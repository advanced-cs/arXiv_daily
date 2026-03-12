# 计算机视觉 cs.CV

- **最新发布 108 篇**

- **更新 93 篇**

## 最新发布

#### [new 001] HyPER-GAN: Hybrid Patch-Based Image-to-Image Translation for Real-Time Photorealism Enhancement
- **分类: cs.CV**

- **简介: 该论文提出HyPER-GAN，用于实时图像增强任务，解决合成数据 photorealism 提升问题，通过混合训练策略提升视觉真实感和语义一致性。**

- **链接: [https://arxiv.org/pdf/2603.10604](https://arxiv.org/pdf/2603.10604)**

> **作者:** Stefanos Pasios; Nikos Nikolaidis
>
> **备注:** 8 pages
>
> **摘要:** Generative models are widely employed to enhance the photorealism of synthetic data for training computer vision algorithms. However, they often introduce visual artifacts that degrade the accuracy of these algorithms and require high computational resources, limiting their applicability in real-time training or evaluation scenarios. In this paper, we propose Hybrid Patch Enhanced Realism Generative Adversarial Network (HyPER-GAN), a lightweight image-to-image translation method based on a U-Net-style generator designed for real-time inference. The model is trained using paired synthetic and photorealism-enhanced images, complemented by a hybrid training strategy that incorporates matched patches from real-world data to improve visual realism and semantic consistency. Experimental results demonstrate that HyPER-GAN outperforms state-of-the-art paired image-to-image translation methods in terms of inference latency, visual realism, and semantic robustness. Moreover, it is illustrated that the proposed hybrid training strategy indeed improves visual quality and semantic consistency compared to training the model solely with paired synthetic and photorealism-enhanced images. Code and pretrained models are publicly available for download at: this https URL
>
---
#### [new 002] WalkGPT: Grounded Vision-Language Conversation with Depth-Aware Segmentation for Pedestrian Navigation
- **分类: cs.CV; cs.CY**

- **简介: 该论文提出WalkGPT，解决城市导航中语义与空间推理问题，通过融合语言与分割实现深度感知的无障碍导航指导。**

- **链接: [https://arxiv.org/pdf/2603.10703](https://arxiv.org/pdf/2603.10703)**

> **作者:** Rafi Ibn Sultan; Hui Zhu; Xiangyu Zhou; Chengyin Li; Prashant Khanduri; Marco Brocanelli; Dongxiao Zhu
>
> **备注:** Accepted by CVPR-2026
>
> **摘要:** Ensuring accessible pedestrian navigation requires reasoning about both semantic and spatial aspects of complex urban scenes, a challenge that existing Large Vision-Language Models (LVLMs) struggle to meet. Although these models can describe visual content, their lack of explicit grounding leads to object hallucinations and unreliable depth reasoning, limiting their usefulness for accessibility guidance. We introduce WalkGPT, a pixel-grounded LVLM for the new task of Grounded Navigation Guide, unifying language reasoning and segmentation within a single architecture for depth-aware accessibility guidance. Given a pedestrian-view image and a navigation query, WalkGPT generates a conversational response with segmentation masks that delineate accessible and harmful features, along with relative depth estimation. The model incorporates a Multi-Scale Query Projector (MSQP) that shapes the final image tokens by aggregating them along text tokens across spatial hierarchies, and a Calibrated Text Projector (CTP), guided by a proposed Region Alignment Loss, that maps language embeddings into segmentation-aware representations. These components enable fine-grained grounding and depth inference without user-provided cues or anchor points, allowing the model to generate complete and realistic navigation guidance. We also introduce PAVE, a large-scale benchmark of 41k pedestrian-view images paired with accessibility-aware questions and depth-grounded answers. Experiments show that WalkGPT achieves strong grounded reasoning and segmentation performance. The source code and dataset are available on the \href{this https URL}{project website}.
>
---
#### [new 003] Multi-Person Pose Estimation Evaluation Using Optimal Transportation and Improved Pose Matching
- **分类: cs.CV**

- **简介: 该论文属于多人体姿态估计任务，旨在解决现有评估指标忽视低置信度误检的问题。提出OCpose方法，通过最优运输理论公平评估所有检测姿态，提升评估可靠性。**

- **链接: [https://arxiv.org/pdf/2603.10398](https://arxiv.org/pdf/2603.10398)**

> **作者:** Takato Moriki; Hiromu Taketsugu; Norimichi Ukita
>
> **备注:** 8 pages, 10 figures. Accepted at MVA 2025
>
> **摘要:** In Multi-Person Pose Estimation, many metrics place importance on ranking of pose detection confidence scores. Current metrics tend to disregard false-positive poses with low confidence, focusing primarily on a larger number of high-confidence poses. Consequently, these metrics may yield high scores even when many false-positive poses with low confidence are detected. For fair evaluation taking into account a tradeoff between true-positive and false-positive poses, this paper proposes Optimal Correction Cost for pose (OCpose), which evaluates detected poses against pose annotations as an optimal transportation. For the fair tradeoff between true-positive and false-positive poses, OCpose equally evaluates all the detected poses regardless of their confidence scores. In OCpose, on the other hand, the confidence score of each pose is utilized to improve the reliability of matching scores between the estimated pose and pose annotations. As a result, OCpose provides a different perspective assessment than other confidence ranking-based metrics.
>
---
#### [new 004] Contrastive learning-based video quality assessment-jointed video vision transformer for video recognition
- **分类: cs.CV**

- **简介: 该论文属于视频分类任务，旨在解决视频质量影响分类效果的问题。通过结合自监督学习与无参考VQA，提出SSL-V3模型，提升分类准确性。**

- **链接: [https://arxiv.org/pdf/2603.10965](https://arxiv.org/pdf/2603.10965)**

> **作者:** Jian Sun; Mohammad H. Mahoor
>
> **备注:** 9 figures, 10 tables,
>
> **摘要:** Video quality significantly affects video classification. We found this problem when we classified Mild Cognitive Impairment well from clear videos, but worse from blurred ones. From then, we realized that referring to Video Quality Assessment (VQA) may improve video classification. This paper proposed Self-Supervised Learning-based Video Vision Transformer combined with No-reference VQA for video classification (SSL-V3) to fulfill the goal. SSL-V3 leverages Combined-SSL mechanism to join VQA into video classification and address the label shortage of VQA, which commonly occurs in video datasets, making it impossible to provide an accurate Video Quality Score. In brief, Combined-SSL takes video quality score as a factor to directly tune the feature map of the video classification. Then, the score, as an intersected point, links VQA and classification, using the supervised classification task to tune the parameters of VQA. SSL-V3 achieved robust experimental results on two datasets. For example, it reached an accuracy of 94.87% on some interview videos in the I-CONECT (a facial video-involved healthcare dataset), verifying SSL-V3's effectiveness.
>
---
#### [new 005] From Imitation to Intuition: Intrinsic Reasoning for Open-Instance Video Classification
- **分类: cs.CV**

- **简介: 该论文属于视频分类任务，解决开放实例下的复杂类别差异问题。通过引入内在推理框架DeepIntuit，提升模型从模仿到直觉的推理能力，增强分类准确性。**

- **链接: [https://arxiv.org/pdf/2603.10300](https://arxiv.org/pdf/2603.10300)**

> **作者:** Ke Zhang; Xiangchen Zhao; Yunjie Tian; Jiayu Zheng; Vishal M. Patel; Di Fu
>
> **备注:** 18 pages, 7 figures
>
> **摘要:** Conventional video classification models, acting as effective imitators, excel in scenarios with homogeneous data distributions. However, real-world applications often present an open-instance challenge, where intra-class variations are vast and complex, beyond existing benchmarks. While traditional video encoder models struggle to fit these diverse distributions, vision-language models (VLMs) offer superior generalization but have not fully leveraged their reasoning capabilities (intuition) for such tasks. In this paper, we bridge this gap with an intrinsic reasoning framework that evolves open-instance video classification from imitation to intuition. Our approach, namely DeepIntuit, begins with a cold-start supervised alignment to initialize reasoning capability, followed by refinement using Group Relative Policy Optimization (GRPO) to enhance reasoning coherence through reinforcement learning. Crucially, to translate this reasoning into accurate classification, DeepIntuit then introduces an intuitive calibration stage. In this stage, a classifier is trained on this intrinsic reasoning traces generated by the refined VLM, ensuring stable knowledge transfer without distribution mismatch. Extensive experiments demonstrate that for open-instance video classification, DeepIntuit benefits significantly from transcending simple feature imitation and evolving toward intrinsic reasoning. Our project is available at this https URL.
>
---
#### [new 006] S2D: Sparse to Dense Lifting for 3D Reconstruction with Minimal Inputs
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决稀疏输入下重建质量差的问题。提出S2D方法，通过扩散模型和重构策略，实现从稀疏点云到高质量3DGS的提升。**

- **链接: [https://arxiv.org/pdf/2603.10893](https://arxiv.org/pdf/2603.10893)**

> **作者:** Yuzhou Ji; Qijian Tian; He Zhu; Xiaoqi Jiang; Guangzhi Cao; Lizhuang Ma; Yuan Xie; Xin Tan
>
> **摘要:** Explicit 3D representations have already become an essential medium for 3D simulation and understanding. However, the most commonly used point cloud and 3D Gaussian Splatting (3DGS) each suffer from non-photorealistic rendering and significant degradation under sparse inputs. In this paper, we introduce Sparse to Dense lifting (S2D), a novel pipeline that bridges the two representations and achieves high-quality 3DGS reconstruction with minimal inputs. Specifically, the S2D lifting is two-fold. We first present an efficient one-step diffusion model that lifts sparse point cloud for high-fidelity image artifact fixing. Meanwhile, to reconstruct 3D consistent scenes, we also design a corresponding reconstruction strategy with random sample drop and weighted gradient for robust model fitting from sparse input views to dense novel views. Extensive experiments show that S2D achieves the best consistency in generating novel view guidance and first-tier sparse view reconstruction quality under different input sparsity. By reconstructing stable scenes with the least possible captures among existing methods, S2D enables minimal input requirements for 3DGS applications.
>
---
#### [new 007] Does AI See like Art Historians? Interpreting How Vision Language Models Recognize Artistic Style
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型（VLM）研究任务，旨在探讨VLM如何识别艺术风格，并评估其与艺术史家标准的契合度。通过分析模型机制和概念相关性，验证其推理合理性。**

- **链接: [https://arxiv.org/pdf/2603.11024](https://arxiv.org/pdf/2603.11024)**

> **作者:** Marvin Limpijankit; Milad Alshomary; Yassin Oulad Daoud; Amith Ananthram; Tim Trombley; Elias Stengel-Eskin; Mohit Bansal; Noam M. Elcott; Kathleen McKeown
>
> **备注:** 12 pages, 12 figures
>
> **摘要:** VLMs have become increasingly proficient at a range of computer vision tasks, such as visual question answering and object detection. This includes increasingly strong capabilities in the domain of art, from analyzing artwork to generation of art. In an interdisciplinary collaboration between computer scientists and art historians, we characterize the mechanisms underlying VLMs' ability to predict artistic style and assess the extent to which they align with the criteria art historians use to reason about artistic style. We employ a latent-space decomposition approach to identify concepts that drive art style prediction and conduct quantitative evaluations, causal analysis and assessment by art historians. Our findings indicate that 73% of the extracted concepts are judged by art historians to exhibit a coherent and semantically meaningful visual feature and 90% of concepts used to predict style of a given artwork were judged relevant. In cases where an irrelevant concept was used to successfully predict style, art historians identified possible reasons for its success; for example, the model might "understand" a concept in more formal terms, such as dark/light contrasts.
>
---
#### [new 008] HG-Lane: High-Fidelity Generation of Lane Scenes under Adverse Weather and Lighting Conditions without Re-annotation
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的车道检测任务，旨在解决极端天气下数据不足导致模型性能下降的问题。提出HG-Lane框架，在无需重新标注的情况下生成高保真车道场景，提升模型在恶劣条件下的可靠性。**

- **链接: [https://arxiv.org/pdf/2603.10128](https://arxiv.org/pdf/2603.10128)**

> **作者:** Daichao Zhao; Qiupu Chen; Feng He; Xin Ning; Qiankun Li
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Lane detection is a crucial task in autonomous driving, as it helps ensure the safe operation of vehicles. However, existing datasets such as CULane and TuSimple contain relatively limited data under extreme weather conditions, including rain, snow, and fog. As a result, detection models trained on these datasets often become unreliable in such environments, which may lead to serious safety-critical failures on the road. To address this issue, we propose HG-Lane, a High-fidelity Generation framework for Lane Scenes under adverse weather and lighting conditions without requiring re-annotation. Based on this framework, we further construct a benchmark that includes adverse weather and lighting scenarios, containing 30,000 images. Experimental results demonstrate that our method consistently and significantly improves the performance of existing lane detection networks. For example, using the state-of-the-art CLRNet, the overall mF1 score on our benchmark increases by 20.87 percent. The F1@50 score for the overall, normal, snow, rain, fog, night, and dusk categories increases by 19.75 percent, 8.63 percent, 38.8 percent, 14.96 percent, 26.84 percent, 21.5 percent, and 12.04 percent, respectively. The code and dataset are available at: this https URL.
>
---
#### [new 009] A dataset of medication images with instance segmentation masks for preventing adverse drug events
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决药物识别中的错误问题。通过构建包含32种药丸的实例分割数据集MEDISEG，提升AI模型在复杂场景下的识别能力。**

- **链接: [https://arxiv.org/pdf/2603.10825](https://arxiv.org/pdf/2603.10825)**

> **作者:** W. I. Chu; S. Hirani; G. Tarroni; L. Li
>
> **备注:** 25 pages, 19 figures. Submitted to Scientific Data (Nature Portfolio)
>
> **摘要:** Medication errors and adverse drug events (ADEs) pose significant risks to patient safety, often arising from difficulties in reliably identifying pharmaceuticals in real-world settings. AI-based pill recognition models offer a promising solution, but the lack of comprehensive datasets hinders their development. Existing pill image datasets rarely capture real-world complexities such as overlapping pills, varied lighting, and occlusions. MEDISEG addresses this gap by providing instance segmentation annotations for 32 distinct pill types across 8262 images, encompassing diverse conditions from individual pill images to cluttered dosette boxes. We trained YOLOv8 and YOLOv9 on MEDISEG to demonstrate their usability, achieving mean average precision at IoU 0.5 of 99.5 percent on the 3-Pills subset and 80.1 percent on the 32-Pills subset. We further evaluate MEDISEG under a few-shot detection protocol, demonstrating that base training on MEDISEG significantly improves recognition of unseen pill classes in occluded multi-pill scenarios compared to existing datasets. These results highlight the dataset's ability not only to support robust supervised training but also to promote transferable representations under limited supervision, making it a valuable resource for developing and benchmarking AI-driven systems for medication safety.
>
---
#### [new 010] Spatial self-supervised Peak Learning and correlation-based Evaluation of peak picking in Mass Spectrometry Imaging
- **分类: cs.CV**

- **简介: 该论文属于质谱成像中的峰挑选任务，旨在解决现有方法在复杂数据中表现不一致的问题。提出一种基于自监督学习的网络，结合空间和光谱信息选择结构化峰，并引入专家标注评估方法。**

- **链接: [https://arxiv.org/pdf/2603.10487](https://arxiv.org/pdf/2603.10487)**

> **作者:** Philipp Weigand; Nikolas Ebert; Shad A. Mohammed; Denis Abu Sammour; Carsten Hopf; Oliver Wasenmüller
>
> **摘要:** Mass spectrometry imaging (MSI) enables label-free visualization of molecular distributions across tissue samples but generates large and complex datasets that require effective peak picking to reduce data size while preserving meaningful biological information. Existing peak picking approaches perform inconsistently across heterogeneous datasets, and their evaluation is often limited to synthetic data or manually selected ion images that do not fully represent real-world challenges in MSI. To address these limitations, we propose an autoencoder-based spatial self-supervised peak learning neural network that selects spatially structured peaks by learning an attention mask leveraging both spatial and spectral information. We further introduce an evaluation procedure based on expert-annotated segmentation masks, allowing a more representative and spatially grounded assessment of peak picking performance. We evaluate our approach on four diverse public MSI datasets using our proposed evaluation procedure. Our approach consistently outperforms state-of-the-art peak picking methods by selecting spatially structured peaks, thus demonstrating its efficacy. These results highlight the value of our spatial self-supervised network in comparison to contemporary state-of-the-art methods. The evaluation procedure can be readily applied to new MSI datasets, thereby providing a consistent and robust framework for the comparison of spatially structured peak picking methods across different datasets.
>
---
#### [new 011] Frames2Residual: Spatiotemporal Decoupling for Self-Supervised Video Denoising
- **分类: cs.CV**

- **简介: 该论文属于视频去噪任务，解决自监督方法中时空一致性与细节恢复的矛盾。提出F2R框架，分阶段处理时空信息，提升去噪效果。**

- **链接: [https://arxiv.org/pdf/2603.10417](https://arxiv.org/pdf/2603.10417)**

> **作者:** Mingjie Ji; Zhan Shi; Kailai Zhou; Zixuan Fu; Xun Cao
>
> **摘要:** Self-supervised video denoising methods typically extend image-based frameworks into the temporal dimension, yet they often struggle to integrate inter-frame temporal consistency with intra-frame spatial specificity. Existing Video Blind-Spot Networks (BSNs) require noise independence by masking the center pixel, this constraint prevents the use of spatial evidence for texture recovery, thereby severing spatiotemporal correlations and causing texture loss. To address this, we propose Frames2Residual (F2R), a spatiotemporal decoupling framework that explicitly divides self-supervised training into two distinct stages: blind temporal consistency modeling and non-blind spatial texture recovery. In Stage 1, a blind temporal estimator learns inter-frame consistency using a frame-wise blind strategy, producing a temporally consistent anchor. In Stage 2, a non-blind spatial refiner leverages this anchor to safely reintroduce the center frame and recover intra-frame high-frequency spatial residuals while preserving temporal stability. Extensive experiments demonstrate that our decoupling strategy allows F2R to outperform existing self-supervised methods on both sRGB and raw video benchmarks.
>
---
#### [new 012] eLasmobranc Dataset: An Image Dataset for Elasmobranch Species Recognition and Biodiversity Monitoring
- **分类: cs.CV**

- **简介: 该论文提出eLasmobranc数据集，用于解决细粒度鲨鱼和鳐鱼识别问题，支持生物多样性监测与保护。**

- **链接: [https://arxiv.org/pdf/2603.10724](https://arxiv.org/pdf/2603.10724)**

> **作者:** Ismael Beviá-Ballesteros; Mario Jerez-Tallón; Nieves Aranda-Garrido; Isabel Abel-Abellán; Irene Antón-Linares; Jorge Azorín-López; Marcelo Saval-Calvo; Andres Fuster-Guilló; Francisca Giménez-Casalduero
>
> **备注:** 9 pages, 6 figures, 5 tables. A future extended version of this work will be submitted to Scientific Data
>
> **摘要:** Elasmobranch populations are experiencing significant global declines, and several species are currently classified as threatened. Reliable monitoring and species-level identification are essential to support conservation and spatial planning initiatives such as Important Shark and Ray Areas (ISRAs). However, existing visual datasets are predominantly detection-oriented, underwater-acquired, or limited to coarse-grained categories, restricting their applicability to fine-grained morphological classification. We present the eLasmobranc Dataset, a curated and publicly available image collection from seven ecologically relevant elasmobranch species inhabiting the eastern Spanish Mediterranean coast, a region where two ISRAs have been identified. Images were obtained through dedicated data collection, including field campaigns and collaborations with local fish markets and projects, as well as from open-access public sources. The dataset was constructed predominantly from images acquired outside the aquatic environment under standardized protocols to ensure clear visualization of diagnostic morphological traits. It integrates expert-validated species annotations, structured spatial and temporal metadata, and complementary species-level information. The eLasmobranc Dataset is specifically designed to support supervised species-level classification, population studies, and the development of artificial intelligence systems for biodiversity monitoring. By combining morphological clarity, taxonomic reliability, and public accessibility, the dataset addresses a critical gap in fine-grained elasmobranch identification and promotes reproducible research in conservation-oriented computer vision. The dataset is publicly available at this https URL.
>
---
#### [new 013] V2M-Zero: Zero-Pair Time-Aligned Video-to-Music Generation
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.SD**

- **简介: 该论文属于视频到音乐生成任务，解决视频与音乐时间对齐问题。通过分析视频和音乐的时序结构，无需配对数据即可生成同步音乐。**

- **链接: [https://arxiv.org/pdf/2603.11042](https://arxiv.org/pdf/2603.11042)**

> **作者:** Yan-Bo Lin; Jonah Casebeer; Long Mai; Aniruddha Mahapatra; Gedas Bertasius; Nicholas J. Bryan
>
> **备注:** Project page: this https URL
>
> **摘要:** Generating music that temporally aligns with video events is challenging for existing text-to-music models, which lack fine-grained temporal control. We introduce V2M-Zero, a zero-pair video-to-music generation approach that outputs time-aligned music for video. Our method is motivated by a key observation: temporal synchronization requires matching when and how much change occurs, not what changes. While musical and visual events differ semantically, they exhibit shared temporal structure that can be captured independently within each modality. We capture this structure through event curves computed from intra-modal similarity using pretrained music and video encoders. By measuring temporal change within each modality independently, these curves provide comparable representations across modalities. This enables a simple training strategy: fine-tune a text-to-music model on music-event curves, then substitute video-event curves at inference without cross-modal training or paired data. Across OES-Pub, MovieGenBench-Music, and AIST++, V2M-Zero achieves substantial gains over paired-data baselines: 5-21% higher audio quality, 13-15% better semantic alignment, 21-52% improved temporal synchronization, and 28% higher beat alignment on dance videos. We find similar results via a large crowd-source subjective listening test. Overall, our results validate that temporal alignment through within-modality features, rather than paired cross-modal supervision, is effective for video-to-music generation. Results are available at this https URL
>
---
#### [new 014] One Adapter for All: Towards Unified Representation in Step-Imbalanced Class-Incremental Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于类增量学习任务，解决步不平衡问题。提出One-A框架，通过统一适配器融合更新，提升模型稳定性与效率。**

- **链接: [https://arxiv.org/pdf/2603.10237](https://arxiv.org/pdf/2603.10237)**

> **作者:** Xiaoyan Zhang; Jiangpeng He
>
> **备注:** Code is available at this https URL
>
> **摘要:** Class-incremental learning (CIL) aims to acquire new classes over time while retaining prior knowledge, yet most setups and methods assume balanced task streams. In practice, the number of classes per task often varies significantly. We refer to this as step imbalance, where large tasks that contain more classes dominate learning and small tasks inject unstable updates. Existing CIL methods assume balanced tasks and therefore treat all tasks uniformly, producing imbalanced updates that degrade overall learning performance. To address this challenge, we propose One-A, a unified and imbalance-aware framework that incrementally merges task updates into a single adapter, maintaining constant inference cost. One-A performs asymmetric subspace alignment to preserve dominant subspaces learned from large tasks while constraining low-information updates within them. An information-adaptive weighting balances the contribution between base and new adapters, and a directional gating mechanism selectively fuses updates along each singular direction, maintaining stability in head directions and plasticity in tail ones. Across multiple benchmarks and step-imbalanced streams, One-A achieves competitive accuracy with significantly low inference overhead, showing that a single, asymmetrically fused adapter can remain both adaptive to dynamic task sizes and efficient at deployment.
>
---
#### [new 015] VCR: Variance-Driven Channel Recalibration for Robust Low-Light Enhancement
- **分类: cs.CV**

- **简介: 该论文属于低光图像增强任务，旨在解决颜色与亮度纠缠及色彩分布不一致问题。提出VCR框架，通过通道自适应调整和色彩分布对齐提升增强效果。**

- **链接: [https://arxiv.org/pdf/2603.10975](https://arxiv.org/pdf/2603.10975)**

> **作者:** Zhixin Cheng; Fangwen Zhang; Xiaotian Yin; Baoqun Yin; Haodian Wang
>
> **摘要:** Most sRGB-based LLIE methods suffer from entangled luminance and color, while the HSV color space offers insufficient decoupling at the cost of introducing significant red and black noise artifacts. Recently, the HVI color space has been proposed to address these limitations by enhancing color fidelity through chrominance polarization and intensity compression. However, existing methods could suffer from channel-level inconsistency between luminance and chrominance, and misaligned color distribution may lead to unnatural enhancement results. To address these challenges, we propose the Variance-Driven Channel Recalibration for Robust Low-Light Enhancement (VCR), a novel framework for low-light image enhancement. VCR consists of two main components, including the Channel Adaptive Adjustment (CAA) module, which employs variance-guided feature filtering to enhance the model's focus on regions with high intensity and color distribution. And the Color Distribution Alignment (CDA) module, which enforces distribution alignment in the color feature space. These designs enhance perceptual quality under low-light conditions. Experimental results on several benchmark datasets demonstrate that the proposed method achieves state-of-the-art performance compared with existing methods.
>
---
#### [new 016] How To Embed Matters: Evaluation of EO Embedding Design Choices
- **分类: cs.CV**

- **简介: 该论文属于遥感图像处理任务，研究GeoFM嵌入设计对EO任务的影响。通过系统分析，探讨了不同架构和策略对嵌入效果的影响，以提升模型性能与效率。**

- **链接: [https://arxiv.org/pdf/2603.10658](https://arxiv.org/pdf/2603.10658)**

> **作者:** Luis Gilch; Isabelle Wittmann; Maximilian Nitsche; Johannes Jakubik; Arne Ewald; Thomas Brunschwiler
>
> **摘要:** Earth observation (EO) missions produce petabytes of multispectral imagery, increasingly analyzed using large Geospatial Foundation Models (GeoFMs). Alongside end-to-end adaptation, workflows make growing use of intermediate representations as task-agnostic embeddings, enabling models to compute representations once and reuse them across downstream tasks. Consequently, when GeoFMs act as feature extractors, decisions about how representations are obtained, aggregated, and combined affect downstream performance and pipeline scalability. Understanding these trade-offs is essential for scalable embedding-based EO workflows, where compact embeddings can replace raw data while remaining broadly useful. We present a systematic analysis of embedding design in GeoFM-based EO workflows. Leveraging NeuCo-Bench, we study how backbone architecture, pretraining strategy, representation depth, spatial aggregation, and representation combination influence EO task performance. We demonstrate the usability of GeoFM embeddings by aggregating them into fixed-size representations more than 500x smaller than the raw input data. Across models, we find consistent trends: transformer backbones with mean pooling provide strong default embeddings, intermediate ResNet layers can outperform final layers, self-supervised objectives exhibit task-specific strengths, and combining embeddings from different objectives often improves robustness.
>
---
#### [new 017] An Automated Radiomics Framework for Postoperative Survival Prediction in Colorectal Liver Metastases using Preoperative MRI
- **分类: cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决结直肠肝转移术后生存预测问题。通过自动化分割与放射组学方法，实现准确的生存期预测。**

- **链接: [https://arxiv.org/pdf/2603.10216](https://arxiv.org/pdf/2603.10216)**

> **作者:** Muhammad Alberb; Jianan Chen; Hossam El-rewaidy; Paul Karanicolas; Arun Seth; Yutaka Amemiya; Anne Martel; Helen Cheung
>
> **摘要:** While colorectal liver metastasis (CRLM) is potentially curable via hepatectomy, patient outcomes remain highly heterogeneous. Postoperative survival prediction is necessary to avoid non-beneficial surgeries and guide personalized therapy. In this study, we present an automated AI-based framework for postoperative CRLM survival prediction using pre- and post-contrast MRI. We performed a retrospective study of 227 CRLM patients who had gadoxetate-enhanced MRI prior to curative-intent hepatectomy between 2013 and 2020. We developed a survival prediction framework comprising an anatomy-aware segmentation pipeline followed by a radiomics pipeline. The segmentation pipeline learns liver, CRLMs, and spleen segmentation from partially-annotated data, leveraging promptable foundation models to generate pseudo-labels. To support this pipeline, we propose SAMONAI, a prompt propagation algorithm that extends Segment Anything Model to 3D point-based segmentation. Predicted pre- and post-contrast segmentations are then fed into our radiomics pipeline, which extracts per-tumor features and predicts survival using SurvAMINN, an autoencoder-based multiple instance neural network for time-to-event survival prediction. SurvAMINN jointly learns dimensionality reduction and survival prediction from right-censored data, emphasizing high-risk metastases. We compared our framework against established methods and biomarkers using univariate and multivariate Cox regression. Our segmentation pipeline achieves median Dice scores of 0.96 (liver) and 0.93 (spleen), driving a CRLM segmentation Dice score of 0.78 and a detection F1-score of 0.79. Accurate segmentation enables our radiomics pipeline to achieve a survival prediction C-index of 0.69. Our results show the potential of integrating segmentation algorithms with radiomics-based survival analysis to deliver accurate and automated CRLM outcome prediction.
>
---
#### [new 018] Why Does It Look There? Structured Explanations for Image Classification
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像分类的可解释性任务，旨在解决模型决策过程不透明的问题。提出I2X框架，通过结构化解释提升模型可信度与优化效果。**

- **链接: [https://arxiv.org/pdf/2603.10234](https://arxiv.org/pdf/2603.10234)**

> **作者:** Jiarui Li; Zixiang Yin; Samuel J Landry; Zhengming Ding; Ramgopal R. Mettu
>
> **摘要:** Deep learning models achieve remarkable predictive performance, yet their black-box nature limits transparency and trustworthiness. Although numerous explainable artificial intelligence (XAI) methods have been proposed, they primarily provide saliency maps or concepts (i.e., unstructured interpretability). Existing approaches often rely on auxiliary models (\eg, GPT, CLIP) to describe model behavior, thereby compromising faithfulness to the original models. We propose Interpretability to Explainability (I2X), a framework that builds structured explanations directly from unstructured interpretability by quantifying progress at selected checkpoints during training using prototypes extracted from post-hoc XAI methods (e.g., GradCAM). I2X answers the question of "why does it look there" by providing a structured view of both intra- and inter-class decision making during training. Experiments on MNIST and CIFAR10 demonstrate effectiveness of I2X to reveal prototype-based inference process of various image classification models. Moreover, we demonstrate that I2X can be used to improve predictions across different model architectures and datasets: we can identify uncertain prototypes recognized by I2X and then use targeted perturbation of samples that allows fine-tuning to ultimately improve accuracy. Thus, I2X not only faithfully explains model behavior but also provides a practical approach to guide optimization toward desired targets.
>
---
#### [new 019] Bilevel Layer-Positioning LoRA for Real Image Dehazing
- **分类: cs.CV**

- **简介: 该论文属于图像去雾任务，旨在解决真实场景下无参考图像的去雾适应问题。通过引入语义对齐损失和BiLaLoRA策略，实现高效无监督模型适配。**

- **链接: [https://arxiv.org/pdf/2603.10872](https://arxiv.org/pdf/2603.10872)**

> **作者:** Yan Zhang; Long Ma; Yuxin Feng; Zhe Huang; Fan Zhou; Zhuo Su
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Learning-based real image dehazing methods have achieved notable progress, yet they still face adaptation challenges in diverse real haze scenes. These challenges mainly stem from the lack of effective unsupervised mechanisms for unlabeled data and the heavy cost of full model fine-tuning. To address these challenges, we propose the haze-to-clear text-directed loss that leverages CLIP's cross-modal capabilities to reformulate real image dehazing as a semantic alignment problem in latent space, thereby providing explicit unsupervised cross-modal guidance in the absence of reference images. Furthermore, we introduce the Bilevel Layer-positioning LoRA (BiLaLoRA) strategy, which learns both the LoRA parameters and automatically search the injection layers, enabling targeted adaptation of critical network layers. Extensive experiments demonstrate our superiority against state-of-the-art methods on multiple real-world dehazing benchmarks. The code is publicly available at this https URL.
>
---
#### [new 020] PET-F2I: A Comprehensive Benchmark and Parameter-Efficient Fine-Tuning of LLMs for PET/CT Report Impression Generation
- **分类: cs.CV**

- **简介: 该论文针对PET/CT报告生成任务，解决医学文本生成中诊断完整性与准确性不足的问题。构建了大规模基准数据集PET-F2I-41K，并提出参数高效微调模型PET-F2I-7B，提升生成质量与临床适用性。**

- **链接: [https://arxiv.org/pdf/2603.10560](https://arxiv.org/pdf/2603.10560)**

> **作者:** Yuchen Liu; Wenbo Zhang; Liling Peng; Yichi Zhang; Yu Fu; Xin Guo; Chao Qu; Yuan Qi; Le Xue
>
> **摘要:** PET/CT imaging is pivotal in oncology and nuclear medicine, yet summarizing complex findings into precise diagnostic impressions is labor-intensive. While LLMs have shown promise in medical text generation, their capability in the highly specialized domain of PET/CT remains underexplored. We introduce PET-F2I-41K (PET Findings-to-Impression Benchmark), a large-scale benchmark for PET/CT impression generation using LLMs, constructed from over 41k real-world reports. Using PET-F2I-41K, we conduct a comprehensive evaluation of 27 models across proprietary frontier LLMs, open-source generalist models, and medical-domain LLMs, and we develop a domain-adapted 7B model (PET-F2I-7B) fine-tuned from Qwen2.5-7B-Instruct via LoRA. Beyond standard NLG metrics (e.g., BLEU-4, ROUGE-L, BERTScore), we propose three clinically grounded metrics - Entity Coverage Rate (ECR), Uncovered Entity Rate (UER), and Factual Consistency Rate (FCR) - to assess diagnostic completeness and factual reliability. Experiments reveal that neither frontier nor medical-domain LLMs perform adequately in zero-shot settings. In contrast, PET-F2I-7B achieves substantial gains (e.g., 0.708 BLEU-4) and a 3.0x improvement in entity coverage over the strongest baseline, while offering advantages in cost, latency, and privacy. Beyond this modeling contribution, PET-F2I-41K establishes a standardized evaluation framework to accelerate the development of reliable and clinically deployable reporting systems for PET/CT.
>
---
#### [new 021] OilSAM2: Memory-Augmented SAM2 for Scalable SAR Oil Spill Detection
- **分类: cs.CV**

- **简介: 该论文属于遥感图像分割任务，旨在解决SAR图像中油污检测的挑战。针对现有方法无法跨场景复用信息的问题，提出OilSAM2框架，引入多尺度记忆库提升分割性能。**

- **链接: [https://arxiv.org/pdf/2603.10231](https://arxiv.org/pdf/2603.10231)**

> **作者:** Shuaiyu Chen; Ming Yin; Peng Ren; Chunbo Luo; Zeyu Fu
>
> **摘要:** Segmenting oil spills from Synthetic Aperture Radar (SAR) imagery remains challenging due to severe appearance variability, scale heterogeneity, and the absence of temporal continuity in real world monitoring scenarios. While foundation models such as Segment Anything (SAM) enable prompt driven segmentation, existing SAM based approaches operate on single images and cannot effectively reuse information across scenes. Memory augmented variants (e.g., SAM2) further assume temporal coherence, making them prone to semantic drift when applied to unordered SAR image collections. We propose OilSAM2, a memory augmented segmentation framework tailored for unordered SAR oil spill monitoring. OilSAM2 introduces a hierarchical feature aware multi scale memory bank that explicitly models texture, structure, and semantic level representations, enabling robust cross image information reuse. To mitigate memory drift, we further propose a structure semantic consistent memory update strategy that selectively refreshes memory based on semantic discrepancy and structural this http URL on two public SAR oil spill datasets demonstrate that OilSAM2 achieves state of the art segmentation performance, delivering stable and accurate results under noisy SAR monitoring scenarios. The source code is available at this https URL.
>
---
#### [new 022] Layer Consistency Matters: Elegant Latent Transition Discrepancy for Generalizable Synthetic Image Detection
- **分类: cs.CV**

- **简介: 该论文属于合成图像检测任务，旨在解决现有方法泛化能力差的问题。通过分析真实与合成图像在潜在层间的一致性差异，提出LTD方法，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.10598](https://arxiv.org/pdf/2603.10598)**

> **作者:** Yawen Yang; Feng Li; Shuqi Kong; Yunfeng Diao; Xinjian Gao; Zenglin Shi; Meng Wang
>
> **摘要:** Recent rapid advancement of generative models has significantly improved the fidelity and accessibility of AI-generated synthetic images. While enabling various innovative applications, the unprecedented realism of these synthetics makes them increasingly indistinguishable from authentic photographs, posing serious security risks, such as media credibility and content manipulation. Although extensive efforts have been dedicated to detecting synthetic images, most existing approaches suffer from poor generalization to unseen data due to their reliance on model-specific artifacts or low-level statistical cues. In this work, we identify a previously unexplored distinction that real images maintain consistent semantic attention and structural coherence in their latent representations, exhibiting more stable feature transitions across network layers, whereas synthetic ones present discernible distinct patterns. Therefore, we propose a novel approach termed latent transition discrepancy (LTD), which captures the inter-layer consistency differences of real and synthetic images. LTD adaptively identifies the most discriminative layers and assesses the transition discrepancies across layers. Benefiting from the proposed inter-layer discriminative modeling, our approach exceeds the base model by 14.35\% in mean Acc across three datasets containing diverse GANs and DMs. Extensive experiments demonstrate that LTD outperforms recent state-of-the-art methods, achieving superior detection accuracy, generalizability, and robustness. The code is available at this https URL
>
---
#### [new 023] Overcoming Visual Clutter in Vision Language Action Models via Concept-Gated Visual Distillation
- **分类: cs.CV; cs.AI; cs.RO; eess.SY**

- **简介: 该论文属于视觉-语言-动作模型任务，解决 clutter 环境下的精度-推理差距问题。提出 CGVD 方法，通过概念门控和傅里叶修复提升操作准确性。**

- **链接: [https://arxiv.org/pdf/2603.10340](https://arxiv.org/pdf/2603.10340)**

> **作者:** Sangmim Song; Sarath Kodagoda; Marc Carmichael; Karthick Thiyagarajan
>
> **备注:** 7 pages, 4 figures, 3 tables
>
> **摘要:** Vision-Language-Action (VLA) models demonstrate impressive zero-shot generalization but frequently suffer from a "Precision-Reasoning Gap" in cluttered environments. This failure is driven by background-induced feature dilution, where high-frequency semantic noise corrupts the geometric grounding required for precise manipulation. To bridge this gap, we propose Concept-Gated Visual Distillation (CGVD), a training-free, model-agnostic inference framework that stabilizes VLA policies. CGVD operates by parsing instructions into safe and distractor sets, utilizing a two-layer target refinement process--combining cross-validation and spatial disambiguation--to explicitly penalize false positives and isolate genuine manipulation targets. We then process the scene via Fourier-based inpainting, generating a clean observation that actively suppresses semantic distractors while preserving critical spatial geometry and visual proprioception. Extensive evaluations in highly cluttered manipulation tasks demonstrate that CGVD prevents performance collapse. In environments with dense semantic distractors, our method significantly outperforms state-of-the-art baselines, achieving a 77.5% success rate compared to the baseline's 43.0%. By enforcing strict attribute adherence, CGVD establishes inference-time visual distillation as a critical prerequisite for robust robotic manipulation in the clutter.
>
---
#### [new 024] SignSparK: Efficient Multilingual Sign Language Production via Sparse Keyframe Learning
- **分类: cs.CV**

- **简介: 该论文属于手语生成任务，解决手语动画自然性和准确性问题。提出SignSparK框架，通过稀疏关键帧学习实现高效多语言手语生成。**

- **链接: [https://arxiv.org/pdf/2603.10446](https://arxiv.org/pdf/2603.10446)**

> **作者:** Jianhe Low; Alexandre Symeonidis-Herzig; Maksym Ivashechkin; Ozge Mercanoglu Sincan; Richard Bowden
>
> **摘要:** Generating natural and linguistically accurate sign language avatars remains a formidable challenge. Current Sign Language Production (SLP) frameworks face a stark trade-off: direct text-to-pose models suffer from regression-to-the-mean effects, while dictionary-retrieval methods produce robotic, disjointed transitions. To resolve this, we propose a novel training paradigm that leverages sparse keyframes to capture the true underlying kinematic distribution of human signing. By predicting dense motion from these discrete anchors, our approach mitigates regression-to-the-mean while ensuring fluid articulation. To realize this paradigm at scale, we first introduce FAST, an ultra-efficient sign segmentation model that automatically mines precise temporal boundaries. We then present SignSparK, a large-scale Conditional Flow Matching (CFM) framework that utilizes these extracted anchors to synthesize 3D signing sequences in SMPL-X and MANO spaces. This keyframe-driven formulation also uniquely unlocks Keyframe-to-Pose (KF2P) generation, making precise spatiotemporal editing of signing sequences possible. Furthermore, our adopted reconstruction-based CFM objective also enables high-fidelity synthesis in fewer than ten sampling steps; this allows SignSparK to scale across four distinct sign languages, establishing the largest multilingual SLP framework to date. Finally, by integrating 3D Gaussian Splatting for photorealistic rendering, we demonstrate through extensive evaluation that SignSparK establishes a new state-of-the-art across diverse SLP tasks and multilingual benchmarks.
>
---
#### [new 025] Beyond Sequential Distance: Inter-Modal Distance Invariant Position Encoding
- **分类: cs.CV**

- **简介: 该论文属于多模态任务，解决长上下文下视觉信息衰减问题。提出DIPE机制，使模型在长文本中保持视觉一致性。**

- **链接: [https://arxiv.org/pdf/2603.10863](https://arxiv.org/pdf/2603.10863)**

> **作者:** Lin Chen; Bolin Ni; Qi Yang; Zili Wang; Kun Ding; Ying Wang; Houwen Peng; Shiming Xiang
>
> **摘要:** Despite the remarkable capabilities of Multimodal Large Language Models (MLLMs), they still suffer from visual fading in long-context scenarios. Specifically, the attention to visual tokens diminishes as the text sequence lengthens, leading to text generation detached from visual constraints. We attribute this degradation to the inherent inductive bias of Multimodal RoPE, which penalizes inter-modal attention as the distance between visual and text tokens increases. To address this, we propose inter-modal Distance Invariant Position Encoding (DIPE), a simple but effective mechanism that disentangles position encoding based on modality interactions. DIPE retains the natural relative positioning for intra-modal interactions to preserve local structure, while enforcing an anchored perceptual proximity for inter-modal interactions. This strategy effectively mitigates the inter-modal distance-based penalty, ensuring that visual signals remain perceptually consistent regardless of the context length. Experimental results demonstrate that by integrating DIPE with Multimodal RoPE, the model maintains stable visual grounding in long-context scenarios, significantly alleviating visual fading while preserving performance on standard short-context benchmarks. Code is available at this https URL.
>
---
#### [new 026] The Quadratic Geometry of Flow Matching: Semantic Granularity Alignment for Text-to-Image Synthesis
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决生成模型训练中的优化效率与质量平衡问题。通过引入语义粒度对齐方法，提升模型收敛速度和结构完整性。**

- **链接: [https://arxiv.org/pdf/2603.10785](https://arxiv.org/pdf/2603.10785)**

> **作者:** Zhinan Xiong; Shunqi Yuan
>
> **备注:** 43 pages
>
> **摘要:** In this work, we analyze the optimization dynamics of generative fine-tuning. We observe that under the Flow Matching framework, the standard MSE objective can be formulated as a Quadratic Form governed by a dynamically evolving Neural Tangent Kernel (NTK). This geometric perspective reveals a latent Data Interaction Matrix, where diagonal terms represent independent sample learning and off-diagonal terms encode residual correlation between heterogeneous features. Although standard training implicitly optimizes these cross-term interferences, it does so without explicit control; moreover, the prevailing data-homogeneity assumption may constrain the model's effective capacity. Motivated by this insight, we propose Semantic Granularity Alignment (SGA), using Text-to-Image synthesis as a testbed. SGA engineers targeted interventions in the vector residual field to mitigate gradient conflicts. Evaluations across DiT and U-Net architectures confirm that SGA advances the efficiency-quality trade-off by accelerating convergence and improving structural integrity.
>
---
#### [new 027] Splat2Real: Novel-view Scaling for Physical AI with 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于单目RGB到3D感知任务，解决训练与部署视角不一致导致的泛化问题。提出Splat2Real方法，通过新颖视角选择提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.10638](https://arxiv.org/pdf/2603.10638)**

> **作者:** Hansol Lim; Jongseong Brad Choi
>
> **摘要:** Physical AI faces viewpoint shift between training and deployment, and novel-view robustness is essential for monocular RGB-to-3D perception. We cast Real2Render2Real monocular depth pretraining as imitation-learning-style supervision from a digital twin oracle: a student depth network imitates expert metric depth/visibility rendered from a scene mesh, while 3DGS supplies scalable novel-view observations. We present Splat2Real, centered on novel-view scaling: performance depends more on which views are added than on raw view count. We introduce CN-Coverage, a coverage+novelty curriculum that greedily selects views by geometry gain and an extrapolation penalty, plus a quality-aware guardrail fallback for low-reliability teachers. Across 20 TUM RGB-D sequences with step-matched budgets (N=0 to 2000 additional rendered views, with N unique <= 500 and resampling for larger budgets), naive scaling is unstable; CN-Coverage mitigates worst-case regressions relative to Robot/Coverage policies, and GOL-Gated CN-Coverage provides the strongest medium-high-budget stability with the lowest high-novelty tail error. Downstream control-proxy results versus N provides embodied-relevance evidence by shifting safety/progress trade-offs under viewpoint shift.
>
---
#### [new 028] World2Act: Latent Action Post-Training via Skill-Compositional World Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言-动作（VLA）策略的后训练任务，旨在解决WM生成视频质量不稳定导致的策略泛化问题。通过引入World2Act框架，将VLA动作与WM视频动态潜在空间对齐，提升策略鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.10422](https://arxiv.org/pdf/2603.10422)**

> **作者:** An Dinh Vuong; Tuan Van Vo; Abdullah Sohail; Haoran Ding; Liang Ma; Xiaodan Liang; Anqing Duan; Ivan Laptev; Ian Reid
>
> **备注:** Project page: this https URL
>
> **摘要:** World Models (WMs) have emerged as a promising approach for post-training Vision-Language-Action (VLA) policies to improve robustness and generalization under environmental changes. However, most WM-based post-training methods rely on pixel-space supervision, making policies sensitive to pixel-level artifacts and hallucination from imperfect WM rollouts. We introduce World2Act, a post-training framework that aligns VLA actions directly with WM video-dynamics latents using a contrastive matching objective, reducing dependence on pixels. Post-training performance is tied to rollout quality, yet current WMs struggle with arbitrary-length video generation as they are mostly trained on fixed-length clips while robotic execution durations vary widely. To address this, we propose an automatic LLM-based skill-decomposition pipeline that segments high-level instructions into low-level prompts. Our pipeline produces RoboCasa-Skill and LIBERO-Skill, supporting skill-compositional WMs that remain temporally consistent across diverse task horizons. Empirically, applying World2Act to VLAs like GR00T-N1.6 and Cosmos Policy achieves state-of-the-art results on RoboCasa and LIBERO, and improves real-world performance by 6.7%, enhancing embodied agent generalization.
>
---
#### [new 029] PolGS++: Physically-Guided Polarimetric Gaussian Splatting for Fast Reflective Surface Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于三维重建任务，旨在解决反射表面重建难题。通过引入物理引导的偏振BRDF模型和深度引导的可见性掩码，提升3DGS在反射表面几何与法线恢复上的效果。**

- **链接: [https://arxiv.org/pdf/2603.10801](https://arxiv.org/pdf/2603.10801)**

> **作者:** Yufei Han; Chu Zhou; Youwei Lyu; Qi Chen; Si Li; Boxin Shi; Yunpeng Jia; Heng Guo; Zhanyu Ma
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2509.19726
>
> **摘要:** Accurate reconstruction of reflective surfaces remains a fundamental challenge in computer vision, with broad applications in real-time virtual reality and digital content creation. Although 3D Gaussian Splatting (3DGS) enables efficient novel-view rendering with explicit representations, its performance on reflective surfaces still lags behind implicit neural methods, especially in recovering fine geometry and surface normals. To address this gap, we propose PolGS++, a physically-guided polarimetric Gaussian Splatting framework for fast reflective surface reconstruction. Specifically, we integrate a polarized BRDF (pBRDF) model into 3DGS to explicitly decouple diffuse and specular components, providing physically grounded reflectance modeling and stronger geometric cues for reflective surface recovery. Furthermore, we introduce a depth-guided visibility mask acquisition mechanism that enables angle-of-polarization (AoP)-based tangent-space consistency constraints in Gaussian Splatting without costly ray-tracing intersections. This physically guided design improves reconstruction quality and efficiency, requiring only about 10 minutes of training. Extensive experiments on both synthetic and real-world datasets validate the effectiveness of our method.
>
---
#### [new 030] Motion Forcing: A Decoupled Framework for Robust Video Generation in Motion Dynamics
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决视觉质量、物理一致性与可控性之间的平衡问题。提出Motion Forcing框架，通过分解生成过程提升复杂场景下的稳定性。**

- **链接: [https://arxiv.org/pdf/2603.10408](https://arxiv.org/pdf/2603.10408)**

> **作者:** Tianshuo Xu; Zhifei Chen; Leyi Wu; Hao Lu; Ying-cong Chen
>
> **备注:** this https URL
>
> **摘要:** The ultimate goal of video generation is to satisfy a fundamental trilemma: achieving high visual quality, maintaining rigorous physical consistency, and enabling precise controllability. While recent models can maintain this balance in simple, isolated scenarios, we observe that this equilibrium is fragile and often breaks down as scene complexity increases (e.g., involving collisions or dense traffic). To address this, we introduce \textbf{Motion Forcing}, a framework designed to stabilize this trilemma even in complex generative tasks. Our key insight is to explicitly decouple physical reasoning from visual synthesis via a hierarchical \textbf{``Point-Shape-Appearance''} paradigm. This approach decomposes generation into verifiable stages: modeling complex dynamics as sparse geometric anchors (\textbf{Point}), expanding them into dynamic depth maps that explicitly resolve 3D geometry (\textbf{Shape}), and finally rendering high-fidelity textures (\textbf{Appearance}). Furthermore, to foster robust physical understanding, we employ a \textbf{Masked Point Recovery} strategy. By randomly masking input anchors during training and enforcing the reconstruction of complete dynamic depth, the model is compelled to move beyond passive pattern matching and learn latent physical laws (e.g., inertia) to infer missing trajectories. Extensive experiments on autonomous driving benchmarks show that Motion Forcing significantly outperforms state-of-the-art baselines, maintaining trilemma stability across complex scenes. Evaluations on physics and robotics further confirm our framework's generality.
>
---
#### [new 031] Fuel Gauge: Estimating Chain-of-Thought Length Ahead of Time in Large Multimodal Models
- **分类: cs.CV**

- **简介: 该论文提出Fuel Gauge，用于提前估计大模型的思维链长度，解决推理过程效率低和资源浪费问题，适用于多模态任务。**

- **链接: [https://arxiv.org/pdf/2603.10335](https://arxiv.org/pdf/2603.10335)**

> **作者:** Yuedong Yang; Xiwen Wei; Mustafa Munir; Radu Marculescu
>
> **摘要:** Reasoning Large Multi-modality Models (LMMs) have become the de facto choice for many applications. However, these models rely on a Chain-of-Thought (CoT) process that is lengthy and unpredictable at runtime, often resulting in inefficient use of computational resources (due to memory fragmentation) and sub-optimal accuracy (due to under- and over-thinking). We observe empirically that the CoT process follows a very simple form, whose behavior is independent of the specific generated samples. This suggests that the CoT length can be estimated ahead of time based on a hidden parameter representing the amount of "fuel" available to support the reasoning process. Based on this insight, we propose Fuel Gauge, the first method which extracts this hidden signal and predicts CoT length ahead of time. We demonstrate the utility on the Fuel Gauge on two downstream tasks: predictive KV cache allocation, which addresses memory fragmentation in LMM serving systems, and CoT length modulation, which mitigates under-thinking and over-thinking. Extensive experiments on LMMs across text-only, image-text, and video-text question answering benchmarks demonstrate the effectiveness, generalizability, and practical value of our Fuel Gauge. For example, on the GPQA-Diamond benchmark, our Fuel Gauge achieves less than half the CoT length prediction error compared to the baseline; this translates into a 13.37x reduction in the memory allocation frequency.
>
---
#### [new 032] Taking Shortcuts for Categorical VQA Using Super Neurons
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉语言模型优化任务，旨在提升分类性能并加速推理。通过直接分析模型激活值，提出Super Neurons方法，实现高效准确的分类。**

- **链接: [https://arxiv.org/pdf/2603.10781](https://arxiv.org/pdf/2603.10781)**

> **作者:** Pierre Musacchio; Jaeyi Jeong; Dahun Kim; Jaesik Park
>
> **备注:** 25 pages, 15 tables, 8 figures
>
> **摘要:** Sparse Attention Vectors (SAVs) have emerged as an excellent training-free alternative to supervised finetuning or low-rank adaptation to improve the performance of Vision Language Models (VLMs). At their heart, SAVs select a few accurate attention heads for a task of interest and use them as classifiers, rather than relying on the model's prediction. In a similar spirit, we find that directly probing the raw activations of the VLM, in the form of scalar values, is sufficient to yield accurate classifiers on diverse visually grounded downstream tasks. Shifting focus from attention vectors to scalar activations dramatically increases the search space for accurate parameters, allowing us to find more discriminative neurons immediately from the first generated token. We call such activations Super Neurons (SNs). In this probing setting, we discover that enough SNs appear in the shallower layers of the large language model to allow for extreme early exiting from the first layer of the model at the first generated token. Compared to the original network, SNs robustly improve the classification performance while achieving a speedup of up to 5.10x.
>
---
#### [new 033] Guiding Diffusion Models with Semantically Degraded Conditions
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决CFG因使用无语义的空提示导致的几何纠缠问题。通过引入CDG，用退化的条件替代空提示，提升组合精度和语义对齐。**

- **链接: [https://arxiv.org/pdf/2603.10780](https://arxiv.org/pdf/2603.10780)**

> **作者:** Shilong Han; Yuming Zhang; Hongxia Wang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Classifier-Free Guidance (CFG) is a cornerstone of modern text-to-image models, yet its reliance on a semantically vacuous null prompt ($\varnothing$) generates a guidance signal prone to geometric entanglement. This is a key factor limiting its precision, leading to well-documented failures in complex compositional tasks. We propose Condition-Degradation Guidance (CDG), a novel paradigm that replaces the null prompt with a strategically degraded condition, $\boldsymbol{c}_{\text{deg}}$. This reframes guidance from a coarse "good vs. null" contrast to a more refined "good vs. almost good" discrimination, thereby compelling the model to capture fine-grained semantic distinctions. We find that tokens in transformer text encoders split into two functional roles: content tokens encoding object semantics, and context-aggregating tokens capturing global context. By selectively degrading only the former, CDG constructs $\boldsymbol{c}_{\text{deg}}$ without external models or training. Validated across diverse architectures including Stable Diffusion 3, FLUX, and Qwen-Image, CDG markedly improves compositional accuracy and text-image alignment. As a lightweight, plug-and-play module, it achieves this with negligible computational overhead. Our work challenges the reliance on static, information-sparse negative samples and establishes a new principle for diffusion guidance: the construction of adaptive, semantically-aware negative samples is critical to achieving precise semantic control. Code is available at this https URL.
>
---
#### [new 034] UHD Image Deblurring via Autoregressive Flow with Ill-conditioned Constraints
- **分类: cs.CV**

- **简介: 该论文属于UHD图像去模糊任务，旨在解决高分辨率图像去模糊中细节恢复与计算效率的平衡问题。提出一种带有病态约束的自回归流方法，实现高效且稳定的多尺度去模糊。**

- **链接: [https://arxiv.org/pdf/2603.10517](https://arxiv.org/pdf/2603.10517)**

> **作者:** Yucheng Xin; Dawei Zhao; Xiang Chen; Chen Wu; Pu Wang; Dianjie Lu; Guijuan Zhang; Xiuyi Jia; Zhuoran Zheng
>
> **备注:** Submitted to ECCV 2026
>
> **摘要:** Ultra-high-definition (UHD) image deblurring poses significant challenges for UHD restoration methods, which must balance fine-grained detail recovery and practical inference efficiency. Although prominent discriminative and generative methods have achieved remarkable results, a trade-off persists between computational cost and the ability to generate fine-grained detail for UHD image deblurring tasks. To further alleviate these issues, we propose a novel autoregressive flow method for UHD image deblurring with an ill-conditioned constraint. Our core idea is to decompose UHD restoration into a progressive, coarse-to-fine process: at each scale, the sharp estimate is formed by upsampling the previous-scale result and adding a current-scale residual, enabling stable, stage-wise refinement from low to high resolution. We further introduce Flow Matching to model residual generation as a conditional vector field and perform few-step ODE sampling with efficient Euler/Heun solvers, enriching details while keeping inference affordable. Since multi-step generation at UHD can be numerically unstable, we propose an ill-conditioning suppression scheme by imposing condition-number regularization on a feature-induced attention matrix, improving convergence and cross-scale consistency. Our method demonstrates promising performance on blurred images at 4K (3840$\times$2160) or higher resolutions.
>
---
#### [new 035] Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决视觉诱导的幻觉问题。提出CIPHER方法，通过对抗性图像扰动抑制幻觉，提升模型输出的准确性。**

- **链接: [https://arxiv.org/pdf/2603.10470](https://arxiv.org/pdf/2603.10470)**

> **作者:** Hamidreza Dastmalchi; Aijun An; Ali Cheraghian; Hamed Barzamini
>
> **备注:** CVPR 2026
>
> **摘要:** While large vision-language models (LVLMs) achieve strong performance on multimodal tasks, they frequently generate hallucinations -- unfaithful outputs misaligned with the visual input. To address this issue, we introduce CIPHER (Counterfactual Image Perturbations for Hallucination Extraction and Removal), a training-free method that suppresses vision-induced hallucinations via lightweight feature-level correction. Unlike prior training-free approaches that primarily focus on text-induced hallucinations, CIPHER explicitly targets hallucinations arising from the visual modality. CIPHER operates in two phases. In the offline phase, we construct OHC-25K (Object-Hallucinated Counterfactuals, 25,000 samples), a counterfactual dataset consisting of diffusion-edited images that intentionally contradict the original ground-truth captions. We pair these edited images with the unchanged ground-truth captions and process them through an LVLM to extract hallucination-related representations. Contrasting these representations with those from authentic (image, caption) pairs reveals structured, systematic shifts spanning a low-rank subspace characterizing vision-induced hallucination. In the inference phase, CIPHER suppresses hallucinations by projecting intermediate hidden states away from this subspace. Experiments across multiple benchmarks show that CIPHER significantly reduces hallucination rates while preserving task performance, demonstrating the effectiveness of counterfactual visual perturbations for improving LVLM faithfulness. Code and additional materials are available at this https URL.
>
---
#### [new 036] CodePercept: Code-Grounded Visual STEM Perception for MLLMs
- **分类: cs.CV**

- **简介: 该论文属于STEM视觉推理任务，旨在解决MLLMs在STEM领域视觉感知不足的问题。通过构建包含代码的图像-描述-代码三元组数据集，提升模型的视觉感知能力。**

- **链接: [https://arxiv.org/pdf/2603.10757](https://arxiv.org/pdf/2603.10757)**

> **作者:** Tongkun Guan; Zhibo Yang; Jianqiang Wan; Mingkun Yang; Zhengtao Guo; Zijian Hu; Ruilin Luo; Ruize Chen; Songtao Jiang; Peng Wang; Wei Shen; Junyang Lin; Xiaokang Yang
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** When MLLMs fail at Science, Technology, Engineering, and Mathematics (STEM) visual reasoning, a fundamental question arises: is it due to perceptual deficiencies or reasoning limitations? Through systematic scaling analysis that independently scales perception and reasoning components, we uncover a critical insight: scaling perception consistently outperforms scaling reasoning. This reveals perception as the true lever limiting current STEM visual reasoning. Motivated by this insight, our work focuses on systematically enhancing the perception capabilities of MLLMs by establishing code as a powerful perceptual medium--executable code provides precise semantics that naturally align with the structured nature of STEM visuals. Specifically, we construct ICC-1M, a large-scale dataset comprising 1M Image-Caption-Code triplets that materializes this code-as-perception paradigm through two complementary approaches: (1) Code-Grounded Caption Generation treats executable code as ground truth for image captions, eliminating the hallucinations inherent in existing knowledge distillation methods; (2) STEM Image-to-Code Translation prompts models to generate reconstruction code, mitigating the ambiguity of natural language for perception enhancement. To validate this paradigm, we further introduce STEM2Code-Eval, a novel benchmark that directly evaluates visual perception in STEM domains. Unlike existing work relying on problem-solving accuracy as a proxy that only measures problem-relevant understanding, our benchmark requires comprehensive visual comprehension through executable code generation for image reconstruction, providing deterministic and verifiable assessment. Code is available at this https URL.
>
---
#### [new 037] UniCom: Unified Multimodal Modeling via Compressed Continuous Semantic Representations
- **分类: cs.CV**

- **简介: 该论文提出UniCom，解决统一多模态模型中语义信息丢失与生成不稳定的问题。通过压缩连续语义表示，提升图像生成与编辑性能。**

- **链接: [https://arxiv.org/pdf/2603.10702](https://arxiv.org/pdf/2603.10702)**

> **作者:** Yaqi Zhao; Wang Lin; Zijian Zhang; Miles Yang; Jingyuan Chen; Wentao Zhang; Zhao Zhong; Liefeng Bo
>
> **摘要:** Current unified multimodal models typically rely on discrete visual tokenizers to bridge the modality gap. However, discretization inevitably discards fine-grained semantic information, leading to suboptimal performance in visual understanding tasks. Conversely, directly modeling continuous semantic representations (e.g., CLIP, SigLIP) poses significant challenges in high-dimensional generative modeling, resulting in slow convergence and training instability. To resolve this dilemma, we introduce UniCom, a unified framework that harmonizes multimodal understanding and generation via compressed continuous representation. We empirically demonstrate that reducing channel dimension is significantly more effective than spatial downsampling for both reconstruction and generation. Accordingly, we design an attention-based semantic compressor to distill dense features into a compact unified representation. Furthermore, we validate that the transfusion architecture surpasses query-based designs in convergence and consistency. Experiments demonstrate that UniCom achieves state-of-the-art generation performance among unified models. Notably, by preserving rich semantic priors, it delivers exceptional controllability in image editing and maintains image consistency even without relying on VAE.
>
---
#### [new 038] Backdoor Directions in Vision Transformers
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于安全与隐私任务，研究视觉Transformer中的后门攻击。通过分析触发方向，揭示后门机制并提出检测方法。**

- **链接: [https://arxiv.org/pdf/2603.10806](https://arxiv.org/pdf/2603.10806)**

> **作者:** Sengim Karayalcin; Marina Krcek; Pin-Yu Chen; Stjepan Picek
>
> **备注:** 31 pages, 16 figures
>
> **摘要:** This paper investigates how Backdoor Attacks are represented within Vision Transformers (ViTs). By assuming knowledge of the trigger, we identify a specific ``trigger direction'' in the model's activations that corresponds to the internal representation of the trigger. We confirm the causal role of this linear direction by showing that interventions in both activation and parameter space consistently modulate the model's backdoor behavior across multiple datasets and attack types. Using this direction as a diagnostic tool, we trace how backdoor features are processed across layers. Our analysis reveals distinct qualitative differences: static-patch triggers follow a different internal logic than stealthy, distributed triggers. We further examine the link between backdoors and adversarial attacks, specifically testing whether PGD-based perturbations (de-)activate the identified trigger mechanism. Finally, we propose a data-free, weight-based detection scheme for stealthy-trigger attacks. Our findings show that mechanistic interpretability offers a robust framework for diagnosing and addressing security vulnerabilities in computer vision.
>
---
#### [new 039] Prompting with the human-touch: evaluating model-sensitivity of foundation models for musculoskeletal CT segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，旨在评估基础模型在人体骨骼CT分割中的敏感性，解决模型性能比较困难及人类提示影响效果的问题。**

- **链接: [https://arxiv.org/pdf/2603.10541](https://arxiv.org/pdf/2603.10541)**

> **作者:** Caroline Magg; Maaike A. ter Wee; Johannes G.G. Dobbe; Geert J. Streekstra; Leendert Blankevoort; Clara I. Sánchez; Hoel Kervadec
>
> **摘要:** Promptable Foundation Models (FMs), initially introduced for natural image segmentation, have also revolutionized medical image segmentation. The increasing number of models, along with evaluations varying in datasets, metrics, and compared models, makes direct performance comparison between models difficult and complicates the selection of the most suitable model for specific clinical tasks. In our study, 11 promptable FMs are tested using non-iterative 2D and 3D prompting strategies on a private and public dataset focusing on bone and implant segmentation in four anatomical regions (wrist, shoulder, hip and lower leg). The Pareto-optimal models are identified and further analyzed using human prompts collected through a dedicated observer study. Our findings are: 1) The segmentation performance varies a lot between FMs and prompting strategies; 2) The Pareto-optimal models in 2D are SAM and SAM2.1, in 3D nnInteractive and Med-SAM2; 3) Localization accuracy and rater consistency vary with anatomical structures, with higher consistency for simple structures (wrist bones) and lower consistency for complex structures (pelvis, tibia, implants); 4) The segmentation performance drops using human prompts, suggesting that performance reported on "ideal" prompts extracted from reference labels might overestimate the performance in a human-driven setting; 5) All models were sensitive to prompt variations. While two models demonstrated intra-rater robustness, it did not scale to inter-rater settings. We conclude that the selection of the most optimal FM for a human-driven setting remains challenging, with even high-performing FMs being sensitive to variations in human input prompts. Our code base for prompt extraction and model inference is available: this https URL
>
---
#### [new 040] Joint Imaging-ROI Representation Learning via Cross-View Contrastive Alignment for Brain Disorder Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于脑疾病分类任务，旨在解决传统方法中全局影像与ROI图表示的互补性不足问题。通过跨视角对比对齐，联合学习影像与ROI表示，提升分类性能。**

- **链接: [https://arxiv.org/pdf/2603.10253](https://arxiv.org/pdf/2603.10253)**

> **作者:** Wei Liang; Lifang He
>
> **摘要:** Brain imaging classification is commonly approached from two perspectives: modeling the full image volume to capture global anatomical context, or constructing ROI-based graphs to encode localized and topological interactions. Although both representations have demonstrated independent efficacy, their relative contributions and potential complementarity remain insufficiently understood. Existing fusion approaches are typically task-specific and do not enable controlled evaluation of each representation under consistent training settings. To address this gap, we propose a unified cross-view contrastive framework for joint imaging-ROI representation learning. Our method learns subject-level global (imaging) and local (ROI-graph) embeddings and aligns them in a shared latent space using a bidirectional contrastive objective, encouraging representations from the same subject to converge while separating those from different subjects. This alignment produces comparable embeddings suitable for downstream fusion and enables systematic evaluation of imaging-only, ROI-only, and joint configurations within a unified training protocol. Extensive experiments on the ADHD-200 and ABIDE datasets demonstrate that joint learning consistently improves classification performance over either branch alone across multiple backbone choices. Moreover, interpretability analyses reveal that imaging-based and ROI-based branches emphasize distinct yet complementary discriminative patterns, explaining the observed performance gains. These findings provide principled evidence that explicitly integrating global volumetric and ROI-level representations is a promising direction for neuroimaging-based brain disorder classification. The source code is available at this https URL.
>
---
#### [new 041] Need for Speed: Zero-Shot Depth Completion with Single-Step Diffusion
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出Marigold-SSD，解决深度补全任务中的效率问题。通过单步扩散方法，提升推理速度并减少训练成本，实现高效且泛化的3D感知。**

- **链接: [https://arxiv.org/pdf/2603.10584](https://arxiv.org/pdf/2603.10584)**

> **作者:** Jakub Gregorek; Paraskevas Pegios; Nando Metzger; Konrad Schindler; Theodora Kontogianni; Lazaros Nalpantidis
>
> **摘要:** We introduce Marigold-SSD, a single-step, late-fusion depth completion framework that leverages strong diffusion priors while eliminating the costly test-time optimization typically associated with diffusion-based methods. By shifting computational burden from inference to finetuning, our approach enables efficient and robust 3D perception under real-world latency constraints. Marigold-SSD achieves significantly faster inference with a training cost of only 4.5 GPU days. We evaluate our method across four indoor and two outdoor benchmarks, demonstrating strong cross-domain generalization and zero-shot performance compared to existing depth completion approaches. Our approach significantly narrows the efficiency gap between diffusion-based and discriminative models. Finally, we challenge common evaluation protocols by analyzing performance under varying input sparsity levels. Page: this https URL
>
---
#### [new 042] IMTBench: A Multi-Scenario Cross-Modal Collaborative Evaluation Benchmark for In-Image Machine Translation
- **分类: cs.CV**

- **简介: 该论文提出IMTBench，一个用于图像内机器翻译的多场景跨模态评估基准，解决现有数据不真实、评估不全面的问题。**

- **链接: [https://arxiv.org/pdf/2603.10495](https://arxiv.org/pdf/2603.10495)**

> **作者:** Jiahao Lyu; Pei Fu; Zhenhang Li; Weichao Zeng; Shaojie Zhan; Jiahui Yang; Can Ma; Yu Zhou; Zhenbo Luo; Jian Luan
>
> **摘要:** End-to-end In-Image Machine Translation (IIMT) aims to convert text embedded within an image into a target language while preserving the original visual context, layout, and rendering style. However, existing IIMT benchmarks are largely synthetic and thus fail to reflect real-world complexity, while current evaluation protocols focus on single-modality metrics and overlook cross-modal faithfulness between rendered text and model outputs. To address these shortcomings, we present In-image Machine Translation Benchmark (IMTBench), a new benchmark of 2,500 image translation samples covering four practical scenarios and nine languages. IMTBench supports multi-aspect evaluation, including translation quality, background preservation, overall image quality, and a cross-modal alignment score that measures consistency between the translated text produced by the model and the text rendered in the translated image. We benchmark strong commercial cascade systems, and both closed- and open-source unified multi-modal models, and observe large performance gaps across scenarios and languages, especially on natural scenes and resource-limited languages, highlighting substantial headroom for end-to-end image text translation. We hope IMTBench establishes a standardized benchmark to accelerate progress in this emerging task.
>
---
#### [new 043] Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，解决扩散Transformer计算成本高的问题。提出JiT框架，在不训练的情况下通过空间加速提升推理速度，实现7倍提速且性能损失小。**

- **链接: [https://arxiv.org/pdf/2603.10744](https://arxiv.org/pdf/2603.10744)**

> **作者:** Wenhao Sun; Ji Li; Zhaoqiang Liu
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Diffusion Transformers have established a new state-of-the-art in image synthesis, but the high computational cost of iterative sampling severely hampers their practical deployment. While existing acceleration methods often focus on the temporal domain, they overlook the substantial spatial redundancy inherent in the generative process, where global structures emerge long before fine-grained details are formed. The uniform computational treatment of all spatial regions represents a critical inefficiency. In this paper, we introduce Just-in-Time (JiT), a novel training-free framework that addresses this challenge by acceleration in the spatial domain. JiT formulates a spatially approximated generative ordinary differential equation (ODE) that drives the full latent state evolution based on computations from a dynamically selected, sparse subset of anchor tokens. To ensure seamless transitions as new tokens are incorporated to expand the dimensions of the latent state, we propose a deterministic micro-flow, a simple and effective finite-time ODE that maintains both structural coherence and statistical correctness. Extensive experiments on the state-of-the-art FLUX.1-dev model demonstrate that JiT achieves up to a 7x speedup with nearly lossless performance, significantly outperforming existing acceleration methods and establishing a new and superior trade-off between inference speed and generation fidelity.
>
---
#### [new 044] Towards Cognitive Defect Analysis in Active Infrared Thermography with Vision-Text Cues
- **分类: cs.CV; cs.AI; eess.SP**

- **简介: 该论文属于缺陷检测任务，旨在解决CFRP检测中数据不足的问题。通过引入视觉-语言模型和适配器，实现无需训练数据的零样本缺陷识别与定位。**

- **链接: [https://arxiv.org/pdf/2603.10549](https://arxiv.org/pdf/2603.10549)**

> **作者:** Mohammed Salah; Eman Ouda; Giuseppe Dell'Avvocato; Fabrizio Sarasini; Ester D'Accardi; Jorge Dias; Davor Svetinovic; Stefano Sfarra; Yusra Abdulrahman
>
> **摘要:** Active infrared thermography (AIRT) is currently witnessing a surge of artificial intelligence (AI) methodologies being deployed for automated subsurface defect analysis of high performance carbon fiber-reinforced polymers (CFRP). Deploying AI-based AIRT methodologies for inspecting CFRPs requires the creation of time consuming and expensive datasets of CFRP inspection sequences to train neural networks. To address this challenge, this work introduces a novel language-guided framework for cognitive defect analysis in CFRPs using AIRT and vision-language models (VLMs). Unlike conventional learning-based approaches, the proposed framework does not require developing training datasets for extensive training of defect detectors, instead it relies solely on pretrained multimodal VLM encoders coupled with a lightweight adapter to enable generative zero-shot understanding and localization of subsurface defects. By leveraging pretrained multimodal encoders, the proposed system enables generative zero-shot understanding of thermographic patterns and automatic detection of subsurface defects. Given the domain gap between thermographic data and natural images used to train VLMs, an AIRT-VLM Adapter is proposed to enhance the visibility of defects while aligning the thermographic domain with the learned representations of VLMs. The proposed framework is validated using three representative VLMs; specifically, GroundingDINO, Qwen-VL-Chat, and CogVLM. Validation is performed on 25 CFRP inspection sequences with impacts introduced at different energy levels, reflecting realistic defects encountered in industrial scenarios. Experimental results demonstrate that the AIRT-VLM adapter achieves signal-to-noise ratio (SNR) gains exceeding 10 dB compared with conventional thermographic dimensionality-reduction methods, while enabling zero-shot defect detection with intersection-over-union values reaching 70%.
>
---
#### [new 045] On the Reliability of Cue Conflict and Beyond
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于模型偏差分析任务，旨在解决现有基准在评估形状与纹理依赖性时的不稳定问题。提出REFINED-BIAS框架，实现更可靠、可解释的偏差诊断。**

- **链接: [https://arxiv.org/pdf/2603.10834](https://arxiv.org/pdf/2603.10834)**

> **作者:** Pum Jun Kim; Seung-Ah Lee; Seongho Park; Dongyoon Han; Jaejun Yoo
>
> **备注:** Shape-Texture Bias, Cue Conflict Benchmark
>
> **摘要:** Understanding how neural networks rely on visual cues offers a human-interpretable view of their internal decision processes. The cue-conflict benchmark has been influential in probing shape-texture preference and in motivating the insight that stronger, human-like shape bias is often associated with improved in-domain performance. However, we find that the current stylization-based instantiation can yield unstable and ambiguous bias estimates. Specifically, stylization may not reliably instantiate perceptually valid and separable cues nor control their relative informativeness, ratio-based bias can obscure absolute cue sensitivity, and restricting evaluation to preselected classes can distort model predictions by ignoring the full decision space. Together, these factors can confound preference with cue validity, cue balance, and recognizability artifacts. We introduce REFINED-BIAS, an integrated dataset and evaluation framework for reliable and interpretable shape-texture bias diagnosis. REFINED-BIAS constructs balanced, human- and model- recognizable cue pairs using explicit definitions of shape and texture, and measures cue-specific sensitivity over the full label space via a ranking-based metric, enabling fairer cross-model comparisons. Across diverse training regimes and architectures, REFINED-BIAS enables fairer cross-model comparison, more faithful diagnosis of shape and texture biases, and clearer empirical conclusions, resolving inconsistencies that prior cue-conflict evaluations could not reliably disambiguate.
>
---
#### [new 046] Evaluating Few-Shot Pill Recognition Under Visual Domain Shift
- **分类: cs.CV**

- **简介: 该论文属于少样本药物识别任务，旨在解决实际部署中视觉域迁移带来的挑战。通过两阶段检测框架，研究少样本微调效果，评估模型在复杂场景下的泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.10833](https://arxiv.org/pdf/2603.10833)**

> **作者:** W. I. Chu; G. Tarroni; L. Li
>
> **备注:** 8 pages, 4 figures. Submitted to IEEE Engineering in Medicine and Biology Conference (EMBC) 2026
>
> **摘要:** Adverse drug events are a significant source of preventable harm, which has led to the development of automated pill recognition systems to enhance medication safety. Real-world deployment of these systems is hindered by visually complex conditions, including cluttered scenes, overlapping pills, reflections, and diverse acquisition environments. This study investigates few-shot pill recognition from a deployment-oriented perspective, prioritizing generalization under realistic cross-dataset domain shifts over architectural innovation. A two-stage object detection framework is employed, involving base training followed by few-shot fine-tuning. Models are adapted to novel pill classes using one, five, or ten labeled examples per class and are evaluated on a separate deployment dataset featuring multi-object, cluttered scenes. The evaluation focuses on classification-centric and error-based metrics to address heterogeneous annotation strategies. Findings indicate that semantic pill recognition adapts rapidly with few-shot supervision, with classification performance reaching saturation even with a single labeled example. However, stress testing under overlapping and occluded conditions demonstrates a marked decline in localization and recall, despite robust semantic classification. Models trained on visually realistic, multi-pill data consistently exhibit greater robustness in low-shot scenarios, underscoring the importance of training data realism and the diagnostic utility of few-shot fine-tuning for deployment readiness.
>
---
#### [new 047] Less is More: Decoder-Free Masked Modeling for Efficient Skeleton Representation Learning
- **分类: cs.CV**

- **简介: 该论文属于骨架动作表示学习任务，解决CL和MAE的局限性。提出SLiM框架，结合掩码建模与对比学习，无需解码器，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.10648](https://arxiv.org/pdf/2603.10648)**

> **作者:** Jeonghyeok Do; Yun Chen; Geunhyuk Youk; Munchurl Kim
>
> **备注:** Please visit our project page at this https URL
>
> **摘要:** The landscape of skeleton-based action representation learning has evolved from Contrastive Learning (CL) to Masked Auto-Encoder (MAE) architectures. However, each paradigm faces inherent limitations: CL often overlooks fine-grained local details, while MAE is burdened by computationally heavy decoders. Moreover, MAE suffers from severe computational asymmetry -- benefiting from efficient masking during pre-training but requiring exhaustive full-sequence processing for downstream tasks. To resolve these bottlenecks, we propose SLiM (Skeleton Less is More), a novel unified framework that harmonizes masked modeling with contrastive learning via a shared encoder. By eschewing the reconstruction decoder, SLiM not only eliminates computational redundancy but also compels the encoder to capture discriminative features directly. SLiM is the first framework with decoder-free masked modeling of representative learning. Crucially, to prevent trivial reconstruction arising from high skeletal-temporal correlation, we introduce semantic tube masking, alongside skeletal-aware augmentations designed to ensure anatomical consistency across diverse temporal granularities. Extensive experiments demonstrate that SLiM consistently achieves state-of-the-art performance across all downstream protocols. Notably, our method delivers this superior accuracy with exceptional efficiency, reducing inference computational cost by 7.89x compared to existing MAE methods.
>
---
#### [new 048] Visually-Guided Controllable Medical Image Generation via Fine-Grained Semantic Disentanglement
- **分类: cs.CV**

- **简介: 该论文属于医学图像生成任务，旨在解决文本与图像间的语义纠缠问题，提升生成可控性。通过引入跨模态对齐和特征融合模块，实现更精确的结构控制。**

- **链接: [https://arxiv.org/pdf/2603.10519](https://arxiv.org/pdf/2603.10519)**

> **作者:** Xin Huang; Junjie Liang; Qingshan Hou; Peng Cao; Jinzhu Yang; Xiaoli Liu; Osmar R. Zaiane
>
> **备注:** 10 pages, 7 figures. Currently under review
>
> **摘要:** Medical image synthesis is crucial for alleviating data scarcity and privacy constraints. However, fine-tuning general text-to-image (T2I) models remains challenging, mainly due to the significant modality gap between complex visual details and abstract clinical text. In addition, semantic entanglement persists, where coarse-grained text embeddings blur the boundary between anatomical structures and imaging styles, thus weakening controllability during generation. To address this, we propose a Visually-Guided Text Disentanglement framework. We introduce a cross-modal latent alignment mechanism that leverages visual priors to explicitly disentangle unstructured text into independent semantic representations. Subsequently, a Hybrid Feature Fusion Module (HFFM) injects these features into a Diffusion Transformer (DiT) through separated channels, enabling fine-grained structural control. Experimental results in three datasets demonstrate that our method outperforms existing approaches in terms of generation quality and significantly improves performance on downstream classification tasks. The source code is available at this https URL.
>
---
#### [new 049] Are Video Reasoning Models Ready to Go Outside?
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频推理任务，旨在解决模型在真实环境扰动下的性能下降问题。提出ROVA框架和PVRBench基准，提升模型在复杂场景中的鲁棒性与推理能力。**

- **链接: [https://arxiv.org/pdf/2603.10652](https://arxiv.org/pdf/2603.10652)**

> **作者:** Yangfan He; Changgyu Boo; Jaehong Yoon
>
> **备注:** Project Page: this https URL
>
> **摘要:** In real-world deployment, vision-language models often encounter disturbances such as weather, occlusion, and camera motion. Under such conditions, their understanding and reasoning degrade substantially, revealing a gap between clean, controlled (i.e., unperturbed) evaluation settings and real-world robustness. To address this limitation, we propose ROVA, a novel training framework that improves robustness by modeling a robustness-aware consistency reward under spatio-temporal corruptions. ROVA introduces a difficulty-aware online training strategy that prioritizes informative samples based on the model's evolving capability. Specifically, it continuously re-estimates sample difficulty via self-reflective evaluation, enabling adaptive training with a robustness-aware consistency reward. We also introduce PVRBench, a new benchmark that injects real-world perturbations into embodied video datasets to assess both accuracy and reasoning quality under realistic disturbances. We evaluate ROVA and baselines on PVRBench, UrbanVideo, and VisBench, where open-source and proprietary models suffer up to 35% and 28% drops in accuracy and reasoning under realistic perturbations. ROVA effectively mitigates performance degradation, boosting relative accuracy by at least 24% and reasoning by over 9% compared with baseline models (QWen2.5/3-VL, InternVL2.5, Embodied-R). These gains transfer to clean standard benchmarks, yielding consistent improvements.
>
---
#### [new 050] BALD-SAM: Disagreement-based Active Prompting in Interactive Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于交互式分割任务，旨在提升标注效率。提出BALD-SAM方法，通过量化不确定性选择最优提示位置，提高分割质量。**

- **链接: [https://arxiv.org/pdf/2603.10828](https://arxiv.org/pdf/2603.10828)**

> **作者:** Prithwijit Chowdhury; Mohit Prabhushankar; Ghassan AlRegib
>
> **摘要:** The Segment Anything Model (SAM) has revolutionized interactive segmentation through spatial prompting. While existing work primarily focuses on automating prompts in various settings, real-world annotation workflows involve iterative refinement where annotators observe model outputs and strategically place prompts to resolve ambiguities. Current pipelines typically rely on the annotator's visual assessment of the predicted mask quality. We postulate that a principled approach for automated interactive prompting is to use a model-derived criterion to identify the most informative region for the next prompt. In this work, we establish active prompting: a spatial active learning approach where locations within images constitute an unlabeled pool and prompts serve as queries to prioritize information-rich regions, increasing the utility of each interaction. We further present BALD-SAM: a principled framework adapting Bayesian Active Learning by Disagreement (BALD) to spatial prompt selection by quantifying epistemic uncertainty. To do so, we freeze the entire model and apply Bayesian uncertainty modeling only to a small learned prediction head, making intractable uncertainty estimation practical for large multi-million parameter foundation models. Across 16 datasets spanning natural, medical, underwater, and seismic domains, BALD-SAM demonstrates strong cross-domain performance, ranking first or second on 14 of 16 benchmarks. We validate these gains through a comprehensive ablation suite covering 3 SAM backbones and 35 Laplace posterior configurations, amounting to 38 distinct ablation settings. Beyond strong average performance, BALD-SAM surpasses human prompting and, in several categories, even oracle prompting, while consistently outperforming one-shot baselines in final segmentation quality, particularly on thin and structurally complex objects.
>
---
#### [new 051] Unbalanced Optimal Transport Dictionary Learning for Unsupervised Hyperspectral Image Clustering
- **分类: cs.CV; cs.LG; math.ST**

- **简介: 该论文属于无监督 hyperspectral 图像聚类任务，旨在解决传统方法依赖数据平衡、鲁棒性差的问题。通过引入非平衡 Wasserstein 中心，学习低维表示并提升聚类效果。**

- **链接: [https://arxiv.org/pdf/2603.10132](https://arxiv.org/pdf/2603.10132)**

> **作者:** Joshua Lentz; Nicholas Karris; Alex Cloninger; James M. Murphy
>
> **备注:** IEEE WHISPERS 2025
>
> **摘要:** Hyperspectral images capture vast amounts of high-dimensional spectral information about a scene, making labeling an intensive task that is resistant to out-of-the-box statistical methods. Unsupervised learning of clusters allows for automated segmentation of the scene, enabling a more rapid understanding of the image. Partitioning the spectral information contained within the data via dictionary learning in Wasserstein space has proven an effective method for unsupervised clustering. However, this approach requires balancing the spectral profiles of the data, blurring the classes, and sacrificing robustness to outliers and noise. In this paper, we suggest improving this approach by utilizing unbalanced Wasserstein barycenters to learn a lower-dimensional representation of the underlying data. The deployment of spectral clustering on the learned representation results in an effective approach for the unsupervised learning of labels.
>
---
#### [new 052] UltrasoundAgents: Hierarchical Multi-Agent Evidence-Chain Reasoning for Breast Ultrasound Diagnosis
- **分类: cs.CV**

- **简介: 该论文属于乳腺超声诊断任务，旨在提升诊断准确性与可解释性。提出Hierarchical Multi-Agent框架，分步处理病变定位、特征分析与证据整合，解决现有方法证据弱、难以审计的问题。**

- **链接: [https://arxiv.org/pdf/2603.10852](https://arxiv.org/pdf/2603.10852)**

> **作者:** Yali Zhu; Kang Zhou; Dingbang Wu; Gaofeng Meng
>
> **摘要:** Breast ultrasound diagnosis typically proceeds from global lesion localization to local sign assessment and then evidence integration to assign a BI-RADS category and determine benignity or malignancy. Many existing methods rely on end-to-end prediction or provide only weakly grounded evidence, which can miss fine-grained lesion cues and limit auditability and clinical review. To align with the clinical workflow and improve evidence traceability, we propose a hierarchical multi-agent framework, termed UltrasoundAgents. A main agent localizes the lesion in the full image and triggers a crop-and-zoom operation. A sub-agent analyzes the local view and predicts four clinically relevant attributes, namely echogenicity pattern, calcification, boundary type, and edge (margin) morphology. The main agent then integrates these structured attributes to perform evidence-based reasoning and output the BI-RADS category and the malignancy prediction, while producing reviewable intermediate evidence. Furthermore, hierarchical multi-agent training often suffers from error propagation, difficult credit assignment, and sparse rewards. To alleviate this and improve training stability, we introduce a decoupled progressive training strategy. We first train the attribute agent, then train the main agent with oracle attributes to learn robust attribute-based reasoning, and finally apply corrective trajectory self-distillation with spatial supervision to build high-quality trajectories for supervised fine-tuning, yielding a deployable end-to-end policy. Experiments show consistent gains over strong vision-language baselines in diagnostic accuracy and attribute agreement, together with structured evidence and traceable reasoning.
>
---
#### [new 053] FusionNet: a frame interpolation network for 4D heart models
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出FusionNet，用于从短时CMR图像生成高时间分辨率的4D心脏模型，解决心脏运动可视化中时间分辨率不足的问题。**

- **链接: [https://arxiv.org/pdf/2603.10212](https://arxiv.org/pdf/2603.10212)**

> **作者:** Chujie Chang; Shoko Miyauchi; Ken'ichi Morooka; Ryo Kurazume; Oscar Martinez Mozos
>
> **备注:** This is the authors' version. The final authenticated version is available online at this https URL. Published in Medical Image Computing and Computer Assisted Intervention - MICCAI 2023 Workshops
>
> **摘要:** Cardiac magnetic resonance (CMR) imaging is widely used to visualise cardiac motion and diagnose heart disease. However, standard CMR imaging requires patients to lie still in a confined space inside a loud machine for 40-60 min, which increases patient discomfort. In addition, shorter scan times decrease either or both the temporal and spatial resolutions of cardiac motion, and thus, the diagnostic accuracy of the procedure. Of these, we focus on reduced temporal resolution and propose a neural network called FusionNet to obtain four-dimensional (4D) cardiac motion with high temporal resolution from CMR images captured in a short period of time. The model estimates intermediate 3D heart shapes based on adjacent shapes. The results of an experimental evaluation of the proposed FusionNet model showed that it achieved a performance of over 0.897 in terms of the Dice coefficient, confirming that it can recover shapes more precisely than existing methods. This code is available at: this https URL
>
---
#### [new 054] COMIC: Agentic Sketch Comedy Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.MA; cs.NE**

- **简介: 该论文提出COMIC系统，属于喜剧视频生成任务，旨在自动化创作高质量喜剧小品。通过模拟制作团队角色，结合LLM评价体系，提升生成内容的趣味性与多样性。**

- **链接: [https://arxiv.org/pdf/2603.11048](https://arxiv.org/pdf/2603.11048)**

> **作者:** Susung Hong; Brian Curless; Ira Kemelmacher-Shlizerman; Steve Seitz
>
> **备注:** Project page: this https URL
>
> **摘要:** We propose a fully automated AI system that produces short comedic videos similar to sketch shows such as Saturday Night Live. Starting with character references, the system employs a population of agents loosely based on real production studio roles, structured to optimize the quality and diversity of ideas and outputs through iterative competition, evaluation, and improvement. A key contribution is the introduction of LLM critics aligned with real viewer preferences through the analysis of a corpus of comedy videos on YouTube to automatically evaluate humor. Our experiments show that our framework produces results approaching the quality of professionally produced sketches while demonstrating state-of-the-art performance in video generation.
>
---
#### [new 055] DynVLA: Learning World Dynamics for Action Reasoning in Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DynVLA模型，解决自动驾驶中的决策问题。通过引入Dynamics CoT，预测世界动态以提升决策质量。**

- **链接: [https://arxiv.org/pdf/2603.11041](https://arxiv.org/pdf/2603.11041)**

> **作者:** Shuyao Shang; Bing Zhan; Yunfei Yan; Yuqi Wang; Yingyan Li; Yasong An; Xiaoman Wang; Jierui Liu; Lu Hou; Lue Fan; Zhaoxiang Zhang; Tieniu Tan
>
> **备注:** 18 pages, 10 figures
>
> **摘要:** We propose DynVLA, a driving VLA model that introduces a new CoT paradigm termed Dynamics CoT. DynVLA forecasts compact world dynamics before action generation, enabling more informed and physically grounded decision-making. To obtain compact dynamics representations, DynVLA introduces a Dynamics Tokenizer that compresses future evolution into a small set of dynamics tokens. Considering the rich environment dynamics in interaction-intensive driving scenarios, DynVLA decouples ego-centric and environment-centric dynamics, yielding more accurate world dynamics modeling. We then train DynVLA to generate dynamics tokens before actions through SFT and RFT, improving decision quality while maintaining latency-efficient inference. Compared to Textual CoT, which lacks fine-grained spatiotemporal understanding, and Visual CoT, which introduces substantial redundancy due to dense image prediction, Dynamics CoT captures the evolution of the world in a compact, interpretable, and efficient form. Extensive experiments on NAVSIM, Bench2Drive, and a large-scale in-house dataset demonstrate that DynVLA consistently outperforms Textual CoT and Visual CoT methods, validating the effectiveness and practical value of Dynamics CoT.
>
---
#### [new 056] Geometric Autoencoder for Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决扩散模型中潜在空间设计的挑战。提出Geometric Autoencoder（GAE），优化语义监督、重建精度与压缩效率，提升生成质量与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.10365](https://arxiv.org/pdf/2603.10365)**

> **作者:** Hangyu Liu; Jianyong Wang; Yutao Sun
>
> **备注:** Code and models are publicly available at this https URL
>
> **摘要:** Latent diffusion models have established a new state-of-the-art in high-resolution visual generation. Integrating Vision Foundation Model priors improves generative efficiency, yet existing latent designs remain largely heuristic. These approaches often struggle to unify semantic discriminability, reconstruction fidelity, and latent compactness. In this paper, we propose Geometric Autoencoder (GAE), a principled framework that systematically addresses these challenges. By analyzing various alignment paradigms, GAE constructs an optimized low-dimensional semantic supervision target from VFMs to provide guidance for the autoencoder. Furthermore, we leverage latent normalization that replaces the restrictive KL-divergence of standard VAEs, enabling a more stable latent manifold specifically optimized for diffusion learning. To ensure robust reconstruction under high-intensity noise, GAE incorporates a dynamic noise sampling mechanism. Empirically, GAE achieves compelling performance on the ImageNet-1K $256 \times 256$ benchmark, reaching a gFID of 1.82 at only 80 epochs and 1.31 at 800 epochs without Classifier-Free Guidance, significantly surpassing existing state-of-the-art methods. Beyond generative quality, GAE establishes a superior equilibrium between compression, semantic depth and robust reconstruction stability. These results validate our design considerations, offering a promising paradigm for latent diffusion modeling. Code and models are publicly available at this https URL.
>
---
#### [new 057] Bioinspired CNNs for border completion in occluded images
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，旨在解决 occluded 图像的边界补全问题。通过借鉴视觉皮层的数学模型，设计增强鲁棒性的 CNN 滤波器，并在多个数据集上验证效果。**

- **链接: [https://arxiv.org/pdf/2603.10694](https://arxiv.org/pdf/2603.10694)**

> **作者:** Catarina P. Coutinho; Aneeqa Merhab; Janko Petkovic; Ferdinando Zanchetta; Rita Fioresi
>
> **备注:** Submitted for Publication
>
> **摘要:** We exploit the mathematical modeling of the border completion problem in the visual cortex to design convolutional neural network (CNN) filters that enhance robustness to image occlusions. We evaluate our CNN architecture, BorderNet, on three occluded datasets (MNIST, Fashion-MNIST, and EMNIST) under two types of occlusions: stripes and grids. In all cases, BorderNet demonstrates improved performance, with gains varying depending on the severity of the occlusions and the dataset.
>
---
#### [new 058] P-GSVC: Layered Progressive 2D Gaussian Splatting for Scalable Image and Video
- **分类: cs.CV; cs.MM**

- **简介: 该论文提出P-GSVC，解决图像和视频的可扩展高斯表示问题。通过分层渐进结构实现从粗到细的重建，提升重建质量。**

- **链接: [https://arxiv.org/pdf/2603.10551](https://arxiv.org/pdf/2603.10551)**

> **作者:** Longan Wang; Yuang Shi; Wei Tsang Ooi
>
> **备注:** MMSys 2026; Project Website: see this https URL
>
> **摘要:** Gaussian splatting has emerged as a competitive explicit representation for image and video reconstruction. In this work, we present P-GSVC, the first layered progressive 2D Gaussian splatting framework that provides a unified solution for scalable Gaussian representation in both images and videos. P-GSVC organizes 2D Gaussian splats into a base layer and successive enhancement layers, enabling coarse-to-fine reconstructions. To effectively optimize this layered representation, we propose a joint training strategy that simultaneously updates Gaussians across layers, aligning their optimization trajectories to ensure inter-layer compatibility and a stable progressive reconstruction. P-GSVC supports scalability in terms of both quality and resolution. Our experiments show that the joint training strategy can gain up to 1.9 dB improvement in PSNR for video and 2.6 dB improvement in PSNR for image when compared to methods that perform sequential layer-wise training. Project page: this https URL
>
---
#### [new 059] LiTo: Surface Light Field Tokenization
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文提出一种3D潜在表示方法，解决同时建模物体几何与视点相关外观的问题。通过编码RGB-D图像中的表面光场，学习统一的3D潜在空间，生成高质量且符合输入光照的3D物体。**

- **链接: [https://arxiv.org/pdf/2603.11047](https://arxiv.org/pdf/2603.11047)**

> **作者:** Jen-Hao Rick Chang; Xiaoming Zhao; Dorian Chan; Oncel Tuzel
>
> **备注:** ICLR 2026; Project page: this https URL
>
> **摘要:** We propose a 3D latent representation that jointly models object geometry and view-dependent appearance. Most prior works focus on either reconstructing 3D geometry or predicting view-independent diffuse appearance, and thus struggle to capture realistic view-dependent effects. Our approach leverages that RGB-depth images provide samples of a surface light field. By encoding random subsamples of this surface light field into a compact set of latent vectors, our model learns to represent both geometry and appearance within a unified 3D latent space. This representation reproduces view-dependent effects such as specular highlights and Fresnel reflections under complex lighting. We further train a latent flow matching model on this representation to learn its distribution conditioned on a single input image, enabling the generation of 3D objects with appearances consistent with the lighting and materials in the input. Experiments show that our approach achieves higher visual quality and better input fidelity than existing methods.
>
---
#### [new 060] Pointy - A Lightweight Transformer for Point Cloud Foundation Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出一种轻量级Transformer点云架构Pointy，用于点云基础模型任务。解决传统模型依赖大量数据和跨模态监督的问题，通过精心设计的训练集和结构，实现高效性能。**

- **链接: [https://arxiv.org/pdf/2603.10963](https://arxiv.org/pdf/2603.10963)**

> **作者:** Konrad Szafer; Marek Kraft; Dominik Belter
>
> **备注:** To appear in the proceedings of ACIVS 2025. An earlier version was presented at the SCI-FM workshop at ICLR 2025
>
> **摘要:** Foundation models for point cloud data have recently grown in capability, often leveraging extensive representation learning from language or vision. In this work, we take a more controlled approach by introducing a lightweight transformer-based point cloud architecture. In contrast to the heavy reliance on cross-modal supervision, our model is trained only on 39k point clouds - yet it outperforms several larger foundation models trained on over 200k training samples. Interestingly, our method approaches state-of-the-art results from models that have seen over a million point clouds, images, and text samples, demonstrating the value of a carefully curated training setup and architecture. To ensure rigorous evaluation, we conduct a comprehensive replication study that standardizes the training regime and benchmarks across multiple point cloud architectures. This unified experimental framework isolates the impact of architectural choices, allowing for transparent comparisons and highlighting the benefits of our design and other tokenizer-free architectures. Our results show that simple backbones can deliver competitive results to more complex or data-rich strategies. The implementation, including code, pre-trained models, and training protocols, is available at this https URL.
>
---
#### [new 061] One Token, Two Fates: A Unified Framework via Vision Token Manipulation Against MLLMs Hallucination
- **分类: cs.CV**

- **简介: 该论文属于多模态大模型 hallucination 问题研究，旨在解决视觉与语言不平衡导致的错误生成。通过统一框架，利用视觉 token 增强和修正模型偏差，有效减少对象幻觉。**

- **链接: [https://arxiv.org/pdf/2603.10360](https://arxiv.org/pdf/2603.10360)**

> **作者:** Zhan Fa; Yue Duan; Jian Zhang; Lei Qi; Yinghuan Shi
>
> **备注:** 10 pages
>
> **摘要:** Current training-free methods tackle MLLM hallucination with separate strategies: either enhancing visual signals or suppressing text inertia. However, these separate methods are insufficient due to critical trade-offs: simply enhancing vision often fails against strong language prior, while suppressing language can introduce extra image-irrelevant noise. Moreover, we find their naive combination is also ineffective, necessitating a unified framework. We propose such a framework by focusing on the core asset: the vision token. Our design leverages two key insights: (1) augmented images offer complementary visual semantics, and (2) removing vision tokens (information-gap) isolates hallucination tendencies more precisely than distorting images (modality-gap). Based on these, our framework uses vision tokens in two distinct ways, both operating on latent representations: our Synergistic Visual Calibration (SVC) module incorporates augmented tokens to strengthen visual representations, while our Causal Representation Calibration (CRC) module uses pruned tokens to create latent-space negative samples for correcting internal model biases. By harmonizing these two roles, our framework effectively restores the vision-language balance, significantly reducing object hallucinations, improving POPE accuracy by an average of 2% absolute on LLaVA-1.5 across multiple benchmarks with only a 1.06x inference latency overhead.
>
---
#### [new 062] StyleGallery: Training-free and Semantic-aware Personalized Style Transfer from Arbitrary Image References
- **分类: cs.CV**

- **简介: 该论文属于图像风格迁移任务，解决现有方法在语义对齐、依赖约束和全局局部平衡上的不足。提出StyleGallery框架，实现无需训练、语义感知的个性化风格迁移。**

- **链接: [https://arxiv.org/pdf/2603.10354](https://arxiv.org/pdf/2603.10354)**

> **作者:** Boyu He; Yunfan Ye; Chang Liu; Weishang Wu; Fang Liu; Zhiping Cai
>
> **备注:** 10 pages, 23 figures, Conference on Computer Vision and Pattern Recognition 2026
>
> **摘要:** Despite the advancements in diffusion-based image style transfer, existing methods are commonly limited by 1) semantic gap: the style reference could miss proper content semantics, causing uncontrollable stylization; 2) reliance on extra constraints (e.g., semantic masks) restricting applicability; 3) rigid feature associations lacking adaptive global-local alignment, failing to balance fine-grained stylization and global content preservation. These limitations, particularly the inability to flexibly leverage style inputs, fundamentally restrict style transfer in terms of personalization, accuracy, and adaptability. To address these, we propose StyleGallery, a training-free and semantic-aware framework that supports arbitrary reference images as input and enables effective personalized customization. It comprises three core stages: semantic region segmentation (adaptive clustering on latent diffusion features to divide regions without extra inputs); clustered region matching (block filtering on extracted features for precise alignment); and style transfer optimization (energy function-guided diffusion sampling with regional style loss to optimize stylization). Experiments on our introduced benchmark demonstrate that StyleGallery outperforms state-of-the-art methods in content structure preservation, regional stylization, interpretability, and personalized customization, particularly when leveraging multiple style references.
>
---
#### [new 063] R4-CGQA: Retrieval-based Vision Language Models for Computer Graphics Image Quality Assessment
- **分类: cs.CV; cs.DB**

- **简介: 该论文属于图像质量评估任务，旨在解决CG图像质量评价缺乏系统描述和解释的问题。通过构建数据集并改进VLMs的评估能力，提升细粒度CG质量判断。**

- **链接: [https://arxiv.org/pdf/2603.10578](https://arxiv.org/pdf/2603.10578)**

> **作者:** Zhuangzi Li; Jian Jin; Shilv Cai; Weisi Lin
>
> **摘要:** Immersive Computer Graphics (CGs) rendering has become ubiquitous in modern daily life. However, comprehensively evaluating CG quality remains challenging for two reasons: First, existing CG datasets lack systematic descriptions of rendering quality; and second existing CG quality assessment methods cannot provide reasonable text-based explanations. To address these issues, we first identify six key perceptual dimensions of CG quality from the user perspective and construct a dataset of 3500 CG images with corresponding quality descriptions. Each description covers CG style, content, and perceived quality along the selected dimensions. Furthermore, we use a subset of the dataset to build several question-answer benchmarks based on the descriptions in order to evaluate the responses of existing Vision Language Models (VLMs). We find that current VLMs are not sufficiently accurate in judging fine-grained CG quality, but that descriptions of visually similar images can significantly improve a VLM's understanding of a given CG image. Motivated by this observation, we adopt retrieval-augmented generation and propose a two-stream retrieval framework that effectively enhances the CG quality assessment capabilities of VLMs. Experiments on several representative VLMs demonstrate that our method substantially improves their performance on CG quality assessment.
>
---
#### [new 064] LCAMV: High-Accuracy 3D Reconstruction of Color-Varying Objects Using LCA Correction and Minimum-Variance Fusion in Structured Light
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决彩色物体在结构光下的深度误差问题。通过LCA校正和最小方差融合，提升重建精度。**

- **链接: [https://arxiv.org/pdf/2603.10456](https://arxiv.org/pdf/2603.10456)**

> **作者:** Wonbeen Oh; Jae-Sang Hyun
>
> **摘要:** Accurate 3D reconstruction of colored objects with structured light (SL) is hindered by lateral chromatic aberration (LCA) in optical components and uneven noise characteristics across RGB channels. This paper introduces lateral chromatic aberration correction and minimum-variance fusion (LCAMV), a robust 3D reconstruction method that operates with a single projector-camera pair without additional hardware or acquisition constraints. LCAMV analytically models and pixel-wise compensates LCA in both the projector and camera, then adaptively fuses multi-channel phase data using a Poisson-Gaussian noise model and minimum-variance estimation. Unlike existing methods that require extra hardware or multiple exposures, LCAMV enables fast acquisition. Experiments on planar and non-planar colored surfaces show that LCAMV outperforms grayscale conversion and conventional channel-weighting, reducing depth error by up to 43.6\%. These results establish LCAMV as an effective solution for high-precision 3D reconstruction of nonuniformly colored objects.
>
---
#### [new 065] Novel Architecture of RPA In Oral Cancer Lesion Detection
- **分类: cs.CV**

- **简介: 该论文属于口腔癌病灶检测任务，旨在提升检测效率。通过改进RPA架构，提出OC-RPAv1和OC-RPAv2，显著减少预测时间，提高处理速度。**

- **链接: [https://arxiv.org/pdf/2603.10928](https://arxiv.org/pdf/2603.10928)**

> **作者:** Revana Magdy; Joy Naoum; Ali Hamdi
>
> **摘要:** Accurate and early detection of oral cancer lesions is crucial for effective diagnosis and treatment. This study evaluates two RPA implementations, OC-RPAv1 and OC-RPAv2, using a test set of 31 images. OC-RPAv1 processes one image per prediction in an average of 0.29 seconds, while OCRPAv2 employs a Singleton design pattern and batch processing, reducing prediction time to just 0.06 seconds per image. This represents a 60-100x efficiency improvement over standard RPA methods, showcasing that design patterns and batch processing can enhance scalability and reduce costs in oral cancer detection
>
---
#### [new 066] UniStitch: Unifying Semantic and Geometric Features for Image Stitching
- **分类: cs.CV**

- **简介: 该论文属于图像拼接任务，旨在解决传统方法与学习方法分离的问题。提出UniStitch框架，融合语义与几何特征，提升拼接效果。**

- **链接: [https://arxiv.org/pdf/2603.10568](https://arxiv.org/pdf/2603.10568)**

> **作者:** Yuan Mei; Lang Nie; Kang Liao; Yunqiu Xu; Chunyu Lin; Bin Xiao
>
> **备注:** Code:this https URL
>
> **摘要:** Traditional image stitching methods estimate warps from hand-crafted geometric features, whereas recent learning-based solutions leverage semantic features from neural networks instead. These two lines of research have largely diverged along separate evolution, with virtually no meaningful convergence to date. In this paper, we take a pioneering step to bridge this gap by unifying semantic and geometric features with UniStitch, a unified image stitching framework from multimodal features. To align discrete geometric features (i.e., keypoint) with continuous semantic feature maps, we present a Neural Point Transformer (NPT) module, which transforms unordered, sparse 1D geometric keypoints into ordered, dense 2D semantic maps. Then, to integrate the advantages of both representations, an Adaptive Mixture of Experts (AMoE) module is designed to fuse geometric and semantic representations. It dynamically shifts focus toward more reliable features during the fusion process, allowing the model to handle complex scenes, especially when either modality might be compromised. The fused representation can be adopted into common deep stitching pipelines, delivering significant performance gains over any single feature. Experiments show that UniStitch outperforms existing state-of-the-art methods with a large margin, paving the way for a unified paradigm between traditional and learning-based image stitching.
>
---
#### [new 067] UAV traffic scene understanding: A cross-spectral guided approach and a unified benchmark
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于交通场景理解任务，解决UAV在恶劣光照下性能下降及复杂行为评估不足的问题。提出CTCNet模型和Traffic-VQA基准，提升UAV交通认知能力。**

- **链接: [https://arxiv.org/pdf/2603.10722](https://arxiv.org/pdf/2603.10722)**

> **作者:** Yu Zhang; Zhicheng Zhao; Ze Luo; Chenglong Li; Jin Tang
>
> **摘要:** Traffic scene understanding from unmanned aerial vehicle (UAV) platforms is crucial for intelligent transportation systems due to its flexible deployment and wide-area monitoring capabilities. However, existing methods face significant challenges in real-world surveillance, as their heavy reliance on optical imagery leads to severe performance degradation under adverse illumination conditions like nighttime and fog. Furthermore, current Visual Question Answering (VQA) models are restricted to elementary perception tasks, lacking the domain-specific regulatory knowledge required to assess complex traffic behaviors. To address these limitations, we propose a novel Cross-spectral Traffic Cognition Network (CTCNet) for robust UAV traffic scene understanding. Specifically, we design a Prototype-Guided Knowledge Embedding (PGKE) module that leverages high-level semantic prototypes from an external Traffic Regulation Memory (TRM) to anchor domain-specific knowledge into visual representations, enabling the model to comprehend complex behaviors and distinguish fine-grained traffic violations. Moreover, we develop a Quality-Aware Spectral Compensation (QASC) module that exploits the complementary characteristics of optical and thermal modalities to perform bidirectional context exchange, effectively compensating for degraded features to ensure robust representation in complex environments. In addition, we construct Traffic-VQA, the first large-scale optical-thermal infrared benchmark for cognitive UAV traffic understanding, comprising 8,180 aligned image pairs and 1.3 million question-answer pairs across 31 diverse types. Extensive experiments demonstrate that CTCNet significantly outperforms state-of-the-art methods in both cognition and perception scenarios. The dataset is available at this https URL.
>
---
#### [new 068] HanMoVLM: Large Vision-Language Models for Professional Artistic Painting Evaluation
- **分类: cs.CV**

- **简介: 该论文提出HanMoVLM，解决艺术绘画专业评估问题，通过构建数据集和推理链，提升模型在中文绘画领域的评估能力。**

- **链接: [https://arxiv.org/pdf/2603.10814](https://arxiv.org/pdf/2603.10814)**

> **作者:** Hongji Yang; Yucheng Zhou; Wencheng Han; Songlian Li; Xiaotong Zhao; Jianbing Shen
>
> **备注:** 14 pages
>
> **摘要:** While Large Vision-Language Models (VLMs) demonstrate impressive general visual capabilities, they remain artistically blind and unable to offer professional evaluation of artworks within specific artistic domains like human experts. To bridge this gap, we transform VLMs into experts capable of professional-grade painting evaluation in the Chinese Artistic Domain, which is more abstract and demands extensive artistic training for evaluation. We introduce HanMo-Bench, a new dataset that features authentic auction-grade masterpieces and AI-generated works, grounded in real-world market valuations. To realize the rigorous judgment, we propose the HanMoVLM and construct a Chain-of-Thought (CoT) validated by experts. This CoT guides the model to perform expert-level reasoning: from content identification and Region of Interest (RoI) localization to professional evaluation, guided by both theme-specific evaluation and typical three-tier evaluation in Chinese paintings. Furthermore, we design a reward function to refine the reasoning process of the HanMoVLM to improve the accuracy. We demonstrate that HanMoVLM can serve as a critical backbone for Test-time Scaling in image generation. By acting as a high-quality verifier, HanMoVLM enables generative models to select the most artistically superior outputs from multiple candidates. Experimental results and human studies confirm that the proposed HanMoVLM effectively bridges the gap, achieving a high consistency with professional experts and significantly improving the quality of Chinese Painting generation.
>
---
#### [new 069] Event-based Photometric Stereo via Rotating Illumination and Per-Pixel Learning
- **分类: cs.CV**

- **简介: 该论文属于三维重建任务，解决传统光度立体方法在动态光照和高动态范围下的局限性。提出基于事件相机的光度立体系统，通过旋转光源和像素级神经网络直接预测表面法向量。**

- **链接: [https://arxiv.org/pdf/2603.10748](https://arxiv.org/pdf/2603.10748)**

> **作者:** Hyunwoo Kim; Won-Hoe Kim; Sanghoon Lee; Jianfei Cai; Giljoo Nam; Jae-Sang Hyun
>
> **摘要:** Photometric stereo is a technique for estimating surface normals using images captured under varying illumination. However, conventional frame-based photometric stereo methods are limited in real-world applications due to their reliance on controlled lighting, and susceptibility to ambient illumination. To address these limitations, we propose an event-based photometric stereo system that leverages an event camera, which is effective in scenarios with continuously varying scene radiance and high dynamic range conditions. Our setup employs a single light source moving along a predefined circular trajectory, eliminating the need for multiple synchronized light sources and enabling a more compact and scalable design. We further introduce a lightweight per-pixel multi-layer neural network that directly predicts surface normals from event signals generated by intensity changes as the light source rotates, without system calibration. Experimental results on benchmark datasets and real-world data collected with our data acquisition system demonstrate the effectiveness of our method, achieving a 7.12\% reduction in mean angular error compared to existing event-based photometric stereo methods. In addition, our method demonstrates robustness in regions with sparse event activity, strong ambient illumination, and scenes affected by specularities.
>
---
#### [new 070] Bridging the Skill Gap in Clinical CBCT Interpretation with CBCTRepD
- **分类: cs.CV**

- **简介: 该论文属于医学影像报告生成任务，旨在解决口腔颌面CBCT报告质量与效率问题。通过构建数据集并开发CBCTRepD系统，提升不同层级放射科医生的报告水平。**

- **链接: [https://arxiv.org/pdf/2603.10933](https://arxiv.org/pdf/2603.10933)**

> **作者:** Qinxin Wu; Fucheng Niu; Hengchuan Zhu; Yifan Sun; Ye Shen; Xu Li; Han Wu; Leqi Liu; Zhiwen Pan; Zuozhu Liu; Fudong Zhu; Bin Feng
>
> **摘要:** Generative AI has advanced rapidly in medical report generation; however, its application to oral and maxillofacial CBCT reporting remains limited, largely because of the scarcity of high-quality paired CBCT-report data and the intrinsic complexity of volumetric CBCT interpretation. To address this, we introduce CBCTRepD, a bilingual oral and maxillofacial CBCT report-generation system designed for integration into routine radiologist-AI co-authoring workflows. We curated a large-scale, high-quality paired CBCT-report dataset comprising approximately 7,408 studies, covering 55 oral disease entities across diverse acquisition settings, and used it to develop the system. We further established a clinically grounded, multi-level evaluation framework that assesses both direct AI-generated drafts and radiologist-edited collaboration reports using automatic metrics together with radiologist- and clinician-centered evaluation. Using this framework, we show that CBCTRepD achieves superior report-generation performance and produces drafts with writing quality and standardization comparable to those of intermediate radiologists. More importantly, in radiologist-AI collaboration, CBCTRepD provides consistent and clinically meaningful benefits across experience levels: it helps novice radiologists improve toward intermediate-level reporting, enables intermediate radiologists to approach senior-level performance, and even assists senior radiologists by reducing omission-related errors, including clinically important missed lesions. By improving report structure, reducing omissions, and promoting attention to co-existing lesions across anatomical regions, CBCTRepD shows strong and reliable potential as a practical assistant for real-world CBCT reporting across multi-level care settings.
>
---
#### [new 071] TractoRC: A Unified Probabilistic Learning Framework for Joint Tractography Registration and Clustering
- **分类: cs.CV**

- **简介: 该论文提出TractoRC，解决扩散MRI中轨迹配准与聚类的联合优化问题。通过统一的概率框架，提升两者性能。**

- **链接: [https://arxiv.org/pdf/2603.10418](https://arxiv.org/pdf/2603.10418)**

> **作者:** Yijie Li; Xi Zhu; Junyi Wang; Ye Wu; Lauren J. O'Donnell; Fan Zhang
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Diffusion MRI tractography enables in vivo reconstruction of white matter (WM) pathways. Two key tasks in tractography analysis include: 1) tractogram registration that aligns streamlines across individuals, and 2) streamline clustering that groups streamlines into compact fiber bundles. Although both tasks share the goal of capturing geometrically similar structures to characterize consistent WM organization, they are typically performed independently. In this work, we propose TractoRC, a unified probabilistic framework that jointly performs tractogram registration and streamline clustering within a single optimization scheme, enabling the two tasks to leverage complementary information. TractoRC learns a latent embedding space for streamline points, which serves as a shared representation for both tasks. Within this space, both tasks are formulated as probabilistic inference over structural representations: registration learns the distribution of anatomical landmarks as probabilistic keypoints to align tractograms across subjects, and clustering learns streamline structural prototypes that capture geometric similarity to form coherent streamline clusters. To support effective learning of this shared space, we introduce a transformation-equivariant self-supervised strategy to learn geometry-aware and transformation-invariant embeddings. Experiments demonstrate that jointly optimizing registration and clustering significantly improves performance in both tasks over state-of-the-art methods that treat them independently. Code will be made publicly available at this https URL .
>
---
#### [new 072] GroundCount: Grounding Vision-Language Models with Object Detection for Mitigating Counting Hallucinations
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型任务，旨在解决计数幻觉问题。通过引入目标检测模型进行空间定位，提升模型计数准确性。**

- **链接: [https://arxiv.org/pdf/2603.10978](https://arxiv.org/pdf/2603.10978)**

> **作者:** Boyuan Chen; Minghao Shao; Siddharth Garg; Ramesh Karri; Muhammad Shafique
>
> **摘要:** Vision Language Models (VLMs) exhibit persistent hallucinations in counting tasks, with accuracy substantially lower than other visual reasoning tasks (excluding sentiment). This phenomenon persists even in state-of-the-art reasoning-capable VLMs. Conversely, CNN-based object detection models (ODMs) such as YOLO excel at spatial localization and instance counting with minimal computational overhead. We propose GroundCount, a framework that augments VLMs with explicit spatial grounding from ODMs to mitigate counting hallucinations. In the best case, our prompt-based augmentation strategy achieves 81.3% counting accuracy on the best-performing model (Ovis2.5-2B) - a 6.6pp improvement - while reducing inference time by 22% through elimination of hallucination-driven reasoning loops for stronger models. We conduct comprehensive ablation studies demonstrating that positional encoding is a critical component, being beneficial for stronger models but detrimental for weaker ones. Confidence scores, by contrast, introduce noise for most architectures and their removal improves performance in four of five evaluated models. We further evaluate feature-level fusion architectures, finding that explicit symbolic grounding via structured prompts outperforms implicit feature fusion despite sophisticated cross-attention mechanisms. Our approach yields consistent improvements across four of five evaluated VLM architectures (6.2--7.5pp), with one architecture exhibiting degraded performance due to incompatibility between its iterative reflection mechanisms and structured prompts. These results suggest that counting failures stem from fundamental spatial-semantic integration limitations rather than architecture-specific deficiencies, while highlighting the importance of architectural compatibility in augmentation strategies.
>
---
#### [new 073] RandMark: On Random Watermarking of Visual Foundation Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于模型版权保护任务，旨在解决视觉基础模型的归属验证问题。通过随机水印嵌入方法，在模型内部表示中嵌入数字水印，实现有效所有权验证。**

- **链接: [https://arxiv.org/pdf/2603.10695](https://arxiv.org/pdf/2603.10695)**

> **作者:** Anna Chistyakova; Mikhail Pautov
>
> **摘要:** Being trained on large and diverse datasets, visual foundation models (VFMs) can be fine-tuned to achieve remarkable performance and efficiency in various downstream computer vision tasks. The high computational cost of data collection and training makes these models valuable assets, which motivates some VFM owners to distribute them alongside a license to protect their intellectual property rights. In this paper, we propose an approach to ownership verification of visual foundation models that leverages a small encoder-decoder network to embed digital watermarks into an internal representation of a hold-out set of input images. The method is based on random watermark embedding, which makes the watermark statistics detectable in functional copies of the watermarked model. Both theoretically and experimentally, we demonstrate that the proposed method yields a low probability of false detection for non-watermarked models and a low probability of false misdetection for watermarked models.
>
---
#### [new 074] A$^2$-Edit: Precise Reference-Guided Image Editing of Arbitrary Objects and Ambiguous Masks
- **分类: cs.CV**

- **简介: 该论文提出A²-Edit，解决任意物体编辑任务中的遮罩模糊和类别覆盖不足问题，构建了UniEdit-500K数据集，并引入Transformer混合模块和掩码渐进训练策略，提升编辑效果与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.10685](https://arxiv.org/pdf/2603.10685)**

> **作者:** Huayu Zheng; Guangzhao Li; Baixuan Zhao; Siqi Luo; Hantao Jiang; Guangtao Zhai; Xiaohong Liu
>
> **摘要:** We propose \textbf{A$^2$-Edit}, a unified inpainting framework for arbitrary object categories, which allows users to replace any target region with a reference object using only a coarse mask. To address the issues of severe homogenization and limited category coverage in existing datasets, we construct a large-scale, multi-category dataset \textbf{UniEdit-500K}, which includes 8 major categories, 209 fine-grained subcategories, and a total of 500,104 image pairs. Such rich category diversity poses new challenges for the model, requiring it to automatically learn semantic relationships and distinctions across categories. To this end, we introduce the \textbf{Mixture of Transformer} module, which performs differentiated modeling of various object categories through dynamic expert selection, and further enhances cross-category semantic transfer and generalization through collaboration among experts. In addition, we propose a \textbf{Mask Annealing Training Strategy} (MATS) that progressively relaxes mask precision during training, reducing the model's reliance on accurate masks and improving robustness across diverse editing tasks. Extensive experiments on benchmarks such as VITON-HD and AnyInsertion demonstrate that A$^2$-Edit consistently outperforms existing approaches across all metrics, providing a new and efficient solution for arbitrary object editing.
>
---
#### [new 075] A Robust Deep Learning Framework for Bangla License Plate Recognition Using YOLO and Vision-Language OCR
- **分类: cs.CV**

- **简介: 该论文属于车牌识别任务，解决Bangla车牌检测与识别问题。通过YOLO和Vision-Language OCR方法，提升识别准确率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.10267](https://arxiv.org/pdf/2603.10267)**

> **作者:** Nayeb Hasin; Md. Arafath Rahman Nishat; Mainul Islam; Khandakar Shakib Al Hasan; Asif Newaz
>
> **备注:** Accepted at the 2026 IEEE International Conference on AI and Data Analytics (ICAD 2026). Final version will appear in IEEE Xplore
>
> **摘要:** An Automatic License Plate Recognition (ALPR) system constitutes a crucial element in an intelligent traffic management system. However, the detection of Bangla license plates remains challenging because of the complicated character scheme and uneven layouts. This paper presents a robust Bangla License Plate Recognition system that integrates a deep learning-based object detection model for license plate localization with Optical Character Recognition for text extraction. Multiple object detection architectures, including U-Net and several YOLO (You Only Look Once) variants, are compared for license plate localization. This study proposes a novel two-stage adaptive training strategy built upon the YOLOv8 architecture to improve localization performance. The proposed approach outperforms the established models, achieving an accuracy of 97.83% and an Intersection over Union (IoU) of 91.3%. The text recognition problem is phrased as a sequence generation problem with a VisionEncoderDecoder architecture, with a combination of encoder-decoders evaluated. It was demonstrated that the ViT + BanglaBERT model gives better results at the character level, with a Character Error Rate of 0.1323 and Word Error Rate of 0.1068. The proposed system also shows a consistent performance when tested on an external dataset that has been curated for this study purpose. The dataset offers completely different environment and lighting conditions compared to the training sample, indicating the robustness of the proposed framework. Overall, our proposed system provides a robust and reliable solution for Bangla license plate recognition and performs effectively across diverse real-world scenarios, including variations in lighting, noise, and plate styles. These strengths make it well suited for deployment in intelligent transportation applications such as automated law enforcement and access control.
>
---
#### [new 076] UniPINN: A Unified PINN Framework for Multi-task Learning of Diverse Navier-Stokes Equations
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出UniPINN，解决多流Navier-Stokes方程的联合学习问题，通过共享-专用架构、跨流注意力和动态权重分配提升精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.10466](https://arxiv.org/pdf/2603.10466)**

> **作者:** Dengdi Sun; Jie Chen; Xiao Wang; Jin Tang
>
> **摘要:** Physics-Informed Neural Networks (PINNs) have shown promise in solving incompressible Navier-Stokes equations, yet existing approaches are predominantly designed for single-flow settings. When extended to multi-flow scenarios, these methods face three key challenges: (1) difficulty in simultaneously capturing both shared physical principles and flow-specific characteristics, (2) susceptibility to inter-task negative transfer that degrades prediction accuracy, and (3) unstable training dynamics caused by disparate loss magnitudes across heterogeneous flow regimes. To address these limitations, we propose UniPINN, a unified multi-flow PINN framework that integrates three complementary components: a shared-specialized architecture that disentangles universal physical laws from flow-specific features, a cross-flow attention mechanism that selectively reinforces relevant patterns while suppressing task-irrelevant interference, and a dynamic weight allocation strategy that adaptively balances loss contributions to stabilize multi-objective optimization. Extensive experiments on three canonical flows demonstrate that UniPINN effectively unifies multi-flow learning, achieving superior prediction accuracy and balanced performance across heterogeneous regimes while successfully mitigating negative transfer. The source code of this paper will be released on this https URL
>
---
#### [new 077] StructDamage:A Large Scale Unified Crack and Surface Defect Dataset for Robust Structural Damage Detection
- **分类: cs.CV**

- **简介: 该论文提出StructDamage数据集，解决结构裂缝和表面缺陷检测问题。通过整合多源数据，构建大规模、多样化数据集，支持深度学习模型训练与评估。**

- **链接: [https://arxiv.org/pdf/2603.10484](https://arxiv.org/pdf/2603.10484)**

> **作者:** Misbah Ijaz; Saif Ur Rehman Khan; Abd Ur Rehman; Sebastian Vollmer; Andreas Dengel; Muhammad Nabeel Asim
>
> **摘要:** Automated detection and classification of structural cracks and surface defects is a critical challenge in civil engineering, infrastructure maintenance, and heritage preservation. Recent advances in Computer Vision (CV) and Deep Learning (DL) have significantly improved automatic crack detection. However, these methods rely heavily on large, diverse, and carefully curated datasets that include various crack types across different surface materials. Many existing public crack datasets lack geographic diversity, surface types, scale, and labeling consistency, making it challenging for trained algorithms to generalize effectively in real world conditions. We provide a novel dataset, StructDamage, a curated collection of approximately 78,093 images spanning nine surface types: walls, tile, stone, road, pavement, deck, concrete, and brick. The dataset was constructed by systematically aggregating, harmonizing, and reannotating images from 32 publicly available datasets covering concrete structures, asphalt pavements, masonry walls, bridges, and historic buildings. All images are organized in a folder level classification hierarchy suitable for training Convolutional Neural Networks (CNNs) and Vision Transformers. To highlight the practical value of the dataset, we present baseline classification results using fifteen DL architectures from six model families, with twelve achieving macro F1-scores over 0.96. The best performing model DenseNet201 achieves 98.62% accuracy. The proposed dataset provides a comprehensive and versatile resource suitable for classification tasks. With thorough documentation and a standard structure, it is designed to promote reproducible research and support the development and fair evaluation of robust crack damage detection approaches.
>
---
#### [new 078] GeoSense: Internalizing Geometric Necessity Perception for Multimodal Reasoning
- **分类: cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决MLLMs空间理解不足的问题。通过引入几何感知机制，使模型能自主判断是否需要几何信息，提升空间推理能力。**

- **链接: [https://arxiv.org/pdf/2603.10370](https://arxiv.org/pdf/2603.10370)**

> **作者:** Ruiheng Liu; Haihong Hao; Mingfei Han; Xin Gu; Kecheng Zhang; Changlin Li; Xiaojun Chang
>
> **摘要:** Advancing towards artificial superintelligence requires rich and intelligent perceptual capabilities. A critical frontier in this pursuit is overcoming the limited spatial understanding of Multimodal Large Language Models (MLLMs), where geometry information is essential. Existing methods often address this by rigidly injecting geometric signals into every input, while ignoring their necessity and adding computation overhead. Contrary to this paradigm, our framework endows the model with an awareness of perceptual insufficiency, empowering it to autonomously engage geometric features in reasoning when 2D cues are deemed insufficient. To achieve this, we first introduce an independent geometry input channel to the model architecture and conduct alignment training, enabling the effective utilization of geometric features. Subsequently, to endow the model with perceptual awareness, we curate a dedicated spatial-aware supervised fine-tuning dataset. This serves to activate the model's latent internal cues, empowering it to autonomously determine the necessity of geometric information. Experiments across multiple spatial reasoning benchmarks validate this approach, demonstrating significant spatial gains without compromising 2D visual reasoning capabilities, offering a path toward more robust, efficient and self-aware multi-modal intelligence.
>
---
#### [new 079] EmoStory: Emotion-Aware Story Generation
- **分类: cs.CV**

- **简介: 该论文提出EmoStory，属于情感感知的故事生成任务，旨在解决现有方法忽视情感影响的问题，通过两阶段框架生成具有明确情感方向的连贯视觉故事。**

- **链接: [https://arxiv.org/pdf/2603.10349](https://arxiv.org/pdf/2603.10349)**

> **作者:** Jingyuan Yang; Rucong Chen; Hui Huang
>
> **摘要:** Story generation aims to produce image sequences that depict coherent narratives while maintaining subject consistency across frames. Although existing methods have excelled in producing coherent and expressive stories, they remain largely emotion-neutral, focusing on what subject appears in a story while overlooking how emotions shape narrative interpretation and visual presentation. As stories are intended to engage audiences emotionally, we introduce emotion-aware story generation, a new task that aims to generate subject-consistent visual stories with explicit emotional directions. This task is challenging due to the abstract nature of emotions, which must be grounded in concrete visual elements and consistently expressed across a narrative through visual composition. To address these challenges, we propose EmoStory, a two-stage framework that integrates agent-based story planning and region-aware story generation. The planning stage transforms target emotions into coherent story prompts with emotion agent and writer agent, while the generation stage preserves subject consistency and injects emotion-related elements through region-aware composition. We evaluate EmoStory on a newly constructed dataset covering 25 subjects and 600 emotional stories. Extensive quantitative and qualitative results, along with user studies, show that EmoStory outperforms state-of-the-art story generation methods in emotion accuracy, prompt alignment, and subject consistency.
>
---
#### [new 080] Attribution as Retrieval: Model-Agnostic AI-Generated Image Attribution
- **分类: cs.CV**

- **简介: 该论文属于AI生成图像溯源任务，旨在解决模型依赖和泛化性差的问题。提出LIDA框架，通过实例检索实现高效溯源。**

- **链接: [https://arxiv.org/pdf/2603.10583](https://arxiv.org/pdf/2603.10583)**

> **作者:** Hongsong Wang; Renxi Cheng; Chaolei Han; Jie Gui
>
> **备注:** To appear in CVPR 2026, Code is at this https URL
>
> **摘要:** With the rapid advancement of AIGC technologies, image forensics will encounter unprecedented challenges. Traditional methods are incapable of dealing with increasingly realistic images generated by rapidly evolving image generation techniques. To facilitate the identification of AI-generated images and the attribution of their source models, generative image watermarking and AI-generated image attribution have emerged as key research focuses in recent years. However, existing methods are model-dependent, requiring access to the generative models and lacking generality and scalability to new and unseen generators. To address these limitations, this work presents a new paradigm for AI-generated image attribution by formulating it as an instance retrieval problem instead of a conventional image classification problem. We propose an efficient model-agnostic framework, called Low-bIt-plane-based Deepfake Attribution (LIDA). The input to LIDA is produced by Low-Bit Fingerprint Generation module, while the training involves Unsupervised Pre-Training followed by subsequent Few-Shot Attribution Adaptation. Comprehensive experiments demonstrate that LIDA achieves state-of-the-art performance for both Deepfake detection and image attribution under zero- and few-shot settings. The code is at this https URL
>
---
#### [new 081] DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime
- **分类: cs.CV**

- **简介: 该论文提出DSFlash，用于实时全景场景图生成任务，解决资源受限设备上的速度与效率问题，通过低延迟和轻量级设计实现高性能推理。**

- **链接: [https://arxiv.org/pdf/2603.10538](https://arxiv.org/pdf/2603.10538)**

> **作者:** Julian Lorenz; Vladyslav Kovganko; Elias Kohout; Mrunmai Phatak; Daniel Kienzle; Rainer Lienhart
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Scene Graph Generation (SGG) aims to extract a detailed graph structure from an image, a representation that holds significant promise as a robust intermediate step for complex downstream tasks like reasoning for embodied agents. However, practical deployment in real-world applications - especially on resource constrained edge devices - requires speed and resource efficiency, challenges that have received limited attention in existing research. To bridge this gap, we introduce DSFlash, a low-latency model for panoptic scene graph generation designed to overcome these limitations. DSFlash can process a video stream at 56 frames per second on a standard RTX 3090 GPU, without compromising performance against existing state-of-the-art methods. Crucially, unlike prior approaches that often restrict themselves to salient relationships, DSFlash computes comprehensive scene graphs, offering richer contextual information while maintaining its superior latency. Furthermore, DSFlash is light on resources, requiring less than 24 hours to train on a single, nine-year-old GTX 1080 GPU. This accessibility makes DSFlash particularly well-suited for researchers and practitioners operating with limited computational resources, empowering them to adapt and fine-tune SGG models for specialized applications.
>
---
#### [new 082] Sparse Task Vector Mixup with Hypernetworks for Efficient Knowledge Transfer in Whole-Slide Image Prognosis
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决癌症 prognosis 预测中因样本稀缺导致的模型泛化能力差问题。提出 STEPH 方法，通过稀疏任务向量混合与超网络实现高效知识迁移。**

- **链接: [https://arxiv.org/pdf/2603.10526](https://arxiv.org/pdf/2603.10526)**

> **作者:** Pei Liu; Xiangxiang Zeng; Tengfei Ma; Yucheng Xing; Xuanbai Ren; Yiping Liu
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Whole-Slide Images (WSIs) are widely used for estimating the prognosis of cancer patients. Current studies generally follow a cancer-specific learning paradigm. However, the available training samples for one cancer type are usually scarce in pathology. Consequently, the model often struggles to learn generalizable knowledge, thus performing worse on the tumor samples with inherent high heterogeneity. Although multi-cancer joint learning and knowledge transfer approaches have been explored recently to address it, they either rely on large-scale joint training or extensive inference across multiple models, posing new challenges in computational efficiency. To this end, this paper proposes a new scheme, Sparse Task Vector Mixup with Hypernetworks (STEPH). Unlike previous ones, it efficiently absorbs generalizable knowledge from other cancers for the target via model merging: i) applying task vector mixup to each source-target pair and then ii) sparsely aggregating task vector mixtures to obtain an improved target model, driven by hypernetworks. Extensive experiments on 13 cancer datasets show that STEPH improves over cancer-specific learning and an existing knowledge transfer baseline by 5.14% and 2.01%, respectively. Moreover, it is a more efficient solution for learning prognostic knowledge from other cancers, without requiring large-scale joint training or extensive multi-model inference. Code is publicly available at this https URL.
>
---
#### [new 083] Delta-K: Boosting Multi-Instance Generation via Cross-Attention Augmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成任务，解决多实例场景中概念遗漏问题。提出Delta-K框架，通过跨注意力键空间增强语义表示，提升合成一致性。**

- **链接: [https://arxiv.org/pdf/2603.10210](https://arxiv.org/pdf/2603.10210)**

> **作者:** Zitong Wang; Zijun Shen; Haohao Xu; Zhengjie Luo; Weibin Wu
>
> **摘要:** While Diffusion Models excel in text-to-image synthesis, they often suffer from concept omission when synthesizing complex multi-instance scenes. Existing training-free methods attempt to resolve this by rescaling attention maps, which merely exacerbates unstructured noise without establishing coherent semantic representations. To address this, we propose Delta-K, a backbone-agnostic and plug-and-play inference framework that tackles omission by operating directly in the shared cross-attention Key space. Specifically, with Vision-language model, we extract a differential key $\Delta K$ that encodes the semantic signature of missing concepts. This signal is then injected during the early semantic planning stage of the diffusion process. Governed by a dynamically optimized scheduling mechanism, Delta-K grounds diffuse noise into stable structural anchors while preserving existing concepts. Extensive experiments demonstrate the generality of our approach: Delta-K consistently improves compositional alignment across both modern DiT models and classical U-Net architectures, without requiring spatial masks, additional training, or architectural modifications.
>
---
#### [new 084] Too Vivid to Be Real? Benchmarking and Calibrating Generative Color Fidelity
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决生成图像过于鲜艳不真实的问题。提出CFD数据集和CFM指标评估颜色真实性，并引入CFR方法提升生成质量。**

- **链接: [https://arxiv.org/pdf/2603.10990](https://arxiv.org/pdf/2603.10990)**

> **作者:** Zhengyao Fang; Zexi Jia; Yijia Zhong; Pengcheng Luo; Jinchao Zhang; Guangming Lu; Jun Yu; Wenjie Pei
>
> **备注:** accepted by CVPR2026
>
> **摘要:** Recent advances in text-to-image (T2I) generation have greatly improved visual quality, yet producing images that appear visually authentic to real-world photography remains challenging. This is partly due to biases in existing evaluation paradigms: human ratings and preference-trained metrics often favor visually vivid images with exaggerated saturation and contrast, which make generations often too vivid to be real even when prompted for realistic-style images. To address this issue, we present Color Fidelity Dataset (CFD) and Color Fidelity Metric (CFM) for objective evaluation of color fidelity in realistic-style generations. CFD contains over 1.3M real and synthetic images with ordered levels of color realism, while CFM employs a multimodal encoder to learn perceptual color fidelity. In addition, we propose a training-free Color Fidelity Refinement (CFR) that adaptively modulates spatial-temporal guidance scale in generation, thereby enhancing color authenticity. Together, CFD supports CFM for assessment, whose learned attention further guides CFR to refine T2I fidelity, forming a progressive framework for assessing and improving color fidelity in realistic-style T2I generation. The dataset and code are available at this https URL.
>
---
#### [new 085] Robotic Ultrasound Makes CBCT Alive
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于医学影像任务，旨在解决CBCT静态图像无法实时反映软组织变形的问题。通过结合机器人超声与深度学习，实现动态更新CBCT图像。**

- **链接: [https://arxiv.org/pdf/2603.10220](https://arxiv.org/pdf/2603.10220)**

> **作者:** Feng Li; Ziyuan Li; Zhongliang Jiang; Nassir Navab; Yuan Bi
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Intraoperative Cone Beam Computed Tomography (CBCT) provides a reliable 3D anatomical context essential for interventional planning. However, its static nature fails to provide continuous monitoring of soft-tissue deformations induced by respiration, probe pressure, and surgical manipulation, leading to navigation discrepancies. We propose a deformation-aware CBCT updating framework that leverages robotic ultrasound as a dynamic proxy to infer tissue motion and update static CBCT slices in real time. Starting from calibration-initialized alignment with linear correlation of linear combination (LC2)-based rigid refinement, our method establishes accurate multimodal correspondence. To capture intraoperative dynamics, we introduce the ultrasound correlation UNet (USCorUNet), a lightweight network trained with optical flow-guided supervision to learn deformation-aware correlation representations, enabling accurate, real-time dense deformation field estimation from ultrasound streams. The inferred deformation is spatially regularized and transferred to the CBCT reference to produce deformation-consistent visualizations without repeated radiation exposure. We validate the proposed approach through deformation estimation and ultrasound-guided CBCT updating experiments. Results demonstrate real-time end-to-end CBCT slice updating and physically plausible deformation estimation, enabling dynamic refinement of static CBCT guidance during robotic ultrasound-assisted interventions. The source code is publicly available at this https URL.
>
---
#### [new 086] Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于连续模仿学习任务，解决多任务下策略持续优化问题。通过多模态潜在空间存储信息，并引入增量调整机制，提升学习稳定性与任务区分性。**

- **链接: [https://arxiv.org/pdf/2603.10929](https://arxiv.org/pdf/2603.10929)**

> **作者:** Fanqi Yu; Matteo Tiezzi; Tommaso Apicella; Cigdem Beyan; Vittorio Murino
>
> **摘要:** We introduce a lifelong imitation learning framework that enables continual policy refinement across sequential tasks under realistic memory and data constraints. Our approach departs from conventional experience replay by operating entirely in a multimodal latent space, where compact representations of visual, linguistic, and robot's state information are stored and reused to support future learning. To further stabilize adaptation, we introduce an incremental feature adjustment mechanism that regularizes the evolution of task embeddings through an angular margin constraint, preserving inter-task distinctiveness. Our method establishes a new state of the art in the LIBERO benchmarks, achieving 10-17 point gains in AUC and up to 65% less forgetting compared to previous leading methods. Ablation studies confirm the effectiveness of each component, showing consistent gains over alternative strategies. The code is available at: this https URL.
>
---
#### [new 087] Med-DualLoRA: Local Adaptation of Foundation Models for 3D Cardiac MRI
- **分类: cs.CV**

- **简介: 该论文属于医学影像任务，解决多中心3D心脏MRI疾病检测中模型适应问题。提出Med-DualLoRA框架，实现高效联邦微调，提升性能并降低通信成本。**

- **链接: [https://arxiv.org/pdf/2603.10967](https://arxiv.org/pdf/2603.10967)**

> **作者:** Joan Perramon-Llussà; Amelia Jiménez-Sánchez; Grzegorz Skorupko; Fotis Avgoustidis; Carlos Martín-Isla; Karim Lekadir; Polyxeni Gkontra
>
> **备注:** 11 pages, 2 figures. Submitted to MICCAI 2026
>
> **摘要:** Foundation models (FMs) show great promise for robust downstream performance across medical imaging tasks and modalities, including cardiac magnetic resonance (CMR), following task-specific adaptation. However, adaptation using single-site data may lead to suboptimal performance and increased model bias, while centralized fine-tuning on clinical data is often infeasible due to privacy constraints. Federated fine-tuning offers a privacy-preserving alternative; yet conventional approaches struggle under heterogeneous, non-IID multi-center data and incur substantial communication overhead when adapting large models. In this work, we study federated FM fine-tuning for 3D CMR disease detection and propose Med-DualLoRA, a client-aware parameter-efficient fine-tuning (PEFT) federated framework that disentangles globally shared and local low-rank adaptations (LoRA) through additive decomposition. Global and local LoRA modules are trained locally, but only the global component is shared and aggregated across sites, keeping local adapters private. This design improves personalization while significantly reducing communication cost, and experiments show that adapting only two transformer blocks preserves performance while further improving efficiency. We evaluate our method on a multi-center state-of-the-art cine 3D CMR FM fine-tuned for disease detection using ACDC and combined M\&Ms datasets, treating each vendor as a federated client. Med-DualLoRA achieves statistically significant improved performance (balanced accuracy 0.768, specificity 0.612) compared to other federated PEFT baselines, while maintaining communication efficiency. Our approach provides a scalable solution for local federated adaptation of medical FMs under realistic clinical constraints.
>
---
#### [new 088] Learning to Wander: Improving the Global Image Geolocation Ability of LMMs via Actionable Reasoning
- **分类: cs.CV**

- **简介: 该论文聚焦于图像全球定位任务，解决LMMs在该任务上的表现不足。提出WanderBench基准和GeoAoT框架，通过行动推理提升定位精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.10463](https://arxiv.org/pdf/2603.10463)**

> **作者:** Yushuo Zheng; Huiyu Duan; Zicheng Zhang; Xiaohong Liu; Xiongkuo Min
>
> **摘要:** Geolocation, the task of identifying the geographic location of an image, requires abundant world knowledge and complex reasoning abilities. Though advanced large multimodal models (LMMs) have shown superior aforementioned capabilities, their performance on the geolocation task remains unexplored. To this end, we introduce \textbf{WanderBench}, the first open access global geolocation benchmark designed for actionable geolocation reasoning in embodied scenarios. WanderBench contains over 32K panoramas across six continents, organized as navigable graphs that enable physical actions such as rotation and movement, transforming geolocation from static recognition into interactive exploration. Building on this foundation, we propose \textbf{GeoAoT} (Action of Thought), a \underline{Geo}location framework with \underline{A}ction of \underline{T}hough, which couples reasoning with embodied actions. Instead of generating textual reasoning chains, GeoAoT produces actionable plans such as, approaching landmarks or adjusting viewpoints, to actively reduce uncertainty. We further establish an evaluation protocol that jointly measures geolocation accuracy and difficulty-aware geolocation questioning ability. Experiments on 19 large multimodal models show that GeoAoT achieves superior fine-grained localization and stronger generalization in dynamic environments. WanderBench and GeoAoT define a new paradigm for actionable, reasoning driven geolocation in embodied visual understanding.
>
---
#### [new 089] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video
- **分类: cs.CV**

- **简介: 该论文属于4D动物重建任务，解决单目视频中马匹运动与外观分离重建问题。提出4DEquine框架，分步处理运动与外观重建，提升效率与精度。**

- **链接: [https://arxiv.org/pdf/2603.10125](https://arxiv.org/pdf/2603.10125)**

> **作者:** Jin Lyu; Liang An; Pujin Cheng; Yebin Liu; Xiaoying Tang
>
> **备注:** Accepted to CVPR2026
>
> **摘要:** 4D reconstruction of equine family (e.g. horses) from monocular video is important for animal welfare. Previous mainstream 4D animal reconstruction methods require joint optimization of motion and appearance over a whole video, which is time-consuming and sensitive to incomplete observation. In this work, we propose a novel framework called 4DEquine by disentangling the 4D reconstruction problem into two sub-problems: dynamic motion reconstruction and static appearance reconstruction. For motion, we introduce a simple yet effective spatio-temporal transformer with a post-optimization stage to regress smooth and pixel-aligned pose and shape sequences from video. For appearance, we design a novel feed-forward network that reconstructs a high-fidelity, animatable 3D Gaussian avatar from as few as a single image. To assist training, we create a large-scale synthetic motion dataset, VarenPoser, which features high-quality surface motions and diverse camera trajectories, as well as a synthetic appearance dataset, VarenTex, comprising realistic multi-view images generated through multi-view diffusion. While training only on synthetic datasets, 4DEquine achieves state-of-the-art performance on real-world APT36K and AiM datasets, demonstrating the superiority of 4DEquine and our new datasets for both geometry and appearance reconstruction. Comprehensive ablation studies validate the effectiveness of both the motion and appearance reconstruction network. Project page: this https URL.
>
---
#### [new 090] Video-Based Reward Modeling for Computer-Use Agents
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于计算机使用代理的评估任务，解决如何准确判断代理行为是否符合用户指令的问题。通过视频执行建模和奖励预测，提出ExeVRM模型，提升评估准确性与时间定位精度。**

- **链接: [https://arxiv.org/pdf/2603.10178](https://arxiv.org/pdf/2603.10178)**

> **作者:** Linxin Song; Jieyu Zhang; Huanxin Sheng; Taiwei Shi; Gupta Rahul; Yang Liu; Ranjay Krishna; Jian Kang; Jieyu Zhao
>
> **摘要:** Computer-using agents (CUAs) are becoming increasingly capable; however, it remains difficult to scale evaluation of whether a trajectory truly fulfills a user instruction. In this work, we study reward modeling from execution video: a sequence of keyframes from an agent trajectory that is independent of the agent's internal reasoning or actions. Although video-execution modeling is method-agnostic, it presents key challenges, including highly redundant layouts and subtle, localized cues that determine success. We introduce Execution Video Reward 53k (ExeVR-53k), a dataset of 53k high-quality video--task--reward triplets. We further propose adversarial instruction translation to synthesize negative samples with step-level annotations. To enable learning from long, high-resolution execution videos, we design spatiotemporal token pruning, which removes homogeneous regions and persistent tokens while preserving decisive UI changes. Building on these components, we fine-tune an Execution Video Reward Model (ExeVRM) that takes only a user instruction and a video-execution sequence to predict task success. Our ExeVRM 8B achieves 84.7% accuracy and 87.7% recall on video-execution assessment, outperforming strong proprietary models such as GPT-5.2 and Gemini-3 Pro across Ubuntu, macOS, Windows, and Android, while providing more precise temporal attribution. These results show that video-execution reward modeling can serve as a scalable, model-agnostic evaluator for CUAs.
>
---
#### [new 091] Phase-Interface Instance Segmentation as a Visual Sensor for Laboratory Process Monitoring
- **分类: cs.CV**

- **简介: 该论文属于相界面实例分割任务，旨在解决透明玻璃器皿中相界分割困难的问题。提出LGA-RCM-YOLO模型，并构建CTG 2.0数据集，实现高效准确的实验室过程监控。**

- **链接: [https://arxiv.org/pdf/2603.10782](https://arxiv.org/pdf/2603.10782)**

> **作者:** Mingyue Li; Xin Yang; Shilin Yan; Jinye Ran; Morui Zhu; Zirui Peng; Huanqing Peng; Wei Peng; Guanghua Zhang; Shuo Li; Hao Zhang
>
> **摘要:** Reliable visual monitoring of chemical experiments remains challenging in transparent glassware, where weak phase boundaries and optical artifacts degrade conventional segmentation. We formulate laboratory phenomena as the time evolution of phase interfaces and introduce the Chemical Transparent Glasses dataset 2.0 (CTG 2.0), a vessel-aware benchmark with 3,668 images, 23 glassware categories, and five multiphase interface types for phase-interface instance segmentation. Building on YOLO11m-seg, we propose LGA-RCM-YOLO, which combines Local-Global Attention (LGA) for robust semantic representation and a Rectangular Self-Calibration Module (RCM) for boundary refinement of thin, elongated interfaces. On CTG 2.0, the proposed model achieves 84.4% AP@0.5 and 58.43% AP@0.5-0.95, improving over the YOLO11m baseline by 6.42 and 8.75 AP points, respectively, while maintaining near real-time inference (13.67 FPS, RTX 3060). An auxiliary color-attribute head further labels liquid instances as colored or colorless with 98.71% precision and 98.32% recall. Finally, we demonstrate continuous process monitoring in separatory-funnel phase separation and crystallization, showing that phase-interface instance segmentation can serve as a practical visual sensor for laboratory automation.
>
---
#### [new 092] Agentar-Fin-OCR
- **分类: cs.CV**

- **简介: 该论文提出Agentar-Fin-OCR，解决金融文档解析问题，通过算法和模块设计提升表格解析准确性和结构一致性。**

- **链接: [https://arxiv.org/pdf/2603.11044](https://arxiv.org/pdf/2603.11044)**

> **作者:** Siyi Qian; Xiongfei Bai; Bingtao Fu; Yichen Lu; Gaoyang Zhang; Xudong Yang; Peng Zhang
>
> **摘要:** In this paper, we propose Agentar-Fin-OCR, a document parsing system tailored to financial-domain documents, transforming ultra-long financial PDFs into semantically consistent, highly accurate, structured outputs with auditing-grade provenance. To address finance-specific challenges such as complex layouts, cross-page structural discontinuities, and cell-level referencing capability, Agentar-Fin-OCR combines (1) a Cross-page Contents Consolidation algorithm to restore continuity across pages and a Document-level Heading Hierarchy Reconstruction (DHR) module to build a globally consistent Table of Contents (TOC) tree for structure-aware retrieval, and (2) a difficulty-adaptive curriculum learning training strategy for table parsing, together with a CellBBoxRegressor module that uses structural anchor tokens to localize table cells from decoder hidden states without external detectors. Experiments demonstrate that our model shows high performance on the table parsing metrics of OmniDocBench. To enable realistic evaluation in the financial vertical, we further introduce FinDocBench, a benchmark that includes six financial document categories with expert-verified annotations and evaluation metrics including Table of Contents edit-distance-based similarity (TocEDS), cross-page concatenated TEDS, and Table Cell Intersection over Union (C-IoU). We evaluate a wide range of state-of-the-art models on FinDocBench to assess their capabilities and remaining limitations on financial documents. Overall, Agentar-Fin-OCR and FinDocBench provide a practical foundation for reliable downstream financial document applications.
>
---
#### [new 093] Human Presence Detection via Wi-Fi Range-Filtered Doppler Spectrum on Commodity Laptops
- **分类: eess.SP; cs.AI; cs.CV**

- **简介: 该论文属于人体存在检测任务，解决传统方法依赖外部设备或隐私问题。提出一种基于Wi-Fi的低复杂度检测方法，利用内置硬件实现无外设、无隐私风险的用户位置检测。**

- **链接: [https://arxiv.org/pdf/2603.10845](https://arxiv.org/pdf/2603.10845)**

> **作者:** Jessica Sanson; Rahul C. Shah; Valerio Frascolla
>
> **备注:** 6 pages, Conference
>
> **摘要:** Human Presence Detection (HPD) is key to enable intelligent power management and security features in everyday devices. In this paper we propose the first HPD solution that leverages monostatic Wi-Fi sensing and detects user position using only the built-in Wi-Fi hardware of a device, with no need for external devices, access points, or additional sensors. In contrast, existing HPD solutions for laptops require external dedicated sensors which add cost and complexity, or rely on camera-based approaches that introduce significant privacy concerns. We herewith introduce the Range-Filtered Doppler Spectrum (RF-DS), a novel Wi-Fi sensing technique for presence estimation that enables both range-selective and temporally windowed detection of user presence. By applying targeted range-area filtering in the Channel Impulse Response (CIR) domain before Doppler analysis, our method focuses processing on task-relevant spatial zones, significantly reducing computational complexity. In addition, the use of temporal windows in the spectrum domain provides greater estimator stability compared to conventional 2D Range-Doppler detectors. Furthermore, we propose an adaptive multi-rate processing framework that dynamically adjusts Channel State Information (CSI) sampling rates-operating at low frame rates (10Hz) during idle periods and high rates (100Hz) only when motion is detected. To our knowledge, this is the first low-complexity solution for occupancy detection using monostatic Wi-Fi sensing on a built-in Wi-Fi network interface controller (NIC) of a commercial off-the-shelf laptop that requires no external network infrastructure or specialized sensors. Our solution can scale across different environments and devices without calibration or retraining.
>
---
#### [new 094] Variance-Aware Adaptive Weighting for Diffusion Model Training
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于生成模型任务，解决扩散模型训练中不同噪声水平下的优化不平衡问题。通过引入方差感知的自适应加权策略，提升生成性能与训练稳定性。**

- **链接: [https://arxiv.org/pdf/2603.10391](https://arxiv.org/pdf/2603.10391)**

> **作者:** Nanlong Sun; Lei Shi
>
> **备注:** 15 pages, 8 figures, 1 table
>
> **摘要:** Diffusion models have recently achieved remarkable success in generative modeling, yet their training dynamics across different noise levels remain highly imbalanced, which can lead to inefficient optimization and unstable learning behavior. In this work, we investigate this imbalance from the perspective of loss variance across log-SNR levels and propose a variance-aware adaptive weighting strategy to address it. The proposed approach dynamically adjusts training weights based on the observed variance distribution, encouraging a more balanced optimization process across noise levels. Extensive experiments on CIFAR-10 and CIFAR-100 demonstrate that the proposed method consistently improves generative performance over standard training schemes, achieving lower Fréchet Inception Distance (FID) while also reducing performance variance across random seeds. Additional analysis, including loss-log-SNR visualization, variance heatmaps, and ablation studies, further reveal that the adaptive weighting effectively stabilizes training dynamics. These results highlight the potential of variance-aware training strategies for improving diffusion model optimization.
>
---
#### [new 095] An FPGA Implementation of Displacement Vector Search for Intra Pattern Copy in JPEG XS
- **分类: cs.AR; cs.CV; eess.IV**

- **简介: 该论文属于图像压缩任务，解决IPC中位移向量搜索的计算效率问题，提出一种高效FPGA架构设计以提升性能并降低功耗。**

- **链接: [https://arxiv.org/pdf/2603.10671](https://arxiv.org/pdf/2603.10671)**

> **作者:** Qiyue Chen; Yao Li; Jie Tao; Song Chen; Li Li; Dong Liu
>
> **摘要:** Recently, progress has been made on the Intra Pattern Copy (IPC) tool for JPEG XS, an image compression standard designed for low-latency and low-complexity coding. IPC performs wavelet-domain intra compensation predictions to reduce spatial redundancy in screen content. A key module of IPC is the displacement vector (DV) search, which aims to solve the optimal prediction reference offset. However, the DV search process is computationally intensive, posing challenges for practical hardware deployment. In this paper, we propose an efficient pipelined FPGA architecture design for the DV search module to promote the practical deployment of IPC. Optimized memory organization, which leverages the IPC computational characteristics and data inherent reuse patterns, is further introduced to enhance the performance. Experimental results show that our proposed architecture achieves a throughput of 38.3 Mpixels/s with a power consumption of 277 mW, demonstrating its feasibility for practical hardware implementation in IPC and other predictive coding tools, and providing a promising foundation for ASIC deployment.
>
---
#### [new 096] Neural Field Thermal Tomography: A Differentiable Physics Framework for Non-Destructive Evaluation
- **分类: cs.LG; cond-mat.mtrl-sci; cs.AI; cs.CV; physics.ins-det**

- **简介: 该论文提出NeFTY，用于从瞬态表面温度数据中定量重建材料三维属性，解决非破坏性检测中的缺陷定位问题。通过神经场和可微物理求解器实现高精度3D重建。**

- **链接: [https://arxiv.org/pdf/2603.11045](https://arxiv.org/pdf/2603.11045)**

> **作者:** Tao Zhong; Yixun Hu; Dongzhe Zheng; Aditya Sood; Christine Allen-Blanchette
>
> **备注:** 27 pages, 15 figures
>
> **摘要:** We propose Neural Field Thermal Tomography (NeFTY), a differentiable physics framework for the quantitative 3D reconstruction of material properties from transient surface temperature measurements. While traditional thermography relies on pixel-wise 1D approximations that neglect lateral diffusion, and soft-constrained Physics-Informed Neural Networks (PINNs) often fail in transient diffusion scenarios due to gradient stiffness, NeFTY parameterizes the 3D diffusivity field as a continuous neural field optimized through a rigorous numerical solver. By leveraging a differentiable physics solver, our approach enforces thermodynamic laws as hard constraints while maintaining the memory efficiency required for high-resolution 3D tomography. Our discretize-then-optimize paradigm effectively mitigates the spectral bias and ill-posedness inherent in inverse heat conduction, enabling the recovery of subsurface defects at arbitrary scales. Experimental validation on synthetic data demonstrates that NeFTY significantly improves the accuracy of subsurface defect localization over baselines. Additional details at this https URL
>
---
#### [new 097] Historical Consensus: Preventing Posterior Collapse via Iterative Selection of Gaussian Mixture Priors
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于深度学习任务，解决VAE中的后验崩溃问题。通过迭代选择高斯混合先验，构建历史共识训练机制，防止后验退化，无需依赖特定条件或架构。**

- **链接: [https://arxiv.org/pdf/2603.10935](https://arxiv.org/pdf/2603.10935)**

> **作者:** Zegu Zhang; Jian Zhang
>
> **备注:** 15 pages, 6 figures
>
> **摘要:** Variational autoencoders (VAEs) frequently suffer from posterior collapse, where latent variables become uninformative and the approximate posterior degenerates to the prior. Recent work has characterized this phenomenon as a phase transition governed by the spectral properties of the data covariance matrix. In this paper, we propose a fundamentally different approach: instead of avoiding collapse through architectural constraints or hyperparameter tuning, we eliminate the possibility of collapse altogether by leveraging the multiplicity of Gaussian mixture model (GMM) clusterings. We introduce Historical Consensus Training, an iterative selection procedure that progressively refines a set of candidate GMM priors through alternating optimization and selection. The key insight is that models trained to satisfy multiple distinct clustering constraints develop a historical barrier -- a region in parameter space that remains stable even when subsequently trained with a single objective. We prove that this barrier excludes the collapsed solution, and demonstrate through extensive experiments on synthetic and real-world datasets that our method achieves non-collapsed representations regardless of decoder variance or regularization strength. Our approach requires no explicit stability conditions (e.g., $\sigma^{\prime 2} < \lambda_{\max}$) and works with arbitrary neural architectures. The code is available at this https URL.
>
---
#### [new 098] ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于图像压缩任务，旨在解决传统编码效率低与计算成本高的问题。提出ARCHE框架，通过高效卷积设计实现高率失真效率和计算效率的平衡。**

- **链接: [https://arxiv.org/pdf/2603.10188](https://arxiv.org/pdf/2603.10188)**

> **作者:** Sofia Iliopoulou; Dimitris Ampeliotis; Athanassios Skodras
>
> **备注:** 16 pages, 12 figures
>
> **摘要:** Recent progress in learning-based image compression has demonstrated that end-to-end optimization can substantially outperform traditional codecs by jointly learning compact latent representations and probabilistic entropy models. However, many existing approaches achieve high rate-distortion efficiency at the expense of increased computational cost and limited parallelism. This paper presents ARCHE - Autoregressive Residual Compression with Hyperprior and Excitation, an end-to-end learned image compression framework that balances modeling accuracy and computational efficiency. The proposed architecture unifies hierarchical, spatial, and channel-based priors within a single probabilistic framework, capturing both global and local dependencies in the latent representation of the image, while employing adaptive feature recalibration and residual refinement to enhance latent representation quality. Without relying on recurrent or transformer-based components, ARCHE attains state-of-the-art rate-distortion efficiency: it reduces the BD-Rate by approximately 48% relative to the commonly used benchmark model of Balle et al., 30% relative to the channel-wise autoregressive model of Minnen & Singh and 5% against the VVC Intra codec on the Kodak benchmark dataset. The framework maintains computational efficiency with 95M parameters and 222ms running time per image. Visual comparisons confirm sharper textures and improved color fidelity, particularly at lower bit rates, demonstrating that accurate entropy modeling can be achieved through efficient convolutional designs suitable for practical deployment.
>
---
#### [new 099] MoXaRt: Audio-Visual Object-Guided Sound Interaction for XR
- **分类: cs.SD; cs.CV; cs.HC**

- **简介: 该论文提出MoXaRt系统，解决XR中复杂声景干扰问题，通过音视频协同分离声音源，提升语音可懂度与用户体验。**

- **链接: [https://arxiv.org/pdf/2603.10465](https://arxiv.org/pdf/2603.10465)**

> **作者:** Tianyu Xu; Sieun Kim; Qianhui Zheng; Ruoyu Xu; Tejasvi Ravi; Anuva Kulkarni; Katrina Passarella-Ward; Junyi Zhu; Adarsh Kowdle
>
> **摘要:** In Extended Reality (XR), complex acoustic environments often overwhelm users, compromising both scene awareness and social engagement due to entangled sound sources. We introduce MoXaRt, a real-time XR system that uses audio-visual cues to separate these sources and enable fine-grained sound interaction. MoXaRt's core is a cascaded architecture that performs coarse, audio-only separation in parallel with visual detection of sources (e.g., faces, instruments). These visual anchors then guide refinement networks to isolate individual sources, separating complex mixes of up to 5 concurrent sources (e.g., 2 voices + 3 instruments) with ~2 second processing latency. We validate MoXaRt through a technical evaluation on a new dataset of 30 one-minute recordings featuring concurrent speech and music, and a 22-participant user study. Empirical results indicate that our system significantly enhances speech intelligibility, yielding a 36.2% (p < 0.01) increase in listening comprehension within adversarial acoustic environments while substantially reducing cognitive load (p < 0.001), thereby paving the way for more perceptive and socially adept XR experiences.
>
---
#### [new 100] ID-LoRA: Identity-Driven Audio-Video Personalization with In-Context LoRA
- **分类: cs.SD; cs.CV; cs.GR**

- **简介: 该论文提出ID-LoRA，解决音视频个性化生成任务，通过联合生成外观和声音，提升语音相似度与风格控制。**

- **链接: [https://arxiv.org/pdf/2603.10256](https://arxiv.org/pdf/2603.10256)**

> **作者:** Aviad Dahan; Moran Yanuka; Noa Kraicer; Lior Wolf; Raja Giryes
>
> **摘要:** Existing video personalization methods preserve visual likeness but treat video and audio separately. Without access to the visual scene, audio models cannot synchronize sounds with on-screen actions; and because classical voice-cloning models condition only on a reference recording, a text prompt cannot redirect speaking style or acoustic environment. We propose ID-LoRA (Identity-Driven In-Context LoRA), which jointly generates a subject's appearance and voice in a single model, letting a text prompt, a reference image, and a short audio clip govern both modalities together. ID-LoRA adapts the LTX-2 joint audio-video diffusion backbone via parameter-efficient In-Context LoRA and, to our knowledge, is the first method to personalize visual appearance and voice in a single generative pass. Two challenges arise. Reference and generation tokens share the same positional-encoding space, making them hard to distinguish; we address this with negative temporal positions, placing reference tokens in a disjoint RoPE region while preserving their internal temporal structure. Speaker characteristics also tend to be diluted during denoising; we introduce identity guidance, a classifier-free guidance variant that amplifies speaker-specific features by contrasting predictions with and without the reference signal. In human preference studies, ID-LoRA is preferred over Kling 2.6 Pro by 73% of annotators for voice similarity and 65% for speaking style. On cross-environment settings, speaker similarity improves by 24% over Kling, with the gap widening as conditions diverge. A preliminary user study further suggests that joint generation provides a useful inductive bias for physically grounded sound synthesis. ID-LoRA achieves these results with only ~3K training pairs on a single GPU. Code, models, and data will be released.
>
---
#### [new 101] Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于图像恢复任务，解决将分数模型集成到ADMM中的收敛性问题。提出AC-DC denoiser框架，提升重建质量。**

- **链接: [https://arxiv.org/pdf/2603.10281](https://arxiv.org/pdf/2603.10281)**

> **作者:** Rajesh Shrestha; Xiao Fu
>
> **摘要:** While score-based generative models have emerged as powerful priors for solving inverse problems, directly integrating them into optimization algorithms such as ADMM remains nontrivial. Two central challenges arise: i) the mismatch between the noisy data manifolds used to train the score functions and the geometry of ADMM iterates, especially due to the influence of dual variables, and ii) the lack of convergence understanding when ADMM is equipped with score-based denoisers. To address the manifold mismatch issue, we propose ADMM plug-and-play (ADMM-PnP) with the AC-DC denoiser, a new framework that embeds a three-stage denoiser into ADMM: (1) auto-correction (AC) via additive Gaussian noise, (2) directional correction (DC) using conditional Langevin dynamics, and (3) score-based denoising. In terms of convergence, we establish two results: first, under proper denoiser parameters, each ADMM iteration is a weakly nonexpansive operator, ensuring high-probability fixed-point $\textit{ball convergence}$ using a constant step size; second, under more relaxed conditions, the AC-DC denoiser is a bounded denoiser, which leads to convergence under an adaptive step size schedule. Experiments on a range of inverse problems demonstrate that our method consistently improves solution quality over a variety of baselines.
>
---
#### [new 102] Daily-Omni: Towards Audio-Visual Reasoning with Temporal Alignment across Modalities
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态任务，旨在解决跨模态时间对齐问题。提出Daily-Omni基准，评估模型在音频视频联合推理中的表现。**

- **链接: [https://arxiv.org/pdf/2505.17862](https://arxiv.org/pdf/2505.17862)**

> **作者:** Ziwei Zhou; Rui Wang; Zuxuan Wu; Yu-Gang Jiang
>
> **摘要:** Recent Multimodal Large Language Models (MLLMs) achieve promising performance on visual and audio benchmarks independently. However, the ability of these models to process cross-modal information synchronously remains largely unexplored. We introduce Daily-Omni, a multiple-choice Audio-Visual QA benchmark featuring 684 real-world videos and 1,197 questions spanning 6 task families that explicitly require cross-modal temporal reasoning. To support scalable benchmark construction, we develop a semi-automatic pipeline for annotation, cross-modal consistency refinement, temporal alignment elicitation, and text-only leakage filtering, followed by human verification. We further provide a diagnostic evaluation suite and extensively evaluate 24 foundation models under 37 model--modality settings (Audio+Video / Audio-only / Video-only / Text-only). Finally, we include a training-free modular diagnostic baseline that composes off-the-shelf unimodal models to serve as a diagnostic baseline and to illustrate how explicit temporal alignment signals affect performance. Results indicate that many end-to-end MLLMs still struggle on alignment-critical questions, suggesting that robust cross-modal temporal alignment remains an important open challenge.
>
---
#### [new 103] AsyncMDE: Real-Time Monocular Depth Estimation via Asynchronous Spatial Memory
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出AsyncMDE，解决实时单目深度估计问题，通过异步机制降低计算成本，提升边缘部署可行性。**

- **链接: [https://arxiv.org/pdf/2603.10438](https://arxiv.org/pdf/2603.10438)**

> **作者:** Lianjie Ma; Yuquan Li; Bingzheng Jiang; Ziming Zhong; Han Ding; Lijun Zhu
>
> **备注:** 8 pages, 5 figures, 5 tables
>
> **摘要:** Foundation-model-based monocular depth estimation offers a viable alternative to active sensors for robot perception, yet its computational cost often prohibits deployment on edge platforms. Existing methods perform independent per-frame inference, wasting the substantial computational redundancy between adjacent viewpoints in continuous robot operation. This paper presents AsyncMDE, an asynchronous depth perception system consisting of a foundation model and a lightweight model that amortizes the foundation model's computational cost over time. The foundation model produces high-quality spatial features in the background, while the lightweight model runs asynchronously in the foreground, fusing cached memory with current observations through complementary fusion, outputting depth estimates, and autoregressively updating the memory. This enables cross-frame feature reuse with bounded accuracy degradation. At a mere 3.83M parameters, it operates at 237 FPS on an RTX 4090, recovering 77% of the accuracy gap to the foundation model while achieving a 25X parameter reduction. Validated across indoor static, dynamic, and synthetic extreme-motion benchmarks, AsyncMDE degrades gracefully between refreshes and achieves 161FPS on a Jetson AGX Orin with TensorRT, clearly demonstrating its feasibility for real-time edge deployment.
>
---
#### [new 104] MUNIChus: Multilingual News Image Captioning Benchmark
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出多语言新闻图像描述基准MUNIChus，解决多语言新闻图像生成任务中的数据稀缺问题，涵盖9种语言，包含低资源语言，并评估多种模型。**

- **链接: [https://arxiv.org/pdf/2603.10613](https://arxiv.org/pdf/2603.10613)**

> **作者:** Yuji Chen; Alistair Plum; Hansi Hettiarachchi; Diptesh Kanojia; Saroj Basnet; Marcos Zampieri; Tharindu Ranasinghe
>
> **备注:** Accepted to LREC 2026 (The Fifteenth biennial Language Resources and Evaluation Conference)
>
> **摘要:** The goal of news image captioning is to generate captions by integrating news article content with corresponding images, highlighting the relationship between textual context and visual elements. The majority of research on news image captioning focuses on English, primarily because datasets in other languages are scarce. To address this limitation, we create the first multilingual news image captioning benchmark, MUNIChus, comprising 9 languages, including several low-resource languages such as Sinhala and Urdu. We evaluate various state-of-the-art neural news image captioning models on MUNIChus and find that news image captioning remains challenging. We also make MUNIChus publicly available with over 20 models already benchmarked. MUNIChus opens new avenues for further advancements in developing and evaluating multilingual news image captioning models.
>
---
#### [new 105] Unlearning the Unpromptable: Prompt-free Instance Unlearning in Diffusion Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于机器学习中的模型遗忘任务，旨在解决无法通过文本提示消除的特定输出问题。通过图像编辑等方法实现对特定实例的遗忘，同时保持模型整体性能。**

- **链接: [https://arxiv.org/pdf/2603.10445](https://arxiv.org/pdf/2603.10445)**

> **作者:** Kyungryeol Lee; Kyeonghyun Lee; Seongmin Hong; Byung Hyun Lee; Se Young Chun
>
> **备注:** 12 pages
>
> **摘要:** Machine unlearning aims to remove specific outputs from trained models, often at the concept level, such as forgetting all occurrences of a particular celebrity or filtering content via text prompts. However, many undesired outputs, such as an individual's face or generations culturally or factually misinterpreted, cannot often be specified by text prompts. We address this underexplored setting of instance unlearning for outputs that are undesired but unpromptable, where the goal is to forget target outputs selectively while preserving the rest. To this end, we introduce an effective surrogate-based unlearning method that leverages image editing, timestep-aware weighting, and gradient surgery to guide trained diffusion models toward forgetting specific outputs. Experiments on conditional (Stable Diffusion 3) and unconditional (DDPM-CelebA) diffusion models demonstrate that our prompt-free method uniquely unlearns unpromptable outputs, such as faces and culturally inaccurate depictions, with preserved integrity, unlike prompt-based and prompt-free baselines. Our proposed method would serve as a practical hotfix for diffusion model providers to ensure privacy protection and ethical compliance.
>
---
#### [new 106] MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于在线高精度地图构建任务，解决标注成本高的问题。通过引入地理空间对比学习，提升特征表示，实现半监督训练，提升地图感知性能。**

- **链接: [https://arxiv.org/pdf/2603.10688](https://arxiv.org/pdf/2603.10688)**

> **作者:** Jonas Merkert; Alexander Blumberg; Jan-Hendrik Pauls; Christoph Stiller
>
> **摘要:** Autonomous vehicles rely on map information to understand the world around them. However, the creation and maintenance of offline high-definition (HD) maps remains costly. A more scalable alternative lies in online HD map construction, which only requires map annotations at training time. To further reduce the need for annotating vast training labels, self-supervised training provides an alternative. This work focuses on improving the latent birds-eye-view (BEV) feature grid representation within a vectorized online HD map construction model by enforcing geospatial consistency between overlapping BEV feature grids as part of a contrastive loss function. To ensure geospatial overlap for contrastive pairs, we introduce an approach to analyze the overlap between traversals within a given dataset and generate subsidiary dataset splits following adjustable multi-traversal requirements. We train the same model supervised using a reduced set of single-traversal labeled data and self-supervised on a broader unlabeled set of data following our multi-traversal requirements, effectively implementing a semi-supervised approach. Our approach outperforms the supervised baseline across the board, both quantitatively in terms of the downstream tasks vectorized map perception performance and qualitatively in terms of segmentation in the principal component analysis (PCA) visualization of the BEV feature space.
>
---
#### [new 107] Naïve Exposure of Generative AI Capabilities Undermines Deepfake Detection
- **分类: cs.CR; cs.AI; cs.CV**

- **简介: 论文研究生成式AI对深度伪造检测的威胁，指出其通过语义保留的图像优化绕过检测系统。任务是评估检测有效性，解决真实场景下AI能力被滥用的问题。**

- **链接: [https://arxiv.org/pdf/2603.10504](https://arxiv.org/pdf/2603.10504)**

> **作者:** Sunpill Kim; Chanwoo Hwang; Minsu Kim; Jae Hong Seo
>
> **摘要:** Generative AI systems increasingly expose powerful reasoning and image refinement capabilities through user-facing chatbot interfaces. In this work, we show that the naïve exposure of such capabilities fundamentally undermines modern deepfake detectors. Rather than proposing a new image manipulation technique, we study a realistic and already-deployed usage scenario in which an adversary uses only benign, policy-compliant prompts and commercial generative AI systems. We demonstrate that state-of-the-art deepfake detection methods fail under semantic-preserving image refinement. Specifically, we show that generative AI systems articulate explicit authenticity criteria and inadvertently externalize them through unrestricted reasoning, enabling their direct reuse as refinement objectives. As a result, refined images simultaneously evade detection, preserve identity as verified by commercial face recognition APIs, and exhibit substantially higher perceptual quality. Importantly, we find that widely accessible commercial chatbot services pose a significantly greater security risk than open-source models, as their superior realism, semantic controllability, and low-barrier interfaces enable effective evasion by non-expert users. Our findings reveal a structural mismatch between the threat models assumed by current detection frameworks and the actual capabilities of real-world generative AI. While detection baselines are largely shaped by prior benchmarks, deployed systems expose unrestricted authenticity reasoning and refinement despite stringent safety controls in other domains.
>
---
#### [new 108] The Orthogonal Vulnerabilities of Generative AI Watermarks: A Comparative Empirical Benchmark of Spatial and Latent Provenance
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于数字水印安全任务，旨在解决生成式AI水印的脆弱性问题。通过对比空间与潜在域水印，发现其存在数学正交漏洞，提出多域加密架构必要性。**

- **链接: [https://arxiv.org/pdf/2603.10323](https://arxiv.org/pdf/2603.10323)**

> **作者:** Jesse Yu; Nicholas Wei
>
> **摘要:** As open-weights generative AI rapidly proliferates, the ability to synthesize hyper-realistic media has introduced profound challenges to digital trust. Automated disinformation and AI-generated imagery have made robust digital provenance a critical cybersecurity imperative. Currently, state-of-the-art invisible watermarks operate within one of two primary mathematical manifolds: the spatial domain (post-generation pixel embedding) or the latent domain (pre-generation frequency embedding). While existing literature frequently evaluates these models against isolated, classical distortions, there is a critical lack of rigorous, comparative benchmarking against modern generative AI editing tools. In this study, we empirically evaluate two leading representative paradigms, RivaGAN (Spatial) and Tree-Ring (Latent), utilizing an automated Attack Simulation Engine across 30 intensity intervals of geometric and generative perturbations. We formalize an "Adversarial Evasion Region" (AER) framework to measure cryptographic degradation against semantic visual retention (OpenCLIP > 70.0). Our statistical analysis ($n=100$ per interval, $MOE = \pm 3.92\%$) reveals that these domains possess mutually exclusive, mathematically orthogonal vulnerabilities. Spatial watermarks experience severe cryptographic degradation under algorithmic pixel-rewriting (exhibiting a 67.47% AER evasion rate under Img2Img translation), whereas latent watermarks exhibit profound fragility against geometric misalignment (yielding a 43.20% AER evasion rate under static cropping). By proving that single-domain watermarking is fundamentally insufficient against modern adversarial toolsets, this research exposes a systemic vulnerability in current digital provenance standards and establishes the foundational exigence for future multi-domain cryptographic architectures.
>
---
## 更新

#### [replaced 001] GOUHFI 2.0: A Next-Generation Toolbox for Brain Segmentation and Cortex Parcellation at Ultra-High Field MRI
- **分类: eess.IV; cs.CV; physics.med-ph**

- **链接: [https://arxiv.org/pdf/2601.09006](https://arxiv.org/pdf/2601.09006)**

> **作者:** Marc-Antoine Fortin; Anne Louise Kristoffersen; Paal Erik Goa
>
> **摘要:** Ultra-High Field MRI (UHF-MRI) is increasingly used in large-scale neuroimaging studies, yet automatic brain segmentation and cortical parcellation remain challenging due to signal inhomogeneities, heterogeneous contrasts and resolutions, and the limited availability of tools optimized for UHF data. Standard software packages such as FastSurferVINN and SynthSeg+ often yield suboptimal results when applied directly to UHF images, thereby restricting region-based quantitative analyses. To address this need, we introduce GOUHFI 2.0, an updated implementation of GOUHFI that incorporates increased training data variability and additional functionalities, including cortical parcellation and volumetry. GOUHFI 2.0 preserves the contrast- and resolution-agnostic design of the original toolbox while introducing two independently trained 3D U-Net segmentation tasks. The first performs whole-brain segmentation into 35 labels across contrasts, resolutions, field strengths and populations, using a domain-randomization strategy and a training dataset of 238 subjects. Using the same training data, the second network performs cortical parcellation into 62 labels following the Desikan-Killiany-Tourville (DKT) protocol. Across multiple datasets, GOUHFI 2.0 demonstrated improved segmentation accuracy relative to the original toolbox, particularly in heterogeneous cohorts, and produced reliable cortical parcellations. In addition, the integrated volumetry pipeline yielded results consistent with standard volumetric workflows. Overall, GOUHFI 2.0 provides a comprehensive solution for brain segmentation, parcellation and volumetry across field strengths, and constitutes the first deep-learning toolbox enabling robust cortical parcellation at UHF-MRI.
>
---
#### [replaced 002] Mindstorms in Natural Language-Based Societies of Mind
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.MA**

- **简介: 该论文探讨自然语言驱动的多智能体系统（NLSOM），解决复杂AI任务。通过模块化设计提升多模态推理能力，实验验证其在视觉、生成等任务中的有效性。**

- **链接: [https://arxiv.org/pdf/2305.17066](https://arxiv.org/pdf/2305.17066)**

> **作者:** Mingchen Zhuge; Haozhe Liu; Francesco Faccio; Dylan R. Ashley; Róbert Csordás; Anand Gopalakrishnan; Abdullah Hamdi; Hasan Abed Al Kader Hammoud; Vincent Herrmann; Kazuki Irie; Louis Kirsch; Bing Li; Guohao Li; Shuming Liu; Jinjie Mai; Piotr Piękos; Aditya Ramesh; Imanol Schlag; Weimin Shi; Aleksandar Stanić; Wenyi Wang; Yuhui Wang; Mengmeng Xu; Deng-Ping Fan; Bernard Ghanem; Jürgen Schmidhuber
>
> **备注:** published in Computational Visual Media Journal (CVMJ); 9 pages in main text + 7 pages of references + 38 pages of appendices, 14 figures in main text + 13 in appendices, 7 tables in appendices
>
> **摘要:** Both Minsky's "society of mind" and Schmidhuber's "learning to think" inspire diverse societies of large multimodal neural networks (NNs) that solve problems by interviewing each other in a "mindstorm." Recent implementations of NN-based societies of minds consist of large language models (LLMs) and other NN-based experts communicating through a natural language interface. In doing so, they overcome the limitations of single LLMs, improving multimodal zero-shot reasoning. In these natural language-based societies of mind (NLSOMs), new agents -- all communicating through the same universal symbolic language -- are easily added in a modular fashion. To demonstrate the power of NLSOMs, we assemble and experiment with several of them (having up to 129 members), leveraging mindstorms in them to solve some practical AI tasks: visual question answering, image captioning, text-to-image synthesis, 3D generation, egocentric retrieval, embodied AI, and general language-based task solving. We view this as a starting point towards much larger NLSOMs with billions of agents-some of which may be humans. And with this emergence of great societies of heterogeneous minds, many new research questions have suddenly become paramount to the future of artificial intelligence. What should be the social structure of an NLSOM? What would be the (dis)advantages of having a monarchical rather than a democratic structure? How can principles of NN economies be used to maximize the total reward of a reinforcement learning NLSOM? In this work, we identify, discuss, and try to answer some of these questions.
>
---
#### [replaced 003] ENIGMA-360: An Ego-Exo Dataset for Human Behavior Understanding in Industrial Scenarios
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.09741](https://arxiv.org/pdf/2603.09741)**

> **作者:** Francesco Ragusa; Rosario Leonardi; Michele Mazzamuto; Daniele Di Mauro; Camillo Quattrocchi; Alessandro Passanisi; Irene D'Ambra; Antonino Furnari; Giovanni Maria Farinella
>
> **摘要:** Understanding human behavior from complementary egocentric (ego) and exocentric (exo) points of view enables the development of systems that can support workers in industrial environments and enhance their safety. However, progress in this area is hindered by the lack of datasets capturing both views in realistic industrial scenarios. To address this gap, we propose ENIGMA-360, a new ego-exo dataset acquired in a real industrial scenario. The dataset is composed of 180 egocentric and 180 exocentric procedural videos temporally synchronized offering complementary information of the same scene. The 360 videos have been labeled with temporal and spatial annotations, enabling the study of different aspects of human behavior in industrial domain. We provide baseline experiments for 3 foundational tasks for human behavior understanding: 1) Temporal Action Segmentation, 2) Keystep Recognition and 3) Egocentric Human-Object Interaction Detection, showing the limits of state-of-the-art approaches on this challenging scenario. These results highlight the need for new models capable of robust ego-exo understanding in real-world environments. We publicly release the dataset and its annotations at this https URL.
>
---
#### [replaced 004] Unsupervised training of keypoint-agnostic descriptors for flexible retinal image registration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.02787](https://arxiv.org/pdf/2505.02787)**

> **作者:** David Rivas-Villar; Álvaro S. Hervella; José Rouco; Jorge Novo
>
> **摘要:** Current color fundus image registration approaches are limited, among other things, by the lack of labeled data, which is even more significant in the medical domain, motivating the use of unsupervised learning. Therefore, in this work, we develop a novel unsupervised descriptor learning method that does not rely on keypoint detection. This enables the resulting descriptor network to be agnostic to the keypoint detector used during the registration inference. To validate this approach, we perform an extensive and comprehensive comparison on the reference public retinal image registration dataset. Additionally, we test our method with multiple keypoint detectors of varied nature, even proposing some novel ones. Our results demonstrate that the proposed approach offers accurate registration, not incurring in any performance loss versus supervised methods. Additionally, it demonstrates accurate performance regardless of the keypoint detector used. Thus, this work represents a notable step towards leveraging unsupervised learning in the medical domain.
>
---
#### [replaced 005] Similarity-as-Evidence: Calibrating Overconfident VLMs for Interpretable and Label-Efficient Medical Active Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.18867](https://arxiv.org/pdf/2602.18867)**

> **作者:** Zhuofan Xie; Zishan Lin; Jinliang Lin; Jie Qi; Shaohua Hong; Shuo Li
>
> **备注:** Accepted to CVPR 2026 (to appear)
>
> **摘要:** Active Learning (AL) reduces annotation costs in medical imaging by selecting only the most informative samples for labeling, but suffers from cold-start when labeled data are scarce. Vision-Language Models (VLMs) address the cold-start problem via zero-shot predictions, yet their temperature-scaled softmax outputs treat text-image similarities as deterministic scores while ignoring inherent uncertainty, leading to overconfidence. This overconfidence misleads sample selection, wasting annotation budgets on uninformative cases. To overcome these limitations, the Similarity-as-Evidence (SaE) framework calibrates text-image similarities by introducing a Similarity Evidence Head (SEH), which reinterprets the similarity vector as evidence and parameterizes a Dirichlet distribution over labels. In contrast to a standard softmax that enforces confident predictions even under weak signals, the Dirichlet formulation explicitly quantifies lack of evidence (vacuity) and conflicting evidence (dissonance), thereby mitigating overconfidence caused by rigid softmax normalization. Building on this, SaE employs a dual-factor acquisition strategy: high-vacuity samples (e.g., rare diseases) are prioritized in early rounds to ensure coverage, while high-dissonance samples (e.g., ambiguous diagnoses) are prioritized later to refine boundaries, providing clinically interpretable selection rationales. Experiments on ten public medical imaging datasets with a 20% label budget show that SaE attains state-of-the-art macro-averaged accuracy of 82.57%. On the representative BTMRI dataset, SaE also achieves superior calibration, with a negative log-likelihood (NLL) of 0.425.
>
---
#### [replaced 006] PatchDenoiser: Parameter-efficient multi-scale patch learning and fusion denoiser for Low-dose CT imaging
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.21987](https://arxiv.org/pdf/2602.21987)**

> **作者:** Jitindra Fartiyal; Pedro Freire; Sergei K. Turitsyn; Sergei G. Solovski
>
> **摘要:** Low-dose CT images are essential for reducing radiation exposure in cancer screening, pediatric imaging, and longitudinal monitoring protocols, but their quality is often degraded by noise from low-dose acquisition, patient motion, or scanner limitations, affecting both clinical interpretation and downstream analysis. Traditional filtering approaches often over-smooth and lose fine anatomical details, while deep learning methods, including CNNs, GANs, and transformers, may struggle to preserve such details or require large, computationally expensive models, limiting clinical practicality. We propose PatchDenoiser, a lightweight, energy-efficient multi-scale patch-based denoising framework. It decomposes denoising into local texture extraction and global context aggregation, fused via a spatially aware patch fusion strategy. This design enables effective noise suppression while preserving fine structural and anatomical details. PatchDenoiser is ultra-lightweight, with far fewer parameters and lower computational complexity than CNN, GAN, and transformer based denoisers. On the 2016 Mayo Low-Dose CT dataset, PatchDenoiser consistently outperforms state-of-the-art CNN- and GAN-based methods in PSNR and SSIM. It is robust to variations in slice thickness, reconstruction kernels, and HU windows, generalizes across scanners without fine-tuning, and reduces parameters by ~9x and energy consumption per inference by ~27x compared with conventional CNN denoisers. PatchDenoiser thus provides a practical, scalable, and computationally efficient solution for medical image denoising, balancing performance, robustness, and clinical deployability.
>
---
#### [replaced 007] Transformer-Based Multi-Region Segmentation and Radiomic Analysis of HR-pQCT Imaging for Osteoporosis Classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.09137](https://arxiv.org/pdf/2603.09137)**

> **作者:** Mohseu Rashid Subah; Mohammed Abdul Gani Zilani; Thomas L. Nickolas; Matthew R. Allen; Stuart J. Warden; Rachel K. Surowiec
>
> **摘要:** Osteoporosis is a skeletal disease typically diagnosed using dual-energy X-ray absorptiometry (DXA), which quantifies areal bone mineral density but overlooks bone microarchitecture and surrounding soft tissues. High-resolution peripheral quantitative computed tomography (HR-pQCT) enables three-dimensional microstructural imaging with minimal radiation. However, current analysis pipelines largely focus on mineralized bone compartments, leaving much of the acquired image data underutilized. We introduce a fully automated framework for binary osteoporosis classification using radiomics features extracted from anatomically segmented HR-pQCT images. To our knowledge, this work is the first to leverage a transformer-based segmentation architecture, i.e., the SegFormer, for fully automated multi-region HR-pQCT analysis. The SegFormer model simultaneously delineated the cortical and trabecular bone of the tibia and fibula along with surrounding soft tissues and achieved a mean F1 score of 95.36%. Soft tissues were further subdivided into skin, myotendinous, and adipose regions through post-processing. From each region, 939 radiomic features were extracted and dimensionally reduced to train six machine learning classifiers on an independent dataset comprising 20,496 images from 122 HR-pQCT scans. The best image level performance was achieved using myotendinous tissue features, yielding an accuracy of 80.08% and an area under the receiver operating characteristic curve (AUROC) of 0.85, outperforming bone-based models. At the patient level, replacing standard biological, DXA, and HR-pQCT parameters with soft tissue radiomics improved AUROC from 0.792 to 0.875. These findings demonstrate that automated, multi-region HR-pQCT segmentation enables the extraction of clinically informative signals beyond bone alone, highlighting the importance of integrated tissue assessment for osteoporosis detection.
>
---
#### [replaced 008] A Saccade-inspired Approach to Image Classification using Vision Transformer Attention Maps
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.09613](https://arxiv.org/pdf/2603.09613)**

> **作者:** Matthis Dallain; Laurent Rodriguez; Laurent Udo Perrinet; Benoît Miramond
>
> **备注:** 16 page, 11 figure main paper + 3 pages, 6 appendix
>
> **摘要:** Human vision achieves remarkable perceptual performance while operating under strict metabolic constraints. A key ingredient is the selective attention mechanism, driven by rapid saccadic eye movements that constantly reposition the high-resolution fovea onto task-relevant locations, unlike conventional AI systems that process entire images with equal emphasis. Our work aims to draw inspiration from the human visual system to create smarter, more efficient image processing models. Using DINO, a self-supervised Vision Transformer that produces attention maps strikingly similar to human gaze patterns, we explore a saccade inspired method to focus the processing of information on key regions in visual space. To do so, we use the ImageNet dataset in a standard classification task and measure how each successive saccade affects the model's class scores. This selective-processing strategy preserves most of the full-image classification performance and can even outperform it in certain cases. By benchmarking against established saliency models built for human gaze prediction, we demonstrate that DINO provides superior fixation guidance for selecting informative regions. These findings highlight Vision Transformer attention as a promising basis for biologically inspired active vision and open new directions for efficient, neuromorphic visual processing.
>
---
#### [replaced 009] KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.04268](https://arxiv.org/pdf/2602.04268)**

> **作者:** Siyu Jiang; Feiyang Chen; Xiaojin Zhang; Kun He
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Despite the significant progress of Multimodal Large Language Models (MLLMs) across diverse tasks, hallucination -- corresponding to the generation of visually inconsistent objects, attributes, or relations -- remains a major obstacle to their reliable deployment. Unlike pure language models, MLLMs must ground their generation process in visual inputs. However, existing models often suffer from semantic drift during decoding, causing outputs to diverge from visual facts as the sequence length increases. To address this issue, we propose KVSmooth, a training-free and plug-and-play method that mitigates hallucination by performing attention-entropy-guided adaptive smoothing on hidden states. Specifically, KVSmooth applies an exponential moving average (EMA) to both keys and values in the KV-Cache, while dynamically quantifying the sink degree of each token through the entropy of its attention distribution to adaptively adjust the smoothing strength. Unlike computationally expensive retraining or contrastive decoding methods, KVSmooth operates efficiently during inference without additional training or model modification. Extensive experiments demonstrate that KVSmooth significantly reduces hallucination ($\mathit{CHAIR}_{S}$ from $41.8 \rightarrow 18.2$) while improving overall performance ($F_1$ score from $77.5 \rightarrow 79.2$), achieving higher precision and recall simultaneously. In contrast, prior methods often improve one at the expense of the other, validating the effectiveness and generality of our approach.
>
---
#### [replaced 010] Consistency-based Abductive Reasoning over Perceptual Errors of Multiple Pre-trained Models in Novel Environments
- **分类: cs.AI; cs.CV; cs.LG; cs.LO**

- **链接: [https://arxiv.org/pdf/2505.19361](https://arxiv.org/pdf/2505.19361)**

> **作者:** Mario Leiva; Noel Ngu; Joshua Shay Kricheli; Aditya Taparia; Ransalu Senanayake; Paulo Shakarian; Nathaniel Bastian; John Corcoran; Gerardo Simari
>
> **备注:** Accepted to AAAI 2026. Code available at this https URL
>
> **摘要:** The deployment of pre-trained perception models in novel environments often leads to performance degradation due to distributional shifts. Although recent artificial intelligence approaches for metacognition use logical rules to characterize and filter model errors, improving precision often comes at the cost of reduced recall. This paper addresses the hypothesis that leveraging multiple pre-trained models can mitigate this recall reduction. We formulate the challenge of identifying and managing conflicting predictions from various models as a consistency-based abduction problem, building on the idea of abductive learning (ABL) but applying it to test-time instead of training. The input predictions and the learned error detection rules derived from each model are encoded in a logic program. We then seek an abductive explanation--a subset of model predictions--that maximizes prediction coverage while ensuring the rate of logical inconsistencies (derived from domain constraints) remains below a specified threshold. We propose two algorithms for this knowledge representation task: an exact method based on Integer Programming (IP) and an efficient Heuristic Search (HS). Through extensive experiments on a simulated aerial imagery dataset featuring controlled, complex distributional shifts, we demonstrate that our abduction-based framework outperforms individual models and standard ensemble baselines, achieving, for instance, average relative improvements of approximately 13.6\% in F1-score and 16.6\% in accuracy across 15 diverse test datasets when compared to the best individual model. Our results validate the use of consistency-based abduction as an effective mechanism to robustly integrate knowledge from multiple imperfect models in challenging, novel scenarios.
>
---
#### [replaced 011] REMSA: Foundation Model Selection for Remote Sensing via a Constraint-Aware Agent
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.17442](https://arxiv.org/pdf/2511.17442)**

> **作者:** Binger Chen; Tacettin Emre Bök; Behnood Rasti; Volker Markl; Begüm Demir
>
> **备注:** Code and data available at this https URL
>
> **摘要:** Foundation Models (FMs) are increasingly integrated into remote sensing (RS) pipelines. These models include unimodal vision encoders and multimodal architectures. FMs are adapted to diverse perception tasks, such as image classification, change detection, and visual question answering. However, selecting the most suitable remote sensing foundation model (RSFM) for a specific task remains challenging due to scattered documentation, heterogeneous formats, and complex deployment constraints. To address this, we first introduce the RSFM Database (RS-FMD), the first structured and schema-guided resource covering over 160 RSFMs trained on various data modalities, spanning different spatial, spectral, and temporal resolutions, considering different learning paradigms. Built upon RS-FMD, we further present REMSA, a constraint-aware agent that enables automated RSFM selection from natural language queries. REMSA combines structured FM metadata retrieval with a task-driven decision workflow. In detail, it interprets user input, clarifies missing constraints, ranks models via in-context learning, and provides transparent justifications. Our system supports various RS tasks and data modalities, enabling personalized, reproducible, and efficient FM selection. To evaluate REMSA, we construct a benchmark of 100 expert-verified RS query scenarios. Each query is evaluated across 4 systems and 3 LLM backbones, with the top-3 selected models manually assessed by domain experts. This results in 3,000 expert-scored task--system--model configurations under our novel expert-centered evaluation protocol. REMSA outperforms multiple baselines, showing its practical utility in real decision-making applications. REMSA operates entirely on publicly available metadata of open source RSFMs, without accessing private or sensitive data.
>
---
#### [replaced 012] AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.22740](https://arxiv.org/pdf/2602.22740)**

> **作者:** Tongfei Chen; Shuo Yang; Yuguang Yang; Linlin Yang; Runtang Guo; Changbai Li; He Long; Chunyu Xie; Dawei Leng; Baochang Zhang
>
> **备注:** ICLR 2026 conference paper
>
> **摘要:** Referring Image Segmentation (RIS) aims to segment the object in an image uniquely referred to by a natural language expression. However, RIS training often contains hard-to-align and instance-specific visual signals; optimizing on such pixels injects misleading gradients and drives the model in the wrong direction. By explicitly estimating pixel-level vision-language alignment, the learner can suppress low-alignment regions, concentrate on reliable cues, and acquire more generalizable alignment features. In this paper, we propose Alignment-Aware Masked Learning (AML), a simple yet effective training strategy that quantifies region-referent alignment (PMME) and filters out unreliable pixels during optimization (AFM). Specifically, each sample first computes a similarity map between visual and textual features, and then masks out pixels falling below an adaptive similarity threshold, thereby excluding poorly aligned regions from the training process. AML does not require architectural changes and incurs no inference overhead, directing attention to the areas aligned with the textual description. Experiments on the RefCOCO (vanilla/+/g) datasets show that AML achieves state-of-the-art results across all 8 splits, and beyond improving RIS performance, AML also enhances the model's robustness to diverse descriptions and scenarios. Code is available at this https URL.
>
---
#### [replaced 013] SGI: Structured 2D Gaussians for Efficient and Compact Large Image Representation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.07789](https://arxiv.org/pdf/2603.07789)**

> **作者:** Zixuan Pan; Kaiyuan Tang; Jun Xia; Yifan Qin; Lin Gu; Chaoli Wang; Jianxu Chen; Yiyu Shi
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** 2D Gaussian Splatting has emerged as a novel image representation technique that can support efficient rendering on low-end devices. However, scaling to high-resolution images requires optimizing and storing millions of unstructured Gaussian primitives independently, leading to slow convergence and redundant parameters. To address this, we propose Structured Gaussian Image (SGI), a compact and efficient framework for representing high-resolution images. SGI decomposes a complex image into multi-scale local spaces defined by a set of seeds. Each seed corresponds to a spatially coherent region and, together with lightweight multi-layer perceptrons (MLPs), generates structured implicit 2D neural Gaussians. This seed-based formulation imposes structural regularity on otherwise unstructured Gaussian primitives, which facilitates entropy-based compression at the seed level to reduce the total storage. However, optimizing seed parameters directly on high-resolution images is a challenging and non-trivial task. Therefore, we designed a multi-scale fitting strategy that refines the seed representation in a coarse-to-fine manner, substantially accelerating convergence. Quantitative and qualitative evaluations demonstrate that SGI achieves up to 7.5x compression over prior non-quantized 2D Gaussian methods and 1.6x over quantized ones, while also delivering 1.6x and 6.5x faster optimization, respectively, without degrading, and often improving, image fidelity. Code is available at this https URL.
>
---
#### [replaced 014] UniWeTok: An Unified Binary Tokenizer with Codebook Size $\mathit{2^{128}}$ for Unified Multimodal Large Language Model
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.14178](https://arxiv.org/pdf/2602.14178)**

> **作者:** Shaobin Zhuang; Yuang Ai; Jiaming Han; Weijia Mao; Xiaohui Li; Fangyikang Wang; Xiao Wang; Yan Li; Shanchuan Lin; Kun Xu; Zhenheng Yang; Huaibo Huang; Xiangyu Yue; Hao Chen; Yali Wang
>
> **备注:** 29 pages, 9 figures, 33 tables
>
> **摘要:** Unified Multimodal Large Language Models (MLLMs) require a visual representation that simultaneously supports high-fidelity reconstruction, complex semantic extraction, and generative suitability. However, existing visual tokenizers typically struggle to satisfy these conflicting objectives within a single framework. In this paper, we introduce UniWeTok, a unified discrete tokenizer designed to bridge this gap using a massive binary codebook ($\mathit{2^{128}}$). For training framework, we introduce Pre-Post Distillation and a Generative-Aware Prior to enhance the semantic extraction and generative prior of the discrete tokens. In terms of model architecture, we propose a convolution-attention hybrid architecture with the SigLu activation function. SigLu activation not only bounds the encoder output and stabilizes the semantic distillation process but also effectively addresses the optimization conflict between token entropy loss and commitment loss. We further propose a three-stage training framework designed to enhance UniWeTok's adaptability cross various image resolutions and perception-sensitive scenarios, such as those involving human faces and textual content. On ImageNet, UniWeTok achieves state-of-the-art image generation performance (FID: UniWeTok 1.38 vs. REPA 1.42) while requiring a remarkably low training compute (Training Tokens: UniWeTok 33B vs. REPA 262B). On general-domain, UniWeTok demonstrates highly competitive capabilities across a broad range of tasks, including multimodal understanding, image generation (DPG Score: UniWeTok 86.63 vs. FLUX.1 [Dev] 83.84), and editing (GEdit Overall Score: UniWeTok 5.09 vs. OmniGen 5.06). We release code and models to facilitate community exploration of unified tokenizer and MLLM.
>
---
#### [replaced 015] Ego: Embedding-Guided Personalization of Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.09771](https://arxiv.org/pdf/2603.09771)**

> **作者:** Soroush Seifi; Simon Gardier; Vaggelis Dorovatas; Daniel Olmeda Reino; Rahaf Aljundi
>
> **备注:** Accepted at the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** AI assistants that support humans in daily life are becoming increasingly feasible, driven by the rapid advancements in multimodal language models. A key challenge lies in overcoming the generic nature of these models to deliver personalized experiences. Existing approaches to personalizing large vision language models often rely on additional training stages, which limit generality and scalability, or on engineered pipelines with external pre-trained modules, which hinder deployment efficiency. In this work, we propose an efficient personalization method that leverages the model's inherent ability to capture personalized concepts. Specifically, we extract visual tokens that predominantly represent the target concept by utilizing the model's internal attention mechanisms. These tokens serve as a memory of that specific concept, enabling the model to recall and describe it when it appears in test images. We conduct a comprehensive and unified evaluation of our approach and SOTA methods across various personalization settings including single-concept, multi-concept, and video personalization, demonstrating strong performance gains with minimal personalization overhead.
>
---
#### [replaced 016] Sketch-Guided Stylized Landscape Cinemagraph Synthesis
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2412.00638](https://arxiv.org/pdf/2412.00638)**

> **作者:** Hao Jin; Hengyuan Chang; Xiaoxuan Xie; Zhengyang Wang; Xusheng Du; Shaojun Hu; Haoran Xie
>
> **备注:** 16 pages, 18 figures, accepted in Computer and Graphics
>
> **摘要:** Designing stylized cinemagraphs is challenging due to the difficulty in customizing complex and expressive flow elements. To achieve intuitive and detailed control of the generated cinemagraphs, sketches provide a feasible solution to convey personalized design requirements beyond text inputs. In this paper, we propose Sketch2Cinemagraph, a sketch-guided framework that enables the conditional generation of stylized cinemagraphs from freehand sketches. Sketch2Cinemagraph adopts text prompts for initial landscape generation and provides sketch controls for both spatial and motion cues. The latent diffusion model first generates target stylized landscape images along with realistic versions. Then, a pre-trained object detection model obtains masks for the flow regions. We propose a latent motion diffusion model to estimate motion field in fluid regions of the generated landscape images. The input motion sketches serve as the conditions to control the generated motion fields in the masked fluid regions with the prompt. To synthesize cinemagraph frames, the pixels within fluid regions are warped to target locations at each timestep using a U-Net based frame generator. The results verified that Sketch2Cinemagraph can generate aesthetically appealing stylized cinemagraphs with continuous temporal flow from sketch inputs. We showcase the advantages of Sketch2Cinemagraph through qualitative and quantitative comparisons against the state-of-the-art approaches.
>
---
#### [replaced 017] In Pursuit of Many: A Review of Modern Multiple Object Tracking Systems
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2209.04796](https://arxiv.org/pdf/2209.04796)**

> **作者:** Mk Bashar; Samia Islam; Kashifa Kawaakib Hussain; Md. Bakhtiar Hasan; A.B.M. Ashikur Rahman; Md. Hasanul Kabir
>
> **摘要:** Multiple Object Tracking (MOT) is a core capability in modern computer vision, essential to autonomous driving, surveillance, sports analytics, robotics, and biomedical imaging. Persistent identity assignment across frames remains challenging in real scenes because of occlusion, dense crowds, appearance ambiguity, scale variation, camera motion, and identity switching. In this survey we synthesize recent progress by organizing methods around the problems they target and the paradigms they adopt. We cover the historical progression from tracking-by-detection to hybrid and end-to-end designs, and we summarize major architectural directions including transformer-based trackers, generative/diffusion formulations, state-space predictors, Siamese and graph-based models, and the growing impact of foundation models for detection and representation. We review benchmark trends that motivate method design, documenting the shift from saturated pedestrian benchmarks to challenge-driven and domain-specific datasets and we analyze evaluation practice by comparing classic and newer motion- and safety-centric metrics. Finally, we connect algorithmic trends to practical deployment constraints and outline emerging directions, foundation-model integration, open-vocabulary and multimodal tracking, unified evaluation, and domain-adaptive methods, that we believe will shape MOT research and real-world adoption.
>
---
#### [replaced 018] X-WIN: Building Chest Radiograph World Model via Predictive Sensing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.14918](https://arxiv.org/pdf/2511.14918)**

> **作者:** Zefan Yang; Ge Wang; James Hendler; Mannudeep K. Kalra; Pingkun Yan
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Chest X-ray radiography (CXR) is an essential medical imaging technique for disease diagnosis. However, as 2D projectional images, CXRs are limited by structural superposition and hence fail to capture 3D anatomies. This limitation makes representation learning and disease diagnosis challenging. To address this challenge, we propose a novel CXR world model named X-WIN, which distills volumetric knowledge from chest computed tomography (CT) by learning to predict its 2D projections in latent space. The core idea is that a world model with internalized knowledge of 3D anatomical structure can predict CXRs under various transformations in 3D space. During projection prediction, we introduce an affinity-guided contrastive alignment loss that leverages mutual similarities to capture rich, correlated information across projections from the same volume. To improve model adaptability, we incorporate real CXRs into training through masked image modeling and employ a domain classifier to encourage statistically similar representations for real and simulated CXRs. Comprehensive experiments show that X-WIN outperforms existing foundation models on diverse downstream tasks using linear probing and few-shot fine-tuning. X-WIN also demonstrates the ability to render 2D projections for reconstructing a 3D CT volume.
>
---
#### [replaced 019] Is CLIP ideal? No. Can we fix it? Yes!
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2503.08723](https://arxiv.org/pdf/2503.08723)**

> **作者:** Raphi Kang; Yue Song; Georgia Gkioxari; Pietro Perona
>
> **备注:** ICCV 2025
>
> **摘要:** Contrastive Language-Image Pre-Training (CLIP) is a popular method for learning multimodal latent spaces with well-organized semantics. Despite its wide range of applications, CLIP's latent space is known to fail at handling complex visual-textual interactions. Recent works attempt to address its shortcomings with data-centric or algorithmic approaches. But what if the problem is more fundamental, and lies in the geometry of CLIP? Toward this end, we rigorously analyze CLIP's latent space properties, and prove that no CLIP-like joint embedding space exists which can correctly do any two of the following at the same time: 1. represent basic descriptions and image content, 2. represent attribute binding, 3. represent spatial location and relationships, 4. represent negation. Informed by this analysis, we propose Dense Cosine Similarity Maps (DCSMs) as a principled and interpretable scoring method for CLIP-like models, which solves the fundamental limitations of CLIP by retaining the semantic topology of the image patches and text tokens. This method improves upon the performance of classical CLIP-like joint encoder models on a wide array of benchmarks. We share our code and data here for reproducibility: this https URL
>
---
#### [replaced 020] DeepEyesV2: Toward Agentic Multimodal Model
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.05271](https://arxiv.org/pdf/2511.05271)**

> **作者:** Jack Hong; Chenxiao Zhao; ChengLin Zhu; Weiheng Lu; Guohai Xu; Xing Yu
>
> **备注:** Accepted to ICLR2026. Homepage: this https URL
>
> **摘要:** Agentic multimodal models should not only comprehend text and images, but also actively invoke external tools, such as code execution environments and web search, and integrate these operations into reasoning. In this work, we introduce DeepEyesV2 and explore how to build an agentic multimodal model from the perspectives of data construction, training methods, and model evaluation. We observe that direct reinforcement learning alone fails to induce robust tool-use behavior. This phenomenon motivates a two-stage training pipeline: a cold-start stage to establish tool-use patterns, and reinforcement learning stage to further refine tool invocation. We curate a diverse, moderately challenging training dataset, specifically including examples where tool use is beneficial. We further introduce RealX-Bench, a comprehensive benchmark designed to evaluate real-world multimodal reasoning, which inherently requires the integration of multiple capabilities, including perception, search, and reasoning. We evaluate DeepEyesV2 on RealX-Bench and other representative benchmarks, demonstrating its effectiveness across real-world understanding, mathematical reasoning, and search-intensive tasks. Moreover, DeepEyesV2 exhibits task-adaptive tool invocation, tending to use image operations for perception tasks and numerical computations for reasoning tasks. Reinforcement learning further enables complex tool combinations and allows model to selectively invoke tools based on context. We hope our study can provide guidance for community in developing agentic multimodal models.
>
---
#### [replaced 021] VIVID-Med: LLM-Supervised Structured Pretraining for Deployable Medical ViTs
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.09109](https://arxiv.org/pdf/2603.09109)**

> **作者:** Xiyao Wang; Xiaoyu Tan; Yang Dai; Yuxuan Fu; Shuo Li; Xihe Qiu
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Vision-language pretraining has driven significant progress in medical image analysis. However, current methods typically supervise visual encoders using one-hot labels or free-form text, neither of which effectively captures the complex semantic relationships among clinical findings. In this study, we introduce VIVID-Med, a novel framework that leverages a frozen large language model (LLM) as a structured semantic teacher to pretrain medical vision transformers (ViTs). VIVID-Med translates clinical findings into verifiable JSON field-state pairs via a Unified Medical Schema (UMS), utilizing answerability-aware masking to focus optimization. It then employs Structured Prediction Decomposition (SPD) to partition cross-attention into orthogonality-regularized query groups, extracting complementary visual aspects. Crucially, the LLM is discarded post-training, yielding a lightweight, deployable ViT-only backbone. We evaluated VIVID-Med across multiple settings: on CheXpert linear probing, it achieves a macro-AUC of 0.8588, outperforming BiomedCLIP by +6.65 points while using 500x less data. It also demonstrates robust zero-shot cross-domain transfer to NIH ChestX-ray14 (0.7225 macro-AUC) and strong cross-modality generalization to CT, achieving 0.8413 AUC on LIDC-IDRI lung nodule classification and 0.9969 macro-AUC on OrganAMNIST 11-organ classification. VIVID-Med offers a highly efficient, scalable alternative to deploying resource-heavy vision-language models in clinical settings.
>
---
#### [replaced 022] Enhanced Continual Learning of Vision-Language Models with Model Fusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.10705](https://arxiv.org/pdf/2503.10705)**

> **作者:** Haoyuan Gao; Zicong Zhang; Yuqi Wei; Linglan Zhao; Guilin Li; Yexin Li; Bo Wang; Linghe Kong; Weiran Huang
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** Vision-Language Models (VLMs) represent a significant breakthrough in artificial intelligence by integrating visual and textual modalities to achieve impressive zero-shot capabilities. However, VLMs are susceptible to catastrophic forgetting when sequentially fine-tuned on multiple downstream tasks. Existing continual learning methods for VLMs face various limitations, often relying on additional reference datasets, compromising zero-shot performance, or being restricted to parameter-efficient fine-tuning scenarios. In this paper, we propose a novel Continual Decoupling-Unifying (ConDU) approach that pioneers the use of model fusion for continual learning in VLMs. Specifically, ConDU maintains a unified model along with task triggers and prototype sets, employing an iterative process of decoupling task experts for previous tasks and unifying them with the task expert for the newly learned task. Additionally, we introduce an inference strategy for zero-shot scenarios by aggregating predictions from multiple decoupled task experts. Extensive experiments on the MTIL benchmark show that ConDU achieves up to a 2\% improvement in average performance across all seen tasks compared to state-of-the-art baselines, while also enhancing zero-shot capabilities relative to the original VLM. Our code is available at this https URL.
>
---
#### [replaced 023] Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.01957](https://arxiv.org/pdf/2507.01957)**

> **作者:** Zhuoyang Zhang; Luke J. Huang; Chengyue Wu; Shang Yang; Kelly Peng; Yao Lu; Song Han
>
> **备注:** ICLR 2026 Oral. The first two authors contributed equally to this work
>
> **摘要:** We present Locality-aware Parallel Decoding (LPD) to accelerate autoregressive image generation. Traditional autoregressive image generation relies on next-patch prediction, a memory-bound process that leads to high latency. Existing works have tried to parallelize next-patch prediction by shifting to multi-patch prediction to accelerate the process, but only achieved limited parallelization. To achieve high parallelization while maintaining generation quality, we introduce two key techniques: (1) Flexible Parallelized Autoregressive Modeling, a novel architecture that enables arbitrary generation ordering and degrees of parallelization. It uses learnable position query tokens to guide generation at target positions while ensuring mutual visibility among concurrently generated tokens for consistent parallel decoding. (2) Locality-aware Generation Ordering, a novel schedule that forms groups to minimize intra-group dependencies and maximize contextual support, enhancing generation quality. With these designs, we reduce the generation steps from 256 to 20 (256$\times$256 res.) and 1024 to 48 (512$\times$512 res.) without compromising quality on the ImageNet class-conditional generation, and achieving at least 3.4$\times$ lower latency than previous parallelized autoregressive models.
>
---
#### [replaced 024] Leveraging Spatial Context for Positive Pair Sampling in Histopathology Image Representation Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.05170](https://arxiv.org/pdf/2503.05170)**

> **作者:** Willmer Rafell Quinones Robles; Sakonporn Noree; Jongwoo Kim; Young Sin Ko; Bryan Wong; Mun Yong Yi
>
> **摘要:** Deep learning has shown strong potential in cancer classification from whole-slide images (WSIs), but the need for extensive expert annotations often limits its success. Annotation-free approaches, such as multiple instance learning (MIL) and self-supervised learning (SSL), have emerged as promising alternatives to traditional annotation-based methods. However, conventional SSL methods typically rely on synthetic data augmentations, which may fail to capture the spatial structure critical to histopathology. In this work, we propose a spatial context-driven positive pair sampling strategy that enhances SSL by leveraging the morphological coherence of spatially adjacent patches within WSIs. Our method is modular and compatible with established joint embedding SSL frameworks, including Barlow Twins, BYOL, VICReg, and DINOv2. We evaluate its effectiveness on both slide-level classification using MIL and patch-level linear probing. Experiments across four datasets demonstrate consistent performance improvements, with accuracy gains of 5\% to 10\% compared to standard augmentation-based sampling. These findings highlight the value of spatial context in improving representation learning for computational pathology and provide a biologically meaningful enhancement for pretraining models in annotation-limited settings. The code is available at this https URL.
>
---
#### [replaced 025] Prune Redundancy, Preserve Essence: Vision Token Compression in VLMs via Synergistic Importance-Diversity
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.09480](https://arxiv.org/pdf/2603.09480)**

> **作者:** Zhengyao Fang; Pengyuan Lyu; Chengquan Zhang; Guangming Lu; Jun Yu; Wenjie Pei
>
> **备注:** accepted by ICLR2026
>
> **摘要:** Vision-language models (VLMs) face significant computational inefficiencies caused by excessive generation of visual tokens. While prior work shows that a large fraction of visual tokens are redundant, existing compression methods struggle to balance importance preservation and information diversity. To address this, we propose PruneSID, a training-free Synergistic Importance-Diversity approach featuring a two-stage pipeline: (1) Principal Semantic Components Analysis (PSCA) for clustering tokens into semantically coherent groups, ensuring comprehensive concept coverage, and (2) Intra-group Non-Maximum Suppression (NMS) for pruning redundant tokens while preserving key representative tokens within each group. Additionally, PruneSID incorporates an information-aware dynamic compression ratio mechanism that optimizes token compression rates based on image complexity, enabling more effective average information preservation across diverse scenes. Extensive experiments demonstrate state-of-the-art performance, achieving 96.3% accuracy on LLaVA-1.5 with only 11.1% token retention, and 92.8% accuracy at extreme compression rates (5.6%) on LLaVA-NeXT, outperforming prior methods by 2.5% with 7.8 $\times$ faster prefilling speed compared to the original model. Our framework generalizes across diverse VLMs and both image and video modalities, showcasing strong cross-modal versatility. Code is available at this https URL.
>
---
#### [replaced 026] An Overview about Emerging Technologies of Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2306.13302](https://arxiv.org/pdf/2306.13302)**

> **作者:** Yu Huang; Yue Chen; Zijiang Yang
>
> **摘要:** Since DARPA started Grand Challenges in 2004 and Urban Challenges in 2007, autonomous driving has been the most active field of AI applications. This paper gives an overview about technical aspects of autonomous driving technologies and open problems. We investigate the major fields of self-driving systems, such as perception, mapping and localization, prediction, planning and control, simulation, V2X and safety etc. Especially we elaborate on all these issues in a framework of data closed loop, a popular platform to solve the long tailed autonomous driving problems.
>
---
#### [replaced 027] Enhancing Tree Species Classification: Insights from YOLOv8 and Explainable AI Applied to TLS Point Cloud Projections
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.16950](https://arxiv.org/pdf/2512.16950)**

> **作者:** Adrian Straker; Paul Magdon; Marco Zullich; Maximilian Freudenberg; Christoph Kleinn; Johannes Breidenbach; Stefano Puliti; Nils Noelke
>
> **备注:** 34 pages, 17 figures, submitted to Forestry: An International Journal of Forest Research
>
> **摘要:** Aiming to advance research in the field of interpretability of deep learning models for tree species classification using TLS 3D point clouds we present insights in the classification abilities of YOLOv8 through a new framework which enables systematic analysis of saliency maps derived from CAM (Class Activation Mapping). To investigate the contribution of structural tree features to the classification decisions of the models, we link regions with high saliency derived from the application of Finer-CAM to segments of 2D side-view images that correspond to structural tree features. Using TLS 3D point clouds from 2445 trees across seven European tree species, we trained five YOLOv8 models with cross-validation, reaching a mean accuracy of 96% (SD = 0.24%) when applied to the test data. Our results demonstrate that Finer-CAM can be considered faithful in identifying discriminative regions that discriminate target tree species. This renders Finer-CAM suitable for enhancing the interpretability of the tree species classification models. Analysis of 630 saliency maps indicate that the models primarily rely on image regions associated with tree crowns for species classification. While this result is pronounced in Silver Birch, European Beech, English oak, and Norway Spruce, image regions associated with stems contribute more frequently to the differentiation of European ash, Scots pine, and Douglas-fir. We demonstrate that the visibility of detailed structural tree features in the 2D side-view images enhances the discriminative performances of the models, indicating YOLOv8`s abilities to leverage detailed point cloud representations. Our results represent a first step toward enhancing the understanding of the classification decision processes of tree species classification models, aiding in the identification of data set and model limitations, and building confidence in model predictions.
>
---
#### [replaced 028] D-GAP: Improving Out-of-Domain Robustness via Dataset-Agnostic and Gradient-Guided Augmentation in Frequency and Pixel Spaces
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.11286](https://arxiv.org/pdf/2511.11286)**

> **作者:** Ruoqi Wang; Haitao Wang; Shaojie Guo; Qiong Luo
>
> **摘要:** Out-of-domain (OOD) robustness is challenging to achieve in real-world computer vision applications, where shifts in image background, style, and acquisition instruments always degrade model performance. Generic augmentations show inconsistent gains under such shifts, whereas dataset-specific augmentations require expert knowledge and prior analysis. Moreover, prior studies show that neural networks adapt poorly to domain shifts because they exhibit a learning bias to domain-specific frequency components. Perturbing frequency values can mitigate such bias but overlooks pixel-level details, leading to suboptimal performance. To address these problems, we propose D-GAP, a Dataset-agnostic and Gradient-guided augmentation method for the Amplitude spectrum (in frequency space) and the Pixel values, improving OOD robustness by introducing targeted augmentation in both frequency and pixel spaces. Unlike conventional handcrafted augmentations, D-GAP computes sensitivity maps in the frequency space from task gradients, which reflect how strongly the deep models respond to different frequency components, and uses the maps to adaptively interpolate amplitudes between source and target samples. This way, D-GAP reduces the learning bias in frequency space, while a complementary pixel-space blending procedure restores fine spatial details. Extensive experiments on four real-world datasets and three domain-adaptation benchmarks show that D-GAP consistently outperforms both generic and dataset-specific domain adaptation methods, improving average OOD performance by +5.3% on real-world datasets and +1.9% on benchmark datasets.
>
---
#### [replaced 029] Adaptive Event Stream Slicing for Open-Vocabulary Event-Based Object Detection via Vision-Language Knowledge Distillation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.00681](https://arxiv.org/pdf/2510.00681)**

> **作者:** Jinchang Zhang; Zijun Li; Jiakai Lin; Guoyu Lu
>
> **摘要:** Event cameras offer advantages in object detection tasks due to high-speed response, low latency, and robustness to motion blur. However, event cameras lack texture and color information, making open-vocabulary detection particularly challenging. Current event-based detection methods are typically trained on predefined categories, limiting their ability to generalize to novel objects, where encountering previously unseen objects is common. Vision-language models (VLMs) have enabled open-vocabulary object detection in RGB images. However, the modality gap between images and event streams makes it ineffective to directly transfer CLIP to event data, as CLIP was not designed for event streams. To bridge this gap, we propose an event-image knowledge distillation framework that leverages CLIP's semantic understanding to achieve open-vocabulary object detection on event data. Instead of training CLIP directly on event streams, we use image frames as inputs to a teacher model, guiding the event-based student model to learn CLIP's rich visual representations. Through spatial attention-based distillation, the student network learns meaningful visual features directly from raw event inputs while inheriting CLIP's broad visual knowledge. Furthermore, to prevent information loss due to event data segmentation, we design a hybrid spiking neural network (SNN) and convolutional neural network (CNN) framework. Unlike fixed-group event segmentation methods, which often discard crucial temporal information, our SNN adaptively determines the optimal event segmentation moments, ensuring that key temporal features are extracted. The extracted event features are then processed by CNNs for object detection.
>
---
#### [replaced 030] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.08224](https://arxiv.org/pdf/2603.08224)**

> **作者:** Ruixiang Zhao; Zhihao Xu; Bangxiang Lan; Zijie Xin; Jingyu Liu; Xirong Li
>
> **备注:** Accepted to CVPR2026
>
> **摘要:** For video-text retrieval, the use of CLIP has been a de facto choice. Since CLIP provides only image and text encoders, this consensus has led to a biased paradigm that entirely ignores the sound track of videos. While several attempts have been made to reintroduce audio -- typically by incorporating an audio encoder and fusing its output with visual features -- these methods face two challenges: ineffective representation of speech content and suboptimal vision-audio fusion. To address these issues jointly, we propose SAVE, a Speech Aware Video rEpresentation learning method. SAVE improves upon AVIGATE, a SOTA audiovisual method, with a dedicated speech branch for more effective speech embedding. Furthermore, we introduce soft-ALBEF for early vision-audio alignment that facilitates fusion. Extensive experiments on five benchmarks show that SAVE compares favorably against the SOTA, outperforming AVIGATE by +4.1% on MSRVTT-9k, +1.9% on MSRVTT-7k, +2.5% on VATEX, +9.8% on Charades, and +2.1% on LSMDC, in light of the SumR metric.
>
---
#### [replaced 031] SVBench: Evaluation of Video Generation Models on Social Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.21507](https://arxiv.org/pdf/2512.21507)**

> **作者:** Wenshuo Peng; Gongxuan Wang; Tianmeng Yang; Chuanhao Li; Xiaojie Xu; Hui He; Kaipeng Zhang
>
> **备注:** 10pages
>
> **摘要:** Recent text-to-video generation models have made remarkable progress in visual realism, motion fidelity, and text-video alignment, yet they still struggle to produce socially coherent behavior. Unlike humans, who readily infer intentions, beliefs, emotions, and social norms from brief visual cues, current models often generate literal scenes without capturing the underlying causal and psychological dynamics. To systematically assess this limitation, we introduce the first benchmark for social reasoning in video generation. Grounded in developmental and social psychology, the benchmark covers thirty classic social cognition paradigms spanning seven core dimensions: mental-state inference, goal-directed action, joint attention, social coordination, prosocial behavior, social norms, and multi-agent strategy. To operationalize these paradigms, we build a fully training-free agent-based pipeline that distills the reasoning structure of each paradigm, synthesizes diverse video-ready scenarios, enforces conceptual neutrality and difficulty control through cue-based critique, and evaluates generated videos with a high-capacity VLM judge along five interpretable dimensions of social reasoning. Using this framework, we conduct the first large-scale evaluation of seven state-of-the-art video generation systems. Results show a clear gap between surface-level plausibility and deeper social reasoning, suggesting that current models remain limited in their ability to generate socially grounded behavior. this https URL
>
---
#### [replaced 032] DSER: Spectral Epipolar Representation for Efficient Light Field Depth Estimation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.08900](https://arxiv.org/pdf/2508.08900)**

> **作者:** Noor Islam S. Mohammad; Md Muntaqim Meherab
>
> **摘要:** Dense light field depth estimation remains challenging due to sparse angular sampling, occlusion boundaries, textureless regions, and the cost of exhaustive multi-view matching. We propose \emph{Deep Spectral Epipolar Representation} (DSER), a geometry-aware framework that introduces spectral regularization in the epipolar domain for dense disparity reconstruction. DSER models frequency-consistent EPI structure to constrain correspondence estimation and couples this prior with a hybrid inference pipeline that combines least squares gradient initialization, plane-sweeping cost aggregation, and multiscale EPI refinement. An occlusion-aware directed random walk further propagates reliable disparity along edge-consistent paths, improving boundary sharpness and weak-texture stability. Experiments on benchmark and real-world light field datasets show that DSER achieves a strong accuracy-efficiency trade-off, producing more structurally consistent depth maps than representative classical and hybrid baselines. These results establish spectral epipolar regularization as an effective inductive bias for scalable and noise-robust light field depth estimation.
>
---
#### [replaced 033] vS-Graphs: Tightly Coupling Visual SLAM and 3D Scene Graphs Exploiting Hierarchical Scene Understanding
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉SLAM任务，旨在解决传统系统地图语义不足的问题。通过融合视觉场景理解和3D场景图，提升地图的语义丰富性和可理解性。**

- **链接: [https://arxiv.org/pdf/2503.01783](https://arxiv.org/pdf/2503.01783)**

> **作者:** Ali Tourani; Saad Ejaz; Hriday Bavle; Miguel Fernandez-Cortizas; David Morilla-Cabello; Jose Luis Sanchez-Lopez; Holger Voos
>
> **备注:** 20 pages, 10 figures, 5 tables
>
> **摘要:** Current Visual Simultaneous Localization and Mapping (VSLAM) systems often struggle to create maps that are both semantically rich and easily interpretable. While incorporating semantic scene knowledge aids in building richer maps with contextual associations among mapped objects, representing them in structured formats, such as scene graphs, has not been widely addressed, resulting in complex map comprehension and limited scalability. This paper introduces vS-Graphs, a novel real-time VSLAM framework that integrates vision-based scene understanding with map reconstruction and comprehensible graph-based representation. The framework infers structural elements (i.e., rooms and floors) from detected building components (i.e., walls and ground surfaces) and incorporates them into optimizable 3D scene graphs. This solution enhances the reconstructed map's semantic richness, comprehensibility, and localization accuracy. Extensive experiments on standard benchmarks and real-world datasets demonstrate that vS-Graphs achieves an average of 15.22% accuracy gain across all tested datasets compared to state-of-the-art VSLAM methods. Furthermore, the proposed framework achieves environment-driven semantic entity detection accuracy comparable to that of precise LiDAR-based frameworks, using only visual features. The code is publicly available at this https URL and is actively being improved. Moreover, a web page containing more media and evaluation outcomes is available on this https URL.
>
---
#### [replaced 034] Don't Mind the Gaps: Implicit Neural Representations for Resolution-Agnostic Retinal OCT Analysis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.02447](https://arxiv.org/pdf/2601.02447)**

> **作者:** Bennet Kahrs; Julia Andresen; Fenja Falta; Monty Santarossa; Heinz Handels; Timo Kepp
>
> **备注:** MELBA-BVM 2025 Special Issue. Extended journal version of the paper "Bridging Gaps in Retinal Imaging" presented at the German Conference on Medical Image Computing - BVM2025 (DOI:https://doi.org/10.1007/978-3-658-47422-5_24)
>
> **摘要:** Routine clinical imaging of the retina using optical coherence tomography (OCT) is performed with large slice spacing, resulting in highly anisotropic images and a sparsely scanned retina. Most learning-based methods circumvent the problems arising from the anisotropy by using 2D approaches rather than performing volumetric analyses. These approaches inherently bear the risk of generating inconsistent results for neighboring B-scans. For example, 2D retinal layer segmentations can have irregular surfaces in 3D. Furthermore, the typically used convolutional neural networks are bound to the resolution of the training data, which prevents their usage for images acquired with a different imaging protocol. Implicit neural representations (INRs) have recently emerged as a tool to store voxelized data as a continuous representation. Using coordinates as input, INRs are resolution-agnostic, which allows them to be applied to anisotropic data. In this paper, we propose two frameworks that make use of this characteristic of INRs for dense 3D analyses of retinal OCT volumes. 1) We perform inter-B-scan interpolation by incorporating additional information from en-face modalities, that help retain relevant structures between B-scans. 2) We create a resolution-agnostic retinal atlas that enables general analysis without strict requirements for the data. Both methods leverage generalizable INRs, improving retinal shape representation through population-based training and allowing predictions for unseen cases. Our resolution-independent frameworks facilitate the analysis of OCT images with large B-scan distances, opening up possibilities for the volumetric evaluation of retinal structures and pathologies.
>
---
#### [replaced 035] A Survey on Interpretability in Visual Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.11099](https://arxiv.org/pdf/2507.11099)**

> **作者:** Qiyang Wan; Chengzhi Gao; Ruiping Wang; Xilin Chen
>
> **备注:** 20 pages, 8 figures, 7 tables. Accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
>
> **摘要:** Visual recognition models have achieved unprecedented success in various tasks. While researchers aim to understand the underlying mechanisms of these models, the growing demand for deployment in safety-critical areas like autonomous driving and medical diagnostics has accelerated the development of eXplainable AI (XAI). Distinct from generic XAI, visual recognition XAI is positioned at the intersection of vision and language, which represent the two most fundamental human modalities and form the cornerstones of multimodal intelligence. This paper provides a systematic survey of XAI in visual recognition by establishing a multi-dimensional taxonomy from a human-centered perspective based on intent, object, presentation, and methodology. Beyond categorization, we summarize critical evaluation desiderata and metrics, conducting an extensive qualitative assessment across different categories and demonstrating quantitative benchmarks within specific dimensions. Furthermore, we explore the interpretability of Multimodal Large Language Models and practical applications, identifying emerging trends and opportunities. By synthesizing these diverse perspectives, this survey provides an insightful roadmap to inspire future research on the interpretability of visual recognition models.
>
---
#### [replaced 036] ZACH-ViT: Regime-Dependent Inductive Bias in Compact Vision Transformers for Medical Imaging
- **分类: cs.CV; cs.LG; eess.IV**

- **链接: [https://arxiv.org/pdf/2602.17929](https://arxiv.org/pdf/2602.17929)**

> **作者:** Athanasios Angelakis
>
> **备注:** 24 pages, 15 figures, 5 tables. Code and models available at this https URL
>
> **摘要:** Vision Transformers rely on positional embeddings and class tokens encoding fixed spatial priors. While effective for natural images, these priors may be suboptimal when spatial layout is weakly informative, a frequent condition in medical imaging. We introduce ZACH-ViT (Zero-token Adaptive Compact Hierarchical Vision Transformer), a compact Vision Transformer that removes positional embeddings and the [CLS] token, achieving permutation-invariant patch processing via global average pooling. Zero-token denotes removal of the dedicated aggregation token and positional encodings. Patch tokens remain unchanged. Adaptive residual projections preserve training stability under strict parameter constraints. We evaluate ZACH-ViT across seven MedMNIST datasets under a strict few-shot protocol (50 samples/class, fixed hyperparameters, five seeds). Results reveal regime-dependent behavior: ZACH-ViT (0.25M parameters, trained from scratch) achieves strongest advantage on BloodMNIST and remains competitive on PathMNIST, while relative advantage decreases on datasets with stronger anatomical priors (OCTMNIST, OrganAMNIST), consistent with our hypothesis. Component and pooling ablations show positional support becomes mildly beneficial as spatial structure increases, whereas reintroducing a [CLS] token is consistently unfavorable. These findings support that architectural alignment with data structure can outweigh universal benchmark dominance. Despite minimal size and no pretraining, ZACH-ViT achieves competitive performance under data-scarce conditions, relevant for compact medical imaging and low-resource settings. Code: this https URL
>
---
#### [replaced 037] MVCustom: Multi-View Customized Diffusion via Geometric Latent Rendering and Completion
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.13702](https://arxiv.org/pdf/2510.13702)**

> **作者:** Minjung Shin; Hyunin Cho; Sooyeon Go; Jin-Hwa Kim; Youngjung Uh
>
> **备注:** ICLR 2026, Project page: this https URL
>
> **摘要:** Multi-view generation with camera pose control and prompt-based customization are both essential elements for achieving controllable generative models. However, existing multi-view generation models do not support customization with geometric consistency, whereas customization models lack explicit viewpoint control, making them challenging to unify. Motivated by these gaps, we introduce a novel task, multi-view customization, which aims to jointly achieve multi-view camera pose control and customization. Due to the scarcity of training data in customization, existing multi-view generation models, which inherently rely on large-scale datasets, struggle to generalize to diverse prompts. To address this, we propose MVCustom, a novel diffusion-based framework explicitly designed to achieve both multi-view consistency and customization fidelity. In the training stage, MVCustom learns the subject's identity and geometry using a feature-field representation, incorporating the text-to-video diffusion backbone enhanced with dense spatio-temporal attention, which leverages temporal coherence for multi-view consistency. In the inference stage, we introduce two novel techniques: depth-aware feature rendering explicitly enforces geometric consistency, and consistent-aware latent completion ensures accurate perspective alignment of the customized subject and surrounding backgrounds. Extensive experiments demonstrate that MVCustom achieves the most balanced and consistent competitive performance across multi-view consistency, customization fidelity, demonstrating effective solution of multi-objective generation task.
>
---
#### [replaced 038] Streaming Autoregressive Video Generation via Diagonal Distillation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.09488](https://arxiv.org/pdf/2603.09488)**

> **作者:** Jinxiu Liu; Xuanming Liu; Kangfu Mei; Yandong Wen; Ming-Hsuan Yang; Weiyang Liu
>
> **备注:** ICLR 2026 (31 pages, 10 figures, project page: this https URL)
>
> **摘要:** Large pretrained diffusion models have significantly enhanced the quality of generated videos, and yet their use in real-time streaming remains limited. Autoregressive models offer a natural framework for sequential frame synthesis but require heavy computation to achieve high fidelity. Diffusion distillation can compress these models into efficient few-step variants, but existing video distillation approaches largely adapt image-specific methods that neglect temporal dependencies. These techniques often excel in image generation but underperform in video synthesis, exhibiting reduced motion coherence, error accumulation over long sequences, and a latency-quality trade-off. We identify two factors that result in these limitations: insufficient utilization of temporal context during step reduction and implicit prediction of subsequent noise levels in next-chunk prediction (i.e., exposure bias). To address these issues, we propose Diagonal Distillation, which operates orthogonally to existing approaches and better exploits temporal information across both video chunks and denoising steps. Central to our approach is an asymmetric generation strategy: more steps early, fewer steps later. This design allows later chunks to inherit rich appearance information from thoroughly processed early chunks, while using partially denoised chunks as conditional inputs for subsequent synthesis. By aligning the implicit prediction of subsequent noise levels during chunk generation with the actual inference conditions, our approach mitigates error propagation and reduces oversaturation in long-range sequences. We further incorporate implicit optical flow modeling to preserve motion quality under strict step constraints. Our method generates a 5-second video in 2.61 seconds (up to 31 FPS), achieving a 277.3x speedup over the undistilled model.
>
---
#### [replaced 039] CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.03281](https://arxiv.org/pdf/2603.03281)**

> **作者:** Hanyang Wang; Yiyang Liu; Jiawei Chi; Fangfu Liu; Ran Xue; Yueqi Duan
>
> **备注:** Accepted by CVPR 2026; Project Page: this https URL
>
> **摘要:** Classifier-Free Guidance (CFG) has emerged as a central approach for enhancing semantic alignment in flow-based diffusion models. In this paper, we explore a unified framework called CFG-Ctrl, which reinterprets CFG as a control applied to the first-order continuous-time generative flow, using the conditional-unconditional discrepancy as an error signal to adjust the velocity field. From this perspective, we summarize vanilla CFG as a proportional controller (P-control) with fixed gain, and typical follow-up variants develop extended control-law designs derived from it. However, existing methods mainly rely on linear control, inherently leading to instability, overshooting, and degraded semantic fidelity especially on large guidance scales. To address this, we introduce Sliding Mode Control CFG (SMC-CFG), which enforces the generative flow toward a rapidly convergent sliding manifold. Specifically, we define an exponential sliding mode surface over the semantic prediction error and introduce a switching control term to establish nonlinear feedback-guided correction. Moreover, we provide a Lyapunov stability analysis to theoretically support finite-time convergence. Experiments across text-to-image generation models including Stable Diffusion 3.5, Flux, and Qwen-Image demonstrate that SMC-CFG outperforms standard CFG in semantic alignment and enhances robustness across a wide range of guidance scales. Project Page: this https URL
>
---
#### [replaced 040] BrandFusion: A Multi-Agent Framework for Seamless Brand Integration in Text-to-Video Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.02816](https://arxiv.org/pdf/2603.02816)**

> **作者:** Zihao Zhu; Ruotong Wang; Siwei Lyu; Min Zhang; Baoyuan Wu
>
> **摘要:** The rapid advancement of text-to-video (T2V) models has revolutionized content creation, yet their commercial potential remains largely untapped. We introduce, for the first time, the task of seamless brand integration in T2V: automatically embedding advertiser brands into prompt-generated videos while preserving semantic fidelity to user intent. This task confronts three core challenges: maintaining prompt fidelity, ensuring brand recognizability, and achieving contextually natural integration. To address them, we propose BrandFusion, a novel multi-agent framework comprising two synergistic phases. In the offline phase (advertiser-facing), we construct a Brand Knowledge Base by probing model priors and adapting to novel brands via lightweight fine-tuning. In the online phase (user-facing), five agents jointly refine user prompts through iterative refinement, leveraging the shared knowledge base and real-time contextual tracking to ensure brand visibility and semantic alignment. Experiments on 18 established and 2 custom brands across multiple state-of-the-art T2V models demonstrate that BrandFusion significantly outperforms baselines in semantic preservation, brand recognizability, and integration naturalness. Human evaluations further confirm higher user satisfaction, establishing a practical pathway for sustainable T2V monetization.
>
---
#### [replaced 041] Pixel Motion Diffusion is What We Need for Robot Control
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DAWN框架，用于机器人控制任务，解决如何将高层指令转化为低层动作的问题。通过扩散模型和像素运动表示，实现端到端学习与可靠现实迁移。**

- **链接: [https://arxiv.org/pdf/2509.22652](https://arxiv.org/pdf/2509.22652)**

> **作者:** E-Ro Nguyen; Yichi Zhang; Kanchana Ranasinghe; Xiang Li; Michael S. Ryoo
>
> **备注:** Accepted to CVPR 2026. Project page: this https URL
>
> **摘要:** We present DAWN (Diffusion is All We Need for robot control), a unified diffusion-based framework for language-conditioned robotic manipulation that bridges high-level motion intent and low-level robot action via structured pixel motion representation. In DAWN, both the high-level and low-level controllers are modeled as diffusion processes, yielding a fully trainable, end-to-end system with interpretable intermediate motion abstractions. DAWN achieves state-of-the-art results on the challenging CALVIN benchmark, demonstrating strong multi-task performance, and further validates its effectiveness on MetaWorld. Despite the substantial domain gap between simulation and reality and limited real-world data, we demonstrate reliable real-world transfer with only minimal finetuning, illustrating the practical viability of diffusion-based motion abstractions for robotic control. Our results show the effectiveness of combining diffusion modeling with motion-centric representations as a strong baseline for scalable and robust robot learning. Project page: this https URL
>
---
#### [replaced 042] No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.19248](https://arxiv.org/pdf/2602.19248)**

> **作者:** Zunkai Dai; Ke Li; Jiajia Liu; Jie Yang; Yuanyuan Qiao
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** The collection and detection of video anomaly data has long been a challenging problem due to its rare occurrence and spatio-temporal scarcity. Existing video anomaly detection (VAD) methods under perform in open-world scenarios. Key contributing factors include limited dataset diversity, and inadequate understanding of context-dependent anomalous semantics. To address these issues, i) we propose LAVIDA, an end-to-end zero-shot video anomaly detection framework. ii) LAVIDA employs an Anomaly Exposure Sampler that transforms segmented objects into pseudo-anomalies to enhance model adaptability to unseen anomaly categories. It further integrates a Multimodal Large Language Model (MLLM) to bolster semantic comprehension capabilities. Additionally, iii) we design a token compression approach based on reverse attention to handle the spatio-temporal scarcity of anomalous patterns and decrease computational cost. The training process is conducted solely on pseudo anomalies without any VAD data. Evaluations across four benchmark VAD datasets demonstrate that LAVIDA achieves SOTA performance in both frame-level and pixel-level anomaly detection under the zero-shot setting. Our code is available in this https URL.
>
---
#### [replaced 043] PLANING: A Loosely Coupled Triangle-Gaussian Framework for Streaming 3D Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.22046](https://arxiv.org/pdf/2601.22046)**

> **作者:** Changjian Jiang; Kerui Ren; Xudong Li; Kaiwen Song; Guanghao Li; Linning Xu; Tao Lu; Junting Dong; Yu Zhang; Bo Dai; Mulin Yu
>
> **备注:** Project page: this https URL
>
> **摘要:** Streaming reconstruction from monocular image sequences remains challenging, as existing methods typically favor either high-quality rendering or accurate geometry, but rarely both. We present PLANING, an efficient on-the-fly reconstruction framework built on a hybrid representation that loosely couples explicit geometric primitives with neural Gaussians, enabling geometry and appearance to be modeled in a decoupled manner. This decoupling supports an online initialization and optimization strategy that separates geometry and appearance updates, yielding stable streaming reconstruction with substantially reduced structural redundancy. PLANING improves dense mesh Chamfer-L2 by 18.52% over PGSR, surpasses ARTDECO by 1.31 dB PSNR, and reconstructs ScanNetV2 scenes in under 100 seconds, over 5x faster than 2D Gaussian Splatting, while matching the quality of offline per-scene optimization. Beyond reconstruction quality, the structural clarity and computational efficiency of PLANING make it well suited for a broad range of downstream applications, such as enabling large-scale scene modeling and simulation-ready environments for embodied AI. Project page: this https URL .
>
---
#### [replaced 044] InstantSfM: Towards GPU-Native SfM for the Deep Learning Era
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.13310](https://arxiv.org/pdf/2510.13310)**

> **作者:** Jiankun Zhong; Zitong Zhan; Quankai Gao; Ziyu Chen; Haozhe Lou; Jiageng Mao; Ulrich Neumann; Chen Wang; Yue Wang
>
> **摘要:** Structure-from-Motion (SfM) is a fundamental technique for recovering camera poses and scene structure from multi-view imagery, serving as a critical upstream component for applications ranging from 3D reconstruction to modern neural scene representations such as 3D Gaussian Splatting. However, most mature SfM systems remain CPU-centric and built upon traditional optimization toolchains, creating a growing mismatch with modern GPU-based, learning-driven pipelines and limiting scalability in large-scale scenes. While recent advances in GPU-accelerated bundle adjustment (BA) have demonstrated the potential of parallel sparse optimization, extending these techniques to build a complete global SfM system remains challenging due to unresolved issues in metric scale recovery and numerical robustness. In this paper, we implement a fully GPU-based and PyTorch-compatible global SfM system, named InstantSfM, to integrate seamlessly with modern learning pipelines. InstantSfM embeds metric depth priors directly into both global positioning and BA through a depth-constrained Jacobian structure, thereby resolving scale ambiguity within the optimization framework. To ensure numerical stability, we employ explicit filtering of under-constrained variables for the Jacobian matrix in an optimized GPU-friendly manner. Extensive experiments on diverse datasets demonstrate that InstantSfM achieves state-of-the-art efficiency while maintaining reconstruction accuracy comparable to both established classical pipelines and recent learning-based methods, showing up to ${\sim40\times}$ speedup over COLMAP on large-scale scenes.
>
---
#### [replaced 045] Cosmos-H-Surgical: Learning Surgical Robot Policies from Videos via World Modeling
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于手术机器人领域，旨在解决数据稀缺问题。通过构建世界模型和合成数据，提升手术机器人自主学习能力。**

- **链接: [https://arxiv.org/pdf/2512.23162](https://arxiv.org/pdf/2512.23162)**

> **作者:** Yufan He; Pengfei Guo; Mengya Xu; Zhaoshuo Li; Andriy Myronenko; Dillan Imans; Bingjie Liu; Dongren Yang; Mingxue Gu; Yongnan Ji; Yueming Jin; Ren Zhao; Baiyong Shen; Daguang Xu
>
> **摘要:** Data scarcity remains a fundamental barrier to achieving fully autonomous surgical robots. While large scale vision language action (VLA) models have shown impressive generalization in household and industrial manipulation by leveraging paired video action data from diverse domains, surgical robotics suffers from the paucity of datasets that include both visual observations and accurate robot kinematics. In contrast, vast corpora of surgical videos exist, but they lack corresponding action labels, preventing direct application of imitation learning or VLA training. In this work, we aim to alleviate this problem by learning policy models from Cosmos-H-Surgical, a world model designed for surgical physical AI. We curated the Surgical Action Text Alignment (SATA) dataset with detailed action description specifically for surgical robots. Then we built Cosmos-H-Surgical based on the most advanced physical AI world model and SATA. It's able to generate diverse, generalizable and realistic surgery videos. We are also the first to use an inverse dynamics model to infer pseudokinematics from synthetic surgical videos, producing synthetic paired video action data. We demonstrate that a surgical VLA policy trained with these augmented data significantly outperforms models trained only on real demonstrations on a real surgical robot platform. Our approach offers a scalable path toward autonomous surgical skill acquisition by leveraging the abundance of unlabeled surgical video and generative world modeling, thus opening the door to generalizable and data efficient surgical robot policies.
>
---
#### [replaced 046] TikArt: Stabilizing Aperture-Guided Fine-Grained Visual Reasoning with Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.14482](https://arxiv.org/pdf/2602.14482)**

> **作者:** Hao Ding; Zhichuan Yang; Weijie Ge; Ziqin Gao; Chaoyi Lu; Lei Zhao
>
> **摘要:** Fine-grained visual reasoning in multimodal large language models (MLLMs) is bottlenecked by single-pass global image encoding: key evidence often lies in tiny objects, cluttered regions, subtle markings, or dense charts. We present \textbf{TikArt} (\textbf{T}h\textbf{i}n\textbf{k}ing \textbf{A}pe\textbf{rt}ure), an aperture-guided agent that formulates multimodal reasoning as sequential evidence acquisition over regions of interest. TikArt follows a Think--Aperture--Observe (TAO) loop that interleaves language reasoning with two aperture actions: Zoom, which extracts rectangular crops, and Segment, which invokes an off-the-shelf segmenter to produce object-centric mask-based views for irregular targets. A mandatory Observation step after every aperture action writes local evidence back into text, yielding interpretable aperture trajectories and persistent linguistic memory. Built on Qwen3-VL-8B, TikArt is trained with GRPO-style reinforcement learning under a two-stage curriculum. To stabilize long-horizon tool-integrated learning, we introduce Relative Uncertainty Reduction (RUR), a dense reward computed by a frozen evaluator that favors evidence-building trajectories and mitigates degenerate tool use. Experiments on high-resolution reasoning, general multimodal understanding, and both referring and reasoning-oriented segmentation show consistent gains over the backbone, demonstrating that aperture-guided observation improves fine-grained visual reasoning and transfers naturally to pixel-level grounding.
>
---
#### [replaced 047] Agentic AI as a Network Control-Plane Intelligence Layer for Federated Learning over 6G
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.09141](https://arxiv.org/pdf/2603.09141)**

> **作者:** Loc X. Nguyen; Ji Su Yoon; Huy Q. Le; Yu Qiao; Avi Deb Raha; Eui-Nam Huh; Nguyen H. Tran; Zhu Han; Choong Seon Hong
>
> **摘要:** The shift toward user-customized on-device learning places new demands on wireless systems: models must be trained on diverse, distributed data while meeting strict latency, bandwidth, and reliability constraints. To address this, we propose an Agentic AI as the control layer for managing federated learning (FL) over 6G networks, which translates high-level task goals into actions that are aware of network conditions. Rather than simply viewing FL as a learning challenge, our system sees it as a combined task of learning and network management. A set of specialized agents focused on retrieval, planning, coding, and evaluation utilizes monitoring tools and optimization methods to handle client selection, incentive structuring, scheduling, resource allocation, adaptive local training, and code generation. The use of closed-loop evaluation and memory allows the system to consistently refine its decisions, taking into account varying signal-to-noise ratios, bandwidth conditions, and device capabilities. Finally, our case study has demonstrated the effectiveness of the Agentic AI system's use of tools for achieving high performance.
>
---
#### [replaced 048] SIMSPINE: A Biomechanics-Aware Simulation Framework for 3D Spine Motion Annotation and Benchmarking
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.20792](https://arxiv.org/pdf/2602.20792)**

> **作者:** Muhammad Saif Ullah Khan; Didier Stricker
>
> **备注:** Camera-ready version
>
> **摘要:** Modeling spinal motion is fundamental to understanding human biomechanics, yet remains underexplored in computer vision due to the spine's complex multi-joint kinematics and the lack of large-scale 3D annotations. We present a biomechanics-aware keypoint simulation framework that augments existing human pose datasets with anatomically consistent 3D spinal keypoints derived from musculoskeletal modeling. Using this framework, we create the first open dataset, named SIMSPINE, which provides sparse vertebra-level 3D spinal annotations for natural full-body motions in indoor multi-camera capture without external restraints. With 2.14 million frames, this enables data-driven learning of vertebral kinematics from subtle posture variations and bridges the gap between musculoskeletal simulation and computer vision. In addition, we release pretrained baselines covering fine-tuned 2D detectors, monocular 3D pose lifting models, and multi-view reconstruction pipelines, establishing a unified benchmark for biomechanically valid spine motion estimation. Specifically, our 2D spine baselines improve the state-of-the-art from 0.63 to 0.80 AUC in controlled environments, and from 0.91 to 0.93 AP for in-the-wild spine tracking. Together, the simulation framework and SIMSPINE dataset advance research in vision-based biomechanics, motion analysis, and digital human modeling by enabling reproducible, anatomically grounded 3D spine estimation under natural conditions.
>
---
#### [replaced 049] Generating a Paracosm for Training-Free Zero-Shot Composed Image Retrieval
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.00813](https://arxiv.org/pdf/2602.00813)**

> **作者:** Tong Wang; Yunhan Zhao; Shu Kong
>
> **摘要:** Composed Image Retrieval (CIR) is the task of retrieving a target image from a database using a multimodal query, which consists of a reference image and a modification text. The text specifies how to alter the reference image to form a ''mental image'', based on which CIR should find the target image in the database. The fundamental challenge of CIR is that this ''mental image'' is not physically available and is only implicitly defined by the query. The contemporary literature pursues zero-shot methods and uses a Large Multimodal Model (LMM) to generate a textual description for a given multimodal query, and then employs a Vision-Language Model (VLM) for textual-visual matching to search for the target image. In contrast, we address CIR from first principles by directly generating the ''mental image'' for more accurate matching. Particularly, we prompt an LMM to generate a ''mental image'' for a given multimodal query and propose to use this ''mental image'' to search for the target image. As the ''mental image'' has a synthetic-to-real domain gap with real images, we also generate a synthetic counterpart for each real image in the database to facilitate matching. In this sense, our method uses LMM to construct a ``paracosm'', where it matches the multimodal query and database images. Hence, we call this method Paracosm. Notably, Paracosm is a training-free zero-shot CIR method. It significantly outperforms existing zero-shot methods on challenging benchmarks, achieving state-of-the-art performance for zero-shot CIR.
>
---
#### [replaced 050] Token-Level Constraint Boundary Search for Jailbreaking Text-to-Image Models
- **分类: cs.CV; cs.CR**

- **链接: [https://arxiv.org/pdf/2504.11106](https://arxiv.org/pdf/2504.11106)**

> **作者:** Jiangtao Liu; Zhaoxin Wang; Handing Wang; Cong Tian; Yaochu Jin
>
> **摘要:** Text-to-Image (T2I) generation has advanced rapidly in recent years, but they also raise safety concerns due to the potential production of harmful content. In the practical deployments, T2I services typically adopt full-chain defenses that combine a prompt checker, a securely trained generator, and a post-hoc image checker. Jailbreaking such full-chain systems is challenging in the black-box settings because prompt tokens form a discrete combinatorial space and the attack must satisfy multiple coupled constraints under sparse feedback and limited queries. To address these challenges, we propose Token-level Constraint Boundary Search (TCBS)-Attack, a novel query-based black-box jailbreak attack that searches for tokens located near the decision boundaries defined by text and image checkers. TCBS-Attack incorporates decision boundaries as constraint conditions to guide the evolutionary search of token populations, iteratively optimize tokens near these boundaries. Such evolutionary search process reduces the effective search space and improves query efficiency while preserving semantic coherence. Extensive experiments demonstrate that TCBS-Attack consistently outperforms state-of-the-art jailbreak attacks across various T2I models, including securely trained open-source models and commercial online services like DALL-E 3. TCBS-Attack achieves an ASR-4 of 52.5% and an ASR-1 of 22.0% on jailbreaking full-chain T2I models, significantly surpassing baseline methods.
>
---
#### [replaced 051] WebAccessVL: Violation-Aware VLM for Web Accessibility
- **分类: cs.HC; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2602.03850](https://arxiv.org/pdf/2602.03850)**

> **作者:** Amber Yijia Zheng; Jae Joong Lee; Bedrich Benes; Raymond A. Yeh
>
> **摘要:** We present a vision-language model (VLM) that automatically edits website HTML to address violations of the Web Content Accessibility Guidelines 2 (WCAG2) while preserving the original design. We formulate this as a supervised image-conditioned program synthesis task, where the model learns to correct HTML given both the code and its visual rendering. We create WebAccessVL, a website dataset with manually corrected accessibility violations. We then propose a violation-conditioned VLM that further takes the detected violations' descriptions from a checker as input. This conditioning enables an iterative checker-in-the-loop refinement strategy at test time. We conduct extensive evaluation on both open API and open-weight models. Empirically, our method achieves 0.211 violations per website, a 96.0\% reduction from the 5.34 violations in raw data and 87\% better than GPT-5. A perceptual study also confirms that our edited websites better maintain the original visual appearance and content.
>
---
#### [replaced 052] AutoViVQA: A Large-Scale Automatically Constructed Dataset for Vietnamese Visual Question Answering
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.09689](https://arxiv.org/pdf/2603.09689)**

> **作者:** Nguyen Anh Tuong; Phan Ba Duc; Nguyen Trung Quoc; Tran Dac Thinh; Dang Duy Lan; Nguyen Quoc Thinh; Tung Le
>
> **摘要:** Visual Question Answering (VQA) is a fundamental multimodal task that requires models to jointly understand visual and textual information. Early VQA systems relied heavily on language biases, motivating subsequent work to emphasize visual grounding and balanced datasets. With the success of large-scale pre-trained transformers for both text and vision domains -- such as PhoBERT for Vietnamese language understanding and Vision Transformers (ViT) for image representation learning -- multimodal fusion has achieved remarkable progress. For Vietnamese VQA, several datasets have been introduced to promote research in low-resource multimodal learning, including ViVQA, OpenViVQA, and the recently proposed ViTextVQA. These resources enable benchmarking of models that integrate linguistic and visual features in the Vietnamese context. Evaluation of VQA systems often employs automatic metrics originally designed for image captioning or machine translation, such as BLEU, METEOR, CIDEr, Recall, Precision, and F1-score. However, recent research suggests that large language models can further improve the alignment between automatic evaluation and human judgment in VQA tasks. In this work, we explore Vietnamese Visual Question Answering using transformer-based architectures, leveraging both textual and visual pre-training while systematically comparing automatic evaluation metrics under multilingual settings.
>
---
#### [replaced 053] MediRound: Multi-Round Entity-Level Reasoning Segmentation in Medical Images
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.12110](https://arxiv.org/pdf/2511.12110)**

> **作者:** Qinyue Tong; Ziqian Lu; Jun Liu; Rui Zuo; Zheming Lu; Yueming Jin
>
> **备注:** 15pages, 9 figures
>
> **摘要:** Despite recent progress in text-prompt-based medical image segmentation, these methods are limited to single-round dialogues and fail to support multi-round reasoning, which is important for medical education scenarios. In this work, we introduce Multi-Round Entity-Level Medical Reasoning Segmentation (MEMR-Seg), a new task that requires generating segmentation masks through multi-round queries with entity-level reasoning, helping learners progressively develop their understanding of medical knowledge. To support this task, we construct MR-MedSeg, a large-scale dataset of 177K multi-round medical segmentation dialogues, featuring entity-based reasoning across rounds. Furthermore, we propose MediRound, an effective baseline model designed for multi-round medical reasoning segmentation. To mitigate the inherent error propagation within the chain-like pipeline of multi-round segmentation, we introduce a lightweight yet effective Judgment & Correction Mechanism during model inference. Experimental results demonstrate that our method effectively addresses the MEMR-Seg task and outperforms conventional medical referring segmentation methods. The project is available at this https URL.
>
---
#### [replaced 054] PathoScribe: Transforming Pathology Data into a Living Library with a Unified LLM-Driven Framework for Semantic Retrieval and Clinical Integration
- **分类: cs.CV; cs.AI; cs.CL; cs.DL; cs.IR**

- **简介: 该论文提出PathoScribe，解决病理数据难以检索与利用的问题。通过统一的LLM框架，实现病例检索、智能分析和临床整合，提升诊断效率。**

- **链接: [https://arxiv.org/pdf/2603.08935](https://arxiv.org/pdf/2603.08935)**

> **作者:** Abdul Rehman Akbar; Samuel Wales-McGrath; Alejadro Levya; Lina Gokhale; Rajendra Singh; Wei Chen; Anil Parwani; Muhammad Khalid Khan Niazi
>
> **摘要:** Pathology underpins modern diagnosis and cancer care, yet its most valuable asset, the accumulated experience encoded in millions of narrative reports, remains largely inaccessible. Although institutions are rapidly digitizing pathology workflows, storing data without effective mechanisms for retrieval and reasoning risks transforming archives into a passive data repository, where institutional knowledge exists but cannot meaningfully inform patient care. True progress requires not only digitization, but the ability for pathologists to interrogate prior similar cases in real time while evaluating a new diagnostic dilemma. We present PathoScribe, a unified retrieval-augmented large language model (LLM) framework designed to transform static pathology archives into a searchable, reasoning-enabled living library. PathoScribe enables natural language case exploration, automated cohort construction, clinical question answering, immunohistochemistry (IHC) panel recommendation, and prompt-controlled report transformation within a single architecture. Evaluated on 70,000 multi-institutional surgical pathology reports, PathoScribe achieved perfect Recall@10 for natural language case retrieval and demonstrated high-quality retrieval-grounded reasoning (mean reviewer score 4.56/5). Critically, the system operationalized automated cohort construction from free-text eligibility criteria, assembling research-ready cohorts in minutes (mean 9.2 minutes) with 91.3% agreement to human reviewers and no eligible cases incorrectly excluded, representing orders-of-magnitude reductions in time and cost compared to traditional manual chart review. This work establishes a scalable foundation for converting digital pathology archives from passive storage systems into active clinical intelligence platforms.
>
---
#### [replaced 055] Rethinking Two-Stage Referring-by-Tracking in Referring Multi-Object Tracking: Make it Strong Again
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.07516](https://arxiv.org/pdf/2503.07516)**

> **作者:** Weize Li; Yunhao Du; Qixiang Yin; Zhicheng Zhao; Fei Su
>
> **备注:** Accepted to the CVPR 2026
>
> **摘要:** Referring Multi-Object Tracking (RMOT) aims to track multiple objects specified by natural language expressions in videos. With the recent significant progress of one-stage methods, the two-stage Referring-by-Tracking (RBT) paradigm has gradually lost its popularity. However, its lower training cost and flexible incremental deployment remain irreplaceable. Rethinking existing two-stage RBT frameworks, we identify two fundamental limitations: the overly heuristic feature construction and fragile correspondence modeling. To address these issues, we propose FlexHook, a novel two-stage RBT framework. In FlexHook, the proposed Conditioning Hook (C-Hook) redefines the feature construction by a sampling-based strategy and language-conditioned cue injection. Then, we introduce a Pairwise Correspondence Decoder (PCD) that replaces CLIP-based similarity matching with active correspondence modeling, yielding a more flexible and robust strategy. Extensive experiments on multiple benchmarks (Refer-KITTI/v2, Refer-Dance, and LaMOT) demonstrate that FlexHook becomes the first two-stage RBT approach to comprehensively outperform current state-of-the-art methods. Code can be found in the this https URL.
>
---
#### [replaced 056] Clair Obscur: an Illumination-Aware Method for Real-World Image Vectorization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20034](https://arxiv.org/pdf/2511.20034)**

> **作者:** Xingyue Lin; Shuai Peng; Xiangyu Xie; Jianhua Zhu; Yuxuan Zhou; Liangcai Gao
>
> **摘要:** Image vectorization aims to convert raster images into editable, scalable vector representations while preserving visual fidelity. Existing vectorization methods struggle to represent complex real-world images, often producing fragmented shapes at the cost of semantic conciseness. In this paper, we propose COVec, an illumination-aware vectorization framework inspired by the Clair-Obscur principle of light-shade contrast. COVec is the first to introduce intrinsic image decomposition in the vector domain, separating an image into albedo, shade, and light layers in a unified vector representation. A semantic-guided initialization and two-stage optimization refine these layers with differentiable rendering. Experiments on various datasets demonstrate that COVec achieves higher visual fidelity and significantly improved editability compared to existing methods. The code will be released at this https URL.
>
---
#### [replaced 057] SOTA: Self-adaptive Optimal Transport for Zero-Shot Classification with Multiple Foundation Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.13723](https://arxiv.org/pdf/2506.13723)**

> **作者:** Zhanxuan Hu; Qiyu Xu; Yu Duan; Yonghang Tai; Huafeng Li
>
> **摘要:** Foundation models have attracted widespread attention across domains due to their powerful zero-shot classification capabilities. This work is motivated by two key observations: (1) \textit{Vision-Language Models} (VLMs), such as CLIP, often over-rely on class-level textual priors and struggle to capture fine-grained visual cues, whereas \textit{Vision-only Foundation Models} (VFMs), such as DINO, provide rich and discriminative visual features but lack semantic alignment; (2) the performance of different VLMs varies considerably across datasets owing to differences in pre-training. To address these challenges, we propose \textbf{SOTA} (\textit{Self-adaptive Optimal TrAnsport}), a \textit{training-free} ensemble framework that integrates the outputs of multiple foundation models~(VFMs or VLMs) by learning a self-adaptive transport plan. Notably, \textbf{SOTA} is prior-free and automatically balances model contributions. Extensive experiments across diverse domains, including natural images, medical pathology, and remote sensing, validate the generalizability of \textbf{SOTA}. The results consistently show that it effectively leverages the complementary strengths of different foundation models and achieves substantial improvements over individual models. The implementation code is available at: this https URL.
>
---
#### [replaced 058] Pre-training vision models for the classification of alerts from wide-field time-domain surveys
- **分类: astro-ph.IM; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.11957](https://arxiv.org/pdf/2512.11957)**

> **作者:** Nabeel Rehemtulla; Adam A. Miller; Mike Walmsley; Ved G. Shah; Theophile Jegou du Laz; Michael W. Coughlin; Argyro Sasli; Joshua Bloom; Christoffer Fremling; Matthew J. Graham; Steven L. Groom; David Hale; Ashish A. Mahabal; Daniel A. Perley; Josiah Purdum; Ben Rusholme; Jesper Sollerman; Mansi M. Kasliwal
>
> **备注:** Accepted for publication in PASP
>
> **摘要:** Modern wide-field time-domain surveys facilitate the study of transient, variable and moving phenomena by conducting image differencing and relaying alerts to their communities. Machine learning tools have been used on data from these surveys and their precursors for more than a decade, and convolutional neural networks (CNNs), which make predictions directly from input images, saw particularly broad adoption through the 2010s. Since then, continually rapid advances in computer vision have transformed the standard practices around using such models. It is now commonplace to use standardized architectures pre-trained on large corpora of everyday images (e.g., ImageNet). In contrast, time-domain astronomy studies still typically design custom CNN architectures and train them from scratch. Here, we explore the effects of adopting various pre-training regimens and standardized model architectures on the performance of alert classification. We find that the resulting models match or outperform a custom, specialized CNN like what is typically used for filtering alerts. Moreover, our results show that pre-training on galaxy images from Galaxy Zoo tends to yield better performance than pre-training on ImageNet or training from scratch. We observe that the design of standardized architectures are much better optimized than the custom CNN baseline, requiring significantly less time and memory for inference despite having more trainable parameters. On the eve of the Legacy Survey of Space and Time and other image-differencing surveys, these findings advocate for a paradigm shift in the creation of vision models for alerts, demonstrating that greater performance and efficiency, in time and in data, can be achieved by adopting the latest practices from the computer vision field.
>
---
#### [replaced 059] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于多模态学习领域，旨在分析多模态数据中不同模态间的依赖关系。研究通过实验证明，现有基准在模态交互上存在偏差，提出量化方法以改进多模态基准设计。**

- **链接: [https://arxiv.org/pdf/2509.23499](https://arxiv.org/pdf/2509.23499)**

> **作者:** Divyam Madaan; Varshan Muhunthan; Kyunghyun Cho; Sumit Chopra
>
> **备注:** Accepted to ICLR 2026. Code available at this https URL
>
> **摘要:** Understanding the interplay between intra-modality dependencies (the contribution of an individual modality to a target task) and inter-modality dependencies (the relationships between modalities and the target task) is fundamental to advancing multi-modal learning. However, the nature of and interaction between these dependencies within current benchmark evaluations remains poorly characterized. In this work, we present a large-scale empirical study to quantify these dependencies across 23 visual question-answering benchmarks using multi-modal large language models (MLLMs) covering domains such as general and expert knowledge reasoning, optical character recognition, and document understanding. Our findings show that the reliance on vision, question (text), and their interaction varies significantly, both across and within benchmarks. We discover that numerous benchmarks intended to mitigate text-only biases have inadvertently amplified image-only dependencies. This characterization persists across model sizes and types, with models often obtaining high performance by using each modality independently and showing limited dependence on their interaction. We provide a quantitative characterization of multi-modal datasets, enabling a principled approach to multi-modal benchmark design and evaluation.
>
---
#### [replaced 060] MonitorVLM:A Vision Language Framework for Safety Violation Detection in Mining Operations
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.03666](https://arxiv.org/pdf/2510.03666)**

> **作者:** Jiang Wu; Sichao Wu; Yinsong Ma; Guangyuan Yu; Haoyuan Xu; Lifang Zheng; Jingliang Duan
>
> **摘要:** Industrial accidents, particularly in high-risk domains such as surface and underground mining, are frequently caused by unsafe worker behaviors. Traditional manual inspection remains labor-intensive, error-prone, and insufficient for large-scale, dynamic environments, highlighting the urgent need for intelligent and automated safety monitoring. In this paper, we present MonitorVLM, a novel vision--language framework designed to detect safety violations directly from surveillance video streams. MonitorVLM introduces three key innovations: (1) a domain-specific violation dataset comprising 9,000 vision--question--answer (VQA) samples across 40 high-frequency mining regulations, enriched with augmentation and auxiliary detection cues; (2) a clause filter (CF) module that dynamically selects the Top-$K$ most relevant clauses, reducing inference latency by 13.56\% while maintaining accuracy; and (3) a behavior magnifier (BM) module that enhances worker regions to improve fine-grained action recognition, yielding additional gains of 3.45% in precision and 8.62% in recall. Experimental results demonstrate that MonitorVLM significantly outperforms baseline vision--language models, achieving improvements of 22.01% in precision, 34.22\% in recall, and 28.37% in F1 score over the 72B unfine-tuned baseline. A lightweight web-based interface further integrates MonitorVLM into practical workflows, enabling automatic violation reporting with video timestamping. This study highlights the potential of multimodal large models to enhance occupational safety monitoring in mining and beyond.
>
---
#### [replaced 061] Chain-of-Thought Compression Should Not Be Blind: V-Skip for Efficient Multimodal Reasoning via Dual-Path Anchoring
- **分类: cs.MM; cs.CL; cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决CoT推理的高延迟问题。通过V-Skip方法，结合语言和视觉信息优化token压缩，提升效率并保持精度。**

- **链接: [https://arxiv.org/pdf/2601.13879](https://arxiv.org/pdf/2601.13879)**

> **作者:** Dongxu Zhang; Yiding Sun; Cheng Tan; Wenbiao Yan; Ning Yang; Jihua Zhu; Haijun Zhang
>
> **摘要:** While Chain-of-Thought (CoT) reasoning significantly enhances the performance of Multimodal Large Language Models (MLLMs), its autoregressive nature incurs prohibitive latency constraints. Current efforts to mitigate this via token compression often fail by blindly applying text-centric metrics to multimodal contexts. We identify a critical failure mode termed Visual Amnesia, where linguistically redundant tokens are erroneously pruned, leading to hallucinations. To address this, we introduce V-Skip that reformulates token pruning as a Visual-Anchored Information Bottleneck (VA-IB) optimization problem. V-Skip employs a dual-path gating mechanism that weighs token importance through both linguistic surprisal and cross-modal attention flow, effectively rescuing visually salient anchors. Extensive experiments on Qwen2-VL and Llama-3.2 families demonstrate that V-Skip achieves a $2.9\times$ speedup with negligible accuracy loss. Specifically, it preserves fine-grained visual details, outperforming other baselines over 30\% on the DocVQA.
>
---
#### [replaced 062] Seeing Space and Motion: Enhancing Latent Actions with Geometric and Dynamic Awareness for Vision-Language-Action Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.26251](https://arxiv.org/pdf/2509.26251)**

> **作者:** Zhejia Cai; Yandan Yang; Xinyuan Chang; Shiyi Liang; Ronghan Chen; Feng Xiong; Mu Xu; Ruqi Huang
>
> **备注:** 8 pages, correct errors, clarify details
>
> **摘要:** Latent Action Models (LAMs) enable Vision- Language-Action (VLA) systems to learn semantic action representations from large-scale unannotated data. Yet, we identify two bottlenecks of LAMs: 1) the commonly adopted end-to-end trained image encoder suffers from poor spatial understanding; 2) LAMs can be fragile when input frames are temporally distant, leading to limited temporal percep- tion. Such factors inevitably hinder stable and clear action modeling. To this end, we propose Farsighted-LAM, a latent action framework with geometry-aware spatial encoding and multi-scale temporal modeling, capturing structural priors and dynamic motion patterns from consecutive frames. We further propose SSM-VLA, an end-to-end VLA framework built upon Farsighted-LAM, which integrates structured perception with a visual Chain-of-Thought module to explicitly reason about environmental dynamics, enhancing decision consistency and interpretability. We validate SSM-VLA on multiple VLA tasks in both simulation and real-world settings, and achieve state-of- the-art performance. Our results demonstrate that our strategy of combining geometry-aware modeling, temporal coherence, and explicit reasoning is effective in enhancing the robustness and generalizability of embodied intelligence.
>
---
#### [replaced 063] TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21145](https://arxiv.org/pdf/2511.21145)**

> **作者:** Jiaming He; Guanyu Hou; Hongwei Li; Zhicong Huang; Kangjie Chen; Yi Yu; Wenbo Jiang; Guowen Xu; Tianwei Zhang
>
> **备注:** CVPR 2026
>
> **摘要:** Text-to-Video (T2V) models are capable of synthesizing high-quality, temporally coherent dynamic video content, but the diverse generation also inherently introduces critical safety challenges. Existing safety evaluation methods,which focus on static image and text generation, are insufficient to capture the complex temporal dynamics in video generation. To address this, we propose a TEmporal-aware Automated Red-teaming framework, named TEAR, an automated framework designed to uncover safety risks specifically linked to the dynamic temporal sequencing of T2V models. TEAR employs a temporal-aware test generator optimized via a two-stage approach: initial generator training and temporal-aware online preference learning, to craft textually innocuous prompts that exploit temporal dynamics to elicit policy-violating video output. And a refine model is adopted to improve the prompt stealthiness and adversarial effectiveness cyclically. Extensive experimental evaluation demonstrates the effectiveness of TEAR across open-source and commercial T2V systems with over 80% attack success rate, a significant boost from prior best result of 57%.
>
---
#### [replaced 064] AD-R1: Closed-Loop Reinforcement Learning for End-to-End Autonomous Driving with Impartial World Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20325](https://arxiv.org/pdf/2511.20325)**

> **作者:** Tianyi Yan; Tao Tang; Xingtai Gui; Yongkang Li; Jiasen Zhesng; Weiyao Huang; Lingdong Kong; Wencheng Han; Xia Zhou; Xueyang Zhang; Yifei Zhan; Kun Zhan; Cheng-zhong Xu; Jianbing Shen
>
> **摘要:** End-to-end models for autonomous driving hold the promise of learning complex behaviors directly from sensor data, but face critical challenges in safety and handling long-tail events. Reinforcement Learning (RL) offers a promising path to overcome these limitations, yet its success in autonomous driving has been elusive. We identify a fundamental flaw hindering this progress: a deep seated optimistic bias in the world models used for RL. To address this, we introduce a framework for post-training policy refinement built around an Impartial World Model. Our primary contribution is to teach this model to be honest about danger. We achieve this with a novel data synthesis pipeline, Counterfactual Synthesis, which systematically generates a rich curriculum of plausible collisions and off-road events. This transforms the model from a passive scene completer into a veridical forecaster that remains faithful to the causal link between actions and outcomes. We then integrate this Impartial World Model into our closed-loop RL framework, where it serves as an internal critic. During refinement, the agent queries the critic to ``dream" of the outcomes for candidate actions. We demonstrate through extensive experiments, including on a new Risk Foreseeing Benchmark, that our model significantly outperforms baselines in predicting failures. Consequently, when used as a critic, it enables a substantial reduction in safety violations in challenging simulations, proving that teaching a model to dream of danger is a critical step towards building truly safe and intelligent autonomous agents.
>
---
#### [replaced 065] Inferring Clinically Relevant Molecular Subtypes of Pancreatic Cancer from Routine Histopathology Using Deep Learning
- **分类: cs.LG; cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2601.03410](https://arxiv.org/pdf/2601.03410)**

> **作者:** Abdul Rehman Akbar; Alejandro Levya; Ashwini Esnakula; Elshad Hasanov; Anne Noonan; Lingbin Meng; Susan Tsai; Vaibhav Sahai; Midhun Malla; Sarbajit Mukherjee; Upender Manne; Anil Parwani; Wei Chen; Ashish Manne; Muhammad Khalid Khan Niazi
>
> **摘要:** Molecular subtyping of PDAC into basal-like and classical has established prognostic and predictive value. However, its use in clinical practice is limited by cost, turnaround time, and tissue requirements, thereby restricting its application in the management of PDAC. We introduce PanSubNet, an interpretable deep learning framework that predicts therapy-relevant molecular subtypes directly from standard H&E-stained WSIs. PanSubNet was developed using data from 1,055 patients across two multi-institutional cohorts (PANCAN, n=846; TCGA, n=209) with paired histology and RNA-seq data. Ground-truth labels were derived using the validated Moffitt 50-gene signature refined by GATA6 expression. The model employs dual-scale architecture that fuses cellular-level morphology with tissue-level architecture, leveraging attention mechanisms for multi-scale representation learning and transparent feature attribution. On internal validation within PANCAN using five-fold cross-validation, PanSubNet achieved mean AUC of 88.5% with balanced sensitivity and specificity. External validation on the independent TCGA cohort without fine-tuning demonstrated robust generalizability (AUC 84.0%). PanSubNet preserved and, in metastatic disease, strengthened prognostic stratification compared to RNA-seq based labels. Prediction uncertainty linked to intermediate transcriptional states, not classification noise. Model predictions are aligned with established transcriptomic programs, differentiation markers, and DNA damage repair signatures. By enabling rapid, cost-effective molecular stratification from routine H&E-stained slides, PanSubNet offers a clinically deployable and interpretable tool for genetic subtyping. We are gathering data from two institutions to validate and assess real-world performance, supporting integration into digital pathology workflows and advancing precision oncology for PDAC.
>
---
#### [replaced 066] Rethinking Few-Shot Image Fusion: Granular Ball Priors Enable General-Purpose Deep Fusion
- **分类: cs.GR; cs.CV; cs.LG; eess.IV; stat.ML**

- **链接: [https://arxiv.org/pdf/2504.08937](https://arxiv.org/pdf/2504.08937)**

> **作者:** Minjie Deng; Yan Wei; An Wu; Yuncan Ouyang; Hao Zhai; Qianyao Peng
>
> **摘要:** In image fusion tasks, the absence of real fused images as supervision signals poses significant challenges for supervised learning. Existing deep learning methods typically address this issue either by designing handcrafted priors or by relying on large-scale datasets to learn model parameters. Different from previous approaches, this paper introduces the concept of incomplete priors, which formally describe handcrafted priors at the algorithmic level and estimate their confidence. Based on this idea, we couple incomplete priors with the neural network through a sample-level adaptive loss function, enabling the network to learn and re-infer fusion rules under conditions that approximate the real fusion this http URL generate incomplete priors, we propose a Granular Ball Pixel Computation (GBPC) algorithm based on the principles of granular computing. The algorithm models fused-image pixels as information units, estimating pixel weights at a fine-grained level while statistically evaluating prior reliability at a coarse-grained level. This design enables the algorithm to perceive cross-modal discrepancies and perform adaptive this http URL results demonstrate that even under few-shot conditions, a lightweight neural network can still learn effective fusion rules by training only on image patches extracted from ten image pairs. Extensive experiments across multiple fusion tasks and datasets further show that the proposed method achieves superior performance in both visual quality and model compactness. The code is available at: this https URL
>
---
#### [replaced 067] Context-Nav: Context-Driven Exploration and Viewpoint-Aware 3D Spatial Reasoning for Instance Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出Context-Nav，解决文本引导的实例导航任务，通过上下文驱动探索和3D空间推理，提升在复杂场景中精准定位目标实例的能力。**

- **链接: [https://arxiv.org/pdf/2603.09506](https://arxiv.org/pdf/2603.09506)**

> **作者:** Won Shik Jang; Ue-Hwan Kim
>
> **备注:** Camera-ready version. Accepted to CVPR 2026
>
> **摘要:** Text-goal instance navigation (TGIN) asks an agent to resolve a single, free-form description into actions that reach the correct object instance among same-category distractors. We present \textit{Context-Nav} that elevates long, contextual captions from a local matching cue to a global exploration prior and verifies candidates through 3D spatial reasoning. First, we compute dense text-image alignments for a value map that ranks frontiers -- guiding exploration toward regions consistent with the entire description rather than early detections. Second, upon observing a candidate, we perform a viewpoint-aware relation check: the agent samples plausible observer poses, aligns local frames, and accepts a target only if the spatial relations can be satisfied from at least one viewpoint. The pipeline requires no task-specific training or fine-tuning; we attain state-of-the-art performance on InstanceNav and CoIN-Bench. Ablations show that (i) encoding full captions into the value map avoids wasted motion and (ii) explicit, viewpoint-aware 3D verification prevents semantically plausible but incorrect stops. This suggests that geometry-grounded spatial reasoning is a scalable alternative to heavy policy training or human-in-the-loop interaction for fine-grained instance disambiguation in cluttered 3D scenes.
>
---
#### [replaced 068] Content-Aware Mamba for Learned Image Compression
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.02192](https://arxiv.org/pdf/2508.02192)**

> **作者:** Yunuo Chen; Zezheng Lyu; Bing He; Hongwei Hu; Qi Wang; Yuan Tian; Li Song; Wenjun Zhang; Guo Lu
>
> **备注:** ICLR2026 poster
>
> **摘要:** Recent learned image compression (LIC) leverages Mamba-style state-space models (SSMs) for global receptive fields with linear complexity. However, the standard Mamba adopts content-agnostic, predefined raster (or multi-directional) scans under strict causality. This rigidity hinders its ability to effectively eliminate redundancy between tokens that are content-correlated but spatially distant. We introduce Content-Aware Mamba (CAM), an SSM that dynamically adapts its processing to the image content. Specifically, CAM overcomes prior limitations with two novel mechanisms. First, it replaces the rigid scan with a content-adaptive token permutation strategy to prioritize interactions between content-similar tokens regardless of their location. Second, it overcomes the sequential dependency by injecting sample-specific global priors into the state-space model, which effectively mitigates the strict causality without multi-directional scans. These innovations enable CAM to better capture global redundancy while preserving computational efficiency. Our Content-Aware Mamba-based LIC model (CMIC) achieves state-of-the-art rate-distortion performance, surpassing VTM-21.0 by 15.91%, 21.34%, and 17.58% in BD-rate on the Kodak, Tecnick, and CLIC datasets, respectively. Code will be released at this https URL.
>
---
#### [replaced 069] Average Calibration Losses for Reliable Uncertainty in Medical Image Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.03942](https://arxiv.org/pdf/2506.03942)**

> **作者:** Theodore Barfoot; Luis C. Garcia-Peraza-Herrera; Samet Akcay; Ben Glocker; Tom Vercauteren
>
> **备注:** 15 pages, 6 figures, IEEE TMI submission. This version originally appeared in error as arXiv:2403.06759(v2)
>
> **摘要:** Deep neural networks for medical image segmentation are often overconfident, compromising both reliability and clinical utility. In this work, we propose differentiable formulations of marginal L1 Average Calibration Error (mL1-ACE) as an auxiliary loss that can be computed on a per-image basis. We compare both hard- and soft-binning approaches to directly improve pixel-wise calibration. Our experiments on four datasets (ACDC, AMOS, KiTS, BraTS) demonstrate that incorporating mL1-ACE significantly reduces calibration errors, particularly Average Calibration Error (ACE) and Maximum Calibration Error (MCE), while largely maintaining high Dice Similarity Coefficients (DSCs). We find that the soft-binned variant yields the greatest improvements in calibration over the DSC plus cross-entropy loss baseline but often compromises segmentation performance, with hard-binned mL1-ACE maintaining segmentation performance, albeit with weaker calibration improvement. To gain further insight into calibration performance and its variability across an imaging dataset, we introduce dataset reliability histograms, an aggregation of per-image reliability diagrams. The resulting analysis highlights improved alignment between predicted confidences and true accuracies. Overall, our approach provides practitioners with explicit control over the calibration-accuracy trade-off, enabling more reliable integration of deep learning methods into clinical workflows. We share our code here: this https URL
>
---
#### [replaced 070] GOT-JEPA: Generic Object Tracking with Model Adaptation and Occlusion Handling using Joint-Embedding Predictive Architecture
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.NE**

- **链接: [https://arxiv.org/pdf/2602.14771](https://arxiv.org/pdf/2602.14771)**

> **作者:** Shih-Fang Chen; Jun-Cheng Chen; I-Hong Jhuo; Yen-Yu Lin
>
> **备注:** Accepted in IEEE Transactions on Circuits and Systems for Video Technology (TCSVT). Learning Model Adaptation for Adverse and Dynamic Environments and Fine-Grained Occlusion Perception for Tracker
>
> **摘要:** The human visual system tracks objects by integrating current observations with previously observed information, adapting to target and scene changes, and reasoning about occlusion at fine granularity. In contrast, recent generic object trackers are often optimized for training targets, which limits robustness and generalization in unseen scenarios, and their occlusion reasoning remains coarse, lacking detailed modeling of occlusion patterns. To address these limitations in generalization and occlusion perception, we propose GOT-JEPA, a model-predictive pretraining framework that extends JEPA from predicting image features to predicting tracking models. Given identical historical information, a teacher predictor generates pseudo-tracking models from a clean current frame, and a student predictor learns to predict the same pseudo-tracking models from a corrupted version of the current frame. This design provides stable pseudo supervision and explicitly trains the predictor to produce reliable tracking models under occlusions, distractors, and other adverse observations, improving generalization to dynamic environments. Building on GOT-JEPA, we further propose OccuSolver to enhance occlusion perception for object tracking. OccuSolver adapts a point-centric point tracker for object-aware visibility estimation and detailed occlusion-pattern capture. Conditioned on object priors iteratively generated by the tracker, OccuSolver incrementally refines visibility states, strengthens occlusion handling, and produces higher-quality reference labels that progressively improve subsequent model predictions. Extensive evaluations on seven benchmarks show that our method effectively enhances tracker generalization and robustness.
>
---
#### [replaced 071] Mind the Way You Select Negative Texts: Pursuing the Distance Consistency in OOD Detection with VLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.02618](https://arxiv.org/pdf/2603.02618)**

> **作者:** Zhikang Xu; Qianqian Xu; Zitai Wang; Cong Hua; Sicong Li; Zhiyong Yang; Qingming Huang
>
> **备注:** Accepted by the main track of CVPR 2026
>
> **摘要:** Out-of-distribution (OOD) detection seeks to identify samples from unknown classes, a critical capability for deploying machine learning models in open-world scenarios. Recent research has demonstrated that Vision-Language Models (VLMs) can effectively leverage their multi-modal representations for OOD detection. However, current methods often incorporate intra-modal distance during OOD detection, such as comparing negative texts with ID labels or comparing test images with image proxies. This design paradigm creates an inherent inconsistency against the inter-modal distance that CLIP-like VLMs are optimized for, potentially leading to suboptimal performance. To address this limitation, we propose InterNeg, a simple yet effective framework that systematically utilizes consistent inter-modal distance enhancement from textual and visual perspectives. From the textual perspective, we devise an inter-modal criterion for selecting negative texts. From the visual perspective, we dynamically identify high-confidence OOD images and invert them into the textual space, generating extra negative text embeddings guided by inter-modal distance. Extensive experiments across multiple benchmarks demonstrate the superiority of our approach. Notably, our InterNeg achieves state-of-the-art performance compared to existing works, with a 3.47% reduction in FPR95 on the large-scale ImageNet benchmark and a 5.50% improvement in AUROC on the challenging Near-OOD benchmark.
>
---
#### [replaced 072] Structured Bitmap-to-Mesh Triangulation for Geometry-Aware Discretization of Image-Derived Domains
- **分类: cs.CG; cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2602.19474](https://arxiv.org/pdf/2602.19474)**

> **作者:** Wei Feng; Haiyong Zheng
>
> **备注:** This version updates the Gmsh baseline configuration and comparative statistics, revises the downstream heat-diffusion comparison, expands the threshold-sensitivity study in the supplementary material, and corrects minor numerical values in the star-domain results without changing any conclusions. Code: this https URL
>
> **摘要:** We propose a template-driven triangulation framework that embeds raster- or segmentation-derived boundaries into a regular triangular grid for stable PDE discretization on image-derived domains. Unlike constrained Delaunay triangulation (CDT), which may trigger global connectivity updates, our method retriangulates only triangles intersected by the boundary, preserves the base mesh, and supports synchronization-free parallel execution. To ensure determinism and scalability, we classify all local boundary-intersection configurations up to discrete equivalence and triangle symmetries, yielding a finite symbolic lookup table that maps each case to a conflict-free retriangulation template. We prove that the resulting mesh is closed, has bounded angles, and is compatible with cotangent-based discretizations and standard finite element methods. Experiments on elliptic and parabolic PDEs, signal interpolation, and structural metrics show fewer sliver elements, more regular triangles, and improved geometric fidelity near complex boundaries. The framework is well suited for real-time geometric analysis and physically based simulation over image-derived domains.
>
---
#### [replaced 073] CostNav: A Navigation Benchmark for Real-World Economic-Cost Evaluation of Physical AI Agents
- **分类: cs.AI; cs.CE; cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出CostNav，一个用于评估物理AI代理经济成本的导航基准，解决传统导航任务忽略经济约束的问题。通过整合真实商业数据，评估方法的经济可行性。**

- **链接: [https://arxiv.org/pdf/2511.20216](https://arxiv.org/pdf/2511.20216)**

> **作者:** Haebin Seong; Sungmin Kim; Yongjun Cho; Myunchul Joe; Geunwoo Kim; Yubeen Park; Sunhoo Kim; Yoonshik Kim; Suhwan Choi; Jaeyoon Jung; Jiyong Youn; Jinmyung Kwak; Sunghee Ahn; Jaemin Lee; Younggil Do; Seungyeop Yi; Woojin Cheong; Minhyeok Oh; Minchan Kim; Seongjae Kang; Samwoo Seong; Youngjae Yu; Yunsung Lee
>
> **摘要:** While current navigation benchmarks prioritize task success in simplified settings, they neglect the multidimensional economic constraints essential for the real-world commercialization of autonomous delivery systems. We introduce CostNav, an Economic Navigation Benchmark that evaluates physical AI agents through comprehensive economic cost-revenue analysis aligned with real-world business operations. By integrating industry-standard data--such as Securities and Exchange Commission (SEC) filings and Abbreviated Injury Scale (AIS) injury reports--with Isaac Sim's detailed collision and cargo dynamics, CostNav transcends simple task completion to accurately evaluate business value in complex, real-world scenarios. To our knowledge, CostNav is the first physics-grounded economic benchmark that uses industry-standard regulatory and financial data to quantitatively expose the gap between navigation research metrics and commercial viability, revealing that optimizing for task success on a simplified task fundamentally differs from optimizing for real-world economic deployment. Evaluating seven baselines--two rule-based and five imitation learning--we find that no current method is economically viable, all yielding negative contribution margins. The best-performing method, CANVAS (-27.36\$/run), equipped with only an RGB camera and GPS, outperforms LiDAR-equipped Nav2 w/ GPS (-35.46\$/run). We challenge the community to develop navigation policies that achieve economic viability on CostNav. We remain method-agnostic, evaluating success solely on cost rather than the underlying architecture. All resources are available at this https URL.
>
---
#### [replaced 074] OmniVTON++: Training-Free Universal Virtual Try-On with Principal Pose Guidance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.14552](https://arxiv.org/pdf/2602.14552)**

> **作者:** Zhaotong Yang; Yong Du; Shengfeng He; Yuhui Li; Xinzhe Li; Yangyang Xu; Junyu Dong; Jian Yang
>
> **摘要:** Image-based Virtual Try-On (VTON) concerns the synthesis of realistic person imagery through garment re-rendering under human pose and body constraints. In practice, however, existing approaches are typically optimized for specific data conditions, making their deployment reliant on retraining and limiting their generalization as a unified solution. We present OmniVTON++, a training-free VTON framework designed for universal applicability. It addresses the intertwined challenges of garment alignment, human structural coherence, and boundary continuity by coordinating Structured Garment Morphing for correspondence-driven garment adaptation, Principal Pose Guidance for step-wise structural regulation during diffusion sampling, and Continuous Boundary Stitching for boundary-aware refinement, forming a cohesive pipeline without task-specific retraining. Experimental results demonstrate that OmniVTON++ achieves state-of-the-art performance across diverse generalization settings, including cross-dataset and cross-garment-type evaluations, while reliably operating across scenarios and diffusion backbones within a single formulation. In addition to single-garment, single-human cases, the framework supports multi-garment, multi-human, and anime character virtual try-on, expanding the scope of virtual try-on applications. The code is available at this https URL.
>
---
#### [replaced 075] FreeFly-Thinking : Aligning Chain-of-Thought Reasoning with Continuous UAV Navigation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.07181](https://arxiv.org/pdf/2603.07181)**

> **作者:** Jiaxu Zhou; Shaobo Wang; Zhiyuan Yang; Zhenjun Yu; Tao Li
>
> **备注:** 10 pages, 5 figures, ECCV review
>
> **摘要:** Vision-Language Navigation aims to enable agents to understand natural language instructions and carry out appropriate navigation actions in real-world environments. Most work focuses on indoor settings, with little research in complex outdoor scenes. Current UAV Vision-and-Language Navigation models typically act as black boxes without explicit reasoning. We introduce FreeFly-thinking, an end-to-end VLN framework that converts the UAV agent's egocentric images and language instructions into a series of actions, inspired by environment of urban architecture proposed by OpenFly. We first construct a UAV dataset for navigation task, and then performing natural language chain of thought. We adopt a two-stage training strategy: Supervised fine-tuning and Reinforcement fine-tuning. Experiments on unseen test demonstrate a strong performance, presenting robustness and efficiency in UAV navigation issue.
>
---
#### [replaced 076] PD-Diag-Net: Clinical-Priors guided Network on Brain MRI for Auxiliary Diagnosis of Parkinson's Disease
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.23719](https://arxiv.org/pdf/2509.23719)**

> **作者:** Shuai Shao; Yan Wang; Shu Jiang; Shiyuan Zhao; Di Yang; Jiangtao Wang; Yutong Bai; Jianguo Zhang
>
> **摘要:** Parkinson's disease (PD) is a common neurodegenerative disorder that severely diminishes patients' quality of life. Its global prevalence has increased markedly in recent decades. Current diagnostic workflows are complex and heavily reliant on neurologists' expertise, often resulting in delays in early detection and missed opportunities for timely intervention. To address these issues, we propose an end-to-end automated diagnostic method for PD, termed PD-Diag-Net, which performs risk assessment and auxiliary diagnosis directly from raw MRI scans. This framework first introduces an MRI Pre-processing Module (MRI-Processor) to mitigate inter-subject and inter-scanner variability by flexibly integrating established medical imaging preprocessing tools. It then incorporates two forms of clinical prior knowledge: (1) Brain-Region-Relevance-Prior (Relevance-Prior), which specifies brain regions strongly associated with PD; and (2) Brain-Region-Aging-Prior (Aging-Prior), which reflects the accelerated aging typically observed in PD-associated regions. Building on these priors, we design two dedicated modules: the Relevance-Prior Guided Feature Aggregation Module (Aggregator), which guides the model to focus on PD-associated regions at the inter-subject level, and the Age-Prior Guided Diagnosis Module (Diagnoser), which leverages brain age gaps as auxiliary constraints at the intra-subject level to enhance diagnostic accuracy and clinical interpretability. Furthermore, we collected external test data from our collaborating hospital. Experimental results show that PD-Diag-Net achieves 86\% accuracy on external tests and over 96% accuracy in early-stage diagnosis, outperforming existing advanced methods by more than 20%.
>
---
#### [replaced 077] UltraGen: Efficient Ultra-High-Resolution Image Generation with Hierarchical Local Attention
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.16325](https://arxiv.org/pdf/2510.16325)**

> **作者:** Yuyao Zhang; Yu-Wing Tai
>
> **备注:** 28 pages
>
> **摘要:** Ultra-high-resolution text-to-image generation is increasingly vital for applications requiring fine-grained textures and global structural fidelity, yet state-of-the-art text-to-image diffusion models such as FLUX and SD3 remain confined to sub 2MP (< $1K\times2K$) resolutions due to the quadratic complexity of attention mechanisms and the scarcity of high-quality high-resolution training data. We present \textbf{\ourwork}, a novel framework that introduces hierarchical local attention with low-resolution global guidance, enabling efficient, scalable, and semantically coherent image synthesis at ultra-high resolutions. Specifically, high-resolution latents are divided into hardware aligned fixed-size local windows to reduce attention complexity from quadratic to near-linear, while a low-resolution latent equipped with scaled positional embeddings injects global semantics as an anchor. A lightweight LoRA adaptation bridges global and local pathways during denoising, ensuring consistency across structure and detail. To maximize inference efficiency and achieve scalable ultra-high-resolution generation, we repermute token sequence in window-first order, so that the GPU-friendly dense local blocks in attention calculation equals to the fixed-size local window in 2D regardless of resolution. Together~\ourwork~reliably scales the pretrained model to resolutions higher than $8K$ with more than $10\times$ speed up and significantly lower memory usage. Extensive experiments demonstrate that~\ourwork~achieves superior quality while maintaining computational efficiency, establishing a practical paradigm for advancing ultra-high-resolution image generation.
>
---
#### [replaced 078] Uncovering Semantic Selectivity of Latent Groups in Higher Visual Cortex with Mutual Information-Guided Diffusion
- **分类: q-bio.NC; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.02182](https://arxiv.org/pdf/2510.02182)**

> **作者:** Yule Wang; Joseph Yu; Chengrui Li; Weihan Li; Anqi Wu
>
> **摘要:** Understanding how neural populations in higher visual areas encode object-centered visual information remains a central challenge in computational neuroscience. Prior works have investigated representational alignment between artificial neural networks and the visual cortex. Nevertheless, these findings are indirect and offer limited insights to the structure of neural populations themselves. Similarly, decoding-based methods have quantified semantic features from neural populations but have not uncovered their underlying organizations. This leaves open a scientific question: "how feature-specific visual information is distributed across neural populations in higher visual areas, and whether it is organized into structured, semantically meaningful subspaces." To tackle this problem, we present MIG-Vis, a method that leverages the generative power of diffusion models to visualize and validate the visual-semantic attributes encoded in neural latent subspaces. Our method first uses a variational autoencoder to infer a group-wise disentangled neural latent subspace from neural populations. Subsequently, we propose a mutual information (MI)-guided diffusion synthesis procedure to visualize the specific visual-semantic features encoded by each latent group. We validate MIG-Vis on multi-session neural spiking datasets from the inferior temporal (IT) cortex of two macaques. The synthesized results demonstrate that our method identifies neural latent groups with clear semantic selectivity to diverse visual features, including object pose, inter-category transformations, and intra-class content. These findings provide direct, interpretable evidence of structured semantic representation in the higher visual cortex and advance our understanding of its encoding principles.
>
---
#### [replaced 079] Speech-to-LaTeX: New Models and Datasets for Converting Spoken Equations and Sentences
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.03542](https://arxiv.org/pdf/2508.03542)**

> **作者:** Dmitrii Korzh; Dmitrii Tarasov; Artyom Iudin; Elvir Karimov; Matvey Skripkin; Nikita Kuzmin; Andrey Kuznetsov; Oleg Y. Rogov; Ivan Oseledets
>
> **备注:** 22 pages, 2 figures, 16 Tables
>
> **摘要:** Conversion of spoken mathematical expressions is a challenging task that involves transcribing speech into a strictly structured symbolic representation while addressing the ambiguity inherent in the pronunciation of equations. Although significant progress has been achieved in automatic speech recognition (ASR) and language models (LM), the problem of converting spoken mathematics into LaTeX remains underexplored. This task directly applies to educational and research domains, such as lecture transcription or note creation. Based on ASR post-correction, prior work requires 2 transcriptions, focuses only on isolated equations, has a limited test set, and provides neither training data nor multilingual coverage. To address these issues, we present the first fully open-source large-scale dataset, comprising over 66,000 human-annotated audio samples of mathematical equations and sentences in English and Russian, drawn from diverse scientific domains. In addition to the ASR post-correction models and few-shot prompting, we apply audio language models, demonstrating comparable character error rate (CER) results on the MathSpeech benchmark (28% vs. 30%) for the equations conversion. In contrast, on the proposed S2L-equations benchmark, our models outperform the MathSpeech model by a substantial margin of more than 36 percentage points, even after accounting for LaTeX formatting artifacts (27% vs. 64%). We establish the first benchmark for mathematical sentence recognition (S2L-sentences) and achieve an equation CER of 40%. This work lays the groundwork for future advances in multimodal AI, with a particular focus on mathematical content recognition.
>
---
#### [replaced 080] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.13043](https://arxiv.org/pdf/2512.13043)**

> **作者:** Tong Wei; Yijun Yang; Changhao Zhang; Junliang Xing; Yuanchun Shi; Zongqing Lu; Deheng Ye
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Multi-turn reinforcement learning (RL) for multi-modal agents built upon vision-language models (VLMs) is hampered by sparse rewards and long-horizon credit assignment. Recent methods densify the reward by querying a teacher that provides step-level feedback, e.g., Guided Thought Reinforcement (GTR) and On-Policy Distillation, but rely on costly, often privileged models as the teacher, limiting practicality and reproducibility. We introduce GTR-Turbo, a highly efficient upgrade to GTR that matches its performance without training on or querying an expensive teacher model. Specifically, GTR-Turbo merges the weights of checkpoints produced during ongoing RL training and then uses the resulting merged model as a "free" teacher to guide subsequent RL via supervised fine-tuning or soft logit distillation. This design removes dependence on privileged VLMs (e.g., GPT or Gemini), mitigates the "entropy collapse" observed in prior work, and maintains stable training. Across diverse visual agentic tasks, GTR-Turbo improves the accuracy of the baseline model by 10-30% while reducing wall-clock training time by 50% and compute cost by 60% relative to GTR.
>
---
#### [replaced 081] Efficient Audio-Visual Speech Separation with Discrete Lip Semantics and Multi-Scale Global-Local Attention
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于音频-视觉语音分离任务，旨在提升分离效率。针对现有方法参数多、计算成本高的问题，提出Dolphin模型，结合轻量编码器与多尺度注意力机制，实现高效准确的语音分离。**

- **链接: [https://arxiv.org/pdf/2509.23610](https://arxiv.org/pdf/2509.23610)**

> **作者:** Kai Li; Kejun Gao; Xiaolin Hu
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Audio-visual speech separation (AVSS) methods leverage visual cues to extract target speech and have demonstrated strong separation quality in noisy acoustic environments. However, these methods usually involve a large number of parameters and require high computational cost, which is unacceptable in many applications where speech separation serves as only a preprocessing step for further speech processing. To address this issue, we propose an efficient AVSS method, named Dolphin. For visual feature extraction, we develop DP-LipCoder, a dual-path lightweight video encoder that transforms lip-motion into discrete audio-aligned semantic tokens. For audio separation, we construct a lightweight encoder-decoder separator, in which each layer incorporates a global-local attention (GLA) block to efficiently capture multi-scale dependencies. Experiments on three benchmark datasets showed that Dolphin not only surpassed the current state-of-the-art (SOTA) model in separation quality but also achieved remarkable improvements in efficiency: over 50% fewer parameters, more than 2.4x reduction in MACs, and over 6x faster GPU inference speed. These results indicate that Dolphin offers a practical and deployable solution for high-performance AVSS in real-world scenarios. Our code and demo page are publicly available at this http URL.
>
---
#### [replaced 082] Ultra-Low Bitrate Perceptual Image Compression with Shallow Encoder
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.12229](https://arxiv.org/pdf/2512.12229)**

> **作者:** Tianyu Zhang; Dong Liu; Chang Wen Chen
>
> **备注:** Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026
>
> **摘要:** Ultra-low bitrate image compression (below 0.05 bits per pixel) is increasingly critical for bandwidth-constrained and computation-limited encoding scenarios such as edge devices. Existing frameworks typically rely on large pretrained encoders (e.g., VAEs or tokenizer-based models) and perform transform coding within their generative latent space. While these approaches achieve impressive perceptual fidelity, their reliance on heavy encoder networks makes them unsuitable for deployment on weak sender devices. In this work, we explore the feasibility of applying shallow encoders for ultra-low bitrate compression and propose a novel Asymmetric Extreme Image Compression (AEIC) framework that pursues simultaneously encoding simplicity and decoding quality. Specifically, AEIC employs moderate or even shallow encoder networks, while leveraging an one-step diffusion decoder to maintain high-fidelity and high-realism reconstructions under extreme bitrates. To further enhance the efficiency of shallow encoders, we design a dual-side feature distillation scheme that transfers knowledge from AEIC with moderate encoders to its shallow encoder variants. Experiments show that AEIC not only outperforms existing methods on rate-distortion-perception performance at ultra-low bitrates, but also delivers exceptional encoding efficiency for 35.8 FPS on 1080P images, while maintaining competitive decoding speed compared to existing methods. Code is available at this https URL.
>
---
#### [replaced 083] Data relativistic uncertainty framework for low-illumination anime scenery image enhancement
- **分类: cs.CV; cs.LG; cs.MM**

- **链接: [https://arxiv.org/pdf/2512.21944](https://arxiv.org/pdf/2512.21944)**

> **作者:** Yiquan Gao; John See
>
> **备注:** Add data
>
> **摘要:** By contrast with the prevailing works of low-light enhancement in natural images and videos, this study copes with the low-illumination quality degradation in anime scenery images to bridge the domain gap. For such an underexplored enhancement task, we first curate images from various sources and construct an unpaired anime scenery dataset with diverse environments and illumination conditions to address the data scarcity. To exploit the power of uncertainty information inherent with the diverse illumination conditions, we propose a Data Relativistic Uncertainty (DRU) framework, motivated by the idea from Relativistic GAN. By analogy with the wave-particle duality of light, our framework interpretably defines and quantifies the illumination uncertainty of dark/bright samples, which is leveraged to dynamically adjust the objective functions to recalibrate the model learning under data uncertainty. Extensive experiments demonstrate the effectiveness of DRU framework by training several versions of EnlightenGANs, yielding superior perceptual and aesthetic qualities beyond the state-of-the-art methods that are incapable of learning from data uncertainty perspective. We hope our framework can expose a novel paradigm of data-centric learning for potential visual and language domains. Code is available.
>
---
#### [replaced 084] Segmentation of Retinal Low-Cost Optical Coherence Tomography Images using Deep Learning
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2001.08480](https://arxiv.org/pdf/2001.08480)**

> **作者:** Timo Kepp; Helge Sudkamp; Claus von der Burchard; Hendrik Schenke; Peter Koch; Gereon Hüttmann; Johann Roider; Mattias P. Heinrich; Heinz Handels
>
> **备注:** Accepted for SPIE Medical Imaging 2020: Computer-Aided Diagnosis
>
> **摘要:** The treatment of age-related macular degeneration (AMD) requires continuous eye exams using optical coherence tomography (OCT). The need for treatment is determined by the presence or change of disease-specific OCT-based biomarkers. Therefore, the monitoring frequency has a significant influence on the success of AMD therapy. However, the monitoring frequency of current treatment schemes is not individually adapted to the patient and therefore often insufficient. While a higher monitoring frequency would have a positive effect on the success of treatment, in practice it can only be achieved with a home monitoring solution. One of the key requirements of a home monitoring OCT system is a computer-aided diagnosis to automatically detect and quantify pathological changes using specific OCT-based biomarkers. In this paper, for the first time, retinal scans of a novel self-examination low-cost full-field OCT (SELF-OCT) are segmented using a deep learning-based approach. A convolutional neural network (CNN) is utilized to segment the total retina as well as pigment epithelial detachments (PED). It is shown that the CNN-based approach can segment the retina with high accuracy, whereas the segmentation of the PED proves to be challenging. In addition, a convolutional denoising autoencoder (CDAE) refines the CNN prediction, which has previously learned retinal shape information. It is shown that the CDAE refinement can correct segmentation errors caused by artifacts in the OCT image.
>
---
#### [replaced 085] SEGA: Drivable 3D Gaussian Head Avatar from a Single Image
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2504.14373](https://arxiv.org/pdf/2504.14373)**

> **作者:** Chen Guo; Zhuo Su; Liao Wang; Jian Wang; Shuang Li; Xu Chang; Zhaohu Li; Yang Zhao; Guidong Wang; Yebin Liu; Ruqi Huang
>
> **摘要:** Creating photorealistic 3D head avatars from limited input has become increasingly important for applications in virtual reality, telepresence, and digital entertainment. While recent advances like neural rendering and 3D Gaussian splatting have enabled high-quality digital human avatar creation and animation, most methods rely on multiple images or multi-view inputs, limiting their practicality for real-world use. In this paper, we propose SEGA, a novel approach for Single-imagE-based 3D drivable Gaussian head Avatar creation that combines generalized prior models with a new hierarchical UV-space Gaussian Splatting framework. SEGA seamlessly combines priors derived from large-scale 2D datasets with 3D priors learned from multi-view, multi-expression, and multi-ID data, achieving robust generalization to unseen identities while ensuring 3D consistency across novel viewpoints and expressions. We further present a hierarchical UV-space Gaussian Splatting framework that leverages FLAME-based structural priors and employs a dual-branch architecture to disentangle dynamic and static facial components effectively. The dynamic branch encodes expression-driven fine details, while the static branch focuses on expression-invariant regions, enabling efficient parameter inference and precomputation. This design maximizes the utility of limited 3D data and achieves real-time performance for animation and rendering. Additionally, SEGA performs person-specific fine-tuning to further enhance the fidelity and realism of the generated avatars. Experiments show our method outperforms state-of-the-art approaches in generalization ability, identity preservation, and expression realism, advancing one-shot avatar creation for practical applications.
>
---
#### [replaced 086] Class Incremental Learning with Task-Specific Batch Normalization and Out-of-Distribution Detection
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2411.00430](https://arxiv.org/pdf/2411.00430)**

> **作者:** Zhiping Zhou; Xuchen Xie; Yiqiao Qiu; Run Lin; Weishi Zheng; Ruixuan Wang
>
> **备注:** accepted by Neurocomputing Journal, camera ready version
>
> **摘要:** This study focuses on incremental learning for image classification, exploring how to reduce catastrophic forgetting of all learned knowledge when access to old data is restricted. The challenge lies in balancing plasticity (learning new knowledge) and stability (retaining old knowledge). Based on whether the task identifier (task-ID) is available during testing, incremental learning is divided into task incremental learning (TIL) and class incremental learning (CIL). The TIL paradigm often uses multiple classifier heads, selecting the corresponding head based on the task-ID. Since the CIL paradigm cannot access task-ID, methods originally developed for TIL require explicit task-ID prediction to bridge this gap and enable their adaptation to the CIL paradigm. {In this study, a novel continual learning framework extends the TIL method for CIL by introducing out-of-distribution detection for task-ID prediction. Our framework utilizes task-specific Batch Normalization (BN) and task-specific classification heads to effectively adjust feature map distributions for each task, enhancing plasticity. With far fewer parameters than convolutional kernels, task-specific BN helps minimize parameter growth, preserving stability. Based on multiple task-specific classification heads, we introduce an ``unknow'' class for each head. During training, data from other tasks are mapped to this unknown class. During inference, the task-ID is predicted by selecting the classification head with the lowest probability assigned to the unknown class. Our method achieves state-of-the-art performance on two medical image datasets and two natural image datasets. The source code is available at this https URL.
>
---
#### [replaced 087] REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.16410](https://arxiv.org/pdf/2510.16410)**

> **作者:** Changyue Shi; Minghao Chen; Yiping Mao; Chuxiao Yang; Xinyuan Hu; Jiajun Ding; Zhou Yu
>
> **备注:** CVPR 2026 Accepted
>
> **摘要:** Bridging the gap between complex human instructions and precise 3D object grounding remains a significant challenge in vision and robotics. Existing 3D segmentation methods often struggle to interpret ambiguous, reasoning-based instructions, while 2D vision-language models that excel at such reasoning lack intrinsic 3D spatial understanding. In this paper, we introduce REALM, an innovative MLLM-agent framework that enables open-world reasoning-based segmentation without requiring extensive 3D-specific post-training. We perform segmentation directly on 3D Gaussian Splatting representations, capitalizing on their ability to render photorealistic novel views that are highly suitable for MLLM comprehension. As directly feeding one or more rendered views to the MLLM can lead to high sensitivity to viewpoint selection, we propose a novel Global-to-Local Spatial Grounding strategy. Specifically, multiple global views are first fed into the MLLM agent in parallel for coarse-level localization, aggregating responses to robustly identify the target object. Then, several close-up novel views of the object are synthesized to perform fine-grained local segmentation, yielding accurate and consistent 3D masks. Extensive experiments show that REALM achieves remarkable performance in interpreting both explicit and implicit instructions across LERF, 3D-OVS, and our newly introduced REALM3D benchmarks. Furthermore, our agent framework seamlessly supports a range of 3D interaction tasks, including object removal, replacement, and style transfer, demonstrating its practical utility and versatility. Project page: this https URL.
>
---
#### [replaced 088] World Models That Know When They Don't Know - Controllable Video Generation with Calibrated Uncertainty
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视频生成任务，旨在解决可控视频模型 hallucinate 的问题。通过引入不确定性量化方法，提升模型对不确定性的估计能力，增强可靠性。**

- **链接: [https://arxiv.org/pdf/2512.05927](https://arxiv.org/pdf/2512.05927)**

> **作者:** Zhiting Mei; Tenny Yin; Micah Baker; Ola Shorinwa; Anirudha Majumdar
>
> **摘要:** Recent advances in generative video models have led to significant breakthroughs in high-fidelity video synthesis, specifically in controllable video generation where the generated video is conditioned on text and action inputs, e.g., in instruction-guided video editing and world modeling in robotics. Despite these exceptional capabilities, controllable video models often hallucinate - generating future video frames that are misaligned with physical reality - which raises serious concerns in many tasks such as robot policy evaluation and planning. However, state-of-the-art video models lack the ability to assess and express their confidence, impeding hallucination mitigation. To rigorously address this challenge, we propose C3, an uncertainty quantification (UQ) method for training continuous-scale calibrated controllable video models for dense confidence estimation at the subpatch level, precisely localizing the uncertainty in each generated video frame. Our UQ method introduces three core innovations to empower video models to estimate their uncertainty. First, our method develops a novel framework that trains video models for correctness and calibration via strictly proper scoring rules. Second, we estimate the video model's uncertainty in latent space, avoiding training instability and prohibitive training costs associated with pixel-space approaches. Third, we map the dense latent-space uncertainty to interpretable pixel-level uncertainty in the RGB space for intuitive visualization, providing high-resolution uncertainty heatmaps that identify untrustworthy regions. Through extensive experiments on large-scale robot learning datasets (Bridge and DROID) and real-world evaluations, we demonstrate that our method not only provides calibrated uncertainty estimates within the training distribution, but also enables effective out-of-distribution detection.
>
---
#### [replaced 089] SPIRAL: A Closed-Loop Framework for Self-Improving Action World Models via Reflective Planning Agents
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.08403](https://arxiv.org/pdf/2603.08403)**

> **作者:** Yu Yang; Yue Liao; Jianbiao Mei; Baisen Wang; Xuemeng Yang; Licheng Wen; Jiangning Zhang; Xiangtai Li; Hanlin Chen; Botian Shi; Yong Liu; Shuicheng Yan; Gim Hee Lee
>
> **备注:** 22 Pages, 11 Figures
>
> **摘要:** We introduce SPIRAL, a self-improving planning and iterative reflective action world modeling closed-loop framework that enables controllable long-horizon video generation conditioned on high-level semantic actions. Existing one-shot video generation models operate in open-loop, often resulting in incomplete action execution, weak semantic grounding, and temporal drift. SPIRAL formulates ActWM as a closed-loop think-act-reflect process, where generation proceeds step by step under explicit planning and feedback. A PlanAgent decomposes abstract actions into object-centric sub-actions, while a CriticAgent evaluates intermediate results and guides iterative refinement with long-horizon memory. This closed-loop design naturally supports RL evolving optimization, improving semantic alignment and temporal consistency over extended horizons. We further introduce the ActWM-Dataset and ActWM-Bench for training and evaluation. Experiments across multiple TI2V backbones demonstrate consistent gains on ActWM-Bench and mainstream video generation benchmarks, validating SPIRAL's effectiveness.
>
---
#### [replaced 090] A Systematic Comparison of Training Objectives for Out-of-Distribution Detection in Image Classification
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.07571](https://arxiv.org/pdf/2603.07571)**

> **作者:** Furkan Genç; Onat Özdemir; Emre Akbaş
>
> **摘要:** Out-of-distribution (OOD) detection is critical in safety-sensitive applications. While this challenge has been addressed from various perspectives, the influence of training objectives on OOD behavior remains comparatively underexplored. In this paper, we present a systematic comparison of four widely used training objectives: Cross-Entropy Loss, Prototype Loss, Triplet Loss, and Average Precision (AP) Loss, spanning probabilistic, prototype-based, metric-learning, and ranking-based supervision, for OOD detection in image classification under standardized OpenOOD protocols. Across CIFAR-10/100 and ImageNet-200, we find that Cross-Entropy Loss, Prototype Loss, and AP Loss achieve comparable in-distribution accuracy, while Cross-Entropy Loss provides the most consistent near- and far-OOD performance overall; the other objectives can be competitive in specific settings.
>
---
#### [replaced 091] UrbanAlign: Post-hoc Semantic Calibration for VLM-Human Preference Alignment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.19442](https://arxiv.org/pdf/2602.19442)**

> **作者:** Yecheng Zhang; Rong Zhao; Zhizhou Sha; Yong Li; Lei Wang; Ce Hou; Wen Ji; Hao Huang; Yunshan Wan; Jian Yu; Junhao Xia; Yuru Zhang; Chunlei Shi
>
> **备注:** 26 pages
>
> **摘要:** Vision-language models (VLMs) can describe urban scenes in rich detail, yet consistently fail to produce reliable human preference labels in domain-specific tasks such as safety assessment and aesthetic evaluation. The standard fix, fine-tuning or RLHF, requires large-scale annotations and model retraining. We ask a different question: can a frozen VLM be aligned with human preferences without modifying any weights? Our key insight is that VLMs are strong concept extractors but poor decision calibrators. We propose a three-stage post-hoc pipeline that exploits this asymmetry: (i) interpretable evaluation dimensions are automatically mined from consensus exemplars; (ii) an Observer-Debater-Judge chain extracts robust concept scores from the frozen VLM; and (iii) locally-weighted ridge regression on a hybrid manifold calibrates these scores to human ratings. Applied as UrbanAlign on Place Pulse 2.0, the framework reaches 72.2% accuracy (kappa=0.45) across six perception categories, outperforming all baselines by +11.0 pp and zero-shot VLM by +15.5 pp, with full interpretability and zero weight modification.
>
---
#### [replaced 092] Equivariant Splitting: Self-supervised learning from incomplete data
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.00929](https://arxiv.org/pdf/2510.00929)**

> **作者:** Victor Sechaud; Jérémy Scanvic; Quentin Barthélemy; Patrice Abry; Julián Tachella
>
> **摘要:** Self-supervised learning for inverse problems allows to train a reconstruction network from noise and/or incomplete data alone. These methods have the potential of enabling learning-based solutions when obtaining ground-truth references for training is expensive or even impossible. In this paper, we propose a new self-supervised learning strategy devised for the challenging setting where measurements are observed via a single incomplete observation model. We introduce a new definition of equivariance in the context of reconstruction networks, and show that the combination of self-supervised splitting losses and equivariant reconstruction networks results in unbiased estimates of the supervised loss. Through a series of experiments on image inpainting, accelerated magnetic resonance imaging, sparse-view computed tomography, and compressive sensing, we demonstrate that the proposed loss achieves state-of-the-art performance in settings with highly rank-deficient forward models. The code is available at this https URL
>
---
#### [replaced 093] MA-EgoQA: Question Answering over Egocentric Videos from Multiple Embodied Agents
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.09827](https://arxiv.org/pdf/2603.09827)**

> **作者:** Kangsan Kim; Yanlai Yang; Suji Kim; Woongyeong Yeo; Youngwan Lee; Mengye Ren; Sung Ju Hwang
>
> **备注:** Under review
>
> **摘要:** As embodied models become powerful, humans will collaborate with multiple embodied AI agents at their workplace or home in the future. To ensure better communication between human users and the multi-agent system, it is crucial to interpret incoming information from agents in parallel and refer to the appropriate context for each query. Existing challenges include effectively compressing and communicating high volumes of individual sensory inputs in the form of video and correctly aggregating multiple egocentric videos to construct system-level memory. In this work, we first formally define a novel problem of understanding multiple long-horizon egocentric videos simultaneously collected from embodied agents. To facilitate research in this direction, we introduce MultiAgent-EgoQA (MA-EgoQA), a benchmark designed to systemically evaluate existing models in our scenario. MA-EgoQA provides 1.7k questions unique to multiple egocentric streams, spanning five categories: social interaction, task coordination, theory-of-mind, temporal reasoning, and environmental interaction. We further propose a simple baseline model for MA-EgoQA named EgoMAS, which leverages shared memory across embodied agents and agent-wise dynamic retrieval. Through comprehensive evaluation across diverse baselines and EgoMAS on MA-EgoQA, we find that current approaches are unable to effectively handle multiple egocentric streams, highlighting the need for future advances in system-level understanding across the agents. The code and benchmark are available at this https URL.
>
---
