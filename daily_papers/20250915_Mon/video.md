# 计算机视觉 cs.CV

- **最新发布 73 篇**

- **更新 46 篇**

## 最新发布

#### [new 001] LayerLock: Non-collapsing Representation Learning with Progressive Freezing
- **分类: cs.CV**

- **简介: 论文提出LayerLock方法，用于自监督视觉表征学习，通过渐进式冻结网络层解决表征崩溃问题。该方法在视频MAE模型中按深度逐步冻结层，提升训练效率并优化潜在空间预测，应用于大模型取得优异结果。属于视觉表征学习任务。**

- **链接: [http://arxiv.org/pdf/2509.10156v1](http://arxiv.org/pdf/2509.10156v1)**

> **作者:** Goker Erdogan; Nikhil Parthasarathy; Catalin Ionescu; Drew Hudson; Alexander Lerchner; Andrew Zisserman; Mehdi Sajjadi; Joao Carreira
>
> **备注:** ICCV 2025
>
> **摘要:** We introduce LayerLock, a simple yet effective approach for self-supervised visual representation learning, that gradually transitions from pixel to latent prediction through progressive layer freezing. First, we make the observation that during training of video masked-autoencoding (MAE) models, ViT layers converge in the order of their depth: shallower layers converge early, deeper layers converge late. We then show that this observation can be exploited to accelerate standard MAE by progressively freezing the model according to an explicit schedule, throughout training. Furthermore, this same schedule can be used in a simple and scalable approach to latent prediction that does not suffer from "representation collapse". We apply our proposed approach, LayerLock, to large models of up to 4B parameters with results surpassing those of non-latent masked prediction on the 4DS perception suite.
>
---
#### [new 002] BEVTraj: Map-Free End-to-End Trajectory Prediction in Bird's-Eye View with Deformable Attention and Sparse Goal Proposals
- **分类: cs.CV; I.2.9; I.4.8**

- **简介: 该论文提出BEVTraj，用于自动驾驶中的轨迹预测任务。其旨在无需依赖预建高清地图的情况下提升预测精度。通过可变形注意力和稀疏目标候选模块，实现端到端预测，提高灵活性与适应性。**

- **链接: [http://arxiv.org/pdf/2509.10080v1](http://arxiv.org/pdf/2509.10080v1)**

> **作者:** Minsang Kong; Myeongjun Kim; Sang Gu Kang; Sang Hun Lee
>
> **备注:** Submitted to IEEE Transactions on Intelligent Transportation Systems (under review)
>
> **摘要:** In autonomous driving, trajectory prediction is essential for ensuring safe and efficient navigation. To improve prediction accuracy, recent approaches often rely on pre-built high-definition (HD) maps or real-time local map construction modules to incorporate static environmental information. However, pre-built HD maps are limited to specific regions and cannot adapt to transient changes. In addition, local map construction modules, which recognize only predefined elements, may fail to capture critical scene details or introduce errors that degrade prediction performance. To overcome these limitations, we propose Bird's-Eye View Trajectory Prediction (BEVTraj), a novel trajectory prediction framework that operates directly in the bird's-eye view (BEV) space utilizing real-time sensor data without relying on any pre-built maps. The BEVTraj leverages deformable attention to efficiently extract relevant context from dense BEV features. Furthermore, we introduce a Sparse Goal Candidate Proposal (SGCP) module, which enables full end-to-end prediction without requiring any post-processing steps. Extensive experiments demonstrate that the BEVTraj achieves performance comparable to state-of-the-art HD map-based models while offering greater flexibility by eliminating the dependency on pre-built maps. The source code is available at https://github.com/Kongminsang/bevtraj.
>
---
#### [new 003] Early Detection of Visual Impairments at Home Using a Smartphone Red-Eye Reflex Test
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出一种基于智能手机的红眼反射检测方法，用于家庭早期发现儿童视力问题。任务是实现无需专业设备的视力筛查。通过深度学习模型，准确率达90%，并优化数据采集条件，提供用户反馈，推动全球儿童视力早筛与干预。**

- **链接: [http://arxiv.org/pdf/2509.09808v1](http://arxiv.org/pdf/2509.09808v1)**

> **作者:** Judith Massmann; Alexander Lichtenstein; Francisco M. López
>
> **备注:** Accepted at IEEE ICDL 2025. 6 pages, 7 figures, 2 tables
>
> **摘要:** Numerous visual impairments can be detected in red-eye reflex images from young children. The so-called Bruckner test is traditionally performed by ophthalmologists in clinical settings. Thanks to the recent technological advances in smartphones and artificial intelligence, it is now possible to recreate the Bruckner test using a mobile device. In this paper, we present a first study conducted during the development of KidsVisionCheck, a free application that can perform vision screening with a mobile device using red-eye reflex images. The underlying model relies on deep neural networks trained on children's pupil images collected and labeled by an ophthalmologist. With an accuracy of 90% on unseen test data, our model provides highly reliable performance without the necessity of specialist equipment. Furthermore, we can identify the optimal conditions for data collection, which can in turn be used to provide immediate feedback to the users. In summary, this work marks a first step toward accessible pediatric vision screenings and early intervention for vision abnormalities worldwide.
>
---
#### [new 004] A Co-Training Semi-Supervised Framework Using Faster R-CNN and YOLO Networks for Object Detection in Densely Packed Retail Images
- **分类: cs.CV; cs.AI**

- **简介: 论文提出一种半监督协同训练框架，结合Faster R-CNN与YOLO网络，解决零售环境中密集商品图像目标检测问题。通过伪标签交换与集成分类器提升检测精度，降低标注依赖，适用于库存管理等实际场景。**

- **链接: [http://arxiv.org/pdf/2509.09750v1](http://arxiv.org/pdf/2509.09750v1)**

> **作者:** Hossein Yazdanjouei; Arash Mansouri; Mohammad Shokouhifar
>
> **摘要:** This study proposes a semi-supervised co-training framework for object detection in densely packed retail environments, where limited labeled data and complex conditions pose major challenges. The framework combines Faster R-CNN (utilizing a ResNet backbone) for precise localization with YOLO (employing a Darknet backbone) for global context, enabling mutual pseudo-label exchange that improves accuracy in scenes with occlusion and overlapping objects. To strengthen classification, it employs an ensemble of XGBoost, Random Forest, and SVM, utilizing diverse feature representations for higher robustness. Hyperparameters are optimized using a metaheuristic-driven algorithm, enhancing precision and efficiency across models. By minimizing reliance on manual labeling, the approach reduces annotation costs and adapts effectively to frequent product and layout changes common in retail. Experiments on the SKU-110k dataset demonstrate strong performance, highlighting the scalability and practicality of the proposed framework for real-world retail applications such as automated inventory tracking, product monitoring, and checkout systems.
>
---
#### [new 005] A Lightweight Ensemble-Based Face Image Quality Assessment Method with Correlation-Aware Loss
- **分类: cs.CV**

- **简介: 该论文提出一种轻量级的面向真实场景的人脸图像质量评估方法，解决现有方法计算复杂且难以捕捉人脸特有退化的问题。通过集成MobileNetV3-Small和ShuffleNetV2，并引入相关性感知损失函数，实现高精度与低计算成本的平衡。**

- **链接: [http://arxiv.org/pdf/2509.10114v1](http://arxiv.org/pdf/2509.10114v1)**

> **作者:** MohammadAli Hamidi; Hadi Amirpour; Luigi Atzori; Christian Timmerer
>
> **摘要:** Face image quality assessment (FIQA) plays a critical role in face recognition and verification systems, especially in uncontrolled, real-world environments. Although several methods have been proposed, general-purpose no-reference image quality assessment techniques often fail to capture face-specific degradations. Meanwhile, state-of-the-art FIQA models tend to be computationally intensive, limiting their practical applicability. We propose a lightweight and efficient method for FIQA, designed for the perceptual evaluation of face images in the wild. Our approach integrates an ensemble of two compact convolutional neural networks, MobileNetV3-Small and ShuffleNetV2, with prediction-level fusion via simple averaging. To enhance alignment with human perceptual judgments, we employ a correlation-aware loss (MSECorrLoss), combining mean squared error (MSE) with a Pearson correlation regularizer. Our method achieves a strong balance between accuracy and computational cost, making it suitable for real-world deployment. Experiments on the VQualA FIQA benchmark demonstrate that our model achieves a Spearman rank correlation coefficient (SRCC) of 0.9829 and a Pearson linear correlation coefficient (PLCC) of 0.9894, remaining within competition efficiency constraints.
>
---
#### [new 006] Ordinality of Visible-Thermal Image Intensities for Intrinsic Image Decomposition
- **分类: cs.CV**

- **简介: 论文提出一种无需训练的内在图像分解方法，利用可见光与热成像图像强度的序关系，解决真实场景中缺乏标注数据的问题，实现对光照与反射率的恢复。**

- **链接: [http://arxiv.org/pdf/2509.10388v1](http://arxiv.org/pdf/2509.10388v1)**

> **作者:** Zeqing Leo Yuan; Mani Ramanagopal; Aswin C. Sankaranarayanan; Srinivasa G. Narasimhan
>
> **摘要:** Decomposing an image into its intrinsic photometric factors--shading and reflectance--is a long-standing challenge due to the lack of extensive ground-truth data for real-world scenes. Recent methods rely on synthetic data or sparse annotations for limited indoor and even fewer outdoor scenes. We introduce a novel training-free approach for intrinsic image decomposition using only a pair of visible and thermal images. We leverage the principle that light not reflected from an opaque surface is absorbed and detected as heat by a thermal camera. This allows us to relate the ordinalities between visible and thermal image intensities to the ordinalities of shading and reflectance, which can densely self-supervise an optimizing neural network to recover shading and reflectance. We perform quantitative evaluations with known reflectance and shading under natural and artificial lighting, and qualitative experiments across diverse outdoor scenes. The results demonstrate superior performance over recent learning-based models and point toward a scalable path to curating real-world ordinal supervision, previously infeasible via manual labeling.
>
---
#### [new 007] GLAM: Geometry-Guided Local Alignment for Multi-View VLP in Mammography
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出GLAM模型，用于改进多视角乳腺X光图像的视觉语言预训练。针对现有模型忽略医学影像领域特性的缺陷，利用几何先验知识建模跨视图对齐与局部特征，提升预测性能。**

- **链接: [http://arxiv.org/pdf/2509.10344v1](http://arxiv.org/pdf/2509.10344v1)**

> **作者:** Yuexi Du; Lihui Chen; Nicha C. Dvornek
>
> **备注:** Accepted by MICCAI 2025
>
> **摘要:** Mammography screening is an essential tool for early detection of breast cancer. The speed and accuracy of mammography interpretation have the potential to be improved with deep learning methods. However, the development of a foundation visual language model (VLM) is hindered by limited data and domain differences between natural and medical images. Existing mammography VLMs, adapted from natural images, often ignore domain-specific characteristics, such as multi-view relationships in mammography. Unlike radiologists who analyze both views together to process ipsilateral correspondence, current methods treat them as independent images or do not properly model the multi-view correspondence learning, losing critical geometric context and resulting in suboptimal prediction. We propose GLAM: Global and Local Alignment for Multi-view mammography for VLM pretraining using geometry guidance. By leveraging the prior knowledge about the multi-view imaging process of mammograms, our model learns local cross-view alignments and fine-grained local features through joint global and local, visual-visual, and visual-language contrastive learning. Pretrained on EMBED [14], one of the largest open mammography datasets, our model outperforms baselines across multiple datasets under different settings.
>
---
#### [new 008] Multimodal SAM-adapter for Semantic Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于语义分割任务，旨在解决复杂环境下（如光照差、遮挡）分割性能下降的问题。提出MM SAM-adapter框架，通过融合多模态数据增强模型鲁棒性，在多个基准测试中取得最优效果。**

- **链接: [http://arxiv.org/pdf/2509.10408v1](http://arxiv.org/pdf/2509.10408v1)**

> **作者:** Iacopo Curti; Pierluigi Zama Ramirez; Alioscia Petrelli; Luigi Di Stefano
>
> **摘要:** Semantic segmentation, a key task in computer vision with broad applications in autonomous driving, medical imaging, and robotics, has advanced substantially with deep learning. Nevertheless, current approaches remain vulnerable to challenging conditions such as poor lighting, occlusions, and adverse weather. To address these limitations, multimodal methods that integrate auxiliary sensor data (e.g., LiDAR, infrared) have recently emerged, providing complementary information that enhances robustness. In this work, we present MM SAM-adapter, a novel framework that extends the capabilities of the Segment Anything Model (SAM) for multimodal semantic segmentation. The proposed method employs an adapter network that injects fused multimodal features into SAM's rich RGB features. This design enables the model to retain the strong generalization ability of RGB features while selectively incorporating auxiliary modalities only when they contribute additional cues. As a result, MM SAM-adapter achieves a balanced and efficient use of multimodal information. We evaluate our approach on three challenging benchmarks, DeLiVER, FMB, and MUSES, where MM SAM-adapter delivers state-of-the-art performance. To further analyze modality contributions, we partition DeLiVER and FMB into RGB-easy and RGB-hard subsets. Results consistently demonstrate that our framework outperforms competing methods in both favorable and adverse conditions, highlighting the effectiveness of multimodal adaptation for robust scene understanding. The code is available at the following link: https://github.com/iacopo97/Multimodal-SAM-Adapter.
>
---
#### [new 009] Fine-Grained Cross-View Localization via Local Feature Matching and Monocular Depth Priors
- **分类: cs.CV**

- **简介: 该论文提出一种细粒度跨视角定位方法，通过局部特征匹配和单目深度先验估计地面图像的3D姿态。解决传统方法因视角转换导致的信息丢失问题，实现高精度、强解释性的跨视角定位。**

- **链接: [http://arxiv.org/pdf/2509.09792v1](http://arxiv.org/pdf/2509.09792v1)**

> **作者:** Zimin Xia; Chenghao Xu; Alexandre Alahi
>
> **摘要:** We propose an accurate and highly interpretable fine-grained cross-view localization method that estimates the 3 Degrees of Freedom pose of a ground-level image by matching its local features with a reference aerial image. Previous methods typically transform the ground image into a bird's-eye view (BEV) representation and then align it with the aerial image for localization. However, this transformation often leads to information loss due to perspective distortion or compression of height information, thereby degrading alignment quality with the aerial view. In contrast, our method directly establishes correspondences between ground and aerial images and lifts only the matched keypoints to BEV space using monocular depth prior. Notably, modern depth predictors can provide reliable metric depth when the test samples are similar to the training data. When the depth distribution differs, they still produce consistent relative depth, i.e., depth accurate up to an unknown scale. Our method supports both metric and relative depth. It employs a scale-aware Procrustes alignment to estimate the camera pose from the correspondences and optionally recover the scale when using relative depth. Experimental results demonstrate that, with only weak supervision on camera pose, our method learns accurate local feature correspondences and achieves superior localization performance under challenging conditions, such as cross-area generalization and unknown orientation. Moreover, our method is compatible with various relative depth models without requiring per-model finetuning. This flexibility, combined with strong localization performance, makes it well-suited for real-world deployment.
>
---
#### [new 010] TUNI: Real-time RGB-T Semantic Segmentation with Unified Multi-Modal Feature Extraction and Cross-Modal Feature Fusion
- **分类: cs.CV**

- **简介: 该论文提出TUNI模型，用于RGB-T语义分割任务，解决传统方法中热成像特征提取不足、跨模态融合效果差及实时性低的问题。通过统一多模态特征提取与融合模块，提升性能并实现27 FPS的实时推理速度。**

- **链接: [http://arxiv.org/pdf/2509.10005v1](http://arxiv.org/pdf/2509.10005v1)**

> **作者:** Xiaodong Guo; Tong Liu; Yike Li; Zi'ang Lin; Zhihong Deng
>
> **摘要:** RGB-thermal (RGB-T) semantic segmentation improves the environmental perception of autonomous platforms in challenging conditions. Prevailing models employ encoders pre-trained on RGB images to extract features from both RGB and infrared inputs, and design additional modules to achieve cross-modal feature fusion. This results in limited thermal feature extraction and suboptimal cross-modal fusion, while the redundant encoders further compromises the model's real-time efficiency. To address the above issues, we propose TUNI, with an RGB-T encoder consisting of multiple stacked blocks that simultaneously perform multi-modal feature extraction and cross-modal fusion. By leveraging large-scale pre-training with RGB and pseudo-thermal data, the RGB-T encoder learns to integrate feature extraction and fusion in a unified manner. By slimming down the thermal branch, the encoder achieves a more compact architecture. Moreover, we introduce an RGB-T local module to strengthen the encoder's capacity for cross-modal local feature fusion. The RGB-T local module employs adaptive cosine similarity to selectively emphasize salient consistent and distinct local features across RGB-T modalities. Experimental results show that TUNI achieves competitive performance with state-of-the-art models on FMB, PST900 and CART, with fewer parameters and lower computational cost. Meanwhile, it achieves an inference speed of 27 FPS on a Jetson Orin NX, demonstrating its real-time capability in deployment. Codes are available at https://github.com/xiaodonguo/TUNI.
>
---
#### [new 011] Improving MLLM Historical Record Extraction with Test-Time Image
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于历史文档文本提取任务，旨在解决噪声文档中文字识别不准确的问题。提出一种集成框架，通过多图像变体增强与自定义对齐器融合，提升提取准确性，并验证了方法的有效性与可扩展性。**

- **链接: [http://arxiv.org/pdf/2509.09722v1](http://arxiv.org/pdf/2509.09722v1)**

> **作者:** Taylor Archibald; Tony Martinez
>
> **摘要:** We present a novel ensemble framework that stabilizes LLM based text extraction from noisy historical documents. We transcribe multiple augmented variants of each image with Gemini 2.0 Flash and fuse these outputs with a custom Needleman Wunsch style aligner that yields both a consensus transcription and a confidence score. We present a new dataset of 622 Pennsylvania death records, and demonstrate our method improves transcription accuracy by 4 percentage points relative to a single shot baseline. We find that padding and blurring are the most useful for improving accuracy, while grid warp perturbations are best for separating high and low confidence cases. The approach is simple, scalable, and immediately deployable to other document collections and transcription models.
>
---
#### [new 012] Augment to Segment: Tackling Pixel-Level Imbalance in Wheat Disease and Pest Segmentation
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务，旨在解决小麦病虫害分割中的像素级类别不平衡问题。提出RPCP增强方法，通过复制粘贴稀有损伤区域并随机变换，提升模型对罕见类的识别能力。**

- **链接: [http://arxiv.org/pdf/2509.09961v1](http://arxiv.org/pdf/2509.09961v1)**

> **作者:** Tianqi Wei; Xin Yu; Zhi Chen; Scott Chapman; Zi Huang
>
> **摘要:** Accurate segmentation of foliar diseases and insect damage in wheat is crucial for effective crop management and disease control. However, the insect damage typically occupies only a tiny fraction of annotated pixels. This extreme pixel-level imbalance poses a significant challenge to the segmentation performance, which can result in overfitting to common classes and insufficient learning of rare classes, thereby impairing overall performance. In this paper, we propose a Random Projected Copy-and-Paste (RPCP) augmentation technique to address the pixel imbalance problem. Specifically, we extract rare insect-damage patches from annotated training images and apply random geometric transformations to simulate variations. The transformed patches are then pasted in appropriate regions while avoiding overlaps with lesions or existing damaged regions. In addition, we apply a random projection filter to the pasted regions, refining local features and ensuring a natural blend with the new background. Experiments show that our method substantially improves segmentation performance on the insect damage class, while maintaining or even slightly enhancing accuracy on other categories. Our results highlight the effectiveness of targeted augmentation in mitigating extreme pixel imbalance, offering a straightforward yet effective solution for agricultural segmentation problems.
>
---
#### [new 013] Patch-based Automatic Rosacea Detection Using the ResNet Deep Learning Framework
- **分类: cs.CV**

- **简介: 该论文提出基于ResNet-18的补丁级自动检测系统，用于改善玫瑰痤疮的早期诊断。通过分析面部局部区域，提升模型准确性与隐私保护，解决传统全图方法在敏感性和解释性上的不足。**

- **链接: [http://arxiv.org/pdf/2509.09841v1](http://arxiv.org/pdf/2509.09841v1)**

> **作者:** Chengyu Yang; Rishik Reddy Yesgari; Chengjun Liu
>
> **摘要:** Rosacea, which is a chronic inflammatory skin condition that manifests with facial redness, papules, and visible blood vessels, often requirs precise and early detection for significantly improving treatment effectiveness. This paper presents new patch-based automatic rosacea detection strategies using the ResNet-18 deep learning framework. The contributions of the proposed strategies come from the following aspects. First, various image pateches are extracted from the facial images of people in different sizes, shapes, and locations. Second, a number of investigation studies are carried out to evaluate how the localized visual information influences the deep learing model performance. Third, thorough experiments are implemented to reveal that several patch-based automatic rosacea detection strategies achieve competitive or superior accuracy and sensitivity than the full-image based methods. And finally, the proposed patch-based strategies, which use only localized patches, inherently preserve patient privacy by excluding any identifiable facial features from the data. The experimental results indicate that the proposed patch-based strategies guide the deep learning model to focus on clinically relevant regions, enhance robustness and interpretability, and protect patient privacy. As a result, the proposed strategies offer practical insights for improving automated dermatological diagnostics.
>
---
#### [new 014] WAVE-DETR Multi-Modal Visible and Acoustic Real-Life Drone Detector
- **分类: cs.CV; cs.LG; 68W99**

- **简介: 该论文提出WAVE-DETR，一种融合可见光与声学信号的无人机检测模型，解决复杂环境下无人机检测难题。基于Deformable DETR和Wav2Vec2，采用四种融合方式提升检测性能，尤其在小尺寸无人机上mAP提升11.1%-15.3%。**

- **链接: [http://arxiv.org/pdf/2509.09859v1](http://arxiv.org/pdf/2509.09859v1)**

> **作者:** Razvan Stefanescu; Ethan Oh; Ruben Vazquez; Chris Mesterharm; Constantin Serban; Ritu Chadha
>
> **备注:** 11 pages, 11 figures
>
> **摘要:** We introduce a multi-modal WAVE-DETR drone detector combining visible RGB and acoustic signals for robust real-life UAV object detection. Our approach fuses visual and acoustic features in a unified object detector model relying on the Deformable DETR and Wav2Vec2 architectures, achieving strong performance under challenging environmental conditions. Our work leverage the existing Drone-vs-Bird dataset and the newly generated ARDrone dataset containing more than 7,500 synchronized images and audio segments. We show how the acoustic information is used to improve the performance of the Deformable DETR object detector on the real ARDrone dataset. We developed, trained and tested four different fusion configurations based on a gated mechanism, linear layer, MLP and cross attention. The Wav2Vec2 acoustic embeddings are fused with the multi resolution feature mappings of the Deformable DETR and enhance the object detection performance over all drones dimensions. The best performer is the gated fusion approach, which improves the mAP of the Deformable DETR object detector on our in-distribution and out-of-distribution ARDrone datasets by 11.1% to 15.3% for small drones across all IoU thresholds between 0.5 and 0.9. The mAP scores for medium and large drones are also enhanced, with overall gains across all drone sizes ranging from 3.27% to 5.84%.
>
---
#### [new 015] GARD: Gamma-based Anatomical Restoration and Denoising for Retinal OCT
- **分类: cs.CV**

- **简介: 该论文提出GARD方法，用于解决视网膜OCT图像的去噪问题。通过基于Gamma分布的扩散模型和保真项，有效去除斑点噪声，同时保留解剖细节，在PSNR、SSIM等指标上优于现有方法。属于医学图像去噪任务。**

- **链接: [http://arxiv.org/pdf/2509.10341v1](http://arxiv.org/pdf/2509.10341v1)**

> **作者:** Botond Fazekas; Thomas Pinetz; Guilherme Aresta; Taha Emre; Hrvoje Bogunovic
>
> **摘要:** Optical Coherence Tomography (OCT) is a vital imaging modality for diagnosing and monitoring retinal diseases. However, OCT images are inherently degraded by speckle noise, which obscures fine details and hinders accurate interpretation. While numerous denoising methods exist, many struggle to balance noise reduction with the preservation of crucial anatomical structures. This paper introduces GARD (Gamma-based Anatomical Restoration and Denoising), a novel deep learning approach for OCT image despeckling that leverages the strengths of diffusion probabilistic models. Unlike conventional diffusion models that assume Gaussian noise, GARD employs a Denoising Diffusion Gamma Model to more accurately reflect the statistical properties of speckle. Furthermore, we introduce a Noise-Reduced Fidelity Term that utilizes a pre-processed, less-noisy image to guide the denoising process. This crucial addition prevents the reintroduction of high-frequency noise. We accelerate the inference process by adapting the Denoising Diffusion Implicit Model framework to our Gamma-based model. Experiments on a dataset with paired noisy and less-noisy OCT B-scans demonstrate that GARD significantly outperforms traditional denoising methods and state-of-the-art deep learning models in terms of PSNR, SSIM, and MSE. Qualitative results confirm that GARD produces sharper edges and better preserves fine anatomical details.
>
---
#### [new 016] FLARE-SSM: Deep State Space Models with Influence-Balanced Loss for 72-Hour Solar Flare Prediction
- **分类: cs.CV; astro-ph.SR**

- **简介: 该论文属于太阳耀斑预测任务，旨在解决72小时内最大耀斑类别预测中的类别不平衡问题。提出基于多深度状态空间模型和FLARE损失函数的新方法，在多个指标上优于基线方法。**

- **链接: [http://arxiv.org/pdf/2509.09988v1](http://arxiv.org/pdf/2509.09988v1)**

> **作者:** Yusuke Takagi; Shunya Nagashima; Komei Sugiura
>
> **备注:** Accepted for presentation at ICONIP2025
>
> **摘要:** Accurate and reliable solar flare predictions are essential to mitigate potential impacts on critical infrastructure. However, the current performance of solar flare forecasting is insufficient. In this study, we address the task of predicting the class of the largest solar flare expected to occur within the next 72 hours. Existing methods often fail to adequately address the severe class imbalance across flare classes. To address this issue, we propose a solar flare prediction model based on multiple deep state space models. In addition, we introduce the frequency & local-boundary-aware reliability loss (FLARE loss) to improve predictive performance and reliability under class imbalance. Experiments were conducted on a multi-wavelength solar image dataset covering a full 11-year solar activity cycle. As a result, our method outperformed baseline approaches in terms of both the Gandin-Murphy-Gerrity score and the true skill statistic, which are standard metrics in terms of the performance and reliability.
>
---
#### [new 017] A Multimodal RAG Framework for Housing Damage Assessment: Collaborative Optimization of Image Encoding and Policy Vector Retrieval
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出一种多模态RAG框架，用于灾后房屋损毁评估。通过图像与文本分支协同优化，实现损毁图像理解与保险政策匹配，提升检索与分类性能。属于多模态信息检索与生成任务，解决灾后快速准确评估房屋损坏程度的问题。**

- **链接: [http://arxiv.org/pdf/2509.09721v1](http://arxiv.org/pdf/2509.09721v1)**

> **作者:** Jiayi Miao; Dingxin Lu; Zhuqi Wang
>
> **摘要:** After natural disasters, accurate evaluations of damage to housing are important for insurance claims response and planning of resources. In this work, we introduce a novel multimodal retrieval-augmented generation (MM-RAG) framework. On top of classical RAG architecture, we further the framework to devise a two-branch multimodal encoder structure that the image branch employs a visual encoder composed of ResNet and Transformer to extract the characteristic of building damage after disaster, and the text branch harnesses a BERT retriever for the text vectorization of posts as well as insurance policies and for the construction of a retrievable restoration index. To impose cross-modal semantic alignment, the model integrates a cross-modal interaction module to bridge the semantic representation between image and text via multi-head attention. Meanwhile, in the generation module, the introduced modal attention gating mechanism dynamically controls the role of visual evidence and text prior information during generation. The entire framework takes end-to-end training, and combines the comparison loss, the retrieval loss and the generation loss to form multi-task optimization objectives, and achieves image understanding and policy matching in collaborative learning. The results demonstrate superior performance in retrieval accuracy and classification index on damage severity, where the Top-1 retrieval accuracy has been improved by 9.6%.
>
---
#### [new 018] LaV-CoT: Language-Aware Visual CoT with Multi-Aspect Reward Optimization for Real-World Multilingual VQA
- **分类: cs.CV**

- **简介: 该论文提出LaV-CoT框架，解决多语言视觉问答（mVQA）中语言感知与多模态推理不足的问题。通过多阶段推理流程和多方面奖励优化，提升模型在多语言场景下的准确性和泛化能力，并在多个数据集上取得显著性能提升。**

- **链接: [http://arxiv.org/pdf/2509.10026v1](http://arxiv.org/pdf/2509.10026v1)**

> **作者:** Jing Huang; Zhiya Tan; Shutao Gong; Fanwei Zeng; Jianshu Li
>
> **备注:** 12 Pages, 12 Figures, 2 Tables
>
> **摘要:** As large vision language models (VLMs) advance, their capabilities in multilingual visual question answering (mVQA) have significantly improved. Chain-of-thought (CoT) reasoning has been proven to enhance interpretability and complex reasoning. However, most existing approaches rely primarily on textual CoT and provide limited support for multilingual multimodal reasoning, constraining their deployment in real-world applications. To address this gap, we introduce \textbf{LaV-CoT}, the first Language-aware Visual CoT framework with Multi-Aspect Reward Optimization. LaV-CoT incorporates an interpretable multi-stage reasoning pipeline consisting of Text Summary with Bounding Box (BBox), Language Identification, Spatial Object-level Captioning, and Step-by-step Logical Reasoning. Following this reasoning pipeline, we design an automated data curation method that generates multilingual CoT annotations through iterative generation, correction, and refinement, enabling scalable and high-quality training data. To improve reasoning and generalization, LaV-CoT adopts a two-stage training paradigm combining Supervised Fine-Tuning (SFT) with Language-aware Group Relative Policy Optimization (GRPO), guided by verifiable multi-aspect rewards including language consistency, structural accuracy, and semantic alignment. Extensive evaluations on public datasets including MMMB, Multilingual MMBench, and MTVQA show that LaV-CoT achieves up to \(\sim\)9.5\% accuracy improvements over open-source baselines of similar size and even surpasses models with 2$\times$ larger scales by \(\sim\)2.6\%. Moreover, LaV-CoT outperforms advanced proprietary models such as GPT-4o-0513 and Gemini-2.5-flash. We further conducted an online A/B test to validate our method on real-world data, highlighting its effectiveness for industrial deployment. Our code is available at this link: \href{https://github.com/HJNVR/LaV-CoT}
>
---
#### [new 019] Detecting Text Manipulation in Images using Vision Language Models
- **分类: cs.CV**

- **简介: 该论文研究图像中文字篡改检测任务，旨在解决现有视觉语言模型在文本篡改检测中的不足。通过对比开源与闭源模型，发现开源模型表现接近但仍有差距，并指出专用模型存在泛化问题，实验涵盖真实场景与幻想ID卡等复杂情况。**

- **链接: [http://arxiv.org/pdf/2509.10278v1](http://arxiv.org/pdf/2509.10278v1)**

> **作者:** Vidit Vidit; Pavel Korshunov; Amir Mohammadi; Christophe Ecabert; Ketan Kotwal; Sébastien Marcel
>
> **备注:** Accepted in Synthetic Realities and Biometric Security Workshop BMVC-2025. For paper page see https://www.idiap.ch/paper/textvlmdet/
>
> **摘要:** Recent works have shown the effectiveness of Large Vision Language Models (VLMs or LVLMs) in image manipulation detection. However, text manipulation detection is largely missing in these studies. We bridge this knowledge gap by analyzing closed- and open-source VLMs on different text manipulation datasets. Our results suggest that open-source models are getting closer, but still behind closed-source ones like GPT- 4o. Additionally, we benchmark image manipulation detection-specific VLMs for text manipulation detection and show that they suffer from the generalization problem. We benchmark VLMs for manipulations done on in-the-wild scene texts and on fantasy ID cards, where the latter mimic a challenging real-world misuse.
>
---
#### [new 020] An Autoencoder and Vision Transformer-based Interpretability Analysis of the Differences in Automated Staging of Second and Third Molars
- **分类: cs.CV; cs.AI; 68T07 (Primary)**

- **简介: 论文提出结合自编码器与视觉Transformer的框架，用于提升下颌第二、第三磨牙自动分期的准确性和可解释性。通过分析潜在空间和重建图像，揭示数据集内类变异性是性能瓶颈，为法医年龄估计提供更可靠的决策支持。**

- **链接: [http://arxiv.org/pdf/2509.09911v1](http://arxiv.org/pdf/2509.09911v1)**

> **作者:** Barkin Buyukcakir; Jannick De Tobel; Patrick Thevissen; Dirk Vandermeulen; Peter Claes
>
> **备注:** 21 pages, 11 figures, Scientific Reports
>
> **摘要:** The practical adoption of deep learning in high-stakes forensic applications, such as dental age estimation, is often limited by the 'black box' nature of the models. This study introduces a framework designed to enhance both performance and transparency in this context. We use a notable performance disparity in the automated staging of mandibular second (tooth 37) and third (tooth 38) molars as a case study. The proposed framework, which combines a convolutional autoencoder (AE) with a Vision Transformer (ViT), improves classification accuracy for both teeth over a baseline ViT, increasing from 0.712 to 0.815 for tooth 37 and from 0.462 to 0.543 for tooth 38. Beyond improving performance, the framework provides multi-faceted diagnostic insights. Analysis of the AE's latent space metrics and image reconstructions indicates that the remaining performance gap is data-centric, suggesting high intra-class morphological variability in the tooth 38 dataset is a primary limiting factor. This work highlights the insufficiency of relying on a single mode of interpretability, such as attention maps, which can appear anatomically plausible yet fail to identify underlying data issues. By offering a methodology that both enhances accuracy and provides evidence for why a model may be uncertain, this framework serves as a more robust tool to support expert decision-making in forensic age estimation.
>
---
#### [new 021] World Modeling with Probabilistic Structure Integration
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出Probabilistic Structure Integration（PSI）系统，用于从数据中学习可控且灵活提示的世界模型。通过概率预测、结构提取与集成三步骤，提取数据中的低维结构并增强模型能力，应用于视频预测与理解任务。**

- **链接: [http://arxiv.org/pdf/2509.09737v1](http://arxiv.org/pdf/2509.09737v1)**

> **作者:** Klemen Kotar; Wanhee Lee; Rahul Venkatesh; Honglin Chen; Daniel Bear; Jared Watrous; Simon Kim; Khai Loong Aw; Lilian Naing Chen; Stefan Stojanov; Kevin Feigelis; Imran Thobani; Alex Durango; Khaled Jedoui; Atlas Kazemian; Dan Yamins
>
> **摘要:** We present Probabilistic Structure Integration (PSI), a system for learning richly controllable and flexibly promptable world models from data. PSI consists of a three-step cycle. The first step, Probabilistic prediction, involves building a probabilistic graphical model Psi of the data, in the form of a random-access autoregressive sequence model. Psi supports a complete set of learned conditional distributions describing the dependence of any variables in the data on any other set of variables. In step 2, Structure extraction, we show how to extract underlying low-dimensional properties in the data, corresponding to a diverse set of meaningful "intermediate structures", in a zero-shot fashion via causal inference on Psi. Step 3, Integration, completes the cycle by converting these structures into new token types that are then continually mixed back into the training diet as conditioning signals and prediction targets. Each such cycle augments the capabilities of Psi, both allowing it to model the underlying data better, and creating new control handles -- akin to an LLM-like universal prompting language. We train an instance of Psi on 1.4 trillion tokens of internet video data; we use it to perform a variety of useful video prediction and understanding inferences; we extract state-of-the-art optical flow, self-supervised depth and object segmentation; and we use these structures to support a full cycle of predictive improvements.
>
---
#### [new 022] Compute Only 16 Tokens in One Timestep: Accelerating Diffusion Transformers with Cluster-Driven Feature Caching
- **分类: cs.CV**

- **简介: 该论文提出Cluster-Driven Feature Caching（ClusCa）方法，用于加速扩散Transformer模型。通过空间聚类减少计算量，提升生成效率，在文本到图像和视频任务中显著加速，无需额外训练。属于图像/视频生成任务，解决计算成本高的问题。**

- **链接: [http://arxiv.org/pdf/2509.10312v1](http://arxiv.org/pdf/2509.10312v1)**

> **作者:** Zhixin Zheng; Xinyu Wang; Chang Zou; Shaobo Wang; Linfeng Zhang
>
> **备注:** 11 pages, 11 figures; Accepted by ACM MM2025; Mainly focus on feature caching for diffusion transformers acceleration
>
> **摘要:** Diffusion transformers have gained significant attention in recent years for their ability to generate high-quality images and videos, yet still suffer from a huge computational cost due to their iterative denoising process. Recently, feature caching has been introduced to accelerate diffusion transformers by caching the feature computation in previous timesteps and reusing it in the following timesteps, which leverage the temporal similarity of diffusion models while ignoring the similarity in the spatial dimension. In this paper, we introduce Cluster-Driven Feature Caching (ClusCa) as an orthogonal and complementary perspective for previous feature caching. Specifically, ClusCa performs spatial clustering on tokens in each timestep, computes only one token in each cluster and propagates their information to all the other tokens, which is able to reduce the number of tokens by over 90%. Extensive experiments on DiT, FLUX and HunyuanVideo demonstrate its effectiveness in both text-to-image and text-to-video generation. Besides, it can be directly applied to any diffusion transformer without requirements for training. For instance, ClusCa achieves 4.96x acceleration on FLUX with an ImageReward of 99.49%, surpassing the original model by 0.51%. The code is available at https://github.com/Shenyi-Z/Cache4Diffusion.
>
---
#### [new 023] Adversarial robustness through Lipschitz-Guided Stochastic Depth in Neural Networks
- **分类: cs.CV**

- **简介: 论文提出一种基于Lipschitz常数的随机深度方法，用于提升神经网络对抗鲁棒性。该方法通过深度相关的DropPath策略，在保持准确率的同时降低计算量，有效增强模型对FGSM、PGD等攻击的防御能力。属于计算机视觉中的对抗样本防御任务。**

- **链接: [http://arxiv.org/pdf/2509.10298v1](http://arxiv.org/pdf/2509.10298v1)**

> **作者:** Laith Nayal; Mahmoud Mousatat; Bader Rasheed
>
> **备注:** 8 pages, 2 tables
>
> **摘要:** Deep neural networks and Vision Transformers achieve state-of-the-art performance in computer vision but are highly vulnerable to adversarial perturbations. Standard defenses often incur high computational cost or lack formal guarantees. We propose a Lipschitz-guided stochastic depth (DropPath) method, where drop probabilities increase with depth to control the effective Lipschitz constant of the network. This approach regularizes deeper layers, improving robustness while preserving clean accuracy and reducing computation. Experiments on CIFAR-10 with ViT-Tiny show that our custom depth-dependent schedule maintains near-baseline clean accuracy, enhances robustness under FGSM, PGD-20, and AutoAttack, and significantly reduces FLOPs compared to baseline and linear DropPath schedules.
>
---
#### [new 024] Towards Understanding Visual Grounding in Visual Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文综述了视觉语言模型中的视觉 grounding 任务，旨在解决文本与视觉内容对齐的问题。论文梳理了相关研究，分析了核心组件、应用场景及挑战，为未来研究提供方向。**

- **链接: [http://arxiv.org/pdf/2509.10345v1](http://arxiv.org/pdf/2509.10345v1)**

> **作者:** Georgios Pantazopoulos; Eda B. Özyiğit
>
> **摘要:** Visual grounding refers to the ability of a model to identify a region within some visual input that matches a textual description. Consequently, a model equipped with visual grounding capabilities can target a wide range of applications in various domains, including referring expression comprehension, answering questions pertinent to fine-grained details in images or videos, caption visual context by explicitly referring to entities, as well as low and high-level control in simulated and real environments. In this survey paper, we review representative works across the key areas of research on modern general-purpose vision language models (VLMs). We first outline the importance of grounding in VLMs, then delineate the core components of the contemporary paradigm for developing grounded models, and examine their practical applications, including benchmarks and evaluation metrics for grounded multimodal generation. We also discuss the multifaceted interrelations among visual grounding, multimodal chain-of-thought, and reasoning in VLMs. Finally, we analyse the challenges inherent to visual grounding and suggest promising directions for future research.
>
---
#### [new 025] Zero-Shot Referring Expression Comprehension via Visual-Language True/False Verification
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究零样本指代表达理解任务，旨在无需特定训练即可理解语言描述与图像区域的对应关系。提出通过视觉-语言真伪验证方法，利用通用检测器和VLM实现高精度匹配，优于现有零样本及微调方法。**

- **链接: [http://arxiv.org/pdf/2509.09958v1](http://arxiv.org/pdf/2509.09958v1)**

> **作者:** Jeffrey Liu; Rongbin Hu
>
> **摘要:** Referring Expression Comprehension (REC) is usually addressed with task-trained grounding models. We show that a zero-shot workflow, without any REC-specific training, can achieve competitive or superior performance. Our approach reformulates REC as box-wise visual-language verification: given proposals from a COCO-clean generic detector (YOLO-World), a general-purpose VLM independently answers True/False queries for each region. This simple procedure reduces cross-box interference, supports abstention and multiple matches, and requires no fine-tuning. On RefCOCO, RefCOCO+, and RefCOCOg, our method not only surpasses a zero-shot GroundingDINO baseline but also exceeds reported results for GroundingDINO trained on REC and GroundingDINO+CRG. Controlled studies with identical proposals confirm that verification significantly outperforms selection-based prompting, and results hold with open VLMs. Overall, we show that workflow design, rather than task-specific pretraining, drives strong zero-shot REC performance.
>
---
#### [new 026] Immunizing Images from Text to Image Editing via Adversarial Cross-Attention
- **分类: cs.CV**

- **简介: 该论文提出一种针对文本到图像编辑方法的对抗攻击——Attention Attack，通过生成源图像描述干扰跨模态对齐，降低编辑效果。属于图像编辑安全任务，解决编辑方法易受攻击的问题，提出新评估策略并验证攻击有效性。**

- **链接: [http://arxiv.org/pdf/2509.10359v1](http://arxiv.org/pdf/2509.10359v1)**

> **作者:** Matteo Trippodo; Federico Becattini; Lorenzo Seidenari
>
> **备注:** Accepted as Regular Paper at ACM Multimedia 2025
>
> **摘要:** Recent advances in text-based image editing have enabled fine-grained manipulation of visual content guided by natural language. However, such methods are susceptible to adversarial attacks. In this work, we propose a novel attack that targets the visual component of editing methods. We introduce Attention Attack, which disrupts the cross-attention between a textual prompt and the visual representation of the image by using an automatically generated caption of the source image as a proxy for the edit prompt. This breaks the alignment between the contents of the image and their textual description, without requiring knowledge of the editing method or the editing prompt. Reflecting on the reliability of existing metrics for immunization success, we propose two novel evaluation strategies: Caption Similarity, which quantifies semantic consistency between original and adversarial edits, and semantic Intersection over Union (IoU), which measures spatial layout disruption via segmentation masks. Experiments conducted on the TEDBench++ benchmark demonstrate that our attack significantly degrades editing performance while remaining imperceptible.
>
---
#### [new 027] Event Camera Guided Visual Media Restoration & 3D Reconstruction: A Survey
- **分类: cs.CV**

- **简介: 该论文综述了事件相机与传统视觉技术融合在视频修复与3D重建中的应用，重点探讨深度学习在时空增强和质量提升中的作用，并汇总公开数据集以推动可复现研究。**

- **链接: [http://arxiv.org/pdf/2509.09971v1](http://arxiv.org/pdf/2509.09971v1)**

> **作者:** Aupendu Kar; Vishnu Raj; Guan-Ming Su
>
> **摘要:** Event camera sensors are bio-inspired sensors which asynchronously capture per-pixel brightness changes and output a stream of events encoding the polarity, location and time of these changes. These systems are witnessing rapid advancements as an emerging field, driven by their low latency, reduced power consumption, and ultra-high capture rates. This survey explores the evolution of fusing event-stream captured with traditional frame-based capture, highlighting how this synergy significantly benefits various video restoration and 3D reconstruction tasks. The paper systematically reviews major deep learning contributions to image/video enhancement and restoration, focusing on two dimensions: temporal enhancement (such as frame interpolation and motion deblurring) and spatial enhancement (including super-resolution, low-light and HDR enhancement, and artifact reduction). This paper also explores how the 3D reconstruction domain evolves with the advancement of event driven fusion. Diverse topics are covered, with in-depth discussions on recent works for improving visual quality under challenging conditions. Additionally, the survey compiles a comprehensive list of openly available datasets, enabling reproducible research and benchmarking. By consolidating recent progress and insights, this survey aims to inspire further research into leveraging event camera systems, especially in combination with deep learning, for advanced visual media restoration and enhancement.
>
---
#### [new 028] Few-Part-Shot Font Generation
- **分类: cs.CV**

- **简介: 该论文属于字体生成任务，旨在解决传统方法需完整字符形状的问题。提出一种基于部分形状的少样本字体生成模型，提升效率并揭示局部设计对整体结构的影响。**

- **链接: [http://arxiv.org/pdf/2509.10006v1](http://arxiv.org/pdf/2509.10006v1)**

> **作者:** Masaki Akiba; Shumpei Takezaki; Daichi Haraguchi; Seiichi Uchida
>
> **备注:** ICDAR 2025 Workshop on Machine Learning
>
> **摘要:** This paper proposes a novel model of few-part-shot font generation, which designs an entire font based on a set of partial design elements, i.e., partial shapes. Unlike conventional few-shot font generation, which requires entire character shapes for a couple of character classes, our approach only needs partial shapes as input. The proposed model not only improves the efficiency of font creation but also provides insights into how partial design details influence the entire structure of the individual characters.
>
---
#### [new 029] Images in Motion?: A First Look into Video Leakage in Collaborative Deep Learning
- **分类: cs.CV**

- **简介: 该论文研究联邦学习中视频数据泄露问题，分析梯度反演攻击对视频分类模型的影响。通过实验发现特征提取器可增强安全性，但复杂度不足仍存在泄露风险，指出视频数据泄露是可行威胁，需进一步研究。**

- **链接: [http://arxiv.org/pdf/2509.09742v1](http://arxiv.org/pdf/2509.09742v1)**

> **作者:** Md Fazle Rasul; Alanood Alqobaisi; Bruhadeshwar Bezawada; Indrakshi Ray
>
> **摘要:** Federated learning (FL) allows multiple entities to train a shared model collaboratively. Its core, privacy-preserving principle is that participants only exchange model updates, such as gradients, and never their raw, sensitive data. This approach is fundamental for applications in domains where privacy and confidentiality are important. However, the security of this very mechanism is threatened by gradient inversion attacks, which can reverse-engineer private training data directly from the shared gradients, defeating the purpose of FL. While the impact of these attacks is known for image, text, and tabular data, their effect on video data remains an unexamined area of research. This paper presents the first analysis of video data leakage in FL using gradient inversion attacks. We evaluate two common video classification approaches: one employing pre-trained feature extractors and another that processes raw video frames with simple transformations. Our initial results indicate that the use of feature extractors offers greater resilience against gradient inversion attacks. We also demonstrate that image super-resolution techniques can enhance the frames extracted through gradient inversion attacks, enabling attackers to reconstruct higher-quality videos. Our experiments validate this across scenarios where the attacker has access to zero, one, or more reference frames from the target environment. We find that although feature extractors make attacks more challenging, leakage is still possible if the classifier lacks sufficient complexity. We, therefore, conclude that video data leakage in FL is a viable threat, and the conditions under which it occurs warrant further investigation.
>
---
#### [new 030] Privacy-Preserving Automated Rosacea Detection Based on Medically Inspired Region of Interest Selection
- **分类: cs.CV**

- **简介: 该论文提出一种隐私保护的自动化玫瑰痤疮检测方法，基于医学启发的感兴趣区域选择。通过合成数据训练ResNet-18模型，在保留诊断区域的同时去除身份信息，提升检测性能，解决隐私与数据稀缺问题。**

- **链接: [http://arxiv.org/pdf/2509.09844v1](http://arxiv.org/pdf/2509.09844v1)**

> **作者:** Chengyu Yang; Rishik Reddy Yesgari; Chengjun Liu
>
> **摘要:** Rosacea is a common but underdiagnosed inflammatory skin condition that primarily affects the central face and presents with subtle redness, pustules, and visible blood vessels. Automated detection remains challenging due to the diffuse nature of symptoms, the scarcity of labeled datasets, and privacy concerns associated with using identifiable facial images. A novel privacy-preserving automated rosacea detection method inspired by clinical priors and trained entirely on synthetic data is presented in this paper. Specifically, the proposed method, which leverages the observation that rosacea manifests predominantly through central facial erythema, first constructs a fixed redness-informed mask by selecting regions with consistently high red channel intensity across facial images. The mask thus is able to focus on diagnostically relevant areas such as the cheeks, nose, and forehead and exclude identity-revealing features. Second, the ResNet-18 deep learning method, which is trained on the masked synthetic images, achieves superior performance over the full-face baselines with notable gains in terms of accuracy, recall and F1 score when evaluated using the real-world test data. The experimental results demonstrate that the synthetic data and clinical priors can jointly enable accurate and ethical dermatological AI systems, especially for privacy sensitive applications in telemedicine and large-scale screening.
>
---
#### [new 031] VARCO-VISION-2.0 Technical Report
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出VARCO-VISION-2.0，一种支持韩英双语的视觉语言模型，提升多图像理解与布局感知OCR能力。通过四阶段训练和优化，增强多模态对齐与安全性，发布14B和1.7B两个版本，推动双语VLM发展与应用。**

- **链接: [http://arxiv.org/pdf/2509.10105v1](http://arxiv.org/pdf/2509.10105v1)**

> **作者:** Young-rok Cha; Jeongho Ju; SunYoung Park; Jong-Hyeon Lee; Younghyun Yu; Youngjune Kim
>
> **备注:** 19 pages, 1 figure, 14 tables. Technical report for VARCO-VISION-2.0, a Korean-English bilingual VLM in 14B and 1.7B variants. Key features: multi-image understanding, OCR with text localization, improved Korean capabilities
>
> **摘要:** We introduce VARCO-VISION-2.0, an open-weight bilingual vision-language model (VLM) for Korean and English with improved capabilities compared to the previous model VARCO-VISION-14B. The model supports multi-image understanding for complex inputs such as documents, charts, and tables, and delivers layoutaware OCR by predicting both textual content and its spatial location. Trained with a four-stage curriculum with memory-efficient techniques, the model achieves enhanced multimodal alignment, while preserving core language abilities and improving safety via preference optimization. Extensive benchmark evaluations demonstrate strong spatial grounding and competitive results for both languages, with the 14B model achieving 8th place on the OpenCompass VLM leaderboard among models of comparable scale. Alongside the 14B-scale model, we release a 1.7B version optimized for on-device deployment. We believe these models advance the development of bilingual VLMs and their practical applications. Two variants of VARCO-VISION-2.0 are available at Hugging Face: a full-scale 14B model and a lightweight 1.7B model.
>
---
#### [new 032] Grad-CL: Source Free Domain Adaptation with Gradient Guided Feature Disalignment
- **分类: cs.CV**

- **简介: 该论文提出Grad-CL，一种无需源数据的领域自适应框架，用于解决跨领域视网膜图像中视盘和杯分割性能下降的问题。通过梯度引导伪标签优化与对比学习策略，提升分割精度与边界描绘能力。**

- **链接: [http://arxiv.org/pdf/2509.10134v1](http://arxiv.org/pdf/2509.10134v1)**

> **作者:** Rini Smita Thakur; Rajeev Ranjan Dwivedi; Vinod K Kurmi
>
> **备注:** Accepted in BMVC 2025
>
> **摘要:** Accurate segmentation of the optic disc and cup is critical for the early diagnosis and management of ocular diseases such as glaucoma. However, segmentation models trained on one dataset often suffer significant performance degradation when applied to target data acquired under different imaging protocols or conditions. To address this challenge, we propose \textbf{Grad-CL}, a novel source-free domain adaptation framework that leverages a pre-trained source model and unlabeled target data to robustly adapt segmentation performance without requiring access to the original source data. Grad-CL combines a gradient-guided pseudolabel refinement module with a cosine similarity-based contrastive learning strategy. In the first stage, salient class-specific features are extracted via a gradient-based mechanism, enabling more accurate uncertainty quantification and robust prototype estimation for refining noisy pseudolabels. In the second stage, a contrastive loss based on cosine similarity is employed to explicitly enforce inter-class separability between the gradient-informed features of the optic cup and disc. Extensive experiments on challenging cross-domain fundus imaging datasets demonstrate that Grad-CL outperforms state-of-the-art unsupervised and source-free domain adaptation methods, achieving superior segmentation accuracy and improved boundary delineation. Project and code are available at https://visdomlab.github.io/GCL/.
>
---
#### [new 033] SSL-AD: Spatiotemporal Self-Supervised Learning for Generalizability and Adaptability Across Alzheimer's Prediction Tasks and Datasets
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出SSL-AD模型，用于阿尔茨海默病预测任务。针对数据标注不足、泛化能力差等问题，采用时序自监督学习方法，提升模型在不同数据集和任务中的适应性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.10453v1](http://arxiv.org/pdf/2509.10453v1)**

> **作者:** Emily Kaczmarek; Justin Szeto; Brennan Nichyporuk; Tal Arbel
>
> **摘要:** Alzheimer's disease is a progressive, neurodegenerative disorder that causes memory loss and cognitive decline. While there has been extensive research in applying deep learning models to Alzheimer's prediction tasks, these models remain limited by lack of available labeled data, poor generalization across datasets, and inflexibility to varying numbers of input scans and time intervals between scans. In this study, we adapt three state-of-the-art temporal self-supervised learning (SSL) approaches for 3D brain MRI analysis, and add novel extensions designed to handle variable-length inputs and learn robust spatial features. We aggregate four publicly available datasets comprising 3,161 patients for pre-training, and show the performance of our model across multiple Alzheimer's prediction tasks including diagnosis classification, conversion detection, and future conversion prediction. Importantly, our SSL model implemented with temporal order prediction and contrastive learning outperforms supervised learning on six out of seven downstream tasks. It demonstrates adaptability and generalizability across tasks and number of input images with varying time intervals, highlighting its capacity for robust performance across clinical applications. We release our code and model publicly at https://github.com/emilykaczmarek/SSL-AD.
>
---
#### [new 034] On the Geometric Accuracy of Implicit and Primitive-based Representations Derived from View Rendering Constraints
- **分类: cs.CV**

- **简介: 论文比较隐式与显式新视角合成方法在空间3D重建中的几何精度。研究发现外观嵌入虽提升图像质量，但对几何准确性无明显帮助，并指出凸分割比高斯分割更高效、适合航天应用。**

- **链接: [http://arxiv.org/pdf/2509.10241v1](http://arxiv.org/pdf/2509.10241v1)**

> **作者:** Elias De Smijter; Renaud Detry; Christophe De Vleeschouwer
>
> **备注:** 9 pages, 3 figures, to be presented at ASTRA25,
>
> **摘要:** We present the first systematic comparison of implicit and explicit Novel View Synthesis methods for space-based 3D object reconstruction, evaluating the role of appearance embeddings. While embeddings improve photometric fidelity by modeling lighting variation, we show they do not translate into meaningful gains in geometric accuracy - a critical requirement for space robotics applications. Using the SPEED+ dataset, we compare K-Planes, Gaussian Splatting, and Convex Splatting, and demonstrate that embeddings primarily reduce the number of primitives needed for explicit methods rather than enhancing geometric fidelity. Moreover, convex splatting achieves more compact and clutter-free representations than Gaussian splatting, offering advantages for safety-critical applications such as interaction and collision avoidance. Our findings clarify the limits of appearance embeddings for geometry-centric tasks and highlight trade-offs between reconstruction quality and representation efficiency in space scenarios.
>
---
#### [new 035] GAMMA: Generalizable Alignment via Multi-task and Manipulation-Augmented Training for AI-Generated Image Detection
- **分类: cs.CV**

- **简介: 论文提出GAMMA框架，用于检测AI生成图像。旨在解决现有检测器泛化能力差的问题，通过多任务训练和内容操控增强语义对齐，提升跨模型检测性能。**

- **链接: [http://arxiv.org/pdf/2509.10250v1](http://arxiv.org/pdf/2509.10250v1)**

> **作者:** Haozhen Yan; Yan Hong; Suning Lang; Jiahui Zhan; Yikun Ji; Yujie Gao; Jun Lan; Huijia Zhu; Weiqiang Wang; Jianfu Zhang
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** With generative models becoming increasingly sophisticated and diverse, detecting AI-generated images has become increasingly challenging. While existing AI-genereted Image detectors achieve promising performance on in-distribution generated images, their generalization to unseen generative models remains limited. This limitation is largely attributed to their reliance on generation-specific artifacts, such as stylistic priors and compression patterns. To address these limitations, we propose GAMMA, a novel training framework designed to reduce domain bias and enhance semantic alignment. GAMMA introduces diverse manipulation strategies, such as inpainting-based manipulation and semantics-preserving perturbations, to ensure consistency between manipulated and authentic content. We employ multi-task supervision with dual segmentation heads and a classification head, enabling pixel-level source attribution across diverse generative domains. In addition, a reverse cross-attention mechanism is introduced to allow the segmentation heads to guide and correct biased representations in the classification branch. Our method achieves state-of-the-art generalization performance on the GenImage benchmark, imporving accuracy by 5.8%, but also maintains strong robustness on newly released generative model such as GPT-4o.
>
---
#### [new 036] MITS: A Large-Scale Multimodal Benchmark Dataset for Intelligent Traffic Surveillance
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MITS，一个专为智能交通监控设计的多模态数据集，包含标注图像和问答对，用于提升大模型在交通场景中的性能。解决了ITS领域缺乏专用数据集的问题，显著提升了多个LMM在ITS任务中的表现。**

- **链接: [http://arxiv.org/pdf/2509.09730v1](http://arxiv.org/pdf/2509.09730v1)**

> **作者:** Kaikai Zhao; Zhaoxiang Liu; Peng Wang; Xin Wang; Zhicheng Ma; Yajun Xu; Wenjing Zhang; Yibing Nan; Kai Wang; Shiguo Lian
>
> **备注:** accepted by Image and Vision Computing
>
> **摘要:** General-domain large multimodal models (LMMs) have achieved significant advances in various image-text tasks. However, their performance in the Intelligent Traffic Surveillance (ITS) domain remains limited due to the absence of dedicated multimodal datasets. To address this gap, we introduce MITS (Multimodal Intelligent Traffic Surveillance), the first large-scale multimodal benchmark dataset specifically designed for ITS. MITS includes 170,400 independently collected real-world ITS images sourced from traffic surveillance cameras, annotated with eight main categories and 24 subcategories of ITS-specific objects and events under diverse environmental conditions. Additionally, through a systematic data generation pipeline, we generate high-quality image captions and 5 million instruction-following visual question-answer pairs, addressing five critical ITS tasks: object and event recognition, object counting, object localization, background analysis, and event reasoning. To demonstrate MITS's effectiveness, we fine-tune mainstream LMMs on this dataset, enabling the development of ITS-specific applications. Experimental results show that MITS significantly improves LMM performance in ITS applications, increasing LLaVA-1.5's performance from 0.494 to 0.905 (+83.2%), LLaVA-1.6's from 0.678 to 0.921 (+35.8%), Qwen2-VL's from 0.584 to 0.926 (+58.6%), and Qwen2.5-VL's from 0.732 to 0.930 (+27.0%). We release the dataset, code, and models as open-source, providing high-value resources to advance both ITS and LMM research.
>
---
#### [new 037] Efficient and Accurate Downfacing Visual Inertial Odometry
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文提出一种高效精确的下视视觉惯性里程计（VIO）方案，针对微纳无人机优化。采用先进特征检测方法并量化至RISC-V SoC，提升实时性与精度，降低计算需求，实现轻量化部署。**

- **链接: [http://arxiv.org/pdf/2509.10021v1](http://arxiv.org/pdf/2509.10021v1)**

> **作者:** Jonas Kühne; Christian Vogt; Michele Magno; Luca Benini
>
> **备注:** This article has been accepted for publication in the IEEE Internet of Things Journal (IoT-J)
>
> **摘要:** Visual Inertial Odometry (VIO) is a widely used computer vision method that determines an agent's movement through a camera and an IMU sensor. This paper presents an efficient and accurate VIO pipeline optimized for applications on micro- and nano-UAVs. The proposed design incorporates state-of-the-art feature detection and tracking methods (SuperPoint, PX4FLOW, ORB), all optimized and quantized for emerging RISC-V-based ultra-low-power parallel systems on chips (SoCs). Furthermore, by employing a rigid body motion model, the pipeline reduces estimation errors and achieves improved accuracy in planar motion scenarios. The pipeline's suitability for real-time VIO is assessed on an ultra-low-power SoC in terms of compute requirements and tracking accuracy after quantization. The pipeline, including the three feature tracking methods, was implemented on the SoC for real-world validation. This design bridges the gap between high-accuracy VIO pipelines that are traditionally run on computationally powerful systems and lightweight implementations suitable for microcontrollers. The optimized pipeline on the GAP9 low-power SoC demonstrates an average reduction in RMSE of up to a factor of 3.65x over the baseline pipeline when using the ORB feature tracker. The analysis of the computational complexity of the feature trackers further shows that PX4FLOW achieves on-par tracking accuracy with ORB at a lower runtime for movement speeds below 24 pixels/frame.
>
---
#### [new 038] SCoDA: Self-supervised Continual Domain Adaptation
- **分类: cs.CV**

- **简介: 论文提出SCoDA方法，解决无源域数据的领域适应问题。采用自监督预训练模型，结合几何流形对齐与空间相似性损失，提升目标域适应效果，避免灾难性遗忘。属于领域自适应任务。**

- **链接: [http://arxiv.org/pdf/2509.09935v1](http://arxiv.org/pdf/2509.09935v1)**

> **作者:** Chirayu Agrawal; Snehasis Mukherjee
>
> **备注:** Submitted to ICVGIP 2025
>
> **摘要:** Source-Free Domain Adaptation (SFDA) addresses the challenge of adapting a model to a target domain without access to the data of the source domain. Prevailing methods typically start with a source model pre-trained with full supervision and distill the knowledge by aligning instance-level features. However, these approaches, relying on cosine similarity over L2-normalized feature vectors, inadvertently discard crucial geometric information about the latent manifold of the source model. We introduce Self-supervised Continual Domain Adaptation (SCoDA) to address these limitations. We make two key departures from standard practice: first, we avoid the reliance on supervised pre-training by initializing the proposed framework with a teacher model pre-trained entirely via self-supervision (SSL). Second, we adapt the principle of geometric manifold alignment to the SFDA setting. The student is trained with a composite objective combining instance-level feature matching with a Space Similarity Loss. To combat catastrophic forgetting, the teacher's parameters are updated via an Exponential Moving Average (EMA) of the student's parameters. Extensive experiments on benchmark datasets demonstrate that SCoDA significantly outperforms state-of-the-art SFDA methods.
>
---
#### [new 039] An HMM-based framework for identity-aware long-term multi-object tracking from sparse and uncertain identification: use case on long-term tracking in livestock
- **分类: cs.CV**

- **简介: 该论文提出基于HMM的框架，解决长时多目标跟踪中因身份切换导致性能下降的问题。利用稀疏不确定的身份信息提升跟踪效果，并在猪只跟踪数据集及MOT基准上验证有效性。属于多目标跟踪任务。**

- **链接: [http://arxiv.org/pdf/2509.09962v1](http://arxiv.org/pdf/2509.09962v1)**

> **作者:** Anne Marthe Sophie Ngo Bibinbe; Chiron Bang; Patrick Gagnon; Jamie Ahloy-Dallaire; Eric R. Paquet
>
> **备注:** 13 pages, 7 figures, 1 table, accepted at CVPR animal workshop 2024, submitted to IJCV
>
> **摘要:** The need for long-term multi-object tracking (MOT) is growing due to the demand for analyzing individual behaviors in videos that span several minutes. Unfortunately, due to identity switches between objects, the tracking performance of existing MOT approaches decreases over time, making them difficult to apply for long-term tracking. However, in many real-world applications, such as in the livestock sector, it is possible to obtain sporadic identifications for some of the animals from sources like feeders. To address the challenges of long-term MOT, we propose a new framework that combines both uncertain identities and tracking using a Hidden Markov Model (HMM) formulation. In addition to providing real-world identities to animals, our HMM framework improves the F1 score of ByteTrack, a leading MOT approach even with re-identification, on a 10 minute pig tracking dataset with 21 identifications at the pen's feeding station. We also show that our approach is robust to the uncertainty of identifications, with performance increasing as identities are provided more frequently. The improved performance of our HMM framework was also validated on the MOT17 and MOT20 benchmark datasets using both ByteTrack and FairMOT. The code for this new HMM framework and the new 10-minute pig tracking video dataset are available at: https://github.com/ngobibibnbe/uncertain-identity-aware-tracking
>
---
#### [new 040] Australian Supermarket Object Set (ASOS): A Benchmark Dataset of Physical Objects and 3D Models for Robotics and Computer Vision
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文提出ASOS数据集，包含50种超市常见物品的高质量3D模型，用于机器人和计算机视觉的基准测试。旨在解决现实场景中物体检测与姿态估计问题，提供真实、易获取的数据支持相关研究。**

- **链接: [http://arxiv.org/pdf/2509.09720v1](http://arxiv.org/pdf/2509.09720v1)**

> **作者:** Akansel Cosgun; Lachlan Chumbley; Benjamin J. Meyer
>
> **摘要:** This paper introduces the Australian Supermarket Object Set (ASOS), a comprehensive dataset comprising 50 readily available supermarket items with high-quality 3D textured meshes designed for benchmarking in robotics and computer vision applications. Unlike existing datasets that rely on synthetic models or specialized objects with limited accessibility, ASOS provides a cost-effective collection of common household items that can be sourced from a major Australian supermarket chain. The dataset spans 10 distinct categories with diverse shapes, sizes, and weights. 3D meshes are acquired by a structure-from-motion techniques with high-resolution imaging to generate watertight meshes. The dataset's emphasis on accessibility and real-world applicability makes it valuable for benchmarking object detection, pose estimation, and robotics applications.
>
---
#### [new 041] Compressed Video Quality Enhancement: Classifying and Benchmarking over Standards
- **分类: cs.CV**

- **简介: 该论文属于视频质量增强任务，旨在解决现有CVQE方法分类不系统、评估不统一的问题。论文提出新分类体系、统一基准框架，并分析性能与复杂度的权衡，为研究和应用提供指导。**

- **链接: [http://arxiv.org/pdf/2509.10407v1](http://arxiv.org/pdf/2509.10407v1)**

> **作者:** Xiem HoangVan; Dang BuiDinh; Sang NguyenQuang; Wen-Hsiao Peng
>
> **摘要:** Compressed video quality enhancement (CVQE) is crucial for improving user experience with lossy video codecs like H.264/AVC, H.265/HEVC, and H.266/VVC. While deep learning based CVQE has driven significant progress, existing surveys still suffer from limitations: lack of systematic classification linking methods to specific standards and artifacts, insufficient comparative analysis of architectural paradigms across coding types, and underdeveloped benchmarking practices. To address these gaps, this paper presents three key contributions. First, it introduces a novel taxonomy classifying CVQE methods across architectural paradigms, coding standards, and compressed-domain feature utilization. Second, it proposes a unified benchmarking framework integrating modern compression protocols and standard test sequences for fair multi-criteria evaluation. Third, it provides a systematic analysis of the critical trade-offs between reconstruction performance and computational complexity observed in state-of-the-art methods and highlighting promising directions for future research. This comprehensive review aims to establish a foundation for consistent assessment and informed model selection in CVQE research and deployment.
>
---
#### [new 042] Investigating the Impact of Various Loss Functions and Learnable Wiener Filter for Laparoscopic Image Desmoking
- **分类: cs.CV**

- **简介: 论文研究腹腔镜图像去烟任务，评估ULW框架中各组件（如损失函数和可学习维纳滤波器）的有效性。通过消融实验分析各部分对性能的贡献，使用定量与定性指标进行评估。**

- **链接: [http://arxiv.org/pdf/2509.09849v1](http://arxiv.org/pdf/2509.09849v1)**

> **作者:** Chengyu Yang; Chengjun Liu
>
> **摘要:** To rigorously assess the effectiveness and necessity of individual components within the recently proposed ULW framework for laparoscopic image desmoking, this paper presents a comprehensive ablation study. The ULW approach combines a U-Net based backbone with a compound loss function that comprises mean squared error (MSE), structural similarity index (SSIM) loss, and perceptual loss. The framework also incorporates a differentiable, learnable Wiener filter module. In this study, each component is systematically ablated to evaluate its specific contribution to the overall performance of the whole framework. The analysis includes: (1) removal of the learnable Wiener filter, (2) selective use of individual loss terms from the composite loss function. All variants are benchmarked on a publicly available paired laparoscopic images dataset using quantitative metrics (SSIM, PSNR, MSE and CIEDE-2000) alongside qualitative visual comparisons.
>
---
#### [new 043] A Stochastic Birth-and-Death Approach for Street Furniture Geolocation in Urban Environments
- **分类: cs.CV**

- **简介: 论文提出一种基于能量图的随机出生-死亡优化算法，用于城市环境中街道家具的精确定位。该方法结合GIS信息提升定位精度，适用于公共设施的高效监测与维护。**

- **链接: [http://arxiv.org/pdf/2509.10310v1](http://arxiv.org/pdf/2509.10310v1)**

> **作者:** Evan Murphy; Marco Viola; Vladimir A. Krylov
>
> **备注:** Accepted for publication in the Proceedings of the 27th Irish Machine Vision and Image Processing Conference (IMVIP 2025)
>
> **摘要:** In this paper we address the problem of precise geolocation of street furniture in complex urban environments, which is a critical task for effective monitoring and maintenance of public infrastructure by local authorities and private stakeholders. To this end, we propose a probabilistic framework based on energy maps that encode the spatial likelihood of object locations. Representing the energy in a map-based geopositioned format allows the optimisation process to seamlessly integrate external geospatial information, such as GIS layers, road maps, or placement constraints, which improves contextual awareness and localisation accuracy. A stochastic birth-and-death optimisation algorithm is introduced to infer the most probable configuration of assets. We evaluate our approach using a realistic simulation informed by a geolocated dataset of street lighting infrastructure in Dublin city centre, demonstrating its potential for scalable and accurate urban asset mapping. The implementation of the algorithm will be made available in the GitHub repository https://github.com/EMurphy0108/SBD_Street_Furniture.
>
---
#### [new 044] Online 3D Multi-Camera Perception through Robust 2D Tracking and Depth-based Late Aggregation
- **分类: cs.CV**

- **简介: 该论文属于多目标多摄像头跟踪（MTMC）任务，旨在将现有2D跟踪系统扩展至3D空间。通过深度信息重建点云并聚类恢复3D框，结合增强的数据关联机制，实现在线3D感知，提升大规模监控的自动化水平。**

- **链接: [http://arxiv.org/pdf/2509.09946v1](http://arxiv.org/pdf/2509.09946v1)**

> **作者:** Vu-Minh Le; Thao-Anh Tran; Duc Huy Do; Xuan Canh Do; Huong Ninh; Hai Tran
>
> **备注:** Accepted at ICCVW 2025
>
> **摘要:** Multi-Target Multi-Camera Tracking (MTMC) is an essential computer vision task for automating large-scale surveillance. With camera calibration and depth information, the targets in the scene can be projected into 3D space, offering unparalleled levels of automatic perception of a 3D environment. However, tracking in the 3D space requires replacing all 2D tracking components from the ground up, which may be infeasible for existing MTMC systems. In this paper, we present an approach for extending any online 2D multi-camera tracking system into 3D space by utilizing depth information to reconstruct a target in point-cloud space, and recovering its 3D box through clustering and yaw refinement following tracking. We also introduced an enhanced online data association mechanism that leverages the target's local ID consistency to assign global IDs across frames. The proposed framework is evaluated on the 2025 AI City Challenge's 3D MTMC dataset, achieving 3rd place on the leaderboard.
>
---
#### [new 045] MCL-AD: Multimodal Collaboration Learning for Zero-Shot 3D Anomaly Detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出MCL-AD框架，解决零样本3D异常检测问题。通过融合点云、RGB图像和文本语义，利用多模态协作学习提升检测性能，引入MPLM和CMM机制，实现跨模态表征与协作优化，在ZS-3D任务中取得最优效果。**

- **链接: [http://arxiv.org/pdf/2509.10282v1](http://arxiv.org/pdf/2509.10282v1)**

> **作者:** Gang Li; Tianjiao Chen; Mingle Zhou; Min Li; Delong Han; Jin Wan
>
> **备注:** Page 14, 5 pictures
>
> **摘要:** Zero-shot 3D (ZS-3D) anomaly detection aims to identify defects in 3D objects without relying on labeled training data, making it especially valuable in scenarios constrained by data scarcity, privacy, or high annotation cost. However, most existing methods focus exclusively on point clouds, neglecting the rich semantic cues available from complementary modalities such as RGB images and texts priors. This paper introduces MCL-AD, a novel framework that leverages multimodal collaboration learning across point clouds, RGB images, and texts semantics to achieve superior zero-shot 3D anomaly detection. Specifically, we propose a Multimodal Prompt Learning Mechanism (MPLM) that enhances the intra-modal representation capability and inter-modal collaborative learning by introducing an object-agnostic decoupled text prompt and a multimodal contrastive loss. In addition, a collaborative modulation mechanism (CMM) is proposed to fully leverage the complementary representations of point clouds and RGB images by jointly modulating the RGB image-guided and point cloud-guided branches. Extensive experiments demonstrate that the proposed MCL-AD framework achieves state-of-the-art performance in ZS-3D anomaly detection.
>
---
#### [new 046] MagicMirror: A Large-Scale Dataset and Benchmark for Fine-Grained Artifacts Assessment in Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文提出MagicMirror框架，用于评估文本到图像生成中的细粒度缺陷。任务是解决生成图像中物理缺陷缺乏系统评估的问题，构建了标注数据集、训练评估模型并建立基准测试，推动T2I模型的缺陷优化。**

- **链接: [http://arxiv.org/pdf/2509.10260v1](http://arxiv.org/pdf/2509.10260v1)**

> **作者:** Jia Wang; Jie Hu; Xiaoqi Ma; Hanghang Ma; Yanbing Zeng; Xiaoming Wei
>
> **摘要:** Text-to-image (T2I) generation has achieved remarkable progress in instruction following and aesthetics. However, a persistent challenge is the prevalence of physical artifacts, such as anatomical and structural flaws, which severely degrade perceptual quality and limit application. Given the diversity and complexity of these artifacts, a systematic and fine-grained evaluation framework is required, which is lacking in current benchmarks. To fill this gap, we introduce MagicMirror, a comprehensive framework for artifacts assessment. We first establish a detailed taxonomy of generated image artifacts. Guided by this taxonomy, we manually annotate MagicData340K, the first human-annotated large-scale dataset of 340K generated images with fine-grained artifact labels. Building on this dataset, we train MagicAssessor, a Vision-Language Model (VLM) that provides detailed assessments and corresponding labels. To overcome challenges like class imbalance and reward hacking, we design a novel data sampling strategy and a multi-level reward system for Group Relative Policy Optimization (GRPO). Finally, we leverage MagicAssessor to construct MagicBench, an automated benchmark for evaluating the image artifacts of current T2I models. Our evaluation with MagicBench reveals that despite their widespread adoption, even top-tier models like GPT-image-1 are consistently plagued by significant artifacts, highlighting artifact reduction as a critical frontier for future T2I development. Project page: https://wj-inf.github.io/MagicMirror-page/.
>
---
#### [new 047] ISTASTrack: Bridging ANN and SNN via ISTA Adapter for RGB-Event Tracking
- **分类: cs.CV**

- **简介: 该论文提出ISTASTrack，一种基于Transformer的ANN-SNN混合跟踪器，用于RGB-Event视觉跟踪。通过ISTA适配器融合两种模态特征，解决异构范式下特征对齐难题，实现高性能与高能效的跟踪。**

- **链接: [http://arxiv.org/pdf/2509.09977v1](http://arxiv.org/pdf/2509.09977v1)**

> **作者:** Siying Liu; Zikai Wang; Hanle Zheng; Yifan Hu; Xilin Wang; Qingkai Yang; Jibin Wu; Hao Guo; Lei Deng
>
> **备注:** 15 pages, 8 figures
>
> **摘要:** RGB-Event tracking has become a promising trend in visual object tracking to leverage the complementary strengths of both RGB images and dynamic spike events for improved performance. However, existing artificial neural networks (ANNs) struggle to fully exploit the sparse and asynchronous nature of event streams. Recent efforts toward hybrid architectures combining ANNs and spiking neural networks (SNNs) have emerged as a promising solution in RGB-Event perception, yet effectively fusing features across heterogeneous paradigms remains a challenge. In this work, we propose ISTASTrack, the first transformer-based \textbf{A}NN-\textbf{S}NN hybrid \textbf{Track}er equipped with \textbf{ISTA} adapters for RGB-Event tracking. The two-branch model employs a vision transformer to extract spatial context from RGB inputs and a spiking transformer to capture spatio-temporal dynamics from event streams. To bridge the modality and paradigm gap between ANN and SNN features, we systematically design a model-based ISTA adapter for bidirectional feature interaction between the two branches, derived from sparse representation theory by unfolding the iterative shrinkage thresholding algorithm. Additionally, we incorporate a temporal downsampling attention module within the adapter to align multi-step SNN features with single-step ANN features in the latent space, improving temporal fusion. Experimental results on RGB-Event tracking benchmarks, such as FE240hz, VisEvent, COESOT, and FELT, have demonstrated that ISTASTrack achieves state-of-the-art performance while maintaining high energy efficiency, highlighting the effectiveness and practicality of hybrid ANN-SNN designs for robust visual tracking. The code is publicly available at https://github.com/lsying009/ISTASTrack.git.
>
---
#### [new 048] SignClip: Leveraging Mouthing Cues for Sign Language Translation by Multimodal Contrastive Fusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SignClip框架，用于手语翻译任务，旨在提升翻译准确性。通过融合手势与口型信息，并引入层次对比学习，实现多模态对齐。实验表明其在PHOENIX14T数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.10266v1](http://arxiv.org/pdf/2509.10266v1)**

> **作者:** Wenfang Wu; Tingting Yuan; Yupeng Li; Daling Wang; Xiaoming Fu
>
> **摘要:** Sign language translation (SLT) aims to translate natural language from sign language videos, serving as a vital bridge for inclusive communication. While recent advances leverage powerful visual backbones and large language models, most approaches mainly focus on manual signals (hand gestures) and tend to overlook non-manual cues like mouthing. In fact, mouthing conveys essential linguistic information in sign languages and plays a crucial role in disambiguating visually similar signs. In this paper, we propose SignClip, a novel framework to improve the accuracy of sign language translation. It fuses manual and non-manual cues, specifically spatial gesture and lip movement features. Besides, SignClip introduces a hierarchical contrastive learning framework with multi-level alignment objectives, ensuring semantic consistency across sign-lip and visual-text modalities. Extensive experiments on two benchmark datasets, PHOENIX14T and How2Sign, demonstrate the superiority of our approach. For example, on PHOENIX14T, in the Gloss-free setting, SignClip surpasses the previous state-of-the-art model SpaMo, improving BLEU-4 from 24.32 to 24.71, and ROUGE from 46.57 to 48.38.
>
---
#### [new 049] Realism Control One-step Diffusion for Real-World Image Super-Resolution
- **分类: cs.CV; cs.AI**

- **简介: 论文提出RCOD框架，解决单步扩散模型在图像超分辨率中难以平衡保真度与真实感的问题。通过引入隐空间分组策略、退化感知采样和视觉提示注入，实现灵活控制，提升重建质量与效率。属于图像超分辨率任务。**

- **链接: [http://arxiv.org/pdf/2509.10122v1](http://arxiv.org/pdf/2509.10122v1)**

> **作者:** Zongliang Wu; Siming Zheng; Peng-Tao Jiang; Xin Yuan
>
> **摘要:** Pre-trained diffusion models have shown great potential in real-world image super-resolution (Real-ISR) tasks by enabling high-resolution reconstructions. While one-step diffusion (OSD) methods significantly improve efficiency compared to traditional multi-step approaches, they still have limitations in balancing fidelity and realism across diverse scenarios. Since the OSDs for SR are usually trained or distilled by a single timestep, they lack flexible control mechanisms to adaptively prioritize these competing objectives, which are inherently manageable in multi-step methods through adjusting sampling steps. To address this challenge, we propose a Realism Controlled One-step Diffusion (RCOD) framework for Real-ISR. RCOD provides a latent domain grouping strategy that enables explicit control over fidelity-realism trade-offs during the noise prediction phase with minimal training paradigm modifications and original training data. A degradation-aware sampling strategy is also introduced to align distillation regularization with the grouping strategy and enhance the controlling of trade-offs. Moreover, a visual prompt injection module is used to replace conventional text prompts with degradation-aware visual tokens, enhancing both restoration accuracy and semantic consistency. Our method achieves superior fidelity and perceptual quality while maintaining computational efficiency. Extensive experiments demonstrate that RCOD outperforms state-of-the-art OSD methods in both quantitative metrics and visual qualities, with flexible realism control capabilities in the inference stage. The code will be released.
>
---
#### [new 050] DGFusion: Depth-Guided Sensor Fusion for Robust Semantic Perception
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出DGFusion，一种基于深度引导的多传感器融合方法，用于自动驾驶的语义感知任务。通过结合激光雷达深度信息，提升复杂环境下多模态分割的鲁棒性，实现更精准的场景理解。**

- **链接: [http://arxiv.org/pdf/2509.09828v1](http://arxiv.org/pdf/2509.09828v1)**

> **作者:** Tim Broedermannn; Christos Sakaridis; Luigi Piccinelli; Wim Abbeloos; Luc Van Gool
>
> **备注:** Code and models will be available at https://github.com/timbroed/DGFusion
>
> **摘要:** Robust semantic perception for autonomous vehicles relies on effectively combining multiple sensors with complementary strengths and weaknesses. State-of-the-art sensor fusion approaches to semantic perception often treat sensor data uniformly across the spatial extent of the input, which hinders performance when faced with challenging conditions. By contrast, we propose a novel depth-guided multimodal fusion method that upgrades condition-aware fusion by integrating depth information. Our network, DGFusion, poses multimodal segmentation as a multi-task problem, utilizing the lidar measurements, which are typically available in outdoor sensor suites, both as one of the model's inputs and as ground truth for learning depth. Our corresponding auxiliary depth head helps to learn depth-aware features, which are encoded into spatially varying local depth tokens that condition our attentive cross-modal fusion. Together with a global condition token, these local depth tokens dynamically adapt sensor fusion to the spatially varying reliability of each sensor across the scene, which largely depends on depth. In addition, we propose a robust loss for our depth, which is essential for learning from lidar inputs that are typically sparse and noisy in adverse conditions. Our method achieves state-of-the-art panoptic and semantic segmentation performance on the challenging MUSES and DELIVER datasets. Code and models will be available at https://github.com/timbroed/DGFusion
>
---
#### [new 051] Hierarchical MLANet: Multi-level Attention for 3D Face Reconstruction From Single Images
- **分类: cs.CV**

- **简介: 论文提出MLANet，用于从单张真实图像中重建3D人脸模型。属于3D人脸重建任务，解决缺乏标注数据和复杂环境的问题，采用多级注意力机制与半监督训练策略，提升重建精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.10024v1](http://arxiv.org/pdf/2509.10024v1)**

> **作者:** Danling Cao
>
> **备注:** This work was completed during the author's MPhil studies at the University of Manchester
>
> **摘要:** Recovering 3D face models from 2D in-the-wild images has gained considerable attention in the computer vision community due to its wide range of potential applications. However, the lack of ground-truth labeled datasets and the complexity of real-world environments remain significant challenges. In this chapter, we propose a convolutional neural network-based approach, the Hierarchical Multi-Level Attention Network (MLANet), for reconstructing 3D face models from single in-the-wild images. Our model predicts detailed facial geometry, texture, pose, and illumination parameters from a single image. Specifically, we employ a pre-trained hierarchical backbone network and introduce multi-level attention mechanisms at different stages of 2D face image feature extraction. A semi-supervised training strategy is employed, incorporating 3D Morphable Model (3DMM) parameters from publicly available datasets along with a differentiable renderer, enabling an end-to-end training process. Extensive experiments, including both comparative and ablation studies, were conducted on two benchmark datasets, AFLW2000-3D and MICC Florence, focusing on 3D face reconstruction and 3D face alignment tasks. The effectiveness of the proposed method was evaluated both quantitatively and qualitatively.
>
---
#### [new 052] I-Segmenter: Integer-Only Vision Transformer for Efficient Semantic Segmentation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出I-Segmenter，一种全整数运算的视觉Transformer分割框架，用于高效语义分割。解决ViT在低精度下精度下降的问题，通过整数运算和新激活函数提升效率与稳定性。**

- **链接: [http://arxiv.org/pdf/2509.10334v1](http://arxiv.org/pdf/2509.10334v1)**

> **作者:** Jordan Sassoon; Michal Szczepanski; Martyna Poreba
>
> **摘要:** Vision Transformers (ViTs) have recently achieved strong results in semantic segmentation, yet their deployment on resource-constrained devices remains limited due to their high memory footprint and computational cost. Quantization offers an effective strategy to improve efficiency, but ViT-based segmentation models are notoriously fragile under low precision, as quantization errors accumulate across deep encoder-decoder pipelines. We introduce I-Segmenter, the first fully integer-only ViT segmentation framework. Building on the Segmenter architecture, I-Segmenter systematically replaces floating-point operations with integer-only counterparts. To further stabilize both training and inference, we propose $\lambda$-ShiftGELU, a novel activation function that mitigates the limitations of uniform quantization in handling long-tailed activation distributions. In addition, we remove the L2 normalization layer and replace bilinear interpolation in the decoder with nearest neighbor upsampling, ensuring integer-only execution throughout the computational graph. Extensive experiments show that I-Segmenter achieves accuracy within a reasonable margin of its FP32 baseline (5.1 % on average), while reducing model size by up to 3.8x and enabling up to 1.2x faster inference with optimized runtimes. Notably, even in one-shot PTQ with a single calibration image, I-Segmenter delivers competitive accuracy, underscoring its practicality for real-world deployment.
>
---
#### [new 053] Mask Consistency Regularization in Object Removal
- **分类: cs.CV**

- **简介: 该论文属于图像修复中的物体移除任务，旨在解决模型生成无关内容（mask hallucination）和形状偏差（mask-shape bias）问题。提出Mask Consistency Regularization（MCR）方法，通过引入掩码扰动策略，提升修复结果的上下文一致性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.10259v1](http://arxiv.org/pdf/2509.10259v1)**

> **作者:** Hua Yuan; Jin Yuan; Yicheng Jiang; Yao Zhang; Xin Geng; Yong Rui
>
> **摘要:** Object removal, a challenging task within image inpainting, involves seamlessly filling the removed region with content that matches the surrounding context. Despite advancements in diffusion models, current methods still face two critical challenges. The first is mask hallucination, where the model generates irrelevant or spurious content inside the masked region, and the second is mask-shape bias, where the model fills the masked area with an object that mimics the mask's shape rather than surrounding content. To address these issues, we propose Mask Consistency Regularization (MCR), a novel training strategy designed specifically for object removal tasks. During training, our approach introduces two mask perturbations: dilation and reshape, enforcing consistency between the outputs of these perturbed branches and the original mask. The dilated masks help align the model's output with the surrounding content, while reshaped masks encourage the model to break the mask-shape bias. This combination of strategies enables MCR to produce more robust and contextually coherent inpainting results. Our experiments demonstrate that MCR significantly reduces hallucinations and mask-shape bias, leading to improved performance in object removal.
>
---
#### [new 054] Surrogate Supervision for Robust and Generalizable Deformable Image Registration
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像配准任务，旨在解决深度学习配准模型对输入变化敏感的问题。提出“代理监督”方法，通过解耦输入与监督域，提升模型鲁棒性与泛化能力，并在多种医学影像场景中验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2509.09869v1](http://arxiv.org/pdf/2509.09869v1)**

> **作者:** Yihao Liu; Junyu Chen; Lianrui Zuo; Shuwen Wei; Brian D. Boyd; Carmen Andreescu; Olusola Ajilore; Warren D. Taylor; Aaron Carass; Bennett A. Landman
>
> **摘要:** Objective: Deep learning-based deformable image registration has achieved strong accuracy, but remains sensitive to variations in input image characteristics such as artifacts, field-of-view mismatch, or modality difference. We aim to develop a general training paradigm that improves the robustness and generalizability of registration networks. Methods: We introduce surrogate supervision, which decouples the input domain from the supervision domain by applying estimated spatial transformations to surrogate images. This allows training on heterogeneous inputs while ensuring supervision is computed in domains where similarity is well defined. We evaluate the framework through three representative applications: artifact-robust brain MR registration, mask-agnostic lung CT registration, and multi-modal MR registration. Results: Across tasks, surrogate supervision demonstrated strong resilience to input variations including inhomogeneity field, inconsistent field-of-view, and modality differences, while maintaining high performance on well-curated data. Conclusions: Surrogate supervision provides a principled framework for training robust and generalizable deep learning-based registration models without increasing complexity. Significance: Surrogate supervision offers a practical pathway to more robust and generalizable medical image registration, enabling broader applicability in diverse biomedical imaging scenarios.
>
---
#### [new 055] Color Me Correctly: Bridging Perceptual Color Spaces and Text Embeddings for Improved Diffusion Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决颜色描述模糊导致的生成颜色偏差问题。提出一种无需训练的方法，利用大语言模型解析颜色术语，并在文本嵌入空间中优化颜色融合，提升生成图像的颜色准确性。**

- **链接: [http://arxiv.org/pdf/2509.10058v1](http://arxiv.org/pdf/2509.10058v1)**

> **作者:** Sung-Lin Tsai; Bo-Lun Huang; Yu Ting Shen; Cheng Yu Yeo; Chiang Tseng; Bo-Kai Ruan; Wen-Sheng Lien; Hong-Han Shuai
>
> **备注:** Accepted to ACM Multimedia 2025 (MM '25)
>
> **摘要:** Accurate color alignment in text-to-image (T2I) generation is critical for applications such as fashion, product visualization, and interior design, yet current diffusion models struggle with nuanced and compound color terms (e.g., Tiffany blue, lime green, hot pink), often producing images that are misaligned with human intent. Existing approaches rely on cross-attention manipulation, reference images, or fine-tuning but fail to systematically resolve ambiguous color descriptions. To precisely render colors under prompt ambiguity, we propose a training-free framework that enhances color fidelity by leveraging a large language model (LLM) to disambiguate color-related prompts and guiding color blending operations directly in the text embedding space. Our method first employs a large language model (LLM) to resolve ambiguous color terms in the text prompt, and then refines the text embeddings based on the spatial relationships of the resulting color terms in the CIELAB color space. Unlike prior methods, our approach improves color accuracy without requiring additional training or external reference images. Experimental results demonstrate that our framework improves color alignment without compromising image quality, bridging the gap between text semantics and visual generation.
>
---
#### [new 056] Purge-Gate: Backpropagation-Free Test-Time Adaptation for Point Clouds Classification via Token Purging
- **分类: cs.CV**

- **简介: 该论文提出Purge-Gate方法，用于点云分类的测试时自适应任务，解决分布偏移导致的性能下降问题。通过去除受领域偏移影响的token，无需反向传播即可实现高效、鲁棒的模型适应。**

- **链接: [http://arxiv.org/pdf/2509.09785v1](http://arxiv.org/pdf/2509.09785v1)**

> **作者:** Moslem Yazdanpanah; Ali Bahri; Mehrdad Noori; Sahar Dastani; Gustavo Adolfo Vargas Hakim; David Osowiechi; Ismail Ben Ayed; Christian Desrosiers
>
> **摘要:** Test-time adaptation (TTA) is crucial for mitigating performance degradation caused by distribution shifts in 3D point cloud classification. In this work, we introduce Token Purging (PG), a novel backpropagation-free approach that removes tokens highly affected by domain shifts before they reach attention layers. Unlike existing TTA methods, PG operates at the token level, ensuring robust adaptation without iterative updates. We propose two variants: PG-SP, which leverages source statistics, and PG-SF, a fully source-free version relying on CLS-token-driven adaptation. Extensive evaluations on ModelNet40-C, ShapeNet-C, and ScanObjectNN-C demonstrate that PG-SP achieves an average of +10.3\% higher accuracy than state-of-the-art backpropagation-free methods, while PG-SF sets new benchmarks for source-free adaptation. Moreover, PG is 12.4 times faster and 5.5 times more memory efficient than our baseline, making it suitable for real-world deployment. Code is available at \hyperlink{https://github.com/MosyMosy/Purge-Gate}{https://github.com/MosyMosy/Purge-Gate}
>
---
#### [new 057] Leveraging Multi-View Weak Supervision for Occlusion-Aware Multi-Human Parsing
- **分类: cs.CV**

- **简介: 该论文属于多人体解析任务，旨在解决遮挡场景下人体部分分割问题。提出利用多视角弱监督方法，结合实例分割与一致性损失，提升遮挡情况下的分割效果，并设计半自动标注策略生成训练数据。**

- **链接: [http://arxiv.org/pdf/2509.10093v1](http://arxiv.org/pdf/2509.10093v1)**

> **作者:** Laura Bragagnolo; Matteo Terreran; Leonardo Barcellona; Stefano Ghidoni
>
> **备注:** ICIAP 2025
>
> **摘要:** Multi-human parsing is the task of segmenting human body parts while associating each part to the person it belongs to, combining instance-level and part-level information for fine-grained human understanding. In this work, we demonstrate that, while state-of-the-art approaches achieved notable results on public datasets, they struggle considerably in segmenting people with overlapping bodies. From the intuition that overlapping people may appear separated from a different point of view, we propose a novel training framework exploiting multi-view information to improve multi-human parsing models under occlusions. Our method integrates such knowledge during the training process, introducing a novel approach based on weak supervision on human instances and a multi-view consistency loss. Given the lack of suitable datasets in the literature, we propose a semi-automatic annotation strategy to generate human instance segmentation masks from multi-view RGB+D data and 3D human skeletons. The experiments demonstrate that the approach can achieve up to a 4.20\% relative improvement on human parsing over the baseline model in occlusion scenarios.
>
---
#### [new 058] InfGen: A Resolution-Agnostic Paradigm for Scalable Image Synthesis
- **分类: cs.CV**

- **简介: 论文提出InfGen，一种无需重新训练即可生成任意分辨率图像的框架，解决扩散模型高分辨率生成计算复杂度高的问题，通过替换VAE解码器为一步生成器，显著提升生成速度与扩展性。**

- **链接: [http://arxiv.org/pdf/2509.10441v1](http://arxiv.org/pdf/2509.10441v1)**

> **作者:** Tao Han; Wanghan Xu; Junchao Gong; Xiaoyu Yue; Song Guo; Luping Zhou; Lei Bai
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Arbitrary resolution image generation provides a consistent visual experience across devices, having extensive applications for producers and consumers. Current diffusion models increase computational demand quadratically with resolution, causing 4K image generation delays over 100 seconds. To solve this, we explore the second generation upon the latent diffusion models, where the fixed latent generated by diffusion models is regarded as the content representation and we propose to decode arbitrary resolution images with a compact generated latent using a one-step generator. Thus, we present the \textbf{InfGen}, replacing the VAE decoder with the new generator, for generating images at any resolution from a fixed-size latent without retraining the diffusion models, which simplifies the process, reducing computational complexity and can be applied to any model using the same latent space. Experiments show InfGen is capable of improving many models into the arbitrary high-resolution era while cutting 4K image generation time to under 10 seconds.
>
---
#### [new 059] Decomposing Visual Classification: Assessing Tree-Based Reasoning in VLMs
- **分类: cs.CV**

- **简介: 论文研究视觉语言模型（VLMs）在树状结构分类任务中的表现，探索是否通过决策树增强其性能。提出分解分类为可解释决策的框架，在细粒度和粗粒度数据集上评估，发现树状推理效果不如零样本提示，但加入描述后有所提升。**

- **链接: [http://arxiv.org/pdf/2509.09732v1](http://arxiv.org/pdf/2509.09732v1)**

> **作者:** Sary Elmansoury; Islam Mesabah; Gerrit Großmann; Peter Neigel; Raj Bhalwankar; Daniel Kondermann; Sebastian J. Vollmer
>
> **摘要:** Vision language models (VLMs) excel at zero-shot visual classification, but their performance on fine-grained tasks and large hierarchical label spaces is understudied. This paper investigates whether structured, tree-based reasoning can enhance VLM performance. We introduce a framework that decomposes classification into interpretable decisions using decision trees and evaluates it on fine-grained (GTSRB) and coarse-grained (CIFAR-10) datasets. Although the model achieves 98.2% accuracy in understanding the tree knowledge, tree-based reasoning consistently underperforms standard zero-shot prompting. We also explore enhancing the tree prompts with LLM-generated classes and image descriptions to improve alignment. The added description enhances the performance of the tree-based and zero-shot methods. Our findings highlight limitations of structured reasoning in visual classification and offer insights for designing more interpretable VLM systems.
>
---
#### [new 060] Segment Anything for Cell Tracking
- **分类: cs.CV**

- **简介: 该论文提出一种零样本细胞追踪框架，利用SAM2模型解决显微镜图像中细胞追踪与有丝分裂检测难题。无需标注数据，实现跨数据集的通用性，提升2D和3D视频的追踪准确性。**

- **链接: [http://arxiv.org/pdf/2509.09943v1](http://arxiv.org/pdf/2509.09943v1)**

> **作者:** Zhu Chen; Mert Edgü; Er Jin; Johannes Stegmaier
>
> **摘要:** Tracking cells and detecting mitotic events in time-lapse microscopy image sequences is a crucial task in biomedical research. However, it remains highly challenging due to dividing objects, low signal-tonoise ratios, indistinct boundaries, dense clusters, and the visually similar appearance of individual cells. Existing deep learning-based methods rely on manually labeled datasets for training, which is both costly and time-consuming. Moreover, their generalizability to unseen datasets remains limited due to the vast diversity of microscopy data. To overcome these limitations, we propose a zero-shot cell tracking framework by integrating Segment Anything 2 (SAM2), a large foundation model designed for general image and video segmentation, into the tracking pipeline. As a fully-unsupervised approach, our method does not depend on or inherit biases from any specific training dataset, allowing it to generalize across diverse microscopy datasets without finetuning. Our approach achieves competitive accuracy in both 2D and large-scale 3D time-lapse microscopy videos while eliminating the need for dataset-specific adaptation.
>
---
#### [new 061] Efficient Learned Image Compression Through Knowledge Distillation
- **分类: cs.CV**

- **简介: 该论文属于图像压缩任务，旨在解决神经网络压缩模型资源消耗大的问题。通过知识蒸馏训练小型网络，实现高效压缩，提升不同架构下的性能与能效。**

- **链接: [http://arxiv.org/pdf/2509.10366v1](http://arxiv.org/pdf/2509.10366v1)**

> **作者:** Fabien Allemand; Attilio Fiandrotti; Sumanta Chaudhuri; Alaa Eddine Mazouz
>
> **备注:** 19 pages, 21 figures
>
> **摘要:** Learned image compression sits at the intersection of machine learning and image processing. With advances in deep learning, neural network-based compression methods have emerged. In this process, an encoder maps the image to a low-dimensional latent space, which is then quantized, entropy-coded into a binary bitstream, and transmitted to the receiver. At the receiver end, the bitstream is entropy-decoded, and a decoder reconstructs an approximation of the original image. Recent research suggests that these models consistently outperform conventional codecs. However, they require significant processing power, making them unsuitable for real-time use on resource-constrained platforms, which hinders their deployment in mainstream applications. This study aims to reduce the resource requirements of neural networks used for image compression by leveraging knowledge distillation, a training paradigm where smaller neural networks, partially trained on the outputs of larger, more complex models, can achieve better performance than when trained independently. Our work demonstrates that knowledge distillation can be effectively applied to image compression tasks: i) across various architecture sizes, ii) to achieve different image quality/bit rate tradeoffs, and iii) to save processing and energy resources. This approach introduces new settings and hyperparameters, and future research could explore the impact of different teacher models, as well as alternative loss functions. Knowledge distillation could also be extended to transformer-based models. The code is publicly available at: https://github.com/FABallemand/PRIM .
>
---
#### [new 062] Scalable Training for Vector-Quantized Networks with 100% Codebook Utilization
- **分类: cs.CV**

- **简介: 论文提出FVQ方法，解决VQ网络训练中代码簿使用率低、重建性能差的问题。通过VQBridge优化代码向量，实现100%代码簿利用，提升图像生成效果。属于图像生成中的离散编码器优化任务。**

- **链接: [http://arxiv.org/pdf/2509.10140v1](http://arxiv.org/pdf/2509.10140v1)**

> **作者:** Yifan Chang; Jie Qin; Limeng Qiao; Xiaofeng Wang; Zheng Zhu; Lin Ma; Xingang Wang
>
> **摘要:** Vector quantization (VQ) is a key component in discrete tokenizers for image generation, but its training is often unstable due to straight-through estimation bias, one-step-behind updates, and sparse codebook gradients, which lead to suboptimal reconstruction performance and low codebook usage. In this work, we analyze these fundamental challenges and provide a simple yet effective solution. To maintain high codebook usage in VQ networks (VQN) during learning annealing and codebook size expansion, we propose VQBridge, a robust, scalable, and efficient projector based on the map function method. VQBridge optimizes code vectors through a compress-process-recover pipeline, enabling stable and effective codebook training. By combining VQBridge with learning annealing, our VQN achieves full (100%) codebook usage across diverse codebook configurations, which we refer to as FVQ (FullVQ). Through extensive experiments, we demonstrate that FVQ is effective, scalable, and generalizable: it attains 100% codebook usage even with a 262k-codebook, achieves state-of-the-art reconstruction performance, consistently improves with larger codebooks, higher vector channels, or longer training, and remains effective across different VQ variants. Moreover, when integrated with LlamaGen, FVQ significantly enhances image generation performance, surpassing visual autoregressive models (VAR) by 0.5 and diffusion models (DiT) by 0.2 rFID, highlighting the importance of high-quality tokenizers for strong autoregressive image generation.
>
---
#### [new 063] Robustness and Diagnostic Performance of Super-Resolution Fetal Brain MRI
- **分类: cs.CV**

- **简介: 该论文研究胎儿脑MRI的超分辨率重建方法，解决低分辨率与运动伪影问题。比较三种方法在健康与病理性病例中的表现，评估其对体积测量和诊断分类的影响，发现NeSVoR最稳健且诊断性能不受影响。**

- **链接: [http://arxiv.org/pdf/2509.10257v1](http://arxiv.org/pdf/2509.10257v1)**

> **作者:** Ema Masterl; Tina Vipotnik Vesnaver; Žiga Špiclin
>
> **备注:** Accepted at the PIPPI Workshop of MICCAI 2025
>
> **摘要:** Fetal brain MRI relies on rapid multi-view 2D slice acquisitions to reduce motion artifacts caused by fetal movement. However, these stacks are typically low resolution, may suffer from motion corruption, and do not adequately capture 3D anatomy. Super-resolution reconstruction (SRR) methods aim to address these limitations by combining slice-to-volume registration and super-resolution techniques to generate high-resolution (HR) 3D volumes. While several SRR methods have been proposed, their comparative performance - particularly in pathological cases - and their influence on downstream volumetric analysis and diagnostic tasks remain underexplored. In this study, we applied three state-of-the-art SRR method - NiftyMIC, SVRTK, and NeSVoR - to 140 fetal brain MRI scans, including both healthy controls (HC) and pathological cases (PC) with ventriculomegaly (VM). Each HR reconstruction was segmented using the BoUNTi algorithm to extract volumes of nine principal brain structures. We evaluated visual quality, SRR success rates, volumetric measurement agreement, and diagnostic classification performance. NeSVoR demonstrated the highest and most consistent reconstruction success rate (>90%) across both HC and PC groups. Although significant differences in volumetric estimates were observed between SRR methods, classification performance for VM was not affected by the choice of SRR method. These findings highlight NeSVoR's robustness and the resilience of diagnostic performance despite SRR-induced volumetric variability.
>
---
#### [new 064] Multimodal Mathematical Reasoning Embedded in Aerial Vehicle Imagery: Benchmarking, Analysis, and Exploration
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出AVI-Math基准，用于评估无人机图像中的多模态数学推理能力。任务是解决当前视觉语言模型在几何、代数等领域的推理不足问题，通过构建包含3773题的数据集并测试14种VLM，分析其局限性，并探索链式思维与微调方法的改进效果。**

- **链接: [http://arxiv.org/pdf/2509.10059v1](http://arxiv.org/pdf/2509.10059v1)**

> **作者:** Yue Zhou; Litong Feng; Mengcheng Lan; Xue Yang; Qingyun Li; Yiping Ke; Xue Jiang; Wayne Zhang
>
> **备注:** 17 pages, 16 figures
>
> **摘要:** Mathematical reasoning is critical for tasks such as precise distance and area computations, trajectory estimations, and spatial analysis in unmanned aerial vehicle (UAV) based remote sensing, yet current vision-language models (VLMs) have not been adequately tested in this domain. To address this gap, we introduce AVI-Math, the first benchmark to rigorously evaluate multimodal mathematical reasoning in aerial vehicle imagery, moving beyond simple counting tasks to include domain-specific knowledge in areas such as geometry, logic, and algebra. The dataset comprises 3,773 high-quality vehicle-related questions captured from UAV views, covering 6 mathematical subjects and 20 topics. The data, collected at varying altitudes and from multiple UAV angles, reflects real-world UAV scenarios, ensuring the diversity and complexity of the constructed mathematical problems. In this paper, we benchmark 14 prominent VLMs through a comprehensive evaluation and demonstrate that, despite their success on previous multimodal benchmarks, these models struggle with the reasoning tasks in AVI-Math. Our detailed analysis highlights significant limitations in the mathematical reasoning capabilities of current VLMs and suggests avenues for future research. Furthermore, we explore the use of Chain-of-Thought prompting and fine-tuning techniques, which show promise in addressing the reasoning challenges in AVI-Math. Our findings not only expose the limitations of VLMs in mathematical reasoning but also offer valuable insights for advancing UAV-based trustworthy VLMs in real-world applications. The code, and datasets will be released at https://github.com/VisionXLab/avi-math
>
---
#### [new 065] LoFT: Parameter-Efficient Fine-Tuning for Long-tailed Semi-Supervised Learning in Open-World Scenarios
- **分类: cs.LG; cs.CV**

- **简介: 论文提出LoFT框架，解决长尾半监督学习中的伪标签质量与过自信问题，并扩展至开放世界场景（LoFT-OW）。通过参数高效的微调方法提升模型性能，实验表明其在少量未标记数据下表现优异。属于长尾分类与半监督学习任务。**

- **链接: [http://arxiv.org/pdf/2509.09926v1](http://arxiv.org/pdf/2509.09926v1)**

> **作者:** Jiahao Chen; Zhiyuan Huang; Yurou Liu; Bing Su
>
> **摘要:** Long-tailed learning has garnered increasing attention due to its wide applicability in real-world scenarios. Among existing approaches, Long-Tailed Semi-Supervised Learning (LTSSL) has emerged as an effective solution by incorporating a large amount of unlabeled data into the imbalanced labeled dataset. However, most prior LTSSL methods are designed to train models from scratch, which often leads to issues such as overconfidence and low-quality pseudo-labels. To address these challenges, we extend LTSSL into the foundation model fine-tuning paradigm and propose a novel framework: LoFT (Long-tailed semi-supervised learning via parameter-efficient Fine-Tuning). We demonstrate that fine-tuned foundation models can generate more reliable pseudolabels, thereby benefiting imbalanced learning. Furthermore, we explore a more practical setting by investigating semi-supervised learning under open-world conditions, where the unlabeled data may include out-of-distribution (OOD) samples. To handle this problem, we propose LoFT-OW (LoFT under Open-World scenarios) to improve the discriminative ability. Experimental results on multiple benchmarks demonstrate that our method achieves superior performance compared to previous approaches, even when utilizing only 1\% of the unlabeled data compared with previous works.
>
---
#### [new 066] Drone-Based Multispectral Imaging and Deep Learning for Timely Detection of Branched Broomrape in Tomato Farms
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 论文提出利用无人机多光谱成像与LSTM深度学习结合的方法，解决番茄田中枝状独尾草早期检测难题。通过SMOTE处理类别不平衡问题，实现高准确率和召回率的早期寄生植物识别，为精准农业提供新工具。**

- **链接: [http://arxiv.org/pdf/2509.09972v1](http://arxiv.org/pdf/2509.09972v1)**

> **作者:** Mohammadreza Narimani; Alireza Pourreza; Ali Moghimi; Mohsen Mesgaran; Parastoo Farajpoor; Hamid Jafarbiglu
>
> **备注:** Author-accepted version (no publisher header/footer). 10 pages + presentation. Published in Proceedings of SPIE Defense + Commercial Sensing 2024, Vol. 13053, Paper 1305304. Event: National Harbor, Maryland, USA. Official version: https://doi.org/10.1117/12.3021219
>
> **摘要:** This study addresses the escalating threat of branched broomrape (Phelipanche ramosa) to California's tomato industry, which supplies over 90 percent of U.S. processing tomatoes. The parasite's largely underground life cycle makes early detection difficult, while conventional chemical controls are costly, environmentally harmful, and often ineffective. To address this, we combined drone-based multispectral imagery with Long Short-Term Memory (LSTM) deep learning networks, using the Synthetic Minority Over-sampling Technique (SMOTE) to handle class imbalance. Research was conducted on a known broomrape-infested tomato farm in Woodland, Yolo County, CA, across five key growth stages determined by growing degree days (GDD). Multispectral images were processed to isolate tomato canopy reflectance. At 897 GDD, broomrape could be detected with 79.09 percent overall accuracy and 70.36 percent recall without integrating later stages. Incorporating sequential growth stages with LSTM improved detection substantially. The best-performing scenario, which integrated all growth stages with SMOTE augmentation, achieved 88.37 percent overall accuracy and 95.37 percent recall. These results demonstrate the strong potential of temporal multispectral analysis and LSTM networks for early broomrape detection. While further real-world data collection is needed for practical deployment, this study shows that UAV-based multispectral sensing coupled with deep learning could provide a powerful precision agriculture tool to reduce losses and improve sustainability in tomato production.
>
---
#### [new 067] Multi-pathology Chest X-ray Classification with Rejection Mechanisms
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于多标签医学图像分类任务，旨在解决深度学习模型在胸部X光诊断中的过度自信问题。研究提出结合熵和置信区间两种拒绝机制的框架，提升模型可靠性，并在多个数据集上验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.10348v1](http://arxiv.org/pdf/2509.10348v1)**

> **作者:** Yehudit Aperstein; Amit Tzahar; Alon Gottlib; Tal Verber; Ravit Shagan Damti; Alexander Apartsin
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Overconfidence in deep learning models poses a significant risk in high-stakes medical imaging tasks, particularly in multi-label classification of chest X-rays, where multiple co-occurring pathologies must be detected simultaneously. This study introduces an uncertainty-aware framework for chest X-ray diagnosis based on a DenseNet-121 backbone, enhanced with two selective prediction mechanisms: entropy-based rejection and confidence interval-based rejection. Both methods enable the model to abstain from uncertain predictions, improving reliability by deferring ambiguous cases to clinical experts. A quantile-based calibration procedure is employed to tune rejection thresholds using either global or class-specific strategies. Experiments conducted on three large public datasets (PadChest, NIH ChestX-ray14, and MIMIC-CXR) demonstrate that selective rejection improves the trade-off between diagnostic accuracy and coverage, with entropy-based rejection yielding the highest average AUC across all pathologies. These results support the integration of selective prediction into AI-assisted diagnostic workflows, providing a practical step toward safer, uncertainty-aware deployment of deep learning in clinical settings.
>
---
#### [new 068] Polarization Denoising and Demosaicking: Dataset and Baseline Method
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于图像处理任务，解决DoFP偏振成像中的去噪与解马赛克问题。提出新数据集和基线方法，采用去噪后解马赛克策略，提升图像重建性能。**

- **链接: [http://arxiv.org/pdf/2509.10098v1](http://arxiv.org/pdf/2509.10098v1)**

> **作者:** Muhamad Daniel Ariff Bin Abdul Rahman; Yusuke Monno; Masayuki Tanaka; Masatoshi Okutomi
>
> **备注:** Published in ICIP2025; Project page: http://www.ok.sc.e.titech.ac.jp/res/PolarDem/PDD.html
>
> **摘要:** A division-of-focal-plane (DoFP) polarimeter enables us to acquire images with multiple polarization orientations in one shot and thus it is valuable for many applications using polarimetric information. The image processing pipeline for a DoFP polarimeter entails two crucial tasks: denoising and demosaicking. While polarization demosaicking for a noise-free case has increasingly been studied, the research for the joint task of polarization denoising and demosaicking is scarce due to the lack of a suitable evaluation dataset and a solid baseline method. In this paper, we propose a novel dataset and method for polarization denoising and demosaicking. Our dataset contains 40 real-world scenes and three noise-level conditions, consisting of pairs of noisy mosaic inputs and noise-free full images. Our method takes a denoising-then-demosaicking approach based on well-accepted signal processing components to offer a reproducible method. Experimental results demonstrate that our method exhibits higher image reconstruction performance than other alternative methods, offering a solid baseline.
>
---
#### [new 069] Chord: Chain of Rendering Decomposition for PBR Material Estimation from Generated Texture Images
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出一种两阶段框架用于生成和估计PBR材质。通过微调扩散模型生成纹理图像，并采用链式分解方案逐步预测SVBRDF通道，提升生成质量与用户控制灵活性，适用于多种材质生成与编辑任务。**

- **链接: [http://arxiv.org/pdf/2509.09952v1](http://arxiv.org/pdf/2509.09952v1)**

> **作者:** Zhi Ying; Boxiang Rong; Jingyu Wang; Maoyuan Xu
>
> **备注:** Accepted to SIGGRAPH Asia 2025. Project page: https://ubisoft-laforge.github.io/world/chord
>
> **摘要:** Material creation and reconstruction are crucial for appearance modeling but traditionally require significant time and expertise from artists. While recent methods leverage visual foundation models to synthesize PBR materials from user-provided inputs, they often fall short in quality, flexibility, and user control. We propose a novel two-stage generate-and-estimate framework for PBR material generation. In the generation stage, a fine-tuned diffusion model synthesizes shaded, tileable texture images aligned with user input. In the estimation stage, we introduce a chained decomposition scheme that sequentially predicts SVBRDF channels by passing previously extracted representation as input into a single-step image-conditional diffusion model. Our method is efficient, high quality, and enables flexible user control. We evaluate our approach against existing material generation and estimation methods, demonstrating superior performance. Our material estimation method shows strong robustness on both generated textures and in-the-wild photographs. Furthermore, we highlight the flexibility of our framework across diverse applications, including text-to-material, image-to-material, structure-guided generation, and material editing.
>
---
#### [new 070] GC-VLN: Instruction as Graph Constraints for Training-free Vision-and-Language Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出一种无需训练的视觉-语言导航框架GC-VLN，通过将指令转化为图约束优化，实现零样本适应。解决了现有方法在连续环境中的泛化与部署难题，提升了导航成功率与效率。**

- **链接: [http://arxiv.org/pdf/2509.10454v1](http://arxiv.org/pdf/2509.10454v1)**

> **作者:** Hang Yin; Haoyu Wei; Xiuwei Xu; Wenxuan Guo; Jie Zhou; Jiwen Lu
>
> **备注:** Accepted to CoRL 2025. Project page: [this https URL](https://bagh2178.github.io/GC-VLN/)
>
> **摘要:** In this paper, we propose a training-free framework for vision-and-language navigation (VLN). Existing zero-shot VLN methods are mainly designed for discrete environments or involve unsupervised training in continuous simulator environments, which makes it challenging to generalize and deploy them in real-world scenarios. To achieve a training-free framework in continuous environments, our framework formulates navigation guidance as graph constraint optimization by decomposing instructions into explicit spatial constraints. The constraint-driven paradigm decodes spatial semantics through constraint solving, enabling zero-shot adaptation to unseen environments. Specifically, we construct a spatial constraint library covering all types of spatial relationship mentioned in VLN instructions. The human instruction is decomposed into a directed acyclic graph, with waypoint nodes, object nodes and edges, which are used as queries to retrieve the library to build the graph constraints. The graph constraint optimization is solved by the constraint solver to determine the positions of waypoints, obtaining the robot's navigation path and final goal. To handle cases of no solution or multiple solutions, we construct a navigation tree and the backtracking mechanism. Extensive experiments on standard benchmarks demonstrate significant improvements in success rate and navigation efficiency compared to state-of-the-art zero-shot VLN methods. We further conduct real-world experiments to show that our framework can effectively generalize to new environments and instruction sets, paving the way for a more robust and autonomous navigation framework.
>
---
#### [new 071] Automated Tuning for Diffusion Inverse Problem Solvers without Generative Prior Retraining
- **分类: eess.IV; cs.AI; cs.CV; cs.LG; physics.med-ph**

- **简介: 该论文提出ZADS方法，解决扩散模型在逆问题求解中数据保真度权重调优问题。无需重训练，通过测试时优化实现自适应调整，在MRI重建任务中表现优于传统和最新方法。**

- **链接: [http://arxiv.org/pdf/2509.09880v1](http://arxiv.org/pdf/2509.09880v1)**

> **作者:** Yaşar Utku Alçalar; Junno Yun; Mehmet Akçakaya
>
> **备注:** IEEE International Workshop on Computational Advances in Multi-Sensor Adaptive Processing (CAMSAP), 2025
>
> **摘要:** Diffusion/score-based models have recently emerged as powerful generative priors for solving inverse problems, including accelerated MRI reconstruction. While their flexibility allows decoupling the measurement model from the learned prior, their performance heavily depends on carefully tuned data fidelity weights, especially under fast sampling schedules with few denoising steps. Existing approaches often rely on heuristics or fixed weights, which fail to generalize across varying measurement conditions and irregular timestep schedules. In this work, we propose Zero-shot Adaptive Diffusion Sampling (ZADS), a test-time optimization method that adaptively tunes fidelity weights across arbitrary noise schedules without requiring retraining of the diffusion prior. ZADS treats the denoising process as a fixed unrolled sampler and optimizes fidelity weights in a self-supervised manner using only undersampled measurements. Experiments on the fastMRI knee dataset demonstrate that ZADS consistently outperforms both traditional compressed sensing and recent diffusion-based methods, showcasing its ability to deliver high-fidelity reconstructions across varying noise schedules and acquisition settings.
>
---
#### [new 072] HHI-Assist: A Dataset and Benchmark of Human-Human Interaction in Physical Assistance Scenario
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于人机交互任务，旨在解决助人机器人中的人体运动预测问题。论文提出了HHI-Assist数据集和基于Transformer的去噪扩散模型，用于预测物理协助场景中互动双方的姿态，提升机器人辅助策略的效果。**

- **链接: [http://arxiv.org/pdf/2509.10096v1](http://arxiv.org/pdf/2509.10096v1)**

> **作者:** Saeed Saadatnejad; Reyhaneh Hosseininejad; Jose Barreiros; Katherine M. Tsui; Alexandre Alahi
>
> **备注:** Accepted to RA-L 2025
>
> **摘要:** The increasing labor shortage and aging population underline the need for assistive robots to support human care recipients. To enable safe and responsive assistance, robots require accurate human motion prediction in physical interaction scenarios. However, this remains a challenging task due to the variability of assistive settings and the complexity of coupled dynamics in physical interactions. In this work, we address these challenges through two key contributions: (1) HHI-Assist, a dataset comprising motion capture clips of human-human interactions in assistive tasks; and (2) a conditional Transformer-based denoising diffusion model for predicting the poses of interacting agents. Our model effectively captures the coupled dynamics between caregivers and care receivers, demonstrating improvements over baselines and strong generalization to unseen scenarios. By advancing interaction-aware motion prediction and introducing a new dataset, our work has the potential to significantly enhance robotic assistance policies. The dataset and code are available at: https://sites.google.com/view/hhi-assist/home
>
---
#### [new 073] Adaptive Token Merging for Efficient Transformer Semantic Communication at the Edge
- **分类: cs.LG; cs.AI; cs.CV; eess.IV**

- **简介: 论文提出一种无需训练的自适应token合并框架，解决大模型在边缘设备上的计算与通信开销问题。通过运行时合并语义冗余token，实现高效推理，在保持性能的同时显著降低计算和通信成本。**

- **链接: [http://arxiv.org/pdf/2509.09955v1](http://arxiv.org/pdf/2509.09955v1)**

> **作者:** Omar Erak; Omar Alhussein; Hatem Abou-Zeid; Mehdi Bennis; Sami Muhaidat
>
> **备注:** Submitted to IEEE Journals
>
> **摘要:** Large-scale transformers are central to modern semantic communication, yet their high computational and communication costs hinder deployment on resource-constrained edge devices. This paper introduces a training-free framework for adaptive token merging, a novel mechanism that compresses transformer representations at runtime by selectively merging semantically redundant tokens under per-layer similarity thresholds. Unlike prior fixed-ratio reduction, our approach couples merging directly to input redundancy, enabling data-dependent adaptation that balances efficiency and task relevance without retraining. We cast the discovery of merging strategies as a multi-objective optimization problem and leverage Bayesian optimization to obtain Pareto-optimal trade-offs between accuracy, inference cost, and communication cost. On ImageNet classification, we match the accuracy of the unmodified transformer with 30\% fewer floating-point operations per second and under 20\% of the original communication cost, while for visual question answering our method achieves performance competitive with the full LLaVA model at less than one-third of the compute and one-tenth of the bandwidth. Finally, we show that our adaptive merging is robust across varying channel conditions and provides inherent privacy benefits, substantially degrading the efficacy of model inversion attacks. Our framework provides a practical and versatile solution for deploying powerful transformer models in resource-limited edge intelligence scenarios.
>
---
## 更新

#### [replaced 001] PATS: Proficiency-Aware Temporal Sampling for Multi-View Sports Skill Assessment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.04996v4](http://arxiv.org/pdf/2506.04996v4)**

> **作者:** Edoardo Bianchi; Antonio Liotta
>
> **备注:** Accepted at the 2025 4th IEEE International Workshop on Sport Technology and Research
>
> **摘要:** Automated sports skill assessment requires capturing fundamental movement patterns that distinguish expert from novice performance, yet current video sampling methods disrupt the temporal continuity essential for proficiency evaluation. To this end, we introduce Proficiency-Aware Temporal Sampling (PATS), a novel sampling strategy that preserves complete fundamental movements within continuous temporal segments for multi-view skill assessment. PATS adaptively segments videos to ensure each analyzed portion contains full execution of critical performance components, repeating this process across multiple segments to maximize information coverage while maintaining temporal coherence. Evaluated on the EgoExo4D benchmark with SkillFormer, PATS surpasses the state-of-the-art accuracy across all viewing configurations (+0.65% to +3.05%) and delivers substantial gains in challenging domains (+26.22% bouldering, +2.39% music, +1.13% basketball). Systematic analysis reveals that PATS successfully adapts to diverse activity characteristics-from high-frequency sampling for dynamic sports to fine-grained segmentation for sequential skills-demonstrating its effectiveness as an adaptive approach to temporal sampling that advances automated skill assessment for real-world applications.
>
---
#### [replaced 002] HiddenObject: Modality-Agnostic Fusion for Multimodal Hidden Object Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.21135v2](http://arxiv.org/pdf/2508.21135v2)**

> **作者:** Harris Song; Tuan-Anh Vu; Sanjith Menon; Sriram Narasimhan; M. Khalid Jawed
>
> **备注:** fix typos
>
> **摘要:** Detecting hidden or partially concealed objects remains a fundamental challenge in multimodal environments, where factors like occlusion, camouflage, and lighting variations significantly hinder performance. Traditional RGB-based detection methods often fail under such adverse conditions, motivating the need for more robust, modality-agnostic approaches. In this work, we present HiddenObject, a fusion framework that integrates RGB, thermal, and depth data using a Mamba-based fusion mechanism. Our method captures complementary signals across modalities, enabling enhanced detection of obscured or camouflaged targets. Specifically, the proposed approach identifies modality-specific features and fuses them in a unified representation that generalizes well across challenging scenarios. We validate HiddenObject across multiple benchmark datasets, demonstrating state-of-the-art or competitive performance compared to existing methods. These results highlight the efficacy of our fusion design and expose key limitations in current unimodal and na\"ive fusion strategies. More broadly, our findings suggest that Mamba-based fusion architectures can significantly advance the field of multimodal object detection, especially under visually degraded or complex conditions.
>
---
#### [replaced 003] When and How Does CLIP Enable Domain and Compositional Generalization?
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.09507v3](http://arxiv.org/pdf/2502.09507v3)**

> **作者:** Elias Kempf; Simon Schrodi; Max Argus; Thomas Brox
>
> **备注:** ICML 2025 (Spotlight)
>
> **摘要:** The remarkable generalization performance of contrastive vision-language models like CLIP is often attributed to the diversity of their training distributions. However, key questions remain unanswered: Can CLIP generalize to an entirely unseen domain when trained on a diverse mixture of domains (domain generalization)? Can it generalize to unseen classes within partially seen domains (compositional generalization)? What factors affect such generalization? To answer these questions, we trained CLIP models on systematically constructed training distributions with controlled domain diversity and object class exposure. Our experiments show that domain diversity is essential for both domain and compositional generalization, yet compositional generalization can be surprisingly weaker than domain generalization when the training distribution contains a suboptimal subset of the test domain. Through data-centric and mechanistic analyses, we find that successful generalization requires the learning of sufficiently shared representations in intermediate layers and circuits.
>
---
#### [replaced 004] Similarity-based Outlier Detection for Noisy Object Re-Identification Using Beta Mixtures
- **分类: cs.CV; cs.AI; cs.LG; math.ST; stat.ML; stat.TH**

- **链接: [http://arxiv.org/pdf/2509.08926v2](http://arxiv.org/pdf/2509.08926v2)**

> **作者:** Waqar Ahmad; Evan Murphy; Vladimir A. Krylov
>
> **摘要:** Object re-identification (Re-ID) methods are highly sensitive to label noise, which typically leads to significant performance degradation. We address this challenge by reframing Re-ID as a supervised image similarity task and adopting a Siamese network architecture trained to capture discriminative pairwise relationships. Central to our approach is a novel statistical outlier detection (OD) framework, termed Beta-SOD (Beta mixture Similarity-based Outlier Detection), which models the distribution of cosine similarities between embedding pairs using a two-component Beta distribution mixture model. We establish a novel identifiability result for mixtures of two Beta distributions, ensuring that our learning task is well-posed. The proposed OD step complements the Re-ID architecture combining binary cross-entropy, contrastive, and cosine embedding losses that jointly optimize feature-level similarity learning.We demonstrate the effectiveness of Beta-SOD in de-noising and Re-ID tasks for person Re-ID, on CUHK03 and Market-1501 datasets, and vehicle Re-ID, on VeRi-776 dataset. Our method shows superior performance compared to the state-of-the-art methods across various noise levels (10-30\%), demonstrating both robustness and broad applicability in noisy Re-ID scenarios. The implementation of Beta-SOD is available at: github.com/waqar3411/Beta-SOD
>
---
#### [replaced 005] Talk2PC: Enhancing 3D Visual Grounding through LiDAR and Radar Point Clouds Fusion for Autonomous Driving
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.08336v2](http://arxiv.org/pdf/2503.08336v2)**

> **作者:** Runwei Guan; Jianan Liu; Ningwei Ouyang; Shaofeng Liang; Daizong Liu; Xiaolou Sun; Lianqing Zheng; Ming Xu; Yutao Yue; Guoqiang Mao; Hui Xiong
>
> **备注:** 13 pages, 12 figures
>
> **摘要:** Embodied outdoor scene understanding forms the foundation for autonomous agents to perceive, analyze, and react to dynamic driving environments. However, existing 3D understanding is predominantly based on 2D Vision-Language Models (VLMs), which collect and process limited scene-aware contexts. In contrast, compared to the 2D planar visual information, point cloud sensors such as LiDAR provide rich depth and fine-grained 3D representations of objects. Even better the emerging 4D millimeter-wave radar detects the motion trend, velocity, and reflection intensity of each object. The integration of these two modalities provides more flexible querying conditions for natural language, thereby supporting more accurate 3D visual grounding. To this end, we propose a novel method called TPCNet, the first outdoor 3D visual grounding model upon the paradigm of prompt-guided point cloud sensor combination, including both LiDAR and radar sensors. To optimally combine the features of these two sensors required by the prompt, we design a multi-fusion paradigm called Two-Stage Heterogeneous Modal Adaptive Fusion. Specifically, this paradigm initially employs Bidirectional Agent Cross-Attention (BACA), which feeds both-sensor features, characterized by global receptive fields, to the text features for querying. Moreover, we design a Dynamic Gated Graph Fusion (DGGF) module to locate the regions of interest identified by the queries. To further enhance accuracy, we devise an C3D-RECHead, based on the nearest object edge to the ego-vehicle. Experimental results demonstrate that our TPCNet, along with its individual modules, achieves the state-of-the-art performance on both the Talk2Radar and Talk2Car datasets. We release the code at https://github.com/GuanRunwei/TPCNet.
>
---
#### [replaced 006] Earth Observation Foundation Model PhilEO: Pretraining on the MajorTOM and FastTOM Datasets
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.14765v2](http://arxiv.org/pdf/2506.14765v2)**

> **作者:** Nikolaos Dionelis; Jente Bosmans; Riccardo Musto; Giancarlo Paoletti; Simone Sarti; Giacomo Cascarano; Casper Fibaek; Luke Camilleri; Bertrand Le Saux; Nicolas Longépé
>
> **备注:** 15 pages, 22 figures, 2 tables, 64 references
>
> **摘要:** Today, Earth Observation (EO) satellites generate massive volumes of data, with the Copernicus Sentinel-2 constellation alone producing approximately 1.6TB per day. To fully exploit this information, it is essential to pretrain EO Foundation Models (FMs) on large unlabeled datasets, enabling efficient fine-tuning for several different downstream tasks with minimal labeled data. In this work, we present the scaling-up of our recently proposed EO Foundation Model, PhilEO Geo-Aware U-Net, on the unlabeled 23TB dataset MajorTOM, which covers the vast majority of the Earth's surface, as well as on the specialized subset FastTOM 2TB that does not include oceans and ice. We develop and study various PhilEO model variants with different numbers of parameters and architectures. We fine-tune the models on the PhilEO Bench for road density estimation, building density pixel-wise regression, and land cover semantic segmentation, and we evaluate the performance. Our results demonstrate that for all n-shots for road density regression, the PhilEO 44M MajorTOM 23TB model outperforms PhilEO Globe 0.5TB 44M. We also show that for most n-shots for road density estimation and building density regression, PhilEO 200M FastTOM outperforms all the other models. The effectiveness of both dataset and model scaling is validated using the PhilEO Bench. We also study the impact of architecture scaling, transitioning from U-Net Convolutional Neural Networks (CNN) to Vision Transformers (ViT).
>
---
#### [replaced 007] AdaFusion: Prompt-Guided Inference with Adaptive Fusion of Pathology Foundation Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.05084v2](http://arxiv.org/pdf/2508.05084v2)**

> **作者:** Yuxiang Xiao; Yang Hu; Bin Li; Tianyang Zhang; Zexi Li; Huazhu Fu; Jens Rittscher; Kaixiang Yang
>
> **备注:** 6 Tables, 11 Figures
>
> **摘要:** Pathology foundation models (PFMs) have demonstrated strong representational capabilities through self-supervised pre-training on large-scale, unannotated histopathology image datasets. However, their diverse yet opaque pretraining contexts, shaped by both data-related and structural/training factors, introduce latent biases that hinder generalisability and transparency in downstream applications. In this paper, we propose AdaFusion, a novel prompt-guided inference framework that, to our knowledge, is among the very first to dynamically integrate complementary knowledge from multiple PFMs. Our method compresses and aligns tile-level features from diverse models and employs a lightweight attention mechanism to adaptively fuse them based on tissue phenotype context. We evaluate AdaFusion on three real-world benchmarks spanning treatment response prediction, tumour grading, and spatial gene expression inference. Our approach consistently surpasses individual PFMs across both classification and regression tasks, while offering interpretable insights into each model's biosemantic specialisation. These results highlight AdaFusion's ability to bridge heterogeneous PFMs, achieving both enhanced performance and interpretability of model-specific inductive biases.
>
---
#### [replaced 008] JARVIS-VLA: Post-Training Large-Scale Vision Language Models to Play Visual Games with Keyboards and Mouse
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.16365v2](http://arxiv.org/pdf/2503.16365v2)**

> **作者:** Muyao Li; Zihao Wang; Kaichen He; Xiaojian Ma; Yitao Liang
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Recently, action-based decision-making in open-world environments has gained significant attention. Visual Language Action (VLA) models, pretrained on large-scale web datasets, have shown promise in decision-making tasks. However, previous work has primarily focused on action post-training, often neglecting enhancements to the foundational model itself. In response, we introduce a novel approach, Act from Visual Language Post-Training, which refines Visual Language Models (VLMs) through visual and linguistic guidance in a self-supervised manner. This enhancement improves the models' capabilities in world knowledge, visual recognition, and spatial grounding in open-world environments. Following the above post-training paradigms, we obtain the first VLA models in Minecraft that can follow human instructions on over 1k different atomic tasks, including crafting, smelting, cooking, mining, and killing. Our experiments demonstrate that post-training on non-trajectory tasks leads to a significant 40% improvement over the best agent baseline on a diverse set of atomic tasks. Furthermore, we demonstrate that our approach surpasses traditional imitation learning-based policies in Minecraft, achieving state-of-the-art performance. We have open-sourced the code, models, and datasets to foster further research. The project page can be found in https://craftjarvis.github.io/JarvisVLA.
>
---
#### [replaced 009] Taccel: Scaling Up Vision-based Tactile Robotics via High-performance GPU Simulation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.12908v2](http://arxiv.org/pdf/2504.12908v2)**

> **作者:** Yuyang Li; Wenxin Du; Chang Yu; Puhao Li; Zihang Zhao; Tengyu Liu; Chenfanfu Jiang; Yixin Zhu; Siyuan Huang
>
> **摘要:** Tactile sensing is crucial for achieving human-level robotic capabilities in manipulation tasks. As a promising solution, Vision-Based Tactile Sensors (VBTSs) offer high spatial resolution and cost-effectiveness, but present unique challenges in robotics for their complex physical characteristics and visual signal processing requirements. The lack of efficient and accurate simulation tools for VBTSs has significantly limited the scale and scope of tactile robotics research. We present Taccel, a high-performance simulation platform that integrates IPC and ABD to model robots, tactile sensors, and objects with both accuracy and unprecedented speed, achieving an 18-fold acceleration over real-time across thousands of parallel environments. Unlike previous simulators that operate at sub-real-time speeds with limited parallelization, Taccel provides precise physics simulation and realistic tactile signals while supporting flexible robot-sensor configurations through user-friendly APIs. Through extensive validation in object recognition, robotic grasping, and articulated object manipulation, we demonstrate precise simulation and successful sim-to-real transfer. These capabilities position Taccel as a powerful tool for scaling up tactile robotics research and development, potentially transforming how robots interact with and understand their physical environment.
>
---
#### [replaced 010] AdvI2I: Adversarial Image Attack on Image-to-Image Diffusion models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.21471v3](http://arxiv.org/pdf/2410.21471v3)**

> **作者:** Yaopei Zeng; Yuanpu Cao; Bochuan Cao; Yurui Chang; Jinghui Chen; Lu Lin
>
> **摘要:** Recent advances in diffusion models have significantly enhanced the quality of image synthesis, yet they have also introduced serious safety concerns, particularly the generation of Not Safe for Work (NSFW) content. Previous research has demonstrated that adversarial prompts can be used to generate NSFW content. However, such adversarial text prompts are often easily detectable by text-based filters, limiting their efficacy. In this paper, we expose a previously overlooked vulnerability: adversarial image attacks targeting Image-to-Image (I2I) diffusion models. We propose AdvI2I, a novel framework that manipulates input images to induce diffusion models to generate NSFW content. By optimizing a generator to craft adversarial images, AdvI2I circumvents existing defense mechanisms, such as Safe Latent Diffusion (SLD), without altering the text prompts. Furthermore, we introduce AdvI2I-Adaptive, an enhanced version that adapts to potential countermeasures and minimizes the resemblance between adversarial images and NSFW concept embeddings, making the attack more resilient against defenses. Through extensive experiments, we demonstrate that both AdvI2I and AdvI2I-Adaptive can effectively bypass current safeguards, highlighting the urgent need for stronger security measures to address the misuse of I2I diffusion models.
>
---
#### [replaced 011] LoFi: Vision-Aided Label Generator for Wi-Fi Localization and Tracking
- **分类: cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2412.05074v4](http://arxiv.org/pdf/2412.05074v4)**

> **作者:** Zijian Zhao; Tingwei Chen; Fanyi Meng; Zhijie Cai; Hang Li; Xiaoyang Li; Guangxu Zhu
>
> **摘要:** Data-driven Wi-Fi localization and tracking have shown great promise due to their lower reliance on specialized hardware compared to model-based methods. However, most existing data collection techniques provide only coarse-grained ground truth or a limited number of labeled points, significantly hindering the advancement of data-driven approaches. While systems like lidar can deliver precise ground truth, their high costs make them inaccessible to many users. To address these challenges, we propose LoFi, a vision-aided label generator for Wi-Fi localization and tracking. LoFi can generate ground truth position coordinates solely from 2D images, offering high precision, low cost, and ease of use. Utilizing our method, we have compiled a Wi-Fi tracking and localization dataset using the ESP32-S3 and a webcam. The code and dataset of this paper are available at https://github.com/RS2002/LoFi.
>
---
#### [replaced 012] SPECS: Specificity-Enhanced CLIP-Score for Long Image Caption Evaluation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.03897v2](http://arxiv.org/pdf/2509.03897v2)**

> **作者:** Xiaofu Chen; Israfel Salazar; Yova Kementchedjhieva
>
> **摘要:** As interest grows in generating long, detailed image captions, standard evaluation metrics become increasingly unreliable. N-gram-based metrics though efficient, fail to capture semantic correctness. Representational Similarity (RS) metrics, designed to address this, initially saw limited use due to high computational costs, while today, despite advances in hardware, they remain unpopular due to low correlation to human judgments. Meanwhile, metrics based on large language models (LLMs) show strong correlation with human judgments, but remain too expensive for iterative use during model development. We introduce SPECS (Specificity-Enhanced CLIPScore), a reference-free RS metric tailored to long image captioning. SPECS modifies CLIP with a new objective that emphasizes specificity: rewarding correct details and penalizing incorrect ones. We show that SPECS matches the performance of open-source LLM-based metrics in correlation to human judgments, while being far more efficient. This makes it a practical alternative for iterative checkpoint evaluation during image captioning model development.Our code can be found at https://github.com/mbzuai-nlp/SPECS.
>
---
#### [replaced 013] Uncovering Neuroimaging Biomarkers of Brain Tumor Surgery with AI-Driven Methods
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.04881v2](http://arxiv.org/pdf/2507.04881v2)**

> **作者:** Carmen Jimenez-Mesa; Yizhou Wan; Guilio Sansone; Francisco J. Martinez-Murcia; Javier Ramirez; Pietro Lio; Juan M. Gorriz; Stephen J. Price; John Suckling; Michail Mamalakis
>
> **备注:** 18 pages, 6 figures
>
> **摘要:** Brain tumor resection is a highly complex procedure with profound implications for survival and quality of life. Predicting patient outcomes is crucial to guide clinicians in balancing oncological control with preservation of neurological function. However, building reliable prediction models is severely limited by the rarity of curated datasets that include both pre- and post-surgery imaging, given the clinical, logistical and ethical challenges of collecting such data. In this study, we develop a novel framework that integrates explainable artificial intelligence (XAI) with neuroimaging-based feature engineering for survival assessment in brain tumor patients. We curated structural MRI data from 49 patients scanned pre- and post-surgery, providing a rare resource for identifying survival-related biomarkers. A key methodological contribution is the development of a global explanation optimizer, which refines survival-related feature attribution in deep learning models, thereby improving both the interpretability and reliability of predictions. From a clinical perspective, our findings provide important evidence that survival after oncological surgery is influenced by alterations in regions related to cognitive and sensory functions. These results highlight the importance of preserving areas involved in decision-making and emotional regulation to improve long-term outcomes. From a technical perspective, the proposed optimizer advances beyond state-of-the-art XAI methods by enhancing both the fidelity and comprehensibility of model explanations, thus reinforcing trust in the recognition patterns driving survival prediction. This work demonstrates the utility of XAI-driven neuroimaging analysis in identifying survival-related variability and underscores its potential to inform precision medicine strategies in brain tumor treatment.
>
---
#### [replaced 014] DiFlow-TTS: Discrete Flow Matching with Factorized Speech Tokens for Low-Latency Zero-Shot Text-To-Speech
- **分类: cs.SD; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.09631v2](http://arxiv.org/pdf/2509.09631v2)**

> **作者:** Ngoc-Son Nguyen; Hieu-Nghia Huynh-Nguyen; Thanh V. T. Tran; Truong-Son Hy; Van Nguyen
>
> **摘要:** Zero-shot Text-to-Speech (TTS) aims to synthesize high-quality speech that mimics the voice of an unseen speaker using only a short reference sample, requiring not only speaker adaptation but also accurate modeling of prosodic attributes. Recent approaches based on language models, diffusion, and flow matching have shown promising results in zero-shot TTS, but still suffer from slow inference and repetition artifacts. Discrete codec representations have been widely adopted for speech synthesis, and recent works have begun to explore diffusion models in purely discrete settings, suggesting the potential of discrete generative modeling for speech synthesis. However, existing flow-matching methods typically embed these discrete tokens into a continuous space and apply continuous flow matching, which may not fully leverage the advantages of discrete representations. To address these challenges, we introduce DiFlow-TTS, which, to the best of our knowledge, is the first model to explore purely Discrete Flow Matching for speech synthesis. DiFlow-TTS explicitly models factorized speech attributes within a compact and unified architecture. It leverages in-context learning by conditioning on textual content, along with prosodic and acoustic attributes extracted from a reference speech, enabling effective attribute cloning in a zero-shot setting. In addition, the model employs a factorized flow prediction mechanism with distinct heads for prosody and acoustic details, allowing it to learn aspect-specific distributions. Experimental results demonstrate that DiFlow-TTS achieves promising performance in several key metrics, including naturalness, prosody, preservation of speaker style, and energy control. It also maintains a compact model size and achieves low-latency inference, generating speech up to 25.8 times faster than the latest existing baselines.
>
---
#### [replaced 015] Just Say the Word: Annotation-Free Fine-Grained Object Counting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.11705v3](http://arxiv.org/pdf/2504.11705v3)**

> **作者:** Adriano D'Alessandro; Ali Mahdavi-Amiri; Ghassan Hamarneh
>
> **备注:** data - https://dalessandro.dev/datasets/lookalikes/
>
> **摘要:** Fine-grained object counting remains a major challenge for class-agnostic counting models, which overcount visually similar but incorrect instances (e.g., jalape\~no vs. poblano). Addressing this by annotating new data and fully retraining the model is time-consuming and does not guarantee generalization to additional novel categories at test time. Instead, we propose an alternative paradigm: Given a category name, tune a compact concept embedding derived from the prompt using synthetic images and pseudo-labels generated by a text-to-image diffusion model. This embedding conditions a specialization module that refines raw overcounts from any frozen counter into accurate, category-specific estimates\textemdash without requiring real images or human annotations. We validate our approach on \textsc{Lookalikes}, a challenging new benchmark containing 1,037 images across 27 fine-grained subcategories, and show substantial improvements over strong baselines. Code will be released upon acceptance. Dataset - https://dalessandro.dev/datasets/lookalikes/
>
---
#### [replaced 016] TSGCNeXt: Dynamic-Static Multi-Graph Convolution for Efficient Skeleton-Based Action Recognition with Long-term Learning Potential
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2304.11631v2](http://arxiv.org/pdf/2304.11631v2)**

> **作者:** Dongjingdin Liu; Pengpeng Chen; Miao Yao; Yijing Lu; Zijie Cai; Yuxin Tian
>
> **摘要:** Skeleton-based action recognition has achieved remarkable results in human action recognition with the development of graph convolutional networks (GCNs). However, the recent works tend to construct complex learning mechanisms with redundant training and exist a bottleneck for long time-series. To solve these problems, we propose the Temporal-Spatio Graph ConvNeXt (TSGCNeXt) to explore efficient learning mechanism of long temporal skeleton sequences. Firstly, a new graph learning mechanism with simple structure, Dynamic-Static Separate Multi-graph Convolution (DS-SMG) is proposed to aggregate features of multiple independent topological graphs and avoid the node information being ignored during dynamic convolution. Next, we construct a graph convolution training acceleration mechanism to optimize the back-propagation computing of dynamic graph learning with 55.08\% speed-up. Finally, the TSGCNeXt restructure the overall structure of GCN with three Spatio-temporal learning modules,efficiently modeling long temporal features. In comparison with existing previous methods on large-scale datasets NTU RGB+D 60 and 120, TSGCNeXt outperforms on single-stream networks. In addition, with the ema model introduced into the multi-stream fusion, TSGCNeXt achieves SOTA levels. On the cross-subject and cross-set of the NTU 120, accuracies reach 90.22% and 91.74%.
>
---
#### [replaced 017] Bridging the Gap: A Framework for Real-World Video Deepfake Detection via Social Network Compression Emulation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.08765v2](http://arxiv.org/pdf/2508.08765v2)**

> **作者:** Andrea Montibeller; Dasara Shullani; Daniele Baracchi; Alessandro Piva; Giulia Boato
>
> **摘要:** The growing presence of AI-generated videos on social networks poses new challenges for deepfake detection, as detectors trained under controlled conditions often fail to generalize to real-world scenarios. A key factor behind this gap is the aggressive, proprietary compression applied by platforms like YouTube and Facebook, which launder low-level forensic cues. However, replicating these transformations at scale is difficult due to API limitations and data-sharing constraints. For these reasons, we propose a first framework that emulates the video sharing pipelines of social networks by estimating compression and resizing parameters from a small set of uploaded videos. These parameters enable a local emulator capable of reproducing platform-specific artifacts on large datasets without direct API access. Experiments on FaceForensics++ videos shared via social networks demonstrate that our emulated data closely matches the degradation patterns of real uploads. Furthermore, detectors fine-tuned on emulated videos achieve comparable performance to those trained on actual shared media. Our approach offers a scalable and practical solution for bridging the gap between lab-based training and real-world deployment of deepfake detectors, particularly in the underexplored domain of compressed video content.
>
---
#### [replaced 018] Can Generative Geospatial Diffusion Models Excel as Discriminative Geospatial Foundation Models?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07890v2](http://arxiv.org/pdf/2503.07890v2)**

> **作者:** Yuru Jia; Valerio Marsocci; Ziyang Gong; Xue Yang; Maarten Vergauwen; Andrea Nascetti
>
> **备注:** ICCV 2025, camera ready
>
> **摘要:** Self-supervised learning (SSL) has revolutionized representation learning in Remote Sensing (RS), advancing Geospatial Foundation Models (GFMs) to leverage vast unlabeled satellite imagery for diverse downstream tasks. Currently, GFMs primarily employ objectives like contrastive learning or masked image modeling, owing to their proven success in learning transferable representations. However, generative diffusion models, which demonstrate the potential to capture multi-grained semantics essential for RS tasks during image generation, remain underexplored for discriminative applications. This prompts the question: can generative diffusion models also excel and serve as GFMs with sufficient discriminative power? In this work, we answer this question with SatDiFuser, a framework that transforms a diffusion-based generative geospatial foundation model into a powerful pretraining tool for discriminative RS. By systematically analyzing multi-stage, noise-dependent diffusion features, we develop three fusion strategies to effectively leverage these diverse representations. Extensive experiments on remote sensing benchmarks show that SatDiFuser outperforms state-of-the-art GFMs, achieving gains of up to +5.7% mIoU in semantic segmentation and +7.9% F1-score in classification, demonstrating the capacity of diffusion-based generative foundation models to rival or exceed discriminative GFMs. The source code is available at: https://github.com/yurujaja/SatDiFuser.
>
---
#### [replaced 019] MoPD: Mixture-of-Prompts Distillation for Vision-Language Models
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.19087v2](http://arxiv.org/pdf/2412.19087v2)**

> **作者:** Yang Chen; Shuai Fu; Yu Zhang
>
> **摘要:** Soft prompt learning methods are effective for adapting vision-language models (VLMs) to downstream tasks. Nevertheless, empirical evidence reveals a tendency of existing methods that they overfit seen classes and exhibit degraded performance on unseen classes. This limitation is due to the inherent bias in the training data towards the seen classes. To address this issue, we propose a novel soft prompt learning method, named Mixture-of-Prompts Distillation (MoPD), which can effectively transfer useful knowledge from hard prompts manually hand-crafted (a.k.a. teacher prompts) to the learnable soft prompt (a.k.a. student prompt), thereby enhancing the generalization ability of soft prompts on unseen classes. Moreover, the proposed MoPD method utilizes a gating network that learns to select hard prompts used for prompt distillation. Extensive experiments demonstrate that the proposed MoPD method outperforms state-of-the-art baselines especially on on unseen classes.
>
---
#### [replaced 020] ANTS: Shaping the Adaptive Negative Textual Space by MLLM for OOD Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.03951v2](http://arxiv.org/pdf/2509.03951v2)**

> **作者:** Wenjie Zhu; Yabin Zhang; Xin Jin; Wenjun Zeng; Lei Zhang
>
> **摘要:** The introduction of negative labels (NLs) has proven effective in enhancing Out-of-Distribution (OOD) detection. However, existing methods often lack an understanding of OOD images, making it difficult to construct an accurate negative space. In addition, the presence of false negative labels significantly degrades their near-OOD performance. To address these issues, we propose shaping an Adaptive Negative Textual Space (ANTS) by leveraging the understanding and reasoning capabilities of multimodal large language models (MLLMs). Specifically, we identify images likely to be OOD samples as negative images and prompt the MLLM to describe these images, generating expressive negative sentences that precisely characterize the OOD distribution and enhance far-OOD detection. For the near-OOD setting, where OOD samples resemble the in-distribution (ID) subset, we first identify the subset of ID classes that are visually similar to negative images and then leverage the reasoning capability of MLLMs to generate visually similar negative labels tailored to this subset, effectively reducing false negatives and improving near-OOD detection. To balance these two types of negative textual spaces, we design an adaptive weighted score that enables the method to handle different OOD task settings (near-OOD and far-OOD) without relying on task-specific prior knowledge, making it highly adaptable in open environments. On the ImageNet benchmark, our ANTS significantly reduces the FPR95 by 4.2\%, establishing a new state-of-the-art. Furthermore, our method is training-free and zero-shot, enabling high scalability.
>
---
#### [replaced 021] PL-Net: Progressive Learning Network for Medical Image Segmentation
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2110.14484v3](http://arxiv.org/pdf/2110.14484v3)**

> **作者:** Kunpeng Mao; Ruoyu Li; Junlong Cheng; Danmei Huang; Zhiping Song; ZeKui Liu
>
> **摘要:** In recent years, deep convolutional neural network-based segmentation methods have achieved state-of-the-art performance for many medical analysis tasks. However, most of these approaches rely on optimizing the U-Net structure or adding new functional modules, which overlooks the complementation and fusion of coarse-grained and fine-grained semantic information. To address these issues, we propose a 2D medical image segmentation framework called Progressive Learning Network (PL-Net), which comprises Internal Progressive Learning (IPL) and External Progressive Learning (EPL). PL-Net offers the following advantages: (1) IPL divides feature extraction into two steps, allowing for the mixing of different size receptive fields and capturing semantic information from coarse to fine granularity without introducing additional parameters; (2) EPL divides the training process into two stages to optimize parameters and facilitate the fusion of coarse-grained information in the first stage and fine-grained information in the second stage. We conducted comprehensive evaluations of our proposed method on five medical image segmentation datasets, and the experimental results demonstrate that PL-Net achieves competitive segmentation performance. It is worth noting that PL-Net does not introduce any additional learnable parameters compared to other U-Net variants.
>
---
#### [replaced 022] MedM-VL: What Makes a Good Medical LVLM?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.04323v3](http://arxiv.org/pdf/2504.04323v3)**

> **作者:** Yiming Shi; Shaoshuai Yang; Xun Zhu; Haoyu Wang; Xiangling Fu; Miao Li; Ji Wu
>
> **摘要:** Medical image analysis is essential in modern healthcare. Deep learning has redirected research focus toward complex medical multimodal tasks, including report generation and visual question answering. Traditional task-specific models often fall short in handling these challenges. Large vision-language models (LVLMs) offer new solutions for solving such tasks. In this study, we build on the popular LLaVA framework to systematically explore model architectures and training strategies for both 2D and 3D medical LVLMs. We present extensive empirical findings and practical guidance. To support reproducibility and future research, we release a modular codebase, MedM-VL, and two pre-trained models: MedM-VL-2D for 2D medical image analysis and MedM-VL-CT-Chest for 3D CT-based applications. The code is available at: https://github.com/MSIIP/MedM-VL
>
---
#### [replaced 023] Integrative Variational Autoencoders for Generative Modeling of an Image Outcome with Multiple Input Images
- **分类: eess.IV; cs.CV; cs.NE; stat.AP; stat.ML**

- **链接: [http://arxiv.org/pdf/2402.02734v2](http://arxiv.org/pdf/2402.02734v2)**

> **作者:** Bowen Lei; Yeseul Jeon; Rajarshi Guhaniyogi; Aaron Scheffler; Bani Mallick; Alzheimer's Disease Neuroimaging Initiatives
>
> **摘要:** Understanding relationships across multiple imaging modalities is central to neuroimaging research. We introduce the Integrative Variational Autoencoder (InVA), the first hierarchical VAE framework for image-on-image regression in multimodal neuroimaging. Unlike standard VAEs, which are not designed for predictive integration across modalities, InVA models outcome images as functions of both shared and modality-specific features. This flexible, data-driven approach avoids rigid assumptions of classical tensor regression and outperforms conventional VAEs and nonlinear models such as BART. As a key application, InVA accurately predicts costly PET scans from structural MRI, offering an efficient and powerful tool for multimodal neuroimaging.
>
---
#### [replaced 024] Dynamic Motion Blending for Versatile Motion Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.20724v2](http://arxiv.org/pdf/2503.20724v2)**

> **作者:** Nan Jiang; Hongjie Li; Ziye Yuan; Zimo He; Yixin Chen; Tengyu Liu; Yixin Zhu; Siyuan Huang
>
> **摘要:** Text-guided motion editing enables high-level semantic control and iterative modifications beyond traditional keyframe animation. Existing methods rely on limited pre-collected training triplets, which severely hinders their versatility in diverse editing scenarios. We introduce MotionCutMix, an online data augmentation technique that dynamically generates training triplets by blending body part motions based on input text. While MotionCutMix effectively expands the training distribution, the compositional nature introduces increased randomness and potential body part incoordination. To model such a rich distribution, we present MotionReFit, an auto-regressive diffusion model with a motion coordinator. The auto-regressive architecture facilitates learning by decomposing long sequences, while the motion coordinator mitigates the artifacts of motion composition. Our method handles both spatial and temporal motion edits directly from high-level human instructions, without relying on additional specifications or Large Language Models. Through extensive experiments, we show that MotionReFit achieves state-of-the-art performance in text-guided motion editing.
>
---
#### [replaced 025] Towards Reliable Audio Deepfake Attribution and Model Recognition: A Multi-Level Autoencoder-Based Framework
- **分类: cs.SD; cs.CV; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.02521v3](http://arxiv.org/pdf/2508.02521v3)**

> **作者:** Andrea Di Pierno; Luca Guarnera; Dario Allegra; Sebastiano Battiato
>
> **摘要:** The proliferation of audio deepfakes poses a growing threat to trust in digital communications. While detection methods have advanced, attributing audio deepfakes to their source models remains an underexplored yet crucial challenge. In this paper we introduce LAVA (Layered Architecture for Voice Attribution), a hierarchical framework for audio deepfake detection and model recognition that leverages attention-enhanced latent representations extracted by a convolutional autoencoder trained solely on fake audio. Two specialized classifiers operate on these features: Audio Deepfake Attribution (ADA), which identifies the generation technology, and Audio Deepfake Model Recognition (ADMR), which recognize the specific generative model instance. To improve robustness under open-set conditions, we incorporate confidence-based rejection thresholds. Experiments on ASVspoof2021, FakeOrReal, and CodecFake show strong performance: the ADA classifier achieves F1-scores over 95% across all datasets, and the ADMR module reaches 96.31% macro F1 across six classes. Additional tests on unseen attacks from ASVpoof2019 LA and error propagation analysis confirm LAVA's robustness and reliability. The framework advances the field by introducing a supervised approach to deepfake attribution and model recognition under open-set conditions, validated on public benchmarks and accompanied by publicly released models and code. Models and code are available at https://www.github.com/adipiz99/lava-framework.
>
---
#### [replaced 026] Self-Rewarding Large Vision-Language Models for Optimizing Prompts in Text-to-Image Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16763v2](http://arxiv.org/pdf/2505.16763v2)**

> **作者:** Hongji Yang; Yucheng Zhou; Wencheng Han; Jianbing Shen
>
> **备注:** Accepted by ACL2025 Findings
>
> **摘要:** Text-to-image models are powerful for producing high-quality images based on given text prompts, but crafting these prompts often requires specialized vocabulary. To address this, existing methods train rewriting models with supervision from large amounts of manually annotated data and trained aesthetic assessment models. To alleviate the dependence on data scale for model training and the biases introduced by trained models, we propose a novel prompt optimization framework, designed to rephrase a simple user prompt into a sophisticated prompt to a text-to-image model. Specifically, we employ the large vision language models (LVLMs) as the solver to rewrite the user prompt, and concurrently, employ LVLMs as a reward model to score the aesthetics and alignment of the images generated by the optimized prompt. Instead of laborious human feedback, we exploit the prior knowledge of the LVLM to provide rewards, i.e., AI feedback. Simultaneously, the solver and the reward model are unified into one model and iterated in reinforcement learning to achieve self-improvement by giving a solution and judging itself. Results on two popular datasets demonstrate that our method outperforms other strong competitors.
>
---
#### [replaced 027] Region-Wise Correspondence Prediction between Manga Line Art Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.09501v2](http://arxiv.org/pdf/2509.09501v2)**

> **作者:** Yingxuan Li; Jiafeng Mao; Qianru Qiu; Yusuke Matsui
>
> **摘要:** Understanding region-wise correspondence between manga line art images is a fundamental task in manga processing, enabling downstream applications such as automatic line art colorization and in-between frame generation. However, this task remains largely unexplored, especially in realistic scenarios without pre-existing segmentation or annotations. In this paper, we introduce a novel and practical task: predicting region-wise correspondence between raw manga line art images without any pre-existing labels or masks. To tackle this problem, we divide each line art image into a set of patches and propose a Transformer-based framework that learns patch-level similarities within and across images. We then apply edge-aware clustering and a region matching algorithm to convert patch-level predictions into coherent region-level correspondences. To support training and evaluation, we develop an automatic annotation pipeline and manually refine a subset of the data to construct benchmark datasets. Experiments on multiple datasets demonstrate that our method achieves high patch-level accuracy (e.g., 96.34%) and generates consistent region-level correspondences, highlighting its potential for real-world manga applications.
>
---
#### [replaced 028] Backdoor Poisoning Attack Against Face Spoofing Attack Detection Methods
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.03108v2](http://arxiv.org/pdf/2509.03108v2)**

> **作者:** Shota Iwamatsu; Koichi Ito; Takafumi Aoki
>
> **备注:** 2025 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)
>
> **摘要:** Face recognition systems are robust against environmental changes and noise, and thus may be vulnerable to illegal authentication attempts using user face photos, such as spoofing attacks. To prevent such spoofing attacks, it is crucial to discriminate whether the input image is a live user image or a spoofed image prior to the face recognition process. Most existing spoofing attack detection methods utilize deep learning, which necessitates a substantial amount of training data. Consequently, if malicious data is injected into a portion of the training dataset, a specific spoofing attack may be erroneously classified as live, leading to false positives. In this paper, we propose a novel backdoor poisoning attack method to demonstrate the latent threat of backdoor poisoning within face anti-spoofing detection. The proposed method enables certain spoofing attacks to bypass detection by embedding features extracted from the spoofing attack's face image into a live face image without inducing any perceptible visual alterations. Through experiments conducted on public datasets, we demonstrate that the proposed method constitutes a realistic threat to existing spoofing attack detection systems.
>
---
#### [replaced 029] SFD-Mamba2Net: Structure-Guided Frequency-Enhanced Dual-Stream Mamba2 Network for Coronary Artery Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.08934v2](http://arxiv.org/pdf/2509.08934v2)**

> **作者:** Nan Mu; Ruiqi Song; Zhihui Xu; Jingfeng Jiang; Chen Zhao
>
> **摘要:** Background: Coronary Artery Disease (CAD) is one of the leading causes of death worldwide. Invasive Coronary Angiography (ICA), regarded as the gold standard for CAD diagnosis, necessitates precise vessel segmentation and stenosis detection. However, ICA images are typically characterized by low contrast, high noise levels, and complex, fine-grained vascular structures, which pose significant challenges to the clinical adoption of existing segmentation and detection methods. Objective: This study aims to improve the accuracy of coronary artery segmentation and stenosis detection in ICA images by integrating multi-scale structural priors, state-space-based long-range dependency modeling, and frequency-domain detail enhancement strategies. Methods: We propose SFD-Mamba2Net, an end-to-end framework tailored for ICA-based vascular segmentation and stenosis detection. In the encoder, a Curvature-Aware Structural Enhancement (CASE) module is embedded to leverage multi-scale responses for highlighting slender tubular vascular structures, suppressing background interference, and directing attention toward vascular regions. In the decoder, we introduce a Progressive High-Frequency Perception (PHFP) module that employs multi-level wavelet decomposition to progressively refine high-frequency details while integrating low-frequency global structures. Results and Conclusions: SFD-Mamba2Net consistently outperformed state-of-the-art methods across eight segmentation metrics, and achieved the highest true positive rate and positive predictive value in stenosis detection.
>
---
#### [replaced 030] OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval
- **分类: cs.IR; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.07879v3](http://arxiv.org/pdf/2505.07879v3)**

> **作者:** Wei Yang; Jingjing Fu; Rui Wang; Jinyu Wang; Lei Song; Jiang Bian
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** Vision-language retrieval-augmented generation (RAG) has become an effective approach for tackling Knowledge-Based Visual Question Answering (KB-VQA), which requires external knowledge beyond the visual content presented in images. The effectiveness of Vision-language RAG systems hinges on multimodal retrieval, which is inherently challenging due to the diverse modalities and knowledge granularities in both queries and knowledge bases. Existing methods have not fully tapped into the potential interplay between these elements. We propose a multimodal RAG system featuring a coarse-to-fine, multi-step retrieval that harmonizes multiple granularities and modalities to enhance efficacy. Our system begins with a broad initial search aligning knowledge granularity for cross-modal retrieval, followed by a multimodal fusion reranking to capture the nuanced multimodal information for top entity selection. A text reranker then filters out the most relevant fine-grained section for augmented generation. Extensive experiments on the InfoSeek and Encyclopedic-VQA benchmarks show our method achieves state-of-the-art retrieval performance and highly competitive answering results, underscoring its effectiveness in advancing KB-VQA systems.
>
---
#### [replaced 031] GROVE: A Generalized Reward for Learning Open-Vocabulary Physical Skill
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.04191v2](http://arxiv.org/pdf/2504.04191v2)**

> **作者:** Jieming Cui; Tengyu Liu; Ziyu Meng; Jiale Yu; Ran Song; Wei Zhang; Yixin Zhu; Siyuan Huang
>
> **摘要:** Learning open-vocabulary physical skills for simulated agents presents a significant challenge in artificial intelligence. Current reinforcement learning approaches face critical limitations: manually designed rewards lack scalability across diverse tasks, while demonstration-based methods struggle to generalize beyond their training distribution. We introduce GROVE, a generalized reward framework that enables open-vocabulary physical skill learning without manual engineering or task-specific demonstrations. Our key insight is that Large Language Models(LLMs) and Vision Language Models(VLMs) provide complementary guidance -- LLMs generate precise physical constraints capturing task requirements, while VLMs evaluate motion semantics and naturalness. Through an iterative design process, VLM-based feedback continuously refines LLM-generated constraints, creating a self-improving reward system. To bridge the domain gap between simulation and natural images, we develop Pose2CLIP, a lightweight mapper that efficiently projects agent poses directly into semantic feature space without computationally expensive rendering. Extensive experiments across diverse embodiments and learning paradigms demonstrate GROVE's effectiveness, achieving 22.2% higher motion naturalness and 25.7% better task completion scores while training 8.4x faster than previous methods. These results establish a new foundation for scalable physical skill acquisition in simulated environments.
>
---
#### [replaced 032] OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.09332v2](http://arxiv.org/pdf/2509.09332v2)**

> **作者:** Yuecheng Liu; Dafeng Chi; Shiguang Wu; Zhanguang Zhang; Yuzheng Zhuang; Bowen Yang; He Zhu; Lingfeng Zhang; Pengwei Xie; David Gamaliel Arcos Bravo; Yingxue Zhang; Jianye Hao; Xingyue Quan
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have opened new opportunities for embodied intelligence, enabling multimodal understanding, reasoning, and interaction, as well as continuous spatial decision-making. Nevertheless, current MLLM-based embodied systems face two critical limitations. First, Geometric Adaptability Gap: models trained solely on 2D inputs or with hard-coded 3D geometry injection suffer from either insufficient spatial information or restricted 2D generalization, leading to poor adaptability across tasks with diverse spatial demands. Second, Embodiment Constraint Gap: prior work often neglects the physical constraints and capacities of real robots, resulting in task plans that are theoretically valid but practically infeasible. To address these gaps, we introduce OmniEVA -- an embodied versatile planner that enables advanced embodied reasoning and task planning through two pivotal innovations: (1) a Task-Adaptive 3D Grounding mechanism, which introduces a gated router to perform explicit selective regulation of 3D fusion based on contextual requirements, enabling context-aware 3D grounding for diverse embodied tasks. (2) an Embodiment-Aware Reasoning framework that jointly incorporates task goals and embodiment constraints into the reasoning loop, resulting in planning decisions that are both goal-directed and executable. Extensive experimental results demonstrate that OmniEVA not only achieves state-of-the-art general embodied reasoning performance, but also exhibits a strong ability across a wide range of downstream scenarios. Evaluations of a suite of proposed embodied benchmarks, including both primitive and composite tasks, confirm its robust and versatile planning capabilities. Project page: https://omnieva.github.io
>
---
#### [replaced 033] Out-Of-Distribution Detection for Audio-visual Generalized Zero-Shot Learning: A General Framework
- **分类: cs.MM; cs.CV; cs.SD; eess.AS; eess.IV**

- **链接: [http://arxiv.org/pdf/2408.01284v2](http://arxiv.org/pdf/2408.01284v2)**

> **作者:** Liuyuan Wen
>
> **备注:** Accepted to BMVC 2024
>
> **摘要:** Generalized Zero-Shot Learning (GZSL) is a challenging task requiring accurate classification of both seen and unseen classes. Within this domain, Audio-visual GZSL emerges as an extremely exciting yet difficult task, given the inclusion of both visual and acoustic features as multi-modal inputs. Existing efforts in this field mostly utilize either embedding-based or generative-based methods. However, generative training is difficult and unstable, while embedding-based methods often encounter domain shift problem. Thus, we find it promising to integrate both methods into a unified framework to leverage their advantages while mitigating their respective disadvantages. Our study introduces a general framework employing out-of-distribution (OOD) detection, aiming to harness the strengths of both approaches. We first employ generative adversarial networks to synthesize unseen features, enabling the training of an OOD detector alongside classifiers for seen and unseen classes. This detector determines whether a test feature belongs to seen or unseen classes, followed by classification utilizing separate classifiers for each feature type. We test our framework on three popular audio-visual datasets and observe a significant improvement comparing to existing state-of-the-art works. Codes can be found in https://github.com/liuyuan-wen/AV-OOD-GZSL.
>
---
#### [replaced 034] Afford-X: Generalizable and Slim Affordance Reasoning for Task-oriented Manipulation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.03556v2](http://arxiv.org/pdf/2503.03556v2)**

> **作者:** Xiaomeng Zhu; Yuyang Li; Leiyao Cui; Pengfei Li; Huan-ang Gao; Yixin Zhu; Hao Zhao
>
> **摘要:** Object affordance reasoning, the ability to infer object functionalities based on physical properties, is fundamental for task-oriented planning and activities in both humans and Artificial Intelligence (AI). This capability, required for planning and executing daily activities in a task-oriented manner, relies on commonsense knowledge of object physics and functionalities, extending beyond simple object recognition. Current computational models for affordance reasoning from perception lack generalizability, limiting their applicability in novel scenarios. Meanwhile, comprehensive Large Language Models (LLMs) with emerging reasoning capabilities are challenging to deploy on local devices for task-oriented manipulations. Here, we introduce LVIS-Aff, a large-scale dataset comprising 1,496 tasks and 119k images, designed to enhance the generalizability of affordance reasoning from perception. Utilizing this dataset, we develop Afford-X, an end-to-end trainable affordance reasoning model that incorporates Verb Attention and Bi-Fusion modules to improve multi-modal understanding. This model achieves up to a 12.1% performance improvement over the best-reported results from non-LLM methods, while also demonstrating a 1.2% enhancement compared to our previous conference paper. Additionally, it maintains a compact 187M parameter size and infers nearly 50 times faster than the GPT-4V API. Our work demonstrates the potential for efficient, generalizable affordance reasoning models that can be deployed on local devices for task-oriented manipulations. We showcase Afford-X's effectiveness in enabling task-oriented manipulations for robots across various tasks and environments, underscoring its efficiency and broad implications for advancing robotics and AI systems in real-world applications.
>
---
#### [replaced 035] DRespNeT: A UAV Dataset and YOLOv8-DRN Model for Aerial Instance Segmentation of Building Access Points for Post-Earthquake Search-and-Rescue Missions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.16016v2](http://arxiv.org/pdf/2508.16016v2)**

> **作者:** Aykut Sirma; Angelos Plastropoulos; Gilbert Tang; Argyrios Zolotas
>
> **备注:** Technical Paper of Scientific data paper: UAV imagery dataset from 2023 Turkiye earthquakes, annotated for instance segmentation to support SAR robotics. Initial version of the Dataset is released: https://figshare.com/s/66d3116a0de5b7d827fb and https://universe.roboflow.com/cranfield-university-dwusz/phd-project-instance-segmentation
>
> **摘要:** Recent advancements in computer vision and deep learning have enhanced disaster-response capabilities, particularly in the rapid assessment of earthquake-affected urban environments. Timely identification of accessible entry points and structural obstacles is essential for effective search-and-rescue (SAR) operations. To address this need, we introduce DRespNeT, a high-resolution dataset specifically developed for aerial instance segmentation of post-earthquake structural environments. Unlike existing datasets, which rely heavily on satellite imagery or coarse semantic labeling, DRespNeT provides detailed polygon-level instance segmentation annotations derived from high-definition (1080p) aerial footage captured in disaster zones, including the 2023 Turkiye earthquake and other impacted regions. The dataset comprises 28 operationally critical classes, including structurally compromised buildings, access points such as doors, windows, and gaps, multiple debris levels, rescue personnel, vehicles, and civilian visibility. A distinctive feature of DRespNeT is its fine-grained annotation detail, enabling differentiation between accessible and obstructed areas, thereby improving operational planning and response efficiency. Performance evaluations using YOLO-based instance segmentation models, specifically YOLOv8-seg, demonstrate significant gains in real-time situational awareness and decision-making. Our optimized YOLOv8-DRN model achieves 92.7% mAP50 with an inference speed of 27 FPS on an RTX-4090 GPU for multi-target detection, meeting real-time operational requirements. The dataset and models support SAR teams and robotic systems, providing a foundation for enhancing human-robot collaboration, streamlining emergency response, and improving survivor outcomes.
>
---
#### [replaced 036] Geometry and Perception Guided Gaussians for Multiview-consistent 3D Generation from a Single Image
- **分类: cs.CV; 68; I.4.0**

- **链接: [http://arxiv.org/pdf/2506.21152v2](http://arxiv.org/pdf/2506.21152v2)**

> **作者:** Pufan Li; Bi'an Du; Wei Hu
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Generating realistic 3D objects from single-view images requires natural appearance, 3D consistency, and the ability to capture multiple plausible interpretations of unseen regions. Existing approaches often rely on fine-tuning pretrained 2D diffusion models or directly generating 3D information through fast network inference or 3D Gaussian Splatting, but their results generally suffer from poor multiview consistency and lack geometric detail. To tackle these issues, we present a novel method that seamlessly integrates geometry and perception information without requiring additional model training to reconstruct detailed 3D objects from a single image. Specifically, we incorporate geometry and perception priors to initialize the Gaussian branches and guide their parameter optimization. The geometry prior captures the rough 3D shapes, while the perception prior utilizes the 2D pretrained diffusion model to enhance multiview information. Subsequently, we introduce a stable Score Distillation Sampling for fine-grained prior distillation to ensure effective knowledge transfer. The model is further enhanced by a reprojection-based strategy that enforces depth consistency. Experimental results show that we outperform existing methods on novel view synthesis and 3D reconstruction, demonstrating robust and consistent 3D object generation.
>
---
#### [replaced 037] Survivability of Backdoor Attacks on Unconstrained Face Recognition Systems
- **分类: cs.CV; cs.AI; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.01607v3](http://arxiv.org/pdf/2507.01607v3)**

> **作者:** Quentin Le Roux; Yannick Teglia; Teddy Furon; Philippe Loubet-Moundi; Eric Bourbao
>
> **摘要:** The widespread deployment of Deep Learning-based Face Recognition Systems raises multiple security concerns. While prior research has identified backdoor vulnerabilities on isolated components, Backdoor Attacks on real-world, unconstrained pipelines remain underexplored. This paper presents the first comprehensive system-level analysis of Backdoor Attacks targeting Face Recognition Systems and provides three contributions. We first show that face feature extractors trained with large margin metric learning losses are susceptible to Backdoor Attacks. By analyzing 20 pipeline configurations and 15 attack scenarios, we then reveal that a single backdoor can compromise an entire Face Recognition System. Finally, we propose effective best practices and countermeasures for stakeholders.
>
---
#### [replaced 038] Efficient and Effective Adaptation of Multimodal Foundation Models in Sequential Recommendation
- **分类: cs.IR; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.02992v2](http://arxiv.org/pdf/2411.02992v2)**

> **作者:** Junchen Fu; Xuri Ge; Xin Xin; Alexandros Karatzoglou; Ioannis Arapakis; Kaiwen Zheng; Yongxin Ni; Joemon M. Jose
>
> **备注:** Accepted by IEEE Transactions on Knowledge and Data Engineering (TKDE)
>
> **摘要:** Multimodal foundation models (MFMs) have revolutionized sequential recommender systems through advanced representation learning. While Parameter-efficient Fine-tuning (PEFT) is commonly used to adapt these models, studies often prioritize parameter efficiency, neglecting GPU memory and training speed. To address this, we introduced the IISAN framework, significantly enhancing efficiency. However, IISAN was limited to symmetrical MFMs and identical text and image encoders, preventing the use of state-of-the-art Large Language Models. To overcome this, we developed IISAN-Versa, a versatile plug-and-play architecture compatible with both symmetrical and asymmetrical MFMs. IISAN-Versa employs a Decoupled PEFT structure and utilizes both intra- and inter-modal adaptation. It effectively handles asymmetry through a simple yet effective combination of group layer-dropping and dimension transformation alignment. Our research demonstrates that IISAN-Versa effectively adapts large text encoders, and we further identify a scaling effect where larger encoders generally perform better. IISAN-Versa also demonstrates strong versatility in our defined multimodal scenarios, which include raw titles and captions generated from images and videos. Additionally, IISAN-Versa achieved state-of-the-art performance on the Microlens public benchmark. We release our code at https://github.com/GAIR-Lab/IISAN.
>
---
#### [replaced 039] GeoDE: a Geographically Diverse Evaluation Dataset for Object Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2301.02560v4](http://arxiv.org/pdf/2301.02560v4)**

> **作者:** Vikram V. Ramaswamy; Sing Yu Lin; Dora Zhao; Aaron B. Adcock; Laurens van der Maaten; Deepti Ghadiyaram; Olga Russakovsky
>
> **备注:** Published at NeurIPS D&B, 2023
>
> **摘要:** Current dataset collection methods typically scrape large amounts of data from the web. While this technique is extremely scalable, data collected in this way tends to reinforce stereotypical biases, can contain personally identifiable information, and typically originates from Europe and North America. In this work, we rethink the dataset collection paradigm and introduce GeoDE, a geographically diverse dataset with 61,940 images from 40 classes and 6 world regions, with no personally identifiable information, collected by soliciting images from people around the world. We analyse GeoDE to understand differences in images collected in this manner compared to web-scraping. We demonstrate its use as both an evaluation and training dataset, allowing us to highlight and begin to mitigate the shortcomings in current models, despite GeoDE's relatively small size. We release the full dataset and code at https://geodiverse-data-collection.cs.princeton.edu
>
---
#### [replaced 040] Orientation Scores should be a Piece of Cake
- **分类: math.DG; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.00702v3](http://arxiv.org/pdf/2504.00702v3)**

> **作者:** Finn M. Sherry; Chase van de Geijn; Erik J. Bekkers; Remco Duits
>
> **备注:** Accepted in the 7th International Conference on Geometric Science of Information
>
> **摘要:** We axiomatically derive a family of wavelets for an orientation score, lifting from position space $\mathbb{R}^2$ to position and orientation space $\mathbb{R}^2\times S^1$, with fast reconstruction property, that minimise position-orientation uncertainty. We subsequently show that these minimum uncertainty states are well-approximated by cake wavelets: for standard parameters, the uncertainty gap of cake wavelets is less than 1.1, and in the limit, we prove the uncertainty gap tends to the minimum of 1. Next, we complete a previous theoretical argument that one does not have to train the lifting layer in (PDE-)G-CNNs, but can instead use cake wavelets. Finally, we show experimentally that in this way we can reduce the network complexity and improve the interpretability of (PDE-)G-CNNs, with only a slight impact on the model's performance.
>
---
#### [replaced 041] OneCAT: Decoder-Only Auto-Regressive Model for Unified Understanding and Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.03498v2](http://arxiv.org/pdf/2509.03498v2)**

> **作者:** Han Li; Xinyu Peng; Yaoming Wang; Zelin Peng; Xin Chen; Rongxiang Weng; Jingang Wang; Xunliang Cai; Wenrui Dai; Hongkai Xiong
>
> **备注:** technical report, project url:https://onecat-ai.github.io/
>
> **摘要:** We introduce OneCAT, a unified multimodal model that seamlessly integrates understanding, generation, and editing within a novel, pure decoder-only transformer architecture. Our framework uniquely eliminates the need for external components such as Vision Transformers (ViT) or vision tokenizer during inference, leading to significant efficiency gains, especially for high-resolution inputs. This is achieved through a modality-specific Mixture-of-Experts (MoE) structure trained with a single autoregressive (AR) objective, which also natively supports dynamic resolutions. Furthermore, we pioneer a multi-scale visual autoregressive mechanism within the Large Language Model (LLM) that drastically reduces decoding steps compared to diffusion-based methods while maintaining state-of-the-art performance. Our findings demonstrate the powerful potential of pure autoregressive modeling as a sufficient and elegant foundation for unified multimodal intelligence. As a result, OneCAT sets a new performance standard, outperforming existing open-source unified multimodal models across benchmarks for multimodal generation, editing, and understanding.
>
---
#### [replaced 042] Building Age Estimation: A New Multi-Modal Benchmark Dataset and Community Challenge
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13818v4](http://arxiv.org/pdf/2502.13818v4)**

> **作者:** Nikolaos Dionelis; Alessandra Feliciotti; Mattia Marconcini; Devis Peressutti; Nika Oman Kadunc; JaeWan Park; Hagai Raja Sinulingga; Steve Andreas Immanuel; Ba Tran; Caroline Arnold; Nicolas Longépé
>
> **备注:** 16 pages, 20 figures, 1 table, Submitted
>
> **摘要:** Estimating the construction year of buildings is critical for advancing sustainability, as older structures often lack energy-efficient features. Sustainable urban planning relies on accurate building age data to reduce energy consumption and mitigate climate change. In this work, we introduce MapYourCity, a novel multi-modal benchmark dataset comprising top-view Very High Resolution (VHR) imagery, multi-spectral Earth Observation (EO) data from the Copernicus Sentinel-2 satellite constellation, and co-localized street-view images across various European cities. Each building is labeled with its construction epoch, and the task is formulated as a seven-class classification problem covering periods from 1900 to the present. To advance research in EO generalization and multi-modal learning, we organized a community-driven data challenge in 2024, hosted by ESA $\Phi$-lab, which ran for four months and attracted wide participation. This paper presents the Top-4 performing models from the challenge and their evaluation results. We assess model generalization on cities excluded from training to prevent data leakage, and evaluate performance under missing modality scenarios, particularly when street-view data is unavailable. Results demonstrate that building age estimation is both feasible and effective, even in previously unseen cities and when relying solely on top-view satellite imagery (i.e. with VHR and Sentinel-2 images). The MapYourCity dataset thus provides a valuable resource for developing scalable, real-world solutions in sustainable urban analytics.
>
---
#### [replaced 043] The Weighting Game: Evaluating Quality of Explainability Methods
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2208.06175v2](http://arxiv.org/pdf/2208.06175v2)**

> **作者:** Lassi Raatikainen; Esa Rahtu
>
> **备注:** Published in: Image Analysis (SCIA 2025), Lecture Notes in Computer Science (LNCS), vol. 15726, pp. 325-338 (2025). This is the submitted-manuscript (pre-review) version. v2: added required preprint notice and updated metadata. Version of Record: see DOI 10.1007/978-3-031-95918-9_23
>
> **摘要:** The objective of this paper is to assess the quality of explanation heatmaps for image classification tasks. To assess the quality of explainability methods, we approach the task through the lens of accuracy and stability. In this work, we make the following contributions. Firstly, we introduce the Weighting Game, which measures how much of a class-guided explanation is contained within the correct class' segmentation mask. Secondly, we introduce a metric for explanation stability, using zooming/panning transformations to measure differences between saliency maps with similar contents. Quantitative experiments are produced, using these new metrics, to evaluate the quality of explanations provided by commonly used CAM methods. The quality of explanations is also contrasted between different model architectures, with findings highlighting the need to consider model architecture when choosing an explainability method.
>
---
#### [replaced 044] Evaluating the Evaluators: Towards Human-aligned Metrics for Missing Markers Reconstruction
- **分类: cs.CV; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.14334v4](http://arxiv.org/pdf/2410.14334v4)**

> **作者:** Taras Kucherenko; Derek Peristy; Judith Bütepage
>
> **备注:** Accepted at the ACM International Conference on Multimedia 2025 (ACM MM'25)
>
> **摘要:** Animation data is often obtained through optical motion capture systems, which utilize a multitude of cameras to establish the position of optical markers. However, system errors or occlusions can result in missing markers, the manual cleaning of which can be time-consuming. This has sparked interest in machine learning-based solutions for missing marker reconstruction in the academic community. Most academic papers utilize a simplistic mean square error as the main metric. In this paper, we show that this metric does not correlate with subjective perception of the fill quality. Additionally, we introduce and evaluate a set of better-correlated metrics that can drive progress in the field.
>
---
#### [replaced 045] Your Image is Secretly the Last Frame of a Pseudo Video
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.20158v3](http://arxiv.org/pdf/2410.20158v3)**

> **作者:** Wenlong Chen; Wenlin Chen; Lapo Rastrelli; Yingzhen Li
>
> **备注:** Presented at the ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy (DeLTa). 1-frame results for CIFAR10 in Table 2 corrected. Code released
>
> **摘要:** Diffusion models, which can be viewed as a special case of hierarchical variational autoencoders (HVAEs), have shown profound success in generating photo-realistic images. In contrast, standard HVAEs often produce images of inferior quality compared to diffusion models. In this paper, we hypothesize that the success of diffusion models can be partly attributed to the additional self-supervision information for their intermediate latent states provided by corrupted images, which along with the original image form a pseudo video. Based on this hypothesis, we explore the possibility of improving other types of generative models with such pseudo videos. Specifically, we first extend a given image generative model to their video generative model counterpart, and then train the video generative model on pseudo videos constructed by applying data augmentation to the original images. Furthermore, we analyze the potential issues of first-order Markov data augmentation methods, which are typically used in diffusion models, and propose to use more expressive data augmentation to construct more useful information in pseudo videos. Our empirical results on the CIFAR10 and CelebA datasets demonstrate that improved image generation quality can be achieved with additional self-supervised information from pseudo videos.
>
---
#### [replaced 046] Hybrid Swin Attention Networks for Simultaneously Low-Dose PET and CT Denoising
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.06591v4](http://arxiv.org/pdf/2509.06591v4)**

> **作者:** Yichao Liu; Hengzhi Xue; YueYang Teng
>
> **摘要:** Low-dose computed tomography (LDCT) and positron emission tomography (PET) have emerged as safer alternatives to conventional imaging modalities by significantly reducing radiation exposure. However, this reduction often results in increased noise and artifacts, which can compromise diagnostic accuracy. Consequently, denoising for LDCT/PET has become a vital area of research aimed at enhancing image quality while maintaining radiation safety. In this study, we introduce a novel Hybrid Swin Attention Network (HSANet), which incorporates Efficient Global Attention (EGA) modules and a hybrid upsampling module. The EGA modules enhance both spatial and channel-wise interaction, improving the network's capacity to capture relevant features, while the hybrid upsampling module mitigates the risk of overfitting to noise. We validate the proposed approach using a publicly available LDCT/PET dataset. Experimental results demonstrate that HSANet achieves superior denoising performance compared to existing methods, while maintaining a lightweight model size suitable for deployment on GPUs with standard memory configurations. This makes our approach highly practical for real-world clinical applications.
>
---
