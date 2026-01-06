# 计算机视觉 cs.CV

- **最新发布 205 篇**

- **更新 92 篇**

## 最新发布

#### [new 001] Adaptive Hybrid Optimizer based Framework for Lumpy Skin Disease Identification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于疾病识别任务，旨在解决LSD早期检测问题。提出LUMPNet框架，结合YOLOv11和EfficientNet，使用自适应优化器提升检测精度与效率。**

- **链接: [https://arxiv.org/pdf/2601.01807v1](https://arxiv.org/pdf/2601.01807v1)**

> **作者:** Ubaidullah; Muhammad Abid Hussain; Mohsin Raza Jafri; Rozi Khan; Moid Sandhu; Abd Ullah Khan; Hyundong Shin
>
> **摘要:** Lumpy Skin Disease (LSD) is a contagious viral infection that significantly deteriorates livestock health, thereby posing a serious threat to the global economy and food security. Owing to its rapid spread characteristics, early and precise identification is crucial to prevent outbreaks and ensure timely intervention. In this paper, we propose a hybrid deep learning-based approach called LUMPNet for the early detection of LSD. LUMPNet utilizes image data to detect and classify skin nodules -- the primary indicator of LSD. To this end, LUMPNet uses YOLOv11, EfficientNet-based CNN classifier with compound scaling, and a novel adaptive hybrid optimizer. More precisely, LUMPNet detects and localizes LSD skin nodules and lesions on cattle images. It exploits EfficientNet to classify the localized cattle images into LSD-affected or healthy categories. To stabilize and accelerate the training of YOLOv11 and EfficientNet hybrid model, a novel adaptive hybrid optimizer is proposed and utilized. We evaluate LUMPNet at various stages of LSD using a publicly available dataset. Results indicate that the proposed scheme achieves 99% LSD detection training accuracy, and outperforms existing schemes. The model also achieves validation accuracy of 98%. Moreover, for further evaluation, we conduct a case study using an optimized EfficientNet-B0 model trained with the AdamW optimizer, and compare its performance with LUMPNet. The results show that LUMPNet achieves superior performance.
>
---
#### [new 002] Few-Shot Video Object Segmentation in X-Ray Angiography Using Local Matching and Spatio-Temporal Consistency Loss
- **分类: cs.CV**

- **简介: 该论文属于视频目标分割任务，解决X射线血管造影视频中少样本分割问题。提出局部匹配策略和时空一致性损失，提升分割精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.00988v1](https://arxiv.org/pdf/2601.00988v1)**

> **作者:** Lin Xi; Yingliang Ma; Xiahai Zhuang
>
> **摘要:** We introduce a novel FSVOS model that employs a local matching strategy to restrict the search space to the most relevant neighboring pixels. Rather than relying on inefficient standard im2col-like implementations (e.g., spatial convolutions, depthwise convolutions and feature-shifting mechanisms) or hardware-specific CUDA kernels (e.g., deformable and neighborhood attention), which often suffer from limited portability across non-CUDA devices, we reorganize the local sampling process through a direction-based sampling perspective. Specifically, we implement a non-parametric sampling mechanism that enables dynamically varying sampling regions. This approach provides the flexibility to adapt to diverse spatial structures without the computational costs of parametric layers and the need for model retraining. To further enhance feature coherence across frames, we design a supervised spatio-temporal contrastive learning scheme that enforces consistency in feature representations. In addition, we introduce a publicly available benchmark dataset for multi-object segmentation in X-ray angiography videos (MOSXAV), featuring detailed, manually labeled segmentation ground truth. Extensive experiments on the CADICA, XACV, and MOSXAV datasets show that our proposed FSVOS method outperforms current state-of-the-art video segmentation methods in terms of segmentation accuracy and generalization capability (i.e., seen and unseen categories). This work offers enhanced flexibility and potential for a wide range of clinical applications.
>
---
#### [new 003] Crowded Video Individual Counting Informed by Social Grouping and Spatial-Temporal Displacement Priors
- **分类: cs.CV**

- **简介: 该论文属于视频个体计数任务，解决拥挤场景下行人匹配与计数问题。提出新数据集和OMAN++模型，引入社会分组与时空位移先验提升计数精度。**

- **链接: [https://arxiv.org/pdf/2601.01192v1](https://arxiv.org/pdf/2601.01192v1)**

> **作者:** Hao Lu; Xuhui Zhu; Wenjing Zhang; Yanan Li; Xiang Bai
>
> **备注:** Journal Extension of arXiv:2506.13067
>
> **摘要:** Video Individual Counting (VIC) is a recently introduced task aiming to estimate pedestrian flux from a video. It extends Video Crowd Counting (VCC) beyond the per-frame pedestrian count. In contrast to VCC that learns to count pedestrians across frames, VIC must identify co-existent pedestrians between frames, which turns out to be a correspondence problem. Existing VIC approaches, however, can underperform in congested scenes such as metro commuting. To address this, we build WuhanMetroCrowd, one of the first VIC datasets that characterize crowded, dynamic pedestrian flows. It features sparse-to-dense density levels, short-to-long video clips, slow-to-fast flow variations, front-to-back appearance changes, and light-to-heavy occlusions. To better adapt VIC approaches to crowds, we rethink the nature of VIC and recognize two informative priors: i) the social grouping prior that indicates pedestrians tend to gather in groups and ii) the spatial-temporal displacement prior that informs an individual cannot teleport physically. The former inspires us to relax the standard one-to-one (O2O) matching used by VIC to one-to-many (O2M) matching, implemented by an implicit context generator and a O2M matcher; the latter facilitates the design of a displacement prior injector, which strengthens not only O2M matching but also feature extraction and model training. These designs jointly form a novel and strong VIC baseline OMAN++. Extensive experiments show that OMAN++ not only outperforms state-of-the-art VIC baselines on the standard SenseCrowd, CroHD, and MovingDroneCrowd benchmarks, but also indicates a clear advantage in crowded scenes, with a 38.12% error reduction on our WuhanMetroCrowd dataset. Code, data, and pretrained models are available at https://github.com/tiny-smart/OMAN.
>
---
#### [new 004] Leveraging 2D-VLM for Label-Free 3D Segmentation in Large-Scale Outdoor Scene Understanding
- **分类: cs.CV**

- **简介: 该论文属于3D语义分割任务，解决无标注数据和RGB配对的问题。通过2D模型和自然语言提示实现3D分割，支持开放词汇识别。**

- **链接: [https://arxiv.org/pdf/2601.02029v1](https://arxiv.org/pdf/2601.02029v1)**

> **作者:** Toshihiko Nishimura; Hirofumi Abe; Kazuhiko Murasaki; Taiga Yoshida; Ryuichi Tanida
>
> **备注:** 19
>
> **摘要:** This paper presents a novel 3D semantic segmentation method for large-scale point cloud data that does not require annotated 3D training data or paired RGB images. The proposed approach projects 3D point clouds onto 2D images using virtual cameras and performs semantic segmentation via a foundation 2D model guided by natural language prompts. 3D segmentation is achieved by aggregating predictions from multiple viewpoints through weighted voting. Our method outperforms existing training-free approaches and achieves segmentation accuracy comparable to supervised methods. Moreover, it supports open-vocabulary recognition, enabling users to detect objects using arbitrary text queries, thus overcoming the limitations of traditional supervised approaches.
>
---
#### [new 005] Enhanced Leukemic Cell Classification Using Attention-Based CNN and Data Augmentation
- **分类: cs.CV; cs.AI; cs.LG; cs.SE**

- **简介: 该论文属于医学图像分类任务，旨在解决急性淋巴细胞白血病细胞自动分类问题。通过结合注意力机制和数据增强，提出高效模型提升分类准确率。**

- **链接: [https://arxiv.org/pdf/2601.01026v1](https://arxiv.org/pdf/2601.01026v1)**

> **作者:** Douglas Costa Braga; Daniel Oliveira Dantas
>
> **备注:** 9 pages, 5 figures, 4 tables. Submitted to VISAPP 2025
>
> **摘要:** We present a reproducible deep learning pipeline for leukemic cell classification, focusing on system architecture, experimental robustness, and software design choices for medical image analysis. Acute lymphoblastic leukemia (ALL) is the most common childhood cancer, requiring expert microscopic diagnosis that suffers from inter-observer variability and time constraints. The proposed system integrates an attention-based convolutional neural network combining EfficientNetV2-B3 with Squeeze-and-Excitation mechanisms for automated ALL cell classification. Our approach employs comprehensive data augmentation, focal loss for class imbalance, and patient-wise data splitting to ensure robust and reproducible evaluation. On the C-NMC 2019 dataset (12,528 original images from 62 patients), the system achieves a 97.89% F1-score and 97.89% accuracy on the test set, with statistical validation through 100-iteration Monte Carlo experiments confirming significant improvements (p < 0.001) over baseline methods. The proposed pipeline outperforms existing approaches by up to 4.67% while using 89% fewer parameters than VGG16 (15.2M vs. 138M). The attention mechanism provides interpretable visualizations of diagnostically relevant cellular features, demonstrating that modern attention-based architectures can improve leukemic cell classification while maintaining computational efficiency suitable for clinical deployment.
>
---
#### [new 006] Improved Object-Centric Diffusion Learning with Registers and Contrastive Alignment
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于对象中心学习任务，解决对象槽混淆和对齐不足问题。提出CODA方法，通过注册槽和对比对齐损失提升对象发现与生成效果。**

- **链接: [https://arxiv.org/pdf/2601.01224v1](https://arxiv.org/pdf/2601.01224v1)**

> **作者:** Bac Nguyen; Yuhta Takida; Naoki Murata; Chieh-Hsin Lai; Toshimitsu Uesaka; Stefano Ermon; Yuki Mitsufuji
>
> **摘要:** Slot Attention (SA) with pretrained diffusion models has recently shown promise for object-centric learning (OCL), but suffers from slot entanglement and weak alignment between object slots and image content. We propose Contrastive Object-centric Diffusion Alignment (CODA), a simple extension that (i) employs register slots to absorb residual attention and reduce interference between object slots, and (ii) applies a contrastive alignment loss to explicitly encourage slot-image correspondence. The resulting training objective serves as a tractable surrogate for maximizing mutual information (MI) between slots and inputs, strengthening slot representation quality. On both synthetic (MOVi-C/E) and real-world datasets (VOC, COCO), CODA improves object discovery (e.g., +6.1% FG-ARI on COCO), property prediction, and compositional image generation over strong baselines. Register slots add negligible overhead, keeping CODA efficient and scalable. These results indicate potential applications of CODA as an effective framework for robust OCL in complex, real-world scenes.
>
---
#### [new 007] TalkPhoto: A Versatile Training-Free Conversational Assistant for Intelligent Image Editing
- **分类: cs.CV**

- **简介: 该论文提出TalkPhoto，一种无需训练的图像编辑框架，解决多任务编辑效率低的问题，通过对话交互调用现有方法实现精准编辑。**

- **链接: [https://arxiv.org/pdf/2601.01915v1](https://arxiv.org/pdf/2601.01915v1)**

> **作者:** Yujie Hu; Zecheng Tang; Xu Jiang; Weiqi Li; Jian Zhang
>
> **备注:** a Conversational Assistant for Intelligent Image Editing
>
> **摘要:** Thanks to the powerful language comprehension capabilities of Large Language Models (LLMs), existing instruction-based image editing methods have introduced Multimodal Large Language Models (MLLMs) to promote information exchange between instructions and images, ensuring the controllability and flexibility of image editing. However, these frameworks often build a multi-instruction dataset to train the model to handle multiple editing tasks, which is not only time-consuming and labor-intensive but also fails to achieve satisfactory results. In this paper, we present TalkPhoto, a versatile training-free image editing framework that facilitates precise image manipulation through conversational interaction. We instruct the open-source LLM with a specially designed prompt template to analyze user needs after receiving instructions and hierarchically invoke existing advanced editing methods, all without additional training. Moreover, we implement a plug-and-play and efficient invocation of image editing methods, allowing complex and unseen editing tasks to be integrated into the current framework, achieving stable and high-quality editing results. Extensive experiments demonstrate that our method not only provides more accurate invocation with fewer token consumption but also achieves higher editing quality across various image editing tasks.
>
---
#### [new 008] Domain Adaptation of Carotid Ultrasound Images using Generative Adversarial Network
- **分类: cs.CV**

- **简介: 该论文属于医学图像域适应任务，旨在解决不同设备或参数下超声图像的分布差异问题。通过改进的GAN模型，实现图像纹理调整和噪声去除，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.01460v1](https://arxiv.org/pdf/2601.01460v1)**

> **作者:** Mohd Usama; Belal Ahmad; Christer Gronlund; Faleh Menawer R Althiyabi
>
> **备注:** 15 pages, 9 figures, 4 tables
>
> **摘要:** Deep learning has been extensively used in medical imaging applications, assuming that the test and training datasets belong to the same probability distribution. However, a common challenge arises when working with medical images generated by different systems or even the same system with different parameter settings. Such images contain diverse textures and reverberation noise that violate the aforementioned assumption. Consequently, models trained on data from one device or setting often struggle to perform effectively with data from other devices or settings. In addition, retraining models for each specific device or setting is labor-intensive and costly. To address these issues in ultrasound images, we propose a novel Generative Adversarial Network (GAN)-based model. We formulated the domain adaptation tasks as an image-to-image translation task, in which we modified the texture patterns and removed reverberation noise in the test data images from the source domain to align with those in the target domain images while keeping the image content unchanged. We applied the proposed method to two datasets containing carotid ultrasound images from three different domains. The experimental results demonstrate that the model successfully translated the texture pattern of images and removed reverberation noise from the ultrasound images. Furthermore, we evaluated the CycleGAN approaches for a comparative study with the proposed model. The experimental findings conclusively demonstrated that the proposed model achieved domain adaptation (histogram correlation (0.960 (0.019), & 0.920 (0.043) and bhattacharya distance (0.040 (0.020), & 0.085 (0.048)), compared to no adaptation (0.916 (0.062) & 0.890 (0.077), 0.090 (0.070) & 0.121 (0.095)) for both datasets.
>
---
#### [new 009] EdgeNeRF: Edge-Guided Regularization for Neural Radiance Fields from Sparse Views
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决稀疏视角下NeRF几何失真问题。提出EdgeNeRF，通过边缘引导的深度和法线正则化，提升重建质量并保留边界细节。**

- **链接: [https://arxiv.org/pdf/2601.01431v1](https://arxiv.org/pdf/2601.01431v1)**

> **作者:** Weiqi Yu; Yiyang Yao; Lin He; Jianming Lv
>
> **备注:** PRCV 2025
>
> **摘要:** Neural Radiance Fields (NeRF) achieve remarkable performance in dense multi-view scenarios, but their reconstruction quality degrades significantly under sparse inputs due to geometric artifacts. Existing methods utilize global depth regularization to mitigate artifacts, leading to the loss of geometric boundary details. To address this problem, we propose EdgeNeRF, an edge-guided sparse-view 3D reconstruction algorithm. Our method leverages the prior that abrupt changes in depth and normals generate edges. Specifically, we first extract edges from input images, then apply depth and normal regularization constraints to non-edge regions, enhancing geometric consistency while preserving high-frequency details at boundaries. Experiments on LLFF and DTU datasets demonstrate EdgeNeRF's superior performance, particularly in retaining sharp geometric boundaries and suppressing artifacts. Additionally, the proposed edge-guided depth regularization module can be seamlessly integrated into other methods in a plug-and-play manner, significantly improving their performance without substantially increasing training time. Code is available at https://github.com/skyhigh404/edgenerf.
>
---
#### [new 010] MCD-Net: A Lightweight Deep Learning Baseline for Optical-Only Moraine Segmentation
- **分类: cs.CV**

- **简介: 该论文属于遥感图像分割任务，旨在解决光学影像中冰川垄（moraine）自动分割难题。通过构建数据集并提出轻量级网络MCD-Net，提升分割效率与精度。**

- **链接: [https://arxiv.org/pdf/2601.02091v1](https://arxiv.org/pdf/2601.02091v1)**

> **作者:** Zhehuan Cao; Fiseha Berhanu Tesema; Ping Fu; Jianfeng Ren; Ahmed Nasr
>
> **备注:** 13 pages, 10 figures. This manuscript is under review at IEEE Transactions on Geoscience and Remote Sensing
>
> **摘要:** Glacial segmentation is essential for reconstructing past glacier dynamics and evaluating climate-driven landscape change. However, weak optical contrast and the limited availability of high-resolution DEMs hinder automated mapping. This study introduces the first large-scale optical-only moraine segmentation dataset, comprising 3,340 manually annotated high-resolution images from Google Earth covering glaciated regions of Sichuan and Yunnan, China. We develop MCD-Net, a lightweight baseline that integrates a MobileNetV2 encoder, a Convolutional Block Attention Module (CBAM), and a DeepLabV3+ decoder. Benchmarking against deeper backbones (ResNet152, Xception) shows that MCD-Net achieves 62.3\% mean Intersection over Union (mIoU) and 72.8\% Dice coefficient while reducing computational cost by more than 60\%. Although ridge delineation remains constrained by sub-pixel width and spectral ambiguity, the results demonstrate that optical imagery alone can provide reliable moraine-body segmentation. The dataset and code are publicly available at https://github.com/Lyra-alpha/MCD-Net, establishing a reproducible benchmark for moraine-specific segmentation and offering a deployable baseline for high-altitude glacial monitoring.
>
---
#### [new 011] UnrealPose: Leveraging Game Engine Kinematics for Large-Scale Synthetic Human Pose Data
- **分类: cs.CV**

- **简介: 该论文提出UnrealPose-Gen工具，生成大规模高质量3D人体姿态数据，解决真实数据稀缺与标注困难问题，用于姿态估计等任务。**

- **链接: [https://arxiv.org/pdf/2601.00991v1](https://arxiv.org/pdf/2601.00991v1)**

> **作者:** Joshua Kawaguchi; Saad Manzur; Emily Gao Wang; Maitreyi Sinha; Bryan Vela; Yunxi Wang; Brandon Vela; Wayne B. Hayes
>
> **备注:** CVPR 2026 submission. Introduces UnrealPose-1M dataset and UnrealPose-Gen pipeline
>
> **摘要:** Diverse, accurately labeled 3D human pose data is expensive and studio-bound, while in-the-wild datasets lack known ground truth. We introduce UnrealPose-Gen, an Unreal Engine 5 pipeline built on Movie Render Queue for high-quality offline rendering. Our generated frames include: (i) 3D joints in world and camera coordinates, (ii) 2D projections and COCO-style keypoints with occlusion and joint-visibility flags, (iii) person bounding boxes, and (iv) camera intrinsics and extrinsics. We use UnrealPose-Gen to present UnrealPose-1M, an approximately one million frame corpus comprising eight sequences: five scripted "coherent" sequences spanning five scenes, approximately 40 actions, and five subjects; and three randomized sequences across three scenes, approximately 100 actions, and five subjects, all captured from diverse camera trajectories for broad viewpoint coverage. As a fidelity check, we report real-to-synthetic results on four tasks: image-to-3D pose, 2D keypoint detection, 2D-to-3D lifting, and person detection/segmentation. Though time and resources constrain us from an unlimited dataset, we release the UnrealPose-1M dataset, as well as the UnrealPose-Gen pipeline to support third-party generation of human pose data.
>
---
#### [new 012] Cross-Layer Attentive Feature Upsampling for Low-latency Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于语义分割任务，旨在解决低延迟下高分辨率特征生成的问题。提出GAI方法，通过注意力机制提升特征质量，实现高效准确的分割。**

- **链接: [https://arxiv.org/pdf/2601.01167v1](https://arxiv.org/pdf/2601.01167v1)**

> **作者:** Tianheng Cheng; Xinggang Wang; Junchao Liao; Wenyu Liu
>
> **摘要:** Semantic segmentation is a fundamental problem in computer vision and it requires high-resolution feature maps for dense prediction. Current coordinate-guided low-resolution feature interpolation methods, e.g., bilinear interpolation, produce coarse high-resolution features which suffer from feature misalignment and insufficient context information. Moreover, enriching semantics to high-resolution features requires a high computation burden, so that it is challenging to meet the requirement of lowlatency inference. We propose a novel Guided Attentive Interpolation (GAI) method to adaptively interpolate fine-grained high-resolution features with semantic features to tackle these issues. Guided Attentive Interpolation determines both spatial and semantic relations of pixels from features of different resolutions and then leverages these relations to interpolate high-resolution features with rich semantics. GAI can be integrated with any deep convolutional network for efficient semantic segmentation. In experiments, the GAI-based semantic segmentation networks, i.e., GAIN, can achieve78.8 mIoU with 22.3 FPS on Cityscapes and 80.6 mIoU with 64.5 on CamVid using an NVIDIA 1080Ti GPU, which are the new state-of-the-art results of low-latency semantic segmentation. Code and models are available at: https://github.com/hustvl/simpleseg.
>
---
#### [new 013] Efficient Unrolled Networks for Large-Scale 3D Inverse Problems
- **分类: cs.CV**

- **简介: 该论文属于3D成像逆问题任务，解决大尺度问题中无法有效集成成像算子的难题。通过域划分和算子近似，实现高效端到端重建模型。**

- **链接: [https://arxiv.org/pdf/2601.02141v1](https://arxiv.org/pdf/2601.02141v1)**

> **作者:** Romain Vo; Julián Tachella
>
> **摘要:** Deep learning-based methods have revolutionized the field of imaging inverse problems, yielding state-of-the-art performance across various imaging domains. The best performing networks incorporate the imaging operator within the network architecture, typically in the form of deep unrolling. However, in large-scale problems, such as 3D imaging, most existing methods fail to incorporate the operator in the architecture due to the prohibitive amount of memory required by global forward operators, which hinder typical patching strategies. In this work, we present a domain partitioning strategy and normal operator approximations that enable the training of end-to-end reconstruction models incorporating forward operators of arbitrarily large problems into their architecture. The proposed method achieves state-of-the-art performance on 3D X-ray cone-beam tomography and 3D multi-coil accelerated MRI, while requiring only a single GPU for both training and inference.
>
---
#### [new 014] In defense of the two-stage framework for open-set domain adaptive semantic segmentation
- **分类: cs.CV**

- **简介: 该论文属于开放集域适应语义分割任务，解决已知与未知类不平衡导致的负迁移问题。提出两阶段策略SATS，分离后适应，提升模型对未知类的识别能力。**

- **链接: [https://arxiv.org/pdf/2601.01439v1](https://arxiv.org/pdf/2601.01439v1)**

> **作者:** Wenqi Ren; Weijie Wang; Meng Zheng; Ziyan Wu; Yang Tang; Zhun Zhong; Nicu Sebe
>
> **摘要:** Open-Set Domain Adaptation for Semantic Segmentation (OSDA-SS) presents a significant challenge, as it requires both domain adaptation for known classes and the distinction of unknowns. Existing methods attempt to address both tasks within a single unified stage. We question this design, as the annotation imbalance between known and unknown classes often leads to negative transfer of known classes and underfitting for unknowns. To overcome these issues, we propose SATS, a Separating-then-Adapting Training Strategy, which addresses OSDA-SS through two sequential steps: known/unknown separation and unknown-aware domain adaptation. By providing the model with more accurate and well-aligned unknown classes, our method ensures a balanced learning of discriminative features for both known and unknown classes, steering the model toward discovering truly unknown objects. Additionally, we present hard unknown exploration, an innovative data augmentation method that exposes the model to more challenging unknowns, strengthening its ability to capture more comprehensive understanding of target unknowns. We evaluate our method on public OSDA-SS benchmarks. Experimental results demonstrate that our method achieves a substantial advancement, with a +3.85% H-Score improvement for GTA5-to-Cityscapes and +18.64% for SYNTHIA-to-Cityscapes, outperforming previous state-of-the-art methods.
>
---
#### [new 015] CTIS-QA: Clinical Template-Informed Slide-level Question Answering for Pathology
- **分类: cs.CV**

- **简介: 该论文提出CTIS-QA，用于病理学的滑片级问答任务，解决病理信息结构化与视觉-语言对齐问题。构建了CTIS-Align和CTIS-Bench数据集，设计双流模型提升诊断准确性。**

- **链接: [https://arxiv.org/pdf/2601.01769v1](https://arxiv.org/pdf/2601.01769v1)**

> **作者:** Hao Lu; Ziniu Qian; Yifu Li; Yang Zhou; Bingzheng Wei; Yan Xu
>
> **备注:** The paper has been accepted by BIBM 2025
>
> **摘要:** In this paper, we introduce a clinical diagnosis template-based pipeline to systematically collect and structure pathological information. In collaboration with pathologists and guided by the the College of American Pathologists (CAP) Cancer Protocols, we design a Clinical Pathology Report Template (CPRT) that ensures comprehensive and standardized extraction of diagnostic elements from pathology reports. We validate the effectiveness of our pipeline on TCGA-BRCA. First, we extract pathological features from reports using CPRT. These features are then used to build CTIS-Align, a dataset of 80k slide-description pairs from 804 WSIs for vision-language alignment training, and CTIS-Bench, a rigorously curated VQA benchmark comprising 977 WSIs and 14,879 question-answer pairs. CTIS-Bench emphasizes clinically grounded, closed-ended questions (e.g., tumor grade, receptor status) that reflect real diagnostic workflows, minimize non-visual reasoning, and require genuine slide understanding. We further propose CTIS-QA, a Slide-level Question Answering model, featuring a dual-stream architecture that mimics pathologists' diagnostic approach. One stream captures global slide-level context via clustering-based feature aggregation, while the other focuses on salient local regions through attention-guided patch perception module. Extensive experiments on WSI-VQA, CTIS-Bench, and slide-level diagnostic tasks show that CTIS-QA consistently outperforms existing state-of-the-art models across multiple metrics. Code and data are available at https://github.com/HLSvois/CTIS-QA.
>
---
#### [new 016] Parameter-Efficient Domain Adaption for CSI Crowd-Counting via Self-Supervised Learning with Adapter Modules
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于WiFi-based人群计数任务，解决领域适应问题。通过自监督学习和适配器模块，实现高效模型迁移与高精度计数。**

- **链接: [https://arxiv.org/pdf/2601.02203v1](https://arxiv.org/pdf/2601.02203v1)**

> **作者:** Oliver Custance; Saad Khan; Simon Parkinson; Quan Z. Sheng
>
> **摘要:** Device-free crowd-counting using WiFi Channel State Information (CSI) is a key enabling technology for a new generation of privacy-preserving Internet of Things (IoT) applications. However, practical deployment is severely hampered by the domain shift problem, where models trained in one environment fail to generalise to another. To overcome this, we propose a novel two-stage framework centred on a CSI-ResNet-A architecture. This model is pre-trained via self-supervised contrastive learning to learn domain-invariant representations and leverages lightweight Adapter modules for highly efficient fine-tuning. The resulting event sequence is then processed by a stateful counting machine to produce a final, stable occupancy estimate. We validate our framework extensively. On our WiFlow dataset, our unsupervised approach excels in a 10-shot learning scenario, achieving a final Mean Absolute Error (MAE) of just 0.44--a task where supervised baselines fail. To formally quantify robustness, we introduce the Generalisation Index (GI), on which our model scores near-perfectly, confirming its ability to generalise. Furthermore, our framework sets a new state-of-the-art public WiAR benchmark with 98.8\% accuracy. Our ablation studies reveal the core strength of our design: adapter-based fine-tuning achieves performance within 1\% of a full fine-tune (98.84\% vs. 99.67\%) while training 97.2\% fewer parameters. Our work provides a practical and scalable solution for developing robust sensing systems ready for real-world IoT deployments.
>
---
#### [new 017] VideoCuRL: Video Curriculum Reinforcement Learning with Orthogonal Difficulty Decomposition
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，解决RL中难度度量不准确的问题。提出VideoCuRL框架，分解难度为视觉和认知两轴，提升视频推理与感知性能。**

- **链接: [https://arxiv.org/pdf/2601.00887v1](https://arxiv.org/pdf/2601.00887v1)**

> **作者:** Hongbo Jin; Kuanwei Lin; Wenhao Zhang; Yichen Jin; Ge Li
>
> **摘要:** Reinforcement Learning (RL) is crucial for empowering VideoLLMs with complex spatiotemporal reasoning. However, current RL paradigms predominantly rely on random data shuffling or naive curriculum strategies based on scalar difficulty metrics. We argue that scalar metrics fail to disentangle two orthogonal challenges in video understanding: Visual Temporal Perception Load and Cognitive Reasoning Depth. To address this, we propose VideoCuRL, a novel framework that decomposes difficulty into these two axes. We employ efficient, training-free proxies, optical flow and keyframe entropy for visual complexity, Calibrated Surprisal for cognitive complexity, to map data onto a 2D curriculum grid. A competence aware Diagonal Wavefront strategy then schedules training from base alignment to complex reasoning. Furthermore, we introduce Dynamic Sparse KL and Structured Revisiting to stabilize training against reward collapse and catastrophic forgetting. Extensive experiments show that VideoCuRL surpasses strong RL baselines on reasoning (+2.5 on VSI-Bench) and perception (+2.9 on VideoMME) tasks. Notably, VideoCuRL eliminates the prohibitive inference overhead of generation-based curricula, offering a scalable solution for robust video post-training.
>
---
#### [new 018] VINO: A Unified Visual Generator with Interleaved OmniModal Context
- **分类: cs.CV**

- **简介: 该论文提出VINO，一个统一的视觉生成模型，解决图像和视频生成与编辑任务。通过共享扩散架构，融合多模态输入，实现高效、连贯的视觉创作。**

- **链接: [https://arxiv.org/pdf/2601.02358v1](https://arxiv.org/pdf/2601.02358v1)**

> **作者:** Junyi Chen; Tong He; Zhoujie Fu; Pengfei Wan; Kun Gai; Weicai Ye
>
> **备注:** Project page: https://sotamak1r.github.io/VINO-web/
>
> **摘要:** We present VINO, a unified visual generator that performs image and video generation and editing within a single framework. Instead of relying on task-specific models or independent modules for each modality, VINO uses a shared diffusion backbone that conditions on text, images and videos, enabling a broad range of visual creation and editing tasks under one model. Specifically, VINO couples a vision-language model (VLM) with a Multimodal Diffusion Transformer (MMDiT), where multimodal inputs are encoded as interleaved conditioning tokens, and then used to guide the diffusion process. This design supports multi-reference grounding, long-form instruction following, and coherent identity preservation across static and dynamic content, while avoiding modality-specific architectural components. To train such a unified system, we introduce a multi-stage training pipeline that progressively expands a video generation base model into a unified, multi-task generator capable of both image and video input and output. Across diverse generation and editing benchmarks, VINO demonstrates strong visual quality, faithful instruction following, improved reference and attribute preservation, and more controllable multi-identity edits. Our results highlight a practical path toward scalable unified visual generation, and the promise of interleaved, in-context computation as a foundation for general-purpose visual creation.
>
---
#### [new 019] BARE: Towards Bias-Aware and Reasoning-Enhanced One-Tower Visual Grounding
- **分类: cs.CV**

- **简介: 该论文属于视觉定位任务，解决多模态表示过拟合和语义推理不足的问题，提出BARE框架增强推理能力和减少模态偏差。**

- **链接: [https://arxiv.org/pdf/2601.01526v1](https://arxiv.org/pdf/2601.01526v1)**

> **作者:** Hongbing Li; Linhui Xiao; Zihan Zhao; Qi Shen; Yixiang Huang; Bo Xiao; Zhanyu Ma
>
> **摘要:** Visual Grounding (VG), which aims to locate a specific region referred to by expressions, is a fundamental yet challenging task in the multimodal understanding fields. While recent grounding transfer works have advanced the field through one-tower architectures, they still suffer from two primary limitations: (1) over-entangled multimodal representations that exacerbate deceptive modality biases, and (2) insufficient semantic reasoning that hinders the comprehension of referential cues. In this paper, we propose BARE, a bias-aware and reasoning-enhanced framework for one-tower visual grounding. BARE introduces a mechanism that preserves modality-specific features and constructs referential semantics through three novel modules: (i) language salience modulator, (ii) visual bias correction and (iii) referential relationship enhancement, which jointly mitigate multimodal distractions and enhance referential comprehension. Extensive experimental results on five benchmarks demonstrate that BARE not only achieves state-of-the-art performance but also delivers superior computational efficiency compared to existing approaches. The code is publicly accessible at https://github.com/Marloweeee/BARE.
>
---
#### [new 020] FAR-AMTN: Attention Multi-Task Network for Face Attribute Recognition
- **分类: cs.CV**

- **简介: 该论文属于人脸属性识别任务，旨在解决多任务网络泛化能力不足的问题。通过引入注意力机制和特征融合策略，提升模型性能并减少参数量。**

- **链接: [https://arxiv.org/pdf/2601.01537v1](https://arxiv.org/pdf/2601.01537v1)**

> **作者:** Gong Gao; Zekai Wang; Xianhui Liu; Weidong Zhao
>
> **备注:** 28 pages, 8figures
>
> **摘要:** To enhance the generalization performance of Multi-Task Networks (MTN) in Face Attribute Recognition (FAR), it is crucial to share relevant information across multiple related prediction tasks effectively. Traditional MTN methods create shared low-level modules and distinct high-level modules, causing an exponential increase in model parameters with the addition of tasks. This approach also limits feature interaction at the high level, hindering the exploration of semantic relations among attributes, thereby affecting generalization negatively. In response, this study introduces FAR-AMTN, a novel Attention Multi-Task Network for FAR. It incorporates a Weight-Shared Group-Specific Attention (WSGSA) module with shared parameters to minimize complexity while improving group feature representation. Furthermore, a Cross-Group Feature Fusion (CGFF) module is utilized to foster interactions between attribute groups, enhancing feature learning. A Dynamic Weighting Strategy (DWS) is also introduced for synchronized task convergence. Experiments on the CelebA and LFWA datasets demonstrate that the proposed FAR-AMTN demonstrates superior accuracy with significantly fewer parameters compared to existing models.
>
---
#### [new 021] Towards Any-Quality Image Segmentation via Generative and Adaptive Latent Space Enhancement
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务，旨在解决低质量图像分割性能下降的问题。通过引入生成潜空间增强和退化感知机制，提升模型在不同质量图像上的鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.02018v1](https://arxiv.org/pdf/2601.02018v1)**

> **作者:** Guangqian Guo; Aixi Ren; Yong Guo; Xuehui Yu; Jiacheng Tian; Wenli Li; Yaoxing Wang; Shan Gao
>
> **备注:** Diffusion-based latent space enhancement helps improve the robustness of SAM
>
> **摘要:** Segment Anything Models (SAMs), known for their exceptional zero-shot segmentation performance, have garnered significant attention in the research community. Nevertheless, their performance drops significantly on severely degraded, low-quality images, limiting their effectiveness in real-world scenarios. To address this, we propose GleSAM++, which utilizes Generative Latent space Enhancement to boost robustness on low-quality images, thus enabling generalization across various image qualities. Additionally, to improve compatibility between the pre-trained diffusion model and the segmentation framework, we introduce two techniques, i.e., Feature Distribution Alignment (FDA) and Channel Replication and Expansion (CRE). However, the above components lack explicit guidance regarding the degree of degradation. The model is forced to implicitly fit a complex noise distribution that spans conditions from mild noise to severe artifacts, which substantially increases the learning burden and leads to suboptimal reconstructions. To address this issue, we further introduce a Degradation-aware Adaptive Enhancement (DAE) mechanism. The key principle of DAE is to decouple the reconstruction process for arbitrary-quality features into two stages: degradation-level prediction and degradation-aware reconstruction. Our method can be applied to pre-trained SAM and SAM2 with only minimal additional learnable parameters, allowing for efficient optimization. Extensive experiments demonstrate that GleSAM++ significantly improves segmentation robustness on complex degradations while maintaining generalization to clear images. Furthermore, GleSAM++ also performs well on unseen degradations, underscoring the versatility of our approach and dataset.
>
---
#### [new 022] VReID-XFD: Video-based Person Re-identification at Extreme Far Distance Challenge Results
- **分类: cs.CV**

- **简介: 该论文属于视频行人重识别任务，解决极端远距离下跨视角的行人再识别问题。构建了VReID-XFD数据集，并分析了性能退化规律与挑战。**

- **链接: [https://arxiv.org/pdf/2601.01312v1](https://arxiv.org/pdf/2601.01312v1)**

> **作者:** Kailash A. Hambarde; Hugo Proença; Md Rashidunnabi; Pranita Samale; Qiwei Yang; Pingping Zhang; Zijing Gong; Yuhao Wang; Xi Zhang; Ruoshui Qu; Qiaoyun He; Yuhang Zhang; Thi Ngoc Ha Nguyen; Tien-Dung Mai; Cheng-Jun Kang; Yu-Fan Lin; Jin-Hui Jiang; Chih-Chung Hsu; Tamás Endrei; György Cserey; Ashwat Rajbhandari
>
> **摘要:** Person re-identification (ReID) across aerial and ground views at extreme far distances introduces a distinct operating regime where severe resolution degradation, extreme viewpoint changes, unstable motion cues, and clothing variation jointly undermine the appearance-based assumptions of existing ReID systems. To study this regime, we introduce VReID-XFD, a video-based benchmark and community challenge for extreme far-distance (XFD) aerial-to-ground person re-identification. VReID-XFD is derived from the DetReIDX dataset and comprises 371 identities, 11,288 tracklets, and 11.75 million frames, captured across altitudes from 5.8 m to 120 m, viewing angles from oblique (30 degrees) to nadir (90 degrees), and horizontal distances up to 120 m. The benchmark supports aerial-to-aerial, aerial-to-ground, and ground-to-aerial evaluation under strict identity-disjoint splits, with rich physical metadata. The VReID-XFD-25 Challenge attracted 10 teams with hundreds of submissions. Systematic analysis reveals monotonic performance degradation with altitude and distance, a universal disadvantage of nadir views, and a trade-off between peak performance and robustness. Even the best-performing SAS-PReID method achieves only 43.93 percent mAP in the aerial-to-ground setting. The dataset, annotations, and official evaluation protocols are publicly available at https://www.it.ubi.pt/DetReIDX/ .
>
---
#### [new 023] VerLM: Explaining Face Verification Using Natural Language
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于人脸验证任务，旨在解决系统缺乏透明度的问题。提出一种视觉-语言模型，既能准确验证人脸，又能提供解释。**

- **链接: [https://arxiv.org/pdf/2601.01798v1](https://arxiv.org/pdf/2601.01798v1)**

> **作者:** Syed Abdul Hannan; Hazim Bukhari; Thomas Cantalapiedra; Eman Ansar; Massa Baali; Rita Singh; Bhiksha Raj
>
> **摘要:** Face verification systems have seen substantial advancements; however, they often lack transparency in their decision-making processes. In this paper, we introduce an innovative Vision-Language Model (VLM) for Face Verification, which not only accurately determines if two face images depict the same individual but also explicitly explains the rationale behind its decisions. Our model is uniquely trained using two complementary explanation styles: (1) concise explanations that summarize the key factors influencing its decision, and (2) comprehensive explanations detailing the specific differences observed between the images. We adapt and enhance a state-of-the-art modeling approach originally designed for audio-based differentiation to suit visual inputs effectively. This cross-modal transfer significantly improves our model's accuracy and interpretability. The proposed VLM integrates sophisticated feature extraction techniques with advanced reasoning capabilities, enabling clear articulation of its verification process. Our approach demonstrates superior performance, surpassing baseline methods and existing models. These findings highlight the immense potential of vision language models in face verification set up, contributing to more transparent, reliable, and explainable face verification systems.
>
---
#### [new 024] Real-Time LiDAR Point Cloud Densification for Low-Latency Spatial Data Transmission
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于三维重建任务，旨在解决LiDAR点云稀疏和实时处理问题。通过融合多传感器数据，提出一种快速点云增强方法，实现高分辨率、低延迟的深度图生成。**

- **链接: [https://arxiv.org/pdf/2601.01210v1](https://arxiv.org/pdf/2601.01210v1)**

> **作者:** Kazuhiko Murasaki; Shunsuke Konagai; Masakatsu Aoki; Taiga Yoshida; Ryuichi Tanida
>
> **摘要:** To realize low-latency spatial transmission system for immersive telepresence, there are two major problems: capturing dynamic 3D scene densely and processing them in real time. LiDAR sensors capture 3D in real time, but produce sparce point clouds. Therefore, this paper presents a high-speed LiDAR point cloud densification method to generate dense 3D scene with minimal latency, addressing the need for on-the-fly depth completion while maintaining real-time performance. Our approach combines multiple LiDAR inputs with high-resolution color images and applies a joint bilateral filtering strategy implemented through a convolutional neural network architecture. Experiments demonstrate that the proposed method produces dense depth maps at full HD resolution in real time (30 fps), which is over 15x faster than a recent training-based depth completion approach. The resulting dense point clouds exhibit accurate geometry without multiview inconsistencies or ghosting artifacts.
>
---
#### [new 025] RFAssigner: A Generic Label Assignment Strategy for Dense Object Detection
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，解决小物体正样本不足导致的尺度不平衡问题。提出RFAssigner策略，通过GRF距离自适应选择正样本，提升多尺度检测性能。**

- **链接: [https://arxiv.org/pdf/2601.01240v1](https://arxiv.org/pdf/2601.01240v1)**

> **作者:** Ziqian Guan; Xieyi Fu; Yuting Wang; Haowen Xiao; Jiarui Zhu; Yingying Zhu; Yongtao Liu; Lin Gu
>
> **摘要:** Label assignment is a critical component in training dense object detectors. State-of-the-art methods typically assign each training sample a positive and a negative weight, optimizing the assignment scheme during training. However, these strategies often assign an insufficient number of positive samples to small objects, leading to a scale imbalance during training. To address this limitation, we introduce RFAssigner, a novel assignment strategy designed to enhance the multi-scale learning capabilities of dense detectors. RFAssigner first establishes an initial set of positive samples using a point-based prior. It then leverages a Gaussian Receptive Field (GRF) distance to measure the similarity between the GRFs of unassigned candidate locations and the ground-truth objects. Based on this metric, RFAssigner adaptively selects supplementary positive samples from the unassigned pool, promoting a more balanced learning process across object scales. Comprehensive experiments on three datasets with distinct object scale distributions validate the effectiveness and generalizability of our method. Notably, a single FCOS-ResNet-50 detector equipped with RFAssigner achieves state-of-the-art performance across all object scales, consistently outperforming existing strategies without requiring auxiliary modules or heuristics.
>
---
#### [new 026] InpaintHuman: Reconstructing Occluded Humans with Multi-Scale UV Mapping and Identity-Preserving Diffusion Inpainting
- **分类: cs.CV**

- **简介: 该论文属于3D人体重建任务，旨在解决单目视频中严重遮挡下生成完整、可动画化的人体模型的问题。提出InpaintHuman方法，结合多尺度UV映射和身份保持的扩散修复模块，提升重建质量与一致性。**

- **链接: [https://arxiv.org/pdf/2601.02098v1](https://arxiv.org/pdf/2601.02098v1)**

> **作者:** Jinlong Fan; Shanshan Zhao; Liang Zheng; Jing Zhang; Yuxiang Yang; Mingming Gong
>
> **摘要:** Reconstructing complete and animatable 3D human avatars from monocular videos remains challenging, particularly under severe occlusions. While 3D Gaussian Splatting has enabled photorealistic human rendering, existing methods struggle with incomplete observations, often producing corrupted geometry and temporal inconsistencies. We present InpaintHuman, a novel method for generating high-fidelity, complete, and animatable avatars from occluded monocular videos. Our approach introduces two key innovations: (i) a multi-scale UV-parameterized representation with hierarchical coarse-to-fine feature interpolation, enabling robust reconstruction of occluded regions while preserving geometric details; and (ii) an identity-preserving diffusion inpainting module that integrates textual inversion with semantic-conditioned guidance for subject-specific, temporally coherent completion. Unlike SDS-based methods, our approach employs direct pixel-level supervision to ensure identity fidelity. Experiments on synthetic benchmarks (PeopleSnapshot, ZJU-MoCap) and real-world scenarios (OcMotion) demonstrate competitive performance with consistent improvements in reconstruction quality across diverse poses and viewpoints.
>
---
#### [new 027] ExposeAnyone: Personalized Audio-to-Expression Diffusion Models Are Robust Zero-Shot Face Forgery Detectors
- **分类: cs.CV**

- **简介: 该论文属于人脸伪造检测任务，旨在解决未知伪造方法检测困难的问题。通过自监督扩散模型生成表情序列，利用重建误差进行身份距离计算，实现鲁棒的伪造检测。**

- **链接: [https://arxiv.org/pdf/2601.02359v1](https://arxiv.org/pdf/2601.02359v1)**

> **作者:** Kaede Shiohara; Toshihiko Yamasaki; Vladislav Golyanik
>
> **备注:** 17 pages, 8 figures, 11 tables; project page: https://mapooon.github.io/ExposeAnyonePage/
>
> **摘要:** Detecting unknown deepfake manipulations remains one of the most challenging problems in face forgery detection. Current state-of-the-art approaches fail to generalize to unseen manipulations, as they primarily rely on supervised training with existing deepfakes or pseudo-fakes, which leads to overfitting to specific forgery patterns. In contrast, self-supervised methods offer greater potential for generalization, but existing work struggles to learn discriminative representations only from self-supervision. In this paper, we propose ExposeAnyone, a fully self-supervised approach based on a diffusion model that generates expression sequences from audio. The key idea is, once the model is personalized to specific subjects using reference sets, it can compute the identity distances between suspected videos and personalized subjects via diffusion reconstruction errors, enabling person-of-interest face forgery detection. Extensive experiments demonstrate that 1) our method outperforms the previous state-of-the-art method by 4.22 percentage points in the average AUC on DF-TIMIT, DFDCP, KoDF, and IDForge datasets, 2) our model is also capable of detecting Sora2-generated videos, where the previous approaches perform poorly, and 3) our method is highly robust to corruptions such as blur and compression, highlighting the applicability in real-world face forgery detection.
>
---
#### [new 028] DeepInv: A Novel Self-supervised Learning Approach for Fast and Accurate Diffusion Inversion
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于扩散模型中的图像去噪任务，旨在解决缺乏监督信号导致的逆向难题。提出DeepInv方法，通过自监督学习和数据增强生成伪噪声，实现快速准确的图像到噪声映射。**

- **链接: [https://arxiv.org/pdf/2601.01487v1](https://arxiv.org/pdf/2601.01487v1)**

> **作者:** Ziyue Zhang; Luxi Lin; Xiaolin Hu; Chao Chang; HuaiXi Wang; Yiyi Zhou; Rongrong Ji
>
> **摘要:** Diffusion inversion is a task of recovering the noise of an image in a diffusion model, which is vital for controllable diffusion image editing. At present, diffusion inversion still remains a challenging task due to the lack of viable supervision signals. Thus, most existing methods resort to approximation-based solutions, which however are often at the cost of performance or efficiency. To remedy these shortcomings, we propose a novel self-supervised diffusion inversion approach in this paper, termed Deep Inversion (DeepInv). Instead of requiring ground-truth noise annotations, we introduce a self-supervised objective as well as a data augmentation strategy to generate high-quality pseudo noises from real images without manual intervention. Based on these two innovative designs, DeepInv is also equipped with an iterative and multi-scale training regime to train a parameterized inversion solver, thereby achieving the fast and accurate image-to-noise mapping. To the best of our knowledge, this is the first attempt of presenting a trainable solver to predict inversion noise step by step. The extensive experiments show that our DeepInv can achieve much better performance and inference speed than the compared methods, e.g., +40.435% SSIM than EasyInv and +9887.5% speed than ReNoise on COCO dataset. Moreover, our careful designs of trainable solvers can also provide insights to the community. Codes and model parameters will be released in https://github.com/potato-kitty/DeepInv.
>
---
#### [new 029] 600k-ks-ocr: a large-scale synthetic dataset for optical character recognition in kashmiri script
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出600K-KS-OCR数据集，用于解决克什米尔语OCR任务中的资源不足问题。通过合成数据和增强技术，提供高质量训练素材。**

- **链接: [https://arxiv.org/pdf/2601.01088v1](https://arxiv.org/pdf/2601.01088v1)**

> **作者:** Haq Nawaz Malik
>
> **摘要:** This technical report presents the 600K-KS-OCR Dataset, a large-scale synthetic corpus comprising approximately 602,000 word-level segmented images designed for training and evaluating optical character recognition systems targeting Kashmiri script. The dataset addresses a critical resource gap for Kashmiri, an endangered Dardic language utilizing a modified Perso-Arabic writing system spoken by approximately seven million people. Each image is rendered at 256x64 pixels with corresponding ground-truth transcriptions provided in multiple formats compatible with CRNN, TrOCR, and generalpurpose machine learning pipelines. The generation methodology incorporates three traditional Kashmiri typefaces, comprehensive data augmentation simulating real-world document degradation, and diverse background textures to enhance model robustness. The dataset is distributed across ten partitioned archives totaling approximately 10.6 GB and is released under the CC-BY-4.0 license to facilitate research in low-resource language optical character recognition.
>
---
#### [new 030] VIT-Ped: Visionary Intention Transformer for Pedestrian Behavior Analysis
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于行人行为分析任务，旨在预测行人意图以提升自动驾驶安全性。提出基于Transformer的算法，融合多模态数据，在JAAD数据集上取得优异效果。**

- **链接: [https://arxiv.org/pdf/2601.01989v1](https://arxiv.org/pdf/2601.01989v1)**

> **作者:** Aly R. Elkammar; Karim M. Gamaleldin; Catherine M. Elias
>
> **摘要:** Pedestrian Intention prediction is one of the key technologies in the transition from level 3 to level 4 autonomous driving. To understand pedestrian crossing behaviour, several elements and features should be taken into consideration to make the roads of tomorrow safer for everybody. We introduce a transformer / video vision transformer based algorithm of different sizes which uses different data modalities .We evaluated our algorithms on popular pedestrian behaviour dataset, JAAD, and have reached SOTA performance and passed the SOTA in metrics like Accuracy, AUC and F1-score. The advantages brought by different model design choices are investigated via extensive ablation studies.
>
---
#### [new 031] Four-Stage Alzheimer's Disease Classification from MRI Using Topological Feature Extraction, Feature Selection, and Ensemble Learning
- **分类: cs.CV**

- **简介: 该论文属于阿尔茨海默病分级任务，旨在准确高效分类MRI图像中的疾病严重程度。通过拓扑特征提取和集成学习，提出TDA-Alz方法，解决数据有限与模型可解释性问题。**

- **链接: [https://arxiv.org/pdf/2601.00918v1](https://arxiv.org/pdf/2601.00918v1)**

> **作者:** Faisal Ahmed
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** Accurate and efficient classification of Alzheimer's disease (AD) severity from brain magnetic resonance imaging (MRI) remains a critical challenge, particularly when limited data and model interpretability are of concern. In this work, we propose TDA-Alz, a novel framework for four-stage Alzheimer's disease severity classification (non-demented, moderate dementia, mild, and very mild) using topological data analysis (TDA) and ensemble learning. Instead of relying on deep convolutional architectures or extensive data augmentation, our approach extracts topological descriptors that capture intrinsic structural patterns of brain MRI, followed by feature selection to retain the most discriminative topological features. These features are then classified using an ensemble learning strategy to achieve robust multiclass discrimination. Experiments conducted on the OASIS-1 MRI dataset demonstrate that the proposed method achieves an accuracy of 98.19% and an AUC of 99.75%, outperforming or matching state-of-the-art deep learning--based methods reported on OASIS and OASIS-derived datasets. Notably, the proposed framework does not require data augmentation, pretrained networks, or large-scale computational resources, making it computationally efficient and fast compared to deep neural network approaches. Furthermore, the use of topological descriptors provides greater interpretability, as the extracted features are directly linked to the underlying structural characteristics of brain MRI rather than opaque latent representations. These results indicate that TDA-Alz offers a powerful, lightweight, and interpretable alternative to deep learning models for MRI-based Alzheimer's disease severity classification, with strong potential for real-world clinical decision-support systems.
>
---
#### [new 032] MagicFight: Personalized Martial Arts Combat Video Generation
- **分类: cs.CV**

- **简介: 该论文提出MagicFight任务，解决两人武术对战视频生成问题。针对现有模型在双人交互中的身份混淆和动作不匹配问题，构建专用数据集并优化模型，生成高质量、连贯的对战视频。**

- **链接: [https://arxiv.org/pdf/2601.02107v1](https://arxiv.org/pdf/2601.02107v1)**

> **作者:** Jiancheng Huang; Mingfu Yan; Songyan Chen; Yi Huang; Shifeng Chen
>
> **备注:** Accepted by ACM MM 2024
>
> **摘要:** Amid the surge in generic text-to-video generation, the field of personalized human video generation has witnessed notable advancements, primarily concentrated on single-person scenarios. However, to our knowledge, the domain of two-person interactions, particularly in the context of martial arts combat, remains uncharted. We identify a significant gap: existing models for single-person dancing generation prove insufficient for capturing the subtleties and complexities of two engaged fighters, resulting in challenges such as identity confusion, anomalous limbs, and action mismatches. To address this, we introduce a pioneering new task, Personalized Martial Arts Combat Video Generation. Our approach, MagicFight, is specifically crafted to overcome these hurdles. Given this pioneering task, we face a lack of appropriate datasets. Thus, we generate a bespoke dataset using the game physics engine Unity, meticulously crafting a multitude of 3D characters, martial arts moves, and scenes designed to represent the diversity of combat. MagicFight refines and adapts existing models and strategies to generate high-fidelity two-person combat videos that maintain individual identities and ensure seamless, coherent action sequences, thereby laying the groundwork for future innovations in the realm of interactive video content creation. Website: https://MingfuYAN.github.io/MagicFight/ Dataset: https://huggingface.co/datasets/MingfuYAN/KungFu-Fiesta
>
---
#### [new 033] DreamID-V:Bridging the Image-to-Video Gap for High-Fidelity Face Swapping via Diffusion Transformer
- **分类: cs.CV**

- **简介: 该论文属于视频人脸交换任务，旨在解决身份相似性与属性保持的问题。提出DreamID-V框架，结合扩散Transformer和强化学习，提升视频换脸的逼真度与一致性。**

- **链接: [https://arxiv.org/pdf/2601.01425v1](https://arxiv.org/pdf/2601.01425v1)**

> **作者:** Xu Guo; Fulong Ye; Xinghui Li; Pengqi Tu; Pengze Zhang; Qichao Sun; Songtao Zhao; Xiangwang Hou; Qian He
>
> **备注:** Project: https://guoxu1233.github.io/DreamID-V/
>
> **摘要:** Video Face Swapping (VFS) requires seamlessly injecting a source identity into a target video while meticulously preserving the original pose, expression, lighting, background, and dynamic information. Existing methods struggle to maintain identity similarity and attribute preservation while preserving temporal consistency. To address the challenge, we propose a comprehensive framework to seamlessly transfer the superiority of Image Face Swapping (IFS) to the video domain. We first introduce a novel data pipeline SyncID-Pipe that pre-trains an Identity-Anchored Video Synthesizer and combines it with IFS models to construct bidirectional ID quadruplets for explicit supervision. Building upon paired data, we propose the first Diffusion Transformer-based framework DreamID-V, employing a core Modality-Aware Conditioning module to discriminatively inject multi-model conditions. Meanwhile, we propose a Synthetic-to-Real Curriculum mechanism and an Identity-Coherence Reinforcement Learning strategy to enhance visual realism and identity consistency under challenging scenarios. To address the issue of limited benchmarks, we introduce IDBench-V, a comprehensive benchmark encompassing diverse scenes. Extensive experiments demonstrate DreamID-V outperforms state-of-the-art methods and further exhibits exceptional versatility, which can be seamlessly adapted to various swap-related tasks.
>
---
#### [new 034] FALCON: Few-Shot Adversarial Learning for Cross-Domain Medical Image Segmentation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出FALCON框架，解决跨域医学图像分割任务中的标注数据少、计算成本高问题，通过2D切片处理和对抗微调实现高精度分割。**

- **链接: [https://arxiv.org/pdf/2601.01687v1](https://arxiv.org/pdf/2601.01687v1)**

> **作者:** Abdur R. Fayjie; Pankhi Kashyap; Jutika Borah; Patrick Vandewalle
>
> **备注:** 20 pages, 6 figures, 7 tables
>
> **摘要:** Precise delineation of anatomical and pathological structures within 3D medical volumes is crucial for accurate diagnosis, effective surgical planning, and longitudinal disease monitoring. Despite advancements in AI, clinically viable segmentation is often hindered by the scarcity of 3D annotations, patient-specific variability, data privacy concerns, and substantial computational overhead. In this work, we propose FALCON, a cross-domain few-shot segmentation framework that achieves high-precision 3D volume segmentation by processing data as 2D slices. The framework is first meta-trained on natural images to learn-to-learn generalizable segmentation priors, then transferred to the medical domain via adversarial fine-tuning and boundary-aware learning. Task-aware inference, conditioned on support cues, allows FALCON to adapt dynamically to patient-specific anatomical variations across slices. Experiments on four benchmarks demonstrate that FALCON consistently achieves the lowest Hausdorff Distance scores, indicating superior boundary accuracy while maintaining a Dice Similarity Coefficient comparable to the state-of-the-art models. Notably, these results are achieved with significantly less labeled data, no data augmentation, and substantially lower computational overhead.
>
---
#### [new 035] EscherVerse: An Open World Benchmark and Dataset for Teleo-Spatial Intelligence with Physical-Dynamic and Intent-Driven Understanding
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出Teleo-Spatial Intelligence任务，解决物理动态与意图推理问题。构建EscherVerse基准和数据集，推动智能体对动态场景中物体变化和人类意图的理解。**

- **链接: [https://arxiv.org/pdf/2601.01547v1](https://arxiv.org/pdf/2601.01547v1)**

> **作者:** Tianjun Gu; Chenghua Gong; Jingyu Gong; Zhizhong Zhang; Yuan Xie; Lizhuang Ma; Xin Tan
>
> **摘要:** The ability to reason about spatial dynamics is a cornerstone of intelligence, yet current research overlooks the human intent behind spatial changes. To address these limitations, we introduce Teleo-Spatial Intelligence (TSI), a new paradigm that unifies two critical pillars: Physical-Dynamic Reasoning--understanding the physical principles of object interactions--and Intent-Driven Reasoning--inferring the human goals behind these actions. To catalyze research in TSI, we present EscherVerse, consisting of a large-scale, open-world benchmark (Escher-Bench), a dataset (Escher-35k), and models (Escher series). Derived from real-world videos, EscherVerse moves beyond constrained settings to explicitly evaluate an agent's ability to reason about object permanence, state transitions, and trajectory prediction in dynamic, human-centric scenarios. Crucially, it is the first benchmark to systematically assess Intent-Driven Reasoning, challenging models to connect physical events to their underlying human purposes. Our work, including a novel data curation pipeline, provides a foundational resource to advance spatial intelligence from passive scene description toward a holistic, purpose-driven understanding of the world.
>
---
#### [new 036] AR-MOT: Autoregressive Multi-object Tracking
- **分类: cs.CV**

- **简介: 该论文属于多目标跟踪任务，旨在解决传统方法灵活性差、难以扩展的问题。提出AR-MOT，将跟踪任务转化为序列生成，提升模型适应性与扩展性。**

- **链接: [https://arxiv.org/pdf/2601.01925v1](https://arxiv.org/pdf/2601.01925v1)**

> **作者:** Lianjie Jia; Yuhan Wu; Binghao Ran; Yifan Wang; Lijun Wang; Huchuan Lu
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** As multi-object tracking (MOT) tasks continue to evolve toward more general and multi-modal scenarios, the rigid and task-specific architectures of existing MOT methods increasingly hinder their applicability across diverse tasks and limit flexibility in adapting to new tracking formulations. Most approaches rely on fixed output heads and bespoke tracking pipelines, making them difficult to extend to more complex or instruction-driven tasks. To address these limitations, we propose AR-MOT, a novel autoregressive paradigm that formulates MOT as a sequence generation task within a large language model (LLM) framework. This design enables the model to output structured results through flexible sequence construction, without requiring any task-specific heads. To enhance region-level visual perception, we introduce an Object Tokenizer based on a pretrained detector. To mitigate the misalignment between global and regional features, we propose a Region-Aware Alignment (RAA) module, and to support long-term tracking, we design a Temporal Memory Fusion (TMF) module that caches historical object tokens. AR-MOT offers strong potential for extensibility, as new modalities or instructions can be integrated by simply modifying the output sequence format without altering the model architecture. Extensive experiments on MOT17 and DanceTrack validate the feasibility of our approach, achieving performance comparable to state-of-the-art methods while laying the foundation for more general and flexible MOT systems.
>
---
#### [new 037] Entity-Guided Multi-Task Learning for Infrared and Visible Image Fusion
- **分类: cs.CV**

- **简介: 该论文属于红外与可见光图像融合任务，旨在解决文本语义噪声和语义深度不足的问题。提出EGMT方法，通过实体引导的多任务学习提升融合效果。**

- **链接: [https://arxiv.org/pdf/2601.01870v1](https://arxiv.org/pdf/2601.01870v1)**

> **作者:** Wenyu Shao; Hongbo Liu; Yunchuan Ma; Ruili Wang
>
> **备注:** Accepted by IEEE Transactions on Multimedia
>
> **摘要:** Existing text-driven infrared and visible image fusion approaches often rely on textual information at the sentence level, which can lead to semantic noise from redundant text and fail to fully exploit the deeper semantic value of textual information. To address these issues, we propose a novel fusion approach named Entity-Guided Multi-Task learning for infrared and visible image fusion (EGMT). Our approach includes three key innovative components: (i) A principled method is proposed to extract entity-level textual information from image captions generated by large vision-language models, eliminating semantic noise from raw text while preserving critical semantic information; (ii) A parallel multi-task learning architecture is constructed, which integrates image fusion with a multi-label classification task. By using entities as pseudo-labels, the multi-label classification task provides semantic supervision, enabling the model to achieve a deeper understanding of image content and significantly improving the quality and semantic density of the fused image; (iii) An entity-guided cross-modal interactive module is also developed to facilitate the fine-grained interaction between visual and entity-level textual features, which enhances feature representation by capturing cross-modal dependencies at both inter-visual and visual-entity levels. To promote the wide application of the entity-guided image fusion framework, we release the entity-annotated version of four public datasets (i.e., TNO, RoadScene, M3FD, and MSRS). Extensive experiments demonstrate that EGMT achieves superior performance in preserving salient targets, texture details, and semantic consistency, compared to the state-of-the-art methods. The code and dataset will be publicly available at https://github.com/wyshao-01/EGMT.
>
---
#### [new 038] Application of deep learning techniques in non-contrast computed tomography pulmonary angiogram for pulmonary embolism diagnosis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于肺栓塞诊断任务，旨在解决传统方法依赖对比剂的问题。通过3D卷积神经网络，在无对比剂CT图像中实现肺栓塞自动分类。**

- **链接: [https://arxiv.org/pdf/2601.00925v1](https://arxiv.org/pdf/2601.00925v1)**

> **作者:** I-Hsien Ting; Yi-Jun Tseng; Yu-Sheng Lin
>
> **摘要:** Pulmonary embolism is a life-threatening disease, early detection and treatment can significantly reduce mortality. In recent years, many studies have been using deep learning in the diagnosis of pulmonary embolism with contrast medium computed tomography pulmonary angiography, but the contrast medium is likely to cause acute kidney injury in patients with pulmonary embolism and chronic kidney disease, and the contrast medium takes time to work, patients with acute pulmonary embolism may miss the golden treatment time. This study aims to use deep learning techniques to automatically classify pulmonary embolism in CT images without contrast medium by using a 3D convolutional neural network model. The deep learning model used in this study had a significant impact on the pulmonary embolism classification of computed tomography images without contrast with 85\% accuracy and 0.84 AUC, which confirms the feasibility of the model in the diagnosis of pulmonary embolism.
>
---
#### [new 039] Lightweight Channel Attention for Efficient CNNs
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉领域，研究如何提升CNN的效率与精度平衡。针对通道注意力机制的效率与准确率问题，提出轻量级LCA模块，在ResNet 18和MobileNetV2上验证其有效性。**

- **链接: [https://arxiv.org/pdf/2601.01002v1](https://arxiv.org/pdf/2601.01002v1)**

> **作者:** Prem Babu Kanaparthi; Tulasi Venkata Sri Varshini Padamata
>
> **备注:** 6 pages, 5 figures
>
> **摘要:** Attention mechanisms have become integral to modern convolutional neural networks (CNNs), delivering notable performance improvements with minimal computational overhead. However, the efficiency accuracy trade off of different channel attention designs remains underexplored. This work presents an empirical study comparing Squeeze and Excitation (SE), Efficient Channel Attention (ECA), and a proposed Lite Channel Attention (LCA) module across ResNet 18 and MobileNetV2 architectures on CIFAR 10. LCA employs adaptive one dimensional convolutions with grouped operations to reduce parameter usage while preserving effective attention behavior. Experimental results show that LCA achieves competitive accuracy, reaching 94.68 percent on ResNet 18 and 93.10 percent on MobileNetV2, while matching ECA in parameter efficiency and maintaining favorable inference latency. Comprehensive benchmarks including FLOPs, parameter counts, and GPU latency measurements are provided, offering practical insights for deploying attention enhanced CNNs in resource constrained environments.
>
---
#### [new 040] Rank-based Geographical Regularization: Revisiting Contrastive Self-Supervised Learning for Multispectral Remote Sensing Imagery
- **分类: cs.CV**

- **简介: 该论文属于多光谱遥感图像的自监督学习任务，旨在解决地理信息融入特征空间的问题。提出GeoRank方法，通过优化球面距离提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.02289v1](https://arxiv.org/pdf/2601.02289v1)**

> **作者:** Tom Burgert; Leonard Hackel; Paolo Rota; Begüm Demir
>
> **备注:** accepted for publication at IEEE/CVF Winter Conference on Applications of Computer Vision
>
> **摘要:** Self-supervised learning (SSL) has become a powerful paradigm for learning from large, unlabeled datasets, particularly in computer vision (CV). However, applying SSL to multispectral remote sensing (RS) images presents unique challenges and opportunities due to the geographical and temporal variability of the data. In this paper, we introduce GeoRank, a novel regularization method for contrastive SSL that improves upon prior techniques by directly optimizing spherical distances to embed geographical relationships into the learned feature space. GeoRank outperforms or matches prior methods that integrate geographical metadata and consistently improves diverse contrastive SSL algorithms (e.g., BYOL, DINO). Beyond this, we present a systematic investigation of key adaptations of contrastive SSL for multispectral RS images, including the effectiveness of data augmentations, the impact of dataset cardinality and image size on performance, and the task dependency of temporal views. Code is available at https://github.com/tomburgert/georank.
>
---
#### [new 041] Unified Generation and Self-Verification for Vision-Language Models via Advantage Decoupled Preference Optimization
- **分类: cs.CV**

- **简介: 该论文提出ADPO框架，解决视觉语言模型生成与自验证的联合优化问题，通过创新奖励机制和解耦优化，提升验证效果并降低推理时间。**

- **链接: [https://arxiv.org/pdf/2601.01483v1](https://arxiv.org/pdf/2601.01483v1)**

> **作者:** Xinyu Qiu; Heng Jia; Zhengwen Zeng; Shuheng Shen; Changhua Meng; Yi Yang; Linchao Zhu
>
> **摘要:** Parallel test-time scaling typically trains separate generation and verification models, incurring high training and inference costs. We propose Advantage Decoupled Preference Optimization (ADPO), a unified reinforcement learning framework that jointly learns answer generation and self-verification within a single policy. ADPO introduces two innovations: a preference verification reward improving verification capability and a decoupled optimization mechanism enabling synergistic optimization of generation and verification. Specifically, the preference verification reward computes mean verification scores from positive and negative samples as decision thresholds, providing positive feedback when prediction correctness aligns with answer correctness. Meanwhile, the advantage decoupled optimization computes separate advantages for generation and verification, applies token masks to isolate gradients, and combines masked GRPO objectives, preserving generation quality while calibrating verification scores. ADPO achieves up to +34.1% higher verification AUC and -53.5% lower inference time, with significant gains of +2.8%/+1.4% accuracy on MathVista/MMMU, +1.9 cIoU on ReasonSeg, and +1.7%/+1.0% step success rate on AndroidControl/GUI Odyssey.
>
---
#### [new 042] Nighttime Hazy Image Enhancement via Progressively and Mutually Reinforcing Night-Haze Priors
- **分类: cs.CV**

- **简介: 该论文属于夜间雾霾图像增强任务，旨在解决夜间图像因雾霾和低光共同作用导致的可见度不足问题。通过融合雾霾与低光先验，提升图像质量。**

- **链接: [https://arxiv.org/pdf/2601.01998v1](https://arxiv.org/pdf/2601.01998v1)**

> **作者:** Chen Zhu; Huiwen Zhang; Mu He; Yujie Li; Xiaotian Qiao
>
> **摘要:** Enhancing the visibility of nighttime hazy images is challenging due to the complex degradation distributions. Existing methods mainly address a single type of degradation (e.g., haze or low-light) at a time, ignoring the interplay of different degradation types and resulting in limited visibility improvement. We observe that the domain knowledge shared between low-light and haze priors can be reinforced mutually for better visibility. Based on this key insight, in this paper, we propose a novel framework that enhances visibility in nighttime hazy images by reinforcing the intrinsic consistency between haze and low-light priors mutually and progressively. In particular, our model utilizes image-, patch-, and pixel-level experts that operate across visual and frequency domains to recover global scene structure, regional patterns, and fine-grained details progressively. A frequency-aware router is further introduced to adaptively guide the contribution of each expert, ensuring robust image restoration. Extensive experiments demonstrate the superior performance of our model on nighttime dehazing benchmarks both quantitatively and qualitatively. Moreover, we showcase the generalizability of our model in daytime dehazing and low-light enhancement tasks.
>
---
#### [new 043] Can Generative Models Actually Forge Realistic Identity Documents?
- **分类: cs.CV**

- **简介: 论文探讨生成模型是否能伪造真实身份文件，属于文档伪造检测任务。研究分析多种生成模型，发现其无法复制文件的结构和真实性，表明伪造风险可能被高估。**

- **链接: [https://arxiv.org/pdf/2601.00829v1](https://arxiv.org/pdf/2601.00829v1)**

> **作者:** Alexander Vinogradov
>
> **备注:** 11 pages, 16 figures
>
> **摘要:** Generative image models have recently shown significant progress in image realism, leading to public concerns about their potential misuse for document forgery. This paper explores whether contemporary open-source and publicly accessible diffusion-based generative models can produce identity document forgeries that could realistically bypass human or automated verification systems. We evaluate text-to-image and image-to-image generation pipelines using multiple publicly available generative model families, including Stable Diffusion, Qwen, Flux, Nano-Banana, and others. The findings indicate that while current generative models can simulate surface-level document aesthetics, they fail to reproduce structural and forensic authenticity. Consequently, the risk of generative identity document deepfakes achieving forensic-level authenticity may be overestimated, underscoring the value of collaboration between machine learning practitioners and document-forensics experts in realistic risk assessment.
>
---
#### [new 044] Deep Clustering with Associative Memories
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于深度聚类任务，旨在解决表示学习与聚类分离的问题。提出DCAM方法，通过关联记忆构建统一目标，提升聚类效果。**

- **链接: [https://arxiv.org/pdf/2601.00963v1](https://arxiv.org/pdf/2601.00963v1)**

> **作者:** Bishwajit Saha; Dmitry Krotov; Mohammed J. Zaki; Parikshit Ram
>
> **摘要:** Deep clustering - joint representation learning and latent space clustering - is a well studied problem especially in computer vision and text processing under the deep learning framework. While the representation learning is generally differentiable, clustering is an inherently discrete optimization task, requiring various approximations and regularizations to fit in a standard differentiable pipeline. This leads to a somewhat disjointed representation learning and clustering. In this work, we propose a novel loss function utilizing energy-based dynamics via Associative Memories to formulate a new deep clustering method, DCAM, which ties together the representation learning and clustering aspects more intricately in a single objective. Our experiments showcase the advantage of DCAM, producing improved clustering quality for various architecture choices (convolutional, residual or fully-connected) and data modalities (images or text).
>
---
#### [new 045] Clean-GS: Semantic Mask-Guided Pruning for 3D Gaussian Splatting
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于3D重建任务，解决3D高斯点云中冗余高斯分布的问题。通过语义掩码引导的剪枝方法，有效压缩模型并提升渲染质量。**

- **链接: [https://arxiv.org/pdf/2601.00913v1](https://arxiv.org/pdf/2601.00913v1)**

> **作者:** Subhankar Mishra
>
> **摘要:** 3D Gaussian Splatting produces high-quality scene reconstructions but generates hundreds of thousands of spurious Gaussians (floaters) scattered throughout the environment. These artifacts obscure objects of interest and inflate model sizes, hindering deployment in bandwidth-constrained applications. We present Clean-GS, a method for removing background clutter and floaters from 3DGS reconstructions using sparse semantic masks. Our approach combines whitelist-based spatial filtering with color-guided validation and outlier removal to achieve 60-80\% model compression while preserving object quality. Unlike existing 3DGS pruning methods that rely on global importance metrics, Clean-GS uses semantic information from as few as 3 segmentation masks (1\% of views) to identify and remove Gaussians not belonging to the target object. Our multi-stage approach consisting of (1) whitelist filtering via projection to masked regions, (2) depth-buffered color validation, and (3) neighbor-based outlier removal isolates monuments and objects from complex outdoor scenes. Experiments on Tanks and Temples show that Clean-GS reduces file sizes from 125MB to 47MB while maintaining rendering quality, making 3DGS models practical for web deployment and AR/VR applications. Our code is available at https://github.com/smlab-niser/clean-gs
>
---
#### [new 046] Enhancing Object Detection with Privileged Information: A Model-Agnostic Teacher-Student Approach
- **分类: cs.CV; cs.AI; cs.ET; cs.LG**

- **简介: 该论文属于目标检测任务，旨在利用训练阶段的特权信息提升检测性能。通过教师-学生架构注入边界框掩码等信息，提升模型精度且不增加推理负担。**

- **链接: [https://arxiv.org/pdf/2601.02016v1](https://arxiv.org/pdf/2601.02016v1)**

> **作者:** Matthias Bartolo; Dylan Seychell; Gabriel Hili; Matthew Montebello; Carl James Debono; Saviour Formosa; Konstantinos Makantasis
>
> **备注:** Code available on GitHub: https://github.com/mbar0075/lupi-for-object-detection
>
> **摘要:** This paper investigates the integration of the Learning Using Privileged Information (LUPI) paradigm in object detection to exploit fine-grained, descriptive information available during training but not at inference. We introduce a general, model-agnostic methodology for injecting privileged information-such as bounding box masks, saliency maps, and depth cues-into deep learning-based object detectors through a teacher-student architecture. Experiments are conducted across five state-of-the-art object detection models and multiple public benchmarks, including UAV-based litter detection datasets and Pascal VOC 2012, to assess the impact on accuracy, generalization, and computational efficiency. Our results demonstrate that LUPI-trained students consistently outperform their baseline counterparts, achieving significant boosts in detection accuracy with no increase in inference complexity or model size. Performance improvements are especially marked for medium and large objects, while ablation studies reveal that intermediate weighting of teacher guidance optimally balances learning from privileged and standard inputs. The findings affirm that the LUPI framework provides an effective and practical strategy for advancing object detection systems in both resource-constrained and real-world settings.
>
---
#### [new 047] CardioMOD-Net: A Modal Decomposition-Neural Network Framework for Diagnosis and Prognosis of HFpEF from Echocardiography Cine Loops
- **分类: cs.CV**

- **简介: 该论文提出CardioMOD-Net框架，用于从超声心动图电影中多类诊断和预测HFpEF的发病时间，解决早期诊断与预后评估难题。**

- **链接: [https://arxiv.org/pdf/2601.01176v1](https://arxiv.org/pdf/2601.01176v1)**

> **作者:** Andrés Bell-Navas; Jesús Garicano-Mena; Antonella Ausiello; Soledad Le Clainche; María Villalba-Orero; Enrique Lara-Pezzi
>
> **备注:** 9 pages; 1 figure; letter
>
> **摘要:** Introduction: Heart failure with preserved ejection fraction (HFpEF) arises from diverse comorbidities and progresses through prolonged subclinical stages, making early diagnosis and prognosis difficult. Current echocardiography-based Artificial Intelligence (AI) models focus primarily on binary HFpEF detection in humans and do not provide comorbidity-specific phenotyping or temporal estimates of disease progression towards decompensation. We aimed to develop a unified AI framework, CardioMOD-Net, to perform multiclass diagnosis and continuous prediction of HFpEF onset directly from standard echocardiography cine loops in preclinical models. Methods: Mouse echocardiography videos from four groups were used: control (CTL), hyperglycaemic (HG), obesity (OB), and systemic arterial hypertension (SAH). Two-dimensional parasternal long-axis cine loops were decomposed using Higher Order Dynamic Mode Decomposition (HODMD) to extract temporal features for downstream analysis. A shared latent representation supported Vision Transformers, one for a classifier for diagnosis and another for a regression module for predicting the age at HFpEF onset. Results: Overall diagnostic accuracy across the four groups was 65%, with all classes exceeding 50% accuracy. Misclassifications primarily reflected early-stage overlap between OB or SAH and CTL. The prognostic module achieved a root-mean-square error of 21.72 weeks for time-to-HFpEF prediction, with OB and SAH showing the most accurate estimates. Predicted HFpEF onset closely matched true distributions in all groups. Discussion: This unified framework demonstrates that multiclass phenotyping and continuous HFpEF onset prediction can be obtained from a single cine loop, even under small-data conditions. The approach offers a foundation for integrating diagnostic and prognostic modelling in preclinical HFpEF research.
>
---
#### [new 048] Evaluating transfer learning strategies for improving dairy cattle body weight prediction in small farms using depth-image and point-cloud data
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于农业计算机视觉任务，旨在提升小农场奶牛体重预测效果。通过对比深度图像与点云数据，评估迁移学习策略的有效性，解决小数据环境下模型泛化问题。**

- **链接: [https://arxiv.org/pdf/2601.01044v1](https://arxiv.org/pdf/2601.01044v1)**

> **作者:** Jin Wang; Angelo De Castro; Yuxi Zhang; Lucas Basolli Borsatto; Yuechen Guo; Victoria Bastos Primo; Ana Beatriz Montevecchio Bernardino; Gota Morota; Ricardo C Chebel; Haipeng Yu
>
> **摘要:** Computer vision provides automated, non-invasive, and scalable tools for monitoring dairy cattle, thereby supporting management, health assessment, and phenotypic data collection. Although transfer learning is commonly used for predicting body weight from images, its effectiveness and optimal fine-tuning strategies remain poorly understood in livestock applications, particularly beyond the use of pretrained ImageNet or COCO weights. In addition, while both depth images and three-dimensional point-cloud data have been explored for body weight prediction, direct comparisons of these two modalities in dairy cattle are limited. Therefore, the objectives of this study were to 1) evaluate whether transfer learning from a large farm enhances body weight prediction on a small farm with limited data, and 2) compare the predictive performance of depth-image- and point-cloud-based approaches under three experimental designs. Top-view depth images and point-cloud data were collected from 1,201, 215, and 58 cows at large, medium, and small dairy farms, respectively. Four deep learning models were evaluated: ConvNeXt and MobileViT for depth images, and PointNet and DGCNN for point clouds. Transfer learning markedly improved body weight prediction on the small farm across all four models, outperforming single-source learning and achieving gains comparable to or greater than joint learning. These results indicate that pretrained representations generalize well across farms with differing imaging conditions and dairy cattle populations. No consistent performance difference was observed between depth-image- and point-cloud-based models. Overall, these findings suggest that transfer learning is well suited for small farm prediction scenarios where cross-farm data sharing is limited by privacy, logistical, or policy constraints, as it requires access only to pretrained model weights rather than raw data.
>
---
#### [new 049] Luminark: Training-free, Probabilistically-Certified Watermarking for General Vision Generative Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Luminark，一种无需训练的视觉生成模型水印方法，解决水印嵌入与检测问题。通过亮度统计实现概率认证，确保检测可靠性。**

- **链接: [https://arxiv.org/pdf/2601.01085v1](https://arxiv.org/pdf/2601.01085v1)**

> **作者:** Jiayi Xu; Zhang Zhang; Yuanrui Zhang; Ruitao Chen; Yixian Xu; Tianyu He; Di He
>
> **摘要:** In this paper, we introduce \emph{Luminark}, a training-free and probabilistically-certified watermarking method for general vision generative models. Our approach is built upon a novel watermark definition that leverages patch-level luminance statistics. Specifically, the service provider predefines a binary pattern together with corresponding patch-level thresholds. To detect a watermark in a given image, we evaluate whether the luminance of each patch surpasses its threshold and then verify whether the resulting binary pattern aligns with the target one. A simple statistical analysis demonstrates that the false positive rate of the proposed method can be effectively controlled, thereby ensuring certified detection. To enable seamless watermark injection across different paradigms, we leverage the widely adopted guidance technique as a plug-and-play mechanism and develop the \emph{watermark guidance}. This design enables Luminark to achieve generality across state-of-the-art generative models without compromising image quality. Empirically, we evaluate our approach on nine models spanning diffusion, autoregressive, and hybrid frameworks. Across all evaluations, Luminark consistently demonstrates high detection accuracy, strong robustness against common image transformations, and good performance on visual quality.
>
---
#### [new 050] Point-SRA: Self-Representation Alignment for 3D Representation Learning
- **分类: cs.CV**

- **简介: 该论文提出Point-SRA，用于3D表示学习，解决点云多样性带来的重建难题，通过自蒸馏和概率建模提升性能。**

- **链接: [https://arxiv.org/pdf/2601.01746v1](https://arxiv.org/pdf/2601.01746v1)**

> **作者:** Lintong Wei; Jian Lu; Haozhe Cheng; Jihua Zhu; Kaibing Zhang
>
> **备注:** This is an AAAI 2026 accepted paper titled "Point-SRA: Self-Representation Alignment for 3D Representation Learning", spanning 13 pages in total. The submission includes 7 figures (fig1 to fig7) that visually support the technical analysis
>
> **摘要:** Masked autoencoders (MAE) have become a dominant paradigm in 3D representation learning, setting new performance benchmarks across various downstream tasks. Existing methods with fixed mask ratio neglect multi-level representational correlations and intrinsic geometric structures, while relying on point-wise reconstruction assumptions that conflict with the diversity of point cloud. To address these issues, we propose a 3D representation learning method, termed Point-SRA, which aligns representations through self-distillation and probabilistic modeling. Specifically, we assign different masking ratios to the MAE to capture complementary geometric and semantic information, while the MeanFlow Transformer (MFT) leverages cross-modal conditional embeddings to enable diverse probabilistic reconstruction. Our analysis further reveals that representations at different time steps in MFT also exhibit complementarity. Therefore, a Dual Self-Representation Alignment mechanism is proposed at both the MAE and MFT levels. Finally, we design a Flow-Conditioned Fine-Tuning Architecture to fully exploit the point cloud distribution learned via MeanFlow. Point-SRA outperforms Point-MAE by 5.37% on ScanObjectNN. On intracranial aneurysm segmentation, it reaches 96.07% mean IoU for arteries and 86.87% for aneurysms. For 3D object detection, Point-SRA achieves 47.3% AP@50, surpassing MaskPoint by 5.12%.
>
---
#### [new 051] Agentic AI in Remote Sensing: Foundations, Taxonomy, and Emerging Systems
- **分类: cs.CV**

- **简介: 该论文属于遥感领域，探讨如何将自主代理AI应用于地球观测分析。解决传统模型缺乏顺序规划与工具协调的问题，提出分类体系并分析架构基础。**

- **链接: [https://arxiv.org/pdf/2601.01891v1](https://arxiv.org/pdf/2601.01891v1)**

> **作者:** Niloufar Alipour Talemi; Julia Boone; Fatemeh Afghah
>
> **备注:** Accepted to the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026, GeoCV Workshop
>
> **摘要:** The paradigm of Earth Observation analysis is shifting from static deep learning models to autonomous agentic AI. Although recent vision foundation models and multimodal large language models advance representation learning, they often lack the sequential planning and active tool orchestration required for complex geospatial workflows. This survey presents the first comprehensive review of agentic AI in remote sensing. We introduce a unified taxonomy distinguishing between single-agent copilots and multi-agent systems while analyzing architectural foundations such as planning mechanisms, retrieval-augmented generation, and memory structures. Furthermore, we review emerging benchmarks that move the evaluation from pixel-level accuracy to trajectory-aware reasoning correctness. By critically examining limitations in grounding, safety, and orchestration, this work outlines a strategic roadmap for the development of robust, autonomous geospatial intelligence.
>
---
#### [new 052] A Comparative Study of Custom CNNs, Pre-trained Models, and Transfer Learning Across Multiple Visual Datasets
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像分类任务，比较了自定义CNN、预训练模型和迁移学习在多个数据集上的表现，旨在找出最优的模型选择策略。**

- **链接: [https://arxiv.org/pdf/2601.02246v1](https://arxiv.org/pdf/2601.02246v1)**

> **作者:** Annoor Sharara Akhand
>
> **摘要:** Convolutional Neural Networks (CNNs) are a standard approach for visual recognition due to their capacity to learn hierarchical representations from raw pixels. In practice, practitioners often choose among (i) training a compact custom CNN from scratch, (ii) using a large pre-trained CNN as a fixed feature extractor, and (iii) performing transfer learning via partial or full fine-tuning of a pre-trained backbone. This report presents a controlled comparison of these three paradigms across five real-world image classification datasets spanning road-surface defect recognition, agricultural variety identification, fruit/leaf disease recognition, pedestrian walkway encroachment recognition, and unauthorized vehicle recognition. Models are evaluated using accuracy and macro F1-score, complemented by efficiency metrics including training time per epoch and parameter counts. The results show that transfer learning consistently yields the strongest predictive performance, while the custom CNN provides an attractive efficiency--accuracy trade-off, especially when compute and memory budgets are constrained.
>
---
#### [new 053] Forget Less by Learning from Parents Through Hierarchical Relationships
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于持续学习任务，旨在解决模型在顺序学习新概念时的灾难性遗忘问题。提出FLLP框架，通过超平面中的父子关系机制，提升模型的泛化与稳定性。**

- **链接: [https://arxiv.org/pdf/2601.01892v1](https://arxiv.org/pdf/2601.01892v1)**

> **作者:** Arjun Ramesh Kaushik; Naresh Kumar Devulapally; Vishnu Suresh Lokhande; Nalini K. Ratha; Venu Govindaraju
>
> **备注:** Accepted at AAAI-26
>
> **摘要:** Custom Diffusion Models (CDMs) offer impressive capabilities for personalization in generative modeling, yet they remain vulnerable to catastrophic forgetting when learning new concepts sequentially. Existing approaches primarily focus on minimizing interference between concepts, often neglecting the potential for positive inter-concept interactions. In this work, we present Forget Less by Learning from Parents (FLLP), a novel framework that introduces a parent-child inter-concept learning mechanism in hyperbolic space to mitigate forgetting. By embedding concept representations within a Lorentzian manifold, naturally suited to modeling tree-like hierarchies, we define parent-child relationships in which previously learned concepts serve as guidance for adapting to new ones. Our method not only preserves prior knowledge but also supports continual integration of new concepts. We validate FLLP on three public datasets and one synthetic benchmark, showing consistent improvements in both robustness and generalization.
>
---
#### [new 054] FMVP: Masked Flow Matching for Adversarial Video Purification
- **分类: cs.CV**

- **简介: 该论文属于视频对抗净化任务，解决视频识别模型受对抗攻击的问题。提出FMVP方法，通过流匹配和掩码策略有效去除噪声，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.02228v1](https://arxiv.org/pdf/2601.02228v1)**

> **作者:** Duoxun Tang; Xueyi Zhang; Chak Hin Wang; Xi Xiao; Dasen Dai; Xinhang Jiang; Wentao Shi; Rui Li; Qing Li
>
> **摘要:** Video recognition models remain vulnerable to adversarial attacks, while existing diffusion-based purification methods suffer from inefficient sampling and curved trajectories. Directly regressing clean videos from adversarial inputs often fails to recover faithful content due to the subtle nature of perturbations; this necessitates physically shattering the adversarial structure. Therefore, we propose Flow Matching for Adversarial Video Purification FMVP. FMVP physically shatters global adversarial structures via a masking strategy and reconstructs clean video dynamics using Conditional Flow Matching (CFM) with an inpainting objective. To further decouple semantic content from adversarial noise, we design a Frequency-Gated Loss (FGL) that explicitly suppresses high-frequency adversarial residuals while preserving low-frequency fidelity. We design Attack-Aware and Generalist training paradigms to handle known and unknown threats, respectively. Extensive experiments on UCF-101 and HMDB-51 demonstrate that FMVP outperforms state-of-the-art methods (DiffPure, Defense Patterns (DP), Temporal Shuffling (TS) and FlowPure), achieving robust accuracy exceeding 87% against PGD and 89% against CW attacks. Furthermore, FMVP demonstrates superior robustness against adaptive attacks (DiffHammer) and functions as a zero-shot adversarial detector, attaining detection accuracies of 98% for PGD and 79% for highly imperceptible CW attacks.
>
---
#### [new 055] Animated 3DGS Avatars in Diverse Scenes with Consistent Lighting and Shadows
- **分类: cs.CV**

- **简介: 该论文属于三维动画任务，解决动态3DGS角色与场景光照阴影一致性问题。提出DGSM和SH relighting方法，实现无网格的阴影计算与光照渲染。**

- **链接: [https://arxiv.org/pdf/2601.01660v1](https://arxiv.org/pdf/2601.01660v1)**

> **作者:** Aymen Mir; Riza Alp Guler; Jian Wang; Gerard Pons-Moll; Bing Zhou
>
> **备注:** Our project page is available at https://miraymen.github.io/dgsm
>
> **摘要:** We present a method for consistent lighting and shadows when animated 3D Gaussian Splatting (3DGS) avatars interact with 3DGS scenes or with dynamic objects inserted into otherwise static scenes. Our key contribution is Deep Gaussian Shadow Maps (DGSM), a modern analogue of the classical shadow mapping algorithm tailored to the volumetric 3DGS representation. Building on the classic deep shadow mapping idea, we show that 3DGS admits closed form light accumulation along light rays, enabling volumetric shadow computation without meshing. For each estimated light, we tabulate transmittance over concentric radial shells and store them in octahedral atlases, which modern GPUs can sample in real time per query to attenuate affected scene Gaussians and thus cast and receive shadows consistently. To relight moving avatars, we approximate the local environment illumination with HDRI probes represented in a spherical harmonic (SH) basis and apply a fast per Gaussian radiance transfer, avoiding explicit BRDF estimation or offline optimization. We demonstrate environment consistent lighting for avatars from AvatarX and ActorsHQ, composited into ScanNet++, DL3DV, and SuperSplat scenes, and show interactions with inserted objects. Across single and multi avatar settings, DGSM and SH relighting operate fully in the volumetric 3DGS representation, yielding coherent shadows and relighting while avoiding meshing.
>
---
#### [new 056] InfiniteVGGT: Visual Geometry Grounded Transformer for Endless Streams
- **分类: cs.CV**

- **简介: 该论文提出InfiniteVGGT，解决3D视觉几何理解在持续流数据中的可扩展性与长期稳定性问题。通过滚动记忆机制和优化策略，实现无限时序处理，并引入Long3D基准进行评估。**

- **链接: [https://arxiv.org/pdf/2601.02281v1](https://arxiv.org/pdf/2601.02281v1)**

> **作者:** Shuai Yuan; Yantai Yang; Xiaotian Yang; Xupeng Zhang; Zhonghao Zhao; Lingming Zhang; Zhipeng Zhang
>
> **摘要:** The grand vision of enabling persistent, large-scale 3D visual geometry understanding is shackled by the irreconcilable demands of scalability and long-term stability. While offline models like VGGT achieve inspiring geometry capability, their batch-based nature renders them irrelevant for live systems. Streaming architectures, though the intended solution for live operation, have proven inadequate. Existing methods either fail to support truly infinite-horizon inputs or suffer from catastrophic drift over long sequences. We shatter this long-standing dilemma with InfiniteVGGT, a causal visual geometry transformer that operationalizes the concept of a rolling memory through a bounded yet adaptive and perpetually expressive KV cache. Capitalizing on this, we devise a training-free, attention-agnostic pruning strategy that intelligently discards obsolete information, effectively ``rolling'' the memory forward with each new frame. Fully compatible with FlashAttention, InfiniteVGGT finally alleviates the compromise, enabling infinite-horizon streaming while outperforming existing streaming methods in long-term stability. The ultimate test for such a system is its performance over a truly infinite horizon, a capability that has been impossible to rigorously validate due to the lack of extremely long-term, continuous benchmarks. To address this critical gap, we introduce the Long3D benchmark, which, for the first time, enables a rigorous evaluation of continuous 3D geometry estimation on sequences about 10,000 frames. This provides the definitive evaluation platform for future research in long-term 3D geometry understanding. Code is available at: https://github.com/AutoLab-SAI-SJTU/InfiniteVGGT
>
---
#### [new 057] EgoGrasp: World-Space Hand-Object Interaction Estimation from Egocentric Videos
- **分类: cs.CV; cs.AI; cs.GR**

- **简介: 该论文提出EgoGrasp，解决从第一视角视频中重建世界空间手物交互的问题。通过多阶段框架和新模型，提升动态场景下的交互估计精度。**

- **链接: [https://arxiv.org/pdf/2601.01050v1](https://arxiv.org/pdf/2601.01050v1)**

> **作者:** Hongming Fu; Wenjia Wang; Xiaozhen Qiao; Shuo Yang; Zheng Liu; Bo Zhao
>
> **摘要:** We propose EgoGrasp, the first method to reconstruct world-space hand-object interactions (W-HOI) from egocentric monocular videos with dynamic cameras in the wild. Accurate W-HOI reconstruction is critical for understanding human behavior and enabling applications in embodied intelligence and virtual reality. However, existing hand-object interactions (HOI) methods are limited to single images or camera coordinates, failing to model temporal dynamics or consistent global trajectories. Some recent approaches attempt world-space hand estimation but overlook object poses and HOI constraints. Their performance also suffers under severe camera motion and frequent occlusions common in egocentric in-the-wild videos. To address these challenges, we introduce a multi-stage framework with a robust pre-process pipeline built on newly developed spatial intelligence models, a whole-body HOI prior model based on decoupled diffusion models, and a multi-objective test-time optimization paradigm. Our HOI prior model is template-free and scalable to multiple objects. In experiments, we prove our method achieving state-of-the-art performance in W-HOI reconstruction.
>
---
#### [new 058] SortWaste: A Densely Annotated Dataset for Object Detection in Industrial Waste Sorting
- **分类: cs.CV**

- **简介: 该论文属于工业废弃物检测任务，旨在解决自动化分拣效率低的问题。提出SortWaste数据集和ClutterScore评估指标，以提升检测模型性能。**

- **链接: [https://arxiv.org/pdf/2601.02299v1](https://arxiv.org/pdf/2601.02299v1)**

> **作者:** Sara Inácio; Hugo Proença; João C. Neves
>
> **备注:** 9 pages
>
> **摘要:** The increasing production of waste, driven by population growth, has created challenges in managing and recycling materials effectively. Manual waste sorting is a common practice; however, it remains inefficient for handling large-scale waste streams and presents health risks for workers. On the other hand, existing automated sorting approaches still struggle with the high variability, clutter, and visual complexity of real-world waste streams. The lack of real-world datasets for waste sorting is a major reason automated systems for this problem are underdeveloped. Accordingly, we introduce SortWaste, a densely annotated object detection dataset collected from a Material Recovery Facility. Additionally, we contribute to standardizing waste detection in sorting lines by proposing ClutterScore, an objective metric that gauges the scene's hardness level using a set of proxies that affect visual complexity (e.g., object count, class and size entropy, and spatial overlap). In addition to these contributions, we provide an extensive benchmark of state-of-the-art object detection models, detailing their results with respect to the hardness level assessed by the proposed metric. Despite achieving promising results (mAP of 59.7% in the plastic-only detection task), performance significantly decreases in highly cluttered scenes. This highlights the need for novel and more challenging datasets on the topic.
>
---
#### [new 059] DiffProxy: Multi-View Human Mesh Recovery via Diffusion-Generated Dense Proxies
- **分类: cs.CV**

- **简介: 该论文属于人体网格重建任务，解决多视角图像中因标注不准确导致的模型训练偏差问题。通过生成一致的人体代理，结合扩散模型提升重建效果。**

- **链接: [https://arxiv.org/pdf/2601.02267v1](https://arxiv.org/pdf/2601.02267v1)**

> **作者:** Renke Wang; Zhenyu Zhang; Ying Tai; Jian Yang
>
> **备注:** Page: https://wrk226.github.io/DiffProxy.html, Code: https://github.com/wrk226/DiffProxy
>
> **摘要:** Human mesh recovery from multi-view images faces a fundamental challenge: real-world datasets contain imperfect ground-truth annotations that bias the models' training, while synthetic data with precise supervision suffers from domain gap. In this paper, we propose DiffProxy, a novel framework that generates multi-view consistent human proxies for mesh recovery. Central to DiffProxy is leveraging the diffusion-based generative priors to bridge the synthetic training and real-world generalization. Its key innovations include: (1) a multi-conditional mechanism for generating multi-view consistent, pixel-aligned human proxies; (2) a hand refinement module that incorporates flexible visual prompts to enhance local details; and (3) an uncertainty-aware test-time scaling method that increases robustness to challenging cases during optimization. These designs ensure that the mesh recovery process effectively benefits from the precise synthetic ground truth and generative advantages of the diffusion-based pipeline. Trained entirely on synthetic data, DiffProxy achieves state-of-the-art performance across five real-world benchmarks, demonstrating strong zero-shot generalization particularly on challenging scenarios with occlusions and partial views. Project page: https://wrk226.github.io/DiffProxy.html
>
---
#### [new 060] Forget Less by Learning Together through Concept Consolidation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于增量学习任务，解决持续学习中新概念导致的灾难性遗忘问题。提出FL2T框架，通过概念间交互提升知识保留与迁移能力。**

- **链接: [https://arxiv.org/pdf/2601.01963v1](https://arxiv.org/pdf/2601.01963v1)**

> **作者:** Arjun Ramesh Kaushik; Naresh Kumar Devulapally; Vishnu Suresh Lokhande; Nalini Ratha; Venu Govindaraju
>
> **备注:** Accepted at WACV-26
>
> **摘要:** Custom Diffusion Models (CDMs) have gained significant attention due to their remarkable ability to personalize generative processes. However, existing CDMs suffer from catastrophic forgetting when continuously learning new concepts. Most prior works attempt to mitigate this issue under the sequential learning setting with a fixed order of concept inflow and neglect inter-concept interactions. In this paper, we propose a novel framework - Forget Less by Learning Together (FL2T) - that enables concurrent and order-agnostic concept learning while addressing catastrophic forgetting. Specifically, we introduce a set-invariant inter-concept learning module where proxies guide feature selection across concepts, facilitating improved knowledge retention and transfer. By leveraging inter-concept guidance, our approach preserves old concepts while efficiently incorporating new ones. Extensive experiments, across three datasets, demonstrates that our method significantly improves concept retention and mitigates catastrophic forgetting, highlighting the effectiveness of inter-concept catalytic behavior in incremental concept learning of ten tasks with at least 2% gain on average CLIP Image Alignment scores.
>
---
#### [new 061] SLGNet: Synergizing Structural Priors and Language-Guided Modulation for Multimodal Object Detection
- **分类: cs.CV**

- **简介: 该论文属于多模态目标检测任务，旨在解决RGB与红外图像融合中的结构一致性与环境适应性问题。通过引入结构感知适配器和语言引导调制模块，提升检测性能并减少参数量。**

- **链接: [https://arxiv.org/pdf/2601.02249v1](https://arxiv.org/pdf/2601.02249v1)**

> **作者:** Xiantai Xiang; Guangyao Zhou; Zixiao Wen; Wenshuai Li; Ben Niu; Feng Wang; Lijia Huang; Qiantong Wang; Yuhan Liu; Zongxu Pan; Yuxin Hu
>
> **摘要:** Multimodal object detection leveraging RGB and Infrared (IR) images is pivotal for robust perception in all-weather scenarios. While recent adapter-based approaches efficiently transfer RGB-pretrained foundation models to this task, they often prioritize model efficiency at the expense of cross-modal structural consistency. Consequently, critical structural cues are frequently lost when significant domain gaps arise, such as in high-contrast or nighttime environments. Moreover, conventional static multimodal fusion mechanisms typically lack environmental awareness, resulting in suboptimal adaptation and constrained detection performance under complex, dynamic scene variations. To address these limitations, we propose SLGNet, a parameter-efficient framework that synergizes hierarchical structural priors and language-guided modulation within a frozen Vision Transformer (ViT)-based foundation model. Specifically, we design a Structure-Aware Adapter to extract hierarchical structural representations from both modalities and dynamically inject them into the ViT to compensate for structural degradation inherent in ViT-based backbones. Furthermore, we propose a Language-Guided Modulation module that exploits VLM-driven structured captions to dynamically recalibrate visual features, thereby endowing the model with robust environmental awareness. Extensive experiments on the LLVIP, FLIR, KAIST, and DroneVehicle datasets demonstrate that SLGNet establishes new state-of-the-art performance. Notably, on the LLVIP benchmark, our method achieves an mAP of 66.1, while reducing trainable parameters by approximately 87% compared to traditional full fine-tuning. This confirms SLGNet as a robust and efficient solution for multimodal perception.
>
---
#### [new 062] Subimage Overlap Prediction: Task-Aligned Self-Supervised Pretraining For Semantic Segmentation In Remote Sensing Imagery
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于遥感图像语义分割任务，旨在解决标注数据不足的问题。通过引入子图重叠预测作为自监督预训练任务，减少对大量预训练数据的依赖，提升模型性能与收敛速度。**

- **链接: [https://arxiv.org/pdf/2601.01781v1](https://arxiv.org/pdf/2601.01781v1)**

> **作者:** Lakshay Sharma; Alex Marin
>
> **备注:** Accepted at CV4EO Workshop at WACV 2026
>
> **摘要:** Self-supervised learning (SSL) methods have become a dominant paradigm for creating general purpose models whose capabilities can be transferred to downstream supervised learning tasks. However, most such methods rely on vast amounts of pretraining data. This work introduces Subimage Overlap Prediction, a novel self-supervised pretraining task to aid semantic segmentation in remote sensing imagery that uses significantly lesser pretraining imagery. Given an image, a sub-image is extracted and the model is trained to produce a semantic mask of the location of the extracted sub-image within the original image. We demonstrate that pretraining with this task results in significantly faster convergence, and equal or better performance (measured via mIoU) on downstream segmentation. This gap in convergence and performance widens when labeled training data is reduced. We show this across multiple architecture types, and with multiple downstream datasets. We also show that our method matches or exceeds performance while requiring significantly lesser pretraining data relative to other SSL methods. Code and model weights are provided at \href{https://github.com/sharmalakshay93/subimage-overlap-prediction}{github.com/sharmalakshay93/subimage-overlap-prediction}.
>
---
#### [new 063] Agentic Retoucher for Text-To-Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成任务，解决生成图像中的细小失真问题。提出Agentic Retoucher框架，通过感知、推理和行动代理实现精准修复。**

- **链接: [https://arxiv.org/pdf/2601.02046v1](https://arxiv.org/pdf/2601.02046v1)**

> **作者:** Shaocheng Shen; Jianfeng Liang. Chunlei Cai; Cong Geng; Huiyu Duan; Xiaoyun Zhang; Qiang Hu; Guangtao Zhai
>
> **摘要:** Text-to-image (T2I) diffusion models such as SDXL and FLUX have achieved impressive photorealism, yet small-scale distortions remain pervasive in limbs, face, text and so on. Existing refinement approaches either perform costly iterative re-generation or rely on vision-language models (VLMs) with weak spatial grounding, leading to semantic drift and unreliable local edits. To close this gap, we propose Agentic Retoucher, a hierarchical decision-driven framework that reformulates post-generation correction as a human-like perception-reasoning-action loop. Specifically, we design (1) a perception agent that learns contextual saliency for fine-grained distortion localization under text-image consistency cues, (2) a reasoning agent that performs human-aligned inferential diagnosis via progressive preference alignment, and (3) an action agent that adaptively plans localized inpainting guided by user preference. This design integrates perceptual evidence, linguistic reasoning, and controllable correction into a unified, self-corrective decision process. To enable fine-grained supervision and quantitative evaluation, we further construct GenBlemish-27K, a dataset of 6K T2I images with 27K annotated artifact regions across 12 categories. Extensive experiments demonstrate that Agentic Retoucher consistently outperforms state-of-the-art methods in perceptual quality, distortion localization and human preference alignment, establishing a new paradigm for self-corrective and perceptually reliable T2I generation.
>
---
#### [new 064] AFTER: Mitigating the Object Hallucination of LVLM via Adaptive Factual-Guided Activation Editing
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决对象幻觉问题。通过提出AFTER方法，利用事实引导的激活编辑来减少语言偏差带来的幻觉现象。**

- **链接: [https://arxiv.org/pdf/2601.01957v1](https://arxiv.org/pdf/2601.01957v1)**

> **作者:** Tianbo Wang; Yuqing Ma; Kewei Liao; Zhange Zhang; Simin Li; Jinyang Guo; Xianglong Liu
>
> **摘要:** Large Vision-Language Models (LVLMs) have achieved substantial progress in cross-modal tasks. However, due to language bias, LVLMs are susceptible to object hallucination, which can be primarily divided into category, attribute, and relation hallucination, significantly impeding the trustworthy AI applications. Editing the internal activations of LVLMs has shown promising effectiveness in mitigating hallucinations with minimal cost. However, previous editing approaches neglect the effective guidance offered by factual textual semantics, thereby struggling to explicitly mitigate language bias. To address these issues, we propose Adaptive Factual-guided Visual-Textual Editing for hallucination mitigation (AFTER), which comprises Factual-Augmented Activation Steering (FAS) and Query-Adaptive Offset Optimization (QAO), to adaptively guides the original biased activations towards factual semantics. Specifically, FAS is proposed to provide factual and general guidance for activation editing, thereby explicitly modeling the precise visual-textual associations. Subsequently, QAO introduces a query-aware offset estimator to establish query-specific editing from the general steering vector, enhancing the diversity and granularity of editing. Extensive experiments on standard hallucination benchmarks across three widely adopted LVLMs validate the efficacy of the proposed AFTER, notably achieving up to a 16.3% reduction of hallucination over baseline on the AMBER benchmark. Our code and data will be released for reproducibility.
>
---
#### [new 065] Seeing the Unseen: Zooming in the Dark with Event Cameras
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于低光视频超分辨率任务，旨在提升低光、低分辨率视频的细节和质量。通过引入事件相机数据和Retinex模型，提出RetinexEVSR框架，有效增强视频清晰度并减少运行时间。**

- **链接: [https://arxiv.org/pdf/2601.02206v1](https://arxiv.org/pdf/2601.02206v1)**

> **作者:** Dachun Kai; Zeyu Xiao; Huyue Zhu; Jiaxiao Wang; Yueyi Zhang; Xiaoyan Sun
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** This paper addresses low-light video super-resolution (LVSR), aiming to restore high-resolution videos from low-light, low-resolution (LR) inputs. Existing LVSR methods often struggle to recover fine details due to limited contrast and insufficient high-frequency information. To overcome these challenges, we present RetinexEVSR, the first event-driven LVSR framework that leverages high-contrast event signals and Retinex-inspired priors to enhance video quality under low-light scenarios. Unlike previous approaches that directly fuse degraded signals, RetinexEVSR introduces a novel bidirectional cross-modal fusion strategy to extract and integrate meaningful cues from noisy event data and degraded RGB frames. Specifically, an illumination-guided event enhancement module is designed to progressively refine event features using illumination maps derived from the Retinex model, thereby suppressing low-light artifacts while preserving high-contrast details. Furthermore, we propose an event-guided reflectance enhancement module that utilizes the enhanced event features to dynamically recover reflectance details via a multi-scale fusion mechanism. Experimental results show that our RetinexEVSR achieves state-of-the-art performance on three datasets. Notably, on the SDSD benchmark, our method can get up to 2.95 dB gain while reducing runtime by 65% compared to prior event-based methods. Code: https://github.com/DachunKai/RetinexEVSR.
>
---
#### [new 066] 360DVO: Deep Visual Odometry for Monocular 360-Degree Camera
- **分类: cs.CV**

- **简介: 该论文属于视觉里程计任务，解决单目360度相机在复杂场景下的定位问题。提出360DVO框架，提升鲁棒性和精度。**

- **链接: [https://arxiv.org/pdf/2601.02309v1](https://arxiv.org/pdf/2601.02309v1)**

> **作者:** Xiaopeng Guo; Yinzhe Xu; Huajian Huang; Sai-Kit Yeung
>
> **备注:** 12 pages. Received by RA-L
>
> **摘要:** Monocular omnidirectional visual odometry (OVO) systems leverage 360-degree cameras to overcome field-of-view limitations of perspective VO systems. However, existing methods, reliant on handcrafted features or photometric objectives, often lack robustness in challenging scenarios, such as aggressive motion and varying illumination. To address this, we present 360DVO, the first deep learning-based OVO framework. Our approach introduces a distortion-aware spherical feature extractor (DAS-Feat) that adaptively learns distortion-resistant features from 360-degree images. These sparse feature patches are then used to establish constraints for effective pose estimation within a novel omnidirectional differentiable bundle adjustment (ODBA) module. To facilitate evaluation in realistic settings, we also contribute a new real-world OVO benchmark. Extensive experiments on this benchmark and public synthetic datasets (TartanAir V2 and 360VO) demonstrate that 360DVO surpasses state-of-the-art baselines (including 360VO and OpenVSLAM), improving robustness by 50% and accuracy by 37.5%. Homepage: https://chris1004336379.github.io/360DVO-homepage
>
---
#### [new 067] Free Energy-Based Modeling of Emotional Dynamics in Video Advertisements
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于情感建模任务，旨在无需外部信息即可解释性地估计视频广告中的情绪。通过自由能原理，利用场景特征量化情绪维度，验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2601.00812v1](https://arxiv.org/pdf/2601.00812v1)**

> **作者:** Takashi Ushio; Kazuhiro Onishi; Hideyoshi Yanagisawa
>
> **备注:** This article has been accepted for publication in IEEE Access and will be published shortly
>
> **摘要:** Emotional responses during advertising video viewing are recognized as essential for understanding media effects because they have influenced attention, memory, and purchase intention. To establish a methodological basis for explainable emotion estimation without relying on external information such as physiological signals or subjective ratings, we have quantified "pleasantness," "surprise," and "habituation" solely from scene-level expression features of advertising videos, drawing on the free energy(FE) principle, which has provided a unified account of perception, learning, and behavior. In this framework, Kullback-Leibler divergence (KLD) has captured prediction error, Bayesian surprise (BS) has captured belief updates, and uncertainty (UN) has reflected prior ambiguity, and together they have formed the core components of FE. Using 1,059 15 s food video advertisements, the experiments have shown that KLD has reflected "pleasantness" associated with brand presentation, BS has captured "surprise" arising from informational complexity, and UN has reflected "surprise" driven by uncertainty in element types and spatial arrangements, as well as by the variability and quantity of presented elements. This study also identified three characteristic emotional patterns, namely uncertain stimulus, sustained high emotion, and momentary peak and decay, demonstrating the usefulness of the proposed method. Robustness across nine hyperparameter settings and generalization tests with six types of Japanese advertising videos (three genres and two durations) confirmed that these tendencies remained stable. This work can be extended by integrating a wider range of expression elements and validating the approach through subjective ratings, ultimately guiding the development of technologies that can support the creation of more engaging advertising videos.
>
---
#### [new 068] GenCAMO: Scene-Graph Contextual Decoupling for Environment-aware and Mask-free Camouflage Image-Dense Annotation Generation
- **分类: cs.CV**

- **简介: 该论文属于环境感知的伪装图像密集标注生成任务，旨在解决高质量伪装数据稀缺问题。通过生成模型合成真实数据，提升复杂场景下的密集预测性能。**

- **链接: [https://arxiv.org/pdf/2601.01181v1](https://arxiv.org/pdf/2601.01181v1)**

> **作者:** Chenglizhao Chen; Shaojiang Yuan; Xiaoxue Lu; Mengke Song; Jia Song; Zhenyu Wu; Wenfeng Song; Shuai Li
>
> **摘要:** Conceal dense prediction (CDP), especially RGB-D camouflage object detection and open-vocabulary camouflage object segmentation, plays a crucial role in advancing the understanding and reasoning of complex camouflage scenes. However, high-quality and large-scale camouflage datasets with dense annotation remain scarce due to expensive data collection and labeling costs. To address this challenge, we explore leveraging generative models to synthesize realistic camouflage image-dense data for training CDP models with fine-grained representations, prior knowledge, and auxiliary reasoning. Concretely, our contributions are threefold: (i) we introduce GenCAMO-DB, a large-scale camouflage dataset with multi-modal annotations, including depth maps, scene graphs, attribute descriptions, and text prompts; (ii) we present GenCAMO, an environment-aware and mask-free generative framework that produces high-fidelity camouflage image-dense annotations; (iii) extensive experiments across multiple modalities demonstrate that GenCAMO significantly improves dense prediction performance on complex camouflage scenes by providing high-quality synthetic data. The code and datasets will be released after paper acceptance.
>
---
#### [new 069] API: Empowering Generalizable Real-World Image Dehazing via Adaptive Patch Importance Learning
- **分类: cs.CV**

- **简介: 该论文属于图像去雾任务，旨在解决真实场景下去雾效果不佳的问题。通过引入API框架和MNCD损失函数，提升模型的泛化能力和去雾效果。**

- **链接: [https://arxiv.org/pdf/2601.01992v1](https://arxiv.org/pdf/2601.01992v1)**

> **作者:** Chen Zhu; Huiwen Zhang; Yujie Li; Mu He; Xiaotian Qiao
>
> **摘要:** Real-world image dehazing is a fundamental yet challenging task in low-level vision. Existing learning-based methods often suffer from significant performance degradation when applied to complex real-world hazy scenes, primarily due to limited training data and the intrinsic complexity of haze density distributions.To address these challenges, we introduce a novel Adaptive Patch Importance-aware (API) framework for generalizable real-world image dehazing. Specifically, our framework consists of an Automatic Haze Generation (AHG) module and a Density-aware Haze Removal (DHR) module. AHG provides a hybrid data augmentation strategy by generating realistic and diverse hazy images as additional high-quality training data. DHR considers hazy regions with varying haze density distributions for generalizable real-world image dehazing in an adaptive patch importance-aware manner. To alleviate the ambiguity of the dehazed image details, we further introduce a new Multi-Negative Contrastive Dehazing (MNCD) loss, which fully utilizes information from multiple negative samples across both spatial and frequency domains. Extensive experiments demonstrate that our framework achieves state-of-the-art performance across multiple real-world benchmarks, delivering strong results in both quantitative metrics and qualitative visual quality, and exhibiting robust generalization across diverse haze distributions.
>
---
#### [new 070] Rethinking Multimodal Few-Shot 3D Point Cloud Segmentation: From Fused Refinement to Decoupled Arbitration
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于3D点云语义分割任务，解决少样本学习中的“塑形-稳定困境”和语义混淆问题，提出DA-FSS模型实现几何与语义路径的解耦优化。**

- **链接: [https://arxiv.org/pdf/2601.01456v1](https://arxiv.org/pdf/2601.01456v1)**

> **作者:** Wentao Bian; Fenglei Xu
>
> **备注:** 10 pages, 4 figures, 3 tables
>
> **摘要:** In this paper, we revisit multimodal few-shot 3D point cloud semantic segmentation (FS-PCS), identifying a conflict in "Fuse-then-Refine" paradigms: the "Plasticity-Stability Dilemma." In addition, CLIP's inter-class confusion can result in semantic blindness. To address these issues, we present the Decoupled-experts Arbitration Few-Shot SegNet (DA-FSS), a model that effectively distinguishes between semantic and geometric paths and mutually regularizes their gradients to achieve better generalization. DA-FSS employs the same backbone and pre-trained text encoder as MM-FSS to generate text embeddings, which can increase free modalities' utilization rate and better leverage each modality's information space. To achieve this, we propose a Parallel Expert Refinement module to generate each modal correlation. We also propose a Stacked Arbitration Module (SAM) to perform convolutional fusion and arbitrate correlations for each modality pathway. The Parallel Experts decouple two paths: a Geometric Expert maintains plasticity, and a Semantic Expert ensures stability. They are coordinated via a Decoupled Alignment Module (DAM) that transfers knowledge without propagating confusion. Experiments on popular datasets (S3DIS, ScanNet) demonstrate the superiority of DA-FSS over MM-FSS. Meanwhile, geometric boundaries, completeness, and texture differentiation are all superior to the baseline. The code is available at: https://github.com/MoWenQAQ/DA-FSS.
>
---
#### [new 071] Pediatric Pneumonia Detection from Chest X-Rays:A Comparative Study of Transfer Learning and Custom CNNs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于儿科肺炎检测任务，旨在提升X光片诊断准确性。通过比较迁移学习与自定义CNN，发现微调方法效果更优，可作为资源有限地区的筛查工具。**

- **链接: [https://arxiv.org/pdf/2601.00837v1](https://arxiv.org/pdf/2601.00837v1)**

> **作者:** Agniv Roy Choudhury
>
> **摘要:** Pneumonia is a leading cause of mortality in children under five, with over 700,000 deaths annually. Accurate diagnosis from chest X-rays is limited by radiologist availability and variability. Objective: This study compares custom CNNs trained from scratch with transfer learning (ResNet50, DenseNet121, EfficientNet-B0) for pediatric pneumonia detection, evaluating frozen-backbone and fine-tuning regimes. Methods: A dataset of 5,216 pediatric chest X-rays was split 80/10/10 for training, validation, and testing. Seven models were trained and assessed using accuracy, F1-score, and AUC. Grad-CAM visualizations provided explainability. Results: Fine-tuned ResNet50 achieved the best performance: 99.43\% accuracy, 99.61\% F1-score, and 99.93\% AUC, with only 3 misclassifications. Fine-tuning outperformed frozen-backbone models by 5.5 percentage points on average. Grad-CAM confirmed clinically relevant lung regions guided predictions. Conclusions: Transfer learning with fine-tuning substantially outperforms CNNs trained from scratch for pediatric pneumonia detection, showing near-perfect accuracy. This system has strong potential as a screening tool in resource-limited settings. Future work should validate these findings on multi-center and adult datasets. Keywords: Pneumonia detection, deep learning, transfer learning, CNN, chest X-ray, pediatric diagnosis, ResNet, DenseNet, EfficientNet, Grad-CAM.
>
---
#### [new 072] Car Drag Coefficient Prediction from 3D Point Clouds Using a Slice-Based Surrogate Model
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于 aerodynamic drag coefficient 预测任务，旨在解决传统方法计算成本高、效率低的问题。通过3D点云的切片处理和轻量模型，实现快速准确的Cd预测。**

- **链接: [https://arxiv.org/pdf/2601.02112v1](https://arxiv.org/pdf/2601.02112v1)**

> **作者:** Utkarsh Singh; Absaar Ali; Adarsh Roy
>
> **备注:** 14 pages, 5 figures. Published in: Bramer M., Stahl F. (eds) Artificial Intelligence XLII. SGAI 2025. Lecture Notes in Computer Science, vol 16302. Springer, Cham
>
> **摘要:** The automotive industry's pursuit of enhanced fuel economy and performance necessitates efficient aerodynamic design. However, traditional evaluation methods such as computational fluid dynamics (CFD) and wind tunnel testing are resource intensive, hindering rapid iteration in the early design stages. Machine learning-based surrogate models offer a promising alternative, yet many existing approaches suffer from high computational complexity, limited interpretability, or insufficient accuracy for detailed geometric inputs. This paper introduces a novel lightweight surrogate model for the prediction of the aerodynamic drag coefficient (Cd) based on a sequential slice-wise processing of the geometry of the 3D vehicle. Inspired by medical imaging, 3D point clouds of vehicles are decomposed into an ordered sequence of 2D cross-sectional slices along the stream-wise axis. Each slice is encoded by a lightweight PointNet2D module, and the sequence of slice embeddings is processed by a bidirectional LSTM to capture longitudinal geometric evolution. The model, trained and evaluated on the DrivAerNet++ dataset, achieves a high coefficient of determination (R^2 > 0.9528) and a low mean absolute error (MAE approx 6.046 x 10^{-3}) in Cd prediction. With an inference time of approximately 0.025 seconds per sample on a consumer-grade GPU, our approach provides fast, accurate, and interpretable aerodynamic feedback, facilitating more agile and informed automotive design exploration.
>
---
#### [new 073] Why Commodity WiFi Sensors Fail at Multi-Person Gait Identification: A Systematic Analysis Using ESP32
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于多人员步态识别任务，探讨商品WiFi传感器在多人情况下的性能问题。通过分析六种信号分离方法，发现其准确率低且受硬件限制，无法可靠区分多人。**

- **链接: [https://arxiv.org/pdf/2601.02177v1](https://arxiv.org/pdf/2601.02177v1)**

> **作者:** Oliver Custance; Saad Khan; Simon Parkinson
>
> **摘要:** WiFi Channel State Information (CSI) has shown promise for single-person gait identification, with numerous studies reporting high accuracy. However, multi-person identification remains largely unexplored, with the limited existing work relying on complex, expensive setups requiring modified firmware. A critical question remains unanswered: is poor multi-person performance an algorithmic limitation or a fundamental hardware constraint? We systematically evaluate six diverse signal separation methods (FastICA, SOBI, PCA, NMF, Wavelet, Tensor Decomposition) across seven scenarios with 1-10 people using commodity ESP32 WiFi sensors--a simple, low-cost, off-the-shelf solution. Through novel diagnostic metrics (intra-subject variability, inter-subject distinguishability, performance degradation rate), we reveal that all methods achieve similarly low accuracy (45-56\%, $σ$=3.74\%) with statistically insignificant differences (p $>$ 0.05). Even the best-performing method, NMF, achieves only 56\% accuracy. Our analysis reveals high intra-subject variability, low inter-subject distinguishability, and severe performance degradation as person count increases, indicating that commodity ESP32 sensors cannot provide sufficient signal quality for reliable multi-person separation.
>
---
#### [new 074] CornViT: A Multi-Stage Convolutional Vision Transformer Framework for Hierarchical Corn Kernel Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CornViT，用于玉米籽粒分级的多阶段视觉Transformer框架，解决人工检测效率低的问题。通过三个阶段分类实现籽粒纯度、形态和胚芽方向的自动识别。**

- **链接: [https://arxiv.org/pdf/2601.00897v1](https://arxiv.org/pdf/2601.00897v1)**

> **作者:** Sai Teja Erukude; Jane Mascarenhas; Lior Shamir
>
> **备注:** 23 pages
>
> **摘要:** Accurate grading of corn kernels is critical for seed certification, directional seeding, and breeding, yet it is still predominantly performed by manual inspection. This work introduces CornViT, a three-stage Convolutional Vision Transformer (CvT) framework that emulates the hierarchical reasoning of human seed analysts for single-kernel evaluation. Three sequential CvT-13 classifiers operate on 384x384 RGB images: Stage 1 distinguishes pure from impure kernels; Stage 2 categorizes pure kernels into flat and round morphologies; and Stage 3 determines the embryo orientation (up vs. down) for pure, flat kernels. Starting from a public corn seed image collection, we manually relabeled and filtered images to construct three stage-specific datasets: 7265 kernels for purity, 3859 pure kernels for morphology, and 1960 pure-flat kernels for embryo orientation, all released as benchmarks. Head-only fine-tuning of ImageNet-22k pretrained CvT-13 backbones yields test accuracies of 93.76% for purity, 94.11% for shape, and 91.12% for embryo-orientation detection. Under identical training conditions, ResNet-50 reaches only 76.56 to 81.02 percent, whereas DenseNet-121 attains 86.56 to 89.38 percent accuracy. These results highlight the advantages of convolution-augmented self-attention for kernel analysis. To facilitate adoption, we deploy CornViT in a Flask-based web application that performs stage-wise inference and exposes interpretable outputs through a browser interface. Together, the CornViT framework, curated datasets, and web application provide a deployable solution for automated corn kernel quality assessment in seed quality workflows. Source code and data are publicly available.
>
---
#### [new 075] ParkGaussian: Surround-view 3D Gaussian Splatting for Autonomous Parking
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自动驾驶中的停车任务，解决停车场场景3D重建问题。提出ParkGaussian框架，结合3D高斯泼溅技术，提升停车线感知一致性。**

- **链接: [https://arxiv.org/pdf/2601.01386v1](https://arxiv.org/pdf/2601.01386v1)**

> **作者:** Xiaobao Wei; Zhangjie Ye; Yuxiang Gu; Zunjie Zhu; Yunfei Guo; Yingying Shen; Shan Zhao; Ming Lu; Haiyang Sun; Bing Wang; Guang Chen; Rongfeng Lu; Hangjun Ye
>
> **摘要:** Parking is a critical task for autonomous driving systems (ADS), with unique challenges in crowded parking slots and GPS-denied environments. However, existing works focus on 2D parking slot perception, mapping, and localization, 3D reconstruction remains underexplored, which is crucial for capturing complex spatial geometry in parking scenarios. Naively improving the visual quality of reconstructed parking scenes does not directly benefit autonomous parking, as the key entry point for parking is the slots perception module. To address these limitations, we curate the first benchmark named ParkRecon3D, specifically designed for parking scene reconstruction. It includes sensor data from four surround-view fisheye cameras with calibrated extrinsics and dense parking slot annotations. We then propose ParkGaussian, the first framework that integrates 3D Gaussian Splatting (3DGS) for parking scene reconstruction. To further improve the alignment between reconstruction and downstream parking slot detection, we introduce a slot-aware reconstruction strategy that leverages existing parking perception methods to enhance the synthesis quality of slot regions. Experiments on ParkRecon3D demonstrate that ParkGaussian achieves state-of-the-art reconstruction quality and better preserves perception consistency for downstream tasks. The code and dataset will be released at: https://github.com/wm-research/ParkGaussian
>
---
#### [new 076] PartImageNet++ Dataset: Enhancing Visual Models with High-Quality Part Annotations
- **分类: cs.CV**

- **简介: 该论文提出PartImageNet++数据集，解决现有数据集缺乏高质量部件标注的问题。通过引入多尺度部件监督模型，提升图像分类性能，并验证了部件标注在下游任务中的有效性。**

- **链接: [https://arxiv.org/pdf/2601.01454v1](https://arxiv.org/pdf/2601.01454v1)**

> **作者:** Xiao Li; Zilong Liu; Yining Liu; Zhuhong Li; Na Dong; Sitian Qin; Xiaolin Hu
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2407.10918
>
> **摘要:** To address the scarcity of high-quality part annotations in existing datasets, we introduce PartImageNet++ (PIN++), a dataset that provides detailed part annotations for all categories in ImageNet-1K. With 100 annotated images per category, totaling 100K images, PIN++ represents the most comprehensive dataset covering a diverse range of object categories. Leveraging PIN++, we propose a Multi-scale Part-supervised recognition Model (MPM) for robust classification on ImageNet-1K. We first trained a part segmentation network using PIN++ and used it to generate pseudo part labels for the remaining unannotated images. MPM then integrated a conventional recognition architecture with auxiliary bypass layers, jointly supervised by both pseudo part labels and the original part annotations. Furthermore, we conducted extensive experiments on PIN++, including part segmentation, object segmentation, and few-shot learning, exploring various ways to leverage part annotations in downstream tasks. Experimental results demonstrated that our approach not only enhanced part-based models for robust object recognition but also established strong baselines for multiple downstream tasks, highlighting the potential of part annotations in improving model performance. The dataset and the code are available at https://github.com/LixiaoTHU/PartImageNetPP.
>
---
#### [new 077] HeadLighter: Disentangling Illumination in Generative 3D Gaussian Heads via Lightstage Captures
- **分类: cs.CV**

- **简介: 该论文属于3D头像生成任务，解决光照与外观纠缠问题。通过设计双分支架构和渐进解耦训练，实现光照与外观的分离，支持可控重光照。**

- **链接: [https://arxiv.org/pdf/2601.02103v1](https://arxiv.org/pdf/2601.02103v1)**

> **作者:** Yating Wang; Yuan Sun; Xuan Wang; Ran Yi; Boyao Zhou; Yipengjing Sun; Hongyu Liu; Yinuo Wang; Lizhuang Ma
>
> **摘要:** Recent 3D-aware head generative models based on 3D Gaussian Splatting achieve real-time, photorealistic and view-consistent head synthesis. However, a fundamental limitation persists: the deep entanglement of illumination and intrinsic appearance prevents controllable relighting. Existing disentanglement methods rely on strong assumptions to enable weakly supervised learning, which restricts their capacity for complex illumination. To address this challenge, we introduce HeadLighter, a novel supervised framework that learns a physically plausible decomposition of appearance and illumination in head generative models. Specifically, we design a dual-branch architecture that separately models lighting-invariant head attributes and physically grounded rendering components. A progressive disentanglement training is employed to gradually inject head appearance priors into the generative architecture, supervised by multi-view images captured under controlled light conditions with a light stage setup. We further introduce a distillation strategy to generate high-quality normals for realistic rendering. Experiments demonstrate that our method preserves high-quality generation and real-time rendering, while simultaneously supporting explicit lighting and viewpoint editing. We will publicly release our code and dataset.
>
---
#### [new 078] BEDS: Bayesian Emergent Dissipative Structures
- **分类: cs.CV**

- **简介: 该论文提出BEDS框架，融合热力学与贝叶斯推理，解决学习系统可持续性问题，通过数学常数和网络架构验证理论。**

- **链接: [https://arxiv.org/pdf/2601.02329v1](https://arxiv.org/pdf/2601.02329v1)**

> **作者:** Laurent Caraffa
>
> **备注:** 19 pages
>
> **摘要:** We present BEDS (Bayesian Emergent Dissipative Structures), a theoretical framework that unifies concepts from non-equilibrium thermodynamics, Bayesian inference, information geometry, and machine learning. The central thesis proposes that learning, across physical, biological, and computational systems, fundamentally constitutes the conversion of flux into structure through entropy export. Building on Prigogine's theory of dissipative structures, we establish a formal isomorphism between thermodynamic processes and Bayesian updating, demonstrating that sustainable learning systems must follow dissipative patterns where crystallized posteriors become priors for subsequent levels of emergence. We derive fundamental mathematical constants (e, π, φ) as fixed points of Bayesian inference under minimal axioms, suggesting these constants emerge necessarily from any system capable of representing and updating uncertainty. Furthermore, we propose a conjecture linking Gödel's incompleteness theorems to thermodynamic constraints, hypothesizing that pathologies of formal systems (incompleteness, undecidability) are structurally analogous to dissipation deficits in physical systems. As practical validation, we present a peer-to-peer network architecture implementing BEDS principles, achieving six orders of magnitude improvement in energy efficiency compared to existing distributed consensus systems while enabling continuous learning. This work bridges fundamental physics, mathematical logic, and practical system design, offering both theoretical insights into the nature of learning and computation, and a concrete pathway toward sustainable artificial intelligence.
>
---
#### [new 079] Unified Review and Benchmark of Deep Segmentation Architectures for Cardiac Ultrasound on CAMUS
- **分类: cs.CV**

- **简介: 该论文属于心脏超声分割任务，旨在比较不同深度学习架构的性能。通过基准测试，评估U-Net、Attention U-Net和TransUNet在CAMUS数据集上的表现，并探索数据预处理与自监督学习的影响。**

- **链接: [https://arxiv.org/pdf/2601.00839v1](https://arxiv.org/pdf/2601.00839v1)**

> **作者:** Zahid Ullah; Muhammad Hilal; Eunsoo Lee; Dragan Pamucar; Jihie Kim
>
> **摘要:** Several review papers summarize cardiac imaging and DL advances, few works connect this overview to a unified and reproducible experimental benchmark. In this study, we combine a focused review of cardiac ultrasound segmentation literature with a controlled comparison of three influential architectures, U-Net, Attention U-Net, and TransUNet, on the Cardiac Acquisitions for Multi-Structure Ultrasound Segmentation (CAMUS) echocardiography dataset. Our benchmark spans multiple preprocessing routes, including native NIfTI volumes, 16-bit PNG exports, GPT-assisted polygon-based pseudo-labels, and self-supervised pretraining (SSL) on thousands of unlabeled cine frames. Using identical training splits, losses, and evaluation criteria, a plain U-Net achieved a 94% mean Dice when trained directly on NIfTI data (preserving native dynamic range), while the PNG-16-bit workflow reached 91% under similar conditions. Attention U-Net provided modest improvements on small or low-contrast regions, reducing boundary leakage, whereas TransUNet demonstrated the strongest generalization on challenging frames due to its ability to model global spatial context, particularly when initialized with SSL. Pseudo-labeling expanded the training set and improved robustness after confidence filtering. Overall, our contributions are threefold: a harmonized, apples-to-apples benchmark of U-Net, Attention U-Net, and TransUNet under standardized CAMUS preprocessing and evaluation; practical guidance on maintaining intensity fidelity, resolution consistency, and alignment when preparing ultrasound data; and an outlook on scalable self-supervision and emerging multimodal GPT-based annotation pipelines for rapid labeling, quality assurance, and targeted dataset curation.
>
---
#### [new 080] WildIng: A Wildlife Image Invariant Representation Model for Geographical Domain Shift
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于野生动物识别任务，解决模型在不同地理区域泛化能力差的问题。通过结合文本与图像特征，提升模型对地理域偏移的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.00993v1](https://arxiv.org/pdf/2601.00993v1)**

> **作者:** Julian D. Santamaria; Claudia Isaza; Jhony H. Giraldo
>
> **摘要:** Wildlife monitoring is crucial for studying biodiversity loss and climate change. Camera trap images provide a non-intrusive method for analyzing animal populations and identifying ecological patterns over time. However, manual analysis is time-consuming and resource-intensive. Deep learning, particularly foundation models, has been applied to automate wildlife identification, achieving strong performance when tested on data from the same geographical locations as their training sets. Yet, despite their promise, these models struggle to generalize to new geographical areas, leading to significant performance drops. For example, training an advanced vision-language model, such as CLIP with an adapter, on an African dataset achieves an accuracy of 84.77%. However, this performance drops significantly to 16.17% when the model is tested on an American dataset. This limitation partly arises because existing models rely predominantly on image-based representations, making them sensitive to geographical data distribution shifts, such as variation in background, lighting, and environmental conditions. To address this, we introduce WildIng, a Wildlife image Invariant representation model for geographical domain shift. WildIng integrates text descriptions with image features, creating a more robust representation to geographical domain shifts. By leveraging textual descriptions, our approach captures consistent semantic information, such as detailed descriptions of the appearance of the species, improving generalization across different geographical locations. Experiments show that WildIng enhances the accuracy of foundation models such as BioCLIP by 30% under geographical domain shift conditions. We evaluate WildIng on two datasets collected from different regions, namely America and Africa. The code and models are publicly available at https://github.com/Julian075/CATALOG/tree/WildIng.
>
---
#### [new 081] AI-Powered Deepfake Detection Using CNN and Vision Transformer Architectures
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于深度伪造检测任务，旨在解决AI生成虚假图像的识别问题。通过CNN和Vision Transformer模型进行检测，提升准确性和效率。**

- **链接: [https://arxiv.org/pdf/2601.01281v1](https://arxiv.org/pdf/2601.01281v1)**

> **作者:** Sifatullah Sheikh Urmi; Kirtonia Nuzath Tabassum Arthi; Md Al-Imran
>
> **备注:** 6 pages, 6 figures, 3 tables. Conference paper
>
> **摘要:** The increasing use of artificial intelligence generated deepfakes creates major challenges in maintaining digital authenticity. Four AI-based models, consisting of three CNNs and one Vision Transformer, were evaluated using large face image datasets. Data preprocessing and augmentation techniques improved model performance across different scenarios. VFDNET demonstrated superior accuracy with MobileNetV3, showing efficient performance, thereby demonstrating AI's capabilities for dependable deepfake detection.
>
---
#### [new 082] NarrativeTrack: Evaluating Video Language Models Beyond the Frame
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视频语言模型的叙事理解任务，旨在解决模型在视频中跟踪实体和理解时间连贯性的问题。工作包括构建NarrativeTrack基准和CRP评估框架，分析模型在实体持续性、变化和模糊性上的表现。**

- **链接: [https://arxiv.org/pdf/2601.01095v1](https://arxiv.org/pdf/2601.01095v1)**

> **作者:** Hyeonjeong Ha; Jinjin Ge; Bo Feng; Kaixin Ma; Gargi Chakraborty
>
> **备注:** VideoLLM Fine-Grained Evaluation
>
> **摘要:** Multimodal large language models (MLLMs) have achieved impressive progress in vision-language reasoning, yet their ability to understand temporally unfolding narratives in videos remains underexplored. True narrative understanding requires grounding who is doing what, when, and where, maintaining coherent entity representations across dynamic visual and temporal contexts. We introduce NarrativeTrack, the first benchmark to evaluate narrative understanding in MLLMs through fine-grained entity-centric reasoning. Unlike existing benchmarks limited to short clips or coarse scene-level semantics, we decompose videos into constituent entities and examine their continuity via a Compositional Reasoning Progression (CRP), a structured evaluation framework that progressively increases narrative complexity across three dimensions: entity existence, entity changes, and entity ambiguity. CRP challenges models to advance from temporal persistence to contextual evolution and fine-grained perceptual reasoning. A fully automated entity-centric pipeline enables scalable extraction of temporally grounded entity representations, providing the foundation for CRP. Evaluations of state-of-the-art MLLMs reveal that models fail to robustly track entities across visual transitions and temporal dynamics, often hallucinating identity under context shifts. Open-source general-purpose MLLMs exhibit strong perceptual grounding but weak temporal coherence, while video-specific MLLMs capture temporal context yet hallucinate entity's contexts. These findings uncover a fundamental trade-off between perceptual grounding and temporal reasoning, indicating that narrative understanding emerges only from their integration. NarrativeTrack provides the first systematic framework to diagnose and advance temporally grounded narrative comprehension in MLLMs.
>
---
#### [new 083] Prior-Guided DETR for Ultrasound Nodule Detection
- **分类: cs.CV**

- **简介: 该论文属于医学图像检测任务，旨在解决超声图像中结节检测的难题，通过引入先验知识提升检测精度。**

- **链接: [https://arxiv.org/pdf/2601.02212v1](https://arxiv.org/pdf/2601.02212v1)**

> **作者:** Jingjing Wang; Zhuo Xiao; Xinning Yao; Bo Liu; Lijuan Niu; Xiangzhi Bai; Fugen Zhou
>
> **摘要:** Accurate detection of ultrasound nodules is essential for the early diagnosis and treatment of thyroid and breast cancers. However, this task remains challenging due to irregular nodule shapes, indistinct boundaries, substantial scale variations, and the presence of speckle noise that degrades structural visibility. To address these challenges, we propose a prior-guided DETR framework specifically designed for ultrasound nodule detection. Instead of relying on purely data-driven feature learning, the proposed framework progressively incorporates different prior knowledge at multiple stages of the network. First, a Spatially-adaptive Deformable FFN with Prior Regularization (SDFPR) is embedded into the CNN backbone to inject geometric priors into deformable sampling, stabilizing feature extraction for irregular and blurred nodules. Second, a Multi-scale Spatial-Frequency Feature Mixer (MSFFM) is designed to extract multi-scale structural priors, where spatial-domain processing emphasizes contour continuity and boundary cues, while frequency-domain modeling captures global morphology and suppresses speckle noise. Furthermore, a Dense Feature Interaction (DFI) mechanism propagates and exploits these prior-modulated features across all encoder layers, enabling the decoder to enhance query refinement under consistent geometric and structural guidance. Experiments conducted on two clinically collected thyroid ultrasound datasets (Thyroid I and Thyroid II) and two public benchmarks (TN3K and BUSI) for thyroid and breast nodules demonstrate that the proposed method achieves superior accuracy compared with 18 detection methods, particularly in detecting morphologically complex nodules.The source code is publicly available at https://github.com/wjj1wjj/Ultrasound-DETR.
>
---
#### [new 084] An Empirical Study of Monocular Human Body Measurement Under Weak Calibration
- **分类: cs.CV**

- **简介: 该论文属于单目人体测量任务，解决尺度模糊和视角敏感问题。通过三种弱标定策略分析测量稳定性与用户校准的关系，为轻量级系统设计提供参考。**

- **链接: [https://arxiv.org/pdf/2601.01639v1](https://arxiv.org/pdf/2601.01639v1)**

> **作者:** Gaurav Sekar
>
> **备注:** The paper consists of 8 pages, 2 figures (on pages 4 and 7), and 2 tables (both on page 6)
>
> **摘要:** Estimating human body measurements from monocular RGB imagery remains challenging due to scale ambiguity, viewpoint sensitivity, and the absence of explicit depth information. This work presents a systematic empirical study of three weakly calibrated monocular strategies: landmark-based geometry, pose-driven regression, and object-calibrated silhouettes, evaluated under semi-constrained conditions using consumer-grade cameras. Rather than pursuing state-of-the-art accuracy, the study analyzes how differing calibration assumptions influence measurement behavior, robustness, and failure modes across varied body types. The results reveal a clear trade-off between user effort during calibration and the stability of resulting circumferential quantities. This paper serves as an empirical design reference for lightweight monocular human measurement systems intended for deployment on consumer devices.
>
---
#### [new 085] Face Normal Estimation from Rags to Riches
- **分类: cs.CV**

- **简介: 该论文属于人脸法向量估计任务，旨在减少对大规模配对数据的依赖。通过粗到细的框架和自注意力机制，提升估计质量并降低训练成本。**

- **链接: [https://arxiv.org/pdf/2601.01950v1](https://arxiv.org/pdf/2601.01950v1)**

> **作者:** Meng Wang; Wenjing Dai; Jiawan Zhang; Xiaojie Guo
>
> **摘要:** Although recent approaches to face normal estimation have achieved promising results, their effectiveness heavily depends on large-scale paired data for training. This paper concentrates on relieving this requirement via developing a coarse-to-fine normal estimator. Concretely, our method first trains a neat model from a small dataset to produce coarse face normals that perform as guidance (called exemplars) for the following refinement. A self-attention mechanism is employed to capture long-range dependencies, thus remedying severe local artifacts left in estimated coarse facial normals. Then, a refinement network is customized for the sake of mapping input face images together with corresponding exemplars to fine-grained high-quality facial normals. Such a logical function split can significantly cut the requirement of massive paired data and computational resource. Extensive experiments and ablation studies are conducted to demonstrate the efficacy of our design and reveal its superiority over state-of-the-art methods in terms of both training expense as well as estimation quality. Our code and models are open-sourced at: https://github.com/AutoHDR/FNR2R.git.
>
---
#### [new 086] FastV-RAG: Towards Fast and Fine-Grained Video QA with Retrieval-Augmented Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频问答任务，旨在提升检索增强生成的效率与准确性。通过引入推测解码和相似性过滤，提高推理速度并减少错误。**

- **链接: [https://arxiv.org/pdf/2601.01513v1](https://arxiv.org/pdf/2601.01513v1)**

> **作者:** Gen Li; Peiyu Liu
>
> **摘要:** Vision-Language Models (VLMs) excel at visual reasoning but still struggle with integrating external knowledge. Retrieval-Augmented Generation (RAG) is a promising solution, but current methods remain inefficient and often fail to maintain high answer quality. To address these challenges, we propose VideoSpeculateRAG, an efficient VLM-based RAG framework built on two key ideas. First, we introduce a speculative decoding pipeline: a lightweight draft model quickly generates multiple answer candidates, which are then verified and refined by a more accurate heavyweight model, substantially reducing inference latency without sacrificing correctness. Second, we identify a major source of error - incorrect entity recognition in retrieved knowledge - and mitigate it with a simple yet effective similarity-based filtering strategy that improves entity alignment and boosts overall answer accuracy. Experiments demonstrate that VideoSpeculateRAG achieves comparable or higher accuracy than standard RAG approaches while accelerating inference by approximately 2x. Our framework highlights the potential of combining speculative decoding with retrieval-augmented reasoning to enhance efficiency and reliability in complex, knowledge-intensive multimodal tasks.
>
---
#### [new 087] Meta-Learning Guided Pruning for Few-Shot Plant Pathology on Edge Devices
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于植物病害识别任务，解决边缘设备上模型过大和数据稀缺问题，通过结合剪枝与元学习，提出DACIS方法，显著压缩模型并保持高精度。**

- **链接: [https://arxiv.org/pdf/2601.02353v1](https://arxiv.org/pdf/2601.02353v1)**

> **作者:** Shahnawaz Alam; Mohammed Mudassir Uddin; Mohammed Kaif Pasha
>
> **摘要:** Farmers in remote areas need quick and reliable methods for identifying plant diseases, yet they often lack access to laboratories or high-performance computing resources. Deep learning models can detect diseases from leaf images with high accuracy, but these models are typically too large and computationally expensive to run on low-cost edge devices such as Raspberry Pi. Furthermore, collecting thousands of labeled disease images for training is both expensive and time-consuming. This paper addresses both challenges by combining neural network pruning -- removing unnecessary parts of the model -- with few-shot learning, which enables the model to learn from limited examples. This paper proposes Disease-Aware Channel Importance Scoring (DACIS), a method that identifies which parts of the neural network are most important for distinguishing between different plant diseases, integrated into a three-stage Prune-then-Meta-Learn-then-Prune (PMP) pipeline. Experiments on PlantVillage and PlantDoc datasets demonstrate that the proposed approach reduces model size by 78\% while maintaining 92.3\% of the original accuracy, with the compressed model running at 7 frames per second on a Raspberry Pi 4, making real-time field diagnosis practical for smallholder farmers.
>
---
#### [new 088] Real-Time Lane Detection via Efficient Feature Alignment and Covariance Optimization for Low-Power Embedded Systems
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于实时车道检测任务，针对嵌入式系统中计算资源有限的问题，提出CDO模块提升检测精度，无需增加计算复杂度。**

- **链接: [https://arxiv.org/pdf/2601.01696v1](https://arxiv.org/pdf/2601.01696v1)**

> **作者:** Yian Liu; Xiong Wang; Ping Xu; Lei Zhu; Ming Yan; Linyun Xue
>
> **摘要:** Real-time lane detection in embedded systems encounters significant challenges due to subtle and sparse visual signals in RGB images, often constrained by limited computational resources and power consumption. Although deep learning models for lane detection categorized into segmentation-based, anchor-based, and curve-based methods there remains a scarcity of universally applicable optimization techniques tailored for low-power embedded environments. To overcome this, we propose an innovative Covariance Distribution Optimization (CDO) module specifically designed for efficient, real-time applications. The CDO module aligns lane feature distributions closely with ground-truth labels, significantly enhancing detection accuracy without increasing computational complexity. Evaluations were conducted on six diverse models across all three method categories, including two optimized for real-time applications and four state-of-the-art (SOTA) models, tested comprehensively on three major datasets: CULane, TuSimple, and LLAMAS. Experimental results demonstrate accuracy improvements ranging from 0.01% to 1.5%. The proposed CDO module is characterized by ease of integration into existing systems without structural modifications and utilizes existing model parameters to facilitate ongoing training, thus offering substantial benefits in performance, power efficiency, and operational flexibility in embedded systems.
>
---
#### [new 089] A Deep Learning Approach for Automated Skin Lesion Diagnosis with Explainable AI
- **分类: cs.CV**

- **简介: 该论文属于皮肤病变分类任务，旨在提高皮肤癌诊断的准确性和透明度。通过深度学习与可解释AI技术，实现高效、可靠的多类皮肤病变自动诊断。**

- **链接: [https://arxiv.org/pdf/2601.00964v1](https://arxiv.org/pdf/2601.00964v1)**

> **作者:** Md. Maksudul Haque; Rahnuma Akter; A S M Ahsanul Sarkar Akib; Abdul Hasib
>
> **摘要:** Skin cancer is also one of the most common and dangerous types of cancer in the world that requires timely and precise diagnosis. In this paper, a deep-learning architecture of the multi-class skin lesion classification on the HAM10000 dataset will be described. The system suggested combines high-quality data balancing methods, large-scale data augmentation, hybridized EfficientNetV2-L framework with channel attention, and a three-stage progressive learning approach. Moreover, we also use explainable AI (XAI) techniques such as Grad-CAM and saliency maps to come up with intelligible visual representations of model predictions. Our strategy is with a total accuracy of 91.15 per cent, macro F1 of 85.45\% and micro-average AUC of 99.33\%. The model has shown high performance in all the seven lesion classes with specific high performance of melanoma and melanocytic nevi. In addition to enhancing diagnostic transparency, XAI also helps to find out the visual characteristics that cause the classifications, which enhances clinical trustworthiness.
>
---
#### [new 090] Beyond Patches: Global-aware Autoregressive Model for Multimodal Few-Shot Font Generation
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于少样本字体生成任务，旨在解决结构与风格一致性问题。提出GAR-Font框架，结合全局感知和多模态控制，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2601.01593v1](https://arxiv.org/pdf/2601.01593v1)**

> **作者:** Haonan Cai; Yuxuan Luo; Zhouhui Lian
>
> **备注:** 25 pages
>
> **摘要:** Manual font design is an intricate process that transforms a stylistic visual concept into a coherent glyph set. This challenge persists in automated Few-shot Font Generation (FFG), where models often struggle to preserve both the structural integrity and stylistic fidelity from limited references. While autoregressive (AR) models have demonstrated impressive generative capabilities, their application to FFG is constrained by conventional patch-level tokenization, which neglects global dependencies crucial for coherent font synthesis. Moreover, existing FFG methods remain within the image-to-image paradigm, relying solely on visual references and overlooking the role of language in conveying stylistic intent during font design. To address these limitations, we propose GAR-Font, a novel AR framework for multimodal few-shot font generation. GAR-Font introduces a global-aware tokenizer that effectively captures both local structures and global stylistic patterns, a multimodal style encoder offering flexible style control through a lightweight language-style adapter without requiring intensive multimodal pretraining, and a post-refinement pipeline that further enhances structural fidelity and style coherence. Extensive experiments show that GAR-Font outperforms existing FFG methods, excelling in maintaining global style faithfulness and achieving higher-quality results with textual stylistic guidance.
>
---
#### [new 091] RSwinV2-MD: An Enhanced Residual SwinV2 Transformer for Monkeypox Detection from Skin Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分类任务，旨在提升猴痘皮肤病变的检测与分类能力。通过改进的Residual SwinV2 Transformer模型，结合局部与全局特征，提高诊断准确性。**

- **链接: [https://arxiv.org/pdf/2601.01835v1](https://arxiv.org/pdf/2601.01835v1)**

> **作者:** Rashid Iqbal; Saddam Hussain Khan
>
> **备注:** 15 Pages, 7 Figures, 4 Tables
>
> **摘要:** In this paper, a deep learning approach for Mpox diagnosis named Customized Residual SwinTransformerV2 (RSwinV2) has been proposed, trying to enhance the capability of lesion classification by employing the RSwinV2 tool-assisted vision approach. In the RSwinV2 method, a hierarchical structure of the transformer has been customized based on the input dimensionality, embedding structure, and output targeted by the method. In this RSwinV2 approach, the input image has been split into non-overlapping patches and processed using shifted windows and attention in these patches. This process has helped the method link all the windows efficiently by avoiding the locality issues of non-overlapping regions in attention, while being computationally efficient. RSwinV2 has further developed based on SwinTransformer and has included patch and position embeddings to take advantage of the transformer global-linking capability by employing multi-head attention in these embeddings. Furthermore, RSwinV2 has developed and incorporated the Inverse Residual Block (IRB) into this method, which utilizes convolutional skip connections with these inclusive designs to address the vanishing gradient issues during processing. RSwinV2 inclusion of IRB has therefore facilitated this method to link global patterns as well as local patterns; hence, its integrity has helped improve lesion classification capability by minimizing variability of Mpox and increasing differences of Mpox, chickenpox, measles, and cowpox. In testing SwinV2, its accuracy of 96.21 and an F1score of 95.62 have been achieved on the Kaggle public dataset, which has outperformed standard CNN models and SwinTransformers; RSwinV2 vector has thus proved its valiance as a computer-assisted tool for Mpox lesion observation interpretation.
>
---
#### [new 092] DiffKD-DCIS: Predicting Upgrade of Ductal Carcinoma In Situ with Diffusion Augmentation and Knowledge Distillation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决DCIS升级预测问题。通过扩散增强和知识蒸馏方法，提升模型的泛化能力和效率。**

- **链接: [https://arxiv.org/pdf/2601.01507v1](https://arxiv.org/pdf/2601.01507v1)**

> **作者:** Tao Li; Qing Li; Na Li; Hui Xie
>
> **摘要:** Accurately predicting the upgrade of ductal carcinoma in situ (DCIS) to invasive ductal carcinoma (IDC) is crucial for surgical planning. However, traditional deep learning methods face challenges due to limited ultrasound data and poor generalization ability. This study proposes the DiffKD-DCIS framework, integrating conditional diffusion modeling with teacher-student knowledge distillation. The framework operates in three stages: First, a conditional diffusion model generates high-fidelity ultrasound images using multimodal conditions for data augmentation. Then, a deep teacher network extracts robust features from both original and synthetic data. Finally, a compact student network learns from the teacher via knowledge distillation, balancing generalization and computational efficiency. Evaluated on a multi-center dataset of 1,435 cases, the synthetic images were of good quality. The student network had fewer parameters and faster inference. On external test sets, it outperformed partial combinations, and its accuracy was comparable to senior radiologists and superior to junior ones, showing significant clinical potential.
>
---
#### [new 093] Robust Egocentric Visual Attention Prediction Through Language-guided Scene Context-aware Learning
- **分类: cs.CV**

- **简介: 该论文属于视觉注意力预测任务，旨在解决动态第一人称场景中注意力预测的挑战。通过语言引导的场景上下文学习，提升预测的准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.01818v1](https://arxiv.org/pdf/2601.01818v1)**

> **作者:** Sungjune Park; Hongda Mao; Qingshuang Chen; Yong Man Ro; Yelin Kim
>
> **备注:** 11 pages, 7 figures, 4 tables
>
> **摘要:** As the demand for analyzing egocentric videos grows, egocentric visual attention prediction, anticipating where a camera wearer will attend, has garnered increasing attention. However, it remains challenging due to the inherent complexity and ambiguity of dynamic egocentric scenes. Motivated by evidence that scene contextual information plays a crucial role in modulating human attention, in this paper, we present a language-guided scene context-aware learning framework for robust egocentric visual attention prediction. We first design a context perceiver which is guided to summarize the egocentric video based on a language-based scene description, generating context-aware video representations. We then introduce two training objectives that: 1) encourage the framework to focus on the target point-of-interest regions and 2) suppress distractions from irrelevant regions which are less likely to attract first-person attention. Extensive experiments on Ego4D and Aria Everyday Activities (AEA) datasets demonstrate the effectiveness of our approach, achieving state-of-the-art performance and enhanced robustness across diverse, dynamic egocentric scenarios.
>
---
#### [new 094] UniSH: Unifying Scene and Human Reconstruction in a Feed-Forward Pass
- **分类: cs.CV**

- **简介: 该论文提出UniSH，解决真实场景中人体与环境联合重建问题。通过统一框架和创新训练策略，提升重建精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.01222v1](https://arxiv.org/pdf/2601.01222v1)**

> **作者:** Mengfei Li; Peng Li; Zheng Zhang; Jiahao Lu; Chengfeng Zhao; Wei Xue; Qifeng Liu; Sida Peng; Wenxiao Zhang; Wenhan Luo; Yuan Liu; Yike Guo
>
> **摘要:** We present UniSH, a unified, feed-forward framework for joint metric-scale 3D scene and human reconstruction. A key challenge in this domain is the scarcity of large-scale, annotated real-world data, forcing a reliance on synthetic datasets. This reliance introduces a significant sim-to-real domain gap, leading to poor generalization, low-fidelity human geometry, and poor alignment on in-the-wild videos. To address this, we propose an innovative training paradigm that effectively leverages unlabeled in-the-wild data. Our framework bridges strong, disparate priors from scene reconstruction and HMR, and is trained with two core components: (1) a robust distillation strategy to refine human surface details by distilling high-frequency details from an expert depth model, and (2) a two-stage supervision scheme, which first learns coarse localization on synthetic data, then fine-tunes on real data by directly optimizing the geometric correspondence between the SMPL mesh and the human point cloud. This approach enables our feed-forward model to jointly recover high-fidelity scene geometry, human point clouds, camera parameters, and coherent, metric-scale SMPL bodies, all in a single forward pass. Extensive experiments demonstrate that our model achieves state-of-the-art performance on human-centric scene reconstruction and delivers highly competitive results on global human motion estimation, comparing favorably against both optimization-based frameworks and HMR-only methods. Project page: https://murphylmf.github.io/UniSH/
>
---
#### [new 095] ESGaussianFace: Emotional and Stylized Audio-Driven Facial Animation via 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于音频驱动的面部动画任务，旨在解决情感与风格融合的高质量说话头视频生成问题。提出ESGaussianFace框架，结合3D高斯点云与注意力机制，实现高效、逼真的面部表情与风格化动画。**

- **链接: [https://arxiv.org/pdf/2601.01847v1](https://arxiv.org/pdf/2601.01847v1)**

> **作者:** Chuhang Ma; Shuai Tan; Ye Pan; Jiaolong Yang; Xin Tong
>
> **备注:** 13 pages, 10 figures
>
> **摘要:** Most current audio-driven facial animation research primarily focuses on generating videos with neutral emotions. While some studies have addressed the generation of facial videos driven by emotional audio, efficiently generating high-quality talking head videos that integrate both emotional expressions and style features remains a significant challenge. In this paper, we propose ESGaussianFace, an innovative framework for emotional and stylized audio-driven facial animation. Our approach leverages 3D Gaussian Splatting to reconstruct 3D scenes and render videos, ensuring efficient generation of 3D consistent results. We propose an emotion-audio-guided spatial attention method that effectively integrates emotion features with audio content features. Through emotion-guided attention, the model is able to reconstruct facial details across different emotional states more accurately. To achieve emotional and stylized deformations of the 3D Gaussian points through emotion and style features, we introduce two 3D Gaussian deformation predictors. Futhermore, we propose a multi-stage training strategy, enabling the step-by-step learning of the character's lip movements, emotional variations, and style features. Our generated results exhibit high efficiency, high quality, and 3D consistency. Extensive experimental results demonstrate that our method outperforms existing state-of-the-art techniques in terms of lip movement accuracy, expression variation, and style feature expressiveness.
>
---
#### [new 096] QuIC: A Quantum-Inspired Interaction Classifier for Revitalizing Shallow CNNs in Fine-Grained Recognition
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于细粒度视觉分类任务，旨在解决浅层CNN在区分相似子类别时性能不足的问题。提出QuIC模块，通过模拟量子机制提升特征交互能力，显著提高模型精度。**

- **链接: [https://arxiv.org/pdf/2601.02189v1](https://arxiv.org/pdf/2601.02189v1)**

> **作者:** Cheng Ying Wu; Yen Jui Chang
>
> **摘要:** Deploying deep learning models for Fine-Grained Visual Classification (FGVC) on resource-constrained edge devices remains a significant challenge. While deep architectures achieve high accuracy on benchmarks like CUB-200-2011, their computational cost is often prohibitive. Conversely, shallow networks (e.g., AlexNet, VGG) offer efficiency but fail to distinguish visually similar sub-categories. This is because standard Global Average Pooling (GAP) heads capture only first-order statistics, missing the subtle high-order feature interactions required for FGVC. While Bilinear CNNs address this, they suffer from high feature dimensionality and instability during training. To bridge this gap, we propose the Quantum-inspired Interaction Classifier (QuIC). Drawing inspiration from quantum mechanics, QuIC models feature channels as interacting quantum states and captures second-order feature covariance via a learnable observable operator. Designed as a lightweight, plug-and-play module, QuIC supports stable, single-stage end-to-end training without exploding feature dimensions. Experimental results demonstrate that QuIC significantly revitalizes shallow backbones: it boosts the Top-1 accuracy of VGG16 by nearly 20% and outperforms state-of-the-art attention mechanisms (SE-Block) on ResNet18. Qualitative analysis, including t-SNE visualization, further confirms that QuIC resolves ambiguous cases by explicitly attending to fine-grained discriminative features and enforcing compact intra-class clustering.
>
---
#### [new 097] RRNet: Configurable Real-Time Video Enhancement with Arbitrary Local Lighting Variations
- **分类: cs.CV**

- **简介: 论文提出RRNet，解决实时视频增强中的光照不均问题。通过轻量框架实现高效局部调光，提升画质与性能。**

- **链接: [https://arxiv.org/pdf/2601.01865v1](https://arxiv.org/pdf/2601.01865v1)**

> **作者:** Wenlong Yang; Canran Jin; Weihang Yuan; Chao Wang; Lifeng Sun
>
> **摘要:** With the growing demand for real-time video enhancement in live applications, existing methods often struggle to balance speed and effective exposure control, particularly under uneven lighting. We introduce RRNet (Rendering Relighting Network), a lightweight and configurable framework that achieves a state-of-the-art tradeoff between visual quality and efficiency. By estimating parameters for a minimal set of virtual light sources, RRNet enables localized relighting through a depth-aware rendering module without requiring pixel-aligned training data. This object-aware formulation preserves facial identity and supports real-time, high-resolution performance using a streamlined encoder and lightweight prediction head. To facilitate training, we propose a generative AI-based dataset creation pipeline that synthesizes diverse lighting conditions at low cost. With its interpretable lighting control and efficient architecture, RRNet is well suited for practical applications such as video conferencing, AR-based portrait enhancement, and mobile photography. Experiments show that RRNet consistently outperforms prior methods in low-light enhancement, localized illumination adjustment, and glare removal.
>
---
#### [new 098] Adapting Depth Anything to Adverse Imaging Conditions with Events
- **分类: cs.CV**

- **简介: 该论文属于深度估计任务，旨在解决恶劣光照和运动模糊下的深度估计问题。通过融合事件相机与帧相机信息，提升模型在复杂环境中的性能。**

- **链接: [https://arxiv.org/pdf/2601.02020v1](https://arxiv.org/pdf/2601.02020v1)**

> **作者:** Shihan Peng; Yuyang Xiong; Hanyu Zhou; Zhiwei Shi; Haoyue Liu; Gang Chen; Luxin Yan; Yi Chang
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Robust depth estimation under dynamic and adverse lighting conditions is essential for robotic systems. Currently, depth foundation models, such as Depth Anything, achieve great success in ideal scenes but remain challenging under adverse imaging conditions such as extreme illumination and motion blur. These degradations corrupt the visual signals of frame cameras, weakening the discriminative features of frame-based depths across the spatial and temporal dimensions. Typically, existing approaches incorporate event cameras to leverage their high dynamic range and temporal resolution, aiming to compensate for corrupted frame features. However, such specialized fusion models are predominantly trained from scratch on domain-specific datasets, thereby failing to inherit the open-world knowledge and robust generalization inherent to foundation models. In this work, we propose ADAE, an event-guided spatiotemporal fusion framework for Depth Anything in degraded scenes. Our design is guided by two key insights: 1) Entropy-Aware Spatial Fusion. We adaptively merge frame-based and event-based features using an information entropy strategy to indicate illumination-induced degradation. 2) Motion-Guided Temporal Correction. We resort to the event-based motion cue to recalibrate ambiguous features in blurred regions. Under our unified framework, the two components are complementary to each other and jointly enhance Depth Anything under adverse imaging conditions. Extensive experiments have been performed to verify the superiority of the proposed method. Our code will be released upon acceptance.
>
---
#### [new 099] Comparative Evaluation of CNN Architectures for Neural Style Transfer in Indonesian Batik Motif Generation: A Comprehensive Study
- **分类: cs.CV**

- **简介: 该论文属于神经风格迁移任务，旨在解决资源受限环境下印尼蜡染图案生成的效率与结构保持问题。通过对比不同CNN架构，发现ResNet在速度和计算效率上优于VGG。**

- **链接: [https://arxiv.org/pdf/2601.00888v1](https://arxiv.org/pdf/2601.00888v1)**

> **作者:** Happy Gery Pangestu; Andi Prademon Yunus; Siti Khomsah
>
> **备注:** 29 pages, 9 figures, submitted in VCIBA
>
> **摘要:** Neural Style Transfer (NST) provides a computational framework for the digital preservation and generative exploration of Indonesian batik motifs; however, existing approaches remain largely centered on VGG-based architectures whose strong stylistic expressiveness comes at the cost of high computational and memory demands, that limits practical deployment in resource-limited environments. This study presents a systematic comparative analysis of five widely used CNN backbones, namely VGG16, VGG19, Inception V3, ResNet50, and ResNet101, based on 245 controlled experiments combining quantitative metrics, qualitative assessment, and statistical analysis to examine the trade-off between structural preservation, stylistic behavior, and computational efficiency. The results show that backbone selection does not yield statistically significant differences in structural similarity, as confirmed by ANOVA on SSIM (p= 0.83), indicating comparable levels of structural preservation rather than equivalent stylistic quality. Within this context, ResNet-based architectures achieve approximately 5-6x faster convergence than VGG models while maintaining similar perceptual similarity (LPIPS = 0.53) and requiring over 16x fewer FLOPs (0.63 vs 10.12 GFLOPs). Qualitative analysis reveals consistent stylistic trade-offs, with VGG producing denser painterly textures, ResNet favoring geometric stability and canting stroke preservation with milder stylization, and Inception V3 exhibiting intermediate but noisier behavior. These findings reposition architectural choice in NST from maximizing stylistic intensity toward efficiency-aware and structure-preserving deployment, highlighting ResNet-based backbones as a practical foundation for scalable, industry-oriented batik generation.
>
---
#### [new 100] S2M-Net: Spectral-Spatial Mixing for Medical Image Segmentation with Morphology-Aware Adaptive Loss
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，解决精度、上下文与效率的平衡问题。提出S2M-Net，结合频谱-空间混合与形态感知损失，提升分割性能并减少参数量。**

- **链接: [https://arxiv.org/pdf/2601.01285v1](https://arxiv.org/pdf/2601.01285v1)**

> **作者:** Md. Sanaullah Chowdhury Lameya Sabrin
>
> **摘要:** Medical image segmentation requires balancing local precision for boundary-critical clinical applications, global context for anatomical coherence, and computational efficiency for deployment on limited data and hardware a trilemma that existing architectures fail to resolve. Although convolutional networks provide local precision at $\mathcal{O}(n)$ cost but limited receptive fields, vision transformers achieve global context through $\mathcal{O}(n^2)$ self-attention at prohibitive computational expense, causing overfitting on small clinical datasets. We propose S2M-Net, a 4.7M-parameter architecture that achieves $\mathcal{O}(HW \log HW)$ global context through two synergistic innovations: (i) Spectral-Selective Token Mixer (SSTM), which exploits the spectral concentration of medical images via truncated 2D FFT with learnable frequency filtering and content-gated spatial projection, avoiding quadratic attention cost while maintaining global receptive fields; and (ii) Morphology-Aware Adaptive Segmentation Loss (MASL), which automatically analyzes structure characteristics (compactness, tubularity, irregularity, scale) to modulate five complementary loss components through constrained learnable weights, eliminating manual per-dataset tuning. Comprehensive evaluation in 16 medical imaging datasets that span 8 modalities demonstrates state-of-the-art performance: 96.12\% Dice on polyp segmentation, 83.77\% on surgical instruments (+17.85\% over the prior art) and 80.90\% on brain tumors, with consistent 3-18\% improvements over specialized baselines while using 3.5--6$\times$ fewer parameters than transformer-based methods.
>
---
#### [new 101] Language as Prior, Vision as Calibration: Metric Scale Recovery for Monocular Depth Estimation
- **分类: cs.CV**

- **简介: 该论文属于单目深度估计任务，解决度量尺度恢复问题。通过语言预测不确定性范围，结合视觉特征进行校准，提升模型精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.01457v1](https://arxiv.org/pdf/2601.01457v1)**

> **作者:** Mingxing Zhan; Li Zhang; Beibei Wang; Yingjie Wang; Zenglin Shi
>
> **摘要:** Relative-depth foundation models transfer well, yet monocular metric depth remains ill-posed due to unidentifiable global scale and heightened domain-shift sensitivity. Under a frozen-backbone calibration setting, we recover metric depth via an image-specific affine transform in inverse depth and train only lightweight calibration heads while keeping the relative-depth backbone and the CLIP text encoder fixed. Since captions provide coarse but noisy scale cues that vary with phrasing and missing objects, we use language to predict an uncertainty-aware envelope that bounds feasible calibration parameters in an unconstrained space, rather than committing to a text-only point estimate. We then use pooled multi-scale frozen visual features to select an image-specific calibration within this envelope. During training, a closed-form least-squares oracle in inverse depth provides per-image supervision for learning the envelope and the selected calibration. Experiments on NYUv2 and KITTI improve in-domain accuracy, while zero-shot transfer to SUN-RGBD and DDAD demonstrates improved robustness over strong language-only baselines.
>
---
#### [new 102] MANGO:Natural Multi-speaker 3D Talking Head Generation via 2D-Lifted Enhancement
- **分类: cs.CV**

- **简介: 该论文属于3D对话头生成任务，解决多说话人自然交互与伪3D标签噪声问题。提出MANGO框架，通过两阶段训练提升对话行为的逼真度与准确性。**

- **链接: [https://arxiv.org/pdf/2601.01749v1](https://arxiv.org/pdf/2601.01749v1)**

> **作者:** Lei Zhu; Lijian Lin; Ye Zhu; Jiahao Wu; Xuehan Hou; Yu Li; Yunfei Liu; Jie Chen
>
> **备注:** 20 pages, 11i figures
>
> **摘要:** Current audio-driven 3D head generation methods mainly focus on single-speaker scenarios, lacking natural, bidirectional listen-and-speak interaction. Achieving seamless conversational behavior, where speaking and listening states transition fluidly remains a key challenge. Existing 3D conversational avatar approaches rely on error-prone pseudo-3D labels that fail to capture fine-grained facial dynamics. To address these limitations, we introduce a novel two-stage framework MANGO, which leveraging pure image-level supervision by alternately training to mitigate the noise introduced by pseudo-3D labels, thereby achieving better alignment with real-world conversational behaviors. Specifically, in the first stage, a diffusion-based transformer with a dual-audio interaction module models natural 3D motion from multi-speaker audio. In the second stage, we use a fast 3D Gaussian Renderer to generate high-fidelity images and provide 2D-level photometric supervision for the 3D motions through alternate training. Additionally, we introduce MANGO-Dialog, a high-quality dataset with over 50 hours of aligned 2D-3D conversational data across 500+ identities. Extensive experiments demonstrate that our method achieves exceptional accuracy and realism in modeling two-person 3D dialogue motion, significantly advancing the fidelity and controllability of audio-driven talking heads.
>
---
#### [new 103] FFP-300K: Scaling First-Frame Propagation for Generalizable Video Editing
- **分类: cs.CV**

- **简介: 该论文属于视频编辑任务，解决现有方法依赖运行时引导的问题。通过构建大规模数据集FFP-300K和提出新框架，实现无需引导的首次帧传播。**

- **链接: [https://arxiv.org/pdf/2601.01720v1](https://arxiv.org/pdf/2601.01720v1)**

> **作者:** Xijie Huang; Chengming Xu; Donghao Luo; Xiaobin Hu; Peng Tang; Xu Peng; Jiangning Zhang; Chengjie Wang; Yanwei Fu
>
> **摘要:** First-Frame Propagation (FFP) offers a promising paradigm for controllable video editing, but existing methods are hampered by a reliance on cumbersome run-time guidance. We identify the root cause of this limitation as the inadequacy of current training datasets, which are often too short, low-resolution, and lack the task diversity required to teach robust temporal priors. To address this foundational data gap, we first introduce FFP-300K, a new large-scale dataset comprising 300K high-fidelity video pairs at 720p resolution and 81 frames in length, constructed via a principled two-track pipeline for diverse local and global edits. Building on this dataset, we propose a novel framework designed for true guidance-free FFP that resolves the critical tension between maintaining first-frame appearance and preserving source video motion. Architecturally, we introduce Adaptive Spatio-Temporal RoPE (AST-RoPE), which dynamically remaps positional encodings to disentangle appearance and motion references. At the objective level, we employ a self-distillation strategy where an identity propagation task acts as a powerful regularizer, ensuring long-term temporal stability and preventing semantic drift. Comprehensive experiments on the EditVerseBench benchmark demonstrate that our method significantly outperforming existing academic and commercial models by receiving about 0.2 PickScore and 0.3 VLM score improvement against these competitors.
>
---
#### [new 104] VIBE: Visual Instruction Based Editor
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出VIBE，一种基于视觉指令的图像编辑系统，解决高效、高质量图像编辑问题。使用轻量模型实现低开销推理，支持多种编辑任务。**

- **链接: [https://arxiv.org/pdf/2601.02242v1](https://arxiv.org/pdf/2601.02242v1)**

> **作者:** Grigorii Alekseenko; Aleksandr Gordeev; Irina Tolstykh; Bulat Suleimanov; Vladimir Dokholyan; Georgii Fedorov; Sergey Yakubson; Aleksandra Tsybina; Mikhail Chernyshov; Maksim Kuprashevich
>
> **摘要:** Instruction-based image editing is among the fastest developing areas in generative AI. Over the past year, the field has reached a new level, with dozens of open-source models released alongside highly capable commercial systems. However, only a limited number of open-source approaches currently achieve real-world quality. In addition, diffusion backbones, the dominant choice for these pipelines, are often large and computationally expensive for many deployments and research settings, with widely used variants typically containing 6B to 20B parameters. This paper presents a compact, high-throughput instruction-based image editing pipeline that uses a modern 2B-parameter Qwen3-VL model to guide the editing process and the 1.6B-parameter diffusion model Sana1.5 for image generation. Our design decisions across architecture, data processing, training configuration, and evaluation target low-cost inference and strict source consistency while maintaining high quality across the major edit categories feasible at this scale. Evaluated on the ImgEdit and GEdit benchmarks, the proposed method matches or exceeds the performance of substantially heavier baselines, including models with several times as many parameters and higher inference cost, and is particularly strong on edits that require preserving the input image, such as an attribute adjustment, object removal, background edits, and targeted replacement. The model fits within 24 GB of GPU memory and generates edited images at up to 2K resolution in approximately 4 seconds on an NVIDIA H100 in BF16, without additional inference optimizations or distillation.
>
---
#### [new 105] Improving Flexible Image Tokenizers for Autoregressive Image Generation
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决灵活分词器信息集中于前部 tokens 的问题。提出 ReToK 方法，通过冗余填充和语义正则化提升生成效果。**

- **链接: [https://arxiv.org/pdf/2601.01535v1](https://arxiv.org/pdf/2601.01535v1)**

> **作者:** Zixuan Fu; Lanqing Guo; Chong Wang; Binbin Song; Ding Liu; Bihan Wen
>
> **摘要:** Flexible image tokenizers aim to represent an image using an ordered 1D variable-length token sequence. This flexible tokenization is typically achieved through nested dropout, where a portion of trailing tokens is randomly truncated during training, and the image is reconstructed using the remaining preceding sequence. However, this tail-truncation strategy inherently concentrates the image information in the early tokens, limiting the effectiveness of downstream AutoRegressive (AR) image generation as the token length increases. To overcome these limitations, we propose \textbf{ReToK}, a flexible tokenizer with \underline{Re}dundant \underline{Tok}en Padding and Hierarchical Semantic Regularization, designed to fully exploit all tokens for enhanced latent modeling. Specifically, we introduce \textbf{Redundant Token Padding} to activate tail tokens more frequently, thereby alleviating information over-concentration in the early tokens. In addition, we apply \textbf{Hierarchical Semantic Regularization} to align the decoding features of earlier tokens with those from a pre-trained vision foundation model, while progressively reducing the regularization strength toward the tail to allow finer low-level detail reconstruction. Extensive experiments demonstrate the effectiveness of ReTok: on ImageNet 256$\times$256, our method achieves superior generation performance compared with both flexible and fixed-length tokenizers. Code will be available at: \href{https://github.com/zfu006/ReTok}{https://github.com/zfu006/ReTok}
>
---
#### [new 106] A UAV-Based Multispectral and RGB Dataset for Multi-Stage Paddy Crop Monitoring in Indian Agricultural Fields
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于农业监测任务，旨在解决水稻生长阶段的精准监测问题。通过无人机采集多光谱和RGB图像，构建高分辨率数据集，支持病害分析与产量预测。**

- **链接: [https://arxiv.org/pdf/2601.01084v1](https://arxiv.org/pdf/2601.01084v1)**

> **作者:** Adari Rama Sukanya; Puvvula Roopesh Naga Sri Sai; Kota Moses; Rimalapudi Sarvendranath
>
> **备注:** 10-page dataset explanation paper
>
> **摘要:** We present a large-scale unmanned aerial vehicle (UAV)-based RGB and multispectral image dataset collected over paddy fields in the Vijayawada region, Andhra Pradesh, India, covering nursery to harvesting stages. We used a 20-megapixel RGB camera and a 5-megapixel four-band multispectral camera capturing red, green, red-edge, and near-infrared bands. Standardised operating procedure (SOP) and checklists were developed to ensure repeatable data acquisition. Our dataset comprises of 42,430 raw images (415 GB) captured over 5 acres with 1 cm/pixel ground sampling distance (GSD) with associated metadata such as GPS coordinates, flight altitude, and environmental conditions. Captured images were validated using Pix4D Fields to generate orthomosaic maps and vegetation index maps, such as normalised difference vegetation index (NDVI) and normalised difference red-edge (NDRE) index. Our dataset is one of the few datasets that provide high-resolution images with rich metadata that cover all growth stages of Indian paddy crops. The dataset is available on IEEE DataPort with DOI, . It can support studies on targeted spraying, disease analysis, and yield estimation.
>
---
#### [new 107] Remote Sensing Change Detection via Weak Temporal Supervision
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于遥感变化检测任务，旨在解决标注数据稀缺的问题。通过弱时间监督策略，利用现有数据生成变化样本，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.02126v1](https://arxiv.org/pdf/2601.02126v1)**

> **作者:** Xavier Bou; Elliot Vincent; Gabriele Facciolo; Rafael Grompone von Gioi; Jean-Michel Morel; Thibaud Ehret
>
> **摘要:** Semantic change detection in remote sensing aims to identify land cover changes between bi-temporal image pairs. Progress in this area has been limited by the scarcity of annotated datasets, as pixel-level annotation is costly and time-consuming. To address this, recent methods leverage synthetic data or generate artificial change pairs, but out-of-domain generalization remains limited. In this work, we introduce a weak temporal supervision strategy that leverages additional temporal observations of existing single-temporal datasets, without requiring any new annotations. Specifically, we extend single-date remote sensing datasets with new observations acquired at different times and train a change detection model by assuming that real bi-temporal pairs mostly contain no change, while pairing images from different locations to generate change examples. To handle the inherent noise in these weak labels, we employ an object-aware change map generation and an iterative refinement process. We validate our approach on extended versions of the FLAIR and IAILD aerial datasets, achieving strong zero-shot and low-data regime performance across different benchmarks. Lastly, we showcase results over large areas in France, highlighting the scalability potential of our method.
>
---
#### [new 108] Causality-Aware Temporal Projection for Video Understanding in Video-LLMs
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决Video-LLMs中时间顺序和因果关系不明确的问题。提出V-CORE框架，通过显式时间约束提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.01804v1](https://arxiv.org/pdf/2601.01804v1)**

> **作者:** Zhengjian Kang; Qi Chen; Rui Liu; Kangtong Mo; Xingyu Zhang; Xiaoyu Deng; Ye Zhang
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Recent Video Large Language Models (Video-LLMs) have shown strong multimodal reasoning capabilities, yet remain challenged by video understanding tasks that require consistent temporal ordering and causal coherence. Many parameter-efficient Video-LLMs rely on unconstrained bidirectional projectors to model inter-frame interactions, which can blur temporal ordering by allowing later frames to influence earlier representations, without explicit architectural mechanisms to respect the directional nature of video reasoning. To address this limitation, we propose V-CORE, a parameter-efficient framework that introduces explicit temporal ordering constraints for video understanding. V-CORE consists of two key components: (1) Learnable Spatial Aggregation (LSA), which adaptively selects salient spatial tokens to reduce redundancy, and (2) a Causality-Aware Temporal Projector (CATP), which enforces structured unidirectional information flow via block-causal attention and a terminal dynamic summary token acting as a causal sink. This design preserves intra-frame spatial interactions while ensuring that temporal information is aggregated in a strictly ordered manner. With 4-bit QLoRA and a frozen LLM backbone, V-CORE can be trained efficiently on a single consumer GPU. Experiments show that V-CORE achieves strong performance on the challenging NExT-QA benchmark, reaching 61.2% accuracy, and remains competitive across MSVD-QA, MSRVTT-QA, and TGIF-QA, with gains concentrated in temporal and causal reasoning subcategories (+3.5% and +5.2% respectively), directly validating the importance of explicit temporal ordering constraints.
>
---
#### [new 109] Mind the Gap: Continuous Magnification Sampling for Pathology Foundation Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于病理学图像分析任务，解决基础模型在不同放大倍数下的性能问题。通过连续采样优化放大覆盖，提升中间倍数下的分类准确率。**

- **链接: [https://arxiv.org/pdf/2601.02198v1](https://arxiv.org/pdf/2601.02198v1)**

> **作者:** Alexander Möllers; Julius Hense; Florian Schulz; Timo Milbich; Maximilian Alber; Lukas Ruff
>
> **摘要:** In histopathology, pathologists examine both tissue architecture at low magnification and fine-grained morphology at high magnification. Yet, the performance of pathology foundation models across magnifications and the effect of magnification sampling during training remain poorly understood. We model magnification sampling as a multi-source domain adaptation problem and develop a simple theoretical framework that reveals systematic trade-offs between sampling strategies. We show that the widely used discrete uniform sampling of magnifications (0.25, 0.5, 1.0, 2.0 mpp) leads to degradation at intermediate magnifications. We introduce continuous magnification sampling, which removes gaps in magnification coverage while preserving performance at standard scales. Further, we derive sampling distributions that optimize representation quality across magnification scales. To evaluate these strategies, we introduce two new benchmarks (TCGA-MS, BRACS-MS) with appropriate metrics. Our experiments show that continuous sampling substantially improves over discrete sampling at intermediate magnifications, with gains of up to 4 percentage points in balanced classification accuracy, and that optimized distributions can further improve performance. Finally, we evaluate current histopathology foundation models, finding that magnification is a primary driver of performance variation across models. Our work paves the way towards future pathology foundation models that perform reliably across magnifications.
>
---
#### [new 110] Joint Semantic and Rendering Enhancements in 3D Gaussian Modeling with Anisotropic Local Encoding
- **分类: cs.CV**

- **简介: 该论文属于3D场景理解任务，旨在解决语义分割与渲染效率问题。通过联合优化语义与渲染分支，引入各向异性特征描述符和局部语义调整策略，提升分割精度与渲染质量。**

- **链接: [https://arxiv.org/pdf/2601.02339v1](https://arxiv.org/pdf/2601.02339v1)**

> **作者:** Jingming He; Chongyi Li; Shiqi Wang; Sam Kwong
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Recent works propose extending 3DGS with semantic feature vectors for simultaneous semantic segmentation and image rendering. However, these methods often treat the semantic and rendering branches separately, relying solely on 2D supervision while ignoring the 3D Gaussian geometry. Moreover, current adaptive strategies adapt the Gaussian set depending solely on rendering gradients, which can be insufficient in subtle or textureless regions. In this work, we propose a joint enhancement framework for 3D semantic Gaussian modeling that synergizes both semantic and rendering branches. Firstly, unlike conventional point cloud shape encoding, we introduce an anisotropic 3D Gaussian Chebyshev descriptor using the Laplace-Beltrami operator to capture fine-grained 3D shape details, thereby distinguishing objects with similar appearances and reducing reliance on potentially noisy 2D guidance. In addition, without relying solely on rendering gradient, we adaptively adjust Gaussian allocation and spherical harmonics with local semantic and shape signals, enhancing rendering efficiency through selective resource allocation. Finally, we employ a cross-scene knowledge transfer module to continuously update learned shape patterns, enabling faster convergence and robust representations without relearning shape information from scratch for each new scene. Experiments on multiple datasets demonstrate improvements in segmentation accuracy and rendering quality while maintaining high rendering frame rates.
>
---
#### [new 111] BiPrompt: Bilateral Prompt Optimization for Visual and Textual Debiasing in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉-语言模型的去偏任务，旨在解决模型对非因果特征的依赖问题。通过双模态提示优化，提升模型的因果推理能力与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.02147v1](https://arxiv.org/pdf/2601.02147v1)**

> **作者:** Sunny Gupta; Shounak Das; Amit Sethi
>
> **备注:** Accepted at the AAAI 2026 Workshop AIR-FM, Assessing and Improving Reliability of Foundation Models in the Real World
>
> **摘要:** Vision language foundation models such as CLIP exhibit impressive zero-shot generalization yet remain vulnerable to spurious correlations across visual and textual modalities. Existing debiasing approaches often address a single modality either visual or textual leading to partial robustness and unstable adaptation under distribution shifts. We propose a bilateral prompt optimization framework (BiPrompt) that simultaneously mitigates non-causal feature reliance in both modalities during test-time adaptation. On the visual side, it employs structured attention-guided erasure to suppress background activations and enforce orthogonal prediction consistency between causal and spurious regions. On the textual side, it introduces balanced prompt normalization, a learnable re-centering mechanism that aligns class embeddings toward an isotropic semantic space. Together, these modules jointly minimize conditional mutual information between spurious cues and predictions, steering the model toward causal, domain invariant reasoning without retraining or domain supervision. Extensive evaluations on real-world and synthetic bias benchmarks demonstrate consistent improvements in both average and worst-group accuracies over prior test-time debiasing methods, establishing a lightweight yet effective path toward trustworthy and causally grounded vision-language adaptation.
>
---
#### [new 112] GCR: Geometry-Consistent Routing for Task-Agnostic Continual Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于异常检测任务，解决持续类别扩展下的任务无关异常检测问题。提出GCR框架，通过几何一致路由提升路由稳定性，避免跨头评分比较问题。**

- **链接: [https://arxiv.org/pdf/2601.01856v1](https://arxiv.org/pdf/2601.01856v1)**

> **作者:** Joongwon Chae; Lihui Luo; Yang Liu; Runming Wang; Dongmei Yu; Zeming Liang; Xi Yuan; Dayan Zhang; Zhenglin Chen; Peiwu Qin; Ilmoon Chae
>
> **摘要:** Feature-based anomaly detection is widely adopted in industrial inspection due to the strong representational power of large pre-trained vision encoders. While most existing methods focus on improving within-category anomaly scoring, practical deployments increasingly require task-agnostic operation under continual category expansion, where the category identity is unknown at test time. In this setting, overall performance is often dominated by expert selection, namely routing an input to an appropriate normality model before any head-specific scoring is applied. However, routing rules that compare head-specific anomaly scores across independently constructed heads are unreliable in practice, as score distributions can differ substantially across categories in scale and tail behavior. We propose GCR, a lightweight mixture-of-experts framework for stabilizing task-agnostic continual anomaly detection through geometry-consistent routing. GCR routes each test image directly in a shared frozen patch-embedding space by minimizing an accumulated nearest-prototype distance to category-specific prototype banks, and then computes anomaly maps only within the routed expert using a standard prototype-based scoring rule. By separating cross-head decision making from within-head anomaly scoring, GCR avoids cross-head score comparability issues without requiring end-to-end representation learning. Experiments on MVTec AD and VisA show that geometry-consistent routing substantially improves routing stability and mitigates continual performance collapse, achieving near-zero forgetting while maintaining competitive detection and localization performance. These results indicate that many failures previously attributed to representation forgetting can instead be explained by decision-rule instability in cross-head routing. Code is available at https://github.com/jw-chae/GCR
>
---
#### [new 113] MS-ISSM: Objective Quality Assessment of Point Clouds Using Multi-scale Implicit Structural Similarity
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于点云质量评估任务，旨在解决点云结构不规则导致的特征匹配难题。提出MS-ISSM方法，利用RBF和ResGrouped-MLP网络提升评估准确性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.01200v1](https://arxiv.org/pdf/2601.01200v1)**

> **作者:** Zhang Chen; Shuai Wan; Yuezhe Zhang; Siyu Ren; Fuzheng Yang; Junhui Hou
>
> **摘要:** The unstructured and irregular nature of point clouds poses a significant challenge for objective quality assessment (PCQA), particularly in establishing accurate perceptual feature correspondence. To tackle this, we propose the Multi-scale Implicit Structural Similarity Measurement (MS-ISSM). Unlike traditional point-to-point matching, MS-ISSM utilizes Radial Basis Functions (RBF) to represent local features continuously, transforming distortion measurement into a comparison of implicit function coefficients. This approach effectively circumvents matching errors inherent in irregular data. Additionally, we propose a ResGrouped-MLP quality assessment network, which robustly maps multi-scale feature differences to perceptual scores. The network architecture departs from traditional flat MLPs by adopting a grouped encoding strategy integrated with Residual Blocks and Channel-wise Attention mechanisms. This hierarchical design allows the model to preserve the distinct physical semantics of luma, chroma, and geometry while adaptively focusing on the most salient distortion features across High, Medium, and Low scales. Experimental results on multiple benchmarks demonstrate that MS-ISSM outperforms state-of-the-art metrics in both reliability and generalization. The source code is available at: https://github.com/ZhangChen2022/MS-ISSM.
>
---
#### [new 114] LabelAny3D: Label Any Object 3D in the Wild
- **分类: cs.CV**

- **简介: 该论文属于单目3D目标检测任务，解决真实场景中3D标注数据不足的问题。提出LabelAny3D框架，生成高质量3D标注，并构建COCO3D基准。**

- **链接: [https://arxiv.org/pdf/2601.01676v1](https://arxiv.org/pdf/2601.01676v1)**

> **作者:** Jin Yao; Radowan Mahmud Redoy; Sebastian Elbaum; Matthew B. Dwyer; Zezhou Cheng
>
> **备注:** NeurIPS 2025. Project page: https://uva-computer-vision-lab.github.io/LabelAny3D/
>
> **摘要:** Detecting objects in 3D space from monocular input is crucial for applications ranging from robotics to scene understanding. Despite advanced performance in the indoor and autonomous driving domains, existing monocular 3D detection models struggle with in-the-wild images due to the lack of 3D in-the-wild datasets and the challenges of 3D annotation. We introduce LabelAny3D, an \emph{analysis-by-synthesis} framework that reconstructs holistic 3D scenes from 2D images to efficiently produce high-quality 3D bounding box annotations. Built on this pipeline, we present COCO3D, a new benchmark for open-vocabulary monocular 3D detection, derived from the MS-COCO dataset and covering a wide range of object categories absent from existing 3D datasets. Experiments show that annotations generated by LabelAny3D improve monocular 3D detection performance across multiple benchmarks, outperforming prior auto-labeling approaches in quality. These results demonstrate the promise of foundation-model-driven annotation for scaling up 3D recognition in realistic, open-world settings.
>
---
#### [new 115] Promptable Foundation Models for SAR Remote Sensing: Adapting the Segment Anything Model for Snow Avalanche Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于遥感图像分割任务，旨在解决雪崩区域自动标注难题。通过改进Segment Anything Model，适应SAR图像特点，提升标注效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.01213v1](https://arxiv.org/pdf/2601.01213v1)**

> **作者:** Riccardo Gelato; Carlo Sgaravatti; Jakob Grahn; Giacomo Boracchi; Filippo Maria Bianchi
>
> **摘要:** Remote sensing solutions for avalanche segmentation and mapping are key to supporting risk forecasting and mitigation in mountain regions. Synthetic Aperture Radar (SAR) imagery from Sentinel-1 can be effectively used for this task, but training an effective detection model requires gathering a large dataset with high-quality annotations from domain experts, which is prohibitively time-consuming. In this work, we aim to facilitate and accelerate the annotation of SAR images for avalanche mapping. We build on the Segment Anything Model (SAM), a segmentation foundation model trained on natural images, and tailor it to Sentinel-1 SAR data. Adapting SAM to our use-case requires addressing several domain-specific challenges: (i) domain mismatch, since SAM was not trained on satellite/SAR imagery; (ii) input adaptation, because SAR products typically provide more than three channels, while SAM is constrained to RGB images; (iii) robustness to imprecise prompts that can affect target identification and degrade the segmentation quality, an issue exacerbated in small, low-contrast avalanches; and (iv) training efficiency, since standard fine-tuning is computationally demanding for SAM. We tackle these challenges through a combination of adapters to mitigate the domain gap, multiple encoders to handle multi-channel SAR inputs, prompt-engineering strategies to improve avalanche localization accuracy, and a training algorithm that limits the training time of the encoder, which is recognized as the major bottleneck. We integrate the resulting model into an annotation tool and show experimentally that it speeds up the annotation of SAR images.
>
---
#### [new 116] MacVQA: Adaptive Memory Allocation and Global Noise Filtering for Continual Visual Question Answering
- **分类: cs.CV**

- **简介: 该论文属于视觉问答任务，旨在解决持续学习中的知识保留与适应问题。提出MacVQA框架，通过自适应记忆分配和全局噪声过滤，提升模型性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2601.01926v1](https://arxiv.org/pdf/2601.01926v1)**

> **作者:** Zhifei Li; Yiran Wang; Chenyi Xiong; Yujing Xia; Xiaoju Hou; Yue Zhao; Miao Zhang; Kui Xiao; Bing Yang
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Visual Question Answering (VQA) requires models to reason over multimodal information, combining visual and textual data. With the development of continual learning, significant progress has been made in retaining knowledge and adapting to new information in the VQA domain. However, current methods often struggle with balancing knowledge retention, adaptation, and robust feature representation. To address these challenges, we propose a novel framework with adaptive memory allocation and global noise filtering called MacVQA for visual question answering. MacVQA fuses visual and question information while filtering noise to ensure robust representations, and employs prototype-based memory allocation to optimize feature quality and memory usage. These designs enable MacVQA to balance knowledge acquisition, retention, and compositional generalization in continual VQA learning. Experiments on ten continual VQA tasks show that MacVQA outperforms existing baselines, achieving 43.38% average accuracy and 2.32% average forgetting on standard tasks, and 42.53% average accuracy and 3.60% average forgetting on novel composition tasks.
>
---
#### [new 117] Advanced Machine Learning Approaches for Enhancing Person Re-Identification Performance
- **分类: cs.CV**

- **简介: 该论文属于人物重识别任务，旨在解决外观变化、领域迁移和数据不足等问题。提出了三种方法，在监督、无监督域适应和完全无监督设置下提升性能。**

- **链接: [https://arxiv.org/pdf/2601.01356v1](https://arxiv.org/pdf/2601.01356v1)**

> **作者:** Dang H. Pham; Tu N. Nguyen; Hoa N. Nguyen
>
> **备注:** in Vietnamese language
>
> **摘要:** Person re-identification (ReID) plays a critical role in intelligent surveillance systems by linking identities across multiple cameras in complex environments. However, ReID faces significant challenges such as appearance variations, domain shifts, and limited labeled data. This dissertation proposes three advanced approaches to enhance ReID performance under supervised, unsupervised domain adaptation (UDA), and fully unsupervised settings. First, SCM-ReID integrates supervised contrastive learning with hybrid loss optimization (classification, center, triplet, and centroid-triplet losses), improving discriminative feature representation and achieving state-of-the-art accuracy on Market-1501 and CUHK03 datasets. Second, for UDA, IQAGA and DAPRH combine GAN-based image augmentation, domain-invariant mapping, and pseudo-label refinement to mitigate domain discrepancies and enhance cross-domain generalization. Experiments demonstrate substantial gains over baseline methods, with mAP and Rank-1 improvements up to 12% in challenging transfer scenarios. Finally, ViTC-UReID leverages Vision Transformer-based feature encoding and camera-aware proxy learning to boost unsupervised ReID. By integrating global and local attention with camera identity constraints, this method significantly outperforms existing unsupervised approaches on large-scale benchmarks. Comprehensive evaluations across CUHK03, Market-1501, DukeMTMC-reID, and MSMT17 confirm the effectiveness of the proposed methods. The contributions advance ReID research by addressing key limitations in feature learning, domain adaptation, and label noise handling, paving the way for robust deployment in real-world surveillance systems.
>
---
#### [new 118] Evaluation of Convolutional Neural Network For Image Classification with Agricultural and Urban Datasets
- **分类: cs.CV**

- **简介: 论文提出一种定制卷积神经网络，用于农业和城市图像分类任务，解决多领域图像分类问题，通过优化架构设计提升性能与效率。**

- **链接: [https://arxiv.org/pdf/2601.01393v1](https://arxiv.org/pdf/2601.01393v1)**

> **作者:** Shamik Shafkat Avro; Nazira Jesmin Lina; Shahanaz Sharmin
>
> **备注:** All authors contributed equally to this work
>
> **摘要:** This paper presents the development and evaluation of a custom Convolutional Neural Network (CustomCNN) created to study how architectural design choices affect multi-domain image classification tasks. The network uses residual connections, Squeeze-and-Excitation attention mechanisms, progressive channel scaling, and Kaiming initialization to improve its ability to represent data and speed up training. The model is trained and tested on five publicly available datasets: unauthorized vehicle detection, footpath encroachment detection, polygon-annotated road damage and manhole detection, MangoImageBD and PaddyVarietyBD. A comparison with popular CNN architectures shows that the CustomCNN delivers competitive performance while remaining efficient in computation. The results underscore the importance of thoughtful architectural design for real-world Smart City and agricultural imaging applications.
>
---
#### [new 119] SwinIFS: Landmark Guided Swin Transformer For Identity Preserving Face Super Resolution
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于人脸超分辨率任务，旨在解决低分辨率人脸图像恢复中细节丢失和身份特征保持的问题。提出SwinIFS框架，结合关键点引导与Transformer结构，提升重建质量与身份一致性。**

- **链接: [https://arxiv.org/pdf/2601.01406v1](https://arxiv.org/pdf/2601.01406v1)**

> **作者:** Habiba Kausar; Saeed Anwar; Omar Jamal Hammad; Abdul Bais
>
> **摘要:** Face super-resolution aims to recover high-quality facial images from severely degraded low-resolution inputs, but remains challenging due to the loss of fine structural details and identity-specific features. This work introduces SwinIFS, a landmark-guided super-resolution framework that integrates structural priors with hierarchical attention mechanisms to achieve identity-preserving reconstruction at both moderate and extreme upscaling factors. The method incorporates dense Gaussian heatmaps of key facial landmarks into the input representation, enabling the network to focus on semantically important facial regions from the earliest stages of processing. A compact Swin Transformer backbone is employed to capture long-range contextual information while preserving local geometry, allowing the model to restore subtle facial textures and maintain global structural consistency. Extensive experiments on the CelebA benchmark demonstrate that SwinIFS achieves superior perceptual quality, sharper reconstructions, and improved identity retention; it consistently produces more photorealistic results and exhibits strong performance even under 8x magnification, where most methods fail to recover meaningful structure. SwinIFS also provides an advantageous balance between reconstruction accuracy and computational efficiency, making it suitable for real-world applications in facial enhancement, surveillance, and digital restoration. Our code, model weights, and results are available at https://github.com/Habiba123-stack/SwinIFS.
>
---
#### [new 120] A Novel Deep Learning Method for Segmenting the Left Ventricle in Cardiac Cine MRI
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分割任务，旨在提高心脏MRI中左心室的分割精度。通过改进的GBU-Net模型，提升上下文理解与分割效果。**

- **链接: [https://arxiv.org/pdf/2601.01512v1](https://arxiv.org/pdf/2601.01512v1)**

> **作者:** Wenhui Chu; Aobo Jin; Hardik A. Gohel
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** This research aims to develop a novel deep learning network, GBU-Net, utilizing a group-batch-normalized U-Net framework, specifically designed for the precise semantic segmentation of the left ventricle in short-axis cine MRI scans. The methodology includes a down-sampling pathway for feature extraction and an up-sampling pathway for detail restoration, enhanced for medical imaging. Key modifications include techniques for better contextual understanding crucial in cardiac MRI segmentation. The dataset consists of 805 left ventricular MRI scans from 45 patients, with comparative analysis using established metrics such as the dice coefficient and mean perpendicular distance. GBU-Net significantly improves the accuracy of left ventricle segmentation in cine MRI scans. Its innovative design outperforms existing methods in tests, surpassing standard metrics like the dice coefficient and mean perpendicular distance. The approach is unique in its ability to capture contextual information, often missed in traditional CNN-based segmentation. An ensemble of the GBU-Net attains a 97% dice score on the SunnyBrook testing dataset. GBU-Net offers enhanced precision and contextual understanding in left ventricle segmentation for surgical robotics and medical analysis.
>
---
#### [new 121] Thinking with Blueprints: Assisting Vision-Language Models in Spatial Reasoning via Structured Object Representation
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的 spatial reasoning 任务，旨在提升模型对空间关系的理解。通过引入结构化对象蓝图，增强模型的空间推理能力。**

- **链接: [https://arxiv.org/pdf/2601.01984v1](https://arxiv.org/pdf/2601.01984v1)**

> **作者:** Weijian Ma; Shizhao Sun; Tianyu Yu; Ruiyu Wang; Tat-Seng Chua; Jiang Bian
>
> **备注:** Preprint. Under review
>
> **摘要:** Spatial reasoning -- the ability to perceive and reason about relationships in space -- advances vision-language models (VLMs) from visual perception toward spatial semantic understanding. Existing approaches either revisit local image patches, improving fine-grained perception but weakening global spatial awareness, or mark isolated coordinates, which capture object locations but overlook their overall organization. In this work, we integrate the cognitive concept of an object-centric blueprint into VLMs to enhance spatial reasoning. Given an image and a question, the model first constructs a JSON-style blueprint that records the positions, sizes, and attributes of relevant objects, and then reasons over this structured representation to produce the final answer. To achieve this, we introduce three key techniques: (1) blueprint-embedded reasoning traces for supervised fine-tuning to elicit basic reasoning skills; (2) blueprint-aware rewards in reinforcement learning to encourage the blueprint to include an appropriate number of objects and to align final answers with this causal reasoning; and (3) anti-shortcut data augmentation that applies targeted perturbations to images and questions, discouraging reliance on superficial visual or linguistic cues. Experiments show that our method consistently outperforms existing VLMs and specialized spatial reasoning models.
>
---
#### [new 122] AirSpatialBot: A Spatially-Aware Aerial Agent for Fine-Grained Vehicle Attribute Recognization and Retrieval
- **分类: cs.CV**

- **简介: 该论文提出AirSpatialBot，解决遥感中车辆细粒度属性识别与检索问题。构建了包含3DBB的AirSpatial数据集，采用两阶段训练策略提升模型空间理解能力。**

- **链接: [https://arxiv.org/pdf/2601.01416v1](https://arxiv.org/pdf/2601.01416v1)**

> **作者:** Yue Zhou; Ran Ding; Xue Yang; Xue Jiang; Xingzhao Liu
>
> **备注:** 12 pages, 9 figures
>
> **摘要:** Despite notable advancements in remote sensing vision-language models (VLMs), existing models often struggle with spatial understanding, limiting their effectiveness in real-world applications. To push the boundaries of VLMs in remote sensing, we specifically address vehicle imagery captured by drones and introduce a spatially-aware dataset AirSpatial, which comprises over 206K instructions and introduces two novel tasks: Spatial Grounding and Spatial Question Answering. It is also the first remote sensing grounding dataset to provide 3DBB. To effectively leverage existing image understanding of VLMs to spatial domains, we adopt a two-stage training strategy comprising Image Understanding Pre-training and Spatial Understanding Fine-tuning. Utilizing this trained spatially-aware VLM, we develop an aerial agent, AirSpatialBot, which is capable of fine-grained vehicle attribute recognition and retrieval. By dynamically integrating task planning, image understanding, spatial understanding, and task execution capabilities, AirSpatialBot adapts to diverse query requirements. Experimental results validate the effectiveness of our approach, revealing the spatial limitations of existing VLMs while providing valuable insights. The model, code, and datasets will be released at https://github.com/VisionXLab/AirSpatialBot
>
---
#### [new 123] HyDRA: Hybrid Denoising Regularization for Measurement-Only DEQ Training
- **分类: cs.CV; math.NA**

- **简介: 该论文属于图像重建任务，解决测量数据下DEQ模型训练问题。提出HyDRA框架，结合测量一致性与自适应去噪正则，实现高质量重建。**

- **链接: [https://arxiv.org/pdf/2601.01228v1](https://arxiv.org/pdf/2601.01228v1)**

> **作者:** Markus Haltmeier; Lukas Neumann; Nadja Gruber; Johannes Schwab; Gyeongha Hwang
>
> **摘要:** Solving image reconstruction problems of the form \(\mathbf{A} \mathbf{x} = \mathbf{y}\) remains challenging due to ill-posedness and the lack of large-scale supervised datasets. Deep Equilibrium (DEQ) models have been used successfully but typically require supervised pairs \((\mathbf{x},\mathbf{y})\). In many practical settings, only measurements \(\mathbf{y}\) are available. We introduce HyDRA (Hybrid Denoising Regularization Adaptation), a measurement-only framework for DEQ training that combines measurement consistency with an adaptive denoising regularization term, together with a data-driven early stopping criterion. Experiments on sparse-view CT demonstrate competitive reconstruction quality and fast inference.
>
---
#### [new 124] ITSELF: Attention Guided Fine-Grained Alignment for Vision-Language Retrieval
- **分类: cs.CV; cs.AI; cs.IR**

- **简介: 该论文属于视觉-语言检索任务，旨在解决细粒度对齐问题。提出ITSELF框架，通过注意力机制实现无监督的局部对齐，提升模型性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.01024v1](https://arxiv.org/pdf/2601.01024v1)**

> **作者:** Tien-Huy Nguyen; Huu-Loc Tran; Thanh Duc Ngo
>
> **备注:** Accepted at WACV Main Track 2026
>
> **摘要:** Vision Language Models (VLMs) have rapidly advanced and show strong promise for text-based person search (TBPS), a task that requires capturing fine-grained relationships between images and text to distinguish individuals. Previous methods address these challenges through local alignment, yet they are often prone to shortcut learning and spurious correlations, yielding misalignment. Moreover, injecting prior knowledge can distort intra-modality structure. Motivated by our finding that encoder attention surfaces spatially precise evidence from the earliest training epochs, and to alleviate these issues, we introduceITSELF, an attention-guided framework for implicit local alignment. At its core, Guided Representation with Attentive Bank (GRAB) converts the model's own attention into an Attentive Bank of high-saliency tokens and applies local objectives on this bank, learning fine-grained correspondences without extra supervision. To make the selection reliable and non-redundant, we introduce Multi-Layer Attention for Robust Selection (MARS), which aggregates attention across layers and performs diversity-aware top-k selection; and Adaptive Token Scheduler (ATS), which schedules the retention budget from coarse to fine over training, preserving context early while progressively focusing on discriminative details. Extensive experiments on three widely used TBPS benchmarks showstate-of-the-art performance and strong cross-dataset generalization, confirming the effectiveness and robustness of our approach without additional prior supervision. Our project is publicly available at https://trhuuloc.github.io/itself
>
---
#### [new 125] VAR RL Done Right: Tackling Asynchronous Policy Conflicts in Visual Autoregressive Generation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉生成任务，解决VAR模型在强化学习中因异构输入导致的策略冲突问题。提出改进的GRPO框架，包含奖励引导、时间加权和掩码传播机制，提升生成质量与对齐效果。**

- **链接: [https://arxiv.org/pdf/2601.02256v1](https://arxiv.org/pdf/2601.02256v1)**

> **作者:** Shikun Sun; Liao Qu; Huichao Zhang; Yiheng Liu; Yangyang Song; Xian Li; Xu Wang; Yi Jiang; Daniel K. Du; Xinglong Wu; Jia Jia
>
> **备注:** Project page: https://github.com/ByteVisionLab/NextFlow
>
> **摘要:** Visual generation is dominated by three paradigms: AutoRegressive (AR), diffusion, and Visual AutoRegressive (VAR) models. Unlike AR and diffusion, VARs operate on heterogeneous input structures across their generation steps, which creates severe asynchronous policy conflicts. This issue becomes particularly acute in reinforcement learning (RL) scenarios, leading to unstable training and suboptimal alignment. To resolve this, we propose a novel framework to enhance Group Relative Policy Optimization (GRPO) by explicitly managing these conflicts. Our method integrates three synergistic components: 1) a stabilizing intermediate reward to guide early-stage generation; 2) a dynamic time-step reweighting scheme for precise credit assignment; and 3) a novel mask propagation algorithm, derived from principles of Reward Feedback Learning (ReFL), designed to isolate optimization effects both spatially and temporally. Our approach demonstrates significant improvements in sample quality and objective alignment over the vanilla GRPO baseline, enabling robust and effective optimization for VAR models.
>
---
#### [new 126] Histogram Assisted Quality Aware Generative Model for Resolution Invariant NIR Image Colorization
- **分类: cs.CV; eess.IV**

- **简介: 该论文提出HAQAGen，解决NIR图像到RGB的色彩化任务，通过结合直方图匹配和结构保持策略，提升色彩真实性和细节精度。**

- **链接: [https://arxiv.org/pdf/2601.01103v1](https://arxiv.org/pdf/2601.01103v1)**

> **作者:** Abhinav Attri; Rajeev Ranjan Dwivedi; Samiran Das; Vinod Kumar Kurmi
>
> **备注:** Accepted at WACV 2026
>
> **摘要:** We present HAQAGen, a unified generative model for resolution-invariant NIR-to-RGB colorization that balances chromatic realism with structural fidelity. The proposed model introduces (i) a combined loss term aligning the global color statistics through differentiable histogram matching, perceptual image quality measure, and feature based similarity to preserve texture information, (ii) local hue-saturation priors injected via Spatially Adaptive Denormalization (SPADE) to stabilize chromatic reconstruction, and (iii) texture-aware supervision within a Mamba backbone to preserve fine details. We introduce an adaptive-resolution inference engine that further enables high-resolution translation without sacrificing quality. Our proposed NIR-to-RGB translation model simultaneously enforces global color statistics and local chromatic consistency, while scaling to native resolutions without compromising texture fidelity or generalization. Extensive evaluations on FANVID, OMSIV, VCIP2020, and RGB2NIR using different evaluation metrics demonstrate consistent improvements over state-of-the-art baseline methods. HAQAGen produces images with sharper textures, natural colors, attaining significant gains as per perceptual metrics. These results position HAQAGen as a scalable and effective solution for NIR-to-RGB translation across diverse imaging scenarios. Project Page: https://rajeev-dw9.github.io/HAQAGen/
>
---
#### [new 127] DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶领域，旨在解决生成式视频世界模型的评估问题。提出DrivingGen基准，涵盖多样数据和新指标，评估视觉真实、轨迹合理性等，推动可靠模型发展。**

- **链接: [https://arxiv.org/pdf/2601.01528v1](https://arxiv.org/pdf/2601.01528v1)**

> **作者:** Yang Zhou; Hao Shao; Letian Wang; Zhuofan Zong; Hongsheng Li; Steven L. Waslander
>
> **备注:** 10 pages, 4 figures; Project Website: https://drivinggen-bench.github.io/
>
> **摘要:** Video generation models, as one form of world models, have emerged as one of the most exciting frontiers in AI, promising agents the ability to imagine the future by modeling the temporal evolution of complex scenes. In autonomous driving, this vision gives rise to driving world models: generative simulators that imagine ego and agent futures, enabling scalable simulation, safe testing of corner cases, and rich synthetic data generation. Yet, despite fast-growing research activity, the field lacks a rigorous benchmark to measure progress and guide priorities. Existing evaluations remain limited: generic video metrics overlook safety-critical imaging factors; trajectory plausibility is rarely quantified; temporal and agent-level consistency is neglected; and controllability with respect to ego conditioning is ignored. Moreover, current datasets fail to cover the diversity of conditions required for real-world deployment. To address these gaps, we present DrivingGen, the first comprehensive benchmark for generative driving world models. DrivingGen combines a diverse evaluation dataset curated from both driving datasets and internet-scale video sources, spanning varied weather, time of day, geographic regions, and complex maneuvers, with a suite of new metrics that jointly assess visual realism, trajectory plausibility, temporal coherence, and controllability. Benchmarking 14 state-of-the-art models reveals clear trade-offs: general models look better but break physics, while driving-specific ones capture motion realistically but lag in visual quality. DrivingGen offers a unified evaluation framework to foster reliable, controllable, and deployable driving world models, enabling scalable simulation, planning, and data-driven decision-making.
>
---
#### [new 128] 360-GeoGS: Geometrically Consistent Feed-Forward 3D Gaussian Splatting Reconstruction for 360 Images
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决传统方法在几何一致性上的不足。提出一种新的3DGS框架，通过深度-法线正则化提升表面重建精度。**

- **链接: [https://arxiv.org/pdf/2601.02102v1](https://arxiv.org/pdf/2601.02102v1)**

> **作者:** Jiaqi Yao; Zhongmiao Yan; Jingyi Xu; Songpengcheng Xia; Yan Xiang; Ling Pei
>
> **摘要:** 3D scene reconstruction is fundamental for spatial intelligence applications such as AR, robotics, and digital twins. Traditional multi-view stereo struggles with sparse viewpoints or low-texture regions, while neural rendering approaches, though capable of producing high-quality results, require per-scene optimization and lack real-time efficiency. Explicit 3D Gaussian Splatting (3DGS) enables efficient rendering, but most feed-forward variants focus on visual quality rather than geometric consistency, limiting accurate surface reconstruction and overall reliability in spatial perception tasks. This paper presents a novel feed-forward 3DGS framework for 360 images, capable of generating geometrically consistent Gaussian primitives while maintaining high rendering quality. A Depth-Normal geometric regularization is introduced to couple rendered depth gradients with normal information, supervising Gaussian rotation, scale, and position to improve point cloud and surface accuracy. Experimental results show that the proposed method maintains high rendering quality while significantly improving geometric consistency, providing an effective solution for 3D reconstruction in spatial perception tasks.
>
---
#### [new 129] LinMU: Multimodal Understanding Made Linear
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; eess.IV**

- **简介: 该论文提出LinMU，解决视觉语言模型因自注意力复杂度高而难以部署于边缘设备的问题。通过线性复杂度模块实现高效多模态理解。**

- **链接: [https://arxiv.org/pdf/2601.01322v1](https://arxiv.org/pdf/2601.01322v1)**

> **作者:** Hongjie Wang; Niraj K. Jha
>
> **备注:** 23 pages, 7 figures
>
> **摘要:** Modern Vision-Language Models (VLMs) achieve impressive performance but are limited by the quadratic complexity of self-attention, which prevents their deployment on edge devices and makes their understanding of high-resolution images and long-context videos prohibitively expensive. To address this challenge, we introduce LinMU (Linear-complexity Multimodal Understanding), a VLM design that achieves linear complexity without using any quadratic-complexity modules while maintaining the performance of global-attention-based VLMs. LinMU replaces every self-attention layer in the VLM with the M-MATE block: a dual-branch module that combines a bidirectional state-space model for global context (Flex-MA branch) with localized Swin-style window attention (Local-Swin branch) for adjacent correlations. To transform a pre-trained VLM into the LinMU architecture, we propose a three-stage distillation framework that (i) initializes both branches with self-attention weights and trains the Flex-MA branch alone, (ii) unfreezes the Local-Swin branch and fine-tunes it jointly with the Flex-MA branch, and (iii) unfreezes the remaining blocks and fine-tunes them using LoRA adapters, while regressing on hidden states and token-level logits of the frozen VLM teacher. On MMMU, TextVQA, LongVideoBench, Video-MME, and other benchmarks, LinMU matches the performance of teacher models, yet reduces Time-To-First-Token (TTFT) by up to 2.7$\times$ and improves token throughput by up to 9.0$\times$ on minute-length videos. Ablations confirm the importance of each distillation stage and the necessity of the two branches of the M-MATE block. The proposed framework demonstrates that state-of-the-art multimodal reasoning can be achieved without quadratic attention, thus opening up avenues for long-context VLMs that can deal with high-resolution images and long videos.
>
---
#### [new 130] Learning Action Hierarchies via Hybrid Geometric Diffusion
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，解决时间动作分割问题。通过结合欧几里得与双曲几何的混合扩散模型，利用动作的层次结构提升分割效果。**

- **链接: [https://arxiv.org/pdf/2601.01914v1](https://arxiv.org/pdf/2601.01914v1)**

> **作者:** Arjun Ramesh Kaushik; Nalini K. Ratha; Venu Govindaraju
>
> **备注:** Accepted at WACV-26
>
> **摘要:** Temporal action segmentation is a critical task in video understanding, where the goal is to assign action labels to each frame in a video. While recent advances leverage iterative refinement-based strategies, they fail to explicitly utilize the hierarchical nature of human actions. In this work, we propose HybridTAS - a novel framework that incorporates a hybrid of Euclidean and hyperbolic geometries into the denoising process of diffusion models to exploit the hierarchical structure of actions. Hyperbolic geometry naturally provides tree-like relationships between embeddings, enabling us to guide the action label denoising process in a coarse-to-fine manner: higher diffusion timesteps are influenced by abstract, high-level action categories (root nodes), while lower timesteps are refined using fine-grained action classes (leaf nodes). Extensive experiments on three benchmark datasets, GTEA, 50Salads, and Breakfast, demonstrate that our method achieves state-of-the-art performance, validating the effectiveness of hyperbolic-guided denoising for the temporal action segmentation task.
>
---
#### [new 131] Mask-Guided Multi-Task Network for Face Attribute Recognition
- **分类: cs.CV**

- **简介: 该论文属于人脸属性识别任务，旨在解决传统方法因依赖全局区域导致的冗余特征问题。提出MGMTN模型，结合自适应掩码学习和组-全局特征融合，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2601.01408v1](https://arxiv.org/pdf/2601.01408v1)**

> **作者:** Gong Gao; Zekai Wang; Jian Zhao; Ziqi Xie; Xianhui Liu; Weidong Zhao
>
> **备注:** 23 pages, 9 figures
>
> **摘要:** Face Attribute Recognition (FAR) plays a crucial role in applications such as person re-identification, face retrieval, and face editing. Conventional multi-task attribute recognition methods often process the entire feature map for feature extraction and attribute classification, which can produce redundant features due to reliance on global regions. To address these challenges, we propose a novel approach emphasizing the selection of specific feature regions for efficient feature learning. We introduce the Mask-Guided Multi-Task Network (MGMTN), which integrates Adaptive Mask Learning (AML) and Group-Global Feature Fusion (G2FF) to address the aforementioned limitations. Leveraging a pre-trained keypoint annotation model and a fully convolutional network, AML accurately localizes critical facial parts (e.g., eye and mouth groups) and generates group masks that delineate meaningful feature regions, thereby mitigating negative transfer from global region usage. Furthermore, G2FF combines group and global features to enhance FAR learning, enabling more precise attribute identification. Extensive experiments on two challenging facial attribute recognition datasets demonstrate the effectiveness of MGMTN in improving FAR performance.
>
---
#### [new 132] Slot-ID: Identity-Preserving Video Generation from Reference Videos via Slot-Based Temporal Identity Encoding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成任务，旨在解决身份保留与动作自然性之间的矛盾。通过使用参考视频而非单图，引入动态编码以提升身份一致性。**

- **链接: [https://arxiv.org/pdf/2601.01352v1](https://arxiv.org/pdf/2601.01352v1)**

> **作者:** Yixuan Lai; He Wang; Kun Zhou; Tianjia Shao
>
> **摘要:** Producing prompt-faithful videos that preserve a user-specified identity remains challenging: models need to extrapolate facial dynamics from sparse reference while balancing the tension between identity preservation and motion naturalness. Conditioning on a single image completely ignores the temporal signature, which leads to pose-locked motions, unnatural warping, and "average" faces when viewpoints and expressions change. To this end, we introduce an identity-conditioned variant of a diffusion-transformer video generator which uses a short reference video rather than a single portrait. Our key idea is to incorporate the dynamics in the reference. A short clip reveals subject-specific patterns, e.g., how smiles form, across poses and lighting. From this clip, a Sinkhorn-routed encoder learns compact identity tokens that capture characteristic dynamics while remaining pretrained backbone-compatible. Despite adding only lightweight conditioning, the approach consistently improves identity retention under large pose changes and expressive facial behavior, while maintaining prompt faithfulness and visual realism across diverse subjects and prompts.
>
---
#### [new 133] Mono3DV: Monocular 3D Object Detection with 3D-Aware Bipartite Matching and Variational Query DeNoising
- **分类: cs.CV**

- **简介: 该论文属于单目3D目标检测任务，解决2D匹配标准抑制高质量3D预测的问题。提出Mono3DV框架，引入3D感知匹配和变分查询去噪机制，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2601.01036v1](https://arxiv.org/pdf/2601.01036v1)**

> **作者:** Kiet Dang Vu; Trung Thai Tran; Kien Nguyen Do Trung; Duc Dung Nguyen
>
> **摘要:** While DETR-like architectures have demonstrated significant potential for monocular 3D object detection, they are often hindered by a critical limitation: the exclusion of 3D attributes from the bipartite matching process. This exclusion arises from the inherent ill-posed nature of 3D estimation from monocular image, which introduces instability during training. Consequently, high-quality 3D predictions can be erroneously suppressed by 2D-only matching criteria, leading to suboptimal results. To address this, we propose Mono3DV, a novel Transformer-based framework. Our approach introduces three key innovations. First, we develop a 3D-Aware Bipartite Matching strategy that directly incorporates 3D geometric information into the matching cost, resolving the misalignment caused by purely 2D criteria. Second, it is important to stabilize the Bipartite Matching to resolve the instability occurring when integrating 3D attributes. Therefore, we propose 3D-DeNoising scheme in the training phase. Finally, recognizing the gradient vanishing issue associated with conventional denoising techniques, we propose a novel Variational Query DeNoising mechanism to overcome this limitation, which significantly enhances model performance. Without leveraging any external data, our method achieves state-of-the-art results on the KITTI 3D object detection benchmark.
>
---
#### [new 134] Achieving Fine-grained Cross-modal Understanding through Brain-inspired Hierarchical Representation Learning
- **分类: cs.CV**

- **简介: 该论文属于跨模态对齐任务，旨在解决神经数据与视觉输入间的模态差异问题。提出NeuroAlign框架，通过层次化结构实现细粒度fMRI与视频对齐。**

- **链接: [https://arxiv.org/pdf/2601.01339v1](https://arxiv.org/pdf/2601.01339v1)**

> **作者:** Weihang You; Hanqi Jiang; Yi Pan; Junhao Chen; Tianming Liu; Fei Dou
>
> **摘要:** Understanding neural responses to visual stimuli remains challenging due to the inherent complexity of brain representations and the modality gap between neural data and visual inputs. Existing methods, mainly based on reducing neural decoding to generation tasks or simple correlations, fail to reflect the hierarchical and temporal processes of visual processing in the brain. To address these limitations, we present NeuroAlign, a novel framework for fine-grained fMRI-video alignment inspired by the hierarchical organization of the human visual system. Our framework implements a two-stage mechanism that mirrors biological visual pathways: global semantic understanding through Neural-Temporal Contrastive Learning (NTCL) and fine-grained pattern matching through enhanced vector quantization. NTCL explicitly models temporal dynamics through bidirectional prediction between modalities, while our DynaSyncMM-EMA approach enables dynamic multi-modal fusion with adaptive weighting. Experiments demonstrate that NeuroAlign significantly outperforms existing methods in cross-modal retrieval tasks, establishing a new paradigm for understanding visual cognitive mechanisms.
>
---
#### [new 135] Garment Inertial Denoiser (GID): Endowing Accurate Motion Capture via Loose IMU Denoiser
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于运动捕捉任务，解决松散穿戴IMU数据噪声问题。提出GID模型，通过分阶段处理提升松散穿戴的运动捕捉精度。**

- **链接: [https://arxiv.org/pdf/2601.01360v1](https://arxiv.org/pdf/2601.01360v1)**

> **作者:** Jiawei Fang; Ruonan Zheng; Xiaoxia Gao; Shifan Jiang; Anjun Chen; Qi Ye; Shihui Guo
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Wearable inertial motion capture (MoCap) provides a portable, occlusion-free, and privacy-preserving alternative to camera-based systems, but its accuracy depends on tightly attached sensors - an intrusive and uncomfortable requirement for daily use. Embedding IMUs into loose-fitting garments is a desirable alternative, yet sensor-body displacement introduces severe, structured, and location-dependent corruption that breaks standard inertial pipelines. We propose GID (Garment Inertial Denoiser), a lightweight, plug-and-play Transformer that factorizes loose-wear MoCap into three stages: (i) location-specific denoising, (ii) adaptive cross-wear fusion, and (iii) general pose prediction. GID uses a location-aware expert architecture, where a shared spatio-temporal backbone models global motion while per-IMU expert heads specialize in local garment dynamics, and a lightweight fusion module ensures cross-part consistency. This inductive bias enables stable training and effective learning from limited paired loose-tight IMU data. We also introduce GarMoCap, a combined public and newly collected dataset covering diverse users, motions, and garments. Experiments show that GID enables accurate, real-time denoising from single-user training and generalizes across unseen users, motions, and garment types, consistently improving state-of-the-art inertial MoCap methods when used as a drop-in module.
>
---
#### [new 136] Evolving CNN Architectures: From Custom Designs to Deep Residual Models for Diverse Image Classification and Detection Tasks
- **分类: cs.CV; cs.AI**

- **简介: 论文比较了自定义CNN与预训练模型在图像分类和检测任务中的表现，分析了网络深度等因素的影响，旨在为不同任务选择合适架构。**

- **链接: [https://arxiv.org/pdf/2601.01099v1](https://arxiv.org/pdf/2601.01099v1)**

> **作者:** Mahmudul Hasan; Mabsur Fatin Bin Hossain
>
> **摘要:** This paper presents a comparative study of a custom convolutional neural network (CNN) architecture against widely used pretrained and transfer learning CNN models across five real-world image datasets. The datasets span binary classification, fine-grained multiclass recognition, and object detection scenarios. We analyze how architectural factors, such as network depth, residual connections, and feature extraction strategies, influence classification and localization performance. The results show that deeper CNN architectures provide substantial performance gains on fine-grained multiclass datasets, while lightweight pretrained and transfer learning models remain highly effective for simpler binary classification tasks. Additionally, we extend the proposed architecture to an object detection setting, demonstrating its adaptability in identifying unauthorized auto-rickshaws in real-world traffic scenes. Building upon a systematic analysis of custom CNN architectures alongside pretrained and transfer learning models, this study provides practical guidance for selecting suitable network designs based on task complexity and resource constraints.
>
---
#### [new 137] Prithvi-Complimentary Adaptive Fusion Encoder (CAFE): unlocking full-potential for flood inundation mapping
- **分类: cs.CV**

- **简介: 该论文属于洪水淹没地图生成任务，旨在解决GFMs在捕捉局部细节上的不足。通过融合Prithvi模型与CNN分支，提出CAFE架构，提升分割精度。**

- **链接: [https://arxiv.org/pdf/2601.02315v1](https://arxiv.org/pdf/2601.02315v1)**

> **作者:** Saurabh Kaushik; Lalit Maurya; Beth Tellman
>
> **备注:** Accepted at CV4EO Workshop @ WACV 2026
>
> **摘要:** Geo-Foundation Models (GFMs), have proven effective in diverse downstream applications, including semantic segmentation, classification, and regression tasks. However, in case of flood mapping using Sen1Flood11 dataset as a downstream task, GFMs struggles to outperform the baseline U-Net, highlighting model's limitation in capturing critical local nuances. To address this, we present the Prithvi-Complementary Adaptive Fusion Encoder (CAFE), which integrate Prithvi GFM pretrained encoder with a parallel CNN residual branch enhanced by Convolutional Attention Modules (CAM). Prithvi-CAFE enables fast and efficient fine-tuning through adapters in Prithvi and performs multi-scale, multi-level fusion with CNN features, capturing critical local details while preserving long-range dependencies. We achieve state-of-the-art results on two comprehensive flood mapping datasets: Sen1Flood11 and FloodPlanet. On Sen1Flood11 test data, Prithvi-CAFE (IoU 83.41) outperforms the original Prithvi (IoU 82.50) and other major GFMs (TerraMind 82.90, DOFA 81.54, spectralGPT: 81.02). The improvement is even more pronounced on the hold-out test site, where Prithvi-CAFE achieves an IoU of 81.37 compared to the baseline U-Net (70.57) and original Prithvi (72.42). On FloodPlanet, Prithvi-CAFE also surpasses the baseline U-Net and other GFMs, achieving an IoU of 64.70 compared to U-Net (60.14), Terramind (62.33), DOFA (59.15) and Prithvi 2.0 (61.91). Our proposed simple yet effective Prithvi-CAFE demonstrates strong potential for improving segmentation tasks where multi-channel and multi-modal data provide complementary information and local details are critical. The code is released on \href{https://github.com/Sk-2103/Prithvi-CAFE}{Prithvi-CAFE Github}
>
---
#### [new 138] Learnability-Driven Submodular Optimization for Active Roadside 3D Detection
- **分类: cs.CV**

- **简介: 该论文属于 roadside 3D 检测任务，解决标注困难和模型性能问题。通过主动学习选择可学习且易标注的样本，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.01695v1](https://arxiv.org/pdf/2601.01695v1)**

> **作者:** Ruiyu Mao; Baoming Zhang; Nicholas Ruozzi; Yunhui Guo
>
> **备注:** 10 pages, 7 figures. Submitted to CVPR 2026
>
> **摘要:** Roadside perception datasets are typically constructed via cooperative labeling between synchronized vehicle and roadside frame pairs. However, real deployment often requires annotation of roadside-only data due to hardware and privacy constraints. Even human experts struggle to produce accurate labels without vehicle-side data (image, LIDAR), which not only increases annotation difficulty and cost, but also reveals a fundamental learnability problem: many roadside-only scenes contain distant, blurred, or occluded objects whose 3D properties are ambiguous from a single view and can only be reliably annotated by cross-checking paired vehicle--roadside frames. We refer to such cases as inherently ambiguous samples. To reduce wasted annotation effort on inherently ambiguous samples while still obtaining high-performing models, we turn to active learning. This work focuses on active learning for roadside monocular 3D object detection and proposes a learnability-driven framework that selects scenes which are both informative and reliably labelable, suppressing inherently ambiguous samples while ensuring coverage. Experiments demonstrate that our method, LH3D, achieves 86.06%, 67.32%, and 78.67% of full-performance for vehicles, pedestrians, and cyclists respectively, using only 25% of the annotation budget on DAIR-V2X-I, significantly outperforming uncertainty-based baselines. This confirms that learnability, not uncertainty, matters for roadside 3D perception.
>
---
#### [new 139] Evaluating Deep Learning-Based Face Recognition for Infants and Toddlers: Impact of Age Across Developmental Stages
- **分类: cs.CV**

- **简介: 该论文属于人脸识别任务，针对婴儿和幼儿面部识别的挑战进行研究，评估不同模型在不同年龄阶段的性能，并提出改进方法以提高识别准确性。**

- **链接: [https://arxiv.org/pdf/2601.01680v1](https://arxiv.org/pdf/2601.01680v1)**

> **作者:** Afzal Hossain; Mst Rumana Sumi; Stephanie Schuckers
>
> **备注:** Accepted and presented at IEEE IJCB 2025 conference; final published version forthcoming
>
> **摘要:** Face recognition for infants and toddlers presents unique challenges due to rapid facial morphology changes, high inter-class similarity, and limited dataset availability. This study evaluates the performance of four deep learning-based face recognition models FaceNet, ArcFace, MagFace, and CosFace on a newly developed longitudinal dataset collected over a 24 month period in seven sessions involving children aged 0 to 3 years. Our analysis examines recognition accuracy across developmental stages, showing that the True Accept Rate (TAR) is only 30.7% at 0.1% False Accept Rate (FAR) for infants aged 0 to 6 months, due to unstable facial features. Performance improves significantly in older children, reaching 64.7% TAR at 0.1% FAR in the 2.5 to 3 year age group. We also evaluate verification performance over different time intervals, revealing that shorter time gaps result in higher accuracy due to reduced embedding drift. To mitigate this drift, we apply a Domain Adversarial Neural Network (DANN) approach that improves TAR by over 12%, yielding features that are more temporally stable and generalizable. These findings are critical for building biometric systems that function reliably over time in smart city applications such as public healthcare, child safety, and digital identity services. The challenges observed in early age groups highlight the importance of future research on privacy preserving biometric authentication systems that can address temporal variability, particularly in secure and regulated urban environments where child verification is essential.
>
---
#### [new 140] CAP-IQA: Context-Aware Prompt-Guided CT Image Quality Assessment
- **分类: cs.CV**

- **简介: 该论文属于CT图像质量评估任务，旨在解决传统提示方法引入偏差的问题。提出CAP-IQA框架，结合文本与上下文信息，提升评估准确性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.01613v1](https://arxiv.org/pdf/2601.01613v1)**

> **作者:** Kazi Ramisa Rifa; Jie Zhang; Abdullah Imran
>
> **备注:** 18 pages, 9 figures, 5 tables
>
> **摘要:** Prompt-based methods, which encode medical priors through descriptive text, have been only minimally explored for CT Image Quality Assessment (IQA). While such prompts can embed prior knowledge about diagnostic quality, they often introduce bias by reflecting idealized definitions that may not hold under real-world degradations such as noise, motion artifacts, or scanner variability. To address this, we propose the Context-Aware Prompt-guided Image Quality Assessment (CAP-IQA) framework, which integrates text-level priors with instance-level context prompts and applies causal debiasing to separate idealized knowledge from factual, image-specific degradations. Our framework combines a CNN-based visual encoder with a domain-specific text encoder to assess diagnostic visibility, anatomical clarity, and noise perception in abdominal CT images. The model leverages radiology-style prompts and context-aware fusion to align semantic and perceptual representations. On the 2023 LDCTIQA challenge benchmark, CAP-IQA achieves an overall correlation score of 2.8590 (sum of PLCC, SROCC, and KROCC), surpassing the top-ranked leaderboard team (2.7427) by 4.24%. Moreover, our comprehensive ablation experiments confirm that prompt-guided fusion and the simplified encoder-only design jointly enhance feature alignment and interpretability. Furthermore, evaluation on an in-house dataset of 91,514 pediatric CT images demonstrates the true generalizability of CAP-IQA in assessing perceptual fidelity in a different patient population.
>
---
#### [new 141] PhysSFI-Net: Physics-informed Geometric Learning of Skeletal and Facial Interactions for Orthognathic Surgical Outcome Prediction
- **分类: cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决正颌手术后面部形态预测问题。通过构建物理信息的几何深度学习框架PhysSFI-Net，实现高精度、可解释的软组织变形预测。**

- **链接: [https://arxiv.org/pdf/2601.02088v1](https://arxiv.org/pdf/2601.02088v1)**

> **作者:** Jiahao Bao; Huazhen Liu; Yu Zhuang; Leran Tao; Xinyu Xu; Yongtao Shi; Mengjia Cheng; Yiming Wang; Congshuang Ku; Ting Zeng; Yilang Du; Siyi Chen; Shunyao Shen; Suncheng Xiang; Hongbo Yu
>
> **备注:** 31 pages, 8 figures
>
> **摘要:** Orthognathic surgery repositions jaw bones to restore occlusion and enhance facial aesthetics. Accurate simulation of postoperative facial morphology is essential for preoperative planning. However, traditional biomechanical models are computationally expensive, while geometric deep learning approaches often lack interpretability. In this study, we develop and validate a physics-informed geometric deep learning framework named PhysSFI-Net for precise prediction of soft tissue deformation following orthognathic surgery. PhysSFI-Net consists of three components: a hierarchical graph module with craniofacial and surgical plan encoders combined with attention mechanisms to extract skeletal-facial interaction features; a Long Short-Term Memory (LSTM)-based sequential predictor for incremental soft tissue deformation; and a biomechanics-inspired module for high-resolution facial surface reconstruction. Model performance was assessed using point cloud shape error (Hausdorff distance), surface deviation error, and landmark localization error (Euclidean distances of craniomaxillofacial landmarks) between predicted facial shapes and corresponding ground truths. A total of 135 patients who underwent combined orthodontic and orthognathic treatment were included for model training and validation. Quantitative analysis demonstrated that PhysSFI-Net achieved a point cloud shape error of 1.070 +/- 0.088 mm, a surface deviation error of 1.296 +/- 0.349 mm, and a landmark localization error of 2.445 +/- 1.326 mm. Comparative experiments indicated that PhysSFI-Net outperformed the state-of-the-art method ACMT-Net in prediction accuracy. In conclusion, PhysSFI-Net enables interpretable, high-resolution prediction of postoperative facial morphology with superior accuracy, showing strong potential for clinical application in orthognathic surgical planning and simulation.
>
---
#### [new 142] VL-OrdinalFormer: Vision Language Guided Ordinal Transformers for Interpretable Knee Osteoarthritis Grading
- **分类: cs.CV**

- **简介: 该论文属于膝骨关节炎分级任务，旨在解决KL1与KL2阶段区分困难的问题。提出VLOrdinalFormer框架，结合视觉语言模型提升分类准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2601.00879v1](https://arxiv.org/pdf/2601.00879v1)**

> **作者:** Zahid Ullah; Jihie Kim
>
> **摘要:** Knee osteoarthritis (KOA) is a leading cause of disability worldwide, and accurate severity assessment using the Kellgren Lawrence (KL) grading system is critical for clinical decision making. However, radiographic distinctions between early disease stages, particularly KL1 and KL2, are subtle and frequently lead to inter-observer variability among radiologists. To address these challenges, we propose VLOrdinalFormer, a vision language guided ordinal learning framework for fully automated KOA grading from knee radiographs. The proposed method combines a ViT L16 backbone with CORAL based ordinal regression and a Contrastive Language Image Pretraining (CLIP) driven semantic alignment module, allowing the model to incorporate clinically meaningful textual concepts related to joint space narrowing, osteophyte formation, and subchondral sclerosis. To improve robustness and mitigate overfitting, we employ stratified five fold cross validation, class aware re weighting to emphasize challenging intermediate grades, and test time augmentation with global threshold optimization. Experiments conducted on the publicly available OAI kneeKL224 dataset demonstrate that VLOrdinalFormer achieves state of the art performance, outperforming CNN and ViT baselines in terms of macro F1 score and overall accuracy. Notably, the proposed framework yields substantial performance gains for KL1 and KL2 without compromising classification accuracy for mild or severe cases. In addition, interpretability analyses using Grad CAM and CLIP similarity maps confirm that the model consistently attends to clinically relevant anatomical regions. These results highlight the potential of vision language aligned ordinal transformers as reliable and interpretable tools for KOA grading and disease progression assessment in routine radiological practice.
>
---
#### [new 143] Unraveling MMDiT Blocks: Training-free Analysis and Enhancement of Text-conditioned Diffusion
- **分类: cs.CV**

- **简介: 该论文研究文本条件扩散模型中的MMDiT块，分析其结构与文本条件的交互机制，并提出无需训练的优化策略，以提升文本对齐和生成效果。任务为文本到图像生成。**

- **链接: [https://arxiv.org/pdf/2601.02211v1](https://arxiv.org/pdf/2601.02211v1)**

> **作者:** Binglei Li; Mengping Yang; Zhiyu Tan; Junping Zhang; Hao Li
>
> **备注:** 11 pages
>
> **摘要:** Recent breakthroughs of transformer-based diffusion models, particularly with Multimodal Diffusion Transformers (MMDiT) driven models like FLUX and Qwen Image, have facilitated thrilling experiences in text-to-image generation and editing. To understand the internal mechanism of MMDiT-based models, existing methods tried to analyze the effect of specific components like positional encoding and attention layers. Yet, a comprehensive understanding of how different blocks and their interactions with textual conditions contribute to the synthesis process remains elusive. In this paper, we first develop a systematic pipeline to comprehensively investigate each block's functionality by removing, disabling and enhancing textual hidden-states at corresponding blocks. Our analysis reveals that 1) semantic information appears in earlier blocks and finer details are rendered in later blocks, 2) removing specific blocks is usually less disruptive than disabling text conditions, and 3) enhancing textual conditions in selective blocks improves semantic attributes. Building on these observations, we further propose novel training-free strategies for improved text alignment, precise editing, and acceleration. Extensive experiments demonstrated that our method outperforms various baselines and remains flexible across text-to-image generation, image editing, and inference acceleration. Our method improves T2I-Combench++ from 56.92% to 63.00% and GenEval from 66.42% to 71.63% on SD3.5, without sacrificing synthesis quality. These results advance understanding of MMDiT models and provide valuable insights to unlock new possibilities for further improvements.
>
---
#### [new 144] Analyzing the Shopping Journey: Computing Shelf Browsing Visits in a Physical Retail Store
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于行为分析任务，旨在通过算法识别顾客在实体店的货架浏览行为，解决如何理解购物者意图的问题。工作包括算法开发、模型校准与评估，以及对浏览模式与购买关系的分析。**

- **链接: [https://arxiv.org/pdf/2601.00928v1](https://arxiv.org/pdf/2601.00928v1)**

> **作者:** Luis Yoichi Morales; Francesco Zanlungo; David M. Woollard
>
> **摘要:** Motivated by recent challenges in the deployment of robots into customer-facing roles within retail, this work introduces a study of customer activity in physical stores as a step toward autonomous understanding of shopper intent. We introduce an algorithm that computes shoppers' ``shelf visits'' -- capturing their browsing behavior in the store. Shelf visits are extracted from trajectories obtained via machine vision-based 3D tracking and overhead cameras. We perform two independent calibrations of the shelf visit algorithm, using distinct sets of trajectories (consisting of 8138 and 15129 trajectories), collected in different stores and labeled by human reviewers. The calibrated models are then evaluated on trajectories held out of the calibration process both from the same store on which calibration was performed and from the other store. An analysis of the results shows that the algorithm can recognize customers' browsing activity when evaluated in an environment different from the one on which calibration was performed. We then use the model to analyze the customers' ``browsing patterns'' on a large set of trajectories and their relation to actual purchases in the stores. Finally, we discuss how shelf browsing information could be used for retail planning and in the domain of human-robot interaction scenarios.
>
---
#### [new 145] Unsupervised SE(3) Disentanglement for in situ Macromolecular Morphology Identification from Cryo-Electron Tomography
- **分类: cs.CV**

- **简介: 该论文属于图像分析任务，解决cryo-ET中macromolecular形态识别问题。提出一种分离SE(3)变换与形态内容的深度学习框架，提升形态识别准确性和效率。**

- **链接: [https://arxiv.org/pdf/2601.01364v1](https://arxiv.org/pdf/2601.01364v1)**

> **作者:** Mostofa Rafid Uddin; Mahek Vora; Qifeng Wu; Muyuan Chen; Min Xu
>
> **摘要:** Cryo-electron tomography (cryo-ET) provides direct 3D visualization of macromolecules inside the cell, enabling analysis of their in situ morphology. This morphology can be regarded as an SE(3)-invariant, denoised volumetric representation of subvolumes extracted from tomograms. Inferring morphology is therefore an inverse problem of estimating both a template morphology and its SE(3) transformation. Existing expectation-maximization based solution to this problem often misses rare but important morphologies and requires extensive manual hyperparameter tuning. Addressing this issue, we present a disentangled deep representation learning framework that separates SE(3) transformations from morphological content in the representation space. The framework includes a novel multi-choice learning module that enables this disentanglement for highly noisy cryo-ET data, and the learned morphological content is used to generate template morphologies. Experiments on simulated and real cryo-ET datasets demonstrate clear improvements over prior methods, including the discovery of previously unidentified macromolecular morphologies.
>
---
#### [new 146] Trustworthy Data-Driven Wildfire Risk Prediction and Understanding in Western Canada
- **分类: cs.CV**

- **简介: 该论文属于 wildfire 风险预测任务，旨在解决数据驱动模型可靠性与可解释性问题。提出一种可信框架，融合多源数据并量化不确定性，提升预测精度与理解能力。**

- **链接: [https://arxiv.org/pdf/2601.01677v1](https://arxiv.org/pdf/2601.01677v1)**

> **作者:** Zhengsen Xu; Lanying Wang; Sibo Cheng; Xue Rui; Kyle Gao; Yimin Zhu; Mabel Heffring; Zack Dewis; Saeid Taleghanidoozdoozan; Megan Greenwood; Motasem Alkayid; Quinn Ledingham; Hongjie He; Jonathan Li; Lincoln Linlin Xu
>
> **摘要:** In recent decades, the intensification of wildfire activity in western Canada has resulted in substantial socio-economic and environmental losses. Accurate wildfire risk prediction is hindered by the intrinsic stochasticity of ignition and spread and by nonlinear interactions among fuel conditions, meteorology, climate variability, topography, and human activities, challenging the reliability and interpretability of purely data-driven models. We propose a trustworthy data-driven wildfire risk prediction framework based on long-sequence, multi-scale temporal modeling, which integrates heterogeneous drivers while explicitly quantifying predictive uncertainty and enabling process-level interpretation. Evaluated over western Canada during the record-breaking 2023 and 2024 fire seasons, the proposed model outperforms existing time-series approaches, achieving an F1 score of 0.90 and a PR-AUC of 0.98 with low computational cost. Uncertainty-aware analysis reveals structured spatial and seasonal patterns in predictive confidence, highlighting increased uncertainty associated with ambiguous predictions and spatiotemporal decision boundaries. SHAP-based interpretation provides mechanistic understanding of wildfire controls, showing that temperature-related drivers dominate wildfire risk in both years, while moisture-related constraints play a stronger role in shaping spatial and land-cover-specific contrasts in 2024 compared to the widespread hot and dry conditions of 2023. Data and code are available at https://github.com/SynUW/mmFire.
>
---
#### [new 147] PhyEduVideo: A Benchmark for Evaluating Text-to-Video Models for Physics Education
- **分类: cs.CV**

- **简介: 该论文属于文本到视频生成任务，旨在评估T2V模型在物理教育中的表现。工作包括构建基准，测试模型生成准确物理概念视频的能力。**

- **链接: [https://arxiv.org/pdf/2601.00943v1](https://arxiv.org/pdf/2601.00943v1)**

> **作者:** Megha Mariam K. M; Aditya Arun; Zakaria Laskar; C. V. Jawahar
>
> **备注:** Accepted at IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026
>
> **摘要:** Generative AI models, particularly Text-to-Video (T2V) systems, offer a promising avenue for transforming science education by automating the creation of engaging and intuitive visual explanations. In this work, we take a first step toward evaluating their potential in physics education by introducing a dedicated benchmark for explanatory video generation. The benchmark is designed to assess how well T2V models can convey core physics concepts through visual illustrations. Each physics concept in our benchmark is decomposed into granular teaching points, with each point accompanied by a carefully crafted prompt intended for visual explanation of the teaching point. T2V models are evaluated on their ability to generate accurate videos in response to these prompts. Our aim is to systematically explore the feasibility of using T2V models to generate high-quality, curriculum-aligned educational content-paving the way toward scalable, accessible, and personalized learning experiences powered by AI. Our evaluation reveals that current models produce visually coherent videos with smooth motion and minimal flickering, yet their conceptual accuracy is less reliable. Performance in areas such as mechanics, fluids, and optics is encouraging, but models struggle with electromagnetism and thermodynamics, where abstract interactions are harder to depict. These findings underscore the gap between visual quality and conceptual correctness in educational video generation. We hope this benchmark helps the community close that gap and move toward T2V systems that can deliver accurate, curriculum-aligned physics content at scale. The benchmark and accompanying codebase are publicly available at https://github.com/meghamariamkm/PhyEduVideo.
>
---
#### [new 148] Learning to Segment Liquids in Real-world Images
- **分类: cs.CV**

- **简介: 该论文属于液体分割任务，旨在解决机器人安全交互液体的难题。构建了LQDS数据集，并提出LQDM模型提升分割效果。**

- **链接: [https://arxiv.org/pdf/2601.00940v1](https://arxiv.org/pdf/2601.00940v1)**

> **作者:** Jonas Li; Michelle Li; Luke Liu; Heng Fan
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Different types of liquids such as water, wine and medicine appear in all aspects of daily life. However, limited attention has been given to the task, hindering the ability of robots to avoid or interact with liquids safely. The segmentation of liquids is difficult because liquids come in diverse appearances and shapes; moreover, they can be both transparent or reflective, taking on arbitrary objects and scenes from the background or surroundings. To take on this challenge, we construct a large-scale dataset of liquids named LQDS consisting of 5000 real-world images annotated into 14 distinct classes, and design a novel liquid detection model named LQDM, which leverages cross-attention between a dedicated boundary branch and the main segmentation branch to enhance segmentation predictions. Extensive experiments demonstrate the effectiveness of LQDM on the test set of LQDS, outperforming state-of-the-art methods and establishing a strong baseline for the semantic segmentation of liquids.
>
---
#### [new 149] XStreamVGGT: Extremely Memory-Efficient Streaming Vision Geometry Grounded Transformer with KV Cache Compression
- **分类: cs.CV**

- **简介: 该论文提出XStreamVGGT，解决3D视觉流任务中KV缓存占用过大的问题，通过剪枝和量化实现高效内存管理。**

- **链接: [https://arxiv.org/pdf/2601.01204v1](https://arxiv.org/pdf/2601.01204v1)**

> **作者:** Zunhai Su; Weihao Ye; Hansen Feng; Keyu Fan; Jing Zhang; Dahai Yu; Zhengwu Liu; Ngai Wong
>
> **摘要:** Learning-based 3D visual geometry models have benefited substantially from large-scale transformers. Among these, StreamVGGT leverages frame-wise causal attention for strong streaming reconstruction, but suffers from unbounded KV cache growth, leading to escalating memory consumption and inference latency as input frames accumulate. We propose XStreamVGGT, a tuning-free approach that systematically compresses the KV cache through joint pruning and quantization, enabling extremely memory-efficient streaming inference. Specifically, redundant KVs originating from multi-view inputs are pruned through efficient token importance identification, enabling a fixed memory budget. Leveraging the unique distribution of KV tensors, we incorporate KV quantization to further reduce memory consumption. Extensive evaluations show that XStreamVGGT achieves mostly negligible performance degradation while substantially reducing memory usage by 4.42$\times$ and accelerating inference by 5.48$\times$, enabling scalable and practical streaming 3D applications. The code is available at https://github.com/ywh187/XStreamVGGT/.
>
---
#### [new 150] Higher-Order Domain Generalization in Magnetic Resonance-Based Assessment of Alzheimer's Disease
- **分类: cs.CV**

- **简介: 该论文属于阿尔茨海默病诊断任务，旨在解决模型在不同数据域间泛化能力不足的问题。通过引入Extended MixStyle框架，提升模型跨域性能。**

- **链接: [https://arxiv.org/pdf/2601.01485v1](https://arxiv.org/pdf/2601.01485v1)**

> **作者:** Zobia Batool; Diala Lteif; Vijaya B. Kolachalama; Huseyin Ozkan; Erchan Aptoula
>
> **摘要:** Despite progress in deep learning for Alzheimer's disease (AD) diagnostics, models trained on structural magnetic resonance imaging (sMRI) often do not perform well when applied to new cohorts due to domain shifts from varying scanners, protocols and patient demographics. AD, the primary driver of dementia, manifests through progressive cognitive and neuroanatomical changes like atrophy and ventricular expansion, making robust, generalizable classification essential for real-world use. While convolutional neural networks and transformers have advanced feature extraction via attention and fusion techniques, single-domain generalization (SDG) remains underexplored yet critical, given the fragmented nature of AD datasets. To bridge this gap, we introduce Extended MixStyle (EM), a framework for blending higher-order feature moments (skewness and kurtosis) to mimic diverse distributional variations. Trained on sMRI data from the National Alzheimer's Coordinating Center (NACC; n=4,647) to differentiate persons with normal cognition (NC) from those with mild cognitive impairment (MCI) or AD and tested on three unseen cohorts (total n=3,126), EM yields enhanced cross-domain performance, improving macro-F1 on average by 2.4 percentage points over state-of-the-art SDG benchmarks, underscoring its promise for invariant, reliable AD detection in heterogeneous real-world settings. The source code will be made available upon acceptance at https://github.com/zobia111/Extended-Mixstyle.
>
---
#### [new 151] AlignVTOFF: Texture-Spatial Feature Alignment for High-Fidelity Virtual Try-Off
- **分类: cs.CV**

- **简介: 该论文属于虚拟试穿任务，旨在解决复杂变形和高频率纹理下的细节丢失问题。提出AlignVTOFF框架，通过特征对齐提升生成质量。**

- **链接: [https://arxiv.org/pdf/2601.02038v1](https://arxiv.org/pdf/2601.02038v1)**

> **作者:** Yihan Zhu; Mengying Ge
>
> **摘要:** Virtual Try-Off (VTOFF) is a challenging multimodal image generation task that aims to synthesize high-fidelity flat-lay garments under complex geometric deformation and rich high-frequency textures. Existing methods often rely on lightweight modules for fast feature extraction, which struggles to preserve structured patterns and fine-grained details, leading to texture attenuation during generation.To address these issues, we propose AlignVTOFF, a novel parallel U-Net framework built upon a Reference U-Net and Texture-Spatial Feature Alignment (TSFA). The Reference U-Net performs multi-scale feature extraction and enhances geometric fidelity, enabling robust modeling of deformation while retaining complex structured patterns. TSFA then injects the reference garment features into a frozen denoising U-Net via a hybrid attention design, consisting of a trainable cross-attention module and a frozen self-attention module. This design explicitly aligns texture and spatial cues and alleviates the loss of high-frequency information during the denoising process.Extensive experiments across multiple settings demonstrate that AlignVTOFF consistently outperforms state-of-the-art methods, producing flat-lay garment results with improved structural realism and high-frequency detail fidelity.
>
---
#### [new 152] Deepfake Detection with Multi-Artifact Subspace Fine-Tuning and Selective Layer Masking
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于深度伪造检测任务，旨在解决跨数据集和复杂场景下的检测难题。通过分解语义与伪影子空间并引入选择性层掩码，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.01041v1](https://arxiv.org/pdf/2601.01041v1)**

> **作者:** Xiang Zhang; Wenliang Weng; Daoyong Fu; Ziqiang Li; Zhangjie Fu
>
> **摘要:** Deepfake detection still faces significant challenges in cross-dataset and real-world complex scenarios. The root cause lies in the high diversity of artifact distributions introduced by different forgery methods, while pretrained models tend to disrupt their original general semantic structures when adapting to new artifacts. Existing approaches usually rely on indiscriminate global parameter updates or introduce additional supervision signals, making it difficult to effectively model diverse forgery artifacts while preserving semantic stability. To address these issues, this paper proposes a deepfake detection method based on Multi-Artifact Subspaces and selective layer masks (MASM), which explicitly decouples semantic representations from artifact representations and constrains the fitting strength of artifact subspaces, thereby improving generalization robustness in cross-dataset scenarios. Specifically, MASM applies singular value decomposition to model weights, partitioning pretrained weights into a stable semantic principal subspace and multiple learnable artifact subspaces. This design enables decoupled modeling of different forgery artifact patterns while preserving the general semantic subspace. On this basis, a selective layer mask strategy is introduced to adaptively regulate the update behavior of corresponding network layers according to the learning state of each artifact subspace, suppressing overfitting to any single forgery characteristic. Furthermore, orthogonality constraints and spectral consistency constraints are imposed to jointly regularize multiple artifact subspaces, guiding them to learn complementary and diverse artifact representations while maintaining a stable overall spectral structure.
>
---
#### [new 153] Decoupling Amplitude and Phase Attention in Frequency Domain for RGB-Event based Visual Object Tracking
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于RGB-Event视觉目标跟踪任务，旨在解决传统方法未能充分利用事件相机优势的问题。通过频域早期融合与注意力机制，提升特征表示并减少计算量。**

- **链接: [https://arxiv.org/pdf/2601.01022v1](https://arxiv.org/pdf/2601.01022v1)**

> **作者:** Shiao Wang; Xiao Wang; Haonan Zhao; Jiarui Xu; Bo Jiang; Lin Zhu; Xin Zhao; Yonghong Tian; Jin Tang
>
> **摘要:** Existing RGB-Event visual object tracking approaches primarily rely on conventional feature-level fusion, failing to fully exploit the unique advantages of event cameras. In particular, the high dynamic range and motion-sensitive nature of event cameras are often overlooked, while low-information regions are processed uniformly, leading to unnecessary computational overhead for the backbone network. To address these issues, we propose a novel tracking framework that performs early fusion in the frequency domain, enabling effective aggregation of high-frequency information from the event modality. Specifically, RGB and event modalities are transformed from the spatial domain to the frequency domain via the Fast Fourier Transform, with their amplitude and phase components decoupled. High-frequency event information is selectively fused into RGB modality through amplitude and phase attention, enhancing feature representation while substantially reducing backbone computation. In addition, a motion-guided spatial sparsification module leverages the motion-sensitive nature of event cameras to capture the relationship between target motion cues and spatial probability distribution, filtering out low-information regions and enhancing target-relevant features. Finally, a sparse set of target-relevant features is fed into the backbone network for learning, and the tracking head predicts the final target position. Extensive experiments on three widely used RGB-Event tracking benchmark datasets, including FE108, FELT, and COESOT, demonstrate the high performance and efficiency of our method. The source code of this paper will be released on https://github.com/Event-AHU/OpenEvTracking
>
---
#### [new 154] CogFlow: Bridging Perception and Reasoning through Knowledge Internalization for Visual Mathematical Problem Solving
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CogFlow，解决视觉数学问题求解中感知与推理脱节的问题，通过知识内化框架提升模型对视觉信息的整合与利用能力。**

- **链接: [https://arxiv.org/pdf/2601.01874v1](https://arxiv.org/pdf/2601.01874v1)**

> **作者:** Shuhang Chen; Yunqiu Xu; Junjie Xie; Aojun Lu; Tao Feng; Zeying Huang; Ning Zhang; Yi Sun; Yi Yang; Hangjie Yuan
>
> **摘要:** Despite significant progress, multimodal large language models continue to struggle with visual mathematical problem solving. Some recent works recognize that visual perception is a bottleneck in visual mathematical reasoning, but their solutions are limited to improving the extraction and interpretation of visual inputs. Notably, they all ignore the key issue of whether the extracted visual cues are faithfully integrated and properly utilized in subsequent reasoning. Motivated by this, we present CogFlow, a novel cognitive-inspired three-stage framework that incorporates a knowledge internalization stage, explicitly simulating the hierarchical flow of human reasoning: perception$\Rightarrow$internalization$\Rightarrow$reasoning. Inline with this hierarchical flow, we holistically enhance all its stages. We devise Synergistic Visual Rewards to boost perception capabilities in parametric and semantic spaces, jointly improving visual information extraction from symbols and diagrams. To guarantee faithful integration of extracted visual cues into subsequent reasoning, we introduce a Knowledge Internalization Reward model in the internalization stage, bridging perception and reasoning. Moreover, we design a Visual-Gated Policy Optimization algorithm to further enforce the reasoning is grounded with the visual knowledge, preventing models seeking shortcuts that appear coherent but are visually ungrounded reasoning chains. Moreover, we contribute a new dataset MathCog for model training, which contains samples with over 120K high-quality perception-reasoning aligned annotations. Comprehensive experiments and analysis on commonly used visual mathematical reasoning benchmarks validate the superiority of the proposed CogFlow.
>
---
#### [new 155] Evaluating Contextual Intelligence in Recyclability: A Comprehensive Study of Image-Based Reasoning Systems
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像推理任务，旨在解决物品可回收性判断问题。通过评估视觉语言模型在不同场景下的表现，提升公众垃圾分类准确性。**

- **链接: [https://arxiv.org/pdf/2601.00905v1](https://arxiv.org/pdf/2601.00905v1)**

> **作者:** Eliot Park; Abhi Kumar; Pranav Rajpurkar
>
> **备注:** x
>
> **摘要:** While the importance of efficient recycling is widely acknowledged, accurately determining the recyclability of items and their proper disposal remains a complex task for the general public. In this study, we explore the application of cutting-edge vision-language models (GPT-4o, GPT-4o-mini, and Claude 3.5) for predicting the recyclability of commonly disposed items. Utilizing a curated dataset of images, we evaluated the models' ability to match objects to appropriate recycling bins, including assessing whether the items could physically fit into the available bins. Additionally, we investigated the models' performance across several challenging scenarios: (i) adjusting predictions based on location-specific recycling guidelines; (ii) accounting for contamination or structural damage; and (iii) handling objects composed of multiple materials. Our findings highlight the significant advancements in contextual understanding offered by these models compared to previous iterations, while also identifying areas where they still fall short. The continued refinement of context-aware models is crucial for enhancing public recycling practices and advancing environmental sustainability.
>
---
#### [new 156] RefSR-Adv: Adversarial Attack on Reference-based Image Super-Resolution Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像超分辨率任务，针对参考图像引导的超分辨率模型（RefSR）提出对抗攻击方法RefSR-Adv，通过扰动参考图像降低模型性能，揭示其安全漏洞。**

- **链接: [https://arxiv.org/pdf/2601.01202v1](https://arxiv.org/pdf/2601.01202v1)**

> **作者:** Jiazhu Dai; Huihui Jiang
>
> **摘要:** Single Image Super-Resolution (SISR) aims to recover high-resolution images from low-resolution inputs. Unlike SISR, Reference-based Super-Resolution (RefSR) leverages an additional high-resolution reference image to facilitate the recovery of high-frequency textures. However, existing research mainly focuses on backdoor attacks targeting RefSR, while the vulnerability of the adversarial attacks targeting RefSR has not been fully explored. To fill this research gap, we propose RefSR-Adv, an adversarial attack that degrades SR outputs by perturbing only the reference image. By maximizing the difference between adversarial and clean outputs, RefSR-Adv induces significant performance degradation and generates severe artifacts across CNN, Transformer, and Mamba architectures on the CUFED5, WR-SR, and DRefSR datasets. Importantly, experiments confirm a positive correlation between the similarity of the low-resolution input and the reference image and attack effectiveness, revealing that the model's over-reliance on reference features is a key security flaw. This study reveals a security vulnerability in RefSR systems, aiming to urge researchers to pay attention to the robustness of RefSR.
>
---
#### [new 157] Talk2Move: Reinforcement Learning for Text-Instructed Object-Level Geometric Transformation in Scenes
- **分类: cs.CV**

- **简介: 该论文提出Talk2Move，属于文本指导的场景物体几何变换任务。解决文本引导下物体几何变换困难的问题，通过强化学习实现精准、一致的物体操作。**

- **链接: [https://arxiv.org/pdf/2601.02356v1](https://arxiv.org/pdf/2601.02356v1)**

> **作者:** Jing Tan; Zhaoyang Zhang; Yantao Shen; Jiarui Cai; Shuo Yang; Jiajun Wu; Wei Xia; Zhuowen Tu; Stefano Soatto
>
> **备注:** Project page: https://sparkstj.github.io/talk2move
>
> **摘要:** We introduce Talk2Move, a reinforcement learning (RL) based diffusion framework for text-instructed spatial transformation of objects within scenes. Spatially manipulating objects in a scene through natural language poses a challenge for multimodal generation systems. While existing text-based manipulation methods can adjust appearance or style, they struggle to perform object-level geometric transformations-such as translating, rotating, or resizing objects-due to scarce paired supervision and pixel-level optimization limits. Talk2Move employs Group Relative Policy Optimization (GRPO) to explore geometric actions through diverse rollouts generated from input images and lightweight textual variations, removing the need for costly paired data. A spatial reward guided model aligns geometric transformations with linguistic description, while off-policy step evaluation and active step sampling improve learning efficiency by focusing on informative transformation stages. Furthermore, we design object-centric spatial rewards that evaluate displacement, rotation, and scaling behaviors directly, enabling interpretable and coherent transformations. Experiments on curated benchmarks demonstrate that Talk2Move achieves precise, consistent, and semantically faithful object transformations, outperforming existing text-guided editing approaches in both spatial accuracy and scene coherence.
>
---
#### [new 158] Motion-Compensated Latent Semantic Canvases for Visual Situational Awareness on Edge
- **分类: cs.CV**

- **简介: 该论文提出MCLSC方法，用于边缘设备的视觉情境感知。解决资源受限下频繁分割计算效率低的问题，通过两个语义图层和运动补偿技术，减少分割调用次数并提升处理速度。**

- **链接: [https://arxiv.org/pdf/2601.00854v1](https://arxiv.org/pdf/2601.00854v1)**

> **作者:** Igor Lodin; Sergii Filatov; Vira Filatova; Dmytro Filatov
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** We propose Motion-Compensated Latent Semantic Canvases (MCLSC) for visual situational awareness on resource-constrained edge devices. The core idea is to maintain persistent semantic metadata in two latent canvases - a slowly accumulating static layer and a rapidly updating dynamic layer - defined in a baseline coordinate frame stabilized from the video stream. Expensive panoptic segmentation (Mask2Former) runs asynchronously and is motion-gated: inference is triggered only when motion indicates new information, while stabilization/motion compensation preserves a consistent coordinate system for latent semantic memory. On prerecorded 480p clips, our prototype reduces segmentation calls by >30x and lowers mean end-to-end processing time by >20x compared to naive per-frame segmentation, while maintaining coherent static/dynamic semantic overlays.
>
---
#### [new 159] TopoLoRA-SAM: Topology-Aware Parameter-Efficient Adaptation of Foundation Segmenters for Thin-Structure and Cross-Domain Binary Semantic Segmentation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对基础分割模型在细结构和跨域二值分割中的适应问题，提出TopoLoRA-SAM框架，通过参数高效方式提升分割性能。**

- **链接: [https://arxiv.org/pdf/2601.02273v1](https://arxiv.org/pdf/2601.02273v1)**

> **作者:** Salim Khazem
>
> **摘要:** Foundation segmentation models such as the Segment Anything Model (SAM) exhibit strong zero-shot generalization through large-scale pretraining, but adapting them to domain-specific semantic segmentation remains challenging, particularly for thin structures (e.g., retinal vessels) and noisy modalities (e.g., SAR imagery). Full fine-tuning is computationally expensive and risks catastrophic forgetting. We propose \textbf{TopoLoRA-SAM}, a topology-aware and parameter-efficient adaptation framework for binary semantic segmentation. TopoLoRA-SAM injects Low-Rank Adaptation (LoRA) into the frozen ViT encoder, augmented with a lightweight spatial convolutional adapter and optional topology-aware supervision via differentiable clDice. We evaluate our approach on five benchmarks spanning retinal vessel segmentation (DRIVE, STARE, CHASE\_DB1), polyp segmentation (Kvasir-SEG), and SAR sea/land segmentation (SL-SSDD), comparing against U-Net, DeepLabV3+, SegFormer, and Mask2Former. TopoLoRA-SAM achieves the best retina-average Dice and the best overall average Dice across datasets, while training only \textbf{5.2\%} of model parameters ($\sim$4.9M). On the challenging CHASE\_DB1 dataset, our method substantially improves segmentation accuracy and robustness, demonstrating that topology-aware parameter-efficient adaptation can match or exceed fully fine-tuned specialist models. Code is available at : https://github.com/salimkhazem/Seglab.git
>
---
#### [new 160] Robust Ship Detection and Tracking Using Modified ViBe and Backwash Cancellation Algorithm
- **分类: cs.CV**

- **简介: 该论文属于目标检测与跟踪任务，旨在解决海岸视频中船舶的实时准确检测与跟踪问题。通过改进ViBe算法和提出后浪消除方法，提升检测鲁棒性与精度。**

- **链接: [https://arxiv.org/pdf/2601.01481v1](https://arxiv.org/pdf/2601.01481v1)**

> **作者:** Mohammad Hassan Saghafi; Seyed Majid Noorhosseini; Seyed Abolfazl Seyed Javadein; Hadi Khalili
>
> **摘要:** In this paper, we propose a robust real time detection and tracking method for detecting ships in a coastal video sequences. Since coastal scenarios are unpredictable and scenes have dynamic properties it is essential to apply detection methods that are robust to these conditions. This paper presents modified ViBe for moving object detection which detects ships and backwash. In the modified ViBe the probability of losing ships is decreased in comparison with the original ViBe. It is robust to natural sea waves and variation of lights and is capable of quickly updating the background. Based on geometrical properties of ship and some concepts such as brightness distortion, a new method for backwash cancellation is proposed. Experimental results demonstrate that the proposed strategy and methods have outstanding performance in ship detection and tracking. These results also illustrate real time and precise performance of the proposed strategy.
>
---
#### [new 161] Efficient Hyperspectral Image Reconstruction Using Lightweight Separate Spectral Transformers
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于 hyperspectral image reconstruction 任务，解决从压缩感知测量中高效重建高光谱图像的问题。提出 LSST 架构，结合 spectral 和 spatial 处理模块，提升重建效率与质量。**

- **链接: [https://arxiv.org/pdf/2601.01064v1](https://arxiv.org/pdf/2601.01064v1)**

> **作者:** Jianan Li; Wangcai Zhao; Tingfa Xu
>
> **摘要:** Hyperspectral imaging (HSI) is essential across various disciplines for its capacity to capture rich spectral information. However, efficiently reconstructing hyperspectral images from compressive sensing measurements presents significant challenges. To tackle these, we adopt a divide-and-conquer strategy that capitalizes on the unique spectral and spatial characteristics of hyperspectral images. We introduce the Lightweight Separate Spectral Transformer (LSST), an innovative architecture tailored for efficient hyperspectral image reconstruction. This architecture consists of Separate Spectral Transformer Blocks (SSTB) for modeling spectral relationships and Lightweight Spatial Convolution Blocks (LSCB) for spatial processing. The SSTB employs Grouped Spectral Self-attention and a Spectrum Shuffle operation to effectively manage both local and non-local spectral relationships. Simultaneously, the LSCB utilizes depth-wise separable convolutions and strategic ordering to enhance spatial information processing. Furthermore, we implement the Focal Spectrum Loss, a novel loss weighting mechanism that dynamically adjusts during training to improve reconstruction across spectrally complex bands. Extensive testing demonstrates that our LSST achieves superior performance while requiring fewer FLOPs and parameters, underscoring its efficiency and effectiveness. The source code is available at: https://github.com/wcz1124/LSST.
>
---
#### [new 162] DDNet: A Dual-Stream Graph Learning and Disentanglement Framework for Temporal Forgery Localization
- **分类: cs.CV; cs.MM; eess.IV**

- **简介: 该论文属于视频篡改定位任务，旨在解决传统方法因局部视角导致全局异常检测不足的问题。提出DDNet框架，结合时序与语义流，并引入TDA和CLFE提升检测精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.01784v1](https://arxiv.org/pdf/2601.01784v1)**

> **作者:** Boyang Zhao; Xin Liao; Jiaxin Chen; Xiaoshuai Wu; Yufeng Wu
>
> **摘要:** The rapid evolution of AIGC technology enables misleading viewers by tampering mere small segments within a video, rendering video-level detection inaccurate and unpersuasive. Consequently, temporal forgery localization (TFL), which aims to precisely pinpoint tampered segments, becomes critical. However, existing methods are often constrained by \emph{local view}, failing to capture global anomalies. To address this, we propose a \underline{d}ual-stream graph learning and \underline{d}isentanglement framework for temporal forgery localization (DDNet). By coordinating a \emph{Temporal Distance Stream} for local artifacts and a \emph{Semantic Content Stream} for long-range connections, DDNet prevents global cues from being drowned out by local smoothness. Furthermore, we introduce Trace Disentanglement and Adaptation (TDA) to isolate generic forgery fingerprints, alongside Cross-Level Feature Embedding (CLFE) to construct a robust feature foundation via deep fusion of hierarchical features. Experiments on ForgeryNet and TVIL benchmarks demonstrate that our method outperforms state-of-the-art approaches by approximately 9\% in AP@0.95, with significant improvements in cross-domain robustness.
>
---
#### [new 163] Enhancing Histopathological Image Classification via Integrated HOG and Deep Features with Robust Noise Performance
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像分类任务，旨在提升病理图像分类性能。通过结合HOG与深度特征，使用改进的InceptionResNet-v2网络进行分类，验证了模型在不同噪声条件下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.01056v1](https://arxiv.org/pdf/2601.01056v1)**

> **作者:** Ifeanyi Ezuma; Ugochukwu Ugwu
>
> **备注:** 10 pages, 8 figures. Code and datasets available upon request
>
> **摘要:** The era of digital pathology has advanced histopathological examinations, making automated image analysis essential in clinical practice. This study evaluates the classification performance of machine learning and deep learning models on the LC25000 dataset, which includes five classes of histopathological images. We used the fine-tuned InceptionResNet-v2 network both as a classifier and for feature extraction. Our results show that the fine-tuned InceptionResNet-v2 achieved a classification accuracy of 96.01\% and an average AUC of 96.8\%. Models trained on deep features from InceptionResNet-v2 outperformed those using only the pre-trained network, with the Neural Network model achieving an AUC of 99.99\% and accuracy of 99.84\%. Evaluating model robustness under varying SNR conditions revealed that models using deep features exhibited greater resilience, particularly GBM and KNN. The combination of HOG and deep features showed enhanced performance, however, less so in noisy environments.
>
---
#### [new 164] Guiding Token-Sparse Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，解决稀疏训练扩散模型在推理时性能下降的问题。提出Sparse Guidance方法，利用token级稀疏性提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2601.01608v1](https://arxiv.org/pdf/2601.01608v1)**

> **作者:** Felix Krause; Stefan Andreas Baumann; Johannes Schusterbauer; Olga Grebenkova; Ming Gui; Vincent Tao Hu; Björn Ommer
>
> **摘要:** Diffusion models deliver high quality in image synthesis but remain expensive during training and inference. Recent works have leveraged the inherent redundancy in visual content to make training more affordable by training only on a subset of visual information. While these methods were successful in providing cheaper and more effective training, sparsely trained diffusion models struggle in inference. This is due to their lacking response to Classifier-free Guidance (CFG) leading to underwhelming performance during inference. To overcome this, we propose Sparse Guidance (SG). Instead of using conditional dropout as a signal to guide diffusion models, SG uses token-level sparsity. As a result, SG preserves the high-variance of the conditional prediction better, achieving good quality and high variance outputs. Leveraging token-level sparsity at inference, SG improves fidelity at lower compute, achieving 1.58 FID on the commonly used ImageNet-256 benchmark with 25% fewer FLOPs, and yields up to 58% FLOP savings at matched baseline quality. To demonstrate the effectiveness of Sparse Guidance, we train a 2.5B text-to-image diffusion model using training time sparsity and leverage SG during inference. SG achieves improvements in composition and human preference score while increasing throughput at the same time.
>
---
#### [new 165] NextFlow: Unified Sequential Modeling Activates Multimodal Understanding and Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出NextFlow，一种统一的解码器模型，用于多模态理解和生成。解决传统方法在速度和稳定性上的问题，实现高效图像生成与编辑。**

- **链接: [https://arxiv.org/pdf/2601.02204v1](https://arxiv.org/pdf/2601.02204v1)**

> **作者:** Huichao Zhang; Liao Qu; Yiheng Liu; Hang Chen; Yangyang Song; Yongsheng Dong; Shikun Sun; Xian Li; Xu Wang; Yi Jiang; Hu Ye; Bo Chen; Yiming Gao; Peng Liu; Akide Liu; Zhipeng Yang; Qili Deng; Linjie Xing; Jiyang Liu; Zhao Wang; Yang Zhou; Mingcong Liu; Yi Zhang; Qian He; Xiwei Hu; Zhongqi Qi; Jie Shao; Zhiye Fu; Shuai Wang; Fangmin Chen; Xuezhi Chai; Zhihua Wu; Yitong Wang; Zehuan Yuan; Daniel K. Du; Xinglong Wu
>
> **备注:** Project page: https://github.com/ByteVisionLab/NextFlow
>
> **摘要:** We present NextFlow, a unified decoder-only autoregressive transformer trained on 6 trillion interleaved text-image discrete tokens. By leveraging a unified vision representation within a unified autoregressive architecture, NextFlow natively activates multimodal understanding and generation capabilities, unlocking abilities of image editing, interleaved content and video generation. Motivated by the distinct nature of modalities - where text is strictly sequential and images are inherently hierarchical - we retain next-token prediction for text but adopt next-scale prediction for visual generation. This departs from traditional raster-scan methods, enabling the generation of 1024x1024 images in just 5 seconds - orders of magnitude faster than comparable AR models. We address the instabilities of multi-scale generation through a robust training recipe. Furthermore, we introduce a prefix-tuning strategy for reinforcement learning. Experiments demonstrate that NextFlow achieves state-of-the-art performance among unified models and rivals specialized diffusion baselines in visual quality.
>
---
#### [new 166] Beyond Segmentation: An Oil Spill Change Detection Framework Using Synthetic SAR Imagery
- **分类: cs.CV**

- **简介: 该论文提出OSCD任务，解决油污检测中误报率高的问题。通过生成合成前时影像，提升变化检测精度。**

- **链接: [https://arxiv.org/pdf/2601.02139v1](https://arxiv.org/pdf/2601.02139v1)**

> **作者:** Chenyang Lai; Shuaiyu Chen; Tianjin Huang; Siyang Song; Guangliang Cheng; Chunbo Luo; Zeyu Fu
>
> **摘要:** Marine oil spills are urgent environmental hazards that demand rapid and reliable detection to minimise ecological and economic damage. While Synthetic Aperture Radar (SAR) imagery has become a key tool for large-scale oil spill monitoring, most existing detection methods rely on deep learning-based segmentation applied to single SAR images. These static approaches struggle to distinguish true oil spills from visually similar oceanic features (e.g., biogenic slicks or low-wind zones), leading to high false positive rates and limited generalizability, especially under data-scarce conditions. To overcome these limitations, we introduce Oil Spill Change Detection (OSCD), a new bi-temporal task that focuses on identifying changes between pre- and post-spill SAR images. As real co-registered pre-spill imagery is not always available, we propose the Temporal-Aware Hybrid Inpainting (TAHI) framework, which generates synthetic pre-spill images from post-spill SAR data. TAHI integrates two key components: High-Fidelity Hybrid Inpainting for oil-free reconstruction, and Temporal Realism Enhancement for radiometric and sea-state consistency. Using TAHI, we construct the first OSCD dataset and benchmark several state-of-the-art change detection models. Results show that OSCD significantly reduces false positives and improves detection accuracy compared to conventional segmentation, demonstrating the value of temporally-aware methods for reliable, scalable oil spill monitoring in real-world scenarios.
>
---
#### [new 167] Fusion2Print: Deep Flash-Non-Flash Fusion for Contactless Fingerprint Matching
- **分类: cs.CV**

- **简介: 该论文属于指纹识别任务，解决接触式与非接触式指纹图像质量差的问题。通过融合闪光与非闪光图像，提升纹路清晰度，提高识别准确率。**

- **链接: [https://arxiv.org/pdf/2601.02318v1](https://arxiv.org/pdf/2601.02318v1)**

> **作者:** Roja Sahoo; Anoop Namboodiri
>
> **备注:** 15 pages, 8 figures, 5 tables. Submitted to ICPR 2026
>
> **摘要:** Contactless fingerprint recognition offers a hygienic and convenient alternative to contact-based systems, enabling rapid acquisition without latent prints, pressure artifacts, or hygiene risks. However, contactless images often show degraded ridge clarity due to illumination variation, subcutaneous skin discoloration, and specular reflections. Flash captures preserve ridge detail but introduce noise, whereas non-flash captures reduce noise but lower ridge contrast. We propose Fusion2Print (F2P), the first framework to systematically capture and fuse paired flash-non-flash contactless fingerprints. We construct a custom paired dataset, FNF Database, and perform manual flash-non-flash subtraction to isolate ridge-preserving signals. A lightweight attention-based fusion network also integrates both modalities, emphasizing informative channels and suppressing noise, and then a U-Net enhancement module produces an optimally weighted grayscale image. Finally, a deep embedding model with cross-domain compatibility, generates discriminative and robust representations in a unified embedding space compatible with both contactless and contact-based fingerprints for verification. F2P enhances ridge clarity and achieves superior recognition performance (AUC=0.999, EER=1.12%) over single-capture baselines (Verifinger, DeepPrint).
>
---
#### [new 168] MotionAdapter: Video Motion Transfer via Content-Aware Attention Customization
- **分类: cs.CV**

- **简介: 该论文提出MotionAdapter，解决视频运动迁移问题。通过分离运动与外观，并自适应调整运动以匹配目标内容，提升运动迁移的准确性和一致性。**

- **链接: [https://arxiv.org/pdf/2601.01955v1](https://arxiv.org/pdf/2601.01955v1)**

> **作者:** Zhexin Zhang; Yifeng Zhu; Yangyang Xu; Long Chen; Yong Du; Shengfeng He; Jun Yu
>
> **摘要:** Recent advances in diffusion-based text-to-video models, particularly those built on the diffusion transformer architecture, have achieved remarkable progress in generating high-quality and temporally coherent videos. However, transferring complex motions between videos remains challenging. In this work, we present MotionAdapter, a content-aware motion transfer framework that enables robust and semantically aligned motion transfer within DiT-based T2V models. Our key insight is that effective motion transfer requires \romannumeral1) explicit disentanglement of motion from appearance and \romannumeral 2) adaptive customization of motion to target content. MotionAdapter first isolates motion by analyzing cross-frame attention within 3D full-attention modules to extract attention-derived motion fields. To bridge the semantic gap between reference and target videos, we further introduce a DINO-guided motion customization module that rearranges and refines motion fields based on content correspondences. The customized motion field is then used to guide the DiT denoising process, ensuring that the synthesized video inherits the reference motion while preserving target appearance and semantics. Extensive experiments demonstrate that MotionAdapter outperforms state-of-the-art methods in both qualitative and quantitative evaluations. Moreover, MotionAdapter naturally supports complex motion transfer and motion editing tasks such as zooming.
>
---
#### [new 169] Mitigating Longitudinal Performance Degradation in Child Face Recognition Using Synthetic Data
- **分类: cs.CV**

- **简介: 该论文属于儿童人脸识别任务，旨在解决因面部快速生长导致的识别性能下降问题。通过引入合成数据提升模型的时序鲁棒性，实验表明合成数据能有效降低错误率。**

- **链接: [https://arxiv.org/pdf/2601.01689v1](https://arxiv.org/pdf/2601.01689v1)**

> **作者:** Afzal Hossain; Stephanie Schuckers
>
> **摘要:** Longitudinal face recognition in children remains challenging due to rapid and nonlinear facial growth, which causes template drift and increasing verification errors over time. This work investigates whether synthetic face data can act as a longitudinal stabilizer by improving temporal robustness of child face recognition models. Using an identity disjoint protocol on the Young Face Aging (YFA) dataset, we evaluate three settings: (i) pretrained MagFace embeddings without dataset specific fine-tuning, (ii) MagFace fine-tuned using authentic training faces only, and (iii) MagFace fine-tuned using a combination of authentic and synthetically generated training faces. Synthetic data is generated using StyleGAN2 ADA and incorporated exclusively within the training identities; a post generation filtering step is applied to mitigate identity leakage and remove artifact affected samples. Experimental results across enrollment verification gaps from 6 to 36 months show that synthetic-augmented fine tuning substantially reduces error rates relative to both the pretrained baseline and real only fine tuning. These findings provide a risk aware assessment of synthetic augmentation for improving identity persistence in pediatric face recognition.
>
---
#### [new 170] DVGBench: Implicit-to-Explicit Visual Grounding Benchmark in UAV Imagery with Large Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文提出DVGBench，一个用于无人机图像的隐式视觉定位基准，解决现有数据集依赖显式描述的问题。通过引入I2E-CoT方法提升模型隐式推理能力。**

- **链接: [https://arxiv.org/pdf/2601.00998v1](https://arxiv.org/pdf/2601.00998v1)**

> **作者:** Yue Zhou; Jue Chen; Zilun Zhang; Penghui Huang; Ran Ding; Zhentao Zou; PengFei Gao; Yuchen Wei; Ke Li; Xue Yang; Xue Jiang; Hongxin Yang; Jonathan Li
>
> **备注:** 20 pages, 17 figures
>
> **摘要:** Remote sensing (RS) large vision-language models (LVLMs) have shown strong promise across visual grounding (VG) tasks. However, existing RS VG datasets predominantly rely on explicit referring expressions-such as relative position, relative size, and color cues-thereby constraining performance on implicit VG tasks that require scenario-specific domain knowledge. This article introduces DVGBench, a high-quality implicit VG benchmark for drones, covering six major application scenarios: traffic, disaster, security, sport, social activity, and productive activity. Each object provides both explicit and implicit queries. Based on the dataset, we design DroneVG-R1, an LVLM that integrates the novel Implicit-to-Explicit Chain-of-Thought (I2E-CoT) within a reinforcement learning paradigm. This enables the model to take advantage of scene-specific expertise, converting implicit references into explicit ones and thus reducing grounding difficulty. Finally, an evaluation of mainstream models on both explicit and implicit VG tasks reveals substantial limitations in their reasoning capabilities. These findings provide actionable insights for advancing the reasoning capacity of LVLMs for drone-based agents. The code and datasets will be released at https://github.com/zytx121/DVGBench
>
---
#### [new 171] MambaFormer: Token-Level Guided Routing Mixture-of-Experts for Accurate and Efficient Clinical Assistance
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于医疗问答任务，旨在解决大模型在临床应用中的计算成本与效率矛盾。提出MambaFormer框架，通过动态路由选择不同专家模型，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.01260v1](https://arxiv.org/pdf/2601.01260v1)**

> **作者:** Hamad Khan; Saddam Hussain Khan
>
> **备注:** 28 Pages, Tables 12, Figure 09
>
> **摘要:** The deployment of large language models (LLMs) in real-world clinical applications is constrained by the fundamental trade-off between computational cost and the efficiency of linear-time models. To address this, we propose an LLM-based MambaFormer hybrid Mixture-of-Experts (MoE) framework for efficient medical question-answering (QA) and clinical assistance. The MambaFormer employs a lightweight gating mechanism that performs token-level dynamic routing to a customized Transformer expert (ET5) for short, complex queries or to a State Space Model expert (EMamba) for long, high-throughput sequences. The customized EMamba and ET5 models are tailored to accommodate input sequence dimensionality, embedding structure, sequence length, and target-specific output heads, and are fine-tuned through transfer learning on a new, custom-designed DentalQA dataset. Moreover, intelligent routing decisions are driven by the contextual complexity of token embeddings, normalized sequence length, and domain-aware features, thereby enforcing a Pareto-optimal trade-off between inference latency and prediction accuracy. Furthermore, a novel utility-guided multi-objective loss jointly optimizes decisions, router parameters, routing behavior, expert utilization, and computational cost by adaptively regulating token-level expert activation. Finally, the proposed MambaFormer is cross-validated (holdout) for medical QA on the new, custom-designed DentalQA and PubMedQA datasets and compared with state-of-the-art techniques. The proposed MambaFormer outperforms (BERTScore = 0.9180) with ultra-low latency (0.077 s), delivering a 24.4 speedup over T5-Large and establishing a scalable solution for resource-constrained clinical deployment.
>
---
#### [new 172] Nodule-DETR: A Novel DETR Architecture with Frequency-Channel Attention for Ultrasound Thyroid Nodule Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像检测任务，旨在解决超声甲状腺结节检测中的低对比度和边界模糊问题。提出Nodule-DETR模型，引入三个创新模块提升检测性能。**

- **链接: [https://arxiv.org/pdf/2601.01908v1](https://arxiv.org/pdf/2601.01908v1)**

> **作者:** Jingjing Wang; Qianglin Liu; Zhuo Xiao; Xinning Yao; Bo Liu; Lu Li; Lijuan Niu; Fugen Zhou
>
> **摘要:** Thyroid cancer is the most common endocrine malignancy, and its incidence is rising globally. While ultrasound is the preferred imaging modality for detecting thyroid nodules, its diagnostic accuracy is often limited by challenges such as low image contrast and blurred nodule boundaries. To address these issues, we propose Nodule-DETR, a novel detection transformer (DETR) architecture designed for robust thyroid nodule detection in ultrasound images. Nodule-DETR introduces three key innovations: a Multi-Spectral Frequency-domain Channel Attention (MSFCA) module that leverages frequency analysis to enhance features of low-contrast nodules; a Hierarchical Feature Fusion (HFF) module for efficient multi-scale integration; and Multi-Scale Deformable Attention (MSDA) to flexibly capture small and irregularly shaped nodules. We conducted extensive experiments on a clinical dataset of real-world thyroid ultrasound images. The results demonstrate that Nodule-DETR achieves state-of-the-art performance, outperforming the baseline model by a significant margin of 0.149 in mAP@0.5:0.95. The superior accuracy of Nodule-DETR highlights its significant potential for clinical application as an effective tool in computer-aided thyroid diagnosis. The code of work is available at https://github.com/wjj1wjj/Nodule-DETR.
>
---
#### [new 173] ShadowGS: Shadow-Aware 3D Gaussian Splatting for Satellite Imagery
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决多时相卫星影像中阴影不一致问题。提出ShadowGS框架，通过物理渲染和高效光线投射，实现几何一致的阴影建模与精确重建。**

- **链接: [https://arxiv.org/pdf/2601.00939v1](https://arxiv.org/pdf/2601.00939v1)**

> **作者:** Feng Luo; Hongbo Pan; Xiang Yang; Baoyu Jiang; Fengqing Liu; Tao Huang
>
> **摘要:** 3D Gaussian Splatting (3DGS) has emerged as a novel paradigm for 3D reconstruction from satellite imagery. However, in multi-temporal satellite images, prevalent shadows exhibit significant inconsistencies due to varying illumination conditions. To address this, we propose ShadowGS, a novel framework based on 3DGS. It leverages a physics-based rendering equation from remote sensing, combined with an efficient ray marching technique, to precisely model geometrically consistent shadows while maintaining efficient rendering. Additionally, it effectively disentangles different illumination components and apparent attributes in the scene. Furthermore, we introduce a shadow consistency constraint that significantly enhances the geometric accuracy of 3D reconstruction. We also incorporate a novel shadow map prior to improve performance with sparse-view inputs. Extensive experiments demonstrate that ShadowGS outperforms current state-of-the-art methods in shadow decoupling accuracy, 3D reconstruction precision, and novel view synthesis quality, with only a few minutes of training. ShadowGS exhibits robust performance across various settings, including RGB, pansharpened, and sparse-view satellite inputs.
>
---
#### [new 174] Sim2Real SAR Image Restoration: Metadata-Driven Models for Joint Despeckling and Sidelobes Reduction
- **分类: eess.IV; cs.CV; eess.SP**

- **简介: 该论文属于SAR图像修复任务，旨在解决斑点噪声和旁瓣问题。通过联合训练神经网络，实现同时去斑和抑制旁瓣，并利用元数据提升效果。**

- **链接: [https://arxiv.org/pdf/2601.01541v1](https://arxiv.org/pdf/2601.01541v1)**

> **作者:** Antoine De Paepe; Pascal Nguyen; Michael Mabelle; Cédric Saleun; Antoine Jouadé; Jean-Christophe Louvigne
>
> **备注:** Accepted at the Conference on Artificial Intelligence for Defense (CAID), 2025, Rennes, France
>
> **摘要:** Synthetic aperture radar (SAR) provides valuable information about the Earth's surface under all weather and illumination conditions. However, the inherent phenomenon of speckle and the presence of sidelobes around bright targets pose challenges for accurate interpretation of SAR imagery. Most existing SAR image restoration methods address despeckling and sidelobes reduction as separate tasks. In this paper, we propose a unified framework that jointly performs both tasks using neural networks (NNs) trained on a realistic SAR simulated dataset generated with MOCEM. Inference can then be performed on real SAR images, demonstrating effective simulation to real (Sim2Real) transferability. Additionally, we incorporate acquisition metadata as auxiliary input to the NNs, demonstrating improved restoration performance.
>
---
#### [new 175] GDRO: Group-level Reward Post-training Suitable for Diffusion Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于扩散模型的奖励对齐任务，解决在线强化学习效率低、依赖随机采样器和奖励黑客等问题。提出GDRO方法，实现高效、稳定的离线群体奖励优化。**

- **链接: [https://arxiv.org/pdf/2601.02036v1](https://arxiv.org/pdf/2601.02036v1)**

> **作者:** Yiyang Wang; Xi Chen; Xiaogang Xu; Yu Liu; Hengshuang Zhao
>
> **摘要:** Recent advancements adopt online reinforcement learning (RL) from LLMs to text-to-image rectified flow diffusion models for reward alignment. The use of group-level rewards successfully aligns the model with the targeted reward. However, it faces challenges including low efficiency, dependency on stochastic samplers, and reward hacking. The problem is that rectified flow models are fundamentally different from LLMs: 1) For efficiency, online image sampling takes much more time and dominates the time of training. 2) For stochasticity, rectified flow is deterministic once the initial noise is fixed. Aiming at these problems and inspired by the effects of group-level rewards from LLMs, we design Group-level Direct Reward Optimization (GDRO). GDRO is a new post-training paradigm for group-level reward alignment that combines the characteristics of rectified flow models. Through rigorous theoretical analysis, we point out that GDRO supports full offline training that saves the large time cost for image rollout sampling. Also, it is diffusion-sampler-independent, which eliminates the need for the ODE-to-SDE approximation to obtain stochasticity. We also empirically study the reward hacking trap that may mislead the evaluation, and involve this factor in the evaluation using a corrected score that not only considers the original evaluation reward but also the trend of reward hacking. Extensive experiments demonstrate that GDRO effectively and efficiently improves the reward score of the diffusion model through group-wise offline optimization across the OCR and GenEval tasks, while demonstrating strong stability and robustness in mitigating reward hacking.
>
---
#### [new 176] OpenRT: An Open-Source Red Teaming Framework for Multimodal LLMs
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于AI安全任务，旨在解决MLLMs的安全评估问题。提出OpenRT框架，通过多种攻击方法系统评估模型安全性，揭示前沿模型的漏洞。**

- **链接: [https://arxiv.org/pdf/2601.01592v1](https://arxiv.org/pdf/2601.01592v1)**

> **作者:** Xin Wang; Yunhao Chen; Juncheng Li; Yixu Wang; Yang Yao; Tianle Gu; Jie Li; Yan Teng; Xingjun Ma; Yingchun Wang; Xia Hu
>
> **摘要:** The rapid integration of Multimodal Large Language Models (MLLMs) into critical applications is increasingly hindered by persistent safety vulnerabilities. However, existing red-teaming benchmarks are often fragmented, limited to single-turn text interactions, and lack the scalability required for systematic evaluation. To address this, we introduce OpenRT, a unified, modular, and high-throughput red-teaming framework designed for comprehensive MLLM safety evaluation. At its core, OpenRT architects a paradigm shift in automated red-teaming by introducing an adversarial kernel that enables modular separation across five critical dimensions: model integration, dataset management, attack strategies, judging methods, and evaluation metrics. By standardizing attack interfaces, it decouples adversarial logic from a high-throughput asynchronous runtime, enabling systematic scaling across diverse models. Our framework integrates 37 diverse attack methodologies, spanning white-box gradients, multi-modal perturbations, and sophisticated multi-agent evolutionary strategies. Through an extensive empirical study on 20 advanced models (including GPT-5.2, Claude 4.5, and Gemini 3 Pro), we expose critical safety gaps: even frontier models fail to generalize across attack paradigms, with leading models exhibiting average Attack Success Rates as high as 49.14%. Notably, our findings reveal that reasoning models do not inherently possess superior robustness against complex, multi-turn jailbreaks. By open-sourcing OpenRT, we provide a sustainable, extensible, and continuously maintained infrastructure that accelerates the development and standardization of AI safety.
>
---
#### [new 177] A Global Atlas of Digital Dermatology to Map Innovation and Disparities
- **分类: cs.DL; cs.AI; cs.CV**

- **简介: 该论文属于数据质量评估任务，旨在解决皮肤病学数据的创新性与代表性问题。通过构建SkinMap框架，分析数据新颖性、冗余度及覆盖缺口，揭示数据分布不均现象。**

- **链接: [https://arxiv.org/pdf/2601.00840v1](https://arxiv.org/pdf/2601.00840v1)**

> **作者:** Fabian Gröger; Simone Lionetti; Philippe Gottfrois; Alvaro Gonzalez-Jimenez; Lea Habermacher; Labelling Consortium; Ludovic Amruthalingam; Matthew Groh; Marc Pouly; Alexander A. Navarini
>
> **摘要:** The adoption of artificial intelligence in dermatology promises democratized access to healthcare, but model reliability depends on the quality and comprehensiveness of the data fueling these models. Despite rapid growth in publicly available dermatology images, the field lacks quantitative key performance indicators to measure whether new datasets expand clinical coverage or merely replicate what is already known. Here we present SkinMap, a multi-modal framework for the first comprehensive audit of the field's entire data basis. We unify the publicly available dermatology datasets into a single, queryable semantic atlas comprising more than 1.1 million images of skin conditions and quantify (i) informational novelty over time, (ii) dataset redundancy, and (iii) representation gaps across demographics and diagnoses. Despite exponential growth in dataset sizes, informational novelty across time has somewhat plateaued: Some clusters, such as common neoplasms on fair skin, are densely populated, while underrepresented skin types and many rare diseases remain unaddressed. We further identify structural gaps in coverage: Darker skin tones (Fitzpatrick V-VI) constitute only 5.8% of images and pediatric patients only 3.0%, while many rare diseases and phenotype combinations remain sparsely represented. SkinMap provides infrastructure to measure blind spots and steer strategic data acquisition toward undercovered regions of clinical space.
>
---
#### [new 178] Image Synthesis Using Spintronic Deep Convolutional Generative Adversarial Network
- **分类: physics.app-ph; cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决传统架构能耗高的问题，提出一种基于自旋电子学的DCGAN架构，实现高效图像合成。**

- **链接: [https://arxiv.org/pdf/2601.01441v1](https://arxiv.org/pdf/2601.01441v1)**

> **作者:** Saumya Gupta; Abhinandan; Venkatesh vadde; Bhaskaran Muralidharan; Abhishek Sharma
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** The computational requirements of generative adversarial networks (GANs) exceed the limit of conventional Von Neumann architectures, necessitating energy efficient alternatives such as neuromorphic spintronics. This work presents a hybrid CMOS-spintronic deep convolutional generative adversarial network (DCGAN) architecture for synthetic image generation. The proposed generative vision model approach follows the standard framework, leveraging generator and discriminators adversarial training with our designed spintronics hardware for deconvolution, convolution, and activation layers of the DCGAN architecture. To enable hardware aware spintronic implementation, the generator's deconvolution layers are restructured as zero padded convolution, allowing seamless integration with a 6-bit skyrmion based synapse in a crossbar, without compromising training performance. Nonlinear activation functions are implemented using a hybrid CMOS domain wall based Rectified linear unit (ReLU) and Leaky ReLU units. Our proposed tunable Leaky ReLU employs domain wall position coded, continuous resistance states and a piecewise uniaxial parabolic anisotropy profile with a parallel MTJ readout, exhibiting energy consumption of 0.192 pJ. Our spintronic DCGAN model demonstrates adaptability across both grayscale and colored datasets, achieving Fr'echet Inception Distances (FID) of 27.5 for the Fashion MNIST and 45.4 for Anime Face datasets, with testing energy (training energy) of 4.9 nJ (14.97~nJ/image) and 24.72 nJ (74.7 nJ/image).
>
---
#### [new 179] Seamlessly Natural: Image Stitching with Natural Appearance Preservation
- **分类: eess.IV; cs.AI; cs.CV; cs.GR; eess.SP**

- **简介: 该论文属于图像拼接任务，解决双相机场景下的结构失真问题。提出SENA方法，通过几何驱动策略提升拼接的结构保真度和视觉真实感。**

- **链接: [https://arxiv.org/pdf/2601.01257v1](https://arxiv.org/pdf/2601.01257v1)**

> **作者:** Gaetane Lorna N. Tchana; Damaris Belle M. Fotso; Antonio Hendricks; Christophe Bobda
>
> **摘要:** This paper introduces SENA (SEamlessly NAtural), a geometry-driven image stitching approach that prioritizes structural fidelity in challenging real-world scenes characterized by parallax and depth variation. Conventional image stitching relies on homographic alignment, but this rigid planar assumption often fails in dual-camera setups with significant scene depth, leading to distortions such as visible warps and spherical bulging. SENA addresses these fundamental limitations through three key contributions. First, we propose a hierarchical affine-based warping strategy, combining global affine initialization with local affine refinement and smooth free-form deformation. This design preserves local shape, parallelism, and aspect ratios, thereby avoiding the hallucinated structural distortions commonly introduced by homography-based models. Second, we introduce a geometry-driven adequate zone detection mechanism that identifies parallax-minimized regions directly from the disparity consistency of RANSAC-filtered feature correspondences, without relying on semantic segmentation. Third, building upon this adequate zone, we perform anchor-based seamline cutting and segmentation, enforcing a one-to-one geometric correspondence across image pairs by construction, which effectively eliminates ghosting, duplication, and smearing artifacts in the final panorama. Extensive experiments conducted on challenging datasets demonstrate that SENA achieves alignment accuracy comparable to leading homography-based methods, while significantly outperforming them in critical visual metrics such as shape preservation, texture integrity, and overall visual realism.
>
---
#### [new 180] Quantifying Local Strain Field and Deformation in Active Contraction of Bladder Using a Pretrained Transformer Model: A Speckle-Free Approach
- **分类: q-bio.TO; cs.AI; cs.CV**

- **简介: 该论文属于生物力学分析任务，旨在解决传统DIC方法需人工斑点的问题。研究采用无斑点的Transformer模型，准确量化膀胱收缩时的局部应变场。**

- **链接: [https://arxiv.org/pdf/2601.01315v1](https://arxiv.org/pdf/2601.01315v1)**

> **作者:** Alireza Asadbeygi; Anne M. Robertson; Yasutaka Tobe; Masoud Zamani; Sean D. Stocker; Paul Watton; Naoki Yoshimura; Simon C Watkins
>
> **摘要:** Accurate quantification of local strain fields during bladder contraction is essential for understanding the biomechanics of bladder micturition, in both health and disease. Conventional digital image correlation (DIC) methods have been successfully applied to various biological tissues; however, this approach requires artificial speckling, which can alter both passive and active properties of the tissue. In this study, we introduce a speckle-free framework for quantifying local strain fields using a state-of-the-art, zero-shot transformer model, CoTracker3. We utilized a custom-designed, portable isotonic biaxial apparatus compatible with multiphoton microscopy (MPM) to demonstrate this approach, successfully tracking natural bladder lumen textures without artificial markers. Benchmark tests validated the method's high pixel accuracy and low strain errors. Our framework effectively captured heterogeneous deformation patterns, despite complex folding and buckling, which conventional DIC often fails to track. Application to in vitro active bladder contractions in four rat specimens (n=4) revealed statistically significant anisotropy (p<0.01), with higher contraction longitudinally compared to circumferentially. Multiphoton microscopy further illustrated and confirmed heterogeneous morphological changes, such as large fold formation during active contraction. This non-invasive approach eliminates speckle-induced artifacts, enabling more physiologically relevant measurements, and has broad applicability for material testing of other biological and engineered systems.
>
---
#### [new 181] Crafting Adversarial Inputs for Large Vision-Language Models Using Black-Box Optimization
- **分类: cs.CR; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于安全攻击任务，旨在解决LVLMs在黑盒环境下易受对抗攻击的问题。通过ZO-SPSA方法实现高效且隐蔽的对抗输入生成。**

- **链接: [https://arxiv.org/pdf/2601.01747v1](https://arxiv.org/pdf/2601.01747v1)**

> **作者:** Jiwei Guan; Haibo Jin; Haohan Wang
>
> **备注:** EACL
>
> **摘要:** Recent advancements in Large Vision-Language Models (LVLMs) have shown groundbreaking capabilities across diverse multimodal tasks. However, these models remain vulnerable to adversarial jailbreak attacks, where adversaries craft subtle perturbations to bypass safety mechanisms and trigger harmful outputs. Existing white-box attacks methods require full model accessibility, suffer from computing costs and exhibit insufficient adversarial transferability, making them impractical for real-world, black-box settings. To address these limitations, we propose a black-box jailbreak attack on LVLMs via Zeroth-Order optimization using Simultaneous Perturbation Stochastic Approximation (ZO-SPSA). ZO-SPSA provides three key advantages: (i) gradient-free approximation by input-output interactions without requiring model knowledge, (ii) model-agnostic optimization without the surrogate model and (iii) lower resource requirements with reduced GPU memory consumption. We evaluate ZO-SPSA on three LVLMs, including InstructBLIP, LLaVA and MiniGPT-4, achieving the highest jailbreak success rate of 83.0% on InstructBLIP, while maintaining imperceptible perturbations comparable to white-box methods. Moreover, adversarial examples generated from MiniGPT-4 exhibit strong transferability to other LVLMs, with ASR reaching 64.18%. These findings underscore the real-world feasibility of black-box jailbreaks and expose critical weaknesses in the safety mechanisms of current LVLMs
>
---
#### [new 182] DST-Calib: A Dual-Path, Self-Supervised, Target-Free LiDAR-Camera Extrinsic Calibration Network
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于LiDAR-相机外参标定任务，解决传统方法依赖标定目标和静态场景的问题。提出一种自监督、双路径的在线标定网络，提升泛化性和适应性。**

- **链接: [https://arxiv.org/pdf/2601.01188v1](https://arxiv.org/pdf/2601.01188v1)**

> **作者:** Zhiwei Huang; Yanwei Fu; Yi Zhou; Xieyuanli Chen; Qijun Chen; Rui Fan
>
> **摘要:** LiDAR-camera extrinsic calibration is essential for multi-modal data fusion in robotic perception systems. However, existing approaches typically rely on handcrafted calibration targets (e.g., checkerboards) or specific, static scene types, limiting their adaptability and deployment in real-world autonomous and robotic applications. This article presents the first self-supervised LiDAR-camera extrinsic calibration network that operates in an online fashion and eliminates the need for specific calibration targets. We first identify a significant generalization degradation problem in prior methods, caused by the conventional single-sided data augmentation strategy. To overcome this limitation, we propose a novel double-sided data augmentation technique that generates multi-perspective camera views using estimated depth maps, thereby enhancing robustness and diversity during training. Built upon this augmentation strategy, we design a dual-path, self-supervised calibration framework that reduces the dependence on high-precision ground truth labels and supports fully adaptive online calibration. Furthermore, to improve cross-modal feature association, we replace the traditional dual-branch feature extraction design with a difference map construction process that explicitly correlates LiDAR and camera features. This not only enhances calibration accuracy but also reduces model complexity. Extensive experiments conducted on five public benchmark datasets, as well as our own recorded dataset, demonstrate that the proposed method significantly outperforms existing approaches in terms of generalizability.
>
---
#### [new 183] An Explainable Agentic AI Framework for Uncertainty-Aware and Abstention-Enabled Acute Ischemic Stroke Imaging Decisions
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于急性缺血性卒中影像诊断任务，旨在解决AI模型缺乏不确定性感知和拒判能力的问题。提出一种可解释的智能体框架，实现安全、透明的决策支持。**

- **链接: [https://arxiv.org/pdf/2601.01008v1](https://arxiv.org/pdf/2601.01008v1)**

> **作者:** Md Rashadul Islam
>
> **备注:** Preprint. Conceptual and exploratory framework focusing on uncertainty-aware and abstention-enabled decision support for acute ischemic stroke imaging
>
> **摘要:** Artificial intelligence models have shown strong potential in acute ischemic stroke imaging, particularly for lesion detection and segmentation using computed tomography and magnetic resonance imaging. However, most existing approaches operate as black box predictors, producing deterministic outputs without explicit uncertainty awareness or structured mechanisms to abstain under ambiguous conditions. This limitation raises serious safety and trust concerns in high risk emergency radiology settings. In this paper, we propose an explainable agentic AI framework for uncertainty aware and abstention enabled decision support in acute ischemic stroke imaging. The framework follows a modular agentic pipeline in which a perception agent performs lesion aware image analysis, an uncertainty estimation agent computes slice level predictive reliability, and a decision agent determines whether to issue a prediction or abstain based on predefined uncertainty thresholds. Unlike prior stroke imaging systems that primarily focus on improving segmentation or classification accuracy, the proposed framework explicitly prioritizes clinical safety, transparency, and clinician aligned decision behavior. Qualitative and case based analyses across representative stroke imaging scenarios demonstrate that uncertainty driven abstention naturally emerges in diagnostically ambiguous regions and low information slices. The framework further integrates visual explanation mechanisms to support both predictive and abstention decisions, addressing a key limitation of existing uncertainty aware medical imaging systems. Rather than introducing a new performance benchmark, this work presents agentic control, uncertainty awareness, and selective abstention as essential design principles for developing safe and trustworthy medical imaging AI systems.
>
---
#### [new 184] DisCo-FLoc: Using Dual-Level Visual-Geometric Contrasts to Disambiguate Depth-Aware Visual Floorplan Localization
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉楼图定位任务，旨在解决简约楼图中因重复结构导致的定位模糊问题。提出DisCo-FLoc方法，通过双级视觉-几何对比消除歧义，提升定位精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.01822v1](https://arxiv.org/pdf/2601.01822v1)**

> **作者:** Shiyong Meng; Tao Zou; Bolei Chen; Chaoxu Mu; Jianxin Wang
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Since floorplan data is readily available, long-term persistent, and robust to changes in visual appearance, visual Floorplan Localization (FLoc) has garnered significant attention. Existing methods either ingeniously match geometric priors or utilize sparse semantics to reduce FLoc uncertainty. However, they still suffer from ambiguous FLoc caused by repetitive structures within minimalist floorplans. Moreover, expensive but limited semantic annotations restrict their applicability. To address these issues, we propose DisCo-FLoc, which utilizes dual-level visual-geometric Contrasts to Disambiguate depth-aware visual Floc, without requiring additional semantic labels. Our solution begins with a ray regression predictor tailored for ray-casting-based FLoc, predicting a series of FLoc candidates using depth estimation expertise. In addition, a novel contrastive learning method with position-level and orientation-level constraints is proposed to strictly match depth-aware visual features with the corresponding geometric structures in the floorplan. Such matches can effectively eliminate FLoc ambiguity and select the optimal imaging pose from FLoc candidates. Exhaustive comparative studies on two standard visual Floc benchmarks demonstrate that our method outperforms the state-of-the-art semantic-based method, achieving significant improvements in both robustness and accuracy.
>
---
#### [new 185] Uncertainty-Calibrated Explainable AI for Fetal Ultrasound Plane Classification
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于胎儿超声平面分类任务，旨在解决领域漂移、图像噪声和概率校准问题，通过融合不确定性估计与可解释方法，提升模型在实际应用中的可靠性。**

- **链接: [https://arxiv.org/pdf/2601.00990v1](https://arxiv.org/pdf/2601.00990v1)**

> **作者:** Olaf Yunus Laitinen Imanov
>
> **备注:** 9 pages, 1 figure, 4 tables
>
> **摘要:** Fetal ultrasound standard-plane classification underpins reliable prenatal biometry and anomaly screening, yet real-world deployment is limited by domain shift, image noise, and poor calibration of predicted probabilities. This paper presents a practical framework for uncertainty-calibrated explainable AI in fetal plane classification. We synthesize uncertainty estimation methods (Monte Carlo dropout, deep ensembles, evidential learning, and conformal prediction) with post-hoc and uncertainty-aware explanations (Grad-CAM variants, LIME-style local surrogates, and uncertainty-weighted multi-resolution activation maps), and we map these components to a clinician-facing workflow. Using FETAL_PLANES_DB as a reference benchmark, we define a reporting protocol that couples accuracy with calibration and selective prediction, including expected calibration error, Brier score, coverage-risk curves, and structured error analysis with explanations. We also discuss integration points for quality control and human-in-the-loop review, where uncertainty flags trigger re-acquisition or expert confirmation. The goal is a reproducible, clinically aligned blueprint for building fetal ultrasound classifiers whose confidence and explanations remain trustworthy under noisy acquisition conditions.
>
---
#### [new 186] YODA: Yet Another One-step Diffusion-based Video Compressor
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于视频压缩任务，解决视频中时空相关性利用不足的问题。通过引入多尺度时序特征和线性DiT模型，提升压缩效果。**

- **链接: [https://arxiv.org/pdf/2601.01141v1](https://arxiv.org/pdf/2601.01141v1)**

> **作者:** Xingchen Li; Junzhe Zhang; Junqi Shi; Ming Lu; Zhan Ma
>
> **备注:** Code will be available at https://github.com/NJUVISION/YODA
>
> **摘要:** While one-step diffusion models have recently excelled in perceptual image compression, their application to video remains limited. Prior efforts typically rely on pretrained 2D autoencoders that generate per-frame latent representations independently, thereby neglecting temporal dependencies. We present YODA--Yet Another One-step Diffusion-based Video Compressor--which embeds multiscale features from temporal references for both latent generation and latent coding to better exploit spatial-temporal correlations for more compact representation, and employs a linear Diffusion Transformer (DiT) for efficient one-step denoising. YODA achieves state-of-the-art perceptual performance, consistently outperforming traditional and deep-learning baselines on LPIPS, DISTS, FID, and KID. Source code will be publicly available at https://github.com/NJUVISION/YODA.
>
---
#### [new 187] Flow Equivariant World Models: Memory for Partially Observed Dynamic Environments
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于世界建模任务，旨在解决部分观测动态环境中的表示问题。通过引入流等变世界模型，统一处理自我运动与外部物体运动，提升长期预测性能。**

- **链接: [https://arxiv.org/pdf/2601.01075v1](https://arxiv.org/pdf/2601.01075v1)**

> **作者:** Hansen Jin Lillemark; Benhao Huang; Fangneng Zhan; Yilun Du; Thomas Anderson Keller
>
> **备注:** 11 main text pages, 10 figures
>
> **摘要:** Embodied systems experience the world as 'a symphony of flows': a combination of many continuous streams of sensory input coupled to self-motion, interwoven with the dynamics of external objects. These streams obey smooth, time-parameterized symmetries, which combine through a precisely structured algebra; yet most neural network world models ignore this structure and instead repeatedly re-learn the same transformations from data. In this work, we introduce 'Flow Equivariant World Models', a framework in which both self-motion and external object motion are unified as one-parameter Lie group 'flows'. We leverage this unification to implement group equivariance with respect to these transformations, thereby providing a stable latent world representation over hundreds of timesteps. On both 2D and 3D partially observed video world modeling benchmarks, we demonstrate that Flow Equivariant World Models significantly outperform comparable state-of-the-art diffusion-based and memory-augmented world modeling architectures -- particularly when there are predictable world dynamics outside the agent's current field of view. We show that flow equivariance is particularly beneficial for long rollouts, generalizing far beyond the training horizon. By structuring world model representations with respect to internal and external motion, flow equivariance charts a scalable route to data efficient, symmetry-guided, embodied intelligence. Project link: https://flowequivariantworldmodels.github.io.
>
---
#### [new 188] SketchRodGS: Sketch-based Extraction of Slender Geometries for Animating Gaussian Splatting Scenes
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于几何提取任务，旨在从高斯飞溅场景中提取细长结构的折线表示。针对缺乏连接信息和噪声问题，提出基于用户草图和屏幕空间最短路径的方法。**

- **链接: [https://arxiv.org/pdf/2601.02072v1](https://arxiv.org/pdf/2601.02072v1)**

> **作者:** Haato Watanabe; Nobuyuki Umetani
>
> **备注:** Presented at SIGGRAPH Asia 2025 (Technical Communications). Best Technical Communications Award
>
> **摘要:** Physics simulation of slender elastic objects often requires discretization as a polyline. However, constructing a polyline from Gaussian splatting is challenging as Gaussian splatting lacks connectivity information and the configuration of Gaussian primitives contains much noise. This paper presents a method to extract a polyline representation of the slender part of the objects in a Gaussian splatting scene from the user's sketching input. Our method robustly constructs a polyline mesh that represents the slender parts using the screen-space shortest path analysis that can be efficiently solved using dynamic programming. We demonstrate the effectiveness of our approach in several in-the-wild examples.
>
---
#### [new 189] Simulations of MRI Guided and Powered Ferric Applicators for Tetherless Delivery of Therapeutic Interventions
- **分类: cs.RO; cs.CV; eess.SY**

- **简介: 该论文属于医学机器人领域，旨在解决MRI引导下无导线治疗设备的术前规划问题。通过构建计算平台，实现血管路径模拟与安全评估。**

- **链接: [https://arxiv.org/pdf/2601.00981v1](https://arxiv.org/pdf/2601.00981v1)**

> **作者:** Wenhui Chu; Khang Tran; Nikolaos V. Tsekos
>
> **备注:** 9 pages, 8 figures, published in ICBBB 2022
>
> **摘要:** Magnetic Resonance Imaging (MRI) is a well-established modality for pre-operative planning and is also explored for intra-operative guidance of procedures such as intravascular interventions. Among the experimental robot-assisted technologies, the magnetic field gradients of the MRI scanner are used to power and maneuver ferromagnetic applicators for accessing sites in the patient's body via the vascular network. In this work, we propose a computational platform for preoperative planning and modeling of MRI-powered applicators inside blood vessels. This platform was implemented as a two-way data and command pipeline that links the MRI scanner, the computational core, and the operator. The platform first processes multi-slice MR data to extract the vascular bed and then fits a virtual corridor inside the vessel. This corridor serves as a virtual fixture (VF), a forbidden region for the applicators to avoid vessel perforation or collision. The geometric features of the vessel centerline, the VF, and MRI safety compliance (dB/dt, max available gradient) are then used to generate magnetic field gradient waveforms. Different blood flow profiles can be user-selected, and those parameters are used for modeling the applicator's maneuvering. The modeling module further generates cues about whether the selected vascular path can be safely maneuvered. Given future experimental studies that require a real-time operation, the platform was implemented on the Qt framework (C/C++) with software modules performing specific tasks running on dedicated threads: PID controller, generation of VF, generation of MR gradient waveforms.
>
---
#### [new 190] XAI-MeD: Explainable Knowledge Guided Neuro-Symbolic Framework for Domain Generalization and Rare Class Detection in Medical Imaging
- **分类: cs.AI; cs.CV**

- **简介: 该论文提出XAIMeD框架，解决医学影像中的领域泛化和罕见类检测问题，通过结合临床知识与深度学习提升模型鲁棒性和可解释性。**

- **链接: [https://arxiv.org/pdf/2601.02008v1](https://arxiv.org/pdf/2601.02008v1)**

> **作者:** Midhat Urooj; Ayan Banerjee; Sandeep Gupta
>
> **备注:** Accepted at AAAI Bridge Program 2026
>
> **摘要:** Explainability domain generalization and rare class reliability are critical challenges in medical AI where deep models often fail under real world distribution shifts and exhibit bias against infrequent clinical conditions This paper introduces XAIMeD an explainable medical AI framework that integrates clinically accurate expert knowledge into deep learning through a unified neuro symbolic architecture XAIMeD is designed to improve robustness under distribution shift enhance rare class sensitivity and deliver transparent clinically aligned interpretations The framework encodes clinical expertise as logical connectives over atomic medical propositions transforming them into machine checkable class specific rules Their diagnostic utility is quantified through weighted feature satisfaction scores enabling a symbolic reasoning branch that complements neural predictions A confidence weighted fusion integrates symbolic and deep outputs while a Hunt inspired adaptive routing mechanism guided by Entropy Imbalance Gain EIG and Rare Class Gini mitigates class imbalance high intra class variability and uncertainty We evaluate XAIMeD across diverse modalities on four challenging tasks i Seizure Onset Zone SOZ localization from rs fMRI ii Diabetic Retinopathy grading across 6 multicenter datasets demonstrate substantial performance improvements including 6 percent gains in cross domain generalization and a 10 percent improved rare class F1 score far outperforming state of the art deep learning baselines Ablation studies confirm that the clinically grounded symbolic components act as effective regularizers ensuring robustness to distribution shifts XAIMeD thus provides a principled clinically faithful and interpretable approach to multimodal medical AI.
>
---
#### [new 191] Scale-aware Adaptive Supervised Network with Limited Medical Annotations
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决标注数据稀缺、标注差异大及多尺度特征整合不足的问题。提出SASNet，通过双分支结构和三项创新方法提升分割精度。**

- **链接: [https://arxiv.org/pdf/2601.01005v1](https://arxiv.org/pdf/2601.01005v1)**

> **作者:** Zihan Li; Dandan Shan; Yunxiang Li; Paul E. Kinahan; Qingqi Hong
>
> **备注:** Accepted by Pattern Recognition, 8 figures, 11 tables
>
> **摘要:** Medical image segmentation faces critical challenges in semi-supervised learning scenarios due to severe annotation scarcity requiring expert radiological knowledge, significant inter-annotator variability across different viewpoints and expertise levels, and inadequate multi-scale feature integration for precise boundary delineation in complex anatomical structures. Existing semi-supervised methods demonstrate substantial performance degradation compared to fully supervised approaches, particularly in small target segmentation and boundary refinement tasks. To address these fundamental challenges, we propose SASNet (Scale-aware Adaptive Supervised Network), a dual-branch architecture that leverages both low-level and high-level feature representations through novel scale-aware adaptive reweight mechanisms. Our approach introduces three key methodological innovations, including the Scale-aware Adaptive Reweight strategy that dynamically weights pixel-wise predictions using temporal confidence accumulation, the View Variance Enhancement mechanism employing 3D Fourier domain transformations to simulate annotation variability, and segmentation-regression consistency learning through signed distance map algorithms for enhanced boundary precision. These innovations collectively address the core limitations of existing semi-supervised approaches by integrating spatial, temporal, and geometric consistency principles within a unified optimization framework. Comprehensive evaluation across LA, Pancreas-CT, and BraTS datasets demonstrates that SASNet achieves superior performance with limited labeled data, surpassing state-of-the-art semi-supervised methods while approaching fully supervised performance levels. The source code for SASNet is available at https://github.com/HUANGLIZI/SASNet.
>
---
#### [new 192] CORE: Code-based Inverse Self-Training Framework with Graph Expansion for Virtual Agents
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于虚拟代理训练任务，解决行为克隆与强化学习的冲突问题。提出CORE框架，通过语义代码抽象和图扩展提升行为多样性，无需人工设计奖励函数。**

- **链接: [https://arxiv.org/pdf/2601.02201v1](https://arxiv.org/pdf/2601.02201v1)**

> **作者:** Keyu Wang; Bingchen Miao; Wendong Bu; Yu Wu; Juncheng Li; Shengyu Zhang; Wenqiao Zhang; Siliang Tang; Jun Xiao; Yueting Zhuang
>
> **备注:** 19 pages, 12 figures
>
> **摘要:** The development of Multimodal Virtual Agents has made significant progress through the integration of Multimodal Large Language Models. However, mainstream training paradigms face key challenges: Behavior Cloning is simple and effective through imitation but suffers from low behavioral diversity, while Reinforcement Learning is capable of discovering novel strategies through exploration but heavily relies on manually designed reward functions. To address the conflict between these two methods, we present CORE, a Code-based Inverse Self-Training Framework with Graph Expansion that bridges imitation and exploration, offering a novel training framework that promotes behavioral diversity while eliminating the reliance on manually reward design. Specifically, we introduce Semantic Code Abstraction to automatically infers reward functions from expert demonstrations without manual design. The inferred reward function, referred to as the Label Function, is executable code that verifies one key step within a task. Building on this, we propose Strategy Graph Expansion to enhance in-domain behavioral diversity, which constructs a multi-path graph called Strategy Graph that captures diverse valid solutions beyond expert demonstrations. Furthermore, we introduce Trajectory-Guided Extrapolation, which enriches out-of-domain behavioral diversity by utilizing both successful and failed trajectories to expand the task space. Experiments on Web and Android platforms demonstrate that CORE significantly improves both overall performance and generalization, highlighting its potential as a robust and generalizable training paradigm for building powerful virtual agents.
>
---
#### [new 193] T3C: Test-Time Tensor Compression with Consistency Guarantees
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出T3C，一种测试时张量压缩框架，解决模型部署中的精度与效率平衡问题。通过弹性分解和混合精度量化，实现可控制的压缩效果。**

- **链接: [https://arxiv.org/pdf/2601.01299v1](https://arxiv.org/pdf/2601.01299v1)**

> **作者:** Ismail Lamaakal; Chaymae Yahyati; Yassine Maleh; Khalid El Makkaoui; Ibrahim Ouahbi
>
> **摘要:** We present T3C, a train-once, test-time budget-conditioned compression framework that exposes rank and precision as a controllable deployment knob. T3C combines elastic tensor factorization (maintained up to a maximal rank) with rank-tied mixed-precision quantization and a lightweight controller that maps a latency/energy/size budget token to per-layer rank/bit assignments; the policy snaps to hardware-aligned profiles and is monotone in the budget. A fast, layerwise consistency certificate, computed from spectral proxies and activation statistics, upper-bounds logit drift and regularizes training, yielding a practical reliability signal with negligible overhead. On ImageNet-1k, T3C shifts the vision Pareto frontier: for ResNet-50 at matched accuracy (\leq 0.5% drop), p50 latency is 1.18ms with a 38MB model, outperforming PTQ-8b (1.44ms, 88MB); for ViT-B/16, T3C reaches 2.30ms p50 with 59MB, improving over strong PTQ/QAT baselines. A single T3C checkpoint therefore provides predictable, certificate-backed accuracy-latency-size trade-offs on demand across devices.
>
---
#### [new 194] Noise-Aware and Dynamically Adaptive Federated Defense Framework for SAR Image Target Recognition
- **分类: cs.CR; cs.CV; cs.LG**

- **简介: 该论文属于SAR图像目标识别任务，解决联邦学习中的后门攻击问题。提出NADAFD框架，通过多域分析和动态调整提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.00900v1](https://arxiv.org/pdf/2601.00900v1)**

> **作者:** Yuchao Hou; Zixuan Zhang; Jie Wang; Wenke Huang; Lianhui Liang; Di Wu; Zhiquan Liu; Youliang Tian; Jianming Zhu; Jisheng Dang; Junhao Dong; Zhongliang Guo
>
> **备注:** This work was supported in part by the National Key Research and Development Program of China under Grant 2021YFB3101100, in part by the National Natural Science Foundation of China under Grant 62272123, 42371470, and 42461057, in part by the Fundamental Research Program of Shanxi Province under Grant 202303021212164. Corresponding authors: Zhongliang Guo and Junhao Dong
>
> **摘要:** As a critical application of computational intelligence in remote sensing, deep learning-based synthetic aperture radar (SAR) image target recognition facilitates intelligent perception but typically relies on centralized training, where multi-source SAR data are uploaded to a single server, raising privacy and security concerns. Federated learning (FL) provides an emerging computational intelligence paradigm for SAR image target recognition, enabling cross-site collaboration while preserving local data privacy. However, FL confronts critical security risks, where malicious clients can exploit SAR's multiplicative speckle noise to conceal backdoor triggers, severely challenging the robustness of the computational intelligence model. To address this challenge, we propose NADAFD, a noise-aware and dynamically adaptive federated defense framework that integrates frequency-domain, spatial-domain, and client-behavior analyses to counter SAR-specific backdoor threats. Specifically, we introduce a frequency-domain collaborative inversion mechanism to expose cross-client spectral inconsistencies indicative of hidden backdoor triggers. We further design a noise-aware adversarial training strategy that embeds $Γ$-distributed speckle characteristics into mask-guided adversarial sample generation to enhance robustness against both backdoor attacks and SAR speckle noise. In addition, we present a dynamic health assessment module that tracks client update behaviors across training rounds and adaptively adjusts aggregation weights to mitigate evolving malicious contributions. Experiments on MSTAR and OpenSARShip datasets demonstrate that NADAFD achieves higher accuracy on clean test samples and a lower backdoor attack success rate on triggered inputs than existing federated backdoor defenses for SAR target recognition.
>
---
#### [new 195] AlignDrive: Aligned Lateral-Longitudinal Planning for End-to-End Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，解决路径与速度协调不足的问题。提出一种级联框架，将纵向规划基于路径，提升协同性和安全性。**

- **链接: [https://arxiv.org/pdf/2601.01762v1](https://arxiv.org/pdf/2601.01762v1)**

> **作者:** Yanhao Wu; Haoyang Zhang; Fei He; Rui Wu; Congpei Qiu; Liang Gao; Wei Ke; Tong Zhang
>
> **备注:** underreview
>
> **摘要:** End-to-end autonomous driving has rapidly progressed, enabling joint perception and planning in complex environments. In the planning stage, state-of-the-art (SOTA) end-to-end autonomous driving models decouple planning into parallel lateral and longitudinal predictions. While effective, this parallel design can lead to i) coordination failures between the planned path and speed, and ii) underutilization of the drive path as a prior for longitudinal planning, thus redundantly encoding static information. To address this, we propose a novel cascaded framework that explicitly conditions longitudinal planning on the drive path, enabling coordinated and collision-aware lateral and longitudinal planning. Specifically, we introduce a path-conditioned formulation that explicitly incorporates the drive path into longitudinal planning. Building on this, the model predicts longitudinal displacements along the drive path rather than full 2D trajectory waypoints. This design simplifies longitudinal reasoning and more tightly couples it with lateral planning. Additionally, we introduce a planning-oriented data augmentation strategy that simulates rare safety-critical events, such as vehicle cut-ins, by adding agents and relabeling longitudinal targets to avoid collision. Evaluated on the challenging Bench2Drive benchmark, our method sets a new SOTA, achieving a driving score of 89.07 and a success rate of 73.18%, demonstrating significantly improved coordination and safety
>
---
#### [new 196] Real-Time Human Detection for Aerial Captured Video Sequences via Deep Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于视频中人体检测任务，旨在解决传统方法在动态环境下的不足。通过结合光流与深度模型，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.00391v1](https://arxiv.org/pdf/2601.00391v1)**

> **作者:** Nouar AlDahoul; Aznul Qalid Md Sabri; Ali Mohammed Mansoor
>
> **摘要:** Human detection in videos plays an important role in various real-life applications. Most traditional approaches depend on utilizing handcrafted features, which are problem-dependent and optimal for specific tasks. Moreover, they are highly susceptible to dynamical events such as illumination changes, camera jitter, and variations in object sizes. On the other hand, the proposed feature learning approaches are cheaper and easier because highly abstract and discriminative features can be produced automatically without the need of expert knowledge. In this paper, we utilize automatic feature learning methods, which combine optical flow and three different deep models (i.e., supervised convolutional neural network (S-CNN), pretrained CNN feature extractor, and hierarchical extreme learning machine) for human detection in videos captured using a nonstatic camera on an aerial platform with varying altitudes. The models are trained and tested on the publicly available and highly challenging UCF-ARG aerial dataset. The comparison between these models in terms of training, testing accuracy, and learning speed is analyzed. The performance evaluation considers five human actions (digging, waving, throwing, walking, and running). Experimental results demonstrated that the proposed methods are successful for the human detection task. The pretrained CNN produces an average accuracy of 98.09%. S-CNN produces an average accuracy of 95.6% with softmax and 91.7% with Support Vector Machines (SVM). H-ELM has an average accuracy of 95.9%. Using a normal Central Processing Unit (CPU), H-ELM's training time takes 445 seconds. Learning in S-CNN takes 770 seconds with a high-performance Graphical Processing Unit (GPU).
>
---
#### [new 197] An Energy-Efficient Smart Bus Transport Management System with Blind-Spot Collision Detection Ability
- **分类: eess.SY; cs.CV**

- **简介: 该论文属于智能交通管理系统任务，旨在解决公交准点率低、安全风险高和能源消耗大的问题。通过引入深度学习和物联网技术，提升公交安全与效率。**

- **链接: [https://arxiv.org/pdf/2601.01274v1](https://arxiv.org/pdf/2601.01274v1)**

> **作者:** Md. Sadman Haque; Zobaer Ibn Razzaque; Robiul Awoul Robin; Fahim Hafiz; Riasat Azim
>
> **备注:** 29 pages, 11 figures
>
> **摘要:** Public bus transport systems in developing countries often suffer from a lack of real-time location updates and for users, making commuting inconvenient and unreliable for passengers. Furthermore, stopping at undesired locations rather than designated bus stops creates safety risks and contributes to roadblocks, often causing traffic congestion. Additionally, issues such as blind spots, along with a lack of following traffic laws, increase the chances of accidents. In this work, we address these challenges by proposing a smart public bus system along with intelligent bus stops that enhance safety, efficiency, and sustainability. Our approach includes a deep learning-based blind-spot warning system to help drivers avoid accidents with automated bus-stop detection to accurately identify bus stops, improving transit efficiency. We also introduce IoT-based solar-powered smart bus stops that show real-time passenger counts, along with an RFID-based card system to track where passengers board and exit. A smart door system ensures safer and more organised boarding, while real-time bus tracking keeps passengers informed. To connect all these features, we use an HTTP-based server for seamless communication between the interconnected network systems. Our proposed system demonstrated approximately 99% efficiency in real-time blind spot detection while stopping precisely at the bus stops. Furthermore, the server showed real-time location updates both to the users and at the bus stops, enhancing commuting efficiency. The proposed energy-efficient bus stop demonstrated 12.71kWh energy saving, promoting sustainable architecture. Full implementation and source code are available at: https://github.com/sadman-adib/MoveMe-IoT
>
---
#### [new 198] MM-Sonate: Multimodal Controllable Audio-Video Generation with Zero-Shot Voice Cloning
- **分类: cs.SD; cs.AI; cs.CV; cs.MM; eess.AS**

- **简介: 该论文属于多模态生成任务，旨在解决音频视频同步生成中语音控制与零样本语音克隆的问题。提出MM-Sonate框架，实现精准语言对齐和高质量语音克隆。**

- **链接: [https://arxiv.org/pdf/2601.01568v1](https://arxiv.org/pdf/2601.01568v1)**

> **作者:** Chunyu Qiang; Jun Wang; Xiaopeng Wang; Kang Yin; Yuxin Guo; Xijuan Zeng; Nan Li; Zihan Li; Yuzhe Liang; Ziyu Zhang; Teng Ma; Yushen Chen; Zhongliang Liu; Feng Deng; Chen Zhang; Pengfei Wan
>
> **摘要:** Joint audio-video generation aims to synthesize synchronized multisensory content, yet current unified models struggle with fine-grained acoustic control, particularly for identity-preserving speech. Existing approaches either suffer from temporal misalignment due to cascaded generation or lack the capability to perform zero-shot voice cloning within a joint synthesis framework. In this work, we present MM-Sonate, a multimodal flow-matching framework that unifies controllable audio-video joint generation with zero-shot voice cloning capabilities. Unlike prior works that rely on coarse semantic descriptions, MM-Sonate utilizes a unified instruction-phoneme input to enforce strict linguistic and temporal alignment. To enable zero-shot voice cloning, we introduce a timbre injection mechanism that effectively decouples speaker identity from linguistic content. Furthermore, addressing the limitations of standard classifier-free guidance in multimodal settings, we propose a noise-based negative conditioning strategy that utilizes natural noise priors to significantly enhance acoustic fidelity. Empirical evaluations demonstrate that MM-Sonate establishes new state-of-the-art performance in joint generation benchmarks, significantly outperforming baselines in lip synchronization and speech intelligibility, while achieving voice cloning fidelity comparable to specialized Text-to-Speech systems.
>
---
#### [new 199] MetaFormer-driven Encoding Network for Robust Medical Semantic Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在提升分割精度同时降低计算成本。提出MFEnNet框架，结合MetaFormer和改进模块，实现高效准确的医学图像分割。**

- **链接: [https://arxiv.org/pdf/2601.00922v1](https://arxiv.org/pdf/2601.00922v1)**

> **作者:** Le-Anh Tran; Chung Nguyen Tran; Nhan Cach Dang; Anh Le Van Quoc; Jordi Carrabina; David Castells-Rufas; Minh Son Nguyen
>
> **备注:** 10 pages, 5 figures, MCT4SD 2025
>
> **摘要:** Semantic segmentation is crucial for medical image analysis, enabling precise disease diagnosis and treatment planning. However, many advanced models employ complex architectures, limiting their use in resource-constrained clinical settings. This paper proposes MFEnNet, an efficient medical image segmentation framework that incorporates MetaFormer in the encoding phase of the U-Net backbone. MetaFormer, an architectural abstraction of vision transformers, provides a versatile alternative to convolutional neural networks by transforming tokenized image patches into sequences for global context modeling. To mitigate the substantial computational cost associated with self-attention, the proposed framework replaces conventional transformer modules with pooling transformer blocks, thereby achieving effective global feature aggregation at reduced complexity. In addition, Swish activation is used to achieve smoother gradients and faster convergence, while spatial pyramid pooling is incorporated at the bottleneck to improve multi-scale feature extraction. Comprehensive experiments on different medical segmentation benchmarks demonstrate that the proposed MFEnNet approach attains competitive accuracy while significantly lowering computational cost compared to state-of-the-art models. The source code for this work is available at https://github.com/tranleanh/mfennet.
>
---
#### [new 200] Placenta Accreta Spectrum Detection using Multimodal Deep Learning
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于医学影像分析任务，旨在解决胎盘植入谱的早期诊断问题。通过融合MRI和超声数据，构建深度学习模型提升诊断准确性。**

- **链接: [https://arxiv.org/pdf/2601.00907v1](https://arxiv.org/pdf/2601.00907v1)**

> **作者:** Sumaiya Ali; Areej Alhothali; Sameera Albasri; Ohoud Alzamzami; Ahmed Abduljabbar; Muhammad Alwazzan
>
> **摘要:** Placenta Accreta Spectrum (PAS) is a life-threatening obstetric complication involving abnormal placental invasion into the uterine wall. Early and accurate prenatal diagnosis is essential to reduce maternal and neonatal risks. This study aimed to develop and validate a deep learning framework that enhances PAS detection by integrating multiple imaging modalities. A multimodal deep learning model was designed using an intermediate feature-level fusion architecture combining 3D Magnetic Resonance Imaging (MRI) and 2D Ultrasound (US) scans. Unimodal feature extractors, a 3D DenseNet121-Vision Transformer for MRI and a 2D ResNet50 for US, were selected after systematic comparative analysis. Curated datasets comprising 1,293 MRI and 1,143 US scans were used to train the unimodal models and paired samples of patient-matched MRI-US scans was isolated for multimodal model development and evaluation. On an independent test set, the multimodal fusion model achieved superior performance, with an accuracy of 92.5% and an Area Under the Receiver Operating Characteristic Curve (AUC) of 0.927, outperforming the MRI-only (82.5%, AUC 0.825) and US-only (87.5%, AUC 0.879) models. Integrating MRI and US features provides complementary diagnostic information, demonstrating strong potential to enhance prenatal risk assessment and improve patient outcomes.
>
---
#### [new 201] ShrimpXNet: A Transfer Learning Framework for Shrimp Disease Classification with Augmented Regularization, Adversarial Training, and Explainable AI
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于 shrimp disease classification 任务，旨在通过深度学习实现虾病的自动识别。研究采用迁移学习、对抗训练和可解释AI方法，提升模型准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2601.00832v1](https://arxiv.org/pdf/2601.00832v1)**

> **作者:** Israk Hasan Jone; D. M. Rafiun Bin Masud; Promit Sarker; Sayed Fuad Al Labib; Nazmul Islam; Farhad Billah
>
> **备注:** 8 Page, fugure 11
>
> **摘要:** Shrimp is one of the most widely consumed aquatic species globally, valued for both its nutritional content and economic importance. Shrimp farming represents a significant source of income in many regions; however, like other forms of aquaculture, it is severely impacted by disease outbreaks. These diseases pose a major challenge to sustainable shrimp production. To address this issue, automated disease classification methods can offer timely and accurate detection. This research proposes a deep learning-based approach for the automated classification of shrimp diseases. A dataset comprising 1,149 images across four disease classes was utilized. Six pretrained deep learning models, ResNet50, EfficientNet, DenseNet201, MobileNet, ConvNeXt-Tiny, and Xception were deployed and evaluated for performance. The images background was removed, followed by standardized preprocessing through the Keras image pipeline. Fast Gradient Sign Method (FGSM) was used for enhancing the model robustness through adversarial training. While advanced augmentation strategies, including CutMix and MixUp, were implemented to mitigate overfitting and improve generalization. To support interpretability, and to visualize regions of model attention, post-hoc explanation methods such as Grad-CAM, Grad-CAM++, and XGrad-CAM were applied. Exploratory results demonstrated that ConvNeXt-Tiny achieved the highest performance, attaining a 96.88% accuracy on the test dataset. After 1000 iterations, the 99% confidence interval for the model is [0.953,0.971].
>
---
#### [new 202] SPoRC-VIST: A Benchmark for Evaluating Generative Natural Narrative in Vision-Language Models
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于视觉语言生成任务，旨在提升模型生成自然对话式叙事的能力。通过构建数据集并 fine-tune 模型，解决传统评估指标不足的问题。**

- **链接: [https://arxiv.org/pdf/2601.01062v1](https://arxiv.org/pdf/2601.01062v1)**

> **作者:** Yunlin Zeng
>
> **备注:** 14 pages, 3 figures. Accepted to WVAQ 2026, WACV 2026
>
> **摘要:** Vision-Language Models (VLMs) have achieved remarkable success in descriptive tasks such as image captioning and visual question answering (VQA). However, their ability to generate engaging, long-form narratives -- specifically multi-speaker podcast dialogues -- remains under-explored and difficult to evaluate. Standard metrics like BLEU and ROUGE fail to capture the nuances of conversational naturalness, personality, and narrative flow, often rewarding safe, repetitive outputs over engaging storytelling. In this work, we present a novel pipeline for end-to-end visual podcast generation, and fine-tune a Qwen3-VL-32B model on a curated dataset of 4,000 image-dialogue pairs. Crucially, we use a synthetic-to-real training strategy: we train on high-quality podcast dialogues from the Structured Podcast Research Corpus (SPoRC) paired with synthetically generated imagery, and evaluate on real-world photo sequences from the Visual Storytelling Dataset (VIST). This rigorous setup tests the model's ability to generalize from synthetic training data to real-world visual domains. We propose a comprehensive evaluation framework that moves beyond textual overlap, and use AI-as-a-judge (Gemini 3 Pro, Claude Opus 4.5, GPT 5.2) and novel style metrics (average turn length, speaker switch rate) to assess quality. Our experiments demonstrate that our fine-tuned 32B model significantly outperforms a 235B base model in conversational naturalness ($>$80\% win rate) and narrative depth (+50\% turn length), while maintaining identical visual grounding capabilities (CLIPScore: 20.39).
>
---
#### [new 203] Hierarchical topological clustering
- **分类: cs.LG; cs.CV; physics.data-an; stat.ME; stat.ML**

- **简介: 该论文提出一种分层拓扑聚类算法，用于识别数据中的异常值和任意形状的簇。属于聚类任务，解决传统方法在复杂数据结构中失效的问题。**

- **链接: [https://arxiv.org/pdf/2601.00892v1](https://arxiv.org/pdf/2601.00892v1)**

> **作者:** Ana Carpio; Gema Duro
>
> **备注:** not peer reviewed, reviewed version to appear in Soft Computing
>
> **摘要:** Topological methods have the potential of exploring data clouds without making assumptions on their the structure. Here we propose a hierarchical topological clustering algorithm that can be implemented with any distance choice. The persistence of outliers and clusters of arbitrary shape is inferred from the resulting hierarchy. We demonstrate the potential of the algorithm on selected datasets in which outliers play relevant roles, consisting of images, medical and economic data. These methods can provide meaningful clusters in situations in which other techniques fail to do so.
>
---
#### [new 204] Dancing Points: Synthesizing Ballroom Dancing with Three-Point Inputs
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于动作生成任务，旨在解决舞者互动建模难题。通过三点轨迹输入，使用MLP网络直接预测舞者动作，实现高效、数据节约的舞蹈合成。**

- **链接: [https://arxiv.org/pdf/2601.02096v1](https://arxiv.org/pdf/2601.02096v1)**

> **作者:** Peizhuo Li; Sebastian Starke; Yuting Ye; Olga Sorkine-Hornung
>
> **摘要:** Ballroom dancing is a structured yet expressive motion category. Its highly diverse movement and complex interactions between leader and follower dancers make the understanding and synthesis challenging. We demonstrate that the three-point trajectory available from a virtual reality (VR) device can effectively serve as a dancer's motion descriptor, simplifying the modeling and synthesis of interplay between dancers' full-body motions down to sparse trajectories. Thanks to the low dimensionality, we can employ an efficient MLP network to predict the follower's three-point trajectory directly from the leader's three-point input for certain types of ballroom dancing, addressing the challenge of modeling high-dimensional full-body interaction. It also prevents our method from overfitting thanks to its compact yet explicit representation. By leveraging the inherent structure of the movements and carefully planning the autoregressive procedure, we show a deterministic neural network is able to translate three-point trajectories into a virtual embodied avatar, which is typically considered under-constrained and requires generative models for common motions. In addition, we demonstrate this deterministic approach generalizes beyond small, structured datasets like ballroom dancing, and performs robustly on larger, more diverse datasets such as LaFAN. Our method provides a computationally- and data-efficient solution, opening new possibilities for immersive paired dancing applications. Code and pre-trained models for this paper are available at https://peizhuoli.github.io/dancing-points.
>
---
#### [new 205] Neuro-Channel Networks: A Multiplication-Free Architecture by Biological Signal Transmission
- **分类: cs.LG; cs.AR; cs.CV**

- **简介: 该论文提出一种无需乘法的神经网络架构NCN，旨在解决AI对高成本硬件的依赖问题，通过模拟生物信号传输机制实现高效计算。**

- **链接: [https://arxiv.org/pdf/2601.02253v1](https://arxiv.org/pdf/2601.02253v1)**

> **作者:** Emrah Mete; Emin Erkan Korkmaz
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** The rapid proliferation of Deep Learning is increasingly constrained by its heavy reliance on high-performance hardware, particularly Graphics Processing Units (GPUs). These specialized accelerators are not only prohibitively expensive and energy-intensive but also suffer from significant supply scarcity, limiting the ubiquity of Artificial Intelligence (AI) deployment on edge devices. The core of this inefficiency stems from the standard artificial perceptron's dependence on intensive matrix multiplications. However, biological nervous systems achieve unparalleled efficiency without such arithmetic intensity; synaptic signal transmission is regulated by physical ion channel limits and chemical neurotransmitter levels rather than a process that can be analogous to arithmetic multiplication. Inspired by this biological mechanism, we propose Neuro-Channel Networks (NCN), a novel multiplication-free architecture designed to decouple AI from expensive hardware dependencies. In our model, weights are replaced with Channel Widths that physically limit the signal magnitude, while a secondary parameter acts as a Neurotransmitter to regulate Signal Transmission based on sign logic. The forward pass relies exclusively on addition, subtraction, and bitwise operations (minimum, sign), eliminating floating-point multiplication entirely. In this proof-of-concept study, we demonstrate that NCNs can solve non-linearly separable problems like XOR and the Majority function with 100% accuracy using standard backpropagation, proving their capability to form complex decision boundaries without multiplicative weights. This architecture offers a highly efficient alternative for next-generation neuromorphic hardware, paving the way for running complex models on commodity CPUs or ultra-low-power chips without relying on costly GPU clusters.
>
---
## 更新

#### [replaced 001] Towards Vision-Language Geo-Foundation Model: A Survey
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2406.09385v2](https://arxiv.org/pdf/2406.09385v2)**

> **作者:** Yue Zhou; Zhihang Zhong; Xue Yang
>
> **备注:** 18 pages, 4 figures
>
> **摘要:** Vision-Language Foundation Models (VLFMs) have made remarkable progress on various multimodal tasks, such as image captioning, image-text retrieval, visual question answering, and visual grounding. However, most methods rely on training with general image datasets, and the lack of geospatial data leads to poor performance on earth observation. Numerous geospatial image-text pair datasets and VLFMs fine-tuned on them have been proposed recently. These new approaches aim to leverage large-scale, multimodal geospatial data to build versatile intelligent models with diverse geo-perceptive capabilities, which we refer to as Vision-Language Geo-Foundation Models (VLGFMs). This paper thoroughly reviews VLGFMs, summarizing and analyzing recent developments in the field. In particular, we introduce the background and motivation behind the rise of VLGFMs, highlighting their unique research significance. Then, we systematically summarize the core technologies employed in VLGFMs, including data construction, model architectures, and applications of various multimodal geospatial tasks. Finally, we conclude with insights, issues, and discussions regarding future research directions. To the best of our knowledge, this is the first comprehensive literature review of VLGFMs. We keep tracing related works at https://github.com/zytx121/Awesome-VLGFM.
>
---
#### [replaced 002] Multimodal RewardBench 2: Evaluating Omni Reward Models for Interleaved Text and Image
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出MMRB2基准，用于评估处理文本与图像交织的通用奖励模型。解决多模态奖励模型评估不足的问题，通过四个任务和专家标注数据进行实验分析。**

- **链接: [https://arxiv.org/pdf/2512.16899v2](https://arxiv.org/pdf/2512.16899v2)**

> **作者:** Yushi Hu; Reyhane Askari-Hemmat; Melissa Hall; Emily Dinan; Luke Zettlemoyer; Marjan Ghazvininejad
>
> **备注:** Code and data available at https://github.com/facebookresearch/MMRB2
>
> **摘要:** Reward models (RMs) are essential for training large language models (LLMs), but remain underexplored for omni models that handle interleaved image and text sequences. We introduce Multimodal RewardBench 2 (MMRB2), the first comprehensive benchmark for reward models on multimodal understanding and (interleaved) generation. MMRB2 spans four tasks: text-to-image, image editing, interleaved generation, and multimodal reasoning ("thinking-with-images"), providing 1,000 expert-annotated preference pairs per task from 23 models and agents across 21 source tasks. MMRB2 is designed with: (1) practical but challenging prompts; (2) responses from state-of-the-art models and agents; and (3) preference pairs with strong human-expert consensus, curated via an ensemble filtering strategy. Using MMRB2, we study existing judges for each subtask, including multimodal LLM-as-a-judge and models trained with human preferences. The latest Gemini 3 Pro attains 75-80% accuracy. GPT-5 and Gemini 2.5 Pro reach 66-75% accuracy, compared to >90% for humans, yet surpass the widely used GPT-4o (59%). The best performing open-source model Qwen3-VL-32B achieves similar accuracies as Gemini 2.5 Flash (64%). We also show that MMRB2 performance strongly correlates with downstream task success using Best-of-N sampling and conduct an in-depth analysis that shows key areas to improve the reward models going forward.
>
---
#### [replaced 003] VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning for Image and Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.21059v4](https://arxiv.org/pdf/2412.21059v4)**

> **作者:** Jiazheng Xu; Yu Huang; Jiale Cheng; Yuanming Yang; Jiajun Xu; Yuan Wang; Wenbo Duan; Shen Yang; Qunlin Jin; Shurun Li; Jiayan Teng; Zhuoyi Yang; Wendi Zheng; Xiao Liu; Dan Zhang; Ming Ding; Xiaohan Zhang; Xiaotao Gu; Shiyu Huang; Minlie Huang; Jie Tang; Yuxiao Dong
>
> **备注:** 27 pages
>
> **摘要:** Visual generative models have achieved remarkable progress in synthesizing photorealistic images and videos, yet aligning their outputs with human preferences across critical dimensions remains a persistent challenge. Though reinforcement learning from human feedback offers promise for preference alignment, existing reward models for visual generation face limitations, including black-box scoring without interpretability and potentially resultant unexpected biases. We present VisionReward, a general framework for learning human visual preferences in both image and video generation. Specifically, we employ a hierarchical visual assessment framework to capture fine-grained human preferences, and leverages linear weighting to enable interpretable preference learning. Furthermore, we propose a multi-dimensional consistent strategy when using VisionReward as a reward model during preference optimization for visual generation. Experiments show that VisionReward can significantly outperform existing image and video reward models on both machine metrics and human evaluation. Notably, VisionReward surpasses VideoScore by 17.2% in preference prediction accuracy, and text-to-video models with VisionReward achieve a 31.6% higher pairwise win rate compared to the same models using VideoScore. All code and datasets are provided at https://github.com/THUDM/VisionReward.
>
---
#### [replaced 004] UNIDOC-BENCH: A Unified Benchmark for Document-Centric Multimodal RAG
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出UniDoc-Bench，一个用于文档导向多模态RAG的基准，解决多模态信息融合不足的问题，通过构建真实PDF数据集并生成多模态问答对进行评估。**

- **链接: [https://arxiv.org/pdf/2510.03663v3](https://arxiv.org/pdf/2510.03663v3)**

> **作者:** Xiangyu Peng; Can Qin; Zeyuan Chen; Ran Xu; Caiming Xiong; Chien-Sheng Wu
>
> **摘要:** Multimodal retrieval-augmented Generation (MM-RAG) is a key approach for applying large language models (LLMs) and agents to real-world knowledge bases, yet current evaluations are fragmented -- focusing on either text or images in isolation, or simplified multimodal setup, failing to capture document-centric multimodal use cases. In this paper, we introduce UniDoc-Bench, the first large-scale, realistic benchmark for MM-RAG built from $k$ real-world PDF pages across domains. Our pipeline extracts and links evidence from text, tables, and figures, then generates multimodal QA pairs spanning factual retrieval, comparison, summarization, and logical reasoning queries. To ensure reliability, all of QA pairs are validated by multiple human annotators and expert adjudication. UniDoc-Bench supports apples-to-apples comparison across four paradigms: 1) text-only, 2) image-only, 3) \emph{multimodal} text-image fusion and 4) multimodal joint retrieval -- under a unified protocol with standardized candidate pools, prompts, and evaluation metrics. UniDoc-Bench can also be used to evaluate Visual Question Answering (VQA) tasks. Our experiments show that multimodal text-image fusion RAG systems consistently outperform both unimodal and jointly multimodal embedding-based retrieval, indicating that neither text nor images alone are sufficient and that current multimodal embeddings remain inadequate. Beyond benchmarking, our analysis reveals when and how visual context complements textual evidence, uncovers systematic failure modes, and offers actionable guidance for developing more robust MM-RAG pipelines.
>
---
#### [replaced 005] SinBasis Networks: Matrix-Equivalent Feature Extraction for Wave-Like Optical Spectrograms
- **分类: cs.LG; cs.AI; cs.CV; physics.optics**

- **链接: [https://arxiv.org/pdf/2505.06275v3](https://arxiv.org/pdf/2505.06275v3)**

> **作者:** Yuzhou Zhu; Zheng Zhang; Ruyi Zhang; Liang Zhou
>
> **备注:** AAAI26 Poster
>
> **摘要:** Wave-like images--from attosecond streaking spectrograms to optical spectra, audio mel-spectrograms and periodic video frames--encode critical harmonic structures that elude conventional feature extractors. We propose a unified, matrix-equivalent framework that reinterprets convolution and attention as linear transforms on flattened inputs, revealing filter weights as basis vectors spanning latent feature subspaces. To infuse spectral priors we apply elementwise \(\sin(\cdot)\) mappings to each weight matrix. Embedding these transforms into CNN, ViT and Capsule architectures yields Sin-Basis Networks with heightened sensitivity to periodic motifs and built-in invariance to spatial shifts. Experiments on a diverse collection of wave-like image datasets--including 80,000 synthetic attosecond streaking spectrograms, thousands of Raman, photoluminescence and FTIR spectra, mel-spectrograms from AudioSet and cycle-pattern frames from Kinetics--demonstrate substantial gains in reconstruction accuracy, translational robustness and zero-shot cross-domain transfer. Theoretical analysis via matrix isomorphism and Mercer-kernel truncation quantifies how sinusoidal reparametrization enriches expressivity while preserving stability in data-scarce regimes. Sin-Basis Networks thus offer a lightweight, physics-informed approach to deep learning across all wave-form imaging modalities.
>
---
#### [replaced 006] Test-Time Modification: Inverse Domain Transformation for Robust Perception
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13454v2](https://arxiv.org/pdf/2512.13454v2)**

> **作者:** Arpit Jadon; Joshua Niemeijer; Yuki M. Asano
>
> **备注:** Preprint
>
> **摘要:** Generative foundation models contain broad visual knowledge and can produce diverse image variations, making them particularly promising for advancing domain generalization tasks. While they can be used for training data augmentation, synthesizing comprehensive target-domain variations remains slow, expensive, and incomplete. We propose an alternative: using diffusion models at test time to map target images back to the source distribution where the downstream model was trained. This approach requires only a source domain description, preserves the task model, and eliminates large-scale synthetic data generation. We demonstrate consistent improvements across segmentation, detection, and classification tasks under challenging environmental shifts in real-to-real domain generalization scenarios with unknown target distributions. Our analysis spans multiple generative and downstream models, including an ensemble variant for enhanced robustness. The method achieves substantial relative gains: 137% on BDD100K-Night, 68% on ImageNet-R, and 62% on DarkZurich.
>
---
#### [replaced 007] RAD: A Dataset and Benchmark for Real-Life Anomaly Detection with Robotic Observations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2410.00713v3](https://arxiv.org/pdf/2410.00713v3)**

> **作者:** Kaichen Zhou; Xinhai Chang; Taewhan Kim; Jiadong Zhang; Yang Cao; Chufei Peng; Fangneng Zhan; Hao Zhao; Hao Dong; Kai Ming Ting; Ye Zhu
>
> **摘要:** Anomaly detection is a core capability for robotic perception and industrial inspection, yet most existing benchmarks are collected under controlled conditions with fixed viewpoints and stable illumination, failing to reflect real deployment scenarios. We introduce RAD (Realistic Anomaly Detection), a robot-captured, multi-view dataset designed to stress pose variation, reflective materials, and viewpoint-dependent defect visibility. RAD covers 13 everyday object categories and four realistic defect types--scratched, missing, stained, and squeezed--captured from over 60 robot viewpoints per object under uncontrolled lighting. We benchmark a wide range of state-of-the-art approaches, including 2D feature-based methods, 3D reconstruction pipelines, and vision-language models (VLMs), under a pose-agnostic setting. Surprisingly, we find that mature 2D feature-embedding methods consistently outperform recent 3D and VLM-based approaches at the image level, while the performance gap narrows for pixel-level localization. Our analysis reveals that reflective surfaces, geometric symmetry, and sparse viewpoint coverage fundamentally limit current geometry-based and zero-shot methods. RAD establishes a challenging and realistic benchmark for robotic anomaly detection, highlighting critical open problems beyond controlled laboratory settings.
>
---
#### [replaced 008] A Survey on 3D Skeleton Based Person Re-Identification: Taxonomy, Advances, Challenges, and Interdisciplinary Prospects
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2401.15296v3](https://arxiv.org/pdf/2401.15296v3)**

> **作者:** Haocong Rao; Chunyan Miao
>
> **备注:** A curated collection of valuable resources is available at https://github.com/Kali-Hac/3D-SRID-Survey
>
> **摘要:** Person re-identification via 3D skeletons is an important emerging research area that attracts increasing attention within the pattern recognition community. With distinctive advantages across various application scenarios, numerous 3D skeleton based person re-identification (SRID) methods with diverse skeleton modeling and learning paradigms have been proposed in recent years. In this paper, we provide a comprehensive review and analysis of recent SRID advances. First of all, we define the SRID task and provide an overview of its origin and major advancements. Secondly, we formulate a systematic taxonomy that organizes existing methods into three categories centered on hand-crafted, sequence-based, and graph-based modeling. Then, we elaborate on the representative models along these three types with an illustration of foundational mechanisms. Meanwhile, we provide an overview of mainstream supervised, self-supervised, and unsupervised SRID learning paradigms and corresponding common methods. A thorough evaluation of state-of-the-art SRID methods is further conducted over various types of benchmarks and protocols to compare their effectiveness, efficiency, and key properties. Finally, we present the key challenges and prospects to advance future research, and highlight interdisciplinary applications of SRID with a case study.
>
---
#### [replaced 009] Enhancing Blind Video Quality Assessment with Rich Quality-aware Features
- **分类: eess.IV; cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2405.08745v2](https://arxiv.org/pdf/2405.08745v2)**

> **作者:** Wei Sun; Linhan Cao; Jun Jia; Zhichao Zhang; Zicheng Zhang; Xiongkuo Min; Guangtao Zhai
>
> **备注:** RQ-VQA won first place in the CVPR NTIRE 2024 Short-form UGC Video Quality Assessment Challenge
>
> **摘要:** Blind video quality assessment (BVQA) is a highly challenging task due to the intrinsic complexity of video content and visual distortions, especially given the high popularity of social media videos, which originate from a wide range of sources, and are often processed by various compression and enhancement algorithms. While recent BVQA and blind image quality assessment (BIQA) studies have made remarkable progress, their models typically perform well on the datasets they were trained on but generalize poorly to unseen videos, making them less effective for accurately evaluating the perceptual quality of diverse social media videos. In this paper, we propose Rich Quality-aware features enabled Video Quality Assessment (RQ-VQA), a simple yet effective method to enhance BVQA by leveraging rich quality-aware features extracted from off-the-shelf BIQA and BVQA models. Our approach exploits the expertise of existing quality assessment models within their trained domains to improve generalization. Specifically, we design a multi-source feature framework that integrates:(1) Learnable spatial features} from a base model fine-tuned on the target VQA dataset to capture domain-specific quality cues; (2) Temporal motion features from the fast pathway of SlowFast pre-trained on action recognition datasets to model motion-related distortions; (3) Spatial quality-aware features from BIQA models trained on diverse IQA datasets to enhance frame-level distortion representation; and (4) Spatiotemporal quality-aware features from a BVQA model trained on large-scale VQA datasets to jointly encode spatial structure and temporal dynamics. These features are concatenated and fed into a multi-layer perceptron (MLP) to regress them into quality scores. Experimental results demonstrate that our model achieves state-of-the-art performance on three public social media VQA datasets.
>
---
#### [replaced 010] Vision-Enhanced Large Language Models for High-Resolution Image Synthesis and Multimodal Data Interpretation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.12595v2](https://arxiv.org/pdf/2512.12595v2)**

> **作者:** Karthikeya KV
>
> **摘要:** This research introduces a transformative framework for integrating Vision-Enhanced Large Language Models (LLMs) with advanced transformer-based architectures to tackle challenges in high-resolution image synthesis and multimodal data interpretation. The proposed model incorporates a rectified flow mechanism that connects noise and data with linear paths, enabling efficient and high-quality generation. A bidirectional tokenization strategy is employed to seamlessly merge inputs from text, image, and video modalities, fostering a unified understanding across diverse data types. By embedding spatial-temporal features and leveraging a hybrid text-image sequence modeling approach, the framework achieves unparalleled fidelity in synthesized images and coherent multimodal representations. The architecture is optimized with a noise-aware learning algorithm, addressing discrepancies in noisy data distributions and improving generative performance under varying input conditions. Rigorous evaluations on benchmark datasets demonstrate a 25% increase in image resolution clarity and a 20% reduction in computational requirements compared to diffusion-based methods. Furthermore, the model exhibits robust scalability and adaptability, showcasing its potential in applications like autonomous systems, creative content generation, and advanced video analysis. This work underscores the role of vision-centric LLMs in redefining capabilities in computer vision and multimodal artificial intelligence.
>
---
#### [replaced 011] Data-Augmented Multimodal Feature Fusion for Multiclass Visual Recognition of Oral Cancer Lesions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21582v3](https://arxiv.org/pdf/2511.21582v3)**

> **作者:** Joy Naoum; Revana Salama; Ali Hamdi
>
> **摘要:** Oral cancer is frequently diagnosed at later stages due to its similarity to other lesions. Existing research on computer aided diagnosis has made progress using deep learning; however, most approaches remain limited by small, imbalanced datasets and a dependence on single-modality features, which restricts model generalization in real-world clinical settings. To address these limitations, this study proposes a novel data-augmentation driven multimodal feature-fusion framework integrated within a (Vision Recognition)VR assisted oral cancer recognition system. Our method combines extensive data centric augmentation with fused clinical and image-based representations to enhance model robustness and reduce diagnostic ambiguity. Using a stratified training pipeline and an EfficientNetV2 B1 backbone, the system improves feature diversity, mitigates imbalance, and strengthens the learned multimodal embeddings. Experimental evaluation demonstrates that the proposed framework achieves an overall accuracy of 82.57 percent on 2 classes, 65.13 percent on 3 classes, and 54.97 percent on 4 classes, outperforming traditional single stream CNN models. These results highlight the effectiveness of multimodal feature fusion combined with strategic augmentation for reliable early oral cancer lesion recognition and serve as a foundation for immersive VR based clinical decision support tools.
>
---
#### [replaced 012] InfMasking: Unleashing Synergistic Information by Contrastive Multimodal Interactions
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.25270v4](https://arxiv.org/pdf/2509.25270v4)**

> **作者:** Liangjian Wen; Qun Dai; Jianzhuang Liu; Jiangtao Zheng; Yong Dai; Dongkai Wang; Zhao Kang; Jun Wang; Zenglin Xu; Jiang Duan
>
> **备注:** Conference on Neural Information Processing Systems (NeurIPS) 2025 (Spotlight)
>
> **摘要:** In multimodal representation learning, synergistic interactions between modalities not only provide complementary information but also create unique outcomes through specific interaction patterns that no single modality could achieve alone. Existing methods may struggle to effectively capture the full spectrum of synergistic information, leading to suboptimal performance in tasks where such interactions are critical. This is particularly problematic because synergistic information constitutes the fundamental value proposition of multimodal representation. To address this challenge, we introduce InfMasking, a contrastive synergistic information extraction method designed to enhance synergistic information through an Infinite Masking strategy. InfMasking stochastically occludes most features from each modality during fusion, preserving only partial information to create representations with varied synergistic patterns. Unmasked fused representations are then aligned with masked ones through mutual information maximization to encode comprehensive synergistic information. This infinite masking strategy enables capturing richer interactions by exposing the model to diverse partial modality combinations during training. As computing mutual information estimates with infinite masking is computationally prohibitive, we derive an InfMasking loss to approximate this calculation. Through controlled experiments, we demonstrate that InfMasking effectively enhances synergistic information between modalities. In evaluations on large-scale real-world datasets, InfMasking achieves state-of-the-art performance across seven benchmarks. Code is released at https://github.com/brightest66/InfMasking.
>
---
#### [replaced 013] AH-GS: Augmented 3D Gaussian Splatting for High-Frequency Detail Representation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2503.22324v2](https://arxiv.org/pdf/2503.22324v2)**

> **作者:** Chenyang Xu; XingGuo Deng; Rui Zhong
>
> **备注:** need to revsie
>
> **摘要:** The 3D Gaussian Splatting (3D-GS) is a novel method for scene representation and view synthesis. Although Scaffold-GS achieves higher quality real-time rendering compared to the original 3D-GS, its fine-grained rendering of the scene is extremely dependent on adequate viewing angles. The spectral bias of neural network learning results in Scaffold-GS's poor ability to perceive and learn high-frequency information in the scene. In this work, we propose enhancing the manifold complexity of input features and using network-based feature map loss to improve the image reconstruction quality of 3D-GS models. We introduce AH-GS, which enables 3D Gaussians in structurally complex regions to obtain higher-frequency encodings, allowing the model to more effectively learn the high-frequency information of the scene. Additionally, we incorporate high-frequency reinforce loss to further enhance the model's ability to capture detailed frequency information. Our result demonstrates that our model significantly improves rendering fidelity, and in specific scenarios (e.g., MipNeRf360-garden), our method exceeds the rendering quality of Scaffold-GS in just 15K iterations.
>
---
#### [replaced 014] COLT: Enhancing Video Large Language Models with Continual Tool Usage
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.18754v3](https://arxiv.org/pdf/2509.18754v3)**

> **作者:** Yuyang Liu; Meng Cao; Xinyuan Shi; Xiaondan Liang
>
> **备注:** 16 pages
>
> **摘要:** The success of Large Language Models (LLMs) has significantly propelled the research of video understanding. To harvest the benefits of well-trained expert models (i.e., tools), video LLMs prioritize the exploration of tool usage capabilities. Existing methods either prompt closed-source LLMs or employ the instruction tuning paradigm for tool-use fine-tuning. These methods, however, assume an established repository of fixed tools and struggle to generalize to real-world environments where tool data is perpetually evolving and streaming in. To this end, we propose to enhance open-source video LLMs with COntinuaL Tool usage (termed COLT), which automatically acquires tool-use ability in a successive tool stream without suffering 'catastrophic forgetting' of the past learned tools. Specifically, our COLT incorporates a learnable tool codebook as a tool-specific memory system. Then relevant tools are dynamically selected based on the similarity between user instruction and tool features within the codebook. To unleash the tool usage potential of video LLMs, we collect a video-centric tool-use instruction tuning dataset VideoToolBench. Extensive experiments on both previous video LLM benchmarks and the tool-use-specific VideoToolBench dataset demonstrate the state-of-the-art performance of our proposed COLT.
>
---
#### [replaced 015] DGE-YOLO: Dual-Branch Gathering and Attention for Accurate UAV Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.23252v2](https://arxiv.org/pdf/2506.23252v2)**

> **作者:** Kunwei Lv; Zhiren Xiao; Hang Ren; Ping Lan
>
> **备注:** 5 pages, 5 figures
>
> **摘要:** The rapid proliferation of unmanned aerial vehicles (UAVs) has highlighted the importance of robust and efficient object detection in diverse aerial scenarios. Detecting small objects under complex conditions, however, remains a significant challenge.To address this, we present DGE-YOLO, an enhanced YOLO-based detection framework designed to effectively fuse multi-modal information. We introduce a dual-branch architecture for modality-specific feature extraction, enabling the model to process both infrared and visible images. To further enrich semantic representation, we propose an Efficient Multi-scale Attention (EMA) mechanism that enhances feature learning across spatial scales. Additionally, we replace the conventional neck with a Gather-and-Distribute(GD) module to mitigate information loss during feature aggregation. Extensive experiments on the Drone Vehicle dataset demonstrate that DGE-YOLO achieves superior performance over state-of-the-art methods, validating its effectiveness in multi-modal UAV object detection tasks.
>
---
#### [replaced 016] Quantifying task-relevant representational similarity using decision variable correlation
- **分类: cs.CV; cs.LG; q-bio.NC; q-bio.QM**

- **链接: [https://arxiv.org/pdf/2506.02164v2](https://arxiv.org/pdf/2506.02164v2)**

> **作者:** Yu; Qian; Wilson S. Geisler; Xue-Xin Wei
>
> **备注:** Camera-ready version; accepted at NeurIPS 2025
>
> **摘要:** Previous studies have compared neural activities in the visual cortex to representations in deep neural networks trained on image classification. Interestingly, while some suggest that their representations are highly similar, others argued the opposite. Here, we propose a new approach to characterize the similarity of the decision strategies of two observers (models or brains) using decision variable correlation (DVC). DVC quantifies the image-by-image correlation between the decoded decisions based on the internal neural representations in a classification task. Thus, it can capture task-relevant information rather than general representational alignment. We evaluate DVC using monkey V4/IT recordings and network models trained on image classification tasks. We find that model-model similarity is comparable to monkey-monkey similarity, whereas model-monkey similarity is consistently lower. Strikingly, DVC decreases with increasing network performance on ImageNet-1k. Adversarial training does not improve model-monkey similarity in task-relevant dimensions assessed using DVC, although it markedly increases the model-model similarity. Similarly, pre-training on larger datasets does not improve model-monkey similarity. These results suggest a divergence between the task-relevant representations in monkey V4/IT and those learned by models trained on image classification tasks.
>
---
#### [replaced 017] OVSeg3R: Learn Open-vocabulary Instance Segmentation from 2D via 3D Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.23541v2](https://arxiv.org/pdf/2509.23541v2)**

> **作者:** Hongyang Li; Jinyuan Qu; Lei Zhang
>
> **摘要:** In this paper, we propose a training scheme called OVSeg3R to learn open-vocabulary 3D instance segmentation from well-studied 2D perception models with the aid of 3D reconstruction. OVSeg3R directly adopts reconstructed scenes from 2D videos as input, avoiding costly manual adjustment while aligning input with real-world applications. By exploiting the 2D to 3D correspondences provided by 3D reconstruction models, OVSeg3R projects each view's 2D instance mask predictions, obtained from an open-vocabulary 2D model, onto 3D to generate annotations for the view's corresponding sub-scene. To avoid incorrectly introduced false positives as supervision due to partial annotations from 2D to 3D, we propose a View-wise Instance Partition algorithm, which partitions predictions to their respective views for supervision, stabilizing the training process. Furthermore, since 3D reconstruction models tend to over-smooth geometric details, clustering reconstructed points into representative super-points based solely on geometry, as commonly done in mainstream 3D segmentation methods, may overlook geometrically non-salient objects. We therefore introduce 2D Instance Boundary-aware Superpoint, which leverages 2D masks to constrain the superpoint clustering, preventing superpoints from violating instance boundaries. With these designs, OVSeg3R not only extends a state-of-the-art closed-vocabulary 3D instance segmentation model to open-vocabulary, but also substantially narrows the performance gap between tail and head classes, ultimately leading to an overall improvement of +2.3 mAP on the ScanNet200 benchmark. Furthermore, under the standard open-vocabulary setting, OVSeg3R surpasses previous methods by about +7.1 mAP on the novel classes, further validating its effectiveness.
>
---
#### [replaced 018] Unsupervised Stereo via Multi-Baseline Geometry-Consistent Self-Training
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.10838v2](https://arxiv.org/pdf/2508.10838v2)**

> **作者:** Peng Xu; Zhiyu Xiang; Tingming Bai; Tianyu Pu; Kai Wang; Chaojie Ji; Zhihao Yang; Eryun Liu
>
> **摘要:** Photometric loss and pseudo-label-based self-training are two widely used methods for training stereo networks on unlabeled data. However, they both struggle to provide accurate supervision in occluded regions. The former lacks valid correspondences, while the latter's pseudo labels are often unreliable. To overcome these limitations, we present S$^3$, a simple yet effective framework based on multi-baseline geometry consistency. Unlike conventional self-training where teacher and student share identical stereo pairs, S$^3$ assigns them different target images, introducing natural visibility asymmetry. Regions occluded in the student's view often remain visible and matchable to the teacher, enabling reliable pseudo labels even in regions where photometric supervision fails. The teacher's disparities are rescaled to align with the student's baseline and used to guide student learning. An occlusion-aware weighting strategy is further proposed to mitigate unreliable supervision in teacher-occluded regions and to encourage the student to learn robust occlusion completion. To support training, we construct MBS20K, a multi-baseline stereo dataset synthesized using the CARLA simulator. Extensive experiments demonstrate that S$^3$ provides effective supervision in both occluded and non-occluded regions, achieves strong generalization performance, and surpasses previous state-of-the-art methods on the KITTI 2015 and 2012 benchmarks.
>
---
#### [replaced 019] PICABench: How Far Are We from Physically Realistic Image Editing?
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.17681v3](https://arxiv.org/pdf/2510.17681v3)**

> **作者:** Yuandong Pu; Le Zhuo; Songhao Han; Jinbo Xing; Kaiwen Zhu; Shuo Cao; Bin Fu; Si Liu; Hongsheng Li; Yu Qiao; Wenlong Zhang; Xi Chen; Yihao Liu
>
> **摘要:** Image editing has achieved remarkable progress recently. Modern editing models could already follow complex instructions to manipulate the original content. However, beyond completing the editing instructions, the accompanying physical effects are the key to the generation realism. For example, removing an object should also remove its shadow, reflections, and interactions with nearby objects. Unfortunately, existing models and benchmarks mainly focus on instruction completion but overlook these physical effects. So, at this moment, how far are we from physically realistic image editing? To answer this, we introduce PICABench, which systematically evaluates physical realism across eight sub-dimension (spanning optics, mechanics, and state transitions) for most of the common editing operations (add, remove, attribute change, etc.). We further propose the PICAEval, a reliable evaluation protocol that uses VLM-as-a-judge with per-case, region-level human annotations and questions. Beyond benchmarking, we also explore effective solutions by learning physics from videos and construct a training dataset PICA-100K. After evaluating most of the mainstream models, we observe that physical realism remains a challenging problem with large rooms to explore. We hope that our benchmark and proposed solutions can serve as a foundation for future work moving from naive content editing toward physically consistent realism.
>
---
#### [replaced 020] Energy Propagation in Scattering Convolution Networks Can Be Arbitrarily Slow
- **分类: math.FA; cs.CV**

- **链接: [https://arxiv.org/pdf/2406.05121v2](https://arxiv.org/pdf/2406.05121v2)**

> **作者:** Hartmut Führ; Max Getter
>
> **备注:** 44 pages; updated to match published OA version (ACHA); added a brief reference to related SampTA 2025 paper
>
> **摘要:** We analyze energy decay for deep convolutional neural networks employed as feature extractors, including Mallat's wavelet scattering transform. For time-frequency scattering transforms based on Gabor filters, previous work has established that energy decay is exponential for arbitrary square-integrable input signals. In contrast, our main results allow proving that this is false for wavelet scattering in arbitrary dimensions. Specifically, we show that the energy decay of wavelet and wavelet-like scattering transforms acting on generic square-integrable signals can be arbitrarily slow. Importantly, this slow decay behavior holds for dense subsets of $L^2(\mathbb{R}^d)$, indicating that rapid energy decay is generally an unstable property of signals. We complement these findings with positive results that allow us to infer fast (up to exponential) energy decay for generalized Sobolev spaces tailored to the frequency localization of the underlying filter bank. Both negative and positive results highlight that energy decay in scattering networks critically depends on the interplay between the respective frequency localizations of both the signal and the filters used.
>
---
#### [replaced 021] LRANet++: Low-Rank Approximation Network for Accurate and Efficient Text Spotting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.05818v2](https://arxiv.org/pdf/2511.05818v2)**

> **作者:** Yuchen Su; Zhineng Chen; Yongkun Du; Zuxuan Wu; Hongtao Xie; Yu-Gang Jiang
>
> **备注:** Accepted by IEEE TPAMI
>
> **摘要:** End-to-end text spotting aims to jointly optimize text detection and recognition within a unified framework. Despite significant progress, designing an accurate and efficient end-to-end text spotter for arbitrary-shaped text remains challenging. We identify the primary bottleneck as the lack of a reliable and efficient text detection method. To address this, we propose a novel parameterized text shape representation based on low-rank approximation for precise detection and a triple assignment detection head for fast inference. Specifically, unlike current data-irrelevant shape representation methods, we exploit shape correlations among labeled text boundaries to construct a robust low-rank subspace. By minimizing an $\ell_1$-norm objective, we extract orthogonal vectors that capture the intrinsic text shape from noisy annotations, enabling precise reconstruction via the linear combination of only a few basis vectors. Next, the triple assignment scheme decouples training complexity from inference speed. It utilizes a deep sparse branch to guide an ultra-lightweight inference branch, while a dense branch provides rich parallel supervision. Building upon these advancements, we integrate the enhanced detection module with a lightweight recognition branch to form an end-to-end text spotting framework, termed LRANet++, capable of accurately and efficiently spotting arbitrary-shaped text. Extensive experiments on challenging benchmarks demonstrate the superiority of LRANet++ compared to state-of-the-art methods. Code is available at: https://github.com/ychensu/LRANet-PP.
>
---
#### [replaced 022] Diminishing Returns in Self-Supervised Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.03862v2](https://arxiv.org/pdf/2512.03862v2)**

> **作者:** Oli Bridge; Huey Sun; Botond Branyicskai-Nagy; Charles D'Ornano; Shomit Basu
>
> **摘要:** Transformer-based architectures have become a dominant paradigm in vision and language, but their success is often attributed to large model capacity and massive training data. In this work, we examine how self-supervised pre-training, intermediate fine-tuning, and downstream fine-tuning interact in a low-capacity regime, using a 5M-parameter Vision Transformer for semantic segmentation. Across multiple data scales, we find that masked image modeling pre-training and downstream fine-tuning reliably improve performance, but with clear diminishing returns as supervision increases. In contrast, inserting an intermediate classification fine-tuning stage consistently degrades downstream performance, with the largest drops occurring precisely where pre-training is most effective. Through an analysis of patch-level representation geometry, we show that classification-based intermediate supervision actively interferes with representations learned during pre-training by collapsing spatial structure critical for dense prediction. These results indicate that, in small models, the geometry of supervision matters more than the number of training stages: misaligned intermediate objectives can negate the benefits of pre-training rather than amplify them.
>
---
#### [replaced 023] CADMorph: Geometry-Driven Parametric CAD Editing via a Plan-Generate-Verify Loop
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.11480v2](https://arxiv.org/pdf/2512.11480v2)**

> **作者:** Weijian Ma; Shizhao Sun; Ruiyu Wang; Jiang Bian
>
> **备注:** NeurIPS 2025
>
> **摘要:** A Computer-Aided Design (CAD) model encodes an object in two coupled forms: a parametric construction sequence and its resulting visible geometric shape. During iterative design, adjustments to the geometric shape inevitably require synchronized edits to the underlying parametric sequence, called geometry-driven parametric CAD editing. The task calls for 1) preserving the original sequence's structure, 2) ensuring each edit's semantic validity, and 3) maintaining high shape fidelity to the target shape, all under scarce editing data triplets. We present CADMorph, an iterative plan-generate-verify framework that orchestrates pretrained domain-specific foundation models during inference: a parameter-to-shape (P2S) latent diffusion model and a masked-parameter-prediction (MPP) model. In the planning stage, cross-attention maps from the P2S model pinpoint the segments that need modification and offer editing masks. The MPP model then infills these masks with semantically valid edits in the generation stage. During verification, the P2S model embeds each candidate sequence in shape-latent space, measures its distance to the target shape, and selects the closest one. The three stages leverage the inherent geometric consciousness and design knowledge in pretrained priors, and thus tackle structure preservation, semantic validity, and shape fidelity respectively. Besides, both P2S and MPP models are trained without triplet data, bypassing the data-scarcity bottleneck. CADMorph surpasses GPT-4o and specialized CAD baselines, and supports downstream applications such as iterative editing and reverse-engineering enhancement.
>
---
#### [replaced 024] Point Cloud to Mesh Reconstruction: Methods, Trade-offs, and Implementation Guide
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2412.10977v2](https://arxiv.org/pdf/2412.10977v2)**

> **作者:** Fatima Zahra Iguenfer; Achraf Hsain; Hiba Amissa; Yousra Chtouki
>
> **摘要:** Reconstructing meshes from point clouds is a fundamental task in computer vision with applications spanning robotics, autonomous systems, and medical imaging. Selecting an appropriate learning-based method requires understanding trade-offs between computational efficiency, geometric accuracy, and output constraints. This paper categorizes over fifteen methods into five paradigms -- PointNet family, autoencoder architectures, deformation-based methods, point-move techniques, and primitive-based approaches -- and provides practical guidance for method selection. We contribute: (1) a decision framework mapping input/output requirements to suitable paradigms, (2) a failure mode analysis to assist practitioners in debugging implementations, (3) standardized comparisons on ShapeNet benchmarks, and (4) a curated list of maintained codebases with implementation resources. By synthesizing both theoretical foundations and practical considerations, this work serves as an entry point for practitioners and researchers new to learning-based 3D mesh reconstruction.
>
---
#### [replaced 025] AHA! Animating Human Avatars in Diverse Scenes with Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.09827v2](https://arxiv.org/pdf/2511.09827v2)**

> **作者:** Aymen Mir; Jian Wang; Riza Alp Guler; Chuan Guo; Gerard Pons-Moll; Bing Zhou
>
> **备注:** Project page available at: https://miraymen.github.io/aha/
>
> **摘要:** We present a novel framework for animating humans in 3D scenes using 3D Gaussian Splatting (3DGS), a neural scene representation that has recently achieved state-of-the-art photorealistic results for novel-view synthesis but remains under-explored for human-scene animation and interaction. Unlike existing animation pipelines that use meshes or point clouds as the underlying 3D representation, our approach introduces the use of 3DGS as the 3D representation for animating humans in scenes. By representing humans and scenes as Gaussians, our approach allows geometry-consistent free-viewpoint rendering of humans interacting with 3D scenes. Our key insight is that rendering can be decoupled from motion synthesis, and each sub-problem can be addressed independently without the need for paired human-scene data. Central to our method is a Gaussian-aligned motion module that synthesizes motion without explicit scene geometry, using opacity-based cues and projected Gaussian structures to guide human placement and pose alignment. To ensure natural interactions, we further propose a human-scene Gaussian refinement optimization that enforces realistic contact and navigation. We evaluate our approach on scenes from Scannet++ and the SuperSplat library, and on avatars reconstructed from sparse and dense multi-view human capture. Finally, we demonstrate that our framework enables novel applications such as geometry-consistent free-viewpoint rendering of edited monocular RGB videos with newly animated humans, showcasing the unique advantages of 3DGS for monocular video-based human animation. To assess the full quality of our results, we encourage readers to view the supplementary material available at https://miraymen.github.io/aha/ .
>
---
#### [replaced 026] RoboMirror: Understand Before You Imitate for Video to Humanoid Locomotion
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出RoboMirror，解决视频到人形机器人运动的控制问题，通过视觉理解生成合理动作，无需姿态重建。**

- **链接: [https://arxiv.org/pdf/2512.23649v3](https://arxiv.org/pdf/2512.23649v3)**

> **作者:** Zhe Li; Cheng Chi; Boan Zhu; Yangyang Wei; Shuanghao Bai; Yuheng Ji; Yibo Peng; Tao Huang; Pengwei Wang; Zhongyuan Wang; S. -H. Gary Chan; Chang Xu; Shanghang Zhang
>
> **摘要:** Humans learn locomotion through visual observation, interpreting visual content first before imitating actions. However, state-of-the-art humanoid locomotion systems rely on either curated motion capture trajectories or sparse text commands, leaving a critical gap between visual understanding and control. Text-to-motion methods suffer from semantic sparsity and staged pipeline errors, while video-based approaches only perform mechanical pose mimicry without genuine visual understanding. We propose RoboMirror, the first retargeting-free video-to-locomotion framework embodying "understand before you imitate". Leveraging VLMs, it distills raw egocentric/third-person videos into visual motion intents, which directly condition a diffusion-based policy to generate physically plausible, semantically aligned locomotion without explicit pose reconstruction or retargeting. Extensive experiments validate the effectiveness of RoboMirror, it enables telepresence via egocentric videos, drastically reduces third-person control latency by 80%, and achieves a 3.7% higher task success rate than baselines. By reframing humanoid control around video understanding, we bridge the visual understanding and action gap.
>
---
#### [replaced 027] Ideal Observer for Segmentation of Dead Leaves Images
- **分类: cs.CV; math.ST; stat.ME**

- **链接: [https://arxiv.org/pdf/2512.05539v2](https://arxiv.org/pdf/2512.05539v2)**

> **作者:** Swantje Mahncke; Malte Ott
>
> **备注:** 41 pages, 16 figures
>
> **摘要:** The human visual environment is comprised of different surfaces that are distributed in space. The parts of a scene that are visible at any one time are governed by the occlusion of overlapping objects. In this work we consider "dead leaves" models, which replicate these occlusions when generating images by layering objects on top of each other. A dead leaves model is a generative model comprised of distributions for object position, shape, color and texture. An image is generated from a dead leaves model by sampling objects ("leaves") from these distributions until a stopping criterion is reached, usually when the image is fully covered or until a given number of leaves was sampled. Here, we describe a theoretical approach, based on previous work, to derive a Bayesian ideal observer for the partition of a given set of pixels based on independent dead leaves model distributions. Extending previous work, we provide step-by-step explanations for the computation of the posterior probability as well as describe factors that determine the feasibility of practically applying this computation. The dead leaves image model and the associated ideal observer can be applied to study segmentation decisions in a limited number of pixels, providing a principled upper-bound on performance, to which humans and vision algorithms could be compared.
>
---
#### [replaced 028] On Exact Editing of Flow-Based Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.24015v2](https://arxiv.org/pdf/2512.24015v2)**

> **作者:** Zixiang Li; Yue Song; Jianing Peng; Ting Liu; Jun Huang; Xiaochao Qu; Luoqi Liu; Wei Wang; Yao Zhao; Yunchao Wei
>
> **摘要:** Recent methods in flow-based diffusion editing have enabled direct transformations between source and target image distribution without explicit inversion. However, the latent trajectories in these methods often exhibit accumulated velocity errors, leading to semantic inconsistency and loss of structural fidelity. We propose Conditioned Velocity Correction (CVC), a principled framework that reformulates flow-based editing as a distribution transformation problem driven by a known source prior. CVC rethinks the role of velocity in inter-distribution transformation by introducing a dual-perspective velocity conversion mechanism. This mechanism explicitly decomposes the latent evolution into two components: a structure-preserving branch that remains consistent with the source trajectory, and a semantically-guided branch that drives a controlled deviation toward the target distribution. The conditional velocity field exhibits an absolute velocity error relative to the true underlying distribution trajectory, which inherently introduces potential instability and trajectory drift in the latent space. To address this quantifiable deviation and maintain fidelity to the true flow, we apply a posterior-consistent update to the resulting conditional velocity field. This update is derived from Empirical Bayes Inference and Tweedie correction, which ensures a mathematically grounded error compensation over time. Our method yields stable and interpretable latent dynamics, achieving faithful reconstruction alongside smooth local semantic conversion. Comprehensive experiments demonstrate that CVC consistently achieves superior fidelity, better semantic alignment, and more reliable editing behavior across diverse tasks.
>
---
#### [replaced 029] TI-PREGO: Chain of Thought and In-Context Learning for Online Mistake Detection in PRocedural EGOcentric Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.02570v3](https://arxiv.org/pdf/2411.02570v3)**

> **作者:** Leonardo Plini; Luca Scofano; Edoardo De Matteis; Guido Maria D'Amely di Melendugno; Alessandro Flaborea; Andrea Sanchietti; Giovanni Maria Farinella; Fabio Galasso; Antonino Furnari
>
> **摘要:** Identifying procedural errors online from egocentric videos is a critical yet challenging task across various domains, including manufacturing, healthcare, and skill-based training. The nature of such mistakes is inherently open-set, as unforeseen or novel errors may occur, necessitating robust detection systems that do not rely on prior examples of failure. Currently, however, no technique effectively detects open-set procedural mistakes online. We propose a dual branch architecture to address this problem in an online fashion: one branch continuously performs step recognition from the input egocentric video, while the other anticipates future steps based on the recognition module's output. Mistakes are detected as mismatches between the currently recognized action and the action predicted by the anticipation module. The recognition branch takes input frames, predicts the current action, and aggregates frame-level results into action tokens. The anticipation branch, specifically, leverages the solid pattern-matching capabilities of Large Language Models (LLMs) to predict action tokens based on previously predicted ones. Given the online nature of the task, we also thoroughly benchmark the difficulties associated with per-frame evaluations, particularly the need for accurate and timely predictions in dynamic online scenarios. Extensive experiments on two procedural datasets demonstrate the challenges and opportunities of leveraging a dual-branch architecture for mistake detection, showcasing the effectiveness of our proposed approach. In a thorough evaluation including recognition and anticipation variants and state-of-the-art models, our method reveals its robustness and effectiveness in online applications.
>
---
#### [replaced 030] VALLR: Visual ASR Language Model for Lip Reading
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.21408v2](https://arxiv.org/pdf/2503.21408v2)**

> **作者:** Marshall Thomas; Edward Fish; Richard Bowden
>
> **摘要:** Lip Reading, or Visual Automatic Speech Recognition (V-ASR), is a complex task requiring the interpretation of spoken language exclusively from visual cues, primarily lip movements and facial expressions. This task is especially challenging due to the absence of auditory information and the inherent ambiguity when visually distinguishing phonemes that have overlapping visemes where different phonemes appear identical on the lips. Current methods typically attempt to predict words or characters directly from these visual cues, but this approach frequently encounters high error rates due to coarticulation effects and viseme ambiguity. We propose a novel two-stage, phoneme-centric framework for Visual Automatic Speech Recognition (V-ASR) that addresses these longstanding challenges. First, our model predicts a compact sequence of phonemes from visual inputs using a Video Transformer with a CTC head, thereby reducing the task complexity and achieving robust speaker invariance. This phoneme output then serves as the input to a fine-tuned Large Language Model (LLM), which reconstructs coherent words and sentences by leveraging broader linguistic context. Unlike existing methods that either predict words directly-often faltering on visually similar phonemes-or rely on large-scale multimodal pre-training, our approach explicitly encodes intermediate linguistic structure while remaining highly data efficient. We demonstrate state-of-the-art performance on two challenging datasets, LRS2 and LRS3, where our method achieves significant reductions in Word Error Rate (WER) achieving a SOTA WER of 18.7 on LRS3 despite using 99.4% less labelled data than the next best approach.
>
---
#### [replaced 031] Answering from Sure to Uncertain: Uncertainty-Aware Curriculum Learning for Video Question Answering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2401.01510v2](https://arxiv.org/pdf/2401.01510v2)**

> **作者:** Haopeng Li; Mohammed Bennamoun; Jun Liu; Hossein Rahmani; Qiuhong Ke
>
> **备注:** Accepted by BMVC 2025
>
> **摘要:** While significant advancements have been made in video question answering (VideoQA), the potential benefits of enhancing model generalization through tailored difficulty scheduling have been largely overlooked in existing research. This paper seeks to bridge that gap by incorporating VideoQA into a curriculum learning (CL) framework that progressively trains models from simpler to more complex data. Recognizing that conventional self-paced CL methods rely on training loss for difficulty measurement, which might not accurately reflect the intricacies of video-question pairs, we introduce the concept of uncertainty-aware CL. Here, uncertainty serves as the guiding principle for dynamically adjusting the difficulty. Furthermore, we address the challenge posed by uncertainty by presenting a probabilistic modeling approach for VideoQA. Specifically, we conceptualize VideoQA as a stochastic computation graph, where the hidden representations are treated as stochastic variables. This yields two distinct types of uncertainty: one related to the inherent uncertainty in the data and another pertaining to the model's confidence. In practice, we seamlessly integrate the VideoQA model into our framework and conduct comprehensive experiments. The findings affirm that our approach not only achieves enhanced performance but also effectively quantifies uncertainty in the context of VideoQA.
>
---
#### [replaced 032] Spinal Line Detection for Posture Evaluation through Train-ing-free 3D Human Body Reconstruction with 2D Depth Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.12718v2](https://arxiv.org/pdf/2512.12718v2)**

> **作者:** Sehyun Kim; Hye Jun Lee; Jiwoo Lee; Changgyun Kim; Taemin Lee
>
> **备注:** GitHub, see https://github.com/DevChoco/TF3D_SpineDetect
>
> **摘要:** The spinal angle is an important indicator of body balance. It is important to restore the 3D shape of the human body and estimate the spine center line. Existing mul-ti-image-based body restoration methods require expensive equipment and complex pro-cedures, and single image-based body restoration methods have limitations in that it is difficult to accurately estimate the internal structure such as the spine center line due to occlusion and viewpoint limitation. This study proposes a method to compensate for the shortcomings of the multi-image-based method and to solve the limitations of the sin-gle-image method. We propose a 3D body posture analysis system that integrates depth images from four directions to restore a 3D human model and automatically estimate the spine center line. Through hierarchical matching of global and fine registration, restora-tion to noise and occlusion is performed. Also, the Adaptive Vertex Reduction is applied to maintain the resolution and shape reliability of the mesh, and the accuracy and stabil-ity of spinal angle estimation are simultaneously secured by using the Level of Detail en-semble. The proposed method achieves high-precision 3D spine registration estimation without relying on training data or complex neural network models, and the verification confirms the improvement of matching quality.
>
---
#### [replaced 033] SAM-aware Test-time Adaptation for Universal Medical Image Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.05221v2](https://arxiv.org/pdf/2506.05221v2)**

> **作者:** Jianghao Wu; Yicheng Wu; Yutong Xie; Wenjia Bai; You Zhang; Feilong Tang; Yulong Li; Imran Razzak; Daniel F Schmidt; Yasmeen George
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Leveraging the Segment Anything Model (SAM) for medical image segmentation remains challenging due to its limited adaptability across diverse medical domains. Although fine-tuned variants, such as MedSAM, improve performance in scenarios similar to the training modalities or organs, they may lack generalizability to unseen data. To overcome this limitation, we propose SAM-aware Test-time Adaptation (SAM-TTA), a lightweight and flexible framework that preserves SAM's inherent generalization ability while enhancing segmentation accuracy for medical images. SAM-TTA tackles two major challenges: (1) input-level discrepancy caused by channel mismatches between natural and medical images, and (2) semantic-level discrepancy due to different object characteristics in natural versus medical images (e.g., with clear boundaries vs. ambiguous structures). To this end, we introduce two complementary components: a self-adaptive Bezier Curve-based Transformation (SBCT), which maps single-channel medical images into SAM-compatible three-channel images via a few learnable parameters to be optimized at test time; and IoU-guided Multi-scale Adaptation (IMA), which leverages SAM's intrinsic IoU scores to enforce high output confidence, dual-scale prediction consistency, and intermediate feature consistency, to improve semantic-level alignments. Extensive experiments on eight public medical image segmentation tasks, covering six grayscale and two color (endoscopic) tasks, demonstrate that SAM-TTA consistently outperforms state-of-the-art test-time adaptation methods. Notably, on six grayscale datasets, SAM-TTA even surpasses fully fine-tuned models, achieving significant Dice improvements (i.e., average 4.8% and 7.4% gains over MedSAM and SAM-Med2D) and establishing a new paradigm for universal medical image segmentation. Code is available at https://github.com/JianghaoWu/SAM-TTA.
>
---
#### [replaced 034] EgoReAct: Egocentric Video-Driven 3D Human Reaction Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.22808v2](https://arxiv.org/pdf/2512.22808v2)**

> **作者:** Libo Zhang; Zekun Li; Tianyu Li; Zeyu Cao; Rui Xu; Xiaoxiao Long; Wenjia Wang; Jingbo Wang; Yuan Liu; Wenping Wang; Daquan Zhou; Taku Komura; Zhiyang Dou
>
> **备注:** 12 pages, 9 figures
>
> **摘要:** Humans exhibit adaptive, context-sensitive responses to egocentric visual input. However, faithfully modeling such reactions from egocentric video remains challenging due to the dual requirements of strictly causal generation and precise 3D spatial alignment. To tackle this problem, we first construct the Human Reaction Dataset (HRD) to address data scarcity and misalignment by building a spatially aligned egocentric video-reaction dataset, as existing datasets (e.g., ViMo) suffer from significant spatial inconsistency between the egocentric video and reaction motion, e.g., dynamically moving motions are always paired with fixed-camera videos. Leveraging HRD, we present EgoReAct, the first autoregressive framework that generates 3D-aligned human reaction motions from egocentric video streams in real-time. We first compress the reaction motion into a compact yet expressive latent space via a Vector Quantised-Variational AutoEncoder and then train a Generative Pre-trained Transformer for reaction generation from the visual input. EgoReAct incorporates 3D dynamic features, i.e., metric depth, and head dynamics during the generation, which effectively enhance spatial grounding. Extensive experiments demonstrate that EgoReAct achieves remarkably higher realism, spatial consistency, and generation efficiency compared with prior methods, while maintaining strict causality during generation. We will release code, models, and data upon acceptance.
>
---
#### [replaced 035] Dream-VL & Dream-VLA: Open Vision-Language and Vision-Language-Action Models with Diffusion Language Model Backbone
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出基于扩散语言模型的视觉-语言（Dream-VL）和视觉-语言-动作（Dream-VLA）模型，解决复杂视觉规划与动态控制问题。通过连续预训练提升任务表现。**

- **链接: [https://arxiv.org/pdf/2512.22615v2](https://arxiv.org/pdf/2512.22615v2)**

> **作者:** Jiacheng Ye; Shansan Gong; Jiahui Gao; Junming Fan; Shuang Wu; Wei Bi; Haoli Bai; Lifeng Shang; Lingpeng Kong
>
> **备注:** Add real-world experiments
>
> **摘要:** While autoregressive Large Vision-Language Models (VLMs) have achieved remarkable success, their sequential generation often limits their efficacy in complex visual planning and dynamic robotic control. In this work, we investigate the potential of constructing Vision-Language Models upon diffusion-based large language models (dLLMs) to overcome these limitations. We introduce Dream-VL, an open diffusion-based VLM (dVLM) that achieves state-of-the-art performance among previous dVLMs. Dream-VL is comparable to top-tier AR-based VLMs trained on open data on various benchmarks but exhibits superior potential when applied to visual planning tasks. Building upon Dream-VL, we introduce Dream-VLA, a dLLM-based Vision-Language-Action model (dVLA) developed through continuous pre-training on open robotic datasets. We demonstrate that the natively bidirectional nature of this diffusion backbone serves as a superior foundation for VLA tasks, inherently suited for action chunking and parallel generation, leading to significantly faster convergence in downstream fine-tuning. Dream-VLA achieves top-tier performance of 97.2% average success rate on LIBERO, 71.4% overall average on SimplerEnv-Bridge, and 60.5% overall average on SimplerEnv-Fractal, surpassing leading models such as $π_0$ and GR00T-N1. We also validate that dVLMs surpass AR baselines on downstream tasks across different training objectives. We release both Dream-VL and Dream-VLA to facilitate further research in the community.
>
---
#### [replaced 036] DISCODE: Distribution-Aware Score Decoder for Robust Automatic Evaluation of Image Captioning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.14420v2](https://arxiv.org/pdf/2512.14420v2)**

> **作者:** Nakamasa Inoue; Kanoko Goto; Masanari Oi; Martyna Gruszka; Mahiro Ukai; Takumi Hirose; Yusuke Sekikawa
>
> **备注:** Paper accepted to AAAI 2026
>
> **摘要:** Large vision-language models (LVLMs) have shown impressive performance across a broad range of multimodal tasks. However, robust image caption evaluation using LVLMs remains challenging, particularly under domain-shift scenarios. To address this issue, we introduce the Distribution-Aware Score Decoder (DISCODE), a novel finetuning-free method that generates robust evaluation scores better aligned with human judgments across diverse domains. The core idea behind DISCODE lies in its test-time adaptive evaluation approach, which introduces the Adaptive Test-Time (ATT) loss, leveraging a Gaussian prior distribution to improve robustness in evaluation score estimation. This loss is efficiently minimized at test time using an analytical solution that we derive. Furthermore, we introduce the Multi-domain Caption Evaluation (MCEval) benchmark, a new image captioning evaluation benchmark covering six distinct domains, designed to assess the robustness of evaluation metrics. In our experiments, we demonstrate that DISCODE achieves state-of-the-art performance as a reference-free evaluation metric across MCEval and four representative existing benchmarks.
>
---
#### [replaced 037] PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.05353v2](https://arxiv.org/pdf/2508.05353v2)**

> **作者:** Kang Liu; Zhuoqi Ma; Zikang Fang; Yunan Li; Kun Xie; Qiguang Miao
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Chest X-ray report generation aims to reduce radiologists' workload by automatically producing high-quality preliminary reports. A critical yet underexplored aspect of this task is the effective use of patient-specific prior knowledge -- including clinical context (e.g., symptoms, medical history) and the most recent prior image -- which radiologists routinely rely on for diagnostic reasoning. Most existing methods generate reports from single images, neglecting this essential prior information and thus failing to capture diagnostic intent or disease progression. To bridge this gap, we propose PriorRG, a novel chest X-ray report generation framework that emulates real-world clinical workflows via a two-stage training pipeline. In Stage 1, we introduce a prior-guided contrastive pre-training scheme that leverages clinical context to guide spatiotemporal feature extraction, allowing the model to align more closely with the intrinsic spatiotemporal semantics in radiology reports. In Stage 2, we present a prior-aware coarse-to-fine decoding for report generation that progressively integrates patient-specific prior knowledge with the vision encoder's hidden states. This decoding allows the model to align with diagnostic focus and track disease progression, thereby enhancing the clinical accuracy and fluency of the generated reports. Extensive experiments on MIMIC-CXR and MIMIC-ABN datasets demonstrate that PriorRG outperforms state-of-the-art methods, achieving a 3.6% BLEU-4 and 3.8% F1 score improvement on MIMIC-CXR, and a 5.9% BLEU-1 gain on MIMIC-ABN. Code and checkpoints will be released upon acceptance.
>
---
#### [replaced 038] Improving VisNet for Object Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.08897v2](https://arxiv.org/pdf/2511.08897v2)**

> **作者:** Mehdi Fatan Serj; C. Alejandro Parraga; Xavier Otazu
>
> **摘要:** Object recognition plays a fundamental role in how biological organisms perceive and interact with their environment. While the human visual system performs this task with remarkable efficiency, reproducing similar capabilities in artificial systems remains challenging. This study investigates VisNet, a biologically inspired neural network model, and several enhanced variants incorporating radial basis function neurons, Mahalanobis distance based learning, and retinal like preprocessing for both general object recognition and symmetry classification. By leveraging principles of Hebbian learning and temporal continuity associating temporally adjacent views to build invariant representations. VisNet and its extensions capture robust and transformation invariant features. Experimental results across multiple datasets, including MNIST, CIFAR10, and custom symmetric object sets, show that these enhanced VisNet variants substantially improve recognition accuracy compared with the baseline model. These findings underscore the adaptability and biological relevance of VisNet inspired architectures, offering a powerful and interpretable framework for visual recognition in both neuroscience and artificial intelligence. Keywords: VisNet, Object Recognition, Symmetry Detection, Hebbian Learning, RBF Neurons, Mahalanobis Distance, Biologically Inspired Models, Invariant Representations
>
---
#### [replaced 039] RingMo-Agent: A Unified Remote Sensing Foundation Model for Multi-Platform and Multi-Modal Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.20776v2](https://arxiv.org/pdf/2507.20776v2)**

> **作者:** Huiyang Hu; Peijin Wang; Yingchao Feng; Kaiwen Wei; Wenxin Yin; Wenhui Diao; Mengyu Wang; Hanbo Bi; Kaiyue Kang; Tong Ling; Kun Fu; Xian Sun
>
> **备注:** 23 pages, 6 figures, 20 tables
>
> **摘要:** Remote sensing (RS) images from multiple modalities and platforms exhibit diverse details due to differences in sensor characteristics and imaging perspectives. Existing vision-language research in RS largely relies on relatively homogeneous data sources. Moreover, they still remain limited to conventional visual perception tasks such as classification or captioning. As a result, these methods fail to serve as a unified and standalone framework capable of effectively handling RS imagery from diverse sources in real-world applications. To address these issues, we propose RingMo-Agent, a model designed to handle multi-modal and multi-platform data that performs perception and reasoning tasks based on user textual instructions. Compared with existing models, RingMo-Agent 1) is supported by a large-scale vision-language dataset named RS-VL3M, comprising over 3 million image-text pairs, spanning optical, SAR, and infrared (IR) modalities collected from both satellite and UAV platforms, covering perception and challenging reasoning tasks; 2) learns modality adaptive representations by incorporating separated embedding layers to construct isolated features for heterogeneous modalities and reduce cross-modal interference; 3) unifies task modeling by introducing task-specific tokens and employing a token-based high-dimensional hidden state decoding mechanism designed for long-horizon spatial tasks. Extensive experiments on various RS vision-language tasks demonstrate that RingMo-Agent not only proves effective in both visual understanding and sophisticated analytical tasks, but also exhibits strong generalizability across different platforms and sensing modalities.
>
---
#### [replaced 040] Adaptive Dual-Weighted Gravitational Point Cloud Denoising Method
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.10386v2](https://arxiv.org/pdf/2512.10386v2)**

> **作者:** Ge Zhang; Chunyang Wang; Bin Liu; Guan Xi
>
> **摘要:** High-quality point cloud data is a critical foundation for tasks such as autonomous driving and 3D reconstruction. However, LiDAR-based point cloud acquisition is often affected by various disturbances, resulting in a large number of noise points that degrade the accuracy of subsequent point cloud object detection and recognition. Moreover, existing point cloud denoising methods typically sacrifice computational efficiency in pursuit of higher denoising accuracy, or, conversely, improve processing speed at the expense of preserving object boundaries and fine structural details, making it difficult to simultaneously achieve high denoising accuracy, strong edge preservation, and real-time performance. To address these limitations, this paper proposes an adaptive dualweight gravitational-based point cloud denoising method. First, an octree is employed to perform spatial partitioning of the global point cloud, enabling parallel acceleration. Then, within each leaf node, adaptive voxel-based occupancy statistics and k-nearest neighbor (kNN) density estimation are applied to rapidly remove clearly isolated and low-density noise points, thereby reducing the effective candidate set. Finally, a gravitational scoring function that combines density weights with adaptive distance weights is constructed to finely distinguish noise points from object points. Experiments conducted on the Stanford 3D Scanning Repository, the Canadian Adverse Driving Conditions (CADC) dataset, and in-house RUBY PLUS LiDAR point clouds acquired in our laboratory demonstrate that, compared with existing methods, the proposed approach achieves consistent improvements in F1, PSNR, and Chamfer Distance (CD) across various noise conditions while reducing the single-frame processing time, thereby validating its high accuracy, robustness, and real-time performance in multi-noise scenarios.
>
---
#### [replaced 041] Seal2Real: Prompt Prior Learning on Diffusion Model for Unsupervised Document Seal Data Generation and Realisation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2310.00546v3](https://arxiv.org/pdf/2310.00546v3)**

> **作者:** Mingfu Yan; Jiancheng Huang; Shifeng Chen
>
> **摘要:** Seal-related tasks in document processing-such as seal segmentation, authenticity verification, seal removal, and text recognition under seals-hold substantial commercial importance. However, progress in these areas has been hindered by the scarcity of labeled document seal datasets, which are essential for supervised learning. To address this limitation, we propose Seal2Real, a novel generative framework designed to synthesize large-scale labeled document seal data. As part of this work, we also present Seal-DB, a comprehensive dataset containing 20,000 labeled images to support seal-related research. Seal2Real introduces a prompt prior learning architecture built upon a pre-trained Stable Diffusion model, effectively transferring its generative capability to the unsupervised domain of seal image synthesis. By producing highly realistic synthetic seal images, Seal2Real significantly enhances the performance of downstream seal-related tasks on real-world data. Experimental evaluations on the Seal-DB dataset demonstrate the effectiveness and practical value of the proposed framework. The dataset is available at https://github.com/liuyifan6613/DocBank-Document-Enhancement-Dataset.
>
---
#### [replaced 042] TalkingEyes: Pluralistic Speech-Driven 3D Eye Gaze Animation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.09921v2](https://arxiv.org/pdf/2501.09921v2)**

> **作者:** Yixiang Zhuang; Chunshan Ma; Yao Cheng; Xuan Cheng; Jing Liao; Juncong Lin
>
> **摘要:** Although significant progress has been made in the field of speech-driven 3D facial animation recently, the speech-driven animation of an indispensable facial component, eye gaze, has been overlooked by recent research. This is primarily due to the weak correlation between speech and eye gaze, as well as the scarcity of audio-gaze data, making it very challenging to generate 3D eye gaze motion from speech alone. In this paper, we propose a novel data-driven method which can generate diverse 3D eye gaze motions in harmony with the speech. To achieve this, we firstly construct an audio-gaze dataset that contains about 14 hours of audio-mesh sequences featuring high-quality eye gaze motion, head motion and facial motion simultaneously. The motion data is acquired by performing lightweight eye gaze fitting and face reconstruction on videos from existing audio-visual datasets. We then tailor a novel speech-to-motion translation framework in which the head motions and eye gaze motions are jointly generated from speech but are modeled in two separate latent spaces. This design stems from the physiological knowledge that the rotation range of eyeballs is less than that of head. Through mapping the speech embedding into the two latent spaces, the difficulty in modeling the weak correlation between speech and non-verbal motion is thus attenuated. Finally, our TalkingEyes, integrated with a speech-driven 3D facial motion generator, can synthesize eye gaze motion, eye blinks, head motion and facial motion collectively from speech. Extensive quantitative and qualitative evaluations demonstrate the superiority of the proposed method in generating diverse and natural 3D eye gaze motions from speech. The project page of this paper is: https://lkjkjoiuiu.github.io/TalkingEyes_Home/
>
---
#### [replaced 043] On Pitfalls of $\textit{RemOve-And-Retrain}$: Data Processing Inequality Perspective
- **分类: cs.LG; cs.AI; cs.CV; stat.ME**

- **链接: [https://arxiv.org/pdf/2304.13836v4](https://arxiv.org/pdf/2304.13836v4)**

> **作者:** Junhwa Song; Keumgang Cha; Junghoon Seo
>
> **摘要:** Approaches for appraising feature importance approximations, alternatively referred to as attribution methods, have been established across an extensive array of contexts. The development of resilient techniques for performance benchmarking constitutes a critical concern in the sphere of explainable deep learning. This study scrutinizes the dependability of the RemOve-And-Retrain (ROAR) procedure, which is prevalently employed for gauging the performance of feature importance estimates. The insights gleaned from our theoretical foundation and empirical investigations reveal that attributions containing lesser information about the decision function may yield superior results in ROAR benchmarks, contradicting the original intent of ROAR. This occurrence is similarly observed in the recently introduced variant RemOve-And-Debias (ROAD), and we posit a persistent pattern of blurriness bias in ROAR attribution metrics. Our findings serve as a warning against indiscriminate use on ROAR metrics.
>
---
#### [replaced 044] Neural Surface Reconstruction from Sparse Views Using Epipolar Geometry
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2406.04301v2](https://arxiv.org/pdf/2406.04301v2)**

> **作者:** Xinhai Chang; Kaichen Zhou
>
> **摘要:** Reconstructing accurate surfaces from sparse multi-view images remains challenging due to severe geometric ambiguity and occlusions. Existing generalizable neural surface reconstruction methods primarily rely on cost volumes that summarize multi-view features using simple statistics (e.g., mean and variance), which discard critical view-dependent geometric structure and often lead to over-smoothed reconstructions. We propose EpiS, a generalizable neural surface reconstruction framework that explicitly leverages epipolar geometry for sparse-view inputs. Instead of directly regressing geometry from cost-volume statistics, EpiS uses coarse cost-volume features to guide the aggregation of fine-grained epipolar features sampled along corresponding epipolar lines across source views. An epipolar transformer fuses multi-view information, followed by ray-wise aggregation to produce SDF-aware features for surface estimation. To further mitigate information loss under sparse views, we introduce a geometry regularization strategy that leverages a pretrained monocular depth model through scale-invariant global and local constraints. Extensive experiments on DTU and BlendedMVS demonstrate that EpiS significantly outperforms state-of-the-art generalizable surface reconstruction methods under sparse-view settings, while maintaining strong generalization without per-scene optimization.
>
---
#### [replaced 045] Sports-QA: A Large-Scale Video Question Answering Benchmark for Complex and Professional Sports
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2401.01505v5](https://arxiv.org/pdf/2401.01505v5)**

> **作者:** Haopeng Li; Andong Deng; Jun Liu; Hossein Rahmani; Yulan Guo; Bernt Schiele; Mohammed Bennamoun; Qiuhong Ke
>
> **摘要:** Reasoning over sports videos for question answering is an important task with numerous applications, such as player training and information retrieval. However, this task has not been explored due to the lack of relevant datasets and the challenging nature it presents. Most datasets for video question answering (VideoQA) focus mainly on general and coarse-grained understanding of daily-life videos, which is not applicable to sports scenarios requiring professional action understanding and fine-grained motion analysis. In this paper, we introduce the first dataset, named Sports-QA, specifically designed for the sports VideoQA task. The Sports-QA dataset includes various types of questions, such as descriptions, chronologies, causalities, and counterfactual conditions, covering multiple sports. Furthermore, to address the characteristics of the sports VideoQA task, we propose a new Auto-Focus Transformer (AFT) capable of automatically focusing on particular scales of temporal information for question answering. We conduct extensive experiments on Sports-QA, including baseline studies and the evaluation of different methods. The results demonstrate that our AFT achieves state-of-the-art performance.
>
---
#### [replaced 046] GTPBD: A Fine-Grained Global Terraced Parcel and Boundary Dataset
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.14697v3](https://arxiv.org/pdf/2507.14697v3)**

> **作者:** Zhiwei Zhang; Zi Ye; Yibin Wen; Shuai Yuan; Haohuan Fu; Jianxi Huang; Juepeng Zheng
>
> **备注:** 40 pages, 40 figures, Accepted to the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Agricultural parcels serve as basic units for conducting agricultural practices and applications, which is vital for land ownership registration, food security assessment, soil erosion monitoring, etc. However, existing agriculture parcel extraction studies only focus on mid-resolution mapping or regular plain farmlands while lacking representation of complex terraced terrains due to the demands of precision agriculture.In this paper, we introduce a more fine-grained terraced parcel dataset named GTPBD (Global Terraced Parcel and Boundary Dataset), which is the first fine-grained dataset covering major worldwide terraced regions with more than 200,000 complex terraced parcels with manual annotation. GTPBD comprises 47,537 high-resolution images with three-level labels, including pixel-level boundary labels, mask labels, and parcel labels. It covers seven major geographic zones in China and transcontinental climatic regions around the world.Compared to the existing datasets, the GTPBD dataset brings considerable challenges due to the: (1) terrain diversity; (2) complex and irregular parcel objects; and (3) multiple domain styles. Our proposed GTPBD dataset is suitable for four different tasks, including semantic segmentation, edge detection, terraced parcel extraction, and unsupervised domain adaptation (UDA) tasks.Accordingly, we benchmark the GTPBD dataset on eight semantic segmentation methods, four edge extraction methods, three parcel extraction methods, and five UDA methods, along with a multi-dimensional evaluation framework integrating pixel-level and object-level metrics. GTPBD fills a critical gap in terraced remote sensing research, providing a basic infrastructure for fine-grained agricultural terrain analysis and cross-scenario knowledge transfer.
>
---
#### [replaced 047] Damba-ST: Domain-Adaptive Mamba for Efficient Urban Spatio-Temporal Prediction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.18939v3](https://arxiv.org/pdf/2506.18939v3)**

> **作者:** Rui An; Yifeng Zhang; Ziran Liang; Wenqi Fan; Yuxuan Liang; Xuequn Shang; Qing Li
>
> **备注:** Accepted by ICDE 2026
>
> **摘要:** Training urban spatio-temporal foundation models that generalize well across diverse regions and cities is critical for deploying urban services in unseen or data-scarce regions. Recent studies have typically focused on fusing cross-domain spatio-temporal data to train unified Transformer-based models. However, these models suffer from quadratic computational complexity and high memory overhead, limiting their scalability and practical deployment. Inspired by the efficiency of Mamba, a state space model with linear time complexity, we explore its potential for efficient urban spatio-temporal prediction. However, directly applying Mamba as a spatio-temporal backbone leads to negative transfer and severe performance degradation. This is primarily due to spatio-temporal heterogeneity and the recursive mechanism of Mamba's hidden state updates, which limit cross-domain generalization. To overcome these challenges, we propose Damba-ST, a novel domain-adaptive Mamba-based model for efficient urban spatio-temporal prediction. Damba-ST retains Mamba's linear complexity advantage while significantly enhancing its adaptability to heterogeneous domains. Specifically, we introduce two core innovations: (1) a domain-adaptive state space model that partitions the latent representation space into a shared subspace for learning cross-domain commonalities and independent, domain-specific subspaces for capturing intra-domain discriminative features; (2) three distinct Domain Adapters, which serve as domain-aware proxies to bridge disparate domain distributions and facilitate the alignment of cross-domain commonalities. Extensive experiments demonstrate the generalization and efficiency of Damba-ST. It achieves state-of-the-art performance on prediction tasks and demonstrates strong zero-shot generalization, enabling seamless deployment in new urban environments without extensive retraining or fine-tuning.
>
---
#### [replaced 048] Bridging Cognitive Gap: Hierarchical Description Learning for Artistic Image Aesthetics Assessment
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.23413v2](https://arxiv.org/pdf/2512.23413v2)**

> **作者:** Henglin Liu; Nisha Huang; Chang Liu; Jiangpeng Yan; Huijuan Huang; Jixuan Ying; Tong-Yee Lee; Pengfei Wan; Xiangyang Ji
>
> **备注:** AAAI2026,Project Page:https://github.com/Henglin-Liu/ArtQuant
>
> **摘要:** The aesthetic quality assessment task is crucial for developing a human-aligned quantitative evaluation system for AIGC. However, its inherently complex nature, spanning visual perception, cognition, and emotion, poses fundamental challenges. Although aesthetic descriptions offer a viable representation of this complexity, two critical challenges persist: (1) data scarcity and imbalance: existing dataset overly focuses on visual perception and neglects deeper dimensions due to the expensive manual annotation; and (2) model fragmentation: current visual networks isolate aesthetic attributes with multi-branch encoder, while multimodal methods represented by contrastive learning struggle to effectively process long-form textual descriptions. To resolve challenge (1), we first present the Refined Aesthetic Description (RAD) dataset, a large-scale (70k), multi-dimensional structured dataset, generated via an iterative pipeline without heavy annotation costs and easy to scale. To address challenge (2), we propose ArtQuant, an aesthetics assessment framework for artistic images which not only couples isolated aesthetic dimensions through joint description generation, but also better models long-text semantics with the help of LLM decoders. Besides, theoretical analysis confirms this symbiosis: RAD's semantic adequacy (data) and generation paradigm (model) collectively minimize prediction entropy, providing mathematical grounding for the framework. Our approach achieves state-of-the-art performance on several datasets while requiring only 33% of conventional training epochs, narrowing the cognitive gap between artistic images and aesthetic judgment. We will release both code and dataset to support future research.
>
---
#### [replaced 049] SurgWorld: Learning Surgical Robot Policies from Videos via World Modeling
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于手术机器人领域，解决数据稀缺问题。通过构建SurgWorld模型和SATA数据集，生成合成视频动作数据，提升手术机器人自主学习效果。**

- **链接: [https://arxiv.org/pdf/2512.23162v3](https://arxiv.org/pdf/2512.23162v3)**

> **作者:** Yufan He; Pengfei Guo; Mengya Xu; Zhaoshuo Li; Andriy Myronenko; Dillan Imans; Bingjie Liu; Dongren Yang; Mingxue Gu; Yongnan Ji; Yueming Jin; Ren Zhao; Baiyong Shen; Daguang Xu
>
> **摘要:** Data scarcity remains a fundamental barrier to achieving fully autonomous surgical robots. While large scale vision language action (VLA) models have shown impressive generalization in household and industrial manipulation by leveraging paired video action data from diverse domains, surgical robotics suffers from the paucity of datasets that include both visual observations and accurate robot kinematics. In contrast, vast corpora of surgical videos exist, but they lack corresponding action labels, preventing direct application of imitation learning or VLA training. In this work, we aim to alleviate this problem by learning policy models from SurgWorld, a world model designed for surgical physical AI. We curated the Surgical Action Text Alignment (SATA) dataset with detailed action description specifically for surgical robots. Then we built SurgeWorld based on the most advanced physical AI world model and SATA. It's able to generate diverse, generalizable and realistic surgery videos. We are also the first to use an inverse dynamics model to infer pseudokinematics from synthetic surgical videos, producing synthetic paired video action data. We demonstrate that a surgical VLA policy trained with these augmented data significantly outperforms models trained only on real demonstrations on a real surgical robot platform. Our approach offers a scalable path toward autonomous surgical skill acquisition by leveraging the abundance of unlabeled surgical video and generative world modeling, thus opening the door to generalizable and data efficient surgical robot policies.
>
---
#### [replaced 050] HCVP: Leveraging Hierarchical Contrastive Visual Prompt for Domain Generalization
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2401.09716v2](https://arxiv.org/pdf/2401.09716v2)**

> **作者:** Guanglin Zhou; Zhongyi Han; Shiming Chen; Biwei Huang; Liming Zhu; Tongliang Liu; Lina Yao; Kun Zhang
>
> **摘要:** Domain Generalization (DG) endeavors to create machine learning models that excel in unseen scenarios by learning invariant features. In DG, the prevalent practice of constraining models to a fixed structure or uniform parameterization to encapsulate invariant features can inadvertently blend specific aspects. Such an approach struggles with nuanced differentiation of inter-domain variations and may exhibit bias towards certain domains, hindering the precise learning of domain-invariant features. Recognizing this, we introduce a novel method designed to supplement the model with domain-level and task-specific characteristics. This approach aims to guide the model in more effectively separating invariant features from specific characteristics, thereby boosting the generalization. Building on the emerging trend of visual prompts in the DG paradigm, our work introduces the novel \textbf{H}ierarchical \textbf{C}ontrastive \textbf{V}isual \textbf{P}rompt (HCVP) methodology. This represents a significant advancement in the field, setting itself apart with a unique generative approach to prompts, alongside an explicit model structure and specialized loss functions. Differing from traditional visual prompts that are often shared across entire datasets, HCVP utilizes a hierarchical prompt generation network enhanced by prompt contrastive learning. These generative prompts are instance-dependent, catering to the unique characteristics inherent to different domains and tasks. Additionally, we devise a prompt modulation network that serves as a bridge, effectively incorporating the generated visual prompts into the vision transformer backbone. Experiments conducted on five DG datasets demonstrate the effectiveness of HCVP, outperforming both established DG algorithms and adaptation protocols.
>
---
#### [replaced 051] Training-Free Adaptive Quantization for Variable Rate Image Coding for Machines
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.05836v3](https://arxiv.org/pdf/2511.05836v3)**

> **作者:** Yui Tatsumi; Ziyue Zeng; Hiroshi Watanabe
>
> **备注:** Accepted to IEEE 44th International Conference on Consumer Electronics (ICCE 2026)
>
> **摘要:** Image Coding for Machines (ICM) has become increasingly important with the rapid integration of computer vision technology into real-world applications. However, most neural network-based ICM frameworks operate at a fixed rate, thus requiring individual training for each target bitrate. This limitation may restrict their practical usage. Existing variable rate image compression approaches mitigate this issue but often rely on additional training, which increases computational costs and complicates deployment. Moreover, variable rate control has not been thoroughly explored for ICM. To address these challenges, we propose a training-free framework for quantization strength control which enables flexible bitrate adjustment. By exploiting the scale parameter predicted by the hyperprior network, the proposed method adaptively modulates quantization step sizes across both channel and spatial dimensions. This allows the model to preserve semantically important regions while coarsely quantizing less critical areas. Our architectural design further enables continuous bitrate control through a single parameter. Experimental results demonstrate the effectiveness of our proposed method, achieving up to 11.07% BD-rate savings over the non-adaptive variable rate baseline. The code is available at https://github.com/qwert-top/AQVR-ICM.
>
---
#### [replaced 052] Fine-Grained Preference Optimization Improves Spatial Reasoning in VLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文聚焦于视觉语言模型的空间推理任务，解决其在细粒度空间逻辑推理上的不足。通过引入fDPO方法和SpatialReasoner-R1模型，提升空间对齐与逻辑一致性。**

- **链接: [https://arxiv.org/pdf/2506.21656v3](https://arxiv.org/pdf/2506.21656v3)**

> **作者:** Yifan Shen; Yuanzhe Liu; Jingyuan Zhu; Xu Cao; Xiaofeng Zhang; Yixiao He; Wenming Ye; James Matthew Rehg; Ismini Lourentzou
>
> **摘要:** Current Vision-Language Models (VLMs) struggle with fine-grained spatial reasoning, particularly when multi-step logic and precise spatial alignment are required. In this work, we introduce SpatialReasoner-R1, a vision-language reasoning model designed to address these limitations. To construct high-quality supervision for spatial reasoning, we design a Multi-Model Monte Carlo Tree Search (M3CTS) method that generates diverse, logically consistent Long Chain-of-Thought (LongCOT) reasoning trajectories. In addition, we propose a fine-grained Direct Preference Optimization (fDPO) method that introduces segment-specific preference granularity for descriptive grounding and logical reasoning, guided by a spatial reward mechanism that evaluates candidate responses based on visual consistency, spatial grounding, and logical coherence. Experimental results demonstrate that fDPO achieves relative performance gains of 4.1% and 9.0% over standard DPO on spatial qualitative and quantitative tasks, respectively. SpatialReasoner-R1, trained with fDPO, sets a new SoTA on SpatialRGPT-Bench, outperforming the strongest baseline by 9.4% in average accuracy, while maintaining competitive performance on general vision-language tasks.
>
---
#### [replaced 053] CountCluster: Training-Free Object Quantity Guidance with Cross-Attention Map Clustering for Text-to-Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.10710v2](https://arxiv.org/pdf/2508.10710v2)**

> **作者:** Joohyeon Lee; Jin-Seop Lee; Jee-Hyong Lee
>
> **备注:** Under review
>
> **摘要:** Diffusion-based text-to-image generation models have demonstrated strong performance in terms of image quality and diversity. However, they still struggle to generate images that accurately reflect the number of objects specified in the input prompt. Several approaches have been proposed that rely on either external counting modules for iterative refinement or quantity representations derived from learned tokens or latent features. However, they still have limitations in accurately reflecting the specified number of objects and overlook an important structural characteristic--The number of object instances in the generated image is largely determined in the early timesteps of the denoising process. To correctly reflect the object quantity for image generation, the highly activated regions in the object cross-attention map at the early timesteps should match the input object quantity, while each region should be clearly separated. To address this issue, we propose \textit{CountCluster}, a method that guides the object cross-attention map to be clustered according to the specified object count in the input, without relying on any external tools or additional training. The proposed method partitions the object cross-attention map into $k$ clusters at inference time based on attention scores, defines an ideal distribution in which each cluster is spatially well-separated, and optimizes the latent to align with this target distribution. Our method achieves an average improvement of 18.5\%p in object count accuracy compared to existing methods, and demonstrates superior quantity control performance across a variety of prompts. Code will be released at: https://github.com/JoohyeonL22/CountCluster
>
---
#### [replaced 054] MotionCharacter: Fine-Grained Motion Controllable Human Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.18281v3](https://arxiv.org/pdf/2411.18281v3)**

> **作者:** Haopeng Fang; Di Qiu; Binjie Mao; He Tang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Recent advancements in personalized Text-to-Video (T2V) generation have made significant strides in synthesizing character-specific content. However, these methods face a critical limitation: the inability to perform fine-grained control over motion intensity. This limitation stems from an inherent entanglement of action semantics and their corresponding magnitudes within coarse textual descriptions, hindering the generation of nuanced human videos and limiting their applicability in scenarios demanding high precision, such as animating virtual avatars or synthesizing subtle micro-expressions. Furthermore, existing approaches often struggle to preserve high identity fidelity when other attributes are modified. To address these challenges, we introduce MotionCharacter, a framework for high-fidelity human video generation with precise motion control. At its core, MotionCharacter explicitly decouples motion into two independently controllable components: action type and motion intensity. This is achieved through two key technical contributions: (1) a Motion Control Module that leverages textual phrases to specify the action type and a quantifiable metric derived from optical flow to modulate its intensity, guided by a region-aware loss that localizes motion to relevant subject areas; and (2) an ID Content Insertion Module coupled with an ID-Consistency loss to ensure robust identity preservation during dynamic motions. To facilitate training for such fine-grained control, we also curate Human-Motion, a new large-scale dataset with detailed annotations for both motion and facial features. Extensive experiments demonstrate that MotionCharacter achieves substantial improvements over existing methods. Our framework excels in generating videos that are not only identity-consistent but also precisely adhere to specified motion types and intensities.
>
---
#### [replaced 055] RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉-语言导航任务，旨在解决机器人在复杂3D场景中准确理解空间指代和动态推理的问题。提出RoboRefer模型，结合深度编码与强化学习，提升空间理解与多步推理能力。**

- **链接: [https://arxiv.org/pdf/2506.04308v4](https://arxiv.org/pdf/2506.04308v4)**

> **作者:** Enshen Zhou; Jingkun An; Cheng Chi; Yi Han; Shanyu Rong; Chi Zhang; Pengwei Wang; Zhongyuan Wang; Tiejun Huang; Lu Sheng; Shanghang Zhang
>
> **备注:** Accepted by NeurIPS 2025. Project page: https://zhoues.github.io/RoboRefer/
>
> **摘要:** Spatial referring is a fundamental capability of embodied robots to interact with the 3D physical world. However, even with the powerful pretrained vision language models (VLMs), recent approaches are still not qualified to accurately understand the complex 3D scenes and dynamically reason about the instruction-indicated locations for interaction. To this end, we propose RoboRefer, a 3D-aware VLM that can first achieve precise spatial understanding by integrating a disentangled but dedicated depth encoder via supervised fine-tuning (SFT). Moreover, RoboRefer advances generalized multi-step spatial reasoning via reinforcement fine-tuning (RFT), with metric-sensitive process reward functions tailored for spatial referring tasks. To support SFT and RFT training, we introduce RefSpatial, a large-scale dataset of 20M QA pairs (2x prior), covering 31 spatial relations (vs. 15 prior) and supporting complex reasoning processes (up to 5 steps). In addition, we introduce RefSpatial-Bench, a challenging benchmark filling the gap in evaluating spatial referring with multi-step reasoning. Experiments show that SFT-trained RoboRefer achieves state-of-the-art spatial understanding, with an average success rate of 89.6%. RFT-trained RoboRefer further outperforms all other baselines by a large margin, even surpassing Gemini-2.5-Pro by 17.4% in average accuracy on RefSpatial-Bench. Notably, RoboRefer can be integrated with various control policies to execute long-horizon, dynamic tasks across diverse robots (e,g., UR5, G1 humanoid) in cluttered real-world scenes.
>
---
#### [replaced 056] Joint Distillation for Fast Likelihood Evaluation and Sampling in Flow-based Models
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2512.02636v2](https://arxiv.org/pdf/2512.02636v2)**

> **作者:** Xinyue Ai; Yutong He; Albert Gu; Ruslan Salakhutdinov; J Zico Kolter; Nicholas Matthew Boffi; Max Simchowitz
>
> **摘要:** Log-likelihood evaluation enables important capabilities in generative models, including model comparison, certain fine-tuning objectives, and many downstream applications. Yet paradoxically, some of today's best generative models -- diffusion and flow-based models -- still require hundreds to thousands of neural function evaluations (NFEs) to compute a single likelihood. While recent distillation methods have successfully accelerated sampling to just a few steps, they achieve this at the cost of likelihood tractability: existing approaches either abandon likelihood computation entirely or still require expensive integration over full trajectories. We present fast flow joint distillation (F2D2), a framework that simultaneously reduces the number of NFEs required for both sampling and likelihood evaluation by two orders of magnitude. Our key insight is that in continuous normalizing flows, the coupled ODEs for sampling and likelihood are computed from a shared underlying velocity field, allowing us to jointly distill both the sampling trajectory and cumulative divergence using a single model. F2D2 is modular, compatible with existing flow-based few-step sampling models, and requires only an additional divergence prediction head. Experiments demonstrate F2D2's capability of achieving accurate log-likelihood with few-step evaluations while maintaining high sample quality, solving a long-standing computational bottleneck in flow-based generative models. As an application of our approach, we propose a lightweight self-guidance method that enables a 2-step MeanFlow to outperform a 1024 step flow matching model with only a single additional backward NFE.
>
---
#### [replaced 057] Learning the Language of Histopathology Images reveals Prognostic Subgroups in Invasive Lung Adenocarcinoma Patients
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.16742v2](https://arxiv.org/pdf/2508.16742v2)**

> **作者:** Abdul Rehman Akbar; Usama Sajjad; Ziyu Su; Wencheng Li; Fei Xing; Jimmy Ruiz; Wei Chen; Muhammad Khalid Khan Niazi
>
> **摘要:** Recurrence remains a major clinical challenge in surgically resected invasive lung adenocarcinoma, where existing grading and staging systems fail to capture the cellular complexity that underlies tumor aggressiveness. We present PathRosetta, a novel AI model that conceptualizes histopathology as a language, where cells serve as words, spatial neighborhoods form syntactic structures, and tissue architecture composes sentences. By learning this language of histopathology, PathRosetta predicts five-year recurrence directly from hematoxylin-and-eosin (H&E) slides, treating them as documents representing the state of the disease. In a multi-cohort dataset of 289 patients (600 slides), PathRosetta achieved an area under the curve (AUC) of 0.78 +- 0.04 on the internal cohort, significantly outperforming IASLC grading (AUC:0.71), AJCC staging (AUC:0.64), and other state-of-the-art AI models (AUC:0.62-0.67). It yielded a hazard ratio of 9.54 and a concordance index of 0.70, generalized robustly to external TCGA (AUC:0.75) and CPTAC (AUC:0.76) cohorts, and performed consistently across demographic and clinical subgroups. Beyond whole-slide prediction, PathRosetta uncovered prognostic subgroups within individual cell types, revealing that even within benign epithelial, stromal, or other cells, distinct morpho-spatial phenotypes correspond to divergent outcomes. Moreover, because the model explicitly understands what it is looking at, including cell types, cellular neighborhoods, and higher-order tissue morphology, it is inherently interpretable and can articulate the rationale behind its predictions. These findings establish that representing histopathology as a language enables interpretable and generalizable prognostication from routine histology.
>
---
#### [replaced 058] Attire-Based Anomaly Detection in Restricted Areas Using YOLOv8 for Enhanced CCTV Security
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2404.00645v2](https://arxiv.org/pdf/2404.00645v2)**

> **作者:** Abdul Aziz A. B; Aindri Bajpai
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** This research introduces an innovative security enhancement approach, employing advanced image analysis and soft computing. The focus is on an intelligent surveillance system that detects unauthorized individuals in restricted areas by analyzing attire. Traditional security measures face challenges in monitoring unauthorized access. Leveraging YOLOv8, an advanced object detection algorithm, our system identifies authorized personnel based on their attire in CCTV footage. The methodology involves training the YOLOv8 model on a comprehensive dataset of uniform patterns, ensuring precise recognition in specific regions. Soft computing techniques enhance adaptability to dynamic environments and varying lighting conditions. This research contributes to image analysis and soft computing, providing a sophisticated security solution. Emphasizing uniform-based anomaly detection, it establishes a foundation for robust security systems in restricted areas. The outcomes highlight the potential of YOLOv8-based surveillance in ensuring safety in sensitive locations.
>
---
#### [replaced 059] PrevMatch: Revisiting and Maximizing Temporal Knowledge in Semi-Supervised Semantic Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2405.20610v2](https://arxiv.org/pdf/2405.20610v2)**

> **作者:** Wooseok Shin; Hyun Joon Park; Jin Sob Kim; Juan Yun; Se Hong Park; Sung Won Han
>
> **备注:** To appear in WACV 2026. Code: https://github.com/wooseok-shin/PrevMatch
>
> **摘要:** In semi-supervised semantic segmentation, the Mean Teacher- and co-training-based approaches are employed to mitigate confirmation bias and coupling problems. However, despite their high performance, these approaches frequently involve complex training pipelines and a substantial computational burden, limiting the scalability and compatibility of these methods. In this paper, we propose a PrevMatch framework that effectively mitigates the aforementioned limitations by maximizing the utilization of the temporal knowledge obtained during the training process. The PrevMatch framework relies on two core strategies: (1) we reconsider the use of temporal knowledge and thus directly utilize previous models obtained during training to generate additional pseudo-label guidance, referred to as previous guidance. (2) we design a highly randomized ensemble strategy to maximize the effectiveness of the previous guidance. PrevMatch, a simple yet effective plug-in method, can be seamlessly integrated into existing semi-supervised learning frameworks with minimal computational overhead. Experimental results on three benchmark semantic segmentation datasets show that incorporating PrevMatch into existing methods significantly improves their performance. Furthermore, our analysis indicates that PrevMatch facilitates stable optimization during training, resulting in improved generalization performance.
>
---
#### [replaced 060] RaffeSDG: Random Frequency Filtering enabled Single-source Domain Generalization for Medical Image Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2405.01228v3](https://arxiv.org/pdf/2405.01228v3)**

> **作者:** Heng Li; Haojin Li; Jianyu Chen; Mingyang Ou; Hai Shu; Heng Miao
>
> **摘要:** Deep learning models often encounter challenges in making accurate inferences when there are domain shifts between the source and target data. This issue is particularly pronounced in clinical settings due to the scarcity of annotated data resulting from the professional and private nature of medical data. Although various cross-domain strategies have been explored, including frequency-based approaches that vary appearance while preserving semantics, many remain limited by data constraints and computational cost. To tackle domain shifts in data-scarce medical scenarios, we propose a Random frequency filtering enabled Single-source Domain Generalization algorithm (RaffeSDG), which promises robust out-of-domain inference with segmentation models trained on a single-source domain. A frequency filter-based data augmentation strategy is first proposed to promote domain variability within a single-source domain by introducing variations in frequency space and blending homologous samples. Then Gaussian filter-based structural saliency is also leveraged to learn robust representations across augmented samples, further facilitating the training of generalizable segmentation models. To validate the effectiveness of RaffeSDG, we conducted extensive experiments involving out-of-domain inference on segmentation tasks for three human tissues imaged by four diverse modalities. Through thorough investigations and comparisons, compelling evidence was observed in these experiments, demonstrating the potential and generalizability of RaffeSDG. The code is available at https://github.com/liamheng/Non-IID_Medical_Image_Segmentation.
>
---
#### [replaced 061] Conditional Diffusion Model with Anatomical-Dose Dual Constraints for End-to-End Multi-Tumor Dose Prediction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.02043v2](https://arxiv.org/pdf/2508.02043v2)**

> **作者:** Hui Xie; Haiqin Hu; Lijuan Ding; Qing Li; Yue Sun; Tao Tan
>
> **摘要:** Radiotherapy treatment planning often relies on time-consuming, trial-and-error adjustments that heavily depend on the expertise of specialists, while existing deep learning methods face limitations in generalization, prediction accuracy, and clinical applicability. To tackle these challenges, we propose ADDiff-Dose, an Anatomical-Dose Dual Constraints Conditional Diffusion Model for end-to-end multi-tumor dose prediction. The model employs LightweightVAE3D to compress high-dimensional CT data and integrates multimodal inputs, including target and organ-at-risk (OAR) masks and beam parameters, within a progressive noise addition and denoising framework. It incorporates conditional features via a multi-head attention mechanism and utilizes a composite loss function combining MSE, conditional terms, and KL divergence to ensure both dosimetric accuracy and compliance with clinical constraints. Evaluation on a large-scale public dataset (2,877 cases) and three external institutional cohorts (450 cases in total) demonstrates that ADDiff-Dose significantly outperforms traditional baselines, achieving an MAE of 0.101-0.154 (compared to 0.316 for UNet and 0.169 for GAN models), a DICE coefficient of 0.927 (a 6.8% improvement), and limiting spinal cord maximum dose error to within 0.1 Gy. The average plan generation time per case is reduced to 22 seconds. Ablation studies confirm that the structural encoder enhances compliance with clinical dose constraints by 28.5%. To our knowledge, this is the first study to introduce a conditional diffusion model framework for radiotherapy dose prediction, offering a generalizable and efficient solution for automated treatment planning across diverse tumor sites, with the potential to substantially reduce planning time and improve clinical workflow efficiency.
>
---
#### [replaced 062] SJTU:Spatial judgments in multimodal models towards unified segmentation through coordinate detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.02565v3](https://arxiv.org/pdf/2412.02565v3)**

> **作者:** Joongwon Chae; Zhenyu Wang; Peiwu Qin
>
> **备注:** A flaw was discovered in the experimental setup. Therefore, we are retracting the paper
>
> **摘要:** Despite significant advances in vision-language understanding, implementing image segmentation within multimodal architectures remains a fundamental challenge in modern artificial intelligence systems. Existing vision-language models, which primarily rely on backbone architectures or CLIP-based embedding learning, demonstrate inherent limitations in fine-grained spatial localization and operational capabilities. This paper introduces SJTU: Spatial Judgments in Multimodal Models - Towards Unified Segmentation through Coordinate Detection, a framework that leverages spatial coordinate understanding to bridge vision-language interaction and precise segmentation, enabling accurate target identification through natural language instructions. The framework presents an approach for integrating segmentation techniques with vision-language models through spatial inference in multimodal space. By utilizing normalized coordinate detection for bounding boxes and transforming them into actionable segmentation outputs, we establish a connection between spatial and language representations in multimodal architectures. Experimental results demonstrate superior performance across benchmark datasets, achieving IoU scores of 0.5958 on COCO 2017 and 0.6758 on Pascal VOC. Testing on a single NVIDIA RTX 3090 GPU with 512x512 resolution images yields an average inference time of 7 seconds per image, demonstrating the framework's effectiveness in both accuracy and practical deployability. The project code is available at https://github.com/jw-chae/SJTU
>
---
#### [replaced 063] MPJudge: Towards Perceptual Assessment of Music-Induced Paintings
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.07137v2](https://arxiv.org/pdf/2511.07137v2)**

> **作者:** Shiqi Jiang; Tianyi Liang; Huayuan Ye; Changbo Wang; Chenhui Li
>
> **摘要:** Music induced painting is a unique artistic practice, where visual artworks are created under the influence of music. Evaluating whether a painting faithfully reflects the music that inspired it poses a challenging perceptual assessment task. Existing methods primarily rely on emotion recognition models to assess the similarity between music and painting, but such models introduce considerable noise and overlook broader perceptual cues beyond emotion. To address these limitations, we propose a novel framework for music induced painting assessment that directly models perceptual coherence between music and visual art. We introduce MPD, the first large scale dataset of music painting pairs annotated by domain experts based on perceptual coherence. To better handle ambiguous cases, we further collect pairwise preference annotations. Building on this dataset, we present MPJudge, a model that integrates music features into a visual encoder via a modulation based fusion mechanism. To effectively learn from ambiguous cases, we adopt Direct Preference Optimization for training. Extensive experiments demonstrate that our method outperforms existing approaches. Qualitative results further show that our model more accurately identifies music relevant regions in paintings.
>
---
#### [replaced 064] Autoregressive Semantic Visual Reconstruction Helps VLMs Understand Better
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态学习任务，旨在解决LVLMs对视觉信息理解不足的问题。通过引入ASVR，联合学习视觉与文本模态，提升模型对视觉内容的感知能力。**

- **链接: [https://arxiv.org/pdf/2506.09040v2](https://arxiv.org/pdf/2506.09040v2)**

> **作者:** Dianyi Wang; Wei Song; Yikun Wang; Siyuan Wang; Kaicheng Yu; Zhongyu Wei; Jiaqi Wang
>
> **摘要:** Typical large vision-language models (LVLMs) apply autoregressive supervision solely to textual sequences, without fully incorporating the visual modality into the learning process. This results in three key limitations: (1) an inability to utilize images without accompanying captions, (2) the risk that captions omit critical visual details, and (3) the challenge that certain vision-centric content cannot be adequately conveyed through text. As a result, current LVLMs often prioritize vision-to-language alignment while potentially overlooking fine-grained visual information. While some prior works have explored autoregressive image generation, effectively leveraging autoregressive visual supervision to enhance image understanding remains an open challenge. In this paper, we introduce Autoregressive Semantic Visual Reconstruction (ASVR), which enables joint learning of visual and textual modalities within a unified autoregressive framework. We show that autoregressively reconstructing the raw visual appearance of images does not enhance and may even impair multimodal understanding. In contrast, autoregressively reconstructing the semantic representation of images consistently improves comprehension. Notably, we find that even when models are given continuous image features as input, they can effectively reconstruct discrete semantic tokens, resulting in stable and consistent improvements across a wide range of multimodal understanding benchmarks. Our approach delivers significant performance gains across varying data scales (556k-2M) and types of LLM bacbones. Specifically, ASVR improves LLaVA-1.5 by 5% in average scores across 14 multimodal benchmarks. The code is available at https://github.com/AlenjandroWang/ASVR.
>
---
#### [replaced 065] CAT: Circular-Convolutional Attention for Sub-Quadratic Transformers
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于自然语言处理任务，旨在解决Transformer模型复杂度高、难以扩展的问题。提出CAT方法，通过循环卷积降低计算复杂度至O(NlogN)，提升效率与可扩展性。**

- **链接: [https://arxiv.org/pdf/2504.06704v2](https://arxiv.org/pdf/2504.06704v2)**

> **作者:** Yoshihiro Yamada
>
> **备注:** Accepted as a poster at NeurIPS 2025
>
> **摘要:** Transformers have driven remarkable breakthroughs in natural language processing and computer vision, yet their standard attention mechanism still imposes O(N^2) complexity, hindering scalability to longer sequences. We introduce Circular-convolutional ATtention (CAT), a Fourier-based approach that efficiently applies circular convolutions to reduce complexity without sacrificing representational power. CAT achieves O(NlogN) computations, requires fewer learnable parameters by streamlining fully connected layers, and introduces no additional heavy operations, resulting in consistent accuracy improvements and about a 10% speedup in naive PyTorch implementations. Based on the Engineering-Isomorphic Transformers (EITs) framework, CAT's design not only offers practical efficiency and ease of implementation, but also provides insights to guide the development of future high-performance Transformer architectures. Finally, our ablation studies highlight the key conditions underlying CAT's success, shedding light on broader principles for scalable attention mechanisms.
>
---
#### [replaced 066] MemeMind: A Large-Scale Multimodal Dataset with Chain-of-Thought Reasoning for Harmful Meme Detection
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于有害表情包检测任务，旨在解决隐含有害内容识别困难的问题。构建了MemeMind数据集并提出MemeGuard模型，提升检测准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2506.18919v2](https://arxiv.org/pdf/2506.18919v2)**

> **作者:** Hexiang Gu; Qifan Yu; Yuan Liu; Zikang Li; Saihui Hou; Jian Zhao; Zhaofeng He
>
> **摘要:** As a multimodal medium combining images and text, memes frequently convey implicit harmful content through metaphors and humor, rendering the detection of harmful memes a complex and challenging task. Although recent studies have made progress in detection accuracy and interpretability, large-scale, high-quality datasets for harmful memes remain scarce, and current methods still struggle to capture implicit risks and nuanced semantics. Thus, we construct MemeMind, a large-scale harmful meme dataset. Aligned with the international standards and the context of internet, MemeMind provides detailed Chain-of-Thought (CoT) reasoning annotations to support fine-grained analysis of implicit intentions in memes. Based on this dataset, we further propose MemeGuard, a reasoning-oriented multimodal detection model that significantly improves both the accuracy of harmful meme detection and the interpretability of model decisions. Extensive experimental results demonstrate that MemeGuard outperforms existing state-of-the-art methods on the MemeMind dataset, establishing a solid foundation for future research in harmful meme detection.
>
---
#### [replaced 067] SDEval: Safety Dynamic Evaluation for Multimodal Large Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.06142v2](https://arxiv.org/pdf/2508.06142v2)**

> **作者:** Hanqing Wang; Yuan Tian; Mingyu Liu; Zhenhao Zhang; Xiangyang Zhu
>
> **备注:** AAAI 2026 poster
>
> **摘要:** In the rapidly evolving landscape of Multimodal Large Language Models (MLLMs), the safety concerns of their outputs have earned significant attention. Although numerous datasets have been proposed, they may become outdated with MLLM advancements and are susceptible to data contamination issues. To address these problems, we propose \textbf{SDEval}, the \textit{first} safety dynamic evaluation framework to controllably adjust the distribution and complexity of safety benchmarks. Specifically, SDEval mainly adopts three dynamic strategies: text, image, and text-image dynamics to generate new samples from original benchmarks. We first explore the individual effects of text and image dynamics on model safety. Then, we find that injecting text dynamics into images can further impact safety, and conversely, injecting image dynamics into text also leads to safety risks. SDEval is general enough to be applied to various existing safety and even capability benchmarks. Experiments across safety benchmarks, MLLMGuard and VLSBench, and capability benchmarks, MMBench and MMVet, show that SDEval significantly influences safety evaluation, mitigates data contamination, and exposes safety limitations of MLLMs. Code is available at https://github.com/hq-King/SDEval
>
---
#### [replaced 068] MeSS: City Mesh-Guided Outdoor Scene Generation with Cross-View Consistent Diffusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.15169v3](https://arxiv.org/pdf/2508.15169v3)**

> **作者:** Xuyang Chen; Zhijun Zhai; Kaixuan Zhou; Zengmao Wang; Jianan He; Dong Wang; Yanfeng Zhang; mingwei Sun; Rüdiger Westermann; Konrad Schindler; Liqiu Meng
>
> **摘要:** Mesh models have become increasingly accessible for numerous cities; however, the lack of realistic textures restricts their application in virtual urban navigation and autonomous driving. To address this, this paper proposes MeSS (Meshbased Scene Synthesis) for generating high-quality, styleconsistent outdoor scenes with city mesh models serving as the geometric prior. While image and video diffusion models can leverage spatial layouts (such as depth maps or HD maps) as control conditions to generate street-level perspective views, they are not directly applicable to 3D scene generation. Video diffusion models excel at synthesizing consistent view sequences that depict scenes but often struggle to adhere to predefined camera paths or align accurately with rendered control videos. In contrast, image diffusion models, though unable to guarantee cross-view visual consistency, can produce more geometry-aligned results when combined with ControlNet. Building on this insight, our approach enhances image diffusion models by improving cross-view consistency. The pipeline comprises three key stages: first, we generate geometrically consistent sparse views using Cascaded Outpainting ControlNets; second, we propagate denser intermediate views via a component dubbed AGInpaint; and third, we globally eliminate visual inconsistencies (e.g., varying exposure) using the GCAlign module. Concurrently with generation, a 3D Gaussian Splatting (3DGS) scene is reconstructed by initializing Gaussian balls on the mesh surface. Our method outperforms existing approaches in both geometric alignment and generation quality. Once synthesized, the scene can be rendered in diverse styles through relighting and style transfer techniques. project page: https://albertchen98.github.io/mess/
>
---
#### [replaced 069] Training-Free Video Editing via Optical Flow-Enhanced Score Distillation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2406.04888v3](https://arxiv.org/pdf/2406.04888v3)**

> **作者:** Lianghan Zhu; Yanqi Bao; Jing Huo; Jing Wu; Yu-Kun Lai; Wenbin Li; Yang Gao
>
> **摘要:** The rapid advancement in visual generation, particularly the emergence of pre-trained text-to-image and text-to-video models, has catalyzed growing interest in training-free video editing research. Mirroring training-free image editing techniques, current approaches preserve original video information through video input inversion and manipulating intermediate features and attention during the inference process to achieve content editing. Although they have demonstrated promising results, the lossy nature of the inversion process poses significant challenges in maintaining unedited regions of the video. Furthermore, feature and attention manipulation during inference can lead to unintended over-editing and face challenges in both local temporal continuity and global content consistency. To address these challenges, this study proposes a score distillation paradigm based on pre-trained text-to-video models, where the original video is iteratively optimized through multiple steps guided by editing gradients provided by score distillation to ultimately obtain the target video. The iterative optimization starting from the original video, combined with content preservation loss, ensures the maintenance of unedited regions in the original video and suppresses over-editing. To further guarantee video content consistency and temporal continuity, we additionally introduce a global consistency auxiliary loss and optical flow prediction-based local editing gradient smoothing. Experiments demonstrate that these strategies effectively address the aforementioned challenges, achieving comparable or superior performance across multiple dimensions including preservation of unedited regions, local temporal continuity, and global content consistency of editing results, compared to state-of-the-art methods.
>
---
#### [replaced 070] AdaVLN: Towards Visual Language Navigation in Continuous Indoor Environments with Moving Humans
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决动态室内环境中人类障碍物带来的导航挑战。提出AdaVLN框架及相应数据集，增强导航模型的适应性。**

- **链接: [https://arxiv.org/pdf/2411.18539v2](https://arxiv.org/pdf/2411.18539v2)**

> **作者:** Dillon Loh; Tomasz Bednarz; Xinxing Xia; Frank Guan
>
> **摘要:** Visual Language Navigation is a task that challenges robots to navigate in realistic environments based on natural language instructions. While previous research has largely focused on static settings, real-world navigation must often contend with dynamic human obstacles. Hence, we propose an extension to the task, termed Adaptive Visual Language Navigation (AdaVLN), which seeks to narrow this gap. AdaVLN requires robots to navigate complex 3D indoor environments populated with dynamically moving human obstacles, adding a layer of complexity to navigation tasks that mimic the real-world. To support exploration of this task, we also present AdaVLN simulator and AdaR2R datasets. The AdaVLN simulator enables easy inclusion of fully animated human models directly into common datasets like Matterport3D. We also introduce a "freeze-time" mechanism for both the navigation task and simulator, which pauses world state updates during agent inference, enabling fair comparisons and experimental reproducibility across different hardware. We evaluate several baseline models on this task, analyze the unique challenges introduced by AdaVLN, and demonstrate its potential to bridge the sim-to-real gap in VLN research.
>
---
#### [replaced 071] Hierarchical Relation-augmented Representation Generalization for Few-shot Action Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.10079v3](https://arxiv.org/pdf/2504.10079v3)**

> **作者:** Hongyu Qu; Ling Xing; Jiachao Zhang; Rui Yan; Yazhou Yao; Xiangbo Shu
>
> **摘要:** Few-shot action recognition (FSAR) aims to recognize novel action categories with few exemplars. Existing methods typically learn frame-level representations for each video by designing inter-frame temporal modeling strategies or inter-video interaction at the coarse video-level granularity. However, they treat each episode task in isolation and neglect fine-grained temporal relation modeling between videos, thus failing to capture shared fine-grained temporal patterns across videos and reuse temporal knowledge from historical tasks. In light of this, we propose HR2G-shot, a Hierarchical Relation-augmented Representation Generalization framework for FSAR, which unifies three types of relation modeling (inter-frame, inter-video, and inter-task) to learn task-specific temporal patterns from a holistic view. Going beyond conducting inter-frame temporal interactions, we further devise two components to respectively explore inter-video and inter-task relationships: i) Inter-video Semantic Correlation (ISC) performs cross-video frame-level interactions in a fine-grained manner, thereby capturing task-specific query features and enhancing both intra-class consistency and inter-class separability; ii) Inter-task Knowledge Transfer (IKT) retrieves and aggregates relevant temporal knowledge from the bank, which stores diverse temporal patterns from historical episode tasks. Extensive experiments on five benchmarks show that HR2G-shot outperforms current top-leading FSAR methods.
>
---
#### [replaced 072] TraveLLaMA: A Multimodal Travel Assistant with Large-Scale Dataset and Structured Reasoning
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2504.16505v2](https://arxiv.org/pdf/2504.16505v2)**

> **作者:** Meng Chu; Yukang Chen; Haokun Gui; Shaozuo Yu; Yi Wang; Jiaya Jia
>
> **备注:** AAAI 2026 Oral
>
> **摘要:** Tourism and travel planning increasingly rely on digital assistance, yet existing multimodal AI systems often lack specialized knowledge and contextual understanding of urban environments. We present TraveLLaMA, a specialized multimodal language model designed for comprehensive travel assistance. Our work addresses the fundamental challenge of developing practical AI travel assistants through three key contributions: (1) TravelQA, a novel dataset of 265k question-answer pairs combining 160k text QA from authentic travel sources, 100k vision-language QA featuring maps and location imagery, and 5k expert-annotated Chain-of-Thought reasoning examples; (2) Travel-CoT, a structured reasoning framework that decomposes travel queries into spatial, temporal, and practical dimensions, improving answer accuracy by 10.8\% while providing interpretable decision paths; and (3) an interactive agent system validated through extensive user studies. Through fine-tuning experiments on state-of-the-art vision-language models (LLaVA, Qwen-VL, Shikra), we achieve 6.2-9.4\% base improvements, further enhanced by Travel-CoT reasoning. Our model demonstrates superior capabilities in contextual travel recommendations, map interpretation, and scene understanding while providing practical information such as operating hours and cultural insights. User studies with 500 participants show TraveLLaMA achieves a System Usability Scale score of 82.5, significantly outperforming general-purpose models and establishing new standards for multimodal travel assistance systems.
>
---
#### [replaced 073] Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.22496v3](https://arxiv.org/pdf/2509.22496v3)**

> **作者:** Ruoyu Chen; Xiaoqing Guo; Kangwei Liu; Siyuan Liang; Shiming Liu; Qunli Zhang; Laiyuan Wang; Hua Zhang; Xiaochun Cao
>
> **摘要:** Multimodal large language models (MLLMs) have demonstrated remarkable capabilities in aligning visual inputs with natural language outputs. Yet, the extent to which generated tokens depend on visual modalities remains poorly understood, limiting interpretability and reliability. In this work, we present EAGLE, a lightweight black-box framework for explaining autoregressive token generation in MLLMs. EAGLE attributes any selected tokens to compact perceptual regions while quantifying the relative influence of language priors and perceptual evidence. The framework introduces an objective function that unifies sufficiency (insight score) and indispensability (necessity score), optimized via greedy search over sparsified image regions for faithful and efficient attribution. Beyond spatial attribution, EAGLE performs modality-aware analysis that disentangles what tokens rely on, providing fine-grained interpretability of model decisions. Extensive experiments across open-source MLLMs show that EAGLE consistently outperforms existing methods in faithfulness, localization, and hallucination diagnosis, while requiring substantially less GPU memory. These results highlight its effectiveness and practicality for advancing the interpretability of MLLMs.
>
---
#### [replaced 074] PMGS: Reconstruction of Projectile Motion Across Large Spatiotemporal Spans via 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.02660v3](https://arxiv.org/pdf/2508.02660v3)**

> **作者:** Yijun Xu; Jingrui Zhang; Yuhan Chen; Dingwen Wang; Lei Yu; Chu He
>
> **摘要:** Modeling complex rigid motion across large spatiotemporal spans remains an unresolved challenge in dynamic reconstruction. Existing paradigms are mainly confined to short-term, small-scale deformation and offer limited consideration for physical consistency. This study proposes PMGS, focusing on reconstructing Projectile Motion via 3D Gaussian Splatting. The workflow comprises two stages: 1) Target Modeling: achieving object-centralized reconstruction through dynamic scene decomposition and an improved point density control; 2) Motion Recovery: restoring full motion sequences by learning per-frame SE(3) poses. We introduce an acceleration consistency constraint to bridge Newtonian mechanics and pose estimation, and design a dynamic simulated annealing strategy that adaptively schedules learning rates based on motion states. Furthermore, we devise a Kalman fusion scheme to optimize error accumulation from multi-source observations to mitigate disturbances. Experiments show PMGS's superior performance in reconstructing high-speed nonlinear rigid motion compared to mainstream dynamic methods.
>
---
#### [replaced 075] VisualActBench: Can VLMs See and Act like a Human?
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.09907v2](https://arxiv.org/pdf/2512.09907v2)**

> **作者:** Daoan Zhang; Pai Liu; Xiaofei Zhou; Yuan Ge; Guangchen Lan; Jing Bi; Christopher Brinton; Ehsan Hoque; Jiebo Luo
>
> **摘要:** Vision-Language Models (VLMs) have achieved impressive progress in perceiving and describing visual environments. However, their ability to proactively reason and act based solely on visual inputs, without explicit textual prompts, remains underexplored. We introduce a new task, Visual Action Reasoning, and propose VisualActBench, a large-scale benchmark comprising 1,074 videos and 3,733 human-annotated actions across four real-world scenarios. Each action is labeled with an Action Prioritization Level (APL) and a proactive-reactive type to assess models' human-aligned reasoning and value sensitivity. We evaluate 29 VLMs on VisualActBench and find that while frontier models like GPT4o demonstrate relatively strong performance, a significant gap remains compared to human-level reasoning, particularly in generating proactive, high-priority actions. Our results highlight limitations in current VLMs' ability to interpret complex context, anticipate outcomes, and align with human decision-making frameworks. VisualActBench establishes a comprehensive foundation for assessing and improving the real-world readiness of proactive, vision-centric AI agents.
>
---
#### [replaced 076] ULTra: Unveiling Latent Token Interpretability in Transformer-Based Understanding and Segmentation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2411.12589v3](https://arxiv.org/pdf/2411.12589v3)**

> **作者:** Hesam Hosseini; Ghazal Hosseini Mighan; Amirabbas Afzali; Sajjad Amini; Amir Houmansadr
>
> **摘要:** Transformers have revolutionized Computer Vision (CV) through self-attention mechanisms. However, their complexity makes latent token representations difficult to interpret. We introduce ULTra, a framework for interpreting Transformer embeddings and uncovering meaningful semantic patterns within them. ULTra enables unsupervised semantic segmentation using pre-trained models without requiring fine-tuning. Additionally, we propose a self-supervised training approach that refines segmentation performance by learning an external transformation matrix without modifying the underlying model. Our method achieves state-of-the-art performance in unsupervised semantic segmentation, outperforming existing segmentation methods. Furthermore, we validate ULTra for model interpretation on both synthetic and real-world scenarios, including Object Selection and interpretable text summarization using LLMs, demonstrating its broad applicability in explaining the semantic structure of latent token representations.
>
---
#### [replaced 077] P2U-SLAM: A Monocular Wide-FoV SLAM System Based on Point Uncertainty and Pose Uncertainty
- **分类: cs.RO; cs.CV; eess.IV**

- **简介: 该论文提出P2U-SLAM，解决宽视场角视觉SLAM中的长期定位性能问题。通过引入点不确定性与位姿不确定性，提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2409.10143v2](https://arxiv.org/pdf/2409.10143v2)**

> **作者:** Yufan Zhang; Kailun Yang; Ze Wang; Kaiwei Wang
>
> **备注:** Accepted to IEEE Transactions on Intelligent Transportation Systems (T-ITS). The source code will be made publicly available at https://github.com/BambValley/P2U-SLAM
>
> **摘要:** This paper presents P2U-SLAM, a visual Simultaneous Localization And Mapping (SLAM) system with a wide Field of View (FoV) camera, which utilizes pose uncertainty and point uncertainty. While the wide FoV enables considerable repetitive observations of historical map points for matching cross-view features, the data properties of the historical map points and the poses of historical keyframes have changed during the optimization process. The neglect of data property changes results in the lack of partial information matrices in optimization, increasing the risk of long-term positioning performance degradation. The purpose of our research is to mitigate the risks posed by wide-FoV visual input to the SLAM system. Based on the conditional probability model, this work reveals the definite impacts of the above data properties changes on the optimization process, concretizes these impacts as point uncertainty and pose uncertainty, and gives their specific mathematical form. P2U-SLAM embeds point uncertainty into the tracking module and pose uncertainty into the local mapping module respectively, and updates these uncertainties after each optimization operation including local mapping, map merging, and loop closing. We present an exhaustive evaluation on 27 sequences from two popular public datasets with wide-FoV visual input. P2U-SLAM shows excellent performance compared with other state-of-the-art methods. The source code will be made publicly available at https://github.com/BambValley/P2U-SLAM.
>
---
#### [replaced 078] G2L:From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Knowledge Distillation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.11176v3](https://arxiv.org/pdf/2510.11176v3)**

> **作者:** Yesung Cho; Sungmin Lee; Geongyu Lee; Minkyung Lee; Jongbae Park; Dongmyung Shin
>
> **备注:** Accepted in AAAI 2026 workshop in Health Intelligence Special Theme on Foundation Models and AI Agents
>
> **摘要:** Recent studies in pathology foundation models have shown that scaling training data, diversifying cancer types, and increasing model size consistently improve their performance. However, giga-scale foundation models, which are trained on hundreds of thousands of slides covering tens of cancer types and contain billions of parameters, pose significant challenges for practical use due to their tremendous computational costs in both development and deployment. In this work, we present a novel strategy, named the G2L framework, to increase the performance of large-scale foundation models, which consist of only $15\%$ of the parameters of giga-scale models, to a comparable performance level of giga-scale models in cancer-specific tasks. Our approach applies knowledge distillation, transferring the capabilities of a giga-scale model to a large-scale model, using just 1K pathology slides of a target cancer (e.g., breast, prostate, etc.). The resulting distilled model not only outperformed state-of-the-art models of the same size (i.e., large-scale) across several benchmarks but also, interestingly, surpassed the giga-scale teacher and huge-scale models in some benchmarks. In addition, the distilled model exhibited a higher robustness index, indicating improved resilience to image variations originating from multiple institutions. These findings suggest that the proposed distillation approach for a large-scale model is a data- and parameter-efficient way to achieve giga-scale-level performance for cancer-specific applications without prohibitive computational burden.
>
---
#### [replaced 079] RS-Prune: Training-Free Data Pruning at High Ratios for Efficient Remote Sensing Diffusion Foundation Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.23239v2](https://arxiv.org/pdf/2512.23239v2)**

> **作者:** Fan Wei; Runmin Dong; Yushan Lai; Yixiang Yang; Zhaoyang Luo; Jinxiao Zhang; Miao Yang; Shuai Yuan; Jiyao Zhao; Bin Luo; Haohuan Fu
>
> **摘要:** Diffusion-based remote sensing (RS) generative foundation models are cruial for downstream tasks. However, these models rely on large amounts of globally representative data, which often contain redundancy, noise, and class imbalance, reducing training efficiency and preventing convergence. Existing RS diffusion foundation models typically aggregate multiple classification datasets or apply simplistic deduplication, overlooking the distributional requirements of generation modeling and the heterogeneity of RS imagery. To address these limitations, we propose a training-free, two-stage data pruning approach that quickly select a high-quality subset under high pruning ratios, enabling a preliminary foundation model to converge rapidly and serve as a versatile backbone for generation, downstream fine-tuning, and other applications. Our method jointly considers local information content with global scene-level diversity and representativeness. First, an entropy-based criterion efficiently removes low-information samples. Next, leveraging RS scene classification datasets as reference benchmarks, we perform scene-aware clustering with stratified sampling to improve clustering effectiveness while reducing computational costs on large-scale unlabeled data. Finally, by balancing cluster-level uniformity and sample representativeness, the method enables fine-grained selection under high pruning ratios while preserving overall diversity and representativeness. Experiments show that, even after pruning 85\% of the training data, our method significantly improves convergence and generation quality. Furthermore, diffusion foundation models trained with our method consistently achieve state-of-the-art performance across downstream tasks, including super-resolution and semantic image synthesis. This data pruning paradigm offers practical guidance for developing RS generative foundation models.
>
---
#### [replaced 080] Explainable AI Technique in Lung Cancer Detection Using Convolutional Neural Networks
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.10196v2](https://arxiv.org/pdf/2508.10196v2)**

> **作者:** Nishan Rai; Sujan Khatri; Devendra Risal
>
> **备注:** 11 pages, 9 figures, 4 tables. Undergraduate research project report
>
> **摘要:** Early detection of lung cancer is critical to improving survival outcomes. We present a deep learning framework for automated lung cancer screening from chest computed tomography (CT) images with integrated explainability. Using the IQ-OTH/NCCD dataset (1,197 scans across Normal, Benign, and Malignant classes), we evaluate a custom convolutional neural network (CNN) and three fine-tuned transfer learning backbones: DenseNet121, ResNet152, and VGG19. Models are trained with cost-sensitive learning to mitigate class imbalance and evaluated via accuracy, precision, recall, F1-score, and ROC-AUC. While ResNet152 achieved the highest accuracy (97.3%), DenseNet121 provided the best overall balance in precision, recall, and F1 (up to 92%, 90%, 91%, respectively). We further apply Shapley Additive Explanations (SHAP) to visualize evidence contributing to predictions, improving clinical transparency. Results indicate that CNN-based approaches augmented with explainability can provide fast, accurate, and interpretable support for lung cancer screening, particularly in resource-limited settings.
>
---
#### [replaced 081] Virtual Multiplex Staining for Histological Images using a Marker-wise Conditioned Diffusion Model
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.14681v4](https://arxiv.org/pdf/2508.14681v4)**

> **作者:** Hyun-Jic Oh; Junsik Kim; Zhiyi Shi; Yichen Wu; Yu-An Chen; Peter K Sorger; Hanspeter Pfister; Won-Ki Jeong
>
> **备注:** Accepted at AAAI 2026
>
> **摘要:** Multiplex imaging is revolutionizing pathology by enabling the simultaneous visualization of multiple biomarkers within tissue samples, providing molecular-level insights that traditional hematoxylin and eosin (H&E) staining cannot provide. However, the complexity and cost of multiplex data acquisition have hindered its widespread adoption. Additionally, most existing large repositories of H&E images lack corresponding multiplex images, limiting opportunities for multimodal analysis. To address these challenges, we leverage recent advances in latent diffusion models (LDMs), which excel at modeling complex data distributions by utilizing their powerful priors for fine-tuning to a target domain. In this paper, we introduce a novel framework for virtual multiplex staining that utilizes pretrained LDM parameters to generate multiplex images from H&E images using a conditional diffusion model. Our approach enables marker-by-marker generation by conditioning the diffusion model on each marker, while sharing the same architecture across all markers. To tackle the challenge of varying pixel value distributions across different marker stains and to improve inference speed, we fine-tune the model for single-step sampling, enhancing both color contrast fidelity and inference efficiency through pixel-level loss functions. We validate our framework on two publicly available datasets, notably demonstrating its effectiveness in generating up to 18 different marker types with improved accuracy, a substantial increase over the 2-3 marker types achieved in previous approaches. This validation highlights the potential of our framework, pioneering virtual multiplex staining. Finally, this paper bridges the gap between H&E and multiplex imaging, potentially enabling retrospective studies and large-scale analyses of existing H&E image repositories.
>
---
#### [replaced 082] Pretraining Frame Preservation in Autoregressive Video Memory Compression
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.23851v2](https://arxiv.org/pdf/2512.23851v2)**

> **作者:** Lvmin Zhang; Shengqu Cai; Muyang Li; Chong Zeng; Beijia Lu; Anyi Rao; Song Han; Gordon Wetzstein; Maneesh Agrawala
>
> **备注:** Github: https://github.com/lllyasviel/PFP ; Project: https://lllyasviel.github.io/pfp_gitpage/
>
> **摘要:** We present PFP, a neural network structure to compress long videos into short contexts, with an explicit pretraining objective to preserve the high-frequency details of single frames at arbitrary temporal positions. The baseline model can compress a 20-second video into a context at about 5k length, where random frames can be retrieved with perceptually preserved appearances. Such pretrained models can be directly fine-tuned as memory encoders for autoregressive video models, enabling long history memory with low context cost and relatively low fidelity loss. We evaluate the framework with ablative settings and discuss the trade-offs of possible neural architecture designs.
>
---
#### [replaced 083] AdaptInfer: Adaptive Token Pruning for Vision-Language Model Inference with Dynamical Text Guidance
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.06084v2](https://arxiv.org/pdf/2508.06084v2)**

> **作者:** Weichen Zhang; Zhui Zhu; Ningbo Li; Shilong Tao; Kebin Liu; Yunhao Liu
>
> **摘要:** Vision-language models (VLMs) have achieved impressive performance on multimodal reasoning tasks such as visual question answering, image captioning and so on, but their inference cost remains a significant challenge due to the large number of vision tokens processed during the prefill stage. Existing pruning methods often rely on directly using the attention patterns or static text prompt guidance, failing to exploit the dynamic internal signals generated during inference. To address these issues, we propose AdaptInfer, a plug-and-play framework for adaptive vision token pruning in VLMs. First, we introduce a fine-grained, dynamic text-guided pruning mechanism that reuses layer-wise text-to-text attention maps to construct soft priors over text-token importance, allowing more informed scoring of vision tokens at each stage. Second, we perform an offline analysis of cross-modal attention shifts and identify consistent inflection locations in inference, which inspire us to propose a more principled and efficient pruning schedule. Our method is lightweight and plug-and-play, also generalizable across multi-modal tasks. Experimental results have verified the effectiveness of the proposed method. For example, it reduces CUDA latency by 61.3% while maintaining an average accuracy of 93.1% on vanilla LLaVA-1.5-7B. Under the same token budget, AdaptInfer surpasses SOTA in accuracy.
>
---
#### [replaced 084] Video Detective: Seek Critical Clues Recurrently to Answer Question from Long Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.17229v2](https://arxiv.org/pdf/2512.17229v2)**

> **作者:** Henghui Du; Chunjie Zhang; Xi Chen; Chang Zhou; Di Hu
>
> **摘要:** Long Video Question-Answering (LVQA) presents a significant challenge for Multi-modal Large Language Models (MLLMs) due to immense context and overloaded information, which could also lead to prohibitive memory consumption. While existing methods attempt to address these issues by reducing visual tokens or extending model's context length, they may miss useful information or take considerable computation. In fact, when answering given questions, only a small amount of crucial information is required. Therefore, we propose an efficient question-aware memory mechanism, enabling MLLMs to recurrently seek these critical clues. Our approach, named VideoDetective, simplifies this task by iteratively processing video sub-segments. For each sub-segment, a question-aware compression strategy is employed by introducing a few special memory tokens to achieve purposefully compression. This allows models to effectively seek critical clues while reducing visual tokens. Then, due to history context could have a significant impact, we recurrently aggregate and store these memory tokens to update history context, which would be reused for subsequent sub-segments. Furthermore, to more effectively measure model's long video understanding ability, we introduce GLVC (Grounding Long Video Clues), a long video question-answering dataset, which features grounding critical and concrete clues scattered throughout entire videos. Experimental results demonstrate our method enables MLLMs with limited context length of 32K to efficiently process 100K tokens (3600 frames, an hour-long video sampled at 1fps), requiring only 2 minutes and 37GB GPU memory usage. Evaluation results across multiple long video benchmarks illustrate our method can more effectively seek critical clues from massive information.
>
---
#### [replaced 085] How Robot Dogs See the Unseeable: Improving Visual Interpretability via Peering for Exploratory Robots
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉感知任务，旨在解决机器人在植被环境中因遮挡导致的视觉理解问题。通过模仿昆虫的“窥视”动作，结合信号处理与大模型，提升机器人在部分遮挡下的感知能力。**

- **链接: [https://arxiv.org/pdf/2511.16262v4](https://arxiv.org/pdf/2511.16262v4)**

> **作者:** Oliver Bimber; Karl Dietrich von Ellenrieder; Michael Haller; Rakesh John Amala Arokia Nathan; Gianni Lunardi; Mohamed Youssef; Marco Camurri; Santos Miguel Orozco Soto; Jeremy E. Niven
>
> **摘要:** In vegetated environments, such as forests, exploratory robots play a vital role in navigating complex, cluttered environments where human access is limited and traditional equipment struggles. Visual occlusion from obstacles, such as foliage, can severely obstruct a robot's sensors, impairing scene understanding. We show that "peering", a characteristic side-to-side movement used by insects to overcome their visual limitations, can also allow robots to markedly improve visual reasoning under partial occlusion. This is accomplished by applying core signal processing principles, specifically optical synthetic aperture sensing, together with the vision reasoning capabilities of modern large multimodal models. Peering enables real-time, high-resolution, and wavelength-independent perception, which is crucial for vision-based scene understanding across a wide range of applications. The approach is low-cost and immediately deployable on any camera-equipped robot. We investigated different peering motions and occlusion masking strategies, demonstrating that, unlike peering, state-of-the-art multi-view 3D vision techniques fail in these conditions due to their high susceptibility to occlusion. Our experiments were carried out on an industrial-grade quadrupedal robot. However, the ability to peer is not limited to such platforms, but potentially also applicable to bipedal, hexapod, wheeled, or crawling platforms. Robots that can effectively see through partial occlusion will gain superior perception abilities - including enhanced scene understanding, situational awareness, camouflage breaking, and advanced navigation in complex environments.
>
---
#### [replaced 086] Wukong's 72 Transformations: High-fidelity Textured 3D Morphing via Flow Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22425v4](https://arxiv.org/pdf/2511.22425v4)**

> **作者:** Minghao Yin; Yukang Cao; Kai Han
>
> **摘要:** We present WUKONG, a novel training-free framework for high-fidelity textured 3D morphing that takes a pair of source and target prompts (image or text) as input. Unlike conventional methods -- which rely on manual correspondence matching and deformation trajectory estimation (limiting generalization and requiring costly preprocessing) -- WUKONG leverages the generative prior of flow-based transformers to produce high-fidelity 3D transitions with rich texture details. To ensure smooth shape transitions, we exploit the inherent continuity of flow-based generative processes and formulate morphing as an optimal transport barycenter problem. We further introduce a sequential initialization strategy to prevent abrupt geometric distortions and preserve identity coherence. For faithful texture preservation, we propose a similarity-guided semantic consistency mechanism that selectively retains high-frequency details and enables precise control over blending dynamics. This empowers WUKONG to support both global texture transitions and identity-preserving texture morphing, catering to diverse generation needs. Extensive quantitative and qualitative evaluations demonstrate that WUKONG significantly outperforms state-of-the-art methods, achieving superior results across diverse geometry and texture variations.
>
---
#### [replaced 087] TD3Net: A temporal densely connected multi-dilated convolutional network for lipreading
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.16073v4](https://arxiv.org/pdf/2506.16073v4)**

> **作者:** Byung Hoon Lee; Wooseok Shin; Sung Won Han
>
> **备注:** Accepted for publication in Journal of Visual Communication and Image Representation. DOI: https://doi.org/10.1016/j.jvcir.2025.104540
>
> **摘要:** The word-level lipreading approach typically employs a two-stage framework with separate frontend and backend architectures to model dynamic lip movements. Each component has been extensively studied, and in the backend architecture, temporal convolutional networks (TCNs) have been widely adopted in state-of-the-art methods. Recently, dense skip connections have been introduced in TCNs to mitigate the limited density of the receptive field, thereby improving the modeling of complex temporal representations. However, their performance remains constrained owing to potential information loss regarding the continuous nature of lip movements, caused by blind spots in the receptive field. To address this limitation, we propose TD3Net, a temporal densely connected multi-dilated convolutional network that combines dense skip connections and multi-dilated temporal convolutions as the backend architecture. TD3Net covers a wide and dense receptive field without blind spots by applying different dilation factors to skip-connected features. Experimental results on a word-level lipreading task using two large publicly available datasets, Lip Reading in the Wild (LRW) and LRW-1000, indicate that the proposed method achieves performance comparable to state-of-the-art methods. It achieved higher accuracy with fewer parameters and lower floating-point operations compared to existing TCN-based backend architectures. Moreover, visualization results suggest that our approach effectively utilizes diverse temporal features while preserving temporal continuity, presenting notable advantages in lipreading systems. The code is available at our GitHub repository (https://github.com/Leebh-kor/TD3Net).
>
---
#### [replaced 088] Loupe: A Generalizable and Adaptive Framework for Image Forgery Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.16819v2](https://arxiv.org/pdf/2506.16819v2)**

> **作者:** Yuchu Jiang; Jiaming Chu; Jian Zhao; Xin Zhang; Xu Yang; Lei Jin; Chi Zhang; Xuelong Li
>
> **备注:** There is some controversy over the methods of the content
>
> **摘要:** The proliferation of generative models has raised serious concerns about visual content forgery. Existing deepfake detection methods primarily target either image-level classification or pixel-wise localization. While some achieve high accuracy, they often suffer from limited generalization across manipulation types or rely on complex architectures. In this paper, we propose Loupe, a lightweight yet effective framework for joint deepfake detection and localization. Loupe integrates a patch-aware classifier and a segmentation module with conditional queries, allowing simultaneous global authenticity classification and fine-grained mask prediction. To enhance robustness against distribution shifts of test set, Loupe introduces a pseudo-label-guided test-time adaptation mechanism by leveraging patch-level predictions to supervise the segmentation head. Extensive experiments on the DDL dataset demonstrate that Loupe achieves state-of-the-art performance, securing the first place in the IJCAI 2025 Deepfake Detection and Localization Challenge with an overall score of 0.846. Our results validate the effectiveness of the proposed patch-level fusion and conditional query design in improving both classification accuracy and spatial localization under diverse forgery patterns. The code is available at https://github.com/Kamichanw/Loupe.
>
---
#### [replaced 089] COMPASS: High-Efficiency Deep Image Compression with Arbitrary-scale Spatial Scalability
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2309.07926v2](https://arxiv.org/pdf/2309.07926v2)**

> **作者:** Jongmin Park; Jooyoung Lee; Munchurl Kim
>
> **备注:** Accepted in ICCV 2023. Please visit our project page at https://kaist-viclab.github.io/compass-site/
>
> **摘要:** Recently, neural network (NN)-based image compression studies have actively been made and has shown impressive performance in comparison to traditional methods. However, most of the works have focused on non-scalable image compression (single-layer coding) while spatially scalable image compression has drawn less attention although it has many applications. In this paper, we propose a novel NN-based spatially scalable image compression method, called COMPASS, which supports arbitrary-scale spatial scalability. Our proposed COMPASS has a very flexible structure where the number of layers and their respective scale factors can be arbitrarily determined during inference. To reduce the spatial redundancy between adjacent layers for arbitrary scale factors, our COMPASS adopts an inter-layer arbitrary scale prediction method, called LIFF, based on implicit neural representation. We propose a combined RD loss function to effectively train multiple layers. Experimental results show that our COMPASS achieves BD-rate gain of -58.33% and -47.17% at maximum compared to SHVC and the state-of-the-art NN-based spatially scalable image compression method, respectively, for various combinations of scale factors. Our COMPASS also shows comparable or even better coding efficiency than the single-layer coding for various scale factors.
>
---
#### [replaced 090] EMLoC: Emulator-based Memory-efficient Fine-tuning with LoRA Correction
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.12015v2](https://arxiv.org/pdf/2506.12015v2)**

> **作者:** Hsi-Che Lin; Yu-Chu Yu; Kai-Po Chang; Yu-Chiang Frank Wang
>
> **备注:** Accepted to the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Project page: https://hsi-che-lin.github.io/EMLoC/
>
> **摘要:** Open-source foundation models have seen rapid adoption and development, enabling powerful general-purpose capabilities across diverse domains. However, fine-tuning large foundation models for domain-specific or personalized tasks remains prohibitively expensive for most users due to the significant memory overhead beyond that of inference. We introduce EMLoC, an Emulator-based Memory-efficient fine-tuning framework with LoRA Correction, which enables model fine-tuning within the same memory budget required for inference. EMLoC constructs a task-specific light-weight emulator using activation-aware singular value decomposition (SVD) on a small downstream calibration set. Fine-tuning then is performed on this lightweight emulator via LoRA. To tackle the misalignment between the original model and the compressed emulator, we propose a novel compensation algorithm to correct the fine-tuned LoRA module, which thus can be merged into the original model for inference. EMLoC supports flexible compression ratios and standard training pipelines, making it adaptable to a wide range of applications. Extensive experiments demonstrate that EMLoC outperforms other baselines across multiple datasets and modalities. Moreover, without quantization, EMLoC enables fine-tuning of a 38B model, which originally required 95GB of memory, on a single 24GB consumer GPU-bringing efficient and practical model adaptation to individual users.
>
---
#### [replaced 091] A Mutual-Structure Weighted Sub-Pixel Multimodal Optical Remote Sensing Image Matching Method
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.10294v2](https://arxiv.org/pdf/2508.10294v2)**

> **作者:** Tao Huang; Hongbo Pan; Nanxi Zhou; Siyuan Zou; Shun Zhou
>
> **摘要:** Sub-pixel matching of multimodal optical images is a critical step in combined application of multiple sensors. However structural noise and inconsistencies arising from variations in multimodal image responses usually limit the accuracy of matching. Phase congruency mutual-structure weighted least absolute deviation (PCWLAD) is developed as a coarse-to-fine framework. In the coarse matching stage, we preserve the complete structure and use an enhanced cross-modal similarity criterion to mitigate structural information loss by PC noise filtering. In the fine matching stage, a mutual-structure filtering and weighted least absolute deviation-based is introduced to enhance inter-modal structural consistency and accurately estimate sub-pixel displacements adaptively. Experiments on three multimodal datasets-Landsat visible-infrared, short-range visible-near-infrared, and UAV optical image pairs demonstrate that PCWLAD consistently outperforms eight state-of-the-art methods, achieving an average matching accuracy of approximately 0.4 pixels. The software and datasets are publicly available at https://github.com/huangtaocsu/PCWLAD.
>
---
#### [replaced 092] Bridging Geometry and Appearance: Topological Features for Robust Self-Supervised Segmentation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.04666v2](https://arxiv.org/pdf/2412.04666v2)**

> **作者:** Kebin Peng; Haotang Li; Zhenyu Qi; Huashan Chen; Zi Wang; Wei Zhang; Sen He; Huanrui Yang; Qing Guo
>
> **摘要:** Self-supervised semantic segmentation methods often fail when faced with appearance ambiguities. We argue that this is due to an over-reliance on unstable, appearance-based features such as shadows, glare, and local textures. We propose \textbf{GASeg}, a novel framework that bridges appearance and geometry by leveraging stable topological information. The core of our method is Differentiable Box-Counting (\textbf{DBC}) module, which quantifies multi-scale topological statistics from two parallel streams: geometric-based features and appearance-based features. To force the model to learn these stable structural representations, we introduce Topological Augmentation (\textbf{TopoAug}), an adversarial strategy that simulates real-world ambiguities by applying morphological operators to the input images. A multi-objective loss, \textbf{GALoss}, then explicitly enforces cross-modal alignment between geometric-based and appearance-based features. Extensive experiments demonstrate that GASeg achieves state-of-the-art performance on four benchmarks, including COCO-Stuff, Cityscapes, and PASCAL, validating our approach of bridging geometry and appearance via topological information.
>
---
