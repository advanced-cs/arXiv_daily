# 计算机视觉 cs.CV

- **最新发布 118 篇**

- **更新 86 篇**

## 最新发布

#### [new 001] GVCCS: A Dataset for Contrail Identification and Tracking on Visible Whole Sky Camera Sequences
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于计算机视觉与气候科学交叉任务，旨在解决飞机尾迹（contrail）识别与追踪问题。现有数据集缺乏时间跟踪和来源关联，限制了物理模型的校准。论文贡献了GVCCS数据集，包含122个可见光全天空相机视频序列，提供尾迹像素识别、个体分割与时间追踪标注，并提出统一的深度学习框架进行尾迹分析，支持更精准的气候影响评估。**

- **链接: [http://arxiv.org/pdf/2507.18330v1](http://arxiv.org/pdf/2507.18330v1)**

> **作者:** Gabriel Jarry; Ramon Dalmau; Philippe Very; Franck Ballerini; Stephania-Denisa Bocu
>
> **摘要:** Aviation's climate impact includes not only CO2 emissions but also significant non-CO2 effects, especially from contrails. These ice clouds can alter Earth's radiative balance, potentially rivaling the warming effect of aviation CO2. Physics-based models provide useful estimates of contrail formation and climate impact, but their accuracy depends heavily on the quality of atmospheric input data and on assumptions used to represent complex processes like ice particle formation and humidity-driven persistence. Observational data from remote sensors, such as satellites and ground cameras, could be used to validate and calibrate these models. However, existing datasets don't explore all aspect of contrail dynamics and formation: they typically lack temporal tracking, and do not attribute contrails to their source flights. To address these limitations, we present the Ground Visible Camera Contrail Sequences (GVCCS), a new open data set of contrails recorded with a ground-based all-sky camera in the visible range. Each contrail is individually labeled and tracked over time, allowing a detailed analysis of its lifecycle. The dataset contains 122 video sequences (24,228 frames) and includes flight identifiers for contrails that form above the camera. As reference, we also propose a unified deep learning framework for contrail analysis using a panoptic segmentation model that performs semantic segmentation (contrail pixel identification), instance segmentation (individual contrail separation), and temporal tracking in a single architecture. By providing high-quality, temporally resolved annotations and a benchmark for model evaluation, our work supports improved contrail monitoring and will facilitate better calibration of physical models. This sets the groundwork for more accurate climate impact understanding and assessments.
>
---
#### [new 002] Identifying Prompted Artist Names from Generated Images
- **分类: cs.CV**

- **简介: 该论文属于图像识别任务，旨在解决从生成图像中识别提示中指定的艺术家名称的问题。作者构建了一个包含195万张图像的数据集，并评估多种模型在不同泛化设置下的表现，推动对文本到图像模型的负责任监管。**

- **链接: [http://arxiv.org/pdf/2507.18633v1](http://arxiv.org/pdf/2507.18633v1)**

> **作者:** Grace Su; Sheng-Yu Wang; Aaron Hertzmann; Eli Shechtman; Jun-Yan Zhu; Richard Zhang
>
> **备注:** Project page: https://graceduansu.github.io/IdentifyingPromptedArtists
>
> **摘要:** A common and controversial use of text-to-image models is to generate pictures by explicitly naming artists, such as "in the style of Greg Rutkowski". We introduce a benchmark for prompted-artist recognition: predicting which artist names were invoked in the prompt from the image alone. The dataset contains 1.95M images covering 110 artists and spans four generalization settings: held-out artists, increasing prompt complexity, multiple-artist prompts, and different text-to-image models. We evaluate feature similarity baselines, contrastive style descriptors, data attribution methods, supervised classifiers, and few-shot prototypical networks. Generalization patterns vary: supervised and few-shot models excel on seen artists and complex prompts, whereas style descriptors transfer better when the artist's style is pronounced; multi-artist prompts remain the most challenging. Our benchmark reveals substantial headroom and provides a public testbed to advance the responsible moderation of text-to-image models. We release the dataset and benchmark to foster further research: https://graceduansu.github.io/IdentifyingPromptedArtists/
>
---
#### [new 003] FishDet-M: A Unified Large-Scale Benchmark for Robust Fish Detection and CLIP-Guided Model Selection in Diverse Aquatic Visual Domains
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于目标检测任务，旨在解决水下鱼类检测中数据碎片化、成像条件多样和评估标准不统一的问题。论文构建了大规模统一基准 FishDet-M，整合13个公开数据集，并系统评估28种检测模型。同时提出基于CLIP的零样本模型选择框架，提升跨域检测的适应性和效率。**

- **链接: [http://arxiv.org/pdf/2507.17859v1](http://arxiv.org/pdf/2507.17859v1)**

> **作者:** Muayad Abujabal; Lyes Saad Saoud; Irfan Hussain
>
> **摘要:** Accurate fish detection in underwater imagery is essential for ecological monitoring, aquaculture automation, and robotic perception. However, practical deployment remains limited by fragmented datasets, heterogeneous imaging conditions, and inconsistent evaluation protocols. To address these gaps, we present \textit{FishDet-M}, the largest unified benchmark for fish detection, comprising 13 publicly available datasets spanning diverse aquatic environments including marine, brackish, occluded, and aquarium scenes. All data are harmonized using COCO-style annotations with both bounding boxes and segmentation masks, enabling consistent and scalable cross-domain evaluation. We systematically benchmark 28 contemporary object detection models, covering the YOLOv8 to YOLOv12 series, R-CNN based detectors, and DETR based models. Evaluations are conducted using standard metrics including mAP, mAP@50, and mAP@75, along with scale-specific analyses (AP$_S$, AP$_M$, AP$_L$) and inference profiling in terms of latency and parameter count. The results highlight the varying detection performance across models trained on FishDet-M, as well as the trade-off between accuracy and efficiency across models of different architectures. To support adaptive deployment, we introduce a CLIP-based model selection framework that leverages vision-language alignment to dynamically identify the most semantically appropriate detector for each input image. This zero-shot selection strategy achieves high performance without requiring ensemble computation, offering a scalable solution for real-time applications. FishDet-M establishes a standardized and reproducible platform for evaluating object detection in complex aquatic scenes. All datasets, pretrained models, and evaluation tools are publicly available to facilitate future research in underwater computer vision and intelligent marine systems.
>
---
#### [new 004] Real-Time Object Detection and Classification using YOLO for Edge FPGAs
- **分类: cs.CV; cs.AR**

- **简介: 该论文属于目标检测与分类任务，旨在解决现有YOLO模型在边缘FPGA设备上资源效率低的问题。作者提出了一种资源高效的YOLOv5模型，优化后部署在Xilinx Kria KV260 FPGA上，实现了高精度（99%）、低功耗（3.5W）和实时性（9 FPS）的目标检测与分类。**

- **链接: [http://arxiv.org/pdf/2507.18174v1](http://arxiv.org/pdf/2507.18174v1)**

> **作者:** Rashed Al Amin; Roman Obermaisser
>
> **备注:** This paper has been accepted for the 67th International Symposium on ELMAR 2025
>
> **摘要:** Object detection and classification are crucial tasks across various application domains, particularly in the development of safe and reliable Advanced Driver Assistance Systems (ADAS). Existing deep learning-based methods such as Convolutional Neural Networks (CNNs), Single Shot Detectors (SSDs), and You Only Look Once (YOLO) have demonstrated high performance in terms of accuracy and computational speed when deployed on Field-Programmable Gate Arrays (FPGAs). However, despite these advances, state-of-the-art YOLO-based object detection and classification systems continue to face challenges in achieving resource efficiency suitable for edge FPGA platforms. To address this limitation, this paper presents a resource-efficient real-time object detection and classification system based on YOLOv5 optimized for FPGA deployment. The proposed system is trained on the COCO and GTSRD datasets and implemented on the Xilinx Kria KV260 FPGA board. Experimental results demonstrate a classification accuracy of 99%, with a power consumption of 3.5W and a processing speed of 9 frames per second (FPS). These findings highlight the effectiveness of the proposed approach in enabling real-time, resource-efficient object detection and classification for edge computing applications.
>
---
#### [new 005] Degradation-Consistent Learning via Bidirectional Diffusion for Low-Light Image Enhancement
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于低光图像增强任务，旨在解决低光图像中结构不一致和像素错位的问题。通过提出双向扩散优化机制与自适应特征交互模块，联合建模低光与正常光图像的退化过程，实现更精确的退化参数匹配和高质量图像生成。**

- **链接: [http://arxiv.org/pdf/2507.18144v1](http://arxiv.org/pdf/2507.18144v1)**

> **作者:** Jinhong He; Minglong Xue; Zhipu Liu; Mingliang Zhou; Aoxiang Ning; Palaiahnakote Shivakumara
>
> **备注:** 10page
>
> **摘要:** Low-light image enhancement aims to improve the visibility of degraded images to better align with human visual perception. While diffusion-based methods have shown promising performance due to their strong generative capabilities. However, their unidirectional modelling of degradation often struggles to capture the complexity of real-world degradation patterns, leading to structural inconsistencies and pixel misalignments. To address these challenges, we propose a bidirectional diffusion optimization mechanism that jointly models the degradation processes of both low-light and normal-light images, enabling more precise degradation parameter matching and enhancing generation quality. Specifically, we perform bidirectional diffusion-from low-to-normal light and from normal-to-low light during training and introduce an adaptive feature interaction block (AFI) to refine feature representation. By leveraging the complementarity between these two paths, our approach imposes an implicit symmetry constraint on illumination attenuation and noise distribution, facilitating consistent degradation learning and improving the models ability to perceive illumination and detail degradation. Additionally, we design a reflection-aware correction module (RACM) to guide color restoration post-denoising and suppress overexposed regions, ensuring content consistency and generating high-quality images that align with human visual perception. Extensive experiments on multiple benchmark datasets demonstrate that our method outperforms state-of-the-art methods in both quantitative and qualitative evaluations while generalizing effectively to diverse degradation scenarios. Code at https://github.com/hejh8/BidDiff
>
---
#### [new 006] GaussianFusionOcc: A Seamless Sensor Fusion Approach for 3D Occupancy Prediction Using 3D Gaussians
- **分类: cs.CV**

- **简介: 论文属于自动驾驶中的3D语义占用预测任务，旨在解决多传感器融合下环境建模不准确的问题。作者提出GaussianFusionOcc，采用3D高斯表示和模态无关的可变形注意力机制，融合相机、激光雷达和雷达数据，提升预测精度与推理效率。**

- **链接: [http://arxiv.org/pdf/2507.18522v1](http://arxiv.org/pdf/2507.18522v1)**

> **作者:** Tomislav Pavković; Mohammad-Ali Nikouei Mahani; Johannes Niedermayer; Johannes Betz
>
> **摘要:** 3D semantic occupancy prediction is one of the crucial tasks of autonomous driving. It enables precise and safe interpretation and navigation in complex environments. Reliable predictions rely on effective sensor fusion, as different modalities can contain complementary information. Unlike conventional methods that depend on dense grid representations, our approach, GaussianFusionOcc, uses semantic 3D Gaussians alongside an innovative sensor fusion mechanism. Seamless integration of data from camera, LiDAR, and radar sensors enables more precise and scalable occupancy prediction, while 3D Gaussian representation significantly improves memory efficiency and inference speed. GaussianFusionOcc employs modality-agnostic deformable attention to extract essential features from each sensor type, which are then used to refine Gaussian properties, resulting in a more accurate representation of the environment. Extensive testing with various sensor combinations demonstrates the versatility of our approach. By leveraging the robustness of multi-modal fusion and the efficiency of Gaussian representation, GaussianFusionOcc outperforms current state-of-the-art models.
>
---
#### [new 007] MatSSL: Robust Self-Supervised Representation Learning for Metallographic Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务，旨在解决金属材料显微图像分割中依赖大量标注数据的问题。作者提出MatSSL，一种融合多级特征的自监督学习框架，通过在小规模无标签数据上预训练并微调，提升了分割性能，尤其在少量标注数据场景下优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.18184v1](http://arxiv.org/pdf/2507.18184v1)**

> **作者:** Hoang Hai Nam Nguyen; Phan Nguyen Duc Hieu; Ho Won Lee
>
> **摘要:** MatSSL is a streamlined self-supervised learning (SSL) architecture that employs Gated Feature Fusion at each stage of the backbone to integrate multi-level representations effectively. Current micrograph analysis of metallic materials relies on supervised methods, which require retraining for each new dataset and often perform inconsistently with only a few labeled samples. While SSL offers a promising alternative by leveraging unlabeled data, most existing methods still depend on large-scale datasets to be effective. MatSSL is designed to overcome this limitation. We first perform self-supervised pretraining on a small-scale, unlabeled dataset and then fine-tune the model on multiple benchmark datasets. The resulting segmentation models achieve 69.13% mIoU on MetalDAM, outperforming the 66.73% achieved by an ImageNet-pretrained encoder, and delivers consistently up to nearly 40% improvement in average mIoU on the Environmental Barrier Coating benchmark dataset (EBC) compared to models pretrained with MicroNet. This suggests that MatSSL enables effective adaptation to the metallographic domain using only a small amount of unlabeled data, while preserving the rich and transferable features learned from large-scale pretraining on natural images.
>
---
#### [new 008] Differential-UMamba: Rethinking Tumor Segmentation Under Limited Data Scenarios
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，旨在解决数据不足场景下模型易过拟合的问题。作者提出Diff-UMamba网络，结合UNet与Mamba机制，并引入噪声抑制模块，提升肿瘤分割精度与鲁棒性。实验表明其在多数据集上表现优于基线方法。**

- **链接: [http://arxiv.org/pdf/2507.18177v1](http://arxiv.org/pdf/2507.18177v1)**

> **作者:** Dhruv Jain; Romain Modzelewski; Romain Hérault; Clement Chatelain; Eva Torfeh; Sebastien Thureau
>
> **摘要:** In data-scarce scenarios, deep learning models often overfit to noise and irrelevant patterns, which limits their ability to generalize to unseen samples. To address these challenges in medical image segmentation, we introduce Diff-UMamba, a novel architecture that combines the UNet framework with the mamba mechanism for modeling long-range dependencies. At the heart of Diff-UMamba is a Noise Reduction Module (NRM), which employs a signal differencing strategy to suppress noisy or irrelevant activations within the encoder. This encourages the model to filter out spurious features and enhance task-relevant representations, thereby improving its focus on clinically meaningful regions. As a result, the architecture achieves improved segmentation accuracy and robustness, particularly in low-data settings. Diff-UMamba is evaluated on multiple public datasets, including MSD (lung and pancreas) and AIIB23, demonstrating consistent performance gains of 1-3% over baseline methods across diverse segmentation tasks. To further assess performance under limited-data conditions, additional experiments are conducted on the BraTS-21 dataset by varying the proportion of available training samples. The approach is also validated on a small internal non-small cell lung cancer (NSCLC) dataset for gross tumor volume (GTV) segmentation in cone beam CT (CBCT), where it achieves a 4-5% improvement over the baseline.
>
---
#### [new 009] GRR-CoCa: Leveraging LLM Mechanisms in Multimodal Model Architectures
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态模型任务，旨在解决当前多模态模型在架构上落后于大语言模型的问题。论文提出了GRR-CoCa模型，在文本解码器和视觉编码器中引入LLM中的组件如高斯激活、归一化和位置编码，提升了模型在预训练和微调任务上的表现。**

- **链接: [http://arxiv.org/pdf/2507.18009v1](http://arxiv.org/pdf/2507.18009v1)**

> **作者:** Jake R. Patock; Nicole Catherine Lewis; Kevin McCoy; Christina Gomez; Canling Chen; Lorenzo Luzi
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** State-of-the-art (SOTA) image and text generation models are multimodal models that have many similarities to large language models (LLMs). Despite achieving strong performances, leading foundational multimodal model architectures frequently lag behind the architectural sophistication of contemporary LLMs. We propose GRR-CoCa, an improved SOTA Contrastive Captioner (CoCa) model that incorporates Gaussian error gated linear units, root mean squared normalization, and rotary positional embedding into the textual decoders and the vision transformer (ViT) encoder. Each architectural modification has been shown to improve model performance in LLMs, but has yet to be adopted in CoCa. We benchmarked GRR-CoCa against Baseline CoCa, a model with the same modified textual decoders but with CoCa's original ViT encoder. We used standard pretraining and fine-tuning workflows to benchmark the models on contrastive and generative tasks. Our GRR-CoCa significantly outperformed Baseline CoCa on the pretraining dataset and three diverse fine-tuning datasets. Pretraining improvements were 27.25% in contrastive loss, 3.71% in perplexity, and 7.15% in CoCa loss. The average fine-tuning improvements were 13.66% in contrastive loss, 5.18% in perplexity, and 5.55% in CoCa loss. We show that GRR-CoCa's modified architecture improves performance and generalization across vision-language domains.
>
---
#### [new 010] Towards Facilitated Fairness Assessment of AI-based Skin Lesion Classifiers Through GenAI-based Image Synthesis
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文旨在通过生成式AI合成图像，提升皮肤病变分类模型的公平性评估。任务是解决因数据偏差导致的公平性评估不足问题。工作内容为利用LightningDiT模型生成合成数据，用于评估现有黑色素瘤分类模型的公平性，并探讨合成数据在公平性评估中的潜力与局限性。**

- **链接: [http://arxiv.org/pdf/2507.17860v1](http://arxiv.org/pdf/2507.17860v1)**

> **作者:** Ko Watanabe. Stanislav Frolov. Adriano Lucieri. Andreas Dengel
>
> **摘要:** Recent advancements in Deep Learning and its application on the edge hold great potential for the revolution of routine screenings for skin cancers like Melanoma. Along with the anticipated benefits of this technology, potential dangers arise from unforseen and inherent biases. Thus, assessing and improving the fairness of such systems is of utmost importance. A key challenge in fairness assessment is to ensure that the evaluation dataset is sufficiently representative of different Personal Identifiable Information (PII) (sex, age, and race) and other minority groups. Against the backdrop of this challenge, this study leverages the state-of-the-art Generative AI (GenAI) LightningDiT model to assess the fairness of publicly available melanoma classifiers. The results suggest that fairness assessment using highly realistic synthetic data is a promising direction. Yet, our findings indicate that verifying fairness becomes difficult when the melanoma-detection model used for evaluation is trained on data that differ from the dataset underpinning the synthetic images. Nonetheless, we propose that our approach offers a valuable new avenue for employing synthetic data to gauge and enhance fairness in medical-imaging GenAI systems.
>
---
#### [new 011] LONG3R: Long Sequence Streaming 3D Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D场景重建任务，旨在解决现有方法处理长序列图像流效率低、无法实时推理的问题。作者提出LONG3R模型，通过记忆门控机制和3D时空记忆实现长时间序列的实时多视角重建，并采用两阶段训练策略提升性能。实验表明其在长序列上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.18255v1](http://arxiv.org/pdf/2507.18255v1)**

> **作者:** Zhuoguang Chen; Minghui Qin; Tianyuan Yuan; Zhe Liu; Hang Zhao
>
> **备注:** Accepted by ICCV 2025. Project page: https://zgchen33.github.io/LONG3R/
>
> **摘要:** Recent advancements in multi-view scene reconstruction have been significant, yet existing methods face limitations when processing streams of input images. These methods either rely on time-consuming offline optimization or are restricted to shorter sequences, hindering their applicability in real-time scenarios. In this work, we propose LONG3R (LOng sequence streaming 3D Reconstruction), a novel model designed for streaming multi-view 3D scene reconstruction over longer sequences. Our model achieves real-time processing by operating recurrently, maintaining and updating memory with each new observation. We first employ a memory gating mechanism to filter relevant memory, which, together with a new observation, is fed into a dual-source refined decoder for coarse-to-fine interaction. To effectively capture long-sequence memory, we propose a 3D spatio-temporal memory that dynamically prunes redundant spatial information while adaptively adjusting resolution along the scene. To enhance our model's performance on long sequences while maintaining training efficiency, we employ a two-stage curriculum training strategy, each stage targeting specific capabilities. Experiments demonstrate that LONG3R outperforms state-of-the-art streaming methods, particularly for longer sequences, while maintaining real-time inference speed. Project page: https://zgchen33.github.io/LONG3R/.
>
---
#### [new 012] Boosting Multi-View Indoor 3D Object Detection via Adaptive 3D Volume Construction
- **分类: cs.CV**

- **简介: 该论文属于室内3D目标检测任务，旨在解决多视角3D检测中特征表示与计算效率问题。作者提出SGCDet框架，通过自适应3D体素构建、几何与上下文感知聚合模块及稀疏体素优化策略，实现高效检测。方法仅需3D框标注，无需场景几何真值，提升了性能与实用性。**

- **链接: [http://arxiv.org/pdf/2507.18331v1](http://arxiv.org/pdf/2507.18331v1)**

> **作者:** Runmin Zhang; Zhu Yu; Si-Yuan Cao; Lingyu Zhu; Guangyi Zhang; Xiaokai Bai; Hui-Liang Shen
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** This work presents SGCDet, a novel multi-view indoor 3D object detection framework based on adaptive 3D volume construction. Unlike previous approaches that restrict the receptive field of voxels to fixed locations on images, we introduce a geometry and context aware aggregation module to integrate geometric and contextual information within adaptive regions in each image and dynamically adjust the contributions from different views, enhancing the representation capability of voxel features. Furthermore, we propose a sparse volume construction strategy that adaptively identifies and selects voxels with high occupancy probabilities for feature refinement, minimizing redundant computation in free space. Benefiting from the above designs, our framework achieves effective and efficient volume construction in an adaptive way. Better still, our network can be supervised using only 3D bounding boxes, eliminating the dependence on ground-truth scene geometry. Experimental results demonstrate that SGCDet achieves state-of-the-art performance on the ScanNet, ScanNet200 and ARKitScenes datasets. The source code is available at https://github.com/RM-Zhang/SGCDet.
>
---
#### [new 013] Explaining How Visual, Textual and Multimodal Encoders Share Concepts
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态表示学习任务，旨在解决如何量化比较视觉、文本和多模态编码器之间的特征共享问题。作者提出了一种新的指标和“特征共享度”方法，对21个不同类型的编码器进行研究，揭示了多模态训练对特征共享的影响，并指出视觉语言模型的视觉特征与文本编码器共享，反映文本预训练的重要性。**

- **链接: [http://arxiv.org/pdf/2507.18512v1](http://arxiv.org/pdf/2507.18512v1)**

> **作者:** Clément Cornet; Romaric Besançon; Hervé Le Borgne
>
> **摘要:** Sparse autoencoders (SAEs) have emerged as a powerful technique for extracting human-interpretable features from neural networks activations. Previous works compared different models based on SAE-derived features but those comparisons have been restricted to models within the same modality. We propose a novel indicator allowing quantitative comparison of models across SAE features, and use it to conduct a comparative study of visual, textual and multimodal encoders. We also propose to quantify the Comparative Sharedness of individual features between different classes of models. With these two new tools, we conduct several studies on 21 encoders of the three types, with two significantly different sizes, and considering generalist and domain specific datasets. The results allow to revisit previous studies at the light of encoders trained in a multimodal context and to quantify to which extent all these models share some representations or features. They also suggest that visual features that are specific to VLMs among vision encoders are shared with text encoders, highlighting the impact of text pretraining. The code is available at https://github.com/CEA-LIST/SAEshareConcepts
>
---
#### [new 014] Lumina-mGPT 2.0: Stand-Alone AutoRegressive Image Modeling
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决现有扩散模型依赖预训练组件或混合架构的问题。论文提出Lumina-mGPT 2.0，一种从零训练的自回归图像生成模型，支持多种任务（如编辑、可控合成等），并提升生成质量与解码效率，实现与扩散模型相当甚至更优的性能。**

- **链接: [http://arxiv.org/pdf/2507.17801v1](http://arxiv.org/pdf/2507.17801v1)**

> **作者:** Yi Xin; Juncheng Yan; Qi Qin; Zhen Li; Dongyang Liu; Shicheng Li; Victor Shea-Jay Huang; Yupeng Zhou; Renrui Zhang; Le Zhuo; Tiancheng Han; Xiaoqing Sun; Siqi Luo; Mengmeng Wang; Bin Fu; Yuewen Cao; Hongsheng Li; Guangtao Zhai; Xiaohong Liu; Yu Qiao; Peng Gao
>
> **备注:** Tech Report, 23 pages, 11 figures, 7 tables
>
> **摘要:** We present Lumina-mGPT 2.0, a stand-alone, decoder-only autoregressive model that revisits and revitalizes the autoregressive paradigm for high-quality image generation and beyond. Unlike existing approaches that rely on pretrained components or hybrid architectures, Lumina-mGPT 2.0 is trained entirely from scratch, enabling unrestricted architectural design and licensing freedom. It achieves generation quality on par with state-of-the-art diffusion models such as DALL-E 3 and SANA, while preserving the inherent flexibility and compositionality of autoregressive modeling. Our unified tokenization scheme allows the model to seamlessly handle a wide spectrum of tasks-including subject-driven generation, image editing, controllable synthesis, and dense prediction-within a single generative framework. To further boost usability, we incorporate efficient decoding strategies like inference-time scaling and speculative Jacobi sampling to improve quality and speed, respectively. Extensive evaluations on standard text-to-image benchmarks (e.g., GenEval, DPG) demonstrate that Lumina-mGPT 2.0 not only matches but in some cases surpasses diffusion-based models. Moreover, we confirm its multi-task capabilities on the Graph200K benchmark, with the native Lumina-mGPT 2.0 performing exceptionally well. These results position Lumina-mGPT 2.0 as a strong, flexible foundation model for unified multimodal generation. We have released our training details, code, and models at https://github.com/Alpha-VLLM/Lumina-mGPT-2.0.
>
---
#### [new 015] A Multi-Dataset Benchmark for Semi-Supervised Semantic Segmentation in ECG Delineation
- **分类: cs.CV; cs.AI; cs.LG; eess.SP**

- **简介: 该论文属于医学信号处理与深度学习交叉任务，旨在解决心电图（ECG）波形自动分割中标注数据不足的问题。作者构建了首个用于半监督语义分割的ECG基准数据集，整合多个公开数据源，采用五种半监督算法，在卷积网络与Transformer架构上进行实验，并提出ECG专用训练策略与评估框架。结果表明Transformer表现更优，为后续研究提供了标准参考。**

- **链接: [http://arxiv.org/pdf/2507.18323v1](http://arxiv.org/pdf/2507.18323v1)**

> **作者:** Minje Park; Jeonghwa Lim; Taehyung Yu; Sunghoon Joo
>
> **备注:** 6 pages, 2 figures
>
> **摘要:** Electrocardiogram (ECG) delineation, the segmentation of meaningful waveform features, is critical for clinical diagnosis. Despite recent advances using deep learning, progress has been limited by the scarcity of publicly available annotated datasets. Semi-supervised learning presents a promising solution by leveraging abundant unlabeled ECG data. In this study, we present the first systematic benchmark for semi-supervised semantic segmentation (SemiSeg) in ECG delineation. We curated and unified multiple public datasets, including previously underused sources, to support robust and diverse evaluation. We adopted five representative SemiSeg algorithms from computer vision, implemented them on two different architectures: the convolutional network and the transformer, and evaluated them in two different settings: in-domain and cross-domain. Additionally, we propose ECG-specific training configurations and augmentation strategies and introduce a standardized evaluation framework. Our results show that the transformer outperforms the convolutional network in semi-supervised ECG delineation. We anticipate that our benchmark will serve as a foundation for advancing semi-supervised ECG delineation methods and will facilitate further research in this domain.
>
---
#### [new 016] Exploiting Gaussian Agnostic Representation Learning with Diffusion Priors for Enhanced Infrared Small Target Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于红外小目标检测任务，旨在解决实际场景中因缺乏高质量标注数据导致检测性能下降的问题。作者提出高斯无关表示学习方法，并结合扩散模型提升合成样本质量，从而增强检测模型在数据稀缺场景下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.18260v1](http://arxiv.org/pdf/2507.18260v1)**

> **作者:** Junyao Li; Yahao Lu; Xingyuan Guo; Xiaoyu Xian; Tiantian Wang; Yukai Shi
>
> **备注:** Submitted to Neural Networks. We propose the Gaussian Group Squeezer, leveraging Gaussian sampling and compression with diffusion models for channel-based data augmentation
>
> **摘要:** Infrared small target detection (ISTD) plays a vital role in numerous practical applications. In pursuit of determining the performance boundaries, researchers employ large and expensive manual-labeling data for representation learning. Nevertheless, this approach renders the state-of-the-art ISTD methods highly fragile in real-world challenges. In this paper, we first study the variation in detection performance across several mainstream methods under various scarcity -- namely, the absence of high-quality infrared data -- that challenge the prevailing theories about practical ISTD. To address this concern, we introduce the Gaussian Agnostic Representation Learning. Specifically, we propose the Gaussian Group Squeezer, leveraging Gaussian sampling and compression for non-uniform quantization. By exploiting a diverse array of training samples, we enhance the resilience of ISTD models against various challenges. Then, we introduce two-stage diffusion models for real-world reconstruction. By aligning quantized signals closely with real-world distributions, we significantly elevate the quality and fidelity of the synthetic samples. Comparative evaluations against state-of-the-art detection methods in various scarcity scenarios demonstrate the efficacy of the proposed approach.
>
---
#### [new 017] LEAF: Latent Diffusion with Efficient Encoder Distillation for Aligned Features in Medical Image Segmentation
- **分类: cs.CV**

- **简介: 论文提出LEAF模型，用于医学图像分割任务，旨在解决现有扩散模型在特征提取和分割结果稳定性上的不足。通过直接预测分割图和引入特征蒸馏方法，提升分割性能，且不增加推理复杂度。**

- **链接: [http://arxiv.org/pdf/2507.18214v1](http://arxiv.org/pdf/2507.18214v1)**

> **作者:** Qilin Huang; Tianyu Lin; Zhiguang Chen; Fudan Zheng
>
> **备注:** Accepted at MICCAI 2025
>
> **摘要:** Leveraging the powerful capabilities of diffusion models has yielded quite effective results in medical image segmentation tasks. However, existing methods typically transfer the original training process directly without specific adjustments for segmentation tasks. Furthermore, the commonly used pre-trained diffusion models still have deficiencies in feature extraction. Based on these considerations, we propose LEAF, a medical image segmentation model grounded in latent diffusion models. During the fine-tuning process, we replace the original noise prediction pattern with a direct prediction of the segmentation map, thereby reducing the variance of segmentation results. We also employ a feature distillation method to align the hidden states of the convolutional layers with the features from a transformer-based vision encoder. Experimental results demonstrate that our method enhances the performance of the original diffusion model across multiple segmentation datasets for different disease types. Notably, our approach does not alter the model architecture, nor does it increase the number of parameters or computation during the inference phase, making it highly efficient.
>
---
#### [new 018] VideoMind: An Omni-Modal Video Dataset with Intent Grounding for Deep-Cognitive Video Understanding
- **分类: cs.CV; cs.AI; 68T45, 68T50, 68U35,; I.4.8; I.2.7; I.2.10; H.5.1**

- **简介: 该论文属于多模态视频理解任务，旨在解决深度认知视频理解问题。论文构建了VideoMind数据集，包含103K视频样本及其音频和文本描述，提供意图表达与多层次标注，支持细粒度跨模态对齐与认知理解研究。**

- **链接: [http://arxiv.org/pdf/2507.18552v1](http://arxiv.org/pdf/2507.18552v1)**

> **作者:** Baoyao Yang; Wanyun Li; Dixin Chen; Junxiang Chen; Wenbin Yao; Haifeng Lin
>
> **备注:** 7 pages; 14 figures
>
> **摘要:** This paper introduces VideoMind, a video-centric omni-modal dataset designed for deep video content cognition and enhanced multi-modal feature representation. The dataset comprises 103K video samples (3K reserved for testing), each paired with audio and systematically detailed textual descriptions. Specifically, every video and its audio is described across three hierarchical layers (factual, abstract, and intent), progressing from surface to depth. It contains over 22 million words, averaging ~225 words per sample. VideoMind's key distinction from existing datasets is its provision of intent expressions, which require contextual integration across the entire video and are not directly observable. These deep-cognitive expressions are generated using a Chain-of-Thought (COT) approach, prompting the mLLM through step-by-step reasoning. Each description includes annotations for subject, place, time, event, action, and intent, supporting downstream recognition tasks. Crucially, we establish a gold-standard benchmark with 3,000 manually validated samples for evaluating deep-cognitive video understanding. We design hybrid-cognitive retrieval experiments, scored by multi-level retrieval metrics, to appropriately assess deep video comprehension. Evaluation results for models (e.g., InternVideo, VAST, UMT-L) are released. VideoMind serves as a powerful benchmark for fine-grained cross-modal alignment and advances fields requiring in-depth video understanding, such as emotion and intent recognition. The data is publicly available on GitHub, HuggingFace, and OpenDataLab, https://github.com/cdx-cindy/VideoMind.
>
---
#### [new 019] Human Scanpath Prediction in Target-Present Visual Search with Semantic-Foveal Bayesian Attention
- **分类: cs.CV**

- **简介: 该论文属于视觉注意力建模任务，旨在解决目标存在下的视觉搜索中人类扫描路径预测问题。作者提出了SemBA-FAST模型，结合语义与中央凹机制，动态生成注意力图，以提升注视点预测准确性，并在COCO-Search18数据集上验证其性能。**

- **链接: [http://arxiv.org/pdf/2507.18503v1](http://arxiv.org/pdf/2507.18503v1)**

> **作者:** João Luzio; Alexandre Bernardino; Plinio Moreno
>
> **备注:** To be published in the 2025 IEEE International Conference on Development and Learning (ICDL)
>
> **摘要:** In goal-directed visual tasks, human perception is guided by both top-down and bottom-up cues. At the same time, foveal vision plays a crucial role in directing attention efficiently. Modern research on bio-inspired computational attention models has taken advantage of advancements in deep learning by utilizing human scanpath data to achieve new state-of-the-art performance. In this work, we assess the performance of SemBA-FAST, i.e. Semantic-based Bayesian Attention for Foveal Active visual Search Tasks, a top-down framework designed for predicting human visual attention in target-present visual search. SemBA-FAST integrates deep object detection with a probabilistic semantic fusion mechanism to generate attention maps dynamically, leveraging pre-trained detectors and artificial foveation to update top-down knowledge and improve fixation prediction sequentially. We evaluate SemBA-FAST on the COCO-Search18 benchmark dataset, comparing its performance against other scanpath prediction models. Our methodology achieves fixation sequences that closely match human ground-truth scanpaths. Notably, it surpasses baseline and other top-down approaches and competes, in some cases, with scanpath-informed models. These findings provide valuable insights into the capabilities of semantic-foveal probabilistic frameworks for human-like attention modelling, with implications for real-time cognitive computing and robotics.
>
---
#### [new 020] ViGText: Deepfake Image Detection with Vision-Language Model Explanations and Graph Neural Networks
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像检测任务，旨在解决深度伪造图像的检测难题。针对现有方法在复杂场景下的泛化性和鲁棒性不足，论文提出ViGText模型，结合视觉语言模型解释与图神经网络，对图像及其文本解释进行多级特征分析，提升检测准确性与抗攻击能力。**

- **链接: [http://arxiv.org/pdf/2507.18031v1](http://arxiv.org/pdf/2507.18031v1)**

> **作者:** Ahmad ALBarqawi; Mahmoud Nazzal; Issa Khalil; Abdallah Khreishah; NhatHai Phan
>
> **摘要:** The rapid rise of deepfake technology, which produces realistic but fraudulent digital content, threatens the authenticity of media. Traditional deepfake detection approaches often struggle with sophisticated, customized deepfakes, especially in terms of generalization and robustness against malicious attacks. This paper introduces ViGText, a novel approach that integrates images with Vision Large Language Model (VLLM) Text explanations within a Graph-based framework to improve deepfake detection. The novelty of ViGText lies in its integration of detailed explanations with visual data, as it provides a more context-aware analysis than captions, which often lack specificity and fail to reveal subtle inconsistencies. ViGText systematically divides images into patches, constructs image and text graphs, and integrates them for analysis using Graph Neural Networks (GNNs) to identify deepfakes. Through the use of multi-level feature extraction across spatial and frequency domains, ViGText captures details that enhance its robustness and accuracy to detect sophisticated deepfakes. Extensive experiments demonstrate that ViGText significantly enhances generalization and achieves a notable performance boost when it detects user-customized deepfakes. Specifically, average F1 scores rise from 72.45% to 98.32% under generalization evaluation, and reflects the model's superior ability to generalize to unseen, fine-tuned variations of stable diffusion models. As for robustness, ViGText achieves an increase of 11.1% in recall compared to other deepfake detection approaches. When facing targeted attacks that exploit its graph-based architecture, ViGText limits classification performance degradation to less than 4%. ViGText uses detailed visual and textual analysis to set a new standard for detecting deepfakes, helping ensure media authenticity and information integrity.
>
---
#### [new 021] Bearded Dragon Activity Recognition Pipeline: An AI-Based Approach to Behavioural Monitoring
- **分类: cs.CV**

- **简介: 论文提出了一种基于AI的系统，用于自动识别鬃狮蜥行为。该系统使用YOLO模型进行实时视频分析，旨在解决传统人工监测效率低且易出错的问题。研究团队训练多个YOLO模型，识别“晒太阳”和“捕猎”行为，并选择YOLOv8s作为最优模型。系统在晒太阳识别上效果良好，但捕猎识别因蟋蟀检测精度低而受限。未来将改进小物体检测以提升性能。**

- **链接: [http://arxiv.org/pdf/2507.17987v1](http://arxiv.org/pdf/2507.17987v1)**

> **作者:** Arsen Yermukan; Pedro Machado; Feliciano Domingos; Isibor Kennedy Ihianle; Jordan J. Bird; Stefano S. K. Kaburu; Samantha J. Ward
>
> **摘要:** Traditional monitoring of bearded dragon (Pogona Viticeps) behaviour is time-consuming and prone to errors. This project introduces an automated system for real-time video analysis, using You Only Look Once (YOLO) object detection models to identify two key behaviours: basking and hunting. We trained five YOLO variants (v5, v7, v8, v11, v12) on a custom, publicly available dataset of 1200 images, encompassing bearded dragons (600), heating lamps (500), and crickets (100). YOLOv8s was selected as the optimal model due to its superior balance of accuracy (mAP@0.5:0.95 = 0.855) and speed. The system processes video footage by extracting per-frame object coordinates, applying temporal interpolation for continuity, and using rule-based logic to classify specific behaviours. Basking detection proved reliable. However, hunting detection was less accurate, primarily due to weak cricket detection (mAP@0.5 = 0.392). Future improvements will focus on enhancing cricket detection through expanded datasets or specialised small-object detectors. This automated system offers a scalable solution for monitoring reptile behaviour in controlled environments, significantly improving research efficiency and data quality.
>
---
#### [new 022] LMM-Det: Make Large Multimodal Models Excel in Object Detection
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决大型多模态模型（LMM）在目标检测中性能不足的问题。作者提出了LMM-Det方法，通过调整数据分布和优化推理过程，提升LMM在不依赖专用检测模块情况下的检测能力，验证了LMM本身具备目标检测潜力。**

- **链接: [http://arxiv.org/pdf/2507.18300v1](http://arxiv.org/pdf/2507.18300v1)**

> **作者:** Jincheng Li; Chunyu Xie; Ji Ao; Dawei Leng; Yuhui Yin
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Large multimodal models (LMMs) have garnered wide-spread attention and interest within the artificial intelligence research and industrial communities, owing to their remarkable capability in multimodal understanding, reasoning, and in-context learning, among others. While LMMs have demonstrated promising results in tackling multimodal tasks like image captioning, visual question answering, and visual grounding, the object detection capabilities of LMMs exhibit a significant gap compared to specialist detectors. To bridge the gap, we depart from the conventional methods of integrating heavy detectors with LMMs and propose LMM-Det, a simple yet effective approach that leverages a Large Multimodal Model for vanilla object Detection without relying on specialized detection modules. Specifically, we conduct a comprehensive exploratory analysis when a large multimodal model meets with object detection, revealing that the recall rate degrades significantly compared with specialist detection models. To mitigate this, we propose to increase the recall rate by introducing data distribution adjustment and inference optimization tailored for object detection. We re-organize the instruction conversations to enhance the object detection capabilities of large multimodal models. We claim that a large multimodal model possesses detection capability without any extra detection modules. Extensive experiments support our claim and show the effectiveness of the versatile LMM-Det. The datasets, models, and codes are available at https://github.com/360CVGroup/LMM-Det.
>
---
#### [new 023] HumanMaterial: Human Material Estimation from a Single Image via Progressive Training
- **分类: cs.CV**

- **简介: 该论文属于人体逆向渲染任务，旨在从单张图像估计高精度人体材质。为解决材质估计中因数据和训练方法限制导致的渲染真实性不足问题，作者构建了高质量数据集OpenHumanBRDF，并提出HumanMaterial模型，采用渐进式训练策略优化材质预测，提升了渲染效果，尤其在皮肤细节上效果显著。**

- **链接: [http://arxiv.org/pdf/2507.18385v1](http://arxiv.org/pdf/2507.18385v1)**

> **作者:** Yu Jiang; Jiahao Xia; Jiongming Qin; Yusen Wang; Tuo Cao; Chunxia Xiao
>
> **备注:** 14
>
> **摘要:** Full-body Human inverse rendering based on physically-based rendering aims to acquire high-quality materials, which helps achieve photo-realistic rendering under arbitrary illuminations. This task requires estimating multiple material maps and usually relies on the constraint of rendering result. The absence of constraints on the material maps makes inverse rendering an ill-posed task. Previous works alleviated this problem by building material dataset for training, but their simplified material data and rendering equation lead to rendering results with limited realism, especially that of skin. To further alleviate this problem, we construct a higher-quality dataset (OpenHumanBRDF) based on scanned real data and statistical material data. In addition to the normal, diffuse albedo, roughness, specular albedo, we produce displacement and subsurface scattering to enhance the realism of rendering results, especially for the skin. With the increase in prediction tasks for more materials, using an end-to-end model as in the previous work struggles to balance the importance among various material maps, and leads to model underfitting. Therefore, we design a model (HumanMaterial) with progressive training strategy to make full use of the supervision information of the material maps and improve the performance of material estimation. HumanMaterial first obtain the initial material results via three prior models, and then refine the results by a finetuning model. Prior models estimate different material maps, and each map has different significance for rendering results. Thus, we design a Controlled PBR Rendering (CPR) loss, which enhances the importance of the materials to be optimized during the training of prior models. Extensive experiments on OpenHumanBRDF dataset and real data demonstrate that our method achieves state-of-the-art performance.
>
---
#### [new 024] High-fidelity 3D Gaussian Inpainting: preserving multi-view consistency and photorealistic details
- **分类: cs.CV**

- **简介: 该论文属于3D场景修复任务，旨在解决3D高斯散射表示中多视角不一致与细节缺失的问题。作者提出了一种新的3D高斯修复框架，结合自动掩码优化与不确定性引导的区域优化策略，提升了修复质量与多视角一致性。**

- **链接: [http://arxiv.org/pdf/2507.18023v1](http://arxiv.org/pdf/2507.18023v1)**

> **作者:** Jun Zhou; Dinghao Li; Nannan Li; Mingjie Wang
>
> **摘要:** Recent advancements in multi-view 3D reconstruction and novel-view synthesis, particularly through Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), have greatly enhanced the fidelity and efficiency of 3D content creation. However, inpainting 3D scenes remains a challenging task due to the inherent irregularity of 3D structures and the critical need for maintaining multi-view consistency. In this work, we propose a novel 3D Gaussian inpainting framework that reconstructs complete 3D scenes by leveraging sparse inpainted views. Our framework incorporates an automatic Mask Refinement Process and region-wise Uncertainty-guided Optimization. Specifically, we refine the inpainting mask using a series of operations, including Gaussian scene filtering and back-projection, enabling more accurate localization of occluded regions and realistic boundary restoration. Furthermore, our Uncertainty-guided Fine-grained Optimization strategy, which estimates the importance of each region across multi-view images during training, alleviates multi-view inconsistencies and enhances the fidelity of fine details in the inpainted results. Comprehensive experiments conducted on diverse datasets demonstrate that our approach outperforms existing state-of-the-art methods in both visual quality and view consistency.
>
---
#### [new 025] DCFFSNet: Deep Connectivity Feature Fusion Separation Network for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决现有模型中拓扑连通性特征与其他特征耦合、缺乏量化机制导致的边缘精度和区域一致性不足问题。作者提出DCFFSNet，通过解耦特征空间、量化特征强度，并构建多尺度特征融合-分离架构，有效提升分割性能。**

- **链接: [http://arxiv.org/pdf/2507.18407v1](http://arxiv.org/pdf/2507.18407v1)**

> **作者:** Xun Ye; Ruixiang Tang; Mingda Zhang; Jianglong Qin
>
> **备注:** 16 pages , 11 figures
>
> **摘要:** Medical image segmentation leverages topological connectivity theory to enhance edge precision and regional consistency. However, existing deep networks integrating connectivity often forcibly inject it as an additional feature module, resulting in coupled feature spaces with no standardized mechanism to quantify different feature strengths. To address these issues, we propose DCFFSNet (Dual-Connectivity Feature Fusion-Separation Network). It introduces an innovative feature space decoupling strategy. This strategy quantifies the relative strength between connectivity features and other features. It then builds a deep connectivity feature fusion-separation architecture. This architecture dynamically balances multi-scale feature expression. Experiments were conducted on the ISIC2018, DSB2018, and MoNuSeg datasets. On ISIC2018, DCFFSNet outperformed the next best model (CMUNet) by 1.3% (Dice) and 1.2% (IoU). On DSB2018, it surpassed TransUNet by 0.7% (Dice) and 0.9% (IoU). On MoNuSeg, it exceeded CSCAUNet by 0.8% (Dice) and 0.9% (IoU). The results demonstrate that DCFFSNet exceeds existing mainstream methods across all metrics. It effectively resolves segmentation fragmentation and achieves smooth edge transitions. This significantly enhances clinical usability.
>
---
#### [new 026] Facial Demorphing from a Single Morph Using a Latent Conditional GAN
- **分类: cs.CV**

- **简介: 该论文属于图像生成与生物特征安全任务，旨在解决“面部变形”图像的还原问题。现有方法存在复现变形图像或依赖特定生成技术的问题，为此，论文提出一种基于潜在条件生成对抗网络的方法，可在未知面部风格和变形技术下，从单一变形图像中还原出高保真原图。**

- **链接: [http://arxiv.org/pdf/2507.18566v1](http://arxiv.org/pdf/2507.18566v1)**

> **作者:** Nitish Shukla; Arun Ross
>
> **摘要:** A morph is created by combining two (or more) face images from two (or more) identities to create a composite image that is highly similar to both constituent identities, allowing the forged morph to be biometrically associated with more than one individual. Morph Attack Detection (MAD) can be used to detect a morph, but does not reveal the constituent images. Demorphing - the process of deducing the constituent images - is thus vital to provide additional evidence about a morph. Existing demorphing methods suffer from the morph replication problem, where the outputs tend to look very similar to the morph itself, or assume that train and test morphs are generated using the same morph technique. The proposed method overcomes these issues. The method decomposes a morph in latent space allowing it to demorph images created from unseen morph techniques and face styles. We train our method on morphs created from synthetic faces and test on morphs created from real faces using arbitrary morph techniques. Our method outperforms existing methods by a considerable margin and produces high fidelity demorphed face images.
>
---
#### [new 027] Revisiting Physically Realizable Adversarial Object Attack against LiDAR-based Detection: Clarifying Problem Formulation and Experimental Protocols
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自动驾驶安全任务，旨在解决LiDAR目标检测中的物理对抗攻击问题。现有攻击方法缺乏实际可行性与可重复性。论文提出了一种通用、标准化的物理对抗攻击框架，支持多种攻击方法，并提供开源代码与实验协议，实现了仿真与真实环境中的攻击迁移，推动对抗鲁棒性研究。**

- **链接: [http://arxiv.org/pdf/2507.18457v1](http://arxiv.org/pdf/2507.18457v1)**

> **作者:** Luo Cheng; Hanwei Zhang; Lijun Zhang; Holger Hermanns
>
> **摘要:** Adversarial robustness in LiDAR-based 3D object detection is a critical research area due to its widespread application in real-world scenarios. While many digital attacks manipulate point clouds or meshes, they often lack physical realizability, limiting their practical impact. Physical adversarial object attacks remain underexplored and suffer from poor reproducibility due to inconsistent setups and hardware differences. To address this, we propose a device-agnostic, standardized framework that abstracts key elements of physical adversarial object attacks, supports diverse methods, and provides open-source code with benchmarking protocols in simulation and real-world settings. Our framework enables fair comparison, accelerates research, and is validated by successfully transferring simulated attacks to a physical LiDAR system. Beyond the framework, we offer insights into factors influencing attack success and advance understanding of adversarial robustness in real-world LiDAR perception.
>
---
#### [new 028] 3D Software Synthesis Guided by Constraint-Expressive Intermediate Representation
- **分类: cs.CV; cs.AI; cs.MM; cs.SE**

- **简介: 该论文属于3D软件合成任务，旨在解决现有方法难以精细控制3D元素及处理复杂约束的问题。作者提出Scenethesis及其语言ScenethesisLang，通过约束感知的中间表示实现需求敏感的3D软件生成，支持细粒度修改与复杂约束满足。**

- **链接: [http://arxiv.org/pdf/2507.18625v1](http://arxiv.org/pdf/2507.18625v1)**

> **作者:** Shuqing Li; Anson Y. Lam; Yun Peng; Wenxuan Wang; Michael R. Lyu
>
> **摘要:** Graphical user interface (UI) software has undergone a fundamental transformation from traditional two-dimensional (2D) desktop/web/mobile interfaces to spatial three-dimensional (3D) environments. While existing work has made remarkable success in automated 2D software generation, such as HTML/CSS and mobile app interface code synthesis, the generation of 3D software still remains under-explored. Current methods for 3D software generation usually generate the 3D environments as a whole and cannot modify or control specific elements in the software. Furthermore, these methods struggle to handle the complex spatial and semantic constraints inherent in the real world. To address the challenges, we present Scenethesis, a novel requirement-sensitive 3D software synthesis approach that maintains formal traceability between user specifications and generated 3D software. Scenethesis is built upon ScenethesisLang, a domain-specific language that serves as a granular constraint-aware intermediate representation (IR) to bridge natural language requirements and executable 3D software. It serves both as a comprehensive scene description language enabling fine-grained modification of 3D software elements and as a formal constraint-expressive specification language capable of expressing complex spatial constraints. By decomposing 3D software synthesis into stages operating on ScenethesisLang, Scenethesis enables independent verification, targeted modification, and systematic constraint satisfaction. Our evaluation demonstrates that Scenethesis accurately captures over 80% of user requirements and satisfies more than 90% of hard constraints while handling over 100 constraints simultaneously. Furthermore, Scenethesis achieves a 42.8% improvement in BLIP-2 visual evaluation scores compared to the state-of-the-art method.
>
---
#### [new 029] Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像恢复任务，旨在解决传统扩散模型因固定高斯噪声导致的图像退化问题。作者提出EDA框架，扩展了噪声模式的设计空间，支持任意噪声类型，同时保持模型灵活性与计算效率。实验验证了EDA在多种图像恢复任务中的优越性能。**

- **链接: [http://arxiv.org/pdf/2507.18534v1](http://arxiv.org/pdf/2507.18534v1)**

> **作者:** Xingyu Qiu; Mengying Yang; Xinghua Ma; Dong Liang; Yuzhen Li; Fanding Li; Gongning Luo; Wei Wang; Kuanquan Wang; Shuo Li
>
> **备注:** 21 pages, 4 figures
>
> **摘要:** EDM elucidates the unified design space of diffusion models, yet its fixed noise patterns restricted to pure Gaussian noise, limit advancements in image restoration. Our study indicates that forcibly injecting Gaussian noise corrupts the degraded images, overextends the image transformation distance, and increases restoration complexity. To address this problem, our proposed EDA Elucidates the Design space of Arbitrary-noise-based diffusion models. Theoretically, EDA expands the freedom of noise pattern while preserving the original module flexibility of EDM, with rigorous proof that increased noise complexity incurs no additional computational overhead during restoration. EDA is validated on three typical tasks: MRI bias field correction (global smooth noise), CT metal artifact reduction (global sharp noise), and natural image shadow removal (local boundary-aware noise). With only 5 sampling steps, EDA outperforms most task-specific methods and achieves state-of-the-art performance in bias field correction and shadow removal.
>
---
#### [new 030] NLML-HPE: Head Pose Estimation with Limited Data via Manifold Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于计算机视觉中的头部姿态估计（HPE）任务，旨在解决训练数据有限且标注不准确的问题。作者提出NLML-HPE方法，结合张量分解与神经网络，将HPE视为回归问题，通过非线性流形学习提升估计精度与速度，并生成精确的2D数据集用于训练。**

- **链接: [http://arxiv.org/pdf/2507.18429v1](http://arxiv.org/pdf/2507.18429v1)**

> **作者:** Mahdi Ghafourian; Federico M. Sukno
>
> **摘要:** Head pose estimation (HPE) plays a critical role in various computer vision applications such as human-computer interaction and facial recognition. In this paper, we propose a novel deep learning approach for head pose estimation with limited training data via non-linear manifold learning called NLML-HPE. This method is based on the combination of tensor decomposition (i.e., Tucker decomposition) and feed forward neural networks. Unlike traditional classification-based approaches, our method formulates head pose estimation as a regression problem, mapping input landmarks into a continuous representation of pose angles. To this end, our method uses tensor decomposition to split each Euler angle (yaw, pitch, roll) to separate subspaces and models each dimension of the underlying manifold as a cosine curve. We address two key challenges: 1. Almost all HPE datasets suffer from incorrect and inaccurate pose annotations. Hence, we generated a precise and consistent 2D head pose dataset for our training set by rotating 3D head models for a fixed set of poses and rendering the corresponding 2D images. 2. We achieved real-time performance with limited training data as our method accurately captures the nature of rotation of an object from facial landmarks. Once the underlying manifold for rotation around each axis is learned, the model is very fast in predicting unseen data. Our training and testing code is available online along with our trained models: https: //github.com/MahdiGhafoorian/NLML_HPE.
>
---
#### [new 031] Emotion Recognition from Skeleton Data: A Comprehensive Survey
- **分类: cs.CV**

- **简介: 该论文属于情感识别任务，旨在通过身体动作解决传统方法依赖面部或生理信号的问题。论文系统综述了基于骨架数据的情感识别技术，分析了心理模型、数据集、方法分类及技术框架，提出了统一分类体系并探讨了其在心理健康评估中的应用。**

- **链接: [http://arxiv.org/pdf/2507.18026v1](http://arxiv.org/pdf/2507.18026v1)**

> **作者:** Haifeng Lu; Jiuyi Chen; Zhen Zhang; Ruida Liu; Runhao Zeng; Xiping Hu
>
> **备注:** 34 pages, 5 figures, 13 tables
>
> **摘要:** Emotion recognition through body movements has emerged as a compelling and privacy-preserving alternative to traditional methods that rely on facial expressions or physiological signals. Recent advancements in 3D skeleton acquisition technologies and pose estimation algorithms have significantly enhanced the feasibility of emotion recognition based on full-body motion. This survey provides a comprehensive and systematic review of skeleton-based emotion recognition techniques. First, we introduce psychological models of emotion and examine the relationship between bodily movements and emotional expression. Next, we summarize publicly available datasets, highlighting the differences in data acquisition methods and emotion labeling strategies. We then categorize existing methods into posture-based and gait-based approaches, analyzing them from both data-driven and technical perspectives. In particular, we propose a unified taxonomy that encompasses four primary technical paradigms: Traditional approaches, Feat2Net, FeatFusionNet, and End2EndNet. Representative works within each category are reviewed and compared, with benchmarking results across commonly used datasets. Finally, we explore the extended applications of emotion recognition in mental health assessment, such as detecting depression and autism, and discuss the open challenges and future research directions in this rapidly evolving field.
>
---
#### [new 032] Comparison of Segmentation Methods in Remote Sensing for Land Use Land Cover
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于遥感图像处理任务，旨在解决土地利用与覆盖（LULC）分类问题。研究比较了多种分割方法，结合大气校正、监督与半监督学习模型（如DeeplabV3+和改进的CPS），提升LULC映射精度，并通过印度海得拉巴的案例展示其在城市规划中的应用价值。**

- **链接: [http://arxiv.org/pdf/2507.18099v1](http://arxiv.org/pdf/2507.18099v1)**

> **作者:** Naman Srivastava; Joel D Joy; Yash Dixit; Swarup E; Rakshit Ramesh
>
> **摘要:** Land Use Land Cover (LULC) mapping is essential for urban and resource planning, and is one of the key elements in developing smart and sustainable cities.This study evaluates advanced LULC mapping techniques, focusing on Look-Up Table (LUT)-based Atmospheric Correction applied to Cartosat Multispectral (MX) sensor images, followed by supervised and semi-supervised learning models for LULC prediction. We explore DeeplabV3+ and Cross-Pseudo Supervision (CPS). The CPS model is further refined with dynamic weighting, enhancing pseudo-label reliability during training. This comprehensive approach analyses the accuracy and utility of LULC mapping techniques for various urban planning applications. A case study of Hyderabad, India, illustrates significant land use changes due to rapid urbanization. By analyzing Cartosat MX images over time, we highlight shifts such as urban sprawl, shrinking green spaces, and expanding industrial areas. This demonstrates the practical utility of these techniques for urban planners and policymakers.
>
---
#### [new 033] Synthetic Data Augmentation for Enhanced Chicken Carcass Instance Segmentation
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于实例分割任务，旨在解决鸡肉分割数据不足导致模型性能差的问题。作者构建了首个逼真合成数据生成流程，并发布300张真实标注数据集。通过引入合成数据增强，在少量真实数据情况下显著提升了模型性能，为禽类自动化检测提供了有效方案。**

- **链接: [http://arxiv.org/pdf/2507.18558v1](http://arxiv.org/pdf/2507.18558v1)**

> **作者:** Yihong Feng; Chaitanya Pallerla; Xiaomin Lin; Pouya Sohrabipour Sr; Philip Crandall; Wan Shou; Yu She; Dongyi Wang
>
> **备注:** Submitted for journal reviewing
>
> **摘要:** The poultry industry has been driven by broiler chicken production and has grown into the world's largest animal protein sector. Automated detection of chicken carcasses on processing lines is vital for quality control, food safety, and operational efficiency in slaughterhouses and poultry processing plants. However, developing robust deep learning models for tasks like instance segmentation in these fast-paced industrial environments is often hampered by the need for laborious acquisition and annotation of large-scale real-world image datasets. We present the first pipeline generating photo-realistic, automatically labeled synthetic images of chicken carcasses. We also introduce a new benchmark dataset containing 300 annotated real-world images, curated specifically for poultry segmentation research. Using these datasets, this study investigates the efficacy of synthetic data and automatic data annotation to enhance the instance segmentation of chicken carcasses, particularly when real annotated data from the processing line is scarce. A small real dataset with varying proportions of synthetic images was evaluated in prominent instance segmentation models. Results show that synthetic data significantly boosts segmentation performance for chicken carcasses across all models. This research underscores the value of synthetic data augmentation as a viable and effective strategy to mitigate data scarcity, reduce manual annotation efforts, and advance the development of robust AI-driven automated detection systems for chicken carcasses in the poultry processing industry.
>
---
#### [new 034] Distributional Uncertainty for Out-of-Distribution Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于语义分割中的分布外检测（OoD）任务，旨在解决现有方法对模型或数据不确定性建模不足的问题。作者提出了一种基于自由能的后验网络框架，结合Beta分布密度估计与残差预测分支，实现对未知区域的精细不确定性估计，提升检测效果。**

- **链接: [http://arxiv.org/pdf/2507.18106v1](http://arxiv.org/pdf/2507.18106v1)**

> **作者:** JinYoung Kim; DaeUng Jo; Kimin Yun; Jeonghyo Song; Youngjoon Yoo
>
> **备注:** 6 pages , 3 figures , IEEE International Conference on Advanced Visual and Signal-Based Systems
>
> **摘要:** Estimating uncertainty from deep neural networks is a widely used approach for detecting out-of-distribution (OoD) samples, which typically exhibit high predictive uncertainty. However, conventional methods such as Monte Carlo (MC) Dropout often focus solely on either model or data uncertainty, failing to align with the semantic objective of OoD detection. To address this, we propose the Free-Energy Posterior Network, a novel framework that jointly models distributional uncertainty and identifying OoD and misclassified regions using free energy. Our method introduces two key contributions: (1) a free-energy-based density estimator parameterized by a Beta distribution, which enables fine-grained uncertainty estimation near ambiguous or unseen regions; and (2) a loss integrated within a posterior network, allowing direct uncertainty estimation from learned parameters without requiring stochastic sampling. By integrating our approach with the residual prediction branch (RPL) framework, the proposed method goes beyond post-hoc energy thresholding and enables the network to learn OoD regions by leveraging the variance of the Beta distribution, resulting in a semantically meaningful and computationally efficient solution for uncertainty-aware segmentation. We validate the effectiveness of our method on challenging real-world benchmarks, including Fishyscapes, RoadAnomaly, and Segment-Me-If-You-Can.
>
---
#### [new 035] 3D Test-time Adaptation via Graph Spectral Driven Point Shift
- **分类: cs.CV**

- **简介: 该论文属于3D点云分类任务，旨在解决测试时域适应（TTA）中点云结构不规则导致的适应效率低问题。作者提出GSDTTA方法，通过图谱域变换优化低频分量，结合自训练策略，实现高效模型适应，提升跨域分类性能。**

- **链接: [http://arxiv.org/pdf/2507.18225v1](http://arxiv.org/pdf/2507.18225v1)**

> **作者:** Xin Wei; Qin Yang; Yijie Fang; Mingrui Zhu; Nannan Wang
>
> **摘要:** While test-time adaptation (TTA) methods effectively address domain shifts by dynamically adapting pre-trained models to target domain data during online inference, their application to 3D point clouds is hindered by their irregular and unordered structure. Current 3D TTA methods often rely on computationally expensive spatial-domain optimizations and may require additional training data. In contrast, we propose Graph Spectral Domain Test-Time Adaptation (GSDTTA), a novel approach for 3D point cloud classification that shifts adaptation to the graph spectral domain, enabling more efficient adaptation by capturing global structural properties with fewer parameters. Point clouds in target domain are represented as outlier-aware graphs and transformed into graph spectral domain by Graph Fourier Transform (GFT). For efficiency, adaptation is performed by optimizing only the lowest 10% of frequency components, which capture the majority of the point cloud's energy. An inverse GFT (IGFT) is then applied to reconstruct the adapted point cloud with the graph spectral-driven point shift. This process is enhanced by an eigenmap-guided self-training strategy that iteratively refines both the spectral adjustments and the model parameters. Experimental results and ablation studies on benchmark datasets demonstrate the effectiveness of GSDTTA, outperforming existing TTA methods for 3D point cloud classification.
>
---
#### [new 036] TeEFusion: Blending Text Embeddings to Distill Classifier-Free Guidance
- **分类: cs.CV**

- **简介: 论文提出TeEFusion，用于文本到图像生成任务，旨在解决分类器无关引导（CFG）带来的高推理成本问题。通过融合条件与无条件文本嵌入，无需额外参数即可重建引导效果，使学生模型高效学习教师模型的复杂采样策略，在保持图像质量的同时显著提升推理速度。**

- **链接: [http://arxiv.org/pdf/2507.18192v1](http://arxiv.org/pdf/2507.18192v1)**

> **作者:** Minghao Fu; Guo-Hua Wang; Xiaohao Chen; Qing-Guo Chen; Zhao Xu; Weihua Luo; Kaifu Zhang
>
> **备注:** Accepted by ICCV 2025. The code is publicly available at https://github.com/AIDC-AI/TeEFusion
>
> **摘要:** Recent advances in text-to-image synthesis largely benefit from sophisticated sampling strategies and classifier-free guidance (CFG) to ensure high-quality generation. However, CFG's reliance on two forward passes, especially when combined with intricate sampling algorithms, results in prohibitively high inference costs. To address this, we introduce TeEFusion (\textbf{Te}xt \textbf{E}mbeddings \textbf{Fusion}), a novel and efficient distillation method that directly incorporates the guidance magnitude into the text embeddings and distills the teacher model's complex sampling strategy. By simply fusing conditional and unconditional text embeddings using linear operations, TeEFusion reconstructs the desired guidance without adding extra parameters, simultaneously enabling the student model to learn from the teacher's output produced via its sophisticated sampling approach. Extensive experiments on state-of-the-art models such as SD3 demonstrate that our method allows the student to closely mimic the teacher's performance with a far simpler and more efficient sampling strategy. Consequently, the student model achieves inference speeds up to 6$\times$ faster than the teacher model, while maintaining image quality at levels comparable to those obtained through the teacher's complex sampling approach. The code is publicly available at \href{https://github.com/AIDC-AI/TeEFusion}{github.com/AIDC-AI/TeEFusion}.
>
---
#### [new 037] CRUISE: Cooperative Reconstruction and Editing in V2X Scenarios using Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶与V2X通信任务，旨在解决V2X场景中数据生成与增强不足的问题。论文提出了CRUISE框架，基于高斯点绘技术实现真实场景重建与灵活编辑，支持多视角渲染与大规模数据增强，提升了3D检测与跟踪性能，并能生成复杂边缘案例。**

- **链接: [http://arxiv.org/pdf/2507.18473v1](http://arxiv.org/pdf/2507.18473v1)**

> **作者:** Haoran Xu; Saining Zhang; Peishuo Li; Baijun Ye; Xiaoxue Chen; Huan-ang Gao; Jv Zheng; Xiaowei Song; Ziqiao Peng; Run Miao; Jinrang Jia; Yifeng Shi; Guangqi Yi; Hang Zhao; Hao Tang; Hongyang Li; Kaicheng Yu; Hao Zhao
>
> **备注:** IROS 2025, Code: https://github.com/SainingZhang/CRUISE
>
> **摘要:** Vehicle-to-everything (V2X) communication plays a crucial role in autonomous driving, enabling cooperation between vehicles and infrastructure. While simulation has significantly contributed to various autonomous driving tasks, its potential for data generation and augmentation in V2X scenarios remains underexplored. In this paper, we introduce CRUISE, a comprehensive reconstruction-and-synthesis framework designed for V2X driving environments. CRUISE employs decomposed Gaussian Splatting to accurately reconstruct real-world scenes while supporting flexible editing. By decomposing dynamic traffic participants into editable Gaussian representations, CRUISE allows for seamless modification and augmentation of driving scenes. Furthermore, the framework renders images from both ego-vehicle and infrastructure views, enabling large-scale V2X dataset augmentation for training and evaluation. Our experimental results demonstrate that: 1) CRUISE reconstructs real-world V2X driving scenes with high fidelity; 2) using CRUISE improves 3D detection across ego-vehicle, infrastructure, and cooperative views, as well as cooperative 3D tracking on the V2X-Seq benchmark; and 3) CRUISE effectively generates challenging corner cases.
>
---
#### [new 038] Enhancing Scene Transition Awareness in Video Generation via Post-Training
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成任务，旨在解决生成长视频时场景转换不连贯的问题。现有模型因训练数据多为单场景片段，难以理解多场景提示。作者构建了包含多场景转换的TAV数据集，通过后训练提升模型对场景转换的理解与生成能力。**

- **链接: [http://arxiv.org/pdf/2507.18046v1](http://arxiv.org/pdf/2507.18046v1)**

> **作者:** Hanwen Shen; Jiajie Lu; Yupeng Cao; Xiaonan Yang
>
> **摘要:** Recent advances in AI-generated video have shown strong performance on \emph{text-to-video} tasks, particularly for short clips depicting a single scene. However, current models struggle to generate longer videos with coherent scene transitions, primarily because they cannot infer when a transition is needed from the prompt. Most open-source models are trained on datasets consisting of single-scene video clips, which limits their capacity to learn and respond to prompts requiring multiple scenes. Developing scene transition awareness is essential for multi-scene generation, as it allows models to identify and segment videos into distinct clips by accurately detecting transitions. To address this, we propose the \textbf{Transition-Aware Video} (TAV) dataset, which consists of preprocessed video clips with multiple scene transitions. Our experiment shows that post-training on the \textbf{TAV} dataset improves prompt-based scene transition understanding, narrows the gap between required and generated scenes, and maintains image quality.
>
---
#### [new 039] Improving Large Vision-Language Models' Understanding for Field Data
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型任务，旨在提升大模型对科学领域数据的理解。现有模型在自然图像和文本任务表现好，但处理科学场数据（如流场、涡旋）效果不足。论文提出FieldLVLM框架，包含场感知语言生成和数据压缩的模型调优，通过提取物理特征并生成结构化文本，优化模型输入，从而提升性能。**

- **链接: [http://arxiv.org/pdf/2507.18311v1](http://arxiv.org/pdf/2507.18311v1)**

> **作者:** Xiaomei Zhang; Hanyu Zheng; Xiangyu Zhu; Jinghuan Wei; Junhong Zou; Zhen Lei; Zhaoxiang Zhang
>
> **摘要:** Large Vision-Language Models (LVLMs) have shown impressive capabilities across a range of tasks that integrate visual and textual understanding, such as image captioning and visual question answering. These models are trained on large-scale image and video datasets paired with text, enabling them to bridge visual perception and natural language processing. However, their application to scientific domains, especially in interpreting complex field data commonly used in the natural sciences, remains underexplored. In this work, we introduce FieldLVLM, a novel framework designed to improve large vision-language models' understanding of field data. FieldLVLM consists of two main components: a field-aware language generation strategy and a data-compressed multimodal model tuning. The field-aware language generation strategy leverages a special-purpose machine learning pipeline to extract key physical features from field data, such as flow classification, Reynolds number, and vortex patterns. This information is then converted into structured textual descriptions that serve as a dataset. The data-compressed multimodal model tuning focuses on LVLMs with these generated datasets, using a data compression strategy to reduce the complexity of field inputs and retain only the most informative values. This ensures compatibility with the models language decoder and guides its learning more effectively. Experimental results on newly proposed benchmark datasets demonstrate that FieldLVLM significantly outperforms existing methods in tasks involving scientific field data. Our findings suggest that this approach opens up new possibilities for applying large vision-language models to scientific research, helping bridge the gap between large models and domain-specific discovery.
>
---
#### [new 040] Exploring the interplay of label bias with subgroup size and separability: A case study in mammographic density classification
- **分类: cs.CV**

- **简介: 该论文研究医疗影像数据集中标签偏差对深度学习模型特征学习和性能的影响，属于医学图像分类任务。通过模拟不同子组大小和可分性下的标签偏差，发现标签偏差会导致特征表示显著变化，并影响子组性能。使用清洁验证集定义分类阈值可缓解偏差影响。**

- **链接: [http://arxiv.org/pdf/2507.17996v1](http://arxiv.org/pdf/2507.17996v1)**

> **作者:** Emma A. M. Stanley; Raghav Mehta; Mélanie Roschewitz; Nils D. Forkert; Ben Glocker
>
> **备注:** Accepted at MICCAI Workshop on Fairness of AI in Medical Imaging (FAIMI) 2025
>
> **摘要:** Systematic mislabelling affecting specific subgroups (i.e., label bias) in medical imaging datasets represents an understudied issue concerning the fairness of medical AI systems. In this work, we investigated how size and separability of subgroups affected by label bias influence the learned features and performance of a deep learning model. Therefore, we trained deep learning models for binary tissue density classification using the EMory BrEast imaging Dataset (EMBED), where label bias affected separable subgroups (based on imaging manufacturer) or non-separable "pseudo-subgroups". We found that simulated subgroup label bias led to prominent shifts in the learned feature representations of the models. Importantly, these shifts within the feature space were dependent on both the relative size and the separability of the subgroup affected by label bias. We also observed notable differences in subgroup performance depending on whether a validation set with clean labels was used to define the classification threshold for the model. For instance, with label bias affecting the majority separable subgroup, the true positive rate for that subgroup fell from 0.898, when the validation set had clean labels, to 0.518, when the validation set had biased labels. Our work represents a key contribution toward understanding the consequences of label bias on subgroup fairness in medical imaging AI.
>
---
#### [new 041] SIDA: Synthetic Image Driven Zero-shot Domain Adaptation
- **分类: cs.CV; cs.AI; cs.LG; cs.MM**

- **简介: 该论文属于零样本域适应任务，旨在不使用目标域图像的情况下提升模型跨域性能。现有方法依赖文本描述，难以捕捉复杂变化且效率低。论文提出SIDA，利用合成图像生成目标域风格特征，通过域混合与局部风格迁移建模真实世界变化，实现了高效且性能优越的零样本域适应。**

- **链接: [http://arxiv.org/pdf/2507.18632v1](http://arxiv.org/pdf/2507.18632v1)**

> **作者:** Ye-Chan Kim; SeungJu Cha; Si-Woo Kim; Taewhan Kim; Dong-Jin Kim
>
> **备注:** Accepted to ACM MM 2025
>
> **摘要:** Zero-shot domain adaptation is a method for adapting a model to a target domain without utilizing target domain image data. To enable adaptation without target images, existing studies utilize CLIP's embedding space and text description to simulate target-like style features. Despite the previous achievements in zero-shot domain adaptation, we observe that these text-driven methods struggle to capture complex real-world variations and significantly increase adaptation time due to their alignment process. Instead of relying on text descriptions, we explore solutions leveraging image data, which provides diverse and more fine-grained style cues. In this work, we propose SIDA, a novel and efficient zero-shot domain adaptation method leveraging synthetic images. To generate synthetic images, we first create detailed, source-like images and apply image translation to reflect the style of the target domain. We then utilize the style features of these synthetic images as a proxy for the target domain. Based on these features, we introduce Domain Mix and Patch Style Transfer modules, which enable effective modeling of real-world variations. In particular, Domain Mix blends multiple styles to expand the intra-domain representations, and Patch Style Transfer assigns different styles to individual patches. We demonstrate the effectiveness of our method by showing state-of-the-art performance in diverse zero-shot adaptation scenarios, particularly in challenging domains. Moreover, our approach achieves high efficiency by significantly reducing the overall adaptation time.
>
---
#### [new 042] Object segmentation in the wild with foundation models: application to vision assisted neuro-prostheses for upper limbs
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决自然场景中的语义物体分割问题，以支持上肢神经假体的视觉辅助。论文提出了一种基于注视点生成提示的方法，引导Segment Anything Model进行分割，并在真实数据集Grasping-in-the-Wild上进行了评估，提升了IoU指标。**

- **链接: [http://arxiv.org/pdf/2507.18517v1](http://arxiv.org/pdf/2507.18517v1)**

> **作者:** Bolutife Atoki; Jenny Benois-Pineau; Renaud Péteri; Fabien Baldacci; Aymar de Rugy
>
> **摘要:** In this work, we address the problem of semantic object segmentation using foundation models. We investigate whether foundation models, trained on a large number and variety of objects, can perform object segmentation without fine-tuning on specific images containing everyday objects, but in highly cluttered visual scenes. The ''in the wild'' context is driven by the target application of vision guided upper limb neuroprostheses. We propose a method for generating prompts based on gaze fixations to guide the Segment Anything Model (SAM) in our segmentation scenario, and fine-tune it on egocentric visual data. Evaluation results of our approach show an improvement of the IoU segmentation quality metric by up to 0.51 points on real-world challenging data of Grasping-in-the-Wild corpus which is made available on the RoboFlow Platform (https://universe.roboflow.com/iwrist/grasping-in-the-wild)
>
---
#### [new 043] DATA: Domain-And-Time Alignment for High-Quality Feature Fusion in Collaborative Perception
- **分类: cs.CV**

- **简介: 该论文属于协同感知任务，旨在解决特征融合中的域间差异和时序不对齐问题。提出了DATA网络，包含域对齐、时序对齐和特征聚合模块，提升融合特征的质量。实验表明方法在多个数据集上表现优异，具有鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.18237v1](http://arxiv.org/pdf/2507.18237v1)**

> **作者:** Chengchang Tian; Jianwei Ma; Yan Huang; Zhanye Chen; Honghao Wei; Hui Zhang; Wei Hong
>
> **备注:** ICCV 2025, accepted as poster. 22 pages including supplementary materials
>
> **摘要:** Feature-level fusion shows promise in collaborative perception (CP) through balanced performance and communication bandwidth trade-off. However, its effectiveness critically relies on input feature quality. The acquisition of high-quality features faces domain gaps from hardware diversity and deployment conditions, alongside temporal misalignment from transmission delays. These challenges degrade feature quality with cumulative effects throughout the collaborative network. In this paper, we present the Domain-And-Time Alignment (DATA) network, designed to systematically align features while maximizing their semantic representations for fusion. Specifically, we propose a Consistency-preserving Domain Alignment Module (CDAM) that reduces domain gaps through proximal-region hierarchical downsampling and observability-constrained discriminator. We further propose a Progressive Temporal Alignment Module (PTAM) to handle transmission delays via multi-scale motion modeling and two-stage compensation. Building upon the aligned features, an Instance-focused Feature Aggregation Module (IFAM) is developed to enhance semantic representations. Extensive experiments demonstrate that DATA achieves state-of-the-art performance on three typical datasets, maintaining robustness with severe communication delays and pose errors. The code will be released at https://github.com/ChengchangTian/DATA.
>
---
#### [new 044] HybridTM: Combining Transformer and Mamba for 3D Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于3D语义分割任务，旨在解决Transformer计算复杂度高和Mamba特征表达能力弱的问题。论文提出HybridTM，首次将Transformer与Mamba结合，并引入细粒度混合策略，以同时捕捉长距离依赖和局部特征，提升了分割性能。**

- **链接: [http://arxiv.org/pdf/2507.18575v1](http://arxiv.org/pdf/2507.18575v1)**

> **作者:** Xinyu Wang; Jinghua Hou; Zhe Liu; Yingying Zhu
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** Transformer-based methods have demonstrated remarkable capabilities in 3D semantic segmentation through their powerful attention mechanisms, but the quadratic complexity limits their modeling of long-range dependencies in large-scale point clouds. While recent Mamba-based approaches offer efficient processing with linear complexity, they struggle with feature representation when extracting 3D features. However, effectively combining these complementary strengths remains an open challenge in this field. In this paper, we propose HybridTM, the first hybrid architecture that integrates Transformer and Mamba for 3D semantic segmentation. In addition, we propose the Inner Layer Hybrid Strategy, which combines attention and Mamba at a finer granularity, enabling simultaneous capture of long-range dependencies and fine-grained local features. Extensive experiments demonstrate the effectiveness and generalization of our HybridTM on diverse indoor and outdoor datasets. Furthermore, our HybridTM achieves state-of-the-art performance on ScanNet, ScanNet200, and nuScenes benchmarks. The code will be made available at https://github.com/deepinact/HybridTM.
>
---
#### [new 045] Adapting Large VLMs with Iterative and Manual Instructions for Generative Low-light Enhancement
- **分类: cs.CV**

- **简介: 该论文属于低光图像增强任务，旨在解决现有方法忽略正常光图像语义引导、难以处理复杂光照条件的问题。作者提出VLM-IMI框架，利用大视觉-语言模型结合迭代手动指令，通过融合文本与图像特征，提升低光图像的语义连贯性与细节恢复效果。**

- **链接: [http://arxiv.org/pdf/2507.18064v1](http://arxiv.org/pdf/2507.18064v1)**

> **作者:** Xiaoran Sun; Liyan Wang; Cong Wang; Yeying Jin; Kin-man Lam; Zhixun Su; Yang Yang; Jinshan Pan
>
> **摘要:** Most existing low-light image enhancement (LLIE) methods rely on pre-trained model priors, low-light inputs, or both, while neglecting the semantic guidance available from normal-light images. This limitation hinders their effectiveness in complex lighting conditions. In this paper, we propose VLM-IMI, a novel framework that leverages large vision-language models (VLMs) with iterative and manual instructions (IMIs) for LLIE. VLM-IMI incorporates textual descriptions of the desired normal-light content as enhancement cues, enabling semantically informed restoration. To effectively integrate cross-modal priors, we introduce an instruction prior fusion module, which dynamically aligns and fuses image and text features, promoting the generation of detailed and semantically coherent outputs. During inference, we adopt an iterative and manual instruction strategy to refine textual instructions, progressively improving visual quality. This refinement enhances structural fidelity, semantic alignment, and the recovery of fine details under extremely low-light conditions. Extensive experiments across diverse scenarios demonstrate that VLM-IMI outperforms state-of-the-art methods in both quantitative metrics and perceptual quality. The source code is available at https://github.com/sunxiaoran01/VLM-IMI.
>
---
#### [new 046] Delving into Mapping Uncertainty for Mapless Trajectory Prediction
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的轨迹预测任务，旨在解决在线生成地图的不确定性对轨迹预测的影响。论文分析了地图不确定性在哪些场景中对预测有益，提出基于自车运动状态的不确定性融合方法，并改进地图不确定性建模，提升了预测性能。**

- **链接: [http://arxiv.org/pdf/2507.18498v1](http://arxiv.org/pdf/2507.18498v1)**

> **作者:** Zongzheng Zhang; Xuchong Qiu; Boran Zhang; Guantian Zheng; Xunjiang Gu; Guoxuan Chi; Huan-ang Gao; Leichen Wang; Ziming Liu; Xinrun Li; Igor Gilitschenski; Hongyang Li; Hang Zhao; Hao Zhao
>
> **备注:** Accepted to IROS 2025, Project Page: https://ethan-zheng136.github.io/Dev-Unc/
>
> **摘要:** Recent advances in autonomous driving are moving towards mapless approaches, where High-Definition (HD) maps are generated online directly from sensor data, reducing the need for expensive labeling and maintenance. However, the reliability of these online-generated maps remains uncertain. While incorporating map uncertainty into downstream trajectory prediction tasks has shown potential for performance improvements, current strategies provide limited insights into the specific scenarios where this uncertainty is beneficial. In this work, we first analyze the driving scenarios in which mapping uncertainty has the greatest positive impact on trajectory prediction and identify a critical, previously overlooked factor: the agent's kinematic state. Building on these insights, we propose a novel Proprioceptive Scenario Gating that adaptively integrates map uncertainty into trajectory prediction based on forecasts of the ego vehicle's future kinematics. This lightweight, self-supervised approach enhances the synergy between online mapping and trajectory prediction, providing interpretability around where uncertainty is advantageous and outperforming previous integration methods. Additionally, we introduce a Covariance-based Map Uncertainty approach that better aligns with map geometry, further improving trajectory prediction. Extensive ablation studies confirm the effectiveness of our approach, achieving up to 23.6% improvement in mapless trajectory prediction performance over the state-of-the-art method using the real-world nuScenes driving dataset. Our code, data, and models are publicly available at https://github.com/Ethan-Zheng136/Map-Uncertainty-for-Trajectory-Prediction.
>
---
#### [new 047] Unposed 3DGS Reconstruction with Probabilistic Procrustes Mapping
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决无姿态图像序列下的3D高斯重建问题。现有方法依赖预训练MVS模型，但在处理大量图像时面临内存与精度问题。论文提出一种结合概率Procrustes映射与联合优化的新框架，实现高效全局对齐与重建，在Waymo和KITTI数据集上取得SOTA效果。**

- **链接: [http://arxiv.org/pdf/2507.18541v1](http://arxiv.org/pdf/2507.18541v1)**

> **作者:** Chong Cheng; Zijian Wang; Sicheng Yu; Yu Hu; Nanjie Yao; Hao Wang
>
> **摘要:** 3D Gaussian Splatting (3DGS) has emerged as a core technique for 3D representation. Its effectiveness largely depends on precise camera poses and accurate point cloud initialization, which are often derived from pretrained Multi-View Stereo (MVS) models. However, in unposed reconstruction task from hundreds of outdoor images, existing MVS models may struggle with memory limits and lose accuracy as the number of input images grows. To address this limitation, we propose a novel unposed 3DGS reconstruction framework that integrates pretrained MVS priors with the probabilistic Procrustes mapping strategy. The method partitions input images into subsets, maps submaps into a global space, and jointly optimizes geometry and poses with 3DGS. Technically, we formulate the mapping of tens of millions of point clouds as a probabilistic Procrustes problem and solve a closed-form alignment. By employing probabilistic coupling along with a soft dustbin mechanism to reject uncertain correspondences, our method globally aligns point clouds and poses within minutes across hundreds of images. Moreover, we propose a joint optimization framework for 3DGS and camera poses. It constructs Gaussians from confidence-aware anchor points and integrates 3DGS differentiable rendering with an analytical Jacobian to jointly refine scene and poses, enabling accurate reconstruction and pose estimation. Experiments on Waymo and KITTI datasets show that our method achieves accurate reconstruction from unposed image sequences, setting a new state of the art for unposed 3DGS reconstruction.
>
---
#### [new 048] AG-VPReID.VIR: Bridging Aerial and Ground Platforms for Video-based Visible-Infrared Person Re-ID
- **分类: cs.CV**

- **简介: 该论文属于跨模态行人重识别任务，旨在解决地面与空中视角、可见光与红外图像之间的行人匹配问题。作者提出了首个空地跨模态视频数据集AG-VPReID.VIR，并设计了TCC-VPReID模型以应对视角、模态和时间差异带来的挑战。**

- **链接: [http://arxiv.org/pdf/2507.17995v1](http://arxiv.org/pdf/2507.17995v1)**

> **作者:** Huy Nguyen; Kien Nguyen; Akila Pemasiri; Akmal Jahan; Clinton Fookes; Sridha Sridharan
>
> **备注:** Accepted atIEEE International Joint Conference on Biometrics (IJCB) 2025
>
> **摘要:** Person re-identification (Re-ID) across visible and infrared modalities is crucial for 24-hour surveillance systems, but existing datasets primarily focus on ground-level perspectives. While ground-based IR systems offer nighttime capabilities, they suffer from occlusions, limited coverage, and vulnerability to obstructions--problems that aerial perspectives uniquely solve. To address these limitations, we introduce AG-VPReID.VIR, the first aerial-ground cross-modality video-based person Re-ID dataset. This dataset captures 1,837 identities across 4,861 tracklets (124,855 frames) using both UAV-mounted and fixed CCTV cameras in RGB and infrared modalities. AG-VPReID.VIR presents unique challenges including cross-viewpoint variations, modality discrepancies, and temporal dynamics. Additionally, we propose TCC-VPReID, a novel three-stream architecture designed to address the joint challenges of cross-platform and cross-modality person Re-ID. Our approach bridges the domain gaps between aerial-ground perspectives and RGB-IR modalities, through style-robust feature learning, memory-based cross-view adaptation, and intermediary-guided temporal modeling. Experiments show that AG-VPReID.VIR presents distinctive challenges compared to existing datasets, with our TCC-VPReID framework achieving significant performance gains across multiple evaluation protocols. Dataset and code are available at https://github.com/agvpreid25/AG-VPReID.VIR.
>
---
#### [new 049] Towards Effective Human-in-the-Loop Assistive AI Agents
- **分类: cs.CV**

- **简介: 该论文研究人类与AI协作完成物理任务，旨在提升任务表现、减少错误并促进学习。作者构建了评估框架与多模态数据集，并开发了具备增强现实功能的AI助手，应用于烹饪、战场医疗等场景。通过实验证明AI辅助能有效提升任务完成效果。**

- **链接: [http://arxiv.org/pdf/2507.18374v1](http://arxiv.org/pdf/2507.18374v1)**

> **作者:** Filippos Bellos; Yayuan Li; Cary Shu; Ruey Day; Jeffrey M. Siskind; Jason J. Corso
>
> **备注:** 10 pages, 5 figures, 2 tables
>
> **摘要:** Effective human-AI collaboration for physical task completion has significant potential in both everyday activities and professional domains. AI agents equipped with informative guidance can enhance human performance, but evaluating such collaboration remains challenging due to the complexity of human-in-the-loop interactions. In this work, we introduce an evaluation framework and a multimodal dataset of human-AI interactions designed to assess how AI guidance affects procedural task performance, error reduction and learning outcomes. Besides, we develop an augmented reality (AR)-equipped AI agent that provides interactive guidance in real-world tasks, from cooking to battlefield medicine. Through human studies, we share empirical insights into AI-assisted human performance and demonstrate that AI-assisted collaboration improves task completion.
>
---
#### [new 050] DRWKV: Focusing on Object Edges for Low-Light Image Enhancement
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于低光照图像增强任务，旨在解决光照不足导致的物体边缘模糊和结构细节丢失问题。论文提出了DRWKV模型，融合GER理论、Evolving WKV注意力机制、Bi-SAB和MS2-Loss，有效提升图像质量并保持边缘连续性。**

- **链接: [http://arxiv.org/pdf/2507.18594v1](http://arxiv.org/pdf/2507.18594v1)**

> **作者:** Xuecheng Bai; Yuxiang Wang; Boyu Hu; Qinyuan Jie; Chuanzhi Xu; Hongru Xiao; Kechen Li; Vera Chung
>
> **摘要:** Low-light image enhancement remains a challenging task, particularly in preserving object edge continuity and fine structural details under extreme illumination degradation. In this paper, we propose a novel model, DRWKV (Detailed Receptance Weighted Key Value), which integrates our proposed Global Edge Retinex (GER) theory, enabling effective decoupling of illumination and edge structures for enhanced edge fidelity. Secondly, we introduce Evolving WKV Attention, a spiral-scanning mechanism that captures spatial edge continuity and models irregular structures more effectively. Thirdly, we design the Bilateral Spectrum Aligner (Bi-SAB) and a tailored MS2-Loss to jointly align luminance and chrominance features, improving visual naturalness and mitigating artifacts. Extensive experiments on five LLIE benchmarks demonstrate that DRWKV achieves leading performance in PSNR, SSIM, and NIQE while maintaining low computational complexity. Furthermore, DRWKV enhances downstream performance in low-light multi-object tracking tasks, validating its generalization capabilities.
>
---
#### [new 051] Captain Cinema: Towards Short Movie Generation
- **分类: cs.CV**

- **简介: 该论文提出“Captain Cinema”，用于短视频生成任务。解决长叙事视频生成中视觉与情节连贯性不足的问题。通过“自上而下关键帧规划”生成关键帧序列，再结合“自下而上视频合成”生成帧间动态内容。引入适用于长视频的多模态扩散模型训练策略，实现高质量、高效率的短片自动生成。**

- **链接: [http://arxiv.org/pdf/2507.18634v1](http://arxiv.org/pdf/2507.18634v1)**

> **作者:** Junfei Xiao; Ceyuan Yang; Lvmin Zhang; Shengqu Cai; Yang Zhao; Yuwei Guo; Gordon Wetzstein; Maneesh Agrawala; Alan Yuille; Lu Jiang
>
> **备注:** Under review. Project page: https://thecinema.ai
>
> **摘要:** We present Captain Cinema, a generation framework for short movie generation. Given a detailed textual description of a movie storyline, our approach firstly generates a sequence of keyframes that outline the entire narrative, which ensures long-range coherence in both the storyline and visual appearance (e.g., scenes and characters). We refer to this step as top-down keyframe planning. These keyframes then serve as conditioning signals for a video synthesis model, which supports long context learning, to produce the spatio-temporal dynamics between them. This step is referred to as bottom-up video synthesis. To support stable and efficient generation of multi-scene long narrative cinematic works, we introduce an interleaved training strategy for Multimodal Diffusion Transformers (MM-DiT), specifically adapted for long-context video data. Our model is trained on a specially curated cinematic dataset consisting of interleaved data pairs. Our experiments demonstrate that Captain Cinema performs favorably in the automated creation of visually coherent and narrative consistent short movies in high quality and efficiency. Project page: https://thecinema.ai
>
---
#### [new 052] T2VWorldBench: A Benchmark for Evaluating World Knowledge in Text-to-Video Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到视频生成任务，旨在解决现有模型缺乏世界知识理解与应用能力的问题。作者构建了T2VWorldBench基准，涵盖6大类、60小类、1200个提示，结合人工与自动评估方法，评估10个先进模型，发现其普遍存在事实性错误，揭示了模型在常识推理和事实生成上的不足。**

- **链接: [http://arxiv.org/pdf/2507.18107v1](http://arxiv.org/pdf/2507.18107v1)**

> **作者:** Yubin Chen; Xuyang Guo; Zhenmei Shi; Zhao Song; Jiahao Zhang
>
> **摘要:** Text-to-video (T2V) models have shown remarkable performance in generating visually reasonable scenes, while their capability to leverage world knowledge for ensuring semantic consistency and factual accuracy remains largely understudied. In response to this challenge, we propose T2VWorldBench, the first systematic evaluation framework for evaluating the world knowledge generation abilities of text-to-video models, covering 6 major categories, 60 subcategories, and 1,200 prompts across a wide range of domains, including physics, nature, activity, culture, causality, and object. To address both human preference and scalable evaluation, our benchmark incorporates both human evaluation and automated evaluation using vision-language models (VLMs). We evaluated the 10 most advanced text-to-video models currently available, ranging from open source to commercial models, and found that most models are unable to understand world knowledge and generate truly correct videos. These findings point out a critical gap in the capability of current text-to-video models to leverage world knowledge, providing valuable research opportunities and entry points for constructing models with robust capabilities for commonsense reasoning and factual generation.
>
---
#### [new 053] Beyond Low-rankness: Guaranteed Matrix Recovery via Modified Nuclear Norm
- **分类: cs.CV**

- **简介: 该论文属于矩阵恢复任务，旨在解决传统方法难以同时利用局部信息与全局低秩结构的问题。作者提出一种新的修正核范数框架，无需调参即可融合两者，并提供理论保证。实验表明其有效性。**

- **链接: [http://arxiv.org/pdf/2507.18327v1](http://arxiv.org/pdf/2507.18327v1)**

> **作者:** Jiangjun Peng; Yisi Luo; Xiangyong Cao; Shuang Xu; Deyu Meng
>
> **备注:** 15 pages, 14 figures
>
> **摘要:** The nuclear norm (NN) has been widely explored in matrix recovery problems, such as Robust PCA and matrix completion, leveraging the inherent global low-rank structure of the data. In this study, we introduce a new modified nuclear norm (MNN) framework, where the MNN family norms are defined by adopting suitable transformations and performing the NN on the transformed matrix. The MNN framework offers two main advantages: (1) it jointly captures both local information and global low-rankness without requiring trade-off parameter tuning; (2) Under mild assumptions on the transformation, we provided exact theoretical recovery guarantees for both Robust PCA and MC tasks-an achievement not shared by existing methods that combine local and global information. Thanks to its general and flexible design, MNN can accommodate various proven transformations, enabling a unified and effective approach to structured low-rank recovery. Extensive experiments demonstrate the effectiveness of our method. Code and supplementary material are available at https://github.com/andrew-pengjj/modified_nuclear_norm.
>
---
#### [new 054] DSFormer: A Dual-Scale Cross-Learning Transformer for Visual Place Recognition
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉地点识别（VPR）任务，旨在解决环境和视角变化下识别不准确的问题。作者提出DSFormer模型，通过双尺度特征交互和跨尺度学习增强特征表示，并采用新的数据聚类策略优化训练集，提升识别鲁棒性和计算效率。**

- **链接: [http://arxiv.org/pdf/2507.18444v1](http://arxiv.org/pdf/2507.18444v1)**

> **作者:** Haiyang Jiang; Songhao Piao; Chao Gao; Lei Yu; Liguo Chen
>
> **摘要:** Visual Place Recognition (VPR) is crucial for robust mobile robot localization, yet it faces significant challenges in maintaining reliable performance under varying environmental conditions and viewpoints. To address this, we propose a novel framework that integrates Dual-Scale-Former (DSFormer), a Transformer-based cross-learning module, with an innovative block clustering strategy. DSFormer enhances feature representation by enabling bidirectional information transfer between dual-scale features extracted from the final two CNN layers, capturing both semantic richness and spatial details through self-attention for long-range dependencies within each scale and shared cross-attention for cross-scale learning. Complementing this, our block clustering strategy repartitions the widely used San Francisco eXtra Large (SF-XL) training dataset from multiple distinct perspectives, optimizing data organization to further bolster robustness against viewpoint variations. Together, these innovations not only yield a robust global embedding adaptable to environmental changes but also reduce the required training data volume by approximately 30\% compared to previous partitioning methods. Comprehensive experiments demonstrate that our approach achieves state-of-the-art performance across most benchmark datasets, surpassing advanced reranking methods like DELG, Patch-NetVLAD, TransVPR, and R2Former as a global retrieval solution using 512-dim global descriptors, while significantly improving computational efficiency.
>
---
#### [new 055] VB-Mitigator: An Open-source Framework for Evaluating and Advancing Visual Bias Mitigation
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉中的视觉偏差缓解任务，旨在解决模型偏差导致的不公平、不可靠问题。论文提出了VB-Mitigator开源框架，集成12种方法和7个数据集，支持扩展，用于统一评估和比较偏差缓解技术，推动公平性研究。**

- **链接: [http://arxiv.org/pdf/2507.18348v1](http://arxiv.org/pdf/2507.18348v1)**

> **作者:** Ioannis Sarridis; Christos Koutlis; Symeon Papadopoulos; Christos Diou
>
> **摘要:** Bias in computer vision models remains a significant challenge, often resulting in unfair, unreliable, and non-generalizable AI systems. Although research into bias mitigation has intensified, progress continues to be hindered by fragmented implementations and inconsistent evaluation practices. Disparate datasets and metrics used across studies complicate reproducibility, making it difficult to fairly assess and compare the effectiveness of various approaches. To overcome these limitations, we introduce the Visual Bias Mitigator (VB-Mitigator), an open-source framework designed to streamline the development, evaluation, and comparative analysis of visual bias mitigation techniques. VB-Mitigator offers a unified research environment encompassing 12 established mitigation methods, 7 diverse benchmark datasets. A key strength of VB-Mitigator is its extensibility, allowing for seamless integration of additional methods, datasets, metrics, and models. VB-Mitigator aims to accelerate research toward fairness-aware computer vision models by serving as a foundational codebase for the research community to develop and assess their approaches. To this end, we also recommend best evaluation practices and provide a comprehensive performance comparison among state-of-the-art methodologies.
>
---
#### [new 056] AFRDA: Attentive Feature Refinement for Domain Adaptive Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于无监督域自适应语义分割任务，旨在解决现有方法在局部细节与全局信息平衡上的不足。作者提出AFR模块，通过低分辨率语义先验优化高分辨率特征，并引入高频成分和注意力机制，提升分割精度。在多个数据集上取得SOTA结果。**

- **链接: [http://arxiv.org/pdf/2507.17957v1](http://arxiv.org/pdf/2507.17957v1)**

> **作者:** Md. Al-Masrur Khan; Durgakant Pushp; Lantao Liu
>
> **摘要:** In Unsupervised Domain Adaptive Semantic Segmentation (UDA-SS), a model is trained on labeled source domain data (e.g., synthetic images) and adapted to an unlabeled target domain (e.g., real-world images) without access to target annotations. Existing UDA-SS methods often struggle to balance fine-grained local details with global contextual information, leading to segmentation errors in complex regions. To address this, we introduce the Adaptive Feature Refinement (AFR) module, which enhances segmentation accuracy by refining highresolution features using semantic priors from low-resolution logits. AFR also integrates high-frequency components, which capture fine-grained structures and provide crucial boundary information, improving object delineation. Additionally, AFR adaptively balances local and global information through uncertaintydriven attention, reducing misclassifications. Its lightweight design allows seamless integration into HRDA-based UDA methods, leading to state-of-the-art segmentation performance. Our approach improves existing UDA-SS methods by 1.05% mIoU on GTA V --> Cityscapes and 1.04% mIoU on Synthia-->Cityscapes. The implementation of our framework is available at: https://github.com/Masrur02/AFRDA
>
---
#### [new 057] SV3.3B: A Sports Video Understanding Model for Action Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频理解任务，旨在解决体育视频中动作识别与分析的难题。现有方法计算量大、缺乏细粒度动作理解。论文提出SV3.3B模型，采用轻量化设计与自监督学习，实现高效设备端部署，并在篮球数据集上表现出色，生成技术详尽的运动描述。**

- **链接: [http://arxiv.org/pdf/2507.17844v1](http://arxiv.org/pdf/2507.17844v1)**

> **作者:** Sai Varun Kodathala; Yashwanth Reddy Vutukoori; Rakesh Vunnam
>
> **备注:** 8 pages, 6 figures, 4 tables. Submitted to AIxSET 2025
>
> **摘要:** This paper addresses the challenge of automated sports video analysis, which has traditionally been limited by computationally intensive models requiring server-side processing and lacking fine-grained understanding of athletic movements. Current approaches struggle to capture the nuanced biomechanical transitions essential for meaningful sports analysis, often missing critical phases like preparation, execution, and follow-through that occur within seconds. To address these limitations, we introduce SV3.3B, a lightweight 3.3B parameter video understanding model that combines novel temporal motion difference sampling with self-supervised learning for efficient on-device deployment. Our approach employs a DWT-VGG16-LDA based keyframe extraction mechanism that intelligently identifies the 16 most representative frames from sports sequences, followed by a V-DWT-JEPA2 encoder pretrained through mask-denoising objectives and an LLM decoder fine-tuned for sports action description generation. Evaluated on a subset of the NSVA basketball dataset, SV3.3B achieves superior performance across both traditional text generation metrics and sports-specific evaluation criteria, outperforming larger closed-source models including GPT-4o variants while maintaining significantly lower computational requirements. Our model demonstrates exceptional capability in generating technically detailed and analytically rich sports descriptions, achieving 29.2% improvement over GPT-4o in ground truth validation metrics, with substantial improvements in information density, action complexity, and measurement precision metrics essential for comprehensive athletic analysis. Model Available at https://huggingface.co/sportsvision/SV3.3B.
>
---
#### [new 058] EgoExoBench: A Benchmark for First- and Third-person View Video Understanding in MLLMs
- **分类: cs.CV**

- **简介: 该论文提出了EgoExoBench，首个用于第一视角与第三视角视频理解的基准。任务是评估多模态大模型在跨视角语义对齐、视角关联和时序推理上的能力。论文构建了包含7,300个问答对的评测集，发现当前模型在单视角任务表现好，但跨视角理解仍存在挑战。**

- **链接: [http://arxiv.org/pdf/2507.18342v1](http://arxiv.org/pdf/2507.18342v1)**

> **作者:** Yuping He; Yifei Huang; Guo Chen; Baoqi Pei; Jilan Xu; Tong Lu; Jiangmiao Pang
>
> **摘要:** Transferring and integrating knowledge across first-person (egocentric) and third-person (exocentric) viewpoints is intrinsic to human intelligence, enabling humans to learn from others and convey insights from their own experiences. Despite rapid progress in multimodal large language models (MLLMs), their ability to perform such cross-view reasoning remains unexplored. To address this, we introduce EgoExoBench, the first benchmark for egocentric-exocentric video understanding and reasoning. Built from publicly available datasets, EgoExoBench comprises over 7,300 question-answer pairs spanning eleven sub-tasks organized into three core challenges: semantic alignment, viewpoint association, and temporal reasoning. We evaluate 13 state-of-the-art MLLMs and find that while these models excel on single-view tasks, they struggle to align semantics across perspectives, accurately associate views, and infer temporal dynamics in the ego-exo context. We hope EgoExoBench can serve as a valuable resource for research on embodied agents and intelligent assistants seeking human-like cross-view intelligence.
>
---
#### [new 059] PDB-Eval: An Evaluation of Large Multimodal Models for Description and Explanation of Personalized Driving Behavior
- **分类: cs.CV**

- **简介: 该论文属于多模态模型评估任务，旨在解决驾驶员行为理解和意图预测的问题。论文构建了PDB-Eval基准，包括PDB-X和PDB-QA，用于评估和优化大语言模型对驾驶行为的理解与推理能力，并通过微调提升其在实际驾驶任务中的表现。**

- **链接: [http://arxiv.org/pdf/2507.18447v1](http://arxiv.org/pdf/2507.18447v1)**

> **作者:** Junda Wu; Jessica Echterhoff; Kyungtae Han; Amr Abdelraouf; Rohit Gupta; Julian McAuley
>
> **摘要:** Understanding a driver's behavior and intentions is important for potential risk assessment and early accident prevention. Safety and driver assistance systems can be tailored to individual drivers' behavior, significantly enhancing their effectiveness. However, existing datasets are limited in describing and explaining general vehicle movements based on external visual evidence. This paper introduces a benchmark, PDB-Eval, for a detailed understanding of Personalized Driver Behavior, and aligning Large Multimodal Models (MLLMs) with driving comprehension and reasoning. Our benchmark consists of two main components, PDB-X and PDB-QA. PDB-X can evaluate MLLMs' understanding of temporal driving scenes. Our dataset is designed to find valid visual evidence from the external view to explain the driver's behavior from the internal view. To align MLLMs' reasoning abilities with driving tasks, we propose PDB-QA as a visual explanation question-answering task for MLLM instruction fine-tuning. As a generic learning task for generative models like MLLMs, PDB-QA can bridge the domain gap without harming MLLMs' generalizability. Our evaluation indicates that fine-tuning MLLMs on fine-grained descriptions and explanations can effectively bridge the gap between MLLMs and the driving domain, which improves zero-shot performance on question-answering tasks by up to 73.2%. We further evaluate the MLLMs fine-tuned on PDB-X in Brain4Cars' intention prediction and AIDE's recognition tasks. We observe up to 12.5% performance improvements on the turn intention prediction task in Brain4Cars, and consistent performance improvements up to 11.0% on all tasks in AIDE.
>
---
#### [new 060] MVG4D: Image Matrix-Based Multi-View and Motion Generation for 4D Content Creation from a Single Image
- **分类: cs.CV**

- **简介: 该论文提出MVG4D，旨在从单张图像生成高质量4D内容。任务为4D场景生成，解决动态内容时空不一致、细节模糊等问题。方法结合多视角合成与4D高斯点阵变形，提升时间连贯性与几何精度，优化AR/VR体验。**

- **链接: [http://arxiv.org/pdf/2507.18371v1](http://arxiv.org/pdf/2507.18371v1)**

> **作者:** Xiaotian Chen; DongFu Yin; Fei Richard Yu; Xuanchen Li; Xinhao Zhang
>
> **摘要:** Advances in generative modeling have significantly enhanced digital content creation, extending from 2D images to complex 3D and 4D scenes. Despite substantial progress, producing high-fidelity and temporally consistent dynamic 4D content remains a challenge. In this paper, we propose MVG4D, a novel framework that generates dynamic 4D content from a single still image by combining multi-view synthesis with 4D Gaussian Splatting (4D GS). At its core, MVG4D employs an image matrix module that synthesizes temporally coherent and spatially diverse multi-view images, providing rich supervisory signals for downstream 3D and 4D reconstruction. These multi-view images are used to optimize a 3D Gaussian point cloud, which is further extended into the temporal domain via a lightweight deformation network. Our method effectively enhances temporal consistency, geometric fidelity, and visual realism, addressing key challenges in motion discontinuity and background degradation that affect prior 4D GS-based methods. Extensive experiments on the Objaverse dataset demonstrate that MVG4D outperforms state-of-the-art baselines in CLIP-I, PSNR, FVD, and time efficiency. Notably, it reduces flickering artifacts and sharpens structural details across views and time, enabling more immersive AR/VR experiences. MVG4D sets a new direction for efficient and controllable 4D generation from minimal inputs.
>
---
#### [new 061] Reinforced Embodied Active Defense: Exploiting Adaptive Interaction for Robust Visual Perception in Adversarial 3D Environments
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉感知安全任务，旨在解决3D环境中对抗攻击威胁感知系统可靠性的问题。作者提出Rein-EAD主动防御框架，通过自适应探索与环境交互，优化多步策略以提升鲁棒性，并设计不确定性奖励机制降低计算开销。实验表明其在多种任务中有效抵御攻击，具备良好泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.18484v1](http://arxiv.org/pdf/2507.18484v1)**

> **作者:** Xiao Yang; Lingxuan Wu; Lizhong Wang; Chengyang Ying; Hang Su; Jun Zhu
>
> **备注:** arXiv admin note: text overlap with arXiv:2404.00540
>
> **摘要:** Adversarial attacks in 3D environments have emerged as a critical threat to the reliability of visual perception systems, particularly in safety-sensitive applications such as identity verification and autonomous driving. These attacks employ adversarial patches and 3D objects to manipulate deep neural network (DNN) predictions by exploiting vulnerabilities within complex scenes. Existing defense mechanisms, such as adversarial training and purification, primarily employ passive strategies to enhance robustness. However, these approaches often rely on pre-defined assumptions about adversarial tactics, limiting their adaptability in dynamic 3D settings. To address these challenges, we introduce Reinforced Embodied Active Defense (Rein-EAD), a proactive defense framework that leverages adaptive exploration and interaction with the environment to improve perception robustness in 3D adversarial contexts. By implementing a multi-step objective that balances immediate prediction accuracy with predictive entropy minimization, Rein-EAD optimizes defense strategies over a multi-step horizon. Additionally, Rein-EAD involves an uncertainty-oriented reward-shaping mechanism that facilitates efficient policy updates, thereby reducing computational overhead and supporting real-world applicability without the need for differentiable environments. Comprehensive experiments validate the effectiveness of Rein-EAD, demonstrating a substantial reduction in attack success rates while preserving standard accuracy across diverse tasks. Notably, Rein-EAD exhibits robust generalization to unseen and adaptive attacks, making it suitable for real-world complex tasks, including 3D object classification, face recognition and autonomous driving.
>
---
#### [new 062] Iwin Transformer: Hierarchical Vision Transformer using Interleaved Windows
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于计算机视觉任务，旨在解决视觉Transformer中全局信息交互受限的问题。论文提出Iwin Transformer，结合交错窗口注意力与深度可分离卷积，实现单模块内全局信息交互，无需位置嵌入，支持从低到高分辨率的直接微调，并在图像分类、语义分割等任务中表现出色。**

- **链接: [http://arxiv.org/pdf/2507.18405v1](http://arxiv.org/pdf/2507.18405v1)**

> **作者:** Simin Huo; Ning Li
>
> **备注:** 14 pages, 10 figures, Submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence
>
> **摘要:** We introduce Iwin Transformer, a novel position-embedding-free hierarchical vision transformer, which can be fine-tuned directly from low to high resolution, through the collaboration of innovative interleaved window attention and depthwise separable convolution. This approach uses attention to connect distant tokens and applies convolution to link neighboring tokens, enabling global information exchange within a single module, overcoming Swin Transformer's limitation of requiring two consecutive blocks to approximate global attention. Extensive experiments on visual benchmarks demonstrate that Iwin Transformer exhibits strong competitiveness in tasks such as image classification (87.4 top-1 accuracy on ImageNet-1K), semantic segmentation and video action recognition. We also validate the effectiveness of the core component in Iwin as a standalone module that can seamlessly replace the self-attention module in class-conditional image generation. The concepts and methods introduced by the Iwin Transformer have the potential to inspire future research, like Iwin 3D Attention in video generation. The code and models are available at https://github.com/cominder/Iwin-Transformer.
>
---
#### [new 063] Improving Bird Classification with Primary Color Additives
- **分类: cs.CV; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于鸟类分类任务，旨在解决因环境噪声、重叠鸣叫和标签缺失导致的分类困难。作者通过将频率信息转化为颜色增强频谱图，提升深度学习模型对鸟类物种的区分能力，显著提高了分类准确率。**

- **链接: [http://arxiv.org/pdf/2507.18334v1](http://arxiv.org/pdf/2507.18334v1)**

> **作者:** Ezhini Rasendiran R; Chandresh Kumar Maurya
>
> **备注:** 5 pages (Accepted to Interspeech 2025)
>
> **摘要:** We address the problem of classifying bird species using their song recordings, a challenging task due to environmental noise, overlapping vocalizations, and missing labels. Existing models struggle with low-SNR or multi-species recordings. We hypothesize that birds can be classified by visualizing their pitch pattern, speed, and repetition, collectively called motifs. Deep learning models applied to spectrogram images help, but similar motifs across species cause confusion. To mitigate this, we embed frequency information into spectrograms using primary color additives. This enhances species distinction and improves classification accuracy. Our experiments show that the proposed approach achieves statistically significant gains over models without colorization and surpasses the BirdCLEF 2024 winner, improving F1 by 7.3%, ROC-AUC by 6.2%, and CMAP by 6.6%. These results demonstrate the effectiveness of incorporating frequency information via colorization.
>
---
#### [new 064] Celeb-DF++: A Large-scale Challenging Video DeepFake Benchmark for Generalizable Forensics
- **分类: cs.CV**

- **简介: 该论文属于图像篡改检测任务，旨在解决现有DeepFake数据集伪造类型单一、难以评估检测模型泛化能力的问题。作者构建了大规模、多样化的Celeb-DF++数据集，涵盖多种伪造方法生成的视频，并提出评估协议，用于衡量检测方法在未见伪造类型上的泛化性能。**

- **链接: [http://arxiv.org/pdf/2507.18015v1](http://arxiv.org/pdf/2507.18015v1)**

> **作者:** Yuezun Li; Delong Zhu; Xinjie Cui; Siwei Lyu
>
> **备注:** https://github.com/OUC-VAS/Celeb-DF-PP
>
> **摘要:** The rapid advancement of AI technologies has significantly increased the diversity of DeepFake videos circulating online, posing a pressing challenge for \textit{generalizable forensics}, \ie, detecting a wide range of unseen DeepFake types using a single model. Addressing this challenge requires datasets that are not only large-scale but also rich in forgery diversity. However, most existing datasets, despite their scale, include only a limited variety of forgery types, making them insufficient for developing generalizable detection methods. Therefore, we build upon our earlier Celeb-DF dataset and introduce {Celeb-DF++}, a new large-scale and challenging video DeepFake benchmark dedicated to the generalizable forensics challenge. Celeb-DF++ covers three commonly encountered forgery scenarios: Face-swap (FS), Face-reenactment (FR), and Talking-face (TF). Each scenario contains a substantial number of high-quality forged videos, generated using a total of 22 various recent DeepFake methods. These methods differ in terms of architectures, generation pipelines, and targeted facial regions, covering the most prevalent DeepFake cases witnessed in the wild. We also introduce evaluation protocols for measuring the generalizability of 24 recent detection methods, highlighting the limitations of existing detection methods and the difficulty of our new dataset.
>
---
#### [new 065] Self-Supervised Ultrasound-Video Segmentation with Feature Prediction and 3D Localised Loss
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决超声影像数据标注困难导致的分割性能受限问题。论文采用自监督学习框架V-JEPA进行视频分割，并提出一种3D局部化辅助任务，以提升基于ViT模型的局部特征理解，从而在标注数据有限的情况下显著提高分割性能。**

- **链接: [http://arxiv.org/pdf/2507.18424v1](http://arxiv.org/pdf/2507.18424v1)**

> **作者:** Edward Ellis; Robert Mendel; Andrew Bulpitt; Nasim Parsa; Michael F Byrne; Sharib Ali
>
> **摘要:** Acquiring and annotating large datasets in ultrasound imaging is challenging due to low contrast, high noise, and susceptibility to artefacts. This process requires significant time and clinical expertise. Self-supervised learning (SSL) offers a promising solution by leveraging unlabelled data to learn useful representations, enabling improved segmentation performance when annotated data is limited. Recent state-of-the-art developments in SSL for video data include V-JEPA, a framework solely based on feature prediction, avoiding pixel level reconstruction or negative samples. We hypothesise that V-JEPA is well-suited to ultrasound imaging, as it is less sensitive to noisy pixel-level detail while effectively leveraging temporal information. To the best of our knowledge, this is the first study to adopt V-JEPA for ultrasound video data. Similar to other patch-based masking SSL techniques such as VideoMAE, V-JEPA is well-suited to ViT-based models. However, ViTs can underperform on small medical datasets due to lack of inductive biases, limited spatial locality and absence of hierarchical feature learning. To improve locality understanding, we propose a novel 3D localisation auxiliary task to improve locality in ViT representations during V-JEPA pre-training. Our results show V-JEPA with our auxiliary task improves segmentation performance significantly across various frozen encoder configurations, with gains up to 3.4\% using 100\% and up to 8.35\% using only 10\% of the training data.
>
---
#### [new 066] WaveMamba: Wavelet-Driven Mamba Fusion for RGB-Infrared Object Detection
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于RGB-红外目标检测任务，旨在解决跨模态特征融合问题。作者提出WaveMamba，结合离散小波变换与Mamba框架，设计WaveMamba融合块，有效融合低频和高频特征，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2507.18173v1](http://arxiv.org/pdf/2507.18173v1)**

> **作者:** Haodong Zhu; Wenhao Dong; Linlin Yang; Hong Li; Yuguang Yang; Yangyang Ren; Qingcheng Zhu; Zichao Feng; Changbai Li; Shaohui Lin; Runqi Wang; Xiaoyan Luo; Baochang Zhang
>
> **摘要:** Leveraging the complementary characteristics of visible (RGB) and infrared (IR) imagery offers significant potential for improving object detection. In this paper, we propose WaveMamba, a cross-modality fusion method that efficiently integrates the unique and complementary frequency features of RGB and IR decomposed by Discrete Wavelet Transform (DWT). An improved detection head incorporating the Inverse Discrete Wavelet Transform (IDWT) is also proposed to reduce information loss and produce the final detection results. The core of our approach is the introduction of WaveMamba Fusion Block (WMFB), which facilitates comprehensive fusion across low-/high-frequency sub-bands. Within WMFB, the Low-frequency Mamba Fusion Block (LMFB), built upon the Mamba framework, first performs initial low-frequency feature fusion with channel swapping, followed by deep fusion with an advanced gated attention mechanism for enhanced integration. High-frequency features are enhanced using a strategy that applies an ``absolute maximum" fusion approach. These advancements lead to significant performance gains, with our method surpassing state-of-the-art approaches and achieving average mAP improvements of 4.5% on four benchmarks.
>
---
#### [new 067] A 3D Cross-modal Keypoint Descriptor for MR-US Matching and Registration
- **分类: cs.CV**

- **简介: 该论文属于医学图像配准任务，旨在解决术中实时超声（iUS）与术前磁共振成像（MRI）因模态差异导致的配准难题。论文提出一种3D跨模态关键点描述符，通过合成iUS图像、对比学习与关键点匹配实现鲁棒配准，提升了精度与解释性。**

- **链接: [http://arxiv.org/pdf/2507.18551v1](http://arxiv.org/pdf/2507.18551v1)**

> **作者:** Daniil Morozov; Reuben Dorent; Nazim Haouchine
>
> **备注:** Under review
>
> **摘要:** Intraoperative registration of real-time ultrasound (iUS) to preoperative Magnetic Resonance Imaging (MRI) remains an unsolved problem due to severe modality-specific differences in appearance, resolution, and field-of-view. To address this, we propose a novel 3D cross-modal keypoint descriptor for MRI-iUS matching and registration. Our approach employs a patient-specific matching-by-synthesis approach, generating synthetic iUS volumes from preoperative MRI. This enables supervised contrastive training to learn a shared descriptor space. A probabilistic keypoint detection strategy is then employed to identify anatomically salient and modality-consistent locations. During training, a curriculum-based triplet loss with dynamic hard negative mining is used to learn descriptors that are i) robust to iUS artifacts such as speckle noise and limited coverage, and ii) rotation-invariant . At inference, the method detects keypoints in MR and real iUS images and identifies sparse matches, which are then used to perform rigid registration. Our approach is evaluated using 3D MRI-iUS pairs from the ReMIND dataset. Experiments show that our approach outperforms state-of-the-art keypoint matching methods across 11 patients, with an average precision of $69.8\%$. For image registration, our method achieves a competitive mean Target Registration Error of 2.39 mm on the ReMIND2Reg benchmark. Compared to existing iUS-MR registration approach, our framework is interpretable, requires no manual initialization, and shows robustness to iUS field-of-view variation. Code is available at https://github.com/morozovdd/CrossKEY.
>
---
#### [new 068] DepthDark: Robust Monocular Depth Estimation for Low-Light Environments
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于单目深度估计任务，旨在解决低光环境下深度估计效果差的问题。论文提出了DepthDark模型，通过模拟夜间成像过程生成高质量低光深度数据集，并设计了低光环境下的参数高效微调策略，提升了模型在低光条件下的鲁棒性与估计精度。**

- **链接: [http://arxiv.org/pdf/2507.18243v1](http://arxiv.org/pdf/2507.18243v1)**

> **作者:** Longjian Zeng; Zunjie Zhu; Rongfeng Lu; Ming Lu; Bolun Zheng; Chenggang Yan; Anke Xue
>
> **备注:** Accepted by ACM MM 2025 conference
>
> **摘要:** In recent years, foundation models for monocular depth estimation have received increasing attention. Current methods mainly address typical daylight conditions, but their effectiveness notably decreases in low-light environments. There is a lack of robust foundational models for monocular depth estimation specifically designed for low-light scenarios. This largely stems from the absence of large-scale, high-quality paired depth datasets for low-light conditions and the effective parameter-efficient fine-tuning (PEFT) strategy. To address these challenges, we propose DepthDark, a robust foundation model for low-light monocular depth estimation. We first introduce a flare-simulation module and a noise-simulation module to accurately simulate the imaging process under nighttime conditions, producing high-quality paired depth datasets for low-light conditions. Additionally, we present an effective low-light PEFT strategy that utilizes illumination guidance and multiscale feature fusion to enhance the model's capability in low-light environments. Our method achieves state-of-the-art depth estimation performance on the challenging nuScenes-Night and RobotCar-Night datasets, validating its effectiveness using limited training data and computing resources.
>
---
#### [new 069] Q-Former Autoencoder: A Modern Framework for Medical Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决无监督医学异常检测问题。作者提出了基于Q-Former的自编码框架，利用预训练视觉模型（如DINO、Masked Autoencoder）提取特征，并通过Q-Former瓶颈聚合多尺度信息，实现高效重建。方法在多个医学数据集上达到先进性能，验证了预训练模型在医学领域的泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.18481v1](http://arxiv.org/pdf/2507.18481v1)**

> **作者:** Francesco Dalmonte; Emirhan Bayar; Emre Akbas; Mariana-Iuliana Georgescu
>
> **备注:** 15 pages
>
> **摘要:** Anomaly detection in medical images is an important yet challenging task due to the diversity of possible anomalies and the practical impossibility of collecting comprehensively annotated data sets. In this work, we tackle unsupervised medical anomaly detection proposing a modernized autoencoder-based framework, the Q-Former Autoencoder, that leverages state-of-the-art pretrained vision foundation models, such as DINO, DINOv2 and Masked Autoencoder. Instead of training encoders from scratch, we directly utilize frozen vision foundation models as feature extractors, enabling rich, multi-stage, high-level representations without domain-specific fine-tuning. We propose the usage of the Q-Former architecture as the bottleneck, which enables the control of the length of the reconstruction sequence, while efficiently aggregating multiscale features. Additionally, we incorporate a perceptual loss computed using features from a pretrained Masked Autoencoder, guiding the reconstruction towards semantically meaningful structures. Our framework is evaluated on four diverse medical anomaly detection benchmarks, achieving state-of-the-art results on BraTS2021, RESC, and RSNA. Our results highlight the potential of vision foundation model encoders, pretrained on natural images, to generalize effectively to medical image analysis tasks without further fine-tuning. We release the code and models at https://github.com/emirhanbayar/QFAE.
>
---
#### [new 070] Towards Consistent Long-Term Pose Generation
- **分类: cs.CV**

- **简介: 该论文属于姿态生成任务，旨在解决长期姿态生成中因中间表示和误差累积导致的时空不连贯问题。作者提出一种新颖的一阶段模型，直接在连续坐标空间中生成姿态，通过相对运动预测机制和统一占位符标记方法，实现训练与推理行为一致，提升了长期姿态生成的准确性与连贯性。**

- **链接: [http://arxiv.org/pdf/2507.18382v1](http://arxiv.org/pdf/2507.18382v1)**

> **作者:** Yayuan Li; Filippos Bellos; Jason Corso
>
> **备注:** 10 pages, 5 figures, 4 tables
>
> **摘要:** Current approaches to pose generation rely heavily on intermediate representations, either through two-stage pipelines with quantization or autoregressive models that accumulate errors during inference. This fundamental limitation leads to degraded performance, particularly in long-term pose generation where maintaining temporal coherence is crucial. We propose a novel one-stage architecture that directly generates poses in continuous coordinate space from minimal context - a single RGB image and text description - while maintaining consistent distributions between training and inference. Our key innovation is eliminating the need for intermediate representations or token-based generation by operating directly on pose coordinates through a relative movement prediction mechanism that preserves spatial relationships, and a unified placeholder token approach that enables single-forward generation with identical behavior during training and inference. Through extensive experiments on Penn Action and First-Person Hand Action Benchmark (F-PHAB) datasets, we demonstrate that our approach significantly outperforms existing quantization-based and autoregressive methods, especially in long-term generation scenarios.
>
---
#### [new 071] IntentVCNet: Bridging Spatio-Temporal Gaps for Intention-Oriented Controllable Video Captioning
- **分类: cs.CV**

- **简介: 该论文属于视频描述生成任务，旨在根据用户意图生成对视频中特定目标的可控描述。现有大视觉语言模型在时空理解上存在细粒度控制不足的问题。论文提出IntentVCNet，通过提示组合策略和高效的框适配器，统一时空理解，增强模型对空间细节的建模能力，实现更精确的意图导向视频描述。**

- **链接: [http://arxiv.org/pdf/2507.18531v1](http://arxiv.org/pdf/2507.18531v1)**

> **作者:** Tianheng Qiu; Jingchun Gao; Jingyu Li; Huiyi Leong; Xuan Huang; Xi Wang; Xiaocheng Zhang; Kele Xu; Lan Zhang
>
> **摘要:** Intent-oriented controlled video captioning aims to generate targeted descriptions for specific targets in a video based on customized user intent. Current Large Visual Language Models (LVLMs) have gained strong instruction following and visual comprehension capabilities. Although the LVLMs demonstrated proficiency in spatial and temporal understanding respectively, it was not able to perform fine-grained spatial control in time sequences in direct response to instructions. This substantial spatio-temporal gap complicates efforts to achieve fine-grained intention-oriented control in video. Towards this end, we propose a novel IntentVCNet that unifies the temporal and spatial understanding knowledge inherent in LVLMs to bridge the spatio-temporal gap from both prompting and model perspectives. Specifically, we first propose a prompt combination strategy designed to enable LLM to model the implicit relationship between prompts that characterize user intent and video sequences. We then propose a parameter efficient box adapter that augments the object semantic information in the global visual context so that the visual token has a priori information about the user intent. The final experiment proves that the combination of the two strategies can further enhance the LVLM's ability to model spatial details in video sequences, and facilitate the LVLMs to accurately generate controlled intent-oriented captions. Our proposed method achieved state-of-the-art results in several open source LVLMs and was the runner-up in the IntentVC challenge. Our code is available on https://github.com/thqiu0419/IntentVCNet.
>
---
#### [new 072] Deformable Convolution Module with Globally Learned Relative Offsets for Fundus Vessel Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决视网膜血管分割中复杂边缘特征的建模问题。作者提出了一种新的可变形卷积模块，通过注意力和前馈网络学习全局相对偏移，实现跨通道特征变形，提升了模型表达与泛化能力。基于该模块构建的GDCUnet在公开数据集上取得了最优性能。**

- **链接: [http://arxiv.org/pdf/2507.18354v1](http://arxiv.org/pdf/2507.18354v1)**

> **作者:** Lexuan Zhu; Yuxuan Li; Yuning Ren
>
> **摘要:** Deformable convolution can adaptively change the shape of convolution kernel by learning offsets to deal with complex shape features. We propose a novel plug and play deformable convolutional module that uses attention and feedforward networks to learn offsets, so that the deformable patterns can capture long-distance global features. Compared with previously existing deformable convolutions, the proposed module learns the sub pixel displacement field and adaptively warps the feature maps across all channels rather than directly deforms the convolution kernel , which is equivalent to a relative deformation of the kernel sampling grids, achieving global feature deformation and the decoupling of kernel size and learning network. Considering that the fundus blood vessels have globally self similar complex edges, we design a deep learning model for fundus blood vessel segmentation, GDCUnet, based on the proposed convolutional module. Empirical evaluations under the same configuration and unified framework show that GDCUnet has achieved state of the art performance on public datasets. Further ablation experiments demonstrated that the proposed deformable convolutional module could more significantly learn the complex features of fundus blood vessels, enhancing the model representation and generalization capabilities.The proposed module is similar to the interface of conventional convolution, we suggest applying it to more machine vision tasks with complex global self similar features.
>
---
#### [new 073] A Multimodal Seq2Seq Transformer for Predicting Brain Responses to Naturalistic Stimuli
- **分类: cs.CV; q-bio.NC**

- **简介: 该论文属于脑活动预测任务，旨在解决如何准确预测自然刺激下全脑fMRI响应的问题。作者提出了一种多模态序列到序列Transformer模型，结合视觉、听觉和语言输入，利用预训练模型提取特征，并通过双交叉注意力机制整合感知与叙事信息，同时结合共享编码器与个体化解码器以兼顾共性与个体差异，有效捕捉刺激与神经响应的长时程结构。**

- **链接: [http://arxiv.org/pdf/2507.18104v1](http://arxiv.org/pdf/2507.18104v1)**

> **作者:** Qianyi He; Yuan Chang Leong
>
> **摘要:** The Algonauts 2025 Challenge called on the community to develop encoding models that predict whole-brain fMRI responses to naturalistic multimodal movies. In this submission, we propose a sequence-to-sequence Transformer that autoregressively predicts fMRI activity from visual, auditory, and language inputs. Stimulus features were extracted using pretrained models including VideoMAE, HuBERT, Qwen, and BridgeTower. The decoder integrates information from prior brain states, current stimuli, and episode-level summaries via dual cross-attention mechanisms that attend to both perceptual information extracted from the stimulus as well as narrative information provided by high-level summaries of narrative content. One core innovation of our approach is the use of sequences of multimodal context to predict sequences of brain activity, enabling the model to capture long-range temporal structure in both stimuli and neural responses. Another is the combination of a shared encoder with partial subject-specific decoder, which leverages common structure across subjects while accounting for individual variability. Our model achieves strong performance on both in-distribution and out-of-distribution data, demonstrating the effectiveness of temporally-aware, multimodal sequence modeling for brain activity prediction. The code is available at https://github.com/Angelneer926/Algonauts_challenge.
>
---
#### [new 074] Towards Large Scale Geostatistical Methane Monitoring with Part-based Object Detection
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉与遥感任务，旨在解决大规模稀疏目标检测问题。针对生物沼气池稀少且分布广、难以监测的问题，作者提出一种基于部件的目标检测方法，利用少量样本提升检测效果，并应用于法国生物沼气池的甲烷排放监测。**

- **链接: [http://arxiv.org/pdf/2507.18513v1](http://arxiv.org/pdf/2507.18513v1)**

> **作者:** Adhemar de Senneville; Xavier Bou; Thibaud Ehret; Rafael Grompone; Jean Louis Bonne; Nicolas Dumelie; Thomas Lauvaux; Gabriele Facciolo
>
> **摘要:** Object detection is one of the main applications of computer vision in remote sensing imagery. Despite its increasing availability, the sheer volume of remote sensing data poses a challenge when detecting rare objects across large geographic areas. Paradoxically, this common challenge is crucial to many applications, such as estimating environmental impact of certain human activities at scale. In this paper, we propose to address the problem by investigating the methane production and emissions of bio-digesters in France. We first introduce a novel dataset containing bio-digesters, with small training and validation sets, and a large test set with a high imbalance towards observations without objects since such sites are rare. We develop a part-based method that considers essential bio-digester sub-elements to boost initial detections. To this end, we apply our method to new, unseen regions to build an inventory of bio-digesters. We then compute geostatistical estimates of the quantity of methane produced that can be attributed to these infrastructures in a given area at a given time.
>
---
#### [new 075] A COCO-Formatted Instance-Level Dataset for Plasmodium Falciparum Detection in Giemsa-Stained Blood Smears
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决疟疾检测中缺乏高质量实例标注数据的问题。作者改进了现有疟疾数据集，提供了COCO格式的详细标注，并通过训练Faster R-CNN模型验证其有效性，提升了检测性能。**

- **链接: [http://arxiv.org/pdf/2507.18483v1](http://arxiv.org/pdf/2507.18483v1)**

> **作者:** Frauke Wilm; Luis Carlos Rivera Monroy; Mathias Öttl; Lukas Mürdter; Leonid Mill; Andreas Maier
>
> **备注:** 7 pages, 4 figures, 2 tables, accepted at MICCAI 2025 Open Data
>
> **摘要:** Accurate detection of Plasmodium falciparum in Giemsa-stained blood smears is an essential component of reliable malaria diagnosis, especially in developing countries. Deep learning-based object detection methods have demonstrated strong potential for automated Malaria diagnosis, but their adoption is limited by the scarcity of datasets with detailed instance-level annotations. In this work, we present an enhanced version of the publicly available NIH malaria dataset, with detailed bounding box annotations in COCO format to support object detection training. We validated the revised annotations by training a Faster R-CNN model to detect infected and non-infected red blood cells, as well as white blood cells. Cross-validation on the original dataset yielded F1 scores of up to 0.88 for infected cell detection. These results underscore the importance of annotation volume and consistency, and demonstrate that automated annotation refinement combined with targeted manual correction can produce training data of sufficient quality for robust detection performance. The updated annotations set is publicly available via GitHub: https://github.com/MIRA-Vision-Microscopy/malaria-thin-smear-coco.
>
---
#### [new 076] SynC: Synthetic Image Caption Dataset Refinement with One-to-many Mapping for Zero-shot Image Captioning
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于零样本图像描述生成（ZIC）任务，旨在解决合成图像与描述语义不一致的问题。作者提出SynC方法，通过多候选图像检索与循环一致性评分重新匹配图像与描述，提升合成数据质量。实验证明该方法在多个基准数据集上显著提升ZIC性能。**

- **链接: [http://arxiv.org/pdf/2507.18616v1](http://arxiv.org/pdf/2507.18616v1)**

> **作者:** Si-Woo Kim; MinJu Jeon; Ye-Chan Kim; Soeun Lee; Taewhan Kim; Dong-Jin Kim
>
> **备注:** Accepted to ACM Multimedia 2025
>
> **摘要:** Zero-shot Image Captioning (ZIC) increasingly utilizes synthetic datasets generated by text-to-image (T2I) models to mitigate the need for costly manual annotation. However, these T2I models often produce images that exhibit semantic misalignments with their corresponding input captions (e.g., missing objects, incorrect attributes), resulting in noisy synthetic image-caption pairs that can hinder model training. Existing dataset pruning techniques are largely designed for removing noisy text in web-crawled data. However, these methods are ill-suited for the distinct challenges of synthetic data, where captions are typically well-formed, but images may be inaccurate representations. To address this gap, we introduce SynC, a novel framework specifically designed to refine synthetic image-caption datasets for ZIC. Instead of conventional filtering or regeneration, SynC focuses on reassigning captions to the most semantically aligned images already present within the synthetic image pool. Our approach employs a one-to-many mapping strategy by initially retrieving multiple relevant candidate images for each caption. We then apply a cycle-consistency-inspired alignment scorer that selects the best image by verifying its ability to retrieve the original caption via image-to-text retrieval. Extensive evaluations demonstrate that SynC consistently and significantly improves performance across various ZIC models on standard benchmarks (MS-COCO, Flickr30k, NoCaps), achieving state-of-the-art results in several scenarios. SynC offers an effective strategy for curating refined synthetic data to enhance ZIC.
>
---
#### [new 077] Deep Learning-Based Age Estimation and Gender Deep Learning-Based Age Estimation and Gender Classification for Targeted Advertisement
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉与深度学习任务，旨在解决基于人脸图像的年龄估计与性别分类问题，以提升广告投放效果。论文提出了一种定制的CNN架构，联合学习年龄和性别信息，利用其相关性提升性能。实验显示性别分类准确率达95%，年龄估计平均误差为5.77年，并分析了不同年龄组的表现差异。**

- **链接: [http://arxiv.org/pdf/2507.18565v1](http://arxiv.org/pdf/2507.18565v1)**

> **作者:** Muhammad Imran Zaman; Nisar Ahmed
>
> **备注:** 6
>
> **摘要:** This paper presents a novel deep learning-based approach for simultaneous age and gender classification from facial images, designed to enhance the effectiveness of targeted advertising campaigns. We propose a custom Convolutional Neural Network (CNN) architecture, optimized for both tasks, which leverages the inherent correlation between age and gender information present in facial features. Unlike existing methods that often treat these tasks independently, our model learns shared representations, leading to improved performance. The network is trained on a large, diverse dataset of facial images, carefully pre-processed to ensure robustness against variations in lighting, pose, and image quality. Our experimental results demonstrate a significant improvement in gender classification accuracy, achieving 95%, and a competitive mean absolute error of 5.77 years for age estimation. Critically, we analyze the performance across different age groups, identifying specific challenges in accurately estimating the age of younger individuals. This analysis reveals the need for targeted data augmentation and model refinement to address these biases. Furthermore, we explore the impact of different CNN architectures and hyperparameter settings on the overall performance, providing valuable insights for future research.
>
---
#### [new 078] Information Entropy-Based Framework for Quantifying Tortuosity in Meibomian Gland Uneven Atrophy
- **分类: cs.CV; cs.IT; math.IT**

- **简介: 该论文属于医学图像分析任务，旨在解决量化评估曲线迂曲度（tortuosity）的问题。作者提出了一种基于信息熵的迂曲度量化框架，通过与参考曲线比较，实现对睑板腺萎缩均匀性的客观评估，并验证了其在区分蠕形螨阴性和阳性患者中的有效性。**

- **链接: [http://arxiv.org/pdf/2507.18135v1](http://arxiv.org/pdf/2507.18135v1)**

> **作者:** Kesheng Wang; Xiaoyu Chen; Chunlei He; Fenfen Li; Xinxin Yu; Dexing Kong; Shoujun Huang; Qi Dai
>
> **备注:** This manuscript contains 7 figures. All comments are welcome
>
> **摘要:** In the medical image analysis field, precise quantification of curve tortuosity plays a critical role in the auxiliary diagnosis and pathological assessment of various diseases. In this study, we propose a novel framework for tortuosity quantification and demonstrate its effectiveness through the evaluation of meibomian gland atrophy uniformity,serving as a representative application scenario. We introduce an information entropy-based tortuosity quantification framework that integrates probability modeling with entropy theory and incorporates domain transformation of curve data. Unlike traditional methods such as curvature or arc-chord ratio, this approach evaluates the tortuosity of a target curve by comparing it to a designated reference curve. Consequently, it is more suitable for tortuosity assessment tasks in medical data where biologically plausible reference curves are available, providing a more robust and objective evaluation metric without relying on idealized straight-line comparisons. First, we conducted numerical simulation experiments to preliminarily assess the stability and validity of the method. Subsequently, the framework was applied to quantify the spatial uniformity of meibomian gland atrophy and to analyze the difference in this uniformity between \textit{Demodex}-negative and \textit{Demodex}-positive patient groups. The results demonstrated a significant difference in tortuosity-based uniformity between the two groups, with an area under the curve of 0.8768, sensitivity of 0.75, and specificity of 0.93. These findings highlight the clinical utility of the proposed framework in curve tortuosity analysis and its potential as a generalizable tool for quantitative morphological evaluation in medical diagnostics.
>
---
#### [new 079] TextSAM-EUS: Text Prompt Learning for SAM to Accurately Segment Pancreatic Tumor in Endoscopic Ultrasound
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，旨在解决胰腺肿瘤在内窥镜超声（EUS）图像中因噪声、低对比度和形态复杂导致的分割难题。作者提出TextSAM-EUS，结合文本提示学习与轻量级模型适配，实现无需手动几何提示的自动分割，仅微调0.86%的参数，性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.18082v1](http://arxiv.org/pdf/2507.18082v1)**

> **作者:** Pascal Spiegler; Taha Koleilat; Arash Harirpoush; Corey S. Miller; Hassan Rivaz; Marta Kersten-Oertel; Yiming Xiao
>
> **备注:** Accepted to ICCV 2025 Workshop CVAMD
>
> **摘要:** Pancreatic cancer carries a poor prognosis and relies on endoscopic ultrasound (EUS) for targeted biopsy and radiotherapy. However, the speckle noise, low contrast, and unintuitive appearance of EUS make segmentation of pancreatic tumors with fully supervised deep learning (DL) models both error-prone and dependent on large, expert-curated annotation datasets. To address these challenges, we present TextSAM-EUS, a novel, lightweight, text-driven adaptation of the Segment Anything Model (SAM) that requires no manual geometric prompts at inference. Our approach leverages text prompt learning (context optimization) through the BiomedCLIP text encoder in conjunction with a LoRA-based adaptation of SAM's architecture to enable automatic pancreatic tumor segmentation in EUS, tuning only 0.86% of the total parameters. On the public Endoscopic Ultrasound Database of the Pancreas, TextSAM-EUS with automatic prompts attains 82.69% Dice and 85.28% normalized surface distance (NSD), and with manual geometric prompts reaches 83.10% Dice and 85.70% NSD, outperforming both existing state-of-the-art (SOTA) supervised DL models and foundation models (e.g., SAM and its variants). As the first attempt to incorporate prompt learning in SAM-based medical image segmentation, TextSAM-EUS offers a practical option for efficient and robust automatic EUS segmentation. Our code will be publicly available upon acceptance.
>
---
#### [new 080] TTS-VAR: A Test-Time Scaling Framework for Visual Auto-Regressive Generation
- **分类: cs.CV**

- **简介: 该论文属于视觉生成任务，旨在解决大规模模型部署成本高的问题。提出TTS-VAR框架，在测试阶段动态调整计算资源，通过自适应批处理和多尺度搜索策略，提升生成质量与效率。核心创新包括聚类多样性搜索与重采样潜力选择。**

- **链接: [http://arxiv.org/pdf/2507.18537v1](http://arxiv.org/pdf/2507.18537v1)**

> **作者:** Zhekai Chen; Ruihang Chu; Yukang Chen; Shiwei Zhang; Yujie Wei; Yingya Zhang; Xihui Liu
>
> **备注:** 10 Tables, 9 Figures
>
> **摘要:** Scaling visual generation models is essential for real-world content creation, yet requires substantial training and computational expenses. Alternatively, test-time scaling has garnered growing attention due to resource efficiency and promising performance. In this work, we present TTS-VAR, the first general test-time scaling framework for visual auto-regressive (VAR) models, modeling the generation process as a path searching problem. To dynamically balance computational efficiency with exploration capacity, we first introduce an adaptive descending batch size schedule throughout the causal generation process. Besides, inspired by VAR's hierarchical coarse-to-fine multi-scale generation, our framework integrates two key components: (i) At coarse scales, we observe that generated tokens are hard for evaluation, possibly leading to erroneous acceptance of inferior samples or rejection of superior samples. Noticing that the coarse scales contain sufficient structural information, we propose clustering-based diversity search. It preserves structural variety through semantic feature clustering, enabling later selection on samples with higher potential. (ii) In fine scales, resampling-based potential selection prioritizes promising candidates using potential scores, which are defined as reward functions incorporating multi-scale generation history. Experiments on the powerful VAR model Infinity show a notable 8.7% GenEval score improvement (from 0.69 to 0.75). Key insights reveal that early-stage structural features effectively influence final quality, and resampling efficacy varies across generation scales. Code is available at https://github.com/ali-vilab/TTS-VAR.
>
---
#### [new 081] Registration beyond Points: General Affine Subspace Alignment via Geodesic Distance on Grassmann Manifold
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉中的点云配准任务，旨在解决仿射子空间对齐问题。现有方法无法将刚体变换参数融入距离度量，限制了在配准中的应用。本文首次推导出可优化的成本函数，基于Grassmann流形上的基变换直接最小化测地距离，实现全局最优解，并提出基于BnB的求解器，提升了配准效果和收敛性。**

- **链接: [http://arxiv.org/pdf/2507.17998v1](http://arxiv.org/pdf/2507.17998v1)**

> **作者:** Jaeho Shin; Hyeonjae Gil; Junwoo Jang; Maani Ghaffari; Ayoung Kim
>
> **摘要:** Affine Grassmannian has been favored for expressing proximity between lines and planes due to its theoretical exactness in measuring distances among features. Despite this advantage, the existing method can only measure the proximity without yielding the distance as an explicit function of rigid body transformation. Thus, an optimizable distance function on the manifold has remained underdeveloped, stifling its application in registration problems. This paper is the first to explicitly derive an optimizable cost function between two Grassmannian features with respect to rigid body transformation ($\mathbf{R}$ and $\mathbf{t}$). Specifically, we present a rigorous mathematical proof demonstrating that the bases of high-dimensional linear subspaces can serve as an explicit representation of the cost. Finally, we propose an optimizable cost function based on the transformed bases that can be applied to the registration problem of any affine subspace. Compared to vector parameter-based approaches, our method is able to find a globally optimal solution by directly minimizing the geodesic distance which is agnostic to representation ambiguity. The resulting cost function and its extension to the inlier-set maximizing \ac{BnB} solver have been demonstrated to improve the convergence of existing solutions or outperform them in various computer vision tasks. The code is available on https://github.com/joomeok/GrassmannRegistration.
>
---
#### [new 082] COT-AD: Cotton Analysis Dataset
- **分类: cs.CV; I.4.9; I.5.4; H.2.8**

- **简介: 该论文属于农业图像数据集构建任务，旨在解决棉花作物分析中缺乏专业数据集的问题。论文工作是创建了COT-AD数据集，包含25,000张图像，涵盖棉花生长周期、病害、杂草等标注信息，支持多种计算机视觉任务，推动棉花种植的智能化管理。**

- **链接: [http://arxiv.org/pdf/2507.18532v1](http://arxiv.org/pdf/2507.18532v1)**

> **作者:** Akbar Ali; Mahek Vyas; Soumyaratna Debnath; Chanda Grover Kamra; Jaidev Sanjay Khalane; Reuben Shibu Devanesan; Indra Deep Mastan; Subramanian Sankaranarayanan; Pankaj Khanna; Shanmuganathan Raman
>
> **备注:** Dataset publicly available at: https://ieee-dataport.org/documents/cot-adcotton-analysis-dataset. Accepted to IEEE International Conference on Image Processing (ICIP) 2025
>
> **摘要:** This paper presents COT-AD, a comprehensive Dataset designed to enhance cotton crop analysis through computer vision. Comprising over 25,000 images captured throughout the cotton growth cycle, with 5,000 annotated images, COT-AD includes aerial imagery for field-scale detection and segmentation and high-resolution DSLR images documenting key diseases. The annotations cover pest and disease recognition, vegetation, and weed analysis, addressing a critical gap in cotton-specific agricultural datasets. COT-AD supports tasks such as classification, segmentation, image restoration, enhancement, deep generative model-based cotton crop synthesis, and early disease management, advancing data-driven crop management
>
---
#### [new 083] Dissecting the Dental Lung Cancer Axis via Mendelian Randomization and Mediation Analysis
- **分类: cs.CV**

- **简介: 该论文旨在通过孟德尔随机化和中介分析，探讨牙科疾病（龋齿和牙周炎）与肺癌之间的因果关系及肺功能的中介作用。研究发现龋齿显著增加肺癌风险，部分由肺功能下降中介，而牙周炎无显著影响，提示龋齿与肺癌存在因果关联。**

- **链接: [http://arxiv.org/pdf/2507.18287v1](http://arxiv.org/pdf/2507.18287v1)**

> **作者:** Wenran Zhang; Huihuan Luo; Linda Wei; Ping Nie; Yiqun Wu; Dedong Yu
>
> **摘要:** Periodontitis and dental caries are common oral diseases affecting billions globally. While observational studies suggest links between these conditions and lung cancer, causality remains uncertain. This study used two sample Mendelian randomization (MR) to explore causal relationships between dental traits (periodontitis, dental caries) and lung cancer subtypes, and to assess mediation by pulmonary function. Genetic instruments were derived from the largest available genome wide association studies, including data from 487,823 dental caries and 506,594 periodontitis cases, as well as lung cancer data from the Transdisciplinary Research of Cancer in Lung consortium. Inverse variance weighting was the main analytical method; lung function mediation was assessed using the delta method. The results showed a significant positive causal effect of dental caries on overall lung cancer and its subtypes. Specifically, a one standard deviation increase in dental caries incidence was associated with a 188.0% higher risk of squamous cell lung carcinoma (OR = 2.880, 95% CI = 1.236--6.713, p = 0.014), partially mediated by declines in forced vital capacity (FVC) and forced expiratory volume in one second (FEV1), accounting for 5.124% and 5.890% of the total effect. No causal effect was found for periodontitis. These findings highlight a causal role of dental caries in lung cancer risk and support integrating dental care and pulmonary function monitoring into cancer prevention strategies.
>
---
#### [new 084] Detail++: Training-Free Detail Enhancer for Text-to-Image Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成任务，旨在解决复杂提示下多主体生成效果差的问题。提出Detail++框架，通过渐进式细节注入策略，分阶段生成并优化细节，结合注意力机制与中心对齐损失，提升生成质量与属性一致性。**

- **链接: [http://arxiv.org/pdf/2507.17853v1](http://arxiv.org/pdf/2507.17853v1)**

> **作者:** Lifeng Chen; Jiner Wang; Zihao Pan; Beier Zhu; Xiaofeng Yang; Chi Zhang
>
> **摘要:** Recent advances in text-to-image (T2I) generation have led to impressive visual results. However, these models still face significant challenges when handling complex prompt, particularly those involving multiple subjects with distinct attributes. Inspired by the human drawing process, which first outlines the composition and then incrementally adds details, we propose Detail++, a training-free framework that introduces a novel Progressive Detail Injection (PDI) strategy to address this limitation. Specifically, we decompose a complex prompt into a sequence of simplified sub-prompts, guiding the generation process in stages. This staged generation leverages the inherent layout-controlling capacity of self-attention to first ensure global composition, followed by precise refinement. To achieve accurate binding between attributes and corresponding subjects, we exploit cross-attention mechanisms and further introduce a Centroid Alignment Loss at test time to reduce binding noise and enhance attribute consistency. Extensive experiments on T2I-CompBench and a newly constructed style composition benchmark demonstrate that Detail++ significantly outperforms existing methods, particularly in scenarios involving multiple objects and complex stylistic conditions.
>
---
#### [new 085] BokehDiff: Neural Lens Blur with One-Step Diffusion
- **分类: cs.CV**

- **简介: 该论文属于图像处理任务，旨在解决现有镜头模糊渲染方法在深度不连续处易产生伪影的问题。作者提出BokehDiff，结合扩散模型与物理启发的注意力机制，实现高质量、逼真的镜头模糊效果，并通过合成数据增强策略提升模型训练效果。**

- **链接: [http://arxiv.org/pdf/2507.18060v1](http://arxiv.org/pdf/2507.18060v1)**

> **作者:** Chengxuan Zhu; Qingnan Fan; Qi Zhang; Jinwei Chen; Huaqi Zhang; Chao Xu; Boxin Shi
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** We introduce BokehDiff, a novel lens blur rendering method that achieves physically accurate and visually appealing outcomes, with the help of generative diffusion prior. Previous methods are bounded by the accuracy of depth estimation, generating artifacts in depth discontinuities. Our method employs a physics-inspired self-attention module that aligns with the image formation process, incorporating depth-dependent circle of confusion constraint and self-occlusion effects. We adapt the diffusion model to the one-step inference scheme without introducing additional noise, and achieve results of high quality and fidelity. To address the lack of scalable paired data, we propose to synthesize photorealistic foregrounds with transparency with diffusion models, balancing authenticity and scene diversity.
>
---
#### [new 086] OPEN: A Benchmark Dataset and Baseline for Older Adult Patient Engagement Recognition in Virtual Rehabilitation Learning Environments
- **分类: cs.CV**

- **简介: 该论文属于人工智能在医疗康复领域的应用任务，旨在解决虚拟康复环境中老年人参与度自动识别的问题。为填补相关数据集的空白，作者构建了名为OPEN的新型数据集，并基于此训练模型，实现了最高81%准确率的参与度识别。**

- **链接: [http://arxiv.org/pdf/2507.17959v1](http://arxiv.org/pdf/2507.17959v1)**

> **作者:** Ali Abedi; Sadaf Safa; Tracey J. F. Colella; Shehroz S. Khan
>
> **备注:** 14 pages, 3 figures, 7 tables
>
> **摘要:** Engagement in virtual learning is essential for participant satisfaction, performance, and adherence, particularly in online education and virtual rehabilitation, where interactive communication plays a key role. Yet, accurately measuring engagement in virtual group settings remains a challenge. There is increasing interest in using artificial intelligence (AI) for large-scale, real-world, automated engagement recognition. While engagement has been widely studied in younger academic populations, research and datasets focused on older adults in virtual and telehealth learning settings remain limited. Existing methods often neglect contextual relevance and the longitudinal nature of engagement across sessions. This paper introduces OPEN (Older adult Patient ENgagement), a novel dataset supporting AI-driven engagement recognition. It was collected from eleven older adults participating in weekly virtual group learning sessions over six weeks as part of cardiac rehabilitation, producing over 35 hours of data, making it the largest dataset of its kind. To protect privacy, raw video is withheld; instead, the released data include facial, hand, and body joint landmarks, along with affective and behavioral features extracted from video. Annotations include binary engagement states, affective and behavioral labels, and context-type indicators, such as whether the instructor addressed the group or an individual. The dataset offers versions with 5-, 10-, 30-second, and variable-length samples. To demonstrate utility, multiple machine learning and deep learning models were trained, achieving engagement recognition accuracy of up to 81 percent. OPEN provides a scalable foundation for personalized engagement modeling in aging populations and contributes to broader engagement recognition research.
>
---
#### [new 087] Datasets and Recipes for Video Temporal Grounding via Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频时序定位（VTG）任务，旨在通过自然语言查询定位视频中的时间片段。现有方法存在时间感知不足和泛化能力差的问题。论文提出一种结合监督微调和难度控制强化学习的两阶段训练框架，提升模型准确性和鲁棒性，并通过释放数据、模型和代码推动研究与应用。**

- **链接: [http://arxiv.org/pdf/2507.18100v1](http://arxiv.org/pdf/2507.18100v1)**

> **作者:** Ruizhe Chen; Zhiting Fan; Tianze Luo; Heqing Zou; Zhaopeng Feng; Guiyang Xie; Hansheng Zhang; Zhuochen Wang; Zuozhu Liu; Huaijian Zhang
>
> **摘要:** Video Temporal Grounding (VTG) aims to localize relevant temporal segments in videos given natural language queries. Despite recent progress with large vision-language models (LVLMs) and instruction-tuning, existing approaches often suffer from limited temporal awareness and poor generalization. In this work, we introduce a two-stage training framework that integrates supervised fine-tuning with reinforcement learning (RL) to improve both the accuracy and robustness of VTG models. Our approach first leverages high-quality curated cold start data for SFT initialization, followed by difficulty-controlled RL to further enhance temporal localization and reasoning abilities. Comprehensive experiments on multiple VTG benchmarks demonstrate that our method consistently outperforms existing models, particularly in challenging and open-domain scenarios. We conduct an in-depth analysis of training strategies and dataset curation, highlighting the importance of both high-quality cold start data and difficulty-controlled RL. To facilitate further research and industrial adoption, we release all intermediate datasets, models, and code to the community.
>
---
#### [new 088] Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis
- **分类: cs.CV**

- **简介: 该论文属于图像和视频生成任务，旨在解决扩散模型蒸馏中的模式坍塌问题。作者提出了一种基于对抗分布匹配（ADM）的蒸馏框架，并结合对抗蒸馏与混合判别器，形成DMDX方法，有效提升了生成效率与质量。**

- **链接: [http://arxiv.org/pdf/2507.18569v1](http://arxiv.org/pdf/2507.18569v1)**

> **作者:** Yanzuo Lu; Yuxi Ren; Xin Xia; Shanchuan Lin; Xing Wang; Xuefeng Xiao; Andy J. Ma; Xiaohua Xie; Jian-Huang Lai
>
> **备注:** Accepted by ICCV 2025 (Highlight)
>
> **摘要:** Distribution Matching Distillation (DMD) is a promising score distillation technique that compresses pre-trained teacher diffusion models into efficient one-step or multi-step student generators. Nevertheless, its reliance on the reverse Kullback-Leibler (KL) divergence minimization potentially induces mode collapse (or mode-seeking) in certain applications. To circumvent this inherent drawback, we propose Adversarial Distribution Matching (ADM), a novel framework that leverages diffusion-based discriminators to align the latent predictions between real and fake score estimators for score distillation in an adversarial manner. In the context of extremely challenging one-step distillation, we further improve the pre-trained generator by adversarial distillation with hybrid discriminators in both latent and pixel spaces. Different from the mean squared error used in DMD2 pre-training, our method incorporates the distributional loss on ODE pairs collected from the teacher model, and thus providing a better initialization for score distillation fine-tuning in the next stage. By combining the adversarial distillation pre-training with ADM fine-tuning into a unified pipeline termed DMDX, our proposed method achieves superior one-step performance on SDXL compared to DMD2 while consuming less GPU time. Additional experiments that apply multi-step ADM distillation on SD3-Medium, SD3.5-Large, and CogVideoX set a new benchmark towards efficient image and video synthesis.
>
---
#### [new 089] Unsupervised Domain Adaptation for 3D LiDAR Semantic Segmentation Using Contrastive Learning and Multi-Model Pseudo Labeling
- **分类: cs.CV**

- **简介: 该论文属于3D LiDAR语义分割任务，旨在解决跨域数据分布差异导致的性能下降问题。通过自监督对比学习和多模型伪标签策略，实现无需目标域标注的域适应方法，提升了分割准确性。**

- **链接: [http://arxiv.org/pdf/2507.18176v1](http://arxiv.org/pdf/2507.18176v1)**

> **作者:** Abhishek Kaushik; Norbert Haala; Uwe Soergel
>
> **摘要:** Addressing performance degradation in 3D LiDAR semantic segmentation due to domain shifts (e.g., sensor type, geographical location) is crucial for autonomous systems, yet manual annotation of target data is prohibitive. This study addresses the challenge using Unsupervised Domain Adaptation (UDA) and introduces a novel two-stage framework to tackle it. Initially, unsupervised contrastive learning at the segment level is used to pre-train a backbone network, enabling it to learn robust, domain-invariant features without labels. Subsequently, a multi-model pseudo-labeling strategy is introduced, utilizing an ensemble of diverse state-of-the-art architectures (including projection, voxel, hybrid, and cylinder-based methods). Predictions from these models are aggregated via hard voting to generate high-quality, refined pseudo-labels for the unlabeled target domain, mitigating single-model biases. The contrastively pre-trained network is then fine-tuned using these robust pseudo-labels. Experiments adapting from SemanticKITTI to unlabeled target datasets (SemanticPOSS, SemanticSlamantic) demonstrate significant improvements in segmentation accuracy compared to direct transfer and single-model UDA approaches. These results highlight the effectiveness of combining contrastive pre-training with refined ensemble pseudo-labeling for bridging complex domain gaps without requiring target domain annotations.
>
---
#### [new 090] DiNAT-IR: Exploring Dilated Neighborhood Attention for High-Quality Image Restoration
- **分类: cs.CV**

- **简介: 本文属于图像恢复任务，旨在解决传统注意力机制在高分辨率图像处理中的效率与质量权衡问题。作者提出DiNAT-IR，结合膨胀邻域注意力与通道感知模块，在保持局部精度的同时增强全局上下文理解，实现高效高质量的图像去模糊等恢复任务。**

- **链接: [http://arxiv.org/pdf/2507.17892v1](http://arxiv.org/pdf/2507.17892v1)**

> **作者:** Hanzhou Liu; Binghan Li; Chengkai Liu; Mi Lu
>
> **摘要:** Transformers, with their self-attention mechanisms for modeling long-range dependencies, have become a dominant paradigm in image restoration tasks. However, the high computational cost of self-attention limits scalability to high-resolution images, making efficiency-quality trade-offs a key research focus. To address this, Restormer employs channel-wise self-attention, which computes attention across channels instead of spatial dimensions. While effective, this approach may overlook localized artifacts that are crucial for high-quality image restoration. To bridge this gap, we explore Dilated Neighborhood Attention (DiNA) as a promising alternative, inspired by its success in high-level vision tasks. DiNA balances global context and local precision by integrating sliding-window attention with mixed dilation factors, effectively expanding the receptive field without excessive overhead. However, our preliminary experiments indicate that directly applying this global-local design to the classic deblurring task hinders accurate visual restoration, primarily due to the constrained global context understanding within local attention. To address this, we introduce a channel-aware module that complements local attention, effectively integrating global context without sacrificing pixel-level precision. The proposed DiNAT-IR, a Transformer-based architecture specifically designed for image restoration, achieves competitive results across multiple benchmarks, offering a high-quality solution for diverse low-level computer vision problems.
>
---
#### [new 091] ChronoSelect: Robust Learning with Noisy Labels via Dynamics Temporal Memory
- **分类: cs.LG; cs.CV**

- **简介: 论文提出ChronoSelect，属于学习含噪标签任务，旨在解决深度模型因记忆噪声标签导致性能下降的问题。通过四阶段记忆架构与滑动更新机制，压缩预测历史并分析时间轨迹，实现样本的准确划分与噪声抑制，提升模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.18183v1](http://arxiv.org/pdf/2507.18183v1)**

> **作者:** Jianchao Wang; Qingfeng Li; Pengcheng Zheng; Xiaorong Pu; Yazhou Ren
>
> **摘要:** Training deep neural networks on real-world datasets is often hampered by the presence of noisy labels, which can be memorized by over-parameterized models, leading to significant degradation in generalization performance. While existing methods for learning with noisy labels (LNL) have made considerable progress, they fundamentally suffer from static snapshot evaluations and fail to leverage the rich temporal dynamics of learning evolution. In this paper, we propose ChronoSelect (chrono denoting its temporal nature), a novel framework featuring an innovative four-stage memory architecture that compresses prediction history into compact temporal distributions. Our unique sliding update mechanism with controlled decay maintains only four dynamic memory units per sample, progressively emphasizing recent patterns while retaining essential historical knowledge. This enables precise three-way sample partitioning into clean, boundary, and noisy subsets through temporal trajectory analysis and dual-branch consistency. Theoretical guarantees prove the mechanism's convergence and stability under noisy conditions. Extensive experiments demonstrate ChronoSelect's state-of-the-art performance across synthetic and real-world benchmarks.
>
---
#### [new 092] UniSegDiff: Boosting Unified Lesion Segmentation via a Staged Diffusion Model
- **分类: eess.IV; cs.CV**

- **简介: 论文提出UniSegDiff，用于医学图像中病灶分割任务。解决现有扩散模型训练和推理策略导致的注意力分布不均问题。通过分阶段训练和推理方法，动态调整预测目标，提高各时间步注意力，实现多模态、多器官统一病灶分割。**

- **链接: [http://arxiv.org/pdf/2507.18362v1](http://arxiv.org/pdf/2507.18362v1)**

> **作者:** Yilong Hu; Shijie Chang; Lihe Zhang; Feng Tian; Weibing Sun; Huchuan Lu
>
> **备注:** MICCAI2025
>
> **摘要:** The Diffusion Probabilistic Model (DPM) has demonstrated remarkable performance across a variety of generative tasks. The inherent randomness in diffusion models helps address issues such as blurring at the edges of medical images and labels, positioning Diffusion Probabilistic Models (DPMs) as a promising approach for lesion segmentation. However, we find that the current training and inference strategies of diffusion models result in an uneven distribution of attention across different timesteps, leading to longer training times and suboptimal solutions. To this end, we propose UniSegDiff, a novel diffusion model framework designed to address lesion segmentation in a unified manner across multiple modalities and organs. This framework introduces a staged training and inference approach, dynamically adjusting the prediction targets at different stages, forcing the model to maintain high attention across all timesteps, and achieves unified lesion segmentation through pre-training the feature extraction network for segmentation. We evaluate performance on six different organs across various imaging modalities. Comprehensive experimental results demonstrate that UniSegDiff significantly outperforms previous state-of-the-art (SOTA) approaches. The code is available at https://github.com/HUYILONG-Z/UniSegDiff.
>
---
#### [new 093] Caching Techniques for Reducing the Communication Cost of Federated Learning in IoT Environments
- **分类: cs.DC; cs.AI; cs.CV; cs.LG**

- **简介: 该论文研究物联网环境中联邦学习的通信开销问题，提出使用FIFO、LRU和基于优先级的缓存策略，选择性传输重要模型更新，以减少带宽消耗并保持模型精度，适用于智能城市和医疗等边缘计算场景。**

- **链接: [http://arxiv.org/pdf/2507.17772v1](http://arxiv.org/pdf/2507.17772v1)**

> **作者:** Ahmad Alhonainy; Praveen Rao
>
> **备注:** Journal
>
> **摘要:** Federated Learning (FL) allows multiple distributed devices to jointly train a shared model without centralizing data, but communication cost remains a major bottleneck, especially in resource-constrained environments. This paper introduces caching strategies - FIFO, LRU, and Priority-Based - to reduce unnecessary model update transmissions. By selectively forwarding significant updates, our approach lowers bandwidth usage while maintaining model accuracy. Experiments on CIFAR-10 and medical datasets show reduced communication with minimal accuracy loss. Results confirm that intelligent caching improves scalability, memory efficiency, and supports reliable FL in edge IoT networks, making it practical for deployment in smart cities, healthcare, and other latency-sensitive applications.
>
---
#### [new 094] DiagR1: A Vision-Language Model Trained via Reinforcement Learning for Digestive Pathology Diagnosis
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决胃肠病理诊断中数据质量差和推理不透明的问题。作者构建了大规模胃肠病理数据集，提出提示论证策略，并结合监督微调与GRPO优化模型推理和生成质量。实验表明，该方法在临床相关性、结构完整性和诊断准确性方面均优于现有模型。**

- **链接: [http://arxiv.org/pdf/2507.18433v1](http://arxiv.org/pdf/2507.18433v1)**

> **作者:** Minxi Ouyang; Lianghui Zhu; Yaqing Bao; Qiang Huang; Jingli Ouyang; Tian Guan; Xitong Ling; Jiawen Li; Song Duan; Wenbin Dai; Li Zheng; Xuemei Zhang; Yonghong He
>
> **摘要:** Multimodal large models have shown great potential in automating pathology image analysis. However, current multimodal models for gastrointestinal pathology are constrained by both data quality and reasoning transparency: pervasive noise and incomplete annotations in public datasets predispose vision language models to factual hallucinations when generating diagnostic text, while the absence of explicit intermediate reasoning chains renders the outputs difficult to audit and thus less trustworthy in clinical practice. To address these issues, we construct a large scale gastrointestinal pathology dataset containing both microscopic descriptions and diagnostic conclusions, and propose a prompt argumentation strategy that incorporates lesion classification and anatomical site information. This design guides the model to better capture image specific features and maintain semantic consistency in generation. Furthermore, we employ a post training pipeline that combines supervised fine tuning with Group Relative Policy Optimization (GRPO) to improve reasoning quality and output structure. Experimental results on real world pathology report generation tasks demonstrate that our approach significantly outperforms state of the art open source and proprietary baselines in terms of generation quality, structural completeness, and clinical relevance. Our solution outperforms state of the art models with 18.7% higher clinical relevance, 32.4% improved structural completeness, and 41.2% fewer diagnostic errors, demonstrating superior accuracy and clinical utility compared to existing solutions.
>
---
#### [new 095] Multimodal Recurrent Ensembles for Predicting Brain Responses to Naturalistic Movies (Algonauts 2025)
- **分类: q-bio.NC; cs.CV; cs.LG**

- **简介: 该论文属于脑响应预测任务，旨在解决自然电影刺激下多模态信息（视觉、听觉、语义）融合建模问题。作者构建了一个层次化多模态循环集成模型，结合视频、音频和语言嵌入，通过双向RNN提取时序特征，融合后输入第二层循环网络，最终输出皮层分区响应。模型在Algonauts 2025挑战赛中表现优异，取得第三名，皮尔逊相关系数达0.2094。**

- **链接: [http://arxiv.org/pdf/2507.17897v1](http://arxiv.org/pdf/2507.17897v1)**

> **作者:** Semih Eren; Deniz Kucukahmetler; Nico Scherf
>
> **备注:** 8 pages, 2 figures, 1 table. Invited report, CCN 2025 Algonauts Project session (3rd-place team). Code: https://github.com/erensemih/Algonauts2025_ModalityRNN
>
> **摘要:** Accurately predicting distributed cortical responses to naturalistic stimuli requires models that integrate visual, auditory and semantic information over time. We present a hierarchical multimodal recurrent ensemble that maps pretrained video, audio, and language embeddings to fMRI time series recorded while four subjects watched almost 80 hours of movies provided by the Algonauts 2025 challenge. Modality-specific bidirectional RNNs encode temporal dynamics; their hidden states are fused and passed to a second recurrent layer, and lightweight subject-specific heads output responses for 1000 cortical parcels. Training relies on a composite MSE-correlation loss and a curriculum that gradually shifts emphasis from early sensory to late association regions. Averaging 100 model variants further boosts robustness. The resulting system ranked third on the competition leaderboard, achieving an overall Pearson r = 0.2094 and the highest single-parcel peak score (mean r = 0.63) among all participants, with particularly strong gains for the most challenging subject (Subject 5). The approach establishes a simple, extensible baseline for future multimodal brain-encoding benchmarks.
>
---
#### [new 096] U-Net Based Healthy 3D Brain Tissue Inpainting
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决3D脑组织MRI图像中缺失或受损区域的健康组织修复问题。作者采用基于U-Net的模型，通过数据增强策略训练，实现对脑组织图像的高质量重建，并在相关数据集上取得了优异性能。**

- **链接: [http://arxiv.org/pdf/2507.18126v1](http://arxiv.org/pdf/2507.18126v1)**

> **作者:** Juexin Zhang; Ying Weng; Ke Chen
>
> **备注:** Accepted by the International Brain Tumor Segmentation (BraTS) challenge organized at MICCAI 2024 conference. Included 7 pages, 2 figures
>
> **摘要:** This paper introduces a novel approach to synthesize healthy 3D brain tissue from masked input images, specifically focusing on the task of 'ASNR-MICCAI BraTS Local Synthesis of Tissue via Inpainting'. Our proposed method employs a U-Net-based architecture, which is designed to effectively reconstruct the missing or corrupted regions of brain MRI scans. To enhance our model's generalization capabilities and robustness, we implement a comprehensive data augmentation strategy that involves randomly masking healthy images during training. Our model is trained on the BraTS-Local-Inpainting dataset and demonstrates the exceptional performance in recovering healthy brain tissue. The evaluation metrics employed, including Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR), and Mean Squared Error (MSE), consistently yields impressive results. On the BraTS-Local-Inpainting validation set, our model achieved an SSIM score of 0.841, a PSNR score of 23.257, and an MSE score of 0.007. Notably, these evaluation metrics exhibit relatively low standard deviations, i.e., 0.103 for SSIM score, 4.213 for PSNR score and 0.007 for MSE score, which indicates that our model's reliability and consistency across various input scenarios. Our method also secured first place in the challenge.
>
---
#### [new 097] On the Performance of Concept Probing: The Influence of the Data (Extended Version)
- **分类: cs.AI; cs.CV; cs.LG; cs.NE**

- **简介: 该论文属于模型解释任务，旨在解决概念探测中数据影响的问题。它研究了用于训练探测模型的数据对性能的影响，并发布了两个常用数据集的概念标签。**

- **链接: [http://arxiv.org/pdf/2507.18550v1](http://arxiv.org/pdf/2507.18550v1)**

> **作者:** Manuel de Sousa Ribeiro; Afonso Leote; João Leite
>
> **备注:** Extended version of the paper published in Proceedings of the European Conference on Artificial Intelligence (ECAI 2025)
>
> **摘要:** Concept probing has recently garnered increasing interest as a way to help interpret artificial neural networks, dealing both with their typically large size and their subsymbolic nature, which ultimately renders them unfeasible for direct human interpretation. Concept probing works by training additional classifiers to map the internal representations of a model into human-defined concepts of interest, thus allowing humans to peek inside artificial neural networks. Research on concept probing has mainly focused on the model being probed or the probing model itself, paying limited attention to the data required to train such probing models. In this paper, we address this gap. Focusing on concept probing in the context of image classification tasks, we investigate the effect of the data used to train probing models on their performance. We also make available concept labels for two widely used datasets.
>
---
#### [new 098] GeoAvatar: Adaptive Geometrical Gaussian Splatting for 3D Head Avatar
- **分类: cs.GR; cs.CV; cs.LG**

- **简介: 该论文属于3D头像生成任务，旨在解决身份保持与新姿态、表情生成之间的平衡问题。作者提出GeoAvatar，通过自适应几何高斯点绘制，结合预分配策略和嘴部结构优化，提高重建与动画质量，并发布新数据集DynamicFace。**

- **链接: [http://arxiv.org/pdf/2507.18155v1](http://arxiv.org/pdf/2507.18155v1)**

> **作者:** SeungJun Moon; Hah Min Lew; Seungeun Lee; Ji-Su Kang; Gyeong-Moon Park
>
> **备注:** ICCV 2025, Project page: https://hahminlew.github.io/geoavatar/
>
> **摘要:** Despite recent progress in 3D head avatar generation, balancing identity preservation, i.e., reconstruction, with novel poses and expressions, i.e., animation, remains a challenge. Existing methods struggle to adapt Gaussians to varying geometrical deviations across facial regions, resulting in suboptimal quality. To address this, we propose GeoAvatar, a framework for adaptive geometrical Gaussian Splatting. GeoAvatar leverages Adaptive Pre-allocation Stage (APS), an unsupervised method that segments Gaussians into rigid and flexible sets for adaptive offset regularization. Then, based on mouth anatomy and dynamics, we introduce a novel mouth structure and the part-wise deformation strategy to enhance the animation fidelity of the mouth. Finally, we propose a regularization loss for precise rigging between Gaussians and 3DMM faces. Moreover, we release DynamicFace, a video dataset with highly expressive facial motions. Extensive experiments show the superiority of GeoAvatar compared to state-of-the-art methods in reconstruction and novel animation scenarios.
>
---
#### [new 099] GrAInS: Gradient-based Attribution for Inference-Time Steering of LLMs and VLMs
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 论文提出GrAInS，一种推理时 steering 方法，用于调整大语言模型和多模态模型的行为。它通过基于梯度的归因识别关键输入 token，构建方向性 steering 向量，实现对模型输出的细粒度控制。旨在解决现有方法忽视 token 级因果影响和梯度信息的问题，提升模型在问答、减少幻觉和对齐任务中的表现。**

- **链接: [http://arxiv.org/pdf/2507.18043v1](http://arxiv.org/pdf/2507.18043v1)**

> **作者:** Duy Nguyen; Archiki Prasad; Elias Stengel-Eskin; Mohit Bansal
>
> **备注:** 21 pages. Code: https://github.com/duykhuongnguyen/GrAInS
>
> **摘要:** Inference-time steering methods offer a lightweight alternative to fine-tuning large language models (LLMs) and vision-language models (VLMs) by modifying internal activations at test time without updating model weights. However, most existing approaches rely on fixed, global intervention vectors, overlook the causal influence of individual input tokens, and fail to leverage informative gradients from the model's logits, particularly in multimodal settings where visual and textual inputs contribute unevenly. To address these limitations, we introduce GrAInS, an inference-time steering approach that operates across both language-only and vision-language models and tasks. GrAInS uses contrastive, gradient-based attribution via Integrated Gradients to identify the top-k most influential tokens, both positively and negatively attributed based on their contribution to preferred versus dispreferred outputs. These tokens are then used to construct directional steering vectors that capture semantic shifts from undesirable to desirable behavior. During inference, GrAInS adjusts hidden activations at transformer layers guided by token-level attribution signals, and normalizes activations to preserve representational scale. This enables fine-grained, interpretable, and modular control over model behavior, without retraining or auxiliary supervision. Empirically, GrAInS consistently outperforms both fine-tuning and existing steering baselines: it achieves a 13.22% accuracy gain on TruthfulQA using Llama-3.1-8B, reduces hallucination rates on MMHal-Bench from 0.624 to 0.514 with LLaVA-1.6-7B, and improves alignment win rates on SPA-VL by 8.11%, all while preserving the model's fluency and general capabilities.
>
---
#### [new 100] Towards Robust Foundation Models for Digital Pathology
- **分类: eess.IV; cs.AI; cs.CV; cs.LG; q-bio.QM**

- **简介: 该论文属于医学图像分析任务，旨在解决病理学基础模型对非生物特征（如技术差异）的敏感性问题。工作包括构建评估基准PathoROB，提出鲁棒性度量指标，并验证多个模型的鲁棒性缺陷，强调鲁棒性在临床部署中的重要性。**

- **链接: [http://arxiv.org/pdf/2507.17845v1](http://arxiv.org/pdf/2507.17845v1)**

> **作者:** Jonah Kömen; Edwin D. de Jong; Julius Hense; Hannah Marienwald; Jonas Dippel; Philip Naumann; Eric Marcus; Lukas Ruff; Maximilian Alber; Jonas Teuwen; Frederick Klauschen; Klaus-Robert Müller
>
> **摘要:** Biomedical Foundation Models (FMs) are rapidly transforming AI-enabled healthcare research and entering clinical validation. However, their susceptibility to learning non-biological technical features -- including variations in surgical/endoscopic techniques, laboratory procedures, and scanner hardware -- poses risks for clinical deployment. We present the first systematic investigation of pathology FM robustness to non-biological features. Our work (i) introduces measures to quantify FM robustness, (ii) demonstrates the consequences of limited robustness, and (iii) proposes a framework for FM robustification to mitigate these issues. Specifically, we developed PathoROB, a robustness benchmark with three novel metrics, including the robustness index, and four datasets covering 28 biological classes from 34 medical centers. Our experiments reveal robustness deficits across all 20 evaluated FMs, and substantial robustness differences between them. We found that non-robust FM representations can cause major diagnostic downstream errors and clinical blunders that prevent safe clinical adoption. Using more robust FMs and post-hoc robustification considerably reduced (but did not yet eliminate) the risk of such errors. This work establishes that robustness evaluation is essential for validating pathology FMs before clinical adoption and demonstrates that future FM development must integrate robustness as a core design principle. PathoROB provides a blueprint for assessing robustness across biomedical domains, guiding FM improvement efforts towards more robust, representative, and clinically deployable AI systems that prioritize biological information over technical artifacts.
>
---
#### [new 101] Improving Multislice Electron Ptychography with a Generative Prior
- **分类: eess.IV; cond-mat.mtrl-sci; cs.CV; physics.optics**

- **简介: 该论文属于电子断层成像任务，旨在解决多层电子叠层成像（MEP）中重建速度慢、效果不佳的问题。作者提出了MEP-Diffusion方法，通过在现有迭代算法中引入基于扩散模型的先验知识，显著提升了3D原子结构重建质量，SSIM指标提高了90.50%。**

- **链接: [http://arxiv.org/pdf/2507.17800v1](http://arxiv.org/pdf/2507.17800v1)**

> **作者:** Christian K. Belardi; Chia-Hao Lee; Yingheng Wang; Justin Lovelace; Kilian Q. Weinberger; David A. Muller; Carla P. Gomes
>
> **备注:** 16 pages, 10 figures, 5 tables
>
> **摘要:** Multislice electron ptychography (MEP) is an inverse imaging technique that computationally reconstructs the highest-resolution images of atomic crystal structures from diffraction patterns. Available algorithms often solve this inverse problem iteratively but are both time consuming and produce suboptimal solutions due to their ill-posed nature. We develop MEP-Diffusion, a diffusion model trained on a large database of crystal structures specifically for MEP to augment existing iterative solvers. MEP-Diffusion is easily integrated as a generative prior into existing reconstruction methods via Diffusion Posterior Sampling (DPS). We find that this hybrid approach greatly enhances the quality of the reconstructed 3D volumes, achieving a 90.50% improvement in SSIM over existing methods.
>
---
#### [new 102] Zero-Shot Dynamic Concept Personalization with Grid-Based LoRA
- **分类: cs.GR; cs.CV; cs.LG**

- **简介: 该论文属于文本到视频生成任务，旨在解决动态概念个性化生成中的泛化性与效率问题。现有方法需逐实例微调，限制了扩展性，而该文提出一种完全零样本的动态概念个性化框架，通过基于网格的LoRA适配器实现输入输出的结构化组织，并设计Grid Fill模块完成推理时的布局补全，从而在无需优化的情况下实现跨概念、高质量、时序一致的视频生成。**

- **链接: [http://arxiv.org/pdf/2507.17963v1](http://arxiv.org/pdf/2507.17963v1)**

> **作者:** Rameen Abdal; Or Patashnik; Ekaterina Deyneka; Hao Chen; Aliaksandr Siarohin; Sergey Tulyakov; Daniel Cohen-Or; Kfir Aberman
>
> **备注:** Project Page and Video : https://snap-research.github.io/zero-shot-dynamic-concepts/
>
> **摘要:** Recent advances in text-to-video generation have enabled high-quality synthesis from text and image prompts. While the personalization of dynamic concepts, which capture subject-specific appearance and motion from a single video, is now feasible, most existing methods require per-instance fine-tuning, limiting scalability. We introduce a fully zero-shot framework for dynamic concept personalization in text-to-video models. Our method leverages structured 2x2 video grids that spatially organize input and output pairs, enabling the training of lightweight Grid-LoRA adapters for editing and composition within these grids. At inference, a dedicated Grid Fill module completes partially observed layouts, producing temporally coherent and identity preserving outputs. Once trained, the entire system operates in a single forward pass, generalizing to previously unseen dynamic concepts without any test-time optimization. Extensive experiments demonstrate high-quality and consistent results across a wide range of subjects beyond trained concepts and editing scenarios.
>
---
#### [new 103] Deep Learning for Glioblastoma Morpho-pathological Features Identification: A BraTS-Pathology Challenge Solution
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在利用深度学习识别胶质母细胞瘤的病理特征。论文为解决肿瘤异质性带来的诊断难题，基于 BraTS-Path 数据集微调预训练模型，并在挑战赛中取得第二名。尽管模型敏感度较低，但特异性表现优异。**

- **链接: [http://arxiv.org/pdf/2507.18133v1](http://arxiv.org/pdf/2507.18133v1)**

> **作者:** Juexin Zhang; Ying Weng; Ke Chen
>
> **备注:** Accepted by the International Brain Tumor Segmentation (BraTS) challenge organized at MICCAI 2024 conference
>
> **摘要:** Glioblastoma, a highly aggressive brain tumor with diverse molecular and pathological features, poses a diagnostic challenge due to its heterogeneity. Accurate diagnosis and assessment of this heterogeneity are essential for choosing the right treatment and improving patient outcomes. Traditional methods rely on identifying specific features in tissue samples, but deep learning offers a promising approach for improved glioblastoma diagnosis. In this paper, we present our approach to the BraTS-Path Challenge 2024. We leverage a pre-trained model and fine-tune it on the BraTS-Path training dataset. Our model demonstrates poor performance on the challenging BraTS-Path validation set, as rigorously assessed by the Synapse online platform. The model achieves an accuracy of 0.392229, a recall of 0.392229, and a F1-score of 0.392229, indicating a consistent ability to correctly identify instances under the target condition. Notably, our model exhibits perfect specificity of 0.898704, showing an exceptional capacity to correctly classify negative cases. Moreover, a Matthews Correlation Coefficient (MCC) of 0.255267 is calculated, to signify a limited positive correlation between predicted and actual values and highlight our model's overall predictive power. Our solution also achieves the second place during the testing phase.
>
---
#### [new 104] Parameter-Efficient Fine-Tuning of 3D DDPM for MRI Image Generation Using Tensor Networks
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像生成任务，旨在解决3D扩散模型微调中参数效率低的问题。作者提出TenVOO方法，利用张量网络建模降低3D卷积核参数维度，在脑MRI图像生成中实现高效微调，仅用0.3%参数量即取得优越性能。**

- **链接: [http://arxiv.org/pdf/2507.18112v1](http://arxiv.org/pdf/2507.18112v1)**

> **作者:** Binghua Li; Ziqing Chang; Tong Liang; Chao Li; Toshihisa Tanaka; Shigeki Aoki; Qibin Zhao; Zhe Sun
>
> **摘要:** We address the challenge of parameter-efficient fine-tuning (PEFT) for three-dimensional (3D) U-Net-based denoising diffusion probabilistic models (DDPMs) in magnetic resonance imaging (MRI) image generation. Despite its practical significance, research on parameter-efficient representations of 3D convolution operations remains limited. To bridge this gap, we propose Tensor Volumetric Operator (TenVOO), a novel PEFT method specifically designed for fine-tuning DDPMs with 3D convolutional backbones. Leveraging tensor network modeling, TenVOO represents 3D convolution kernels with lower-dimensional tensors, effectively capturing complex spatial dependencies during fine-tuning with few parameters. We evaluate TenVOO on three downstream brain MRI datasets-ADNI, PPMI, and BraTS2021-by fine-tuning a DDPM pretrained on 59,830 T1-weighted brain MRI scans from the UK Biobank. Our results demonstrate that TenVOO achieves state-of-the-art performance in multi-scale structural similarity index measure (MS-SSIM), outperforming existing approaches in capturing spatial dependencies while requiring only 0.3% of the trainable parameters of the original model. Our code is available at: https://github.com/xiaovhua/tenvoo
>
---
#### [new 105] Benchmarking of Deep Learning Methods for Generic MRI Multi-OrganAbdominal Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决腹部多器官MRI分割中因数据标注困难导致的模型泛化能力不足问题。论文对比了三种先进开源模型，并提出了一种无需真实MRI标注数据的合成训练模型ABDSynth，评估其在多个公开MRI数据集上的性能。**

- **链接: [http://arxiv.org/pdf/2507.17971v1](http://arxiv.org/pdf/2507.17971v1)**

> **作者:** Deepa Krishnaswamy; Cosmin Ciausu; Steve Pieper; Ron Kikinis; Benjamin Billot; Andrey Fedorov
>
> **摘要:** Recent advances in deep learning have led to robust automated tools for segmentation of abdominal computed tomography (CT). Meanwhile, segmentation of magnetic resonance imaging (MRI) is substantially more challenging due to the inherent signal variability and the increased effort required for annotating training datasets. Hence, existing approaches are trained on limited sets of MRI sequences, which might limit their generalizability. To characterize the landscape of MRI abdominal segmentation tools, we present here a comprehensive benchmarking of the three state-of-the-art and open-source models: MRSegmentator, MRISegmentator-Abdomen, and TotalSegmentator MRI. Since these models are trained using labor-intensive manual annotation cycles, we also introduce and evaluate ABDSynth, a SynthSeg-based model purely trained on widely available CT segmentations (no real images). More generally, we assess accuracy and generalizability by leveraging three public datasets (not seen by any of the evaluated methods during their training), which span all major manufacturers, five MRI sequences, as well as a variety of subject conditions, voxel resolutions, and fields-of-view. Our results reveal that MRSegmentator achieves the best performance and is most generalizable. In contrast, ABDSynth yields slightly less accurate results, but its relaxed requirements in training data make it an alternative when the annotation budget is limited. The evaluation code and datasets are given for future benchmarking at https://github.com/deepakri201/AbdoBench, along with inference code and weights for ABDSynth.
>
---
#### [new 106] TCM-Tongue: A Standardized Tongue Image Dataset with Pathological Annotations for AI-Assisted TCM Diagnosis
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像数据集构建任务，旨在解决中医舌诊因缺乏标准化和大规模标注数据而难以发展AI辅助诊断的问题。作者创建了包含6,719张高质量舌部图像的TCM-Tongue数据集，每张图像标注了20种病理症状类别，并支持多种标注格式，适用于AI模型训练与评估，为推动中医舌诊智能化提供了重要基础资源。**

- **链接: [http://arxiv.org/pdf/2507.18288v1](http://arxiv.org/pdf/2507.18288v1)**

> **作者:** Xuebo Jin; Longfei Gao; Anshuo Tong; Zhengyang Chen; Jianlei Kong; Ning Sun; Huijun Ma; Qiang Wang; Yuting Bai; Tingli Su
>
> **备注:** 16 pages, 11 figures, 2 Tables
>
> **摘要:** Traditional Chinese medicine (TCM) tongue diagnosis, while clinically valuable, faces standardization challenges due to subjective interpretation and inconsistent imaging protocols, compounded by the lack of large-scale, annotated datasets for AI development. To address this gap, we present the first specialized dataset for AI-driven TCM tongue diagnosis, comprising 6,719 high-quality images captured under standardized conditions and annotated with 20 pathological symptom categories (averaging 2.54 clinically validated labels per image, all verified by licensed TCM practitioners). The dataset supports multiple annotation formats (COCO, TXT, XML) for broad usability and has been benchmarked using nine deep learning models (YOLOv5/v7/v8 variants, SSD, and MobileNetV2) to demonstrate its utility for AI development. This resource provides a critical foundation for advancing reliable computational tools in TCM, bridging the data shortage that has hindered progress in the field, and facilitating the integration of AI into both research and clinical practice through standardized, high-quality diagnostic data.
>
---
#### [new 107] Diffusion-Assisted Frequency Attention Model for Whole-body Low-field MRI Reconstruction
- **分类: physics.med-ph; cs.CV**

- **简介: 该论文属于医学图像重建任务，旨在解决低信噪比下全身低场磁共振成像的重建难题。作者提出了DFAM模型，结合扩散模型与频域注意力机制，有效提升重建效果，尤其适用于资源有限的临床场景。**

- **链接: [http://arxiv.org/pdf/2507.17764v1](http://arxiv.org/pdf/2507.17764v1)**

> **作者:** Xin Xie; Yu Guan; Zhuoxu Cui; Dong Liang; Qiegen Liu
>
> **备注:** 29 pages,7 figures
>
> **摘要:** By integrating the generative strengths of diffusion models with the representation capabilities of frequency-domain attention, DFAM effectively enhances reconstruction performance under low-SNR condi-tions. Experimental results demonstrate that DFAM consistently outperforms both conventional reconstruction algorithms and recent learning-based approaches. These findings highlight the potential of DFAM as a promising solution to advance low-field MRI reconstruction, particularly in resource-constrained or underdeveloped clinical settings.
>
---
#### [new 108] PS-GS: Gaussian Splatting for Multi-View Photometric Stereo
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于三维重建任务，旨在解决多视角光度立体逆渲染效率与精度问题。作者提出了PS-GS方法，结合高斯点绘与延迟逆渲染，联合优化几何、材质与光照，利用正则化与多视角多光源图像缓解逆渲染病态问题，实现高质量重建。**

- **链接: [http://arxiv.org/pdf/2507.18231v1](http://arxiv.org/pdf/2507.18231v1)**

> **作者:** Yixiao Chen; Bin Liang; Hanzhi Guo; Yongqing Cheng; Jiayi Zhao; Dongdong Weng
>
> **摘要:** Integrating inverse rendering with multi-view photometric stereo (MVPS) yields more accurate 3D reconstructions than the inverse rendering approaches that rely on fixed environment illumination. However, efficient inverse rendering with MVPS remains challenging. To fill this gap, we introduce the Gaussian Splatting for Multi-view Photometric Stereo (PS-GS), which efficiently and jointly estimates the geometry, materials, and lighting of the object that is illuminated by diverse directional lights (multi-light). Our method first reconstructs a standard 2D Gaussian splatting model as the initial geometry. Based on the initialization model, it then proceeds with the deferred inverse rendering by the full rendering equation containing a lighting-computing multi-layer perceptron. During the whole optimization, we regularize the rendered normal maps by the uncalibrated photometric stereo estimated normals. We also propose the 2D Gaussian ray-tracing for single directional light to refine the incident lighting. The regularizations and the use of multi-view and multi-light images mitigate the ill-posed problem of inverse rendering. After optimization, the reconstructed object can be used for novel-view synthesis, relighting, and material and shape editing. Experiments on both synthetic and real datasets demonstrate that our method outperforms prior works in terms of reconstruction accuracy and computational efficiency.
>
---
#### [new 109] Adaptive Articulated Object Manipulation On The Fly with Foundation Model Reasoning and Part Grounding
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决真实场景中多类联结物体操作的泛化问题。现有方法因物体几何多样性和功能差异难以适应。论文提出AdaRPG框架，利用基础模型提取具有局部几何相似性的物体部件，提升功能技能的视觉可操作性泛化，并通过部件功能推理生成高层控制指令，实现跨新联结物体类别的强泛化操作。**

- **链接: [http://arxiv.org/pdf/2507.18276v1](http://arxiv.org/pdf/2507.18276v1)**

> **作者:** Xiaojie Zhang; Yuanfei Wang; Ruihai Wu; Kunqi Xu; Yu Li; Liuyu Xiang; Hao Dong; Zhaofeng He
>
> **备注:** ICCV 2025
>
> **摘要:** Articulated objects pose diverse manipulation challenges for robots. Since their internal structures are not directly observable, robots must adaptively explore and refine actions to generate successful manipulation trajectories. While existing works have attempted cross-category generalization in adaptive articulated object manipulation, two major challenges persist: (1) the geometric diversity of real-world articulated objects complicates visual perception and understanding, and (2) variations in object functions and mechanisms hinder the development of a unified adaptive manipulation strategy. To address these challenges, we propose AdaRPG, a novel framework that leverages foundation models to extract object parts, which exhibit greater local geometric similarity than entire objects, thereby enhancing visual affordance generalization for functional primitive skills. To support this, we construct a part-level affordance annotation dataset to train the affordance model. Additionally, AdaRPG utilizes the common knowledge embedded in foundation models to reason about complex mechanisms and generate high-level control codes that invoke primitive skill functions based on part affordance inference. Simulation and real-world experiments demonstrate AdaRPG's strong generalization ability across novel articulated object categories.
>
---
#### [new 110] Direct Dual-Energy CT Material Decomposition using Model-based Denoising Diffusion Model
- **分类: eess.IV; cs.CV; physics.med-ph; 92C55, 94A08; I.4.5; J.3**

- **简介: 该论文属于医学成像任务，旨在解决双能CT材料分解中因后处理导致的次优结果和硬化伪影问题。工作提出DEcomp-MoD方法，结合模型扩散与深度学习，直接从投影数据分解材料，提升准确性与临床应用潜力。**

- **链接: [http://arxiv.org/pdf/2507.18012v1](http://arxiv.org/pdf/2507.18012v1)**

> **作者:** Hang Xu; Alexandre Bousse; Alessandro Perelli
>
> **备注:** 13 pages, 10 figures, 2 tables
>
> **摘要:** Dual-energy X-ray Computed Tomography (DECT) constitutes an advanced technology which enables automatic decomposition of materials in clinical images without manual segmentation using the dependency of the X-ray linear attenuation with energy. However, most methods perform material decomposition in the image domain as a post-processing step after reconstruction but this procedure does not account for the beam-hardening effect and it results in sub-optimal results. In this work, we propose a deep learning procedure called Dual-Energy Decomposition Model-based Diffusion (DEcomp-MoD) for quantitative material decomposition which directly converts the DECT projection data into material images. The algorithm is based on incorporating the knowledge of the spectral DECT model into the deep learning training loss and combining a score-based denoising diffusion learned prior in the material image domain. Importantly the inference optimization loss takes as inputs directly the sinogram and converts to material images through a model-based conditional diffusion model which guarantees consistency of the results. We evaluate the performance with both quantitative and qualitative estimation of the proposed DEcomp-MoD method on synthetic DECT sinograms from the low-dose AAPM dataset. Finally, we show that DEcomp-MoD outperform state-of-the-art unsupervised score-based model and supervised deep learning networks, with the potential to be deployed for clinical diagnosis.
>
---
#### [new 111] Evaluation of facial landmark localization performance in a surgical setting
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于计算机视觉与医疗技术交叉任务，旨在解决手术环境中面部关键点定位准确性问题。论文通过控制实验，评估MediaPipe算法在不同姿态和光照条件下的检测效果，验证其在固定手术光照下的性能提升，探讨其在医疗场景中的应用潜力。**

- **链接: [http://arxiv.org/pdf/2507.18248v1](http://arxiv.org/pdf/2507.18248v1)**

> **作者:** Ines Frajtag; Marko Švaco; Filip Šuligoj
>
> **摘要:** The use of robotics, computer vision, and their applications is becoming increasingly widespread in various fields, including medicine. Many face detection algorithms have found applications in neurosurgery, ophthalmology, and plastic surgery. A common challenge in using these algorithms is variable lighting conditions and the flexibility of detection positions to identify and precisely localize patients. The proposed experiment tests the MediaPipe algorithm for detecting facial landmarks in a controlled setting, using a robotic arm that automatically adjusts positions while the surgical light and the phantom remain in a fixed position. The results of this study demonstrate that the improved accuracy of facial landmark detection under surgical lighting significantly enhances the detection performance at larger yaw and pitch angles. The increase in standard deviation/dispersion occurs due to imprecise detection of selected facial landmarks. This analysis allows for a discussion on the potential integration of the MediaPipe algorithm into medical procedures.
>
---
#### [new 112] SafeWork-R1: Coevolving Safety and Intelligence under the AI-45$^{\circ}$ Law
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于人工智能任务，旨在解决AI模型在提升能力的同时保障安全性。论文提出了SafeLadder框架，通过渐进式安全强化学习和多原则验证器，使模型SafeWork-R1具备内在安全推理和自我反思能力。实验表明其在安全相关基准上大幅提升性能，且不损害通用能力，展示了安全与能力可协同进化。**

- **链接: [http://arxiv.org/pdf/2507.18576v1](http://arxiv.org/pdf/2507.18576v1)**

> **作者:** Shanghai AI Lab; :; Yicheng Bao; Guanxu Chen; Mingkang Chen; Yunhao Chen; Chiyu Chen; Lingjie Chen; Sirui Chen; Xinquan Chen; Jie Cheng; Yu Cheng; Dengke Deng; Yizhuo Ding; Dan Ding; Xiaoshan Ding; Yi Ding; Zhichen Dong; Lingxiao Du; Yuyu Fan; Xinshun Feng; Yanwei Fu; Yuxuan Gao; Ruijun Ge; Tianle Gu; Lujun Gui; Jiaxuan Guo; Qianxi He; Yuenan Hou; Xuhao Hu; Hong Huang; Kaichen Huang; Shiyang Huang; Yuxian Jiang; Shanzhe Lei; Jie Li; Lijun Li; Hao Li; Juncheng Li; Xiangtian Li; Yafu Li; Lingyu Li; Xueyan Li; Haotian Liang; Dongrui Liu; Qihua Liu; Zhixuan Liu; Bangwei Liu; Huacan Liu; Yuexiao Liu; Zongkai Liu; Chaochao Lu; Yudong Lu; Xiaoya Lu; Zhenghao Lu; Qitan Lv; Caoyuan Ma; Jiachen Ma; Xiaoya Ma; Zhongtian Ma; Lingyu Meng; Ziqi Miao; Yazhe Niu; Yuezhang Peng; Yuan Pu; Han Qi; Chen Qian; Xingge Qiao; Jingjing Qu; Jiashu Qu; Wanying Qu; Wenwen Qu; Xiaoye Qu; Qihan Ren; Qingnan Ren; Qingyu Ren; Jing Shao; Wenqi Shao; Shuai Shao; Dongxing Shi; Xin Song; Xinhao Song; Yan Teng; Xuan Tong; Yingchun Wang; Xuhong Wang; Shujie Wang; Xin Wang; Yige Wang; Yixu Wang; Yuanfu Wang; Futing Wang; Ruofan Wang; Wenjie Wang; Yajie Wang; Muhao Wei; Xiaoyu Wen; Fenghua Weng; Yuqi Wu; Yingtong Xiong; Xingcheng Xu; Chao Yang; Yue Yang; Yang Yao; Yulei Ye; Zhenyun Yin; Yi Yu; Bo Zhang; Qiaosheng Zhang; Jinxuan Zhang; Yexin Zhang; Yinqiang Zheng; Hefeng Zhou; Zhanhui Zhou; Pengyu Zhu; Qingzi Zhu; Yubo Zhu; Bowen Zhou
>
> **备注:** 47 pages, 18 figures, authors are listed in alphabetical order by their last names
>
> **摘要:** We introduce SafeWork-R1, a cutting-edge multimodal reasoning model that demonstrates the coevolution of capabilities and safety. It is developed by our proposed SafeLadder framework, which incorporates large-scale, progressive, safety-oriented reinforcement learning post-training, supported by a suite of multi-principled verifiers. Unlike previous alignment methods such as RLHF that simply learn human preferences, SafeLadder enables SafeWork-R1 to develop intrinsic safety reasoning and self-reflection abilities, giving rise to safety `aha' moments. Notably, SafeWork-R1 achieves an average improvement of $46.54\%$ over its base model Qwen2.5-VL-72B on safety-related benchmarks without compromising general capabilities, and delivers state-of-the-art safety performance compared to leading proprietary models such as GPT-4.1 and Claude Opus 4. To further bolster its reliability, we implement two distinct inference-time intervention methods and a deliberative search mechanism, enforcing step-level verification. Finally, we further develop SafeWork-R1-InternVL3-78B, SafeWork-R1-DeepSeek-70B, and SafeWork-R1-Qwen2.5VL-7B. All resulting models demonstrate that safety and capability can co-evolve synergistically, highlighting the generalizability of our framework in building robust, reliable, and trustworthy general-purpose AI.
>
---
#### [new 113] VIBE: Video-Input Brain Encoder for fMRI Response Modeling
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出VIBE，一种用于fMRI响应建模的视频输入脑编码器。它通过融合视频、音频和文本多模态特征，预测大脑活动。使用Transformer架构，结合旋转位置嵌入，训练于电影数据集，在Algonauts挑战中表现优异。**

- **链接: [http://arxiv.org/pdf/2507.17958v1](http://arxiv.org/pdf/2507.17958v1)**

> **作者:** Daniel Carlstrom Schad; Shrey Dixit; Janis Keck; Viktor Studenyak; Aleksandr Shpilevoi; Andrej Bicanski
>
> **摘要:** We present VIBE, a two-stage Transformer that fuses multi-modal video, audio, and text features to predict fMRI activity. Representations from open-source models (Qwen2.5, BEATs, Whisper, SlowFast, V-JEPA) are merged by a modality-fusion transformer and temporally decoded by a prediction transformer with rotary embeddings. Trained on 65 hours of movie data from the CNeuroMod dataset and ensembled across 20 seeds, VIBE attains mean parcel-wise Pearson correlations of 32.25 on in-distribution Friends S07 and 21.25 on six out-of-distribution films. An earlier iteration of the same architecture obtained 0.3198 and 0.2096, respectively, winning Phase-1 and placing second overall in the Algonauts 2025 Challenge.
>
---
#### [new 114] Enhancing Quantization-Aware Training on Edge Devices via Relative Entropy Coreset Selection and Cascaded Layer Correction
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于模型压缩任务，旨在解决边缘设备上低比特量化模型性能下降问题。通过提出QuaRC框架，结合相对熵核心集选择与级联层校正策略，有效提升量化感知训练在小规模数据上的表现。**

- **链接: [http://arxiv.org/pdf/2507.17768v1](http://arxiv.org/pdf/2507.17768v1)**

> **作者:** Yujia Tong; Jingling Yuan; Chuang Hu
>
> **摘要:** With the development of mobile and edge computing, the demand for low-bit quantized models on edge devices is increasing to achieve efficient deployment. To enhance the performance, it is often necessary to retrain the quantized models using edge data. However, due to privacy concerns, certain sensitive data can only be processed on edge devices. Therefore, employing Quantization-Aware Training (QAT) on edge devices has become an effective solution. Nevertheless, traditional QAT relies on the complete dataset for training, which incurs a huge computational cost. Coreset selection techniques can mitigate this issue by training on the most representative subsets. However, existing methods struggle to eliminate quantization errors in the model when using small-scale datasets (e.g., only 10% of the data), leading to significant performance degradation. To address these issues, we propose QuaRC, a QAT framework with coresets on edge devices, which consists of two main phases: In the coreset selection phase, QuaRC introduces the ``Relative Entropy Score" to identify the subsets that most effectively capture the model's quantization errors. During the training phase, QuaRC employs the Cascaded Layer Correction strategy to align the intermediate layer outputs of the quantized model with those of the full-precision model, thereby effectively reducing the quantization errors in the intermediate layers. Experimental results demonstrate the effectiveness of our approach. For instance, when quantizing ResNet-18 to 2-bit using a 1% data subset, QuaRC achieves a 5.72% improvement in Top-1 accuracy on the ImageNet-1K dataset compared to state-of-the-art techniques.
>
---
#### [new 115] Hierarchical Diffusion Framework for Pseudo-Healthy Brain MRI Inpainting with Enhanced 3D Consistency
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决病理脑MRI图像修复中伪健康区域生成的问题。现有方法在2D切片上处理导致层间不连续，而3D模型需大量数据。作者提出一种分层扩散框架，结合轴向和冠状面的2D扩散模型，实现高效且具三维一致性的图像修复。**

- **链接: [http://arxiv.org/pdf/2507.17911v1](http://arxiv.org/pdf/2507.17911v1)**

> **作者:** Dou Hoon Kwark; Shirui Luo; Xiyue Zhu; Yudu Li; Zhi-Pei Liang; Volodymyr Kindratenko
>
> **备注:** 11 pages, 2 figures
>
> **摘要:** Pseudo-healthy image inpainting is an essential preprocessing step for analyzing pathological brain MRI scans. Most current inpainting methods favor slice-wise 2D models for their high in-plane fidelity, but their independence across slices produces discontinuities in the volume. Fully 3D models alleviate this issue, but their high model capacity demands extensive training data for reliable, high-fidelity synthesis -- often impractical in medical settings. We address these limitations with a hierarchical diffusion framework by replacing direct 3D modeling with two perpendicular coarse-to-fine 2D stages. An axial diffusion model first yields a coarse, globally consistent inpainting; a coronal diffusion model then refines anatomical details. By combining perpendicular spatial views with adaptive resampling, our method balances data efficiency and volumetric consistency. Our experiments show our approach outperforms state-of-the-art baselines in both realism and volumetric consistency, making it a promising solution for pseudo-healthy image inpainting. Code is available at https://github.com/dou0000/3dMRI-Consistent-Inpaint.
>
---
#### [new 116] ReSem3D: Refinable 3D Spatial Constraints via Fine-Grained Semantic Grounding for Generalizable Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.HC; cs.LG**

- **简介: 该论文属于机器人操作任务，旨在解决语义多样环境下操作任务理解与执行的统一问题。现有方法存在语义粒度粗、缺乏实时闭环规划、鲁棒性差等问题。论文提出ReSem3D框架，结合视觉基础模型和多模态大语言模型，通过分层递归推理实现细粒度语义定位，动态构建3D空间约束，提升操作的适应性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.18262v1](http://arxiv.org/pdf/2507.18262v1)**

> **作者:** Chenyu Su; Weiwei Shang; Chen Qian; Fei Zhang; Shuang Cong
>
> **备注:** 12 pages,9 figures
>
> **摘要:** Semantics-driven 3D spatial constraints align highlevel semantic representations with low-level action spaces, facilitating the unification of task understanding and execution in robotic manipulation. The synergistic reasoning of Multimodal Large Language Models (MLLMs) and Vision Foundation Models (VFMs) enables cross-modal 3D spatial constraint construction. Nevertheless, existing methods have three key limitations: (1) coarse semantic granularity in constraint modeling, (2) lack of real-time closed-loop planning, (3) compromised robustness in semantically diverse environments. To address these challenges, we propose ReSem3D, a unified manipulation framework for semantically diverse environments, leveraging the synergy between VFMs and MLLMs to achieve fine-grained visual grounding and dynamically constructs hierarchical 3D spatial constraints for real-time manipulation. Specifically, the framework is driven by hierarchical recursive reasoning in MLLMs, which interact with VFMs to automatically construct 3D spatial constraints from natural language instructions and RGB-D observations in two stages: part-level extraction and region-level refinement. Subsequently, these constraints are encoded as real-time optimization objectives in joint space, enabling reactive behavior to dynamic disturbances. Extensive simulation and real-world experiments are conducted in semantically rich household and sparse chemical lab environments. The results demonstrate that ReSem3D performs diverse manipulation tasks under zero-shot conditions, exhibiting strong adaptability and generalization. Code and videos at https://resem3d.github.io.
>
---
#### [new 117] NWaaS: Nonintrusive Watermarking as a Service for X-to-Image DNN
- **分类: cs.CR; cs.CV**

- **简介: 论文提出NWaaS，一种非侵入式水印即服务框架，用于X到图像的深度神经网络。它通过模型输出而非修改模型本身嵌入水印，解决模型版权保护问题，同时避免模型行为改变和额外调优成本。**

- **链接: [http://arxiv.org/pdf/2507.18036v1](http://arxiv.org/pdf/2507.18036v1)**

> **作者:** Haonan An; Guang Hua; Yu Guo; Hangcheng Cao; Susanto Rahardja; Yuguang Fang
>
> **摘要:** The intellectual property of deep neural network (DNN) models can be protected with DNN watermarking, which embeds copyright watermarks into model parameters (white-box), model behavior (black-box), or model outputs (box-free), and the watermarks can be subsequently extracted to verify model ownership or detect model theft. Despite recent advances, these existing methods are inherently intrusive, as they either modify the model parameters or alter the structure. This natural intrusiveness raises concerns about watermarking-induced shifts in model behavior and the additional cost of fine-tuning, further exacerbated by the rapidly growing model size. As a result, model owners are often reluctant to adopt DNN watermarking in practice, which limits the development of practical Watermarking as a Service (WaaS) systems. To address this issue, we introduce Nonintrusive Watermarking as a Service (NWaaS), a novel trustless paradigm designed for X-to-Image models, in which we hypothesize that with the model untouched, an owner-defined watermark can still be extracted from model outputs. Building on this concept, we propose ShadowMark, a concrete implementation of NWaaS which addresses critical deployment challenges by establishing a robust and nonintrusive side channel in the protected model's black-box API, leveraging a key encoder and a watermark decoder. It is significantly distinctive from existing solutions by attaining the so-called absolute fidelity and being applicable to different DNN architectures, while being also robust against existing attacks, eliminating the fidelity-robustness trade-off. Extensive experiments on image-to-image, noise-to-image, noise-and-text-to-image, and text-to-image models, demonstrate the efficacy and practicality of ShadowMark for real-world deployment of nonintrusive DNN watermarking.
>
---
#### [new 118] Integrating Feature Selection and Machine Learning for Nitrogen Assessment in Grapevine Leaves using In-Field Hyperspectral Imaging
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于农业遥感与机器学习结合的任务，旨在解决葡萄园中氮素精准评估的问题。通过田间高光谱成像和特征选择，结合梯度提升与XGBoost模型，预测叶片与冠层水平的氮浓度，验证了光谱数据与机器学习结合在氮素监测中的潜力。**

- **链接: [http://arxiv.org/pdf/2507.17869v1](http://arxiv.org/pdf/2507.17869v1)**

> **作者:** Atif Bilal Asad; Achyut Paudel; Safal Kshetri; Chenchen Kang; Salik Ram Khanal; Nataliya Shcherbatyuk; Pierre Davadant; R. Paul Schreiner; Santosh Kalauni; Manoj Karkee; Markus Keller
>
> **摘要:** Nitrogen (N) is one of the most crucial nutrients in vineyards, affecting plant growth and subsequent products such as wine and juice. Because soil N has high spatial and temporal variability, it is desirable to accurately estimate the N concentration of grapevine leaves and manage fertilization at the individual plant level to optimally meet plant needs. In this study, we used in-field hyperspectral images with wavelengths ranging from $400 to 1000nm of four different grapevine cultivars collected from distinct vineyards and over two growth stages during two growing seasons to develop models for predicting N concentration at the leaf-level and canopy-level. After image processing, two feature selection methods were employed to identify the optimal set of spectral bands that were responsive to leaf N concentrations. The selected spectral bands were used to train and test two different Machine Learning (ML) models, Gradient Boosting and XGBoost, for predicting nitrogen concentrations. The comparison of selected bands for both leaf-level and canopy-level datasets showed that most of the spectral regions identified by the feature selection methods were across both methods and the dataset types (leaf- and canopy-level datasets), particularly in the key regions, 500-525nm, 650-690nm, 750-800nm, and 900-950nm. These findings indicated the robustness of these spectral regions for predicting nitrogen content. The results for N prediction demonstrated that the ML model achieved an R square of 0.49 for canopy-level data and an R square of 0.57 for leaf-level data, despite using different sets of selected spectral bands for each analysis level. The study demonstrated the potential of using in-field hyperspectral imaging and the use of spectral data in integrated feature selection and ML techniques to monitor N status in vineyards.
>
---
## 更新

#### [replaced 001] Att-Adapter: A Robust and Precise Domain-Specific Multi-Attributes T2I Diffusion Adapter via Conditional Variational Autoencoder
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.11937v4](http://arxiv.org/pdf/2503.11937v4)**

> **作者:** Wonwoong Cho; Yan-Ying Chen; Matthew Klenk; David I. Inouye; Yanxia Zhang
>
> **备注:** ICCV'25 (Highlight), The project page is available at https://tri-mac.github.io/att-adapter/
>
> **摘要:** Text-to-Image (T2I) Diffusion Models have achieved remarkable performance in generating high quality images. However, enabling precise control of continuous attributes, especially multiple attributes simultaneously, in a new domain (e.g., numeric values like eye openness or car width) with text-only guidance remains a significant challenge. To address this, we introduce the Attribute (Att) Adapter, a novel plug-and-play module designed to enable fine-grained, multi-attributes control in pretrained diffusion models. Our approach learns a single control adapter from a set of sample images that can be unpaired and contain multiple visual attributes. The Att-Adapter leverages the decoupled cross attention module to naturally harmonize the multiple domain attributes with text conditioning. We further introduce Conditional Variational Autoencoder (CVAE) to the Att-Adapter to mitigate overfitting, matching the diverse nature of the visual world. Evaluations on two public datasets show that Att-Adapter outperforms all LoRA-based baselines in controlling continuous attributes. Additionally, our method enables a broader control range and also improves disentanglement across multiple attributes, surpassing StyleGAN-based techniques. Notably, Att-Adapter is flexible, requiring no paired synthetic data for training, and is easily scalable to multiple attributes within a single model.
>
---
#### [replaced 002] VolDoGer: LLM-assisted Datasets for Domain Generalization in Vision-Language Tasks
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2407.19795v2](http://arxiv.org/pdf/2407.19795v2)**

> **作者:** Juhwan Choi; Junehyoung Kwon; JungMin Yun; Seunguk Yu; YoungBin Kim
>
> **备注:** ICCV 2025 Workshop on Curated Data for Efficient Learning (CDEL)
>
> **摘要:** Domain generalizability is a crucial aspect of a deep learning model since it determines the capability of the model to perform well on data from unseen domains. However, research on the domain generalizability of deep learning models for vision-language tasks remains limited, primarily because of the lack of required datasets. To address these challenges, we propose VolDoGer: Vision-Language Dataset for Domain Generalization, a dedicated dataset designed for domain generalization that addresses three vision-language tasks: image captioning, visual question answering, and visual entailment. We constructed VolDoGer by extending LLM-based data annotation techniques to vision-language tasks, thereby alleviating the burden of recruiting human annotators. We evaluated the domain generalizability of various models, ranging from fine-tuned models to a recent multimodal large language model, through VolDoGer.
>
---
#### [replaced 003] Aligning Vision to Language: Annotation-Free Multimodal Knowledge Graph Construction for Enhanced LLMs Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.12972v2](http://arxiv.org/pdf/2503.12972v2)**

> **作者:** Junming Liu; Siyuan Meng; Yanting Gao; Song Mao; Pinlong Cai; Guohang Yan; Yirong Chen; Zilin Bian; Ding Wang; Botian Shi
>
> **备注:** 14 pages, 7 figures, 6 tables; Accepted to ICCV 2025
>
> **摘要:** Multimodal reasoning in Large Language Models (LLMs) struggles with incomplete knowledge and hallucination artifacts, challenges that textual Knowledge Graphs (KGs) only partially mitigate due to their modality isolation. While Multimodal Knowledge Graphs (MMKGs) promise enhanced cross-modal understanding, their practical construction is impeded by semantic narrowness of manual text annotations and inherent noise in visual-semantic entity linkages. In this paper, we propose Vision-align-to-Language integrated Knowledge Graph (VaLiK), a novel approach for constructing MMKGs that enhances LLMs reasoning through cross-modal information supplementation. Specifically, we cascade pre-trained Vision-Language Models (VLMs) to align image features with text, transforming them into descriptions that encapsulate image-specific information. Furthermore, we developed a cross-modal similarity verification mechanism to quantify semantic consistency, effectively filtering out noise introduced during feature alignment. Even without manually annotated image captions, the refined descriptions alone suffice to construct the MMKG. Compared to conventional MMKGs construction paradigms, our approach achieves substantial storage efficiency gains while maintaining direct entity-to-image linkage capability. Experimental results on multimodal reasoning tasks demonstrate that LLMs augmented with VaLiK outperform previous state-of-the-art models. Our code is published at https://github.com/Wings-Of-Disaster/VaLiK.
>
---
#### [replaced 004] Rectifying Magnitude Neglect in Linear Attention
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.00698v2](http://arxiv.org/pdf/2507.00698v2)**

> **作者:** Qihang Fan; Huaibo Huang; Yuang Ai; ran He
>
> **备注:** Accepted by ICCV2025, highlight paper
>
> **摘要:** As the core operator of Transformers, Softmax Attention exhibits excellent global modeling capabilities. However, its quadratic complexity limits its applicability to vision tasks. In contrast, Linear Attention shares a similar formulation with Softmax Attention while achieving linear complexity, enabling efficient global information modeling. Nevertheless, Linear Attention suffers from a significant performance degradation compared to standard Softmax Attention. In this paper, we analyze the underlying causes of this issue based on the formulation of Linear Attention. We find that, unlike Softmax Attention, Linear Attention entirely disregards the magnitude information of the Query. This prevents the attention score distribution from dynamically adapting as the Query scales. As a result, despite its structural similarity to Softmax Attention, Linear Attention exhibits a significantly different attention score distribution. Based on this observation, we propose Magnitude-Aware Linear Attention (MALA), which modifies the computation of Linear Attention to fully incorporate the Query's magnitude. This adjustment allows MALA to generate an attention score distribution that closely resembles Softmax Attention while exhibiting a more well-balanced structure. We evaluate the effectiveness of MALA on multiple tasks, including image classification, object detection, instance segmentation, semantic segmentation, natural language processing, speech recognition, and image generation. Our MALA achieves strong results on all of these tasks. Code will be available at https://github.com/qhfan/MALA
>
---
#### [replaced 005] PolarAnything: Diffusion-based Polarimetric Image Synthesis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.17268v2](http://arxiv.org/pdf/2507.17268v2)**

> **作者:** Kailong Zhang; Youwei Lyu; Heng Guo; Si Li; Zhanyu Ma; Boxin Shi
>
> **备注:** 11 pages
>
> **摘要:** Polarization images facilitate image enhancement and 3D reconstruction tasks, but the limited accessibility of polarization cameras hinders their broader application. This gap drives the need for synthesizing photorealistic polarization images. The existing polarization simulator Mitsuba relies on a parametric polarization image formation model and requires extensive 3D assets covering shape and PBR materials, preventing it from generating large-scale photorealistic images. To address this problem, we propose PolarAnything, capable of synthesizing polarization images from a single RGB input with both photorealism and physical accuracy, eliminating the dependency on 3D asset collections. Drawing inspiration from the zero-shot performance of pretrained diffusion models, we introduce a diffusion-based generative framework with an effective representation strategy that preserves the fidelity of polarization properties. Experiments show that our model generates high-quality polarization images and supports downstream tasks like shape from polarization.
>
---
#### [replaced 006] LPTR-AFLNet: Lightweight Integrated Chinese License Plate Rectification and Recognition Network
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.16362v2](http://arxiv.org/pdf/2507.16362v2)**

> **作者:** Guangzhu Xu; Pengcheng Zuo; Zhi Ke; Bangjun Lei
>
> **备注:** 28 pages, 33 figures
>
> **摘要:** Chinese License Plate Recognition (CLPR) faces numerous challenges in unconstrained and complex environments, particularly due to perspective distortions caused by various shooting angles and the correction of single-line and double-line license plates. Considering the limited computational resources of edge devices, developing a low-complexity, end-to-end integrated network for both correction and recognition is essential for achieving real-time and efficient deployment. In this work, we propose a lightweight, unified network named LPTR-AFLNet for correcting and recognizing Chinese license plates, which combines a perspective transformation correction module (PTR) with an optimized license plate recognition network, AFLNet. The network leverages the recognition output as a weak supervisory signal to effectively guide the correction process, ensuring accurate perspective distortion correction. To enhance recognition accuracy, we introduce several improvements to LPRNet, including an improved attention module to reduce confusion among similar characters and the use of Focal Loss to address class imbalance during training. Experimental results demonstrate the exceptional performance of LPTR-AFLNet in rectifying perspective distortion and recognizing double-line license plate images, maintaining high recognition accuracy across various challenging scenarios. Moreover, on lower-mid-range GPUs platform, the method runs in less than 10 milliseconds, indicating its practical efficiency and broad applicability.
>
---
#### [replaced 007] DAA*: Deep Angular A Star for Image-based Path Planning
- **分类: cs.CV; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2507.09305v3](http://arxiv.org/pdf/2507.09305v3)**

> **作者:** Zhiwei Xu
>
> **备注:** International Conference on Computer Vision (ICCV), 2025
>
> **摘要:** Path smoothness is often overlooked in path imitation learning from expert demonstrations. In this paper, we introduce a novel learning method, termed deep angular A* (DAA*), by incorporating the proposed path angular freedom (PAF) into A* to improve path similarity through adaptive path smoothness. The PAF aims to explore the effect of move angles on path node expansion by finding the trade-off between their minimum and maximum values, allowing for high adaptiveness for imitation learning. DAA* improves path optimality by closely aligning with the reference path through joint optimization of path shortening and smoothing, which correspond to heuristic distance and PAF, respectively. Throughout comprehensive evaluations on 7 datasets, including 4 maze datasets, 2 video-game datasets, and a real-world drone-view dataset containing 2 scenarios, we demonstrate remarkable improvements of our DAA* over neural A* in path similarity between the predicted and reference paths with a shorter path length when the shortest path is plausible, improving by 9.0% SPR, 6.9% ASIM, and 3.9% PSIM. Furthermore, when jointly learning pathfinding with both path loss and path probability map loss, DAA* significantly outperforms the state-of-the-art TransPath by 6.3% SPR, 6.0% PSIM, and 3.7% ASIM. We also discuss the minor trade-off between path optimality and search efficiency where applicable. Our code and model weights are available at https://github.com/zwxu064/DAAStar.git.
>
---
#### [replaced 008] Scalable Frame Sampling for Video Classification: A Semi-Optimal Policy Approach with Reduced Search Space
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.05260v3](http://arxiv.org/pdf/2409.05260v3)**

> **作者:** Junho Lee; Jeongwoo Shin; Seung Woo Ko; Seongsu Ha; Joonseok Lee
>
> **摘要:** Given a video with $T$ frames, frame sampling is a task to select $N \ll T$ frames, so as to maximize the performance of a fixed video classifier. Not just brute-force search, but most existing methods suffer from its vast search space of $\binom{T}{N}$, especially when $N$ gets large. To address this challenge, we introduce a novel perspective of reducing the search space from $O(T^N)$ to $O(T)$. Instead of exploring the entire $O(T^N)$ space, our proposed semi-optimal policy selects the top $N$ frames based on the independently estimated value of each frame using per-frame confidence, significantly reducing the computational complexity. We verify that our semi-optimal policy can efficiently approximate the optimal policy, particularly under practical settings. Additionally, through extensive experiments on various datasets and model architectures, we demonstrate that learning our semi-optimal policy ensures stable and high performance regardless of the size of $N$ and $T$.
>
---
#### [replaced 009] EndoControlMag: Robust Endoscopic Vascular Motion Magnification with Periodic Reference Resetting and Hierarchical Tissue-aware Dual-Mask Control
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.15292v4](http://arxiv.org/pdf/2507.15292v4)**

> **作者:** An Wang; Rulin Zhou; Mengya Xu; Yiru Ye; Longfei Gou; Yiting Chang; Hao Chen; Chwee Ming Lim; Jiankun Wang; Hongliang Ren
>
> **摘要:** Visualizing subtle vascular motions in endoscopic surgery is crucial for surgical precision and decision-making, yet remains challenging due to the complex and dynamic nature of surgical scenes. To address this, we introduce EndoControlMag, a training-free, Lagrangian-based framework with mask-conditioned vascular motion magnification tailored to endoscopic environments. Our approach features two key modules: a Periodic Reference Resetting (PRR) scheme that divides videos into short overlapping clips with dynamically updated reference frames to prevent error accumulation while maintaining temporal coherence, and a Hierarchical Tissue-aware Magnification (HTM) framework with dual-mode mask dilation. HTM first tracks vessel cores using a pretrained visual tracking model to maintain accurate localization despite occlusions and view changes. It then applies one of two adaptive softening strategies to surrounding tissues: motion-based softening that modulates magnification strength proportional to observed tissue displacement, or distance-based exponential decay that simulates biomechanical force attenuation. This dual-mode approach accommodates diverse surgical scenarios-motion-based softening excels with complex tissue deformations while distance-based softening provides stability during unreliable optical flow conditions. We evaluate EndoControlMag on our EndoVMM24 dataset spanning four different surgery types and various challenging scenarios, including occlusions, instrument disturbance, view changes, and vessel deformations. Quantitative metrics, visual assessments, and expert surgeon evaluations demonstrate that EndoControlMag significantly outperforms existing methods in both magnification accuracy and visual quality while maintaining robustness across challenging surgical conditions. The code, dataset, and video results are available at https://szupc.github.io/EndoControlMag/.
>
---
#### [replaced 010] A Transfer Learning-Based Method for Water Body Segmentation in Remote Sensing Imagery: A Case Study of the Zhada Tulin Area
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.10084v2](http://arxiv.org/pdf/2507.10084v2)**

> **作者:** Haonan Chen; Xin Tong
>
> **备注:** 13 pages, 6 figures, 2 tables
>
> **摘要:** The Tibetan Plateau, known as the Asian Water Tower, faces significant water security challenges due to its high sensitivity to climate change. Advancing Earth observation for sustainable water monitoring is thus essential for building climate resilience in this region. This study proposes a two-stage transfer learning strategy using the SegFormer model to overcome domain shift and data scarcit--key barriers in developing robust AI for climate-sensitive applications. After pre-training on a diverse source domain, our model was fine-tuned for the arid Zhada Tulin area. Experimental results show a substantial performance boost: the Intersection over Union (IoU) for water body segmentation surged from 25.50% (direct transfer) to 64.84%. This AI-driven accuracy is crucial for disaster risk reduction, particularly in monitoring flash flood-prone systems. More importantly, the high-precision map reveals a highly concentrated spatial distribution of water, with over 80% of the water area confined to less than 20% of the river channel length. This quantitative finding provides crucial evidence for understanding hydrological processes and designing targeted water management and climate adaptation strategies. Our work thus demonstrates an effective technical solution for monitoring arid plateau regions and contributes to advancing AI-powered Earth observation for disaster preparedness in critical transboundary river headwaters.
>
---
#### [replaced 011] Self-Reinforcing Prototype Evolution with Dual-Knowledge Cooperation for Semi-Supervised Lifelong Person Re-Identification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.01884v2](http://arxiv.org/pdf/2507.01884v2)**

> **作者:** Kunlun Xu; Fan Zhuo; Jiangmeng Li; Xu Zou; Jiahuan Zhou
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Current lifelong person re-identification (LReID) methods predominantly rely on fully labeled data streams. However, in real-world scenarios where annotation resources are limited, a vast amount of unlabeled data coexists with scarce labeled samples, leading to the Semi-Supervised LReID (Semi-LReID) problem where LReID methods suffer severe performance degradation. Existing LReID methods, even when combined with semi-supervised strategies, suffer from limited long-term adaptation performance due to struggling with the noisy knowledge occurring during unlabeled data utilization. In this paper, we pioneer the investigation of Semi-LReID, introducing a novel Self-Reinforcing Prototype Evolution with Dual-Knowledge Cooperation framework (SPRED). Our key innovation lies in establishing a self-reinforcing cycle between dynamic prototype-guided pseudo-label generation and new-old knowledge collaborative purification to enhance the utilization of unlabeled data. Specifically, learnable identity prototypes are introduced to dynamically capture the identity distributions and generate high-quality pseudo-labels. Then, the dual-knowledge cooperation scheme integrates current model specialization and historical model generalization, refining noisy pseudo-labels. Through this cyclic design, reliable pseudo-labels are progressively mined to improve current-stage learning and ensure positive knowledge propagation over long-term learning. Experiments on the established Semi-LReID benchmarks show that our SPRED achieves state-of-the-art performance. Our source code is available at https://github.com/zhoujiahuan1991/ICCV2025-SPRED
>
---
#### [replaced 012] Orthogonal Constrained Minimization with Tensor $\ell_{2,p}$ Regularization for HSI Denoising and Destriping
- **分类: math.OC; cs.CV; 68U10, 90C26, 15A18, 65F22**

- **链接: [http://arxiv.org/pdf/2407.03605v3](http://arxiv.org/pdf/2407.03605v3)**

> **作者:** Xiaoxia Liu; Shijie Yu; Jian Lu; Xiaojun Chen
>
> **摘要:** Hyperspectral images~(HSIs) are often contaminated by a mixture of noise such as Gaussian noise, dead lines, stripes, and so on. In this paper, we propose a multi-scale low-rank tensor regularized $\ell_{2,p}$ (MLTL2p) approach for HSI denoising and destriping, which consists of an orthogonal constrained minimization model and an iterative algorithm with convergence guarantees. The model of the proposed MLTL2p approach is built based on a new sparsity-enhanced Multi-scale Low-rank Tensor regularization and a tensor $\ell_{2,p}$ norm with \(p\in (0,1)\). The multi-scale low-rank regularization for HSI denoising utilizes the global and local spectral correlation as well as the spatial nonlocal self-similarity priors of HSIs. The corresponding low-rank constraints are formulated based on independent higher-order singular value decomposition with sparsity enhancement on its core tensor to prompt more low-rankness. The tensor $\ell_{2,p}$ norm for HSI destriping is extended from the matrix $\ell_{2,p}$ norm. A proximal block coordinate descent algorithm is proposed in the MLTL2p approach to solve the resulting nonconvex nonsmooth minimization with orthogonal constraints. We show any accumulation point of the sequence generated by the proposed algorithm converges to a first-order stationary point, which is defined using three equalities of substationarity, symmetry, and feasibility for orthogonal constraints. In the numerical experiments, we compare the proposed method with state-of-the-art methods including a deep learning based method, and test the methods on both simulated and real HSI datasets. Our proposed MLTL2p method demonstrates outperformance in terms of metrics such as mean peak signal-to-noise ratio as well as visual quality.
>
---
#### [replaced 013] CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.16319v3](http://arxiv.org/pdf/2411.16319v3)**

> **作者:** Leon Sick; Dominik Engel; Sebastian Hartwig; Pedro Hermosilla; Timo Ropinski
>
> **备注:** Accepted at ICCV 2025. Project Page with Code, Models & Demo: https://leonsick.github.io/cuts3d/
>
> **摘要:** Traditionally, algorithms that learn to segment object instances in 2D images have heavily relied on large amounts of human-annotated data. Only recently, novel approaches have emerged tackling this problem in an unsupervised fashion. Generally, these approaches first generate pseudo-masks and then train a class-agnostic detector. While such methods deliver the current state of the art, they often fail to correctly separate instances overlapping in 2D image space since only semantics are considered. To tackle this issue, we instead propose to cut the semantic masks in 3D to obtain the final 2D instances by utilizing a point cloud representation of the scene. Furthermore, we derive a Spatial Importance function, which we use to resharpen the semantics along the 3D borders of instances. Nevertheless, these pseudo-masks are still subject to mask ambiguity. To address this issue, we further propose to augment the training of a class-agnostic detector with three Spatial Confidence components aiming to isolate a clean learning signal. With these contributions, our approach outperforms competing methods across multiple standard benchmarks for unsupervised instance segmentation and object detection.
>
---
#### [replaced 014] Inversion-DPO: Precise and Efficient Post-Training for Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.11554v3](http://arxiv.org/pdf/2507.11554v3)**

> **作者:** Zejian Li; Yize Li; Chenye Meng; Zhongni Liu; Yang Ling; Shengyuan Zhang; Guang Yang; Changyuan Yang; Zhiyuan Yang; Lingyun Sun
>
> **备注:** Accepted by ACM MM25
>
> **摘要:** Recent advancements in diffusion models (DMs) have been propelled by alignment methods that post-train models to better conform to human preferences. However, these approaches typically require computation-intensive training of a base model and a reward model, which not only incurs substantial computational overhead but may also compromise model accuracy and training efficiency. To address these limitations, we propose Inversion-DPO, a novel alignment framework that circumvents reward modeling by reformulating Direct Preference Optimization (DPO) with DDIM inversion for DMs. Our method conducts intractable posterior sampling in Diffusion-DPO with the deterministic inversion from winning and losing samples to noise and thus derive a new post-training paradigm. This paradigm eliminates the need for auxiliary reward models or inaccurate appromixation, significantly enhancing both precision and efficiency of training. We apply Inversion-DPO to a basic task of text-to-image generation and a challenging task of compositional image generation. Extensive experiments show substantial performance improvements achieved by Inversion-DPO compared to existing post-training methods and highlight the ability of the trained generative models to generate high-fidelity compositionally coherent images. For the post-training of compostitional image geneation, we curate a paired dataset consisting of 11,140 images with complex structural annotations and comprehensive scores, designed to enhance the compositional capabilities of generative models. Inversion-DPO explores a new avenue for efficient, high-precision alignment in diffusion models, advancing their applicability to complex realistic generation tasks. Our code is available at https://github.com/MIGHTYEZ/Inversion-DPO
>
---
#### [replaced 015] Diffusion Beats Autoregressive in Data-Constrained Settings
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.15857v2](http://arxiv.org/pdf/2507.15857v2)**

> **作者:** Mihir Prabhudesai; Menging Wu; Amir Zadeh; Katerina Fragkiadaki; Deepak Pathak
>
> **备注:** Project Webpage: https://diffusion-scaling.github.io
>
> **摘要:** Autoregressive (AR) models have long dominated the landscape of large language models, driving progress across a wide range of tasks. Recently, diffusion-based language models have emerged as a promising alternative, though their advantages over AR models remain underexplored. In this paper, we systematically study masked diffusion models in data-constrained settings-where training involves repeated passes over limited data-and find that they significantly outperform AR models when compute is abundant but data is scarce. Diffusion models make better use of repeated data, achieving lower validation loss and superior downstream performance. We interpret this advantage as implicit data augmentation: masked diffusion exposes the model to a diverse distribution of token orderings and prediction tasks, unlike AR's fixed left-to-right factorization. We find new scaling laws for diffusion models and derive a closed-form expression for the critical compute threshold at which diffusion begins to outperform AR. These results suggest that when data, not compute, is the bottleneck, diffusion models offer a compelling alternative to the standard AR paradigm. Our code is available at: https://diffusion-scaling.github.io.
>
---
#### [replaced 016] Swin-TUNA : A Novel PEFT Approach for Accurate Food Image Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.17347v2](http://arxiv.org/pdf/2507.17347v2)**

> **作者:** Haotian Chen; Zhiyong Xiao
>
> **备注:** After discussion among the authors, some parts of the paper are deemed inappropriate and will be revised and resubmitted
>
> **摘要:** In the field of food image processing, efficient semantic segmentation techniques are crucial for industrial applications. However, existing large-scale Transformer-based models (such as FoodSAM) face challenges in meeting practical deploymentrequirements due to their massive parameter counts and high computational resource demands. This paper introduces TUNable Adapter module (Swin-TUNA), a Parameter Efficient Fine-Tuning (PEFT) method that integrates multiscale trainable adapters into the Swin Transformer architecture, achieving high-performance food image segmentation by updating only 4% of the parameters. The core innovation of Swin-TUNA lies in its hierarchical feature adaptation mechanism: it designs separable convolutions in depth and dimensional mappings of varying scales to address the differences in features between shallow and deep networks, combined with a dynamic balancing strategy for tasks-agnostic and task-specific features. Experiments demonstrate that this method achieves mIoU of 50.56% and 74.94% on the FoodSeg103 and UECFoodPix Complete datasets, respectively, surpassing the fully parameterized FoodSAM model while reducing the parameter count by 98.7% (to only 8.13M). Furthermore, Swin-TUNA exhibits faster convergence and stronger generalization capabilities in low-data scenarios, providing an efficient solution for assembling lightweight food image.
>
---
#### [replaced 017] L-FUSION: Laplacian Fetal Ultrasound Segmentation & Uncertainty Estimation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.05245v3](http://arxiv.org/pdf/2503.05245v3)**

> **作者:** Johanna P. Müller; Robert Wright; Thomas G. Day; Lorenzo Venturini; Samuel F. Budd; Hadrien Reynaud; Joseph V. Hajnal; Reza Razavi; Bernhard Kainz
>
> **备注:** Accepted at MICCAI ASMUS 2025
>
> **摘要:** Accurate analysis of prenatal ultrasound (US) is essential for early detection of developmental anomalies. However, operator dependency and technical limitations (e.g. intrinsic artefacts and effects, setting errors) can complicate image interpretation and the assessment of diagnostic uncertainty. We present L-FUSION (Laplacian Fetal US Segmentation with Integrated FoundatiON models), a framework that integrates uncertainty quantification through unsupervised, normative learning and large-scale foundation models for robust segmentation of fetal structures in normal and pathological scans. We propose to utilise the aleatoric logit distributions of Stochastic Segmentation Networks and Laplace approximations with fast Hessian estimations to estimate epistemic uncertainty only from the segmentation head. This enables us to achieve reliable abnormality quantification for instant diagnostic feedback. Combined with an integrated Dropout component, L-FUSION enables reliable differentiation of lesions from normal fetal anatomy with enhanced uncertainty maps and segmentation counterfactuals in US imaging. It improves epistemic and aleatoric uncertainty interpretation and removes the need for manual disease-labelling. Evaluations across multiple datasets show that L-FUSION achieves superior segmentation accuracy and consistent uncertainty quantification, supporting on-site decision-making and offering a scalable solution for advancing fetal ultrasound analysis in clinical settings.
>
---
#### [replaced 018] Residual Prior-driven Frequency-aware Network for Image Fusion
- **分类: cs.CV; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2507.06735v2](http://arxiv.org/pdf/2507.06735v2)**

> **作者:** Guan Zheng; Xue Wang; Wenhua Qian; Peng Liu; Runzhuo Ma
>
> **摘要:** Image fusion aims to integrate complementary information across modalities to generate high-quality fused images, thereby enhancing the performance of high-level vision tasks. While global spatial modeling mechanisms show promising results, constructing long-range feature dependencies in the spatial domain incurs substantial computational costs. Additionally, the absence of ground-truth exacerbates the difficulty of capturing complementary features effectively. To tackle these challenges, we propose a Residual Prior-driven Frequency-aware Network, termed as RPFNet. Specifically, RPFNet employs a dual-branch feature extraction framework: the Residual Prior Module (RPM) extracts modality-specific difference information from residual maps, thereby providing complementary priors for fusion; the Frequency Domain Fusion Module (FDFM) achieves efficient global feature modeling and integration through frequency-domain convolution. Additionally, the Cross Promotion Module (CPM) enhances the synergistic perception of local details and global structures through bidirectional feature interaction. During training, we incorporate an auxiliary decoder and saliency structure loss to strengthen the model's sensitivity to modality-specific differences. Furthermore, a combination of adaptive weight-based frequency contrastive loss and SSIM loss effectively constrains the solution space, facilitating the joint capture of local details and global features while ensuring the retention of complementary information. Extensive experiments validate the fusion performance of RPFNet, which effectively integrates discriminative features, enhances texture details and salient objects, and can effectively facilitate the deployment of the high-level vision task.
>
---
#### [replaced 019] ELITE: Enhanced Language-Image Toxicity Evaluation for Safety
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.04757v3](http://arxiv.org/pdf/2502.04757v3)**

> **作者:** Wonjun Lee; Doehyeon Lee; Eugene Choi; Sangyoon Yu; Ashkan Yousefpour; Haon Park; Bumsub Ham; Suhyun Kim
>
> **备注:** ICML 2025. Project page at https://velpegor.github.io/ELITE/
>
> **摘要:** Current Vision Language Models (VLMs) remain vulnerable to malicious prompts that induce harmful outputs. Existing safety benchmarks for VLMs primarily rely on automated evaluation methods, but these methods struggle to detect implicit harmful content or produce inaccurate evaluations. Therefore, we found that existing benchmarks have low levels of harmfulness, ambiguous data, and limited diversity in image-text pair combinations. To address these issues, we propose the ELITE benchmark, a high-quality safety evaluation benchmark for VLMs, underpinned by our enhanced evaluation method, the ELITE evaluator. The ELITE evaluator explicitly incorporates a toxicity score to accurately assess harmfulness in multimodal contexts, where VLMs often provide specific, convincing, but unharmful descriptions of images. We filter out ambiguous and low-quality image-text pairs from existing benchmarks using the ELITE evaluator and generate diverse combinations of safe and unsafe image-text pairs. Our experiments demonstrate that the ELITE evaluator achieves superior alignment with human evaluations compared to prior automated methods, and the ELITE benchmark offers enhanced benchmark quality and diversity. By introducing ELITE, we pave the way for safer, more robust VLMs, contributing essential tools for evaluating and mitigating safety risks in real-world applications.
>
---
#### [replaced 020] MambaNeXt-YOLO: A Hybrid State Space Model for Real-time Object Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.03654v3](http://arxiv.org/pdf/2506.03654v3)**

> **作者:** Xiaochun Lei; Siqi Wu; Weilin Wu; Zetao Jiang
>
> **备注:** This paper is under consideration at Image and Vision Computing
>
> **摘要:** Real-time object detection is a fundamental but challenging task in computer vision, particularly when computational resources are limited. Although YOLO-series models have set strong benchmarks by balancing speed and accuracy, the increasing need for richer global context modeling has led to the use of Transformer-based architectures. Nevertheless, Transformers have high computational complexity because of their self-attention mechanism, which limits their practicality for real-time and edge deployments. To overcome these challenges, recent developments in linear state space models, such as Mamba, provide a promising alternative by enabling efficient sequence modeling with linear complexity. Building on this insight, we propose MambaNeXt-YOLO, a novel object detection framework that balances accuracy and efficiency through three key contributions: (1) MambaNeXt Block: a hybrid design that integrates CNNs with Mamba to effectively capture both local features and long-range dependencies; (2) Multi-branch Asymmetric Fusion Pyramid Network (MAFPN): an enhanced feature pyramid architecture that improves multi-scale object detection across various object sizes; and (3) Edge-focused Efficiency: our method achieved 66.6% mAP at 31.9 FPS on the PASCAL VOC dataset without any pre-training and supports deployment on edge devices such as the NVIDIA Jetson Xavier NX and Orin NX.
>
---
#### [replaced 021] External Knowledge Injection for CLIP-Based Class-Incremental Learning
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.08510v2](http://arxiv.org/pdf/2503.08510v2)**

> **作者:** Da-Wei Zhou; Kai-Wen Li; Jingyi Ning; Han-Jia Ye; Lijun Zhang; De-Chuan Zhan
>
> **备注:** Accepted to ICCV 2025. Code is available at: https://github.com/LAMDA-CL/ICCV25-ENGINE
>
> **摘要:** Class-Incremental Learning (CIL) enables learning systems to continuously adapt to evolving data streams. With the advancement of pre-training, leveraging pre-trained vision-language models (e.g., CLIP) offers a promising starting point for CIL. However, CLIP makes decisions by matching visual embeddings to class names, overlooking the rich contextual information conveyed through language. For instance, the concept of ``cat'' can be decomposed into features like tail, fur, and face for recognition. Besides, since the model is continually updated, these detailed features are overwritten in CIL, requiring external knowledge for compensation. In this paper, we introduce ExterNal knowledGe INjEction (ENGINE) for CLIP-based CIL. To enhance knowledge transfer from outside the dataset, we propose a dual-branch injection tuning framework that encodes informative knowledge from both visual and textual modalities. The visual branch is enhanced with data augmentation to enrich the visual features, while the textual branch leverages GPT-4 to rewrite discriminative descriptors. In addition to this on-the-fly knowledge injection, we also implement post-tuning knowledge by re-ranking the prediction results during inference. With the injected knowledge, the model can better capture informative features for downstream tasks as data evolves. Extensive experiments demonstrate the state-of-the-art performance of ENGINE. Code is available at: https://github.com/LAMDA-CL/ICCV25-ENGINE
>
---
#### [replaced 022] NSegment : Label-specific Deformations for Remote Sensing Image Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.19634v4](http://arxiv.org/pdf/2504.19634v4)**

> **作者:** Yechan Kim; DongHo Yoon; SooYeon Kim; Moongu Jeon
>
> **备注:** The paper is being revised substantially and will be resubmitted.
>
> **摘要:** Labeling errors in remote sensing (RS) image segmentation datasets often remain implicit and subtle due to ambiguous class boundaries, mixed pixels, shadows, complex terrain features, and subjective annotator bias. Furthermore, the scarcity of annotated RS data due to high image acquisition and labeling costs complicates training noise-robust models. While sophisticated mechanisms such as label selection or noise correction might address this issue, they tend to increase training time and add implementation complexity. In this letter, we propose NSegment-a simple yet effective data augmentation solution to mitigate this issue. Unlike traditional methods, it applies elastic transformations only to segmentation labels, varying deformation intensity per sample in each training epoch to address annotation inconsistencies. Experimental results demonstrate that our approach improves the performance of RS image segmentation on various state-of-the-art models.
>
---
#### [replaced 023] PerceptionLM: Open-Access Data and Models for Detailed Visual Understanding
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.13180v3](http://arxiv.org/pdf/2504.13180v3)**

> **作者:** Jang Hyun Cho; Andrea Madotto; Effrosyni Mavroudi; Triantafyllos Afouras; Tushar Nagarajan; Muhammad Maaz; Yale Song; Tengyu Ma; Shuming Hu; Suyog Jain; Miguel Martin; Huiyu Wang; Hanoona Rasheed; Peize Sun; Po-Yao Huang; Daniel Bolya; Nikhila Ravi; Shashank Jain; Tammy Stark; Shane Moon; Babak Damavandi; Vivian Lee; Andrew Westbury; Salman Khan; Philipp Krähenbühl; Piotr Dollár; Lorenzo Torresani; Kristen Grauman; Christoph Feichtenhofer
>
> **备注:** Technical Report
>
> **摘要:** Vision-language models are integral to computer vision research, yet many high-performing models remain closed-source, obscuring their data, design and training recipe. The research community has responded by using distillation from black-box models to label training data, achieving strong benchmark results, at the cost of measurable scientific progress. However, without knowing the details of the teacher model and its data sources, scientific progress remains difficult to measure. In this paper, we study building a Perception Language Model (PLM) in a fully open and reproducible framework for transparent research in image and video understanding. We analyze standard training pipelines without distillation from proprietary models and explore large-scale synthetic data to identify critical data gaps, particularly in detailed video understanding. To bridge these gaps, we release 2.8M human-labeled instances of fine-grained video question-answer pairs and spatio-temporally grounded video captions. Additionally, we introduce PLM-VideoBench, a suite for evaluating challenging video understanding tasks focusing on the ability to reason about "what", "where", "when", and "how" of a video. We make our work fully reproducible by providing data, training recipes, code & models. https://github.com/facebookresearch/perception_models
>
---
#### [replaced 024] Tackling Hallucination from Conditional Models for Medical Image Reconstruction with DynamicDPS
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.01075v2](http://arxiv.org/pdf/2503.01075v2)**

> **作者:** Seunghoi Kim; Henry F. J. Tregidgo; Matteo Figini; Chen Jin; Sarang Joshi; Daniel C. Alexander
>
> **摘要:** Hallucinations are spurious structures not present in the ground truth, posing a critical challenge in medical image reconstruction, especially for data-driven conditional models. We hypothesize that combining an unconditional diffusion model with data consistency, trained on a diverse dataset, can reduce these hallucinations. Based on this, we propose DynamicDPS, a diffusion-based framework that integrates conditional and unconditional diffusion models to enhance low-quality medical images while systematically reducing hallucinations. Our approach first generates an initial reconstruction using a conditional model, then refines it with an adaptive diffusion-based inverse problem solver. DynamicDPS skips early stage in the reverse process by selecting an optimal starting time point per sample and applies Wolfe's line search for adaptive step sizes, improving both efficiency and image fidelity. Using diffusion priors and data consistency, our method effectively reduces hallucinations from any conditional model output. We validate its effectiveness in Image Quality Transfer for low-field MRI enhancement. Extensive evaluations on synthetic and real MR scans, including a downstream task for tissue volume estimation, show that DynamicDPS reduces hallucinations, improving relative volume estimation by over 15% for critical tissues while using only 5% of the sampling steps required by baseline diffusion models. As a model-agnostic and fine-tuning-free approach, DynamicDPS offers a robust solution for hallucination reduction in medical imaging. The code will be made publicly available upon publication.
>
---
#### [replaced 025] History-Guided Video Diffusion
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.06764v2](http://arxiv.org/pdf/2502.06764v2)**

> **作者:** Kiwhan Song; Boyuan Chen; Max Simchowitz; Yilun Du; Russ Tedrake; Vincent Sitzmann
>
> **备注:** ICML 2025. Project website: https://boyuan.space/history-guidance
>
> **摘要:** Classifier-free guidance (CFG) is a key technique for improving conditional generation in diffusion models, enabling more accurate control while enhancing sample quality. It is natural to extend this technique to video diffusion, which generates video conditioned on a variable number of context frames, collectively referred to as history. However, we find two key challenges to guiding with variable-length history: architectures that only support fixed-size conditioning, and the empirical observation that CFG-style history dropout performs poorly. To address this, we propose the Diffusion Forcing Transformer (DFoT), a video diffusion architecture and theoretically grounded training objective that jointly enable conditioning on a flexible number of history frames. We then introduce History Guidance, a family of guidance methods uniquely enabled by DFoT. We show that its simplest form, vanilla history guidance, already significantly improves video generation quality and temporal consistency. A more advanced method, history guidance across time and frequency further enhances motion dynamics, enables compositional generalization to out-of-distribution history, and can stably roll out extremely long videos. Project website: https://boyuan.space/history-guidance
>
---
#### [replaced 026] Advancing Multimodal LLMs by Large-Scale 3D Visual Instruction Dataset Generation
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.08513v2](http://arxiv.org/pdf/2507.08513v2)**

> **作者:** Liu He; Xiao Zeng; Yizhi Song; Albert Y. C. Chen; Lu Xia; Shashwat Verma; Sankalp Dayal; Min Sun; Cheng-Hao Kuo; Daniel Aliaga
>
> **摘要:** Multimodal Large Language Models (MLLMs) struggle with accurately capturing camera-object relations, especially for object orientation, camera viewpoint, and camera shots. This stems from the fact that existing MLLMs are trained on images with limited diverse camera-object relations and corresponding textual descriptions. To address this, we propose a synthetic generation pipeline to create large-scale 3D visual instruction datasets. Our framework takes 3D assets as input and uses rendering and diffusion-based image generation models to create photorealistic images preserving precise camera-object relations. Additionally, large language models (LLMs) are used to generate text prompts for guiding visual instruction tuning and controlling image generation. We create Ultimate3D, a dataset of 240K VQAs with precise camera-object annotations, and corresponding benchmark. MLLMs fine-tuned on our proposed dataset outperform commercial models by a large margin, achieving an average accuracy improvement of 33.4% on camera-object relation recognition tasks. Our code, dataset, and benchmark will contribute to broad MLLM applications.
>
---
#### [replaced 027] Vision Transformers in Precision Agriculture: A Comprehensive Survey
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.21706v3](http://arxiv.org/pdf/2504.21706v3)**

> **作者:** Saber Mehdipour; Seyed Abolghasem Mirroshandel; Seyed Amirhossein Tabatabaei
>
> **摘要:** Detecting plant diseases is a crucial aspect of modern agriculture, as it plays a key role in maintaining crop health and increasing overall yield. Traditional approaches, though still valuable, often rely on manual inspection or conventional machine learning techniques, both of which face limitations in scalability and accuracy. Recently, Vision Transformers (ViTs) have emerged as a promising alternative, offering advantages such as improved handling of long-range dependencies and better scalability for visual tasks. This review explores the application of ViTs in precision agriculture, covering a range of tasks. We begin by introducing the foundational architecture of ViTs and discussing their transition from Natural Language Processing (NLP) to Computer Vision. The discussion includes the concept of inductive bias in traditional models like Convolutional Neural Networks (CNNs), and how ViTs mitigate these biases. We provide a comprehensive review of recent literature, focusing on key methodologies, datasets, and performance metrics. This study also includes a comparative analysis of CNNs and ViTs, along with a review of hybrid models and performance enhancements. Technical challenges such as data requirements, computational demands, and model interpretability are addressed, along with potential solutions. Finally, we outline future research directions and technological advancements that could further support the integration of ViTs in real-world agricultural settings. Our goal with this study is to offer practitioners and researchers a deeper understanding of how ViTs are poised to transform smart and precision agriculture.
>
---
#### [replaced 028] Distilling Diffusion Models to Efficient 3D LiDAR Scene Completion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.03515v2](http://arxiv.org/pdf/2412.03515v2)**

> **作者:** Shengyuan Zhang; An Zhao; Ling Yang; Zejian Li; Chenye Meng; Haoran Xu; Tianrun Chen; AnYang Wei; Perry Pengyun GU; Lingyun Sun
>
> **备注:** This paper is accept by ICCV'25(Oral), the model and code are publicly available on https: //github.com/happyw1nd/ScoreLiDAR
>
> **摘要:** Diffusion models have been applied to 3D LiDAR scene completion due to their strong training stability and high completion quality. However, the slow sampling speed limits the practical application of diffusion-based scene completion models since autonomous vehicles require an efficient perception of surrounding environments. This paper proposes a novel distillation method tailored for 3D Li- DAR scene completion models, dubbed ScoreLiDAR, which achieves efficient yet high-quality scene completion. Score- LiDAR enables the distilled model to sample in significantly fewer steps after distillation. To improve completion quality, we also introduce a novel Structural Loss, which encourages the distilled model to capture the geometric structure of the 3D LiDAR scene. The loss contains a scene-wise term constraining the holistic structure and a point-wise term constraining the key landmark points and their relative configuration. Extensive experiments demonstrate that ScoreLiDAR significantly accelerates the completion time from 30.55 to 5.37 seconds per frame (>5x) on SemanticKITTI and achieves superior performance compared to state-of-the-art 3D LiDAR scene completion models. Our model and code are publicly available on https: //github.com/happyw1nd/ScoreLiDAR.
>
---
#### [replaced 029] TextCrafter: Accurately Rendering Multiple Texts in Complex Visual Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.23461v4](http://arxiv.org/pdf/2503.23461v4)**

> **作者:** Nikai Du; Zhennan Chen; Zhizhou Chen; Shan Gao; Xi Chen; Zhengkai Jiang; Jian Yang; Ying Tai
>
> **摘要:** This paper explores the task of Complex Visual Text Generation (CVTG), which centers on generating intricate textual content distributed across diverse regions within visual images. In CVTG, image generation models often rendering distorted and blurred visual text or missing some visual text. To tackle these challenges, we propose TextCrafter, a novel multi-visual text rendering method. TextCrafter employs a progressive strategy to decompose complex visual text into distinct components while ensuring robust alignment between textual content and its visual carrier. Additionally, it incorporates a token focus enhancement mechanism to amplify the prominence of visual text during the generation process. TextCrafter effectively addresses key challenges in CVTG tasks, such as text confusion, omissions, and blurriness. Moreover, we present a new benchmark dataset, CVTG-2K, tailored to rigorously evaluate the performance of generative models on CVTG tasks. Extensive experiments demonstrate that our method surpasses state-of-the-art approaches.
>
---
#### [replaced 030] Learning Gentle Grasping Using Vision, Sound, and Touch
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.07926v2](http://arxiv.org/pdf/2503.07926v2)**

> **作者:** Ken Nakahara; Roberto Calandra
>
> **备注:** 8 pages. Accepted by 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** In our daily life, we often encounter objects that are fragile and can be damaged by excessive grasping force, such as fruits. For these objects, it is paramount to grasp gently -- not using the maximum amount of force possible, but rather the minimum amount of force necessary. This paper proposes using visual, tactile, and auditory signals to learn to grasp and regrasp objects stably and gently. Specifically, we use audio signals as an indicator of gentleness during the grasping, and then train an end-to-end action-conditional model from raw visuo-tactile inputs that predicts both the stability and the gentleness of future grasping candidates, thus allowing the selection and execution of the most promising action. Experimental results on a multi-fingered hand over 1,500 grasping trials demonstrated that our model is useful for gentle grasping by validating the predictive performance (3.27% higher accuracy than the vision-only variant) and providing interpretations of their behavior. Finally, real-world experiments confirmed that the grasping performance with the trained multi-modal model outperformed other baselines (17% higher rate for stable and gentle grasps than vision-only). Our approach requires neither tactile sensor calibration nor analytical force modeling, drastically reducing the engineering effort to grasp fragile objects. Dataset and videos are available at https://lasr.org/research/gentle-grasping.
>
---
#### [replaced 031] ViLU: Learning Vision-Language Uncertainties for Failure Prediction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.07620v3](http://arxiv.org/pdf/2507.07620v3)**

> **作者:** Marc Lafon; Yannis Karmim; Julio Silva-Rodríguez; Paul Couairon; Clément Rambour; Raphaël Fournier-Sniehotta; Ismail Ben Ayed; Jose Dolz; Nicolas Thome
>
> **摘要:** Reliable Uncertainty Quantification (UQ) and failure prediction remain open challenges for Vision-Language Models (VLMs). We introduce ViLU, a new Vision-Language Uncertainty quantification framework that contextualizes uncertainty estimates by leveraging all task-relevant textual representations. ViLU constructs an uncertainty-aware multi-modal representation by integrating the visual embedding, the predicted textual embedding, and an image-conditioned textual representation via cross-attention. Unlike traditional UQ methods based on loss prediction, ViLU trains an uncertainty predictor as a binary classifier to distinguish correct from incorrect predictions using a weighted binary cross-entropy loss, making it loss-agnostic. In particular, our proposed approach is well-suited for post-hoc settings, where only vision and text embeddings are available without direct access to the model itself. Extensive experiments on diverse datasets show the significant gains of our method compared to state-of-the-art failure prediction methods. We apply our method to standard classification datasets, such as ImageNet-1k, as well as large-scale image-caption datasets like CC12M and LAION-400M. Ablation studies highlight the critical role of our architecture and training in achieving effective uncertainty quantification. Our code is publicly available and can be found here: https://github.com/ykrmm/ViLU.
>
---
#### [replaced 032] MLRU++: Multiscale Lightweight Residual UNETR++ with Attention for Efficient 3D Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.16122v2](http://arxiv.org/pdf/2507.16122v2)**

> **作者:** Nand Kumar Yadav; Rodrigue Rizk; William CW Chen; KC
>
> **摘要:** Accurate and efficient medical image segmentation is crucial but challenging due to anatomical variability and high computational demands on volumetric data. Recent hybrid CNN-Transformer architectures achieve state-of-the-art results but add significant complexity. In this paper, we propose MLRU++, a Multiscale Lightweight Residual UNETR++ architecture designed to balance segmentation accuracy and computational efficiency. It introduces two key innovations: a Lightweight Channel and Bottleneck Attention Module (LCBAM) that enhances contextual feature encoding with minimal overhead, and a Multiscale Bottleneck Block (M2B) in the decoder that captures fine-grained details via multi-resolution feature aggregation. Experiments on four publicly available benchmark datasets (Synapse, BTCV, ACDC, and Decathlon Lung) demonstrate that MLRU++ achieves state-of-the-art performance, with average Dice scores of 87.57% (Synapse), 93.00% (ACDC), and 81.12% (Lung). Compared to existing leading models, MLRU++ improves Dice scores by 5.38% and 2.12% on Synapse and ACDC, respectively, while significantly reducing parameter count and computational cost. Ablation studies evaluating LCBAM and M2B further confirm the effectiveness of the proposed architectural components. Results suggest that MLRU++ offers a practical and high-performing solution for 3D medical image segmentation tasks. Source code is available at: https://github.com/1027865/MLRUPP
>
---
#### [replaced 033] ODES: Domain Adaptation with Expert Guidance for Online Medical Image Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2312.05407v3](http://arxiv.org/pdf/2312.05407v3)**

> **作者:** Md Shazid Islam; Sayak Nag; Arindam Dutta; Miraj Ahmed; Fahim Faisal Niloy; Amit K. Roy-Chowdhury
>
> **摘要:** Unsupervised domain adaptive segmentation typically relies on self-training using pseudo labels predicted by a pre-trained network on an unlabeled target dataset. However, the noisy nature of such pseudo-labels presents a major bottleneck in adapting a network to the distribution shift between source and target datasets. This challenge is exaggerated when the network encounters an incoming data stream in online fashion, where the network is constrained to adapt to incoming streams of target domain data in exactly one round of forward and backward passes. In this scenario, relying solely on inaccurate pseudo-labels can lead to low-quality segmentation, which is detrimental to medical image analysis where accuracy and precision are of utmost priority. We hypothesize that a small amount of pixel-level annotation obtained from an expert can address this problem, thereby enhancing the performance of domain adaptation of online streaming data, even in the absence of dedicated training data. We call our method ODES: Domain Adaptation with Expert Guidance for Online Medical Image Segmentation that adapts to each incoming data batch in an online setup, incorporating feedback from an expert through active learning. Through active learning, the most informative pixels in each image can be selected for expert annotation. However, the acquisition of pixel-level annotations across all images in a batch often leads to redundant information while increasing temporal overhead in online learning. To reduce the annotation acquisition time and make the adaptation process more online-friendly, we further propose a novel image-pruning strategy that selects the most useful subset of images from the current batch for active learning. Our proposed approach outperforms existing online adaptation approaches and produces competitive results compared to offline domain adaptive active learning methods.
>
---
#### [replaced 034] Robust Multi-View Learning via Representation Fusion of Sample-Level Attention and Alignment of Simulated Perturbation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.04151v2](http://arxiv.org/pdf/2503.04151v2)**

> **作者:** Jie Xu; Na Zhao; Gang Niu; Masashi Sugiyama; Xiaofeng Zhu
>
> **摘要:** Recently, multi-view learning (MVL) has garnered significant attention due to its ability to fuse discriminative information from multiple views. However, real-world multi-view datasets are often heterogeneous and imperfect, which usually causes MVL methods designed for specific combinations of views to lack application potential and limits their effectiveness. To address this issue, we propose a novel robust MVL method (namely RML) with simultaneous representation fusion and alignment. Specifically, we introduce a simple yet effective multi-view transformer fusion network where we transform heterogeneous multi-view data into homogeneous word embeddings, and then integrate multiple views by the sample-level attention mechanism to obtain a fused representation. Furthermore, we propose a simulated perturbation based multi-view contrastive learning framework that dynamically generates the noise and unusable perturbations for simulating imperfect data conditions. The simulated noisy and unusable data obtain two distinct fused representations, and we utilize contrastive learning to align them for learning discriminative and robust representations. Our RML is self-supervised and can also be applied for downstream tasks as a regularization. In experiments, we employ it in multi-view unsupervised clustering, noise-label classification, and as a plug-and-play module for cross-modal hashing retrieval. Extensive comparison experiments and ablation studies validate RML's effectiveness. Code is available at https://github.com/SubmissionsIn/RML.
>
---
#### [replaced 035] AI Workflow, External Validation, and Development in Eye Disease Diagnosis
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.15087v2](http://arxiv.org/pdf/2409.15087v2)**

> **作者:** Qingyu Chen; Tiarnan D L Keenan; Elvira Agron; Alexis Allot; Emily Guan; Bryant Duong; Amr Elsawy; Benjamin Hou; Cancan Xue; Sanjeeb Bhandari; Geoffrey Broadhead; Chantal Cousineau-Krieger; Ellen Davis; William G Gensheimer; David Grasic; Seema Gupta; Luis Haddock; Eleni Konstantinou; Tania Lamba; Michele Maiberger; Dimosthenis Mantopoulos; Mitul C Mehta; Ayman G Nahri; Mutaz AL-Nawaflh; Arnold Oshinsky; Brittany E Powell; Boonkit Purt; Soo Shin; Hillary Stiefel; Alisa T Thavikulwat; Keith James Wroblewski; Tham Yih Chung; Chui Ming Gemmy Cheung; Ching-Yu Cheng; Emily Y Chew; Michelle R. Hribar; Michael F. Chiang; Zhiyong Lu
>
> **备注:** Published in JAMA Network Open, doi:10.1001/jamanetworkopen.2025.17204
>
> **摘要:** Timely disease diagnosis is challenging due to increasing disease burdens and limited clinician availability. AI shows promise in diagnosis accuracy but faces real-world application issues due to insufficient validation in clinical workflows and diverse populations. This study addresses gaps in medical AI downstream accountability through a case study on age-related macular degeneration (AMD) diagnosis and severity classification. We designed and implemented an AI-assisted diagnostic workflow for AMD, comparing diagnostic performance with and without AI assistance among 24 clinicians from 12 institutions with real patient data sampled from the Age-Related Eye Disease Study (AREDS). Additionally, we demonstrated continual enhancement of an existing AI model by incorporating approximately 40,000 additional medical images (named AREDS2 dataset). The improved model was then systematically evaluated using both AREDS and AREDS2 test sets, as well as an external test set from Singapore. AI assistance markedly enhanced diagnostic accuracy and classification for 23 out of 24 clinicians, with the average F1-score increasing by 20% from 37.71 (Manual) to 45.52 (Manual + AI) (P-value < 0.0001), achieving an improvement of over 50% in some cases. In terms of efficiency, AI assistance reduced diagnostic times for 17 out of the 19 clinicians tracked, with time savings of up to 40%. Furthermore, a model equipped with continual learning showed robust performance across three independent datasets, recording a 29% increase in accuracy, and elevating the F1-score from 42 to 54 in the Singapore population.
>
---
#### [replaced 036] When Large Vision-Language Model Meets Large Remote Sensing Imagery: Coarse-to-Fine Text-Guided Token Pruning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.07588v3](http://arxiv.org/pdf/2503.07588v3)**

> **作者:** Junwei Luo; Yingying Zhang; Xue Yang; Kang Wu; Qi Zhu; Lei Liang; Jingdong Chen; Yansheng Li
>
> **备注:** 18 pages, 6 figures, 18 tables
>
> **摘要:** Efficient vision-language understanding of large Remote Sensing Images (RSIs) is meaningful but challenging. Current Large Vision-Language Models (LVLMs) typically employ limited pre-defined grids to process images, leading to information loss when handling gigapixel RSIs. Conversely, using unlimited grids significantly increases computational costs. To preserve image details while reducing computational complexity, we propose a text-guided token pruning method with Dynamic Image Pyramid (DIP) integration. Our method introduces: (i) a Region Focus Module (RFM) that leverages text-aware region localization capability to identify critical vision tokens, and (ii) a coarse-to-fine image tile selection and vision token pruning strategy based on DIP, which is guided by RFM outputs and avoids directly processing the entire large imagery. Additionally, existing benchmarks for evaluating LVLMs' perception ability on large RSI suffer from limited question diversity and constrained image sizes. We construct a new benchmark named LRS-VQA, which contains 7,333 QA pairs across 8 categories, with image length up to 27,328 pixels. Our method outperforms existing high-resolution strategies on four datasets using the same data. Moreover, compared to existing token reduction methods, our approach demonstrates higher efficiency under high-resolution settings. Dataset and code are in https://github.com/VisionXLab/LRS-VQA.
>
---
#### [replaced 037] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.13176v3](http://arxiv.org/pdf/2503.13176v3)**

> **作者:** Rui Wang; Quentin Lohmeyer; Mirko Meboldt; Siyu Tang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Reconstructing clean, distractor-free 3D scenes from real-world captures remains a significant challenge, particularly in highly dynamic and cluttered settings such as egocentric videos. To tackle this problem, we introduce DeGauss, a simple and robust self-supervised framework for dynamic scene reconstruction based on a decoupled dynamic-static Gaussian Splatting design. DeGauss models dynamic elements with foreground Gaussians and static content with background Gaussians, using a probabilistic mask to coordinate their composition and enable independent yet complementary optimization. DeGauss generalizes robustly across a wide range of real-world scenarios, from casual image collections to long, dynamic egocentric videos, without relying on complex heuristics or extensive supervision. Experiments on benchmarks including NeRF-on-the-go, ADT, AEA, Hot3D, and EPIC-Fields demonstrate that DeGauss consistently outperforms existing methods, establishing a strong baseline for generalizable, distractor-free 3D reconstructionin highly dynamic, interaction-rich environments. Project page: https://batfacewayne.github.io/DeGauss.io/
>
---
#### [replaced 038] Faithful, Interpretable Chest X-ray Diagnosis with Anti-Aliased B-cos Networks
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.16761v2](http://arxiv.org/pdf/2507.16761v2)**

> **作者:** Marcel Kleinmann; Shashank Agnihotri; Margret Keuper
>
> **摘要:** Faithfulness and interpretability are essential for deploying deep neural networks (DNNs) in safety-critical domains such as medical imaging. B-cos networks offer a promising solution by replacing standard linear layers with a weight-input alignment mechanism, producing inherently interpretable, class-specific explanations without post-hoc methods. While maintaining diagnostic performance competitive with state-of-the-art DNNs, standard B-cos models suffer from severe aliasing artifacts in their explanation maps, making them unsuitable for clinical use where clarity is essential. In this work, we address these limitations by introducing anti-aliasing strategies using FLCPooling (FLC) and BlurPool (BP) to significantly improve explanation quality. Our experiments on chest X-ray datasets demonstrate that the modified $\text{B-cos}_\text{FLC}$ and $\text{B-cos}_\text{BP}$ preserve strong predictive performance while providing faithful and artifact-free explanations suitable for clinical application in multi-class and multi-label settings. Code available at: GitHub repository (url: https://github.com/mkleinma/B-cos-medical-paper).
>
---
#### [replaced 039] Optimizing against Infeasible Inclusions from Data for Semantic Segmentation through Morphology
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.14672v5](http://arxiv.org/pdf/2408.14672v5)**

> **作者:** Shamik Basu; Luc Van Gool; Christos Sakaridis
>
> **摘要:** State-of-the-art semantic segmentation models are typically optimized in a data-driven fashion, minimizing solely per-pixel or per-segment classification objectives on their training data. This purely data-driven paradigm often leads to absurd segmentations, especially when the domain of input images is shifted from the one encountered during training. For instance, state-of-the-art models may assign the label "road" to a segment that is included by another segment that is respectively labeled as "sky". However, the ground truth of the existing dataset at hand dictates that such inclusion is not feasible. Our method, Infeasible Semantic Inclusions (InSeIn), first extracts explicit inclusion constraints that govern spatial class relations from the semantic segmentation training set at hand in an offline, data-driven fashion, and then enforces a morphological yet differentiable loss that penalizes violations of these constraints during training to promote prediction feasibility. InSeIn is a light-weight plug-and-play method, constitutes a novel step towards minimizing infeasible semantic inclusions in the predictions of learned segmentation models, and yields consistent and significant performance improvements over diverse state-of-the-art networks across the ADE20K, Cityscapes, and ACDC datasets. https://github.com/SHAMIK-97/InSeIn/tree/main
>
---
#### [replaced 040] Flash-VStream: Efficient Real-Time Understanding for Long Video Streams
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.23825v2](http://arxiv.org/pdf/2506.23825v2)**

> **作者:** Haoji Zhang; Yiqin Wang; Yansong Tang; Yong Liu; Jiashi Feng; Xiaojie Jin
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Benefiting from the advances in large language models and cross-modal alignment, existing multimodal large language models have achieved prominent performance in image and short video understanding. However, the understanding of long videos is still challenging, as their long-context nature results in significant computational and memory overhead. Most existing work treats long videos in the same way as short videos, which is inefficient for real-world applications and hard to generalize to even longer videos. To address these issues, we propose Flash-VStream, an efficient video language model capable of processing extremely long videos and responding to user queries in real time. Particularly, we design a Flash Memory module, containing a low-capacity context memory to aggregate long-context temporal information and model the distribution of information density, and a high-capacity augmentation memory to retrieve detailed spatial information based on this distribution. Compared to existing models, Flash-VStream achieves significant reductions in inference latency. Extensive experiments on long video benchmarks and comprehensive video benchmarks, i.e., EgoSchema, MLVU, LVBench, MVBench and Video-MME, demonstrate the state-of-the-art performance and outstanding efficiency of our method. Code is available at https://github.com/IVGSZ/Flash-VStream.
>
---
#### [replaced 041] PALADIN : Robust Neural Fingerprinting for Text-to-Image Diffusion Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.03170v2](http://arxiv.org/pdf/2506.03170v2)**

> **作者:** Murthy L; Subarna Tripathi
>
> **摘要:** The risk of misusing text-to-image generative models for malicious uses, especially due to the open-source development of such models, has become a serious concern. As a risk mitigation strategy, attributing generative models with neural fingerprinting is emerging as a popular technique. There has been a plethora of recent work that aim for addressing neural fingerprinting. A trade-off between the attribution accuracy and generation quality of such models has been studied extensively. None of the existing methods yet achieved 100% attribution accuracy. However, any model with less than cent percent accuracy is practically non-deployable. In this work, we propose an accurate method to incorporate neural fingerprinting for text-to-image diffusion models leveraging the concepts of cyclic error correcting codes from the literature of coding theory.
>
---
#### [replaced 042] A Survey of Deep Learning for Geometry Problem Solving
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.11936v3](http://arxiv.org/pdf/2507.11936v3)**

> **作者:** Jianzhe Ma; Wenxuan Wang; Qin Jin
>
> **备注:** Work in progress
>
> **摘要:** Geometry problem solving is a key area of mathematical reasoning, which is widely involved in many important fields such as education, mathematical ability assessment of artificial intelligence, and multimodal ability assessment. In recent years, the rapid development of deep learning technology, especially the rise of multimodal large language models, has triggered a widespread research boom. This paper provides a survey of the applications of deep learning in geometry problem solving, including (i) a comprehensive summary of the relevant tasks in geometry problem solving; (ii) a thorough review of related deep learning methods; (iii) a detailed analysis of evaluation metrics and methods; and (iv) a critical discussion of the current challenges and future directions that can be explored. Our goal is to provide a comprehensive and practical reference of deep learning for geometry problem solving to promote further developments in this field. We create a continuously updated list of papers on GitHub: https://github.com/majianz/dl4gps.
>
---
#### [replaced 043] Label Anything: Multi-Class Few-Shot Semantic Segmentation with Visual Prompts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.02075v2](http://arxiv.org/pdf/2407.02075v2)**

> **作者:** Pasquale De Marinis; Nicola Fanelli; Raffaele Scaringi; Emanuele Colonna; Giuseppe Fiameni; Gennaro Vessio; Giovanna Castellano
>
> **摘要:** We present Label Anything, an innovative neural network architecture designed for few-shot semantic segmentation (FSS) that demonstrates remarkable generalizability across multiple classes with minimal examples required per class. Diverging from traditional FSS methods that predominantly rely on masks for annotating support images, Label Anything introduces varied visual prompts -- points, bounding boxes, and masks -- thereby enhancing the framework's versatility and adaptability. Unique to our approach, Label Anything is engineered for end-to-end training across multi-class FSS scenarios, efficiently learning from diverse support set configurations without retraining. This approach enables a "universal" application to various FSS challenges, ranging from $1$-way $1$-shot to complex $N$-way $K$-shot configurations while remaining agnostic to the specific number of class examples. This innovative training strategy reduces computational requirements and substantially improves the model's adaptability and generalization across diverse segmentation tasks. Our comprehensive experimental validation, particularly achieving state-of-the-art results on the COCO-$20^i$ benchmark, underscores Label Anything's robust generalization and flexibility. The source code is publicly available at: https://github.com/pasqualedem/LabelAnything.
>
---
#### [replaced 044] Rethinking Occlusion in FER: A Semantic-Aware Perspective and Go Beyond
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.15401v3](http://arxiv.org/pdf/2507.15401v3)**

> **作者:** Huiyu Zhai; Xingxing Yang; Yalan Ye; Chenyang Li; Bin Fan; Changze Li
>
> **摘要:** Facial expression recognition (FER) is a challenging task due to pervasive occlusion and dataset biases. Especially when facial information is partially occluded, existing FER models struggle to extract effective facial features, leading to inaccurate classifications. In response, we present ORSANet, which introduces the following three key contributions: First, we introduce auxiliary multi-modal semantic guidance to disambiguate facial occlusion and learn high-level semantic knowledge, which is two-fold: 1) we introduce semantic segmentation maps as dense semantics prior to generate semantics-enhanced facial representations; 2) we introduce facial landmarks as sparse geometric prior to mitigate intrinsic noises in FER, such as identity and gender biases. Second, to facilitate the effective incorporation of these two multi-modal priors, we customize a Multi-scale Cross-interaction Module (MCM) to adaptively fuse the landmark feature and semantics-enhanced representations within different scales. Third, we design a Dynamic Adversarial Repulsion Enhancement Loss (DARELoss) that dynamically adjusts the margins of ambiguous classes, further enhancing the model's ability to distinguish similar expressions. We further construct the first occlusion-oriented FER dataset to facilitate specialized robustness analysis on various real-world occlusion conditions, dubbed Occlu-FER. Extensive experiments on both public benchmarks and Occlu-FER demonstrate that our proposed ORSANet achieves SOTA recognition performance. Code is publicly available at https://github.com/Wenyuzhy/ORSANet-master.
>
---
#### [replaced 045] Unsupervised Feature Disentanglement and Augmentation Network for One-class Face Anti-spoofing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.22929v2](http://arxiv.org/pdf/2503.22929v2)**

> **作者:** Pei-Kai Huang; Jun-Xiong Chong; Ming-Tsung Hsu; Fang-Yu Hsu; Yi-Ting Lin; Kai-Heng Chien; Hao-Chiang Shao; Chiou-Ting Hsu
>
> **摘要:** Face anti-spoofing (FAS) techniques aim to enhance the security of facial identity authentication by distinguishing authentic live faces from deceptive attempts. While two-class FAS methods risk overfitting to training attacks to achieve better performance, one-class FAS approaches handle unseen attacks well but are less robust to domain information entangled within the liveness features. To address this, we propose an Unsupervised Feature Disentanglement and Augmentation Network (\textbf{UFDANet}), a one-class FAS technique that enhances generalizability by augmenting face images via disentangled features. The \textbf{UFDANet} employs a novel unsupervised feature disentangling method to separate the liveness and domain features, facilitating discriminative feature learning. It integrates an out-of-distribution liveness feature augmentation scheme to synthesize new liveness features of unseen spoof classes, which deviate from the live class, thus enhancing the representability and discriminability of liveness features. Additionally, \textbf{UFDANet} incorporates a domain feature augmentation routine to synthesize unseen domain features, thereby achieving better generalizability. Extensive experiments demonstrate that the proposed \textbf{UFDANet} outperforms previous one-class FAS methods and achieves comparable performance to state-of-the-art two-class FAS methods.
>
---
#### [replaced 046] Diffuse and Disperse: Image Generation with Representation Regularization
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.09027v2](http://arxiv.org/pdf/2506.09027v2)**

> **作者:** Runqian Wang; Kaiming He
>
> **摘要:** The development of diffusion-based generative models over the past decade has largely proceeded independently of progress in representation learning. These diffusion models typically rely on regression-based objectives and generally lack explicit regularization. In this work, we propose \textit{Dispersive Loss}, a simple plug-and-play regularizer that effectively improves diffusion-based generative models. Our loss function encourages internal representations to disperse in the hidden space, analogous to contrastive self-supervised learning, with the key distinction that it requires no positive sample pairs and therefore does not interfere with the sampling process used for regression. Compared to the recent method of representation alignment (REPA), our approach is self-contained and minimalist, requiring no pre-training, no additional parameters, and no external data. We evaluate Dispersive Loss on the ImageNet dataset across a range of models and report consistent improvements over widely used and strong baselines. We hope our work will help bridge the gap between generative modeling and representation learning.
>
---
#### [replaced 047] MAD-AD: Masked Diffusion for Unsupervised Brain Anomaly Detection
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2502.16943v3](http://arxiv.org/pdf/2502.16943v3)**

> **作者:** Farzad Beizaee; Gregory Lodygensky; Christian Desrosiers; Jose Dolz
>
> **摘要:** Unsupervised anomaly detection in brain images is crucial for identifying injuries and pathologies without access to labels. However, the accurate localization of anomalies in medical images remains challenging due to the inherent complexity and variability of brain structures and the scarcity of annotated abnormal data. To address this challenge, we propose a novel approach that incorporates masking within diffusion models, leveraging their generative capabilities to learn robust representations of normal brain anatomy. During training, our model processes only normal brain MRI scans and performs a forward diffusion process in the latent space that adds noise to the features of randomly-selected patches. Following a dual objective, the model learns to identify which patches are noisy and recover their original features. This strategy ensures that the model captures intricate patterns of normal brain structures while isolating potential anomalies as noise in the latent space. At inference, the model identifies noisy patches corresponding to anomalies and generates a normal counterpart for these patches by applying a reverse diffusion process. Our method surpasses existing unsupervised anomaly detection techniques, demonstrating superior performance in generating accurate normal counterparts and localizing anomalies. The code is available at hhttps://github.com/farzad-bz/MAD-AD.
>
---
#### [replaced 048] Learning to Generalize without Bias for Open-Vocabulary Action Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.20158v2](http://arxiv.org/pdf/2502.20158v2)**

> **作者:** Yating Yu; Congqi Cao; Yifan Zhang; Yanning Zhang
>
> **备注:** Accepted by ICCV2025 (Highlight)
>
> **摘要:** Leveraging the effective visual-text alignment and static generalizability from CLIP, recent video learners adopt CLIP initialization with further regularization or recombination for generalization in open-vocabulary action recognition in-context. However, due to the static bias of CLIP, such video learners tend to overfit on shortcut static features, thereby compromising their generalizability, especially to novel out-of-context actions. To address this issue, we introduce Open-MeDe, a novel Meta-optimization framework with static Debiasing for Open-vocabulary action recognition. From a fresh perspective of generalization, Open-MeDe adopts a meta-learning approach to improve known-to-open generalizing and image-to-video debiasing in a cost-effective manner. Specifically, Open-MeDe introduces a cross-batch meta-optimization scheme that explicitly encourages video learners to quickly generalize to arbitrary subsequent data via virtual evaluation, steering a smoother optimization landscape. In effect, the free of CLIP regularization during optimization implicitly mitigates the inherent static bias of the video meta-learner. We further apply self-ensemble over the optimization trajectory to obtain generic optimal parameters that can achieve robust generalization to both in-context and out-of-context novel data. Extensive evaluations show that Open-MeDe not only surpasses state-of-the-art regularization methods tailored for in-context open-vocabulary action recognition but also substantially excels in out-of-context scenarios.Code is released at https://github.com/Mia-YatingYu/Open-MeDe.
>
---
#### [replaced 049] Frequency-Dynamic Attention Modulation for Dense Prediction
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.12006v3](http://arxiv.org/pdf/2507.12006v3)**

> **作者:** Linwei Chen; Lin Gu; Ying Fu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Vision Transformers (ViTs) have significantly advanced computer vision, demonstrating strong performance across various tasks. However, the attention mechanism in ViTs makes each layer function as a low-pass filter, and the stacked-layer architecture in existing transformers suffers from frequency vanishing. This leads to the loss of critical details and textures. We propose a novel, circuit-theory-inspired strategy called Frequency-Dynamic Attention Modulation (FDAM), which can be easily plugged into ViTs. FDAM directly modulates the overall frequency response of ViTs and consists of two techniques: Attention Inversion (AttInv) and Frequency Dynamic Scaling (FreqScale). Since circuit theory uses low-pass filters as fundamental elements, we introduce AttInv, a method that generates complementary high-pass filtering by inverting the low-pass filter in the attention matrix, and dynamically combining the two. We further design FreqScale to weight different frequency components for fine-grained adjustments to the target response function. Through feature similarity analysis and effective rank evaluation, we demonstrate that our approach avoids representation collapse, leading to consistent performance improvements across various models, including SegFormer, DeiT, and MaskDINO. These improvements are evident in tasks such as semantic segmentation, object detection, and instance segmentation. Additionally, we apply our method to remote sensing detection, achieving state-of-the-art results in single-scale settings. The code is available at https://github.com/Linwei-Chen/FDAM.
>
---
#### [replaced 050] Trigger without Trace: Towards Stealthy Backdoor Attack on Text-to-Image Diffusion Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.17724v2](http://arxiv.org/pdf/2503.17724v2)**

> **作者:** Jie Zhang; Zhongqi Wang; Shiguang Shan; Xilin Chen
>
> **摘要:** Backdoor attacks targeting text-to-image diffusion models have advanced rapidly. However, current backdoor samples often exhibit two key abnormalities compared to benign samples: 1) Semantic Consistency, where backdoor prompts tend to generate images with similar semantic content even with significant textual variations to the prompts; 2) Attention Consistency, where the trigger induces consistent structural responses in the cross-attention maps. These consistencies leave detectable traces for defenders, making backdoors easier to identify. In this paper, toward stealthy backdoor samples, we propose Trigger without Trace (TwT) by explicitly mitigating these consistencies. Specifically, our approach leverages syntactic structures as backdoor triggers to amplify the sensitivity to textual variations, effectively breaking down the semantic consistency. Besides, a regularization method based on Kernel Maximum Mean Discrepancy (KMMD) is proposed to align the distribution of cross-attention responses between backdoor and benign samples, thereby disrupting attention consistency. Extensive experiments demonstrate that our method achieves a 97.5% attack success rate while exhibiting stronger resistance to defenses. It achieves an average of over 98% backdoor samples bypassing three state-of-the-art detection mechanisms, revealing the vulnerabilities of current backdoor defense methods. The code is available at https://github.com/Robin-WZQ/TwT.
>
---
#### [replaced 051] Towards a Universal 3D Medical Multi-modality Generalization via Learning Personalized Invariant Representation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.06106v4](http://arxiv.org/pdf/2411.06106v4)**

> **作者:** Zhaorui Tan; Xi Yang; Tan Pan; Tianyi Liu; Chen Jiang; Xin Guo; Qiufeng Wang; Anh Nguyen; Yuan Qi; Kaizhu Huang; Yuan Cheng
>
> **备注:** Accepted by ICCV25
>
> **摘要:** Variations in medical imaging modalities and individual anatomical differences pose challenges to cross-modality generalization in multi-modal tasks. Existing methods often concentrate exclusively on common anatomical patterns, thereby neglecting individual differences and consequently limiting their generalization performance. This paper emphasizes the critical role of learning individual-level invariance, i.e., personalized representation $\mathbb{X}_h$, to enhance multi-modality generalization under both homogeneous and heterogeneous settings. It reveals that mappings from individual biological profile to different medical modalities remain static across the population, which is implied in the personalization process. We propose a two-stage approach: pre-training with invariant representation $\mathbb{X}_h$ for personalization, then fine-tuning for diverse downstream tasks. We provide both theoretical and empirical evidence demonstrating the feasibility and advantages of personalization, showing that our approach yields greater generalizability and transferability across diverse multi-modal medical tasks compared to methods lacking personalization. Extensive experiments further validate that our approach significantly enhances performance in various generalization scenarios.
>
---
#### [replaced 052] QR-LoRA: Efficient and Disentangled Fine-tuning via QR Decomposition for Customized Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.04599v2](http://arxiv.org/pdf/2507.04599v2)**

> **作者:** Jiahui Yang; Yongjia Ma; Donglin Di; Hao Li; Wei Chen; Yan Xie; Jianxun Cui; Xun Yang; Wangmeng Zuo
>
> **备注:** ICCV 2025, 30 pages, 26 figures
>
> **摘要:** Existing text-to-image models often rely on parameter fine-tuning techniques such as Low-Rank Adaptation (LoRA) to customize visual attributes. However, when combining multiple LoRA models for content-style fusion tasks, unstructured modifications of weight matrices often lead to undesired feature entanglement between content and style attributes. We propose QR-LoRA, a novel fine-tuning framework leveraging QR decomposition for structured parameter updates that effectively separate visual attributes. Our key insight is that the orthogonal Q matrix naturally minimizes interference between different visual features, while the upper triangular R matrix efficiently encodes attribute-specific transformations. Our approach fixes both Q and R matrices while only training an additional task-specific $\Delta R$ matrix. This structured design reduces trainable parameters to half of conventional LoRA methods and supports effective merging of multiple adaptations without cross-contamination due to the strong disentanglement properties between $\Delta R$ matrices. Experiments demonstrate that QR-LoRA achieves superior disentanglement in content-style fusion tasks, establishing a new paradigm for parameter-efficient, disentangled fine-tuning in generative models. The project page is available at: https://luna-ai-lab.github.io/QR-LoRA/.
>
---
#### [replaced 053] Towards Holistic Surgical Scene Graph
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.15541v2](http://arxiv.org/pdf/2507.15541v2)**

> **作者:** Jongmin Shin; Enki Cho; Ka Young Kim; Jung Yong Kim; Seong Tae Kim; Namkee Oh
>
> **备注:** Accepted to MICCAI 2025
>
> **摘要:** Surgical scene understanding is crucial for computer-assisted intervention systems, requiring visual comprehension of surgical scenes that involves diverse elements such as surgical tools, anatomical structures, and their interactions. To effectively represent the complex information in surgical scenes, graph-based approaches have been explored to structurally model surgical entities and their relationships. Previous surgical scene graph studies have demonstrated the feasibility of representing surgical scenes using graphs. However, certain aspects of surgical scenes-such as diverse combinations of tool-action-target and the identity of the hand operating the tool-remain underexplored in graph-based representations, despite their importance. To incorporate these aspects into graph representations, we propose Endoscapes-SG201 dataset, which includes annotations for tool-action-target combinations and hand identity. We also introduce SSG-Com, a graph-based method designed to learn and represent these critical elements. Through experiments on downstream tasks such as critical view of safety assessment and action triplet recognition, we demonstrated the importance of integrating these essential scene graph components, highlighting their significant contribution to surgical scene understanding. The code and dataset are available at https://github.com/ailab-kyunghee/SSG-Com
>
---
#### [replaced 054] I-CEE: Tailoring Explanations of Image Classification Models to User Expertise
- **分类: cs.AI; cs.CV; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2312.12102v3](http://arxiv.org/pdf/2312.12102v3)**

> **作者:** Yao Rong; Peizhu Qian; Vaibhav Unhelkar; Enkelejda Kasneci
>
> **摘要:** Effectively explaining decisions of black-box machine learning models is critical to responsible deployment of AI systems that rely on them. Recognizing their importance, the field of explainable AI (XAI) provides several techniques to generate these explanations. Yet, there is relatively little emphasis on the user (the explainee) in this growing body of work and most XAI techniques generate "one-size-fits-all" explanations. To bridge this gap and achieve a step closer towards human-centered XAI, we present I-CEE, a framework that provides Image Classification Explanations tailored to User Expertise. Informed by existing work, I-CEE explains the decisions of image classification models by providing the user with an informative subset of training data (i.e., example images), corresponding local explanations, and model decisions. However, unlike prior work, I-CEE models the informativeness of the example images to depend on user expertise, resulting in different examples for different users. We posit that by tailoring the example set to user expertise, I-CEE can better facilitate users' understanding and simulatability of the model. To evaluate our approach, we conduct detailed experiments in both simulation and with human participants (N = 100) on multiple datasets. Experiments with simulated users show that I-CEE improves users' ability to accurately predict the model's decisions (simulatability) compared to baselines, providing promising preliminary results. Experiments with human participants demonstrate that our method significantly improves user simulatability accuracy, highlighting the importance of human-centered XAI
>
---
#### [replaced 055] Outdoor Monocular SLAM with Global Scale-Consistent 3D Gaussian Pointmaps
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.03737v2](http://arxiv.org/pdf/2507.03737v2)**

> **作者:** Chong Cheng; Sicheng Yu; Zijian Wang; Yifan Zhou; Hao Wang
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** 3D Gaussian Splatting (3DGS) has become a popular solution in SLAM due to its high-fidelity and real-time novel view synthesis performance. However, some previous 3DGS SLAM methods employ a differentiable rendering pipeline for tracking, lack geometric priors in outdoor scenes. Other approaches introduce separate tracking modules, but they accumulate errors with significant camera movement, leading to scale drift. To address these challenges, we propose a robust RGB-only outdoor 3DGS SLAM method: S3PO-GS. Technically, we establish a self-consistent tracking module anchored in the 3DGS pointmap, which avoids cumulative scale drift and achieves more precise and robust tracking with fewer iterations. Additionally, we design a patch-based pointmap dynamic mapping module, which introduces geometric priors while avoiding scale ambiguity. This significantly enhances tracking accuracy and the quality of scene reconstruction, making it particularly suitable for complex outdoor environments. Our experiments on the Waymo, KITTI, and DL3DV datasets demonstrate that S3PO-GS achieves state-of-the-art results in novel view synthesis and outperforms other 3DGS SLAM methods in tracking accuracy. Project page: https://3dagentworld.github.io/S3PO-GS/.
>
---
#### [replaced 056] crossMoDA Challenge: Evolution of Cross-Modality Domain Adaptation Techniques for Vestibular Schwannoma and Cochlea Segmentation from 2021 to 2023
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.12006v3](http://arxiv.org/pdf/2506.12006v3)**

> **作者:** Navodini Wijethilake; Reuben Dorent; Marina Ivory; Aaron Kujawa; Stefan Cornelissen; Patrick Langenhuizen; Mohamed Okasha; Anna Oviedova; Hexin Dong; Bogyeong Kang; Guillaume Sallé; Luyi Han; Ziyuan Zhao; Han Liu; Yubo Fan; Tao Yang; Shahad Hardan; Hussain Alasmawi; Santosh Sanjeev; Yuzhou Zhuang; Satoshi Kondo; Maria Baldeon Calisto; Shaikh Muhammad Uzair Noman; Cancan Chen; Ipek Oguz; Rongguo Zhang; Mina Rezaei; Susana K. Lai-Yuen; Satoshi Kasai; Yunzhi Huang; Chih-Cheng Hung; Mohammad Yaqub; Lisheng Wang; Benoit M. Dawant; Cuntai Guan; Ritse Mann; Vincent Jaouen; Tae-Eui Kam; Li Zhang; Jonathan Shapey; Tom Vercauteren
>
> **摘要:** The cross-Modality Domain Adaptation (crossMoDA) challenge series, initiated in 2021 in conjunction with the International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), focuses on unsupervised cross-modality segmentation, learning from contrast-enhanced T1 (ceT1) and transferring to T2 MRI. The task is an extreme example of domain shift chosen to serve as a meaningful and illustrative benchmark. From a clinical application perspective, it aims to automate Vestibular Schwannoma (VS) and cochlea segmentation on T2 scans for more cost-effective VS management. Over time, the challenge objectives have evolved to enhance its clinical relevance. The challenge evolved from using single-institutional data and basic segmentation in 2021 to incorporating multi-institutional data and Koos grading in 2022, and by 2023, it included heterogeneous routine data and sub-segmentation of intra- and extra-meatal tumour components. In this work, we report the findings of the 2022 and 2023 editions and perform a retrospective analysis of the challenge progression over the years. The observations from the successive challenge contributions indicate that the number of outliers decreases with an expanding dataset. This is notable since the diversity of scanning protocols of the datasets concurrently increased. The winning approach of the 2023 edition reduced the number of outliers on the 2021 and 2022 testing data, demonstrating how increased data heterogeneity can enhance segmentation performance even on homogeneous data. However, the cochlea Dice score declined in 2023, likely due to the added complexity from tumour sub-annotations affecting overall segmentation performance. While progress is still needed for clinically acceptable VS segmentation, the plateauing performance suggests that a more challenging cross-modal task may better serve future benchmarking.
>
---
#### [replaced 057] Benchmarking Cross-Domain Audio-Visual Deception Detection
- **分类: cs.SD; cs.CV; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2405.06995v3](http://arxiv.org/pdf/2405.06995v3)**

> **作者:** Xiaobao Guo; Zitong Yu; Nithish Muthuchamy Selvaraj; Bingquan Shen; Adams Wai-Kin Kong; Alex C. Kot
>
> **备注:** 15 pages
>
> **摘要:** Automated deception detection is crucial for assisting humans in accurately assessing truthfulness and identifying deceptive behavior. Conventional contact-based techniques, like polygraph devices, rely on physiological signals to determine the authenticity of an individual's statements. Nevertheless, recent developments in automated deception detection have demonstrated that multimodal features derived from both audio and video modalities may outperform human observers on publicly available datasets. Despite these positive findings, the generalizability of existing audio-visual deception detection approaches across different scenarios remains largely unexplored. To close this gap, we present the first cross-domain audio-visual deception detection benchmark, that enables us to assess how well these methods generalize for use in real-world scenarios. We used widely adopted audio and visual features and different architectures for benchmarking, comparing single-to-single and multi-to-single domain generalization performance. To further exploit the impacts using data from multiple source domains for training, we investigate three types of domain sampling strategies, including domain-simultaneous, domain-alternating, and domain-by-domain for multi-to-single domain generalization evaluation. We also propose an algorithm to enhance the generalization performance by maximizing the gradient inner products between modality encoders, named ``MM-IDGM". Furthermore, we proposed the Attention-Mixer fusion method to improve performance, and we believe that this new cross-domain benchmark will facilitate future research in audio-visual deception detection.
>
---
#### [replaced 058] FUDOKI: Discrete Flow-based Unified Understanding and Generation via Kinetic-Optimal Velocities
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20147v3](http://arxiv.org/pdf/2505.20147v3)**

> **作者:** Jin Wang; Yao Lai; Aoxue Li; Shifeng Zhang; Jiacheng Sun; Ning Kang; Chengyue Wu; Zhenguo Li; Ping Luo
>
> **备注:** 37 pages, 12 figures
>
> **摘要:** The rapid progress of large language models (LLMs) has catalyzed the emergence of multimodal large language models (MLLMs) that unify visual understanding and image generation within a single framework. However, most existing MLLMs rely on autoregressive (AR) architectures, which impose inherent limitations on future development, such as the raster-scan order in image generation and restricted reasoning abilities in causal context modeling. In this work, we challenge the dominance of AR-based approaches by introducing FUDOKI, a unified multimodal model purely based on discrete flow matching, as an alternative to conventional AR paradigms. By leveraging metric-induced probability paths with kinetic optimal velocities, our framework goes beyond the previous masking-based corruption process, enabling iterative refinement with self-correction capability and richer bidirectional context integration during generation. To mitigate the high cost of training from scratch, we initialize FUDOKI from pre-trained AR-based MLLMs and adaptively transition to the discrete flow matching paradigm. Experimental results show that FUDOKI achieves performance comparable to state-of-the-art AR-based MLLMs across both visual understanding and image generation tasks, highlighting its potential as a foundation for next-generation unified multimodal models. Furthermore, we show that applying test-time scaling techniques to FUDOKI yields significant performance gains, further underscoring its promise for future enhancement through reinforcement learning.
>
---
#### [replaced 059] Advances in 4D Generation: A Survey
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.14501v3](http://arxiv.org/pdf/2503.14501v3)**

> **作者:** Qiaowei Miao; Kehan Li; Jinsheng Quan; Zhiyuan Min; Shaojie Ma; Yichao Xu; Yi Yang; Ping Liu; Yawei Luo
>
> **摘要:** Generative artificial intelligence has recently progressed from static image and video synthesis to 3D content generation, culminating in the emergence of 4D generation-the task of synthesizing temporally coherent dynamic 3D assets guided by user input. As a burgeoning research frontier, 4D generation enables richer interactive and immersive experiences, with applications ranging from digital humans to autonomous driving. Despite rapid progress, the field lacks a unified understanding of 4D representations, generative frameworks, basic paradigms, and the core technical challenges it faces. This survey provides a systematic and in-depth review of the 4D generation landscape. To comprehensively characterize 4D generation, we first categorize fundamental 4D representations and outline associated techniques for 4D generation. We then present an in-depth analysis of representative generative pipelines based on conditions and representation methods. Subsequently, we discuss how motion and geometry priors are integrated into 4D outputs to ensure spatio-temporal consistency under various control schemes. From an application perspective, this paper summarizes 4D generation tasks in areas such as dynamic object/scene generation, digital human synthesis, editable 4D content, and embodied AI. Furthermore, we summarize and multi-dimensionally compare four basic paradigms for 4D generation: End-to-End, Generated-Data-Based, Implicit-Distillation-Based, and Explicit-Supervision-Based. Concluding our analysis, we highlight five key challenges-consistency, controllability, diversity, efficiency, and fidelity-and contextualize these with current approaches.By distilling recent advances and outlining open problems, this work offers a comprehensive and forward-looking perspective to guide future research in 4D generation.
>
---
#### [replaced 060] Leveraging the Structure of Medical Data for Improved Representation Learning
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.02987v3](http://arxiv.org/pdf/2507.02987v3)**

> **作者:** Andrea Agostini; Sonia Laguna; Alain Ryser; Samuel Ruiperez-Campillo; Moritz Vandenhirtz; Nicolas Deperrois; Farhad Nooralahzadeh; Michael Krauthammer; Thomas M. Sutter; Julia E. Vogt
>
> **摘要:** Building generalizable medical AI systems requires pretraining strategies that are data-efficient and domain-aware. Unlike internet-scale corpora, clinical datasets such as MIMIC-CXR offer limited image counts and scarce annotations, but exhibit rich internal structure through multi-view imaging. We propose a self-supervised framework that leverages the inherent structure of medical datasets. Specifically, we treat paired chest X-rays (i.e., frontal and lateral views) as natural positive pairs, learning to reconstruct each view from sparse patches while aligning their latent embeddings. Our method requires no textual supervision and produces informative representations. Evaluated on MIMIC-CXR, we show strong performance compared to supervised objectives and baselines being trained without leveraging structure. This work provides a lightweight, modality-agnostic blueprint for domain-specific pretraining where data is structured but scarce
>
---
#### [replaced 061] Robust sensitivity control in digital pathology via tile score distribution matching
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.20144v3](http://arxiv.org/pdf/2502.20144v3)**

> **作者:** Arthur Pignet; John Klein; Genevieve Robin; Antoine Olivier
>
> **备注:** Camera ready version. Accepted at MICCAI 2025
>
> **摘要:** Deploying digital pathology models across medical centers is challenging due to distribution shifts. Recent advances in domain generalization improve model transferability in terms of aggregated performance measured by the Area Under Curve (AUC). However, clinical regulations often require to control the transferability of other metrics, such as prescribed sensitivity levels. We introduce a novel approach to control the sensitivity of whole slide image (WSI) classification models, based on optimal transport and Multiple Instance Learning (MIL). Validated across multiple cohorts and tasks, our method enables robust sensitivity control with only a handful of calibration samples, providing a practical solution for reliable deployment of computational pathology systems.
>
---
#### [replaced 062] SyncMapV2: Robust and Adaptive Unsupervised Segmentation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.16297v3](http://arxiv.org/pdf/2506.16297v3)**

> **作者:** Heng Zhang; Zikang Wan; Danilo Vasconcellos Vargas
>
> **摘要:** Human vision excels at segmenting visual cues without the need for explicit training, and it remains remarkably robust even as noise severity increases. In contrast, existing AI algorithms struggle to maintain accuracy under similar conditions. Here, we present SyncMapV2, the first to solve unsupervised segmentation with state-of-the-art robustness. SyncMapV2 exhibits a minimal drop in mIoU, only 0.01%, under digital corruption, compared to a 23.8% drop observed in SOTA methods. This superior performance extends across various types of corruption: noise (7.3% vs. 37.7%), weather (7.5% vs. 33.8%), and blur (7.0% vs. 29.5%). Notably, SyncMapV2 accomplishes this without any robust training, supervision, or loss functions. It is based on a learning paradigm that uses self-organizing dynamical equations combined with concepts from random networks. Moreover, unlike conventional methods that require re-initialization for each new input, SyncMapV2 adapts online, mimicking the continuous adaptability of human vision. Thus, we go beyond the accurate and robust results, and present the first algorithm that can do all the above online, adapting to input rather than re-initializing. In adaptability tests, SyncMapV2 demonstrates near-zero performance degradation, which motivates and fosters a new generation of robust and adaptive intelligence in the near future.
>
---
#### [replaced 063] Cloud gap-filling with deep learning for improved grassland monitoring
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2403.09554v2](http://arxiv.org/pdf/2403.09554v2)**

> **作者:** Iason Tsardanidis; Alkiviadis Koukos; Vasileios Sitokonstantinou; Thanassis Drivas; Charalampos Kontoes
>
> **备注:** Published in Computers and Electronics in Agriculture
>
> **摘要:** Uninterrupted optical image time series are crucial for the timely monitoring of agricultural land changes, particularly in grasslands. However, the continuity of such time series is often disrupted by clouds. In response to this challenge, we propose an innovative deep learning method that integrates cloud-free optical (Sentinel-2) observations and weather-independent (Sentinel-1) Synthetic Aperture Radar (SAR) data. Our approach employs a hybrid architecture combining Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) to generate continuous Normalized Difference Vegetation Index (NDVI) time series, highlighting the role of NDVI in the synergy between SAR and optical data. We demonstrate the significance of observation continuity by assessing the impact of the generated NDVI time series on the downstream task of grassland mowing event detection. We conducted our study in Lithuania, a country characterized by extensive cloud coverage, and compared our approach with alternative interpolation techniques (i.e., linear, Akima, quadratic). Our method outperformed these techniques, achieving an average Mean Absolute Error (MAE) of 0.024 and a coefficient of determination R^2 of 0.92. Additionally, our analysis revealed improvement in the performance of the mowing event detection, with F1-score up to 84% using two widely applied mowing detection methodologies. Our method also effectively mitigated sudden shifts and noise originating from cloudy observations, which are often missed by conventional cloud masks and adversely affect mowing detection precision.
>
---
#### [replaced 064] Choosing Public Datasets for Private Machine Learning via Gradient Subspace Distance
- **分类: stat.ML; cs.CR; cs.CV; cs.DS; cs.LG**

- **链接: [http://arxiv.org/pdf/2303.01256v2](http://arxiv.org/pdf/2303.01256v2)**

> **作者:** Xin Gu; Gautam Kamath; Zhiwei Steven Wu
>
> **备注:** Accepted to SaTML 2025
>
> **摘要:** Differentially private stochastic gradient descent privatizes model training by injecting noise into each iteration, where the noise magnitude increases with the number of model parameters. Recent works suggest that we can reduce the noise by leveraging public data for private machine learning, by projecting gradients onto a subspace prescribed by the public data. However, given a choice of public datasets, it is not a priori clear which one may be most appropriate for the private task. We give an algorithm for selecting a public dataset by measuring a low-dimensional subspace distance between gradients of the public and private examples. We provide theoretical analysis demonstrating that the excess risk scales with this subspace distance. This distance is easy to compute and robust to modifications in the setting. Empirical evaluation shows that trained model accuracy is monotone in this distance.
>
---
#### [replaced 065] PLOT-TAL: Prompt Learning with Optimal Transport for Few-Shot Temporal Action Localization
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.18915v2](http://arxiv.org/pdf/2403.18915v2)**

> **作者:** Edward Fish; Andrew Gilbert
>
> **备注:** Accepted to ICCVWS
>
> **摘要:** Few-shot temporal action localization (TAL) methods that adapt large models via single-prompt tuning often fail to produce precise temporal boundaries. This stems from the model learning a non-discriminative mean representation of an action from sparse data, which compromises generalization. We address this by proposing a new paradigm based on multi-prompt ensembles, where a set of diverse, learnable prompts for each action is encouraged to specialize on compositional sub-events. To enforce this specialization, we introduce PLOT-TAL, a framework that leverages Optimal Transport (OT) to find a globally optimal alignment between the prompt ensemble and the video's temporal features. Our method establishes a new state-of-the-art on the challenging few-shot benchmarks of THUMOS'14 and EPIC-Kitchens, without requiring complex meta-learning. The significant performance gains, particularly at high IoU thresholds, validate our hypothesis and demonstrate the superiority of learning distributed, compositional representations for precise temporal localization.
>
---
#### [replaced 066] PF3plat: Pose-Free Feed-Forward 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.22128v2](http://arxiv.org/pdf/2410.22128v2)**

> **作者:** Sunghwan Hong; Jaewoo Jung; Heeseong Shin; Jisang Han; Jiaolong Yang; Chong Luo; Seungryong Kim
>
> **备注:** Accepted by ICML'25
>
> **摘要:** We consider the problem of novel view synthesis from unposed images in a single feed-forward. Our framework capitalizes on fast speed, scalability, and high-quality 3D reconstruction and view synthesis capabilities of 3DGS, where we further extend it to offer a practical solution that relaxes common assumptions such as dense image views, accurate camera poses, and substantial image overlaps. We achieve this through identifying and addressing unique challenges arising from the use of pixel-aligned 3DGS: misaligned 3D Gaussians across different views induce noisy or sparse gradients that destabilize training and hinder convergence, especially when above assumptions are not met. To mitigate this, we employ pre-trained monocular depth estimation and visual correspondence models to achieve coarse alignments of 3D Gaussians. We then introduce lightweight, learnable modules to refine depth and pose estimates from the coarse alignments, improving the quality of 3D reconstruction and novel view synthesis. Furthermore, the refined estimates are leveraged to estimate geometry confidence scores, which assess the reliability of 3D Gaussian centers and condition the prediction of Gaussian parameters accordingly. Extensive evaluations on large-scale real-world datasets demonstrate that PF3plat sets a new state-of-the-art across all benchmarks, supported by comprehensive ablation studies validating our design choices. project page: https://cvlab-kaist.github.io/PF3plat/
>
---
#### [replaced 067] X-ray2CTPA: Leveraging Diffusion Models to Enhance Pulmonary Embolism Classification
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2406.16109v4](http://arxiv.org/pdf/2406.16109v4)**

> **作者:** Noa Cahan; Eyal Klang; Galit Aviram; Yiftach Barash; Eli Konen; Raja Giryes; Hayit Greenspan
>
> **备注:** preprint, project code: https://github.com/NoaCahan/X-ray2CTPA
>
> **摘要:** Chest X-rays or chest radiography (CXR), commonly used for medical diagnostics, typically enables limited imaging compared to computed tomography (CT) scans, which offer more detailed and accurate three-dimensional data, particularly contrast-enhanced scans like CT Pulmonary Angiography (CTPA). However, CT scans entail higher costs, greater radiation exposure, and are less accessible than CXRs. In this work we explore cross-modal translation from a 2D low contrast-resolution X-ray input to a 3D high contrast and spatial-resolution CTPA scan. Driven by recent advances in generative AI, we introduce a novel diffusion-based approach to this task. We evaluate the models performance using both quantitative metrics and qualitative feedback from radiologists, ensuring diagnostic relevance of the generated images. Furthermore, we employ the synthesized 3D images in a classification framework and show improved AUC in a PE categorization task, using the initial CXR input. The proposed method is generalizable and capable of performing additional cross-modality translations in medical imaging. It may pave the way for more accessible and cost-effective advanced diagnostic tools. The code for this project is available: https://github.com/NoaCahan/X-ray2CTPA .
>
---
#### [replaced 068] Quantifying and Narrowing the Unknown: Interactive Text-to-Video Retrieval via Uncertainty Minimization
- **分类: cs.CV; 68T45; I.2.10; H.3.3**

- **链接: [http://arxiv.org/pdf/2507.15504v2](http://arxiv.org/pdf/2507.15504v2)**

> **作者:** Bingqing Zhang; Zhuo Cao; Heming Du; Yang Li; Xue Li; Jiajun Liu; Sen Wang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Despite recent advances, Text-to-video retrieval (TVR) is still hindered by multiple inherent uncertainties, such as ambiguous textual queries, indistinct text-video mappings, and low-quality video frames. Although interactive systems have emerged to address these challenges by refining user intent through clarifying questions, current methods typically rely on heuristic or ad-hoc strategies without explicitly quantifying these uncertainties, limiting their effectiveness. Motivated by this gap, we propose UMIVR, an Uncertainty-Minimizing Interactive Text-to-Video Retrieval framework that explicitly quantifies three critical uncertainties-text ambiguity, mapping uncertainty, and frame uncertainty-via principled, training-free metrics: semantic entropy-based Text Ambiguity Score (TAS), Jensen-Shannon divergence-based Mapping Uncertainty Score (MUS), and a Temporal Quality-based Frame Sampler (TQFS). By adaptively generating targeted clarifying questions guided by these uncertainty measures, UMIVR iteratively refines user queries, significantly reducing retrieval ambiguity. Extensive experiments on multiple benchmarks validate UMIVR's effectiveness, achieving notable gains in Recall@1 (69.2\% after 10 interactive rounds) on the MSR-VTT-1k dataset, thereby establishing an uncertainty-minimizing foundation for interactive TVR.
>
---
#### [replaced 069] EVEv2: Improved Baselines for Encoder-Free Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.06788v2](http://arxiv.org/pdf/2502.06788v2)**

> **作者:** Haiwen Diao; Xiaotong Li; Yufeng Cui; Yueze Wang; Haoge Deng; Ting Pan; Wenxuan Wang; Huchuan Lu; Xinlong Wang
>
> **备注:** 20 pages, 10 figures, Accepted by ICCV2025 (highlight)
>
> **摘要:** Existing encoder-free vision-language models (VLMs) are rapidly narrowing the performance gap with their encoder-based counterparts, highlighting the promising potential for unified multimodal systems with structural simplicity and efficient deployment. We systematically clarify the performance gap between VLMs using pre-trained vision encoders, discrete tokenizers, and minimalist visual layers from scratch, deeply excavating the under-examined characteristics of encoder-free VLMs. We develop efficient strategies for encoder-free VLMs that rival mainstream encoder-based ones. After an in-depth investigation, we launch EVEv2.0, a new and improved family of encoder-free VLMs. We show that: (i) Properly decomposing and hierarchically associating vision and language within a unified model reduces interference between modalities. (ii) A well-designed training strategy enables effective optimization for encoder-free VLMs. Through extensive evaluation, our EVEv2.0 represents a thorough study for developing a decoder-only architecture across modalities, demonstrating superior data efficiency and strong vision-reasoning capability. Code is publicly available at: https://github.com/baaivision/EVE.
>
---
#### [replaced 070] Personalization Toolkit: Training Free Personalization of Large Vision Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.02452v3](http://arxiv.org/pdf/2502.02452v3)**

> **作者:** Soroush Seifi; Vaggelis Dorovatas; Daniel Olmeda Reino; Rahaf Aljundi
>
> **摘要:** Personalization of Large Vision-Language Models (LVLMs) involves customizing models to recognize specific users and object instances, and to generate contextually tailored responses. Existing approaches typically rely on time-consuming test-time training for each user or object, making them impractical for real-world deployment, a limitation reflected in current personalization benchmarks, which are focused on object-centric, single-concept evaluations. In this paper, we present a novel training-free approach to LVLM personalization and introduce a comprehensive real-world benchmark designed to rigorously evaluate various aspects of the personalization task. Our method leverages pre-trained vision foundation models to extract distinctive features, applies retrieval-augmented generation (RAG) techniques to identify instances within visual inputs, and employs visual prompting strategies to guide model outputs. Our model-agnostic vision toolkit enables efficient and flexible multi-concept personalization across both images and videos, without any additional training. We achieve state-of-the-art results, surpassing existing training-based methods.
>
---
#### [replaced 071] Zero-Shot Skeleton-Based Action Recognition With Prototype-Guided Feature Alignment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.00566v2](http://arxiv.org/pdf/2507.00566v2)**

> **作者:** Kai Zhou; Shuhai Zhang; Zeng You; Jinwu Hu; Mingkui Tan; Fei Liu
>
> **备注:** This paper is accepted by IEEE TIP 2025 (The journal version is available at https://doi.org/10.1109/TIP.2025.3586487). Code is publicly available at https://github.com/kaai520/PGFA
>
> **摘要:** Zero-shot skeleton-based action recognition aims to classify unseen skeleton-based human actions without prior exposure to such categories during training. This task is extremely challenging due to the difficulty in generalizing from known to unknown actions. Previous studies typically use two-stage training: pre-training skeleton encoders on seen action categories using cross-entropy loss and then aligning pre-extracted skeleton and text features, enabling knowledge transfer to unseen classes through skeleton-text alignment and language models' generalization. However, their efficacy is hindered by 1) insufficient discrimination for skeleton features, as the fixed skeleton encoder fails to capture necessary alignment information for effective skeleton-text alignment; 2) the neglect of alignment bias between skeleton and unseen text features during testing. To this end, we propose a prototype-guided feature alignment paradigm for zero-shot skeleton-based action recognition, termed PGFA. Specifically, we develop an end-to-end cross-modal contrastive training framework to improve skeleton-text alignment, ensuring sufficient discrimination for skeleton features. Additionally, we introduce a prototype-guided text feature alignment strategy to mitigate the adverse impact of the distribution discrepancy during testing. We provide a theoretical analysis to support our prototype-guided text feature alignment strategy and empirically evaluate our overall PGFA on three well-known datasets. Compared with the top competitor SMIE method, our PGFA achieves absolute accuracy improvements of 22.96%, 12.53%, and 18.54% on the NTU-60, NTU-120, and PKU-MMD datasets, respectively.
>
---
#### [replaced 072] One Look is Enough: A Novel Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation Models on High-Resolution Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.22351v2](http://arxiv.org/pdf/2503.22351v2)**

> **作者:** Byeongjun Kwon; Munchurl Kim
>
> **备注:** ICCV 2025 (camera-ready version). [Project page](https://kaist-viclab.github.io/One-Look-is-Enough_site)
>
> **摘要:** Zero-shot depth estimation (DE) models exhibit strong generalization performance as they are trained on large-scale datasets. However, existing models struggle with high-resolution images due to the discrepancy in image resolutions of training (with smaller resolutions) and inference (for high resolutions). Processing them at full resolution leads to decreased estimation accuracy on depth with tremendous memory consumption, while downsampling to the training resolution results in blurred edges in the estimated depth images. Prevailing high-resolution depth estimation methods adopt a patch-based approach, which introduces depth discontinuity issues when reassembling the estimated depth patches, resulting in test-time inefficiency. Additionally, to obtain fine-grained depth details, these methods rely on synthetic datasets due to the real-world sparse ground truth depth, leading to poor generalizability. To tackle these limitations, we propose Patch Refine Once (PRO), an efficient and generalizable tile-based framework. Our PRO consists of two key components: (i) Grouped Patch Consistency Training that enhances test-time efficiency while mitigating the depth discontinuity problem by jointly processing four overlapping patches and enforcing a consistency loss on their overlapping regions within a single backpropagation step, and (ii) Bias Free Masking that prevents the DE models from overfitting to dataset-specific biases, enabling better generalization to real-world datasets even after training on synthetic data. Zero-shot evaluations on Booster, ETH3D, Middlebury 2014, and NuScenes demonstrate that our PRO can be seamlessly integrated into existing depth estimation models.
>
---
#### [replaced 073] Degradation-Agnostic Statistical Facial Feature Transformation for Blind Face Restoration in Adverse Weather Conditions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.07464v2](http://arxiv.org/pdf/2507.07464v2)**

> **作者:** Chang-Hwan Son
>
> **摘要:** With the increasing deployment of intelligent CCTV systems in outdoor environments, there is a growing demand for face recognition systems optimized for challenging weather conditions. Adverse weather significantly degrades image quality, which in turn reduces recognition accuracy. Although recent face image restoration (FIR) models based on generative adversarial networks (GANs) and diffusion models have shown progress, their performance remains limited due to the lack of dedicated modules that explicitly address weather-induced degradations. This leads to distorted facial textures and structures. To address these limitations, we propose a novel GAN-based blind FIR framework that integrates two key components: local Statistical Facial Feature Transformation (SFFT) and Degradation-Agnostic Feature Embedding (DAFE). The local SFFT module enhances facial structure and color fidelity by aligning the local statistical distributions of low-quality (LQ) facial regions with those of high-quality (HQ) counterparts. Complementarily, the DAFE module enables robust statistical facial feature extraction under adverse weather conditions by aligning LQ and HQ encoder representations, thereby making the restoration process adaptive to severe weather-induced degradations. Experimental results demonstrate that the proposed degradation-agnostic SFFT model outperforms existing state-of-the-art FIR methods based on GAN and diffusion models, particularly in suppressing texture distortions and accurately reconstructing facial structures. Furthermore, both the SFFT and DAFE modules are empirically validated in enhancing structural fidelity and perceptual quality in face restoration under challenging weather scenarios.
>
---
#### [replaced 074] ToonifyGB: StyleGAN-based Gaussian Blendshapes for 3D Stylized Head Avatars
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10072v2](http://arxiv.org/pdf/2505.10072v2)**

> **作者:** Rui-Yang Ju; Sheng-Yen Huang; Yi-Ping Hung
>
> **摘要:** The introduction of 3D Gaussian blendshapes has enabled the real-time reconstruction of animatable head avatars from monocular video. Toonify, a StyleGAN-based method, has become widely used for facial image stylization. To extend Toonify for synthesizing diverse stylized 3D head avatars using Gaussian blendshapes, we propose an efficient two-stage framework, ToonifyGB. In Stage 1 (stylized video generation), we adopt an improved StyleGAN to generate the stylized video from the input video frames, which overcomes the limitation of cropping aligned faces at a fixed resolution as preprocessing for normal StyleGAN. This process provides a more stable stylized video, which enables Gaussian blendshapes to better capture the high-frequency details of the video frames, facilitating the synthesis of high-quality animations in the next stage. In Stage 2 (Gaussian blendshapes synthesis), our method learns a stylized neutral head model and a set of expression blendshapes from the generated stylized video. By combining the neutral head model with expression blendshapes, ToonifyGB can efficiently render stylized avatars with arbitrary expressions. We validate the effectiveness of ToonifyGB on benchmark datasets using two representative styles: Arcane and Pixar.
>
---
#### [replaced 075] CA-Cut: Crop-Aligned Cutout for Data Augmentation to Learn More Robust Under-Canopy Navigation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.17727v2](http://arxiv.org/pdf/2507.17727v2)**

> **作者:** Robel Mamo; Taeyeong Choi
>
> **备注:** Accepted for publication at the 12th European Conference on Mobile Robots (ECMR 2025)
>
> **摘要:** State-of-the-art visual under-canopy navigation methods are designed with deep learning-based perception models to distinguish traversable space from crop rows. While these models have demonstrated successful performance, they require large amounts of training data to ensure reliability in real-world field deployment. However, data collection is costly, demanding significant human resources for in-field sampling and annotation. To address this challenge, various data augmentation techniques are commonly employed during model training, such as color jittering, Gaussian blur, and horizontal flip, to diversify training data and enhance model robustness. In this paper, we hypothesize that utilizing only these augmentation techniques may lead to suboptimal performance, particularly in complex under-canopy environments with frequent occlusions, debris, and non-uniform spacing of crops. Instead, we propose a novel augmentation method, so-called Crop-Aligned Cutout (CA-Cut) which masks random regions out in input images that are spatially distributed around crop rows on the sides to encourage trained models to capture high-level contextual features even when fine-grained information is obstructed. Our extensive experiments with a public cornfield dataset demonstrate that masking-based augmentations are effective for simulating occlusions and significantly improving robustness in semantic keypoint predictions for visual navigation. In particular, we show that biasing the mask distribution toward crop rows in CA-Cut is critical for enhancing both prediction accuracy and generalizability across diverse environments achieving up to a 36.9% reduction in prediction error. In addition, we conduct ablation studies to determine the number of masks, the size of each mask, and the spatial distribution of masks to maximize overall performance.
>
---
#### [replaced 076] Dynamic mapping from static labels: remote sensing dynamic sample generation with temporal-spectral embedding
- **分类: eess.IV; cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2506.02574v2](http://arxiv.org/pdf/2506.02574v2)**

> **作者:** Shuai Yuan; Shuang Chen; Tianwu Lin; Jincheng Yuan; Geng Tian; Yang Xu; Jie Wang; Peng Gong
>
> **摘要:** Accurate remote sensing geographic mapping requires timely and representative samples. However, rapid land surface changes often render static samples obsolete within months, making manual sample updates labor-intensive and unsustainable. To address this challenge, we propose TasGen, a two-stage Temporal spectral-aware Automatic Sample Generation method for generating dynamic training samples from single-date static labels without human intervention. Land surface dynamics often manifest as anomalies in temporal-spectral sequences. %These anomalies are multivariate yet unified: temporal, spectral, or joint anomalies stem from different mechanisms and cannot be naively coupled, as this may obscure the nature of changes. Yet, any land surface state corresponds to a coherent temporal-spectral signature, which would be lost if the two dimensions are modeled separately. To effectively capture these dynamics, TasGen first disentangles temporal and spectral features to isolate their individual contributions, and then couples them to model their synergistic interactions. In the first stage, we introduce a hierarchical temporal-spectral variational autoencoder (HTS-VAE) with a dual-dimension embedding to learn low-dimensional latent patterns of normal samples by first disentangling and then jointly embedding temporal and spectral information. This temporal-spectral embedding enables robust anomaly detection by identifying deviations from learned joint patterns. In the second stage, a classifier trained on stable samples relabels change points across time to generate dynamic samples. To not only detect but also explain surface dynamics, we further propose an anomaly interpretation method based on Gibbs sampling, which attributes changes to specific spectral-temporal dimensions.
>
---
#### [replaced 077] DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.10510v4](http://arxiv.org/pdf/2412.10510v4)**

> **作者:** Tobias Braun; Mark Rothermel; Marcus Rohrbach; Anna Rohrbach
>
> **备注:** ICML 2025 version. 9 pages main paper, 35 pages with appendix, 18 figures and 7 tables. Corrected two inconsistent numbers in Table 2
>
> **摘要:** The proliferation of disinformation demands reliable and scalable fact-checking solutions. We present Dynamic Evidence-based FAct-checking with Multimodal Experts (DEFAME), a modular, zero-shot MLLM pipeline for open-domain, text-image claim verification. DEFAME operates in a six-stage process, dynamically selecting the tools and search depth to extract and evaluate textual and visual evidence. Unlike prior approaches that are text-only, lack explainability, or rely solely on parametric knowledge, DEFAME performs end-to-end verification, accounting for images in claims and evidence while generating structured, multimodal reports. Evaluation on the popular benchmarks VERITE, AVerITeC, and MOCHEG shows that DEFAME surpasses all previous methods, establishing itself as the new state-of-the-art fact-checking system for uni- and multimodal fact-checking. Moreover, we introduce a new multimodal benchmark, ClaimReview2024+, featuring claims after the knowledge cutoff of GPT-4o, avoiding data leakage. Here, DEFAME drastically outperforms the GPT-4o baselines, showing temporal generalizability and the potential for real-time fact-checking.
>
---
#### [replaced 078] PARTE: Part-Guided Texturing for 3D Human Reconstruction from a Single Image
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.17332v2](http://arxiv.org/pdf/2507.17332v2)**

> **作者:** Hyeongjin Nam; Donghwan Kim; Gyeongsik Moon; Kyoung Mu Lee
>
> **备注:** Published at ICCV 2025, 22 pages including the supplementary material
>
> **摘要:** The misaligned human texture across different human parts is one of the main limitations of existing 3D human reconstruction methods. Each human part, such as a jacket or pants, should maintain a distinct texture without blending into others. The structural coherence of human parts serves as a crucial cue to infer human textures in the invisible regions of a single image. However, most existing 3D human reconstruction methods do not explicitly exploit such part segmentation priors, leading to misaligned textures in their reconstructions. In this regard, we present PARTE, which utilizes 3D human part information as a key guide to reconstruct 3D human textures. Our framework comprises two core components. First, to infer 3D human part information from a single image, we propose a 3D part segmentation module (PartSegmenter) that initially reconstructs a textureless human surface and predicts human part labels based on the textureless surface. Second, to incorporate part information into texture reconstruction, we introduce a part-guided texturing module (PartTexturer), which acquires prior knowledge from a pre-trained image generation network on texture alignment of human parts. Extensive experiments demonstrate that our framework achieves state-of-the-art quality in 3D human reconstruction. The project page is available at https://hygenie1228.github.io/PARTE/.
>
---
#### [replaced 079] LagKV: Lag-Relative Information of the KV Cache Tells Which Tokens Are Important
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.04704v2](http://arxiv.org/pdf/2504.04704v2)**

> **作者:** Manlai Liang; JiaMing Zhang; Xiong Li; Jinlong Li
>
> **摘要:** The increasing size of the Key-Value (KV) cache during the Large Language Models long-context inference is the main obstacle for its balance between the deployment cost and task accuracy. To reduce the KV cache size in such scenarios, most previous efforts leveraged on the attention weight to evict non-critical cache tokens. But there is a trade-off in those methods, they usually require major modification of the inference infrastructure and significant computation overhead. Based on the fact that the Large Language models are autoregressive models, we propose LagKV, a KV compression strategy only relying on straight forward comparison among KV themselves. It is a totally attention free method which offers easy integration to the main stream inference platform and comparable performance comparing to other complicated KV compression methods. Results on RULER benchmark show that, our approach outperforms SnapKV and StreamingLLM in different compression ratios. Especially in the 64-digit passkey retrieval task, our method outperforms the attention weight based method $H_2O$ over $50\%$ with same compression ratios. Our code is available at https://github.com/AI-Lab-China-Merchants-Bank/LagKV.
>
---
#### [replaced 080] SR-NeRV: Improving Embedding Efficiency of Neural Video Representation via Super-Resolution
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.00046v2](http://arxiv.org/pdf/2505.00046v2)**

> **作者:** Taiga Hayami; Kakeru Koizumi; Hiroshi Watanabe
>
> **摘要:** Implicit Neural Representations (INRs) have garnered significant attention for their ability to model complex signals in various domains. Recently, INR-based frameworks have shown promise in neural video compression by embedding video content into compact neural networks. However, these methods often struggle to reconstruct high-frequency details under stringent constraints on model size, which are critical in practical compression scenarios. To address this limitation, we propose an INR-based video representation framework that integrates a general-purpose super-resolution (SR) network. This design is motivated by the observation that high-frequency components tend to exhibit low temporal redundancy across frames. By offloading the reconstruction of fine details to a dedicated SR network pre-trained on natural images, the proposed method improves visual fidelity. Experimental results demonstrate that the proposed method outperforms conventional INR-based baselines in reconstruction quality, while maintaining a comparable model size.
>
---
#### [replaced 081] MC3D-AD: A Unified Geometry-aware Reconstruction Model for Multi-category 3D Anomaly Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.01969v2](http://arxiv.org/pdf/2505.01969v2)**

> **作者:** Jiayi Cheng; Can Gao; Jie Zhou; Jiajun Wen; Tao Dai; Jinbao Wang
>
> **备注:** 7 pages of main text, 3 pages of appendix, accepted to IJCAI 2025
>
> **摘要:** 3D Anomaly Detection (AD) is a promising means of controlling the quality of manufactured products. However, existing methods typically require carefully training a task-specific model for each category independently, leading to high cost, low efficiency, and weak generalization. Therefore, this paper presents a novel unified model for Multi-Category 3D Anomaly Detection (MC3D-AD) that aims to utilize both local and global geometry-aware information to reconstruct normal representations of all categories. First, to learn robust and generalized features of different categories, we propose an adaptive geometry-aware masked attention module that extracts geometry variation information to guide mask attention. Then, we introduce a local geometry-aware encoder reinforced by the improved mask attention to encode group-level feature tokens. Finally, we design a global query decoder that utilizes point cloud position embeddings to improve the decoding process and reconstruction ability. This leads to local and global geometry-aware reconstructed feature tokens for the AD task. MC3D-AD is evaluated on two publicly available Real3D-AD and Anomaly-ShapeNet datasets, and exhibits significant superiority over current state-of-the-art single-category methods, achieving 3.1\% and 9.3\% improvement in object-level AUROC over Real3D-AD and Anomaly-ShapeNet, respectively. The code is available at https://github.com/iCAN-SZU/MC3D-AD.
>
---
#### [replaced 082] Scaling RL to Long Videos
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.07966v2](http://arxiv.org/pdf/2507.07966v2)**

> **作者:** Yukang Chen; Wei Huang; Baifeng Shi; Qinghao Hu; Hanrong Ye; Ligeng Zhu; Zhijian Liu; Pavlo Molchanov; Jan Kautz; Xiaojuan Qi; Sifei Liu; Hongxu Yin; Yao Lu; Song Han
>
> **备注:** Code at https://github.com/NVlabs/Long-RL and model at https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B
>
> **摘要:** We introduce a full-stack framework that scales up reasoning in vision-language models (VLMs) to long videos, leveraging reinforcement learning. We address the unique challenges of long video reasoning by integrating three critical components: (1) a large-scale dataset, LongVideo-Reason, comprising 104K long video QA pairs with high-quality reasoning annotations across diverse domains such as sports, games, and vlogs; (2) a two-stage training pipeline that extends VLMs with chain-of-thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL); and (3) a training infrastructure for long video RL, named Multi-modal Reinforcement Sequence Parallelism (MR-SP), which incorporates sequence parallelism and a vLLM-based engine tailored for long video, using cached video embeddings for efficient rollout and prefilling. In our experiments, LongVILA-R1-7B achieves strong performance on video benchmarks, reaching 65.0% and 70.7% accuracy on VideoMME without and with subtitles, respectively, and consistently outperforming LongVILA-R1 across multiple benchmarks. Moreover, LongVILA-R1 shows steady performance improvements as the number of input video frames increases. Notably, our MR-SP system achieves up to 2.1x speedup on long video RL training. In addition, we release our training system for public availability that supports RL training on various modalities (video, text, and audio), various models (VILA and Qwen series), and even image and video generation models. On a single A100 node (8 GPUs), it supports RL training on hour-long videos (e.g., 3,600 frames / around 256k tokens).
>
---
#### [replaced 083] PreMix: Label-Efficient Multiple Instance Learning via Non-Contrastive Pre-training and Feature Mixing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.01162v3](http://arxiv.org/pdf/2408.01162v3)**

> **作者:** Bryan Wong; Mun Yong Yi
>
> **备注:** Under review
>
> **摘要:** Multiple instance learning (MIL) has emerged as a powerful framework for weakly supervised whole slide image (WSI) classification, enabling slide-level predictions without requiring detailed patch-level annotations. Despite its success, a critical limitation of current MIL methods lies in the underutilization of pre-training for the MIL aggregator. Most existing approaches initialize the aggregator randomly and train it from scratch, making performance highly sensitive to the quantity of labeled WSIs and ignoring the abundance of unlabeled WSIs commonly available in clinical settings. To address this, we propose PreMix, a novel framework that leverages a non-contrastive pre-training method, Barlow Twins, augmented with the Slide Mixing approach to generate additional positive pairs and enhance feature learning, particularly under limited labeled WSI conditions. Fine-tuning with Mixup and Manifold Mixup further enhances robustness by effectively handling the diverse sizes of gigapixel WSIs. Experimental results demonstrate that integrating PreMix as a plug-in module into HIPT yields an average F1 improvement of 4.7% over the baseline HIPT across various WSI training sizes and datasets. These findings underscore its potential to advance WSI classification with limited labeled data and its applicability to real-world histopathology practices. The code is available at https://github.com/bryanwong17/PreMix
>
---
#### [replaced 084] BiECVC: Gated Diversification of Bidirectional Contexts for Learned Video Compression
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.09193v4](http://arxiv.org/pdf/2505.09193v4)**

> **作者:** Wei Jiang; Junru Li; Kai Zhang; Li Zhang
>
> **备注:** Accepted to ACMMM 2025
>
> **摘要:** Recent forward prediction-based learned video compression (LVC) methods have achieved impressive results, even surpassing VVC reference software VTM under the Low Delay B (LDB) configuration. In contrast, learned bidirectional video compression (BVC) remains underexplored and still lags behind its forward-only counterparts. This performance gap is mainly due to the limited ability to extract diverse and accurate contexts: most existing BVCs primarily exploit temporal motion while neglecting non-local correlations across frames. Moreover, they lack the adaptability to dynamically suppress harmful contexts arising from fast motion or occlusion. To tackle these challenges, we propose BiECVC, a BVC framework that incorporates diversified local and non-local context modeling along with adaptive context gating. For local context enhancement, BiECVC reuses high-quality features from lower layers and aligns them using decoded motion vectors without introducing extra motion overhead. To model non-local dependencies efficiently, we adopt a linear attention mechanism that balances performance and complexity. To further mitigate the impact of inaccurate context prediction, we introduce Bidirectional Context Gating, inspired by data-dependent decay in recent autoregressive language models, to dynamically filter contextual information based on conditional coding results. Extensive experiments demonstrate that BiECVC achieves state-of-the-art performance, reducing the bit-rate by 13.4% and 15.7% compared to VTM 13.2 under the Random Access (RA) configuration with intra periods of 32 and 64, respectively. To our knowledge, BiECVC is the first learned video codec to surpass VTM 13.2 RA across all standard test datasets.
>
---
#### [replaced 085] RadioDUN: A Physics-Inspired Deep Unfolding Network for Radio Map Estimation
- **分类: cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2506.08418v2](http://arxiv.org/pdf/2506.08418v2)**

> **作者:** Taiqin Chen; Zikun Zhou; Zheng Fang; Wenzhen Zou; Kangjun Liu; Ke Chen; Yongbing Zhang; Yaowei Wang
>
> **摘要:** The radio map represents the spatial distribution of spectrum resources within a region, supporting efficient resource allocation and interference mitigation. However, it is difficult to construct a dense radio map as a limited number of samples can be measured in practical scenarios. While existing works have used deep learning to estimate dense radio maps from sparse samples, they are hard to integrate with the physical characteristics of the radio map. To address this challenge, we cast radio map estimation as the sparse signal recovery problem. A physical propagation model is further incorporated to decompose the problem into multiple factor optimization sub-problems, thereby reducing recovery complexity. Inspired by the existing compressive sensing methods, we propose the Radio Deep Unfolding Network (RadioDUN) to unfold the optimization process, achieving adaptive parameter adjusting and prior fitting in a learnable manner. To account for the radio propagation characteristics, we develop a dynamic reweighting module (DRM) to adaptively model the importance of each factor for the radio map. Inspired by the shadowing factor in the physical propagation model, we integrate obstacle-related factors to express the obstacle-induced signal stochastic decay. The shadowing loss is further designed to constrain the factor prediction and act as a supplementary supervised objective, which enhances the performance of RadioDUN. Extensive experiments have been conducted to demonstrate that the proposed method outperforms the state-of-the-art methods. Our code will be made publicly available upon publication.
>
---
#### [replaced 086] PRIX: Learning to Plan from Raw Pixels for End-to-End Autonomous Driving
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.17596v2](http://arxiv.org/pdf/2507.17596v2)**

> **作者:** Maciej K. Wozniak; Lianhang Liu; Yixi Cai; Patric Jensfelt
>
> **备注:** under review
>
> **摘要:** While end-to-end autonomous driving models show promising results, their practical deployment is often hindered by large model sizes, a reliance on expensive LiDAR sensors and computationally intensive BEV feature representations. This limits their scalability, especially for mass-market vehicles equipped only with cameras. To address these challenges, we propose PRIX (Plan from Raw Pixels). Our novel and efficient end-to-end driving architecture operates using only camera data, without explicit BEV representation and forgoing the need for LiDAR. PRIX leverages a visual feature extractor coupled with a generative planning head to predict safe trajectories from raw pixel inputs directly. A core component of our architecture is the Context-aware Recalibration Transformer (CaRT), a novel module designed to effectively enhance multi-level visual features for more robust planning. We demonstrate through comprehensive experiments that PRIX achieves state-of-the-art performance on the NavSim and nuScenes benchmarks, matching the capabilities of larger, multimodal diffusion planners while being significantly more efficient in terms of inference speed and model size, making it a practical solution for real-world deployment. Our work is open-source and the code will be at https://maxiuw.github.io/prix.
>
---
