# 计算机视觉 cs.CV

- **最新发布 76 篇**

- **更新 47 篇**

## 最新发布

#### [new 001] Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video
- **分类: cs.CV**

- **简介: 该论文提出一种基于比特率控制扩散模型的视频表示学习框架，用于分离视频中的运动与内容。通过自监督方式生成离散运动空间，实现运动迁移和生成任务，适用于真实人脸及卡通动画等多类型视频数据。**

- **链接: [http://arxiv.org/pdf/2509.08376v1](http://arxiv.org/pdf/2509.08376v1)**

> **作者:** Xiao Li; Qi Chen; Xiulian Peng; Kai Yu; Xie Chen; Yan Lu
>
> **摘要:** We propose a novel and general framework to disentangle video data into its dynamic motion and static content components. Our proposed method is a self-supervised pipeline with less assumptions and inductive biases than previous works: it utilizes a transformer-based architecture to jointly generate flexible implicit features for frame-wise motion and clip-wise content, and incorporates a low-bitrate vector quantization as an information bottleneck to promote disentanglement and form a meaningful discrete motion space. The bitrate-controlled latent motion and content are used as conditional inputs to a denoising diffusion model to facilitate self-supervised representation learning. We validate our disentangled representation learning framework on real-world talking head videos with motion transfer and auto-regressive motion generation tasks. Furthermore, we also show that our method can generalize to other types of video data, such as pixel sprites of 2D cartoon characters. Our work presents a new perspective on self-supervised learning of disentangled video representations, contributing to the broader field of video analysis and generation.
>
---
#### [new 002] VRAE: Vertical Residual Autoencoder for License Plate Denoising and Deblurring
- **分类: cs.CV**

- **简介: 论文提出VRAE模型，用于交通监控中车牌图像的去噪与去模糊。针对恶劣条件下车牌识别率低的问题，设计垂直残差自编码器，通过辅助模块提升特征学习效果，实验表明其在PSNR、NMSE和SSIM指标上优于传统方法。**

- **链接: [http://arxiv.org/pdf/2509.08392v1](http://arxiv.org/pdf/2509.08392v1)**

> **作者:** Cuong Nguyen; Dung T. Tran; Hong Nguyen; Xuan-Vu Phan; Nam-Phong Nguyen
>
> **摘要:** In real-world traffic surveillance, vehicle images captured under adverse weather, poor lighting, or high-speed motion often suffer from severe noise and blur. Such degradations significantly reduce the accuracy of license plate recognition systems, especially when the plate occupies only a small region within the full vehicle image. Restoring these degraded images a fast realtime manner is thus a crucial pre-processing step to enhance recognition performance. In this work, we propose a Vertical Residual Autoencoder (VRAE) architecture designed for the image enhancement task in traffic surveillance. The method incorporates an enhancement strategy that employs an auxiliary block, which injects input-aware features at each encoding stage to guide the representation learning process, enabling better general information preservation throughout the network compared to conventional autoencoders. Experiments on a vehicle image dataset with visible license plates demonstrate that our method consistently outperforms Autoencoder (AE), Generative Adversarial Network (GAN), and Flow-Based (FB) approaches. Compared with AE at the same depth, it improves PSNR by about 20\%, reduces NMSE by around 50\%, and enhances SSIM by 1\%, while requiring only a marginal increase of roughly 1\% in parameters.
>
---
#### [new 003] Chirality in Action: Time-Aware Video Representation Learning by Latent Straightening
- **分类: cs.CV**

- **简介: 该论文提出一种时间感知视频表示学习方法，解决视频中时间敏感性不足的问题。通过引入“手性动作识别”任务，利用自监督方式增强图像特征的时间感知能力，提升视频分类性能。**

- **链接: [http://arxiv.org/pdf/2509.08502v1](http://arxiv.org/pdf/2509.08502v1)**

> **作者:** Piyush Bagad; Andrew Zisserman
>
> **备注:** 24 pages, 10 figures
>
> **摘要:** Our objective is to develop compact video representations that are sensitive to visual change over time. To measure such time-sensitivity, we introduce a new task: chiral action recognition, where one needs to distinguish between a pair of temporally opposite actions, such as "opening vs. closing a door", "approaching vs. moving away from something", "folding vs. unfolding paper", etc. Such actions (i) occur frequently in everyday life, (ii) require understanding of simple visual change over time (in object state, size, spatial position, count . . . ), and (iii) are known to be poorly represented by many video embeddings. Our goal is to build time aware video representations which offer linear separability between these chiral pairs. To that end, we propose a self-supervised adaptation recipe to inject time-sensitivity into a sequence of frozen image features. Our model is based on an auto-encoder with a latent space with inductive bias inspired by perceptual straightening. We show that this results in a compact but time-sensitive video representation for the proposed task across three datasets: Something-Something, EPIC-Kitchens, and Charade. Our method (i) outperforms much larger video models pre-trained on large-scale video datasets, and (ii) leads to an improvement in classification performance on standard benchmarks when combined with these existing models.
>
---
#### [new 004] SimCroP: Radiograph Representation Learning with Similarity-driven Cross-granularity Pre-training
- **分类: cs.CV**

- **简介: 该论文提出SimCroP框架，用于医学影像理解任务，解决CT扫描中病灶结构稀疏及报告与影像关联复杂的问题。通过相似性驱动对齐和跨粒度融合，提升多尺度下游任务性能。**

- **链接: [http://arxiv.org/pdf/2509.08311v1](http://arxiv.org/pdf/2509.08311v1)**

> **作者:** Rongsheng Wang; Fenghe Tang; Qingsong Yao; Rui Yan; Xu Zhang; Zhen Huang; Haoran Lai; Zhiyang He; Xiaodong Tao; Zihang Jiang; Shaohua Kevin Zhou
>
> **备注:** Accepted by MICCAI 2025
>
> **摘要:** Medical vision-language pre-training shows great potential in learning representative features from massive paired radiographs and reports. However, in computed tomography (CT) scans, the distribution of lesions which contain intricate structures is characterized by spatial sparsity. Besides, the complex and implicit relationships between different pathological descriptions in each sentence of the report and their corresponding sub-regions in radiographs pose additional challenges. In this paper, we propose a Similarity-Driven Cross-Granularity Pre-training (SimCroP) framework on chest CTs, which combines similarity-driven alignment and cross-granularity fusion to improve radiograph interpretation. We first leverage multi-modal masked modeling to optimize the encoder for understanding precise low-level semantics from radiographs. Then, similarity-driven alignment is designed to pre-train the encoder to adaptively select and align the correct patches corresponding to each sentence in reports. The cross-granularity fusion module integrates multimodal information across instance level and word-patch level, which helps the model better capture key pathology structures in sparse radiographs, resulting in improved performance for multi-scale downstream tasks. SimCroP is pre-trained on a large-scale paired CT-reports dataset and validated on image classification and segmentation tasks across five public datasets. Experimental results demonstrate that SimCroP outperforms both cutting-edge medical self-supervised learning methods and medical vision-language pre-training methods. Codes and models are available at https://github.com/ToniChopp/SimCroP.
>
---
#### [new 005] Improving Greenland Bed Topography Mapping with Uncertainty-Aware Graph Learning on Sparse Radar Data
- **分类: cs.CV**

- **简介: 论文提出GraphTopoNet框架，利用图学习和不确定性建模，提升格陵兰冰床地形图精度。任务为从稀疏雷达数据中生成高精度冰床地图，解决观测稀疏导致的建模难题，通过融合多源数据与动态损失函数显著降低误差。**

- **链接: [http://arxiv.org/pdf/2509.08571v1](http://arxiv.org/pdf/2509.08571v1)**

> **作者:** Bayu Adhi Tama; Homayra Alam; Mostafa Cham; Omar Faruque; Jianwu Wang; Vandana Janeja
>
> **摘要:** Accurate maps of Greenland's subglacial bed are essential for sea-level projections, but radar observations are sparse and uneven. We introduce GraphTopoNet, a graph-learning framework that fuses heterogeneous supervision and explicitly models uncertainty via Monte Carlo dropout. Spatial graphs built from surface observables (elevation, velocity, mass balance) are augmented with gradient features and polynomial trends to capture both local variability and broad structure. To handle data gaps, we employ a hybrid loss that combines confidence-weighted radar supervision with dynamically balanced regularization. Applied to three Greenland subregions, GraphTopoNet outperforms interpolation, convolutional, and graph-based baselines, reducing error by up to 60 percent while preserving fine-scale glacial features. The resulting bed maps improve reliability for operational modeling, supporting agencies engaged in climate forecasting and policy. More broadly, GraphTopoNet shows how graph machine learning can convert sparse, uncertain geophysical observations into actionable knowledge at continental scale.
>
---
#### [new 006] Retrieval-Augmented VLMs for Multimodal Melanoma Diagnosis
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文提出一种检索增强的视觉语言模型（VLM），用于多模态黑色素瘤诊断。该任务旨在提升皮肤癌的早期准确诊断，解决传统CNN忽略临床数据和VLM泛化能力不足的问题。通过引入相似病例信息进行提示，提升分类精度与错误纠正能力。**

- **链接: [http://arxiv.org/pdf/2509.08338v1](http://arxiv.org/pdf/2509.08338v1)**

> **作者:** Jihyun Moon; Charmgil Hong
>
> **备注:** Medical Image Computing and Computer-Assisted Intervention (MICCAI) ISIC Skin Image Analysis Workshop (MICCAI ISIC) 2025; 10 pages
>
> **摘要:** Accurate and early diagnosis of malignant melanoma is critical for improving patient outcomes. While convolutional neural networks (CNNs) have shown promise in dermoscopic image analysis, they often neglect clinical metadata and require extensive preprocessing. Vision-language models (VLMs) offer a multimodal alternative but struggle to capture clinical specificity when trained on general-domain data. To address this, we propose a retrieval-augmented VLM framework that incorporates semantically similar patient cases into the diagnostic prompt. Our method enables informed predictions without fine-tuning and significantly improves classification accuracy and error correction over conventional baselines. These results demonstrate that retrieval-augmented prompting provides a robust strategy for clinical decision support.
>
---
#### [new 007] LADB: Latent Aligned Diffusion Bridges for Semi-Supervised Domain Translation
- **分类: cs.CV**

- **简介: 论文提出LADB框架，解决半监督域翻译问题，利用部分配对数据在共享潜在空间对齐分布，结合预训练源域扩散模型与目标域LADM，实现无需全监督的样本到样本翻译，提升生成质量与多样性。**

- **链接: [http://arxiv.org/pdf/2509.08628v1](http://arxiv.org/pdf/2509.08628v1)**

> **作者:** Xuqin Wang; Tao Wu; Yanfeng Zhang; Lu Liu; Dong Wang; Mingwei Sun; Yongliang Wang; Niclas Zeller; Daniel Cremers
>
> **摘要:** Diffusion models excel at generating high-quality outputs but face challenges in data-scarce domains, where exhaustive retraining or costly paired data are often required. To address these limitations, we propose Latent Aligned Diffusion Bridges (LADB), a semi-supervised framework for sample-to-sample translation that effectively bridges domain gaps using partially paired data. By aligning source and target distributions within a shared latent space, LADB seamlessly integrates pretrained source-domain diffusion models with a target-domain Latent Aligned Diffusion Model (LADM), trained on partially paired latent representations. This approach enables deterministic domain mapping without the need for full supervision. Compared to unpaired methods, which often lack controllability, and fully paired approaches that require large, domain-specific datasets, LADB strikes a balance between fidelity and diversity by leveraging a mixture of paired and unpaired latent-target couplings. Our experimental results demonstrate superior performance in depth-to-image translation under partial supervision. Furthermore, we extend LADB to handle multi-source translation (from depth maps and segmentation masks) and multi-target translation in a class-conditioned style transfer task, showcasing its versatility in handling diverse and heterogeneous use cases. Ultimately, we present LADB as a scalable and versatile solution for real-world domain translation, particularly in scenarios where data annotation is costly or incomplete.
>
---
#### [new 008] Dual-Thresholding Heatmaps to Cluster Proposals for Weakly Supervised Object Detection
- **分类: cs.CV**

- **简介: 该论文属于弱监督目标检测任务，旨在解决无需框标注的检测问题。提出双阈值热图算法选择候选框，并设计WSBDN网络和负样本损失函数，提升检测性能与收敛速度。**

- **链接: [http://arxiv.org/pdf/2509.08289v1](http://arxiv.org/pdf/2509.08289v1)**

> **作者:** Yuelin Guo; Haoyu He; Zhiyuan Chen; Zitong Huang; Renhao Lu; Lu Shi; Zejun Wang; Weizhe Zhang
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Weakly supervised object detection (WSOD) has attracted significant attention in recent years, as it does not require box-level annotations. State-of-the-art methods generally adopt a multi-module network, which employs WSDDN as the multiple instance detection network module and multiple instance refinement modules to refine performance. However, these approaches suffer from three key limitations. First, existing methods tend to generate pseudo GT boxes that either focus only on discriminative parts, failing to capture the whole object, or cover the entire object but fail to distinguish between adjacent intra-class instances. Second, the foundational WSDDN architecture lacks a crucial background class representation for each proposal and exhibits a large semantic gap between its branches. Third, prior methods discard ignored proposals during optimization, leading to slow convergence. To address these challenges, we first design a heatmap-guided proposal selector (HGPS) algorithm, which utilizes dual thresholds on heatmaps to pre-select proposals, enabling pseudo GT boxes to both capture the full object extent and distinguish between adjacent intra-class instances. We then present a weakly supervised basic detection network (WSBDN), which augments each proposal with a background class representation and uses heatmaps for pre-supervision to bridge the semantic gap between matrices. At last, we introduce a negative certainty supervision loss on ignored proposals to accelerate convergence. Extensive experiments on the challenging PASCAL VOC 2007 and 2012 datasets demonstrate the effectiveness of our framework. We achieve mAP/mCorLoc scores of 58.5%/81.8% on VOC 2007 and 55.6%/80.5% on VOC 2012, performing favorably against the state-of-the-art WSOD methods. Our code is publicly available at https://github.com/gyl2565309278/DTH-CP.
>
---
#### [new 009] RepViT-CXR: A Channel Replication Strategy for Vision Transformers in Chest X-ray Tuberculosis and Pneumonia Classification
- **分类: cs.CV; cs.LG; F.2.2; I.2.7**

- **简介: 论文提出RepViT-CXR方法，解决ViT在单通道CXR图像中的应用问题。通过通道复制策略，将灰度图像适配ViT输入，提升TB和肺炎分类性能，在多个数据集上取得SOTA结果。属于医学图像分类任务。**

- **链接: [http://arxiv.org/pdf/2509.08234v1](http://arxiv.org/pdf/2509.08234v1)**

> **作者:** Faisal Ahmed
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Chest X-ray (CXR) imaging remains one of the most widely used diagnostic tools for detecting pulmonary diseases such as tuberculosis (TB) and pneumonia. Recent advances in deep learning, particularly Vision Transformers (ViTs), have shown strong potential for automated medical image analysis. However, most ViT architectures are pretrained on natural images and require three-channel inputs, while CXR scans are inherently grayscale. To address this gap, we propose RepViT-CXR, a channel replication strategy that adapts single-channel CXR images into a ViT-compatible format without introducing additional information loss. We evaluate RepViT-CXR on three benchmark datasets. On the TB-CXR dataset,our method achieved an accuracy of 99.9% and an AUC of 99.9%, surpassing prior state-of-the-art methods such as Topo-CXR (99.3% accuracy, 99.8% AUC). For the Pediatric Pneumonia dataset, RepViT-CXR obtained 99.0% accuracy, with 99.2% recall, 99.3% precision, and an AUC of 99.0%, outperforming strong baselines including DCNN and VGG16. On the Shenzhen TB dataset, our approach achieved 91.1% accuracy and an AUC of 91.2%, marking a performance improvement over previously reported CNN-based methods. These results demonstrate that a simple yet effective channel replication strategy allows ViTs to fully leverage their representational power on grayscale medical imaging tasks. RepViT-CXR establishes a new state of the art for TB and pneumonia detection from chest X-rays, showing strong potential for deployment in real-world clinical screening systems.
>
---
#### [new 010] An End-to-End Deep Learning Framework for Arsenicosis Diagnosis Using Mobile-Captured Skin Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种基于移动端皮肤图像的砷中毒诊断深度学习框架。任务为皮肤病分类诊断，解决农村地区缺乏皮肤科医生的问题。通过构建数据集、对比多种模型，最终采用Transformer模型实现86%准确率，并提供可视化解释工具。**

- **链接: [http://arxiv.org/pdf/2509.08780v1](http://arxiv.org/pdf/2509.08780v1)**

> **作者:** Asif Newaz; Asif Ur Rahman Adib; Rajit Sahil; Mashfique Mehzad
>
> **摘要:** Background: Arsenicosis is a serious public health concern in South and Southeast Asia, primarily caused by long-term consumption of arsenic-contaminated water. Its early cutaneous manifestations are clinically significant but often underdiagnosed, particularly in rural areas with limited access to dermatologists. Automated, image-based diagnostic solutions can support early detection and timely interventions. Methods: In this study, we propose an end-to-end framework for arsenicosis diagnosis using mobile phone-captured skin images. A dataset comprising 20 classes and over 11000 images of arsenic-induced and other dermatological conditions was curated. Multiple deep learning architectures, including convolutional neural networks (CNNs) and Transformer-based models, were benchmarked for arsenicosis detection. Model interpretability was integrated via LIME and Grad-CAM, while deployment feasibility was demonstrated through a web-based diagnostic tool. Results: Transformer-based models significantly outperformed CNNs, with the Swin Transformer achieving the best results (86\\% accuracy). LIME and Grad-CAM visualizations confirmed that the models attended to lesion-relevant regions, increasing clinical transparency and aiding in error analysis. The framework also demonstrated strong performance on external validation samples, confirming its ability to generalize beyond the curated dataset. Conclusion: The proposed framework demonstrates the potential of deep learning for non-invasive, accessible, and explainable diagnosis of arsenicosis from mobile-acquired images. By enabling reliable image-based screening, it can serve as a practical diagnostic aid in rural and resource-limited communities, where access to dermatologists is scarce, thereby supporting early detection and timely intervention.
>
---
#### [new 011] Video Parallel Scaling: Aggregating Diverse Frame Subsets for VideoLLMs
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出视频并行扩展（VPS）方法，解决视频大语言模型因输入帧数增加导致的计算成本高和性能下降问题。通过并行处理不同帧子集并聚合结果，提升模型时序推理能力，无需扩大上下文窗口或额外训练。**

- **链接: [http://arxiv.org/pdf/2509.08016v1](http://arxiv.org/pdf/2509.08016v1)**

> **作者:** Hyungjin Chung; Hyelin Nam; Jiyeon Kim; Hyojun Go; Byeongjun Park; Junho Kim; Joonseok Lee; Seongsu Ha; Byung-Hoon Kim
>
> **备注:** https://github.com/hyungjin-chung/VPS
>
> **摘要:** Video Large Language Models (VideoLLMs) face a critical bottleneck: increasing the number of input frames to capture fine-grained temporal detail leads to prohibitive computational costs and performance degradation from long context lengths. We introduce Video Parallel Scaling (VPS), an inference-time method that expands a model's perceptual bandwidth without increasing its context window. VPS operates by running multiple parallel inference streams, each processing a unique, disjoint subset of the video's frames. By aggregating the output probabilities from these complementary streams, VPS integrates a richer set of visual information than is possible with a single pass. We theoretically show that this approach effectively contracts the Chinchilla scaling law by leveraging uncorrelated visual evidence, thereby improving performance without additional training. Extensive experiments across various model architectures and scales (2B-32B) on benchmarks such as Video-MME and EventHallusion demonstrate that VPS consistently and significantly improves performance. It scales more favorably than other parallel alternatives (e.g. Self-consistency) and is complementary to other decoding strategies, offering a memory-efficient and robust framework for enhancing the temporal reasoning capabilities of VideoLLMs.
>
---
#### [new 012] Two Stage Context Learning with Large Language Models for Multimodal Stance Detection on Climate Change
- **分类: cs.CV; cs.CY**

- **简介: 该论文属于多模态立场检测任务，旨在解决社交媒体中结合文本与视觉内容的气候变化立场识别问题。提出两阶段上下文学习框架，融合文本与图像信息，通过大语言模型和图像描述生成器提取特征，并使用专用Transformer模块进行联合建模，实现更准确的立场分类。**

- **链接: [http://arxiv.org/pdf/2509.08024v1](http://arxiv.org/pdf/2509.08024v1)**

> **作者:** Lata Pangtey; Omkar Kabde; Shahid Shafi Dar; Nagendra Kumar
>
> **摘要:** With the rapid proliferation of information across digital platforms, stance detection has emerged as a pivotal challenge in social media analysis. While most of the existing approaches focus solely on textual data, real-world social media content increasingly combines text with visual elements creating a need for advanced multimodal methods. To address this gap, we propose a multimodal stance detection framework that integrates textual and visual information through a hierarchical fusion approach. Our method first employs a Large Language Model to retrieve stance-relevant summaries from source text, while a domain-aware image caption generator interprets visual content in the context of the target topic. These modalities are then jointly modeled along with the reply text, through a specialized transformer module that captures interactions between the texts and images. The proposed modality fusion framework integrates diverse modalities to facilitate robust stance classification. We evaluate our approach on the MultiClimate dataset, a benchmark for climate change-related stance detection containing aligned video frames and transcripts. We achieve accuracy of 76.2%, precision of 76.3%, recall of 76.2% and F1-score of 76.2%, respectively, outperforming existing state-of-the-art approaches.
>
---
#### [new 013] AdsQA: Towards Advertisement Video Understanding
- **分类: cs.CV**

- **简介: 该论文提出AdsQA基准，用于评估LLMs对广告视频的理解能力。通过设计5项任务，构建包含1,544个广告视频的数据集，并提出ReAd-R模型，在广告视频问答任务中取得最优效果。**

- **链接: [http://arxiv.org/pdf/2509.08621v1](http://arxiv.org/pdf/2509.08621v1)**

> **作者:** Xinwei Long; Kai Tian; Peng Xu; Guoli Jia; Jingxuan Li; Sa Yang; Yihua Shao; Kaiyan Zhang; Che Jiang; Hao Xu; Yang Liu; Jiaheng Ma; Bowen Zhou
>
> **备注:** ICCV-2025
>
> **摘要:** Large language models (LLMs) have taken a great step towards AGI. Meanwhile, an increasing number of domain-specific problems such as math and programming boost these general-purpose models to continuously evolve via learning deeper expertise. Now is thus the time further to extend the diversity of specialized applications for knowledgeable LLMs, though collecting high quality data with unexpected and informative tasks is challenging. In this paper, we propose to use advertisement (ad) videos as a challenging test-bed to probe the ability of LLMs in perceiving beyond the objective physical content of common visual domain. Our motivation is to take full advantage of the clue-rich and information-dense ad videos' traits, e.g., marketing logic, persuasive strategies, and audience engagement. Our contribution is three-fold: (1) To our knowledge, this is the first attempt to use ad videos with well-designed tasks to evaluate LLMs. We contribute AdsQA, a challenging ad Video QA benchmark derived from 1,544 ad videos with 10,962 clips, totaling 22.7 hours, providing 5 challenging tasks. (2) We propose ReAd-R, a Deepseek-R1 styled RL model that reflects on questions, and generates answers via reward-driven optimization. (3) We benchmark 14 top-tier LLMs on AdsQA, and our \texttt{ReAd-R}~achieves the state-of-the-art outperforming strong competitors equipped with long-chain reasoning capabilities by a clear margin.
>
---
#### [new 014] Spherical Brownian Bridge Diffusion Models for Conditional Cortical Thickness Forecasting
- **分类: cs.CV; cs.AI; cs.LG; q-bio.NC**

- **简介: 该论文提出SBDM模型，用于预测个体皮层厚度轨迹。针对皮层非欧几里得结构和多模态数据整合难题，设计双向布朗桥扩散过程与CoS-UNet网络，实现高精度预测及事实/反事实轨迹生成，提升神经退行性疾病早期干预能力。**

- **链接: [http://arxiv.org/pdf/2509.08442v1](http://arxiv.org/pdf/2509.08442v1)**

> **作者:** Ivan Stoyanov; Fabian Bongratz; Christian Wachinger
>
> **摘要:** Accurate forecasting of individualized, high-resolution cortical thickness (CTh) trajectories is essential for detecting subtle cortical changes, providing invaluable insights into neurodegenerative processes and facilitating earlier and more precise intervention strategies. However, CTh forecasting is a challenging task due to the intricate non-Euclidean geometry of the cerebral cortex and the need to integrate multi-modal data for subject-specific predictions. To address these challenges, we introduce the Spherical Brownian Bridge Diffusion Model (SBDM). Specifically, we propose a bidirectional conditional Brownian bridge diffusion process to forecast CTh trajectories at the vertex level of registered cortical surfaces. Our technical contribution includes a new denoising model, the conditional spherical U-Net (CoS-UNet), which combines spherical convolutions and dense cross-attention to integrate cortical surfaces and tabular conditions seamlessly. Compared to previous approaches, SBDM achieves significantly reduced prediction errors, as demonstrated by our experiments based on longitudinal datasets from the ADNI and OASIS. Additionally, we demonstrate SBDM's ability to generate individual factual and counterfactual CTh trajectories, offering a novel framework for exploring hypothetical scenarios of cortical development.
>
---
#### [new 015] ArgoTweak: Towards Self-Updating HD Maps through Structured Priors
- **分类: cs.CV**

- **简介: 该论文提出ArgoTweak数据集，用于解决高精度地图自我更新中缺乏真实先验信息的问题。通过结构化先验与细粒度标注，提升模型泛化能力，缩小仿真与现实差距，推动可解释的高精度地图构建。**

- **链接: [http://arxiv.org/pdf/2509.08764v1](http://arxiv.org/pdf/2509.08764v1)**

> **作者:** Lena Wild; Rafael Valencia; Patric Jensfelt
>
> **备注:** ICCV 2025
>
> **摘要:** Reliable integration of prior information is crucial for self-verifying and self-updating HD maps. However, no public dataset includes the required triplet of prior maps, current maps, and sensor data. As a result, existing methods must rely on synthetic priors, which create inconsistencies and lead to a significant sim2real gap. To address this, we introduce ArgoTweak, the first dataset to complete the triplet with realistic map priors. At its core, ArgoTweak employs a bijective mapping framework, breaking down large-scale modifications into fine-grained atomic changes at the map element level, thus ensuring interpretability. This paradigm shift enables accurate change detection and integration while preserving unchanged elements with high fidelity. Experiments show that training models on ArgoTweak significantly reduces the sim2real gap compared to synthetic priors. Extensive ablations further highlight the impact of structured priors and detailed change annotations. By establishing a benchmark for explainable, prior-aided HD mapping, ArgoTweak advances scalable, self-improving mapping solutions. The dataset, baselines, map modification toolbox, and further resources are available at https://kth-rpl.github.io/ArgoTweak/.
>
---
#### [new 016] Calibrating MLLM-as-a-judge via Multimodal Bayesian Prompt Ensembles
- **分类: cs.CV; cs.CL**

- **简介: 论文提出MMB方法，用于校准多模态大语言模型在图像生成评估中的判断能力。针对模型存在偏差、过拟合和跨领域性能不一致的问题，通过多模态贝叶斯提示集成提升准确性和校准效果。**

- **链接: [http://arxiv.org/pdf/2509.08777v1](http://arxiv.org/pdf/2509.08777v1)**

> **作者:** Eric Slyman; Mehrab Tanjim; Kushal Kafle; Stefan Lee
>
> **备注:** 17 pages, 8 figures, Accepted at ICCV 2025
>
> **摘要:** Multimodal large language models (MLLMs) are increasingly used to evaluate text-to-image (TTI) generation systems, providing automated judgments based on visual and textual context. However, these "judge" models often suffer from biases, overconfidence, and inconsistent performance across diverse image domains. While prompt ensembling has shown promise for mitigating these issues in unimodal, text-only settings, our experiments reveal that standard ensembling methods fail to generalize effectively for TTI tasks. To address these limitations, we propose a new multimodal-aware method called Multimodal Mixture-of-Bayesian Prompt Ensembles (MMB). Our method uses a Bayesian prompt ensemble approach augmented by image clustering, allowing the judge to dynamically assign prompt weights based on the visual characteristics of each sample. We show that MMB improves accuracy in pairwise preference judgments and greatly enhances calibration, making it easier to gauge the judge's true uncertainty. In evaluations on two TTI benchmarks, HPSv2 and MJBench, MMB outperforms existing baselines in alignment with human annotations and calibration across varied image content. Our findings highlight the importance of multimodal-specific strategies for judge calibration and suggest a promising path forward for reliable large-scale TTI evaluation.
>
---
#### [new 017] FractalPINN-Flow: A Fractal-Inspired Network for Unsupervised Optical Flow Estimation with Total Variation Regularization
- **分类: cs.CV**

- **简介: 该论文提出FractalPINN-Flow，用于无监督光流估计任务。其解决无需标注数据的密集光流预测问题，采用分形变形网络与总变差正则化，提升精度与平滑性。**

- **链接: [http://arxiv.org/pdf/2509.08670v1](http://arxiv.org/pdf/2509.08670v1)**

> **作者:** Sara Behnamian; Rasoul Khaksarinezhad; Andreas Langer
>
> **摘要:** We present FractalPINN-Flow, an unsupervised deep learning framework for dense optical flow estimation that learns directly from consecutive grayscale frames without requiring ground truth. The architecture centers on the Fractal Deformation Network (FDN) - a recursive encoder-decoder inspired by fractal geometry and self-similarity. Unlike traditional CNNs with sequential downsampling, FDN uses repeated encoder-decoder nesting with skip connections to capture both fine-grained details and long-range motion patterns. The training objective is based on a classical variational formulation using total variation (TV) regularization. Specifically, we minimize an energy functional that combines $L^1$ and $L^2$ data fidelity terms to enforce brightness constancy, along with a TV term that promotes spatial smoothness and coherent flow fields. Experiments on synthetic and benchmark datasets show that FractalPINN-Flow produces accurate, smooth, and edge-preserving optical flow fields. The model is especially effective for high-resolution data and scenarios with limited annotations.
>
---
#### [new 018] Boosted Training of Lightweight Early Exits for Optimizing CNN Image Classification Inference
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决CNN在资源受限设备上的实时推理效率与精度平衡问题。提出BTS-EE训练方案，通过顺序训练和轻量分支设计，减少计算量并提升准确率，适用于工业检测等场景。**

- **链接: [http://arxiv.org/pdf/2509.08318v1](http://arxiv.org/pdf/2509.08318v1)**

> **作者:** Yehudit Aperstein; Alexander Apartsin
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Real-time image classification on resource-constrained platforms demands inference methods that balance accuracy with strict latency and power budgets. Early-exit strategies address this need by attaching auxiliary classifiers to intermediate layers of convolutional neural networks (CNNs), allowing "easy" samples to terminate inference early. However, conventional training of early exits introduces a covariance shift: downstream branches are trained on full datasets, while at inference they process only the harder, non-exited samples. This mismatch limits efficiency--accuracy trade-offs in practice. We introduce the Boosted Training Scheme for Early Exits (BTS-EE), a sequential training approach that aligns branch training with inference-time data distributions. Each branch is trained and calibrated before the next, ensuring robustness under selective inference conditions. To further support embedded deployment, we propose a lightweight branch architecture based on 1D convolutions and a Class Precision Margin (CPM) calibration method that enables per-class threshold tuning for reliable exit decisions. Experiments on the CINIC-10 dataset with a ResNet18 backbone demonstrate that BTS-EE consistently outperforms non-boosted training across 64 configurations, achieving up to 45 percent reduction in computation with only 2 percent accuracy degradation. These results expand the design space for deploying CNNs in real-time image processing systems, offering practical efficiency gains for applications such as industrial inspection, embedded vision, and UAV-based monitoring.
>
---
#### [new 019] UOPSL: Unpaired OCT Predilection Sites Learning for Fundus Image Diagnosis Augmentation
- **分类: cs.CV; cs.AI; I.4.10**

- **简介: 该论文提出UOPSL框架，解决无配对OCT与眼底图像诊断问题。通过OCT空间先验动态识别病灶区域，提升眼底图像分类效果，无需配对数据即可增强疾病识别能力。**

- **链接: [http://arxiv.org/pdf/2509.08624v1](http://arxiv.org/pdf/2509.08624v1)**

> **作者:** Zhihao Zhao; Yinzheng Zhao; Junjie Yang; Xiangtong Yao; Quanmin Liang; Daniel Zapp; Kai Huang; Nassir Navab; M. Ali Nasseri
>
> **备注:** BIBM
>
> **摘要:** Significant advancements in AI-driven multimodal medical image diagnosis have led to substantial improvements in ophthalmic disease identification in recent years. However, acquiring paired multimodal ophthalmic images remains prohibitively expensive. While fundus photography is simple and cost-effective, the limited availability of OCT data and inherent modality imbalance hinder further progress. Conventional approaches that rely solely on fundus or textual features often fail to capture fine-grained spatial information, as each imaging modality provides distinct cues about lesion predilection sites. In this study, we propose a novel unpaired multimodal framework \UOPSL that utilizes extensive OCT-derived spatial priors to dynamically identify predilection sites, enhancing fundus image-based disease recognition. Our approach bridges unpaired fundus and OCTs via extended disease text descriptions. Initially, we employ contrastive learning on a large corpus of unpaired OCT and fundus images while simultaneously learning the predilection sites matrix in the OCT latent space. Through extensive optimization, this matrix captures lesion localization patterns within the OCT feature space. During the fine-tuning or inference phase of the downstream classification task based solely on fundus images, where paired OCT data is unavailable, we eliminate OCT input and utilize the predilection sites matrix to assist in fundus image classification learning. Extensive experiments conducted on 9 diverse datasets across 28 critical categories demonstrate that our framework outperforms existing benchmarks.
>
---
#### [new 020] 3D and 4D World Modeling: A Survey
- **分类: cs.CV; cs.RO**

- **简介: 该论文综述了3D和4D世界建模技术，明确其定义与分类，总结相关数据集与评估指标，探讨应用与挑战，旨在为该领域提供系统性参考。属于AI环境建模任务，解决现有研究碎片化问题。**

- **链接: [http://arxiv.org/pdf/2509.07996v1](http://arxiv.org/pdf/2509.07996v1)**

> **作者:** Lingdong Kong; Wesley Yang; Jianbiao Mei; Youquan Liu; Ao Liang; Dekai Zhu; Dongyue Lu; Wei Yin; Xiaotao Hu; Mingkai Jia; Junyuan Deng; Kaiwen Zhang; Yang Wu; Tianyi Yan; Shenyuan Gao; Song Wang; Linfeng Li; Liang Pan; Yong Liu; Jianke Zhu; Wei Tsang Ooi; Steven C. H. Hoi; Ziwei Liu
>
> **备注:** Survey; 34 pages, 10 figures, 14 tables; GitHub Repo at https://github.com/worldbench/survey
>
> **摘要:** World modeling has become a cornerstone in AI research, enabling agents to understand, represent, and predict the dynamic environments they inhabit. While prior work largely emphasizes generative methods for 2D image and video data, they overlook the rapidly growing body of work that leverages native 3D and 4D representations such as RGB-D imagery, occupancy grids, and LiDAR point clouds for large-scale scene modeling. At the same time, the absence of a standardized definition and taxonomy for ``world models'' has led to fragmented and sometimes inconsistent claims in the literature. This survey addresses these gaps by presenting the first comprehensive review explicitly dedicated to 3D and 4D world modeling and generation. We establish precise definitions, introduce a structured taxonomy spanning video-based (VideoGen), occupancy-based (OccGen), and LiDAR-based (LiDARGen) approaches, and systematically summarize datasets and evaluation metrics tailored to 3D/4D settings. We further discuss practical applications, identify open challenges, and highlight promising research directions, aiming to provide a coherent and foundational reference for advancing the field. A systematic summary of existing literature is available at https://github.com/worldbench/survey
>
---
#### [new 021] Prompt-Driven Image Analysis with Multimodal Generative AI: Detection, Segmentation, Inpainting, and Interpretation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出一种基于提示的图像分析系统，整合检测、分割、修复与描述功能，解决多模态任务统一处理问题，提供透明、可重复的工作流及优化策略。**

- **链接: [http://arxiv.org/pdf/2509.08489v1](http://arxiv.org/pdf/2509.08489v1)**

> **作者:** Kaleem Ahmad
>
> **备注:** 14 pages. Preprint
>
> **摘要:** Prompt-driven image analysis converts a single natural-language instruction into multiple steps: locate, segment, edit, and describe. We present a practical case study of a unified pipeline that combines open-vocabulary detection, promptable segmentation, text-conditioned inpainting, and vision-language description into a single workflow. The system works end to end from a single prompt, retains intermediate artifacts for transparent debugging (such as detections, masks, overlays, edited images, and before and after composites), and provides the same functionality through an interactive UI and a scriptable CLI for consistent, repeatable runs. We highlight integration choices that reduce brittleness, including threshold adjustments, mask inspection with light morphology, and resource-aware defaults. In a small, single-word prompt segment, detection and segmentation produced usable masks in over 90% of cases with an accuracy above 85% based on our criteria. On a high-end GPU, inpainting makes up 60 to 75% of total runtime under typical guidance and sampling settings, which highlights the need for careful tuning. The study offers implementation-guided advice on thresholds, mask tightness, and diffusion parameters, and details version pinning, artifact logging, and seed control to support replay. Our contribution is a transparent, reliable pattern for assembling modern vision and multimodal models behind a single prompt, with clear guardrails and operational practices that improve reliability in object replacement, scene augmentation, and removal.
>
---
#### [new 022] Generalized Zero-Shot Learning for Point Cloud Segmentation with Evidence-Based Dynamic Calibration
- **分类: cs.CV**

- **简介: 论文提出E3DPC-GZSL方法，解决3D点云语义分割中对已见类过度自信的问题。通过引入基于证据的不确定性估计和动态校准策略，提升对未见类的预测能力，实验证明其在多个数据集上达到SOTA性能。属于广义零样本学习任务。**

- **链接: [http://arxiv.org/pdf/2509.08280v1](http://arxiv.org/pdf/2509.08280v1)**

> **作者:** Hyeonseok Kim; Byeongkeun Kang; Yeejin Lee
>
> **备注:** 20 pages, 12 figures, AAAI 2025
>
> **摘要:** Generalized zero-shot semantic segmentation of 3D point clouds aims to classify each point into both seen and unseen classes. A significant challenge with these models is their tendency to make biased predictions, often favoring the classes encountered during training. This problem is more pronounced in 3D applications, where the scale of the training data is typically smaller than in image-based tasks. To address this problem, we propose a novel method called E3DPC-GZSL, which reduces overconfident predictions towards seen classes without relying on separate classifiers for seen and unseen data. E3DPC-GZSL tackles the overconfidence problem by integrating an evidence-based uncertainty estimator into a classifier. This estimator is then used to adjust prediction probabilities using a dynamic calibrated stacking factor that accounts for pointwise prediction uncertainty. In addition, E3DPC-GZSL introduces a novel training strategy that improves uncertainty estimation by refining the semantic space. This is achieved by merging learnable parameters with text-derived features, thereby improving model optimization for unseen data. Extensive experiments demonstrate that the proposed approach achieves state-of-the-art performance on generalized zero-shot semantic segmentation datasets, including ScanNet v2 and S3DIS.
>
---
#### [new 023] Semantic Causality-Aware Vision-Based 3D Occupancy Prediction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D语义占用预测任务，旨在解决模块化流程中独立优化导致的误差问题。提出因果损失函数，实现端到端训练，并设计三个组件提升性能，实验表明方法在Occ3D上达到SOTA。**

- **链接: [http://arxiv.org/pdf/2509.08388v1](http://arxiv.org/pdf/2509.08388v1)**

> **作者:** Dubing Chen; Huan Zheng; Yucheng Zhou; Xianfei Li; Wenlong Liao; Tao He; Pai Peng; Jianbing Shen
>
> **备注:** ICCV 2025
>
> **摘要:** Vision-based 3D semantic occupancy prediction is a critical task in 3D vision that integrates volumetric 3D reconstruction with semantic understanding. Existing methods, however, often rely on modular pipelines. These modules are typically optimized independently or use pre-configured inputs, leading to cascading errors. In this paper, we address this limitation by designing a novel causal loss that enables holistic, end-to-end supervision of the modular 2D-to-3D transformation pipeline. Grounded in the principle of 2D-to-3D semantic causality, this loss regulates the gradient flow from 3D voxel representations back to the 2D features. Consequently, it renders the entire pipeline differentiable, unifying the learning process and making previously non-trainable components fully learnable. Building on this principle, we propose the Semantic Causality-Aware 2D-to-3D Transformation, which comprises three components guided by our causal loss: Channel-Grouped Lifting for adaptive semantic mapping, Learnable Camera Offsets for enhanced robustness against camera perturbations, and Normalized Convolution for effective feature propagation. Extensive experiments demonstrate that our method achieves state-of-the-art performance on the Occ3D benchmark, demonstrating significant robustness to camera perturbations and improved 2D-to-3D semantic consistency.
>
---
#### [new 024] Symmetry Interactive Transformer with CNN Framework for Diagnosis of Alzheimer's Disease Using Structural MRI
- **分类: cs.CV**

- **简介: 该论文提出一种结合3D CNN和Symmetry Interactive Transformer的端到端网络，用于阿尔茨海默病的诊断。通过关注左右脑不对称萎缩区域，提升诊断准确率至92.5%，解决传统方法忽略脑部不对称性的问题。**

- **链接: [http://arxiv.org/pdf/2509.08243v1](http://arxiv.org/pdf/2509.08243v1)**

> **作者:** Zheng Yang; Yanteng Zhang; Xupeng Kou; Yang Liu; Chao Ren
>
> **摘要:** Structural magnetic resonance imaging (sMRI) combined with deep learning has achieved remarkable progress in the prediction and diagnosis of Alzheimer's disease (AD). Existing studies have used CNN and transformer to build a well-performing network, but most of them are based on pretraining or ignoring the asymmetrical character caused by brain disorders. We propose an end-to-end network for the detection of disease-based asymmetric induced by left and right brain atrophy which consist of 3D CNN Encoder and Symmetry Interactive Transformer (SIT). Following the inter-equal grid block fetch operation, the corresponding left and right hemisphere features are aligned and subsequently fed into the SIT for diagnostic analysis. SIT can help the model focus more on the regions of asymmetry caused by structural changes, thus improving diagnostic performance. We evaluated our method based on the ADNI dataset, and the results show that the method achieves better diagnostic accuracy (92.5\%) compared to several CNN methods and CNNs combined with a general transformer. The visualization results show that our network pays more attention in regions of brain atrophy, especially for the asymmetric pathological characteristics induced by AD, demonstrating the interpretability and effectiveness of the method.
>
---
#### [new 025] Implicit Shape-Prior for Few-Shot Assisted 3D Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于3D医学图像分割任务，旨在减少人工标注负担。提出隐式形状先验方法，结合少量切片标注实现多器官分割，并自动选择关键切片以提升效率。实验验证了其在脑癌风险器官分割和肌少症数据库构建中的有效性。**

- **链接: [http://arxiv.org/pdf/2509.08580v1](http://arxiv.org/pdf/2509.08580v1)**

> **作者:** Mathilde Monvoisin; Louise Piecuch; Blanche Texier; Cédric Hémon; Anaïs Barateau; Jérémie Huet; Antoine Nordez; Anne-Sophie Boureau; Jean-Claude Nunes; Diana Mateus
>
> **备注:** Both first Authors contributed equally to this work, lastnames in alphabetical order. This preprint has not undergone peer review or any post-submission improvements or corrections. The Version of Record of this contribution will be published in a Springer Nature Computer Science book series (CCIS, LNAI, LNBI, LNBIP, LNCS) and the doi will soon be released
>
> **摘要:** The objective of this paper is to significantly reduce the manual workload required from medical professionals in complex 3D segmentation tasks that cannot be yet fully automated. For instance, in radiotherapy planning, organs at risk must be accurately identified in computed tomography (CT) or magnetic resonance imaging (MRI) scans to ensure they are spared from harmful radiation. Similarly, diagnosing age-related degenerative diseases such as sarcopenia, which involve progressive muscle volume loss and strength, is commonly based on muscular mass measurements often obtained from manual segmentation of medical volumes. To alleviate the manual-segmentation burden, this paper introduces an implicit shape prior to segment volumes from sparse slice manual annotations generalized to the multi-organ case, along with a simple framework for automatically selecting the most informative slices to guide and minimize the next interactions. The experimental validation shows the method's effectiveness on two medical use cases: assisted segmentation in the context of at risks organs for brain cancer patients, and acceleration of the creation of a new database with unseen muscle shapes for patients with sarcopenia.
>
---
#### [new 026] APML: Adaptive Probabilistic Matching Loss for Robust 3D Point Cloud Reconstruction
- **分类: cs.CV; cs.AI**

- **简介: 论文提出APML损失函数，用于改进3D点云重建任务中的匹配损失。针对现有损失函数存在的点拥堵、非微分操作及计算复杂等问题，APML通过温度缩放相似矩阵和Sinkhorn迭代实现可微的一对一匹配，提升重建质量与效率。**

- **链接: [http://arxiv.org/pdf/2509.08104v1](http://arxiv.org/pdf/2509.08104v1)**

> **作者:** Sasan Sharifipour; Constantino Álvarez Casado; Mohammad Sabokrou; Miguel Bordallo López
>
> **备注:** 22 pages, 6 figures, conference, 7 tables, 15 formulas
>
> **摘要:** Training deep learning models for point cloud prediction tasks such as shape completion and generation depends critically on loss functions that measure discrepancies between predicted and ground-truth point sets. Commonly used functions such as Chamfer Distance (CD), HyperCD, and InfoCD rely on nearest-neighbor assignments, which often induce many-to-one correspondences, leading to point congestion in dense regions and poor coverage in sparse regions. These losses also involve non-differentiable operations due to index selection, which may affect gradient-based optimization. Earth Mover Distance (EMD) enforces one-to-one correspondences and captures structural similarity more effectively, but its cubic computational complexity limits its practical use. We propose the Adaptive Probabilistic Matching Loss (APML), a fully differentiable approximation of one-to-one matching that leverages Sinkhorn iterations on a temperature-scaled similarity matrix derived from pairwise distances. We analytically compute the temperature to guarantee a minimum assignment probability, eliminating manual tuning. APML achieves near-quadratic runtime, comparable to Chamfer-based losses, and avoids non-differentiable operations. When integrated into state-of-the-art architectures (PoinTr, PCN, FoldingNet) on ShapeNet benchmarks and on a spatiotemporal Transformer (CSI2PC) that generates 3D human point clouds from WiFi CSI measurements, APM loss yields faster convergence, superior spatial distribution, especially in low-density regions, and improved or on-par quantitative performance without additional hyperparameter search. The code is available at: https://github.com/apm-loss/apml.
>
---
#### [new 027] Handling Multiple Hypotheses in Coarse-to-Fine Dense Image Matching
- **分类: cs.CV**

- **简介: 该论文属于密集图像匹配任务，旨在解决在深度不连续或强缩放情况下单假设匹配易出错的问题。提出BEAMER架构，通过每尺度预测多个假设并集成到交叉注意力层中，提升匹配鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.08805v1](http://arxiv.org/pdf/2509.08805v1)**

> **作者:** Matthieu Vilain; Rémi Giraud; Yannick Berthoumieu; Guillaume Bourmaud
>
> **摘要:** Dense image matching aims to find a correspondent for every pixel of a source image in a partially overlapping target image. State-of-the-art methods typically rely on a coarse-to-fine mechanism where a single correspondent hypothesis is produced per source location at each scale. In challenging cases -- such as at depth discontinuities or when the target image is a strong zoom-in of the source image -- the correspondents of neighboring source locations are often widely spread and predicting a single correspondent hypothesis per source location at each scale may lead to erroneous matches. In this paper, we investigate the idea of predicting multiple correspondent hypotheses per source location at each scale instead. We consider a beam search strategy to propagat multiple hypotheses at each scale and propose integrating these multiple hypotheses into cross-attention layers, resulting in a novel dense matching architecture called BEAMER. BEAMER learns to preserve and propagate multiple hypotheses across scales, making it significantly more robust than state-of-the-art methods, especially at depth discontinuities or when the target image is a strong zoom-in of the source image.
>
---
#### [new 028] Computational Imaging for Enhanced Computer Vision
- **分类: cs.CV**

- **简介: 该论文综述计算成像技术对计算机视觉的影响，分析其在图像增强与重建中的应用，解决传统成像在复杂环境下的性能限制，探讨其与核心视觉任务的协同关系，并指出未来研究方向。**

- **链接: [http://arxiv.org/pdf/2509.08712v1](http://arxiv.org/pdf/2509.08712v1)**

> **作者:** Humera Shaikh; Kaur Jashanpreet
>
> **摘要:** This paper presents a comprehensive survey of computational imaging (CI) techniques and their transformative impact on computer vision (CV) applications. Conventional imaging methods often fail to deliver high-fidelity visual data in challenging conditions, such as low light, motion blur, or high dynamic range scenes, thereby limiting the performance of state-of-the-art CV systems. Computational imaging techniques, including light field imaging, high dynamic range (HDR) imaging, deblurring, high-speed imaging, and glare mitigation, address these limitations by enhancing image acquisition and reconstruc- tion processes. This survey systematically explores the synergies between CI techniques and core CV tasks, including object detection, depth estimation, optical flow, face recognition, and keypoint detection. By analyzing the relationships between CI methods and their practical contributions to CV applications, this work highlights emerging opportunities, challenges, and future research directions. We emphasize the potential for task-specific, adaptive imaging pipelines that improve robustness, accuracy, and efficiency in real-world scenarios, such as autonomous navigation, surveillance, augmented reality, and robotics.
>
---
#### [new 029] Multi-Modal Robust Enhancement for Coastal Water Segmentation: A Systematic HSV-Guided Framework
- **分类: cs.CV**

- **简介: 该论文提出一种基于HSV颜色空间的多模态鲁棒增强框架——Robust U-Net，用于提升卫星图像中沿海水域分割的稳定性与精度。通过五种协同组件优化分割效果，显著提升模型泛化能力与计算效率。**

- **链接: [http://arxiv.org/pdf/2509.08694v1](http://arxiv.org/pdf/2509.08694v1)**

> **作者:** Zhen Tian; Christos Anagnostopoulos; Qiyuan Wang; Zhiwei Gao
>
> **摘要:** Coastal water segmentation from satellite imagery presents unique challenges due to complex spectral characteristics and irregular boundary patterns. Traditional RGB-based approaches often suffer from training instability and poor generalization in diverse maritime environments. This paper introduces a systematic robust enhancement framework, referred to as Robust U-Net, that leverages HSV color space supervision and multi-modal constraints for improved coastal water segmentation. Our approach integrates five synergistic components: HSV-guided color supervision, gradient-based coastline optimization, morphological post-processing, sea area cleanup, and connectivity control. Through comprehensive ablation studies, we demonstrate that HSV supervision provides the highest impact (0.85 influence score), while the complete framework achieves superior training stability (84\% variance reduction) and enhanced segmentation quality. Our method shows consistent improvements across multiple evaluation metrics while maintaining computational efficiency. For reproducibility, our training configurations and code are available here: https://github.com/UofgCoastline/ICASSP-2026-Robust-Unet.
>
---
#### [new 030] Sparse Transformer for Ultra-sparse Sampled Video Compressive Sensing
- **分类: cs.CV**

- **简介: 论文提出一种稀疏Transformer（BSTFormer）用于超稀疏采样视频压缩感知。旨在解决高帧率 gigapixel 相机的高能耗问题，通过 Ultra-Sparse Sampling (USS) 策略和 DMD 编码系统实现高效视频恢复，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.08228v1](http://arxiv.org/pdf/2509.08228v1)**

> **作者:** Miao Cao; Siming Zheng; Lishun Wang; Ziyang Chen; David Brady; Xin Yuan
>
> **摘要:** Digital cameras consume ~0.1 microjoule per pixel to capture and encode video, resulting in a power usage of ~20W for a 4K sensor operating at 30 fps. Imagining gigapixel cameras operating at 100-1000 fps, the current processing model is unsustainable. To address this, physical layer compressive measurement has been proposed to reduce power consumption per pixel by 10-100X. Video Snapshot Compressive Imaging (SCI) introduces high frequency modulation in the optical sensor layer to increase effective frame rate. A commonly used sampling strategy of video SCI is Random Sampling (RS) where each mask element value is randomly set to be 0 or 1. Similarly, image inpainting (I2P) has demonstrated that images can be recovered from a fraction of the image pixels. Inspired by I2P, we propose Ultra-Sparse Sampling (USS) regime, where at each spatial location, only one sub-frame is set to 1 and all others are set to 0. We then build a Digital Micro-mirror Device (DMD) encoding system to verify the effectiveness of our USS strategy. Ideally, we can decompose the USS measurement into sub-measurements for which we can utilize I2P algorithms to recover high-speed frames. However, due to the mismatch between the DMD and CCD, the USS measurement cannot be perfectly decomposed. To this end, we propose BSTFormer, a sparse TransFormer that utilizes local Block attention, global Sparse attention, and global Temporal attention to exploit the sparsity of the USS measurement. Extensive results on both simulated and real-world data show that our method significantly outperforms all previous state-of-the-art algorithms. Additionally, an essential advantage of the USS strategy is its higher dynamic range than that of the RS strategy. Finally, from the application perspective, the USS strategy is a good choice to implement a complete video SCI system on chip due to its fixed exposure time.
>
---
#### [new 031] HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning
- **分类: cs.CV; cs.MM**

- **简介: 该论文提出HuMo框架，解决多模态协同生成人类中心视频的问题。通过构建高质量数据集与两阶段训练策略，实现文本、图像和音频的联合控制，提升视频生成质量与同步性。**

- **链接: [http://arxiv.org/pdf/2509.08519v1](http://arxiv.org/pdf/2509.08519v1)**

> **作者:** Liyang Chen; Tianxiang Ma; Jiawei Liu; Bingchuan Li; Zhuowei Chen; Lijie Liu; Xu He; Gen Li; Qian He; Zhiyong Wu
>
> **摘要:** Human-Centric Video Generation (HCVG) methods seek to synthesize human videos from multimodal inputs, including text, image, and audio. Existing methods struggle to effectively coordinate these heterogeneous modalities due to two challenges: the scarcity of training data with paired triplet conditions and the difficulty of collaborating the sub-tasks of subject preservation and audio-visual sync with multimodal inputs. In this work, we present HuMo, a unified HCVG framework for collaborative multimodal control. For the first challenge, we construct a high-quality dataset with diverse and paired text, reference images, and audio. For the second challenge, we propose a two-stage progressive multimodal training paradigm with task-specific strategies. For the subject preservation task, to maintain the prompt following and visual generation abilities of the foundation model, we adopt the minimal-invasive image injection strategy. For the audio-visual sync task, besides the commonly adopted audio cross-attention layer, we propose a focus-by-predicting strategy that implicitly guides the model to associate audio with facial regions. For joint learning of controllabilities across multimodal inputs, building on previously acquired capabilities, we progressively incorporate the audio-visual sync task. During inference, for flexible and fine-grained multimodal control, we design a time-adaptive Classifier-Free Guidance strategy that dynamically adjusts guidance weights across denoising steps. Extensive experimental results demonstrate that HuMo surpasses specialized state-of-the-art methods in sub-tasks, establishing a unified framework for collaborative multimodal-conditioned HCVG. Project Page: https://phantom-video.github.io/HuMo.
>
---
#### [new 032] MESH -- Understanding Videos Like Human: Measuring Hallucinations in Large Video Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MESH基准，用于系统评估大视频模型中的幻觉问题。通过问答框架，从基础对象到复杂动作对进行测试，揭示模型在细粒度和多主体场景下的幻觉倾向，改进视频理解的准确性。**

- **链接: [http://arxiv.org/pdf/2509.08538v1](http://arxiv.org/pdf/2509.08538v1)**

> **作者:** Garry Yang; Zizhe Chen; Man Hon Wong; Haoyu Lei; Yongqiang Chen; Zhenguo Li; Kaiwen Zhou; James Cheng
>
> **摘要:** Large Video Models (LVMs) build on the semantic capabilities of Large Language Models (LLMs) and vision modules by integrating temporal information to better understand dynamic video content. Despite their progress, LVMs are prone to hallucinations-producing inaccurate or irrelevant descriptions. Current benchmarks for video hallucination depend heavily on manual categorization of video content, neglecting the perception-based processes through which humans naturally interpret videos. We introduce MESH, a benchmark designed to evaluate hallucinations in LVMs systematically. MESH uses a Question-Answering framework with binary and multi-choice formats incorporating target and trap instances. It follows a bottom-up approach, evaluating basic objects, coarse-to-fine subject features, and subject-action pairs, aligning with human video understanding. We demonstrate that MESH offers an effective and comprehensive approach for identifying hallucinations in videos. Our evaluations show that while LVMs excel at recognizing basic objects and features, their susceptibility to hallucinations increases markedly when handling fine details or aligning multiple actions involving various subjects in longer videos.
>
---
#### [new 033] An Explainable Deep Neural Network with Frequency-Aware Channel and Spatial Refinement for Flood Prediction in Sustainable Cities
- **分类: cs.CV**

- **简介: 该论文提出XFloodNet，用于城市洪水分类任务。针对传统方法在多模态数据融合与噪声环境下的不足，设计了跨模态注意力、频域特征提取和级联特征优化模块，显著提升了洪水预测性能。**

- **链接: [http://arxiv.org/pdf/2509.08003v1](http://arxiv.org/pdf/2509.08003v1)**

> **作者:** Shahid Shafi Dar; Bharat Kaurav; Arnav Jain; Chandravardhan Singh Raghaw; Mohammad Zia Ur Rehman; Nagendra Kumar
>
> **摘要:** In an era of escalating climate change, urban flooding has emerged as a critical challenge for sustainable cities, threatening lives, infrastructure, and ecosystems. Traditional flood detection methods are constrained by their reliance on unimodal data and static rule-based systems, which fail to capture the dynamic, non-linear relationships inherent in flood events. Furthermore, existing attention mechanisms and ensemble learning approaches exhibit limitations in hierarchical refinement, cross-modal feature integration, and adaptability to noisy or unstructured environments, resulting in suboptimal flood classification performance. To address these challenges, we present XFloodNet, a novel framework that redefines urban flood classification through advanced deep-learning techniques. XFloodNet integrates three novel components: (1) a Hierarchical Cross-Modal Gated Attention mechanism that dynamically aligns visual and textual features, enabling precise multi-granularity interactions and resolving contextual ambiguities; (2) a Heterogeneous Convolutional Adaptive Multi-Scale Attention module, which leverages frequency-enhanced channel attention and frequency-modulated spatial attention to extract and prioritize discriminative flood-related features across spectral and spatial domains; and (3) a Cascading Convolutional Transformer Feature Refinement technique that harmonizes hierarchical features through adaptive scaling and cascading operations, ensuring robust and noise-resistant flood detection. We evaluate our proposed method on three benchmark datasets, such as Chennai Floods, Rhine18 Floods, and Harz17 Floods, XFloodNet achieves state-of-the-art F1-scores of 93.33%, 82.24%, and 88.60%, respectively, surpassing existing methods by significant margins.
>
---
#### [new 034] Skeleton-based sign language recognition using a dual-stream spatio-temporal dynamic graph convolutional network
- **分类: cs.CV; cs.AI; I.2.m; I.2.0**

- **简介: 该论文属于手语识别任务，旨在解决形态相似但语义不同的手势识别难题。提出DSLNet双流网络，分别建模手势形态与轨迹，通过几何融合机制提升性能，在多个数据集上取得新SOTA。**

- **链接: [http://arxiv.org/pdf/2509.08661v1](http://arxiv.org/pdf/2509.08661v1)**

> **作者:** Liangjin Liu; Haoyang Zheng; Pei Zhou
>
> **备注:** 5 pages, 3 figures, ICASSP
>
> **摘要:** Isolated Sign Language Recognition (ISLR) is challenged by gestures that are morphologically similar yet semantically distinct, a problem rooted in the complex interplay between hand shape and motion trajectory. Existing methods, often relying on a single reference frame, struggle to resolve this geometric ambiguity. This paper introduces Dual-SignLanguageNet (DSLNet), a dual-reference, dual-stream architecture that decouples and models gesture morphology and trajectory in separate, complementary coordinate systems. Our approach utilizes a wrist-centric frame for view-invariant shape analysis and a facial-centric frame for context-aware trajectory modeling. These streams are processed by specialized networks-a topology-aware graph convolution for shape and a Finsler geometry-based encoder for trajectory-and are integrated via a geometry-driven optimal transport fusion mechanism. DSLNet sets a new state-of-the-art, achieving 93.70%, 89.97% and 99.79% accuracy on the challenging WLASL-100, WLASL-300 and LSA64 datasets, respectively, with significantly fewer parameters than competing models.
>
---
#### [new 035] GTA-Crime: A Synthetic Dataset and Generation Framework for Fatal Violence Detection with Adversarial Snippet-Level Domain Adaptation
- **分类: cs.CV**

- **简介: 该论文属于视频异常检测任务，旨在解决致命暴力事件（如枪击、刺杀）检测因数据稀缺而难以训练的问题。研究构建了GTA-Crime合成数据集与生成框架，并提出片段级对抗域适应策略，提升真实场景检测性能。**

- **链接: [http://arxiv.org/pdf/2509.08232v1](http://arxiv.org/pdf/2509.08232v1)**

> **作者:** Seongho Kim; Sejong Ryu; Hyoukjun You; Je Hyeong Hong
>
> **摘要:** Recent advancements in video anomaly detection (VAD) have enabled identification of various criminal activities in surveillance videos, but detecting fatal incidents such as shootings and stabbings remains difficult due to their rarity and ethical issues in data collection. Recognizing this limitation, we introduce GTA-Crime, a fatal video anomaly dataset and generation framework using Grand Theft Auto 5 (GTA5). Our dataset contains fatal situations such as shootings and stabbings, captured from CCTV multiview perspectives under diverse conditions including action types, weather, time of day, and viewpoints. To address the rarity of such scenarios, we also release a framework for generating these types of videos. Additionally, we propose a snippet-level domain adaptation strategy using Wasserstein adversarial training to bridge the gap between synthetic GTA-Crime features and real-world features like UCF-Crime. Experimental results validate our GTA-Crime dataset and demonstrate that incorporating GTA-Crime with our domain adaptation strategy consistently enhances real world fatal violence detection accuracy. Our dataset and the data generation framework are publicly available at https://github.com/ta-ho/GTA-Crime.
>
---
#### [new 036] Hyperspectral Mamba for Hyperspectral Object Tracking
- **分类: cs.CV**

- **简介: 该论文提出HyMamba网络用于高光谱目标跟踪任务，旨在解决现有方法未能有效捕捉光谱信息、时间依赖性和跨深度交互的问题。通过引入SSM模块，实现光谱、时空特征的统一建模，实验表明其在多个数据集上达到最优性能。**

- **链接: [http://arxiv.org/pdf/2509.08265v1](http://arxiv.org/pdf/2509.08265v1)**

> **作者:** Long Gao; Yunhe Zhang; Yan Jiang; Weiying Xie; Yunsong Li
>
> **摘要:** Hyperspectral object tracking holds great promise due to the rich spectral information and fine-grained material distinctions in hyperspectral images, which are beneficial in challenging scenarios. While existing hyperspectral trackers have made progress by either transforming hyperspectral data into false-color images or incorporating modality fusion strategies, they often fail to capture the intrinsic spectral information, temporal dependencies, and cross-depth interactions. To address these limitations, a new hyperspectral object tracking network equipped with Mamba (HyMamba), is proposed. It unifies spectral, cross-depth, and temporal modeling through state space modules (SSMs). The core of HyMamba lies in the Spectral State Integration (SSI) module, which enables progressive refinement and propagation of spectral features with cross-depth and temporal spectral information. Embedded within each SSI, the Hyperspectral Mamba (HSM) module is introduced to learn spatial and spectral information synchronously via three directional scanning SSMs. Based on SSI and HSM, HyMamba constructs joint features from false-color and hyperspectral inputs, and enhances them through interaction with original spectral features extracted from raw hyperspectral images. Extensive experiments conducted on seven benchmark datasets demonstrate that HyMamba achieves state-of-the-art performance. For instance, it achieves 73.0\% of the AUC score and 96.3\% of the DP@20 score on the HOTC2020 dataset. The code will be released at https://github.com/lgao001/HyMamba.
>
---
#### [new 037] EVDI++: Event-based Video Deblurring and Interpolation via Self-Supervised Learning
- **分类: cs.CV**

- **简介: 该论文提出EVDI++，用于事件相机的视频去模糊与插值。通过自监督学习，利用事件相机高时间分辨率，解决传统相机因长曝光导致的运动模糊与信息丢失问题，实现高质量中间帧预测。**

- **链接: [http://arxiv.org/pdf/2509.08260v1](http://arxiv.org/pdf/2509.08260v1)**

> **作者:** Chi Zhang; Xiang Zhang; Chenxu Jiang; Gui-Song Xia; Lei Yu
>
> **备注:** 18 pages
>
> **摘要:** Frame-based cameras with extended exposure times often produce perceptible visual blurring and information loss between frames, significantly degrading video quality. To address this challenge, we introduce EVDI++, a unified self-supervised framework for Event-based Video Deblurring and Interpolation that leverages the high temporal resolution of event cameras to mitigate motion blur and enable intermediate frame prediction. Specifically, the Learnable Double Integral (LDI) network is designed to estimate the mapping relation between reference frames and sharp latent images. Then, we refine the coarse results and optimize overall training efficiency by introducing a learning-based division reconstruction module, enabling images to be converted with varying exposure intervals. We devise an adaptive parameter-free fusion strategy to obtain the final results, utilizing the confidence embedded in the LDI outputs of concurrent events. A self-supervised learning framework is proposed to enable network training with real-world blurry videos and events by exploring the mutual constraints among blurry frames, latent images, and event streams. We further construct a dataset with real-world blurry images and events using a DAVIS346c camera, demonstrating the generalizability of the proposed EVDI++ in real-world scenarios. Extensive experiments on both synthetic and real-world datasets show that our method achieves state-of-the-art performance in video deblurring and interpolation tasks.
>
---
#### [new 038] EfficientIML: Efficient High-Resolution Image Manipulation Localization
- **分类: cs.CV**

- **简介: 论文提出EfficientIML模型，用于高分辨率图像篡改定位。针对扩散模型生成的新型伪造图像，构建SIF数据集，并设计轻量级三阶段EfficientRWKV网络，在保证性能的同时提升推理速度，适用于实时图像取证任务。**

- **链接: [http://arxiv.org/pdf/2509.08583v1](http://arxiv.org/pdf/2509.08583v1)**

> **作者:** Jinhan Li; Haoyang He; Lei Xie; Jiangning Zhang
>
> **摘要:** With imaging devices delivering ever-higher resolutions and the emerging diffusion-based forgery methods, current detectors trained only on traditional datasets (with splicing, copy-moving and object removal forgeries) lack exposure to this new manipulation type. To address this, we propose a novel high-resolution SIF dataset of 1200+ diffusion-generated manipulations with semantically extracted masks. However, this also imposes a challenge on existing methods, as they face significant computational resource constraints due to their prohibitive computational complexities. Therefore, we propose a novel EfficientIML model with a lightweight, three-stage EfficientRWKV backbone. EfficientRWKV's hybrid state-space and attention network captures global context and local details in parallel, while a multi-scale supervision strategy enforces consistency across hierarchical predictions. Extensive evaluations on our dataset and standard benchmarks demonstrate that our approach outperforms ViT-based and other SOTA lightweight baselines in localization performance, FLOPs and inference speed, underscoring its suitability for real-time forensic applications.
>
---
#### [new 039] Examining Vision Language Models through Multi-dimensional Experiments with Vision and Text Features
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型（VLMs）在处理图像和文本时的表现差异，分析其对图像特征和提示词的依赖。通过多维实验，探讨输入参数如何影响模型行为，揭示其性能变化原因。**

- **链接: [http://arxiv.org/pdf/2509.08266v1](http://arxiv.org/pdf/2509.08266v1)**

> **作者:** Saurav Sengupta; Nazanin Moradinasab; Jiebei Liu; Donald E. Brown
>
> **摘要:** Recent research on Vision Language Models (VLMs) suggests that they rely on inherent biases learned during training to respond to questions about visual properties of an image. These biases are exacerbated when VLMs are asked highly specific questions that require focusing on specific areas of the image. For example, a VLM tasked with counting stars on a modified American flag (e.g., with more than 50 stars) will often disregard the visual evidence and fail to answer accurately. We build upon this research and develop a multi-dimensional examination framework to systematically determine which characteristics of the input data, including both the image and the accompanying prompt, lead to such differences in performance. Using open-source VLMs, we further examine how attention values fluctuate with varying input parameters (e.g., image size, number of objects in the image, background color, prompt specificity). This research aims to learn how the behavior of vision language models changes and to explore methods for characterizing such changes. Our results suggest, among other things, that even minor modifications in image characteristics and prompt specificity can lead to large changes in how a VLM formulates its answer and, subsequently, its overall performance.
>
---
#### [new 040] Maximally Useful and Minimally Redundant: The Key to Self Supervised Learning for Imbalanced Data
- **分类: cs.CV**

- **简介: 论文研究自监督学习在不平衡数据集中的应用，提出多视角框架以提升尾部类别表现。通过互信息理论设计损失函数，过滤极端特征，实现更优表示学习，在多个数据集上取得新SOTA结果。属于图像分类任务，解决数据不平衡问题。**

- **链接: [http://arxiv.org/pdf/2509.08469v1](http://arxiv.org/pdf/2509.08469v1)**

> **作者:** Yash Kumar Sharma; Vineet Nair; Wilson Naik
>
> **摘要:** The robustness of contrastive self-supervised learning (CSSL) for imbalanced datasets is largely unexplored. CSSL usually makes use of \emph{multi-view} assumptions to learn discriminatory features via similar and dissimilar data samples. CSSL works well on balanced datasets, but does not generalize well for imbalanced datasets. In a very recent paper, as part of future work, Yann LeCun pointed out that the self-supervised multiview framework can be extended to cases involving \emph{more than two views}. Taking a cue from this insight we propose a theoretical justification based on the concept of \emph{mutual information} to support the \emph{more than two views} objective and apply it to the problem of dataset imbalance in self-supervised learning. The proposed method helps extract representative characteristics of the tail classes by segregating between \emph{intra} and \emph{inter} discriminatory characteristics. We introduce a loss function that helps us to learn better representations by filtering out extreme features. Experimental evaluation on a variety of self-supervised frameworks (both contrastive and non-contrastive) also prove that the \emph{more than two view} objective works well for imbalanced datasets. We achieve a new state-of-the-art accuracy in self-supervised imbalanced dataset classification (2\% improvement in Cifar10-LT using Resnet-18, 5\% improvement in Cifar100-LT using Resnet-18, 3\% improvement in Imagenet-LT (1k) using Resnet-50).
>
---
#### [new 041] Two-Stage Swarm Intelligence Ensemble Deep Transfer Learning (SI-EDTL) for Vehicle Detection Using Unmanned Aerial Vehicles
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SI-EDTL模型，用于无人机图像中的多车辆检测任务。通过集成三种预训练特征提取器与五种分类器，结合鲸鱼优化算法优化参数，实现对汽车、卡车等目标的高精度分类，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2509.08026v1](http://arxiv.org/pdf/2509.08026v1)**

> **作者:** Zeinab Ghasemi Darehnaei; Mohammad Shokouhifar; Hossein Yazdanjouei; S. M. J. Rastegar Fatemi
>
> **摘要:** This paper introduces SI-EDTL, a two-stage swarm intelligence ensemble deep transfer learning model for detecting multiple vehicles in UAV images. It combines three pre-trained Faster R-CNN feature extractor models (InceptionV3, ResNet50, GoogLeNet) with five transfer classifiers (KNN, SVM, MLP, C4.5, Na\"ive Bayes), resulting in 15 different base learners. These are aggregated via weighted averaging to classify regions as Car, Van, Truck, Bus, or background. Hyperparameters are optimized with the whale optimization algorithm to balance accuracy, precision, and recall. Implemented in MATLAB R2020b with parallel processing, SI-EDTL outperforms existing methods on the AU-AIR UAV dataset.
>
---
#### [new 042] Sparse BEV Fusion with Self-View Consistency for Multi-View Detection and Tracking
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多视角目标检测与跟踪任务，旨在解决跨摄像头身份一致性问题。提出SCFusion框架，通过稀疏投影、密度感知融合和多视角一致性损失，提升BEV特征融合效果，显著提高检测与跟踪精度。**

- **链接: [http://arxiv.org/pdf/2509.08421v1](http://arxiv.org/pdf/2509.08421v1)**

> **作者:** Keisuke Toida; Taigo Sakai; Naoki Kato; Kazutoyo Yokota; Takeshi Nakamura; Kazuhiro Hotta
>
> **摘要:** Multi-View Multi-Object Tracking (MVMOT) is essential for applications such as surveillance, autonomous driving, and sports analytics. However, maintaining consistent object identities across multiple cameras remains challenging due to viewpoint changes, lighting variations, and occlusions, which often lead to tracking errors.Recent methods project features from multiple cameras into a unified Bird's-Eye-View (BEV) space to improve robustness against occlusion. However, this projection introduces feature distortion and non-uniform density caused by variations in object scale with distance. These issues degrade the quality of the fused representation and reduce detection and tracking accuracy.To address these problems, we propose SCFusion, a framework that combines three techniques to improve multi-view feature integration. First, it applies a sparse transformation to avoid unnatural interpolation during projection. Next, it performs density-aware weighting to adaptively fuse features based on spatial confidence and camera distance. Finally, it introduces a multi-view consistency loss that encourages each camera to learn discriminative features independently before fusion.Experiments show that SCFusion achieves state-of-the-art performance, reaching an IDF1 score of 95.9% on WildTrack and a MODP of 89.2% on MultiviewX, outperforming the baseline method TrackTacular. These results demonstrate that SCFusion effectively mitigates the limitations of conventional BEV projection and provides a robust and accurate solution for multi-view object detection and tracking.
>
---
#### [new 043] CLAPS: A CLIP-Unified Auto-Prompt Segmentation for Multi-Modal Retinal Imaging
- **分类: cs.CV; I.4.6**

- **简介: 该论文提出CLAPS方法，解决多模态视网膜图像分割中的模态模糊、依赖手动提示和缺乏统一框架的问题。通过预训练CLIP编码器、自动生空间提示和增强文本提示，实现自动化统一分割，提升模型泛化能力。属于医学图像分割任务。**

- **链接: [http://arxiv.org/pdf/2509.08618v1](http://arxiv.org/pdf/2509.08618v1)**

> **作者:** Zhihao Zhao; Yinzheng Zhao; Junjie Yang; Xiangtong Yao; Quanmin Liang; Shahrooz Faghihroohi; Kai Huang; Nassir Navab; M. Ali Nasseri
>
> **备注:** BIBM
>
> **摘要:** Recent advancements in foundation models, such as the Segment Anything Model (SAM), have significantly impacted medical image segmentation, especially in retinal imaging, where precise segmentation is vital for diagnosis. Despite this progress, current methods face critical challenges: 1) modality ambiguity in textual disease descriptions, 2) a continued reliance on manual prompting for SAM-based workflows, and 3) a lack of a unified framework, with most methods being modality- and task-specific. To overcome these hurdles, we propose CLIP-unified Auto-Prompt Segmentation (\CLAPS), a novel method for unified segmentation across diverse tasks and modalities in retinal imaging. Our approach begins by pre-training a CLIP-based image encoder on a large, multi-modal retinal dataset to handle data scarcity and distribution imbalance. We then leverage GroundingDINO to automatically generate spatial bounding box prompts by detecting local lesions. To unify tasks and resolve ambiguity, we use text prompts enhanced with a unique "modality signature" for each imaging modality. Ultimately, these automated textual and spatial prompts guide SAM to execute precise segmentation, creating a fully automated and unified pipeline. Extensive experiments on 12 diverse datasets across 11 critical segmentation categories show that CLAPS achieves performance on par with specialized expert models while surpassing existing benchmarks across most metrics, demonstrating its broad generalizability as a foundation model.
>
---
#### [new 044] InsFusion: Rethink Instance-level LiDAR-Camera Fusion for 3D Object Detection
- **分类: cs.CV**

- **简介: 论文提出InsFusion方法，用于解决多视角相机与LiDAR融合中的误差累积问题。通过从原始和融合特征中提取提议，并利用注意力机制优化特征查询，提升3D目标检测性能。属于自动驾驶中的3D目标检测任务。**

- **链接: [http://arxiv.org/pdf/2509.08374v1](http://arxiv.org/pdf/2509.08374v1)**

> **作者:** Zhongyu Xia; Hansong Yang; Yongtao Wang
>
> **摘要:** Three-dimensional Object Detection from multi-view cameras and LiDAR is a crucial component for autonomous driving and smart transportation. However, in the process of basic feature extraction, perspective transformation, and feature fusion, noise and error will gradually accumulate. To address this issue, we propose InsFusion, which can extract proposals from both raw and fused features and utilizes these proposals to query the raw features, thereby mitigating the impact of accumulated errors. Additionally, by incorporating attention mechanisms applied to the raw features, it thereby mitigates the impact of accumulated errors. Experiments on the nuScenes dataset demonstrate that InsFusion is compatible with various advanced baseline methods and delivers new state-of-the-art performance for 3D object detection.
>
---
#### [new 045] GeneVA: A Dataset of Human Annotations for Generative Text to Video Artifacts
- **分类: cs.CV**

- **简介: 该论文提出GeneVA数据集，用于评估生成视频中的时空异常。属于视频生成质量评估任务，解决生成视频中物理不合理和时间不一致的问题，通过大量人工标注提供基准。**

- **链接: [http://arxiv.org/pdf/2509.08818v1](http://arxiv.org/pdf/2509.08818v1)**

> **作者:** Jenna Kang; Maria Silva; Patsorn Sangkloy; Kenneth Chen; Niall Williams; Qi Sun
>
> **摘要:** Recent advances in probabilistic generative models have extended capabilities from static image synthesis to text-driven video generation. However, the inherent randomness of their generation process can lead to unpredictable artifacts, such as impossible physics and temporal inconsistency. Progress in addressing these challenges requires systematic benchmarks, yet existing datasets primarily focus on generative images due to the unique spatio-temporal complexities of videos. To bridge this gap, we introduce GeneVA, a large-scale artifact dataset with rich human annotations that focuses on spatio-temporal artifacts in videos generated from natural text prompts. We hope GeneVA can enable and assist critical applications, such as benchmarking model performance and improving generative video quality.
>
---
#### [new 046] A Structured Review of Underwater Object Detection Challenges and Solutions: From Traditional to Large Vision Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文综述了水下目标检测（UOD）的挑战与解决方案，从传统方法到大视觉语言模型（LVLMs）。其任务是系统分析UOD问题，提出改进方法，并探讨LVLMs的应用潜力，以提升检测性能。**

- **链接: [http://arxiv.org/pdf/2509.08490v1](http://arxiv.org/pdf/2509.08490v1)**

> **作者:** Edwine Nabahirwa; Wei Song; Minghua Zhang; Yi Fang; Zhou Ni
>
> **备注:** 72 Pages, 11 Figures
>
> **摘要:** Underwater object detection (UOD) is vital to diverse marine applications, including oceanographic research, underwater robotics, and marine conservation. However, UOD faces numerous challenges that compromise its performance. Over the years, various methods have been proposed to address these issues, but they often fail to fully capture the complexities of underwater environments. This review systematically categorizes UOD challenges into five key areas: Image quality degradation, target-related issues, data-related challenges, computational and processing constraints, and limitations in detection methodologies. To address these challenges, we analyze the progression from traditional image processing and object detection techniques to modern approaches. Additionally, we explore the potential of large vision-language models (LVLMs) in UOD, leveraging their multi-modal capabilities demonstrated in other domains. We also present case studies, including synthetic dataset generation using DALL-E 3 and fine-tuning Florence-2 LVLM for UOD. This review identifies three key insights: (i) Current UOD methods are insufficient to fully address challenges like image degradation and small object detection in dynamic underwater environments. (ii) Synthetic data generation using LVLMs shows potential for augmenting datasets but requires further refinement to ensure realism and applicability. (iii) LVLMs hold significant promise for UOD, but their real-time application remains under-explored, requiring further research on optimization techniques.
>
---
#### [new 047] Beyond Distribution Shifts: Adaptive Hyperspectral Image Classification at Test Time
- **分类: cs.CV**

- **简介: 论文提出HyperTTA框架，解决高光谱图像分类中因现实退化导致的分布偏移问题。构建多退化数据集，设计SSTC分类器，并引入轻量级测试时自适应策略CELA，提升模型鲁棒性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.08436v1](http://arxiv.org/pdf/2509.08436v1)**

> **作者:** Xia Yue; Anfeng Liu; Ning Chen; Chenjia Huang; Hui Liu; Zhou Huang; Leyuan Fang
>
> **摘要:** Hyperspectral image (HSI) classification models are highly sensitive to distribution shifts caused by various real-world degradations such as noise, blur, compression, and atmospheric effects. To address this challenge, we propose HyperTTA, a unified framework designed to enhance model robustness under diverse degradation conditions. Specifically, we first construct a multi-degradation hyperspectral dataset that systematically simulates nine representative types of degradations, providing a comprehensive benchmark for robust classification evaluation. Based on this, we design a spectral-spatial transformer classifier (SSTC) enhanced with a multi-level receptive field mechanism and label smoothing regularization to jointly capture multi-scale spatial context and improve generalization. Furthermore, HyperTTA incorporates a lightweight test-time adaptation (TTA) strategy, the confidence-aware entropy-minimized LayerNorm adapter (CELA), which updates only the affine parameters of LayerNorm layers by minimizing prediction entropy on high-confidence unlabeled target samples. This confidence-aware adaptation prevents unreliable updates from noisy predictions, enabling robust and dynamic adaptation without access to source data or target annotations. Extensive experiments on two benchmark datasets demonstrate that HyperTTA outperforms existing baselines across a wide range of degradation scenarios, validating the effectiveness of both its classification backbone and the proposed TTA scheme. Code will be made available publicly.
>
---
#### [new 048] First-order State Space Model for Lightweight Image Super-resolution
- **分类: cs.CV**

- **简介: 论文提出一种轻量级图像超分辨率方法FSSM，改进Mamba模块以提升性能。通过引入一阶保持条件和离散化形式，无需增加参数即可在五个基准数据集上取得最优结果。属于图像超分辨率任务，解决轻量模型性能不足问题。**

- **链接: [http://arxiv.org/pdf/2509.08458v1](http://arxiv.org/pdf/2509.08458v1)**

> **作者:** Yujie Zhu; Xinyi Zhang; Yekai Lu; Guang Yang; Faming Fang; Guixu Zhang
>
> **备注:** Accept by ICASSP 2025 (Oral)
>
> **摘要:** State space models (SSMs), particularly Mamba, have shown promise in NLP tasks and are increasingly applied to vision tasks. However, most Mamba-based vision models focus on network architecture and scan paths, with little attention to the SSM module. In order to explore the potential of SSMs, we modified the calculation process of SSM without increasing the number of parameters to improve the performance on lightweight super-resolution tasks. In this paper, we introduce the First-order State Space Model (FSSM) to improve the original Mamba module, enhancing performance by incorporating token correlations. We apply a first-order hold condition in SSMs, derive the new discretized form, and analyzed cumulative error. Extensive experimental results demonstrate that FSSM improves the performance of MambaIR on five benchmark datasets without additionally increasing the number of parameters, and surpasses current lightweight SR methods, achieving state-of-the-art results.
>
---
#### [new 049] MCTED: A Machine-Learning-Ready Dataset for Digital Elevation Model Generation From Mars Imagery
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出MCTED数据集，用于火星数字高程模型生成任务。解决大规模DEM处理中的数据缺陷问题，构建包含光学图像、DEM及掩码的样本，并训练U-Net模型验证其性能。数据集开源，支持机器学习应用。**

- **链接: [http://arxiv.org/pdf/2509.08027v1](http://arxiv.org/pdf/2509.08027v1)**

> **作者:** Rafał Osadnik; Pablo Gómez; Eleni Bohacek; Rickbir Bahia
>
> **备注:** 22 pages, 21 figures
>
> **摘要:** This work presents a new dataset for the Martian digital elevation model prediction task, ready for machine learning applications called MCTED. The dataset has been generated using a comprehensive pipeline designed to process high-resolution Mars orthoimage and DEM pairs from Day et al., yielding a dataset consisting of 80,898 data samples. The source images are data gathered by the Mars Reconnaissance Orbiter using the CTX instrument, providing a very diverse and comprehensive coverage of the Martian surface. Given the complexity of the processing pipelines used in large-scale DEMs, there are often artefacts and missing data points in the original data, for which we developed tools to solve or mitigate their impact. We divide the processed samples into training and validation splits, ensuring samples in both splits cover no mutual areas to avoid data leakage. Every sample in the dataset is represented by the optical image patch, DEM patch, and two mask patches, indicating values that were originally missing or were altered by us. This allows future users of the dataset to handle altered elevation regions as they please. We provide statistical insights of the generated dataset, including the spatial distribution of samples, the distributions of elevation values, slopes and more. Finally, we train a small U-Net architecture on the MCTED dataset and compare its performance to a monocular depth estimation foundation model, DepthAnythingV2, on the task of elevation prediction. We find that even a very small architecture trained on this dataset specifically, beats a zero-shot performance of a depth estimation foundation model like DepthAnythingV2. We make the dataset and code used for its generation completely open source in public repositories.
>
---
#### [new 050] LD-ViCE: Latent Diffusion Model for Video Counterfactual Explanations
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出LD-ViCE框架，用于生成视频的反事实解释，解决现有方法在时间一致性、鲁棒性和因果洞察力方面的不足。通过潜空间扩散模型降低计算成本，提升解释的语义准确性和实用性，适用于安全关键领域。**

- **链接: [http://arxiv.org/pdf/2509.08422v1](http://arxiv.org/pdf/2509.08422v1)**

> **作者:** Payal Varshney; Adriano Lucieri; Christoph Balada; Sheraz Ahmed; Andreas Dengel
>
> **备注:** 30 pages
>
> **摘要:** Video-based AI systems are increasingly adopted in safety-critical domains such as autonomous driving and healthcare. However, interpreting their decisions remains challenging due to the inherent spatiotemporal complexity of video data and the opacity of deep learning models. Existing explanation techniques often suffer from limited temporal coherence, insufficient robustness, and a lack of actionable causal insights. Current counterfactual explanation methods typically do not incorporate guidance from the target model, reducing semantic fidelity and practical utility. We introduce Latent Diffusion for Video Counterfactual Explanations (LD-ViCE), a novel framework designed to explain the behavior of video-based AI models. Compared to previous approaches, LD-ViCE reduces the computational costs of generating explanations by operating in latent space using a state-of-the-art diffusion model, while producing realistic and interpretable counterfactuals through an additional refinement step. Our experiments demonstrate the effectiveness of LD-ViCE across three diverse video datasets, including EchoNet-Dynamic (cardiac ultrasound), FERV39k (facial expression), and Something-Something V2 (action recognition). LD-ViCE outperforms a recent state-of-the-art method, achieving an increase in R2 score of up to 68% while reducing inference time by half. Qualitative analysis confirms that LD-ViCE generates semantically meaningful and temporally coherent explanations, offering valuable insights into the target model behavior. LD-ViCE represents a valuable step toward the trustworthy deployment of AI in safety-critical domains.
>
---
#### [new 051] Vision-Language Semantic Aggregation Leveraging Foundation Model for Generalizable Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决多模态融合中的语义鸿沟与特征分散问题。提出EM聚合机制与文本引导像素解码器，提升模型跨域泛化能力，实验表明优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.08570v1](http://arxiv.org/pdf/2509.08570v1)**

> **作者:** Wenjun Yu; Yinchen Zhou; Jia-Xuan Jiang; Shubin Zeng; Yuee Li; Zhong Wang
>
> **备注:** 29 pages and 8 figures
>
> **摘要:** Multimodal models have achieved remarkable success in natural image segmentation, yet they often underperform when applied to the medical domain. Through extensive study, we attribute this performance gap to the challenges of multimodal fusion, primarily the significant semantic gap between abstract textual prompts and fine-grained medical visual features, as well as the resulting feature dispersion. To address these issues, we revisit the problem from the perspective of semantic aggregation. Specifically, we propose an Expectation-Maximization (EM) Aggregation mechanism and a Text-Guided Pixel Decoder. The former mitigates feature dispersion by dynamically clustering features into compact semantic centers to enhance cross-modal correspondence. The latter is designed to bridge the semantic gap by leveraging domain-invariant textual knowledge to effectively guide deep visual representations. The synergy between these two mechanisms significantly improves the model's generalization ability. Extensive experiments on public cardiac and fundus datasets demonstrate that our method consistently outperforms existing SOTA approaches across multiple domain generalization benchmarks.
>
---
#### [new 052] CrowdQuery: Density-Guided Query Module for Enhanced 2D and 3D Detection in Crowded Scenes
- **分类: cs.CV**

- **简介: 该论文提出CrowdQuery模块，利用密度信息增强2D和3D检测器在拥挤场景中的性能。通过将密度图嵌入解码器，提升检测精度，适用于多种检测模型，在多个数据集上表现优异。**

- **链接: [http://arxiv.org/pdf/2509.08738v1](http://arxiv.org/pdf/2509.08738v1)**

> **作者:** Marius Dähling; Sebastian Krebs; J. Marius Zöllner
>
> **备注:** 8 pages, 5 figures, accepted by IROS 2025
>
> **摘要:** This paper introduces a novel method for end-to-end crowd detection that leverages object density information to enhance existing transformer-based detectors. We present CrowdQuery (CQ), whose core component is our CQ module that predicts and subsequently embeds an object density map. The embedded density information is then systematically integrated into the decoder. Existing density map definitions typically depend on head positions or object-based spatial statistics. Our method extends these definitions to include individual bounding box dimensions. By incorporating density information into object queries, our method utilizes density-guided queries to improve detection in crowded scenes. CQ is universally applicable to both 2D and 3D detection without requiring additional data. Consequently, we are the first to design a method that effectively bridges 2D and 3D detection in crowded environments. We demonstrate the integration of CQ into both a general 2D and 3D transformer-based object detector, introducing the architectures CQ2D and CQ3D. CQ is not limited to the specific transformer models we selected. Experiments on the STCrowd dataset for both 2D and 3D domains show significant performance improvements compared to the base models, outperforming most state-of-the-art methods. When integrated into a state-of-the-art crowd detector, CQ can further improve performance on the challenging CrowdHuman dataset, demonstrating its generalizability. The code is released at https://github.com/mdaehl/CrowdQuery.
>
---
#### [new 053] ViewSparsifier: Killing Redundancy in Multi-View Plant Phenotyping
- **分类: cs.CV**

- **简介: 论文提出ViewSparsifier方法，解决多视角植物表型分析中的冗余问题。针对植物年龄预测和叶数估计任务，通过随机选择视角学习不变嵌入，提升模型性能，并探索更多视角组合以优化未来研究方向。**

- **链接: [http://arxiv.org/pdf/2509.08550v1](http://arxiv.org/pdf/2509.08550v1)**

> **作者:** Robin-Nico Kampa; Fabian Deuser; Konrad Habel; Norbert Oswald
>
> **摘要:** Plant phenotyping involves analyzing observable characteristics of plants to better understand their growth, health, and development. In the context of deep learning, this analysis is often approached through single-view classification or regression models. However, these methods often fail to capture all information required for accurate estimation of target phenotypic traits, which can adversely affect plant health assessment and harvest readiness prediction. To address this, the Growth Modelling (GroMo) Grand Challenge at ACM Multimedia 2025 provides a multi-view dataset featuring multiple plants and two tasks: Plant Age Prediction and Leaf Count Estimation. Each plant is photographed from multiple heights and angles, leading to significant overlap and redundancy in the captured information. To learn view-invariant embeddings, we incorporate 24 views, referred to as the selection vector, in a random selection. Our ViewSparsifier approach won both tasks. For further improvement and as a direction for future research, we also experimented with randomized view selection across all five height levels (120 views total), referred to as selection matrices.
>
---
#### [new 054] BcQLM: Efficient Vision-Language Understanding with Distilled Q-Gated Cross-Modal Fusion
- **分类: cs.CV**

- **简介: 该论文提出轻量级多模态大语言模型BcQLM，用于端到端视觉问答任务。旨在解决大模型在资源受限环境下的部署难题，通过优化的BreezeCLIP编码器和Q门控融合机制，在保持性能的同时显著降低计算成本。**

- **链接: [http://arxiv.org/pdf/2509.08715v1](http://arxiv.org/pdf/2509.08715v1)**

> **作者:** Sike Xiang; Shuang Chen; Amir Atapour-Abarghouei
>
> **摘要:** As multimodal large language models (MLLMs) advance, their large-scale architectures pose challenges for deployment in resource-constrained environments. In the age of large models, where energy efficiency, computational scalability and environmental sustainability are paramount, the development of lightweight and high-performance models is critical for real-world applications. As such, we propose a lightweight MLLM framework for end-to-end visual question answering. Our proposed approach centres on BreezeCLIP, a compact yet powerful vision-language encoder optimised for efficient multimodal understanding. With only 1.2 billion parameters overall, our model significantly reduces computational cost while achieving performance comparable to standard-size MLLMs. Experiments conducted on multiple datasets further validate its effectiveness in balancing accuracy and efficiency. The modular and extensible design enables generalisation to broader multimodal tasks. The proposed lightweight vision-language framework is denoted as BcQLM (BreezeCLIP-enhanced Q-Gated Multimodal Language Model). It offers a promising path toward deployable MLLMs under practical hardware constraints. The source code is available at https://github.com/thico0224/BcQLM.
>
---
#### [new 055] Lightweight Deep Unfolding Networks with Enhanced Robustness for Infrared Small Target Detection
- **分类: cs.CV**

- **简介: 该论文针对红外小目标检测任务，提出轻量级鲁棒深度展开网络L-RPCANet，通过层次瓶颈结构和噪声抑制模块提升参数轻量化与抗噪能力，实验验证其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.08205v1](http://arxiv.org/pdf/2509.08205v1)**

> **作者:** Jingjing Liu; Yinchao Han; Xianchao Xiu; Jianhua Zhang; Wanquan Liu
>
> **摘要:** Infrared small target detection (ISTD) is one of the key techniques in image processing. Although deep unfolding networks (DUNs) have demonstrated promising performance in ISTD due to their model interpretability and data adaptability, existing methods still face significant challenges in parameter lightweightness and noise robustness. In this regard, we propose a highly lightweight framework based on robust principal component analysis (RPCA) called L-RPCANet. Technically, a hierarchical bottleneck structure is constructed to reduce and increase the channel dimension in the single-channel input infrared image to achieve channel-wise feature refinement, with bottleneck layers designed in each module to extract features. This reduces the number of channels in feature extraction and improves the lightweightness of network parameters. Furthermore, a noise reduction module is embedded to enhance the robustness against complex noise. In addition, squeeze-and-excitation networks (SENets) are leveraged as a channel attention mechanism to focus on the varying importance of different features across channels, thereby achieving excellent performance while maintaining both lightweightness and robustness. Extensive experiments on the ISTD datasets validate the superiority of our proposed method compared with state-of-the-art methods covering RPCANet, DRPCANet, and RPCANet++. The code will be available at https://github.com/xianchaoxiu/L-RPCANet.
>
---
#### [new 056] Quantifying Accuracy of an Event-Based Star Tracker via Earth's Rotation
- **分类: cs.CV**

- **简介: 该论文评估事件相机星 tracker 的精度，利用地球自转作为基准。通过静态观测星空，对比 IERS 数据，验证其定位误差。研究证明事件相机在低成本、低延迟星跟踪中具有应用潜力。**

- **链接: [http://arxiv.org/pdf/2509.08794v1](http://arxiv.org/pdf/2509.08794v1)**

> **作者:** Dennis Melamed; Connor Hashemi; Scott McCloskey
>
> **摘要:** Event-based cameras (EBCs) are a promising new technology for star tracking-based attitude determination, but prior studies have struggled to determine accurate ground truth for real data. We analyze the accuracy of an EBC star tracking system utilizing the Earth's motion as the ground truth for comparison. The Earth rotates in a regular way with very small irregularities which are measured to the level of milli-arcseconds. By keeping an event camera static and pointing it through a ground-based telescope at the night sky, we create a system where the only camera motion in the celestial reference frame is that induced by the Earth's rotation. The resulting event stream is processed to generate estimates of orientation which we compare to the International Earth Rotation and Reference System (IERS) measured orientation of the Earth. The event camera system is able to achieve a root mean squared across error of 18.47 arcseconds and an about error of 78.84 arcseconds. Combined with the other benefits of event cameras over framing sensors (reduced computation due to sparser data streams, higher dynamic range, lower energy consumption, faster update rates), this level of accuracy suggests the utility of event cameras for low-cost and low-latency star tracking. We provide all code and data used to generate our results: https://gitlab.kitware.com/nest-public/telescope_accuracy_quantification.
>
---
#### [new 057] SAFT: Shape and Appearance of Fabrics from Template via Differentiable Physical Simulations from Monocular Video
- **分类: cs.CV**

- **简介: 该论文属于三维动态场景重建任务，旨在从单目视频中同时恢复织物的几何形状与外观。通过引入物理模拟和可微渲染，并设计新正则项解决深度模糊问题，显著提升了重建精度与质量。**

- **链接: [http://arxiv.org/pdf/2509.08828v1](http://arxiv.org/pdf/2509.08828v1)**

> **作者:** David Stotko; Reinhard Klein
>
> **备注:** Project page: https://cg.cs.uni-bonn.de/publication/stotko-2025-saft Video: https://www.youtube.com/watch?v=EvioNjBOARc GitHub: https://github.com/vc-bonn/saft
>
> **摘要:** The reconstruction of three-dimensional dynamic scenes is a well-established yet challenging task within the domain of computer vision. In this paper, we propose a novel approach that combines the domains of 3D geometry reconstruction and appearance estimation for physically based rendering and present a system that is able to perform both tasks for fabrics, utilizing only a single monocular RGB video sequence as input. In order to obtain realistic and high-quality deformations and renderings, a physical simulation of the cloth geometry and differentiable rendering are employed. In this paper, we introduce two novel regularization terms for the 3D reconstruction task that improve the plausibility of the reconstruction by addressing the depth ambiguity problem in monocular video. In comparison with the most recent methods in the field, we have reduced the error in the 3D reconstruction by a factor of 2.64 while requiring a medium runtime of 30 min per scene. Furthermore, the optimized motion achieves sufficient quality to perform an appearance estimation of the deforming object, recovering sharp details from this single monocular RGB video.
>
---
#### [new 058] An Open Benchmark Dataset for GeoAI Foundation Models for Oil Palm Mapping in Indonesia
- **分类: cs.CV**

- **简介: 该论文构建了一个开放的地理AI基准数据集，用于印尼油棕榈种植监测。旨在解决油棕种植导致的森林砍伐问题，通过高分辨率卫星影像标注，提供多阶段油棕种植及土地覆盖类型数据，支持模型训练与评估，助力可持续发展和监管。**

- **链接: [http://arxiv.org/pdf/2509.08303v1](http://arxiv.org/pdf/2509.08303v1)**

> **作者:** M. Warizmi Wafiq; Peter Cutter; Ate Poortinga; Daniel Marc G. dela Torre; Karis Tenneson; Vanna Teck; Enikoe Bihari; Chanarun Saisaward; Weraphong Suaruang; Andrea McMahon; Andi Vika Faradiba Muin; Karno B. Batiran; Chairil A; Nurul Qomar; Arya Arismaya Metananda; David Ganz; David Saah
>
> **摘要:** Oil palm cultivation remains one of the leading causes of deforestation in Indonesia. To better track and address this challenge, detailed and reliable mapping is needed to support sustainability efforts and emerging regulatory frameworks. We present an open-access geospatial dataset of oil palm plantations and related land cover types in Indonesia, produced through expert labeling of high-resolution satellite imagery from 2020 to 2024. The dataset provides polygon-based, wall-to-wall annotations across a range of agro-ecological zones and includes a hierarchical typology that distinguishes oil palm planting stages as well as similar perennial crops. Quality was ensured through multi-interpreter consensus and field validation. The dataset was created using wall-to-wall digitization over large grids, making it suitable for training and benchmarking both conventional convolutional neural networks and newer geospatial foundation models. Released under a CC-BY license, it fills a key gap in training data for remote sensing and aims to improve the accuracy of land cover types mapping. By supporting transparent monitoring of oil palm expansion, the resource contributes to global deforestation reduction goals and follows FAIR data principles.
>
---
#### [new 059] RewardDance: Reward Scaling in Visual Generation
- **分类: cs.CV**

- **简介: 该论文提出RewardDance框架，解决视觉生成中奖励模型（RM）扩展性差和奖励黑客问题。通过将奖励分数与VLM架构对齐，实现模型和上下文双重扩展，显著提升生成质量并缓解模式崩溃。属于视觉生成任务。**

- **链接: [http://arxiv.org/pdf/2509.08826v1](http://arxiv.org/pdf/2509.08826v1)**

> **作者:** Jie Wu; Yu Gao; Zilyu Ye; Ming Li; Liang Li; Hanzhong Guo; Jie Liu; Zeyue Xue; Xiaoxia Hou; Wei Liu; Yan Zeng; Weilin Huang
>
> **备注:** Bytedance Seed Technical Report
>
> **摘要:** Reward Models (RMs) are critical for improving generation models via Reinforcement Learning (RL), yet the RM scaling paradigm in visual generation remains largely unexplored. It primarily due to fundamental limitations in existing approaches: CLIP-based RMs suffer from architectural and input modality constraints, while prevalent Bradley-Terry losses are fundamentally misaligned with the next-token prediction mechanism of Vision-Language Models (VLMs), hindering effective scaling. More critically, the RLHF optimization process is plagued by Reward Hacking issue, where models exploit flaws in the reward signal without improving true quality. To address these challenges, we introduce RewardDance, a scalable reward modeling framework that overcomes these barriers through a novel generative reward paradigm. By reformulating the reward score as the model's probability of predicting a "yes" token, indicating that the generated image outperforms a reference image according to specific criteria, RewardDance intrinsically aligns reward objectives with VLM architectures. This alignment unlocks scaling across two dimensions: (1) Model Scaling: Systematic scaling of RMs up to 26 billion parameters; (2) Context Scaling: Integration of task-specific instructions, reference examples, and chain-of-thought (CoT) reasoning. Extensive experiments demonstrate that RewardDance significantly surpasses state-of-the-art methods in text-to-image, text-to-video, and image-to-video generation. Crucially, we resolve the persistent challenge of "reward hacking": Our large-scale RMs exhibit and maintain high reward variance during RL fine-tuning, proving their resistance to hacking and ability to produce diverse, high-quality outputs. It greatly relieves the mode collapse problem that plagues smaller models.
>
---
#### [new 060] Expert-Guided Explainable Few-Shot Learning for Medical Image Diagnosis
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出一种专家引导的可解释少样本学习框架，用于医学图像诊断。针对标注数据不足的问题，融合放射科医生提供的ROI，结合Grad-CAM与Dice损失提升模型性能与可解释性，在两个医学图像数据集上取得显著提升。**

- **链接: [http://arxiv.org/pdf/2509.08007v1](http://arxiv.org/pdf/2509.08007v1)**

> **作者:** Ifrat Ikhtear Uddin; Longwei Wang; KC Santosh
>
> **备注:** Accepted for publication in the proceedings of MICCAI Workshop on Data Engineering in Medical Imaging 2025
>
> **摘要:** Medical image analysis often faces significant challenges due to limited expert-annotated data, hindering both model generalization and clinical adoption. We propose an expert-guided explainable few-shot learning framework that integrates radiologist-provided regions-of-interests (ROIs) into model training to simultaneously enhance classification performance and interpretability. Leveraging Grad-CAM for spatial attention supervision, we introduce an explanation loss based on Dice similarity to align model attention with diagnostically relevant regions during training. This explanation loss is jointly optimized with a standard prototypical network objective, encouraging the model to focus on clinically meaningful features even under limited data conditions. We evaluate our framework on two distinct datasets: BraTS (MRI) and VinDr-CXR (Chest X-ray), achieving significant accuracy improvements from 77.09% to 83.61% on BraTS and from 54.33% to 73.29% on VinDr-CXR compared to non-guided models. Grad-CAM visualizations further confirm that expert-guided training consistently aligns attention with diagnostic regions, improving both predictive reliability and clinical trustworthiness. Our findings demonstrate the effectiveness of incorporating expert-guided attention supervision to bridge the gap between performance and interpretability in few-shot medical image diagnosis.
>
---
#### [new 061] PianoVAM: A Multimodal Piano Performance Dataset
- **分类: cs.SD; cs.AI; cs.CV; cs.MM; eess.AS**

- **简介: 该论文提出PianoVAM数据集，用于多模态钢琴演奏分析。任务是解决音乐信息检索中跨模态对齐与标注问题。工作包括采集视频、音频、MIDI及手部关键点数据，并建立标注方法与基准测试。**

- **链接: [http://arxiv.org/pdf/2509.08800v1](http://arxiv.org/pdf/2509.08800v1)**

> **作者:** Yonghyun Kim; Junhyung Park; Joonhyung Bae; Kirak Kim; Taegyun Kwon; Alexander Lerch; Juhan Nam
>
> **备注:** Accepted to the 26th International Society for Music Information Retrieval (ISMIR) Conference, 2025
>
> **摘要:** The multimodal nature of music performance has driven increasing interest in data beyond the audio domain within the music information retrieval (MIR) community. This paper introduces PianoVAM, a comprehensive piano performance dataset that includes videos, audio, MIDI, hand landmarks, fingering labels, and rich metadata. The dataset was recorded using a Disklavier piano, capturing audio and MIDI from amateur pianists during their daily practice sessions, alongside synchronized top-view videos in realistic and varied performance conditions. Hand landmarks and fingering labels were extracted using a pretrained hand pose estimation model and a semi-automated fingering annotation algorithm. We discuss the challenges encountered during data collection and the alignment process across different modalities. Additionally, we describe our fingering annotation method based on hand landmarks extracted from videos. Finally, we present benchmarking results for both audio-only and audio-visual piano transcription using the PianoVAM dataset and discuss additional potential applications.
>
---
#### [new 062] Foundation Models for Autonomous Driving Perception: A Survey Through Core Capabilities
- **分类: cs.RO; cs.CV**

- **简介: 该论文综述基础模型在自动驾驶感知中的应用，分析其在泛化、鲁棒性等方面的优势与挑战，提出四大核心能力框架，指导模型设计，并探讨未来研究方向。属于自动驾驶感知技术研究任务。**

- **链接: [http://arxiv.org/pdf/2509.08302v1](http://arxiv.org/pdf/2509.08302v1)**

> **作者:** Rajendramayavan Sathyam; Yueqi Li
>
> **备注:** 32 pages, 14 figures, accepted at IEEE Open Journal of Vehicular Technology (OJVT)
>
> **摘要:** Foundation models are revolutionizing autonomous driving perception, transitioning the field from narrow, task-specific deep learning models to versatile, general-purpose architectures trained on vast, diverse datasets. This survey examines how these models address critical challenges in autonomous perception, including limitations in generalization, scalability, and robustness to distributional shifts. The survey introduces a novel taxonomy structured around four essential capabilities for robust performance in dynamic driving environments: generalized knowledge, spatial understanding, multi-sensor robustness, and temporal reasoning. For each capability, the survey elucidates its significance and comprehensively reviews cutting-edge approaches. Diverging from traditional method-centric surveys, our unique framework prioritizes conceptual design principles, providing a capability-driven guide for model development and clearer insights into foundational aspects. We conclude by discussing key challenges, particularly those associated with the integration of these capabilities into real-time, scalable systems, and broader deployment challenges related to computational demands and ensuring model reliability against issues like hallucinations and out-of-distribution failures. The survey also outlines crucial future research directions to enable the safe and effective deployment of foundation models in autonomous driving systems.
>
---
#### [new 063] TANGO: Traversability-Aware Navigation with Local Metric Control for Topological Goals
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.SY; eess.SY**

- **简介: 该论文提出TANGO系统，解决机器人视觉导航中无需3D地图和预训练控制器的问题。通过融合拓扑路径规划与局部轨迹控制，实现零样本、长时距导航，提升适应性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.08699v1](http://arxiv.org/pdf/2509.08699v1)**

> **作者:** Stefan Podgorski; Sourav Garg; Mehdi Hosseinzadeh; Lachlan Mares; Feras Dayoub; Ian Reid
>
> **备注:** 9 pages, 5 figures, ICRA 2025
>
> **摘要:** Visual navigation in robotics traditionally relies on globally-consistent 3D maps or learned controllers, which can be computationally expensive and difficult to generalize across diverse environments. In this work, we present a novel RGB-only, object-level topometric navigation pipeline that enables zero-shot, long-horizon robot navigation without requiring 3D maps or pre-trained controllers. Our approach integrates global topological path planning with local metric trajectory control, allowing the robot to navigate towards object-level sub-goals while avoiding obstacles. We address key limitations of previous methods by continuously predicting local trajectory using monocular depth and traversability estimation, and incorporating an auto-switching mechanism that falls back to a baseline controller when necessary. The system operates using foundational models, ensuring open-set applicability without the need for domain-specific fine-tuning. We demonstrate the effectiveness of our method in both simulated environments and real-world tests, highlighting its robustness and deployability. Our approach outperforms existing state-of-the-art methods, offering a more adaptable and effective solution for visual navigation in open-set environments. The source code is made publicly available: https://github.com/podgorki/TANGO.
>
---
#### [new 064] Physics-Guided Rectified Flow for Low-light RAW Image Enhancement
- **分类: eess.IV; cs.CV**

- **简介: 论文提出PGRF框架，用于低光RAW图像增强。针对现有方法忽略乘性噪声和像素级差异的问题，建立物理噪声模型，结合物理引导的rectified flow生成高质量清洁图像，并构建LLID数据集验证效果。**

- **链接: [http://arxiv.org/pdf/2509.08330v1](http://arxiv.org/pdf/2509.08330v1)**

> **作者:** Juntai Zeng
>
> **备注:** 21pages,7figures
>
> **摘要:** Enhancing RAW images captured under low light conditions is a challenging task. Recent deep learning based RAW enhancement methods have shifted from using real paired data to relying on synthetic datasets. These synthetic datasets are typically generated by physically modeling sensor noise, but existing approaches often consider only additive noise, ignore multiplicative components, and rely on global calibration that overlooks pixel level manufacturing variations. As a result, such methods struggle to accurately reproduce real sensor noise. To address these limitations, this paper derives a noise model from the physical noise generation mechanisms that occur under low illumination and proposes a novel composite model that integrates both additive and multiplicative noise. To solve the model, we introduce a physics based per pixel noise simulation and calibration scheme that estimates and synthesizes noise for each individual pixel, thereby overcoming the restrictions of traditional global calibration and capturing spatial noise variations induced by microscopic CMOS manufacturing differences. Motivated by the strong performance of rectified flow methods in image generation and processing, we further combine the physics-based noise synthesis with a rectified flow generative framework and present PGRF a physics-guided rectified flow framework for low light image enhancement. PGRF leverages the ability of rectified flows to model complex data distributions and uses physical guidance to steer the generation toward the desired clean image. To validate the effectiveness of the proposed model, we established the LLID dataset, an indoor low light benchmark captured with the Sony A7S II camera. Experimental results demonstrate that the proposed framework achieves significant improvements in low light RAW image enhancement.
>
---
#### [new 065] Revisiting Deepfake Detection: Chronological Continual Learning and the Limits of Generalization
- **分类: cs.LG; cs.AI; cs.CV; cs.GR**

- **简介: 该论文将深度伪造检测视为持续学习问题，提出一个能适应新技术并保留历史知识的高效框架。通过模拟7年技术演变，引入新指标评估性能与泛化能力，发现现有方法对未来生成器泛化能力有限，提出非通用深度伪造分布假设。**

- **链接: [http://arxiv.org/pdf/2509.07993v1](http://arxiv.org/pdf/2509.07993v1)**

> **作者:** Federico Fontana; Anxhelo Diko; Romeo Lanzino; Marco Raoul Marini; Bachir Kaddar; Gian Luca Foresti; Luigi Cinque
>
> **摘要:** The rapid evolution of deepfake generation technologies poses critical challenges for detection systems, as non-continual learning methods demand frequent and expensive retraining. We reframe deepfake detection (DFD) as a Continual Learning (CL) problem, proposing an efficient framework that incrementally adapts to emerging visual manipulation techniques while retaining knowledge of past generators. Our framework, unlike prior approaches that rely on unreal simulation sequences, simulates the real-world chronological evolution of deepfake technologies in extended periods across 7 years. Simultaneously, our framework builds upon lightweight visual backbones to allow for the real-time performance of DFD systems. Additionally, we contribute two novel metrics: Continual AUC (C-AUC) for historical performance and Forward Transfer AUC (FWT-AUC) for future generalization. Through extensive experimentation (over 600 simulations), we empirically demonstrate that while efficient adaptation (+155 times faster than full retraining) and robust retention of historical knowledge is possible, the generalization of current approaches to future generators without additional training remains near-random (FWT-AUC $\approx$ 0.5) due to the unique imprint characterizing each existing generator. Such observations are the foundation of our newly proposed Non-Universal Deepfake Distribution Hypothesis. \textbf{Code will be released upon acceptance.}
>
---
#### [new 066] CNN-ViT Hybrid for Pneumonia Detection: Theory and Empiric on Limited Data without Pretraining
- **分类: eess.IV; cs.CV**

- **简介: 论文提出一种CNN-ViT混合模型，用于在小样本和类别不平衡数据下检测肺炎。该模型结合CNN与ViT优势，在有限数据下实现高召回率与稳定F1分数，解决了传统模型在数据不足时性能下降的问题。**

- **链接: [http://arxiv.org/pdf/2509.08586v1](http://arxiv.org/pdf/2509.08586v1)**

> **作者:** Prashant Singh Basnet; Roshan Chitrakar
>
> **备注:** 8 pages, 5 Tables, 5 Figures. Manuscript submitted to ICOIICS 2025 Conference. Currently, under peer review
>
> **摘要:** This research explored the hybridization of CNN and ViT within a training dataset of limited size, and introduced a distinct class imbalance. The training was made from scratch with a mere focus on theoretically and experimentally exploring the architectural strengths of the proposed hybrid model. Experiments were conducted across varied data fractions with balanced and imbalanced training datasets. Comparatively, the hybrid model, complementing the strengths of CNN and ViT, achieved the highest recall of 0.9443 (50% data fraction in balanced) and consistency in F1 score around 0.85, suggesting reliability in diagnosis. Additionally, the model was successful in outperforming CNN and ViT in imbalanced datasets. Despite its complex architecture, it required comparable training time to the transformers in all data fractions.
>
---
#### [new 067] SocialNav-SUB: Benchmarking VLMs for Scene Understanding in Social Robot Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SocialNav-SUB基准，用于评估VLM在社交机器人导航场景理解中的能力。任务是通过VQA测试VLM的空间、时空与社交推理能力，发现其与人类和规则基线仍有差距，为未来研究提供框架。**

- **链接: [http://arxiv.org/pdf/2509.08757v1](http://arxiv.org/pdf/2509.08757v1)**

> **作者:** Michael J. Munje; Chen Tang; Shuijing Liu; Zichao Hu; Yifeng Zhu; Jiaxun Cui; Garrett Warnell; Joydeep Biswas; Peter Stone
>
> **备注:** Conference on Robot Learning (CoRL) 2025 Project site: https://larg.github.io/socialnav-sub
>
> **摘要:** Robot navigation in dynamic, human-centered environments requires socially-compliant decisions grounded in robust scene understanding. Recent Vision-Language Models (VLMs) exhibit promising capabilities such as object recognition, common-sense reasoning, and contextual understanding-capabilities that align with the nuanced requirements of social robot navigation. However, it remains unclear whether VLMs can accurately understand complex social navigation scenes (e.g., inferring the spatial-temporal relations among agents and human intentions), which is essential for safe and socially compliant robot navigation. While some recent works have explored the use of VLMs in social robot navigation, no existing work systematically evaluates their ability to meet these necessary conditions. In this paper, we introduce the Social Navigation Scene Understanding Benchmark (SocialNav-SUB), a Visual Question Answering (VQA) dataset and benchmark designed to evaluate VLMs for scene understanding in real-world social robot navigation scenarios. SocialNav-SUB provides a unified framework for evaluating VLMs against human and rule-based baselines across VQA tasks requiring spatial, spatiotemporal, and social reasoning in social robot navigation. Through experiments with state-of-the-art VLMs, we find that while the best-performing VLM achieves an encouraging probability of agreeing with human answers, it still underperforms simpler rule-based approach and human consensus baselines, indicating critical gaps in social scene understanding of current VLMs. Our benchmark sets the stage for further research on foundation models for social robot navigation, offering a framework to explore how VLMs can be tailored to meet real-world social robot navigation needs. An overview of this paper along with the code and data can be found at https://larg.github.io/socialnav-sub .
>
---
#### [new 068] Validation of a CT-brain analysis tool for measuring global cortical atrophy in older patient cohorts
- **分类: eess.IV; cs.AI; cs.CV; I.2; I.4**

- **简介: 该论文验证了一种基于深度学习的CT脑分析工具，用于测量老年患者的大脑皮层萎缩。旨在替代耗时的人工评分方法，通过与人工评分对比，证明其准确性，并展示其在大规模健康数据分析中的应用潜力。**

- **链接: [http://arxiv.org/pdf/2509.08012v1](http://arxiv.org/pdf/2509.08012v1)**

> **作者:** Sukhdeep Bal; Emma Colbourne; Jasmine Gan; Ludovica Griffanti; Taylor Hanayik; Nele Demeyere; Jim Davies; Sarah T Pendlebury; Mark Jenkinson
>
> **备注:** 6 figures
>
> **摘要:** Quantification of brain atrophy currently requires visual rating scales which are time consuming and automated brain image analysis is warranted. We validated our automated deep learning (DL) tool measuring the Global Cerebral Atrophy (GCA) score against trained human raters, and associations with age and cognitive impairment, in representative older (>65 years) patients. CT-brain scans were obtained from patients in acute medicine (ORCHARD-EPR), acute stroke (OCS studies) and a legacy sample. Scans were divided in a 60/20/20 ratio for training, optimisation and testing. CT-images were assessed by two trained raters (rater-1=864 scans, rater-2=20 scans). Agreement between DL tool-predicted GCA scores (range 0-39) and the visual ratings was evaluated using mean absolute error (MAE) and Cohen's weighted kappa. Among 864 scans (ORCHARD-EPR=578, OCS=200, legacy scans=86), MAE between the DL tool and rater-1 GCA scores was 3.2 overall, 3.1 for ORCHARD-EPR, 3.3 for OCS and 2.6 for the legacy scans and half had DL-predicted GCA error between -2 and 2. Inter-rater agreement was Kappa=0.45 between the DL-tool and rater-1, and 0.41 between the tool and rater- 2 whereas it was lower at 0.28 for rater-1 and rater-2. There was no difference in GCA scores from the DL-tool and the two raters (one-way ANOVA, p=0.35) or in mean GCA scores between the DL-tool and rater-1 (paired t-test, t=-0.43, p=0.66), the tool and rater-2 (t=1.35, p=0.18) or between rater-1 and rater-2 (t=0.99, p=0.32). DL-tool GCA scores correlated with age and cognitive scores (both p<0.001). Our DL CT-brain analysis tool measured GCA score accurately and without user input in real-world scans acquired from older patients. Our tool will enable extraction of standardised quantitative measures of atrophy at scale for use in health data research and will act as proof-of-concept towards a point-of-care clinically approved tool.
>
---
#### [new 069] Enhancing Privacy Preservation and Reducing Analysis Time with Federated Transfer Learning in Digital Twins-based Computed Tomography Scan Analysis
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文提出基于数字孪生的联邦迁移学习（FTL）方法，用于CT扫描分析。旨在解决数据隐私、计算资源有限和数据异质性问题，提升模型性能与效率，适用于非独立同分布数据场景。**

- **链接: [http://arxiv.org/pdf/2509.08018v1](http://arxiv.org/pdf/2509.08018v1)**

> **作者:** Avais Jan; Qasim Zia; Murray Patterson
>
> **摘要:** The application of Digital Twin (DT) technology and Federated Learning (FL) has great potential to change the field of biomedical image analysis, particularly for Computed Tomography (CT) scans. This paper presents Federated Transfer Learning (FTL) as a new Digital Twin-based CT scan analysis paradigm. FTL uses pre-trained models and knowledge transfer between peer nodes to solve problems such as data privacy, limited computing resources, and data heterogeneity. The proposed framework allows real-time collaboration between cloud servers and Digital Twin-enabled CT scanners while protecting patient identity. We apply the FTL method to a heterogeneous CT scan dataset and assess model performance using convergence time, model accuracy, precision, recall, F1 score, and confusion matrix. It has been shown to perform better than conventional FL and Clustered Federated Learning (CFL) methods with better precision, accuracy, recall, and F1-score. The technique is beneficial in settings where the data is not independently and identically distributed (non-IID), and it offers reliable, efficient, and secure solutions for medical diagnosis. These findings highlight the possibility of using FTL to improve decision-making in digital twin-based CT scan analysis, secure and efficient medical image analysis, promote privacy, and open new possibilities for applying precision medicine and smart healthcare systems.
>
---
#### [new 070] Quadrotor Navigation using Reinforcement Learning with Privileged Information
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出一种基于强化学习的四旋翼导航方法，利用特权信息解决大障碍物环境下的路径规划问题。通过ToA地图和偏航对齐损失函数，实现高成功率的避障导航，并在真实环境中验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2509.08177v1](http://arxiv.org/pdf/2509.08177v1)**

> **作者:** Jonathan Lee; Abhishek Rathod; Kshitij Goel; John Stecklein; Wennie Tabib
>
> **摘要:** This paper presents a reinforcement learning-based quadrotor navigation method that leverages efficient differentiable simulation, novel loss functions, and privileged information to navigate around large obstacles. Prior learning-based methods perform well in scenes that exhibit narrow obstacles, but struggle when the goal location is blocked by large walls or terrain. In contrast, the proposed method utilizes time-of-arrival (ToA) maps as privileged information and a yaw alignment loss to guide the robot around large obstacles. The policy is evaluated in photo-realistic simulation environments containing large obstacles, sharp corners, and dead-ends. Our approach achieves an 86% success rate and outperforms baseline strategies by 34%. We deploy the policy onboard a custom quadrotor in outdoor cluttered environments both during the day and night. The policy is validated across 20 flights, covering 589 meters without collisions at speeds up to 4 m/s.
>
---
#### [new 071] X-Part: high fidelity and structure coherent shape decomposition
- **分类: cs.GR; cs.CV**

- **简介: 该论文提出X-Part模型，用于高保真、结构连贯的3D形状部件分解。解决现有方法语义分解差、可控性弱的问题，通过边界框提示和点级语义特征实现可控生成，并设计可编辑流程，提升生成质量与实用性。**

- **链接: [http://arxiv.org/pdf/2509.08643v1](http://arxiv.org/pdf/2509.08643v1)**

> **作者:** Xinhao Yan; Jiachen Xu; Yang Li; Changfeng Ma; Yunhan Yang; Chunshi Wang; Zibo Zhao; Zeqiang Lai; Yunfei Zhao; Zhuo Chen; Chunchao Guo
>
> **备注:** Tech Report
>
> **摘要:** Generating 3D shapes at part level is pivotal for downstream applications such as mesh retopology, UV mapping, and 3D printing. However, existing part-based generation methods often lack sufficient controllability and suffer from poor semantically meaningful decomposition. To this end, we introduce X-Part, a controllable generative model designed to decompose a holistic 3D object into semantically meaningful and structurally coherent parts with high geometric fidelity. X-Part exploits the bounding box as prompts for the part generation and injects point-wise semantic features for meaningful decomposition. Furthermore, we design an editable pipeline for interactive part generation. Extensive experimental results show that X-Part achieves state-of-the-art performance in part-level shape generation. This work establishes a new paradigm for creating production-ready, editable, and structurally sound 3D assets. Codes will be released for public research.
>
---
#### [new 072] Good Deep Features to Track: Self-Supervised Feature Extraction and Tracking in Visual Odometry
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉里程计任务，旨在解决大尺度、户外及长期场景下的特征提取与跟踪问题。通过自监督学习与任务反馈优化深度特征，提升模型在光照变化、动态场景等挑战环境中的泛化能力与稳定性。**

- **链接: [http://arxiv.org/pdf/2509.08333v1](http://arxiv.org/pdf/2509.08333v1)**

> **作者:** Sai Puneeth Reddy Gottam; Haoming Zhang; Eivydas Keras
>
> **备注:** This short paper has been accepted as a workshop paper at European Conference on Mobile Robots 2025
>
> **摘要:** Visual-based localization has made significant progress, yet its performance often drops in large-scale, outdoor, and long-term settings due to factors like lighting changes, dynamic scenes, and low-texture areas. These challenges degrade feature extraction and tracking, which are critical for accurate motion estimation. While learning-based methods such as SuperPoint and SuperGlue show improved feature coverage and robustness, they still face generalization issues with out-of-distribution data. We address this by enhancing deep feature extraction and tracking through self-supervised learning with task specific feedback. Our method promotes stable and informative features, improving generalization and reliability in challenging environments.
>
---
#### [new 073] CardioComposer: Flexible and Compositional Anatomical Structure Generation with Disentangled Geometric Guidance
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出CardioComposer框架，解决3D解剖结构生成中可控性与真实感的矛盾。通过可解释的椭球体原语和几何矩损失引导扩散模型，实现对大小、形状、位置的独立控制及多组件约束组合。属于医学图像生成任务。**

- **链接: [http://arxiv.org/pdf/2509.08015v1](http://arxiv.org/pdf/2509.08015v1)**

> **作者:** Karim Kadry; Shoaib Goraya; Ajay Manicka; Abdalla Abdelwahed; Farhad Nezami; Elazer Edelman
>
> **备注:** 10 pages, 13 figures
>
> **摘要:** Generative models of 3D anatomy, when integrated with biophysical simulators, enable the study of structure-function relationships for clinical research and medical device design. However, current models face a trade-off between controllability and anatomical realism. We propose a programmable and compositional framework for guiding unconditional diffusion models of human anatomy using interpretable ellipsoidal primitives embedded in 3D space. Our method involves the selection of certain tissues within multi-tissue segmentation maps, upon which we apply geometric moment losses to guide the reverse diffusion process. This framework supports the independent control over size, shape, and position, as well as the composition of multi-component constraints during inference.
>
---
#### [new 074] Adapting Vision-Language Models for Neutrino Event Classification in High-Energy Physics
- **分类: cs.LG; cs.AI; cs.CV; hep-ex**

- **简介: 论文将视觉语言模型（VLM）应用于高能物理中的中微子事件分类任务，旨在提升分类性能与可解释性。研究对比了VLM与CNN，发现VLM在准确率和灵活性上表现更优，展示了其在物理事件分类中的潜力。**

- **链接: [http://arxiv.org/pdf/2509.08461v1](http://arxiv.org/pdf/2509.08461v1)**

> **作者:** Dikshant Sagar; Kaiwen Yu; Alejandro Yankelevich; Jianming Bian; Pierre Baldi
>
> **摘要:** Recent advances in Large Language Models (LLMs) have demonstrated their remarkable capacity to process and reason over structured and unstructured data modalities beyond natural language. In this work, we explore the applications of Vision Language Models (VLMs), specifically a fine-tuned variant of LLaMa 3.2, to the task of identifying neutrino interactions in pixelated detector data from high-energy physics (HEP) experiments. We benchmark this model against a state-of-the-art convolutional neural network (CNN) architecture, similar to those used in the NOvA and DUNE experiments, which have achieved high efficiency and purity in classifying electron and muon neutrino events. Our evaluation considers both the classification performance and interpretability of the model predictions. We find that VLMs can outperform CNNs, while also providing greater flexibility in integrating auxiliary textual or semantic information and offering more interpretable, reasoning-based predictions. This work highlights the potential of VLMs as a general-purpose backbone for physics event classification, due to their high performance, interpretability, and generalizability, which opens new avenues for integrating multimodal reasoning in experimental neutrino physics.
>
---
#### [new 075] RoentMod: A Synthetic Chest X-Ray Modification Model to Identify and Correct Image Interpretation Model Shortcuts
- **分类: eess.IV; cs.AI; cs.CV; I.4, I.2, J.3**

- **简介: 该论文提出RoentMod，一种生成合成胸部X光图像的模型，用于识别和纠正医学影像AI的捷径学习问题。通过生成具有指定病理特征的逼真图像，提升模型对真实病变的识别能力与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.08640v1](http://arxiv.org/pdf/2509.08640v1)**

> **作者:** Lauren H. Cooke; Matthias Jung; Jan M. Brendel; Nora M. Kerkovits; Borek Foldyna; Michael T. Lu; Vineet K. Raghu
>
> **备注:** 25 + 8 pages, 4 + 7 figures
>
> **摘要:** Chest radiographs (CXRs) are among the most common tests in medicine. Automated image interpretation may reduce radiologists\' workload and expand access to diagnostic expertise. Deep learning multi-task and foundation models have shown strong performance for CXR interpretation but are vulnerable to shortcut learning, where models rely on spurious and off-target correlations rather than clinically relevant features to make decisions. We introduce RoentMod, a counterfactual image editing framework that generates anatomically realistic CXRs with user-specified, synthetic pathology while preserving unrelated anatomical features of the original scan. RoentMod combines an open-source medical image generator (RoentGen) with an image-to-image modification model without requiring retraining. In reader studies with board-certified radiologists and radiology residents, RoentMod-produced images appeared realistic in 93\% of cases, correctly incorporated the specified finding in 89-99\% of cases, and preserved native anatomy comparable to real follow-up CXRs. Using RoentMod, we demonstrate that state-of-the-art multi-task and foundation models frequently exploit off-target pathology as shortcuts, limiting their specificity. Incorporating RoentMod-generated counterfactual images during training mitigated this vulnerability, improving model discrimination across multiple pathologies by 3-19\% AUC in internal validation and by 1-11\% for 5 out of 6 tested pathologies in external testing. These findings establish RoentMod as a broadly applicable tool for probing and correcting shortcut learning in medical AI. By enabling controlled counterfactual interventions, RoentMod enhances the robustness and interpretability of CXR interpretation models and provides a generalizable strategy for improving foundation models in medical imaging.
>
---
#### [new 076] STROKEVISION-BENCH: A Multimodal Video And 2D Pose Benchmark For Tracking Stroke Recovery
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文提出StrokeVision-Bench数据集，用于跟踪中风康复。针对现有数据集缺乏临床结构化评估任务的问题，构建包含1000个标注视频的多模态数据集，推动自动化中风康复评估研究。**

- **链接: [http://arxiv.org/pdf/2509.07994v1](http://arxiv.org/pdf/2509.07994v1)**

> **作者:** David Robinson; Animesh Gupta; Rizwan Quershi; Qiushi Fu; Mubarak Shah
>
> **备注:** 6 pages
>
> **摘要:** Despite advancements in rehabilitation protocols, clinical assessment of upper extremity (UE) function after stroke largely remains subjective, relying heavily on therapist observation and coarse scoring systems. This subjectivity limits the sensitivity of assessments to detect subtle motor improvements, which are critical for personalized rehabilitation planning. Recent progress in computer vision offers promising avenues for enabling objective, quantitative, and scalable assessment of UE motor function. Among standardized tests, the Box and Block Test (BBT) is widely utilized for measuring gross manual dexterity and tracking stroke recovery, providing a structured setting that lends itself well to computational analysis. However, existing datasets targeting stroke rehabilitation primarily focus on daily living activities and often fail to capture clinically structured assessments such as block transfer tasks. Furthermore, many available datasets include a mixture of healthy and stroke-affected individuals, limiting their specificity and clinical utility. To address these critical gaps, we introduce StrokeVision-Bench, the first-ever dedicated dataset of stroke patients performing clinically structured block transfer tasks. StrokeVision-Bench comprises 1,000 annotated videos categorized into four clinically meaningful action classes, with each sample represented in two modalities: raw video frames and 2D skeletal keypoints. We benchmark several state-of-the-art video action recognition and skeleton-based action classification methods to establish performance baselines for this domain and facilitate future research in automated stroke rehabilitation assessment.
>
---
## 更新

#### [replaced 001] Towards properties of adversarial image perturbations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.14111v2](http://arxiv.org/pdf/2503.14111v2)**

> **作者:** Egor Kuznetsov; Kirill Aistov; Maxim Koroteev
>
> **备注:** 13 pages, 40 figures
>
> **摘要:** Using stochastic gradient approach we study the properties of adversarial perturbations resulting in noticeable growth of VMAF image quality metric. The structure of the perturbations is investigated depending on the acceptable PSNR values and based on the Fourier power spectrum computations for the perturbations. It is demonstrated that moderate variation of image brightness ($\sim 10$ pixel units in a restricted region of an image can result in VMAF growth by $\sim 60\%$). Unlike some other methods demonstrating similar VMAF growth, the subjective quality of an image remains almost unchanged. It is also shown that the adversarial perturbations may demonstrate approximately linear dependence of perturbation amplitudes on the image brightness. The perturbations are studied based on the direct VMAF optimization in PyTorch. The significant discrepancies between the metric values and subjective judgements are also demonstrated when image restoration from noise is carried out using the same direct VMAF optimization.
>
---
#### [replaced 002] LLaDA-VLA: Vision Language Diffusion Action Models
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.06932v2](http://arxiv.org/pdf/2509.06932v2)**

> **作者:** Yuqing Wen; Hebei Li; Kefan Gu; Yucheng Zhao; Tiancai Wang; Xiaoyan Sun
>
> **摘要:** The rapid progress of auto-regressive vision-language models (VLMs) has inspired growing interest in vision-language-action models (VLA) for robotic manipulation. Recently, masked diffusion models, a paradigm distinct from autoregressive models, have begun to demonstrate competitive performance in text generation and multimodal applications, leading to the development of a series of diffusion-based VLMs (d-VLMs). However, leveraging such models for robot policy learning remains largely unexplored. In this work, we present LLaDA-VLA, the first Vision-Language-Diffusion-Action model built upon pretrained d-VLMs for robotic manipulation. To effectively adapt d-VLMs to robotic domain, we introduce two key designs: (1) a localized special-token classification strategy that replaces full-vocabulary classification with special action token classification, reducing adaptation difficulty; (2) a hierarchical action-structured decoding strategy that decodes action sequences hierarchically considering the dependencies within and across actions. Extensive experiments demonstrate that LLaDA-VLA significantly outperforms state-of-the-art VLAs on both simulation and real-world robots.
>
---
#### [replaced 003] PromptEnhancer: A Simple Approach to Enhance Text-to-Image Models via Chain-of-Thought Prompt Rewriting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.04545v3](http://arxiv.org/pdf/2509.04545v3)**

> **作者:** Linqing Wang; Ximing Xing; Yiji Cheng; Zhiyuan Zhao; Jiale Tao; Qixun Wang; Ruihuang Li; Comi Chen; Xin Li; Mingrui Wu; Xinchi Deng; Chunyu Wang; Qinglin Lu
>
> **备注:** technical report
>
> **摘要:** Recent advancements in text-to-image (T2I) diffusion models have demonstrated remarkable capabilities in generating high-fidelity images. However, these models often struggle to faithfully render complex user prompts, particularly in aspects like attribute binding, negation, and compositional relationships. This leads to a significant mismatch between user intent and the generated output. To address this challenge, we introduce PromptEnhancer, a novel and universal prompt rewriting framework that enhances any pretrained T2I model without requiring modifications to its weights. Unlike prior methods that rely on model-specific fine-tuning or implicit reward signals like image-reward scores, our framework decouples the rewriter from the generator. We achieve this by training a Chain-of-Thought (CoT) rewriter through reinforcement learning, guided by a dedicated reward model we term the AlignEvaluator. The AlignEvaluator is trained to provide explicit and fine-grained feedback based on a systematic taxonomy of 24 key points, which are derived from a comprehensive analysis of common T2I failure modes. By optimizing the CoT rewriter to maximize the reward from our AlignEvaluator, our framework learns to generate prompts that are more precisely interpreted by T2I models. Extensive experiments on the HunyuanImage 2.1 model demonstrate that PromptEnhancer significantly improves image-text alignment across a wide range of semantic and compositional challenges. Furthermore, we introduce a new, high-quality human preference benchmark to facilitate future research in this direction.
>
---
#### [replaced 004] Rethinking Random Masking in Self-Distillation on ViT
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.10582v3](http://arxiv.org/pdf/2506.10582v3)**

> **作者:** Jihyeon Seong; Hyunkyung Han
>
> **备注:** 4 pages
>
> **摘要:** Vision Transformers (ViTs) have demonstrated remarkable performance across a wide range of vision tasks. In particular, self-distillation frameworks such as DINO have contributed significantly to these advances. Within such frameworks, random masking is often utilized to improve training efficiency and introduce regularization. However, recent studies have raised concerns that indiscriminate random masking may inadvertently eliminate critical semantic information, motivating the development of more informed masking strategies. In this study, we explore the role of random masking in the self-distillation setting, focusing on the DINO framework. Specifically, we apply random masking exclusively to the student's global view, while preserving the student's local views and the teacher's global view in their original, unmasked forms. This design leverages DINO's multi-view augmentation scheme to retain clean supervision while inducing robustness through masked inputs. We evaluate our approach using DINO-Tiny on the mini-ImageNet dataset and show that random masking under this asymmetric setup yields more robust and fine-grained attention maps, ultimately enhancing downstream performance.
>
---
#### [replaced 005] GloFinder: AI-empowered QuPath Plugin for WSI-level Glomerular Detection, Visualization, and Curation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.18795v2](http://arxiv.org/pdf/2411.18795v2)**

> **作者:** Jialin Yue; Tianyuan Yao; Ruining Deng; Siqi Lu; Junlin Guo; Quan Liu; Mengmeng Yin; Juming Xiong; Haichun Yang; Yuankai Huo
>
> **摘要:** Artificial intelligence (AI) has demonstrated significant success in automating the detection of glomeruli, the key functional units of the kidney, from whole slide images (WSIs) in kidney pathology. However, existing open-source tools are often distributed as source code or Docker containers, requiring advanced programming skills that hinder accessibility for non-programmers, such as clinicians. Additionally, current models are typically trained on a single dataset and lack flexibility in adjusting confidence levels for predictions. To overcome these challenges, we introduce GloFinder, a QuPath plugin designed for single-click automated glomeruli detection across entire WSIs with online editing through the graphical user interface (GUI). GloFinder employs CircleNet, an anchor-free detection framework utilizing circle representations for precise object localization, with models trained on approximately 160,000 manually annotated glomeruli. To further enhance accuracy, the plugin incorporates Weighted Circle Fusion (WCF), an ensemble method that combines confidence scores from multiple CircleNet models to produce refined predictions, achieving superior performance in glomerular detection. GloFinder enables direct visualization and editing of results in QuPath, facilitating seamless interaction for clinicians and providing a powerful tool for nephropathology research and clinical practice. Code and the QuPath plugin are available at https://github.com/hrlblab/GloFinder
>
---
#### [replaced 006] RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.01907v3](http://arxiv.org/pdf/2509.01907v3)**

> **作者:** Zhenyuan Chen; Chenxi Wang; Feng Zhang
>
> **备注:** under review
>
> **摘要:** Remote sensing is critical for disaster monitoring, yet existing datasets lack temporal image pairs and detailed textual annotations. While single-snapshot imagery dominates current resources, it fails to capture dynamic disaster impacts over time. To address this gap, we introduce the Remote Sensing Change Caption (RSCC) dataset, a large-scale benchmark comprising 62,315 pre-/post-disaster image pairs (spanning earthquakes, floods, wildfires, and more) paired with rich, human-like change captions. By bridging the temporal and semantic divide in remote sensing data, RSCC enables robust training and evaluation of vision-language models for disaster-aware bi-temporal understanding. Our results highlight RSCC's ability to facilitate detailed disaster-related analysis, paving the way for more accurate, interpretable, and scalable vision-language applications in remote sensing. Code and dataset are available at https://github.com/Bili-Sakura/RSCC.
>
---
#### [replaced 007] Moment- and Power-Spectrum-Based Gaussianity Regularization for Text-to-Image Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.07027v2](http://arxiv.org/pdf/2509.07027v2)**

> **作者:** Jisung Hwang; Jaihoon Kim; Minhyuk Sung
>
> **摘要:** We propose a novel regularization loss that enforces standard Gaussianity, encouraging samples to align with a standard Gaussian distribution. This facilitates a range of downstream tasks involving optimization in the latent space of text-to-image models. We treat elements of a high-dimensional sample as one-dimensional standard Gaussian variables and define a composite loss that combines moment-based regularization in the spatial domain with power spectrum-based regularization in the spectral domain. Since the expected values of moments and power spectrum distributions are analytically known, the loss promotes conformity to these properties. To ensure permutation invariance, the losses are applied to randomly permuted inputs. Notably, existing Gaussianity-based regularizations fall within our unified framework: some correspond to moment losses of specific orders, while the previous covariance-matching loss is equivalent to our spectral loss but incurs higher time complexity due to its spatial-domain computation. We showcase the application of our regularization in generative modeling for test-time reward alignment with a text-to-image model, specifically to enhance aesthetics and text alignment. Our regularization outperforms previous Gaussianity regularization, effectively prevents reward hacking and accelerates convergence.
>
---
#### [replaced 008] TransitReID: Transit OD Data Collection with Occlusion-Resistant Dynamic Passenger Re-Identification
- **分类: cs.CV; cs.AI; eess.IV**

- **链接: [http://arxiv.org/pdf/2504.11500v2](http://arxiv.org/pdf/2504.11500v2)**

> **作者:** Kaicong Huang; Talha Azfar; Jack Reilly; Ruimin Ke
>
> **摘要:** Transit Origin-Destination (OD) data are fundamental for optimizing public transit services, yet current collection methods, such as manual surveys, Bluetooth and WiFi tracking, or Automated Passenger Counters, are either costly, device-dependent, or incapable of individual-level matching. Meanwhile, onboard surveillance cameras already deployed on most transit vehicles provide an underutilized opportunity for automated OD data collection. Leveraging this, we present TransitReID, a novel framework for individual-level and occlusion-resistant passenger re-identification tailored to transit environments. Our approach introduces three key innovations: (1) an occlusion-robust ReID algorithm that integrates a variational autoencoder-guided region-attention mechanism and selective quality feature averaging to dynamically emphasize visible and discriminative body regions under severe occlusions and viewpoint variations; (2) a Hierarchical Storage and Dynamic Matching HSDM mechanism that transforms static gallery matching into a dynamic process for robustness, accuracy, and speed in real-world bus operations; and (3) a multi-threaded edge implementation that enables near real-time OD estimation while ensuring privacy by processing all data locally. To support research in this domain, we also construct a new TransitReID dataset with over 17,000 images captured from bus front and rear cameras under diverse occlusion and viewpoint conditions. Experimental results demonstrate that TransitReID achieves state-of-the-art performance, with R-1 accuracy of 88.3 percent and mAP of 92.5 percent, and further sustains 90 percent OD estimation accuracy in bus route simulations on NVIDIA Jetson edge devices. This work advances both the algorithmic and system-level foundations of automated transit OD collection, paving the way for scalable, privacy-preserving deployment in intelligent transportation systems.
>
---
#### [replaced 009] GNF: Gaussian Neural Fields for Multidimensional Signal Representation and Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06762v2](http://arxiv.org/pdf/2503.06762v2)**

> **作者:** Abdelaziz Bouzidi; Hamid Laga; Hazem Wannous; Ferdous Sohel
>
> **备注:** The source code is publicly available at \url{https://grbfnet.github.io/}
>
> **摘要:** Neural fields have emerged as a powerful framework for representing continuous multidimensional signals such as images and videos, 3D and 4D objects and scenes, and radiance fields. While efficient, achieving high-quality representation requires the use of wide and deep neural networks. These, however, are slow to train and evaluate. Although several acceleration techniques have been proposed, they either trade memory for faster training and/or inference, rely on thousands of fitted primitives with considerable optimization time, or compromise the smooth, continuous nature of neural fields. In this paper, we introduce Gaussian Neural Fields (GNF), a novel compact neural decoder that maps learned feature grids into continuous non-linear signals, such as RGB images, Signed Distance Functions (SDFs), and radiance fields, using a single compact layer of Gaussian kernels defined in a high-dimensional feature space. Our key observation is that neurons in traditional MLPs perform simple computations, usually a dot product followed by an activation function, necessitating wide and deep MLPs or high-resolution feature grids to model complex functions. In this paper, we show that replacing MLP-based decoders with Gaussian kernels whose centers are learned features yields highly accurate representations of 2D (RGB), 3D (geometry), and 5D (radiance fields) signals with just a single layer of such kernels. This representation is highly parallelizable, operates on low-resolution grids, and trains in under $15$ seconds for 3D geometry and under $11$ minutes for view synthesis. GNF matches the accuracy of deep MLP-based decoders with far fewer parameters and significantly higher inference throughput.
>
---
#### [replaced 010] P3-SAM: Native 3D Part Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.06784v3](http://arxiv.org/pdf/2509.06784v3)**

> **作者:** Changfeng Ma; Yang Li; Xinhao Yan; Jiachen Xu; Yunhan Yang; Chunshi Wang; Zibo Zhao; Yanwen Guo; Zhuo Chen; Chunchao Guo
>
> **备注:** Tech Report
>
> **摘要:** Segmenting 3D assets into their constituent parts is crucial for enhancing 3D understanding, facilitating model reuse, and supporting various applications such as part generation. However, current methods face limitations such as poor robustness when dealing with complex objects and cannot fully automate the process. In this paper, we propose a native 3D point-promptable part segmentation model termed P3-SAM, designed to fully automate the segmentation of any 3D objects into components. Inspired by SAM, P3-SAM consists of a feature extractor, multiple segmentation heads, and an IoU predictor, enabling interactive segmentation for users. We also propose an algorithm to automatically select and merge masks predicted by our model for part instance segmentation. Our model is trained on a newly built dataset containing nearly 3.7 million models with reasonable segmentation labels. Comparisons show that our method achieves precise segmentation results and strong robustness on any complex objects, attaining state-of-the-art performance. Our code will be released soon.
>
---
#### [replaced 011] Alternating Minimization Schemes for Computing Rate-Distortion-Perception Functions with $f$-Divergence Perception Constraints
- **分类: cs.IT; cs.CV; eess.SP; math.IT**

- **链接: [http://arxiv.org/pdf/2408.15015v2](http://arxiv.org/pdf/2408.15015v2)**

> **作者:** Giuseppe Serra; Photios A. Stavrou; Marios Kountouris
>
> **备注:** This work has been submitted for possible publication
>
> **摘要:** We study the computation of the rate-distortion-perception function (RDPF) for discrete memoryless sources subject to a single-letter average distortion constraint and a perception constraint belonging to the family of $f$-divergences. In this setting, the RDPF forms a convex programming problem for which we characterize optimal parametric solutions. We employ the developed solutions in an alternating minimization scheme, namely Optimal Alternating Minimization (OAM), for which we provide convergence guarantees. Nevertheless, the OAM scheme does not lead to a direct implementation of a generalized Blahut-Arimoto (BA) type of algorithm due to implicit equations in the iteration's structure. To overcome this difficulty, we propose two alternative minimization approaches whose applicability depends on the smoothness of the used perception metric: a Newton-based Alternating Minimization (NAM) scheme, relying on Newton's root-finding method for the approximation of the optimal solution of the iteration, and a Relaxed Alternating Minimization (RAM) scheme, based on relaxing the OAM iterates. We show, by deriving necessary and sufficient conditions, that both schemes guarantee convergence to a globally optimal solution. We also provide sufficient conditions on the distortion and perception constraints, which guarantee that the proposed algorithms converge exponentially fast in the number of iteration steps. We corroborate our theoretical results with numerical simulations and establish connections with existing results.
>
---
#### [replaced 012] Detection of trade in products derived from threatened species using machine learning and a smartphone
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.06585v2](http://arxiv.org/pdf/2509.06585v2)**

> **作者:** Ritwik Kulkarni; WU Hanqin; Enrico Di Minin
>
> **摘要:** Unsustainable trade in wildlife is a major threat to biodiversity and is now increasingly prevalent in digital marketplaces and social media. With the sheer volume of digital content, the need for automated methods to detect wildlife trade listings is growing. These methods are especially needed for the automatic identification of wildlife products, such as ivory. We developed machine learning-based object recognition models that can identify wildlife products within images and highlight them. The data consists of images of elephant, pangolin, and tiger products that were identified as being sold illegally or that were confiscated by authorities. Specifically, the wildlife products included elephant ivory and skins, pangolin scales, and claws (raw and crafted), and tiger skins and bones. We investigated various combinations of training strategies and two loss functions to identify the best model to use in the automatic detection of these wildlife products. Models were trained for each species while also developing a single model to identify products from all three species. The best model showed an overall accuracy of 84.2% with accuracies of 71.1%, 90.2% and 93.5% in detecting products derived from elephants, pangolins, and tigers, respectively. We further demonstrate that the machine learning model can be made easily available to stakeholders, such as government authorities and law enforcement agencies, by developing a smartphone-based application that had an overall accuracy of 91.3%. The application can be used in real time to click images and help identify potentially prohibited products of target species. Thus, the proposed method is not only applicable for monitoring trade on the web but can also be used e.g. in physical markets for monitoring wildlife trade.
>
---
#### [replaced 013] CamC2V: Context-aware Controllable Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.06022v2](http://arxiv.org/pdf/2504.06022v2)**

> **作者:** Luis Denninger; Sina Mokhtarzadeh Azar; Juergen Gall
>
> **摘要:** Recently, image-to-video (I2V) diffusion models have demonstrated impressive scene understanding and generative quality, incorporating image conditions to guide generation. However, these models primarily animate static images without extending beyond their provided context. Introducing additional constraints, such as camera trajectories, can enhance diversity but often degrade visual quality, limiting their applicability for tasks requiring faithful scene representation. We propose CamC2V, a context-to-video (C2V) model that integrates multiple image conditions as context with 3D constraints alongside camera control to enrich both global semantics and fine-grained visual details. This enables more coherent and context-aware video generation. Moreover, we motivate the necessity of temporal awareness for an effective context representation. Our comprehensive study on the RealEstate10K dataset demonstrates improvements in visual quality and camera controllability. We will publish our code upon acceptance.
>
---
#### [replaced 014] SGDFuse: SAM-Guided Diffusion for High-Fidelity Infrared and Visible Image Fusion
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.05264v3](http://arxiv.org/pdf/2508.05264v3)**

> **作者:** Xiaoyang Zhang; jinjiang Li; Guodong Fan; Yakun Ju; Linwei Fan; Jun Liu; Alex C. Kot
>
> **备注:** Submitted to Information Fusion
>
> **摘要:** Infrared and visible image fusion (IVIF) aims to combine the thermal radiation information from infrared images with the rich texture details from visible images to enhance perceptual capabilities for downstream visual tasks. However, existing methods often fail to preserve key targets due to a lack of deep semantic understanding of the scene, while the fusion process itself can also introduce artifacts and detail loss, severely compromising both image quality and task performance. To address these issues, this paper proposes SGDFuse, a conditional diffusion model guided by the Segment Anything Model (SAM), to achieve high-fidelity and semantically-aware image fusion. The core of our method is to utilize high-quality semantic masks generated by SAM as explicit priors to guide the optimization of the fusion process via a conditional diffusion model. Specifically, the framework operates in a two-stage process: it first performs a preliminary fusion of multi-modal features, and then utilizes the semantic masks from SAM jointly with the preliminary fused image as a condition to drive the diffusion model's coarse-to-fine denoising generation. This ensures the fusion process not only has explicit semantic directionality but also guarantees the high fidelity of the final result. Extensive experiments demonstrate that SGDFuse achieves state-of-the-art performance in both subjective and objective evaluations, as well as in its adaptability to downstream tasks, providing a powerful solution to the core challenges in image fusion. The code of SGDFuse is available at https://github.com/boshizhang123/SGDFuse.
>
---
#### [replaced 015] Have Large Vision-Language Models Mastered Art History?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.03521v2](http://arxiv.org/pdf/2409.03521v2)**

> **作者:** Ombretta Strafforello; Derya Soydaner; Michiel Willems; Anne-Sofie Maerten; Stefanie De Winter
>
> **摘要:** The emergence of large Vision-Language Models (VLMs) has established new baselines in image classification across multiple domains. We examine whether their multimodal reasoning can also address a challenge mastered by human experts. Specifically, we test whether VLMs can classify the style, author and creation date of paintings, a domain traditionally mastered by art historians. Artworks pose a unique challenge compared to natural images due to their inherently complex and diverse structures, characterized by variable compositions and styles. This requires a contextual and stylistic interpretation rather than straightforward object recognition. Art historians have long studied the unique aspects of artworks, with style prediction being a crucial component of their discipline. This paper investigates whether large VLMs, which integrate visual and textual data, can effectively reason about the historical and stylistic attributes of paintings. We present the first study of its kind, conducting an in-depth analysis of three VLMs, namely CLIP, LLaVA, and GPT-4o, evaluating their zero-shot classification of art style, author and time period. Using two image benchmarks of artworks, we assess the models' ability to interpret style, evaluate their sensitivity to prompts, and examine failure cases. Additionally, we focus on how these models compare to human art historical expertise by analyzing misclassifications, providing insights into their reasoning and classification patterns.
>
---
#### [replaced 016] A Chinese Continuous Sign Language Dataset Based on Complex Environments
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.11960v2](http://arxiv.org/pdf/2409.11960v2)**

> **作者:** Qidan Zhu; Jing Li; Fei Yuan; Jiaojiao Fan; Quan Gan
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** The current bottleneck in continuous sign language recognition (CSLR) research lies in the fact that most publicly available datasets are limited to laboratory environments or television program recordings, resulting in a single background environment with uniform lighting, which significantly deviates from the diversity and complexity found in real-life scenarios. To address this challenge, we have constructed a new, large-scale dataset for Chinese continuous sign language (CSL) based on complex environments, termed the complex environment - chinese sign language dataset (CE-CSL). This dataset encompasses 5,988 continuous CSL video clips collected from daily life scenes, featuring more than 70 different complex backgrounds to ensure representativeness and generalization capability. To tackle the impact of complex backgrounds on CSLR performance, we propose a time-frequency network (TFNet) model for continuous sign language recognition. This model extracts frame-level features and then utilizes both temporal and spectral information to separately derive sequence features before fusion, aiming to achieve efficient and accurate CSLR. Experimental results demonstrate that our approach achieves significant performance improvements on the CE-CSL, validating its effectiveness under complex background conditions. Additionally, our proposed method has also yielded highly competitive results when applied to three publicly available CSL datasets.
>
---
#### [replaced 017] LED: LLM Enhanced Open-Vocabulary Object Detection without Human Curated Data Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.13794v5](http://arxiv.org/pdf/2503.13794v5)**

> **作者:** Yang Zhou; Shiyu Zhao; Yuxiao Chen; Zhenting Wang; Can Jin; Dimitris N. Metaxas
>
> **摘要:** Large foundation models trained on large-scale vision-language data can boost Open-Vocabulary Object Detection (OVD) via synthetic training data, yet the hand-crafted pipelines often introduce bias and overfit to specific prompts. We sidestep this issue by directly fusing hidden states from Large Language Models (LLMs) into detectors-an avenue surprisingly under-explored. This paper presents a systematic method to enhance visual grounding by utilizing decoder layers of the LLM of an MLLM. We introduce a zero-initialized cross-attention adapter to enable efficient knowledge fusion from LLMs to object detectors, a new approach called LED (LLM Enhanced Open-Vocabulary Object Detection). We find that intermediate LLM layers already encode rich spatial semantics; adapting only the early layers yields most of the gain. With Swin-T as the vision encoder, Qwen2-0.5B + LED lifts GroundingDINO by 3.82 % on OmniLabel at just 8.7 % extra GFLOPs, and a larger vision backbone pushes the improvement to 6.22 %. Extensive ablations on adapter variants, LLM scales and fusion depths further corroborate our design.
>
---
#### [replaced 018] PrediTree: A Multi-Temporal Sub-meter Dataset of Multi-Spectral Imagery Aligned With Canopy Height Maps
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.01202v2](http://arxiv.org/pdf/2509.01202v2)**

> **作者:** Hiyam Debary; Mustansar Fiaz; Levente Klein
>
> **备注:** Accepted at GAIA 2025. Dataset available at https://huggingface.co/datasets/hiyam-d/PrediTree
>
> **摘要:** We present PrediTree, the first comprehensive open-source dataset designed for training and evaluating tree height prediction models at sub-meter resolution. This dataset combines very high-resolution (0.5m) LiDAR-derived canopy height maps, spatially aligned with multi-temporal and multi-spectral imagery, across diverse forest ecosystems in France, totaling 3,141,568 images. PrediTree addresses a critical gap in forest monitoring capabilities by enabling the training of deep learning methods that can predict tree growth based on multiple past observations. To make use of this PrediTree dataset, we propose an encoder-decoder framework that requires the multi-temporal multi-spectral imagery and the relative time differences in years between the canopy height map timestamp (target) and each image acquisition date for which this framework predicts the canopy height. The conducted experiments demonstrate that a U-Net architecture trained on the PrediTree dataset provides the highest masked mean squared error of $11.78\%$, outperforming the next-best architecture, ResNet-50, by around $12\%$, and cutting the error of the same experiments but on fewer bands (red, green, blue only), by around $30\%$. This dataset is publicly available on https://huggingface.co/datasets/hiyam-d/PrediTree, and both processing and training codebases are available on {GitHub}.
>
---
#### [replaced 019] Learning Robust Representations via Bidirectional Transition for Visual Reinforcement Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2312.01915v2](http://arxiv.org/pdf/2312.01915v2)**

> **作者:** Xiaobo Hu; Youfang Lin; Yue Liu; Jinwen Wang; Shuo Wang; Hehe Fan; Kai Lv
>
> **摘要:** Visual reinforcement learning has proven effective in solving control tasks with high-dimensional observations. However, extracting reliable and generalizable representations from vision-based observations remains a central challenge. Inspired by the human thought process, when the representation extracted from the observation can predict the future and trace history, the representation is reliable and accurate in comprehending the environment. Based on this concept, we introduce a Bidirectional Transition (BiT) model, which leverages the ability to bidirectionally predict environmental transitions both forward and backward to extract reliable representations. Our model demonstrates competitive generalization performance and sample efficiency on two settings of the DeepMind Control suite. Additionally, we utilize robotic manipulation and CARLA simulators to demonstrate the wide applicability of our method.
>
---
#### [replaced 020] F-Bench: Rethinking Human Preference Evaluation Metrics for Benchmarking Face Generation, Customization, and Restoration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.13155v2](http://arxiv.org/pdf/2412.13155v2)**

> **作者:** Lu Liu; Huiyu Duan; Qiang Hu; Liu Yang; Chunlei Cai; Tianxiao Ye; Huayu Liu; Xiaoyun Zhang; Guangtao Zhai
>
> **摘要:** Artificial intelligence generative models exhibit remarkable capabilities in content creation, particularly in face image generation, customization, and restoration. However, current AI-generated faces (AIGFs) often fall short of human preferences due to unique distortions, unrealistic details, and unexpected identity shifts, underscoring the need for a comprehensive quality evaluation framework for AIGFs. To address this need, we introduce FaceQ, a large-scale, comprehensive database of AI-generated Face images with fine-grained Quality annotations reflecting human preferences. The FaceQ database comprises 12,255 images generated by 29 models across three tasks: (1) face generation, (2) face customization, and (3) face restoration. It includes 32,742 mean opinion scores (MOSs) from 180 annotators, assessed across multiple dimensions: quality, authenticity, identity (ID) fidelity, and text-image correspondence. Using the FaceQ database, we establish F-Bench, a benchmark for comparing and evaluating face generation, customization, and restoration models, highlighting strengths and weaknesses across various prompts and evaluation dimensions. Additionally, we assess the performance of existing image quality assessment (IQA), face quality assessment (FQA), AI-generated content image quality assessment (AIGCIQA), and preference evaluation metrics, manifesting that these standard metrics are relatively ineffective in evaluating authenticity, ID fidelity, and text-image correspondence. The FaceQ database will be publicly available upon publication.
>
---
#### [replaced 021] Physics-Driven Local-Whole Elastic Deformation Modeling for Point Cloud Representation Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.13812v2](http://arxiv.org/pdf/2505.13812v2)**

> **作者:** Zhongyu Chen; Rong Zhao; Xie Han; Xindong Guo; Song Wang; Zherui Qiao
>
> **摘要:** Existing point cloud representation learning methods primarily rely on data-driven strategies to extract geometric information from large amounts of scattered data. However, most methods focus solely on the spatial distribution features of point clouds while overlooking the relationship between local information and the whole structure, which limits the accuracy of point cloud representation. Local information reflect the fine-grained variations of an object, while the whole structure is determined by the interaction and combination of these local features, collectively defining the object's shape. In real-world, objects undergo deformation under external forces, and this deformation gradually affects the whole structure through the propagation of forces from local regions, thereby altering the object's geometric features. Therefore, the appropriate introduction of physics-driven mechanism can effectively compensate for the limitations of data-driven methods in structural modeling and significantly enhance the generalization and interpretability of point cloud representations in downstream tasks such as understanding and recognition. Inspired by this, we incorporate a physics-driven mechanism into the data-driven method to learn fine-grained features in point clouds and model the structural relationship between local regions and the whole shape. Specifically, we design a dual-task encoder-decoder framework that combines the geometric modeling capability of data-driven implicit fields with physics-driven elastic deformation. Through the integration of physics-based loss functions, the framework is guided to predict localized deformation and explicitly capture the correspondence between local structural changes and whole shape variations. Experimental results show that our method outperforms existing approaches in object classification and segmentation, demonstrating its effectiveness.
>
---
#### [replaced 022] Bidirectional Sparse Attention for Faster Video Diffusion Training
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.01085v2](http://arxiv.org/pdf/2509.01085v2)**

> **作者:** Chenlu Zhan; Wen Li; Chuyu Shen; Jun Zhang; Suhui Wu; Hao Zhang
>
> **摘要:** Video diffusion Transformer (DiT) models excel in generative quality but hit major computational bottlenecks when producing high-resolution, long-duration videos. The quadratic complexity of full attention leads to prohibitively high training and inference costs. Full attention inefficiency stems from two key challenges: excessive computation due to the inherent sparsity of Queries and Key-Value pairs, and redundant computation as fixed sparse patterns fail to leverage DiT's dynamic attention. To overcome this limitation, we propose a Bidirectional Sparse Attention (BSA) framework for faster video DiT training, the first to dynamically sparsify both Queries and Key-Value pairs within 3D full attention, thereby substantially improving training and inference efficiency. BSA addresses these issues through two key components. Query sparsity is optimized by selecting the most informative query tokens via semantic similarity and with a dynamic spatial-time training strategy, while KV sparsity is achieved by computing a statistical dynamic threshold to retain only the most salient KV blocks for computation. Extensive experiments demonstrate that BSA significantly accelerates DiT training across long sequences, reducing FLOPs by up to 20x and achieving 17.79x faster attention training, while preserving or even surpassing the generative quality of full attention.
>
---
#### [replaced 023] PriorCLIP: Visual Prior Guided Vision-Language Model for Remote Sensing Image-Text Retrieval
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.10160v3](http://arxiv.org/pdf/2405.10160v3)**

> **作者:** Jiancheng Pan; Muyuan Ma; Qing Ma; Cong Bai; Shengyong Chen
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** Remote sensing image-text retrieval plays a crucial role in remote sensing interpretation, yet remains challenging under both closed-domain and open-domain scenarios due to semantic noise and domain shifts. To address these issues, we propose a visual prior-guided vision-language model, PriorCLIP, which leverages visual priors for unbiased representation learning and adaptive vision-language alignment. In the closed-domain setting, PriorCLIP introduces two Progressive Attention Encoder (PAE) structures: Spatial-PAE constructs a belief matrix with instruction embeddings to filter key features and mitigate semantic bias. At the same time, Temporal-PAE exploits cyclic activation across time steps to enhance text representation. For the open-domain setting, we design a two-stage prior representation learning strategy, consisting of large-scale pre-training on coarse-grained image-text pairs, followed by fine-tuning on fine-grained pairs using vision-instruction, which enables robust retrieval across long-tail concepts and vocabulary shifts. Furthermore, a cluster-based symmetric contrastive Attribution Loss is proposed to constrain inter-class relations and alleviate semantic confusion in the shared embedding space. Extensive experiments on RSICD and RSITMD benchmarks demonstrate that PriorCLIP achieves substantial improvements, outperforming existing methods by 4.9% and 4.0% in closed-domain retrieval, and by 7.3% and 9.4% in open-domain retrieval, respectively.
>
---
#### [replaced 024] A Survey of World Models for Autonomous Driving
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.11260v4](http://arxiv.org/pdf/2501.11260v4)**

> **作者:** Tuo Feng; Wenguan Wang; Yi Yang
>
> **备注:** Ongoing project. Paper list: https://github.com/FengZicai/AwesomeWMAD Benchmark: https://github.com/FengZicai/WMAD-Benchmarks
>
> **摘要:** Recent breakthroughs in autonomous driving have been propelled by advances in robust world modeling, fundamentally transforming how vehicles interpret dynamic scenes and execute safe decision-making. World models have emerged as a linchpin technology, offering high-fidelity representations of the driving environment that integrate multi-sensor data, semantic cues, and temporal dynamics. This paper systematically reviews recent advances in world models for autonomous driving, proposing a three-tiered taxonomy: (i) Generation of Future Physical World, covering Image-, BEV-, OG-, and PC-based generation methods that enhance scene evolution modeling through diffusion models and 4D occupancy forecasting; (ii) Behavior Planning for Intelligent Agents, combining rule-driven and learning-based paradigms with cost map optimization and reinforcement learning for trajectory generation in complex traffic conditions; (ii) Interaction between Prediction and Planning, achieving multi-agent collaborative decision-making through latent space diffusion and memory-augmented architectures. The study further analyzes training paradigms, including self-supervised learning, multimodal pretraining, and generative data augmentation, while evaluating world models' performance in scene understanding and motion prediction tasks. Future research must address key challenges in self-supervised representation learning, multimodal fusion, and advanced simulation to advance the practical deployment of world models in complex urban environments. Overall, the comprehensive analysis provides a technical roadmap for harnessing the transformative potential of world models in advancing safe and reliable autonomous driving solutions.
>
---
#### [replaced 025] From Channel Bias to Feature Redundancy: Uncovering the "Less is More" Principle in Few-Shot Learning
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2310.03843v2](http://arxiv.org/pdf/2310.03843v2)**

> **作者:** Ji Zhang; Xu Luo; Lianli Gao; Difan Zou; Hengtao Shen; Jingkuan Song
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2206.08126
>
> **摘要:** Deep neural networks often fail to adapt representations to novel tasks under distribution shifts, especially when only a few examples are available. This paper identifies a core obstacle behind this failure: channel bias, where networks develop a rigid emphasis on feature dimensions that were discriminative for the source task, but this emphasis is misaligned and fails to adapt to the distinct needs of a novel task. This bias leads to a striking and detrimental consequence: feature redundancy. We demonstrate that for few-shot tasks, classification accuracy is significantly improved by using as few as 1-5% of the most discriminative feature dimensions, revealing that the vast majority are actively harmful. Our theoretical analysis confirms that this redundancy originates from confounding feature dimensions-those with high intra-class variance but low inter-class separability-which are especially problematic in low-data regimes. This "less is more" phenomenon is a defining characteristic of the few-shot setting, diminishing as more samples become available. To address this, we propose a simple yet effective soft-masking method, Augmented Feature Importance Adjustment (AFIA), which estimates feature importance from augmented data to mitigate the issue. By establishing the cohesive link from channel bias to its consequence of extreme feature redundancy, this work provides a foundational principle for few-shot representation transfer and a practical method for developing more robust few-shot learning algorithms.
>
---
#### [replaced 026] GenFlow: Interactive Modular System for Image Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21369v2](http://arxiv.org/pdf/2506.21369v2)**

> **作者:** Duc-Hung Nguyen; Huu-Phuc Huynh; Minh-Triet Tran; Trung-Nghia Le
>
> **备注:** CBMI 2025
>
> **摘要:** Generative art unlocks boundless creative possibilities, yet its full potential remains untapped due to the technical expertise required for advanced architectural concepts and computational workflows. To bridge this gap, we present GenFlow, a novel modular framework that empowers users of all skill levels to generate images with precision and ease. Featuring a node-based editor for seamless customization and an intelligent assistant powered by natural language processing, GenFlow transforms the complexity of workflow creation into an intuitive and accessible experience. By automating deployment processes and minimizing technical barriers, our framework makes cutting-edge generative art tools available to everyone. A user study demonstrated GenFlow's ability to optimize workflows, reduce task completion times, and enhance user understanding through its intuitive interface and adaptive features. These results position GenFlow as a groundbreaking solution that redefines accessibility and efficiency in the realm of generative art.
>
---
#### [replaced 027] GCAV: A Global Concept Activation Vector Framework for Cross-Layer Consistency in Interpretability
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.21197v2](http://arxiv.org/pdf/2508.21197v2)**

> **作者:** Zhenghao He; Sanchit Sinha; Guangzhi Xiong; Aidong Zhang
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Concept Activation Vectors (CAVs) provide a powerful approach for interpreting deep neural networks by quantifying their sensitivity to human-defined concepts. However, when computed independently at different layers, CAVs often exhibit inconsistencies, making cross-layer comparisons unreliable. To address this issue, we propose the Global Concept Activation Vector (GCAV), a novel framework that unifies CAVs into a single, semantically consistent representation. Our method leverages contrastive learning to align concept representations across layers and employs an attention-based fusion mechanism to construct a globally integrated CAV. By doing so, our method significantly reduces the variance in TCAV scores while preserving concept relevance, ensuring more stable and reliable concept attributions. To evaluate the effectiveness of GCAV, we introduce Testing with Global Concept Activation Vectors (TGCAV) as a method to apply TCAV to GCAV-based representations. We conduct extensive experiments on multiple deep neural networks, demonstrating that our method effectively mitigates concept inconsistency across layers, enhances concept localization, and improves robustness against adversarial perturbations. By integrating cross-layer information into a coherent framework, our method offers a more comprehensive and interpretable understanding of how deep learning models encode human-defined concepts. Code and models are available at https://github.com/Zhenghao-He/GCAV.
>
---
#### [replaced 028] Déjà Vu: Efficient Video-Language Query Engine with Learning-based Inter-Frame Computation Reuse
- **分类: cs.DC; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.14107v2](http://arxiv.org/pdf/2506.14107v2)**

> **作者:** Jinwoo Hwang; Daeun Kim; Sangyeop Lee; Yoonsung Kim; Guseul Heo; Hojoon Kim; Yunseok Jeong; Tadiwos Meaza; Eunhyeok Park; Jeongseob Ahn; Jongse Park
>
> **备注:** Accepted to 2025 VLDB
>
> **摘要:** Recently, Video-Language Models (VideoLMs) have demonstrated remarkable capabilities, offering significant potential for flexible and powerful video query systems. These models typically rely on Vision Transformers (ViTs), which process video frames individually to extract visual embeddings. However, generating embeddings for large-scale videos requires ViT inferencing across numerous frames, posing a major hurdle to real-world deployment and necessitating solutions for integration into scalable video data management systems. This paper introduces D\'ej\`a Vu, a video-language query engine that accelerates ViT-based VideoLMs by reusing computations across consecutive frames. At its core is ReuseViT, a modified ViT model specifically designed for VideoLM tasks, which learns to detect inter-frame reuse opportunities, striking an effective balance between accuracy and reuse. Although ReuseViT significantly reduces computation, these savings do not directly translate into performance gains on GPUs. To overcome this, D\'ej\`a Vu integrates memory-compute joint compaction techniques that convert the FLOP savings into tangible performance gains. Evaluations on three VideoLM tasks show that D\'ej\`a Vu accelerates embedding generation by up to a 2.64x within a 2% error bound, dramatically enhancing the practicality of VideoLMs for large-scale video analytics.
>
---
#### [replaced 029] UAR-NVC: A Unified AutoRegressive Framework for Memory-Efficient Neural Video Compression
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.02733v2](http://arxiv.org/pdf/2503.02733v2)**

> **作者:** Jia Wang; Xinfeng Zhang; Gai Zhang; Jun Zhu; Lv Tang; Li Zhang
>
> **备注:** Accepted to TCSVT2025
>
> **摘要:** Implicit Neural Representations (INRs) have demonstrated significant potential in video compression by representing videos as neural networks. However, as the number of frames increases, the memory consumption for training and inference increases substantially, posing challenges in resource-constrained scenarios. Inspired by the success of traditional video compression frameworks, which process video frame by frame and can efficiently compress long videos, we adopt this modeling strategy for INRs to decrease memory consumption, while aiming to unify the frameworks from the perspective of timeline-based autoregressive modeling. In this work, we present a novel understanding of INR models from an autoregressive (AR) perspective and introduce a Unified AutoRegressive Framework for memory-efficient Neural Video Compression (UAR-NVC). UAR-NVC integrates timeline-based and INR-based neural video compression under a unified autoregressive paradigm. It partitions videos into several clips and processes each clip using a different INR model instance, leveraging the advantages of both compression frameworks while allowing seamless adaptation to either in form. To further reduce temporal redundancy between clips, we design two modules to optimize the initialization, training, and compression of these model parameters. UAR-NVC supports adjustable latencies by varying the clip length. Extensive experimental results demonstrate that UAR-NVC, with its flexible video clip setting, can adapt to resource-constrained environments and significantly improve performance compared to different baseline models. The project page: "https://wj-inf.github.io/UAR-NVC-page/".
>
---
#### [replaced 030] Sigma: Siamese Mamba Network for Multi-Modal Semantic Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2404.04256v3](http://arxiv.org/pdf/2404.04256v3)**

> **作者:** Zifu Wan; Pingping Zhang; Yuhao Wang; Silong Yong; Simon Stepputtis; Katia Sycara; Yaqi Xie
>
> **备注:** Accepted by WACV 2025. Project page: https://zifuwan.github.io/Sigma/
>
> **摘要:** Multi-modal semantic segmentation significantly enhances AI agents' perception and scene understanding, especially under adverse conditions like low-light or overexposed environments. Leveraging additional modalities (X-modality) like thermal and depth alongside traditional RGB provides complementary information, enabling more robust and reliable prediction. In this work, we introduce Sigma, a Siamese Mamba network for multi-modal semantic segmentation utilizing the advanced Mamba. Unlike conventional methods that rely on CNNs, with their limited local receptive fields, or Vision Transformers (ViTs), which offer global receptive fields at the cost of quadratic complexity, our model achieves global receptive fields with linear complexity. By employing a Siamese encoder and innovating a Mamba-based fusion mechanism, we effectively select essential information from different modalities. A decoder is then developed to enhance the channel-wise modeling ability of the model. Our proposed method is rigorously evaluated on both RGB-Thermal and RGB-Depth semantic segmentation tasks, demonstrating its superiority and marking the first successful application of State Space Models (SSMs) in multi-modal perception tasks. Code is available at https://github.com/zifuwan/Sigma.
>
---
#### [replaced 031] Task-based Loss Functions in Computer Vision: A Comprehensive Review
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.04242v2](http://arxiv.org/pdf/2504.04242v2)**

> **作者:** Omar Elharrouss; Yasir Mahmood; Yassine Bechqito; Mohamed Adel Serhani; Elarbi Badidi; Jamal Riffi; Hamid Tairi
>
> **摘要:** Loss functions are at the heart of deep learning, shaping how models learn and perform across diverse tasks. They are used to quantify the difference between predicted outputs and ground truth labels, guiding the optimization process to minimize errors. Selecting the right loss function is critical, as it directly impacts model convergence, generalization, and overall performance across various applications, from computer vision to time series forecasting. This paper presents a comprehensive review of loss functions, covering fundamental metrics like Mean Squared Error and Cross-Entropy to advanced functions such as Adversarial and Diffusion losses. We explore their mathematical foundations, impact on model training, and strategic selection for various applications, including computer vision (Discriminative and generative), tabular data prediction, and time series forecasting. For each of these categories, we discuss the most used loss functions in the recent advancements of deep learning techniques. Also, this review explore the historical evolution, computational efficiency, and ongoing challenges in loss function design, underlining the need for more adaptive and robust solutions. Emphasis is placed on complex scenarios involving multi-modal data, class imbalances, and real-world constraints. Finally, we identify key future directions, advocating for loss functions that enhance interpretability, scalability, and generalization, leading to more effective and resilient deep learning models.
>
---
#### [replaced 032] BranchGRPO: Stable and Efficient GRPO with Structured Branching in Diffusion Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.06040v3](http://arxiv.org/pdf/2509.06040v3)**

> **作者:** Yuming Li; Yikai Wang; Yuying Zhu; Zhongyu Zhao; Ming Lu; Qi She; Shanghang Zhang
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Recent advancements in aligning image and video generative models via GRPO have achieved remarkable gains in enhancing human preference alignment. However, these methods still face high computational costs from on-policy rollouts and excessive SDE sampling steps, as well as training instability due to sparse rewards. In this paper, we propose BranchGRPO, a novel method that introduces a branch sampling policy updating the SDE sampling process. By sharing computation across common prefixes and pruning low-reward paths and redundant depths, BranchGRPO substantially lowers the per-update compute cost while maintaining or improving exploration diversity. This work makes three main contributions: (1) a branch sampling scheme that reduces rollout and training cost; (2) a tree-based advantage estimator incorporating dense process-level rewards; and (3) pruning strategies exploiting path and depth redundancy to accelerate convergence and boost performance. Experiments on image and video preference alignment show that BranchGRPO improves alignment scores by 16% over strong baselines, while cutting training time by 50%.
>
---
#### [replaced 033] Delta Velocity Rectified Flow for Text-to-Image Editing
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.05342v2](http://arxiv.org/pdf/2509.05342v2)**

> **作者:** Gaspard Beaudouin; Minghan Li; Jaeyeon Kim; Sung-Hoon Yoon; Mengyu Wang
>
> **摘要:** We propose Delta Velocity Rectified Flow (DVRF), a novel inversion-free, path-aware editing framework within rectified flow models for text-to-image editing. DVRF is a distillation-based method that explicitly models the discrepancy between the source and target velocity fields in order to mitigate over-smoothing artifacts rampant in prior distillation sampling approaches. We further introduce a time-dependent shift term to push noisy latents closer to the target trajectory, enhancing the alignment with the target distribution. We theoretically demonstrate that when this shift is disabled, DVRF reduces to Delta Denoising Score, thereby bridging score-based diffusion optimization and velocity-based rectified-flow optimization. Moreover, when the shift term follows a linear schedule under rectified-flow dynamics, DVRF generalizes the Inversion-free method FlowEdit and provides a principled theoretical interpretation for it. Experimental results indicate that DVRF achieves superior editing quality, fidelity, and controllability while requiring no architectural modifications, making it efficient and broadly applicable to text-to-image editing tasks. Code is available at https://github.com/Harvard-AI-and-Robotics-Lab/DeltaVelocityRectifiedFlow.
>
---
#### [replaced 034] VIM-GS: Visual-Inertial Monocular Gaussian Splatting via Object-level Guidance in Large Scenes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.06685v3](http://arxiv.org/pdf/2509.06685v3)**

> **作者:** Shengkai Zhang; Yuhe Liu; Guanjun Wu; Jianhua He; Xinggang Wang; Mozi Chen; Kezhong Liu
>
> **备注:** Withdrawn due to an error in the author list & incomplete experimental results
>
> **摘要:** VIM-GS is a Gaussian Splatting (GS) framework using monocular images for novel-view synthesis (NVS) in large scenes. GS typically requires accurate depth to initiate Gaussian ellipsoids using RGB-D/stereo cameras. Their limited depth sensing range makes it difficult for GS to work in large scenes. Monocular images, however, lack depth to guide the learning and lead to inferior NVS results. Although large foundation models (LFMs) for monocular depth estimation are available, they suffer from cross-frame inconsistency, inaccuracy for distant scenes, and ambiguity in deceptive texture cues. This paper aims to generate dense, accurate depth images from monocular RGB inputs for high-definite GS rendering. The key idea is to leverage the accurate but sparse depth from visual-inertial Structure-from-Motion (SfM) to refine the dense but coarse depth from LFMs. To bridge the sparse input and dense output, we propose an object-segmented depth propagation algorithm that renders the depth of pixels of structured objects. Then we develop a dynamic depth refinement module to handle the crippled SfM depth of dynamic objects and refine the coarse LFM depth. Experiments using public and customized datasets demonstrate the superior rendering quality of VIM-GS in large scenes.
>
---
#### [replaced 035] TextSSR: Diffusion-based Data Synthesis for Scene Text Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.01137v2](http://arxiv.org/pdf/2412.01137v2)**

> **作者:** Xingsong Ye; Yongkun Du; Yunbo Tao; Zhineng Chen
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Scene text recognition (STR) suffers from challenges of either less realistic synthetic training data or the difficulty of collecting sufficient high-quality real-world data, limiting the effectiveness of trained models. Meanwhile, despite producing holistically appealing text images, diffusion-based visual text generation methods struggle to synthesize accurate and realistic instance-level text at scale. To tackle this, we introduce TextSSR: a novel pipeline for Synthesizing Scene Text Recognition training data. TextSSR targets three key synthesizing characteristics: accuracy, realism, and scalability. It achieves accuracy through a proposed region-centric text generation with position-glyph enhancement, ensuring proper character placement. It maintains realism by guiding style and appearance generation using contextual hints from surrounding text or background. This character-aware diffusion architecture enjoys precise character-level control and semantic coherence preservation, without relying on natural language prompts. Therefore, TextSSR supports large-scale generation through combinatorial text permutations. Based on these, we present TextSSR-F, a dataset of 3.55 million quality-screened text instances. Extensive experiments show that STR models trained on TextSSR-F outperform those trained on existing synthetic datasets by clear margins on common benchmarks, and further improvements are observed when mixed with real-world training data. Code is available at https://github.com/YesianRohn/TextSSR.
>
---
#### [replaced 036] Self-Correcting Decoding with Generative Feedback for Mitigating Hallucinations in Large Vision-Language Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.06130v2](http://arxiv.org/pdf/2502.06130v2)**

> **作者:** Ce Zhang; Zifu Wan; Zhehan Kan; Martin Q. Ma; Simon Stepputtis; Deva Ramanan; Russ Salakhutdinov; Louis-Philippe Morency; Katia Sycara; Yaqi Xie
>
> **备注:** Accepted by ICLR 2025. Project page: https://zhangce01.github.io/DeGF/
>
> **摘要:** While recent Large Vision-Language Models (LVLMs) have shown remarkable performance in multi-modal tasks, they are prone to generating hallucinatory text responses that do not align with the given visual input, which restricts their practical applicability in real-world scenarios. In this work, inspired by the observation that the text-to-image generation process is the inverse of image-conditioned response generation in LVLMs, we explore the potential of leveraging text-to-image generative models to assist in mitigating hallucinations in LVLMs. We discover that generative models can offer valuable self-feedback for mitigating hallucinations at both the response and token levels. Building on this insight, we introduce self-correcting Decoding with Generative Feedback (DeGF), a novel training-free algorithm that incorporates feedback from text-to-image generative models into the decoding process to effectively mitigate hallucinations in LVLMs. Specifically, DeGF generates an image from the initial response produced by LVLMs, which acts as an auxiliary visual reference and provides self-feedback to verify and correct the initial response through complementary or contrastive decoding. Extensive experimental results validate the effectiveness of our approach in mitigating diverse types of hallucinations, consistently surpassing state-of-the-art methods across six benchmarks. Code is available at https://github.com/zhangce01/DeGF.
>
---
#### [replaced 037] ALOcc: Adaptive Lifting-Based 3D Semantic Occupancy and Cost Volume-Based Flow Predictions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.07725v2](http://arxiv.org/pdf/2411.07725v2)**

> **作者:** Dubing Chen; Jin Fang; Wencheng Han; Xinjing Cheng; Junbo Yin; Chenzhong Xu; Fahad Shahbaz Khan; Jianbing Shen
>
> **备注:** ICCV 2025
>
> **摘要:** 3D semantic occupancy and flow prediction are fundamental to spatiotemporal scene understanding. This paper proposes a vision-based framework with three targeted improvements. First, we introduce an occlusion-aware adaptive lifting mechanism incorporating depth denoising. This enhances the robustness of 2D-to-3D feature transformation while mitigating reliance on depth priors. Second, we enforce 3D-2D semantic consistency via jointly optimized prototypes, using confidence- and category-aware sampling to address the long-tail classes problem. Third, to streamline joint prediction, we devise a BEV-centric cost volume to explicitly correlate semantic and flow features, supervised by a hybrid classification-regression scheme that handles diverse motion scales. Our purely convolutional architecture establishes new SOTA performance on multiple benchmarks for both semantic occupancy and joint occupancy semantic-flow prediction. We also present a family of models offering a spectrum of efficiency-performance trade-offs. Our real-time version exceeds all existing real-time methods in speed and accuracy, ensuring its practical viability.
>
---
#### [replaced 038] Event Camera Meets Resource-Aware Mobile Computing: Abstraction, Algorithm, Acceleration, Application
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.22943v3](http://arxiv.org/pdf/2503.22943v3)**

> **作者:** Haoyang Wang; Ruishan Guo; Pengtao Ma; Ciyu Ruan; Xinyu Luo; Wenhua Ding; Tianyang Zhong; Jingao Xu; Yunhao Liu; Xinlei Chen
>
> **备注:** 35 pages
>
> **摘要:** With the increasing complexity of mobile device applications, these devices are evolving toward high agility. This shift imposes new demands on mobile sensing, particularly in achieving high-accuracy and low-latency. Event-based vision has emerged as a disruptive paradigm, offering high temporal resolution and low latency, making it well-suited for high-accuracy and low-latency sensing tasks on high-agility platforms. However, the presence of substantial noisy events, lack of stable, persistent semantic information, and large data volume pose challenges for event-based data processing on resource-constrained mobile devices. This paper surveys the literature from 2014 to 2025 and presents a comprehensive overview of event-based mobile sensing, encompassing its fundamental principles, event \textit{abstraction} methods, \textit{algorithm} advancements, and both hardware and software \textit{acceleration} strategies. We discuss key \textit{applications} of event cameras in mobile sensing, including visual odometry, object tracking, optical flow, and 3D reconstruction, while highlighting challenges associated with event data processing, sensor fusion, and real-time deployment. Furthermore, we outline future research directions, such as improving the event camera with advanced optics, leveraging neuromorphic computing for efficient processing, and integrating bio-inspired algorithms. To support ongoing research, we provide an open-source \textit{Online Sheet} with recent developments. We hope this survey serves as a reference, facilitating the adoption of event-based vision across diverse applications.
>
---
#### [replaced 039] Maximizing Information in Domain-Invariant Representation Improves Transfer Learning
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2306.00262v5](http://arxiv.org/pdf/2306.00262v5)**

> **作者:** Adrian Shuai Li; Elisa Bertino; Xuan-Hong Dang; Ankush Singla; Yuhai Tu; Mark N Wegman
>
> **摘要:** We propose MaxDIRep, a domain adaptation method that improves the decomposition of data representations into domain-independent and domain-dependent components. Existing methods, such as Domain-Separation Networks (DSN), use a weak orthogonality constraint between these components, which can lead to label-relevant features being partially encoded in the domain-dependent representation (DDRep) rather than the domain-independent representation (DIRep). As a result, information crucial for target-domain classification may be missing from the DIRep. MaxDIRep addresses this issue by applying a Kullback-Leibler (KL) divergence constraint to minimize the information content of the DDRep, thereby encouraging the DIRep to retain features that are both domain-invariant and predictive of target labels. Through geometric analysis and an ablation study on synthetic datasets, we show why DSN's weaker constraint can lead to suboptimal adaptation. Experiments on standard image benchmarks and a network intrusion detection task demonstrate that MaxDIRep achieves strong performance, works with pretrained models, and generalizes to non-image classification tasks.
>
---
#### [replaced 040] TerraMind: Large-Scale Generative Multimodality for Earth Observation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.11171v4](http://arxiv.org/pdf/2504.11171v4)**

> **作者:** Johannes Jakubik; Felix Yang; Benedikt Blumenstiel; Erik Scheurer; Rocco Sedona; Stefano Maurogiovanni; Jente Bosmans; Nikolaos Dionelis; Valerio Marsocci; Niklas Kopp; Rahul Ramachandran; Paolo Fraccaro; Thomas Brunschwiler; Gabriele Cavallaro; Juan Bernabe-Moreno; Nicolas Longépé
>
> **备注:** Accepted at ICCV'25
>
> **摘要:** We present TerraMind, the first any-to-any generative, multimodal foundation model for Earth observation (EO). Unlike other multimodal models, TerraMind is pretrained on dual-scale representations combining both token-level and pixel-level data across modalities. On a token level, TerraMind encodes high-level contextual information to learn cross-modal relationships, while on a pixel level, TerraMind leverages fine-grained representations to capture critical spatial nuances. We pretrained TerraMind on nine geospatial modalities of a global, large-scale dataset. In this paper, we demonstrate that (i) TerraMind's dual-scale early fusion approach unlocks a range of zero-shot and few-shot applications for Earth observation, (ii) TerraMind introduces "Thinking-in-Modalities" (TiM) -- the capability of generating additional artificial data during finetuning and inference to improve the model output -- and (iii) TerraMind achieves beyond state-of-the-art performance in community-standard benchmarks for EO like PANGAEA. The pretraining dataset, the model weights, and our code are open-sourced under a permissive license.
>
---
#### [replaced 041] MSNav: Zero-Shot Vision-and-Language Navigation with Dynamic Memory and LLM Spatial Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.16654v3](http://arxiv.org/pdf/2508.16654v3)**

> **作者:** Chenghao Liu; Zhimu Zhou; Jiachen Zhang; Minghao Zhang; Songfang Huang; Huiling Duan
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Vision-and-Language Navigation (VLN) requires an agent to interpret natural language instructions and navigate complex environments. Current approaches often adopt a "black-box" paradigm, where a single Large Language Model (LLM) makes end-to-end decisions. However, it is plagued by critical vulnerabilities, including poor spatial reasoning, weak cross-modal grounding, and memory overload in long-horizon tasks. To systematically address these issues, we propose Memory Spatial Navigation(MSNav), a framework that fuses three modules into a synergistic architecture, which transforms fragile inference into a robust, integrated intelligence. MSNav integrates three modules: Memory Module, a dynamic map memory module that tackles memory overload through selective node pruning, enhancing long-range exploration; Spatial Module, a module for spatial reasoning and object relationship inference that improves endpoint recognition; and Decision Module, a module using LLM-based path planning to execute robust actions. Powering Spatial Module, we also introduce an Instruction-Object-Space (I-O-S) dataset and fine-tune the Qwen3-4B model into Qwen-Spatial (Qwen-Sp), which outperforms leading commercial LLMs in object list extraction, achieving higher F1 and NDCG scores on the I-O-S test set. Extensive experiments on the Room-to-Room (R2R) and REVERIE datasets demonstrate MSNav's state-of-the-art performance with significant improvements in Success Rate (SR) and Success weighted by Path Length (SPL).
>
---
#### [replaced 042] RetinaGuard: Obfuscating Retinal Age in Fundus Images for Biometric Privacy Preserving
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.06142v2](http://arxiv.org/pdf/2509.06142v2)**

> **作者:** Zhengquan Luo; Chi Liu; Dongfu Xiao; Zhen Yu; Yueye Wang; Tianqing Zhu
>
> **摘要:** The integration of AI with medical images enables the extraction of implicit image-derived biomarkers for a precise health assessment. Recently, retinal age, a biomarker predicted from fundus images, is a proven predictor of systemic disease risks, behavioral patterns, aging trajectory and even mortality. However, the capability to infer such sensitive biometric data raises significant privacy risks, where unauthorized use of fundus images could lead to bioinformation leakage, breaching individual privacy. In response, we formulate a new research problem of biometric privacy associated with medical images and propose RetinaGuard, a novel privacy-enhancing framework that employs a feature-level generative adversarial masking mechanism to obscure retinal age while preserving image visual quality and disease diagnostic utility. The framework further utilizes a novel multiple-to-one knowledge distillation strategy incorporating a retinal foundation model and diverse surrogate age encoders to enable a universal defense against black-box age prediction models. Comprehensive evaluations confirm that RetinaGuard successfully obfuscates retinal age prediction with minimal impact on image quality and pathological feature representation. RetinaGuard is also flexible for extension to other medical image derived biomarkers. RetinaGuard is also flexible for extension to other medical image biomarkers.
>
---
#### [replaced 043] Vision Transformer with Sparse Scan Prior
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2405.13335v2](http://arxiv.org/pdf/2405.13335v2)**

> **作者:** Yuguang Zhang; Qihang Fan; Huaibo Huang
>
> **摘要:** In recent years, Transformers have achieved remarkable progress in computer vision tasks. However, their global modeling often comes with substantial computational overhead, in stark contrast to the human eye's efficient information processing. Inspired by the human eye's sparse scanning mechanism, we propose a \textbf{S}parse \textbf{S}can \textbf{S}elf-\textbf{A}ttention mechanism ($\rm{S}^3\rm{A}$). This mechanism predefines a series of Anchors of Interest for each token and employs local attention to efficiently model the spatial information around these anchors, avoiding redundant global modeling and excessive focus on local information. This approach mirrors the human eye's functionality and significantly reduces the computational load of vision models. Building on $\rm{S}^3\rm{A}$, we introduce the \textbf{S}parse \textbf{S}can \textbf{Vi}sion \textbf{T}ransformer (SSViT). Extensive experiments demonstrate the outstanding performance of SSViT across a variety of tasks. Specifically, on ImageNet classification, without additional supervision or training data, SSViT achieves top-1 accuracies of \textbf{84.4\%/85.7\%} with \textbf{4.4G/18.2G} FLOPs. SSViT also excels in downstream tasks such as object detection, instance segmentation, and semantic segmentation. Its robustness is further validated across diverse datasets.
>
---
#### [replaced 044] TextlessRAG: End-to-End Visual Document RAG by Speech Without Text
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.07538v2](http://arxiv.org/pdf/2509.07538v2)**

> **作者:** Peijin Xie; Shun Qian; Bingquan Liu; Dexin Wang; Lin Sun; Xiangzheng Zhang
>
> **备注:** 5 pages, 4 figures,
>
> **摘要:** Document images encapsulate a wealth of knowledge, while the portability of spoken queries enables broader and flexible application scenarios. Yet, no prior work has explored knowledge base question answering over visual document images with queries provided directly in speech. We propose TextlessRAG, the first end-to-end framework for speech-based question answering over large-scale document images. Unlike prior methods, TextlessRAG eliminates ASR, TTS and OCR, directly interpreting speech, retrieving relevant visual knowledge, and generating answers in a fully textless pipeline. To further boost performance, we integrate a layout-aware reranking mechanism to refine retrieval. Experiments demonstrate substantial improvements in both efficiency and accuracy. To advance research in this direction, we also release the first bilingual speech--document RAG dataset, featuring Chinese and English voice queries paired with multimodal document content. Both the dataset and our pipeline will be made available at repository:https://github.com/xiepeijinhit-hue/textlessrag
>
---
#### [replaced 045] Hybrid Swin Attention Networks for Simultaneously Low-Dose PET and CT Denoising
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.06591v3](http://arxiv.org/pdf/2509.06591v3)**

> **作者:** Yichao Liu; Hengzhi Xue; YueYang Teng
>
> **摘要:** Low-dose computed tomography (LDCT) and positron emission tomography (PET) have emerged as safer alternatives to conventional imaging modalities by significantly reducing radiation exposure. However, this reduction often results in increased noise and artifacts, which can compromise diagnostic accuracy. Consequently, denoising for LDCT/PET has become a vital area of research aimed at enhancing image quality while maintaining radiation safety. In this study, we introduce a novel Hybrid Swin Attention Network (HSANet), which incorporates Efficient Global Attention (EGA) modules and a hybrid upsampling module. The EGA modules enhance both spatial and channel-wise interaction, improving the network's capacity to capture relevant features, while the hybrid upsampling module mitigates the risk of overfitting to noise. We validate the proposed approach using a publicly available LDCT/PET dataset. Experimental results demonstrate that HSANet achieves superior denoising performance compared to existing methods, while maintaining a lightweight model size suitable for deployment on GPUs with standard memory configurations. This makes our approach highly practical for real-world clinical applications.
>
---
#### [replaced 046] Nearest Neighbor Projection Removal Adversarial Training
- **分类: cs.CV; cs.LG; 68T45 (Primary), 68T10 (Secondary); I.5.4**

- **链接: [http://arxiv.org/pdf/2509.07673v2](http://arxiv.org/pdf/2509.07673v2)**

> **作者:** Himanshu Singh; A. V. Subramanyam; Shivank Rajput; Mohan Kankanhalli
>
> **摘要:** Deep neural networks have exhibited impressive performance in image classification tasks but remain vulnerable to adversarial examples. Standard adversarial training enhances robustness but typically fails to explicitly address inter-class feature overlap, a significant contributor to adversarial susceptibility. In this work, we introduce a novel adversarial training framework that actively mitigates inter-class proximity by projecting out inter-class dependencies from adversarial and clean samples in the feature space. Specifically, our approach first identifies the nearest inter-class neighbors for each adversarial sample and subsequently removes projections onto these neighbors to enforce stronger feature separability. Theoretically, we demonstrate that our proposed logits correction reduces the Lipschitz constant of neural networks, thereby lowering the Rademacher complexity, which directly contributes to improved generalization and robustness. Extensive experiments across standard benchmarks including CIFAR-10, CIFAR-100, and SVHN show that our method demonstrates strong performance that is competitive with leading adversarial training techniques, highlighting significant achievements in both robust and clean accuracy. Our findings reveal the importance of addressing inter-class feature proximity explicitly to bolster adversarial robustness in DNNs.
>
---
#### [replaced 047] Reangle-A-Video: 4D Video Generation as Video-to-Video Translation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.09151v3](http://arxiv.org/pdf/2503.09151v3)**

> **作者:** Hyeonho Jeong; Suhyeon Lee; Jong Chul Ye
>
> **备注:** ICCV 2025, Project page: https://hyeonho99.github.io/reangle-a-video/
>
> **摘要:** We introduce Reangle-A-Video, a unified framework for generating synchronized multi-view videos from a single input video. Unlike mainstream approaches that train multi-view video diffusion models on large-scale 4D datasets, our method reframes the multi-view video generation task as video-to-videos translation, leveraging publicly available image and video diffusion priors. In essence, Reangle-A-Video operates in two stages. (1) Multi-View Motion Learning: An image-to-video diffusion transformer is synchronously fine-tuned in a self-supervised manner to distill view-invariant motion from a set of warped videos. (2) Multi-View Consistent Image-to-Images Translation: The first frame of the input video is warped and inpainted into various camera perspectives under an inference-time cross-view consistency guidance using DUSt3R, generating multi-view consistent starting images. Extensive experiments on static view transport and dynamic camera control show that Reangle-A-Video surpasses existing methods, establishing a new solution for multi-view video generation. We will publicly release our code and data. Project page: https://hyeonho99.github.io/reangle-a-video/
>
---
