# 计算机视觉 cs.CV

- **最新发布 164 篇**

- **更新 90 篇**

## 最新发布

#### [new 001] EdgeVidSum: Real-Time Personalized Video Summarization at the Edge
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频摘要任务，旨在解决长视频内容冗余、计算资源消耗大及隐私问题。作者提出了EdgeVidSum，一种基于边缘设备的轻量级个性化视频摘要方法，通过缩略图分析和轻量神经网络实现高效、实时的视频摘要生成。**

- **链接: [http://arxiv.org/pdf/2506.03171v1](http://arxiv.org/pdf/2506.03171v1)**

> **作者:** Ghulam Mujtaba; Eun-Seok Ryu
>
> **摘要:** EdgeVidSum is a lightweight method that generates personalized, fast-forward summaries of long-form videos directly on edge devices. The proposed approach enables real-time video summarization while safeguarding user privacy through local data processing using innovative thumbnail-based techniques and efficient neural architectures. Unlike conventional methods that process entire videos frame by frame, the proposed method uses thumbnail containers to significantly reduce computational complexity without sacrificing semantic relevance. The framework employs a hierarchical analysis approach, where a lightweight 2D CNN model identifies user-preferred content from thumbnails and generates timestamps to create fast-forward summaries. Our interactive demo highlights the system's ability to create tailored video summaries for long-form videos, such as movies, sports events, and TV shows, based on individual user preferences. The entire computation occurs seamlessly on resource-constrained devices like Jetson Nano, demonstrating how EdgeVidSum addresses the critical challenges of computational efficiency, personalization, and privacy in modern video consumption environments.
>
---
#### [new 002] SAAT: Synergistic Alternating Aggregation Transformer for Image Super-Resolution
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像超分辨率任务，旨在解决现有Transformer模型在特征融合和结构信息提取上的不足。作者提出了SAAT模型，通过CWSAG和SWSAG模块协同利用通道与空间注意力，提升超分辨率效果。实验表明其性能与当前最优方法相当。**

- **链接: [http://arxiv.org/pdf/2506.03740v1](http://arxiv.org/pdf/2506.03740v1)**

> **作者:** Jianfeng Wu; Nannan Xu
>
> **摘要:** Single image super-resolution is a well-known downstream task which aims to restore low-resolution images into high-resolution images. At present, models based on Transformers have shone brightly in the field of super-resolution due to their ability to capture long-term dependencies in information. However, current methods typically compute self-attention in nonoverlapping windows to save computational costs, and the standard self-attention computation only focuses on its results, thereby neglecting the useful information across channels and the rich spatial structural information generated in the intermediate process. Channel attention and spatial attention have, respectively, brought significant improvements to various downstream visual tasks in terms of extracting feature dependency and spatial structure relationships, but the synergistic relationship between channel and spatial attention has not been fully explored yet.To address these issues, we propose a novel model. Synergistic Alternating Aggregation Transformer (SAAT), which can better utilize the potential information of features. In SAAT, we introduce the Efficient Channel & Window Synergistic Attention Group (CWSAG) and the Spatial & Window Synergistic Attention Group (SWSAG). On the one hand, CWSAG combines efficient channel attention with shifted window attention, enhancing non-local feature fusion, and producing more visually appealing results. On the other hand, SWSAG leverages spatial attention to capture rich structured feature information, thereby enabling SAAT to more effectively extract structural features.Extensive experimental results and ablation studies demonstrate the effectiveness of SAAT in the field of super-resolution. SAAT achieves performance comparable to that of the state-of-the-art (SOTA) under the same quantity of parameters.
>
---
#### [new 003] ConText: Driving In-context Learning for Text Removal and Segmentation
- **分类: cs.CV**

- **简介: 该论文属于光学字符识别任务，旨在解决文本移除与分割的上下文学习问题。现有方法依赖单步推理，效果受限。作者提出ConText模型，通过任务链组合器和上下文感知聚合，增强模型推理能力，并采用自提示策略应对视觉异质性问题，提升了领域内与领域外性能。**

- **链接: [http://arxiv.org/pdf/2506.03799v1](http://arxiv.org/pdf/2506.03799v1)**

> **作者:** Fei Zhang; Pei Zhang; Baosong Yang; Fei Huang; Yanfeng Wang; Ya Zhang
>
> **备注:** 19 pages, 9 figures, Accepted at ICML 2025
>
> **摘要:** This paper presents the first study on adapting the visual in-context learning (V-ICL) paradigm to optical character recognition tasks, specifically focusing on text removal and segmentation. Most existing V-ICL generalists employ a reasoning-as-reconstruction approach: they turn to using a straightforward image-label compositor as the prompt and query input, and then masking the query label to generate the desired output. This direct prompt confines the model to a challenging single-step reasoning process. To address this, we propose a task-chaining compositor in the form of image-removal-segmentation, providing an enhanced prompt that elicits reasoning with enriched intermediates. Additionally, we introduce context-aware aggregation, integrating the chained prompt pattern into the latent query representation, thereby strengthening the model's in-context reasoning. We also consider the issue of visual heterogeneity, which complicates the selection of homogeneous demonstrations in text recognition. Accordingly, this is effectively addressed through a simple self-prompting strategy, preventing the model's in-context learnability from devolving into specialist-like, context-free inference. Collectively, these insights culminate in our ConText model, which achieves new state-of-the-art across both in- and out-of-domain benchmarks. The code is available at https://github.com/Ferenas/ConText.
>
---
#### [new 004] Average Calibration Losses for Reliable Uncertainty in Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决模型预测不确定性不可靠的问题。作者提出了可微分的平均校准误差损失（mL1-ACE），通过硬/软分箱策略优化像素级校准，降低了校准误差并保持分割性能。实验在多个数据集上验证了方法有效性，并引入数据集可靠性直方图评估整体校准表现。**

- **链接: [http://arxiv.org/pdf/2506.03942v1](http://arxiv.org/pdf/2506.03942v1)**

> **作者:** Theodore Barfoot; Luis C. Garcia-Peraza-Herrera; Samet Akcay; Ben Glocker; Tom Vercauteren
>
> **备注:** 12 pages, 5 figures, IEEE TMI submission
>
> **摘要:** Deep neural networks for medical image segmentation are often overconfident, compromising both reliability and clinical utility. In this work, we propose differentiable formulations of marginal L1 Average Calibration Error (mL1-ACE) as an auxiliary loss that can be computed on a per-image basis. We compare both hard- and soft-binning approaches to directly improve pixel-wise calibration. Our experiments on four datasets (ACDC, AMOS, KiTS, BraTS) demonstrate that incorporating mL1-ACE significantly reduces calibration errors, particularly Average Calibration Error (ACE) and Maximum Calibration Error (MCE), while largely maintaining high Dice Similarity Coefficients (DSCs). We find that the soft-binned variant yields the greatest improvements in calibration, over the Dice plus cross-entropy loss baseline, but often compromises segmentation performance, with hard-binned mL1-ACE maintaining segmentation performance, albeit with weaker calibration improvement. To gain further insight into calibration performance and its variability across an imaging dataset, we introduce dataset reliability histograms, an aggregation of per-image reliability diagrams. The resulting analysis highlights improved alignment between predicted confidences and true accuracies. Overall, our approach not only enhances the trustworthiness of segmentation predictions but also shows potential for safer integration of deep learning methods into clinical workflows. We share our code here: https://github.com/cai4cai/Average-Calibration-Losses
>
---
#### [new 005] Target Semantics Clustering via Text Representations for Robust Universal Domain Adaptation
- **分类: cs.CV**

- **简介: 该论文属于通用域适应任务，旨在解决域迁移和未知类别迁移下的知识迁移问题。通过在文本表示空间中搜索目标语义中心，提出TASC方法进行聚类，并设计UniMS评分函数检测开放集样本，实现鲁棒的域对齐与分类。**

- **链接: [http://arxiv.org/pdf/2506.03521v1](http://arxiv.org/pdf/2506.03521v1)**

> **作者:** Weinan He; Zilei Wang; Yixin Zhang
>
> **备注:** Camera-ready version for AAAI 2025
>
> **摘要:** Universal Domain Adaptation (UniDA) focuses on transferring source domain knowledge to the target domain under both domain shift and unknown category shift. Its main challenge lies in identifying common class samples and aligning them. Current methods typically obtain target domain semantics centers from an unconstrained continuous image representation space. Due to domain shift and the unknown number of clusters, these centers often result in complex and less robust alignment algorithm. In this paper, based on vision-language models, we search for semantic centers in a semantically meaningful and discrete text representation space. The constrained space ensures almost no domain bias and appropriate semantic granularity for these centers, enabling a simple and robust adaptation algorithm. Specifically, we propose TArget Semantics Clustering (TASC) via Text Representations, which leverages information maximization as a unified objective and involves two stages. First, with the frozen encoders, a greedy search-based framework is used to search for an optimal set of text embeddings to represent target semantics. Second, with the search results fixed, encoders are refined based on gradient descent, simultaneously achieving robust domain alignment and private class clustering. Additionally, we propose Universal Maximum Similarity (UniMS), a scoring function tailored for detecting open-set samples in UniDA. Experimentally, we evaluate the universality of UniDA algorithms under four category shift scenarios. Extensive experiments on four benchmarks demonstrate the effectiveness and robustness of our method, which has achieved state-of-the-art performance.
>
---
#### [new 006] Struct2D: A Perception-Guided Framework for Spatial Reasoning in Large Multimodal Models
- **分类: cs.CV**

- **简介: 该论文属于多模态模型的空间推理任务，旨在解决如何在不依赖显式3D输入的情况下，提升大型多模态模型（LMMs）的空间推理能力。作者提出了Struct2D框架，通过结合鸟瞰图、对象标记和元数据进行感知引导，验证了LMMs在零样本设置下的空间推理能力，并构建了Struct2D-Set数据集用于微调模型，最终在多个基准任务上取得了良好表现。**

- **链接: [http://arxiv.org/pdf/2506.04220v1](http://arxiv.org/pdf/2506.04220v1)**

> **作者:** Fangrui Zhu; Hanhui Wang; Yiming Xie; Jing Gu; Tianye Ding; Jianwei Yang; Huaizu Jiang
>
> **备注:** https://github.com/neu-vi/struct2d
>
> **摘要:** Unlocking spatial reasoning in Large Multimodal Models (LMMs) is crucial for enabling intelligent interaction with 3D environments. While prior efforts often rely on explicit 3D inputs or specialized model architectures, we ask: can LMMs reason about 3D space using only structured 2D representations derived from perception? We introduce Struct2D, a perception-guided prompting framework that combines bird's-eye-view (BEV) images with object marks and object-centric metadata, optionally incorporating egocentric keyframes when needed. Using Struct2D, we conduct an in-depth zero-shot analysis of closed-source LMMs (e.g., GPT-o3) and find that they exhibit surprisingly strong spatial reasoning abilities when provided with structured 2D inputs, effectively handling tasks such as relative direction estimation and route planning. Building on these insights, we construct Struct2D-Set, a large-scale instruction tuning dataset with 200K fine-grained QA pairs across eight spatial reasoning categories, generated automatically from 3D indoor scenes. We fine-tune an open-source LMM (Qwen2.5VL) on Struct2D-Set, achieving competitive performance on multiple benchmarks, including 3D question answering, dense captioning, and object grounding. Our approach demonstrates that structured 2D inputs can effectively bridge perception and language reasoning in LMMs-without requiring explicit 3D representations as input. We will release both our code and dataset to support future research.
>
---
#### [new 007] Accelerating SfM-based Pose Estimation with Dominating Set
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于三维重建与姿态估计任务，旨在解决基于SfM的位姿估计速度慢的问题。作者提出一种基于图论中支配集的预处理方法，加快姿态估计过程，在保持精度的同时显著减少参考图像和点云规模。**

- **链接: [http://arxiv.org/pdf/2506.03667v1](http://arxiv.org/pdf/2506.03667v1)**

> **作者:** Joji Joseph; Bharadwaj Amrutur; Shalabh Bhatnagar
>
> **摘要:** This paper introduces a preprocessing technique to speed up Structure-from-Motion (SfM) based pose estimation, which is critical for real-time applications like augmented reality (AR), virtual reality (VR), and robotics. Our method leverages the concept of a dominating set from graph theory to preprocess SfM models, significantly enhancing the speed of the pose estimation process without losing significant accuracy. Using the OnePose dataset, we evaluated our method across various SfM-based pose estimation techniques. The results demonstrate substantial improvements in processing speed, ranging from 1.5 to 14.48 times, and a reduction in reference images and point cloud size by factors of 17-23 and 2.27-4, respectively. This work offers a promising solution for efficient and accurate 3D pose estimation, balancing speed and accuracy in real-time applications.
>
---
#### [new 008] RoNFA: Robust Neural Field-based Approach for Few-Shot Image Classification with Noisy Labels
- **分类: cs.CV; 68Txx; I.5.1**

- **简介: 该论文属于小样本图像分类任务，旨在解决标签噪声下的分类鲁棒性问题。作者提出了RoNFA方法，利用两个神经场分别进行特征和类别表示，并通过自适应感受野机制提升模型在标签噪声下的准确率。实验表明其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.03461v1](http://arxiv.org/pdf/2506.03461v1)**

> **作者:** Nan Xiang; Lifeng Xing; Dequan Jin
>
> **备注:** 7 pages, 1 figure
>
> **摘要:** In few-shot learning (FSL), the labeled samples are scarce. Thus, label errors can significantly reduce classification accuracy. Since label errors are inevitable in realistic learning tasks, improving the robustness of the model in the presence of label errors is critical. This paper proposes a new robust neural field-based image approach (RoNFA) for few-shot image classification with noisy labels. RoNFA consists of two neural fields for feature and category representation. They correspond to the feature space and category set. Each neuron in the field for category representation (FCR) has a receptive field (RF) on the field for feature representation (FFR) centered at the representative neuron for its category generated by soft clustering. In the prediction stage, the range of these receptive fields adapts according to the neuronal activation in FCR to ensure prediction accuracy. These learning strategies provide the proposed model with excellent few-shot learning capability and strong robustness against label noises. The experimental results on real-world FSL datasets with three different types of label noise demonstrate that the proposed method significantly outperforms state-of-the-art FSL methods. Its accuracy obtained in the presence of noisy labels even surpasses the results obtained by state-of-the-art FSL methods trained on clean support sets, indicating its strong robustness against noisy labels.
>
---
#### [new 009] Dual Branch VideoMamba with Gated Class Token Fusion for Violence Detection
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视频暴力检测任务，旨在解决现有模型在长时序依赖和计算效率上的不足。作者提出了一种双分支架构VideoMamba，并引入门控类别令牌融合机制，结合状态空间模型提升时空特征建模能力。此外，构建了一个新的数据集基准，实现了更优的性能与效率平衡。**

- **链接: [http://arxiv.org/pdf/2506.03162v1](http://arxiv.org/pdf/2506.03162v1)**

> **作者:** Damith Chamalke Senadeera; Xiaoyun Yang; Dimitrios Kollias; Gregory Slabaugh
>
> **摘要:** The rapid proliferation of surveillance cameras has increased the demand for automated violence detection. While CNNs and Transformers have shown success in extracting spatio-temporal features, they struggle with long-term dependencies and computational efficiency. We propose Dual Branch VideoMamba with Gated Class Token Fusion (GCTF), an efficient architecture combining a dual-branch design and a state-space model (SSM) backbone where one branch captures spatial features, while the other focuses on temporal dynamics, with continuous fusion via a gating mechanism. We also present a new benchmark by merging RWF-2000, RLVS, and VioPeru datasets in video violence detection, ensuring strict separation between training and testing sets. Our model achieves state-of-the-art performance on this benchmark offering an optimal balance between accuracy and computational efficiency, demonstrating the promise of SSMs for scalable, real-time surveillance violence detection.
>
---
#### [new 010] GlobalBuildingAtlas: An Open Global and Complete Dataset of Building Polygons, Heights and LoD1 3D Models
- **分类: cs.CV**

- **简介: 该论文属于地理信息科学与遥感任务，旨在解决全球建筑物数据不完整、精度低的问题。作者利用机器学习方法，从卫星影像中提取建筑物轮廓和高度，并融合多源数据生成高质量的2D与3D建筑数据集GlobalBuildingAtlas，包含2.75亿个建筑多边形、高精度高度图及LoD1级三维模型，支持全球尺度的空间分析与可持续发展目标监测。**

- **链接: [http://arxiv.org/pdf/2506.04106v1](http://arxiv.org/pdf/2506.04106v1)**

> **作者:** Xiao Xiang Zhu; Sining Chen; Fahong Zhang; Yilei Shi; Yuanyuan Wang
>
> **摘要:** We introduce GlobalBuildingAtlas, a publicly available dataset providing global and complete coverage of building polygons, heights and Level of Detail 1 (LoD1) 3D building models. This is the first open dataset to offer high quality, consistent, and complete building data in 2D and 3D form at the individual building level on a global scale. Towards this dataset, we developed machine learning-based pipelines to derive building polygons and heights (called GBA.Height) from global PlanetScope satellite data, respectively. Also a quality-based fusion strategy was employed to generate higher-quality polygons (called GBA.Polygon) based on existing open building polygons, including our own derived one. With more than 2.75 billion buildings worldwide, GBA.Polygon surpasses the most comprehensive database to date by more than 1 billion buildings. GBA.Height offers the most detailed and accurate global 3D building height maps to date, achieving a spatial resolution of 3x3 meters-30 times finer than previous global products (90 m), enabling a high-resolution and reliable analysis of building volumes at both local and global scales. Finally, we generated a global LoD1 building model (called GBA.LoD1) from the resulting GBA.Polygon and GBA.Height. GBA.LoD1 represents the first complete global LoD1 building models, including 2.68 billion building instances with predicted heights, i.e., with a height completeness of more than 97%, achieving RMSEs ranging from 1.5 m to 8.9 m across different continents. With its height accuracy, comprehensive global coverage and rich spatial details, GlobalBuildingAltas offers novel insights on the status quo of global buildings, which unlocks unprecedented geospatial analysis possibilities, as showcased by a better illustration of where people live and a more comprehensive monitoring of the progress on the 11th Sustainable Development Goal of the United Nations.
>
---
#### [new 011] DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决文本到视频扩散模型中偏好优化的粗粒度和运动偏差问题。作者提出DenseDPO方法，通过使用真实视频的去噪副本来构建对齐的视频对，并在短片段上进行细粒度偏好标注，从而提供更精确的学习信号。此外，还探索了利用现有视觉语言模型实现自动偏好标注的可能性。**

- **链接: [http://arxiv.org/pdf/2506.03517v1](http://arxiv.org/pdf/2506.03517v1)**

> **作者:** Ziyi Wu; Anil Kag; Ivan Skorokhodov; Willi Menapace; Ashkan Mirzaei; Igor Gilitschenski; Sergey Tulyakov; Aliaksandr Siarohin
>
> **备注:** Project page: https://snap-research.github.io/DenseDPO/
>
> **摘要:** Direct Preference Optimization (DPO) has recently been applied as a post-training technique for text-to-video diffusion models. To obtain training data, annotators are asked to provide preferences between two videos generated from independent noise. However, this approach prohibits fine-grained comparisons, and we point out that it biases the annotators towards low-motion clips as they often contain fewer visual artifacts. In this work, we introduce DenseDPO, a method that addresses these shortcomings by making three contributions. First, we create each video pair for DPO by denoising corrupted copies of a ground truth video. This results in aligned pairs with similar motion structures while differing in local details, effectively neutralizing the motion bias. Second, we leverage the resulting temporal alignment to label preferences on short segments rather than entire clips, yielding a denser and more precise learning signal. With only one-third of the labeled data, DenseDPO greatly improves motion generation over vanilla DPO, while matching it in text alignment, visual quality, and temporal consistency. Finally, we show that DenseDPO unlocks automatic preference annotation using off-the-shelf Vision Language Models (VLMs): GPT accurately predicts segment-level preferences similar to task-specifically fine-tuned video reward models, and DenseDPO trained on these labels achieves performance close to using human labels.
>
---
#### [new 012] Animal Pose Labeling Using General-Purpose Point Trackers
- **分类: cs.CV**

- **简介: 该论文属于动物姿态估计任务，旨在解决现有方法因训练数据不足导致的不可靠问题。通过测试时优化策略，在少量标注帧上微调轻量级外观嵌入模型，实现视频中动物姿态的自动标注，降低了数据收集难度并提升了性能。**

- **链接: [http://arxiv.org/pdf/2506.03868v1](http://arxiv.org/pdf/2506.03868v1)**

> **作者:** Zhuoyang Pan; Boxiao Pan; Guandao Yang; Adam W. Harley; Leonidas Guibas
>
> **摘要:** Automatically estimating animal poses from videos is important for studying animal behaviors. Existing methods do not perform reliably since they are trained on datasets that are not comprehensive enough to capture all necessary animal behaviors. However, it is very challenging to collect such datasets due to the large variations in animal morphology. In this paper, we propose an animal pose labeling pipeline that follows a different strategy, i.e. test time optimization. Given a video, we fine-tune a lightweight appearance embedding inside a pre-trained general-purpose point tracker on a sparse set of annotated frames. These annotations can be obtained from human labelers or off-the-shelf pose detectors. The fine-tuned model is then applied to the rest of the frames for automatic labeling. Our method achieves state-of-the-art performance at a reasonable annotation cost. We believe our pipeline offers a valuable tool for the automatic quantification of animal behavior. Visit our project webpage at https://zhuoyang-pan.github.io/animal-labeling.
>
---
#### [new 013] Cross-Modal Urban Sensing: Evaluating Sound-Vision Alignment Across Street-Level and Aerial Imagery
- **分类: cs.CV**

- **简介: 该论文属于跨模态城市感知任务，旨在解决如何将城市声音与视觉信息对齐的问题。研究通过结合街景和遥感图像及声音数据，使用多模态模型评估不同视觉表示方法在捕捉声学语义上的效果，发现嵌入模型更优，而分割方法在生态分类上更具解释性。**

- **链接: [http://arxiv.org/pdf/2506.03388v1](http://arxiv.org/pdf/2506.03388v1)**

> **作者:** Pengyu Chen; Xiao Huang; Teng Fei; Sicheng Wang
>
> **摘要:** Environmental soundscapes convey substantial ecological and social information regarding urban environments; however, their potential remains largely untapped in large-scale geographic analysis. In this study, we investigate the extent to which urban sounds correspond with visual scenes by comparing various visual representation strategies in capturing acoustic semantics. We employ a multimodal approach that integrates geo-referenced sound recordings with both street-level and remote sensing imagery across three major global cities: London, New York, and Tokyo. Utilizing the AST model for audio, along with CLIP and RemoteCLIP for imagery, as well as CLIPSeg and Seg-Earth OV for semantic segmentation, we extract embeddings and class-level features to evaluate cross-modal similarity. The results indicate that street view embeddings demonstrate stronger alignment with environmental sounds compared to segmentation outputs, whereas remote sensing segmentation is more effective in interpreting ecological categories through a Biophony--Geophony--Anthrophony (BGA) framework. These findings imply that embedding-based models offer superior semantic alignment, while segmentation-based methods provide interpretable links between visual structure and acoustic ecology. This work advances the burgeoning field of multimodal urban sensing by offering novel perspectives for incorporating sound into geospatial analysis.
>
---
#### [new 014] Improvement of human health lifespan with hybrid group pose estimation methods
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉中的人体姿态估计任务，旨在解决多人实时姿态估计的准确性与鲁棒性问题。作者提出了一种基于混合集成的群体姿态估计算法，融合多种方法提升性能，并通过实验验证其在健康监测等实时应用中的有效性。**

- **链接: [http://arxiv.org/pdf/2506.03169v1](http://arxiv.org/pdf/2506.03169v1)**

> **作者:** Arindam Chaudhuri
>
> **摘要:** Human beings rely heavily on estimation of poses in order to access their body movements. Human pose estimation methods take advantage of computer vision advances in order to track human body movements in real life applications. This comes from videos which are recorded through available devices. These para-digms provide potential to make human movement measurement more accessible to users. The consumers of pose estimation movements believe that human poses content tend to supplement available videos. This has increased pose estimation software usage to estimate human poses. In order to address this problem, we develop hybrid-ensemble-based group pose estimation method to improve human health. This proposed hybrid-ensemble-based group pose estimation method aims to detect multi-person poses using modified group pose estimation and modified real time pose estimation. This ensemble allows fusion of performance of stated methods in real time. The input poses from images are fed into individual meth-ods. The pose transformation method helps to identify relevant features for en-semble to perform training effectively. After this, customized pre-trained hybrid ensemble is trained on public benchmarked datasets which is being evaluated through test datasets. The effectiveness and viability of proposed method is estab-lished based on comparative analysis of group pose estimation methods and ex-periments conducted on benchmarked datasets. It provides best optimized results in real-time pose estimation. It makes pose estimation method more robust to oc-clusion and improves dense regression accuracy. These results have affirmed po-tential application of this method in several real-time situations with improvement in human health life span
>
---
#### [new 015] OpenCarbon: A Contrastive Learning-based Cross-Modality Neural Approach for High-Resolution Carbon Emission Prediction Using Open Data
- **分类: cs.CV; cs.AI; physics.soc-ph**

- **简介: 该论文属于碳排放预测任务，旨在解决高分辨率城市碳排放估计问题。利用卫星图像和POI数据，通过跨模态信息提取与融合模块及邻域感知聚合模块，提升预测精度。实验表明其性能优于现有方法，有助于碳治理和减排规划。**

- **链接: [http://arxiv.org/pdf/2506.03224v1](http://arxiv.org/pdf/2506.03224v1)**

> **作者:** Jinwei Zeng; Yu Liu; Guozhen Zhang; Jingtao Ding; Yuming Lin; Jian Yuan; Yong Li
>
> **备注:** Accepted by IJCAI 2025
>
> **摘要:** Accurately estimating high-resolution carbon emissions is crucial for effective emission governance and mitigation planning. While conventional methods for precise carbon accounting are hindered by substantial data collection efforts, the rise of open data and advanced learning techniques offers a promising solution. Once an open data-based prediction model is developed and trained, it can easily infer emissions for new areas based on available open data. To address this, we incorporate two modalities of open data, satellite images and point-of-interest (POI) data, to predict high-resolution urban carbon emissions, with satellite images providing macroscopic and static and POI data offering fine-grained and relatively dynamic functionality information. However, estimating high-resolution carbon emissions presents two significant challenges: the intertwined and implicit effects of various functionalities on carbon emissions, and the complex spatial contiguity correlations that give rise to the agglomeration effect. Our model, OpenCarbon, features two major designs that target the challenges: a cross-modality information extraction and fusion module to extract complementary functionality information from two modules and model their interactions, and a neighborhood-informed aggregation module to capture the spatial contiguity correlations. Extensive experiments demonstrate our model's superiority, with a significant performance gain of 26.6\% on R2. Further generalizability tests and case studies also show OpenCarbon's capacity to capture the intrinsic relation between urban functionalities and carbon emissions, validating its potential to empower efficient carbon governance and targeted carbon mitigation planning. Codes and data are available: https://github.com/JinweiZzz/OpenCarbon.
>
---
#### [new 016] Semiconductor SEM Image Defect Classification Using Supervised and Semi-Supervised Learning with Vision Transformers
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决半导体缺陷检测中人工分类效率低、易出错的问题。作者使用视觉Transformer（ViT）结合监督与半监督学习方法，实现了对扫描电子显微镜（SEM）图像中的缺陷自动分类，在IBM工厂的真实数据上取得了90%以上的准确率，并验证了模型的高效性与实用性。**

- **链接: [http://arxiv.org/pdf/2506.03345v1](http://arxiv.org/pdf/2506.03345v1)**

> **作者:** Chien-Fu; Huang; Katherine Sieg; Leonid Karlinksy; Nash Flores; Rebekah Sheraw; Xin Zhang
>
> **备注:** Published at 36th Annual SEMI Advanced Semiconductor Manufacturing Conference (ASMC) 2025
>
> **摘要:** Controlling defects in semiconductor processes is important for maintaining yield, improving production cost, and preventing time-dependent critical component failures. Electron beam-based imaging has been used as a tool to survey wafers in the line and inspect for defects. However, manual classification of images for these nano-scale defects is limited by time, labor constraints, and human biases. In recent years, deep learning computer vision algorithms have shown to be effective solutions for image-based inspection applications in industry. This work proposes application of vision transformer (ViT) neural networks for automatic defect classification (ADC) of scanning electron microscope (SEM) images of wafer defects. We evaluated our proposed methods on 300mm wafer semiconductor defect data from our fab in IBM Albany. We studied 11 defect types from over 7400 total images and investigated the potential of transfer learning of DinoV2 and semi-supervised learning for improved classification accuracy and efficient computation. We were able to achieve classification accuracies of over 90% with less than 15 images per defect class. Our work demonstrates the potential to apply the proposed framework for a platform agnostic in-house classification tool with faster turnaround time and flexibility.
>
---
#### [new 017] Isharah: A Large-Scale Multi-Scene Dataset for Continuous Sign Language Recognition
- **分类: cs.CV**

- **简介: 该论文属于连续手语识别任务，旨在解决现有数据集受限于孤立手语识别且场景单一的问题。作者提出了Isharah数据集，包含30,000个视频片段，由18名聋人专业手语者在非受控环境下录制，并提供词汇级标注，适用于连续手语识别和翻译系统研究。**

- **链接: [http://arxiv.org/pdf/2506.03615v1](http://arxiv.org/pdf/2506.03615v1)**

> **作者:** Sarah Alyami; Hamzah Luqman; Sadam Al-Azani; Maad Alowaifeer; Yazeed Alharbi; Yaser Alonaizan
>
> **摘要:** Current benchmarks for sign language recognition (SLR) focus mainly on isolated SLR, while there are limited datasets for continuous SLR (CSLR), which recognizes sequences of signs in a video. Additionally, existing CSLR datasets are collected in controlled settings, which restricts their effectiveness in building robust real-world CSLR systems. To address these limitations, we present Isharah, a large multi-scene dataset for CSLR. It is the first dataset of its type and size that has been collected in an unconstrained environment using signers' smartphone cameras. This setup resulted in high variations of recording settings, camera distances, angles, and resolutions. This variation helps with developing sign language understanding models capable of handling the variability and complexity of real-world scenarios. The dataset consists of 30,000 video clips performed by 18 deaf and professional signers. Additionally, the dataset is linguistically rich as it provides a gloss-level annotation for all dataset's videos, making it useful for developing CSLR and sign language translation (SLT) systems. This paper also introduces multiple sign language understanding benchmarks, including signer-independent and unseen-sentence CSLR, along with gloss-based and gloss-free SLT. The Isharah dataset is available on https://snalyami.github.io/Isharah_CSLR/.
>
---
#### [new 018] Analyzing Transformer Models and Knowledge Distillation Approaches for Image Captioning on Edge AI
- **分类: cs.CV**

- **简介: 该论文研究在边缘AI设备上部署高效的图像描述生成模型。任务是解决边缘设备资源受限导致的实时性与计算需求之间的矛盾。通过评估轻量化Transformer模型和应用知识蒸馏技术，实现了低资源消耗下的高效推理，保持了模型性能。**

- **链接: [http://arxiv.org/pdf/2506.03607v1](http://arxiv.org/pdf/2506.03607v1)**

> **作者:** Wing Man Casca Kwok; Yip Chiu Tung; Kunal Bhagchandani
>
> **摘要:** Edge computing decentralizes processing power to network edge, enabling real-time AI-driven decision-making in IoT applications. In industrial automation such as robotics and rugged edge AI, real-time perception and intelligence are critical for autonomous operations. Deploying transformer-based image captioning models at the edge can enhance machine perception, improve scene understanding for autonomous robots, and aid in industrial inspection. However, these edge or IoT devices are often constrained in computational resources for physical agility, yet they have strict response time requirements. Traditional deep learning models can be too large and computationally demanding for these devices. In this research, we present findings of transformer-based models for image captioning that operate effectively on edge devices. By evaluating resource-effective transformer models and applying knowledge distillation techniques, we demonstrate inference can be accelerated on resource-constrained devices while maintaining model performance using these techniques.
>
---
#### [new 019] MamFusion: Multi-Mamba with Temporal Fusion for Partially Relevant Video Retrieval
- **分类: cs.CV**

- **简介: 该论文属于多媒体检索任务，旨在解决部分相关视频检索（PRVR）中的信息冗余问题。作者提出MamFusion框架，结合多Mamba模块与时间融合策略，提升长视频内容理解与文本-视频相关性建模，增强检索效果。**

- **链接: [http://arxiv.org/pdf/2506.03473v1](http://arxiv.org/pdf/2506.03473v1)**

> **作者:** Xinru Ying; Jiaqi Mo; Jingyang Lin; Canghong Jin; Fangfang Wang; Lina Wei
>
> **摘要:** Partially Relevant Video Retrieval (PRVR) is a challenging task in the domain of multimedia retrieval. It is designed to identify and retrieve untrimmed videos that are partially relevant to the provided query. In this work, we investigate long-sequence video content understanding to address information redundancy issues. Leveraging the outstanding long-term state space modeling capability and linear scalability of the Mamba module, we introduce a multi-Mamba module with temporal fusion framework (MamFusion) tailored for PRVR task. This framework effectively captures the state-relatedness in long-term video content and seamlessly integrates it into text-video relevance understanding, thereby enhancing the retrieval process. Specifically, we introduce Temporal T-to-V Fusion and Temporal V-to-T Fusion to explicitly model temporal relationships between text queries and video moments, improving contextual awareness and retrieval accuracy. Extensive experiments conducted on large-scale datasets demonstrate that MamFusion achieves state-of-the-art performance in retrieval effectiveness. Code is available at the link: https://github.com/Vision-Multimodal-Lab-HZCU/MamFusion.
>
---
#### [new 020] Person Re-Identification System at Semantic Level based on Pedestrian Attributes Ontology
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频监控中行人重识别（Re-ID）任务，旨在解决大规模数据集、属性不平衡、视角变化及细粒度特征利用不足等问题。作者提出了一种结合行人属性本体（PAO）、局部多任务卷积神经网络（Local MDCNN）和不平衡数据解决模块（IDS）的统一系统，通过语义信息预筛选匹配候选，提升了Re-ID性能。实验表明在Market1501数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.04143v1](http://arxiv.org/pdf/2506.04143v1)**

> **作者:** Ngoc Q. Ly; Hieu N. M. Cao; Thi T. Nguyen
>
> **摘要:** Person Re-Identification (Re-ID) is a very important task in video surveillance systems such as tracking people, finding people in public places, or analysing customer behavior in supermarkets. Although there have been many works to solve this problem, there are still remaining challenges such as large-scale datasets, imbalanced data, viewpoint, fine grained data (attributes), the Local Features are not employed at semantic level in online stage of Re-ID task, furthermore, the imbalanced data problem of attributes are not taken into consideration. This paper has proposed a Unified Re-ID system consisted of three main modules such as Pedestrian Attribute Ontology (PAO), Local Multi-task DCNN (Local MDCNN), Imbalance Data Solver (IDS). The new main point of our Re-ID system is the power of mutual support of PAO, Local MDCNN and IDS to exploit the inner-group correlations of attributes and pre-filter the mismatch candidates from Gallery set based on semantic information as Fashion Attributes and Facial Attributes, to solve the imbalanced data of attributes without adjusting network architecture and data augmentation. We experimented on the well-known Market1501 dataset. The experimental results have shown the effectiveness of our Re-ID system and it could achieve the higher performance on Market1501 dataset in comparison to some state-of-the-art Re-ID methods.
>
---
#### [new 021] HUMOF: Human Motion Forecasting in Interactive Social Scenes
- **分类: cs.CV**

- **简介: 该论文属于人类运动预测任务，旨在解决复杂社交场景中因人与人、人与环境交互导致的行为不确定性问题。作者提出了层次化交互特征表示和由粗到细的交互推理模块，结合空间与频率视角提升预测准确性，并在多个数据集上达到最优性能。**

- **链接: [http://arxiv.org/pdf/2506.03753v1](http://arxiv.org/pdf/2506.03753v1)**

> **作者:** Caiyi Sun; Yujing Sun; Xiao Han; Zemin Yang; Jiawei Liu; Xinge Zhu; Siu Ming Yiu; Yuexin Ma
>
> **摘要:** Complex scenes present significant challenges for predicting human behaviour due to the abundance of interaction information, such as human-human and humanenvironment interactions. These factors complicate the analysis and understanding of human behaviour, thereby increasing the uncertainty in forecasting human motions. Existing motion prediction methods thus struggle in these complex scenarios. In this paper, we propose an effective method for human motion forecasting in interactive scenes. To achieve a comprehensive representation of interactions, we design a hierarchical interaction feature representation so that high-level features capture the overall context of the interactions, while low-level features focus on fine-grained details. Besides, we propose a coarse-to-fine interaction reasoning module that leverages both spatial and frequency perspectives to efficiently utilize hierarchical features, thereby enhancing the accuracy of motion predictions. Our method achieves state-of-the-art performance across four public datasets. Code will be released when this paper is published.
>
---
#### [new 022] Multimodal Foundation Model for Cross-Modal Retrieval and Activity Recognition Tasks
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于跨模态检索与活动识别任务，旨在解决现有模型对全身人类活动分析不足的问题。作者提出了AURA-MFM多模态基础模型，融合第三人称视频、动作捕捉、IMU和文本四种模态，并采用基于Transformer的IMU编码器提升性能，在检索和活动识别任务中表现出色，尤其在零样本动作分类中显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.03174v1](http://arxiv.org/pdf/2506.03174v1)**

> **作者:** Koki Matsuishi; Kosuke Ukita; Tsuyoshi Okita
>
> **备注:** 25 pages, 8 figures
>
> **摘要:** In recent years, the widespread adoption of wearable devices has highlighted the growing importance of behavior analysis using IMU. While applications span diverse fields such as healthcare and robotics, recent studies have increasingly focused on multimodal analysis, in addition to unimodal analysis. Several studies have proposed multimodal foundation models that incorporate first-person video and text data; however, these models still fall short in providing a detailed analysis of full-body human activity. To address this limitation, we propose Activity Understanding and Representations Alignment - Multimodal Foundation Model (AURA-MFM), a foundational model integrating four modalities: third-person video, motion capture, IMU, and text. By incorporating third-person video and motion capture data, the model enables a detailed and multidimensional understanding of human activity, which first-person perspectives alone fail to capture. Additionally, a Transformer-based IMU encoder is employed to enhance the model's overall performance. Experimental evaluations on retrieval and activity recognition tasks demonstrate that our model surpasses existing methods. Notably, in the zero-shot classification for action recognition, our method achieved significantly higher performance, with an F1-score of 0.6226 and an accuracy of 0.7320, whereas the existing method recorded an F1-score of 0.0747 and an accuracy of 0.1961.
>
---
#### [new 023] Contour Errors: An Ego-Centric Metric for Reliable 3D Multi-Object Tracking
- **分类: cs.CV**

- **简介: 论文属于3D多目标跟踪任务，旨在解决传统2D匹配度量（如IoU和CPD）在复杂3D场景中失效的问题。作者提出“轮廓误差”（CE）这一以物体为中心的新型评估指标，通过对比自动驾驶车辆坐标系中的边界框，提升匹配可靠性，显著减少误检与漏检。实验表明其效果优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.04122v1](http://arxiv.org/pdf/2506.04122v1)**

> **作者:** Sharang Kaul; Mario Berk; Thiemo Gerbich; Abhinav Valada
>
> **摘要:** Finding reliable matches is essential in multi-object tracking to ensure the accuracy and reliability of perception systems in safety-critical applications such as autonomous vehicles. Effective matching mitigates perception errors, enhancing object identification and tracking for improved performance and safety. However, traditional metrics such as Intersection over Union (IoU) and Center Point Distances (CPDs), which are effective in 2D image planes, often fail to find critical matches in complex 3D scenes. To address this limitation, we introduce Contour Errors (CEs), an ego or object-centric metric for identifying matches of interest in tracking scenarios from a functional perspective. By comparing bounding boxes in the ego vehicle's frame, contour errors provide a more functionally relevant assessment of object matches. Extensive experiments on the nuScenes dataset demonstrate that contour errors improve the reliability of matches over the state-of-the-art 2D IoU and CPD metrics in tracking-by-detection methods. In 3D car tracking, our results show that Contour Errors reduce functional failures (FPs/FNs) by 80% at close ranges and 60% at far ranges compared to IoU in the evaluation stage.
>
---
#### [new 024] RAID: A Dataset for Testing the Adversarial Robustness of AI-Generated Image Detectors
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于AI生成图像检测任务，旨在解决检测器在对抗攻击下的鲁棒性问题。作者构建了RAID数据集，包含72k个对抗样本，用于评估检测方法的鲁棒性，发现当前方法易受攻击，强调需提升鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.03988v1](http://arxiv.org/pdf/2506.03988v1)**

> **作者:** Hicham Eddoubi; Jonas Ricker; Federico Cocchi; Lorenzo Baraldi; Angelo Sotgiu; Maura Pintor; Marcella Cornia; Lorenzo Baraldi; Asja Fischer; Rita Cucchiara; Battista Biggio
>
> **备注:** Under review for NeurIPS 2025 Datasets and Benchmarks Track
>
> **摘要:** AI-generated images have reached a quality level at which humans are incapable of reliably distinguishing them from real images. To counteract the inherent risk of fraud and disinformation, the detection of AI-generated images is a pressing challenge and an active research topic. While many of the presented methods claim to achieve high detection accuracy, they are usually evaluated under idealized conditions. In particular, the adversarial robustness is often neglected, potentially due to a lack of awareness or the substantial effort required to conduct a comprehensive robustness analysis. In this work, we tackle this problem by providing a simpler means to assess the robustness of AI-generated image detectors. We present RAID (Robust evaluation of AI-generated image Detectors), a dataset of 72k diverse and highly transferable adversarial examples. The dataset is created by running attacks against an ensemble of seven state-of-the-art detectors and images generated by four different text-to-image models. Extensive experiments show that our methodology generates adversarial images that transfer with a high success rate to unseen detectors, which can be used to quickly provide an approximate yet still reliable estimate of a detector's adversarial robustnessOur findings indicate that current state-of-the-art AI-generated image detectors can be easily deceived by adversarial examples, highlighting the critical need for the development of more robust methods. We release our dataset at https://huggingface.co/datasets/aimagelab/RAID and evaluation code at https://github.com/pralab/RAID.
>
---
#### [new 025] Vocabulary-free few-shot learning for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的少样本学习任务，旨在解决无预定义类别名称时的分类问题。提出了一种无需手动设计提示的相似性映射方法（SiM），通过通用提示进行分类，提升了模型在缺乏明确词汇场景下的适应能力。**

- **链接: [http://arxiv.org/pdf/2506.04005v1](http://arxiv.org/pdf/2506.04005v1)**

> **作者:** Maxime Zanella; Clément Fuchs; Ismail Ben Ayed; Christophe De Vleeschouwer
>
> **备注:** Accepted at CVPR Workshops 2025
>
> **摘要:** Recent advances in few-shot adaptation for Vision-Language Models (VLMs) have greatly expanded their ability to generalize across tasks using only a few labeled examples. However, existing approaches primarily build upon the strong zero-shot priors of these models by leveraging carefully designed, task-specific prompts. This dependence on predefined class names can restrict their applicability, especially in scenarios where exact class names are unavailable or difficult to specify. To address this limitation, we introduce vocabulary-free few-shot learning for VLMs, a setting where target class instances - that is, images - are available but their corresponding names are not. We propose Similarity Mapping (SiM), a simple yet effective baseline that classifies target instances solely based on similarity scores with a set of generic prompts (textual or visual), eliminating the need for carefully handcrafted prompts. Although conceptually straightforward, SiM demonstrates strong performance, operates with high computational efficiency (learning the mapping typically takes less than one second), and provides interpretability by linking target classes to generic prompts. We believe that our approach could serve as an important baseline for future research in vocabulary-free few-shot learning. Code is available at https://github.com/MaxZanella/vocabulary-free-FSL.
>
---
#### [new 026] Video, How Do Your Tokens Merge?
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决视频Transformer模型计算资源消耗大的问题。通过探索无需训练的视频Token合并方法，在多个视频Transformer和数据集上进行实验，验证其在保持精度的同时提升推理速度的效果。**

- **链接: [http://arxiv.org/pdf/2506.03885v1](http://arxiv.org/pdf/2506.03885v1)**

> **作者:** Sam Pollard; Michael Wray
>
> **备注:** Accepted at eLVM workshop at CVPR 2025
>
> **摘要:** Video transformer models require huge amounts of compute resources due to the spatio-temporal scaling of the input. Tackling this, recent methods have proposed to drop or merge tokens for image models, whether randomly or via learned methods. Merging tokens has many benefits: it can be plugged into any vision transformer, does not require model re-training, and it propagates information that would otherwise be dropped through the model. Before now, video token merging has not been evaluated on temporally complex datasets for video understanding. In this work, we explore training-free token merging for video to provide comprehensive experiments and find best practices across four video transformers on three datasets that exhibit coarse and fine-grained action recognition. Our results showcase the benefits of video token merging with a speedup of around $2.5$X while maintaining accuracy (avg. $-0.55\%$ for ViViT). Code available at https://github.com/sjpollard/video-how-do-your-tokens-merge.
>
---
#### [new 027] ComRoPE: Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于自然语言处理任务中的位置编码研究。旨在解决传统RoPE方法中旋转矩阵手动定义、变换空间受限的问题。作者提出ComRoPE，通过可训练的交换角度矩阵实现更灵活和鲁棒的位置编码，提高了模型性能，并在ImageNet-1K数据集上取得了更好的效果。**

- **链接: [http://arxiv.org/pdf/2506.03737v1](http://arxiv.org/pdf/2506.03737v1)**

> **作者:** Hao Yu; Tangyu Jiang; Shuning Jia; Shannan Yan; Shunning Liu; Haolong Qian; Guanghao Li; Shuting Dong; Huaisong Zhang; Chun Yuan
>
> **摘要:** The Transformer architecture has revolutionized various regions since it was proposed, and its effectiveness largely depends on the ability to encode positional information. Traditional position encoding methods exhibit significant limitations due to lack of robustness and flexibility of position. Therefore, Rotary Positional Encoding (RoPE) was proposed to alleviate these issues, which integrates positional information by rotating the embeddings in the attention mechanism. However, RoPE requires manually defined rotation matrices with limited transformation space, constraining the model's capacity. In this work, we propose ComRoPE, which generalizes RoPE by defining it in terms of trainable commuting angle matrices. Specifically, we demonstrate that pairwise commutativity of these matrices is essential for RoPE to achieve scalability and positional robustness. We formally define the RoPE Equation, which is an essential condition that ensures consistent performance with position offsets. Based on the theoretical analysis, we present two types of trainable commuting angle matrices as sufficient solutions to the RoPE equation, which significantly improve performance, surpassing the current state-of-the-art method by 1.6% at training resolution and 2.9% at higher resolution on the ImageNet-1K dataset. Furthermore, our framework shows versatility in generalizing to existing RoPE formulations and offering new insights for future positional encoding research. To ensure reproducibility, the source code and instructions are available at https://github.com/Longin-Yu/ComRoPE
>
---
#### [new 028] Advancements in Artificial Intelligence Applications for Cardiovascular Disease Research
- **分类: cs.CV**

- **简介: 该论文综述了人工智能在心血管疾病研究中的应用进展，属于医学与AI交叉领域的任务。旨在解决心血管疾病诊断准确性与效率问题，重点分析了深度学习在CT、MRI、ECG和超声中的应用，并指出数据验证不足的挑战。工作包括总结现有模型优势，强调开发多模态融合与自适应算法对未来精准医疗的重要性。**

- **链接: [http://arxiv.org/pdf/2506.03698v1](http://arxiv.org/pdf/2506.03698v1)**

> **作者:** Yuanlin Mo; Haishan Huang; Bocheng Liang; Weibo Ma
>
> **摘要:** Recent advancements in artificial intelligence (AI) have revolutionized cardiovascular medicine, particularly through integration with computed tomography (CT), magnetic resonance imaging (MRI), electrocardiography (ECG) and ultrasound (US). Deep learning architectures, including convolutional neural networks and generative adversarial networks, enable automated analysis of medical imaging and physiological signals, surpassing human capabilities in diagnostic accuracy and workflow efficiency. However, critical challenges persist, including the inability to validate input data accuracy, which may propagate diagnostic errors. This review highlights AI's transformative potential in precision diagnostics while underscoring the need for robust validation protocols to ensure clinical reliability. Future directions emphasize hybrid models integrating multimodal data and adaptive algorithms to refine personalized cardiovascular care.
>
---
#### [new 029] MS-YOLO: A Multi-Scale Model for Accurate and Efficient Blood Cell Detection
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决血液细胞检测中的重叠细胞与多尺度目标识别问题。作者提出了MS-YOLO模型，在YOLOv11基础上引入三种模块（MS-DRM、DCFEM、LADS），提升检测精度与效率。实验表明其在CBC和WBCDD数据集上表现优异，尤其对小目标如血小板检测效果突出，具备临床应用潜力。**

- **链接: [http://arxiv.org/pdf/2506.03972v1](http://arxiv.org/pdf/2506.03972v1)**

> **作者:** Guohua Wu; Shengqi Chen; Pengchao Deng; Wenting Yu
>
> **摘要:** Complete blood cell detection holds significant value in clinical diagnostics. Conventional manual microscopy methods suffer from time inefficiency and diagnostic inaccuracies. Existing automated detection approaches remain constrained by high deployment costs and suboptimal accuracy. While deep learning has introduced powerful paradigms to this field, persistent challenges in detecting overlapping cells and multi-scale objects hinder practical deployment. This study proposes the multi-scale YOLO (MS-YOLO), a blood cell detection model based on the YOLOv11 framework, incorporating three key architectural innovations to enhance detection performance. Specifically, the multi-scale dilated residual module (MS-DRM) replaces the original C3K2 modules to improve multi-scale discriminability; the dynamic cross-path feature enhancement module (DCFEM) enables the fusion of hierarchical features from the backbone with aggregated features from the neck to enhance feature representations; and the light adaptive-weight downsampling module (LADS) improves feature downsampling through adaptive spatial weighting while reducing computational complexity. Experimental results on the CBC benchmark demonstrate that MS-YOLO achieves precise detection of overlapping cells and multi-scale objects, particularly small targets such as platelets, achieving an mAP@50 of 97.4% that outperforms existing models. Further validation on the supplementary WBCDD dataset confirms its robust generalization capability. Additionally, with a lightweight architecture and real-time inference efficiency, MS-YOLO meets clinical deployment requirements, providing reliable technical support for standardized blood pathology assessment.
>
---
#### [new 030] ViTSGMM: A Robust Semi-Supervised Image Recognition Network Using Sparse Labels
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像识别任务，旨在解决标签数据极少情况下的半监督学习问题。作者提出ViTSGMM网络，通过优化特征与类别间的互信息，构建分层混合密度分类机制，在保留关键判别信息的同时压缩冗余。实验表明其在多个数据集上表现优异，并修正了STL-10数据集中的数据泄漏问题。**

- **链接: [http://arxiv.org/pdf/2506.03582v1](http://arxiv.org/pdf/2506.03582v1)**

> **作者:** Rui Yann; Xianglei Xing
>
> **摘要:** We present ViTSGMM, an image recognition network that leverages semi-supervised learning in a highly efficient manner. Existing works often rely on complex training techniques and architectures, while their generalization ability when dealing with extremely limited labeled data remains to be improved. To address these limitations, we construct a hierarchical mixture density classification decision mechanism by optimizing mutual information between feature representations and target classes, compressing redundant information while retaining crucial discriminative components. Experimental results demonstrate that our method achieves state-of-the-art performance on STL-10 and CIFAR-10/100 datasets when using negligible labeled samples. Notably, this paper also reveals a long-overlooked data leakage issue in the STL-10 dataset for semi-supervised learning tasks and removes duplicates to ensure the reliability of experimental results. Code available at https://github.com/Shu1L0n9/ViTSGMM.
>
---
#### [new 031] TerraIncognita: A Dynamic Benchmark for Species Discovery Using Frontier Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出了TerraIncognita，一个用于物种发现的动态基准，旨在评估前沿AI模型在识别未知昆虫物种方面的能力。任务是多模态图像分类与新类别检测，解决昆虫物种鉴定效率低、依赖专家的问题。工作包括构建包含已知与未知物种图像的基准数据集，并评估模型在层级分类、新物种识别及解释生成方面的能力。**

- **链接: [http://arxiv.org/pdf/2506.03182v1](http://arxiv.org/pdf/2506.03182v1)**

> **作者:** Shivani Chiranjeevi; Hossein Zaremehrjerdi; Zi K. Deng; Talukder Z. Jubery; Ari Grele; Arti Singh; Asheesh K Singh; Soumik Sarkar; Nirav Merchant; Harold F. Greeney; Baskar Ganapathysubramanian; Chinmay Hegde
>
> **摘要:** The rapid global loss of biodiversity, particularly among insects, represents an urgent ecological crisis. Current methods for insect species discovery are manual, slow, and severely constrained by taxonomic expertise, hindering timely conservation actions. We introduce TerraIncognita, a dynamic benchmark designed to evaluate state-of-the-art multimodal models for the challenging problem of identifying unknown, potentially undescribed insect species from image data. Our benchmark dataset combines a mix of expertly annotated images of insect species likely known to frontier AI models, and images of rare and poorly known species, for which few/no publicly available images exist. These images were collected from underexplored biodiversity hotspots, realistically mimicking open-world discovery scenarios faced by ecologists. The benchmark assesses models' proficiency in hierarchical taxonomic classification, their capability to detect and abstain from out-of-distribution (OOD) samples representing novel species, and their ability to generate explanations aligned with expert taxonomic knowledge. Notably, top-performing models achieve over 90\% F1 at the Order level on known species, but drop below 2\% at the Species level, highlighting the sharp difficulty gradient from coarse to fine taxonomic prediction (Order $\rightarrow$ Family $\rightarrow$ Genus $\rightarrow$ Species). TerraIncognita will be updated regularly, and by committing to quarterly dataset expansions (of both known and novel species), will provide an evolving platform for longitudinal benchmarking of frontier AI methods. All TerraIncognita data, results, and future updates are available \href{https://baskargroup.github.io/TerraIncognita/}{here}.
>
---
#### [new 032] Images are Worth Variable Length of Representations
- **分类: cs.CV**

- **简介: 该论文属于视觉编码任务，旨在解决固定长度编码效率低的问题。现有方法用相同数量的token表示所有图像，忽视了不同图像的信息差异。作者提出DOVE动态编码器，根据图像复杂度生成可变数量的token，在减少token数的同时保持高质量重建，并提升语义表达能力。**

- **链接: [http://arxiv.org/pdf/2506.03643v1](http://arxiv.org/pdf/2506.03643v1)**

> **作者:** Lingjun Mao; Rodolfo Corona; Xin Liang; Wenhao Yan; Zineng Tang
>
> **摘要:** Most existing vision encoders map images into a fixed-length sequence of tokens, overlooking the fact that different images contain varying amounts of information. For example, a visually complex image (e.g., a cluttered room) inherently carries more information and thus deserves more tokens than a simple image (e.g., a blank wall). To address this inefficiency, we propose DOVE, a dynamic vision encoder that produces a variable number of visual tokens (i.e., continuous representation vectors) to reconstruct each image. Our results show that DOVE significantly reduces the average number of tokens while maintaining high reconstruction quality. In several linear probing and downstream multimodal tasks, it outperforms existing autoencoder-based tokenization methods when using far fewer tokens, capturing more expressive semantic features compared to fixed-length encoding. We further extend DOVE with query-conditioned tokenization. By guiding the model to focus on query-relevant regions, it achieves more efficient and targeted semantic extraction. Our code and checkpoints are available at https://dove-encoder.github.io/dove-encoder.
>
---
#### [new 033] Generating 6DoF Object Manipulation Trajectories from Action Description in Egocentric Vision
- **分类: cs.CV**

- **简介: 该论文属于机器人操作任务，旨在根据第一视角视觉中的动作描述生成6自由度物体操作轨迹。为解决缺乏大规模多样本操作数据的问题，作者利用Exo-Ego4D视频数据提取操作轨迹，并基于视觉与点云语言模型生成轨迹，在HOT3D数据集上验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2506.03605v1](http://arxiv.org/pdf/2506.03605v1)**

> **作者:** Tomoya Yoshida; Shuhei Kurita; Taichi Nishimura; Shinsuke Mori
>
> **备注:** CVPR 2025
>
> **摘要:** Learning to use tools or objects in common scenes, particularly handling them in various ways as instructed, is a key challenge for developing interactive robots. Training models to generate such manipulation trajectories requires a large and diverse collection of detailed manipulation demonstrations for various objects, which is nearly unfeasible to gather at scale. In this paper, we propose a framework that leverages large-scale ego- and exo-centric video datasets -- constructed globally with substantial effort -- of Exo-Ego4D to extract diverse manipulation trajectories at scale. From these extracted trajectories with the associated textual action description, we develop trajectory generation models based on visual and point cloud-based language models. In the recently proposed egocentric vision-based in-a-quality trajectory dataset of HOT3D, we confirmed that our models successfully generate valid object trajectories, establishing a training dataset and baseline models for the novel task of generating 6DoF manipulation trajectories from action descriptions in egocentric vision.
>
---
#### [new 034] OSGNet @ Ego4D Episodic Memory Challenge 2025
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频定位任务，旨在解决第一视角视频中精确时间区间定位问题。针对现有方法依赖后期融合导致效果不佳的情况，作者提出基于早期融合的模型OSGNet，并在三项挑战赛中取得第一名，验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2506.03710v1](http://arxiv.org/pdf/2506.03710v1)**

> **作者:** Yisen Feng; Haoyu Zhang; Qiaohui Chu; Meng Liu; Weili Guan; Yaowei Wang; Liqiang Nie
>
> **备注:** The champion solutions for the three egocentric video localization tracks(Natural Language Queries, Goal Step, and Moment Queries tracks) of the Ego4D Episodic Memory Challenge at CVPR EgoVis Workshop 2025
>
> **摘要:** In this report, we present our champion solutions for the three egocentric video localization tracks of the Ego4D Episodic Memory Challenge at CVPR 2025. All tracks require precise localization of the interval within an untrimmed egocentric video. Previous unified video localization approaches often rely on late fusion strategies, which tend to yield suboptimal results. To address this, we adopt an early fusion-based video localization model to tackle all three tasks, aiming to enhance localization accuracy. Ultimately, our method achieved first place in the Natural Language Queries, Goal Step, and Moment Queries tracks, demonstrating its effectiveness. Our code can be found at https://github.com/Yisen-Feng/OSGNet.
>
---
#### [new 035] How PARTs assemble into wholes: Learning the relative composition of images
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于自监督学习任务，旨在解决现有基于网格的预训练方法在建模对象及其部分间连续关系上的局限性。作者提出PART方法，通过学习图像块之间的连续相对变换关系，实现对图像结构的更灵活建模，提升了在物体检测、时间序列预测等任务上的性能。**

- **链接: [http://arxiv.org/pdf/2506.03682v1](http://arxiv.org/pdf/2506.03682v1)**

> **作者:** Melika Ayoughi; Samira Abnar; Chen Huang; Chris Sandino; Sayeri Lala; Eeshan Gunesh Dhekane; Dan Busbridge; Shuangfei Zhai; Vimal Thilak; Josh Susskind; Pascal Mettes; Paul Groth; Hanlin Goh
>
> **摘要:** The composition of objects and their parts, along with object-object positional relationships, provides a rich source of information for representation learning. Hence, spatial-aware pretext tasks have been actively explored in self-supervised learning. Existing works commonly start from a grid structure, where the goal of the pretext task involves predicting the absolute position index of patches within a fixed grid. However, grid-based approaches fall short of capturing the fluid and continuous nature of real-world object compositions. We introduce PART, a self-supervised learning approach that leverages continuous relative transformations between off-grid patches to overcome these limitations. By modeling how parts relate to each other in a continuous space, PART learns the relative composition of images-an off-grid structural relative positioning process that generalizes beyond occlusions and deformations. In tasks requiring precise spatial understanding such as object detection and time series prediction, PART outperforms strong grid-based methods like MAE and DropPos, while also maintaining competitive performance on global classification tasks with minimal hyperparameter tuning. By breaking free from grid constraints, PART opens up an exciting new trajectory for universal self-supervised pretraining across diverse datatypes-from natural images to EEG signals-with promising potential in video, medical imaging, and audio.
>
---
#### [new 036] PALADIN : Robust Neural Fingerprinting for Text-to-Image Diffusion Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于文本到图像扩散模型的神经指纹识别任务，旨在解决模型滥用的可追溯性问题。现有方法在归因准确性和生成质量间存在权衡，但未达到100%准确率，实用性受限。本文提出基于循环纠错码的新方法，实现高准确性的神经指纹嵌入，提升模型部署可行性。**

- **链接: [http://arxiv.org/pdf/2506.03170v1](http://arxiv.org/pdf/2506.03170v1)**

> **作者:** Murthy L; Subarna Tripathi
>
> **摘要:** The risk of misusing text-to-image generative models for malicious uses, especially due to the open-source development of such models, has become a serious concern. As a risk mitigation strategy, attributing generative models with neural fingerprinting is emerging as a popular technique. There has been a plethora of recent work that aim for addressing neural fingerprinting. A trade-off between the attribution accuracy and generation quality of such models has been studied extensively. None of the existing methods yet achieved $100\%$ attribution accuracy. However, any model with less than \emph{perfect} accuracy is practically non-deployable. In this work, we propose an accurate method to incorporate neural fingerprinting for text-to-image diffusion models leveraging the concepts of cyclic error correcting codes from the literature of coding theory.
>
---
#### [new 037] Diffusion Domain Teacher: Diffusion Guided Domain Adaptive Object Detector
- **分类: cs.CV**

- **简介: 论文提出了一种名为Diffusion Domain Teacher（DDT）的领域自适应目标检测方法，旨在解决训练数据（源域）与真实世界数据（目标域）之间存在较大领域差异导致检测性能下降的问题。通过利用扩散模型提取跨领域特征表示，构建教师模型为目标域生成伪标签，从而指导学生模型的学习过程。实验表明，该方法在多个数据集上显著提升了跨领域目标检测性能，并具有广泛的适用性和有效性。**

- **链接: [http://arxiv.org/pdf/2506.04211v1](http://arxiv.org/pdf/2506.04211v1)**

> **作者:** Boyong He; Yuxiang Ji; Zhuoyue Tan; Liaoni Wu
>
> **备注:** MM2024 poster, with appendix and codes
>
> **摘要:** Object detectors often suffer a decrease in performance due to the large domain gap between the training data (source domain) and real-world data (target domain). Diffusion-based generative models have shown remarkable abilities in generating high-quality and diverse images, suggesting their potential for extracting valuable feature from various domains. To effectively leverage the cross-domain feature representation of diffusion models, in this paper, we train a detector with frozen-weight diffusion model on the source domain, then employ it as a teacher model to generate pseudo labels on the unlabeled target domain, which are used to guide the supervised learning of the student model on the target domain. We refer to this approach as Diffusion Domain Teacher (DDT). By employing this straightforward yet potent framework, we significantly improve cross-domain object detection performance without compromising the inference speed. Our method achieves an average mAP improvement of 21.2% compared to the baseline on 6 datasets from three common cross-domain detection benchmarks (Cross-Camera, Syn2Real, Real2Artistic}, surpassing the current state-of-the-art (SOTA) methods by an average of 5.7% mAP. Furthermore, extensive experiments demonstrate that our method consistently brings improvements even in more powerful and complex models, highlighting broadly applicable and effective domain adaptation capability of our DDT. The code is available at https://github.com/heboyong/Diffusion-Domain-Teacher.
>
---
#### [new 038] FOLIAGE: Towards Physical Intelligence World Models Via Unbounded Surface Evolution
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出FOLIAGE，一种基于物理智能的多模态世界模型，用于无限制表面增长建模。论文属于世界模型与物理智能任务，旨在从部分多感官观测中预测并塑造物理世界。工作包括构建统一上下文编码器、物理感知预测器及AGN网络，并在SURF-GARDEN平台上验证其在SURF-BENCH任务中的性能，展现其在动态环境中的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.03173v1](http://arxiv.org/pdf/2506.03173v1)**

> **作者:** Xiaoyi Liu; Hao Tang
>
> **摘要:** Physical intelligence -- anticipating and shaping the world from partial, multisensory observations -- is critical for next-generation world models. We propose FOLIAGE, a physics-informed multimodal world model for unbounded accretive surface growth. In its Action-Perception loop, a unified context encoder maps images, mesh connectivity, and point clouds to a shared latent state. A physics-aware predictor, conditioned on physical control actions, advances this latent state in time to align with the target latent of the surface, yielding a Modality-Agnostic Growth Embedding (MAGE) that interfaces with critic heads for downstream objectives. FOLIAGE's Accretive Graph Network (AGN) captures dynamic connectivity through Age Positional Encoding and Energy-Gated Message-Passing. Geometry-Correspondence Fusion and Cross-Patch Masking enhance MAGE's expressiveness, while Hierarchical Pooling balances global context with local dynamics. We create SURF-GARDEN, a world model learning platform comprising a Counterfactual Physics Simulator, a Multimodal Correspondence Extractor, and Evolution Tracing, which generates 7,200 diverse surface-growth sequences. SURF-BENCH, our physical-intelligence evaluation suite, evaluates six core tasks -- topology recognition, inverse material estimation, growth-stage classification, latent roll-out, cross-modal retrieval, and dense correspondence -- and four stress tests -- sensor dropout, zero-shot modality transfer, long-horizon prediction, and physics ablation -- to probe resilience. FOLIAGE outperforms specialized baselines while remaining robust across dynamic environments, establishing a new world-model based, multimodal pathway to physical intelligence.
>
---
#### [new 039] Mitigating Hallucinations in Large Vision-Language Models via Entity-Centric Multimodal Preference Optimization
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉-语言模型任务，旨在解决大模型在跨模态理解中出现的幻觉问题。作者提出了一种实体中心的多模态偏好优化方法（EMPO），通过提升图像与文本的对齐能力，并利用开源数据构建高质量偏好数据集，有效减少了模型的幻觉现象。**

- **链接: [http://arxiv.org/pdf/2506.04039v1](http://arxiv.org/pdf/2506.04039v1)**

> **作者:** Jiulong Wu; Zhengliang Shi; Shuaiqiang Wang; Jizhou Huang; Dawei Yin; Lingyong Yan; Min Cao; Min Zhang
>
> **摘要:** Large Visual Language Models (LVLMs) have demonstrated impressive capabilities across multiple tasks. However, their trustworthiness is often challenged by hallucinations, which can be attributed to the modality misalignment and the inherent hallucinations of their underlying Large Language Models (LLMs) backbone. Existing preference alignment methods focus on aligning model responses with human preferences while neglecting image-text modality alignment, resulting in over-reliance on LLMs and hallucinations. In this paper, we propose Entity-centric Multimodal Preference Optimization (EMPO), which achieves enhanced modality alignment than existing human preference alignment methods. Besides, to overcome the scarcity of high-quality multimodal preference data, we utilize open-source instruction datasets to automatically construct high-quality preference data across three aspects: image, instruction, and response. Experiments on two human preference datasets and five multimodal hallucination benchmarks demonstrate the effectiveness of EMPO, e.g., reducing hallucination rates by 85.9% on Object-HalBench and 49.8% on MM-HalBench.
>
---
#### [new 040] Multiple Stochastic Prompt Tuning for Practical Cross-Domain Few Shot Learning
- **分类: cs.CV**

- **简介: 该论文属于实用跨域少样本学习（pCDFSL）任务，旨在解决在极端领域迁移下，仅用少量标签样本同时分类所有未见类的问题。作者提出MIST框架，通过多随机提示调优，学习每个类别的多个提示，并以高斯分布建模参数，提升模型泛化能力，缓解过拟合。**

- **链接: [http://arxiv.org/pdf/2506.03926v1](http://arxiv.org/pdf/2506.03926v1)**

> **作者:** Debarshi Brahma; Soma Biswas
>
> **摘要:** In this work, we propose a practical cross-domain few-shot learning (pCDFSL) task, where a large-scale pre-trained model like CLIP can be easily deployed on a target dataset. The goal is to simultaneously classify all unseen classes under extreme domain shifts, by utilizing only a few labeled samples per class. The pCDFSL paradigm is source-free and moves beyond artificially created episodic training and testing regimes followed by existing CDFSL frameworks, making it more challenging and relevant to real-world applications. Towards that goal, we propose a novel framework, termed MIST (MultIple STochastic Prompt tuning), where multiple stochastic prompts are utilized to handle significant domain and semantic shifts. Specifically, multiple prompts are learnt for each class, effectively capturing multiple peaks in the input data. Furthermore, instead of representing the weights of the multiple prompts as point-estimates, we model them as learnable Gaussian distributions with two different strategies, encouraging an efficient exploration of the prompt parameter space, which mitigate overfitting due to the few labeled training samples. Extensive experiments and comparison with the state-of-the-art methods on four CDFSL benchmarks adapted to this setting, show the effectiveness of the proposed framework.
>
---
#### [new 041] PDSE: A Multiple Lesion Detector for CT Images using PANet and Deformable Squeeze-and-Excitation Block
- **分类: cs.CV**

- **简介: 该论文属于医学图像处理中的病变检测任务，旨在解决CT图像中多类型、多尺度病灶检测难题。作者提出PDSE框架，改进Retinanet结构，通过增强特征融合与注意力机制，提升检测精度与效率，尤其改善小病灶和多尺度病灶的检测效果。**

- **链接: [http://arxiv.org/pdf/2506.03608v1](http://arxiv.org/pdf/2506.03608v1)**

> **作者:** Di Fan; Heng Yu; Zhiyuan Xu
>
> **备注:** MIUA 2024
>
> **摘要:** Detecting lesions in Computed Tomography (CT) scans is a challenging task in medical image processing due to the diverse types, sizes, and locations of lesions. Recently, various one-stage and two-stage framework networks have been developed to focus on lesion localization. We introduce a one-stage lesion detection framework, PDSE, by redesigning Retinanet to achieve higher accuracy and efficiency for detecting lesions in multimodal CT images. Specifically, we enhance the path aggregation flow by incorporating a low-level feature map. Additionally, to improve model representation, we utilize the adaptive Squeeze-and-Excitation (SE) block and integrate channel feature map attention. This approach has resulted in achieving new state-of-the-art performance. Our method significantly improves the detection of small and multiscaled objects. When evaluated against other advanced algorithms on the public DeepLesion benchmark, our algorithm achieved an mAP of over 0.20.
>
---
#### [new 042] FingerVeinSyn-5M: A Million-Scale Dataset and Benchmark for Finger Vein Recognition
- **分类: cs.CV**

- **简介: 该论文属于生物特征识别任务，旨在解决手指静脉识别中缺乏大规模公开数据集的问题。作者提出了FVeinSyn合成生成器，并构建了包含500万样本的FingerVeinSyn-5M数据集，支持深度学习模型训练与研究。**

- **链接: [http://arxiv.org/pdf/2506.03635v1](http://arxiv.org/pdf/2506.03635v1)**

> **作者:** Yinfan Wang; Jie Gui; Baosheng Yu; Qi Li; Zhenan Sun; Juho Kannala; Guoying Zhao
>
> **摘要:** A major challenge in finger vein recognition is the lack of large-scale public datasets. Existing datasets contain few identities and limited samples per finger, restricting the advancement of deep learning-based methods. To address this, we introduce FVeinSyn, a synthetic generator capable of producing diverse finger vein patterns with rich intra-class variations. Using FVeinSyn, we created FingerVeinSyn-5M -- the largest available finger vein dataset -- containing 5 million samples from 50,000 unique fingers, each with 100 variations including shift, rotation, scale, roll, varying exposure levels, skin scattering blur, optical blur, and motion blur. FingerVeinSyn-5M is also the first to offer fully annotated finger vein images, supporting deep learning applications in this field. Models pretrained on FingerVeinSyn-5M and fine-tuned with minimal real data achieve an average 53.91\% performance gain across multiple benchmarks. The dataset is publicly available at: https://github.com/EvanWang98/FingerVeinSyn-5M.
>
---
#### [new 043] Learning Optical Flow Field via Neural Ordinary Differential Equation
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务中的光流估计问题。现有方法使用固定步数的循环网络进行优化，可能不适应不同输入。论文提出用神经常微分方程（ODE）建模光流，动态调整计算步数，实现更高效的优化。实验表明该方法性能优越，仅需一个优化步骤。**

- **链接: [http://arxiv.org/pdf/2506.03290v1](http://arxiv.org/pdf/2506.03290v1)**

> **作者:** Leyla Mirvakhabova; Hong Cai; Jisoo Jeong; Hanno Ackermann; Farhad Zanjani; Fatih Porikli
>
> **备注:** CVPRW 2025
>
> **摘要:** Recent works on optical flow estimation use neural networks to predict the flow field that maps positions of one image to positions of the other. These networks consist of a feature extractor, a correlation volume, and finally several refinement steps. These refinement steps mimic the iterative refinements performed by classical optimization algorithms and are usually implemented by neural layers (e.g., GRU) which are recurrently executed for a fixed and pre-determined number of steps. However, relying on a fixed number of steps may result in suboptimal performance because it is not tailored to the input data. In this paper, we introduce a novel approach for predicting the derivative of the flow using a continuous model, namely neural ordinary differential equations (ODE). One key advantage of this approach is its capacity to model an equilibrium process, dynamically adjusting the number of compute steps based on the data at hand. By following a particular neural architecture, ODE solver, and associated hyperparameters, our proposed model can replicate the exact same updates as recurrent cells used in existing works, offering greater generality. Through extensive experimental analysis on optical flow benchmarks, we demonstrate that our approach achieves an impressive improvement over baseline and existing models, all while requiring only a single refinement step.
>
---
#### [new 044] Farm-LightSeek: An Edge-centric Multimodal Agricultural IoT Data Analytics Framework with Lightweight LLMs
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于农业物联网与人工智能交叉任务，旨在解决传统农业物联网系统在多模态数据融合、实时决策及环境适应性方面的不足。论文提出Farm-LightSeek框架，结合边缘计算与轻量级大语言模型，实现农田多源数据的实时分析与智能决策，并通过云协作优化模型更新。**

- **链接: [http://arxiv.org/pdf/2506.03168v1](http://arxiv.org/pdf/2506.03168v1)**

> **作者:** Dawen Jiang; Zhishu Shen; Qiushi Zheng; Tiehua Zhang; Wei Xiang; Jiong Jin
>
> **备注:** Accepted by IEEE Internet of Things Magazine
>
> **摘要:** Amid the challenges posed by global population growth and climate change, traditional agricultural Internet of Things (IoT) systems is currently undergoing a significant digital transformation to facilitate efficient big data processing. While smart agriculture utilizes artificial intelligence (AI) technologies to enable precise control, it still encounters significant challenges, including excessive reliance on agricultural expert knowledge, difficulties in fusing multimodal data, poor adaptability to dynamic environments, and bottlenecks in real-time decision-making at the edge. Large language models (LLMs), with their exceptional capabilities in knowledge acquisition and semantic understanding, provide a promising solution to address these challenges. To this end, we propose Farm-LightSeek, an edge-centric multimodal agricultural IoT data analytics framework that integrates LLMs with edge computing. This framework collects real-time farmland multi-source data (images, weather, geographic information) via sensors, performs cross-modal reasoning and disease detection at edge nodes, conducts low-latency management decisions, and enables cloud collaboration for model updates. The main innovations of Farm-LightSeek include: (1) an agricultural "perception-decision-action" closed-loop architecture; (2) cross-modal adaptive monitoring; and (3)a lightweight LLM deployment strategy balancing performance and efficiency. Experiments conducted on two real-world datasets demonstrate that Farm-LightSeek consistently achieves reliable performance in mission-critical tasks, even under the limitations of edge computing resources. This work advances intelligent real-time agricultural solutions and highlights the potential for deeper integration of agricultural IoT with LLMs.
>
---
#### [new 045] FSHNet: Fully Sparse Hybrid Network for 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文属于3D目标检测任务，旨在解决全稀疏网络在长距离交互和特征缺失的问题。作者提出了FSHNet，包含SlotFormer模块增强特征提取、动态标签分配策略优化训练，以及稀疏上采样模块提升小物体检测效果。**

- **链接: [http://arxiv.org/pdf/2506.03714v1](http://arxiv.org/pdf/2506.03714v1)**

> **作者:** Shuai Liu; Mingyue Cui; Boyang Li; Quanmin Liang; Tinghe Hong; Kai Huang; Yunxiao Shan; Kai Huang
>
> **备注:** Accepted by CVPR2025
>
> **摘要:** Fully sparse 3D detectors have recently gained significant attention due to their efficiency in long-range detection. However, sparse 3D detectors extract features only from non-empty voxels, which impairs long-range interactions and causes the center feature missing. The former weakens the feature extraction capability, while the latter hinders network optimization. To address these challenges, we introduce the Fully Sparse Hybrid Network (FSHNet). FSHNet incorporates a proposed SlotFormer block to enhance the long-range feature extraction capability of existing sparse encoders. The SlotFormer divides sparse voxels using a slot partition approach, which, compared to traditional window partition, provides a larger receptive field. Additionally, we propose a dynamic sparse label assignment strategy to deeply optimize the network by providing more high-quality positive samples. To further enhance performance, we introduce a sparse upsampling module to refine downsampled voxels, preserving fine-grained details crucial for detecting small objects. Extensive experiments on the Waymo, nuScenes, and Argoverse2 benchmarks demonstrate the effectiveness of FSHNet. The code is available at https://github.com/Say2L/FSHNet.
>
---
#### [new 046] Negative-Guided Subject Fidelity Optimization for Zero-Shot Subject-Driven Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于零样本主体驱动生成任务，旨在提升生成图像与指定主体的保真度。方法通过引入合成负样本并进行对比学习优化模型，提出CDNS算法自动生成负样本，并调整扩散步骤权重以聚焦关键阶段，最终在基准数据集上显著优于基线方法。**

- **链接: [http://arxiv.org/pdf/2506.03621v1](http://arxiv.org/pdf/2506.03621v1)**

> **作者:** Chaehun Shin; Jooyoung Choi; Johan Barthelemy; Jungbeom Lee; Sungroh Yoon
>
> **摘要:** We present Subject Fidelity Optimization (SFO), a novel comparative learning framework for zero-shot subject-driven generation that enhances subject fidelity. Beyond supervised fine-tuning methods that rely only on positive targets and use the diffusion loss as in the pre-training stage, SFO introduces synthetic negative targets and explicitly guides the model to favor positives over negatives through pairwise comparison. For negative targets, we propose Condition-Degradation Negative Sampling (CDNS), which automatically generates distinctive and informative negatives by intentionally degrading visual and textual cues without expensive human annotations. Moreover, we reweight the diffusion timesteps to focus finetuning on intermediate steps where subject details emerge. Extensive experiments demonstrate that SFO with CDNS significantly outperforms baselines in terms of both subject fidelity and text alignment on a subject-driven generation benchmark. Project page: https://subjectfidelityoptimization.github.io/
>
---
#### [new 047] FlexGS: Train Once, Deploy Everywhere with Many-in-One Flexible 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D场景表示与渲染任务，旨在解决3D高斯泼溅（3DGS）模型在不同设备上部署时内存占用过高的问题。作者提出FlexGS，一种弹性推理方法，通过可学习模块选择和调整高斯分布，实现无需微调的模型压缩与适应性部署，兼顾性能与内存限制。**

- **链接: [http://arxiv.org/pdf/2506.04174v1](http://arxiv.org/pdf/2506.04174v1)**

> **作者:** Hengyu Liu; Yuehao Wang; Chenxin Li; Ruisi Cai; Kevin Wang; Wuyang Li; Pavlo Molchanov; Peihao Wang; Zhangyang Wang
>
> **备注:** CVPR 2025; Project Page: https://flexgs.github.io
>
> **摘要:** 3D Gaussian splatting (3DGS) has enabled various applications in 3D scene representation and novel view synthesis due to its efficient rendering capabilities. However, 3DGS demands relatively significant GPU memory, limiting its use on devices with restricted computational resources. Previous approaches have focused on pruning less important Gaussians, effectively compressing 3DGS but often requiring a fine-tuning stage and lacking adaptability for the specific memory needs of different devices. In this work, we present an elastic inference method for 3DGS. Given an input for the desired model size, our method selects and transforms a subset of Gaussians, achieving substantial rendering performance without additional fine-tuning. We introduce a tiny learnable module that controls Gaussian selection based on the input percentage, along with a transformation module that adjusts the selected Gaussians to complement the performance of the reduced model. Comprehensive experiments on ZipNeRF, MipNeRF and Tanks\&Temples scenes demonstrate the effectiveness of our approach. Code is available at https://flexgs.github.io.
>
---
#### [new 048] Resolving Task Objective Conflicts in Unified Multimodal Understanding and Generation via Task-Aware Mixture-of-Experts
- **分类: cs.CV**

- **简介: 该论文属于多模态任务，旨在解决统一多模态大语言模型中理解与生成任务的目标冲突问题。作者提出UTAMoE框架，通过任务感知的专家混合层解耦自回归模型内部组件，并设计两阶段训练策略以提升任务协调性，实验证明其性能优越。**

- **链接: [http://arxiv.org/pdf/2506.03591v1](http://arxiv.org/pdf/2506.03591v1)**

> **作者:** Jiaxing Zhang; Xinyi Zeng; Hao Tang
>
> **摘要:** Unified multimodal large language models (MLLMs) based on end-to-end autoregressive (AR) transformers effectively integrate both understanding and generation tasks within a single framework. However, intrinsic Task Objective Conflicts between high-level semantic abstraction in understanding and fine-grained detail preservation in generation pose significant challenges, often leading to suboptimal trade-offs and task interference. Existing solutions, such as decoupling shared visual encoders, fall short of fundamentally resolving these conflicts due to inherent AR architecture. In this paper, we propose a novel approach that decouples internal components of AR to resolve task objective conflicts. Specifically, we design UTAMoE, a Unified Task-Aware Mixture-of-Experts (MoE) framework that decouples internal AR modules via a Task-Aware MoE Layer to create task-specific optimization subpaths. To enhance task differentiation while maintaining overall coordination, we introduce a novel Two-Stage Training Strategy. Extensive experiments on multimodal benchmarks demonstrate that UTAMoE mitigates task objective conflicts, achieving state-of-the-art performance across various tasks. Visualizations and ablation studies further validate the effectiveness of our approach.
>
---
#### [new 049] Image Editing As Programs with Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，旨在解决扩散模型在结构不一致的指令驱动编辑中的不足。作者提出IEAP框架，将编辑操作分解为原子操作序列，通过轻量适配器实现，由视觉语言模型代理编程执行复杂编辑，显著提升了编辑准确性和语义保真度。**

- **链接: [http://arxiv.org/pdf/2506.04158v1](http://arxiv.org/pdf/2506.04158v1)**

> **作者:** Yujia Hu; Songhua Liu; Zhenxiong Tan; Xingyi Yang; Xinchao Wang
>
> **摘要:** While diffusion models have achieved remarkable success in text-to-image generation, they encounter significant challenges with instruction-driven image editing. Our research highlights a key challenge: these models particularly struggle with structurally inconsistent edits that involve substantial layout changes. To mitigate this gap, we introduce Image Editing As Programs (IEAP), a unified image editing framework built upon the Diffusion Transformer (DiT) architecture. At its core, IEAP approaches instructional editing through a reductionist lens, decomposing complex editing instructions into sequences of atomic operations. Each operation is implemented via a lightweight adapter sharing the same DiT backbone and is specialized for a specific type of edit. Programmed by a vision-language model (VLM)-based agent, these operations collaboratively support arbitrary and structurally inconsistent transformations. By modularizing and sequencing edits in this way, IEAP generalizes robustly across a wide range of editing tasks, from simple adjustments to substantial structural changes. Extensive experiments demonstrate that IEAP significantly outperforms state-of-the-art methods on standard benchmarks across various editing scenarios. In these evaluations, our framework delivers superior accuracy and semantic fidelity, particularly for complex, multi-step instructions. Codes are available at https://github.com/YujiaHu1109/IEAP.
>
---
#### [new 050] INP-Former++: Advancing Universal Anomaly Detection via Intrinsic Normal Prototypes and Residual Learning
- **分类: cs.CV**

- **简介: 该论文属于异常检测任务，旨在解决因测试图像与训练集正常样本对齐困难导致的检测精度问题。论文提出INP-Former++，通过提取测试图像内部的“内在正常原型”并结合残差学习进行重建，利用重建误差作为异常得分，实现了在多种设置下的高性能异常检测。**

- **链接: [http://arxiv.org/pdf/2506.03660v1](http://arxiv.org/pdf/2506.03660v1)**

> **作者:** Wei Luo; Haiming Yao; Yunkang Cao; Qiyu Chen; Ang Gao; Weiming Shen; Weihang Zhang; Wenyong Yu
>
> **备注:** 15 pages, 11 figures, 13 tables
>
> **摘要:** Anomaly detection (AD) is essential for industrial inspection and medical diagnosis, yet existing methods typically rely on ``comparing'' test images to normal references from a training set. However, variations in appearance and positioning often complicate the alignment of these references with the test image, limiting detection accuracy. We observe that most anomalies manifest as local variations, meaning that even within anomalous images, valuable normal information remains. We argue that this information is useful and may be more aligned with the anomalies since both the anomalies and the normal information originate from the same image. Therefore, rather than relying on external normality from the training set, we propose INP-Former, a novel method that extracts Intrinsic Normal Prototypes (INPs) directly from the test image. Specifically, we introduce the INP Extractor, which linearly combines normal tokens to represent INPs. We further propose an INP Coherence Loss to ensure INPs can faithfully represent normality for the testing image. These INPs then guide the INP-guided Decoder to reconstruct only normal tokens, with reconstruction errors serving as anomaly scores. Additionally, we propose a Soft Mining Loss to prioritize hard-to-optimize samples during training. INP-Former achieves state-of-the-art performance in single-class, multi-class, and few-shot AD tasks across MVTec-AD, VisA, and Real-IAD, positioning it as a versatile and universal solution for AD. Remarkably, INP-Former also demonstrates some zero-shot AD capability. Furthermore, we propose a soft version of the INP Coherence Loss and enhance INP-Former by incorporating residual learning, leading to the development of INP-Former++. The proposed method significantly improves detection performance across single-class, multi-class, semi-supervised, few-shot, and zero-shot settings.
>
---
#### [new 051] OV-COAST: Cost Aggregation with Optimal Transport for Open-Vocabulary Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于开放词汇语义分割（OVSS）任务，旨在通过文本描述为图像每个像素分配语义标签。为提升模型泛化能力，作者提出OV-COAST方法，利用最优传输理论对视觉-语言特征进行成本聚合优化。实验表明该方法显著提升了CAT-Seg模型的性能。**

- **链接: [http://arxiv.org/pdf/2506.03706v1](http://arxiv.org/pdf/2506.03706v1)**

> **作者:** Aditya Gandhamal; Aniruddh Sikdar; Suresh Sundaram
>
> **备注:** Accepted at CVPR 2025 Workshop on Transformers for Vision (Non-archival track)
>
> **摘要:** Open-vocabulary semantic segmentation (OVSS) entails assigning semantic labels to each pixel in an image using textual descriptions, typically leveraging world models such as CLIP. To enhance out-of-domain generalization, we propose Cost Aggregation with Optimal Transport (OV-COAST) for open-vocabulary semantic segmentation. To align visual-language features within the framework of optimal transport theory, we employ cost volume to construct a cost matrix, which quantifies the distance between two distributions. Our approach adopts a two-stage optimization strategy: in the first stage, the optimal transport problem is solved using cost volume via Sinkhorn distance to obtain an alignment solution; in the second stage, this solution is used to guide the training of the CAT-Seg model. We evaluate state-of-the-art OVSS models on the MESS benchmark, where our approach notably improves the performance of the cost-aggregation model CAT-Seg with ViT-B backbone, achieving superior results, surpassing CAT-Seg by 1.72 % and SAN-B by 4.9 % mIoU. The code is available at https://github.com/adityagandhamal/OV-COAST/}{https://github.com/adityagandhamal/OV-COAST/ .
>
---
#### [new 052] UNIC: Unified In-Context Video Editing
- **分类: cs.CV**

- **简介: 该论文属于视频编辑任务，旨在解决现有方法依赖任务特定架构、难以统一处理多编辑任务的问题。作者提出UNIC框架，通过统一输入表示和改进模型设计，实现多种视频编辑任务的集成处理，并验证了其优越性能与任务组合能力。**

- **链接: [http://arxiv.org/pdf/2506.04216v1](http://arxiv.org/pdf/2506.04216v1)**

> **作者:** Zixuan Ye; Xuanhua He; Quande Liu; Qiulin Wang; Xintao Wang; Pengfei Wan; Di Zhang; Kun Gai; Qifeng Chen; Wenhan Luo
>
> **备注:** The project page is at \href{https://zixuan-ye.github.io/UNIC}{https://zixuan-ye.github.io/UNIC}
>
> **摘要:** Recent advances in text-to-video generation have sparked interest in generative video editing tasks. Previous methods often rely on task-specific architectures (e.g., additional adapter modules) or dedicated customizations (e.g., DDIM inversion), which limit the integration of versatile editing conditions and the unification of various editing tasks. In this paper, we introduce UNified In-Context Video Editing (UNIC), a simple yet effective framework that unifies diverse video editing tasks within a single model in an in-context manner. To achieve this unification, we represent the inputs of various video editing tasks as three types of tokens: the source video tokens, the noisy video latent, and the multi-modal conditioning tokens that vary according to the specific editing task. Based on this formulation, our key insight is to integrate these three types into a single consecutive token sequence and jointly model them using the native attention operations of DiT, thereby eliminating the need for task-specific adapter designs. Nevertheless, direct task unification under this framework is challenging, leading to severe token collisions and task confusion due to the varying video lengths and diverse condition modalities across tasks. To address these, we introduce task-aware RoPE to facilitate consistent temporal positional encoding, and condition bias that enables the model to clearly differentiate different editing tasks. This allows our approach to adaptively perform different video editing tasks by referring the source video and varying condition tokens "in context", and support flexible task composition. To validate our method, we construct a unified video editing benchmark containing six representative video editing tasks. Results demonstrate that our unified approach achieves superior performance on each task and exhibits emergent task composition abilities.
>
---
#### [new 053] FullDiT2: Efficient In-Context Conditioning for Video Diffusion Transformers
- **分类: cs.CV**

- **简介: 该论文属于视频生成与编辑任务，旨在解决现有方法计算效率低的问题。通过分析冗余来源，提出FullDiT2框架，采用动态token选择和上下文缓存机制，显著提升处理速度并减少计算开销，同时保持生成质量。**

- **链接: [http://arxiv.org/pdf/2506.04213v1](http://arxiv.org/pdf/2506.04213v1)**

> **作者:** Xuanhua He; Quande Liu; Zixuan Ye; Wecai Ye; Qiulin Wang; Xintao Wang; Qifeng Chen; Pengfei Wan; Di Zhang; Kun Gai
>
> **摘要:** Fine-grained and efficient controllability on video diffusion transformers has raised increasing desires for the applicability. Recently, In-context Conditioning emerged as a powerful paradigm for unified conditional video generation, which enables diverse controls by concatenating varying context conditioning signals with noisy video latents into a long unified token sequence and jointly processing them via full-attention, e.g., FullDiT. Despite their effectiveness, these methods face quadratic computation overhead as task complexity increases, hindering practical deployment. In this paper, we study the efficiency bottleneck neglected in original in-context conditioning video generation framework. We begin with systematic analysis to identify two key sources of the computation inefficiencies: the inherent redundancy within context condition tokens and the computational redundancy in context-latent interactions throughout the diffusion process. Based on these insights, we propose FullDiT2, an efficient in-context conditioning framework for general controllability in both video generation and editing tasks, which innovates from two key perspectives. Firstly, to address the token redundancy, FullDiT2 leverages a dynamic token selection mechanism to adaptively identify important context tokens, reducing the sequence length for unified full-attention. Additionally, a selective context caching mechanism is devised to minimize redundant interactions between condition tokens and video latents. Extensive experiments on six diverse conditional video editing and generation tasks demonstrate that FullDiT2 achieves significant computation reduction and 2-3 times speedup in averaged time cost per diffusion step, with minimal degradation or even higher performance in video generation quality. The project page is at \href{https://fulldit2.github.io/}{https://fulldit2.github.io/}.
>
---
#### [new 054] Chipmunk: Training-Free Acceleration of Diffusion Transformers with Dynamic Column-Sparse Deltas
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像和视频生成任务，旨在加速扩散Transformer（DiT）的推理过程。通过分析激活变化，发现部分中间激活值变化缓慢，提出Chipmunk方法，在不重新训练的情况下利用动态稀疏性减少冗余计算，并优化GPU计算与缓存更新，实现推理加速，同时保持生成质量。**

- **链接: [http://arxiv.org/pdf/2506.03275v1](http://arxiv.org/pdf/2506.03275v1)**

> **作者:** Austin Silveria; Soham V. Govande; Daniel Y. Fu
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Diffusion Transformers (DiTs) have achieved state-of-the-art performance in high-quality image and video generation but incur substantial compute cost at inference. A common observation is that DiT latent noise vectors change slowly across inference steps, which suggests that the DiT compute may be redundant across steps. In this paper, we aim to speed up inference by reducing this redundancy, without additional training. We first study how activations change between steps in two state-of-the-art open-source DiTs. We find that just 5-25% of the values in attention and MLP explain 70-90% of the change in activations across steps. This finding motivates our approach, Chipmunk, which uses dynamic sparsity at inference time to recompute only the fastest-changing intermediate activations, while caching the rest. Dynamic sparsity introduces two systems challenges: (1) sparse attention and MLP operations tend to underutilize GPU tensor cores; and (2) computing dynamic sparsity patterns at runtime and caching activations both introduce overhead. To address these challenges, Chipmunk first uses a voxel-based reordering of input tokens to introduce column-wise sparsity. We implement column-sparse kernels utilizing efficient sparse gathers from global to shared GPU memory, achieving a 9.3x speedup at 93% sparsity compared to highly-optimized dense baselines. Second, Chipmunk overlaps the computation of sparsity patterns and cache updates with other parts of the computation (e.g., second layer of the MLP) to hide the extra latency. Chipmunk achieves up to 2.16x speedup on HunyuanVideo and 1.41x on FLUX.1-dev without compromising generation quality. Furthermore, we show that Chipmunk can be stacked on top of full step caching, achieving a 3.72x speedup on HunyuanVideo, a 2.67x speedup on WAN2.1, and a 2.25x speedup on FLUX.1-dev with minimal quality impact.
>
---
#### [new 055] Multi-view Surface Reconstruction Using Normal and Reflectance Cues
- **分类: cs.CV**

- **简介: 该论文属于三维重建任务，旨在解决复杂材质和稀疏视角下高精度表面重建难题。作者提出了一种融合多视角法线和反射率信息的框架，通过像素级联合参数化实现表面法线与反射率的统一表示，并集成到传统MVS或现代NVR流程中，显著提升了细粒度重建效果及对复杂光照条件的适应性。**

- **链接: [http://arxiv.org/pdf/2506.04115v1](http://arxiv.org/pdf/2506.04115v1)**

> **作者:** Robin Bruneau; Baptiste Brument; Yvain Quéau; Jean Mélou; François Bernard Lauze; Jean-Denis Durou; Lilian Calvet
>
> **备注:** 22 pages, 15 figures, 11 tables. A thorough qualitative and quantitive study is available in the supplementary material at https://drive.google.com/file/d/1KDfCKediXNP5Os954TL_QldaUWS0nKcD/view?usp=drive_link
>
> **摘要:** Achieving high-fidelity 3D surface reconstruction while preserving fine details remains challenging, especially in the presence of materials with complex reflectance properties and without a dense-view setup. In this paper, we introduce a versatile framework that incorporates multi-view normal and optionally reflectance maps into radiance-based surface reconstruction. Our approach employs a pixel-wise joint re-parametrization of reflectance and surface normals, representing them as a vector of radiances under simulated, varying illumination. This formulation enables seamless incorporation into standard surface reconstruction pipelines, such as traditional multi-view stereo (MVS) frameworks or modern neural volume rendering (NVR) ones. Combined with the latter, our approach achieves state-of-the-art performance on multi-view photometric stereo (MVPS) benchmark datasets, including DiLiGenT-MV, LUCES-MV and Skoltech3D. In particular, our method excels in reconstructing fine-grained details and handling challenging visibility conditions. The present paper is an extended version of the earlier conference paper by Brument et al. (in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024), featuring an accelerated and more robust algorithm as well as a broader empirical evaluation. The code and data relative to this article is available at https://github.com/RobinBruneau/RNb-NeuS2.
>
---
#### [new 056] ControlThinker: Unveiling Latent Semantics for Controllable Image Generation through Visual Reasoning
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决文本提示与目标图像间的语义鸿沟问题。作者提出ControlThinker框架，通过视觉推理挖掘控制图像中的潜在语义，丰富文本提示，并利用输出奖励模型选择最佳生成路径，从而提升生成图像的质量与语义一致性。**

- **链接: [http://arxiv.org/pdf/2506.03596v1](http://arxiv.org/pdf/2506.03596v1)**

> **作者:** Feng Han; Yang Jiao; Shaoxiang Chen; Junhao Xu; Jingjing Chen; Yu-Gang Jiang
>
> **摘要:** The field of controllable image generation has seen significant advancements, with various architectures improving generation layout consistency with control signals. However, contemporary methods still face challenges in bridging the semantic gap between input text prompts with sparse semantics and the target images, often over-relying on low-level control signals to infer regional details. To address this challenge, we propose ControlThinker, a novel framework that employs a "comprehend-then-generate" paradigm. Firstly, by incentivizing the visual reasoning capability of a MLLM, latent semantics from control images are mined to enrich text prompts. This enriched semantic understanding then seamlessly aids in image generation without the need for additional complex modifications. To further tackle the uncertainty arising from the ambiguity of control images, we encourage broader exploration of reasoning trajectories and select the optimal one using a metric-based output reward model (ORM). Extensive experimental results demonstrate that ControlThinker effectively mitigates the semantic gap between raw text prompts and target images, resulting in improved visual quality and semantic consistency across a wide range of benchmarks. The code and models are available at https://github.com/Maplebb/ControlThinker.
>
---
#### [new 057] Voyager: Long-Range and World-Consistent Video Diffusion for Explorable 3D Scene Generation
- **分类: cs.CV**

- **简介: 该论文属于3D场景生成任务，旨在解决根据单张图像和自定义相机路径生成长距离、世界一致的可探索3D场景问题。作者提出Voyager框架，通过世界一致的视频扩散、长距离探索机制和可扩展数据引擎，实现端到端的3D点云序列生成，提升了视觉质量和几何准确性。**

- **链接: [http://arxiv.org/pdf/2506.04225v1](http://arxiv.org/pdf/2506.04225v1)**

> **作者:** Tianyu Huang; Wangguandong Zheng; Tengfei Wang; Yuhao Liu; Zhenwei Wang; Junta Wu; Jie Jiang; Hui Li; Rynson W. H. Lau; Wangmeng Zuo; Chunchao Guo
>
> **摘要:** Real-world applications like video gaming and virtual reality often demand the ability to model 3D scenes that users can explore along custom camera trajectories. While significant progress has been made in generating 3D objects from text or images, creating long-range, 3D-consistent, explorable 3D scenes remains a complex and challenging problem. In this work, we present Voyager, a novel video diffusion framework that generates world-consistent 3D point-cloud sequences from a single image with user-defined camera path. Unlike existing approaches, Voyager achieves end-to-end scene generation and reconstruction with inherent consistency across frames, eliminating the need for 3D reconstruction pipelines (e.g., structure-from-motion or multi-view stereo). Our method integrates three key components: 1) World-Consistent Video Diffusion: A unified architecture that jointly generates aligned RGB and depth video sequences, conditioned on existing world observation to ensure global coherence 2) Long-Range World Exploration: An efficient world cache with point culling and an auto-regressive inference with smooth video sampling for iterative scene extension with context-aware consistency, and 3) Scalable Data Engine: A video reconstruction pipeline that automates camera pose estimation and metric depth prediction for arbitrary videos, enabling large-scale, diverse training data curation without manual 3D annotations. Collectively, these designs result in a clear improvement over existing methods in visual quality and geometric accuracy, with versatile applications.
>
---
#### [new 058] Joint Video Enhancement with Deblurring, Super-Resolution, and Frame Interpolation Network
- **分类: cs.CV**

- **简介: 该论文属于视频增强任务，旨在同时解决视频的模糊、低分辨率和低帧率问题。作者提出了一种联合增强网络DSFN，包含联合去模糊与超分辨率模块（JDSR）和三帧插值模块（TFBFI），可同时提升视频清晰度、分辨率和帧率，相比顺序处理方法更高效且效果更好。**

- **链接: [http://arxiv.org/pdf/2506.03892v1](http://arxiv.org/pdf/2506.03892v1)**

> **作者:** Giyong Choi; HyunWook Park
>
> **摘要:** Video quality is often severely degraded by multiple factors rather than a single factor. These low-quality videos can be restored to high-quality videos by sequentially performing appropriate video enhancement techniques. However, the sequential approach was inefficient and sub-optimal because most video enhancement approaches were designed without taking into account that multiple factors together degrade video quality. In this paper, we propose a new joint video enhancement method that mitigates multiple degradation factors simultaneously by resolving an integrated enhancement problem. Our proposed network, named DSFN, directly produces a high-resolution, high-frame-rate, and clear video from a low-resolution, low-frame-rate, and blurry video. In the DSFN, low-resolution and blurry input frames are enhanced by a joint deblurring and super-resolution (JDSR) module. Meanwhile, intermediate frames between input adjacent frames are interpolated by a triple-frame-based frame interpolation (TFBFI) module. The proper combination of the proposed modules of DSFN can achieve superior performance on the joint video enhancement task. Experimental results show that the proposed method outperforms other sequential state-of-the-art techniques on public datasets with a smaller network size and faster processing time.
>
---
#### [new 059] The effects of using created synthetic images in computer vision training
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，旨在解决训练数据不足和网络图像污染问题。论文通过使用Unreal Engine生成合成图像，探索其在补充真实图像数据集中的效果，并提出利用预训练模型评估合成图像质量的方法。实验表明，合成图像可显著减少所需真实数据量，提高训练效率。**

- **链接: [http://arxiv.org/pdf/2506.03449v1](http://arxiv.org/pdf/2506.03449v1)**

> **作者:** John W. Smutny
>
> **备注:** Nine pages long. Main content in pages one through eight. References start at page nine
>
> **摘要:** This paper investigates how rendering engines, like Unreal Engine 4 (UE), can be used to create synthetic images to supplement datasets for deep computer vision (CV) models in image abundant and image limited use cases. Using rendered synthetic images from UE can provide developers and businesses with a method of accessing nearly unlimited, reproducible, agile, and cheap training sets for their customers and applications without the threat of poisoned images from the internet or the cost of collecting them. The validity of these generated images are examined by testing the change in model test accuracy in two different sized CV models across two binary classification cases (Cat vs Dog and Weld Defect Detection). In addition, this paper provides an implementation of how to measure the quality of synthetic images by using pre-trained CV models as auditors. Results imply that for large (VGG16) and small (MobileNetV3-small) parameter deep CV models, adding >60% additional synthetic images to a real image dataset during model training can narrow the test-training accuracy gap to ~1-2% without a conclusive effect on test accuracy compared to using real world images alone. Likewise, adding <10% additional real training images to synthetic only training sets decreased the classification error rate in half, then decreasing further when adding more real training images. For these cases tested, using synthetic images from rendering engines allow researchers to only use 10% of their real images during training, compared to the traditional 50-70%. This research serves as an example of how to create synthetic images, guidelines on how to use the images, potential restrictions and possible performance improvements for data-scarce projects.
>
---
#### [new 060] Vision Remember: Alleviating Visual Forgetting in Efficient MLLM with Vision Feature Resample
- **分类: cs.CV**

- **简介: 该论文属于多模态大语言模型任务，旨在解决视觉信息遗忘问题。通过在解码层间插入“Vision Remember”模块，对视觉特征进行重记忆与局部注意力优化，提升细粒度视觉理解效果，同时保持计算效率。**

- **链接: [http://arxiv.org/pdf/2506.03928v1](http://arxiv.org/pdf/2506.03928v1)**

> **作者:** Ze Feng; Jiang-Jiang Liu; Sen Yang; Lingyu Xiao; Xiaofan Li; Wankou Yang; Jingdong Wang
>
> **摘要:** In this work, we study the Efficient Multimodal Large Language Model. Redundant vision tokens consume a significant amount of computational memory and resources. Therefore, many previous works compress them in the Vision Projector to reduce the number of vision tokens. However, simply compressing in the Vision Projector can lead to the loss of visual information, especially for tasks that rely on fine-grained spatial relationships, such as OCR and Chart \& Table Understanding. To address this problem, we propose Vision Remember, which is inserted between the LLM decoder layers to allow vision tokens to re-memorize vision features. Specifically, we retain multi-level vision features and resample them with the vision tokens that have interacted with the text token. During the resampling process, each vision token only attends to a local region in vision features, which is referred to as saliency-enhancing local attention. Saliency-enhancing local attention not only improves computational efficiency but also captures more fine-grained contextual information and spatial relationships within the region. Comprehensive experiments on multiple visual understanding benchmarks validate the effectiveness of our method when combined with various Efficient Vision Projectors, showing performance gains without sacrificing efficiency. Based on Vision Remember, LLaVA-VR with only 2B parameters is also superior to previous representative MLLMs such as Tokenpacker-HD-7B and DeepSeek-VL-7B.
>
---
#### [new 061] Spatial Understanding from Videos: Structured Prompts Meet Simulation Data
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型的三维空间推理任务，旨在解决现有方法在空间不确定性和数据稀缺性下的局限性。作者提出了一种无需修改模型架构的框架，结合结构化提示策略SpatialMind与新构建的ScanForgeQA数据集，提升预训练模型的空间理解能力，并通过实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2506.03642v1](http://arxiv.org/pdf/2506.03642v1)**

> **作者:** Haoyu Zhang; Meng Liu; Zaijing Li; Haokun Wen; Weili Guan; Yaowei Wang; Liqiang Nie
>
> **摘要:** Visual-spatial understanding, the ability to infer object relationships and layouts from visual input, is fundamental to downstream tasks such as robotic navigation and embodied interaction. However, existing methods face spatial uncertainty and data scarcity, limiting the 3D spatial reasoning capability of pre-trained vision-language models (VLMs). To address these challenges, we present a unified framework for enhancing 3D spatial reasoning in pre-trained VLMs without modifying their architecture. This framework combines SpatialMind, a structured prompting strategy that decomposes complex scenes and questions into interpretable reasoning steps, with ScanForgeQA, a scalable question-answering dataset built from diverse 3D simulation scenes through an automated construction process designed for fine-tuning. Extensive experiments across multiple benchmarks demonstrate the individual and combined effectiveness of our prompting and fine-tuning strategies, and yield insights that may inspire future research on visual-spatial understanding.
>
---
#### [new 062] VLMs Can Aggregate Scattered Training Patches
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究视觉-语言模型（VLMs）在训练中整合分散图像碎片的能力，称为“视觉拼接”。任务是验证VLMs是否能通过学习带有相同文本描述的碎片图像，在推理阶段还原完整内容。论文展示了该能力，并模拟了恶意数据攻击场景，揭示了VLM的安全风险。**

- **链接: [http://arxiv.org/pdf/2506.03614v1](http://arxiv.org/pdf/2506.03614v1)**

> **作者:** Zhanhui Zhou; Lingjie Chen; Chao Yang; Chaochao Lu
>
> **摘要:** One way to mitigate risks in vision-language models (VLMs) is to remove dangerous samples in their training data. However, such data moderation can be easily bypassed when harmful images are split into small, benign-looking patches, scattered across many training samples. VLMs may then learn to piece these fragments together during training and generate harmful responses at inference, either from full images or text references. For instance, if trained on image patches from a bloody scene paired with the descriptions "safe," VLMs may later describe, the full image or a text reference to the scene, as "safe." We define the core ability of VLMs enabling this attack as $\textit{visual stitching}$ -- the ability to integrate visual information spread across multiple training samples that share the same textual descriptions. In our work, we first demonstrate visual stitching abilities in common open-source VLMs on three datasets where each image is labeled with a unique synthetic ID: we split each $(\texttt{image}, \texttt{ID})$ pair into $\{(\texttt{patch}, \texttt{ID})\}$ pairs at different granularity for finetuning, and we find that tuned models can verbalize the correct IDs from full images or text reference. Building on this, we simulate the adversarial data poisoning scenario mentioned above by using patches from dangerous images and replacing IDs with text descriptions like ``safe'' or ``unsafe'', demonstrating how harmful content can evade moderation in patches and later be reconstructed through visual stitching, posing serious VLM safety risks. Code is available at https://github.com/ZHZisZZ/visual-stitching.
>
---
#### [new 063] Learning from Noise: Enhancing DNNs for Event-Based Vision through Controlled Noise Injection
- **分类: cs.CV**

- **简介: 该论文属于事件视觉任务，旨在解决事件数据噪声影响深度学习模型性能的问题。作者提出了一种可控噪声注入的训练方法，使模型学习抗噪表示，并在多个数据集和网络架构上验证了其有效性，结果表明该方法优于传统滤波技术。**

- **链接: [http://arxiv.org/pdf/2506.03918v1](http://arxiv.org/pdf/2506.03918v1)**

> **作者:** Marcin Kowalczyk; Kamil Jeziorek; Tomasz Kryjak
>
> **摘要:** Event-based sensors offer significant advantages over traditional frame-based cameras, especially in scenarios involving rapid motion or challenging lighting conditions. However, event data frequently suffers from considerable noise, negatively impacting the performance and robustness of deep learning models. Traditionally, this problem has been addressed by applying filtering algorithms to the event stream, but this may also remove some of relevant data. In this paper, we propose a novel noise-injection training methodology designed to enhance the neural networks robustness against varying levels of event noise. Our approach introduces controlled noise directly into the training data, enabling models to learn noise-resilient representations. We have conducted extensive evaluations of the proposed method using multiple benchmark datasets (N-Caltech101, N-Cars, and Mini N-ImageNet) and various network architectures, including Convolutional Neural Networks, Vision Transformers, Spiking Neural Networks, and Graph Convolutional Networks. Experimental results show that our noise-injection training strategy achieves stable performance over a range of noise intensities, consistently outperforms event-filtering techniques, and achieves the highest average classification accuracy, making it a viable alternative to traditional event-data filtering methods in an object classification system. Code: https://github.com/vision-agh/DVS_Filtering
>
---
#### [new 064] CoLa: Chinese Character Decomposition with Compositional Latent Components
- **分类: cs.CV**

- **简介: 该论文属于中文字符识别任务，旨在解决零样本字符识别问题。现有方法依赖人工定义的分解规则，限制了泛化能力。本文提出CoLa模型，通过学习字符的组合隐变量实现自主分解与重组，支持跨数据集的零样本识别，并具备良好的可解释性。**

- **链接: [http://arxiv.org/pdf/2506.03798v1](http://arxiv.org/pdf/2506.03798v1)**

> **作者:** Fan Shi; Haiyang Yu; Bin Li; Xiangyang Xue
>
> **摘要:** Humans can decompose Chinese characters into compositional components and recombine them to recognize unseen characters. This reflects two cognitive principles: Compositionality, the idea that complex concepts are built on simpler parts; and Learning-to-learn, the ability to learn strategies for decomposing and recombining components to form new concepts. These principles provide inductive biases that support efficient generalization. They are critical to Chinese character recognition (CCR) in solving the zero-shot problem, which results from the common long-tail distribution of Chinese character datasets. Existing methods have made substantial progress in modeling compositionality via predefined radical or stroke decomposition. However, they often ignore the learning-to-learn capability, limiting their ability to generalize beyond human-defined schemes. Inspired by these principles, we propose a deep latent variable model that learns Compositional Latent components of Chinese characters (CoLa) without relying on human-defined decomposition schemes. Recognition and matching can be performed by comparing compositional latent components in the latent space, enabling zero-shot character recognition. The experiments illustrate that CoLa outperforms previous methods in both character the radical zero-shot CCR. Visualization indicates that the learned components can reflect the structure of characters in an interpretable way. Moreover, despite being trained on historical documents, CoLa can analyze components of oracle bone characters, highlighting its cross-dataset generalization ability.
>
---
#### [new 065] RefEdit: A Benchmark and Method for Improving Instruction-based Image Editing Model on Referring Expressions
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，旨在解决基于指令的图像编辑模型在复杂场景中多实体编辑效果差的问题。作者提出了RefEdit-Bench作为评估基准，并构建了RefEdit模型及其合成数据生成方法，在少量数据上实现了超越大规模训练基线模型的效果。**

- **链接: [http://arxiv.org/pdf/2506.03448v1](http://arxiv.org/pdf/2506.03448v1)**

> **作者:** Bimsara Pathiraja; Maitreya Patel; Shivam Singh; Yezhou Yang; Chitta Baral
>
> **备注:** Project page: \url{http://refedit.vercel.app}
>
> **摘要:** Despite recent advances in inversion and instruction-based image editing, existing approaches primarily excel at editing single, prominent objects but significantly struggle when applied to complex scenes containing multiple entities. To quantify this gap, we first introduce RefEdit-Bench, a rigorous real-world benchmark rooted in RefCOCO, where even baselines trained on millions of samples perform poorly. To overcome this limitation, we introduce RefEdit -- an instruction-based editing model trained on our scalable synthetic data generation pipeline. Our RefEdit, trained on only 20,000 editing triplets, outperforms the Flux/SD3 model-based baselines trained on millions of data. Extensive evaluations across various benchmarks demonstrate that our model not only excels in referring expression tasks but also enhances performance on traditional benchmarks, achieving state-of-the-art results comparable to closed-source methods. We release data \& checkpoint for reproducibility.
>
---
#### [new 066] EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation
- **分类: cs.CV**

- **简介: 该论文属于事件相机光流估计任务，旨在解决现有方法计算冗余、分辨率扩展性差的问题。作者提出EDCFlow，结合时间密集特征差异与成本体，通过多尺度注意力机制和自适应融合提升运动表征，实现高效高分辨率光流估计，并可作为插件增强RAFT-like方法。**

- **链接: [http://arxiv.org/pdf/2506.03512v1](http://arxiv.org/pdf/2506.03512v1)**

> **作者:** Daikun Liu; Lei Cheng; Teng Wang; changyin Sun
>
> **备注:** 14 pages, 8 figures
>
> **摘要:** Recent learning-based methods for event-based optical flow estimation utilize cost volumes for pixel matching but suffer from redundant computations and limited scalability to higher resolutions for flow refinement. In this work, we take advantage of the complementarity between temporally dense feature differences of adjacent event frames and cost volume and present a lightweight event-based optical flow network (EDCFlow) to achieve high-quality flow estimation at a higher resolution. Specifically, an attention-based multi-scale temporal feature difference layer is developed to capture diverse motion patterns at high resolution in a computation-efficient manner. An adaptive fusion of high-resolution difference motion features and low-resolution correlation motion features is performed to enhance motion representation and model generalization. Notably, EDCFlow can serve as a plug-and-play refinement module for RAFT-like event-based methods to enhance flow details. Extensive experiments demonstrate that EDCFlow achieves better performance with lower complexity compared to existing methods, offering superior generalization.
>
---
#### [new 067] EmoArt: A Multidimensional Dataset for Emotion-Aware Artistic Generation
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决情感表达艺术图像生成缺乏高质量数据的问题。作者构建了EmoArt数据集，包含13万+艺术作品及多维情感标注，并评估了主流扩散模型的情感生成能力，推动情感驱动图像合成及相关领域发展。**

- **链接: [http://arxiv.org/pdf/2506.03652v1](http://arxiv.org/pdf/2506.03652v1)**

> **作者:** Cheng Zhang; Hongxia xie; Bin Wen; Songhan Zuo; Ruoxuan Zhang; Wen-huang Cheng
>
> **摘要:** With the rapid advancement of diffusion models, text-to-image generation has achieved significant progress in image resolution, detail fidelity, and semantic alignment, particularly with models like Stable Diffusion 3.5, Stable Diffusion XL, and FLUX 1. However, generating emotionally expressive and abstract artistic images remains a major challenge, largely due to the lack of large-scale, fine-grained emotional datasets. To address this gap, we present the EmoArt Dataset -- one of the most comprehensive emotion-annotated art datasets to date. It contains 132,664 artworks across 56 painting styles (e.g., Impressionism, Expressionism, Abstract Art), offering rich stylistic and cultural diversity. Each image includes structured annotations: objective scene descriptions, five key visual attributes (brushwork, composition, color, line, light), binary arousal-valence labels, twelve emotion categories, and potential art therapy effects. Using EmoArt, we systematically evaluate popular text-to-image diffusion models for their ability to generate emotionally aligned images from text. Our work provides essential data and benchmarks for emotion-driven image synthesis and aims to advance fields such as affective computing, multimodal learning, and computational art, enabling applications in art therapy and creative design. The dataset and more details can be accessed via our project website.
>
---
#### [new 068] DSSAU-Net:U-Shaped Hybrid Network for Pubic Symphysis and Fetal Head Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决分娩过程中胎儿头部和耻骨联合的准确分割问题。通过设计高效的DSSAU-Net网络结构，结合稀疏自注意力机制与多尺度特征融合，提升计算效率及分割精度，应用于产科超声图像分析。**

- **链接: [http://arxiv.org/pdf/2506.03684v1](http://arxiv.org/pdf/2506.03684v1)**

> **作者:** Zunhui Xia; Hongxing Li; Libin Lan
>
> **备注:** 14 pages, 3 figures, 5 tables.Accepted by MICCAI Workshop on IUGC 2024
>
> **摘要:** In the childbirth process, traditional methods involve invasive vaginal examinations, but research has shown that these methods are both subjective and inaccurate. Ultrasound-assisted diagnosis offers an objective yet effective way to assess fetal head position via two key parameters: Angle of Progression (AoP) and Head-Symphysis Distance (HSD), calculated by segmenting the fetal head (FH) and pubic symphysis (PS), which aids clinicians in ensuring a smooth delivery process. Therefore, accurate segmentation of FH and PS is crucial. In this work, we propose a sparse self-attention network architecture with good performance and high computational efficiency, named DSSAU-Net, for the segmentation of FH and PS. Specifically, we stack varying numbers of Dual Sparse Selection Attention (DSSA) blocks at each stage to form a symmetric U-shaped encoder-decoder network architecture. For a given query, DSSA is designed to explicitly perform one sparse token selection at both the region and pixel levels, respectively, which is beneficial for further reducing computational complexity while extracting the most relevant features. To compensate for the information loss during the upsampling process, skip connections with convolutions are designed. Additionally, multiscale feature fusion is employed to enrich the model's global and local information. The performance of DSSAU-Net has been validated using the Intrapartum Ultrasound Grand Challenge (IUGC) 2024 \textit{test set} provided by the organizer in the MICCAI IUGC 2024 competition\footnote{\href{https://codalab.lisn.upsaclay.fr/competitions/18413\#learn\_the\_details}{https://codalab.lisn.upsaclay.fr/competitions/18413\#learn\_the\_details}}, where we win the fourth place on the tasks of classification and segmentation, demonstrating its effectiveness. The codes will be available at https://github.com/XiaZunhui/DSSAU-Net.
>
---
#### [new 069] Vid-SME: Membership Inference Attacks against Large Video Understanding Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频理解模型的安全分析任务，旨在解决视频数据在大模型训练中的隐私泄露问题。针对现有方法无法有效处理视频时序特性和帧数变化的问题，作者提出了Vid-SME方法，利用Sharma-Mittal熵和模型输出置信度进行成员推理攻击，判断视频是否参与训练，提升了攻击效果。**

- **链接: [http://arxiv.org/pdf/2506.03179v1](http://arxiv.org/pdf/2506.03179v1)**

> **作者:** Qi Li; Runpeng Yu; Xinchao Wang
>
> **摘要:** Multimodal large language models (MLLMs) demonstrate remarkable capabilities in handling complex multimodal tasks and are increasingly adopted in video understanding applications. However, their rapid advancement raises serious data privacy concerns, particularly given the potential inclusion of sensitive video content, such as personal recordings and surveillance footage, in their training datasets. Determining improperly used videos during training remains a critical and unresolved challenge. Despite considerable progress on membership inference attacks (MIAs) for text and image data in MLLMs, existing methods fail to generalize effectively to the video domain. These methods suffer from poor scalability as more frames are sampled and generally achieve negligible true positive rates at low false positive rates (TPR@Low FPR), mainly due to their failure to capture the inherent temporal variations of video frames and to account for model behavior differences as the number of frames varies. To address these challenges, we introduce Vid-SME, the first membership inference method tailored for video data used in video understanding LLMs (VULLMs). Vid-SME leverages the confidence of model output and integrates adaptive parameterization to compute Sharma-Mittal entropy (SME) for video inputs. By leveraging the SME difference between natural and temporally-reversed video frames, Vid-SME derives robust membership scores to determine whether a given video is part of the model's training set. Experiments on various self-trained and open-sourced VULLMs demonstrate the strong effectiveness of Vid-SME.
>
---
#### [new 070] SportMamba: Adaptive Non-Linear Multi-Object Tracking with State Space Models for Team Sports
- **分类: cs.CV**

- **简介: 该论文属于多目标跟踪任务，旨在解决团队体育场景中因快速运动和频繁遮挡导致的身份切换和运动模糊问题。论文提出了SportMamba方法，结合状态空间模型，引入mamba-attention机制和高度自适应的空间关联度量，提升了复杂场景下的跟踪性能，并在多个数据集上验证了效果。**

- **链接: [http://arxiv.org/pdf/2506.03335v1](http://arxiv.org/pdf/2506.03335v1)**

> **作者:** Dheeraj Khanna; Jerrin Bright; Yuhao Chen; John S. Zelek
>
> **备注:** Paper accepted at CVSports IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW'25). The paper has 8 pages, including 6 Figures and 5 Tables
>
> **摘要:** Multi-object tracking (MOT) in team sports is particularly challenging due to the fast-paced motion and frequent occlusions resulting in motion blur and identity switches, respectively. Predicting player positions in such scenarios is particularly difficult due to the observed highly non-linear motion patterns. Current methods are heavily reliant on object detection and appearance-based tracking, which struggle to perform in complex team sports scenarios, where appearance cues are ambiguous and motion patterns do not necessarily follow a linear pattern. To address these challenges, we introduce SportMamba, an adaptive hybrid MOT technique specifically designed for tracking in dynamic team sports. The technical contribution of SportMamba is twofold. First, we introduce a mamba-attention mechanism that models non-linear motion by implicitly focusing on relevant embedding dependencies. Second, we propose a height-adaptive spatial association metric to reduce ID switches caused by partial occlusions by accounting for scale variations due to depth changes. Additionally, we extend the detection search space with adaptive buffers to improve associations in fast-motion scenarios. Our proposed technique, SportMamba, demonstrates state-of-the-art performance on various metrics in the SportsMOT dataset, which is characterized by complex motion and severe occlusion. Furthermore, we demonstrate its generalization capability through zero-shot transfer to VIP-HTD, an ice hockey dataset.
>
---
#### [new 071] DiagNet: Detecting Objects using Diagonal Constraints on Adjacency Matrix of Graph Neural Network
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于目标检测任务，旨在解决传统方法需预设锚框的问题。作者提出DiagNet，通过图卷积网络邻接矩阵的对角约束实现边界框预测，无需锚框设计。实验表明其在多个数据集上优于YOLO系列模型。**

- **链接: [http://arxiv.org/pdf/2506.03571v1](http://arxiv.org/pdf/2506.03571v1)**

> **作者:** Chong Hyun Lee; Kibae Lee
>
> **摘要:** We propose DaigNet, a new approach to object detection with which we can detect an object bounding box using diagonal constraints on adjacency matrix of a graph convolutional network (GCN). We propose two diagonalization algorithms based on hard and soft constraints on adjacency matrix and two loss functions using diagonal constraint and complementary constraint. The DaigNet eliminates the need for designing a set of anchor boxes commonly used. To prove feasibility of our novel detector, we adopt detection head in YOLO models. Experiments show that the DiagNet achieves 7.5% higher mAP50 on Pascal VOC than YOLOv1. The DiagNet also shows 5.1% higher mAP on MS COCO than YOLOv3u, 3.7% higher mAP than YOLOv5u, and 2.9% higher mAP than YOLOv8.
>
---
#### [new 072] ViT-Split: Unleashing the Power of Vision Foundation Models via Efficient Splitting Heads
- **分类: cs.CV**

- **简介: 该论文属于视觉基础模型（VFM）适配任务。针对现有适配方法效率低、需调整个全部组件且削弱先验知识的问题，提出ViT-Split，通过分离特征提取与任务适配，引入任务头和先验头，减少训练时间和参数，提升性能。**

- **链接: [http://arxiv.org/pdf/2506.03433v1](http://arxiv.org/pdf/2506.03433v1)**

> **作者:** Yifan Li; Xin Li; Tianqin Li; Wenbin He; Yu Kong; Liu Ren
>
> **备注:** The project is available: https://jackyfl.github.io/vitsplit.github.io/
>
> **摘要:** Vision foundation models (VFMs) have demonstrated remarkable performance across a wide range of downstream tasks. While several VFM adapters have shown promising results by leveraging the prior knowledge of VFMs, we identify two inefficiencies in these approaches. First, the interaction between convolutional neural network (CNN) and VFM backbone triggers early layer gradient backpropagation. Second, existing methods require tuning all components, adding complexity. Besides, these adapters alter VFM features, underutilizing the prior knowledge. To tackle these challenges, we propose a new approach called ViT-Split, based on a key observation: the layers of several VFMs, like DINOv2, can be divided into two distinct components: an extractor for learning low-level features and an adapter for learning task-specific features. Leveraging this insight, we eliminate the CNN branch and introduce two heads, task head and prior head, to the frozen VFM. The task head is designed to learn task-specific features, mitigating the early gradient propagation issue. The prior head is used to leverage the multi-scale prior features from the frozen VFM, reducing tuning parameters and overfitting. Extensive experiments on various tasks (e.g., segmentation, detection, depth estimation, and visual question answering) validate the effectiveness and efficiency of ViT-Split. Specifically, ViT-Split reduces training time up to $4\times$ while achieving comparable or even better results on ADE20K, compared to other VFM adapters.
>
---
#### [new 073] MambaNeXt-YOLO: A Hybrid State Space Model for Real-time Object Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于实时目标检测任务，旨在解决在计算资源受限下模型速度与精度的平衡问题。作者提出MambaNeXt-YOLO，结合CNN与Mamba模型，设计了高效结构和多尺度融合网络，实现在边缘设备上的高性能检测。**

- **链接: [http://arxiv.org/pdf/2506.03654v1](http://arxiv.org/pdf/2506.03654v1)**

> **作者:** Xiaochun Lei; Siqi Wu; Weilin Wu; Zetao Jiang
>
> **摘要:** Real-time object detection is a fundamental but challenging task in computer vision, particularly when computational resources are limited. Although YOLO-series models have set strong benchmarks by balancing speed and accuracy, the increasing need for richer global context modeling has led to the use of Transformer-based architectures. Nevertheless, Transformers have high computational complexity because of their self-attention mechanism, which limits their practicality for real-time and edge deployments. To overcome these challenges, recent developments in linear state space models, such as Mamba, provide a promising alternative by enabling efficient sequence modeling with linear complexity. Building on this insight, we propose MambaNeXt-YOLO, a novel object detection framework that balances accuracy and efficiency through three key contributions: (1) MambaNeXt Block: a hybrid design that integrates CNNs with Mamba to effectively capture both local features and long-range dependencies; (2) Multi-branch Asymmetric Fusion Pyramid Network (MAFPN): an enhanced feature pyramid architecture that improves multi-scale object detection across various object sizes; and (3) Edge-focused Efficiency: our method achieved 66.6\% mAP at 31.9 FPS on the PASCAL VOC dataset without any pre-training and supports deployment on edge devices such as the NVIDIA Jetson Xavier NX and Orin NX.
>
---
#### [new 074] DiffCAP: Diffusion-based Cumulative Adversarial Purification for Vision Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型（VLM）的鲁棒性任务，旨在解决VLM在面对对抗扰动时输出不稳定的问题。作者提出DiffCAP方法，通过扩散模型逐步去噪对抗样本，直到其嵌入稳定，从而提升模型鲁棒性。实验表明该方法优于现有防御技术，具有高效性和实用性。**

- **链接: [http://arxiv.org/pdf/2506.03933v1](http://arxiv.org/pdf/2506.03933v1)**

> **作者:** Jia Fu; Yongtao Wu; Yihang Chen; Kunyu Peng; Xiao Zhang; Volkan Cevher; Sepideh Pashami; Anders Holst
>
> **摘要:** Vision Language Models (VLMs) have shown remarkable capabilities in multimodal understanding, yet their susceptibility to perturbations poses a significant threat to their reliability in real-world applications. Despite often being imperceptible to humans, these perturbations can drastically alter model outputs, leading to erroneous interpretations and decisions. This paper introduces DiffCAP, a novel diffusion-based purification strategy that can effectively neutralize adversarial corruptions in VLMs. We observe that adding minimal noise to an adversarially corrupted image significantly alters its latent embedding with respect to VLMs. Building on this insight, DiffCAP cumulatively injects random Gaussian noise into adversarially perturbed input data. This process continues until the embeddings of two consecutive noisy images reach a predefined similarity threshold, indicating a potential approach to neutralize the adversarial effect. Subsequently, a pretrained diffusion model is employed to denoise the stabilized image, recovering a clean representation suitable for the VLMs to produce an output. Through extensive experiments across six datasets with three VLMs under varying attack strengths in three task scenarios, we show that DiffCAP consistently outperforms existing defense techniques by a substantial margin. Notably, DiffCAP significantly reduces both hyperparameter tuning complexity and the required diffusion time, thereby accelerating the denoising process. Equipped with strong theoretical and empirical support, DiffCAP provides a robust and practical solution for securely deploying VLMs in adversarial environments.
>
---
#### [new 075] Seeing in the Dark: Benchmarking Egocentric 3D Vision with the Oxford Day-and-Night Dataset
- **分类: cs.CV**

- **简介: 论文提出了Oxford Day-and-Night数据集，用于解决光照变化下以自我为中心的3D视觉任务。该数据集填补了现有数据在真实3D几何、广泛光照变化和完整6自由度运动方面的空白，支持新视角合成与重定位基准，为相关研究提供了大规模、多样化的测试平台。**

- **链接: [http://arxiv.org/pdf/2506.04224v1](http://arxiv.org/pdf/2506.04224v1)**

> **作者:** Zirui Wang; Wenjing Bian; Xinghui Li; Yifu Tao; Jianeng Wang; Maurice Fallon; Victor Adrian Prisacariu
>
> **备注:** Project page: https://oxdan.active.vision/
>
> **摘要:** We introduce Oxford Day-and-Night, a large-scale, egocentric dataset for novel view synthesis (NVS) and visual relocalisation under challenging lighting conditions. Existing datasets often lack crucial combinations of features such as ground-truth 3D geometry, wide-ranging lighting variation, and full 6DoF motion. Oxford Day-and-Night addresses these gaps by leveraging Meta ARIA glasses to capture egocentric video and applying multi-session SLAM to estimate camera poses, reconstruct 3D point clouds, and align sequences captured under varying lighting conditions, including both day and night. The dataset spans over 30 $\mathrm{km}$ of recorded trajectories and covers an area of 40,000 $\mathrm{m}^2$, offering a rich foundation for egocentric 3D vision research. It supports two core benchmarks, NVS and relocalisation, providing a unique platform for evaluating models in realistic and diverse environments.
>
---
#### [new 076] YOND: Practical Blind Raw Image Denoising Free from Camera-Specific Data Dependency
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于图像去噪任务，旨在解决相机无关的盲原始图像去噪问题。现有方法依赖特定相机数据，性能受限。论文提出YOND方法，通过合成数据训练，包含噪声估计、方差稳定变换和信噪比引导去噪模块，实现对未知相机噪声图像的有效去噪，提升实用性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.03645v1](http://arxiv.org/pdf/2506.03645v1)**

> **作者:** Hansen Feng; Lizhi Wang; Yiqi Huang; Tong Li; Lin Zhu; Hua Huang
>
> **备注:** 17 pages, 19 figures, TPAMI under review
>
> **摘要:** The rapid advancement of photography has created a growing demand for a practical blind raw image denoising method. Recently, learning-based methods have become mainstream due to their excellent performance. However, most existing learning-based methods suffer from camera-specific data dependency, resulting in performance drops when applied to data from unknown cameras. To address this challenge, we introduce a novel blind raw image denoising method named YOND, which represents You Only Need a Denoiser. Trained solely on synthetic data, YOND can generalize robustly to noisy raw images captured by diverse unknown cameras. Specifically, we propose three key modules to guarantee the practicality of YOND: coarse-to-fine noise estimation (CNE), expectation-matched variance-stabilizing transform (EM-VST), and SNR-guided denoiser (SNR-Net). Firstly, we propose CNE to identify the camera noise characteristic, refining the estimated noise parameters based on the coarse denoised image. Secondly, we propose EM-VST to eliminate camera-specific data dependency, correcting the bias expectation of VST according to the noisy image. Finally, we propose SNR-Net to offer controllable raw image denoising, supporting adaptive adjustments and manual fine-tuning. Extensive experiments on unknown cameras, along with flexible solutions for challenging cases, demonstrate the superior practicality of our method. The source code will be publicly available at the \href{https://fenghansen.github.io/publication/YOND}{project homepage}.
>
---
#### [new 077] WIFE-Fusion:Wavelet-aware Intra-inter Frequency Enhancement for Multi-model Image Fusion
- **分类: cs.CV**

- **简介: 该论文属于多模态图像融合任务，旨在解决现有方法忽略频域特征探索与交互关系的问题。作者提出WIFE-Fusion框架，通过Intra-Frequency Self-Attention（IFSA）和Inter-Frequency Interaction（IFI）机制，实现频域内及跨频域的特征增强与融合，提升了融合效果，并在多个数据集上验证了其优越性。**

- **链接: [http://arxiv.org/pdf/2506.03555v1](http://arxiv.org/pdf/2506.03555v1)**

> **作者:** Tianpei Zhang; Jufeng Zhao; Yiming Zhu; Guangmang Cui
>
> **摘要:** Multimodal image fusion effectively aggregates information from diverse modalities, with fused images playing a crucial role in vision systems. However, existing methods often neglect frequency-domain feature exploration and interactive relationships. In this paper, we propose wavelet-aware Intra-inter Frequency Enhancement Fusion (WIFE-Fusion), a multimodal image fusion framework based on frequency-domain components interactions. Its core innovations include: Intra-Frequency Self-Attention (IFSA) that leverages inherent cross-modal correlations and complementarity through interactive self-attention mechanisms to extract enriched frequency-domain features, and Inter-Frequency Interaction (IFI) that enhances enriched features and filters latent features via combinatorial interactions between heterogeneous frequency-domain components across modalities. These processes achieve precise source feature extraction and unified modeling of feature extraction-aggregation. Extensive experiments on five datasets across three multimodal fusion tasks demonstrate WIFE-Fusion's superiority over current specialized and unified fusion methods. Our code is available at https://github.com/Lmmh058/WIFE-Fusion.
>
---
#### [new 078] Seeing the Arrow of Time in Large Multimodal Models
- **分类: cs.CV**

- **简介: 该论文研究大型多模态模型（LMMs）在视频理解中对时间箭头（AoT）的感知问题，属于视频语言理解任务。现有模型难以识别视频的时间方向性，影响深层时序理解。作者提出ArrowRL训练策略，通过强化学习和反向奖励机制提升模型对时间流动的理解能力，并构建新基准AoTBench评估时序理解效果。实验表明该方法显著提升了模型在时序相关任务上的表现。**

- **链接: [http://arxiv.org/pdf/2506.03340v1](http://arxiv.org/pdf/2506.03340v1)**

> **作者:** Zihui Xue; Mi Luo; Kristen Grauman
>
> **备注:** Project website: https://vision.cs.utexas.edu/projects/SeeAoT
>
> **摘要:** The Arrow of Time (AoT)-time's irreversible flow shaping physical events-is fundamental to video comprehension, yet remains a significant challenge for modern large multimodal models (LMMs). Current LMMs struggle to perceive and utilize temporal directionality in video when responding to language queries, obstructing deeper temporal understanding. We tackle this deficiency by first providing a critical analysis of existing benchmarks and models. We then introduce ArrowRL, a reinforcement learning (RL)-based training strategy with an innovative reverse reward that instills AoT awareness by encouraging divergent video interpretations between forward and reversed visual frames. For rigorous evaluation, we additionally develop AoTBench, a new multi-faceted benchmark probing temporally challenging questions. Experiments show ArrowRL greatly advances temporal perception: it not only achieves substantial improvements on our challenging AoTBench but also demonstrably boosts performance on standard video question answering (VQA) benchmarks (with peak accuracy gains reaching over 20% and 10% respectively). This validates ArrowRL's effectiveness and highlights the critical need for dedicated AoT understanding in LMMs.
>
---
#### [new 079] Sounding that Object: Interactive Object-Aware Image to Audio Generation
- **分类: cs.CV; cs.LG; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于图像到音频生成任务，旨在解决复杂场景中多对象声音生成的问题。作者提出了一种交互式物体感知的音频生成模型，通过结合物体中心学习和条件扩散模型，使用户能基于图像中的特定物体生成对应声音，并验证了其注意力机制的有效性。**

- **链接: [http://arxiv.org/pdf/2506.04214v1](http://arxiv.org/pdf/2506.04214v1)**

> **作者:** Tingle Li; Baihe Huang; Xiaobin Zhuang; Dongya Jia; Jiawei Chen; Yuping Wang; Zhuo Chen; Gopala Anumanchipalli; Yuxuan Wang
>
> **备注:** ICML 2025
>
> **摘要:** Generating accurate sounds for complex audio-visual scenes is challenging, especially in the presence of multiple objects and sound sources. In this paper, we propose an {\em interactive object-aware audio generation} model that grounds sound generation in user-selected visual objects within images. Our method integrates object-centric learning into a conditional latent diffusion model, which learns to associate image regions with their corresponding sounds through multi-modal attention. At test time, our model employs image segmentation to allow users to interactively generate sounds at the {\em object} level. We theoretically validate that our attention mechanism functionally approximates test-time segmentation masks, ensuring the generated audio aligns with selected objects. Quantitative and qualitative evaluations show that our model outperforms baselines, achieving better alignment between objects and their associated sounds. Project page: https://tinglok.netlify.app/files/avobject/
>
---
#### [new 080] A Large-Scale Referring Remote Sensing Image Segmentation Dataset and Benchmark
- **分类: cs.CV**

- **简介: 该论文属于 referring remote sensing image segmentation（RRSIS）任务，旨在解决现有数据集分辨率低、场景多样性和类别覆盖不足的问题。作者构建了大规模数据集 NWPU-Refer，并提出 MRSNet 模型，通过 IFIM 和 HFIM 模块提升多尺度特征交互，实现了更优性能。**

- **链接: [http://arxiv.org/pdf/2506.03583v1](http://arxiv.org/pdf/2506.03583v1)**

> **作者:** Zhigang Yang; Huiguang Yao; Linmao Tian; Xuezhi Zhao; Qiang Li; Qi Wang
>
> **摘要:** Referring Remote Sensing Image Segmentation is a complex and challenging task that integrates the paradigms of computer vision and natural language processing. Existing datasets for RRSIS suffer from critical limitations in resolution, scene diversity, and category coverage, which hinders the generalization and real-world applicability of refer segmentation models. To facilitate the development of this field, we introduce NWPU-Refer, the largest and most diverse RRSIS dataset to date, comprising 15,003 high-resolution images (1024-2048px) spanning 30+ countries with 49,745 annotated targets supporting single-object, multi-object, and non-object segmentation scenarios. Additionally, we propose the Multi-scale Referring Segmentation Network (MRSNet), a novel framework tailored for the unique demands of RRSIS. MRSNet introduces two key innovations: (1) an Intra-scale Feature Interaction Module (IFIM) that captures fine-grained details within each encoder stage, and (2) a Hierarchical Feature Interaction Module (HFIM) to enable seamless cross-scale feature fusion, preserving spatial integrity while enhancing discriminative power. Extensive experiments conducte on the proposed NWPU-Refer dataset demonstrate that MRSNet achieves state-of-the-art performance across multiple evaluation metrics, validating its effectiveness. The dataset and code are publicly available at https://github.com/CVer-Yang/NWPU-Refer.
>
---
#### [new 081] BiXFormer: A Robust Framework for Maximizing Modality Effectiveness in Multi-Modal Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于多模态语义分割任务，旨在解决现有方法限制各模态优势发挥及对缺失模态敏感的问题。作者提出了BiXFormer框架，通过统一模态匹配和跨模态对齐策略，最大化各模态有效性，并提升了模型在不同模态输入下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.03675v1](http://arxiv.org/pdf/2506.03675v1)**

> **作者:** Jialei Chen; Xu Zheng; Danda Pani Paudel; Luc Van Gool; Hiroshi Murase; Daisuke Deguchi
>
> **摘要:** Utilizing multi-modal data enhances scene understanding by providing complementary semantic and geometric information. Existing methods fuse features or distill knowledge from multiple modalities into a unified representation, improving robustness but restricting each modality's ability to fully leverage its strengths in different situations. We reformulate multi-modal semantic segmentation as a mask-level classification task and propose BiXFormer, which integrates Unified Modality Matching (UMM) and Cross Modality Alignment (CMA) to maximize modality effectiveness and handle missing modalities. Specifically, BiXFormer first categorizes multi-modal inputs into RGB and X, where X represents any non-RGB modalities, e.g., depth, allowing separate processing for each. This design leverages the well-established pretraining for RGB, while addressing the relative lack of attention to X modalities. Then, we propose UMM, which includes Modality Agnostic Matching (MAM) and Complementary Matching (CM). MAM assigns labels to features from all modalities without considering modality differences, leveraging each modality's strengths. CM then reassigns unmatched labels to remaining unassigned features within their respective modalities, ensuring that each available modality contributes to the final prediction and mitigating the impact of missing modalities. Moreover, to further facilitate UMM, we introduce CMA, which enhances the weaker queries assigned in CM by aligning them with optimally matched queries from MAM. Experiments on both synthetic and real-world multi-modal benchmarks demonstrate the effectiveness of our method, achieving significant improvements in mIoU of +2.75% and +22.74% over the prior arts.
>
---
#### [new 082] BiMa: Towards Biases Mitigation for Text-Video Retrieval via Scene Element Guidance
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于文本-视频检索（TVR）任务，旨在缓解数据集中的视觉-语言偏差问题。作者提出BiMa框架，通过场景元素引导，分别在视觉和文本表示中进行去偏处理，以提升模型对关键细节的关注能力，并在多个基准上验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2506.03589v1](http://arxiv.org/pdf/2506.03589v1)**

> **作者:** Huy Le; Nhat Chung; Tung Kieu; Anh Nguyen; Ngan Le
>
> **备注:** 22 pages, 14 figures
>
> **摘要:** Text-video retrieval (TVR) systems often suffer from visual-linguistic biases present in datasets, which cause pre-trained vision-language models to overlook key details. To address this, we propose BiMa, a novel framework designed to mitigate biases in both visual and textual representations. Our approach begins by generating scene elements that characterize each video by identifying relevant entities/objects and activities. For visual debiasing, we integrate these scene elements into the video embeddings, enhancing them to emphasize fine-grained and salient details. For textual debiasing, we introduce a mechanism to disentangle text features into content and bias components, enabling the model to focus on meaningful content while separately handling biased information. Extensive experiments and ablation studies across five major TVR benchmarks (i.e., MSR-VTT, MSVD, LSMDC, ActivityNet, and DiDeMo) demonstrate the competitive performance of BiMa. Additionally, the model's bias mitigation capability is consistently validated by its strong results on out-of-distribution retrieval tasks.
>
---
#### [new 083] Language-Image Alignment with Fixed Text Encoders
- **分类: cs.CV**

- **简介: 该论文属于多模态任务，旨在解决语言-图像对齐问题。传统方法如CLIP需联合训练文本和图像编码器，计算成本高。本文提出LIFT，仅训练图像编码器，使用预训练大语言模型作为固定文本编码器。实验表明，LIFT在多数场景下优于CLIP，尤其在处理长文本和复杂语义时，同时提升了计算效率。**

- **链接: [http://arxiv.org/pdf/2506.04209v1](http://arxiv.org/pdf/2506.04209v1)**

> **作者:** Jingfeng Yang; Ziyang Wu; Yue Zhao; Yi Ma
>
> **摘要:** Currently, the most dominant approach to establishing language-image alignment is to pre-train text and image encoders jointly through contrastive learning, such as CLIP and its variants. In this work, we question whether such a costly joint training is necessary. In particular, we investigate if a pre-trained fixed large language model (LLM) offers a good enough text encoder to guide visual representation learning. That is, we propose to learn Language-Image alignment with a Fixed Text encoder (LIFT) from an LLM by training only the image encoder. Somewhat surprisingly, through comprehensive benchmarking and ablation studies, we find that this much simplified framework LIFT is highly effective and it outperforms CLIP in most scenarios that involve compositional understanding and long captions, while achieving considerable gains in computational efficiency. Our work takes a first step towards systematically exploring how text embeddings from LLMs can guide visual learning and suggests an alternative design choice for learning language-aligned visual representations.
>
---
#### [new 084] Geometric Visual Fusion Graph Neural Networks for Multi-Person Human-Object Interaction Recognition in Videos
- **分类: cs.CV**

- **简介: 该论文属于视频中多人人-物交互识别任务，旨在解决视觉与几何特征融合困难及复杂场景下交互识别效果差的问题。作者提出了GeoVis-GNN模型，采用双注意力机制和实体图学习，逐步构建从个体特征到交互理解的表达，并构建了新数据集MPHOI-120以推动真实场景下的研究进展。**

- **链接: [http://arxiv.org/pdf/2506.03440v1](http://arxiv.org/pdf/2506.03440v1)**

> **作者:** Tanqiu Qiao; Ruochen Li; Frederick W. B. Li; Yoshiki Kubotani; Shigeo Morishima; Hubert P. H. Shum
>
> **备注:** Accepted by Expert Systems with Applications (ESWA)
>
> **摘要:** Human-Object Interaction (HOI) recognition in videos requires understanding both visual patterns and geometric relationships as they evolve over time. Visual and geometric features offer complementary strengths. Visual features capture appearance context, while geometric features provide structural patterns. Effectively fusing these multimodal features without compromising their unique characteristics remains challenging. We observe that establishing robust, entity-specific representations before modeling interactions helps preserve the strengths of each modality. Therefore, we hypothesize that a bottom-up approach is crucial for effective multimodal fusion. Following this insight, we propose the Geometric Visual Fusion Graph Neural Network (GeoVis-GNN), which uses dual-attention feature fusion combined with interdependent entity graph learning. It progressively builds from entity-specific representations toward high-level interaction understanding. To advance HOI recognition to real-world scenarios, we introduce the Concurrent Partial Interaction Dataset (MPHOI-120). It captures dynamic multi-person interactions involving concurrent actions and partial engagement. This dataset helps address challenges like complex human-object dynamics and mutual occlusions. Extensive experiments demonstrate the effectiveness of our method across various HOI scenarios. These scenarios include two-person interactions, single-person activities, bimanual manipulations, and complex concurrent partial interactions. Our method achieves state-of-the-art performance.
>
---
#### [new 085] Channel-adaptive Cross-modal Generative Semantic Communication for Point Cloud Transmission
- **分类: cs.CV; cs.NI**

- **简介: 该论文属于点云传输任务，旨在解决高效可靠传输点云的问题。提出了GenSeC-PC框架，采用跨模态语义通信与生成模型，融合图像与点云信息，实现高压缩效率和鲁棒重建，支持实时通信并适应不同信道条件。**

- **链接: [http://arxiv.org/pdf/2506.03211v1](http://arxiv.org/pdf/2506.03211v1)**

> **作者:** Wanting Yang; Zehui Xiong; Qianqian Yang; Ping Zhang; Merouane Debbah; Rahim Tafazolli
>
> **摘要:** With the rapid development of autonomous driving and extended reality, efficient transmission of point clouds (PCs) has become increasingly important. In this context, we propose a novel channel-adaptive cross-modal generative semantic communication (SemCom) for PC transmission, called GenSeC-PC. GenSeC-PC employs a semantic encoder that fuses images and point clouds, where images serve as non-transmitted side information. Meanwhile, the decoder is built upon the backbone of PointDif. Such a cross-modal design not only ensures high compression efficiency but also delivers superior reconstruction performance compared to PointDif. Moreover, to ensure robust transmission and reduce system complexity, we design a streamlined and asymmetric channel-adaptive joint semantic-channel coding architecture, where only the encoder needs the feedback of average signal-to-noise ratio (SNR) and available bandwidth. In addition, rectified denoising diffusion implicit models is employed to accelerate the decoding process to the millisecond level, enabling real-time PC communication. Unlike existing methods, GenSeC-PC leverages generative priors to ensure reliable reconstruction even from noisy or incomplete source PCs. More importantly, it supports fully analog transmission, improving compression efficiency by eliminating the need for error-free side information transmission common in prior SemCom approaches. Simulation results confirm the effectiveness of cross-modal semantic extraction and dual-metric guided fine-tuning, highlighting the framework's robustness across diverse conditions, including low SNR, bandwidth limitations, varying numbers of 2D images, and previously unseen objects.
>
---
#### [new 086] Video Deblurring with Deconvolution and Aggregation Networks
- **分类: cs.CV**

- **简介: 该论文属于视频去模糊任务，旨在解决现有方法未能充分利用邻近帧导致效果不佳的问题。作者提出了DAN网络，包含预处理、对齐反卷积和帧聚合三个子网络，有效利用邻近帧信息进行去模糊，取得了优于现有方法的效果。**

- **链接: [http://arxiv.org/pdf/2506.04054v1](http://arxiv.org/pdf/2506.04054v1)**

> **作者:** Giyong Choi; HyunWook Park
>
> **摘要:** In contrast to single-image deblurring, video deblurring has the advantage that neighbor frames can be utilized to deblur a target frame. However, existing video deblurring algorithms often fail to properly employ the neighbor frames, resulting in sub-optimal performance. In this paper, we propose a deconvolution and aggregation network (DAN) for video deblurring that utilizes the information of neighbor frames well. In DAN, both deconvolution and aggregation strategies are achieved through three sub-networks: the preprocessing network (PPN) and the alignment-based deconvolution network (ABDN) for the deconvolution scheme; the frame aggregation network (FAN) for the aggregation scheme. In the deconvolution part, blurry inputs are first preprocessed by the PPN with non-local operations. Then, the output frames from the PPN are deblurred by the ABDN based on the frame alignment. In the FAN, these deblurred frames from the deconvolution part are combined into a latent frame according to reliability maps which infer pixel-wise sharpness. The proper combination of three sub-networks can achieve favorable performance on video deblurring by using the neighbor frames suitably. In experiments, the proposed DAN was demonstrated to be superior to existing state-of-the-art methods through both quantitative and qualitative evaluations on the public datasets.
>
---
#### [new 087] PlückeRF: A Line-based 3D Representation for Few-view Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决少视角输入下重建质量不足的问题。作者提出PlückeRF，一种基于线结构的3D表示方法，通过连接像素射线与3D特征，实现多视角信息的有效融合，提升了重建效果。**

- **链接: [http://arxiv.org/pdf/2506.03713v1](http://arxiv.org/pdf/2506.03713v1)**

> **作者:** Sam Bahrami; Dylan Campbell
>
> **摘要:** Feed-forward 3D reconstruction methods aim to predict the 3D structure of a scene directly from input images, providing a faster alternative to per-scene optimization approaches. Significant progress has been made in single-view and few-view reconstruction using learned priors that infer object shape and appearance, even for unobserved regions. However, there is substantial potential to enhance these methods by better leveraging information from multiple views when available. To address this, we propose a few-view reconstruction model that more effectively harnesses multi-view information. Our approach introduces a simple mechanism that connects the 3D representation with pixel rays from the input views, allowing for preferential sharing of information between nearby 3D locations and between 3D locations and nearby pixel rays. We achieve this by defining the 3D representation as a set of structured, feature-augmented lines; the Pl\"uckeRF representation. Using this representation, we demonstrate improvements in reconstruction quality over the equivalent triplane representation and state-of-the-art feedforward reconstruction methods.
>
---
#### [new 088] Video-Skill-CoT: Skill-based Chain-of-Thoughts for Domain-Adaptive Video Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频理解任务，旨在解决现有方法在领域特定技能（如事件检测、情感理解）上的适应性不足问题。作者提出了Video-SKoT框架，通过构建基于技能的CoT标注和技能专用专家学习模型，提升跨领域视频推理能力，并在多个基准上验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2506.03525v1](http://arxiv.org/pdf/2506.03525v1)**

> **作者:** Daeun Lee; Jaehong Yoon; Jaemin Cho; Mohit Bansal
>
> **备注:** Project website: https://video-skill-cot.github.io/
>
> **摘要:** Recent advances in Chain-of-Thought (CoT) reasoning have improved complex video understanding, but existing methods often struggle to adapt to domain-specific skills (e.g., event detection, spatial relation understanding, emotion understanding) over various video content. To address this, we propose Video-Skill-CoT (a.k.a. Video-SKoT), a framework that automatically constructs and leverages skill-aware CoT supervisions for domain-adaptive video reasoning. First, we construct skill-based CoT annotations: we extract domain-relevant reasoning skills from training questions, cluster them into a shared skill taxonomy, and create detailed multi-step CoT rationale tailored to each video-question pair for training. Second, we introduce a skill-specific expert learning framework. Each expert module specializes in a subset of reasoning skills and is trained with lightweight adapters using the collected CoT supervision. We demonstrate the effectiveness of the proposed approach on three video understanding benchmarks, where Video-SKoT consistently outperforms strong baselines. We also provide in-depth analyses on comparing different CoT annotation pipelines and learned skills over multiple video domains.
>
---
#### [new 089] CHIME: Conditional Hallucination and Integrated Multi-scale Enhancement for Time Series Diffusion Model
- **分类: cs.CV; cs.SY; eess.SY**

- **简介: 该论文属于时间序列生成任务，旨在解决现有扩散模型在多尺度特征对齐和跨实体、长时间尺度生成能力不足的问题。作者提出了CHIME框架，通过多尺度分解、自适应集成和特征幻觉模块，提升生成效果与泛化能力，实验证明其在少样本场景下表现优异。**

- **链接: [http://arxiv.org/pdf/2506.03502v1](http://arxiv.org/pdf/2506.03502v1)**

> **作者:** Yuxuan Chen; Haipeng Xie
>
> **摘要:** The denoising diffusion probabilistic model has become a mainstream generative model, achieving significant success in various computer vision tasks. Recently, there has been initial exploration of applying diffusion models to time series tasks. However, existing studies still face challenges in multi-scale feature alignment and generative capabilities across different entities and long-time scales. In this paper, we propose CHIME, a conditional hallucination and integrated multi-scale enhancement framework for time series diffusion models. By employing multi-scale decomposition and adaptive integration, CHIME captures the decomposed features of time series, achieving in-domain distribution alignment between generated and original samples. In addition, we introduce a feature hallucination module in the conditional denoising process, enabling the transfer of temporal features through the training of category-independent transformation layers. Experimental results on publicly available real-world datasets demonstrate that CHIME achieves state-of-the-art performance and exhibits excellent generative generalization capabilities in few-shot scenarios.
>
---
#### [new 090] Human Fall Detection using Transfer Learning-based 3D CNN
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视频行为识别任务，旨在解决老年人意外跌倒检测问题。利用预训练的3D CNN模型提取时空特征，通过迁移学习与SVM分类器结合，实现对“跌倒”与“日常活动（ADL）”的准确区分，并在两个数据集上进行了实验验证。**

- **链接: [http://arxiv.org/pdf/2506.03193v1](http://arxiv.org/pdf/2506.03193v1)**

> **作者:** Ekram Alam; Abu Sufian; Paramartha Dutta; Marco Leo
>
> **摘要:** Unintentional or accidental falls are one of the significant health issues in senior persons. The population of senior persons is increasing steadily. So, there is a need for an automated fall detection monitoring system. This paper introduces a vision-based fall detection system using a pre-trained 3D CNN. Unlike 2D CNN, 3D CNN extracts not only spatial but also temporal features. The proposed model leverages the original learned weights of a 3D CNN model pre-trained on the Sports1M dataset to extract the spatio-temporal features. Only the SVM classifier was trained, which saves the time required to train the 3D CNN. Stratified shuffle five split cross-validation has been used to split the dataset into training and testing data. Extracted features from the proposed 3D CNN model were fed to an SVM classifier to classify the activity as fall or ADL. Two datasets, GMDCSA and CAUCAFall, were utilized to conduct the experiment. The source code for this work can be accessed via the following link: https://github.com/ekramalam/HFD_3DCNN.
>
---
#### [new 091] Heterogeneous Skeleton-Based Action Representation Learning
- **分类: cs.CV**

- **简介: 该论文属于动作识别任务，旨在解决不同结构骨架数据带来的异质性问题。现有方法多针对同质骨架，忽视了实际中骨架来源多样、结构不一的情况。论文提出一种异质骨架动作表征学习框架，通过将二维骨架转化为三维，并引入语义运动编码和统一表征学习模块，实现对多种骨架数据的有效动作识别。**

- **链接: [http://arxiv.org/pdf/2506.03481v1](http://arxiv.org/pdf/2506.03481v1)**

> **作者:** Hongsong Wang; Xiaoyan Ma; Jidong Kuang; Jie Gui
>
> **备注:** To appear in CVPR 2025
>
> **摘要:** Skeleton-based human action recognition has received widespread attention in recent years due to its diverse range of application scenarios. Due to the different sources of human skeletons, skeleton data naturally exhibit heterogeneity. The previous works, however, overlook the heterogeneity of human skeletons and solely construct models tailored for homogeneous skeletons. This work addresses the challenge of heterogeneous skeleton-based action representation learning, specifically focusing on processing skeleton data that varies in joint dimensions and topological structures. The proposed framework comprises two primary components: heterogeneous skeleton processing and unified representation learning. The former first converts two-dimensional skeleton data into three-dimensional skeleton via an auxiliary network, and then constructs a prompted unified skeleton using skeleton-specific prompts. We also design an additional modality named semantic motion encoding to harness the semantic information within skeletons. The latter module learns a unified action representation using a shared backbone network that processes different heterogeneous skeletons. Extensive experiments on the NTU-60, NTU-120, and PKU-MMD II datasets demonstrate the effectiveness of our method in various tasks of action understanding. Our approach can be applied to action recognition in robots with different humanoid structures.
>
---
#### [new 092] Point Cloud Quality Assessment Using the Perceptual Clustering Weighted Graph (PCW-Graph) and Attention Fusion Network
- **分类: cs.CV**

- **简介: 该论文属于无参考点云质量评估任务，旨在解决缺乏参考模型时3D内容质量评估的问题。作者提出了一种基于感知聚类加权图（PCW-Graph）和注意力融合网络的方法，用于更准确地评估点云的质量。**

- **链接: [http://arxiv.org/pdf/2506.04081v1](http://arxiv.org/pdf/2506.04081v1)**

> **作者:** Abdelouahed Laazoufi; Mohammed El Hassouni; Hocine Cherifi
>
> **摘要:** No-Reference Point Cloud Quality Assessment (NR-PCQA) is critical for evaluating 3D content in real-world applications where reference models are unavailable.
>
---
#### [new 093] LayerFlow: A Unified Model for Layer-aware Video Generation
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决层感知视频生成问题。论文提出LayerFlow模型，通过层嵌入区分不同图层及其提示，统一生成前景、背景及合成视频。针对缺乏高质量分层视频数据的问题，设计了多阶段训练策略，结合低质量视频和高质量静态图像进行训练，实现生成效果良好的层感知视频。**

- **链接: [http://arxiv.org/pdf/2506.04228v1](http://arxiv.org/pdf/2506.04228v1)**

> **作者:** Sihui Ji; Hao Luo; Xi Chen; Yuanpeng Tu; Yiyang Wang; Hengshuang Zhao
>
> **备注:** Project Page: https://sihuiji.github.io/LayerFlow-Page/
>
> **摘要:** We present LayerFlow, a unified solution for layer-aware video generation. Given per-layer prompts, LayerFlow generates videos for the transparent foreground, clean background, and blended scene. It also supports versatile variants like decomposing a blended video or generating the background for the given foreground and vice versa. Starting from a text-to-video diffusion transformer, we organize the videos for different layers as sub-clips, and leverage layer embeddings to distinguish each clip and the corresponding layer-wise prompts. In this way, we seamlessly support the aforementioned variants in one unified framework. For the lack of high-quality layer-wise training videos, we design a multi-stage training strategy to accommodate static images with high-quality layer annotations. Specifically, we first train the model with low-quality video data. Then, we tune a motion LoRA to make the model compatible with static frames. Afterward, we train the content LoRA on the mixture of image data with high-quality layered images along with copy-pasted video data. During inference, we remove the motion LoRA thus generating smooth videos with desired layers.
>
---
#### [new 094] Pre-trained Vision-Language Models Assisted Noisy Partial Label Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于噪声部分标签学习任务，旨在解决预训练视觉-语言模型生成的实例相关噪声标注问题。作者提出了一种协作一致性正则化方法（Co-Reg），通过协同训练两个网络实现标签净化，并结合少量人工标注提升性能，验证了弱监督学习与知识蒸馏结合的潜力。**

- **链接: [http://arxiv.org/pdf/2506.03229v1](http://arxiv.org/pdf/2506.03229v1)**

> **作者:** Qian-Wei Wang; Yuqiu Xie; Letian Zhang; Zimo Liu; Shu-Tao Xia
>
> **摘要:** In the context of noisy partial label learning (NPLL), each training sample is associated with a set of candidate labels annotated by multiple noisy annotators. With the emergence of high-performance pre-trained vision-language models (VLMs) such as CLIP, LLaVa and GPT-4V, the direction of using these models to replace time-consuming manual annotation workflows and achieve "manual-annotation-free" training for downstream tasks has become a highly promising research avenue. This paper focuses on learning from noisy partial labels annotated by pre-trained VLMs and proposes an innovative collaborative consistency regularization (Co-Reg) method. Unlike the symmetric noise primarily addressed in traditional noisy label learning, the noise generated by pre-trained models is instance-dependent, embodying the underlying patterns of the pre-trained models themselves, which significantly increases the learning difficulty for the model. To address this, we simultaneously train two neural networks that implement collaborative purification of training labels through a "Co-Pseudo-Labeling" mechanism, while enforcing consistency regularization constraints in both the label space and feature representation space. Our method can also leverage few-shot manually annotated valid labels to further enhance its performances. Comparative experiments with different denoising and disambiguation algorithms, annotation manners, and pre-trained model application schemes fully validate the effectiveness of the proposed method, while revealing the broad prospects of integrating weakly-supervised learning techniques into the knowledge distillation process of pre-trained models.
>
---
#### [new 095] MMR-V: What's Left Unsaid? A Benchmark for Multimodal Deep Reasoning in Videos
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视频多模态推理任务，旨在解决现有模型在长程、多帧推理上的不足。作者构建了MMR-V基准，包含317个视频和1257个任务，强调需跨多帧推理及理解隐藏信息。实验表明当前模型表现不佳，最佳模型仅达52.5%准确率，并指出多模态推理与文本推理的不同之处。**

- **链接: [http://arxiv.org/pdf/2506.04141v1](http://arxiv.org/pdf/2506.04141v1)**

> **作者:** Kejian Zhu; Zhuoran Jin; Hongbang Yuan; Jiachun Li; Shangqing Tu; Pengfei Cao; Yubo Chen; Kang Liu; Jun Zhao
>
> **备注:** Project Page: https://mmr-v.github.io
>
> **摘要:** The sequential structure of videos poses a challenge to the ability of multimodal large language models (MLLMs) to locate multi-frame evidence and conduct multimodal reasoning. However, existing video benchmarks mainly focus on understanding tasks, which only require models to match frames mentioned in the question (hereafter referred to as "question frame") and perceive a few adjacent frames. To address this gap, we propose MMR-V: A Benchmark for Multimodal Deep Reasoning in Videos. The benchmark is characterized by the following features. (1) Long-range, multi-frame reasoning: Models are required to infer and analyze evidence frames that may be far from the question frame. (2) Beyond perception: Questions cannot be answered through direct perception alone but require reasoning over hidden information. (3) Reliability: All tasks are manually annotated, referencing extensive real-world user understanding to align with common perceptions. (4) Confusability: Carefully designed distractor annotation strategies to reduce model shortcuts. MMR-V consists of 317 videos and 1,257 tasks. Our experiments reveal that current models still struggle with multi-modal reasoning; even the best-performing model, o4-mini, achieves only 52.5% accuracy. Additionally, current reasoning enhancement strategies (Chain-of-Thought and scaling test-time compute) bring limited gains. Further analysis indicates that the CoT demanded for multi-modal reasoning differs from it in textual reasoning, which partly explains the limited performance gains. We hope that MMR-V can inspire further research into enhancing multi-modal reasoning capabilities.
>
---
#### [new 096] Impact of Tuning Parameters in Deep Convolutional Neural Network Using a Crack Image Dataset
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 论文研究了深度卷积神经网络（DCNN）中调参对分类性能的影响，使用包含裂缝图像数据集的二分类任务。重点分析了池化方法、激活函数和优化器参数的影响。实验表明，采用maxpooling、adam优化器和tanh激活函数效果更优。**

- **链接: [http://arxiv.org/pdf/2506.03184v1](http://arxiv.org/pdf/2506.03184v1)**

> **作者:** Mahe Zabin; Ho-Jin Choi; Md. Monirul Islam; Jia Uddin
>
> **备注:** 8 pages, 2 figures, published at Proceedings of the 15th KIPS International Conference on Ubiquitous Information Technologies and Applications (CUTE 2021), Jeju, Repubilc of Korea
>
> **摘要:** The performance of a classifier depends on the tuning of its parame ters. In this paper, we have experimented the impact of various tuning parameters on the performance of a deep convolutional neural network (DCNN). In the ex perimental evaluation, we have considered a DCNN classifier that consists of 2 convolutional layers (CL), 2 pooling layers (PL), 1 dropout, and a dense layer. To observe the impact of pooling, activation function, and optimizer tuning pa rameters, we utilized a crack image dataset having two classes: negative and pos itive. The experimental results demonstrate that with the maxpooling, the DCNN demonstrates its better performance for adam optimizer and tanh activation func tion.
>
---
#### [new 097] EV-Flying: an Event-based Dataset for In-The-Wild Recognition of Flying Objects
- **分类: cs.CV**

- **简介: 该论文属于飞行物体识别任务，旨在解决传统RGB方法在识别小型快速移动物体（如昆虫和无人机）时的模糊、尺度变化等问题。论文提出EV-Flying事件数据集，采用基于点云的轻量模型处理异步事件流，提升野外环境下飞行物识别效率与准确性。**

- **链接: [http://arxiv.org/pdf/2506.04048v1](http://arxiv.org/pdf/2506.04048v1)**

> **作者:** Gabriele Magrini; Federico Becattini; Giovanni Colombo; Pietro Pala
>
> **摘要:** Monitoring aerial objects is crucial for security, wildlife conservation, and environmental studies. Traditional RGB-based approaches struggle with challenges such as scale variations, motion blur, and high-speed object movements, especially for small flying entities like insects and drones. In this work, we explore the potential of event-based vision for detecting and recognizing flying objects, in particular animals that may not follow short and long-term predictable patters. Event cameras offer high temporal resolution, low latency, and robustness to motion blur, making them well-suited for this task. We introduce EV-Flying, an event-based dataset of flying objects, comprising manually annotated birds, insects and drones with spatio-temporal bounding boxes and track identities. To effectively process the asynchronous event streams, we employ a point-based approach leveraging lightweight architectures inspired by PointNet. Our study investigates the classification of flying objects using point cloud-based event representations. The proposed dataset and methodology pave the way for more efficient and reliable aerial object recognition in real-world scenarios.
>
---
#### [new 098] ConMamba: Contrastive Vision Mamba for Plant Disease Detection
- **分类: cs.CV**

- **简介: 论文属于植物病害检测任务，旨在解决现有方法依赖大量标注数据、计算成本高及特征对齐效果差的问题。作者提出ConMamba框架，结合视觉Mamba编码器与双层级对比损失，实现高效自监督学习，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2506.03213v1](http://arxiv.org/pdf/2506.03213v1)**

> **作者:** Abdullah Al Mamun; Miaohua Zhang; David Ahmedt-Aristizabal; Zeeshan Hayder; Mohammad Awrangjeb
>
> **摘要:** Plant Disease Detection (PDD) is a key aspect of precision agriculture. However, existing deep learning methods often rely on extensively annotated datasets, which are time-consuming and costly to generate. Self-supervised Learning (SSL) offers a promising alternative by exploiting the abundance of unlabeled data. However, most existing SSL approaches suffer from high computational costs due to convolutional neural networks or transformer-based architectures. Additionally, they struggle to capture long-range dependencies in visual representation and rely on static loss functions that fail to align local and global features effectively. To address these challenges, we propose ConMamba, a novel SSL framework specially designed for PDD. ConMamba integrates the Vision Mamba Encoder (VME), which employs a bidirectional State Space Model (SSM) to capture long-range dependencies efficiently. Furthermore, we introduce a dual-level contrastive loss with dynamic weight adjustment to optimize local-global feature alignment. Experimental results on three benchmark datasets demonstrate that ConMamba significantly outperforms state-of-the-art methods across multiple evaluation metrics. This provides an efficient and robust solution for PDD.
>
---
#### [new 099] Multimodal Generative AI with Autoregressive LLMs for Human Motion Understanding and Generation: A Way Forward
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本驱动的人体动作生成任务，旨在解决如何利用文本描述生成复杂、逼真的人体运动序列。论文综述了基于自回归大语言模型和多模态生成AI的方法，分析了不同模型（如扩散模型、GANs、VAEs、Transformer）在动作质量、效率和适应性方面的优劣，并探讨了其在医疗、游戏、动画等领域的应用潜力与挑战。**

- **链接: [http://arxiv.org/pdf/2506.03191v1](http://arxiv.org/pdf/2506.03191v1)**

> **作者:** Muhammad Islam; Tao Huang; Euijoon Ahn; Usman Naseem
>
> **摘要:** This paper presents an in-depth survey on the use of multimodal Generative Artificial Intelligence (GenAI) and autoregressive Large Language Models (LLMs) for human motion understanding and generation, offering insights into emerging methods, architectures, and their potential to advance realistic and versatile motion synthesis. Focusing exclusively on text and motion modalities, this research investigates how textual descriptions can guide the generation of complex, human-like motion sequences. The paper explores various generative approaches, including autoregressive models, diffusion models, Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and transformer-based models, by analyzing their strengths and limitations in terms of motion quality, computational efficiency, and adaptability. It highlights recent advances in text-conditioned motion generation, where textual inputs are used to control and refine motion outputs with greater precision. The integration of LLMs further enhances these models by enabling semantic alignment between instructions and motion, improving coherence and contextual relevance. This systematic survey underscores the transformative potential of text-to-motion GenAI and LLM architectures in applications such as healthcare, humanoids, gaming, animation, and assistive technologies, while addressing ongoing challenges in generating efficient and realistic human motion.
>
---
#### [new 100] UniCUE: Unified Recognition and Generation Framework for Chinese Cued Speech Video-to-Speech Generation
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 论文提出UniCUE，统一中文手势语音视频到语音生成框架，解决因数据不足和中间文本依赖导致的语音生成误差与同步问题。**

- **链接: [http://arxiv.org/pdf/2506.04134v1](http://arxiv.org/pdf/2506.04134v1)**

> **作者:** Jinting Wang; Shan Yang; Li Liu
>
> **备注:** 10 pages, 10 figures
>
> **摘要:** Cued Speech (CS) enhances lipreading through hand coding, providing precise speech perception support for the hearing-impaired. CS Video-to-Speech generation (CSV2S) task aims to convert the CS visual expressions (CS videos) of hearing-impaired individuals into comprehensible speech signals. Direct generation of speech from CS video (called single CSV2S) yields poor performance due to insufficient CS data. Current research mostly focuses on CS Recognition (CSR), which convert video content into linguistic text. Based on this, one straightforward way of CSV2S is to combine CSR with a Text-to-Speech system. This combined architecture relies on text as an intermediate medium for stepwise cross-modal alignment, which may lead to error propagation and temporal misalignment between speech and video dynamics. To address these challenges, we propose a novel approach that directly generates speech from CS videos without relying on intermediate text. Building upon this, we propose UniCUE, the first unified framework for CSV2S, whose core innovation lies in the integration of the CSR task that provides fine-grained visual-semantic information to facilitate speech generation from CS videos. More precisely, (1) a novel fine-grained semantic alignment pool to ensure precise mapping between visual features and speech contents; (2) a VisioPhonetic adapter to bridge cross-task representations, ensuring seamless compatibility between two distinct tasks (i.e., CSV2S and CSR); (3) a pose-aware visual processor is introduced to enhance fine-grained spatiotemporal correlations between lip and hand movements in CS video. Experiments on our new established Chinese CS dataset (14 cuers1: 8 hearing-impaired and 6 normal-hearing) show that our UniCUE significantly reduces Word Error Rate by 78.3% and improves lip-speech synchronization by 32% compared to the single CSV2S.
>
---
#### [new 101] Robust Neural Rendering in the Wild with Asymmetric Dual 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决真实场景中因光照不一致和干扰物导致的重建不稳定问题。作者提出Asymmetric Dual 3DGS方法，通过并行训练两个3D高斯溅射模型并引入一致性约束与差异化掩码策略，抑制不稳定伪影。此外，还设计了轻量级Dynamic EMA Proxy提升训练效率。实验表明该方法在真实数据集上表现优异且高效。**

- **链接: [http://arxiv.org/pdf/2506.03538v1](http://arxiv.org/pdf/2506.03538v1)**

> **作者:** Chengqi Li; Zhihao Shi; Yangdi Lu; Wenbo He; Xiangyu Xu
>
> **摘要:** 3D reconstruction from in-the-wild images remains a challenging task due to inconsistent lighting conditions and transient distractors. Existing methods typically rely on heuristic strategies to handle the low-quality training data, which often struggle to produce stable and consistent reconstructions, frequently resulting in visual artifacts. In this work, we propose Asymmetric Dual 3DGS, a novel framework that leverages the stochastic nature of these artifacts: they tend to vary across different training runs due to minor randomness. Specifically, our method trains two 3D Gaussian Splatting (3DGS) models in parallel, enforcing a consistency constraint that encourages convergence on reliable scene geometry while suppressing inconsistent artifacts. To prevent the two models from collapsing into similar failure modes due to confirmation bias, we introduce a divergent masking strategy that applies two complementary masks: a multi-cue adaptive mask and a self-supervised soft mask, which leads to an asymmetric training process of the two models, reducing shared error modes. In addition, to improve the efficiency of model training, we introduce a lightweight variant called Dynamic EMA Proxy, which replaces one of the two models with a dynamically updated Exponential Moving Average (EMA) proxy, and employs an alternating masking strategy to preserve divergence. Extensive experiments on challenging real-world datasets demonstrate that our method consistently outperforms existing approaches while achieving high efficiency. Codes and trained models will be released.
>
---
#### [new 102] Toward Reliable VLM: A Fine-Grained Benchmark and Framework for Exposure, Bias, and Inference in Korean Street Views
- **分类: cs.CV**

- **简介: 该论文属于图像-语言模型（VLM）在地理定位任务中的隐私与偏差分析。旨在解决现有基准过于粗糙、存在语言偏见，缺乏多模态和隐私意识评估的问题。作者构建了KoreaGEO Bench，包含1,080张韩国街景图像及多上下文标注，提出三路径评估协议，分析10种主流VLM在不同输入下的定位精度、空间偏见与推理行为。**

- **链接: [http://arxiv.org/pdf/2506.03371v1](http://arxiv.org/pdf/2506.03371v1)**

> **作者:** Xiaonan Wang; Bo Shao; Hansaem Kim
>
> **摘要:** Recent advances in vision-language models (VLMs) have enabled accurate image-based geolocation, raising serious concerns about location privacy risks in everyday social media posts. However, current benchmarks remain coarse-grained, linguistically biased, and lack multimodal and privacy-aware evaluations. To address these gaps, we present KoreaGEO Bench, the first fine-grained, multimodal geolocation benchmark for Korean street views. Our dataset comprises 1,080 high-resolution images sampled across four urban clusters and nine place types, enriched with multi-contextual annotations and two styles of Korean captions simulating real-world privacy exposure. We introduce a three-path evaluation protocol to assess ten mainstream VLMs under varying input modalities and analyze their accuracy, spatial bias, and reasoning behavior. Results reveal modality-driven shifts in localization precision and highlight structural prediction biases toward core cities.
>
---
#### [new 103] Zero-Shot Temporal Interaction Localization for Egocentric Videos
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视频行为定位任务，旨在解决第一视角视频中零样本下的人-物交互动作时间定位问题。现有方法依赖标注数据、泛化性差，而本文提出EgoLoc，利用视觉语言模型与自适应采样策略，结合2D/3D信息和闭环反馈，提升定位准确性与时效性。**

- **链接: [http://arxiv.org/pdf/2506.03662v1](http://arxiv.org/pdf/2506.03662v1)**

> **作者:** Erhang Zhang; Junyi Ma; Yin-Dong Zheng; Yixuan Zhou; Hesheng Wang
>
> **摘要:** Locating human-object interaction (HOI) actions within video serves as the foundation for multiple downstream tasks, such as human behavior analysis and human-robot skill transfer. Current temporal action localization methods typically rely on annotated action and object categories of interactions for optimization, which leads to domain bias and low deployment efficiency. Although some recent works have achieved zero-shot temporal action localization (ZS-TAL) with large vision-language models (VLMs), their coarse-grained estimations and open-loop pipelines hinder further performance improvements for temporal interaction localization (TIL). To address these issues, we propose a novel zero-shot TIL approach dubbed EgoLoc to locate the timings of grasp actions for human-object interaction in egocentric videos. EgoLoc introduces a self-adaptive sampling strategy to generate reasonable visual prompts for VLM reasoning. By absorbing both 2D and 3D observations, it directly samples high-quality initial guesses around the possible contact/separation timestamps of HOI according to 3D hand velocities, leading to high inference accuracy and efficiency. In addition, EgoLoc generates closed-loop feedback from visual and dynamic cues to further refine the localization results. Comprehensive experiments on the publicly available dataset and our newly proposed benchmark demonstrate that EgoLoc achieves better temporal interaction localization for egocentric videos compared to state-of-the-art baselines. We will release our code and relevant data as open-source at https://github.com/IRMVLab/EgoLoc.
>
---
#### [new 104] MINT: Memory-Infused Prompt Tuning at Test-time for CLIP
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型的测试时自适应任务，旨在提升模型在分布偏移下的泛化能力。论文提出MINT方法，通过引入记忆提示库（MPB）存储可学习提示对，利用测试图像的层次特征检索并构建关联提示，动态调整图像编码器。同时结合可学习文本提示，实现无需源数据或重新训练的快速精确自适应。**

- **链接: [http://arxiv.org/pdf/2506.03190v1](http://arxiv.org/pdf/2506.03190v1)**

> **作者:** Jiaming Yi; Ruirui Pan; Jishen Yang; Xiulong Yang
>
> **备注:** 14 pages, 3 figures
>
> **摘要:** Improving the generalization ability of Vision-Language Pre-trained Models (VLMs) under test-time data distribution shifts remains a critical challenge. The existing Test-Time Adaptation (TTA) methods fall short in fully leveraging the model's internal knowledge, particularly in dynamically adapting to complex and hierarchical visual semantic information. In this paper, we propose Memory-Infused Prompt Tuning (MINT), a novel framework to address this issue. Inspired by human associative memory theory, MINT introduces a Memory Prompt Bank (MPB), which stores learnable key-value prompt pairs that work as a memory of previously seen samples. During the test time, relevant prompt pairs in the MPB are retrieved by the hierarchical visual features of test images to dynamically assemble Associative Prompts. The associative prompts are then injected into the image encoder for fine-grained, customized visual contextual guidance. MINT also utilizes learnable text prompts. MINT thus enables rapid, precise VLM adaptation at test time by leveraging this MPB-acquired memory, without source data or retraining. The code is available at https://github.com/Jamieyi2004/MINT.
>
---
#### [new 105] PRJ: Perception-Retrieval-Judgement for Generated Images
- **分类: cs.CV**

- **简介: 该论文属于图像安全检测任务，旨在解决现有系统难以识别上下文相关及隐性有害内容的问题。作者提出PRJ框架，通过感知、检索、判断三阶段结构，结合外部知识与规则推理，提升对显性和隐性危害的检测精度与解释性，并引入动态评分机制量化毒性风险。**

- **链接: [http://arxiv.org/pdf/2506.03683v1](http://arxiv.org/pdf/2506.03683v1)**

> **作者:** Qiang Fu; Zonglei Jing; Zonghao Ying; Xiaoqian Li
>
> **摘要:** The rapid progress of generative AI has enabled remarkable creative capabilities, yet it also raises urgent concerns regarding the safety of AI-generated visual content in real-world applications such as content moderation, platform governance, and digital media regulation. This includes unsafe material such as sexually explicit images, violent scenes, hate symbols, propaganda, and unauthorized imitations of copyrighted artworks. Existing image safety systems often rely on rigid category filters and produce binary outputs, lacking the capacity to interpret context or reason about nuanced, adversarially induced forms of harm. In addition, standard evaluation metrics (e.g., attack success rate) fail to capture the semantic severity and dynamic progression of toxicity. To address these limitations, we propose Perception-Retrieval-Judgement (PRJ), a cognitively inspired framework that models toxicity detection as a structured reasoning process. PRJ follows a three-stage design: it first transforms an image into descriptive language (perception), then retrieves external knowledge related to harm categories and traits (retrieval), and finally evaluates toxicity based on legal or normative rules (judgement). This language-centric structure enables the system to detect both explicit and implicit harms with improved interpretability and categorical granularity. In addition, we introduce a dynamic scoring mechanism based on a contextual toxicity risk matrix to quantify harmfulness across different semantic dimensions. Experiments show that PRJ surpasses existing safety checkers in detection accuracy and robustness while uniquely supporting structured category-level toxicity interpretation.
>
---
#### [new 106] Unlabeled Data Improves Fine-Grained Image Zero-shot Classification with Multimodal LLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于细粒度图像零样本分类任务，旨在解决多模态大语言模型（MLLM）在无监督情况下难以捕捉细微视觉差异的问题。作者提出了AutoSEP框架，利用无标签数据迭代优化描述提示，提升分类性能，无需训练或微调MLLM。**

- **链接: [http://arxiv.org/pdf/2506.03195v1](http://arxiv.org/pdf/2506.03195v1)**

> **作者:** Yunqi Hong; Sohyun An; Andrew Bai; Neil Y. C. Lin; Cho-Jui Hsieh
>
> **摘要:** Despite Multimodal Large Language Models (MLLMs) showing promising results on general zero-shot image classification tasks, fine-grained image classification remains challenging. It demands precise attention to subtle visual details to distinguish between visually similar subcategories--details that MLLMs may easily overlook without explicit guidance. To address this, we introduce AutoSEP, an iterative self-supervised prompt learning framework designed to enhance MLLM fine-grained classification capabilities in a fully unsupervised manner. Our core idea is to leverage unlabeled data to learn a description prompt that guides MLLMs in identifying crucial discriminative features within an image, and boosts classification accuracy. We developed an automatic self-enhancing prompt learning framework called AutoSEP to iteratively improve the description prompt using unlabeled data, based on instance-level classification scoring function. AutoSEP only requires black-box access to MLLMs, eliminating the need for any training or fine-tuning. We evaluate our approach on multiple fine-grained classification datasets. It consistently outperforms other unsupervised baselines, demonstrating the effectiveness of our self-supervised optimization framework. Notably, AutoSEP on average improves 13 percent over standard zero-shot classification and 5 percent over the best-performing baselines. Code is available at: https://github.com/yq-hong/AutoSEP
>
---
#### [new 107] Infinity Parser: Layout Aware Reinforcement Learning for Scanned Document Parsing
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于文档解析任务，旨在解决扫描文档结构化解析的瓶颈问题。通过提出layoutRL强化学习框架和构建Infinity-Parser模型，实现对文档布局的感知与理解。论文工作包括发布新数据集Infinity-Doc-55K，并在多语言基准上验证了模型在OCR、表格公式提取及阅读顺序检测中的优越性能。**

- **链接: [http://arxiv.org/pdf/2506.03197v1](http://arxiv.org/pdf/2506.03197v1)**

> **作者:** Baode Wang; Biao Wu; Weizhen Li; Meng Fang; Yanjie Liang; Zuming Huang; Haozhe Wang; Jun Huang; Ling Chen; Wei Chu; Yuan Qi
>
> **备注:** 16 pages, 12 figures
>
> **摘要:** Automated parsing of scanned documents into richly structured, machine-readable formats remains a critical bottleneck in Document AI, as traditional multi-stage pipelines suffer from error propagation and limited adaptability to diverse layouts. We introduce layoutRL, an end-to-end reinforcement learning framework that trains models to be explicitly layout-aware by optimizing a composite reward of normalized edit distance, paragraph count accuracy, and reading order preservation. Leveraging our newly released dataset, Infinity-Doc-55K, which combines 55K high-fidelity synthetic scanned document parsing data with expert-filtered real-world documents, we instantiate layoutRL in a vision-language-model-based parser called Infinity-Parser. Evaluated on English and Chinese benchmarks for OCR, table and formula extraction, and reading order detection, Infinity-Parser achieves new state-of-the-art performance in both accuracy and structural fidelity, outpacing specialist pipelines and general-purpose vision-language models. We will publicly release our code and dataset to accelerate progress in robust document understanding.
>
---
#### [new 108] Intersectional Bias in Pre-Trained Image Recognition Models
- **分类: cs.CV; cs.CY; cs.HC; cs.LG**

- **简介: 该论文研究预训练图像识别模型中的交叉偏差问题，属于计算机视觉与公平性分析任务。旨在揭示模型在年龄、种族和性别交叉属性上的潜在偏见。通过线性分类探针评估偏差，并用拓扑图可视化激活区域，发现模型尤其能区分年龄，在中年人群中可辨识性别和种族。**

- **链接: [http://arxiv.org/pdf/2506.03664v1](http://arxiv.org/pdf/2506.03664v1)**

> **作者:** Valerie Krug; Sebastian Stober
>
> **备注:** Summary paper accepted at the 3rd TRR 318 Conference: Contextualizing Explanations 2025
>
> **摘要:** Deep Learning models have achieved remarkable success. Training them is often accelerated by building on top of pre-trained models which poses the risk of perpetuating encoded biases. Here, we investigate biases in the representations of commonly used ImageNet classifiers for facial images while considering intersections of sensitive variables age, race and gender. To assess the biases, we use linear classifier probes and visualize activations as topographic maps. We find that representations in ImageNet classifiers particularly allow differentiation between ages. Less strongly pronounced, the models appear to associate certain ethnicities and distinguish genders in middle-aged groups.
>
---
#### [new 109] A Foundation Model for Spatial Proteomics
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像分析与蛋白质组学任务，旨在解决空间蛋白质组学中数据异质性高、标注样本少的问题。作者提出了KRONOS，一种基于自监督学习的空间蛋白质组基础模型，通过多尺度表征学习实现细胞分型、区域分类等多项任务，具备高效性和跨机构可扩展性。**

- **链接: [http://arxiv.org/pdf/2506.03373v1](http://arxiv.org/pdf/2506.03373v1)**

> **作者:** Muhammad Shaban; Yuzhou Chang; Huaying Qiu; Yao Yu Yeo; Andrew H. Song; Guillaume Jaume; Yuchen Wang; Luca L. Weishaupt; Tong Ding; Anurag Vaidya; Abdallah Lamane; Daniel Shao; Mohammed Zidane; Yunhao Bai; Paige McCallum; Shuli Luo; Wenrui Wu; Yang Wang; Precious Cramer; Chi Ngai Chan; Pierre Stephan; Johanna Schaffenrath; Jia Le Lee; Hendrik A. Michel; Caiwei Tian; Cristina Almagro-Perez; Sophia J. Wagner; Sharifa Sahai; Ming Y. Lu; Richard J. Chen; Andrew Zhang; Mark Edward M. Gonzales; Ahmad Makky; Jia-Ying Joey Lee; Hao Cheng; Nourhan El Ahmar; Sayed Matar; Maximilian Haist; Darci Phillips; Yuqi Tan; Garry P. Nolan; W. Richard Burack; Jacob D. Estes; Jonathan T. C. Liu; Toni K Choueiri; Neeraj Agarwal; Marc Barry; Scott J. Rodig; Long Phi Le; Georg Gerber; Christian M. Schürch; Fabian J. Theis; Youn H Kim; Joe Yeong; Sabina Signoretti; Brooke E. Howitt; Lit-Hsin Loo; Qin Ma; Sizun Jiang; Faisal Mahmood
>
> **摘要:** Foundation models have begun to transform image analysis by acting as pretrained generalist backbones that can be adapted to many tasks even when post-training data are limited, yet their impact on spatial proteomics, imaging that maps proteins at single-cell resolution, remains limited. Here, we introduce KRONOS, a foundation model built for spatial proteomics. KRONOS was trained in a self-supervised manner on over 47 million image patches covering 175 protein markers, 16 tissue types, and 8 fluorescence-based imaging platforms. We introduce key architectural adaptations to address the high-dimensional, multi-channel, and heterogeneous nature of multiplex imaging. We demonstrate that KRONOS learns biologically meaningful representations across multiple scales, ranging from cellular and microenvironment to tissue levels, enabling it to address diverse downstream tasks, including cell phenotyping, region classification, and patient stratification. Evaluated across 11 independent cohorts, KRONOS achieves state-of-the-art performance across cell phenotyping, treatment response prediction, and retrieval tasks, and is highly data-efficient. KRONOS also introduces the paradigm of segmentation-free patch-level processing for efficient and scalable spatial proteomics analysis, allowing cross-institutional comparisons, and as an image reverse search engine for spatial patterns. Together, these results position KRONOS as a flexible and scalable tool for spatial proteomics. The model is publicly accessible at https://github.com/mahmoodlab/KRONOS.
>
---
#### [new 110] FLEX: A Large-Scale Multi-Modal Multi-Action Dataset for Fitness Action Quality Assessment
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于动作质量评估（AQA）任务，旨在解决当前健身动作评估数据集和方法的局限性。作者提出了FLEX数据集，包含多模态、多视角和精细标注的健身动作数据，结合知识图谱构建评估规则，提升模型性能，推动人工智能在健身领域的应用。**

- **链接: [http://arxiv.org/pdf/2506.03198v1](http://arxiv.org/pdf/2506.03198v1)**

> **作者:** Hao Yin; Lijun Gu; Paritosh Parmar; Lin Xu; Tianxiao Guo; Weiwei Fu; Yang Zhang; Tianyou Zheng
>
> **摘要:** With the increasing awareness of health and the growing desire for aesthetic physique, fitness has become a prevailing trend. However, the potential risks associated with fitness training, especially with weight-loaded fitness actions, cannot be overlooked. Action Quality Assessment (AQA), a technology that quantifies the quality of human action and provides feedback, holds the potential to assist fitness enthusiasts of varying skill levels in achieving better training outcomes. Nevertheless, current AQA methodologies and datasets are limited to single-view competitive sports scenarios and RGB modality and lack professional assessment and guidance of fitness actions. To address this gap, we propose the FLEX dataset, the first multi-modal, multi-action, large-scale dataset that incorporates surface electromyography (sEMG) signals into AQA. FLEX utilizes high-precision MoCap to collect 20 different weight-loaded actions performed by 38 subjects across 3 different skill levels for 10 repetitions each, containing 5 different views of the RGB video, 3D pose, sEMG, and physiological information. Additionally, FLEX incorporates knowledge graphs into AQA, constructing annotation rules in the form of penalty functions that map weight-loaded actions, action keysteps, error types, and feedback. We conducted various baseline methodologies on FLEX, demonstrating that multimodal data, multiview data, and fine-grained annotations significantly enhance model performance. FLEX not only advances AQA methodologies and datasets towards multi-modal and multi-action scenarios but also fosters the integration of artificial intelligence within the fitness domain. Dataset and code are available at https://haoyin116.github.io/FLEX_Dataset.
>
---
#### [new 111] HueManity: Probing Fine-Grained Visual Perception in MLLMs
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉感知任务，旨在评估多模态大语言模型（MLLMs）在精细视觉识别中的表现。作者构建了HueManity数据集，包含83,850张带有字符的点阵图像，测试MLLM在模式识别上的能力。实验表明，当前MLLM表现远不如人类和传统视觉模型，揭示其感知缺陷，并开源数据与代码以促进研究改进。**

- **链接: [http://arxiv.org/pdf/2506.03194v1](http://arxiv.org/pdf/2506.03194v1)**

> **作者:** Rynaa Grover; Jayant Sravan Tamarapalli; Sahiti Yerramilli; Nilay Pande
>
> **摘要:** Multimodal Large Language Models (MLLMs) excel at high-level visual reasoning, but their performance on nuanced perceptual tasks remains surprisingly limited. We present HueManity, a benchmark designed to assess visual perception in MLLMs. The dataset comprises 83,850 images featuring two-character alphanumeric strings embedded in Ishihara test style dot patterns, challenging models on precise pattern recognition. Our evaluation of nine state-of-the-art MLLMs on HueManity demonstrates a significant performance deficit compared to human and traditional computer vision baselines. The best-performing MLLM achieved a 33.6% accuracy on the numeric `easy' task and a striking 3% on the alphanumeric `hard' task. In contrast, human participants achieved near-perfect scores (100% and 95.6%), and a fine-tuned ResNet50 model reached accuracies of 96.5% and 94.5%. These results highlight a critical gap in the visual capabilities of current MLLMs. Our analysis further explores potential architectural and training-paradigm factors contributing to this perceptual gap in MLLMs. We open-source HueManity dataset and code to foster further research in improving perceptual robustness of MLLMs.
>
---
#### [new 112] AetherVision-Bench: An Open-Vocabulary RGB-Infrared Benchmark for Multi-Angle Segmentation across Aerial and Ground Perspectives
- **分类: cs.CV**

- **简介: 该论文属于开放词汇语义分割任务，旨在解决跨域泛化问题。作者构建了AetherVision-Bench基准，支持空中与地面视角的多角度分割评估，测试了现有模型并分析影响零样本迁移性能的关键因素，为未来研究提供基础。**

- **链接: [http://arxiv.org/pdf/2506.03709v1](http://arxiv.org/pdf/2506.03709v1)**

> **作者:** Aniruddh Sikdar; Aditya Gandhamal; Suresh Sundaram
>
> **备注:** Accepted at Workshop on Foundation Models Meet Embodied Agents at CVPR 2025 (Non-archival Track)
>
> **摘要:** Open-vocabulary semantic segmentation (OVSS) involves assigning labels to each pixel in an image based on textual descriptions, leveraging world models like CLIP. However, they encounter significant challenges in cross-domain generalization, hindering their practical efficacy in real-world applications. Embodied AI systems are transforming autonomous navigation for ground vehicles and drones by enhancing their perception abilities, and in this study, we present AetherVision-Bench, a benchmark for multi-angle segmentation across aerial, and ground perspectives, which facilitates an extensive evaluation of performance across different viewing angles and sensor modalities. We assess state-of-the-art OVSS models on the proposed benchmark and investigate the key factors that impact the performance of zero-shot transfer models. Our work pioneers the creation of a robustness benchmark, offering valuable insights and establishing a foundation for future research.
>
---
#### [new 113] Continual Learning in Vision-Language Models via Aligned Model Merging
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型的持续学习任务，旨在解决灾难性遗忘问题。现有方法因顺序微调导致模型偏向新任务而遗忘旧知识。本文提出一种基于模型合并的新方法，在更新模型时融合新旧任务参数，并设计对齐机制减少干扰，从而在保持稳定性的同时提升泛化能力和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.03189v1](http://arxiv.org/pdf/2506.03189v1)**

> **作者:** Ghada Sokar; Gintare Karolina Dziugaite; Anurag Arnab; Ahmet Iscen; Pablo Samuel Castro; Cordelia Schmid
>
> **摘要:** Continual learning is conventionally tackled through sequential fine-tuning, a process that, while enabling adaptation, inherently favors plasticity over the stability needed to retain prior knowledge. While existing approaches attempt to mitigate catastrophic forgetting, a bias towards recent tasks persists as they build upon this sequential nature. In this work we present a new perspective based on model merging to maintain stability while still retaining plasticity. Rather than just sequentially updating the model weights, we propose merging newly trained task parameters with previously learned ones, promoting a better balance. To maximize the effectiveness of the merging process, we propose a simple mechanism that promotes learning aligned weights with previous ones, thereby avoiding interference when merging. We evaluate this approach on large Vision-Language Models (VLMs), and demonstrate its effectiveness in reducing forgetting, increasing robustness to various task orders and similarities, and improving generalization.
>
---
#### [new 114] Rex-Thinker: Grounded Object Referring via Chain-of-Thought Reasoning
- **分类: cs.CV**

- **简介: 论文提出Rex-Thinker，用于解决视觉对象指代表达任务。该模型通过链式推理实现可解释、可靠的对象匹配，并构建了HumanRef-CoT数据集进行训练。方法包含两阶段训练：监督微调与强化学习，提升了精度与泛化能力，同时能拒绝无匹配的表达。**

- **链接: [http://arxiv.org/pdf/2506.04034v1](http://arxiv.org/pdf/2506.04034v1)**

> **作者:** Qing Jiang; Xingyu Chen; Zhaoyang Zeng; Junzhi Yu; Lei Zhang
>
> **备注:** homepage: https://rexthinker.github.io/
>
> **摘要:** Object referring aims to detect all objects in an image that match a given natural language description. We argue that a robust object referring model should be grounded, meaning its predictions should be both explainable and faithful to the visual content. Specifically, it should satisfy two key properties: 1) Verifiable, by producing interpretable reasoning that justifies its predictions and clearly links them to visual evidence; and 2) Trustworthy, by learning to abstain when no object in the image satisfies the given expression. However, most methods treat referring as a direct bounding box prediction task, offering limited interpretability and struggling to reject expressions with no matching object. In this work, we propose Rex-Thinker, a model that formulates object referring as an explicit CoT reasoning task. Given a referring expression, we first identify all candidate object instances corresponding to the referred object category. Rex-Thinker then performs step-by-step reasoning over each candidate to assess whether it matches the given expression, before making a final prediction. To support this paradigm, we construct a large-scale CoT-style referring dataset named HumanRef-CoT by prompting GPT-4o on the HumanRef dataset. Each reasoning trace follows a structured planning, action, and summarization format, enabling the model to learn decomposed, interpretable reasoning over object candidates. We then train Rex-Thinker in two stages: a cold-start supervised fine-tuning phase to teach the model how to perform structured reasoning, followed by GRPO-based RL learning to improve accuracy and generalization. Experiments show that our approach outperforms standard baselines in both precision and interpretability on in-domain evaluation, while also demonstrating improved ability to reject hallucinated outputs and strong generalization in out-of-domain settings.
>
---
#### [new 115] Temporal Vegetation Index-Based Unsupervised Crop Stress Detection via Eigenvector-Guided Contrastive Learning
- **分类: cs.CV**

- **简介: 该论文属于农业遥感与作物监测任务，旨在解决作物胁迫早期检测问题。现有方法依赖标签或仅在症状明显后检测，限制了应用。论文提出EigenCL，一种基于时序植被指数的无监督对比学习框架，利用NDRE时间序列和特征分解捕捉作物胁迫轨迹，实现早期检测与高精度分类，适用于数据稀缺的实际农业场景。**

- **链接: [http://arxiv.org/pdf/2506.03394v1](http://arxiv.org/pdf/2506.03394v1)**

> **作者:** Shafqaat Ahmad
>
> **摘要:** Early detection of crop stress is vital for minimizing yield loss and enabling timely intervention in precision agriculture. Traditional approaches using NDRE often detect stress only after visible symptoms appear or require labeled datasets, limiting scalability. This study introduces EigenCL, a novel unsupervised contrastive learning framework guided by temporal NDRE dynamics and biologically grounded eigen decomposition. Using over 10,000 Sentinel-2 NDRE image patches from drought-affected Iowa cornfields, we constructed five-point NDRE time series per patch and derived an RBF similarity matrix. The principal eigenvector explaining 76% of the variance and strongly correlated (r = 0.95) with raw NDRE values was used to define stress-aware similarity for contrastive embedding learning. Unlike existing methods that rely on visual augmentations, EigenCL pulls embeddings together based on biologically similar stress trajectories and pushes apart divergent ones. The learned embeddings formed physiologically meaningful clusters, achieving superior clustering metrics (Silhouette: 0.748, DBI: 0.35) and enabling 76% early stress detection up to 12 days before conventional NDRE thresholds. Downstream classification yielded 95% k-NN and 91% logistic regression accuracy. Validation on an independent 2023 Nebraska dataset confirmed generalizability without retraining. EigenCL offers a label-free, scalable approach for early stress detection that aligns with underlying plant physiology and is suitable for real-world deployment in data-scarce agricultural environments.
>
---
#### [new 116] JointSplat: Probabilistic Joint Flow-Depth Optimization for Sparse-View Gaussian Splatting
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于稀疏视角三维重建任务，旨在解决低纹理或重复区域中的误匹配、噪声和全局不一致问题。作者提出JointSplat，通过概率联合光流与深度优化，融合像素级信息并引入多视角深度一致性损失，提升了重建效果，在RealEstate10K和ACID数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.03872v1](http://arxiv.org/pdf/2506.03872v1)**

> **作者:** Yang Xiao; Guoan Xu; Qiang Wu; Wenjing Jia
>
> **摘要:** Reconstructing 3D scenes from sparse viewpoints is a long-standing challenge with wide applications. Recent advances in feed-forward 3D Gaussian sparse-view reconstruction methods provide an efficient solution for real-time novel view synthesis by leveraging geometric priors learned from large-scale multi-view datasets and computing 3D Gaussian centers via back-projection. Despite offering strong geometric cues, both feed-forward multi-view depth estimation and flow-depth joint estimation face key limitations: the former suffers from mislocation and artifact issues in low-texture or repetitive regions, while the latter is prone to local noise and global inconsistency due to unreliable matches when ground-truth flow supervision is unavailable. To overcome this, we propose JointSplat, a unified framework that leverages the complementarity between optical flow and depth via a novel probabilistic optimization mechanism. Specifically, this pixel-level mechanism scales the information fusion between depth and flow based on the matching probability of optical flow during training. Building upon the above mechanism, we further propose a novel multi-view depth-consistency loss to leverage the reliability of supervision while suppressing misleading gradients in uncertain areas. Evaluated on RealEstate10K and ACID, JointSplat consistently outperforms state-of-the-art (SOTA) methods, demonstrating the effectiveness and robustness of our proposed probabilistic joint flow-depth optimization approach for high-fidelity sparse-view 3D reconstruction.
>
---
#### [new 117] Kinship in Speech: Leveraging Linguistic Relatedness for Zero-Shot TTS in Indian Languages
- **分类: cs.CL; cs.CV; I.5.4**

- **简介: 该论文属于语音合成任务，旨在解决印度多语言环境下缺乏语料导致难以训练TTS系统的问题。工作重点是利用语言间的语音关联，构建共享音素表示并调整文本解析规则，实现零样本跨语言语音合成，成功生成多种印度语言的可懂自然语音。**

- **链接: [http://arxiv.org/pdf/2506.03884v1](http://arxiv.org/pdf/2506.03884v1)**

> **作者:** Utkarsh Pathak; Chandra Sai Krishna Gunda; Anusha Prakash; Keshav Agarwal; Hema A. Murthy
>
> **备注:** Accepted at INTERSPEECH 2025
>
> **摘要:** Text-to-speech (TTS) systems typically require high-quality studio data and accurate transcriptions for training. India has 1369 languages, with 22 official using 13 scripts. Training a TTS system for all these languages, most of which have no digital resources, seems a Herculean task. Our work focuses on zero-shot synthesis, particularly for languages whose scripts and phonotactics come from different families. The novelty of our work is in the augmentation of a shared phone representation and modifying the text parsing rules to match the phonotactics of the target language, thus reducing the synthesiser overhead and enabling rapid adaptation. Intelligible and natural speech was generated for Sanskrit, Maharashtrian and Canara Konkani, Maithili and Kurukh by leveraging linguistic connections across languages with suitable synthesisers. Evaluations confirm the effectiveness of this approach, highlighting its potential to expand speech technology access for under-represented languages.
>
---
#### [new 118] DynTok: Dynamic Compression of Visual Tokens for Efficient and Effective Video Understanding
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决视频建模中视觉token数量过多导致的计算开销问题。作者提出了DynTok方法，通过动态压缩低信息密度区域的token，在保持性能的同时显著减少token数量，提升了效率。**

- **链接: [http://arxiv.org/pdf/2506.03990v1](http://arxiv.org/pdf/2506.03990v1)**

> **作者:** Hongzhi Zhang; Jingyuan Zhang; Xingguang Ji; Qi Wang; Fuzheng Zhang
>
> **摘要:** Typical video modeling methods, such as LLava, represent videos as sequences of visual tokens, which are then processed by the LLM backbone for effective video understanding. However, this approach leads to a massive number of visual tokens, especially for long videos. A practical solution is to first extract relevant visual information from the large visual context before feeding it into the LLM backbone, thereby reducing computational overhead. In this work, we introduce DynTok, a novel \textbf{Dyn}amic video \textbf{Tok}en compression strategy. DynTok adaptively splits visual tokens into groups and merges them within each group, achieving high compression in regions with low information density while preserving essential content. Our method reduces the number of tokens to 44.4% of the original size while maintaining comparable performance. It further benefits from increasing the number of video frames and achieves 65.3% on Video-MME and 72.5% on MLVU. By applying this simple yet effective compression method, we expose the redundancy in video token representations and offer insights for designing more efficient video modeling techniques.
>
---
#### [new 119] Facial Appearance Capture at Home with Patch-Level Reflectance Prior
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于面部外观重建任务，旨在解决家用设备（如智能手机和手电筒）在非理想环境下采集人脸反射属性质量低的问题。通过引入基于补丁级反射先验的扩散模型，并结合后验采样技术，显著提升重建效果，使其接近专业影棚水平。**

- **链接: [http://arxiv.org/pdf/2506.03478v1](http://arxiv.org/pdf/2506.03478v1)**

> **作者:** Yuxuan Han; Junfeng Lyu; Kuan Sheng; Minghao Que; Qixuan Zhang; Lan Xu; Feng Xu
>
> **备注:** ACM Transactions on Graphics (Proc. of SIGGRAPH), 2025. Code: https://github.com/yxuhan/DoRA; Project Page: https://yxuhan.github.io/DoRA
>
> **摘要:** Existing facial appearance capture methods can reconstruct plausible facial reflectance from smartphone-recorded videos. However, the reconstruction quality is still far behind the ones based on studio recordings. This paper fills the gap by developing a novel daily-used solution with a co-located smartphone and flashlight video capture setting in a dim room. To enhance the quality, our key observation is to solve facial reflectance maps within the data distribution of studio-scanned ones. Specifically, we first learn a diffusion prior over the Light Stage scans and then steer it to produce the reflectance map that best matches the captured images. We propose to train the diffusion prior at the patch level to improve generalization ability and training stability, as current Light Stage datasets are in ultra-high resolution but limited in data size. Tailored to this prior, we propose a patch-level posterior sampling technique to sample seamless full-resolution reflectance maps from this patch-level diffusion model. Experiments demonstrate our method closes the quality gap between low-cost and studio recordings by a large margin, opening the door for everyday users to clone themselves to the digital world. Our code will be released at https://github.com/yxuhan/DoRA.
>
---
#### [new 120] Deep Learning-Based Breast Cancer Detection in Mammography: A Multi-Center Validation Study in Thai Population
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决乳腺癌在乳腺X线图像中的检测问题。作者开发了一种基于深度学习的系统，采用改进的EfficientNetV2架构并增强注意力机制，进行了多中心验证，展示了良好的癌症检测性能和临床应用潜力。**

- **链接: [http://arxiv.org/pdf/2506.03177v1](http://arxiv.org/pdf/2506.03177v1)**

> **作者:** Isarun Chamveha; Supphanut Chaiyungyuen; Sasinun Worakriangkrai; Nattawadee Prasawang; Warasinee Chaisangmongkon; Pornpim Korpraphong; Voraparee Suvannarerg; Shanigarn Thiravit; Chalermdej Kannawat; Kewalin Rungsinaporn; Suwara Issaragrisil; Payia Chadbunchachai; Pattiya Gatechumpol; Chawiporn Muktabhant; Patarachai Sereerat
>
> **摘要:** This study presents a deep learning system for breast cancer detection in mammography, developed using a modified EfficientNetV2 architecture with enhanced attention mechanisms. The model was trained on mammograms from a major Thai medical center and validated on three distinct datasets: an in-domain test set (9,421 cases), a biopsy-confirmed set (883 cases), and an out-of-domain generalizability set (761 cases) collected from two different hospitals. For cancer detection, the model achieved AUROCs of 0.89, 0.96, and 0.94 on the respective datasets. The system's lesion localization capability, evaluated using metrics including Lesion Localization Fraction (LLF) and Non-Lesion Localization Fraction (NLF), demonstrated robust performance in identifying suspicious regions. Clinical validation through concordance tests showed strong agreement with radiologists: 83.5% classification and 84.0% localization concordance for biopsy-confirmed cases, and 78.1% classification and 79.6% localization concordance for out-of-domain cases. Expert radiologists' acceptance rate also averaged 96.7% for biopsy-confirmed cases, and 89.3% for out-of-domain cases. The system achieved a System Usability Scale score of 74.17 for source hospital, and 69.20 for validation hospitals, indicating good clinical acceptance. These results demonstrate the model's effectiveness in assisting mammogram interpretation, with the potential to enhance breast cancer screening workflows in clinical practice.
>
---
#### [new 121] Rethinking Whole-Body CT Image Interpretation: An Abnormality-Centric Approach
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决全身体部CT图像中异常定位与描述的挑战。论文构建了包含404种异常的分类体系及14.5K图像的数据集，并提出模型OminiAbnorm-CT，实现基于文本查询和视觉提示的异常自动识别与描述，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.03238v1](http://arxiv.org/pdf/2506.03238v1)**

> **作者:** Ziheng Zhao; Lisong Dai; Ya Zhang; Yanfeng Wang; Weidi Xie
>
> **摘要:** Automated interpretation of CT images-particularly localizing and describing abnormal findings across multi-plane and whole-body scans-remains a significant challenge in clinical radiology. This work aims to address this challenge through four key contributions: (i) On taxonomy, we collaborate with senior radiologists to propose a comprehensive hierarchical classification system, with 404 representative abnormal findings across all body regions; (ii) On data, we contribute a dataset containing over 14.5K CT images from multiple planes and all human body regions, and meticulously provide grounding annotations for over 19K abnormalities, each linked to the detailed description and cast into the taxonomy; (iii) On model development, we propose OminiAbnorm-CT, which can automatically ground and describe abnormal findings on multi-plane and whole-body CT images based on text queries, while also allowing flexible interaction through visual prompts; (iv) On benchmarks, we establish three representative evaluation tasks based on real clinical scenarios. Through extensive experiments, we show that OminiAbnorm-CT can significantly outperform existing methods on all the tasks and metrics.
>
---
#### [new 122] A combined Machine Learning and Finite Element Modelling tool for the surgical planning of craniosynostosis correction
- **分类: eess.IV; cs.CV; cs.LG; physics.med-ph**

- **简介: 该论文旨在开发一种结合机器学习与有限元建模的实时手术规划工具，用于改善颅缝早闭矫正手术效果。通过三维照片生成个性化合成颅骨模型，避免使用CT扫描，减少辐射暴露。利用支持向量回归模型预测手术结果，实现高精度（R²达0.95），并可优化手术参数以提高颅骨指数。**

- **链接: [http://arxiv.org/pdf/2506.03202v1](http://arxiv.org/pdf/2506.03202v1)**

> **作者:** Itxasne Antúnez Sáenz; Ane Alberdi Aramendi; David Dunaway; Juling Ong; Lara Deliège; Amparo Sáenz; Anita Ahmadi Birjandi; Noor UI Owase Jeelani; Silvia Schievano; Alessandro Borghi
>
> **备注:** 11 pages, 16 figures
>
> **摘要:** Craniosynostosis is a medical condition that affects the growth of babies' heads, caused by an early fusion of cranial sutures. In recent decades, surgical treatments for craniosynostosis have significantly improved, leading to reduced invasiveness, faster recovery, and less blood loss. At Great Ormond Street Hospital (GOSH), the main surgical treatment for patients diagnosed with sagittal craniosynostosis (SC) is spring assisted cranioplasty (SAC). This procedure involves a 15x15 mm2 osteotomy, where two springs are inserted to induce distraction. Despite the numerous advantages of this surgical technique for patients, the outcome remains unpredictable due to the lack of efficient preoperative planning tools. The surgeon's experience and the baby's age are currently relied upon to determine the osteotomy location and spring selection. Previous tools for predicting the surgical outcome of SC relied on finite element modeling (FEM), which involved computed tomography (CT) imaging and required engineering expertise and lengthy calculations. The main goal of this research is to develop a real-time prediction tool for the surgical outcome of patients, eliminating the need for CT scans to minimise radiation exposure during preoperative planning. The proposed methodology involves creating personalised synthetic skulls based on three-dimensional (3D) photographs, incorporating population average values of suture location, skull thickness, and soft tissue properties. A machine learning (ML) surrogate model is employed to achieve the desired surgical outcome. The resulting multi-output support vector regressor model achieves a R2 metric of 0.95 and MSE and MAE below 0.13. Furthermore, in the future, this model could not only simulate various surgical scenarios but also provide optimal parameters for achieving a maximum cranial index (CI).
>
---
#### [new 123] Adapt before Continual Learning
- **分类: cs.LG; cs.CV**

- **简介: 论文属于持续学习任务，旨在解决预训练模型在增量学习中稳定性与可塑性失衡问题。作者提出ACL框架，在每次新任务学习前对模型进行适配优化，提升跨域适应能力，同时减少知识遗忘，有效平衡了稳定性和可塑性。**

- **链接: [http://arxiv.org/pdf/2506.03956v1](http://arxiv.org/pdf/2506.03956v1)**

> **作者:** Aojun Lu; Tao Feng; Hangjie Yuan; Chunhui Ding; Yanan Sun
>
> **摘要:** Continual Learning (CL) seeks to enable neural networks to incrementally acquire new knowledge (plasticity) while retaining existing knowledge (stability). While pre-trained models (PTMs) have become pivotal in CL, prevailing approaches freeze the PTM backbone to preserve stability, limiting their plasticity, particularly when encountering significant domain gaps in incremental tasks. Conversely, sequentially finetuning the entire PTM risks catastrophic forgetting of generalizable knowledge, exposing a critical stability-plasticity trade-off. To address this challenge, we propose Adapting PTMs before the core CL process (ACL), a novel framework that refines the PTM backbone through a plug-and-play adaptation phase before learning each new task with existing CL approaches (e.g., prompt tuning). ACL enhances plasticity by aligning embeddings with their original class prototypes while distancing them from others, theoretically and empirically shown to balance stability and plasticity. Extensive experiments demonstrate that ACL significantly improves CL performance across benchmarks and integrated methods, offering a versatile solution for PTM-based CL.
>
---
#### [new 124] petBrain: A New Pipeline for Amyloid, Tau Tangles and Neurodegeneration Quantification Using PET and MRI
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决阿尔茨海默病（AD）中淀粉样蛋白、tau蛋白和神经退化的量化问题。作者开发了petBrain这一端到端的深度学习管道，支持多模态PET和MRI数据的自动化处理与生物标志物标准化量化，并实现了快速、可靠且无需本地设备的在线分析。**

- **链接: [http://arxiv.org/pdf/2506.03217v1](http://arxiv.org/pdf/2506.03217v1)**

> **作者:** Pierrick Coupé; Boris Mansencal; Floréal Morandat; Sergio Morell-Ortega; Nicolas Villain; Jose V. Manjón; Vincent Planche
>
> **摘要:** INTRODUCTION: Quantification of amyloid plaques (A), neurofibrillary tangles (T2), and neurodegeneration (N) using PET and MRI is critical for Alzheimer's disease (AD) diagnosis and prognosis. Existing pipelines face limitations regarding processing time, variability in tracer types, and challenges in multimodal integration. METHODS: We developed petBrain, a novel end-to-end processing pipeline for amyloid-PET, tau-PET, and structural MRI. It leverages deep learning-based segmentation, standardized biomarker quantification (Centiloid, CenTauR, HAVAs), and simultaneous estimation of A, T2, and N biomarkers. The pipeline is implemented as a web-based platform, requiring no local computational infrastructure or specialized software knowledge. RESULTS: petBrain provides reliable and rapid biomarker quantification, with results comparable to existing pipelines for A and T2. It shows strong concordance with data processed in ADNI databases. The staging and quantification of A/T2/N by petBrain demonstrated good agreement with CSF/plasma biomarkers, clinical status, and cognitive performance. DISCUSSION: petBrain represents a powerful and openly accessible platform for standardized AD biomarker analysis, facilitating applications in clinical research.
>
---
#### [new 125] mRAG: Elucidating the Design Space of Multi-modal Retrieval-Augmented Generation
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态检索增强生成（RAG）任务，旨在解决大视觉-语言模型（LVLMs）因依赖静态训练数据、易产生幻觉和缺乏实时验证能力而影响实际应用的问题。作者系统分析了RAG流程中的检索、重排序与生成阶段，并提出统一的代理框架，通过自反思机制动态选择证据，提升了模型性能约5%。**

- **链接: [http://arxiv.org/pdf/2505.24073v1](http://arxiv.org/pdf/2505.24073v1)**

> **作者:** Chan-Wei Hu; Yueqi Wang; Shuo Xing; Chia-Ju Chen; Zhengzhong Tu
>
> **备注:** 16 pages, 11 figures
>
> **摘要:** Large Vision-Language Models (LVLMs) have made remarkable strides in multimodal tasks such as visual question answering, visual grounding, and complex reasoning. However, they remain limited by static training data, susceptibility to hallucinations, and inability to verify claims against up-to-date, external evidence, compromising their performance in dynamic real-world applications. Retrieval-Augmented Generation (RAG) offers a practical solution to mitigate these challenges by allowing the LVLMs to access large-scale knowledge databases via retrieval mechanisms, thereby grounding model outputs in factual, contextually relevant information. Here in this paper, we conduct the first systematic dissection of the multimodal RAG pipeline for LVLMs, explicitly investigating (1) the retrieval phase: on the modality configurations and retrieval strategies, (2) the re-ranking stage: on strategies to mitigate positional biases and improve the relevance of retrieved evidence, and (3) the generation phase: we further investigate how to best integrate retrieved candidates into the final generation process. Finally, we extend to explore a unified agentic framework that integrates re-ranking and generation through self-reflection, enabling LVLMs to select relevant evidence and suppress irrelevant context dynamically. Our full-stack exploration of RAG for LVLMs yields substantial insights, resulting in an average performance boost of 5% without any fine-tuning.
>
---
#### [new 126] How Far Are We from Predicting Missing Modalities with Foundation Models?
- **分类: cs.MM; cs.CV**

- **简介: 该论文属于多模态任务，旨在解决缺失模态预测问题。现有基础模型在细粒度语义提取和生成模态验证方面存在不足，导致预测效果不佳。作者提出了一种基于智能体的框架，包含动态策略制定与自精炼机制，有效提升缺失图像与文本预测性能。**

- **链接: [http://arxiv.org/pdf/2506.03530v1](http://arxiv.org/pdf/2506.03530v1)**

> **作者:** Guanzhou Ke; Yi Xie; Xiaoli Wang; Guoqing Chao; Bo Wang; Shengfeng He
>
> **摘要:** Multimodal foundation models have demonstrated impressive capabilities across diverse tasks. However, their potential as plug-and-play solutions for missing modality prediction remains underexplored. To investigate this, we categorize existing approaches into three representative paradigms, encompassing a total of 42 model variants, and conduct a comprehensive evaluation in terms of prediction accuracy and adaptability to downstream tasks. Our analysis reveals that current foundation models often fall short in two critical aspects: (i) fine-grained semantic extraction from the available modalities, and (ii) robust validation of generated modalities. These limitations lead to suboptimal and, at times, misaligned predictions. To address these challenges, we propose an agentic framework tailored for missing modality prediction. This framework dynamically formulates modality-aware mining strategies based on the input context, facilitating the extraction of richer and more discriminative semantic features. In addition, we introduce a \textit{self-refinement mechanism}, which iteratively verifies and enhances the quality of generated modalities through internal feedback. Experimental results show that our method reduces FID for missing image prediction by at least 14% and MER for missing text prediction by at least 10% compared to baselines.
>
---
#### [new 127] Multi-Spectral Gaussian Splatting with Neural Color Representation
- **分类: cs.GR; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于多光谱三维重建任务，旨在解决跨模态相机标定复杂、多光谱信息利用不足的问题。作者提出MS-Splatting，采用神经颜色表示方法统一建模多光谱信息，通过MLP解码提升渲染质量，并应用于农业植被指数生成。**

- **链接: [http://arxiv.org/pdf/2506.03407v1](http://arxiv.org/pdf/2506.03407v1)**

> **作者:** Lukas Meyer; Josef Grün; Maximilian Weiherer; Bernhard Egger; Marc Stamminger; Linus Franke
>
> **摘要:** We present MS-Splatting -- a multi-spectral 3D Gaussian Splatting (3DGS) framework that is able to generate multi-view consistent novel views from images of multiple, independent cameras with different spectral domains. In contrast to previous approaches, our method does not require cross-modal camera calibration and is versatile enough to model a variety of different spectra, including thermal and near-infra red, without any algorithmic changes. Unlike existing 3DGS-based frameworks that treat each modality separately (by optimizing per-channel spherical harmonics) and therefore fail to exploit the underlying spectral and spatial correlations, our method leverages a novel neural color representation that encodes multi-spectral information into a learned, compact, per-splat feature embedding. A shallow multi-layer perceptron (MLP) then decodes this embedding to obtain spectral color values, enabling joint learning of all bands within a unified representation. Our experiments show that this simple yet effective strategy is able to improve multi-spectral rendering quality, while also leading to improved per-spectra rendering quality over state-of-the-art methods. We demonstrate the effectiveness of this new technique in agricultural applications to render vegetation indices, such as normalized difference vegetation index (NDVI).
>
---
#### [new 128] Dc-EEMF: Pushing depth-of-field limit of photoacoustic microscopy via decision-level constrained learning
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学成像任务，旨在解决光声显微镜（OR-PAM）景深受限的问题。作者提出Dc-EEMF方法，通过端到端的轻量级网络融合多焦点图像，提升景深，同时保持横向分辨率，无需后期处理，适用于临床前和临床研究。**

- **链接: [http://arxiv.org/pdf/2506.03181v1](http://arxiv.org/pdf/2506.03181v1)**

> **作者:** Wangting Zhou; Jiangshan He; Tong Cai; Lin Wang; Zhen Yuan; Xunbin Wei; Xueli Chen
>
> **摘要:** Photoacoustic microscopy holds the potential to measure biomarkers' structural and functional status without labels, which significantly aids in comprehending pathophysiological conditions in biomedical research. However, conventional optical-resolution photoacoustic microscopy (OR-PAM) is hindered by a limited depth-of-field (DoF) due to the narrow depth range focused on a Gaussian beam. Consequently, it fails to resolve sufficient details in the depth direction. Herein, we propose a decision-level constrained end-to-end multi-focus image fusion (Dc-EEMF) to push DoF limit of PAM. The DC-EEMF method is a lightweight siamese network that incorporates an artifact-resistant channel-wise spatial frequency as its feature fusion rule. The meticulously crafted U-Net-based perceptual loss function for decision-level focus properties in end-to-end fusion seamlessly integrates the complementary advantages of spatial domain and transform domain methods within Dc-EEMF. This approach can be trained end-to-end without necessitating post-processing procedures. Experimental results and numerical analyses collectively demonstrate our method's robust performance, achieving an impressive fusion result for PAM images without a substantial sacrifice in lateral resolution. The utilization of Dc-EEMF-powered PAM has the potential to serve as a practical tool in preclinical and clinical studies requiring extended DoF for various applications.
>
---
#### [new 129] Super-temporal-resolution Photoacoustic Imaging with Dynamic Reconstruction through Implicit Neural Representation in Sparse-view
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学成像任务，旨在解决动态光声成像中因传感器稀疏和低时间分辨率导致的图像质量问题。作者提出一种基于隐式神经表示（INR）的方法，通过利用时空坐标的隐式连续表示，结合低秩与稀疏正则化，实现从稀疏数据中高质量重建动态图像，提升了时间分辨率并抑制了伪影。**

- **链接: [http://arxiv.org/pdf/2506.03175v1](http://arxiv.org/pdf/2506.03175v1)**

> **作者:** Youshen Xiao; Yiling Shi; Ruixi Sun; Hongjiang Wei; Fei Gao; Yuyao Zhang
>
> **摘要:** Dynamic Photoacoustic Computed Tomography (PACT) is an important imaging technique for monitoring physiological processes, capable of providing high-contrast images of optical absorption at much greater depths than traditional optical imaging methods. However, practical instrumentation and geometric constraints limit the number of acoustic sensors available around the imaging target, leading to sparsity in sensor data. Traditional photoacoustic (PA) image reconstruction methods, when directly applied to sparse PA data, produce severe artifacts. Additionally, these traditional methods do not consider the inter-frame relationships in dynamic imaging. Temporal resolution is crucial for dynamic photoacoustic imaging, which is fundamentally limited by the low repetition rate (e.g., 20 Hz) and high cost of high-power laser technology. Recently, Implicit Neural Representation (INR) has emerged as a powerful deep learning tool for solving inverse problems with sparse data, by characterizing signal properties as continuous functions of their coordinates in an unsupervised manner. In this work, we propose an INR-based method to improve dynamic photoacoustic image reconstruction from sparse-views and enhance temporal resolution, using only spatiotemporal coordinates as input. Specifically, the proposed INR represents dynamic photoacoustic images as implicit functions and encodes them into a neural network. The weights of the network are learned solely from the acquired sparse sensor data, without the need for external training datasets or prior images. Benefiting from the strong implicit continuity regularization provided by INR, as well as explicit regularization for low-rank and sparsity, our proposed method outperforms traditional reconstruction methods under two different sparsity conditions, effectively suppressing artifacts and ensuring image quality.
>
---
#### [new 130] Personalized MR-Informed Diffusion Models for 3D PET Image Reconstruction
- **分类: physics.med-ph; cs.CV**

- **简介: 该论文属于医学图像重建任务，旨在解决低计数数据下PET图像重建精度不足的问题。利用MR扫描生成个性化“伪PET”图像，通过扩散模型进行预训练，提升重建效果，在保留PET独特特征的同时更好地结合MR解剖信息。**

- **链接: [http://arxiv.org/pdf/2506.03804v1](http://arxiv.org/pdf/2506.03804v1)**

> **作者:** George Webber; Alexander Hammers; Andrew P. King; Andrew J. Reader
>
> **备注:** 10 pages, 10 figures
>
> **摘要:** Recent work has shown improved lesion detectability and flexibility to reconstruction hyperparameters (e.g. scanner geometry or dose level) when PET images are reconstructed by leveraging pre-trained diffusion models. Such methods train a diffusion model (without sinogram data) on high-quality, but still noisy, PET images. In this work, we propose a simple method for generating subject-specific PET images from a dataset of multi-subject PET-MR scans, synthesizing "pseudo-PET" images by transforming between different patients' anatomy using image registration. The images we synthesize retain information from the subject's MR scan, leading to higher resolution and the retention of anatomical features compared to the original set of PET images. With simulated and real [$^{18}$F]FDG datasets, we show that pre-training a personalized diffusion model with subject-specific "pseudo-PET" images improves reconstruction accuracy with low-count data. In particular, the method shows promise in combining information from a guidance MR scan without overly imposing anatomical features, demonstrating an improved trade-off between reconstructing PET-unique image features versus features present in both PET and MR. We believe this approach for generating and utilizing synthetic data has further applications to medical imaging tasks, particularly because patient-specific PET images can be generated without resorting to generative deep learning or large training datasets.
>
---
#### [new 131] SNIFR : Boosting Fine-Grained Child Harmful Content Detection Through Audio-Visual Alignment with Cascaded Cross-Transformer
- **分类: eess.AS; cs.CV; cs.MM**

- **简介: 该论文属于多模态内容审核任务，旨在提升儿童有害内容的细粒度检测。为解决恶意用户通过少量敏感帧逃避检测的问题，论文提出SNIFR框架，融合音视频信息，利用级联交叉Transformer实现跨模态对齐，显著提升了检测效果。**

- **链接: [http://arxiv.org/pdf/2506.03378v1](http://arxiv.org/pdf/2506.03378v1)**

> **作者:** Orchid Chetia Phukan; Mohd Mujtaba Akhtar; Girish; Swarup Ranjan Behera; Abu Osama Siddiqui; Sarthak Jain; Priyabrata Mallick; Jaya Sai Kiran Patibandla; Pailla Balakrishna Reddy; Arun Balaji Buduru; Rajesh Sharma
>
> **备注:** Accepted to INTERSPEECH 2025
>
> **摘要:** As video-sharing platforms have grown over the past decade, child viewership has surged, increasing the need for precise detection of harmful content like violence or explicit scenes. Malicious users exploit moderation systems by embedding unsafe content in minimal frames to evade detection. While prior research has focused on visual cues and advanced such fine-grained detection, audio features remain underexplored. In this study, we embed audio cues with visual for fine-grained child harmful content detection and introduce SNIFR, a novel framework for effective alignment. SNIFR employs a transformer encoder for intra-modality interaction, followed by a cascaded cross-transformer for inter-modality alignment. Our approach achieves superior performance over unimodal and baseline fusion methods, setting a new state-of-the-art.
>
---
#### [new 132] Optimal Transport-based Domain Alignment as a Preprocessing Step for Federated Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于联邦学习任务，旨在解决数据集不平衡导致的模型性能下降问题。通过引入基于最优传输的预处理算法，利用Wasserstein重心对齐边缘设备上的数据分布，减少全局样本方差，从而提升模型泛化能力与通信效率。**

- **链接: [http://arxiv.org/pdf/2506.04071v1](http://arxiv.org/pdf/2506.04071v1)**

> **作者:** Luiz Manella Pereira; M. Hadi Amini
>
> **摘要:** Federated learning (FL) is a subfield of machine learning that avoids sharing local data with a central server, which can enhance privacy and scalability. The inability to consolidate data leads to a unique problem called dataset imbalance, where agents in a network do not have equal representation of the labels one is trying to learn to predict. In FL, fusing locally-trained models with unbalanced datasets may deteriorate the performance of global model aggregation, and reduce the quality of updated local models and the accuracy of the distributed agents' decisions. In this work, we introduce an Optimal Transport-based preprocessing algorithm that aligns the datasets by minimizing the distributional discrepancy of data along the edge devices. We accomplish this by leveraging Wasserstein barycenters when computing channel-wise averages. These barycenters are collected in a trusted central server where they collectively generate a target RGB space. By projecting our dataset towards this target space, we minimize the distributional discrepancy on a global level, which facilitates the learning process due to a minimization of variance across the samples. We demonstrate the capabilities of the proposed approach over the CIFAR-10 dataset, where we show its capability of reaching higher degrees of generalization in fewer communication rounds.
>
---
#### [new 133] ROSA: Addressing text understanding challenges in photographs via ROtated SAmpling
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉问答（VQA）任务，旨在解决视障人士拍摄照片中文本方向不正导致的识别难题。通过分析视障用户拍摄习惯，提出ROSA解码策略，显著提升模型对倾斜文本的理解效果。**

- **链接: [http://arxiv.org/pdf/2506.03665v1](http://arxiv.org/pdf/2506.03665v1)**

> **作者:** Hernán Maina; Guido Ivetta; Mateo Lione Stuto; Julian Martin Eisenschlos; Jorge Sánchez; Luciana Benotti
>
> **摘要:** Visually impaired people could benefit from Visual Question Answering (VQA) systems to interpret text in their surroundings. However, current models often struggle with recognizing text in the photos taken by this population. Through in-depth interviews with visually impaired individuals, we identified common framing conventions that frequently result in misaligned text. Existing VQA benchmarks primarily feature well-oriented text captured by sighted users, under-representing these challenges. To address this gap, we introduce ROtated SAmpling (ROSA), a decoding strategy that enhances VQA performance in text-rich images with incorrectly oriented text. ROSA outperforms Greedy decoding by 11.7 absolute points in the best-performing model.
>
---
#### [new 134] Analytical Reconstruction of Periodically Deformed Objects in Time-resolved CT
- **分类: physics.med-ph; cs.CV**

- **简介: 该论文属于医学成像任务，旨在解决动态周期物体在CT重建中的辐射效率低和图像噪声问题。作者提出了两种解析重建方法，利用全数据而非分阶段子集，提升图像质量或降低辐射剂量，并通过同步辐射显微数据验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2506.03792v1](http://arxiv.org/pdf/2506.03792v1)**

> **作者:** Qianwei Qu; Christian M. Schlepütz; Marco Stampanoni
>
> **摘要:** Time-resolved CT is an advanced measurement technique that has been widely used to observe dynamic objects, including periodically varying structures such as hearts, lungs, or hearing structures. To reconstruct these objects from CT projections, a common approach is to divide the projections into several collections based on their motion phases and perform reconstruction within each collection, assuming they originate from a static object. This describes the gating-based method, which is the standard approach for time-periodic reconstruction. However, the gating-based reconstruction algorithm only utilizes a limited subset of projections within each collection and ignores the correlation between different collections, leading to inefficient use of the radiation dose. To address this issue, we propose two analytical reconstruction pipelines in this paper, and validate them with experimental data captured using tomographic synchrotron microscopy. We demonstrate that our approaches significantly reduce random noise in the reconstructed images without blurring the sharp features of the observed objects. Equivalently, our methods can achieve the same reconstruction quality as gating-based methods but with a lower radiation dose. Our code is available at github.com/PeriodRecon.
>
---
#### [new 135] LLaMA-XR: A Novel Framework for Radiology Report Generation using LLaMA and QLoRA Fine Tuning
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像处理与自然语言生成任务，旨在解决自动生成准确、上下文相关的放射科报告问题。作者提出了LLaMA-XR框架，结合LLaMA 3.1和DenseNet-121图像嵌入，并采用QLoRA微调方法，提升了生成报告的连贯性和临床准确性，同时保持计算效率，在IU X-ray数据集上取得了优异的ROUGE-L和METEOR评分。**

- **链接: [http://arxiv.org/pdf/2506.03178v1](http://arxiv.org/pdf/2506.03178v1)**

> **作者:** Md. Zihad Bin Jahangir; Muhammad Ashad Kabir; Sumaiya Akter; Israt Jahan; Minh Chau
>
> **备注:** 25 pages
>
> **摘要:** Automated radiology report generation holds significant potential to reduce radiologists' workload and enhance diagnostic accuracy. However, generating precise and clinically meaningful reports from chest radiographs remains challenging due to the complexity of medical language and the need for contextual understanding. Existing models often struggle with maintaining both accuracy and contextual relevance. In this paper, we present LLaMA-XR, a novel framework that integrates LLaMA 3.1 with DenseNet-121-based image embeddings and Quantized Low-Rank Adaptation (QLoRA) fine-tuning. LLaMA-XR achieves improved coherence and clinical accuracy while maintaining computational efficiency. This efficiency is driven by an optimization strategy that enhances parameter utilization and reduces memory overhead, enabling faster report generation with lower computational resource demands. Extensive experiments conducted on the IU X-ray benchmark dataset demonstrate that LLaMA-XR outperforms a range of state-of-the-art methods. Our model achieves a ROUGE-L score of 0.433 and a METEOR score of 0.336, establishing new performance benchmarks in the domain. These results underscore LLaMA-XR's potential as an effective and efficient AI system for automated radiology reporting, offering enhanced clinical utility and reliability.
>
---
#### [new 136] Edge Computing for Physics-Driven AI in Computational MRI: A Feasibility Study
- **分类: eess.IV; cs.AI; cs.AR; cs.CV; cs.LG; physics.med-ph**

- **简介: 该论文属于医学影像重建任务，旨在解决高分辨率MRI数据处理中的传输、存储和实时性挑战。工作提出了一种面向FPGA边缘计算设备的物理驱动AI方法，通过8位复数量化和去除冗余FFT/IFFT操作，提升计算效率，同时保持重建质量，实现了资源受限设备上的高质量MRI重建。**

- **链接: [http://arxiv.org/pdf/2506.03183v1](http://arxiv.org/pdf/2506.03183v1)**

> **作者:** Yaşar Utku Alçalar; Yu Cao; Mehmet Akçakaya
>
> **备注:** IEEE International Conference on Future Internet of Things and Cloud (FiCloud), 2025
>
> **摘要:** Physics-driven artificial intelligence (PD-AI) reconstruction methods have emerged as the state-of-the-art for accelerating MRI scans, enabling higher spatial and temporal resolutions. However, the high resolution of these scans generates massive data volumes, leading to challenges in transmission, storage, and real-time processing. This is particularly pronounced in functional MRI, where hundreds of volumetric acquisitions further exacerbate these demands. Edge computing with FPGAs presents a promising solution for enabling PD-AI reconstruction near the MRI sensors, reducing data transfer and storage bottlenecks. However, this requires optimization of PD-AI models for hardware efficiency through quantization and bypassing traditional FFT-based approaches, which can be a limitation due to their computational demands. In this work, we propose a novel PD-AI computational MRI approach optimized for FPGA-based edge computing devices, leveraging 8-bit complex data quantization and eliminating redundant FFT/IFFT operations. Our results show that this strategy improves computational efficiency while maintaining reconstruction quality comparable to conventional PD-AI methods, and outperforms standard clinical methods. Our approach presents an opportunity for high-resolution MRI reconstruction on resource-constrained devices, highlighting its potential for real-world deployment.
>
---
#### [new 137] Conformal coronary calcification volume estimation with conditional coverage via histogram clustering
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决冠状动脉钙化自动评分的可靠性问题。通过提出一种基于聚类的条件共形预测框架，为现有分割模型生成校准的预测区间，从而提升风险分类的可信度与分诊效果。**

- **链接: [http://arxiv.org/pdf/2506.04030v1](http://arxiv.org/pdf/2506.04030v1)**

> **作者:** Olivier Jaubert; Salman Mohammadi; Keith A. Goatman; Shadia S. Mikhael; Conor Bradley; Rebecca Hughes; Richard Good; John H. Hipwell; Sonia Dahdouh
>
> **备注:** IEEE 22nd International Symposium on Biomedical Imaging (ISBI)
>
> **摘要:** Incidental detection and quantification of coronary calcium in CT scans could lead to the early introduction of lifesaving clinical interventions. However, over-reporting could negatively affect patient wellbeing and unnecessarily burden the medical system. Therefore, careful considerations should be taken when automatically reporting coronary calcium scores. A cluster-based conditional conformal prediction framework is proposed to provide score intervals with calibrated coverage from trained segmentation networks without retraining. The proposed method was tuned and used to calibrate predictive intervals for 3D UNet models (deterministic, MCDropout and deep ensemble) reaching similar coverage with better triage metrics compared to conventional conformal prediction. Meaningful predictive intervals of calcium scores could help triage patients according to the confidence of their risk category prediction.
>
---
#### [new 138] Advancing Multimodal Reasoning: From Optimized Cold Start to Staged Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决多模态大语言模型（MLLM）在复杂推理上表现不足的问题。通过优化冷启动、改进强化学习（RL）策略，提出ReVisual-R1模型，结合分阶段训练提升多模态推理能力，在多个基准测试中达到新SOTA。**

- **链接: [http://arxiv.org/pdf/2506.04207v1](http://arxiv.org/pdf/2506.04207v1)**

> **作者:** Shuang Chen; Yue Guo; Zhaochen Su; Yafu Li; Yulun Wu; Jiacheng Chen; Jiayu Chen; Weijie Wang; Xiaoye Qu; Yu Cheng
>
> **备注:** 19 pages, 6 figures
>
> **摘要:** Inspired by the remarkable reasoning capabilities of Deepseek-R1 in complex textual tasks, many works attempt to incentivize similar capabilities in Multimodal Large Language Models (MLLMs) by directly applying reinforcement learning (RL). However, they still struggle to activate complex reasoning. In this paper, rather than examining multimodal RL in isolation, we delve into current training pipelines and identify three crucial phenomena: 1) Effective cold start initialization is critical for enhancing MLLM reasoning. Intriguingly, we find that initializing with carefully selected text data alone can lead to performance surpassing many recent multimodal reasoning models, even before multimodal RL. 2) Standard GRPO applied to multimodal RL suffers from gradient stagnation, which degrades training stability and performance. 3) Subsequent text-only RL training, following the multimodal RL phase, further enhances multimodal reasoning. This staged training approach effectively balances perceptual grounding and cognitive reasoning development. By incorporating the above insights and addressing multimodal RL issues, we introduce ReVisual-R1, achieving a new state-of-the-art among open-source 7B MLLMs on challenging benchmarks including MathVerse, MathVision, WeMath, LogicVista, DynaMath, and challenging AIME2024 and AIME2025.
>
---
#### [new 139] Lightweight Convolutional Neural Networks for Retinal Disease Classification
- **分类: eess.IV; cs.AI; cs.CV; cs.LG; cs.NE**

- **简介: 该论文属于医学图像分类任务，旨在解决糖尿病视网膜病变和黄斑裂孔的早期检测问题。作者使用轻量级卷积神经网络MobileNet和NASNetMobile，在RFMiD数据集上进行训练与评估，通过迁移学习与数据增强提升模型性能。实验表明MobileNetV2准确率达90.8%，优于NASNetMobile。**

- **链接: [http://arxiv.org/pdf/2506.03186v1](http://arxiv.org/pdf/2506.03186v1)**

> **作者:** Duaa Kareem Qasim; Sabah Abdulazeez Jebur; Lafta Raheem Ali; Abdul Jalil M. Khalaf; Abir Jaafar Hussain
>
> **摘要:** Retinal diseases such as Diabetic Retinopathy (DR) and Macular Hole (MH) significantly impact vision and affect millions worldwide. Early detection is crucial, as DR, a complication of diabetes, damages retinal blood vessels, potentially leading to blindness, while MH disrupts central vision, affecting tasks like reading and facial recognition. This paper employed two lightweight and efficient Convolution Neural Network architectures, MobileNet and NASNetMobile, for the classification of Normal, DR, and MH retinal images. The models were trained on the RFMiD dataset, consisting of 3,200 fundus images, after undergoing preprocessing steps such as resizing, normalization, and augmentation. To address data scarcity, this study leveraged transfer learning and data augmentation techniques, enhancing model generalization and performance. The experimental results demonstrate that MobileNetV2 achieved the highest accuracy of 90.8%, outperforming NASNetMobile, which achieved 89.5% accuracy. These findings highlight the effectiveness of CNNs in retinal disease classification, providing a foundation for AI-assisted ophthalmic diagnosis and early intervention.
>
---
#### [new 140] Encoding of Demographic and Anatomical Information in Chest X-Ray-based Severe Left Ventricular Hypertrophy Classifiers
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学影像分类任务，旨在解决通过胸部X光片直接预测严重左心室肥厚的问题。作者提出了一种不依赖解剖测量或人口统计输入的分类框架，使用互信息神经估计量化特征表达能力，实现了高准确性的模型并增强了可解释性。**

- **链接: [http://arxiv.org/pdf/2506.03192v1](http://arxiv.org/pdf/2506.03192v1)**

> **作者:** Basudha Pal; Rama Chellappa; Muhammad Umair
>
> **摘要:** While echocardiography and MRI are clinical standards for evaluating cardiac structure, their use is limited by cost and accessibility.We introduce a direct classification framework that predicts severe left ventricular hypertrophy from chest X-rays, without relying on anatomical measurements or demographic inputs. Our approach achieves high AUROC and AUPRC, and employs Mutual Information Neural Estimation to quantify feature expressivity. This reveals clinically meaningful attribute encoding and supports transparent model interpretation.
>
---
#### [new 141] Adaptive and Robust Image Processing on CubeSats
- **分类: eess.IV; cs.CV; cs.DC; cs.LG**

- **简介: 该论文属于嵌入式系统与图像处理任务，旨在解决资源受限的CubeSats在空间图像处理中面临的适应性与鲁棒性不足的问题。论文提出了DIPP框架实现模块化、可配置的图像处理流水线，提升适应性与鲁棒性；并提出DISH语言及其运行时系统，用于高效调度复杂成像任务。实验验证了其低开销、低网络需求和内存效率优势。**

- **链接: [http://arxiv.org/pdf/2506.03152v1](http://arxiv.org/pdf/2506.03152v1)**

> **作者:** Robert Bayer; Julian Priest; Daniel Kjellberg; Jeppe Lindhard; Nikolaj Sørenesen; Nicolaj Valsted; Ívar Óli; Pınar Tözün
>
> **摘要:** CubeSats offer a low-cost platform for space research, particularly for Earth observation. However, their resource-constrained nature and being in space, challenge the flexibility and complexity of the deployed image processing pipelines and their orchestration. This paper introduces two novel systems, DIPP and DISH, to address these challenges. DIPP is a modular and configurable image processing pipeline framework that allows for adaptability to changing mission goals even after deployment, while preserving robustness. DISH is a domain-specific language (DSL) and runtime system designed to schedule complex imaging workloads on low-power and memory-constrained processors. Our experiments demonstrate that DIPP's decomposition of the processing pipelines adds negligible overhead, while significantly reducing the network requirements of updating pipelines and being robust against erroneous module uploads. Furthermore, we compare DISH to Lua, a general purpose scripting language, and demonstrate its comparable expressiveness and lower memory requirement.
>
---
#### [new 142] Multi-Analyte, Swab-based Automated Wound Monitor with AI
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医疗监测任务，旨在解决糖尿病足溃疡（DFUs）早期识别与监测问题。研究开发了一种低成本、多分析物的3D打印传感器集成棉签及配套iOS应用，通过图像对比和计算机视觉技术，实现伤口严重程度的自动分析与实时监测，提升慢性伤口护理效率。**

- **链接: [http://arxiv.org/pdf/2506.03188v1](http://arxiv.org/pdf/2506.03188v1)**

> **作者:** Madhu Babu Sikha; Lalith Appari; Gurudatt Nanjanagudu Ganesh; Amay Bandodkar; Imon Banerjee
>
> **备注:** 4 pages conference paper
>
> **摘要:** Diabetic foot ulcers (DFUs), a class of chronic wounds, affect ~750,000 individuals every year in the US alone and identifying non-healing DFUs that develop to chronic wounds early can drastically reduce treatment costs and minimize risks of amputation. There is therefore a pressing need for diagnostic tools that can detect non-healing DFUs early. We develop a low cost, multi-analyte 3D printed assays seamlessly integrated on swabs that can identify non-healing DFUs and a Wound Sensor iOS App - an innovative mobile application developed for the controlled acquisition and automated analysis of wound sensor data. By comparing both the original base image (before exposure to the wound) and the wound-exposed image, we developed automated computer vision techniques to compare density changes between the two assay images, which allow us to automatically determine the severity of the wound. The iOS app ensures accurate data collection and presents actionable insights, despite challenges such as variations in camera configurations and ambient conditions. The proposed integrated sensor and iOS app will allow healthcare professionals to monitor wound conditions real-time, track healing progress, and assess critical parameters related to wound care.
>
---
#### [new 143] Knowledge Graphs for Digitized Manuscripts in Jagiellonian Digital Library Application
- **分类: cs.DL; cs.CV**

- **简介: 该论文旨在解决数字化手稿元数据不完整和标准化不足的问题，属于知识图谱构建任务。通过结合计算机视觉、人工智能与语义网技术，对雅盖隆数字图书馆中的手稿和摇篮本进行元数据增强，并构建知识图谱以提升检索与关联能力。**

- **链接: [http://arxiv.org/pdf/2506.03180v1](http://arxiv.org/pdf/2506.03180v1)**

> **作者:** Jan Ignatowicz; Krzysztof Kutt; Grzegorz J. Nalepa
>
> **摘要:** Digitizing cultural heritage collections has become crucial for preservation of historical artifacts and enhancing their availability to the wider public. Galleries, libraries, archives and museums (GLAM institutions) are actively digitizing their holdings and creates extensive digital collections. Those collections are often enriched with metadata describing items but not exactly their contents. The Jagiellonian Digital Library, standing as a good example of such an effort, offers datasets accessible through protocols like OAI-PMH. Despite these improvements, metadata completeness and standardization continue to pose substantial obstacles, limiting the searchability and potential connections between collections. To deal with these challenges, we explore an integrated methodology of computer vision (CV), artificial intelligence (AI), and semantic web technologies to enrich metadata and construct knowledge graphs for digitized manuscripts and incunabula.
>
---
#### [new 144] Pseudo-Simulation for Autonomous Driving
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 论文提出“伪仿真”方法，用于自动驾驶评估任务，解决现有方法在真实性、计算成本与误差累积上的不足。工作包括：基于真实数据生成多样化的合成观测，通过权重机制匹配车辆行为，并建立公开榜单。**

- **链接: [http://arxiv.org/pdf/2506.04218v1](http://arxiv.org/pdf/2506.04218v1)**

> **作者:** Wei Cao; Marcel Hallgarten; Tianyu Li; Daniel Dauner; Xunjiang Gu; Caojun Wang; Yakov Miron; Marco Aiello; Hongyang Li; Igor Gilitschenski; Boris Ivanovic; Marco Pavone; Andreas Geiger; Kashyap Chitta
>
> **摘要:** Existing evaluation paradigms for Autonomous Vehicles (AVs) face critical limitations. Real-world evaluation is often challenging due to safety concerns and a lack of reproducibility, whereas closed-loop simulation can face insufficient realism or high computational costs. Open-loop evaluation, while being efficient and data-driven, relies on metrics that generally overlook compounding errors. In this paper, we propose pseudo-simulation, a novel paradigm that addresses these limitations. Pseudo-simulation operates on real datasets, similar to open-loop evaluation, but augments them with synthetic observations generated prior to evaluation using 3D Gaussian Splatting. Our key idea is to approximate potential future states the AV might encounter by generating a diverse set of observations that vary in position, heading, and speed. Our method then assigns a higher importance to synthetic observations that best match the AV's likely behavior using a novel proximity-based weighting scheme. This enables evaluating error recovery and the mitigation of causal confusion, as in closed-loop benchmarks, without requiring sequential interactive simulation. We show that pseudo-simulation is better correlated with closed-loop simulations (R^2=0.8) than the best existing open-loop approach (R^2=0.7). We also establish a public leaderboard for the community to benchmark new methodologies with pseudo-simulation. Our code is available at https://github.com/autonomousvision/navsim.
>
---
#### [new 145] A Diffusion-Driven Temporal Super-Resolution and Spatial Consistency Enhancement Framework for 4D MRI imaging
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决4D MRI成像中空间与时间分辨率的权衡问题。传统方法在大形变下易产生配准错误和伪影。作者提出了TSSC-Net框架，结合扩散驱动的时间超分辨网络和基于Mamba的三向模块，在提升时间分辨率的同时增强空间一致性，有效改善动态MRI成像质量。**

- **链接: [http://arxiv.org/pdf/2506.04116v1](http://arxiv.org/pdf/2506.04116v1)**

> **作者:** Xuanru Zhou; Jiarun Liu; Shoujun Yu; Hao Yang; Cheng Li; Tao Tan; Shanshan Wang
>
> **摘要:** In medical imaging, 4D MRI enables dynamic 3D visualization, yet the trade-off between spatial and temporal resolution requires prolonged scan time that can compromise temporal fidelity--especially during rapid, large-amplitude motion. Traditional approaches typically rely on registration-based interpolation to generate intermediate frames. However, these methods struggle with large deformations, resulting in misregistration, artifacts, and diminished spatial consistency. To address these challenges, we propose TSSC-Net, a novel framework that generates intermediate frames while preserving spatial consistency. To improve temporal fidelity under fast motion, our diffusion-based temporal super-resolution network generates intermediate frames using the start and end frames as key references, achieving 6x temporal super-resolution in a single inference step. Additionally, we introduce a novel tri-directional Mamba-based module that leverages long-range contextual information to effectively resolve spatial inconsistencies arising from cross-slice misalignment, thereby enhancing volumetric coherence and correcting cross-slice errors. Extensive experiments were performed on the public ACDC cardiac MRI dataset and a real-world dynamic 4D knee joint dataset. The results demonstrate that TSSC-Net can generate high-resolution dynamic MRI from fast-motion data while preserving structural fidelity and spatial consistency.
>
---
#### [new 146] Object-centric 3D Motion Field for Robot Learning from Human Videos
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.SY; eess.SY**

- **简介: 该论文属于机器人学习任务，旨在从人类视频中提取动作知识用于机器人控制。论文提出了一种基于物体中心的3D运动场表示方法，并设计了相应的预测框架与训练流程，有效提升了动作捕捉精度与策略泛化能力，实现了优于现有方法的表现。**

- **链接: [http://arxiv.org/pdf/2506.04227v1](http://arxiv.org/pdf/2506.04227v1)**

> **作者:** Zhao-Heng Yin; Sherry Yang; Pieter Abbeel
>
> **备注:** Project: https://zhaohengyin.github.io/3DMF
>
> **摘要:** Learning robot control policies from human videos is a promising direction for scaling up robot learning. However, how to extract action knowledge (or action representations) from videos for policy learning remains a key challenge. Existing action representations such as video frames, pixelflow, and pointcloud flow have inherent limitations such as modeling complexity or loss of information. In this paper, we propose to use object-centric 3D motion field to represent actions for robot learning from human videos, and present a novel framework for extracting this representation from videos for zero-shot control. We introduce two novel components in its implementation. First, a novel training pipeline for training a ''denoising'' 3D motion field estimator to extract fine object 3D motions from human videos with noisy depth robustly. Second, a dense object-centric 3D motion field prediction architecture that favors both cross-embodiment transfer and policy generalization to background. We evaluate the system in real world setups. Experiments show that our method reduces 3D motion estimation error by over 50% compared to the latest method, achieve 55% average success rate in diverse tasks where prior approaches fail~($\lesssim 10$\%), and can even acquire fine-grained manipulation skills like insertion.
>
---
#### [new 147] Urban Visibility Hotspots: Quantifying Building Vertex Visibility from Connected Vehicle Trajectories using Spatial Indexing
- **分类: eess.SY; cs.CV; cs.SY; stat.CO**

- **简介: 该论文属于城市视觉热点分析任务，旨在解决广告和街道设施选址问题。通过分析车辆轨迹数据与建筑顶点位置，构建空间索引量化视觉曝光度，发现视觉热点并揭示其分布规律。**

- **链接: [http://arxiv.org/pdf/2506.03365v1](http://arxiv.org/pdf/2506.03365v1)**

> **作者:** Artur Grigorev; Adriana-Simona Mihaita
>
> **摘要:** Effective placement of Out-of-Home advertising and street furniture requires accurate identification of locations offering maximum visual exposure to target audiences, particularly vehicular traffic. Traditional site selection methods often rely on static traffic counts or subjective assessments. This research introduces a data-driven methodology to objectively quantify location visibility by analyzing large-scale connected vehicle trajectory data (sourced from Compass IoT) within urban environments. We model the dynamic driver field-of-view using a forward-projected visibility area for each vehicle position derived from interpolated trajectories. By integrating this with building vertex locations extracted from OpenStreetMap, we quantify the cumulative visual exposure, or ``visibility count'', for thousands of potential points of interest near roadways. The analysis reveals that visibility is highly concentrated, identifying specific ``visual hotspots'' that receive disproportionately high exposure compared to average locations. The core technical contribution involves the construction of a BallTree spatial index over building vertices. This enables highly efficient (O(logN) complexity) radius queries to determine which vertices fall within the viewing circles of millions of trajectory points across numerous trips, significantly outperforming brute-force geometric checks. Analysis reveals two key findings: 1) Visibility is highly concentrated, identifying distinct 'visual hotspots' receiving disproportionately high exposure compared to average locations. 2) The aggregated visibility counts across vertices conform to a Log-Normal distribution.
>
---
#### [new 148] Hybrid Ensemble of Segmentation-Assisted Classification and GBDT for Skin Cancer Detection with Engineered Metadata and Synthetic Lesions from ISIC 2024 Non-Dermoscopic 3D-TBP Images
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在解决皮肤癌早期检测问题。作者结合视觉Transformer与自设计模型提取特征，采用分割辅助分类和GBDT集成，并引入合成病变数据缓解类别不平衡，最终在ISIC 2024数据集上取得最优性能。**

- **链接: [http://arxiv.org/pdf/2506.03420v1](http://arxiv.org/pdf/2506.03420v1)**

> **作者:** Muhammad Zubair Hasan; Fahmida Yasmin Rifat
>
> **备注:** Written as per the requirements of CVPR 2025. It is a 8 page paper without reference
>
> **摘要:** Skin cancer is among the most prevalent and life-threatening diseases worldwide, with early detection being critical to patient outcomes. This work presents a hybrid machine and deep learning-based approach for classifying malignant and benign skin lesions using the SLICE-3D dataset from ISIC 2024, which comprises 401,059 cropped lesion images extracted from 3D Total Body Photography (TBP), emulating non-dermoscopic, smartphone-like conditions. Our method combines vision transformers (EVA02) and our designed convolutional ViT hybrid (EdgeNeXtSAC) to extract robust features, employing a segmentation-assisted classification pipeline to enhance lesion localization. Predictions from these models are fused with a gradient-boosted decision tree (GBDT) ensemble enriched by engineered features and patient-specific relational metrics. To address class imbalance and improve generalization, we augment malignant cases with Stable Diffusion-generated synthetic lesions and apply a diagnosis-informed relabeling strategy to harmonize external datasets into a 3-class format. Using partial AUC (pAUC) above 80 percent true positive rate (TPR) as the evaluation metric, our approach achieves a pAUC of 0.1755 -- the highest among all configurations. These results underscore the potential of hybrid, interpretable AI systems for skin cancer triage in telemedicine and resource-constrained settings.
>
---
#### [new 149] Dreaming up scale invariance via inverse renormalization group
- **分类: cond-mat.stat-mech; cs.CV; cs.LG**

- **简介: 该论文属于物理与机器学习交叉任务，旨在解决如何通过神经网络逆向重构二维伊辛模型的微观构型。他们利用逆重整化群方法，训练极简神经网络从粗粒化状态“生成”微观配置，成功再现临界现象的标度不变性和关键物理量，表明简单规则足以编码复杂系统的普适性。**

- **链接: [http://arxiv.org/pdf/2506.04016v1](http://arxiv.org/pdf/2506.04016v1)**

> **作者:** Adam Rançon; Ulysse Rançon; Tomislav Ivek; Ivan Balog
>
> **备注:** v1: 12 pages, 11 figures, 55 references
>
> **摘要:** We explore how minimal neural networks can invert the renormalization group (RG) coarse-graining procedure in the two-dimensional Ising model, effectively "dreaming up" microscopic configurations from coarse-grained states. This task-formally impossible at the level of configurations-can be approached probabilistically, allowing machine learning models to reconstruct scale-invariant distributions without relying on microscopic input. We demonstrate that even neural networks with as few as three trainable parameters can learn to generate critical configurations, reproducing the scaling behavior of observables such as magnetic susceptibility, heat capacity, and Binder ratios. A real-space renormalization group analysis of the generated configurations confirms that the models capture not only scale invariance but also reproduce nontrivial eigenvalues of the RG transformation. Surprisingly, we find that increasing network complexity by introducing multiple layers offers no significant benefit. These findings suggest that simple local rules, akin to those generating fractal structures, are sufficient to encode the universality of critical phenomena, opening the door to efficient generative models of statistical ensembles in physics.
>
---
#### [new 150] A Comprehensive Study on Medical Image Segmentation using Deep Neural Networks
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在通过深度神经网络提升疾病诊断效率。论文分析了当前智能视觉系统在数据、信息、知识、智能和智慧层面的表现，重点研究可解释人工智能（XAI）以提高模型透明度，并探讨了其在癌症早期检测中的应用，强调了XAI与早期预测对实现“智能”到“智慧”的转变意义。**

- **链接: [http://arxiv.org/pdf/2506.04121v1](http://arxiv.org/pdf/2506.04121v1)**

> **作者:** Loan Dao; Ngoc Quoc Ly
>
> **摘要:** Over the past decade, Medical Image Segmentation (MIS) using Deep Neural Networks (DNNs) has achieved significant performance improvements and holds great promise for future developments. This paper presents a comprehensive study on MIS based on DNNs. Intelligent Vision Systems are often evaluated based on their output levels, such as Data, Information, Knowledge, Intelligence, and Wisdom (DIKIW),and the state-of-the-art solutions in MIS at these levels are the focus of research. Additionally, Explainable Artificial Intelligence (XAI) has become an important research direction, as it aims to uncover the "black box" nature of previous DNN architectures to meet the requirements of transparency and ethics. The study emphasizes the importance of MIS in disease diagnosis and early detection, particularly for increasing the survival rate of cancer patients through timely diagnosis. XAI and early prediction are considered two important steps in the journey from "intelligence" to "wisdom." Additionally, the paper addresses existing challenges and proposes potential solutions to enhance the efficiency of implementing DNN-based MIS.
>
---
#### [new 151] Solving Inverse Problems via Diffusion-Based Priors: An Approximation-Free Ensemble Sampling Approach
- **分类: cs.LG; cs.CV; cs.NA; eess.IV; math.NA; stat.ML**

- **简介: 该论文属于逆问题求解任务，旨在利用扩散模型（DM）进行贝叶斯推断。现有方法依赖启发式近似，影响生成效果。为此，作者提出一种无需近似的集成采样算法，结合扩散模型与序贯蒙特卡洛方法，通过分析先验在扩散过程中的演化，推导出描述后验分布演化的修正偏微分方程，并用加权粒子方法模拟。理论上证明了误差界，实验表明其图像重建效果优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.03979v1](http://arxiv.org/pdf/2506.03979v1)**

> **作者:** Haoxuan Chen; Yinuo Ren; Martin Renqiang Min; Lexing Ying; Zachary Izzo
>
> **备注:** 45 pages
>
> **摘要:** Diffusion models (DMs) have proven to be effective in modeling high-dimensional distributions, leading to their widespread adoption for representing complex priors in Bayesian inverse problems (BIPs). However, current DM-based posterior sampling methods proposed for solving common BIPs rely on heuristic approximations to the generative process. To exploit the generative capability of DMs and avoid the usage of such approximations, we propose an ensemble-based algorithm that performs posterior sampling without the use of heuristic approximations. Our algorithm is motivated by existing works that combine DM-based methods with the sequential Monte Carlo (SMC) method. By examining how the prior evolves through the diffusion process encoded by the pre-trained score function, we derive a modified partial differential equation (PDE) governing the evolution of the corresponding posterior distribution. This PDE includes a modified diffusion term and a reweighting term, which can be simulated via stochastic weighted particle methods. Theoretically, we prove that the error between the true posterior distribution can be bounded in terms of the training error of the pre-trained score function and the number of particles in the ensemble. Empirically, we validate our algorithm on several inverse problems in imaging to show that our method gives more accurate reconstructions compared to existing DM-based methods.
>
---
#### [new 152] A Survey of Deep Learning Video Super-Resolution
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于视频超分辨率任务，旨在通过深度学习技术提升低质量视频的分辨率。论文系统分析了现有方法，总结关键技术，提出分类体系，并探讨了挑战与趋势，以指导未来研究方向。**

- **链接: [http://arxiv.org/pdf/2506.03216v1](http://arxiv.org/pdf/2506.03216v1)**

> **作者:** Arbind Agrahari Baniya; Tsz-Kwan Lee; Peter Eklund; Sunil Aryal
>
> **备注:** This paper has been published in IEEE Transactions on Emerging Topics in Computational Intelligence, vol. 8, no. 4, pp. 2655-2676, Aug. 2024, doi: 10.1109/TETCI.2024.3398015
>
> **摘要:** Video super-resolution (VSR) is a prominent research topic in low-level computer vision, where deep learning technologies have played a significant role. The rapid progress in deep learning and its applications in VSR has led to a proliferation of tools and techniques in the literature. However, the usage of these methods is often not adequately explained, and decisions are primarily driven by quantitative improvements. Given the significance of VSR's potential influence across multiple domains, it is imperative to conduct a comprehensive analysis of the elements and deep learning methodologies employed in VSR research. This methodical analysis will facilitate the informed development of models tailored to specific application needs. In this paper, we present an overarching overview of deep learning-based video super-resolution models, investigating each component and discussing its implications. Furthermore, we provide a synopsis of key components and technologies employed by state-of-the-art and earlier VSR models. By elucidating the underlying methodologies and categorising them systematically, we identified trends, requirements, and challenges in the domain. As a first-of-its-kind survey of deep learning-based VSR models, this work also establishes a multi-level taxonomy to guide current and future VSR research, enhancing the maturation and interpretation of VSR practices for various practical applications.
>
---
#### [new 153] Trajectory Prediction Meets Large Language Models: A Survey
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于轨迹预测任务，旨在解决如何利用大语言模型提升轨迹预测的准确性与可解释性。论文系统梳理了五类方法：语言建模范式、预训练语言模型直接预测、语言引导场景理解、语言驱动数据生成、语言推理与可解释性分析，探讨其设计与挑战。**

- **链接: [http://arxiv.org/pdf/2506.03408v1](http://arxiv.org/pdf/2506.03408v1)**

> **作者:** Yi Xu; Ruining Yang; Yitian Zhang; Yizhou Wang; Jianglin Lu; Mingyuan Zhang; Lili Su; Yun Fu
>
> **备注:** 16 pages, GitHub: https://github.com/colorfulfuture/Awesome-Trajectory-Motion-Prediction-Papers
>
> **摘要:** Recent advances in large language models (LLMs) have sparked growing interest in integrating language-driven techniques into trajectory prediction. By leveraging their semantic and reasoning capabilities, LLMs are reshaping how autonomous systems perceive, model, and predict trajectories. This survey provides a comprehensive overview of this emerging field, categorizing recent work into five directions: (1) Trajectory prediction via language modeling paradigms, (2) Direct trajectory prediction with pretrained language models, (3) Language-guided scene understanding for trajectory prediction, (4) Language-driven data generation for trajectory prediction, (5) Language-based reasoning and interpretability for trajectory prediction. For each, we analyze representative methods, highlight core design choices, and identify open challenges. This survey bridges natural language processing and trajectory prediction, offering a unified perspective on how language can enrich trajectory prediction.
>
---
#### [new 154] Identifying Alzheimer's Disease Prediction Strategies of Convolutional Neural Network Classifiers using R2* Maps and Spectral Clustering
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决阿尔茨海默病（AD）的可解释性预测问题。作者使用R2*图和深度学习模型，结合Layer-wise Relevance Propagation与谱聚类方法，分析不同预处理和训练配置下卷积神经网络的决策策略差异，并通过t-SNE可视化评估聚类结构，强调模型解释性在医疗AI中的重要性。**

- **链接: [http://arxiv.org/pdf/2506.03890v1](http://arxiv.org/pdf/2506.03890v1)**

> **作者:** Christian Tinauer; Maximilian Sackl; Stefan Ropele; Christian Langkammer
>
> **备注:** Accepted for the conference EUSIPCO2025 (https://eusipco2025.org/)
>
> **摘要:** Deep learning models have shown strong performance in classifying Alzheimer's disease (AD) from R2* maps, but their decision-making remains opaque, raising concerns about interpretability. Previous studies suggest biases in model decisions, necessitating further analysis. This study uses Layer-wise Relevance Propagation (LRP) and spectral clustering to explore classifier decision strategies across preprocessing and training configurations using R2* maps. We trained a 3D convolutional neural network on R2* maps, generating relevance heatmaps via LRP and applied spectral clustering to identify dominant patterns. t-Stochastic Neighbor Embedding (t-SNE) visualization was used to assess clustering structure. Spectral clustering revealed distinct decision patterns, with the relevance-guided model showing the clearest separation between AD and normal control (NC) cases. The t-SNE visualization confirmed that this model aligned heatmap groupings with the underlying subject groups. Our findings highlight the significant impact of preprocessing and training choices on deep learning models trained on R2* maps, even with similar performance metrics. Spectral clustering offers a structured method to identify classification strategy differences, emphasizing the importance of explainability in medical AI.
>
---
#### [new 155] Recent Advances in Medical Image Classification
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决医疗影像诊断中自动识别与分类的问题。论文综述了基于人工智能的技术进展，涵盖深度学习模型如卷积神经网络、视觉变换器及视觉语言模型的应用，并探讨了如何应对标注数据不足与结果可解释性等挑战。**

- **链接: [http://arxiv.org/pdf/2506.04129v1](http://arxiv.org/pdf/2506.04129v1)**

> **作者:** Loan Dao; Ngoc Quoc Ly
>
> **摘要:** Medical image classification is crucial for diagnosis and treatment, benefiting significantly from advancements in artificial intelligence. The paper reviews recent progress in the field, focusing on three levels of solutions: basic, specific, and applied. It highlights advances in traditional methods using deep learning models like Convolutional Neural Networks and Vision Transformers, as well as state-of-the-art approaches with Vision Language Models. These models tackle the issue of limited labeled data, and enhance and explain predictive results through Explainable Artificial Intelligence.
>
---
#### [new 156] Robustness in Both Domains: CLIP Needs a Robust Text Encoder
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于自然语言处理与多模态学习任务，旨在解决CLIP模型中文本编码器在对抗攻击下的鲁棒性问题。作者提出了LEAF方法，提升文本编码器的对抗鲁棒性，同时保持图像性能，并验证了其在文本到图像生成和多模态检索中的有效性。**

- **链接: [http://arxiv.org/pdf/2506.03355v1](http://arxiv.org/pdf/2506.03355v1)**

> **作者:** Elias Abad Rocamora; Christian Schlarmann; Naman Deep Singh; Yongtao Wu; Matthias Hein; Volkan Cevher
>
> **摘要:** Adversarial input attacks can cause a significant shift of CLIP embeddings. This can affect the downstream robustness of models incorporating CLIP in the pipeline, such as text-to-image generative models or large vision language models. While some efforts have been done towards making the CLIP image encoders robust, the robustness of text encoders remains unexplored. In this work, we cover this gap in the literature. We propose LEAF: an efficient adversarial finetuning method for the text domain, with the ability to scale to large CLIP models. Our models significantly improve the zero-shot adversarial accuracy in the text domain, while maintaining the vision performance provided by robust image encoders. When combined with text-to-image diffusion models, we can improve the generation quality under adversarial noise. When employing our robust CLIP encoders in multimodal retrieval tasks, we improve the recall under adversarial noise over standard CLIP models. Finally, we show that robust text encoders facilitate better reconstruction of input text from its embedding via direct optimization.
>
---
#### [new 157] Multimodal Tabular Reasoning with Privileged Structured Information
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态表格推理任务，旨在解决从表格图像中进行逻辑推理的问题。由于实际场景中表格常以图像形式存在，缺乏高质量文本表示，作者提出了Turbo框架，利用训练时的结构化信息提升多模态大模型的推理能力。**

- **链接: [http://arxiv.org/pdf/2506.04088v1](http://arxiv.org/pdf/2506.04088v1)**

> **作者:** Jun-Peng Jiang; Yu Xia; Hai-Long Sun; Shiyin Lu; Qing-Guo Chen; Weihua Luo; Kaifu Zhang; De-Chuan Zhan; Han-Jia Ye
>
> **摘要:** Tabular reasoning involves multi-step information extraction and logical inference over tabular data. While recent advances have leveraged large language models (LLMs) for reasoning over structured tables, such high-quality textual representations are often unavailable in real-world settings, where tables typically appear as images. In this paper, we tackle the task of tabular reasoning from table images, leveraging privileged structured information available during training to enhance multimodal large language models (MLLMs). The key challenges lie in the complexity of accurately aligning structured information with visual representations, and in effectively transferring structured reasoning skills to MLLMs despite the input modality gap. To address these, we introduce TabUlar Reasoning with Bridged infOrmation ({\sc Turbo}), a new framework for multimodal tabular reasoning with privileged structured tables. {\sc Turbo} benefits from a structure-aware reasoning trace generator based on DeepSeek-R1, contributing to high-quality modality-bridged data. On this basis, {\sc Turbo} repeatedly generates and selects the advantageous reasoning paths, further enhancing the model's tabular reasoning ability. Experimental results demonstrate that, with limited ($9$k) data, {\sc Turbo} achieves state-of-the-art performance ($+7.2\%$ vs. previous SOTA) across multiple datasets.
>
---
#### [new 158] Towards generating more interpretable counterfactuals via concept vectors: a preliminary study on chest X-rays
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像解释任务，旨在提升模型可解释性。通过概念向量将临床知识映射到生成模型的潜在空间，提取与图像特征关联的可解释概念，并利用这些概念生成强调或减弱特定病理特征的反事实图像，初步验证了方法在胸部X光片中的可行性。**

- **链接: [http://arxiv.org/pdf/2506.04058v1](http://arxiv.org/pdf/2506.04058v1)**

> **作者:** Bulat Maksudov; Kathleen Curran; Alessandra Mileo
>
> **摘要:** An essential step in deploying medical imaging models is ensuring alignment with clinical knowledge and interpretability. We focus on mapping clinical concepts into the latent space of generative models to identify Concept Activation Vectors (CAVs). Using a simple reconstruction autoencoder, we link user-defined concepts to image-level features without explicit label training. The extracted concepts are stable across datasets, enabling visual explanations that highlight clinically relevant features. By traversing latent space along concept directions, we produce counterfactuals that exaggerate or reduce specific clinical features. Preliminary results on chest X-rays show promise for large pathologies like cardiomegaly, while smaller pathologies remain challenging due to reconstruction limits. Although not outperforming baselines, this approach offers a path toward interpretable, concept-based explanations aligned with clinical knowledge.
>
---
#### [new 159] DLiPath: A Benchmark for the Comprehensive Assessment of Donor Liver Based on Histopathological Image Dataset
- **分类: eess.IV; cs.AI; cs.CV; q-bio.QM**

- **简介: 该论文属于医学图像分析任务，旨在解决肝移植中病理特征评估主观性高、效率低的问题。作者构建了DLiPath数据集，包含636张肝脏活检全切片图像及专家标注，并基于此设立9种多实例学习模型作为基线，推动自动化肝脏评估研究。**

- **链接: [http://arxiv.org/pdf/2506.03185v1](http://arxiv.org/pdf/2506.03185v1)**

> **作者:** Liangrui Pan; Xingchen Li; Zhongyi Chen; Ling Chu; Shaoliang Peng
>
> **备注:** Submit to ACM MM2025
>
> **摘要:** Pathologists comprehensive evaluation of donor liver biopsies provides crucial information for accepting or discarding potential grafts. However, rapidly and accurately obtaining these assessments intraoperatively poses a significant challenge for pathologists. Features in donor liver biopsies, such as portal tract fibrosis, total steatosis, macrovesicular steatosis, and hepatocellular ballooning are correlated with transplant outcomes, yet quantifying these indicators suffers from substantial inter- and intra-observer variability. To address this, we introduce DLiPath, the first benchmark for comprehensive donor liver assessment based on a histopathology image dataset. We collected and publicly released 636 whole slide images from 304 donor liver patients at the Department of Pathology, the Third Xiangya Hospital, with expert annotations for key pathological features (including cholestasis, portal tract fibrosis, portal inflammation, total steatosis, macrovesicular steatosis, and hepatocellular ballooning). We selected nine state-of-the-art multiple-instance learning (MIL) models based on the DLiPath dataset as baselines for extensive comparative analysis. The experimental results demonstrate that several MIL models achieve high accuracy across donor liver assessment indicators on DLiPath, charting a clear course for future automated and intelligent donor liver assessment research. Data and code are available at https://github.com/panliangrui/ACM_MM_2025.
>
---
#### [new 160] DUAL: Dynamic Uncertainty-Aware Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出DUAL框架，解决深度学习中单模态与多模态场景下的特征不确定性问题。通过动态建模不确定性、自适应调整样本影响及跨模态关系学习，提升模型性能与可靠性，在多个视觉与多模态任务上取得显著效果。**

- **链接: [http://arxiv.org/pdf/2506.03158v1](http://arxiv.org/pdf/2506.03158v1)**

> **作者:** Jiahao Qin; Bei Peng; Feng Liu; Guangliang Cheng; Lu Zong
>
> **备注:** 12 pages, 3 figures
>
> **摘要:** Deep learning models frequently encounter feature uncertainty in diverse learning scenarios, significantly impacting their performance and reliability. This challenge is particularly complex in multi-modal scenarios, where models must integrate information from different sources with inherent uncertainties. We propose Dynamic Uncertainty-Aware Learning (DUAL), a unified framework that effectively handles feature uncertainty in both single-modal and multi-modal scenarios. DUAL introduces three key innovations: Dynamic Feature Uncertainty Modeling, which continuously refines uncertainty estimates through joint consideration of feature characteristics and learning dynamics; Adaptive Distribution-Aware Modulation, which maintains balanced feature distributions through dynamic sample influence adjustment; and Uncertainty-aware Cross-Modal Relationship Learning, which explicitly models uncertainties in cross-modal interactions. Through extensive experiments, we demonstrate DUAL's effectiveness across multiple domains: in computer vision tasks, it achieves substantial improvements of 7.1% accuracy on CIFAR-10, 6.5% accuracy on CIFAR-100, and 2.3% accuracy on Tiny-ImageNet; in multi-modal learning, it demonstrates consistent gains of 4.1% accuracy on CMU-MOSEI and 2.8% accuracy on CMU-MOSI for sentiment analysis, while achieving 1.4% accuracy improvements on MISR. The code will be available on GitHub soon.
>
---
#### [new 161] Structural Vibration Monitoring with Diffractive Optical Processors
- **分类: physics.optics; cs.CV; cs.LG; physics.app-ph**

- **简介: 该论文属于结构健康监测任务，旨在解决传统方法在成本、功耗和数据处理方面的局限性。作者提出了一种基于衍射光学处理器的振动监测系统，结合优化的衍射层与浅层神经网络，实现低功耗、高精度的3D结构振动谱提取，验证了其在实验室建筑模型上的有效性，并展示了其在多个领域的应用潜力。**

- **链接: [http://arxiv.org/pdf/2506.03317v1](http://arxiv.org/pdf/2506.03317v1)**

> **作者:** Yuntian Wang; Zafer Yilmaz; Yuhang Li; Edward Liu; Eric Ahlberg; Farid Ghahari; Ertugrul Taciroglu; Aydogan Ozcan
>
> **备注:** 33 Pages, 8 Figures, 1 Table
>
> **摘要:** Structural Health Monitoring (SHM) is vital for maintaining the safety and longevity of civil infrastructure, yet current solutions remain constrained by cost, power consumption, scalability, and the complexity of data processing. Here, we present a diffractive vibration monitoring system, integrating a jointly optimized diffractive layer with a shallow neural network-based backend to remotely extract 3D structural vibration spectra, offering a low-power, cost-effective and scalable solution. This architecture eliminates the need for dense sensor arrays or extensive data acquisition; instead, it uses a spatially-optimized passive diffractive layer that encodes 3D structural displacements into modulated light, captured by a minimal number of detectors and decoded in real-time by shallow and low-power neural networks to reconstruct the 3D displacement spectra of structures. The diffractive system's efficacy was demonstrated both numerically and experimentally using millimeter-wave illumination on a laboratory-scale building model with a programmable shake table. Our system achieves more than an order-of-magnitude improvement in accuracy over conventional optics or separately trained modules, establishing a foundation for high-throughput 3D monitoring of structures. Beyond SHM, the 3D vibration monitoring capabilities of this cost-effective and data-efficient framework establish a new computational sensing modality with potential applications in disaster resilience, aerospace diagnostics, and autonomous navigation, where energy efficiency, low latency, and high-throughput are critical.
>
---
#### [new 162] Rethinking the Stability-Plasticity Trade-off in Continual Learning from an Architectural Perspective
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于持续学习任务，旨在解决神经网络在增量学习中的稳定性-可塑性权衡问题。现有方法多关注参数层面的权衡，忽视了网络结构的影响。本文从结构层面重新审视该问题，提出深度网络更可塑、宽度网络更稳定，并设计了一个双网络框架Dual-Arch，分别优化稳定性与可塑性。实验表明该方法在提升性能的同时大幅减少参数量。**

- **链接: [http://arxiv.org/pdf/2506.03951v1](http://arxiv.org/pdf/2506.03951v1)**

> **作者:** Aojun Lu; Hangjie Yuan; Tao Feng; Yanan Sun
>
> **摘要:** The quest for Continual Learning (CL) seeks to empower neural networks with the ability to learn and adapt incrementally. Central to this pursuit is addressing the stability-plasticity dilemma, which involves striking a balance between two conflicting objectives: preserving previously learned knowledge and acquiring new knowledge. While numerous CL methods aim to achieve this trade-off, they often overlook the impact of network architecture on stability and plasticity, restricting the trade-off to the parameter level. In this paper, we delve into the conflict between stability and plasticity at the architectural level. We reveal that under an equal parameter constraint, deeper networks exhibit better plasticity, while wider networks are characterized by superior stability. To address this architectural-level dilemma, we introduce a novel framework denoted Dual-Arch, which serves as a plug-in component for CL. This framework leverages the complementary strengths of two distinct and independent networks: one dedicated to plasticity and the other to stability. Each network is designed with a specialized and lightweight architecture, tailored to its respective objective. Extensive experiments demonstrate that Dual-Arch enhances the performance of existing CL methods while being up to 87% more compact in terms of parameters.
>
---
#### [new 163] Seeing What Tastes Good: Revisiting Multimodal Distributional Semantics in the Billion Parameter Era
- **分类: cs.CL; cs.CV**

- **简介: 本文研究大规模基础模型如何表示具体物体概念的语义特征，属于自然语言处理与多模态学习任务。论文旨在探究这些模型是否能捕捉如颜色、气味等功能属性。作者通过探测任务评估单模态和多模态模型对McRae规范及Binder数据集属性评分的预测能力，发现多模态模型表现略优，图像模型在非视觉属性上也具竞争力。**

- **链接: [http://arxiv.org/pdf/2506.03994v1](http://arxiv.org/pdf/2506.03994v1)**

> **作者:** Dan Oneata; Desmond Elliott; Stella Frank
>
> **备注:** ACL Findings 2025
>
> **摘要:** Human learning and conceptual representation is grounded in sensorimotor experience, in contrast to state-of-the-art foundation models. In this paper, we investigate how well such large-scale models, trained on vast quantities of data, represent the semantic feature norms of concrete object concepts, e.g. a ROSE is red, smells sweet, and is a flower. More specifically, we use probing tasks to test which properties of objects these models are aware of. We evaluate image encoders trained on image data alone, as well as multimodally-trained image encoders and language-only models, on predicting an extended denser version of the classic McRae norms and the newer Binder dataset of attribute ratings. We find that multimodal image encoders slightly outperform language-only approaches, and that image-only encoders perform comparably to the language models, even on non-visual attributes that are classified as "encyclopedic" or "function". These results offer new insights into what can be learned from pure unimodal learning, and the complementarity of the modalities.
>
---
#### [new 164] SplArt: Articulation Estimation and Part-Level Reconstruction with 3D Gaussian Splatting
- **分类: cs.GR; cs.CV; cs.LG; cs.MM; cs.RO**

- **简介: 该论文属于三维重建与运动估计任务，旨在解决日常场景中关节物体的精细重建与运动状态估计问题。现有方法依赖三维标注或易陷入局部最优。SplArt提出一种自监督、类别无关的框架，基于3D高斯点绘制技术，仅需两组不同姿态的RGB图像即可实现高质量部分分割、重建与运动估计，并支持实时渲染。**

- **链接: [http://arxiv.org/pdf/2506.03594v1](http://arxiv.org/pdf/2506.03594v1)**

> **作者:** Shengjie Lin; Jiading Fang; Muhammad Zubair Irshad; Vitor Campagnolo Guizilini; Rares Andrei Ambrus; Greg Shakhnarovich; Matthew R. Walter
>
> **备注:** https://github.com/ripl/splart
>
> **摘要:** Reconstructing articulated objects prevalent in daily environments is crucial for applications in augmented/virtual reality and robotics. However, existing methods face scalability limitations (requiring 3D supervision or costly annotations), robustness issues (being susceptible to local optima), and rendering shortcomings (lacking speed or photorealism). We introduce SplArt, a self-supervised, category-agnostic framework that leverages 3D Gaussian Splatting (3DGS) to reconstruct articulated objects and infer kinematics from two sets of posed RGB images captured at different articulation states, enabling real-time photorealistic rendering for novel viewpoints and articulations. SplArt augments 3DGS with a differentiable mobility parameter per Gaussian, achieving refined part segmentation. A multi-stage optimization strategy is employed to progressively handle reconstruction, part segmentation, and articulation estimation, significantly enhancing robustness and accuracy. SplArt exploits geometric self-supervision, effectively addressing challenging scenarios without requiring 3D annotations or category-specific priors. Evaluations on established and newly proposed benchmarks, along with applications to real-world scenarios using a handheld RGB camera, demonstrate SplArt's state-of-the-art performance and real-world practicality. Code is publicly available at https://github.com/ripl/splart.
>
---
## 更新

#### [replaced 001] CMAR-Net: Accurate Cross-Modal 3D SAR Reconstruction of Vehicle Targets with Sparse-Aspect Multi-Baseline Data
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2406.04158v4](http://arxiv.org/pdf/2406.04158v4)**

> **作者:** Da Li; Guoqiang Zhao; Chen Yao; Kaiqiang Zhu; Houjun Sun; Jiacheng Bao
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Sparse-aspect multi-baseline Synthetic Aperture Radar (SAR) three-dimensional (3D) tomography is a crucial remote sensing technique. Compared to full-aspect observation, it needs only a few observation aspects to achieve a sufficiently clear 3D scene reconstruction, providing a cost-effective alternative. In the past, compressive sensing (CS) was the mainstream approach for sparse 3D SAR imaging. Recently, deep learning (DL) revolutionizes this field through its powerful data-driven representation capabilities and efficient inference characteristics. However, existing DL methods primarily depend on high-resolution radar images for supervising the training of deep neural networks (DNNs). This unimodal approach precludes the incorporation of complementary information from other data sources, thereby limiting potential improvements in imaging performance. In this paper, we propose a Cross-Modal 3D-SAR Reconstruction Network (CMAR-Net) that enhances 3D SAR imaging by fusing heterogeneous information. Leveraging cross-modal supervision from 2D optical images and error transfer guaranteed by differentiable rendering, CMAR-Net achieves efficient training and reconstructs highly sparse-aspect multi-baseline SAR image into visually structured and accurate 3D images, particularly for vehicle targets. Extensive experiments on simulated and real-world datasets demonstrate that CMAR-Net significantly outperforms state-of-the-art sparse reconstruction algorithms based on CS and DL, with average improvements of 75.83% in PSNR and 47.85% in SSIM. Furthermore, our method eliminates the need for time-consuming full-aperture data preprocessing and relies solely on computer-rendered optical images, significantly reducing dataset construction costs. This work highlights the potential of cross-modal learning for multi-baseline SAR 3D imaging and introduces a novel framework for radar imaging research.
>
---
#### [replaced 002] Rate-In: Information-Driven Adaptive Dropout Rates for Improved Inference-Time Uncertainty Estimation
- **分类: cs.LG; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2412.07169v4](http://arxiv.org/pdf/2412.07169v4)**

> **作者:** Tal Zeevi; Ravid Shwartz-Ziv; Yann LeCun; Lawrence H. Staib; John A. Onofrey
>
> **备注:** Accepted to the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2025. Code available at: https://github.com/code-supplement-25/rate-in
>
> **摘要:** Accurate uncertainty estimation is crucial for deploying neural networks in risk-sensitive applications such as medical diagnosis. Monte Carlo Dropout is a widely used technique for approximating predictive uncertainty by performing stochastic forward passes with dropout during inference. However, using static dropout rates across all layers and inputs can lead to suboptimal uncertainty estimates, as it fails to adapt to the varying characteristics of individual inputs and network layers. Existing approaches optimize dropout rates during training using labeled data, resulting in fixed inference-time parameters that cannot adjust to new data distributions, compromising uncertainty estimates in Monte Carlo simulations. In this paper, we propose Rate-In, an algorithm that dynamically adjusts dropout rates during inference by quantifying the information loss induced by dropout in each layer's feature maps. By treating dropout as controlled noise injection and leveraging information-theoretic principles, Rate-In adapts dropout rates per layer and per input instance without requiring ground truth labels. By quantifying the functional information loss in feature maps, we adaptively tune dropout rates to maintain perceptual quality across diverse medical imaging tasks and architectural configurations. Our extensive empirical study on synthetic data and real-world medical imaging tasks demonstrates that Rate-In improves calibration and sharpens uncertainty estimates compared to fixed or heuristic dropout rates without compromising predictive performance. Rate-In offers a practical, unsupervised, inference-time approach to optimizing dropout for more reliable predictive uncertainty estimation in critical applications.
>
---
#### [replaced 003] Flow-GRPO: Training Flow Matching Models via Online RL
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.05470v3](http://arxiv.org/pdf/2505.05470v3)**

> **作者:** Jie Liu; Gongye Liu; Jiajun Liang; Yangguang Li; Jiaheng Liu; Xintao Wang; Pengfei Wan; Di Zhang; Wanli Ouyang
>
> **备注:** Code: https://github.com/yifan123/flow_grpo
>
> **摘要:** We propose Flow-GRPO, the first method integrating online reinforcement learning (RL) into flow matching models. Our approach uses two key strategies: (1) an ODE-to-SDE conversion that transforms a deterministic Ordinary Differential Equation (ODE) into an equivalent Stochastic Differential Equation (SDE) that matches the original model's marginal distribution at all timesteps, enabling statistical sampling for RL exploration; and (2) a Denoising Reduction strategy that reduces training denoising steps while retaining the original inference timestep number, significantly improving sampling efficiency without performance degradation. Empirically, Flow-GRPO is effective across multiple text-to-image tasks. For complex compositions, RL-tuned SD3.5 generates nearly perfect object counts, spatial relations, and fine-grained attributes, boosting GenEval accuracy from 63% to 95%. In visual text rendering, its accuracy improves from 59% to 92%, significantly enhancing text generation. Flow-GRPO also achieves substantial gains in human preference alignment. Notably, very little reward hacking occurred, meaning rewards did not increase at the cost of appreciable image quality or diversity degradation.
>
---
#### [replaced 004] A Flag Decomposition for Hierarchical Datasets
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.07782v2](http://arxiv.org/pdf/2502.07782v2)**

> **作者:** Nathan Mankovich; Ignacio Santamaria; Gustau Camps-Valls; Tolga Birdal
>
> **摘要:** Flag manifolds encode nested sequences of subspaces and serve as powerful structures for various computer vision and machine learning applications. Despite their utility in tasks such as dimensionality reduction, motion averaging, and subspace clustering, current applications are often restricted to extracting flags using common matrix decomposition methods like the singular value decomposition. Here, we address the need for a general algorithm to factorize and work with hierarchical datasets. In particular, we propose a novel, flag-based method that decomposes arbitrary hierarchical real-valued data into a hierarchy-preserving flag representation in Stiefel coordinates. Our work harnesses the potential of flag manifolds in applications including denoising, clustering, and few-shot learning.
>
---
#### [replaced 005] Skywork R1V2: Multimodal Hybrid Reinforcement Learning for Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.16656v3](http://arxiv.org/pdf/2504.16656v3)**

> **作者:** Chris; Yichen Wei; Yi Peng; Xiaokun Wang; Weijie Qiu; Wei Shen; Tianyidan Xie; Jiangbo Pei; Jianhao Zhang; Yunzhuo Hao; Xuchen Song; Yang Liu; Yahui Zhou
>
> **摘要:** We present Skywork R1V2, a next-generation multimodal reasoning model and a major leap forward from its predecessor, Skywork R1V. At its core, R1V2 introduces a hybrid reinforcement learning paradigm that jointly leverages the Mixed Preference Optimization (MPO) and the Group Relative Policy Optimization (GRPO), which harmonizes reward-model guidance with rule-based strategies, thereby addressing the long-standing challenge of balancing sophisticated reasoning capabilities with broad generalization. To further enhance training efficiency, we propose the Selective Sample Buffer (SSB) mechanism, which effectively addresses the vanishing advantages dilemma inherent in GRPO by prioritizing high-value samples throughout the optimization process. Notably, we observe that excessive reinforcement signals can induce visual hallucinations--a phenomenon we systematically monitor and mitigate through calibrated reward thresholds throughout the training process. Empirical results affirm the exceptional capability of R1V2, with benchmark-leading performances such as 62.6 on OlympiadBench, 78.9 on AIME2024, 63.6 on LiveCodeBench, and 73.6 on MMMU. These results underscore R1V2's superiority over existing open-source models and demonstrate significant progress in closing the performance gap with premier proprietary systems, including Gemini 2.5 and OpenAI-o4-mini. The Skywork R1V2 model weights have been publicly released to promote openness and reproducibility https://huggingface.co/Skywork/Skywork-R1V2-38B.
>
---
#### [replaced 006] LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning
- **分类: cs.LG; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16933v2](http://arxiv.org/pdf/2505.16933v2)**

> **作者:** Zebin You; Shen Nie; Xiaolu Zhang; Jun Hu; Jun Zhou; Zhiwu Lu; Ji-Rong Wen; Chongxuan Li
>
> **备注:** Project page and codes: \url{https://ml-gsai.github.io/LLaDA-V-demo/}
>
> **摘要:** In this work, we introduce LLaDA-V, a purely diffusion-based Multimodal Large Language Model (MLLM) that integrates visual instruction tuning with masked diffusion models, representing a departure from the autoregressive paradigms dominant in current multimodal approaches. Built upon LLaDA, a representative large language diffusion model, LLaDA-V incorporates a vision encoder and MLP connector that projects visual features into the language embedding space, enabling effective multimodal alignment. Our empirical investigation reveals several intriguing results: First, LLaDA-V demonstrates promising multimodal performance despite its language model being weaker on purely textual tasks than counterparts like LLaMA3-8B and Qwen2-7B. When trained on the same instruction data, LLaDA-V is highly competitive to LLaMA3-V across multimodal tasks with better data scalability. It also narrows the performance gap to Qwen2-VL, suggesting the effectiveness of its architecture for multimodal tasks. Second, LLaDA-V achieves state-of-the-art performance in multimodal understanding compared to existing hybrid autoregressive-diffusion and purely diffusion-based MLLMs. Our findings suggest that large language diffusion models show promise in multimodal contexts and warrant further investigation in future research. Project page and codes: https://ml-gsai.github.io/LLaDA-V-demo/.
>
---
#### [replaced 007] Galileo: Learning Global & Local Features of Many Remote Sensing Modalities
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.09356v3](http://arxiv.org/pdf/2502.09356v3)**

> **作者:** Gabriel Tseng; Anthony Fuller; Marlena Reil; Henry Herzog; Patrick Beukema; Favyen Bastani; James R. Green; Evan Shelhamer; Hannah Kerner; David Rolnick
>
> **摘要:** We introduce a highly multimodal transformer to represent many remote sensing modalities - multispectral optical, synthetic aperture radar, elevation, weather, pseudo-labels, and more - across space and time. These inputs are useful for diverse remote sensing tasks, such as crop mapping and flood detection. However, learning shared representations of remote sensing data is challenging, given the diversity of relevant data modalities, and because objects of interest vary massively in scale, from small boats (1-2 pixels and fast) to glaciers (thousands of pixels and slow). We present a novel self-supervised learning algorithm that extracts multi-scale features across a flexible set of input modalities through masked modeling. Our dual global and local contrastive losses differ in their targets (deep representations vs. shallow input projections) and masking strategies (structured vs. not). Our Galileo is a single generalist model that outperforms SoTA specialist models for satellite images and pixel time series across eleven benchmarks and multiple tasks.
>
---
#### [replaced 008] A Survey on (M)LLM-Based GUI Agents
- **分类: cs.HC; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.13865v2](http://arxiv.org/pdf/2504.13865v2)**

> **作者:** Fei Tang; Haolei Xu; Hang Zhang; Siqi Chen; Xingyu Wu; Yongliang Shen; Wenqi Zhang; Guiyang Hou; Zeqi Tan; Yuchen Yan; Kaitao Song; Jian Shao; Weiming Lu; Jun Xiao; Yueting Zhuang
>
> **摘要:** Graphical User Interface (GUI) Agents have emerged as a transformative paradigm in human-computer interaction, evolving from rule-based automation scripts to sophisticated AI-driven systems capable of understanding and executing complex interface operations. This survey provides a comprehensive examination of the rapidly advancing field of LLM-based GUI Agents, systematically analyzing their architectural foundations, technical components, and evaluation methodologies. We identify and analyze four fundamental components that constitute modern GUI Agents: (1) perception systems that integrate text-based parsing with multimodal understanding for comprehensive interface comprehension; (2) exploration mechanisms that construct and maintain knowledge bases through internal modeling, historical experience, and external information retrieval; (3) planning frameworks that leverage advanced reasoning methodologies for task decomposition and execution; and (4) interaction systems that manage action generation with robust safety controls. Through rigorous analysis of these components, we reveal how recent advances in large language models and multimodal learning have revolutionized GUI automation across desktop, mobile, and web platforms. We critically examine current evaluation frameworks, highlighting methodological limitations in existing benchmarks while proposing directions for standardization. This survey also identifies key technical challenges, including accurate element localization, effective knowledge retrieval, long-horizon planning, and safety-aware execution control, while outlining promising research directions for enhancing GUI Agents' capabilities. Our systematic review provides researchers and practitioners with a thorough understanding of the field's current state and offers insights into future developments in intelligent interface automation.
>
---
#### [replaced 009] High Performance Space Debris Tracking in Complex Skylight Backgrounds with a Large-Scale Dataset
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.02614v2](http://arxiv.org/pdf/2506.02614v2)**

> **作者:** Guohang Zhuang; Weixi Song; Jinyang Huang; Chenwei Yang; Yan Lu
>
> **摘要:** With the rapid development of space exploration, space debris has attracted more attention due to its potential extreme threat, leading to the need for real-time and accurate debris tracking. However, existing methods are mainly based on traditional signal processing, which cannot effectively process the complex background and dense space debris. In this paper, we propose a deep learning-based Space Debris Tracking Network~(SDT-Net) to achieve highly accurate debris tracking. SDT-Net effectively represents the feature of debris, enhancing the efficiency and stability of end-to-end model learning. To train and evaluate this model effectively, we also produce a large-scale dataset Space Debris Tracking Dataset (SDTD) by a novel observation-based data simulation scheme. SDTD contains 18,040 video sequences with a total of 62,562 frames and covers 250,000 synthetic space debris. Extensive experiments validate the effectiveness of our model and the challenging of our dataset. Furthermore, we test our model on real data from the Antarctic Station, achieving a MOTA score of 70.6%, which demonstrates its strong transferability to real-world scenarios. Our dataset and code will be released soon.
>
---
#### [replaced 010] Graph Flow Matching: Enhancing Image Generation with Neighbor-Aware Flow Fields
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.24434v2](http://arxiv.org/pdf/2505.24434v2)**

> **作者:** Md Shahriar Rahim Siddiqui; Moshe Eliasof; Eldad Haber
>
> **摘要:** Flow matching casts sample generation as learning a continuous-time velocity field that transports noise to data. Existing flow matching networks typically predict each point's velocity independently, considering only its location and time along its flow trajectory, and ignoring neighboring points. However, this pointwise approach may overlook correlations between points along the generation trajectory that could enhance velocity predictions, thereby improving downstream generation quality. To address this, we propose Graph Flow Matching (GFM), a lightweight enhancement that decomposes the learned velocity into a reaction term -- any standard flow matching network -- and a diffusion term that aggregates neighbor information via a graph neural module. This reaction-diffusion formulation retains the scalability of deep flow models while enriching velocity predictions with local context, all at minimal additional computational cost. Operating in the latent space of a pretrained variational autoencoder, GFM consistently improves Fr\'echet Inception Distance (FID) and recall across five image generation benchmarks (LSUN Church, LSUN Bedroom, FFHQ, AFHQ-Cat, and CelebA-HQ at $256\times256$), demonstrating its effectiveness as a modular enhancement to existing flow matching architectures.
>
---
#### [replaced 011] Sonic: Shifting Focus to Global Audio Perception in Portrait Animation
- **分类: cs.MM; cs.CV; cs.GR; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.16331v2](http://arxiv.org/pdf/2411.16331v2)**

> **作者:** Xiaozhong Ji; Xiaobin Hu; Zhihong Xu; Junwei Zhu; Chuming Lin; Qingdong He; Jiangning Zhang; Donghao Luo; Yi Chen; Qin Lin; Qinglin Lu; Chengjie Wang
>
> **备注:** refer to our main-page \url{https://jixiaozhong.github.io/Sonic/}
>
> **摘要:** The study of talking face generation mainly explores the intricacies of synchronizing facial movements and crafting visually appealing, temporally-coherent animations. However, due to the limited exploration of global audio perception, current approaches predominantly employ auxiliary visual and spatial knowledge to stabilize the movements, which often results in the deterioration of the naturalness and temporal inconsistencies.Considering the essence of audio-driven animation, the audio signal serves as the ideal and unique priors to adjust facial expressions and lip movements, without resorting to interference of any visual signals. Based on this motivation, we propose a novel paradigm, dubbed as Sonic, to {s}hift f{o}cus on the exploration of global audio per{c}ept{i}o{n}.To effectively leverage global audio knowledge, we disentangle it into intra- and inter-clip audio perception and collaborate with both aspects to enhance overall perception.For the intra-clip audio perception, 1). \textbf{Context-enhanced audio learning}, in which long-range intra-clip temporal audio knowledge is extracted to provide facial expression and lip motion priors implicitly expressed as the tone and speed of speech. 2). \textbf{Motion-decoupled controller}, in which the motion of the head and expression movement are disentangled and independently controlled by intra-audio clips. Most importantly, for inter-clip audio perception, as a bridge to connect the intra-clips to achieve the global perception, \textbf{Time-aware position shift fusion}, in which the global inter-clip audio information is considered and fused for long-audio inference via through consecutively time-aware shifted windows. Extensive experiments demonstrate that the novel audio-driven paradigm outperform existing SOTA methodologies in terms of video quality, temporally consistency, lip synchronization precision, and motion diversity.
>
---
#### [replaced 012] ABCDEFGH: An Adaptation-Based Convolutional Neural Network-CycleGAN Disease-Courses Evolution Framework Using Generative Models in Health Education
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.00605v2](http://arxiv.org/pdf/2506.00605v2)**

> **作者:** Ruiming Min; Minghao Liu
>
> **备注:** All authors did not agree to submitting this work. This version of the report contains misinformation and is not ready to share
>
> **摘要:** With the advancement of modern medicine and the development of technologies such as MRI, CT, and cellular analysis, it has become increasingly critical for clinicians to accurately interpret various diagnostic images. However, modern medical education often faces challenges due to limited access to high-quality teaching materials, stemming from privacy concerns and a shortage of educational resources (Balogh et al., 2015). In this context, image data generated by machine learning models, particularly generative models, presents a promising solution. These models can create diverse and comparable imaging datasets without compromising patient privacy, thereby supporting modern medical education. In this study, we explore the use of convolutional neural networks (CNNs) and CycleGAN (Zhu et al., 2017) for generating synthetic medical images. The source code is available at https://github.com/mliuby/COMP4211-Project.
>
---
#### [replaced 013] UniWorld: High-Resolution Semantic Encoders for Unified Visual Understanding and Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.03147v2](http://arxiv.org/pdf/2506.03147v2)**

> **作者:** Bin Lin; Zongjian Li; Xinhua Cheng; Yuwei Niu; Yang Ye; Xianyi He; Shenghai Yuan; Wangbo Yu; Shaodong Wang; Yunyang Ge; Yatian Pang; Li Yuan
>
> **摘要:** Although existing unified models achieve strong performance in vision-language understanding and text-to-image generation, they remain limited in addressing image perception and manipulation -- capabilities increasingly demanded in practical applications. Recently, OpenAI introduced the powerful GPT-4o-Image model, which showcases advanced capabilities in comprehensive image perception and manipulation, sparking widespread interest. Through carefully designed experiments, we observe that GPT-4o-Image likely relies on semantic encoders rather than VAEs for feature extraction, despite VAEs being commonly regarded as crucial for image manipulation tasks. Inspired by this insight, we propose UniWorld, a unified generative framework built upon semantic features extracted from powerful multimodal large language models and contrastive semantic encoders. Using only 2.7M training data, UniWorld achieves impressive performance across diverse tasks, including image understanding, generation, manipulation, and perception. We fully open-source the UniWorld framework, including model weights, training and evaluation scripts, and datasets to promote reproducibility and further research.
>
---
#### [replaced 014] Rapid Bone Scintigraphy Enhancement via Semantic Prior Distillation from Segment Anything Model
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.02321v3](http://arxiv.org/pdf/2503.02321v3)**

> **作者:** Pengchen Liang; Leijun Shi; Huiping Yao; Bin Pu; Jianguo Chen; Lei Zhao; Haishan Huang; Zhuangzhuang Chen; Zhaozhao Xu; Lite Xu; Qing Chang; Yiwei Li
>
> **备注:** 12 pages, 9 figures, 8 tables
>
> **摘要:** Rapid bone scintigraphy is crucial for diagnosing skeletal disorders and detecting tumor metastases in children, as it shortens scan duration and reduces discomfort. However, accelerated acquisition often degrades image quality, impairing the visibility of fine anatomical details and potentially compromising diagnosis. To overcome this limitation, we introduce the first application of SAM-based semantic priors for medical image restoration, utilizing the Segment Anything Model (SAM) to enhance pediatric rapid bone scintigraphy. Our approach employs two cascaded networks, $f^{IR1}$ and $f^{IR2}$, supported by three specialized modules: a Semantic Prior Integration (SPI) module, a Semantic Knowledge Distillation (SKD) module, and a Semantic Consistency Module (SCM). The SPI and SKD modules inject domain-specific semantic cues from a fine-tuned SAM, while the SCM preserves coherent semantic feature representations across both cascaded stages. Moreover, we present RBS, a novel Rapid Bone Scintigraphy dataset comprising paired standard (20 cm/min) and rapid (40 cm/min) scans from 137 pediatric patients aged 0.5 - 16 years, making it the first dataset tailored for pediatric rapid bone scintigraphy restoration. Extensive experiments on both a public endoscopic dataset and our RBS dataset demonstrate that our method consistently surpasses existing techniques in PSNR, SSIM, FID, and LPIPS metrics.
>
---
#### [replaced 015] Generalized Diffusion Detector: Mining Robust Features from Diffusion Models for Domain-Generalized Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.02101v2](http://arxiv.org/pdf/2503.02101v2)**

> **作者:** Boyong He; Yuxiang Ji; Qianwen Ye; Zhuoyue Tan; Liaoni Wu
>
> **备注:** CVPR2025 camera-ready version with supplementary material
>
> **摘要:** Domain generalization (DG) for object detection aims to enhance detectors' performance in unseen scenarios. This task remains challenging due to complex variations in real-world applications. Recently, diffusion models have demonstrated remarkable capabilities in diverse scene generation, which inspires us to explore their potential for improving DG tasks. Instead of generating images, our method extracts multi-step intermediate features during the diffusion process to obtain domain-invariant features for generalized detection. Furthermore, we propose an efficient knowledge transfer framework that enables detectors to inherit the generalization capabilities of diffusion models through feature and object-level alignment, without increasing inference time. We conduct extensive experiments on six challenging DG benchmarks. The results demonstrate that our method achieves substantial improvements of 14.0% mAP over existing DG approaches across different domains and corruption types. Notably, our method even outperforms most domain adaptation methods without accessing any target domain data. Moreover, the diffusion-guided detectors show consistent improvements of 15.9% mAP on average compared to the baseline. Our work aims to present an effective approach for domain-generalized detection and provide potential insights for robust visual recognition in real-world scenarios. The code is available at https://github.com/heboyong/Generalized-Diffusion-Detector.
>
---
#### [replaced 016] ATI: Any Trajectory Instruction for Controllable Video Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.22944v2](http://arxiv.org/pdf/2505.22944v2)**

> **作者:** Angtian Wang; Haibin Huang; Jacob Zhiyuan Fang; Yiding Yang; Chongyang Ma
>
> **摘要:** We propose a unified framework for motion control in video generation that seamlessly integrates camera movement, object-level translation, and fine-grained local motion using trajectory-based inputs. In contrast to prior methods that address these motion types through separate modules or task-specific designs, our approach offers a cohesive solution by projecting user-defined trajectories into the latent space of pre-trained image-to-video generation models via a lightweight motion injector. Users can specify keypoints and their motion paths to control localized deformations, entire object motion, virtual camera dynamics, or combinations of these. The injected trajectory signals guide the generative process to produce temporally consistent and semantically aligned motion sequences. Our framework demonstrates superior performance across multiple video motion control tasks, including stylized motion effects (e.g., motion brushes), dynamic viewpoint changes, and precise local motion manipulation. Experiments show that our method provides significantly better controllability and visual quality compared to prior approaches and commercial solutions, while remaining broadly compatible with various state-of-the-art video generation backbones. Project page: https://anytraj.github.io/.
>
---
#### [replaced 017] Dirty and Clean-Label attack detection using GAN discriminators
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2506.01224v2](http://arxiv.org/pdf/2506.01224v2)**

> **作者:** John W. Smutny
>
> **备注:** 13 pages total. Appendix starts on page 10
>
> **摘要:** Gathering enough images to train a deep computer vision model is a constant challenge. Unfortunately, collecting images from unknown sources can leave your model s behavior at risk of being manipulated by a dirty-label or clean-label attack unless the images are properly inspected. Manually inspecting each image-label pair is impractical and common poison-detection methods that involve re-training your model can be time consuming. This research uses GAN discriminators to protect a single class against mislabeled and different levels of modified images. The effect of said perturbation on a basic convolutional neural network classifier is also included for reference. The results suggest that after training on a single class, GAN discriminator s confidence scores can provide a threshold to identify mislabeled images and identify 100% of the tested poison starting at a perturbation epsilon magnitude of 0.20, after decision threshold calibration using in-class samples. Developers can use this report as a basis to train their own discriminators to protect high valued classes in their CV models.
>
---
#### [replaced 018] Exploring Sparsity for Parameter Efficient Fine Tuning Using Wavelets
- **分类: cs.CV; cs.AI; cs.LG; eess.IV; eess.SP**

- **链接: [http://arxiv.org/pdf/2505.12532v2](http://arxiv.org/pdf/2505.12532v2)**

> **作者:** Ahmet Bilican; M. Akın Yılmaz; A. Murat Tekalp; R. Gökberk Cinbiş
>
> **摘要:** Efficiently adapting large foundation models is critical, especially with tight compute and memory budgets. Parameter-Efficient Fine-Tuning (PEFT) methods such as LoRA offer limited granularity and effectiveness in few-parameter regimes. We propose Wavelet Fine-Tuning (WaveFT), a novel PEFT method that learns highly sparse updates in the wavelet domain of residual matrices. WaveFT allows precise control of trainable parameters, offering fine-grained capacity adjustment and excelling with remarkably low parameter count, potentially far fewer than LoRA's minimum, ideal for extreme parameter-efficient scenarios. Evaluated on personalized text-to-image generation using Stable Diffusion XL as baseline, WaveFT significantly outperforms LoRA and other PEFT methods, especially at low parameter counts; achieving superior subject fidelity, prompt alignment, and image diversity.
>
---
#### [replaced 019] Fact-R1: Towards Explainable Video Misinformation Detection with Deep Reasoning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.16836v2](http://arxiv.org/pdf/2505.16836v2)**

> **作者:** Fanrui Zhang; Dian Li; Qiang Zhang; Chenjun; sinbadliu; Junxiong Lin; Jiahong Yan; Jiawei Liu; Zheng-Jun Zha
>
> **备注:** 28 pages, 27 figures
>
> **摘要:** The rapid spread of multimodal misinformation on social media has raised growing concerns, while research on video misinformation detection remains limited due to the lack of large-scale, diverse datasets. Existing methods often overfit to rigid templates and lack deep reasoning over deceptive content. To address these challenges, we introduce FakeVV, a large-scale benchmark comprising over 100,000 video-text pairs with fine-grained, interpretable annotations. In addition, we further propose Fact-R1, a novel framework that integrates deep reasoning with collaborative rule-based reinforcement learning. Fact-R1 is trained through a three-stage process: (1) misinformation long-Chain-of-Thought (CoT) instruction tuning, (2) preference alignment via Direct Preference Optimization (DPO), and (3) Group Relative Policy Optimization (GRPO) using a novel verifiable reward function. This enables Fact-R1 to exhibit emergent reasoning behaviors comparable to those observed in advanced text-based reinforcement learning systems, but in the more complex multimodal misinformation setting. Our work establishes a new paradigm for misinformation detection, bridging large-scale video understanding, reasoning-guided alignment, and interpretable verification.
>
---
#### [replaced 020] CellFlux: Simulating Cellular Morphology Changes via Flow Matching
- **分类: q-bio.QM; cs.CV; cs.LG; q-bio.BM; q-bio.CB**

- **链接: [http://arxiv.org/pdf/2502.09775v3](http://arxiv.org/pdf/2502.09775v3)**

> **作者:** Yuhui Zhang; Yuchang Su; Chenyu Wang; Tianhong Li; Zoe Wefers; Jeffrey Nirschl; James Burgess; Daisy Ding; Alejandro Lozano; Emma Lundberg; Serena Yeung-Levy
>
> **备注:** Published at ICML 2025
>
> **摘要:** Building a virtual cell capable of accurately simulating cellular behaviors in silico has long been a dream in computational biology. We introduce CellFlux, an image-generative model that simulates cellular morphology changes induced by chemical and genetic perturbations using flow matching. Unlike prior methods, CellFlux models distribution-wise transformations from unperturbed to perturbed cell states, effectively distinguishing actual perturbation effects from experimental artifacts such as batch effects -- a major challenge in biological data. Evaluated on chemical (BBBC021), genetic (RxRx1), and combined perturbation (JUMP) datasets, CellFlux generates biologically meaningful cell images that faithfully capture perturbation-specific morphological changes, achieving a 35% improvement in FID scores and a 12% increase in mode-of-action prediction accuracy over existing methods. Additionally, CellFlux enables continuous interpolation between cellular states, providing a potential tool for studying perturbation dynamics. These capabilities mark a significant step toward realizing virtual cell modeling for biomedical research. Project page: https://yuhui-zh15.github.io/CellFlux/.
>
---
#### [replaced 021] SViMo: Synchronized Diffusion for Video and Motion Generation in Hand-object Interaction Scenarios
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.02444v2](http://arxiv.org/pdf/2506.02444v2)**

> **作者:** Lingwei Dang; Ruizhi Shao; Hongwen Zhang; Wei Min; Yebin Liu; Qingyao Wu
>
> **摘要:** Hand-Object Interaction (HOI) generation has significant application potential. However, current 3D HOI motion generation approaches heavily rely on predefined 3D object models and lab-captured motion data, limiting generalization capabilities. Meanwhile, HOI video generation methods prioritize pixel-level visual fidelity, often sacrificing physical plausibility. Recognizing that visual appearance and motion patterns share fundamental physical laws in the real world, we propose a novel framework that combines visual priors and dynamic constraints within a synchronized diffusion process to generate the HOI video and motion simultaneously. To integrate the heterogeneous semantics, appearance, and motion features, our method implements tri-modal adaptive modulation for feature aligning, coupled with 3D full-attention for modeling inter- and intra-modal dependencies. Furthermore, we introduce a vision-aware 3D interaction diffusion model that generates explicit 3D interaction sequences directly from the synchronized diffusion outputs, then feeds them back to establish a closed-loop feedback cycle. This architecture eliminates dependencies on predefined object models or explicit pose guidance while significantly enhancing video-motion consistency. Experimental results demonstrate our method's superiority over state-of-the-art approaches in generating high-fidelity, dynamically plausible HOI sequences, with notable generalization capabilities in unseen real-world scenarios. Project page at https://github.com/Droliven/SViMo\_project.
>
---
#### [replaced 022] SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook for Multimodal Understanding and Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.06764v4](http://arxiv.org/pdf/2503.06764v4)**

> **作者:** Zisheng Chen; Chunwei Wang; Xiuwei Chen; Hongbin Xu; Runhui Huang; Jun Zhou; Jianhua Han; Hang Xu; Xiaodan Liang
>
> **备注:** Under Review, Refer to the latest version
>
> **摘要:** In this paper, we introduce SemHiTok, a unified image Tokenizer via Semantic-Guided Hierarchical codebook that provides consistent discrete representations for multimodal understanding and generation. Recently, unified image tokenizers have sparked exploration within research community, which is designed to capture high-level semantic features for understanding and retaining low-level pixel features for generation. Previous works attempt to train a unified image tokenizer by combining loss for semantic distillation and pixel reconstruction. However, due to the differing levels of features prioritized by multimodal understanding and generation, joint training methods face significant challenges in achieving a good trade-off. SemHiTok addresses this challenge through a novel semantic-guided hierarchical codebook, which builds pixel sub-codebooks on a pretrained semantic codebook. This design decouples semantic and pixel both in terms of structure and training strategy, enabling the tokenizer to capture pixel features while retaining its ability to comprehend high-level semantic information. Our experiments demonstrate that SemHiTok achieves SOTA performance in image reconstruction and multimodal understanding under LLaVA-v1.5 setting. Further, we develop a unified MLLM with SemHiTok, which exhibits superior performance across multimodal understanding and generation tasks. For understanding, SemHiTok achieves impressive performance on most benchmarks. For generation, our model achieves SOTA performance on MJHQ30K in unified MLLMs.
>
---
#### [replaced 023] Detecting Dataset Bias in Medical AI: A Generalized and Modality-Agnostic Auditing Framework
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.09969v2](http://arxiv.org/pdf/2503.09969v2)**

> **作者:** Nathan Drenkow; Mitchell Pavlak; Keith Harrigian; Ayah Zirikly; Adarsh Subbaswamy; Mohammad Mehdi Farhangi; Nicholas Petrick; Mathias Unberath
>
> **摘要:** Artificial Intelligence (AI) is now firmly at the center of evidence-based medicine. Despite many success stories that edge the path of AI's rise in healthcare, there are comparably many reports of significant shortcomings and unexpected behavior of AI in deployment. A major reason for these limitations is AI's reliance on association-based learning, where non-representative machine learning datasets can amplify latent bias during training and/or hide it during testing. To unlock new tools capable of foreseeing and preventing such AI bias issues, we present G-AUDIT. Generalized Attribute Utility and Detectability-Induced bias Testing (G-AUDIT) for datasets is a modality-agnostic dataset auditing framework that allows for generating targeted hypotheses about sources of bias in training or testing data. Our method examines the relationship between task-level annotations (commonly referred to as ``labels'') and data properties including patient attributes (e.g., age, sex) and environment/acquisition characteristics (e.g., clinical site, imaging protocols). G-AUDIT quantifies the extent to which the observed data attributes pose a risk for shortcut learning, or in the case of testing data, might hide predictions made based on spurious associations. We demonstrate the broad applicability of our method by analyzing large-scale medical datasets for three distinct modalities and machine learning tasks: skin lesion classification in images, stigmatizing language classification in Electronic Health Records (EHR), and mortality prediction for ICU tabular data. In each setting, G-AUDIT successfully identifies subtle biases commonly overlooked by traditional qualitative methods, underscoring its practical value in exposing dataset-level risks and supporting the downstream development of reliable AI systems.
>
---
#### [replaced 024] Two-stage deep learning framework for the restoration of incomplete-ring PET images
- **分类: cs.CV; physics.med-ph**

- **链接: [http://arxiv.org/pdf/2504.00816v3](http://arxiv.org/pdf/2504.00816v3)**

> **作者:** Yeqi Fang; Rong Zhou
>
> **备注:** 20 pages, 7 figures
>
> **摘要:** Positron Emission Tomography (PET) is an important molecular imaging tool widely used in medicine. Traditional PET systems rely on complete detector rings for full angular coverage and reliable data collection. However, incomplete-ring PET scanners have emerged due to hardware failures, cost constraints, or specific clinical needs. Standard reconstruction algorithms often suffer from performance degradation with these systems because of reduced data completeness and geometric inconsistencies. We present a two-stage deep-learning framework that, without incorporating any time-of-flight (TOF) information, restores high-quality images from data with about 50% missing coincidences - double the loss levels previously addressed by CNN-based methods. The pipeline operates in two stages: a projection-domain Attention U-Net first predicts the missing sections of the sinogram by leveraging spatial context from neighbouring slices, after which the completed data are reconstructed with OSEM algorithm and passed to a U-Net-diffusion module that removes residual artefacts while reinstating high-frequency detail. Using 206 brain volumes from a public dataset, the result shows that our model successfully preserves most anatomical structures and tracer distribution features with PSNR of 30.92 dB and SSIM of 0.9708. We also achieve higher inference speed, thus providing an effective solution for incomplete-ring PET imaging.
>
---
#### [replaced 025] Right Side Up? Disentangling Orientation Understanding in MLLMs with Fine-grained Multi-axis Perception Tasks
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21649v4](http://arxiv.org/pdf/2505.21649v4)**

> **作者:** Keanu Nichols; Nazia Tasnim; Yuting Yan; Nicholas Ikechukwu; Elva Zou; Deepti Ghadiyaram; Bryan A. Plummer
>
> **摘要:** Object orientation understanding represents a fundamental challenge in visual perception critical for applications like robotic manipulation and augmented reality. Current vision-language benchmarks fail to isolate this capability, often conflating it with positional relationships and general scene understanding. We introduce DORI (Discriminative Orientation Reasoning Intelligence), a comprehensive benchmark establishing object orientation perception as a primary evaluation target. DORI assesses four dimensions of orientation comprehension: frontal alignment, rotational transformations, relative directional relationships, and canonical orientation understanding. Through carefully curated tasks from 11 datasets spanning 67 object categories across synthetic and real-world scenarios, DORI provides insights on how multi-modal systems understand object orientations. Our evaluation of 15 state-of-the-art vision-language models reveals critical limitations: even the best models achieve only 54.2% accuracy on coarse tasks and 33.0% on granular orientation judgments, with performance deteriorating for tasks requiring reference frame shifts or compound rotations. These findings demonstrate the need for dedicated orientation representation mechanisms, as models show systematic inability to perform precise angular estimations, track orientation changes across viewpoints, and understand compound rotations - suggesting limitations in their internal 3D spatial representations. As the first diagnostic framework specifically designed for orientation awareness in multimodal systems, DORI offers implications for improving robotic control, 3D scene reconstruction, and human-AI interaction in physical environments. DORI data: https://huggingface.co/datasets/appledora/DORI-Benchmark
>
---
#### [replaced 026] UltraBones100k: A reliable automated labeling method and large-scale dataset for ultrasound-based bone surface extraction
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.03783v4](http://arxiv.org/pdf/2502.03783v4)**

> **作者:** Luohong Wu; Nicola A. Cavalcanti; Matthias Seibold; Giuseppe Loggia; Lisa Reissner; Jonas Hein; Silvan Beeler; Arnd Viehöfer; Stephan Wirth; Lilian Calvet; Philipp Fürnstahl
>
> **备注:** accepted by Computers in Biology and Medicine
>
> **摘要:** Ultrasound-based bone surface segmentation is crucial in computer-assisted orthopedic surgery. However, ultrasound images have limitations, including a low signal-to-noise ratio, and acoustic shadowing, which make interpretation difficult. Existing deep learning models for bone segmentation rely primarily on costly manual labeling by experts, limiting dataset size and model generalizability. Additionally, the complexity of ultrasound physics and acoustic shadow makes the images difficult for humans to interpret, leading to incomplete labels in anechoic regions and limiting model performance. To advance ultrasound bone segmentation and establish effective model benchmarks, larger and higher-quality datasets are needed. We propose a methodology for collecting ex-vivo ultrasound datasets with automatically generated bone labels, including anechoic regions. The proposed labels are derived by accurately superimposing tracked bone CT models onto the tracked ultrasound images. These initial labels are refined to account for ultrasound physics. A clinical evaluation is conducted by an expert physician specialized on orthopedic sonography to assess the quality of the generated bone labels. A neural network for bone segmentation is trained on the collected dataset and its predictions are compared to expert manual labels, evaluating accuracy, completeness, and F1-score. We collected the largest known dataset of 100k ultrasound images of human lower limbs with bone labels, called UltraBones100k. A Wilcoxon signed-rank test with Bonferroni correction confirmed that the bone alignment after our method significantly improved the quality of bone labeling (p < 0.001). The model trained on UltraBones100k consistently outperforms manual labeling in all metrics, particularly in low-intensity regions (320% improvement in completeness at a distance threshold of 0.5 mm).
>
---
#### [replaced 027] Generative Emotion Cause Explanation in Multimodal Conversations
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.02430v3](http://arxiv.org/pdf/2411.02430v3)**

> **作者:** Lin Wang; Xiaocui Yang; Shi Feng; Daling Wang; Yifei Zhang; Zhitao Zhang
>
> **摘要:** Multimodal conversation, a crucial form of human communication, carries rich emotional content, making the exploration of the causes of emotions within it a research endeavor of significant importance. However, existing research on the causes of emotions typically employs an utterance selection method within a single textual modality to locate causal utterances. This approach remains limited to coarse-grained assessments, lacks nuanced explanations of emotional causation, and demonstrates inadequate capability in identifying multimodal emotional triggers. Therefore, we introduce a task-\textbf{Multimodal Emotion Cause Explanation in Conversation (MECEC)}. This task aims to generate a summary based on the multimodal context of conversations, clearly and intuitively describing the reasons that trigger a given emotion. To adapt to this task, we develop a new dataset (ECEM) based on the MELD dataset. ECEM combines video clips with detailed explanations of character emotions, helping to explore the causal factors behind emotional expression in multimodal conversations. A novel approach, FAME-Net, is further proposed, that harnesses the power of Large Language Models (LLMs) to analyze visual data and accurately interpret the emotions conveyed through facial expressions in videos. By exploiting the contagion effect of facial emotions, FAME-Net effectively captures the emotional causes of individuals engaged in conversations. Our experimental results on the newly constructed dataset show that FAME-Net outperforms several excellent baselines. Code and dataset are available at https://github.com/3222345200/FAME-Net.
>
---
#### [replaced 028] ExeChecker: Where Did I Go Wrong?
- **分类: cs.CV; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.10573v2](http://arxiv.org/pdf/2412.10573v2)**

> **作者:** Yiwen Gu; Mahir Patel; Margrit Betke
>
> **摘要:** In this paper, we present a contrastive learning based framework, ExeChecker, for the interpretation of rehabilitation exercises. Our work builds upon state-of-the-art advances in the area of human pose estimation, graph-attention neural networks, and transformer interpretablity. The downstream task is to assist rehabilitation by providing informative feedback to users while they are performing prescribed exercises. We utilize a contrastive learning strategy during training. Given a tuple of correctly and incorrectly executed exercises, our model is able to identify and highlight those joints that are involved in an incorrect movement and thus require the user's attention. We collected an in-house dataset, ExeCheck, with paired recordings of both correct and incorrect execution of exercises. In our experiments, we tested our method on this dataset as well as the UI-PRMD dataset and found ExeCheck outperformed the baseline method using pairwise sequence alignment in identifying joints of physical relevance in rehabilitation exercises.
>
---
#### [replaced 029] Diffusion-VLA: Generalizable and Interpretable Robot Foundation Model via Self-Generated Reasoning
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.03293v3](http://arxiv.org/pdf/2412.03293v3)**

> **作者:** Junjie Wen; Minjie Zhu; Yichen Zhu; Zhibin Tang; Jinming Li; Zhongyi Zhou; Chengmeng Li; Xiaoyu Liu; Yaxin Peng; Chaomin Shen; Feifei Feng
>
> **备注:** Accepted by ICML 2025. The project page is available at: http://diffusion-vla.github.io
>
> **摘要:** In this paper, we present DiffusionVLA, a novel framework that seamlessly combines the autoregression model with the diffusion model for learning visuomotor policy. Central to our approach is a next-token prediction objective, enabling the model to reason effectively over the user's query in the context of current observations. Subsequently, a diffusion model is attached to generate robust action outputs. To enhance policy learning through self-reasoning, we introduce a novel reasoning injection module that integrates reasoning phrases directly into the policy learning process. The whole framework is simple and flexible, making it easy to deploy and upgrade. We conduct extensive experiments using multiple real robots to validate the effectiveness of DiffusionVLA. Our tests include a challenging factory sorting task, where DiffusionVLA successfully categorizes objects, including those not seen during training. We observe that the reasoning module makes the model interpretable. It allows observers to understand the model thought process and identify potential causes of policy failures. Additionally, we test DiffusionVLA on a zero-shot bin-picking task, achieving 63.7\% accuracy on 102 previously unseen objects. Our method demonstrates robustness to visual changes, such as distractors and new backgrounds, and easily adapts to new embodiments. Furthermore, DiffusionVLA can follow novel instructions and retain conversational ability. Notably, DiffusionVLA is data-efficient and fast at inference; our smallest DiffusionVLA-2B runs 82Hz on a single A6000 GPU and can train from scratch on less than 50 demonstrations for a complex task. Finally, we scale the model from 2B to 72B parameters, showcasing improved generalization capabilities with increased model size.
>
---
#### [replaced 030] InterRVOS: Interaction-aware Referring Video Object Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.02356v2](http://arxiv.org/pdf/2506.02356v2)**

> **作者:** Woojeong Jin; Seongchan Kim; Seungryong Kim
>
> **摘要:** Referring video object segmentation aims to segment the object in a video corresponding to a given natural language expression. While prior works have explored various referring scenarios, including motion-centric or multi-instance expressions, most approaches still focus on localizing a single target object in isolation. However, in comprehensive video understanding, an object's role is often defined by its interactions with other entities, which are largely overlooked in existing datasets and models. In this work, we introduce Interaction-aware referring video object sgementation (InterRVOS), a new task that requires segmenting both actor and target entities involved in an interaction. Each interactoin is described through a pair of complementary expressions from different semantic perspectives, enabling fine-grained modeling of inter-object relationships. To tackle this task, we propose InterRVOS-8K, the large-scale and automatically constructed dataset containing diverse interaction-aware expressions with corresponding masks, including challenging cases such as motion-only multi-instance expressions. We also present a baseline architecture, ReVIOSa, designed to handle actor-target segmentation from a single expression, achieving strong performance in both standard and interaction-focused settings. Furthermore, we introduce an actor-target-aware evalaution setting that enables a more targeted assessment of interaction understanding. Experimental results demonstrate that our approach outperforms prior methods in modeling complex object interactions for referring video object segmentation task, establishing a strong foundation for future research in interaction-centric video understanding. Our project page is available at https://cvlab-kaist.github.io/InterRVOS.
>
---
#### [replaced 031] EasyInv: Toward Fast and Better DDIM Inversion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.05159v4](http://arxiv.org/pdf/2408.05159v4)**

> **作者:** Ziyue Zhang; Mingbao Lin; Shuicheng Yan; Rongrong Ji
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** This paper introduces EasyInv, an easy yet novel approach that significantly advances the field of DDIM Inversion by addressing the inherent inefficiencies and performance limitations of traditional iterative optimization methods. At the core of our EasyInv is a refined strategy for approximating inversion noise, which is pivotal for enhancing the accuracy and reliability of the inversion process. By prioritizing the initial latent state, which encapsulates rich information about the original images, EasyInv steers clear of the iterative refinement of noise items. Instead, we introduce a methodical aggregation of the latent state from the preceding time step with the current state, effectively increasing the influence of the initial latent state and mitigating the impact of noise. We illustrate that EasyInv is capable of delivering results that are either on par with or exceed those of the conventional DDIM Inversion approach, especially under conditions where the model's precision is limited or computational resources are scarce. Concurrently, our EasyInv offers an approximate threefold enhancement regarding inference efficiency over off-the-shelf iterative optimization techniques. It can be easily combined with most existing inversion methods by only four lines of code. See code at https://github.com/potato-kitty/EasyInv.
>
---
#### [replaced 032] MMAR: Towards Lossless Multi-Modal Auto-Regressive Probabilistic Modeling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.10798v3](http://arxiv.org/pdf/2410.10798v3)**

> **作者:** Jian Yang; Dacheng Yin; Yizhou Zhou; Fengyun Rao; Wei Zhai; Yang Cao; Zheng-Jun Zha
>
> **摘要:** Recent advancements in multi-modal large language models have propelled the development of joint probabilistic models capable of both image understanding and generation. However, we have identified that recent methods suffer from loss of image information during understanding task, due to either image discretization or diffusion denoising steps. To address this issue, we propose a novel Multi-Modal Auto-Regressive (MMAR) probabilistic modeling framework. Unlike discretization line of method, MMAR takes in continuous-valued image tokens to avoid information loss in an efficient way. Differing from diffusion-based approaches, we disentangle the diffusion process from auto-regressive backbone model by employing a light-weight diffusion head on top each auto-regressed image patch embedding. In this way, when the model transits from image generation to understanding through text generation, the backbone model's hidden representation of the image is not limited to the last denoising step. To successfully train our method, we also propose a theoretically proven technique that addresses the numerical stability issue and a training strategy that balances the generation and understanding task goals. Extensive evaluations on 18 image understanding benchmarks show that MMAR significantly outperforms most of the existing joint multi-modal models, surpassing the method that employs pre-trained CLIP vision encoder. Meanwhile, MMAR is able to generate high quality images. We also show that our method is scalable with larger data and model size.
>
---
#### [replaced 033] Diffusing DeBias: Synthetic Bias Amplification for Model Debiasing
- **分类: cs.LG; cs.CV; I.4; I.5**

- **链接: [http://arxiv.org/pdf/2502.09564v4](http://arxiv.org/pdf/2502.09564v4)**

> **作者:** Massimiliano Ciranni; Vito Paolo Pastore; Roberto Di Via; Enzo Tartaglione; Francesca Odone; Vittorio Murino
>
> **备注:** 18 Pages, 9 Figures
>
> **摘要:** Deep learning model effectiveness in classification tasks is often challenged by the quality and quantity of training data whenever they are affected by strong spurious correlations between specific attributes and target labels. This results in a form of bias affecting training data, which typically leads to unrecoverable weak generalization in prediction. This paper aims at facing this problem by leveraging bias amplification with generated synthetic data: we introduce Diffusing DeBias (DDB), a novel approach acting as a plug-in for common methods of unsupervised model debiasing exploiting the inherent bias-learning tendency of diffusion models in data generation. Specifically, our approach adopts conditional diffusion models to generate synthetic bias-aligned images, which replace the original training set for learning an effective bias amplifier model that we subsequently incorporate into an end-to-end and a two-step unsupervised debiasing approach. By tackling the fundamental issue of bias-conflicting training samples memorization in learning auxiliary models, typical of this type of techniques, our proposed method beats current state-of-the-art in multiple benchmark datasets, demonstrating its potential as a versatile and effective tool for tackling bias in deep learning models.
>
---
#### [replaced 034] KAN-HyperpointNet for Point Cloud Sequence-Based 3D Human Action Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.09444v2](http://arxiv.org/pdf/2409.09444v2)**

> **作者:** Zhaoyu Chen; Xing Li; Qian Huang; Qiang Geng; Tianjin Yang; Shihao Han
>
> **摘要:** Point cloud sequence-based 3D action recognition has achieved impressive performance and efficiency. However, existing point cloud sequence modeling methods cannot adequately balance the precision of limb micro-movements with the integrity of posture macro-structure, leading to the loss of crucial information cues in action inference. To overcome this limitation, we introduce D-Hyperpoint, a novel data type generated through a D-Hyperpoint Embedding module. D-Hyperpoint encapsulates both regional-momentary motion and global-static posture, effectively summarizing the unit human action at each moment. In addition, we present a D-Hyperpoint KANsMixer module, which is recursively applied to nested groupings of D-Hyperpoints to learn the action discrimination information and creatively integrates Kolmogorov-Arnold Networks (KAN) to enhance spatio-temporal interaction within D-Hyperpoints. Finally, we propose KAN-HyperpointNet, a spatio-temporal decoupled network architecture for 3D action recognition. Extensive experiments on two public datasets: MSR Action3D and NTU-RGB+D 60, demonstrate the state-of-the-art performance of our method.
>
---
#### [replaced 035] FaceSleuth: Learning-Driven Single-Orientation Attention Verifies Vertical Dominance in Micro-Expression Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.02695v2](http://arxiv.org/pdf/2506.02695v2)**

> **作者:** Linquan Wu; Tianxiang Jiang; Wenhao Duan; Yini Fang; Jacky Keung
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** Micro-expression recognition (MER) demands models that can amplify millisecond-level, low-amplitude facial motions while suppressing identity-specific appearance. We introduce FaceSleuth, a dual-stream architecture that (1) enhances motion along the empirically dominant vertical axix through a Continuously Vertical Attention (CVA) block, (2) localises the resulting signals with a Facial Position Focalizer built on hierarchical cross-window attention, and (3) steers feature learning toward physiologically meaningful regions via lightweight Action-Unit embeddings. To examine whether the hand-chosen vertical axis is indeed optimal, we further propose a Single-Orientation Attention (SOA) module that learns its own pooling direction end-to-end. SOA is differentiable, adds only 0.16 % parameters, and collapses to CVA when the learned angle converges to {\Pi}/2. In practice, SOA reliably drifts to 88{\deg}, confirming the effectiveness of the vertical prior while delivering consistent gains. On three standard MER benchmarks, FaceSleuth with CVA already surpasses previous state-of-the-art methods; plugging in SOA lifts accuracy and F1 score performance to 95.1 % / 0.918 on CASME II, 87.1 % / 0.840 on SAMM, and 92.9 % / 0.917 on MMEW without sacrificing model compactness. These results establish a new state of the art and, for the first time, provide empirical evidence that the vertical attention bias is the most discriminative orientation for MER.
>
---
#### [replaced 036] Data-Juicer Sandbox: A Feedback-Driven Suite for Multimodal Data-Model Co-development
- **分类: cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.11784v3](http://arxiv.org/pdf/2407.11784v3)**

> **作者:** Daoyuan Chen; Haibin Wang; Yilun Huang; Ce Ge; Yaliang Li; Bolin Ding; Jingren Zhou
>
> **备注:** Accepted by ICML 2025 (Spotlight). 33 pages, 16 tables, 14 figures
>
> **摘要:** The emergence of multimodal large models has advanced artificial intelligence, introducing unprecedented levels of performance and functionality. However, optimizing these models remains challenging due to historically isolated paths of model-centric and data-centric developments, leading to suboptimal outcomes and inefficient resource utilization. In response, we present a new sandbox suite tailored for integrated data-model co-development. This sandbox provides a feedback-driven experimental platform, enabling cost-effective iteration and guided refinement of both data and models. Our proposed ``Probe-Analyze-Refine'' workflow, validated through practical use cases on multimodal tasks such as image-text pre-training with CLIP, image-to-text generation with LLaVA-like models, and text-to-video generation with DiT-based models, yields transferable and notable performance boosts, such as topping the VBench leaderboard. A comprehensive set of over 100 experiments demonstrated the suite's usability and extensibility, while also uncovering insights into the interplay between data quality, diversity, model behavior, and computational costs. All codes, datasets, and models are open-sourced to foster future research and applications that would otherwise be infeasible due to the lack of a dedicated co-development infrastructure.
>
---
#### [replaced 037] Go Beyond Earth: Understanding Human Actions and Scenes in Microgravity Environments
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.02845v2](http://arxiv.org/pdf/2506.02845v2)**

> **作者:** Di Wen; Lei Qi; Kunyu Peng; Kailun Yang; Fei Teng; Ao Luo; Jia Fu; Yufan Chen; Ruiping Liu; Yitian Shi; M. Saquib Sarfraz; Rainer Stiefelhagen
>
> **备注:** 15 pages, 3 figures, code are available at https://github.com/LEI-QI-233/HAR-in-Space
>
> **摘要:** Despite substantial progress in video understanding, most existing datasets are limited to Earth's gravitational conditions. However, microgravity alters human motion, interactions, and visual semantics, revealing a critical gap for real-world vision systems. This presents a challenge for domain-robust video understanding in safety-critical space applications. To address this, we introduce MicroG-4M, the first benchmark for spatio-temporal and semantic understanding of human activities in microgravity. Constructed from real-world space missions and cinematic simulations, the dataset includes 4,759 clips covering 50 actions, 1,238 context-rich captions, and over 7,000 question-answer pairs on astronaut activities and scene understanding. MicroG-4M supports three core tasks: fine-grained multi-label action recognition, temporal video captioning, and visual question answering, enabling a comprehensive evaluation of both spatial localization and semantic reasoning in microgravity contexts. We establish baselines using state-of-the-art models. All data, annotations, and code are available at https://github.com/LEI-QI-233/HAR-in-Space.
>
---
#### [replaced 038] DAS3D: Dual-modality Anomaly Synthesis for 3D Anomaly Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.09821v2](http://arxiv.org/pdf/2410.09821v2)**

> **作者:** Kecen Li; Bingquan Dai; Jingjing Fu; Xinwen Hou
>
> **备注:** Code available at https://github.com/SunnierLee/DAS3D
>
> **摘要:** Synthesizing anomaly samples has proven to be an effective strategy for self-supervised 2D industrial anomaly detection. However, this approach has been rarely explored in multi-modality anomaly detection, particularly involving 3D and RGB images. In this paper, we propose a novel dual-modality augmentation method for 3D anomaly synthesis, which is simple and capable of mimicking the characteristics of 3D defects. Incorporating with our anomaly synthesis method, we introduce a reconstruction-based discriminative anomaly detection network, in which a dual-modal discriminator is employed to fuse the original and reconstructed embedding of two modalities for anomaly detection. Additionally, we design an augmentation dropout mechanism to enhance the generalizability of the discriminator. Extensive experiments show that our method outperforms the state-of-the-art methods on detection precision and achieves competitive segmentation performance on both MVTec 3D-AD and Eyescandies datasets.
>
---
#### [replaced 039] MAC-Gaze: Motion-Aware Continual Calibration for Mobile Gaze Tracking
- **分类: cs.HC; cs.CV; 68T10, 68U35; H.5.2; H.1.2; C.2.4; I.5.4**

- **链接: [http://arxiv.org/pdf/2505.22769v2](http://arxiv.org/pdf/2505.22769v2)**

> **作者:** Yaxiong Lei; Mingyue Zhao; Yuheng Wang; Shijing He; Yusuke Sugano; Mohamed Khamis; Juan Ye
>
> **备注:** 24 pages, 7 figures
>
> **摘要:** Mobile gaze tracking faces a fundamental challenge: maintaining accuracy as users naturally change their postures and device orientations. Traditional calibration approaches, like one-off, fail to adapt to these dynamic conditions, leading to degraded performance over time. We present MAC-Gaze, a Motion-Aware continual Calibration approach that leverages smartphone Inertial measurement unit (IMU) sensors and continual learning techniques to automatically detect changes in user motion states and update the gaze tracking model accordingly. Our system integrates a pre-trained visual gaze estimator and an IMU-based activity recognition model with a clustering-based hybrid decision-making mechanism that triggers recalibration when motion patterns deviate significantly from previously encountered states. To enable accumulative learning of new motion conditions while mitigating catastrophic forgetting, we employ replay-based continual learning, allowing the model to maintain performance across previously encountered motion conditions. We evaluate our system through extensive experiments on the publicly available RGBDGaze dataset and our own 10-hour multimodal MotionGaze dataset (481K+ images, 800K+ IMU readings), encompassing a wide range of postures under various motion conditions including sitting, standing, lying, and walking. Results demonstrate that our method reduces gaze estimation error by 19.9% on RGBDGaze (from 1.73 cm to 1.41 cm) and by 31.7% on MotionGaze (from 2.81 cm to 1.92 cm) compared to traditional calibration approaches. Our framework provides a robust solution for maintaining gaze estimation accuracy in mobile scenarios.
>
---
#### [replaced 040] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2406.09295v3](http://arxiv.org/pdf/2406.09295v3)**

> **作者:** Yuhang Wu; Wenmeng Yu; Yean Cheng; Yan Wang; Xiaohan Zhang; Jiazheng Xu; Ming Ding; Yuxiao Dong
>
> **摘要:** Evaluating the alignment capabilities of large Vision-Language Models (VLMs) is essential for determining their effectiveness as helpful assistants. However, existing benchmarks primarily focus on basic abilities using nonverbal methods, such as yes-no and multiple-choice questions. In this paper, we address this gap by introducing AlignMMBench, which provides more nuanced evaluations of alignment capabilities and is the first benchmark specifically designed for Chinese visual contexts. This benchmark is meticulously curated from real-world scenarios and internet sources, encompassing thirteen specific tasks across three categories, and includes both single-turn and multi-turn dialogue scenarios. Incorporating a prompt rewrite strategy, AlignMMBench encompasses 1,054 images and 4,978 question-answer pairs. To facilitate the evaluation pipeline, we develop CritiqueVLM, a rule-calibrated evaluator that exceeds GPT-4's evaluation ability. Additionally, we measure the "alignment score", a quantitative metric designed to assess the robustness and stability of models across diverse prompts. Finally, we evaluate the performance of representative VLMs on AlignMMBench, offering insights into the capabilities and limitations of different VLM architectures. The evaluation code and data are available at https://github.com/THUDM/AlignMMBench.
>
---
#### [replaced 041] SAB3R: Semantic-Augmented Backbone in 3D Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.02112v2](http://arxiv.org/pdf/2506.02112v2)**

> **作者:** Xuweiyi Chen; Tian Xia; Sihan Xu; Jianing Yang; Joyce Chai; Zezhou Cheng
>
> **备注:** 3D-LLM/VLA @ CVPR2025 | Project page: https://uva-computer-vision-lab.github.io/sab3r/
>
> **摘要:** We introduce a new task, Map and Locate, which unifies the traditionally distinct objectives of open-vocabulary segmentation - detecting and segmenting object instances based on natural language queries - and 3D reconstruction, the process of estimating a scene's 3D structure from visual inputs. Specifically, Map and Locate involves generating a point cloud from an unposed video and segmenting object instances based on open-vocabulary queries. This task serves as a critical step toward real-world embodied AI applications and introduces a practical task that bridges reconstruction, recognition and reorganization. To tackle this task, we introduce a simple yet effective baseline, which we denote as SAB3R. Our approach builds upon MASt3R, a recent breakthrough in 3D computer vision, and incorporates a lightweight distillation strategy. This method transfers dense, per-pixel semantic features from 2D vision backbones (eg, CLIP and DINOv2) to enhance MASt3R's capabilities. Without introducing any auxiliary frozen networks, our model generates per-pixel semantic features and constructs cohesive point maps in a single forward pass. Compared to separately deploying MASt3R and CLIP, our unified model, SAB3R, achieves superior performance on the Map and Locate benchmark. Furthermore, we evaluate SAB3R on both 2D semantic segmentation and 3D tasks to comprehensively validate its effectiveness.
>
---
#### [replaced 042] DreamFrame: Enhancing Video Understanding via Automatically Generated QA and Style-Consistent Keyframes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.01422v3](http://arxiv.org/pdf/2403.01422v3)**

> **作者:** Zhende Song; Chenchen Wang; Jiamu Sheng; Chi Zhang; Shengji Tang; Jiayuan Fan; Tao Chen
>
> **摘要:** Recent large vision-language models (LVLMs) for video understanding are primarily fine-tuned with various videos scraped from online platforms. Existing datasets, such as ActivityNet, require considerable human labor for structuring and annotation before effectively utilized for tuning LVLMs. While current LVLMs are primarily trained on existing datasets in broad, general-purpose settings, adapting them to specific downstream scenarios remains challenging, as collecting and annotating task-specific videos is highly labor-intensive and time-consuming. To address this issue, we propose a three-stage framework named DreamFrame for automatically generating style-consistent keyframes and corresponding question-answer (QA) pairs to support LVLM instruction tuning. DreamFrame generates datasets in a movie-like manner. First, we utilize an LLM to generate structured movie plots including movie prior information (like overview and style), frame descriptions and plot-related QA pairs, with a story expansion strategy to mitigate context length limitations.Then, to ensure visual consistency across generated frames, we design a Style Immobilization Process which maintains consistent style through an embedding learning strategy. Finally, frame descriptions and style embeddings are integrated to produce coherent keyframes. Using DreamFrame, we construct a dataset comprising approximately 1k stylized keyframe-like videos and 100k diverse QA pairs. Extensive fine-tuned experiments on various LVLM architectures demonstrate the effectiveness of the proposed dataset. Furthermore, based on the proposed dataset, we fine-tune a new LVLM named DreamFrame-7B, which significantly surpasses the previous similar-sized LVLMs across different benchmarks.
>
---
#### [replaced 043] DropCluster: A structured dropout for convolutional networks
- **分类: cs.LG; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2002.02997v2](http://arxiv.org/pdf/2002.02997v2)**

> **作者:** Liyan Chen; Philippos Mordohai; Sergul Aydore
>
> **备注:** 11 pages, 10 figures, under review
>
> **摘要:** Dropout as a common regularizer to prevent overfitting in deep neural networks has been less effective in convolutional layers than in fully connected layers. This is because Dropout drops features randomly, without considering local structure. When features are spatially correlated, as in the case of convolutional layers, information from the dropped features can still propagate to subsequent layers via neighboring features. To address this problem, structured forms of Dropout have been proposed. A drawback of these methods is that they do not adapt to the data. In this work, we leverage the structure in the outputs of convolutional layers and introduce a novel structured regularization method named DropCluster. Our approach clusters features in convolutional layers, and drops the resulting clusters randomly during training iterations. Experiments on CIFAR-10/100, SVHN, and APPA-REAL datasets demonstrate that our approach is effective and controls overfitting better than other approaches.
>
---
#### [replaced 044] VCT: Training Consistency Models with Variational Noise Coupling
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.18197v2](http://arxiv.org/pdf/2502.18197v2)**

> **作者:** Gianluigi Silvestri; Luca Ambrogioni; Chieh-Hsin Lai; Yuhta Takida; Yuki Mitsufuji
>
> **备注:** 23 pages, 11 figures
>
> **摘要:** Consistency Training (CT) has recently emerged as a strong alternative to diffusion models for image generation. However, non-distillation CT often suffers from high variance and instability, motivating ongoing research into its training dynamics. We propose Variational Consistency Training (VCT), a flexible and effective framework compatible with various forward kernels, including those in flow matching. Its key innovation is a learned noise-data coupling scheme inspired by Variational Autoencoders, where a data-dependent encoder models noise emission. This enables VCT to adaptively learn noise-todata pairings, reducing training variance relative to the fixed, unsorted pairings in classical CT. Experiments on multiple image datasets demonstrate significant improvements: our method surpasses baselines, achieves state-of-the-art FID among non-distillation CT approaches on CIFAR-10, and matches SoTA performance on ImageNet 64 x 64 with only two sampling steps. Code is available at https://github.com/sony/vct.
>
---
#### [replaced 045] Moving Beyond Discrete Categories: Continuous Demographic Labels for Fair Facial Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.01532v2](http://arxiv.org/pdf/2506.01532v2)**

> **作者:** Pedro C. Neto; Naser Damer; Jaime S. Cardoso; Ana F. Sequeira
>
> **备注:** Under review
>
> **摘要:** Bias has been a constant in face recognition models. Over the years, researchers have looked at it from both the model and the data point of view. However, their approach to mitigation of data bias was limited and lacked insight on the real nature of the problem. Here, in this document, we propose to revise our use of ethnicity labels as a continuous variable instead of a discrete value per identity. We validate our formulation both experimentally and theoretically, showcasing that not all identities from one ethnicity contribute equally to the balance of the dataset; thus, having the same number of identities per ethnicity does not represent a balanced dataset. We further show that models trained on datasets balanced in the continuous space consistently outperform models trained on data balanced in the discrete space. We trained more than 65 different models, and created more than 20 subsets of the original datasets.
>
---
#### [replaced 046] Open-PMC-18M: A High-Fidelity Large Scale Medical Dataset for Multimodal Representation Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.02738v2](http://arxiv.org/pdf/2506.02738v2)**

> **作者:** Negin Baghbanzadeh; Sajad Ashkezari; Elham Dolatabadi; Arash Afkanpour
>
> **备注:** 15 pages
>
> **摘要:** Compound figures, which are multi-panel composites containing diverse subfigures, are ubiquitous in biomedical literature, yet large-scale subfigure extraction remains largely unaddressed. Prior work on subfigure extraction has been limited in both dataset size and generalizability, leaving a critical open question: How does high-fidelity image-text alignment via large-scale subfigure extraction impact representation learning in vision-language models? We address this gap by introducing a scalable subfigure extraction pipeline based on transformer-based object detection, trained on a synthetic corpus of 500,000 compound figures, and achieving state-of-the-art performance on both ImageCLEF 2016 and synthetic benchmarks. Using this pipeline, we release OPEN-PMC-18M, a large-scale high quality biomedical vision-language dataset comprising 18 million clinically relevant subfigure-caption pairs spanning radiology, microscopy, and visible light photography. We train and evaluate vision-language models on our curated datasets and show improved performance across retrieval, zero-shot classification, and robustness benchmarks, outperforming existing baselines. We release our dataset, models, and code to support reproducible benchmarks and further study into biomedical vision-language modeling and representation learning.
>
---
#### [replaced 047] Bézier Splatting for Fast and Differentiable Vector Graphics Rendering
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.16424v3](http://arxiv.org/pdf/2503.16424v3)**

> **作者:** Xi Liu; Chaoyi Zhou; Nanxuan Zhao; Siyu Huang
>
> **备注:** Project page: https://xiliu8006.github.io/Bezier_splatting_project/
>
> **摘要:** Differentiable vector graphics (VGs) are widely used in image vectorization and vector synthesis, while existing representations are costly to optimize and struggle to achieve high-quality rendering results for high-resolution images. This work introduces a new differentiable VG representation, dubbed B\'ezier Splatting, that enables fast yet high-fidelity VG rasterization. B\'ezier Splatting samples 2D Gaussians along B\'ezier curves, which naturally provide positional gradients at object boundaries. Thanks to the efficient splatting-based differentiable rasterizer, B\'ezier Splatting achieves 30x and 150x faster per forward and backward rasterization step for open curves compared to DiffVG. Additionally, we introduce an adaptive pruning and densification strategy that dynamically adjusts the spatial distribution of curves to escape local minima, further improving VG quality. Furthermore, our new VG representation supports conversion to standard XML-based SVG format, enhancing interoperability with existing VG tools and pipelines. Experimental results show that B\'ezier Splatting significantly outperforms existing methods with better visual fidelity and significant optimization speedup.
>
---
#### [replaced 048] MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.05237v2](http://arxiv.org/pdf/2412.05237v2)**

> **作者:** Jarvis Guo; Tuney Zheng; Yuelin Bai; Bo Li; Yubo Wang; King Zhu; Yizhi Li; Graham Neubig; Wenhu Chen; Xiang Yue
>
> **备注:** ACL 2025 Main
>
> **摘要:** Open-source multimodal large language models (MLLMs) have shown significant potential in a broad range of multimodal tasks. However, their reasoning capabilities remain constrained by existing instruction-tuning datasets, which were predominately repurposed from academic datasets such as VQA, AI2D, and ChartQA. These datasets target simplistic tasks, and only provide phrase-level answers without any intermediate rationales. To address these challenges, we introduce a scalable and cost-effective method to construct a large-scale multimodal instruction-tuning dataset with rich intermediate rationales designed to elicit CoT reasoning. Using only open models, we create a dataset containing 12M instruction-response pairs to cover diverse, reasoning-intensive tasks with detailed and faithful rationales. Experiments demonstrate that training MLLMs on this dataset significantly improves reasoning capabilities, achieving state-of-the-art performance on benchmarks such as MathVerse (+8.1%), MMMU-Pro (+7%), and MuirBench (+13.3%). Additionally, the model demonstrates notable improvements of up to 4% on non-reasoning-based benchmarks. Ablation studies further highlight the importance of key components, such as rewriting and self-filtering, in the dataset construction process.
>
---
#### [replaced 049] CondiMen: Conditional Multi-Person Mesh Recovery
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.13058v2](http://arxiv.org/pdf/2412.13058v2)**

> **作者:** Brégier Romain; Baradel Fabien; Lucas Thomas; Galaaoui Salma; Armando Matthieu; Weinzaepfel Philippe; Rogez Grégory
>
> **备注:** accepted to the RHOBIN workshop at CVPR 2025
>
> **摘要:** Multi-person human mesh recovery (HMR) consists in detecting all individuals in a given input image, and predicting the body shape, pose, and 3D location for each detected person. The dominant approaches to this task rely on neural networks trained to output a single prediction for each detected individual. In contrast, we propose CondiMen, a method that outputs a joint parametric distribution over likely poses, body shapes, intrinsics and distances to the camera, using a Bayesian network. This approach offers several advantages. First, a probability distribution can handle some inherent ambiguities of this task -- such as the uncertainty between a person's size and their distance to the camera, or simply the loss of information when projecting 3D data onto the 2D image plane. Second, the output distribution can be combined with additional information to produce better predictions, by using e.g. known camera or body shape parameters, or by exploiting multi-view observations. Third, one can efficiently extract the most likely predictions from the output distribution, making our proposed approach suitable for real-time applications. Empirically we find that our model i) achieves performance on par with or better than the state-of-the-art, ii) captures uncertainties and correlations inherent in pose estimation and iii) can exploit additional information at test time, such as multi-view consistency or body shape priors. CondiMen spices up the modeling of ambiguity, using just the right ingredients on hand.
>
---
#### [replaced 050] Towards a deep learning approach for classifying treatment response in glioblastomas
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.18268v2](http://arxiv.org/pdf/2504.18268v2)**

> **作者:** Ana Matoso; Catarina Passarinho; Marta P. Loureiro; José Maria Moreira; Patrícia Figueiredo; Rita G. Nunes
>
> **摘要:** Glioblastomas are the most aggressive type of glioma, having a 5-year survival rate of 6.9%. Treatment typically involves surgery, followed by radiotherapy and chemotherapy, and frequent magnetic resonance imaging (MRI) scans to monitor disease progression. To assess treatment response, radiologists use the Response Assessment in Neuro-Oncology (RANO) criteria to categorize the tumor into one of four labels based on imaging and clinical features: complete response, partial response, stable disease, and progressive disease. This assessment is very complex and time-consuming. Since deep learning (DL) has been widely used to tackle classification problems, this work aimed to implement the first DL pipeline for the classification of RANO criteria based on two consecutive MRI acquisitions. The models were trained and tested on the open dataset LUMIERE. Five approaches were tested: 1) subtraction of input images, 2) different combinations of modalities, 3) different model architectures, 4) different pretraining tasks, and 5) adding clinical data. The pipeline that achieved the best performance used a Densenet264 considering only T1-weighted, T2-weighted, and Fluid Attenuated Inversion Recovery (FLAIR) images as input without any pretraining. A median Balanced Accuracy of 50.96% was achieved. Additionally, explainability methods were applied. Using Saliency Maps, the tumor region was often successfully highlighted. In contrast, Grad-CAM typically failed to highlight the tumor region, with some exceptions observed in the Complete Response and Progressive Disease classes, where it effectively identified the tumor region. These results set a benchmark for future studies on glioblastoma treatment response assessment based on the RANO criteria while emphasizing the heterogeneity of factors that might play a role when assessing the tumor's response to treatment.
>
---
#### [replaced 051] DiffoRA: Enabling Parameter-Efficient Fine-Tuning via Differential Module Selection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.08905v2](http://arxiv.org/pdf/2502.08905v2)**

> **作者:** Tangyu Jiang; Haodi Wang; Chun Yuan
>
> **摘要:** The Parameter-Efficient Fine-Tuning (PEFT) methods have been extensively researched for large language models in downstream tasks. Among all the existing approaches, the Low-Rank Adaptation (LoRA) has gained popularity for its streamlined design by incorporating low-rank matrices into existing pre-trained models. Though effective, LoRA, as well as its adaptive optimizations, either allocate the same matrix to all the modules or adjust the interior rank of the components based on importance scoring indicators. In this paper, we argue that not all the modules in LLMs are suitable and necessary to be fine-tuned. Enlightened by this insight, we propose a new PEFT scheme called DiffoRA, which enables adaptive adoption of the low-rank decomposition matrices. At the core of DiffoRA lies a Differential Adaptation Matrix (DAM) to determine which module is the most suitable and essential for fine-tuning. We theoretically explain how the designed matrix impacts the convergence rate and generalization capability of a pre-trained model. We then construct the DAM via continuous relaxation and discretization with weight-sharing optimizations. We fully implement DiffoRA and design comprehensive experiments to evaluate its performance. The experimental results demonstrate that DiffoRA delivers state-of-the-art results across multiple benchmarks.
>
---
#### [replaced 052] FlySearch: Exploring how vision-language models explore
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.02896v2](http://arxiv.org/pdf/2506.02896v2)**

> **作者:** Adam Pardyl; Dominik Matuszek; Mateusz Przebieracz; Marek Cygan; Bartosz Zieliński; Maciej Wołczyk
>
> **摘要:** The real world is messy and unstructured. Uncovering critical information often requires active, goal-driven exploration. It remains to be seen whether Vision-Language Models (VLMs), which recently emerged as a popular zero-shot tool in many difficult tasks, can operate effectively in such conditions. In this paper, we answer this question by introducing FlySearch, a 3D, outdoor, photorealistic environment for searching and navigating to objects in complex scenes. We define three sets of scenarios with varying difficulty and observe that state-of-the-art VLMs cannot reliably solve even the simplest exploration tasks, with the gap to human performance increasing as the tasks get harder. We identify a set of central causes, ranging from vision hallucination, through context misunderstanding, to task planning failures, and we show that some of them can be addressed by finetuning. We publicly release the benchmark, scenarios, and the underlying codebase.
>
---
#### [replaced 053] ReactDiff: Latent Diffusion for Facial Reaction Generation
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.14151v3](http://arxiv.org/pdf/2505.14151v3)**

> **作者:** Jiaming Li; Sheng Wang; Xin Wang; Yitao Zhu; Honglin Xiong; Zixu Zhuang; Qian Wang
>
> **备注:** Accepted by Neural Networks
>
> **摘要:** Given the audio-visual clip of the speaker, facial reaction generation aims to predict the listener's facial reactions. The challenge lies in capturing the relevance between video and audio while balancing appropriateness, realism, and diversity. While prior works have mostly focused on uni-modal inputs or simplified reaction mappings, recent approaches such as PerFRDiff have explored multi-modal inputs and the one-to-many nature of appropriate reaction mappings. In this work, we propose the Facial Reaction Diffusion (ReactDiff) framework that uniquely integrates a Multi-Modality Transformer with conditional diffusion in the latent space for enhanced reaction generation. Unlike existing methods, ReactDiff leverages intra- and inter-class attention for fine-grained multi-modal interaction, while the latent diffusion process between the encoder and decoder enables diverse yet contextually appropriate outputs. Experimental results demonstrate that ReactDiff significantly outperforms existing approaches, achieving a facial reaction correlation of 0.26 and diversity score of 0.094 while maintaining competitive realism. The code is open-sourced at \href{https://github.com/Hunan-Tiger/ReactDiff}{github}.
>
---
#### [replaced 054] NTIRE 2025 Challenge on RAW Image Restoration and Super-Resolution
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.02197v2](http://arxiv.org/pdf/2506.02197v2)**

> **作者:** Marcos V. Conde; Radu Timofte; Zihao Lu; Xiangyu Kong; Xiaoxia Xing; Fan Wang; Suejin Han; MinKyu Park; Tianyu Zhang; Xin Luo; Yeda Chen; Dong Liu; Li Pang; Yuhang Yang; Hongzhong Wang; Xiangyong Cao; Ruixuan Jiang; Senyan Xu; Siyuan Jiang; Xueyang Fu; Zheng-Jun Zha; Tianyu Hao; Yuhong He; Ruoqi Li; Yueqi Yang; Xiang Yu; Guanlan Hong; Minmin Yi; Yuanjia Chen; Liwen Zhang; Zijie Jin; Cheng Li; Lian Liu; Wei Song; Heng Sun; Yubo Wang; Jinghua Wang; Jiajie Lu; Watchara Ruangsan
>
> **备注:** CVPR 2025 - New Trends in Image Restoration and Enhancement (NTIRE)
>
> **摘要:** This paper reviews the NTIRE 2025 RAW Image Restoration and Super-Resolution Challenge, highlighting the proposed solutions and results. New methods for RAW Restoration and Super-Resolution could be essential in modern Image Signal Processing (ISP) pipelines, however, this problem is not as explored as in the RGB domain. The goal of this challenge is two fold, (i) restore RAW images with blur and noise degradations, (ii) upscale RAW Bayer images by 2x, considering unknown noise and blur. In the challenge, a total of 230 participants registered, and 45 submitted results during thee challenge period. This report presents the current state-of-the-art in RAW Restoration.
>
---
#### [replaced 055] Learning 3D Representations from Procedural 3D Programs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.17467v2](http://arxiv.org/pdf/2411.17467v2)**

> **作者:** Xuweiyi Chen; Zezhou Cheng
>
> **备注:** SynData4CV @ CVPR2025 | Project Page: https://point-mae-zero.cs.virginia.edu/
>
> **摘要:** Self-supervised learning has emerged as a promising approach for acquiring transferable 3D representations from unlabeled 3D point clouds. Unlike 2D images, which are widely accessible, acquiring 3D assets requires specialized expertise or professional 3D scanning equipment, making it difficult to scale and raising copyright concerns. To address these challenges, we propose learning 3D representations from procedural 3D programs that automatically generate 3D shapes using simple primitives and augmentations. Remarkably, despite lacking semantic content, the 3D representations learned from the procedurally generated 3D shapes perform on par with state-of-the-art representations learned from semantically recognizable 3D models (e.g., airplanes) across various downstream 3D tasks, including shape classification, part segmentation, and masked point cloud completion. We provide a detailed analysis on factors that make a good 3D procedural program. Extensive experiments further suggest that current self-supervised learning methods on point clouds do not rely on the semantics of 3D shapes, shedding light on the nature of 3D representations learned.
>
---
#### [replaced 056] RAC3: Retrieval-Augmented Corner Case Comprehension for Autonomous Driving with Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.11050v3](http://arxiv.org/pdf/2412.11050v3)**

> **作者:** Yujin Wang; Quanfeng Liu; Jiaqi Fan; Jinlong Hong; Hongqing Chu; Mengjian Tian; Bingzhao Gao; Hong Chen
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** Understanding and addressing corner cases is essential for ensuring the safety and reliability of autonomous driving systems. Vision-language models (VLMs) play a crucial role in enhancing scenario comprehension, yet they face significant challenges, such as hallucination and insufficient real-world grounding, which compromise their performance in critical driving scenarios. In this work, RAC3, a novel framework designed to enhance the performance of VLMs in corner case comprehension, is proposed. RAC3 integrates a frequency-spatial fusion (FSF) image encoder, a cross-modal alignment training method for embedding models with hard and semi-hard negative mining, and a fast querying and retrieval pipeline based on K-Means clustering and hierarchical navigable small world (HNSW) indexing. A multimodal chain-of-thought (CoT) prompting strategy to guide analogical reasoning and reduce hallucinations during inference is introduced. Moreover, an update mechanism is integrated into RAC3 to ensure continual learning within the framework. Extensive experiments on the CODA and nuScenes datasets demonstrate that RAC3 significantly improves corner case comprehension across multiple downstream tasks. Compared to prior state-of-the-art methods, RAC3 achieves the highest final score of 74.46 on the CODA-LM benchmark and shows consistent performance gains when integrated with end-to-end frameworks like DriveLM. These results demonstrate the effectiveness of retrieval-augmented strategies and cross-modal alignment for safer and more interpretable autonomous driving.
>
---
#### [replaced 057] Objective drives the consistency of representational similarity across datasets
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.05561v2](http://arxiv.org/pdf/2411.05561v2)**

> **作者:** Laure Ciernik; Lorenz Linhardt; Marco Morik; Jonas Dippel; Simon Kornblith; Lukas Muttenthaler
>
> **备注:** 26 pages
>
> **摘要:** The Platonic Representation Hypothesis claims that recent foundation models are converging to a shared representation space as a function of their downstream task performance, irrespective of the objectives and data modalities used to train these models (Huh et al., 2024). Representational similarity is generally measured for individual datasets and is not necessarily consistent across datasets. Thus, one may wonder whether this convergence of model representations is confounded by the datasets commonly used in machine learning. Here, we propose a systematic way to measure how representational similarity between models varies with the set of stimuli used to construct the representations. We find that the objective function is a crucial factor in determining the consistency of representational similarities across datasets. Specifically, self-supervised vision models learn representations whose relative pairwise similarities generalize better from one dataset to another compared to those of image classification or image-text models. Moreover, the correspondence between representational similarities and the models' task behavior is dataset-dependent, being most strongly pronounced for single-domain datasets. Our work provides a framework for analyzing similarities of model representations across datasets and linking those similarities to differences in task behavior.
>
---
#### [replaced 058] Single-Pass Object-Focused Data Selection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.10032v2](http://arxiv.org/pdf/2412.10032v2)**

> **作者:** Niclas Popp; Dan Zhang; Jan Hendrik Metzen; Matthias Hein; Lukas Schott
>
> **摘要:** While unlabeled image data is often plentiful, the costs of high-quality labels pose an important practical challenge: Which images should one select for labeling to use the annotation budget for a particular target task most effectively? To address this problem, we focus on single-pass data selection, which refers to the process of selecting all data to be annotated at once before training a downstream model. Prior methods for single-pass data selection rely on image-level representations and fail to reliably outperform random selection for object detection and segmentation. We propose Object-Focused Data Selection (OFDS) which leverages object-level features from foundation models and ensures semantic coverage of all target classes. In extensive experiments across tasks and target domains, OFDS consistently outperforms random selection and all baselines. The best results for constrained annotation budgets are obtained by combining human labels from OFDS with autolabels from foundation models. Moreover, using OFDS to select the initial labeled set for active learning yields consistent improvements
>
---
#### [replaced 059] DualMap: Online Open-Vocabulary Semantic Mapping for Natural Language Navigation in Dynamic Changing Scenes
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.01950v2](http://arxiv.org/pdf/2506.01950v2)**

> **作者:** Jiajun Jiang; Yiming Zhu; Zirui Wu; Jie Song
>
> **备注:** 8 pages, 5 figures. Code: https://github.com/Eku127/DualMap Project page: https://eku127.github.io/DualMap/
>
> **摘要:** We introduce DualMap, an online open-vocabulary mapping system that enables robots to understand and navigate dynamically changing environments through natural language queries. Designed for efficient semantic mapping and adaptability to changing environments, DualMap meets the essential requirements for real-world robot navigation applications. Our proposed hybrid segmentation frontend and object-level status check eliminate the costly 3D object merging required by prior methods, enabling efficient online scene mapping. The dual-map representation combines a global abstract map for high-level candidate selection with a local concrete map for precise goal-reaching, effectively managing and updating dynamic changes in the environment. Through extensive experiments in both simulation and real-world scenarios, we demonstrate state-of-the-art performance in 3D open-vocabulary segmentation, efficient scene mapping, and online language-guided navigation.
>
---
#### [replaced 060] M3-AGIQA: Multimodal, Multi-Round, Multi-Aspect AI-Generated Image Quality Assessment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.15167v2](http://arxiv.org/pdf/2502.15167v2)**

> **作者:** Chuan Cui; Kejiang Chen; Zhihua Wei; Wen Shen; Weiming Zhang; Nenghai Yu
>
> **备注:** 24 pages. This work has been submitted to the ACM for possible publication
>
> **摘要:** The rapid advancement of AI-generated image (AIGI) models presents new challenges for evaluating image quality, particularly across three aspects: perceptual quality, prompt correspondence, and authenticity. To address these challenges, we introduce M3-AGIQA, a comprehensive framework that leverages Multimodal Large Language Models (MLLMs) to enable more human-aligned, holistic evaluation of AI-generated images across both visual and textual domains. Besides, our framework features a structured multi-round evaluation process, generating and analyzing intermediate image descriptions to provide deeper insight into these three aspects. By aligning model outputs more closely with human judgment, M3-AGIQA delivers robust and interpretable quality scores. Extensive experiments on multiple benchmarks demonstrate that our method achieves state-of-the-art performance on tested datasets and aspects, and exhibits strong generalizability in most cross-dataset settings. Code is available at https://github.com/strawhatboy/M3-AGIQA.
>
---
#### [replaced 061] Grid-LOGAT: Grid Based Local and Global Area Transcription for Video Question Answering
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.24371v2](http://arxiv.org/pdf/2505.24371v2)**

> **作者:** Md Intisar Chowdhury; Kittinun Aukkapinyo; Hiroshi Fujimura; Joo Ann Woo; Wasu Wasusatein; Fadoua Ghourabi
>
> **备注:** Copyright 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** In this paper, we propose a Grid-based Local and Global Area Transcription (Grid-LoGAT) system for Video Question Answering (VideoQA). The system operates in two phases. First, extracting text transcripts from video frames using a Vision-Language Model (VLM). Next, processing questions using these transcripts to generate answers through a Large Language Model (LLM). This design ensures image privacy by deploying the VLM on edge devices and the LLM in the cloud. To improve transcript quality, we propose grid-based visual prompting, which extracts intricate local details from each grid cell and integrates them with global information. Evaluation results show that Grid-LoGAT, using the open-source VLM (LLaVA-1.6-7B) and LLM (Llama-3.1-8B), outperforms state-of-the-art methods with similar baseline models on NExT-QA and STAR-QA datasets with an accuracy of 65.9% and 50.11% respectively. Additionally, our method surpasses the non-grid version by 24 points on localization-based questions we created using NExT-QA. (This paper is accepted by IEEE ICIP 2025.)
>
---
#### [replaced 062] Your Turn: At Home Turning Angle Estimation for Parkinson's Disease Severity Assessment
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2408.08182v3](http://arxiv.org/pdf/2408.08182v3)**

> **作者:** Qiushuo Cheng; Catherine Morgan; Arindam Sikdar; Alessandro Masullo; Alan Whone; Majid Mirmehdi
>
> **摘要:** People with Parkinson's Disease (PD) often experience progressively worsening gait, including changes in how they turn around, as the disease progresses. Existing clinical rating tools are not capable of capturing hour-by-hour variations of PD symptoms, as they are confined to brief assessments within clinic settings. Measuring gait turning angles continuously and passively is a component step towards using gait characteristics as sensitive indicators of disease progression in PD. This paper presents a deep learning-based approach to automatically quantify turning angles by extracting 3D skeletons from videos and calculating the rotation of hip and knee joints. We utilise state-of-the-art human pose estimation models, Fastpose and Strided Transformer, on a total of 1386 turning video clips from 24 subjects (12 people with PD and 12 healthy control volunteers), trimmed from a PD dataset of unscripted free-living videos in a home-like setting (Turn-REMAP). We also curate a turning video dataset, Turn-H3.6M, from the public Human3.6M human pose benchmark with 3D ground truth, to further validate our method. Previous gait research has primarily taken place in clinics or laboratories evaluating scripted gait outcomes, but this work focuses on free-living home settings where complexities exist, such as baggy clothing and poor lighting. Due to difficulties in obtaining accurate ground truth data in a free-living setting, we quantise the angle into the nearest bin $45^\circ$ based on the manual labelling of expert clinicians. Our method achieves a turning calculation accuracy of 41.6%, a Mean Absolute Error (MAE) of 34.7{\deg}, and a weighted precision WPrec of 68.3% for Turn-REMAP. This is the first work to explore the use of single monocular camera data to quantify turns by PD patients in a home setting.
>
---
#### [replaced 063] SEM: Enhancing Spatial Understanding for Robust Robot Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16196v2](http://arxiv.org/pdf/2505.16196v2)**

> **作者:** Xuewu Lin; Tianwei Lin; Lichao Huang; Hongyu Xie; Yiwei Jin; Keyu Li; Zhizhong Su
>
> **摘要:** A key challenge in robot manipulation lies in developing policy models with strong spatial understanding, the ability to reason about 3D geometry, object relations, and robot embodiment. Existing methods often fall short: 3D point cloud models lack semantic abstraction, while 2D image encoders struggle with spatial reasoning. To address this, we propose SEM (Spatial Enhanced Manipulation model), a novel diffusion-based policy framework that explicitly enhances spatial understanding from two complementary perspectives. A spatial enhancer augments visual representations with 3D geometric context, while a robot state encoder captures embodiment-aware structure through graphbased modeling of joint dependencies. By integrating these modules, SEM significantly improves spatial understanding, leading to robust and generalizable manipulation across diverse tasks that outperform existing baselines.
>
---
#### [replaced 064] Large-Scale Text-to-Image Model with Inpainting is a Zero-Shot Subject-Driven Image Generator
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.15466v2](http://arxiv.org/pdf/2411.15466v2)**

> **作者:** Chaehun Shin; Jooyoung Choi; Heeseung Kim; Sungroh Yoon
>
> **备注:** CVPR 2025
>
> **摘要:** Subject-driven text-to-image generation aims to produce images of a new subject within a desired context by accurately capturing both the visual characteristics of the subject and the semantic content of a text prompt. Traditional methods rely on time- and resource-intensive fine-tuning for subject alignment, while recent zero-shot approaches leverage on-the-fly image prompting, often sacrificing subject alignment. In this paper, we introduce Diptych Prompting, a novel zero-shot approach that reinterprets as an inpainting task with precise subject alignment by leveraging the emergent property of diptych generation in large-scale text-to-image models. Diptych Prompting arranges an incomplete diptych with the reference image in the left panel, and performs text-conditioned inpainting on the right panel. We further prevent unwanted content leakage by removing the background in the reference image and improve fine-grained details in the generated subject by enhancing attention weights between the panels during inpainting. Experimental results confirm that our approach significantly outperforms zero-shot image prompting methods, resulting in images that are visually preferred by users. Additionally, our method supports not only subject-driven generation but also stylized image generation and subject-driven image editing, demonstrating versatility across diverse image generation applications. Project page: https://diptychprompting.github.io/
>
---
#### [replaced 065] T2I-FactualBench: Benchmarking the Factuality of Text-to-Image Models with Knowledge-Intensive Concepts
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.04300v3](http://arxiv.org/pdf/2412.04300v3)**

> **作者:** Ziwei Huang; Wanggui He; Quanyu Long; Yandi Wang; Haoyuan Li; Zhelun Yu; Fangxun Shu; Long Chan; Hao Jiang; Fei Wu; Leilei Gan
>
> **摘要:** Evaluating the quality of synthesized images remains a significant challenge in the development of text-to-image (T2I) generation. Most existing studies in this area primarily focus on evaluating text-image alignment, image quality, and object composition capabilities, with comparatively fewer studies addressing the evaluation of the factuality of T2I models, particularly when the concepts involved are knowledge-intensive. To mitigate this gap, we present T2I-FactualBench in this work - the largest benchmark to date in terms of the number of concepts and prompts specifically designed to evaluate the factuality of knowledge-intensive concept generation. T2I-FactualBench consists of a three-tiered knowledge-intensive text-to-image generation framework, ranging from the basic memorization of individual knowledge concepts to the more complex composition of multiple knowledge concepts. We further introduce a multi-round visual question answering (VQA) based evaluation framework to assess the factuality of three-tiered knowledge-intensive text-to-image generation tasks. Experiments on T2I-FactualBench indicate that current state-of-the-art (SOTA) T2I models still leave significant room for improvement.
>
---
#### [replaced 066] MDPE: A Multimodal Deception Dataset with Personality and Emotional Characteristics
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.12274v2](http://arxiv.org/pdf/2407.12274v2)**

> **作者:** Cong Cai; Shan Liang; Xuefei Liu; Kang Zhu; Zhengqi Wen; Jianhua Tao; Heng Xie; Jizhou Cui; Yiming Ma; Zhenhua Cheng; Hanzhe Xu; Ruibo Fu; Bin Liu; Yongwei Li
>
> **备注:** Code and data are available; Submitted to ACM Multimedia 2025 Dataset Track
>
> **摘要:** Deception detection has garnered increasing attention in recent years due to the significant growth of digital media and heightened ethical and security concerns. It has been extensively studied using multimodal methods, including video, audio, and text. In addition, individual differences in deception production and detection are believed to play a crucial role.Although some studies have utilized individual information such as personality traits to enhance the performance of deception detection, current systems remain limited, partly due to a lack of sufficient datasets for evaluating performance. To address this issue, we introduce a multimodal deception dataset MDPE. Besides deception features, this dataset also includes individual differences information in personality and emotional expression characteristics. It can explore the impact of individual differences on deception behavior. It comprises over 104 hours of deception and emotional videos from 193 subjects. Furthermore, we conducted numerous experiments to provide valuable insights for future deception detection research. MDPE not only supports deception detection, but also provides conditions for tasks such as personality recognition and emotion recognition, and can even study the relationships between them. We believe that MDPE will become a valuable resource for promoting research in the field of affective computing.
>
---
#### [replaced 067] Prescribing the Right Remedy: Mitigating Hallucinations in Large Vision-Language Models via Targeted Instruction Tuning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2404.10332v2](http://arxiv.org/pdf/2404.10332v2)**

> **作者:** Rui Hu; Yahan Tu; Shuyu Wei; Dongyuan Lu; Jitao Sang
>
> **备注:** Accepted in Information Sciences 2025
>
> **摘要:** Despite achieving outstanding performance on various cross-modal tasks, current large vision-language models (LVLMs) still suffer from hallucination issues, manifesting as inconsistencies between their generated responses and the corresponding images. Prior research has implicated that the low quality of instruction data, particularly the skewed balance between positive and negative samples, is a significant contributor to model hallucinations. Recently, researchers have proposed high-quality instruction datasets, such as LRV-Instruction, to mitigate model hallucination. Nonetheless, our investigation reveals that hallucinatory concepts from different LVLMs exhibit specificity, i.e. the distribution of hallucinatory concepts varies significantly across models. Existing datasets did not consider the hallucination specificity of different models in the design processes, thereby diminishing their efficacy in mitigating model hallucination. In this paper, we propose a targeted instruction data generation framework named DFTG that tailored to the hallucination specificity of different models. Concretely, DFTG consists of two stages: hallucination diagnosis, which extracts the necessary information from the model's responses and images for hallucination diagnosis; and targeted data generation, which generates targeted instruction data based on diagnostic results. The experimental results on hallucination benchmarks demonstrate that the targeted instruction data generated by our method are more effective in mitigating hallucinations compared to previous datasets.
>
---
#### [replaced 068] FlexTok: Resampling Images into 1D Token Sequences of Flexible Length
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13967v2](http://arxiv.org/pdf/2502.13967v2)**

> **作者:** Roman Bachmann; Jesse Allardice; David Mizrahi; Enrico Fini; Oğuzhan Fatih Kar; Elmira Amirloo; Alaaeldin El-Nouby; Amir Zamir; Afshin Dehghan
>
> **备注:** ICML 2025. Project page at https://flextok.epfl.ch/
>
> **摘要:** Image tokenization has enabled major advances in autoregressive image generation by providing compressed, discrete representations that are more efficient to process than raw pixels. While traditional approaches use 2D grid tokenization, recent methods like TiTok have shown that 1D tokenization can achieve high generation quality by eliminating grid redundancies. However, these methods typically use a fixed number of tokens and thus cannot adapt to an image's inherent complexity. We introduce FlexTok, a tokenizer that projects 2D images into variable-length, ordered 1D token sequences. For example, a 256x256 image can be resampled into anywhere from 1 to 256 discrete tokens, hierarchically and semantically compressing its information. By training a rectified flow model as the decoder and using nested dropout, FlexTok produces plausible reconstructions regardless of the chosen token sequence length. We evaluate our approach in an autoregressive generation setting using a simple GPT-style Transformer. On ImageNet, this approach achieves an FID<2 across 8 to 128 tokens, outperforming TiTok and matching state-of-the-art methods with far fewer tokens. We further extend the model to support to text-conditioned image generation and examine how FlexTok relates to traditional 2D tokenization. A key finding is that FlexTok enables next-token prediction to describe images in a coarse-to-fine "visual vocabulary", and that the number of tokens to generate depends on the complexity of the generation task.
>
---
#### [replaced 069] Crowd Scene Analysis using Deep Learning Techniques
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.08834v2](http://arxiv.org/pdf/2505.08834v2)**

> **作者:** Muhammad Junaid Asif
>
> **备注:** MS Graduate Research Thesis
>
> **摘要:** Our research is focused on two main applications of crowd scene analysis crowd counting and anomaly detection In recent years a large number of researches have been presented in the domain of crowd counting We addressed two main challenges in this domain 1 Deep learning models are datahungry paradigms and always need a large amount of annotated data for the training of algorithm It is timeconsuming and costly task to annotate such large amount of data Selfsupervised training is proposed to deal with this challenge 2 MCNN consists of multicolumns of CNN with different sizes of filters by presenting a novel approach based on a combination of selfsupervised training and MultiColumn CNN This enables the model to learn features at different levels and makes it effective in dealing with challenges of occluded scenes nonuniform density complex backgrounds and scale invariation The proposed model was evaluated on publicly available data sets such as ShanghaiTech and UCFQNRF by means of MAE and MSE A spatiotemporal model based on VGG19 is proposed for crowd anomaly detection addressing challenges like lighting environmental conditions unexpected objects and scalability The model extracts spatial and temporal features allowing it to be generalized to realworld scenes Spatial features are learned using CNN while temporal features are learned using LSTM blocks The model works on binary classification and can detect normal or abnormal behavior The models performance is improved by replacing fully connected layers with dense residual blocks Experiments on the Hockey Fight dataset and SCVD dataset show our models outperform other stateoftheart approaches
>
---
#### [replaced 070] Beyond Entropy: Region Confidence Proxy for Wild Test-Time Adaptation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20704v2](http://arxiv.org/pdf/2505.20704v2)**

> **作者:** Zixuan Hu; Yichun Hu; Xiaotong Li; Shixiang Tang; Ling-Yu Duan
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Wild Test-Time Adaptation (WTTA) is proposed to adapt a source model to unseen domains under extreme data scarcity and multiple shifts. Previous approaches mainly focused on sample selection strategies, while overlooking the fundamental problem on underlying optimization. Initially, we critically analyze the widely-adopted entropy minimization framework in WTTA and uncover its significant limitations in noisy optimization dynamics that substantially hinder adaptation efficiency. Through our analysis, we identify region confidence as a superior alternative to traditional entropy, however, its direct optimization remains computationally prohibitive for real-time applications. In this paper, we introduce a novel region-integrated method ReCAP that bypasses the lengthy process. Specifically, we propose a probabilistic region modeling scheme that flexibly captures semantic changes in embedding space. Subsequently, we develop a finite-to-infinite asymptotic approximation that transforms the intractable region confidence into a tractable and upper-bounded proxy. These innovations significantly unlock the overlooked potential dynamics in local region in a concise solution. Our extensive experiments demonstrate the consistent superiority of ReCAP over existing methods across various datasets and wild scenarios.
>
---
#### [replaced 071] Visual Attention Never Fades: Selective Progressive Attention ReCalibration for Detailed Image Captioning in Multimodal Large Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.01419v2](http://arxiv.org/pdf/2502.01419v2)**

> **作者:** Mingi Jung; Saehyung Lee; Eunji Kim; Sungroh Yoon
>
> **备注:** ICML 2025
>
> **摘要:** Detailed image captioning is essential for tasks like data generation and aiding visually impaired individuals. High-quality captions require a balance between precision and recall, which remains challenging for current multimodal large language models (MLLMs). In this work, we hypothesize that this limitation stems from weakening and increasingly noisy visual attention as responses lengthen. To address this issue, we propose SPARC (Selective Progressive Attention ReCalibration), a training-free method that enhances the contribution of visual tokens during decoding. SPARC is founded on three key observations: (1) increasing the influence of all visual tokens reduces recall; thus, SPARC selectively amplifies visual tokens; (2) as captions lengthen, visual attention becomes noisier, so SPARC identifies critical visual tokens by leveraging attention differences across time steps; (3) as visual attention gradually weakens, SPARC reinforces it to preserve its influence. Our experiments, incorporating both automated and human evaluations, demonstrate that existing methods improve the precision of MLLMs at the cost of recall. In contrast, our proposed method enhances both precision and recall with minimal computational overhead.
>
---
#### [replaced 072] Improving Knowledge Distillation Under Unknown Covariate Shift Through Confidence-Guided Data Augmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.02294v2](http://arxiv.org/pdf/2506.02294v2)**

> **作者:** Niclas Popp; Kevin Alexander Laube; Matthias Hein; Lukas Schott
>
> **摘要:** Large foundation models trained on extensive datasets demonstrate strong zero-shot capabilities in various domains. To replicate their success when data and model size are constrained, knowledge distillation has become an established tool for transferring knowledge from foundation models to small student networks. However, the effectiveness of distillation is critically limited by the available training data. This work addresses the common practical issue of covariate shift in knowledge distillation, where spurious features appear during training but not at test time. We ask the question: when these spurious features are unknown, yet a robust teacher is available, is it possible for a student to also become robust to them? We address this problem by introducing a novel diffusion-based data augmentation strategy that generates images by maximizing the disagreement between the teacher and the student, effectively creating challenging samples that the student struggles with. Experiments demonstrate that our approach significantly improves worst group and mean group accuracy on CelebA and SpuCo Birds as well as the spurious mAUC on spurious ImageNet under covariate shift, outperforming state-of-the-art diffusion-based data augmentation baselines
>
---
#### [replaced 073] MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.06141v4](http://arxiv.org/pdf/2412.06141v4)**

> **作者:** Kangyu Zhu; Peng Xia; Yun Li; Hongtu Zhu; Sheng Wang; Huaxiu Yao
>
> **备注:** ICML 2025
>
> **摘要:** The advancement of Large Vision-Language Models (LVLMs) has propelled their application in the medical field. However, Medical LVLMs (Med-LVLMs) encounter factuality challenges due to modality misalignment, where the models prioritize textual knowledge over visual input, leading to hallucinations that contradict information in medical images. Previous attempts to enhance modality alignment in Med-LVLMs through preference optimization have inadequately mitigated clinical relevance in preference data, making these samples easily distinguishable and reducing alignment effectiveness. To address this challenge, we propose MMedPO, a novel multimodal medical preference optimization approach that considers the clinical relevance of preference samples to enhance Med-LVLM alignment. MMedPO curates multimodal preference data by introducing two types of dispreference: (1) plausible hallucinations injected through target Med-LVLMs or GPT-4o to produce medically inaccurate responses, and (2) lesion region neglect achieved through local lesion-noising, disrupting visual understanding of critical areas. We then calculate clinical relevance for each sample based on scores from multiple Med-LLMs and visual tools, and integrate these scores into the preference optimization process as weights, enabling effective alignment. Our experiments demonstrate that MMedPO significantly enhances factual accuracy in Med-LVLMs, achieving substantial improvements over existing preference optimization methods by averaging 14.2% and 51.7% across the Med-VQA and report generation tasks. Our code are available in https://github.com/aiming-lab/MMedPO.
>
---
#### [replaced 074] The Role of Visual Modality in Multimodal Mathematical Reasoning: Challenges and Insights
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.04167v2](http://arxiv.org/pdf/2503.04167v2)**

> **作者:** Yufang Liu; Yao Du; Tao Ji; Jianing Wang; Yang Liu; Yuanbin Wu; Aimin Zhou; Mengdi Zhang; Xunliang Cai
>
> **摘要:** Recent research has increasingly focused on multimodal mathematical reasoning, particularly emphasizing the creation of relevant datasets and benchmarks. Despite this, the role of visual information in reasoning has been underexplored. Our findings show that existing multimodal mathematical models minimally leverage visual information, and model performance remains largely unaffected by changes to or removal of images in the dataset. We attribute this to the dominance of textual information and answer options that inadvertently guide the model to correct answers. To improve evaluation methods, we introduce the HC-M3D dataset, specifically designed to require image reliance for problem-solving and to challenge models with similar, yet distinct, images that change the correct answer. In testing leading models, their failure to detect these subtle visual differences suggests limitations in current visual perception capabilities. Additionally, we observe that the common approach of improving general VQA capabilities by combining various types of image encoders does not contribute to math reasoning performance. This finding also presents a challenge to enhancing visual reliance during math reasoning. Our benchmark and code would be available at \href{https://github.com/Yufang-Liu/visual_modality_role}{https://github.com/Yufang-Liu/visual\_modality\_role}.
>
---
#### [replaced 075] MammAlps: A multi-view video behavior monitoring dataset of wild mammals in the Swiss Alps
- **分类: cs.CV; cs.IR; q-bio.NC; q-bio.QM**

- **链接: [http://arxiv.org/pdf/2503.18223v2](http://arxiv.org/pdf/2503.18223v2)**

> **作者:** Valentin Gabeff; Haozhe Qi; Brendan Flaherty; Gencer Sumbül; Alexander Mathis; Devis Tuia
>
> **备注:** CVPR 2025; Benchmark and code at: https://github.com/eceo-epfl/MammAlps. After submission of v1, we noticed that a few audio files were not correctly aligned with the corresponding video. We fixed the issue, which had little to no impact on performance. We also now report results for three runs
>
> **摘要:** Monitoring wildlife is essential for ecology and ethology, especially in light of the increasing human impact on ecosystems. Camera traps have emerged as habitat-centric sensors enabling the study of wildlife populations at scale with minimal disturbance. However, the lack of annotated video datasets limits the development of powerful video understanding models needed to process the vast amount of fieldwork data collected. To advance research in wild animal behavior monitoring we present MammAlps, a multimodal and multi-view dataset of wildlife behavior monitoring from 9 camera-traps in the Swiss National Park. MammAlps contains over 14 hours of video with audio, 2D segmentation maps and 8.5 hours of individual tracks densely labeled for species and behavior. Based on 6135 single animal clips, we propose the first hierarchical and multimodal animal behavior recognition benchmark using audio, video and reference scene segmentation maps as inputs. Furthermore, we also propose a second ecology-oriented benchmark aiming at identifying activities, species, number of individuals and meteorological conditions from 397 multi-view and long-term ecological events, including false positive triggers. We advocate that both tasks are complementary and contribute to bridging the gap between machine learning and ecology. Code and data are available at: https://github.com/eceo-epfl/MammAlps
>
---
#### [replaced 076] FlowMo: Variance-Based Flow Guidance for Coherent Motion in Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.01144v2](http://arxiv.org/pdf/2506.01144v2)**

> **作者:** Ariel Shaulov; Itay Hazan; Lior Wolf; Hila Chefer
>
> **摘要:** Text-to-video diffusion models are notoriously limited in their ability to model temporal aspects such as motion, physics, and dynamic interactions. Existing approaches address this limitation by retraining the model or introducing external conditioning signals to enforce temporal consistency. In this work, we explore whether a meaningful temporal representation can be extracted directly from the predictions of a pre-trained model without any additional training or auxiliary inputs. We introduce FlowMo, a novel training-free guidance method that enhances motion coherence using only the model's own predictions in each diffusion step. FlowMo first derives an appearance-debiased temporal representation by measuring the distance between latents corresponding to consecutive frames. This highlights the implicit temporal structure predicted by the model. It then estimates motion coherence by measuring the patch-wise variance across the temporal dimension and guides the model to reduce this variance dynamically during sampling. Extensive experiments across multiple text-to-video models demonstrate that FlowMo significantly improves motion coherence without sacrificing visual quality or prompt alignment, offering an effective plug-and-play solution for enhancing the temporal fidelity of pre-trained video diffusion models.
>
---
#### [replaced 077] MM-IQ: Benchmarking Human-Like Abstraction and Reasoning in Multimodal Models
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.00698v2](http://arxiv.org/pdf/2502.00698v2)**

> **作者:** Huanqia Cai; Yijun Yang; Winston Hu
>
> **摘要:** IQ testing has served as a foundational methodology for evaluating human cognitive capabilities, deliberately decoupling assessment from linguistic background, language proficiency, or domain-specific knowledge to isolate core competencies in abstraction and reasoning. Yet, artificial intelligence research currently lacks systematic benchmarks to quantify these critical cognitive capabilities in multimodal systems. To address this crucial gap, we propose MM-IQ, a comprehensive evaluation framework, which comprises a large-scale training set with 4,776 visual reasoning problems and 2,710 meticulously curated test items spanning 8 distinct reasoning paradigms. Through systematic evaluation of existing open-source and proprietary multimodal models, our benchmark reveals striking limitations: even state-of-the-art architectures achieve only marginally superior performance to random chance (33.17% vs. 25% baseline accuracy). This substantial performance chasm highlights the inadequacy of current multimodal models in approximating fundamental human reasoning capacities, underscoring the need for paradigm-shifting advancements to bridge this cognitive divide. Moreover, inspired by the recent surge of large reasoning models, we also release a multimodal reasoning model as the baseline that is trained via reinforcement learning with verifiable reward functions, reaching competitive performance to the state-of-the-art with a notably smaller model size.
>
---
#### [replaced 078] Multi-Source Collaborative Style Augmentation and Domain-Invariant Learning for Federated Domain Generalization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10152v2](http://arxiv.org/pdf/2505.10152v2)**

> **作者:** Yikang Wei
>
> **备注:** IJCAI 2025
>
> **摘要:** Federated domain generalization aims to learn a generalizable model from multiple decentralized source domains for deploying on the unseen target domain. The style augmentation methods have achieved great progress on domain generalization. However, the existing style augmentation methods either explore the data styles within isolated source domain or interpolate the style information across existing source domains under the data decentralization scenario, which leads to limited style space. To address this issue, we propose a Multi-source Collaborative Style Augmentation and Domain-invariant learning method (MCSAD) for federated domain generalization. Specifically, we propose a multi-source collaborative style augmentation module to generate data in the broader style space. Furthermore, we conduct domain-invariant learning between the original data and augmented data by cross-domain feature alignment within the same class and classes relation ensemble distillation between different classes to learn a domain-invariant model. By alternatively conducting collaborative style augmentation and domain-invariant learning, the model can generalize well on unseen target domain. Extensive experiments on multiple domain generalization datasets indicate that our method significantly outperforms the state-of-the-art federated domain generalization methods.
>
---
#### [replaced 079] Escaping Plato's Cave: Towards the Alignment of 3D and Text Latent Spaces
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.05283v2](http://arxiv.org/pdf/2503.05283v2)**

> **作者:** Souhail Hadgi; Luca Moschella; Andrea Santilli; Diego Gomez; Qixing Huang; Emanuele Rodolà; Simone Melzi; Maks Ovsjanikov
>
> **备注:** CVPR 2025
>
> **摘要:** Recent works have shown that, when trained at scale, uni-modal 2D vision and text encoders converge to learned features that share remarkable structural properties, despite arising from different representations. However, the role of 3D encoders with respect to other modalities remains unexplored. Furthermore, existing 3D foundation models that leverage large datasets are typically trained with explicit alignment objectives with respect to frozen encoders from other representations. In this work, we investigate the possibility of a posteriori alignment of representations obtained from uni-modal 3D encoders compared to text-based feature spaces. We show that naive post-training feature alignment of uni-modal text and 3D encoders results in limited performance. We then focus on extracting subspaces of the corresponding feature spaces and discover that by projecting learned representations onto well-chosen lower-dimensional subspaces the quality of alignment becomes significantly higher, leading to improved accuracy on matching and retrieval tasks. Our analysis further sheds light on the nature of these shared subspaces, which roughly separate between semantic and geometric data representations. Overall, ours is the first work that helps to establish a baseline for post-training alignment of 3D uni-modal and text feature spaces, and helps to highlight both the shared and unique properties of 3D data compared to other representations. Our code and weights are available at https://github.com/Souhail-01/3d-text-alignment
>
---
#### [replaced 080] Reasoning is All You Need for Video Generalization: A Counterfactual Benchmark with Sub-question Evaluation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10691v2](http://arxiv.org/pdf/2503.10691v2)**

> **作者:** Qiji Zhou; Yifan Gong; Guangsheng Bao; Hongjie Qiu; Jinqiang Li; Xiangrong Zhu; Huajian Zhang; Yue Zhang
>
> **备注:** It has been accepted to the ACL-2025 Findings
>
> **摘要:** Counterfactual reasoning is crucial for robust video understanding but remains underexplored in existing multimodal benchmarks. In this paper, we introduce \textbf{COVER} (\textbf{\underline{CO}}unterfactual \textbf{\underline{V}}id\textbf{\underline{E}}o \textbf{\underline{R}}easoning), a multidimensional multimodal benchmark that systematically evaluates MLLMs across the abstract-concrete and perception-cognition dimensions. Beyond prior multimodal benchmarks, COVER decomposes complex queries into structured sub-questions, enabling fine-grained reasoning analysis. Experiments on commercial and open-source models reveal a strong correlation between sub-question accuracy and counterfactual reasoning performance, highlighting the role of structured inference in video understanding. Furthermore, our results suggest a key insight: enhancing the reasoning capability of models is essential for improving the robustness of video understanding. COVER establishes a new standard for assessing MLLMs' logical reasoning abilities in dynamic environments. Our work is available at https://github.com/gongyifan-hash/COVER-Benchmark.
>
---
#### [replaced 081] MedEBench: Revisiting Text-instructed Image Editing on Medical Domain
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.01921v3](http://arxiv.org/pdf/2506.01921v3)**

> **作者:** Minghao Liu; Zhitao He; Zhiyuan Fan; Qingyun Wang; Yi R. Fung
>
> **备注:** Project website: https://mliuby.github.io/MedEBench_Website/
>
> **摘要:** Text-guided image editing has seen rapid progress in natural image domains, but its adaptation to medical imaging remains limited and lacks standardized evaluation. Clinically, such editing holds promise for simulating surgical outcomes, creating personalized teaching materials, and enhancing patient communication. To bridge this gap, we introduce MedEBench, a comprehensive benchmark for evaluating text-guided medical image editing. It consists of 1,182 clinically sourced image-prompt triplets spanning 70 tasks across 13 anatomical regions. MedEBench offers three key contributions: (1) a clinically relevant evaluation framework covering Editing Accuracy, Contextual Preservation, and Visual Quality, supported by detailed descriptions of expected change and ROI (Region of Interest) masks; (2) a systematic comparison of seven state-of-the-art models, revealing common failure patterns; and (3) a failure analysis protocol based on attention grounding, using IoU between attention maps and ROIs to identify mislocalization. MedEBench provides a solid foundation for developing and evaluating reliable, clinically meaningful medical image editing systems. Project website: https://mliuby.github.io/MedEBench_Website/
>
---
#### [replaced 082] EnergyMoGen: Compositional Human Motion Generation with Energy-Based Diffusion Model in Latent Space
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.14706v2](http://arxiv.org/pdf/2412.14706v2)**

> **作者:** Jianrong Zhang; Hehe Fan; Yi Yang
>
> **备注:** Accepted to CVPR 2025. Project page: https://jiro-zhang.github.io/EnergyMoGen/
>
> **摘要:** Diffusion models, particularly latent diffusion models, have demonstrated remarkable success in text-driven human motion generation. However, it remains challenging for latent diffusion models to effectively compose multiple semantic concepts into a single, coherent motion sequence. To address this issue, we propose EnergyMoGen, which includes two spectrums of Energy-Based Models: (1) We interpret the diffusion model as a latent-aware energy-based model that generates motions by composing a set of diffusion models in latent space; (2) We introduce a semantic-aware energy model based on cross-attention, which enables semantic composition and adaptive gradient descent for text embeddings. To overcome the challenges of semantic inconsistency and motion distortion across these two spectrums, we introduce Synergistic Energy Fusion. This design allows the motion latent diffusion model to synthesize high-quality, complex motions by combining multiple energy terms corresponding to textual descriptions. Experiments show that our approach outperforms existing state-of-the-art models on various motion generation tasks, including text-to-motion generation, compositional motion generation, and multi-concept motion generation. Additionally, we demonstrate that our method can be used to extend motion datasets and improve the text-to-motion task.
>
---
#### [replaced 083] Comparing the Effects of Persistence Barcodes Aggregation and Feature Concatenation on Medical Imaging
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.23637v2](http://arxiv.org/pdf/2505.23637v2)**

> **作者:** Dashti A. Ali; Richard K. G. Do; William R. Jarnagin; Aras T. Asaad; Amber L. Simpson
>
> **备注:** 16 pages, 8 figures
>
> **摘要:** In medical image analysis, feature engineering plays an important role in the design and performance of machine learning models. Persistent homology (PH), from the field of topological data analysis (TDA), demonstrates robustness and stability to data perturbations and addresses the limitation from traditional feature extraction approaches where a small change in input results in a large change in feature representation. Using PH, we store persistent topological and geometrical features in the form of the persistence barcode whereby large bars represent global topological features and small bars encapsulate geometrical information of the data. When multiple barcodes are computed from 2D or 3D medical images, two approaches can be used to construct the final topological feature vector in each dimension: aggregating persistence barcodes followed by featurization or concatenating topological feature vectors derived from each barcode. In this study, we conduct a comprehensive analysis across diverse medical imaging datasets to compare the effects of the two aforementioned approaches on the performance of classification models. The results of this analysis indicate that feature concatenation preserves detailed topological information from individual barcodes, yields better classification performance and is therefore a preferred approach when conducting similar experiments.
>
---
#### [replaced 084] Implicit Inversion turns CLIP into a Decoder
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.23161v2](http://arxiv.org/pdf/2505.23161v2)**

> **作者:** Antonio D'Orazio; Maria Rosaria Briglia; Donato Crisostomi; Dario Loi; Emanuele Rodolà; Iacopo Masi
>
> **摘要:** CLIP is a discriminative model trained to align images and text in a shared embedding space. Due to its multimodal structure, it serves as the backbone of many generative pipelines, where a decoder is trained to map from the shared space back to images. In this work, we show that image synthesis is nevertheless possible using CLIP alone -- without any decoder, training, or fine-tuning. Our approach optimizes a frequency-aware implicit neural representation that encourages coarse-to-fine generation by stratifying frequencies across network layers. To stabilize this inverse mapping, we introduce adversarially robust initialization, a lightweight Orthogonal Procrustes projection to align local text and image embeddings, and a blending loss that anchors outputs to natural image statistics. Without altering CLIP's weights, this framework unlocks capabilities such as text-to-image generation, style transfer, and image reconstruction. These findings suggest that discriminative models may hold untapped generative potential, hidden in plain sight.
>
---
#### [replaced 085] Mixed Non-linear Quantization for Vision Transformers
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2407.18437v2](http://arxiv.org/pdf/2407.18437v2)**

> **作者:** Gihwan Kim; Jemin Lee; Sihyeong Park; Yongin Kwon; Hyungshin Kim
>
> **备注:** 16 pages, 4 figures, Accepted in ECCV Workshops 2024
>
> **摘要:** The majority of quantization methods have been proposed to reduce the model size of Vision Transformers, yet most of them have overlooked the quantization of non-linear operations. Only a few works have addressed quantization for non-linear operations, but they applied a single quantization method across all non-linear operations. We believe that this can be further improved by employing a different quantization method for each non-linear operation. Therefore, to assign the most error-minimizing quantization method from the known methods to each non-linear layer, we propose a mixed non-linear quantization that considers layer-wise quantization sensitivity measured by SQNR difference metric. The results show that our method outperforms I-BERT, FQ-ViT, and I-ViT in both 8-bit and 6-bit settings for ViT, DeiT, and Swin models by an average of 0.6%p and 19.6%p, respectively. Our method outperforms I-BERT and I-ViT by 0.6%p and 20.8%p, respectively, when training time is limited. We plan to release our code at https://gitlab.com/ones-ai/mixed-non-linear-quantization.
>
---
#### [replaced 086] Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.09586v2](http://arxiv.org/pdf/2412.09586v2)**

> **作者:** Fiona Ryan; Ajay Bati; Sangmin Lee; Daniel Bolya; Judy Hoffman; James M. Rehg
>
> **备注:** CVPR 2025 Highlight
>
> **摘要:** We address the problem of gaze target estimation, which aims to predict where a person is looking in a scene. Predicting a person's gaze target requires reasoning both about the person's appearance and the contents of the scene. Prior works have developed increasingly complex, hand-crafted pipelines for gaze target estimation that carefully fuse features from separate scene encoders, head encoders, and auxiliary models for signals like depth and pose. Motivated by the success of general-purpose feature extractors on a variety of visual tasks, we propose Gaze-LLE, a novel transformer framework that streamlines gaze target estimation by leveraging features from a frozen DINOv2 encoder. We extract a single feature representation for the scene, and apply a person-specific positional prompt to decode gaze with a lightweight module. We demonstrate state-of-the-art performance across several gaze benchmarks and provide extensive analysis to validate our design choices. Our code is available at: http://github.com/fkryan/gazelle .
>
---
#### [replaced 087] SemEval-2025 Task 1: AdMIRe -- Advancing Multimodal Idiomaticity Representation
- **分类: cs.CL; cs.CV; I.2.7; I.4.m**

- **链接: [http://arxiv.org/pdf/2503.15358v3](http://arxiv.org/pdf/2503.15358v3)**

> **作者:** Thomas Pickard; Aline Villavicencio; Maggie Mi; Wei He; Dylan Phelps; Marco Idiart
>
> **备注:** Author accepted version; SemEval-2025 proceedings to appear at ACL 2025. This version corrects a typo in the results table
>
> **摘要:** Idiomatic expressions present a unique challenge in NLP, as their meanings are often not directly inferable from their constituent words. Despite recent advancements in Large Language Models (LLMs), idiomaticity remains a significant obstacle to robust semantic representation. We present datasets and tasks for SemEval-2025 Task 1: AdMiRe (Advancing Multimodal Idiomaticity Representation), which challenges the community to assess and improve models' ability to interpret idiomatic expressions in multimodal contexts and in multiple languages. Participants competed in two subtasks: ranking images based on their alignment with idiomatic or literal meanings, and predicting the next image in a sequence. The most effective methods achieved human-level performance by leveraging pretrained LLMs and vision-language models in mixture-of-experts settings, with multiple queries used to smooth over the weaknesses in these models' representations of idiomaticity.
>
---
#### [replaced 088] EscapeCraft: A 3D Room Escape Environment for Benchmarking Complex Multimodal Reasoning Ability
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10042v4](http://arxiv.org/pdf/2503.10042v4)**

> **作者:** Ziyue Wang; Yurui Dong; Fuwen Luo; Minyuan Ruan; Zhili Cheng; Chi Chen; Peng Li; Yang Liu
>
> **摘要:** The rapid advancing of Multimodal Large Language Models (MLLMs) has spurred interest in complex multimodal reasoning tasks in the real-world and virtual environment, which require coordinating multiple abilities, including visual perception, visual reasoning, spatial awareness, and target deduction. However, existing evaluations primarily assess the final task completion, often degrading assessments to isolated abilities such as visual grounding and visual question answering. Less attention is given to comprehensively and quantitatively analyzing reasoning process in multimodal environments, which is crucial for understanding model behaviors and underlying reasoning mechanisms beyond merely task success. To address this, we introduce MM-Escape, an extensible benchmark for investigating multimodal reasoning, inspired by real-world escape games. MM-Escape emphasizes intermediate model behaviors alongside final task completion. To achieve this, we develop EscapeCraft, a customizable and open environment that enables models to engage in free-form exploration for assessing multimodal reasoning. Extensive experiments show that MLLMs, regardless of scale, can successfully complete the simplest room escape tasks, with some exhibiting human-like exploration strategies. Yet, performance dramatically drops as task difficulty increases. Moreover, we observe that performance bottlenecks vary across models, revealing distinct failure modes and limitations in their multimodal reasoning abilities, such as repetitive trajectories without adaptive exploration, getting stuck in corners due to poor visual spatial awareness, and ineffective use of acquired props, such as the key. We hope our work sheds light on new challenges in multimodal reasoning, and uncovers potential improvements in MLLMs capabilities.
>
---
#### [replaced 089] MemoryOut: Learning Principal Features via Multimodal Sparse Filtering Network for Semi-supervised Video Anomaly Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.02535v2](http://arxiv.org/pdf/2506.02535v2)**

> **作者:** Juntong Li; Lingwei Dang; Yukun Su; Yun Hao; Qingxin Xiao; Yongwei Nie; Qingyao Wu
>
> **摘要:** Video Anomaly Detection (VAD) methods based on reconstruction or prediction face two critical challenges: (1) strong generalization capability often results in accurate reconstruction or prediction of abnormal events, making it difficult to distinguish normal from abnormal patterns; (2) reliance only on low-level appearance and motion cues limits their ability to identify high-level semantic in abnormal events from complex scenes. To address these limitations, we propose a novel VAD framework with two key innovations. First, to suppress excessive generalization, we introduce the Sparse Feature Filtering Module (SFFM) that employs bottleneck filters to dynamically and adaptively remove abnormal information from features. Unlike traditional memory modules, it does not need to memorize the normal prototypes across the training dataset. Further, we design the Mixture of Experts (MoE) architecture for SFFM. Each expert is responsible for extracting specialized principal features during running time, and different experts are selectively activated to ensure the diversity of the learned principal features. Second, to overcome the neglect of semantics in existing methods, we integrate a Vision-Language Model (VLM) to generate textual descriptions for video clips, enabling comprehensive joint modeling of semantic, appearance, and motion cues. Additionally, we enforce modality consistency through semantic similarity constraints and motion frame-difference contrastive loss. Extensive experiments on multiple public datasets validate the effectiveness of our multimodal joint modeling framework and sparse feature filtering paradigm. Project page at https://qzfm.github.io/sfn_vad_project_page/.
>
---
#### [replaced 090] CARL: Camera-Agnostic Representation Learning for Spectral Image Analysis
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.19223v2](http://arxiv.org/pdf/2504.19223v2)**

> **作者:** Alexander Baumann; Leonardo Ayala; Silvia Seidlitz; Jan Sellner; Alexander Studier-Fischer; Berkin Özdemir; Lena Maier-Hein; Slobodan Ilic
>
> **摘要:** Spectral imaging offers promising applications across diverse domains, including medicine and urban scene understanding, and is already established as a critical modality in remote sensing. However, variability in channel dimensionality and captured wavelengths among spectral cameras impede the development of AI-driven methodologies, leading to camera-specific models with limited generalizability and inadequate cross-camera applicability. To address this bottleneck, we introduce $\textbf{CARL}$, a model for $\textbf{C}$amera-$\textbf{A}$gnostic $\textbf{R}$epresentation $\textbf{L}$earning across RGB, multispectral, and hyperspectral imaging modalities. To enable the conversion of a spectral image with any channel dimensionality to a camera-agnostic embedding, we introduce wavelength positional encoding and a self-attention-cross-attention mechanism to compress spectral information into learned query representations. Spectral-spatial pre-training is achieved with a novel spectral self-supervised JEPA-inspired strategy tailored to CARL. Large-scale experiments across the domains of medical imaging, autonomous driving, and satellite imaging demonstrate our model's unique robustness to spectral heterogeneity, outperforming on datasets with simulated and real-world cross-camera spectral variations. The scalability and versatility of the proposed approach position our model as a backbone for future spectral foundation models.
>
---
