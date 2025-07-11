# 计算机视觉 cs.CV

- **最新发布 117 篇**

- **更新 59 篇**

## 最新发布

#### [new 001] Attend-and-Refine: Interactive keypoint estimation and quantitative cervical vertebrae analysis for bone age assessment
- **分类: cs.CV**

- **简介: 该论文属于骨龄评估任务，旨在通过分析颈椎形态准确估计儿童生长潜力。提出ARNet模型，提升关键点标注效率与准确性。**

- **链接: [http://arxiv.org/pdf/2507.07670v1](http://arxiv.org/pdf/2507.07670v1)**

> **作者:** Jinhee Kim; Taesung Kim; Taewoo Kim; Dong-Wook Kim; Byungduk Ahn; Yoon-Ji Kim; In-Seok Song; Jaegul Choo
>
> **备注:** Accepted to Medical Image Analysis (2025)
>
> **摘要:** In pediatric orthodontics, accurate estimation of growth potential is essential for developing effective treatment strategies. Our research aims to predict this potential by identifying the growth peak and analyzing cervical vertebra morphology solely through lateral cephalometric radiographs. We accomplish this by comprehensively analyzing cervical vertebral maturation (CVM) features from these radiographs. This methodology provides clinicians with a reliable and efficient tool to determine the optimal timings for orthodontic interventions, ultimately enhancing patient outcomes. A crucial aspect of this approach is the meticulous annotation of keypoints on the cervical vertebrae, a task often challenged by its labor-intensive nature. To mitigate this, we introduce Attend-and-Refine Network (ARNet), a user-interactive, deep learning-based model designed to streamline the annotation process. ARNet features Interaction-guided recalibration network, which adaptively recalibrates image features in response to user feedback, coupled with a morphology-aware loss function that preserves the structural consistency of keypoints. This novel approach substantially reduces manual effort in keypoint identification, thereby enhancing the efficiency and accuracy of the process. Extensively validated across various datasets, ARNet demonstrates remarkable performance and exhibits wide-ranging applicability in medical imaging. In conclusion, our research offers an effective AI-assisted diagnostic tool for assessing growth potential in pediatric orthodontics, marking a significant advancement in the field.
>
---
#### [new 002] Motion-Aware Adaptive Pixel Pruning for Efficient Local Motion Deblurring
- **分类: cs.CV; I.4.3**

- **简介: 该论文属于图像去模糊任务，解决局部运动模糊问题。通过提出可训练掩码预测器和帧内运动分析器，实现像素级剪枝与自适应模糊修复。**

- **链接: [http://arxiv.org/pdf/2507.07708v1](http://arxiv.org/pdf/2507.07708v1)**

> **作者:** Wei Shang; Dongwei Ren; Wanying Zhang; Pengfei Zhu; Qinghua Hu; Wangmeng Zuo
>
> **备注:** Accepted by ACMMM 2025
>
> **摘要:** Local motion blur in digital images originates from the relative motion between dynamic objects and static imaging systems during exposure. Existing deblurring methods face significant challenges in addressing this problem due to their inefficient allocation of computational resources and inadequate handling of spatially varying blur patterns. To overcome these limitations, we first propose a trainable mask predictor that identifies blurred regions in the image. During training, we employ blur masks to exclude sharp regions. For inference optimization, we implement structural reparameterization by converting $3\times 3$ convolutions to computationally efficient $1\times 1$ convolutions, enabling pixel-level pruning of sharp areas to reduce computation. Second, we develop an intra-frame motion analyzer that translates relative pixel displacements into motion trajectories, establishing adaptive guidance for region-specific blur restoration. Our method is trained end-to-end using a combination of reconstruction loss, reblur loss, and mask loss guided by annotated blur masks. Extensive experiments demonstrate superior performance over state-of-the-art methods on both local and global blur datasets while reducing FLOPs by 49\% compared to SOTA models (e.g., LMD-ViT). The source code is available at https://github.com/shangwei5/M2AENet.
>
---
#### [new 003] Explainable Artificial Intelligence in Biomedical Image Analysis: A Comprehensive Survey
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分析任务，旨在解决深度学习模型缺乏可解释性的问题。通过系统综述XAI方法，提出模态分类体系，探讨多模态与视觉语言模型的应用，总结评估指标与挑战。**

- **链接: [http://arxiv.org/pdf/2507.07148v1](http://arxiv.org/pdf/2507.07148v1)**

> **作者:** Getamesay Haile Dagnaw; Yanming Zhu; Muhammad Hassan Maqsood; Wencheng Yang; Xingshuai Dong; Xuefei Yin; Alan Wee-Chung Liew
>
> **摘要:** Explainable artificial intelligence (XAI) has become increasingly important in biomedical image analysis to promote transparency, trust, and clinical adoption of DL models. While several surveys have reviewed XAI techniques, they often lack a modality-aware perspective, overlook recent advances in multimodal and vision-language paradigms, and provide limited practical guidance. This survey addresses this gap through a comprehensive and structured synthesis of XAI methods tailored to biomedical image analysis.We systematically categorize XAI methods, analyzing their underlying principles, strengths, and limitations within biomedical contexts. A modality-centered taxonomy is proposed to align XAI methods with specific imaging types, highlighting the distinct interpretability challenges across modalities. We further examine the emerging role of multimodal learning and vision-language models in explainable biomedical AI, a topic largely underexplored in previous work. Our contributions also include a summary of widely used evaluation metrics and open-source frameworks, along with a critical discussion of persistent challenges and future directions. This survey offers a timely and in-depth foundation for advancing interpretable DL in biomedical image analysis.
>
---
#### [new 004] Martian World Models: Controllable Video Synthesis with Physically Accurate 3D Reconstructions
- **分类: cs.CV**

- **简介: 该论文属于火星场景视频生成任务，解决数据稀缺与领域差异问题。提出M3arsSynth和MarsGen，生成高精度3D视频，提升视觉与结构一致性。**

- **链接: [http://arxiv.org/pdf/2507.07978v1](http://arxiv.org/pdf/2507.07978v1)**

> **作者:** Longfei Li; Zhiwen Fan; Wenyan Cong; Xinhang Liu; Yuyang Yin; Matt Foutter; Panwang Pan; Chenyu You; Yue Wang; Zhangyang Wang; Yao Zhao; Marco Pavone; Yunchao Wei
>
> **备注:** Project Page: https://marsgenai.github.io
>
> **摘要:** Synthesizing realistic Martian landscape videos is crucial for mission rehearsal and robotic simulation. However, this task poses unique challenges due to the scarcity of high-quality Martian data and the significant domain gap between Martian and terrestrial imagery. To address these challenges, we propose a holistic solution composed of two key components: 1) A data curation pipeline Multimodal Mars Synthesis (M3arsSynth), which reconstructs 3D Martian environments from real stereo navigation images, sourced from NASA's Planetary Data System (PDS), and renders high-fidelity multiview 3D video sequences. 2) A Martian terrain video generator, MarsGen, which synthesizes novel videos visually realistic and geometrically consistent with the 3D structure encoded in the data. Our M3arsSynth engine spans a wide range of Martian terrains and acquisition dates, enabling the generation of physically accurate 3D surface models at metric-scale resolution. MarsGen, fine-tuned on M3arsSynth data, synthesizes videos conditioned on an initial image frame and, optionally, camera trajectories or textual prompts, allowing for video generation in novel environments. Experimental results show that our approach outperforms video synthesis models trained on terrestrial datasets, achieving superior visual fidelity and 3D structural consistency.
>
---
#### [new 005] D-CNN and VQ-VAE Autoencoders for Compression and Denoising of Industrial X-ray Computed Tomography Images
- **分类: cs.CV**

- **简介: 该论文属于图像压缩与去噪任务，研究如何用D-CNN和VQ-VAE压缩工业XCT数据，比较不同架构对图像质量的影响。**

- **链接: [http://arxiv.org/pdf/2507.07704v1](http://arxiv.org/pdf/2507.07704v1)**

> **作者:** Bardia Hejazi; Keerthana Chand; Tobias Fritsch; Giovanni Bruno
>
> **摘要:** The ever-growing volume of data in imaging sciences stemming from the advancements in imaging technologies, necessitates efficient and reliable storage solutions for such large datasets. This study investigates the compression of industrial X-ray computed tomography (XCT) data using deep learning autoencoders and examines how these compression algorithms affect the quality of the recovered data. Two network architectures with different compression rates were used, a deep convolution neural network (D-CNN) and a vector quantized variational autoencoder (VQ-VAE). The XCT data used was from a sandstone sample with a complex internal pore network. The quality of the decoded images obtained from the two different deep learning architectures with different compression rates were quantified and compared to the original input data. In addition, to improve image decoding quality metrics, we introduced a metric sensitive to edge preservation, which is crucial for three-dimensional data analysis. We showed that different architectures and compression rates are required depending on the specific characteristics needed to be preserved for later analysis. The findings presented here can aid scientists to determine the requirements and strategies for their data storage and analysis needs.
>
---
#### [new 006] MGVQ: Could VQ-VAE Beat VAE? A Generalizable Tokenizer with Multi-group Quantization
- **分类: cs.CV**

- **简介: 该论文属于图像重建任务，旨在提升VQ-VAE的重构质量。提出MGVQ方法，通过多组量化增强离散代码本表示能力，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.07997v1](http://arxiv.org/pdf/2507.07997v1)**

> **作者:** Mingkai Jia; Wei Yin; Xiaotao Hu; Jiaxin Guo; Xiaoyang Guo; Qian Zhang; Xiao-Xiao Long; Ping Tan
>
> **摘要:** Vector Quantized Variational Autoencoders (VQ-VAEs) are fundamental models that compress continuous visual data into discrete tokens. Existing methods have tried to improve the quantization strategy for better reconstruction quality, however, there still exists a large gap between VQ-VAEs and VAEs. To narrow this gap, we propose \NickName, a novel method to augment the representation capability of discrete codebooks, facilitating easier optimization for codebooks and minimizing information loss, thereby enhancing reconstruction quality. Specifically, we propose to retain the latent dimension to preserve encoded features and incorporate a set of sub-codebooks for quantization. Furthermore, we construct comprehensive zero-shot benchmarks featuring resolutions of 512p and 2k to evaluate the reconstruction performance of existing methods rigorously. \NickName~achieves the \textbf{state-of-the-art performance on both ImageNet and $8$ zero-shot benchmarks} across all VQ-VAEs. Notably, compared with SD-VAE, we outperform them on ImageNet significantly, with rFID $\textbf{0.49}$ v.s. $\textbf{0.91}$, and achieve superior PSNR on all zero-shot benchmarks. These results highlight the superiority of \NickName~in reconstruction and pave the way for preserving fidelity in HD image processing tasks. Code will be publicly available at https://github.com/MKJia/MGVQ.
>
---
#### [new 007] Interpretable EEG-to-Image Generation with Semantic Prompts
- **分类: cs.CV; cs.LG; eess.SP**

- **简介: 该论文属于脑信号到图像的解码任务，旨在解决EEG空间细节不足导致的图像重建难题。通过语义提示对齐EEG与多层级语义描述，实现可解释的视觉解码。**

- **链接: [http://arxiv.org/pdf/2507.07157v1](http://arxiv.org/pdf/2507.07157v1)**

> **作者:** Arshak Rezvani; Ali Akbari; Kosar Sanjar Arani; Maryam Mirian; Emad Arasteh; Martin J. McKeown
>
> **备注:** Actionable Interpretability Workshop (non-archival) at the 42 International Conference on Machine Learning
>
> **摘要:** Decoding visual experience from brain signals offers exciting possibilities for neuroscience and interpretable AI. While EEG is accessible and temporally precise, its limitations in spatial detail hinder image reconstruction. Our model bypasses direct EEG-to-image generation by aligning EEG signals with multilevel semantic captions -- ranging from object-level to abstract themes -- generated by a large language model. A transformer-based EEG encoder maps brain activity to these captions through contrastive learning. During inference, caption embeddings retrieved via projection heads condition a pretrained latent diffusion model for image generation. This text-mediated framework yields state-of-the-art visual decoding on the EEGCVPR dataset, with interpretable alignment to known neurocognitive pathways. Dominant EEG-caption associations reflected the importance of different semantic levels extracted from perceived images. Saliency maps and t-SNE projections reveal semantic topography across the scalp. Our model demonstrates how structured semantic mediation enables cognitively aligned visual decoding from EEG.
>
---
#### [new 008] EscherNet++: Simultaneous Amodal Completion and Scalable View Synthesis through Masked Fine-Tuning and Enhanced Feed-Forward 3D Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决物体在遮挡下的视图合成与完整重建问题。通过masked fine-tuning和增强的前馈重建，实现高效、准确的3D重建。**

- **链接: [http://arxiv.org/pdf/2507.07410v1](http://arxiv.org/pdf/2507.07410v1)**

> **作者:** Xinan Zhang; Muhammad Zubair Irshad; Anthony Yezzi; Yi-Chang Tsai; Zsolt Kira
>
> **摘要:** We propose EscherNet++, a masked fine-tuned diffusion model that can synthesize novel views of objects in a zero-shot manner with amodal completion ability. Existing approaches utilize multiple stages and complex pipelines to first hallucinate missing parts of the image and then perform novel view synthesis, which fail to consider cross-view dependencies and require redundant storage and computing for separate stages. Instead, we apply masked fine-tuning including input-level and feature-level masking to enable an end-to-end model with the improved ability to synthesize novel views and conduct amodal completion. In addition, we empirically integrate our model with other feed-forward image-to-mesh models without extra training and achieve competitive results with reconstruction time decreased by 95%, thanks to its ability to synthesize arbitrary query views. Our method's scalable nature further enhances fast 3D reconstruction. Despite fine-tuning on a smaller dataset and batch size, our method achieves state-of-the-art results, improving PSNR by 3.9 and Volume IoU by 0.28 on occluded tasks in 10-input settings, while also generalizing to real-world occluded reconstruction.
>
---
#### [new 009] MUVOD: A Novel Multi-view Video Object Segmentation Dataset and A Benchmark for 3D Segmentation
- **分类: cs.CV**

- **简介: 该论文属于多视角视频目标分割任务，旨在解决动态场景4D分割数据不足的问题。提出MUVOD数据集及3D分割基准，包含多视角标注数据和评估方法。**

- **链接: [http://arxiv.org/pdf/2507.07519v1](http://arxiv.org/pdf/2507.07519v1)**

> **作者:** Bangning Wei; Joshua Maraval; Meriem Outtas; Kidiyo Kpalma; Nicolas Ramin; Lu Zhang
>
> **摘要:** The application of methods based on Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3D GS) have steadily gained popularity in the field of 3D object segmentation in static scenes. These approaches demonstrate efficacy in a range of 3D scene understanding and editing tasks. Nevertheless, the 4D object segmentation of dynamic scenes remains an underexplored field due to the absence of a sufficiently extensive and accurately labelled multi-view video dataset. In this paper, we present MUVOD, a new multi-view video dataset for training and evaluating object segmentation in reconstructed real-world scenarios. The 17 selected scenes, describing various indoor or outdoor activities, are collected from different sources of datasets originating from various types of camera rigs. Each scene contains a minimum of 9 views and a maximum of 46 views. We provide 7830 RGB images (30 frames per video) with their corresponding segmentation mask in 4D motion, meaning that any object of interest in the scene could be tracked across temporal frames of a given view or across different views belonging to the same camera rig. This dataset, which contains 459 instances of 73 categories, is intended as a basic benchmark for the evaluation of multi-view video segmentation methods. We also present an evaluation metric and a baseline segmentation approach to encourage and evaluate progress in this evolving field. Additionally, we propose a new benchmark for 3D object segmentation task with a subset of annotated multi-view images selected from our MUVOD dataset. This subset contains 50 objects of different conditions in different scenarios, providing a more comprehensive analysis of state-of-the-art 3D object segmentation methods. Our proposed MUVOD dataset is available at https://volumetric-repository.labs.b-com.com/#/muvod.
>
---
#### [new 010] Automated Video Segmentation Machine Learning Pipeline
- **分类: cs.CV**

- **简介: 该论文属于视频分割任务，旨在解决VFX中手动分割效率低的问题。通过机器学习实现自动化、时序一致的实例分割，提升生产效率。**

- **链接: [http://arxiv.org/pdf/2507.07242v1](http://arxiv.org/pdf/2507.07242v1)**

> **作者:** Johannes Merz; Lucien Fostier
>
> **摘要:** Visual effects (VFX) production often struggles with slow, resource-intensive mask generation. This paper presents an automated video segmentation pipeline that creates temporally consistent instance masks. It employs machine learning for: (1) flexible object detection via text prompts, (2) refined per-frame image segmentation and (3) robust video tracking to ensure temporal stability. Deployed using containerization and leveraging a structured output format, the pipeline was quickly adopted by our artists. It significantly reduces manual effort, speeds up the creation of preliminary composites, and provides comprehensive segmentation data, thereby enhancing overall VFX production efficiency.
>
---
#### [new 011] HOTA: Hierarchical Overlap-Tiling Aggregation for Large-Area 3D Flood Mapping
- **分类: cs.CV**

- **简介: 该论文属于3D洪水制图任务，解决大范围洪水范围与深度信息不足的问题。提出HOTA方法，结合SegFormer和深度估计模块，实现高精度大范围洪水三维地图生成。**

- **链接: [http://arxiv.org/pdf/2507.07585v1](http://arxiv.org/pdf/2507.07585v1)**

> **作者:** Wenfeng Jia; Bin Liang; Yuxi Lu; Attavit Wilaiwongsakul; Muhammad Arif Khan; Lihong Zheng
>
> **摘要:** Floods are among the most frequent natural hazards and cause significant social and economic damage. Timely, large-scale information on flood extent and depth is essential for disaster response; however, existing products often trade spatial detail for coverage or ignore flood depth altogether. To bridge this gap, this work presents HOTA: Hierarchical Overlap-Tiling Aggregation, a plug-and-play, multi-scale inference strategy. When combined with SegFormer and a dual-constraint depth estimation module, this approach forms a complete 3D flood-mapping pipeline. HOTA applies overlapping tiles of different sizes to multispectral Sentinel-2 images only during inference, enabling the SegFormer model to capture both local features and kilometre-scale inundation without changing the network weights or retraining. The subsequent depth module is based on a digital elevation model (DEM) differencing method, which refines the 2D mask and estimates flood depth by enforcing (i) zero depth along the flood boundary and (ii) near-constant flood volume with respect to the DEM. A case study on the March 2021 Kempsey (Australia) flood shows that HOTA, when coupled with SegFormer, improves IoU from 73\% (U-Net baseline) to 84\%. The resulting 3D surface achieves a mean absolute boundary error of less than 0.5 m. These results demonstrate that HOTA can produce accurate, large-area 3D flood maps suitable for rapid disaster response.
>
---
#### [new 012] Bridging the gap in FER: addressing age bias in deep learning
- **分类: cs.CV**

- **简介: 该论文属于面部表情识别任务，旨在解决深度学习模型中的年龄偏见问题。通过分析不同年龄组的表现差异，提出三种缓解策略以提高老年人表情识别的准确性。**

- **链接: [http://arxiv.org/pdf/2507.07638v1](http://arxiv.org/pdf/2507.07638v1)**

> **作者:** F. Xavier Gaya-Morey; Julia Sanchez-Perez; Cristina Manresa-Yee; Jose M. Buades-Rubio
>
> **摘要:** Facial Expression Recognition (FER) systems based on deep learning have achieved impressive performance in recent years. However, these models often exhibit demographic biases, particularly with respect to age, which can compromise their fairness and reliability. In this work, we present a comprehensive study of age-related bias in deep FER models, with a particular focus on the elderly population. We first investigate whether recognition performance varies across age groups, which expressions are most affected, and whether model attention differs depending on age. Using Explainable AI (XAI) techniques, we identify systematic disparities in expression recognition and attention patterns, especially for "neutral", "sadness", and "anger" in elderly individuals. Based on these findings, we propose and evaluate three bias mitigation strategies: Multi-task Learning, Multi-modal Input, and Age-weighted Loss. Our models are trained on a large-scale dataset, AffectNet, with automatically estimated age labels and validated on balanced benchmark datasets that include underrepresented age groups. Results show consistent improvements in recognition accuracy for elderly individuals, particularly for the most error-prone expressions. Saliency heatmap analysis reveals that models trained with age-aware strategies attend to more relevant facial regions for each age group, helping to explain the observed improvements. These findings suggest that age-related bias in FER can be effectively mitigated using simple training modifications, and that even approximate demographic labels can be valuable for promoting fairness in large-scale affective computing systems.
>
---
#### [new 013] EEvAct: Early Event-Based Action Recognition with High-Rate Two-Stream Spiking Neural Networks
- **分类: cs.CV; cs.NE**

- **简介: 该论文属于早期动作识别任务，旨在提升事件相机的实时性与准确性。通过高率双流脉冲神经网络，提高早期预测性能并优化最终识别准确率。**

- **链接: [http://arxiv.org/pdf/2507.07734v1](http://arxiv.org/pdf/2507.07734v1)**

> **作者:** Michael Neumeier; Jules Lecomte; Nils Kazinski; Soubarna Banik; Bing Li; Axel von Arnim
>
> **备注:** International Conference on Neuromorphic Systems (ICONS) 2025
>
> **摘要:** Recognizing human activities early is crucial for the safety and responsiveness of human-robot and human-machine interfaces. Due to their high temporal resolution and low latency, event-based vision sensors are a perfect match for this early recognition demand. However, most existing processing approaches accumulate events to low-rate frames or space-time voxels which limits the early prediction capabilities. In contrast, spiking neural networks (SNNs) can process the events at a high-rate for early predictions, but most works still fall short on final accuracy. In this work, we introduce a high-rate two-stream SNN which closes this gap by outperforming previous work by 2% in final accuracy on the large-scale THU EACT-50 dataset. We benchmark the SNNs within a novel early event-based recognition framework by reporting Top-1 and Top-5 recognition scores for growing observation time. Finally, we exemplify the impact of these methods on a real-world task of early action triggering for human motion capture in sports.
>
---
#### [new 014] MIRA: A Novel Framework for Fusing Modalities in Medical RAG
- **分类: cs.CV**

- **简介: 该论文属于医疗多模态任务，旨在解决MLLM在医学诊断中事实不一致的问题。提出MIRA框架，优化检索与生成，提升准确性。**

- **链接: [http://arxiv.org/pdf/2507.07902v1](http://arxiv.org/pdf/2507.07902v1)**

> **作者:** Jinhong Wang; Tajamul Ashraf; Zongyan Han; Jorma Laaksonen; Rao Mohammad Anwer
>
> **备注:** ACM Multimedia 2025
>
> **摘要:** Multimodal Large Language Models (MLLMs) have significantly advanced AI-assisted medical diagnosis, but they often generate factually inconsistent responses that deviate from established medical knowledge. Retrieval-Augmented Generation (RAG) enhances factual accuracy by integrating external sources, but it presents two key challenges. First, insufficient retrieval can miss critical information, whereas excessive retrieval can introduce irrelevant or misleading content, disrupting model output. Second, even when the model initially provides correct answers, over-reliance on retrieved data can lead to factual errors. To address these issues, we introduce the Multimodal Intelligent Retrieval and Augmentation (MIRA) framework, designed to optimize factual accuracy in MLLM. MIRA consists of two key components: (1) a calibrated Rethinking and Rearrangement module that dynamically adjusts the number of retrieved contexts to manage factual risk, and (2) A medical RAG framework integrating image embeddings and a medical knowledge base with a query-rewrite module for efficient multimodal reasoning. This enables the model to effectively integrate both its inherent knowledge and external references. Our evaluation of publicly available medical VQA and report generation benchmarks demonstrates that MIRA substantially enhances factual accuracy and overall performance, achieving new state-of-the-art results. Code is released at https://github.com/mbzuai-oryx/MIRA.
>
---
#### [new 015] Adaptive Particle-Based Shape Modeling for Anatomical Surface Correspondence
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决 anatomical surface correspondence 的问题。通过引入新机制提升粒子模型的自适应性，以更准确地表示解剖结构的复杂变化。**

- **链接: [http://arxiv.org/pdf/2507.07379v1](http://arxiv.org/pdf/2507.07379v1)**

> **作者:** Hong Xu; Shireen Y. Elhabian
>
> **摘要:** Particle-based shape modeling (PSM) is a family of approaches that automatically quantifies shape variability across anatomical cohorts by positioning particles (pseudo landmarks) on shape surfaces in a consistent configuration. Recent advances incorporate implicit radial basis function representations as self-supervised signals to better capture the complex geometric properties of anatomical structures. However, these methods still lack self-adaptivity -- that is, the ability to automatically adjust particle configurations to local geometric features of each surface, which is essential for accurately representing complex anatomical variability. This paper introduces two mechanisms to increase surface adaptivity while maintaining consistent particle configurations: (1) a novel neighborhood correspondence loss to enable high adaptivity and (2) a geodesic correspondence algorithm that regularizes optimization to enforce geodesic neighborhood consistency. We evaluate the efficacy and scalability of our approach on challenging datasets, providing a detailed analysis of the adaptivity-correspondence trade-off and benchmarking against existing methods on surface representation accuracy and correspondence metrics.
>
---
#### [new 016] Robust and Generalizable Heart Rate Estimation via Deep Learning for Remote Photoplethysmography in Complex Scenarios
- **分类: cs.CV; F.2.2**

- **简介: 该论文属于心率估计任务，旨在提升远程光电容积描记（rPPG）在复杂场景下的准确性与泛化能力。通过引入3D CNN、差分帧融合模块和动态混合损失函数，提高了模型性能。**

- **链接: [http://arxiv.org/pdf/2507.07795v1](http://arxiv.org/pdf/2507.07795v1)**

> **作者:** Kang Cen; Chang-Hong Fu; Hong Hong
>
> **备注:** 7 pages, 3 figures
>
> **摘要:** Non-contact remote photoplethysmography (rPPG) technology enables heart rate measurement from facial videos. However, existing network models still face challenges in accu racy, robustness, and generalization capability under complex scenarios. This paper proposes an end-to-end rPPG extraction network that employs 3D convolutional neural networks to reconstruct accurate rPPG signals from raw facial videos. We introduce a differential frame fusion module that integrates differential frames with original frames, enabling frame-level representations to capture blood volume pulse (BVP) variations. Additionally, we incorporate Temporal Shift Module (TSM) with self-attention mechanisms, which effectively enhance rPPG features with minimal computational overhead. Furthermore, we propose a novel dynamic hybrid loss function that provides stronger supervision for the network, effectively mitigating over fitting. Comprehensive experiments were conducted on not only the PURE and UBFC-rPPG datasets but also the challenging MMPD dataset under complex scenarios, involving both intra dataset and cross-dataset evaluations, which demonstrate the superior robustness and generalization capability of our network. Specifically, after training on PURE, our model achieved a mean absolute error (MAE) of 7.58 on the MMPD test set, outperforming the state-of-the-art models.
>
---
#### [new 017] Towards High-Resolution 3D Anomaly Detection: A Scalable Dataset and Real-Time Framework for Subtle Industrial Defects
- **分类: cs.CV**

- **简介: 该论文属于3D异常检测任务，旨在解决工业中细微缺陷检测问题。提出MiniShift数据集和Simple3D框架，实现高分辨率、实时准确的检测。**

- **链接: [http://arxiv.org/pdf/2507.07435v1](http://arxiv.org/pdf/2507.07435v1)**

> **作者:** Yuqi Cheng; Yihan Sun; Hui Zhang; Weiming Shen; Yunkang Cao
>
> **备注:** 14 pages, 8figures
>
> **摘要:** In industrial point cloud analysis, detecting subtle anomalies demands high-resolution spatial data, yet prevailing benchmarks emphasize low-resolution inputs. To address this disparity, we propose a scalable pipeline for generating realistic and subtle 3D anomalies. Employing this pipeline, we developed MiniShift, the inaugural high-resolution 3D anomaly detection dataset, encompassing 2,577 point clouds, each with 500,000 points and anomalies occupying less than 1\% of the total. We further introduce Simple3D, an efficient framework integrating Multi-scale Neighborhood Descriptors (MSND) and Local Feature Spatial Aggregation (LFSA) to capture intricate geometric details with minimal computational overhead, achieving real-time inference exceeding 20 fps. Extensive evaluations on MiniShift and established benchmarks demonstrate that Simple3D surpasses state-of-the-art methods in both accuracy and speed, highlighting the pivotal role of high-resolution data and effective feature aggregation in advancing practical 3D anomaly detection.
>
---
#### [new 018] SCOOTER: A Human Evaluation Framework for Unrestricted Adversarial Examples
- **分类: cs.CV**

- **简介: 该论文属于对抗样本评估任务，解决无约束对抗样本的不可察觉性验证问题。提出SCOOTER框架，包含评估指南、实验对比、工具和数据集，以人类评价为核心。**

- **链接: [http://arxiv.org/pdf/2507.07776v1](http://arxiv.org/pdf/2507.07776v1)**

> **作者:** Dren Fazlija; Monty-Maximilian Zühlke; Johanna Schrader; Arkadij Orlov; Clara Stein; Iyiola E. Olatunji; Daniel Kudenko
>
> **备注:** 42 pages, 16 figures, 11 tables, Under Review, Code: https://github.com/DrenFazlija/Scooter, Data: https://doi.org/10.5281/zenodo.15771501
>
> **摘要:** Unrestricted adversarial attacks aim to fool computer vision models without being constrained by $\ell_p$-norm bounds to remain imperceptible to humans, for example, by changing an object's color. This allows attackers to circumvent traditional, norm-bounded defense strategies such as adversarial training or certified defense strategies. However, due to their unrestricted nature, there are also no guarantees of norm-based imperceptibility, necessitating human evaluations to verify just how authentic these adversarial examples look. While some related work assesses this vital quality of adversarial attacks, none provide statistically significant insights. This issue necessitates a unified framework that supports and streamlines such an assessment for evaluating and comparing unrestricted attacks. To close this gap, we introduce SCOOTER - an open-source, statistically powered framework for evaluating unrestricted adversarial examples. Our contributions are: $(i)$ best-practice guidelines for crowd-study power, compensation, and Likert equivalence bounds to measure imperceptibility; $(ii)$ the first large-scale human vs. model comparison across 346 human participants showing that three color-space attacks and three diffusion-based attacks fail to produce imperceptible images. Furthermore, we found that GPT-4o can serve as a preliminary test for imperceptibility, but it only consistently detects adversarial examples for four out of six tested attacks; $(iii)$ open-source software tools, including a browser-based task template to collect annotations and analysis scripts in Python and R; $(iv)$ an ImageNet-derived benchmark dataset containing 3K real images, 7K adversarial examples, and over 34K human ratings. Our findings demonstrate that automated vision systems do not align with human perception, reinforcing the need for a ground-truth SCOOTER benchmark.
>
---
#### [new 019] Entity Re-identification in Visual Storytelling via Contrastive Reinforcement Learning
- **分类: cs.CV; I.2; I.4; I.5; I.7**

- **简介: 该论文属于视觉叙事任务，解决模型在多帧中无法保持实体一致性的问题。通过对比强化学习方法提升实体识别与连接能力。**

- **链接: [http://arxiv.org/pdf/2507.07340v1](http://arxiv.org/pdf/2507.07340v1)**

> **作者:** Daniel A. P. Oliveira; David Martins de Matos
>
> **备注:** 7 pages
>
> **摘要:** Visual storytelling systems, particularly large vision-language models, struggle to maintain character and object identity across frames, often failing to recognize when entities in different images represent the same individuals or objects, leading to inconsistent references and referential hallucinations. This occurs because models lack explicit training on when to establish entity connections across frames. We propose a contrastive reinforcement learning approach that trains models to discriminate between coherent image sequences and stories from unrelated images. We extend the Story Reasoning dataset with synthetic negative examples to teach appropriate entity connection behavior. We employ Direct Preference Optimization with a dual-component reward function that promotes grounding and re-identification of entities in real stories while penalizing incorrect entity connections in synthetic contexts. Using this contrastive framework, we fine-tune Qwen Storyteller (based on Qwen2.5-VL 7B). Evaluation shows improvements in grounding mAP from 0.27 to 0.31 (+14.8%), F1 from 0.35 to 0.41 (+17.1%). Pronoun grounding accuracy improved across all pronoun types except ``its'', and cross-frame character and object persistence increased across all frame counts, with entities appearing in 5 or more frames advancing from 29.3% to 33.3% (+13.7%). Well-structured stories, containing the chain-of-thought and grounded story, increased from 79.1% to 97.5% (+23.3%).
>
---
#### [new 020] Synergistic Prompting for Robust Visual Recognition with Missing Modalities
- **分类: cs.CV**

- **简介: 该论文属于视觉识别任务，解决多模态数据缺失导致的性能下降问题。提出SyP框架，通过动态适配器和协同提示策略提升模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.07802v1](http://arxiv.org/pdf/2507.07802v1)**

> **作者:** Zhihui Zhang; Luanyuan Dai; Qika Lin; Yunfeng Diao; Guangyin Jin; Yufei Guo; Jing Zhang; Xiaoshuai Hao
>
> **摘要:** Large-scale multi-modal models have demonstrated remarkable performance across various visual recognition tasks by leveraging extensive paired multi-modal training data. However, in real-world applications, the presence of missing or incomplete modality inputs often leads to significant performance degradation. Recent research has focused on prompt-based strategies to tackle this issue; however, existing methods are hindered by two major limitations: (1) static prompts lack the flexibility to adapt to varying missing-data conditions, and (2) basic prompt-tuning methods struggle to ensure reliable performance when critical modalities are missing.To address these challenges, we propose a novel Synergistic Prompting (SyP) framework for robust visual recognition with missing modalities. The proposed SyP introduces two key innovations: (I) a Dynamic Adapter, which computes adaptive scaling factors to dynamically generate prompts, replacing static parameters for flexible multi-modal adaptation, and (II) a Synergistic Prompting Strategy, which combines static and dynamic prompts to balance information across modalities, ensuring robust reasoning even when key modalities are missing. The proposed SyP achieves significant performance improvements over existing approaches across three widely-used visual recognition datasets, demonstrating robustness under diverse missing rates and conditions. Extensive experiments and ablation studies validate its effectiveness in handling missing modalities, highlighting its superior adaptability and reliability.
>
---
#### [new 021] Multi-Granular Spatio-Temporal Token Merging for Training-Free Acceleration of Video LLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频语言模型加速任务，解决视频LLMs因大量时空token导致的计算效率问题。提出STTM方法，在不训练的情况下合并冗余token，提升推理速度。**

- **链接: [http://arxiv.org/pdf/2507.07990v1](http://arxiv.org/pdf/2507.07990v1)**

> **作者:** Jeongseok Hyun; Sukjun Hwang; Su Ho Han; Taeoh Kim; Inwoong Lee; Dongyoon Wee; Joon-Young Lee; Seon Joo Kim; Minho Shim
>
> **备注:** Accepted at ICCV2025; Project page: https://www.jshyun.me/projects/sttm
>
> **摘要:** Video large language models (LLMs) achieve strong video understanding by leveraging a large number of spatio-temporal tokens, but suffer from quadratic computational scaling with token count. To address this, we propose a training-free spatio-temporal token merging method, named STTM. Our key insight is to exploit local spatial and temporal redundancy in video data which has been overlooked in prior work. STTM first transforms each frame into multi-granular spatial tokens using a coarse-to-fine search over a quadtree structure, then performs directed pairwise merging across the temporal dimension. This decomposed merging approach outperforms existing token reduction methods across six video QA benchmarks. Notably, STTM achieves a 2$\times$ speed-up with only a 0.5% accuracy drop under a 50% token budget, and a 3$\times$ speed-up with just a 2% drop under a 30% budget. Moreover, STTM is query-agnostic, allowing KV cache reuse across different questions for the same video. The project page is available at https://www.jshyun.me/projects/sttm.
>
---
#### [new 022] Impact of Pretraining Word Co-occurrence on Compositional Generalization in Multimodal Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究预训练词共现对多模态模型组合泛化的影响，旨在解决概念组合如何影响模型性能的问题。通过分析PMI与准确率的关系，验证了共现统计的重要性。**

- **链接: [http://arxiv.org/pdf/2507.08000v1](http://arxiv.org/pdf/2507.08000v1)**

> **作者:** Helen Qu; Sang Michael Xie
>
> **摘要:** CLIP and large multimodal models (LMMs) have better accuracy on examples involving concepts that are highly represented in the training data. However, the role of concept combinations in the training data on compositional generalization is largely unclear -- for instance, how does accuracy vary when a common object appears in an uncommon pairing with another object? In this paper, we investigate how word co-occurrence statistics in the pretraining dataset (a proxy for co-occurrence of visual concepts) impacts CLIP/LMM performance. To disentangle the effects of word co-occurrence frequencies from single-word frequencies, we measure co-occurrence with pointwise mutual information (PMI), which normalizes the joint probability of two words co-occurring by the probability of co-occurring independently. Using synthetically generated images with a variety of concept pairs, we show a strong correlation between PMI in the CLIP pretraining data and zero-shot accuracy in CLIP models trained on LAION-400M (r=0.97 and 14% accuracy gap between images in the top and bottom 5% of PMI values), demonstrating that even accuracy on common concepts is affected by the combination of concepts in the image. Leveraging this finding, we reproduce this effect in natural images by editing them to contain pairs with varying PMI, resulting in a correlation of r=0.75. Finally, we demonstrate that this behavior in CLIP transfers to LMMs built on top of CLIP (r=0.70 for TextVQA, r=0.62 for VQAv2). Our findings highlight the need for algorithms and architectures that improve compositional generalization in multimodal models without scaling the training data combinatorially. Our code is available at https://github.com/helenqu/multimodal-pretraining-pmi.
>
---
#### [new 023] CLIP Won't Learn Object-Attribute Binding from Natural Data and Here is Why
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型研究，解决CLIP模型在对象属性绑定上的不足。通过合成数据实验，发现自然数据特性影响绑定学习，提出数据属性对模型性能的关键作用。**

- **链接: [http://arxiv.org/pdf/2507.07985v1](http://arxiv.org/pdf/2507.07985v1)**

> **作者:** Bijay Gurung; David T. Hoffmann; Thomas Brox
>
> **摘要:** Contrastive vision-language models like CLIP are used for a large variety of applications, such as zero-shot classification or as vision encoder for multi-modal models. Despite their popularity, their representations show major limitations. For instance, CLIP models learn bag-of-words representations and, as a consequence, fail to distinguish whether an image is of "a yellow submarine and a blue bus" or "a blue submarine and a yellow bus". Previous attempts to fix this issue added hard negatives during training or modified the architecture, but failed to resolve the problem in its entirety. We suspect that the missing insights to solve the binding problem for CLIP are hidden in the arguably most important part of learning algorithms: the data. In this work, we fill this gap by rigorously identifying the influence of data properties on CLIP's ability to learn binding using a synthetic dataset. We find that common properties of natural data such as low attribute density, incomplete captions, and the saliency bias, a tendency of human captioners to describe the object that is "most salient" to them have a detrimental effect on binding performance. In contrast to common belief, we find that neither scaling the batch size, i.e., implicitly adding more hard negatives, nor explicitly creating hard negatives enables CLIP to learn reliable binding. Only when the data expresses our identified data properties CLIP learns almost perfect binding.
>
---
#### [new 024] Corvid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning
- **分类: cs.CV**

- **简介: 该论文属于多模态语言模型任务，旨在解决复杂推理能力不足的问题。通过构建Corvid模型和MCoT-Instruct-287K数据集，提升模型的链式推理能力。**

- **链接: [http://arxiv.org/pdf/2507.07424v1](http://arxiv.org/pdf/2507.07424v1)**

> **作者:** Jingjing Jiang; Chao Ma; Xurui Song; Hanwang Zhang; Jun Luo
>
> **备注:** ICCV 2025
>
> **摘要:** Recent advancements in multimodal large language models (MLLMs) have demonstrated exceptional performance in multimodal perception and understanding. However, leading open-source MLLMs exhibit significant limitations in complex and structured reasoning, particularly in tasks requiring deep reasoning for decision-making and problem-solving. In this work, we present Corvid, an MLLM with enhanced chain-of-thought (CoT) reasoning capabilities. Architecturally, Corvid incorporates a hybrid vision encoder for informative visual representation and a meticulously designed connector (GateMixer) to facilitate cross-modal alignment. To enhance Corvid's CoT reasoning capabilities, we introduce MCoT-Instruct-287K, a high-quality multimodal CoT instruction-following dataset, refined and standardized from diverse public reasoning sources. Leveraging this dataset, we fine-tune Corvid with a two-stage CoT-formatted training approach to progressively enhance its step-by-step reasoning abilities. Furthermore, we propose an effective inference-time scaling strategy that enables Corvid to mitigate over-reasoning and under-reasoning through self-verification. Extensive experiments demonstrate that Corvid outperforms existing o1-like MLLMs and state-of-the-art MLLMs with similar parameter scales, with notable strengths in mathematical reasoning and science problem-solving. Project page: https://mm-vl.github.io/corvid.
>
---
#### [new 025] MolCLIP: A Molecular-Auxiliary CLIP Framework for Identifying Drug Mechanism of Action Based on Time-Lapsed Mitochondrial Images
- **分类: cs.CV**

- **简介: 该论文属于药物机制识别任务，解决传统方法忽视细胞时间动态的问题。通过结合显微视频与分子信息，提出MolCLIP框架提升药物识别与机制分析效果。**

- **链接: [http://arxiv.org/pdf/2507.07663v1](http://arxiv.org/pdf/2507.07663v1)**

> **作者:** Fengqian Pang; Chunyue Lei; Hongfei Zhao; Chenghao Liu; Zhiqiang Xing; Huafeng Wang; Chuyang Ye
>
> **摘要:** Drug Mechanism of Action (MoA) mainly investigates how drug molecules interact with cells, which is crucial for drug discovery and clinical application. Recently, deep learning models have been used to recognize MoA by relying on high-content and fluorescence images of cells exposed to various drugs. However, these methods focus on spatial characteristics while overlooking the temporal dynamics of live cells. Time-lapse imaging is more suitable for observing the cell response to drugs. Additionally, drug molecules can trigger cellular dynamic variations related to specific MoA. This indicates that the drug molecule modality may complement the image counterpart. This paper proposes MolCLIP, the first visual language model to combine microscopic cell video- and molecule-modalities. MolCLIP designs a molecule-auxiliary CLIP framework to guide video features in learning the distribution of the molecular latent space. Furthermore, we integrate a metric learning strategy with MolCLIP to optimize the aggregation of video features. Experimental results on the MitoDataset demonstrate that MolCLIP achieves improvements of 51.2% and 20.5% in mAP for drug identification and MoA recognition, respectively.
>
---
#### [new 026] Patient-specific vs Multi-Patient Vision Transformer for Markerless Tumor Motion Forecasting
- **分类: cs.CV**

- **简介: 该论文属于肿瘤运动预测任务，旨在解决质子治疗中精准剂量递送问题。通过对比患者特异性与多患者ViT模型，探索 markerless 方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.07811v1](http://arxiv.org/pdf/2507.07811v1)**

> **作者:** Gauthier Rotsart de Hertaing; Dani Manjah; Benoit Macq
>
> **摘要:** Background: Accurate forecasting of lung tumor motion is essential for precise dose delivery in proton therapy. While current markerless methods mostly rely on deep learning, transformer-based architectures remain unexplored in this domain, despite their proven performance in trajectory forecasting. Purpose: This work introduces a markerless forecasting approach for lung tumor motion using Vision Transformers (ViT). Two training strategies are evaluated under clinically realistic constraints: a patient-specific (PS) approach that learns individualized motion patterns, and a multi-patient (MP) model designed for generalization. The comparison explicitly accounts for the limited number of images that can be generated between planning and treatment sessions. Methods: Digitally reconstructed radiographs (DRRs) derived from planning 4DCT scans of 31 patients were used to train the MP model; a 32nd patient was held out for evaluation. PS models were trained using only the target patient's planning data. Both models used 16 DRRs per input and predicted tumor motion over a 1-second horizon. Performance was assessed using Average Displacement Error (ADE) and Final Displacement Error (FDE), on both planning (T1) and treatment (T2) data. Results: On T1 data, PS models outperformed MP models across all training set sizes, especially with larger datasets (up to 25,000 DRRs, p < 0.05). However, MP models demonstrated stronger robustness to inter-fractional anatomical variability and achieved comparable performance on T2 data without retraining. Conclusions: This is the first study to apply ViT architectures to markerless tumor motion forecasting. While PS models achieve higher precision, MP models offer robust out-of-the-box performance, well-suited for time-constrained clinical settings.
>
---
#### [new 027] Driving by Hybrid Navigation: An Online HD-SD Map Association Framework and Benchmark for Autonomous Vehicles
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的地图关联任务，旨在解决在线HD地图与全局SD地图的关联问题。提出OMA基准和Map Association Transformer框架，提升导航能力。**

- **链接: [http://arxiv.org/pdf/2507.07487v1](http://arxiv.org/pdf/2507.07487v1)**

> **作者:** Jiaxu Wan; Xu Wang; Mengwei Xie; Xinyuan Chang; Xinran Liu; Zheng Pan; Mu Xu; Ding Yuan
>
> **备注:** 23 pages, 10 figures, 9 tables
>
> **摘要:** Autonomous vehicles rely on global standard-definition (SD) maps for road-level route planning and online local high-definition (HD) maps for lane-level navigation. However, recent work concentrates on construct online HD maps, often overlooking the association of global SD maps with online HD maps for hybrid navigation, making challenges in utilizing online HD maps in the real world. Observing the lack of the capability of autonomous vehicles in navigation, we introduce \textbf{O}nline \textbf{M}ap \textbf{A}ssociation, the first benchmark for the association of hybrid navigation-oriented online maps, which enhances the planning capabilities of autonomous vehicles. Based on existing datasets, the OMA contains 480k of roads and 260k of lane paths and provides the corresponding metrics to evaluate the performance of the model. Additionally, we propose a novel framework, named Map Association Transformer, as the baseline method, using path-aware attention and spatial attention mechanisms to enable the understanding of geometric and topological correspondences. The code and dataset can be accessed at https://github.com/WallelWan/OMA-MAT.
>
---
#### [new 028] Towards Continuous Home Cage Monitoring: An Evaluation of Tracking and Identification Strategies for Laboratory Mice
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于动物行为监测任务，旨在解决高密度环境下小鼠个体识别难题。通过开发跟踪与识别算法，实现连续、准确的个体监控。**

- **链接: [http://arxiv.org/pdf/2507.07929v1](http://arxiv.org/pdf/2507.07929v1)**

> **作者:** Juan Pablo Oberhauser; Daniel Grzenda
>
> **摘要:** Continuous, automated monitoring of laboratory mice enables more accurate data collection and improves animal welfare through real-time insights. Researchers can achieve a more dynamic and clinically relevant characterization of disease progression and therapeutic effects by integrating behavioral and physiological monitoring in the home cage. However, providing individual mouse metrics is difficult because of their housing density, similar appearances, high mobility, and frequent interactions. To address these challenges, we develop a real-time identification (ID) algorithm that accurately assigns ID predictions to mice wearing custom ear tags in digital home cages monitored by cameras. Our pipeline consists of three parts: (1) a custom multiple object tracker (MouseTracks) that combines appearance and motion cues from mice; (2) a transformer-based ID classifier (Mouseformer); and (3) a tracklet associator linear program to assign final ID predictions to tracklets (MouseMap). Our models assign an animal ID based on custom ear tags at 30 frames per second with 24/7 cage coverage. We show that our custom tracking and ID pipeline improves tracking efficiency and lowers ID switches across mouse strains and various environmental factors compared to current mouse tracking methods.
>
---
#### [new 029] Sparse-Dense Side-Tuner for efficient Video Temporal Grounding
- **分类: cs.CV**

- **简介: 该论文属于视频时间定位任务，解决传统方法依赖固定特征、适应性差的问题，提出SDST框架提升模型效率与性能。**

- **链接: [http://arxiv.org/pdf/2507.07744v1](http://arxiv.org/pdf/2507.07744v1)**

> **作者:** David Pujol-Perich; Sergio Escalera; Albert Clapés
>
> **摘要:** Video Temporal Grounding (VTG) involves Moment Retrieval (MR) and Highlight Detection (HD) based on textual queries. For this, most methods rely solely on final-layer features of frozen large pre-trained backbones, limiting their adaptability to new domains. While full fine-tuning is often impractical, parameter-efficient fine-tuning -- and particularly side-tuning (ST) -- has emerged as an effective alternative. However, prior ST approaches this problem from a frame-level refinement perspective, overlooking the inherent sparse nature of MR. To address this, we propose the Sparse-Dense Side-Tuner (SDST), the first anchor-free ST architecture for VTG. We also introduce the Reference-based Deformable Self-Attention, a novel mechanism that enhances the context modeling of the deformable attention -- a key limitation of existing anchor-free methods. Additionally, we present the first effective integration of InternVideo2 backbone into an ST framework, showing its profound implications in performance. Overall, our method significantly improves existing ST methods, achieving highly competitive or SOTA results on QVHighlights, TACoS, and Charades-STA, while reducing up to a 73% the parameter count w.r.t. the existing SOTA methods. The code is publicly accessible at https://github.com/davidpujol/SDST.
>
---
#### [new 030] Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Methodology
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉推理任务，旨在解决缺乏评估基准的问题。提出TreeBench基准和TreeVGR方法，提升模型的可解释性和推理能力。**

- **链接: [http://arxiv.org/pdf/2507.07999v1](http://arxiv.org/pdf/2507.07999v1)**

> **作者:** Haochen Wang; Xiangtai Li; Zilong Huang; Anran Wang; Jiacong Wang; Tao Zhang; Jiani Zheng; Sule Bai; Zijian Kang; Jiashi Feng; Zhuochen Wang; Zhaoxiang Zhang
>
> **摘要:** Models like OpenAI-o3 pioneer visual grounded reasoning by dynamically referencing visual regions, just like human "thinking with images". However, no benchmark exists to evaluate these capabilities holistically. To bridge this gap, we propose TreeBench (Traceable Evidence Evaluation Benchmark), a diagnostic benchmark built on three principles: (1) focused visual perception of subtle targets in complex scenes, (2) traceable evidence via bounding box evaluation, and (3) second-order reasoning to test object interactions and spatial hierarchies beyond simple object localization. Prioritizing images with dense objects, we initially sample 1K high-quality images from SA-1B, and incorporate eight LMM experts to manually annotate questions, candidate options, and answers for each image. After three stages of quality control, TreeBench consists of 405 challenging visual question-answering pairs, even the most advanced models struggle with this benchmark, where none of them reach 60% accuracy, e.g., OpenAI-o3 scores only 54.87. Furthermore, we introduce TreeVGR (Traceable Evidence Enhanced Visual Grounded Reasoning), a training paradigm to supervise localization and reasoning jointly with reinforcement learning, enabling accurate localizations and explainable reasoning pathways. Initialized from Qwen2.5-VL-7B, it improves V* Bench (+16.8), MME-RealWorld (+12.6), and TreeBench (+13.4), proving traceability is key to advancing vision-grounded reasoning. The code is available at https://github.com/Haochen-Wang409/TreeVGR.
>
---
#### [new 031] OST-Bench: Evaluating the Capabilities of MLLMs in Online Spatio-temporal Scene Understanding
- **分类: cs.CV**

- **简介: 该论文属于多模态大模型的在线时空场景理解任务，旨在评估模型在动态环境中的空间和时间推理能力，通过构建OST-Bench数据集进行实验分析。**

- **链接: [http://arxiv.org/pdf/2507.07984v1](http://arxiv.org/pdf/2507.07984v1)**

> **作者:** JingLi Lin; Chenming Zhu; Runsen Xu; Xiaohan Mao; Xihui Liu; Tai Wang; Jiangmiao Pang
>
> **备注:** 28 pages, a benchmark designed to evaluate Online Spatio-Temporal understanding from the perspective of an agent actively exploring a scene. Project Page: https://rbler1234.github.io/OSTBench.github.io/
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have shown remarkable capabilities in integrating vision and language for complex reasoning. While most existing benchmarks evaluate models under offline settings with a fixed set of pre-recorded inputs, we introduce OST-Bench, a benchmark designed to evaluate Online Spatio-Temporal understanding from the perspective of an agent actively exploring a scene. The Online aspect emphasizes the need to process and reason over incrementally acquired observations, while the Spatio-Temporal component requires integrating current visual inputs with historical memory to support dynamic spatial reasoning. OST-Bench better reflects the challenges of real-world embodied perception. Built on an efficient data collection pipeline, OST-Bench consists of 1.4k scenes and 10k question-answer pairs collected from ScanNet, Matterport3D, and ARKitScenes. We evaluate several leading MLLMs on OST-Bench and observe that they fall short on tasks requiring complex spatio-temporal reasoning. Under the online setting, their accuracy declines as the exploration horizon extends and the memory grows. Through further experimental analysis, we identify common error patterns across models and find that both complex clue-based spatial reasoning demands and long-term memory retrieval requirements significantly drop model performance along two separate axes, highlighting the core challenges that must be addressed to improve online embodied reasoning. To foster further research and development in the field, our codes, dataset, and benchmark are available. Our project page is: https://rbler1234.github.io/OSTBench.github.io/
>
---
#### [new 032] Diffusion-Guided Knowledge Distillation for Weakly-Supervised Low-Light Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于弱监督低光语义分割任务，旨在解决低光环境下图像质量差和监督信息不足导致的分割性能下降问题。通过结合扩散引导知识蒸馏和深度引导特征融合，提升模型的分割效果。**

- **链接: [http://arxiv.org/pdf/2507.07578v1](http://arxiv.org/pdf/2507.07578v1)**

> **作者:** Chunyan Wang; Dong Zhang; Jinhui Tang
>
> **摘要:** Weakly-supervised semantic segmentation aims to assign category labels to each pixel using weak annotations, significantly reducing manual annotation costs. Although existing methods have achieved remarkable progress in well-lit scenarios, their performance significantly degrades in low-light environments due to two fundamental limitations: severe image quality degradation (e.g., low contrast, noise, and color distortion) and the inherent constraints of weak supervision. These factors collectively lead to unreliable class activation maps and semantically ambiguous pseudo-labels, ultimately compromising the model's ability to learn discriminative feature representations. To address these problems, we propose Diffusion-Guided Knowledge Distillation for Weakly-Supervised Low-light Semantic Segmentation (DGKD-WLSS), a novel framework that synergistically combines Diffusion-Guided Knowledge Distillation (DGKD) with Depth-Guided Feature Fusion (DGF2). DGKD aligns normal-light and low-light features via diffusion-based denoising and knowledge distillation, while DGF2 integrates depth maps as illumination-invariant geometric priors to enhance structural feature learning. Extensive experiments demonstrate the effectiveness of DGKD-WLSS, which achieves state-of-the-art performance in weakly supervised semantic segmentation tasks under low-light conditions. The source codes have been released at:https://github.com/ChunyanWang1/DGKD-WLSS.
>
---
#### [new 033] Scalable and Realistic Virtual Try-on Application for Foundation Makeup with Kubelka-Munk Theory
- **分类: cs.CV; I.4.9**

- **简介: 该论文属于虚拟试妆任务，解决基础化妆品与皮肤色调融合的准确合成问题，提出基于Kubelka-Munk理论的快速图像合成方法。**

- **链接: [http://arxiv.org/pdf/2507.07333v1](http://arxiv.org/pdf/2507.07333v1)**

> **作者:** Hui Pang; Sunil Hadap; Violetta Shevchenko; Rahul Suresh; Amin Banitalebi-Dehkordi
>
> **备注:** Presented at the workshop Three questions about virtual try-on at CVPR 2025
>
> **摘要:** Augmented reality is revolutionizing beauty industry with virtual try-on (VTO) applications, which empowers users to try a wide variety of products using their phones without the hassle of physically putting on real products. A critical technical challenge in foundation VTO applications is the accurate synthesis of foundation-skin tone color blending while maintaining the scalability of the method across diverse product ranges. In this work, we propose a novel method to approximate well-established Kubelka-Munk (KM) theory for faster image synthesis while preserving foundation-skin tone color blending realism. Additionally, we build a scalable end-to-end framework for realistic foundation makeup VTO solely depending on the product information available on e-commerce sites. We validate our method using real-world makeup images, demonstrating that our framework outperforms other techniques.
>
---
#### [new 034] Compressive Imaging Reconstruction via Tensor Decomposed Multi-Resolution Grid Encoding
- **分类: cs.CV**

- **简介: 该论文属于压缩成像重建任务，解决高维图像从低维测量中恢复的问题。提出GridTD框架，结合张量分解与多分辨率网格编码，实现高效准确的重建。**

- **链接: [http://arxiv.org/pdf/2507.07707v1](http://arxiv.org/pdf/2507.07707v1)**

> **作者:** Zhenyu Jin; Yisi Luo; Xile Zhao; Deyu Meng
>
> **摘要:** Compressive imaging (CI) reconstruction, such as snapshot compressive imaging (SCI) and compressive sensing magnetic resonance imaging (MRI), aims to recover high-dimensional images from low-dimensional compressed measurements. This process critically relies on learning an accurate representation of the underlying high-dimensional image. However, existing unsupervised representations may struggle to achieve a desired balance between representation ability and efficiency. To overcome this limitation, we propose Tensor Decomposed multi-resolution Grid encoding (GridTD), an unsupervised continuous representation framework for CI reconstruction. GridTD optimizes a lightweight neural network and the input tensor decomposition model whose parameters are learned via multi-resolution hash grid encoding. It inherently enjoys the hierarchical modeling ability of multi-resolution grid encoding and the compactness of tensor decomposition, enabling effective and efficient reconstruction of high-dimensional images. Theoretical analyses for the algorithm's Lipschitz property, generalization error bound, and fixed-point convergence reveal the intrinsic superiority of GridTD as compared with existing continuous representation models. Extensive experiments across diverse CI tasks, including video SCI, spectral SCI, and compressive dynamic MRI reconstruction, consistently demonstrate the superiority of GridTD over existing methods, positioning GridTD as a versatile and state-of-the-art CI reconstruction method.
>
---
#### [new 035] Bluish Veil Detection and Lesion Classification using Custom Deep Learnable Layers with Explainable Artificial Intelligence (XAI)
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于皮肤病变分类任务，旨在解决BWV检测问题。通过自定义深度学习层和XAI技术提升 melanoma 早期诊断准确性。**

- **链接: [http://arxiv.org/pdf/2507.07453v1](http://arxiv.org/pdf/2507.07453v1)**

> **作者:** M. A. Rasel; Sameem Abdul Kareem; Zhenli Kwan; Shin Shen Yong; Unaizah Obaidellah
>
> **备注:** Accepted version. Published in Computers in Biology and Medicine, 14 June 2024. DOI: 10.1016/j.compbiomed.2024.108758
>
> **摘要:** Melanoma, one of the deadliest types of skin cancer, accounts for thousands of fatalities globally. The bluish, blue-whitish, or blue-white veil (BWV) is a critical feature for diagnosing melanoma, yet research into detecting BWV in dermatological images is limited. This study utilizes a non-annotated skin lesion dataset, which is converted into an annotated dataset using a proposed imaging algorithm based on color threshold techniques on lesion patches and color palettes. A Deep Convolutional Neural Network (DCNN) is designed and trained separately on three individual and combined dermoscopic datasets, using custom layers instead of standard activation function layers. The model is developed to categorize skin lesions based on the presence of BWV. The proposed DCNN demonstrates superior performance compared to conventional BWV detection models across different datasets. The model achieves a testing accuracy of 85.71% on the augmented PH2 dataset, 95.00% on the augmented ISIC archive dataset, 95.05% on the combined augmented (PH2+ISIC archive) dataset, and 90.00% on the Derm7pt dataset. An explainable artificial intelligence (XAI) algorithm is subsequently applied to interpret the DCNN's decision-making process regarding BWV detection. The proposed approach, coupled with XAI, significantly improves the detection of BWV in skin lesions, outperforming existing models and providing a robust tool for early melanoma diagnosis.
>
---
#### [new 036] NexViTAD: Few-shot Unsupervised Cross-Domain Defect Detection via Vision Foundation Models and Multi-Task Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于工业缺陷检测任务，解决跨域异常检测问题。通过视觉基础模型和多任务学习，提出NexViTAD框架，提升模型泛化能力与检测精度。**

- **链接: [http://arxiv.org/pdf/2507.07579v1](http://arxiv.org/pdf/2507.07579v1)**

> **作者:** Tianwei Mu; Feiyu Duan; Bo Zhou; Dan Xue; Manhong Huang
>
> **摘要:** This paper presents a novel few-shot cross-domain anomaly detection framework, Nexus Vision Transformer for Anomaly Detection (NexViTAD), based on vision foundation models, which effectively addresses domain-shift challenges in industrial anomaly detection through innovative shared subspace projection mechanisms and multi-task learning (MTL) module. The main innovations include: (1) a hierarchical adapter module that adaptively fuses complementary features from Hiera and DINO-v2 pre-trained models, constructing more robust feature representations; (2) a shared subspace projection strategy that enables effective cross-domain knowledge transfer through bottleneck dimension constraints and skip connection mechanisms; (3) a MTL Decoder architecture supports simultaneous processing of multiple source domains, significantly enhancing model generalization capabilities; (4) an anomaly score inference method based on Sinkhorn-K-means clustering, combined with Gaussian filtering and adaptive threshold processing for precise pixel level. Valuated on the MVTec AD dataset, NexViTAD delivers state-of-the-art performance with an AUC of 97.5%, AP of 70.4%, and PRO of 95.2% in the target domains, surpassing other recent models, marking a transformative advance in cross-domain defect detection.
>
---
#### [new 037] Image Can Bring Your Memory Back: A Novel Multi-Modal Guided Attack against Image Generation Model Unlearning
- **分类: cs.CV; cs.CR; cs.LG**

- **简介: 该论文属于图像生成模型安全任务，旨在解决模型遗忘机制的脆弱性问题。通过设计多模态对抗攻击框架Recall，有效破坏遗忘后的模型性能。**

- **链接: [http://arxiv.org/pdf/2507.07139v1](http://arxiv.org/pdf/2507.07139v1)**

> **作者:** Renyang Liu; Guanlin Li; Tianwei Zhang; See-Kiong Ng
>
> **摘要:** Recent advances in image generation models (IGMs), particularly diffusion-based architectures such as Stable Diffusion (SD), have markedly enhanced the quality and diversity of AI-generated visual content. However, their generative capability has also raised significant ethical, legal, and societal concerns, including the potential to produce harmful, misleading, or copyright-infringing content. To mitigate these concerns, machine unlearning (MU) emerges as a promising solution by selectively removing undesirable concepts from pretrained models. Nevertheless, the robustness and effectiveness of existing unlearning techniques remain largely unexplored, particularly in the presence of multi-modal adversarial inputs. To bridge this gap, we propose Recall, a novel adversarial framework explicitly designed to compromise the robustness of unlearned IGMs. Unlike existing approaches that predominantly rely on adversarial text prompts, Recall exploits the intrinsic multi-modal conditioning capabilities of diffusion models by efficiently optimizing adversarial image prompts with guidance from a single semantically relevant reference image. Extensive experiments across ten state-of-the-art unlearning methods and diverse tasks show that Recall consistently outperforms existing baselines in terms of adversarial effectiveness, computational efficiency, and semantic fidelity with the original textual prompt. These findings reveal critical vulnerabilities in current unlearning mechanisms and underscore the need for more robust solutions to ensure the safety and reliability of generative models. Code and data are publicly available at \textcolor{blue}{https://github.com/ryliu68/RECALL}.
>
---
#### [new 038] KeyRe-ID: Keypoint-Guided Person Re-Identification using Part-Aware Representation in Videos
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频行人重识别任务，旨在提升跨摄像头追踪的准确性。通过关键点引导的全局与局部分支结构，增强时空特征表示。**

- **链接: [http://arxiv.org/pdf/2507.07393v1](http://arxiv.org/pdf/2507.07393v1)**

> **作者:** Jinseong Kim; Junghoon Song; Gyeongseon Baek; Byeongjoon Noh
>
> **备注:** 10 pages, 2 figures,
>
> **摘要:** We propose \textbf{KeyRe-ID}, a keypoint-guided video-based person re-identification framework consisting of global and local branches that leverage human keypoints for enhanced spatiotemporal representation learning. The global branch captures holistic identity semantics through Transformer-based temporal aggregation, while the local branch dynamically segments body regions based on keypoints to generate fine-grained, part-aware features. Extensive experiments on MARS and iLIDS-VID benchmarks demonstrate state-of-the-art performance, achieving 91.73\% mAP and 97.32\% Rank-1 accuracy on MARS, and 96.00\% Rank-1 and 100.0\% Rank-5 accuracy on iLIDS-VID. The code for this work will be publicly available on GitHub upon publication.
>
---
#### [new 039] Objectomaly: Objectness-Aware Refinement for OoD Segmentation with Structural Consistency and Boundary Precision
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于OoD分割任务，解决边界不精确和异常评分不一致问题。提出Objectomaly框架，通过三阶段优化提升分割精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.07460v1](http://arxiv.org/pdf/2507.07460v1)**

> **作者:** Jeonghoon Song; Sunghun Kim; Jaegyun Im; Byeongjoon Noh
>
> **摘要:** Out-of-Distribution (OoD) segmentation is critical for safety-sensitive applications like autonomous driving. However, existing mask-based methods often suffer from boundary imprecision, inconsistent anomaly scores within objects, and false positives from background noise. We propose \textbf{\textit{Objectomaly}}, an objectness-aware refinement framework that incorporates object-level priors. Objectomaly consists of three stages: (1) Coarse Anomaly Scoring (CAS) using an existing OoD backbone, (2) Objectness-Aware Score Calibration (OASC) leveraging SAM-generated instance masks for object-level score normalization, and (3) Meticulous Boundary Precision (MBP) applying Laplacian filtering and Gaussian smoothing for contour refinement. Objectomaly achieves state-of-the-art performance on key OoD segmentation benchmarks, including SMIYC AnomalyTrack/ObstacleTrack and RoadAnomaly, improving both pixel-level (AuPRC up to 96.99, FPR$_{95}$ down to 0.07) and component-level (F1$-$score up to 83.44) metrics. Ablation studies and qualitative results on real-world driving videos further validate the robustness and generalizability of our method. Code will be released upon publication.
>
---
#### [new 040] Action Unit Enhance Dynamic Facial Expression Recognition
- **分类: cs.CV**

- **简介: 该论文属于动态面部表情识别任务，旨在解决模型效果不足和数据标签不平衡问题。通过引入AU增强机制和重新设计损失函数，提升识别性能。**

- **链接: [http://arxiv.org/pdf/2507.07678v1](http://arxiv.org/pdf/2507.07678v1)**

> **作者:** Feng Liu; Lingna Gu; Chen Shi; Xiaolan Fu
>
> **摘要:** Dynamic Facial Expression Recognition(DFER) is a rapidly evolving field of research that focuses on the recognition of time-series facial expressions. While previous research on DFER has concentrated on feature learning from a deep learning perspective, we put forward an AU-enhanced Dynamic Facial Expression Recognition architecture, namely AU-DFER, that incorporates AU-expression knowledge to enhance the effectiveness of deep learning modeling. In particular, the contribution of the Action Units(AUs) to different expressions is quantified, and a weight matrix is designed to incorporate a priori knowledge. Subsequently, the knowledge is integrated with the learning outcomes of a conventional deep learning network through the introduction of AU loss. The design is incorporated into the existing optimal model for dynamic expression recognition for the purpose of validation. Experiments are conducted on three recent mainstream open-source approaches to DFER on the principal datasets in this field. The results demonstrate that the proposed architecture outperforms the state-of-the-art(SOTA) methods without the need for additional arithmetic and generally produces improved results. Furthermore, we investigate the potential of AU loss function redesign to address data label imbalance issues in established dynamic expression datasets. To the best of our knowledge, this is the first attempt to integrate quantified AU-expression knowledge into various DFER models. We also devise strategies to tackle label imbalance, or minor class problems. Our findings suggest that employing a diverse strategy of loss function design can enhance the effectiveness of DFER. This underscores the criticality of addressing data imbalance challenges in mainstream datasets within this domain. The source code is available at https://github.com/Cross-Innovation-Lab/AU-DFER.
>
---
#### [new 041] Beyond the Linear Separability Ceiling
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型的线性可分性瓶颈问题，属于模型分析任务。通过引入LSC指标，发现推理路径缺陷是主要原因，并提出针对性调整方案。**

- **链接: [http://arxiv.org/pdf/2507.07574v1](http://arxiv.org/pdf/2507.07574v1)**

> **作者:** Enrico Vompa; Tanel Tammet; Mohit Vaishnav
>
> **摘要:** Most state-of-the-art Visual-Language Models (VLMs) are seemingly limited by the linear separabilty of their visual embeddings on abstract reasoning tasks. This work investigates this "linear reasoning bottleneck" by introducing the Linear Separability Ceiling (LSC), the performance of a simple linear classifier on a VLM's visual embeddings. We find this bottleneck is widespread and stems not from poor perception, but from failures in the language model's reasoning pathways. We demonstrate this is a solvable alignment issue. The required intervention, however, is task-dependent: activating existing pathways suffices for semantic concepts, while complex relational reasoning requires adapting core model weights. Using postfix tuning as a methodological control, we find strong evidence for powerful, dormant reasoning pathways within VLMs. However, for complex relational tasks requiring deeper adaptation, explicitly improving representation quality causes the model to fail on new prompt formats despite its embeddings remaining well separated. Ultimately, this work provides a new lens for VLM analysis, showing that robust reasoning is a matter of targeted alignment, not simply improved representation learning.
>
---
#### [new 042] Robust Multimodal Large Language Models Against Modality Conflict
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态任务，解决MLLMs在模态冲突下的幻觉问题。通过构建数据集并提出三种方法缓解冲突导致的错误。**

- **链接: [http://arxiv.org/pdf/2507.07151v1](http://arxiv.org/pdf/2507.07151v1)**

> **作者:** Zongmeng Zhang; Wengang Zhou; Jie Zhao; Houqiang Li
>
> **备注:** ICML 2025
>
> **摘要:** Despite the impressive capabilities of multimodal large language models (MLLMs) in vision-language tasks, they are prone to hallucinations in real-world scenarios. This paper investigates the hallucination phenomenon in MLLMs from the perspective of modality conflict. Unlike existing works focusing on the conflicts between model responses and inputs, we study the inherent conflicts in inputs from different modalities that place MLLMs in a dilemma and directly lead to hallucinations. We formally define the modality conflict and construct a dataset named Multimodal Modality Conflict (MMMC) to simulate this phenomenon in vision-language tasks. Three methods based on prompt engineering, supervised fine-tuning, and reinforcement learning are proposed to alleviate the hallucination caused by modality conflict. Extensive experiments are conducted on the MMMC dataset to analyze the merits and demerits of these methods. Our results show that the reinforcement learning method achieves the best performance in mitigating the hallucination under modality conflict, while the supervised fine-tuning method shows promising and stable performance. Our work sheds light on the unnoticed modality conflict that leads to hallucinations and provides more insights into the robustness of MLLMs.
>
---
#### [new 043] Stable-Hair v2: Real-World Hair Transfer via Multiple-View Diffusion Model
- **分类: cs.CV**

- **简介: 该论文属于多视角头发迁移任务，旨在解决真实场景下头发生成的一致性问题。提出Stable-Hair v2框架，结合多视角扩散模型与姿态嵌入，实现高质量、视图一致的头发迁移。**

- **链接: [http://arxiv.org/pdf/2507.07591v1](http://arxiv.org/pdf/2507.07591v1)**

> **作者:** Kuiyuan Sun; Yuxuan Zhang; Jichao Zhang; Jiaming Liu; Wei Wang; Niculae Sebe; Yao Zhao
>
> **备注:** 14 pages
>
> **摘要:** While diffusion-based methods have shown impressive capabilities in capturing diverse and complex hairstyles, their ability to generate consistent and high-quality multi-view outputs -- crucial for real-world applications such as digital humans and virtual avatars -- remains underexplored. In this paper, we propose Stable-Hair v2, a novel diffusion-based multi-view hair transfer framework. To the best of our knowledge, this is the first work to leverage multi-view diffusion models for robust, high-fidelity, and view-consistent hair transfer across multiple perspectives. We introduce a comprehensive multi-view training data generation pipeline comprising a diffusion-based Bald Converter, a data-augment inpainting model, and a face-finetuned multi-view diffusion model to generate high-quality triplet data, including bald images, reference hairstyles, and view-aligned source-bald pairs. Our multi-view hair transfer model integrates polar-azimuth embeddings for pose conditioning and temporal attention layers to ensure smooth transitions between views. To optimize this model, we design a novel multi-stage training strategy consisting of pose-controllable latent IdentityNet training, hair extractor training, and temporal attention training. Extensive experiments demonstrate that our method accurately transfers detailed and realistic hairstyles to source subjects while achieving seamless and consistent results across views, significantly outperforming existing methods and establishing a new benchmark in multi-view hair transfer. Code is publicly available at https://github.com/sunkymepro/StableHairV2.
>
---
#### [new 044] SURPRISE3D: A Dataset for Spatial Understanding and Reasoning in Complex 3D Scenes
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D视觉语言任务，旨在解决空间推理不足的问题。构建了SURPRISE3D数据集，用于评估语言引导的空间分割，推动更精准的AI交互与机器人规划。**

- **链接: [http://arxiv.org/pdf/2507.07781v1](http://arxiv.org/pdf/2507.07781v1)**

> **作者:** Jiaxin Huang; Ziwen Li; Hanlve Zhang; Runnan Chen; Xiao He; Yandong Guo; Wenping Wang; Tongliang Liu; Mingming Gong
>
> **摘要:** The integration of language and 3D perception is critical for embodied AI and robotic systems to perceive, understand, and interact with the physical world. Spatial reasoning, a key capability for understanding spatial relationships between objects, remains underexplored in current 3D vision-language research. Existing datasets often mix semantic cues (e.g., object name) with spatial context, leading models to rely on superficial shortcuts rather than genuinely interpreting spatial relationships. To address this gap, we introduce S\textsc{urprise}3D, a novel dataset designed to evaluate language-guided spatial reasoning segmentation in complex 3D scenes. S\textsc{urprise}3D consists of more than 200k vision language pairs across 900+ detailed indoor scenes from ScanNet++ v2, including more than 2.8k unique object classes. The dataset contains 89k+ human-annotated spatial queries deliberately crafted without object name, thereby mitigating shortcut biases in spatial understanding. These queries comprehensively cover various spatial reasoning skills, such as relative position, narrative perspective, parametric perspective, and absolute distance reasoning. Initial benchmarks demonstrate significant challenges for current state-of-the-art expert 3D visual grounding methods and 3D-LLMs, underscoring the necessity of our dataset and the accompanying 3D Spatial Reasoning Segmentation (3D-SRS) benchmark suite. S\textsc{urprise}3D and 3D-SRS aim to facilitate advancements in spatially aware AI, paving the way for effective embodied interaction and robotic planning. The code and datasets can be found in https://github.com/liziwennba/SUPRISE.
>
---
#### [new 045] Deep Learning based 3D Volume Correlation for Additive Manufacturing Using High-Resolution Industrial X-ray Computed Tomography
- **分类: cs.CV**

- **简介: 该论文属于三维图像配准任务，旨在解决AM中CAD与XCT数据的精确对齐问题。通过深度学习方法提升配准精度与效率。**

- **链接: [http://arxiv.org/pdf/2507.07757v1](http://arxiv.org/pdf/2507.07757v1)**

> **作者:** Keerthana Chand; Tobias Fritsch; Bardia Hejazi; Konstantin Poka; Giovanni Bruno
>
> **摘要:** Quality control in additive manufacturing (AM) is vital for industrial applications in areas such as the automotive, medical and aerospace sectors. Geometric inaccuracies caused by shrinkage and deformations can compromise the life and performance of additively manufactured components. Such deviations can be quantified using Digital Volume Correlation (DVC), which compares the computer-aided design (CAD) model with the X-ray Computed Tomography (XCT) geometry of the components produced. However, accurate registration between the two modalities is challenging due to the absence of a ground truth or reference deformation field. In addition, the extremely large data size of high-resolution XCT volumes makes computation difficult. In this work, we present a deep learning-based approach for estimating voxel-wise deformations between CAD and XCT volumes. Our method uses a dynamic patch-based processing strategy to handle high-resolution volumes. In addition to the Dice Score, we introduce a Binary Difference Map (BDM) that quantifies voxel-wise mismatches between binarized CAD and XCT volumes to evaluate the accuracy of the registration. Our approach shows a 9.2\% improvement in the Dice Score and a 9.9\% improvement in the voxel match rate compared to classic DVC methods, while reducing the interaction time from days to minutes. This work sets the foundation for deep learning-based DVC methods to generate compensation meshes that can then be used in closed-loop correlations during the AM production process. Such a system would be of great interest to industries since the manufacturing process will become more reliable and efficient, saving time and material.
>
---
#### [new 046] Scaling RL to Long Videos
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频理解任务，旨在解决长视频推理问题。通过构建数据集、两阶段训练框架和高效训练系统，提升视觉语言模型在长视频上的表现。**

- **链接: [http://arxiv.org/pdf/2507.07966v1](http://arxiv.org/pdf/2507.07966v1)**

> **作者:** Yukang Chen; Wei Huang; Baifeng Shi; Qinghao Hu; Hanrong Ye; Ligeng Zhu; Zhijian Liu; Pavlo Molchanov; Jan Kautz; Xiaojuan Qi; Sifei Liu; Hongxu Yin; Yao Lu; Song Han
>
> **备注:** Code and models are available at https://github.com/NVlabs/Long-RL
>
> **摘要:** We introduce a full-stack framework that scales up reasoning in vision-language models (VLMs) to long videos, leveraging reinforcement learning. We address the unique challenges of long video reasoning by integrating three critical components: (1) a large-scale dataset, LongVideo-Reason, comprising 52K long video QA pairs with high-quality reasoning annotations across diverse domains such as sports, games, and vlogs; (2) a two-stage training pipeline that extends VLMs with chain-of-thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL); and (3) a training infrastructure for long video RL, named Multi-modal Reinforcement Sequence Parallelism (MR-SP), which incorporates sequence parallelism and a vLLM-based engine tailored for long video, using cached video embeddings for efficient rollout and prefilling. In experiments, LongVILA-R1-7B achieves strong performance on long video QA benchmarks such as VideoMME. It also outperforms Video-R1-7B and even matches Gemini-1.5-Pro across temporal reasoning, goal and purpose reasoning, spatial reasoning, and plot reasoning on our LongVideo-Reason-eval benchmark. Notably, our MR-SP system achieves up to 2.1x speedup on long video RL training. LongVILA-R1 demonstrates consistent performance gains as the number of input video frames scales. LongVILA-R1 marks a firm step towards long video reasoning in VLMs. In addition, we release our training system for public availability that supports RL training on various modalities (video, text, and audio), various models (VILA and Qwen series), and even image and video generation models. On a single A100 node (8 GPUs), it supports RL training on hour-long videos (e.g., 3,600 frames / around 256k tokens).
>
---
#### [new 047] Seg-Wild: Interactive Segmentation based on 3D Gaussian Splatting for Unconstrained Image Collections
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务，解决无约束图像集合中的分割难题。通过3D高斯点云和交互式方法提升分割精度与场景重建质量。**

- **链接: [http://arxiv.org/pdf/2507.07395v1](http://arxiv.org/pdf/2507.07395v1)**

> **作者:** Yongtang Bao; Chengjie Tang; Yuze Wang; Haojie Li
>
> **摘要:** Reconstructing and segmenting scenes from unconstrained photo collections obtained from the Internet is a novel but challenging task. Unconstrained photo collections are easier to get than well-captured photo collections. These unconstrained images suffer from inconsistent lighting and transient occlusions, which makes segmentation challenging. Previous segmentation methods cannot address transient occlusions or accurately restore the scene's lighting conditions. Therefore, we propose Seg-Wild, an interactive segmentation method based on 3D Gaussian Splatting for unconstrained image collections, suitable for in-the-wild scenes. We integrate multi-dimensional feature embeddings for each 3D Gaussian and calculate the feature similarity between the feature embeddings and the segmentation target to achieve interactive segmentation in the 3D scene. Additionally, we introduce the Spiky 3D Gaussian Cutter (SGC) to smooth abnormal 3D Gaussians. We project the 3D Gaussians onto a 2D plane and calculate the ratio of 3D Gaussians that need to be cut using the SAM mask. We also designed a benchmark to evaluate segmentation quality in in-the-wild scenes. Experimental results demonstrate that compared to previous methods, Seg-Wild achieves better segmentation results and reconstruction quality. Our code will be available at https://github.com/Sugar0725/Seg-Wild.
>
---
#### [new 048] SpatialViz-Bench: Automatically Generated Spatial Visualization Reasoning Tasks for MLLMs
- **分类: cs.CV; cs.CL; cs.HC**

- **简介: 该论文属于多模态语言模型任务，旨在解决空间可视化评估不足的问题。提出SpatialViz-Bench基准，包含12个任务，评估模型的空间推理能力。**

- **链接: [http://arxiv.org/pdf/2507.07610v1](http://arxiv.org/pdf/2507.07610v1)**

> **作者:** Siting Wang; Luoyang Sun; Cheng Deng; Kun Shao; Minnan Pei; Zheng Tian; Haifeng Zhang; Jun Wang
>
> **摘要:** Humans can directly imagine and manipulate visual images in their minds, a capability known as spatial visualization. While multi-modal Large Language Models (MLLMs) support imagination-based reasoning, spatial visualization remains insufficiently evaluated, typically embedded within broader mathematical and logical assessments. Existing evaluations often rely on IQ tests or math competitions that may overlap with training data, compromising assessment reliability. To this end, we introduce SpatialViz-Bench, a comprehensive multi-modal benchmark for spatial visualization with 12 tasks across 4 sub-abilities, comprising 1,180 automatically generated problems. Our evaluation of 33 state-of-the-art MLLMs not only reveals wide performance variations and demonstrates the benchmark's strong discriminative power, but also uncovers counter-intuitive findings: models exhibit unexpected behaviors by showing difficulty perception that misaligns with human intuition, displaying dramatic 2D-to-3D performance cliffs, and defaulting to formula derivation despite spatial tasks requiring visualization alone. SpatialVizBench empirically demonstrates that state-of-the-art MLLMs continue to exhibit deficiencies in spatial visualization tasks, thereby addressing a significant lacuna in the field. The benchmark is publicly available.
>
---
#### [new 049] T-GVC: Trajectory-Guided Generative Video Coding at Ultra-Low Bitrates
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于视频编码任务，旨在解决超低码率下语义准确重建问题。提出T-GVC框架，通过轨迹引导提升运动细节和真实性。**

- **链接: [http://arxiv.org/pdf/2507.07633v1](http://arxiv.org/pdf/2507.07633v1)**

> **作者:** Zhitao Wang; Hengyu Man; Wenrui Li; Xingtao Wang; Xiaopeng Fan; Debin Zhao
>
> **摘要:** Recent advances in video generation techniques have given rise to an emerging paradigm of generative video coding, aiming to achieve semantically accurate reconstructions in Ultra-Low Bitrate (ULB) scenarios by leveraging strong generative priors. However, most existing methods are limited by domain specificity (e.g., facial or human videos) or an excessive dependence on high-level text guidance, which often fails to capture motion details and results in unrealistic reconstructions. To address these challenges, we propose a Trajectory-Guided Generative Video Coding framework (dubbed T-GVC). T-GVC employs a semantic-aware sparse motion sampling pipeline to effectively bridge low-level motion tracking with high-level semantic understanding by extracting pixel-wise motion as sparse trajectory points based on their semantic importance, not only significantly reducing the bitrate but also preserving critical temporal semantic information. In addition, by incorporating trajectory-aligned loss constraints into diffusion processes, we introduce a training-free latent space guidance mechanism to ensure physically plausible motion patterns without sacrificing the inherent capabilities of generative models. Experimental results demonstrate that our framework outperforms both traditional codecs and state-of-the-art end-to-end video compression methods under ULB conditions. Furthermore, additional experiments confirm that our approach achieves more precise motion control than existing text-guided methods, paving the way for a novel direction of generative video coding guided by geometric motion modeling.
>
---
#### [new 050] Single-Step Latent Diffusion for Underwater Image Restoration
- **分类: cs.CV**

- **简介: 该论文属于 underwater image restoration 任务，解决复杂水下场景修复问题。提出 SLURPP 网络与合成数据生成方法，提升修复速度与质量。**

- **链接: [http://arxiv.org/pdf/2507.07878v1](http://arxiv.org/pdf/2507.07878v1)**

> **作者:** Jiayi Wu; Tianfu Wang; Md Abu Bakr Siddique; Md Jahidul Islam; Cornelia Fermuller; Yiannis Aloimonos; Christopher A. Metzler
>
> **摘要:** Underwater image restoration algorithms seek to restore the color, contrast, and appearance of a scene that is imaged underwater. They are a critical tool in applications ranging from marine ecology and aquaculture to underwater construction and archaeology. While existing pixel-domain diffusion-based image restoration approaches are effective at restoring simple scenes with limited depth variation, they are computationally intensive and often generate unrealistic artifacts when applied to scenes with complex geometry and significant depth variation. In this work we overcome these limitations by combining a novel network architecture (SLURPP) with an accurate synthetic data generation pipeline. SLURPP combines pretrained latent diffusion models -- which encode strong priors on the geometry and depth of scenes -- with an explicit scene decomposition -- which allows one to model and account for the effects of light attenuation and backscattering. To train SLURPP we design a physics-based underwater image synthesis pipeline that applies varied and realistic underwater degradation effects to existing terrestrial image datasets. This approach enables the generation of diverse training data with dense medium/degradation annotations. We evaluate our method extensively on both synthetic and real-world benchmarks and demonstrate state-of-the-art performance. Notably, SLURPP is over 200X faster than existing diffusion-based methods while offering ~ 3 dB improvement in PSNR on synthetic benchmarks. It also offers compelling qualitative improvements on real-world data. Project website https://tianfwang.github.io/slurpp/.
>
---
#### [new 051] Not Only Consistency: Enhance Test-Time Adaptation with Spatio-temporal Inconsistency for Remote Physiological Measurement
- **分类: cs.CV**

- **简介: 该论文属于远程生理信号监测任务，旨在解决rPPG模型在未知环境下的适应性问题。通过引入时空不一致性先验，提出CiCi框架提升测试时自监督适应能力。**

- **链接: [http://arxiv.org/pdf/2507.07908v1](http://arxiv.org/pdf/2507.07908v1)**

> **作者:** Xiao Yang; Yuxuan Fan; Can Liu; Houcheng Su; Weichen Guo; Jiyao Wang; Dengbo He
>
> **摘要:** Remote photoplethysmography (rPPG) has emerged as a promising non-invasive method for monitoring physiological signals using the camera. Although various domain adaptation and generalization methods were proposed to promote the adaptability of deep-based rPPG models in unseen deployment environments, considerations in aspects like privacy concerns and real-time adaptation restrict their application in real-world deployment. Thus, we aim to propose a novel fully Test-Time Adaptation (TTA) strategy tailored for rPPG tasks in this work. Specifically, based on prior knowledge in physiology and our observations, we noticed not only there is spatio-temporal consistency in the frequency domain of rPPG signals, but also that inconsistency in the time domain was significant. Given this, by leveraging both consistency and inconsistency priors, we introduce an innovative expert knowledge-based self-supervised \textbf{C}onsistency-\textbf{i}n\textbf{C}onsistency-\textbf{i}ntegration (\textbf{CiCi}) framework to enhances model adaptation during inference. Besides, our approach further incorporates a gradient dynamic control mechanism to mitigate potential conflicts between priors, ensuring stable adaptation across instances. Through extensive experiments on five diverse datasets under the TTA protocol, our method consistently outperforms existing techniques, presenting state-of-the-art performance in real-time self-supervised adaptation without accessing source data. The code will be released later.
>
---
#### [new 052] Colors See Colors Ignore: Clothes Changing ReID with Color Disentanglement
- **分类: cs.CV**

- **简介: 该论文属于CC-ReID任务，解决衣物变化下的行人重识别问题。通过颜色解耦方法CSCI，提升模型对衣物不变特征的识别能力。**

- **链接: [http://arxiv.org/pdf/2507.07230v1](http://arxiv.org/pdf/2507.07230v1)**

> **作者:** Priyank Pathak; Yogesh S. Rawat
>
> **备注:** ICCV'25 paper
>
> **摘要:** Clothes-Changing Re-Identification (CC-ReID) aims to recognize individuals across different locations and times, irrespective of clothing. Existing methods often rely on additional models or annotations to learn robust, clothing-invariant features, making them resource-intensive. In contrast, we explore the use of color - specifically foreground and background colors - as a lightweight, annotation-free proxy for mitigating appearance bias in ReID models. We propose Colors See, Colors Ignore (CSCI), an RGB-only method that leverages color information directly from raw images or video frames. CSCI efficiently captures color-related appearance bias ('Color See') while disentangling it from identity-relevant ReID features ('Color Ignore'). To achieve this, we introduce S2A self-attention, a novel self-attention to prevent information leak between color and identity cues within the feature space. Our analysis shows a strong correspondence between learned color embeddings and clothing attributes, validating color as an effective proxy when explicit clothing labels are unavailable. We demonstrate the effectiveness of CSCI on both image and video ReID with extensive experiments on four CC-ReID datasets. We improve the baseline by Top-1 2.9% on LTCC and 5.0% on PRCC for image-based ReID, and 1.0% on CCVID and 2.5% on MeVID for video-based ReID without relying on additional supervision. Our results highlight the potential of color as a cost-effective solution for addressing appearance bias in CC-ReID. Github: https://github.com/ppriyank/ICCV-CSCI-Person-ReID.
>
---
#### [new 053] Multi-Scale Attention and Gated Shifting for Fine-Grained Event Spotting in Videos
- **分类: cs.CV**

- **简介: 该论文属于视频细粒度事件检测任务，旨在提升体育视频中精细动作的帧级识别。针对现有模型时间感知和空间适应性不足的问题，提出MSAGSM模块，结合多尺度时间和多头空间注意力，有效建模长短时依赖，提升检测精度。**

- **链接: [http://arxiv.org/pdf/2507.07381v1](http://arxiv.org/pdf/2507.07381v1)**

> **作者:** Hao Xu; Arbind Agrahari Baniya; Sam Wells; Mohamed Reda Bouadjenek; Richard Dazeley; Sunil Aryal
>
> **摘要:** Precise Event Spotting (PES) in sports videos requires frame-level recognition of fine-grained actions from single-camera footage. Existing PES models typically incorporate lightweight temporal modules such as Gate Shift Module (GSM) or Gate Shift Fuse (GSF) to enrich 2D CNN feature extractors with temporal context. However, these modules are limited in both temporal receptive field and spatial adaptability. We propose a Multi-Scale Attention Gate Shift Module (MSAGSM) that enhances GSM with multi-scale temporal dilations and multi-head spatial attention, enabling efficient modeling of both short- and long-term dependencies while focusing on salient regions. MSAGSM is a lightweight plug-and-play module that can be easily integrated with various 2D backbones. To further advance the field, we introduce the Table Tennis Australia (TTA) dataset-the first PES benchmark for table tennis-containing over 4800 precisely annotated events. Extensive experiments across five PES benchmarks demonstrate that MSAGSM consistently improves performance with minimal overhead, setting new state-of-the-art results.
>
---
#### [new 054] DisenQ: Disentangling Q-Former for Activity-Biometrics
- **分类: cs.CV**

- **简介: 该论文属于活动生物识别任务，解决身份识别中运动与外观干扰问题。提出DisenQ框架，通过语言引导分离生物特征与非生物特征，提升识别准确性。**

- **链接: [http://arxiv.org/pdf/2507.07262v1](http://arxiv.org/pdf/2507.07262v1)**

> **作者:** Shehreen Azad; Yogesh S Rawat
>
> **备注:** Accepted in ICCV 2025
>
> **摘要:** In this work, we address activity-biometrics, which involves identifying individuals across diverse set of activities. Unlike traditional person identification, this setting introduces additional challenges as identity cues become entangled with motion dynamics and appearance variations, making biometrics feature learning more complex. While additional visual data like pose and/or silhouette help, they often struggle from extraction inaccuracies. To overcome this, we propose a multimodal language-guided framework that replaces reliance on additional visual data with structured textual supervision. At its core, we introduce \textbf{DisenQ} (\textbf{Disen}tangling \textbf{Q}-Former), a unified querying transformer that disentangles biometrics, motion, and non-biometrics features by leveraging structured language guidance. This ensures identity cues remain independent of appearance and motion variations, preventing misidentifications. We evaluate our approach on three activity-based video benchmarks, achieving state-of-the-art performance. Additionally, we demonstrate strong generalization to complex real-world scenario with competitive performance on a traditional video-based identification benchmark, showing the effectiveness of our framework.
>
---
#### [new 055] Multi-level Mixture of Experts for Multimodal Entity Linking
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **简介: 该论文属于多模态实体链接任务，解决提及歧义和动态选择模态内容的问题，提出MMoE模型提升链接性能。**

- **链接: [http://arxiv.org/pdf/2507.07108v1](http://arxiv.org/pdf/2507.07108v1)**

> **作者:** Zhiwei Hu; Víctor Gutiérrez-Basulto; Zhiliang Xiang; Ru Li; Jeff Z. Pan
>
> **备注:** Accepted at KDD 2025
>
> **摘要:** Multimodal Entity Linking (MEL) aims to link ambiguous mentions within multimodal contexts to associated entities in a multimodal knowledge base. Existing approaches to MEL introduce multimodal interaction and fusion mechanisms to bridge the modality gap and enable multi-grained semantic matching. However, they do not address two important problems: (i) mention ambiguity, i.e., the lack of semantic content caused by the brevity and omission of key information in the mention's textual context; (ii) dynamic selection of modal content, i.e., to dynamically distinguish the importance of different parts of modal information. To mitigate these issues, we propose a Multi-level Mixture of Experts (MMoE) model for MEL. MMoE has four components: (i) the description-aware mention enhancement module leverages large language models to identify the WikiData descriptions that best match a mention, considering the mention's textual context; (ii) the multimodal feature extraction module adopts multimodal feature encoders to obtain textual and visual embeddings for both mentions and entities; (iii)-(iv) the intra-level mixture of experts and inter-level mixture of experts modules apply a switch mixture of experts mechanism to dynamically and adaptively select features from relevant regions of information. Extensive experiments demonstrate the outstanding performance of MMoE compared to the state-of-the-art. MMoE's code is available at: https://github.com/zhiweihu1103/MEL-MMoE.
>
---
#### [new 056] ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation
- **分类: cs.CV**

- **简介: 该论文属于图像编辑评估任务，解决自动化评估不足问题。提出ADIEE方法生成数据集并训练评分模型，提升评估准确性与效率。**

- **链接: [http://arxiv.org/pdf/2507.07317v1](http://arxiv.org/pdf/2507.07317v1)**

> **作者:** Sherry X. Chen; Yi Wei; Luowei Zhou; Suren Kumar
>
> **备注:** International Conference on Computer Vision (ICCV) 2025
>
> **摘要:** Recent advances in instruction-guided image editing underscore the need for effective automated evaluation. While Vision-Language Models (VLMs) have been explored as judges, open-source models struggle with alignment, and proprietary models lack transparency and cost efficiency. Additionally, no public training datasets exist to fine-tune open-source VLMs, only small benchmarks with diverse evaluation schemes. To address this, we introduce ADIEE, an automated dataset creation approach which is then used to train a scoring model for instruction-guided image editing evaluation. We generate a large-scale dataset with over 100K samples and use it to fine-tune a LLaVA-NeXT-8B model modified to decode a numeric score from a custom token. The resulting scorer outperforms all open-source VLMs and Gemini-Pro 1.5 across all benchmarks, achieving a 0.0696 (+17.24%) gain in score correlation with human ratings on AURORA-Bench, and improving pair-wise comparison accuracy by 4.03% (+7.21%) on GenAI-Bench and 4.75% (+9.35%) on AURORA-Bench, respectively, compared to the state-of-the-art. The scorer can act as a reward model, enabling automated best edit selection and model fine-tuning. Notably, the proposed scorer can boost MagicBrush model's average evaluation score on ImagenHub from 5.90 to 6.43 (+8.98%).
>
---
#### [new 057] Visual Instance-aware Prompt Tuning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉任务，解决VPT在不同输入实例间性能不稳定的问题，提出ViaPT生成实例感知提示，融合数据集级提示，提升模型表现。**

- **链接: [http://arxiv.org/pdf/2507.07796v1](http://arxiv.org/pdf/2507.07796v1)**

> **作者:** Xi Xiao; Yunbei Zhang; Xingjian Li; Tianyang Wang; Xiao Wang; Yuxiang Wei; Jihun Hamm; Min Xu
>
> **摘要:** Visual Prompt Tuning (VPT) has emerged as a parameter-efficient fine-tuning paradigm for vision transformers, with conventional approaches utilizing dataset-level prompts that remain the same across all input instances. We observe that this strategy results in sub-optimal performance due to high variance in downstream datasets. To address this challenge, we propose Visual Instance-aware Prompt Tuning (ViaPT), which generates instance-aware prompts based on each individual input and fuses them with dataset-level prompts, leveraging Principal Component Analysis (PCA) to retain important prompting information. Moreover, we reveal that VPT-Deep and VPT-Shallow represent two corner cases based on a conceptual understanding, in which they fail to effectively capture instance-specific information, while random dimension reduction on prompts only yields performance between the two extremes. Instead, ViaPT overcomes these limitations by balancing dataset-level and instance-level knowledge, while reducing the amount of learnable parameters compared to VPT-Deep. Extensive experiments across 34 diverse datasets demonstrate that our method consistently outperforms state-of-the-art baselines, establishing a new paradigm for analyzing and optimizing visual prompts for vision transformers.
>
---
#### [new 058] X-RAFT: Cross-Modal Non-Rigid Registration of Blue and White Light Neurosurgical Hyperspectral Images
- **分类: cs.CV**

- **简介: 该论文属于跨模态图像配准任务，解决蓝光与白光神经外科高光谱图像的对应问题。提出X-RAFT模型提升配准精度。**

- **链接: [http://arxiv.org/pdf/2507.07747v1](http://arxiv.org/pdf/2507.07747v1)**

> **作者:** Charlie Budd; Silvère Ségaud; Matthew Elliot; Graeme Stasiuk; Yijing Xie; Jonathan Shapey; Tom Vercauteren
>
> **摘要:** Integration of hyperspectral imaging into fluorescence-guided neurosurgery has the potential to improve surgical decision making by providing quantitative fluorescence measurements in real-time. Quantitative fluorescence requires paired spectral data in fluorescence (blue light) and reflectance (white light) mode. Blue and white image acquisition needs to be performed sequentially in a potentially dynamic surgical environment. A key component to the fluorescence quantification process is therefore the ability to find dense cross-modal image correspondences between two hyperspectral images taken under these drastically different lighting conditions. We address this challenge with the introduction of X-RAFT, a Recurrent All-Pairs Field Transforms (RAFT) optical flow model modified for cross-modal inputs. We propose using distinct image encoders for each modality pair, and fine-tune these in a self-supervised manner using flow-cycle-consistency on our neurosurgical hyperspectral data. We show an error reduction of 36.6% across our evaluation metrics when comparing to a naive baseline and 27.83% reduction compared to an existing cross-modal optical flow method (CrossRAFT). Our code and models will be made publicly available after the review process.
>
---
#### [new 059] Single-pass Adaptive Image Tokenization for Minimum Program Search
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像表示学习任务，旨在解决固定长度表示与复杂度不匹配的问题。提出KARL模型，在单次通过中自适应确定图像令牌数量，以逼近最小描述长度。**

- **链接: [http://arxiv.org/pdf/2507.07995v1](http://arxiv.org/pdf/2507.07995v1)**

> **作者:** Shivam Duggal; Sanghyun Byun; William T. Freeman; Antonio Torralba; Phillip Isola
>
> **备注:** Code at: https://github.com/ShivamDuggal4/karl Keywords: Representation Learning, Adaptive Tokenization, Compression, Algorithmic Information Theory, Kolmogorov Complexity, Upside-Down RL
>
> **摘要:** According to Algorithmic Information Theory (AIT) -- Intelligent representations compress data into the shortest possible program that can reconstruct its content, exhibiting low Kolmogorov Complexity (KC). In contrast, most visual representation learning systems use fixed-length representations for all inputs, ignoring variations in complexity or familiarity. Recent adaptive tokenization methods address this by allocating variable-length representations but typically require test-time search over multiple encodings to find the most predictive one. Inspired by Kolmogorov Complexity principles, we propose a single-pass adaptive tokenizer, KARL, which predicts the appropriate number of tokens for an image in a single forward pass, halting once its approximate KC is reached. The token count serves as a proxy for the minimum description length. KARL's training procedure closely resembles the Upside-Down Reinforcement Learning paradigm, as it learns to conditionally predict token halting based on a desired reconstruction quality. KARL matches the performance of recent adaptive tokenizers while operating in a single pass. We present scaling laws for KARL, analyzing the role of encoder/decoder size, continuous vs. discrete tokenization and more. Additionally, we offer a conceptual study drawing an analogy between Adaptive Image Tokenization and Algorithmic Information Theory, examining the predicted image complexity (KC) across axes such as structure vs. noise and in- vs. out-of-distribution familiarity -- revealing alignment with human intuition.
>
---
#### [new 060] Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于多模态推理任务，解决LVLMs在CoT中忽略生成理性的问题。提出RED方法，通过融合视觉与理性信息提升推理准确性与可靠性。**

- **链接: [http://arxiv.org/pdf/2507.07685v1](http://arxiv.org/pdf/2507.07685v1)**

> **作者:** Shin'ya Yamaguchi; Kosuke Nishida; Daiki Chijiwa
>
> **备注:** 17 pages, 4 figures
>
> **摘要:** Large vision-language models (LVLMs) have demonstrated remarkable capabilities by integrating pre-trained vision encoders with large language models (LLMs). Similar to single-modal LLMs, chain-of-thought (CoT) prompting has been adapted for LVLMs to enhance multi-modal reasoning by generating intermediate rationales based on visual and textual inputs. While CoT is assumed to improve grounding and accuracy in LVLMs, our experiments reveal a key challenge: existing LVLMs often ignore the contents of generated rationales in CoT reasoning. To address this, we re-formulate multi-modal CoT reasoning as a KL-constrained reward maximization focused on rationale-conditional log-likelihood. As the optimal solution, we propose rationale-enhanced decoding (RED), a novel plug-and-play inference-time decoding strategy. RED harmonizes visual and rationale information by multiplying distinct image-conditional and rationale-conditional next token distributions. Extensive experiments show that RED consistently and significantly improves reasoning over standard CoT and other decoding methods across multiple benchmarks and LVLMs. Our work offers a practical and effective approach to improve both the faithfulness and accuracy of CoT reasoning in LVLMs, paving the way for more reliable rationale-grounded multi-modal systems.
>
---
#### [new 061] Hardware-Aware Feature Extraction Quantisation for Real-Time Visual Odometry on FPGA Platforms
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于视觉里程计任务，旨在提升FPGA平台上的实时特征提取效率。通过量化SuperPoint网络，实现低计算需求下的高精度特征检测。**

- **链接: [http://arxiv.org/pdf/2507.07903v1](http://arxiv.org/pdf/2507.07903v1)**

> **作者:** Mateusz Wasala; Mateusz Smolarczyk; Michal Danilowicz; Tomasz Kryjak
>
> **备注:** Accepted for the DSD 2025 conference in Salerno, Italy
>
> **摘要:** Accurate position estimation is essential for modern navigation systems deployed in autonomous platforms, including ground vehicles, marine vessels, and aerial drones. In this context, Visual Simultaneous Localisation and Mapping (VSLAM) - which includes Visual Odometry - relies heavily on the reliable extraction of salient feature points from the visual input data. In this work, we propose an embedded implementation of an unsupervised architecture capable of detecting and describing feature points. It is based on a quantised SuperPoint convolutional neural network. Our objective is to minimise the computational demands of the model while preserving high detection quality, thus facilitating efficient deployment on platforms with limited resources, such as mobile or embedded systems. We implemented the solution on an FPGA System-on-Chip (SoC) platform, specifically the AMD/Xilinx Zynq UltraScale+, where we evaluated the performance of Deep Learning Processing Units (DPUs) and we also used the Brevitas library and the FINN framework to perform model quantisation and hardware-aware optimisation. This allowed us to process 640 x 480 pixel images at up to 54 fps on an FPGA platform, outperforming state-of-the-art solutions in the field. We conducted experiments on the TUM dataset to demonstrate and discuss the impact of different quantisation techniques on the accuracy and performance of the model in a visual odometry task.
>
---
#### [new 062] One Object, Multiple Lies: A Benchmark for Cross-task Adversarial Attack on Unified Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型安全研究，解决跨任务对抗攻击问题。构建了CrossVLAD数据集，并提出CRAFT方法，提升对统一VLM的跨任务攻击效果。**

- **链接: [http://arxiv.org/pdf/2507.07709v1](http://arxiv.org/pdf/2507.07709v1)**

> **作者:** Jiale Zhao; Xinyang Jiang; Junyao Gao; Yuhao Xue; Cairong Zhao
>
> **摘要:** Unified vision-language models(VLMs) have recently shown remarkable progress, enabling a single model to flexibly address diverse tasks through different instructions within a shared computational architecture. This instruction-based control mechanism creates unique security challenges, as adversarial inputs must remain effective across multiple task instructions that may be unpredictably applied to process the same malicious content. In this paper, we introduce CrossVLAD, a new benchmark dataset carefully curated from MSCOCO with GPT-4-assisted annotations for systematically evaluating cross-task adversarial attacks on unified VLMs. CrossVLAD centers on the object-change objective-consistently manipulating a target object's classification across four downstream tasks-and proposes a novel success rate metric that measures simultaneous misclassification across all tasks, providing a rigorous evaluation of adversarial transferability. To tackle this challenge, we present CRAFT (Cross-task Region-based Attack Framework with Token-alignment), an efficient region-centric attack method. Extensive experiments on Florence-2 and other popular unified VLMs demonstrate that our method outperforms existing approaches in both overall cross-task attack performance and targeted object-change success rates, highlighting its effectiveness in adversarially influencing unified VLMs across diverse tasks.
>
---
#### [new 063] Understanding Dataset Bias in Medical Imaging: A Case Study on Chest X-rays
- **分类: cs.CV**

- **简介: 该论文研究医疗影像数据集偏差问题，通过变换数据探索胸部X光数据集中是否存在偏差，旨在评估AI是否依赖数据特征而非病理特征。**

- **链接: [http://arxiv.org/pdf/2507.07722v1](http://arxiv.org/pdf/2507.07722v1)**

> **作者:** Ethan Dack; Chengliang Dai
>
> **摘要:** Recent work has revisited the infamous task Name that dataset and established that in non-medical datasets, there is an underlying bias and achieved high Accuracies on the dataset origin task. In this work, we revisit the same task applied to popular open-source chest X-ray datasets. Medical images are naturally more difficult to release for open-source due to their sensitive nature, which has led to certain open-source datasets being extremely popular for research purposes. By performing the same task, we wish to explore whether dataset bias also exists in these datasets. % We deliberately try to increase the difficulty of the task by dataset transformations. We apply simple transformations of the datasets to try to identify bias. Given the importance of AI applications in medical imaging, it's vital to establish whether modern methods are taking shortcuts or are focused on the relevant pathology. We implement a range of different network architectures on the datasets: NIH, CheXpert, MIMIC-CXR and PadChest. We hope this work will encourage more explainable research being performed in medical imaging and the creation of more open-source datasets in the medical domain. The corresponding code will be released upon acceptance.
>
---
#### [new 064] Multigranular Evaluation for Brain Visual Decoding
- **分类: cs.CV; cs.AI; eess.IV; q-bio.NC**

- **简介: 该论文属于脑视觉解码任务，旨在解决现有评估方法粗糙、缺乏神经科学依据的问题。提出BASIC框架，从结构、语义和上下文层面进行多粒度评估。**

- **链接: [http://arxiv.org/pdf/2507.07993v1](http://arxiv.org/pdf/2507.07993v1)**

> **作者:** Weihao Xia; Cengiz Oztireli
>
> **备注:** Project: https://weihaox.github.io/BASIC
>
> **摘要:** Existing evaluation protocols for brain visual decoding predominantly rely on coarse metrics that obscure inter-model differences, lack neuroscientific foundation, and fail to capture fine-grained visual distinctions. To address these limitations, we introduce BASIC, a unified, multigranular evaluation framework that jointly quantifies structural fidelity, inferential alignment, and contextual coherence between decoded and ground truth images. For the structural level, we introduce a hierarchical suite of segmentation-based metrics, including foreground, semantic, instance, and component masks, anchored in granularity-aware correspondence across mask structures. For the semantic level, we extract structured scene representations encompassing objects, attributes, and relationships using multimodal large language models, enabling detailed, scalable, and context-rich comparisons with ground-truth stimuli. We benchmark a diverse set of visual decoding methods across multiple stimulus-neuroimaging datasets within this unified evaluation framework. Together, these criteria provide a more discriminative, interpretable, and comprehensive foundation for measuring brain visual decoding methods.
>
---
#### [new 065] HiM2SAM: Enhancing SAM2 with Hierarchical Motion Estimation and Memory Optimization towards Long-term Tracking
- **分类: cs.CV**

- **简介: 该论文属于视频目标跟踪任务，解决遮挡、背景干扰和目标重现问题。通过引入分层运动估计和记忆优化，提升SAM2的长期跟踪性能。**

- **链接: [http://arxiv.org/pdf/2507.07603v1](http://arxiv.org/pdf/2507.07603v1)**

> **作者:** Ruixiang Chen; Guolei Sun; Yawei Li; Jie Qin; Luca Benini
>
> **摘要:** This paper presents enhancements to the SAM2 framework for video object tracking task, addressing challenges such as occlusions, background clutter, and target reappearance. We introduce a hierarchical motion estimation strategy, combining lightweight linear prediction with selective non-linear refinement to improve tracking accuracy without requiring additional training. In addition, we optimize the memory bank by distinguishing long-term and short-term memory frames, enabling more reliable tracking under long-term occlusions and appearance changes. Experimental results show consistent improvements across different model scales. Our method achieves state-of-the-art performance on LaSOT and LaSOText with the large model, achieving 9.6% and 7.2% relative improvements in AUC over the original SAM2, and demonstrates even larger relative gains on smaller models, highlighting the effectiveness of our trainless, low-overhead improvements for boosting long-term tracking performance. The code is available at https://github.com/LouisFinner/HiM2SAM.
>
---
#### [new 066] Semi-supervised learning and integration of multi-sequence MR-images for carotid vessel wall and plaque segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分割任务，旨在解决多序列MRI数据中颈动脉斑块分割难题。通过半监督学习和多模态融合策略提升分割精度。**

- **链接: [http://arxiv.org/pdf/2507.07496v1](http://arxiv.org/pdf/2507.07496v1)**

> **作者:** Marie-Christine Pali; Christina Schwaiger; Malik Galijasevic; Valentin K. Ladenhauf; Stephanie Mangesius; Elke R. Gizewski
>
> **摘要:** The analysis of carotid arteries, particularly plaques, in multi-sequence Magnetic Resonance Imaging (MRI) data is crucial for assessing the risk of atherosclerosis and ischemic stroke. In order to evaluate metrics and radiomic features, quantifying the state of atherosclerosis, accurate segmentation is important. However, the complex morphology of plaques and the scarcity of labeled data poses significant challenges. In this work, we address these problems and propose a semi-supervised deep learning-based approach designed to effectively integrate multi-sequence MRI data for the segmentation of carotid artery vessel wall and plaque. The proposed algorithm consists of two networks: a coarse localization model identifies the region of interest guided by some prior knowledge on the position and number of carotid arteries, followed by a fine segmentation model for precise delineation of vessel walls and plaques. To effectively integrate complementary information across different MRI sequences, we investigate different fusion strategies and introduce a multi-level multi-sequence version of U-Net architecture. To address the challenges of limited labeled data and the complexity of carotid artery MRI, we propose a semi-supervised approach that enforces consistency under various input transformations. Our approach is evaluated on 52 patients with arteriosclerosis, each with five MRI sequences. Comprehensive experiments demonstrate the effectiveness of our approach and emphasize the role of fusion point selection in U-Net-based architectures. To validate the accuracy of our results, we also include an expert-based assessment of model performance. Our findings highlight the potential of fusion strategies and semi-supervised learning for improving carotid artery segmentation in data-limited MRI applications.
>
---
#### [new 067] Behave Your Motion: Habit-preserved Cross-category Animal Motion Transfer
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于跨类别动物运动迁移任务，旨在解决动物特有行为习惯在运动转移中的丢失问题。提出一种保留习惯的框架，并引入大语言模型提升泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.07394v1](http://arxiv.org/pdf/2507.07394v1)**

> **作者:** Zhimin Zhang; Bi'an Du; Caoyuan Ma; Zheng Wang; Wei Hu
>
> **摘要:** Animal motion embodies species-specific behavioral habits, making the transfer of motion across categories a critical yet complex task for applications in animation and virtual reality. Existing motion transfer methods, primarily focused on human motion, emphasize skeletal alignment (motion retargeting) or stylistic consistency (motion style transfer), often neglecting the preservation of distinct habitual behaviors in animals. To bridge this gap, we propose a novel habit-preserved motion transfer framework for cross-category animal motion. Built upon a generative framework, our model introduces a habit-preservation module with category-specific habit encoder, allowing it to learn motion priors that capture distinctive habitual characteristics. Furthermore, we integrate a large language model (LLM) to facilitate the motion transfer to previously unobserved species. To evaluate the effectiveness of our approach, we introduce the DeformingThings4D-skl dataset, a quadruped dataset with skeletal bindings, and conduct extensive experiments and quantitative analyses, which validate the superiority of our proposed model.
>
---
#### [new 068] Dual Semantic-Aware Network for Noise Suppressed Ultrasound Video Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决超声视频中的噪声干扰问题。提出DSANet框架，通过双语义感知模块提升分割精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.07443v1](http://arxiv.org/pdf/2507.07443v1)**

> **作者:** Ling Zhou; Runtian Yuan; Yi Liu; Yuejie Zhang; Rui Feng; Shang Gao
>
> **摘要:** Ultrasound imaging is a prevalent diagnostic tool known for its simplicity and non-invasiveness. However, its inherent characteristics often introduce substantial noise, posing considerable challenges for automated lesion or organ segmentation in ultrasound video sequences. To address these limitations, we propose the Dual Semantic-Aware Network (DSANet), a novel framework designed to enhance noise robustness in ultrasound video segmentation by fostering mutual semantic awareness between local and global features. Specifically, we introduce an Adjacent-Frame Semantic-Aware (AFSA) module, which constructs a channel-wise similarity matrix to guide feature fusion across adjacent frames, effectively mitigating the impact of random noise without relying on pixel-level relationships. Additionally, we propose a Local-and-Global Semantic-Aware (LGSA) module that reorganizes and fuses temporal unconditional local features, which capture spatial details independently at each frame, with conditional global features that incorporate temporal context from adjacent frames. This integration facilitates multi-level semantic representation, significantly improving the model's resilience to noise interference. Extensive evaluations on four benchmark datasets demonstrate that DSANet substantially outperforms state-of-the-art methods in segmentation accuracy. Moreover, since our model avoids pixel-level feature dependencies, it achieves significantly higher inference FPS than video-based methods, and even surpasses some image-based models. Code can be found in \href{https://github.com/ZhouL2001/DSANet}{DSANet}
>
---
#### [new 069] TinierHAR: Towards Ultra-Lightweight Deep Learning Models for Efficient Human Activity Recognition on Edge Devices
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于人体活动识别任务，旨在解决边缘设备上模型轻量化问题。通过设计TinierHAR架构，提升计算效率并保持性能。**

- **链接: [http://arxiv.org/pdf/2507.07949v1](http://arxiv.org/pdf/2507.07949v1)**

> **作者:** Sizhen Bian; Mengxi Liu; Vitor Fortes Rey; Daniel Geissler; Paul Lukowicz
>
> **摘要:** Human Activity Recognition (HAR) on resource-constrained wearable devices demands inference models that harmonize accuracy with computational efficiency. This paper introduces TinierHAR, an ultra-lightweight deep learning architecture that synergizes residual depthwise separable convolutions, gated recurrent units (GRUs), and temporal aggregation to achieve SOTA efficiency without compromising performance. Evaluated across 14 public HAR datasets, TinierHAR reduces Parameters by 2.7x (vs. TinyHAR) and 43.3x (vs. DeepConvLSTM), and MACs by 6.4x and 58.6x, respectively, while maintaining the averaged F1-scores. Beyond quantitative gains, this work provides the first systematic ablation study dissecting the contributions of spatial-temporal components across proposed TinierHAR, prior SOTA TinyHAR, and the classical DeepConvLSTM, offering actionable insights for designing efficient HAR systems. We finally discussed the findings and suggested principled design guidelines for future efficient HAR. To catalyze edge-HAR research, we open-source all materials in this work for future benchmarking\footnote{https://github.com/zhaxidele/TinierHAR}
>
---
#### [new 070] Aerial Maritime Vessel Detection and Identification
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自主海上目标识别任务，解决GNSS缺失环境下通过视觉定位和识别船舶的问题。工作包括使用YOLOv8检测船舶，结合特征匹配与几何原理实现定位。**

- **链接: [http://arxiv.org/pdf/2507.07153v1](http://arxiv.org/pdf/2507.07153v1)**

> **作者:** Antonella Barisic Kulas; Frano Petric; Stjepan Bogdan
>
> **备注:** Preprint. ICUAS 2025
>
> **摘要:** Autonomous maritime surveillance and target vessel identification in environments where Global Navigation Satellite Systems (GNSS) are not available is critical for a number of applications such as search and rescue and threat detection. When the target vessel is only described by visual cues and its last known position is not available, unmanned aerial vehicles (UAVs) must rely solely on on-board vision to scan a large search area under strict computational constraints. To address this challenge, we leverage the YOLOv8 object detection model to detect all vessels in the field of view. We then apply feature matching and hue histogram distance analysis to determine whether any detected vessel corresponds to the target. When found, we localize the target using simple geometric principles. We demonstrate the proposed method in real-world experiments during the MBZIRC2023 competition, integrated into a fully autonomous system with GNSS-denied navigation. We also evaluate the impact of perspective on detection accuracy and localization precision and compare it with the oracle approach.
>
---
#### [new 071] Spline Deformation Field
- **分类: cs.CV**

- **简介: 该论文属于轨迹建模任务，解决隐式变形场在稀疏数据下的空间一致性问题。提出基于样条的轨迹表示，提升动态场景重建质量与运动连贯性。**

- **链接: [http://arxiv.org/pdf/2507.07521v1](http://arxiv.org/pdf/2507.07521v1)**

> **作者:** Mingyang Song; Yang Zhang; Marko Mihajlovic; Siyu Tang; Markus Gross; Tunç Ozan Aydın
>
> **摘要:** Trajectory modeling of dense points usually employs implicit deformation fields, represented as neural networks that map coordinates to relate canonical spatial positions to temporal offsets. However, the inductive biases inherent in neural networks can hinder spatial coherence in ill-posed scenarios. Current methods focus either on enhancing encoding strategies for deformation fields, often resulting in opaque and less intuitive models, or adopt explicit techniques like linear blend skinning, which rely on heuristic-based node initialization. Additionally, the potential of implicit representations for interpolating sparse temporal signals remains under-explored. To address these challenges, we propose a spline-based trajectory representation, where the number of knots explicitly determines the degrees of freedom. This approach enables efficient analytical derivation of velocities, preserving spatial coherence and accelerations, while mitigating temporal fluctuations. To model knot characteristics in both spatial and temporal domains, we introduce a novel low-rank time-variant spatial encoding, replacing conventional coupled spatiotemporal techniques. Our method demonstrates superior performance in temporal interpolation for fitting continuous fields with sparse inputs. Furthermore, it achieves competitive dynamic scene reconstruction quality compared to state-of-the-art methods while enhancing motion coherence without relying on linear blend skinning or as-rigid-as-possible constraints.
>
---
#### [new 072] THUNDER: Tile-level Histopathology image UNDERstanding benchmark
- **分类: cs.CV**

- **简介: 该论文属于数字病理学领域，旨在解决基础模型评估问题。提出THUNDER基准，用于高效比较多种模型在不同数据集上的表现及鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.07860v1](http://arxiv.org/pdf/2507.07860v1)**

> **作者:** Pierre Marza; Leo Fillioux; Sofiène Boutaj; Kunal Mahatha; Christian Desrosiers; Pablo Piantanida; Jose Dolz; Stergios Christodoulidis; Maria Vakalopoulou
>
> **摘要:** Progress in a research field can be hard to assess, in particular when many concurrent methods are proposed in a short period of time. This is the case in digital pathology, where many foundation models have been released recently to serve as feature extractors for tile-level images, being used in a variety of downstream tasks, both for tile- and slide-level problems. Benchmarking available methods then becomes paramount to get a clearer view of the research landscape. In particular, in critical domains such as healthcare, a benchmark should not only focus on evaluating downstream performance, but also provide insights about the main differences between methods, and importantly, further consider uncertainty and robustness to ensure a reliable usage of proposed models. For these reasons, we introduce THUNDER, a tile-level benchmark for digital pathology foundation models, allowing for efficient comparison of many models on diverse datasets with a series of downstream tasks, studying their feature spaces and assessing the robustness and uncertainty of predictions informed by their embeddings. THUNDER is a fast, easy-to-use, dynamic benchmark that can already support a large variety of state-of-the-art foundation, as well as local user-defined models for direct tile-based comparison. In this paper, we provide a comprehensive comparison of 23 foundation models on 16 different datasets covering diverse tasks, feature analysis, and robustness. The code for THUNDER is publicly available at https://github.com/MICS-Lab/thunder.
>
---
#### [new 073] Energy-Guided Decoding for Object Hallucination Mitigation
- **分类: cs.CV**

- **简介: 该论文属于视觉问答任务，旨在解决大模型中的对象幻觉问题。提出一种基于能量的解码方法，有效降低“是”回答比例偏差并提升性能。**

- **链接: [http://arxiv.org/pdf/2507.07731v1](http://arxiv.org/pdf/2507.07731v1)**

> **作者:** Xixi Liu; Ailin Deng; Christopher Zach
>
> **摘要:** Mitigating object hallucination in large vision-language models (LVLMs) is critical to their safe deployment. Existing methods either are restricted to specific decoding methods, or demand sophisticated modifications to visual inputs, or rely on knowledge from external models. In this work, we first reveal the phenomenon that VLMs exhibit significant imbalance in the ``Yes'' ratio ( \ie, the fraction of ``Yes'' answers among the total number of questions) across three different visual question answering (VQA) datasets. Furthermore, we propose an energy-based decoding method, which dynamically selects the hidden states from the layer with minimal energy score. It is simple yet effective in reducing the bias for the yes ratio while boosting performance across three benchmarks (POPE, MME, and MMVP). Our method consistently improves accuracy and F1 score on three VQA datasets across three commonly used VLMs over several baseline methods. The average accuracy improvement is 4.82% compared to greedy decoding. Moreover, the average yes-ratio gap reduction is 8.81%, meaning the proposed method is less biased as shown in Figure 1.
>
---
#### [new 074] Tree-Mamba: A Tree-Aware Mamba for Underwater Monocular Depth Estimation
- **分类: cs.CV**

- **简介: 该论文属于 underwater monocular depth estimation 任务，解决水下图像深度估计精度低的问题。提出 Tree-Mamba 方法，通过树状扫描策略提升多尺度特征表示，并构建了 BlueDepth 数据集。**

- **链接: [http://arxiv.org/pdf/2507.07687v1](http://arxiv.org/pdf/2507.07687v1)**

> **作者:** Peixian Zhuang; Yijian Wang; Zhenqi Fu; Hongliang Zhang; Sam Kwong; Chongyi Li
>
> **摘要:** Underwater Monocular Depth Estimation (UMDE) is a critical task that aims to estimate high-precision depth maps from underwater degraded images caused by light absorption and scattering effects in marine environments. Recently, Mamba-based methods have achieved promising performance across various vision tasks; however, they struggle with the UMDE task because their inflexible state scanning strategies fail to model the structural features of underwater images effectively. Meanwhile, existing UMDE datasets usually contain unreliable depth labels, leading to incorrect object-depth relationships between underwater images and their corresponding depth maps. To overcome these limitations, we develop a novel tree-aware Mamba method, dubbed Tree-Mamba, for estimating accurate monocular depth maps from underwater degraded images. Specifically, we propose a tree-aware scanning strategy that adaptively constructs a minimum spanning tree based on feature similarity. The spatial topological features among the tree nodes are then flexibly aggregated through bottom-up and top-down traversals, enabling stronger multi-scale feature representation capabilities. Moreover, we construct an underwater depth estimation benchmark (called BlueDepth), which consists of 38,162 underwater image pairs with reliable depth labels. This benchmark serves as a foundational dataset for training existing deep learning-based UMDE methods to learn accurate object-depth relationships. Extensive experiments demonstrate the superiority of the proposed Tree-Mamba over several leading methods in both qualitative results and quantitative evaluations with competitive computational efficiency. Code and dataset will be available at https://wyjgr.github.io/Tree-Mamba.html.
>
---
#### [new 075] Geometry Forcing: Marrying Video Diffusion and 3D Representation for Consistent World Modeling
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成任务，旨在解决视频扩散模型缺乏几何结构的问题。通过引入几何引导机制，增强模型对3D结构的感知与一致性。**

- **链接: [http://arxiv.org/pdf/2507.07982v1](http://arxiv.org/pdf/2507.07982v1)**

> **作者:** Haoyu Wu; Diankun Wu; Tianyu He; Junliang Guo; Yang Ye; Yueqi Duan; Jiang Bian
>
> **备注:** 18 pages, project page: https://GeometryForcing.github.io
>
> **摘要:** Videos inherently represent 2D projections of a dynamic 3D world. However, our analysis suggests that video diffusion models trained solely on raw video data often fail to capture meaningful geometric-aware structure in their learned representations. To bridge this gap between video diffusion models and the underlying 3D nature of the physical world, we propose Geometry Forcing, a simple yet effective method that encourages video diffusion models to internalize latent 3D representations. Our key insight is to guide the model's intermediate representations toward geometry-aware structure by aligning them with features from a pretrained geometric foundation model. To this end, we introduce two complementary alignment objectives: Angular Alignment, which enforces directional consistency via cosine similarity, and Scale Alignment, which preserves scale-related information by regressing unnormalized geometric features from normalized diffusion representation. We evaluate Geometry Forcing on both camera view-conditioned and action-conditioned video generation tasks. Experimental results demonstrate that our method substantially improves visual quality and 3D consistency over the baseline methods. Project page: https://GeometryForcing.github.io.
>
---
#### [new 076] Where are we with calibration under dataset shift in image classification?
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像分类任务，研究数据集偏移下的校准问题。通过实验比较不同校准方法，提出有效提升校准鲁棒性的实践建议。**

- **链接: [http://arxiv.org/pdf/2507.07780v1](http://arxiv.org/pdf/2507.07780v1)**

> **作者:** Mélanie Roschewitz; Raghav Mehta; Fabio de Sousa Ribeiro; Ben Glocker
>
> **备注:** Code available at https://github.com/biomedia-mira/calibration_under_shifts
>
> **摘要:** We conduct an extensive study on the state of calibration under real-world dataset shift for image classification. Our work provides important insights on the choice of post-hoc and in-training calibration techniques, and yields practical guidelines for all practitioners interested in robust calibration under shift. We compare various post-hoc calibration methods, and their interactions with common in-training calibration strategies (e.g., label smoothing), across a wide range of natural shifts, on eight different classification tasks across several imaging domains. We find that: (i) simultaneously applying entropy regularisation and label smoothing yield the best calibrated raw probabilities under dataset shift, (ii) post-hoc calibrators exposed to a small amount of semantic out-of-distribution data (unrelated to the task) are most robust under shift, (iii) recent calibration methods specifically aimed at increasing calibration under shifts do not necessarily offer significant improvements over simpler post-hoc calibration methods, (iv) improving calibration under shifts often comes at the cost of worsening in-distribution calibration. Importantly, these findings hold for randomly initialised classifiers, as well as for those finetuned from foundation models, the latter being consistently better calibrated compared to models trained from scratch. Finally, we conduct an in-depth analysis of ensembling effects, finding that (i) applying calibration prior to ensembling (instead of after) is more effective for calibration under shifts, (ii) for ensembles, OOD exposure deteriorates the ID-shifted calibration trade-off, (iii) ensembling remains one of the most effective methods to improve calibration robustness and, combined with finetuning from foundation models, yields best calibration results overall.
>
---
#### [new 077] CL-Polyp: A Contrastive Learning-Enhanced Network for Accurate Polyp Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决结肠镜图像中息肉精准分割问题。提出CL-Polyp网络，结合对比学习与轻量模块，提升分割效果。**

- **链接: [http://arxiv.org/pdf/2507.07154v1](http://arxiv.org/pdf/2507.07154v1)**

> **作者:** Desheng Li; Chaoliang Liu; Zhiyong Xiao
>
> **摘要:** Accurate segmentation of polyps from colonoscopy images is crucial for the early diagnosis and treatment of colorectal cancer. Most existing deep learning-based polyp segmentation methods adopt an Encoder-Decoder architecture, and some utilize multi-task frameworks that incorporate auxiliary tasks such as classification to enhance segmentation performance. However, these approaches often require additional labeled data and rely on task similarity, which can limit their generalizability. To address these challenges, we propose CL-Polyp, a contrastive learning-enhanced polyp segmentation network. Our method leverages contrastive learning to improve the encoder's ability to extract discriminative features by contrasting positive and negative sample pairs derived from polyp images. This self-supervised strategy enhances visual representation without requiring additional annotations. In addition, we introduce two lightweight and effective modules: the Modified Atrous Spatial Pyramid Pooling (MASPP) module for better multi-scale feature fusion, and the Channel Concatenate and Element Add (CA) module to fuse low-level and upsampled features for improved boundary reconstruction. Extensive experiments on five benchmark datasets-Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, CVC-300, and ETIS-demonstrate that CL-Polyp consistently outperforms state-of-the-art methods. Specifically, it improves the IoU metric by 0.011 and 0.020 on the Kvasir-SEG and CVC-ClinicDB datasets, respectively, validating its effectiveness in clinical polyp segmentation tasks.
>
---
#### [new 078] EPIC: Efficient Prompt Interaction for Text-Image Classification
- **分类: cs.CV**

- **简介: 该论文属于文本-图像分类任务，旨在解决大模型微调计算成本高的问题。提出EPIC方法，通过高效提示交互减少资源消耗并提升性能。**

- **链接: [http://arxiv.org/pdf/2507.07415v1](http://arxiv.org/pdf/2507.07415v1)**

> **作者:** Xinyao Yu; Hao Sun; Zeyu Ling; Ziwei Niu; Zhenjia Bai; Rui Qin; Yen-Wei Chen; Lanfen Lin
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2401.14856
>
> **摘要:** In recent years, large-scale pre-trained multimodal models (LMMs) generally emerge to integrate the vision and language modalities, achieving considerable success in multimodal tasks, such as text-image classification. The growing size of LMMs, however, results in a significant computational cost for fine-tuning these models for downstream tasks. Hence, prompt-based interaction strategy is studied to align modalities more efficiently. In this context, we propose a novel efficient prompt-based multimodal interaction strategy, namely Efficient Prompt Interaction for text-image Classification (EPIC). Specifically, we utilize temporal prompts on intermediate layers, and integrate different modalities with similarity-based prompt interaction, to leverage sufficient information exchange between modalities. Utilizing this approach, our method achieves reduced computational resource consumption and fewer trainable parameters (about 1\% of the foundation model) compared to other fine-tuning strategies. Furthermore, it demonstrates superior performance on the UPMC-Food101 and SNLI-VE datasets, while achieving comparable performance on the MM-IMDB dataset.
>
---
#### [new 079] Rethinking Query-based Transformer for Continual Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于持续图像分割任务，解决类别增量学习中的灾难性遗忘问题。提出SimCIS方法，通过查询与特征对齐提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.07831v1](http://arxiv.org/pdf/2507.07831v1)**

> **作者:** Yuchen Zhu; Cheng Shi; Dingyou Wang; Jiajin Tang; Zhengxuan Wei; Yu Wu; Guanbin Li; Sibei Yang
>
> **备注:** This work is accepted by CVPR 2025
>
> **摘要:** Class-incremental/Continual image segmentation (CIS) aims to train an image segmenter in stages, where the set of available categories differs at each stage. To leverage the built-in objectness of query-based transformers, which mitigates catastrophic forgetting of mask proposals, current methods often decouple mask generation from the continual learning process. This study, however, identifies two key issues with decoupled frameworks: loss of plasticity and heavy reliance on input data order. To address these, we conduct an in-depth investigation of the built-in objectness and find that highly aggregated image features provide a shortcut for queries to generate masks through simple feature alignment. Based on this, we propose SimCIS, a simple yet powerful baseline for CIS. Its core idea is to directly select image features for query assignment, ensuring "perfect alignment" to preserve objectness, while simultaneously allowing queries to select new classes to promote plasticity. To further combat catastrophic forgetting of categories, we introduce cross-stage consistency in selection and an innovative "visual query"-based replay mechanism. Experiments demonstrate that SimCIS consistently outperforms state-of-the-art methods across various segmentation tasks, settings, splits, and input data orders. All models and codes will be made publicly available at https://github.com/SooLab/SimCIS.
>
---
#### [new 080] Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection
- **分类: cs.CV; I.4.0; I.4.9**

- **简介: 该论文属于关键点检测任务，解决少样本学习中缺乏源数据的问题。通过引入草图作为无源数据替代方案，提出框架实现跨模态嵌入与域适应，提升新关键点和类别的收敛性能。**

- **链接: [http://arxiv.org/pdf/2507.07994v1](http://arxiv.org/pdf/2507.07994v1)**

> **作者:** Subhajit Maity; Ayan Kumar Bhunia; Subhadeep Koley; Pinaki Nath Chowdhury; Aneeshan Sain; Yi-Zhe Song
>
> **备注:** Accepted at ICCV 2025. Project Page: https://subhajitmaity.me/DYKp
>
> **摘要:** Keypoint detection, integral to modern machine perception, faces challenges in few-shot learning, particularly when source data from the same distribution as the query is unavailable. This gap is addressed by leveraging sketches, a popular form of human expression, providing a source-free alternative. However, challenges arise in mastering cross-modal embeddings and handling user-specific sketch styles. Our proposed framework overcomes these hurdles with a prototypical setup, combined with a grid-based locator and prototypical domain adaptation. We also demonstrate success in few-shot convergence across novel keypoints and classes through extensive experiments.
>
---
#### [new 081] MeD-3D: A Multimodal Deep Learning Framework for Precise Recurrence Prediction in Clear Cell Renal Cell Carcinoma (ccRCC)
- **分类: cs.CV**

- **简介: 该论文属于医学预测任务，旨在解决ccRCC复发预测准确性不足的问题。通过融合多模态数据（影像、病理、临床、基因）构建深度学习框架，提升预测效果。**

- **链接: [http://arxiv.org/pdf/2507.07839v1](http://arxiv.org/pdf/2507.07839v1)**

> **作者:** Hasaan Maqsood; Saif Ur Rehman Khan
>
> **摘要:** Accurate prediction of recurrence in clear cell renal cell carcinoma (ccRCC) remains a major clinical challenge due to the disease complex molecular, pathological, and clinical heterogeneity. Traditional prognostic models, which rely on single data modalities such as radiology, histopathology, or genomics, often fail to capture the full spectrum of disease complexity, resulting in suboptimal predictive accuracy. This study aims to overcome these limitations by proposing a deep learning (DL) framework that integrates multimodal data, including CT, MRI, histopathology whole slide images (WSI), clinical data, and genomic profiles, to improve the prediction of ccRCC recurrence and enhance clinical decision-making. The proposed framework utilizes a comprehensive dataset curated from multiple publicly available sources, including TCGA, TCIA, and CPTAC. To process the diverse modalities, domain-specific models are employed: CLAM, a ResNet50-based model, is used for histopathology WSIs, while MeD-3D, a pre-trained 3D-ResNet18 model, processes CT and MRI images. For structured clinical and genomic data, a multi-layer perceptron (MLP) is used. These models are designed to extract deep feature embeddings from each modality, which are then fused through an early and late integration architecture. This fusion strategy enables the model to combine complementary information from multiple sources. Additionally, the framework is designed to handle incomplete data, a common challenge in clinical settings, by enabling inference even when certain modalities are missing.
>
---
#### [new 082] ArteryX: Advancing Brain Artery Feature Extraction with Vessel-Fused Networks and a Robust Validation Framework
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决脑动脉特征提取的准确性与标准化问题。提出ArteryX框架，结合半监督学习和验证机制，提升血管量化分析效率与灵敏度。**

- **链接: [http://arxiv.org/pdf/2507.07920v1](http://arxiv.org/pdf/2507.07920v1)**

> **作者:** Abrar Faiyaz; Nhat Hoang; Giovanni Schifitto; Md Nasir Uddin
>
> **备注:** 14 Pages, 8 Figures, Preliminary version of the toolbox was presented at the ISMRM 2025 Conference in Hawaii at the "Software Tools" Session
>
> **摘要:** Cerebrovascular pathology significantly contributes to cognitive decline and neurological disorders, underscoring the need for advanced tools to assess vascular integrity. Three-dimensional Time-of-Flight Magnetic Resonance Angiography (3D TOF MRA) is widely used to visualize cerebral vasculature, however, clinical evaluations generally focus on major arterial abnormalities, overlooking quantitative metrics critical for understanding subtle vascular changes. Existing methods for extracting structural, geometrical and morphological arterial features from MRA - whether manual or automated - face challenges including user-dependent variability, steep learning curves, and lack of standardized quantitative validations. We propose a novel semi-supervised artery evaluation framework, named ArteryX, a MATLAB-based toolbox that quantifies vascular features with high accuracy and efficiency, achieving processing times ~10-15 minutes per subject at 0.5 mm resolution with minimal user intervention. ArteryX employs a vessel-fused network based landmarking approach to reliably track and manage tracings, effectively addressing the issue of dangling/disconnected vessels. Validation on human subjects with cerebral small vessel disease demonstrated its improved sensitivity to subtle vascular changes and better performance than an existing semi-automated method. Importantly, the ArteryX toolbox enables quantitative feature validation by integrating an in-vivo like artery simulation framework utilizing vessel-fused graph nodes and predefined ground-truth features for specific artery types. Thus, the ArteryX framework holds promise for benchmarking feature extraction toolboxes and for seamless integration into clinical workflows, enabling early detection of cerebrovascular pathology and standardized comparisons across patient cohorts to advance understanding of vascular contributions to brain health.
>
---
#### [new 083] MagiC: Evaluating Multimodal Cognition Toward Grounded Visual Reasoning
- **分类: cs.CV**

- **简介: 该论文属于视觉推理任务，旨在评估模型是否具备真正的多模态认知能力。通过构建MagiC基准，分析模型在答案准确性、推理有效性等方面的性能，揭示当前方法的局限与改进方向。**

- **链接: [http://arxiv.org/pdf/2507.07297v1](http://arxiv.org/pdf/2507.07297v1)**

> **作者:** Chengfei Wu; Ronald Seoh; Bingxuan Li; Liqiang Zhang; Fengrong Han; Dan Goldwasser
>
> **摘要:** Recent advances in large vision-language models have led to impressive performance in visual question answering and multimodal reasoning. However, it remains unclear whether these models genuinely perform grounded visual reasoning or rely on superficial patterns and dataset biases. In this work, we introduce MagiC, a comprehensive benchmark designed to evaluate grounded multimodal cognition, assessing not only answer accuracy but also the quality of step-by-step reasoning and its alignment with relevant visual evidence. Our benchmark includes approximately 5,500 weakly supervised QA examples generated from strong model outputs and 900 human-curated examples with fine-grained annotations, including answers, rationales, and bounding box groundings. We evaluate 15 vision-language models ranging from 7B to 70B parameters across four dimensions: final answer correctness, reasoning validity, grounding fidelity, and self-correction ability. MagiC further includes diagnostic settings to probe model robustness under adversarial visual cues and assess their capacity for introspective error correction. We introduce new metrics such as MagiScore and StepSense, and provide comprehensive analyses that reveal key limitations and opportunities in current approaches to grounded visual reasoning.
>
---
#### [new 084] Breast Ultrasound Tumor Generation via Mask Generator and Text-Guided Network:A Clinically Controllable Framework with Downstream Evaluation
- **分类: cs.CV**

- **简介: 该论文属于医学图像生成任务，旨在解决乳腺超声数据不足的问题。通过结合临床描述和结构掩码生成肿瘤图像，提升诊断效果。**

- **链接: [http://arxiv.org/pdf/2507.07721v1](http://arxiv.org/pdf/2507.07721v1)**

> **作者:** Haoyu Pan; Hongxin Lin; Zetian Feng; Chuxuan Lin; Junyang Mo; Chu Zhang; Zijian Wu; Yi Wang; Qingqing Zheng
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** The development of robust deep learning models for breast ultrasound (BUS) image analysis is significantly constrained by the scarcity of expert-annotated data. To address this limitation, we propose a clinically controllable generative framework for synthesizing BUS images. This framework integrates clinical descriptions with structural masks to generate tumors, enabling fine-grained control over tumor characteristics such as morphology, echogencity, and shape. Furthermore, we design a semantic-curvature mask generator, which synthesizes structurally diverse tumor masks guided by clinical priors. During inference, synthetic tumor masks serve as input to the generative framework, producing highly personalized synthetic BUS images with tumors that reflect real-world morphological diversity. Quantitative evaluations on six public BUS datasets demonstrate the significant clinical utility of our synthetic images, showing their effectiveness in enhancing downstream breast cancer diagnosis tasks. Furthermore, visual Turing tests conducted by experienced sonographers confirm the realism of the generated images, indicating the framework's potential to support broader clinical applications.
>
---
#### [new 085] ViLU: Learning Vision-Language Uncertainties for Failure Prediction
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言任务，解决VLMs的不确定性量化与故障预测问题。提出ViLU框架，通过多模态表示和二分类器实现准确的不确定性估计。**

- **链接: [http://arxiv.org/pdf/2507.07620v1](http://arxiv.org/pdf/2507.07620v1)**

> **作者:** Marc Lafon; Yannis Karmim; Julio Silva-Rodriguez; Paul Couairon; Clément Rambour; Raphaël Fournier-Sniehotta; Ismail Ben Ayed; Jose Dolz; Nicolas Thome
>
> **摘要:** Reliable Uncertainty Quantification (UQ) and failure prediction remain open challenges for Vision-Language Models (VLMs). We introduce ViLU, a new Vision-Language Uncertainty quantification framework that contextualizes uncertainty estimates by leveraging all task-relevant textual representations. ViLU constructs an uncertainty-aware multi-modal representation by integrating the visual embedding, the predicted textual embedding, and an image-conditioned textual representation via cross-attention. Unlike traditional UQ methods based on loss prediction, ViLU trains an uncertainty predictor as a binary classifier to distinguish correct from incorrect predictions using a weighted binary cross-entropy loss, making it loss-agnostic. In particular, our proposed approach is well-suited for post-hoc settings, where only vision and text embeddings are available without direct access to the model itself. Extensive experiments on diverse datasets show the significant gains of our method compared to state-of-the-art failure prediction methods. We apply our method to standard classification datasets, such as ImageNet-1k, as well as large-scale image-caption datasets like CC12M and LAION-400M. Ablation studies highlight the critical role of our architecture and training in achieving effective uncertainty quantification. Our code is publicly available and can be found here: https://github.com/ykrmm/ViLU.
>
---
#### [new 086] PacGDC: Label-Efficient Generalizable Depth Completion with Projection Ambiguity and Consistency
- **分类: cs.CV**

- **简介: 该论文属于深度补全任务，旨在减少对标注数据的依赖。通过利用投影歧义与一致性，合成多样伪几何数据，提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.07374v1](http://arxiv.org/pdf/2507.07374v1)**

> **作者:** Haotian Wang; Aoran Xiao; Xiaoqin Zhang; Meng Yang; Shijian Lu
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Generalizable depth completion enables the acquisition of dense metric depth maps for unseen environments, offering robust perception capabilities for various downstream tasks. However, training such models typically requires large-scale datasets with metric depth labels, which are often labor-intensive to collect. This paper presents PacGDC, a label-efficient technique that enhances data diversity with minimal annotation effort for generalizable depth completion. PacGDC builds on novel insights into inherent ambiguities and consistencies in object shapes and positions during 2D-to-3D projection, allowing the synthesis of numerous pseudo geometries for the same visual scene. This process greatly broadens available geometries by manipulating scene scales of the corresponding depth maps. To leverage this property, we propose a new data synthesis pipeline that uses multiple depth foundation models as scale manipulators. These models robustly provide pseudo depth labels with varied scene scales, affecting both local objects and global layouts, while ensuring projection consistency that supports generalization. To further diversify geometries, we incorporate interpolation and relocation strategies, as well as unlabeled images, extending the data coverage beyond the individual use of foundation models. Extensive experiments show that PacGDC achieves remarkable generalizability across multiple benchmarks, excelling in diverse scene semantics/scales and depth sparsity/patterns under both zero-shot and few-shot settings. Code: https://github.com/Wang-xjtu/PacGDC.
>
---
#### [new 087] Degradation-Agnostic Statistical Facial Feature Transformation for Blind Face Restoration in Adverse Weather Conditions
- **分类: cs.CV**

- **简介: 该论文属于盲人脸修复任务，旨在解决恶劣天气下图像质量下降导致的识别准确率问题。提出SFFT和DAFE模块以提升修复效果。**

- **链接: [http://arxiv.org/pdf/2507.07464v1](http://arxiv.org/pdf/2507.07464v1)**

> **作者:** Chang-Hwan Son
>
> **摘要:** With the increasing deployment of intelligent CCTV systems in outdoor environments, there is a growing demand for face recognition systems optimized for challenging weather conditions. Adverse weather significantly degrades image quality, which in turn reduces recognition accuracy. Although recent face image restoration (FIR) models based on generative adversarial networks (GANs) and diffusion models have shown progress, their performance remains limited due to the lack of dedicated modules that explicitly address weather-induced degradations. This leads to distorted facial textures and structures. To address these limitations, we propose a novel GAN-based blind FIR framework that integrates two key components: local Statistical Facial Feature Transformation (SFFT) and Degradation-Agnostic Feature Embedding (DAFE). The local SFFT module enhances facial structure and color fidelity by aligning the local statistical distributions of low-quality (LQ) facial regions with those of high-quality (HQ) counterparts. Complementarily, the DAFE module enables robust statistical facial feature extraction under adverse weather conditions by aligning LQ and HQ encoder representations, thereby making the restoration process adaptive to severe weather-induced degradations. Experimental results demonstrate that the proposed degradation-agnostic SFFT model outperforms existing state-of-the-art FIR methods based on GAN and diffusion models, particularly in suppressing texture distortions and accurately reconstructing facial structures. Furthermore, both the SFFT and DAFE modules are empirically validated in enhancing structural fidelity and perceptual quality in face restoration under challenging weather scenarios.
>
---
#### [new 088] 3D-ADAM: A Dataset for 3D Anomaly Detection in Advanced Manufacturing
- **分类: cs.CV**

- **简介: 该论文属于3D异常检测任务，旨在解决工业制造中表面缺陷检测难题。工作包括构建3D-ADAM数据集，包含大量真实场景下的缺陷和机械特征标注，以推动更鲁棒的检测模型发展。**

- **链接: [http://arxiv.org/pdf/2507.07838v1](http://arxiv.org/pdf/2507.07838v1)**

> **作者:** Paul McHard; Florent P. Audonnet; Oliver Summerell; Sebastian Andraos; Paul Henderson; Gerardo Aragon-Camarasa
>
> **摘要:** Surface defects are one of the largest contributors to low yield in the manufacturing sector. Accurate and reliable detection of defects during the manufacturing process is therefore of great value across the sector. State-of-the-art approaches to automated defect detection yield impressive performance on current datasets, yet still fall short in real-world manufacturing settings and developing improved methods relies on large datasets representative of real-world scenarios. Unfortunately, high-quality, high-precision RGB+3D industrial anomaly detection datasets are scarce, and typically do not reflect real-world industrial deployment scenarios. To address this, we introduce 3D-ADAM, the first large-scale industry-relevant dataset for high-precision 3D Anomaly Detection. 3D-ADAM comprises 14,120 high-resolution scans across 217 unique parts, captured using 4 industrial depth imaging sensors. It includes 27,346 annotated defect instances from 12 categories, covering the breadth of industrial surface defects. 3D-ADAM uniquely captures an additional 8,110 annotations of machine element features, spanning the range of relevant mechanical design form factors. Unlike existing datasets, 3D-ADAM is captured in a real industrial environment with variations in part position and orientation, camera positioning, ambient lighting conditions, as well as partial occlusions. Our evaluation of SOTA models across various RGB+3D anomaly detection tasks demonstrates the significant challenge this dataset presents to current approaches. We further validated the industrial relevance and quality of the dataset through an expert labelling survey conducted by industry partners. By providing this challenging benchmark, 3D-ADAM aims to accelerate the development of robust 3D Anomaly Detection models capable of meeting the demands of modern manufacturing environments.
>
---
#### [new 089] A Survey on Long-Video Storytelling Generation: Architectures, Consistency, and Cinematic Quality
- **分类: cs.CV**

- **简介: 该论文属于长视频生成任务，旨在解决长视频中角色一致性、场景连贯性和电影质量的问题。通过分析32篇论文，提出架构分类和性能比较。**

- **链接: [http://arxiv.org/pdf/2507.07202v1](http://arxiv.org/pdf/2507.07202v1)**

> **作者:** Mohamed Elmoghany; Ryan Rossi; Seunghyun Yoon; Subhojyoti Mukherjee; Eslam Bakr; Puneet Mathur; Gang Wu; Viet Dac Lai; Nedim Lipka; Ruiyi Zhang; Varun Manjunatha; Chien Nguyen; Daksh Dangi; Abel Salinas; Mohammad Taesiri; Hongjie Chen; Xiaolei Huang; Joe Barrow; Nesreen Ahmed; Hoda Eldardiry; Namyong Park; Yu Wang; Jaemin Cho; Anh Totti Nguyen; Zhengzhong Tu; Thien Nguyen; Dinesh Manocha; Mohamed Elhoseiny; Franck Dernoncourt
>
> **摘要:** Despite the significant progress that has been made in video generative models, existing state-of-the-art methods can only produce videos lasting 5-16 seconds, often labeled "long-form videos". Furthermore, videos exceeding 16 seconds struggle to maintain consistent character appearances and scene layouts throughout the narrative. In particular, multi-subject long videos still fail to preserve character consistency and motion coherence. While some methods can generate videos up to 150 seconds long, they often suffer from frame redundancy and low temporal diversity. Recent work has attempted to produce long-form videos featuring multiple characters, narrative coherence, and high-fidelity detail. We comprehensively studied 32 papers on video generation to identify key architectural components and training strategies that consistently yield these qualities. We also construct a comprehensive novel taxonomy of existing methods and present comparative tables that categorize papers by their architectural designs and performance characteristics.
>
---
#### [new 090] Temporal Unlearnable Examples: Preventing Personal Video Data from Unauthorized Exploitation by Object Tracking
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于视频数据隐私保护任务，旨在防止个人视频被未经授权使用。通过生成时间不可学习样本（TUEs）和引入时序对比损失，提升模型对隐私数据的保护效果。**

- **链接: [http://arxiv.org/pdf/2507.07483v1](http://arxiv.org/pdf/2507.07483v1)**

> **作者:** Qiangqiang Wu; Yi Yu; Chenqi Kong; Ziquan Liu; Jia Wan; Haoliang Li; Alex C. Kot; Antoni B. Chan
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** With the rise of social media, vast amounts of user-uploaded videos (e.g., YouTube) are utilized as training data for Visual Object Tracking (VOT). However, the VOT community has largely overlooked video data-privacy issues, as many private videos have been collected and used for training commercial models without authorization. To alleviate these issues, this paper presents the first investigation on preventing personal video data from unauthorized exploitation by deep trackers. Existing methods for preventing unauthorized data use primarily focus on image-based tasks (e.g., image classification), directly applying them to videos reveals several limitations, including inefficiency, limited effectiveness, and poor generalizability. To address these issues, we propose a novel generative framework for generating Temporal Unlearnable Examples (TUEs), and whose efficient computation makes it scalable for usage on large-scale video datasets. The trackers trained w/ TUEs heavily rely on unlearnable noises for temporal matching, ignoring the original data structure and thus ensuring training video data-privacy. To enhance the effectiveness of TUEs, we introduce a temporal contrastive loss, which further corrupts the learning of existing trackers when using our TUEs for training. Extensive experiments demonstrate that our approach achieves state-of-the-art performance in video data-privacy protection, with strong transferability across VOT models, datasets, and temporal matching tasks.
>
---
#### [new 091] RAPS-3D: Efficient interactive segmentation for 3D radiological imaging
- **分类: cs.CV**

- **简介: 该论文属于3D医学图像分割任务，旨在解决2D模型难以直接应用于3D数据的问题，提出一种高效交互式分割方法。**

- **链接: [http://arxiv.org/pdf/2507.07730v1](http://arxiv.org/pdf/2507.07730v1)**

> **作者:** Théo Danielou; Daniel Tordjman; Pierre Manceron; Corentin Dancette
>
> **备注:** Abstract accepted at MIUA 2025
>
> **摘要:** Promptable segmentation, introduced by the Segment Anything Model (SAM), is a promising approach for medical imaging, as it enables clinicians to guide and refine model predictions interactively. However, SAM's architecture is designed for 2D images and does not extend naturally to 3D volumetric data such as CT or MRI scans. Adapting 2D models to 3D typically involves autoregressive strategies, where predictions are propagated slice by slice, resulting in increased inference complexity. Processing large 3D volumes also requires significant computational resources, often leading existing 3D methods to also adopt complex strategies like sliding-window inference to manage memory usage, at the cost of longer inference times and greater implementation complexity. In this paper, we present a simplified 3D promptable segmentation method, inspired by SegVol, designed to reduce inference time and eliminate prompt management complexities associated with sliding windows while achieving state-of-the-art performance.
>
---
#### [new 092] MAPEX: Modality-Aware Pruning of Experts for Remote Sensing Foundation Models
- **分类: cs.CV**

- **简介: 该论文属于遥感任务，解决预训练模型与应用模态不匹配及模型过大问题。提出MAPEX模型，通过模态感知剪枝实现高效任务适配。**

- **链接: [http://arxiv.org/pdf/2507.07527v1](http://arxiv.org/pdf/2507.07527v1)**

> **作者:** Joelle Hanna; Linus Scheibenreif; Damian Borth
>
> **摘要:** Remote sensing data is commonly used for tasks such as flood mapping, wildfire detection, or land-use studies. For each task, scientists carefully choose appropriate modalities or leverage data from purpose-built instruments. Recent work on remote sensing foundation models pre-trains computer vision models on large amounts of remote sensing data. These large-scale models tend to focus on specific modalities, often optical RGB or multispectral data. For many important applications, this introduces a mismatch between the application modalities and the pre-training data. Moreover, the large size of foundation models makes them expensive and difficult to fine-tune on typically small datasets for each task. We address this mismatch with MAPEX, a remote sensing foundation model based on mixture-of-modality experts. MAPEX is pre-trained on multi-modal remote sensing data with a novel modality-conditioned token routing mechanism that elicits modality-specific experts. To apply the model on a specific task, we propose a modality aware pruning technique, which only retains experts specialized for the task modalities. This yields efficient modality-specific models while simplifying fine-tuning and deployment for the modalities of interest. We experimentally validate MAPEX on diverse remote sensing datasets and show strong performance compared to fully supervised training and state-of-the-art remote sensing foundation models. Code is available at https://github.com/HSG-AIML/MAPEX.
>
---
#### [new 093] Divergence Minimization Preference Optimization for Diffusion Model Alignment
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于扩散模型对齐任务，解决现有方法陷入次优优化的问题，提出DMPO方法通过最小化反向KL散度实现更优对齐。**

- **链接: [http://arxiv.org/pdf/2507.07510v1](http://arxiv.org/pdf/2507.07510v1)**

> **作者:** Binxu Li; Minkai Xu; Meihua Dang; Stefano Ermon
>
> **备注:** 24 pages, 8 figures
>
> **摘要:** Diffusion models have achieved remarkable success in generating realistic and versatile images from text prompts. Inspired by the recent advancements of language models, there is an increasing interest in further improving the models by aligning with human preferences. However, we investigate alignment from a divergence minimization perspective and reveal that existing preference optimization methods are typically trapped in suboptimal mean-seeking optimization. In this paper, we introduce Divergence Minimization Preference Optimization (DMPO), a novel and principled method for aligning diffusion models by minimizing reverse KL divergence, which asymptotically enjoys the same optimization direction as original RL. We provide rigorous analysis to justify the effectiveness of DMPO and conduct comprehensive experiments to validate its empirical strength across both human evaluations and automatic metrics. Our extensive results show that diffusion models fine-tuned with DMPO can consistently outperform or match existing techniques, specifically outperforming all existing diffusion alignment baselines by at least 64.6% in PickScore across all evaluation datasets, demonstrating the method's superiority in aligning generative behavior with desired outputs. Overall, DMPO unlocks a robust and elegant pathway for preference alignment, bridging principled theory with practical performance in diffusion models.
>
---
#### [new 094] GGMotion: Group Graph Dynamics-Kinematics Networks for Human Motion Prediction
- **分类: cs.CV**

- **简介: 该论文属于人体运动预测任务，旨在解决现有方法忽略关节物理依赖的问题。提出GGMotion模型，通过分组图结构和时空依赖建模提升运动合理性。**

- **链接: [http://arxiv.org/pdf/2507.07515v1](http://arxiv.org/pdf/2507.07515v1)**

> **作者:** Shuaijin Wan; Huaijiang Sun
>
> **摘要:** Human motion is a continuous physical process in 3D space, governed by complex dynamic and kinematic constraints. Existing methods typically represent the human pose as an abstract graph structure, neglecting the intrinsic physical dependencies between joints, which increases learning difficulty and makes the model prone to generating unrealistic motions. In this paper, we propose GGMotion, a group graph dynamics-kinematics network that models human topology in groups to better leverage dynamics and kinematics priors. To preserve the geometric equivariance in 3D space, we propose a novel radial field for the graph network that captures more comprehensive spatio-temporal dependencies by aggregating joint features through spatial and temporal edges. Inter-group and intra-group interaction modules are employed to capture the dependencies of joints at different scales. Combined with equivariant multilayer perceptrons (MLP), joint position features are updated in each group through parallelized dynamics-kinematics propagation to improve physical plausibility. Meanwhile, we introduce an auxiliary loss to supervise motion priors during training. Extensive experiments on three standard benchmarks, including Human3.6M, CMU-Mocap, and 3DPW, demonstrate the effectiveness and superiority of our approach, achieving a significant performance margin in short-term motion prediction. The code is available at https://github.com/inkcat520/GGMotion.git.
>
---
#### [new 095] Benchmarking Content-Based Puzzle Solvers on Corrupted Jigsaw Puzzles
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像拼图任务，研究内容-based拼图求解器在损坏拼图中的鲁棒性，分析不同损坏类型对性能的影响，并提出改进方法。**

- **链接: [http://arxiv.org/pdf/2507.07828v1](http://arxiv.org/pdf/2507.07828v1)**

> **作者:** Richard Dirauf; Florian Wolz; Dario Zanca; Björn Eskofier
>
> **备注:** Accepted at ICIAP 2025
>
> **摘要:** Content-based puzzle solvers have been extensively studied, demonstrating significant progress in computational techniques. However, their evaluation often lacks realistic challenges crucial for real-world applications, such as the reassembly of fragmented artefacts or shredded documents. In this work, we investigate the robustness of State-Of-The-Art content-based puzzle solvers introducing three types of jigsaw puzzle corruptions: missing pieces, eroded edges, and eroded contents. Evaluating both heuristic and deep learning-based solvers, we analyse their ability to handle these corruptions and identify key limitations. Our results show that solvers developed for standard puzzles have a rapid decline in performance if more pieces are corrupted. However, deep learning models can significantly improve their robustness through fine-tuning with augmented data. Notably, the advanced Positional Diffusion model adapts particularly well, outperforming its competitors in most experiments. Based on our findings, we highlight promising research directions for enhancing the automated reconstruction of real-world artefacts.
>
---
#### [new 096] CoPT: Unsupervised Domain Adaptive Segmentation using Domain-Agnostic Text Embeddings
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于无监督域适应任务，解决分割模型在不同域间泛化的问题。通过引入领域无关的文本嵌入，提出CoPT损失函数，提升模型域不变特征学习能力。**

- **链接: [http://arxiv.org/pdf/2507.07125v1](http://arxiv.org/pdf/2507.07125v1)**

> **作者:** Cristina Mata; Kanchana Ranasinghe; Michael S. Ryoo
>
> **备注:** ECCV 2024
>
> **摘要:** Unsupervised domain adaptation (UDA) involves learning class semantics from labeled data within a source domain that generalize to an unseen target domain. UDA methods are particularly impactful for semantic segmentation, where annotations are more difficult to collect than in image classification. Despite recent advances in large-scale vision-language representation learning, UDA methods for segmentation have not taken advantage of the domain-agnostic properties of text. To address this, we present a novel Covariance-based Pixel-Text loss, CoPT, that uses domain-agnostic text embeddings to learn domain-invariant features in an image segmentation encoder. The text embeddings are generated through our LLM Domain Template process, where an LLM is used to generate source and target domain descriptions that are fed to a frozen CLIP model and combined. In experiments on four benchmarks we show that a model trained using CoPT achieves the new state of the art performance on UDA for segmentation. The code can be found at https://github.com/cfmata/CoPT.
>
---
#### [new 097] LOSC: LiDAR Open-voc Segmentation Consolidator
- **分类: cs.CV**

- **简介: 该论文属于开放词汇点云分割任务，解决传统方法标签噪声和稀疏的问题，通过融合视觉语言模型提升分割精度。**

- **链接: [http://arxiv.org/pdf/2507.07605v1](http://arxiv.org/pdf/2507.07605v1)**

> **作者:** Nermin Samet; Gilles Puy; Renaud Marlet
>
> **摘要:** We study the use of image-based Vision-Language Models (VLMs) for open-vocabulary segmentation of lidar scans in driving settings. Classically, image semantics can be back-projected onto 3D point clouds. Yet, resulting point labels are noisy and sparse. We consolidate these labels to enforce both spatio-temporal consistency and robustness to image-level augmentations. We then train a 3D network based on these refined labels. This simple method, called LOSC, outperforms the SOTA of zero-shot open-vocabulary semantic and panoptic segmentation on both nuScenes and SemanticKITTI, with significant margins.
>
---
#### [new 098] LinguaMark: Do Multimodal Models Speak Fairly? A Benchmark-Based Evaluation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态任务，旨在评估LMMs在多语言VQA中的公平性与性能，解决模型语言偏见问题，通过构建LinguaMark基准进行实验分析。**

- **链接: [http://arxiv.org/pdf/2507.07274v1](http://arxiv.org/pdf/2507.07274v1)**

> **作者:** Ananya Raval; Aravind Narayanan; Vahid Reza Khazaie; Shaina Raza
>
> **备注:** Accepted at ASONAM'25
>
> **摘要:** Large Multimodal Models (LMMs) are typically trained on vast corpora of image-text data but are often limited in linguistic coverage, leading to biased and unfair outputs across languages. While prior work has explored multimodal evaluation, less emphasis has been placed on assessing multilingual capabilities. In this work, we introduce LinguaMark, a benchmark designed to evaluate state-of-the-art LMMs on a multilingual Visual Question Answering (VQA) task. Our dataset comprises 6,875 image-text pairs spanning 11 languages and five social attributes. We evaluate models using three key metrics: Bias, Answer Relevancy, and Faithfulness. Our findings reveal that closed-source models generally achieve the highest overall performance. Both closed-source (GPT-4o and Gemini2.5) and open-source models (Gemma3, Qwen2.5) perform competitively across social attributes, and Qwen2.5 demonstrates strong generalization across multiple languages. We release our benchmark and evaluation code to encourage reproducibility and further research.
>
---
#### [new 099] Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Translation
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于文档图像机器翻译任务，解决数据有限和视觉文本交互复杂的问题。提出M4Doc框架，利用多模态大语言模型提升翻译质量与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.07572v1](http://arxiv.org/pdf/2507.07572v1)**

> **作者:** Yupu Liang; Yaping Zhang; Zhiyang Zhang; Yang Zhao; Lu Xiang; Chengqing Zong; Yu Zhou
>
> **备注:** Accepted by ACL 2025 Main
>
> **摘要:** Document Image Machine Translation (DIMT) aims to translate text within document images, facing generalization challenges due to limited training data and the complex interplay between visual and textual information. To address these challenges, we introduce M4Doc, a novel single-to-mix modality alignment framework leveraging Multimodal Large Language Models (MLLMs). M4Doc aligns an image-only encoder with the multimodal representations of an MLLM, pre-trained on large-scale document image datasets. This alignment enables a lightweight DIMT model to learn crucial visual-textual correlations during training. During inference, M4Doc bypasses the MLLM, maintaining computational efficiency while benefiting from its multimodal knowledge. Comprehensive experiments demonstrate substantial improvements in translation quality, especially in cross-domain generalization and challenging document image scenarios.
>
---
#### [new 100] SD-GS: Structured Deformable 3D Gaussians for Efficient Dynamic Scene Reconstruction
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于动态场景重建任务，解决存储成本与复杂运动表征之间的矛盾。提出SD-GS框架，通过可变形锚点网格和自适应密度策略，提升效率与质量。**

- **链接: [http://arxiv.org/pdf/2507.07465v1](http://arxiv.org/pdf/2507.07465v1)**

> **作者:** Wei Yao; Shuzhao Xie; Letian Li; Weixiang Zhang; Zhixin Lai; Shiqi Dai; Ke Zhang; Zhi Wang
>
> **摘要:** Current 4D Gaussian frameworks for dynamic scene reconstruction deliver impressive visual fidelity and rendering speed, however, the inherent trade-off between storage costs and the ability to characterize complex physical motions significantly limits the practical application of these methods. To tackle these problems, we propose SD-GS, a compact and efficient dynamic Gaussian splatting framework for complex dynamic scene reconstruction, featuring two key contributions. First, we introduce a deformable anchor grid, a hierarchical and memory-efficient scene representation where each anchor point derives multiple 3D Gaussians in its local spatiotemporal region and serves as the geometric backbone of the 3D scene. Second, to enhance modeling capability for complex motions, we present a deformation-aware densification strategy that adaptively grows anchors in under-reconstructed high-dynamic regions while reducing redundancy in static areas, achieving superior visual quality with fewer anchors. Experimental results demonstrate that, compared to state-of-the-art methods, SD-GS achieves an average of 60\% reduction in model size and an average of 100\% improvement in FPS, significantly enhancing computational efficiency while maintaining or even surpassing visual quality.
>
---
#### [new 101] Balancing the Past and Present: A Coordinated Replay Framework for Federated Class-Incremental Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于联邦持续学习任务，解决类不平衡问题。提出FedCBDR方法，通过全局协调和温度调整提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.07712v1](http://arxiv.org/pdf/2507.07712v1)**

> **作者:** Zhuang Qi; Lei Meng; Han Yu
>
> **摘要:** Federated Class Incremental Learning (FCIL) aims to collaboratively process continuously increasing incoming tasks across multiple clients. Among various approaches, data replay has become a promising solution, which can alleviate forgetting by reintroducing representative samples from previous tasks. However, their performance is typically limited by class imbalance, both within the replay buffer due to limited global awareness and between replayed and newly arrived classes. To address this issue, we propose a class wise balancing data replay method for FCIL (FedCBDR), which employs a global coordination mechanism for class-level memory construction and reweights the learning objective to alleviate the aforementioned imbalances. Specifically, FedCBDR has two key components: 1) the global-perspective data replay module reconstructs global representations of prior task in a privacy-preserving manner, which then guides a class-aware and importance-sensitive sampling strategy to achieve balanced replay; 2) Subsequently, to handle class imbalance across tasks, the task aware temperature scaling module adaptively adjusts the temperature of logits at both class and instance levels based on task dynamics, which reduces the model's overconfidence in majority classes while enhancing its sensitivity to minority classes. Experimental results verified that FedCBDR achieves balanced class-wise sampling under heterogeneous data distributions and improves generalization under task imbalance between earlier and recent tasks, yielding a 2%-15% Top-1 accuracy improvement over six state-of-the-art methods.
>
---
#### [new 102] Adaptive Attention Residual U-Net for curvilinear structure segmentation in fluorescence microscopy and biomedical images
- **分类: q-bio.QM; cs.CV**

- **简介: 该论文属于生物医学图像分割任务，旨在解决荧光显微镜中曲线结构的分割难题。通过构建数据集并提出新型网络ASE_Res_UNet，提升噪声和低对比条件下的分割性能。**

- **链接: [http://arxiv.org/pdf/2507.07800v1](http://arxiv.org/pdf/2507.07800v1)**

> **作者:** Achraf Ait Laydi; Louis Cueff; Mewen Crespo; Yousef El Mourabit; Hélène Bouvrais
>
> **摘要:** Segmenting curvilinear structures in fluorescence microscopy remains a challenging task, particularly under noisy conditions and in dense filament networks commonly seen in vivo. To address this, we created two original datasets consisting of hundreds of synthetic images of fluorescently labelled microtubules within cells. These datasets are precisely annotated and closely mimic real microscopy images, including realistic noise. The second dataset presents an additional challenge, by simulating varying fluorescence intensities along filaments that complicate segmentation. While deep learning has shown strong potential in biomedical image analysis, its performance often declines in noisy or low-contrast conditions. To overcome this limitation, we developed a novel advanced architecture: the Adaptive Squeeze-and-Excitation Residual U-Net (ASE_Res_UNet). This model enhanced the standard U-Net by integrating residual blocks in the encoder and adaptive SE attention mechanisms in the decoder. Through ablation studies and comprehensive visual and quantitative evaluations, ASE_Res_UNet consistently outperformed its variants, namely standard U-Net, ASE_UNet and Res_UNet architectures. These improvements, particularly in noise resilience and detecting fine, low-intensity structures, were largely attributed to the adaptive SE attention module that we created. We further benchmarked ASE_Res_UNet against various state-of-the-art models, and found it achieved superior performance on our most challenging dataset. Finally, the model also generalized well to real microscopy images of stained microtubules as well as to other curvilinear structures. Indeed, it successfully segmented retinal blood vessels and nerves in noisy or low-contrast biomedical images, demonstrating its strong potential for applications in disease diagnosis and treatment.
>
---
#### [new 103] LangNavBench: Evaluation of Natural Language Understanding in Semantic Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于语义导航任务，旨在解决语言理解与物体定位的问题。构建了LangNav数据集和LangNavBench基准，评估导航系统对语言指令的解析能力。提出MLFM方法提升小物体和空间关系处理效果。**

- **链接: [http://arxiv.org/pdf/2507.07299v1](http://arxiv.org/pdf/2507.07299v1)**

> **作者:** Sonia Raychaudhuri; Enrico Cancelli; Tommaso Campari; Lamberto Ballan; Manolis Savva; Angel X. Chang
>
> **摘要:** Recent progress in large vision-language models has driven improvements in language-based semantic navigation, where an embodied agent must reach a target object described in natural language. Despite these advances, we still lack a clear, language-focused benchmark for testing how well such agents ground the words in their instructions. We address this gap with LangNav, an open-set dataset specifically created to test an agent's ability to locate objects described at different levels of detail, from broad category names to fine attributes and object-object relations. Every description in LangNav was manually checked, yielding a lower error rate than existing lifelong- and semantic-navigation datasets. On top of LangNav we build LangNavBench, a benchmark that measures how well current semantic-navigation methods understand and act on these descriptions while moving toward their targets. LangNavBench allows us to systematically compare models on their handling of attributes, spatial and relational cues, and category hierarchies, offering the first thorough, language-centric evaluation of embodied navigation systems. We also present Multi-Layered Feature Map (MLFM), a method that builds a queryable multi-layered semantic map, particularly effective when dealing with small objects or instructions involving spatial relations. MLFM outperforms state-of-the-art mapping-based navigation baselines on the LangNav dataset.
>
---
#### [new 104] ST-GRIT: Spatio-Temporal Graph Transformer For Internal Ice Layer Thickness Prediction
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于冰层厚度预测任务，解决雷达图像中内部冰层结构分析问题。提出ST-GRIT模型，结合时空图注意力机制，提升预测精度。**

- **链接: [http://arxiv.org/pdf/2507.07389v1](http://arxiv.org/pdf/2507.07389v1)**

> **作者:** Zesheng Liu; Maryam Rahnemoonfar
>
> **备注:** Accepted for 2025 IEEE International Conference on Image Processing (ICIP)
>
> **摘要:** Understanding the thickness and variability of internal ice layers in radar imagery is crucial for monitoring snow accumulation, assessing ice dynamics, and reducing uncertainties in climate models. Radar sensors, capable of penetrating ice, provide detailed radargram images of these internal layers. In this work, we present ST-GRIT, a spatio-temporal graph transformer for ice layer thickness, designed to process these radargrams and capture the spatiotemporal relationships between shallow and deep ice layers. ST-GRIT leverages an inductive geometric graph learning framework to extract local spatial features as feature embeddings and employs a series of temporal and spatial attention blocks separately to model long-range dependencies effectively in both dimensions. Experimental evaluation on radargram data from the Greenland ice sheet demonstrates that ST-GRIT consistently outperforms current state-of-the-art methods and other baseline graph neural networks by achieving lower root mean-squared error. These results highlight the advantages of self-attention mechanisms on graphs over pure graph neural networks, including the ability to handle noise, avoid oversmoothing, and capture long-range dependencies. Moreover, the use of separate spatial and temporal attention blocks allows for distinct and robust learning of spatial relationships and temporal patterns, providing a more comprehensive and effective approach.
>
---
#### [new 105] Weighted Multi-Prompt Learning with Description-free Large Language Model Distillation
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于视觉语言模型任务，解决传统方法依赖描述提取导致的不稳定问题，提出无需描述的多提示学习方法，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.07147v1](http://arxiv.org/pdf/2507.07147v1)**

> **作者:** Sua Lee; Kyubum Shin; Jung Ho Park
>
> **备注:** Published as a conference paper at ICLR 2025
>
> **摘要:** Recent advances in pre-trained Vision Language Models (VLM) have shown promising potential for effectively adapting to downstream tasks through prompt learning, without the need for additional annotated paired datasets. To supplement the text information in VLM trained on correlations with vision data, new approaches leveraging Large Language Models (LLM) in prompts have been proposed, enhancing robustness to unseen and diverse data. Existing methods typically extract text-based responses (i.e., descriptions) from LLM to incorporate into prompts; however, this approach suffers from high variability and low reliability. In this work, we propose Description-free Multi-prompt Learning(DeMul), a novel method that eliminates the process of extracting descriptions and instead directly distills knowledge from LLM into prompts. By adopting a description-free approach, prompts can encapsulate richer semantics while still being represented as continuous vectors for optimization, thereby eliminating the need for discrete pre-defined templates. Additionally, in a multi-prompt setting, we empirically demonstrate the potential of prompt weighting in reflecting the importance of different prompts during training. Experimental results show that our approach achieves superior performance across 11 recognition datasets.
>
---
#### [new 106] mmFlux: Crowd Flow Analytics with Commodity mmWave MIMO Radar
- **分类: eess.SP; cs.CV**

- **简介: 该论文属于人群流动分析任务，旨在通过mmWave雷达提取人群运动模式和语义。工作包括生成高保真流场、构建几何图，并利用curl和divergence分析获取关键语义。**

- **链接: [http://arxiv.org/pdf/2507.07331v1](http://arxiv.org/pdf/2507.07331v1)**

> **作者:** Anurag Pallaprolu; Winston Hurst; Yasamin Mostofi
>
> **摘要:** In this paper, we present a novel framework for extracting underlying crowd motion patterns and inferring crowd semantics using mmWave radar. First, our proposed signal processing pipeline combines optical flow estimation concepts from vision with novel statistical and morphological noise filtering to generate high-fidelity mmWave flow fields - compact 2D vector representations of crowd motion. We then introduce a novel approach that transforms these fields into directed geometric graphs, where edges capture dominant flow currents, vertices mark crowd splitting or merging, and flow distribution is quantified across edges. Finally, we show that by analyzing the local Jacobian and computing the corresponding curl and divergence, we can extract key crowd semantics for both structured and diffused crowds. We conduct 21 experiments on crowds of up to (and including) 20 people across 3 areas, using commodity mmWave radar. Our framework achieves high-fidelity graph reconstruction of the underlying flow structure, even for complex crowd patterns, demonstrating strong spatial alignment and precise quantitative characterization of flow split ratios. Finally, our curl and divergence analysis accurately infers key crowd semantics, e.g., abrupt turns, boundaries where flow directions shift, dispersions, and gatherings. Overall, these findings validate our framework, underscoring its potential for various crowd analytics applications.
>
---
#### [new 107] TRIX- Trading Adversarial Fairness via Mixed Adversarial Training
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于对抗训练任务，旨在解决类别间鲁棒性不均的问题。通过自适应调整对抗样本，提升弱类别的鲁棒性，同时保持整体准确率。**

- **链接: [http://arxiv.org/pdf/2507.07768v1](http://arxiv.org/pdf/2507.07768v1)**

> **作者:** Tejaswini Medi; Steffen Jung; Margret Keuper
>
> **摘要:** Adversarial Training (AT) is a widely adopted defense against adversarial examples. However, existing approaches typically apply a uniform training objective across all classes, overlooking disparities in class-wise vulnerability. This results in adversarial unfairness: classes with well distinguishable features (strong classes) tend to become more robust, while classes with overlapping or shared features(weak classes) remain disproportionately susceptible to adversarial attacks. We observe that strong classes do not require strong adversaries during training, as their non-robust features are quickly suppressed. In contrast, weak classes benefit from stronger adversaries to effectively reduce their vulnerabilities. Motivated by this, we introduce TRIX, a feature-aware adversarial training framework that adaptively assigns weaker targeted adversaries to strong classes, promoting feature diversity via uniformly sampled targets, and stronger untargeted adversaries to weak classes, enhancing their focused robustness. TRIX further incorporates per-class loss weighting and perturbation strength adjustments, building on prior work, to emphasize weak classes during the optimization. Comprehensive experiments on standard image classification benchmarks, including evaluations under strong attacks such as PGD and AutoAttack, demonstrate that TRIX significantly improves worst-case class accuracy on both clean and adversarial data, reducing inter-class robustness disparities, and preserves overall accuracy. Our results highlight TRIX as a practical step toward fair and effective adversarial defense.
>
---
#### [new 108] RTR-GS: 3D Gaussian Splatting for Inverse Rendering with Radiance Transfer and Reflection
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于逆渲染任务，旨在解决反射物体的高质量重建与光照重演问题。通过结合正向与延迟渲染，RTR-GS有效分离光照与材质，提升视图合成与重光照效果。**

- **链接: [http://arxiv.org/pdf/2507.07733v1](http://arxiv.org/pdf/2507.07733v1)**

> **作者:** Yongyang Zhou; Fang-Lue Zhang; Zichen Wang; Lei Zhang
>
> **备注:** 16 pages
>
> **摘要:** 3D Gaussian Splatting (3DGS) has demonstrated impressive capabilities in novel view synthesis. However, rendering reflective objects remains a significant challenge, particularly in inverse rendering and relighting. We introduce RTR-GS, a novel inverse rendering framework capable of robustly rendering objects with arbitrary reflectance properties, decomposing BRDF and lighting, and delivering credible relighting results. Given a collection of multi-view images, our method effectively recovers geometric structure through a hybrid rendering model that combines forward rendering for radiance transfer with deferred rendering for reflections. This approach successfully separates high-frequency and low-frequency appearances, mitigating floating artifacts caused by spherical harmonic overfitting when handling high-frequency details. We further refine BRDF and lighting decomposition using an additional physically-based deferred rendering branch. Experimental results show that our method enhances novel view synthesis, normal estimation, decomposition, and relighting while maintaining efficient training inference process.
>
---
#### [new 109] Label-Efficient Chest X-ray Diagnosis via Partial CLIP Adaptation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像诊断任务，旨在解决标签数据稀缺的问题。通过部分微调CLIP模型，在少量标注数据下提升胸片诊断效果。**

- **链接: [http://arxiv.org/pdf/2507.07254v1](http://arxiv.org/pdf/2507.07254v1)**

> **作者:** Heet Nitinkumar Dalsania
>
> **摘要:** Modern deep learning implementations for medical imaging usually rely on large labeled datasets. These datasets are often difficult to obtain due to privacy concerns, high costs, and even scarcity of cases. In this paper, a label-efficient strategy is proposed for chest X-ray diagnosis that seeks to reflect real-world hospital scenarios. The experiments use the NIH Chest X-ray14 dataset and a pre-trained CLIP ViT-B/32 model. The model is adapted via partial fine-tuning of its visual encoder and then evaluated using zero-shot and few-shot learning with 1-16 labeled examples per disease class. The tests demonstrate that CLIP's pre-trained vision-language features can be effectively adapted to few-shot medical imaging tasks, achieving over 20\% improvement in mean AUC score as compared to the zero-shot baseline. The key aspect of this work is to attempt to simulate internal hospital workflows, where image archives exist but annotations are sparse. This work evaluates a practical and scalable solution for both common and rare disease diagnosis. Additionally this research is intended for academic and experimental purposes only and has not been peer reviewed yet. All code is found at https://github.com/heet007-code/CLIP-disease-xray.
>
---
#### [new 110] Synchronizing Task Behavior: Aligning Multiple Tasks during Test-Time Training
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于多任务学习领域，解决域适应中任务不同步的问题。提出S4T方法，同步多个任务的测试时训练过程。**

- **链接: [http://arxiv.org/pdf/2507.07778v1](http://arxiv.org/pdf/2507.07778v1)**

> **作者:** Wooseong Jeong; Jegyeong Cho; Youngho Yoon; Kuk-Jin Yoon
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Generalizing neural networks to unseen target domains is a significant challenge in real-world deployments. Test-time training (TTT) addresses this by using an auxiliary self-supervised task to reduce the domain gap caused by distribution shifts between the source and target. However, we find that when models are required to perform multiple tasks under domain shifts, conventional TTT methods suffer from unsynchronized task behavior, where the adaptation steps needed for optimal performance in one task may not align with the requirements of other tasks. To address this, we propose a novel TTT approach called Synchronizing Tasks for Test-time Training (S4T), which enables the concurrent handling of multiple tasks. The core idea behind S4T is that predicting task relations across domain shifts is key to synchronizing tasks during test time. To validate our approach, we apply S4T to conventional multi-task benchmarks, integrating it with traditional TTT protocols. Our empirical results show that S4T outperforms state-of-the-art TTT methods across various benchmarks.
>
---
#### [new 111] Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于多任务学习领域，解决任务间负迁移问题。通过动态调整token空间，提升模型适应性与性能。**

- **链接: [http://arxiv.org/pdf/2507.07485v1](http://arxiv.org/pdf/2507.07485v1)**

> **作者:** Wooseong Jeong; Kuk-Jin Yoon
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Multi-Task Learning (MTL) enables multiple tasks to be learned within a shared network, but differences in objectives across tasks can cause negative transfer, where the learning of one task degrades another task's performance. While pre-trained transformers significantly improve MTL performance, their fixed network capacity and rigid structure limit adaptability. Previous dynamic network architectures attempt to address this but are inefficient as they directly convert shared parameters into task-specific ones. We propose Dynamic Token Modulation and Expansion (DTME-MTL), a framework applicable to any transformer-based MTL architecture. DTME-MTL enhances adaptability and reduces overfitting by identifying gradient conflicts in token space and applying adaptive solutions based on conflict type. Unlike prior methods that mitigate negative transfer by duplicating network parameters, DTME-MTL operates entirely in token space, enabling efficient adaptation without excessive parameter growth. Extensive experiments demonstrate that DTME-MTL consistently improves multi-task performance with minimal computational overhead, offering a scalable and effective solution for enhancing transformer-based MTL models.
>
---
#### [new 112] Rainbow Artifacts from Electromagnetic Signal Injection Attacks on Image Sensors
- **分类: cs.CR; cs.CV; B.8; I.4**

- **简介: 该论文属于安全领域，研究电磁信号注入攻击对图像传感器的影响，揭示了导致彩虹色伪影的物理层漏洞，并评估其对目标检测模型的干扰。**

- **链接: [http://arxiv.org/pdf/2507.07773v1](http://arxiv.org/pdf/2507.07773v1)**

> **作者:** Youqian Zhang; Xinyu Ji; Zhihao Wang; Qinhong Jiang
>
> **备注:** 5 pages, 4 figures
>
> **摘要:** Image sensors are integral to a wide range of safety- and security-critical systems, including surveillance infrastructure, autonomous vehicles, and industrial automation. These systems rely on the integrity of visual data to make decisions. In this work, we investigate a novel class of electromagnetic signal injection attacks that target the analog domain of image sensors, allowing adversaries to manipulate raw visual inputs without triggering conventional digital integrity checks. We uncover a previously undocumented attack phenomenon on CMOS image sensors: rainbow-like color artifacts induced in images captured by image sensors through carefully tuned electromagnetic interference. We further evaluate the impact of these attacks on state-of-the-art object detection models, showing that the injected artifacts propagate through the image signal processing pipeline and lead to significant mispredictions. Our findings highlight a critical and underexplored vulnerability in the visual perception stack, highlighting the need for more robust defenses against physical-layer attacks in such systems.
>
---
#### [new 113] Wrist bone segmentation in X-ray images using CT-based simulations
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决腕骨在X光图像中分割困难的问题。通过使用CT模拟的X光图像训练深度学习模型，提升分割精度。**

- **链接: [http://arxiv.org/pdf/2507.07131v1](http://arxiv.org/pdf/2507.07131v1)**

> **作者:** Youssef ElTantawy; Alexia Karantana; Xin Chen
>
> **备注:** 4 pages
>
> **摘要:** Plain X-ray is one of the most common image modalities for clinical diagnosis (e.g. bone fracture, pneumonia, cancer screening, etc.). X-ray image segmentation is an essential step for many computer-aided diagnostic systems, yet it remains challenging. Deep-learning-based methods have achieved superior performance in medical image segmentation tasks but often require a large amount of high-quality annotated data for model training. Providing such an annotated dataset is not only time-consuming but also requires a high level of expertise. This is particularly challenging in wrist bone segmentation in X-rays, due to the interposition of multiple small carpal bones in the image. To overcome the data annotation issue, this work utilizes a large number of simulated X-ray images generated from Computed Tomography (CT) volumes with their corresponding 10 bone labels to train a deep learning-based model for wrist bone segmentation in real X-ray images. The proposed method was evaluated using both simulated images and real images. The method achieved Dice scores ranging from 0.80 to 0.92 for the simulated dataset generated from different view angles. Qualitative analysis of the segmentation results of the real X-ray images also demonstrated the superior performance of the trained model. The trained model and X-ray simulation code are freely available for research purposes: the link will be provided upon acceptance.
>
---
#### [new 114] Capture Stage Environments: A Guide to Better Matting
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于图像matting任务，解决 capture stage 内容的背景分离问题，提出针对性指南与高效处理方法。**

- **链接: [http://arxiv.org/pdf/2507.07623v1](http://arxiv.org/pdf/2507.07623v1)**

> **作者:** Hannah Dröge; Janelle Pfeifer; Saskia Rabich; Markus Plack; Reinhard Klein; Matthias B. Hullin
>
> **摘要:** Capture stages are high-end sources of state-of-the-art recordings for downstream applications in movies, games, and other media. One crucial step in almost all pipelines is the matting of images to isolate the captured performances from the background. While common matting algorithms deliver remarkable performance in other applications like teleconferencing and mobile entertainment, we found that they struggle significantly with the peculiarities of capture stage content. The goal of our work is to share insights into those challenges as a curated list of those characteristics along with a constructive discussion for proactive intervention and present a guideline to practitioners for an improved workflow to mitigate unresolved challenges. To this end, we also demonstrate an efficient pipeline to adapt state-of-the-art approaches to such custom setups without the need of extensive annotations, both offline and real-time. For an objective evaluation, we propose a validation methodology based on a leading diffusion model that highlights the benefits of our approach.
>
---
#### [new 115] Computationally Efficient Information-Driven Optical Design with Interchanging Optimization
- **分类: eess.IV; cs.CE; cs.CV; cs.IT; math.IT; physics.optics**

- **简介: 该论文属于光学设计任务，解决信息驱动设计中的高内存和长运行时问题。通过解耦密度估计与参数优化，提出IDEAL-IO方法，提升效率与设计质量。**

- **链接: [http://arxiv.org/pdf/2507.07789v1](http://arxiv.org/pdf/2507.07789v1)**

> **作者:** Eric Markley; Henry Pinkard; Leyla Kabuli; Nalini Singh; Laura Waller
>
> **摘要:** Recent work has demonstrated that imaging systems can be evaluated through the information content of their measurements alone, enabling application-agnostic optical design that avoids computational decoding challenges. Information-Driven Encoder Analysis Learning (IDEAL) was proposed to automate this process through gradient-based. In this work, we study IDEAL across diverse imaging systems and find that it suffers from high memory usage, long runtimes, and a potentially mismatched objective function due to end-to-end differentiability requirements. We introduce IDEAL with Interchanging Optimization (IDEAL-IO), a method that decouples density estimation from optical parameter optimization by alternating between fitting models to current measurements and updating optical parameters using fixed models for information estimation. This approach reduces runtime and memory usage by up to 6x while enabling more expressive density models that guide optimization toward superior designs. We validate our method on diffractive optics, lensless imaging, and snapshot 3D microscopy applications, establishing information-theoretic optimization as a practical, scalable strategy for real-world imaging system design.
>
---
#### [new 116] Input Conditioned Layer Dropping in Speech Foundation Models
- **分类: cs.SD; cs.CV; eess.AS**

- **简介: 该论文属于语音模型优化任务，解决边缘设备计算资源动态变化的问题。提出输入驱动的层跳过方法，在不改变架构的前提下动态调整计算量，提升模型效率。**

- **链接: [http://arxiv.org/pdf/2507.07954v1](http://arxiv.org/pdf/2507.07954v1)**

> **作者:** Abdul Hannan; Daniele Falavigna; Alessio Brutti
>
> **备注:** Accepted at IEEE MLSP 2025
>
> **摘要:** Curating foundation speech models for edge and IoT settings, where computational resources vary over time, requires dynamic architectures featuring adaptable reduction strategies. One emerging approach is layer dropping ($\mathcal{LD}$) which skips fraction of the layers of a backbone network during inference to reduce the computational load. This allows transforming static models into dynamic ones. However, existing approaches exhibit limitations either in the mode of selecting layers or by significantly modifying the neural architecture. To this end, we propose input-driven $\mathcal{LD}$ that employs the network's input features and a lightweight layer selecting network to determine the optimum combination of processing layers. Extensive experimentation on 4 speech and audio public benchmarks, using two different pre-trained foundation models, demonstrates the effectiveness of our approach, thoroughly outperforming random dropping and producing on-par (or better) results to early exit.
>
---
#### [new 117] PyVision: Agentic Vision with Dynamic Tooling
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于视觉推理任务，解决传统方法依赖静态工具的问题。提出PyVision框架，使模型能动态生成和优化Python工具，提升视觉任务性能。**

- **链接: [http://arxiv.org/pdf/2507.07998v1](http://arxiv.org/pdf/2507.07998v1)**

> **作者:** Shitian Zhao; Haoquan Zhang; Shaoheng Lin; Ming Li; Qilong Wu; Kaipeng Zhang; Chen Wei
>
> **备注:** 26 Pages, 10 Figures, Technical report
>
> **摘要:** LLMs are increasingly deployed as agents, systems capable of planning, reasoning, and dynamically calling external tools. However, in visual reasoning, prior approaches largely remain limited by predefined workflows and static toolsets. In this report, we present PyVision, an interactive, multi-turn framework that enables MLLMs to autonomously generate, execute, and refine Python-based tools tailored to the task at hand, unlocking flexible and interpretable problem-solving. We develop a taxonomy of the tools created by PyVision and analyze their usage across a diverse set of benchmarks. Quantitatively, PyVision achieves consistent performance gains, boosting GPT-4.1 by +7.8% on V* and Claude-4.0-Sonnet by +31.1% on VLMsAreBlind-mini. These results point to a broader shift: dynamic tooling allows models not just to use tools, but to invent them, advancing toward more agentic visual reasoning.
>
---
## 更新

#### [replaced 001] Don't Get Me Wrong: How to Apply Deep Visual Interpretations to Time Series
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2203.07861v3](http://arxiv.org/pdf/2203.07861v3)**

> **作者:** Christoffer Loeffler; Wei-Cheng Lai; Bjoern Eskofier; Dario Zanca; Lukas Schmidt; Christopher Mutschler
>
> **备注:** 48 pages, 12 figues, 7 tables, 6 algorithms
>
> **摘要:** The correct interpretation of convolutional models is a hard problem for time series data. While saliency methods promise visual validation of predictions for image and language processing, they fall short when applied to time series. These tend to be less intuitive and represent highly diverse data, such as the tool-use time series dataset. Furthermore, saliency methods often generate varied, conflicting explanations, complicating the reliability of these methods. Consequently, a rigorous objective assessment is necessary to establish trust in them. This paper investigates saliency methods on time series data to formulate recommendations for interpreting convolutional models and implements them on the tool-use time series problem. To achieve this, we first employ nine gradient-, propagation-, or perturbation-based post-hoc saliency methods across six varied and complex real-world datasets. Next, we evaluate these methods using five independent metrics to generate recommendations. Subsequently, we implement a case study focusing on tool-use time series using convolutional classification models. Our results validate our recommendations that indicate that none of the saliency methods consistently outperforms others on all metrics, while some are sometimes ahead. Our insights and step-by-step guidelines allow experts to choose suitable saliency methods for a given model and dataset.
>
---
#### [replaced 002] Masked Image Modeling: A Survey
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2408.06687v3](http://arxiv.org/pdf/2408.06687v3)**

> **作者:** Vlad Hondru; Florinel Alin Croitoru; Shervin Minaee; Radu Tudor Ionescu; Nicu Sebe
>
> **备注:** Accepted at the International Journal of Computer Vision
>
> **摘要:** In this work, we survey recent studies on masked image modeling (MIM), an approach that emerged as a powerful self-supervised learning technique in computer vision. The MIM task involves masking some information, e.g. pixels, patches, or even latent representations, and training a model, usually an autoencoder, to predicting the missing information by using the context available in the visible part of the input. We identify and formalize two categories of approaches on how to implement MIM as a pretext task, one based on reconstruction and one based on contrastive learning. Then, we construct a taxonomy and review the most prominent papers in recent years. We complement the manually constructed taxonomy with a dendrogram obtained by applying a hierarchical clustering algorithm. We further identify relevant clusters via manually inspecting the resulting dendrogram. Our review also includes datasets that are commonly used in MIM research. We aggregate the performance results of various masked image modeling methods on the most popular datasets, to facilitate the comparison of competing methods. Finally, we identify research gaps and propose several interesting directions of future work. We supplement our survey with the following public repository containing organized references: https://github.com/vladhondru25/MIM-Survey.
>
---
#### [replaced 003] Revisiting Likelihood-Based Out-of-Distribution Detection by Modeling Representations
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.07793v3](http://arxiv.org/pdf/2504.07793v3)**

> **作者:** Yifan Ding; Arturas Aleksandraus; Amirhossein Ahmadian; Jonas Unger; Fredrik Lindsten; Gabriel Eilertsen
>
> **备注:** Scandinavian Conference on Image Analysis 2025 (oral)
>
> **摘要:** Out-of-distribution (OOD) detection is critical for ensuring the reliability of deep learning systems, particularly in safety-critical applications. Likelihood-based deep generative models have historically faced criticism for their unsatisfactory performance in OOD detection, often assigning higher likelihood to OOD data than in-distribution samples when applied to image data. In this work, we demonstrate that likelihood is not inherently flawed. Rather, several properties in the images space prohibit likelihood as a valid detection score. Given a sufficiently good likelihood estimator, specifically using the probability flow formulation of a diffusion model, we show that likelihood-based methods can still perform on par with state-of-the-art methods when applied in the representation space of pre-trained encoders. The code of our work can be found at $\href{https://github.com/limchaos/Likelihood-OOD.git}{\texttt{https://github.com/limchaos/Likelihood-OOD.git}}$.
>
---
#### [replaced 004] Judging from Support-set: A New Way to Utilize Few-Shot Segmentation for Segmentation Refinement Process
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.04519v3](http://arxiv.org/pdf/2407.04519v3)**

> **作者:** Seonghyeon Moon; Qingze; Liu; Haein Kong; Muhammad Haris Khan
>
> **备注:** ICIP 2025
>
> **摘要:** Segmentation refinement aims to enhance the initial coarse masks generated by segmentation algorithms. The refined masks are expected to capture more details and better contours of the target objects. Research on segmentation refinement has developed as a response to the need for high-quality image segmentations. However, to our knowledge, no method has been developed that can determine the success of segmentation refinement. Such a method could ensure the reliability of segmentation in applications where the outcome of the segmentation is important and fosters innovation in image processing technologies. To address this research gap, we propose Judging From Support-set (JFS), a method to judge the success of segmentation refinement leveraging an off-the-shelf few-shot segmentation (FSS) model. The traditional goal of the problem in FSS is to find a target object in a query image utilizing target information given by a support set. However, we propose a novel application of the FSS model in our evaluation pipeline for segmentation refinement methods. Given a coarse mask as input, segmentation refinement methods produce a refined mask; these two masks become new support masks for the FSS model. The existing support mask then serves as the test set for the FSS model to evaluate the quality of the refined segmentation by the segmentation refinement methods. We demonstrate the effectiveness of our proposed JFS framework by evaluating the SAM Enhanced Pseudo-Labels (SEPL) using SegGPT as the choice of FSS model on the PASCAL dataset. The results showed that JFS has the potential to determine whether the segmentation refinement process is successful.
>
---
#### [replaced 005] Solving Inverse Problems using Diffusion with Iterative Colored Renoising
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.17468v3](http://arxiv.org/pdf/2501.17468v3)**

> **作者:** Matt C. Bendel; Saurav K. Shastri; Rizwan Ahmad; Philip Schniter
>
> **摘要:** Imaging inverse problems can be solved in an unsupervised manner using pre-trained diffusion models, but doing so requires approximating the gradient of the measurement-conditional score function in the diffusion reverse process. We show that the approximations produced by existing methods are relatively poor, especially early in the reverse process, and so we propose a new approach that iteratively reestimates and "renoises" the estimate several times per diffusion step. This iterative approach, which we call Fast Iterative REnoising (FIRE), injects colored noise that is shaped to ensure that the pre-trained diffusion model always sees white noise, in accordance with how it was trained. We then embed FIRE into the DDIM reverse process and show that the resulting "DDfire" offers state-of-the-art accuracy and runtime on several linear inverse problems, as well as phase retrieval. Our implementation is at https://github.com/matt-bendel/DDfire
>
---
#### [replaced 006] HadaNorm: Diffusion Transformer Quantization through Mean-Centered Transformations
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.09932v2](http://arxiv.org/pdf/2506.09932v2)**

> **作者:** Marco Federici; Riccardo Del Chiaro; Boris van Breugel; Paul Whatmough; Markus Nagel
>
> **备注:** 8 Pages, 6 Figures
>
> **摘要:** Diffusion models represent the cutting edge in image generation, but their high memory and computational demands hinder deployment on resource-constrained devices. Post-Training Quantization (PTQ) offers a promising solution by reducing the bitwidth of matrix operations. However, standard PTQ methods struggle with outliers, and achieving higher compression often requires transforming model weights and activations before quantization. In this work, we propose HadaNorm, a novel linear transformation that extends existing approaches by both normalizing channels activations and applying Hadamard transforms to effectively mitigate outliers and enable aggressive activation quantization. We demonstrate that HadaNorm consistently reduces quantization error across the various components of transformer blocks, outperforming state-of-the-art methods.
>
---
#### [replaced 007] DLaVA: Document Language and Vision Assistant for Answer Localization with Enhanced Interpretability and Trustworthiness
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.00151v2](http://arxiv.org/pdf/2412.00151v2)**

> **作者:** Ahmad Mohammadshirazi; Pinaki Prasad Guha Neogi; Ser-Nam Lim; Rajiv Ramnath
>
> **摘要:** Document Visual Question Answering (VQA) demands robust integration of text detection, recognition, and spatial reasoning to interpret complex document layouts. In this work, we introduce DLaVA, a novel, training-free pipeline that leverages Multimodal Large Language Models (MLLMs) for zero-shot answer localization in order to improve trustworthiness, interpretability, and explainability. By leveraging an innovative OCR-free approach that organizes text regions with unique bounding box IDs, the proposed method preserves spatial contexts without relying on iterative OCR or chain-of-thought reasoning, thus substantially reducing the computational complexity. We further enhance the evaluation protocol by integrating Intersection over Union (IoU) metrics alongside Average Normalized Levenshtein Similarity (ANLS), thereby ensuring that not only textual accuracy is considered, but spatial accuracy is taken into account, ultimately reducing the risks of AI hallucinations and improving trustworthiness. Experiments on benchmark datasets demonstrate competitive performance compared to state-of-the-art techniques, with significantly lower computational complexity and enhanced accuracies and reliability for high-stakes applications. The code and datasets utilized in this study for DLaVA are accessible at: https://github.com/ahmad-shirazi/AnnotMLLM.
>
---
#### [replaced 008] OVOR: OnePrompt with Virtual Outlier Regularization for Rehearsal-Free Class-Incremental Learning
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2402.04129v2](http://arxiv.org/pdf/2402.04129v2)**

> **作者:** Wei-Cheng Huang; Chun-Fu Chen; Hsiang Hsu
>
> **备注:** Accepted by ICLR 2024
>
> **摘要:** Recent works have shown that by using large pre-trained models along with learnable prompts, rehearsal-free methods for class-incremental learning (CIL) settings can achieve superior performance to prominent rehearsal-based ones. Rehearsal-free CIL methods struggle with distinguishing classes from different tasks, as those are not trained together. In this work we propose a regularization method based on virtual outliers to tighten decision boundaries of the classifier, such that confusion of classes among different tasks is mitigated. Recent prompt-based methods often require a pool of task-specific prompts, in order to prevent overwriting knowledge of previous tasks with that of the new task, leading to extra computation in querying and composing an appropriate prompt from the pool. This additional cost can be eliminated, without sacrificing accuracy, as we reveal in the paper. We illustrate that a simplified prompt-based method can achieve results comparable to previous state-of-the-art (SOTA) methods equipped with a prompt pool, using much less learnable parameters and lower inference cost. Our regularization method has demonstrated its compatibility with different prompt-based methods, boosting those previous SOTA rehearsal-free CIL methods' accuracy on the ImageNet-R and CIFAR-100 benchmarks. Our source code is available at https://github.com/jpmorganchase/ovor.
>
---
#### [replaced 009] Hybrid-View Attention Network for Clinically Significant Prostate Cancer Classification in Transrectal Ultrasound
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.03421v2](http://arxiv.org/pdf/2507.03421v2)**

> **作者:** Zetian Feng; Juan Fu; Xuebin Zou; Hongsheng Ye; Hong Wu; Jianhua Zhou; Yi Wang
>
> **摘要:** Prostate cancer (PCa) is a leading cause of cancer-related mortality in men, and accurate identification of clinically significant PCa (csPCa) is critical for timely intervention. Transrectal ultrasound (TRUS) is widely used for prostate biopsy; however, its low contrast and anisotropic spatial resolution pose diagnostic challenges. To address these limitations, we propose a novel hybrid-view attention (HVA) network for csPCa classification in 3D TRUS that leverages complementary information from transverse and sagittal views. Our approach integrates a CNN-transformer hybrid architecture, where convolutional layers extract fine-grained local features and transformer-based HVA models global dependencies. Specifically, the HVA comprises intra-view attention to refine features within a single view and cross-view attention to incorporate complementary information across views. Furthermore, a hybrid-view adaptive fusion module dynamically aggregates features along both channel and spatial dimensions, enhancing the overall representation. Experiments are conducted on an in-house dataset containing 590 subjects who underwent prostate biopsy. Comparative and ablation results prove the efficacy of our method. The code is available at https://github.com/mock1ngbrd/HVAN.
>
---
#### [replaced 010] Multi-modal Representations for Fine-grained Multi-label Critical View of Safety Recognition
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.05007v2](http://arxiv.org/pdf/2507.05007v2)**

> **作者:** Britty Baby; Vinkle Srivastav; Pooja P. Jain; Kun Yuan; Pietro Mascagni; Nicolas Padoy
>
> **摘要:** The Critical View of Safety (CVS) is crucial for safe laparoscopic cholecystectomy, yet assessing CVS criteria remains a complex and challenging task, even for experts. Traditional models for CVS recognition depend on vision-only models learning with costly, labor-intensive spatial annotations. This study investigates how text can be harnessed as a powerful tool for both training and inference in multi-modal surgical foundation models to automate CVS recognition. Unlike many existing multi-modal models, which are primarily adapted for multi-class classification, CVS recognition requires a multi-label framework. Zero-shot evaluation of existing multi-modal surgical models shows a significant performance gap for this task. To address this, we propose CVS-AdaptNet, a multi-label adaptation strategy that enhances fine-grained, binary classification across multiple labels by aligning image embeddings with textual descriptions of each CVS criterion using positive and negative prompts. By adapting PeskaVLP, a state-of-the-art surgical foundation model, on the Endoscapes-CVS201 dataset, CVS-AdaptNet achieves 57.6 mAP, improving over the ResNet50 image-only baseline (51.5 mAP) by 6 points. Our results show that CVS-AdaptNet's multi-label, multi-modal framework, enhanced by textual prompts, boosts CVS recognition over image-only methods. We also propose text-specific inference methods, that helps in analysing the image-text alignment. While further work is needed to match state-of-the-art spatial annotation-based methods, this approach highlights the potential of adapting generalist models to specialized surgical tasks. Code: https://github.com/CAMMA-public/CVS-AdaptNet
>
---
#### [replaced 011] Learning to Generate Vectorized Maps at Intersections with Multiple Roadside Cameras
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.02899v2](http://arxiv.org/pdf/2507.02899v2)**

> **作者:** Quanxin Zheng; Miao Fan; Shengtong Xu; Linghe Kong; Haoyi Xiong
>
> **备注:** Accepted by IROS'25
>
> **摘要:** Vectorized maps are indispensable for precise navigation and the safe operation of autonomous vehicles. Traditional methods for constructing these maps fall into two categories: offline techniques, which rely on expensive, labor-intensive LiDAR data collection and manual annotation, and online approaches that use onboard cameras to reduce costs but suffer from limited performance, especially at complex intersections. To bridge this gap, we introduce MRC-VMap, a cost-effective, vision-centric, end-to-end neural network designed to generate high-definition vectorized maps directly at intersections. Leveraging existing roadside surveillance cameras, MRC-VMap directly converts time-aligned, multi-directional images into vectorized map representations. This integrated solution lowers the need for additional intermediate modules--such as separate feature extraction and Bird's-Eye View (BEV) conversion steps--thus reducing both computational overhead and error propagation. Moreover, the use of multiple camera views enhances mapping completeness, mitigates occlusions, and provides robust performance under practical deployment constraints. Extensive experiments conducted on 4,000 intersections across 4 major metropolitan areas in China demonstrate that MRC-VMap not only outperforms state-of-the-art online methods but also achieves accuracy comparable to high-cost LiDAR-based approaches, thereby offering a scalable and efficient solution for modern autonomous navigation systems.
>
---
#### [replaced 012] Cosmos World Foundation Model Platform for Physical AI
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2501.03575v3](http://arxiv.org/pdf/2501.03575v3)**

> **作者:** NVIDIA; :; Niket Agarwal; Arslan Ali; Maciej Bala; Yogesh Balaji; Erik Barker; Tiffany Cai; Prithvijit Chattopadhyay; Yongxin Chen; Yin Cui; Yifan Ding; Daniel Dworakowski; Jiaojiao Fan; Michele Fenzi; Francesco Ferroni; Sanja Fidler; Dieter Fox; Songwei Ge; Yunhao Ge; Jinwei Gu; Siddharth Gururani; Ethan He; Jiahui Huang; Jacob Huffman; Pooya Jannaty; Jingyi Jin; Seung Wook Kim; Gergely Klár; Grace Lam; Shiyi Lan; Laura Leal-Taixe; Anqi Li; Zhaoshuo Li; Chen-Hsuan Lin; Tsung-Yi Lin; Huan Ling; Ming-Yu Liu; Xian Liu; Alice Luo; Qianli Ma; Hanzi Mao; Kaichun Mo; Arsalan Mousavian; Seungjun Nah; Sriharsha Niverty; David Page; Despoina Paschalidou; Zeeshan Patel; Lindsey Pavao; Morteza Ramezanali; Fitsum Reda; Xiaowei Ren; Vasanth Rao Naik Sabavat; Ed Schmerling; Stella Shi; Bartosz Stefaniak; Shitao Tang; Lyne Tchapmi; Przemek Tredak; Wei-Cheng Tseng; Jibin Varghese; Hao Wang; Haoxiang Wang; Heng Wang; Ting-Chun Wang; Fangyin Wei; Xinyue Wei; Jay Zhangjie Wu; Jiashu Xu; Wei Yang; Lin Yen-Chen; Xiaohui Zeng; Yu Zeng; Jing Zhang; Qinsheng Zhang; Yuxuan Zhang; Qingqing Zhao; Artur Zolkowski
>
> **摘要:** Physical AI needs to be trained digitally first. It needs a digital twin of itself, the policy model, and a digital twin of the world, the world model. In this paper, we present the Cosmos World Foundation Model Platform to help developers build customized world models for their Physical AI setups. We position a world foundation model as a general-purpose world model that can be fine-tuned into customized world models for downstream applications. Our platform covers a video curation pipeline, pre-trained world foundation models, examples of post-training of pre-trained world foundation models, and video tokenizers. To help Physical AI builders solve the most critical problems of our society, we make Cosmos open-source and our models open-weight with permissive licenses available via https://github.com/nvidia-cosmos/cosmos-predict1.
>
---
#### [replaced 013] Uncertainty-Aware Gradient Stabilization for Small Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2303.01803v2](http://arxiv.org/pdf/2303.01803v2)**

> **作者:** Huixin Sun; Yanjing Li; Linlin Yang; Xianbin Cao; Baochang Zhang
>
> **摘要:** Despite advances in generic object detection, there remains a performance gap in detecting small objects compared to normal-scale objects. We reveal that conventional object localization methods suffer from gradient instability in small objects due to sharper loss curvature, leading to a convergence challenge. To address the issue, we propose Uncertainty-Aware Gradient Stabilization (UGS), a framework that reformulates object localization as a classification task to stabilize gradients. UGS quantizes continuous labels into interval non-uniform discrete representations. Under a classification-based objective, the localization branch generates bounded and confidence-driven gradients, mitigating instability. Furthermore, UGS integrates an uncertainty minimization (UM) loss that reduces prediction variance and an uncertainty-guided refinement (UR) module that identifies and refines high-uncertainty regions via perturbations. Evaluated on four benchmarks, UGS consistently improves anchor-based, anchor-free, and leading small object detectors. Especially, UGS enhances DINO-5scale by 2.6 AP on VisDrone, surpassing previous state-of-the-art results.
>
---
#### [replaced 014] PWD: Prior-Guided and Wavelet-Enhanced Diffusion Model for Limited-Angle CT
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.05317v2](http://arxiv.org/pdf/2507.05317v2)**

> **作者:** Yi Liu; Yiyang Wen; Zekun Zhou; Junqi Ma; Linghang Wang; Yucheng Yao; Liu Shi; Qiegen Liu
>
> **摘要:** Generative diffusion models have received increasing attention in medical imaging, particularly in limited-angle computed tomography (LACT). Standard diffusion models achieve high-quality image reconstruction but require a large number of sampling steps during inference, resulting in substantial computational overhead. Although skip-sampling strategies have been proposed to improve efficiency, they often lead to loss of fine structural details. To address this issue, we propose a prior information embedding and wavelet feature fusion fast sampling diffusion model for LACT reconstruction. The PWD enables efficient sampling while preserving reconstruction fidelity in LACT, and effectively mitigates the degradation typically introduced by skip-sampling. Specifically, during the training phase, PWD maps the distribution of LACT images to that of fully sampled target images, enabling the model to learn structural correspondences between them. During inference, the LACT image serves as an explicit prior to guide the sampling trajectory, allowing for high-quality reconstruction with significantly fewer steps. In addition, PWD performs multi-scale feature fusion in the wavelet domain, effectively enhancing the reconstruction of fine details by leveraging both low-frequency and high-frequency information. Quantitative and qualitative evaluations on clinical dental arch CBCT and periapical datasets demonstrate that PWD outperforms existing methods under the same sampling condition. Using only 50 sampling steps, PWD achieves at least 1.7 dB improvement in PSNR and 10% gain in SSIM.
>
---
#### [replaced 015] Skywork-R1V3 Technical Report
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.06167v3](http://arxiv.org/pdf/2507.06167v3)**

> **作者:** Wei Shen; Jiangbo Pei; Yi Peng; Xuchen Song; Yang Liu; Jian Peng; Haofeng Sun; Yunzhuo Hao; Peiyu Wang; Jianhao Zhang; Yahui Zhou
>
> **摘要:** We introduce Skywork-R1V3, an advanced, open-source vision-language model (VLM) that pioneers a new approach to visual reasoning. Its key innovation lies in effectively transferring reasoning skills from text-only Large Language Models (LLMs) to visual tasks. The strong performance of Skywork-R1V3 primarily stems from our elaborate post-training RL framework, which effectively activates and enhances the model's reasoning ability, without the need for additional continue pre-training. Through this framework, we further uncover the fundamental role of the connector module in achieving robust cross-modal alignment for multimodal reasoning models. In addition, we introduce a unique indicator of reasoning capability, the entropy of critical reasoning tokens, which has proven highly effective for checkpoint selection during RL training. Skywork-R1V3 achieves state-of-the-art results on MMMU, significantly improving from 64.3% to 76.0%. This performance matches entry-level human capabilities. Remarkably, our RL-powered post-training approach enables even the 38B parameter model to rival top closed-source VLMs. The implementation successfully transfers mathematical reasoning to other subject-related reasoning tasks. We also include an analysis of curriculum learning and reinforcement finetuning strategies, along with a broader discussion on multimodal reasoning. Skywork-R1V3 represents a significant leap in multimodal reasoning, showcasing RL as a powerful engine for advancing open-source VLM capabilities.
>
---
#### [replaced 016] MoSiC: Optimal-Transport Motion Trajectory for Dense Self-Supervised Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.08694v2](http://arxiv.org/pdf/2506.08694v2)**

> **作者:** Mohammadreza Salehi; Shashanka Venkataramanan; Ioana Simion; Efstratios Gavves; Cees G. M. Snoek; Yuki M Asano
>
> **备注:** Accepted to ICCV2025
>
> **摘要:** Dense self-supervised learning has shown great promise for learning pixel- and patch-level representations, but extending it to videos remains challenging due to the complexity of motion dynamics. Existing approaches struggle as they rely on static augmentations that fail under object deformations, occlusions, and camera movement, leading to inconsistent feature learning over time. We propose a motion-guided self-supervised learning framework that clusters dense point tracks to learn spatiotemporally consistent representations. By leveraging an off-the-shelf point tracker, we extract long-range motion trajectories and optimize feature clustering through a momentum-encoder-based optimal transport mechanism. To ensure temporal coherence, we propagate cluster assignments along tracked points, enforcing feature consistency across views despite viewpoint changes. Integrating motion as an implicit supervisory signal, our method learns representations that generalize across frames, improving robustness in dynamic scenes and challenging occlusion scenarios. By initializing from strong image-pretrained models and leveraging video data for training, we improve state-of-the-art by 1% to 6% on six image and video datasets and four evaluation benchmarks. The implementation is publicly available at our GitHub repository: https://github.com/SMSD75/MoSiC/tree/main
>
---
#### [replaced 017] Concept Unlearning by Modeling Key Steps of Diffusion Process
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.06526v2](http://arxiv.org/pdf/2507.06526v2)**

> **作者:** Chaoshuo Zhang; Chenhao Lin; Zhengyu Zhao; Le Yang; Qian Wang; Chao Shen
>
> **摘要:** Text-to-image diffusion models (T2I DMs), represented by Stable Diffusion, which generate highly realistic images based on textual input, have been widely used. However, their misuse poses serious security risks. While existing concept unlearning methods aim to mitigate these risks, they struggle to balance unlearning effectiveness with generative retainability.To overcome this limitation, we innovatively propose the Key Step Concept Unlearning (KSCU) method, which ingeniously capitalizes on the unique stepwise sampling characteristic inherent in diffusion models during the image generation process. Unlike conventional approaches that treat all denoising steps equally, KSCU strategically focuses on pivotal steps with the most influence over the final outcome by dividing key steps for different concept unlearning tasks and fine-tuning the model only at those steps. This targeted approach reduces the number of parameter updates needed for effective unlearning, while maximizing the retention of the model's generative capabilities.Through extensive benchmark experiments, we demonstrate that KSCU effectively prevents T2I DMs from generating undesirable images while better retaining the model's generative capabilities. Our code will be released.
>
---
#### [replaced 018] Hallucinating 360°: Panoramic Street-View Generation via Local Scenes Diffusion and Probabilistic Prompting
- **分类: cs.CV; cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2507.06971v2](http://arxiv.org/pdf/2507.06971v2)**

> **作者:** Fei Teng; Kai Luo; Sheng Wu; Siyu Li; Pujun Guo; Jiale Wei; Kunyu Peng; Jiaming Zhang; Kailun Yang
>
> **备注:** The source code will be publicly available at https://github.com/Bryant-Teng/Percep360
>
> **摘要:** Panoramic perception holds significant potential for autonomous driving, enabling vehicles to acquire a comprehensive 360{\deg} surround view in a single shot. However, autonomous driving is a data-driven task. Complete panoramic data acquisition requires complex sampling systems and annotation pipelines, which are time-consuming and labor-intensive. Although existing street view generation models have demonstrated strong data regeneration capabilities, they can only learn from the fixed data distribution of existing datasets and cannot achieve high-quality, controllable panoramic generation. In this paper, we propose the first panoramic generation method Percep360 for autonomous driving. Percep360 enables coherent generation of panoramic data with control signals based on the stitched panoramic data. Percep360 focuses on two key aspects: coherence and controllability. Specifically, to overcome the inherent information loss caused by the pinhole sampling process, we propose the Local Scenes Diffusion Method (LSDM). LSDM reformulates the panorama generation as a spatially continuous diffusion process, bridging the gaps between different data distributions. Additionally, to achieve the controllable generation of panoramic images, we propose a Probabilistic Prompting Method (PPM). PPM dynamically selects the most relevant control cues, enabling controllable panoramic image generation. We evaluate the effectiveness of the generated images from three perspectives: image quality assessment (i.e., no-reference and with reference), controllability, and their utility in real-world Bird's Eye View (BEV) segmentation. Notably, the generated data consistently outperforms the original stitched images in no-reference quality metrics and enhances downstream perception models. The source code will be publicly available at https://github.com/Bryant-Teng/Percep360.
>
---
#### [replaced 019] MCFormer: A Multi-Cost-Volume Network and Comprehensive Benchmark for Particle Image Velocimetry
- **分类: cs.CV; cs.AI; 68T45, 65D18**

- **链接: [http://arxiv.org/pdf/2507.04750v2](http://arxiv.org/pdf/2507.04750v2)**

> **作者:** Zicheng Lin; Xiaoqiang Li; Yichao Wang; Chuang Zhu
>
> **备注:** 20 pages, 13 figures, 5 tables. Comprehensive benchmark evaluation of optical flow models for PIV. Introduces MCFormer architecture with multi-frame temporal processing and multiple cost volumes. Includes large-scale synthetic PIV dataset based on JHTDB and Blasius CFD simulations. Code and dataset will be made publicly available
>
> **摘要:** Particle Image Velocimetry (PIV) is fundamental to fluid dynamics, yet deep learning applications face significant hurdles. A critical gap exists: the lack of comprehensive evaluation of how diverse optical flow models perform specifically on PIV data, largely due to limitations in available datasets and the absence of a standardized benchmark. This prevents fair comparison and hinders progress. To address this, our primary contribution is a novel, large-scale synthetic PIV benchmark dataset generated from diverse CFD simulations (JHTDB and Blasius). It features unprecedented variety in particle densities, flow velocities, and continuous motion, enabling, for the first time, a standardized and rigorous evaluation of various optical flow and PIV algorithms. Complementing this, we propose Multi Cost Volume PIV (MCFormer), a new deep network architecture leveraging multi-frame temporal information and multiple cost volumes, specifically designed for PIV's sparse nature. Our comprehensive benchmark evaluation, the first of its kind, reveals significant performance variations among adapted optical flow models and demonstrates that MCFormer significantly outperforms existing methods, achieving the lowest overall normalized endpoint error (NEPE). This work provides both a foundational benchmark resource essential for future PIV research and a state-of-the-art method tailored for PIV challenges. We make our benchmark dataset and code publicly available to foster future research in this area.
>
---
#### [replaced 020] Adaptation of Multi-modal Representation Models for Multi-task Surgical Computer Vision
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.05020v2](http://arxiv.org/pdf/2507.05020v2)**

> **作者:** Soham Walimbe; Britty Baby; Vinkle Srivastav; Nicolas Padoy
>
> **摘要:** Surgical AI often involves multiple tasks within a single procedure, like phase recognition or assessing the Critical View of Safety in laparoscopic cholecystectomy. Traditional models, built for one task at a time, lack flexibility, requiring a separate model for each. To address this, we introduce MML-SurgAdapt, a unified multi-task framework with Vision-Language Models (VLMs), specifically CLIP, to handle diverse surgical tasks through natural language supervision. A key challenge in multi-task learning is the presence of partial annotations when integrating different tasks. To overcome this, we employ Single Positive Multi-Label (SPML) learning, which traditionally reduces annotation burden by training models with only one positive label per instance. Our framework extends this approach to integrate data from multiple surgical tasks within a single procedure, enabling effective learning despite incomplete or noisy annotations. We demonstrate the effectiveness of our model on a combined dataset consisting of Cholec80, Endoscapes2023, and CholecT50, utilizing custom prompts. Extensive evaluation shows that MML-SurgAdapt performs comparably to task-specific benchmarks, with the added advantage of handling noisy annotations. It also outperforms the existing SPML frameworks for the task. By reducing the required labels by 23%, our approach proposes a more scalable and efficient labeling process, significantly easing the annotation burden on clinicians. To our knowledge, this is the first application of SPML to integrate data from multiple surgical tasks, presenting a novel and generalizable solution for multi-task learning in surgical computer vision. Implementation is available at: https://github.com/CAMMA-public/MML-SurgAdapt
>
---
#### [replaced 021] FunHOI: Annotation-Free 3D Hand-Object Interaction Generation via Functional Text Guidanc
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.20805v2](http://arxiv.org/pdf/2502.20805v2)**

> **作者:** Yongqi Tian; Xueyu Sun; Haoyuan He; Linji Hao; Ning Ding; Caigui Jiang
>
> **摘要:** Hand-object interaction(HOI) is the fundamental link between human and environment, yet its dexterous and complex pose significantly challenges for gesture control. Despite significant advances in AI and robotics, enabling machines to understand and simulate hand-object interactions, capturing the semantics of functional grasping tasks remains a considerable challenge. While previous work can generate stable and correct 3D grasps, they are still far from achieving functional grasps due to unconsidered grasp semantics. To address this challenge, we propose an innovative two-stage framework, Functional Grasp Synthesis Net (FGS-Net), for generating 3D HOI driven by functional text. This framework consists of a text-guided 3D model generator, Functional Grasp Generator (FGG), and a pose optimization strategy, Functional Grasp Refiner (FGR). FGG generates 3D models of hands and objects based on text input, while FGR fine-tunes the poses using Object Pose Approximator and energy functions to ensure the relative position between the hand and object aligns with human intent and remains physically plausible. Extensive experiments demonstrate that our approach achieves precise and high-quality HOI generation without requiring additional 3D annotation data.
>
---
#### [replaced 022] Damba-ST: Domain-Adaptive Mamba for Efficient Urban Spatio-Temporal Prediction
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.18939v2](http://arxiv.org/pdf/2506.18939v2)**

> **作者:** Rui An; Yifeng Zhang; Ziran Liang; Wenqi Fan; Yuxuan Liang; Xuequn Shang; Qing Li
>
> **摘要:** Training urban spatio-temporal foundation models that generalize well across diverse regions and cities is critical for deploying urban services in unseen or data-scarce regions. Recent studies have typically focused on fusing cross-domain spatio-temporal data to train unified Transformer-based models. However, these models suffer from quadratic computational complexity and high memory overhead, limiting their scalability and practical deployment. Inspired by the efficiency of Mamba, a state space model with linear time complexity, we explore its potential for efficient urban spatio-temporal prediction. However, directly applying Mamba as a spatio-temporal backbone leads to negative transfer and severe performance degradation. This is primarily due to spatio-temporal heterogeneity and the recursive mechanism of Mamba's hidden state updates, which limit cross-domain generalization. To overcome these challenges, we propose Damba-ST, a novel domain-adaptive Mamba-based model for efficient urban spatio-temporal prediction. Damba-ST retains Mamba's linear complexity advantage while significantly enhancing its adaptability to heterogeneous domains. Specifically, we introduce two core innovations: (1) a domain-adaptive state space model that partitions the latent representation space into a shared subspace for learning cross-domain commonalities and independent, domain-specific subspaces for capturing intra-domain discriminative features; (2) three distinct Domain Adapters, which serve as domain-aware proxies to bridge disparate domain distributions and facilitate the alignment of cross-domain commonalities. Extensive experiments demonstrate the generalization and efficiency of Damba-ST. It achieves state-of-the-art performance on prediction tasks and demonstrates strong zero-shot generalization, enabling seamless deployment in new urban environments without extensive retraining or fine-tuning.
>
---
#### [replaced 023] Multi-dynamic deep image prior for cardiac MRI
- **分类: physics.med-ph; cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2412.04639v2](http://arxiv.org/pdf/2412.04639v2)**

> **作者:** Marc Vornehm; Chong Chen; Muhammad Ahmad Sultan; Syed Murtaza Arshad; Yuchi Han; Florian Knoll; Rizwan Ahmad
>
> **摘要:** Cardiovascular magnetic resonance imaging is a powerful diagnostic tool for assessing cardiac structure and function. However, traditional breath-held imaging protocols pose challenges for patients with arrhythmias or limited breath-holding capacity. This work aims to overcome these limitations by developing a reconstruction framework that enables high-quality imaging in free-breathing conditions for various dynamic cardiac MRI protocols. Multi-Dynamic Deep Image Prior (M-DIP), a novel unsupervised reconstruction framework for accelerated real-time cardiac MRI, is introduced. To capture contrast or content variation, M-DIP first employs a spatial dictionary to synthesize a time-dependent intermediate image. Then, this intermediate image is further refined using time-dependent deformation fields that model cardiac and respiratory motion. Unlike prior DIP-based methods, M-DIP simultaneously captures physiological motion and frame-to-frame content variations, making it applicable to a wide range of dynamic applications. We validate M-DIP using simulated MRXCAT cine phantom data as well as free-breathing real-time cine, single-shot late gadolinium enhancement (LGE), and first-pass perfusion data from clinical patients. Comparative analyses against state-of-the-art supervised and unsupervised approaches demonstrate M-DIP's performance and versatility. M-DIP achieved better image quality metrics on phantom data, higher reader scores on in-vivo cine and LGE data, and comparable scores on in-vivo perfusion data relative to another DIP-based approach. M-DIP enables high-quality reconstructions of real-time free-breathing cardiac MRI without requiring external training data. Its ability to model physiological motion and content variations makes it a promising approach for various dynamic imaging applications.
>
---
#### [replaced 024] E3D-Bench: A Benchmark for End-to-End 3D Geometric Foundation Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.01933v3](http://arxiv.org/pdf/2506.01933v3)**

> **作者:** Wenyan Cong; Yiqing Liang; Yancheng Zhang; Ziyi Yang; Yan Wang; Boris Ivanovic; Marco Pavone; Chen Chen; Zhangyang Wang; Zhiwen Fan
>
> **备注:** Project Page: https://e3dbench.github.io/
>
> **摘要:** Spatial intelligence, encompassing 3D reconstruction, perception, and reasoning, is fundamental to applications such as robotics, aerial imaging, and extended reality. A key enabler is the real-time, accurate estimation of core 3D attributes (camera parameters, point clouds, depth maps, and 3D point tracks) from unstructured or streaming imagery. Inspired by the success of large foundation models in language and 2D vision, a new class of end-to-end 3D geometric foundation models (GFMs) has emerged, directly predicting dense 3D representations in a single feed-forward pass, eliminating the need for slow or unavailable precomputed camera parameters. Since late 2023, the field has exploded with diverse variants, but systematic evaluation is lacking. In this work, we present the first comprehensive benchmark for 3D GFMs, covering five core tasks: sparse-view depth estimation, video depth estimation, 3D reconstruction, multi-view pose estimation, novel view synthesis, and spanning both standard and challenging out-of-distribution datasets. Our standardized toolkit automates dataset handling, evaluation protocols, and metric computation to ensure fair, reproducible comparisons. We evaluate 16 state-of-the-art GFMs, revealing their strengths and limitations across tasks and domains, and derive key insights to guide future model scaling and optimization. All code, evaluation scripts, and processed data will be publicly released to accelerate research in 3D spatial intelligence.
>
---
#### [replaced 025] Taming the Tri-Space Tension: ARC-Guided Hallucination Modeling and Control for Text-to-Image Generation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.04946v2](http://arxiv.org/pdf/2507.04946v2)**

> **作者:** Jianjiang Yang; Ziyan Huang
>
> **备注:** We withdraw this paper due to significant visualization errors in Figure 3 and 5 that affect the correctness of our core modeling claims and may cause misinterpretation. These figures misrepresent ARC dynamics and trajectory control
>
> **摘要:** Despite remarkable progress in image quality and prompt fidelity, text-to-image (T2I) diffusion models continue to exhibit persistent "hallucinations", where generated content subtly or significantly diverges from the intended prompt semantics. While often regarded as unpredictable artifacts, we argue that these failures reflect deeper, structured misalignments within the generative process. In this work, we propose a cognitively inspired perspective that reinterprets hallucinations as trajectory drift within a latent alignment space. Empirical observations reveal that generation unfolds within a multiaxial cognitive tension field, where the model must continuously negotiate competing demands across three key critical axes: semantic coherence, structural alignment, and knowledge grounding. We then formalize this three-axis space as the \textbf{Hallucination Tri-Space} and introduce the Alignment Risk Code (ARC): a dynamic vector representation that quantifies real-time alignment tension during generation. The magnitude of ARC captures overall misalignment, its direction identifies the dominant failure axis, and its imbalance reflects tension asymmetry. Based on this formulation, we develop the TensionModulator (TM-ARC): a lightweight controller that operates entirely in latent space. TM-ARC monitors ARC signals and applies targeted, axis-specific interventions during the sampling process. Extensive experiments on standard T2I benchmarks demonstrate that our approach significantly reduces hallucination without compromising image quality or diversity. This framework offers a unified and interpretable approach for understanding and mitigating generative failures in diffusion-based T2I systems.
>
---
#### [replaced 026] Mixture of Group Experts for Learning Invariant Representations
- **分类: cs.LG; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.09265v2](http://arxiv.org/pdf/2504.09265v2)**

> **作者:** Lei Kang; Jia Li; Mi Tian; Hua Huang
>
> **摘要:** Sparsely activated Mixture-of-Experts (MoE) models effectively increase the number of parameters while maintaining consistent computational costs per token. However, vanilla MoE models often suffer from limited diversity and specialization among experts, constraining their performance and scalability, especially as the number of experts increases. In this paper, we present a novel perspective on vanilla MoE with top-$k$ routing inspired by sparse representation. This allows us to bridge established theoretical insights from sparse representation into MoE models. Building on this foundation, we propose a group sparse regularization approach for the input of top-$k$ routing, termed Mixture of Group Experts (MoGE). MoGE indirectly regularizes experts by imposing structural constraints on the routing inputs, while preserving the original MoE architecture. Furthermore, we organize the routing input into a 2D topographic map, spatially grouping neighboring elements. This structure enables MoGE to capture representations invariant to minor transformations, thereby significantly enhancing expert diversity and specialization. Comprehensive evaluations across various Transformer models for image classification and language modeling tasks demonstrate that MoGE substantially outperforms its MoE counterpart, with minimal additional memory and computation overhead. Our approach provides a simple yet effective solution to scale the number of experts and reduce redundancy among them. The source code is included in the supplementary material and will be publicly released.
>
---
#### [replaced 027] GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation in End-to-End Autonomous Driving
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.05689v4](http://arxiv.org/pdf/2503.05689v4)**

> **作者:** Zebin Xing; Xingyu Zhang; Yang Hu; Bo Jiang; Tong He; Qian Zhang; Xiaoxiao Long; Wei Yin
>
> **摘要:** We propose GoalFlow, an end-to-end autonomous driving method for generating high-quality multimodal trajectories. In autonomous driving scenarios, there is rarely a single suitable trajectory. Recent methods have increasingly focused on modeling multimodal trajectory distributions. However, they suffer from trajectory selection complexity and reduced trajectory quality due to high trajectory divergence and inconsistencies between guidance and scene information. To address these issues, we introduce GoalFlow, a novel method that effectively constrains the generative process to produce high-quality, multimodal trajectories. To resolve the trajectory divergence problem inherent in diffusion-based methods, GoalFlow constrains the generated trajectories by introducing a goal point. GoalFlow establishes a novel scoring mechanism that selects the most appropriate goal point from the candidate points based on scene information. Furthermore, GoalFlow employs an efficient generative method, Flow Matching, to generate multimodal trajectories, and incorporates a refined scoring mechanism to select the optimal trajectory from the candidates. Our experimental results, validated on the Navsim\cite{Dauner2024_navsim}, demonstrate that GoalFlow achieves state-of-the-art performance, delivering robust multimodal trajectories for autonomous driving. GoalFlow achieved PDMS of 90.3, significantly surpassing other methods. Compared with other diffusion-policy-based methods, our approach requires only a single denoising step to obtain excellent performance. The code is available at https://github.com/YvanYin/GoalFlow.
>
---
#### [replaced 028] FluidNexus: 3D Fluid Reconstruction and Prediction from a Single Video
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.04720v2](http://arxiv.org/pdf/2503.04720v2)**

> **作者:** Yue Gao; Hong-Xing Yu; Bo Zhu; Jiajun Wu
>
> **备注:** CVPR 2025 (oral). The first two authors contributed equally. Project website: https://yuegao.me/FluidNexus
>
> **摘要:** We study reconstructing and predicting 3D fluid appearance and velocity from a single video. Current methods require multi-view videos for fluid reconstruction. We present FluidNexus, a novel framework that bridges video generation and physics simulation to tackle this task. Our key insight is to synthesize multiple novel-view videos as references for reconstruction. FluidNexus consists of two key components: (1) a novel-view video synthesizer that combines frame-wise view synthesis with video diffusion refinement for generating realistic videos, and (2) a physics-integrated particle representation coupling differentiable simulation and rendering to simultaneously facilitate 3D fluid reconstruction and prediction. To evaluate our approach, we collect two new real-world fluid datasets featuring textured backgrounds and object interactions. Our method enables dynamic novel view synthesis, future prediction, and interaction simulation from a single fluid video. Project website: https://yuegao.me/FluidNexus.
>
---
#### [replaced 029] Are Vision Transformer Representations Semantically Meaningful? A Case Study in Medical Imaging
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.01788v2](http://arxiv.org/pdf/2507.01788v2)**

> **作者:** Montasir Shams; Chashi Mahiul Islam; Shaeke Salman; Phat Tran; Xiuwen Liu
>
> **备注:** 9 pages
>
> **摘要:** Vision transformers (ViTs) have rapidly gained prominence in medical imaging tasks such as disease classification, segmentation, and detection due to their superior accuracy compared to conventional deep learning models. However, due to their size and complex interactions via the self-attention mechanism, they are not well understood. In particular, it is unclear whether the representations produced by such models are semantically meaningful. In this paper, using a projected gradient-based algorithm, we show that their representations are not semantically meaningful and they are inherently vulnerable to small changes. Images with imperceptible differences can have very different representations; on the other hand, images that should belong to different semantic classes can have nearly identical representations. Such vulnerability can lead to unreliable classification results; for example, unnoticeable changes cause the classification accuracy to be reduced by over 60\%. %. To the best of our knowledge, this is the first work to systematically demonstrate this fundamental lack of semantic meaningfulness in ViT representations for medical image classification, revealing a critical challenge for their deployment in safety-critical systems.
>
---
#### [replaced 030] Mamba-CL: Optimizing Selective State Space Model in Null Space for Continual Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.15469v2](http://arxiv.org/pdf/2411.15469v2)**

> **作者:** De Cheng; Yue Lu; Lingfeng He; Shizhou Zhang; Xi Yang; Nannan Wang; Xinbo Gao
>
> **摘要:** Continual Learning (CL) aims to equip AI models with the ability to learn a sequence of tasks over time, without forgetting previously learned knowledge. Recently, State Space Models (SSMs), particularly the Mamba model, have achieved notable success in computer vision. Building on the strengths of SSMs, this study explores leveraging the Mamba model for CL. Therefore, we introduce Mamba-CL, a framework that continuously fine-tunes the core SSMs of the large-scale Mamba foundation model by updating parameters orthogonal to the feature subspace of previous tasks. This approach theoretically guarantees the consistency objective aiming to preserves consistent output for each SSM module across both previous and current tasks, so as to overcome catastrophic forgetting issue. Specifically, we achieve this goal by deducing the overall consistency constraints on four key time-invariant parameters in the Mamba model, streamlining its recurrent state-space structure and non-linear discretization process in SSM. In practice, we apply the null-space projection to efficiently implement the orthogonality within Mamba model. Extensive experiments on four class-incremental benchmarks demonstrate the effectiveness of Mamba-CL for anti-forgetting, achieving superior performances to state-of-the-art methods. Code is available in the supplementary materials.
>
---
#### [replaced 031] Adversarial Defenses via Vector Quantization
- **分类: cs.LG; cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2305.13651v2](http://arxiv.org/pdf/2305.13651v2)**

> **作者:** Zhiyi Dong; Yongyi Mao
>
> **备注:** This is the author-accepted version of our paper published in Neurocomputing. The final published version is available at: https://doi.org/10.1016/j.neucom.2025.130703
>
> **摘要:** Adversarial attacks pose significant challenges to the robustness of modern deep neural networks in computer vision, and defending these networks against adversarial attacks has attracted intense research efforts. Among various defense strategies, preprocessing-based defenses are practically appealing since there is no need to train the network under protection. However, such approaches typically do not achieve comparable robustness as other methods such as adversarial training. In this paper, we propose a novel framework for preprocessing-based defenses, where a vector quantizer is used as a preprocessor. This framework, inspired by and extended from Randomized Discretization (RandDisc), is theoretically principled by rate-distortion theory: indeed, RandDisc may be viewed as a scalar quantizer, and rate-distortion theory suggests that such quantization schemes are inferior to vector quantization. In our framework, the preprocessing vector quantizer treats the input image as a collection of patches and finds a set of representative patches based on the patch distributions; each original patch is then modified according to the representative patches close to it. We present two lightweight defenses in this framework, referred to as patched RandDisc (pRD) and sliding-window RandDisc (swRD), where the patches are disjoint in the former and overlapping in the latter. We show that vector-quantization-based defenses have certifiable robust accuracy and that pRD and swRD demonstrate state-of-the-art performances, surpassing RandDisc by a large margin. Notably, the proposed defenses possess the obfuscated gradients property. Our experiments however show that pRD and swRD remain effective under the STE and EOT attacks, which are designed specifically for defenses with gradient obfuscation. ...
>
---
#### [replaced 032] Attention-Enhanced Deep Learning Ensemble for Breast Density Classification in Mammography
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.06410v2](http://arxiv.org/pdf/2507.06410v2)**

> **作者:** Peyman Sharifian; Xiaotong Hong; Alireza Karimian; Mehdi Amini; Hossein Arabi
>
> **备注:** 2025 IEEE Nuclear Science Symposium, Medical Imaging Conference and Room Temperature Semiconductor Detector Conference
>
> **摘要:** Breast density assessment is a crucial component of mammographic interpretation, with high breast density (BI-RADS categories C and D) representing both a significant risk factor for developing breast cancer and a technical challenge for tumor detection. This study proposes an automated deep learning system for robust binary classification of breast density (low: A/B vs. high: C/D) using the VinDr-Mammo dataset. We implemented and compared four advanced convolutional neural networks: ResNet18, ResNet50, EfficientNet-B0, and DenseNet121, each enhanced with channel attention mechanisms. To address the inherent class imbalance, we developed a novel Combined Focal Label Smoothing Loss function that integrates focal loss, label smoothing, and class-balanced weighting. Our preprocessing pipeline incorporated advanced techniques, including contrast-limited adaptive histogram equalization (CLAHE) and comprehensive data augmentation. The individual models were combined through an optimized ensemble voting approach, achieving superior performance (AUC: 0.963, F1-score: 0.952) compared to any single model. This system demonstrates significant potential to standardize density assessments in clinical practice, potentially improving screening efficiency and early cancer detection rates while reducing inter-observer variability among radiologists.
>
---
#### [replaced 033] ReconDreamer++: Harmonizing Generative and Reconstructive Models for Driving Scene Representation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18438v2](http://arxiv.org/pdf/2503.18438v2)**

> **作者:** Guosheng Zhao; Xiaofeng Wang; Chaojun Ni; Zheng Zhu; Wenkang Qin; Guan Huang; Xingang Wang
>
> **备注:** Project Page: https://recondreamer-plus.github.io/
>
> **摘要:** Combining reconstruction models with generative models has emerged as a promising paradigm for closed-loop simulation in autonomous driving. For example, ReconDreamer has demonstrated remarkable success in rendering large-scale maneuvers. However, a significant gap remains between the generated data and real-world sensor observations, particularly in terms of fidelity for structured elements, such as the ground surface. To address these challenges, we propose ReconDreamer++, an enhanced framework that significantly improves the overall rendering quality by mitigating the domain gap and refining the representation of the ground surface. Specifically, ReconDreamer++ introduces the Novel Trajectory Deformable Network (NTDNet), which leverages learnable spatial deformation mechanisms to bridge the domain gap between synthesized novel views and original sensor observations. Moreover, for structured elements such as the ground surface, we preserve geometric prior knowledge in 3D Gaussians, and the optimization process focuses on refining appearance attributes while preserving the underlying geometric structure. Experimental evaluations conducted on multiple datasets (Waymo, nuScenes, PandaSet, and EUVS) confirm the superior performance of ReconDreamer++. Specifically, on Waymo, ReconDreamer++ achieves performance comparable to Street Gaussians for the original trajectory while significantly outperforming ReconDreamer on novel trajectories. In particular, it achieves substantial improvements, including a 6.1% increase in NTA-IoU, a 23. 0% improvement in FID, and a remarkable 4.5% gain in the ground surface metric NTL-IoU, highlighting its effectiveness in accurately reconstructing structured elements such as the road surface.
>
---
#### [replaced 034] EyeTrAES: Fine-grained, Low-Latency Eye Tracking via Adaptive Event Slicing
- **分类: cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2409.18813v3](http://arxiv.org/pdf/2409.18813v3)**

> **作者:** Argha Sen; Nuwan Bandara; Ila Gokarn; Thivya Kandappu; Archan Misra
>
> **备注:** 32 pages,15 figures,
>
> **摘要:** Eye-tracking technology has gained significant attention in recent years due to its wide range of applications in human-computer interaction, virtual and augmented reality, and wearable health. Traditional RGB camera-based eye-tracking systems often struggle with poor temporal resolution and computational constraints, limiting their effectiveness in capturing rapid eye movements. To address these limitations, we propose EyeTrAES, a novel approach using neuromorphic event cameras for high-fidelity tracking of natural pupillary movement that shows significant kinematic variance. One of EyeTrAES's highlights is the use of a novel adaptive windowing/slicing algorithm that ensures just the right amount of descriptive asynchronous event data accumulation within an event frame, across a wide range of eye movement patterns. EyeTrAES then applies lightweight image processing functions over accumulated event frames from just a single eye to perform pupil segmentation and tracking. We show that these methods boost pupil tracking fidelity by 6+%, achieving IoU~=92%, while incurring at least 3x lower latency than competing pure event-based eye tracking alternatives [38]. We additionally demonstrate that the microscopic pupillary motion captured by EyeTrAES exhibits distinctive variations across individuals and can thus serve as a biometric fingerprint. For robust user authentication, we train a lightweight per-user Random Forest classifier using a novel feature vector of short-term pupillary kinematics, comprising a sliding window of pupil (location, velocity, acceleration) triples. Experimental studies with two different datasets demonstrate that the EyeTrAES-based authentication technique can simultaneously achieve high authentication accuracy (~=0.82) and low processing latency (~=12ms), and significantly outperform multiple state-of-the-art competitive baselines.
>
---
#### [replaced 035] Diffusion Model-based Data Augmentation Method for Fetal Head Ultrasound Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.23664v2](http://arxiv.org/pdf/2506.23664v2)**

> **作者:** Fangyijie Wang; Kevin Whelan; Félix Balado; Kathleen M. Curran; Guénolé Silvestre
>
> **备注:** Accepted at Irish Machine Vision and Image Processing Conference (IMVIP) 2025
>
> **摘要:** Medical image data is less accessible than in other domains due to privacy and regulatory constraints. In addition, labeling requires costly, time-intensive manual image annotation by clinical experts. To overcome these challenges, synthetic medical data generation offers a promising solution. Generative AI (GenAI), employing generative deep learning models, has proven effective at producing realistic synthetic images. This study proposes a novel mask-guided GenAI approach using diffusion models to generate synthetic fetal head ultrasound images paired with segmentation masks. These synthetic pairs augment real datasets for supervised fine-tuning of the Segment Anything Model (SAM). Our results show that the synthetic data captures real image features effectively, and this approach reaches state-of-the-art fetal head segmentation, especially when trained with a limited number of real image-mask pairs. In particular, the segmentation reaches Dice Scores of 94.66\% and 94.38\% using a handful of ultrasound images from the Spanish and African cohorts, respectively. Our code, models, and data are available on GitHub.
>
---
#### [replaced 036] video-SALMONN 2: Captioning-Enhanced Audio-Visual Large Language Models
- **分类: cs.CV; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.15220v2](http://arxiv.org/pdf/2506.15220v2)**

> **作者:** Changli Tang; Yixuan Li; Yudong Yang; Jimin Zhuang; Guangzhi Sun; Wei Li; Zejun Ma; Chao Zhang
>
> **摘要:** Videos contain a wealth of information, and generating detailed and accurate descriptions in natural language is a key aspect of video understanding. In this paper, we present video-SALMONN 2, an advanced audio-visual large language model (LLM) with low-rank adaptation (LoRA) designed for enhanced video (with paired audio) captioning through directed preference optimisation (DPO). We propose new metrics to evaluate the completeness and accuracy of video descriptions, which are optimised using DPO. To further improve training, we propose a novel multi-round DPO (MrDPO) approach, which involves periodically updating the DPO reference model, merging and re-initialising the LoRA module as a proxy for parameter updates after each training round (1,000 steps), and incorporating guidance from ground-truth video captions to stabilise the process. Experimental results show that MrDPO significantly enhances video-SALMONN 2's captioning accuracy, reducing the captioning error rates by 28\%. The final video-SALMONN 2 model, with just 7 billion parameters, surpasses leading models such as GPT-4o and Gemini-1.5-Pro in video captioning tasks, while maintaining highly competitive performance to the state-of-the-art on widely used video question-answering benchmarks among models of similar size. Codes are available at \href{https://github.com/bytedance/video-SALMONN-2}{https://github.com/bytedance/video-SALMONN-2}.
>
---
#### [replaced 037] Online Continual Learning via Spiking Neural Networks with Sleep Enhanced Latent Replay
- **分类: cs.NE; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.02901v2](http://arxiv.org/pdf/2507.02901v2)**

> **作者:** Erliang Lin; Wenbin Luo; Wei Jia; Yu Chen; Shaofu Yang
>
> **备注:** 9 pages, 4figures
>
> **摘要:** Edge computing scenarios necessitate the development of hardware-efficient online continual learning algorithms to be adaptive to dynamic environment. However, existing algorithms always suffer from high memory overhead and bias towards recently trained tasks. To tackle these issues, this paper proposes a novel online continual learning approach termed as SESLR, which incorporates a sleep enhanced latent replay scheme with spiking neural networks (SNNs). SESLR leverages SNNs' binary spike characteristics to store replay features in single bits, significantly reducing memory overhead. Furthermore, inspired by biological sleep-wake cycles, SESLR introduces a noise-enhanced sleep phase where the model exclusively trains on replay samples with controlled noise injection, effectively mitigating classification bias towards new classes. Extensive experiments on both conventional (MNIST, CIFAR10) and neuromorphic (NMNIST, CIFAR10-DVS) datasets demonstrate SESLR's effectiveness. On Split CIFAR10, SESLR achieves nearly 30% improvement in average accuracy with only one-third of the memory consumption compared to baseline methods. On Split CIFAR10-DVS, it improves accuracy by approximately 10% while reducing memory overhead by a factor of 32. These results validate SESLR as a promising solution for online continual learning in resource-constrained edge computing scenarios.
>
---
#### [replaced 038] Dance Like a Chicken: Low-Rank Stylization for Human Motion Diffusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.19557v2](http://arxiv.org/pdf/2503.19557v2)**

> **作者:** Haim Sawdayee; Chuan Guo; Guy Tevet; Bing Zhou; Jian Wang; Amit H. Bermano
>
> **备注:** Project page at https://haimsaw.github.io/LoRA-MDM/
>
> **摘要:** Text-to-motion generative models span a wide range of 3D human actions but struggle with nuanced stylistic attributes such as a "Chicken" style. Due to the scarcity of style-specific data, existing approaches pull the generative prior towards a reference style, which often results in out-of-distribution low quality generations. In this work, we introduce LoRA-MDM, a lightweight framework for motion stylization that generalizes to complex actions while maintaining editability. Our key insight is that adapting the generative prior to include the style, while preserving its overall distribution, is more effective than modifying each individual motion during generation. Building on this idea, LoRA-MDM learns to adapt the prior to include the reference style using only a few samples. The style can then be used in the context of different textual prompts for generation. The low-rank adaptation shifts the motion manifold in a semantically meaningful way, enabling realistic style infusion even for actions not present in the reference samples. Moreover, preserving the distribution structure enables advanced operations such as style blending and motion editing. We compare LoRA-MDM to state-of-the-art stylized motion generation methods and demonstrate a favorable balance between text fidelity and style consistency.
>
---
#### [replaced 039] Multi-modal Generative AI: Multi-modal LLMs, Diffusions and the Unification
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.14993v2](http://arxiv.org/pdf/2409.14993v2)**

> **作者:** Xin Wang; Yuwei Zhou; Bin Huang; Hong Chen; Wenwu Zhu
>
> **备注:** 20 pages, 11 figures, 2 tables
>
> **摘要:** Multi-modal generative AI (Artificial Intelligence) has attracted increasing attention from both academia and industry. Particularly, two dominant families of techniques have emerged: i) Multi-modal large language models (LLMs) demonstrate impressive ability for multi-modal understanding; and ii) Diffusion models exhibit remarkable multi-modal powers in terms of multi-modal generation. Therefore, this paper provides a comprehensive overview of multi-modal generative AI, including multi-modal LLMs, diffusions, and the unification for understanding and generation. To lay a solid foundation for unified models, we first provide a detailed review of both multi-modal LLMs and diffusion models respectively, including their probabilistic modeling procedure, multi-modal architecture design, and advanced applications to image/video LLMs as well as text-to-image/video generation. Furthermore, we explore the emerging efforts toward unified models for understanding and generation. To achieve the unification of understanding and generation, we investigate key designs including autoregressive-based and diffusion-based modeling, as well as dense and Mixture-of-Experts (MoE) architectures. We then introduce several strategies for unified models, analyzing their potential advantages and disadvantages. In addition, we summarize the common datasets widely used for multi-modal generative AI pretraining. Last but not least, we present several challenging future research directions which may contribute to the ongoing advancement of multi-modal generative AI.
>
---
#### [replaced 040] Leveraging the Structure of Medical Data for Improved Representation Learning
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.02987v2](http://arxiv.org/pdf/2507.02987v2)**

> **作者:** Andrea Agostini; Sonia Laguna; Alain Ryser; Samuel Ruiperez-Campillo; Moritz Vandenhirtz; Nicolas Deperrois; Farhad Nooralahzadeh; Michael Krauthammer; Thomas M. Sutter; Julia E. Vogt
>
> **摘要:** Building generalizable medical AI systems requires pretraining strategies that are data-efficient and domain-aware. Unlike internet-scale corpora, clinical datasets such as MIMIC-CXR offer limited image counts and scarce annotations, but exhibit rich internal structure through multi-view imaging. We propose a self-supervised framework that leverages the inherent structure of medical datasets. Specifically, we treat paired chest X-rays (i.e., frontal and lateral views) as natural positive pairs, learning to reconstruct each view from sparse patches while aligning their latent embeddings. Our method requires no textual supervision and produces informative representations. Evaluated on MIMIC-CXR, we show strong performance compared to supervised objectives and baselines being trained without leveraging structure. This work provides a lightweight, modality-agnostic blueprint for domain-specific pretraining where data is structured but scarce
>
---
#### [replaced 041] VMem: Consistent Interactive Video Scene Generation with Surfel-Indexed View Memory
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18903v2](http://arxiv.org/pdf/2506.18903v2)**

> **作者:** Runjia Li; Philip Torr; Andrea Vedaldi; Tomas Jakab
>
> **备注:** Project page: https://v-mem.github.io
>
> **摘要:** We propose a novel memory mechanism to build video generators that can explore environments interactively. Similar results have previously been achieved by out-painting 2D views of the scene while incrementally reconstructing its 3D geometry, which quickly accumulates errors, or by video generators with a short context window, which struggle to maintain scene coherence over the long term. To address these limitations, we introduce Surfel-Indexed View Memory (VMem), a mechanism that remembers past views by indexing them geometrically based on the 3D surface elements (surfels) they have observed. VMem enables the efficient retrieval of the most relevant past views when generating new ones. By focusing only on these relevant views, our method produces consistent explorations of imagined environments at a fraction of the computational cost of using all past views as context. We evaluate our approach on challenging long-term scene synthesis benchmarks and demonstrate superior performance compared to existing methods in maintaining scene coherence and camera control.
>
---
#### [replaced 042] GGTalker: Talking Head Systhesis with Generalizable Gaussian Priors and Identity-Specific Adaptation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21513v2](http://arxiv.org/pdf/2506.21513v2)**

> **作者:** Wentao Hu; Shunkai Li; Ziqiao Peng; Haoxian Zhang; Fan Shi; Xiaoqiang Liu; Pengfei Wan; Di Zhang; Hui Tian
>
> **备注:** ICCV 2025, Project page: https://vincenthu19.github.io/GGTalker/
>
> **摘要:** Creating high-quality, generalizable speech-driven 3D talking heads remains a persistent challenge. Previous methods achieve satisfactory results for fixed viewpoints and small-scale audio variations, but they struggle with large head rotations and out-of-distribution (OOD) audio. Moreover, they are constrained by the need for time-consuming, identity-specific training. We believe the core issue lies in the lack of sufficient 3D priors, which limits the extrapolation capabilities of synthesized talking heads. To address this, we propose GGTalker, which synthesizes talking heads through a combination of generalizable priors and identity-specific adaptation. We introduce a two-stage Prior-Adaptation training strategy to learn Gaussian head priors and adapt to individual characteristics. We train Audio-Expression and Expression-Visual priors to capture the universal patterns of lip movements and the general distribution of head textures. During the Customized Adaptation, individual speaking styles and texture details are precisely modeled. Additionally, we introduce a color MLP to generate fine-grained, motion-aligned textures and a Body Inpainter to blend rendered results with the background, producing indistinguishable, photorealistic video frames. Comprehensive experiments show that GGTalker achieves state-of-the-art performance in rendering quality, 3D consistency, lip-sync accuracy, and training efficiency.
>
---
#### [replaced 043] EEPNet-V2: Patch-to-Pixel Solution for Efficient Cross-Modal Registration between LiDAR Point Cloud and Camera Image
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.15285v2](http://arxiv.org/pdf/2503.15285v2)**

> **作者:** Yuanchao Yue; Hui Yuan; Zhengxin Li; Shuai Li; Wei Zhang
>
> **摘要:** The primary requirement for cross-modal data fusion is the precise alignment of data from different sensors. However, the calibration between LiDAR point clouds and camera images is typically time-consuming and needs external calibration board or specific environmental features. Cross-modal registration effectively solves this problem by aligning the data directly without requiring external calibration. However, due to the domain gap between the point cloud and the image, existing methods rarely achieve satisfactory registration accuracy while maintaining real-time performance. To address this issue, we propose a framework that projects point clouds into several 2D representations for matching with camera images, which not only leverages the geometric characteristic of LiDAR point clouds effectively but also bridge the domain gap between the point cloud and image. Moreover, to tackle the challenges of cross modal differences and the limited overlap between LiDAR point clouds and images in the image matching task, we introduce a multi-scale feature extraction network to effectively extract features from both camera images and the projection maps of LiDAR point cloud. Additionally, we propose a patch-to-pixel matching network to provide more effective supervision and achieve high accuracy. We validate the performance of our model through experiments on the KITTI and nuScenes datasets. Experimental results demonstrate the the proposed method achieves real-time performance and extremely high registration accuracy. Specifically, on the KITTI dataset, our model achieves a registration accuracy rate of over 99\%. Our code is released at: https://github.com/ESRSchao/EEPNet-V2.
>
---
#### [replaced 044] MedTrinity-25M: A Large-scale Multimodal Dataset with Multigranular Annotations for Medicine
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.02900v3](http://arxiv.org/pdf/2408.02900v3)**

> **作者:** Yunfei Xie; Ce Zhou; Lang Gao; Juncheng Wu; Xianhang Li; Hong-Yu Zhou; Sheng Liu; Lei Xing; James Zou; Cihang Xie; Yuyin Zhou
>
> **备注:** The dataset is publicly available at https://yunfeixie233.github.io/MedTrinity-25M/. Accepted to ICLR 2025
>
> **摘要:** This paper introduces MedTrinity-25M, a comprehensive, large-scale multimodal dataset for medicine, covering over 25 million images across 10 modalities with multigranular annotations for more than 65 diseases. These multigranular annotations encompass both global information, such as modality and organ detection, and local information like ROI analysis, lesion texture, and region-wise correlations. Unlike the existing multimodal datasets, which are limited by the availability of image-text pairs, we have developed the first automated pipeline that scales up multimodal data by generating multigranular visual and textual annotations in the form of image-ROI-description triplets without the need for any paired text descriptions. Specifically, data from over 30 different sources have been collected, preprocessed, and grounded using domain-specific expert models to identify ROIs related to abnormal regions. We then build a comprehensive knowledge base and prompt multimodal large language models to perform retrieval-augmented generation with the identified ROIs as guidance, resulting in multigranular textual descriptions. Compared to existing datasets, MedTrinity-25M provides the most enriched annotations, supporting a comprehensive range of multimodal tasks such as captioning and report generation, as well as vision-centric tasks like classification and segmentation. We propose LLaVA-Tri by pretraining LLaVA on MedTrinity-25M, achieving state-of-the-art performance on VQA-RAD, SLAKE, and PathVQA, surpassing representative SOTA multimodal large language models. Furthermore, MedTrinity-25M can also be utilized to support large-scale pre-training of multimodal medical AI models, contributing to the development of future foundation models in the medical domain. We will make our dataset available.
>
---
#### [replaced 045] SkipVAR: Accelerating Visual Autoregressive Modeling via Adaptive Frequency-Aware Skipping
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.08908v3](http://arxiv.org/pdf/2506.08908v3)**

> **作者:** Jiajun Li; Yue Ma; Xinyu Zhang; Qingyan Wei; Songhua Liu; Linfeng Zhang
>
> **摘要:** Recent studies on Visual Autoregressive (VAR) models have highlighted that high-frequency components, or later steps, in the generation process contribute disproportionately to inference latency. However, the underlying computational redundancy involved in these steps has yet to be thoroughly investigated. In this paper, we conduct an in-depth analysis of the VAR inference process and identify two primary sources of inefficiency: step redundancy and unconditional branch redundancy. To address step redundancy, we propose an automatic step-skipping strategy that selectively omits unnecessary generation steps to improve efficiency. For unconditional branch redundancy, we observe that the information gap between the conditional and unconditional branches is minimal. Leveraging this insight, we introduce unconditional branch replacement, a technique that bypasses the unconditional branch to reduce computational cost. Notably, we observe that the effectiveness of acceleration strategies varies significantly across different samples. Motivated by this, we propose SkipVAR, a sample-adaptive framework that leverages frequency information to dynamically select the most suitable acceleration strategy for each instance. To evaluate the role of high-frequency information, we introduce high-variation benchmark datasets that test model sensitivity to fine details. Extensive experiments show SkipVAR achieves over 0.88 average SSIM with up to 1.81x overall acceleration and 2.62x speedup on the GenEval benchmark, maintaining model quality. These results confirm the effectiveness of frequency-aware, training-free adaptive acceleration for scalable autoregressive image generation. Our code is available at https://github.com/fakerone-li/SkipVAR and has been publicly released.
>
---
#### [replaced 046] C3T: Cross-modal Transfer Through Time for Sensor-based Human Activity Recognition
- **分类: cs.CV; cs.AI; cs.HC; cs.LG; eess.SP**

- **链接: [http://arxiv.org/pdf/2407.16803v4](http://arxiv.org/pdf/2407.16803v4)**

> **作者:** Abhi Kamboj; Anh Duy Nguyen; Minh N. Do
>
> **摘要:** In order to unlock the potential of diverse sensors, we investigate a method to transfer knowledge between time-series modalities using a multimodal \textit{temporal} representation space for Human Activity Recognition (HAR). Specifically, we explore the setting where the modality used in testing has no labeled data during training, which we refer to as Unsupervised Modality Adaptation (UMA). We categorize existing UMA approaches as Student-Teacher or Contrastive Alignment methods. These methods typically compress continuous-time data samples into single latent vectors during alignment, inhibiting their ability to transfer temporal information through real-world temporal distortions. To address this, we introduce Cross-modal Transfer Through Time (C3T), which preserves temporal information during alignment to handle dynamic sensor data better. C3T achieves this by aligning a set of temporal latent vectors across sensing modalities. Our extensive experiments on various camera+IMU datasets demonstrate that C3T outperforms existing methods in UMA by at least 8% in accuracy and shows superior robustness to temporal distortions such as time-shift, misalignment, and dilation. Our findings suggest that C3T has significant potential for developing generalizable models for time-series sensor data, opening new avenues for various multimodal applications.
>
---
#### [replaced 047] Boundary Learning by Using Weighted Propagation in Convolution Network
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/1905.09226v3](http://arxiv.org/pdf/1905.09226v3)**

> **作者:** Wei Liu; Jiahao Chen; Chuni Liu; Xiaojuan Ban; Boyuan Ma; Hao Wang; Weihua Xue; Yu Guo
>
> **备注:** technical report
>
> **摘要:** In material science, image segmentation is of great significance for quantitative analysis of microstructures. Here, we propose a novel Weighted Propagation Convolution Neural Network based on U-Net (WPU-Net) to detect boundary in poly-crystalline microscopic images. We introduce spatial consistency into network to eliminate the defects in raw microscopic image. And we customize adaptive boundary weight for each pixel in each grain, so that it leads the network to preserve grain's geometric and topological characteristics. Moreover, we provide our dataset with the goal of advancing the development of image processing in materials science. Experiments demonstrate that the proposed method achieves promising performance in both of objective and subjective assessment. In boundary detection task, it reduces the error rate by 7\%, which outperforms state-of-the-art methods by a large margin.
>
---
#### [replaced 048] Open-source automatic pipeline for efficient conversion of large-scale point clouds to IFC format
- **分类: cs.CV; cs.SE**

- **链接: [http://arxiv.org/pdf/2503.11498v3](http://arxiv.org/pdf/2503.11498v3)**

> **作者:** Slávek Zbirovský; Václav Nežerka
>
> **备注:** published version, 23 pages, 25 figures
>
> **摘要:** Building Information Modeling (BIM) is an essential component in the sustainable reconstruction and revitalization of ageing structures. However, model creation usually relies on laborious manual transformation of the unstructured point cloud data provided by laser scans or photogrammetry. This paper presents Cloud2BIM, an open-source software tool designed to automate the conversion of point clouds into BIM models compliant with the Industry Foundation Classes (IFC) standard. Cloud2BIM integrates advanced algorithms for wall and slab segmentation, opening detection, and room zoning based on real wall surfaces, resulting in a comprehensive and fully automated workflow. Unlike existing tools, it avoids computationally- and calibration-intensive techniques such as RANSAC, supports non-orthogonal geometries, and provides unprecedented processing speed-achieving results up to seven times faster than fastest competing solutions. Systematic validation using benchmark datasets confirms that Cloud2BIM is an easy-to-use, efficient, and scalable solution for generating accurate BIM models, capable of converting extensive point cloud datasets for entire buildings into IFC format with minimal user input.
>
---
#### [replaced 049] Localized Concept Erasure for Text-to-Image Diffusion Models Using Training-Free Gated Low-Rank Adaptation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.12356v3](http://arxiv.org/pdf/2503.12356v3)**

> **作者:** Byung Hyun Lee; Sungjin Lim; Se Young Chun
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** Fine-tuning based concept erasing has demonstrated promising results in preventing generation of harmful contents from text-to-image diffusion models by removing target concepts while preserving remaining concepts. To maintain the generation capability of diffusion models after concept erasure, it is necessary to remove only the image region containing the target concept when it locally appears in an image, leaving other regions intact. However, prior arts often compromise fidelity of the other image regions in order to erase the localized target concept appearing in a specific area, thereby reducing the overall performance of image generation. To address these limitations, we first introduce a framework called localized concept erasure, which allows for the deletion of only the specific area containing the target concept in the image while preserving the other regions. As a solution for the localized concept erasure, we propose a training-free approach, dubbed Gated Low-rank adaptation for Concept Erasure (GLoCE), that injects a lightweight module into the diffusion model. GLoCE consists of low-rank matrices and a simple gate, determined only by several generation steps for concepts without training. By directly applying GLoCE to image embeddings and designing the gate to activate only for target concepts, GLoCE can selectively remove only the region of the target concepts, even when target and remaining concepts coexist within an image. Extensive experiments demonstrated GLoCE not only improves the image fidelity to text prompts after erasing the localized target concepts, but also outperforms prior arts in efficacy, specificity, and robustness by large margin and can be extended to mass concept erasure.
>
---
#### [replaced 050] Information-driven design of imaging systems
- **分类: physics.optics; cs.CV; cs.IT; eess.IV; math.IT; physics.data-an**

- **链接: [http://arxiv.org/pdf/2405.20559v4](http://arxiv.org/pdf/2405.20559v4)**

> **作者:** Henry Pinkard; Leyla Kabuli; Eric Markley; Tiffany Chien; Jiantao Jiao; Laura Waller
>
> **摘要:** In modern imaging systems that computationally process raw measurements before or instead of human viewing, information content matters more than visual appearance. However, developing information estimators that can handle the complexity of real-world measurements yet remain practical enough for widespread use has proven challenging. We introduce a data-driven approach for estimating mutual information between unknown objects and their noisy measurements. Our technique fits probabilistic models to measurements and their noise processes, quantifying information content without requiring ground truth data or making assumptions about object structure. We validate our approach across diverse applications-color photography, radio astronomy, lensless imaging, and microscopy-demonstrating that information estimates reliably predict system performance. Finally, we introduce Information-Driven Encoder Analysis Learning (IDEAL), which optimizes imaging systems to maximize information capture. Our work unlocks information theory as a powerful, practical tool for analyzing and designing imaging systems across a broad range of applications. A video summarizing this work can be found at: https://waller-lab.github.io/EncodingInformationWebsite/
>
---
#### [replaced 051] From Images to Signals: Are Large Vision Models Useful for Time Series Analysis?
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.24030v2](http://arxiv.org/pdf/2505.24030v2)**

> **作者:** Ziming Zhao; ChengAo Shen; Hanghang Tong; Dongjin Song; Zhigang Deng; Qingsong Wen; Jingchao Ni
>
> **摘要:** Transformer-based models have gained increasing attention in time series research, driving interest in Large Language Models (LLMs) and foundation models for time series analysis. As the field moves toward multi-modality, Large Vision Models (LVMs) are emerging as a promising direction. In the past, the effectiveness of Transformer and LLMs in time series has been debated. When it comes to LVMs, a similar question arises: are LVMs truely useful for time series analysis? To address it, we design and conduct the first principled study involving 4 LVMs, 8 imaging methods, 18 datasets and 26 baselines across both high-level (classification) and low-level (forecasting) tasks, with extensive ablation analysis. Our findings indicate LVMs are indeed useful for time series classification but face challenges in forecasting. Although effective, the contemporary best LVM forecasters are limited to specific types of LVMs and imaging methods, exhibit a bias toward forecasting periods, and have limited ability to utilize long look-back windows. We hope our findings could serve as a cornerstone for future research on LVM- and multimodal-based solutions to different time series tasks.
>
---
#### [replaced 052] Diffusion Augmented Retrieval: A Training-Free Approach to Interactive Text-to-Image Retrieval
- **分类: cs.IR; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.15379v2](http://arxiv.org/pdf/2501.15379v2)**

> **作者:** Zijun Long; Kangheng Liang; Gerardo Aragon-Camarasa; Richard Mccreadie; Paul Henderson
>
> **摘要:** Interactive Text-to-image retrieval (I-TIR) is an important enabler for a wide range of state-of-the-art services in domains such as e-commerce and education. However, current methods rely on finetuned Multimodal Large Language Models (MLLMs), which are costly to train and update, and exhibit poor generalizability. This latter issue is of particular concern, as: 1) finetuning narrows the pretrained distribution of MLLMs, thereby reducing generalizability; and 2) I-TIR introduces increasing query diversity and complexity. As a result, I-TIR solutions are highly likely to encounter queries and images not well represented in any training dataset. To address this, we propose leveraging Diffusion Models (DMs) for text-to-image mapping, to avoid finetuning MLLMs while preserving robust performance on complex queries. Specifically, we introduce Diffusion Augmented Retrieval (DAR), a framework that generates multiple intermediate representations via LLM-based dialogue refinements and DMs, producing a richer depiction of the user's information needs. This augmented representation facilitates more accurate identification of semantically and visually related images. Extensive experiments on four benchmarks show that for simple queries, DAR achieves results on par with finetuned I-TIR models, yet without incurring their tuning overhead. Moreover, as queries become more complex through additional conversational turns, DAR surpasses finetuned I-TIR models by up to 7.61% in Hits@10 after ten turns, illustrating its improved generalization for more intricate queries.
>
---
#### [replaced 053] VOTE: Vision-Language-Action Optimization with Trajectory Ensemble Voting
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.05116v2](http://arxiv.org/pdf/2507.05116v2)**

> **作者:** Juyi Lin; Amir Taherin; Arash Akbari; Arman Akbari; Lei Lu; Guangyu Chen; Taskin Padir; Xiaomeng Yang; Weiwei Chen; Yiqian Li; Xue Lin; David Kaeli; Pu Zhao; Yanzhi Wang
>
> **摘要:** Recent large-scale Vision Language Action (VLA) models have shown superior performance in robotic manipulation tasks guided by natural language. However, their generalization remains limited when applied to novel objects or unfamiliar environments that lie outside the training distribution. To address this, many existing approaches integrate additional components such as depth estimation, segmentation, or even diffusion to improve generalization, at the cost of adding significant computation overhead, resulting in low efficiency. This motivates the exploration of efficient action prediction methods, which are independent of additional high-level visual representations or diffusion techniques. In this work, we propose VOTE, an efficient and general framework for the optimization and acceleration of VLA models. In details, we propose a novel tokenizer-free fine-tuning approach for parallel accurate action prediction, which reduces computational overhead and accelerates inference speed. Additionally, we adopt an ensemble voting strategy for the action sampling, which significantly improves model performance and enhances generalization. Experimental results show that our method achieves state-of-the-art performance with 35x faster inference and 145 Hz throughput. All the details and codes will be open-sourced.
>
---
#### [replaced 054] SVIP: Semantically Contextualized Visual Patches for Zero-Shot Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10252v2](http://arxiv.org/pdf/2503.10252v2)**

> **作者:** Zhi Chen; Zecheng Zhao; Jingcai Guo; Jingjing Li; Zi Huang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Zero-shot learning (ZSL) aims to recognize unseen classes without labeled training examples by leveraging class-level semantic descriptors such as attributes. A fundamental challenge in ZSL is semantic misalignment, where semantic-unrelated information involved in visual features introduce ambiguity to visual-semantic interaction. Unlike existing methods that suppress semantic-unrelated information post hoc either in the feature space or the model space, we propose addressing this issue at the input stage, preventing semantic-unrelated patches from propagating through the network. To this end, we introduce Semantically contextualized VIsual Patches (SVIP) for ZSL, a transformer-based framework designed to enhance visual-semantic alignment. Specifically, we propose a self-supervised patch selection mechanism that preemptively learns to identify semantic-unrelated patches in the input space. This is trained with the supervision from aggregated attention scores across all transformer layers, which estimate each patch's semantic score. As removing semantic-unrelated patches from the input sequence may disrupt object structure, we replace them with learnable patch embeddings. With initialization from word embeddings, we can ensure they remain semantically meaningful throughout feature extraction. Extensive experiments on ZSL benchmarks demonstrate that SVIP achieves state-of-the-art performance results while providing more interpretable and semantically rich feature representations. Code is available at https://github.com/uqzhichen/SVIP.
>
---
#### [replaced 055] One Trajectory, One Token: Grounded Video Tokenization via Panoptic Sub-object Trajectory
- **分类: cs.CV; cs.AI; cs.GR; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.23617v2](http://arxiv.org/pdf/2505.23617v2)**

> **作者:** Chenhao Zheng; Jieyu Zhang; Mohammadreza Salehi; Ziqi Gao; Vishnu Iyengar; Norimasa Kobori; Quan Kong; Ranjay Krishna
>
> **备注:** ICCV 2025
>
> **摘要:** Effective video tokenization is critical for scaling transformer models for long videos. Current approaches tokenize videos using space-time patches, leading to excessive tokens and computational inefficiencies. The best token reduction strategies degrade performance and barely reduce the number of tokens when the camera moves. We introduce grounded video tokenization, a paradigm that organizes tokens based on panoptic sub-object trajectories rather than fixed patches. Our method aligns with fundamental perceptual principles, ensuring that tokenization reflects scene complexity rather than video duration. We propose TrajViT, a video encoder that extracts object trajectories and converts them into semantically meaningful tokens, significantly reducing redundancy while maintaining temporal coherence. Trained with contrastive learning, TrajViT significantly outperforms space-time ViT (ViT3D) across multiple video understanding benchmarks, e.g., TrajViT outperforms ViT3D by a large margin of 6% top-5 recall in average at video-text retrieval task with 10x token deduction. We also show TrajViT as a stronger model than ViT3D for being the video encoder for modern VideoLLM, obtaining an average of 5.2% performance improvement across 6 VideoQA benchmarks while having 4x faster training time and 18x less inference FLOPs. TrajViT is the first efficient encoder to consistently outperform ViT3D across diverse video analysis tasks, making it a robust and scalable solution.
>
---
#### [replaced 056] Underwater Monocular Metric Depth Estimation: Real-World Benchmarks and Synthetic Fine-Tuning with Vision Foundation Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.02148v2](http://arxiv.org/pdf/2507.02148v2)**

> **作者:** Zijie Cai; Christopher Metzler
>
> **摘要:** Monocular depth estimation has recently progressed beyond ordinal depth to provide metric depth predictions. However, its reliability in underwater environments remains limited due to light attenuation and scattering, color distortion, turbidity, and the lack of high-quality metric ground truth data. In this paper, we present a comprehensive benchmark of zero-shot and fine-tuned monocular metric depth estimation models on real-world underwater datasets with metric depth annotations, including FLSea and SQUID. We evaluated a diverse set of state-of-the-art Vision Foundation Models across a range of underwater conditions and depth ranges. Our results show that large-scale models trained on terrestrial data (real or synthetic) are effective in in-air settings, but perform poorly underwater due to significant domain shifts. To address this, we fine-tune Depth Anything V2 with a ViT-S backbone encoder on a synthetic underwater variant of the Hypersim dataset, which we simulated using a physically based underwater image formation model. Our fine-tuned model consistently improves performance across all benchmarks and outperforms baselines trained only on the clean in-air Hypersim dataset. This study presents a detailed evaluation and visualization of monocular metric depth estimation in underwater scenes, emphasizing the importance of domain adaptation and scale-aware supervision for achieving robust and generalizable metric depth predictions using foundation models in challenging environments.
>
---
#### [replaced 057] Beyond Spatial Frequency: Pixel-wise Temporal Frequency-based Deepfake Video Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.02398v2](http://arxiv.org/pdf/2507.02398v2)**

> **作者:** Taehoon Kim; Jongwook Choi; Yonghyun Jeong; Haeun Noh; Jaejun Yoo; Seungryul Baek; Jongwon Choi
>
> **备注:** accepted by iccv 2025. code is will be available at https://github.com/rama0126/PwTF-DVD
>
> **摘要:** We introduce a deepfake video detection approach that exploits pixel-wise temporal inconsistencies, which traditional spatial frequency-based detectors often overlook. Traditional detectors represent temporal information merely by stacking spatial frequency spectra across frames, resulting in the failure to detect temporal artifacts in the pixel plane. Our approach performs a 1D Fourier transform on the time axis for each pixel, extracting features highly sensitive to temporal inconsistencies, especially in areas prone to unnatural movements. To precisely locate regions containing the temporal artifacts, we introduce an attention proposal module trained in an end-to-end manner. Additionally, our joint transformer module effectively integrates pixel-wise temporal frequency features with spatio-temporal context features, expanding the range of detectable forgery artifacts. Our framework represents a significant advancement in deepfake video detection, providing robust performance across diverse and challenging detection scenarios.
>
---
#### [replaced 058] RT-OVAD: Real-Time Open-Vocabulary Aerial Object Detection via Image-Text Collaboration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.12246v3](http://arxiv.org/pdf/2408.12246v3)**

> **作者:** Guoting Wei; Xia Yuan; Yu Liu; Zhenhao Shang; Xizhe Xue; Peng Wang; Kelu Yao; Chunxia Zhao; Haokui Zhang; Rong Xiao
>
> **摘要:** Aerial object detection plays a crucial role in numerous applications. However, most existing methods focus on detecting predefined object categories, limiting their applicability in real-world open scenarios. In this paper, we extend aerial object detection to open scenarios through image-text collaboration and propose RT-OVAD, the first real-time open-vocabulary detector for aerial scenes. Specifically, we first introduce an image-to-text alignment loss to replace the conventional category regression loss, thereby eliminating category constraints. Next, we propose a lightweight image-text collaboration strategy comprising an image-text collaboration encoder and a text-guided decoder. The encoder simultaneously enhances visual features and refines textual embeddings, while the decoder guides object queries to focus on class-relevant image features. This design further improves detection accuracy without incurring significant computational overhead. Extensive experiments demonstrate that RT-OVAD consistently outperforms existing state-of-the-art methods across open-vocabulary, zero-shot, and traditional closed-set detection tasks. For instance, on the open-vocabulary aerial detection benchmarks DIOR, DOTA-v2.0, and LAE-80C, RT-OVAD achieves 87.7 AP$_{50}$, 53.8 mAP, and 23.7 mAP, respectively, surpassing the previous state-of-the-art (LAE-DINO) by 2.2, 7.0, and 3.5 points. In addition, RT-OVAD achieves an inference speed of 34 FPS on an RTX 4090 GPU, approximately three times faster than LAE-DINO (10 FPS), meeting the real-time detection requirements of diverse applications. The code will be released at https://github.com/GT-Wei/RT-OVAD.
>
---
#### [replaced 059] STAR-R1: Spatial TrAnsformation Reasoning by Reinforcing Multimodal LLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15804v3](http://arxiv.org/pdf/2505.15804v3)**

> **作者:** Zongzhao Li; Zongyang Ma; Mingze Li; Songyou Li; Yu Rong; Tingyang Xu; Ziqi Zhang; Deli Zhao; Wenbing Huang
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities across diverse tasks, yet they lag significantly behind humans in spatial reasoning. We investigate this gap through Transformation-Driven Visual Reasoning (TVR), a challenging task requiring identification of object transformations across images under varying viewpoints. While traditional Supervised Fine-Tuning (SFT) fails to generate coherent reasoning paths in cross-view settings, sparse-reward Reinforcement Learning (RL) suffers from inefficient exploration and slow convergence. To address these limitations, we propose STAR-R1, a novel framework that integrates a single-stage RL paradigm with a fine-grained reward mechanism tailored for TVR. Specifically, STAR-R1 rewards partial correctness while penalizing excessive enumeration and passive inaction, enabling efficient exploration and precise reasoning. Comprehensive evaluations demonstrate that STAR-R1 achieves state-of-the-art performance across all 11 metrics, outperforming SFT by 23% in cross-view scenarios. Further analysis reveals STAR-R1's anthropomorphic behavior and highlights its unique ability to compare all objects for improving spatial reasoning. Our work provides critical insights in advancing the research of MLLMs and reasoning models. The codes, model weights, and data will be publicly available at https://github.com/zongzhao23/STAR-R1.
>
---
