# 计算机视觉 cs.CV

- **最新发布 280 篇**

- **更新 154 篇**

## 最新发布

#### [new 001] How Can Multimodal Remote Sensing Datasets Transform Classification via SpatialNet-ViT?
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于遥感分类任务，旨在解决现有模型泛化能力不足的问题。提出SpatialNet-ViT模型，结合ViTs和MTL提升分类精度与适应性。**

- **链接: [http://arxiv.org/pdf/2506.22501v1](http://arxiv.org/pdf/2506.22501v1)**

> **作者:** Gautam Siddharth Kashyap; Manaswi Kulahara; Nipun Joshi; Usman Naseem
>
> **备注:** Accepted in the 2025 IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2025), scheduled for 3 - 8 August 2025 in Brisbane, Australia
>
> **摘要:** Remote sensing datasets offer significant promise for tackling key classification tasks such as land-use categorization, object presence detection, and rural/urban classification. However, many existing studies tend to focus on narrow tasks or datasets, which limits their ability to generalize across various remote sensing classification challenges. To overcome this, we propose a novel model, SpatialNet-ViT, leveraging the power of Vision Transformers (ViTs) and Multi-Task Learning (MTL). This integrated approach combines spatial awareness with contextual understanding, improving both classification accuracy and scalability. Additionally, techniques like data augmentation, transfer learning, and multi-task learning are employed to enhance model robustness and its ability to generalize across diverse datasets
>
---
#### [new 002] Controllable Reference-Based Real-World Remote Sensing Image Super-Resolution with Generative Diffusion Priors
- **分类: cs.CV**

- **简介: 该论文属于遥感图像超分辨率任务，解决真实场景下参考图像依赖过强和生成不足的问题，提出CRefDiff模型及Real-RefRSSRD数据集。**

- **链接: [http://arxiv.org/pdf/2506.23801v1](http://arxiv.org/pdf/2506.23801v1)**

> **作者:** Ce Wang; Wanjie Sun
>
> **摘要:** Super-resolution (SR) techniques can enhance the spatial resolution of remote sensing images by utilizing low-resolution (LR) images to reconstruct high-resolution (HR) images, enabling more efficient large-scale earth observation applications. While single-image super-resolution (SISR) methods have shown progress, reference-based super-resolution (RefSR) offers superior performance by incorporating historical HR images alongside current LR observations. However, existing RefSR methods struggle with real-world complexities, such as cross-sensor resolution gap and significant land cover changes, often leading to under-generation or over-reliance on reference image. To address these challenges, we propose CRefDiff, a novel controllable reference-based diffusion model for real-world remote sensing image SR. To address the under-generation problem, CRefDiff is built upon the pretrained Stable Diffusion model, leveraging its powerful generative prior to produce accurate structures and textures. To mitigate over-reliance on the reference, we introduce a dual-branch fusion mechanism that adaptively integrates both local and global information from the reference image. Moreover, this novel dual-branch design enables reference strength control during inference, enhancing interactivity and flexibility of the model. Finally, a strategy named Better Start is proposed to significantly reduce the number of denoising steps, thereby accelerating the inference process. To support further research, we introduce Real-RefRSSRD, a new real-world RefSR dataset for remote sensing images, consisting of HR NAIP and LR Sentinel-2 image pairs with diverse land cover changes and significant temporal gaps. Extensive experiments on Real-RefRSSRD show that CRefDiff achieves state-of-the-art performance across various metrics and improves downstream tasks such as scene classification and semantic segmentation.
>
---
#### [new 003] Ovis-U1 Technical Report
- **分类: cs.CV; cs.AI**

- **简介: 该论文介绍Ovis-U1，一个30亿参数的多模态统一模型，解决文本到图像生成与图像编辑问题，通过联合训练提升性能。**

- **链接: [http://arxiv.org/pdf/2506.23044v1](http://arxiv.org/pdf/2506.23044v1)**

> **作者:** Guo-Hua Wang; Shanshan Zhao; Xinjie Zhang; Liangfu Cao; Pengxin Zhan; Lunhao Duan; Shiyin Lu; Minghao Fu; Xiaohao Chen; Jianshan Zhao; Yang Li; Qing-Guo Chen
>
> **备注:** A unified model for multimodal understanding, text-to-image generation, and image editing. GitHub: https://github.com/AIDC-AI/Ovis-U1
>
> **摘要:** In this report, we introduce Ovis-U1, a 3-billion-parameter unified model that integrates multimodal understanding, text-to-image generation, and image editing capabilities. Building on the foundation of the Ovis series, Ovis-U1 incorporates a diffusion-based visual decoder paired with a bidirectional token refiner, enabling image generation tasks comparable to leading models like GPT-4o. Unlike some previous models that use a frozen MLLM for generation tasks, Ovis-U1 utilizes a new unified training approach starting from a language model. Compared to training solely on understanding or generation tasks, unified training yields better performance, demonstrating the enhancement achieved by integrating these two tasks. Ovis-U1 achieves a score of 69.6 on the OpenCompass Multi-modal Academic Benchmark, surpassing recent state-of-the-art models such as Ristretto-3B and SAIL-VL-1.5-2B. In text-to-image generation, it excels with scores of 83.72 and 0.89 on the DPG-Bench and GenEval benchmarks, respectively. For image editing, it achieves 4.00 and 6.42 on the ImgEdit-Bench and GEdit-Bench-EN, respectively. As the initial version of the Ovis unified model series, Ovis-U1 pushes the boundaries of multimodal understanding, generation, and editing.
>
---
#### [new 004] Event-based Tiny Object Detection: A Benchmark Dataset and Baseline
- **分类: cs.CV**

- **简介: 该论文属于小目标检测任务，针对无人机反制中的小目标检测问题，提出EVSOD数据集和EV-SpSegNet方法，解决传统方法在复杂环境下的检测难题。**

- **链接: [http://arxiv.org/pdf/2506.23575v1](http://arxiv.org/pdf/2506.23575v1)**

> **作者:** Nuo Chen; Chao Xiao; Yimian Dai; Shiman He; Miao Li; Wei An
>
> **摘要:** Small object detection (SOD) in anti-UAV task is a challenging problem due to the small size of UAVs and complex backgrounds. Traditional frame-based cameras struggle to detect small objects in complex environments due to their low frame rates, limited dynamic range, and data redundancy. Event cameras, with microsecond temporal resolution and high dynamic range, provide a more effective solution for SOD. However, existing event-based object detection datasets are limited in scale, feature large targets size, and lack diverse backgrounds, making them unsuitable for SOD benchmarks. In this paper, we introduce a Event-based Small object detection (EVSOD) dataset (namely EV-UAV), the first large-scale, highly diverse benchmark for anti-UAV tasks. It includes 147 sequences with over 2.3 million event-level annotations, featuring extremely small targets (averaging 6.8 $\times$ 5.4 pixels) and diverse scenarios such as urban clutter and extreme lighting conditions. Furthermore, based on the observation that small moving targets form continuous curves in spatiotemporal event point clouds, we propose Event based Sparse Segmentation Network (EV-SpSegNet), a novel baseline for event segmentation in point cloud space, along with a Spatiotemporal Correlation (STC) loss that leverages motion continuity to guide the network in retaining target events. Extensive experiments on the EV-UAV dataset demonstrate the superiority of our method and provide a benchmark for future research in EVSOD. The dataset and code are at https://github.com/ChenYichen9527/Ev-UAV.
>
---
#### [new 005] Prompting without Panic: Attribute-aware, Zero-shot, Test-Time Calibration
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉-语言模型的校准任务，解决TPT导致置信度失准的问题，通过属性感知初始化和正则化损失提升校准效果。**

- **链接: [http://arxiv.org/pdf/2506.22819v1](http://arxiv.org/pdf/2506.22819v1)**

> **作者:** Ramya Hebbalaguppe; Tamoghno Kandar; Abhinav Nagpal; Chetan Arora
>
> **备注:** 26 pages
>
> **摘要:** Vision-language models (VLM) have demonstrated impressive performance in image recognition by leveraging self-supervised training on large datasets. Their performance can be further improved by adapting to the test sample using test-time prompt tuning (TPT). Unfortunately, the singular focus of TPT approaches on improving the accuracy suffers from tunnel vision, and leads to degradation in confidence calibration. This limits the applicability of TPT in critical applications. We make three contributions in this work. (1) We posit that random or naive initialization of prompts leads to overfitting on a particular test sample, and is the main reason for miscalibration of the VLM after TPT. To mitigate the problem, we propose careful initialization of test time prompt using prior knowledge about the target label attributes from a large language model (LLM); (2) To further maintain the quality of prompts during \tpt, we propose a novel regularization loss to reduce intraclass distance, and increase inter-class distance between the learnt Through extensive experiments on different CLIP architectures and 15 datasets, we show that our approach can effectively improve the calibration after TPT. We report an average expected calibration error (ECE) of 4.11 with our method, TCA, compared to 11.7 for vanilla TPT, 6.12 for C-TPT (ICLR'24), 6.78 for DiffTPT (CVPR'23), and 8.43 for PromptAlign (NeurIPS'23). The code is publicly accessible at: https://github.com/rhebbalaguppe/TCA_PromptWithoutPanic.
>
---
#### [new 006] Consistent Time-of-Flight Depth Denoising via Graph-Informed Geometric Attention
- **分类: cs.CV**

- **简介: 该论文属于深度图像去噪任务，解决ToF传感器深度图噪声问题，通过图结构融合提升时空一致性与清晰度。**

- **链接: [http://arxiv.org/pdf/2506.23542v1](http://arxiv.org/pdf/2506.23542v1)**

> **作者:** Weida Wang; Changyong He; Jin Zeng; Di Qiu
>
> **备注:** This paper has been accepted for publication at the International Conference on Computer Vision (ICCV) 2025
>
> **摘要:** Depth images captured by Time-of-Flight (ToF) sensors are prone to noise, requiring denoising for reliable downstream applications. Previous works either focus on single-frame processing, or perform multi-frame processing without considering depth variations at corresponding pixels across frames, leading to undesirable temporal inconsistency and spatial ambiguity. In this paper, we propose a novel ToF depth denoising network leveraging motion-invariant graph fusion to simultaneously enhance temporal stability and spatial sharpness. Specifically, despite depth shifts across frames, graph structures exhibit temporal self-similarity, enabling cross-frame geometric attention for graph fusion. Then, by incorporating an image smoothness prior on the fused graph and data fidelity term derived from ToF noise distribution, we formulate a maximum a posterior problem for ToF denoising. Finally, the solution is unrolled into iterative filters whose weights are adaptively learned from the graph-informed geometric attention, producing a high-performance yet interpretable network. Experimental results demonstrate that the proposed scheme achieves state-of-the-art performance in terms of accuracy and consistency on synthetic DVToF dataset and exhibits robust generalization on the real Kinectv2 dataset. Source code will be released at \href{https://github.com/davidweidawang/GIGA-ToF}{https://github.com/davidweidawang/GIGA-ToF}.
>
---
#### [new 007] Towards an Automated Multimodal Approach for Video Summarization: Building a Bridge Between Text, Audio and Facial Cue-Based Summarization
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视频摘要任务，旨在解决传统单模态方法的不足，通过融合文本、音频和视觉信息，提升摘要的语义和情感准确性。**

- **链接: [http://arxiv.org/pdf/2506.23714v1](http://arxiv.org/pdf/2506.23714v1)**

> **作者:** Md Moinul Islam; Sofoklis Kakouros; Janne Heikkilä; Mourad Oussalah
>
> **备注:** Accepted to HHAI WS 2025: Workshops at the Fourth International Conference on Hybrid Human-Artificial Intelligence (HHAI)
>
> **摘要:** The increasing volume of video content in educational, professional, and social domains necessitates effective summarization techniques that go beyond traditional unimodal approaches. This paper proposes a behaviour-aware multimodal video summarization framework that integrates textual, audio, and visual cues to generate timestamp-aligned summaries. By extracting prosodic features, textual cues and visual indicators, the framework identifies semantically and emotionally important moments. A key contribution is the identification of bonus words, which are terms emphasized across multiple modalities and used to improve the semantic relevance and expressive clarity of the summaries. The approach is evaluated against pseudo-ground truth (pGT) summaries generated using LLM-based extractive method. Experimental results demonstrate significant improvements over traditional extractive method, such as the Edmundson method, in both text and video-based evaluation metrics. Text-based metrics show ROUGE-1 increasing from 0.4769 to 0.7929 and BERTScore from 0.9152 to 0.9536, while in video-based evaluation, our proposed framework improves F1-Score by almost 23%. The findings underscore the potential of multimodal integration in producing comprehensive and behaviourally informed video summaries.
>
---
#### [new 008] Automated Defect Identification and Categorization in NDE 4.0 with the Application of Artificial Intelligence
- **分类: cs.CV**

- **简介: 该论文属于工业检测任务，旨在解决传统无损检测效率低的问题。通过AI技术实现缺陷自动识别与分类，提升检测准确性和速度。**

- **链接: [http://arxiv.org/pdf/2506.22513v1](http://arxiv.org/pdf/2506.22513v1)**

> **作者:** Aditya Sharma
>
> **摘要:** This investigation attempts to create an automated framework for fault detection and organization for usage in contemporary radiography, as per NDE 4.0. The review's goals are to address the lack of information that is sufficiently explained, learn how to make the most of virtual defect increase, and determine whether the framework is viable by using NDE measurements. As its basic information source, the technique consists of compiling and categorizing 223 CR photographs of airplane welds. Information expansion systems, such as virtual defect increase and standard increase, are used to work on the preparation dataset. A modified U-net model is prepared using the improved data to produce semantic fault division veils. To assess the effectiveness of the model, NDE boundaries such as Case, estimating exactness, and misleading call rate are used. Tiny a90/95 characteristics, which provide strong differentiating evidence of flaws, reveal that the suggested approach achieves exceptional awareness in defect detection. Considering a 90/95, size error, and fake call rate in the weld area, the consolidated expansion approach clearly wins. Due to the framework's fast derivation speed, large images can be broken down efficiently and quickly. Professional controllers evaluate the transmitted system in the field and believe that it has a guarantee as a support device in the testing cycle, irrespective of particular equipment cut-off points and programming resemblance.
>
---
#### [new 009] Hierarchical Corpus-View-Category Refinement for Carotid Plaque Risk Grading in Ultrasound
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决颈动脉斑块风险分级中的小样本和类内差异问题。提出CVC-RF框架，通过多层级优化提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.23108v1](http://arxiv.org/pdf/2506.23108v1)**

> **作者:** Zhiyuan Zhu; Jian Wang; Yong Jiang; Tong Han; Yuhao Huang; Ang Zhang; Kaiwen Yang; Mingyuan Luo; Zhe Liu; Yaofei Duan; Dong Ni; Tianhong Tang; Xin Yang
>
> **备注:** Accepted at MICCAI 2025
>
> **摘要:** Accurate carotid plaque grading (CPG) is vital to assess the risk of cardiovascular and cerebrovascular diseases. Due to the small size and high intra-class variability of plaque, CPG is commonly evaluated using a combination of transverse and longitudinal ultrasound views in clinical practice. However, most existing deep learning-based multi-view classification methods focus on feature fusion across different views, neglecting the importance of representation learning and the difference in class features. To address these issues, we propose a novel Corpus-View-Category Refinement Framework (CVC-RF) that processes information from Corpus-, View-, and Category-levels, enhancing model performance. Our contribution is four-fold. First, to the best of our knowledge, we are the foremost deep learning-based method for CPG according to the latest Carotid Plaque-RADS guidelines. Second, we propose a novel center-memory contrastive loss, which enhances the network's global modeling capability by comparing with representative cluster centers and diverse negative samples at the Corpus level. Third, we design a cascaded down-sampling attention module to fuse multi-scale information and achieve implicit feature interaction at the View level. Finally, a parameter-free mixture-of-experts weighting strategy is introduced to leverage class clustering knowledge to weight different experts, enabling feature decoupling at the Category level. Experimental results indicate that CVC-RF effectively models global features via multi-level refinement, achieving state-of-the-art performance in the challenging CPG task.
>
---
#### [new 010] Imagine for Me: Creative Conceptual Blending of Real Images and Text via Blended Attention
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于跨模态概念融合任务，旨在解决人类设计中的认知偏差问题。提出IT-Blender模型，结合扩散模型与混合注意力机制，实现图像与文本的高质量融合。**

- **链接: [http://arxiv.org/pdf/2506.24085v1](http://arxiv.org/pdf/2506.24085v1)**

> **作者:** Wonwoong Cho; Yanxia Zhang; Yan-Ying Chen; David I. Inouye
>
> **备注:** Project website is available at https://imagineforme.github.io/
>
> **摘要:** Blending visual and textual concepts into a new visual concept is a unique and powerful trait of human beings that can fuel creativity. However, in practice, cross-modal conceptual blending for humans is prone to cognitive biases, like design fixation, which leads to local minima in the design space. In this paper, we propose a T2I diffusion adapter "IT-Blender" that can automate the blending process to enhance human creativity. Prior works related to cross-modal conceptual blending are limited in encoding a real image without loss of details or in disentangling the image and text inputs. To address these gaps, IT-Blender leverages pretrained diffusion models (SD and FLUX) to blend the latent representations of a clean reference image with those of the noisy generated image. Combined with our novel blended attention, IT-Blender encodes the real reference image without loss of details and blends the visual concept with the object specified by the text in a disentangled way. Our experiment results show that IT-Blender outperforms the baselines by a large margin in blending visual and textual concepts, shedding light on the new application of image generative models to augment human creativity.
>
---
#### [new 011] PixelBoost: Leveraging Brownian Motion for Realistic-Image Super-Resolution
- **分类: cs.CV; cs.AI; cs.MM; eess.IV**

- **简介: 该论文属于图像超分辨率任务，旨在解决生成真实感图像与计算效率的矛盾。通过引入基于布朗运动的PixelBoost模型，提升图像纹理和边缘质量，同时加快推理速度。**

- **链接: [http://arxiv.org/pdf/2506.23254v1](http://arxiv.org/pdf/2506.23254v1)**

> **作者:** Aradhana Mishra; Bumshik Lee
>
> **摘要:** Diffusion-model-based image super-resolution techniques often face a trade-off between realistic image generation and computational efficiency. This issue is exacerbated when inference times by decreasing sampling steps, resulting in less realistic and hazy images. To overcome this challenge, we introduce a novel diffusion model named PixelBoost that underscores the significance of embracing the stochastic nature of Brownian motion in advancing image super-resolution, resulting in a high degree of realism, particularly focusing on texture and edge definitions. By integrating controlled stochasticity into the training regimen, our proposed model avoids convergence to local optima, effectively capturing and reproducing the inherent uncertainty of image textures and patterns. Our proposed model demonstrates superior objective results in terms of learned perceptual image patch similarity (LPIPS), lightness order error (LOE), peak signal-to-noise ratio(PSNR), structural similarity index measure (SSIM), as well as visual quality. To determine the edge enhancement, we evaluated the gradient magnitude and pixel value, and our proposed model exhibited a better edge reconstruction capability. Additionally, our model demonstrates adaptive learning capabilities by effectively adjusting to Brownian noise patterns and introduces a sigmoidal noise sequencing method that simplifies training, resulting in faster inference speeds.
>
---
#### [new 012] Robust Perspective Correction for Real-World Crack Evolution Tracking in Image-Based Structural Health Monitoring
- **分类: cs.CV; 68T45 (Computer Vision)**

- **简介: 该论文属于结构健康监测中的图像对齐任务，解决真实场景下裂缝跟踪的几何校正问题。通过物理启发的框架，提升裂缝定位精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.22437v1](http://arxiv.org/pdf/2506.22437v1)**

> **作者:** Xinxin Sun; Peter Chang
>
> **备注:** 43 pages, 5 figures, 19 tables. Submitted to NDT&E International. This work may also be of interest to researchers in optical NDE and civil engineering SHM
>
> **摘要:** Accurate image alignment is essential for monitoring crack evolution in structural health monitoring (SHM), particularly under real-world conditions involving perspective distortion, occlusion, and low contrast. However, traditional feature detectors such as SIFT and SURF, which rely on Gaussian-based scale spaces, tend to suppress high-frequency edges, making them unsuitable for thin crack localization. Lightweight binary alternatives like ORB and BRISK, while computationally efficient, often suffer from poor keypoint repeatability on textured or shadowed surfaces. This study presents a physics-informed alignment framework that adapts the open KAZE architecture to SHM-specific challenges. By utilizing nonlinear anisotropic diffusion to construct a crack-preserving scale space, and integrating RANSAC-based homography estimation, the framework enables accurate geometric correction without the need for training, parameter tuning, or prior calibration. The method is validated on time-lapse images of masonry and concrete acquired via handheld smartphone under varied field conditions, including shadow interference, cropping, oblique viewing angles, and surface clutter. Compared to classical detectors, the proposed framework reduces crack area and spine length errors by up to 70 percent and 90 percent, respectively, while maintaining sub-5 percent alignment error in key metrics. Unsupervised, interpretable, and computationally lightweight, this approach supports scalable deployment via UAVs and mobile platforms. By tailoring nonlinear scale-space modeling to SHM image alignment, this work offers a robust and physically grounded alternative to conventional techniques for tracking real-world crack evolution.
>
---
#### [new 013] TurboVSR: Fantastic Video Upscalers and Where to Find Them
- **分类: cs.CV**

- **简介: 该论文属于视频超分辨率任务，解决计算效率低的问题。通过设计高效模型，实现快速高质量视频和图像超分辨率。**

- **链接: [http://arxiv.org/pdf/2506.23618v1](http://arxiv.org/pdf/2506.23618v1)**

> **作者:** Zhongdao Wang; Guodongfang Zhao; Jingjing Ren; Bailan Feng; Shifeng Zhang; Wenbo Li
>
> **备注:** ICCV, 2025
>
> **摘要:** Diffusion-based generative models have demonstrated exceptional promise in the video super-resolution (VSR) task, achieving a substantial advancement in detail generation relative to prior methods. However, these approaches face significant computational efficiency challenges. For instance, current techniques may require tens of minutes to super-resolve a mere 2-second, 1080p video. In this paper, we present TurboVSR, an ultra-efficient diffusion-based video super-resolution model. Our core design comprises three key aspects: (1) We employ an autoencoder with a high compression ratio of 32$\times$32$\times$8 to reduce the number of tokens. (2) Highly compressed latents pose substantial challenges for training. We introduce factorized conditioning to mitigate the learning complexity: we first learn to super-resolve the initial frame; subsequently, we condition the super-resolution of the remaining frames on the high-resolution initial frame and the low-resolution subsequent frames. (3) We convert the pre-trained diffusion model to a shortcut model to enable fewer sampling steps, further accelerating inference. As a result, TurboVSR performs on par with state-of-the-art VSR methods, while being 100+ times faster, taking only 7 seconds to process a 2-second long 1080p video. TurboVSR also supports image resolution by considering image as a one-frame video. Our efficient design makes SR beyond 1080p possible, results on 4K (3648$\times$2048) image SR show surprising fine details.
>
---
#### [new 014] Utilizing a Novel Deep Learning Method for Scene Categorization in Remote Sensing Data
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于遥感图像场景分类任务，旨在解决传统方法在高噪声数据中准确率低的问题，提出CO-BRNN模型并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2506.22939v1](http://arxiv.org/pdf/2506.22939v1)**

> **作者:** Ghufran A. Omran; Wassan Saad Abduljabbar Hayale; Ahmad AbdulQadir AlRababah; Israa Ibraheem Al-Barazanchi; Ravi Sekhar; Pritesh Shah; Sushma Parihar; Harshavardhan Reddy Penubadi
>
> **摘要:** Scene categorization (SC) in remotely acquired images is an important subject with broad consequences in different fields, including catastrophe control, ecological observation, architecture for cities, and more. Nevertheless, its several apps, reaching a high degree of accuracy in SC from distant observation data has demonstrated to be difficult. This is because traditional conventional deep learning models require large databases with high variety and high levels of noise to capture important visual features. To address these problems, this investigation file introduces an innovative technique referred to as the Cuttlefish Optimized Bidirectional Recurrent Neural Network (CO- BRNN) for type of scenes in remote sensing data. The investigation compares the execution of CO-BRNN with current techniques, including Multilayer Perceptron- Convolutional Neural Network (MLP-CNN), Convolutional Neural Network-Long Short Term Memory (CNN-LSTM), and Long Short Term Memory-Conditional Random Field (LSTM-CRF), Graph-Based (GB), Multilabel Image Retrieval Model (MIRM-CF), Convolutional Neural Networks Data Augmentation (CNN-DA). The results demonstrate that CO-BRNN attained the maximum accuracy of 97%, followed by LSTM-CRF with 90%, MLP-CNN with 85%, and CNN-LSTM with 80%. The study highlights the significance of physical confirmation to ensure the efficiency of satellite data.
>
---
#### [new 015] Mamba-FETrack V2: Revisiting State Space Model for Frame-Event based Visual Object Tracking
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉目标跟踪任务，旨在解决多模态跟踪中计算复杂度高和跨模交互不足的问题。提出Mamba-FETrack V2框架，利用线性复杂度的Vision Mamba网络实现高效特征提取与融合。**

- **链接: [http://arxiv.org/pdf/2506.23783v1](http://arxiv.org/pdf/2506.23783v1)**

> **作者:** Shiao Wang; Ju Huang; Qingchuan Ma; Jinfeng Gao; Chunyi Xu; Xiao Wang; Lan Chen; Bo Jiang
>
> **备注:** Journal extension of Mamba-FETrack which was published on Pattern Recognition and Computer Vision (PRCV) 2024
>
> **摘要:** Combining traditional RGB cameras with bio-inspired event cameras for robust object tracking has garnered increasing attention in recent years. However, most existing multimodal tracking algorithms depend heavily on high-complexity Vision Transformer architectures for feature extraction and fusion across modalities. This not only leads to substantial computational overhead but also limits the effectiveness of cross-modal interactions. In this paper, we propose an efficient RGB-Event object tracking framework based on the linear-complexity Vision Mamba network, termed Mamba-FETrack V2. Specifically, we first design a lightweight Prompt Generator that utilizes embedded features from each modality, together with a shared prompt pool, to dynamically generate modality-specific learnable prompt vectors. These prompts, along with the modality-specific embedded features, are then fed into a Vision Mamba-based FEMamba backbone, which facilitates prompt-guided feature extraction, cross-modal interaction, and fusion in a unified manner. Finally, the fused representations are passed to the tracking head for accurate target localization. Extensive experimental evaluations on multiple RGB-Event tracking benchmarks, including short-term COESOT dataset and long-term datasets, i.e., FE108 and FELT V2, demonstrate the superior performance and efficiency of the proposed tracking framework. The source code and pre-trained models will be released on https://github.com/Event-AHU/Mamba_FETrack
>
---
#### [new 016] How Semantically Informative is an Image?: Measuring the Covariance-Weighted Norm of Contrastive Learning Embeddings
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言任务，旨在衡量图像和文本的语义信息量。通过对比学习模型计算信息增益，解决语义 informativeness 的量化问题。**

- **链接: [http://arxiv.org/pdf/2506.22881v1](http://arxiv.org/pdf/2506.22881v1)**

> **作者:** Fumiya Uchiyama; Rintaro Yanagi; Shohei Taniguchi; Shota Takashiro; Masahiro Suzuki; Hirokatsu Kataoka; Yusuke Iwasawa; Yutaka Matsuo
>
> **摘要:** Contrastive learning has the capacity to model multimodal probability distributions by embedding and aligning visual representations with semantics from captions. This approach enables the estimation of relational semantic similarity; however, it remains unclear whether it can also represent absolute semantic informativeness. In this work, we introduce a semantic informativeness metric for an image calculated from text samples via a contrastive learning model; similarly, the informativeness of a text is calculated from image samples. We propose a redefinition of the concept of Information Gain, a concept previously explored in natural language processing, extending its application to the domains of vision and language. Our metric quantifies how conditioning on an image distorts the distribution of associated texts, and vice versa for text conditioning on image distributions. In OpenCLIP's empirical results, we observe that images with the lowest Information Gain scores often correspond to placeholder icons such as "image not found." Furthermore, we propose to measure a norm-based metric of the embedding to estimate the Information Gain, following the theoretical results for Skip-Gram with Negative Sampling (SGNS) word embedding. Information Gain can be measured using either CLIP or SigLIP, and the results demonstrate a strong correlation with a coefficient of determination ranging from 0.98 to 1.00. After obtaining the mean and the covariance of the sample embedding, the computational cost of this method is independent of the sample size, and it is compatible with publicly available, open-weight models.
>
---
#### [new 017] Dynamic Contrastive Learning for Hierarchical Retrieval: A Case Study of Distance-Aware Cross-View Geo-Localization
- **分类: cs.CV**

- **简介: 该论文属于跨视角地理定位任务，旨在提升模型对目标周围环境的全面理解及降低定位误差。研究构建了DA-Campus基准，并提出DyCL框架解决空间关系复杂性问题。**

- **链接: [http://arxiv.org/pdf/2506.23077v1](http://arxiv.org/pdf/2506.23077v1)**

> **作者:** Suofei Zhang; Xinxin Wang; Xiaofu Wu; Quan Zhou; Haifeng Hu
>
> **摘要:** Existing deep learning-based cross-view geo-localization methods primarily focus on improving the accuracy of cross-domain image matching, rather than enabling models to comprehensively capture contextual information around the target and minimize the cost of localization errors. To support systematic research into this Distance-Aware Cross-View Geo-Localization (DACVGL) problem, we construct Distance-Aware Campus (DA-Campus), the first benchmark that pairs multi-view imagery with precise distance annotations across three spatial resolutions. Based on DA-Campus, we formulate DACVGL as a hierarchical retrieval problem across different domains. Our study further reveals that, due to the inherent complexity of spatial relationships among buildings, this problem can only be addressed via a contrastive learning paradigm, rather than conventional metric learning. To tackle this challenge, we propose Dynamic Contrastive Learning (DyCL), a novel framework that progressively aligns feature representations according to hierarchical spatial margins. Extensive experiments demonstrate that DyCL is highly complementary to existing multi-scale metric learning methods and yields substantial improvements in both hierarchical retrieval performance and overall cross-view geo-localization accuracy. Our code and benchmark are publicly available at https://github.com/anocodetest1/DyCL.
>
---
#### [new 018] FOCUS: Fine-grained Optimization with Semantic Guided Understanding for Pedestrian Attributes Recognition
- **分类: cs.CV**

- **简介: 该论文属于行人属性识别任务，解决现有方法在细粒度特征提取和未见属性泛化上的不足。提出FOCUS方法，通过多粒度混合令牌和属性引导的视觉特征提取提升性能。**

- **链接: [http://arxiv.org/pdf/2506.22836v1](http://arxiv.org/pdf/2506.22836v1)**

> **作者:** Hongyan An; Kuan Zhu; Xin He; Haiyun Guo; Chaoyang Zhao; Ming Tang; Jinqiao Wang
>
> **备注:** ICME 2025 Oral
>
> **摘要:** Pedestrian attribute recognition (PAR) is a fundamental perception task in intelligent transportation and security. To tackle this fine-grained task, most existing methods focus on extracting regional features to enrich attribute information. However, a regional feature is typically used to predict a fixed set of pre-defined attributes in these methods, which limits the performance and practicality in two aspects: 1) Regional features may compromise fine-grained patterns unique to certain attributes in favor of capturing common characteristics shared across attributes. 2) Regional features cannot generalize to predict unseen attributes in the test time. In this paper, we propose the \textbf{F}ine-grained \textbf{O}ptimization with semanti\textbf{C} g\textbf{U}ided under\textbf{S}tanding (FOCUS) approach for PAR, which adaptively extracts fine-grained attribute-level features for each attribute individually, regardless of whether the attributes are seen or not during training. Specifically, we propose the Multi-Granularity Mix Tokens (MGMT) to capture latent features at varying levels of visual granularity, thereby enriching the diversity of the extracted information. Next, we introduce the Attribute-guided Visual Feature Extraction (AVFE) module, which leverages textual attributes as queries to retrieve their corresponding visual attribute features from the Mix Tokens using a cross-attention mechanism. To ensure that textual attributes focus on the appropriate Mix Tokens, we further incorporate a Region-Aware Contrastive Learning (RACL) method, encouraging attributes within the same region to share consistent attention maps. Extensive experiments on PA100K, PETA, and RAPv1 datasets demonstrate the effectiveness and strong generalization ability of our method.
>
---
#### [new 019] AG-VPReID 2025: Aerial-Ground Video-based Person Re-identification Challenge Results
- **分类: cs.CV**

- **简介: 该论文属于跨视角行人重识别任务，解决航空与地面视角间的识别难题。构建了大规模数据集并组织竞赛，探索多模态与时序建模方法。**

- **链接: [http://arxiv.org/pdf/2506.22843v1](http://arxiv.org/pdf/2506.22843v1)**

> **作者:** Kien Nguyen; Clinton Fookes; Sridha Sridharan; Huy Nguyen; Feng Liu; Xiaoming Liu; Arun Ross; Dana Michalski; Tamás Endrei; Ivan DeAndres-Tame; Ruben Tolosana; Ruben Vera-Rodriguez; Aythami Morales; Julian Fierrez; Javier Ortega-Garcia; Zijing Gong; Yuhao Wang; Xuehu Liu; Pingping Zhang; Md Rashidunnabi; Hugo Proença; Kailash A. Hambarde; Saeid Rezaei
>
> **摘要:** Person re-identification (ReID) across aerial and ground vantage points has become crucial for large-scale surveillance and public safety applications. Although significant progress has been made in ground-only scenarios, bridging the aerial-ground domain gap remains a formidable challenge due to extreme viewpoint differences, scale variations, and occlusions. Building upon the achievements of the AG-ReID 2023 Challenge, this paper introduces the AG-VPReID 2025 Challenge - the first large-scale video-based competition focused on high-altitude (80-120m) aerial-ground ReID. Constructed on the new AG-VPReID dataset with 3,027 identities, over 13,500 tracklets, and approximately 3.7 million frames captured from UAVs, CCTV, and wearable cameras, the challenge featured four international teams. These teams developed solutions ranging from multi-stream architectures to transformer-based temporal reasoning and physics-informed modeling. The leading approach, X-TFCLIP from UAM, attained 72.28% Rank-1 accuracy in the aerial-to-ground ReID setting and 70.77% in the ground-to-aerial ReID setting, surpassing existing baselines while highlighting the dataset's complexity. For additional details, please refer to the official website at https://agvpreid25.github.io.
>
---
#### [new 020] CycleVAR: Repurposing Autoregressive Model for Unsupervised One-Step Image Translation
- **分类: cs.CV**

- **简介: 该论文属于无监督图像翻译任务，旨在解决传统方法中离散量化导致的梯度中断问题。提出CycleVAR模型，通过连续概率混合实现端到端优化，并采用多尺度生成策略提升翻译质量与速度。**

- **链接: [http://arxiv.org/pdf/2506.23347v1](http://arxiv.org/pdf/2506.23347v1)**

> **作者:** Yi Liu; Shengqian Li; Zuzeng Lin; Feng Wang; Si Liu
>
> **摘要:** The current conditional autoregressive image generation methods have shown promising results, yet their potential remains largely unexplored in the practical unsupervised image translation domain, which operates without explicit cross-domain correspondences. A critical limitation stems from the discrete quantization inherent in traditional Vector Quantization-based frameworks, which disrupts gradient flow between the Variational Autoencoder decoder and causal Transformer, impeding end-to-end optimization during adversarial training in image space. To tackle this issue, we propose using Softmax Relaxed Quantization, a novel approach that reformulates codebook selection as a continuous probability mixing process via Softmax, thereby preserving gradient propagation. Building upon this differentiable foundation, we introduce CycleVAR, which reformulates image-to-image translation as image-conditional visual autoregressive generation by injecting multi-scale source image tokens as contextual prompts, analogous to prefix-based conditioning in language models. CycleVAR exploits two modes to generate the target image tokens, including (1) serial multi-step generation, enabling iterative refinement across scales, and (2) parallel one-step generation synthesizing all resolution outputs in a single forward pass. Experimental findings indicate that the parallel one-step generation mode attains superior translation quality with quicker inference speed than the serial multi-step mode in unsupervised scenarios. Furthermore, both quantitative and qualitative results indicate that CycleVAR surpasses previous state-of-the-art unsupervised image translation models, \textit{e}.\textit{g}., CycleGAN-Turbo.
>
---
#### [new 021] Towards Initialization-free Calibrated Bundle Adjustment
- **分类: cs.CV**

- **简介: 该论文属于三维重建任务，旨在解决初始化自由的校准束平差问题。通过引入相对旋转估计，利用已知相机标定，实现接近度量的重建。**

- **链接: [http://arxiv.org/pdf/2506.23808v1](http://arxiv.org/pdf/2506.23808v1)**

> **作者:** Carl Olsson; Amanda Nilsson
>
> **摘要:** A recent series of works has shown that initialization-free BA can be achieved using pseudo Object Space Error (pOSE) as a surrogate objective. The initial reconstruction-step optimizes an objective where all terms are projectively invariant and it cannot incorporate knowledge of the camera calibration. As a result, the solution is only determined up to a projective transformation of the scene and the process requires more data for successful reconstruction. In contrast, we present a method that is able to use the known camera calibration thereby producing near metric solutions, that is, reconstructions that are accurate up to a similarity transformation. To achieve this we introduce pairwise relative rotation estimates that carry information about camera calibration. These are only invariant to similarity transformations, thus encouraging solutions that preserve metric features of the real scene. Our method can be seen as integrating rotation averaging into the pOSE framework striving towards initialization-free calibrated SfM. Our experimental evaluation shows that we are able to reliably optimize our objective, achieving convergence to the global minimum with high probability from random starting solutions, resulting in accurate near metric reconstructions.
>
---
#### [new 022] Improve Underwater Object Detection through YOLOv12 Architecture and Physics-informed Augmentation
- **分类: cs.CV**

- **简介: 该论文属于 underwater object detection 任务，解决低能见度下检测精度不足的问题。通过 YOLOv12 和物理增强技术提升检测性能。**

- **链接: [http://arxiv.org/pdf/2506.23505v1](http://arxiv.org/pdf/2506.23505v1)**

> **作者:** Tinh Nguyen
>
> **摘要:** Underwater object detection is crucial for autonomous navigation, environmental monitoring, and marine exploration, but it is severely hampered by light attenuation, turbidity, and occlusion. Current methods balance accuracy and computational efficiency, but they have trouble deploying in real-time under low visibility conditions. Through the integration of physics-informed augmentation techniques with the YOLOv12 architecture, this study advances underwater detection. With Residual ELAN blocks to preserve structural features in turbid waters and Area Attention to maintain large receptive fields for occluded objects while reducing computational complexity. Underwater optical properties are addressed by domain-specific augmentations such as turbulence adaptive blurring, biologically grounded occlusion simulation, and spectral HSV transformations for color distortion. Extensive tests on four difficult datasets show state-of-the-art performance, with Brackish data registering 98.30% mAP at 142 FPS. YOLOv12 improves occlusion robustness by 18.9%, small-object recall by 22.4%, and detection precision by up to 7.94% compared to previous models. The crucial role of augmentation strategy is validated by ablation studies. This work offers a precise and effective solution for conservation and underwater robotics applications.
>
---
#### [new 023] Towards Explainable Bilingual Multimodal Misinformation Detection and Localization
- **分类: cs.CV**

- **简介: 该论文属于多模态谣言检测任务，旨在解决跨语言、跨模态的虚假信息识别与解释问题。提出BiMi框架，实现区域定位、一致性检测及自然语言解释，并构建了大规模基准数据集。**

- **链接: [http://arxiv.org/pdf/2506.22930v1](http://arxiv.org/pdf/2506.22930v1)**

> **作者:** Yiwei He; Xiangtai Li; Zhenglin Huang; Yi Dong; Hao Fei; Jiangning Zhang; Baoyuan Wu; Guangliang Cheng
>
> **摘要:** The increasing realism of multimodal content has made misinformation more subtle and harder to detect, especially in news media where images are frequently paired with bilingual (e.g., Chinese-English) subtitles. Such content often includes localized image edits and cross-lingual inconsistencies that jointly distort meaning while remaining superficially plausible. We introduce BiMi, a bilingual multimodal framework that jointly performs region-level localization, cross-modal and cross-lingual consistency detection, and natural language explanation for misinformation analysis. To support generalization, BiMi integrates an online retrieval module that supplements model reasoning with up-to-date external context. We further release BiMiBench, a large-scale and comprehensive benchmark constructed by systematically editing real news images and subtitles, comprising 104,000 samples with realistic manipulations across visual and linguistic modalities. To enhance interpretability, we apply Group Relative Policy Optimization (GRPO) to improve explanation quality, marking the first use of GRPO in this domain. Extensive experiments demonstrate that BiMi outperforms strong baselines by up to +8.9 in classification accuracy, +15.9 in localization accuracy, and +2.5 in explanation BERTScore, advancing state-of-the-art performance in realistic, multilingual misinformation detection. Code, models, and datasets will be released.
>
---
#### [new 024] MoCa: Modality-aware Continual Pre-training Makes Better Bidirectional Multimodal Embeddings
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态嵌入任务，旨在解决现有模型在注意力机制、数据依赖性和训练多样性上的不足。提出MoCa框架，通过双向预训练和异构对比微调提升性能。**

- **链接: [http://arxiv.org/pdf/2506.23115v1](http://arxiv.org/pdf/2506.23115v1)**

> **作者:** Haonan Chen; Hong Liu; Yuping Luo; Liang Wang; Nan Yang; Furu Wei; Zhicheng Dou
>
> **备注:** Homepage: https://haon-chen.github.io/MoCa/
>
> **摘要:** Multimodal embedding models, built upon causal Vision Language Models (VLMs), have shown promise in various tasks. However, current approaches face three key limitations: the use of causal attention in VLM backbones is suboptimal for embedding tasks; scalability issues due to reliance on high-quality labeled paired data for contrastive learning; and limited diversity in training objectives and data. To address these issues, we propose MoCa, a two-stage framework for transforming pre-trained VLMs into effective bidirectional multimodal embedding models. The first stage, Modality-aware Continual Pre-training, introduces a joint reconstruction objective that simultaneously denoises interleaved text and image inputs, enhancing bidirectional context-aware reasoning. The second stage, Heterogeneous Contrastive Fine-tuning, leverages diverse, semantically rich multimodal data beyond simple image-caption pairs to enhance generalization and alignment. Our method addresses the stated limitations by introducing bidirectional attention through continual pre-training, scaling effectively with massive unlabeled datasets via joint reconstruction objectives, and utilizing diverse multimodal data for enhanced representation robustness. Experiments demonstrate that MoCa consistently improves performance across MMEB and ViDoRe-v2 benchmarks, achieving new state-of-the-art results, and exhibits strong scalability with both model size and training data on MMEB.
>
---
#### [new 025] Unleashing the Multi-View Fusion Potential: Noise Correction in VLM for Open-Vocabulary 3D Scene Understanding
- **分类: cs.CV**

- **简介: 该论文属于开放词汇3D场景理解任务，旨在解决3D数据不足导致的模型泛化能力差问题。通过减少视觉语言模型中的噪声，提升多视角融合效果。**

- **链接: [http://arxiv.org/pdf/2506.22817v1](http://arxiv.org/pdf/2506.22817v1)**

> **作者:** Xingyilang Yin; Jiale Wang; Xi Yang; Mutian Xu; Xu Gu; Nannan Wang
>
> **摘要:** Recent open-vocabulary 3D scene understanding approaches mainly focus on training 3D networks through contrastive learning with point-text pairs or by distilling 2D features into 3D models via point-pixel alignment. While these methods show considerable performance in benchmarks with limited vocabularies, they struggle to handle diverse object categories as the limited amount of 3D data upbound training strong open-vocabulary 3d models. We observe that 2D multi-view fusion methods take precedence in understanding diverse concepts in 3D scenes. However, inherent noises in vision-language models lead multi-view fusion to sub-optimal performance. To this end, we introduce MVOV3D, a novel approach aimed at unleashing the potential of 2D multi-view fusion for open-vocabulary 3D scene understanding. We focus on reducing the inherent noises without training, thereby preserving the generalizability while enhancing open-world capabilities. Specifically, MVOV3D improves multi-view 2D features by leveraging precise region-level image features and text features encoded by CLIP encoders and incorporates 3D geometric priors to optimize multi-view fusion. Extensive experiments on various datasets demonstrate the effectiveness of our method. Notably, our MVOV3D achieves a new record with 14.7% mIoU on ScanNet200 and 16.2% mIoU on Matterport160 for challenge open-vocabulary semantic segmentation, outperforming current leading trained 3D networks by a significant margin.
>
---
#### [new 026] Visual and Memory Dual Adapter for Multi-Modal Object Tracking
- **分类: cs.CV**

- **简介: 该论文属于多模态目标跟踪任务，旨在解决现有方法在跨频域和时域学习可靠提示的问题。提出视觉与记忆双适配器，提升跟踪性能。**

- **链接: [http://arxiv.org/pdf/2506.23972v1](http://arxiv.org/pdf/2506.23972v1)**

> **作者:** Boyue Xu; Ruichao Hou; Tongwei Ren; Gangshan Wu
>
> **摘要:** Prompt-learning-based multi-modal trackers have achieved promising progress by employing lightweight visual adapters to incorporate auxiliary modality features into frozen foundation models. However, existing approaches often struggle to learn reliable prompts due to limited exploitation of critical cues across frequency and temporal domains. In this paper, we propose a novel visual and memory dual adapter (VMDA) to construct more robust and discriminative representations for multi-modal tracking. Specifically, we develop a simple but effective visual adapter that adaptively transfers discriminative cues from auxiliary modality to dominant modality by jointly modeling the frequency, spatial, and channel-wise features. Additionally, we design the memory adapter inspired by the human memory mechanism, which stores global temporal cues and performs dynamic update and retrieval operations to ensure the consistent propagation of reliable temporal information across video sequences. Extensive experiments demonstrate that our method achieves state-of-the-art performance on the various multi-modal tracking tasks, including RGB-Thermal, RGB-Depth, and RGB-Event tracking. Code and models are available at https://github.com/xuboyue1999/mmtrack.git.
>
---
#### [new 027] FastSeg: Efficient Training-Free Open-Vocabulary Segmentation via Hierarchical Attention Refinement Method
- **分类: cs.CV**

- **简介: 该论文属于开放词汇语义分割任务，旨在解决零样本分割中空间精度不足的问题。提出FastSeg框架，通过注意力机制提升分割质量与效率。**

- **链接: [http://arxiv.org/pdf/2506.23323v1](http://arxiv.org/pdf/2506.23323v1)**

> **作者:** Quang-Huy Che; Vinh-Tiep Nguyen
>
> **摘要:** Open-vocabulary semantic segmentation (OVSS) aims to segment objects from arbitrary text categories without requiring densely annotated datasets. Although contrastive learning based models enable zero-shot segmentation, they often lose fine spatial precision at pixel level, due to global representation bias. In contrast, diffusion-based models naturally encode fine-grained spatial features via attention mechanisms that capture both global context and local details. However, they often face challenges in balancing the number of iterations with the quality of the segmentation. In this work, we propose FastSeg, a novel and efficient training-free framework with only (1+1)-step of reverse process of a pretrained diffusion model (e.g., Stable Diffusion). Moreover, instead of running multiple times for different classes, FastSeg performs segmentation for all classes at once. To further enhance the segmentation quality, FastSeg introduces three key components: (i) a dual-prompt mechanism for discriminative, class-aware attention extraction, (ii) a Hierarchical Attention Refinement Method (HARD) that enhances fused cross-attention using scale-aligned selfattention maps, and (iii) a Test-Time Flipping (TTF) scheme designed to improve spatial consistency. Extensive experiments show that FastSeg achieves state-of-the-art training-free performance, obtaining 43.8% average mIoU across PASCAL VOC, PASCAL Context, and COCO Object benchmarks while maintaining superior inference efficiency. Our results demonstrate that FastSeg provides a strong foundation for extendability, bridging the gap between segmentation quality and inference efficiency.
>
---
#### [new 028] Time-variant Image Inpainting via Interactive Distribution Transition Estimation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出TAMP任务，解决时间差异图像修复问题。通过InDiTE模块和扩散模型，提升修复效果，并构建了TAMP-Street数据集。**

- **链接: [http://arxiv.org/pdf/2506.23461v1](http://arxiv.org/pdf/2506.23461v1)**

> **作者:** Yun Xing; Qing Guo; Xiaoguang Li; Yihao Huang; Xiaofeng Cao; Di Lin; Ivor Tsang; Lei Ma
>
> **摘要:** In this work, we focus on a novel and practical task, i.e., Time-vAriant iMage inPainting (TAMP). The aim of TAMP is to restore a damaged target image by leveraging the complementary information from a reference image, where both images captured the same scene but with a significant time gap in between, i.e., time-variant images. Different from conventional reference-guided image inpainting, the reference image under TAMP setup presents significant content distinction to the target image and potentially also suffers from damages. Such an application frequently happens in our daily lives to restore a damaged image by referring to another reference image, where there is no guarantee of the reference image's source and quality. In particular, our study finds that even state-of-the-art (SOTA) reference-guided image inpainting methods fail to achieve plausible results due to the chaotic image complementation. To address such an ill-posed problem, we propose a novel Interactive Distribution Transition Estimation (InDiTE) module which interactively complements the time-variant images with adaptive semantics thus facilitate the restoration of damaged regions. To further boost the performance, we propose our TAMP solution, namely Interactive Distribution Transition Estimation-driven Diffusion (InDiTE-Diff), which integrates InDiTE with SOTA diffusion model and conducts latent cross-reference during sampling. Moreover, considering the lack of benchmarks for TAMP task, we newly assembled a dataset, i.e., TAMP-Street, based on existing image and mask datasets. We conduct experiments on the TAMP-Street datasets under two different time-variant image inpainting settings, which show our method consistently outperform SOTA reference-guided image inpainting methods for solving TAMP.
>
---
#### [new 029] A High-Throughput Platform to Bench Test Smartphone-Based Heart Rate Measurements Derived From Video
- **分类: cs.CV**

- **简介: 该论文属于移动健康领域，旨在解决智能手机心率监测应用的性能评估与设备兼容性问题。通过构建高通量测试平台，实现自动化、标准化测试。**

- **链接: [http://arxiv.org/pdf/2506.23414v1](http://arxiv.org/pdf/2506.23414v1)**

> **作者:** Ming-Zher Poh; Jonathan Wang; Jonathan Hsu; Lawrence Cai; Eric Teasley; James A. Taylor; Jameson K. Rogers; Anupam Pathak; Shwetak Patel
>
> **摘要:** Smartphone-based heart rate (HR) monitoring apps using finger-over-camera photoplethysmography (PPG) face significant challenges in performance evaluation and device compatibility due to device variability and fragmentation. Manual testing is impractical, and standardized methods are lacking. This paper presents a novel, high-throughput bench-testing platform to address this critical need. We designed a system comprising a test rig capable of holding 12 smartphones for parallel testing, a method for generating synthetic PPG test videos with controllable HR and signal quality, and a host machine for coordinating video playback and data logging. The system achieved a mean absolute percentage error (MAPE) of 0.11% +/- 0.001% between input and measured HR, and a correlation coefficient of 0.92 +/- 0.008 between input and measured PPG signals using a clinically-validated smartphone-based HR app. Bench-testing results of 20 different smartphone models correctly classified all the devices as meeting the ANSI/CTA accuracy standards for HR monitors (MAPE <10%) when compared to a prospective clinical study with 80 participants, demonstrating high positive predictive value. This platform offers a scalable solution for pre-deployment testing of smartphone HR apps to improve app performance, ensure device compatibility, and advance the field of mobile health.
>
---
#### [new 030] Refine Any Object in Any Scene
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决场景中物体视角缺失问题。通过引入RAISE框架，利用3D生成先验提升物体几何与外观的精度。**

- **链接: [http://arxiv.org/pdf/2506.23835v1](http://arxiv.org/pdf/2506.23835v1)**

> **作者:** Ziwei Chen; Ziling Liu; Zitong Huang; Mingqi Gao; Feng Zheng
>
> **备注:** 9 pages with 6 figures
>
> **摘要:** Viewpoint missing of objects is common in scene reconstruction, as camera paths typically prioritize capturing the overall scene structure rather than individual objects. This makes it highly challenging to achieve high-fidelity object-level modeling while maintaining accurate scene-level representation. Addressing this issue is critical for advancing downstream tasks requiring detailed object understanding and appearance modeling. In this paper, we introduce Refine Any object In any ScenE (RAISE), a novel 3D enhancement framework that leverages 3D generative priors to recover fine-grained object geometry and appearance under missing views. Starting from substituting degraded objects with proxies, via a 3D generative model with strong 3D understanding, RAISE progressively refines geometry and texture by aligning each proxy to its degraded counterpart in 7-DOF pose, followed by correcting spatial and appearance inconsistencies via registration-constrained enhancement. This two-stage refinement ensures the high-fidelity geometry and appearance of the original object in unseen views while maintaining consistency in spatial positioning, observed geometry, and appearance. Extensive experiments on challenging benchmarks show that RAISE significantly outperforms state-of-the-art methods in both novel view synthesis and geometry completion tasks. RAISE is made publicly available at https://github.com/PolySummit/RAISE.
>
---
#### [new 031] STD-GS: Exploring Frame-Event Interaction for SpatioTemporal-Disentangled Gaussian Splatting to Reconstruct High-Dynamic Scene
- **分类: cs.CV**

- **简介: 该论文属于高动态场景重建任务，旨在解决静态背景与动态物体在时空特征上的不匹配问题。通过引入事件相机和高斯点云框架，实现时空解耦表示，提升动态场景的连续性渲染效果。**

- **链接: [http://arxiv.org/pdf/2506.23157v1](http://arxiv.org/pdf/2506.23157v1)**

> **作者:** Hanyu Zhou; Haonan Wang; Haoyue Liu; Yuxing Duan; Luxin Yan; Gim Hee Lee
>
> **摘要:** High-dynamic scene reconstruction aims to represent static background with rigid spatial features and dynamic objects with deformed continuous spatiotemporal features. Typically, existing methods adopt unified representation model (e.g., Gaussian) to directly match the spatiotemporal features of dynamic scene from frame camera. However, this unified paradigm fails in the potential discontinuous temporal features of objects due to frame imaging and the heterogeneous spatial features between background and objects. To address this issue, we disentangle the spatiotemporal features into various latent representations to alleviate the spatiotemporal mismatching between background and objects. In this work, we introduce event camera to compensate for frame camera, and propose a spatiotemporal-disentangled Gaussian splatting framework for high-dynamic scene reconstruction. As for dynamic scene, we figure out that background and objects have appearance discrepancy in frame-based spatial features and motion discrepancy in event-based temporal features, which motivates us to distinguish the spatiotemporal features between background and objects via clustering. As for dynamic object, we discover that Gaussian representations and event data share the consistent spatiotemporal characteristic, which could serve as a prior to guide the spatiotemporal disentanglement of object Gaussians. Within Gaussian splatting framework, the cumulative scene-object disentanglement can improve the spatiotemporal discrimination between background and objects to render the time-continuous dynamic scene. Extensive experiments have been performed to verify the superiority of the proposed method.
>
---
#### [new 032] Degradation-Modeled Multipath Diffusion for Tunable Metalens Photography
- **分类: cs.CV**

- **简介: 该论文属于图像恢复任务，解决金属镜头成像中的光学退化问题。通过建模退化过程和引入多路径扩散框架，提升图像质量与细节还原。**

- **链接: [http://arxiv.org/pdf/2506.22753v1](http://arxiv.org/pdf/2506.22753v1)**

> **作者:** Jianing Zhang; Jiayi Zhu; Feiyu Ji; Xiaokang Yang; Xiaoyun Yuan
>
> **摘要:** Metalenses offer significant potential for ultra-compact computational imaging but face challenges from complex optical degradation and computational restoration difficulties. Existing methods typically rely on precise optical calibration or massive paired datasets, which are non-trivial for real-world imaging systems. Furthermore, a lack of control over the inference process often results in undesirable hallucinated artifacts. We introduce Degradation-Modeled Multipath Diffusion for tunable metalens photography, leveraging powerful natural image priors from pretrained models instead of large datasets. Our framework uses positive, neutral, and negative-prompt paths to balance high-frequency detail generation, structural fidelity, and suppression of metalens-specific degradation, alongside \textit{pseudo} data augmentation. A tunable decoder enables controlled trade-offs between fidelity and perceptual quality. Additionally, a spatially varying degradation-aware attention (SVDA) module adaptively models complex optical and sensor-induced degradation. Finally, we design and build a millimeter-scale MetaCamera for real-world validation. Extensive results show that our approach outperforms state-of-the-art methods, achieving high-fidelity and sharp image reconstruction. More materials: https://dmdiff.github.io/.
>
---
#### [new 033] MReg: A Novel Regression Model with MoE-based Video Feature Mining for Mitral Regurgitation Diagnosis
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决二尖瓣反流自动诊断问题。通过回归模型与特征提取方法提升诊断准确性与可解释性。**

- **链接: [http://arxiv.org/pdf/2506.23648v1](http://arxiv.org/pdf/2506.23648v1)**

> **作者:** Zhe Liu; Yuhao Huang; Lian Liu; Chengrui Zhang; Haotian Lin; Tong Han; Zhiyuan Zhu; Yanlin Chen; Yuerui Chen; Dong Ni; Zhongshan Gou; Xin Yang
>
> **备注:** 10 pages, 5 figures, accepted by MICCAI 2025
>
> **摘要:** Color Doppler echocardiography is a crucial tool for diagnosing mitral regurgitation (MR). Recent studies have explored intelligent methods for MR diagnosis to minimize user dependence and improve accuracy. However, these approaches often fail to align with clinical workflow and may lead to suboptimal accuracy and interpretability. In this study, we introduce an automated MR diagnosis model (MReg) developed on the 4-chamber cardiac color Doppler echocardiography video (A4C-CDV). It follows comprehensive feature mining strategies to detect MR and assess its severity, considering clinical realities. Our contribution is threefold. First, we formulate the MR diagnosis as a regression task to capture the continuity and ordinal relationships between categories. Second, we design a feature selection and amplification mechanism to imitate the sonographer's diagnostic logic for accurate MR grading. Third, inspired by the Mixture-of-Experts concept, we introduce a feature summary module to extract the category-level features, enhancing the representational capacity for more accurate grading. We trained and evaluated our proposed MReg on a large in-house A4C-CDV dataset comprising 1868 cases with three graded regurgitation labels. Compared to other weakly supervised video anomaly detection and supervised classification methods, MReg demonstrated superior performance in MR diagnosis. Our code is available at: https://github.com/cskdstz/MReg.
>
---
#### [new 034] Dynamic View Synthesis from Small Camera Motion Videos
- **分类: cs.CV**

- **简介: 该论文属于动态场景新视角合成任务，解决小相机运动下场景几何和相机参数估计问题。提出分布深度正则化方法提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.23153v1](http://arxiv.org/pdf/2506.23153v1)**

> **作者:** Huiqiang Sun; Xingyi Li; Juewen Peng; Liao Shen; Zhiguo Cao; Ke Xian; Guosheng Lin
>
> **备注:** Accepted by TVCG
>
> **摘要:** Novel view synthesis for dynamic $3$D scenes poses a significant challenge. Many notable efforts use NeRF-based approaches to address this task and yield impressive results. However, these methods rely heavily on sufficient motion parallax in the input images or videos. When the camera motion range becomes limited or even stationary (i.e., small camera motion), existing methods encounter two primary challenges: incorrect representation of scene geometry and inaccurate estimation of camera parameters. These challenges make prior methods struggle to produce satisfactory results or even become invalid. To address the first challenge, we propose a novel Distribution-based Depth Regularization (DDR) that ensures the rendering weight distribution to align with the true distribution. Specifically, unlike previous methods that use depth loss to calculate the error of the expectation, we calculate the expectation of the error by using Gumbel-softmax to differentiably sample points from discrete rendering weight distribution. Additionally, we introduce constraints that enforce the volume density of spatial points before the object boundary along the ray to be near zero, ensuring that our model learns the correct geometry of the scene. To demystify the DDR, we further propose a visualization tool that enables observing the scene geometry representation at the rendering weight level. For the second challenge, we incorporate camera parameter learning during training to enhance the robustness of our model to camera parameters. We conduct extensive experiments to demonstrate the effectiveness of our approach in representing scenes with small camera motion input, and our results compare favorably to state-of-the-art methods.
>
---
#### [new 035] Lightning the Night with Generative Artificial Intelligence
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文属于夜间可见光反射率反演任务，旨在解决夜间无法利用可见光数据的问题。通过生成扩散模型，实现了FY4B卫星夜间可见光反射率的高精度估算。**

- **链接: [http://arxiv.org/pdf/2506.22511v1](http://arxiv.org/pdf/2506.22511v1)**

> **作者:** Tingting Zhou; Feng Zhang; Haoyang Fu; Baoxiang Pan; Renhe Zhang; Feng Lu; Zhixin Yang
>
> **摘要:** The visible light reflectance data from geostationary satellites is crucial for meteorological observations and plays an important role in weather monitoring and forecasting. However, due to the lack of visible light at night, it is impossible to conduct continuous all-day weather observations using visible light reflectance data. This study pioneers the use of generative diffusion models to address this limitation. Based on the multi-band thermal infrared brightness temperature data from the Advanced Geostationary Radiation Imager (AGRI) onboard the Fengyun-4B (FY4B) geostationary satellite, we developed a high-precision visible light reflectance retrieval model, called Reflectance Diffusion (RefDiff), which enables 0.47~\mu\mathrm{m}, 0.65~\mu\mathrm{m}, and 0.825~\mu\mathrm{m} bands visible light reflectance retrieval at night. Compared to the classical models, RefDiff not only significantly improves accuracy through ensemble averaging but also provides uncertainty estimation. Specifically, the SSIM index of RefDiff can reach 0.90, with particularly significant improvements in areas with complex cloud structures and thick clouds. The model's nighttime retrieval capability was validated using VIIRS nighttime product, demonstrating comparable performance to its daytime counterpart. In summary, this research has made substantial progress in the ability to retrieve visible light reflectance at night, with the potential to expand the application of nighttime visible light data.
>
---
#### [new 036] Endo-4DGX: Robust Endoscopic Scene Reconstruction and Illumination Correction with Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于医学图像重建任务，旨在解决内窥镜场景中光照变化导致的重建质量下降问题。通过引入光照自适应的高斯点云方法，提升不同光照条件下的渲染效果。**

- **链接: [http://arxiv.org/pdf/2506.23308v1](http://arxiv.org/pdf/2506.23308v1)**

> **作者:** Yiming Huang; Long Bai; Beilei Cui; Yanheng Li; Tong Chen; Jie Wang; Jinlin Wu; Zhen Lei; Hongbin Liu; Hongliang Ren
>
> **备注:** MICCAI 2025. Project Page: https://lastbasket.github.io/MICCAI-2025-Endo-4DGX/
>
> **摘要:** Accurate reconstruction of soft tissue is crucial for advancing automation in image-guided robotic surgery. The recent 3D Gaussian Splatting (3DGS) techniques and their variants, 4DGS, achieve high-quality renderings of dynamic surgical scenes in real-time. However, 3D-GS-based methods still struggle in scenarios with varying illumination, such as low light and over-exposure. Training 3D-GS in such extreme light conditions leads to severe optimization problems and devastating rendering quality. To address these challenges, we present Endo-4DGX, a novel reconstruction method with illumination-adaptive Gaussian Splatting designed specifically for endoscopic scenes with uneven lighting. By incorporating illumination embeddings, our method effectively models view-dependent brightness variations. We introduce a region-aware enhancement module to model the sub-area lightness at the Gaussian level and a spatial-aware adjustment module to learn the view-consistent brightness adjustment. With the illumination adaptive design, Endo-4DGX achieves superior rendering performance under both low-light and over-exposure conditions while maintaining geometric accuracy. Additionally, we employ an exposure control loss to restore the appearance from adverse exposure to the normal level for illumination-adaptive optimization. Experimental results demonstrate that Endo-4DGX significantly outperforms combinations of state-of-the-art reconstruction and restoration methods in challenging lighting environments, underscoring its potential to advance robot-assisted surgical applications. Our code is available at https://github.com/lastbasket/Endo-4DGX.
>
---
#### [new 037] Causal-Entity Reflected Egocentric Traffic Accident Video Synthesis
- **分类: cs.CV**

- **简介: 该论文属于交通事故视频生成任务，旨在解决如何在合成视频中准确反映因果关系的问题。工作包括提出Causal-VidSyn模型和构建Drive-Gaze数据集。**

- **链接: [http://arxiv.org/pdf/2506.23263v1](http://arxiv.org/pdf/2506.23263v1)**

> **作者:** Lei-lei Li; Jianwu Fang; Junbin Xiao; Shanmin Pang; Hongkai Yu; Chen Lv; Jianru Xue; Tat-Seng Chua
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Egocentricly comprehending the causes and effects of car accidents is crucial for the safety of self-driving cars, and synthesizing causal-entity reflected accident videos can facilitate the capability test to respond to unaffordable accidents in reality. However, incorporating causal relations as seen in real-world videos into synthetic videos remains challenging. This work argues that precisely identifying the accident participants and capturing their related behaviors are of critical importance. In this regard, we propose a novel diffusion model, Causal-VidSyn, for synthesizing egocentric traffic accident videos. To enable causal entity grounding in video diffusion, Causal-VidSyn leverages the cause descriptions and driver fixations to identify the accident participants and behaviors, facilitated by accident reason answering and gaze-conditioned selection modules. To support Causal-VidSyn, we further construct Drive-Gaze, the largest driver gaze dataset (with 1.54M frames of fixations) in driving accident scenarios. Extensive experiments show that Causal-VidSyn surpasses state-of-the-art video diffusion models in terms of frame quality and causal sensitivity in various tasks, including accident video editing, normal-to-accident video diffusion, and text-to-video generation.
>
---
#### [new 038] Neural Cellular Automata: From Cells to Pixels
- **分类: cs.CV; cs.GR; cs.LG; cs.MA; eess.IV**

- **简介: 该论文属于图像生成任务，旨在解决NCAs在高分辨率下的效率与性能问题。通过引入隐式解码器和优化损失函数，提升其生成质量与计算效率。**

- **链接: [http://arxiv.org/pdf/2506.22899v1](http://arxiv.org/pdf/2506.22899v1)**

> **作者:** Ehsan Pajouheshgar; Yitao Xu; Ali Abbasi; Alexander Mordvintsev; Wenzel Jakob; Sabine Süsstrunk
>
> **备注:** 6 pages, 5 figures, first draft
>
> **摘要:** Neural Cellular Automata (NCAs) are bio-inspired systems in which identical cells self-organize to form complex and coherent patterns by repeatedly applying simple local rules. NCAs display striking emergent behaviors including self-regeneration, generalization and robustness to unseen situations, and spontaneous motion. Despite their success in texture synthesis and morphogenesis, NCAs remain largely confined to low-resolution grids. This limitation stems from (1) training time and memory requirements that grow quadratically with grid size, (2) the strictly local propagation of information which impedes long-range cell communication, and (3) the heavy compute demands of real-time inference at high resolution. In this work, we overcome this limitation by pairing NCA with a tiny, shared implicit decoder, inspired by recent advances in implicit neural representations. Following NCA evolution on a coarse grid, a lightweight decoder renders output images at arbitrary resolution. We also propose novel loss functions for both morphogenesis and texture synthesis tasks, specifically tailored for high-resolution output with minimal memory and computation overhead. Combining our proposed architecture and loss functions brings substantial improvement in quality, efficiency, and performance. NCAs equipped with our implicit decoder can generate full-HD outputs in real time while preserving their self-organizing, emergent properties. Moreover, because each MLP processes cell states independently, inference remains highly parallelizable and efficient. We demonstrate the applicability of our approach across multiple NCA variants (on 2D, 3D grids, and 3D meshes) and multiple tasks, including texture generation and morphogenesis (growing patterns from a seed), showing that with our proposed framework, NCAs seamlessly scale to high-resolution outputs with minimal computational overhead.
>
---
#### [new 039] PhonemeFake: Redefining Deepfake Realism with Language-Driven Segmental Manipulation and Adaptive Bilevel Detection
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于深度伪造检测任务，旨在解决现有数据不真实、检测效率低的问题。提出PhonemeFake攻击和自适应检测模型，提升检测精度与速度。**

- **链接: [http://arxiv.org/pdf/2506.22783v1](http://arxiv.org/pdf/2506.22783v1)**

> **作者:** Oguzhan Baser; Ahmet Ege Tanriverdi; Sriram Vishwanath; Sandeep P. Chinchali
>
> **备注:** 5 pages, 3 figures, Published at Proceedings of Interspeech 2025, for the dataset see https://huggingface.co/datasets/phonemefake/PhonemeFakeV2, for the code see https://github.com/UTAustin-SwarmLab/ PhonemeFake
>
> **摘要:** Deepfake (DF) attacks pose a growing threat as generative models become increasingly advanced. However, our study reveals that existing DF datasets fail to deceive human perception, unlike real DF attacks that influence public discourse. It highlights the need for more realistic DF attack vectors. We introduce PhonemeFake (PF), a DF attack that manipulates critical speech segments using language reasoning, significantly reducing human perception by up to 42% and benchmark accuracies by up to 94%. We release an easy-to-use PF dataset on HuggingFace and open-source bilevel DF segment detection model that adaptively prioritizes compute on manipulated regions. Our extensive experiments across three known DF datasets reveal that our detection model reduces EER by 91% while achieving up to 90% speed-up, with minimal compute overhead and precise localization beyond existing models as a scalable solution.
>
---
#### [new 040] MusiXQA: Advancing Visual Music Understanding in Multimodal Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于多模态语言模型任务，旨在解决音乐谱图理解问题。提出MusiXQA数据集和Phi-3-MusiX模型，提升MLLM在音乐 sheet 的理解能力。**

- **链接: [http://arxiv.org/pdf/2506.23009v1](http://arxiv.org/pdf/2506.23009v1)**

> **作者:** Jian Chen; Wenye Ma; Penghang Liu; Wei Wang; Tengwei Song; Ming Li; Chenguang Wang; Ruiyi Zhang; Changyou Chen
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved remarkable visual reasoning abilities in natural images, text-rich documents, and graphic designs. However, their ability to interpret music sheets remains underexplored. To bridge this gap, we introduce MusiXQA, the first comprehensive dataset for evaluating and advancing MLLMs in music sheet understanding. MusiXQA features high-quality synthetic music sheets generated via MusiXTeX, with structured annotations covering note pitch and duration, chords, clefs, key/time signatures, and text, enabling diverse visual QA tasks. Through extensive evaluations, we reveal significant limitations of current state-of-the-art MLLMs in this domain. Beyond benchmarking, we developed Phi-3-MusiX, an MLLM fine-tuned on our dataset, achieving significant performance gains over GPT-based methods. The proposed dataset and model establish a foundation for future advances in MLLMs for music sheet understanding. Code, data, and model will be released upon acceptance.
>
---
#### [new 041] MotionGPT3: Human Motion as a Second Modality
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出MotionGPT3，解决人类运动与语言统一建模的问题，通过将运动作为第二模态，实现高效跨模态交互和语言能力保持。**

- **链接: [http://arxiv.org/pdf/2506.24086v1](http://arxiv.org/pdf/2506.24086v1)**

> **作者:** Bingfan Zhu; Biao Jiang; Sunyi Wang; Shixiang Tang; Tao Chen; Linjie Luo; Youyi Zheng; Xin Chen
>
> **备注:** 21 pages, 8 figures
>
> **摘要:** Though recent advances in multimodal models have demonstrated strong capabilities and opportunities in unified understanding and generation, the development of unified motion-language models remains underexplored. To enable such models with high-fidelity human motion, two core challenges must be addressed. The first is the reconstruction gap between the continuous motion modality and discrete representation in an autoregressive manner, and the second is the degradation of language intelligence during unified training. Inspired by the mixture of experts, we propose MotionGPT3, a bimodal motion-language model that treats human motion as a second modality, decoupling motion modeling via separate model parameters and enabling both effective cross-modal interaction and efficient multimodal scaling training. To preserve language intelligence, the text branch retains the original structure and parameters of the pretrained language model, while a new motion branch is integrated via a shared attention mechanism, enabling bidirectional information flow between two modalities. We first employ a motion Variational Autoencoder (VAE) to encode raw human motion into latent representations. Based on this continuous latent space, the motion branch predicts motion latents directly from intermediate hidden states using a diffusion head, bypassing discrete tokenization. Extensive experiments show that our approach achieves competitive performance on both motion understanding and generation tasks while preserving strong language capabilities, establishing a unified bimodal motion diffusion framework within an autoregressive manner.
>
---
#### [new 042] GeoCD: A Differential Local Approximation for Geodesic Chamfer Distance
- **分类: cs.CV**

- **简介: 该论文属于3D点云学习任务，旨在解决传统Chamfer Distance忽略几何结构的问题，提出GeoCD作为拓扑感知的可微分近似方法。**

- **链接: [http://arxiv.org/pdf/2506.23478v1](http://arxiv.org/pdf/2506.23478v1)**

> **作者:** Pedro Alonso; Tianrui Li; Chongshou Li
>
> **摘要:** Chamfer Distance (CD) is a widely adopted metric in 3D point cloud learning due to its simplicity and efficiency. However, it suffers from a fundamental limitation: it relies solely on Euclidean distances, which often fail to capture the intrinsic geometry of 3D shapes. To address this limitation, we propose GeoCD, a topology-aware and fully differentiable approximation of geodesic distance designed to serve as a metric for 3D point cloud learning. Our experiments show that GeoCD consistently improves reconstruction quality over standard CD across various architectures and datasets. We demonstrate this by fine-tuning several models, initially trained with standard CD, using GeoCD. Remarkably, fine-tuning for a single epoch with GeoCD yields significant gains across multiple evaluation metrics.
>
---
#### [new 043] Attention to Burstiness: Low-Rank Bilinear Prompt Tuning
- **分类: cs.CV**

- **简介: 该论文属于视觉提示调优任务，旨在解决非高斯分布带来的学习挑战，通过白化和低秩双线性方法提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.22908v1](http://arxiv.org/pdf/2506.22908v1)**

> **作者:** Yuzhu Wang; Manni Duan; Shu Kong
>
> **备注:** ICCV 2025
>
> **摘要:** Visual Prompt Tuning (VPT) is a parameter-efficient fune-tuning technique that adapts a pre-trained vision Transformer (ViT) by learning a small set of parameters in the input space, known as prompts. In VPT, we uncover ``burstiness'' in the values arising from the interaction of image patch embeddings, and the key and query projectors within Transformer's self-attention module. Furthermore, the values of patch embeddings and the key and query projectors exhibit Laplacian and hyper-Laplacian distribution, respectively. Intuitively, these non-Gaussian distributions pose challenges for learning prompts. To address this, we propose whitening these data, de-correlating them and equalizing their variance towards more Gaussian before learning prompts. We derive the whitening matrix over random image patch embeddings and ViT's key and query projectors, and multiply it with the prompt to be learned in a bilinear manner. Surprisingly, this method significantly accelerates prompt tuning and boosts accuracy, e.g., $>$25 accuracy points on the CUB dataset; interestingly, it learns ``bursty prompts''. Extending the bilinear model which is known to introduce burstiness, we present a compact, low-rank version by learning two smaller matrices whose multiplication yields the final prompts. We call the proposed methods Bilinear Prompt Tuning (BPT). Extensive experiments across multiple benchmark datasets demonstrate that BPT methods not only outperform various VPT methods but also reduce parameter count and computation overhead.
>
---
#### [new 044] Contrastive Learning with Diffusion Features for Weakly Supervised Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于弱监督医学图像分割任务，旨在解决传统方法在对象边界和激活区域不准确的问题。通过结合对比学习与扩散特征，提升分割性能。**

- **链接: [http://arxiv.org/pdf/2506.23460v1](http://arxiv.org/pdf/2506.23460v1)**

> **作者:** Dewen Zeng; Xinrong Hu; Yu-Jen Chen; Yawen Wu; Xiaowei Xu; Yiyu Shi
>
> **摘要:** Weakly supervised semantic segmentation (WSSS) methods using class labels often rely on class activation maps (CAMs) to localize objects. However, traditional CAM-based methods struggle with partial activations and imprecise object boundaries due to optimization discrepancies between classification and segmentation. Recently, the conditional diffusion model (CDM) has been used as an alternative for generating segmentation masks in WSSS, leveraging its strong image generation capabilities tailored to specific class distributions. By modifying or perturbing the condition during diffusion sampling, the related objects can be highlighted in the generated images. Yet, the saliency maps generated by CDMs are prone to noise from background alterations during reverse diffusion. To alleviate the problem, we introduce Contrastive Learning with Diffusion Features (CLDF), a novel method that uses contrastive learning to train a pixel decoder to map the diffusion features from a frozen CDM to a low-dimensional embedding space for segmentation. Specifically, we integrate gradient maps generated from CDM external classifier with CAMs to identify foreground and background pixels with fewer false positives/negatives for contrastive learning, enabling robust pixel embedding learning. Experimental results on four segmentation tasks from two public medical datasets demonstrate that our method significantly outperforms existing baselines.
>
---
#### [new 045] SynMotion: Semantic-Visual Adaptation for Motion Customized Video Generation
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决运动定制中的语义与视觉不一致问题。提出SynMotion模型，结合语义引导和视觉适配，提升运动质量和多样性。**

- **链接: [http://arxiv.org/pdf/2506.23690v1](http://arxiv.org/pdf/2506.23690v1)**

> **作者:** Shuai Tan; Biao Gong; Yujie Wei; Shiwei Zhang; Zhuoxin Liu; Dandan Zheng; Jingdong Chen; Yan Wang; Hao Ouyang; Kecheng Zheng; Yujun Shen
>
> **备注:** Project page: https://lucaria-academy.github.io/SynMotion/
>
> **摘要:** Diffusion-based video motion customization facilitates the acquisition of human motion representations from a few video samples, while achieving arbitrary subjects transfer through precise textual conditioning. Existing approaches often rely on semantic-level alignment, expecting the model to learn new motion concepts and combine them with other entities (e.g., ''cats'' or ''dogs'') to produce visually appealing results. However, video data involve complex spatio-temporal patterns, and focusing solely on semantics cause the model to overlook the visual complexity of motion. Conversely, tuning only the visual representation leads to semantic confusion in representing the intended action. To address these limitations, we propose SynMotion, a new motion-customized video generation model that jointly leverages semantic guidance and visual adaptation. At the semantic level, we introduce the dual-embedding semantic comprehension mechanism which disentangles subject and motion representations, allowing the model to learn customized motion features while preserving its generative capabilities for diverse subjects. At the visual level, we integrate parameter-efficient motion adapters into a pre-trained video generation model to enhance motion fidelity and temporal coherence. Furthermore, we introduce a new embedding-specific training strategy which \textbf{alternately optimizes} subject and motion embeddings, supported by the manually constructed Subject Prior Video (SPV) training dataset. This strategy promotes motion specificity while preserving generalization across diverse subjects. Lastly, we introduce MotionBench, a newly curated benchmark with diverse motion patterns. Experimental results across both T2V and I2V settings demonstrate that \method outperforms existing baselines. Project page: https://lucaria-academy.github.io/SynMotion/
>
---
#### [new 046] Foundation Models for Zero-Shot Segmentation of Scientific Images without AI-Ready Data
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于图像分割任务，解决科学图像数据稀缺导致的零样本分割难题。提出Zenesis平台，结合多模态适应与人机交互，提升分割精度。**

- **链接: [http://arxiv.org/pdf/2506.24039v1](http://arxiv.org/pdf/2506.24039v1)**

> **作者:** Shubhabrata Mukherjee; Jack Lang; Obeen Kwon; Iryna Zenyuk; Valerie Brogden; Adam Weber; Daniela Ushizima
>
> **备注:** This manuscript is a draft on arxiv. A final version has been submitted to the 59th ICPP 2025, DRAI workshop
>
> **摘要:** Zero-shot and prompt-based technologies capitalized on using frequently occurring images to transform visual reasoning tasks, which explains why such technologies struggle with valuable yet scarce scientific image sets. In this work, we propose Zenesis, a comprehensive no-code interactive platform designed to minimize barriers posed by data readiness for scientific images. We develop lightweight multi-modal adaptation techniques that enable zero-shot operation on raw scientific data, along with human-in-the-loop refinement and heuristic-based temporal enhancement options. We demonstrate the performance of our approach through comprehensive comparison and validation on challenging Focused Ion Beam Scanning Electron Microscopy (FIB-SEM) data of catalyst-loaded membranes. Zenesis significantly outperforms baseline methods, achieving an average accuracy of 0.947, an Intersection over Union (IOU) of 0.858, and a Dice score of 0.923 for amorphous catalyst samples and accuracy of 0.987, an IOU of 0.857, and a Dice score of 0.923 for crystalline samples. These results mark a substantial improvement over traditional methods like Otsu thresholding and even advanced models like Segment Anything Model (SAM) when used in isolation. Our results demonstrate that Zenesis is a powerful tool for scientific applications, particularly in fields where high-quality annotated datasets are unavailable, accelerating accurate analysis of experimental imaging.
>
---
#### [new 047] OmniVCus: Feedforward Subject-driven Video Customization with Multimodal Control Conditions
- **分类: cs.CV**

- **简介: 该论文属于视频定制任务，解决多主体视频编辑问题。提出数据生成方法和扩散Transformer框架，实现多模态控制下的视频主体编辑。**

- **链接: [http://arxiv.org/pdf/2506.23361v1](http://arxiv.org/pdf/2506.23361v1)**

> **作者:** Yuanhao Cai; He Zhang; Xi Chen; Jinbo Xing; Yiwei Hu; Yuqian Zhou; Kai Zhang; Zhifei Zhang; Soo Ye Kim; Tianyu Wang; Yulun Zhang; Xiaokang Yang; Zhe Lin; Alan Yuille
>
> **备注:** A data construction pipeline and a diffusion Transformer framework for controllable subject-driven video customization
>
> **摘要:** Existing feedforward subject-driven video customization methods mainly study single-subject scenarios due to the difficulty of constructing multi-subject training data pairs. Another challenging problem that how to use the signals such as depth, mask, camera, and text prompts to control and edit the subject in the customized video is still less explored. In this paper, we first propose a data construction pipeline, VideoCus-Factory, to produce training data pairs for multi-subject customization from raw videos without labels and control signals such as depth-to-video and mask-to-video pairs. Based on our constructed data, we develop an Image-Video Transfer Mixed (IVTM) training with image editing data to enable instructive editing for the subject in the customized video. Then we propose a diffusion Transformer framework, OmniVCus, with two embedding mechanisms, Lottery Embedding (LE) and Temporally Aligned Embedding (TAE). LE enables inference with more subjects by using the training subjects to activate more frame embeddings. TAE encourages the generation process to extract guidance from temporally aligned control signals by assigning the same frame embeddings to the control and noise tokens. Experiments demonstrate that our method significantly surpasses state-of-the-art methods in both quantitative and qualitative evaluations. Video demos are at our project page: https://caiyuanhao1998.github.io/project/OmniVCus/. Our code will be released at https://github.com/caiyuanhao1998/Open-OmniVCus
>
---
#### [new 048] Sanitizing Manufacturing Dataset Labels Using Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于数据清洗任务，旨在解决制造领域图像数据集中的标签噪声问题。通过视觉-语言模型进行标签净化与聚类，提升数据质量。**

- **链接: [http://arxiv.org/pdf/2506.23465v1](http://arxiv.org/pdf/2506.23465v1)**

> **作者:** Nazanin Mahjourian; Vinh Nguyen
>
> **摘要:** The success of machine learning models in industrial applications is heavily dependent on the quality of the datasets used to train the models. However, large-scale datasets, specially those constructed from crowd-sourcing and web-scraping, often suffer from label noise, inconsistencies, and errors. This problem is particularly pronounced in manufacturing domains, where obtaining high-quality labels is costly and time-consuming. This paper introduces Vision-Language Sanitization and Refinement (VLSR), which is a vision-language-based framework for label sanitization and refinement in multi-label manufacturing image datasets. This method embeds both images and their associated textual labels into a shared semantic space leveraging the CLIP vision-language model. Then two key tasks are addressed in this process by computing the cosine similarity between embeddings. First, label sanitization is performed to identify irrelevant, misspelled, or semantically weak labels, and surface the most semantically aligned label for each image by comparing image-label pairs using cosine similarity between image and label embeddings. Second, the method applies density-based clustering on text embeddings, followed by iterative cluster merging, to group semantically similar labels into unified label groups. The Factorynet dataset, which includes noisy labels from both human annotations and web-scraped sources, is employed to evaluate the effectiveness of the proposed framework. Experimental results demonstrate that the VLSR framework successfully identifies problematic labels and improves label consistency. This method enables a significant reduction in label vocabulary through clustering, which ultimately enhances the dataset's quality for training robust machine learning models in industrial applications with minimal human intervention.
>
---
#### [new 049] BrainMT: A Hybrid Mamba-Transformer Architecture for Modeling Long-Range Dependencies in Functional MRI Data
- **分类: cs.CV**

- **简介: 该论文属于神经影像分析任务，旨在解决fMRI数据中长程依赖建模问题。提出BrainMT混合架构，结合Mamba和Transformer，提升分类与回归性能。**

- **链接: [http://arxiv.org/pdf/2506.22591v1](http://arxiv.org/pdf/2506.22591v1)**

> **作者:** Arunkumar Kannan; Martin A. Lindquist; Brian Caffo
>
> **备注:** Accepted at MICCAI 2025
>
> **摘要:** Recent advances in deep learning have made it possible to predict phenotypic measures directly from functional magnetic resonance imaging (fMRI) brain volumes, sparking significant interest in the neuroimaging community. However, existing approaches, primarily based on convolutional neural networks or transformer architectures, often struggle to model the complex relationships inherent in fMRI data, limited by their inability to capture long-range spatial and temporal dependencies. To overcome these shortcomings, we introduce BrainMT, a novel hybrid framework designed to efficiently learn and integrate long-range spatiotemporal attributes in fMRI data. Our framework operates in two stages: (1) a bidirectional Mamba block with a temporal-first scanning mechanism to capture global temporal interactions in a computationally efficient manner; and (2) a transformer block leveraging self-attention to model global spatial relationships across the deep features processed by the Mamba block. Extensive experiments on two large-scale public datasets, UKBioBank and the Human Connectome Project, demonstrate that BrainMT achieves state-of-the-art performance on both classification (sex prediction) and regression (cognitive intelligence prediction) tasks, outperforming existing methods by a significant margin. Our code and implementation details will be made publicly available at this https://github.com/arunkumar-kannan/BrainMT-fMRI
>
---
#### [new 050] Dataset Distillation via Vision-Language Category Prototype
- **分类: cs.CV**

- **简介: 该论文属于数据集蒸馏任务，旨在提升模型泛化能力。通过引入视觉-语言文本原型，融合图像与语言信息，生成逻辑连贯的数据，解决传统方法忽视语义信息的问题。**

- **链接: [http://arxiv.org/pdf/2506.23580v1](http://arxiv.org/pdf/2506.23580v1)**

> **作者:** Yawen Zou; Guang Li; Duo Su; Zi Wang; Jun Yu; Chao Zhang
>
> **备注:** accepted by ICCV2025
>
> **摘要:** Dataset distillation (DD) condenses large datasets into compact yet informative substitutes, preserving performance comparable to the original dataset while reducing storage, transmission costs, and computational consumption. However, previous DD methods mainly focus on distilling information from images, often overlooking the semantic information inherent in the data. The disregard for context hinders the model's generalization ability, particularly in tasks involving complex datasets, which may result in illogical outputs or the omission of critical objects. In this study, we integrate vision-language methods into DD by introducing text prototypes to distill language information and collaboratively synthesize data with image prototypes, thereby enhancing dataset distillation performance. Notably, the text prototypes utilized in this study are derived from descriptive text information generated by an open-source large language model. This framework demonstrates broad applicability across datasets without pre-existing text descriptions, expanding the potential of dataset distillation beyond traditional image-based approaches. Compared to other methods, the proposed approach generates logically coherent images containing target objects, achieving state-of-the-art validation performance and demonstrating robust generalization. Source code and generated data are available in https://github.com/zou-yawen/Dataset-Distillation-via-Vision-Language-Category-Prototype/
>
---
#### [new 051] Partial Forward Blocking: A Novel Data Pruning Paradigm for Lossless Training Acceleration
- **分类: cs.CV**

- **简介: 该论文属于机器学习训练加速任务，旨在解决数据量大导致的计算成本高问题。提出PFB方法，通过特征评估进行无损数据剪枝，提升训练效率。**

- **链接: [http://arxiv.org/pdf/2506.23674v1](http://arxiv.org/pdf/2506.23674v1)**

> **作者:** Dongyue Wu; Zilin Guo; Jialong Zuo; Nong Sang; Changxin Gao
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** The ever-growing size of training datasets enhances the generalization capability of modern machine learning models but also incurs exorbitant computational costs. Existing data pruning approaches aim to accelerate training by removing those less important samples. However, they often rely on gradients or proxy models, leading to prohibitive additional costs of gradient back-propagation and proxy model training. In this paper, we propose Partial Forward Blocking (PFB), a novel framework for lossless training acceleration. The efficiency of PFB stems from its unique adaptive pruning pipeline: sample importance is assessed based on features extracted from the shallow layers of the target model. Less important samples are then pruned, allowing only the retained ones to proceed with the subsequent forward pass and loss back-propagation. This mechanism significantly reduces the computational overhead of deep-layer forward passes and back-propagation for pruned samples, while also eliminating the need for auxiliary backward computations and proxy model training. Moreover, PFB introduces probability density as an indicator of sample importance. Combined with an adaptive distribution estimation module, our method dynamically prioritizes relatively rare samples, aligning with the constantly evolving training state. Extensive experiments demonstrate the significant superiority of PFB in performance and speed. On ImageNet, PFB achieves a 0.5% accuracy improvement and 33% training time reduction with 40% data pruned.
>
---
#### [new 052] Uncertainty-aware Diffusion and Reinforcement Learning for Joint Plane Localization and Anomaly Diagnosis in 3D Ultrasound
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于3D超声图像分析任务，旨在解决子宫畸形自动定位与诊断问题。通过扩散模型、强化学习和不确定性建模实现精准分割与分类。**

- **链接: [http://arxiv.org/pdf/2506.23538v1](http://arxiv.org/pdf/2506.23538v1)**

> **作者:** Yuhao Huang; Yueyue Xu; Haoran Dou; Jiaxiao Deng; Xin Yang; Hongyu Zheng; Dong Ni
>
> **备注:** Accepted by MICCAI 2025;10 pages, 3 figures
>
> **摘要:** Congenital uterine anomalies (CUAs) can lead to infertility, miscarriage, preterm birth, and an increased risk of pregnancy complications. Compared to traditional 2D ultrasound (US), 3D US can reconstruct the coronal plane, providing a clear visualization of the uterine morphology for assessing CUAs accurately. In this paper, we propose an intelligent system for simultaneous automated plane localization and CUA diagnosis. Our highlights are: 1) we develop a denoising diffusion model with local (plane) and global (volume/text) guidance, using an adaptive weighting strategy to optimize attention allocation to different conditions; 2) we introduce a reinforcement learning-based framework with unsupervised rewards to extract the key slice summary from redundant sequences, fully integrating information across multiple planes to reduce learning difficulty; 3) we provide text-driven uncertainty modeling for coarse prediction, and leverage it to adjust the classification probability for overall performance improvement. Extensive experiments on a large 3D uterine US dataset show the efficacy of our method, in terms of plane localization and CUA diagnosis. Code is available at https://github.com/yuhoo0302/CUA-US.
>
---
#### [new 053] Calligrapher: Freestyle Text Image Customization
- **分类: cs.CV**

- **简介: 该论文提出Calligrapher，属于文本图像定制任务，解决风格控制与数据依赖问题，通过自蒸馏和风格注入实现精准字体设计。**

- **链接: [http://arxiv.org/pdf/2506.24123v1](http://arxiv.org/pdf/2506.24123v1)**

> **作者:** Yue Ma; Qingyan Bai; Hao Ouyang; Ka Leong Cheng; Qiuyu Wang; Hongyu Liu; Zichen Liu; Haofan Wang; Jingye Chen; Yujun Shen; Qifeng Chen
>
> **备注:** Project page: https://calligrapher2025.github.io/Calligrapher Code: https://github.com/Calligrapher2025/Calligrapher
>
> **摘要:** We introduce Calligrapher, a novel diffusion-based framework that innovatively integrates advanced text customization with artistic typography for digital calligraphy and design applications. Addressing the challenges of precise style control and data dependency in typographic customization, our framework incorporates three key technical contributions. First, we develop a self-distillation mechanism that leverages the pre-trained text-to-image generative model itself alongside the large language model to automatically construct a style-centric typography benchmark. Second, we introduce a localized style injection framework via a trainable style encoder, which comprises both Qformer and linear layers, to extract robust style features from reference images. An in-context generation mechanism is also employed to directly embed reference images into the denoising process, further enhancing the refined alignment of target styles. Extensive quantitative and qualitative evaluations across diverse fonts and design contexts confirm Calligrapher's accurate reproduction of intricate stylistic details and precise glyph positioning. By automating high-quality, visually consistent typography, Calligrapher surpasses traditional models, empowering creative practitioners in digital art, branding, and contextual typographic design.
>
---
#### [new 054] DMD-Net: Deep Mesh Denoising Network
- **分类: cs.CV**

- **简介: 该论文属于三维网格去噪任务，旨在解决噪声网格的修复问题。提出DMD-Net网络，结合图卷积与Transformer，有效去除各种噪声。**

- **链接: [http://arxiv.org/pdf/2506.22850v1](http://arxiv.org/pdf/2506.22850v1)**

> **作者:** Aalok Gangopadhyay; Shashikant Verma; Shanmuganathan Raman
>
> **摘要:** We present Deep Mesh Denoising Network (DMD-Net), an end-to-end deep learning framework, for solving the mesh denoising problem. DMD-Net consists of a Graph Convolutional Neural Network in which aggregation is performed in both the primal as well as the dual graph. This is realized in the form of an asymmetric two-stream network, which contains a primal-dual fusion block that enables communication between the primal-stream and the dual-stream. We develop a Feature Guided Transformer (FGT) paradigm, which consists of a feature extractor, a transformer, and a denoiser. The feature extractor estimates the local features, that guide the transformer to compute a transformation, which is applied to the noisy input mesh to obtain a useful intermediate representation. This is further processed by the denoiser to obtain the denoised mesh. Our network is trained on a large scale dataset of 3D objects. We perform exhaustive ablation studies to demonstrate that each component in our network is essential for obtaining the best performance. We show that our method obtains competitive or better results when compared with the state-of-the-art mesh denoising algorithms. We demonstrate that our method is robust to various kinds of noise. We observe that even in the presence of extremely high noise, our method achieves excellent performance.
>
---
#### [new 055] Can We Challenge Open-Vocabulary Object Detectors with Generated Content in Street Scenes?
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在挑战开放词汇检测器的局限性。通过生成数据测试模型，发现其对物体位置的依赖性，为提升模型提供依据。**

- **链接: [http://arxiv.org/pdf/2506.23751v1](http://arxiv.org/pdf/2506.23751v1)**

> **作者:** Annika Mütze; Sadia Ilyas; Christian Dörpelkus; Matthias Rottmann
>
> **摘要:** Open-vocabulary object detectors such as Grounding DINO are trained on vast and diverse data, achieving remarkable performance on challenging datasets. Due to that, it is unclear where to find their limitations, which is of major concern when using in safety-critical applications. Real-world data does not provide sufficient control, required for a rigorous evaluation of model generalization. In contrast, synthetically generated data allows to systematically explore the boundaries of model competence/generalization. In this work, we address two research questions: 1) Can we challenge open-vocabulary object detectors with generated image content? 2) Can we find systematic failure modes of those models? To address these questions, we design two automated pipelines using stable diffusion to inpaint unusual objects with high diversity in semantics, by sampling multiple substantives from WordNet and ChatGPT. On the synthetically generated data, we evaluate and compare multiple open-vocabulary object detectors as well as a classical object detector. The synthetic data is derived from two real-world datasets, namely LostAndFound, a challenging out-of-distribution (OOD) detection benchmark, and the NuImages dataset. Our results indicate that inpainting can challenge open-vocabulary object detectors in terms of overlooking objects. Additionally, we find a strong dependence of open-vocabulary models on object location, rather than on object semantics. This provides a systematic approach to challenge open-vocabulary models and gives valuable insights on how data could be acquired to effectively improve these models.
>
---
#### [new 056] AlignCVC: Aligning Cross-View Consistency for Single-Image-to-3D Generation
- **分类: cs.CV**

- **简介: 该论文属于单图像到3D生成任务，解决多视角一致性不足的问题。通过分布对齐提升生成质量与效率。**

- **链接: [http://arxiv.org/pdf/2506.23150v1](http://arxiv.org/pdf/2506.23150v1)**

> **作者:** Xinyue Liang; Zhiyuan Ma; Lingchen Sun; Yanjun Guo; Lei Zhang
>
> **摘要:** Single-image-to-3D models typically follow a sequential generation and reconstruction workflow. However, intermediate multi-view images synthesized by pre-trained generation models often lack cross-view consistency (CVC), significantly degrading 3D reconstruction performance. While recent methods attempt to refine CVC by feeding reconstruction results back into the multi-view generator, these approaches struggle with noisy and unstable reconstruction outputs that limit effective CVC improvement. We introduce AlignCVC, a novel framework that fundamentally re-frames single-image-to-3D generation through distribution alignment rather than relying on strict regression losses. Our key insight is to align both generated and reconstructed multi-view distributions toward the ground-truth multi-view distribution, establishing a principled foundation for improved CVC. Observing that generated images exhibit weak CVC while reconstructed images display strong CVC due to explicit rendering, we propose a soft-hard alignment strategy with distinct objectives for generation and reconstruction models. This approach not only enhances generation quality but also dramatically accelerates inference to as few as 4 steps. As a plug-and-play paradigm, our method, namely AlignCVC, seamlessly integrates various multi-view generation models with 3D reconstruction models. Extensive experiments demonstrate the effectiveness and efficiency of AlignCVC for single-image-to-3D generation.
>
---
#### [new 057] STR-Match: Matching SpatioTemporal Relevance Score for Training-Free Video Editing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频编辑任务，解决时间不一致和运动失真问题。提出STR-Match算法，通过新型STR分数实现无训练的视频编辑。**

- **链接: [http://arxiv.org/pdf/2506.22868v1](http://arxiv.org/pdf/2506.22868v1)**

> **作者:** Junsung Lee; Junoh Kang; Bohyung Han
>
> **备注:** 15 pages, 9 figures, 3 tables
>
> **摘要:** Previous text-guided video editing methods often suffer from temporal inconsistency, motion distortion, and-most notably-limited domain transformation. We attribute these limitations to insufficient modeling of spatiotemporal pixel relevance during the editing process. To address this, we propose STR-Match, a training-free video editing algorithm that produces visually appealing and spatiotemporally coherent videos through latent optimization guided by our novel STR score. The score captures spatiotemporal pixel relevance across adjacent frames by leveraging 2D spatial attention and 1D temporal modules in text-to-video (T2V) diffusion models, without the overhead of computationally expensive 3D attention mechanisms. Integrated into a latent optimization framework with a latent mask, STR-Match generates temporally consistent and visually faithful videos, maintaining strong performance even under significant domain transformations while preserving key visual attributes of the source. Extensive experiments demonstrate that STR-Match consistently outperforms existing methods in both visual quality and spatiotemporal consistency.
>
---
#### [new 058] Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model
- **分类: cs.CV**

- **简介: 该论文属于零样本学习任务，旨在解决视觉语言模型缺乏可解释性的问题。通过局部对齐机制提升模型的可解释性与准确性。**

- **链接: [http://arxiv.org/pdf/2506.23822v1](http://arxiv.org/pdf/2506.23822v1)**

> **作者:** Shiming Chen; Bowen Duan; Salman Khan; Fahad Shahbaz Khan
>
> **备注:** Accepted to ICCV'25
>
> **摘要:** Large-scale vision-language models (VLMs), such as CLIP, have achieved remarkable success in zero-shot learning (ZSL) by leveraging large-scale visual-text pair datasets. However, these methods often lack interpretability, as they compute the similarity between an entire query image and the embedded category words, making it difficult to explain their predictions. One approach to address this issue is to develop interpretable models by integrating language, where classifiers are built using discrete attributes, similar to human perception. This introduces a new challenge: how to effectively align local visual features with corresponding attributes based on pre-trained VLMs. To tackle this, we propose LaZSL, a locally-aligned vision-language model for interpretable ZSL. LaZSL employs local visual-semantic alignment via optimal transport to perform interaction between visual regions and their associated attributes, facilitating effective alignment and providing interpretable similarity without the need for additional training. Extensive experiments demonstrate that our method offers several advantages, including enhanced interpretability, improved accuracy, and strong domain generalization. Codes available at: https://github.com/shiming-chen/LaZSL.
>
---
#### [new 059] On the Domain Robustness of Contrastive Vision-Language Models
- **分类: cs.CV; cs.LG; I.4**

- **简介: 该论文属于视觉语言模型领域，解决模型在特定领域下的鲁棒性问题。提出Deepbench框架，通过生成真实图像干扰评估模型性能。**

- **链接: [http://arxiv.org/pdf/2506.23663v1](http://arxiv.org/pdf/2506.23663v1)**

> **作者:** Mario Koddenbrock; Rudolf Hoffmann; David Brodmann; Erik Rodner
>
> **备注:** Deepbench is available at https://github.com/ml-lab-htw/deepbench
>
> **摘要:** In real-world vision-language applications, practitioners increasingly rely on large, pretrained foundation models rather than custom-built solutions, despite limited transparency regarding their training data and processes. While these models achieve impressive performance on general benchmarks, their effectiveness can decline notably under specialized domain shifts, such as unique imaging conditions or environmental variations. In this work, we introduce Deepbench, a framework designed to assess domain-specific robustness of vision-language models (VLMs). Deepbench leverages a large language model (LLM) to generate realistic, context-aware image corruptions tailored to specific deployment domains without requiring labeled data. We evaluate a range of contrastive vision-language architectures and architectural variants across six real-world domains and observe substantial variability in robustness, highlighting the need for targeted, domain-aware evaluation. Deepbench is released as open-source software to support further research into domain-aware robustness assessment.
>
---
#### [new 060] Transformer-Based Person Search with High-Frequency Augmentation and Multi-Wave Mixing
- **分类: cs.CV**

- **简介: 该论文属于人像检索任务，旨在解决Transformer模型中高频特征抑制和计算成本高的问题。提出HAMW方法，通过增强高频特征和多尺度融合提升性能。**

- **链接: [http://arxiv.org/pdf/2506.23202v1](http://arxiv.org/pdf/2506.23202v1)**

> **作者:** Qilin Shu; Qixian Zhang; Qi Zhang; Hongyun Zhang; Duoqian Miao; Cairong Zhao
>
> **摘要:** The person search task aims to locate a target person within a set of scene images. In recent years, transformer-based models in this field have made some progress. However, they still face three primary challenges: 1) the self-attention mechanism tends to suppress high-frequency components in the features, which severely impacts model performance; 2) the computational cost of transformers is relatively high. To address these issues, we propose a novel High-frequency Augmentation and Multi-Wave mixing (HAMW) method for person search. HAMW is designed to enhance the discriminative feature extraction capabilities of transformers while reducing computational overhead and improving efficiency. Specifically, we develop a three-stage framework that progressively optimizes both detection and re-identification performance. Our model enhances the perception of high-frequency features by learning from augmented inputs containing additional high-frequency components. Furthermore, we replace the self-attention layers in the transformer with a strategy based on multi-level Haar wavelet fusion to capture multi-scale features. This not only lowers the computational complexity but also alleviates the suppression of high-frequency features and enhances the ability to exploit multi-scale information. Extensive experiments demonstrate that HAMW achieves state-of-the-art performance on both the CUHK-SYSU and PRW datasets.
>
---
#### [new 061] FreqDGT: Frequency-Adaptive Dynamic Graph Networks with Transformer for Cross-subject EEG Emotion Recognition
- **分类: cs.CV**

- **简介: 该论文属于跨被试EEG情绪识别任务，旨在解决个体差异导致的泛化能力不足问题。提出FreqDGT模型，结合频率自适应、动态图学习和时序分解网络提升识别准确率。**

- **链接: [http://arxiv.org/pdf/2506.22807v1](http://arxiv.org/pdf/2506.22807v1)**

> **作者:** Yueyang Li; Shengyu Gong; Weiming Zeng; Nizhuan Wang; Wai Ting Siok
>
> **摘要:** Electroencephalography (EEG) serves as a reliable and objective signal for emotion recognition in affective brain-computer interfaces, offering unique advantages through its high temporal resolution and ability to capture authentic emotional states that cannot be consciously controlled. However, cross-subject generalization remains a fundamental challenge due to individual variability, cognitive traits, and emotional responses. We propose FreqDGT, a frequency-adaptive dynamic graph transformer that systematically addresses these limitations through an integrated framework. FreqDGT introduces frequency-adaptive processing (FAP) to dynamically weight emotion-relevant frequency bands based on neuroscientific evidence, employs adaptive dynamic graph learning (ADGL) to learn input-specific brain connectivity patterns, and implements multi-scale temporal disentanglement network (MTDN) that combines hierarchical temporal transformers with adversarial feature disentanglement to capture both temporal dynamics and ensure cross-subject robustness. Comprehensive experiments demonstrate that FreqDGT significantly improves cross-subject emotion recognition accuracy, confirming the effectiveness of integrating frequency-adaptive, spatial-dynamic, and temporal-hierarchical modeling while ensuring robustness to individual differences. The code is available at https://github.com/NZWANG/FreqDGT.
>
---
#### [new 062] Intervening in Black Box: Concept Bottleneck Model for Enhancing Human Neural Network Mutual Understanding
- **分类: cs.CV; cs.HC; cs.LG**

- **简介: 该论文属于模型可解释性任务，旨在提升黑盒模型的可理解性。通过概念瓶颈模型识别并修正有害概念，增强人机互信。**

- **链接: [http://arxiv.org/pdf/2506.22803v1](http://arxiv.org/pdf/2506.22803v1)**

> **作者:** Nuoye Xiong; Anqi Dong; Ning Wang; Cong Hua; Guangming Zhu; Mei Lin; Peiyi Shen; Liang Zhang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Recent advances in deep learning have led to increasingly complex models with deeper layers and more parameters, reducing interpretability and making their decisions harder to understand. While many methods explain black-box reasoning, most lack effective interventions or only operate at sample-level without modifying the model itself. To address this, we propose the Concept Bottleneck Model for Enhancing Human-Neural Network Mutual Understanding (CBM-HNMU). CBM-HNMU leverages the Concept Bottleneck Model (CBM) as an interpretable framework to approximate black-box reasoning and communicate conceptual understanding. Detrimental concepts are automatically identified and refined (removed/replaced) based on global gradient contributions. The modified CBM then distills corrected knowledge back into the black-box model, enhancing both interpretability and accuracy. We evaluate CBM-HNMU on various CNN and transformer-based models across Flower-102, CIFAR-10, CIFAR-100, FGVC-Aircraft, and CUB-200, achieving a maximum accuracy improvement of 2.64% and a maximum increase in average accuracy across 1.03%. Source code is available at: https://github.com/XiGuaBo/CBM-HNMU.
>
---
#### [new 063] Pyramidal Patchification Flow for Visual Generation
- **分类: cs.CV**

- **简介: 该论文属于视觉生成任务，解决扩散模型中计算效率与生成质量的平衡问题。提出Pyramidal Patchification Flow方法，通过动态调整补丁大小提升推理速度。**

- **链接: [http://arxiv.org/pdf/2506.23543v1](http://arxiv.org/pdf/2506.23543v1)**

> **作者:** Hui Li; Baoyou Chen; Liwei Zhang; Jiaye Li; Jingdong Wang; Siyu Zhu
>
> **备注:** 10 pages, 9figures
>
> **摘要:** Diffusion transformers (DiTs) adopt Patchify, mapping patch representations to token representations through linear projections, to adjust the number of tokens input to DiT blocks and thus the computation cost. Instead of a single patch size for all the timesteps, we introduce a Pyramidal Patchification Flow (PPFlow) approach: Large patch sizes are used for high noise timesteps and small patch sizes for low noise timesteps; Linear projections are learned for each patch size; and Unpatchify is accordingly modified. Unlike Pyramidal Flow, our approach operates over full latent representations other than pyramid representations, and adopts the normal denoising process without requiring the renoising trick. We demonstrate the effectiveness of our approach through two training manners. Training from scratch achieves a $1.6\times$ ($2.0\times$) inference speed over SiT-B/2 for 2-level (3-level) pyramid patchification with slightly lower training FLOPs and similar image generation performance. Training from pretrained normal DiTs achieves even better performance with small training time. The code and checkpoint are at https://github.com/fudan-generative-vision/PPFlow.
>
---
#### [new 064] Why Settle for One? Text-to-ImageSet Generation and Evaluation
- **分类: cs.CV; cs.AI**

- **简介: 该论文聚焦于文本到图像集生成任务，解决多一致性要求下的图像集生成问题。提出T2IS-Bench、T2IS-Eval和AutoT2IS，提升生成质量与实用性。**

- **链接: [http://arxiv.org/pdf/2506.23275v1](http://arxiv.org/pdf/2506.23275v1)**

> **作者:** Chengyou Jia; Xin Shen; Zhuohang Dang; Zhuohang Dang; Changliang Xia; Weijia Wu; Xinyu Zhang; Hangwei Qian; Ivor W. Tsang; Minnan Luo
>
> **摘要:** Despite remarkable progress in Text-to-Image models, many real-world applications require generating coherent image sets with diverse consistency requirements. Existing consistent methods often focus on a specific domain with specific aspects of consistency, which significantly constrains their generalizability to broader applications. In this paper, we propose a more challenging problem, Text-to-ImageSet (T2IS) generation, which aims to generate sets of images that meet various consistency requirements based on user instructions. To systematically study this problem, we first introduce $\textbf{T2IS-Bench}$ with 596 diverse instructions across 26 subcategories, providing comprehensive coverage for T2IS generation. Building on this, we propose $\textbf{T2IS-Eval}$, an evaluation framework that transforms user instructions into multifaceted assessment criteria and employs effective evaluators to adaptively assess consistency fulfillment between criteria and generated sets. Subsequently, we propose $\textbf{AutoT2IS}$, a training-free framework that maximally leverages pretrained Diffusion Transformers' in-context capabilities to harmonize visual elements to satisfy both image-level prompt alignment and set-level visual consistency. Extensive experiments on T2IS-Bench reveal that diverse consistency challenges all existing methods, while our AutoT2IS significantly outperforms current generalized and even specialized approaches. Our method also demonstrates the ability to enable numerous underexplored real-world applications, confirming its substantial practical value. Visit our project in https://chengyou-jia.github.io/T2IS-Home.
>
---
#### [new 065] AI-Generated Lecture Slides for Improving Slide Element Detection and Retrieval
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于讲座幻灯片理解任务，旨在解决标注数据不足的问题。通过生成合成幻灯片并进行少样本迁移学习，提升元素检测与检索性能。**

- **链接: [http://arxiv.org/pdf/2506.23605v1](http://arxiv.org/pdf/2506.23605v1)**

> **作者:** Suyash Maniyar; Vishvesh Trivedi; Ajoy Mondal; Anand Mishra; C. V. Jawahar
>
> **备注:** 40 pages including supplementary, accepted at ICDAR 2025
>
> **摘要:** Lecture slide element detection and retrieval are key problems in slide understanding. Training effective models for these tasks often depends on extensive manual annotation. However, annotating large volumes of lecture slides for supervised training is labor intensive and requires domain expertise. To address this, we propose a large language model (LLM)-guided synthetic lecture slide generation pipeline, SynLecSlideGen, which produces high-quality, coherent and realistic slides. We also create an evaluation benchmark, namely RealSlide by manually annotating 1,050 real lecture slides. To assess the utility of our synthetic slides, we perform few-shot transfer learning on real data using models pre-trained on them. Experimental results show that few-shot transfer learning with pretraining on synthetic slides significantly improves performance compared to training only on real data. This demonstrates that synthetic data can effectively compensate for limited labeled lecture slides. The code and resources of our work are publicly available on our project website: https://synslidegen.github.io/.
>
---
#### [new 066] VSRM: A Robust Mamba-Based Framework for Video Super-Resolution
- **分类: cs.CV**

- **简介: 该论文属于视频超分辨率任务，旨在解决长序列建模和特征对齐问题。提出VSRM框架，结合Mamba结构提升性能。**

- **链接: [http://arxiv.org/pdf/2506.22762v1](http://arxiv.org/pdf/2506.22762v1)**

> **作者:** Dinh Phu Tran; Dao Duy Hung; Daeyoung Kim
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Video super-resolution remains a major challenge in low-level vision tasks. To date, CNN- and Transformer-based methods have delivered impressive results. However, CNNs are limited by local receptive fields, while Transformers struggle with quadratic complexity, posing challenges for processing long sequences in VSR. Recently, Mamba has drawn attention for its long-sequence modeling, linear complexity, and large receptive fields. In this work, we propose VSRM, a novel \textbf{V}ideo \textbf{S}uper-\textbf{R}esolution framework that leverages the power of \textbf{M}amba. VSRM introduces Spatial-to-Temporal Mamba and Temporal-to-Spatial Mamba blocks to extract long-range spatio-temporal features and enhance receptive fields efficiently. To better align adjacent frames, we propose Deformable Cross-Mamba Alignment module. This module utilizes a deformable cross-mamba mechanism to make the compensation stage more dynamic and flexible, preventing feature distortions. Finally, we minimize the frequency domain gaps between reconstructed and ground-truth frames by proposing a simple yet effective Frequency Charbonnier-like loss that better preserves high-frequency content and enhances visual quality. Through extensive experiments, VSRM achieves state-of-the-art results on diverse benchmarks, establishing itself as a solid foundation for future research.
>
---
#### [new 067] Towards Markerless Intraoperative Tracking of Deformable Spine Tissue
- **分类: cs.CV**

- **简介: 该论文属于医学影像中的脊柱跟踪任务，旨在解决术中软组织变形跟踪问题。提出SpineAlign系统和CorrespondNet框架，实现术前术后脊柱区域的配准与分割。**

- **链接: [http://arxiv.org/pdf/2506.23657v1](http://arxiv.org/pdf/2506.23657v1)**

> **作者:** Connor Daly; Elettra Marconi; Marco Riva; Jinendra Ekanayake; Daniel S. Elson; Ferdinando Rodriguez y Baena
>
> **备注:** Preprint of paper, submitted
>
> **摘要:** Consumer-grade RGB-D imaging for intraoperative orthopedic tissue tracking is a promising method with high translational potential. Unlike bone-mounted tracking devices, markerless tracking can reduce operating time and complexity. However, its use has been limited to cadaveric studies. This paper introduces the first real-world clinical RGB-D dataset for spine surgery and develops SpineAlign, a system for capturing deformation between preoperative and intraoperative spine states. We also present an intraoperative segmentation network trained on this data and introduce CorrespondNet, a multi-task framework for predicting key regions for registration in both intraoperative and preoperative scenes.
>
---
#### [new 068] Unified Multimodal Understanding via Byte-Pair Visual Encoding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态理解任务，旨在解决视觉与文本对齐难题。通过字节对编码统一视觉与文本表示，提升模型跨模态推理能力。**

- **链接: [http://arxiv.org/pdf/2506.23639v1](http://arxiv.org/pdf/2506.23639v1)**

> **作者:** Wanpeng Zhang; Yicheng Feng; Hao Luo; Yijiang Li; Zihao Yue; Sipeng Zheng; Zongqing Lu
>
> **摘要:** Multimodal large language models (MLLMs) have made significant progress in vision-language understanding, yet effectively aligning different modalities remains a fundamental challenge. We present a framework that unifies multimodal understanding by applying byte-pair encoding to visual tokens. Unlike conventional approaches that rely on modality-specific encoders, our method directly incorporates structural information into visual tokens, mirroring successful tokenization strategies in text-only language models. We introduce a priority-guided encoding scheme that considers both frequency and spatial consistency, coupled with a multi-stage training procedure based on curriculum-driven data composition. These enhancements enable the transformer model to better capture cross-modal relationships and reason with visual information. Comprehensive experiments demonstrate improved performance across diverse vision-language tasks. By bridging the gap between visual and textual representations, our approach contributes to the advancement of more capable and efficient multimodal foundation models.
>
---
#### [new 069] ViFusionTST: Deep Fusion of Time-Series Image Representations from Load Signals for Early Bed-Exit Prediction
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于时间序列分类任务，旨在早期预测患者离床行为。通过融合负载信号生成的图像，提出ViFusionTST模型提升预测准确性。**

- **链接: [http://arxiv.org/pdf/2506.22498v1](http://arxiv.org/pdf/2506.22498v1)**

> **作者:** Hao Liu; Yu Hu; Rakiba Rayhana; Ling Bai; Zheng Liu
>
> **摘要:** Bed-related falls remain a leading source of injury in hospitals and long-term-care facilities, yet many commercial alarms trigger only after a patient has already left the bed. We show that early bed-exit intent can be predicted using only four low-cost load cells mounted under the bed legs. The resulting load signals are first converted into a compact set of complementary images: an RGB line plot that preserves raw waveforms and three texture maps - recurrence plot, Markov transition field, and Gramian angular field - that expose higher-order dynamics. We introduce ViFusionTST, a dual-stream Swin Transformer that processes the line plot and texture maps in parallel and fuses them through cross-attention to learn data-driven modality weights. To provide a realistic benchmark, we collected six months of continuous data from 95 beds in a long-term-care facility. On this real-world dataset ViFusionTST reaches an accuracy of 0.885 and an F1 score of 0.794, surpassing recent 1D and 2D time-series baselines across F1, recall, accuracy, and AUPRC. The results demonstrate that image-based fusion of load-sensor signals for time series classification is a practical and effective solution for real-time, privacy-preserving fall prevention.
>
---
#### [new 070] Metadata, Wavelet, and Time Aware Diffusion Models for Satellite Image Super Resolution
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于卫星图像超分辨率任务，旨在解决高分辨率图像获取困难的问题。提出MWT-Diff框架，结合元数据、小波和时间信息，提升重建质量。**

- **链接: [http://arxiv.org/pdf/2506.23566v1](http://arxiv.org/pdf/2506.23566v1)**

> **作者:** Luigi Sigillo; Renato Giamba; Danilo Comminiello
>
> **备注:** ICLR 2025 Workshop on Machine Learning for Remote Sensing (ML4RS)
>
> **摘要:** The acquisition of high-resolution satellite imagery is often constrained by the spatial and temporal limitations of satellite sensors, as well as the high costs associated with frequent observations. These challenges hinder applications such as environmental monitoring, disaster response, and agricultural management, which require fine-grained and high-resolution data. In this paper, we propose MWT-Diff, an innovative framework for satellite image super-resolution (SR) that combines latent diffusion models with wavelet transforms to address these challenges. At the core of the framework is a novel metadata-, wavelet-, and time-aware encoder (MWT-Encoder), which generates embeddings that capture metadata attributes, multi-scale frequency information, and temporal relationships. The embedded feature representations steer the hierarchical diffusion dynamics, through which the model progressively reconstructs high-resolution satellite imagery from low-resolution inputs. This process preserves critical spatial characteristics including textural patterns, boundary discontinuities, and high-frequency spectral components essential for detailed remote sensing analysis. The comparative analysis of MWT-Diff across multiple datasets demonstrated favorable performance compared to recent approaches, as measured by standard perceptual quality metrics including FID and LPIPS.
>
---
#### [new 071] DenseWorld-1M: Towards Detailed Dense Grounded Caption in the Real World
- **分类: cs.CV**

- **简介: 该论文提出DenseWorld-1M数据集，解决真实世界中密集实体描述不足的问题。通过三阶段标注流程和两个VLM模型，生成详细且具有空间关系的图像描述。**

- **链接: [http://arxiv.org/pdf/2506.24102v1](http://arxiv.org/pdf/2506.24102v1)**

> **作者:** Xiangtai Li; Tao Zhang; Yanwei Li; Haobo Yuan; Shihao Chen; Yikang Zhou; Jiahao Meng; Yueyi Sun; Shilin Xu; Lu Qi; Tianheng Cheng; Yi Lin; Zilong Huang; Wenhao Huang; Jiashi Feng; Guang Shi
>
> **备注:** Datasets and Models: https://github.com/lxtGH/DenseWorld-1M
>
> **摘要:** Multimodal Large Language Models (MLLMs) demonstrate a complex understanding of scenes, benefiting from large-scale and high-quality datasets. Most existing caption datasets lack the ground locations and relations for visual entities. Several grounded caption datasets face the problems of missing detailed descriptions, relations, and massive object descriptions on high-resolution images. To fill this gap for the community, we present DenseWorld-1M, the first massive, detailed, dense grounded caption dataset in the real world. We design a three-stage labeling pipeline, containing open-world perception, detailed object caption generation, and dense caption merging. The first stage obtains entity-level masks and labels. The second stage generates the object-level, detailed captions with the guidance of masks and labels from the first stage. The final stage merges object captions and masks into spatial and relational dense captions. To accelerate the labeling process and improve caption quality, we present two VLM models: the Detailed Region Caption model and the Spatial Caption Merging model. Extensive experiments on various settings, including vision-language understanding, visual grounding, and region caption generation, demonstrate the effectiveness of our DenseWorld-1M dataset and labeling models.
>
---
#### [new 072] WAVE: Warp-Based View Guidance for Consistent Novel View Synthesis Using a Single Image
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决单图生成一致新视角的问题。通过视图引导的变形技术增强扩散模型，提升视图一致性。**

- **链接: [http://arxiv.org/pdf/2506.23518v1](http://arxiv.org/pdf/2506.23518v1)**

> **作者:** Jiwoo Park; Tae Eun Choi; Youngjun Jun; Seong Jae Hwang
>
> **摘要:** Generating high-quality novel views of a scene from a single image requires maintaining structural coherence across different views, referred to as view consistency. While diffusion models have driven advancements in novel view synthesis, they still struggle to preserve spatial continuity across views. Diffusion models have been combined with 3D models to address the issue, but such approaches lack efficiency due to their complex multi-step pipelines. This paper proposes a novel view-consistent image generation method which utilizes diffusion models without additional modules. Our key idea is to enhance diffusion models with a training-free method that enables adaptive attention manipulation and noise reinitialization by leveraging view-guided warping to ensure view consistency. Through our comprehensive metric framework suitable for novel-view datasets, we show that our method improves view consistency across various diffusion models, demonstrating its broader applicability.
>
---
#### [new 073] MILo: Mesh-In-the-Loop Gaussian Splatting for Detailed and Efficient Surface Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决从图像中高效重建精细表面网格的问题。提出MILo框架，通过可微分方式直接从高斯分布生成网格，提升效率与细节保留。**

- **链接: [http://arxiv.org/pdf/2506.24096v1](http://arxiv.org/pdf/2506.24096v1)**

> **作者:** Antoine Guédon; Diego Gomez; Nissim Maruani; Bingchen Gong; George Drettakis; Maks Ovsjanikov
>
> **备注:** 10 pages. A presentation video of our approach is available at https://youtu.be/_SGNhhNz0fE
>
> **摘要:** While recent advances in Gaussian Splatting have enabled fast reconstruction of high-quality 3D scenes from images, extracting accurate surface meshes remains a challenge. Current approaches extract the surface through costly post-processing steps, resulting in the loss of fine geometric details or requiring significant time and leading to very dense meshes with millions of vertices. More fundamentally, the a posteriori conversion from a volumetric to a surface representation limits the ability of the final mesh to preserve all geometric structures captured during training. We present MILo, a novel Gaussian Splatting framework that bridges the gap between volumetric and surface representations by differentiably extracting a mesh from the 3D Gaussians. We design a fully differentiable procedure that constructs the mesh-including both vertex locations and connectivity-at every iteration directly from the parameters of the Gaussians, which are the only quantities optimized during training. Our method introduces three key technical contributions: a bidirectional consistency framework ensuring both representations-Gaussians and the extracted mesh-capture the same underlying geometry during training; an adaptive mesh extraction process performed at each training iteration, which uses Gaussians as differentiable pivots for Delaunay triangulation; a novel method for computing signed distance values from the 3D Gaussians that enables precise surface extraction while avoiding geometric erosion. Our approach can reconstruct complete scenes, including backgrounds, with state-of-the-art quality while requiring an order of magnitude fewer mesh vertices than previous methods. Due to their light weight and empty interior, our meshes are well suited for downstream applications such as physics simulations or animation.
>
---
#### [new 074] Puzzles: Unbounded Video-Depth Augmentation for Scalable End-to-End 3D Reconstruction
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决数据不足问题。通过Puzzles方法生成大量高质量视频深度数据，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.23863v1](http://arxiv.org/pdf/2506.23863v1)**

> **作者:** Jiahao Ma; Lei Wang; Miaomiao liu; David Ahmedt-Aristizabal; Chuong Nguyen
>
> **备注:** Feed-forward 3D reconstruction, Data Augmentation
>
> **摘要:** Multi-view 3D reconstruction remains a core challenge in computer vision. Recent methods, such as DUST3R and its successors, directly regress pointmaps from image pairs without relying on known scene geometry or camera parameters. However, the performance of these models is constrained by the diversity and scale of available training data. In this work, we introduce Puzzles, a data augmentation strategy that synthesizes an unbounded volume of high-quality posed video-depth data from a single image or video clip. By simulating diverse camera trajectories and realistic scene geometry through targeted image transformations, Puzzles significantly enhances data variety. Extensive experiments show that integrating Puzzles into existing video-based 3D reconstruction pipelines consistently boosts performance without modifying the underlying network architecture. Notably, models trained on only ten percent of the original data augmented with Puzzles still achieve accuracy comparable to those trained on the full dataset. Code is available at https://jiahao-ma.github.io/puzzles/.
>
---
#### [new 075] SG-LDM: Semantic-Guided LiDAR Generation via Latent-Aligned Diffusion
- **分类: cs.CV**

- **简介: 该论文属于激光雷达点云生成任务，解决真实数据不足问题。提出SG-LDM模型，实现语义引导的高质量点云生成，并构建翻译框架提升下游任务性能。**

- **链接: [http://arxiv.org/pdf/2506.23606v1](http://arxiv.org/pdf/2506.23606v1)**

> **作者:** Zhengkang Xiang; Zizhao Li; Amir Khodabandeh; Kourosh Khoshelham
>
> **摘要:** Lidar point cloud synthesis based on generative models offers a promising solution to augment deep learning pipelines, particularly when real-world data is scarce or lacks diversity. By enabling flexible object manipulation, this synthesis approach can significantly enrich training datasets and enhance discriminative models. However, existing methods focus on unconditional lidar point cloud generation, overlooking their potential for real-world applications. In this paper, we propose SG-LDM, a Semantic-Guided Lidar Diffusion Model that employs latent alignment to enable robust semantic-to-lidar synthesis. By directly operating in the native lidar space and leveraging explicit semantic conditioning, SG-LDM achieves state-of-the-art performance in generating high-fidelity lidar point clouds guided by semantic labels. Moreover, we propose the first diffusion-based lidar translation framework based on SG-LDM, which enables cross-domain translation as a domain adaptation strategy to enhance downstream perception performance. Systematic experiments demonstrate that SG-LDM significantly outperforms existing lidar diffusion models and the proposed lidar translation framework further improves data augmentation performance in the downstream lidar segmentation task.
>
---
#### [new 076] GeoProg3D: Compositional Visual Reasoning for City-Scale 3D Language Fields
- **分类: cs.CV**

- **简介: 该论文提出GeoProg3D，解决城市级3D场景的自然语言交互问题，通过组合地理信息与视觉工具实现高效推理。**

- **链接: [http://arxiv.org/pdf/2506.23352v1](http://arxiv.org/pdf/2506.23352v1)**

> **作者:** Shunsuke Yasuki; Taiki Miyanishi; Nakamasa Inoue; Shuhei Kurita; Koya Sakamoto; Daichi Azuma; Masato Taki; Yutaka Matsuo
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** The advancement of 3D language fields has enabled intuitive interactions with 3D scenes via natural language. However, existing approaches are typically limited to small-scale environments, lacking the scalability and compositional reasoning capabilities necessary for large, complex urban settings. To overcome these limitations, we propose GeoProg3D, a visual programming framework that enables natural language-driven interactions with city-scale high-fidelity 3D scenes. GeoProg3D consists of two key components: (i) a Geography-aware City-scale 3D Language Field (GCLF) that leverages a memory-efficient hierarchical 3D model to handle large-scale data, integrated with geographic information for efficiently filtering vast urban spaces using directional cues, distance measurements, elevation data, and landmark references; and (ii) Geographical Vision APIs (GV-APIs), specialized geographic vision tools such as area segmentation and object detection. Our framework employs large language models (LLMs) as reasoning engines to dynamically combine GV-APIs and operate GCLF, effectively supporting diverse geographic vision tasks. To assess performance in city-scale reasoning, we introduce GeoEval3D, a comprehensive benchmark dataset containing 952 query-answer pairs across five challenging tasks: grounding, spatial reasoning, comparison, counting, and measurement. Experiments demonstrate that GeoProg3D significantly outperforms existing 3D language fields and vision-language models across multiple tasks. To our knowledge, GeoProg3D is the first framework enabling compositional geographic reasoning in high-fidelity city-scale 3D environments via natural language. The code is available at https://snskysk.github.io/GeoProg3D/.
>
---
#### [new 077] How to Design and Train Your Implicit Neural Representation for Video Compression
- **分类: cs.CV**

- **简介: 该论文属于视频压缩任务，解决INR编码速度慢的问题。通过设计RNeRV模型和引入超网络提升效率与质量。**

- **链接: [http://arxiv.org/pdf/2506.24127v1](http://arxiv.org/pdf/2506.24127v1)**

> **作者:** Matthew Gwilliam; Roy Zhang; Namitha Padmanabhan; Hongyang Du; Abhinav Shrivastava
>
> **备注:** 21 pages, 41 figures, 5 tables
>
> **摘要:** Implicit neural representation (INR) methods for video compression have recently achieved visual quality and compression ratios that are competitive with traditional pipelines. However, due to the need for per-sample network training, the encoding speeds of these methods are too slow for practical adoption. We develop a library to allow us to disentangle and review the components of methods from the NeRV family, reframing their performance in terms of not only size-quality trade-offs, but also impacts on training time. We uncover principles for effective video INR design and propose a state-of-the-art configuration of these components, Rabbit NeRV (RNeRV). When all methods are given equal training time (equivalent to 300 NeRV epochs) for 7 different UVG videos at 1080p, RNeRV achieves +1.27% PSNR on average compared to the best-performing alternative for each video in our NeRV library. We then tackle the encoding speed issue head-on by investigating the viability of hyper-networks, which predict INR weights from video inputs, to disentangle training from encoding to allow for real-time encoding. We propose masking the weights of the predicted INR during training to allow for variable, higher quality compression, resulting in 1.7% improvements to both PSNR and MS-SSIM at 0.037 bpp on the UCF-101 dataset, and we increase hyper-network parameters by 0.4% for 2.5%/2.7% improvements to PSNR/MS-SSIM with equal bpp and similar speeds. Our project website is available at https://mgwillia.github.io/vinrb/ and our code is available at https://github.com/mgwillia/vinrb.
>
---
#### [new 078] Learning Counterfactually Decoupled Attention for Open-World Model Attribution
- **分类: cs.CV; cs.CR; cs.LG**

- **简介: 该论文属于开放世界模型归属任务，旨在解决现有方法依赖人工设计、易受干扰的问题。提出CDAL方法，通过因果解耦注意力学习，提升对未知攻击的泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.23074v1](http://arxiv.org/pdf/2506.23074v1)**

> **作者:** Yu Zheng; Boyang Gong; Fanye Kong; Yueqi Duan; Bingyao Yu; Wenzhao Zheng; Lei Chen; Jiwen Lu; Jie Zhou
>
> **备注:** Accepted by ICCV 2025. Code: \url{https://github.com/yzheng97/CDAL}
>
> **摘要:** In this paper, we propose a Counterfactually Decoupled Attention Learning (CDAL) method for open-world model attribution. Existing methods rely on handcrafted design of region partitioning or feature space, which could be confounded by the spurious statistical correlations and struggle with novel attacks in open-world scenarios. To address this, CDAL explicitly models the causal relationships between the attentional visual traces and source model attribution, and counterfactually decouples the discriminative model-specific artifacts from confounding source biases for comparison. In this way, the resulting causal effect provides a quantification on the quality of learned attention maps, thus encouraging the network to capture essential generation patterns that generalize to unseen source models by maximizing the effect. Extensive experiments on existing open-world model attribution benchmarks show that with minimal computational overhead, our method consistently improves state-of-the-art models by large margins, particularly for unseen novel attacks. Source code: https://github.com/yzheng97/CDAL.
>
---
#### [new 079] CoreMark: Toward Robust and Universal Text Watermarking Technique
- **分类: cs.CV; cs.CR; cs.MM**

- **简介: 该论文属于文本水印任务，旨在解决水印的鲁棒性、通用性和隐蔽性问题。提出CoreMark框架，通过动态提取核心像素段嵌入数据，提升抗攻击能力并保持视觉质量。**

- **链接: [http://arxiv.org/pdf/2506.23066v1](http://arxiv.org/pdf/2506.23066v1)**

> **作者:** Jiale Meng; Yiming Li; Zheming Lu; Zewei He; Hao Luo; Tianwei Zhang
>
> **备注:** 10 pages, 16 figures
>
> **摘要:** Text watermarking schemes have gained considerable attention in recent years, yet still face critical challenges in achieving simultaneous robustness, generalizability, and imperceptibility. This paper introduces a new embedding paradigm,termed CORE, which comprises several consecutively aligned black pixel segments. Its key innovation lies in its inherent noise resistance during transmission and broad applicability across languages and fonts. Based on the CORE, we present a text watermarking framework named CoreMark. Specifically, CoreMark first dynamically extracts COREs from characters. Then, the characters with stronger robustness are selected according to the lengths of COREs. By modifying the thickness of the CORE, the hidden data is embedded into the selected characters without causing significant visual distortions. Moreover, a general plug-and-play embedding strength modulator is proposed, which can adaptively enhance the robustness for small font sizes by adjusting the embedding strength according to the font size. Experimental evaluation indicates that CoreMark demonstrates outstanding generalizability across multiple languages and fonts. Compared to existing methods, CoreMark achieves significant improvements in resisting screenshot, print-scan, and print camera attacks, while maintaining satisfactory imperceptibility.
>
---
#### [new 080] Deterministic Object Pose Confidence Region Estimation
- **分类: cs.CV**

- **简介: 该论文属于6D姿态估计任务，解决姿态置信区域估计问题。提出一种确定性方法，提高效率并减小置信区域体积。**

- **链接: [http://arxiv.org/pdf/2506.22720v1](http://arxiv.org/pdf/2506.22720v1)**

> **作者:** Jinghao Wang; Zhang Li; Zi Wang; Banglei Guan; Yang Shang; Qifeng Yu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** 6D pose confidence region estimation has emerged as a critical direction, aiming to perform uncertainty quantification for assessing the reliability of estimated poses. However, current sampling-based approach suffers from critical limitations that severely impede their practical deployment: 1) the sampling speed significantly decreases as the number of samples increases. 2) the derived confidence regions are often excessively large. To address these challenges, we propose a deterministic and efficient method for estimating pose confidence regions. Our approach uses inductive conformal prediction to calibrate the deterministically regressed Gaussian keypoint distributions into 2D keypoint confidence regions. We then leverage the implicit function theorem to propagate these keypoint confidence regions directly into 6D pose confidence regions. This method avoids the inefficiency and inflated region sizes associated with sampling and ensembling. It provides compact confidence regions that cover the ground-truth poses with a user-defined confidence level. Experimental results on the LineMOD Occlusion and SPEED datasets show that our method achieves higher pose estimation accuracy with reduced computational time. For the same coverage rate, our method yields significantly smaller confidence region volumes, reducing them by up to 99.9\% for rotations and 99.8\% for translations. The code will be available soon.
>
---
#### [new 081] Dual Atrous Separable Convolution for Improving Agricultural Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于农业图像语义分割任务，旨在提升农田异常识别的准确性。通过引入DAS卷积模块和优化跳跃连接，提高模型效率与性能。**

- **链接: [http://arxiv.org/pdf/2506.22570v1](http://arxiv.org/pdf/2506.22570v1)**

> **作者:** Chee Mei Ling; Thangarajah Akilan; Aparna Ravinda Phalke
>
> **备注:** 17 pages, 7 figures, 6 tables
>
> **摘要:** Agricultural image semantic segmentation is a pivotal component of modern agriculture, facilitating accurate visual data analysis to improve crop management, optimize resource utilization, and boost overall productivity. This study proposes an efficient image segmentation method for precision agriculture, focusing on accurately delineating farmland anomalies to support informed decision-making and proactive interventions. A novel Dual Atrous Separable Convolution (DAS Conv) module is integrated within the DeepLabV3-based segmentation framework. The DAS Conv module is meticulously designed to achieve an optimal balance between dilation rates and padding size, thereby enhancing model performance without compromising efficiency. The study also incorporates a strategic skip connection from an optimal stage in the encoder to the decoder to bolster the model's capacity to capture fine-grained spatial features. Despite its lower computational complexity, the proposed model outperforms its baseline and achieves performance comparable to highly complex transformer-based state-of-the-art (SOTA) models on the Agriculture Vision benchmark dataset. It achieves more than 66% improvement in efficiency when considering the trade-off between model complexity and performance, compared to the SOTA model. This study highlights an efficient and effective solution for improving semantic segmentation in remote sensing applications, offering a computationally lightweight model capable of high-quality performance in agricultural imagery.
>
---
#### [new 082] Ella: Embodied Social Agents with Lifelong Memory
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出Ella，一个具备终身记忆的具身社交智能体，解决开放世界中的持续学习与社会交互问题。通过结构化多模态记忆系统，实现知识积累与自主演化。**

- **链接: [http://arxiv.org/pdf/2506.24019v1](http://arxiv.org/pdf/2506.24019v1)**

> **作者:** Hongxin Zhang; Zheyuan Zhang; Zeyuan Wang; Zunzhe Zhang; Lixing Fang; Qinhong Zhou; Chuang Gan
>
> **摘要:** We introduce Ella, an embodied social agent capable of lifelong learning within a community in a 3D open world, where agents accumulate experiences and acquire knowledge through everyday visual observations and social interactions. At the core of Ella's capabilities is a structured, long-term multimodal memory system that stores, updates, and retrieves information effectively. It consists of a name-centric semantic memory for organizing acquired knowledge and a spatiotemporal episodic memory for capturing multimodal experiences. By integrating this lifelong memory system with foundation models, Ella retrieves relevant information for decision-making, plans daily activities, builds social relationships, and evolves autonomously while coexisting with other intelligent beings in the open world. We conduct capability-oriented evaluations in a dynamic 3D open world where 15 agents engage in social activities for days and are assessed with a suite of unseen controlled evaluations. Experimental results show that Ella can influence, lead, and cooperate with other agents well to achieve goals, showcasing its ability to learn effectively through observation and social interaction. Our findings highlight the transformative potential of combining structured memory systems with foundation models for advancing embodied intelligence. More videos can be found at https://umass-embodied-agi.github.io/Ella/.
>
---
#### [new 083] Visual Textualization for Image Prompted Object Detection
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，解决OVLM在罕见类别检测中的不足。通过视觉文本化方法，将视觉样本映射到文本空间，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.23785v1](http://arxiv.org/pdf/2506.23785v1)**

> **作者:** Yongjian Wu; Yang Zhou; Jiya Saiyin; Bingzheng Wei; Yan Xu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** We propose VisTex-OVLM, a novel image prompted object detection method that introduces visual textualization -- a process that projects a few visual exemplars into the text feature space to enhance Object-level Vision-Language Models' (OVLMs) capability in detecting rare categories that are difficult to describe textually and nearly absent from their pre-training data, while preserving their pre-trained object-text alignment. Specifically, VisTex-OVLM leverages multi-scale textualizing blocks and a multi-stage fusion strategy to integrate visual information from visual exemplars, generating textualized visual tokens that effectively guide OVLMs alongside text prompts. Unlike previous methods, our method maintains the original architecture of OVLM, maintaining its generalization capabilities while enhancing performance in few-shot settings. VisTex-OVLM demonstrates superior performance across open-set datasets which have minimal overlap with OVLM's pre-training data and achieves state-of-the-art results on few-shot benchmarks PASCAL VOC and MSCOCO. The code will be released at https://github.com/WitGotFlg/VisTex-OVLM.
>
---
#### [new 084] VisualPrompter: Prompt Optimization with Visual Feedback for Text-to-Image Synthesis
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决生成图像与用户描述语义不一致的问题。提出VisualPrompter框架，通过视觉反馈优化提示词，提升图像与描述的对齐度。**

- **链接: [http://arxiv.org/pdf/2506.23138v1](http://arxiv.org/pdf/2506.23138v1)**

> **作者:** Shiyu Wu; Mingzhen Sun; Weining Wang; Yequan Wang; Jing Liu
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Since there exists a notable gap between user-provided and model-preferred prompts, generating high-quality and satisfactory images using diffusion models often requires prompt engineering to optimize user inputs. Current studies on text-to-image prompt engineering can effectively enhance the style and aesthetics of generated images. However, they often neglect the semantic alignment between generated images and user descriptions, resulting in visually appealing but content-wise unsatisfying outputs. In this work, we propose VisualPrompter, a novel training-free prompt engineering framework that refines user inputs to model-preferred sentences. In particular, VisualPrompter utilizes an automatic self-reflection module to identify the missing concepts in generated images and a target-specific prompt optimization mechanism to revise the prompts in a fine-grained manner. Extensive experiments demonstrate the effectiveness of our VisualPrompter, which achieves new state-of-the-art performance on multiple benchmarks for text-image alignment evaluation. Additionally, our framework features a plug-and-play design, making it highly adaptable to various generative models.
>
---
#### [new 085] Continual Adaptation: Environment-Conditional Parameter Generation for Object Detection in Dynamic Scenarios
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，解决动态环境中模型泛化问题。通过参数生成机制实现持续适应，提升检测器在变化环境中的性能。**

- **链接: [http://arxiv.org/pdf/2506.24063v1](http://arxiv.org/pdf/2506.24063v1)**

> **作者:** Deng Li; Aming Wu; Yang Li; Yaowei Wang; Yahong Han
>
> **摘要:** In practice, environments constantly change over time and space, posing significant challenges for object detectors trained based on a closed-set assumption, i.e., training and test data share the same distribution. To this end, continual test-time adaptation has attracted much attention, aiming to improve detectors' generalization by fine-tuning a few specific parameters, e.g., BatchNorm layers. However, based on a small number of test images, fine-tuning certain parameters may affect the representation ability of other fixed parameters, leading to performance degradation. Instead, we explore a new mechanism, i.e., converting the fine-tuning process to a specific-parameter generation. Particularly, we first design a dual-path LoRA-based domain-aware adapter that disentangles features into domain-invariant and domain-specific components, enabling efficient adaptation. Additionally, a conditional diffusion-based parameter generation mechanism is presented to synthesize the adapter's parameters based on the current environment, preventing the optimization from getting stuck in local optima. Finally, we propose a class-centered optimal transport alignment method to mitigate catastrophic forgetting. Extensive experiments conducted on various continuous domain adaptive object detection tasks demonstrate the effectiveness. Meanwhile, visualization results show that the representation extracted by the generated parameters can capture more object-related information and strengthen the generalization ability.
>
---
#### [new 086] Evaluating the Impact of Khmer Font Types on Text Recognition
- **分类: cs.CV**

- **简介: 该论文属于文本识别任务，研究Khmer字体对OCR准确率的影响，通过实验比较19种字体的识别效果，为优化OCR系统提供参考。**

- **链接: [http://arxiv.org/pdf/2506.23963v1](http://arxiv.org/pdf/2506.23963v1)**

> **作者:** Vannkinh Nom; Souhail Bakkali; Muhammad Muzzamil Luqman; Mickael Coustaty; Jean-Marc Ogier
>
> **摘要:** Text recognition is significantly influenced by font types, especially for complex scripts like Khmer. The variety of Khmer fonts, each with its unique character structure, presents challenges for optical character recognition (OCR) systems. In this study, we evaluate the impact of 19 randomly selected Khmer font types on text recognition accuracy using Pytesseract. The fonts include Angkor, Battambang, Bayon, Bokor, Chenla, Dangrek, Freehand, Kh Kompong Chhnang, Kh SN Kampongsom, Khmer, Khmer CN Stueng Songke, Khmer Savuth Pen, Metal, Moul, Odor MeanChey, Preah Vihear, Siemreap, Sithi Manuss, and iSeth First. Our comparison of OCR performance across these fonts reveals that Khmer, Odor MeanChey, Siemreap, Sithi Manuss, and Battambang achieve high accuracy, while iSeth First, Bayon, and Dangrek perform poorly. This study underscores the critical importance of font selection in optimizing Khmer text recognition and provides valuable insights for developing more robust OCR systems.
>
---
#### [new 087] Inpainting is All You Need: A Diffusion-based Augmentation Method for Semi-supervised Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决标注数据不足的问题。通过引入AugPaint框架，利用扩散模型进行图像修复生成带标签的数据对，提升分割性能。**

- **链接: [http://arxiv.org/pdf/2506.23038v1](http://arxiv.org/pdf/2506.23038v1)**

> **作者:** Xinrong Hu; Yiyu Shi
>
> **摘要:** Collecting pixel-level labels for medical datasets can be a laborious and expensive process, and enhancing segmentation performance with a scarcity of labeled data is a crucial challenge. This work introduces AugPaint, a data augmentation framework that utilizes inpainting to generate image-label pairs from limited labeled data. AugPaint leverages latent diffusion models, known for their ability to generate high-quality in-domain images with low overhead, and adapts the sampling process for the inpainting task without need for retraining. Specifically, given a pair of image and label mask, we crop the area labeled with the foreground and condition on it during reversed denoising process for every noise level. Masked background area would gradually be filled in, and all generated images are paired with the label mask. This approach ensures the accuracy of match between synthetic images and label masks, setting it apart from existing dataset generation methods. The generated images serve as valuable supervision for training downstream segmentation models, effectively addressing the challenge of limited annotations. We conducted extensive evaluations of our data augmentation method on four public medical image segmentation datasets, including CT, MRI, and skin imaging. Results across all datasets demonstrate that AugPaint outperforms state-of-the-art label-efficient methodologies, significantly improving segmentation performance.
>
---
#### [new 088] VolumetricSMPL: A Neural Volumetric Body Model for Efficient Interactions, Contacts, and Collisions
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出VolumetricSMPL，解决人体与环境交互中的效率与精度问题，通过神经体素模型提升计算效率和接触建模能力。**

- **链接: [http://arxiv.org/pdf/2506.23236v1](http://arxiv.org/pdf/2506.23236v1)**

> **作者:** Marko Mihajlovic; Siwei Zhang; Gen Li; Kaifeng Zhao; Lea Müller; Siyu Tang
>
> **备注:** [ICCV 2025] https://markomih.github.io/VolumetricSMPL
>
> **摘要:** Parametric human body models play a crucial role in computer graphics and vision, enabling applications ranging from human motion analysis to understanding human-environment interactions. Traditionally, these models use surface meshes, which pose challenges in efficiently handling interactions with other geometric entities, such as objects and scenes, typically represented as meshes or point clouds. To address this limitation, recent research has explored volumetric neural implicit body models. However, existing works are either insufficiently robust for complex human articulations or impose high computational and memory costs, limiting their widespread use. To this end, we introduce VolumetricSMPL, a neural volumetric body model that leverages Neural Blend Weights (NBW) to generate compact, yet efficient MLP decoders. Unlike prior approaches that rely on large MLPs, NBW dynamically blends a small set of learned weight matrices using predicted shape- and pose-dependent coefficients, significantly improving computational efficiency while preserving expressiveness. VolumetricSMPL outperforms prior volumetric occupancy model COAP with 10x faster inference, 6x lower GPU memory usage, enhanced accuracy, and a Signed Distance Function (SDF) for efficient and differentiable contact modeling. We demonstrate VolumetricSMPL's strengths across four challenging tasks: (1) reconstructing human-object interactions from in-the-wild images, (2) recovering human meshes in 3D scenes from egocentric views, (3) scene-constrained motion synthesis, and (4) resolving self-intersections. Our results highlight its broad applicability and significant performance and efficiency gains.
>
---
#### [new 089] MOTOR: Multimodal Optimal Transport via Grounded Retrieval in Medical Visual Question Answering
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于医学视觉问答任务，旨在解决VLM生成错误答案的问题。通过引入多模态检索与重排序方法MOTOR，提升答案的临床相关性。**

- **链接: [http://arxiv.org/pdf/2506.22900v1](http://arxiv.org/pdf/2506.22900v1)**

> **作者:** Mai A. Shaaban; Tausifa Jan Saleem; Vijay Ram Papineni; Mohammad Yaqub
>
> **摘要:** Medical visual question answering (MedVQA) plays a vital role in clinical decision-making by providing contextually rich answers to image-based queries. Although vision-language models (VLMs) are widely used for this task, they often generate factually incorrect answers. Retrieval-augmented generation addresses this challenge by providing information from external sources, but risks retrieving irrelevant context, which can degrade the reasoning capabilities of VLMs. Re-ranking retrievals, as introduced in existing approaches, enhances retrieval relevance by focusing on query-text alignment. However, these approaches neglect the visual or multimodal context, which is particularly crucial for medical diagnosis. We propose MOTOR, a novel multimodal retrieval and re-ranking approach that leverages grounded captions and optimal transport. It captures the underlying relationships between the query and the retrieved context based on textual and visual information. Consequently, our approach identifies more clinically relevant contexts to augment the VLM input. Empirical analysis and human expert evaluation demonstrate that MOTOR achieves higher accuracy on MedVQA datasets, outperforming state-of-the-art methods by an average of 6.45%. Code is available at https://github.com/BioMedIA-MBZUAI/MOTOR.
>
---
#### [new 090] Where, What, Why: Towards Explainable Driver Attention Prediction
- **分类: cs.CV**

- **简介: 该论文属于驾驶员注意力预测任务，旨在解决现有方法仅预测注视位置而忽略认知动机的问题。提出W3DA数据集和LLada框架，实现空间、语义与原因的联合预测。**

- **链接: [http://arxiv.org/pdf/2506.23088v1](http://arxiv.org/pdf/2506.23088v1)**

> **作者:** Yuchen Zhou; Jiayu Tang; Xiaoyan Xiao; Yueyao Lin; Linkai Liu; Zipeng Guo; Hao Fei; Xiaobo Xia; Chao Gou
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Modeling task-driven attention in driving is a fundamental challenge for both autonomous vehicles and cognitive science. Existing methods primarily predict where drivers look by generating spatial heatmaps, but fail to capture the cognitive motivations behind attention allocation in specific contexts, which limits deeper understanding of attention mechanisms. To bridge this gap, we introduce Explainable Driver Attention Prediction, a novel task paradigm that jointly predicts spatial attention regions (where), parses attended semantics (what), and provides cognitive reasoning for attention allocation (why). To support this, we present W3DA, the first large-scale explainable driver attention dataset. It enriches existing benchmarks with detailed semantic and causal annotations across diverse driving scenarios, including normal conditions, safety-critical situations, and traffic accidents. We further propose LLada, a Large Language model-driven framework for driver attention prediction, which unifies pixel modeling, semantic parsing, and cognitive reasoning within an end-to-end architecture. Extensive experiments demonstrate the effectiveness of LLada, exhibiting robust generalization across datasets and driving conditions. This work serves as a key step toward a deeper understanding of driver attention mechanisms, with significant implications for autonomous driving, intelligent driver training, and human-computer interaction.
>
---
#### [new 091] Thinking with Images for Multimodal Reasoning: Foundations, Methods, and Future Frontiers
- **分类: cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决视觉与语言融合不足的问题。通过构建“思考图像”的框架，推动AI从被动感知向主动认知发展。**

- **链接: [http://arxiv.org/pdf/2506.23918v1](http://arxiv.org/pdf/2506.23918v1)**

> **作者:** Zhaochen Su; Peng Xia; Hangyu Guo; Zhenhua Liu; Yan Ma; Xiaoye Qu; Jiaqi Liu; Yanshu Li; Kaide Zeng; Zhengyuan Yang; Linjie Li; Yu Cheng; Heng Ji; Junxian He; Yi R.; Fung
>
> **备注:** We maintain a real-time GitHub repository tracking progress at: https://github.com/zhaochen0110/Awesome_Think_With_Images
>
> **摘要:** Recent progress in multimodal reasoning has been significantly advanced by textual Chain-of-Thought (CoT), a paradigm where models conduct reasoning within language. This text-centric approach, however, treats vision as a static, initial context, creating a fundamental "semantic gap" between rich perceptual data and discrete symbolic thought. Human cognition often transcends language, utilizing vision as a dynamic mental sketchpad. A similar evolution is now unfolding in AI, marking a fundamental paradigm shift from models that merely think about images to those that can truly think with images. This emerging paradigm is characterized by models leveraging visual information as intermediate steps in their thought process, transforming vision from a passive input into a dynamic, manipulable cognitive workspace. In this survey, we chart this evolution of intelligence along a trajectory of increasing cognitive autonomy, which unfolds across three key stages: from external tool exploration, through programmatic manipulation, to intrinsic imagination. To structure this rapidly evolving field, our survey makes four key contributions. (1) We establish the foundational principles of the think with image paradigm and its three-stage framework. (2) We provide a comprehensive review of the core methods that characterize each stage of this roadmap. (3) We analyze the critical landscape of evaluation benchmarks and transformative applications. (4) We identify significant challenges and outline promising future directions. By providing this structured overview, we aim to offer a clear roadmap for future research towards more powerful and human-aligned multimodal AI.
>
---
#### [new 092] Frequency-enhanced Multi-granularity Context Network for Efficient Vertebrae Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决3D CT/MRI中椎体分割精度不足的问题。通过引入FMC-Net，结合多粒度上下文与频率增强技术，提升分割准确性。**

- **链接: [http://arxiv.org/pdf/2506.23086v1](http://arxiv.org/pdf/2506.23086v1)**

> **作者:** Jian Shi; Tianqi You; Pingping Zhang; Hongli Zhang; Rui Xu; Haojie Li
>
> **备注:** Accepted by MICCAI2025. More modifications my be performed
>
> **摘要:** Automated and accurate segmentation of individual vertebra in 3D CT and MRI images is essential for various clinical applications. Due to the limitations of current imaging techniques and the complexity of spinal structures, existing methods still struggle with reducing the impact of image blurring and distinguishing similar vertebrae. To alleviate these issues, we introduce a Frequency-enhanced Multi-granularity Context Network (FMC-Net) to improve the accuracy of vertebrae segmentation. Specifically, we first apply wavelet transform for lossless downsampling to reduce the feature distortion in blurred images. The decomposed high and low-frequency components are then processed separately. For the high-frequency components, we apply a High-frequency Feature Refinement (HFR) to amplify the prominence of key features and filter out noises, restoring fine-grained details in blurred images. For the low-frequency components, we use a Multi-granularity State Space Model (MG-SSM) to aggregate feature representations with different receptive fields, extracting spatially-varying contexts while capturing long-range dependencies with linear complexity. The utilization of multi-granularity contexts is essential for distinguishing similar vertebrae and improving segmentation accuracy. Extensive experiments demonstrate that our method outperforms state-of-the-art approaches on both CT and MRI vertebrae segmentation datasets. The source code is publicly available at https://github.com/anaanaa/FMCNet.
>
---
#### [new 093] Proteus-ID: ID-Consistent and Motion-Coherent Video Customization
- **分类: cs.CV**

- **简介: 该论文属于视频身份定制任务，旨在保持身份一致性和运动流畅性。提出Proteus-ID框架，融合多模态信息并优化运动学习，提升视频生成质量。**

- **链接: [http://arxiv.org/pdf/2506.23729v1](http://arxiv.org/pdf/2506.23729v1)**

> **作者:** Guiyu Zhang; Chen Shi; Zijian Jiang; Xunzhi Xiang; Jingjing Qian; Shaoshuai Shi; Li Jiang
>
> **备注:** Preprint. Work in progress
>
> **摘要:** Video identity customization seeks to synthesize realistic, temporally coherent videos of a specific subject, given a single reference image and a text prompt. This task presents two core challenges: (1) maintaining identity consistency while aligning with the described appearance and actions, and (2) generating natural, fluid motion without unrealistic stiffness. To address these challenges, we introduce Proteus-ID, a novel diffusion-based framework for identity-consistent and motion-coherent video customization. First, we propose a Multimodal Identity Fusion (MIF) module that unifies visual and textual cues into a joint identity representation using a Q-Former, providing coherent guidance to the diffusion model and eliminating modality imbalance. Second, we present a Time-Aware Identity Injection (TAII) mechanism that dynamically modulates identity conditioning across denoising steps, improving fine-detail reconstruction. Third, we propose Adaptive Motion Learning (AML), a self-supervised strategy that reweights the training loss based on optical-flow-derived motion heatmaps, enhancing motion realism without requiring additional inputs. To support this task, we construct Proteus-Bench, a high-quality dataset comprising 200K curated clips for training and 150 individuals from diverse professions and ethnicities for evaluation. Extensive experiments demonstrate that Proteus-ID outperforms prior methods in identity preservation, text alignment, and motion quality, establishing a new benchmark for video identity customization. Codes and data are publicly available at https://grenoble-zhang.github.io/Proteus-ID/.
>
---
#### [new 094] ActAlign: Zero-Shot Fine-Grained Video Classification via Language-Guided Sequence Alignment
- **分类: cs.CV; cs.LG; cs.MM; I.2.10; I.2.7**

- **简介: 该论文提出ActAlign，解决零样本细粒度视频分类问题，通过语言引导的序列对齐实现视频与动作描述的匹配。**

- **链接: [http://arxiv.org/pdf/2506.22967v1](http://arxiv.org/pdf/2506.22967v1)**

> **作者:** Amir Aghdam; Vincent Tao Hu
>
> **备注:** Preprint manuscript - Project page: https://github.com/aghdamamir/act-align
>
> **摘要:** We address the task of zero-shot fine-grained video classification, where no video examples or temporal annotations are available for unseen action classes. While contrastive vision-language models such as SigLIP demonstrate strong open-set recognition via mean-pooled image-text similarity, they fail to capture the temporal structure critical for distinguishing fine-grained activities. We introduce ActAlign, a zero-shot framework that formulates video classification as sequence alignment. For each class, a large language model generates an ordered sub-action sequence, which is aligned with video frames using Dynamic Time Warping (DTW) in a shared embedding space. Without any video-text supervision or fine-tuning, ActAlign achieves 30.5% accuracy on the extremely challenging ActionAtlas benchmark, where human accuracy is only 61.6%. ActAlign outperforms billion-parameter video-language models while using approximately 8x less parameters. These results demonstrate that structured language priors, combined with classical alignment techniques, offer a scalable and general approach to unlocking the open-set recognition potential of vision-language models for fine-grained video understanding.
>
---
#### [new 095] PathDiff: Histopathology Image Synthesis with Unpaired Text and Mask Conditions
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，解决 histopathology 图像数据稀缺问题。通过引入未配对的文本和掩码条件，提出 PathDiff 框架，提升图像质量和语义准确性。**

- **链接: [http://arxiv.org/pdf/2506.23440v1](http://arxiv.org/pdf/2506.23440v1)**

> **作者:** Mahesh Bhosale; Abdul Wasi; Yuanhao Zhai; Yunjie Tian; Samuel Border; Nan Xi; Pinaki Sarder; Junsong Yuan; David Doermann; Xuan Gong
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Diffusion-based generative models have shown promise in synthesizing histopathology images to address data scarcity caused by privacy constraints. Diagnostic text reports provide high-level semantic descriptions, and masks offer fine-grained spatial structures essential for representing distinct morphological regions. However, public datasets lack paired text and mask data for the same histopathological images, limiting their joint use in image generation. This constraint restricts the ability to fully exploit the benefits of combining both modalities for enhanced control over semantics and spatial details. To overcome this, we propose PathDiff, a diffusion framework that effectively learns from unpaired mask-text data by integrating both modalities into a unified conditioning space. PathDiff allows precise control over structural and contextual features, generating high-quality, semantically accurate images. PathDiff also improves image fidelity, text-image alignment, and faithfulness, enhancing data augmentation for downstream tasks like nuclei segmentation and classification. Extensive experiments demonstrate its superiority over existing methods.
>
---
#### [new 096] Toward Simple and Robust Contrastive Explanations for Image Classification by Leveraging Instance Similarity and Concept Relevance
- **分类: cs.CV; 68T07; I.2; I.4**

- **简介: 该论文属于图像分类的对比解释任务，旨在解决模型决策原因的可解释性问题。通过实例相似性和概念相关性生成简洁鲁棒的解释。**

- **链接: [http://arxiv.org/pdf/2506.23975v1](http://arxiv.org/pdf/2506.23975v1)**

> **作者:** Yuliia Kaidashova; Bettina Finzel; Ute Schmid
>
> **备注:** 17 pages, 6 figures, KI2025 - 48th German Conference on Artificial Intelligence
>
> **摘要:** Understanding why a classification model prefers one class over another for an input instance is the challenge of contrastive explanation. This work implements concept-based contrastive explanations for image classification by leveraging the similarity of instance embeddings and relevance of human-understandable concepts used by a fine-tuned deep learning model. Our approach extracts concepts with their relevance score, computes contrasts for similar instances, and evaluates the resulting contrastive explanations based on explanation complexity. Robustness is tested for different image augmentations. Two research questions are addressed: (1) whether explanation complexity varies across different relevance ranges, and (2) whether explanation complexity remains consistent under image augmentations such as rotation and noise. The results confirm that for our experiments higher concept relevance leads to shorter, less complex explanations, while lower relevance results in longer, more diffuse explanations. Additionally, explanations show varying degrees of robustness. The discussion of these findings offers insights into the potential of building more interpretable and robust AI systems.
>
---
#### [new 097] OcRFDet: Object-Centric Radiance Fields for Multi-View 3D Object Detection in Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于多视角3D目标检测任务，旨在提升自动驾驶中的3D几何估计能力。通过引入以对象为中心的辐射场，优化特征表示，提高检测性能。**

- **链接: [http://arxiv.org/pdf/2506.23565v1](http://arxiv.org/pdf/2506.23565v1)**

> **作者:** Mingqian Ji; Jian Yang; Shanshan Zhang
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Current multi-view 3D object detection methods typically transfer 2D features into 3D space using depth estimation or 3D position encoder, but in a fully data-driven and implicit manner, which limits the detection performance. Inspired by the success of radiance fields on 3D reconstruction, we assume they can be used to enhance the detector's ability of 3D geometry estimation. However, we observe a decline in detection performance, when we directly use them for 3D rendering as an auxiliary task. From our analysis, we find the performance drop is caused by the strong responses on the background when rendering the whole scene. To address this problem, we propose object-centric radiance fields, focusing on modeling foreground objects while discarding background noises. Specifically, we employ Object-centric Radiance Fields (OcRF) to enhance 3D voxel features via an auxiliary task of rendering foreground objects. We further use opacity - the side-product of rendering- to enhance the 2D foreground BEV features via Height-aware Opacity-based Attention (HOA), where attention maps at different height levels are generated separately via multiple networks in parallel. Extensive experiments on the nuScenes validation and test datasets demonstrate that our OcRFDet achieves superior performance, outperforming previous state-of-the-art methods with 57.2$\%$ mAP and 64.8$\%$ NDS on the nuScenes test benchmark. Code will be available at https://github.com/Mingqj/OcRFDet.
>
---
#### [new 098] Why Settle for Mid: A Probabilistic Viewpoint to Spatial Relationship Alignment in Text-to-image Models
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，解决空间关系对齐问题。提出PSE评估指标和PSG生成方法，提升模型生成图像的空间准确性。**

- **链接: [http://arxiv.org/pdf/2506.23418v1](http://arxiv.org/pdf/2506.23418v1)**

> **作者:** Parham Rezaei; Arash Marioriyad; Mahdieh Soleymani Baghshah; Mohammad Hossein Rohban
>
> **备注:** 12 main pages, 18 figures, and 16 tables
>
> **摘要:** Despite the ability of text-to-image models to generate high-quality, realistic, and diverse images, they face challenges in compositional generation, often struggling to accurately represent details specified in the input prompt. A prevalent issue in compositional generation is the misalignment of spatial relationships, as models often fail to faithfully generate images that reflect the spatial configurations specified between objects in the input prompts. To address this challenge, we propose a novel probabilistic framework for modeling the relative spatial positioning of objects in a scene, leveraging the concept of Probability of Superiority (PoS). Building on this insight, we make two key contributions. First, we introduce a novel evaluation metric, PoS-based Evaluation (PSE), designed to assess the alignment of 2D and 3D spatial relationships between text and image, with improved adherence to human judgment. Second, we propose PoS-based Generation (PSG), an inference-time method that improves the alignment of 2D and 3D spatial relationships in T2I models without requiring fine-tuning. PSG employs a Part-of-Speech PoS-based reward function that can be utilized in two distinct ways: (1) as a gradient-based guidance mechanism applied to the cross-attention maps during the denoising steps, or (2) as a search-based strategy that evaluates a set of initial noise vectors to select the best one. Extensive experiments demonstrate that the PSE metric exhibits stronger alignment with human judgment compared to traditional center-based metrics, providing a more nuanced and reliable measure of complex spatial relationship accuracy in text-image alignment. Furthermore, PSG significantly enhances the ability of text-to-image models to generate images with specified spatial configurations, outperforming state-of-the-art methods across multiple evaluation metrics and benchmarks.
>
---
#### [new 099] A Novel Frame Identification and Synchronization Technique for Smartphone Visible Light Communication Systems Based on Convolutional Neural Networks
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于可见光通信中的帧识别与同步任务，旨在解决S2C通信中的图像模糊、裁剪和旋转问题，提出基于CNN的轻量级解决方案。**

- **链接: [http://arxiv.org/pdf/2506.23004v1](http://arxiv.org/pdf/2506.23004v1)**

> **作者:** Vaigai Nayaki Yokar; Hoa Le-Minh; Xicong Li; Wai Lok Woo; Luis Nero Alves; Stanislav Zvanovec; Tran The Son; Zabih Ghassemlooy
>
> **摘要:** This paper proposes a novel, robust, and lightweight supervised Convolutional Neural Network (CNN)-based technique for frame identification and synchronization, designed to enhance short-link communication performance in a screen-to-camera (S2C) based visible light communication (VLC) system. Developed using Python and the TensorFlow Keras framework, the proposed CNN model was trained through three real-time experimental investigations conducted in Jupyter Notebook. These experiments incorporated a dataset created from scratch to address various real-time challenges in S2C communication, including blurring, cropping, and rotated images in mobility scenarios. Overhead frames were introduced for synchronization, which leads to enhanced system performance. The experimental results demonstrate that the proposed model achieves an overall accuracy of approximately 98.74%, highlighting its effectiveness in identifying and synchronizing frames in S2C VLC systems.
>
---
#### [new 100] Visual-Semantic Knowledge Conflicts in Operating Rooms: Synthetic Data Curation for Surgical Risk Perception in Multimodal Large Language Models
- **分类: cs.CV; cs.AI; 68T07, 68U10, 92C55; I.2.10; I.2.7; J.3; I.2.6**

- **简介: 该论文属于医疗AI任务，旨在解决手术室中视觉与语义知识冲突问题。通过生成合成数据集，研究多模态大语言模型在安全违规检测中的表现与不足。**

- **链接: [http://arxiv.org/pdf/2506.22500v1](http://arxiv.org/pdf/2506.22500v1)**

> **作者:** Weiyi Zhao; Xiaoyu Tan; Liang Liu; Sijia Li; Youwei Song; Xihe Qiu
>
> **备注:** 13 pages, 5 figures. The dataset and appendix are available at https://github.com/zgg2577/VS-KC
>
> **摘要:** Surgical risk identification is critical for patient safety and reducing preventable medical errors. While multimodal large language models (MLLMs) show promise for automated operating room (OR) risk detection, they often exhibit visual-semantic knowledge conflicts (VS-KC), failing to identify visual safety violations despite understanding textual rules. To address this, we introduce a dataset comprising over 34,000 synthetic images generated by diffusion models, depicting operating room scenes containing entities that violate established safety rules. These images were created to alleviate data scarcity and examine MLLMs vulnerabilities. In addition, the dataset includes 214 human-annotated images that serve as a gold-standard reference for validation. This comprehensive dataset, spanning diverse perspectives, stages, and configurations, is designed to expose and study VS-KC. Fine-tuning on OR-VSKC significantly improves MLLMs' detection of trained conflict entities and generalizes well to new viewpoints for these entities, but performance on untrained entity types remains poor, highlighting learning specificity and the need for comprehensive training. The main contributions of this work include: (1) a data generation methodology tailored for rule-violation scenarios; (2) the release of the OR-VSKC dataset and its associated benchmark as open-source resources; and (3) an empirical analysis of violation-sensitive knowledge consistency in representative MLLMs. The dataset and appendix are available at https://github.com/zgg2577/VS-KC.
>
---
#### [new 101] Patch2Loc: Learning to Localize Patches for Unsupervised Brain Lesion Detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于无监督脑部病变检测任务，旨在通过学习正常图像块的空间位置来识别异常区域，提升病变分割效果。**

- **链接: [http://arxiv.org/pdf/2506.22504v1](http://arxiv.org/pdf/2506.22504v1)**

> **作者:** Hassan Baker; Austin J. Brockmeier
>
> **摘要:** Detecting brain lesions as abnormalities observed in magnetic resonance imaging (MRI) is essential for diagnosis and treatment. In the search of abnormalities, such as tumors and malformations, radiologists may benefit from computer-aided diagnostics that use computer vision systems trained with machine learning to segment normal tissue from abnormal brain tissue. While supervised learning methods require annotated lesions, we propose a new unsupervised approach (Patch2Loc) that learns from normal patches taken from structural MRI. We train a neural network model to map a patch back to its spatial location within a slice of the brain volume. During inference, abnormal patches are detected by the relatively higher error and/or variance of the location prediction. This generates a heatmap that can be integrated into pixel-wise methods to achieve finer-grained segmentation. We demonstrate the ability of our model to segment abnormal brain tissues by applying our approach to the detection of tumor tissues in MRI on T2-weighted images from BraTS2021 and MSLUB datasets and T1-weighted images from ATLAS and WMH datasets. We show that it outperforms the state-of-the art in unsupervised segmentation. The codebase for this work can be found on our \href{https://github.com/bakerhassan/Patch2Loc}{GitHub page}.
>
---
#### [new 102] BridgeShape: Latent Diffusion Schrödinger Bridge for 3D Shape Completion
- **分类: cs.CV**

- **简介: 该论文属于3D形状补全任务，旨在解决现有方法无法建模全局最优传输路径及分辨率限制的问题。提出BridgeShape框架，通过最优传输和深度增强的VQ-VAE实现更高质量的形状补全。**

- **链接: [http://arxiv.org/pdf/2506.23205v1](http://arxiv.org/pdf/2506.23205v1)**

> **作者:** Dequan Kong; Zhe Zhu; Honghua Chen; Mingqiang Wei
>
> **摘要:** Existing diffusion-based 3D shape completion methods typically use a conditional paradigm, injecting incomplete shape information into the denoising network via deep feature interactions (e.g., concatenation, cross-attention) to guide sampling toward complete shapes, often represented by voxel-based distance functions. However, these approaches fail to explicitly model the optimal global transport path, leading to suboptimal completions. Moreover, performing diffusion directly in voxel space imposes resolution constraints, limiting the generation of fine-grained geometric details. To address these challenges, we propose BridgeShape, a novel framework for 3D shape completion via latent diffusion Schr\"odinger bridge. The key innovations lie in two aspects: (i) BridgeShape formulates shape completion as an optimal transport problem, explicitly modeling the transition between incomplete and complete shapes to ensure a globally coherent transformation. (ii) We introduce a Depth-Enhanced Vector Quantized Variational Autoencoder (VQ-VAE) to encode 3D shapes into a compact latent space, leveraging self-projected multi-view depth information enriched with strong DINOv2 features to enhance geometric structural perception. By operating in a compact yet structurally informative latent space, BridgeShape effectively mitigates resolution constraints and enables more efficient and high-fidelity 3D shape completion. BridgeShape achieves state-of-the-art performance on large-scale 3D shape completion benchmarks, demonstrating superior fidelity at higher resolutions and for unseen object classes.
>
---
#### [new 103] When Test-Time Adaptation Meets Self-Supervised Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于测试时自适应（TTA）任务，解决SSL模型在无源预训练下的适应问题。提出协同学习框架，结合对比学习与知识蒸馏，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.23529v1](http://arxiv.org/pdf/2506.23529v1)**

> **作者:** Jisu Han; Jihee Park; Dongyoon Han; Wonjun Hwang
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** Training on test-time data enables deep learning models to adapt to dynamic environmental changes, enhancing their practical applicability. Online adaptation from source to target domains is promising but it remains highly reliant on the performance of source pretrained model. In this paper, we investigate whether test-time adaptation (TTA) methods can continuously improve models trained via self-supervised learning (SSL) without relying on source pretraining. We introduce a self-supervised TTA protocol after observing that existing TTA approaches struggle when directly applied to self-supervised models with low accuracy on the source domain. Furthermore, we propose a collaborative learning framework that integrates SSL and TTA models, leveraging contrastive learning and knowledge distillation for stepwise representation refinement. We validate our method on diverse self-supervised models, including DINO, MoCo, and iBOT, across TTA benchmarks. Extensive experiments validate the effectiveness of our approach in SSL, showing that it achieves competitive performance even without source pretraining.
>
---
#### [new 104] PCLVis: Visual Analytics of Process Communication Latency in Large-Scale Simulation
- **分类: cs.CV**

- **简介: 该论文属于可视化分析任务，旨在解决大规模模拟中进程通信延迟的分析问题。通过PCLVis框架，利用MPI数据进行延迟事件定位、路径分析和优化策略制定。**

- **链接: [http://arxiv.org/pdf/2506.23257v1](http://arxiv.org/pdf/2506.23257v1)**

> **作者:** Chongke Bi; Xin Gao; Baofeng Fu; Yuheng Zhao; Siming Chen; Ying Zhao; Yunhai Wang
>
> **摘要:** Large-scale simulations on supercomputers have become important tools for users. However, their scalability remains a problem due to the huge communication cost among parallel processes. Most of the existing communication latency analysis methods rely on the physical link layer information, which is only available to administrators. In this paper, a framework called PCLVis is proposed to help general users analyze process communication latency (PCL) events. Instead of the physical link layer information, the PCLVis uses the MPI process communication data for the analysis. First, a spatial PCL event locating method is developed. All processes with high correlation are classified into a single cluster by constructing a process-correlation tree. Second, the propagation path of PCL events is analyzed by constructing a communication-dependency-based directed acyclic graph (DAG), which can help users interactively explore a PCL event from the temporal evolution of a located PCL event cluster. In this graph, a sliding window algorithm is designed to generate the PCL events abstraction. Meanwhile, a new glyph called the communication state glyph (CS-Glyph) is designed for each process to show its communication states, including its in/out messages and load balance. Each leaf node can be further unfolded to view additional information. Third, a PCL event attribution strategy is formulated to help users optimize their simulations. The effectiveness of the PCLVis framework is demonstrated by analyzing the PCL events of several simulations running on the TH-1A supercomputer. By using the proposed framework, users can greatly improve the efficiency of their simulations.
>
---
#### [new 105] VisionScores -- A system-segmented image score dataset for deep learning tasks
- **分类: cs.CV; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文提出VisionScores数据集，用于深度学习任务。旨在解决音乐乐谱图像分析问题，通过构建结构丰富的图像数据集，支持不同作曲家和作品类型的分析。**

- **链接: [http://arxiv.org/pdf/2506.23030v1](http://arxiv.org/pdf/2506.23030v1)**

> **作者:** Alejandro Romero Amezcua; Mariano José Juan Rivera Meraz
>
> **备注:** Comments: 5 pages, 3 figures. Accepted for presentation at the 2025 IEEE International Conference on Image Processing (ICIP). \c{opyright} 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for any other use
>
> **摘要:** VisionScores presents a novel proposal being the first system-segmented image score dataset, aiming to offer structure-rich, high information-density images for machine and deep learning tasks. Delimited to two-handed piano pieces, it was built to consider not only certain graphic similarity but also composition patterns, as this creative process is highly instrument-dependent. It provides two scenarios in relation to composer and composition type. The first, formed by 14k samples, considers works from different authors but the same composition type, specifically, Sonatinas. The latter, consisting of 10.8K samples, presents the opposite case, various composition types from the same author, being the one selected Franz Liszt. All of the 24.8k samples are formatted as grayscale jpg images of $128 \times 512$ pixels. VisionScores supplies the users not only the formatted samples but the systems' order and pieces' metadata. Moreover, unsegmented full-page scores and the pre-formatted images are included for further analysis.
>
---
#### [new 106] Towards foundational LiDAR world models with efficient latent flow matching
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于LiDAR世界模型任务，旨在提升模型在不同场景下的迁移能力。通过改进流匹配框架，提高重建精度与计算效率，减少对标注数据的依赖。**

- **链接: [http://arxiv.org/pdf/2506.23434v1](http://arxiv.org/pdf/2506.23434v1)**

> **作者:** Tianran Liu; Shengwen Zhao; Nicholas Rhinehart
>
> **备注:** 25 pages, 13 figures
>
> **摘要:** LiDAR-based world models offer more structured and geometry-aware representations than their image-based counterparts. However, existing LiDAR world models are narrowly trained; each model excels only in the domain for which it was built. Can we develop LiDAR world models that exhibit strong transferability across multiple domains? We conduct the first systematic domain transfer study across three demanding scenarios: (i) outdoor to indoor generalization, (ii) sparse-beam \& dense-beam adaptation, and (iii) non-semantic to semantic transfer. Given different amounts of fine-tuning data, our experiments show that a single pre-trained model can achieve up to 11% absolute improvement (83\% relative) over training from scratch and outperforms training from scratch in 30/36 of our comparisons. This transferability of dynamic learning significantly reduces the reliance on manually annotated data for semantic occupancy forecasting: our method exceed the previous semantic occupancy forecasting models with only 5% of the labeled training data required by prior models. We also observed inefficiencies of current LiDAR world models, mainly through their under-compression of LiDAR data and inefficient training objectives. To address this, we propose a latent conditional flow matching (CFM)-based frameworks that achieves state-of-the-art reconstruction accuracy using only half the training data and a compression ratio 6 times higher than that of prior methods. Our model achieves SOTA performance on future-trajectory-conditioned semantic occupancy forecasting while being 23x more computationally efficient (a 28x FPS speedup); and achieves SOTA performance on semantic occupancy forecasting while being 2x more computationally efficient (a 1.1x FPS speedup).
>
---
#### [new 107] DDL: A Dataset for Interpretable Deepfake Detection and Localization in Real-World Scenarios
- **分类: cs.CV**

- **简介: 该论文属于深度伪造检测与定位任务，旨在解决现有方法缺乏可解释性的问题。通过构建大规模DDL数据集，提升检测模型的可解释性和实用性。**

- **链接: [http://arxiv.org/pdf/2506.23292v1](http://arxiv.org/pdf/2506.23292v1)**

> **作者:** Changtao Miao; Yi Zhang; Weize Gao; Man Luo; Weiwei Feng; Zhiya Tan; Jianshu Li; Ajian Liu; Yunfeng Diao; Qi Chu; Tao Gong; Zhe Li; Weibin Yao; Joey Tianyi Zhou
>
> **备注:** This paper is a preliminary version, with an extended and comprehensive version currently under development
>
> **摘要:** Recent advances in AIGC have exacerbated the misuse of malicious deepfake content, making the development of reliable deepfake detection methods an essential means to address this challenge. Although existing deepfake detection models demonstrate outstanding performance in detection metrics, most methods only provide simple binary classification results, lacking interpretability. In critical domains such as law, interpretability is crucial for enhancing the credibility and authority of decisions. Recent studies attempt to improve the interpretability of classification results by providing spatial manipulation masks or temporal forgery segments. However, the practical effectiveness of these methods remains suboptimal due to limitations of the forgery data. Most current deepfake datasets predominantly offer binary labels, only a few datasets with localization annotations. However, they suffer from restricted forgery scenarios, limited diversity in deepfake types, and insufficient data scale, making them inadequate for complex real-world scenarios. To address this predicament, we construct a novel large-scale deepfake detection and localization ($\textbf{DDL}$) dataset containing over $\textbf{1.8M}$ forged samples and encompassing up to $\textbf{75}$ distinct deepfake methods. The DDL design incorporates four key innovations: (1) $\textbf{Diverse Forgery Scenarios}$, (2) $\textbf{Comprehensive Deepfake Methods}$, (3) $\textbf{Varied Manipulation Modes}$, and (4) $\textbf{Fine-grained Forgery Annotations}$. Through these improvements, our DDL not only provides a more challenging benchmark for complex real-world forgeries, but also offers crucial support for building next-generation deepfake detection, localization, and interpretability methods. The DDL dataset project page is on https://deepfake-workshop-ijcai2025.github.io/main/index.html.
>
---
#### [new 108] Weakly Supervised Object Segmentation by Background Conditional Divergence
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于弱监督目标分割任务，解决在缺乏大量标注数据的领域中自动分割对象的问题。通过图像级标签训练模型，并生成对比背景图像以提升分割效果。**

- **链接: [http://arxiv.org/pdf/2506.22505v1](http://arxiv.org/pdf/2506.22505v1)**

> **作者:** Hassan Baker; Matthew S. Emigh; Austin J. Brockmeier
>
> **摘要:** As a computer vision task, automatic object segmentation remains challenging in specialized image domains without massive labeled data, such as synthetic aperture sonar images, remote sensing, biomedical imaging, etc. In any domain, obtaining pixel-wise segmentation masks is expensive. In this work, we propose a method for training a masking network to perform binary object segmentation using weak supervision in the form of image-wise presence or absence of an object of interest, which provides less information but may be obtained more quickly from manual or automatic labeling. A key step in our method is that the segmented objects can be placed into background-only images to create realistic, images of the objects with counterfactual backgrounds. To create a contrast between the original and counterfactual background images, we propose to first cluster the background-only images, and then during learning create counterfactual images that blend objects segmented from their original source backgrounds to backgrounds chosen from a targeted cluster. One term in the training loss is the divergence between these counterfactual images and the real object images with backgrounds of the target cluster. The other term is a supervised loss for background-only images. While an adversarial critic could provide the divergence, we use sample-based divergences. We conduct experiments on side-scan and synthetic aperture sonar in which our approach succeeds compared to previous unsupervised segmentation baselines that were only tested on natural images. Furthermore, to show generality we extend our experiments to natural images, obtaining reasonable performance with our method that avoids pretrained networks, generative networks, and adversarial critics. The basecode for this work can be found at \href{GitHub}{https://github.com/bakerhassan/WSOS}.
>
---
#### [new 109] Enhancing Spatial Reasoning in Multimodal Large Language Models through Reasoning-based Segmentation
- **分类: cs.CV**

- **简介: 该论文属于3D点云感知任务，旨在解决复杂指令下的空间推理问题。提出R²S框架和3D ReasonSeg数据集，提升模型的空间推理能力。**

- **链接: [http://arxiv.org/pdf/2506.23120v1](http://arxiv.org/pdf/2506.23120v1)**

> **作者:** Zhenhua Ning; Zhuotao Tian; Shaoshuai Shi; Guangming Lu; Daojing He; Wenjie Pei; Li Jiang
>
> **摘要:** Recent advances in point cloud perception have demonstrated remarkable progress in scene understanding through vision-language alignment leveraging large language models (LLMs). However, existing methods may still encounter challenges in handling complex instructions that require accurate spatial reasoning, even if the 3D point cloud data provides detailed spatial cues such as size and position for identifying the targets. To tackle this issue, we propose Relevant Reasoning Segmentation (R$^2$S), a reasoning-based segmentation framework. The framework emulates human cognitive processes by decomposing spatial reasoning into two sequential stages: first identifying relevant elements, then processing instructions guided by their associated visual priors. Furthermore, acknowledging the inadequacy of existing datasets in complex reasoning tasks, we introduce 3D ReasonSeg, a reasoning-based segmentation dataset comprising 25,185 training samples and 3,966 validation samples with precise annotations. Both quantitative and qualitative experiments demonstrate that the R$^2$S and 3D ReasonSeg effectively endow 3D point cloud perception with stronger spatial reasoning capabilities, and we hope that they can serve as a new baseline and benchmark for future work.
>
---
#### [new 110] Epona: Autoregressive Diffusion World Model for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文提出Epona模型，用于自动驾驶中的世界建模任务，解决长时序预测和轨迹规划整合问题。通过时空解耦和模块化设计实现高质量视频生成与实时运动规划。**

- **链接: [http://arxiv.org/pdf/2506.24113v1](http://arxiv.org/pdf/2506.24113v1)**

> **作者:** Kaiwen Zhang; Zhenyu Tang; Xiaotao Hu; Xingang Pan; Xiaoyang Guo; Yuan Liu; Jingwei Huang; Li Yuan; Qian Zhang; Xiao-Xiao Long; Xun Cao; Wei Yin
>
> **备注:** ICCV2025, Project Page: https://kevin-thu.github.io/Epona/
>
> **摘要:** Diffusion models have demonstrated exceptional visual quality in video generation, making them promising for autonomous driving world modeling. However, existing video diffusion-based world models struggle with flexible-length, long-horizon predictions and integrating trajectory planning. This is because conventional video diffusion models rely on global joint distribution modeling of fixed-length frame sequences rather than sequentially constructing localized distributions at each timestep. In this work, we propose Epona, an autoregressive diffusion world model that enables localized spatiotemporal distribution modeling through two key innovations: 1) Decoupled spatiotemporal factorization that separates temporal dynamics modeling from fine-grained future world generation, and 2) Modular trajectory and video prediction that seamlessly integrate motion planning with visual modeling in an end-to-end framework. Our architecture enables high-resolution, long-duration generation while introducing a novel chain-of-forward training strategy to address error accumulation in autoregressive loops. Experimental results demonstrate state-of-the-art performance with 7.4\% FVD improvement and minutes longer prediction duration compared to prior works. The learned world model further serves as a real-time motion planner, outperforming strong end-to-end planners on NAVSIM benchmarks. Code will be publicly available at \href{https://github.com/Kevin-thu/Epona/}{https://github.com/Kevin-thu/Epona/}.
>
---
#### [new 111] RoboScape: Physics-informed Embodied World Model
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人视觉与物理建模任务，旨在解决世界模型物理感知不足的问题。通过联合学习视频生成与物理知识，提升视频真实性和运动合理性。**

- **链接: [http://arxiv.org/pdf/2506.23135v1](http://arxiv.org/pdf/2506.23135v1)**

> **作者:** Yu Shang; Xin Zhang; Yinzhou Tang; Lei Jin; Chen Gao; Wei Wu; Yong Li
>
> **备注:** 17 pages
>
> **摘要:** World models have become indispensable tools for embodied intelligence, serving as powerful simulators capable of generating realistic robotic videos while addressing critical data scarcity challenges. However, current embodied world models exhibit limited physical awareness, particularly in modeling 3D geometry and motion dynamics, resulting in unrealistic video generation for contact-rich robotic scenarios. In this paper, we present RoboScape, a unified physics-informed world model that jointly learns RGB video generation and physics knowledge within an integrated framework. We introduce two key physics-informed joint training tasks: temporal depth prediction that enhances 3D geometric consistency in video rendering, and keypoint dynamics learning that implicitly encodes physical properties (e.g., object shape and material characteristics) while improving complex motion modeling. Extensive experiments demonstrate that RoboScape generates videos with superior visual fidelity and physical plausibility across diverse robotic scenarios. We further validate its practical utility through downstream applications including robotic policy training with generated data and policy evaluation. Our work provides new insights for building efficient physics-informed world models to advance embodied intelligence research. The code is available at: https://github.com/tsinghua-fib-lab/RoboScape.
>
---
#### [new 112] JAM-Flow: Joint Audio-Motion Synthesis with Flow Matching
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 该论文提出JAM-Flow，解决跨模态生成任务中的语音与面部动作同步问题，通过联合建模实现音频和视觉的统一生成。**

- **链接: [http://arxiv.org/pdf/2506.23552v1](http://arxiv.org/pdf/2506.23552v1)**

> **作者:** Mingi Kwon; Joonghyuk Shin; Jaeseok Jung; Jaesik Park; Youngjung Uh
>
> **备注:** project page: https://joonghyuk.com/jamflow-web Under review. Preprint published on arXiv
>
> **摘要:** The intrinsic link between facial motion and speech is often overlooked in generative modeling, where talking head synthesis and text-to-speech (TTS) are typically addressed as separate tasks. This paper introduces JAM-Flow, a unified framework to simultaneously synthesize and condition on both facial motion and speech. Our approach leverages flow matching and a novel Multi-Modal Diffusion Transformer (MM-DiT) architecture, integrating specialized Motion-DiT and Audio-DiT modules. These are coupled via selective joint attention layers and incorporate key architectural choices, such as temporally aligned positional embeddings and localized joint attention masking, to enable effective cross-modal interaction while preserving modality-specific strengths. Trained with an inpainting-style objective, JAM-Flow supports a wide array of conditioning inputs-including text, reference audio, and reference motion-facilitating tasks such as synchronized talking head generation from text, audio-driven animation, and much more, within a single, coherent model. JAM-Flow significantly advances multi-modal generative modeling by providing a practical solution for holistic audio-visual synthesis. project page: https://joonghyuk.com/jamflow-web
>
---
#### [new 113] When Small Guides Large: Cross-Model Co-Learning for Test-Time Adaptation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于测试时自适应（TTA）任务，解决多模型协同提升适应性能的问题。提出COCA框架，通过交叉学习增强模型适应能力。**

- **链接: [http://arxiv.org/pdf/2506.23724v1](http://arxiv.org/pdf/2506.23724v1)**

> **作者:** Chang'an Yi; Xiaohui Deng; Guohao Chen; Yan Zhou; Qinghua Lu; Shuaicheng Niu
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** Test-time Adaptation (TTA) adapts a given model to testing domain data with potential domain shifts through online unsupervised learning, yielding impressive performance. However, to date, existing TTA methods primarily focus on single-model adaptation. In this work, we investigate an intriguing question: how does cross-model knowledge influence the TTA process? Our findings reveal that, in TTA's unsupervised online setting, each model can provide complementary, confident knowledge to the others, even when there are substantial differences in model size. For instance, a smaller model like MobileViT (10.6M parameters) can effectively guide a larger model like ViT-Base (86.6M parameters). In light of this, we propose COCA, a Cross-Model Co-Learning framework for TTA, which mainly consists of two main strategies. 1) Co-adaptation adaptively integrates complementary knowledge from other models throughout the TTA process, reducing individual model biases. 2) Self-adaptation enhances each model's unique strengths via unsupervised learning, enabling diverse adaptation to the target domain. Extensive experiments show that COCA, which can also serve as a plug-and-play module, significantly boosts existing SOTAs, on models with various sizes--including ResNets, ViTs, and Mobile-ViTs--via cross-model co-learned TTA. For example, with Mobile-ViT's guidance, COCA raises ViT-Base's average adaptation accuracy on ImageNet-C from 51.7% to 64.5%. The code is publicly available at https://github.com/ycarobot/COCA.
>
---
#### [new 114] From Coarse to Fine: Learnable Discrete Wavelet Transforms for Efficient 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决3D高斯点云渲染中高斯基元过多导致的内存问题。通过引入可学习的小波变换，实现粗到细的优化，减少冗余高斯基元。**

- **链接: [http://arxiv.org/pdf/2506.23042v1](http://arxiv.org/pdf/2506.23042v1)**

> **作者:** Hung Nguyen; An Le; Runfa Li; Truong Nguyen
>
> **备注:** Accepted to ICCV Workshop
>
> **摘要:** 3D Gaussian Splatting has emerged as a powerful approach in novel view synthesis, delivering rapid training and rendering but at the cost of an ever-growing set of Gaussian primitives that strains memory and bandwidth. We introduce AutoOpti3DGS, a training-time framework that automatically restrains Gaussian proliferation without sacrificing visual fidelity. The key idea is to feed the input images to a sequence of learnable Forward and Inverse Discrete Wavelet Transforms, where low-pass filters are kept fixed, high-pass filters are learnable and initialized to zero, and an auxiliary orthogonality loss gradually activates fine frequencies. This wavelet-driven, coarse-to-fine process delays the formation of redundant fine Gaussians, allowing 3DGS to capture global structure first and refine detail only when necessary. Through extensive experiments, AutoOpti3DGS requires just a single filter learning-rate hyper-parameter, integrates seamlessly with existing efficient 3DGS frameworks, and consistently produces sparser scene representations more compatible with memory or storage-constrained hardware.
>
---
#### [new 115] SIEDD: Shared-Implicit Encoder with Discrete Decoders
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视频压缩任务，解决INR编码速度慢的问题。提出SIEDD架构，通过共享编码器和并行解码器实现高速编码，保持高质量和控制能力。**

- **链接: [http://arxiv.org/pdf/2506.23382v1](http://arxiv.org/pdf/2506.23382v1)**

> **作者:** Vikram Rangarajan; Shishira Maiya; Max Ehrlich; Abhinav Shrivastava
>
> **备注:** Project page at https://vikramrangarajan.github.io/SIEDD . Project code at https://github.com/VikramRangarajan/SIEDD
>
> **摘要:** Implicit Neural Representations (INRs) offer exceptional fidelity for video compression by learning per-video optimized functions, but their adoption is crippled by impractically slow encoding times. Existing attempts to accelerate INR encoding often sacrifice reconstruction quality or crucial coordinate-level control essential for adaptive streaming and transcoding. We introduce SIEDD (Shared-Implicit Encoder with Discrete Decoders), a novel architecture that fundamentally accelerates INR encoding without these compromises. SIEDD first rapidly trains a shared, coordinate-based encoder on sparse anchor frames to efficiently capture global, low-frequency video features. This encoder is then frozen, enabling massively parallel training of lightweight, discrete decoders for individual frame groups, further expedited by aggressive coordinate-space sampling. This synergistic design delivers a remarkable 20-30X encoding speed-up over state-of-the-art INR codecs on HD and 4K benchmarks, while maintaining competitive reconstruction quality and compression ratios. Critically, SIEDD retains full coordinate-based control, enabling continuous resolution decoding and eliminating costly transcoding. Our approach significantly advances the practicality of high-fidelity neural video compression, demonstrating a scalable and efficient path towards real-world deployment. Our codebase is available at https://github.com/VikramRangarajan/SIEDD .
>
---
#### [new 116] High-quality Pseudo-labeling for Point Cloud Segmentation with Scene-level Annotation
- **分类: cs.CV**

- **简介: 该论文属于点云语义分割任务，解决场景级标注下伪标签生成问题。通过多模态信息与语义一致性提升伪标签质量，优化分割效果。**

- **链接: [http://arxiv.org/pdf/2506.23227v1](http://arxiv.org/pdf/2506.23227v1)**

> **作者:** Lunhao Duan; Shanshan Zhao; Xingxing Weng; Jing Zhang; Gui-Song Xia
>
> **备注:** Accepted by TPAMI. Code: https://github.com/LHDuan/WSegPC
>
> **摘要:** This paper investigates indoor point cloud semantic segmentation under scene-level annotation, which is less explored compared to methods relying on sparse point-level labels. In the absence of precise point-level labels, current methods first generate point-level pseudo-labels, which are then used to train segmentation models. However, generating accurate pseudo-labels for each point solely based on scene-level annotations poses a considerable challenge, substantially affecting segmentation performance. Consequently, to enhance accuracy, this paper proposes a high-quality pseudo-label generation framework by exploring contemporary multi-modal information and region-point semantic consistency. Specifically, with a cross-modal feature guidance module, our method utilizes 2D-3D correspondences to align point cloud features with corresponding 2D image pixels, thereby assisting point cloud feature learning. To further alleviate the challenge presented by the scene-level annotation, we introduce a region-point semantic consistency module. It produces regional semantics through a region-voting strategy derived from point-level semantics, which are subsequently employed to guide the point-level semantic predictions. Leveraging the aforementioned modules, our method can rectify inaccurate point-level semantic predictions during training and obtain high-quality pseudo-labels. Significant improvements over previous works on ScanNet v2 and S3DIS datasets under scene-level annotation can demonstrate the effectiveness. Additionally, comprehensive ablation studies validate the contributions of our approach's individual components. The code is available at https://github.com/LHDuan/WSegPC .
>
---
#### [new 117] Probabilistic Prototype Calibration of Vision-Language Models for Generalized Few-shot Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于通用少样本语义分割任务，旨在解决新类别数据稀缺时模型泛化能力不足的问题。提出FewCLIP框架，通过概率原型校准提升模型适应性。**

- **链接: [http://arxiv.org/pdf/2506.22979v1](http://arxiv.org/pdf/2506.22979v1)**

> **作者:** Jie Liu; Jiayi Shen; Pan Zhou; Jan-Jakob Sonke; Efstratios Gavves
>
> **备注:** ICCV2025 Proceeding
>
> **摘要:** Generalized Few-Shot Semantic Segmentation (GFSS) aims to extend a segmentation model to novel classes with only a few annotated examples while maintaining performance on base classes. Recently, pretrained vision-language models (VLMs) such as CLIP have been leveraged in GFSS to improve generalization on novel classes through multi-modal prototypes learning. However, existing prototype-based methods are inherently deterministic, limiting the adaptability of learned prototypes to diverse samples, particularly for novel classes with scarce annotations. To address this, we propose FewCLIP, a probabilistic prototype calibration framework over multi-modal prototypes from the pretrained CLIP, thus providing more adaptive prototype learning for GFSS. Specifically, FewCLIP first introduces a prototype calibration mechanism, which refines frozen textual prototypes with learnable visual calibration prototypes, leading to a more discriminative and adaptive representation. Furthermore, unlike deterministic prototype learning techniques, FewCLIP introduces distribution regularization over these calibration prototypes. This probabilistic formulation ensures structured and uncertainty-aware prototype learning, effectively mitigating overfitting to limited novel class data while enhancing generalization. Extensive experimental results on PASCAL-5$^i$ and COCO-20$^i$ datasets demonstrate that our proposed FewCLIP significantly outperforms state-of-the-art approaches across both GFSS and class-incremental setting. The code is available at https://github.com/jliu4ai/FewCLIP.
>
---
#### [new 118] UrbanLLaVA: A Multi-modal Large Language Model for Urban Intelligence with Spatial Reasoning and Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出UrbanLLaVA，解决城市多模态数据处理问题，通过多阶段训练提升空间推理与领域知识，增强模型在城市任务中的性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.23219v1](http://arxiv.org/pdf/2506.23219v1)**

> **作者:** Jie Feng; Shengyuan Wang; Tianhui Liu; Yanxin Xi; Yong Li
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Urban research involves a wide range of scenarios and tasks that require the understanding of multi-modal data. Current methods often focus on specific data types and lack a unified framework in urban field for processing them comprehensively. The recent success of multi-modal large language models (MLLMs) presents a promising opportunity to overcome this limitation. In this paper, we introduce $\textit{UrbanLLaVA}$, a multi-modal large language model designed to process these four types of data simultaneously and achieve strong performance across diverse urban tasks compared with general MLLMs. In $\textit{UrbanLLaVA}$, we first curate a diverse urban instruction dataset encompassing both single-modal and cross-modal urban data, spanning from location view to global view of urban environment. Additionally, we propose a multi-stage training framework that decouples spatial reasoning enhancement from domain knowledge learning, thereby improving the compatibility and downstream performance of $\textit{UrbanLLaVA}$ across diverse urban tasks. Finally, we also extend existing benchmark for urban research to assess the performance of MLLMs across a wide range of urban tasks. Experimental results from three cities demonstrate that $\textit{UrbanLLaVA}$ outperforms open-source and proprietary MLLMs in both single-modal tasks and complex cross-modal tasks and shows robust generalization abilities across cities. Source codes and data are openly accessible to the research community via https://github.com/tsinghua-fib-lab/UrbanLLaVA.
>
---
#### [new 119] Three-dimensional end-to-end deep learning for brain MRI analysis
- **分类: cs.CV**

- **简介: 该论文属于脑MRI分析任务，旨在评估不同深度学习模型在年龄和性别预测中的表现，发现简单网络优于复杂架构。**

- **链接: [http://arxiv.org/pdf/2506.23916v1](http://arxiv.org/pdf/2506.23916v1)**

> **作者:** Radhika Juglan; Marta Ligero; Zunamys I. Carrero; Asier Rabasco; Tim Lenz; Leo Misera; Gregory Patrick Veldhuizen; Paul Kuntke; Hagen H. Kitzler; Sven Nebelung; Daniel Truhn; Jakob Nikolas Kather
>
> **摘要:** Deep learning (DL) methods are increasingly outperforming classical approaches in brain imaging, yet their generalizability across diverse imaging cohorts remains inadequately assessed. As age and sex are key neurobiological markers in clinical neuroscience, influencing brain structure and disease risk, this study evaluates three of the existing three-dimensional architectures, namely Simple Fully Connected Network (SFCN), DenseNet, and Shifted Window (Swin) Transformers, for age and sex prediction using T1-weighted MRI from four independent cohorts: UK Biobank (UKB, n=47,390), Dallas Lifespan Brain Study (DLBS, n=132), Parkinson's Progression Markers Initiative (PPMI, n=108 healthy controls), and Information eXtraction from Images (IXI, n=319). We found that SFCN consistently outperformed more complex architectures with AUC of 1.00 [1.00-1.00] in UKB (internal test set) and 0.85-0.91 in external test sets for sex classification. For the age prediction task, SFCN demonstrated a mean absolute error (MAE) of 2.66 (r=0.89) in UKB and 4.98-5.81 (r=0.55-0.70) across external datasets. Pairwise DeLong and Wilcoxon signed-rank tests with Bonferroni corrections confirmed SFCN's superiority over Swin Transformer across most cohorts (p<0.017, for three comparisons). Explainability analysis further demonstrates the regional consistency of model attention across cohorts and specific to each task. Our findings reveal that simpler convolutional networks outperform the denser and more complex attention-based DL architectures in brain image analysis by demonstrating better generalizability across different datasets.
>
---
#### [new 120] StackCLIP: Clustering-Driven Stacked Prompt in Zero-Shot Industrial Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于零样本工业异常检测任务，旨在解决CLIP模型中文本与图像特征对齐不足的问题。通过构建堆叠提示和集成特征对齐模块，提升异常检测性能。**

- **链接: [http://arxiv.org/pdf/2506.23577v1](http://arxiv.org/pdf/2506.23577v1)**

> **作者:** Yanning Hou; Yanran Ruan; Junfa Li; Shanshan Wang; Jianfeng Qiu; Ke Xu
>
> **摘要:** Enhancing the alignment between text and image features in the CLIP model is a critical challenge in zero-shot industrial anomaly detection tasks. Recent studies predominantly utilize specific category prompts during pretraining, which can cause overfitting to the training categories and limit model generalization. To address this, we propose a method that transforms category names through multicategory name stacking to create stacked prompts, forming the basis of our StackCLIP model. Our approach introduces two key components. The Clustering-Driven Stacked Prompts (CSP) module constructs generic prompts by stacking semantically analogous categories, while utilizing multi-object textual feature fusion to amplify discriminative anomalies among similar objects. The Ensemble Feature Alignment (EFA) module trains knowledge-specific linear layers tailored for each stack cluster and adaptively integrates them based on the attributes of test categories. These modules work together to deliver superior training speed, stability, and convergence, significantly boosting anomaly segmentation performance. Additionally, our stacked prompt framework offers robust generalization across classification tasks. To further improve performance, we introduce the Regulating Prompt Learning (RPL) module, which leverages the generalization power of stacked prompts to refine prompt learning, elevating results in anomaly detection classification tasks. Extensive testing on seven industrial anomaly detection datasets demonstrates that our method achieves state-of-the-art performance in both zero-shot anomaly detection and segmentation tasks.
>
---
#### [new 121] Unsupervised 3D Braided Hair Reconstruction from a Single-View Image
- **分类: cs.CV**

- **简介: 该论文属于3D发型重建任务，旨在从单视角图像中重建复杂编织发型。通过提出一种无监督方法，有效捕捉发丝的交织结构，提升重建精度与效率。**

- **链接: [http://arxiv.org/pdf/2506.23072v1](http://arxiv.org/pdf/2506.23072v1)**

> **作者:** Jing Gao
>
> **备注:** 6 pages, 3 figures, accepted to the 2025 International Conference on Machine Vision Applications (MVA 2025)
>
> **摘要:** Reconstructing 3D braided hairstyles from single-view images remains a challenging task due to the intricate interwoven structure and complex topologies of braids. Existing strand-based hair reconstruction methods typically focus on loose hairstyles and often struggle to capture the fine-grained geometry of braided hair. In this paper, we propose a novel unsupervised pipeline for efficiently reconstructing 3D braided hair from single-view RGB images. Leveraging a synthetic braid model inspired by braid theory, our approach effectively captures the complex intertwined structures of braids. Extensive experiments demonstrate that our method outperforms state-of-the-art approaches, providing superior accuracy, realism, and efficiency in reconstructing 3D braided hairstyles, supporting expressive hairstyle modeling in digital humans.
>
---
#### [new 122] FADRM: Fast and Accurate Data Residual Matching for Dataset Distillation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出FADRM方法，解决数据蒸馏任务中的效率与效果问题，通过数据残差匹配提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.24125v1](http://arxiv.org/pdf/2506.24125v1)**

> **作者:** Jiacheng Cui; Xinyue Bi; Yaxin Luo; Xiaohan Zhao; Jiacheng Liu; Zhiqiang Shen
>
> **备注:** Code at: https://github.com/Jiacheng8/FADRM
>
> **摘要:** Residual connection has been extensively studied and widely applied at the model architecture level. However, its potential in the more challenging data-centric approaches remains unexplored. In this work, we introduce the concept of Data Residual Matching for the first time, leveraging data-level skip connections to facilitate data generation and mitigate data information vanishing. This approach maintains a balance between newly acquired knowledge through pixel space optimization and existing core local information identification within raw data modalities, specifically for the dataset distillation task. Furthermore, by incorporating optimization-level refinements, our method significantly improves computational efficiency, achieving superior performance while reducing training time and peak GPU memory usage by 50%. Consequently, the proposed method Fast and Accurate Data Residual Matching for Dataset Distillation (FADRM) establishes a new state-of-the-art, demonstrating substantial improvements over existing methods across multiple dataset benchmarks in both efficiency and effectiveness. For instance, with ResNet-18 as the student model and a 0.8% compression ratio on ImageNet-1K, the method achieves 47.7% test accuracy in single-model dataset distillation and 50.0% in multi-model dataset distillation, surpassing RDED by +5.7% and outperforming state-of-the-art multi-model approaches, EDC and CV-DD, by +1.4% and +4.0%. Code is available at: https://github.com/Jiacheng8/FADRM.
>
---
#### [new 123] RGE-GS: Reward-Guided Expansive Driving Scene Reconstruction via Diffusion Priors
- **分类: cs.CV**

- **简介: 该论文属于场景重建任务，旨在解决单次拍摄导致的场景不完整问题。提出RGE-GS框架，结合扩散先验与奖励引导，提升重建质量与效率。**

- **链接: [http://arxiv.org/pdf/2506.22800v1](http://arxiv.org/pdf/2506.22800v1)**

> **作者:** Sicong Du; Jiarun Liu; Qifeng Chen; Hao-Xiang Chen; Tai-Jiang Mu; Sheng Yang
>
> **摘要:** A single-pass driving clip frequently results in incomplete scanning of the road structure, making reconstructed scene expanding a critical requirement for sensor simulators to effectively regress driving actions. Although contemporary 3D Gaussian Splatting (3DGS) techniques achieve remarkable reconstruction quality, their direct extension through the integration of diffusion priors often introduces cumulative physical inconsistencies and compromises training efficiency. To address these limitations, we present RGE-GS, a novel expansive reconstruction framework that synergizes diffusion-based generation with reward-guided Gaussian integration. The RGE-GS framework incorporates two key innovations: First, we propose a reward network that learns to identify and prioritize consistently generated patterns prior to reconstruction phases, thereby enabling selective retention of diffusion outputs for spatial stability. Second, during the reconstruction process, we devise a differentiated training strategy that automatically adjust Gaussian optimization progress according to scene converge metrics, which achieving better convergence than baseline methods. Extensive evaluations of publicly available datasets demonstrate that RGE-GS achieves state-of-the-art performance in reconstruction quality. Our source-code will be made publicly available at https://github.com/CN-ADLab/RGE-GS. (Camera-ready version incorporating reviewer suggestions will be updated soon.)
>
---
#### [new 124] RGC-VQA: An Exploration Database for Robotic-Generated Video Quality Assessment
- **分类: cs.CV**

- **简介: 该论文属于视频质量评估任务，旨在解决机器人生成内容（RGC）的视觉质量评估问题。研究构建了首个RGC数据库，并测试了现有模型的局限性。**

- **链接: [http://arxiv.org/pdf/2506.23852v1](http://arxiv.org/pdf/2506.23852v1)**

> **作者:** Jianing Jin; Jiangyong Ying; Huiyu Duan; Liu Yang; Sijing Wu; Yunhao Li; Yushuo Zheng; Xiongkuo Min; Guangtao Zhai
>
> **摘要:** As camera-equipped robotic platforms become increasingly integrated into daily life, robotic-generated videos have begun to appear on streaming media platforms, enabling us to envision a future where humans and robots coexist. We innovatively propose the concept of Robotic-Generated Content (RGC) to term these videos generated from egocentric perspective of robots. The perceptual quality of RGC videos is critical in human-robot interaction scenarios, and RGC videos exhibit unique distortions and visual requirements that differ markedly from those of professionally-generated content (PGC) videos and user-generated content (UGC) videos. However, dedicated research on quality assessment of RGC videos is still lacking. To address this gap and to support broader robotic applications, we establish the first Robotic-Generated Content Database (RGCD), which contains a total of 2,100 videos drawn from three robot categories and sourced from diverse platforms. A subjective VQA experiment is conducted subsequently to assess human visual perception of robotic-generated videos. Finally, we conduct a benchmark experiment to evaluate the performance of 11 state-of-the-art VQA models on our database. Experimental results reveal significant limitations in existing VQA models when applied to complex, robotic-generated content, highlighting a critical need for RGC-specific VQA models. Our RGCD is publicly available at: https://github.com/IntMeGroup/RGC-VQA.
>
---
#### [new 125] Detecting What Matters: A Novel Approach for Out-of-Distribution 3D Object Detection in Autonomous Vehicles
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于3D目标检测任务，旨在解决自动驾驶中对分布外（OOD）物体检测不足的问题。通过判断物体是否有害而非仅分类，提升安全决策能力。**

- **链接: [http://arxiv.org/pdf/2506.23426v1](http://arxiv.org/pdf/2506.23426v1)**

> **作者:** Menna Taha; Aya Ahmed; Mohammed Karmoose; Yasser Gadallah
>
> **摘要:** Autonomous vehicles (AVs) use object detection models to recognize their surroundings and make driving decisions accordingly. Conventional object detection approaches classify objects into known classes, which limits the AV's ability to detect and appropriately respond to Out-of-Distribution (OOD) objects. This problem is a significant safety concern since the AV may fail to detect objects or misclassify them, which can potentially lead to hazardous situations such as accidents. Consequently, we propose a novel object detection approach that shifts the emphasis from conventional class-based classification to object harmfulness determination. Instead of object detection by their specific class, our method identifies them as either 'harmful' or 'harmless' based on whether they pose a danger to the AV. This is done based on the object position relative to the AV and its trajectory. With this metric, our model can effectively detect previously unseen objects to enable the AV to make safer real-time decisions. Our results demonstrate that the proposed model effectively detects OOD objects, evaluates their harmfulness, and classifies them accordingly, thus enhancing the AV decision-making effectiveness in dynamic environments.
>
---
#### [new 126] AdFair-CLIP: Adversarial Fair Contrastive Language-Image Pre-training for Chest X-rays
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在解决CLIP模型中的公平性问题，通过对抗性方法减少种族和性别偏见，提升诊断公平性和准确性。**

- **链接: [http://arxiv.org/pdf/2506.23467v1](http://arxiv.org/pdf/2506.23467v1)**

> **作者:** Chenlang Yi; Zizhan Xiong; Qi Qi; Xiyuan Wei; Girish Bathla; Ching-Long Lin; Bobak Jack Mortazavi; Tianbao Yang
>
> **备注:** This preprint has been accepted by MICCAI 2025
>
> **摘要:** Contrastive Language-Image Pre-training (CLIP) models have demonstrated superior performance across various visual tasks including medical image classification. However, fairness concerns, including demographic biases, have received limited attention for CLIP models. This oversight leads to critical issues, particularly those related to race and gender, resulting in disparities in diagnostic outcomes and reduced reliability for underrepresented groups. To address these challenges, we introduce AdFair-CLIP, a novel framework employing adversarial feature intervention to suppress sensitive attributes, thereby mitigating spurious correlations and improving prediction fairness. We conduct comprehensive experiments on chest X-ray (CXR) datasets, and show that AdFair-CLIP significantly enhances both fairness and diagnostic accuracy, while maintaining robust generalization in zero-shot and few-shot scenarios. These results establish new benchmarks for fairness-aware learning in CLIP-based medical diagnostic models, particularly for CXR analysis.
>
---
#### [new 127] Brain Tumor Detection through Thermal Imaging and MobileNET
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于脑肿瘤检测任务，旨在解决传统方法成本高、效率低的问题。通过使用MobileNET模型和图像处理技术，实现高效准确的肿瘤检测。**

- **链接: [http://arxiv.org/pdf/2506.23627v1](http://arxiv.org/pdf/2506.23627v1)**

> **作者:** Roham Maiti; Debasmita Bhoumik
>
> **摘要:** Brain plays a crucial role in regulating body functions and cognitive processes, with brain tumors posing significant risks to human health. Precise and prompt detection is a key factor in proper treatment and better patient outcomes. Traditional methods for detecting brain tumors, that include biopsies, MRI, and CT scans often face challenges due to their high costs and the need for specialized medical expertise. Recent developments in machine learning (ML) and deep learning (DL) has exhibited strong capabilities in automating the identification and categorization of brain tumors from medical images, especially MRI scans. However, these classical ML models have limitations, such as high computational demands, the need for large datasets, and long training times, which hinder their accessibility and efficiency. Our research uses MobileNET model for efficient detection of these tumors. The novelty of this project lies in building an accurate tumor detection model which use less computing re-sources and runs in less time followed by efficient decision making through the use of image processing technique for accurate results. The suggested method attained an average accuracy of 98.5%.
>
---
#### [new 128] Trident: Detecting Face Forgeries with Adversarial Triplet Learning
- **分类: cs.CV**

- **简介: 该论文属于人脸伪造检测任务，旨在解决现有模型对新型伪造方法适应性差的问题。提出Trident框架，结合三元组学习和对抗训练，提升检测鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.23189v1](http://arxiv.org/pdf/2506.23189v1)**

> **作者:** Mustafa Hakan Kara; Aysegul Dundar; Uğur Güdükbay
>
> **备注:** 11 pages, 3 figures, and 7 tables
>
> **摘要:** As face forgeries generated by deep neural networks become increasingly sophisticated, detecting face manipulations in digital media has posed a significant challenge, underscoring the importance of maintaining digital media integrity and combating visual disinformation. Current detection models, predominantly based on supervised training with domain-specific data, often falter against forgeries generated by unencountered techniques. In response to this challenge, we introduce \textit{Trident}, a face forgery detection framework that employs triplet learning with a Siamese network architecture for enhanced adaptability across diverse forgery methods. \textit{Trident} is trained on curated triplets to isolate nuanced differences of forgeries, capturing fine-grained features that distinguish pristine samples from manipulated ones while controlling for other variables. To further enhance generalizability, we incorporate domain-adversarial training with a forgery discriminator. This adversarial component guides our embedding model towards forgery-agnostic representations, improving its robustness to unseen manipulations. In addition, we prevent gradient flow from the classifier head to the embedding model, avoiding overfitting induced by artifacts peculiar to certain forgeries. Comprehensive evaluations across multiple benchmarks and ablation studies demonstrate the effectiveness of our framework. We will release our code in a GitHub repository.
>
---
#### [new 129] IR3D-Bench: Evaluating Vision-Language Model Scene Understanding as Agentic Inverse Rendering
- **分类: cs.CV**

- **简介: 该论文提出IR3D-Bench，用于评估视觉语言模型通过主动创建来理解场景的能力，解决传统基准在视觉精确性上的不足。**

- **链接: [http://arxiv.org/pdf/2506.23329v1](http://arxiv.org/pdf/2506.23329v1)**

> **作者:** Parker Liu; Chenxin Li; Zhengxin Li; Yipeng Wu; Wuyang Li; Zhiqin Yang; Zhenyuan Zhang; Yunlong Lin; Sirui Han; Brandon Y. Feng
>
> **备注:** Project Page: https://ir3d-bench.github.io/
>
> **摘要:** Vision-language models (VLMs) excel at descriptive tasks, but whether they truly understand scenes from visual observations remains uncertain. We introduce IR3D-Bench, a benchmark challenging VLMs to demonstrate understanding through active creation rather than passive recognition. Grounded in the analysis-by-synthesis paradigm, IR3D-Bench tasks Vision-Language Agents (VLAs) with actively using programming and rendering tools to recreate the underlying 3D structure of an input image, achieving agentic inverse rendering through tool use. This "understanding-by-creating" approach probes the tool-using generative capacity of VLAs, moving beyond the descriptive or conversational capacity measured by traditional scene understanding benchmarks. We provide a comprehensive suite of metrics to evaluate geometric accuracy, spatial relations, appearance attributes, and overall plausibility. Initial experiments on agentic inverse rendering powered by various state-of-the-art VLMs highlight current limitations, particularly in visual precision rather than basic tool usage. IR3D-Bench, including data and evaluation protocols, is released to facilitate systematic study and development of tool-using VLAs towards genuine scene understanding by creating.
>
---
#### [new 130] DEL: Dense Event Localization for Multi-modal Audio-Visual Understanding
- **分类: cs.CV**

- **简介: 该论文属于多模态视频理解任务，旨在解决长视频中重叠事件的精细时间定位问题。提出DEL框架，通过音频视觉对齐和跨模态交互优化，提升动作检测精度。**

- **链接: [http://arxiv.org/pdf/2506.23196v1](http://arxiv.org/pdf/2506.23196v1)**

> **作者:** Mona Ahmadian; Amir Shirian; Frank Guerin; Andrew Gilbert
>
> **摘要:** Real-world videos often contain overlapping events and complex temporal dependencies, making multimodal interaction modeling particularly challenging. We introduce DEL, a framework for dense semantic action localization, aiming to accurately detect and classify multiple actions at fine-grained temporal resolutions in long untrimmed videos. DEL consists of two key modules: the alignment of audio and visual features that leverage masked self-attention to enhance intra-mode consistency and a multimodal interaction refinement module that models cross-modal dependencies across multiple scales, enabling high-level semantics and fine-grained details. Our method achieves state-of-the-art performance on multiple real-world Temporal Action Localization (TAL) datasets, UnAV-100, THUMOS14, ActivityNet 1.3, and EPIC-Kitchens-100, surpassing previous approaches with notable average mAP gains of +3.3%, +2.6%, +1.2%, +1.7% (verb), and +1.4% (noun), respectively.
>
---
#### [new 131] Single Image Test-Time Adaptation via Multi-View Co-Training
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，解决单张图像测试时域适应问题。提出一种基于补丁的多视角协同训练方法，仅需一张测试图像即可实现有效分割。**

- **链接: [http://arxiv.org/pdf/2506.23705v1](http://arxiv.org/pdf/2506.23705v1)**

> **作者:** Smriti Joshi; Richard Osuala; Lidia Garrucho; Kaisar Kushibar; Dimitri Kessler; Oliver Diaz; Karim Lekadir
>
> **备注:** MICCAI 2025
>
> **摘要:** Test-time adaptation enables a trained model to adjust to a new domain during inference, making it particularly valuable in clinical settings where such on-the-fly adaptation is required. However, existing techniques depend on large target domain datasets, which are often impractical and unavailable in medical scenarios that demand per-patient, real-time inference. Moreover, current methods commonly focus on two-dimensional images, failing to leverage the volumetric richness of medical imaging data. Bridging this gap, we propose a Patch-Based Multi-View Co-Training method for Single Image Test-Time adaptation. Our method enforces feature and prediction consistency through uncertainty-guided self-training, enabling effective volumetric segmentation in the target domain with only a single test-time image. Validated on three publicly available breast magnetic resonance imaging datasets for tumor segmentation, our method achieves performance close to the upper bound supervised benchmark while also outperforming all existing state-of-the-art methods, on average by a Dice Similarity Coefficient of 3.75%. We publicly share our accessible codebase, readily integrable with the popular nnUNet framework, at https://github.com/smriti-joshi/muvi.git.
>
---
#### [new 132] Oneta: Multi-Style Image Enhancement Using Eigentransformation Functions
- **分类: cs.CV**

- **简介: 该论文提出Oneta算法，用于多风格图像增强任务，解决单一模型处理多种增强需求的问题。通过两个点操作和学习风格令牌实现高效增强。**

- **链接: [http://arxiv.org/pdf/2506.23547v1](http://arxiv.org/pdf/2506.23547v1)**

> **作者:** Jiwon Kim; Soohyun Hwang; Dong-O Kim; Changsu Han; Min Kyu Park; Chang-Su Kim
>
> **摘要:** The first algorithm, called Oneta, for a novel task of multi-style image enhancement is proposed in this work. Oneta uses two point operators sequentially: intensity enhancement with a transformation function (TF) and color correction with a color correction matrix (CCM). This two-step enhancement model, though simple, achieves a high performance upper bound. Also, we introduce eigentransformation function (eigenTF) to represent TF compactly. The Oneta network comprises Y-Net and C-Net to predict eigenTF and CCM parameters, respectively. To support $K$ styles, Oneta employs $K$ learnable tokens. During training, each style token is learned using image pairs from the corresponding dataset. In testing, Oneta selects one of the $K$ style tokens to enhance an image accordingly. Extensive experiments show that the single Oneta network can effectively undertake six enhancement tasks -- retouching, image signal processing, low-light image enhancement, dehazing, underwater image enhancement, and white balancing -- across 30 datasets.
>
---
#### [new 133] 3D Shape Generation: A Survey
- **分类: cs.CV**

- **简介: 该论文属于3D形状生成任务，旨在综述当前技术，解决生成高质量、多样化3D对象的问题，总结了表示方法、生成模型和评估标准。**

- **链接: [http://arxiv.org/pdf/2506.22678v1](http://arxiv.org/pdf/2506.22678v1)**

> **作者:** Nicolas Caytuiro; Ivan Sipiran
>
> **备注:** 20 pages, 5 figures
>
> **摘要:** Recent advances in deep learning have significantly transformed the field of 3D shape generation, enabling the synthesis of complex, diverse, and semantically meaningful 3D objects. This survey provides a comprehensive overview of the current state of the art in 3D shape generation, organizing the discussion around three core components: shape representations, generative modeling approaches, and evaluation protocols. We begin by categorizing 3D representations into explicit, implicit, and hybrid setups, highlighting their structural properties, advantages, and limitations. Next, we review a wide range of generation methods, focusing on feedforward architectures. We further summarize commonly used datasets and evaluation metrics that assess fidelity, diversity, and realism of generated shapes. Finally, we identify open challenges and outline future research directions that could drive progress in controllable, efficient, and high-quality 3D shape generation. This survey aims to serve as a valuable reference for researchers and practitioners seeking a structured and in-depth understanding of this rapidly evolving field.
>
---
#### [new 134] Lightweight Temporal Transformer Decomposition for Federated Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶任务，解决单一图像输入在复杂环境中的不足，通过轻量级时序Transformer分解方法融合时间信息，提升系统性能与实时性。**

- **链接: [http://arxiv.org/pdf/2506.23523v1](http://arxiv.org/pdf/2506.23523v1)**

> **作者:** Tuong Do; Binh X. Nguyen; Quang D. Tran; Erman Tjiputra; Te-Chuan Chiu; Anh Nguyen
>
> **备注:** Accepted in IROS 2025
>
> **摘要:** Traditional vision-based autonomous driving systems often face difficulties in navigating complex environments when relying solely on single-image inputs. To overcome this limitation, incorporating temporal data such as past image frames or steering sequences, has proven effective in enhancing robustness and adaptability in challenging scenarios. While previous high-performance methods exist, they often rely on resource-intensive fusion networks, making them impractical for training and unsuitable for federated learning. To address these challenges, we propose lightweight temporal transformer decomposition, a method that processes sequential image frames and temporal steering data by breaking down large attention maps into smaller matrices. This approach reduces model complexity, enabling efficient weight updates for convergence and real-time predictions while leveraging temporal information to enhance autonomous driving performance. Intensive experiments on three datasets demonstrate that our method outperforms recent approaches by a clear margin while achieving real-time performance. Additionally, real robot experiments further confirm the effectiveness of our method.
>
---
#### [new 135] Low-latency vision transformers via large-scale multi-head attention
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决Transformer模型延迟高和效率低的问题。通过改进多头注意力机制，提升分类准确率并降低延迟。**

- **链接: [http://arxiv.org/pdf/2506.23832v1](http://arxiv.org/pdf/2506.23832v1)**

> **作者:** Ronit D. Gross; Tal Halevi; Ella Koresh; Yarden Tzach; Ido Kanter
>
> **备注:** 23 pages, 4 figures, 7 tables
>
> **摘要:** The emergence of spontaneous symmetry breaking among a few heads of multi-head attention (MHA) across transformer blocks in classification tasks was recently demonstrated through the quantification of single-nodal performance (SNP). This finding indicates that each head focuses its attention on a subset of labels through cooperation among its SNPs. This underlying learning mechanism is generalized to large-scale MHA (LS-MHA) using a single matrix value representing single-head performance (SHP), analogous to single-filter performance in convolutional neural networks (CNNs). The results indicate that each SHP matrix comprises multiple unit clusters such that each label being explicitly recognized by a few heads with negligible noise. This leads to an increased signal-to-noise ratio (SNR) along the transformer blocks, thereby improving classification accuracy. These features give rise to several distinct vision transformer (ViT) architectures that achieve the same accuracy but differ in their LS-MHA structures. As a result, their soft committee yields superior accuracy, an outcome not typically observed in CNNs which rely on hundreds of filters. In addition, a significant reduction in latency is achieved without affecting the accuracy by replacing the initial transformer blocks with convolutional layers. This substitution accelerates early-stage learning, which is then improved by subsequent transformer layers. The extension of this learning mechanism to natural language processing tasks, based on quantitative differences between CNNs and ViT architectures, has the potential to yield new insights in deep learning. The findings are demonstrated using compact convolutional transformer architectures trained on the CIFAR-100 dataset.
>
---
#### [new 136] VMoBA: Mixture-of-Block Attention for Video Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决视频扩散模型中注意力机制计算复杂度高的问题。提出VMoBA方法，通过稀疏注意力提升效率与质量。**

- **链接: [http://arxiv.org/pdf/2506.23858v1](http://arxiv.org/pdf/2506.23858v1)**

> **作者:** Jianzong Wu; Liang Hou; Haotian Yang; Xin Tao; Ye Tian; Pengfei Wan; Di Zhang; Yunhai Tong
>
> **备注:** Code is at https://github.com/KwaiVGI/VMoBA
>
> **摘要:** The quadratic complexity of full attention mechanisms poses a significant bottleneck for Video Diffusion Models (VDMs) aiming to generate long-duration, high-resolution videos. While various sparse attention methods have been proposed, many are designed as training-free inference accelerators or do not optimally capture the unique spatio-temporal characteristics inherent in video data when trained natively. This paper introduces Video Mixture of Block Attention (VMoBA), a novel sparse attention mechanism specifically adapted for VDMs. Motivated by an in-depth analysis of attention patterns within pre-trained video transformers, which revealed strong spatio-temporal locality, varying query importance, and head-specific concentration levels, VMoBA enhances the original MoBA framework with three key modifications: (1) a layer-wise recurrent block partition scheme (1D-2D-3D) to dynamically adapt to diverse spatio-temporal attention patterns and improve efficiency; (2) global block selection to prioritize the most salient query-key block interactions across an entire attention head; and (3) threshold-based block selection to dynamically determine the number of attended blocks based on their cumulative similarity. Extensive experiments demonstrate that VMoBA significantly accelerates the training of VDMs on longer sequences, achieving 2.92x FLOPs and 1.48x latency speedup, while attaining comparable or even superior generation quality to full attention. Furthermore, VMoBA exhibits competitive performance in training-free inference, offering 2.40x FLOPs and 1.35x latency speedup for high-res video generation.
>
---
#### [new 137] LIGHT: Multi-Modal Text Linking on Historical Maps
- **分类: cs.CV**

- **简介: 该论文属于历史地图文本链接任务，旨在解决多词地名识别问题。通过融合语言、图像和几何特征，提出LIGHT模型提升链接效果。**

- **链接: [http://arxiv.org/pdf/2506.22589v1](http://arxiv.org/pdf/2506.22589v1)**

> **作者:** Yijun Lin; Rhett Olson; Junhan Wu; Yao-Yi Chiang; Jerod Weinman
>
> **备注:** Accepted at ICDAR2025
>
> **摘要:** Text on historical maps provides valuable information for studies in history, economics, geography, and other related fields. Unlike structured or semi-structured documents, text on maps varies significantly in orientation, reading order, shape, and placement. Many modern methods can detect and transcribe text regions, but they struggle to effectively ``link'' the recognized text fragments, e.g., determining a multi-word place name. Existing layout analysis methods model word relationships to improve text understanding in structured documents, but they primarily rely on linguistic features and neglect geometric information, which is essential for handling map text. To address these challenges, we propose LIGHT, a novel multi-modal approach that integrates linguistic, image, and geometric features for linking text on historical maps. In particular, LIGHT includes a geometry-aware embedding module that encodes the polygonal coordinates of text regions to capture polygon shapes and their relative spatial positions on an image. LIGHT unifies this geometric information with the visual and linguistic token embeddings from LayoutLMv3, a pretrained layout analysis model. LIGHT uses the cross-modal information to predict the reading-order successor of each text instance directly with a bi-directional learning strategy that enhances sequence robustness. Experimental results show that LIGHT outperforms existing methods on the ICDAR 2024/2025 MapText Competition data, demonstrating the effectiveness of multi-modal learning for historical map text linking.
>
---
#### [new 138] A Hierarchical Slice Attention Network for Appendicitis Classification in 3D CT Scans
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在提高阑尾炎的诊断效率与准确性。通过引入分层切片注意力机制和预训练模型，提升了小病灶检测能力。**

- **链接: [http://arxiv.org/pdf/2506.23209v1](http://arxiv.org/pdf/2506.23209v1)**

> **作者:** Chia-Wen Huang; Haw Hwai; Chien-Chang Lee; Pei-Yuan Wu
>
> **备注:** 8 pages, 1 figure, 3 tables. Published in IEEE ISBI 2025. This version corrects citation numbering errors
>
> **摘要:** Timely and accurate diagnosis of appendicitis is critical in clinical settings to prevent serious complications. While CT imaging remains the standard diagnostic tool, the growing number of cases can overwhelm radiologists, potentially causing delays. In this paper, we propose a deep learning model that leverages 3D CT scans for appendicitis classification, incorporating Slice Attention mechanisms guided by external 2D datasets to enhance small lesion detection. Additionally, we introduce a hierarchical classification framework using pre-trained 2D models to differentiate between simple and complicated appendicitis. Our approach improves AUC by 3% for appendicitis and 5.9% for complicated appendicitis, offering a more efficient and reliable diagnostic solution compared to previous work.
>
---
#### [new 139] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation
- **分类: cs.CV; cs.AI; cs.MM**

- **简介: 该论文属于光流估计任务，旨在解决高分辨率下内存消耗大的问题。通过优化设计，实现高效多帧光流估计。**

- **链接: [http://arxiv.org/pdf/2506.23151v1](http://arxiv.org/pdf/2506.23151v1)**

> **作者:** Vladislav Bargatin; Egor Chistov; Alexander Yakovenko; Dmitriy Vatolin
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Recent advances in optical flow estimation have prioritized accuracy at the cost of growing GPU memory consumption, particularly for high-resolution (FullHD) inputs. We introduce MEMFOF, a memory-efficient multi-frame optical flow method that identifies a favorable trade-off between multi-frame estimation and GPU memory usage. Notably, MEMFOF requires only 2.09 GB of GPU memory at runtime for 1080p inputs, and 28.5 GB during training, which uniquely positions our method to be trained at native 1080p without the need for cropping or downsampling. We systematically revisit design choices from RAFT-like architectures, integrating reduced correlation volumes and high-resolution training protocols alongside multi-frame estimation, to achieve state-of-the-art performance across multiple benchmarks while substantially reducing memory overhead. Our method outperforms more resource-intensive alternatives in both accuracy and runtime efficiency, validating its robustness for flow estimation at high resolutions. At the time of submission, our method ranks first on the Spring benchmark with a 1-pixel (1px) outlier rate of 3.289, leads Sintel (clean) with an endpoint error (EPE) of 0.963, and achieves the best Fl-all error on KITTI-2015 at 2.94%. The code is available at https://github.com/msu-video-group/memfof.
>
---
#### [new 140] Listener-Rewarded Thinking in VLMs for Image Preferences
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言对齐任务，旨在提升生成模型与人类偏好的一致性。针对现有奖励模型泛化能力差的问题，提出一种结合监听器的强化学习框架，增强推理准确性与解释说服力。**

- **链接: [http://arxiv.org/pdf/2506.22832v1](http://arxiv.org/pdf/2506.22832v1)**

> **作者:** Alexander Gambashidze; Li Pengyi; Matvey Skripkin; Andrey Galichin; Anton Gusarov; Konstantin Sobolev; Andrey Kuznetsov; Ivan Oseledets
>
> **摘要:** Training robust and generalizable reward models for human visual preferences is essential for aligning text-to-image and text-to-video generative models with human intent. However, current reward models often fail to generalize, and supervised fine-tuning leads to memorization, demanding complex annotation pipelines. While reinforcement learning (RL), specifically Group Relative Policy Optimization (GRPO), improves generalization, we uncover a key failure mode: a significant drop in reasoning accuracy occurs when a model's reasoning trace contradicts that of an independent, frozen vision-language model ("listener") evaluating the same output. To address this, we introduce a listener-augmented GRPO framework. Here, the listener re-evaluates the reasoner's chain-of-thought to provide a dense, calibrated confidence score, shaping the RL reward signal. This encourages the reasoner not only to answer correctly, but to produce explanations that are persuasive to an independent model. Our listener-shaped reward scheme achieves best accuracy on the ImageReward benchmark (67.4%), significantly improves out-of-distribution (OOD) performance on a large-scale human preference dataset (1.2M votes, up to +6% over naive reasoner), and reduces reasoning contradictions compared to strong GRPO and SFT baselines. These results demonstrate that listener-based rewards provide a scalable, data-efficient path to aligning vision-language models with nuanced human preferences. We will release our reasoning model here: https://huggingface.co/alexgambashidze/qwen2.5vl_image_preference_reasoner.
>
---
#### [new 141] MTADiffusion: Mask Text Alignment Diffusion Model for Object Inpainting
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，旨在解决语义错位、结构扭曲和风格不一致问题。提出MTADiffusion模型及数据集，结合多任务训练与风格一致性损失提升修复效果。**

- **链接: [http://arxiv.org/pdf/2506.23482v1](http://arxiv.org/pdf/2506.23482v1)**

> **作者:** Jun Huang; Ting Liu; Yihang Wu; Xiaochao Qu; Luoqi Liu; Xiaolin Hu
>
> **备注:** CVPR 2025
>
> **摘要:** Advancements in generative models have enabled image inpainting models to generate content within specific regions of an image based on provided prompts and masks. However, existing inpainting methods often suffer from problems such as semantic misalignment, structural distortion, and style inconsistency. In this work, we present MTADiffusion, a Mask-Text Alignment diffusion model designed for object inpainting. To enhance the semantic capabilities of the inpainting model, we introduce MTAPipeline, an automatic solution for annotating masks with detailed descriptions. Based on the MTAPipeline, we construct a new MTADataset comprising 5 million images and 25 million mask-text pairs. Furthermore, we propose a multi-task training strategy that integrates both inpainting and edge prediction tasks to improve structural stability. To promote style consistency, we present a novel inpainting style-consistency loss using a pre-trained VGG network and the Gram matrix. Comprehensive evaluations on BrushBench and EditBench demonstrate that MTADiffusion achieves state-of-the-art performance compared to other methods.
>
---
#### [new 142] Container damage detection using advanced computer vision model Yolov12 vs Yolov11 vs RF-DETR A comparative analysis
- **分类: cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决集装箱损伤识别问题。通过对比Yolov11、Yolov12和RF-DETR模型性能，评估其在损伤检测中的效果。**

- **链接: [http://arxiv.org/pdf/2506.22517v1](http://arxiv.org/pdf/2506.22517v1)**

> **作者:** Subhadip Kumar
>
> **摘要:** Containers are an integral part of the logistics industry and act as a barrier for cargo. A typical service life for a container is more than 20 years. However, overtime containers suffer various types of damage due to the mechanical as well as natural factors. A damaged container is a safety hazard for the employees handling it and a liability for the logistic company. Therefore, a timely inspection and detection of the damaged container is a key for prolonging service life as well as avoiding safety hazards. In this paper, we will compare the performance of the damage detection by three state-of-the-art advanced computer vision models Yolov12, Yolov11 and RF-DETR. We will use a dataset of 278 annotated images to train, validate and test the model. We will compare the mAP and precision of the model. The objective of this paper is to identify the model that is best suited for container damage detection. The result is mixed. mAP@50 score of Yolov11 and 12 was 81.9% compared to RF-DETR, which was 77.7%. However, while testing the model for not-so-common damaged containers, the RF-DETR model outperformed the others overall, exhibiting superiority to accurately detecting both damaged containers as well as damage occurrences with high confidence.
>
---
#### [new 143] LightBSR: Towards Lightweight Blind Super-Resolution via Discriminative Implicit Degradation Representation Learning
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于图像超分辨率任务，旨在解决盲超分辨率中退化表示判别性不足的问题，提出轻量级模型LightBSR，通过知识蒸馏提升效果。**

- **链接: [http://arxiv.org/pdf/2506.22710v1](http://arxiv.org/pdf/2506.22710v1)**

> **作者:** Jiang Yuan; JI Ma; Bo Wang; Guanzhou Ke; Weiming Hu
>
> **摘要:** Implicit degradation estimation-based blind super-resolution (IDE-BSR) hinges on extracting the implicit degradation representation (IDR) of the LR image and adapting it to LR image features to guide HR detail restoration. Although IDE-BSR has shown potential in dealing with noise interference and complex degradations, existing methods ignore the importance of IDR discriminability for BSR and instead over-complicate the adaptation process to improve effect, resulting in a significant increase in the model's parameters and computations. In this paper, we focus on the discriminability optimization of IDR and propose a new powerful and lightweight BSR model termed LightBSR. Specifically, we employ a knowledge distillation-based learning framework. We first introduce a well-designed degradation-prior-constrained contrastive learning technique during teacher stage to make the model more focused on distinguishing different degradation types. Then we utilize a feature alignment technique to transfer the degradation-related knowledge acquired by the teacher to the student for practical inferencing. Extensive experiments demonstrate the effectiveness of IDR discriminability-driven BSR model design. The proposed LightBSR can achieve outstanding performance with minimal complexity across a range of blind SR tasks. Our code is accessible at: https://github.com/MJ-NCEPU/LightBSR.
>
---
#### [new 144] ViewPoint: Panoramic Video Generation with Pretrained Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于全景视频生成任务，解决全景与透视数据之间的模态差异问题。通过设计ViewPoint地图和Pano-Perspective注意力机制，利用预训练模型生成高质量全景视频。**

- **链接: [http://arxiv.org/pdf/2506.23513v1](http://arxiv.org/pdf/2506.23513v1)**

> **作者:** Zixun Fang; Kai Zhu; Zhiheng Liu; Yu Liu; Wei Zhai; Yang Cao; Zheng-Jun Zha
>
> **备注:** https://becauseimbatman0.github.io/ViewPoint
>
> **摘要:** Panoramic video generation aims to synthesize 360-degree immersive videos, holding significant importance in the fields of VR, world models, and spatial intelligence. Existing works fail to synthesize high-quality panoramic videos due to the inherent modality gap between panoramic data and perspective data, which constitutes the majority of the training data for modern diffusion models. In this paper, we propose a novel framework utilizing pretrained perspective video models for generating panoramic videos. Specifically, we design a novel panorama representation named ViewPoint map, which possesses global spatial continuity and fine-grained visual details simultaneously. With our proposed Pano-Perspective attention mechanism, the model benefits from pretrained perspective priors and captures the panoramic spatial correlations of the ViewPoint map effectively. Extensive experiments demonstrate that our method can synthesize highly dynamic and spatially consistent panoramic videos, achieving state-of-the-art performance and surpassing previous methods.
>
---
#### [new 145] Qwen-GUI-3B: A Lightweight Vision-Language Model for Cross-Resolution GUI Grounding
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Qwen-GUI-3B，一个轻量级视觉语言模型，用于跨分辨率GUI定位任务，解决高分辨率桌面环境数据稀缺问题，通过多阶段训练和数据优化提升精度。**

- **链接: [http://arxiv.org/pdf/2506.23491v1](http://arxiv.org/pdf/2506.23491v1)**

> **作者:** ZongHan Hsieh; Tzer-Jen Wei
>
> **摘要:** This paper introduces Qwen-GUI-3B, a lightweight Vision-Language Model (VLM) specifically designed for Graphical User Interface grounding tasks, achieving performance competitive with significantly larger models. Unlike large-scale VLMs (>7B parameters) that are computationally intensive and impractical for consumer-grade hardware, Qwen-GUI-3B delivers strong grounding accuracy while being fully trainable on a single GPU (RTX 4090). The model incorporates several key innovations: (i) combine cross-platform, multi-resolution dataset of 24K examples from diverse sources including mobile, desktop, and web GUI screenshots to effectively address data scarcity in high-resolution desktop environments; (ii) a two-stage fine-tuning strategy, where initial cross-platform training establishes robust GUI understanding, followed by specialized fine-tuning on high-resolution data to significantly enhance model adaptability; and (iii) data curation and redundancy reduction strategies, demonstrating that randomly sampling a smaller subset with reduced redundancy achieves performance comparable to larger datasets, emphasizing data diversity over sheer volume. Empirical evaluation on standard GUI grounding benchmarks-including ScreenSpot, ScreenSpot-v2, and the challenging ScreenSpot-Pro, highlights Qwen-GUI-3B's exceptional accuracy, achieving 84.9% on ScreenSpot and 86.4% on ScreenSpot-v2, surpassing prior models under 4B parameters. Ablation studies validate the critical role of balanced sampling and two-stage fine-tuning in enhancing robustness, particularly in high-resolution desktop scenarios. The Qwen-GUI-3B is available at: https://github.com/Han1018/Qwen-GUI-3B
>
---
#### [new 146] AttentionGS: Towards Initialization-Free 3D Gaussian Splatting via Structural Attention
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决3DGS依赖高质量点云的问题，通过结构注意力机制实现无需初始化的3D高斯泼溅。**

- **链接: [http://arxiv.org/pdf/2506.23611v1](http://arxiv.org/pdf/2506.23611v1)**

> **作者:** Ziao Liu; Zhenjia Li; Yifeng Shi; Xiangang Li
>
> **摘要:** 3D Gaussian Splatting (3DGS) is a powerful alternative to Neural Radiance Fields (NeRF), excelling in complex scene reconstruction and efficient rendering. However, it relies on high-quality point clouds from Structure-from-Motion (SfM), limiting its applicability. SfM also fails in texture-deficient or constrained-view scenarios, causing severe degradation in 3DGS reconstruction. To address this limitation, we propose AttentionGS, a novel framework that eliminates the dependency on high-quality initial point clouds by leveraging structural attention for direct 3D reconstruction from randomly initialization. In the early training stage, we introduce geometric attention to rapidly recover the global scene structure. As training progresses, we incorporate texture attention to refine fine-grained details and enhance rendering quality. Furthermore, we employ opacity-weighted gradients to guide Gaussian densification, leading to improved surface reconstruction. Extensive experiments on multiple benchmark datasets demonstrate that AttentionGS significantly outperforms state-of-the-art methods, particularly in scenarios where point cloud initialization is unreliable. Our approach paves the way for more robust and flexible 3D Gaussian Splatting in real-world applications.
>
---
#### [new 147] PBCAT: Patch-based composite adversarial training against physically realizable attacks on object detection
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于目标检测领域的安全防御任务，旨在解决物理可实现攻击（如对抗补丁和纹理）对检测模型的威胁。工作提出PBCAT方法，通过结合局部和全局对抗扰动提升模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.23581v1](http://arxiv.org/pdf/2506.23581v1)**

> **作者:** Xiao Li; Yiming Zhu; Yifan Huang; Wei Zhang; Yingzhe He; Jie Shi; Xiaolin Hu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Object detection plays a crucial role in many security-sensitive applications. However, several recent studies have shown that object detectors can be easily fooled by physically realizable attacks, \eg, adversarial patches and recent adversarial textures, which pose realistic and urgent threats. Adversarial Training (AT) has been recognized as the most effective defense against adversarial attacks. While AT has been extensively studied in the $l_\infty$ attack settings on classification models, AT against physically realizable attacks on object detectors has received limited exploration. Early attempts are only performed to defend against adversarial patches, leaving AT against a wider range of physically realizable attacks under-explored. In this work, we consider defending against various physically realizable attacks with a unified AT method. We propose PBCAT, a novel Patch-Based Composite Adversarial Training strategy. PBCAT optimizes the model by incorporating the combination of small-area gradient-guided adversarial patches and imperceptible global adversarial perturbations covering the entire image. With these designs, PBCAT has the potential to defend against not only adversarial patches but also unseen physically realizable attacks such as adversarial textures. Extensive experiments in multiple settings demonstrated that PBCAT significantly improved robustness against various physically realizable attacks over state-of-the-art defense methods. Notably, it improved the detection accuracy by 29.7\% over previous defense methods under one recent adversarial texture attack.
>
---
#### [new 148] DiffFit: Disentangled Garment Warping and Texture Refinement for Virtual Try-On
- **分类: cs.CV**

- **简介: 该论文属于虚拟试衣任务，解决服装细节保留、人体对齐和效率问题。提出DiffFit框架，分两阶段优化几何与纹理，提升生成质量与稳定性。**

- **链接: [http://arxiv.org/pdf/2506.23295v1](http://arxiv.org/pdf/2506.23295v1)**

> **作者:** Xiang Xu
>
> **摘要:** Virtual try-on (VTON) aims to synthesize realistic images of a person wearing a target garment, with broad applications in e-commerce and digital fashion. While recent advances in latent diffusion models have substantially improved visual quality, existing approaches still struggle with preserving fine-grained garment details, achieving precise garment-body alignment, maintaining inference efficiency, and generalizing to diverse poses and clothing styles. To address these challenges, we propose DiffFit, a novel two-stage latent diffusion framework for high-fidelity virtual try-on. DiffFit adopts a progressive generation strategy: the first stage performs geometry-aware garment warping, aligning the garment with the target body through fine-grained deformation and pose adaptation. The second stage refines texture fidelity via a cross-modal conditional diffusion model that integrates the warped garment, the original garment appearance, and the target person image for high-quality rendering. By decoupling geometric alignment and appearance refinement, DiffFit effectively reduces task complexity and enhances both generation stability and visual realism. It excels in preserving garment-specific attributes such as textures, wrinkles, and lighting, while ensuring accurate alignment with the human body. Extensive experiments on large-scale VTON benchmarks demonstrate that DiffFit achieves superior performance over existing state-of-the-art methods in both quantitative metrics and perceptual evaluations.
>
---
#### [new 149] TextMesh4D: High-Quality Text-to-4D Mesh Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到4D网格生成任务，解决动态3D内容生成问题。提出TextMesh4D框架，通过分解静态与动态生成阶段，实现高质量、高一致性4D网格生成。**

- **链接: [http://arxiv.org/pdf/2506.24121v1](http://arxiv.org/pdf/2506.24121v1)**

> **作者:** Sisi Dai; Xinxin Su; Boyan Wan; Ruizhen Hu; Kai Xu
>
> **摘要:** Recent advancements in diffusion generative models significantly advanced image, video, and 3D content creation from user-provided text prompts. However, the challenging problem of dynamic 3D content generation (text-to-4D) with diffusion guidance remains largely unexplored. In this paper, we introduce TextMesh4D, a novel framework for high-quality text-to-4D generation. Our approach leverages per-face Jacobians as a differentiable mesh representation and decomposes 4D generation into two stages: static object creation and dynamic motion synthesis. We further propose a flexibility-rigidity regularization term to stabilize Jacobian optimization under video diffusion priors, ensuring robust geometric performance. Experiments demonstrate that TextMesh4D achieves state-of-the-art results in terms of temporal consistency, structural fidelity, and visual realism. Moreover, TextMesh4D operates with a low GPU memory overhead-requiring only a single 24GB GPU-offering a cost-effective yet high-quality solution for text-driven 4D mesh generation. The code will be released to facilitate future research in text-to-4D generation.
>
---
#### [new 150] DGE-YOLO: Dual-Branch Gathering and Attention for Accurate UAV Object Detection
- **分类: cs.CV**

- **简介: 该论文属于无人机目标检测任务，旨在解决复杂环境下小目标检测难题。提出DGE-YOLO框架，融合多模态信息，提升检测精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.23252v1](http://arxiv.org/pdf/2506.23252v1)**

> **作者:** Kunwei Lv; Ping Lan
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** The rapid proliferation of unmanned aerial vehicles (UAVs) has highlighted the importance of robust and efficient object detection in diverse aerial scenarios. Detecting small objects under complex conditions, however, remains a significant challenge. Existing approaches often prioritize inference speed, leading to degraded performance when handling multi-modal inputs. To address this, we present DGE-YOLO, an enhanced YOLO-based detection framework designed to effectively fuse multi-modal information. Specifically, we introduce a dual-branch architecture for modality-specific feature extraction, enabling the model to process both infrared and visible images. To further enrich semantic representation, we propose an Efficient Multi-scale Attention (EMA) mechanism that enhances feature learning across spatial scales. Additionally, we replace the conventional neck with a Gather-and-Distribute module to mitigate information loss during feature aggregation. Extensive experiments on the Drone Vehicle dataset demonstrate that DGE-YOLO achieves superior performance over state-of-the-art methods, validating its effectiveness in multi-modal UAV object detection tasks.
>
---
#### [new 151] Revisiting Audio-Visual Segmentation with Vision-Centric Transformer
- **分类: cs.CV**

- **简介: 该论文属于音频-视觉分割任务，旨在解决音频混淆和视觉细节丢失问题。提出视觉中心的Transformer框架，通过视觉引导查询提升分割精度。**

- **链接: [http://arxiv.org/pdf/2506.23623v1](http://arxiv.org/pdf/2506.23623v1)**

> **作者:** Shaofei Huang; Rui Ling; Tianrui Hui; Hongyu Li; Xu Zhou; Shifeng Zhang; Si Liu; Richang Hong; Meng Wang
>
> **备注:** Accepted by CVPR 2025; Code: https://github.com/spyflying/VCT_AVS; Models: https://huggingface.co/nowherespyfly/VCT_AVS
>
> **摘要:** Audio-Visual Segmentation (AVS) aims to segment sound-producing objects in video frames based on the associated audio signal. Prevailing AVS methods typically adopt an audio-centric Transformer architecture, where object queries are derived from audio features. However, audio-centric Transformers suffer from two limitations: perception ambiguity caused by the mixed nature of audio, and weakened dense prediction ability due to visual detail loss. To address these limitations, we propose a new Vision-Centric Transformer (VCT) framework that leverages vision-derived queries to iteratively fetch corresponding audio and visual information, enabling queries to better distinguish between different sounding objects from mixed audio and accurately delineate their contours. Additionally, we also introduce a Prototype Prompted Query Generation (PPQG) module within our VCT framework to generate vision-derived queries that are both semantically aware and visually rich through audio prototype prompting and pixel context grouping, facilitating audio-visual information aggregation. Extensive experiments demonstrate that our VCT framework achieves new state-of-the-art performances on three subsets of the AVSBench dataset. The code is available at https://github.com/spyflying/VCT_AVS.
>
---
#### [new 152] MagShield: Towards Better Robustness in Sparse Inertial Motion Capture Under Magnetic Disturbances
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于运动捕捉任务，解决磁干扰下惯性传感器姿态估计误差问题，提出MagShield方法通过检测与校正提升系统鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.22907v1](http://arxiv.org/pdf/2506.22907v1)**

> **作者:** Yunzhe Shao; Xinyu Yi; Lu Yin; Shihui Guo; Junhai Yong; Feng Xu
>
> **摘要:** This paper proposes a novel method called MagShield, designed to address the issue of magnetic interference in sparse inertial motion capture (MoCap) systems. Existing Inertial Measurement Unit (IMU) systems are prone to orientation estimation errors in magnetically disturbed environments, limiting their practical application in real-world scenarios. To address this problem, MagShield employs a "detect-then-correct" strategy, first detecting magnetic disturbances through multi-IMU joint analysis, and then correcting orientation errors using human motion priors. MagShield can be integrated with most existing sparse inertial MoCap systems, improving their performance in magnetically disturbed environments. Experimental results demonstrate that MagShield significantly enhances the accuracy of motion capture under magnetic interference and exhibits good compatibility across different sparse inertial MoCap systems.
>
---
#### [new 153] YM-WML: A new Yolo-based segmentation Model with Weighted Multi-class Loss for medical imaging
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决类别不平衡和复杂结构问题。提出YM-WML模型，结合YOLOv11和加权多类损失函数，提升分割精度。**

- **链接: [http://arxiv.org/pdf/2506.22955v1](http://arxiv.org/pdf/2506.22955v1)**

> **作者:** Haniyeh Nikkhah; Jafar Tanha; Mahdi Zarrin; SeyedEhsan Roshan; Amin Kazempour
>
> **备注:** Accepted at The 7th International conference on Pattern Recognition and Image Analysis (IPRIA 2025)
>
> **摘要:** Medical image segmentation poses significant challenges due to class imbalance and the complex structure of medical images. To address these challenges, this study proposes YM-WML, a novel model for cardiac image segmentation. The model integrates a robust backbone for effective feature extraction, a YOLOv11 neck for multi-scale feature aggregation, and an attention-based segmentation head for precise and accurate segmentation. To address class imbalance, we introduce the Weighted Multi-class Exponential (WME) loss function. On the ACDC dataset, YM-WML achieves a Dice Similarity Coefficient of 91.02, outperforming state-of-the-art methods. The model demonstrates stable training, accurate segmentation, and strong generalization, setting a new benchmark in cardiac segmentation tasks.
>
---
#### [new 154] Decoupled Seg Tokens Make Stronger Reasoning Video Segmenter and Grounder
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频分割与定位任务，旨在解决动态视觉与静态语义混淆问题。通过引入解耦机制和文本预训练，提升模型的语义理解与分割性能。**

- **链接: [http://arxiv.org/pdf/2506.22880v1](http://arxiv.org/pdf/2506.22880v1)**

> **作者:** Dang Jisheng; Wu Xudong; Wang Bimei; Lv Ning; Chen Jiayu; Jingwen Zhao; Yichu liu; Jizhao Liu; Juncheng Li; Teng Wang
>
> **摘要:** Existing video segmenter and grounder approaches, exemplified by Sa2VA, directly fuse features within segmentation models. This often results in an undesirable entanglement of dynamic visual information and static semantics, thereby degrading segmentation accuracy. To systematically mitigate this issue, we propose DeSa2VA, a decoupling-enhanced prompting scheme integrating text pre-training and a linear decoupling module to address the information processing limitations inherent in SAM-2. Specifically, first, we devise a pre-training paradigm that converts textual ground-truth labels into point-level prompts while generating corresponding text masks. These masks are refined through a hybrid loss function to strengthen the model's semantic grounding capabilities. Next, we employ linear projection to disentangle hidden states that generated by a large language model into distinct textual and visual feature subspaces. Finally, a dynamic mask fusion strategy synergistically combines these decoupled features through triple supervision from predicted text/visual masks and ground-truth annotations. Extensive experiments demonstrate state-of-the-art performance across diverse tasks, including image segmentation, image question answering, video segmentation, and video question answering. Our codes are available at https://github.com/longmalongma/DeSa2VA.
>
---
#### [new 155] Token Activation Map to Visually Explain Multimodal LLMs
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于多模态大模型解释任务，旨在解决MLLMs解释不可靠问题。提出TAM方法，通过因果推理和滤波减少冗余干扰，提升可视化效果。**

- **链接: [http://arxiv.org/pdf/2506.23270v1](http://arxiv.org/pdf/2506.23270v1)**

> **作者:** Yi Li; Hualiang Wang; Xinpeng Ding; Haonan Wang; Xiaomeng Li
>
> **备注:** ICCV2025 Accepted
>
> **摘要:** Multimodal large language models (MLLMs) are broadly empowering various fields. Despite their advancements, the explainability of MLLMs remains less explored, hindering deeper understanding, model credibility, and effective visualization. Unlike conventional vision models (e.g., CNNs, ViTs, CLIP) that produce a single output, MLLMs generate sequences of tokens progressively, where each generated token depends on the previous context. Therefore, earlier context tokens can introduce redundant activations that interfere with the explanation of later tokens beyond their original information. Existing studies often overlook this issue, but our observations reveal that these redundant correlations can significantly hurt the reliability of explanations. To address this, we propose an estimated causal inference method to mitigate the interference of context to achieve high-quality MLLM explanation, with a novel rank Gaussian filter to further reduce activation noises. We term this method Token Activation Map (TAM) to highlight the consideration of interactions between tokens. TAM also indicates that it excels at explaining multiple tokens of MLLM, which is different from the Class Activation Map (CAM) for a single prediction. Our TAM method significantly outperforms existing SoTA methods, showcasing high-quality visualization results that can be utilized for various scenarios, such as object localization, failure case analysis, video visualization, MLLMs visual comparison, and model understanding (e.g., color, shape, action, location, visual reasoning, multi-turn conversation, etc). The code is available atgithub.com/xmed-lab/TAM.
>
---
#### [new 156] Efficient Multi-Crop Saliency Partitioning for Automatic Image Cropping
- **分类: cs.CV**

- **简介: 该论文属于图像裁剪任务，解决多区域裁剪问题。提出一种高效方法，在线性时间内提取多个不重叠的显著区域。**

- **链接: [http://arxiv.org/pdf/2506.22814v1](http://arxiv.org/pdf/2506.22814v1)**

> **作者:** Andrew Hamara; Andrew C. Freeman
>
> **摘要:** Automatic image cropping aims to extract the most visually salient regions while preserving essential composition elements. Traditional saliency-aware cropping methods optimize a single bounding box, making them ineffective for applications requiring multiple disjoint crops. In this work, we extend the Fixed Aspect Ratio Cropping algorithm to efficiently extract multiple non-overlapping crops in linear time. Our approach dynamically adjusts attention thresholds and removes selected crops from consideration without recomputing the entire saliency map. We discuss qualitative results and introduce the potential for future datasets and benchmarks.
>
---
#### [new 157] ReCo: Reminder Composition Mitigates Hallucinations in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，解决模型幻觉问题。通过引入ReCo模块，缓解模型对视觉输入的记忆衰退，提升生成准确性。**

- **链接: [http://arxiv.org/pdf/2506.22636v1](http://arxiv.org/pdf/2506.22636v1)**

> **作者:** Sotirios Panagiotis Chytas; Miso Choi; Hyunwoo J. Kim; Vikas Singh
>
> **摘要:** Vision Language Models (VLMs) show impressive capabilities in integrating and reasoning with both visual and language data. But these models make mistakes. A common finding -- similar to LLMs -- is their tendency to hallucinate, i.e., generate plausible sounding text which is not grounded in the visual input, or at worst, is contradictory. A growing consensus attributes this behavior to an over-reliance on language -- especially as the generation progresses, the model suffers from a ``fading memory effect'' with respect to the provided visual input. We study mechanisms by which this behavior can be controlled. Specifically, using ideas from geometric algebra and relational compositions, we propose the addition of a small, trainable module (named ReCo) on top of any VLM -- no other modification is needed. We show that such a lightweight module is able to mitigate the fading memory effect on three of the most widely used VLMs (InstructBLIP, LlaVA, MiniGPT4), where we see performance improvements on multiple benchmarks. Additionally, we show that our module can be combined with many of the other approaches for reducing hallucination where we achieve improved results for each one.
>
---
#### [new 158] Deep Learning based Joint Geometry and Attribute Up-sampling for Large-Scale Colored Point Clouds
- **分类: cs.CV**

- **简介: 该论文属于点云上采样任务，旨在提升大尺度彩色点云的几何与属性质量。通过提出JGAU方法，联合优化几何与属性上采样，显著提升了重建效果。**

- **链接: [http://arxiv.org/pdf/2506.22749v1](http://arxiv.org/pdf/2506.22749v1)**

> **作者:** Yun Zhang; Feifan Chen; Na Li; Zhiwei Guo; Xu Wang; Fen Miao; Sam Kwong
>
> **摘要:** Colored point cloud, which includes geometry and attribute components, is a mainstream representation enabling realistic and immersive 3D applications. To generate large-scale and denser colored point clouds, we propose a deep learning-based Joint Geometry and Attribute Up-sampling (JGAU) method that learns to model both geometry and attribute patterns while leveraging spatial attribute correlations. First, we establish and release a large-scale dataset for colored point cloud up-sampling called SYSU-PCUD, containing 121 large-scale colored point clouds with diverse geometry and attribute complexities across six categories and four sampling rates. Second, to improve the quality of up-sampled point clouds, we propose a deep learning-based JGAU framework that jointly up-samples geometry and attributes. It consists of a geometry up-sampling network and an attribute up-sampling network, where the latter leverages the up-sampled auxiliary geometry to model neighborhood correlations of the attributes. Third, we propose two coarse attribute up-sampling methods, Geometric Distance Weighted Attribute Interpolation (GDWAI) and Deep Learning-based Attribute Interpolation (DLAI), to generate coarse up-sampled attributes for each point. Then, an attribute enhancement module is introduced to refine these up-sampled attributes and produce high-quality point clouds by further exploiting intrinsic attribute and geometry patterns. Extensive experiments show that the Peak Signal-to-Noise Ratio (PSNR) achieved by the proposed JGAU method is 33.90 decibels, 32.10 decibels, 31.10 decibels, and 30.39 decibels for up-sampling rates of 4 times, 8 times, 12 times, and 16 times, respectively. Compared to state-of-the-art methods, JGAU achieves average PSNR gains of 2.32 decibels, 2.47 decibels, 2.28 decibels, and 2.11 decibels at these four up-sampling rates, demonstrating significant improvement.
>
---
#### [new 159] Interactive Interface For Semantic Segmentation Dataset Synthesis
- **分类: cs.CV**

- **简介: 该论文属于语义分割数据集合成任务，旨在解决真实数据标注成本高、隐私问题。提出SynthLab平台，提供模块化架构和交互式界面，简化数据生成流程。**

- **链接: [http://arxiv.org/pdf/2506.23470v1](http://arxiv.org/pdf/2506.23470v1)**

> **作者:** Ngoc-Do Tran; Minh-Tuan Huynh; Tam V. Nguyen; Minh-Triet Tran; Trung-Nghia Le
>
> **摘要:** The rapid advancement of AI and computer vision has significantly increased the demand for high-quality annotated datasets, particularly for semantic segmentation. However, creating such datasets is resource-intensive, requiring substantial time, labor, and financial investment, and often raises privacy concerns due to the use of real-world data. To mitigate these challenges, we present SynthLab, consisting of a modular platform for visual data synthesis and a user-friendly interface. The modular architecture of SynthLab enables easy maintenance, scalability with centralized updates, and seamless integration of new features. Each module handles distinct aspects of computer vision tasks, enhancing flexibility and adaptability. Meanwhile, its interactive, user-friendly interface allows users to quickly customize their data pipelines through drag-and-drop actions. Extensive user studies involving a diverse range of users across different ages, professions, and expertise levels, have demonstrated flexible usage, and high accessibility of SynthLab, enabling users without deep technical expertise to harness AI for real-world applications.
>
---
#### [new 160] PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View
- **分类: cs.CV**

- **简介: 该论文属于全景光流任务，旨在解决球面投影导致的极区畸变问题。提出PriOr-Flow框架，结合正交视图提升光流估计精度。**

- **链接: [http://arxiv.org/pdf/2506.23897v1](http://arxiv.org/pdf/2506.23897v1)**

> **作者:** Longliang Liu; Miaojie Feng; Junda Cheng; Jijun Xiang; Xuan Zhu; Xin Yang
>
> **备注:** 11 pages
>
> **摘要:** Panoramic optical flow enables a comprehensive understanding of temporal dynamics across wide fields of view. However, severe distortions caused by sphere-to-plane projections, such as the equirectangular projection (ERP), significantly degrade the performance of conventional perspective-based optical flow methods, especially in polar regions. To address this challenge, we propose PriOr-Flow, a novel dual-branch framework that leverages the low-distortion nature of the orthogonal view to enhance optical flow estimation in these regions. Specifically, we introduce the Dual-Cost Collaborative Lookup (DCCL) operator, which jointly retrieves correlation information from both the primitive and orthogonal cost volumes, effectively mitigating distortion noise during cost volume construction. Furthermore, our Ortho-Driven Distortion Compensation (ODDC) module iteratively refines motion features from both branches, further suppressing polar distortions. Extensive experiments demonstrate that PriOr-Flow is compatible with various perspective-based iterative optical flow methods and consistently achieves state-of-the-art performance on publicly available panoramic optical flow datasets, setting a new benchmark for wide-field motion estimation. The code is publicly available at: https://github.com/longliangLiu/PriOr-Flow.
>
---
#### [new 161] A Unified Framework for Stealthy Adversarial Generation via Latent Optimization and Transferability Enhancement
- **分类: cs.CV**

- **简介: 该论文属于对抗样本生成任务，旨在提升扩散模型生成对抗样本的泛化能力与迁移性。通过融合传统策略，构建统一框架，有效应用于深度伪造检测等任务。**

- **链接: [http://arxiv.org/pdf/2506.23676v1](http://arxiv.org/pdf/2506.23676v1)**

> **作者:** Gaozheng Pei; Ke Ma; Dongpeng Zhang; Chengzhi Sun; Qianqian Xu; Qingming Huang
>
> **摘要:** Due to their powerful image generation capabilities, diffusion-based adversarial example generation methods through image editing are rapidly gaining popularity. However, due to reliance on the discriminative capability of the diffusion model, these diffusion-based methods often struggle to generalize beyond conventional image classification tasks, such as in Deepfake detection. Moreover, traditional strategies for enhancing adversarial example transferability are challenging to adapt to these methods. To address these challenges, we propose a unified framework that seamlessly incorporates traditional transferability enhancement strategies into diffusion model-based adversarial example generation via image editing, enabling their application across a wider range of downstream tasks. Our method won first place in the "1st Adversarial Attacks on Deepfake Detectors: A Challenge in the Era of AI-Generated Media" competition at ACM MM25, which validates the effectiveness of our approach.
>
---
#### [new 162] NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言导航任务，旨在解决复杂环境中导航泛化与适应性问题。提出NavMorph框架，通过自进化世界模型提升环境理解与决策能力。**

- **链接: [http://arxiv.org/pdf/2506.23468v1](http://arxiv.org/pdf/2506.23468v1)**

> **作者:** Xuan Yao; Junyu Gao; Changsheng Xu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Vision-and-Language Navigation in Continuous Environments (VLN-CE) requires agents to execute sequential navigation actions in complex environments guided by natural language instructions. Current approaches often struggle with generalizing to novel environments and adapting to ongoing changes during navigation. Inspired by human cognition, we present NavMorph, a self-evolving world model framework that enhances environmental understanding and decision-making in VLN-CE tasks. NavMorph employs compact latent representations to model environmental dynamics, equipping agents with foresight for adaptive planning and policy refinement. By integrating a novel Contextual Evolution Memory, NavMorph leverages scene-contextual information to support effective navigation while maintaining online adaptability. Extensive experiments demonstrate that our method achieves notable performance improvements on popular VLN-CE benchmarks. Code is available at \href{https://github.com/Feliciaxyao/NavMorph}{this https URL}.
>
---
#### [new 163] CP-Guard: A Unified, Probability-Agnostic, and Adaptive Framework for Malicious Agent Detection and Defense in Multi-Agent Embodied Perception Systems
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于多智能体感知系统的安全防护任务，旨在解决恶意代理检测与防御问题。提出CP-Guard框架，通过共识机制和动态阈值实现有效检测。**

- **链接: [http://arxiv.org/pdf/2506.22890v1](http://arxiv.org/pdf/2506.22890v1)**

> **作者:** Senkang Hu; Yihang Tao; Guowen Xu; Xinyuan Qian; Yiqin Deng; Xianhao Chen; Sam Tak Wu Kwong; Yuguang Fang
>
> **摘要:** Collaborative Perception (CP) has been shown to be a promising technique for multi-agent autonomous driving and multi-agent robotic systems, where multiple agents share their perception information to enhance the overall perception performance and expand the perception range. However, in CP, an ego agent needs to receive messages from its collaborators, which makes it vulnerable to attacks from malicious agents. To address this critical issue, we propose a unified, probability-agnostic, and adaptive framework, namely, CP-Guard, which is a tailored defense mechanism for CP deployed by each agent to accurately detect and eliminate malicious agents in its collaboration network. Our key idea is to enable CP to reach a consensus rather than a conflict against an ego agent's perception results. Based on this idea, we first develop a probability-agnostic sample consensus (PASAC) method to effectively sample a subset of the collaborators and verify the consensus without prior probabilities of malicious agents. Furthermore, we define collaborative consistency loss (CCLoss) for object detection task and bird's eye view (BEV) segmentation task to capture the discrepancy between an ego agent and its collaborators, which is used as a verification criterion for consensus. In addition, we propose online adaptive threshold via dual sliding windows to dynamically adjust the threshold for consensus verification and ensure the reliability of the systems in dynamic environments. Finally, we conduct extensive experiments and demonstrate the effectiveness of our framework. Code will be released at https://github.com/CP-Security/CP-Guard
>
---
#### [new 164] Instant GaussianImage: A Generalizable and Self-Adaptive Image Representation via 2D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于图像表示任务，旨在解决INR资源消耗高和GaussianImage训练慢、适应性差的问题。提出一种通用自适应的2D高斯点云表示方法，提升效率与灵活性。**

- **链接: [http://arxiv.org/pdf/2506.23479v1](http://arxiv.org/pdf/2506.23479v1)**

> **作者:** Zhaojie Zeng; Yuesong Wang; Chao Yang; Tao Guan; Lili Ju
>
> **摘要:** Implicit Neural Representation (INR) has demonstrated remarkable advances in the field of image representation but demands substantial GPU resources. GaussianImage recently pioneered the use of Gaussian Splatting to mitigate this cost, however, the slow training process limits its practicality, and the fixed number of Gaussians per image limits its adaptability to varying information entropy. To address these issues, we propose in this paper a generalizable and self-adaptive image representation framework based on 2D Gaussian Splatting. Our method employs a network to quickly generate a coarse Gaussian representation, followed by minimal fine-tuning steps, achieving comparable rendering quality of GaussianImage while significantly reducing training time. Moreover, our approach dynamically adjusts the number of Gaussian points based on image complexity to further enhance flexibility and efficiency in practice. Experiments on DIV2K and Kodak datasets show that our method matches or exceeds GaussianImage's rendering performance with far fewer iterations and shorter training times. Specifically, our method reduces the training time by up to one order of magnitude while achieving superior rendering performance with the same number of Gaussians.
>
---
#### [new 165] StyleDrive: Towards Driving-Style Aware Benchmarking of End-To-End Autonomous Driving
- **分类: cs.CV; cs.RO; I.4.9**

- **简介: 该论文属于自动驾驶任务，旨在解决E2EAD中个性化缺失的问题。通过构建带驾驶偏好标注的数据集，提出首个个性化评估基准，提升模型与人类驾驶行为的一致性。**

- **链接: [http://arxiv.org/pdf/2506.23982v1](http://arxiv.org/pdf/2506.23982v1)**

> **作者:** Ruiyang Hao; Bowen Jing; Haibao Yu; Zaiqing Nie
>
> **备注:** 14 pages, 4 figures
>
> **摘要:** While personalization has been explored in traditional autonomous driving systems, it remains largely overlooked in end-to-end autonomous driving (E2EAD), despite its growing prominence. This gap is critical, as user-aligned behavior is essential for trust, comfort, and widespread adoption of autonomous vehicles. A core challenge is the lack of large-scale real-world datasets annotated with diverse and fine-grained driving preferences, hindering the development and evaluation of personalized E2EAD models. In this work, we present the first large-scale real-world dataset enriched with annotations capturing diverse driving preferences, establishing a foundation for personalization in E2EAD. We extract static environmental features from real-world road topology and infer dynamic contextual cues using a fine-tuned visual language model (VLM), enabling consistent and fine-grained scenario construction. Based on these scenarios, we derive objective preference annotations through behavioral distribution analysis and rule-based heuristics. To address the inherent subjectivity of driving style, we further employ the VLM to generate subjective annotations by jointly modeling scene semantics and driver behavior. Final high-quality labels are obtained through a human-in-the-loop verification process that fuses both perspectives. Building on this dataset, we propose the first benchmark for evaluating personalized E2EAD models. We assess several state-of-the-art models with and without preference conditioning, demonstrating that incorporating personalized preferences results in behavior more aligned with human driving. Our work lays the foundation for personalized E2EAD by providing a standardized platform to systematically integrate human preferences into data-driven E2EAD systems, catalyzing future research in human-centric autonomy.
>
---
#### [new 166] Spatially Gene Expression Prediction using Dual-Scale Contrastive Learning
- **分类: cs.CV**

- **简介: 该论文属于基因表达预测任务，旨在解决病理图像与基因数据间跨模态关系建模不足的问题。通过引入双尺度对比学习框架NH2ST，提升预测性能。**

- **链接: [http://arxiv.org/pdf/2506.23827v1](http://arxiv.org/pdf/2506.23827v1)**

> **作者:** Mingcheng Qu; Yuncong Wu; Donglin Di; Yue Gao; Tonghua Su; Yang Song; Lei Fan
>
> **备注:** Our paper has been accepted by MICCAI 2025
>
> **摘要:** Spatial transcriptomics (ST) provides crucial insights into tissue micro-environments, but is limited to its high cost and complexity. As an alternative, predicting gene expression from pathology whole slide images (WSI) is gaining increasing attention. However, existing methods typically rely on single patches or a single pathology modality, neglecting the complex spatial and molecular interactions between target and neighboring information (e.g., gene co-expression). This leads to a failure in establishing connections among adjacent regions and capturing intricate cross-modal relationships. To address these issues, we propose NH2ST, a framework that integrates spatial context and both pathology and gene modalities for gene expression prediction. Our model comprises a query branch and a neighbor branch to process paired target patch and gene data and their neighboring regions, where cross-attention and contrastive learning are employed to capture intrinsic associations and ensure alignments between pathology and gene expression. Extensive experiments on six datasets demonstrate that our model consistently outperforms existing methods, achieving over 20% in PCC metrics. Codes are available at https://github.com/MCPathology/NH2ST
>
---
#### [new 167] Peccavi: Visual Paraphrase Attack Safe and Distortion Free Image Watermarking Technique for AI-Generated Images
- **分类: cs.CV**

- **简介: 该论文属于图像水印任务，旨在解决AI生成图像的水印易被视觉改写攻击移除的问题。提出PECCAVI技术，在不改变图像的前提下安全嵌入水印。**

- **链接: [http://arxiv.org/pdf/2506.22960v1](http://arxiv.org/pdf/2506.22960v1)**

> **作者:** Shreyas Dixit; Ashhar Aziz; Shashwat Bajpai; Vasu Sharma; Aman Chadha; Vinija Jain; Amitava Das
>
> **摘要:** A report by the European Union Law Enforcement Agency predicts that by 2026, up to 90 percent of online content could be synthetically generated, raising concerns among policymakers, who cautioned that "Generative AI could act as a force multiplier for political disinformation. The combined effect of generative text, images, videos, and audio may surpass the influence of any single modality." In response, California's Bill AB 3211 mandates the watermarking of AI-generated images, videos, and audio. However, concerns remain regarding the vulnerability of invisible watermarking techniques to tampering and the potential for malicious actors to bypass them entirely. Generative AI-powered de-watermarking attacks, especially the newly introduced visual paraphrase attack, have shown an ability to fully remove watermarks, resulting in a paraphrase of the original image. This paper introduces PECCAVI, the first visual paraphrase attack-safe and distortion-free image watermarking technique. In visual paraphrase attacks, an image is altered while preserving its core semantic regions, termed Non-Melting Points (NMPs). PECCAVI strategically embeds watermarks within these NMPs and employs multi-channel frequency domain watermarking. It also incorporates noisy burnishing to counter reverse-engineering efforts aimed at locating NMPs to disrupt the embedded watermark, thereby enhancing durability. PECCAVI is model-agnostic. All relevant resources and codes will be open-sourced.
>
---
#### [new 168] Improving Token-based Object Detection with Video
- **分类: cs.CV**

- **简介: 该论文属于视频目标检测任务，旨在解决传统方法在训练和推理中的局限性。通过引入可变长度的离散令牌表示和3D跟踪框，提升了检测效果。**

- **链接: [http://arxiv.org/pdf/2506.22562v1](http://arxiv.org/pdf/2506.22562v1)**

> **作者:** Abhineet Singh; Nilanjan Ray
>
> **备注:** Under review for publication in IEEE Access
>
> **摘要:** This paper improves upon the Pix2Seq object detector by extending it for videos. In the process, it introduces a new way to perform end-to-end video object detection that improves upon existing video detectors in two key ways. First, by representing objects as variable-length sequences of discrete tokens, we can succinctly represent widely varying numbers of video objects, with diverse shapes and locations, without having to inject any localization cues in the training process. This eliminates the need to sample the space of all possible boxes that constrains conventional detectors and thus solves the dual problems of loss sparsity during training and heuristics-based postprocessing during inference. Second, it conceptualizes and outputs the video objects as fully integrated and indivisible 3D boxes or tracklets instead of generating image-specific 2D boxes and linking these boxes together to construct the video object, as done in most conventional detectors. This allows it to scale effortlessly with available computational resources by simply increasing the length of the video subsequence that the network takes as input, even generalizing to multi-object tracking if the subsequence can span the entire video. We compare our video detector with the baseline Pix2Seq static detector on several datasets and demonstrate consistent improvement, although with strong signs of being bottlenecked by our limited computational resources. We also compare it with several video detectors on UA-DETRAC to show that it is competitive with the current state of the art even with the computational bottleneck. We make our code and models publicly available.
>
---
#### [new 169] Mask-aware Text-to-Image Retrieval: Referring Expression Segmentation Meets Cross-modal Retrieval
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出MaTIR任务，融合文本到图像检索与指代表达分割，解决传统方法缺乏解释性和计算成本高的问题。通过两阶段框架提升检索与分割效果。**

- **链接: [http://arxiv.org/pdf/2506.22864v1](http://arxiv.org/pdf/2506.22864v1)**

> **作者:** Li-Cheng Shen; Jih-Kang Hsieh; Wei-Hua Li; Chu-Song Chen
>
> **备注:** ICMR 2025
>
> **摘要:** Text-to-image retrieval (TIR) aims to find relevant images based on a textual query, but existing approaches are primarily based on whole-image captions and lack interpretability. Meanwhile, referring expression segmentation (RES) enables precise object localization based on natural language descriptions but is computationally expensive when applied across large image collections. To bridge this gap, we introduce Mask-aware TIR (MaTIR), a new task that unifies TIR and RES, requiring both efficient image search and accurate object segmentation. To address this task, we propose a two-stage framework, comprising a first stage for segmentation-aware image retrieval and a second stage for reranking and object grounding with a multimodal large language model (MLLM). We leverage SAM 2 to generate object masks and Alpha-CLIP to extract region-level embeddings offline at first, enabling effective and scalable online retrieval. Secondly, MLLM is used to refine retrieval rankings and generate bounding boxes, which are matched to segmentation masks. We evaluate our approach on COCO and D$^3$ datasets, demonstrating significant improvements in both retrieval accuracy and segmentation quality over previous methods.
>
---
#### [new 170] Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于生成模型任务，旨在解决扩散模型计算成本高的问题。通过引入Modulated Diffusion框架，实现量化加速并保持生成质量。**

- **链接: [http://arxiv.org/pdf/2506.22463v1](http://arxiv.org/pdf/2506.22463v1)**

> **作者:** Weizhi Gao; Zhichao Hou; Junqi Yin; Feiyi Wang; Linyu Peng; Xiaorui Liu
>
> **备注:** 26 pages, accepted by ICML 2025
>
> **摘要:** Diffusion models have emerged as powerful generative models, but their high computation cost in iterative sampling remains a significant bottleneck. In this work, we present an in-depth and insightful study of state-of-the-art acceleration techniques for diffusion models, including caching and quantization, revealing their limitations in computation error and generation quality. To break these limits, this work introduces Modulated Diffusion (MoDiff), an innovative, rigorous, and principled framework that accelerates generative modeling through modulated quantization and error compensation. MoDiff not only inherents the advantages of existing caching and quantization methods but also serves as a general framework to accelerate all diffusion models. The advantages of MoDiff are supported by solid theoretical insight and analysis. In addition, extensive experiments on CIFAR-10 and LSUN demonstrate that MoDiff significant reduces activation quantization from 8 bits to 3 bits without performance degradation in post-training quantization (PTQ). Our code implementation is available at https://github.com/WeizhiGao/MoDiff.
>
---
#### [new 171] Preserve Anything: Controllable Image Synthesis with Object Preservation
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决对象保留、语义一致性和场景控制问题。提出一种新方法，提升图像质量和用户控制能力。**

- **链接: [http://arxiv.org/pdf/2506.22531v1](http://arxiv.org/pdf/2506.22531v1)**

> **作者:** Prasen Kumar Sharma; Neeraj Matiyali; Siddharth Srivastava; Gaurav Sharma
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** We introduce \textit{Preserve Anything}, a novel method for controlled image synthesis that addresses key limitations in object preservation and semantic consistency in text-to-image (T2I) generation. Existing approaches often fail (i) to preserve multiple objects with fidelity, (ii) maintain semantic alignment with prompts, or (iii) provide explicit control over scene composition. To overcome these challenges, the proposed method employs an N-channel ControlNet that integrates (i) object preservation with size and placement agnosticism, color and detail retention, and artifact elimination, (ii) high-resolution, semantically consistent backgrounds with accurate shadows, lighting, and prompt adherence, and (iii) explicit user control over background layouts and lighting conditions. Key components of our framework include object preservation and background guidance modules, enforcing lighting consistency and a high-frequency overlay module to retain fine details while mitigating unwanted artifacts. We introduce a benchmark dataset consisting of 240K natural images filtered for aesthetic quality and 18K 3D-rendered synthetic images with metadata such as lighting, camera angles, and object relationships. This dataset addresses the deficiencies of existing benchmarks and allows a complete evaluation. Empirical results demonstrate that our method achieves state-of-the-art performance, significantly improving feature-space fidelity (FID 15.26) and semantic alignment (CLIP-S 32.85) while maintaining competitive aesthetic quality. We also conducted a user study to demonstrate the efficacy of the proposed work on unseen benchmark and observed a remarkable improvement of $\sim25\%$, $\sim19\%$, $\sim13\%$, and $\sim14\%$ in terms of prompt alignment, photorealism, the presence of AI artifacts, and natural aesthetics over existing works.
>
---
#### [new 172] CaO$_2$: Rectifying Inconsistencies in Diffusion-Based Dataset Distillation
- **分类: cs.CV**

- **简介: 该论文属于数据蒸馏任务，解决扩散模型在数据蒸馏中的目标与条件不一致问题，提出CaO₂框架提升性能。**

- **链接: [http://arxiv.org/pdf/2506.22637v1](http://arxiv.org/pdf/2506.22637v1)**

> **作者:** Haoxuan Wang; Zhenghao Zhao; Junyi Wu; Yuzhang Shang; Gaowen Liu; Yan Yan
>
> **备注:** ICCV 2025. Code is available at https://github.com/hatchetProject/CaO2
>
> **摘要:** The recent introduction of diffusion models in dataset distillation has shown promising potential in creating compact surrogate datasets for large, high-resolution target datasets, offering improved efficiency and performance over traditional bi-level/uni-level optimization methods. However, current diffusion-based dataset distillation approaches overlook the evaluation process and exhibit two critical inconsistencies in the distillation process: (1) Objective Inconsistency, where the distillation process diverges from the evaluation objective, and (2) Condition Inconsistency, leading to mismatches between generated images and their corresponding conditions. To resolve these issues, we introduce Condition-aware Optimization with Objective-guided Sampling (CaO$_2$), a two-stage diffusion-based framework that aligns the distillation process with the evaluation objective. The first stage employs a probability-informed sample selection pipeline, while the second stage refines the corresponding latent representations to improve conditional likelihood. CaO$_2$ achieves state-of-the-art performance on ImageNet and its subsets, surpassing the best-performing baselines by an average of 2.3% accuracy.
>
---
#### [new 173] Pruning by Block Benefit: Exploring the Properties of Vision Transformer Blocks during Domain Adaptation
- **分类: cs.CV**

- **简介: 该论文属于模型压缩任务，解决域适应中剪枝效果不佳的问题。提出P3B方法，通过块级贡献评估进行参数资源分配，提升剪枝效果。**

- **链接: [http://arxiv.org/pdf/2506.23675v1](http://arxiv.org/pdf/2506.23675v1)**

> **作者:** Patrick Glandorf; Bodo Rosenhahn
>
> **备注:** ICCV'25 Workshops
>
> **摘要:** Vision Transformer have set new benchmarks in several tasks, but these models come with the lack of high computational costs which makes them impractical for resource limited hardware. Network pruning reduces the computational complexity by removing less important operations while maintaining performance. However, pruning a model on an unseen data domain, leads to a misevaluation of weight significance, resulting in suboptimal resource assignment. In this work, we find that task-sensitive layers initially fail to improve the feature representation on downstream tasks, leading to performance loss for early pruning decisions. To address this problem, we introduce Pruning by Block Benefit (P3B), a pruning method that utilizes the relative contribution on block level to globally assign parameter resources. P3B identifies low-impact components to reduce parameter allocation while preserving critical ones. Classical pruning mask optimization struggles to reactivate zero-mask-elements. In contrast, P3B sets a layerwise keep ratio based on global performance metrics, ensuring the reactivation of late-converging blocks. We show in extensive experiments that P3B is a state of the art pruning method with most noticeable gains in transfer learning tasks. Notably, P3B is able to conserve high performance, even in high sparsity regimes of 70% parameter reduction while only losing 0.64% in accuracy.
>
---
#### [new 174] Empowering Small VLMs to Think with Dynamic Memorization and Exploration
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决小规模模型思考能力不足的问题。通过动态选择记忆与探索模式，提升模型性能与可靠性。**

- **链接: [http://arxiv.org/pdf/2506.23061v1](http://arxiv.org/pdf/2506.23061v1)**

> **作者:** Jiazhen Liu; Yuchuan Deng; Long Chen
>
> **摘要:** Empowering Small-scale Vision-Language Models (SVLMs) with reliable thinking capabilities remains fundamentally challenging due to their limited parameter capacity and weak instruction-following abilities. Existing training paradigms, including Supervised Fine-Tuning (SFT) and Reinforcement Learning with Verifiable Reward (RLVR), impose substantial demands on the base VLM, exceeding the capabilities of SVLMs. Consequently, directly applying these paradigms to SVLMs often suffers from severe pseudo thinking traces and advantage collapse, ultimately undermining both thinking reliability and task performance. A natural solution is to combine SFT and RLVR, leveraging their complementarity to reduce the dependence on model capacity. However, the widely adopted two-stage training paradigm still performs poorly on SVLMs, as their tendency toward sub-optimal convergence hinders the trade-off and limits the benefits of the combination. To address this, we propose DyME, a novel training paradigm that Dynamically selects between Memorization (via SFT) and Exploration (via RLVR) modes at each optimization step, ensuring that every update contributes to the trade-off. Extensive experiments across diverse domains demonstrate that DyME consistently achieves this balance, and thus delivers substantial performance improvements. These results establish DyME as a practical and effective solution for empowering SVLMs with reliable thinking capabilities. GitHub: https://github.com/HKUST-LongGroup/DyME
>
---
#### [new 175] HiNeuS: High-fidelity Neural Surface Mitigating Low-texture and Reflective Ambiguity
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于神经表面重建任务，旨在解决低纹理和反射模糊问题。通过引入统一框架，提升几何精度与光照一致性。**

- **链接: [http://arxiv.org/pdf/2506.23854v1](http://arxiv.org/pdf/2506.23854v1)**

> **作者:** Yida Wang; Xueyang Zhang; Kun Zhan; Peng Jia; Xianpeng Lang
>
> **备注:** Published in International Conference on Computer Vision (ICCV) 2025
>
> **摘要:** Neural surface reconstruction faces persistent challenges in reconciling geometric fidelity with photometric consistency under complex scene conditions. We present HiNeuS, a unified framework that holistically addresses three core limitations in existing approaches: multi-view radiance inconsistency, missing keypoints in textureless regions, and structural degradation from over-enforced Eikonal constraints during joint optimization. To resolve these issues through a unified pipeline, we introduce: 1) Differential visibility verification through SDF-guided ray tracing, resolving reflection ambiguities via continuous occlusion modeling; 2) Planar-conformal regularization via ray-aligned geometry patches that enforce local surface coherence while preserving sharp edges through adaptive appearance weighting; and 3) Physically-grounded Eikonal relaxation that dynamically modulates geometric constraints based on local radiance gradients, enabling detail preservation without sacrificing global regularity. Unlike prior methods that handle these aspects through sequential optimizations or isolated modules, our approach achieves cohesive integration where appearance-geometry constraints evolve synergistically throughout training. Comprehensive evaluations across synthetic and real-world datasets demonstrate state-of-the-art performance, including a 21.4% reduction in Chamfer distance over reflection-aware baselines and 2.32 dB PSNR improvement against neural rendering counterparts. Qualitative analyses reveal superior capability in recovering specular instruments, urban layouts with centimeter-scale infrastructure, and low-textured surfaces without local patch collapse. The method's generalizability is further validated through successful application to inverse rendering tasks, including material decomposition and view-consistent relighting.
>
---
#### [new 176] GViT: Representing Images as Gaussians for Visual Recognition
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像分类任务，旨在用2D高斯表示替代传统像素或补丁输入，提升视觉识别性能。**

- **链接: [http://arxiv.org/pdf/2506.23532v1](http://arxiv.org/pdf/2506.23532v1)**

> **作者:** Jefferson Hernandez; Ruozhen He; Guha Balakrishnan; Alexander C. Berg; Vicente Ordonez
>
> **摘要:** We introduce GVIT, a classification framework that abandons conventional pixel or patch grid input representations in favor of a compact set of learnable 2D Gaussians. Each image is encoded as a few hundred Gaussians whose positions, scales, orientations, colors, and opacities are optimized jointly with a ViT classifier trained on top of these representations. We reuse the classifier gradients as constructive guidance, steering the Gaussians toward class-salient regions while a differentiable renderer optimizes an image reconstruction loss. We demonstrate that by 2D Gaussian input representations coupled with our GVIT guidance, using a relatively standard ViT architecture, closely matches the performance of a traditional patch-based ViT, reaching a 76.9% top-1 accuracy on Imagenet-1k using a ViT-B architecture.
>
---
#### [new 177] Part Segmentation and Motion Estimation for Articulated Objects with Dynamic 3D Gaussians
- **分类: cs.CV**

- **简介: 该论文属于关节物体运动分析任务，解决部分分割与运动估计问题。通过动态3D高斯表示，实现无需点对应的方法，提升鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.22718v1](http://arxiv.org/pdf/2506.22718v1)**

> **作者:** Jun-Jee Chao; Qingyuan Jiang; Volkan Isler
>
> **摘要:** Part segmentation and motion estimation are two fundamental problems for articulated object motion analysis. In this paper, we present a method to solve these two problems jointly from a sequence of observed point clouds of a single articulated object. The main challenge in our problem setting is that the point clouds are not assumed to be generated by a fixed set of moving points. Instead, each point cloud in the sequence could be an arbitrary sampling of the object surface at that particular time step. Such scenarios occur when the object undergoes major occlusions, or if the dataset is collected using measurements from multiple sensors asynchronously. In these scenarios, methods that rely on tracking point correspondences are not appropriate. We present an alternative approach based on a compact but effective representation where we represent the object as a collection of simple building blocks modeled as 3D Gaussians. We parameterize the Gaussians with time-dependent rotations, translations, and scales that are shared across all time steps. With our representation, part segmentation can be achieved by building correspondences between the observed points and the Gaussians. Moreover, the transformation of each point across time can be obtained by following the poses of the assigned Gaussian (even when the point is not observed). Experiments show that our method outperforms existing methods that solely rely on finding point correspondences. Additionally, we extend existing datasets to emulate real-world scenarios by considering viewpoint occlusions. We further demonstrate that our method is more robust to missing points as compared to existing approaches on these challenging datasets, even when some parts are completely occluded in some time-steps. Notably, our part segmentation performance outperforms the state-of-the-art method by 13% on point clouds with occlusions.
>
---
#### [new 178] MoMa: Modulating Mamba for Adapting Image Foundation Models to Video Recognition
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决视频动态建模不足的问题。通过引入MoMa框架，结合空间-时间信息，提升视频识别效果。**

- **链接: [http://arxiv.org/pdf/2506.23283v1](http://arxiv.org/pdf/2506.23283v1)**

> **作者:** Yuhuan Yang; Chaofan Ma; Zhenjie Mao; Jiangchao Yao; Ya Zhang; Yanfeng Wang
>
> **备注:** ICML 2025 paper
>
> **摘要:** Video understanding is a complex challenge that requires effective modeling of spatial-temporal dynamics. With the success of image foundation models (IFMs) in image understanding, recent approaches have explored parameter-efficient fine-tuning (PEFT) to adapt IFMs for video. However, most of these methods tend to process spatial and temporal information separately, which may fail to capture the full intricacy of video dynamics. In this paper, we propose MoMa, an efficient adapter framework that achieves full spatial-temporal modeling by integrating Mamba's selective state space modeling into IFMs. We propose a novel SeqMod operation to inject spatial-temporal information into pre-trained IFMs, without disrupting their original features. By incorporating SeqMod into a Divide-and-Modulate architecture, MoMa enhances video understanding while maintaining computational efficiency. Extensive experiments on multiple video benchmarks demonstrate the effectiveness of MoMa, achieving superior performance with reduced computational cost.
>
---
#### [new 179] Point Cloud Compression and Objective Quality Assessment: A Survey
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于点云压缩与质量评估任务，解决3D数据高效存储与传输问题，综述了传统与学习方法，并分析其优缺点。**

- **链接: [http://arxiv.org/pdf/2506.22902v1](http://arxiv.org/pdf/2506.22902v1)**

> **作者:** Yiling Xu; Yujie Zhang; Shuting Xia; Kaifa Yang; He Huang; Ziyu Shan; Wenjie Huang; Qi Yang; Le Yang
>
> **摘要:** The rapid growth of 3D point cloud data, driven by applications in autonomous driving, robotics, and immersive environments, has led to criticals demand for efficient compression and quality assessment techniques. Unlike traditional 2D media, point clouds present unique challenges due to their irregular structure, high data volume, and complex attributes. This paper provides a comprehensive survey of recent advances in point cloud compression (PCC) and point cloud quality assessment (PCQA), emphasizing their significance for real-time and perceptually relevant applications. We analyze a wide range of handcrafted and learning-based PCC algorithms, along with objective PCQA metrics. By benchmarking representative methods on emerging datasets, we offer detailed comparisons and practical insights into their strengths and limitations. Despite notable progress, challenges such as enhancing visual fidelity, reducing latency, and supporting multimodal data remain. This survey outlines future directions, including hybrid compression frameworks and advanced feature extraction strategies, to enable more efficient, immersive, and intelligent 3D applications.
>
---
#### [new 180] Dare to Plagiarize? Plagiarized Painting Recognition and Retrieval
- **分类: cs.CV**

- **简介: 该论文属于艺术抄袭检测任务，旨在识别和检索伪造绘画。通过构建数据集并优化模型提升检索效果。**

- **链接: [http://arxiv.org/pdf/2506.23132v1](http://arxiv.org/pdf/2506.23132v1)**

> **作者:** Sophie Zhou; Shu Kong
>
> **备注:** to appear at AVSS'25
>
> **摘要:** Art plagiarism detection plays a crucial role in protecting artists' copyrights and intellectual property, yet it remains a challenging problem in forensic analysis. In this paper, we address the task of recognizing plagiarized paintings and explaining the detected plagarisms by retrieving visually similar authentic artworks. To support this study, we construct a dataset by collecting painting photos and synthesizing plagiarized versions using generative AI, tailored to specific artists' styles. We first establish a baseline approach using off-the-shelf features from the visual foundation model DINOv2 to retrieve the most similar images in the database and classify plagiarism based on a similarity threshold. Surprisingly, this non-learned method achieves a high recognition accuracy of 97.2\% but suffers from low retrieval precision 29.0\% average precision (AP). To improve retrieval quality, we finetune DINOv2 with a metric learning loss using positive and negative sample pairs sampled in the database. The finetuned model greatly improves retrieval performance by 12\% AP over the baseline, though it unexpectedly results in a lower recognition accuracy (92.7\%). We conclude with insightful discussions and outline directions for future research.
>
---
#### [new 181] SemFaceEdit: Semantic Face Editing on Generative Radiance Manifolds
- **分类: cs.CV**

- **简介: 该论文属于人脸编辑任务，旨在解决3D生成图像中局部编辑困难的问题。提出SemFaceEdit方法，在生成辐射流形上实现语义场的精确编辑。**

- **链接: [http://arxiv.org/pdf/2506.22833v1](http://arxiv.org/pdf/2506.22833v1)**

> **作者:** Shashikant Verma; Shanmuganathan Raman
>
> **摘要:** Despite multiple view consistency offered by 3D-aware GAN techniques, the resulting images often lack the capacity for localized editing. In response, generative radiance manifolds emerge as an efficient approach for constrained point sampling within volumes, effectively reducing computational demands and enabling the learning of fine details. This work introduces SemFaceEdit, a novel method that streamlines the appearance and geometric editing process by generating semantic fields on generative radiance manifolds. Utilizing latent codes, our method effectively disentangles the geometry and appearance associated with different facial semantics within the generated image. In contrast to existing methods that can change the appearance of the entire radiance field, our method enables the precise editing of particular facial semantics while preserving the integrity of other regions. Our network comprises two key modules: the Geometry module, which generates semantic radiance and occupancy fields, and the Appearance module, which is responsible for predicting RGB radiance. We jointly train both modules in adversarial settings to learn semantic-aware geometry and appearance descriptors. The appearance descriptors are then conditioned on their respective semantic latent codes by the Appearance Module, facilitating disentanglement and enhanced control. Our experiments highlight SemFaceEdit's superior performance in semantic field-based editing, particularly in achieving improved radiance field disentanglement.
>
---
#### [new 182] MadCLIP: Few-shot Medical Anomaly Detection with CLIP
- **分类: cs.CV**

- **简介: 该论文属于医学异常检测任务，解决少样本下图像和像素级异常分类与分割问题。通过改进CLIP模型，提出双分支结构和SigLIP损失，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2506.23810v1](http://arxiv.org/pdf/2506.23810v1)**

> **作者:** Mahshid Shiri; Cigdem Beyan; Vittorio Murino
>
> **备注:** Accepted to MICCAI 2025 (this version is not peer-reviewed; it is the submitted version). MICCAI proceedings DOI will appear here
>
> **摘要:** An innovative few-shot anomaly detection approach is presented, leveraging the pre-trained CLIP model for medical data, and adapting it for both image-level anomaly classification (AC) and pixel-level anomaly segmentation (AS). A dual-branch design is proposed to separately capture normal and abnormal features through learnable adapters in the CLIP vision encoder. To improve semantic alignment, learnable text prompts are employed to link visual features. Furthermore, SigLIP loss is applied to effectively handle the many-to-one relationship between images and unpaired text prompts, showcasing its adaptation in the medical field for the first time. Our approach is validated on multiple modalities, demonstrating superior performance over existing methods for AC and AS, in both same-dataset and cross-dataset evaluations. Unlike prior work, it does not rely on synthetic data or memory banks, and an ablation study confirms the contribution of each component. The code is available at https://github.com/mahshid1998/MadCLIP.
>
---
#### [new 183] LH2Face: Loss function for Hard High-quality Face
- **分类: cs.CV**

- **简介: 该论文属于人脸识别任务，旨在解决硬样本识别问题。提出LH2Face损失函数，结合vMF分布和自适应边缘，提升高质人脸识别准确率。**

- **链接: [http://arxiv.org/pdf/2506.23555v1](http://arxiv.org/pdf/2506.23555v1)**

> **作者:** Fan Xie; Pan Cao
>
> **摘要:** In current practical face authentication systems, most face recognition (FR) algorithms are based on cosine similarity with softmax classification. Despite its reliable classification performance, this method struggles with hard samples. A popular strategy to improve FR performance is incorporating angular or cosine margins. However, it does not take face quality or recognition hardness into account, simply increasing the margin value and thus causing an overly uniform training strategy. To address this problem, a novel loss function is proposed, named Loss function for Hard High-quality Face (LH2Face). Firstly, a similarity measure based on the von Mises-Fisher (vMF) distribution is stated, specifically focusing on the logarithm of the Probability Density Function (PDF), which represents the distance between a probability distribution and a vector. Then, an adaptive margin-based multi-classification method using softmax, called the Uncertainty-Aware Margin Function, is implemented in the article. Furthermore, proxy-based loss functions are used to apply extra constraints between the proxy and sample to optimize their representation space distribution. Finally, a renderer is constructed that optimizes FR through face reconstruction and vice versa. Our LH2Face is superior to similiar schemes on hard high-quality face datasets, achieving 49.39% accuracy on the IJB-B dataset, which surpasses the second-place method by 2.37%.
>
---
#### [new 184] Concept Pinpoint Eraser for Text-to-image Diffusion Models via Residual Attention Gate
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于文本生成图像中的概念擦除任务，旨在删除特定概念同时保留其他内容。工作提出CPE框架，通过非线性残差注意力门实现精准擦除。**

- **链接: [http://arxiv.org/pdf/2506.22806v1](http://arxiv.org/pdf/2506.22806v1)**

> **作者:** Byung Hyun Lee; Sungjin Lim; Seunggyu Lee; Dong Un Kang; Se Young Chun
>
> **摘要:** Remarkable progress in text-to-image diffusion models has brought a major concern about potentially generating images on inappropriate or trademarked concepts. Concept erasing has been investigated with the goals of deleting target concepts in diffusion models while preserving other concepts with minimal distortion. To achieve these goals, recent concept erasing methods usually fine-tune the cross-attention layers of diffusion models. In this work, we first show that merely updating the cross-attention layers in diffusion models, which is mathematically equivalent to adding \emph{linear} modules to weights, may not be able to preserve diverse remaining concepts. Then, we propose a novel framework, dubbed Concept Pinpoint Eraser (CPE), by adding \emph{nonlinear} Residual Attention Gates (ResAGs) that selectively erase (or cut) target concepts while safeguarding remaining concepts from broad distributions by employing an attention anchoring loss to prevent the forgetting. Moreover, we adversarially train CPE with ResAG and learnable text embeddings in an iterative manner to maximize erasing performance and enhance robustness against adversarial attacks. Extensive experiments on the erasure of celebrities, artistic styles, and explicit contents demonstrated that the proposed CPE outperforms prior arts by keeping diverse remaining concepts while deleting the target concepts with robustness against attack prompts. Code is available at https://github.com/Hyun1A/CPE
>
---
#### [new 185] Region-Aware CAM: High-Resolution Weakly-Supervised Defect Segmentation via Salient Region Perception
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于工业缺陷分割任务，解决弱监督下高精度分割问题。提出区域感知CAM和伪标签训练方法，提升分割精度。**

- **链接: [http://arxiv.org/pdf/2506.22866v1](http://arxiv.org/pdf/2506.22866v1)**

> **作者:** Hang-Cheng Dong; Lu Zou; Bingguo Liu; Dong Ye; Guodong Liu
>
> **摘要:** Surface defect detection plays a critical role in industrial quality inspection. Recent advances in artificial intelligence have significantly enhanced the automation level of detection processes. However, conventional semantic segmentation and object detection models heavily rely on large-scale annotated datasets, which conflicts with the practical requirements of defect detection tasks. This paper proposes a novel weakly supervised semantic segmentation framework comprising two key components: a region-aware class activation map (CAM) and pseudo-label training. To address the limitations of existing CAM methods, especially low-resolution thermal maps, and insufficient detail preservation, we introduce filtering-guided backpropagation (FGBP), which refines target regions by filtering gradient magnitudes to identify areas with higher relevance to defects. Building upon this, we further develop a region-aware weighted module to enhance spatial precision. Finally, pseudo-label segmentation is implemented to refine the model's performance iteratively. Comprehensive experiments on industrial defect datasets demonstrate the superiority of our method. The proposed framework effectively bridges the gap between weakly supervised learning and high-precision defect segmentation, offering a practical solution for resource-constrained industrial scenarios.
>
---
#### [new 186] GroundingDINO-US-SAM: Text-Prompted Multi-Organ Segmentation in Ultrasound with LoRA-Tuned Vision-Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，旨在解决超声图像中多器官准确分割的问题。通过融合视觉语言模型与SAM2，利用LoRA微调提升泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.23903v1](http://arxiv.org/pdf/2506.23903v1)**

> **作者:** Hamza Rasaee; Taha Koleilat; Hassan Rivaz
>
> **备注:** 11 pages, 3 figures, 6 figures
>
> **摘要:** Accurate and generalizable object segmentation in ultrasound imaging remains a significant challenge due to anatomical variability, diverse imaging protocols, and limited annotated data. In this study, we propose a prompt-driven vision-language model (VLM) that integrates Grounding DINO with SAM2 to enable object segmentation across multiple ultrasound organs. A total of 18 public ultrasound datasets, encompassing the breast, thyroid, liver, prostate, kidney, and paraspinal muscle, were utilized. These datasets were divided into 15 for fine-tuning and validation of Grounding DINO using Low Rank Adaptation (LoRA) to the ultrasound domain, and 3 were held out entirely for testing to evaluate performance in unseen distributions. Comprehensive experiments demonstrate that our approach outperforms state-of-the-art segmentation methods, including UniverSeg, MedSAM, MedCLIP-SAM, BiomedParse, and SAMUS on most seen datasets while maintaining strong performance on unseen datasets without additional fine-tuning. These results underscore the promise of VLMs in scalable and robust ultrasound image analysis, reducing dependence on large, organ-specific annotated datasets. We will publish our code on code.sonography.ai after acceptance.
>
---
#### [new 187] Spurious-Aware Prototype Refinement for Reliable Out-of-Distribution Detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于机器学习中的OOD检测任务，旨在解决模型因虚假相关性导致的检测不准确问题。提出SPROD方法，通过优化类别原型提升检测可靠性。**

- **链接: [http://arxiv.org/pdf/2506.23881v1](http://arxiv.org/pdf/2506.23881v1)**

> **作者:** Reihaneh Zohrabi; Hosein Hasani; Mahdieh Soleymani Baghshah; Anna Rohrbach; Marcus Rohrbach; Mohammad Hossein Rohban
>
> **摘要:** Out-of-distribution (OOD) detection is crucial for ensuring the reliability and safety of machine learning models in real-world applications, where they frequently face data distributions unseen during training. Despite progress, existing methods are often vulnerable to spurious correlations that mislead models and compromise robustness. To address this, we propose SPROD, a novel prototype-based OOD detection approach that explicitly addresses the challenge posed by unknown spurious correlations. Our post-hoc method refines class prototypes to mitigate bias from spurious features without additional data or hyperparameter tuning, and is broadly applicable across diverse backbones and OOD detection settings. We conduct a comprehensive spurious correlation OOD detection benchmarking, comparing our method against existing approaches and demonstrating its superior performance across challenging OOD datasets, such as CelebA, Waterbirds, UrbanCars, Spurious Imagenet, and the newly introduced Animals MetaCoCo. On average, SPROD improves AUROC by 4.7% and FPR@95 by 9.3% over the second best.
>
---
#### [new 188] PointSSIM: A novel low dimensional resolution invariant image-to-image comparison metric
- **分类: cs.CV**

- **简介: 该论文属于图像比较任务，旨在解决不同分辨率二值图像的结构分析问题。通过提取关键点并构建特征向量进行比较，提出PointSSIM方法。**

- **链接: [http://arxiv.org/pdf/2506.23833v1](http://arxiv.org/pdf/2506.23833v1)**

> **作者:** Oscar Ovanger; Ragnar Hauge; Jacob Skauvold; Michael J. Pyrcz; Jo Eidsvik
>
> **备注:** 13 pages, 20 figures
>
> **摘要:** This paper presents PointSSIM, a novel low-dimensional image-to-image comparison metric that is resolution invariant. Drawing inspiration from the structural similarity index measure and mathematical morphology, PointSSIM enables robust comparison across binary images of varying resolutions by transforming them into marked point pattern representations. The key features of the image, referred to as anchor points, are extracted from binary images by identifying locally adaptive maxima from the minimal distance transform. Image comparisons are then performed using a summary vector, capturing intensity, connectivity, complexity, and structural attributes. Results show that this approach provides an efficient and reliable method for image comparison, particularly suited to applications requiring structural analysis across different resolutions.
>
---
#### [new 189] From Sight to Insight: Unleashing Eye-Tracking in Weakly Supervised Video Salient Object Detection
- **分类: cs.CV**

- **简介: 该论文属于视频显著目标检测任务，旨在利用眼动数据在弱监督下提升检测效果。提出PSE模块和SLQ竞争机制，增强时空特征建模。**

- **链接: [http://arxiv.org/pdf/2506.23519v1](http://arxiv.org/pdf/2506.23519v1)**

> **作者:** Qi Qin; Runmin Cong; Gen Zhan; Yiting Liao; Sam Kwong
>
> **备注:** 15 Pages, 9 Figures
>
> **摘要:** The eye-tracking video saliency prediction (VSP) task and video salient object detection (VSOD) task both focus on the most attractive objects in video and show the result in the form of predictive heatmaps and pixel-level saliency masks, respectively. In practical applications, eye tracker annotations are more readily obtainable and align closely with the authentic visual patterns of human eyes. Therefore, this paper aims to introduce fixation information to assist the detection of video salient objects under weak supervision. On the one hand, we ponder how to better explore and utilize the information provided by fixation, and then propose a Position and Semantic Embedding (PSE) module to provide location and semantic guidance during the feature learning process. On the other hand, we achieve spatiotemporal feature modeling under weak supervision from the aspects of feature selection and feature contrast. A Semantics and Locality Query (SLQ) Competitor with semantic and locality constraints is designed to effectively select the most matching and accurate object query for spatiotemporal modeling. In addition, an Intra-Inter Mixed Contrastive (IIMC) model improves the spatiotemporal modeling capabilities under weak supervision by forming an intra-video and inter-video contrastive learning paradigm. Experimental results on five popular VSOD benchmarks indicate that our model outperforms other competitors on various evaluation metrics.
>
---
#### [new 190] A Closer Look at Conditional Prompt Tuning for Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的微调任务，旨在解决Base-New Tradeoff问题。通过引入TCI条件提示，提出CaPT方法提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.23856v1](http://arxiv.org/pdf/2506.23856v1)**

> **作者:** Ji Zhang; Shihan Wu; Lianli Gao; Jingkuan Song; Nicu Sebe; Heng Tao Shen
>
> **备注:** 18 pages
>
> **摘要:** Despite the great promise of Prompt Tuning (PT) in adapting large Vision-Language Pretrained Models (VLPMs) to downstream tasks, they often struggle to overcome the Base-New Tradeoff (BNT) dilemma: as VLPMs are better tuned to a base task, their ability to generalize to new tasks diminishes. Recent work on conditional PT addresses this problem by replacing static prompts with dynamic Visual Image Information (VII)-conditioned prompts, improving the model's generalization to new tasks to some extent. In this work, we first identify a critical issue with existing conditional PT methods: using VII as the "condition" of prompts yields suboptimal performance, and even random noise-conditioned prompts can outperform the VII-conditioned counterparts. On further analysis, we find that learning dynamic prompts conditioned on Textual Class Information (TCI) is the key to solving the BNT problem. Motivated by this, we then propose Class-adaptive Prompt Tuning (CaPT), which enables fast adaptation of tuned models to new classes by learning TCI-conditioned prompts from base classes. Remarkably, CaPT can be used as a plugin to mitigate the BNT problem for existing unconditional PT schemes. Extensive experiments on 11 datasets show that CaPT consistently improves the performance of five strong unconditional PT baselines with negligible additional computational cost. Additionally, by integrating CaPT with our recently proposed DePT framework, we devise a new conditional PT approach, termed DeCaPT, which outperforms the H ACC of the state-of-the-art conditional PT scheme by 3.49%, averaged over the 11 datasets. Code: https://github.com/Koorye/CaPT.
>
---
#### [new 191] Counting with Confidence: Accurate Pest Monitoring in Water Traps
- **分类: cs.CV**

- **简介: 该论文属于目标计数任务，旨在解决实际场景中无法评估计数结果可靠性的问题。通过多因素分析和回归模型提升计数信心度。**

- **链接: [http://arxiv.org/pdf/2506.22438v1](http://arxiv.org/pdf/2506.22438v1)**

> **作者:** Xumin Gao; Mark Stevens; Grzegorz Cielniak
>
> **备注:** \c{opyright} 20XX the authors. This work has been accepted to IFAC for publication under a Creative Commons Licence CC-BY-NC-ND
>
> **摘要:** Accurate pest population monitoring and tracking their dynamic changes are crucial for precision agriculture decision-making. A common limitation in existing vision-based automatic pest counting research is that models are typically evaluated on datasets with ground truth but deployed in real-world scenarios without assessing the reliability of counting results due to the lack of ground truth. To this end, this paper proposed a method for comprehensively evaluating pest counting confidence in the image, based on information related to counting results and external environmental conditions. First, a pest detection network is used for pest detection and counting, extracting counting result-related information. Then, the pest images undergo image quality assessment, image complexity assessment, and pest distribution uniformity assessment. And the changes in image clarity caused by stirring during image acquisition are quantified by calculating the average gradient magnitude. Notably, we designed a hypothesis-driven multi-factor sensitivity analysis method to select the optimal image quality assessment and image complexity assessment methods. And we proposed an adaptive DBSCAN clustering algorithm for pest distribution uniformity assessment. Finally, the obtained information related to counting results and external environmental conditions is input into a regression model for prediction, resulting in the final pest counting confidence. To the best of our knowledge, this is the first study dedicated to comprehensively evaluating counting confidence in counting tasks, and quantifying the relationship between influencing factors and counting confidence through a model. Experimental results show our method reduces MSE by 31.7% and improves R2 by 15.2% on the pest counting confidence test set, compared to the baseline built primarily on information related to counting results.
>
---
#### [new 192] Revisiting CroPA: A Reproducibility Study and Enhancements for Cross-Prompt Adversarial Transferability in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型的对抗攻击研究，旨在提升跨提示攻击的转移性。通过改进初始化、学习通用扰动和设计新损失函数，增强攻击效果。**

- **链接: [http://arxiv.org/pdf/2506.22982v1](http://arxiv.org/pdf/2506.22982v1)**

> **作者:** Atharv Mittal; Agam Pandey; Amritanshu Tiwari; Sukrit Jindal; Swadesh Swain
>
> **备注:** Accepted to MLRC 2025
>
> **摘要:** Large Vision-Language Models (VLMs) have revolutionized computer vision, enabling tasks such as image classification, captioning, and visual question answering. However, they remain highly vulnerable to adversarial attacks, particularly in scenarios where both visual and textual modalities can be manipulated. In this study, we conduct a comprehensive reproducibility study of "An Image is Worth 1000 Lies: Adversarial Transferability Across Prompts on Vision-Language Models" validating the Cross-Prompt Attack (CroPA) and confirming its superior cross-prompt transferability compared to existing baselines. Beyond replication we propose several key improvements: (1) A novel initialization strategy that significantly improves Attack Success Rate (ASR). (2) Investigate cross-image transferability by learning universal perturbations. (3) A novel loss function targeting vision encoder attention mechanisms to improve generalization. Our evaluation across prominent VLMs -- including Flamingo, BLIP-2, and InstructBLIP as well as extended experiments on LLaVA validates the original results and demonstrates that our improvements consistently boost adversarial effectiveness. Our work reinforces the importance of studying adversarial vulnerabilities in VLMs and provides a more robust framework for generating transferable adversarial examples, with significant implications for understanding the security of VLMs in real-world applications.
>
---
#### [new 193] Aggregating Local Saliency Maps for Semi-Global Explainable Image Classification
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像分类解释任务，旨在解决局部解释难以发现模式的问题，提出SATs方法将局部显著图聚合为半全局见解。**

- **链接: [http://arxiv.org/pdf/2506.23247v1](http://arxiv.org/pdf/2506.23247v1)**

> **作者:** James Hinns; David Martens
>
> **摘要:** Deep learning dominates image classification tasks, yet understanding how models arrive at predictions remains a challenge. Much research focuses on local explanations of individual predictions, such as saliency maps, which visualise the influence of specific pixels on a model's prediction. However, reviewing many of these explanations to identify recurring patterns is infeasible, while global methods often oversimplify and miss important local behaviours. To address this, we propose Segment Attribution Tables (SATs), a method for summarising local saliency explanations into (semi-)global insights. SATs take image segments (such as "eyes" in Chihuahuas) and leverage saliency maps to quantify their influence. These segments highlight concepts the model relies on across instances and reveal spurious correlations, such as reliance on backgrounds or watermarks, even when out-of-distribution test performance sees little change. SATs can explain any classifier for which a form of saliency map can be produced, using segmentation maps that provide named segments. SATs bridge the gap between oversimplified global summaries and overly detailed local explanations, offering a practical tool for analysing and debugging image classifiers.
>
---
#### [new 194] FreeDNA: Endowing Domain Adaptation of Diffusion-Based Dense Prediction with Training-Free Domain Noise Alignment
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于密集预测任务中的域适应问题，旨在提升扩散模型在未见域上的性能。提出一种无需训练的噪声对齐方法，实现域适应。**

- **链接: [http://arxiv.org/pdf/2506.22509v1](http://arxiv.org/pdf/2506.22509v1)**

> **作者:** Hang Xu; Jie Huang; Linjiang Huang; Dong Li; Yidi Liu; Feng Zhao
>
> **备注:** ICCV2025
>
> **摘要:** Domain Adaptation(DA) for dense prediction tasks is an important topic, which enhances the dense prediction model's performance when tested on its unseen domain. Recently, with the development of Diffusion-based Dense Prediction (DDP) models, the exploration of DA designs tailored to this framework is worth exploring, since the diffusion model is effective in modeling the distribution transformation that comprises domain information. In this work, we propose a training-free mechanism for DDP frameworks, endowing them with DA capabilities. Our motivation arises from the observation that the exposure bias (e.g., noise statistics bias) in diffusion brings domain shift, and different domains in conditions of DDP models can also be effectively captured by the noise prediction statistics. Based on this, we propose a training-free Domain Noise Alignment (DNA) approach, which alleviates the variations of noise statistics to domain changes during the diffusion sampling process, thereby achieving domain adaptation. Specifically, when the source domain is available, we directly adopt the DNA method to achieve domain adaptation by aligning the noise statistics of the target domain with those of the source domain. For the more challenging source-free DA, inspired by the observation that regions closer to the source domain exhibit higher confidence meeting variations of sampling noise, we utilize the statistics from the high-confidence regions progressively to guide the noise statistic adjustment during the sampling process. Notably, our method demonstrates the effectiveness of enhancing the DA capability of DDP models across four common dense prediction tasks. Code is available at \href{https://github.com/xuhang07/FreeDNA}{https://github.com/xuhang07/FreeDNA}.
>
---
#### [new 195] UniFuse: A Unified All-in-One Framework for Multi-Modal Medical Image Fusion Under Diverse Degradations and Misalignments
- **分类: cs.CV**

- **简介: 该论文属于多模态医学图像融合任务，旨在解决图像退化和错位问题。提出UniFuse框架，统一处理对齐、修复与融合，提升融合效果。**

- **链接: [http://arxiv.org/pdf/2506.22736v1](http://arxiv.org/pdf/2506.22736v1)**

> **作者:** Dayong Su; Yafei Zhang; Huafeng Li; Jinxing Li; Yu Liu
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Current multimodal medical image fusion typically assumes that source images are of high quality and perfectly aligned at the pixel level. Its effectiveness heavily relies on these conditions and often deteriorates when handling misaligned or degraded medical images. To address this, we propose UniFuse, a general fusion framework. By embedding a degradation-aware prompt learning module, UniFuse seamlessly integrates multi-directional information from input images and correlates cross-modal alignment with restoration, enabling joint optimization of both tasks within a unified framework. Additionally, we design an Omni Unified Feature Representation scheme, which leverages Spatial Mamba to encode multi-directional features and mitigate modality differences in feature alignment. To enable simultaneous restoration and fusion within an All-in-One configuration, we propose a Universal Feature Restoration & Fusion module, incorporating the Adaptive LoRA Synergistic Network (ALSN) based on LoRA principles. By leveraging ALSN's adaptive feature representation along with degradation-type guidance, we enable joint restoration and fusion within a single-stage framework. Compared to staged approaches, UniFuse unifies alignment, restoration, and fusion within a single framework. Experimental results across multiple datasets demonstrate the method's effectiveness and significant advantages over existing approaches.
>
---
#### [new 196] XTransfer: Cross-Modality Model Transfer for Human Sensing with Few Data at the Edge
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于边缘计算中的跨模态模型迁移任务，旨在解决数据稀缺和资源受限下的行人感知问题。提出XTransfer方法，通过模型修复和层重组实现高效、通用的模型迁移。**

- **链接: [http://arxiv.org/pdf/2506.22726v1](http://arxiv.org/pdf/2506.22726v1)**

> **作者:** Yu Zhang; Xi Zhang; Hualin zhou; Xinyuan Chen; Shang Gao; Hong Jia; Jianfei Yang; Yuankai Qi; Tao Gu
>
> **摘要:** Deep learning for human sensing on edge systems offers significant opportunities for smart applications. However, its training and development are hindered by the limited availability of sensor data and resource constraints of edge systems. Current methods that rely on transferring pre-trained models often encounter issues such as modality shift and high resource demands, resulting in substantial accuracy loss, resource overhead, and poor adaptability across different sensing applications. In this paper, we propose XTransfer, a first-of-its-kind method for resource-efficient, modality-agnostic model transfer. XTransfer freely leverages single or multiple pre-trained models and transfers knowledge across different modalities by (i) model repairing that safely repairs modality shift in pre-trained model layers with only few sensor data, and (ii) layer recombining that efficiently searches and recombines layers of interest from source models in a layer-wise manner to create compact models. We benchmark various baselines across diverse human sensing datasets spanning different modalities. Comprehensive results demonstrate that XTransfer achieves state-of-the-art performance on human sensing tasks while significantly reducing the costs of sensor data collection, model training, and edge deployment.
>
---
#### [new 197] CAI: Caption-Sensitive Attention Intervention for Mitigating Object Hallucination in Large Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决对象幻觉问题。通过引入CAI方法，在不增加训练成本的情况下提升模型对视觉信息的感知能力。**

- **链接: [http://arxiv.org/pdf/2506.23590v1](http://arxiv.org/pdf/2506.23590v1)**

> **作者:** Qiming Li; Zekai Ye; Xiaocheng Feng; Weihong Zhong; Libo Qin; Ruihan Chen; Baohang Li; Kui Jiang; Yaowei Wang; Ting Liu; Bing Qin
>
> **摘要:** Although Large Vision-Language Models (LVLMs) have demonstrated powerful capabilities in interpreting visual information, they frequently produce content that deviates from visual information, leading to object hallucination. To tackle this, recent works mostly depend on expensive manual annotations and training cost, or significantly increase inference time. In this work, we observe that LVLMs' attention to visual information is significantly stronger when answering caption queries compared to non-caption queries. Inspired by this phenomenon, we propose Caption-sensitive Attention Intervention (CAI), a training-free, plug-and-play hallucination mitigation method that leverages the attention activation pattern in response to caption queries to enhance LVLMs' visual perception capability. Extensive experimental results across four benchmarks covering both discriminative and generative tasks, demonstrate that CAI achieves state-of-the-art (SOTA) hallucination mitigating performance only with minimal additional inference cost.
>
---
#### [new 198] Autoregressive Denoising Score Matching is a Good Video Anomaly Detector
- **分类: cs.CV**

- **简介: 该论文属于视频异常检测任务，旨在解决现有方法对局部异常不敏感的问题。通过构建噪声条件得分模型和引入场景与运动信息，提升异常检测效果。**

- **链接: [http://arxiv.org/pdf/2506.23282v1](http://arxiv.org/pdf/2506.23282v1)**

> **作者:** Hanwen Zhang; Congqi Cao; Qinyi Lv; Lingtong Min; Yanning Zhang
>
> **摘要:** Video anomaly detection (VAD) is an important computer vision problem. Thanks to the mode coverage capabilities of generative models, the likelihood-based paradigm is catching growing interest, as it can model normal distribution and detect out-of-distribution anomalies. However, these likelihood-based methods are blind to the anomalies located in local modes near the learned distribution. To handle these ``unseen" anomalies, we dive into three gaps uniquely existing in VAD regarding scene, motion and appearance. Specifically, we first build a noise-conditioned score transformer for denoising score matching. Then, we introduce a scene-dependent and motion-aware score function by embedding the scene condition of input sequences into our model and assigning motion weights based on the difference between key frames of input sequences. Next, to solve the problem of blindness in principle, we integrate unaffected visual information via a novel autoregressive denoising score matching mechanism for inference. Through autoregressively injecting intensifying Gaussian noise into the denoised data and estimating the corresponding score function, we compare the denoised data with the original data to get a difference and aggregate it with the score function for an enhanced appearance perception and accumulate the abnormal context. With all three gaps considered, we can compute a more comprehensive anomaly indicator. Experiments on three popular VAD benchmarks demonstrate the state-of-the-art performance of our method.
>
---
#### [new 199] WaRA: Wavelet Low Rank Adaptation
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于参数高效微调任务，旨在解决传统方法忽略局部结构的问题。提出WaRA，利用小波变换实现多分辨率低秩适配，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.24092v1](http://arxiv.org/pdf/2506.24092v1)**

> **作者:** Moein Heidari; Yasamin Medghalchi; Mahdi Khoursha; Reza Rezaeian; Ilker Hacihaliloglu
>
> **备注:** Submitted to BMVC 2025
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) has gained widespread adoption across various applications. Among PEFT techniques, Low-Rank Adaptation (LoRA) and its extensions have emerged as particularly effective, allowing efficient model adaptation while significantly reducing computational overhead. However, existing approaches typically rely on global low-rank factorizations, which overlook local or multi-scale structure, failing to capture complex patterns in the weight updates. To address this, we propose WaRA, a novel PEFT method that leverages wavelet transforms to decompose the weight update matrix into a multi-resolution representation. By performing low-rank factorization in the wavelet domain and reconstructing updates through an inverse transform, WaRA obtains compressed adaptation parameters that harness multi-resolution analysis, enabling it to capture both coarse and fine-grained features while providing greater flexibility and sparser representations than standard LoRA. Through comprehensive experiments and analysis, we demonstrate that WaRA performs superior on diverse vision tasks, including image generation, classification, and semantic segmentation, significantly enhancing generated image quality while reducing computational complexity. Although WaRA was primarily designed for vision tasks, we further showcase its effectiveness in language tasks, highlighting its broader applicability and generalizability. The code is publicly available at \href{GitHub}{https://github.com/moeinheidari7829/WaRA}.
>
---
#### [new 200] Layer Decomposition and Morphological Reconstruction for Task-Oriented Infrared Image Enhancement
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于红外图像增强任务，旨在提升复杂天气下自主驾驶的感知能力。针对低对比度和噪声问题，提出层分解与形态重建方法，有效增强目标信息。**

- **链接: [http://arxiv.org/pdf/2506.23353v1](http://arxiv.org/pdf/2506.23353v1)**

> **作者:** Siyuan Chai; Xiaodong Guo; Tong Liu
>
> **摘要:** Infrared image helps improve the perception capabilities of autonomous driving in complex weather conditions such as fog, rain, and low light. However, infrared image often suffers from low contrast, especially in non-heat-emitting targets like bicycles, which significantly affects the performance of downstream high-level vision tasks. Furthermore, achieving contrast enhancement without amplifying noise and losing important information remains a challenge. To address these challenges, we propose a task-oriented infrared image enhancement method. Our approach consists of two key components: layer decomposition and saliency information extraction. First, we design an layer decomposition method for infrared images, which enhances scene details while preserving dark region features, providing more features for subsequent saliency information extraction. Then, we propose a morphological reconstruction-based saliency extraction method that effectively extracts and enhances target information without amplifying noise. Our method improves the image quality for object detection and semantic segmentation tasks. Extensive experiments demonstrate that our approach outperforms state-of-the-art methods.
>
---
#### [new 201] Evaluation of Geolocation Capabilities of Multimodal Large Language Models and Analysis of Associated Privacy Risks
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于视觉地理定位任务，旨在评估多模态大语言模型的地理定位能力及其隐私风险。研究分析了模型在街景图像定位中的表现，并探讨了相关隐私问题及应对措施。**

- **链接: [http://arxiv.org/pdf/2506.23481v1](http://arxiv.org/pdf/2506.23481v1)**

> **作者:** Xian Zhang; Xiang Cheng
>
> **摘要:** Objectives: The rapid advancement of Multimodal Large Language Models (MLLMs) has significantly enhanced their reasoning capabilities, enabling a wide range of intelligent applications. However, these advancements also raise critical concerns regarding privacy and ethics. MLLMs are now capable of inferring the geographic location of images -- such as those shared on social media or captured from street views -- based solely on visual content, thereby posing serious risks of privacy invasion, including doxxing, surveillance, and other security threats. Methods: This study provides a comprehensive analysis of existing geolocation techniques based on MLLMs. It systematically reviews relevant litera-ture and evaluates the performance of state-of-the-art visual reasoning models on geolocation tasks, particularly in identifying the origins of street view imagery. Results: Empirical evaluation reveals that the most advanced visual large models can successfully localize the origin of street-level imagery with up to $49\%$ accuracy within a 1-kilometer radius. This performance underscores the models' powerful capacity to extract and utilize fine-grained geographic cues from visual data. Conclusions: Building on these findings, the study identifies key visual elements that contribute to suc-cessful geolocation, such as text, architectural styles, and environmental features. Furthermore, it discusses the potential privacy implications associated with MLLM-enabled geolocation and discuss several technical and policy-based coun-termeasures to mitigate associated risks. Our code and dataset are available at https://github.com/zxyl1003/MLLM-Geolocation-Evaluation.
>
---
#### [new 202] Seg-R1: Segmentation Can Be Surprisingly Simple with Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文属于图像分割任务，旨在提升模型的像素级理解能力。通过强化学习方法，无需复杂模型修改即可实现高效分割，并在多个数据集上表现优异。**

- **链接: [http://arxiv.org/pdf/2506.22624v1](http://arxiv.org/pdf/2506.22624v1)**

> **作者:** Zuyao You; Zuxuan Wu
>
> **摘要:** We present Seg-R1, a preliminary exploration of using reinforcement learning (RL) to enhance the pixel-level understanding and reasoning capabilities of large multimodal models (LMMs). Starting with foreground segmentation tasks, specifically camouflaged object detection (COD) and salient object detection (SOD), our approach enables the LMM to generate point and bounding box prompts in the next-token fashion, which are then used to guide SAM2 in producing segmentation masks. We introduce Group Relative Policy Optimization (GRPO) into the segmentation domain, equipping the LMM with pixel-level comprehension through a carefully designed training strategy. Notably, Seg-R1 achieves remarkable performance with purely RL-based training, achieving .873 S-measure on COD10K without complex model modification. Moreover, we found that pure RL training demonstrates strong open-world generalization. Despite being trained solely on foreground segmentation image-mask pairs without text supervision, Seg-R1 achieves impressive zero-shot performance on referring segmentation and reasoning segmentation tasks, with 71.4 cIoU on RefCOCOg test and 56.7 gIoU on ReasonSeg test, outperforming models fully supervised on these datasets.
>
---
#### [new 203] Self-Supervised Contrastive Learning for Multi-Label Images
- **分类: cs.CV**

- **简介: 该论文属于多标签图像的自监督对比学习任务，旨在解决单标签数据依赖和多标签信息利用不足的问题，通过块增强和图像感知损失提升表示学习效果。**

- **链接: [http://arxiv.org/pdf/2506.23156v1](http://arxiv.org/pdf/2506.23156v1)**

> **作者:** Jiale Chen
>
> **摘要:** Self-supervised learning (SSL) has demonstrated its effectiveness in learning representations through comparison methods that align with human intuition. However, mainstream SSL methods heavily rely on high body datasets with single label, such as ImageNet, resulting in intolerable pre-training overhead. Besides, more general multi-label images are frequently overlooked in SSL, despite their potential for richer semantic information and broader applicability in downstream scenarios. Therefore, we tailor the mainstream SSL approach to guarantee excellent representation learning capabilities using fewer multi-label images. Firstly, we propose a block-wise augmentation module aimed at extracting additional potential positive view pairs from multi-label images. Subsequently, an image-aware contrastive loss is devised to establish connections between these views, thereby facilitating the extraction of semantically consistent representations. Comprehensive linear fine-tuning and transfer learning validate the competitiveness of our approach despite challenging sample quality and quantity.
>
---
#### [new 204] Blending Concepts with Text-to-Image Diffusion Models
- **分类: cs.CV**

- **简介: 该论文研究文本到图像扩散模型在概念融合任务中的表现，探索如何在无需训练的情况下将不同概念合成新图像，并测试多种融合方法的效果。**

- **链接: [http://arxiv.org/pdf/2506.23630v1](http://arxiv.org/pdf/2506.23630v1)**

> **作者:** Lorenzo Olearo; Giorgio Longari; Alessandro Raganato; Rafael Peñaloza; Simone Melzi
>
> **备注:** Currently under review
>
> **摘要:** Diffusion models have dramatically advanced text-to-image generation in recent years, translating abstract concepts into high-fidelity images with remarkable ease. In this work, we examine whether they can also blend distinct concepts, ranging from concrete objects to intangible ideas, into coherent new visual entities under a zero-shot framework. Specifically, concept blending merges the key attributes of multiple concepts (expressed as textual prompts) into a single, novel image that captures the essence of each concept. We investigate four blending methods, each exploiting different aspects of the diffusion pipeline (e.g., prompt scheduling, embedding interpolation, or layer-wise conditioning). Through systematic experimentation across diverse concept categories, such as merging concrete concepts, synthesizing compound words, transferring artistic styles, and blending architectural landmarks, we show that modern diffusion models indeed exhibit creative blending capabilities without further training or fine-tuning. Our extensive user study, involving 100 participants, reveals that no single approach dominates in all scenarios: each blending technique excels under certain conditions, with factors like prompt ordering, conceptual distance, and random seed affecting the outcome. These findings highlight the remarkable compositional potential of diffusion models while exposing their sensitivity to seemingly minor input variations.
>
---
#### [new 205] PGOV3D: Open-Vocabulary 3D Semantic Segmentation with Partial-to-Global Curriculum
- **分类: cs.CV**

- **简介: 该论文属于3D语义分割任务，解决开放词汇下3D点云分割效果不足的问题。通过引入部分到全局的训练策略，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.23607v1](http://arxiv.org/pdf/2506.23607v1)**

> **作者:** Shiqi Zhang; Sha Zhang; Jiajun Deng; Yedong Shen; Mingxiao MA; Yanyong Zhang
>
> **摘要:** Existing open-vocabulary 3D semantic segmentation methods typically supervise 3D segmentation models by merging text-aligned features (e.g., CLIP) extracted from multi-view images onto 3D points. However, such approaches treat multi-view images merely as intermediaries for transferring open-vocabulary information, overlooking their rich semantic content and cross-view correspondences, which limits model effectiveness. To address this, we propose PGOV3D, a novel framework that introduces a Partial-to-Global curriculum for improving open-vocabulary 3D semantic segmentation. The key innovation lies in a two-stage training strategy. In the first stage, we pre-train the model on partial scenes that provide dense semantic information but relatively simple geometry. These partial point clouds are derived from multi-view RGB-D inputs via pixel-wise depth projection. To enable open-vocabulary learning, we leverage a multi-modal large language model (MLLM) and a 2D segmentation foundation model to generate open-vocabulary labels for each viewpoint, offering rich and aligned supervision. An auxiliary inter-frame consistency module is introduced to enforce feature consistency across varying viewpoints and enhance spatial understanding. In the second stage, we fine-tune the model on complete scene-level point clouds, which are sparser and structurally more complex. We aggregate the partial vocabularies associated with each scene and generate pseudo labels using the pre-trained model, effectively bridging the semantic gap between dense partial observations and large-scale 3D environments. Extensive experiments on ScanNet, ScanNet200, and S3DIS benchmarks demonstrate that PGOV3D achieves competitive performance in open-vocabulary 3D semantic segmentation.
>
---
#### [new 206] VAP-Diffusion: Enriching Descriptions with MLLMs for Enhanced Medical Image Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像生成任务，旨在解决生成图像缺乏详细描述的问题。通过引入多模态大语言模型，生成丰富属性描述，提升图像质量和多样性。**

- **链接: [http://arxiv.org/pdf/2506.23641v1](http://arxiv.org/pdf/2506.23641v1)**

> **作者:** Peng Huang; Junhu Fu; Bowen Guo; Zeju Li; Yuanyuan Wang; Yi Guo
>
> **摘要:** As the appearance of medical images is influenced by multiple underlying factors, generative models require rich attribute information beyond labels to produce realistic and diverse images. For instance, generating an image of skin lesion with specific patterns demands descriptions that go beyond diagnosis, such as shape, size, texture, and color. However, such detailed descriptions are not always accessible. To address this, we explore a framework, termed Visual Attribute Prompts (VAP)-Diffusion, to leverage external knowledge from pre-trained Multi-modal Large Language Models (MLLMs) to improve the quality and diversity of medical image generation. First, to derive descriptions from MLLMs without hallucination, we design a series of prompts following Chain-of-Thoughts for common medical imaging tasks, including dermatologic, colorectal, and chest X-ray images. Generated descriptions are utilized during training and stored across different categories. During testing, descriptions are randomly retrieved from the corresponding category for inference. Moreover, to make the generator robust to unseen combination of descriptions at the test time, we propose a Prototype Condition Mechanism that restricts test embeddings to be similar to those from training. Experiments on three common types of medical imaging across four datasets verify the effectiveness of VAP-Diffusion.
>
---
#### [new 207] RoboPearls: Editable Video Simulation for Robot Manipulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决真实数据收集成本高和模拟与现实差距大的问题。提出RoboPearls框架，通过视频生成高质量仿真环境并支持多种操作。**

- **链接: [http://arxiv.org/pdf/2506.22756v1](http://arxiv.org/pdf/2506.22756v1)**

> **作者:** Tao Tang; Likui Zhang; Youpeng Wen; Kaidong Zhang; Jia-Wang Bian; xia zhou; Tianyi Yan; Kun Zhan; Peng Jia; Hefeng Wu; Liang Lin; Xiaodan Liang
>
> **备注:** ICCV 2025
>
> **摘要:** The development of generalist robot manipulation policies has seen significant progress, driven by large-scale demonstration data across diverse environments. However, the high cost and inefficiency of collecting real-world demonstrations hinder the scalability of data acquisition. While existing simulation platforms enable controlled environments for robotic learning, the challenge of bridging the sim-to-real gap remains. To address these challenges, we propose RoboPearls, an editable video simulation framework for robotic manipulation. Built on 3D Gaussian Splatting (3DGS), RoboPearls enables the construction of photo-realistic, view-consistent simulations from demonstration videos, and supports a wide range of simulation operators, including various object manipulations, powered by advanced modules like Incremental Semantic Distillation (ISD) and 3D regularized NNFM Loss (3D-NNFM). Moreover, by incorporating large language models (LLMs), RoboPearls automates the simulation production process in a user-friendly manner through flexible command interpretation and execution. Furthermore, RoboPearls employs a vision-language model (VLM) to analyze robotic learning issues to close the simulation loop for performance enhancement. To demonstrate the effectiveness of RoboPearls, we conduct extensive experiments on multiple datasets and scenes, including RLBench, COLOSSEUM, Ego4D, Open X-Embodiment, and a real-world robot, which demonstrate our satisfactory simulation performance.
>
---
#### [new 208] Competitive Distillation: A Simple Learning Strategy for Improving Visual Classification
- **分类: cs.CV**

- **简介: 该论文属于视觉分类任务，旨在解决知识蒸馏中学习方向不明确的问题。提出竞争性蒸馏策略，通过网络间竞争提升整体性能。**

- **链接: [http://arxiv.org/pdf/2506.23285v1](http://arxiv.org/pdf/2506.23285v1)**

> **作者:** Daqian Shi; Xiaolei Diao; Xu Chen; Cédric M. John
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Deep Neural Networks (DNNs) have significantly advanced the field of computer vision. To improve DNN training process, knowledge distillation methods demonstrate their effectiveness in accelerating network training by introducing a fixed learning direction from the teacher network to student networks. In this context, several distillation-based optimization strategies are proposed, e.g., deep mutual learning and self-distillation, as an attempt to achieve generic training performance enhancement through the cooperative training of multiple networks. However, such strategies achieve limited improvements due to the poor understanding of the impact of learning directions among networks across different iterations. In this paper, we propose a novel competitive distillation strategy that allows each network in a group to potentially act as a teacher based on its performance, enhancing the overall learning performance. Competitive distillation organizes a group of networks to perform a shared task and engage in competition, where competitive optimization is proposed to improve the parameter updating process. We further introduce stochastic perturbation in competitive distillation, aiming to motivate networks to induce mutations to achieve better visual representations and global optimum. The experimental results show that competitive distillation achieves promising performance in diverse tasks and datasets.
>
---
#### [new 209] TVG-SLAM: Robust Gaussian Splatting SLAM with Tri-view Geometric Constraints
- **分类: cs.CV**

- **简介: 该论文属于SLAM任务，解决RGB-only系统在复杂环境下的跟踪与建图问题。通过引入三视角几何约束和优化策略，提升系统鲁棒性与精度。**

- **链接: [http://arxiv.org/pdf/2506.23207v1](http://arxiv.org/pdf/2506.23207v1)**

> **作者:** Zhen Tan; Xieyuanli Chen; Lei Feng; Yangbing Ge; Shuaifeng Zhi; Jiaxiong Liu; Dewen Hu
>
> **摘要:** Recent advances in 3D Gaussian Splatting (3DGS) have enabled RGB-only SLAM systems to achieve high-fidelity scene representation. However, the heavy reliance of existing systems on photometric rendering loss for camera tracking undermines their robustness, especially in unbounded outdoor environments with severe viewpoint and illumination changes. To address these challenges, we propose TVG-SLAM, a robust RGB-only 3DGS SLAM system that leverages a novel tri-view geometry paradigm to ensure consistent tracking and high-quality mapping. We introduce a dense tri-view matching module that aggregates reliable pairwise correspondences into consistent tri-view matches, forming robust geometric constraints across frames. For tracking, we propose Hybrid Geometric Constraints, which leverage tri-view matches to construct complementary geometric cues alongside photometric loss, ensuring accurate and stable pose estimation even under drastic viewpoint shifts and lighting variations. For mapping, we propose a new probabilistic initialization strategy that encodes geometric uncertainty from tri-view correspondences into newly initialized Gaussians. Additionally, we design a Dynamic Attenuation of Rendering Trust mechanism to mitigate tracking drift caused by mapping latency. Experiments on multiple public outdoor datasets show that our TVG-SLAM outperforms prior RGB-only 3DGS-based SLAM systems. Notably, in the most challenging dataset, our method improves tracking robustness, reducing the average Absolute Trajectory Error (ATE) by 69.0\% while achieving state-of-the-art rendering quality. The implementation of our method will be released as open-source.
>
---
#### [new 210] Flash-VStream: Efficient Real-Time Understanding for Long Video Streams
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决长视频实时处理效率低的问题。提出Flash-VStream模型，通过设计两种记忆模块提升处理速度与效果。**

- **链接: [http://arxiv.org/pdf/2506.23825v1](http://arxiv.org/pdf/2506.23825v1)**

> **作者:** Haoji Zhang; Yiqin Wang; Yansong Tang; Yong Liu; Jiashi Feng; Xiaojie Jin
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Benefiting from the advances in large language models and cross-modal alignment, existing multimodal large language models have achieved prominent performance in image and short video understanding. However, the understanding of long videos is still challenging, as their long-context nature results in significant computational and memory overhead. Most existing work treats long videos in the same way as short videos, which is inefficient for real-world applications and hard to generalize to even longer videos. To address these issues, we propose Flash-VStream, an efficient video language model capable of processing extremely long videos and responding to user queries in real time. Particularly, we design a Flash Memory module, containing a low-capacity context memory to aggregate long-context temporal information and model the distribution of information density, and a high-capacity augmentation memory to retrieve detailed spatial information based on this distribution. Compared to existing models, Flash-VStream achieves significant reductions in inference latency. Extensive experiments on long video benchmarks and comprehensive video benchmarks, i.e., EgoSchema, MLVU, LVBench, MVBench and Video-MME, demonstrate the state-of-the-art performance and outstanding efficiency of our method. Code is available at https://github.com/IVGSZ/Flash-VStream.
>
---
#### [new 211] Seamless Interaction: Dyadic Audiovisual Motion Modeling and Large-Scale Dataset
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于人机交互任务，旨在提升AI对人际互动的理解与生成能力。研究构建了大规模数据集，并开发了能生成同步动作和表情的模型，以实现更自然的虚拟交互。**

- **链接: [http://arxiv.org/pdf/2506.22554v1](http://arxiv.org/pdf/2506.22554v1)**

> **作者:** Vasu Agrawal; Akinniyi Akinyemi; Kathryn Alvero; Morteza Behrooz; Julia Buffalini; Fabio Maria Carlucci; Joy Chen; Junming Chen; Zhang Chen; Shiyang Cheng; Praveen Chowdary; Joe Chuang; Antony D'Avirro; Jon Daly; Ning Dong; Mark Duppenthaler; Cynthia Gao; Jeff Girard; Martin Gleize; Sahir Gomez; Hongyu Gong; Srivathsan Govindarajan; Brandon Han; Sen He; Denise Hernandez; Yordan Hristov; Rongjie Huang; Hirofumi Inaguma; Somya Jain; Raj Janardhan; Qingyao Jia; Christopher Klaiber; Dejan Kovachev; Moneish Kumar; Hang Li; Yilei Li; Pavel Litvin; Wei Liu; Guangyao Ma; Jing Ma; Martin Ma; Xutai Ma; Lucas Mantovani; Sagar Miglani; Sreyas Mohan; Louis-Philippe Morency; Evonne Ng; Kam-Woh Ng; Tu Anh Nguyen; Amia Oberai; Benjamin Peloquin; Juan Pino; Jovan Popovic; Omid Poursaeed; Fabian Prada; Alice Rakotoarison; Alexander Richard; Christophe Ropers; Safiyyah Saleem; Vasu Sharma; Alex Shcherbyna; Jia Shen; Jie Shen; Anastasis Stathopoulos; Anna Sun; Paden Tomasello; Tuan Tran; Arina Turkatenko; Bo Wan; Chao Wang; Jeff Wang; Mary Williamson; Carleigh Wood; Tao Xiang; Yilin Yang; Julien Yao; Chen Zhang; Jiemin Zhang; Xinyue Zhang; Jason Zheng; Pavlo Zhyzheria; Jan Zikes; Michael Zollhoefer
>
> **摘要:** Human communication involves a complex interplay of verbal and nonverbal signals, essential for conveying meaning and achieving interpersonal goals. To develop socially intelligent AI technologies, it is crucial to develop models that can both comprehend and generate dyadic behavioral dynamics. To this end, we introduce the Seamless Interaction Dataset, a large-scale collection of over 4,000 hours of face-to-face interaction footage from over 4,000 participants in diverse contexts. This dataset enables the development of AI technologies that understand dyadic embodied dynamics, unlocking breakthroughs in virtual agents, telepresence experiences, and multimodal content analysis tools. We also develop a suite of models that utilize the dataset to generate dyadic motion gestures and facial expressions aligned with human speech. These models can take as input both the speech and visual behavior of their interlocutors. We present a variant with speech from an LLM model and integrations with 2D and 3D rendering methods, bringing us closer to interactive virtual agents. Additionally, we describe controllable variants of our motion models that can adapt emotional responses and expressivity levels, as well as generating more semantically-relevant gestures. Finally, we discuss methods for assessing the quality of these dyadic motion models, which are demonstrating the potential for more intuitive and responsive human-AI interactions.
>
---
#### [new 212] LLM-enhanced Action-aware Multi-modal Prompt Tuning for Image-Text Matching
- **分类: cs.CV**

- **简介: 该论文属于图像-文本匹配任务，旨在解决CLIP模型在细粒度动作理解上的不足。通过引入LLM增强的多模态提示调优方法，提升模型对动作和状态的感知能力。**

- **链接: [http://arxiv.org/pdf/2506.23502v1](http://arxiv.org/pdf/2506.23502v1)**

> **作者:** Mengxiao Tian; Xinxiao Wu; Shuo Yang
>
> **备注:** accepted by ICCV 2025
>
> **摘要:** Driven by large-scale contrastive vision-language pre-trained models such as CLIP, recent advancements in the image-text matching task have achieved remarkable success in representation learning. Due to image-level visual-language alignment, CLIP falls short in understanding fine-grained details such as object attributes and spatial relationships between objects. Recent efforts have attempted to compel CLIP to acquire structured visual representations by introducing prompt learning to achieve object-level alignment. While achieving promising results, they still lack the capability to perceive actions, which are crucial for describing the states or relationships between objects. Therefore, we propose to endow CLIP with fine-grained action-level understanding by introducing an LLM-enhanced action-aware multi-modal prompt-tuning method, incorporating the action-related external knowledge generated by large language models (LLMs). Specifically, we design an action triplet prompt and an action state prompt to exploit compositional semantic knowledge and state-related causal knowledge implicitly stored in LLMs. Subsequently, we propose an adaptive interaction module to aggregate attentive visual features conditioned on action-aware prompted knowledge for establishing discriminative and action-aware visual representations, which further improves the performance. Comprehensive experimental results on two benchmark datasets demonstrate the effectiveness of our method.
>
---
#### [new 213] DC-TTA: Divide-and-Conquer Framework for Test-Time Adaptation of Interactive Segmentation
- **分类: cs.CV**

- **简介: 该论文属于交互式分割任务，解决SAM在复杂场景下的适应性问题。提出DC-TTA框架，通过分治策略提升分割精度与效率。**

- **链接: [http://arxiv.org/pdf/2506.23104v1](http://arxiv.org/pdf/2506.23104v1)**

> **作者:** Jihun Kim; Hoyong Kwon; Hyeokjun Kweon; Wooseong Jeong; Kuk-Jin Yoon
>
> **摘要:** Interactive segmentation (IS) allows users to iteratively refine object boundaries with minimal cues, such as positive and negative clicks. While the Segment Anything Model (SAM) has garnered attention in the IS community for its promptable segmentation capabilities, it often struggles in specialized domains or when handling complex scenarios (e.g., camouflaged or multi-part objects). To overcome these challenges, we propose DC-TTA, a novel test-time adaptation (TTA) framework that adapts SAM on a per-sample basis by leveraging user interactions as supervision. Instead of forcing a single model to incorporate all user clicks at once, DC-TTA partitions the clicks into more coherent subsets, each processed independently via TTA with a separated model. This Divide-and-Conquer strategy reduces conflicts among diverse cues and enables more localized updates. Finally, we merge the adapted models to form a unified predictor that integrates the specialized knowledge from each subset. Experimental results across various benchmarks demonstrate that DC-TTA significantly outperforms SAM's zero-shot results and conventional TTA methods, effectively handling complex tasks such as camouflaged object segmentation with fewer interactions and improved accuracy.
>
---
#### [new 214] A Survey on Vision-Language-Action Models for Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决车辆理解指令、推理场景和自主决策的问题。通过综述VLA模型，分析其架构与进展，并指出未来挑战。**

- **链接: [http://arxiv.org/pdf/2506.24044v1](http://arxiv.org/pdf/2506.24044v1)**

> **作者:** Sicong Jiang; Zilin Huang; Kangan Qian; Ziang Luo; Tianze Zhu; Yang Zhong; Yihong Tang; Menglin Kong; Yunlong Wang; Siwen Jiao; Hao Ye; Zihao Sheng; Xin Zhao; Tuopu Wen; Zheng Fu; Sikai Chen; Kun Jiang; Diange Yang; Seongjin Choi; Lijun Sun
>
> **摘要:** The rapid progress of multimodal large language models (MLLM) has paved the way for Vision-Language-Action (VLA) paradigms, which integrate visual perception, natural language understanding, and control within a single policy. Researchers in autonomous driving are actively adapting these methods to the vehicle domain. Such models promise autonomous vehicles that can interpret high-level instructions, reason about complex traffic scenes, and make their own decisions. However, the literature remains fragmented and is rapidly expanding. This survey offers the first comprehensive overview of VLA for Autonomous Driving (VLA4AD). We (i) formalize the architectural building blocks shared across recent work, (ii) trace the evolution from early explainer to reasoning-centric VLA models, and (iii) compare over 20 representative models according to VLA's progress in the autonomous driving domain. We also consolidate existing datasets and benchmarks, highlighting protocols that jointly measure driving safety, accuracy, and explanation quality. Finally, we detail open challenges - robustness, real-time efficiency, and formal verification - and outline future directions of VLA4AD. This survey provides a concise yet complete reference for advancing interpretable socially aligned autonomous vehicles. Github repo is available at \href{https://github.com/JohnsonJiang1996/Awesome-VLA4AD}{SicongJiang/Awesome-VLA4AD}.
>
---
#### [new 215] What Makes a Dribble Successful? Insights From 3D Pose Tracking Data
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于体育数据分析任务，旨在提升对足球 dribble 成功因素的理解。通过3D姿态数据提取新特征，改进了传统2D数据的局限性。**

- **链接: [http://arxiv.org/pdf/2506.22503v1](http://arxiv.org/pdf/2506.22503v1)**

> **作者:** Michiel Schepers; Pieter Robberechts; Jan Van Haaren; Jesse Davis
>
> **摘要:** Data analysis plays an increasingly important role in soccer, offering new ways to evaluate individual and team performance. One specific application is the evaluation of dribbles: one-on-one situations where an attacker attempts to bypass a defender with the ball. While previous research has primarily relied on 2D positional tracking data, this fails to capture aspects like balance, orientation, and ball control, limiting the depth of current insights. This study explores how pose tracking data (capturing players' posture and movement in three dimensions) can improve our understanding of dribbling skills. We extract novel pose-based features from 1,736 dribbles in the 2022/23 Champions League season and evaluate their impact on dribble success. Our results indicate that features capturing the attacker's balance and the alignment of the orientation between the attacker and defender are informative for predicting dribble success. Incorporating these pose-based features on top of features derived from traditional 2D positional data leads to a measurable improvement in model performance.
>
---
#### [new 216] Mettle: Meta-Token Learning for Memory-Efficient Audio-Visual Adaptation
- **分类: cs.CV**

- **简介: 该论文提出Mettle方法，用于音频-视觉任务的高效模型适配，解决内存和训练效率问题。通过元令牌学习实现知识蒸馏与特征适应。**

- **链接: [http://arxiv.org/pdf/2506.23271v1](http://arxiv.org/pdf/2506.23271v1)**

> **作者:** Jinxing Zhou; Zhihui Li; Yongqiang Yu; Yanghao Zhou; Ruohao Guo; Guangyao Li; Yuxin Mao; Mingfei Han; Xiaojun Chang; Meng Wang
>
> **备注:** Technical Report
>
> **摘要:** We present \textbf{Met}a-\textbf{T}oken \textbf{Le}arning (Mettle), a simple and memory-efficient method for adapting large-scale pretrained transformer models to downstream audio-visual tasks. Instead of sequentially modifying the output feature distribution of the transformer backbone, Mettle utilizes a lightweight \textit{Layer-Centric Distillation (LCD)} module to distill in parallel the intact audio or visual features embedded by each transformer layer into compact meta-tokens. This distillation process considers both pretrained knowledge preservation and task-specific adaptation. The obtained meta-tokens can be directly applied to classification tasks, such as audio-visual event localization and audio-visual video parsing. To further support fine-grained segmentation tasks, such as audio-visual segmentation, we introduce a \textit{Meta-Token Injection (MTI)} module, which utilizes the audio and visual meta-tokens distilled from the top transformer layer to guide feature adaptation in earlier layers. Extensive experiments on multiple audiovisual benchmarks demonstrate that our method significantly reduces memory usage and training time while maintaining parameter efficiency and competitive accuracy.
>
---
#### [new 217] Unifying Biomedical Vision-Language Expertise: Towards a Generalist Foundation Model via Multi-CLIP Knowledge Distillation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于 biomedical vision-language 任务，旨在解决医学领域数据稀缺与异质性问题。通过多模型知识蒸馏构建通用医学基础模型MMKD-CLIP，提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.22567v1](http://arxiv.org/pdf/2506.22567v1)**

> **作者:** Shansong Wang; Zhecheng Jin; Mingzhe Hu; Mojtaba Safari; Feng Zhao; Chih-Wei Chang; Richard LJ Qiu; Justin Roper; David S. Yu; Xiaofeng Yang
>
> **摘要:** CLIP models pretrained on natural images with billion-scale image-text pairs have demonstrated impressive capabilities in zero-shot classification, cross-modal retrieval, and open-ended visual answering. However, transferring this success to biomedicine is hindered by the scarcity of large-scale biomedical image-text corpora, the heterogeneity of image modalities, and fragmented data standards across institutions. These limitations hinder the development of a unified and generalizable biomedical foundation model trained from scratch. To overcome this, we introduce MMKD-CLIP, a generalist biomedical foundation model developed via Multiple Medical CLIP Knowledge Distillation. Rather than relying on billion-scale raw data, MMKD-CLIP distills knowledge from nine state-of-the-art domain-specific or generalist biomedical CLIP models, each pretrained on millions of biomedical image-text pairs. Our two-stage training pipeline first performs CLIP-style pretraining on over 2.9 million biomedical image-text pairs from 26 image modalities, followed by feature-level distillation using over 19.2 million feature pairs extracted from teacher models. We evaluate MMKD-CLIP on 58 diverse biomedical datasets, encompassing over 10.8 million biomedical images across nine image modalities. The evaluation spans six core task types: zero-shot classification, linear probing, cross-modal retrieval, visual question answering, survival prediction, and cancer diagnosis. MMKD-CLIP consistently outperforms all teacher models while demonstrating remarkable robustness and generalization across image domains and task settings. These results underscore that multi-teacher knowledge distillation is a scalable and effective paradigm for building high-performing biomedical foundation models under the practical constraints of real-world data availability.
>
---
#### [new 218] Recomposed realities: animating still images via patch clustering and randomness
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于图像动画任务，旨在通过局部结构重构使静态图像动态化。工作包括使用聚类和随机采样生成新图像。**

- **链接: [http://arxiv.org/pdf/2506.22556v1](http://arxiv.org/pdf/2506.22556v1)**

> **作者:** Markus Juvonen; Samuli Siltanen
>
> **备注:** 22 pages, 19 figures
>
> **摘要:** We present a patch-based image reconstruction and animation method that uses existing image data to bring still images to life through motion. Image patches from curated datasets are grouped using k-means clustering and a new target image is reconstructed by matching and randomly sampling from these clusters. This approach emphasizes reinterpretation over replication, allowing the source and target domains to differ conceptually while sharing local structures.
>
---
#### [new 219] Subjective Camera: Bridging Human Cognition and Visual Reconstruction through Sequence-Aware Sketch-Guided Diffusion
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决从主观描述和草图生成真实图像的问题。通过序列感知生成和潜在优化，提升语义与空间一致性。**

- **链接: [http://arxiv.org/pdf/2506.23711v1](http://arxiv.org/pdf/2506.23711v1)**

> **作者:** Haoyang Chen; Dongfang Sun; Caoyuan Ma; Shiqin Wang; Kewei Zhang; Zheng Wang; Zhixiang Wang
>
> **摘要:** We propose Subjective Camera, a human-as-imaging-device paradigm that reconstructs real-world scenes from mental impressions through synergistic use of verbal descriptions and progressive rough sketches. This approach overcomes dual limitations of language ambiguity and sketch abstraction by treating the user's drawing sequence as priors, effectively translating subjective perceptual expectations into photorealistic images. Existing approaches face three fundamental barriers: (1) user-specific subjective input biases, (2) huge modality gap between planar sketch and 3D priors in diffusion, and (3) sketch quality-sensitive performance degradation. Current solutions either demand resource-intensive model adaptation or impose impractical requirements on sketch precision. Our framework addresses these challenges through concept-sequential generation. (1) We establish robust appearance priors through text-reward optimization, and then implement sequence-aware disentangled generation that processes concepts in sketching order; these steps accommodate user-specific subjective expectation in a train-free way. (2) We employ latent optimization that effectively bridges the modality gap between planar sketches and 3D priors in diffusion. (3) Our hierarchical reward-guided framework enables the use of rough sketches without demanding artistic expertise. Comprehensive evaluation across diverse datasets demonstrates that our approach achieves state-of-the-art performance in maintaining both semantic and spatial coherence.
>
---
#### [new 220] Computer-Aided Multi-Stroke Character Simplification by Stroke Removal
- **分类: cs.CV**

- **简介: 该论文属于字符简化任务，旨在通过删除笔画提升多笔画汉字的可读性，降低学习难度。工作包括构建简化框架并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2506.23106v1](http://arxiv.org/pdf/2506.23106v1)**

> **作者:** Ryo Ishiyama; Shinnosuke Matsuo; Seiichi Uchida
>
> **备注:** ICDAR2025 (Oral)
>
> **摘要:** Multi-stroke characters in scripts such as Chinese and Japanese can be highly complex, posing significant challenges for both native speakers and, especially, non-native learners. If these characters can be simplified without degrading their legibility, it could reduce learning barriers for non-native speakers, facilitate simpler and legible font designs, and contribute to efficient character-based communication systems. In this paper, we propose a framework to systematically simplify multi-stroke characters by selectively removing strokes while preserving their overall legibility. More specifically, we use a highly accurate character recognition model to assess legibility and remove those strokes that minimally impact it. Experimental results on 1,256 character classes with 5, 10, 15, and 20 strokes reveal several key findings, including the observation that even after removing multiple strokes, many characters remain distinguishable. These findings suggest the potential for more formalized simplification strategies.
>
---
#### [new 221] Single-Frame Point-Pixel Registration via Supervised Cross-Modal Feature Matching
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于点云与图像配准任务，解决单帧LiDAR稀疏性带来的模态差异问题。通过引入无检测器匹配框架和重复性评分机制，实现高效点像素对应。**

- **链接: [http://arxiv.org/pdf/2506.22784v1](http://arxiv.org/pdf/2506.22784v1)**

> **作者:** Yu Han; Zhiwei Huang; Yanting Zhang; Fangjun Ding; Shen Cai; Rui Fan
>
> **摘要:** Point-pixel registration between LiDAR point clouds and camera images is a fundamental yet challenging task in autonomous driving and robotic perception. A key difficulty lies in the modality gap between unstructured point clouds and structured images, especially under sparse single-frame LiDAR settings. Existing methods typically extract features separately from point clouds and images, then rely on hand-crafted or learned matching strategies. This separate encoding fails to bridge the modality gap effectively, and more critically, these methods struggle with the sparsity and noise of single-frame LiDAR, often requiring point cloud accumulation or additional priors to improve reliability. Inspired by recent progress in detector-free matching paradigms (e.g. MatchAnything), we revisit the projection-based approach and introduce the detector-free framework for direct point-pixel matching between LiDAR and camera views. Specifically, we project the LiDAR intensity map into a 2D view from the LiDAR perspective and feed it into an attention-based detector-free matching network, enabling cross-modal correspondence estimation without relying on multi-frame accumulation. To further enhance matching reliability, we introduce a repeatability scoring mechanism that acts as a soft visibility prior. This guides the network to suppress unreliable matches in regions with low intensity variation, improving robustness under sparse input. Extensive experiments on KITTI, nuScenes, and MIAS-LCEC-TF70 benchmarks demonstrate that our method achieves state-of-the-art performance, outperforming prior approaches on nuScenes (even those relying on accumulated point clouds), despite using only single-frame LiDAR.
>
---
#### [new 222] Scalable Dynamic Origin-Destination Demand Estimation Enhanced by High-Resolution Satellite Imagery Data
- **分类: cs.CV; cs.AI; stat.AP**

- **简介: 该论文属于交通需求估计任务，解决多类交通网络中动态OD估计问题，通过融合卫星图像与传统数据提升估计精度和可扩展性。**

- **链接: [http://arxiv.org/pdf/2506.22499v1](http://arxiv.org/pdf/2506.22499v1)**

> **作者:** Jiachao Liu; Pablo Guarda; Koichiro Niinuma; Sean Qian
>
> **摘要:** This study presents a novel integrated framework for dynamic origin-destination demand estimation (DODE) in multi-class mesoscopic network models, leveraging high-resolution satellite imagery together with conventional traffic data from local sensors. Unlike sparse local detectors, satellite imagery offers consistent, city-wide road and traffic information of both parking and moving vehicles, overcoming data availability limitations. To extract information from imagery data, we design a computer vision pipeline for class-specific vehicle detection and map matching, generating link-level traffic density observations by vehicle class. Building upon this information, we formulate a computational graph-based DODE model that calibrates dynamic network states by jointly matching observed traffic counts and travel times from local sensors with density measurements derived from satellite imagery. To assess the accuracy and scalability of the proposed framework, we conduct a series of numerical experiments using both synthetic and real-world data. The results of out-of-sample tests demonstrate that supplementing traditional data with satellite-derived density significantly improves estimation performance, especially for links without local sensors. Real-world experiments also confirm the framework's capability to handle large-scale networks, supporting its potential for practical deployment in cities of varying sizes. Sensitivity analysis further evaluates the impact of data quality related to satellite imagery data.
>
---
#### [new 223] A Clinically-Grounded Two-Stage Framework for Renal CT Report Generation
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学影像报告生成任务，旨在解决肾部CT报告自动化问题。通过两阶段框架提取异常特征并生成临床准确的报告。**

- **链接: [http://arxiv.org/pdf/2506.23584v1](http://arxiv.org/pdf/2506.23584v1)**

> **作者:** Renjie Liang; Zhengkang Fan; Jinqian Pan; Chenkun Sun; Russell Terry; Jie Xu
>
> **摘要:** Generating radiology reports from CT scans remains a complex task due to the nuanced nature of medical imaging and the variability in clinical documentation. In this study, we propose a two-stage framework for generating renal radiology reports from 2D CT slices. First, we extract structured abnormality features using a multi-task learning model trained to identify lesion attributes such as location, size, enhancement, and attenuation. These extracted features are subsequently combined with the corresponding CT image and fed into a fine-tuned vision-language model to generate natural language report sentences aligned with clinical findings. We conduct experiments on a curated dataset of renal CT studies with manually annotated sentence-slice-feature triplets and evaluate performance using both classification metrics and natural language generation metrics. Our results demonstrate that the proposed model outperforms random baselines across all abnormality types, and the generated reports capture key clinical content with reasonable textual accuracy. This exploratory work highlights the feasibility of modular, feature-informed report generation for renal imaging. Future efforts will focus on extending this pipeline to 3D CT volumes and further improving clinical fidelity in multimodal medical AI systems.
>
---
#### [new 224] Supervised Diffusion-Model-Based PET Image Reconstruction
- **分类: physics.med-ph; cs.CV**

- **简介: 该论文属于PET图像重建任务，旨在解决扩散模型与噪声数据交互不足的问题，提出一种监督式扩散模型方法以提高重建精度和不确定性估计。**

- **链接: [http://arxiv.org/pdf/2506.24034v1](http://arxiv.org/pdf/2506.24034v1)**

> **作者:** George Webber; Alexander Hammers; Andrew P King; Andrew J Reader
>
> **备注:** 12 pages, 6 figures. Submitted to MICCAI 2025, not peer-reviewed
>
> **摘要:** Diffusion models (DMs) have recently been introduced as a regularizing prior for PET image reconstruction, integrating DMs trained on high-quality PET images with unsupervised schemes that condition on measured data. While these approaches have potential generalization advantages due to their independence from the scanner geometry and the injected activity level, they forgo the opportunity to explicitly model the interaction between the DM prior and noisy measurement data, potentially limiting reconstruction accuracy. To address this, we propose a supervised DM-based algorithm for PET reconstruction. Our method enforces the non-negativity of PET's Poisson likelihood model and accommodates the wide intensity range of PET images. Through experiments on realistic brain PET phantoms, we demonstrate that our approach outperforms or matches state-of-the-art deep learning-based methods quantitatively across a range of dose levels. We further conduct ablation studies to demonstrate the benefits of the proposed components in our model, as well as its dependence on training data, parameter count, and number of diffusion steps. Additionally, we show that our approach enables more accurate posterior sampling than unsupervised DM-based methods, suggesting improved uncertainty estimation. Finally, we extend our methodology to a practical approach for fully 3D PET and present example results from real [$^{18}$F]FDG brain PET data.
>
---
#### [new 225] maneuverRecognition -- A Python package for Timeseries Classification in the domain of Vehicle Telematics
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于车辆遥测中的时间序列分类任务，旨在解决驾驶行为识别问题。工作包括开发Python包maneuverRecognition，提供数据预处理、建模与评估功能。**

- **链接: [http://arxiv.org/pdf/2506.23147v1](http://arxiv.org/pdf/2506.23147v1)**

> **作者:** Jonathan Schuster; Fabian Transchel
>
> **备注:** 6 pages, 2 figures
>
> **摘要:** In the domain of vehicle telematics the automated recognition of driving maneuvers is used to classify and evaluate driving behaviour. This not only serves as a component to enhance the personalization of insurance policies, but also to increase road safety, reduce accidents and the associated costs as well as to reduce fuel consumption and support environmentally friendly driving. In this context maneuver recognition technically requires a continuous application of time series classification which poses special challenges to the transfer, preprocessing and storage of telematic sensor data, the training of predictive models, and the prediction itself. Although much research has been done in the field of gathering relevant data or regarding the methods to build predictive models for the task of maneuver recognition, there is a practical need for python packages and functions that allow to quickly transform data into the required structure as well as to build and evaluate such models. The maneuverRecognition package was therefore developed to provide the necessary functions for preprocessing, modelling and evaluation and also includes a ready to use LSTM based network structure that can be modified. The implementation of the package is demonstrated using real driving data of three different persons recorded via smartphone sensors.
>
---
#### [new 226] GaVS: 3D-Grounded Video Stabilization via Temporally-Consistent Local Reconstruction and Rendering
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于视频稳定任务，旨在解决现有方法中的几何失真和泛化不足问题。提出GaVS，通过3D重建与渲染实现时间一致的局部重建，提升稳定性与质量。**

- **链接: [http://arxiv.org/pdf/2506.23957v1](http://arxiv.org/pdf/2506.23957v1)**

> **作者:** Zinuo You; Stamatios Georgoulis; Anpei Chen; Siyu Tang; Dengxin Dai
>
> **备注:** siggraph 2025, project website: https://sinoyou.github.io/gavs
>
> **摘要:** Video stabilization is pivotal for video processing, as it removes unwanted shakiness while preserving the original user motion intent. Existing approaches, depending on the domain they operate, suffer from several issues (e.g. geometric distortions, excessive cropping, poor generalization) that degrade the user experience. To address these issues, we introduce \textbf{GaVS}, a novel 3D-grounded approach that reformulates video stabilization as a temporally-consistent `local reconstruction and rendering' paradigm. Given 3D camera poses, we augment a reconstruction model to predict Gaussian Splatting primitives, and finetune it at test-time, with multi-view dynamics-aware photometric supervision and cross-frame regularization, to produce temporally-consistent local reconstructions. The model are then used to render each stabilized frame. We utilize a scene extrapolation module to avoid frame cropping. Our method is evaluated on a repurposed dataset, instilled with 3D-grounded information, covering samples with diverse camera motions and scene dynamics. Quantitatively, our method is competitive with or superior to state-of-the-art 2D and 2.5D approaches in terms of conventional task metrics and new geometry consistency. Qualitatively, our method produces noticeably better results compared to alternatives, validated by the user study.
>
---
#### [new 227] MARBLE: A Hard Benchmark for Multimodal Spatial Reasoning and Planning
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出MARBLE，一个用于多模态空间推理与规划的基准测试，旨在评估模型在复杂多模态问题上的逐步推理能力。**

- **链接: [http://arxiv.org/pdf/2506.22992v1](http://arxiv.org/pdf/2506.22992v1)**

> **作者:** Yulun Jiang; Yekun Chai; Maria Brbić; Michael Moor
>
> **摘要:** The ability to process information from multiple modalities and to reason through it step-by-step remains a critical challenge in advancing artificial intelligence. However, existing reasoning benchmarks focus on text-only reasoning, or employ multimodal questions that can be answered by directly retrieving information from a non-text modality. Thus, complex reasoning remains poorly understood in multimodal domains. Here, we present MARBLE, a challenging multimodal reasoning benchmark that is designed to scrutinize multimodal language models (MLLMs) in their ability to carefully reason step-by-step through complex multimodal problems and environments. MARBLE is composed of two highly challenging tasks, M-Portal and M-Cube, that require the crafting and understanding of multistep plans under spatial, visual, and physical constraints. We find that current MLLMs perform poorly on MARBLE -- all the 12 advanced models obtain near-random performance on M-Portal and 0% accuracy on M-Cube. Only in simplified subtasks some models outperform the random baseline, indicating that complex reasoning is still a challenge for existing MLLMs. Moreover, we show that perception remains a bottleneck, where MLLMs occasionally fail to extract information from the visual inputs. By shedding a light on the limitations of MLLMs, we hope that MARBLE will spur the development of the next generation of models with the ability to reason and plan across many, multimodal reasoning steps.
>
---
#### [new 228] Diffusion Model-based Data Augmentation Method for Fetal Head Ultrasound Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决胎儿头部超声数据不足的问题。通过扩散模型生成合成数据，提升分割性能。**

- **链接: [http://arxiv.org/pdf/2506.23664v1](http://arxiv.org/pdf/2506.23664v1)**

> **作者:** Fangyijie Wang; Kevin Whelan; Félix Balado; Guénolé Silvestre; Kathleen M. Curran
>
> **摘要:** Medical image data is less accessible than in other domains due to privacy and regulatory constraints. In addition, labeling requires costly, time-intensive manual image annotation by clinical experts. To overcome these challenges, synthetic medical data generation offers a promising solution. Generative AI (GenAI), employing generative deep learning models, has proven effective at producing realistic synthetic images. This study proposes a novel mask-guided GenAI approach using diffusion models to generate synthetic fetal head ultrasound images paired with segmentation masks. These synthetic pairs augment real datasets for supervised fine-tuning of the Segment Anything Model (SAM). Our results show that the synthetic data captures real image features effectively, and this approach reaches state-of-the-art fetal head segmentation, especially when trained with a limited number of real image-mask pairs. In particular, the segmentation reaches Dice Scores of 94.66\% and 94.38\% using a handful of ultrasound images from the Spanish and African cohorts, respectively. Our code, models, and data are available on GitHub.
>
---
#### [new 229] Multi-Source COVID-19 Detection via Variance Risk Extrapolation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于多源新冠检测任务，旨在解决跨机构数据域偏移问题。通过引入VREx和Mixup提升模型泛化能力，实现高准确率分类。**

- **链接: [http://arxiv.org/pdf/2506.23208v1](http://arxiv.org/pdf/2506.23208v1)**

> **作者:** Runtian Yuan; Qingqiu Li; Junlin Hou; Jilan Xu; Yuejie Zhang; Rui Feng; Hao Chen
>
> **摘要:** We present our solution for the Multi-Source COVID-19 Detection Challenge, which aims to classify chest CT scans into COVID and Non-COVID categories across data collected from four distinct hospitals and medical centers. A major challenge in this task lies in the domain shift caused by variations in imaging protocols, scanners, and patient populations across institutions. To enhance the cross-domain generalization of our model, we incorporate Variance Risk Extrapolation (VREx) into the training process. VREx encourages the model to maintain consistent performance across multiple source domains by explicitly minimizing the variance of empirical risks across environments. This regularization strategy reduces overfitting to center-specific features and promotes learning of domain-invariant representations. We further apply Mixup data augmentation to improve generalization and robustness. Mixup interpolates both the inputs and labels of randomly selected pairs of training samples, encouraging the model to behave linearly between examples and enhancing its resilience to noise and limited data. Our method achieves an average macro F1 score of 0.96 across the four sources on the validation set, demonstrating strong generalization.
>
---
#### [new 230] FedCLAM: Client Adaptive Momentum with Foreground Intensity Matching for Federated Medical Image Segmentation
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于联邦医学图像分割任务，旨在解决跨机构数据差异导致的模型效果下降问题。提出FedCLAM方法，结合客户端自适应动量和强度对齐损失，提升分割性能。**

- **链接: [http://arxiv.org/pdf/2506.22580v1](http://arxiv.org/pdf/2506.22580v1)**

> **作者:** Vasilis Siomos; Jonathan Passerat-Palmbach; Giacomo Tarroni
>
> **备注:** 10 pages, 2 figures, Accepted at MICCAI 2025
>
> **摘要:** Federated learning is a decentralized training approach that keeps data under stakeholder control while achieving superior performance over isolated training. While inter-institutional feature discrepancies pose a challenge in all federated settings, medical imaging is particularly affected due to diverse imaging devices and population variances, which can diminish the global model's effectiveness. Existing aggregation methods generally fail to adapt across varied circumstances. To address this, we propose FedCLAM, which integrates \textit{client-adaptive momentum} terms derived from each client's loss reduction during local training, as well as a \textit{personalized dampening factor} to curb overfitting. We further introduce a novel \textit{intensity alignment} loss that matches predicted and ground-truth foreground distributions to handle heterogeneous image intensity profiles across institutions and devices. Extensive evaluations on two datasets show that FedCLAM surpasses eight cutting-edge methods in medical segmentation tasks, underscoring its efficacy. The code is available at https://github.com/siomvas/FedCLAM.
>
---
#### [new 231] FedWSQ: Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于联邦学习任务，解决数据异构性和通信约束问题。提出FedWSQ框架，结合权重标准化和分布感知非均匀量化，提升模型性能并减少通信开销。**

- **链接: [http://arxiv.org/pdf/2506.23516v1](http://arxiv.org/pdf/2506.23516v1)**

> **作者:** Seung-Wook Kim; Seongyeol Kim; Jiah Kim; Seowon Ji; Se-Ho Lee
>
> **摘要:** Federated learning (FL) often suffers from performance degradation due to key challenges such as data heterogeneity and communication constraints. To address these limitations, we present a novel FL framework called FedWSQ, which integrates weight standardization (WS) and the proposed distribution-aware non-uniform quantization (DANUQ). WS enhances FL performance by filtering out biased components in local updates during training, thereby improving the robustness of the model against data heterogeneity and unstable client participation. In addition, DANUQ minimizes quantization errors by leveraging the statistical properties of local model updates. As a result, FedWSQ significantly reduces communication overhead while maintaining superior model accuracy. Extensive experiments on FL benchmark datasets demonstrate that FedWSQ consistently outperforms existing FL methods across various challenging FL settings, including extreme data heterogeneity and ultra-low-bit communication scenarios.
>
---
#### [new 232] Spatio-Temporal Representation Decoupling and Enhancement for Federated Instrument Segmentation in Surgical Videos
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于手术器械分割任务，针对联邦学习中手术数据的特殊性，提出FedST方法，通过时空表征解耦与增强提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.23759v1](http://arxiv.org/pdf/2506.23759v1)**

> **作者:** Zheng Fang; Xiaoming Qi; Chun-Mei Feng; Jialun Pei; Weixin Si; Yueming Jin
>
> **摘要:** Surgical instrument segmentation under Federated Learning (FL) is a promising direction, which enables multiple surgical sites to collaboratively train the model without centralizing datasets. However, there exist very limited FL works in surgical data science, and FL methods for other modalities do not consider inherent characteristics in surgical domain: i) different scenarios show diverse anatomical backgrounds while highly similar instrument representation; ii) there exist surgical simulators which promote large-scale synthetic data generation with minimal efforts. In this paper, we propose a novel Personalized FL scheme, Spatio-Temporal Representation Decoupling and Enhancement (FedST), which wisely leverages surgical domain knowledge during both local-site and global-server training to boost segmentation. Concretely, our model embraces a Representation Separation and Cooperation (RSC) mechanism in local-site training, which decouples the query embedding layer to be trained privately, to encode respective backgrounds. Meanwhile, other parameters are optimized globally to capture the consistent representations of instruments, including the temporal layer to capture similar motion patterns. A textual-guided channel selection is further designed to highlight site-specific features, facilitating model adapta tion to each site. Moreover, in global-server training, we propose Synthesis-based Explicit Representation Quantification (SERQ), which defines an explicit representation target based on synthetic data to synchronize the model convergence during fusion for improving model generalization.
>
---
#### [new 233] C3VDv2 -- Colonoscopy 3D video dataset with enhanced realism
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决缺乏高质量3D结肠镜数据的问题。作者构建了C3VDv2数据集，包含高真实感视频及多种标注信息，用于评估和提升3D重建算法。**

- **链接: [http://arxiv.org/pdf/2506.24074v1](http://arxiv.org/pdf/2506.24074v1)**

> **作者:** Mayank V. Golhar; Lucas Sebastian Galeano Fretes; Loren Ayers; Venkata S. Akshintala; Taylor L. Bobrow; Nicholas J. Durr
>
> **备注:** 19 pages, 7 figures
>
> **摘要:** Computer vision techniques have the potential to improve the diagnostic performance of colonoscopy, but the lack of 3D colonoscopy datasets for training and validation hinders their development. This paper introduces C3VDv2, the second version (v2) of the high-definition Colonoscopy 3D Video Dataset, featuring enhanced realism designed to facilitate the quantitative evaluation of 3D colon reconstruction algorithms. 192 video sequences were captured by imaging 60 unique, high-fidelity silicone colon phantom segments. Ground truth depth, surface normals, optical flow, occlusion, six-degree-of-freedom pose, coverage maps, and 3D models are provided for 169 colonoscopy videos. Eight simulated screening colonoscopy videos acquired by a gastroenterologist are provided with ground truth poses. The dataset includes 15 videos featuring colon deformations for qualitative assessment. C3VDv2 emulates diverse and challenging scenarios for 3D reconstruction algorithms, including fecal debris, mucous pools, blood, debris obscuring the colonoscope lens, en-face views, and fast camera motion. The enhanced realism of C3VDv2 will allow for more robust and representative development and evaluation of 3D reconstruction algorithms.
>
---
#### [new 234] Radioactive Watermarks in Diffusion and Autoregressive Image Generative Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于图像生成模型的水印任务，旨在解决水印在训练过程中失效的问题。研究分析了扩散模型和自回归模型的水印放射性，提出一种适用于自回归模型的放射性水印方法。**

- **链接: [http://arxiv.org/pdf/2506.23731v1](http://arxiv.org/pdf/2506.23731v1)**

> **作者:** Michel Meintz; Jan Dubiński; Franziska Boenisch; Adam Dziedzic
>
> **摘要:** Image generative models have become increasingly popular, but training them requires large datasets that are costly to collect and curate. To circumvent these costs, some parties may exploit existing models by using the generated images as training data for their own models. In general, watermarking is a valuable tool for detecting unauthorized use of generated images. However, when these images are used to train a new model, watermarking can only enable detection if the watermark persists through training and remains identifiable in the outputs of the newly trained model - a property known as radioactivity. We analyze the radioactivity of watermarks in images generated by diffusion models (DMs) and image autoregressive models (IARs). We find that existing watermarking methods for DMs fail to retain radioactivity, as watermarks are either erased during encoding into the latent space or lost in the noising-denoising process (during the training in the latent space). Meanwhile, despite IARs having recently surpassed DMs in image generation quality and efficiency, no radioactive watermarking methods have been proposed for them. To overcome this limitation, we propose the first watermarking method tailored for IARs and with radioactivity in mind - drawing inspiration from techniques in large language models (LLMs), which share IARs' autoregressive paradigm. Our extensive experimental evaluation highlights our method's effectiveness in preserving radioactivity within IARs, enabling robust provenance tracking, and preventing unauthorized use of their generated images.
>
---
#### [new 235] CA-Diff: Collaborative Anatomy Diffusion for Brain Tissue Segmentation
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分割任务，旨在解决脑组织分割精度不足的问题。通过引入解剖信息增强扩散模型，提升分割效果。**

- **链接: [http://arxiv.org/pdf/2506.22882v1](http://arxiv.org/pdf/2506.22882v1)**

> **作者:** Qilong Xing; Zikai Song; Yuteng Ye; Yuke Chen; Youjia Zhang; Na Feng; Junqing Yu; Wei Yang
>
> **备注:** ICME 2025
>
> **摘要:** Segmentation of brain structures from MRI is crucial for evaluating brain morphology, yet existing CNN and transformer-based methods struggle to delineate complex structures accurately. While current diffusion models have shown promise in image segmentation, they are inadequate when applied directly to brain MRI due to neglecting anatomical information. To address this, we propose Collaborative Anatomy Diffusion (CA-Diff), a framework integrating spatial anatomical features to enhance segmentation accuracy of the diffusion model. Specifically, we introduce distance field as an auxiliary anatomical condition to provide global spatial context, alongside a collaborative diffusion process to model its joint distribution with anatomical structures, enabling effective utilization of anatomical features for segmentation. Furthermore, we introduce a consistency loss to refine relationships between the distance field and anatomical structures and design a time adapted channel attention module to enhance the U-Net feature fusion procedure. Extensive experiments show that CA-Diff outperforms state-of-the-art (SOTA) methods.
>
---
#### [new 236] MedSAM-CA: A CNN-Augmented ViT with Attention-Enhanced Multi-Scale Fusion for Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决标注数据少和边界模糊的问题。提出MedSAM-CA模型，通过注意力增强的多尺度融合提升分割精度。**

- **链接: [http://arxiv.org/pdf/2506.23700v1](http://arxiv.org/pdf/2506.23700v1)**

> **作者:** Peiting Tian; Xi Chen; Haixia Bi; Fan Li
>
> **摘要:** Medical image segmentation plays a crucial role in clinical diagnosis and treatment planning, where accurate boundary delineation is essential for precise lesion localization, organ identification, and quantitative assessment. In recent years, deep learning-based methods have significantly advanced segmentation accuracy. However, two major challenges remain. First, the performance of these methods heavily relies on large-scale annotated datasets, which are often difficult to obtain in medical scenarios due to privacy concerns and high annotation costs. Second, clinically challenging scenarios, such as low contrast in certain imaging modalities and blurry lesion boundaries caused by malignancy, still pose obstacles to precise segmentation. To address these challenges, we propose MedSAM-CA, an architecture-level fine-tuning approach that mitigates reliance on extensive manual annotations by adapting the pretrained foundation model, Medical Segment Anything (MedSAM). MedSAM-CA introduces two key components: the Convolutional Attention-Enhanced Boundary Refinement Network (CBR-Net) and the Attention-Enhanced Feature Fusion Block (Atte-FFB). CBR-Net operates in parallel with the MedSAM encoder to recover boundary information potentially overlooked by long-range attention mechanisms, leveraging hierarchical convolutional processing. Atte-FFB, embedded in the MedSAM decoder, fuses multi-level fine-grained features from skip connections in CBR-Net with global representations upsampled within the decoder to enhance boundary delineation accuracy. Experiments on publicly available datasets covering dermoscopy, CT, and MRI imaging modalities validate the effectiveness of MedSAM-CA. On dermoscopy dataset, MedSAM-CA achieves 94.43% Dice with only 2% of full training data, reaching 97.25% of full-data training performance, demonstrating strong effectiveness in low-resource clinical settings.
>
---
#### [new 237] KiseKloset: Comprehensive System For Outfit Retrieval, Recommendation, And Try-On
- **分类: cs.IR; cs.CV**

- **简介: 该论文属于服装推荐与虚拟试穿任务，旨在提升在线购物体验。通过构建KiseKloset系统，解决 outfit 检索、推荐及试穿问题，提升用户满意度。**

- **链接: [http://arxiv.org/pdf/2506.23471v1](http://arxiv.org/pdf/2506.23471v1)**

> **作者:** Thanh-Tung Phan-Nguyen; Khoi-Nguyen Nguyen-Ngoc; Tam V. Nguyen; Minh-Triet Tran; Trung-Nghia Le
>
> **摘要:** The global fashion e-commerce industry has become integral to people's daily lives, leveraging technological advancements to offer personalized shopping experiences, primarily through recommendation systems that enhance customer engagement through personalized suggestions. To improve customers' experience in online shopping, we propose a novel comprehensive KiseKloset system for outfit retrieval, recommendation, and try-on. We explore two approaches for outfit retrieval: similar item retrieval and text feedback-guided item retrieval. Notably, we introduce a novel transformer architecture designed to recommend complementary items from diverse categories. Furthermore, we enhance the overall performance of the search pipeline by integrating approximate algorithms to optimize the search process. Additionally, addressing the crucial needs of online shoppers, we employ a lightweight yet efficient virtual try-on framework capable of real-time operation, memory efficiency, and maintaining realistic outputs compared to its predecessors. This virtual try-on module empowers users to visualize specific garments on themselves, enhancing the customers' experience and reducing costs associated with damaged items for retailers. We deployed our end-to-end system for online users to test and provide feedback, enabling us to measure their satisfaction levels. The results of our user study revealed that 84% of participants found our comprehensive system highly useful, significantly improving their online shopping experience.
>
---
#### [new 238] ICP-3DGS: SfM-free 3D Gaussian Splatting for Large-scale Unbounded Scenes
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于3D重建任务，解决无SfM场景下的相机位姿估计与视图合成问题，提出ICP-3DGS方法结合ICP和体素优化提升大场景重建效果。**

- **链接: [http://arxiv.org/pdf/2506.21629v1](http://arxiv.org/pdf/2506.21629v1)**

> **作者:** Chenhao Zhang; Yezhi Shen; Fengqing Zhu
>
> **备注:** 6 pages, Source code is available at https://github.com/Chenhao-Z/ICP-3DGS. To appear at ICIP 2025
>
> **摘要:** In recent years, neural rendering methods such as NeRFs and 3D Gaussian Splatting (3DGS) have made significant progress in scene reconstruction and novel view synthesis. However, they heavily rely on preprocessed camera poses and 3D structural priors from structure-from-motion (SfM), which are challenging to obtain in outdoor scenarios. To address this challenge, we propose to incorporate Iterative Closest Point (ICP) with optimization-based refinement to achieve accurate camera pose estimation under large camera movements. Additionally, we introduce a voxel-based scene densification approach to guide the reconstruction in large-scale scenes. Experiments demonstrate that our approach ICP-3DGS outperforms existing methods in both camera pose estimation and novel view synthesis across indoor and outdoor scenes of various scales. Source code is available at https://github.com/Chenhao-Z/ICP-3DGS.
>
---
#### [new 239] Confident Splatting: Confidence-Based Compression of 3D Gaussian Splatting via Learnable Beta Distributions
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于3D渲染任务，解决高存储和计算开销问题。通过可学习的置信度分布压缩高斯点云，保留视觉质量。**

- **链接: [http://arxiv.org/pdf/2506.22973v1](http://arxiv.org/pdf/2506.22973v1)**

> **作者:** AmirHossein Naghi Razlighi; Elaheh Badali Golezani; Shohreh Kasaei
>
> **摘要:** 3D Gaussian Splatting enables high-quality real-time rendering but often produces millions of splats, resulting in excessive storage and computational overhead. We propose a novel lossy compression method based on learnable confidence scores modeled as Beta distributions. Each splat's confidence is optimized through reconstruction-aware losses, enabling pruning of low-confidence splats while preserving visual fidelity. The proposed approach is architecture-agnostic and can be applied to any Gaussian Splatting variant. In addition, the average confidence values serve as a new metric to assess the quality of the scene. Extensive experiments demonstrate favorable trade-offs between compression and fidelity compared to prior work. Our code and data are publicly available at https://github.com/amirhossein-razlighi/Confident-Splatting
>
---
#### [new 240] UltraTwin: Towards Cardiac Anatomical Twin Generation from Multi-view 2D Ultrasound
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于心脏影像重建任务，旨在从多视角2D超声生成高精度心脏解剖孪生模型，解决数据稀缺与结构复杂问题。**

- **链接: [http://arxiv.org/pdf/2506.23490v1](http://arxiv.org/pdf/2506.23490v1)**

> **作者:** Junxuan Yu; Yaofei Duan; Yuhao Huang; Yu Wang; Rongbo Ling; Weihao Luo; Ang Zhang; Jingxian Xu; Qiongying Ni; Yongsong Zhou; Binghan Li; Haoran Dou; Liping Liu; Yanfen Chu; Feng Geng; Zhe Sheng; Zhifeng Ding; Dingxin Zhang; Rui Huang; Yuhang Zhang; Xiaowei Xu; Tao Tan; Dong Ni; Zhongshan Gou; Xin Yang
>
> **备注:** accepted by miccai 2025
>
> **摘要:** Echocardiography is routine for cardiac examination. However, 2D ultrasound (US) struggles with accurate metric calculation and direct observation of 3D cardiac structures. Moreover, 3D US is limited by low resolution, small field of view and scarce availability in practice. Constructing the cardiac anatomical twin from 2D images is promising to provide precise treatment planning and clinical quantification. However, it remains challenging due to the rare paired data, complex structures, and US noises. In this study, we introduce a novel generative framework UltraTwin, to obtain cardiac anatomical twin from sparse multi-view 2D US. Our contribution is three-fold. First, pioneered the construction of a real-world and high-quality dataset containing strictly paired multi-view 2D US and CT, and pseudo-paired data. Second, we propose a coarse-to-fine scheme to achieve hierarchical reconstruction optimization. Last, we introduce an implicit autoencoder for topology-aware constraints. Extensive experiments show that UltraTwin reconstructs high-quality anatomical twins versus strong competitors. We believe it advances anatomical twin modeling for potential applications in personalized cardiac care.
>
---
#### [new 241] Riemannian-Geometric Fingerprints of Generative Models
- **分类: cs.LG; cs.CR; cs.CV; I.2.6**

- **简介: 该论文属于模型溯源任务，旨在解决生成模型指纹识别问题。通过引入黎曼几何方法，提出新的指纹定义，提升模型辨识与泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.22802v1](http://arxiv.org/pdf/2506.22802v1)**

> **作者:** Hae Jin Song; Laurent Itti
>
> **摘要:** Recent breakthroughs and rapid integration of generative models (GMs) have sparked interest in the problem of model attribution and their fingerprints. For instance, service providers need reliable methods of authenticating their models to protect their IP, while users and law enforcement seek to verify the source of generated content for accountability and trust. In addition, a growing threat of model collapse is arising, as more model-generated data are being fed back into sources (e.g., YouTube) that are often harvested for training ("regurgitative training"), heightening the need to differentiate synthetic from human data. Yet, a gap still exists in understanding generative models' fingerprints, we believe, stemming from the lack of a formal framework that can define, represent, and analyze the fingerprints in a principled way. To address this gap, we take a geometric approach and propose a new definition of artifact and fingerprint of GMs using Riemannian geometry, which allows us to leverage the rich theory of differential geometry. Our new definition generalizes previous work (Song et al., 2024) to non-Euclidean manifolds by learning Riemannian metrics from data and replacing the Euclidean distances and nearest-neighbor search with geodesic distances and kNN-based Riemannian center of mass. We apply our theory to a new gradient-based algorithm for computing the fingerprints in practice. Results show that it is more effective in distinguishing a large array of GMs, spanning across 4 different datasets in 2 different resolutions (64 by 64, 256 by 256), 27 model architectures, and 2 modalities (Vision, Vision-Language). Using our proposed definition significantly improves the performance on model attribution, as well as a generalization to unseen datasets, model types, and modalities, suggesting its practical efficacy.
>
---
#### [new 242] SegmentAnyMuscle: A universal muscle segmentation model across different locations in MRI
- **分类: eess.SP; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决MRI中肌肉精准分割的问题。通过开发一个通用模型，实现跨部位和序列的自动化肌肉分割。**

- **链接: [http://arxiv.org/pdf/2506.22467v1](http://arxiv.org/pdf/2506.22467v1)**

> **作者:** Roy Colglazier; Jisoo Lee; Haoyu Dong; Hanxue Gu; Yaqian Chen; Joseph Cao; Zafer Yildiz; Zhonghao Liu; Nicholas Konz; Jichen Yang; Jikai Zhang; Yuwen Chen; Lin Li; Adrian Camarena; Maciej A. Mazurowski
>
> **备注:** 24 pages, 6 figures
>
> **摘要:** The quantity and quality of muscles are increasingly recognized as important predictors of health outcomes. While MRI offers a valuable modality for such assessments, obtaining precise quantitative measurements of musculature remains challenging. This study aimed to develop a publicly available model for muscle segmentation in MRIs and demonstrate its applicability across various anatomical locations and imaging sequences. A total of 362 MRIs from 160 patients at a single tertiary center (Duke University Health System, 2016-2020) were included, with 316 MRIs from 114 patients used for model development. The model was tested on two separate sets: one with 28 MRIs representing common sequence types, achieving an average Dice Similarity Coefficient (DSC) of 88.45%, and another with 18 MRIs featuring less frequent sequences and abnormalities such as muscular atrophy, hardware, and significant noise, achieving 86.21% DSC. These results demonstrate the feasibility of a fully automated deep learning algorithm for segmenting muscles on MRI across diverse settings. The public release of this model enables consistent, reproducible research into the relationship between musculature and health.
>
---
#### [new 243] ShapeKit
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在提升整体解剖形状的准确性。通过引入ShapeKit工具包，在不重新训练模型的情况下显著提高分割效果。**

- **链接: [http://arxiv.org/pdf/2506.24003v1](http://arxiv.org/pdf/2506.24003v1)**

> **作者:** Junqi Liu; Dongli He; Wenxuan Li; Ningyu Wang; Alan L. Yuille; Zongwei Zhou
>
> **摘要:** In this paper, we present a practical approach to improve anatomical shape accuracy in whole-body medical segmentation. Our analysis shows that a shape-focused toolkit can enhance segmentation performance by over 8%, without the need for model re-training or fine-tuning. In comparison, modifications to model architecture typically lead to marginal gains of less than 3%. Motivated by this observation, we introduce ShapeKit, a flexible and easy-to-integrate toolkit designed to refine anatomical shapes. This work highlights the underappreciated value of shape-based tools and calls attention to their potential impact within the medical segmentation community.
>
---
#### [new 244] CRISP-SAM2: SAM2 with Cross-Modal Interaction and Semantic Prompting for Multi-Organ Segmentation
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于多器官医学图像分割任务，旨在解决模型细节不准确、依赖几何提示和空间信息丢失的问题。通过引入跨模态交互和语义提示机制提升分割性能。**

- **链接: [http://arxiv.org/pdf/2506.23121v1](http://arxiv.org/pdf/2506.23121v1)**

> **作者:** Xinlei Yu; Chanmiao Wang; Hui Jin; Ahmed Elazab; Gangyong Jia; Xiang Wan; Changqing Zou; Ruiquan Ge
>
> **备注:** 19 pages, 9 figures, 10 tables
>
> **摘要:** Multi-organ medical segmentation is a crucial component of medical image processing, essential for doctors to make accurate diagnoses and develop effective treatment plans. Despite significant progress in this field, current multi-organ segmentation models often suffer from inaccurate details, dependence on geometric prompts and loss of spatial information. Addressing these challenges, we introduce a novel model named CRISP-SAM2 with CRoss-modal Interaction and Semantic Prompting based on SAM2. This model represents a promising approach to multi-organ medical segmentation guided by textual descriptions of organs. Our method begins by converting visual and textual inputs into cross-modal contextualized semantics using a progressive cross-attention interaction mechanism. These semantics are then injected into the image encoder to enhance the detailed understanding of visual information. To eliminate reliance on geometric prompts, we use a semantic prompting strategy, replacing the original prompt encoder to sharpen the perception of challenging targets. In addition, a similarity-sorting self-updating strategy for memory and a mask-refining process is applied to further adapt to medical imaging and enhance localized details. Comparative experiments conducted on seven public datasets indicate that CRISP-SAM2 outperforms existing models. Extensive analysis also demonstrates the effectiveness of our method, thereby confirming its superior performance, especially in addressing the limitations mentioned earlier. Our code is available at: https://github.com/YU-deep/CRISP\_SAM2.git.
>
---
#### [new 245] MDPG: Multi-domain Diffusion Prior Guidance for MRI Reconstruction
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于MRI图像重建任务，旨在提升重建图像质量。通过引入多域扩散先验引导，结合高效编码与融合策略，增强数据一致性。**

- **链接: [http://arxiv.org/pdf/2506.23701v1](http://arxiv.org/pdf/2506.23701v1)**

> **作者:** Lingtong Zhang; Mengdie Song; Xiaohan Hao; Huayu Mai; Bensheng Qiu
>
> **备注:** Accept by MICCAI2025
>
> **摘要:** Magnetic Resonance Imaging (MRI) reconstruction is essential in medical diagnostics. As the latest generative models, diffusion models (DMs) have struggled to produce high-fidelity images due to their stochastic nature in image domains. Latent diffusion models (LDMs) yield both compact and detailed prior knowledge in latent domains, which could effectively guide the model towards more effective learning of the original data distribution. Inspired by this, we propose Multi-domain Diffusion Prior Guidance (MDPG) provided by pre-trained LDMs to enhance data consistency in MRI reconstruction tasks. Specifically, we first construct a Visual-Mamba-based backbone, which enables efficient encoding and reconstruction of under-sampled images. Then pre-trained LDMs are integrated to provide conditional priors in both latent and image domains. A novel Latent Guided Attention (LGA) is proposed for efficient fusion in multi-level latent domains. Simultaneously, to effectively utilize a prior in both the k-space and image domain, under-sampled images are fused with generated full-sampled images by the Dual-domain Fusion Branch (DFB) for self-adaption guidance. Lastly, to further enhance the data consistency, we propose a k-space regularization strategy based on the non-auto-calibration signal (NACS) set. Extensive experiments on two public MRI datasets fully demonstrate the effectiveness of the proposed methodology. The code is available at https://github.com/Zolento/MDPG.
>
---
#### [new 246] Pixels-to-Graph: Real-time Integration of Building Information Models and Scene Graphs for Semantic-Geometric Human-Robot Understanding
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于人机协作任务，旨在解决机器人与人类在复杂环境中理解与导航的问题。提出Pix2G方法，实时融合图像和LiDAR生成结构化场景图。**

- **链接: [http://arxiv.org/pdf/2506.22593v1](http://arxiv.org/pdf/2506.22593v1)**

> **作者:** Antonello Longo; Chanyoung Chung; Matteo Palieri; Sung-Kyun Kim; Ali Agha; Cataldo Guaragnella; Shehryar Khattak
>
> **备注:** Paper accepted to 2025 IEEE International Conference on Automation Science and Engineering (CASE)
>
> **摘要:** Autonomous robots are increasingly playing key roles as support platforms for human operators in high-risk, dangerous applications. To accomplish challenging tasks, an efficient human-robot cooperation and understanding is required. While typically robotic planning leverages 3D geometric information, human operators are accustomed to a high-level compact representation of the environment, like top-down 2D maps representing the Building Information Model (BIM). 3D scene graphs have emerged as a powerful tool to bridge the gap between human readable 2D BIM and the robot 3D maps. In this work, we introduce Pixels-to-Graph (Pix2G), a novel lightweight method to generate structured scene graphs from image pixels and LiDAR maps in real-time for the autonomous exploration of unknown environments on resource-constrained robot platforms. To satisfy onboard compute constraints, the framework is designed to perform all operation on CPU only. The method output are a de-noised 2D top-down environment map and a structure-segmented 3D pointcloud which are seamlessly connected using a multi-layer graph abstracting information from object-level up to the building-level. The proposed method is quantitatively and qualitatively evaluated during real-world experiments performed using the NASA JPL NeBula-Spot legged robot to autonomously explore and map cluttered garage and urban office like environments in real-time.
>
---
#### [new 247] EXPERT: An Explainable Image Captioning Evaluation Metric with Structured Explanations
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于图像描述评估任务，旨在解决现有评估指标解释不规范的问题。提出EXPERT模型，通过结构化标准提升解释质量。**

- **链接: [http://arxiv.org/pdf/2506.24016v1](http://arxiv.org/pdf/2506.24016v1)**

> **作者:** Hyunjong Kim; Sangyeop Kim; Jongheon Jeong; Yeongjae Cho; Sungzoon Cho
>
> **备注:** Accepted at ACL 2025 Findings
>
> **摘要:** Recent advances in large language models and vision-language models have led to growing interest in explainable evaluation metrics for image captioning. However, these metrics generate explanations without standardized criteria, and the overall quality of the generated explanations remains unverified. In this paper, we propose EXPERT, a reference-free evaluation metric that provides structured explanations based on three fundamental criteria: fluency, relevance, and descriptiveness. By constructing large-scale datasets of high-quality structured explanations, we develop a two-stage evaluation template to effectively supervise a vision-language model for both scoring and explanation generation. EXPERT achieves state-of-the-art results on benchmark datasets while providing significantly higher-quality explanations than existing metrics, as validated through comprehensive human evaluation. Our code and datasets are available at https://github.com/hjkim811/EXPERT.
>
---
#### [new 248] ReMem: Mutual Information-Aware Fine-tuning of Pretrained Vision Transformers for Effective Knowledge Distillation
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于知识蒸馏任务，旨在解决从强预训练模型中有效迁移知识的问题。通过引入互信息感知的微调方法提升蒸馏效果。**

- **链接: [http://arxiv.org/pdf/2506.23041v1](http://arxiv.org/pdf/2506.23041v1)**

> **作者:** Chengyu Dong; Huan Gui; Noveen Sachdeva; Long Jin; Ke Yin; Jingbo Shang; Lichan Hong; Ed H. Chi; Zhe Zhao
>
> **摘要:** Knowledge distillation from pretrained visual representation models offers an effective approach to improve small, task-specific production models. However, the effectiveness of such knowledge transfer drops significantly when distilling from strong models that are pretrained in a large scale. In this paper, we address this challenge for pretrained Vision Transformers (ViTs) by exploring methods to fine-tune them for more effective knowledge transfer. Motivated by the connection between mutual information and distillation effectiveness, we propose to employ mutual information-aware optimization during finetuning. For small or highly-imbalanced downstream datasets where such optimization becomes less effective, we introduce a simple yet effective heuristic of reweighting MLP blocks. This approach is inspired by our observation that top MLP blocks are primarily responsible for mutual information loss. Our method enables small student models to benefit from those pretrained models among the strongest.
>
---
#### [new 249] Supercm: Revisiting Clustering for Semi-Supervised Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于半监督学习任务，旨在解决如何有效利用少量标注数据和大量未标注数据的问题。通过引入可微聚类模块，提升模型性能并简化训练策略。**

- **链接: [http://arxiv.org/pdf/2506.23824v1](http://arxiv.org/pdf/2506.23824v1)**

> **作者:** Durgesh Singh; Ahcene Boubekki; Robert Jenssen; Michael C. Kampffmeyer
>
> **摘要:** The development of semi-supervised learning (SSL) has in recent years largely focused on the development of new consistency regularization or entropy minimization approaches, often resulting in models with complex training strategies to obtain the desired results. In this work, we instead propose a novel approach that explicitly incorporates the underlying clustering assumption in SSL through extending a recently proposed differentiable clustering module. Leveraging annotated data to guide the cluster centroids results in a simple end-to-end trainable deep SSL approach. We demonstrate that the proposed model improves the performance over the supervised-only baseline and show that our framework can be used in conjunction with other SSL methods to further boost their performance.
>
---
#### [new 250] Teaching Time Series to See and Speak: Forecasting with Aligned Visual and Textual Perspectives
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于时间序列预测任务，旨在解决传统方法难以捕捉语义模式的问题。通过构建视觉与文本模态并进行对比学习，提升预测效果。**

- **链接: [http://arxiv.org/pdf/2506.24124v1](http://arxiv.org/pdf/2506.24124v1)**

> **作者:** Dong Sixun; Fan Wei; Teresa Wu; Fu Yanjie
>
> **备注:** Code: https://github.com/Ironieser/TimesCLIP
>
> **摘要:** Time series forecasting traditionally relies on unimodal numerical inputs, which often struggle to capture high-level semantic patterns due to their dense and unstructured nature. While recent approaches have explored representing time series as text using large language models (LLMs), these methods remain limited by the discrete nature of token sequences and lack the perceptual intuition humans typically apply, such as interpreting visual patterns. In this paper, we propose a multimodal contrastive learning framework that transforms raw time series into structured visual and textual perspectives. Rather than using natural language or real-world images, we construct both modalities directly from numerical sequences. We then align these views in a shared semantic space via contrastive learning, enabling the model to capture richer and more complementary representations. Furthermore, we introduce a variate selection module that leverages the aligned representations to identify the most informative variables for multivariate forecasting. Extensive experiments on fifteen short-term and six long-term forecasting benchmarks demonstrate that our approach consistently outperforms strong unimodal and cross-modal baselines, highlighting the effectiveness of multimodal alignment in enhancing time series forecasting. Code is available at: https://github.com/Ironieser/TimesCLIP.
>
---
#### [new 251] Maximum Dispersion, Maximum Concentration: Enhancing the Quality of MOP Solutions
- **分类: math.OC; cs.CV**

- **简介: 该论文属于多目标优化任务，旨在解决如何平衡解的分散性与集中性问题。通过定义ROI并优化决策空间的均匀性，提升解的质量与多样性。**

- **链接: [http://arxiv.org/pdf/2506.22568v1](http://arxiv.org/pdf/2506.22568v1)**

> **作者:** Gladston Moreira; Ivan Meneghini; Elzabeth Wanner
>
> **备注:** 11 pages
>
> **摘要:** Multi-objective optimization problems (MOPs) often require a trade-off between conflicting objectives, maximizing diversity and convergence in the objective space. This study presents an approach to improve the quality of MOP solutions by optimizing the dispersion in the decision space and the convergence in a specific region of the objective space. Our approach defines a Region of Interest (ROI) based on a cone representing the decision maker's preferences in the objective space, while enhancing the dispersion of solutions in the decision space using a uniformity measure. Combining solution concentration in the objective space with dispersion in the decision space intensifies the search for Pareto-optimal solutions while increasing solution diversity. When combined, these characteristics improve the quality of solutions and avoid the bias caused by clustering solutions in a specific region of the decision space. Preliminary experiments suggest that this method enhances multi-objective optimization by generating solutions that effectively balance dispersion and concentration, thereby mitigating bias in the decision space.
>
---
#### [new 252] Hierarchical Characterization of Brain Dynamics via State Space-based Vector Quantization
- **分类: eess.IV; cs.CV; q-bio.NC**

- **简介: 该论文属于脑动态表征任务，旨在解决脑状态转换建模与稳定表示问题。提出HST网络，通过层次化向量量化实现脑状态与转换的精准表征。**

- **链接: [http://arxiv.org/pdf/2506.22952v1](http://arxiv.org/pdf/2506.22952v1)**

> **作者:** Yanwu Yang; Thomas Wolfers
>
> **摘要:** Understanding brain dynamics through functional Magnetic Resonance Imaging (fMRI) remains a fundamental challenge in neuroscience, particularly in capturing how the brain transitions between various functional states. Recently, metastability, which refers to temporarily stable brain states, has offered a promising paradigm to quantify complex brain signals into interpretable, discretized representations. In particular, compared to cluster-based machine learning approaches, tokenization approaches leveraging vector quantization have shown promise in representation learning with powerful reconstruction and predictive capabilities. However, most existing methods ignore brain transition dependencies and lack a quantification of brain dynamics into representative and stable embeddings. In this study, we propose a Hierarchical State space-based Tokenization network, termed HST, which quantizes brain states and transitions in a hierarchical structure based on a state space-based model. We introduce a refined clustered Vector-Quantization Variational AutoEncoder (VQ-VAE) that incorporates quantization error feedback and clustering to improve quantization performance while facilitating metastability with representative and stable token representations. We validate our HST on two public fMRI datasets, demonstrating its effectiveness in quantifying the hierarchical dynamics of the brain and its potential in disease diagnosis and reconstruction performance. Our method offers a promising framework for the characterization of brain dynamics, facilitating the analysis of metastability.
>
---
#### [new 253] InfGen: Scenario Generation as Next Token Group Prediction
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出InfGen，用于生成动态交通场景，解决传统方法无法建模长期复杂场景的问题。通过Transformer模型实现连续场景生成，提升自动驾驶训练效果。**

- **链接: [http://arxiv.org/pdf/2506.23316v1](http://arxiv.org/pdf/2506.23316v1)**

> **作者:** Zhenghao Peng; Yuxin Liu; Bolei Zhou
>
> **摘要:** Realistic and interactive traffic simulation is essential for training and evaluating autonomous driving systems. However, most existing data-driven simulation methods rely on static initialization or log-replay data, limiting their ability to model dynamic, long-horizon scenarios with evolving agent populations. We propose InfGen, a scenario generation framework that outputs agent states and trajectories in an autoregressive manner. InfGen represents the entire scene as a sequence of tokens, including traffic light signals, agent states, and motion vectors, and uses a transformer model to simulate traffic over time. This design enables InfGen to continuously insert new agents into traffic, supporting infinite scene generation. Experiments demonstrate that InfGen produces realistic, diverse, and adaptive traffic behaviors. Furthermore, reinforcement learning policies trained in InfGen-generated scenarios achieve superior robustness and generalization, validating its utility as a high-fidelity simulation environment for autonomous driving. More information is available at https://metadriverse.github.io/infgen/.
>
---
#### [new 254] Forget-MI: Machine Unlearning for Forgetting Multimodal Information in Healthcare Settings
- **分类: cs.LG; cs.CR; cs.CV**

- **简介: 该论文属于机器遗忘任务，旨在解决医疗多模态数据隐私保护问题。提出Forget-MI方法，有效删除指定数据同时保持模型性能。**

- **链接: [http://arxiv.org/pdf/2506.23145v1](http://arxiv.org/pdf/2506.23145v1)**

> **作者:** Shahad Hardan; Darya Taratynova; Abdelmajid Essofi; Karthik Nandakumar; Mohammad Yaqub
>
> **摘要:** Privacy preservation in AI is crucial, especially in healthcare, where models rely on sensitive patient data. In the emerging field of machine unlearning, existing methodologies struggle to remove patient data from trained multimodal architectures, which are widely used in healthcare. We propose Forget-MI, a novel machine unlearning method for multimodal medical data, by establishing loss functions and perturbation techniques. Our approach unlearns unimodal and joint representations of the data requested to be forgotten while preserving knowledge from the remaining data and maintaining comparable performance to the original model. We evaluate our results using performance on the forget dataset, performance on the test dataset, and Membership Inference Attack (MIA), which measures the attacker's ability to distinguish the forget dataset from the training dataset. Our model outperforms the existing approaches that aim to reduce MIA and the performance on the forget dataset while keeping an equivalent performance on the test set. Specifically, our approach reduces MIA by 0.202 and decreases AUC and F1 scores on the forget set by 0.221 and 0.305, respectively. Additionally, our performance on the test set matches that of the retrained model, while allowing forgetting. Code is available at https://github.com/BioMedIA-MBZUAI/Forget-MI.git
>
---
#### [new 255] General Autonomous Cybersecurity Defense: Learning Robust Policies for Dynamic Topologies and Diverse Attackers
- **分类: cs.CR; cs.AI; cs.CV; stat.ML**

- **简介: 该论文属于自主网络安全防御任务，旨在解决动态网络环境中防御策略泛化不足的问题。通过学习鲁棒策略提升防御系统的适应能力。**

- **链接: [http://arxiv.org/pdf/2506.22706v1](http://arxiv.org/pdf/2506.22706v1)**

> **作者:** Arun Ramamurthy; Neil Dhir
>
> **摘要:** In the face of evolving cyber threats such as malware, ransomware and phishing, autonomous cybersecurity defense (ACD) systems have become essential for real-time threat detection and response with optional human intervention. However, existing ACD systems rely on limiting assumptions, particularly the stationarity of the underlying network dynamics. In real-world scenarios, network topologies can change due to actions taken by attackers or defenders, system failures, or time evolution of networks, leading to failures in the adaptive capabilities of current defense agents. Moreover, many agents are trained on static environments, resulting in overfitting to specific topologies, which hampers their ability to generalize to out-of-distribution network topologies. This work addresses these challenges by exploring methods for developing agents to learn generalizable policies across dynamic network environments -- general ACD (GACD).
>
---
#### [new 256] Deep Learning in Mild Cognitive Impairment Diagnosis using Eye Movements and Image Content in Visual Memory Tasks
- **分类: cs.HC; cs.CV**

- **简介: 该论文属于MCI诊断任务，旨在通过眼动和图像内容数据区分健康人与MCI患者，采用深度学习模型进行分类分析。**

- **链接: [http://arxiv.org/pdf/2506.23016v1](http://arxiv.org/pdf/2506.23016v1)**

> **作者:** Tomás Silva Santos Rocha; Anastasiia Mikhailova; Moreno I. Coco; José Santos-Victor
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** The global prevalence of dementia is projected to double by 2050, highlighting the urgent need for scalable diagnostic tools. This study utilizes digital cognitive tasks with eye-tracking data correlated with memory processes to distinguish between Healthy Controls (HC) and Mild Cognitive Impairment (MCI), a precursor to dementia. A deep learning model based on VTNet was trained using eye-tracking data from 44 participants (24 MCI, 20 HCs) who performed a visual memory task. The model utilizes both time series and spatial data derived from eye-tracking. It was modified to incorporate scan paths, heat maps, and image content. These modifications also enabled testing parameters such as image resolution and task performance, analyzing their impact on model performance. The best model, utilizing $700\times700px$ resolution heatmaps, achieved 68% sensitivity and 76% specificity. Despite operating under more challenging conditions (e.g., smaller dataset size, shorter task duration, or a less standardized task), the model's performance is comparable to an Alzheimer's study using similar methods (70% sensitivity and 73% specificity). These findings contribute to the development of automated diagnostic tools for MCI. Future work should focus on refining the model and using a standardized long-term visual memory task.
>
---
#### [new 257] High Resolution Isotropic 3D Cine imaging with Automated Segmentation using Concatenated 2D Real-time Imaging and Deep Learning
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于医学影像处理任务，旨在通过深度学习从2D实时图像生成3D心脏 cine 数据，解决传统CMR耗时且受限的问题。**

- **链接: [http://arxiv.org/pdf/2506.22532v1](http://arxiv.org/pdf/2506.22532v1)**

> **作者:** Mark Wrobel; Michele Pascale; Tina Yao; Ruaraidh Campbell; Elena Milano; Michael Quail; Jennifer Steeden; Vivek Muthurangu
>
> **摘要:** Background: Conventional cardiovascular magnetic resonance (CMR) in paediatric and congenital heart disease uses 2D, breath-hold, balanced steady state free precession (bSSFP) cine imaging for assessment of function and cardiac-gated, respiratory-navigated, static 3D bSSFP whole-heart imaging for anatomical assessment. Our aim is to concatenate a stack 2D free-breathing real-time cines and use Deep Learning (DL) to create an isotropic a fully segmented 3D cine dataset from these images. Methods: Four DL models were trained on open-source data that performed: a) Interslice contrast correction; b) Interslice respiratory motion correction; c) Super-resolution (slice direction); and d) Segmentation of right and left atria and ventricles (RA, LA, RV, and LV), thoracic aorta (Ao) and pulmonary arteries (PA). In 10 patients undergoing routine cardiovascular examination, our method was validated on prospectively acquired sagittal stacks of real-time cine images. Quantitative metrics (ventricular volumes and vessel diameters) and image quality of the 3D cines were compared to conventional breath hold cine and whole heart imaging. Results: All real-time data were successfully transformed into 3D cines with a total post-processing time of <1 min in all cases. There were no significant biases in any LV or RV metrics with reasonable limits of agreement and correlation. There is also reasonable agreement for all vessel diameters, although there was a small but significant overestimation of RPA diameter. Conclusion: We have demonstrated the potential of creating a 3D-cine data from concatenated 2D real-time cine images using a series of DL models. Our method has short acquisition and reconstruction times with fully segmented data being available within 2 minutes. The good agreement with conventional imaging suggests that our method could help to significantly speed up CMR in clinical practice.
>
---
#### [new 258] TAG-WM: Tamper-Aware Generative Image Watermarking via Diffusion Inversion Sensitivity
- **分类: cs.MM; cs.CV; eess.IV; I.3.3; I.4.9**

- **简介: 该论文属于图像水印任务，旨在解决AI生成内容的版权保护与篡改检测问题。提出TAG-WM方法，实现高鲁棒性和定位能力的水印嵌入与检测。**

- **链接: [http://arxiv.org/pdf/2506.23484v1](http://arxiv.org/pdf/2506.23484v1)**

> **作者:** Yuzhuo Chen; Zehua Ma; Han Fang; Weiming Zhang; Nenghai Yu
>
> **备注:** Accepted by ICCV 2025 (2025 IEEE/CVF International Conference on Computer Vision)
>
> **摘要:** AI-generated content (AIGC) enables efficient visual creation but raises copyright and authenticity risks. As a common technique for integrity verification and source tracing, digital image watermarking is regarded as a potential solution to above issues. Among these, watermarking methods capable of preserving the generation quality are receiving increased attention. However, the proliferation and high performance of generative image editing applications have elevated the risks of malicious tampering, creating new demands. 1) The tamper robustness of current lossless visual quality watermarks remains constrained by the modification-sensitive diffusion inversion process, necessitating enhanced robustness. 2) The improved tampering quality and rapid iteration cycles render passive tampering detection methods inadequate, making proactive tampering localization capability a desired feature for watermarks. To address these requirements, this paper proposes a Tamper-Aware Generative image WaterMarking method named TAG-WM. The proposed method comprises four key modules: a dual-mark joint sampling (DMJS) algorithm for embedding copyright and localization watermarks into the latent space while preserving generative quality, the watermark latent reconstruction (WLR) utilizing reversed DMJS, a dense variation region detector (DVRD) leveraging diffusion inversion sensitivity to identify tampered areas via statistical deviation analysis, and the tamper-aware decoding (TAD) guided by localization results. The experimental results indicate that TAG-WM achieves SOTA tampering robustness and tampering localization capability with distortions while maintaining lossless generation quality and a considerable capacity of 256 bits.
>
---
#### [new 259] MMReason: An Open-Ended Multi-Modal Multi-Step Reasoning Benchmark for MLLMs Toward AGI
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态大模型推理任务，旨在解决现有基准在长链推理评估上的不足。提出MMReason基准，通过多样化问题、开放格式和评分机制提升评估准确性。**

- **链接: [http://arxiv.org/pdf/2506.23563v1](http://arxiv.org/pdf/2506.23563v1)**

> **作者:** Huanjin Yao; Jiaxing Huang; Yawen Qiu; Michael K. Chen; Wenzheng Liu; Wei Zhang; Wenjie Zeng; Xikun Zhang; Jingyi Zhang; Yuxin Song; Wenhao Wu; Dacheng Tao
>
> **备注:** Technical report
>
> **摘要:** Reasoning plays a crucial role in advancing Multimodal Large Language Models (MLLMs) toward Artificial General Intelligence. However, existing MLLM benchmarks often fall short in precisely and comprehensively evaluating long-chain reasoning abilities from three key aspects: (1) lack of difficulty and diversity, (2) susceptibility to guessability and memorization, (3) inadequate assessment of intermediate reasoning steps. To fill this gap, we introduce MMReason, a new benchmark designed to precisely and comprehensively evaluate MLLM long-chain reasoning capability with diverse, open-ended, challenging questions. First, we curate challenging questions requiring multi-step reasoning from various fields (i.e., 6 disciplines) and multiple difficulty levels (i.e., from pre-university to university, and from foundational to competition tiers). Second, these questions are reformulated into an open-ended format and filtered using a multi-model voting technique to eliminate shortcut cases related to guessing and memorization, ensuring robust reasoning evaluations. Third, we annotate the questions with detailed step-by-step solutions, and design a reference-based ternary scoring mechanism to reliably assess intermediate reasoning steps. With MMReason, we benchmark popular leading MLLMs and provide an in-depth analysis of their reasoning capabilities. We hope MMReason will serve as a valuable resource for advancing MLLM reasoning research. Code will be available at https://github.com/HJYao00/MMReason.
>
---
#### [new 260] AFUNet: Cross-Iterative Alignment-Fusion Synergy for HDR Reconstruction via Deep Unfolding Paradigm
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于HDR图像重建任务，旨在解决多曝光LDR图像融合中的对齐与融合问题，提出AFUNet网络通过交替优化实现更优性能。**

- **链接: [http://arxiv.org/pdf/2506.23537v1](http://arxiv.org/pdf/2506.23537v1)**

> **作者:** Xinyue Li; Zhangkai Ni; Wenhan Yang
>
> **备注:** Accepted to International Conference on Computer Vision (ICCV) 2025
>
> **摘要:** Existing learning-based methods effectively reconstruct HDR images from multi-exposure LDR inputs with extended dynamic range and improved detail, but they rely more on empirical design rather than theoretical foundation, which can impact their reliability. To address these limitations, we propose the cross-iterative Alignment and Fusion deep Unfolding Network (AFUNet), where HDR reconstruction is systematically decoupled into two interleaved subtasks -- alignment and fusion -- optimized through alternating refinement, achieving synergy between the two subtasks to enhance the overall performance. Our method formulates multi-exposure HDR reconstruction from a Maximum A Posteriori (MAP) estimation perspective, explicitly incorporating spatial correspondence priors across LDR images and naturally bridging the alignment and fusion subproblems through joint constraints. Building on the mathematical foundation, we reimagine traditional iterative optimization through unfolding -- transforming the conventional solution process into an end-to-end trainable AFUNet with carefully designed modules that work progressively. Specifically, each iteration of AFUNet incorporates an Alignment-Fusion Module (AFM) that alternates between a Spatial Alignment Module (SAM) for alignment and a Channel Fusion Module (CFM) for adaptive feature fusion, progressively bridging misaligned content and exposure discrepancies. Extensive qualitative and quantitative evaluations demonstrate AFUNet's superior performance, consistently surpassing state-of-the-art methods. Our code is available at: https://github.com/eezkni/AFUNet
>
---
#### [new 261] VoteSplat: Hough Voting Gaussian Splatting for 3D Scene Understanding
- **分类: cs.GR; cs.CV; cs.LG**

- **简介: 该论文提出VoteSplat，解决3D场景理解中的对象定位与语义映射问题，结合Hough投票与3DGS，提升效率与准确性。**

- **链接: [http://arxiv.org/pdf/2506.22799v1](http://arxiv.org/pdf/2506.22799v1)**

> **作者:** Minchao Jiang; Shunyu Jia; Jiaming Gu; Xiaoyuan Lu; Guangming Zhu; Anqi Dong; Liang Zhang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** 3D Gaussian Splatting (3DGS) has become horsepower in high-quality, real-time rendering for novel view synthesis of 3D scenes. However, existing methods focus primarily on geometric and appearance modeling, lacking deeper scene understanding while also incurring high training costs that complicate the originally streamlined differentiable rendering pipeline. To this end, we propose VoteSplat, a novel 3D scene understanding framework that integrates Hough voting with 3DGS. Specifically, Segment Anything Model (SAM) is utilized for instance segmentation, extracting objects, and generating 2D vote maps. We then embed spatial offset vectors into Gaussian primitives. These offsets construct 3D spatial votes by associating them with 2D image votes, while depth distortion constraints refine localization along the depth axis. For open-vocabulary object localization, VoteSplat maps 2D image semantics to 3D point clouds via voting points, reducing training costs associated with high-dimensional CLIP features while preserving semantic unambiguity. Extensive experiments demonstrate effectiveness of VoteSplat in open-vocabulary 3D instance localization, 3D point cloud understanding, click-based 3D object localization, hierarchical segmentation, and ablation studies. Our code is available at https://sy-ja.github.io/votesplat/
>
---
#### [new 262] Federated Breast Cancer Detection Enhanced by Synthetic Ultrasound Image Augmentation
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于乳腺癌超声图像分类任务，旨在解决联邦学习中数据不足和分布不均的问题。通过生成对抗网络生成合成图像，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.23334v1](http://arxiv.org/pdf/2506.23334v1)**

> **作者:** Hongyi Pan; Ziliang Hong; Gorkem Durak; Ziyue Xu; Ulas Bagci
>
> **摘要:** Federated learning (FL) has emerged as a promising paradigm for collaboratively training deep learning models across institutions without exchanging sensitive medical data. However, its effectiveness is often hindered by limited data availability and non-independent, identically distributed data across participating clients, which can degrade model performance and generalization. To address these challenges, we propose a generative AI based data augmentation framework that integrates synthetic image sharing into the federated training process for breast cancer diagnosis via ultrasound images. Specifically, we train two simple class-specific Deep Convolutional Generative Adversarial Networks: one for benign and one for malignant lesions. We then simulate a realistic FL setting using three publicly available breast ultrasound image datasets: BUSI, BUS-BRA, and UDIAT. FedAvg and FedProx are adopted as baseline FL algorithms. Experimental results show that incorporating a suitable number of synthetic images improved the average AUC from 0.9206 to 0.9237 for FedAvg and from 0.9429 to 0.9538 for FedProx. We also note that excessive use of synthetic data reduced performance, underscoring the importance of maintaining a balanced ratio of real and synthetic samples. Our findings highlight the potential of generative AI based data augmentation to enhance FL results in the breast ultrasound image classification task.
>
---
#### [new 263] Score-based Diffusion Model for Unpaired Virtual Histology Staining
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于虚拟染色任务，旨在解决H&E与IHC图像配对不足的问题，提出一种基于互信息引导的扩散模型，实现结构一致的IHC生成。**

- **链接: [http://arxiv.org/pdf/2506.23184v1](http://arxiv.org/pdf/2506.23184v1)**

> **作者:** Anran Liu; Xiaofei Wang; Jing Cai; Chao Li
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Hematoxylin and eosin (H&E) staining visualizes histology but lacks specificity for diagnostic markers. Immunohistochemistry (IHC) staining provides protein-targeted staining but is restricted by tissue availability and antibody specificity. Virtual staining, i.e., computationally translating the H&E image to its IHC counterpart while preserving the tissue structure, is promising for efficient IHC generation. Existing virtual staining methods still face key challenges: 1) effective decomposition of staining style and tissue structure, 2) controllable staining process adaptable to diverse tissue and proteins, and 3) rigorous structural consistency modelling to handle the non-pixel-aligned nature of paired H&E and IHC images. This study proposes a mutual-information (MI)-guided score-based diffusion model for unpaired virtual staining. Specifically, we design 1) a global MI-guided energy function that disentangles the tissue structure and staining characteristics across modalities, 2) a novel timestep-customized reverse diffusion process for precise control of the staining intensity and structural reconstruction, and 3) a local MI-driven contrastive learning strategy to ensure the cellular level structural consistency between H&E-IHC images. Extensive experiments demonstrate the our superiority over state-of-the-art approaches, highlighting its biomedical potential. Codes will be open-sourced upon acceptance.
>
---
#### [new 264] Wireless Home Automation Using Social Networking Websites
- **分类: cs.NI; cs.CR; cs.CV**

- **简介: 该论文属于智能家居任务，旨在解决传统家居自动化系统的安全与操作便捷性问题。通过整合社交平台认证，实现对家用设备的智能控制。**

- **链接: [http://arxiv.org/pdf/2506.22482v1](http://arxiv.org/pdf/2506.22482v1)**

> **作者:** Divya Alok Gupta; Dwith Chenna; B. Aditya Vighnesh Ramakanth
>
> **备注:** 20th Annual International Conference on Advanced Computing and Communications (ADCOM) 2014
>
> **摘要:** With the advent of Internet of Things, Wireless Home Automation Systems WHAS are gradually gaining popularity. These systems are faced with multiple challenges such as security; controlling a variety of home appliances with a single interface and user friendliness. In this paper we propose a system that uses secure authentication systems of social networking websites such as Twitter, tracks the end-users activities on the social network and then control his or her domestic appliances. At the end, we highlight the applications of the proposed WHAS and compare the advantages of our proposed system over traditional home automation systems.
>
---
#### [new 265] Single Image Inpainting and Super-Resolution with Simultaneous Uncertainty Guarantees by Universal Reproducing Kernels
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于图像修复与超分辨率任务，解决缺失像素估计问题，并提供不确定性量化。通过RKHS方法实现同时置信区间估计。**

- **链接: [http://arxiv.org/pdf/2506.23221v1](http://arxiv.org/pdf/2506.23221v1)**

> **作者:** Bálint Horváth; Balázs Csanád Csáji
>
> **备注:** 23 pages, 8 figures, 6 tables
>
> **摘要:** The paper proposes a statistical learning approach to the problem of estimating missing pixels of images, crucial for image inpainting and super-resolution problems. One of the main novelties of the method is that it also provides uncertainty quantifications together with the estimated values. Our core assumption is that the underlying data-generating function comes from a Reproducing Kernel Hilbert Space (RKHS). A special emphasis is put on band-limited functions, central to signal processing, which form Paley-Wiener type RKHSs. The proposed method, which we call Simultaneously Guaranteed Kernel Interpolation (SGKI), is an extension and refinement of a recently developed kernel method. An advantage of SGKI is that it not only estimates the missing pixels, but also builds non-asymptotic confidence bands for the unobserved values, which are simultaneously guaranteed for all missing pixels. We also show how to compute these bands efficiently using Schur complements, we discuss a generalization to vector-valued functions, and we present a series of numerical experiments on various datasets containing synthetically generated and benchmark images, as well.
>
---
#### [new 266] Deep Learning-Based Semantic Segmentation for Real-Time Kidney Imaging and Measurements with Augmented Reality-Assisted Ultrasound
- **分类: eess.IV; cs.AI; cs.CV; cs.HC; cs.LG**

- **简介: 该论文属于医学影像分析任务，旨在解决超声检查中手动测量耗时、易疲劳的问题，通过深度学习与增强现实技术实现肾脏实时自动分割与测量。**

- **链接: [http://arxiv.org/pdf/2506.23721v1](http://arxiv.org/pdf/2506.23721v1)**

> **作者:** Gijs Luijten; Roberto Maria Scardigno; Lisle Faray de Paiva; Peter Hoyer; Jens Kleesiek; Domenico Buongiorno; Vitoantonio Bevilacqua; Jan Egger
>
> **摘要:** Ultrasound (US) is widely accessible and radiation-free but has a steep learning curve due to its dynamic nature and non-standard imaging planes. Additionally, the constant need to shift focus between the US screen and the patient poses a challenge. To address these issues, we integrate deep learning (DL)-based semantic segmentation for real-time (RT) automated kidney volumetric measurements, which are essential for clinical assessment but are traditionally time-consuming and prone to fatigue. This automation allows clinicians to concentrate on image interpretation rather than manual measurements. Complementing DL, augmented reality (AR) enhances the usability of US by projecting the display directly into the clinician's field of view, improving ergonomics and reducing the cognitive load associated with screen-to-patient transitions. Two AR-DL-assisted US pipelines on HoloLens-2 are proposed: one streams directly via the application programming interface for a wireless setup, while the other supports any US device with video output for broader accessibility. We evaluate RT feasibility and accuracy using the Open Kidney Dataset and open-source segmentation models (nnU-Net, Segmenter, YOLO with MedSAM and LiteMedSAM). Our open-source GitHub pipeline includes model implementations, measurement algorithms, and a Wi-Fi-based streaming solution, enhancing US training and diagnostics, especially in point-of-care settings.
>
---
#### [new 267] Sample Margin-Aware Recalibration of Temperature Scaling
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于模型校准任务，旨在解决神经网络过自信问题。提出SMART方法，通过logit间隙和软分箱ECE优化，实现高效精准校准。**

- **链接: [http://arxiv.org/pdf/2506.23492v1](http://arxiv.org/pdf/2506.23492v1)**

> **作者:** Haolan Guo; Linwei Tao; Haoyang Luo; Minjing Dong; Chang Xu
>
> **摘要:** Recent advances in deep learning have significantly improved predictive accuracy. However, modern neural networks remain systematically overconfident, posing risks for deployment in safety-critical scenarios. Current post-hoc calibration methods face a fundamental dilemma: global approaches like Temperature Scaling apply uniform adjustments across all samples, introducing high bias despite computational efficiency, while more expressive methods that operate on full logit distributions suffer from high variance due to noisy high-dimensional inputs and insufficient validation data. To address these challenges, we propose Sample Margin-Aware Recalibration of Temperature (SMART), a lightweight, data-efficient recalibration method that precisely scales logits based on the margin between the top two logits -- termed the logit gap. Specifically, the logit gap serves as a denoised, scalar signal directly tied to decision boundary uncertainty, providing a robust indicator that avoids the noise inherent in high-dimensional logit spaces while preserving model prediction invariance. Meanwhile, SMART employs a novel soft-binned Expected Calibration Error (SoftECE) objective that balances model bias and variance through adaptive binning, enabling stable parameter updates even with extremely limited calibration data. Extensive evaluations across diverse datasets and architectures demonstrate that SMART achieves state-of-the-art calibration performance even with substantially fewer parameters compared to existing parametric methods, offering a principled, robust, and highly efficient solution for practical uncertainty quantification in neural network predictions. The source code is available at: https://anonymous.4open.science/r/SMART-8B11.
>
---
#### [new 268] SoMi-ToM: Evaluating Multi-Perspective Theory of Mind in Embodied Social Interactions
- **分类: cs.CL; cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出SoMi-ToM基准，用于评估多视角理论心智在具身社交互动中的能力。旨在解决现有基准与真实社交互动差距大的问题，通过多模态数据和双视角评估方法进行模型评测。**

- **链接: [http://arxiv.org/pdf/2506.23046v1](http://arxiv.org/pdf/2506.23046v1)**

> **作者:** Xianzhe Fan; Xuhui Zhou; Chuanyang Jin; Kolby Nottingham; Hao Zhu; Maarten Sap
>
> **备注:** 23 pages, 6 figures
>
> **摘要:** Humans continuously infer the states, goals, and behaviors of others by perceiving their surroundings in dynamic, real-world social interactions. However, most Theory of Mind (ToM) benchmarks only evaluate static, text-based scenarios, which have a significant gap compared to real interactions. We propose the SoMi-ToM benchmark, designed to evaluate multi-perspective ToM in embodied multi-agent complex social interactions. This benchmark is based on rich multimodal interaction data generated by the interaction environment SoMi, covering diverse crafting goals and social relationships. Our framework supports multi-level evaluation: (1) first-person evaluation provides multimodal (visual, dialogue, action, etc.) input from a first-person perspective during a task for real-time state inference, (2) third-person evaluation provides complete third-person perspective video and text records after a task for goal and behavior inference. This evaluation method allows for a more comprehensive examination of a model's ToM capabilities from both the subjective immediate experience and the objective global observation. We constructed a challenging dataset containing 35 third-person perspective videos, 363 first-person perspective images, and 1225 expert-annotated multiple-choice questions (three options). On this dataset, we systematically evaluated the performance of human subjects and several state-of-the-art large vision-language models (LVLMs). The results show that LVLMs perform significantly worse than humans on SoMi-ToM: the average accuracy gap between humans and models is 40.1% in first-person evaluation and 26.4% in third-person evaluation. This indicates that future LVLMs need to further improve their ToM capabilities in embodied, complex social interactions.
>
---
#### [new 269] BPD-Neo: An MRI Dataset for Lung-Trachea Segmentation with Clinical Data for Neonatal Bronchopulmonary Dysplasia
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决新生儿肺气道分割问题。通过构建包含MRI数据和临床信息的BPD-Neo数据集，支持BPD研究与算法开发。**

- **链接: [http://arxiv.org/pdf/2506.23305v1](http://arxiv.org/pdf/2506.23305v1)**

> **作者:** Rachit Saluja; Arzu Kovanlikaya; Candace Chien; Lauren Kathryn Blatt; Jeffrey M. Perlman; Stefan Worgall; Mert R. Sabuncu; Jonathan P. Dyke
>
> **摘要:** Bronchopulmonary dysplasia (BPD) is a common complication among preterm neonates, with portable X-ray imaging serving as the standard diagnostic modality in neonatal intensive care units (NICUs). However, lung magnetic resonance imaging (MRI) offers a non-invasive alternative that avoids sedation and radiation while providing detailed insights into the underlying mechanisms of BPD. Leveraging high-resolution 3D MRI data, advanced image processing and semantic segmentation algorithms can be developed to assist clinicians in identifying the etiology of BPD. In this dataset, we present MRI scans paired with corresponding semantic segmentations of the lungs and trachea for 40 neonates, the majority of whom are diagnosed with BPD. The imaging data consist of free-breathing 3D stack-of-stars radial gradient echo acquisitions, known as the StarVIBE series. Additionally, we provide comprehensive clinical data and baseline segmentation models, validated against clinical assessments, to support further research and development in neonatal lung imaging.
>
---
#### [new 270] MedRegion-CT: Region-Focused Multimodal LLM for Comprehensive 3D CT Report Generation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学影像报告生成任务，旨在解决现有方法忽略局部区域细节的问题。通过引入区域聚焦的多模态模型，提升报告的准确性和临床相关性。**

- **链接: [http://arxiv.org/pdf/2506.23102v1](http://arxiv.org/pdf/2506.23102v1)**

> **作者:** Sunggu Kyung; Jinyoung Seo; Hyunseok Lim; Dongyeong Kim; Hyungbin Park; Jimin Sung; Jihyun Kim; Wooyoung Jo; Yoojin Nam; Namkug Kim
>
> **备注:** 14 pages, 5 figures, submitted to ICCV 2025
>
> **摘要:** The recent release of RadGenome-Chest CT has significantly advanced CT-based report generation. However, existing methods primarily focus on global features, making it challenging to capture region-specific details, which may cause certain abnormalities to go unnoticed. To address this, we propose MedRegion-CT, a region-focused Multi-Modal Large Language Model (MLLM) framework, featuring three key innovations. First, we introduce Region Representative ($R^2$) Token Pooling, which utilizes a 2D-wise pretrained vision model to efficiently extract 3D CT features. This approach generates global tokens representing overall slice features and region tokens highlighting target areas, enabling the MLLM to process comprehensive information effectively. Second, a universal segmentation model generates pseudo-masks, which are then processed by a mask encoder to extract region-centric features. This allows the MLLM to focus on clinically relevant regions, using six predefined region masks. Third, we leverage segmentation results to extract patient-specific attributions, including organ size, diameter, and locations. These are converted into text prompts, enriching the MLLM's understanding of patient-specific contexts. To ensure rigorous evaluation, we conducted benchmark experiments on report generation using the RadGenome-Chest CT. MedRegion-CT achieved state-of-the-art performance, outperforming existing methods in natural language generation quality and clinical relevance while maintaining interpretability. The code for our framework is publicly available.
>
---
#### [new 271] ICME 2025 Generalizable HDR and SDR Video Quality Measurement Grand Challenge
- **分类: eess.IV; cs.CV; cs.MM**

- **简介: 该论文属于视频质量评估任务，旨在解决HDR和SDR视频质量测量的泛化问题。通过组织竞赛，评估并提升VQA模型在不同动态范围内容上的性能。**

- **链接: [http://arxiv.org/pdf/2506.22790v1](http://arxiv.org/pdf/2506.22790v1)**

> **作者:** Yixu Chen; Bowen Chen; Hai Wei; Alan C. Bovik; Baojun Li; Wei Sun; Linhan Cao; Kang Fu; Dandan Zhu; Jun Jia; Menghan Hu; Xiongkuo Min; Guangtao Zhai; Dounia Hammou; Fei Yin; Rafal Mantiuk; Amritha Premkumar; Prajit T Rajendran; Vignesh V Menon
>
> **备注:** ICME 2025 Grand Challenges
>
> **摘要:** This paper reports IEEE International Conference on Multimedia \& Expo (ICME) 2025 Grand Challenge on Generalizable HDR and SDR Video Quality Measurement. With the rapid development of video technology, especially High Dynamic Range (HDR) and Standard Dynamic Range (SDR) contents, the need for robust and generalizable Video Quality Assessment (VQA) methods has become increasingly demanded. Existing VQA models often struggle to deliver consistent performance across varying dynamic ranges, distortion types, and diverse content. This challenge was established to benchmark and promote VQA approaches capable of jointly handling HDR and SDR content. In the final evaluation phase, five teams submitted seven models along with technical reports to the Full Reference (FR) and No Reference (NR) tracks. Among them, four methods outperformed VMAF baseline, while the top-performing model achieved state-of-the-art performance, setting a new benchmark for generalizable video quality assessment.
>
---
#### [new 272] Denoising Multi-Color QR Codes and Stiefel-Valued Data by Relaxed Regularizations
- **分类: math.OC; cs.CV; cs.NA; math.NA; 94A08, 94A12, 65J22, 90C22, 90C25**

- **简介: 该论文属于图像去噪任务，旨在解决多色QR码和Stiefel值数据的去噪问题，提出基于TV和Tikhonov的去噪模型及高效凸化方法。**

- **链接: [http://arxiv.org/pdf/2506.22826v1](http://arxiv.org/pdf/2506.22826v1)**

> **作者:** Robert Beinert; Jonas Bresch
>
> **备注:** 9 pages, 2 figures, 3 algorithms
>
> **摘要:** The handling of manifold-valued data, for instance, plays a central role in color restoration tasks relying on circle- or sphere-valued color models, in the study of rotational or directional information related to the special orthogonal group, and in Gaussian image processing, where the pixel statistics are interpreted as values on the hyperbolic sheet. Especially, to denoise these kind of data, there have been proposed several generalizations of total variation (TV) and Tikhonov-type denoising models incorporating the underlying manifolds. Recently, a novel, numerically efficient denoising approach has been introduced, where the data are embedded in an Euclidean ambient space, the non-convex manifolds are encoded by a series of positive semi-definite, fixed-rank matrices, and the rank constraint is relaxed to obtain a convexification that can be solved using standard algorithms from convex analysis. The aim of the present paper is to extent this approach to new kinds of data like multi-binary and Stiefel-valued data. Multi-binary data can, for instance, be used to model multi-color QR codes whereas Stiefel-valued data occur in image and video-based recognition. For both new data types, we propose TV- and Tikhonov-based denoising modelstogether with easy-to-solve convexification. All derived methods are evaluated on proof-of-concept, synthetic experiments.
>
---
#### [new 273] DriveBLIP2: Attention-Guided Explanation Generation for Complex Driving Scenarios
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于自动驾驶中的解释生成任务，旨在解决复杂场景下模型解释性不足的问题。通过引入注意力机制，提升模型在关键区域的聚焦能力，从而生成更准确的解释。**

- **链接: [http://arxiv.org/pdf/2506.22494v1](http://arxiv.org/pdf/2506.22494v1)**

> **作者:** Shihong Ling; Yue Wan; Xiaowei Jia; Na Du
>
> **备注:** Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025. 7 pages, 3 figures
>
> **摘要:** This paper introduces a new framework, DriveBLIP2, built upon the BLIP2-OPT architecture, to generate accurate and contextually relevant explanations for emerging driving scenarios. While existing vision-language models perform well in general tasks, they encounter difficulties in understanding complex, multi-object environments, particularly in real-time applications such as autonomous driving, where the rapid identification of key objects is crucial. To address this limitation, an Attention Map Generator is proposed to highlight significant objects relevant to driving decisions within critical video frames. By directing the model's focus to these key regions, the generated attention map helps produce clear and relevant explanations, enabling drivers to better understand the vehicle's decision-making process in critical situations. Evaluations on the DRAMA dataset reveal significant improvements in explanation quality, as indicated by higher BLEU, ROUGE, CIDEr, and SPICE scores compared to baseline models. These findings underscore the potential of targeted attention mechanisms in vision-language models for enhancing explainability in real-time autonomous driving.
>
---
#### [new 274] The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于视觉语言模型的测试时适应任务，旨在解决现有TTA方法评估不全面、效果有限的问题，通过构建基准TTA-VLM进行系统评估与分析。**

- **链接: [http://arxiv.org/pdf/2506.24000v1](http://arxiv.org/pdf/2506.24000v1)**

> **作者:** Lijun Sheng; Jian Liang; Ran He; Zilei Wang; Tieniu Tan
>
> **备注:** Github link: https://github.com/TomSheng21/tta-vlm
>
> **摘要:** Test-time adaptation (TTA) methods have gained significant attention for enhancing the performance of vision-language models (VLMs) such as CLIP during inference, without requiring additional labeled data. However, current TTA researches generally suffer from major limitations such as duplication of baseline results, limited evaluation metrics, inconsistent experimental settings, and insufficient analysis. These problems hinder fair comparisons between TTA methods and obscure their practical strengths and weaknesses. To address these challenges, we introduce TTA-VLM, a comprehensive benchmark for evaluating TTA methods on VLMs. Our benchmark implements 8 episodic TTA and 7 online TTA methods within a unified and reproducible framework, and evaluates them across 15 widely used datasets. Unlike prior studies focused solely on CLIP, we extend the evaluation to SigLIP--a model trained with a Sigmoid loss--and include training-time tuning methods such as CoOp, MaPLe, and TeCoA to assess generality. Beyond classification accuracy, TTA-VLM incorporates various evaluation metrics, including robustness, calibration, out-of-distribution detection, and stability, enabling a more holistic assessment of TTA methods. Through extensive experiments, we find that 1) existing TTA methods produce limited gains compared to the previous pioneering work; 2) current TTA methods exhibit poor collaboration with training-time fine-tuning methods; 3) accuracy gains frequently come at the cost of reduced model trustworthiness. We release TTA-VLM to provide fair comparison and comprehensive evaluation of TTA methods for VLMs, and we hope it encourages the community to develop more reliable and generalizable TTA strategies.
>
---
#### [new 275] Artificial Intelligence-assisted Pixel-level Lung (APL) Scoring for Fast and Accurate Quantification in Ultra-short Echo-time MRI
- **分类: eess.IV; cs.AI; cs.CV; physics.med-ph**

- **简介: 该论文属于医学影像分析任务，旨在解决肺部MRI定量评估难题。通过AI实现像素级肺评分，提升速度与准确性。**

- **链接: [http://arxiv.org/pdf/2506.23506v1](http://arxiv.org/pdf/2506.23506v1)**

> **作者:** Bowen Xin; Rohan Hickey; Tamara Blake; Jin Jin; Claire E Wainwright; Thomas Benkert; Alto Stemmer; Peter Sly; David Coman; Jason Dowling
>
> **备注:** Oral presentation in ISMRM2025
>
> **摘要:** Lung magnetic resonance imaging (MRI) with ultrashort echo-time (UTE) represents a recent breakthrough in lung structure imaging, providing image resolution and quality comparable to computed tomography (CT). Due to the absence of ionising radiation, MRI is often preferred over CT in paediatric diseases such as cystic fibrosis (CF), one of the most common genetic disorders in Caucasians. To assess structural lung damage in CF imaging, CT scoring systems provide valuable quantitative insights for disease diagnosis and progression. However, few quantitative scoring systems are available in structural lung MRI (e.g., UTE-MRI). To provide fast and accurate quantification in lung MRI, we investigated the feasibility of novel Artificial intelligence-assisted Pixel-level Lung (APL) scoring for CF. APL scoring consists of 5 stages, including 1) image loading, 2) AI lung segmentation, 3) lung-bounded slice sampling, 4) pixel-level annotation, and 5) quantification and reporting. The results shows that our APL scoring took 8.2 minutes per subject, which was more than twice as fast as the previous grid-level scoring. Additionally, our pixel-level scoring was statistically more accurate (p=0.021), while strongly correlating with grid-level scoring (R=0.973, p=5.85e-9). This tool has great potential to streamline the workflow of UTE lung MRI in clinical settings, and be extended to other structural lung MRI sequences (e.g., BLADE MRI), and for other lung diseases (e.g., bronchopulmonary dysplasia).
>
---
#### [new 276] SurgTPGS: Semantic 3D Surgical Scene Understanding with Text Promptable Gaussian Splatting
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于3D手术场景理解任务，旨在解决实时文本提示的3D重建与语义理解问题。通过引入语义特征学习和优化方法，提升手术环境的精确重建与分析。**

- **链接: [http://arxiv.org/pdf/2506.23309v1](http://arxiv.org/pdf/2506.23309v1)**

> **作者:** Yiming Huang; Long Bai; Beilei Cui; Kun Yuan; Guankun Wang; Mobarakol Islam; Nicolas Padoy; Nassir Navab; Hongliang Ren
>
> **备注:** MICCAI 2025. Project Page: https://lastbasket.github.io/MICCAI-2025-SurgTPGS/
>
> **摘要:** In contemporary surgical research and practice, accurately comprehending 3D surgical scenes with text-promptable capabilities is particularly crucial for surgical planning and real-time intra-operative guidance, where precisely identifying and interacting with surgical tools and anatomical structures is paramount. However, existing works focus on surgical vision-language model (VLM), 3D reconstruction, and segmentation separately, lacking support for real-time text-promptable 3D queries. In this paper, we present SurgTPGS, a novel text-promptable Gaussian Splatting method to fill this gap. We introduce a 3D semantics feature learning strategy incorporating the Segment Anything model and state-of-the-art vision-language models. We extract the segmented language features for 3D surgical scene reconstruction, enabling a more in-depth understanding of the complex surgical environment. We also propose semantic-aware deformation tracking to capture the seamless deformation of semantic features, providing a more precise reconstruction for both texture and semantic features. Furthermore, we present semantic region-aware optimization, which utilizes regional-based semantic information to supervise the training, particularly promoting the reconstruction quality and semantic smoothness. We conduct comprehensive experiments on two real-world surgical datasets to demonstrate the superiority of SurgTPGS over state-of-the-art methods, highlighting its potential to revolutionize surgical practices. SurgTPGS paves the way for developing next-generation intelligent surgical systems by enhancing surgical precision and safety. Our code is available at: https://github.com/lastbasket/SurgTPGS.
>
---
#### [new 277] Towards Efficient and Accurate Spiking Neural Networks via Adaptive Bit Allocation
- **分类: cs.NE; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于神经网络优化任务，旨在解决多比特SNN中资源浪费与精度不平衡问题，通过自适应位分配策略提升效率与准确性。**

- **链接: [http://arxiv.org/pdf/2506.23717v1](http://arxiv.org/pdf/2506.23717v1)**

> **作者:** Xingting Yao; Qinghao Hu; Fei Zhou; Tielong Liu; Gang Li; Peisong Wang; Jian Cheng
>
> **摘要:** Multi-bit spiking neural networks (SNNs) have recently become a heated research spot, pursuing energy-efficient and high-accurate AI. However, with more bits involved, the associated memory and computation demands escalate to the point where the performance improvements become disproportionate. Based on the insight that different layers demonstrate different importance and extra bits could be wasted and interfering, this paper presents an adaptive bit allocation strategy for direct-trained SNNs, achieving fine-grained layer-wise allocation of memory and computation resources. Thus, SNN's efficiency and accuracy can be improved. Specifically, we parametrize the temporal lengths and the bit widths of weights and spikes, and make them learnable and controllable through gradients. To address the challenges caused by changeable bit widths and temporal lengths, we propose the refined spiking neuron, which can handle different temporal lengths, enable the derivation of gradients for temporal lengths, and suit spike quantization better. In addition, we theoretically formulate the step-size mismatch problem of learnable bit widths, which may incur severe quantization errors to SNN, and accordingly propose the step-size renewal mechanism to alleviate this issue. Experiments on various datasets, including the static CIFAR and ImageNet and the dynamic CIFAR-DVS and DVS-GESTURE, demonstrate that our methods can reduce the overall memory and computation cost while achieving higher accuracy. Particularly, our SEWResNet-34 can achieve a 2.69\% accuracy gain and 4.16$\times$ lower bit budgets over the advanced baseline work on ImageNet. This work will be fully open-sourced.
>
---
#### [new 278] Improving Myocardial Infarction Detection via Synthetic ECG Pretraining
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于心肌梗死检测任务，旨在解决真实ECG数据不足的问题。通过生成合成ECG并预训练模型，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2506.23259v1](http://arxiv.org/pdf/2506.23259v1)**

> **作者:** Lachin Naghashyar
>
> **摘要:** Myocardial infarction is a major cause of death globally, and accurate early diagnosis from electrocardiograms (ECGs) remains a clinical priority. Deep learning models have shown promise for automated ECG interpretation, but require large amounts of labeled data, which are often scarce in practice. We propose a physiology-aware pipeline that (i) synthesizes 12-lead ECGs with tunable MI morphology and realistic noise, and (ii) pre-trains recurrent and transformer classifiers with self-supervised masked-autoencoding plus a joint reconstruction-classification objective. We validate the realism of synthetic ECGs via statistical and visual analysis, confirming that key morphological features are preserved. Pretraining on synthetic data consistently improved classification performance, particularly in low-data settings, with AUC gains of up to 4 percentage points. These results show that controlled synthetic ECGs can help improve MI detection when real clinical data is limited.
>
---
#### [new 279] Navigating with Annealing Guidance Scale in Diffusion Space
- **分类: cs.GR; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于文本到图像生成任务，解决CFG引导尺度选择困难的问题，提出一种动态调整引导尺度的调度器，提升生成质量与文本对齐度。**

- **链接: [http://arxiv.org/pdf/2506.24108v1](http://arxiv.org/pdf/2506.24108v1)**

> **作者:** Shai Yehezkel; Omer Dahary; Andrey Voynov; Daniel Cohen-Or
>
> **备注:** Project page: https://annealing-guidance.github.io/annealing-guidance/
>
> **摘要:** Denoising diffusion models excel at generating high-quality images conditioned on text prompts, yet their effectiveness heavily relies on careful guidance during the sampling process. Classifier-Free Guidance (CFG) provides a widely used mechanism for steering generation by setting the guidance scale, which balances image quality and prompt alignment. However, the choice of the guidance scale has a critical impact on the convergence toward a visually appealing and prompt-adherent image. In this work, we propose an annealing guidance scheduler which dynamically adjusts the guidance scale over time based on the conditional noisy signal. By learning a scheduling policy, our method addresses the temperamental behavior of CFG. Empirical results demonstrate that our guidance scheduler significantly enhances image quality and alignment with the text prompt, advancing the performance of text-to-image generation. Notably, our novel scheduler requires no additional activations or memory consumption, and can seamlessly replace the common classifier-free guidance, offering an improved trade-off between prompt alignment and quality.
>
---
#### [new 280] FD-DiT: Frequency Domain-Directed Diffusion Transformer for Low-Dose CT Reconstruction
- **分类: eess.IV; cs.CV; physics.med-ph**

- **简介: 该论文属于低剂量CT图像重建任务，旨在解决噪声和细节丢失问题。提出FD-DiT模型，结合频率域引导和动态融合策略，提升重建质量。**

- **链接: [http://arxiv.org/pdf/2506.23466v1](http://arxiv.org/pdf/2506.23466v1)**

> **作者:** Qiqing Liu; Guoquan Wei; Zekun Zhou; Yiyang Wen; Liu Shi; Qiegen Liu
>
> **备注:** 11pages, 11 figures
>
> **摘要:** Low-dose computed tomography (LDCT) reduces radiation exposure but suffers from image artifacts and loss of detail due to quantum and electronic noise, potentially impacting diagnostic accuracy. Transformer combined with diffusion models has been a promising approach for image generation. Nevertheless, existing methods exhibit limitations in preserving finegrained image details. To address this issue, frequency domain-directed diffusion transformer (FD-DiT) is proposed for LDCT reconstruction. FD-DiT centers on a diffusion strategy that progressively introduces noise until the distribution statistically aligns with that of LDCT data, followed by denoising processing. Furthermore, we employ a frequency decoupling technique to concentrate noise primarily in high-frequency domain, thereby facilitating effective capture of essential anatomical structures and fine details. A hybrid denoising network is then utilized to optimize the overall data reconstruction process. To enhance the capability in recognizing high-frequency noise, we incorporate sliding sparse local attention to leverage the sparsity and locality of shallow-layer information, propagating them via skip connections for improving feature representation. Finally, we propose a learnable dynamic fusion strategy for optimal component integration. Experimental results demonstrate that at identical dose levels, LDCT images reconstructed by FD-DiT exhibit superior noise and artifact suppression compared to state-of-the-art methods.
>
---
## 更新

#### [replaced 001] Time-R1: Post-Training Large Vision Language Model for Temporal Video Grounding
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.13377v3](http://arxiv.org/pdf/2503.13377v3)**

> **作者:** Ye Wang; Ziheng Wang; Boshen Xu; Yang Du; Kejun Lin; Zihan Xiao; Zihao Yue; Jianzhong Ju; Liang Zhang; Dingyi Yang; Xiangnan Fang; Zewen He; Zhenbo Luo; Wenxuan Wang; Junqi Lin; Jian Luan; Qin Jin
>
> **备注:** Project Page: https://xuboshen.github.io/Time-R1/
>
> **摘要:** Temporal Video Grounding (TVG), the task of locating specific video segments based on language queries, is a core challenge in long-form video understanding. While recent Large Vision-Language Models (LVLMs) have shown early promise in tackling TVG through supervised fine-tuning (SFT), their abilities to generalize remain limited. To address this, we propose a novel post-training framework that enhances the generalization capabilities of LVLMs via reinforcement learning (RL). Specifically, our contributions span three key directions: (1) Time-R1: we introduce a reasoning-guided post-training framework via RL with verifiable reward to enhance the capabilities of LVLMs on the TVG task. (2) TimeRFT: we explore data-efficient post-training strategies on our curated RL-friendly dataset, which trains the model to progressively comprehend difficult samples, leading to better generalization. (3) TVGBench: we carefully construct a small yet comprehensive benchmark for LVLM evaluation, assessing 11 types of queries and featuring balanced distributions across both videos and queries. Extensive experiments demonstrate that Time-R1 achieves state-of-the-art performance across multiple downstream datasets using only 2.5K training data, while improving its general video understanding capabilities.
>
---
#### [replaced 002] MMInA: Benchmarking Multihop Multimodal Internet Agents
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2404.09992v2](http://arxiv.org/pdf/2404.09992v2)**

> **作者:** Shulin Tian; Ziniu Zhang; Liangyu Chen; Ziwei Liu
>
> **备注:** ACL 2025 findings. The live leaderboard is at https://mmina.cliangyu.com/
>
> **摘要:** Autonomous embodied agents live on an Internet of multimedia websites. Can they hop around multimodal websites to complete complex user tasks? Existing benchmarks fail to assess them in a realistic, evolving environment for their embodiment across websites. To answer this question, we present MMInA, a multihop and multimodal benchmark to evaluate the embodied agents for compositional Internet tasks, with several appealing properties: 1) Evolving real-world multimodal websites. Our benchmark uniquely operates on evolving real-world websites, ensuring a high degree of realism and applicability to natural user tasks. Our data includes 1,050 human-written tasks covering various domains such as shopping and travel, with each task requiring the agent to extract multimodal information from web pages as observations autonomously; 2) Multihop web browsing. Our dataset features naturally compositional tasks that require information from or actions on multiple websites to solve, to assess long-range reasoning capabilities on web tasks; 3) Holistic evaluation. We propose a novel protocol for evaluating an agent's progress in completing multihop tasks. We experiment with both standalone (multimodal) language models and heuristic-based web agents. Extensive experiments demonstrate that while long-chain multihop web tasks are easy for humans, they remain challenging for state-of-the-art web agents. We identify that agents are more likely to fail on the early hops when solving tasks with more hops, which results in lower task success rates. To address this issue, we propose a simple memory augmentation approach that replays past action trajectories to reflect. Our method significantly improves the performance of both the single-hop and multihop web browsing abilities. Our code and data are available at github.com/shulin16/MMInA.
>
---
#### [replaced 003] GLIMPSE: Gradient-Layer Importance Mapping for Prompted Visual Saliency Explanation for Generative LVLMs
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.18985v2](http://arxiv.org/pdf/2506.18985v2)**

> **作者:** Guanxi Shen
>
> **摘要:** Recent progress in large vision-language models (LVLMs) has advanced the state of the art in visual question answering (VQA). However, interpreting where LVLMs direct their visual attention while generating free-form responses remains a significant challenge, yet is essential for understanding model behavior. We introduce GLIMPSE (Gradient-Layer Importance Mapping for Prompted Visual Saliency Explanation), a lightweight, model-agnostic framework that jointly attributes LVLM outputs to the most relevant visual evidence and textual signals supporting open-ended VQA. GLIMPSE fuses gradient-weighted attention, adaptive layer propagation, and relevance-weighted token aggregation to produce holistic response-level heat maps for interpreting cross-modal reasoning, outperforming prior interpretability methods and pushing the state-of-the-art in human-alignment. We demonstrate an analytic explainable AI (XAI) approach using GLIMPSE to uncover fine-grained insights into LVLM cross-modal attribution, trace reasoning dynamics, analyze systematic human-attention misalignment, diagnose hallucination, expose bias, and ensure transparency.
>
---
#### [replaced 004] Enhancing Diffusion Posterior Sampling for Inverse Problems by Integrating Crafted Measurements
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.09850v2](http://arxiv.org/pdf/2411.09850v2)**

> **作者:** Shijie Zhou; Huaisheng Zhu; Rohan Sharma; Jiayi Chen; Ruiyi Zhang; Kaiyi Ji; Changyou Chen
>
> **摘要:** Diffusion models have emerged as a powerful foundation model for visual generations. With an appropriate sampling process, it can effectively serve as a generative prior for solving general inverse problems. Current posterior sampling-based methods take the measurement (i.e., degraded image sample) into the posterior sampling to infer the distribution of the target data (i.e., clean image sample). However, in this manner, we show that high-frequency information can be prematurely introduced during the early stages, which could induce larger posterior estimate errors during restoration sampling. To address this observation, we first reveal that forming the log-posterior gradient with the noisy measurement ( i.e., noisy measurement from a diffusion forward process) instead of the clean one can benefit the early posterior sampling. Consequently, we propose a novel diffusion posterior sampling method DPS-CM, which incorporates a Crafted Measurement (i.e., noisy measurement crafted by a reverse denoising process, rather than constructed from the diffusion forward process) to form the posterior estimate. This integration aims to mitigate the misalignment with the diffusion prior caused by cumulative posterior estimate errors. Experimental results demonstrate that our approach significantly improves the overall capacity to solve general and noisy inverse problems, such as Gaussian deblurring, super-resolution, inpainting, nonlinear deblurring, and tasks with Poisson noise, relative to existing approaches. Code is available at: https://github.com/sjz5202/DPS-CM.
>
---
#### [replaced 005] MrTrack: Register Mamba for Needle Tracking with Rapid Reciprocating Motion during Ultrasound-Guided Aspiration Biopsy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.09450v2](http://arxiv.org/pdf/2505.09450v2)**

> **作者:** Yuelin Zhang; Qingpeng Ding; Long Lei; Yongxuan Feng; Raymond Shing-Yan Tang; Shing Shin Cheng
>
> **备注:** Early Accepted by MICCAI 2025
>
> **摘要:** Ultrasound-guided fine needle aspiration (FNA) biopsy is a common minimally invasive diagnostic procedure. However, an aspiration needle tracker addressing rapid reciprocating motion is still missing. MrTrack, an aspiration needle tracker with a mamba-based register mechanism, is proposed. MrTrack leverages a Mamba-based register extractor to sequentially distill global context from each historical search map, storing these temporal cues in a register bank. The Mamba-based register retriever then retrieves temporal prompts from the register bank to provide external cues when current vision features are temporarily unusable due to rapid reciprocating motion and imaging degradation. A self-supervised register diversify loss is proposed to encourage feature diversity and dimension independence within the learned register, mitigating feature collapse. Comprehensive experiments conducted on both robotic and manual aspiration biopsy datasets demonstrate that MrTrack not only outperforms state-of-the-art trackers in accuracy and robustness but also achieves superior inference efficiency. Project page: https://github.com/PieceZhang/MrTrack
>
---
#### [replaced 006] Uncertainty-Aware Remaining Lifespan Prediction from Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.13430v2](http://arxiv.org/pdf/2506.13430v2)**

> **作者:** Tristan Kenneweg; Philip Kenneweg; Barbara Hammer
>
> **备注:** Submitted to ISVC 2025
>
> **摘要:** Predicting mortality-related outcomes from images offers the prospect of accessible, noninvasive, and scalable health screening. We present a method that leverages pretrained vision transformer foundation models to estimate remaining lifespan from facial and whole-body images, alongside robust uncertainty quantification. We show that predictive uncertainty varies systematically with the true remaining lifespan, and that this uncertainty can be effectively modeled by learning a Gaussian distribution for each sample. Our approach achieves state-of-the-art mean absolute error (MAE) of 7.48 years on an established dataset, and further improves to 4.79 and 5.07 years MAE on two new, higher-quality datasets curated and published in this work. Importantly, our models provide well-calibrated uncertainty estimates, as demonstrated by a bucketed expected calibration error of 0.62 years. While not intended for clinical deployment, these results highlight the potential of extracting medically relevant signals from images. We make all code and datasets available to facilitate further research.
>
---
#### [replaced 007] Mitigating Knowledge Discrepancies among Multiple Datasets for Task-agnostic Unified Face Alignment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.22359v2](http://arxiv.org/pdf/2503.22359v2)**

> **作者:** Jiahao Xia; Min Xu; Wenjian Huang; Jianguo Zhang; Haimin Zhang; Chunxia Xiao
>
> **备注:** 24 Pages, 9 Figures, accepted to IJCV-2025
>
> **摘要:** Despite the similar structures of human faces, existing face alignment methods cannot learn unified knowledge from multiple datasets with different landmark annotations. The limited training samples in a single dataset commonly result in fragile robustness in this field. To mitigate knowledge discrepancies among different datasets and train a task-agnostic unified face alignment (TUFA) framework, this paper presents a strategy to unify knowledge from multiple datasets. Specifically, we calculate a mean face shape for each dataset. To explicitly align these mean shapes on an interpretable plane based on their semantics, each shape is then incorporated with a group of semantic alignment embeddings. The 2D coordinates of these aligned shapes can be viewed as the anchors of the plane. By encoding them into structure prompts and further regressing the corresponding facial landmarks using image features, a mapping from the plane to the target faces is finally established, which unifies the learning target of different datasets. Consequently, multiple datasets can be utilized to boost the generalization ability of the model. The successful mitigation of discrepancies also enhances the efficiency of knowledge transferring to a novel dataset, significantly boosts the performance of few-shot face alignment. Additionally, the interpretable plane endows TUFA with a task-agnostic characteristic, enabling it to locate landmarks unseen during training in a zero-shot manner. Extensive experiments are carried on seven benchmarks and the results demonstrate an impressive improvement in face alignment brought by knowledge discrepancies mitigation. The code is available at https://github.com/Jiahao-UTS/TUFA.
>
---
#### [replaced 008] APVR: Hour-Level Long Video Understanding with Adaptive Pivot Visual Information Retrieval
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.04953v2](http://arxiv.org/pdf/2506.04953v2)**

> **作者:** Hong Gao; Yiming Bao; Xuezhen Tu; Bin Zhong; Minling Zhang
>
> **摘要:** Current multimodal large language models (MLLMs) struggle with hour-level video understanding, facing significant challenges not only in modeling the substantial information volume of long videos but also in overcoming the memory wall and resource constraints during both training and inference. Although recent training-free approaches have alleviated resource demands by compressing visual features, their reliance on incomplete visual information limits the performance potential. To address these limitations, we propose \textbf{A}daptive \textbf{P}ivot \textbf{V}isual information \textbf{R}etrieval (\textbf{APVR}), a training-free framework that hierarchically retrieves and retains sufficient and important visual information. It breakthroughs the memory wall limitation via two complementary components: Pivot Frame Retrieval employs query expansion and iterative spatio-semantic confidence scoring to identify relevant video frames, and Pivot Token Retrieval performs query-aware attention-driven token selection within up to 1024 pivot frames. This dual granularity approach enables the processing of hour-long videos while maintaining semantic fidelity. Experimental validations demonstrate significant performance improvements, achieving 64.9\% on LongVideoBench and 68.4\% on VideoMME, which are state-of-the-art results for both training-free and training-based approaches. Meanwhile, our method provides plug-and-play integration capability with existing MLLM architectures.
>
---
#### [replaced 009] AWF: Adaptive Weight Fusion for Enhanced Class Incremental Semantic Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.08516v2](http://arxiv.org/pdf/2409.08516v2)**

> **作者:** Zechao Sun; Shuying Piao; Haolin Jin; Chang Dong; Lin Yue; Weitong Chen; Luping Zhou
>
> **备注:** 10 pages,6 figures
>
> **摘要:** Class Incremental Semantic Segmentation (CISS) aims to mitigate catastrophic forgetting by maintaining a balance between previously learned and newly introduced knowledge. Existing methods, primarily based on regularization techniques like knowledge distillation, help preserve old knowledge but often face challenges in effectively integrating new knowledge, resulting in limited overall improvement. Endpoints Weight Fusion (EWF) method, while simple, effectively addresses some of these limitations by dynamically fusing the model weights from previous steps with those from the current step, using a fusion parameter alpha determined by the relative number of previously known classes and newly introduced classes. However, the simplicity of the alpha calculation may limit its ability to fully capture the complexities of different task scenarios, potentially leading to suboptimal fusion outcomes. In this paper, we propose an enhanced approach called Adaptive Weight Fusion (AWF), which introduces an alternating training strategy for the fusion parameter, allowing for more flexible and adaptive weight integration. AWF achieves superior performance by better balancing the retention of old knowledge with the learning of new classes, significantly improving results on benchmark CISS tasks compared to the original EWF. And our experiment code will be released on Github.
>
---
#### [replaced 010] BandRC: Band Shifted Raised Cosine Activated Implicit Neural Representations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11640v2](http://arxiv.org/pdf/2505.11640v2)**

> **作者:** Pandula Thennakoon; Avishka Ranasinghe; Mario De Silva; Buwaneka Epakanda; Roshan Godaliyadda; Parakrama Ekanayake; Vijitha Herath
>
> **摘要:** In recent years, implicit neural representations (INRs) have gained popularity in the computer vision community. This is mainly due to the strong performance of INRs in many computer vision tasks. These networks can extract a continuous signal representation given a discrete signal representation. In previous studies, it has been repeatedly shown that INR performance has a strong correlation with the activation functions used in its multilayer perceptrons. Although numerous activation functions have been proposed that are competitive with one another, they share some common set of challenges such as spectral bias(Lack of sensitivity to high-frequency content in signals), limited robustness to signal noise and difficulties in simultaneous capturing both local and global features. and furthermore, the requirement for manual parameter tuning. To address these issues, we introduce a novel activation function, Band Shifted Raised Cosine Activated Implicit Neural Networks $\textbf{(BandRC)}$ tailored to enhance signal representation capacity further. We also incorporate deep prior knowledge extracted from the signal to adjust the activation functions through a task-specific model. Through a mathematical analysis and a series of experiments which include image reconstruction (with an average PSNR improvement of +5.67 dB over the nearest counterpart across a diverse image dataset), denoising (with a +0.46 dB increase in PSNR), super-resolution (with a +1.03 dB improvement over the nearest State-Of-The-Art (SOTA) method for 6X super-resolution), inpainting, and 3D shape reconstruction we demonstrate the dominance of BandRC over existing state of the art activation functions.
>
---
#### [replaced 011] DisCoPatch: Taming Adversarially-driven Batch Statistics for Improved Out-of-Distribution Detection
- **分类: cs.CV; cs.AI; eess.IV**

- **链接: [http://arxiv.org/pdf/2501.08005v4](http://arxiv.org/pdf/2501.08005v4)**

> **作者:** Francisco Caetano; Christiaan Viviers; Luis A. Zavala-Mondragón; Peter H. N. de With; Fons van der Sommen
>
> **备注:** ICCV 2025
>
> **摘要:** Out-of-distribution (OOD) detection holds significant importance across many applications. While semantic and domain-shift OOD problems are well-studied, this work focuses on covariate shifts - subtle variations in the data distribution that can degrade machine learning performance. We hypothesize that detecting these subtle shifts can improve our understanding of in-distribution boundaries, ultimately improving OOD detection. In adversarial discriminators trained with Batch Normalization (BN), real and adversarial samples form distinct domains with unique batch statistics - a property we exploit for OOD detection. We introduce DisCoPatch, an unsupervised Adversarial Variational Autoencoder (VAE) framework that harnesses this mechanism. During inference, batches consist of patches from the same image, ensuring a consistent data distribution that allows the model to rely on batch statistics. DisCoPatch uses the VAE's suboptimal outputs (generated and reconstructed) as negative samples to train the discriminator, thereby improving its ability to delineate the boundary between in-distribution samples and covariate shifts. By tightening this boundary, DisCoPatch achieves state-of-the-art results in public OOD detection benchmarks. The proposed model not only excels in detecting covariate shifts, achieving 95.5% AUROC on ImageNet-1K(-C) but also outperforms all prior methods on public Near-OOD (95.0%) benchmarks. With a compact model size of 25MB, it achieves high OOD detection performance at notably lower latency than existing methods, making it an efficient and practical solution for real-world OOD detection applications. The code is publicly available.
>
---
#### [replaced 012] Seg-Zero: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.06520v2](http://arxiv.org/pdf/2503.06520v2)**

> **作者:** Yuqi Liu; Bohao Peng; Zhisheng Zhong; Zihao Yue; Fanbin Lu; Bei Yu; Jiaya Jia
>
> **摘要:** Traditional methods for reasoning segmentation rely on supervised fine-tuning with categorical labels and simple descriptions, limiting its out-of-domain generalization and lacking explicit reasoning processes. To address these limitations, we propose Seg-Zero, a novel framework that demonstrates remarkable generalizability and derives explicit chain-of-thought reasoning through cognitive reinforcement. Seg-Zero introduces a decoupled architecture consisting of a reasoning model and a segmentation model. The reasoning model interprets user intentions, generates explicit reasoning chains, and produces positional prompts, which are subsequently used by the segmentation model to generate precious pixel-level masks. We design a sophisticated reward mechanism that integrates both format and accuracy rewards to effectively guide optimization directions. Trained exclusively via reinforcement learning with GRPO and without explicit reasoning data, Seg-Zero achieves robust zero-shot generalization and exhibits emergent test-time reasoning capabilities. Experiments show that Seg-Zero-7B achieves a zero-shot performance of 57.5 on the ReasonSeg benchmark, surpassing the prior LISA-7B by 18\%. This significant improvement highlights Seg-Zero's ability to generalize across domains while presenting an explicit reasoning process. Code is available at https://github.com/dvlab-research/Seg-Zero.
>
---
#### [replaced 013] Meta-LoRA: Meta-Learning LoRA Components for Domain-Aware ID Personalization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.22352v3](http://arxiv.org/pdf/2503.22352v3)**

> **作者:** Barış Batuhan Topal; Umut Özyurt; Zafer Doğan Budak; Ramazan Gokberk Cinbis
>
> **摘要:** Recent advancements in text-to-image generative models, particularly latent diffusion models (LDMs), have demonstrated remarkable capabilities in synthesizing high-quality images from textual prompts. However, achieving identity personalization-ensuring that a model consistently generates subject-specific outputs from limited reference images-remains a fundamental challenge. To address this, we introduce Meta-Low-Rank Adaptation (Meta-LoRA), a novel framework that leverages meta-learning to encode domain-specific priors into LoRA-based identity personalization. Our method introduces a structured three-layer LoRA architecture that separates identity-agnostic knowledge from identity-specific adaptation. In the first stage, the LoRA Meta-Down layers are meta-trained across multiple subjects, learning a shared manifold that captures general identity-related features. In the second stage, only the LoRA-Mid and LoRA-Up layers are optimized to specialize on a given subject, significantly reducing adaptation time while improving identity fidelity. To evaluate our approach, we introduce Meta-PHD, a new benchmark dataset for identity personalization, and compare Meta-LoRA against state-of-the-art methods. Our results demonstrate that Meta-LoRA achieves superior identity retention, computational efficiency, and adaptability across diverse identity conditions. Our code, model weights, and dataset are released on barisbatuhan.github.io/Meta-LoRA.
>
---
#### [replaced 014] Test-Time Reasoning Through Visual Human Preferences with VLMs and Soft Rewards
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.19948v2](http://arxiv.org/pdf/2503.19948v2)**

> **作者:** Alexander Gambashidze; Konstantin Sobolev; Andrey Kuznetsov; Ivan Oseledets
>
> **备注:** We are withdrawing this paper because the main contributions and methodology have significantly changed after further research and experimental updates. The current version no longer reflects our results and main contribution / topic
>
> **摘要:** Can Visual Language Models (VLMs) effectively capture human visual preferences? This work addresses this question by training VLMs to think about preferences at test time, employing reinforcement learning methods inspired by DeepSeek R1 and OpenAI O1. Using datasets such as ImageReward and Human Preference Score v2 (HPSv2), our models achieve accuracies of 64.9% on the ImageReward test set (trained on ImageReward official split) and 65.4% on HPSv2 (trained on approximately 25% of its data). These results match traditional encoder-based models while providing transparent reasoning and enhanced generalization. This approach allows to use not only rich VLM world knowledge, but also its potential to think, yielding interpretable outcomes that help decision-making processes. By demonstrating that human visual preferences reasonable by current VLMs, we introduce efficient soft-reward strategies for image ranking, outperforming simplistic selection or scoring methods. This reasoning capability enables VLMs to rank arbitrary images-regardless of aspect ratio or complexity-thereby potentially amplifying the effectiveness of visual Preference Optimization. By reducing the need for extensive markup while improving reward generalization and explainability, our findings can be a strong mile-stone that will enhance text-to-vision models even further.
>
---
#### [replaced 015] 3DRealCar: An In-the-wild RGB-D Car Dataset with 360-degree Views
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.04875v2](http://arxiv.org/pdf/2406.04875v2)**

> **作者:** Xiaobiao Du; Yida Wang; Haiyang Sun; Zhuojie Wu; Hongwei Sheng; Shuyun Wang; Jiaying Ying; Ming Lu; Tianqing Zhu; Kun Zhan; Xin Yu
>
> **备注:** Project Page: https://xiaobiaodu.github.io/3drealcar
>
> **摘要:** 3D cars are commonly used in self-driving systems, virtual/augmented reality, and games. However, existing 3D car datasets are either synthetic or low-quality, limiting their applications in practical scenarios and presenting a significant gap toward high-quality real-world 3D car datasets. In this paper, we propose the first large-scale 3D real car dataset, termed 3DRealCar, offering three distinctive features. (1) \textbf{High-Volume}: 2,500 cars are meticulously scanned by smartphones, obtaining car images and point clouds with real-world dimensions; (2) \textbf{High-Quality}: Each car is captured in an average of 200 dense, high-resolution 360-degree RGB-D views, enabling high-fidelity 3D reconstruction; (3) \textbf{High-Diversity}: The dataset contains various cars from over 100 brands, collected under three distinct lighting conditions, including reflective, standard, and dark. Additionally, we offer detailed car parsing maps for each instance to promote research in car parsing tasks. Moreover, we remove background point clouds and standardize the car orientation to a unified axis for the reconstruction only on cars and controllable rendering without background. We benchmark 3D reconstruction results with state-of-the-art methods across different lighting conditions in 3DRealCar. Extensive experiments demonstrate that the standard lighting condition part of 3DRealCar can be used to produce a large number of high-quality 3D cars, improving various 2D and 3D tasks related to cars. Notably, our dataset brings insight into the fact that recent 3D reconstruction methods face challenges in reconstructing high-quality 3D cars under reflective and dark lighting conditions. \textcolor{red}{\href{https://xiaobiaodu.github.io/3drealcar/}{Our dataset is here.}}
>
---
#### [replaced 016] Structure-Aware Radar-Camera Depth Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.05008v3](http://arxiv.org/pdf/2506.05008v3)**

> **作者:** Fuyi Zhang; Zhu Yu; Chunhao Li; Runmin Zhang; Xiaokai Bai; Zili Zhou; Si-Yuan Cao; Fang Wang; Hui-Liang Shen
>
> **摘要:** Radar has gained much attention in autonomous driving due to its accessibility and robustness. However, its standalone application for depth perception is constrained by issues of sparsity and noise. Radar-camera depth estimation offers a more promising complementary solution. Despite significant progress, current approaches fail to produce satisfactory dense depth maps, due to the unsatisfactory processing of the sparse and noisy radar data. They constrain the regions of interest for radar points in rigid rectangular regions, which may introduce unexpected errors and confusions. To address these issues, we develop a structure-aware strategy for radar depth enhancement, which provides more targeted regions of interest by leveraging the structural priors of RGB images. Furthermore, we design a Multi-Scale Structure Guided Network to enhance radar features and preserve detailed structures, achieving accurate and structure-detailed dense metric depth estimation. Building on these, we propose a structure-aware radar-camera depth estimation framework, named SA-RCD. Extensive experiments demonstrate that our SA-RCD achieves state-of-the-art performance on the nuScenes dataset. Our code will be available at https://github.com/FreyZhangYeh/SA-RCD.
>
---
#### [replaced 017] GM-MoE: Low-Light Enhancement with Gated-Mechanism Mixture-of-Experts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07417v3](http://arxiv.org/pdf/2503.07417v3)**

> **作者:** Minwen Liao; Hao Bo Dong; Xinyi Wang; Kurban Ubul; Ziyang Yan; Yihua Shao
>
> **摘要:** Low-light enhancement has wide applications in autonomous driving, 3D reconstruction, remote sensing, surveillance, and so on, which can significantly improve information utilization. However, most existing methods lack generalization and are limited to specific tasks such as image recovery. To address these issues, we propose Gated-Mechanism Mixture-of-Experts (GM-MoE), the first framework to introduce a mixture-of-experts network for low-light image enhancement. GM-MoE comprises a dynamic gated weight conditioning network and three sub-expert networks, each specializing in a distinct enhancement task. Combining a self-designed gated mechanism that dynamically adjusts the weights of the sub-expert networks for different data domains. Additionally, we integrate local and global feature fusion within sub-expert networks to enhance image quality by capturing multi-scale features. Experimental results demonstrate that the GM-MoE achieves superior generalization with respect to 25 compared approaches, reaching state-of-the-art performance on PSNR on 5 benchmarks and SSIM on 4 benchmarks, respectively.
>
---
#### [replaced 018] TinyAlign: Boosting Lightweight Vision-Language Models by Mitigating Modal Alignment Bottlenecks
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12884v2](http://arxiv.org/pdf/2505.12884v2)**

> **作者:** Yuanze Hu; Zhaoxin Fan; Xinyu Wang; Gen Li; Ye Qiu; Zhichao Yang; Wenjun Wu; Kejian Wu; Yifan Sun; Xiaotie Deng; Jin Dong
>
> **摘要:** Lightweight Vision-Language Models (VLMs) are indispensable for resource-constrained applications. The prevailing approach to aligning vision and language models involves freezing both the vision encoder and the language model while training small connector modules. However, this strategy heavily depends on the intrinsic capabilities of the language model, which can be suboptimal for lightweight models with limited representational capacity. In this work, we investigate this alignment bottleneck through the lens of mutual information, demonstrating that the constrained capacity of the language model inherently limits the Effective Mutual Information (EMI) between multimodal inputs and outputs, thereby compromising alignment quality. To address this challenge, we propose TinyAlign, a novel framework inspired by Retrieval-Augmented Generation, which strategically retrieves relevant context from a memory bank to enrich multimodal inputs and enhance their alignment. Extensive empirical evaluations reveal that TinyAlign significantly reduces training loss, accelerates convergence, and enhances task performance. Remarkably, it allows models to achieve baseline-level performance with only 40\% of the fine-tuning data, highlighting exceptional data efficiency. Our work thus offers a practical pathway for developing more capable lightweight VLMs while introducing a fresh theoretical lens to better understand and address alignment bottlenecks in constrained multimodal systems.
>
---
#### [replaced 019] Pretrained Reversible Generation as Unsupervised Visual Representation Learning
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.01787v4](http://arxiv.org/pdf/2412.01787v4)**

> **作者:** Rongkun Xue; Jinouwen Zhang; Yazhe Niu; Dazhong Shen; Bingqi Ma; Yu Liu; Jing Yang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Recent generative models based on score matching and flow matching have significantly advanced generation tasks, but their potential in discriminative tasks remains underexplored. Previous approaches, such as generative classifiers, have not fully leveraged the capabilities of these models for discriminative tasks due to their intricate designs. We propose Pretrained Reversible Generation (PRG), which extracts unsupervised representations by reversing the generative process of a pretrained continuous generation model. PRG effectively reuses unsupervised generative models, leveraging their high capacity to serve as robust and generalizable feature extractors for downstream tasks. This framework enables the flexible selection of feature hierarchies tailored to specific downstream tasks. Our method consistently outperforms prior approaches across multiple benchmarks, achieving state-of-the-art performance among generative model based methods, including 78% top-1 accuracy on ImageNet at a resolution of 64*64. Extensive ablation studies, including out-of-distribution evaluations, further validate the effectiveness of our approach.PRG is available at https://github.com/opendilab/PRG.
>
---
#### [replaced 020] GUNNEL: Guided Mixup Augmentation and Multi-Model Fusion for Aquatic Animal Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2112.06193v4](http://arxiv.org/pdf/2112.06193v4)**

> **作者:** Minh-Quan Le; Trung-Nghia Le; Tam V. Nguyen; Isao Echizen; Minh-Triet Tran
>
> **备注:** Accepted to Neural Computing & Applications
>
> **摘要:** Recent years have witnessed great advances in object segmentation research. In addition to generic objects, aquatic animals have attracted research attention. Deep learning-based methods are widely used for aquatic animal segmentation and have achieved promising performance. However, there is a lack of challenging datasets for benchmarking. In this work, we build a new dataset dubbed "Aquatic Animal Species." We also devise a novel GUided mixup augmeNtatioN and multi-modEl fusion for aquatic animaL segmentation (GUNNEL) that leverages the advantages of multiple segmentation models to segment aquatic animals effectively and improves the training performance by synthesizing hard samples. Extensive experiments demonstrated the superiority of our proposed framework over existing state-of-the-art instance segmentation methods. The code is available at https://github.com/lmquan2000/mask-mixup. The dataset is available at https://doi.org/10.5281/zenodo.8208877.
>
---
#### [replaced 021] Self-supervised Learning of Hybrid Part-aware 3D Representation of 2D Gaussians and Superquadrics
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.10789v3](http://arxiv.org/pdf/2408.10789v3)**

> **作者:** Zhirui Gao; Renjiao Yi; Yuhang Huang; Wei Chen; Chenyang Zhu; Kai Xu
>
> **备注:** Accepted by ICCV 2025 Code: https://github.com/zhirui-gao/PartGS
>
> **摘要:** Low-level 3D representations, such as point clouds, meshes, NeRFs and 3D Gaussians, are commonly used for modeling 3D objects and scenes. However, cognitive studies indicate that human perception operates at higher levels and interprets 3D environments by decomposing them into meaningful structural parts, rather than low-level elements like points or voxels. Structured geometric decomposition enhances scene interpretability and facilitates downstream tasks requiring component-level manipulation. In this work, we introduce PartGS, a self-supervised part-aware reconstruction framework that integrates 2D Gaussians and superquadrics to parse objects and scenes into an interpretable decomposition, leveraging multi-view image inputs to uncover 3D structural information. Our method jointly optimizes superquadric meshes and Gaussians by coupling their parameters within a hybrid representation. On one hand, superquadrics enable the representation of a wide range of shape primitives, facilitating flexible and meaningful decompositions. On the other hand, 2D Gaussians capture detailed texture and geometric details, ensuring high-fidelity appearance and geometry reconstruction. Operating in a self-supervised manner, our approach demonstrates superior performance compared to state-of-the-art methods across extensive experiments on the DTU, ShapeNet, and real-world datasets.
>
---
#### [replaced 022] ForgeLens: Data-Efficient Forgery Focus for Generalizable Forgery Image Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.13697v2](http://arxiv.org/pdf/2408.13697v2)**

> **作者:** Yingjian Chen; Lei Zhang; Yakun Niu
>
> **摘要:** The rise of generative models has raised concerns about image authenticity online, highlighting the urgent need for a detector that is (1) highly generalizable, capable of handling unseen forgery techniques, and (2) data-efficient, achieving optimal performance with minimal training data, enabling it to counter newly emerging forgery techniques effectively. To achieve this, we propose ForgeLens, a data-efficient, feature-guided framework that incorporates two lightweight designs to enable a frozen network to focus on forgery-specific features. First, we introduce the Weight-Shared Guidance Module (WSGM), which guides the extraction of forgery-specific features during training. Second, a forgery-aware feature integrator, FAFormer, is used to effectively integrate forgery information across multi-stage features. ForgeLens addresses a key limitation of previous frozen network-based methods, where general-purpose features extracted from large datasets often contain excessive forgery-irrelevant information. As a result, it achieves strong generalization and reaches optimal performance with minimal training data. Experimental results on 19 generative models, including both GANs and diffusion models, demonstrate improvements of 13.61% in Avg.Acc and 8.69% in Avg.AP over the base model. Notably, ForgeLens outperforms existing forgery detection methods, achieving state-of-the-art performance with just 1% of the training data. Our code is available at https://github.com/Yingjian-Chen/ForgeLens.
>
---
#### [replaced 023] Ground-R1: Incentivizing Grounded Visual Reasoning via Reinforcement Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20272v2](http://arxiv.org/pdf/2505.20272v2)**

> **作者:** Meng Cao; Haoze Zhao; Can Zhang; Xiaojun Chang; Ian Reid; Xiaodan Liang
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated impressive general capabilities across a wide range of multi-modal tasks. However, the reasoning processes of LVLMs often suffer from unreliable outputs and limited interpretability. To address this, grounded visual reasoning has emerged as a promising paradigm that enforces responses anchored on salient visual evidence regions. However, existing approaches typically rely on costly supervision such as bounding box annotations, chain-of-thought rationale or external tool calls, limiting their scalability. In this work, we propose Ground-R1, a reinforcement learning framework that enables grounded visual reasoning without requiring explicit evidence or rationale annotations. Ground-R1 consists of a grounding phase that generates evidence region rollouts based on format constraints, and an answering phase that produces responses guided by both answer correctness and format adherence rewards. Extensive experiments across multiple visual reasoning benchmarks manifest that Ground-R1 achieves superior performance and exhibits emergent cognitive behaviors such as uncertainty awareness, spatial perception, and iterative refinement, offering a scalable and interpretable alternative to existing approaches.
>
---
#### [replaced 024] Finer Disentanglement of Aleatoric Uncertainty Can Accelerate Chemical Histopathology Imaging
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.20532v2](http://arxiv.org/pdf/2502.20532v2)**

> **作者:** Ji-Hun Oh; Kianoush Falahkheirkhah; Rohit Bhargava
>
> **摘要:** Label-free chemical imaging holds significant promise for improving digital pathology workflows, but data acquisition speed remains a limiting factor. To address this gap, we propose an adaptive strategy-initially scan the low information (LI) content of the entire tissue quickly, identify regions with high aleatoric uncertainty (AU), and selectively re-image them at better quality to capture higher information (HI) details. The primary challenge lies in distinguishing between high-AU regions mitigable through HI imaging and those that are not. However, since existing uncertainty frameworks cannot separate such AU subcategories, we propose a fine-grained disentanglement method based on post-hoc latent space analysis to unmix resolvable from irresolvable high-AU regions. We apply our approach to streamline infrared spectroscopic imaging of breast tissues, achieving superior downstream segmentation performance. This marks the first study focused on fine-grained AU disentanglement within dynamic image spaces (LI-to-HI), with novel application to streamline histopathology.
>
---
#### [replaced 025] BFA: Best-Feature-Aware Fusion for Multi-View Fine-grained Manipulation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.11161v3](http://arxiv.org/pdf/2502.11161v3)**

> **作者:** Zihan Lan; Weixin Mao; Haosheng Li; Le Wang; Tiancai Wang; Haoqiang Fan; Osamu Yoshie
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** In real-world scenarios, multi-view cameras are typically employed for fine-grained manipulation tasks. Existing approaches (e.g., ACT) tend to treat multi-view features equally and directly concatenate them for policy learning. However, it will introduce redundant visual information and bring higher computational costs, leading to ineffective manipulation. For a fine-grained manipulation task, it tends to involve multiple stages while the most contributed view for different stages is varied over time. In this paper, we propose a plug-and-play best-feature-aware (BFA) fusion strategy for multi-view manipulation tasks, which is adaptable to various policies. Built upon the visual backbone of the policy network, we design a lightweight network to predict the importance score of each view. Based on the predicted importance scores, the reweighted multi-view features are subsequently fused and input into the end-to-end policy network, enabling seamless integration. Notably, our method demonstrates outstanding performance in fine-grained manipulations. Experimental results show that our approach outperforms multiple baselines by 22-46% success rate on different tasks. Our work provides new insights and inspiration for tackling key challenges in fine-grained manipulations.
>
---
#### [replaced 026] Privacy-Preserving Video Anomaly Detection: A Survey
- **分类: cs.CV; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.14565v2](http://arxiv.org/pdf/2411.14565v2)**

> **作者:** Yang Liu; Siao Liu; Xiaoguang Zhu; Jielin Li; Hao Yang; Liangyu Teng; Juncen Guo; Yan Wang; Dingkang Yang; Jing Liu
>
> **备注:** 22 pages, 9 figures, 7 tables
>
> **摘要:** Video Anomaly Detection (VAD) aims to automatically analyze spatiotemporal patterns in surveillance videos collected from open spaces to detect anomalous events that may cause harm, such as fighting, stealing, and car accidents. However, vision-based surveillance systems such as closed-circuit television often capture personally identifiable information. The lack of transparency and interpretability in video transmission and usage raises public concerns about privacy and ethics, limiting the real-world application of VAD. Recently, researchers have focused on privacy concerns in VAD by conducting systematic studies from various perspectives including data, features, and systems, making Privacy-Preserving Video Anomaly Detection (P2VAD) a hotspot in the AI community. However, current research in P2VAD is fragmented, and prior reviews have mostly focused on methods using RGB sequences, overlooking privacy leakage and appearance bias considerations. To address this gap, this article is the first to systematically reviews the progress of P2VAD, defining its scope and providing an intuitive taxonomy. We outline the basic assumptions, learning frameworks, and optimization objectives of various approaches, analyzing their strengths, weaknesses, and potential correlations. Additionally, we provide open access to research resources such as benchmark datasets and available code. Finally, we discuss key challenges and future opportunities from the perspectives of AI development and P2VAD deployment, aiming to guide future work in the field.
>
---
#### [replaced 027] Multibiometrics Using a Single Face Image
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.20003v2](http://arxiv.org/pdf/2409.20003v2)**

> **作者:** Koichi Ito; Taito Tonosaki; Takafumi Aoki; Tetsushi Ohki; Masakatsu Nishigaki
>
> **备注:** APSIPA ASC 2024
>
> **摘要:** Multibiometrics, which uses multiple biometric traits to improve recognition performance instead of using only one biometric trait to authenticate individuals, has been investigated. Previous studies have combined individually acquired biometric traits or have not fully considered the convenience of the system. Focusing on a single face image, we propose a novel multibiometric method that combines five biometric traits, i.e., face, iris, periocular, nose, eyebrow, that can be extracted from a single face image. The proposed method does not sacrifice the convenience of biometrics since only a single face image is used as input. Through a variety of experiments using the CASIA Iris Distance database, we demonstrate the effectiveness of the proposed multibiometrics method.
>
---
#### [replaced 028] ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.14607v2](http://arxiv.org/pdf/2501.14607v2)**

> **作者:** Tianming Liang; Kun-Yu Lin; Chaolei Tan; Jianguo Zhang; Wei-Shi Zheng; Jian-Fang Hu
>
> **备注:** Accepted to ICCV 2025. Project page: \url{https://isee-laboratory.github.io/ReferDINO}
>
> **摘要:** Referring video object segmentation (RVOS) aims to segment target objects throughout a video based on a text description. This is challenging as it involves deep vision-language understanding, pixel-level dense prediction and spatiotemporal reasoning. Despite notable progress in recent years, existing methods still exhibit a noticeable gap when considering all these aspects. In this work, we propose \textbf{ReferDINO}, a strong RVOS model that inherits region-level vision-language alignment from foundational visual grounding models, and is further endowed with pixel-level dense perception and cross-modal spatiotemporal reasoning. In detail, ReferDINO integrates two key components: 1) a grounding-guided deformable mask decoder that utilizes location prediction to progressively guide mask prediction through differentiable deformation mechanisms; 2) an object-consistent temporal enhancer that injects pretrained time-varying text features into inter-frame interaction to capture object-aware dynamic changes. Moreover, a confidence-aware query pruning strategy is designed to accelerate object decoding without compromising model performance. Extensive experimental results on five benchmarks demonstrate that our ReferDINO significantly outperforms previous methods (e.g., +3.9% (\mathcal{J}&\mathcal{F}) on Ref-YouTube-VOS) with real-time inference speed (51 FPS).
>
---
#### [replaced 029] PoI: A Filter to Extract Pixel of Interest from Novel View Synthesis for Scene Coordinate Regression
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.04843v3](http://arxiv.org/pdf/2502.04843v3)**

> **作者:** Feifei Li; Qi Song; Chi Zhang; Hui Shuai; Rui Huang
>
> **摘要:** Novel View Synthesis (NVS) techniques, notably Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), can augment camera pose estimation by extending and diversifying training data. However, images generated by these methods are often plagued by spatial artifacts such as blurring and ghosting, undermining their reliability as training data for camera pose estimation. This limitation is particularly critical for Scene Coordinate Regression (SCR) methods, which aim at pixel-level 3D coordinate estimation, because rendering artifacts directly lead to estimation inaccuracies. To address this challenge, we propose a dual-criteria filtering mechanism that dynamically identifies and discards suboptimal pixels during training. The dual-criteria filter evaluates two concurrent metrics: (1) real-time SCR reprojection error, and (2) gradient threshold, across the coordinate regression domain. In addition, for visual localization problems in sparse-input scenarios, it becomes even more necessary to use NVS-generated data to assist localization. We design a coarse-to-fine Points of Interest (PoI) variant using sparse-input NVS to solve this problem. Experiments across indoor and outdoor benchmarks confirm our method's efficacy, achieving state-of-the-art localization accuracy while maintaining computational efficiency.
>
---
#### [replaced 030] OpenPath: Open-Set Active Learning for Pathology Image Classification via Pre-trained Vision-Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.15318v3](http://arxiv.org/pdf/2506.15318v3)**

> **作者:** Lanfeng Zhong; Xin Liao; Shichuan Zhang; Shaoting Zhang; Guotai Wang
>
> **备注:** MICCAI 2025 early accept
>
> **摘要:** Pathology image classification plays a crucial role in accurate medical diagnosis and treatment planning. Training high-performance models for this task typically requires large-scale annotated datasets, which are both expensive and time-consuming to acquire. Active Learning (AL) offers a solution by iteratively selecting the most informative samples for annotation, thereby reducing the labeling effort. However, most AL methods are designed under the assumption of a closed-set scenario, where all the unannotated images belong to target classes. In real-world clinical environments, the unlabeled pool often contains a substantial amount of Out-Of-Distribution (OOD) data, leading to low efficiency of annotation in traditional AL methods. Furthermore, most existing AL methods start with random selection in the first query round, leading to a significant waste of labeling costs in open-set scenarios. To address these challenges, we propose OpenPath, a novel open-set active learning approach for pathological image classification leveraging a pre-trained Vision-Language Model (VLM). In the first query, we propose task-specific prompts that combine target and relevant non-target class prompts to effectively select In-Distribution (ID) and informative samples from the unlabeled pool. In subsequent queries, Diverse Informative ID Sampling (DIS) that includes Prototype-based ID candidate Selection (PIS) and Entropy-Guided Stochastic Sampling (EGSS) is proposed to ensure both purity and informativeness in a query, avoiding the selection of OOD samples. Experiments on two public pathology image datasets show that OpenPath significantly enhances the model's performance due to its high purity of selected samples, and outperforms several state-of-the-art open-set AL methods. The code is available at \href{https://github.com/HiLab-git/OpenPath}{https://github.com/HiLab-git/OpenPath}..
>
---
#### [replaced 031] Cluster and Predict Latent Patches for Improved Masked Image Modeling
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.08769v3](http://arxiv.org/pdf/2502.08769v3)**

> **作者:** Timothée Darcet; Federico Baldassarre; Maxime Oquab; Julien Mairal; Piotr Bojanowski
>
> **备注:** 26 pages, 14 figures, accepted in TMLR 2025
>
> **摘要:** Masked Image Modeling (MIM) offers a promising approach to self-supervised representation learning, however existing MIM models still lag behind the state-of-the-art. In this paper, we systematically analyze target representations, loss functions, and architectures, to introduce CAPI - a novel pure-MIM framework that relies on the prediction of latent clusterings. Our approach leverages a clustering-based loss, which is stable to train, and exhibits promising scaling properties. Our ViT-L backbone, CAPI, achieves 83.8% accuracy on ImageNet and 32.1% mIoU on ADE20K with simple linear probes, substantially outperforming previous MIM methods and approaching the performance of the current state-of-the-art, DINOv2. We release all our code and models.
>
---
#### [replaced 032] Visual Position Prompt for MLLM based Visual Grounding
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.15426v3](http://arxiv.org/pdf/2503.15426v3)**

> **作者:** Wei Tang; Yanpeng Sun; Qinying Gu; Zechao Li
>
> **摘要:** Although Multimodal Large Language Models (MLLMs) excel at various image-related tasks, they encounter challenges in precisely aligning coordinates with spatial information within images, particularly in position-aware tasks such as visual grounding. This limitation arises from two key factors. First, MLLMs lack explicit spatial references, making it difficult to associate textual descriptions with precise image locations. Second, their feature extraction processes prioritize global context over fine-grained spatial details, leading to weak localization capability. To address these issues, we introduce VPP-LLaVA, an MLLM enhanced with Visual Position Prompt (VPP) to improve its grounding capability. VPP-LLaVA integrates two complementary mechanisms: the global VPP overlays a learnable, axis-like tensor onto the input image to provide structured spatial cues, while the local VPP incorporates position-aware queries to support fine-grained localization.To effectively train our model with spatial guidance, we further introduce VPP-SFT, a curated dataset of 0.6M high-quality visual grounding samples. Designed in a compact format, it enables efficient training and is significantly smaller than datasets used by other MLLMs (e.g., ~21M samples in MiniGPT-v2), yet still provides a strong performance boost. The resulting model, VPP-LLaVA, not only achieves state-of-the-art results on standard visual grounding benchmarks but also demonstrates strong zero-shot generalization to challenging unseen datasets. Code and dataset will be released upon acceptance at https://github.com/WayneTomas/VPP-LLaVA.
>
---
#### [replaced 033] Iterative approach to reconstructing neural disparity fields from light-field data
- **分类: eess.IV; cs.CV; 68U10; I.4.10; I.4.5**

- **链接: [http://arxiv.org/pdf/2407.15380v2](http://arxiv.org/pdf/2407.15380v2)**

> **作者:** Ligen Shi; Chang Liu; Xing Zhao; Jun Qiu
>
> **备注:** 11 pages, 9 figures
>
> **摘要:** This study proposes a neural disparity field (NDF) that establishes an implicit, continuous representation of scene disparity based on a neural field and an iterative approach to address the inverse problem of NDF reconstruction from light-field data. NDF enables seamless and precise characterization of disparity variations in three-dimensional scenes and can discretize disparity at any arbitrary resolution, overcoming the limitations of traditional disparity maps that are prone to sampling errors and interpolation inaccuracies. The proposed NDF network architecture utilizes hash encoding combined with multilayer perceptrons to capture detailed disparities in texture levels, thereby enhancing its ability to represent the geometric information of complex scenes. By leveraging the spatial-angular consistency inherent in light-field data, a differentiable forward model to generate a central view image from the light-field data is developed. Based on the forward model, an optimization scheme for the inverse problem of NDF reconstruction using differentiable propagation operators is established. Furthermore, an iterative solution method is adopted to reconstruct the NDF in the optimization scheme, which does not require training datasets and applies to light-field data captured by various acquisition methods. Experimental results demonstrate that high-quality NDF can be reconstructed from light-field data using the proposed method. High-resolution disparity can be effectively recovered by NDF, demonstrating its capability for the implicit, continuous representation of scene disparities.
>
---
#### [replaced 034] Disentangled Diffusion-Based 3D Human Pose Estimation with Hierarchical Spatial and Temporal Denoiser
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.04444v2](http://arxiv.org/pdf/2403.04444v2)**

> **作者:** Qingyuan Cai; Xuecai Hu; Saihui Hou; Li Yao; Yongzhen Huang
>
> **备注:** Accepted by AAAI24
>
> **摘要:** Recently, diffusion-based methods for monocular 3D human pose estimation have achieved state-of-the-art (SOTA) performance by directly regressing the 3D joint coordinates from the 2D pose sequence. Although some methods decompose the task into bone length and bone direction prediction based on the human anatomical skeleton to explicitly incorporate more human body prior constraints, the performance of these methods is significantly lower than that of the SOTA diffusion-based methods. This can be attributed to the tree structure of the human skeleton. Direct application of the disentangled method could amplify the accumulation of hierarchical errors, propagating through each hierarchy. Meanwhile, the hierarchical information has not been fully explored by the previous methods. To address these problems, a Disentangled Diffusion-based 3D Human Pose Estimation method with Hierarchical Spatial and Temporal Denoiser is proposed, termed DDHPose. In our approach: (1) We disentangle the 3D pose and diffuse the bone length and bone direction during the forward process of the diffusion model to effectively model the human pose prior. A disentanglement loss is proposed to supervise diffusion model learning. (2) For the reverse process, we propose Hierarchical Spatial and Temporal Denoiser (HSTDenoiser) to improve the hierarchical modeling of each joint. Our HSTDenoiser comprises two components: the Hierarchical-Related Spatial Transformer (HRST) and the Hierarchical-Related Temporal Transformer (HRTT). HRST exploits joint spatial information and the influence of the parent joint on each joint for spatial modeling, while HRTT utilizes information from both the joint and its hierarchical adjacent joints to explore the hierarchical temporal correlations among joints. Code and models are available at https://github.com/Andyen512/DDHPose
>
---
#### [replaced 035] D$^2$ST-Adapter: Disentangled-and-Deformable Spatio-Temporal Adapter for Few-shot Action Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2312.01431v4](http://arxiv.org/pdf/2312.01431v4)**

> **作者:** Wenjie Pei; Qizhong Tan; Guangming Lu; Jiandong Tian; Jun Yu
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Adapting pre-trained image models to video modality has proven to be an effective strategy for robust few-shot action recognition. In this work, we explore the potential of adapter tuning in image-to-video model adaptation and propose a novel video adapter tuning framework, called Disentangled-and-Deformable Spatio-Temporal Adapter (D$^2$ST-Adapter). It features a lightweight design, low adaptation overhead and powerful spatio-temporal feature adaptation capabilities. D$^2$ST-Adapter is structured with an internal dual-pathway architecture that enables built-in disentangled encoding of spatial and temporal features within the adapter, seamlessly integrating into the single-stream feature learning framework of pre-trained image models. In particular, we develop an efficient yet effective implementation of the D$^2$ST-Adapter, incorporating the specially devised anisotropic Deformable Spatio-Temporal Attention as its pivotal operation. This mechanism can be individually tailored for two pathways with anisotropic sampling densities along the spatial and temporal domains in 3D spatio-temporal space, enabling disentangled encoding of spatial and temporal features while maintaining a lightweight design. Extensive experiments by instantiating our method on both pre-trained ResNet and ViT demonstrate the superiority of our method over state-of-the-art methods. Our method is particularly well-suited to challenging scenarios where temporal dynamics are critical for action recognition. Code is available at https://github.com/qizhongtan/D2ST-Adapter.
>
---
#### [replaced 036] MSF: Efficient Diffusion Model Via Multi-Scale Latent Factorize
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.13349v2](http://arxiv.org/pdf/2501.13349v2)**

> **作者:** Haohang Xu; Longyu Chen; Yichen Zhang; Shuangrui Ding; Zhipeng Zhang
>
> **摘要:** While diffusion-based generative models have made significant strides in visual content creation, conventional approaches face computational challenges, especially for high-resolution images, as they denoise the entire image from noisy inputs. This contrasts with signal processing techniques, such as Fourier and wavelet analyses, which often employ hierarchical decompositions. Inspired by such principles, particularly the idea of signal separation, we introduce a diffusion framework leveraging multi-scale latent factorization. Our framework uniquely decomposes the denoising target, typically latent features from a pretrained Variational Autoencoder, into a low-frequency base signal capturing core structural information and a high-frequency residual signal that contributes finer, high-frequency details like textures. This decomposition into base and residual components directly informs our two-stage image generation process, which first produces the low-resolution base, followed by the generation of the high-resolution residual. Our proposed architecture facilitates reduced sampling steps during the residual learning stage, owing to the inherent ease of modeling residual information, which confers advantages over conventional full-resolution generation techniques. This specific approach of decomposing the signal into a base and a residual, conceptually akin to how wavelet analysis can separate different frequency bands, yields a more streamlined and intuitive design distinct from generic hierarchical models. Our method, \name\ (Multi-Scale Factorization), demonstrates its effectiveness by achieving FID scores of 2.08 ($256\times256$) and 2.47 ($512\times512$) on class-conditional ImageNet benchmarks, outperforming the DiT baseline (2.27 and 3.04 respectively) while also delivering a $4\times$ speed-up with the same number of sampling steps.
>
---
#### [replaced 037] HumanGif: Single-View Human Diffusion with Generative Prior
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.12080v3](http://arxiv.org/pdf/2502.12080v3)**

> **作者:** Shoukang Hu; Takuya Narihira; Kazumi Fukuda; Ryosuke Sawata; Takashi Shibuya; Yuki Mitsufuji
>
> **备注:** Project page: https://skhu101.github.io/HumanGif/
>
> **摘要:** Previous 3D human creation methods have made significant progress in synthesizing view-consistent and temporally aligned results from sparse-view images or monocular videos. However, it remains challenging to produce perpetually realistic, view-consistent, and temporally coherent human avatars from a single image, as limited information is available in the single-view input setting. Motivated by the success of 2D character animation, we propose HumanGif, a single-view human diffusion model with generative prior. Specifically, we formulate the single-view-based 3D human novel view and pose synthesis as a single-view-conditioned human diffusion process, utilizing generative priors from foundational diffusion models to complement the missing information. To ensure fine-grained and consistent novel view and pose synthesis, we introduce a Human NeRF module in HumanGif to learn spatially aligned features from the input image, implicitly capturing the relative camera and human pose transformation. Furthermore, we introduce an image-level loss during optimization to bridge the gap between latent and image spaces in diffusion models. Extensive experiments on RenderPeople, DNA-Rendering, THuman 2.1, and TikTok datasets demonstrate that HumanGif achieves the best perceptual performance, with better generalizability for novel view and pose synthesis.
>
---
#### [replaced 038] A Narrative Review on Large AI Models in Lung Cancer Screening, Diagnosis, and Treatment Planning
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.07236v2](http://arxiv.org/pdf/2506.07236v2)**

> **作者:** Jiachen Zhong; Yiting Wang; Di Zhu; Ziwei Wang
>
> **备注:** This request is based on the fact that one of the co-authors is a PhD student whose advisor has informed her that she was not authorized to publicly release this work without his prior approval. Unfortunately, this approval was not obtained, and as such, the submission was made without proper institutional and supervisory consent
>
> **摘要:** Lung cancer remains one of the most prevalent and fatal diseases worldwide, demanding accurate and timely diagnosis and treatment. Recent advancements in large AI models have significantly enhanced medical image understanding and clinical decision-making. This review systematically surveys the state-of-the-art in applying large AI models to lung cancer screening, diagnosis, prognosis, and treatment. We categorize existing models into modality-specific encoders, encoder-decoder frameworks, and joint encoder architectures, highlighting key examples such as CLIP, BLIP, Flamingo, BioViL-T, and GLoRIA. We further examine their performance in multimodal learning tasks using benchmark datasets like LIDC-IDRI, NLST, and MIMIC-CXR. Applications span pulmonary nodule detection, gene mutation prediction, multi-omics integration, and personalized treatment planning, with emerging evidence of clinical deployment and validation. Finally, we discuss current limitations in generalizability, interpretability, and regulatory compliance, proposing future directions for building scalable, explainable, and clinically integrated AI systems. Our review underscores the transformative potential of large AI models to personalize and optimize lung cancer care.
>
---
#### [replaced 039] Benchmarking Spiking Neural Network Learning Methods with Varying Locality
- **分类: cs.NE; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.01782v2](http://arxiv.org/pdf/2402.01782v2)**

> **作者:** Jiaqi Lin; Sen Lu; Malyaban Bal; Abhronil Sengupta
>
> **摘要:** Spiking Neural Networks (SNNs), providing more realistic neuronal dynamics, have been shown to achieve performance comparable to Artificial Neural Networks (ANNs) in several machine learning tasks. Information is processed as spikes within SNNs in an event-based mechanism that significantly reduces energy consumption. However, training SNNs is challenging due to the non-differentiable nature of the spiking mechanism. Traditional approaches, such as Backpropagation Through Time (BPTT), have shown effectiveness but come with additional computational and memory costs and are biologically implausible. In contrast, recent works propose alternative learning methods with varying degrees of locality, demonstrating success in classification tasks. In this work, we show that these methods share similarities during the training process, while they present a trade-off between biological plausibility and performance. Further, given the implicitly recurrent nature of SNNs, this research investigates the influence of the addition of explicit recurrence to SNNs. We experimentally prove that the addition of explicit recurrent weights enhances the robustness of SNNs. We also investigate the performance of local learning methods under gradient and non-gradient-based adversarial attacks.
>
---
#### [replaced 040] Efficiently Serving Large Multimodal Models Using EPD Disaggregation
- **分类: cs.DC; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.05460v4](http://arxiv.org/pdf/2501.05460v4)**

> **作者:** Gursimran Singh; Xinglu Wang; Yifan Hu; Timothy Yu; Linzi Xing; Wei Jiang; Zhefeng Wang; Xiaolong Bai; Yi Li; Ying Xiong; Yong Zhang; Zhenan Fan
>
> **备注:** 17 pages, 12 figures, 9 tables
>
> **摘要:** Large Multimodal Models (LMMs) extend Large Language Models (LLMs) by handling diverse inputs such as images, audio, and video, but at the cost of adding a multimodal encoding stage that increases both computational and memory overhead. This step negatively affects key Service Level Objectives (SLOs), such as time to first token (TTFT) and time per output token (TPOT). We introduce Encode-Prefill-Decode (EPD) Disaggregation, a novel framework that separates the encoding, prefill, and decode stages onto dedicated resources. Unlike current systems, which bundle encoding and prefill together, our approach decouples these steps, unlocking new opportunities and optimizations. These include a mechanism to cache multimedia tokens for efficient transfer, a novel way to parallelize the encoding load within a request, a module for optimal resource allocation for disaggregated serving, and a novel role-switching method to handle changing workload characteristics. Experimental evaluations with popular LMMs show substantial gains in memory efficiency (up to 15x lower peak memory utilization), batch sizes (up to 22x larger), 10x more images per request, and 2.2x larger KV caches. Furthermore, it leads to significant improvements in SLO attainment (up to 90-100% improvement) and TTFT (up to 71% reduction), compared to systems that do not disaggregate. The code is available at https://github.com/vbdi/epdserve.
>
---
#### [replaced 041] Decoding Federated Learning: The FedNAM+ Conformal Revolution
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.17872v2](http://arxiv.org/pdf/2506.17872v2)**

> **作者:** Sree Bhargavi Balija; Amitash Nanda; Debashis Sahoo
>
> **摘要:** Federated learning has significantly advanced distributed training of machine learning models across decentralized data sources. However, existing frameworks often lack comprehensive solutions that combine uncertainty quantification, interpretability, and robustness. To address this, we propose FedNAM+, a federated learning framework that integrates Neural Additive Models (NAMs) with a novel conformal prediction method to enable interpretable and reliable uncertainty estimation. Our method introduces a dynamic level adjustment technique that utilizes gradient-based sensitivity maps to identify key input features influencing predictions. This facilitates both interpretability and pixel-wise uncertainty estimates. Unlike traditional interpretability methods such as LIME and SHAP, which do not provide confidence intervals, FedNAM+ offers visual insights into prediction reliability. We validate our approach through experiments on CT scan, MNIST, and CIFAR datasets, demonstrating high prediction accuracy with minimal loss (e.g., only 0.1% on MNIST), along with transparent uncertainty measures. Visual analysis highlights variable uncertainty intervals, revealing low-confidence regions where model performance can be improved with additional data. Compared to Monte Carlo Dropout, FedNAM+ delivers efficient and global uncertainty estimates with reduced computational overhead, making it particularly suitable for federated learning scenarios. Overall, FedNAM+ provides a robust, interpretable, and computationally efficient framework that enhances trust and transparency in decentralized predictive modeling.
>
---
#### [replaced 042] Fine-Tuning Next-Scale Visual Autoregressive Models with Group Relative Policy Optimization
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.23331v2](http://arxiv.org/pdf/2505.23331v2)**

> **作者:** Matteo Gallici; Haitz Sáez de Ocáriz Borde
>
> **摘要:** Fine-tuning pre-trained generative models with Reinforcement Learning (RL) has emerged as an effective approach for aligning outputs more closely with nuanced human preferences. In this paper, we investigate the application of Group Relative Policy Optimization (GRPO) to fine-tune next-scale visual autoregressive (VAR) models. Our empirical results demonstrate that this approach enables alignment to intricate reward signals derived from aesthetic predictors and CLIP embeddings, significantly enhancing image quality and enabling precise control over the generation style. Interestingly, by leveraging CLIP, our method can help VAR models generalize beyond their initial ImageNet distribution: through RL-driven exploration, these models can generate images aligned with prompts referencing image styles that were absent during pre-training. In summary, we show that RL-based fine-tuning is both efficient and effective for VAR models, benefiting particularly from their fast inference speeds, which are advantageous for online sampling, an aspect that poses significant challenges for diffusion-based alternatives.
>
---
#### [replaced 043] Generalizing vision-language models to novel domains: A comprehensive survey
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.18504v2](http://arxiv.org/pdf/2506.18504v2)**

> **作者:** Xinyao Li; Jingjing Li; Fengling Li; Lei Zhu; Yang Yang; Heng Tao Shen
>
> **摘要:** Recently, vision-language pretraining has emerged as a transformative technique that integrates the strengths of both visual and textual modalities, resulting in powerful vision-language models (VLMs). Leveraging web-scale pretraining data, these models exhibit strong zero-shot capabilities. However, their performance often deteriorates when confronted with domain-specific or specialized generalization tasks. To address this, a growing body of research focuses on transferring or generalizing the rich knowledge embedded in VLMs to various downstream applications. This survey aims to comprehensively summarize the generalization settings, methodologies, benchmarking and results in VLM literatures. Delving into the typical VLM structures, current literatures are categorized into prompt-based, parameter-based and feature-based methods according to the transferred modules. The differences and characteristics in each category are furthered summarized and discussed by revisiting the typical transfer learning (TL) settings, providing novel interpretations for TL in the era of VLMs. Popular benchmarks for VLM generalization are further introduced with thorough performance comparisons among the reviewed methods. Following the advances in large-scale generalizable pretraining, this survey also discusses the relations and differences between VLMs and up-to-date multimodal large language models (MLLM), e.g., DeepSeek-VL. By systematically reviewing the surging literatures in vision-language research from a novel and practical generalization prospective, this survey contributes to a clear landscape of current and future multimodal researches.
>
---
#### [replaced 044] Object detection in adverse weather conditions for autonomous vehicles using Instruct Pix2Pix
- **分类: cs.CV; cs.AI; I.2.6; I.2.10; I.4.8; I.5.1**

- **链接: [http://arxiv.org/pdf/2505.08228v2](http://arxiv.org/pdf/2505.08228v2)**

> **作者:** Unai Gurbindo; Axel Brando; Jaume Abella; Caroline König
>
> **备注:** 8 pages, 5 figures. Accepted at the International Joint Conference on Neural Networks (IJCNN) 2025 (to appear)
>
> **摘要:** Enhancing the robustness of object detection systems under adverse weather conditions is crucial for the advancement of autonomous driving technology. This study presents a novel approach leveraging the diffusion model Instruct Pix2Pix to develop prompting methodologies that generate realistic datasets with weather-based augmentations aiming to mitigate the impact of adverse weather on the perception capabilities of state-of-the-art object detection models, including Faster R-CNN and YOLOv10. Experiments were conducted in two environments, in the CARLA simulator where an initial evaluation of the proposed data augmentation was provided, and then on the real-world image data sets BDD100K and ACDC demonstrating the effectiveness of the approach in real environments. The key contributions of this work are twofold: (1) identifying and quantifying the performance gap in object detection models under challenging weather conditions, and (2) demonstrating how tailored data augmentation strategies can significantly enhance the robustness of these models. This research establishes a solid foundation for improving the reliability of perception systems in demanding environmental scenarios, and provides a pathway for future advancements in autonomous driving.
>
---
#### [replaced 045] HyperPath: Knowledge-Guided Hyperbolic Semantic Hierarchy Modeling for WSI Analysis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.16398v3](http://arxiv.org/pdf/2506.16398v3)**

> **作者:** Peixiang Huang; Yanyan Huang; Weiqin Zhao; Junjun He; Lequan Yu
>
> **摘要:** Pathology is essential for cancer diagnosis, with multiple instance learning (MIL) widely used for whole slide image (WSI) analysis. WSIs exhibit a natural hierarchy -- patches, regions, and slides -- with distinct semantic associations. While some methods attempt to leverage this hierarchy for improved representation, they predominantly rely on Euclidean embeddings, which struggle to fully capture semantic hierarchies. To address this limitation, we propose HyperPath, a novel method that integrates knowledge from textual descriptions to guide the modeling of semantic hierarchies of WSIs in hyperbolic space, thereby enhancing WSI classification. Our approach adapts both visual and textual features extracted by pathology vision-language foundation models to the hyperbolic space. We design an Angular Modality Alignment Loss to ensure robust cross-modal alignment, while a Semantic Hierarchy Consistency Loss further refines feature hierarchies through entailment and contradiction relationships and thus enhance semantic coherence. The classification is performed with geodesic distance, which measures the similarity between entities in the hyperbolic semantic hierarchy. This eliminates the need for linear classifiers and enables a geometry-aware approach to WSI analysis. Extensive experiments show that our method achieves superior performance across tasks compared to existing methods, highlighting the potential of hyperbolic embeddings for WSI analysis.
>
---
#### [replaced 046] Boosting Domain Generalized and Adaptive Detection with Diffusion Models: Fitness, Generalization, and Transferability
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21042v2](http://arxiv.org/pdf/2506.21042v2)**

> **作者:** Boyong He; Yuxiang Ji; Zhuoyue Tan; Liaoni Wu
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Detectors often suffer from performance drop due to domain gap between training and testing data. Recent methods explore diffusion models applied to domain generalization (DG) and adaptation (DA) tasks, but still struggle with large inference costs and have not yet fully leveraged the capabilities of diffusion models. We propose to tackle these problems by extracting intermediate features from a single-step diffusion process, improving feature collection and fusion to reduce inference time by 75% while enhancing performance on source domains (i.e., Fitness). Then, we construct an object-centered auxiliary branch by applying box-masked images with class prompts to extract robust and domain-invariant features that focus on object. We also apply consistency loss to align the auxiliary and ordinary branch, balancing fitness and generalization while preventing overfitting and improving performance on target domains (i.e., Generalization). Furthermore, within a unified framework, standard detectors are guided by diffusion detectors through feature-level and object-level alignment on source domains (for DG) and unlabeled target domains (for DA), thereby improving cross-domain detection performance (i.e., Transferability). Our method achieves competitive results on 3 DA benchmarks and 5 DG benchmarks. Additionally, experiments on COCO generalization benchmark demonstrate that our method maintains significant advantages and show remarkable efficiency in large domain shifts and low-data scenarios. Our work shows the superiority of applying diffusion models to domain generalized and adaptive detection tasks and offers valuable insights for visual perception tasks across diverse domains. The code is available at \href{https://github.com/heboyong/Fitness-Generalization-Transferability}.
>
---
#### [replaced 047] WeatherEdit: Controllable Weather Editing with 4D Gaussian Field
- **分类: cs.CV; cs.AI; cs.ET; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.20471v2](http://arxiv.org/pdf/2505.20471v2)**

> **作者:** Chenghao Qian; Wenjing Li; Yuhu Guo; Gustav Markkula
>
> **摘要:** In this work, we present WeatherEdit, a novel weather editing pipeline for generating realistic weather effects with controllable types and severity in 3D scenes. Our approach is structured into two key components: weather background editing and weather particle construction. For weather background editing, we introduce an all-in-one adapter that integrates multiple weather styles into a single pretrained diffusion model, enabling the generation of diverse weather effects in 2D image backgrounds. During inference, we design a Temporal-View (TV-) attention mechanism that follows a specific order to aggregate temporal and spatial information, ensuring consistent editing across multi-frame and multi-view images. To construct the weather particles, we first reconstruct a 3D scene using the edited images and then introduce a dynamic 4D Gaussian field to generate snowflakes, raindrops and fog in the scene. The attributes and dynamics of these particles are precisely controlled through physical-based modelling and simulation, ensuring realistic weather representation and flexible severity adjustments. Finally, we integrate the 4D Gaussian field with the 3D scene to render consistent and highly realistic weather effects. Experiments on multiple driving datasets demonstrate that WeatherEdit can generate diverse weather effects with controllable condition severity, highlighting its potential for autonomous driving simulation in adverse weather. See project page: https://jumponthemoon.github.io/w-edit
>
---
#### [replaced 048] Incomplete Multi-view Clustering via Diffusion Contrastive Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.09185v2](http://arxiv.org/pdf/2503.09185v2)**

> **作者:** Yuanyang Zhang; Yijie Lin; Weiqing Yan; Li Yao; Xinhang Wan; Guangyuan Li; Chao Zhang; Guanzhou Ke; Jie Xu
>
> **摘要:** Incomplete multi-view clustering (IMVC) has garnered increasing attention in recent years due to the common issue of missing data in multi-view datasets. The primary approach to address this challenge involves recovering the missing views before applying conventional multi-view clustering methods. Although imputation-based IMVC methods have achieved significant improvements, they still encounter notable limitations: 1) heavy reliance on paired data for training the data recovery module, which is impractical in real scenarios with high missing data rates; 2) the generated data often lacks diversity and discriminability, resulting in suboptimal clustering results. To address these shortcomings, we propose a novel IMVC method called Diffusion Contrastive Generation (DCG). Motivated by the consistency between the diffusion and clustering processes, DCG learns the distribution characteristics to enhance clustering by applying forward diffusion and reverse denoising processes to intra-view data. By performing contrastive learning on a limited set of paired multi-view samples, DCG can align the generated views with the real views, facilitating accurate recovery of views across arbitrary missing view scenarios. Additionally, DCG integrates instance-level and category-level interactive learning to exploit the consistent and complementary information available in multi-view data, achieving robust and end-to-end clustering. Extensive experiments demonstrate that our method outperforms state-of-the-art approaches. The code is available at https://github.com/zhangyuanyang21/2025-AAAI-DCG.
>
---
#### [replaced 049] PriorDiffusion: Leverage Language Prior in Diffusion Models for Monocular Depth Estimation
- **分类: cs.CV; cs.CL; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2411.16750v3](http://arxiv.org/pdf/2411.16750v3)**

> **作者:** Ziyao Zeng; Jingcheng Ni; Daniel Wang; Patrick Rim; Younjoon Chung; Fengyu Yang; Byung-Woo Hong; Alex Wong
>
> **摘要:** Traditional monocular depth estimation suffers from inherent ambiguity and visual nuisance. We argue that language prior can enhance monocular depth estimation by leveraging the inductive bias learned during the text-to-image pre-training of diffusion models. The ability of these models to generate images that align with text indicates that they have learned the spatial relationships, size, and shape of specified objects, which can be applied to improve depth estimation. Thus, we propose PriorDiffusion, using a pre-trained text-to-image diffusion model that takes both images and corresponding text descriptions to infer affine-invariant depth through a denoising process. We also show that language prior enhances the model's perception of specific regions of images that users care about and describe. Simultaneously, language prior acts as a constraint to accelerate the convergence of both training and the inference diffusion trajectory. By training on HyperSim and Virtual KITTI, we achieve faster training convergence, fewer inference diffusion steps, and state-of-the-art zero-shot performance across NYUv2, KITTI, ETH3D, and ScanNet. Code will be released upon acceptance.
>
---
#### [replaced 050] Mesh-Learner: Texturing Mesh with Spherical Harmonics
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.19938v2](http://arxiv.org/pdf/2504.19938v2)**

> **作者:** Yunfei Wan; Jianheng Liu; Chunran Zheng; Jiarong Lin; Fu Zhang
>
> **备注:** IROS2025 Accepted
>
> **摘要:** In this paper, we present a 3D reconstruction and rendering framework termed Mesh-Learner that is natively compatible with traditional rasterization pipelines. It integrates mesh and spherical harmonic (SH) texture (i.e., texture filled with SH coefficients) into the learning process to learn each mesh s view-dependent radiance end-to-end. Images are rendered by interpolating surrounding SH Texels at each pixel s sampling point using a novel interpolation method. Conversely, gradients from each pixel are back-propagated to the related SH Texels in SH textures. Mesh-Learner exploits graphic features of rasterization pipeline (texture sampling, deferred rendering) to render, which makes Mesh-Learner naturally compatible with tools (e.g., Blender) and tasks (e.g., 3D reconstruction, scene rendering, reinforcement learning for robotics) that are based on rasterization pipelines. Our system can train vast, unlimited scenes because we transfer only the SH textures within the frustum to the GPU for training. At other times, the SH textures are stored in CPU RAM, which results in moderate GPU memory usage. The rendering results on interpolation and extrapolation sequences in the Replica and FAST-LIVO2 datasets achieve state-of-the-art performance compared to existing state-of-the-art methods (e.g., 3D Gaussian Splatting and M2-Mapping). To benefit the society, the code will be available at https://github.com/hku-mars/Mesh-Learner.
>
---
#### [replaced 051] CarGait: Cross-Attention based Re-ranking for Gait recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.03501v2](http://arxiv.org/pdf/2503.03501v2)**

> **作者:** Gavriel Habib; Noa Barzilay; Or Shimshi; Rami Ben-Ari; Nir Darshan
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Gait recognition is a computer vision task that identifies individuals based on their walking patterns. Gait recognition performance is commonly evaluated by ranking a gallery of candidates and measuring the accuracy at the top Rank-$K$. Existing models are typically single-staged, i.e. searching for the probe's nearest neighbors in a gallery using a single global feature representation. Although these models typically excel at retrieving the correct identity within the top-$K$ predictions, they struggle when hard negatives appear in the top short-list, leading to relatively low performance at the highest ranks (e.g., Rank-1). In this paper, we introduce CarGait, a Cross-Attention Re-ranking method for gait recognition, that involves re-ordering the top-$K$ list leveraging the fine-grained correlations between pairs of gait sequences through cross-attention between gait strips. This re-ranking scheme can be adapted to existing single-stage models to enhance their final results. We demonstrate the capabilities of CarGait by extensive experiments on three common gait datasets, Gait3D, GREW, and OU-MVLP, and seven different gait models, showing consistent improvements in Rank-1,5 accuracy, superior results over existing re-ranking methods, and strong baselines.
>
---
#### [replaced 052] Multimodal Object Detection using Depth and Image Data for Manufacturing Parts
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2411.09062v3](http://arxiv.org/pdf/2411.09062v3)**

> **作者:** Nazanin Mahjourian; Vinh Nguyen
>
> **摘要:** Manufacturing requires reliable object detection methods for precise picking and handling of diverse types of manufacturing parts and components. Traditional object detection methods utilize either only 2D images from cameras or 3D data from lidars or similar 3D sensors. However, each of these sensors have weaknesses and limitations. Cameras do not have depth perception and 3D sensors typically do not carry color information. These weaknesses can undermine the reliability and robustness of industrial manufacturing systems. To address these challenges, this work proposes a multi-sensor system combining an red-green-blue (RGB) camera and a 3D point cloud sensor. The two sensors are calibrated for precise alignment of the multimodal data captured from the two hardware devices. A novel multimodal object detection method is developed to process both RGB and depth data. This object detector is based on the Faster R-CNN baseline that was originally designed to process only camera images. The results show that the multimodal model significantly outperforms the depth-only and RGB-only baselines on established object detection metrics. More specifically, the multimodal model improves mAP by 13% and raises Mean Precision by 11.8% in comparison to the RGB-only baseline. Compared to the depth-only baseline, it improves mAP by 78% and raises Mean Precision by 57%. Hence, this method facilitates more reliable and robust object detection in service to smart manufacturing applications.
>
---
#### [replaced 053] Advancing Textual Prompt Learning with Anchored Attributes
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.09442v2](http://arxiv.org/pdf/2412.09442v2)**

> **作者:** Zheng Li; Yibing Song; Ming-Ming Cheng; Xiang Li; Jian Yang
>
> **备注:** ICCV 2025. Project Page: https://zhengli97.github.io/ATPrompt/
>
> **摘要:** Textual-based prompt learning methods primarily employ multiple learnable soft prompts and hard class tokens in a cascading manner as text inputs, aiming to align image and text (category) spaces for downstream tasks. However, current training is restricted to aligning images with predefined known categories and cannot be associated with unknown categories. In this work, we propose utilizing universal attributes as a bridge to enhance the alignment between images and unknown categories. Specifically, we introduce an Attribute-anchored Textual Prompt learning method for vision-language models, named ATPrompt. This approach expands the learning space of soft prompts from the original one-dimensional category level into the multi-dimensional attribute level by incorporating multiple attribute tokens into the learnable soft prompts. Through this modification, we transform the text prompt from a category-centric form to an attribute-category hybrid form. Additionally, we introduce a straightforward differentiable attribute search method to identify representative and suitable attributes for downstream tasks. As an easy-to-use plug-in technique, ATPrompt can seamlessly replace the existing basic prompt format in textual-based methods, providing general improvements at a negligible computational cost. Extensive experiments across 11 datasets validate the effectiveness of our method.
>
---
#### [replaced 054] Accurate and lightweight dehazing via multi-receptive-field non-local network and novel contrastive regularization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2309.16494v2](http://arxiv.org/pdf/2309.16494v2)**

> **作者:** Zewei He; Zixuan Chen; Jinlei Li; Ziqian Lu; Xuecheng Sun; Hao Luo; Zhe-Ming Lu; Evangelos K. Markakis
>
> **备注:** submitted to the IEEE Journal for possible publication
>
> **摘要:** Recently, deep learning-based methods have dominated image dehazing domain. Although very competitive dehazing performance has been achieved with sophisticated models, effective solutions for extracting useful features are still under-explored. In addition, non-local network, which has made a breakthrough in many vision tasks, has not been appropriately applied to image dehazing. Thus, a multi-receptive-field non-local network (MRFNLN) consisting of the multi-stream feature attention block (MSFAB) and cross non-local block (CNLB) is presented in this paper. We start with extracting richer features for dehazing. Specifically, we design a multi-stream feature extraction (MSFE) sub-block, which contains three parallel convolutions with different receptive fields (i.e., $1\times 1$, $3\times 3$, $5\times 5$) for extracting multi-scale features. Following MSFE, we employ an attention sub-block to make the model adaptively focus on important channels/regions. The MSFE and attention sub-blocks constitute our MSFAB. Then, we design a cross non-local block (CNLB), which can capture long-range dependencies beyond the query. Instead of the same input source of query branch, the key and value branches are enhanced by fusing more preceding features. CNLB is computation-friendly by leveraging a spatial pyramid down-sampling (SPDS) strategy to reduce the computation and memory consumption without sacrificing the performance. Last but not least, a novel detail-focused contrastive regularization (DFCR) is presented by emphasizing the low-level details and ignoring the high-level semantic information in the representation space. Comprehensive experimental results demonstrate that the proposed MRFNLN model outperforms recent state-of-the-art dehazing methods with less than 1.5 Million parameters.
>
---
#### [replaced 055] Seedream 3.0 Technical Report
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.11346v3](http://arxiv.org/pdf/2504.11346v3)**

> **作者:** Yu Gao; Lixue Gong; Qiushan Guo; Xiaoxia Hou; Zhichao Lai; Fanshi Li; Liang Li; Xiaochen Lian; Chao Liao; Liyang Liu; Wei Liu; Yichun Shi; Shiqi Sun; Yu Tian; Zhi Tian; Peng Wang; Rui Wang; Xuanda Wang; Xun Wang; Ye Wang; Guofeng Wu; Jie Wu; Xin Xia; Xuefeng Xiao; Zhonghua Zhai; Xinyu Zhang; Qi Zhang; Yuwei Zhang; Shijia Zhao; Jianchao Yang; Weilin Huang
>
> **备注:** Seedream 3.0 Technical Report
>
> **摘要:** We present Seedream 3.0, a high-performance Chinese-English bilingual image generation foundation model. We develop several technical improvements to address existing challenges in Seedream 2.0, including alignment with complicated prompts, fine-grained typography generation, suboptimal visual aesthetics and fidelity, and limited image resolutions. Specifically, the advancements of Seedream 3.0 stem from improvements across the entire pipeline, from data construction to model deployment. At the data stratum, we double the dataset using a defect-aware training paradigm and a dual-axis collaborative data-sampling framework. Furthermore, we adopt several effective techniques such as mixed-resolution training, cross-modality RoPE, representation alignment loss, and resolution-aware timestep sampling in the pre-training phase. During the post-training stage, we utilize diversified aesthetic captions in SFT, and a VLM-based reward model with scaling, thereby achieving outputs that well align with human preferences. Furthermore, Seedream 3.0 pioneers a novel acceleration paradigm. By employing consistent noise expectation and importance-aware timestep sampling, we achieve a 4 to 8 times speedup while maintaining image quality. Seedream 3.0 demonstrates significant improvements over Seedream 2.0: it enhances overall capabilities, in particular for text-rendering in complicated Chinese characters which is important to professional typography generation. In addition, it provides native high-resolution output (up to 2K), allowing it to generate images with high visual quality.
>
---
#### [replaced 056] Interpretable Interaction Modeling for Trajectory Prediction via Agent Selection and Physical Coefficient
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.13152v5](http://arxiv.org/pdf/2405.13152v5)**

> **作者:** Shiji Huang; Lei Ye; Min Chen; Wenhai Luo; Dihong Wang; Chenqi Xu; Deyuan Liang
>
> **备注:** Accepted by International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** A thorough understanding of the interaction between the target agent and surrounding agents is a prerequisite for accurate trajectory prediction. Although many methods have been explored, they assign correlation coefficients to surrounding agents in a purely learning-based manner. In this study, we present ASPILin, which manually selects interacting agents and replaces the attention scores in Transformer with a newly computed physical correlation coefficient, enhancing the interpretability of interaction modeling. Surprisingly, these simple modifications can significantly improve prediction performance and substantially reduce computational costs. We intentionally simplified our model in other aspects, such as map encoding. Remarkably, experiments conducted on the INTERACTION, highD, and CitySim datasets demonstrate that our method is efficient and straightforward, outperforming other state-of-the-art methods.
>
---
#### [replaced 057] How to Move Your Dragon: Text-to-Motion Synthesis for Large-Vocabulary Objects
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.04257v2](http://arxiv.org/pdf/2503.04257v2)**

> **作者:** Wonkwang Lee; Jongwon Jeong; Taehong Moon; Hyeon-Jong Kim; Jaehyeon Kim; Gunhee Kim; Byeong-Uk Lee
>
> **备注:** Accepted to ICML 2025
>
> **摘要:** Motion synthesis for diverse object categories holds great potential for 3D content creation but remains underexplored due to two key challenges: (1) the lack of comprehensive motion datasets that include a wide range of high-quality motions and annotations, and (2) the absence of methods capable of handling heterogeneous skeletal templates from diverse objects. To address these challenges, we contribute the following: First, we augment the Truebones Zoo dataset, a high-quality animal motion dataset covering over 70 species, by annotating it with detailed text descriptions, making it suitable for text-based motion synthesis. Second, we introduce rig augmentation techniques that generate diverse motion data while preserving consistent dynamics, enabling models to adapt to various skeletal configurations. Finally, we redesign existing motion diffusion models to dynamically adapt to arbitrary skeletal templates, enabling motion synthesis for a diverse range of objects with varying structures. Experiments show that our method learns to generate high-fidelity motions from textual descriptions for diverse and even unseen objects, setting a strong foundation for motion synthesis across diverse object categories and skeletal templates. Qualitative results are available at: $\href{https://t2m4lvo.github.io}{https://t2m4lvo.github.io}$.
>
---
#### [replaced 058] YOLO-LLTS: Real-Time Low-Light Traffic Sign Detection via Prior-Guided Enhancement and Multi-Branch Feature Interaction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.13883v3](http://arxiv.org/pdf/2503.13883v3)**

> **作者:** Ziyu Lin; Yunfan Wu; Yuhang Ma; Junzhou Chen; Ronghui Zhang; Jiaming Wu; Guodong Yin; Liang Lin
>
> **摘要:** Traffic sign detection is essential for autonomous driving and Advanced Driver Assistance Systems (ADAS). However, existing methods struggle with low-light conditions due to issues like indistinct small-object features, limited feature interaction, and poor image quality, which degrade detection accuracy and speed. To address this issue, we propose YOLO-LLTS, an end-to-end real-time traffic sign detection algorithm specifically designed for low-light environments. YOLO-LLTS introduces three main contributions: the High-Resolution Feature Map for Small Object Detection (HRFM-SOD) module to enhance small-object detection by mitigating feature dilution; the Multi-branch Feature Interaction Attention (MFIA) module to improve information extraction through multi-scale features interaction; and the Prior-Guided Feature Enhancement Module (PGFE) to enhance image quality by addressing noise, low contrast, and blurriness. Additionally, we construct a novel dataset, the Chinese Nighttime Traffic Sign Sample Set (CNTSSS), covering diverse nighttime scenarios. Experiments show that YOLO-LLTS achieves state-of-the-art performance, outperforming previous best methods by 2.7% mAP50 and 1.6% mAP50:95 on TT100K-night, 1.3% mAP50 and 1.9% mAP50:95 on CNTSSS, 7.5% mAP50 and 9.8% mAP50:95 on GTSDB-night, and superior results on CCTSDB2021. Deployment on edge devices confirms its real-time applicability and effectiveness.
>
---
#### [replaced 059] ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing
- **分类: eess.AS; cs.CV; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.21448v2](http://arxiv.org/pdf/2506.21448v2)**

> **作者:** Huadai Liu; Jialei Wang; Kaicheng Luo; Wen Wang; Qian Chen; Zhou Zhao; Wei Xue
>
> **摘要:** While end-to-end video-to-audio generation has greatly improved, producing high-fidelity audio that authentically captures the nuances of visual content remains challenging. Like professionals in the creative industries, such generation requires sophisticated reasoning about items such as visual dynamics, acoustic environments, and temporal relationships. We present ThinkSound, a novel framework that leverages Chain-of-Thought (CoT) reasoning to enable stepwise, interactive audio generation and editing for videos. Our approach decomposes the process into three complementary stages: foundational foley generation that creates semantically coherent soundscapes, interactive object-centric refinement through precise user interactions, and targeted editing guided by natural language instructions. At each stage, a multimodal large language model generates contextually aligned CoT reasoning that guides a unified audio foundation model. Furthermore, we introduce AudioCoT, a comprehensive dataset with structured reasoning annotations that establishes connections between visual content, textual descriptions, and sound synthesis. Experiments demonstrate that ThinkSound achieves state-of-the-art performance in video-to-audio generation across both audio metrics and CoT metrics and excels in out-of-distribution Movie Gen Audio benchmark. The demo page is available at https://ThinkSound-Project.github.io.
>
---
#### [replaced 060] FALCON: Resolving Visual Redundancy and Fragmentation in High-resolution Multimodal Large Language Models via Visual Registers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.16297v2](http://arxiv.org/pdf/2501.16297v2)**

> **作者:** Renshan Zhang; Rui Shao; Gongwei Chen; Miao Zhang; Kaiwen Zhou; Weili Guan; Liqiang Nie
>
> **备注:** Accepted to the IEEE/CVF International Conference on Computer Vision (ICCV) 2025
>
> **摘要:** The incorporation of high-resolution visual input equips multimodal large language models (MLLMs) with enhanced visual perception capabilities for real-world tasks. However, most existing high-resolution MLLMs rely on a cropping-based approach to process images, which leads to fragmented visual encoding and a sharp increase in redundant tokens. To tackle these issues, we propose the FALCON model. FALCON introduces a novel visual register technique to simultaneously: 1) Eliminate redundant tokens at the stage of visual encoding. To directly address the visual redundancy present in the output of vision encoder, we propose a Register-based Representation Compacting (ReCompact) mechanism. This mechanism introduces a set of learnable visual registers designed to adaptively aggregate essential information while discarding redundancy. It enables the encoder to produce a more compact visual representation with a minimal number of output tokens, thus eliminating the need for an additional compression module. 2) Ensure continuity in visual encoding. To address the potential encoding errors caused by fragmented visual inputs, we develop a Register Interactive Attention (ReAtten) module. This module facilitates effective and efficient information exchange across sub-images by enabling interactions between visual registers. It ensures the continuity of visual semantics throughout the encoding. We conduct comprehensive experiments with FALCON on high-resolution benchmarks across a wide range of scenarios. FALCON demonstrates superior performance with a remarkable 9-fold reduction in visual tokens.
>
---
#### [replaced 061] Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey and Benchmark
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.02242v5](http://arxiv.org/pdf/2402.02242v5)**

> **作者:** Yi Xin; Jianjiang Yang; Siqi Luo; Yuntao Du; Qi Qin; Kangrui Cen; Yangfan He; Bin Fu; Xiaokang Yang; Guangtao Zhai; Ming-Hsuan Yang; Xiaohong Liu
>
> **备注:** Submitted to IEEE TPAMI
>
> **摘要:** Pre-trained vision models (PVMs) have demonstrated remarkable adaptability across a wide range of downstream vision tasks, showcasing exceptional performance. However, as these models scale to billions or even trillions of parameters, conventional full fine-tuning has become increasingly impractical due to its high computational and storage demands. To address these challenges, parameter-efficient fine-tuning (PEFT) has emerged as a promising alternative, aiming to achieve performance comparable to full fine-tuning while making minimal adjustments to the model parameters. This paper presents a comprehensive survey of the latest advancements in the visual PEFT field, systematically reviewing current methodologies and categorizing them into four primary categories: addition-based, partial-based, unified-based, and multi-task tuning. In addition, this paper offers an in-depth analysis of widely used visual datasets and real-world applications where PEFT methods have been successfully applied. Furthermore, this paper introduces the V-PEFT Bench, a unified benchmark designed to standardize the evaluation of PEFT methods across a diverse set of vision tasks, ensuring consistency and fairness in comparison. Finally, the paper outlines potential directions for future research to propel advances in the PEFT field. A comprehensive collection of resources is available at https://github.com/synbol/Awesome-Parameter-Efficient-Transfer-Learning.
>
---
#### [replaced 062] AEM: Attention Entropy Maximization for Multiple Instance Learning based Whole Slide Image Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2406.15303v3](http://arxiv.org/pdf/2406.15303v3)**

> **作者:** Yunlong Zhang; Honglin Li; Yunxuan Sun; Zhongyi Shui; Jingxiong Li; Chenglu Zhu; Lin Yang
>
> **备注:** Accepted by MICCAI2025
>
> **摘要:** Multiple Instance Learning (MIL) effectively analyzes whole slide images but faces overfitting due to attention over-concentration. While existing solutions rely on complex architectural modifications or additional processing steps, we introduce Attention Entropy Maximization (AEM), a simple yet effective regularization technique. Our investigation reveals the positive correlation between attention entropy and model performance. Building on this insight, we integrate AEM regularization into the MIL framework to penalize excessive attention concentration. To address sensitivity to the AEM weight parameter, we implement Cosine Weight Annealing, reducing parameter dependency. Extensive evaluations demonstrate AEM's superior performance across diverse feature extractors, MIL frameworks, attention mechanisms, and augmentation techniques. Here is our anonymous code: https://github.com/dazhangyu123/AEM.
>
---
#### [replaced 063] Vision-QRWKV: Exploring Quantum-Enhanced RWKV Models for Image Classification
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.06633v2](http://arxiv.org/pdf/2506.06633v2)**

> **作者:** Chi-Sheng Chen
>
> **摘要:** Recent advancements in quantum machine learning have shown promise in enhancing classical neural network architectures, particularly in domains involving complex, high-dimensional data. Building upon prior work in temporal sequence modeling, this paper introduces Vision-QRWKV, a hybrid quantum-classical extension of the Receptance Weighted Key Value (RWKV) architecture, applied for the first time to image classification tasks. By integrating a variational quantum circuit (VQC) into the channel mixing component of RWKV, our model aims to improve nonlinear feature transformation and enhance the expressive capacity of visual representations. We evaluate both classical and quantum RWKV models on a diverse collection of 14 medical and standard image classification benchmarks, including MedMNIST datasets, MNIST, and FashionMNIST. Our results demonstrate that the quantum-enhanced model outperforms its classical counterpart on a majority of datasets, particularly those with subtle or noisy class distinctions (e.g., ChestMNIST, RetinaMNIST, BloodMNIST). This study represents the first systematic application of quantum-enhanced RWKV in the visual domain, offering insights into the architectural trade-offs and future potential of quantum models for lightweight and efficient vision tasks.
>
---
#### [replaced 064] Deepfake Caricatures: Amplifying attention to artifacts increases deepfake detection by humans and machines
- **分类: cs.CV; cs.HC; cs.SI**

- **链接: [http://arxiv.org/pdf/2206.00535v4](http://arxiv.org/pdf/2206.00535v4)**

> **作者:** Camilo Fosco; Emilie Josephs; Alex Andonian; Aude Oliva
>
> **备注:** 11 pages, 5 figures, 4 tables
>
> **摘要:** Deepfakes can fuel online misinformation. As deepfakes get harder to recognize with the naked eye, human users become more reliant on deepfake detection models to help them decide whether a video is real or fake. Currently, models yield a prediction for a video's authenticity, but do not integrate a method for alerting a human user. We introduce a framework for amplifying artifacts in deepfake videos to make them more detectable by people. We propose a novel, semi-supervised Artifact Attention module, which is trained on human responses to create attention maps that highlight video artifacts, and magnify them to create a novel visual indicator we call "Deepfake Caricatures". In a user study, we demonstrate that Caricatures greatly increase human detection, across video presentation times and user engagement levels. We also introduce a deepfake detection model that incorporates the Artifact Attention module to increase its accuracy and robustness. Overall, we demonstrate the success of a human-centered approach to designing deepfake mitigation methods.
>
---
#### [replaced 065] Dehazing Light Microscopy Images with Guided Conditional Flow Matching: finding a sweet spot between fidelity and realism
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.22397v2](http://arxiv.org/pdf/2506.22397v2)**

> **作者:** Anirban Ray; Ashesh; Florian Jug
>
> **备注:** 4 figures, 10 pages + refs, 40 pages total (including supplement), 24 supplementary figures
>
> **摘要:** Fluorescence microscopy is a major driver of scientific progress in the life sciences. Although high-end confocal microscopes are capable of filtering out-of-focus light, cheaper and more accessible microscopy modalities, such as widefield microscopy, can not, which consequently leads to hazy image data. Computational dehazing is trying to combine the best of both worlds, leading to cheap microscopy but crisp-looking images. The perception-distortion trade-off tells us that we can optimize either for data fidelity, e.g. low MSE or high PSNR, or for data realism, measured by perceptual metrics such as LPIPS or FID. Existing methods either prioritize fidelity at the expense of realism, or produce perceptually convincing results that lack quantitative accuracy. In this work, we propose HazeMatching, a novel iterative method for dehazing light microscopy images, which effectively balances these objectives. Our goal was to find a balanced trade-off between the fidelity of the dehazing results and the realism of individual predictions (samples). We achieve this by adapting the conditional flow matching framework by guiding the generative process with a hazy observation in the conditional velocity field. We evaluate HazeMatching on 5 datasets, covering both synthetic and real data, assessing both distortion and perceptual quality. Our method is compared against 7 baselines, achieving a consistent balance between fidelity and realism on average. Additionally, with calibration analysis, we show that HazeMatching produces well-calibrated predictions. Note that our method does not need an explicit degradation operator to exist, making it easily applicable on real microscopy data. All data used for training and evaluation and our code will be publicly available under a permissive license.
>
---
#### [replaced 066] Vision Technologies with Applications in Traffic Surveillance Systems: A Holistic Survey
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.00348v2](http://arxiv.org/pdf/2412.00348v2)**

> **作者:** Wei Zhou; Li Yang; Lei Zhao; Runyu Zhang; Yifan Cui; Hongpu Huang; Kun Qie; Chen Wang
>
> **摘要:** Traffic Surveillance Systems (TSS) have become increasingly crucial in modern intelligent transportation systems, with vision technologies playing a central role for scene perception and understanding. While existing surveys typically focus on isolated aspects of TSS, a comprehensive analytical framework bridging low-level and high-level perception tasks, particularly considering emerging technologies, remains lacking. This paper presents a systematic review of vision technologies in TSS, examining both low-level perception tasks (object detection, classification, and tracking) and high-level perception tasks (parameter estimation, anomaly detection, and behavior understanding). Specifically, we first provide a detailed methodological categorization and comprehensive performance evaluation for each task. Our investigation reveals five fundamental limitations in current TSS: perceptual data degradation in complex scenarios, data-driven learning constraints, semantic understanding gaps, sensing coverage limitations and computational resource demands. To address these challenges, we systematically analyze five categories of current approaches and potential trends: advanced perception enhancement, efficient learning paradigms, knowledge-enhanced understanding, cooperative sensing frameworks and efficient computing frameworks, critically assessing their real-world applicability. Furthermore, we evaluate the transformative potential of foundation models in TSS, which exhibit remarkable zero-shot learning abilities, strong generalization, and sophisticated reasoning capabilities across diverse tasks. This review provides a unified analytical framework bridging low-level and high-level perception tasks, systematically analyzes current limitations and solutions, and presents a structured roadmap for integrating emerging technologies, particularly foundation models, to enhance TSS capabilities.
>
---
#### [replaced 067] InstructionBench: An Instructional Video Understanding Benchmark
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.05040v2](http://arxiv.org/pdf/2504.05040v2)**

> **作者:** Haiwan Wei; Yitian Yuan; Xiaohan Lan; Wei Ke; Lin Ma
>
> **摘要:** Despite progress in video large language models (Video-LLMs), research on instructional video understanding, crucial for enhancing access to instructional content, remains insufficient. To address this, we introduce InstructionBench, an Instructional video understanding Benchmark, which challenges models' advanced temporal reasoning within instructional videos characterized by their strict step-by-step flow. Employing GPT-4, we formulate Q&A pairs in open-ended and multiple-choice formats to assess both Coarse-Grained event-level and Fine-Grained object-level reasoning. Our filtering strategies exclude questions answerable purely by common-sense knowledge, focusing on visual perception and analysis when evaluating Video-LLM models. The benchmark finally contains 5k questions across over 700 videos. We evaluate the latest Video-LLMs on our InstructionBench, finding that closed-source models outperform open-source ones. However, even the best model, GPT-4o, achieves only 53.42% accuracy, indicating significant gaps in temporal reasoning. To advance the field, we also develop a comprehensive instructional video dataset with over 19k Q&A pairs from nearly 2.5k videos, using an automated data generation framework, thereby enriching the community's research resources. All data are available at https://huggingface.co/datasets/sunwhw/InstructionBench.
>
---
#### [replaced 068] Edit360: 2D Image Edits to 3D Assets from Any Angle
- **分类: cs.GR; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.10507v2](http://arxiv.org/pdf/2506.10507v2)**

> **作者:** Junchao Huang; Xinting Hu; Shaoshuai Shi; Zhuotao Tian; Li Jiang
>
> **备注:** 11 pages, 9 figures
>
> **摘要:** Recent advances in diffusion models have significantly improved image generation and editing, but extending these capabilities to 3D assets remains challenging, especially for fine-grained edits that require multi-view consistency. Existing methods typically restrict editing to predetermined viewing angles, severely limiting their flexibility and practical applications. We introduce Edit360, a tuning-free framework that extends 2D modifications to multi-view consistent 3D editing. Built upon video diffusion models, Edit360 enables user-specific editing from arbitrary viewpoints while ensuring structural coherence across all views. The framework selects anchor views for 2D modifications and propagates edits across the entire 360-degree range. To achieve this, Edit360 introduces a novel Anchor-View Editing Propagation mechanism, which effectively aligns and merges multi-view information within the latent and attention spaces of diffusion models. The resulting edited multi-view sequences facilitate the reconstruction of high-quality 3D assets, enabling customizable 3D content creation.
>
---
#### [replaced 069] CountLLM: Towards Generalizable Repetitive Action Counting via Large Language Model
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.17690v2](http://arxiv.org/pdf/2503.17690v2)**

> **作者:** Ziyu Yao; Xuxin Cheng; Zhiqi Huang; Lei Li
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** Repetitive action counting, which aims to count periodic movements in a video, is valuable for video analysis applications such as fitness monitoring. However, existing methods largely rely on regression networks with limited representational capacity, which hampers their ability to accurately capture variable periodic patterns. Additionally, their supervised learning on narrow, limited training sets leads to overfitting and restricts their ability to generalize across diverse scenarios. To address these challenges, we propose CountLLM, the first large language model (LLM)-based framework that takes video data and periodic text prompts as inputs and outputs the desired counting value. CountLLM leverages the rich clues from explicit textual instructions and the powerful representational capabilities of pre-trained LLMs for repetitive action counting. To effectively guide CountLLM, we develop a periodicity-based structured template for instructions that describes the properties of periodicity and implements a standardized answer format to ensure consistency. Additionally, we propose a progressive multimodal training paradigm to enhance the periodicity-awareness of the LLM. Empirical evaluations on widely recognized benchmarks demonstrate CountLLM's superior performance and generalization, particularly in handling novel and out-of-domain actions that deviate significantly from the training data, offering a promising avenue for repetitive action counting.
>
---
#### [replaced 070] Accelerate 3D Object Detection Models via Zero-Shot Attention Key Pruning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.08101v3](http://arxiv.org/pdf/2503.08101v3)**

> **作者:** Lizhen Xu; Xiuxiu Bai; Xiaojun Jia; Jianwu Fang; Shanmin Pang
>
> **备注:** Accepted by ICCV2025. The code can be found at https://github.com/iseri27/tg_gbc
>
> **摘要:** Query-based methods with dense features have demonstrated remarkable success in 3D object detection tasks. However, the computational demands of these models, particularly with large image sizes and multiple transformer layers, pose significant challenges for efficient running on edge devices. Existing pruning and distillation methods either need retraining or are designed for ViT models, which are hard to migrate to 3D detectors. To address this issue, we propose a zero-shot runtime pruning method for transformer decoders in 3D object detection models. The method, termed tgGBC (trim keys gradually Guided By Classification scores), systematically trims keys in transformer modules based on their importance. We expand the classification score to multiply it with the attention map to get the importance score of each key and then prune certain keys after each transformer layer according to their importance scores. Our method achieves a 1.99x speedup in the transformer decoder of the latest ToC3D model, with only a minimal performance loss of less than 1%. Interestingly, for certain models, our method even enhances their performance. Moreover, we deploy 3D detectors with tgGBC on an edge device, further validating the effectiveness of our method. The code can be found at https://github.com/iseri27/tg_gbc.
>
---
#### [replaced 071] USP: Unified Self-Supervised Pretraining for Image Generation and Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06132v2](http://arxiv.org/pdf/2503.06132v2)**

> **作者:** Xiangxiang Chu; Renda Li; Yong Wang
>
> **备注:** Accepted to ICCV2025
>
> **摘要:** Recent studies have highlighted the interplay between diffusion models and representation learning. Intermediate representations from diffusion models can be leveraged for downstream visual tasks, while self-supervised vision models can enhance the convergence and generation quality of diffusion models. However, transferring pretrained weights from vision models to diffusion models is challenging due to input mismatches and the use of latent spaces. To address these challenges, we propose Unified Self-supervised Pretraining (USP), a framework that initializes diffusion models via masked latent modeling in a Variational Autoencoder (VAE) latent space. USP achieves comparable performance in understanding tasks while significantly improving the convergence speed and generation quality of diffusion models. Our code will be publicly available at https://github.com/AMAP-ML/USP.
>
---
#### [replaced 072] FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; eess.IV**

- **链接: [http://arxiv.org/pdf/2412.01064v3](http://arxiv.org/pdf/2412.01064v3)**

> **作者:** Taekyung Ki; Dongchan Min; Gyeongsu Chae
>
> **备注:** ICCV 2025. Project page: https://deepbrainai-research.github.io/float/
>
> **摘要:** With the rapid advancement of diffusion-based generative models, portrait image animation has achieved remarkable results. However, it still faces challenges in temporally consistent video generation and fast sampling due to its iterative sampling nature. This paper presents FLOAT, an audio-driven talking portrait video generation method based on flow matching generative model. Instead of a pixel-based latent space, we take advantage of a learned orthogonal motion latent space, enabling efficient generation and editing of temporally consistent motion. To achieve this, we introduce a transformer-based vector field predictor with an effective frame-wise conditioning mechanism. Additionally, our method supports speech-driven emotion enhancement, enabling a natural incorporation of expressive motions. Extensive experiments demonstrate that our method outperforms state-of-the-art audio-driven talking portrait methods in terms of visual quality, motion fidelity, and efficiency.
>
---
#### [replaced 073] Relating Events and Frames Based on Self-Supervised Learning and Uncorrelated Conditioning for Unsupervised Domain Adaptation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2401.01042v2](http://arxiv.org/pdf/2401.01042v2)**

> **作者:** Mohammad Rostami; Dayuan Jian; Ruitong Sun
>
> **摘要:** Event-based cameras provide accurate and high temporal resolution measurements for performing computer vision tasks in challenging scenarios, such as high-dynamic range environments and fast-motion maneuvers. Despite their advantages, utilizing deep learning for event-based vision encounters a significant obstacle due to the scarcity of annotated data caused by the relatively recent emergence of event-based cameras. To overcome this limitation, leveraging the knowledge available from annotated data obtained with conventional frame-based cameras presents an effective solution based on unsupervised domain adaptation. We propose a new algorithm tailored for adapting a deep neural network trained on annotated frame-based data to generalize well on event-based unannotated data. Our approach incorporates uncorrelated conditioning and self-supervised learning in an adversarial learning scheme to close the gap between the two source and target domains. By applying self-supervised learning, the algorithm learns to align the representations of event-based data with those from frame-based camera data, thereby facilitating knowledge transfer.Furthermore, the inclusion of uncorrelated conditioning ensures that the adapted model effectively distinguishes between event-based and conventional data, enhancing its ability to classify event-based images accurately.Through empirical experimentation and evaluation, we demonstrate that our algorithm surpasses existing approaches designed for the same purpose using two benchmarks. The superior performance of our solution is attributed to its ability to effectively utilize annotated data from frame-based cameras and transfer the acquired knowledge to the event-based vision domain.
>
---
#### [replaced 074] Semantic-Aware Adaptive Video Streaming Using Latent Diffusion Models for Wireless Networks
- **分类: cs.MM; cs.AI; cs.CV; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2502.05695v2](http://arxiv.org/pdf/2502.05695v2)**

> **作者:** Zijiang Yan; Jianhua Pei; Hongda Wu; Hina Tabassum; Ping Wang
>
> **备注:** Accepted in IEEE Wireless Communications
>
> **摘要:** This paper proposes a novel Semantic Communication (SemCom) framework for real-time adaptive-bitrate video streaming by integrating Latent Diffusion Models (LDMs) within the FFmpeg techniques. This solution addresses the challenges of high bandwidth usage, storage inefficiencies, and quality of experience (QoE) degradation associated with traditional Constant Bitrate Streaming (CBS) and Adaptive Bitrate Streaming (ABS). The proposed approach leverages LDMs to compress I-frames into a latent space, offering significant storage and semantic transmission savings without sacrificing high visual quality. While retaining B-frames and P-frames as adjustment metadata to support efficient refinement of video reconstruction at the user side, the proposed framework further incorporates state-of-the-art denoising and Video Frame Interpolation (VFI) techniques. These techniques mitigate semantic ambiguity and restore temporal coherence between frames, even in noisy wireless communication environments. Experimental results demonstrate the proposed method achieves high-quality video streaming with optimized bandwidth usage, outperforming state-of-the-art solutions in terms of QoE and resource efficiency. This work opens new possibilities for scalable real-time video streaming in 5G and future post-5G networks.
>
---
#### [replaced 075] Environment-Driven Online LiDAR-Camera Extrinsic Calibration
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2502.00801v2](http://arxiv.org/pdf/2502.00801v2)**

> **作者:** Zhiwei Huang; Jiaqi Li; Ping Zhong; Rui Fan
>
> **摘要:** LiDAR-camera extrinsic calibration (LCEC) is crucial for multi-modal data fusion in mechatronics. Existing methods, whether target-based or target-free, typically rely on customized calibration targets or fixed scene types, limiting their practicality in real-world applications. To address these challenges, we introduce EdO-LCEC, the first environment-driven online calibration approach. Unlike traditional target-free methods, EdO-LCEC observes the feature density of the application environment through a generalizable scene discriminator. Based on this feature density, EdO-LCEC extracts LiDAR intensity and depth features from varying perspectives to achieve higher calibration accuracy. To overcome the challenges of cross-modal feature matching between LiDAR and camera, we propose dual-path correspondence matching (DPCM), which leverages both structural and textural consistency for reliable 3D-2D correspondences. Additionally, our approach models the calibration process as a joint optimization problem utilizing global constraints from multiple views and scenes to enhance accuracy. Extensive experiments on real-world datasets demonstrate that EdO-LCEC outperforms state-of-the-art methods, particularly in sparse or partially overlapping sensor views.
>
---
#### [replaced 076] LatentMove: Towards Complex Human Movement Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.22046v2](http://arxiv.org/pdf/2505.22046v2)**

> **作者:** Ashkan Taghipour; Morteza Ghahremani; Mohammed Bennamoun; Farid Boussaid; Aref Miri Rekavandi; Zinuo Li; Qiuhong Ke; Hamid Laga
>
> **备注:** The authors are withdrawing this paper due to major issues in the experiments and methodology. To prevent citation of this outdated and flawed version, we have decided to remove it while we work on a substantial revision. Thank you
>
> **摘要:** Image-to-video (I2V) generation seeks to produce realistic motion sequences from a single reference image. Although recent methods exhibit strong temporal consistency, they often struggle when dealing with complex, non-repetitive human movements, leading to unnatural deformations. To tackle this issue, we present LatentMove, a DiT-based framework specifically tailored for highly dynamic human animation. Our architecture incorporates a conditional control branch and learnable face/body tokens to preserve consistency as well as fine-grained details across frames. We introduce Complex-Human-Videos (CHV), a dataset featuring diverse, challenging human motions designed to benchmark the robustness of I2V systems. We also introduce two metrics to assess the flow and silhouette consistency of generated videos with their ground truth. Experimental results indicate that LatentMove substantially improves human animation quality--particularly when handling rapid, intricate movements--thereby pushing the boundaries of I2V generation. The code, the CHV dataset, and the evaluation metrics will be available at https://github.com/ --.
>
---
#### [replaced 077] Assessing workflow impact and clinical utility of AI-assisted brain aneurysm detection: a multi-reader study
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.17786v2](http://arxiv.org/pdf/2503.17786v2)**

> **作者:** Tommaso Di Noto; Sofyan Jankowski; Francesco Puccinelli; Guillaume Marie; Sebastien Tourbier; Yasser Aleman-Gomez; Oscar Esteban; Ricardo Corredor-Jerez; Guillaume Saliou; Patric Hagmann; Meritxell Bach Cuadra; Jonas Richiardi
>
> **备注:** This paper has been accepted for publication in the journal NeuroImage: Clinical (DOI: https://doi.org/10.1016/j.nicl.2025.103835)
>
> **摘要:** Despite the plethora of AI-based algorithms developed for anomaly detection in radiology, subsequent integration into clinical setting is rarely evaluated. In this work, we assess the applicability and utility of an AI-based model for brain aneurysm detection comparing the performance of two readers with different levels of experience (2 and 13 years). We aim to answer the following questions: 1) Do the readers improve their performance when assisted by the AI algorithm? 2) How much does the AI algorithm impact routine clinical workflow? We reuse and enlarge our open-access, Time-Of-Flight Magnetic Resonance Angiography dataset (N=460). We use 360 subjects for training/validating our algorithm and 100 as unseen test set for the reading session. Even though our model reaches state-of-the-art results on the test set (sensitivity=74%, false positive rate=1.6), we show that neither the junior nor the senior reader significantly increase their sensitivity (p=0.59, p=1, respectively). In addition, we find that reading time for both readers is significantly higher in the "AI-assisted" setting than in the "Unassisted" (+15 seconds, on average; p=3x10^(-4) junior, p=3x10^(-5) senior). The confidence reported by the readers is unchanged across the two settings, indicating that the AI assistance does not influence the certainty of the diagnosis. Our findings highlight the importance of clinical validation of AI algorithms in a clinical setting involving radiologists. This study should serve as a reminder to the community to always examine the real-word effectiveness and workflow impact of proposed algorithms.
>
---
#### [replaced 078] OmniHuman-1: Rethinking the Scaling-Up of One-Stage Conditioned Human Animation Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.01061v3](http://arxiv.org/pdf/2502.01061v3)**

> **作者:** Gaojie Lin; Jianwen Jiang; Jiaqi Yang; Zerong Zheng; Chao Liang
>
> **备注:** ICCV 2025, Homepage: https://omnihuman-lab.github.io/
>
> **摘要:** End-to-end human animation, such as audio-driven talking human generation, has undergone notable advancements in the recent few years. However, existing methods still struggle to scale up as large general video generation models, limiting their potential in real applications. In this paper, we propose OmniHuman, a Diffusion Transformer-based framework that scales up data by mixing motion-related conditions into the training phase. To this end, we introduce two training principles for these mixed conditions, along with the corresponding model architecture and inference strategy. These designs enable OmniHuman to fully leverage data-driven motion generation, ultimately achieving highly realistic human video generation. More importantly, OmniHuman supports various portrait contents (face close-up, portrait, half-body, full-body), supports both talking and singing, handles human-object interactions and challenging body poses, and accommodates different image styles. Compared to existing end-to-end audio-driven methods, OmniHuman not only produces more realistic videos, but also offers greater flexibility in inputs. It also supports multiple driving modalities (audio-driven, video-driven and combined driving signals). Video samples are provided on the ttfamily project page (https://omnihuman-lab.github.io)
>
---
#### [replaced 079] DepthART: Monocular Depth Estimation as Autoregressive Refinement Task
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.15010v3](http://arxiv.org/pdf/2409.15010v3)**

> **作者:** Bulat Gabdullin; Nina Konovalova; Nikolay Patakin; Dmitry Senushkin; Anton Konushin
>
> **摘要:** Monocular depth estimation has seen significant advances through discriminative approaches, yet their performance remains constrained by the limitations of training datasets. While generative approaches have addressed this challenge by leveraging priors from internet-scale datasets, with recent studies showing state-of-the-art results using fine-tuned text-to-image diffusion models, there is still room for improvement. Notably, autoregressive generative approaches, particularly Visual AutoRegressive modeling, have demonstrated superior results compared to diffusion models in conditioned image synthesis, while offering faster inference times. In this work, we apply Visual Autoregressive Transformer (VAR) to the monocular depth estimation problem. However, the conventional GPT-2-style training procedure (teacher forcing) inherited by VAR yields suboptimal results for depth estimation. To address this limitation, we introduce DepthART - a novel training method formulated as a Depth Autoregressive Refinement Task. Unlike traditional VAR training with static inputs and targets, our method implements a dynamic target formulation based on model outputs, enabling self-refinement. By utilizing the model's own predictions as inputs instead of ground truth token maps during training, we frame the objective as residual minimization, effectively reducing the discrepancy between training and inference procedures. Our experimental results demonstrate that the proposed training approach significantly enhances the performance of VAR in depth estimation tasks. When trained on Hypersim dataset using our approach, the model achieves superior results across multiple unseen benchmarks compared to existing generative and discriminative baselines.
>
---
#### [replaced 080] G$^{2}$D: Boosting Multimodal Learning with Gradient-Guided Distillation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21514v2](http://arxiv.org/pdf/2506.21514v2)**

> **作者:** Mohammed Rakib; Arunkumar Bagavathi
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Multimodal learning aims to leverage information from diverse data modalities to achieve more comprehensive performance. However, conventional multimodal models often suffer from modality imbalance, where one or a few modalities dominate model optimization, leading to suboptimal feature representation and underutilization of weak modalities. To address this challenge, we introduce Gradient-Guided Distillation (G$^{2}$D), a knowledge distillation framework that optimizes the multimodal model with a custom-built loss function that fuses both unimodal and multimodal objectives. G$^{2}$D further incorporates a dynamic sequential modality prioritization (SMP) technique in the learning process to ensure each modality leads the learning process, avoiding the pitfall of stronger modalities overshadowing weaker ones. We validate G$^{2}$D on multiple real-world datasets and show that G$^{2}$D amplifies the significance of weak modalities while training and outperforms state-of-the-art methods in classification and regression tasks. Our code is available at https://github.com/rAIson-Lab/G2D.
>
---
#### [replaced 081] Compositional Generative Model of Unbounded 4D Cities
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.08983v3](http://arxiv.org/pdf/2501.08983v3)**

> **作者:** Haozhe Xie; Zhaoxi Chen; Fangzhou Hong; Ziwei Liu
>
> **备注:** Project Page: https://www.infinitescript.com/project/city-dreamer-4d/
>
> **摘要:** 3D scene generation has garnered growing attention in recent years and has made significant progress. Generating 4D cities is more challenging than 3D scenes due to the presence of structurally complex, visually diverse objects like buildings and vehicles, and heightened human sensitivity to distortions in urban environments. To tackle these issues, we propose CityDreamer4D, a compositional generative model specifically tailored for generating unbounded 4D cities. Our main insights are 1) 4D city generation should separate dynamic objects (e.g., vehicles) from static scenes (e.g., buildings and roads), and 2) all objects in the 4D scene should be composed of different types of neural fields for buildings, vehicles, and background stuff. Specifically, we propose Traffic Scenario Generator and Unbounded Layout Generator to produce dynamic traffic scenarios and static city layouts using a highly compact BEV representation. Objects in 4D cities are generated by combining stuff-oriented and instance-oriented neural fields for background stuff, buildings, and vehicles. To suit the distinct characteristics of background stuff and instances, the neural fields employ customized generative hash grids and periodic positional embeddings as scene parameterizations. Furthermore, we offer a comprehensive suite of datasets for city generation, including OSM, GoogleEarth, and CityTopia. The OSM dataset provides a variety of real-world city layouts, while the Google Earth and CityTopia datasets deliver large-scale, high-quality city imagery complete with 3D instance annotations. Leveraging its compositional design, CityDreamer4D supports a range of downstream applications, such as instance editing, city stylization, and urban simulation, while delivering state-of-the-art performance in generating realistic 4D cities.
>
---
#### [replaced 082] Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21401v2](http://arxiv.org/pdf/2506.21401v2)**

> **作者:** Zhirui Gao; Renjiao Yi; Yaqiao Dai; Xuening Zhu; Wei Chen; Chenyang Zhu; Kai Xu
>
> **备注:** Accepted by ICCV 2025 Code: https://github.com/zhirui-gao/Curve-Gaussian
>
> **摘要:** This paper presents an end-to-end framework for reconstructing 3D parametric curves directly from multi-view edge maps. Contrasting with existing two-stage methods that follow a sequential ``edge point cloud reconstruction and parametric curve fitting'' pipeline, our one-stage approach optimizes 3D parametric curves directly from 2D edge maps, eliminating error accumulation caused by the inherent optimization gap between disconnected stages. However, parametric curves inherently lack suitability for rendering-based multi-view optimization, necessitating a complementary representation that preserves their geometric properties while enabling differentiable rendering. We propose a novel bi-directional coupling mechanism between parametric curves and edge-oriented Gaussian components. This tight correspondence formulates a curve-aware Gaussian representation, \textbf{CurveGaussian}, that enables differentiable rendering of 3D curves, allowing direct optimization guided by multi-view evidence. Furthermore, we introduce a dynamically adaptive topology optimization framework during training to refine curve structures through linearization, merging, splitting, and pruning operations. Comprehensive evaluations on the ABC dataset and real-world benchmarks demonstrate our one-stage method's superiority over two-stage alternatives, particularly in producing cleaner and more robust reconstructions. Additionally, by directly optimizing parametric curves, our method significantly reduces the parameter count during training, achieving both higher efficiency and superior performance compared to existing approaches.
>
---
#### [replaced 083] Seedance 1.0: Exploring the Boundaries of Video Generation Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.09113v2](http://arxiv.org/pdf/2506.09113v2)**

> **作者:** Yu Gao; Haoyuan Guo; Tuyen Hoang; Weilin Huang; Lu Jiang; Fangyuan Kong; Huixia Li; Jiashi Li; Liang Li; Xiaojie Li; Xunsong Li; Yifu Li; Shanchuan Lin; Zhijie Lin; Jiawei Liu; Shu Liu; Xiaonan Nie; Zhiwu Qing; Yuxi Ren; Li Sun; Zhi Tian; Rui Wang; Sen Wang; Guoqiang Wei; Guohong Wu; Jie Wu; Ruiqi Xia; Fei Xiao; Xuefeng Xiao; Jiangqiao Yan; Ceyuan Yang; Jianchao Yang; Runkai Yang; Tao Yang; Yihang Yang; Zilyu Ye; Xuejiao Zeng; Yan Zeng; Heng Zhang; Yang Zhao; Xiaozheng Zheng; Peihao Zhu; Jiaxin Zou; Feilong Zuo
>
> **备注:** Seedance 1.0 Technical Report
>
> **摘要:** Notable breakthroughs in diffusion modeling have propelled rapid improvements in video generation, yet current foundational model still face critical challenges in simultaneously balancing prompt following, motion plausibility, and visual quality. In this report, we introduce Seedance 1.0, a high-performance and inference-efficient video foundation generation model that integrates several core technical improvements: (i) multi-source data curation augmented with precision and meaningful video captioning, enabling comprehensive learning across diverse scenarios; (ii) an efficient architecture design with proposed training paradigm, which allows for natively supporting multi-shot generation and jointly learning of both text-to-video and image-to-video tasks. (iii) carefully-optimized post-training approaches leveraging fine-grained supervised fine-tuning, and video-specific RLHF with multi-dimensional reward mechanisms for comprehensive performance improvements; (iv) excellent model acceleration achieving ~10x inference speedup through multi-stage distillation strategies and system-level optimizations. Seedance 1.0 can generate a 5-second video at 1080p resolution only with 41.4 seconds (NVIDIA-L20). Compared to state-of-the-art video generation models, Seedance 1.0 stands out with high-quality and fast video generation having superior spatiotemporal fluidity with structural stability, precise instruction adherence in complex multi-subject contexts, native multi-shot narrative coherence with consistent subject representation.
>
---
#### [replaced 084] AQUA20: A Benchmark Dataset for Underwater Species Classification under Challenging Conditions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.17455v2](http://arxiv.org/pdf/2506.17455v2)**

> **作者:** Taufikur Rahman Fuad; Sabbir Ahmed; Shahriar Ivan
>
> **备注:** Submitted to AJSE Springer
>
> **摘要:** Robust visual recognition in underwater environments remains a significant challenge due to complex distortions such as turbidity, low illumination, and occlusion, which severely degrade the performance of standard vision systems. This paper introduces AQUA20, a comprehensive benchmark dataset comprising 8,171 underwater images across 20 marine species reflecting real-world environmental challenges such as illumination, turbidity, occlusions, etc., providing a valuable resource for underwater visual understanding. Thirteen state-of-the-art deep learning models, including lightweight CNNs (SqueezeNet, MobileNetV2) and transformer-based architectures (ViT, ConvNeXt), were evaluated to benchmark their performance in classifying marine species under challenging conditions. Our experimental results show ConvNeXt achieving the best performance, with a Top-3 accuracy of 98.82% and a Top-1 accuracy of 90.69%, as well as the highest overall F1-score of 88.92% with moderately large parameter size. The results obtained from our other benchmark models also demonstrate trade-offs between complexity and performance. We also provide an extensive explainability analysis using GRAD-CAM and LIME for interpreting the strengths and pitfalls of the models. Our results reveal substantial room for improvement in underwater species recognition and demonstrate the value of AQUA20 as a foundation for future research in this domain. The dataset is publicly available at: https://huggingface.co/datasets/taufiktrf/AQUA20.
>
---
#### [replaced 085] I see what you mean: Co-Speech Gestures for Reference Resolution in Multimodal Dialogue
- **分类: cs.CV; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.00071v3](http://arxiv.org/pdf/2503.00071v3)**

> **作者:** Esam Ghaleb; Bulat Khaertdinov; Aslı Özyürek; Raquel Fernández
>
> **备注:** Accepted to Findings of ACL 2025
>
> **摘要:** In face-to-face interaction, we use multiple modalities, including speech and gestures, to communicate information and resolve references to objects. However, how representational co-speech gestures refer to objects remains understudied from a computational perspective. In this work, we address this gap by introducing a multimodal reference resolution task centred on representational gestures, while simultaneously tackling the challenge of learning robust gesture embeddings. We propose a self-supervised pre-training approach to gesture representation learning that grounds body movements in spoken language. Our experiments show that the learned embeddings align with expert annotations and have significant predictive power. Moreover, reference resolution accuracy further improves when (1) using multimodal gesture representations, even when speech is unavailable at inference time, and (2) leveraging dialogue history. Overall, our findings highlight the complementary roles of gesture and speech in reference resolution, offering a step towards more naturalistic models of human-machine interaction.
>
---
#### [replaced 086] Multi-encoder nnU-Net outperforms transformer models with self-supervised pretraining
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.03474v2](http://arxiv.org/pdf/2504.03474v2)**

> **作者:** Seyedeh Sahar Taheri Otaghsara; Reza Rahmanzadeh
>
> **摘要:** This study addresses the essential task of medical image segmentation, which involves the automatic identification and delineation of anatomical structures and pathological regions in medical images. Accurate segmentation is crucial in radiology, as it aids in the precise localization of abnormalities such as tumors, thereby enabling effective diagnosis, treatment planning, and monitoring of disease progression. Specifically, the size, shape, and location of tumors can significantly influence clinical decision-making and therapeutic strategies, making accurate segmentation a key component of radiological workflows. However, challenges posed by variations in MRI modalities, image artifacts, and the scarcity of labeled data complicate the segmentation task and impact the performance of traditional models. To overcome these limitations, we propose a novel self-supervised learning Multi-encoder nnU-Net architecture designed to process multiple MRI modalities independently through separate encoders. This approach allows the model to capture modality-specific features before fusing them for the final segmentation, thus improving accuracy. Our Multi-encoder nnU-Net demonstrates exceptional performance, achieving a Dice Similarity Coefficient (DSC) of 93.72%, which surpasses that of other models such as vanilla nnU-Net, SegResNet, and Swin UNETR. By leveraging the unique information provided by each modality, the model enhances segmentation tasks, particularly in scenarios with limited annotated data. Evaluations highlight the effectiveness of this architecture in improving tumor segmentation outcomes.
>
---
#### [replaced 087] SP$^2$OT: Semantic-Regularized Progressive Partial Optimal Transport for Imbalanced Clustering
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2404.03446v2](http://arxiv.org/pdf/2404.03446v2)**

> **作者:** Chuyu Zhang; Hui Ren; Xuming He
>
> **备注:** under review. Follow-up work of arXiv:2401.09266
>
> **摘要:** Deep clustering, which learns representation and semantic clustering without labels information, poses a great challenge for deep learning-based approaches. Despite significant progress in recent years, most existing methods focus on uniformly distributed datasets, significantly limiting the practical applicability of their methods. In this paper, we propose a more practical problem setting named deep imbalanced clustering, where the underlying classes exhibit an imbalance distribution. To address this challenge, we introduce a novel optimal transport-based pseudo-label learning framework. Our framework formulates pseudo-label generation as a Semantic-regularized Progressive Partial Optimal Transport (SP$^2$OT) problem, which progressively transports each sample to imbalanced clusters under prior and semantic relation constraints, thus generating high-quality and imbalance-aware pseudo-labels. To solve the SP$^2$OT problem, we propose a projected mirror descent algorithm, which alternates between: (1) computing the gradient of the SP$^2$OT objective, and (2) performing gradient descent with projection via an entropy-regularized progressive partial optimal transport formulation. Furthermore, we formulate the second step as an unbalanced optimal transport problem with augmented constraints and develop an efficient solution based on fast matrix scaling algorithms. Experiments on various datasets, including a human-curated long-tailed CIFAR100, challenging ImageNet-R, and large-scale subsets of fine-grained iNaturalist2018 datasets, demonstrate the superiority of our method. Code is available: https://github.com/rhfeiyang/SPPOT
>
---
#### [replaced 088] ZipAR: Parallel Auto-regressive Image Generation through Spatial Locality
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.04062v3](http://arxiv.org/pdf/2412.04062v3)**

> **作者:** Yefei He; Feng Chen; Yuanyu He; Shaoxuan He; Hong Zhou; Kaipeng Zhang; Bohan Zhuang
>
> **备注:** 11 pages
>
> **摘要:** In this paper, we propose ZipAR, a training-free, plug-and-play parallel decoding framework for accelerating auto-regressive (AR) visual generation. The motivation stems from the observation that images exhibit local structures, and spatially distant regions tend to have minimal interdependence. Given a partially decoded set of visual tokens, in addition to the original next-token prediction scheme in the row dimension, the tokens corresponding to spatially adjacent regions in the column dimension can be decoded in parallel, enabling the ``next-set prediction'' paradigm. By decoding multiple tokens simultaneously in a single forward pass, the number of forward passes required to generate an image is significantly reduced, resulting in a substantial improvement in generation efficiency. Experiments demonstrate that ZipAR can reduce the number of model forward passes by up to 91% on the Emu3-Gen model without requiring any additional retraining. Code is available here: https://github.com/ThisisBillhe/ZipAR.
>
---
#### [replaced 089] Composing Parts for Expressive Object Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.10197v2](http://arxiv.org/pdf/2406.10197v2)**

> **作者:** Harsh Rangwani; Aishwarya Agarwal; Kuldeep Kulkarni; R. Venkatesh Babu; Srikrishna Karanam
>
> **备注:** Project Page Will Be Here: https://rangwani-harsh.github.io/PartCraft
>
> **摘要:** Image composition and generation are processes where the artists need control over various parts of the generated images. However, the current state-of-the-art generation models, like Stable Diffusion, cannot handle fine-grained part-level attributes in the text prompts. Specifically, when additional attribute details are added to the base text prompt, these text-to-image models either generate an image vastly different from the image generated from the base prompt or ignore the attribute details. To mitigate these issues, we introduce PartComposer, a training-free method that enables image generation based on fine-grained part-level attributes specified for objects in the base text prompt. This allows more control for artists and enables novel object compositions by combining distinctive object parts. PartComposer first localizes object parts by denoising the object region from a specific diffusion process. This enables each part token to be localized to the right region. After obtaining part masks, we run a localized diffusion process in each part region based on fine-grained part attributes and combine them to produce the final image. All stages of PartComposer are based on repurposing a pre-trained diffusion model, which enables it to generalize across domains. We demonstrate the effectiveness of part-level control provided by PartComposer through qualitative visual examples and quantitative comparisons with contemporary baselines.
>
---
#### [replaced 090] Understanding and Reducing the Class-Dependent Effects of Data Augmentation with A Two-Player Game Approach
- **分类: cs.CY; cs.AI; cs.CV; cs.GT; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.03146v5](http://arxiv.org/pdf/2407.03146v5)**

> **作者:** Yunpeng Jiang; Yutong Ban; Paul Weng
>
> **备注:** Published in Transactions on Machine Learning Research (06/2025)
>
> **摘要:** Data augmentation is widely applied and has shown its benefits in different machine learning tasks. However, as recently observed, it may have an unfair effect in multi-class classification. While data augmentation generally improves the overall performance (and therefore is beneficial for many classes), it can actually be detrimental for other classes, which can be problematic in some application domains. In this paper, to counteract this phenomenon, we propose CLAM, a CLAss-dependent Multiplicative-weights method. To derive it, we first formulate the training of a classifier as a non-linear optimization problem that aims at simultaneously maximizing the individual class performances and balancing them. By rewriting this optimization problem as an adversarial two-player game, we propose a novel multiplicative weight algorithm, for which we prove the convergence. Interestingly, our formulation also reveals that the class-dependent effects of data augmentation is not due to data augmentation only, but is in fact a general phenomenon. Our empirical results over six datasets demonstrate that the performance of learned classifiers is indeed more fairly distributed over classes, with only limited impact on the average accuracy.
>
---
#### [replaced 091] FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Foundation Models
- **分类: cs.DC; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.09432v4](http://arxiv.org/pdf/2410.09432v4)**

> **作者:** Raghav Singhal; Kaustubh Ponkshe; Praneeth Vepakomma
>
> **备注:** ACL 2025 - Oral. Raghav Singhal and Kaustubh Ponkshe contributed equally to this work
>
> **摘要:** Low-Rank Adaptation (LoRA) is a popular technique for efficient fine-tuning of foundation models. However, applying LoRA in federated learning environments, where data is distributed across multiple clients, presents unique challenges. Existing methods rely on traditional federated averaging of LoRA adapters, resulting in inexact updates. To address this, we propose Federated Exact LoRA, or FedEx-LoRA, which adds a residual error term to the pretrained frozen weight matrix. Our approach achieves exact updates with minimal computational and communication overhead, preserving LoRA's efficiency. We evaluate the method on various models across arithmetic reasoning, commonsense reasoning, natural language understanding and natural language generation tasks, showing consistent performance gains over state-of-the-art methods across multiple settings. Through extensive analysis, we quantify that the deviations in updates from the ideal solution are significant, highlighting the need for exact aggregation. Our method's simplicity, efficiency, and broad applicability position it as a promising solution for accurate and effective federated fine-tuning of foundation models. Our code is publicly available at https://github.com/RaghavSinghal10/fedex-lora.
>
---
#### [replaced 092] Fetuses Made Simple: Modeling and Tracking of Fetal Shape and Pose
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.17858v2](http://arxiv.org/pdf/2506.17858v2)**

> **作者:** Yingcheng Liu; Peiqi Wang; Sebastian Diaz; Esra Abaci Turk; Benjamin Billot; Patricia Ellen Grant; Polina Golland
>
> **摘要:** Analyzing fetal body motion and shape is paramount in prenatal diagnostics and monitoring. Existing methods for fetal MRI analysis mainly rely on anatomical keypoints or volumetric body segmentations. Keypoints simplify body structure to facilitate motion analysis, but may ignore important details of full-body shape. Body segmentations capture complete shape information but complicate temporal analysis due to large non-local fetal movements. To address these limitations, we construct a 3D articulated statistical fetal body model based on the Skinned Multi-Person Linear Model (SMPL). Our algorithm iteratively estimates body pose in the image space and body shape in the canonical pose space. This approach improves robustness to MRI motion artifacts and intensity distortions, and reduces the impact of incomplete surface observations due to challenging fetal poses. We train our model on segmentations and keypoints derived from $19,816$ MRI volumes across $53$ subjects. Our model captures body shape and motion across time series and provides intuitive visualization. Furthermore, it enables automated anthropometric measurements traditionally difficult to obtain from segmentations and keypoints. When tested on unseen fetal body shapes, our method yields a surface alignment error of $3.2$ mm for $3$ mm MRI voxel size. To our knowledge, this represents the first 3D articulated statistical fetal body model, paving the way for enhanced fetal motion and shape analysis in prenatal diagnostics. The code is available at https://github.com/MedicalVisionGroup/fetal-smpl .
>
---
#### [replaced 093] FlatFusion: Delving into Details of Sparse Transformer-based Camera-LiDAR Fusion for Autonomous Driving
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.06832v2](http://arxiv.org/pdf/2408.06832v2)**

> **作者:** Yutao Zhu; Xiaosong Jia; Xinyu Yang; Junchi Yan
>
> **备注:** Accepted by ICRA 2025
>
> **摘要:** The integration of data from diverse sensor modalities (e.g., camera and LiDAR) constitutes a prevalent methodology within the ambit of autonomous driving scenarios. Recent advancements in efficient point cloud transformers have underscored the efficacy of integrating information in sparse formats. When it comes to fusion, since image patches are dense in pixel space with ambiguous depth, it necessitates additional design considerations for effective fusion. In this paper, we conduct a comprehensive exploration of design choices for Transformer-based sparse cameraLiDAR fusion. This investigation encompasses strategies for image-to-3D and LiDAR-to-2D mapping, attention neighbor grouping, single modal tokenizer, and micro-structure of Transformer. By amalgamating the most effective principles uncovered through our investigation, we introduce FlatFusion, a carefully designed framework for sparse camera-LiDAR fusion. Notably, FlatFusion significantly outperforms state-of-the-art sparse Transformer-based methods, including UniTR, CMT, and SparseFusion, achieving 73.7 NDS on the nuScenes validation set with 10.1 FPS with PyTorch.
>
---
#### [replaced 094] LW2G: Learning Whether to Grow for Prompt-based Continual Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.18860v2](http://arxiv.org/pdf/2409.18860v2)**

> **作者:** Qian Feng; Da-wei Zhou; Hanbin Zhao; Chao Zhang; Jiahua Dong; Dengxin Dai; Hui Qian
>
> **摘要:** Recent Prompt-based Continual learning (PCL) has achieved remarkable performance with pre-trained models. These approaches expand a prompt pool by adding a new set of prompts while learning and select the correct set during inference. Previous studies have revealed that learning task-wised prompt sets individually and low selection accuracy pose challenges to the performance of PCL. In this paper, we propose a plug-in method, $\textbf{L}$earning $\textbf{W}$hether $\textbf{t}$o $\textbf{G}$row $\textbf{(LW2G)}$, which leverages the disparities between tasks to form an effective and efficient prompt sets pool, thereby achieving intra-task knowledge sharing and cooperation and avoiding the unbounded increase in the cost of the prompt pool. Specifically, a shared set is utilized when several tasks share certain commonalities, and a new set is added when there are significant differences between the new and previous tasks. To achieve this, we develop a metric called Hinder Forward Capability (HFC) to measure the hindrance imposed on learning new tasks by surgically modifying the original gradient onto the orthogonal complement of the old feature space. With HFC, an automated scheme, Dynamic Growing Approach, adaptively learns whether to grow with a dynamic threshold. Furthermore, we design a gradient-based constraint to ensure consistency between the updating prompts and pre-trained knowledge. Extensive experiments show the effectiveness of our method. Code is available at https://github.com/RAIAN08/LW2G.
>
---
#### [replaced 095] OmniEval: A Benchmark for Evaluating Omni-modal Models with Visual, Auditory, and Textual Inputs
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.20960v2](http://arxiv.org/pdf/2506.20960v2)**

> **作者:** Yiman Zhang; Ziheng Luo; Qiangyu Yan; Wei He; Borui Jiang; Xinghao Chen; Kai Han
>
> **摘要:** In this paper, we introduce OmniEval, a benchmark for evaluating omni-modality models like MiniCPM-O 2.6, which encompasses visual, auditory, and textual inputs. Compared with existing benchmarks, our OmniEval has several distinctive features: (i) Full-modal collaboration: We design evaluation tasks that highlight the strong coupling between audio and video, requiring models to effectively leverage the collaborative perception of all modalities; (ii) Diversity of videos: OmniEval includes 810 audio-visual synchronized videos, 285 Chinese videos and 525 English videos; (iii) Diversity and granularity of tasks: OmniEval contains 2617 question-answer pairs, comprising 1412 open-ended questions and 1205 multiple-choice questions. These questions are divided into 3 major task types and 12 sub-task types to achieve comprehensive evaluation. Among them, we introduce a more granular video localization task named Grounding. Then we conduct experiments on OmniEval with several omni-modality models. We hope that our OmniEval can provide a platform for evaluating the ability to construct and understand coherence from the context of all modalities. Codes and data could be found at https://omnieval-benchmark.github.io/.
>
---
#### [replaced 096] OrderChain: Towards General Instruct-Tuning for Stimulating the Ordinal Understanding Ability of MLLM
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.04801v2](http://arxiv.org/pdf/2504.04801v2)**

> **作者:** Jinhong Wang; Shuo Tong; Jian liu; Dongqi Tang; Weiqiang Wang; Wentong Li; Hongxia Xu; Danny Chen; Jintai Chen; Jian Wu
>
> **摘要:** Despite the remarkable progress of multimodal large language models (MLLMs), they continue to face challenges in achieving competitive performance on ordinal regression (OR; a.k.a. ordinal classification). To address this issue, this paper presents OrderChain, a novel and general prompting paradigm that improves the ordinal understanding ability of MLLMs by specificity and commonality modeling. Specifically, our OrderChain consists of a set of task-aware prompts to facilitate the specificity modeling of diverse OR tasks and a new range optimization Chain-of-Thought (RO-CoT), which learns a commonality way of thinking about OR tasks by uniformly decomposing them into multiple small-range optimization subtasks. Further, we propose a category recursive division (CRD) method to generate instruction candidate category prompts to support RO-CoT automatic optimization. Comprehensive experiments show that a Large Language and Vision Assistant (LLaVA) model with our OrderChain improves baseline LLaVA significantly on diverse OR datasets, e.g., from 47.5% to 93.2% accuracy on the Adience dataset for age estimation, and from 30.0% to 85.7% accuracy on the Diabetic Retinopathy dataset. Notably, LLaVA with our OrderChain also remarkably outperforms state-of-the-art methods by 27% on accuracy and 0.24 on MAE on the Adience dataset. To our best knowledge, our OrderChain is the first work that augments MLLMs for OR tasks, and the effectiveness is witnessed across a spectrum of OR datasets.
>
---
#### [replaced 097] General Compression Framework for Efficient Transformer Object Tracking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.17564v2](http://arxiv.org/pdf/2409.17564v2)**

> **作者:** Lingyi Hong; Jinglun Li; Xinyu Zhou; Shilin Yan; Pinxue Guo; Kaixun Jiang; Zhaoyu Chen; Shuyong Gao; Runze Li; Xingdong Sheng; Wei Zhang; Hong Lu; Wenqiang Zhang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Previous works have attempted to improve tracking efficiency through lightweight architecture design or knowledge distillation from teacher models to compact student trackers. However, these solutions often sacrifice accuracy for speed to a great extent, and also have the problems of complex training process and structural limitations. Thus, we propose a general model compression framework for efficient transformer object tracking, named CompressTracker, to reduce model size while preserving tracking accuracy. Our approach features a novel stage division strategy that segments the transformer layers of the teacher model into distinct stages to break the limitation of model structure. Additionally, we also design a unique replacement training technique that randomly substitutes specific stages in the student model with those from the teacher model, as opposed to training the student model in isolation. Replacement training enhances the student model's ability to replicate the teacher model's behavior and simplifies the training process. To further forcing student model to emulate teacher model, we incorporate prediction guidance and stage-wise feature mimicking to provide additional supervision during the teacher model's compression process. CompressTracker is structurally agnostic, making it compatible with any transformer architecture. We conduct a series of experiment to verify the effectiveness and generalizability of our CompressTracker. Our CompressTracker-SUTrack, compressed from SUTrack, retains about 99 performance on LaSOT (72.2 AUC) while achieves 2.42x speed up. Code is available at https://github.com/LingyiHongfd/CompressTracker.
>
---
#### [replaced 098] Deblurring in the Wild: A Real-World Dataset from Smartphone High-Speed Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.19445v2](http://arxiv.org/pdf/2506.19445v2)**

> **作者:** Mahdi Mohd Hossain Noki; Syed Mumtahin Mahmud; Prothito Shovon Majumder; Abdul Mohaimen Al Radi; Md. Haider Ali; Md. Mosaddek Khan
>
> **备注:** 8 pages (without references), 3 figures. Dataset https://huggingface.co/datasets/masterda/SloMoBlur
>
> **摘要:** We introduce the largest real-world image deblurring dataset constructed from smartphone slow-motion videos. Using 240 frames captured over one second, we simulate realistic long-exposure blur by averaging frames to produce blurry images, while using the temporally centered frame as the sharp reference. Our dataset contains over 42,000 high-resolution blur-sharp image pairs, making it approximately 10 times larger than widely used datasets, with 8 times the amount of different scenes, including indoor and outdoor environments, with varying object and camera motions. We benchmark multiple state-of-the-art (SOTA) deblurring models on our dataset and observe significant performance degradation, highlighting the complexity and diversity of our benchmark. Our dataset serves as a challenging new benchmark to facilitate robust and generalizable deblurring models.
>
---
#### [replaced 099] Efficient Diffusion Training through Parallelization with Truncated Karhunen-Loève Expansion
- **分类: cs.CV; I.2.0; I.4.0**

- **链接: [http://arxiv.org/pdf/2503.17657v2](http://arxiv.org/pdf/2503.17657v2)**

> **作者:** Yumeng Ren; Yaofang Liu; Aitor Artola; Laurent Mertz; Raymond H. Chan; Jean-michel Morel
>
> **备注:** 12 pages, 9 figures
>
> **摘要:** Diffusion denoising models have become a popular approach for image generation, but they often suffer from slow convergence during training. In this paper, we identify that this slow convergence is partly due to the complexity of the Brownian motion driving the forward-time process. To address this, we represent the Brownian motion using the Karhunen-Lo\`eve expansion, truncating it to a limited number of eigenfunctions. We propose a novel ordinary differential equation with augmented random initials, termed KL diffusion, as a new forward-time process for training and sampling. By developing an appropriate denoising loss function, we facilitate the integration of our KL-diffusion into existing denoising-based models. Using the widely adopted DDIM framework as our baseline ensures a fair comparison, as our modifications focus solely on the forward process and loss function, leaving the network architecture and sampling methods unchanged. Our method significantly outperforms baseline diffusion models, achieving convergence speeds that are twice faster to reach the best FID score of the baseline and ultimately yielding much lower FID scores. Notably, our approach allows for highly parallelized computation, requires no additional learnable parameters, and can be flexibly integrated into existing diffusion methods. The code will be made publicly available.
>
---
#### [replaced 100] Grid: Omni Visual Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.10718v5](http://arxiv.org/pdf/2412.10718v5)**

> **作者:** Cong Wan; Xiangyang Luo; Hao Luo; Zijian Cai; Yiren Song; Yunlong Zhao; Yifan Bai; Fan Wang; Yuhang He; Yihong Gong
>
> **备注:** Codes: https://github.com/Should-AI-Lab/GRID
>
> **摘要:** Visual generation has witnessed remarkable progress in single-image tasks, yet extending these capabilities to temporal sequences remains challenging. Current approaches either build specialized video models from scratch with enormous computational costs or add separate motion modules to image generators, both requiring learning temporal dynamics anew. We observe that modern image generation models possess underutilized potential in handling structured layouts with implicit temporal understanding. Building on this insight, we introduce GRID, which reformulates temporal sequences as grid layouts, enabling holistic processing of visual sequences while leveraging existing model capabilities. Through a parallel flow-matching training strategy with coarse-to-fine scheduling, our approach achieves up to 67 faster inference speeds while using <1/1000 of the computational resources compared to specialized models. Extensive experiments demonstrate that GRID not only excels in temporal tasks from Text-to-Video to 3D Editing but also preserves strong performance in image generation, establishing itself as an efficient and versatile omni-solution for visual generation.
>
---
#### [replaced 101] Emulating Self-attention with Convolution for Efficient Image Super-Resolution
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06671v2](http://arxiv.org/pdf/2503.06671v2)**

> **作者:** Dongheon Lee; Seokju Yun; Youngmin Ro
>
> **备注:** ICCV 2025
>
> **摘要:** In this paper, we tackle the high computational overhead of Transformers for efficient image super-resolution~(SR). Motivated by the observations of self-attention's inter-layer repetition, we introduce a convolutionized self-attention module named Convolutional Attention~(ConvAttn) that emulates self-attention's long-range modeling capability and instance-dependent weighting with a single shared large kernel and dynamic kernels. By utilizing the ConvAttn module, we significantly reduce the reliance on self-attention and its involved memory-bound operations while maintaining the representational capability of Transformers. Furthermore, we overcome the challenge of integrating flash attention into the lightweight SR regime, effectively mitigating self-attention's inherent memory bottleneck. We scale up the window size to 32$\times$32 with flash attention rather than proposing an intricate self-attention module, significantly improving PSNR by 0.31dB on Urban100$\times$2 while reducing latency and memory usage by 16$\times$ and 12.2$\times$. Building on these approaches, our proposed network, termed Emulating Self-attention with Convolution~(ESC), notably improves PSNR by 0.27 dB on Urban100$\times$4 compared to HiT-SRF, reducing the latency and memory usage by 3.7$\times$ and 6.2$\times$, respectively. Extensive experiments demonstrate that our ESC maintains the ability for long-range modeling, data scalability, and the representational power of Transformers despite most self-attention being replaced by the ConvAttn module.
>
---
#### [replaced 102] Enhancing Adversarial Robustness through Multi-Objective Representation Learning
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.01697v4](http://arxiv.org/pdf/2410.01697v4)**

> **作者:** Sedjro Salomon Hotegni; Sebastian Peitz
>
> **摘要:** Deep neural networks (DNNs) are vulnerable to small adversarial perturbations, which are tiny changes to the input data that appear insignificant but cause the model to produce drastically different outputs. Many defense methods require modifying model architectures during evaluation or performing test-time data purification. This not only introduces additional complexity but is often architecture-dependent. We show, however, that robust feature learning during training can significantly enhance DNN robustness. We propose MOREL, a multi-objective approach that aligns natural and adversarial features using cosine similarity and multi-positive contrastive losses to encourage similar features for same-class inputs. Extensive experiments demonstrate that MOREL significantly improves robustness against both white-box and black-box attacks. Our code is available at https://github.com/salomonhotegni/MOREL
>
---
#### [replaced 103] BraTS-PEDs: Results of the Multi-Consortium International Pediatric Brain Tumor Segmentation Challenge 2023
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2407.08855v3](http://arxiv.org/pdf/2407.08855v3)**

> **作者:** Anahita Fathi Kazerooni; Nastaran Khalili; Xinyang Liu; Debanjan Haldar; Zhifan Jiang; Anna Zapaishchykova; Julija Pavaine; Lubdha M. Shah; Blaise V. Jones; Nakul Sheth; Sanjay P. Prabhu; Aaron S. McAllister; Wenxin Tu; Khanak K. Nandolia; Andres F. Rodriguez; Ibraheem Salman Shaikh; Mariana Sanchez Montano; Hollie Anne Lai; Maruf Adewole; Jake Albrecht; Udunna Anazodo; Hannah Anderson; Syed Muhammed Anwar; Alejandro Aristizabal; Sina Bagheri; Ujjwal Baid; Timothy Bergquist; Austin J. Borja; Evan Calabrese; Verena Chung; Gian-Marco Conte; James Eddy; Ivan Ezhov; Ariana M. Familiar; Keyvan Farahani; Deep Gandhi; Anurag Gottipati; Shuvanjan Haldar; Juan Eugenio Iglesias; Anastasia Janas; Elaine Elaine; Alexandros Karargyris; Hasan Kassem; Neda Khalili; Florian Kofler; Dominic LaBella; Koen Van Leemput; Hongwei B. Li; Nazanin Maleki; Zeke Meier; Bjoern Menze; Ahmed W. Moawad; Sarthak Pati; Marie Piraud; Tina Poussaint; Zachary J. Reitman; Jeffrey D. Rudie; Rachit Saluja; MIcah Sheller; Russell Takeshi Shinohara; Karthik Viswanathan; Chunhao Wang; Benedikt Wiestler; Walter F. Wiggins; Christos Davatzikos; Phillip B. Storm; Miriam Bornhorst; Roger Packer; Trent Hummel; Peter de Blank; Lindsey Hoffman; Mariam Aboian; Ali Nabavizadeh; Jeffrey B. Ware; Benjamin H. Kann; Brian Rood; Adam Resnick; Spyridon Bakas; Arastoo Vossough; Marius George Linguraru
>
> **备注:** Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA)https://melba-journal.org/2025:005
>
> **摘要:** Pediatric central nervous system tumors are the leading cause of cancer-related deaths in children. The five-year survival rate for high-grade glioma in children is less than 20%. The development of new treatments is dependent upon multi-institutional collaborative clinical trials requiring reproducible and accurate centralized response assessment. We present the results of the BraTS-PEDs 2023 challenge, the first Brain Tumor Segmentation (BraTS) challenge focused on pediatric brain tumors. This challenge utilized data acquired from multiple international consortia dedicated to pediatric neuro-oncology and clinical trials. BraTS-PEDs 2023 aimed to evaluate volumetric segmentation algorithms for pediatric brain gliomas from magnetic resonance imaging using standardized quantitative performance evaluation metrics employed across the BraTS 2023 challenges. The top-performing AI approaches for pediatric tumor analysis included ensembles of nnU-Net and Swin UNETR, Auto3DSeg, or nnU-Net with a self-supervised framework. The BraTSPEDs 2023 challenge fostered collaboration between clinicians (neuro-oncologists, neuroradiologists) and AI/imaging scientists, promoting faster data sharing and the development of automated volumetric analysis techniques. These advancements could significantly benefit clinical trials and improve the care of children with brain tumors.
>
---
#### [replaced 104] Tracking by Detection and Query: An Efficient End-to-End Framework for Multi-Object Tracking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.06197v2](http://arxiv.org/pdf/2411.06197v2)**

> **作者:** Shukun Jia; Shiyu Hu; Yichao Cao; Feng Yang; Xin Lu; Xiaobo Lu
>
> **摘要:** Multi-object tracking (MOT) is dominated by two paradigms: tracking-by-detection (TBD) and tracking-by-query (TBQ). While TBD is decoupled and efficient, its fragmented association steps and heuristic matching pipelines often compromise robustness in complex scenarios. TBQ provides stronger semantic modeling through end-to-end learning, but suffers from high training cost and slow inference due to tight coupling between detection and association. To address these challenges, we propose TBDQ-Net, a unified tracking-by-detection-and-query (TBDQ) framework that effectively combines the strengths of both paradigms. Our method efficiently integrates pretrained, high-performance detectors with an MOT-tailored associator. The associator is lightweight and directly fetches information from the inference of detectors, enhancing the overall efficiency of the framework. The associator is also learnable, making it essential for fully end-to-end optimization, ensuring robust tracking capabilities. Specifically, the associator comprises two key modules: basic information interaction (BII) for comprehensive semantic interaction, and content-position alignment (CPA) for semantic and positional consistency. TBDQ-Net's effectiveness is extensively demonstrated on DanceTrack, SportsMOT and MOT20 benchmarks. As a structurally efficient and semantically robust tracking framework, it outperforms the leading TBD method by 6.0 IDF1 points on DanceTrack and achieves at least 37.5% faster inference than prominent TBQ methods.
>
---
#### [replaced 105] PerLDiff: Controllable Street View Synthesis Using Perspective-Layout Diffusion Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.06109v4](http://arxiv.org/pdf/2407.06109v4)**

> **作者:** Jinhua Zhang; Hualian Sheng; Sijia Cai; Bing Deng; Qiao Liang; Wen Li; Ying Fu; Jieping Ye; Shuhang Gu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Controllable generation is considered a potentially vital approach to address the challenge of annotating 3D data, and the precision of such controllable generation becomes particularly imperative in the context of data production for autonomous driving. Existing methods focus on the integration of diverse generative information into controlling inputs, utilizing frameworks such as GLIGEN or ControlNet, to produce commendable outcomes in controllable generation. However, such approaches intrinsically restrict generation performance to the learning capacities of predefined network architectures. In this paper, we explore the innovative integration of controlling information and introduce PerLDiff (\textbf{Per}spective-\textbf{L}ayout \textbf{Diff}usion Models), a novel method for effective street view image generation that fully leverages perspective 3D geometric information. Our PerLDiff employs 3D geometric priors to guide the generation of street view images with precise object-level control within the network learning process, resulting in a more robust and controllable output. Moreover, it demonstrates superior controllability compared to alternative layout control methods. Empirical results justify that our PerLDiff markedly enhances the precision of controllable generation on the NuScenes and KITTI datasets.
>
---
#### [replaced 106] Sculpting Memory: Multi-Concept Forgetting in Diffusion Models via Dynamic Mask and Concept-Aware Optimization
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.09039v2](http://arxiv.org/pdf/2504.09039v2)**

> **作者:** Gen Li; Yang Xiao; Jie Ji; Kaiyuan Deng; Bo Hui; Linke Guo; Xiaolong Ma
>
> **备注:** ICCV2025(Accept)
>
> **摘要:** Text-to-image (T2I) diffusion models have achieved remarkable success in generating high-quality images from textual prompts. However, their ability to store vast amounts of knowledge raises concerns in scenarios where selective forgetting is necessary, such as removing copyrighted content, reducing biases, or eliminating harmful concepts. While existing unlearning methods can remove certain concepts, they struggle with multi-concept forgetting due to instability, residual knowledge persistence, and generation quality degradation. To address these challenges, we propose \textbf{Dynamic Mask coupled with Concept-Aware Loss}, a novel unlearning framework designed for multi-concept forgetting in diffusion models. Our \textbf{Dynamic Mask} mechanism adaptively updates gradient masks based on current optimization states, allowing selective weight modifications that prevent interference with unrelated knowledge. Additionally, our \textbf{Concept-Aware Loss} explicitly guides the unlearning process by enforcing semantic consistency through superclass alignment, while a regularization loss based on knowledge distillation ensures that previously unlearned concepts remain forgotten during sequential unlearning. We conduct extensive experiments to evaluate our approach. Results demonstrate that our method outperforms existing unlearning techniques in forgetting effectiveness, output fidelity, and semantic coherence, particularly in multi-concept scenarios. Our work provides a principled and flexible framework for stable and high-fidelity unlearning in generative models. The code will be released publicly.
>
---
#### [replaced 107] Dense Feature Interaction Network for Image Inpainting Localization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.02191v2](http://arxiv.org/pdf/2408.02191v2)**

> **作者:** Ye Yao; Tingfeng Han; Shan Jia; Siwei Lyu
>
> **摘要:** Image inpainting, the process of filling in missing areas in an image, is a common image editing technique. Inpainting can be used to conceal or alter image contents in malicious manipulation of images, driving the need for research in image inpainting detection. Most existing methods use a basic encoder-decoder structure, which often results in a high number of false positives or misses the inpainted regions, especially when dealing with targets of varying semantics and scales. Additionally, the lack of an effective approach to capture boundary artifacts leads to less accurate edge localization. In this paper, we describe a new method for inpainting detection based on a Dense Feature Interaction Network (DeFI-Net). DeFI-Net uses a novel feature pyramid architecture to capture and amplify multi-scale representations across various stages, thereby improving the detection of image inpainting by better strengthening feature-level interactions. Additionally, the network can adaptively direct the lower-level features, which carry edge and shape information, to refine the localization of manipulated regions while integrating the higher-level semantic features. Using DeFI-Net, we develop a method combining complementary representations to accurately identify inpainted areas. Evaluation on seven image inpainting datasets demonstrates the effectiveness of our approach, which achieves state-of-the-art performance in detecting inpainting across diverse models. Code and models are available at https://github.com/Boombb/DeFI-Net_Inpainting.
>
---
#### [replaced 108] MedSegNet10: A Publicly Accessible Network Repository for Split Federated Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.20830v2](http://arxiv.org/pdf/2503.20830v2)**

> **作者:** Chamani Shiranthika; Zahra Hafezi Kafshgari; Hadi Hadizadeh; Parvaneh Saeedi
>
> **备注:** 20 pages, 14 figures
>
> **摘要:** Machine Learning (ML) and Deep Learning (DL) have shown significant promise in healthcare, particularly in medical image segmentation, which is crucial for accurate disease diagnosis and treatment planning. Despite their potential, challenges such as data privacy concerns, limited annotated data, and inadequate training data persist. Decentralized learning approaches such as federated learning (FL), split learning (SL), and split federated learning (SplitFed/SFL) address these issues effectively. This paper introduces "MedSegNet10," a publicly accessible repository designed for medical image segmentation using split-federated learning. MedSegNet10 provides a collection of pre-trained neural network architectures optimized for various medical image types, including microscopic images of human blastocysts, dermatoscopic images of skin lesions, and endoscopic images of lesions, polyps, and ulcers, with applications extending beyond these examples. By leveraging SplitFed's benefits, MedSegNet10 allows collaborative training on privately stored, horizontally split data, ensuring privacy and integrity. This repository supports researchers, practitioners, trainees, and data scientists, aiming to advance medical image segmentation while maintaining patient data privacy. The repository is available at: https://vault.sfu.ca/index.php/s/ryhf6t12O0sobuX (password upon request to the authors).
>
---
#### [replaced 109] RecConv: Efficient Recursive Convolutions for Multi-Frequency Representations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.19628v2](http://arxiv.org/pdf/2412.19628v2)**

> **作者:** Mingshu Zhao; Yi Luo; Yong Ouyang
>
> **备注:** Tech report; Added supplementary material;
>
> **摘要:** Recent advances in vision transformers (ViTs) have demonstrated the advantage of global modeling capabilities, prompting widespread integration of large-kernel convolutions for enlarging the effective receptive field (ERF). However, the quadratic scaling of parameter count and computational complexity (FLOPs) with respect to kernel size poses significant efficiency and optimization challenges. This paper introduces RecConv, a recursive decomposition strategy that efficiently constructs multi-frequency representations using small-kernel convolutions. RecConv establishes a linear relationship between parameter growth and decomposing levels which determines the effective receptive field $k\times 2^\ell$ for a base kernel $k$ and $\ell$ levels of decomposition, while maintaining constant FLOPs regardless of the ERF expansion. Specifically, RecConv achieves a parameter expansion of only $\ell+2$ times and a maximum FLOPs increase of $5/3$ times, compared to the exponential growth ($4^\ell$) of standard and depthwise convolutions. RecNeXt-M3 outperforms RepViT-M1.1 by 1.9 $AP^{box}$ on COCO with similar FLOPs. This innovation provides a promising avenue towards designing efficient and compact networks across various modalities. Codes and models can be found at https://github.com/suous/RecNeXt.
>
---
#### [replaced 110] Simple-RF: Regularizing Sparse Input Radiance Fields with Simpler Solutions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2404.19015v4](http://arxiv.org/pdf/2404.19015v4)**

> **作者:** Nagabhushan Somraj; Sai Harsha Mupparaju; Adithyan Karanayil; Rajiv Soundararajan
>
> **备注:** The source code for our model can be found on our project page: https://nagabhushansn95.github.io/publications/2024/Simple-RF.html. Extension of arXiv:2309.03955
>
> **摘要:** Neural Radiance Fields (NeRF) show impressive performance in photo-realistic free-view rendering of scenes. Recent improvements on the NeRF such as TensoRF and ZipNeRF employ explicit models for faster optimization and rendering, as compared to the NeRF that employs an implicit representation. However, both implicit and explicit radiance fields require dense sampling of images in the given scene. Their performance degrades significantly when only a sparse set of views is available. Researchers find that supervising the depth estimated by a radiance field helps train it effectively with fewer views. The depth supervision is obtained either using classical approaches or neural networks pre-trained on a large dataset. While the former may provide only sparse supervision, the latter may suffer from generalization issues. As opposed to the earlier approaches, we seek to learn the depth supervision by designing augmented models and training them along with the main radiance field. Further, we aim to design a framework of regularizations that can work across different implicit and explicit radiance fields. We observe that certain features of these radiance field models overfit to the observed images in the sparse-input scenario. Our key finding is that reducing the capability of the radiance fields with respect to positional encoding, the number of decomposed tensor components or the size of the hash table, constrains the model to learn simpler solutions, which estimate better depth in certain regions. By designing augmented models based on such reduced capabilities, we obtain better depth supervision for the main radiance field. We achieve state-of-the-art view-synthesis performance with sparse input views on popular datasets containing forward-facing and 360$^\circ$ scenes by employing the above regularizations.
>
---
#### [replaced 111] Scaling Laws for Black box Adversarial Attacks
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.16782v3](http://arxiv.org/pdf/2411.16782v3)**

> **作者:** Chuan Liu; Huanran Chen; Yichi Zhang; Yinpeng Dong; Jun Zhu
>
> **摘要:** Adversarial examples usually exhibit good cross-model transferability, enabling attacks on black-box models with limited information about their architectures and parameters, which are highly threatening in commercial black-box scenarios. Model ensembling is an effective strategy to improve the transferability of adversarial examples by attacking multiple surrogate models. However, since prior studies usually adopt few models in the ensemble, there remains an open question of whether scaling the number of models can further improve black-box attacks. Inspired by the scaling law of large foundation models, we investigate the scaling laws of black-box adversarial attacks in this work. Through theoretical analysis and empirical evaluations, we conclude with clear scaling laws that using more surrogate models enhances adversarial transferability. Comprehensive experiments verify the claims on standard image classifiers, diverse defended models and multimodal large language models using various adversarial attack methods. Specifically, by scaling law, we achieve 90%+ transfer attack success rate on even proprietary models like GPT-4o. Further visualization indicates that there is also a scaling law on the interpretability and semantics of adversarial perturbations.
>
---
#### [replaced 112] High-Precision Dichotomous Image Segmentation via Probing Diffusion Capacity
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.10105v3](http://arxiv.org/pdf/2410.10105v3)**

> **作者:** Qian Yu; Peng-Tao Jiang; Hao Zhang; Jinwei Chen; Bo Li; Lihe Zhang; Huchuan Lu
>
> **备注:** Published as a conference paper at ICLR 2025
>
> **摘要:** In the realm of high-resolution (HR), fine-grained image segmentation, the primary challenge is balancing broad contextual awareness with the precision required for detailed object delineation, capturing intricate details and the finest edges of objects. Diffusion models, trained on vast datasets comprising billions of image-text pairs, such as SD V2.1, have revolutionized text-to-image synthesis by delivering exceptional quality, fine detail resolution, and strong contextual awareness, making them an attractive solution for high-resolution image segmentation. To this end, we propose DiffDIS, a diffusion-driven segmentation model that taps into the potential of the pre-trained U-Net within diffusion models, specifically designed for high-resolution, fine-grained object segmentation. By leveraging the robust generalization capabilities and rich, versatile image representation prior of the SD models, coupled with a task-specific stable one-step denoising approach, we significantly reduce the inference time while preserving high-fidelity, detailed generation. Additionally, we introduce an auxiliary edge generation task to not only enhance the preservation of fine details of the object boundaries, but reconcile the probabilistic nature of diffusion with the deterministic demands of segmentation. With these refined strategies in place, DiffDIS serves as a rapid object mask generation model, specifically optimized for generating detailed binary maps at high resolutions, while demonstrating impressive accuracy and swift processing. Experiments on the DIS5K dataset demonstrate the superiority of DiffDIS, achieving state-of-the-art results through a streamlined inference process. The source code will be publicly available at https://github.com/qianyu-dlut/DiffDIS.
>
---
#### [replaced 113] Towards Cross-modal Backward-compatible Representation Learning for Vision-Language Models
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.14715v2](http://arxiv.org/pdf/2405.14715v2)**

> **作者:** Young Kyun Jang; Ser-nam Lim
>
> **摘要:** Modern retrieval systems often struggle with upgrading to new and more powerful models due to the incompatibility of embeddings between the old and new models. This necessitates a costly process known as backfilling, which involves re-computing the embeddings for a large number of data samples. In vision, Backward-compatible Training (BT) has been proposed to ensure that the new model aligns with the old model's embeddings. This paper extends the concept of vision-only BT to the field of cross-modal retrieval, marking the first attempt to address Cross-modal BT (XBT). Our goal is to achieve backward-compatibility between Vision-Language Pretraining (VLP) models, such as CLIP, for the cross-modal retrieval task. To address XBT challenges, we propose an efficient solution: a projection module that maps the new model's embeddings to those of the old model. This module, pretrained solely with text data, significantly reduces the number of image-text pairs required for XBT learning, and, once it is pretrained, it avoids using the old model during training. Furthermore, we utilize parameter-efficient training strategies that improve efficiency and preserve the off-the-shelf new model's knowledge by avoiding any modifications. Experimental results on cross-modal retrieval datasets demonstrate the effectiveness of XBT and its potential to enable backfill-free upgrades when a new VLP model emerges.
>
---
#### [replaced 114] Consistency Trajectory Matching for One-Step Generative Super-Resolution
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.20349v3](http://arxiv.org/pdf/2503.20349v3)**

> **作者:** Weiyi You; Mingyang Zhang; Leheng Zhang; Xingyu Zhou; Kexuan Shi; Shuhang Gu
>
> **摘要:** Current diffusion-based super-resolution (SR) approaches achieve commendable performance at the cost of high inference overhead. Therefore, distillation techniques are utilized to accelerate the multi-step teacher model into one-step student model. Nevertheless, these methods significantly raise training costs and constrain the performance of the student model by the teacher model. To overcome these tough challenges, we propose Consistency Trajectory Matching for Super-Resolution (CTMSR), a distillation-free strategy that is able to generate photo-realistic SR results in one step. Concretely, we first formulate a Probability Flow Ordinary Differential Equation (PF-ODE) trajectory to establish a deterministic mapping from low-resolution (LR) images with noise to high-resolution (HR) images. Then we apply the Consistency Training (CT) strategy to directly learn the mapping in one step, eliminating the necessity of pre-trained diffusion model. To further enhance the performance and better leverage the ground-truth during the training process, we aim to align the distribution of SR results more closely with that of the natural images. To this end, we propose to minimize the discrepancy between their respective PF-ODE trajectories from the LR image distribution by our meticulously designed Distribution Trajectory Matching (DTM) loss, resulting in improved realism of our recovered HR images. Comprehensive experimental results demonstrate that the proposed methods can attain comparable or even superior capabilities on both synthetic and real datasets while maintaining minimal inference latency.
>
---
#### [replaced 115] BST: Badminton Stroke-type Transformer for Skeleton-based Action Recognition in Racket Sports
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.21085v2](http://arxiv.org/pdf/2502.21085v2)**

> **作者:** Jing-Yuan Chang
>
> **备注:** 9 pages (excluding references)
>
> **摘要:** Badminton, known for having the fastest ball speeds among all sports, presents significant challenges to the field of computer vision, including player identification, court line detection, shuttlecock trajectory tracking, and player stroke-type classification. In this paper, we introduce a novel video segmentation strategy to extract frames of each player's racket swing in a badminton broadcast match. These segmented frames are then processed by two existing models: one for Human Pose Estimation to obtain player skeletal joints, and the other for shuttlecock trajectory detection to extract shuttlecock trajectories. Leveraging these joints, trajectories, and player positions as inputs, we propose Badminton Stroke-type Transformer (BST) to classify player stroke-types in singles. To the best of our knowledge, experimental results demonstrate that our method outperforms the previous state-of-the-art on the largest publicly available badminton video dataset, ShuttleSet, which shows that effectively leveraging ball trajectory is likely to be a trend for racket sports action recognition.
>
---
#### [replaced 116] Pixel super-resolved virtual staining of label-free tissue using diffusion models
- **分类: eess.IV; cs.CV; cs.LG; physics.med-ph; physics.optics**

- **链接: [http://arxiv.org/pdf/2410.20073v2](http://arxiv.org/pdf/2410.20073v2)**

> **作者:** Yijie Zhang; Luzhe Huang; Nir Pillar; Yuzhu Li; Hanlong Chen; Aydogan Ozcan
>
> **备注:** 39 Pages, 7 Figures
>
> **摘要:** Virtual staining of tissue offers a powerful tool for transforming label-free microscopy images of unstained tissue into equivalents of histochemically stained samples. This study presents a diffusion model-based super-resolution virtual staining approach utilizing a Brownian bridge process to enhance both the spatial resolution and fidelity of label-free virtual tissue staining, addressing the limitations of traditional deep learning-based methods. Our approach integrates novel sampling techniques into a diffusion model-based image inference process to significantly reduce the variance in the generated virtually stained images, resulting in more stable and accurate outputs. Blindly applied to lower-resolution auto-fluorescence images of label-free human lung tissue samples, the diffusion-based super-resolution virtual staining model consistently outperformed conventional approaches in resolution, structural similarity and perceptual accuracy, successfully achieving a super-resolution factor of 4-5x, increasing the output space-bandwidth product by 16-25-fold compared to the input label-free microscopy images. Diffusion-based super-resolved virtual tissue staining not only improves resolution and image quality but also enhances the reliability of virtual staining without traditional chemical staining, offering significant potential for clinical diagnostics.
>
---
#### [replaced 117] Fusing Radiomic Features with Deep Representations for Gestational Age Estimation in Fetal Ultrasound Images
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.20407v2](http://arxiv.org/pdf/2506.20407v2)**

> **作者:** Fangyijie Wang; Yuan Liang; Sourav Bhattacharjee; Abey Campbell; Kathleen M. Curran; Guénolé Silvestre
>
> **备注:** Accepted at MICCAI 2025
>
> **摘要:** Accurate gestational age (GA) estimation, ideally through fetal ultrasound measurement, is a crucial aspect of providing excellent antenatal care. However, deriving GA from manual fetal biometric measurements depends on the operator and is time-consuming. Hence, automatic computer-assisted methods are demanded in clinical practice. In this paper, we present a novel feature fusion framework to estimate GA using fetal ultrasound images without any measurement information. We adopt a deep learning model to extract deep representations from ultrasound images. We extract radiomic features to reveal patterns and characteristics of fetal brain growth. To harness the interpretability of radiomics in medical imaging analysis, we estimate GA by fusing radiomic features and deep representations. Our framework estimates GA with a mean absolute error of 8.0 days across three trimesters, outperforming current machine learning-based methods at these gestational ages. Experimental results demonstrate the robustness of our framework across different populations in diverse geographical regions. Our code is publicly available on \href{https://github.com/13204942/RadiomicsImageFusion_FetalUS}.
>
---
#### [replaced 118] Low-Cost Infrared Vision Systems for Improved Safety of Emergency Vehicle Operations Under Low-Visibility Conditions
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.14078v2](http://arxiv.org/pdf/2504.14078v2)**

> **作者:** M-Mahdi Naddaf-Sh; Andrew Lee; Kin Yen; Eemon Amini; Iman Soltani
>
> **摘要:** This study investigates the potential of infrared (IR) camera technology to enhance driver safety for emergency vehicles operating in low-visibility conditions, particularly at night and in dense fog. Such environments significantly increase the risk of collisions, especially for tow trucks and snowplows that must remain operational in challenging conditions. Conventional driver assistance systems often struggle under these conditions due to limited visibility. In contrast, IR cameras, which detect the thermal signatures of obstacles, offer a promising alternative. The evaluation combines controlled laboratory experiments, real-world field tests, and surveys of emergency vehicle operators. In addition to assessing detection performance, the study examines the feasibility of retrofitting existing Department of Transportation (DoT) fleets with cost-effective IR-based driver assistance systems. Results underscore the utility of IR technology in enhancing driver awareness and provide data-driven recommendations for scalable deployment across legacy emergency vehicle fleets.
>
---
#### [replaced 119] Super-Resolution Generative Adversarial Networks based Video Enhancement
- **分类: cs.CV; cs.AI; eess.IV; I.4.3**

- **链接: [http://arxiv.org/pdf/2505.10589v4](http://arxiv.org/pdf/2505.10589v4)**

> **作者:** Kağan Çetin; Hacer Akça; Ömer Nezih Gerek
>
> **备注:** 28 pages, 14 figures, 3 tables
>
> **摘要:** This study introduces an enhanced approach to video super-resolution by extending ordinary Single-Image Super-Resolution (SISR) Super-Resolution Generative Adversarial Network (SRGAN) structure to handle spatio-temporal data. While SRGAN has proven effective for single-image enhancement, its design does not account for the temporal continuity required in video processing. To address this, a modified framework that incorporates 3D Non-Local Blocks is proposed, which is enabling the model to capture relationships across both spatial and temporal dimensions. An experimental training pipeline is developed, based on patch-wise learning and advanced data degradation techniques, to simulate real-world video conditions and learn from both local and global structures and details. This helps the model generalize better and maintain stability across varying video content while maintaining the general structure besides the pixel-wise correctness. Two model variants-one larger and one more lightweight-are presented to explore the trade-offs between performance and efficiency. The results demonstrate improved temporal coherence, sharper textures, and fewer visual artifacts compared to traditional single-image methods. This work contributes to the development of practical, learning-based solutions for video enhancement tasks, with potential applications in streaming, gaming, and digital restoration.
>
---
#### [replaced 120] HMSViT: A Hierarchical Masked Self-Supervised Vision Transformer for Corneal Nerve Segmentation and Diabetic Neuropathy Diagnosis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.19474v2](http://arxiv.org/pdf/2506.19474v2)**

> **作者:** Xin Zhang; Liangxiu Han; Yue Shi; Yanlin Zheng; Uazman Alam; Maryam Ferdousi; Rayaz Malik
>
> **摘要:** Diabetic Peripheral Neuropathy (DPN) affects nearly half of diabetes patients, requiring early detection. Corneal Confocal Microscopy (CCM) enables non-invasive diagnosis, but automated methods suffer from inefficient feature extraction, reliance on handcrafted priors, and data limitations. We propose HMSViT, a novel Hierarchical Masked Self-Supervised Vision Transformer (HMSViT) designed for corneal nerve segmentation and DPN diagnosis. Unlike existing methods, HMSViT employs pooling-based hierarchical and dual attention mechanisms with absolute positional encoding, enabling efficient multi-scale feature extraction by capturing fine-grained local details in early layers and integrating global context in deeper layers, all at a lower computational cost. A block-masked self supervised learning framework is designed for the HMSViT that reduces reliance on labelled data, enhancing feature robustness, while a multi-scale decoder is used for segmentation and classification by fusing hierarchical features. Experiments on clinical CCM datasets showed HMSViT achieves state-of-the-art performance, with 61.34% mIoU for nerve segmentation and 70.40% diagnostic accuracy, outperforming leading hierarchical models like the Swin Transformer and HiViT by margins of up to 6.39% in segmentation accuracy while using fewer parameters. Detailed ablation studies further reveal that integrating block-masked SSL with hierarchical multi-scale feature extraction substantially enhances performance compared to conventional supervised training. Overall, these comprehensive experiments confirm that HMSViT delivers excellent, robust, and clinically viable results, demonstrating its potential for scalable deployment in real-world diagnostic applications.
>
---
#### [replaced 121] Open World Object Detection: A Survey
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.11301v2](http://arxiv.org/pdf/2410.11301v2)**

> **作者:** Yiming Li; Yi Wang; Wenqian Wang; Dan Lin; Bingbing Li; Kim-Hui Yap
>
> **备注:** Accepted for publication in IEEE TCSVT
>
> **摘要:** Exploring new knowledge is a fundamental human ability that can be mirrored in the development of deep neural networks, especially in the field of object detection. Open world object detection (OWOD) is an emerging area of research that adapts this principle to explore new knowledge. It focuses on recognizing and learning from objects absent from initial training sets, thereby incrementally expanding its knowledge base when new class labels are introduced. This survey paper offers a thorough review of the OWOD domain, covering essential aspects, including problem definitions, benchmark datasets, source codes, evaluation metrics, and a comparative study of existing methods. Additionally, we investigate related areas like open set recognition (OSR) and incremental learning (IL), underlining their relevance to OWOD. Finally, the paper concludes by addressing the limitations and challenges faced by current OWOD algorithms and proposes directions for future research. To our knowledge, this is the first comprehensive survey of the emerging OWOD field with over one hundred references, marking a significant step forward for object detection technology. A comprehensive source code and benchmarks are archived and concluded at https://github.com/ArminLee/OWOD Review.
>
---
#### [replaced 122] Methodology for an Analysis of Influencing Factors on 3D Object Detection Performance
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.08482v3](http://arxiv.org/pdf/2411.08482v3)**

> **作者:** Anton Kuznietsov; Dirk Schweickard; Steven Peters
>
> **备注:** IEEE International Conference on Autonomous and Trusted Computing (IEEE ATC), 2025
>
> **摘要:** In automated driving, object detection is crucial for perceiving the environment. Although deep learning-based detectors offer high performance, their black-box nature complicates safety assurance. We propose a novel methodology to analyze how object- and environment-related factors affect LiDAR- and camera-based 3D object detectors. A statistical univariate analysis relates each factor to pedestrian detection errors. Additionally, a Random Forest (RF) model predicts errors from meta-information, with Shapley Values interpreting feature importance. By capturing feature dependencies, the RF enables a nuanced analysis of detection errors. Understanding these factors reveals detector performance gaps and supports safer object detection system development.
>
---
#### [replaced 123] SqueezeMe: Mobile-Ready Distillation of Gaussian Full-Body Avatars
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.15171v4](http://arxiv.org/pdf/2412.15171v4)**

> **作者:** Forrest Iandola; Stanislav Pidhorskyi; Igor Santesteban; Divam Gupta; Anuj Pahuja; Nemanja Bartolovic; Frank Yu; Emanuel Garbin; Tomas Simon; Shunsuke Saito
>
> **备注:** Accepted to SIGGRAPH 2025
>
> **摘要:** Gaussian-based human avatars have achieved an unprecedented level of visual fidelity. However, existing approaches based on high-capacity neural networks typically require a desktop GPU to achieve real-time performance for a single avatar, and it remains non-trivial to animate and render such avatars on mobile devices including a standalone VR headset due to substantially limited memory and computational bandwidth. In this paper, we present SqueezeMe, a simple and highly effective framework to convert high-fidelity 3D Gaussian full-body avatars into a lightweight representation that supports both animation and rendering with mobile-grade compute. Our key observation is that the decoding of pose-dependent Gaussian attributes from a neural network creates non-negligible memory and computational overhead. Inspired by blendshapes and linear pose correctives widely used in Computer Graphics, we address this by distilling the pose correctives learned with neural networks into linear layers. Moreover, we further reduce the parameters by sharing the correctives among nearby Gaussians. Combining them with a custom splatting pipeline based on Vulkan, we achieve, for the first time, simultaneous animation and rendering of 3 Gaussian avatars in real-time (72 FPS) on a Meta Quest 3 VR headset. Demo videos are available at https://forresti.github.io/squeezeme.
>
---
#### [replaced 124] Advancing Facial Stylization through Semantic Preservation Constraint and Pseudo-Paired Supervision
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.22022v2](http://arxiv.org/pdf/2506.22022v2)**

> **作者:** Zhanyi Lu; Yue Zhou
>
> **摘要:** Facial stylization aims to transform facial images into appealing, high-quality stylized portraits, with the critical challenge of accurately learning the target style while maintaining content consistency with the original image. Although previous StyleGAN-based methods have made significant advancements, the generated results still suffer from artifacts or insufficient fidelity to the source image. We argue that these issues stem from neglecting semantic shift of the generator during stylization. Therefore, we propose a facial stylization method that integrates semantic preservation constraint and pseudo-paired supervision to enhance the content correspondence and improve the stylization effect. Additionally, we develop a methodology for creating multi-level pseudo-paired datasets to implement supervisory constraint. Furthermore, building upon our facial stylization framework, we achieve more flexible multimodal and reference-guided stylization without complex network architecture designs or additional training. Experimental results demonstrate that our approach produces high-fidelity, aesthetically pleasing facial style transfer that surpasses previous methods.
>
---
#### [replaced 125] RefVSR++: Exploiting Reference Inputs for Reference-based Video Super-resolution
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2307.02897v2](http://arxiv.org/pdf/2307.02897v2)**

> **作者:** Han Zou; Masanori Suganuma; Takayuki Okatani
>
> **摘要:** Smartphones with multi-camera systems, featuring cameras with varying field-of-views (FoVs), are increasingly common. This variation in FoVs results in content differences across videos, paving the way for an innovative approach to video super-resolution (VSR). This method enhances the VSR performance of lower resolution (LR) videos by leveraging higher resolution reference (Ref) videos. Previous works, which operate on this principle, generally expand on traditional VSR models by combining LR and Ref inputs over time into a unified stream. However, we can expect that better results are obtained by independently aggregating these Ref image sequences temporally. Therefore, we introduce an improved method, RefVSR++, which performs the parallel aggregation of LR and Ref images in the temporal direction, aiming to optimize the use of the available data. RefVSR++ also incorporates improved mechanisms for aligning image features over time, crucial for effective VSR. Our experiments demonstrate that RefVSR++ outperforms previous works by over 1dB in PSNR, setting a new benchmark in the field.
>
---
#### [replaced 126] Object Retrieval for Visual Question Answering with Outside Knowledge
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.10798v2](http://arxiv.org/pdf/2403.10798v2)**

> **作者:** Shichao Kan; Yuhai Deng; Jiale Fu; Lihui Cen; Zhe Qu; Linna Zhang; Yixiong Liang; Yigang Cen
>
> **摘要:** Retrieval-augmented generation (RAG) with large language models (LLMs) plays a crucial role in question answering, as LLMs possess limited knowledge and are not updated with continuously growing information. Most recent work on RAG has focused primarily on text-based or large-image retrieval, which constrains the broader application of RAG models. We recognize that object-level retrieval is essential for addressing questions that extend beyond image content. To tackle this issue, we propose a task of object retrieval for visual question answering with outside knowledge (OR-OK-VQA), aimed to extend image-based content understanding in conjunction with LLMs. A key challenge in this task is retrieving diverse objects-related images that contribute to answering the questions. To enable accurate and robust general object retrieval, it is necessary to learn embeddings for local objects. This paper introduces a novel unsupervised deep feature embedding technique called multi-scale group collaborative embedding learning (MS-GCEL), developed to learn embeddings for long-tailed objects at different scales. Additionally, we establish an OK-VQA evaluation benchmark using images from the BelgaLogos, Visual Genome, and LVIS datasets. Prior to the OK-VQA evaluation, we construct a benchmark of challenges utilizing objects extracted from the COCO 2017 and VOC 2007 datasets to support the training and evaluation of general object retrieval models. Our evaluations on both general object retrieval and OK-VQA demonstrate the effectiveness of the proposed approach. The code and dataset will be publicly released for future research.
>
---
#### [replaced 127] Visual Re-Ranking with Non-Visual Side Information
- **分类: cs.CV; I.4**

- **链接: [http://arxiv.org/pdf/2504.11134v2](http://arxiv.org/pdf/2504.11134v2)**

> **作者:** Gustav Hanning; Gabrielle Flood; Viktor Larsson
>
> **备注:** Accepted at Scandinavian Conference on Image Analysis (SCIA) 2025
>
> **摘要:** The standard approach for visual place recognition is to use global image descriptors to retrieve the most similar database images for a given query image. The results can then be further improved with re-ranking methods that re-order the top scoring images. However, existing methods focus on re-ranking based on the same image descriptors that were used for the initial retrieval, which we argue provides limited additional signal. In this work we propose Generalized Contextual Similarity Aggregation (GCSA), which is a graph neural network-based re-ranking method that, in addition to the visual descriptors, can leverage other types of available side information. This can for example be other sensor data (such as signal strength of nearby WiFi or BlueTooth endpoints) or geometric properties such as camera poses for database images. In many applications this information is already present or can be acquired with low effort. Our architecture leverages the concept of affinity vectors to allow for a shared encoding of the heterogeneous multi-modal input. Two large-scale datasets, covering both outdoor and indoor localization scenarios, are utilized for training and evaluation. In experiments we show significant improvement not only on image retrieval metrics, but also for the downstream visual localization task.
>
---
#### [replaced 128] Towards Vision-Language-Garment Models for Web Knowledge Garment Understanding and Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.05210v2](http://arxiv.org/pdf/2506.05210v2)**

> **作者:** Jan Ackermann; Kiyohiro Nakayama; Guandao Yang; Tong Wu; Gordon Wetzstein
>
> **备注:** Presented at MMFM CVPRW'25, Project Page: https://www.computationalimaging.org/publications/vision-language-garment-models/
>
> **摘要:** Multimodal foundation models have demonstrated strong generalization, yet their ability to transfer knowledge to specialized domains such as garment generation remains underexplored. We introduce VLG, a vision-language-garment model that synthesizes garments from textual descriptions and visual imagery. Our experiments assess VLG's zero-shot generalization, investigating its ability to transfer web-scale reasoning to unseen garment styles and prompts. Preliminary results indicate promising transfer capabilities, highlighting the potential for multimodal foundation models to adapt effectively to specialized domains like fashion design.
>
---
#### [replaced 129] AlignGuard: Scalable Safety Alignment for Text-to-Image Generation
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.10493v2](http://arxiv.org/pdf/2412.10493v2)**

> **作者:** Runtao Liu; I Chieh Chen; Jindong Gu; Jipeng Zhang; Renjie Pi; Qifeng Chen; Philip Torr; Ashkan Khakzar; Fabio Pizzati
>
> **摘要:** Text-to-image (T2I) models are widespread, but their limited safety guardrails expose end users to harmful content and potentially allow for model misuse. Current safety measures are typically limited to text-based filtering or concept removal strategies, able to remove just a few concepts from the model's generative capabilities. In this work, we introduce AlignGuard, a method for safety alignment of T2I models. We enable the application of Direct Preference Optimization (DPO) for safety purposes in T2I models by synthetically generating a dataset of harmful and safe image-text pairs, which we call CoProV2. Using a custom DPO strategy and this dataset, we train safety experts, in the form of low-rank adaptation (LoRA) matrices, able to guide the generation process away from specific safety-related concepts. Then, we merge the experts into a single LoRA using a novel merging strategy for optimal scaling performance. This expert-based approach enables scalability, allowing us to remove 7x more harmful concepts from T2I models compared to baselines. AlignGuard consistently outperforms the state-of-the-art on many benchmarks and establishes new practices for safety alignment in T2I networks. Code and data will be shared at https://safetydpo.github.io/.
>
---
#### [replaced 130] HalluSegBench: Counterfactual Visual Reasoning for Segmentation Hallucination Evaluation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.21546v2](http://arxiv.org/pdf/2506.21546v2)**

> **作者:** Xinzhuo Li; Adheesh Juvekar; Xingyou Liu; Muntasir Wahed; Kiet A. Nguyen; Ismini Lourentzou
>
> **备注:** Project webpage: https://plan-lab.github.io/hallusegbench/
>
> **摘要:** Recent progress in vision-language segmentation has significantly advanced grounded visual understanding. However, these models often exhibit hallucinations by producing segmentation masks for objects not grounded in the image content or by incorrectly labeling irrelevant regions. Existing evaluation protocols for segmentation hallucination primarily focus on label or textual hallucinations without manipulating the visual context, limiting their capacity to diagnose critical failures. In response, we introduce HalluSegBench, the first benchmark specifically designed to evaluate hallucinations in visual grounding through the lens of counterfactual visual reasoning. Our benchmark consists of a novel dataset of 1340 counterfactual instance pairs spanning 281 unique object classes, and a set of newly introduced metrics that quantify hallucination sensitivity under visually coherent scene edits. Experiments on HalluSegBench with state-of-the-art vision-language segmentation models reveal that vision-driven hallucinations are significantly more prevalent than label-driven ones, with models often persisting in false segmentation, highlighting the need for counterfactual reasoning to diagnose grounding fidelity.
>
---
#### [replaced 131] CBAGAN-RRT: Convolutional Block Attention Generative Adversarial Network for Sampling-Based Path Planning
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2305.10442v2](http://arxiv.org/pdf/2305.10442v2)**

> **作者:** Abhinav Sagar; Sai Teja Gilukara
>
> **摘要:** Sampling-based path planning algorithms play an important role in autonomous robotics. However, a common problem among the RRT-based algorithms is that the initial path generated is not optimal, and the convergence is too slow for real-world applications. In this paper, we propose a novel image-based learning algorithm using a Convolutional Block Attention Generative Adversarial Network (CBAGAN-RRT) with a combination of spatial and channel attention and a novel loss function to design the heuristics, find a better optimal path, and improve the convergence of the algorithm, both concerning time and speed. The probability distribution of the paths generated from our GAN model is used to guide the sampling process for the RRT algorithm. We demonstrate that our algorithm outperforms the previous state-of-the-art algorithms using both the image quality generation metrics, like IOU Score, Dice Score, FID score, and path planning metrics like time cost and the number of nodes. Ablation studies show the effectiveness of various components in our network architecture. The advantage of our approach is that we can avoid the complicated preprocessing in the state space, our model can be generalized to complex environments like those containing turns and narrow passages without loss of accuracy, and our model can be easily integrated with other sampling-based path planning algorithms.
>
---
#### [replaced 132] Unveiling and Mitigating Memorization in Text-to-image Diffusion Models through Cross Attention
- **分类: cs.CV; cs.CR**

- **链接: [http://arxiv.org/pdf/2403.11052v2](http://arxiv.org/pdf/2403.11052v2)**

> **作者:** Jie Ren; Yaxin Li; Shenglai Zeng; Han Xu; Lingjuan Lyu; Yue Xing; Jiliang Tang
>
> **摘要:** Recent advancements in text-to-image diffusion models have demonstrated their remarkable capability to generate high-quality images from textual prompts. However, increasing research indicates that these models memorize and replicate images from their training data, raising tremendous concerns about potential copyright infringement and privacy risks. In our study, we provide a novel perspective to understand this memorization phenomenon by examining its relationship with cross-attention mechanisms. We reveal that during memorization, the cross-attention tends to focus disproportionately on the embeddings of specific tokens. The diffusion model is overfitted to these token embeddings, memorizing corresponding training images. To elucidate this phenomenon, we further identify and discuss various intrinsic findings of cross-attention that contribute to memorization. Building on these insights, we introduce an innovative approach to detect and mitigate memorization in diffusion models. The advantage of our proposed method is that it will not compromise the speed of either the training or the inference processes in these models while preserving the quality of generated images. Our code is available at https://github.com/renjie3/MemAttn .
>
---
#### [replaced 133] DeOcc-1-to-3: 3D De-Occlusion from a Single Image via Self-Supervised Multi-View Diffusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21544v2](http://arxiv.org/pdf/2506.21544v2)**

> **作者:** Yansong Qu; Shaohui Dai; Xinyang Li; Yuze Wang; You Shen; Liujuan Cao; Rongrong Ji
>
> **备注:** Project page: \url{https://quyans.github.io/DeOcc123/}
>
> **摘要:** Reconstructing 3D objects from a single image remains challenging, especially under real-world occlusions. While recent diffusion-based view synthesis models can generate consistent novel views from a single RGB image, they typically assume fully visible inputs and fail when parts of the object are occluded, resulting in degraded 3D reconstruction quality. We propose DeOcc-1-to-3, an end-to-end framework for occlusion-aware multi-view generation that synthesizes six structurally consistent novel views directly from a single occluded image, enabling reliable 3D reconstruction without prior inpainting or manual annotations. Our self-supervised training pipeline leverages occluded-unoccluded image pairs and pseudo-ground-truth views to teach the model structure-aware completion and view consistency. Without modifying the original architecture, we fully fine-tune the view synthesis model to jointly learn completion and multi-view generation. Additionally, we introduce the first benchmark for occlusion-aware reconstruction, covering diverse occlusion levels, object categories, and masking patterns, providing a standardized protocol for future evaluation.
>
---
#### [replaced 134] Visual Encoders for Data-Efficient Imitation Learning in Modern Video Games
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2312.02312v3](http://arxiv.org/pdf/2312.02312v3)**

> **作者:** Lukas Schäfer; Logan Jones; Anssi Kanervisto; Yuhan Cao; Tabish Rashid; Raluca Georgescu; Dave Bignell; Siddhartha Sen; Andrea Treviño Gavito; Sam Devlin
>
> **备注:** Camera-ready paper presented at the Adaptive and Learning Agents Workshop at the AAMAS 2025 conference
>
> **摘要:** Video games have served as useful benchmarks for the decision-making community, but going beyond Atari games towards modern games has been prohibitively expensive for the vast majority of the research community. Prior work in modern video games typically relied on game-specific integration to obtain game features and enable online training, or on existing large datasets. An alternative approach is to train agents using imitation learning to play video games purely from images. However, this setting poses a fundamental question: which visual encoders obtain representations that retain information critical for decision making? To answer this question, we conduct a systematic study of imitation learning with publicly available pre-trained visual encoders compared to the typical task-specific end-to-end training approach in Minecraft, Counter-Strike: Global Offensive, and Minecraft Dungeons. Our results show that end-to-end training can be effective with comparably low-resolution images and only minutes of demonstrations, but significant improvements can be gained by utilising pre-trained encoders such as DINOv2 depending on the game. In addition to enabling effective decision making, we show that pre-trained encoders can make decision-making research in video games more accessible by significantly reducing the cost of training.
>
---
#### [replaced 135] Neurons: Emulating the Human Visual Cortex Improves Fidelity and Interpretability in fMRI-to-Video Reconstruction
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.11167v2](http://arxiv.org/pdf/2503.11167v2)**

> **作者:** Haonan Wang; Qixiang Zhang; Lehan Wang; Xuanqi Huang; Xiaomeng Li
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Decoding visual stimuli from neural activity is essential for understanding the human brain. While fMRI methods have successfully reconstructed static images, fMRI-to-video reconstruction faces challenges due to the need for capturing spatiotemporal dynamics like motion and scene transitions. Recent approaches have improved semantic and perceptual alignment but struggle to integrate coarse fMRI data with detailed visual features. Inspired by the hierarchical organization of the visual system, we propose NEURONS, a novel framework that decouples learning into four correlated sub-tasks: key object segmentation, concept recognition, scene description, and blurry video reconstruction. This approach simulates the visual cortex's functional specialization, allowing the model to capture diverse video content. In the inference stage, NEURONS generates robust conditioning signals for a pre-trained text-to-video diffusion model to reconstruct the videos. Extensive experiments demonstrate that NEURONS outperforms state-of-the-art baselines, achieving solid improvements in video consistency (26.6%) and semantic-level accuracy (19.1%). Notably, NEURONS shows a strong functional correlation with the visual cortex, highlighting its potential for brain-computer interfaces and clinical applications. Code and model weights are available at: https://github.com/xmed-lab/NEURONS.
>
---
#### [replaced 136] Harnessing Shared Relations via Multimodal Mixup Contrastive Learning for Multimodal Classification
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.17777v4](http://arxiv.org/pdf/2409.17777v4)**

> **作者:** Raja Kumar; Raghav Singhal; Pranamya Kulkarni; Deval Mehta; Kshitij Jadhav
>
> **备注:** Transactions on Machine Learning Research (TMLR). Raja Kumar and Raghav Singhal contributed equally to this work
>
> **摘要:** Deep multimodal learning has shown remarkable success by leveraging contrastive learning to capture explicit one-to-one relations across modalities. However, real-world data often exhibits shared relations beyond simple pairwise associations. We propose M3CoL, a Multimodal Mixup Contrastive Learning approach to capture nuanced shared relations inherent in multimodal data. Our key contribution is a Mixup-based contrastive loss that learns robust representations by aligning mixed samples from one modality with their corresponding samples from other modalities thereby capturing shared relations between them. For multimodal classification tasks, we introduce a framework that integrates a fusion module with unimodal prediction modules for auxiliary supervision during training, complemented by our proposed Mixup-based contrastive loss. Through extensive experiments on diverse datasets (N24News, ROSMAP, BRCA, and Food-101), we demonstrate that M3CoL effectively captures shared multimodal relations and generalizes across domains. It outperforms state-of-the-art methods on N24News, ROSMAP, and BRCA, while achieving comparable performance on Food-101. Our work highlights the significance of learning shared relations for robust multimodal learning, opening up promising avenues for future research. Our code is publicly available at https://github.com/RaghavSinghal10/M3CoL.
>
---
#### [replaced 137] Can Robots "Taste" Grapes? Estimating SSC with Simple RGB Sensors
- **分类: cs.CV; cs.RO; J.3; I.5.1; I.2.9; I.2.10**

- **链接: [http://arxiv.org/pdf/2412.20521v2](http://arxiv.org/pdf/2412.20521v2)**

> **作者:** Thomas Alessandro Ciarfuglia; Ionut Marian Motoi; Leonardo Saraceni; Daniele Nardi
>
> **摘要:** In table grape cultivation, harvesting depends on accurately assessing fruit quality. While some characteristics, like color, are visible, others, such as Soluble Solid Content (SSC), or sugar content measured in degrees Brix ({\deg}Brix), require specific tools. SSC is a key quality factor that correlates with ripeness, but lacks a direct causal relationship with color. Hyperspectral cameras can estimate SSC with high accuracy under controlled laboratory conditions, but their practicality in field environments is limited. This study investigates the potential of simple RGB sensors under uncontrolled lighting to estimate SSC and color, enabling cost-effective, robot-assisted harvesting. Over the 2021 and 2022 summer seasons, we collected grape images with corresponding SSC and color labels to evaluate algorithmic solutions for SSC estimation, specifically testing for cross-seasonal and cross-device robustness. We propose two approaches: a computationally efficient histogram-based method for resource-constrained robots and a Deep Neural Network (DNN) model for more complex applications. Our results demonstrate high performance, with the DNN model achieving a Mean Absolute Error (MAE) as low as $1.05$ {\deg}Brix on a challenging cross-device test set. The lightweight histogram-based method also proved effective, reaching an MAE of $1.46$ {\deg}Brix. These results are highly competitive with those from hyperspectral systems, which report errors in the $1.27$--$2.20$ {\deg}Brix range in similar field applications.
>
---
#### [replaced 138] ThinkVideo: High-Quality Reasoning Video Segmentation with Chain of Thoughts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18561v2](http://arxiv.org/pdf/2505.18561v2)**

> **作者:** Shiu-hong Kao; Yu-Wing Tai; Chi-Keung Tang
>
> **备注:** Project page: https://danielshkao.github.io/thinkvideo.html
>
> **摘要:** Reasoning Video Object Segmentation is a challenging task, which generates a mask sequence from an input video and an implicit, complex text query. Existing works probe into the problem by finetuning Multimodal Large Language Models (MLLM) for segmentation-based output, while still falling short in difficult cases on videos given temporally-sensitive queries, primarily due to the failure to integrate temporal and spatial information. In this paper, we propose ThinkVideo, a novel framework which leverages the zero-shot Chain-of-Thought (CoT) capability of MLLM to address these challenges. Specifically, ThinkVideo utilizes the CoT prompts to extract object selectivities associated with particular keyframes, then bridging the reasoning image segmentation model and SAM2 video processor to output mask sequences. The ThinkVideo framework is training-free and compatible with closed-source MLLMs, which can be applied to Reasoning Video Instance Segmentation. We further extend the framework for online video streams, where the CoT is used to update the object of interest when a better target starts to emerge and becomes visible. We conduct extensive experiments on video object segmentation with explicit and implicit queries. The results show that ThinkVideo significantly outperforms previous works in both cases, qualitatively and quantitatively.
>
---
#### [replaced 139] Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.19326v2](http://arxiv.org/pdf/2412.19326v2)**

> **作者:** Ziang Yan; Zhilin Li; Yinan He; Chenting Wang; Kunchang Li; Xinhao Li; Xiangyu Zeng; Zilei Wang; Yali Wang; Yu Qiao; Limin Wang; Yi Wang
>
> **备注:** CVPR2025
>
> **摘要:** Current multimodal large language models (MLLMs) struggle with fine-grained or precise understanding of visuals although they give comprehensive perception and reasoning in a spectrum of vision applications. Recent studies either develop tool-using or unify specific visual tasks into the autoregressive framework, often at the expense of overall multimodal performance. To address this issue and enhance MLLMs with visual tasks in a scalable fashion, we propose Task Preference Optimization (TPO), a novel method that utilizes differentiable task preferences derived from typical fine-grained visual tasks. TPO introduces learnable task tokens that establish connections between multiple task-specific heads and the MLLM. By leveraging rich visual labels during training, TPO significantly enhances the MLLM's multimodal capabilities and task-specific performance. Through multi-task co-training within TPO, we observe synergistic benefits that elevate individual task performance beyond what is achievable through single-task training methodologies. Our instantiation of this approach with VideoChat and LLaVA demonstrates an overall 14.6% improvement in multimodal performance compared to baseline models. Additionally, MLLM-TPO demonstrates robust zero-shot capabilities across various tasks, performing comparably to state-of-the-art supervised models. The code will be released at https://github.com/OpenGVLab/TPO
>
---
#### [replaced 140] Improving Robustness and Reliability in Medical Image Classification with Latent-Guided Diffusion and Nested-Ensembles
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2310.15952v5](http://arxiv.org/pdf/2310.15952v5)**

> **作者:** Xing Shen; Hengguan Huang; Brennan Nichyporuk; Tal Arbel
>
> **备注:** Accepted to IEEE Transactions on Medical Imaging, 2025
>
> **摘要:** Once deployed, medical image analysis methods are often faced with unexpected image corruptions and noise perturbations. These unknown covariate shifts present significant challenges to deep learning based methods trained on "clean" images. This often results in unreliable predictions and poorly calibrated confidence, hence hindering clinical applicability. While recent methods have been developed to address specific issues such as confidence calibration or adversarial robustness, no single framework effectively tackles all these challenges simultaneously. To bridge this gap, we propose LaDiNE, a novel ensemble learning method combining the robustness of Vision Transformers with diffusion-based generative models for improved reliability in medical image classification. Specifically, transformer encoder blocks are used as hierarchical feature extractors that learn invariant features from images for each ensemble member, resulting in features that are robust to input perturbations. In addition, diffusion models are used as flexible density estimators to estimate member densities conditioned on the invariant features, leading to improved modeling of complex data distributions while retaining properly calibrated confidence. Extensive experiments on tuberculosis chest X-rays and melanoma skin cancer datasets demonstrate that LaDiNE achieves superior performance compared to a wide range of state-of-the-art methods by simultaneously improving prediction accuracy and confidence calibration under unseen noise, adversarial perturbations, and resolution degradation.
>
---
#### [replaced 141] INP-Former++: Advancing Universal Anomaly Detection via Intrinsic Normal Prototypes and Residual Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.03660v2](http://arxiv.org/pdf/2506.03660v2)**

> **作者:** Wei Luo; Haiming Yao; Yunkang Cao; Qiyu Chen; Ang Gao; Weiming Shen; Wenyong Yu
>
> **备注:** 15 pages, 11 figures, 13 tables
>
> **摘要:** Anomaly detection (AD) is essential for industrial inspection and medical diagnosis, yet existing methods typically rely on ``comparing'' test images to normal references from a training set. However, variations in appearance and positioning often complicate the alignment of these references with the test image, limiting detection accuracy. We observe that most anomalies manifest as local variations, meaning that even within anomalous images, valuable normal information remains. We argue that this information is useful and may be more aligned with the anomalies since both the anomalies and the normal information originate from the same image. Therefore, rather than relying on external normality from the training set, we propose INP-Former, a novel method that extracts Intrinsic Normal Prototypes (INPs) directly from the test image. Specifically, we introduce the INP Extractor, which linearly combines normal tokens to represent INPs. We further propose an INP Coherence Loss to ensure INPs can faithfully represent normality for the testing image. These INPs then guide the INP-guided Decoder to reconstruct only normal tokens, with reconstruction errors serving as anomaly scores. Additionally, we propose a Soft Mining Loss to prioritize hard-to-optimize samples during training. INP-Former achieves state-of-the-art performance in single-class, multi-class, and few-shot AD tasks across MVTec-AD, VisA, and Real-IAD, positioning it as a versatile and universal solution for AD. Remarkably, INP-Former also demonstrates some zero-shot AD capability. Furthermore, we propose a soft version of the INP Coherence Loss and enhance INP-Former by incorporating residual learning, leading to the development of INP-Former++. The proposed method significantly improves detection performance across single-class, multi-class, semi-supervised, few-shot, and zero-shot settings.
>
---
#### [replaced 142] ProSAM: Enhancing the Robustness of SAM-based Visual Reference Segmentation with Probabilistic Prompts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.21835v2](http://arxiv.org/pdf/2506.21835v2)**

> **作者:** Xiaoqi Wang; Clint Sebastian; Wenbin He; Liu Ren
>
> **摘要:** The recent advancements in large foundation models have driven the success of open-set image segmentation, a task focused on segmenting objects beyond predefined categories. Among various prompt types (such as points, boxes, texts, and visual references), visual reference segmentation stands out for its unique flexibility and strong zero-shot capabilities. Recently, several SAM-based methods have made notable progress in this task by automatically generating prompts to guide SAM. However, these methods often generate prompts at object boundaries due to suboptimal prompt encoder, which results in instability and reduced robustness. In this work, we introduce ProSAM, a simple but effective method to address the stability challenges we identified in existing SAM-based visual reference segmentation approaches. By learning a variational prompt encoder to predict multivariate prompt distributions, ProSAM avoids generating prompts that lie in unstable regions, overcoming the instability caused by less robust prompts. Our approach consistently surpasses state-of-the-art methods on the Pascal-5$^i$ and COCO-20$^i$ datasets, providing a more robust solution for visual reference segmentation.
>
---
#### [replaced 143] Segment as You Wish -- Free-Form Language-Based Segmentation for Medical Images
- **分类: eess.IV; cs.AI; cs.CV; 68T45, 68U10, 92C55; I.2.7; I.4.9; H.3.3; I.2.6**

- **链接: [http://arxiv.org/pdf/2410.12831v2](http://arxiv.org/pdf/2410.12831v2)**

> **作者:** Longchao Da; Rui Wang; Xiaojian Xu; Parminder Bhatia; Taha Kass-Hout; Hua Wei; Cao Xiao
>
> **备注:** 19 pages, 9 as main content. The paper was accepted to KDD2025
>
> **摘要:** Medical imaging is crucial for diagnosing a patient's health condition, and accurate segmentation of these images is essential for isolating regions of interest to ensure precise diagnosis and treatment planning. Existing methods primarily rely on bounding boxes or point-based prompts, while few have explored text-related prompts, despite clinicians often describing their observations and instructions in natural language. To address this gap, we first propose a RAG-based free-form text prompt generator, that leverages the domain corpus to generate diverse and realistic descriptions. Then, we introduce FLanS, a novel medical image segmentation model that handles various free-form text prompts, including professional anatomy-informed queries, anatomy-agnostic position-driven queries, and anatomy-agnostic size-driven queries. Additionally, our model also incorporates a symmetry-aware canonicalization module to ensure consistent, accurate segmentations across varying scan orientations and reduce confusion between the anatomical position of an organ and its appearance in the scan. FLanS is trained on a large-scale dataset of over 100k medical images from 7 public datasets. Comprehensive experiments demonstrate the model's superior language understanding and segmentation precision, along with a deep comprehension of the relationship between them, outperforming SOTA baselines on both in-domain and out-of-domain datasets.
>
---
#### [replaced 144] Efficient Online Inference of Vision Transformers by Training-Free Tokenization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.15397v3](http://arxiv.org/pdf/2411.15397v3)**

> **作者:** Leonidas Gee; Wing Yan Li; Viktoriia Sharmanska; Novi Quadrianto
>
> **摘要:** The cost of deploying vision transformers increasingly represents a barrier to wider industrial adoption. Existing compression techniques require additional end-to-end fine-tuning or incur a significant drawback to runtime, making them ill-suited for online (real-time) inference, where a prediction is made on any new input as it comes in. We introduce the $\textbf{Visual Word Tokenizer}$ (VWT), a training-free method for reducing power costs while retaining performance and runtime. The VWT groups visual subwords (image patches) that are frequently used into visual words while infrequent ones remain intact. To do so, $\textit{intra}$-image or $\textit{inter}$-image statistics are leveraged to identify similar visual concepts for sequence compression. Experimentally, we demonstrate a reduction in wattage of up to 25% with only a 20% increase in runtime at most. Comparative approaches of 8-bit quantization and token merging achieve a lower or similar power efficiency but exact a higher toll on runtime (up to 100% or more). Our results indicate that VWTs are well-suited for efficient online inference with a marginal compromise on performance.
>
---
#### [replaced 145] StereoDiff: Stereo-Diffusion Synergy for Video Depth Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.20756v2](http://arxiv.org/pdf/2506.20756v2)**

> **作者:** Haodong Li; Chen Wang; Jiahui Lei; Kostas Daniilidis; Lingjie Liu
>
> **备注:** Work done in Nov 2024, during an internship at the University of Pennsylvania. Project page: https://stereodiff.github.io/
>
> **摘要:** Recent video depth estimation methods achieve great performance by following the paradigm of image depth estimation, i.e., typically fine-tuning pre-trained video diffusion models with massive data. However, we argue that video depth estimation is not a naive extension of image depth estimation. The temporal consistency requirements for dynamic and static regions in videos are fundamentally different. Consistent video depth in static regions, typically backgrounds, can be more effectively achieved via stereo matching across all frames, which provides much stronger global 3D cues. While the consistency for dynamic regions still should be learned from large-scale video depth data to ensure smooth transitions, due to the violation of triangulation constraints. Based on these insights, we introduce StereoDiff, a two-stage video depth estimator that synergizes stereo matching for mainly the static areas with video depth diffusion for maintaining consistent depth transitions in dynamic areas. We mathematically demonstrate how stereo matching and video depth diffusion offer complementary strengths through frequency domain analysis, highlighting the effectiveness of their synergy in capturing the advantages of both. Experimental results on zero-shot, real-world, dynamic video depth benchmarks, both indoor and outdoor, demonstrate StereoDiff's SoTA performance, showcasing its superior consistency and accuracy in video depth estimation.
>
---
#### [replaced 146] CHARTOM: A Visual Theory-of-Mind Benchmark for LLMs on Misleading Charts
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2408.14419v3](http://arxiv.org/pdf/2408.14419v3)**

> **作者:** Shubham Bharti; Shiyun Cheng; Jihyun Rho; Jianrui Zhang; Mu Cai; Yong Jae Lee; Martina Rau; Xiaojin Zhu
>
> **摘要:** We introduce CHARTOM, a visual theory-of-mind benchmark designed to evaluate multimodal large language models' capability to understand and reason about misleading data visualizations though charts. CHARTOM consists of carefully designed charts and associated questions that require a language model to not only correctly comprehend the factual content in the chart (the FACT question) but also judge whether the chart will be misleading to a human readers (the MIND question), a dual capability with significant societal benefits. We detail the construction of our benchmark including its calibration on human performance and estimation of MIND ground truth called the Human Misleadingness Index. We evaluated several leading LLMs -- including GPT, Claude, Gemini, Qwen, Llama, and Llava series models -- on the CHARTOM dataset and found that it was challenging to all models both on FACT and MIND questions. This highlights the limitations of current LLMs and presents significant opportunity for future LLMs to improve on understanding misleading charts.
>
---
#### [replaced 147] AirSketch: Generative Motion to Sketch
- **分类: cs.CV; cs.AI; cs.GR**

- **链接: [http://arxiv.org/pdf/2407.08906v3](http://arxiv.org/pdf/2407.08906v3)**

> **作者:** Hui Xian Grace Lim; Xuanming Cui; Yogesh S Rawat; Ser-Nam Lim
>
> **摘要:** Illustration is a fundamental mode of human expression and communication. Certain types of motion that accompany speech can provide this illustrative mode of communication. While Augmented and Virtual Reality technologies (AR/VR) have introduced tools for producing drawings with hand motions (air drawing), they typically require costly hardware and additional digital markers, thereby limiting their accessibility and portability. Furthermore, air drawing demands considerable skill to achieve aesthetic results. To address these challenges, we introduce the concept of AirSketch, aimed at generating faithful and visually coherent sketches directly from hand motions, eliminating the need for complicated headsets or markers. We devise a simple augmentation-based self-supervised training procedure, enabling a controllable image diffusion model to learn to translate from highly noisy hand tracking images to clean, aesthetically pleasing sketches, while preserving the essential visual cues from the original tracking data. We present two air drawing datasets to study this problem. Our findings demonstrate that beyond producing photo-realistic images from precise spatial inputs, controllable image diffusion can effectively produce a refined, clear sketch from a noisy input. Our work serves as an initial step towards marker-less air drawing and reveals distinct applications of controllable diffusion models to AirSketch and AR/VR in general.
>
---
#### [replaced 148] CauSkelNet: Causal Representation Learning for Human Behaviour Analysis
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.15564v4](http://arxiv.org/pdf/2409.15564v4)**

> **作者:** Xingrui Gu; Chuyi Jiang; Erte Wang; Qiang Cui; Leimin Tian; Lianlong Wu; Siyang Song; Chuang Yu
>
> **摘要:** Traditional machine learning methods for movement recognition often struggle with limited model interpretability and a lack of insight into human movement dynamics. This study introduces a novel representation learning framework based on causal inference to address these challenges. Our two-stage approach combines the Peter-Clark (PC) algorithm and Kullback-Leibler (KL) divergence to identify and quantify causal relationships between human joints. By capturing joint interactions, the proposed causal Graph Convolutional Network (GCN) produces interpretable and robust representations. Experimental results on the EmoPain dataset demonstrate that the causal GCN outperforms traditional GCNs in accuracy, F1 score, and recall, particularly in detecting protective behaviors. This work contributes to advancing human motion analysis and lays a foundation for adaptive and intelligent healthcare solutions.
>
---
#### [replaced 149] Grounding Creativity in Physics: A Brief Survey of Physical Priors in AIGC
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.07007v2](http://arxiv.org/pdf/2502.07007v2)**

> **作者:** Siwei Meng; Yawei Luo; Ping Liu
>
> **备注:** Accepted by IJCAI 2025 Survey Track
>
> **摘要:** Recent advancements in AI-generated content have significantly improved the realism of 3D and 4D generation. However, most existing methods prioritize appearance consistency while neglecting underlying physical principles, leading to artifacts such as unrealistic deformations, unstable dynamics, and implausible objects interactions. Incorporating physics priors into generative models has become a crucial research direction to enhance structural integrity and motion realism. This survey provides a review of physics-aware generative methods, systematically analyzing how physical constraints are integrated into 3D and 4D generation. First, we examine recent works in incorporating physical priors into static and dynamic 3D generation, categorizing methods based on representation types, including vision-based, NeRF-based, and Gaussian Splatting-based approaches. Second, we explore emerging techniques in 4D generation, focusing on methods that model temporal dynamics with physical simulations. Finally, we conduct a comparative analysis of major methods, highlighting their strengths, limitations, and suitability for different materials and motion dynamics. By presenting an in-depth analysis of physics-grounded AIGC, this survey aims to bridge the gap between generative models and physical realism, providing insights that inspire future research in physically consistent content generation.
>
---
#### [replaced 150] CalFuse: Feature Calibration Enhanced Parameter Fusion for Class-Continual Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18672v5](http://arxiv.org/pdf/2503.18672v5)**

> **作者:** Juncen Guo; Yang Liu; Xiaoguang Zhu; Lianlong Sun; Liangyu Teng; Jingyi Wu; Di Li; Linxiao Gong; Weiwei Jiang; Wei Zhou; Liang Song
>
> **摘要:** Class-Continual Learning (CCL) enables models to continuously learn new class knowledge while retaining previous classes, facilitating adaptation and evolution in dynamic, real-world environments. Traditional CCL methods primarily rely on visual features, which limits their effectiveness in complex, multimodal scenarios. In contrast, Vision-Language Models (VLMs) show promising potential for enhancing CCL by leveraging pre-trained knowledge and fusing multi-modal semantic cues such as text and vision. However, existing approaches struggle to mitigate catastrophic forgetting while preserving the generalization strengths of VLMs across diverse modalities. To address these challenges, we propose CalFuse, a framework for feature Calibration enhanced parameter Fusion, which enhances dynamic knowledge fusion. CalFuse introduces a dynamic feature calibration mechanism that iteratively adjusts the contribution of original visual features to the final class decision, thereby preserving the model's intrinsic generalization capability across modalities. Simultaneously, a parameter fusion strategy effectively fuses newly acquired knowledge with prior task parameters, maintaining a balance between acquiring new class representations and preserving old knowledge. Experimental results on popular benchmarks (e.g., CIFAR100 and ImageNet100) validate the superiority of the proposed method.
>
---
#### [replaced 151] GLS: Geometry-aware 3D Language Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.18066v2](http://arxiv.org/pdf/2411.18066v2)**

> **作者:** Jiaxiong Qiu; Liu Liu; Xinjie Wang; Tianwei Lin; Wei Sui; Zhizhong Su
>
> **备注:** Technical Report
>
> **摘要:** Recently, 3D Gaussian Splatting (3DGS) has achieved impressive performance on indoor surface reconstruction and 3D open-vocabulary segmentation. This paper presents GLS, a unified framework of 3D surface reconstruction and open-vocabulary segmentation based on 3DGS. GLS extends two fields by improving their sharpness and smoothness. For indoor surface reconstruction, we introduce surface normal prior as a geometric cue to guide the rendered normal, and use the normal error to optimize the rendered depth. For 3D open-vocabulary segmentation, we employ 2D CLIP features to guide instance features and enhance the surface smoothness, then utilize DEVA masks to maintain their view consistency. Extensive experiments demonstrate the effectiveness of jointly optimizing surface reconstruction and 3D open-vocabulary segmentation, where GLS surpasses state-of-the-art approaches of each task on MuSHRoom, ScanNet++ and LERF-OVS datasets. Project webpage: https://jiaxiongq.github.io/GLS_ProjectPage.
>
---
#### [replaced 152] Mono-Modalizing Extremely Heterogeneous Multi-Modal Medical Image Registration
- **分类: cs.CV; I.4.5; I.4.9; J.3**

- **链接: [http://arxiv.org/pdf/2506.15596v2](http://arxiv.org/pdf/2506.15596v2)**

> **作者:** Kyobin Choo; Hyunkyung Han; Jinyeong Kim; Chanyong Yoon; Seong Jae Hwang
>
> **备注:** 11 pages, 3 figures, 2 tables, Accepted at Medical Image Computing and Computer Assisted Intervention (MICCAI) 2025
>
> **摘要:** In clinical practice, imaging modalities with functional characteristics, such as positron emission tomography (PET) and fractional anisotropy (FA), are often aligned with a structural reference (e.g., MRI, CT) for accurate interpretation or group analysis, necessitating multi-modal deformable image registration (DIR). However, due to the extreme heterogeneity of these modalities compared to standard structural scans, conventional unsupervised DIR methods struggle to learn reliable spatial mappings and often distort images. We find that the similarity metrics guiding these models fail to capture alignment between highly disparate modalities. To address this, we propose M2M-Reg (Multi-to-Mono Registration), a novel framework that trains multi-modal DIR models using only mono-modal similarity while preserving the established architectural paradigm for seamless integration into existing models. We also introduce GradCyCon, a regularizer that leverages M2M-Reg's cyclic training scheme to promote diffeomorphism. Furthermore, our framework naturally extends to a semi-supervised setting, integrating pre-aligned and unaligned pairs only, without requiring ground-truth transformations or segmentation masks. Experiments on the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset demonstrate that M2M-Reg achieves up to 2x higher DSC than prior methods for PET-MRI and FA-MRI registration, highlighting its effectiveness in handling highly heterogeneous multi-modal DIR. Our code is available at https://github.com/MICV-yonsei/M2M-Reg.
>
---
#### [replaced 153] HalCECE: A Framework for Explainable Hallucination Detection through Conceptual Counterfactuals in Image Captioning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.00436v2](http://arxiv.org/pdf/2503.00436v2)**

> **作者:** Maria Lymperaiou; Giorgos Filandrianos; Angeliki Dimitriou; Athanasios Voulodimos; Giorgos Stamou
>
> **摘要:** In the dynamic landscape of artificial intelligence, the exploration of hallucinations within vision-language (VL) models emerges as a critical frontier. This work delves into the intricacies of hallucinatory phenomena exhibited by widely used image captioners, unraveling interesting patterns. Specifically, we step upon previously introduced techniques of conceptual counterfactual explanations to address VL hallucinations. The deterministic and efficient nature of the employed conceptual counterfactuals backbone is able to suggest semantically minimal edits driven by hierarchical knowledge, so that the transition from a hallucinated caption to a non-hallucinated one is performed in a black-box manner. HalCECE, our proposed hallucination detection framework is highly interpretable, by providing semantically meaningful edits apart from standalone numbers, while the hierarchical decomposition of hallucinated concepts leads to a thorough hallucination analysis. Another novelty tied to the current work is the investigation of role hallucinations, being one of the first works to involve interconnections between visual concepts in hallucination detection. Overall, HalCECE recommends an explainable direction to the crucial field of VL hallucination detection, thus fostering trustworthy evaluation of current and future VL systems.
>
---
#### [replaced 154] Think Before You Segment: High-Quality Reasoning Segmentation with GPT Chain of Thoughts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07503v4](http://arxiv.org/pdf/2503.07503v4)**

> **作者:** Shiu-hong Kao; Yu-Wing Tai; Chi-Keung Tang
>
> **备注:** Project page: https://danielshkao.github.io/thinkfirst.html
>
> **摘要:** Reasoning segmentation is a challenging vision-language task that aims to output the segmentation mask with respect to a complex, implicit, and even non-visual query text. Previous works incorporated multimodal Large Language Models (MLLMs) with segmentation models to approach the difficult problem. However, their segmentation quality often falls short in complex cases, particularly when dealing with out-of-domain objects with intricate structures, blurry boundaries, occlusions, or high similarity with surroundings. In this paper, we introduce ThinkFirst, a training-free reasoning segmentation framework that leverages GPT's chain of thought to address these challenging cases. Our approach allows GPT-4o or other powerful MLLMs to generate a detailed, chain-of-thought description of an image. This summarized description is then passed to a language-instructed segmentation assistant to aid the segmentation process. Our framework allows users to easily interact with the segmentation agent using multimodal inputs, such as easy text and image scribbles, for successive refinement or communication. We evaluate the performance of ThinkFirst on diverse objects. Extensive experiments show that, this zero-shot-CoT approach significantly improves the vanilla reasoning segmentation agent, both qualitatively and quantitatively, while being less sensitive or critical to user-supplied prompts after Thinking First.
>
---
