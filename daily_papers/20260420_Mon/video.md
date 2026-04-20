# 计算机视觉 cs.CV

- **最新发布 106 篇**

- **更新 66 篇**

## 最新发布

#### [new 001] Seeing the Intangible: Survey of Image Classification into High-Level and Abstract Categories
- **分类: cs.CV; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于图像分类任务，旨在解决高层次抽象概念识别问题。通过系统综述，厘清高阶语义，分析相关任务与挑战，提出融合多源信息的解决方案。**

- **链接: [https://arxiv.org/pdf/2308.10562](https://arxiv.org/pdf/2308.10562)**

> **作者:** Delfina Sol Martinez Pandiani; Valentina Presutti
>
> **备注:** Preprint
>
> **摘要:** The field of Computer Vision (CV) is increasingly shifting towards ``high-level'' visual sensemaking tasks, yet the exact nature of these tasks remains unclear and tacit. This survey paper addresses this ambiguity by systematically reviewing research on high-level visual understanding, focusing particularly on Abstract Concepts (ACs) in automatic image classification. Our survey contributes in three main ways: Firstly, it clarifies the tacit understanding of high-level semantics in CV through a multidisciplinary analysis, and categorization into distinct clusters, including commonsense, emotional, aesthetic, and inductive interpretative semantics. Secondly, it identifies and categorizes computer vision tasks associated with high-level visual sensemaking, offering insights into the diverse research areas within this domain. Lastly, it examines how abstract concepts such as values and ideologies are handled in CV, revealing challenges and opportunities in AC-based image classification. Notably, our survey of AC image classification tasks highlights persistent challenges, such as the limited efficacy of massive datasets and the importance of integrating supplementary information and mid-level features. We emphasize the growing relevance of hybrid AI systems in addressing the multifaceted nature of AC image classification tasks. Overall, this survey enhances our understanding of high-level visual reasoning in CV and lays the groundwork for future research endeavors.
>
---
#### [new 002] SPLIT: Self-supervised Partitioning for Learned Inversion in Nonlinear Tomography
- **分类: cs.CV**

- **简介: 该论文提出SPLIT框架，解决非线性断层成像中的自监督图像重建问题，无需真实图像样本即可实现高质量重建。**

- **链接: [https://arxiv.org/pdf/2604.15651](https://arxiv.org/pdf/2604.15651)**

> **作者:** Markus Haltmeier; Lukas Neumann; Nadja Gruber; Gyeongha Hwang
>
> **摘要:** Machine learning has achieved impressive performance in tomographic reconstruction, but supervised training requires paired measurements and ground-truth images that are often unavailable. This has motivated self-supervised approaches, which have primarily addressed denoising and, more recently, linear inverse problems. We address nonlinear inverse problems and introduce SPLIT (Self-supervised Partitioning for Learned Inversion in Nonlinear Tomography), a self-supervised machine-learning framework for reconstructing images from nonlinear, incomplete, and noisy projection data without any samples of ground-truth images. SPLIT enforces cross-partition consistency and measurement-domain fidelity while exploiting complementary information across multiple partitions. Our main theoretical result shows that, under mild conditions, the proposed self-supervised objective is equivalent to its supervised counterpart in expectation. We regularize training with an automatic stopping rule that halts optimization when a no-reference image-quality surrogate saturates. As a concrete application, we derive SPLIT variants for multispectral computed tomography. Experiments on sparse-view acquisitions demonstrate high reconstruction quality and robustness to noise, surpassing classical iterative reconstruction and recent self-supervised baselines.
>
---
#### [new 003] Mind's Eye: A Benchmark of Visual Abstraction, Transformation and Composition for Multimodal LLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态语言模型评估任务，旨在解决视觉认知与空间推理能力不足的问题。构建了"Mind's Eye"基准，测试MLLMs在抽象、关系和变换任务中的表现。**

- **链接: [https://arxiv.org/pdf/2604.16054](https://arxiv.org/pdf/2604.16054)**

> **作者:** Rohit Sinha; Aditya Kanade; Sai Srinivas Kancheti; Vineeth N Balasubramanian; Tanuja Ganu
>
> **摘要:** Multimodal large language models (MLLMs) have achieved impressive progress on vision language benchmarks, yet their capacity for visual cognitive and visuospatial reasoning remains less understood. We introduce "Mind's Eye", a multiple-choice benchmark of eight visuo-cognitive tasks inspired by classic human intelligence tests and organized under a novel "A-R-T" taxonomy: Abstraction, Relation, and Transformation. The tasks probe core processes of fluid intelligence such as pattern induction, analogical relation mapping, and mental transformation. We evaluate a diverse suite of closed-source and open-source MLLMs and compare their performance with human participants. Humans achieve 80% accuracy, while top performing MLLMs remain below 50%. Error analysis reveals failures in: (i) visual attention allocation, (ii) internal perceptual manipulation, and (iii) weak abstraction of underlying visual concepts. Our findings suggest that current MLLMs exhibit limited visuospatial reasoning capabilities, when compared with human participants, highlighting the need for more cognitively grounded evaluation frameworks.
>
---
#### [new 004] Making Image Editing Easier via Adaptive Task Reformulation with Agentic Executions
- **分类: cs.CV**

- **简介: 该论文属于图像编辑任务，旨在解决因任务表述不当导致的编辑失败问题。通过自适应任务重构框架，提升编辑效果，无需修改模型本身。**

- **链接: [https://arxiv.org/pdf/2604.15917](https://arxiv.org/pdf/2604.15917)**

> **作者:** Bo Zhao; Kairui Guo; Runnan Du; Haiyang Sun; Pengshan Wang; Huan Yang; Kun Gai; Yixin Cao; Wei Ji
>
> **备注:** 9pages
>
> **摘要:** Instruction guided image editing has advanced substantially with recent generative models, yet it still fails to produce reliable results across many seemingly simple cases. We observe that a large portion of these failures stem not from insufficient model capacity, but from poorly formulated editing tasks, such as those involving small targets, implicit spatial relations, or under-specified instructions. In this work, we frame image editing failures as a task formulation problem and propose an adaptive task reformulation framework that improves editing performance without modifying the underlying model. Our key idea is to transform the original image-instruction pair into a sequence of operations that are dynamically determined and executed by a MLLM agent through analysis, routing, reformulation, and feedback-driven refinement. Experiments on multiple benchmarks, including ImgEdit, PICA, and RePlan, across diverse editing backbones such as Qwen Image Edit and Nano Banana, show consistent improvements, with especially large gains on challenging cases. These results suggest that task reformulation is a critical but underexplored factor, and that substantial gains can be achieved by better matching editing tasks to the effective operating regime of existing models.
>
---
#### [new 005] The Amazing Stability of Flow Matching
- **分类: cs.CV**

- **简介: 该论文研究流匹配模型的稳定性，探讨数据集和架构变化对生成样本质量与多样性的影响。实验表明，即使数据集减少50%，模型仍保持稳定，生成结果质量不变。任务为生成模型的鲁棒性分析。**

- **链接: [https://arxiv.org/pdf/2604.16079](https://arxiv.org/pdf/2604.16079)**

> **作者:** Rania Briq; Michael Kamp; Ohad Fried; Sarel Cohen; Stefan Kesselheim
>
> **备注:** EurIPS 2025 Workshop on Principles of Generative Modeling (PriGM)
>
> **摘要:** The success of deep generative models in generating high-quality and diverse samples is often attributed to particular architectures and large training datasets. In this paper, we investigate the impact of these factors on the quality and diversity of samples generated by \emph{flow-matching} models. Surprisingly, in our experiments on CelebA-HQ dataset, flow matching remains stable even when pruning 50\% of the dataset. That is, the quality and diversity of generated samples are preserved. Moreover, pruning impacts the latent representation only slightly, that is, samples generated by models trained on the full and pruned dataset map to visually similar outputs for a given seed. We observe similar stability when changing the architecture or training configuration, such that the latent representation is maintained under these changes as well. Our results quantify just how strong this stability can be in practice, and help explain the reliability of flow-matching models under various perturbations.
>
---
#### [new 006] AHS: Adaptive Head Synthesis via Synthetic Data Augmentations
- **分类: cs.CV**

- **简介: 该论文属于头像交换任务，解决现有方法在多样表情、发型和视角下的泛化能力不足问题。提出AHS方法，通过合成数据增强提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.15857](https://arxiv.org/pdf/2604.15857)**

> **作者:** Taewoong Kang; Hyojin Jang; Sohyun Jeong; Seunggi Moon; Gihwi Kim; Hoon Jin Jung; Jaegul choo
>
> **备注:** CVPR 2026, Project Page : this https URL
>
> **摘要:** Recent digital media advancements have created increasing demands for sophisticated portrait manipulation techniques, particularly head swapping, where one's head is seamlessly integrated with another's body. However, current approaches predominantly rely on face-centered cropped data with limited view angles, significantly restricting their real-world applicability. They struggle with diverse head expressions, varying hairstyles, and natural blending beyond facial regions. To address these limitations, we propose Adaptive Head Synthesis (AHS), which effectively handles full upper-body images with varied head poses and expressions. AHS incorporates a novel head reenacted synthetic data augmentation strategy to overcome self-supervised training constraints, enhancing generalization across diverse facial expressions and orientations without requiring paired training data. Comprehensive experiments demonstrate that AHS achieves superior performance in challenging real-world scenarios, producing visually coherent results that preserve identity and expression fidelity across various head orientations and hairstyles. Notably, AHS shows exceptional robustness in maintaining facial identity while drastic expression changes and faithfully preserving accessories while significant head pose variations.
>
---
#### [new 007] LP$^{2}$DH: A Locality-Preserving Pixel-Difference Hashing Framework for Dynamic Texture Recognition
- **分类: cs.CV**

- **简介: 该论文属于动态纹理识别任务，解决高维特征维度和跨平面相关性丢失的问题。提出LP²DH框架，通过编码像素差值并保持局部结构，提升识别性能。**

- **链接: [https://arxiv.org/pdf/2604.15707](https://arxiv.org/pdf/2604.15707)**

> **作者:** Ruxin Ding; Jianfeng Ren; Heng Yu; Jiawei Li; Xudong Jiang
>
> **摘要:** Spatiotemporal Local Binary Pattern (STLBP) is a widely used dynamic texture descriptor, but it suffers from extremely high dimensionality. To tackle this, STLBP features are often extracted on three orthogonal planes, which sacrifice inter-plane correlation. In this work, we propose a Locality-Preserving Pixel-Difference Hashing (LP$^{2}$DH) framework that jointly encodes pixel differences in the full spatiotemporal neighbourhood. LP$^{2}$DH transforms Pixel-Difference Vectors (PDVs) into compact binary codes with maximal discriminative power. Furthermore, we incorporate a locality-preserving embedding to maintain the PDVs' local structure before and after hashing. Then, a curvilinear search strategy is utilized to jointly optimize the hashing matrix and binary codes via gradient descent on the Stiefel manifold. After hashing, dictionary learning is applied to encode the binary vectors into codewords, and the resulting histogram is utilized as the final feature representation. The proposed LP$^{2}$DH achieves state-of-the-art performance on three major dynamic texture recognition benchmarks: 99.80% against DT-GoogleNet's 98.93% on UCLA, 98.52% against HoGF$^{3D}$'s 97.63% on DynTex++, and 96.19% compared to STS's 95.00% on YUPENN. The source code is available at: this https URL.
>
---
#### [new 008] SWNet: A Cross-Spectral Network for Camouflaged Weed Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SWNet，用于检测伪装杂草的跨光谱网络。解决复杂农田中伪装杂草识别难题，融合可见光与近红外信息，并引入边界细化模块提升分割精度。**

- **链接: [https://arxiv.org/pdf/2604.16147](https://arxiv.org/pdf/2604.16147)**

> **作者:** Henry O. Velesaca; Luigi Miranda; Angel D. Sappa
>
> **摘要:** This paper presents SWNet, a bimodal end-to-end cross-spectral network specifically engineered for the detection of camouflaged weeds in dense agricultural environments. Plant camouflage, characterized by homochromatic blending where invasive species mimic the phenotypic traits of primary crops, poses a significant challenge for traditional computer vision systems. To overcome these limitations, SWNet utilizes a Pyramid Vision Transformer v2 backbone to capture long-range dependencies and a Bimodal Gated Fusion Module to dynamically integrate Visible and Near-Infrared information. By leveraging the physiological differences in chlorophyll reflectance captured in the NIR spectrum, the proposed architecture effectively discriminates targets that are otherwise indistinguishable in the visible range. Furthermore, an Edge-Aware Refinement module is employed to produce sharper object boundaries and reduce structural ambiguity. Experimental results on the Weeds-Banana dataset indicate that SWNet outperforms ten state-of-the-art methods. The study demonstrates that the integration of cross-spectral data and boundary-guided refinement is essential for high segmentation accuracy in complex crop canopies. The code is available on GitHub: this https URL
>
---
#### [new 009] Information Router for Mitigating Modality Dominance in Vision-Language Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多模态学习任务，旨在解决视觉语言模型中的模态主导问题。通过提出MoIR方法，显式融合多模态信息以平衡模态贡献并提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.16264](https://arxiv.org/pdf/2604.16264)**

> **作者:** Seulgi Kim; Mohit Prabhushankar; Ghassan AlRegib
>
> **摘要:** Vision Language models (VLMs) have demonstrated strong performance across a wide range of benchmarks, yet they often suffer from modality dominance, where predictions rely disproportionately on a single modality. Prior approaches primarily address this issue by steering model's attention allocation, implicitly assuming that all modalities provide sufficient information. However, attention only determines where the model focuses, and cannot enrich information that is missing or ambiguous. In the real world, input modalities often differ in information density and their signal-to-noise ratios. In such cases, simply adjusting model's attention does not resolve the underlying lack of information. In this paper, we propose \textsc{MoIR}: \textit{Multi-modal Information Router}, an information-level fusion method that explicitly reduces information disparity prior to fusion. \textsc{MoIR} identifies less informative tokens and routes complementary information from a stronger modality, constructing information-dense token representations before they are processed by a large language model. By modifying information availability, \textsc{MoIR} enables reliable shifts in modality dominance, even when one modality is degraded. We evaluate \textsc{MoIR} on three widely used multi-modal benchmarks across multiple model backbones. Experimental results show that \textsc{MoIR} consistently demonstrates more balanced modality contribution, and improves robustness and downstream performance, particularly even under modality degradation. These findings demonstrate that explicitly modifying cross-modal information is an effective and complementary strategy for mitigating modality dominance in multi-modal reasoning models.
>
---
#### [new 010] Learning to Look before Learning to Like: Incorporating Human Visual Cognition into Aesthetic Quality Assessment
- **分类: cs.CV**

- **简介: 该论文属于图像美学评估任务，旨在解决传统方法忽视人类视觉认知的问题。提出AestheticNet，结合视觉注意力与语义感知，提升美学评估效果。**

- **链接: [https://arxiv.org/pdf/2604.15853](https://arxiv.org/pdf/2604.15853)**

> **作者:** Liwen Yu; Chi Liu; Xiaotong Han; Congcong Zhu; Minghao Wang; Sheng Shen
>
> **备注:** Accepted for Poster Presentation at CogSci 2026
>
> **摘要:** Automated Aesthetic Quality Assessment (AQA) treats images primarily as static pixel vectors, aligning predictions with human-rating scores largely through semantic perception. However, this paradigm diverges from human aesthetic cognition, which arises from dynamic visual exploration shaped by scanning paths, processing fluency, and the interplay between bottom-up salience and top-down intention. We introduce AestheticNet, a novel cognitive-inspired AQA paradigm that integrates human-like visual cognition and semantic perception with a two-pathway architecture. The visual attention pathway, implemented as a gaze-aligned visual encoder (GAVE) pre-trained offline on eye-tracking data using resource-efficient contrast gaze alignment, models attention from human vision system. This pathway augments the semantic pathway, which uses a fixed semantic encoder such as CLIP, through cross-attention fusion. Visual attention provides a cognitive prior reflecting foreground/background structure, color cascade, brightness, and lighting, all of which are determinants of aesthetic perception beyond semantics. Experiments validated by hypothesis testing show a consistent improvement over the semantic-alone baselines, and demonstrate the gaze module as a model-agnostic corrector compatible with diverse AQA backbones, supporting the necessity and modularity of human-like visual cognition for AQA. Our code is available at this https URL.
>
---
#### [new 011] Weak-to-Strong Knowledge Distillation Accelerates Visual Learning
- **分类: cs.CV**

- **简介: 该论文属于视觉学习任务，旨在加速训练过程。通过弱到强的知识蒸馏方法，在早期阶段提升学生模型性能，实现训练速度提升。**

- **链接: [https://arxiv.org/pdf/2604.15451](https://arxiv.org/pdf/2604.15451)**

> **作者:** Baiang Li; Wenhao Chai; Felix Heide
>
> **备注:** 18 pages, 7 figures
>
> **摘要:** Large-scale visual learning is increasingly limited by training cost. Existing knowledge distillation methods transfer from a stronger teacher to a weaker student for compression or final-accuracy improvement. We instead investigate distillation to accelerate the training of strong students. We propose a generalizable plug-and-play recipe that freezes a weaker teacher, applies distillation only in early training, and turns it off once the student reaches and surpasses teacher-level performance. For ImageNet and CIFAR classification, this strategy reaches target thresholds much earlier, with up to 4.8 times speedup measured by epochs. We confirm that the method generalizes to other tasks and report 1.7 times epoch speedup for object detection on the COCO dataset, and 2.5 times earlier target-FID crossing for diffusion generation on the CIFAR-10 dataset, measured in steps. These findings validate our method as a universal speedup mechanism for visual learning.
>
---
#### [new 012] Saturation-Aware Space-Variant Blind Image Deblurring
- **分类: cs.CV**

- **简介: 该论文属于图像去模糊任务，解决高动态范围和低光条件下饱和像素带来的去模糊问题。通过分割图像并估计真实辐射值，提升去模糊效果。**

- **链接: [https://arxiv.org/pdf/2604.16200](https://arxiv.org/pdf/2604.16200)**

> **作者:** Muhammad Z. Alam; Larry Stetsiuk; Arooba Zeshan
>
> **备注:** 12 pages, 12 Figure
>
> **摘要:** This paper presents a novel saturation aware space variant blind image deblurring framework designed to address challenges posed by saturated pixels in deblurring under high dynamic range and low light conditions. The proposed approach effectively segments the image based on blur intensity and proximity to saturation, leveraging a pre estimated Light Spread Function to mitigate stray light effects. By accurately estimating the true radiance of saturated regions using the dark channel prior, our method enhances the deblurring process without introducing artifacts like ringing. Experimental evaluations on both synthetic and real world datasets demonstrate that the framework improves deblurring outcomes across various scenarios showcasing superior performance compared to state of the art saturation-aware and general purpose methods. This adaptability highlights the framework potential integration with existing and emerging blind image deblurring techniques.
>
---
#### [new 013] Towards In-Context Tone Style Transfer with A Large-Scale Triplet Dataset
- **分类: cs.CV**

- **简介: 该论文属于图像风格迁移任务，旨在解决缺乏高质量大规模三元组数据的问题。作者构建了TST100K数据集，并提出ICTone框架，实现更准确的色调风格迁移。**

- **链接: [https://arxiv.org/pdf/2604.16114](https://arxiv.org/pdf/2604.16114)**

> **作者:** Yuhai Deng; Huimin She; Wei Shen; Meng Li; Ruoxi Wu; Lunxi Yuan; Xiang Li
>
> **备注:** 33 pages, 14 figures
>
> **摘要:** Tone style transfer for photo retouching aims to adapt the stylistic tone of the reference image to a given content image. However, the lack of high-quality large-scale triplet datasets with stylized ground truth forces existing methods to rely on self-supervised or proxy objectives, which limits model capability. To mitigate this gap, we design a data construction pipeline to build TST100K, a large-scale dataset of 100,000 content-reference-stylized triplets. At the core of this pipeline, we train a tone style scorer to ensure strict stylistic consistency for each triplet. In addition, existing methods typically extract content and reference features independently and then fuse them in a decoder, which may cause semantic loss and lead to inappropriate color transfer and degraded visual aesthetics. Instead, we propose ICTone, a diffusion-based framework that performs tone transfer in an in-context manner by jointly conditioning on both images, leveraging the semantic priors of generative models for semantic-aware transfer. Reward feedback learning using the tone style scorer is further incorporated to improve stylistic fidelity and visual quality. Experiments demonstrate the effectiveness of TST100K, and ICTone achieves state-of-the-art performance on both quantitative metrics and human evaluations.
>
---
#### [new 014] Sketch and Text Synergy: Fusing Structural Contours and Descriptive Attributes for Fine-Grained Image Retrieval
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于细粒度图像检索任务，解决手绘草图与文本描述之间的模态差异问题。通过融合结构轮廓和描述属性，提出STBIR框架提升检索性能。**

- **链接: [https://arxiv.org/pdf/2604.15735](https://arxiv.org/pdf/2604.15735)**

> **作者:** Siyuan Wang; Hanchen Gao; Guangming Zhu; Jiang Lu; Yiyue Ma; Tianci Wu; Jincai Huang; Liang Zhang
>
> **备注:** Image Retrieval, Hand-drawn Sketch, Multi-stage Cross-modal Feature Alignment
>
> **摘要:** Fine-grained image retrieval via hand-drawn sketches or textual descriptions remains a critical challenge due to inherent modality gaps. While hand-drawn sketches capture complex structural contours, they lack color and texture, which text effectively provides despite omitting spatial contours. Motivated by the complementary nature of these modalities, we propose the Sketch and Text Based Image Retrieval (STBIR) framework. By synergizing the rich color and texture cues from text with the structural outlines provided by sketches, STBIR achieves superior fine-grained retrieval performance. First, a curriculum learning driven robustness enhancement module is proposed to enhance the model's robustness when handling queries of varying quality. Second, we introduce a category-knowledge-based feature space optimization module, thereby significantly boosting the model's representational power. Finally, we design a multi-stage cross-modal feature alignment mechanism to effectively mitigate the challenges of cross modal feature alignment. Furthermore, we curate the fine-grained STBIR benchmark dataset to rigorously validate the efficacy of our proposed framework and to provide data support as a reference for subsequent related research. Extensive experiments demonstrate that the proposed STBIR framework significantly outperforms state of the art methods.
>
---
#### [new 015] Diffusion Autoencoder for Unsupervised Artifact Restoration in Handheld Fundus Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像修复任务，旨在解决手持眼底图像中的伪影问题。提出一种无监督扩散自编码器，通过学习语义表示实现伪影恢复，提升诊断准确性。**

- **链接: [https://arxiv.org/pdf/2604.15723](https://arxiv.org/pdf/2604.15723)**

> **作者:** Mathumetha Palani; Kavya Puthumana; Ayantika Das; Ganapathy Krishnamurthi
>
> **备注:** 5 pages, 2 figures, 1 Table - ISBI IEEE 2025 CONFERENCE
>
> **摘要:** The advent of handheld fundus imaging devices has made ophthalmologic diagnosis and disease screening more accessible, efficient, and cost-effective. However, images captured from these setups often suffer from artifacts such as flash reflections, exposure variations, and motion-induced blur, which degrade image quality and hinder downstream analysis. While generative models have been effective in image restoration, most depend on paired supervision or predefined artifact structures, making them less adaptable to unstructured degradations commonly observed in handheld fundus images. To address this, we propose an unsupervised diffusion autoencoder that integrates a context encoder with the denoising process to learn semantically meaningful representations for artifact restoration. The model is trained only on high-quality table-top fundus images and infers to restore artifact-affected handheld acquisitions. We validate the restorations through quantitative and qualitative evaluations, and have shown that diagnostic accuracy increases to 81.17% on an unseen dataset and multiple artifact conditions
>
---
#### [new 016] Chain-of-Thought Degrades Visual Spatial Reasoning Capabilities of Multimodal LLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态模型研究任务，旨在解决CoT在视觉空间推理中的效果下降问题。通过实验发现CoT会损害模型表现，并揭示其依赖文本先验的缺陷。**

- **链接: [https://arxiv.org/pdf/2604.16060](https://arxiv.org/pdf/2604.16060)**

> **作者:** Sai Srinivas Kancheti; Aditya Sanjiv Kanade; Vineeth N. Balasubramanian; Tanuja Ganu
>
> **摘要:** Multimodal Reasoning Models (MRMs) leveraging Chain-of-Thought (CoT) based thinking have revolutionized mathematical and logical problem-solving. However, we show that this paradigm struggles with generalized spatial intelligence. We perform a comprehensive evaluation of seventeen models across thirteen spatial benchmarks and identify a critical gap: CoT prompting consistently degrades performance in visual spatial reasoning. Furthermore, through a novel No-Image++ ablation, we demonstrate that MRMs and CoT prompted MLMs suffer from severe shortcut learning, and hallucinate visual details from textual priors even when the image is absent. These findings challenge the efficacy of text-only CoT for spatial tasks and underscore the need for vision-centric reasoning paradigms.
>
---
#### [new 017] Enhancing Hazy Wildlife Imagery: AnimalHaze3k and IncepDehazeGan
- **分类: cs.CV**

- **简介: 该论文属于图像去雾任务，旨在解决大气 haze 对野生动物图像的干扰问题。通过构建数据集和提出新模型，提升图像质量及后续检测效果。**

- **链接: [https://arxiv.org/pdf/2604.16284](https://arxiv.org/pdf/2604.16284)**

> **作者:** Shivarth Rai; Tejeswar Pokuri
>
> **备注:** Accepted at CV4Animals Workshop, CVPR 2025
>
> **摘要:** Atmospheric haze significantly degrades wildlife imagery, impeding computer vision applications critical for conservation, such as animal detection, tracking, and behavior analysis. To address this challenge, we introduce AnimalHaze3k a synthetic dataset comprising of 3,477 hazy images generated from 1,159 clear wildlife photographs through a physics-based pipeline. Our novel IncepDehazeGan architecture combines inception blocks with residual skip connections in a GAN framework, achieving state-of-the-art performance (SSIM: 0.8914, PSNR: 20.54, and LPIPS: 0.1104), delivering 6.27% higher SSIM and 10.2% better PSNR than competing approaches. When applied to downstream detection tasks, dehazed images improved YOLOv11 detection mAP by 112% and IoU by 67%. These advances can provide ecologists with reliable tools for population monitoring and surveillance in challenging environmental conditions, demonstrating significant potential for enhancing wildlife conservation efforts through robust visual analytics.
>
---
#### [new 018] UA-Net: Uncertainty-Aware Network for TRISO Image Semantic Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出UA-Net，用于TRISO燃料图像的语义分割，解决人工评估费时且主观的问题。通过多阶段预训练和不确定性预测，提高分割精度与缺陷检测能力。**

- **链接: [https://arxiv.org/pdf/2604.15542](https://arxiv.org/pdf/2604.15542)**

> **作者:** Kyle Lucke; Zuzanna Krajewska-Travar; Shoukun Sun; Lu Cai; John D. Stempien; Min Xian
>
> **摘要:** Tristructural isotropic (TRISO)-coated particle fuels undergo dimensional changes and chemical reactions during high-temperature neutron irradiation. Post-irradiation materialography helps understand processes that impact fuel performance, such as coating integrity and fission product retention. Conventionally, experts manually evaluate features in thousands of cross sections of sub-mm-sized samples, which is tedious and subjective. In this work, we propose UA-Net, a deep learning framework that segments five characteristic regions of TRISO fuel micrographs and generates an uncertainty map for predictions. The model uses a multi-stage pretraining strategy, starting with general image representations learned from ImageNet, followed by fine-tuning on TRISO micrographs from various irradiation experiments and AGR-5/6/7 particle cross sections. A meta-model for uncertainty prediction is integrated to identify small defects in TRISO images. UA-Net was evaluated on a test set of 102 images, achieving mean Intersection over Union (mIoU) and mean Precision (mP) of 95.5% and 97.3%, respectively. The meta-model achieved a specificity of 91.8% and sensitivity of 93.5%, demonstrating strong performance in detecting misclassifications. The model was also applied to new TRISO images for qualitative evaluation, showing high accuracy in extracting layer regions.
>
---
#### [new 019] PixDLM: A Dual-Path Multimodal Language Model for UAV Reasoning Segmentation
- **分类: cs.CV**

- **简介: 该论文聚焦于无人机推理分割任务，解决其在视角、分辨率和尺度上的挑战。提出DRSeg基准和PixDLM模型，为该领域提供基础研究支持。**

- **链接: [https://arxiv.org/pdf/2604.15670](https://arxiv.org/pdf/2604.15670)**

> **作者:** Shuyan Ke; Yifan Mei; Changli Wu; Yonghan Zheng; Jiayi Ji; Liujuan Cao; Rongrong Ji
>
> **备注:** Accepted to CVPR 2026 (highlight)
>
> **摘要:** Reasoning segmentation has recently expanded from ground-level scenes to remote-sensing imagery, yet UAV data poses distinct challenges, including oblique viewpoints, ultra-high resolutions, and extreme scale variations. To address these issues, we formally define the UAV Reasoning Segmentation task and organize its semantic requirements into three dimensions: Spatial, Attribute, and Scene-level reasoning. Based on this formulation, we construct DRSeg, a large-scale benchmark for UAV reasoning segmentation, containing 10k high-resolution aerial images paired with Chain-of-Thought QA supervision across all three reasoning types. As a benchmark companion, we introduce PixDLM, a simple yet effective pixel-level multimodal language model that serves as a unified baseline for this task. Experiments on DRSeg establish strong baseline results and highlight the unique challenges of UAV reasoning segmentation, providing a solid foundation for future research.
>
---
#### [new 020] VEFX-Bench: A Holistic Benchmark for Generic Video Editing and Visual Effects
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频编辑任务，旨在解决缺乏高质量数据集和评估标准的问题。工作包括构建VEFX-Dataset和VEFX-Reward模型，以及发布VEFX-Bench基准。**

- **链接: [https://arxiv.org/pdf/2604.16272](https://arxiv.org/pdf/2604.16272)**

> **作者:** Xiangbo Gao; Sicong Jiang; Bangya Liu; Xinghao Chen; Minglai Yang; Siyuan Yang; Mingyang Wu; Jiongze Yu; Qi Zheng; Haozhi Wang; Jiayi Zhang; Jared Yang; Jie Yang; Zihan Wang; Qing Yin; Zhengzhong Tu
>
> **摘要:** As AI-assisted video creation becomes increasingly practical, instruction-guided video editing has become essential for refining generated or captured footage to meet professional requirements. Yet the field still lacks both a large-scale human-annotated dataset with complete editing examples and a standardized evaluator for comparing editing systems. Existing resources are limited by small scale, missing edited outputs, or the absence of human quality labels, while current evaluation often relies on expensive manual inspection or generic vision-language model judges that are not specialized for editing quality. We introduce VEFX-Dataset, a human-annotated dataset containing 5,049 video editing examples across 9 major editing categories and 32 subcategories, each labeled along three decoupled dimensions: Instruction Following, Rendering Quality, and Edit Exclusivity. Building on VEFX-Dataset, we propose VEFX-Reward, a reward model designed specifically for video editing quality assessment. VEFX-Reward jointly processes the source video, the editing instruction, and the edited video, and predicts per-dimension quality scores via ordinal regression. We further release VEFX-Bench, a benchmark of 300 curated video-prompt pairs for standardized comparison of editing systems. Experiments show that VEFX-Reward aligns more strongly with human judgments than generic VLM judges and prior reward models on both standard IQA/VQA metrics and group-wise preference evaluation. Using VEFX-Reward as an evaluator, we benchmark representative commercial and open-source video editing systems, revealing a persistent gap between visual plausibility, instruction following, and edit locality in current models.
>
---
#### [new 021] PolarMAE: Efficient Fetal Ultrasound Pre-training via Semantic Screening and Polar-Guided Masking
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决胎儿超声预训练效率低的问题。通过引入语义筛选、声学约束和极坐标掩码，提升模型性能与效率。**

- **链接: [https://arxiv.org/pdf/2604.15893](https://arxiv.org/pdf/2604.15893)**

> **作者:** Meng Lv; Yapeng Li; Hang Su; Juhua Liu; Bo Du
>
> **备注:** 10 pages, 6 figures, 3 tables
>
> **摘要:** Intelligent fetal ultrasound (US) interpretation is crucial for prenatal diagnosis, but high annotation costs and operator-induced variance make unsupervised pre-training a highly promising paradigm. However, existing pre-training methods largely ignore US-specific characteristics -- severe data redundancy, fan-shaped locality, and polar coordinate beamforming -- limiting their effectiveness in downstream tasks. To address this, we propose PolarMAE, a novel and efficient pre-training framework tailored for US images. Specifically, to mitigate continuous scanning redundancy, we introduce a Progressive Visual-Semantic Screening (PVSS) that adaptively extracts high-value samples, significantly boosting pre-training efficiency. Furthermore, we design an Acoustic-Bounded Region Constraint (ABRC) to accommodate US locality, forcing the model to focus strictly on valid acoustic regions rather than invalid dark backgrounds. Finally, leveraging the beamforming prior and local details, we propose a Polar-Texture Collaborative Masking (PTCM), enabling the model to capture underlying radial imaging patterns and critical tissue structures. Extensive experiments across diverse datasets and downstream interpretation tasks demonstrate that our method achieves state-of-the-art performance with strong pre-training scalability and efficiency.
>
---
#### [new 022] P3T: Prototypical Point-level Prompt Tuning with Enhanced Generalization for 3D Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于3D视觉语言模型的微调任务，旨在解决传统方法计算成本高和泛化能力差的问题。提出P³T方法，通过点级提示增强模型适应性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.15703](https://arxiv.org/pdf/2604.15703)**

> **作者:** Geunyoung Jung; Soohong Kim; Kyungwoo Song; Jiyoung Jung
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** With the rise of pre-trained models in the 3D point cloud domain for a wide range of real-world applications, adapting them to downstream tasks has become increasingly important. However, conventional full fine-tuning methods are computationally expensive and storage-intensive. Although prompt tuning has emerged as an efficient alternative, it often suffers from overfitting, thereby compromising generalization capability. To address this issue, we propose Prototypical Point-level Prompt Tuning (P$^3$T), a parameter-efficient prompt tuning method designed for pre-trained 3D vision-language models (VLMs). P$^3$T consists of two components: 1) \textit{Point Prompter}, which generates instance-aware point-level prompts for the input point cloud, and 2) \textit{Text Prompter}, which employs learnable prompts into the input text instead of hand-crafted ones. Since both prompters operate directly on input data, P$^3$T enables task-specific adaptation of 3D VLMs without sacrificing generalizability. Furthermore, to enhance embedding space alignment, which is key to fine-tuning 3D VLMs, we introduce a prototypical loss that reduces intra-category variance. Extensive experiments demonstrate that our method matches or outperforms full fine-tuning in classification and few-shot learning, and further exhibits robust generalization under data shift in the cross-dataset setting. The code is available at \textcolor{violet}{this https URL}.
>
---
#### [new 023] IA-CLAHE: Image-Adaptive Clip Limit Estimation for CLAHE
- **分类: cs.CV**

- **简介: 该论文提出IA-CLAHE，解决传统CLAHE过度增强的问题。通过自适应估计局部区域的裁剪限值，提升图像对比度和视觉质量，无需任务特定数据。**

- **链接: [https://arxiv.org/pdf/2604.16010](https://arxiv.org/pdf/2604.16010)**

> **作者:** Rikuto Otsuka; Yuho Shoji; Yuka Ogino; Takahiro Toizumi; Atsushi Ito
>
> **备注:** Accepted to NTIRE 2026 Workshop at CVPR 2026
>
> **摘要:** This paper proposes image-adaptive contrast limited adaptive histogram equalization (IA-CLAHE). Conventional CLAHE is widely used to boost the performance of various computer vision tasks and to improve visual quality for human perception in practical industrial applications. CLAHE applies contrast limited histogram equalization to each local region to enhance local contrast. However, CLAHE often leads to over-enhancement, because the contrast-limiting parameter clip limit is fixed regardless of the histogram distribution of each local region. Our IA-CLAHE addresses this limitation by adaptively estimating tile-wise clip limits from the input image. To achieve this, we train a lightweight clip limits estimator with a differentiable extension of CLAHE, enabling end-to-end optimization. Unlike prior learning-based CLAHE methods, IA-CLAHE does not require pre-searched ground-truth clip limits or task-specific datasets, because it learns to map input image histograms toward a domain-invariant uniform distribution, enabling zero-shot generalization across diverse conditions. Experimental results show that IA-CLAHE consistently improves recognition performance, while simultaneously enhancing visual quality for human perception, without requiring any task-specific training data.
>
---
#### [new 024] Fed3D: Federated 3D Object Detection
- **分类: cs.CV**

- **简介: 该论文属于3D目标检测任务，解决多机器人网络中的隐私和通信成本问题。提出Fed3D框架，通过类感知损失和提示模块实现高效联邦学习。**

- **链接: [https://arxiv.org/pdf/2604.15795](https://arxiv.org/pdf/2604.15795)**

> **作者:** Suyan Dai; Chenxi Liu; Fazeng Li; Peican Lin
>
> **摘要:** 3D object detection models trained in one server plays an important role in autonomous driving, robotics manipulation, and augmented reality scenarios. However, most existing methods face severe privacy concern when deployed on a multi-robot perception network to explore large-scale 3D scene. Meanwhile, it is highly challenging to employ conventional federated learning methods on 3D object detection scenes, due to the 3D data heterogeneity and limited communication bandwidth. In this paper, we take the first attempt to propose a novel Federated 3D object detection framework (i.e., Fed3D), to enable distributed learning for 3D object detection with privacy preservation. Specifically, considering the irregular input 3D object in local robot and various category distribution between robots could cause local heterogeneity and global heterogeneity, respectively. We then propose a local-global class-aware loss for the 3D data heterogeneity issue, which could balance gradient back-propagation rate of different 3D categories from local and global aspects. To reduce communication cost on each round, we develop a federated 3D prompt module, which could only learn and communicate the prompts with few learnable parameters. To the end, several extensive experiments on federated 3D object detection show that our Fed3D model significantly outperforms state-of-the-art algorithms with lower communication cost when providing the limited local training data.
>
---
#### [new 025] Dental Panoramic Radiograph Analysis Using YOLO26 From Tooth Detection to Disease Diagnosis
- **分类: cs.CV**

- **简介: 该论文属于牙科影像分析任务，旨在解决手动解读全景牙片效率低、易出错的问题。通过YOLOv26实现牙齿检测、编号及疾病分割，提升诊断效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.16231](https://arxiv.org/pdf/2604.16231)**

> **作者:** Khawaja Azfar Asif; Rafaqat Alam Khan
>
> **摘要:** Panoramic radiography is a fundamental diagnostic tool in dentistry, offering a comprehensive view of the entire dentition with minimal radiation exposure. However, manual interpretation is time-consuming and prone to errors, especially in high-volume clinical settings. This creates a pressing need for efficient automated solutions. This study presents the first application of YOLOv26 for automated tooth detection, FDI-based numbering, and dental disease segmentation in panoramic radiographs. The DENTEX dataset was preprocessed using Roboflow for format conversion and augmentation, yielding 1,082 images for tooth enumeration and 1,040 images for disease segmentation across four pathology classes. Five YOLOv26-seg variants were trained on Google Colab using transfer learning at a resolution of 800x800. Results demonstrate that the YOLOv26m-seg model achieved the best performance for tooth enumeration, with a precision of 0.976, recall of 0.970, and box mAP50 of 0.976. It outperformed the YOLOv8x baseline by 4.9% in precision and 3.3% in mAP50, while also enabling high-quality mask-level segmentation (mask mAP50 = 0.970). For disease segmentation, the YOLOv26l-seg model attained a box mAP50 of 0.591 and a mask mAP50 of 0.547. Impacted teeth showed the highest per-class average precision (0.943), indicating that visual distinctiveness influences detection performance more than annotation quantity. Overall, these findings demonstrate that YOLOv26-based models offer a robust and accurate framework for automated dental image analysis, with strong potential to enhance diagnostic efficiency and consistency in clinical practice.
>
---
#### [new 026] Splats in Splats++: Robust and Generalizable 3D Gaussian Splatting Steganography
- **分类: cs.CV**

- **简介: 该论文属于3D内容隐写任务，解决现有方法在嵌入容量与资产可用性间的矛盾。提出Splats in Splats++框架，实现高效、隐蔽的3D/4D信息嵌入。**

- **链接: [https://arxiv.org/pdf/2604.15862](https://arxiv.org/pdf/2604.15862)**

> **作者:** Yijia Guo; Wenkai Huang; Tong Hu; Gaolei Li; Yang Li; Yuxin Hong; Liwen Hu; Xitong Ling; Jianhua Li; Shengbo Chen; Tiejun Huang; Lei Ma
>
> **摘要:** 3D Gaussian Splatting (3DGS) has recently redefined the paradigm of 3D reconstruction, striking an unprecedented balance between visual fidelity and computational efficiency. As its adoption proliferates, safeguarding the copyright of explicit 3DGS assets has become paramount. However, existing invisible message embedding frameworks struggle to reconcile secure and high-capacity data embedding with intrinsic asset utility, often disrupting the native rendering pipeline or exhibiting vulnerability to structural perturbations. In this work, we present \textbf{\textit{Splats in Splats++}}, a unified and pipeline-agnostic steganography framework that seamlessly embeds high-capacity 3D/4D content directly within the native 3DGS representation. Grounded in a principled analysis of the frequency distribution of Spherical Harmonics (SH), we propose an importance-graded SH coefficient encryption scheme that achieves imperceptible embedding without compromising the original expressive power. To fundamentally resolve the geometric ambiguities that lead to message leakage, we introduce a \textbf{Hash-Grid Guided Opacity Mapping} mechanism. Coupled with a novel \textbf{Gradient-Gated Opacity Consistency Loss}, our formulation enforces a stringent spatial-attribute coupling between the original and hidden scenes, effectively projecting the discrete attribute mapping into a continuous, attack-resilient latent manifold. Extensive experiments demonstrate that our method substantially outperforms existing approaches, achieving up to \textbf{6.28 db} higher message fidelity, \textbf{3$\times$} faster rendering, and exceptional robustness against aggressive 3D-targeted structural attacks (e.g., GSPure). Furthermore, our framework exhibits remarkable versatility, generalizing seamlessly to 2D image embedding, 4D dynamic scene steganography, and diverse downstream tasks.
>
---
#### [new 027] Concept-wise Attention for Fine-grained Concept Bottleneck Models
- **分类: cs.CV**

- **简介: 该论文属于概念瓶颈模型任务，解决概念建模中的粒度不对齐和互斥性忽略问题。提出CoAt-CBM框架，通过概念注意力实现细粒度图像-概念对齐。**

- **链接: [https://arxiv.org/pdf/2604.15748](https://arxiv.org/pdf/2604.15748)**

> **作者:** Minghong Zhong; Guoshuai Zou; Kanghao Chen; Dexia Chen; Ruixuan Wang
>
> **备注:** 10 pages, 7 figures, Accepted by CVPR 2026 Fingdings
>
> **摘要:** Recently impressive performance has been achieved in Concept Bottleneck Models (CBM) by utilizing the image-text alignment learned by a large pre-trained vision-language model (i.e. CLIP). However, there exist two key limitations in concept modeling. Existing methods often suffer from pre-training biases, manifested as granularity misalignment or reliance on structural priors. Moreover, fine-tuning with Binary Cross-Entropy (BCE) loss treats each concept independently, which ignores mutual exclusivity among concepts, leading to suboptimal alignment. To address these limitations, we propose Concept-wise Attention for Fine-grained Concept Bottleneck Models (CoAt-CBM), a novel framework that achieves adaptive fine-grained image-concept alignment and high interpretability. Specifically, CoAt-CBM employs learnable concept-wise visual queries to adaptively obtain fine-grained concept-wise visual embeddings, which are then used to produce a concept score vector. Then, a novel concept contrastive optimization guides the model to handle the relative importance of the concept scores, enabling concept predictions to faithfully reflect the image content and improved alignment. Extensive experiments demonstrate that CoAt-CBM consistently outperforms state-of-the-art methods. The codes will be available upon acceptance.
>
---
#### [new 028] Frequency-Aware Flow Matching for High-Quality Image Generation
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，解决流匹配模型生成细节不足的问题。通过引入频率感知机制，提升生成图像的细节质量。**

- **链接: [https://arxiv.org/pdf/2604.15521](https://arxiv.org/pdf/2604.15521)**

> **作者:** Sucheng Ren; Qihang Yu; Ju He; Xiaohui Shen; Alan Yuille; Liang-Chieh Chen
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Flow matching models have emerged as a powerful framework for realistic image generation by learning to reverse a corruption process that progressively adds Gaussian noise. However, because noise is injected in the latent domain, its impact on different frequency components is non-uniform. As a result, during inference, flow matching models tend to generate low-frequency components (global structure) in the early stages, while high-frequency components (fine details) emerge only later in the reverse process. Building on this insight, we propose Frequency-Aware Flow Matching (FreqFlow), a novel approach that explicitly incorporates frequency-aware conditioning into the flow matching framework via time-dependent adaptive weighting. We introduce a two-branch architecture: (1) a frequency branch that separately processes low- and high-frequency components to capture global structure and refine textures and edges, and (2) a spatial branch that synthesizes images in the latent domain, guided by the frequency branch's output. By explicitly integrating frequency information into the generation process, FreqFlow ensures that both large-scale coherence and fine-grained details are effectively modeled low-frequency conditioning reinforces global structure, while high-frequency conditioning enhances texture fidelity and detail sharpness. On the class-conditional ImageNet-256 generation benchmark, our method achieves state-of-the-art performance with an FID of 1.38, surpassing the prior diffusion model DiT and flow matching model SiT by 0.79 and 0.58 FID, respectively. Code is available at this https URL.
>
---
#### [new 029] Continual Hand-Eye Calibration for Open-world Robotic Manipulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人手眼标定任务，解决开放场景下模型遗忘问题。提出SARS和SPDD方法，实现持续标定与知识保留。**

- **链接: [https://arxiv.org/pdf/2604.15814](https://arxiv.org/pdf/2604.15814)**

> **作者:** Fazeng Li; Gan Sun; Chenxi Liu; Yao He; Wei Cong; Yang Cong
>
> **摘要:** Hand-eye calibration through visual localization is a critical capability for robotic manipulation in open-world environments. However, most deep learning-based calibration models suffer from catastrophic forgetting when adapting into unseen data amongst open-world scene changes, while simple rehearsal-based continual learning strategy cannot well mitigate this issue. To overcome this challenge, we propose a continual hand-eye calibration framework, enabling robots to adapt to sequentially encountered open-world manipulation scenes through spatially replay strategy and structure-preserving distillation. Specifically, a Spatial-Aware Replay Strategy (SARS) constructs a geometrically uniform replay buffer that ensures comprehensive coverage of each scene pose space, replacing redundant adjacent frames with maximally informative viewpoints. Meanwhile, a Structure-Preserving Dual Distillation (SPDD) is proposed to decompose localization knowledge into coarse scene layout and fine pose precision, and distills them separately to alleviate both types of forgetting during continual adaptation. As a new manipulation scene arrives, SARS provides geometrically representative replay samples from all prior scenes, and SPDD applies structured distillation on these samples to retain previously learned knowledge. After training on the new scene, SARS incorporates selected samples from the new scene into the replay buffer for future rehearsal, allowing the model to continuously accumulate multi-scene calibration capability. Experiments on multiple public datasets show significant anti scene forgetting performance, maintaining accuracy on past scenes while preserving adaptation to new scenes, confirming the effectiveness of the framework.
>
---
#### [new 030] Where Do Vision-Language Models Fail? World Scale Analysis for Image Geolocalization
- **分类: cs.CV**

- **简介: 该论文研究视觉语言模型在国家层面图像地理定位任务中的表现，旨在评估其在无监督条件下的地理推理能力，通过测试不同模型的泛化性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.16248](https://arxiv.org/pdf/2604.16248)**

> **作者:** Siddhant Bharadwaj; Ashish Vashist; Fahimul Aleem; Shruti Vyas
>
> **备注:** Accepted to the CVPR EarthVision 2026 Workshop
>
> **摘要:** Image geolocalization has traditionally been addressed through retrieval-based place recognition or geometry-based visual localization pipelines. Recent advances in Vision-Language Models (VLMs) have demonstrated strong zero-shot reasoning capabilities across multimodal tasks, yet their performance in geographic inference remains underexplored. In this work, we present a systematic evaluation of multiple state-of-the-art VLMs for country-level image geolocalization using ground-view imagery only. Instead of relying on image matching, GPS metadata, or task-specific training, we evaluate prompt-based country prediction in a zero-shot setting. The selected models are tested on three geographically diverse datasets to assess their robustness and generalization ability. Our results reveal substantial variation across models, highlighting the potential of semantic reasoning for coarse geolocalization and the limitations of current VLMs in capturing fine-grained geographic cues. This study provides the first focused comparison of modern VLMs for country-level geolocalization and establishes a foundation for future research at the intersection of multimodal reasoning and geographic understanding.
>
---
#### [new 031] SegMix:Shuffle-based Feedback Learning for Semantic Segmentation of Pathology Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，旨在解决标注数据不足的问题。通过shuffle-based反馈学习生成更高质量的伪分割掩码，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.15777](https://arxiv.org/pdf/2604.15777)**

> **作者:** Zhiling Yan; Sicheng Chen; Tianyi Zhang; Nan Ying; Yanli Lei; Guanglei Zhang
>
> **摘要:** Segmentation is a critical task in computational pathology, as it identifies areas affected by disease or abnormal growth and is essential for diagnosis and treatment. However, acquiring high-quality pixel-level supervised segmentation data requires significant workload demands from experienced pathologists, limiting the application of deep learning. To overcome this challenge, relaxing the label conditions to image-level classification labels allows for more data to be used and more scenarios to be enabled. One approach is to leverage Class Activation Map (CAM) to generate pseudo pixel-level annotations for semantic segmentation with only image-level labels. However, this method fails to thoroughly explore the essential characteristics of pathology images, thus identifying only small areas that are insufficient for pseudo masking. In this paper, we propose a novel shuffle-based feedback learning method inspired by curriculum learning to generate higher-quality pseudo-semantic segmentation masks. Specifically, we perform patch level shuffle of pathology images, with the model adaptively adjusting the shuffle strategy based on feedback from previous learning. Experimental results demonstrate that our proposed approach outperforms state-of-the-arts on three different datasets.
>
---
#### [new 032] RefereeBench: Are Video MLLMs Ready to be Multi-Sport Referees
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态模型评估任务，旨在解决MLLM在体育裁判决策中的能力不足问题。构建了RefereeBench基准，评估模型在规则判断、时间定位等能力，发现现有模型表现有限。**

- **链接: [https://arxiv.org/pdf/2604.15736](https://arxiv.org/pdf/2604.15736)**

> **作者:** Yichen Xu; Yuanhang Liu; Chuhan Wang; Zihan Zhao; jinghan luo; Jianzhe Ma; Wenxuan Wang; Qin Jin
>
> **备注:** Work in Progress
>
> **摘要:** While Multimodal Large Language Models (MLLMs) excel at generic video understanding, their ability to support specialized, rule-grounded decision-making remains insufficiently explored. In this paper, we introduce RefereeBench, the first large-scale benchmark for evaluating MLLMs as automatic sports referees. Spanning 11 sports with 925 curated videos and 6,475 QA pairs, RefereeBench evaluates five core officiating abilities: foul existence, foul and penalty classification, foul and penalty reasoning, entity perception, and temporal grounding. The benchmark is fully human-annotated to ensure high-quality annotations grounded in authentic officiating logic and multimodal evidence. Extensive evaluations of state-of-the-art MLLMs show that even the strongest models, such as Doubao-Seed-1.8 and Gemini-3-Pro, achieve only around 60% accuracy, while the strongest open-source model, Qwen3-VL, reaches only 47%. These results indicate that current models remain far from being reliable sports referees. Further analysis shows that while models can often identify incidents and involved entities, they struggle with rule application and temporal grounding, and frequently over-call fouls on normal clips. Our benchmark highlights the need for future MLLMs that better integrate domain knowledge and multimodal understanding, advancing trustworthy AI-assisted officiating and broader multimodal decision-making.
>
---
#### [new 033] Winner of CVPR2026 NTIRE Challenge on Image Shadow Removal: Semantic and Geometric Guidance for Shadow Removal via Cascaded Refinement
- **分类: cs.CV**

- **简介: 该论文属于图像去阴影任务，解决阴影残留问题。提出三阶段渐进式去阴影方法，结合语义与几何信息，提升去阴影效果。**

- **链接: [https://arxiv.org/pdf/2604.16177](https://arxiv.org/pdf/2604.16177)**

> **作者:** Lorenzo Beltrame; Jules Salzinger; Filip Svoboda; Jasmin Lampert; Phillipp Fanta-Jende; Radu Timofte; Marco Koerner
>
> **备注:** 10 pages, 4 figures, 5 tables, accepted at the CVPR 2026 Workshops (NTIRE 2026 Image Shadow Removal Challenge). Code and materials are available at this https URL
>
> **摘要:** We present a three-stage progressive shadow-removal pipeline for the CVPR2026 NTIRE WSRD+ challenge. Built on OmniSR, our method treats deshadowing as iterative direct refinement, where later stages correct residual artefacts left by earlier predictions. The model combines RGB appearance with frozen DINOv2 semantic guidance and geometric cues from monocular depth and surface normals, reused across all stages. To stabilise multi-stage optimisation, we introduce a contraction-constrained objective that encourages non-increasing reconstruction error across the cascade. A staged training pipeline transfers from earlier WSRD pretraining to WSRD+ supervision and final WSRD+ 2026 adaptation with cosine-annealed checkpoint ensembling. On the official WSRD+ 2026 hidden test set, our final ensemble achieved 26.680 PSNR, 0.8740 SSIM, 0.0578 LPIPS, and 26.135 FID, ranked first overall, and won the NTIRE 2026 Image Shadow Removal Challenge. The strong performance of the proposed model is further validated on the ISTD+ and UAV-SC+ datasets.
>
---
#### [new 034] Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，解决模型感知与实际不一致的问题。通过调节信息流，提升模型对关键视觉特征的注意力，改善回答准确性。**

- **链接: [https://arxiv.org/pdf/2604.15809](https://arxiv.org/pdf/2604.15809)**

> **作者:** Chengxin Liu; Wonseok Choi; Chenshuang Zhang; Tae-Hyun Oh
>
> **备注:** CVPR 2026. Project page: this https URL
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated strong capability in a wide range of tasks such as visual recognition, document parsing, and visual grounding. Nevertheless, recent work shows that while VLMs often manage to capture the correct image region corresponding to the question, they do not necessarily produce the correct answers. In this work, we demonstrate that this misalignment could be attributed to suboptimal information flow within VLMs, where text tokens distribute too much attention to irrelevant visual tokens, leading to incorrect answers. Based on the observation, we show that modulating the information flow during inference can improve the perception capability of VLMs. The idea is that text tokens should only be associated with important visual tokens during decoding, eliminating the interference of irrelevant regions. To achieve this, we propose a token dynamics-based method to determine the importance of visual tokens, where visual tokens that exhibit distinct activation patterns during different decoding stages are viewed as important. We apply our approach to representative open-source VLMs and evaluate on various datasets, including visual question answering, visual grounding and counting, optical character recognition, and object hallucination. The results show that our approach significantly improves the performance of baselines. Project page: this https URL.
>
---
#### [new 035] NeuroLip: An Event-driven Spatiotemporal Learning Framework for Cross-Scene Lip-Motion-based Visual Speaker Recognition
- **分类: cs.CV; cs.AI; cs.CR; cs.DB; cs.LG**

- **简介: 该论文属于视觉说话人识别任务，解决传统方法在环境变化下的性能下降问题。提出NeuroLip框架，利用事件驱动机制提升跨场景泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.15718](https://arxiv.org/pdf/2604.15718)**

> **作者:** Junguang Yao; Wenye Liu; Stjepan Picek; Yue Zheng
>
> **摘要:** Visual speaker recognition based on lip motion offers a silent, hands-free, and behavior-driven biometric solution that remains effective even when acoustic cues are unavailable. Compared to traditional methods that rely heavily on appearance-dependent representations, lip motion encodes subject-specific behavioral dynamics driven by consistent articulation patterns and muscle coordination, offering inherent stability across environmental changes. However, capturing these robust, fine-grained dynamics is challenging for conventional frame-based cameras due to motion blur and low dynamic range. To exploit the intrinsic stability of lip motion and address these sensing limitations, we propose NeuroLip, an event-based framework that captures fine-grained lip dynamics under a strict yet practical cross-scene protocol: training is performed under a single controlled condition, while recognition must generalize to unseen viewing and lighting conditions. NeuroLip features a 1) Temporal-aware Voxel Encoding module with adaptive event weighting, 2) Structure-aware Spatial Enhancer that amplifies discriminative behavioral patterns by suppressing noise while preserving vertically structured motion information, and 3) Polarity Consistency Regularization mechanism to retain motion-direction cues encoded in event polarities. To facilitate systematic evaluation, we introduce DVSpeaker, a comprehensive event-based lip-motion dataset comprising 50 subjects recorded under four distinct viewpoint and illumination scenarios. Extensive experiments demonstrate that NeuroLip achieves near-perfect matched-scene accuracy and robust cross-scene generalization, attaining over 71% accuracy on unseen viewpoints and nearly 76% under low-light conditions, outperforming representative existing methods by at least 8.54%. The dataset and code are publicly available at this https URL.
>
---
#### [new 036] Early Detection of Acute Myeloid Leukemia (AML) Using YOLOv12 Deep Learning Model
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在解决AML细胞准确识别问题。通过YOLOv12模型与图像预处理技术，实现高精度的AML细胞分类。**

- **链接: [https://arxiv.org/pdf/2604.16082](https://arxiv.org/pdf/2604.16082)**

> **作者:** Enas E. Ahmed; Salah A. Aly; Mayar Moner
>
> **备注:** 6 pages, 10 figures, 2 tables
>
> **摘要:** Acute Myeloid Leukemia (AML) is one of the most life-threatening type of blood cancers, and its accurate classification is considered and remains a challenging task due to the visual similarity between various cell types. This study addresses the classification of the multiclasses of AML cells Utilizing YOLOv12 deep learning model. We applied two segmentation approaches based on cell and nucleus features, using Hue channel and Otsu thresholding techniques to preprocess the images prior to classification. Our experiments demonstrate that YOLOv12 with Otsu thresholding on cell-based segmentation achieved the highest level of validation and test accuracy, both reaching 99.3%.
>
---
#### [new 037] From Articles to Canopies: Knowledge-Driven Pseudo-Labelling for Tree Species Classification using LLM Experts
- **分类: cs.CV**

- **简介: 该论文属于树种分类任务，解决标签有限、光谱混合和生态异质性问题。融合多源数据与生态知识，提出一种半监督方法，提升分类精度。**

- **链接: [https://arxiv.org/pdf/2604.16115](https://arxiv.org/pdf/2604.16115)**

> **作者:** Michał Romaszewski; Dominik Kopeć; Michał Cholewa; Katarzyna Kołodziej; Przemysław Głomb; Jan Niedzielko; Jakub Charyton; Justyna Wylazłowska; Anna Jarocińska
>
> **摘要:** Hyperspectral tree species classification is challenging due to limited and imbalanced class labels, spectral mixing (overlapping light signatures from multiple species), and ecological heterogeneity (variability among ecological systems). Addressing these challenges requires methods that integrate biological and structural characteristics of vegetation, such as canopy architecture and interspecific interactions, rather than relying solely on spectral signatures. This paper presents a biologically informed, semi-supervised deep learning method that integrates multi-sensor Earth observation data, specifically hyperspectral imaging (HSI) and airborne laser scanning (ALS), with expert, ecological knowledge. The approach relies on biologically inspired pseudo-labelling over a precomputed canopy graph, yielding accurate classification at low training cost. In addition, ecological priors on species cohabitation are automatically derived from reliable sources using large language models (LLMs) and encoded as a cohabitation matrix with likelihoods of species occurring together. These priors are incorporated into the pseudo-labelling strategy, effectively introducing expert knowledge into the model. Experiments on a real-world forest dataset demonstrate 5.6% improvement over the best reference method. Expert evaluation of cohabitation priors reveals high accuracy with differences no larger than 15%.
>
---
#### [new 038] SIMMER: Cross-Modal Food Image--Recipe Retrieval via MLLM-Based Embedding
- **分类: cs.CV; cs.CL; cs.IR; cs.LG; cs.MM**

- **简介: 该论文属于跨模态检索任务，旨在解决食物图像与食谱文本之间的匹配问题。提出SIMMER模型，使用MLLM进行统一编码，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2604.15628](https://arxiv.org/pdf/2604.15628)**

> **作者:** Keisuke Gomi; Keiji Yanai
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Cross-modal retrieval between food images and recipe texts is an important task with applications in nutritional management, dietary logging, and cooking assistance. Existing methods predominantly rely on dual-encoder architectures with separate image and text encoders, requiring complex alignment strategies and task-specific network designs to bridge the semantic gap between modalities. In this work, we propose SIMMER (Single Integrated Multimodal Model for Embedding Recipes), which applies Multimodal Large Language Model (MLLM)-based embedding models, specifically VLM2Vec, to this task, replacing the conventional dual-encoder paradigm with a single unified encoder that processes both food images and recipe texts. We design prompt templates tailored to the structured nature of recipes, which consist of a title, ingredients, and cooking instructions, enabling effective embedding generation by the MLLM. We further introduce a component-aware data augmentation strategy that trains the model on both complete and partial recipes, improving robustness to incomplete inputs. Experiments on the Recipe1M dataset demonstrate that SIMMER achieves state-of-the-art performance across both the 1k and 10k evaluation settings, substantially outperforming all prior methods. In particular, our best model improves the 1k image-to-recipe R@1 from 81.8\% to 87.5\% and the 10k image-to-recipe R@1 from 56.5\% to 65.5\% compared to the previous best method.
>
---
#### [new 039] TableSeq: Unified Generation of Structure, Content, and Layout
- **分类: cs.CV**

- **简介: 该论文提出TableSeq，用于统一处理表格结构、内容和布局的识别任务，解决传统方法依赖OCR和多阶段处理的问题。**

- **链接: [https://arxiv.org/pdf/2604.16070](https://arxiv.org/pdf/2604.16070)**

> **作者:** Laziz Hamdi; Amine Tamasna; Pascal Boisson; Thierry Paquet
>
> **摘要:** We present TableSeq, an image-only, end-to-end framework for joint table structure recognition, content recognition, and cell localization. The model formulates these tasks as a single sequence-generation problem: one decoder produces an interleaved stream of \texttt{HTML} tags, cell text, and discretized coordinate tokens, thereby aligning logical structure, textual content, and cell geometry within a unified autoregressive sequence. This design avoids external OCR, auxiliary decoders, and complex multi-stage post-processing. TableSeq combines a lightweight high-resolution FCN-H16 encoder with a minimal structure-prior head and a single-layer transformer encoder, yielding a compact architecture that remains effective on challenging layouts. Across standard benchmarks, TableSeq achieves competitive or state-of-the-art results while preserving architectural simplicity. It reaches 95.23 TEDS / 96.83 S-TEDS on PubTabNet, 97.45 TEDS / 98.69 S-TEDS on FinTabNet, and 99.79 / 99.54 / 99.66 precision / recall / F1 on SciTSR under the CAR protocol, while remaining competitive on PubTables-1M under GriTS. Beyond TSR/TCR, the same sequence interface generalizes to index-based table querying without task-specific heads, achieving the best IRDR score and competitive ICDR/ICR performance. We also study multi-token prediction for faster blockwise decoding and show that it reduces inference latency with only limited accuracy degradation. Overall, TableSeq provides a practical and reproducible single-stream baseline for unified table recognition, and the source code will be made publicly available at this https URL.
>
---
#### [new 040] Robust Multispectral Semantic Segmentation under Missing or Full Modalities via Structured Latent Projection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态语义分割任务，解决传感器失效导致模态缺失的问题。提出CBC-SLP模型，通过结构化潜在投影保留模态不变与特定信息，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2604.15856](https://arxiv.org/pdf/2604.15856)**

> **作者:** Irem Ulku; Erdem Akagündüz; Ömer Özgür Tanrıöver
>
> **备注:** 15 pages, 7 figures, 9 tables
>
> **摘要:** Multimodal remote sensing data provide complementary information for semantic segmentation, but in real-world deployments, some modalities may be unavailable due to sensor failures, acquisition issues, or challenging atmospheric conditions. Existing multimodal segmentation models typically address missing modalities by learning a shared representation across inputs. However, this approach can introduce a trade-off by compromising modality-specific complementary information and reducing performance when all modalities are available. In this paper, we tackle this limitation with CBC-SLP, a multimodal semantic segmentation model designed to preserve both modality-invariant and modality-specific information. Inspired by the theoretical results on modality alignment, which state that perfectly aligned multimodal representations can lead to sub-optimal performance in downstream prediction tasks, we propose a novel structured latent projection approach as an architectural inductive bias. Rather than enforcing this strategy through a loss term, we incorporate it directly into the architecture. In particular, to use the complementary information effectively while maintaining robustness under random modality dropout, we structure the latent representations into shared and modality-specific components and adaptively transfer them to the decoder according to the random modality availability mask. Extensive experiments on three multimodal remote sensing image sets demonstrate that CBC-SLP consistently outperforms state-of-the-art multimodal models across full and missing modality scenarios. Besides, we empirically demonstrate that the proposed strategy can recover the complementary information that may not be preserved in a shared representation. The code is available at this https URL.
>
---
#### [new 041] Causal Bootstrapped Alignment for Unsupervised Video-Based Visible-Infrared Person Re-Identification
- **分类: cs.CV**

- **简介: 该论文属于视频可见-红外行人重识别任务，解决无监督学习中的跨模态对齐问题。提出CBA框架，通过因果干预和原型引导提升表示质量与对齐效果。**

- **链接: [https://arxiv.org/pdf/2604.15631](https://arxiv.org/pdf/2604.15631)**

> **作者:** Shuang Li; Jiaxu Leng; Changjiang Kuang; Mingpi Tan; Yu Yuan; Xinbo Gao
>
> **备注:** Submit to IEEE TIFS
>
> **摘要:** VVI-ReID is a critical technique for all-day surveillance, where temporal information provides additional cues beyond static images. However, existing approaches rely heavily on fully supervised learning with expensive cross-modality annotations, limiting scalability. To address this issue, we investigate Unsupervised Learning for VVI-ReID (USL-VVI-ReID), which learns identity-discriminative representations directly from unlabeled video tracklets. Directly extending image-based USL-VI-ReID methods to this setting with generic pretrained encoders leads to suboptimal performance. Such encoders suffer from weak identity discrimination and strong modality bias, resulting in severe intra-modality identity confusion and pronounced clustering granularity imbalance between visible and infrared modalities. These issues jointly degrade pseudo-label reliability and hinder effective cross-modality alignment. To address these challenges, we propose a Causal Bootstrapped Alignment (CBA) framework that explicitly exploits inherent video priors. First, we introduce Causal Intervention Warm-up (CIW), which performs sequence-level causal interventions by leveraging temporal identity consistency and cross-modality identity consistency to suppress modality- and motion-induced spurious correlations while preserving identity-relevant semantics, yielding cleaner representations for unsupervised clustering. Second, we propose Prototype-Guided Uncertainty Refinement (PGUR), which employs a coarse-to-fine alignment strategy to resolve cross-modality granularity mismatch, reorganizing under-clustered infrared representations under the guidance of reliable visible prototypes with uncertainty-aware supervision. Extensive experiments on the HITSZ-VCM and BUPTCampus benchmarks demonstrate that CBA significantly outperforms existing USL-VI-ReID methods when extended to the USL-VVI-ReID setting.
>
---
#### [new 042] Hero-Mamba: Mamba-based Dual Domain Learning for Underwater Image Enhancement
- **分类: cs.CV**

- **简介: 该论文属于 underwater image enhancement 任务，解决水下图像颜色失真、对比度低等问题。提出 Hero-Mamba 网络，结合空间与频域信息，提升图像质量。**

- **链接: [https://arxiv.org/pdf/2604.16266](https://arxiv.org/pdf/2604.16266)**

> **作者:** Tejeswar Pokuri; Shivarth Rai
>
> **备注:** Accepted at AI4ES Workshop AAAI 2026
>
> **摘要:** Underwater images often suffer from severe degradation, such as color distortion, low contrast, and blurred details, due to light absorption and scattering in water. While learning-based methods like CNNs and Transformers have shown promise, they face critical limitations: CNNs struggle to model the long-range dependencies needed for non-uniform degradation, and Transformers incur quadratic computational complexity, making them inefficient for high-resolution images. To address these challenges, we propose Hero-Mamba, a novel Mamba-based network that achieves efficient dual-domain learning for underwater image enhancement. Our approach uniquely processes information from both the spatial domain (RGB image) and the spectral domain (FFT components) in parallel. This dual-domain input allows the network to decouple degradation factors, separating color/brightness information from texture/noise. The core of our network utilizes Mamba-based SS2D blocks to capture global receptive fields and long-range dependencies with linear complexity, overcoming the limitations of both CNNs and Transformers. Furthermore, we introduce a ColorFusion block, guided by a background light prior, to restore color information with high fidelity. Extensive experiments on the LSUI and UIEB benchmark datasets demonstrate that Hero-Mamba outperforms state-of-the-art methods. Notably, our model achieves a PSNR of 25.802 and an SSIM of 0.913 on LSUI, validating its superior performance and generalization capabilities.
>
---
#### [new 043] neuralCAD-Edit: An Expert Benchmark for Multimodal-Instructed 3D CAD Model Editing
- **分类: cs.CV; cs.CE**

- **简介: 该论文提出neuralCAD-Edit，首个针对3D CAD模型编辑的基准。任务是解决多模态指令下的CAD编辑问题，通过真实设计师操作数据进行评估，揭示现有模型与人类专家间的性能差距。**

- **链接: [https://arxiv.org/pdf/2604.16170](https://arxiv.org/pdf/2604.16170)**

> **作者:** Toby Perrett; Matthew Bouchard; William McCarthy
>
> **备注:** Project page: this https URL
>
> **摘要:** We introduce neuralCAD-Edit, the first benchmark for editing 3D CAD models collected from expert CAD engineers. Instead of text conditioning as in prior works, we collect realistic CAD editing requests by capturing videos of professional designers, interacting directly with CAD models in CAD software, while talking, pointing and drawing. We recruited ten consenting designers to contribute to this contained study. We benchmark leading foundation models against human CAD experts carrying out edits, and find a large performance gap in both automatic metrics and human evaluations. Even the best foundation model (GPT 5.2) scores 53% lower (absolute) than CAD experts in human acceptance trials, demonstrating the challenge of neuralCAD-Edit. We hope neuralCAD-Edit will provide a solid foundation against which 3D CAD editing approaches and foundation models can be developed. Code/data: this https URL
>
---
#### [new 044] Do Vision-Language Models Truly Perform Vision Reasoning? A Rigorous Study of the Modality Gap
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型研究，旨在解决VLM是否真正进行视觉推理的问题。通过构建CrossMath基准，发现VLM主要依赖文本推理，视觉信息利用不足，并提出训练集提升性能。**

- **链接: [https://arxiv.org/pdf/2604.16256](https://arxiv.org/pdf/2604.16256)**

> **作者:** Yige Xu; Yongjie Wang; Zizhuo Wu; Kaisong Song; Jun Lin; Zhiqi Shen
>
> **摘要:** Reasoning in vision-language models (VLMs) has recently attracted significant attention due to its broad applicability across diverse downstream tasks. However, it remains unclear whether the superior performance of VLMs stems from genuine vision-grounded reasoning or relies predominantly on the reasoning capabilities of their textual backbones. To systematically measure this, we introduce CrossMath, a novel multimodal reasoning benchmark designed for controlled cross-modal comparisons. Specifically, we construct each problem in text-only, image-only, and image+text formats guaranteeing identical task-relevant information, verified by human annotators. This rigorous alignment effectively isolates modality-specific reasoning differences while eliminating confounding factors such as information mismatch. Extensive evaluation of state-of-the-art VLMs reveals a consistent phenomenon: a substantial performance gap between textual and visual reasoning. Notably, VLMs excel with text-only inputs, whereas incorporating visual data (image+text) frequently degrades performance compared to the text-only baseline. These findings indicate that current VLMs conduct reasoning primarily in the textual space, with limited genuine reliance on visual evidence. To mitigate this limitation, we curate a CrossMath training set for VLM fine-tuning. Empirical evaluations demonstrate that fine-tuning on this training set significantly boosts reasoning performance across all individual and joint modalities, while yielding robust gains on two general visual reasoning tasks. Source code is available at this https URL.
>
---
#### [new 045] A Two-Stage, Object-Centric Deep Learning Framework for Robust Exam Cheating Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于考试作弊检测任务，旨在解决传统监考效率低、易出错的问题。提出一个两阶段深度学习框架，结合目标检测与行为分析，提升检测准确性和可扩展性。**

- **链接: [https://arxiv.org/pdf/2604.16234](https://arxiv.org/pdf/2604.16234)**

> **作者:** Van-Truong Le; Le-Khanh Nguyen; Trong-Doanh Nguyen
>
> **备注:** 7 pages, 5 figures. Accepted at the FISU Joint Conference on Artificial Intelligence (FJCAI 2026), Vietnam
>
> **摘要:** Academic integrity continues to face the persistent challenge of examination cheating. Traditional invigilation relies on human observation, which is inefficient, costly, and prone to errors at scale. Although some existing AI-powered monitoring systems have been deployed and trusted, many lack transparency or require multi-layered architectures to achieve the desired performance. To overcome these challenges, we propose an improvement over a simple two-stage framework for exam cheating detection that integrates object detection and behavioral analysis using well-known technologies. First, the state-of-the-art YOLOv8n model is used to localize students in exam-room images. Each detected region is cropped and preprocessed, then classified by a fine-tuned RexNet-150 model as either normal or cheating behavior. The system is trained on a dataset compiled from 10 independent sources with a total of 273,897 samples, achieving 0.95 accuracy, 0.94 recall, 0.96 precision, and 0.95 F1-score - a 13\% increase over a baseline accuracy of 0.82 in video-based cheating detection. In addition, with an average inference time of 13.9 ms per sample, the proposed approach demonstrates robustness and scalability for deployment in large-scale environments. Beyond the technical contribution, the AI-assisted monitoring system also addresses ethical concerns by ensuring that final outcomes are delivered privately to individual students after the examination, for example, via personal email. This prevents public exposure or shaming and offers students an opportunity to reflect on their behavior. For further improvement, it is possible to incorporate additional factors, such as audio data and consecutive frames, to achieve greater accuracy. This study provides a foundation for developing real-time, scalable, ethical, and open-source solutions.
>
---
#### [new 046] Find, Fix, Reason: Context Repair for Video Reasoning
- **分类: cs.CV**

- **简介: 该论文属于视频推理任务，旨在解决多模态模型在视频理解中的上下文不足问题。通过引入教师指导学生补充关键证据，提升模型准确性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.16243](https://arxiv.org/pdf/2604.16243)**

> **作者:** Haojian Huang; Chuanyu Qin; Yinchuan Li; Yingcong Chen
>
> **备注:** 22 pages, 7 figures, 17 tables. Ongoing work
>
> **摘要:** Reinforcement learning has advanced video reasoning in large multi-modal models, yet dominant pipelines either rely on on-policy self-exploration, which plateaus at the model's knowledge boundary, or hybrid replay that mixes policies and demands careful regularization. Dynamic context methods zoom into focused evidence but often require curated pretraining and two-stage tuning, and their context remains bounded by a small model's capability. In contrast, larger models excel at instruction following and multi-modal understanding, can supply richer context to smaller models, and rapidly zoom in on target regions via simple tools. Building on this capability, we introduce an observation-level intervention: a frozen, tool-integrated teacher identifies the missing spatiotemporal dependency and provides a minimal evidence patch (e.g., timestamps, regions etc.) from the original video while the question remains unchanged. The student answers again with the added context, and training updates with a chosen-rollout scheme integrated into Group Relative Policy Optimization (GRPO). We further propose a Robust Improvement Reward (RIR) that aligns optimization with two goals: outcome validity through correct answers and dependency alignment through rationales that reflect the cited evidence. Advantages are group-normalized across the batch, preserving on-policy exploration while directing it along causally meaningful directions with minimal changes to the training stack. Experiments on various related benchmarks show consistent accuracy gains and strong generalization. Web page and source code will be available at this https URL.
>
---
#### [new 047] Towards Realistic Open-Vocabulary Remote Sensing Segmentation: Benchmark and Baseline
- **分类: cs.CV**

- **简介: 该论文属于开放词汇遥感图像分割任务，旨在解决数据分散、评估基准不足的问题。提出OVRSISBenchV2基准和Pi-Seg模型，提升分割性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.15652](https://arxiv.org/pdf/2604.15652)**

> **作者:** Bingyu Li; Tao Huo; Haocheng Dong; Da Zhang; Zhiyuan Zhao; Junyu Gao; Xuelong Li
>
> **摘要:** Open-vocabulary remote sensing image segmentation (OVRSIS) remains underexplored due to fragmented datasets, limited training diversity, and the lack of evaluation benchmarks that reflect realistic geospatial application demands. Our previous \textit{OVRSISBenchV1} established an initial cross-dataset evaluation protocol, but its limited scope is insufficient for assessing realistic open-world generalization. To address this issue, we propose \textit{OVRSISBenchV2}, a large-scale and application-oriented benchmark for OVRSIS. We first construct \textbf{OVRSIS95K}, a balanced dataset of about 95K image--mask pairs covering 35 common semantic categories across diverse remote sensing scenes. Built upon OVRSIS95K and 10 downstream datasets, OVRSISBenchV2 contains 170K images and 128 categories, substantially expanding scene diversity, semantic coverage, and evaluation difficulty. Beyond standard open-vocabulary segmentation, it further includes downstream protocols for building extraction, road extraction, and flood detection, thereby better reflecting realistic geospatial application demands and complex deployment scenarios. We also propose \textbf{Pi-Seg}, a baseline for OVRSIS. Pi-Seg improves transferability through a \textbf{positive-incentive noise} mechanism, where learnable and semantically guided perturbations broaden the visual-text feature space during training. Extensive experiments on OVRSISBenchV1, OVRSISBenchV2, and downstream tasks show that Pi-Seg delivers strong and consistent results, particularly on the more challenging OVRSISBenchV2 benchmark. Our results highlight both the importance of realistic benchmark design and the effectiveness of perturbation-based transfer for OVRSIS. The code and datasets are available at \href{this https URL}{LiBingyu01/RSKT-Seg/tree/Pi-Seg}.
>
---
#### [new 048] FineCog-Nav: Integrating Fine-grained Cognitive Modules for Zero-shot Multimodal UAV Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出FineCog-Nav，解决零样本多模态无人机导航问题。通过细粒度认知模块整合，提升导航准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2604.16298](https://arxiv.org/pdf/2604.16298)**

> **作者:** Dian Shao; Zhengzheng Xu; Peiyang Wang; Like Liu; Yule Wang; Jieqi Shi; Jing Huo
>
> **备注:** Accepted by CVPR 2026 Findings
>
> **摘要:** UAV vision-language navigation (VLN) requires an agent to navigate complex 3D environments from an egocentric perspective while following ambiguous multi-step instructions over long horizons. Existing zero-shot methods remain limited, as they often rely on large base models, generic prompts, and loosely coordinated modules. In this work, we propose FineCog-Nav, a top-down framework inspired by human cognition that organizes navigation into fine-grained modules for language processing, perception, attention, memory, imagination, reasoning, and decision-making. Each module is driven by a moderate-sized foundation model with role-specific prompts and structured input-output protocols, enabling effective collaboration and improved interpretability. To support fine-grained evaluation, we construct AerialVLN-Fine, a curated benchmark of 300 trajectories derived from AerialVLN, with sentence-level instruction-trajectory alignment and refined instructions containing explicit visual endpoints and landmark references. Experiments show that FineCog-Nav consistently outperforms zero-shot baselines in instruction adherence, long-horizon planning, and generalization to unseen environments. These results suggest the effectiveness of fine-grained cognitive modularization for zero-shot aerial navigation. Project page: this https URL.
>
---
#### [new 049] Breakout-picker: Reducing false positives in deep learning-based borehole breakout characterization from acoustic image logs
- **分类: cs.CV; cs.SD; physics.geo-ph**

- **简介: 该论文属于地质图像分析任务，旨在解决深度学习在钻孔崩落识别中误报率高的问题。通过引入负样本和对称性验证，降低误报率，提升识别准确性。**

- **链接: [https://arxiv.org/pdf/2604.16011](https://arxiv.org/pdf/2604.16011)**

> **作者:** Guangyu Wang; Xiaodong Ma; Xinming Wu
>
> **摘要:** Borehole breakouts are stress-induced spalling on the borehole wall, which are identifiable in acoustic image logs as paired zones with near-symmetry azimuths, low acoustic amplitudes, and increased borehole radius. Accurate breakout characterization is crucial for in-situ stress analysis. In recent years, deep learning has been introduced to automate the time-consuming and labor-intensive breakout picking process. However, existing approaches often suffer from misclassification of non-breakout features, leading to high false positive rates. To address this limitation, this study develops a deep learning framework, termed Breakout-picker, with a specific focus on reducing false positives in automatic breakout characterization. Breakout-picker reduces false positives through two strategies. First, the training of Breakout-picker incorporates negative samples of non-breakout features, including natural fractures, keyseats, and logging artifacts. They share similar characteristics with breakouts, such as low acoustic amplitude or locally enlarged borehole radius. These negative training samples enables Breakout-picker to better discriminate true breakouts and similar non-breakout features. Second, candidate breakouts identified by Breakout-picker are further validated by azimuthal symmetry criteria, whereby detections that do not exhibit the near-symmetry characteristics of breakout azimuth are excluded. The performance of Breakout-picker is evaluated using three acoustic image log datasets from different regions. The results demonstrate that Breakout-picker outperforms other automatic methods with higher accuracy and substantially lower false positive rates. By reducing false positives, Breakout-picker enhances the reliability of automatic breakout characterization from acoustic image logs, which in turn benefits in-situ stress analysis based on borehole breakouts.
>
---
#### [new 050] GAViD: A Large-Scale Multimodal Dataset for Context-Aware Group Affect Recognition from Videos
- **分类: cs.CV**

- **简介: 该论文属于群体情感识别任务，旨在解决真实场景下群体情感建模困难的问题。提出GAViD数据集和CAGNet模型，以提升多模态情境感知的群体情感识别效果。**

- **链接: [https://arxiv.org/pdf/2604.16214](https://arxiv.org/pdf/2604.16214)**

> **作者:** Deepak Kumar; Abhishek Pratap Singh; Puneet Kumar; Xiaobai Li; Balasubramanian Raman
>
> **摘要:** Understanding affective dynamics in real-world social systems is fundamental to modeling and analyzing human-human interactions in complex environments. Group affect emerges from intertwined human-human interactions, contextual influences, and behavioral cues, making its quantitative modeling a challenging computational social systems problem. However, computational modeling of group affect in in-the-wild scenarios remains challenging due to limited large-scale annotated datasets and the inherent complexity of multimodal social interactions shaped by contextual and behavioral variability. The lack of comprehensive datasets annotated with multimodal and contextual information further limits advances in the field. To address this, we introduce the Group Affect from ViDeos (GAViD) dataset, comprising 5091 video clips with multimodal data (video, audio and context), annotated with ternary valence and discrete emotion labels and enriched with VideoGPT-generated contextual metadata and human-annotated action cues. We also present Context-Aware Group Affect Recognition Network (CAGNet) for multimodal context-aware group affect recognition. CAGNet achieves 63.20\% test accuracy on GAViD, comparable to state-of-the-art performance. The dataset and code are available at this http URL.
>
---
#### [new 051] Ranking XAI Methods for Head and Neck Cancer Outcome Prediction
- **分类: cs.CV; physics.data-an**

- **简介: 该论文属于医疗AI可解释性研究，旨在评估和排名XAI方法以提升HNC预后预测的可解释性。工作包括全面比较13种XAI方法在24个指标上的表现。**

- **链接: [https://arxiv.org/pdf/2604.16034](https://arxiv.org/pdf/2604.16034)**

> **作者:** Baoqiang Ma; Djennifer K. Madzia-Madzou; Rosa C.J. Kraaijveld; Jin Ouyang
>
> **备注:** 4-page conference paper, accepted at IEEE ISBI 2026 (International Symposium on Biomedical Imaging)
>
> **摘要:** For head and neck cancer (HNC) patients, prognostic outcome prediction can support personalized treatment strategy selection. Improving prediction performance of HNC outcomes has been extensively explored by using advanced artificial intelligence (AI) techniques on PET/CT data. However, the interpretability of AI remains a critical obstacle for its clinical adoption. Unlike previous HNC studies that empirically selected explainable AI (XAI) techniques, we are the first to comprehensively evaluate and rank 13 XAI methods across 24 metrics, covering faithfulness, robustness, complexity and plausibility. Experimental results on the multi-center HECKTOR challenge dataset show large variations across evaluation aspects among different XAI methods, with Integrated Gradients (IG) and DeepLIFT (DL) consistently obtained high rankings for faithfulness, complexity and plausibility. This work highlights the importance of comprehensive XAI method evaluation and can be extended to other medical imaging tasks.
>
---
#### [new 052] Beyond Text Prompts: Precise Concept Erasure through Text-Image Collaboration
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于文本图像生成中的概念擦除任务，旨在解决模型生成不当内容的问题。提出TICoE框架，通过文本图像协作实现精准概念删除，保持内容完整性。**

- **链接: [https://arxiv.org/pdf/2604.15829](https://arxiv.org/pdf/2604.15829)**

> **作者:** Jun Li; Lizhi Xiong; Ziqiang Li; Weiwei Jiang; Zhangjie Fu; Yong Li; Guo-Sen Xie
>
> **备注:** 25 pages, accepted by CVPR 2026
>
> **摘要:** Text-to-image generative models have achieved impressive fidelity and diversity, but can inadvertently produce unsafe or undesirable content due to implicit biases embedded in large-scale training datasets. Existing concept erasure methods, whether text-only or image-assisted, face trade-offs: textual approaches often fail to fully suppress concepts, while naive image-guided methods risk over-erasing unrelated content. We propose TICoE, a text-image Collaborative Erasing framework that achieves precise and faithful concept removal through a continuous convex concept manifold and hierarchical visual representation learning. TICoE precisely removes target concepts while preserving unrelated semantic and visual content. To objectively assess the quality of erasure, we further introduce a fidelity-oriented evaluation strategy that measures post-erasure usability. Experiments on multiple benchmarks show that TICoE surpasses prior methods in concept removal precision and content fidelity, enabling safer, more controllable text-to-image generation. Our code is available at this https URL
>
---
#### [new 053] MambaBack: Bridging Local Features and Global Contexts in Whole Slide Image Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于病理图像分析任务，解决WSI分析中局部特征与全局上下文融合的问题。提出MambaBack架构，结合Mamba和Gated CNN优势，提升模型性能并降低内存占用。**

- **链接: [https://arxiv.org/pdf/2604.15729](https://arxiv.org/pdf/2604.15729)**

> **作者:** Sicheng Chen; Chad Wong; Tianyi Zhang; Enhui Chai; Zeyu Liu; Fei Xia
>
> **摘要:** Whole Slide Image (WSI) analysis is pivotal in computational pathology, enabling cancer diagnosis by integrating morphological and architectural cues across magnifications. Multiple Instance Learning (MIL) serves as the standard framework for WSI analysis. Recently, Mamba has become a promising backbone for MIL, overtaking Transformers due to its efficiency and global context modeling capabilities originating from Natural Language Processing (NLP). However, existing Mamba-based MIL approaches face three critical challenges: (1) disruption of 2D spatial locality during 1D sequence flattening; (2) sub-optimal modeling of fine-grained local cellular structures; and (3) high memory peaks during inference on resource-constrained edge devices. Studies like MambaOut reveal that Mamba's SSM component is redundant for local feature extraction, where Gated CNNs suffice. Recognizing that WSI analysis demands both fine-grained local feature extraction akin to natural images, and global context modeling akin to NLP, we propose MambaBack, a novel hybrid architecture that harmonizes the strengths of Mamba and MambaOut. First, we propose the Hilbert sampling strategy to preserve the 2D spatial locality of tiles within 1D sequences, enhancing the model's spatial perception. Second, we design a hierarchical structure comprising a 1D Gated CNN block based on MambaOut to capture local cellular features, and a BiMamba2 block to aggregate global context, jointly enhancing multi-scale representation. Finally, we implement an asymmetric chunking design, allowing parallel processing during training and chunking-streaming accumulation during inference, minimizing peak memory usage for deployment. Experimental results on five datasets demonstrate that MambaBack outperforms seven state-of-the-art methods. Source code and datasets are publicly available.
>
---
#### [new 054] DINOv3 Beats Specialized Detectors: A Simple Foundation Model Baseline for Image Forensics
- **分类: cs.CV**

- **简介: 该论文属于图像取证任务，旨在解决虚假图像定位问题。提出基于DINOv3的简单基线模型，通过LoRA适配和轻量解码器，提升检测效果并增强泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.16083](https://arxiv.org/pdf/2604.16083)**

> **作者:** Jieming Yu; Qiuxiao Feng; Zhuohan Wang; Xiaochen Ma
>
> **备注:** Technical report
>
> **摘要:** With the rapid advancement of deep generative models, realistic fake images have become increasingly accessible, yet existing localization methods rely on complex designs and still struggle to generalize across manipulation types and imaging conditions. We present a simple but strong baseline based on DINOv3 with LoRA adaptation and a lightweight convolutional decoder. Under the CAT-Net protocol, our best model improves average pixel-level F1 by 17.0 points over the previous state of the art on four standard benchmarks using only 9.1\,M trainable parameters on top of a frozen ViT-L backbone, and even our smallest variant surpasses all prior specialized methods. LoRA consistently outperforms full fine-tuning across all backbone scales. Under the data-scarce MVSS-Net protocol, LoRA reaches an average F1 of 0.774 versus 0.530 for the strongest prior method, while full fine-tuning becomes highly unstable, suggesting that pre-trained representations encode forensic information that is better preserved than overwritten. The baseline also exhibits strong robustness to Gaussian noise, JPEG re-compression, and Gaussian blur. We hope this work can serve as a reliable baseline for the research community and a practical starting point for future image-forensic applications. Code is available at this https URL.
>
---
#### [new 055] Repurposing 3D Generative Model for Autoregressive Layout Generation
- **分类: cs.CV**

- **简介: 该论文提出LaviGen，用于3D布局生成任务，解决传统方法依赖文本描述的问题，直接在3D空间中生成符合物理约束的场景。**

- **链接: [https://arxiv.org/pdf/2604.16299](https://arxiv.org/pdf/2604.16299)**

> **作者:** Haoran Feng; Yifan Niu; Zehuan Huang; Yang-Tian Sun; Chunchao Guo; Yuxin Peng; Lu Sheng
>
> **备注:** this https URL
>
> **摘要:** We introduce LaviGen, a framework that repurposes 3D generative models for 3D layout generation. Unlike previous methods that infer object layouts from textual descriptions, LaviGen operates directly in the native 3D space, formulating layout generation as an autoregressive process that explicitly models geometric relations and physical constraints among objects, producing coherent and physically plausible 3D scenes. To further enhance this process, we propose an adapted 3D diffusion model that integrates scene, object, and instruction information and employs a dual-guidance self-rollout distillation mechanism to improve efficiency and spatial accuracy. Extensive experiments on the LayoutVLM benchmark show LaviGen achieves superior 3D layout generation performance, with 19% higher physical plausibility than the state of the art and 65% faster computation. Our code is publicly available at this https URL.
>
---
#### [new 056] UniEditBench: A Unified and Cost-Effective Benchmark for Image and Video Editing via Distilled MLLMs
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出UniEditBench，解决图像和视频编辑评估不统一的问题，通过轻量级模型实现高效、多维度评估。**

- **链接: [https://arxiv.org/pdf/2604.15871](https://arxiv.org/pdf/2604.15871)**

> **作者:** Lifan Jiang; Tianrun Wu; Yuhang Pei; Chenyang Wang; Boxi Wu; Deng Cai
>
> **摘要:** The evaluation of visual editing models remains fragmented across methods and modalities. Existing benchmarks are often tailored to specific paradigms, making fair cross-paradigm comparisons difficult, while video editing lacks reliable evaluation benchmarks. Furthermore, common automatic metrics often misalign with human preference, yet directly deploying large multimodal models (MLLMs) as evaluators incurs prohibitive computational and financial costs. We present UniEditBench, a unified benchmark for image and video editing that supports reconstruction-based and instruction-driven methods under a shared protocol. UniEditBench includes a structured taxonomy of nine image operations (Add, Remove, Replace, Change, Stroke-based, Extract, Adjust, Count, Reorder) and eight video operations, with coverage of challenging compositional tasks such as counting and spatial reordering. To enable scalable evaluation, we distill a high-capacity MLLM judge (Qwen3-VL-235B-A22B Instruct) into lightweight 4B/8B evaluators that provide multi-dimensional scoring over structural fidelity, text alignment, background consistency, naturalness, and temporal-spatial consistency (for videos). Experiments show that the distilled evaluators maintain strong agreement with human judgments and substantially reduce deployment cost relative to the teacher model. UniEditBench provides a practical and reproducible protocol for benchmarking modern visual editing methods. Our benchmark and the associated reward models are publicly available at this https URL.
>
---
#### [new 057] SSMamba: A Self-Supervised Hybrid State Space Model for Pathological Image Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于病理图像分类任务，针对ROI分析中的域移位、局部全局建模不足和细粒度敏感性差的问题，提出SSMamba框架，通过三个模块提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.15711](https://arxiv.org/pdf/2604.15711)**

> **作者:** Enhui Chai; Sicheng Chen; Tianyi Zhang; Xingyu Li; Tianxiang Cui
>
> **摘要:** Pathological diagnosis is highly reliant on image analysis, where Regions of Interest (ROIs) serve as the primary basis for diagnostic evidence, while whole-slide image (WSI)-level tasks primarily capture aggregated patterns. To extract these critical morphological features, ROI-level Foundation Models (FMs) based on Vision Transformers (ViTs) and large-scale self-supervised learning (SSL) have been widely adopted. However, three core limitations remain in their application to ROI analysis: (1) cross-magnification domain shift, as fixed-scale pretraining hinders adaptation to diverse clinical settings; (2) inadequate local-global relationship modeling, wherein the ViT backbone of FMs suffers from high computational overhead and imprecise local characterization; (3) insufficient fine-grained sensitivity, as traditional self-attention mechanisms tend to overlook subtle diagnostic cues. To address these challenges, we propose SSMamba, a hybrid SSL framework that enables effective fine-grained feature learning without relying on large external datasets. This framework incorporates three domain-adaptive components: Mamba Masked Image Modeling (MAMIM) for mitigating domain shift, a Directional Multi-scale (DMS) module for balanced local-global modeling, and a Local Perception Residual (LPR) module for enhanced fine-grained sensitivity. Employing a two-stage pipeline, SSL pretraining on target ROI datasets followed by supervised fine-tuning (SFT), SSMamba outperforms 11 state-of-the-art (SOTA) pathological FMs on 10 public ROI datasets and surpasses 8 SOTA methods on 6 public WSI datasets. These results validate the superiority of task-specific architectural designs for pathological image analysis.
>
---
#### [new 058] Stylistic-STORM (ST-STORM) : Perceiving the Semantic Nature of Appearance
- **分类: cs.CV; cs.AI; cs.LG; stat.ML**

- **简介: 该论文提出ST-STORM框架，解决自监督学习中区分内容与风格的问题。通过分离内容与风格流，提升对关键外观特征的感知能力，应用于图像分类和医学图像分析。**

- **链接: [https://arxiv.org/pdf/2604.16086](https://arxiv.org/pdf/2604.16086)**

> **作者:** Hamed Ouattara; Pierre Duthon; Pascal Houssam Salmane; Frédéric Bernardin; Omar Ait Aider
>
> **备注:** 20 pages, 16 figures, ICPR 2026 (28th International Conference on Pattern Recognition)
>
> **摘要:** One of the dominant paradigms in self-supervised learning (SSL), illustrated by MoCo or DINO, aims to produce robust representations by capturing features that are insensitive to certain image transformations such as illumination, or geometric changes. This strategy is appropriate when the objective is to recognize objects independently of their appearance. However, it becomes counterproductive as soon as appearance itself constitutes the discriminative signal. In weather analysis, for example, rain streaks, snow granularity, atmospheric scattering, as well as reflections and halos, are not noise: they carry the essential information. In critical applications such as autonomous driving, ignoring these cues is risky, since grip and visibility depend directly on ground conditions and atmospheric conditions. We introduce ST-STORM, a hybrid SSL framework that treats appearance (style) as a semantic modality to be disentangled from content. Our architecture explicitly separates two latent streams, regulated by gating mechanisms. The Content branch aims at a stable semantic representation through a JEPA scheme coupled with a contrastive objective, promoting invariance to appearance variations. In parallel, the Style branch is constrained to capture appearance signatures (textures, contrasts, scattering) through feature prediction and reconstruction under an adversarial constraint. We evaluate ST-STORM on several tasks, including object classification (ImageNet-1K), fine-grained weather characterization, and melanoma detection (ISIC 2024 Challenge). The results show that the Style branch effectively isolates complex appearance phenomena (F1=97% on Multi-Weather and F1=94% on ISIC 2024 with 10% labeled data), without degrading the semantic performance (F1=80% on ImageNet-1K) of the Content branch, and improves the preservation of critical appearance
>
---
#### [new 059] Watching Movies Like a Human: Egocentric Emotion Understanding for Embodied Companions
- **分类: cs.CV**

- **简介: 该论文属于电影情感理解任务，解决机器人在第一视角观看电影时的情感识别问题。提出ESE数据集并构建多模态长上下文模型，提升真实场景下的情感理解性能。**

- **链接: [https://arxiv.org/pdf/2604.15823](https://arxiv.org/pdf/2604.15823)**

> **作者:** Ze Dong; Hao Shi; Zejia Gao; Zhonghua Yi; Kaiwei Wang; Lin Wang
>
> **备注:** 15 pages
>
> **摘要:** Embodied robotic agents often perceive movies through an egocentric screen-view interface rather than native cinematic footage, introducing domain shifts such as viewpoint distortion, scale variation, illumination changes, and environmental interference. However, existing research on movie emotion understanding is almost exclusively conducted on cinematic footage, limiting cross-domain generalization to real-world viewing scenarios. To bridge this gap, we introduce EgoScreen-Emotion (ESE), the first benchmark dataset for egocentric screen-view movie emotion understanding. ESE contains 224 movie trailers captured under controlled egocentric screen-view conditions, producing 28,667 temporally aligned key-frames annotated by multiple raters with a confidence-aware multi-label protocol to address emotional ambiguity. We further build a multimodal long-context emotion reasoning framework that models temporal visual evidence, narrative summaries, compressed historical context, and audio cues. Cross-domain experiments reveal a severe domain gap: models trained on cinematic footage drop from 27.99 to 16.69 Macro-F1 when evaluated on realistic egocentric screen-view observations. Training on ESE substantially improves robustness under realistic viewing conditions. Our approach achieves competitive performance compared with strong closed-source multimodal models, highlighting the importance of domain-specific data and long-context multimodal reasoning.
>
---
#### [new 060] CPU Optimization of a Monocular 3D Biomechanics Pipeline for Low-Resource Deployment
- **分类: cs.CV; cs.PF**

- **简介: 该论文属于计算机视觉与生物力学交叉任务，旨在解决GPU依赖导致的低资源部署难题。通过优化模型结构和并行计算，提升CPU运行效率，实现高效、准确的3D运动分析。**

- **链接: [https://arxiv.org/pdf/2604.15665](https://arxiv.org/pdf/2604.15665)**

> **作者:** Yan Zhang; Xiong Zhao
>
> **摘要:** Markerless 3D movement analysis from monocular video enables accessible biomechanical assessment in clinical and sports settings. However, most research-grade pipelines rely on GPU acceleration, limiting deployment on consumer-grade hardware and in low-resource environments. In this work, we optimize a monocular 3D biomechanics pipeline derived from the MonocularBiomechanics framework for efficient CPU-only execution. Through profiling-driven system optimization, including model initialization restructuring, elimination of disk I/O serialization, and improved CPU parallelization. Experiments on a consumer workstation (AMD Ryzen 7 9700X CPU) show a 2.47x increase in processing throughput and a 59.6\% reduction in total runtime, with initialization latency reduced by 4.6x. Despite these changes, biomechanical outputs remain highly consistent with the baseline implementation (mean joint-angle deviation 0.35$^\circ$, $r=0.998$). These results demonstrate that research-grade vision-based biomechanics pipelines can be deployed on commodity CPU hardware for scalable movement assessment.
>
---
#### [new 061] Efficient Video Diffusion Models: Advancements and Challenges
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决视频扩散模型推理成本高的问题。通过分类分析四种优化方法，提出降低计算开销和提升效率的策略。**

- **链接: [https://arxiv.org/pdf/2604.15911](https://arxiv.org/pdf/2604.15911)**

> **作者:** Shitong Shao; Lichen Bai; Pengfei Wan; James Kwok; Zeke Xie
>
> **摘要:** Video diffusion models have rapidly become the dominant paradigm for high-fidelity generative video synthesis, but their practical deployment remains constrained by severe inference costs. Compared with image generation, video synthesis compounds computation across spatial-temporal token growth and iterative denoising, making attention and memory traffic major bottlenecks in real-world settings. This survey provides a systematic and deployment-oriented review of efficient video diffusion models. We propose a unified categorization that organizes existing methods into four classes of main paradigms, including step distillation, efficient attention, model compression, and cache/trajectory optimization. Building on this categorization, we respectively analyze algorithmic trends of these four paradigms and examine how different design choices target two core objectives: reducing the number of function evaluations and minimizing per-step overhead. Finally, we discuss open challenges and future directions, including quality preservation under composite acceleration, hardware-software co-design, robust real-time long-horizon generation, and open infrastructure for standardized evaluation. To the best of our knowledge, our work is the first comprehensive survey on efficient video diffusion models, offering researchers and engineers a structured overview of the field and its emerging research directions.
>
---
#### [new 062] CXR-LT 2026 Challenge: Multi-Center Long-Tailed and Zero Shot Chest X-ray Classification
- **分类: cs.CV**

- **简介: 该论文属于医学图像分类任务，旨在解决长尾分布和开放世界下的胸片识别问题。通过构建多中心数据集，提出两个核心任务，评估模型在罕见疾病和新发现上的泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.15555](https://arxiv.org/pdf/2604.15555)**

> **作者:** Hexin Dong; Yi Lin; Pengyu Zhou; Fengnian Zhao; Alan Clint Legasto; Juno Cho; Dohui Kim; Justin Namuk Kim; Mingeon Kim; Sunwoo Kwak; Gabriel Moyà-Alcover; Ky Trung Nguyen; Thanh-Huy Nguyen; Ha-Hieu Pham; Huy-Hieu Pham; Huy Le Pham; Nikhileswara Rao Sulake; Aina Tur-Serrano; Ruichi Zhang; Ang Zu; Adam E. Flanders; Zhiyong Lu; Ronald M. Summers; Mingquan Lin; Hao Chen; Yuzhe Yang; George Shih; Yifan Peng
>
> **备注:** 25 pages, 6 figures
>
> **摘要:** Chest X-ray (CXR) interpretation is hindered by the long-tailed distribution of pathologies and the open-world nature of clinical environments. Existing benchmarks often rely on closed-set classes from a single institution, failing to capture the prevalence of rare diseases or the appearance of novel findings. To address this, we present the CXR-LT challenge. The first event, CXR-LT 2023, established a large-scale benchmark for long-tailed multi-label CXR classification and identified key challenges in rare disease recognition. CXR-LT 2024 further expanded the label space and introduced a zero-shot task to study generalization to unseen findings. Building on the success of CXR-LT 2023 and 2024, this third iteration of the benchmark introduces a multi-center dataset comprising over 145,000 images from PadChest and NIH Chest X-ray datasets. Additionally, all development and test sets in CXR-LT 2026 are annotated by radiologists, providing a more reliable and clinically grounded evaluation than report-derived labels. The challenge defines two core tasks this year: (1) Robust Multi-Label Classification on 30 known classes and (2) Open-World Generalization to 6 unseen (out-of-distribution) rare disease classes. This paper summarizes the overview of the CXR-LT 2026 challenge. We describe the data collection and annotation procedures, analyze solution strategies adopted by participating teams, and evaluate head-versus-tail performance, calibration, and cross-center generalization gaps. Our results show that vision-language foundation models improve both in-distribution and zero-shot performance, but detecting rare findings under multi-center shift remains challenging. Our study provides a foundation for developing and evaluating AI systems in realistic long-tailed and open-world clinical conditions.
>
---
#### [new 063] Beyond a Single Frame: Multi-Frame Spatially Grounded Reasoning Across Volumetric MRI
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SGMRI-VQA任务，解决医学影像中多帧空间推理问题。构建了包含41,307对数据的基准，要求模型在多帧MRI中准确定位和描述病变。**

- **链接: [https://arxiv.org/pdf/2604.15808](https://arxiv.org/pdf/2604.15808)**

> **作者:** Lama Moukheiber; Caleb M. Yeung; Haotian Xue; Alec Helbling; Zelin Zhao; Yongxin Chen
>
> **摘要:** Spatial reasoning and visual grounding are core capabilities for vision-language models (VLMs), yet most medical VLMs produce predictions without transparent reasoning or spatial evidence. Existing benchmarks also evaluate VLMs on isolated 2D images, overlooking the volumetric nature of clinical imaging, where findings can span multiple frames or appear on only a few slices. We introduce Spatially Grounded MRI Visual Question Answering (SGMRI-VQA), a 41,307-pair benchmark for multi-frame, spatially grounded reasoning on volumetric MRI. Built from expert radiologist annotations in the fastMRI+ dataset across brain and knee studies, each QA pair includes a clinician-aligned chain-of-thought trace with frame-indexed bounding box coordinates. Tasks are organized hierarchically across detection, localization, counting/classification, and captioning, requiring models to jointly reason about what is present, where it is, and across which frames it extends. We benchmark 10 VLMs and show that supervised fine-tuning of Qwen3-VL-8B with bounding box supervision consistently improves grounding performance over strong zero-shot baselines, indicating that targeted spatial supervision is an effective path toward grounded clinical reasoning.
>
---
#### [new 064] Zoom Consistency: A Free Confidence Signal in Multi-Step Visual Grounding Pipelines
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉接地任务，解决多步Zoom管道中中间预测信息未被充分利用的问题。提出zoom consistency作为免费置信信号，用于评估预测准确性。**

- **链接: [https://arxiv.org/pdf/2604.15376](https://arxiv.org/pdf/2604.15376)**

> **作者:** Keon Kim; Krish Chelikavada
>
> **摘要:** Multi-step zoom-in pipelines are widely used for GUI grounding, yet the intermediate predictions they produce are typically discarded after coordinate remapping. We observe that these intermediate outputs contain a useful confidence signal for free: zoom consistency, the distance between a model's step-2 prediction and the crop center. Unlike log-probabilities or token-level uncertainty, zoom consistency is a geometric quantity in a shared coordinate space, making it directly comparable across architecturally different VLMs without calibration. We prove this quantity is a linear estimator of step-1 spatial error under idealized conditions (perfect step-2, target within crop) and show it correlates with prediction correctness across two VLMs (AUC = 0.60; Spearman rho = -0.14, p < 10^{-6} for KV-Ground-8B; rho = -0.11, p = 0.0003 for Qwen3.5-27B). The correlation is small but consistent across models, application categories, and operating systems. As a proof-of-concept, we use zoom consistency to route between a specialist and generalist model, capturing 16.5% of the oracle headroom between them (+0.8%, McNemar p = 0.19). Code is available at this https URL.
>
---
#### [new 065] AdaVFM: Adaptive Vision Foundation Models for Edge Intelligence via LLM-Guided Execution
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出AdaVFM，解决边缘设备上语言对齐视觉基础模型的高效推理问题。通过动态调整计算，提升准确率与效率。**

- **链接: [https://arxiv.org/pdf/2604.15622](https://arxiv.org/pdf/2604.15622)**

> **作者:** Yiwei Zhao; Yi Zheng; Huapeng Su; Jieyu Lin; Stefano Ambrogio; Cijo Jose; Michaël Ramamonjisoa; Patrick Labatut; Barbara De Salvo; Chiao Liu; Phillip B. Gibbons; Ziyun Li
>
> **摘要:** Language-aligned vision foundation models (VFMs) enable versatile visual understanding for always-on contextual AI, but their deployment on edge devices is hindered by strict latency and power constraints. We present AdaVFM, an adaptive framework for efficient on-device inference of language-aligned VFMs that dynamically adjusts computation based on scene context and task complexity. Our key insight is that the effect of model size reduction on performance is task-dependent in vision applications, motivating a runtime-adaptive execution strategy. AdaVFM integrates neural architecture search (NAS) into the language-aligned VFM backbone to enable lightweight subnet execution during runtime. A multimodal large language model (LLM) deployed on the cloud enables runtime control with a context-aware agent. This synergy allows efficient model adaptation under diverse conditions while maintaining strong accuracy. Extensive experiments on zero-shot classification and open-vocabulary segmentation demonstrate that AdaVFM achieves state-of-the-art accuracy-efficiency trade-offs, surpassing prior baselines by up to $7.9\%$ in acc@1 on IN1K and $5.2\%$ mIoU on ADE20K over the best models of comparable VFM sizes. For models with similar accuracy, AdaVFM further reduces average FLOPs by up to $77.9\%$.
>
---
#### [new 066] Self-Supervised Angular Deblurring in Photoacoustic Reconstruction via Noisier2Inverse
- **分类: cs.CV**

- **简介: 该论文属于图像重建任务，解决有限尺寸探测器导致的图像模糊问题。提出一种无需真实数据的自监督方法，通过角度去模糊提升图像质量。**

- **链接: [https://arxiv.org/pdf/2604.15681](https://arxiv.org/pdf/2604.15681)**

> **作者:** Markus Haltmeier; Nadja Gruber; Gyeongha Hwang
>
> **摘要:** Photoacoustic tomography (PAT) is an emerging imaging modality that combines the complementary strengths of optical contrast and ultrasonic resolution. A central task is image reconstruction, where measured acoustic signals are used to recover the initial pressure distribution. For ideal point-like or line-like detectors, several efficient and fast reconstruction algorithms exist, including Fourier methods, filtered backprojection, and time reversal. However, when applied to data acquired with finite-size detectors, these methods yield systematically blurred images. Although sharper images can be obtained by compensating for finite-detector effects, supervised learning approaches typically require ground-truth images that may not be available in practice. We propose a self-supervised reconstruction method based on Noisier2Inverse that addresses finite-size detector effects without requiring ground-truth data. Our approach operates directly on noisy measurements and learns to recover high-quality PAT images in a ground-truth-free manner. Its key components are: (i) PAT-specific modeling that recasts the problem as angular deblurring; (ii) a Noisier2Inverse formulation in the polar domain that leverages the known angular point-spread function; and (iii) a novel, statistically grounded early-stopping rule. In experiments, the proposed method consistently outperforms alternative approaches that do not use supervised data and achieves performance close to supervised benchmarks, while remaining practical for real acquisitions with finite-size detectors.
>
---
#### [new 067] Neural Gabor Splatting: Enhanced Gaussian Splatting with Neural Gabor for High-frequency Surface Reconstruction
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于3D重建任务，旨在解决高频率表面重建中高斯点数量激增的问题。通过引入神经Gabor方法和频率感知策略，提升重建精度与效率。**

- **链接: [https://arxiv.org/pdf/2604.15941](https://arxiv.org/pdf/2604.15941)**

> **作者:** Haato Watanabe; Nobuyuki Umetani
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Recent years have witnessed the rapid emergence of 3D Gaussian splatting (3DGS) as a powerful approach for 3D reconstruction and novel view synthesis. Its explicit representation with Gaussian primitives enables fast training, real-time rendering, and convenient post-processing such as editing and surface reconstruction. However, 3DGS suffers from a critical drawback: the number of primitives grows drastically for scenes with high-frequency appearance details, since each primitive can represent only a single color, requiring multiple primitives for every sharp color transition. To overcome this limitation, we propose neural Gabor splatting, which augments each Gaussian primitive with a lightweight multi-layer perceptron that models a wide range of color variations within a single primitive. To further control primitive numbers, we introduce a frequency-aware densification strategy that selects mismatch primitives for pruning and cloning based on frequency energy. Our method achieves accurate reconstruction of challenging high-frequency surfaces. We demonstrate its effectiveness through extensive experiments on both standard benchmarks, such as Mip-NeRF360 and High-Frequency datasets (e.g., checkered patterns), supported by comprehensive ablation studies.
>
---
#### [new 068] CLIMB: Controllable Longitudinal Brain Image Generation using Mamba-based Latent Diffusion Model and Gaussian-aligned Autoencoder
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出CLIMB模型，用于可控的纵向脑影像生成，解决脑结构随时间变化的建模问题，通过结合状态空间模型和高斯对齐自编码器，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2604.15611](https://arxiv.org/pdf/2604.15611)**

> **作者:** Duy-Phuong Dao; Muhammad Taqiyuddin; Jahae Kim; Sang-Heon Lee; Hye-Won Jung; Jaehoo Choi; Hyung-Jeong Yang
>
> **备注:** 18 pages, 5 figures, 5 tables
>
> **摘要:** Latent diffusion models have emerged as powerful generative models in medical imaging, enabling the synthesis of high quality brain magnetic resonance imaging scans. In particular, predicting the evolution of a patients brain can aid in early intervention, prognosis, and treatment planning. In this study, we introduce CLIMB, Controllable Longitudinal brain Image generation via state space based latent diffusion model, an advanced framework for modeling temporal changes in brain structure. CLIMB is designed to model the structural evolution of the brain structure over time, utilizing a baseline MRI scan and its acquisition age as foundational inputs. Additionally, multiple conditional variables, including projected age, gender, disease status, genetic information, and brain structure volumes, are incorporated to enhance the temporal modeling of anatomical changes. Unlike existing LDM methods that rely on self attention modules, which effectively capture contextual information from input images but are computationally expensive, our approach leverages state space, a state space model architecture that substantially reduces computational overhead while preserving high-quality image synthesis. Furthermore, we introduce a Gaussian-aligned autoencoder that extracts latent representations conforming to prior distributions without the sampling noise inherent in conventional variational autoencoders. We train and evaluate our proposed model on the Alzheimers Disease Neuroimaging Initiative dataset, consisting of 6,306 MRI scans from 1,390 participants. By comparing generated images with real MRI scans, CLIMB achieves a structural similarity index of 0.9433, demonstrating notable improvements over existing methods.
>
---
#### [new 069] CLOTH-HUGS: Cloth Aware Human Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文提出Cloth-HUGS，用于真实感穿衣人体重建，解决衣物与身体融合困难的问题。通过分离身体和衣物的高斯层，提升衣物细节与动态一致性。**

- **链接: [https://arxiv.org/pdf/2604.15875](https://arxiv.org/pdf/2604.15875)**

> **作者:** Sadia Mubashshira; Nazanin Amini; Kevin Desai
>
> **摘要:** We present Cloth-HUGS, a Gaussian Splatting based neural rendering framework for photorealistic clothed human reconstruction that explicitly disentangles body and clothing. Unlike prior methods that absorb clothing into a single body representation and struggle with loose garments and complex deformations, Cloth-HUGS represents the performer using separate Gaussian layers for body and cloth within a shared canonical space. The canonical volume jointly encodes body, cloth, and scene primitives and is deformed through SMPL-driven articulation with learned linear blend skinning weights. To improve cloth realism, we initialize cloth Gaussians from mesh topology and apply physics-inspired constraints, including simulation-consistency, ARAP regularization, and mask supervision. We further introduce a depth-aware multi-pass rendering strategy for robust body-cloth-scene compositing, enabling real-time rendering at over 60 FPS. Experiments on multiple benchmarks show that Cloth-HUGS improves perceptual quality and geometric fidelity over state-of-the-art baselines, reducing LPIPS by up to 28% while producing temporally coherent cloth dynamics.
>
---
#### [new 070] MMGait: Towards Multi-Modal Gait Recognition
- **分类: cs.CV**

- **简介: 该论文属于多模态步态识别任务，旨在解决单一模态不足和跨模态检索问题。提出MMGait基准及OmniGait模型，实现多模态统一识别。**

- **链接: [https://arxiv.org/pdf/2604.15979](https://arxiv.org/pdf/2604.15979)**

> **作者:** Chenye Wang; Qingyuan Cai; Saihui Hou; Aoqi Li; Yongzhen Huang
>
> **备注:** CVPR 2026
>
> **摘要:** Gait recognition has emerged as a powerful biometric technique for identifying individuals at a distance without requiring user cooperation. Most existing methods focus primarily on RGB-derived modalities, which fall short in real-world scenarios requiring multi-modal collaboration and cross-modal retrieval. To overcome these challenges, we present MMGait, a comprehensive multi-modal gait benchmark integrating data from five heterogeneous sensors, including an RGB camera, a depth camera, an infrared camera, a LiDAR scanner, and a 4D Radar system. MMGait contains twelve modalities and 334,060 sequences from 725 subjects, enabling systematic exploration across geometric, photometric, and motion domains. Based on MMGait, we conduct extensive evaluations on single-modal, cross-modal, and multi-modal paradigms to analyze modality robustness and complementarity. Furthermore, we introduce a new task, Omni Multi-Modal Gait Recognition, which aims to unify the above three gait recognition paradigms within a single model. We also propose a simple yet powerful baseline, OmniGait, which learns a shared embedding space across diverse modalities and achieves promising recognition performance. The MMGait benchmark, codebase, and pretrained checkpoints are publicly available at this https URL.
>
---
#### [new 071] Elucidating the SNR-t Bias of Diffusion Probabilistic Models
- **分类: cs.CV**

- **简介: 该论文属于生成模型领域，针对扩散概率模型中的SNR-t偏差问题进行研究，提出一种差分修正方法以提升生成质量。**

- **链接: [https://arxiv.org/pdf/2604.16044](https://arxiv.org/pdf/2604.16044)**

> **作者:** Meng Yu; Lei Sun; Jianhao Zeng; Xiangxiang Chu; Kun Zhan
>
> **备注:** Accepted to CVPR 2026, 19pages, with appendix
>
> **摘要:** Diffusion Probabilistic Models have demonstrated remarkable performance across a wide range of generative tasks. However, we have observed that these models often suffer from a Signal-to-Noise Ratio-timestep (SNR-t) bias. This bias refers to the misalignment between the SNR of the denoising sample and its corresponding timestep during the inference phase. Specifically, during training, the SNR of a sample is strictly coupled with its timestep. However, this correspondence is disrupted during inference, leading to error accumulation and impairing the generation quality. We provide comprehensive empirical evidence and theoretical analysis to substantiate this phenomenon and propose a simple yet effective differential correction method to mitigate the SNR-t bias. Recognizing that diffusion models typically reconstruct low-frequency components before focusing on high-frequency details during the reverse denoising process, we decompose samples into various frequency components and apply differential correction to each component individually. Extensive experiments show that our approach significantly improves the generation quality of various diffusion models (IDDPM, ADM, DDIM, A-DPM, EA-DPM, EDM, PFGM++, and FLUX) on datasets of various resolutions with negligible computational overhead. The code is at this https URL.
>
---
#### [new 072] HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning
- **分类: cs.CV**

- **简介: 该论文属于少样本持续学习任务，解决跨领域数据不平衡导致的原型漂移问题。提出HyCal方法，通过几何特性稳定原型，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.15678](https://arxiv.org/pdf/2604.15678)**

> **作者:** Eunju Lee; MiHyeon Kim; JuneHyoung Kwon; Yoonji Lee; JiHyun Kim; Soojin Jang; YoungBin Kim
>
> **备注:** Accepted to CVPR 2026. Eunju Lee and MiHyeon Kim contributed equally as co-first authors
>
> **摘要:** Pretrained Vision-Language Models (VLMs) like CLIP show promise in continual learning, but existing Few-Shot Class-Incremental Learning (FSCIL) methods assume homogeneous domains and balanced data distributions, limiting real-world applicability where data arises from heterogeneous disciplines with imbalanced sample availability and varying visual complexity. We identify Domain Gravity, a representational asymmetry where data imbalance across heterogeneous domains causes overrepresented or low-entropy domains to disproportionately influence the embedding space, leading to prototype drift and degraded performance on underrepresented or high-entropy domains. To address this, we introduce Cross-Discipline Variable Few-Shot Class-Incremental Learning (XD-VSCIL), a benchmark capturing real-world heterogeneity and imbalance where Domain Gravity naturally intensifies. We propose Hybrid Prototype Calibration (HyCal), a training-free method combining cosine similarity and Mahalanobis distance to capture complementary geometric properties-directional alignment and covariance-aware magnitude-yielding stable prototypes under imbalanced heterogeneous conditions. Operating on frozen CLIP embeddings, HyCal achieves consistent retention-adaptation improvements while maintaining efficiency. Experiments show HyCal effectively mitigates Domain Gravity and outperforms existing methods in imbalanced cross-domain incremental learning.
>
---
#### [new 073] PLAF: Pixel-wise Language-Aligned Feature Extraction for Efficient 3D Scene Understanding
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D场景理解任务，旨在解决语义表示在2D与3D间语言对齐和冗余问题。提出PLAF框架，实现像素级语义对齐与高效存储查询。**

- **链接: [https://arxiv.org/pdf/2604.15770](https://arxiv.org/pdf/2604.15770)**

> **作者:** Junjie Wen; Junlin He; Fei Ma; Jinqiang Cui
>
> **备注:** Accepted by ICCA 2026
>
> **摘要:** Accurate open-vocabulary 3D scene understanding requires semantic representations that are both language-aligned and spatially precise at the pixel level, while remaining scalable when lifted to 3D space. However, existing representations struggle to jointly satisfy these requirements, and densely propagating pixel-wise semantics to 3D often results in substantial redundancy, leading to inefficient storage and querying in large-scale scenes. To address these challenges, we present \emph{PLAF}, a Pixel-wise Language-Aligned Feature extraction framework that enables dense and accurate semantic alignment in 2D without sacrificing open-vocabulary expressiveness. Building upon this representation, we further design an efficient semantic storage and querying scheme that significantly reduces redundancy across both 2D and 3D domains. Experimental results show that \emph{PLAF} provides a strong semantic foundation for accurate and efficient open-vocabulary 3D scene understanding. The codes are publicly available at this https URL.
>
---
#### [new 074] SENSE: Stereo OpEN Vocabulary SEmantic Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于开放词汇语义分割任务，解决单视角图像精度不足的问题。通过引入立体视觉和视觉语言模型，提升分割准确性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.15946](https://arxiv.org/pdf/2604.15946)**

> **作者:** Thomas Campagnolo; Ezio Malis; Philippe Martinet; Gaétan Bahl
>
> **摘要:** Open-vocabulary semantic segmentation enables models to segment objects or image regions beyond fixed class sets, offering flexibility in dynamic environments. However, existing methods often rely on single-view images and struggle with spatial precision, especially under occlusions and near object boundaries. We propose SENSE, the first work on Stereo OpEN Vocabulary SEmantic Segmentation, which leverages stereo vision and vision-language models to enhance open-vocabulary semantic segmentation. By incorporating stereo image pairs, we introduce geometric cues that improve spatial reasoning and segmentation accuracy. Trained on the PhraseStereo dataset, our approach achieves strong performance in phrase-grounded tasks and demonstrates generalization in zero-shot settings. On PhraseStereo, we show a +2.9% improvement in Average Precision over the baseline method and +0.76% over the best competing method. SENSE also provides a relative improvement of +3.5% mIoU on Cityscapes and +18% on KITTI compared to the baseline work. By jointly reasoning over semantics and geometry, SENSE supports accurate scene understanding from natural language, essential for autonomous robots and Intelligent Transportation Systems.
>
---
#### [new 075] APC: Transferable and Efficient Adversarial Point Counterattack for Robust 3D Point Cloud Recognition
- **分类: cs.CV**

- **简介: 该论文属于3D点云识别任务，旨在解决对抗攻击带来的模型脆弱性问题。提出APC方法，通过输入级净化实现高效且可迁移的防御。**

- **链接: [https://arxiv.org/pdf/2604.15708](https://arxiv.org/pdf/2604.15708)**

> **作者:** Geunyoung Jung; Soohong Kim; Inseok Kong; Jiyoung Jung
>
> **备注:** Accepted by CVPR 2026 Findings
>
> **摘要:** The advent of deep neural networks has led to remarkable progress in 3D point cloud recognition, but they remain vulnerable to adversarial attacks. Although various defense methods have been studied, they suffer from a trade-off between robustness and transferability. We propose Adversarial Point Counterattack (APC) to achieve both simultaneously. APC is a lightweight input-level purification module that generates instance-specific counter-perturbations for each point, effectively neutralizing attacks. Leveraging clean-adversarial pairs, APC enforces geometric consistency in data space and semantic consistency in feature space. To improve generalizability across diverse attacks, we adopt a hybrid training strategy using adversarial point clouds from multiple attack types. Since APC operates purely on input point clouds, it directly transfers to unseen models and defends against attacks targeting them without retraining. At inference, a single APC forward pass provides purified point clouds with negligible time and parameter overhead. Extensive experiments on two 3D recognition benchmarks demonstrate that the APC achieves state-of-the-art defense performance. Furthermore, cross-model evaluations validate its superior transferability. The code is available at this https URL.
>
---
#### [new 076] From Zero to Detail: A Progressive Spectral Decoupling Paradigm for UHD Image Restoration with New Benchmark
- **分类: cs.CV**

- **简介: 该论文属于UHD图像修复任务，旨在解决高分辨率、多样化内容和细节恢复难题。提出分阶段的频谱解耦框架ERR，包含三个子网络，提升修复效果。**

- **链接: [https://arxiv.org/pdf/2604.15654](https://arxiv.org/pdf/2604.15654)**

> **作者:** Chen Zhao; Yunzhe Xu; Zhizhou Chen; Enxuan Gu; Kai Zhang; Xiaoming Liu; Jian Yang; Ying Tai
>
> **备注:** TPAMI
>
> **摘要:** Ultra-high-definition (UHD) image restoration poses unique challenges due to the high spatial resolution, diverse content, and fine-grained structures present in UHD images. To address these issues, we introduce a progressive spectral decomposition for the restoration process, decomposing it into three stages: zero-frequency \textbf{enhancement}, low-frequency \textbf{restoration}, and high-frequency \textbf{refinement}. Based on this formulation, we propose a novel framework, \textbf{ERR}, which integrates three cooperative sub-networks: the zero-frequency enhancer (ZFE), the low-frequency restorer (LFR), and the high-frequency refiner (HFR). The ZFE incorporates global priors to learn holistic mappings, the LFR reconstructs the main content by focusing on coarse-scale information, and the HFR adopts our proposed frequency-windowed Kolmogorov-Arnold Network (FW-KAN) to recover fine textures and intricate details for high-fidelity restoration. To further advance research in UHD image restoration, we also construct a large-scale, high-quality benchmark dataset, \textbf{LSUHDIR}, comprising 82{,}126 UHD images with diverse scenes and rich content. Our proposed methods demonstrate superior performance across a range of UHD image restoration tasks, and extensive ablation studies confirm the contribution and necessity of each module. Project page: this https URL.
>
---
#### [new 077] Motion-Adapter: A Diffusion Model Adapter for Text-to-Motion Generation of Compound Actions
- **分类: cs.CV**

- **简介: 该论文属于文本到动作生成任务，旨在解决复合动作生成中的时间信息处理和注意力机制问题。提出Motion-Adapter模块，通过解耦交叉注意力图提升生成效果。**

- **链接: [https://arxiv.org/pdf/2604.16135](https://arxiv.org/pdf/2604.16135)**

> **作者:** Yue Jiang; Mingyu Yang; Liuyuxin Yang; Yang Xu; Bingxin Yun; Yuhe Zhang
>
> **备注:** 12 pages, 12 figures, Under review for publication in IEEE Transactions on Visualization and Computer Graphics
>
> **摘要:** Recent advances in generative motion synthesis have enabled the production of realistic human motions from diverse input modalities. However, synthesizing compound actions from texts, which integrate multiple concurrent actions into coherent full-body sequences, remains a major challenge. We identify two key limitations in current text-to-motion diffusion models: (i) catastrophic neglect, where earlier actions are overwritten by later ones due to improper handling of temporal information, and (ii) attention collapse, which arises from excessive feature fusion in cross-attention mechanisms. As a result, existing approaches often depend on overly detailed textual descriptions (e.g., raising right hand), explicit body-part specifications (e.g., editing the upper body), or the use of large language models (LLMs) for body-part interpretation. These strategies lead to deficient semantic representations of physical structures and kinematic mechanisms, limiting the ability to incorporate natural behaviors such as greeting while walking. To address these issues, we propose the Motion-Adapter, a plug-and-play module that guides text-to-motion diffusion models in generating compound actions by computing decoupled cross-attention maps, which serve as structural masks during the denoising process. Extensive experiments demonstrate that our method consistently produces more faithful and coherent compound motions across diverse textual prompts, surpassing state-of-the-art approaches.
>
---
#### [new 078] AIFIND: Artifact-Aware Interpreting Fine-Grained Alignment for Incremental Face Forgery Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于增量人脸伪造检测任务，解决特征漂移和灾难性遗忘问题。提出AIFIND方法，通过语义锚点稳定学习，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2604.16207](https://arxiv.org/pdf/2604.16207)**

> **作者:** Hao Wang; Beichen Zhang; Yanpei Gong; Shaoyi Fang; Zhaobo Qi; Yuanrong Xu; Xinyan Liu; Weigang Zhang
>
> **摘要:** As forgery types continue to emerge consistently, Incremental Face Forgery Detection (IFFD) has become a crucial paradigm. However, existing methods typically rely on data replay or coarse binary supervision, which fails to explicitly constrain the feature space, leading to severe feature drift and catastrophic forgetting. To address this, we propose AIFIND, Artifact-Aware Interpreting Fine-Grained Alignment for Incremental Face Forgery Detection, which leverages semantic anchors to stabilize incremental learning. We design the Artifact-Driven Semantic Prior Generator to instantiate invariant semantic anchors, establishing a fixed coordinate system from low-level artifact cues. These anchors are injected into the image encoder via Artifact-Probe Attention, which explicitly constrains volatile visual features to align with stable semantic anchors. Adaptive Decision Harmonizer harmonizes the classifiers by preserving angular relationships of semantic anchors, maintaining geometric consistency across tasks. Extensive experiments on multiple incremental protocols validate the superiority of AIFIND.
>
---
#### [new 079] DenTab: A Dataset for Table Recognition and Visual QA on Real-World Dental Estimates
- **分类: cs.CV**

- **简介: 该论文提出DenTab数据集，用于表格识别和视觉问答任务，解决真实场景下表格结构与语义理解问题。**

- **链接: [https://arxiv.org/pdf/2604.16099](https://arxiv.org/pdf/2604.16099)**

> **作者:** Laziz Hamdi; Amine Tamasna; Thierry Paquet
>
> **摘要:** Tables condense key transactional and administrative information into compact layouts, but practical extraction requires more than text recognition: systems must also recover structure (rows, columns, merged cells, headers) and interpret roles such as line items, subtotals, and totals under common capture artifacts. Many existing resources for table structure recognition and TableVQA are built from clean digital-born sources or rendered tables, and therefore only partially reflect noisy administrative conditions. We introduce DenTab, a dataset of 2{,}000 cropped table images from dental estimates with high-quality HTML annotations, enabling evaluation of table recognition (TR) and table visual question answering (TableVQA) on the same inputs. DenTab includes 2{,}208 questions across eleven categories spanning retrieval, aggregation, and logic/consistency checks. We benchmark 16 systems, including 14 vision--language models (VLMs) and two OCR baselines. Across models, strong structure recovery does not consistently translate into reliable performance on multi-step arithmetic and consistency questions, and these reasoning failures persist even when using ground-truth HTML table inputs. To improve arithmetic reliability without training, we propose the Table Router Pipeline, which routes arithmetic questions to deterministic execution. The pipeline combines (i) a VLM that produces a baseline answer, a structured table representation, and a constrained table program with (ii) a rule-based executor that performs exact computation over the parsed table. The source code and dataset will be made publicly available at this https URL.
>
---
#### [new 080] Polyglot: Multilingual Style Preserving Speech-Driven Facial Animation
- **分类: cs.CV**

- **简介: 该论文属于语音驱动面部动画任务，解决多语言环境下个性化表达问题。提出Polyglot模型，结合语言和风格嵌入，实现跨语言、跨说话者的逼真动画生成。**

- **链接: [https://arxiv.org/pdf/2604.16108](https://arxiv.org/pdf/2604.16108)**

> **作者:** Federico Nocentini; Kwanggyoon Seo; Qingju Liu; Claudio Ferrari; Stefano Berretti; David Ferman; Hyeongwoo Kim; Pablo Garrido; Akin Caliskan
>
> **备注:** The project website is available at this https URL
>
> **摘要:** Speech-Driven Facial Animation (SDFA) has gained significant attention due to its applications in movies, video games, and virtual reality. However, most existing models are trained on single-language data, limiting their effectiveness in real-world multilingual scenarios. In this work, we address multilingual SDFA, which is essential for realistic generation since language influences phonetics, rhythm, intonation, and facial expressions. Speaking style is also shaped by individual differences, not only by language. Existing methods typically rely on either language-specific or speaker-specific conditioning, but not both, limiting their ability to model their interaction. We introduce Polyglot, a unified diffusion-based architecture for personalized multilingual SDFA. Our method uses transcript embeddings to encode language information and style embeddings extracted from reference facial sequences to capture individual speaking characteristics. Polyglot does not require predefined language or speaker labels, enabling generalization across languages and speakers through self-supervised learning. By jointly conditioning on language and style, it captures expressive traits such as rhythm, articulation, and habitual facial movements, producing temporally coherent and realistic animations. Experiments show improved performance in both monolingual and multilingual settings, providing a unified framework for modeling language and personal style in SDFA.
>
---
#### [new 081] (1D) Ordered Tokens Enable Efficient Test-Time Search
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于生成模型任务，研究如何通过测试时搜索提升生成效果。针对传统二维网格结构在搜索中的局限性，提出使用一维有序分层令牌结构，提升搜索效率与生成质量。**

- **链接: [https://arxiv.org/pdf/2604.15453](https://arxiv.org/pdf/2604.15453)**

> **作者:** Zhitong Gao; Parham Rezaei; Ali Cy; Mingqiao Ye; Nataša Jovanović; Jesse Allardice; Afshin Dehghan; Amir Zamir; Roman Bachmann; Oğuzhan Fatih Kar
>
> **备注:** Project page: this https URL
>
> **摘要:** Tokenization is a key component of autoregressive (AR) generative models, converting raw data into more manageable units for modeling. Commonly, tokens describe local information, such as regions of pixels in images or word pieces in text, and AR generation predicts these tokens in a fixed order. A worthwhile question is whether token structures affect the ability to steer the generation through test-time search, where multiple candidate generations are explored and evaluated by a verifier. Using image generation as our testbed, we hypothesize that recent 1D ordered tokenizers with coarse-to-fine structure can be more amenable to search than classical 2D grid structures. This is rooted in the fact that the intermediate states in coarse-to-fine sequences carry semantic meaning that verifiers can reliably evaluate, enabling effective steering during generation. Through controlled experiments, we find that AR models trained on coarse-to-fine ordered tokens exhibit improved test-time scaling behavior compared to grid-based counterparts. Moreover, we demonstrate that, thanks to the ordered structure, pure test-time search over token sequences (i.e., without training an AR model) can perform training-free text-to-image generation when guided by an image-text verifier. Beyond this, we systematically study how classical search algorithms (best-of-N, beam search, lookahead search) interact with different token structures, as well as the role of different verifiers and AR priors. Our results highlight the impact of token structure on inference-time scalability and provide practical guidance for test-time scaling in AR models.
>
---
#### [new 082] AeroDeshadow: Physics-Guided Shadow Synthesis and Penumbra-Aware Deshadowing for Aerospace Imagery
- **分类: cs.CV**

- **简介: 该论文属于航空航天图像去阴影任务，解决高分辨率影像中阴影导致的光谱失真和信息丢失问题。提出AeroDeshadow框架，结合物理引导的阴影合成与半影感知恢复。**

- **链接: [https://arxiv.org/pdf/2604.15903](https://arxiv.org/pdf/2604.15903)**

> **作者:** Wei Lu; Zi-Yang Bo; Fei-Fei Sang; Yi Liu; Xue Yang; Si-Bao Chen
>
> **备注:** 13 pages, 12 figures
>
> **摘要:** Shadows are prevalent in high-resolution aerospace imagery (ASI). They often cause spectral distortion and information loss, which degrade downstream interpretation tasks. While deep learning methods have advanced natural-image shadow removal, their direct application to ASI faces two primary challenges. First, strictly paired training data are severely lacking. Second, homogeneous shadow assumptions fail to handle the broad penumbra transition zones inherent in aerospace scenes. To address these issues, we propose AeroDeshadow, a unified two-stage framework integrating physics-guided shadow synthesis and penumbra-aware restoration. In the first stage, a Physics-aware Degradation Shadow Synthesis Network (PDSS-Net) explicitly models illumination decay and spatial attenuation. This process constructs AeroDS-Syn, a large-scale paired dataset featuring soft boundary transitions. Constrained by this physical formulation, a Penumbra-aware Cascaded DeShadowing Network (PCDS-Net) then decouples the input into umbra and penumbra components. By restoring these regions progressively, PCDS-Net alleviates boundary artifacts and over-correction. Trained solely on the synthetic AeroDS-Syn, the network generalizes to real-world ASI without requiring paired real annotations. Experimental results indicate that AeroDeshadow achieves state-of-the-art quantitative accuracy and visual fidelity across synthetic and real-world datasets. The datasets and code will be made publicly available at: this https URL.
>
---
#### [new 083] From Competition to Coopetition: Coopetitive Training-Free Image Editing Based on Text Guidance
- **分类: cs.CV**

- **简介: 该论文属于文本引导的图像编辑任务，解决传统方法因竞争机制导致的语义冲突问题。提出CoEdit框架，通过协同注意力机制提升编辑质量和结构一致性。**

- **链接: [https://arxiv.org/pdf/2604.15948](https://arxiv.org/pdf/2604.15948)**

> **作者:** Jinhao Shen; Haoqian Du; Xulu Zhang; Xiao-Yong Wei; Qing Li
>
> **摘要:** Text-guided image editing, a pivotal task in modern multimedia content creation, has seen remarkable progress with training-free methods that eliminate the need for additional optimization. Despite recent progress, existing methods are typically constrained by a competitive paradigm in which the editing and reconstruction branches are independently driven by their respective objectives to maximize alignment with target and source prompts. The adversarial strategy causes semantic conflicts and unpredictable outcomes due to the lack of coordination between branches. To overcome these issues, we propose Coopetitive Training-Free Image Editing (CoEdit), a novel zero-shot framework that transforms attention control from competition to coopetitive negotiation, achieving editing harmony across spatial and temporal dimensions. Spatially, CoEdit introduces Dual-Entropy Attention Manipulation, which quantifies directional entropic interactions between branches to reformulate attention control as a harmony-maximization problem, eventually improving the localization of editable and preservable regions. Temporally, we present Entropic Latent Refinement mechanism to dynamically adjust latent representations over time, minimizing accumulated editing errors and ensuring consistent semantic transitions throughout the denoising trajectory. Additionally, we propose the Fidelity-Constrained Editing Score, a composite metric that jointly evaluates semantic editing and background fidelity. Extensive experiments on standard benchmarks demonstrate that CoEdit achieves superior performance in both editing quality and structural preservation, enhancing multimedia information utilization by enabling more effective interaction between visual and textual modalities. The code will be available at this https URL.
>
---
#### [new 084] CollideNet: Hierarchical Multi-scale Video Representation Learning with Disentanglement for Time-To-Collision Forecasting
- **分类: cs.CV**

- **简介: 论文提出CollideNet，用于时间到碰撞预测任务，解决视频中多尺度时空模式理解问题，通过层次化Transformer架构和解耦分析提升预测精度。**

- **链接: [https://arxiv.org/pdf/2604.16240](https://arxiv.org/pdf/2604.16240)**

> **作者:** Nishq Poorav Desai; Ali Etemad; Michael Greenspan
>
> **备注:** Accepted to ICPR 2026
>
> **摘要:** Time-to-Collision (TTC) forecasting is a critical task in collision prevention, requiring precise temporal prediction and comprehending both local and global patterns encapsulated in a video, both spatially and temporally. To address the multi-scale nature of video, we introduce a novel spatiotemporal hierarchical transformer-based architecture called CollideNet, specifically catered for effective TTC forecasting. In the spatial stream, CollideNet aggregates information for each video frame simultaneously at multiple resolutions. In the temporal stream, along with multi-scale feature encoding, CollideNet also disentangles the non-stationarity, trend, and seasonality components. Our method achieves state-of-the-art performance in comparison to prior works on three commonly used public datasets, setting a new state-of-the-art by a considerable margin. We conduct cross-dataset evaluations to analyze the generalization capabilities of our method, and visualize the effects of disentanglement of the trend and seasonality components of the video data. We release our code at this https URL.
>
---
#### [new 085] SSFT: A Lightweight Spectral-Spatial Fusion Transformer for Generic Hyperspectral Classification
- **分类: cs.CV**

- **简介: 该论文属于高光谱图像分类任务，旨在解决高维、冗余、数据稀缺等问题。提出轻量级SSFT模型，通过光谱与空间路径融合提升分类性能。**

- **链接: [https://arxiv.org/pdf/2604.15828](https://arxiv.org/pdf/2604.15828)**

> **作者:** Alexander Musiat; Nikolas Ebert; Oliver Wasenmüller
>
> **备注:** This paper has been accepted at IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2026
>
> **摘要:** Hyperspectral imaging enables fine-grained recognition of materials by capturing rich spectral signatures, but learning robust classifiers is challenging due to high dimensionality, spectral redundancy, limited labeled data, and strong domain shifts. Beyond earth observation, labeled HSI data is often scarce and imbalanced, motivating compact models for generic hyperspectral classification across diverse acquisition regimes. We propose the lightweight Spectral-Spatial Fusion Transformer (SSFT), which factorizes representation learning into spectral and spatial pathways and integrates them via cross-attention to capture complementary wavelength-dependent and structural information. We evaluate our SSFT on the challenging HSI-Benchmark, a heterogeneous multi-dataset benchmark covering earth observation, fruit condition assessment, and fine-grained material recognition. SSFT achieves state-of-the-art overall performance, ranking first while using less than 2% of the parameters of the previous leading method. We further evaluate transfer to the substantially larger SpectralEarth benchmark under the official protocol, where SSFT remains competitive despite its compact size. Ablation studies show that both spectral and spatial pathways are crucial, with spatial modeling contributing most, and that SSFT remains robust without data augmentation.
>
---
#### [new 086] Dual-Modal Lung Cancer AI: Interpretable Radiology and Microscopy with Clinical Risk Integration
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于肺癌诊断任务，旨在解决传统CT影像难以区分良恶性病变的问题。通过融合CT与病理图像，并结合临床数据，提升诊断准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2604.16104](https://arxiv.org/pdf/2604.16104)**

> **作者:** Baramee Sukumal; Aueaphum Aueawatthanaphisut
>
> **备注:** 16 pages, 6 figures, 3 tables, 8 equations
>
> **摘要:** Lung cancer remains one of the leading causes of cancer-related mortality worldwide. Conventional computed tomography (CT) imaging, while essential for detection and staging, has limitations in distinguishing benign from malignant lesions and providing interpretable diagnostic insights. To address this challenge, this study proposes a dual-modal artificial intelligence framework that integrates CT radiology with hematoxylin and eosin (H&E) histopathology for lung cancer diagnosis and subtype classification. The system employs convolutional neural networks to extract radiologic and histopathologic features and incorporates clinical metadata to improve robustness. Predictions from both modalities are fused using a weighted decision-level integration mechanism to classify adenocarcinoma, squamous cell carcinoma, large cell carcinoma, small cell lung cancer, and normal tissue. Explainable AI techniques including Grad-CAM, Grad-CAM++, Integrated Gradients, Occlusion, Saliency Maps, and SmoothGrad are applied to provide visual interpretability. Experimental results show strong performance with accuracy up to 0.87, AUROC above 0.97, and macro F1-score of 0.88. Grad-CAM++ achieved the highest faithfulness and localization accuracy, demonstrating strong correspondence with expert-annotated tumor regions. These results indicate that multimodal fusion of radiology and histopathology can improve diagnostic performance while maintaining model transparency, suggesting potential for future clinical decision support systems in precision oncology.
>
---
#### [new 087] GIST: Multimodal Knowledge Extraction and Spatial Grounding via Intelligent Semantic Topology
- **分类: cs.AI; cs.CV; cs.HC; cs.RO**

- **简介: 该论文提出GIST系统，解决复杂环境中AI的语义空间定位问题。通过多模态知识提取，构建语义导航拓扑，提升人机交互任务性能。**

- **链接: [https://arxiv.org/pdf/2604.15495](https://arxiv.org/pdf/2604.15495)**

> **作者:** Shivendra Agrawal; Bradley Hayes
>
> **摘要:** Navigating complex, densely packed environments like retail stores, warehouses, and hospitals poses a significant spatial grounding challenge for humans and embodied AI. In these spaces, dense visual features quickly become stale given the quasi-static nature of items, and long-tail semantic distributions challenge traditional computer vision. While Vision-Language Models (VLMs) help assistive systems navigate semantically-rich spaces, they still struggle with spatial grounding in cluttered environments. We present GIST (Grounded Intelligent Semantic Topology), a multimodal knowledge extraction pipeline that transforms a consumer-grade mobile point cloud into a semantically annotated navigation topology. Our architecture distills the scene into a 2D occupancy map, extracts its topological layout, and overlays a lightweight semantic layer via intelligent keyframe and semantic selection. We demonstrate the versatility of this structured spatial knowledge through critical downstream Human-AI interaction tasks: (1) an intent-driven Semantic Search engine that actively infers categorical alternatives and zones when exact matches fail; (2) a one-shot Semantic Localizer achieving a 1.04 m top-5 mean translation error; (3) a Zone Classification module that segments the walkable floor plan into high-level semantic regions; and (4) a Visually-Grounded Instruction Generator that synthesizes optimal paths into egocentric, landmark-rich natural language routing. In multi-criteria LLM evaluations, GIST outperforms sequence-based instruction generation baselines. Finally, an in-situ formative evaluation (N=5) yields an 80% navigation success rate relying solely on verbal cues, validating the system's capacity for universal design.
>
---
#### [new 088] AstroVLM: Expert Multi-agent Collaborative Reasoning for Astronomical Imaging Quality Diagnosis
- **分类: cs.MA; cs.CV**

- **简介: 该论文属于天文图像质量诊断任务，旨在解决复杂多步骤天文成像中的质量评估与错误定位问题。提出AstroVLM多智能体协作系统进行诊断。**

- **链接: [https://arxiv.org/pdf/2604.16024](https://arxiv.org/pdf/2604.16024)**

> **作者:** Yaohui Han; Tianshuo Wang; Zixi Zhao; Zhengchun Zhu; Shuo Ren; Yiru Wang; Rongliang Fu; Tinghuan Chen; Tsung-Yi Ho
>
> **摘要:** Vision Language Models (VLMs) have been applied to several specific domains and have shown strong problem-solving capabilities. However, astronomical imaging, a quite complex problem involving multidisciplinary knowledge and several subtasks, has not been adequately studied. Due to the complexity of the astronomical imaging process, both world-class astronomical organizations, such as NASA, and expert enthusiasts devote a great deal of time and effort. This is because the processes in astronomical imaging have complex underlying correlations that significantly influence one another, making the quality diagnosis and error localization of astronomical images challenging. To address this problem, we propose AstroVLM, a collaborative multi-agent system for diagnosing the quality of astronomical images. Experiment results show that AstroVLM outperforms all baselines on real-world astronomical imaging quality diagnosis tasks, providing a reference for language models to handle complicated multi-process tasks.
>
---
#### [new 089] Hierarchical Active Inference using Successor Representations
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于强化学习任务，旨在解决复杂环境下的规划问题。通过结合层次化主动推理与后继表示，提升抽象状态和动作的学习效率。**

- **链接: [https://arxiv.org/pdf/2604.15679](https://arxiv.org/pdf/2604.15679)**

> **作者:** Prashant Rangarajan; Rajesh P. N. Rao
>
> **备注:** Accepted for publication in Neural Computation (MIT Press). 82 pages, 29 figures
>
> **摘要:** Active inference, a neurally-inspired model for inferring actions based on the free energy principle (FEP), has been proposed as a unifying framework for understanding perception, action, and learning in the brain. Active inference has previously been used to model ecologically important tasks such as navigation and planning, but scaling it to solve complex large-scale problems in real-world environments has remained a challenge. Inspired by the existence of multi-scale hierarchical representations in the brain, we propose a model for planning of actions based on hierarchical active inference. Our approach combines a hierarchical model of the environment with successor representations for efficient planning. We present results demonstrating (1) how lower-level successor representations can be used to learn higher-level abstract states, (2) how planning based on active inference at the lower-level can be used to bootstrap and learn higher-level abstract actions, and (3) how these learned higher-level abstract states and actions can facilitate efficient planning. We illustrate the performance of the approach on several planning and reinforcement learning (RL) problems including a variant of the well-known four rooms task, a key-based navigation task, a partially observable planning problem, the Mountain Car problem, and PointMaze, a family of navigation tasks with continuous state and action spaces. Our results represent, to our knowledge, the first application of learned hierarchical state and action abstractions to active inference in FEP-based theories of brain function.
>
---
#### [new 090] AEGIS: Anchor-Enforced Gradient Isolation for Knowledge-Preserving Vision-Language-Action Fine-Tuning
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出AEGIS方法，解决预训练视觉-语言模型在机器人控制微调中的梯度冲突问题，通过正交投影保持视觉问答能力。**

- **链接: [https://arxiv.org/pdf/2604.16067](https://arxiv.org/pdf/2604.16067)**

> **作者:** Guransh Singh
>
> **摘要:** Adapting pre-trained vision-language models (VLMs) for robotic control requires injecting high-magnitude continuous gradients from a flow-matching action expert into a backbone trained exclusively with cross-entropy. This cross-modal gradient asymmetry - the spectral dimensionality mismatch between low-rank MSE regression gradients and the high-dimensional semantic manifold sculpted by CE pre-training, causes rapid, severe erosion of the VLM's visual-question-answering (VQA) capability. Industry-standard defences either sever the gradient pathway entirely via stop gradient, discarding the rich continuous supervision, or restrict parameter capacity through low-rank adapters (LoRA) that constrain the rank of updates but not their direction, and thus still overwrite the pre-trained manifold. We introduce AEGIS (Anchor-Enforced Gradient Isolation System): a buffer-free, layer-wise orthogonal gradient projection framework that enables direct continuous MSE learning while preserving the pre-trained VQA manifold - without any co-training data or replay buffer. AEGIS pre-computes a static Gaussian reference anchor from masked VQA forward passes across all transformer layers, then at each training step constructs a Wasserstein-2 transport penalty that generates an anchor restoration gradient. A sequential dual-backward decomposes the task and anchor gradients; for each transformer layer, AEGIS applies a single Gram-Schmidt orthogonal projection that bends the task gradient away from the destructive direction while preserving its constructive content. The projection sheds less than 1% of gradient energy on average, yet eliminates the cumulative activation drift that drives severe forgetting.
>
---
#### [new 091] M3R: Localized Rainfall Nowcasting with Meteorology-Informed MultiModal Attention
- **分类: cs.LG; cs.CV; cs.MM**

- **简介: 该论文属于降雨短临预测任务，旨在解决多源气象数据融合难题。提出M3R模型，结合雷达图像与气象站数据，提升预测精度与效率。**

- **链接: [https://arxiv.org/pdf/2604.15377](https://arxiv.org/pdf/2604.15377)**

> **作者:** Sanjeev Panta; Rhett M Morvant; Xu Yuan; Li Chen; Nian-Feng Tzeng
>
> **备注:** Accepted at IEEE International Conference on Multimedia and Expo (ICME) 2026
>
> **摘要:** Accurate and timely rainfall nowcasting is crucial for disaster mitigation and water resource management. Despite recent advances in deep learning, precipitation prediction remains challenging due to limitations in effectively leveraging diverse multimedia data sources. We introduce M3R, a Meteorology-informed MultiModal attention-based architecture for direct Rainfall prediction that synergistically combines visual NEXRAD radar imagery with numerical Personal Weather Station (PWS) measurements, using a comprehensive pipeline for temporal alignment of heterogeneous meteorological data. With specialized multimodal attention mechanisms, M3R novelly leverages weather station time series as queries to selectively attend to spatial radar features, enabling focused extraction of precipitation signatures. Experimental results for three spatial areas of 100 km * 100 km centered at NEXRAD radar stations demonstrate that M3R outperforms existing approaches, achieving substantial improvements in accuracy, efficiency, and precipitation detection capabilities. Our work establishes new benchmarks for multimedia-based precipitation nowcasting and provides practical tools for operational weather prediction systems. The source code is available at this https URL
>
---
#### [new 092] Learning Affine-Equivariant Proximal Operators
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于信号处理任务，解决如何构建具有平移和尺度等变性的学习近似算子问题。通过设计等变神经网络，提升近似算子的鲁棒性与实用性。**

- **链接: [https://arxiv.org/pdf/2604.15556](https://arxiv.org/pdf/2604.15556)**

> **作者:** Oriel Savir; Zhenghan Fang; Jeremias Sulam
>
> **备注:** 9 pages, 4 figures, Accepted at ICASSP 2026
>
> **摘要:** Proximal operators are fundamental across many applications in signal processing and machine learning, including solving ill-posed inverse problems. Recent work has introduced Learned Proximal Networks (LPNs), providing parametric functions that compute exact proximals for data-driven and potentially non-convex regularizers. However, in many settings it is important to include additional structure to these regularizers--and their corresponding proximals--such as shift and scale equivariance. In this work, we show how to obtain learned functions parametrized by neural networks that provably compute exact proximal operators while being equivariant to shifts and scaling, which we dub Affine-Equivariant Learned Proximal Networks (AE-LPNs). We demonstrate our results on synthetic, constructive examples, and then on real data via denoising in out-of-distribution settings. Our equivariant learned proximals enhance robustness to noise distributions and affine shifts far beyond training distributions, improving the practical utility of learned proximal operators
>
---
#### [new 093] Hierarchical Codec Diffusion for Video-to-Speech Generation
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于视频到语音生成任务，旨在解决现有方法忽视语音层次结构的问题。提出HiCoDiT模型，利用语音离散编码的层次结构，提升音视频对齐效果。**

- **链接: [https://arxiv.org/pdf/2604.15923](https://arxiv.org/pdf/2604.15923)**

> **作者:** Jiaxin Ye; Gaoxiang Cong; Chenhui Wang; Xin-Cheng Wen; Zhaoyang Li; Boyuan Cao; Hongming Shan
>
> **备注:** CVPR 2026
>
> **摘要:** Video-to-Speech (VTS) generation aims to synthesize speech from a silent video without auditory signals. However, existing VTS methods disregard the hierarchical nature of speech, which spans coarse speaker-aware semantics to fine-grained prosodic details. This oversight hinders direct alignment between visual and speech features at specific hierarchical levels during property matching. In this paper, leveraging the hierarchical structure of Residual Vector Quantization (RVQ)-based codec, we propose HiCoDiT, a novel Hierarchical Codec Diffusion Transformer that exploits the inherent hierarchy of discrete speech tokens to achieve strong audio-visual alignment. Specifically, since lower-level tokens encode coarse speaker-aware semantics and higher-level tokens capture fine-grained prosody, HiCoDiT employs low-level and high-level blocks to generate tokens at different levels. The low-level blocks condition on lip-synchronized motion and facial identity to capture speaker-aware content, while the high-level blocks use facial expression to modulate prosodic dynamics. Finally, to enable more effective coarse-to-fine conditioning, we propose a dual-scale adaptive instance layer normalization that jointly captures global vocal style through channel-wise normalization and local prosody dynamics through temporal-wise normalization. Extensive experiments demonstrate that HiCoDiT outperforms baselines in fidelity and expressiveness, highlighting the potential of discrete modelling for VTS. The code and speech demo are both available at this https URL.
>
---
#### [new 094] Topology-Driven Fusion of nnU-Net and MedNeXt for Accurate Brain Tumor Segmentation on Sub-Saharan Africa Dataset
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于脑肿瘤分割任务，旨在解决LMIC地区MRI数据质量差、协议不统一等问题，通过融合模型并引入拓扑优化模块提升分割精度。**

- **链接: [https://arxiv.org/pdf/2604.15964](https://arxiv.org/pdf/2604.15964)**

> **作者:** Prabin Bohara; Pralhad Kumar Shrestha; Arpan Rai; Usha Poudel Lamgade; Confidence Raymond; Dong Zhang; Aondona Lorumbu; Craig Jones; Mahesh Shakya; Bishesh Khanal; Pratibha Kulung
>
> **摘要:** Accurate automatic brain tumor segmentation in Low and Middle-Income (LMIC) countries is challenging due to the lack of defined national imaging protocols, diverse imaging data, extensive use of low-field Magnetic Resonance Imaging (MRI) scanners and limited health-care resources. As part of the Brain Tumor Segmentation (BraTS) Africa 2025 Challenge, we applied topology refinement to the state-of-the-art segmentation models like nnU-Net, MedNeXt, and a combination of both. Since the BraTS-Africa dataset has low MRI image quality, we incorporated the BraTS 2025 challenge data of pre-treatment adult glioma (Task 1) to pre-train the segmentation model and use it to fine-tune on the BraTS-Africa dataset. We added an extra topology refinement module to address the issue of deformation in prediction that arose due to topological error. With the introduction of this module, we achieved a better Normalized Surface Distance (NSD) of 0.810, 0.829, and 0.895 on Surrounding Non-Enhancing FLAIR Hyperintensity (SNFH) , Non-Enhancing Tumor Core (NETC) and Enhancing tumor (ET).
>
---
#### [new 095] MARCH: Multi-Agent Radiology Clinical Hierarchy for CT Report Generation
- **分类: cs.AI; cs.CV**

- **简介: 该论文提出MARCH框架，用于CT报告生成任务，解决临床幻觉和缺乏迭代验证的问题，通过多智能体协作提升报告的临床准确性和语言质量。**

- **链接: [https://arxiv.org/pdf/2604.16175](https://arxiv.org/pdf/2604.16175)**

> **作者:** Yi Lin; Yihao Ding; Yonghui Wu; Yifan Peng
>
> **备注:** Accepted by ACL 2026 main conference
>
> **摘要:** Automated 3D radiology report generation often suffers from clinical hallucinations and a lack of the iterative verification found in human practice. While recent Vision-Language Models (VLMs) have advanced the field, they typically operate as monolithic "black-box" systems without the collaborative oversight characteristic of clinical workflows. To address these challenges, we propose MARCH (Multi-Agent Radiology Clinical Hierarchy), a multi-agent framework that emulates the professional hierarchy of radiology departments and assigns specialized roles to distinct agents. MARCH utilizes a Resident Agent for initial drafting with multi-scale CT feature extraction, multiple Fellow Agents for retrieval-augmented revision, and an Attending Agent that orchestrates an iterative, stance-based consensus discourse to resolve diagnostic discrepancies. On the RadGenome-ChestCT dataset, MARCH significantly outperforms state-of-the-art baselines in both clinical fidelity and linguistic accuracy. Our work demonstrates that modeling human-like organizational structures enhances the reliability of AI in high-stakes medical domains.
>
---
#### [new 096] Automating Crash Diagram Generation Using Vision-Language Models: A Case Study on Multi-Lane Roundabouts
- **分类: cs.HC; cs.AI; cs.CV; cs.SE**

- **简介: 该论文属于交通工程可视化任务，旨在解决 crash diagram 自动生成问题。通过 VLMs 从事故报告中生成 diagrams，提升效率与一致性。**

- **链接: [https://arxiv.org/pdf/2604.15332](https://arxiv.org/pdf/2604.15332)**

> **作者:** Xiao Lu; Hao Zhen; Jidong J. Yang
>
> **备注:** 16 pages, 5 figures, 3 tables
>
> **摘要:** Crash diagrams are essential tools in transportation safety analysis, yet their manual preparation remains time-consuming and prone to human variability. This study investigates the use of Vision-Language Models (VLMs) to automate crash diagram generation from police crash reports, focusing on multilane roundabouts as a challenging test case. A three-part structured prompt framework was developed to guide model reasoning through interpretation, extraction, and visual synthesis, while a 10-metric evaluation system was designed to assess diagram quality in terms of semantic accuracy, spatial fidelity, and visual clarity. Three popular models, including GPT-4o, Gemini-1.5-Flash, and Janus-4o, were tested on 79 crash reports. GPT-4o achieved the highest average performance (6.29 out of 10), followed by Gemini-1.5-Flash (5.28) and Janus-4o (3.64). The analysis revealed GPT-4o's superior spatial reasoning and alignment between extracted and visualized crash data. These results highlight both the promise and current limitations of VLMs in engineering visualization tasks. The study lays the groundwork for integrating generative AI into crash analysis workflows to improve efficiency, consistency, and interpretability.
>
---
#### [new 097] CTSCAN: Evaluation Leakage in Chest CT Segmentation and a Reproducible Patient-Disjoint Benchmark
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，解决训练与测试数据混用导致性能评估虚高的问题，提出CTSCAN基准以实现患者不重叠的评估。**

- **链接: [https://arxiv.org/pdf/2604.15561](https://arxiv.org/pdf/2604.15561)**

> **作者:** Anton Ivchenko
>
> **摘要:** Reported chest CT segmentation performance can be strongly inflated when train and test partitions mix slices from the same study. We present CTSCAN, a reproducible multi-source chest CT benchmark and research stack designed to measure what survives under patient-disjoint evaluation. The current four-class artifact aggregates 89 cases from PleThora, MedSeg SIRM, and LongCIU, and we show that the original slice-PNG workflow induces near-complete case reuse across train, validation, and test. Using the playground environment, we run a multi-seed protocol sweep with the same FPN plus EfficientNet-B0 control configuration under slice-mixed and case-disjoint evaluation. Across 3 seeds and 12 epochs per seed, the slice-mixed protocol reaches 0.6665 foreground Dice and 0.5031 foreground IoU, whereas the case-disjoint protocol reaches 0.2066 Dice and 0.1181 IoU. Removing patient reuse therefore reduces foreground Dice by 0.4599 absolute (69.00% relative) and foreground IoU by 0.3850 absolute (76.52% relative). CTSCAN packages the corrected benchmark with deterministic split manifests, explicit weak-supervision controls, a scripted multi-seed protocol sweep, and reproducible figure generation, providing a reusable basis for patient-disjoint chest CT evaluation.
>
---
#### [new 098] DENALI: A Dataset Enabling Non-Line-of-Sight Spatial Reasoning with Low-Cost LiDARs
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于非视距感知任务，旨在解决低成本LiDAR在隐藏物体感知中的难题。通过构建DENALI数据集，实现数据驱动的NLOS推理。**

- **链接: [https://arxiv.org/pdf/2604.16201](https://arxiv.org/pdf/2604.16201)**

> **作者:** Nikhil Behari; Diego Rivero; Luke Apostolides; Suman Ghosh; Paul Pu Liang; Ramesh Raskar
>
> **摘要:** Consumer LiDARs in mobile devices and robots typically output a single depth value per pixel. Yet internally, they record full time-resolved histograms containing direct and multi-bounce light returns; these multi-bounce returns encode rich non-line-of-sight (NLOS) cues that can enable perception of hidden objects in a scene. However, severe hardware limitations of consumer LiDARs make NLOS reconstruction with conventional methods difficult. In this work, we motivate a complementary direction: enabling NLOS perception with low-cost LiDARs through data-driven inference. We present DENALI, the first large-scale real-world dataset of space-time histograms from low-cost LiDARs capturing hidden objects. We capture time-resolved LiDAR histograms for 72,000 hidden-object scenes across diverse object shapes, positions, lighting conditions, and spatial resolutions. Using our dataset, we show that consumer LiDARs can enable accurate, data-driven NLOS perception. We further identify key scene and modeling factors that limit performance, as well as simulation-fidelity gaps that hinder current sim-to-real transfer, motivating future work toward scalable NLOS vision with consumer LiDARs.
>
---
#### [new 099] TTL: Test-time Textual Learning for OOD Detection with Pretrained Vision-Language Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于OOD检测任务，旨在解决预训练视觉-语言模型在测试时无法适应开放语义空间的问题。通过动态学习未标记测试数据的文本语义，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2604.15756](https://arxiv.org/pdf/2604.15756)**

> **作者:** Jinlun Ye; Jiang Liao; Runhe Lai; Xinhua Lu; Jiaxin Zhuang; Zhiyong Gan; Ruixuan Wang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Vision-language models (VLMs) such as CLIP exhibit strong Out-of-distribution (OOD) detection capabilities by aligning visual and textual representations. Recent CLIP-based test-time adaptation methods further improve detection performance by incorporating external OOD labels. However, such labels are finite and fixed, while the real OOD semantic space is inherently open-ended. Consequently, fixed labels fail to represent the diverse and evolving OOD semantics encountered in test streams. To address this limitation, we introduce Test-time Textual Learning (TTL), a framework that dynamically learns OOD textual semantics from unlabeled test streams, without relying on external OOD labels. TTL updates learnable prompts using pseudo-labeled test samples to capture emerging OOD knowledge. To suppress noise introduced by pseudo-labels, we introduce an OOD knowledge purification strategy that selects reliable OOD samples for adaptation while suppressing noise. In addition, TTL maintains an OOD Textual Knowledge Bank that stores high-quality textual features, providing stable score calibration across batches. Extensive experiments on two standard benchmarks with nine OOD datasets demonstrate that TTL consistently achieves state-of-the-art performance, highlighting the value of textual adaptation for robust test-time OOD detection. Our code is available at this https URL.
>
---
#### [new 100] GaussianFlow SLAM: Monocular Gaussian Splatting SLAM Guided by GaussianFlow
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于SLAM任务，解决单目SLAM中缺乏几何线索的问题。提出GaussianFlow SLAM，利用光流引导场景结构和相机位姿优化，提升重建精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.15612](https://arxiv.org/pdf/2604.15612)**

> **作者:** Dong-Uk Seo; Jinwoo Jeon; Eungchang Mason Lee; Hyun Myung
>
> **备注:** 8 pages, 5 figures, 7 tables, accepted to IEEE RA-L
>
> **摘要:** Gaussian splatting has recently gained traction as a compelling map representation for SLAM systems, enabling dense and photo-realistic scene modeling. However, its application to monocular SLAM remains challenging due to the lack of reliable geometric cues from monocular input. Without geometric supervision, mapping or tracking could fall in local-minima, resulting in structural degeneracies and inaccuracies. To address this challenge, we propose GaussianFlow SLAM, a monocular 3DGS-SLAM that leverages optical flow as a geometry-aware cue to guide the optimization of both the scene structure and camera poses. By encouraging the projected motion of Gaussians, termed GaussianFlow, to align with the optical flow, our method introduces consistent structural cues to regularize both map reconstruction and pose estimation. Furthermore, we introduce normalized error-based densification and pruning modules to refine inactive and unstable Gaussians, thereby contributing to improved map quality and pose accuracy. Experiments conducted on public datasets demonstrate that our method achieves superior rendering quality and tracking accuracy compared with state-of-the-art algorithms. The source code is available at: this https URL.
>
---
#### [new 101] TwoHamsters: Benchmarking Multi-Concept Compositional Unsafety in Text-to-Image Models
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于文本到图像生成的安全研究任务，旨在解决多概念组合不安全问题。提出TwoHamsters基准，评估模型和防御机制的MCCU风险。**

- **链接: [https://arxiv.org/pdf/2604.15967](https://arxiv.org/pdf/2604.15967)**

> **作者:** Chaoshuo Zhang; Yibo Liang; Mengke Tian; Chenhao Lin; Zhengyu Zhao; Le Yang; Chong Zhang; Yang Zhang; Chao Shen
>
> **摘要:** Despite the remarkable synthesis capabilities of text-to-image (T2I) models, safeguarding them against content violations remains a persistent challenge. Existing safety alignments primarily focus on explicit malicious concepts, often overlooking the subtle yet critical risks of compositional semantics. To address this oversight, we identify and formalize a novel vulnerability: Multi-Concept Compositional Unsafety (MCCU), where unsafe semantics stem from the implicit associations of individually benign concepts. Based on this formulation, we introduce TwoHamsters, a comprehensive benchmark comprising 17.5k prompts curated to probe MCCU vulnerabilities. Through a rigorous evaluation of 10 state-of-the-art models and 16 defense mechanisms, our analysis yields 8 pivotal insights. In particular, we demonstrate that current T2I models and defense mechanisms face severe MCCU risks: on TwoHamsters, FLUX achieves an MCCU generation success rate of 99.52%, while LLaVA-Guard only attains a recall of 41.06%, highlighting a critical limitation of the current paradigm for managing hazardous compositional generation.
>
---
#### [new 102] RelativeFlow: Taming Medical Image Denoising Learning with Noisy Reference
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像去噪任务，解决噪声参考问题。提出RelativeFlow框架，通过相对流匹配学习不同噪声参考，提升去噪效果。**

- **链接: [https://arxiv.org/pdf/2604.15459](https://arxiv.org/pdf/2604.15459)**

> **作者:** Yuxin Liu; Yiqing Dong; Wenxue Yu; Zhan Wu; Rongjun Ge; Yang Chen; Yuting He
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Medical image denoising (MID) lacks absolutely clean images for supervision, leading to a noisy reference problem that fundamentally limits denoising performance. Existing simulated-supervised discriminative learning (SimSDL) and simulated-supervised generative learning (SimSGL) treat noisy references as clean targets, causing suboptimal convergence or reference-biased learning, while self-supervised learning (SSL) imposes restrictive noise assumptions that are seldom satisfied in realistic MID scenarios. We propose \textbf{RelativeFlow}, a flow matching framework that learns from heterogeneous noisy references and drives inputs from arbitrary quality levels toward a unified high-quality target. RelativeFlow reformulates flow matching by decomposing the absolute noise-to-clean mapping into relative noisier-to-noisy mappings, and realizes this formulation through two key components: 1) consistent transport (CoT), a displacement map that constrains relative flows to be components of and progressively compose a unified absolute flow, and 2) simulation-based velocity field (SVF), which constructs a learnable velocity field using modality-specific degradation operators to support different medical imaging modalities. Extensive experiments on Computed Tomography (CT) and Magnetic Resonance (MR) denoising demonstrate that RelativeFlow significantly outperforms existing methods, taming MID with noisy references.
>
---
#### [new 103] Adapting in the Dark: Efficient and Stable Test-Time Adaptation for Black-Box Models
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于测试时适应（TTA）任务，解决黑盒模型在仅能通过API访问时的适应问题。提出BETA框架，通过轻量白盒模型实现高效稳定适应，无需额外API调用。**

- **链接: [https://arxiv.org/pdf/2604.15609](https://arxiv.org/pdf/2604.15609)**

> **作者:** Yunbei Zhang; Shuaicheng Niu; Chengyi Cai; Feng Liu; Jihun Hamm
>
> **备注:** Third Workshop on Test-Time Updates (Oral)
>
> **摘要:** Test-Time Adaptation (TTA) for black-box models accessible only via APIs remains a largely unexplored challenge. Existing approaches such as post-hoc output refinement offer limited adaptive capacity, while Zeroth-Order Optimization (ZOO) enables input-space adaptation but faces high query costs and optimization challenges in the unsupervised TTA setting. We introduce BETA (Black-box Efficient Test-time Adaptation), a framework that addresses these limitations by employing a lightweight, local white-box steering model to create a tractable gradient pathway. Through a prediction harmonization technique combined with consistency regularization and prompt learning-oriented filtering, BETA enables stable adaptation with no additional API calls and negligible latency beyond standard inference. On ImageNet-C, BETA achieves a +7.1% accuracy gain on ViT-B/16 and +3.4% on CLIP, surpassing strong white-box and gray-box methods including TENT and TPT. On a commercial API, BETA achieves comparable performance to ZOO at 250x lower cost while maintaining real-time inference speed, establishing it as a practical and efficient solution for real-world black-box TTA.
>
---
#### [new 104] HyperGVL: Benchmarking and Improving Large Vision-Language Models in Hypergraph Understanding and Reasoning
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决LVLM在超图理解与推理上的能力评估问题。提出首个基准HyperGVL，并引入优化方法WiseHyGR提升性能。**

- **链接: [https://arxiv.org/pdf/2604.15648](https://arxiv.org/pdf/2604.15648)**

> **作者:** Yanbin Wei; Chun Kang; Siwei Li; Haoxuan Che; Yang Chen; Hua Liu; Jian Liu; Zhuang Liu; Can Ouyang; Fei Xing; Lei Sha; Rui Liu; Yu Zhang; James Kwok
>
> **备注:** Under Review; Opensource after accepted
>
> **摘要:** Large Vision-Language Models (LVLMs) consistently require new arenas to guide their expanding boundaries, yet their capabilities with hypergraphs remain unexplored. In the real world, hypergraphs have significant practical applications in areas such as life sciences and social communities. Recent advancements in LVLMs have shown promise in understanding complex topologies, yet there remains a lack of a benchmark to delineate the capabilities of LVLMs with hypergraphs, leaving the boundaries of their abilities unclear. To fill this gap, in this paper, we introduce $\texttt{HyperGVL}$, the first benchmark to evaluate the proficiency of LVLMs in hypergraph understanding and reasoning. $\texttt{HyperGVL}$ provides a comprehensive assessment of 12 advanced LVLMs across 84,000 vision-language question-answering (QA) samples spanning 12 tasks, ranging from basic component counting to complex NP-hard problem reasoning. The involved hypergraphs contain multiscale synthetic structures and real-world citation and protein networks. Moreover, we examine the effects of 12 textual and visual hypergraph representations and introduce a generalizable router $\texttt{WiseHyGR}$ that improves LVLMs in hypergraph via learning adaptive representations. We believe that this work is a step forward in connecting hypergraphs with LVLMs.
>
---
#### [new 105] ProtoTTA: Prototype-Guided Test-Time Adaptation
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于模型适应任务，旨在提升原型模型在分布偏移下的鲁棒性。提出ProtoTTA框架，通过优化原型相似性分布增强模型稳定性与可解释性。**

- **链接: [https://arxiv.org/pdf/2604.15494](https://arxiv.org/pdf/2604.15494)**

> **作者:** Mohammad Mahdi Abootorabi; Parvin Mousavi; Purang Abolmaesumi; Evan Shelhamer
>
> **备注:** ICLR 2026 Test-Time Updates (TTU) Workshop
>
> **摘要:** Deep networks that rely on prototypes-interpretable representations that can be related to the model input-have gained significant attention for balancing high accuracy with inherent interpretability, which makes them suitable for critical domains such as healthcare. However, these models are limited by their reliance on training data, which hampers their robustness to distribution shifts. While test-time adaptation (TTA) improves the robustness of deep networks by updating parameters and statistics, the prototypes of interpretable models have not been explored for this purpose. We introduce ProtoTTA, a general framework for prototypical models that leverages intermediate prototype signals rather than relying solely on model outputs. ProtoTTA minimizes the entropy of the prototype-similarity distribution to encourage more confident and prototype-specific activations on shifted data. To maintain stability, we employ geometric filtering to restrict updates to samples with reliable prototype activations, regularized by prototype-importance weights and model-confidence scores. Experiments across four prototypical backbones on four diverse benchmarks spanning fine-grained vision, histopathology, and NLP demonstrate that ProtoTTA improves robustness over standard output entropy minimization while restoring correct semantic focus in prototype activations. We also introduce novel interpretability metrics and a vision-language model (VLM) evaluation framework to explain TTA dynamics, confirming ProtoTTA restores human-aligned semantic focus and correlates reliably with VLM-rated reasoning quality. Code is available at: this https URL.
>
---
#### [new 106] From Vulnerable Data Subjects to Vulnerabilizing Data Practices: Navigating the Protection Paradox in AI-Based Analyses of Platformized Lives
- **分类: cs.CY; cs.AI; cs.CV; cs.HC**

- **简介: 论文探讨AI分析中数据实践如何加剧个体脆弱性，属于数据伦理研究。解决数据保护与技术操作间的矛盾，通过案例分析提出伦理协议。**

- **链接: [https://arxiv.org/pdf/2604.15990](https://arxiv.org/pdf/2604.15990)**

> **作者:** Delfina S. Martinez Pandiani; Ella Streefkerk; Laurens Naudts; Paula Helm
>
> **备注:** In The 2026 ACM Conference on Fairness, Accountability, and Transparency (FAccT '26), June 25-28, 2026, Montreal, QC, Canada. ACM, New York, NY, USA, 23 pages
>
> **摘要:** This paper traces a conceptual shift from understanding vulnerability as a static, essentialized property of data subjects to examining how it is actively enacted through data practices. Unlike reflexive ethical frameworks focused on missing or counter-data, we address the condition of abundance inherent to platformized life-a context where a near inexhaustible mass of data points already exists, shifting the ethical challenge to the researcher's choices in operating upon this existing mass. We argue that the ethical integrity of data science depends not just on who is studied, but on how technical pipelines transform "vulnerable" individuals into data subjects whose vulnerability can be further precarized. We develop this argument through an AI for Social Good (AI4SG) case: a journalist's request to use computer vision to quantify child presence in monetized YouTube 'family vlogs' for regulatory advocacy. This case reveals a "protection paradox": how data-driven efforts to protect vulnerable subjects can inadvertently impose new forms of computational exposure, reductionism, and extraction. Using this request as a point of departure, we perform a methodological deconstruction of the AI pipeline to show how granular technical decisions are ethically constitutive. We contribute a reflexive ethics protocol that translates these insights into a reflexive roadmap for research ethics surrounding platformized data subjects. Organized around four critical junctures-dataset design, operationalization, inference, and dissemination-the protocol identifies technical questions and ethical tensions where well-intentioned work can slide into renewed extraction or exposure. For every decision point, the protocol offers specific prompts to navigate four cross-cutting vulnerabilizing factors: exposure, monetization, narrative fixing, and algorithmic optimization. Rather than uncritically...
>
---
## 更新

#### [replaced 001] Video-STAR: Reinforcing Open-Vocabulary Action Recognition with Tools
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.08480](https://arxiv.org/pdf/2510.08480)**

> **作者:** Zhenlong Yuan; Xiangyan Qu; Chengxuan Qian; Rui Chen; Jing Tang; Lei Sun; Xiangxiang Chu; Dapeng Zhang; Yiwei Wang; Yujun Cai; Shuo Li
>
> **摘要:** Multimodal large language models (MLLMs) have demonstrated remarkable potential in bridging visual and textual reasoning, yet their reliance on text-centric priors often limits their ability to disentangle semantically similar actions in open-vocabulary scenarios. To address this, we propose Video-STAR, a framework that harmonizes contextual sub-motion decomposition with tool-augmented reinforcement learning for open-vocabulary action recognition (OVAR). Unlike prior methods that treat actions as monolithic entities, our approach innovatively decomposes actions into discriminative sub-motions for fine-grained matching while dynamically invoking domain-specific tools for cross-modal interleaving, thereby enabling category-specific reasoning capacity and reducing cross-modal hallucination. Moreover, by designing a hierarchical reward that balances tool-usage efficiency, sub-motion relevance, and structural coherence in reasoning, our method autonomously leverages external tools to prioritize sub-motion patterns without explicit supervision, transmitting from text-centric reasoning to visually grounded inference. Extensive evaluations on HMDB-51, UCF-101, SSv2, Kinetics-400, and Kinetics-600 datasets demonstrate our state-of-the-art performance, outperforming existing methods in distinguishing fine-grained actions and handling cross-modal hallucination, validating our excellent robustness and generalization.
>
---
#### [replaced 002] 1S-DAug: One-Shot Data Augmentation for Robust Few-Shot Generalization
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.00114](https://arxiv.org/pdf/2602.00114)**

> **作者:** Yunwei Bai; Ying Kiat Tan; Yao Shu; Tsuhan Chen
>
> **摘要:** Few-shot learning (FSL) challenges model generalization to novel classes based on just a few shots of labeled examples, a testbed where traditional test-time augmentations fail to be effective. We introduce 1S-DAug, a one-shot generative augmentation operator that synthesizes diverse yet faithful variants from just one example image at test time. 1S-DAug couples traditional geometric perturbations with controlled noise injection and a denoising diffusion process conditioned on the original image. The generated images are then encoded and aggregated, alongside the original image, into a combined representation for more robust few-shot predictions. Integrated as a training-free model-agnostic plugin, 1S-DAug consistently improves few-shot classification across standard benchmarks of 4 different datasets without any model parameter update, including achieving up to 20\% relative accuracy improvement on the miniImagenet 5-way-1-shot benchmark. Additionally, we provide extension experiments on the larger vision language models as well as theoretical analyses.
>
---
#### [replaced 003] Neural Distribution Prior for LiDAR Out-of-Distribution Detection
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.09232](https://arxiv.org/pdf/2604.09232)**

> **作者:** Zizhao Li; Zhengkang Xiang; Jiayang Ao; Feng Liu; Joseph West; Kourosh Khoshelham
>
> **备注:** CVPR 2026
>
> **摘要:** LiDAR-based perception is critical for autonomous driving due to its robustness to poor lighting and visibility conditions. Yet, current models operate under the closed-set assumption and often fail to recognize unexpected out-of-distribution (OOD) objects in the open world. Existing OOD scoring functions exhibit limited performance because they ignore the pronounced class imbalance inherent in LiDAR OOD detection and assume a uniform class distribution. To address this limitation, we propose the Neural Distribution Prior (NDP), a framework that models the distributional structure of network predictions and adaptively reweights OOD scores based on alignment with a learned distribution prior. NDP dynamically captures the logit distribution patterns of training data and corrects class-dependent confidence bias through an attention-based module. We further introduce a Perlin noise-based OOD synthesis strategy that generates diverse auxiliary OOD samples from input scans, enabling robust OOD training without external datasets. Extensive experiments on the SemanticKITTI and STU benchmarks demonstrate that NDP substantially improves OOD detection performance, achieving a point-level AP of 61.31% on the STU test set, which is more than 10$\times$ higher than the previous best result. Our framework is compatible with various existing OOD scoring formulations, providing an effective solution for open-world LiDAR perception.
>
---
#### [replaced 004] EchoVLM: Dynamic Mixture-of-Experts Vision-Language Model for Universal Ultrasound Intelligence
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.14977](https://arxiv.org/pdf/2509.14977)**

> **作者:** Chaoyin She; Ruifang Lu; Lida Chen; Wei Wang; Qinghua Huang
>
> **摘要:** Ultrasound imaging has become the preferred imaging modality for early cancer screening due to its advantages of non-ionizing radiation, low cost, and real-time imaging capabilities. However, conventional ultrasound diagnosis heavily relies on physician expertise, presenting challenges of high subjectivity and low diagnostic efficiency. Vision-language models (VLMs) offer promising solutions for this issue, but existing general-purpose models demonstrate limited knowledge in ultrasound medical tasks, with poor generalization in multi-organ lesion recognition and low efficiency across multi-task diagnostics. To address these limitations, we propose EchoVLM, a vision-language model specifically designed for ultrasound medical imaging. The model employs a Mixture of Experts (MoE) architecture trained on data spanning seven anatomical regions. This design enables the model to perform multiple tasks, including ultrasound report generation, diagnosis and visual question-answering (VQA). The experimental results demonstrated that EchoVLM achieved significant improvements of 10.15 and 4.77 points in BLEU-1 scores and ROUGE-1 scores respectively compared to Qwen2-VL on the ultrasound report generation task. These findings suggest that EchoVLM has substantial potential to enhance diagnostic accuracy in ultrasound imaging, thereby providing a viable technical solution for future clinical applications. Source code and model weights are available at this https URL.
>
---
#### [replaced 005] InstructTable: Improving Table Structure Recognition Through Instructions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.02880](https://arxiv.org/pdf/2604.02880)**

> **作者:** Boming Chen; Zining Wang; Zhentao Guo; Jianqiang Liu; Chen Duan; Yu Gu; Kai zhou; Pengfei Yan
>
> **备注:** 2026 IEEE/CVF Conference on Computer Vision and Pattern Recognition- FINDINGS Track (CVPRF)
>
> **摘要:** Table structure recognition (TSR) holds widespread practical importance by parsing tabular images into structured representations, yet encounters significant challenges when processing complex layouts involving merged or empty cells. Traditional visual-centric models rely exclusively on visual information while lacking crucial semantic support, thereby impeding accurate structural recognition in complex scenarios. Vision-language models leverage contextual semantics to enhance comprehension; however, these approaches underemphasize the modeling of visual structural information. To address these limitations, this paper introduces InstructTable, an instruction-guided multi-stage training TSR framework. Meticulously designed table instruction pre-training directs attention toward fine-grained structural patterns, enhancing comprehension of complex tables. Complementary TSR fine-tuning preserves robust visual information modeling, maintaining high-precision table parsing across diverse scenarios. Furthermore, we introduce Table Mix Expand (TME), an innovative template-free method for synthesizing large-scale authentic tabular data. Leveraging TME, we construct the Balanced Complex Dense Synthetic Tables (BCDSTab) benchmark, comprising 900 complex table images synthesized through our method to serve as a rigorous benchmark. Extensive experiments on multiple public datasets (FinTabNet, PubTabNet, MUSTARD) and BCDSTab demonstrate that InstructTable achieves state-of-the-art performance in TSR tasks. Ablation studies further confirm the positive impact of the proposed tabular-data-specific instructions and synthetic data.
>
---
#### [replaced 006] BareBones: Benchmarking Zero-Shot Geometric Comprehension in VLMs
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.10528](https://arxiv.org/pdf/2604.10528)**

> **作者:** Aaditya Baranwal; Vishal Yadav; Abhishek Rajora
>
> **备注:** Accepted at CVPR (13th FGVC Workshop) 2026
>
> **摘要:** While Vision-Language Models (VLMs) demonstrate remarkable zero-shot recognition capabilities across a diverse spectrum of multimodal tasks, it yet remains an open question whether these architectures genuinely comprehend geometric structure or merely exploit RGB textures and contextual priors as statistical shortcuts. Existing evaluations fail to isolate this mechanism, conflating semantic reasoning with texture mapping and relying on imprecise annotations that inadvertently leak environmental cues. To address this gap, we introduce $\textbf{BareBones}$, a zero-shot benchmark designed to stress-test pure geometric shape comprehension. We curate pixel-level silhouettes of geometrically distinct classes across six datasets: five established segmentation sources (ImageNet-S, DIS5K, ThinObject5K, PASCAL VOC, CUB-200) and our novel flagship collection, WTP-Bench, establishing a noise-free geometric taxonomy. WTP-Bench is an extreme, fine-grained visual puzzle that forces models to identify inter-class geometric concepts from boundary contours alone. Our evaluation of 26 state-of-the-art proprietary and open-weight VLMs (eg. GPT-4.1, Gemini, Claude Sonnet 4.5, LLaVA) reveals a consistent, severe performance collapse under RGB deprivation, a phenomenon we term the $\textit{Texture Bias Cliff}$. By documenting universal structural blindspots, BareBones establishes a rigorous yardstick for genuine geometric grounding.
>
---
#### [replaced 007] FETAL-GAUGE: A Benchmark for Assessing Vision-Language Models in Fetal Ultrasound
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.22278](https://arxiv.org/pdf/2512.22278)**

> **作者:** Hussain Alasmawi; Numan Saeed; Mohammad Yaqub
>
> **摘要:** The growing demand for prenatal ultrasound imaging has intensified a global shortage of trained sonographers, creating barriers to essential fetal health monitoring. Deep learning has the potential to enhance sonographers' efficiency and support the training of new practitioners. Vision-Language Models (VLMs) are particularly promising for ultrasound interpretation, as they can jointly process images and text to perform multiple clinical tasks within a single framework. However, despite the expansion of VLMs, no standardized benchmark exists to evaluate their performance in fetal ultrasound imaging. This gap is primarily due to the modality's challenging nature, operator dependency, and the limited public availability of datasets. To address this gap, we present Fetal-Gauge, the first and largest visual question answering benchmark specifically designed to evaluate VLMs across various fetal ultrasound tasks. Our benchmark comprises over 42,000 images and 93,000 question-answer pairs, spanning anatomical plane identification, visual grounding of anatomical structures, fetal orientation assessment, clinical view conformity, and clinical diagnosis. We systematically evaluate several state-of-the-art VLMs, including general-purpose and medical-specific models, and reveal a substantial performance gap: the best-performing model achieves only 55\% accuracy, far below clinical requirements. Our analysis identifies critical limitations of current VLMs in fetal ultrasound interpretation, highlighting the urgent need for domain-adapted architectures and specialized training approaches. Fetal-Gauge establishes a rigorous foundation for advancing multimodal deep learning in prenatal care and provides a pathway toward addressing global healthcare accessibility challenges. Our benchmark will be publicly available once the paper gets accepted.
>
---
#### [replaced 008] SurgMotion: A Video-Native Foundation Model for Universal Understanding of Surgical Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.05638](https://arxiv.org/pdf/2602.05638)**

> **作者:** Jinlin Wu; Felix Holm; Chuxi Chen; An Wang; Yaxin Hu; Xiaofan Ye; Zelin Zang; Miao Xu; Lihua Zhou; Huai Liao; Danny T. M. Chan; Ming Feng; Wai S. Poon; Hongliang Ren; Dong Yi; Nassir Navab; Gaofeng Meng; Jiebo Luo; Hongbin Liu; Zhen Lei
>
> **摘要:** While foundation models have advanced surgical video analysis, current approaches rely predominantly on pixel-level reconstruction objectives that waste model capacity on low-level visual details, such as smoke, specular reflections, and fluid motion, rather than semantic structures essential for surgical understanding. We present SurgMotion, a video-native foundation model that shifts the learning paradigm from pixel-level reconstruction to latent motion prediction. Built on the Video Joint Embedding Predictive Architecture (V-JEPA), SurgMotion introduces three key technical innovations tailored to surgical videos: (1) motion-guided latent masked prediction to prioritize semantically meaningful regions, (2) spatiotemporal affinity self-distillation to enforce relational consistency, and (3) spatiotemporal feature diversity regularization (SFDR) to prevent representation collapse in texture-sparse surgical scenes. To enable large-scale pretraining, we curate SurgMotion-15M, the largest surgical video dataset to date, comprising 3,658 hours of video from 50 sources across 13 anatomical regions. Extensive experiments across 17 benchmarks demonstrate that SurgMotion significantly outperforms state-of-the-art methods on surgical workflow recognition, achieving 14.6 percent improvement in F1 score on EgoSurgery and 10.3 percent on PitVis; on action triplet recognition with 39.54 percent mAP-IVT on CholecT50; as well as on skill assessment, polyp segmentation, and depth estimation. These results establish SurgMotion as a new standard for universal, motion-oriented surgical video understanding.
>
---
#### [replaced 009] Mechanisms of Prompt-Induced Hallucination in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型研究，旨在解决提示诱导幻觉问题。通过分析模型注意力机制，发现并验证了减少幻觉的特定注意力头。**

- **链接: [https://arxiv.org/pdf/2601.05201](https://arxiv.org/pdf/2601.05201)**

> **作者:** William Rudman; Michal Golovanevsky; Dana Arad; Yonatan Belinkov; Ritambhara Singh; Carsten Eickhoff; Kyle Mahowald
>
> **备注:** ACL 2026 Main
>
> **摘要:** Large vision-language models (VLMs) are highly capable, yet often hallucinate by favoring textual prompts over visual evidence. We study this failure mode in a controlled object-counting setting, where the prompt overstates the number of objects in the image (e.g., asking a model to describe four waterlilies when only three are present). At low object counts, models often correct the overestimation, but as the number of objects increases, they increasingly conform to the prompt regardless of the discrepancy. Through mechanistic analysis of three VLMs, we identify a small set of attention heads whose ablation substantially reduces prompt-induced hallucinations (PIH) by at least 40% without additional training. Across models, PIH-heads mediate prompt copying in model-specific ways. We characterize these differences and show that PIH ablation increases correction toward visual evidence. Our findings offer insights into the internal mechanisms driving prompt-induced hallucinations, revealing model-specific differences in how these behaviors are implemented.
>
---
#### [replaced 010] EventCrab: Harnessing Frame and Point Synergy for Event-based Action Recognition and Beyond
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.18328](https://arxiv.org/pdf/2411.18328)**

> **作者:** Meiqi Cao; Xiangbo Shu; Jiachao Zhang; Rui Yan; Zechao Li; Jinhui Tang
>
> **备注:** The experiments in this paper are not comprehensive enough to make the conclusions convincing. The authors are adding more experimental scenarios and will resubmit after completion
>
> **摘要:** Event-based Action Recognition (EAR) possesses the advantages of high-temporal resolution capturing and privacy preservation compared with traditional action recognition. Current leading EAR solutions typically follow two regimes: project unconstructed event streams into dense constructed event frames and adopt powerful frame-specific networks, or employ lightweight point-specific networks to handle sparse unconstructed event points directly. However, such two regimes are blind to a fundamental issue: failing to accommodate the unique dense temporal and sparse spatial properties of asynchronous event data. In this article, we present a synergy-aware framework, i.e., EventCrab, that adeptly integrates the "lighter" frame-specific networks for dense event frames with the "heavier" point-specific networks for sparse event points, balancing accuracy and efficiency. Furthermore, we establish a joint frame-text-point representation space to bridge distinct event frames and points. In specific, to better exploit the unique spatiotemporal relationships inherent in asynchronous event points, we devise two strategies for the "heavier" point-specific embedding: i) a Spiking-like Context Learner (SCL) that extracts contextualized event points from raw event streams. ii) an Event Point Encoder (EPE) that further explores event-point long spatiotemporal features in a Hilbert-scan way. Experiments on four datasets demonstrate the significant performance of our proposed EventCrab, particularly gaining improvements of 5.17% on SeAct and 7.01% on HARDVS.
>
---
#### [replaced 011] MMAudioSep: Taming Video-to-Audio Generative Model Towards Video/Text-Queried Sound Separation
- **分类: cs.SD; cs.CV; cs.LG; eess.AS**

- **简介: 该论文提出MMAudioSep，用于视频/文本查询的音源分离任务。通过预训练视频到音频模型提升效率，解决传统方法训练成本高的问题，并验证其在分离与生成任务中的有效性。**

- **链接: [https://arxiv.org/pdf/2510.09065](https://arxiv.org/pdf/2510.09065)**

> **作者:** Akira Takahashi; Shusuke Takahashi; Yuki Mitsufuji
>
> **备注:** Accepted to ICASSP 2026. 4 pages, 4 figures, 2 tables
>
> **摘要:** We introduce MMAudioSep, a generative model for video/text-queried sound separation that is founded on a pretrained video-to-audio model. By leveraging knowledge about the relationship between video/text and audio learned through a pretrained audio generative model, we can train the model more efficiently, i.e., the model does not need to be trained from scratch. We evaluate the performance of MMAudioSep by comparing it to existing separation models, including models based on both deterministic and generative approaches, and find it is superior to the baseline models. Furthermore, we demonstrate that even after acquiring functionality for sound separation via fine-tuning, the model retains the ability for original video-to-audio generation. This highlights the potential of foundational sound generation models to be adopted for sound-related downstream tasks. Our code is available at this https URL.
>
---
#### [replaced 012] SatBLIP: Context Understanding and Feature Identification from Satellite Imagery with Vision-Language Learning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.14373](https://arxiv.org/pdf/2604.14373)**

> **作者:** Xue Wu; Shengting Cao; Shenglin Li; Jiaqi Gong
>
> **摘要:** Rural environmental risks are shaped by place-based conditions (e.g., housing quality, road access, land-surface patterns), yet standard vulnerability indices are coarse and provide limited insight into risk contexts. We propose SatBLIP, a satellite-specific vision-language framework for rural context understanding and feature identification that predicts county-level Social Vulnerability Index (SVI). SatBLIP addresses limitations of prior remote sensing pipelines-handcrafted features, manual virtual audits, and natural-image-trained VLMs-by coupling contrastive image-text alignment with bootstrapped captioning tailored to satellite semantics. We use GPT-4o to generate structured descriptions of satellite tiles (roof type/condition, house size, yard attributes, greenery, and road context), then fine-tune a satellite-adapted BLIP model to generate captions for unseen images. Captions are encoded with CLIP and fused with LLM-derived embeddings via attention for SVI estimation under spatial aggregation. Using SHAP, we identify salient attributes (e.g., roof form/condition, street width, vegetation, cars/open space) that consistently drive robust predictions, enabling interpretable mapping of rural risk environments.
>
---
#### [replaced 013] An Empirical Study of Validating Synthetic Data for Text-Based Person Retrieval
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.22171](https://arxiv.org/pdf/2503.22171)**

> **作者:** Min Cao; Yuxin Lu; Ziyin Zeng; Dong Yi; Jinqiao Wang; Mang Ye
>
> **备注:** 20 pages,13 figures
>
> **摘要:** Data plays a pivotal role in Text-Based Person Retrieval (TBPR) research. Mainstream research paradigm necessitates real-world person images with manual textual annotations for training models, posing privacy concerns and annotation burdens. Several pioneering efforts explore synthetic data generation, and yet still depend on real data as a foundation, inheriting the same limitations. The feasibility of purely synthetic TBPR data remains unexplored, and there is currently no systematic study on the effectiveness boundaries of synthetic data across various real-world scenarios. In this work, we present the first comprehensive empirical study of synthetic data for TBPR, with two key aspects. (1) We propose a unified data synthesis pipeline that can operate entirely without real person data. It combines an inter-class image generation module that produces diverse identity-centric images by means of an automatic prompt construction strategy, and an intra-class augmentation module that enhances identity variation through text-driven image editing. (2) Leveraging this pipeline and an automatic textual description generation, we explore the effectiveness of synthetic data in diverse scenarios through extensive experiments, to reveal its practical utility as either a standalone replacement or a complementary augmentation to real data.
>
---
#### [replaced 014] MFC-RFNet: A Multi-scale Guided Rectified Flow Network for Radar Sequence Prediction
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.03633](https://arxiv.org/pdf/2601.03633)**

> **作者:** Wenjie Luo; Chuanhu Deng; Chaorong Li; Rongyao Deng; Qiang Yang
>
> **摘要:** Accurate and high-resolution precipitation nowcasting from radar echo sequences is crucial for disaster mitigation and economic planning, yet it remains a significant challenge. Key difficulties include modeling complex multi-scale evolution, correcting inter-frame feature misalignment caused by displacement, and efficiently capturing long-range spatiotemporal context without sacrificing spatial fidelity. To address these issues, we present the Multi-scale Feature Communication Rectified Flow (RF) Network (MFC-RFNet), a generative framework that integrates multi-scale communication with guided feature fusion. To enhance multi-scale fusion while retaining fine detail, a Wavelet-Guided Skip Connection (WGSC) preserves high-frequency components, and a Feature Communication Module (FCM) promotes bidirectional cross-scale interaction. To correct inter-frame displacement, a Condition-Guided Spatial Transform Fusion (CGSTF) learns spatial transforms from conditioning echoes to align shallow features. The backbone adopts rectified flow training to learn near-linear probability-flow trajectories, enabling few-step sampling with stable fidelity. Additionally, lightweight Vision-RWKV (RWKV) blocks are placed at the encoder tail, the bottleneck, and the first decoder layer to capture long-range spatiotemporal dependencies at low spatial resolutions with moderate compute. Evaluations on four public datasets (SEVIR, MeteoNet, Shanghai, and CIKM) demonstrate consistent improvements over strong baselines, yielding clearer echo morphology at higher rain-rate thresholds and sustained skill at longer lead times. These results suggest that the proposed synergy of RF training with scale-aware communication, spatial alignment, and frequency-aware fusion presents an effective and robust approach for radar-based nowcasting.
>
---
#### [replaced 015] ChatENV: An Interactive Vision-Language Model for Sensor-Guided Environmental Monitoring and Scenario Simulation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.10635](https://arxiv.org/pdf/2508.10635)**

> **作者:** Hosam Elgendy; Ahmed Sharshar; Ahmed Aboeitta; Mohsen Guizani
>
> **备注:** 11 pages, 5 figures, 7 tables
>
> **摘要:** Understanding environmental changes from remote sensing imagery is vital for climate resilience, urban planning, and ecosystem monitoring. Yet, current vision language models (VLMs) overlook causal signals from environmental sensors, rely on single-source captions prone to stylistic bias, and lack interactive scenario-based reasoning. We present ChatENV, the first interactive VLM that jointly reasons over satellite image pairs and real-world sensor data. Our framework: (i) creates a 177k-image dataset forming 152k temporal pairs across 62 land-use classes in 197 countries with rich sensor metadata (e.g., temperature, PM10, CO); (ii) annotates data using GPT4o and Gemini 2.0 for stylistic and semantic diversity; and (iii) fine-tunes Qwen-2.5-VL using efficient Low-Rank Adaptation (LoRA) adapters for chat purposes. ChatENV achieves strong performance in temporal and "what-if" reasoning (e.g., BERTF1 0.902) and rivals or outperforms state-of-the-art temporal models, while supporting interactive scenario-based analysis. This positions ChatENV as a powerful tool for grounded, sensor-aware environmental monitoring.
>
---
#### [replaced 016] DriveLaW:Unifying Planning and Video Generation in a Latent Driving World
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.23421](https://arxiv.org/pdf/2512.23421)**

> **作者:** Tianze Xia; Yongkang Li; Lijun Zhou; Jingfeng Yao; Kaixin Xiong; Haiyang Sun; Bing Wang; Kun Ma; Guang Chen; Hangjun Ye; Wenyu Liu; Xinggang Wang
>
> **备注:** 18 pages, 6 figures, CVPR 2026
>
> **摘要:** World models have become crucial for autonomous driving, as they learn how scenarios evolve over time to address the long-tail challenges of the real world. However, current approaches relegate world models to limited roles: they operate within ostensibly unified architectures that still keep world prediction and motion planning as decoupled processes. To bridge this gap, we propose DriveLaW, a novel paradigm that unifies video generation and motion planning. By directly injecting the latent representation from its video generator into the planner, DriveLaW ensures inherent consistency between high-fidelity future generation and reliable trajectory planning. Specifically, DriveLaW consists of two core components: DriveLaW-Video, our powerful world model that generates high-fidelity forecasting with expressive latent representations, and DriveLaW-Act, a diffusion planner that generates consistent and reliable trajectories from the latent of DriveLaW-Video, with both components optimized by a three-stage progressive training strategy. The power of our unified paradigm is demonstrated by new state-of-the-art results across both tasks. DriveLaW not only advances video prediction significantly, surpassing best-performing work by 33.3% in FID and 1.8% in FVD, but also achieves a new record on the NAVSIM planning benchmark.
>
---
#### [replaced 017] VIB-Probe: Detecting and Mitigating Hallucinations in Vision-Language Models via Variational Information Bottleneck
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2601.05547](https://arxiv.org/pdf/2601.05547)**

> **作者:** Feiran Zhang; Yixin Wu; Zhenghua Wang; Xiaohua Wang; Changze Lv; Xuanjing Huang; Xiaoqing Zheng
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated remarkable progress in multimodal tasks, but remain susceptible to hallucinations, where generated text deviates from the underlying visual content. Existing hallucination detection methods primarily rely on output logits or external verification tools, often overlooking their internal mechanisms. In this work, we investigate the outputs of internal attention heads, postulating that specific heads carry the primary signals for truthful this http URL, directly probing these high-dimensional states is challenging due to the entanglement of visual-linguistic syntax and noise. To address this, we propose VIB-Probe, a novel hallucination detection and mitigation framework leveraging the Variational Information Bottleneck (VIB) theory. Our method extracts discriminative patterns across layers and heads while filtering out semantic nuisances through the information bottleneck principle. Furthermore, by leveraging the gradients of our VIB probe, we identify attention heads with strong causal influence on hallucinations and introduce an inference-time intervention strategy for hallucination mitigation. Extensive experiments across diverse benchmarks demonstrate that VIB-Probe significantly outperforms existing baselines in both settings. Our code will be made publicly available.
>
---
#### [replaced 018] Scalable spatial point process models for forensic footwear analysis
- **分类: cs.CV; cs.LG; stat.ML**

- **链接: [https://arxiv.org/pdf/2602.07006](https://arxiv.org/pdf/2602.07006)**

> **作者:** Alokesh Manna; Neil Spencer; Dipak K. Dey
>
> **摘要:** Shoe print evidence recovered from crime scenes plays a key role in forensic investigations. By examining shoe prints, investigators can determine details of the footwear worn by suspects. However, establishing that a suspect's shoes match the make and model of a crime scene print may not be sufficient. Typically, thousands of shoes of the same size, make, and model are manufactured, any of which could be responsible for the print. Accordingly, a popular approach used by investigators is to examine the print for signs of ``accidentals,'' i.e., cuts, scrapes, and other features that accumulate on shoe soles after purchase due to wear. While some patterns of accidentals are common on certain types of shoes, others are highly distinctive, potentially distinguishing the suspect's shoe from all others. Quantifying the rarity of a pattern is thus essential to accurately measuring the strength of forensic evidence. In this study, we address this task by developing a hierarchical Bayesian model. Our improvement over existing methods primarily stems from two advancements. First, we frame our approach in terms of a latent Gaussian model, thus enabling inference to be efficiently scaled to large collections of annotated shoe prints via integrated nested Laplace approximations. Second, we incorporate spatially varying coefficients to model the relationship between shoes' tread patterns and accidental locations. We demonstrate these improvements through superior performance on held-out data, which enhances accuracy and reliability in forensic shoe print analysis.
>
---
#### [replaced 019] VRAG-DFD: Verifiable Retrieval-Augmentation for MLLM-based Deepfake Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.13660](https://arxiv.org/pdf/2604.13660)**

> **作者:** Hui Han; Shunli Wang; Yandan Zhao; Taiping Yao; Shouhong Ding
>
> **摘要:** In Deepfake Detection (DFD) tasks, researchers proposed two types of MLLM-based methods: complementary combination with small DFD detectors, or static forgery knowledge injection. The lack of professional forgery knowledge hinders the performance of these DFD-MLLMs. To solve this, we deeply considered two insightful issues: How to provide high-quality associated forgery knowledge for MLLMs? AND How to endow MLLMs with critical reasoning abilities given noisy reference information? Notably, we attempted to address above two questions with preliminary answers by leveraging the combination of Retrieval-Augmented Generation (RAG) and Reinforcement Learning (RL). Through RAG and RL techniques, we propose the VRAG-DFD framework with accurate dynamic forgery knowledge retrieval and powerful critical reasoning capabilities. Specifically, in terms of data, we constructed two datasets with RAG: Forensic Knowledge Database (FKD) for DFD knowledge annotation, and Forensic Chain-of-Thought Dataset (F-CoT), for critical CoT construction. In terms of model training, we adopt a three-stage training method (Alignment->SFT->GRPO) to gradually cultivate the critical reasoning ability of the MLLM. In terms of performance, VRAG-DFD achieved SOTA and competitive performance on DFD generalization testing.
>
---
#### [replaced 020] VeRVE: Versatile Retrieval for Videos via Unified Embeddings
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.12193](https://arxiv.org/pdf/2601.12193)**

> **作者:** Shaunak Halbe; Bhagyashree Puranik; Jayakrishnan Unnikrishnan; Kushan Thakkar; Vimal Bhat; Toufiq Parag
>
> **摘要:** Modern video retrieval systems are expected to handle diverse tasks ranging from corpus-level retrieval, fine-grained moment localization to flexible multimodal querying. Specialized architectures achieve strong retrieval performance by training modality-specific encoders on massive datasets, but they lack the ability to process composed multimodal queries. In contrast, multimodal LLM (MLLM)-based methods support rich multimodal search but their retrieval performance remains well below that of specialized systems. We present VeRVE, an MLLM-based versatile video retrieval framework that integrates corpus and moment-level retrieval capabilities while accommodating composed multimodal queries within a single architecture. We use contrastive alignment of visual and textual embeddings generated using a shared MLLM backbone to facilitate efficient embedding-based candidate search. Our embedding model, trained efficiently using low-rank adaptation (LoRA) on 700K paired visual-text data samples, surpasses other MLLM-based methods on zero-shot video retrieval tasks. Additionally, we demonstrate that the same model can be adapted without further training to achieve competitive results on zero-shot moment retrieval, and state of the art results for zero-shot composed video retrieval. With additional training for reranking candidates identified in the embedding-based search, our model substantially outperforms existing MLLM-based retrieval systems and achieves retrieval performance comparable to state of the art specialized models.
>
---
#### [replaced 021] HiFi-Inpaint: Towards High-Fidelity Reference-Based Inpainting for Generating Detail-Preserving Human-Product Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.02210](https://arxiv.org/pdf/2603.02210)**

> **作者:** Yichen Liu; Donghao Zhou; Jie Wang; Xin Gao; Guisheng Liu; Jiatong Li; Quanwei Zhang; Qiang Lyu; Lanqing Guo; Shilei Wen; Weiqiang Wang; Pheng-Ann Heng
>
> **备注:** Accepted by CVPR 2026 (Project page: this https URL)
>
> **摘要:** Human-product images, which showcase the integration of humans and products, play a vital role in advertising, e-commerce, and digital marketing. The essential challenge of generating such images lies in ensuring the high-fidelity preservation of product details. Among existing paradigms, reference-based inpainting offers a targeted solution by leveraging product reference images to guide the inpainting process. However, limitations remain in three key aspects: the lack of diverse large-scale training data, the struggle of current models to focus on product detail preservation, and the inability of coarse supervision for achieving precise guidance. To address these issues, we propose HiFi-Inpaint, a novel high-fidelity reference-based inpainting framework tailored for generating human-product images. HiFi-Inpaint introduces Shared Enhancement Attention (SEA) to refine fine-grained product features and Detail-Aware Loss (DAL) to enforce precise pixel-level supervision using high-frequency maps. Additionally, we construct a new dataset, HP-Image-40K, with samples curated from self-synthesis data and processed with automatic filtering. Experimental results show that HiFi-Inpaint achieves state-of-the-art performance, delivering detail-preserving human-product images.
>
---
#### [replaced 022] Power to the Clients: Federated Learning in a Dictatorship Setting
- **分类: cs.LG; cs.AI; cs.CL; cs.CR; cs.CV; cs.DC**

- **简介: 该论文属于安全联邦学习任务，研究恶意客户端在集中式FL中的攻击行为，提出“独裁客户端”概念并分析其对模型收敛的影响。**

- **链接: [https://arxiv.org/pdf/2510.22149](https://arxiv.org/pdf/2510.22149)**

> **作者:** Mohammadsajad Alipour; Mohammad Mohammadi Amiri
>
> **摘要:** Federated learning (FL) has emerged as a promising paradigm for decentralized model training, enabling multiple clients to collaboratively learn a shared model without exchanging their local data. However, the decentralized nature of FL also introduces vulnerabilities, as malicious clients can compromise or manipulate the training process. In this work, we introduce dictator clients, a novel, well-defined, and analytically tractable class of malicious participants capable of entirely erasing the contributions of all other clients from the server model, while preserving their own. We propose concrete attack strategies that empower such clients and systematically analyze their effects on the learning process. Furthermore, we explore complex scenarios involving multiple dictator clients, including cases where they collaborate, act independently, or form an alliance in order to ultimately betray one another. For each of these settings, we provide a theoretical analysis of their impact on the global model's convergence. Our theoretical algorithms and findings about the complex scenarios including multiple dictator clients are further supported by empirical evaluations on both computer vision and natural language processing benchmarks.
>
---
#### [replaced 023] AutoDrive-R$^2$: Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自主驾驶任务，旨在提升VLA模型的推理与自省能力。通过构建CoT数据集和GRPO算法，增强决策过程的可解释性与动作序列合理性。**

- **链接: [https://arxiv.org/pdf/2509.01944](https://arxiv.org/pdf/2509.01944)**

> **作者:** Zhenlong Yuan; Chengxuan Qian; Jing Tang; Rui Chen; Zijian Song; Lei Sun; Xiangxiang Chu; Yujun Cai; Dapeng Zhang; Shuo Li
>
> **摘要:** Vision-Language-Action (VLA) models in autonomous driving systems have recently demonstrated transformative potential by integrating multimodal perception with decision-making capabilities. However, the interpretability and coherence of the decision process and the plausibility of action sequences remain largely underexplored. To address these issues, we propose AutoDrive-R$^2$, a novel VLA framework that enhances both reasoning and self-reflection capabilities of autonomous driving systems through chain-of-thought (CoT) processing and reinforcement learning (RL). Specifically, we first propose an innovative CoT dataset named nuScenesR$^2$-6K for supervised fine-tuning, which effectively builds cognitive bridges between input information and output trajectories through a four-step logical chain with self-reflection for validation. Moreover, to maximize both reasoning and self-reflection during the RL stage, we further employ the Group Relative Policy Optimization (GRPO) algorithm within a physics-grounded reward framework that incorporates spatial alignment, vehicle dynamic, and temporal smoothness criteria to ensure reliable and realistic trajectory planning. Extensive evaluation results across both nuScenes and Waymo datasets demonstrates the state-of-the-art performance and robust generalization capacity of our proposed method.
>
---
#### [replaced 024] Intelligent Healthcare Imaging Platform: A VLM-Based Framework for Automated Medical Image Analysis and Clinical Report Generation
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.13590](https://arxiv.org/pdf/2509.13590)**

> **作者:** Samer Al-Hamadani
>
> **备注:** 32 pages, 14 figures, 6 tables
>
> **摘要:** The rapid advancement of artificial intelligence (AI) in healthcare imaging has revolutionized diagnostic medicine and clinical decision-making processes. This work presents an intelligent multimodal framework for medical image analysis that leverages Vision-Language Models (VLMs) in healthcare diagnostics. The framework integrates Google Gemini 2.5 Flash for automated tumor detection and clinical report generation across multiple imaging modalities including CT, MRI, X-ray, and Ultrasound. The system combines visual feature extraction with natural language processing to enable contextual image interpretation, incorporating coordinate verification mechanisms and probabilistic Gaussian modeling for anomaly distribution. Multi-layered visualization techniques generate detailed medical illustrations, overlay comparisons, and statistical representations to enhance clinical confidence, with location measurement achieving 80 pixels average deviation. Result processing utilizes precise prompt engineering and textual analysis to extract structured clinical information while maintaining interpretability. Experimental evaluations demonstrated high performance in anomaly detection across multiple modalities. The system features a user-friendly Gradio interface for clinical workflow integration and demonstrates zero-shot learning capabilities to reduce dependence on large datasets. This framework represents a significant advancement in automated diagnostic support and radiological workflow efficiency, though clinical validation and multi-center evaluation are necessary prior to widespread adoption.
>
---
#### [replaced 025] Anthropogenic Regional Adaptation in Multimodal Vision-Language Model
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于视觉语言模型任务，解决区域文化对齐问题。提出Anthropogenic Regional Adaptation框架及GG-EZ方法，提升模型在特定地区的文化相关性，同时保持全局性能。**

- **链接: [https://arxiv.org/pdf/2604.11490](https://arxiv.org/pdf/2604.11490)**

> **作者:** Samuel Cahyawijaya; Peerat Limkonchotiwat; Tack Hwa Wong; Hitesh Laxmichand Patel; Amit Agarwal; Manuel Antonio Rufino; Carlos Rafael Catalan; Muhammad Reza Qorib; Vicky Feliren; Holy Lovenia; Aye Hninn Khine; Frederikus Hudi; David Anugraha; Alham Fikri Aji; Romrawin Chumpu; Viet-Thanh Pham; Minghan Wang; Mohamed Fazli Imam; Ruochen Zhang; Joseph Marvin Imperial; Khumaisa Nur'aini; Do Xuan Long; Musa Izzanardi Wijanarko; Joel Ruben Antony Moniz; Patrick Amadeus Irawan; Hanif Muhammad Zhafran; Isaiah Flores; Salsabila Zahirah Pranida; Jun Kevin; Jostin Jerico Rosal; Patricia Nicole Monderin; Kun Kerdthaisong; Ahmad Mustafid; My Chiffon Nguyen; Natchapon Jongwiriyanurak; Siva Worajitwannakul; Haochen Li; Adrian Xuan Wei Lim; Bin Wang; Muhammad Ravi Shulthan Habibi; Lynnette Hui Xian Ng; Mithil Bangera; Yeshil Bangera; Priyaranjan Pattnayak; Dun Li Chan; Sherissa Caren Djuniwar; Cho Chan Myei Oo; Hee Ming Shan
>
> **摘要:** While the field of vision-language (VL) has achieved remarkable success in integrating visual and textual information across multiple languages and domains, there is still no dedicated framework for assessing human-centric alignment in vision-language systems. We offer two contributions to address this gap. First, we introduce Anthropogenic Regional Adaptation: a novel paradigm that aims to optimize model relevance to specific regional contexts while ensuring the retention of global generalization capabilities. Second, we present a simple, but effective adaptation method named Geographical-generalization-made-easy (GG-EZ), which utilizes regional data filtering and model merging. Through comprehensive experiments on 3 VL architectures: large vision-language models, text-to-image diffusion models, and vision-language embedding models, and a case study in Southeast Asia (SEA) regional adaptation, we demonstrate the importance of Anthropogenic Regional Adaptation and the effectiveness of GG-EZ, showing 5-15% gains in cultural relevance metrics across SEA while maintaining over 98% of global performance and even occasionally surpassing it. Our findings establish Anthropogenic Regional Alignment as a foundational paradigm towards applicability of multimodal vision-language models in diverse regions and demonstrate a simple-yet-effective baseline method that optimizes regional value alignment while preserving global generalization.
>
---
#### [replaced 026] Social-JEPA: Emergent Geometric Isomorphism
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.02263](https://arxiv.org/pdf/2603.02263)**

> **作者:** Haoran Zhang; Youjin Wang; Yi Duan; Rong Fu; Dianyu Zhao; Sicheng Fan; Shuaishuai Cao; Wentao Guo; Xiao Zhou
>
> **备注:** This preprint is withdrawn due to significant errors in the emergent geometric isomorphism results that necessitate full rewriting, coupled with unresolved author disagreement on authorship. A corrected and revised manuscript will be released separately
>
> **摘要:** World models compress rich sensory streams into compact latent codes that anticipate future observations. We let separate agents acquire such models from distinct viewpoints of the same environment without any parameter sharing or coordination. After training, their internal representations exhibit a striking emergent property: the two latent spaces are related by an approximate linear isometry, enabling transparent translation between them. This geometric consensus survives large viewpoint shifts and scant overlap in raw pixels. Leveraging the learned alignment, a classifier trained on one agent can be ported to the other with no additional gradient steps, while distillation-like migration accelerates later learning and markedly reduces total compute. The findings reveal that predictive learning objectives impose strong regularities on representation geometry, suggesting a lightweight path to interoperability among decentralized vision systems. The code is available at this https URL.
>
---
#### [replaced 027] Two-Stage Framework for Efficient UAV-Based Wildfire Video Analysis with Adaptive Compression and Fire Source Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.16739](https://arxiv.org/pdf/2508.16739)**

> **作者:** Yanbing Bai; Rui-Yang Ju; Lemeng Zhao; Junjie Hu; Jianchao Bi; Erick Mas; Shunichi Koshimura
>
> **备注:** IEEE JSTARS; Extended Journal Version of IROS 2024
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) have become increasingly important in disaster emergency response by facilitating aerial video analysis. Due to the limited computational resources available on UAVs, large models cannot be run efficiently for on-board analysis. To overcome this challenge, we propose a lightweight and efficient two-stage framework for wildfire monitoring and fire source detection on UAV platforms. Specifically, in Stage 1, we utilize a policy network to identify and discard redundant video clips, thereby reducing computational costs. We also introduce a station point mechanism that incorporates future frame information within the sequential policy network to improve prediction accuracy. This mechanism allows Stage 1 to operate in a near-real-time manner. In Stage 2, for frames classified as containing fire, we apply an improved YOLOv8 model to accurately localize the fire source in real-time on selected frames. We evaluate Stage 1 using the FLAME and HMDB51 datasets, and Stage 2 using the Fire & Smoke Detection Dataset. Experimental results show that our method significantly reduces computational costs while maintaining classification accuracy in Stage 1, and achieves high detection accuracy with real-time inference in Stage 2.
>
---
#### [replaced 028] ABot-Claw: A Foundation for Persistent, Cooperative, and Self-Evolving Robotic Agents
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.10096](https://arxiv.org/pdf/2604.10096)**

> **作者:** Dongjie Huo; Haoyun Liu; Guoqing Liu; Dekang Qi; Zhiming Sun; Maoguo Gao; Jianxin He; Yandan Yang; Xinyuan Chang; Feng Xiong; Xing Wei; Zhiheng Ma; Mu Xu
>
> **摘要:** Current embodied intelligent systems still face a substantial gap between high-level reasoning and low-level physical execution in open-world environments. Although Vision-Language-Action (VLA) models provide strong perception and intuitive responses, their open-loop nature limits long-horizon performance. Agents incorporating System 2 cognitive mechanisms improve planning, but usually operate in closed sandboxes with predefined toolkits and limited real-system control. OpenClaw provides a localized runtime with full system privileges, but lacks the embodied control architecture required for long-duration, multi-robot execution. We therefore propose ABot-Claw, an embodied extension of OpenClaw that integrates: 1) a unified embodiment interface with capability-driven scheduling for heterogeneous robot coordination; 2) a visual-centric cross-embodiment multimodal memory for persistent context retention and grounded retrieval; and 3) a critic-based closed-loop feedback mechanism with a generalist reward model for online progress evaluation, local correction, and replanning. With a decoupled architecture spanning the OpenClaw layer, shared service layer, and robot embodiment layer, ABot-Claw enables real-world interaction, closes the loop from natural language intent to physical action, and supports progressively self-evolving robotic agents in open, dynamic environments.
>
---
#### [replaced 029] Noise Aggregation Analysis Driven by Small-Noise Injection: Efficient Membership Inference for Diffusion Models
- **分类: cs.CV; cs.AI; cs.CR**

- **链接: [https://arxiv.org/pdf/2510.21783](https://arxiv.org/pdf/2510.21783)**

> **作者:** Guo Li; Weihong Chen; Yongfu Fan
>
> **摘要:** Diffusion models have demonstrated powerful performance in generating high-quality images. A typical example is text-to-image generator like Stable Diffusion. However, their widespread use also poses potential privacy risks. A key concern is membership inference attacks, which attempt to determine whether a particular data sample was used in the model training process. Existing membership inference attacks against diffusion models either directly exploit sample loss differences or rely on image-level reconstruction differences. Both approaches commonly ignore the consistency characteristics of noise prediction during the diffusion process, resulting in either low inference accuracy or high computational costs. To address these shortcomings, we propose a membership inference method based on noise aggregation analysis, and introduce a single-step, low-intensity noise injection diffusion strategy to amplify differences between member and non-member samples. Our proposed approach substantially reduces model query requirements while delivering more efficient and accurate membership inference.
>
---
#### [replaced 030] From Limited Labels to Open Domains:An Efficient Learning Method for Drone-view Geo-Localization
- **分类: cs.CV; cs.IR**

- **链接: [https://arxiv.org/pdf/2503.07520](https://arxiv.org/pdf/2503.07520)**

> **作者:** Zhongwei Chen; Zhao-Xu Yang; Hai-Jun Rong; Jiawei Lang; Guoqi Li
>
> **备注:** Accepted by IEEE Transactions on Multimedia 2026
>
> **摘要:** Traditional supervised drone-view geo-localization (DVGL) methods heavily depend on paired training data and encounter difficulties in learning cross-view correlations from unpaired data. Moreover, when deployed in a new domain, these methods require obtaining the new paired data and subsequent retraining for model adaptation, which significantly increases computational overhead. Existing unsupervised methods have enabled to generate pseudo-labels based on cross-view similarity to infer the pairing relationships. However, geographical similarity and spatial continuity often cause visually analogous features at different geographical locations. The feature confusion compromises the reliability of pseudo-label generation, where incorrect pseudo-labels drive negative optimization. Given these challenges inherent in both supervised and unsupervised DVGL methods, we propose a novel cross-domain invariant knowledge transfer network (CDIKTNet) with limited supervision, whose architecture consists of a cross-domain invariance sub-network (CDIS) and a cross-domain transfer sub-network (CDTS). This architecture facilitates a closed-loop framework for invariance feature learning and knowledge transfer. The CDIS is designed to learn cross-view structural and spatial invariance from a small amount of paired data that serves as prior knowledge. It endows the shared feature space of unpaired data with similar implicit cross-view correlations at initialization, which alleviates feature confusion. Based on this, the CDTS employs dual-path contrastive learning to further optimize each subspace while preserving consistency in a shared feature space. Extensive experiments demonstrate that CDIKTNet achieves state-of-the-art performance under full supervision compared with those supervised methods, and further surpasses existing unsupervised methods in both few-shot and cross-domain initialization.
>
---
#### [replaced 031] Reasoning over Video: Evaluating How MLLMs Extract, Integrate, and Reconstruct Spatiotemporal Evidence
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.13091](https://arxiv.org/pdf/2603.13091)**

> **作者:** Seunghwan Bang; Hwanjun Song
>
> **备注:** Project page: this https URL
>
> **摘要:** The growing interest in embodied agents increases the demand for spatiotemporal video understanding, yet existing benchmarks largely emphasize extractive reasoning, where answers can be explicitly presented within spatiotemporal events. It remains unclear whether multimodal large language models can instead perform abstractive spatiotemporal reasoning, which requires integrating observations over time, combining dispersed cues, and inferring implicit spatial and contextual structure. To address this gap, we formalize abstractive spatiotemporal reasoning from videos by introducing a structured evaluation taxonomy that systematically targets its core dimensions and constructs a controllable, scenario-driven synthetic egocentric video dataset tailored to evaluate abstractive spatiotemporal reasoning capabilities, spanning object-, room-, and floor-plan-level scenarios. Based on this framework, we present VAEX-BENCH, a benchmark comprising five abstractive reasoning tasks together with their extractive counterparts. Our extensive experiments compare the performance of state-of-the-art MLLMs under extractive and abstractive settings, exposing their limitations on abstractive tasks and providing a fine-grained analysis of the underlying bottlenecks. The dataset will be released soon.
>
---
#### [replaced 032] UniDoc-RL: Coarse-to-Fine Visual RAG with Hierarchical Actions and Dense Rewards
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.14967](https://arxiv.org/pdf/2604.14967)**

> **作者:** Jun Wang; Shuo Tan; Zelong Sun; Tiancheng Gu; Yongle Zhao; Ziyong Feng; Kaicheng Yang; Zhiwu Lu
>
> **备注:** 17 pages, 11 figures
>
> **摘要:** Retrieval-Augmented Generation (RAG) extends Large Vision-Language Models (LVLMs) with external visual knowledge. However, existing visual RAG systems typically rely on generic retrieval signals that overlook the fine-grained visual semantics essential for complex reasoning. To address this limitation, we propose UniDoc-RL, a unified reinforcement learning framework in which an LVLM agent jointly performs retrieval, reranking, active visual perception, and reasoning. UniDoc-RL formulates visual information acquisition as a sequential decision-making problem with a hierarchical action space. Specifically, it progressively refines visual evidence from coarse-grained document retrieval to fine-grained image selection and active region cropping, allowing the model to suppress irrelevant content and attend to information-dense regions. For effective end-to-end training, we introduce a dense multi-reward scheme that provides task-aware supervision for each action. Based on Group Relative Policy Optimization (GRPO), UniDoc-RL aligns agent behavior with multiple objectives without relying on a separate value network. To support this training paradigm, we curate a comprehensive dataset of high-quality reasoning trajectories with fine-grained action annotations. Experiments on three benchmarks demonstrate that UniDoc-RL consistently surpasses state-of-the-art baselines, yielding up to 17.7% gains over prior RL-based methods.
>
---
#### [replaced 033] Towards Design Compositing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.14605](https://arxiv.org/pdf/2604.14605)**

> **作者:** Abhinav Mahajan; Abhikhya Tripathy; Sudeeksha Reddy Pala; Vaibhav Methi; K J Joseph; Balaji Vasan Srinivasan
>
> **备注:** Accepted to CVEU workshop at CVPR 2026
>
> **摘要:** Graphic design creation involves harmoniously assembling multimodal components such as images, text, logos, and other visual assets collected from diverse sources, into a visually-appealing and cohesive design. Recent methods have largely focused on layout prediction or complementary element generation, while retaining input elements exactly, implicitly assuming that provided components are already stylistically harmonious. In practice, inputs often come from disparate sources and exhibit visual mismatch, making this assumption limiting. We argue that identity-preserving stylization and compositing of input elements is a critical missing ingredient for truly harmonized components-to-design pipelines. To this end, we propose GIST, a training-free, identity-preserving image compositor that sits between layout prediction and typography generation, and can be plugged into any existing components-to-design or design-refining pipeline without modification. We demonstrate this by integrating GIST with two substantially different existing methods, LaDeCo and Design-o-meter. GIST shows significant improvements in visual harmony and aesthetic quality across both pipelines, as validated by LLaVA-OV and GPT-4V on aspect-wise ratings and pairwise preference over naive pasting. Project Page: this http URL.
>
---
#### [replaced 034] Art3D: Training-Free 3D Generation from Flat-Colored Illustration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.10466](https://arxiv.org/pdf/2504.10466)**

> **作者:** Xiaoyan Cong; Jiayi Shen; Zekun Li; Rao Fu; Tao Lu; Srinath Sridhar
>
> **备注:** Technical Report. Project Page: this https URL
>
> **摘要:** Large-scale pre-trained image-to-3D generative models have exhibited remarkable capabilities in diverse shape generations. However, most of them struggle to synthesize plausible 3D assets when the reference image is flat-colored like hand drawings due to the lack of 3D illusion, which are often the most user-friendly input modalities in art content creation. To this end, we propose Art3D, a training-free method that can lift flat-colored 2D designs into 3D. By leveraging structural and semantic features with pre-trained 2D image generation models and a VLM-based realism evaluation, Art3D successfully enhances the three-dimensional illusion in reference images, thus simplifying the process of generating 3D from 2D, and proves adaptable to a wide range of painting styles. To benchmark the generalization performance of existing image-to-3D models on flat-colored images without 3D feeling, we collect a new dataset, Flat-2D, with over 100 samples. Experimental results demonstrate the performance and robustness of Art3D, exhibiting superior generalizable capacity and promising practical applicability. Our source code and dataset will be publicly available on our project page: this https URL .
>
---
#### [replaced 035] LLaMo: Scaling Pretrained Language Models for Unified Motion Understanding and Generation with Continuous Autoregressive Tokens
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.12370](https://arxiv.org/pdf/2602.12370)**

> **作者:** Zekun Li; Sizhe An; Chengcheng Tang; Chuan Guo; Ivan Shugurov; Linguang Zhang; Amy Zhao; Srinath Sridhar; Lingling Tao; Abhay Mittal
>
> **备注:** Project page: this https URL
>
> **摘要:** Recent progress in large models has led to significant advances in unified multimodal generation and understanding. However, the development of models that unify motion-language generation and understanding remains largely underexplored. Existing approaches often fine-tune large language models (LLMs) on paired motion-text data, which can result in catastrophic forgetting of linguistic capabilities due to the limited scale of available text-motion pairs. Furthermore, prior methods typically convert motion into discrete representations via quantization to integrate with language models, introducing substantial jitter artifacts from discrete tokenization. To address these challenges, we propose LLaMo, a unified framework that extends pretrained LLMs through a modality-specific Mixture-of-Transformers (MoT) architecture. This design inherently preserves the language understanding of the base model while enabling scalable multimodal adaptation. We encode human motion into a causal continuous latent space and maintain the next-token prediction paradigm in the decoder-only backbone through a lightweight flow-matching head, allowing for streaming motion generation in real-time (>30 FPS). Leveraging the comprehensive language understanding of pretrained LLMs and large-scale motion-text pretraining, our experiments demonstrate that LLaMo achieves high-fidelity text-to-motion generation and motion-to-text captioning in general settings, especially zero-shot motion generation, marking a significant step towards a general unified motion-language large model.
>
---
#### [replaced 036] PILOT: A Promptable Interleaved Layout-aware OCR Transformer
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2504.03621](https://arxiv.org/pdf/2504.03621)**

> **作者:** Laziz Hamdi; Amine Tamasna; Pascal Boisson; Thierry Paquet
>
> **摘要:** Classical OCR pipelines decompose document reading into detection, segmentation, and recognition stages, which makes them sensitive to localization errors and difficult to extend to interactive querying. This work investigates whether a single compact model can jointly perform text recognition and spatial grounding on both handwritten and printed documents. We introduce PILOT, a 155M-parameter prompt-conditioned generative model that formulates document OCR as unified sequence generation. A lightweight depthwise-separable CNN encodes the page, and a Transformer decoder autoregressively emits a single stream of subword and quantized absolute-coordinate tokens on a 10\,px grid, enabling full-page OCR, region-conditioned reading, and query-by-string spotting within the same architecture. A three-stage curriculum, progressing from plain transcription to joint text-and-box generation and finally to prompt-controlled extraction, stabilizes training and improves spatial grounding. Experiments on IAM, RIMES~2009, SROIE~2019, and the heterogeneous MAURDOR benchmark show that PILOT achieves competitive or superior performance in text recognition and line-level detection compared with traditional OCR systems, recent end-to-end HTR models, and compact vision--language models, while remaining substantially smaller than billion-scale multimodal models. Additional evaluations on fine-grained OCR and query-by-string spotting further confirm that a unified text--layout decoder can provide accurate and efficient promptable OCR in a compact setting. To support reproducibility, we release the synthetic SROIE generator, the 500k annotated IDL/PDFA pages, the harmonized line-level annotations for IAM, RIMES~2009, and MAURDOR, and the source code at this https URL.
>
---
#### [replaced 037] SignX: Continuous Sign Recognition in Compact Pose-Rich Latent Space
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于连续手语识别任务，旨在解决手语视频处理复杂和计算消耗大的问题。通过构建紧凑的姿势潜在空间，实现高效准确的手语识别。**

- **链接: [https://arxiv.org/pdf/2504.16315](https://arxiv.org/pdf/2504.16315)**

> **作者:** Sen Fang; Yalin Feng; Chunyu Sui; Hongbin Zhong; Yanxin Zhang; Hongwei Yi; Hezhen Hu; Dimitris N. Metaxas
>
> **备注:** 33 pages, CSLR SOTA (2026). More demo at this https URL
>
> **摘要:** The complexity of Sign Language (SL) data processing brings many challenges. The current approach to recognition of SL signs aims to translate RGB sign language videos through pose information into Word-based ID Glosses, which serve to uniquely identify signs. This paper proposes SignX, a novel framework for continuous sign language recognition (SLR) in compact pose-rich latent space. First, we construct a unified latent representation that encodes heterogeneous pose formats (SMPLer-X, DWPose, Mediapipe, PrimeDepth, and Sapiens Segmentation) into a compact, information-dense space. Second, we train a ViT-based Video-to-Pose module to extract this latent representation directly from raw videos. Finally, we develop a temporal modeling and sequence refinement method that operates entirely in this latent space. This multi-stage design achieves end-to-end SLR while significantly reducing computational consumption. Experimental results demonstrate that SignX achieves SOTA accuracy on continuous SLR and Translation task, delivering nearly a 50-fold acceleration over pixel-space baselines.
>
---
#### [replaced 038] VAGNet: Vision-based Accident Anticipation with Global Features
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.09305](https://arxiv.org/pdf/2604.09305)**

> **作者:** Vipooshan Vipulananthan; Charith D. Chitraranjan
>
> **摘要:** Traffic accidents are a leading cause of fatalities and injuries across the globe. Therefore, the ability to anticipate hazardous situations in advance is essential. Automated accident anticipation enables timely intervention through driver alerts and collision avoidance maneuvers, forming a key component of advanced driver assistance systems. In autonomous driving, such predictive capabilities support proactive safety behaviors, such as initiating defensive driving and human takeover when required. Using dashcam video as input offers a cost-effective solution, but it is challenging due to the complexity of real-world driving scenes. Accident anticipation systems need to operate in real-time. However, current methods involve extracting features from each detected object, which is computationally intensive. We propose VAGNet, a deep neural network that learns to predict accidents from dash-cam video using global features of traffic scenes without requiring explicit object-level features. The network consists of transformer and graph modules, and we use the vision foundation model VideoMAE-V2 for global feature extraction. Experiments on four benchmark datasets (DAD, DoTA, DADA, and Nexar) show that our method anticipates accidents with higher average precision and mean time-to-accident while being computationally more efficient compared to existing methods.
>
---
#### [replaced 039] Hybrid Latents: Geometry-Appearance-Aware Surfel Splatting
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2604.14928](https://arxiv.org/pdf/2604.14928)**

> **作者:** Neel Kelkar; Simon Niedermayr; Klaus Engel; Rüdiger Westermann
>
> **备注:** 22 pages, 9 figures
>
> **摘要:** We introduce a hybrid Gaussian-hash-grid radiance representation for reconstructing 2D Gaussian scene models from multi-view images. Similar to NeST splatting, our approach reduces the entanglement between geometry and appearance common in NeRF-based models, but adds per-Gaussian latent features alongside hash-grid features to bias the optimizer toward a separation of low- and high-frequency scene components. This explicit frequency-based decomposition reduces the tendency of high-frequency texture to compensate for geometric errors. Encouraging Gaussians with hard opacity falloffs further strengthens the separation between geometry and appearance, improving both geometry reconstruction and rendering efficiency. Finally, probabilistic pruning combined with a sparsity-inducing BCE opacity loss allows redundant Gaussians to be turned off, yielding a minimal set of Gaussians sufficient to represent the scene. Using both synthetic and real-world datasets, we compare against the state of the art in Gaussian-based novel-view synthesis and demonstrate superior reconstruction fidelity with an order of magnitude fewer primitives.
>
---
#### [replaced 040] Cross-modal learning for plankton recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.16427](https://arxiv.org/pdf/2603.16427)**

> **作者:** Joona Kareinen; Veikka Immonen; Tuomas Eerola; Lumi Haraguchi; Lasse Lensu; Kaisa Kraft; Sanna Suikkanen; Heikki Kälviäinen
>
> **摘要:** This paper considers self-supervised cross-modal coordination as a strategy enabling utilization of multiple modalities and large volumes of unlabeled plankton data to build models for plankton recognition. Automated imaging instruments facilitate the continuous collection of plankton image data on a large scale. Current methods for automatic plankton image recognition rely primarily on supervised approaches, which require labeled training sets that are labor-intensive to collect. On the other hand, some modern plankton imaging instruments complement image information with optical measurement data, such as scatter and fluorescence profiles, which currently are not widely utilized in plankton recognition. In this work, we explore the possibility of using such measurement data to guide the learning process without requiring manual labeling. Inspired by the concepts behind Contrastive Language-Image Pre-training, we train encoders for both modalities using only binary supervisory information indicating whether a given image and profile originate from the same particle or from different particles. For plankton recognition, we employ a small labeled gallery of known plankton species combined with a $k$-NN classifier. This approach yields a recognition model that is inherently multimodal, i.e., capable of utilizing information extracted from both image and profile data. We demonstrate that the proposed method achieves high recognition accuracy while requiring only a minimal number of labeled images. Furthermore, we show that the approach outperforms an image-only self-supervised baseline. Code available at this https URL.
>
---
#### [replaced 041] Comparison Study: Glacier Calving Front Delineation in Synthetic Aperture Radar Images With Deep Learning
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2501.05281](https://arxiv.org/pdf/2501.05281)**

> **作者:** Nora Gourmelon; Konrad Heidler; Erik Loebel; Daniel Cheng; Julian Klink; Anda Dong; Fei Wu; Noah Maul; Moritz Koch; Marcel Dreier; Dakota Pyles; Thorsten Seehaus; Matthias Braun; Andreas Maier; Vincent Christlein
>
> **备注:** Accepted as short paper in IEEE Transactions on Pattern Analysis and Machine Intelligence
>
> **摘要:** Continuous monitoring of glacier calving fronts is essential for sea level rise projections. This study benchmarks Deep Learning systems for front delineation in Synthetic Aperture Radar imagery. While Deep Learning systems exhibit errors up to 221 m, human annotators deviate by only 38 m, underscoring the need for further research.
>
---
#### [replaced 042] When Cultures Meet: Multicultural Text-to-Image Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2502.15972](https://arxiv.org/pdf/2502.15972)**

> **作者:** Parth Bhalerao; Mounika Yalamarty; Brian Trinh; Oana Ignat
>
> **摘要:** Text-to-image generation models have achieved strong performance in culturally homogeneous settings, yet their ability to generate multicultural scenes, where people and landmarks originate from different cultures, remains largely unexplored. We introduce multicultural text-to-image generation as a new task and present the first benchmark designed to study this setting. Our dataset contains 9,000 images spanning five countries, three age groups, two genders, 25 historical landmarks, and five languages. Using this benchmark, we analyze the behavior of state-of-the-art text-to-image models across multiple dimensions, including alignment, image quality, aesthetics, knowledge, and fairness. As one strategy for composing cultural and demographic information, we explore MosAIG, a Multi-Agent framework that enhances multicultural Image Generation by leveraging LLMs with distinct cultural personas. Our analysis shows that richer prompt composition can improve image quality and cultural grounding compared to simple prompts, while revealing substantial disparities across languages and demographic groups. We release our dataset and code at this https URL.
>
---
#### [replaced 043] FoodSense: A Multisensory Food Dataset and Benchmark for Predicting Taste, Smell, Texture, and Sound from Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.14388](https://arxiv.org/pdf/2604.14388)**

> **作者:** Sabab Ishraq; Aarushi Aarushi; Juncai Jiang; Chen Chen
>
> **摘要:** Humans routinely infer taste, smell, texture, and even sound from food images a phenomenon well studied in cognitive science. However, prior vision language research on food has focused primarily on recognition tasks such as meal identification, ingredient detection, and nutrition estimation. Image-based prediction of multisensory experience remains largely unexplored. We introduce FoodSense, a human-annotated dataset for cross-sensory inference containing 66,842 participant-image pairs across 2,987 unique food images. Each pair includes numeric ratings (1-5) and free-text descriptors for four sensory dimensions: taste, smell, texture, and sound. To enable models to both predict and explain sensory expectations, we expand short human annotations into image-grounded reasoning traces. A large language model generates visual justifications conditioned on the image, ratings, and descriptors. Using these annotations, we train FoodSense-VL, a vision language benchmark model to produce both multisensory ratings and grounded explanations directly from food images. This work connects cognitive science findings on cross-sensory perception with modern instruction tuning for multimodal models and shows that many popular evaluation metrics are insufficient for visually sensory inference.
>
---
#### [replaced 044] When Surfaces Lie: Exploiting Wrinkle-Induced Attention Shift to Attack Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.27759](https://arxiv.org/pdf/2603.27759)**

> **作者:** Chengyin Hu; Xuemeng Sun; Jiaju Han; Qike Zhang; Xiang Chen; Xin Wang; Yiwei Wei; Jiahua Long
>
> **摘要:** Visual-Language Models (VLMs) have demonstrated exceptional cross-modal understanding across various tasks, including zero-shot classification, image captioning, and visual question answering. However, their robustness to physically plausible non-rigid deformations-such as wrinkles on flexible surfaces-remains poorly understood. In this work, we propose a parametric structural perturbation method inspired by the mechanics of three-dimensional fabric wrinkles. Specifically, our method generates photorealistic non-rigid perturbations by constructing multi-scale wrinkle fields and integrating displacement field distortion with surface-consistent appearance variations. To achieve an optimal balance between visual naturalness and adversarial effectiveness, we design a hierarchical fitness function in a low-dimensional parameter space and employ an optimization-based search strategy. We evaluate our approach using a two-stage framework: perturbations are first optimized on a zero-shot classification proxy task and subsequently assessed for transferability on generative tasks. Experimental results demonstrate that our method significantly degrades the performance of various state-of-the-art VLMs, consistently outperforming baselines in both image captioning and visual question-answering tasks.
>
---
#### [replaced 045] GlobalSplat: Efficient Feed-Forward 3D Gaussian Splatting via Global Scene Tokens
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.15284](https://arxiv.org/pdf/2604.15284)**

> **作者:** Roni Itkin; Noam Issachar; Yehonatan Keypur; Xingyu Chen; Anpei Chen; Sagie Benaim
>
> **摘要:** The efficient spatial allocation of primitives serves as the foundation of 3D Gaussian Splatting, as it directly dictates the synergy between representation compactness, reconstruction speed, and rendering fidelity. Previous solutions, whether based on iterative optimization or feed-forward inference, suffer from significant trade-offs between these goals, mainly due to the reliance on local, heuristic-driven allocation strategies that lack global scene awareness. Specifically, current feed-forward methods are largely pixel-aligned or voxel-aligned. By unprojecting pixels into dense, view-aligned primitives, they bake redundancy into the 3D asset. As more input views are added, the representation size increases and global consistency becomes fragile. To this end, we introduce GlobalSplat, a framework built on the principle of align first, decode later. Our approach learns a compact, global, latent scene representation that encodes multi-view input and resolves cross-view correspondences before decoding any explicit 3D geometry. Crucially, this formulation enables compact, globally consistent reconstructions without relying on pretrained pixel-prediction backbones or reusing latent features from dense baselines. Utilizing a coarse-to-fine training curriculum that gradually increases decoded capacity, GlobalSplat natively prevents representation bloat. On RealEstate10K and ACID, our model achieves competitive novel-view synthesis performance while utilizing as few as 16K Gaussians, significantly less than required by dense pipelines, obtaining a light 4MB footprint. Further, GlobalSplat enables significantly faster inference than the baselines, operating under 78 milliseconds in a single forward pass. Project page is available at this https URL
>
---
#### [replaced 046] OmniShow: Unifying Multimodal Conditions for Human-Object Interaction Video Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.11804](https://arxiv.org/pdf/2604.11804)**

> **作者:** Donghao Zhou; Guisheng Liu; Hao Yang; Jiatong Li; Jingyu Lin; Xiaohu Huang; Yichen Liu; Xin Gao; Cunjian Chen; Shilei Wen; Chi-Wing Fu; Pheng-Ann Heng
>
> **备注:** Project page: this https URL
>
> **摘要:** In this work, we study Human-Object Interaction Video Generation (HOIVG), which aims to synthesize high-quality human-object interaction videos conditioned on text, reference images, audio, and pose. This task holds significant practical value for automating content creation in real-world applications, such as e-commerce demonstrations, short video production, and interactive entertainment. However, existing approaches fail to accommodate all these requisite conditions. We present OmniShow, an end-to-end framework tailored for this practical yet challenging task, capable of harmonizing multimodal conditions and delivering industry-grade performance. To overcome the trade-off between controllability and quality, we introduce Unified Channel-wise Conditioning for efficient image and pose injection, and Gated Local-Context Attention to ensure precise audio-visual synchronization. To effectively address data scarcity, we develop a Decoupled-Then-Joint Training strategy that leverages a multi-stage training process with model merging to efficiently harness heterogeneous sub-task datasets. Furthermore, to fill the evaluation gap in this field, we establish HOIVG-Bench, a dedicated and comprehensive benchmark for HOIVG. Extensive experiments demonstrate that OmniShow achieves overall state-of-the-art performance across various multimodal conditioning settings, setting a solid standard for the emerging HOIVG task.
>
---
#### [replaced 047] Cross-Modal Emotion Transfer for Emotion Editing in Talking Face Video
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2604.07786](https://arxiv.org/pdf/2604.07786)**

> **作者:** Chanhyuk Choi; Taesoo Kim; Donggyu Lee; Siyeol Jung; Taehwan Kim
>
> **备注:** Accepted to CVPR 2026. Project Page: this https URL
>
> **摘要:** Talking face generation has gained significant attention as a core application of generative models. To enhance the expressiveness and realism of synthesized videos, emotion editing in talking face video plays a crucial role. However, existing approaches often limit expressive flexibility and struggle to generate extended emotions. Label-based methods represent emotions with discrete categories, which fail to capture a wide range of emotions. Audio-based methods can leverage emotionally rich speech signals - and even benefit from expressive text-to-speech (TTS) synthesis - but they fail to express the target emotions because emotions and linguistic contents are entangled in emotional speeches. Images-based methods, on the other hand, rely on target reference images to guide emotion transfer, yet they require high-quality frontal views and face challenges in acquiring reference data for extended emotions (e.g., sarcasm). To address these limitations, we propose Cross-Modal Emotion Transfer (C-MET), a novel approach that generates facial expressions based on speeches by modeling emotion semantic vectors between speech and visual feature spaces. C-MET leverages a large-scale pretrained audio encoder and a disentangled facial expression encoder to learn emotion semantic vectors that represent the difference between two different emotional embeddings across modalities. Extensive experiments on the MEAD and CREMA-D datasets demonstrate that our method improves emotion accuracy by 14% over state-of-the-art methods, while generating expressive talking face videos - even for unseen extended emotions. Code, checkpoint, and demo are available at this https URL
>
---
#### [replaced 048] Enhancing Mixture-of-Experts Specialization via Cluster-Aware Upcycling
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.13508](https://arxiv.org/pdf/2604.13508)**

> **作者:** Sanghyeok Chu; Pyunghwan Ahn; Gwangmo Song; SeungHwan Kim; Honglak Lee; Bohyung Han
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Sparse Upcycling provides an efficient way to initialize a Mixture-of-Experts (MoE) model from pretrained dense weights instead of training from scratch. However, since all experts start from identical weights and the router is randomly initialized, the model suffers from expert symmetry and limited early specialization. We propose Cluster-aware Upcycling, a strategy that incorporates semantic structure into MoE initialization. Our method first partitions the dense model's input activations into semantic clusters. Each expert is then initialized using the subspace representations of its corresponding cluster via truncated SVD, while setting the router's initial weights to the cluster centroids. This cluster-aware initialization breaks expert symmetry and encourages early specialization aligned with the data distribution. Furthermore, we introduce an expert-ensemble self-distillation loss that stabilizes training by providing reliable routing guidance using an ensemble teacher. When evaluated on CLIP ViT-B/32 and ViT-B/16, Cluster-aware Upcycling consistently outperforms existing methods across both zero-shot and few-shot benchmarks. The proposed method also produces more diverse and disentangled expert representations, reduces inter-expert similarity, and leads to more confident routing behavior. Project page: this https URL
>
---
#### [replaced 049] CASR: A Robust Cyclic Framework for Arbitrary Large-Scale Super-Resolution with Distribution Alignment and Self-Similarity Awareness
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.22159](https://arxiv.org/pdf/2602.22159)**

> **作者:** Wenhao Guo; Zhaoran Zhao; Peng Lu; Sheng Li; Qian Qiao; DeRui Li
>
> **摘要:** Arbitrary-Scale SR (ASISR) remains fundamentally limited by cross-scale distribution shift: once the inference scale leaves the training range, noise, blur, and artifacts accumulate sharply. We revisit this challenge from a cross-scale distribution transition perspective and propose CASR, a simple yet highly efficient cyclic SR framework that reformulates ultra-magnification as a sequence of in-distribution scale transitions. This design ensures stable inference at arbitrary scales while requiring only a single model. CASR tackles two major bottlenecks: distribution drift across iterations and patch-wise diffusion inconsistencies. The proposed SSAM module aligns structural distributions via superpixel aggregation, preventing error accumulation, while SARM module restores high-frequency textures by enforcing correlation-guided consistency and preserving self-similarity structure through correlation alignment. Despite using only a single model, our approach significantly reduces distribution drift, preserves long-range texture consistency, and achieves superior generalization even at extreme magnification.
>
---
#### [replaced 050] ReXSonoVQA: A Video QA Benchmark for Procedure-Centric Ultrasound Understanding
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.10916](https://arxiv.org/pdf/2604.10916)**

> **作者:** Xucheng Wang; Xiaoman Zhang; Sung Eun Kim; Ankit Pal; Pranav Rajpurkar
>
> **摘要:** Ultrasound acquisition requires skilled probe manipulation and real-time adjustments. Vision-language models (VLMs) could enable autonomous ultrasound systems, but existing benchmarks evaluate only static images, not dynamic procedural understanding. We introduce ReXSonoVQA, a video QA benchmark with 514 video clips and 514 questions (249 MCQ, 265 free-response) targeting three competencies: Action-Goal Reasoning, Artifact Resolution & Optimization, and Procedure Context & Planning. Zero-shot evaluation of Gemini 3 Pro, Qwen3.5-397B, LLaVA-Video-72B, and Seed 2.0 Pro shows VLMs can extract some procedural information, but troubleshooting questions remain challenging with minimal gains over text-only baselines, exposing limitations in causal reasoning. ReXSonoVQA enables developing perception systems for ultrasound training, guidance, and robotic automation.
>
---
#### [replaced 051] A Single Image and Multimodality Is All You Need for Novel View Synthesis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.17909](https://arxiv.org/pdf/2602.17909)**

> **作者:** Amirhosein Javadi; Chi-Shiang Gau; Konstantinos D. Polyzos; Tara Javidi
>
> **摘要:** Diffusion-based approaches have recently demonstrated strong performance for single-image novel view synthesis by conditioning generative models on geometry inferred from monocular depth estimation. However, in practice, the quality and consistency of the synthesized views are fundamentally limited by the reliability of the underlying depth estimates, which are often fragile under low-texture, adverse weather, and occlusion-heavy real-world conditions. In this work, we show that incorporating sparse multimodal range measurements provides a simple yet effective way to overcome these limitations. We introduce a multimodal depth reconstruction framework that leverages extremely sparse range sensing data, such as automotive radar or LiDAR, to produce dense depth maps that serve as robust geometric conditioning for diffusion-based novel view synthesis. Our approach models depth in an angular domain using a localized Gaussian Process formulation, enabling computationally efficient inference while explicitly quantifying uncertainty in regions with limited observations. The reconstructed depth and uncertainty are used as a drop-in replacement for monocular depth estimators in existing diffusion-based rendering pipelines, without modifying the generative model itself. Experiments on real-world multimodal driving scenes demonstrate that replacing vision-only depth with our sparse range-based reconstruction substantially improves both geometric consistency and visual quality in single-image novel-view video generation. These results highlight the importance of reliable geometric priors for diffusion-based view synthesis and demonstrate the practical benefits of multimodal sensing even at extreme levels of sparsity. Code is publicly available at: this https URL
>
---
#### [replaced 052] Training Flow Matching: The Role of Weighting and Parameterization
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.06454](https://arxiv.org/pdf/2603.06454)**

> **作者:** Anne Gagneux; Ségolène Martin; Rémi Gribonval; Mathurin Massias
>
> **备注:** Published as a paper at the 2nd DeLTa Workshop, ICLR 2026
>
> **摘要:** We study the training objectives of denoising-based generative models, with a particular focus on loss weighting and output parameterization, including noise-, clean image-, and velocity-based formulations. Through a systematic numerical study, we analyze how these training choices interact with the intrinsic dimensionality of the data manifold, model architecture, and dataset size. Our experiments span synthetic datasets with controlled geometry as well as image data, and compare training objectives using quantitative metrics for denoising accuracy (PSNR across noise levels) and generative quality (FID). Rather than proposing a new method, our goal is to disentangle the various factors that matter when training a flow matching model, in order to provide practical insights on design choices.
>
---
#### [replaced 053] HQF-Net: A Hybrid Quantum-Classical Multi-Scale Fusion Network for Remote Sensing Image Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2604.06715](https://arxiv.org/pdf/2604.06715)**

> **作者:** Md Aminur Hossain; Ayush V. Patel; Siddhant Gole; Sanjay K. Singh; Biplab Banerjee
>
> **备注:** 17 pages
>
> **摘要:** Remote sensing semantic segmentation requires models that can jointly capture fine spatial details and high-level semantic context across complex scenes. While classical encoder-decoder architectures such as U-Net remain strong baselines, they often struggle to fully exploit global semantics and structured feature interactions. In this work, we propose HQF-Net, a hybrid quantum-classical multi-scale fusion network for remote sensing image segmentation. HQF-Net integrates multi-scale semantic guidance from a frozen DINOv3 ViT-L/16 backbone with a customized U-Net architecture through a Deformable Multiscale Cross-Attention Fusion (DMCAF) module. To enhance feature refinement, the framework further introduces quantum-enhanced skip connections (QSkip) and a Quantum bottleneck with Mixture-of-Experts (QMoE), which combines complementary local, global, and directional quantum circuits within an adaptive routing mechanism. Experiments on three remote sensing benchmarks show consistent improvements with the proposed design. HQF-Net achieves 0.8568 mIoU and 96.87% overall accuracy on this http URL, 71.82% mIoU on OpenEarthMap, and 55.28% mIoU with 99.37% overall accuracy on SeasoNet. An architectural ablation study further confirms the contribution of each major component. These results show that structured hybrid quantum-classical feature processing is a promising direction for improving remote sensing semantic segmentation under near-term quantum constraints.
>
---
#### [replaced 054] Think in Latent Thoughts: A New Paradigm for Gloss-Free Sign Language Translation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.15301](https://arxiv.org/pdf/2604.15301)**

> **作者:** Yiyang Jiang; Li Zhang; Xiao-Yong Wei; Li Qing
>
> **备注:** Accepted to ACL 2026 Main
>
> **摘要:** Many SLT systems quietly assume that brief chunks of signing map directly to spoken-language words. That assumption breaks down because signers often create meaning on the fly using context, space, and movement. We revisit SLT and argue that it is mainly a cross-modal reasoning task, not just a straightforward video-to-text conversion. We thus introduce a reasoning-driven SLT framework that uses an ordered sequence of latent thoughts as an explicit middle layer between the video and the generated text. These latent thoughts gradually extract and organize meaning over time. On top of this, we use a plan-then-ground decoding method: the model first decides what it wants to say, and then looks back at the video to find the evidence. This separation improves coherence and faithfulness. We also built and released a new large-scale gloss-free SLT dataset with stronger context dependencies and more realistic meanings. Experiments across several benchmarks show consistent gains over existing gloss-free methods. Our code and data are available at this https URL.
>
---
#### [replaced 055] GenHSI: Controllable Generation of Human-Scene Interaction Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.19840](https://arxiv.org/pdf/2506.19840)**

> **作者:** Zekun Li; Rui Zhou; Rahul Sajnani; Xiaoyan Cong; Daniel Ritchie; Srinath Sridhar
>
> **摘要:** Large-scale pre-trained video diffusion models have exhibited remarkable capabilities in diverse video generation. However, existing solutions face several challenges in generating long videos with rich human-scene interactions (HSI), including unrealistic dynamics and affordance, lack of subject identity preservation, and the need for expensive training. To this end, we propose GenHSI, a training-free method for controllable generation of long HSI videos with 3D awareness. Taking inspiration from movie animation, we subdivide the video synthesis into three stages: (1) script writing, (2) pre-visualization, and (3) animation. Given an image of a scene and a character with a user description, we use these three stages to generate long videos that preserve human identity and provide rich and plausible HSI. Script writing converts a complex text prompt involving a chain of HSI into simple atomic actions that are used in the pre-visualization stage to generate 3D keyframes. To synthesize plausible human interaction poses in 3D keyframes, we utilize pre-trained 2D inpainting diffusion models to generate plausible 2D human interactions based on view canonicalization, which eliminates the need for multi-view fitting in previous works. We then extend these interactions to 3D using robust iterative optimization, informed by contact cues and reasoning from VLMs. Prompted by these 3D keyframes, the pretrained video diffusion models can better generate consistent long videos with plausible dynamics and affordance in a 3D-aware manner. We are the first to synthesize a long video sequence with a chain of HSI actions without training based on the image references of the scene and character. Experiments demonstrate that our method can generate HSI videos that effectively preserve scene content and character identity with plausible human-scene interaction from a single image scene.
>
---
#### [replaced 056] Scalable Unseen Objects 6-DoF Absolute Pose Estimation with Robotic Integration
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人操作任务，解决未知物体6-DoF位姿估计问题。通过单张带标注的RGB-D图像实现可扩展的位姿估计，并构建软硬件结合的机器人系统。**

- **链接: [https://arxiv.org/pdf/2503.05578](https://arxiv.org/pdf/2503.05578)**

> **作者:** Jian Liu; Wei Sun; Kai Zeng; Jin Zheng; Hui Yang; Hossein Rahmani; Ajmal Mian; Lin Wang
>
> **备注:** Accepted by TRO 2026, 18 pages, 9 figures
>
> **摘要:** Pose estimation-guided unseen object 6-DoF robotic manipulation is a key task in robotics. However, the scalability of current pose estimation methods to unseen objects remains a fundamental challenge, as they generally rely on CAD models or dense reference views of unseen objects, which are difficult to acquire, ultimately limit their scalability. In this paper, we introduce a novel task setup, referred to as SinRef-6D, which addresses 6-DoF absolute pose estimation for unseen objects using only a single pose-labeled reference RGB-D image captured during robotic manipulation. This setup is more scalable yet technically nontrivial due to large pose discrepancies and the limited geometric and spatial information contained in a single view. To address these issues, our key idea is to iteratively establish point-wise alignment in a common coordinate system with state space models (SSMs) as backbones. Specifically, to handle large pose discrepancies, we introduce an iterative object-space point-wise alignment strategy. Then, Point and RGB SSMs are proposed to capture long-range spatial dependencies from a single view, offering superior spatial modeling capability with linear complexity. Once pre-trained on synthetic data, SinRef-6D can estimate the 6-DoF absolute pose of an unseen object using only a single reference view. With the estimated pose, we further develop a hardware-software robotic system and integrate the proposed SinRef-6D into it in real-world settings. Extensive experiments on six benchmarks and in diverse real-world scenarios demonstrate that our SinRef-6D offers superior scalability. Additional robotic grasping experiments further validate the effectiveness of the developed robotic system. The code and robotic demos are available at this https URL.
>
---
#### [replaced 057] TokenLight: Precise Lighting Control in Images using Attribute Tokens
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2604.15310](https://arxiv.org/pdf/2604.15310)**

> **作者:** Sumit Chaturvedi; Yannick Hold-Geoffroy; Mengwei Ren; Jingyuan Liu; He Zhang; Yiqun Mei; Julie Dorsey; Zhixin Shu
>
> **备注:** 32 pages, CVPR 2026, Project Page: this https URL
>
> **摘要:** This paper presents a method for image relighting that enables precise and continuous control over multiple illumination attributes in a photograph. We formulate relighting as a conditional image generation task and introduce attribute tokens to encode distinct lighting factors such as intensity, color, ambient illumination, diffuse level, and 3D light positions. The model is trained on a large-scale synthetic dataset with ground-truth lighting annotations, supplemented by a small set of real captures to enhance realism and generalization. We validate our approach across a variety of relighting tasks, including controlling in-scene lighting fixtures and editing environment illumination using virtual light sources, on synthetic and real images. Our method achieves state-of-the-art quantitative and qualitative performance compared to prior work. Remarkably, without explicit inverse rendering supervision, the model exhibits an inherent understanding of how light interacts with scene geometry, occlusion, and materials, yielding convincing lighting effects even in traditionally challenging scenarios such as placing lights within objects or relighting transparent materials plausibly. Project page: this http URL
>
---
#### [replaced 058] Proper Body Landmark Subset Enables More Accurate and 5X Faster Recognition of Isolated Signs in LIBRAS
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.24887](https://arxiv.org/pdf/2510.24887)**

> **作者:** Daniele L. V. dos Santos; Thiago B. Pereira; Carlos Eduardo G. R. Alves; Richard J. M. G. Tello; Francisco de A. Boldt; Thiago M. Paixão
>
> **备注:** This work was accepted for presentation at IEEE SAS 2026
>
> **摘要:** This paper examines the feasibility of utilizing lightweight body landmark detection for recognizing isolated signs in Brazilian Sign Language (LIBRAS). Although the use of skeleton-image representation has enabled substantial improvements in recognition performance, the use of OpenPose for landmark extraction hindered time performance. In a preliminary investigation, we observed that simply replacing OpenPose with lightweight MediaPipe, while improving processing speed, significantly reduced accuracy. To overcome this limitation, we explored landmark subset selection strategies to optimize recognition performance. Experimental results show that a proper landmark subset achieves comparable or superior performance to state-of-the-art methods while reducing processing time by more than 5X. As an additional contribution, we demonstrate that spline-based imputation effectively mitigates missing landmark issues, leading to substantial accuracy gains.
>
---
#### [replaced 059] StreamCacheVGGT: Streaming Visual Geometry Transformers with Robust Scoring and Hybrid Cache Compression
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2604.15237](https://arxiv.org/pdf/2604.15237)**

> **作者:** Xuanyi Liu; Chunan Yu; Deyi Ji; Qi Zhu; Lingyun Sun; Xuanfu Li; Jin Ma; Tianrun Chen; Lanyun Zhu
>
> **摘要:** Reconstructing dense 3D geometry from continuous video streams requires stable inference under a constant memory budget. Existing $O(1)$ frameworks primarily rely on a ``pure eviction'' paradigm, which suffers from significant information destruction due to binary token deletion and evaluation noise from localized, single-layer scoring. To address these bottlenecks, we propose StreamCacheVGGT, a training-free framework that reimagines cache management through two synergistic modules: Cross-Layer Consistency-Enhanced Scoring (CLCES) and Hybrid Cache Compression (HCC). CLCES mitigates activation noise by tracking token importance trajectories across the Transformer hierarchy, employing order-statistical analysis to identify sustained geometric salience. Leveraging these robust scores, HCC transcends simple eviction by introducing a three-tier triage strategy that merges moderately important tokens into retained anchors via nearest-neighbor assignment on the key-vector manifold. This approach preserves essential geometric context that would otherwise be lost. Extensive evaluations on five benchmarks (7-Scenes, NRGBD, ETH3D, Bonn, and KITTI) demonstrate that StreamCacheVGGT sets a new state-of-the-art, delivering superior reconstruction accuracy and long-term stability while strictly adhering to constant-cost constraints.
>
---
#### [replaced 060] OSCBench: Benchmarking Object State Change in Text-to-Video Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于文本到视频生成任务，旨在解决对象状态变化（OSC）评估问题。提出OSCBench基准，评估模型在不同场景下的 OSC 性能。**

- **链接: [https://arxiv.org/pdf/2603.11698](https://arxiv.org/pdf/2603.11698)**

> **作者:** Xianjing Han; Bin Zhu; Shiqi Hu; Franklin Mingzhe Li; Patrick Carrington; Roger Zimmermann; Jingjing Chen
>
> **备注:** ACL 2026 Main Conference, Project page: this https URL
>
> **摘要:** Text-to-video (T2V) generation models have made rapid progress in producing visually high-quality and temporally coherent videos. However, existing benchmarks primarily focus on perceptual quality, text-video alignment, or physical plausibility, leaving a critical aspect of action understanding largely unexplored: object state change (OSC) explicitly specified in the text prompt. OSC refers to the transformation of an object's state induced by an action, such as peeling a potato or slicing a lemon. In this paper, we introduce OSCBench, a benchmark specifically designed to assess OSC performance in T2V models. OSCBench is constructed from instructional cooking data and systematically organizes action-object interactions into regular, novel, and compositional scenarios to probe both in-distribution performance and generalization. We evaluate six representative open-source and proprietary T2V models using both human user study and multimodal large language model (MLLM)-based automatic evaluation. Our results show that, despite strong performance on semantic and scene alignment, current T2V models consistently struggle with accurate and temporally consistent object state changes, especially in novel and compositional settings. These findings position OSC as a key bottleneck in text-to-video generation and establish OSCBench as a diagnostic benchmark for advancing state-aware video generation models.
>
---
#### [replaced 061] MM-Telco: Benchmarks and Multimodal Large Language Models for Telecom Applications
- **分类: cs.AI; cs.CV; cs.ET; cs.NI**

- **链接: [https://arxiv.org/pdf/2511.13131](https://arxiv.org/pdf/2511.13131)**

> **作者:** Anshul Kumar; Gagan Raj Gupta; Manish Rai; Apu Chakraborty; Ashutosh Modi; Abdelaali Chaoub; Soumajit Pramanik; Moyank Giri; Yashwanth Holla; Sunny Kumar; M. V. Kiran Sooraj
>
> **摘要:** Large Language Models (LLMs) have emerged as powerful tools for automating complex reasoning and decision-making tasks. In telecommunications, they hold the potential to transform network optimization, automate troubleshooting, enhance customer support, and ensure regulatory compliance. However, their deployment in telecom is hindered by domain-specific challenges that demand specialized adaptation. To overcome these challenges and to accelerate the adaptation of LLMs for telecom, we propose MM-Telco, a comprehensive suite of multimodal benchmarks and models tailored for the telecom domain. The benchmark introduces various tasks (both text based and image based) that address various practical real-life use cases such as network operations, network management, improving documentation quality, and retrieval of relevant text and images. Further, we perform baseline experiments with various LLMs and VLMs. The models fine-tuned on our dataset exhibit a significant boost in performance. Our experiments also help analyze the weak areas in the working of current state-of-art multimodal LLMs, thus guiding towards further development and research.
>
---
#### [replaced 062] DualTrack: Sensorless 3D Ultrasound needs Local and Global Context
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.09530](https://arxiv.org/pdf/2509.09530)**

> **作者:** Paul F. R. Wilson; Matteo Ronchetti; Rüdiger Göbl; Viktoria Markova; Sebastian Rosenzweig; Raphael Prevost; Parvin Mousavi; Oliver Zettinig
>
> **摘要:** Three-dimensional ultrasound (US) offers many clinical advantages over conventional 2D imaging, yet its widespread adoption is limited by the cost and complexity of traditional 3D systems. Sensorless 3D US, which uses deep learning to estimate a 3D probe trajectory from a sequence of 2D US images, is a promising alternative. Local features, such as speckle patterns, can help predict frame-to-frame motion, while global features, such as coarse shapes and anatomical structures, can situate the scan relative to anatomy and help predict its general shape. In prior approaches, global features are either ignored or tightly coupled with local feature extraction, restricting the ability to robustly model these two complementary aspects. We propose DualTrack, a novel dual-encoder architecture that leverages decoupled local and global encoders specialized for their respective scales of feature extraction. The local encoder uses dense spatiotemporal convolutions to capture fine-grained features, while the global encoder utilizes an image backbone (e.g., a 2D CNN or foundation model) and temporal attention layers to embed high-level anatomical features and long-range dependencies. A lightweight fusion module then combines these features to estimate the trajectory. Experimental results on a large public benchmark show that DualTrack achieves state-of-the-art accuracy and globally consistent 3D reconstructions, outperforming previous methods and yielding an average reconstruction error below 5 mm.
>
---
#### [replaced 063] Is this chart lying to me? Automating the detection of misleading visualizations
- **分类: cs.CL; cs.CV; cs.GR**

- **简介: 该论文属于检测误导性图表的任务，旨在解决虚假可视化传播 misinformation 的问题。工作包括构建真实和合成数据集，并评估多种模型的检测效果。**

- **链接: [https://arxiv.org/pdf/2508.21675](https://arxiv.org/pdf/2508.21675)**

> **作者:** Jonathan Tonglet; Jan Zimny; Tinne Tuytelaars; Iryna Gurevych
>
> **备注:** Camera-ready version accepted at ACL 2026 Main conference. Code and data available at: this https URL
>
> **摘要:** Misleading visualizations are a potent driver of misinformation on social media and the web. By violating chart design principles, they distort data and lead readers to draw inaccurate conclusions. Prior work has shown that both humans and multimodal large language models (MLLMs) are frequently deceived by such visualizations. Automatically detecting misleading visualizations and identifying the specific design rules they violate could help protect readers and reduce the spread of misinformation. However, the training and evaluation of AI models has been limited by the absence of large, diverse, and openly available datasets. In this work, we introduce Misviz, a benchmark of 2,604 real-world visualizations annotated with 12 types of misleaders. To support model training, we also create Misviz-synth, a synthetic dataset of 57,665 visualizations generated using Matplotlib and based on real-world data tables. We perform a comprehensive evaluation on both datasets using state-of-the-art MLLMs, rule-based systems, and image-axis classifiers. Our results reveal that the task remains highly challenging. We release Misviz, Misviz-synth, and the accompanying code.
>
---
#### [replaced 064] DyTact: Capturing Dynamic Contacts in Hand-Object Manipulation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.03103](https://arxiv.org/pdf/2506.03103)**

> **作者:** Xiaoyan Cong; Angela Xing; Chandradeep Pokhariya; Rao Fu; Srinath Sridhar
>
> **备注:** 3DV 2026 Oral, Webpage: this https URL
>
> **摘要:** Reconstructing dynamic hand-object contacts is essential for realistic manipulation in AI character animation, XR, and robotics, yet it remains challenging due to heavy occlusions, complex surface details, and limitations in existing capture techniques. In this paper, we introduce DyTact, a markerless capture method for accurately capturing dynamic contact in hand-object manipulations in a non-intrusive manner. Our approach leverages a dynamic, articulated representation based on 2D Gaussian surfels to model complex manipulations. By binding these surfels to MANO meshes, DyTact harnesses the inductive bias of template models to stabilize and accelerate optimization. A refinement module addresses time-dependent high-frequency deformations, while a contact-guided adaptive sampling strategy selectively increases surfel density in contact regions to handle heavy occlusion. Extensive experiments demonstrate that DyTact not only achieves state-of-the-art dynamic contact estimation accuracy but also significantly improves novel view synthesis quality, all while operating with fast optimization and efficient memory usage. Project Page: this https URL .
>
---
#### [replaced 065] DVP-MVS++: Synergize Depth-Normal-Edge and Harmonized Visibility Prior for Multi-View Stereo
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.13215](https://arxiv.org/pdf/2506.13215)**

> **作者:** Zhenlong Yuan; Dapeng Zhang; Zehao Li; Chengxuan Qian; Jianing Chen; Yinda Chen; Kehua Chen; Tianlu Mao; Zhaoxin Li; Hao Jiang; Zhaoqi Wang
>
> **摘要:** Recently, patch deformation-based methods have demonstrated significant effectiveness in multi-view stereo due to their incorporation of deformable and expandable perception for reconstructing textureless areas. However, these methods generally focus on identifying reliable pixel correlations to mitigate matching ambiguity of patch deformation, while neglecting the deformation instability caused by edge-skipping and visibility occlusions, which may cause potential estimation deviations. To address these issues, we propose DVP-MVS++, an innovative approach that synergizes both depth-normal-edge aligned and harmonized cross-view priors for robust and visibility-aware patch deformation. Specifically, to avoid edge-skipping, we first apply DepthPro, Metric3Dv2 and Roberts operator to generate coarse depth maps, normal maps and edge maps, respectively. These maps are then aligned via an erosion-dilation strategy to produce fine-grained homogeneous boundaries for facilitating robust patch deformation. Moreover, we reformulate view selection weights as visibility maps, and then implement both an enhanced cross-view depth reprojection and an area-maximization strategy to help reliably restore visible areas and effectively balance deformed patch, thus acquiring harmonized cross-view priors for visibility-aware patch deformation. Additionally, we obtain geometry consistency by adopting both aggregated normals via view selection and projection depth differences via epipolar lines, and then employ SHIQ for highlight correction to enable geometry consistency with highlight-aware perception, thus improving reconstruction quality during propagation and refinement stage. Evaluation results on ETH3D, Tanks & Temples and Strecha datasets exhibit the state-of-the-art performance and robust generalization capability of our proposed method.
>
---
#### [replaced 066] Differential privacy representation geometry for medical image analysis
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.01098](https://arxiv.org/pdf/2603.01098)**

> **作者:** Soroosh Tayebi Arasteh; Marziyeh Mohammadi; Sven Nebelung; Daniel Truhn
>
> **摘要:** Differential privacy (DP)'s effect in medical imaging is typically evaluated only through end-to-end performance, leaving the mechanism of privacy-induced utility loss unclear. We introduce Differential Privacy Representation Geometry for Medical Imaging (DP-RGMI), a framework that interprets DP as a structured transformation of representation space and decomposes performance degradation into encoder geometry and task-head utilization. Geometry is quantified by representation displacement from initialization and spectral effective dimension, while utilization is measured as the gap between linear-probe and end-to-end utility. Across over 594,000 images from four chest X-ray datasets and multiple pretrained initializations, we show that DP is consistently associated with a utilization gap even when linear separability is largely preserved. At the same time, displacement and spectral dimension exhibit non-monotonic, initialization- and dataset-dependent reshaping, indicating that DP alters representation anisotropy rather than uniformly collapsing features. Correlation analysis reveals that the association between end-to-end performance and utilization is robust across datasets but can vary by initialization, while geometric quantities capture additional prior- and dataset-conditioned variation. These findings position DP-RGMI as a reproducible framework for diagnosing privacy-induced failure modes and informing privacy model selection.
>
---
