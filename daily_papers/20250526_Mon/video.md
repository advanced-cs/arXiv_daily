# 计算机视觉 cs.CV

- **最新发布 173 篇**

- **更新 65 篇**

## 最新发布

#### [new 001] ICPL-ReID: Identity-Conditional Prompt Learning for Multi-Spectral Object Re-Identification
- **分类: cs.CV**

- **简介: 该论文属于多光谱目标重识别（ReID）任务，旨在解决异构光谱间模态差异导致的互补信息利用不足问题。提出ICPL框架，利用CLIP的跨模态对齐能力，通过身份条件文本提示学习统一光谱特征，并设计多光谱适配器缓解风格差异，实验证明优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.17821v1](http://arxiv.org/pdf/2505.17821v1)**

> **作者:** Shihao Li; Chenglong Li; Aihua Zheng; Jin Tang; Bin Luo
>
> **备注:** Accepted by IEEE Transactions on Multimedia (TMM)
>
> **摘要:** Multi-spectral object re-identification (ReID) brings a new perception perspective for smart city and intelligent transportation applications, effectively addressing challenges from complex illumination and adverse weather. However, complex modal differences between heterogeneous spectra pose challenges to efficiently utilizing complementary and discrepancy of spectra information. Most existing methods fuse spectral data through intricate modal interaction modules, lacking fine-grained semantic understanding of spectral information (\textit{e.g.}, text descriptions, part masks, and object keypoints). To solve this challenge, we propose a novel Identity-Conditional text Prompt Learning framework (ICPL), which exploits the powerful cross-modal alignment capability of CLIP, to unify different spectral visual features from text semantics. Specifically, we first propose the online prompt learning using learnable text prompt as the identity-level semantic center to bridge the identity semantics of different spectra in online manner. Then, in lack of concrete text descriptions, we propose the multi-spectral identity-condition module to use identity prototype as spectral identity condition to constraint prompt learning. Meanwhile, we construct the alignment loop mutually optimizing the learnable text prompt and spectral visual encoder to avoid online prompt learning disrupting the pre-trained text-image alignment distribution. In addition, to adapt to small-scale multi-spectral data and mitigate style differences between spectra, we propose multi-spectral adapter that employs a low-rank adaption method to learn spectra-specific features. Comprehensive experiments on 5 benchmarks, including RGBNT201, Market-MM, MSVR310, RGBN300, and RGBNT100, demonstrate that the proposed method outperforms the state-of-the-art methods.
>
---
#### [new 002] F-ANcGAN: An Attention-Enhanced Cycle Consistent Generative Adversarial Architecture for Synthetic Image Generation of Nanoparticles
- **分类: cs.CV; cond-mat.mtrl-sci; cs.LG; eess.IV**

- **简介: 该论文提出F-ANcGAN模型，解决纳米粒子图像数据不足导致分割模型训练困难的问题。通过结合注意力机制、Style U-Net生成器和CycleGAN架构，利用有限标注数据生成高质量SEM图像，FID分数达10.39，提升下游任务效果并拓展资源受限场景应用。**

- **链接: [http://arxiv.org/pdf/2505.18106v1](http://arxiv.org/pdf/2505.18106v1)**

> **作者:** Varun Ajith; Anindya Pal; Saumik Bhattacharya; Sayantari Ghosh
>
> **备注:** 11 pages, 9 figures, 2 tables, conference paper
>
> **摘要:** Nanomaterial research is becoming a vital area for energy, medicine, and materials science, and accurate analysis of the nanoparticle topology is essential to determine their properties. Unfortunately, the lack of high-quality annotated datasets drastically hinders the creation of strong segmentation models for nanoscale imaging. To alleviate this problem, we introduce F-ANcGAN, an attention-enhanced cycle consistent generative adversarial system that can be trained using a limited number of data samples and generates realistic scanning electron microscopy (SEM) images directly from segmentation maps. Our model uses a Style U-Net generator and a U-Net segmentation network equipped with self-attention to capture structural relationships and applies augmentation methods to increase the variety of the dataset. The architecture reached a raw FID score of 17.65 for TiO$_2$ dataset generation, with a further reduction in FID score to nearly 10.39 by using efficient post-processing techniques. By facilitating scalable high-fidelity synthetic dataset generation, our approach can improve the effectiveness of downstream segmentation task training, overcoming severe data shortage issues in nanoparticle analysis, thus extending its applications to resource-limited fields.
>
---
#### [new 003] 5G-DIL: Domain Incremental Learning with Similarity-Aware Sampling for Dynamic 5G Indoor Localization
- **分类: cs.CV; 62D05, 62J99, 62P12, 68T37; G.3; H.3.3; I.2.4; I.4; I.5.1**

- **简介: 该论文针对动态5G室内定位中环境变化导致模型性能下降的问题，提出5G-DIL方法。通过Chebyshev距离的相似性感知采样，选择先前环境的关键样本并仅训练新环境变化区域，实现高效适应。实验表明其以50样本快速适应，保持0.261米MAE的高精度。**

- **链接: [http://arxiv.org/pdf/2505.17684v1](http://arxiv.org/pdf/2505.17684v1)**

> **作者:** Nisha Lakshmana Raichur; Lucas Heublein; Christopher Mutschler; Felix Ott
>
> **备注:** 7 pages, 6 figures
>
> **摘要:** Indoor positioning based on 5G data has achieved high accuracy through the adoption of recent machine learning (ML) techniques. However, the performance of learning-based methods degrades significantly when environmental conditions change, thereby hindering their applicability to new scenarios. Acquiring new training data for each environmental change and fine-tuning ML models is both time-consuming and resource-intensive. This paper introduces a domain incremental learning (DIL) approach for dynamic 5G indoor localization, called 5G-DIL, enabling rapid adaptation to environmental changes. We present a novel similarity-aware sampling technique based on the Chebyshev distance, designed to efficiently select specific exemplars from the previous environment while training only on the modified regions of the new environment. This avoids the need to train on the entire region, significantly reducing the time and resources required for adaptation without compromising localization accuracy. This approach requires as few as 50 exemplars from adaptation domains, significantly reducing training time while maintaining high positioning accuracy in previous environments. Comparative evaluations against state-of-the-art DIL techniques on a challenging real-world indoor dataset demonstrate the effectiveness of the proposed sample selection method. Our approach is adaptable to real-world non-line-of-sight propagation scenarios and achieves an MAE positioning error of 0.261 meters, even under dynamic environmental conditions. Code: https://gitlab.cc-asp.fraunhofer.de/5g-pos/5g-dil
>
---
#### [new 004] Diagnosing Vision Language Models' Perception by Leveraging Human Methods for Color Vision Deficiencies
- **分类: cs.CV; cs.CL**

- **简介: 该论文评估视觉语言模型（LVLMs）处理色觉差异的能力，解决其能否模拟色觉缺陷者（CVD）感知的问题。通过Ishihara色盲测试发现，LVLMs可解释CVD但无法模拟其图像感知，强调需开发更具色彩感知包容性的多模态系统。**

- **链接: [http://arxiv.org/pdf/2505.17461v1](http://arxiv.org/pdf/2505.17461v1)**

> **作者:** Kazuki Hayashi; Shintaro Ozaki; Yusuke Sakai; Hidetaka Kamigaito; Taro Watanabe
>
> **摘要:** Large-scale Vision Language Models (LVLMs) are increasingly being applied to a wide range of real-world multimodal applications, involving complex visual and linguistic reasoning. As these models become more integrated into practical use, they are expected to handle complex aspects of human interaction. Among these, color perception is a fundamental yet highly variable aspect of visual understanding. It differs across individuals due to biological factors such as Color Vision Deficiencies (CVDs), as well as differences in culture and language. Despite its importance, perceptual diversity has received limited attention. In our study, we evaluate LVLMs' ability to account for individual level perceptual variation using the Ishihara Test, a widely used method for detecting CVDs. Our results show that LVLMs can explain CVDs in natural language, but they cannot simulate how people with CVDs perceive color in image based tasks. These findings highlight the need for multimodal systems that can account for color perceptual diversity and support broader discussions on perceptual inclusiveness and fairness in multimodal AI.
>
---
#### [new 005] EMRA-proxy: Enhancing Multi-Class Region Semantic Segmentation in Remote Sensing Images with Attention Proxy
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出RAPNet，针对高分辨率遥感图像多类语义分割任务。解决CNN局部特征局限与Transformer全局建模忽略细节及计算昂贵的问题，通过区域级上下文注意力（CRA）生成语义掩码，结合全局类优化（GCR）模块融合注意力图，提升分割精度，实验验证其优势。**

- **链接: [http://arxiv.org/pdf/2505.17665v1](http://arxiv.org/pdf/2505.17665v1)**

> **作者:** Yichun Yu; Yuqing Lan; Zhihuan Xing; Xiaoyi Yang; Tingyue Tang; Dan Yu
>
> **备注:** Proceedings of the 20th International Conference on Intelligent Computing (ICIC 2024): Poster Volume I. Tianjin, China, 2024: 538-562
>
> **摘要:** High-resolution remote sensing (HRRS) image segmentation is challenging due to complex spatial layouts and diverse object appearances. While CNNs excel at capturing local features, they struggle with long-range dependencies, whereas Transformers can model global context but often neglect local details and are computationally expensive.We propose a novel approach, Region-Aware Proxy Network (RAPNet), which consists of two components: Contextual Region Attention (CRA) and Global Class Refinement (GCR). Unlike traditional methods that rely on grid-based layouts, RAPNet operates at the region level for more flexible segmentation. The CRA module uses a Transformer to capture region-level contextual dependencies, generating a Semantic Region Mask (SRM). The GCR module learns a global class attention map to refine multi-class information, combining the SRM and attention map for accurate segmentation.Experiments on three public datasets show that RAPNet outperforms state-of-the-art methods, achieving superior multi-class segmentation accuracy.
>
---
#### [new 006] Do You Keep an Eye on What I Ask? Mitigating Multimodal Hallucination via Attention-Guided Ensemble Decoding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态处理任务，旨在解决大视觉语言模型（LVLMs）生成描述时的物体幻觉问题。提出Ensemble Decoding（ED）方法，通过分割图像为子区域并结合注意力权重融合预测结果，同时引入可信度约束和快速变体FastED，实验显示其效果达当前最优。**

- **链接: [http://arxiv.org/pdf/2505.17529v1](http://arxiv.org/pdf/2505.17529v1)**

> **作者:** Yeongjae Cho; Keonwoo Kim; Taebaek Hwang; Sungzoon Cho
>
> **摘要:** Recent advancements in Large Vision-Language Models (LVLMs) have significantly expanded their utility in tasks like image captioning and visual question answering. However, they still struggle with object hallucination, where models generate descriptions that inaccurately reflect the visual content by including nonexistent objects or misrepresenting existing ones. While previous methods, such as data augmentation and training-free approaches, strive to tackle this issue, they still encounter scalability challenges and often depend on additional external modules. In this work, we propose Ensemble Decoding (ED), a novel strategy that splits the input image into sub-images and combines logit distributions by assigning weights through the attention map. Furthermore, we introduce ED adaptive plausibility constraint to calibrate logit distribution and FastED, a variant designed for speed-critical applications. Extensive experiments across hallucination benchmarks demonstrate that our proposed method achieves state-of-the-art performance, validating the effectiveness of our approach.
>
---
#### [new 007] Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出Deep Video Discovery(DVD)代理，针对长视频理解中LLMs处理信息密集视频的局限，采用自主代理搜索策略，通过多粒度视频数据库工具自主规划、选择参数并迭代优化推理，显著提升LVBench等数据集表现。**

- **链接: [http://arxiv.org/pdf/2505.18079v1](http://arxiv.org/pdf/2505.18079v1)**

> **作者:** Xiaoyi Zhang; Zhaoyang Jia; Zongyu Guo; Jiahao Li; Bin Li; Houqiang Li; Yan Lu
>
> **备注:** Under review
>
> **摘要:** Long-form video understanding presents significant challenges due to extensive temporal-spatial complexity and the difficulty of question answering under such extended contexts. While Large Language Models (LLMs) have demonstrated considerable advancements in video analysis capabilities and long context handling, they continue to exhibit limitations when processing information-dense hour-long videos. To overcome such limitations, we propose the Deep Video Discovery agent to leverage an agentic search strategy over segmented video clips. Different from previous video agents manually designing a rigid workflow, our approach emphasizes the autonomous nature of agents. By providing a set of search-centric tools on multi-granular video database, our DVD agent leverages the advanced reasoning capability of LLM to plan on its current observation state, strategically selects tools, formulates appropriate parameters for actions, and iteratively refines its internal reasoning in light of the gathered information. We perform comprehensive evaluation on multiple long video understanding benchmarks that demonstrates the advantage of the entire system design. Our DVD agent achieves SOTA performance, significantly surpassing prior works by a large margin on the challenging LVBench dataset. Comprehensive ablation studies and in-depth tool analyses are also provided, yielding insights to further advance intelligent agents tailored for long-form video understanding tasks. The code will be released later.
>
---
#### [new 008] Mitigate One, Skew Another? Tackling Intersectional Biases in Text-to-Image Models
- **分类: cs.CV**

- **简介: 该论文属于文本到图像（TTI）模型公平性研究，旨在解决多维度偏见间的相互影响问题。现有方法单独处理偏见时可能加剧其他维度的不平等。作者提出BiasConnect工具量化偏见交互，并开发InterMit算法，通过用户定义优先级和分布实现交叉偏见协同缓解，效果更优且步骤更少。**

- **链接: [http://arxiv.org/pdf/2505.17280v1](http://arxiv.org/pdf/2505.17280v1)**

> **作者:** Pushkar Shukla; Aditya Chinchure; Emily Diana; Alexander Tolbert; Kartik Hosanagar; Vineeth N Balasubramanian; Leonid Sigal; Matthew Turk
>
> **摘要:** The biases exhibited by text-to-image (TTI) models are often treated as independent, though in reality, they may be deeply interrelated. Addressing bias along one dimension - such as ethnicity or age - can inadvertently affect another, like gender, either mitigating or exacerbating existing disparities. Understanding these interdependencies is crucial for designing fairer generative models, yet measuring such effects quantitatively remains a challenge. To address this, we introduce BiasConnect, a novel tool for analyzing and quantifying bias interactions in TTI models. BiasConnect uses counterfactual interventions along different bias axes to reveal the underlying structure of these interactions and estimates the effect of mitigating one bias axis on another. These estimates show strong correlation (+0.65) with observed post-mitigation outcomes. Building on BiasConnect, we propose InterMit, an intersectional bias mitigation algorithm guided by user-defined target distributions and priority weights. InterMit achieves lower bias (0.33 vs. 0.52) with fewer mitigation steps (2.38 vs. 3.15 average steps), and yields superior image quality compared to traditional techniques. Although our implementation is training-free, InterMit is modular and can be integrated with many existing debiasing approaches for TTI models, making it a flexible and extensible solution.
>
---
#### [new 009] Harnessing EHRs for Diffusion-based Anomaly Detection on Chest X-rays
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于无监督异常检测任务，旨在解决现有扩散模型无法有效区分胸部X光中的正常解剖变异与病理异常的问题。提出多模态框架Diff3M，通过图像-EHR交叉注意力模块融合X光与结构化电子健康记录，并采用静态掩码策略优化正常图像重建，提升异常检测性能。**

- **链接: [http://arxiv.org/pdf/2505.17311v1](http://arxiv.org/pdf/2505.17311v1)**

> **作者:** Harim Kim; Yuhan Wang; Minkyu Ahn; Heeyoul Choi; Yuyin Zhou; Charmgil Hong
>
> **备注:** MICCAI 2025 early accept
>
> **摘要:** Unsupervised anomaly detection (UAD) in medical imaging is crucial for identifying pathological abnormalities without requiring extensive labeled data. However, existing diffusion-based UAD models rely solely on imaging features, limiting their ability to distinguish between normal anatomical variations and pathological anomalies. To address this, we propose Diff3M, a multi-modal diffusion-based framework that integrates chest X-rays and structured Electronic Health Records (EHRs) for enhanced anomaly detection. Specifically, we introduce a novel image-EHR cross-attention module to incorporate structured clinical context into the image generation process, improving the model's ability to differentiate normal from abnormal features. Additionally, we develop a static masking strategy to enhance the reconstruction of normal-like images from anomalies. Extensive evaluations on CheXpert and MIMIC-CXR/IV demonstrate that Diff3M achieves state-of-the-art performance, outperforming existing UAD methods in medical imaging. Our code is available at this http URL https://github.com/nth221/Diff3M
>
---
#### [new 010] To Glue or Not to Glue? Classical vs Learned Image Matching for Mobile Mapping Cameras to Textured Semantic 3D Building Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文对比经典与深度学习特征匹配方法在移动测绘相机与语义3D建筑模型匹配中的表现，解决视觉定位精度与鲁棒性问题。通过标准数据集及自建 facade-图像数据，评估不同方法的PnP定位精度，结果表明学习方法显著优于传统方法。**

- **链接: [http://arxiv.org/pdf/2505.17973v1](http://arxiv.org/pdf/2505.17973v1)**

> **作者:** Simone Gaisbauer; Prabin Gyawali; Qilin Zhang; Olaf Wysocki; Boris Jutzi
>
> **备注:** Accepted to MMT, Xiamen, China; ISPRS Annals
>
> **摘要:** Feature matching is a necessary step for many computer vision and photogrammetry applications such as image registration, structure-from-motion, and visual localization. Classical handcrafted methods such as SIFT feature detection and description combined with nearest neighbour matching and RANSAC outlier removal have been state-of-the-art for mobile mapping cameras. With recent advances in deep learning, learnable methods have been introduced and proven to have better robustness and performance under complex conditions. Despite their growing adoption, a comprehensive comparison between classical and learnable feature matching methods for the specific task of semantic 3D building camera-to-model matching is still missing. This submission systematically evaluates the effectiveness of different feature-matching techniques in visual localization using textured CityGML LoD2 models. We use standard benchmark datasets (HPatches, MegaDepth-1500) and custom datasets consisting of facade textures and corresponding camera images (terrestrial and drone). For the latter, we evaluate the achievable accuracy of the absolute pose estimated using a Perspective-n-Point (PnP) algorithm, with geometric ground truth derived from geo-referenced trajectory data. The results indicate that the learnable feature matching methods vastly outperform traditional approaches regarding accuracy and robustness on our challenging custom datasets with zero to 12 RANSAC-inliers and zero to 0.16 area under the curve. We believe that this work will foster the development of model-based visual localization methods. Link to the code: https://github.com/simBauer/To\_Glue\_or\_not\_to\_Glue
>
---
#### [new 011] Optimizing Image Capture for Computer Vision-Powered Taxonomic Identification and Trait Recognition of Biodiversity Specimens
- **分类: cs.CV; q-bio.QM**

- **简介: 该论文提出优化生物标本图像采集方法，以支持计算机视觉在物种识别与性状分析中的应用。针对传统成像侧重人眼需求、忽视自动化分析的问题，通过多学科合作，提出10项关键建议（如标准化摆放、统一背景、光照控制等），构建整合框架，提升图像数据质量以推动大规模生态与进化研究。**

- **链接: [http://arxiv.org/pdf/2505.17317v1](http://arxiv.org/pdf/2505.17317v1)**

> **作者:** Alyson East; Elizabeth G. Campolongo; Luke Meyers; S M Rayeed; Samuel Stevens; Iuliia Zarubiieva; Isadora E. Fluck; Jennifer C. Girón; Maximiliane Jousse; Scott Lowe; Kayla I Perry; Isabelle Betancourt; Noah Charney; Evan Donoso; Nathan Fox; Kim J. Landsbergen; Ekaterina Nepovinnykh; Michelle Ramirez; Parkash Singh; Khum Thapa-Magar; Matthew Thompson; Evan Waite; Tanya Berger-Wolf; Hilmar Lapp; Paula Mabee; Graham Taylor; Sydne Record
>
> **摘要:** Biological collections house millions of specimens documenting Earth's biodiversity, with digital images increasingly available through open-access platforms. Most imaging protocols were developed for human visual interpretation without considering computational analysis requirements. This paper aims to bridge the gap between current imaging practices and the potential for automated analysis by presenting key considerations for creating biological specimen images optimized for computer vision applications. We provide conceptual computer vision topics for context, addressing fundamental concerns including model generalization, data leakage, and comprehensive metadata documentation, and outline practical guidance on specimen imagine, and data storage. These recommendations were synthesized through interdisciplinary collaboration between taxonomists, collection managers, ecologists, and computer scientists. Through this synthesis, we have identified ten interconnected considerations that form a framework for successfully integrating biological specimen images into computer vision pipelines. The key elements include: (1) comprehensive metadata documentation, (2) standardized specimen positioning, (3) consistent size and color calibration, (4) protocols for handling multiple specimens in one image, (5) uniform background selection, (6) controlled lighting, (7) appropriate resolution and magnification, (8) optimal file formats, (9) robust data archiving strategies, and (10) accessible data sharing practices. By implementing these recommendations, collection managers, taxonomists, and biodiversity informaticians can generate images that support automated trait extraction, species identification, and novel ecological and evolutionary analyses at unprecedented scales. Successful implementation lies in thorough documentation of methodological choices.
>
---
#### [new 012] Mind the Domain Gap: Measuring the Domain Gap Between Real-World and Synthetic Point Clouds for Automated Driving Development
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于领域差距测量任务，旨在解决真实与合成点云间差异分析不足的问题。提出DoGSS-PCL指标，评估同一场景下真实与模拟点云的几何及语义差距，验证合成数据在50%混合训练中的有效性，推动自动驾驶仿真数据可信度研究。**

- **链接: [http://arxiv.org/pdf/2505.17959v1](http://arxiv.org/pdf/2505.17959v1)**

> **作者:** Nguyen Duc; Yan-Ling Lai; Patrick Madlindl; Xinyuan Zhu; Benedikt Schwab; Olaf Wysocki; Ludwig Hoegner; Thomas H. Kolbe
>
> **备注:** Submitted to PFG Journal of Photogrammetry, Remote Sensing and Geoinformation Science
>
> **摘要:** Owing to the typical long-tail data distribution issues, simulating domain-gap-free synthetic data is crucial in robotics, photogrammetry, and computer vision research. The fundamental challenge pertains to credibly measuring the difference between real and simulated data. Such a measure is vital for safety-critical applications, such as automated driving, where out-of-domain samples may impact a car's perception and cause fatal accidents. Previous work has commonly focused on simulating data on one scene and analyzing performance on a different, real-world scene, hampering the disjoint analysis of domain gap coming from networks' deficiencies, class definitions, and object representation. In this paper, we propose a novel approach to measuring the domain gap between the real world sensor observations and simulated data representing the same location, enabling comprehensive domain gap analysis. To measure such a domain gap, we introduce a novel metric DoGSS-PCL and evaluation assessing the geometric and semantic quality of the simulated point cloud. Our experiments corroborate that the introduced approach can be used to measure the domain gap. The tests also reveal that synthetic semantic point clouds may be used for training deep neural networks, maintaining the performance at the 50/50 real-to-synthetic ratio. We strongly believe that this work will facilitate research on credible data simulation and allow for at-scale deployment in automated driving testing and digital twinning.
>
---
#### [new 013] Hyperspectral Anomaly Detection Fused Unified Nonconvex Tensor Ring Factors Regularization
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出HAD-EUNTRFR方法，针对高光谱异常检测中背景成分全局相关性和局部平滑性利用不足的问题，通过融合非凸张量环分解与梯度因子正则化，同时捕捉背景的空间-光谱关联并约束其低秩性与平滑性，结合异常组稀疏正则化，采用ADMM优化，提升检测精度。**

- **链接: [http://arxiv.org/pdf/2505.17881v1](http://arxiv.org/pdf/2505.17881v1)**

> **作者:** Wenjin Qin; Hailin Wang; Hao Shu; Feng Zhang; Jianjun Wang; Xiangyong Cao; Xi-Le Zhao; Gemine Vivone
>
> **摘要:** In recent years, tensor decomposition-based approaches for hyperspectral anomaly detection (HAD) have gained significant attention in the field of remote sensing. However, existing methods often fail to fully leverage both the global correlations and local smoothness of the background components in hyperspectral images (HSIs), which exist in both the spectral and spatial domains. This limitation results in suboptimal detection performance. To mitigate this critical issue, we put forward a novel HAD method named HAD-EUNTRFR, which incorporates an enhanced unified nonconvex tensor ring (TR) factors regularization. In the HAD-EUNTRFR framework, the raw HSIs are first decomposed into background and anomaly components. The TR decomposition is then employed to capture the spatial-spectral correlations within the background component. Additionally, we introduce a unified and efficient nonconvex regularizer, induced by tensor singular value decomposition (TSVD), to simultaneously encode the low-rankness and sparsity of the 3-D gradient TR factors into a unique concise form. The above characterization scheme enables the interpretable gradient TR factors to inherit the low-rankness and smoothness of the original background. To further enhance anomaly detection, we design a generalized nonconvex regularization term to exploit the group sparsity of the anomaly component. To solve the resulting doubly nonconvex model, we develop a highly efficient optimization algorithm based on the alternating direction method of multipliers (ADMM) framework. Experimental results on several benchmark datasets demonstrate that our proposed method outperforms existing state-of-the-art (SOTA) approaches in terms of detection accuracy.
>
---
#### [new 014] CGS-GAN: 3D Consistent Gaussian Splatting GANs for High Resolution Human Head Synthesis
- **分类: cs.CV**

- **简介: 该论文提出CGS-GAN，解决3D GAN合成高分辨率人头时视角变化导致的3D不一致及训练不稳定问题。采用多视角正则化与优化生成器架构，无需视角条件实现稳定训练和高质量渲染（2048²），并构建新数据集提升一致性，效果显著。**

- **链接: [http://arxiv.org/pdf/2505.17590v1](http://arxiv.org/pdf/2505.17590v1)**

> **作者:** Florian Barthel; Wieland Morgenstern; Paul Hinzer; Anna Hilsmann; Peter Eisert
>
> **备注:** Main paper 12 pages, supplementary materials 8 pages
>
> **摘要:** Recently, 3D GANs based on 3D Gaussian splatting have been proposed for high quality synthesis of human heads. However, existing methods stabilize training and enhance rendering quality from steep viewpoints by conditioning the random latent vector on the current camera position. This compromises 3D consistency, as we observe significant identity changes when re-synthesizing the 3D head with each camera shift. Conversely, fixing the camera to a single viewpoint yields high-quality renderings for that perspective but results in poor performance for novel views. Removing view-conditioning typically destabilizes GAN training, often causing the training to collapse. In response to these challenges, we introduce CGS-GAN, a novel 3D Gaussian Splatting GAN framework that enables stable training and high-quality 3D-consistent synthesis of human heads without relying on view-conditioning. To ensure training stability, we introduce a multi-view regularization technique that enhances generator convergence with minimal computational overhead. Additionally, we adapt the conditional loss used in existing 3D Gaussian splatting GANs and propose a generator architecture designed to not only stabilize training but also facilitate efficient rendering and straightforward scaling, enabling output resolutions up to $2048^2$. To evaluate the capabilities of CGS-GAN, we curate a new dataset derived from FFHQ. This dataset enables very high resolutions, focuses on larger portions of the human head, reduces view-dependent artifacts for improved 3D consistency, and excludes images where subjects are obscured by hands or other objects. As a result, our approach achieves very high rendering quality, supported by competitive FID scores, while ensuring consistent 3D scene generation. Check our our project page here: https://fraunhoferhhi.github.io/cgs-gan/
>
---
#### [new 015] PawPrint: Whose Footprints Are These? Identifying Animal Individuals by Their Footprints
- **分类: cs.CV**

- **简介: 该论文提出通过动物足迹识别个体，解决传统GPS标签易丢失、依赖他人报告的问题。构建首个公开数据集PawPrint/PawPrint+，对比深度学习与传统特征方法，提出结合全局与局部特征提升识别可靠性，用于宠物管理和 wildlife conservation。**

- **链接: [http://arxiv.org/pdf/2505.17445v1](http://arxiv.org/pdf/2505.17445v1)**

> **作者:** Inpyo Song; Hyemin Hwang; Jangwon Lee
>
> **备注:** Accepted to ICIP 2025
>
> **摘要:** In the United States, as of 2023, pet ownership has reached 66% of households and continues to rise annually. This trend underscores the critical need for effective pet identification and monitoring methods, particularly as nearly 10 million cats and dogs are reported stolen or lost each year. However, traditional methods for finding lost animals like GPS tags or ID photos have limitations-they can be removed, face signal issues, and depend on someone finding and reporting the pet. To address these limitations, we introduce PawPrint and PawPrint+, the first publicly available datasets focused on individual-level footprint identification for dogs and cats. Through comprehensive benchmarking of both modern deep neural networks (e.g., CNN, Transformers) and classical local features, we observe varying advantages and drawbacks depending on substrate complexity and data availability. These insights suggest future directions for combining learned global representations with local descriptors to enhance reliability across diverse, real-world conditions. As this approach provides a non-invasive alternative to traditional ID tags, we anticipate promising applications in ethical pet management and wildlife conservation efforts.
>
---
#### [new 016] SHARDeg: A Benchmark for Skeletal Human Action Recognition in Degraded Scenarios
- **分类: cs.CV**

- **简介: 该论文属于骨骼人体动作识别（SHAR）任务，针对现有模型在退化数据（如低帧率、噪声等）中鲁棒性不足的问题，构建首个退化基准SHARDeg，基于NTU-120数据集评估五种模型在三种退化场景下的表现，发现退化类型显著影响精度（差异超40%），提出插值法提升性能，并发现LogSigRNN在低帧率下优于SOTA模型。**

- **链接: [http://arxiv.org/pdf/2505.18048v1](http://arxiv.org/pdf/2505.18048v1)**

> **作者:** Simon Malzard; Nitish Mital; Richard Walters; Victoria Nockles; Raghuveer Rao; Celso M. De Melo
>
> **备注:** 19 pages, 2 images
>
> **摘要:** Computer vision (CV) models for detection, prediction or classification tasks operate on video data-streams that are often degraded in the real world, due to deployment in real-time or on resource-constrained hardware. It is therefore critical that these models are robust to degraded data, but state of the art (SoTA) models are often insufficiently assessed with these real-world constraints in mind. This is exemplified by Skeletal Human Action Recognition (SHAR), which is critical in many CV pipelines operating in real-time and at the edge, but robustness to degraded data has previously only been shallowly and inconsistently assessed. Here we address this issue for SHAR by providing an important first data degradation benchmark on the most detailed and largest 3D open dataset, NTU-RGB+D-120, and assess the robustness of five leading SHAR models to three forms of degradation that represent real-world issues. We demonstrate the need for this benchmark by showing that the form of degradation, which has not previously been considered, has a large impact on model accuracy; at the same effective frame rate, model accuracy can vary by >40% depending on degradation type. We also identify that temporal regularity of frames in degraded SHAR data is likely a major driver of differences in model performance, and harness this to improve performance of existing models by up to >40%, through employing a simple mitigation approach based on interpolation. Finally, we highlight how our benchmark has helped identify an important degradation-resistant SHAR model based in Rough Path Theory; the LogSigRNN SHAR model outperforms the SoTA DeGCN model in five out of six cases at low frame rates by an average accuracy of 6%, despite trailing the SoTA model by 11-12% on un-degraded data at high frame rates (30 FPS).
>
---
#### [new 017] RestoreVAR: Visual Autoregressive Generation for All-in-One Image Restoration
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出RestoreVAR，用于全功能图像修复（AiOR）任务。针对现有潜扩散模型（LDM）推理速度慢的问题，其采用视觉自回归（VAR）架构，设计交叉注意力机制与潜空间优化模块，实现10倍加速且性能更优，达生成类AiOR方法SOTA，兼具泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.18047v1](http://arxiv.org/pdf/2505.18047v1)**

> **作者:** Sudarshan Rajagopalan; Kartik Narayan; Vishal M. Patel
>
> **备注:** Project page: https://sudraj2002.github.io/restorevarpage/
>
> **摘要:** The use of latent diffusion models (LDMs) such as Stable Diffusion has significantly improved the perceptual quality of All-in-One image Restoration (AiOR) methods, while also enhancing their generalization capabilities. However, these LDM-based frameworks suffer from slow inference due to their iterative denoising process, rendering them impractical for time-sensitive applications. To address this, we propose RestoreVAR, a novel generative approach for AiOR that significantly outperforms LDM-based models in restoration performance while achieving over $\mathbf{10\times}$ faster inference. RestoreVAR leverages visual autoregressive modeling (VAR), a recently introduced approach which performs scale-space autoregression for image generation. VAR achieves comparable performance to that of state-of-the-art diffusion transformers with drastically reduced computational costs. To optimally exploit these advantages of VAR for AiOR, we propose architectural modifications and improvements, including intricately designed cross-attention mechanisms and a latent-space refinement module, tailored for the AiOR task. Extensive experiments show that RestoreVAR achieves state-of-the-art performance among generative AiOR methods, while also exhibiting strong generalization capabilities.
>
---
#### [new 018] ViP$^2$-CLIP: Visual-Perception Prompting with Unified Alignment for Zero-Shot Anomaly Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于零样本异常检测（ZSAD）任务，旨在解决现有CLIP基模型依赖人工提示、语义覆盖不足及对类名敏感的问题。提出ViP²-CLIP，通过融合全局与多尺度局部视觉信息的ViP-Prompt机制，自适应生成细粒度文本提示，消除手动模板和类名依赖，提升异常区域定位精度，在工业/医疗数据中实现最优性能。**

- **链接: [http://arxiv.org/pdf/2505.17692v1](http://arxiv.org/pdf/2505.17692v1)**

> **作者:** Ziteng Yang; Jingzehua Xu; Yanshu Li; Zepeng Li; Yeqiang Wang; Xinghui Li
>
> **摘要:** Zero-shot anomaly detection (ZSAD) aims to detect anomalies without any target domain training samples, relying solely on external auxiliary data. Existing CLIP-based methods attempt to activate the model's ZSAD potential via handcrafted or static learnable prompts. The former incur high engineering costs and limited semantic coverage, whereas the latter apply identical descriptions across diverse anomaly types, thus fail to adapt to complex variations. Furthermore, since CLIP is originally pretrained on large-scale classification tasks, its anomaly segmentation quality is highly sensitive to the exact wording of class names, severely constraining prompting strategies that depend on class labels. To address these challenges, we introduce ViP$^{2}$-CLIP. The key insight of ViP$^{2}$-CLIP is a Visual-Perception Prompting (ViP-Prompt) mechanism, which fuses global and multi-scale local visual context to adaptively generate fine-grained textual prompts, eliminating manual templates and class-name priors. This design enables our model to focus on precise abnormal regions, making it particularly valuable when category labels are ambiguous or privacy-constrained. Extensive experiments on 15 industrial and medical benchmarks demonstrate that ViP$^{2}$-CLIP achieves state-of-the-art performance and robust cross-domain generalization.
>
---
#### [new 019] ExpertGen: Training-Free Expert Guidance for Controllable Text-to-Face Generation
- **分类: cs.CV**

- **简介: 论文属于可控文本到人脸生成任务，解决现有方法需额外训练模块导致的灵活性与资源问题。提出ExpertGen框架，利用预训练专家模型（如人脸识别、属性识别）及潜一致性模型，在扩散过程中无训练引导生成，实现多特征协同控制，提升精准度与效率。**

- **链接: [http://arxiv.org/pdf/2505.17256v1](http://arxiv.org/pdf/2505.17256v1)**

> **作者:** Liang Shi; Yun Fu
>
> **摘要:** Recent advances in diffusion models have significantly improved text-to-face generation, but achieving fine-grained control over facial features remains a challenge. Existing methods often require training additional modules to handle specific controls such as identity, attributes, or age, making them inflexible and resource-intensive. We propose ExpertGen, a training-free framework that leverages pre-trained expert models such as face recognition, facial attribute recognition, and age estimation networks to guide generation with fine control. Our approach uses a latent consistency model to ensure realistic and in-distribution predictions at each diffusion step, enabling accurate guidance signals to effectively steer the diffusion process. We show qualitatively and quantitatively that expert models can guide the generation process with high precision, and multiple experts can collaborate to enable simultaneous control over diverse facial aspects. By allowing direct integration of off-the-shelf expert models, our method transforms any such model into a plug-and-play component for controllable face generation.
>
---
#### [new 020] Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling
- **分类: cs.CV**

- **简介: 该论文提出HiVE-MIL框架，解决巨像素图像的少样本弱监督分类任务。针对现有方法在跨尺度模态交互和同尺度视觉-文本对齐上的不足，构建层次图结构整合多尺度视觉/文本节点，设计动态过滤机制与层次对比损失，提升病理图像分类效果，在癌症数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.17982v1](http://arxiv.org/pdf/2505.17982v1)**

> **作者:** Bryan Wong; Jong Woo Kim; Huazhu Fu; Mun Yong Yi
>
> **摘要:** Vision-language models (VLMs) have recently been integrated into multiple instance learning (MIL) frameworks to address the challenge of few-shot, weakly supervised classification of whole slide images (WSIs). A key trend involves leveraging multi-scale information to better represent hierarchical tissue structures. However, existing methods often face two key limitations: (1) insufficient modeling of interactions within the same modalities across scales (e.g., 5x and 20x) and (2) inadequate alignment between visual and textual modalities on the same scale. To address these gaps, we propose HiVE-MIL, a hierarchical vision-language framework that constructs a unified graph consisting of (1) parent-child links between coarse (5x) and fine (20x) visual/textual nodes to capture hierarchical relationships, and (2) heterogeneous intra-scale edges linking visual and textual nodes on the same scale. To further enhance semantic consistency, HiVE-MIL incorporates a two-stage, text-guided dynamic filtering mechanism that removes weakly correlated patch-text pairs, and introduces a hierarchical contrastive loss to align textual semantics across scales. Extensive experiments on TCGA breast, lung, and kidney cancer datasets demonstrate that HiVE-MIL consistently outperforms both traditional MIL and recent VLM-based MIL approaches, achieving gains of up to 4.1% in macro F1 under 16-shot settings. Our results demonstrate the value of jointly modeling hierarchical structure and multimodal alignment for efficient and scalable learning from limited pathology data. The code is available at https://github.com/bryanwong17/HiVE-MIL
>
---
#### [new 021] Temporal Consistency Constrained Transferable Adversarial Attacks with Background Mixup for Action Recognition
- **分类: cs.CV**

- **简介: 该论文提出BMTC方法，针对动作/图像识别模型提升对抗攻击的迁移性。解决现有方法依赖源-目标模型相似性及梯度震荡问题，通过背景混合模块（强化学习选优背景帧）降低模型依赖，并设计时序梯度一致性损失稳定攻击方向，增强跨模型攻击效果。**

- **链接: [http://arxiv.org/pdf/2505.17807v1](http://arxiv.org/pdf/2505.17807v1)**

> **作者:** Ping Li; Jianan Ni; Bo Pang
>
> **备注:** Accepted in IJCAI'25
>
> **摘要:** Action recognition models using deep learning are vulnerable to adversarial examples, which are transferable across other models trained on the same data modality. Existing transferable attack methods face two major challenges: 1) they heavily rely on the assumption that the decision boundaries of the surrogate (a.k.a., source) model and the target model are similar, which limits the adversarial transferability; and 2) their decision boundary difference makes the attack direction uncertain, which may result in the gradient oscillation, weakening the adversarial attack. This motivates us to propose a Background Mixup-induced Temporal Consistency (BMTC) attack method for action recognition. From the input transformation perspective, we design a model-agnostic background adversarial mixup module to reduce the surrogate-target model dependency. In particular, we randomly sample one video from each category and make its background frame, while selecting the background frame with the top attack ability for mixup with the clean frame by reinforcement learning. Moreover, to ensure an explicit attack direction, we leverage the background category as guidance for updating the gradient of adversarial example, and design a temporal gradient consistency loss, which strengthens the stability of the attack direction on subsequent frames. Empirical studies on two video datasets, i.e., UCF101 and Kinetics-400, and one image dataset, i.e., ImageNet, demonstrate that our method significantly boosts the transferability of adversarial examples across several action/image recognition models. Our code is available at https://github.com/mlvccn/BMTC_TransferAttackVid.
>
---
#### [new 022] EmoSign: A Multimodal Dataset for Understanding Emotions in American Sign Language
- **分类: cs.CV**

- **简介: 该论文属于多模态情感识别任务，旨在解决手语中情感表达研究不足的问题。通过构建首个含200条ASL视频的情感标注数据集EmoSign（含专业聋人签译者标注的开放式情感描述及基线模型），填补手语情感研究空白，建立新基准。**

- **链接: [http://arxiv.org/pdf/2505.17090v1](http://arxiv.org/pdf/2505.17090v1)**

> **作者:** Phoebe Chua; Cathy Mengying Fang; Takehiko Ohkawa; Raja Kushalnagar; Suranga Nanayakkara; Pattie Maes
>
> **摘要:** Unlike spoken languages where the use of prosodic features to convey emotion is well studied, indicators of emotion in sign language remain poorly understood, creating communication barriers in critical settings. Sign languages present unique challenges as facial expressions and hand movements simultaneously serve both grammatical and emotional functions. To address this gap, we introduce EmoSign, the first sign video dataset containing sentiment and emotion labels for 200 American Sign Language (ASL) videos. We also collect open-ended descriptions of emotion cues. Annotations were done by 3 Deaf ASL signers with professional interpretation experience. Alongside the annotations, we include baseline models for sentiment and emotion classification. This dataset not only addresses a critical gap in existing sign language research but also establishes a new benchmark for understanding model capabilities in multimodal emotion recognition for sign languages. The dataset is made available at https://huggingface.co/datasets/catfang/emosign.
>
---
#### [new 023] Reflectance Prediction-based Knowledge Distillation for Robust 3D Object Detection in Compressed Point Clouds
- **分类: cs.CV**

- **简介: 该论文针对智能交通系统中压缩点云传输导致的带宽压力与检测精度下降问题，提出基于反射率预测与知识蒸馏的3D检测框架RPKD。通过仅压缩点坐标、丢弃反射率降低传输负担，利用几何预测模块重建反射率，并结合教师模型的知识蒸馏提升压缩点云检测鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.17442v1](http://arxiv.org/pdf/2505.17442v1)**

> **作者:** Hao Jing; Anhong Wang; Yifan Zhang; Donghan Bu; Junhui Hou
>
> **摘要:** Regarding intelligent transportation systems for vehicle networking, low-bitrate transmission via lossy point cloud compression is vital for facilitating real-time collaborative perception among vehicles with restricted bandwidth. In existing compression transmission systems, the sender lossily compresses point coordinates and reflectance to generate a transmission code stream, which faces transmission burdens from reflectance encoding and limited detection robustness due to information loss. To address these issues, this paper proposes a 3D object detection framework with reflectance prediction-based knowledge distillation (RPKD). We compress point coordinates while discarding reflectance during low-bitrate transmission, and feed the decoded non-reflectance compressed point clouds into a student detector. The discarded reflectance is then reconstructed by a geometry-based reflectance prediction (RP) module within the student detector for precise detection. A teacher detector with the same structure as student detector is designed for performing reflectance knowledge distillation (RKD) and detection knowledge distillation (DKD) from raw to compressed point clouds. Our RPKD framework jointly trains detectors on both raw and compressed point clouds to improve the student detector's robustness. Experimental results on the KITTI dataset and Waymo Open Dataset demonstrate that our method can boost detection accuracy for compressed point clouds across multiple code rates. Notably, at a low code rate of 2.146 Bpp on the KITTI dataset, our RPKD-PV achieves the highest mAP of 73.6, outperforming existing detection methods with the PV-RCNN baseline.
>
---
#### [new 024] Semantic segmentation with reward
- **分类: cs.CV**

- **简介: 该论文提出基于奖励的强化学习方法RSS，用于语义分割任务。旨在解决缺乏像素级标注时的模型训练问题，通过引入分层奖励（像素级和图像级）及PSR、PSD技术，确保网络收敛。实验显示其在图像级奖励下优于现有弱监督方法。**

- **链接: [http://arxiv.org/pdf/2505.17905v1](http://arxiv.org/pdf/2505.17905v1)**

> **作者:** Xie Ting; Ye Huang; Zhilin Liu; Lixin Duan
>
> **备注:** Tech report
>
> **摘要:** In real-world scenarios, pixel-level labeling is not always available. Sometimes, we need a semantic segmentation network, and even a visual encoder can have a high compatibility, and can be trained using various types of feedback beyond traditional labels, such as feedback that indicates the quality of the parsing results. To tackle this issue, we proposed RSS (Reward in Semantic Segmentation), the first practical application of reward-based reinforcement learning on pure semantic segmentation offered in two granular levels (pixel-level and image-level). RSS incorporates various novel technologies, such as progressive scale rewards (PSR) and pair-wise spatial difference (PSD), to ensure that the reward facilitates the convergence of the semantic segmentation network, especially under image-level rewards. Experiments and visualizations on benchmark datasets demonstrate that the proposed RSS can successfully ensure the convergence of the semantic segmentation network on two levels of rewards. Additionally, the RSS, which utilizes an image-level reward, outperforms existing weakly supervised methods that also rely solely on image-level signals during training.
>
---
#### [new 025] A Framework for Multi-View Multiple Object Tracking using Single-View Multi-Object Trackers on Fish Data
- **分类: cs.CV**

- **简介: 该论文属于多视角多目标跟踪任务，针对水下鱼类因复杂3D运动和数据噪声导致传统单视角模型精度不足的问题，提出基于FairMOT和YOLOv8的多视角框架，通过立体视频输入与匹配技术提升追踪准确率（47%），并生成3D输出以增强鱼群行为分析。**

- **链接: [http://arxiv.org/pdf/2505.17201v1](http://arxiv.org/pdf/2505.17201v1)**

> **作者:** Chaim Chai Elchik; Fatemeh Karimi Nejadasl; Seyed Sahand Mohammadi Ziabari; Ali Mohammed Mansoor Alsahag
>
> **摘要:** Multi-object tracking (MOT) in computer vision has made significant advancements, yet tracking small fish in underwater environments presents unique challenges due to complex 3D motions and data noise. Traditional single-view MOT models often fall short in these settings. This thesis addresses these challenges by adapting state-of-the-art single-view MOT models, FairMOT and YOLOv8, for underwater fish detecting and tracking in ecological studies. The core contribution of this research is the development of a multi-view framework that utilizes stereo video inputs to enhance tracking accuracy and fish behavior pattern recognition. By integrating and evaluating these models on underwater fish video datasets, the study aims to demonstrate significant improvements in precision and reliability compared to single-view approaches. The proposed framework detects fish entities with a relative accuracy of 47% and employs stereo-matching techniques to produce a novel 3D output, providing a more comprehensive understanding of fish movements and interactions
>
---
#### [new 026] Are GNNs Worth the Effort for IoT Botnet Detection? A Comparative Study of VAE-GNN vs. ViT-MLP and VAE-MLP Approaches
- **分类: cs.CV**

- **简介: 该论文研究IoT僵尸网络检测任务，比较VAE-GNN（GCN/GAT）、VAE-MLP和ViT-MLP四种模型。在N-BaIoT数据集中，二分类任务各模型性能接近（>99.93%），但多分类时GNN基模型准确率低于VAE-MLP（86.42% vs 99.72%），表明GNN在多分类场景效果欠佳，需权衡其应用价值。**

- **链接: [http://arxiv.org/pdf/2505.17363v1](http://arxiv.org/pdf/2505.17363v1)**

> **作者:** Hassan Wasswa; Hussein Abbass; Timothy Lynar
>
> **摘要:** Due to the exponential rise in IoT-based botnet attacks, researchers have explored various advanced techniques for both dimensionality reduction and attack detection to enhance IoT security. Among these, Variational Autoencoders (VAE), Vision Transformers (ViT), and Graph Neural Networks (GNN), including Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT), have garnered significant research attention in the domain of attack detection. This study evaluates the effectiveness of four state-of-the-art deep learning architectures for IoT botnet detection: a VAE encoder with a Multi-Layer Perceptron (MLP), a VAE encoder with a GCN, a VAE encoder with a GAT, and a ViT encoder with an MLP. The evaluation is conducted on a widely studied IoT benchmark dataset--the N-BaIoT dataset for both binary and multiclass tasks. For the binary classification task, all models achieved over 99.93% in accuracy, recall, precision, and F1-score, with no notable differences in performance. In contrast, for the multiclass classification task, GNN-based models showed significantly lower performance compared to VAE-MLP and ViT-MLP, with accuracies of 86.42%, 89.46%, 99.72%, and 98.38% for VAE-GCN, VAE-GAT, VAE-MLP, and ViT-MLP, respectively.
>
---
#### [new 027] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文属于统一多模态理解和生成任务，旨在解决现有模型难以同时优化这两项能力的问题。提出CoRL框架，通过协同强化学习实现联合优化与任务细化，提升文本到图像生成和多模态理解性能。**

- **链接: [http://arxiv.org/pdf/2505.17534v1](http://arxiv.org/pdf/2505.17534v1)**

> **作者:** Jingjing Jiang; Chongjie Si; Jun Luo; Hanwang Zhang; Chao Ma
>
> **摘要:** This paper presents a pioneering exploration of reinforcement learning (RL) via group relative policy optimization for unified multimodal large language models (ULMs), aimed at simultaneously reinforcing generation and understanding capabilities. Through systematic pilot studies, we uncover the significant potential of ULMs to enable the synergistic co-evolution of dual capabilities within a shared policy optimization framework. Building on this insight, we introduce \textbf{CoRL}, a co-reinforcement learning framework comprising a unified RL stage for joint optimization and a refined RL stage for task-specific enhancement. With the proposed CoRL, our resulting model, \textbf{ULM-R1}, achieves average improvements of \textbf{7%} on three text-to-image generation datasets and \textbf{23%} on nine multimodal understanding benchmarks. These results demonstrate the effectiveness of CoRL and highlight the substantial benefit of reinforcement learning in facilitating cross-task synergy and optimization for ULMs.
>
---
#### [new 028] Enhancing Adversarial Robustness of Vision Language Models via Adversarial Mixture Prompt Tuning
- **分类: cs.CV**

- **简介: 该论文属于提升视觉语言模型（VLMs）对抗鲁棒性的任务，针对单一文本提示在对抗攻击下泛化不足的问题，提出Adversarial Mixture Prompt Tuning（AMPT）。通过学习多组混合提示并结合条件权重路由机制，动态聚合文本特征以适应不同对抗图像，实验显示其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.17509v1](http://arxiv.org/pdf/2505.17509v1)**

> **作者:** Shiji Zhao; Qihui Zhu; Shukun Xiong; Shouwei Ruan; Yize Fan; Ranjie Duan; Qing Guo; Xingxing Wei
>
> **摘要:** Large pre-trained Vision Language Models (VLMs) have excellent generalization capabilities but are highly susceptible to adversarial examples, presenting potential security risks. To improve the robustness of VLMs against adversarial examples, adversarial prompt tuning methods are proposed to align the text feature with the adversarial image feature without changing model parameters. However, when facing various adversarial attacks, a single learnable text prompt has insufficient generalization to align well with all adversarial image features, which finally leads to the overfitting phenomenon. To address the above challenge, in this paper, we empirically find that increasing the number of learned prompts can bring more robustness improvement than a longer prompt. Then we propose an adversarial tuning method named Adversarial Mixture Prompt Tuning (AMPT) to enhance the generalization towards various adversarial attacks for VLMs. AMPT aims to learn mixture text prompts to obtain more robust text features. To further enhance the adaptability, we propose a conditional weight router based on the input adversarial image to predict the mixture weights of multiple learned prompts, which helps obtain sample-specific aggregated text features aligning with different adversarial image features. A series of experiments show that our method can achieve better adversarial robustness than state-of-the-art methods on 11 datasets under different experimental settings.
>
---
#### [new 029] Dual Ascent Diffusion for Inverse Problems
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **简介: 该论文针对逆问题（如图像恢复）中现有MAP或后验采样方法因计算近似导致精度不足的问题，提出基于对偶上升优化框架的扩散模型新方法，提升图像质量、抗噪性、速度及保真度，超越现有最优方法。**

- **链接: [http://arxiv.org/pdf/2505.17353v1](http://arxiv.org/pdf/2505.17353v1)**

> **作者:** Minseo Kim; Axel Levy; Gordon Wetzstein
>
> **备注:** 23 pages, 15 figures, 5 tables
>
> **摘要:** Ill-posed inverse problems are fundamental in many domains, ranging from astrophysics to medical imaging. Emerging diffusion models provide a powerful prior for solving these problems. Existing maximum-a-posteriori (MAP) or posterior sampling approaches, however, rely on different computational approximations, leading to inaccurate or suboptimal samples. To address this issue, we introduce a new approach to solving MAP problems with diffusion model priors using a dual ascent optimization framework. Our framework achieves better image quality as measured by various metrics for image restoration problems, it is more robust to high levels of measurement noise, it is faster, and it estimates solutions that represent the observations more faithfully than the state of the art.
>
---
#### [new 030] Center-aware Residual Anomaly Synthesis for Multi-class Industrial Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于多类工业异常检测任务，旨在解决现有方法需为每类单独建模导致成本高、类间干扰漏检及类内样本重叠过检的问题。提出CRAS方法，通过中心感知残差学习统一类别表征，结合距离引导的自适应噪声合成减少类内干扰，提升检测精度与速度。**

- **链接: [http://arxiv.org/pdf/2505.17551v1](http://arxiv.org/pdf/2505.17551v1)**

> **作者:** Qiyu Chen; Huiyuan Luo; Haiming Yao; Wei Luo; Zhen Qu; Chengkan Lv; Zhengtao Zhang
>
> **备注:** Accepted by IEEE Transactions on Industrial Informatics (TII)
>
> **摘要:** Anomaly detection plays a vital role in the inspection of industrial images. Most existing methods require separate models for each category, resulting in multiplied deployment costs. This highlights the challenge of developing a unified model for multi-class anomaly detection. However, the significant increase in inter-class interference leads to severe missed detections. Furthermore, the intra-class overlap between normal and abnormal samples, particularly in synthesis-based methods, cannot be ignored and may lead to over-detection. To tackle these issues, we propose a novel Center-aware Residual Anomaly Synthesis (CRAS) method for multi-class anomaly detection. CRAS leverages center-aware residual learning to couple samples from different categories into a unified center, mitigating the effects of inter-class interference. To further reduce intra-class overlap, CRAS introduces distance-guided anomaly synthesis that adaptively adjusts noise variance based on normal data distribution. Experimental results on diverse datasets and real-world industrial applications demonstrate the superior detection accuracy and competitive inference speed of CRAS. The source code and the newly constructed dataset are publicly available at https://github.com/cqylunlun/CRAS.
>
---
#### [new 031] RePrompt: Reasoning-Augmented Reprompting for Text-to-Image Generation via Reinforcement Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于文本到图像生成任务，旨在解决现有模型难以准确捕捉用户简短模糊提示的问题。提出RePrompt框架，通过强化学习引入推理机制优化提示生成，训练语言模型生成结构化提示，利用图像级奖励模型（如人类偏好、语义对齐）间接监督，实现无监督端到端训练，提升图像布局与组合生成效果，达新SOTA。**

- **链接: [http://arxiv.org/pdf/2505.17540v1](http://arxiv.org/pdf/2505.17540v1)**

> **作者:** Mingrui Wu; Lu Wang; Pu Zhao; Fangkai Yang; Jianjin Zhang; Jianfeng Liu; Yuefeng Zhan; Weihao Han; Hao Sun; Jiayi Ji; Xiaoshuai Sun; Qingwei Lin; Weiwei Deng; Dongmei Zhang; Feng Sun; Qi Zhang; Rongrong Ji
>
> **备注:** Code is available at: https://github.com/microsoft/DKI_LLM/tree/main/RePrompt
>
> **摘要:** Despite recent progress in text-to-image (T2I) generation, existing models often struggle to faithfully capture user intentions from short and under-specified prompts. While prior work has attempted to enhance prompts using large language models (LLMs), these methods frequently generate stylistic or unrealistic content due to insufficient grounding in visual semantics and real-world composition. Inspired by recent advances in reasoning for language model, we propose RePrompt, a novel reprompting framework that introduces explicit reasoning into the prompt enhancement process via reinforcement learning. Instead of relying on handcrafted rules or stylistic rewrites, our method trains a language model to generate structured, self-reflective prompts by optimizing for image-level outcomes. The tailored reward models assesse the generated images in terms of human preference, semantic alignment, and visual composition, providing indirect supervision to refine prompt generation. Our approach enables end-to-end training without human-annotated data. Experiments on GenEval and T2I-Compbench show that RePrompt significantly boosts spatial layout fidelity and compositional generalization across diverse T2I backbones, establishing new state-of-the-art results.
>
---
#### [new 032] HoloLLM: Multisensory Foundation Model for Language-Grounded Human Sensing and Reasoning
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.MM**

- **简介: 该论文提出HoloLLM模型，属于多模态语言引导的人类感知与推理任务。针对视觉主导模型在现实场景中受遮挡、光照或隐私限制的问题，整合LiDAR、红外等多传感器数据。通过设计UMIP模块解决传感器数据与文本对齐难题，并构建协作数据 pipeline 提升标注质量，在新基准测试中提升30%感知精度。**

- **链接: [http://arxiv.org/pdf/2505.17645v1](http://arxiv.org/pdf/2505.17645v1)**

> **作者:** Chuhao Zhou; Jianfei Yang
>
> **备注:** 18 pages, 13 figures, 6 tables
>
> **摘要:** Embodied agents operating in smart homes must understand human behavior through diverse sensory inputs and communicate via natural language. While Vision-Language Models (VLMs) have enabled impressive language-grounded perception, their reliance on visual data limits robustness in real-world scenarios with occlusions, poor lighting, or privacy constraints. In this paper, we introduce HoloLLM, a Multimodal Large Language Model (MLLM) that integrates uncommon but powerful sensing modalities, such as LiDAR, infrared, mmWave radar, and WiFi, to enable seamless human perception and reasoning across heterogeneous environments. We address two key challenges: (1) the scarcity of aligned modality-text data for rare sensors, and (2) the heterogeneity of their physical signal representations. To overcome these, we design a Universal Modality-Injection Projector (UMIP) that enhances pre-aligned modality embeddings with fine-grained, text-aligned features from tailored encoders via coarse-to-fine cross-attention without introducing significant alignment overhead. We further introduce a human-VLM collaborative data curation pipeline to generate paired textual annotations for sensing datasets. Extensive experiments on two newly constructed benchmarks show that HoloLLM significantly outperforms existing MLLMs, improving language-grounded human sensing accuracy by up to 30%. This work establishes a new foundation for real-world, language-informed multisensory embodied intelligence.
>
---
#### [new 033] Debiasing CLIP: Interpreting and Correcting Bias in Attention Heads
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于CLIP模型去偏任务，旨在解决其学习虚假关联（如背景/性别偏差）的问题。提出LTC框架，通过对比机制识别并消除视觉Transformer中的虚假注意力头，同时强化任务相关头，提升分类性能。实验显示最差群体准确率提升超50%。**

- **链接: [http://arxiv.org/pdf/2505.17425v1](http://arxiv.org/pdf/2505.17425v1)**

> **作者:** Wei Jie Yeo; Rui Mao; Moloud Abdar; Erik Cambria; Ranjan Satapathy
>
> **备注:** Under review
>
> **摘要:** Multimodal models like CLIP have gained significant attention due to their remarkable zero-shot performance across various tasks. However, studies have revealed that CLIP can inadvertently learn spurious associations between target variables and confounding factors. To address this, we introduce \textsc{Locate-Then-Correct} (LTC), a contrastive framework that identifies spurious attention heads in Vision Transformers via mechanistic insights and mitigates them through targeted ablation. Furthermore, LTC identifies salient, task-relevant attention heads, enabling the integration of discriminative features through orthogonal projection to improve classification performance. We evaluate LTC on benchmarks with inherent background and gender biases, achieving over a $>50\%$ gain in worst-group accuracy compared to non-training post-hoc baselines. Additionally, we visualize the representation of selected heads and find that the presented interpretation corroborates our contrastive mechanism for identifying both spurious and salient attention heads. Code available at https://github.com/wj210/CLIP_LTC.
>
---
#### [new 034] OrionBench: A Benchmark for Chart and Human-Recognizable Object Detection in Infographics
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出OrionBench基准，旨在提升视觉语言模型在信息图表中检测图表和人类可识别对象（如图标）的准确性。针对现有模型在信息图元素定位上的不足，构建含26,250张真实/78,750张合成信息图及690万标注的数据集，并通过改进VLM推理方案、模型对比及布局检测应用验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.17473v1](http://arxiv.org/pdf/2505.17473v1)**

> **作者:** Jiangning Zhu; Yuxing Zhou; Zheng Wang; Juntao Yao; Yima Gu; Yuhui Yuan; Shixia Liu
>
> **摘要:** Given the central role of charts in scientific, business, and communication contexts, enhancing the chart understanding capabilities of vision-language models (VLMs) has become increasingly critical. A key limitation of existing VLMs lies in their inaccurate visual grounding of infographic elements, including charts and human-recognizable objects (HROs) such as icons and images. However, chart understanding often requires identifying relevant elements and reasoning over them. To address this limitation, we introduce OrionBench, a benchmark designed to support the development of accurate object detection models for charts and HROs in infographics. It contains 26,250 real and 78,750 synthetic infographics, with over 6.9 million bounding box annotations. These annotations are created by combining the model-in-the-loop and programmatic methods. We demonstrate the usefulness of OrionBench through three applications: 1) constructing a Thinking-with-Boxes scheme to boost the chart understanding performance of VLMs, 2) comparing existing object detection models, and 3) applying the developed detection model to document layout and UI element detection.
>
---
#### [new 035] Ocular Authentication: Fusion of Gaze and Periocular Modalities
- **分类: cs.CV; cs.HC**

- **简介: 该论文提出融合眼动与眼周图像的无校准眼部认证系统，旨在提升生物识别性能。通过结合两种模态的互补特征，解决单一模态认证精度局限问题，利用9202人数据集验证，结果显示多模态方法超越单模态及FIDO标准，采用先进机器学习模型优化鉴别能力。**

- **链接: [http://arxiv.org/pdf/2505.17343v1](http://arxiv.org/pdf/2505.17343v1)**

> **作者:** Dillon Lohr; Michael J. Proulx; Mehedi Hasan Raju; Oleg V. Komogortsev
>
> **备注:** Supplementary material is available
>
> **摘要:** This paper investigates the feasibility of fusing two eye-centric authentication modalities-eye movements and periocular images-within a calibration-free authentication system. While each modality has independently shown promise for user authentication, their combination within a unified gaze-estimation pipeline has not been thoroughly explored at scale. In this report, we propose a multimodal authentication system and evaluate it using a large-scale in-house dataset comprising 9202 subjects with an eye tracking (ET) signal quality equivalent to a consumer-facing virtual reality (VR) device. Our results show that the multimodal approach consistently outperforms both unimodal systems across all scenarios, surpassing the FIDO benchmark. The integration of a state-of-the-art machine learning architecture contributed significantly to the overall authentication performance at scale, driven by the model's ability to capture authentication representations and the complementary discriminative characteristics of the fused modalities.
>
---
#### [new 036] Deeper Diffusion Models Amplify Bias
- **分类: cs.CV**

- **简介: 该论文研究扩散模型的偏差-方差权衡问题，揭示深层模型可能放大训练数据固有偏差并威胁隐私。提出无训练优化方法：通过部分绕过生成过程的中间层，引入高方差提升图像质量，经理论和实证验证有效。**

- **链接: [http://arxiv.org/pdf/2505.17560v1](http://arxiv.org/pdf/2505.17560v1)**

> **作者:** Shahin Hakemi; Naveed Akhtar; Ghulam Mubashar Hassan; Ajmal Mian
>
> **摘要:** Despite the impressive performance of generative Diffusion Models (DMs), their internal working is still not well understood, which is potentially problematic. This paper focuses on exploring the important notion of bias-variance tradeoff in diffusion models. Providing a systematic foundation for this exploration, it establishes that at one extreme the diffusion models may amplify the inherent bias in the training data and, on the other, they may compromise the presumed privacy of the training samples. Our exploration aligns with the memorization-generalization understanding of the generative models, but it also expands further along this spectrum beyond ``generalization'', revealing the risk of bias amplification in deeper models. Building on the insights, we also introduce a training-free method to improve output quality in text-to-image and image-to-image generation. By progressively encouraging temporary high variance in the generation process with partial bypassing of the mid-block's contribution in the denoising process of DMs, our method consistently improves generative image quality with zero training cost. Our claims are validated both theoretically and empirically.
>
---
#### [new 037] SemSegBench & DetecBench: Benchmarking Reliability and Generalization Beyond Classification
- **分类: cs.CV; cs.LG**

- **简介: 该论文针对语义分割与目标检测任务，解决其在分布偏移和对抗攻击下的可靠性与泛化性评估不足问题，提出SemSegBench和DetecBench基准工具，评估76种分割模型（4数据集）和61种检测模型（2数据集）的鲁棒性表现，揭示模型弱点并开源6139项评测数据。**

- **链接: [http://arxiv.org/pdf/2505.18015v1](http://arxiv.org/pdf/2505.18015v1)**

> **作者:** Shashank Agnihotri; David Schader; Jonas Jakubassa; Nico Sharei; Simon Kral; Mehmet Ege Kaçar; Ruben Weber; Margret Keuper
>
> **备注:** First seven listed authors have equal contribution. GitHub: https://github.com/shashankskagnihotri/benchmarking_reliability_generalization. arXiv admin note: text overlap with arXiv:2505.05091
>
> **摘要:** Reliability and generalization in deep learning are predominantly studied in the context of image classification. Yet, real-world applications in safety-critical domains involve a broader set of semantic tasks, such as semantic segmentation and object detection, which come with a diverse set of dedicated model architectures. To facilitate research towards robust model design in segmentation and detection, our primary objective is to provide benchmarking tools regarding robustness to distribution shifts and adversarial manipulations. We propose the benchmarking tools SEMSEGBENCH and DETECBENCH, along with the most extensive evaluation to date on the reliability and generalization of semantic segmentation and object detection models. In particular, we benchmark 76 segmentation models across four datasets and 61 object detectors across two datasets, evaluating their performance under diverse adversarial attacks and common corruptions. Our findings reveal systematic weaknesses in state-of-the-art models and uncover key trends based on architecture, backbone, and model capacity. SEMSEGBENCH and DETECBENCH are open-sourced in our GitHub repository (https://github.com/shashankskagnihotri/benchmarking_reliability_generalization) along with our complete set of total 6139 evaluations. We anticipate the collected data to foster and encourage future research towards improved model reliability beyond classification.
>
---
#### [new 038] Towards Dynamic 3D Reconstruction of Hand-Instrument Interaction in Ophthalmic Surgery
- **分类: cs.CV**

- **简介: 该论文聚焦眼科手术中手-器械动态3D重建任务，针对数据与标注工具不足问题，构建了含710万帧的OphNet-3D数据集，并设计自动标注流程；提出H-Net和OH-Net模型，实现高精度双手姿态及交互重建，显著提升MPJPE和ADD-S指标。**

- **链接: [http://arxiv.org/pdf/2505.17677v1](http://arxiv.org/pdf/2505.17677v1)**

> **作者:** Ming Hu; Zhendi Yu; Feilong Tang; Kaiwen Chen; Yulong Li; Imran Razzak; Junjun He; Tolga Birdal; Kaijing Zhou; Zongyuan Ge
>
> **摘要:** Accurate 3D reconstruction of hands and instruments is critical for vision-based analysis of ophthalmic microsurgery, yet progress has been hampered by the lack of realistic, large-scale datasets and reliable annotation tools. In this work, we introduce OphNet-3D, the first extensive RGB-D dynamic 3D reconstruction dataset for ophthalmic surgery, comprising 41 sequences from 40 surgeons and totaling 7.1 million frames, with fine-grained annotations of 12 surgical phases, 10 instrument categories, dense MANO hand meshes, and full 6-DoF instrument poses. To scalably produce high-fidelity labels, we design a multi-stage automatic annotation pipeline that integrates multi-view data observation, data-driven motion prior with cross-view geometric consistency and biomechanical constraints, along with a combination of collision-aware interaction constraints for instrument interactions. Building upon OphNet-3D, we establish two challenging benchmarks-bimanual hand pose estimation and hand-instrument interaction reconstruction-and propose two dedicated architectures: H-Net for dual-hand mesh recovery and OH-Net for joint reconstruction of two-hand-two-instrument interactions. These models leverage a novel spatial reasoning module with weak-perspective camera modeling and collision-aware center-based representation. Both architectures outperform existing methods by substantial margins, achieving improvements of over 2mm in Mean Per Joint Position Error (MPJPE) and up to 23% in ADD-S metrics for hand and instrument reconstruction, respectively.
>
---
#### [new 039] Building Floor Number Estimation from Crowdsourced Street-Level Images: Munich Dataset and Baseline Method
- **分类: cs.CV**

- **简介: 该论文提出基于街景图像的建筑楼层数估计方法，解决城市数据库中垂直结构信息缺失问题。构建包含6800+标注图像的慕尼黑数据集，开发分类-回归深度网络模型，实现81.2%精确匹配率和97.9%±1层误差范围，提供开源数据与代码支持城市三维建模研究。**

- **链接: [http://arxiv.org/pdf/2505.18021v1](http://arxiv.org/pdf/2505.18021v1)**

> **作者:** Yao Sun; Sining Chen; Yifan Tian; Xiao Xiang Zhu
>
> **备注:** Code and data: https://github.com/ya0-sun/Munich-SVI-Floor-Benchmark
>
> **摘要:** Accurate information on the number of building floors, or above-ground storeys, is essential for household estimation, utility provision, risk assessment, evacuation planning, and energy modeling. Yet large-scale floor-count data are rarely available in cadastral and 3D city databases. This study proposes an end-to-end deep learning framework that infers floor numbers directly from unrestricted, crowdsourced street-level imagery, avoiding hand-crafted features and generalizing across diverse facade styles. To enable benchmarking, we release the Munich Building Floor Dataset, a public set of over 6800 geo-tagged images collected from Mapillary and targeted field photography, each paired with a verified storey label. On this dataset, the proposed classification-regression network attains 81.2% exact accuracy and predicts 97.9% of buildings within +/-1 floor. The method and dataset together offer a scalable route to enrich 3D city models with vertical information and lay a foundation for future work in urban informatics, remote sensing, and geographic information science. Source code and data will be released under an open license at https://github.com/ya0-sun/Munich-SVI-Floor-Benchmark.
>
---
#### [new 040] Real-time Traffic Accident Anticipation with Feature Reuse
- **分类: cs.CV**

- **简介: 论文属于实时交通事故预测任务，解决现有方法计算复杂度高、难以实时部署的问题。提出RARE框架，复用预训练物体检测器的中间特征，消除冗余计算以降低延迟，并设计Attention Score Ranking Loss提升事故相关目标的注意力与模型可解释性。实现73.3 FPS的实时性能，达state-of-the-art精度。**

- **链接: [http://arxiv.org/pdf/2505.17449v1](http://arxiv.org/pdf/2505.17449v1)**

> **作者:** Inpyo Song; Jangwon Lee
>
> **备注:** Accepted to ICIP 2025
>
> **摘要:** This paper addresses the problem of anticipating traffic accidents, which aims to forecast potential accidents before they happen. Real-time anticipation is crucial for safe autonomous driving, yet most methods rely on computationally heavy modules like optical flow and intermediate feature extractors, making real-world deployment challenging. In this paper, we thus introduce RARE (Real-time Accident anticipation with Reused Embeddings), a lightweight framework that capitalizes on intermediate features from a single pre-trained object detector. By eliminating additional feature-extraction pipelines, RARE significantly reduces latency. Furthermore, we introduce a novel Attention Score Ranking Loss, which prioritizes higher attention on accident-related objects over non-relevant ones. This loss enhances both accuracy and interpretability. RARE demonstrates a 4-8 times speedup over existing approaches on the DAD and CCD benchmarks, achieving a latency of 13.6ms per frame (73.3 FPS) on an RTX 6000. Moreover, despite its reduced complexity, it attains state-of-the-art Average Precision and reliably anticipates imminent collisions in real time. These results highlight RARE's potential for safety-critical applications where timely and explainable anticipation is essential.
>
---
#### [new 041] Evaluation of Few-Shot Learning Methods for Kidney Stone Type Recognition in Ureteroscopy
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于肾结石类型识别任务，旨在解决深度学习模型因训练数据不足导致的分类难题。研究提出基于原型网络的少样本学习方法，在输尿管镜图像稀缺或存在罕见类别时，利用有限数据（如25%训练集）实现与传统模型相当或更优的分类性能。**

- **链接: [http://arxiv.org/pdf/2505.17921v1](http://arxiv.org/pdf/2505.17921v1)**

> **作者:** Carlos Salazar-Ruiz; Francisco Lopez-Tiro; Ivan Reyes-Amezcua; Clement Larose; Gilberto Ochoa-Ruiz; Christian Daul
>
> **备注:** 6 pages, 3 figures, 3 tables, conference, cbms25
>
> **摘要:** Determining the type of kidney stones is crucial for prescribing appropriate treatments to prevent recurrence. Currently, various approaches exist to identify the type of kidney stones. However, obtaining results through the reference ex vivo identification procedure can take several weeks, while in vivo visual recognition requires highly trained specialists. For this reason, deep learning models have been developed to provide urologists with an automated classification of kidney stones during ureteroscopies. Nevertheless, a common issue with these models is the lack of training data. This contribution presents a deep learning method based on few-shot learning, aimed at producing sufficiently discriminative features for identifying kidney stone types in endoscopic images, even with a very limited number of samples. This approach was specifically designed for scenarios where endoscopic images are scarce or where uncommon classes are present, enabling classification even with a limited training dataset. The results demonstrate that Prototypical Networks, using up to 25% of the training data, can achieve performance equal to or better than traditional deep learning models trained with the complete dataset.
>
---
#### [new 042] Game-invariant Features Through Contrastive and Domain-adversarial Learning
- **分类: cs.CV**

- **简介: 该论文属于跨游戏视觉特征学习任务，旨在解决现有模型因过度拟合特定游戏视觉风格而影响跨游戏泛化的问题。提出结合对比学习（聚合同一内容）与域对抗训练（抑制游戏特异性），使模型学习游戏无关的通用特征。实验表明其特征可跨多种游戏通用，提升故障检测等任务的迁移能力。**

- **链接: [http://arxiv.org/pdf/2505.17328v1](http://arxiv.org/pdf/2505.17328v1)**

> **作者:** Dylan Kline
>
> **摘要:** Foundational game-image encoders often overfit to game-specific visual styles, undermining performance on downstream tasks when applied to new games. We present a method that combines contrastive learning and domain-adversarial training to learn game-invariant visual features. By simultaneously encouraging similar content to cluster and discouraging game-specific cues via an adversarial domain classifier, our approach produces embeddings that generalize across diverse games. Experiments on the Bingsu game-image dataset (10,000 screenshots from 10 games) demonstrate that after only a few training epochs, our model's features no longer cluster by game, indicating successful invariance and potential for improved cross-game transfer (e.g., glitch detection) with minimal fine-tuning. This capability paves the way for more generalizable game vision models that require little to no retraining on new games.
>
---
#### [new 043] Track Anything Annotate: Video annotation and dataset generation of computer vision models
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉数据集生成任务，旨在解决标注数据耗时费力的问题。提出基于视频跟踪与分割的标注工具原型，通过技术选型与实现加速数据集生成，替代手动标注。**

- **链接: [http://arxiv.org/pdf/2505.17884v1](http://arxiv.org/pdf/2505.17884v1)**

> **作者:** Nikita Ivanov; Mark Klimov; Dmitry Glukhikh; Tatiana Chernysheva; Igor Glukhikh
>
> **备注:** 9 pages, 11 figures
>
> **摘要:** Modern machine learning methods require significant amounts of labelled data, making the preparation process time-consuming and resource-intensive. In this paper, we propose to consider the process of prototyping a tool for annotating and generating training datasets based on video tracking and segmentation. We examine different approaches to solving this problem, from technology selection through to final implementation. The developed prototype significantly accelerates dataset generation compared to manual annotation. All resources are available at https://github.com/lnikioffic/track-anything-annotate
>
---
#### [new 044] SpikeGen: Generative Framework for Visual Spike Stream Processing
- **分类: cs.CV**

- **简介: 该论文提出生成模型框架SpikeGen，解决脉冲相机空间稀疏视觉数据问题。通过融合脉冲流与RGB信息，实现去模糊、密集帧重建及新视角合成，利用生成模型的潜在空间操作提升多模态协同效果。**

- **链接: [http://arxiv.org/pdf/2505.18049v1](http://arxiv.org/pdf/2505.18049v1)**

> **作者:** Gaole Dai; Menghang Dong; Rongyu Zhang; Ruichuan An; Shanghang Zhang; Tiejun Huang
>
> **摘要:** Neuromorphic Visual Systems, such as spike cameras, have attracted considerable attention due to their ability to capture clear textures under dynamic conditions. This capability effectively mitigates issues related to motion and aperture blur. However, in contrast to conventional RGB modalities that provide dense spatial information, these systems generate binary, spatially sparse frames as a trade-off for temporally rich visual streams. In this context, generative models emerge as a promising solution to address the inherent limitations of sparse data. These models not only facilitate the conditional fusion of existing information from both spike and RGB modalities but also enable the conditional generation based on latent priors. In this study, we introduce a robust generative processing framework named SpikeGen, designed for visual spike streams captured by spike cameras. We evaluate this framework across multiple tasks involving mixed spike-RGB modalities, including conditional image/video deblurring, dense frame reconstruction from spike streams, and high-speed scene novel-view synthesis. Supported by comprehensive experimental results, we demonstrate that leveraging the latent space operation abilities of generative models allows us to effectively address the sparsity of spatial information while fully exploiting the temporal richness of spike streams, thereby promoting a synergistic enhancement of different visual modalities.
>
---
#### [new 045] RQR3D: Reparametrizing the regression targets for BEV-based 3D object detection
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于BEV（鸟瞰图）基于的3D物体检测任务，旨在解决角度表示导致的回归不连续问题。提出RQR3D方法，通过回归最小包围盒及角点偏移将旋转框检测转为关键点回归，同时设计简化雷达融合网络（用2D卷积替代稀疏卷积），提升检测精度与效率，在nuScenes数据集达SOTA。**

- **链接: [http://arxiv.org/pdf/2505.17732v1](http://arxiv.org/pdf/2505.17732v1)**

> **作者:** Ozsel Kilinc; Cem Tarhan
>
> **摘要:** Accurate, fast, and reliable 3D perception is essential for autonomous driving. Recently, bird's-eye view (BEV)-based perception approaches have emerged as superior alternatives to perspective-based solutions, offering enhanced spatial understanding and more natural outputs for planning. Existing BEV-based 3D object detection methods, typically adhering to angle-based representation, directly estimate the size and orientation of rotated bounding boxes. We observe that BEV-based 3D object detection is analogous to aerial oriented object detection, where angle-based methods are recognized for being affected by discontinuities in their loss functions. Drawing inspiration from this domain, we propose Restricted Quadrilateral Representation to define 3D regression targets. RQR3D regresses the smallest horizontal bounding box encapsulating the oriented box, along with the offsets between the corners of these two boxes, thereby transforming the oriented object detection problem into a keypoint regression task. RQR3D is compatible with any 3D object detection approach. We employ RQR3D within an anchor-free single-stage object detection method and introduce an objectness head to address class imbalance problem. Furthermore, we introduce a simplified radar fusion backbone that eliminates the need for voxel grouping and processes the BEV-mapped point cloud with standard 2D convolutions, rather than sparse convolutions. Extensive evaluations on the nuScenes dataset demonstrate that RQR3D achieves state-of-the-art performance in camera-radar 3D object detection, outperforming the previous best method by +4% in NDS and +2.4% in mAP, and significantly reducing the translation and orientation errors, which are crucial for safe autonomous driving. These consistent gains highlight the robustness, precision, and real-world readiness of our approach.
>
---
#### [new 046] Render-FM: A Foundation Model for Real-time Photorealistic Volumetric Rendering
- **分类: cs.CV; cs.AI**

- **简介: 论文提出Render-FM，一种实时光追体积渲染基础模型，解决CT扫描现有方法需耗时逐场景优化的问题。通过编码器-解码器架构直接回归6DGS参数，预训练消除优化步骤，将准备时间从小时级压缩至秒级，实现高质量临床3D可视化。**

- **链接: [http://arxiv.org/pdf/2505.17338v1](http://arxiv.org/pdf/2505.17338v1)**

> **作者:** Zhongpai Gao; Meng Zheng; Benjamin Planche; Anwesa Choudhuri; Terrence Chen; Ziyan Wu
>
> **摘要:** Volumetric rendering of Computed Tomography (CT) scans is crucial for visualizing complex 3D anatomical structures in medical imaging. Current high-fidelity approaches, especially neural rendering techniques, require time-consuming per-scene optimization, limiting clinical applicability due to computational demands and poor generalizability. We propose Render-FM, a novel foundation model for direct, real-time volumetric rendering of CT scans. Render-FM employs an encoder-decoder architecture that directly regresses 6D Gaussian Splatting (6DGS) parameters from CT volumes, eliminating per-scan optimization through large-scale pre-training on diverse medical data. By integrating robust feature extraction with the expressive power of 6DGS, our approach efficiently generates high-quality, real-time interactive 3D visualizations across diverse clinical CT data. Experiments demonstrate that Render-FM achieves visual fidelity comparable or superior to specialized per-scan methods while drastically reducing preparation time from nearly an hour to seconds for a single inference step. This advancement enables seamless integration into real-time surgical planning and diagnostic workflows. The project page is: https://gaozhongpai.github.io/renderfm/.
>
---
#### [new 047] CAS-IQA: Teaching Vision-Language Models for Synthetic Angiography Quality Assessment
- **分类: cs.CV**

- **简介: 该论文提出CAS-IQA框架，解决合成血管造影质量评估中辅助图像未利用及临床指标缺失的问题。通过视觉语言模型融合多模态信息，设计MUST模块优化特征，构建CAS-3K数据集并定义临床相关指标，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.17619v1](http://arxiv.org/pdf/2505.17619v1)**

> **作者:** Bo Wang; De-Xing Huang; Xiao-Hu Zhou; Mei-Jiang Gui; Nu-Fang Xiao; Jian-Long Hao; Ming-Yuan Liu; Zeng-Guang Hou
>
> **备注:** Under review
>
> **摘要:** Synthetic X-ray angiographies generated by modern generative models hold great potential to reduce the use of contrast agents in vascular interventional procedures. However, low-quality synthetic angiographies can significantly increase procedural risk, underscoring the need for reliable image quality assessment (IQA) methods. Existing IQA models, however, fail to leverage auxiliary images as references during evaluation and lack fine-grained, task-specific metrics necessary for clinical relevance. To address these limitations, this paper proposes CAS-IQA, a vision-language model (VLM)-based framework that predicts fine-grained quality scores by effectively incorporating auxiliary information from related images. In the absence of angiography datasets, CAS-3K is constructed, comprising 3,565 synthetic angiographies along with score annotations. To ensure clinically meaningful assessment, three task-specific evaluation metrics are defined. Furthermore, a Multi-path featUre fuSion and rouTing (MUST) module is designed to enhance image representations by adaptively fusing and routing visual tokens to metric-specific branches. Extensive experiments on the CAS-3K dataset demonstrate that CAS-IQA significantly outperforms state-of-the-art IQA methods by a considerable margin.
>
---
#### [new 048] EVM-Fusion: An Explainable Vision Mamba Architecture with Neural Algorithmic Fusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对多器官医学图像分类任务，解决准确性、可解释性与泛化性挑战。提出EVM-Fusion架构，结合DenseNet、U-Net路径与Vision Mamba模块，通过两阶段融合（交叉注意力+迭代神经算法融合块）动态整合特征，并嵌入多层级可解释性模块（如注意力图与Δ值），在9类医疗数据集上达99.75%准确率，提升医疗AI可信度。**

- **链接: [http://arxiv.org/pdf/2505.17367v1](http://arxiv.org/pdf/2505.17367v1)**

> **作者:** Zichuan Yang
>
> **备注:** 16 pages, 4 figures
>
> **摘要:** Medical image classification is critical for clinical decision-making, yet demands for accuracy, interpretability, and generalizability remain challenging. This paper introduces EVM-Fusion, an Explainable Vision Mamba architecture featuring a novel Neural Algorithmic Fusion (NAF) mechanism for multi-organ medical image classification. EVM-Fusion leverages a multipath design, where DenseNet and U-Net based pathways, enhanced by Vision Mamba (Vim) modules, operate in parallel with a traditional feature pathway. These diverse features are dynamically integrated via a two-stage fusion process: cross-modal attention followed by the iterative NAF block, which learns an adaptive fusion algorithm. Intrinsic explainability is embedded through path-specific spatial attention, Vim {\Delta}-value maps, traditional feature SE-attention, and cross-modal attention weights. Experiments on a diverse 9-class multi-organ medical image dataset demonstrate EVM-Fusion's strong classification performance, achieving 99.75% test accuracy and provide multi-faceted insights into its decision-making process, highlighting its potential for trustworthy AI in medical diagnostics.
>
---
#### [new 049] Hephaestus Minicubes: A Global, Multi-Modal Dataset for Volcanic Unrest Monitoring
- **分类: cs.CV**

- **简介: 该论文构建多模态数据集Hephaestus Minicubes，用于火山 unrest 监测任务。针对InSAR地面变形分析中缺乏深度学习数据集的问题，整合了38个时空立方体（含InSAR、地形及大气数据），覆盖44座火山的7年观测，添加专家标注的变形事件信息，并通过分类和分割基准测试验证其有效性，推动机器学习在火山监测中的应用。**

- **链接: [http://arxiv.org/pdf/2505.17782v1](http://arxiv.org/pdf/2505.17782v1)**

> **作者:** Nikolas Papadopoulos; Nikolaos Ioannis Bountos; Maria Sdraka; Andreas Karavias; Ioannis Papoutsis
>
> **摘要:** Ground deformation is regarded in volcanology as a key precursor signal preceding volcanic eruptions. Satellite-based Interferometric Synthetic Aperture Radar (InSAR) enables consistent, global-scale deformation tracking; however, deep learning methods remain largely unexplored in this domain, mainly due to the lack of a curated machine learning dataset. In this work, we build on the existing Hephaestus dataset, and introduce Hephaestus Minicubes, a global collection of 38 spatiotemporal datacubes offering high resolution, multi-source and multi-temporal information, covering 44 of the world's most active volcanoes over a 7-year period. Each spatiotemporal datacube integrates InSAR products, topographic data, as well as atmospheric variables which are known to introduce signal delays that can mimic ground deformation in InSAR imagery. Furthermore, we provide expert annotations detailing the type, intensity and spatial extent of deformation events, along with rich text descriptions of the observed scenes. Finally, we present a comprehensive benchmark, demonstrating Hephaestus Minicubes' ability to support volcanic unrest monitoring as a multi-modal, multi-temporal classification and semantic segmentation task, establishing strong baselines with state-of-the-art architectures. This work aims to advance machine learning research in volcanic monitoring, contributing to the growing integration of data-driven methods within Earth science applications.
>
---
#### [new 050] Proto-FG3D: Prototype-based Interpretable Fine-Grained 3D Shape Classification
- **分类: cs.CV; I.4.0; I.5.0**

- **简介: 该论文属于细粒度3D形状分类任务，针对多视角特征聚合信息不足、类间细微差异识别困难及模型不可解释性问题，提出Proto-FG3D框架。通过原型关联、在线聚类优化及原型引导监督学习，提升分类精度与可解释性，实验验证其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.17666v1](http://arxiv.org/pdf/2505.17666v1)**

> **作者:** Shuxian Ma; Zihao Dong; Runmin Cong; Sam Kwong; Xiuli Shao
>
> **备注:** 11 pages, 2 figures, 5 tablets; Submitted to BMVC2025
>
> **摘要:** Deep learning-based multi-view coarse-grained 3D shape classification has achieved remarkable success over the past decade, leveraging the powerful feature learning capabilities of CNN-based and ViT-based backbones. However, as a challenging research area critical for detailed shape understanding, fine-grained 3D classification remains understudied due to the limited discriminative information captured during multi-view feature aggregation, particularly for subtle inter-class variations, class imbalance, and inherent interpretability limitations of parametric model. To address these problems, we propose the first prototype-based framework named Proto-FG3D for fine-grained 3D shape classification, achieving a paradigm shift from parametric softmax to non-parametric prototype learning. Firstly, Proto-FG3D establishes joint multi-view and multi-category representation learning via Prototype Association. Secondly, prototypes are refined via Online Clustering, improving both the robustness of multi-view feature allocation and inter-subclass balance. Finally, prototype-guided supervised learning is established to enhance fine-grained discrimination via prototype-view correlation analysis and enables ad-hoc interpretability through transparent case-based reasoning. Experiments on FG3D and ModelNet40 show Proto-FG3D surpasses state-of-the-art methods in accuracy, transparent predictions, and ad-hoc interpretability with visualizations, challenging conventional fine-grained 3D recognition approaches.
>
---
#### [new 051] PathoSCOPE: Few-Shot Pathology Detection via Self-Supervised Contrastive Learning and Pathology-Informed Synthetic Embeddings
- **分类: cs.CV**

- **简介: 该论文提出PathoSCOPE框架，解决无监督病理检测中健康数据不足的问题。通过Global-Local对比损失（GLCL）减少正常样本变异并增强病灶区分，结合病理引导合成嵌入（PiEG）模块，仅需少量正常样本实现高效检测，在医疗数据集达 state-of-the-art性能。**

- **链接: [http://arxiv.org/pdf/2505.17614v1](http://arxiv.org/pdf/2505.17614v1)**

> **作者:** Sinchee Chin; Yinuo Ma; Xiaochen Yang; Jing-Hao Xue; Wenming Yang
>
> **摘要:** Unsupervised pathology detection trains models on non-pathological data to flag deviations as pathologies, offering strong generalizability for identifying novel diseases and avoiding costly annotations. However, building reliable normality models requires vast healthy datasets, as hospitals' data is inherently biased toward symptomatic populations, while privacy regulations hinder the assembly of representative healthy cohorts. To address this limitation, we propose PathoSCOPE, a few-shot unsupervised pathology detection framework that requires only a small set of non-pathological samples (minimum 2 shots), significantly improving data efficiency. We introduce Global-Local Contrastive Loss (GLCL), comprised of a Local Contrastive Loss to reduce the variability of non-pathological embeddings and a Global Contrastive Loss to enhance the discrimination of pathological regions. We also propose a Pathology-informed Embedding Generation (PiEG) module that synthesizes pathological embeddings guided by the global loss, better exploiting the limited non-pathological samples. Evaluated on the BraTS2020 and ChestXray8 datasets, PathoSCOPE achieves state-of-the-art performance among unsupervised methods while maintaining computational efficiency (2.48 GFLOPs, 166 FPS).
>
---
#### [new 052] FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶视觉推理任务。针对现有视觉语言模型（VLM）因离散文本推理导致的时空关系模糊及信息损失问题，提出时空CoT方法。通过VLM生成统一图像帧预测未来时空状态，构建逆动力学模型进行路径规划，并设计融合生成与理解的预训练范式及渐进式图像生成技术，提升自动驾驶的视觉推理能力。**

- **链接: [http://arxiv.org/pdf/2505.17685v1](http://arxiv.org/pdf/2505.17685v1)**

> **作者:** Shuang Zeng; Xinyuan Chang; Mengwei Xie; Xinran Liu; Yifan Bai; Zheng Pan; Mu Xu; Xing Wei
>
> **摘要:** Visual language models (VLMs) have attracted increasing interest in autonomous driving due to their powerful reasoning capabilities. However, existing VLMs typically utilize discrete text Chain-of-Thought (CoT) tailored to the current scenario, which essentially represents highly abstract and symbolic compression of visual information, potentially leading to spatio-temporal relationship ambiguity and fine-grained information loss. Is autonomous driving better modeled on real-world simulation and imagination than on pure symbolic logic? In this paper, we propose a spatio-temporal CoT reasoning method that enables models to think visually. First, VLM serves as a world model to generate unified image frame for predicting future world states: where perception results (e.g., lane divider and 3D detection) represent the future spatial relationships, and ordinary future frame represent the temporal evolution relationships. This spatio-temporal CoT then serves as intermediate reasoning steps, enabling the VLM to function as an inverse dynamics model for trajectory planning based on current observations and future predictions. To implement visual generation in VLMs, we propose a unified pretraining paradigm integrating visual generation and understanding, along with a progressive visual CoT enhancing autoregressive image generation. Extensive experimental results demonstrate the effectiveness of the proposed method, advancing autonomous driving towards visual reasoning.
>
---
#### [new 053] Slot-MLLM: Object-Centric Visual Tokenization for Multimodal LLM
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Slot-MLLM，针对多模态大语言模型（MLLM）在视觉-语言任务中细节处理不足的问题，提出基于Slot Attention的对象级视觉编码方法。通过结合Q-Former、扩散解码器和矢量量化技术，生成既能保留局部细节又具语义的离散视觉token，提升模型对图像内容的理解与生成能力，首次实现MLLM在自然图像中对象级slot attention的应用，显著提升多模态任务性能。**

- **链接: [http://arxiv.org/pdf/2505.17726v1](http://arxiv.org/pdf/2505.17726v1)**

> **作者:** Donghwan Chi; Hyomin Kim; Yoonjin Oh; Yongjin Kim; Donghoon Lee; Daejin Jo; Jongmin Kim; Junyeob Baek; Sungjin Ahn; Sungwoong Kim
>
> **摘要:** Recently, multimodal large language models (MLLMs) have emerged as a key approach in achieving artificial general intelligence. In particular, vision-language MLLMs have been developed to generate not only text but also visual outputs from multimodal inputs. This advancement requires efficient image tokens that LLMs can process effectively both in input and output. However, existing image tokenization methods for MLLMs typically capture only global abstract concepts or uniformly segmented image patches, restricting MLLMs' capability to effectively understand or generate detailed visual content, particularly at the object level. To address this limitation, we propose an object-centric visual tokenizer based on Slot Attention specifically for MLLMs. In particular, based on the Q-Former encoder, diffusion decoder, and residual vector quantization, our proposed discretized slot tokens can encode local visual details while maintaining high-level semantics, and also align with textual data to be integrated seamlessly within a unified next-token prediction framework of LLMs. The resulting Slot-MLLM demonstrates significant performance improvements over baselines with previous visual tokenizers across various vision-language tasks that entail local detailed comprehension and generation. Notably, this work is the first demonstration of the feasibility of object-centric slot attention performed with MLLMs and in-the-wild natural images.
>
---
#### [new 054] Seek-CAD: A Self-refined Generative Modeling for 3D Parametric CAD Using Local Inference via DeepSeek
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出Seek-CAD，基于本地开源LLM（DeepSeek-R1）通过视觉反馈与CoT迭代优化，实现无需训练的3D参数化CAD生成。旨在解决闭源大模型部署成本高问题，创新性融合SSR数据集与自优化机制，提升生成模型质量与工业适用性。**

- **链接: [http://arxiv.org/pdf/2505.17702v1](http://arxiv.org/pdf/2505.17702v1)**

> **作者:** Xueyang Li; Jiahao Li; Yu Song; Yunzhong Lou; Xiangdong Zhou
>
> **摘要:** The advent of Computer-Aided Design (CAD) generative modeling will significantly transform the design of industrial products. The recent research endeavor has extended into the realm of Large Language Models (LLMs). In contrast to fine-tuning methods, training-free approaches typically utilize the advanced closed-source LLMs, thereby offering enhanced flexibility and efficiency in the development of AI agents for generating CAD parametric models. However, the substantial cost and limitations of local deployment of the top-tier closed-source LLMs pose challenges in practical applications. The Seek-CAD is the pioneer exploration of locally deployed open-source inference LLM DeepSeek-R1 for CAD parametric model generation with a training-free methodology. This study is the first investigation to incorporate both visual and Chain-of-Thought (CoT) feedback within the self-refinement mechanism for generating CAD models. Specifically, the initial generated parametric CAD model is rendered into a sequence of step-wise perspective images, which are subsequently processed by a Vision Language Model (VLM) alongside the corresponding CoTs derived from DeepSeek-R1 to assess the CAD model generation. Then, the feedback is utilized by DeepSeek-R1 to refine the initial generated model for the next round of generation. Moreover, we present an innovative 3D CAD model dataset structured around the SSR (Sketch, Sketch-based feature, and Refinements) triple design paradigm. This dataset encompasses a wide range of CAD commands, thereby aligning effectively with industrial application requirements and proving suitable for the generation of LLMs. Extensive experiments validate the effectiveness of Seek-CAD under various metrics.
>
---
#### [new 055] Seeing It or Not? Interpretable Vision-aware Latent Steering to Mitigate Object Hallucinations
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对大型视觉语言模型（LVLMs）的物体幻觉（OH）问题，提出VaLSe框架。通过生成视觉贡献图解析输入图像对输出的影响，利用潜空间引导机制调整模型关注区域，减少不一致输出。同时揭示现有评估指标的局限性，推动更精细的OH评估发展。**

- **链接: [http://arxiv.org/pdf/2505.17812v1](http://arxiv.org/pdf/2505.17812v1)**

> **作者:** Boxu Chen; Ziwei Zheng; Le Yang; Zeyu Geng; Zhengyu Zhao; Chenhao Lin; Chao Shen
>
> **摘要:** Large Vision-Language Models (LVLMs) have achieved remarkable success but continue to struggle with object hallucination (OH), generating outputs inconsistent with visual inputs. While previous work has proposed methods to reduce OH, the visual decision-making mechanisms that lead to hallucinations remain poorly understood. In this paper, we propose VaLSe, a Vision-aware Latent Steering framework that adopts an interpretation-then-mitigation strategy to address OH in LVLMs. By tackling dual challenges of modeling complex vision-language interactions and eliminating spurious activation artifacts, VaLSe can generate visual contribution maps that trace how specific visual inputs influence individual output tokens. These maps reveal the model's vision-aware focus regions, which are then used to perform latent space steering, realigning internal representations toward semantically relevant content and reducing hallucinated outputs. Extensive experiments demonstrate that VaLSe is a powerful interpretability tool and an effective method for enhancing model robustness against OH across multiple benchmarks. Furthermore, our analysis uncovers limitations in existing OH evaluation metrics, underscoring the need for more nuanced, interpretable, and visually grounded OH benchmarks in future work. Code is available at: https://github.com/Ziwei-Zheng/VaLSe.
>
---
#### [new 056] DanceTogether! Identity-Preserving Multi-Person Interactive Video Generation
- **分类: cs.CV**

- **简介: 该论文属于可控视频生成任务，解决多角色互动视频中的身份漂移与外观混淆问题。提出DanceTogether框架，通过MaskPoseAdapter融合姿态热图与跟踪掩码，结合PairFS-4K等新数据集和评估基准，实现长时高保真多角色交互视频生成，支持人机交互等泛化场景。**

- **链接: [http://arxiv.org/pdf/2505.18078v1](http://arxiv.org/pdf/2505.18078v1)**

> **作者:** Junhao Chen; Mingjin Chen; Jianjin Xu; Xiang Li; Junting Dong; Mingze Sun; Puhua Jiang; Hongxiang Li; Yuhang Yang; Hao Zhao; Xiaoxiao Long; Ruqi Huang
>
> **备注:** Our video demos and code are available at https://DanceTog.github.io/
>
> **摘要:** Controllable video generation (CVG) has advanced rapidly, yet current systems falter when more than one actor must move, interact, and exchange positions under noisy control signals. We address this gap with DanceTogether, the first end-to-end diffusion framework that turns a single reference image plus independent pose-mask streams into long, photorealistic videos while strictly preserving every identity. A novel MaskPoseAdapter binds "who" and "how" at every denoising step by fusing robust tracking masks with semantically rich-but noisy-pose heat-maps, eliminating the identity drift and appearance bleeding that plague frame-wise pipelines. To train and evaluate at scale, we introduce (i) PairFS-4K, 26 hours of dual-skater footage with 7,000+ distinct IDs, (ii) HumanRob-300, a one-hour humanoid-robot interaction set for rapid cross-domain transfer, and (iii) TogetherVideoBench, a three-track benchmark centered on the DanceTogEval-100 test suite covering dance, boxing, wrestling, yoga, and figure skating. On TogetherVideoBench, DanceTogether outperforms the prior arts by a significant margin. Moreover, we show that a one-hour fine-tune yields convincing human-robot videos, underscoring broad generalization to embodied-AI and HRI tasks. Extensive ablations confirm that persistent identity-action binding is critical to these gains. Together, our model, datasets, and benchmark lift CVG from single-subject choreography to compositionally controllable, multi-actor interaction, opening new avenues for digital production, simulation, and embodied intelligence. Our video demos and code are available at https://DanceTog.github.io/.
>
---
#### [new 057] SafeMVDrive: Multi-view Safety-Critical Driving Video Synthesis in the Real World Domain
- **分类: cs.CV**

- **简介: 该论文提出SafeMVDrive框架，旨在生成真实场景下的多视角安全关键驾驶视频，解决现有方法无法满足端到端自动驾驶系统对多视角真实数据需求的问题。通过融合视觉语言模型增强轨迹生成、两阶段可控避撞轨迹设计及扩散模型生成高保真视频，提升安全场景测试效果。**

- **链接: [http://arxiv.org/pdf/2505.17727v1](http://arxiv.org/pdf/2505.17727v1)**

> **作者:** Jiawei Zhou; Linye Lyu; Zhuotao Tian; Cheng Zhuo; Yu Li
>
> **摘要:** Safety-critical scenarios are rare yet pivotal for evaluating and enhancing the robustness of autonomous driving systems. While existing methods generate safety-critical driving trajectories, simulations, or single-view videos, they fall short of meeting the demands of advanced end-to-end autonomous systems (E2E AD), which require real-world, multi-view video data. To bridge this gap, we introduce SafeMVDrive, the first framework designed to generate high-quality, safety-critical, multi-view driving videos grounded in real-world domains. SafeMVDrive strategically integrates a safety-critical trajectory generator with an advanced multi-view video generator. To tackle the challenges inherent in this integration, we first enhance scene understanding ability of the trajectory generator by incorporating visual context -- which is previously unavailable to such generator -- and leveraging a GRPO-finetuned vision-language model to achieve more realistic and context-aware trajectory generation. Second, recognizing that existing multi-view video generators struggle to render realistic collision events, we introduce a two-stage, controllable trajectory generation mechanism that produces collision-evasion trajectories, ensuring both video quality and safety-critical fidelity. Finally, we employ a diffusion-based multi-view video generator to synthesize high-quality safety-critical driving videos from the generated trajectories. Experiments conducted on an E2E AD planner demonstrate a significant increase in collision rate when tested with our generated data, validating the effectiveness of SafeMVDrive in stress-testing planning modules. Our code, examples, and datasets are publicly available at: https://zhoujiawei3.github.io/SafeMVDrive/.
>
---
#### [new 058] Multi-task Learning For Joint Action and Gesture Recognition
- **分类: cs.CV**

- **简介: 论文属于多任务学习任务，旨在解决动作与手势识别分离导致效率低、泛化差的问题。提出联合训练单一网络架构，利用任务间协同优化视觉表征，实验表明优于单任务方法。**

- **链接: [http://arxiv.org/pdf/2505.17867v1](http://arxiv.org/pdf/2505.17867v1)**

> **作者:** Konstantinos Spathis; Nikolaos Kardaris; Petros Maragos
>
> **摘要:** In practical applications, computer vision tasks often need to be addressed simultaneously. Multitask learning typically achieves this by jointly training a single deep neural network to learn shared representations, providing efficiency and improving generalization. Although action and gesture recognition are closely related tasks, since they focus on body and hand movements, current state-of-the-art methods handle them separately. In this paper, we show that employing a multi-task learning paradigm for action and gesture recognition results in more efficient, robust and generalizable visual representations, by leveraging the synergies between these tasks. Extensive experiments on multiple action and gesture datasets demonstrate that handling actions and gestures in a single architecture can achieve better performance for both tasks in comparison to their single-task learning variants.
>
---
#### [new 059] Dual-sensing driving detection model
- **分类: cs.CV; cs.AI; 68T07, 68T45, 68U10; I.2.10; I.4.8; J.7**

- **简介: 该论文提出结合计算机视觉与生理信号分析的双传感模型，解决单一模态方法的局限性。通过融合面部特征分析与生理信号处理，设计高效架构实现 robust 疲劳检测，在真实场景中提升准确率，减少疲劳事故，提供可靠低成本的解决方案。**

- **链接: [http://arxiv.org/pdf/2505.17392v1](http://arxiv.org/pdf/2505.17392v1)**

> **作者:** Leon C. C. K; Zeng Hui
>
> **备注:** 19 pages
>
> **摘要:** In this paper, a novel dual-sensing driver fatigue detection method combining computer vision and physiological signal analysis is proposed. The system exploits the complementary advantages of the two sensing modalities and breaks through the limitations of existing single-modality methods. We introduce an innovative architecture that combines real-time facial feature analysis with physiological signal processing, combined with advanced fusion strategies, for robust fatigue detection. The system is designed to run efficiently on existing hardware while maintaining high accuracy and reliability. Through comprehensive experiments, we demonstrate that our method outperforms traditional methods in both controlled environments and real-world conditions, while maintaining high accuracy. The practical applicability of the system has been verified through extensive tests in various driving scenarios and shows great potential in reducing fatigue-related accidents. This study contributes to the field by providing a more reliable, cost-effective, and humane solution for driver fatigue detection.
>
---
#### [new 060] Direct3D-S2: Gigascale 3D Generation Made Easy with Spatial Sparse Attention
- **分类: cs.CV**

- **简介: 该论文提出Direct3D-S2框架，解决高分辨率3D生成中的计算与内存瓶颈。通过Spatial Sparse Attention优化稀疏体积数据的扩散计算，结合统一稀疏格式的变分自编码器，提升训练效率与稳定性。实验显示其在1024分辨率下仅需8GPU，超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.17412v1](http://arxiv.org/pdf/2505.17412v1)**

> **作者:** Shuang Wu; Youtian Lin; Feihu Zhang; Yifei Zeng; Yikang Yang; Yajie Bao; Jiachen Qian; Siyu Zhu; Philip Torr; Xun Cao; Yao Yao
>
> **备注:** Project page: https://nju3dv.github.io/projects/Direct3D-S2/
>
> **摘要:** Generating high resolution 3D shapes using volumetric representations such as Signed Distance Functions presents substantial computational and memory challenges. We introduce Direct3D S2, a scalable 3D generation framework based on sparse volumes that achieves superior output quality with dramatically reduced training costs. Our key innovation is the Spatial Sparse Attention mechanism, which greatly enhances the efficiency of Diffusion Transformer computations on sparse volumetric data. SSA allows the model to effectively process large token sets within sparse volumes, significantly reducing computational overhead and achieving a 3.9x speedup in the forward pass and a 9.6x speedup in the backward pass. Our framework also includes a variational autoencoder that maintains a consistent sparse volumetric format across input, latent, and output stages. Compared to previous methods with heterogeneous representations in 3D VAE, this unified design significantly improves training efficiency and stability. Our model is trained on public available datasets, and experiments demonstrate that Direct3D S2 not only surpasses state-of-the-art methods in generation quality and efficiency, but also enables training at 1024 resolution using only 8 GPUs, a task typically requiring at least 32 GPUs for volumetric representations at 256 resolution, thus making gigascale 3D generation both practical and accessible. Project page: https://nju3dv.github.io/projects/Direct3D-S2/.
>
---
#### [new 061] Scaling Image and Video Generation via Test-Time Evolutionary Search
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于图像和视频生成任务，旨在解决现有测试时扩展方法在视觉生成中的局限性（如领域受限、可扩展性差、多样性不足）。提出EvoSearch方法，通过进化算法优化去噪过程，提升扩散模型和流模型的生成质量和多样性，无需额外训练且通用性强。**

- **链接: [http://arxiv.org/pdf/2505.17618v1](http://arxiv.org/pdf/2505.17618v1)**

> **作者:** Haoran He; Jiajun Liang; Xintao Wang; Pengfei Wan; Di Zhang; Kun Gai; Ling Pan
>
> **备注:** 37 pages. Project: https://tinnerhrhe.github.io/evosearch
>
> **摘要:** As the marginal cost of scaling computation (data and parameters) during model pre-training continues to increase substantially, test-time scaling (TTS) has emerged as a promising direction for improving generative model performance by allocating additional computation at inference time. While TTS has demonstrated significant success across multiple language tasks, there remains a notable gap in understanding the test-time scaling behaviors of image and video generative models (diffusion-based or flow-based models). Although recent works have initiated exploration into inference-time strategies for vision tasks, these approaches face critical limitations: being constrained to task-specific domains, exhibiting poor scalability, or falling into reward over-optimization that sacrifices sample diversity. In this paper, we propose \textbf{Evo}lutionary \textbf{Search} (EvoSearch), a novel, generalist, and efficient TTS method that effectively enhances the scalability of both image and video generation across diffusion and flow models, without requiring additional training or model expansion. EvoSearch reformulates test-time scaling for diffusion and flow models as an evolutionary search problem, leveraging principles from biological evolution to efficiently explore and refine the denoising trajectory. By incorporating carefully designed selection and mutation mechanisms tailored to the stochastic differential equation denoising process, EvoSearch iteratively generates higher-quality offspring while preserving population diversity. Through extensive evaluation across both diffusion and flow architectures for image and video generation tasks, we demonstrate that our method consistently outperforms existing approaches, achieves higher diversity, and shows strong generalizability to unseen evaluation metrics. Our project is available at the website https://tinnerhrhe.github.io/evosearch.
>
---
#### [new 062] Diffusion Classifiers Understand Compositionality, but Conditions Apply
- **分类: cs.CV**

- **简介: 该论文研究扩散模型在视觉组合任务中的判别能力。针对现有研究基准不足及条件分析浅显的问题，团队测试了SD 1.5/2.0/3-m模型在10个数据集、30+任务上的表现，提出Self-Bench基准（用生成图像隔离领域影响），分析了领域差距与时间步权重的关系，揭示扩散分类器的组合理解能力依赖特定条件。**

- **链接: [http://arxiv.org/pdf/2505.17955v1](http://arxiv.org/pdf/2505.17955v1)**

> **作者:** Yujin Jeong; Arnas Uselis; Seong Joon Oh; Anna Rohrbach
>
> **摘要:** Understanding visual scenes is fundamental to human intelligence. While discriminative models have significantly advanced computer vision, they often struggle with compositional understanding. In contrast, recent generative text-to-image diffusion models excel at synthesizing complex scenes, suggesting inherent compositional capabilities. Building on this, zero-shot diffusion classifiers have been proposed to repurpose diffusion models for discriminative tasks. While prior work offered promising results in discriminative compositional scenarios, these results remain preliminary due to a small number of benchmarks and a relatively shallow analysis of conditions under which the models succeed. To address this, we present a comprehensive study of the discriminative capabilities of diffusion classifiers on a wide range of compositional tasks. Specifically, our study covers three diffusion models (SD 1.5, 2.0, and, for the first time, 3-m) spanning 10 datasets and over 30 tasks. Further, we shed light on the role that target dataset domains play in respective performance; to isolate the domain effects, we introduce a new diagnostic benchmark Self-Bench comprised of images created by diffusion models themselves. Finally, we explore the importance of timestep weighting and uncover a relationship between domain gap and timestep sensitivity, particularly for SD3-m. To sum up, diffusion classifiers understand compositionality, but conditions apply! Code and dataset are available at https://github.com/eugene6923/Diffusion-Classifiers-Compositionality.
>
---
#### [new 063] Pixels Versus Priors: Controlling Knowledge Priors in Vision-Language Models through Visual Counterfacts
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究多模态模型在视觉任务中视觉输入与先验知识的交互机制，通过构建Visual CounterFact数据集（含视觉反事实样本），揭示模型推理从依赖先验转向视觉特征的动态过程，并提出PvP方法控制输出偏向视觉或先验，成功调整92.5%颜色和74.6%尺寸预测。**

- **链接: [http://arxiv.org/pdf/2505.17127v1](http://arxiv.org/pdf/2505.17127v1)**

> **作者:** Michal Golovanevsky; William Rudman; Michael Lepori; Amir Bar; Ritambhara Singh; Carsten Eickhoff
>
> **摘要:** Multimodal Large Language Models (MLLMs) perform well on tasks such as visual question answering, but it remains unclear whether their reasoning relies more on memorized world knowledge or on the visual information present in the input image. To investigate this, we introduce Visual CounterFact, a new dataset of visually-realistic counterfactuals that put world knowledge priors (e.g, red strawberry) into direct conflict with visual input (e.g, blue strawberry). Using Visual CounterFact, we show that model predictions initially reflect memorized priors, but shift toward visual evidence in mid-to-late layers. This dynamic reveals a competition between the two modalities, with visual input ultimately overriding priors during evaluation. To control this behavior, we propose Pixels Versus Priors (PvP) steering vectors, a mechanism for controlling model outputs toward either world knowledge or visual input through activation-level interventions. On average, PvP successfully shifts 92.5% of color and 74.6% of size predictions from priors to counterfactuals. Together, these findings offer new tools for interpreting and controlling factual behavior in multimodal models.
>
---
#### [new 064] Semantic Correspondence: Unified Benchmarking and a Strong Baseline
- **分类: cs.CV**

- **简介: 该论文聚焦语义对应任务，解决领域缺乏系统综述与统一基准的问题。工作包括：分类现有方法、汇总多基准测试结果、分析模型组件有效性，并提出简单高效的基线模型，达当前最优性能，推动领域发展。**

- **链接: [http://arxiv.org/pdf/2505.18060v1](http://arxiv.org/pdf/2505.18060v1)**

> **作者:** Kaiyan Zhang; Xinghui Li; Jingyi Lu; Kai Han
>
> **摘要:** Establishing semantic correspondence is a challenging task in computer vision, aiming to match keypoints with the same semantic information across different images. Benefiting from the rapid development of deep learning, remarkable progress has been made over the past decade. However, a comprehensive review and analysis of this task remains absent. In this paper, we present the first extensive survey of semantic correspondence methods. We first propose a taxonomy to classify existing methods based on the type of their method designs. These methods are then categorized accordingly, and we provide a detailed analysis of each approach. Furthermore, we aggregate and summarize the results of methods in literature across various benchmarks into a unified comparative table, with detailed configurations to highlight performance variations. Additionally, to provide a detailed understanding on existing methods for semantic matching, we thoroughly conduct controlled experiments to analyse the effectiveness of the components of different methods. Finally, we propose a simple yet effective baseline that achieves state-of-the-art performance on multiple benchmarks, providing a solid foundation for future research in this field. We hope this survey serves as a comprehensive reference and consolidated baseline for future development. Code is publicly available at: https://github.com/Visual-AI/Semantic-Correspondence.
>
---
#### [new 065] LookWhere? Efficient Visual Recognition by Learning Where to Look and What to See from Self-Supervision
- **分类: cs.CV**

- **简介: 该论文属于高效视觉识别任务，旨在解决高分辨率图像计算成本过高的问题。提出LookWhere方法，通过低分辨率选择器定位关键区域，高分辨率提取器处理局部细节，结合自监督预训练同步学习“看哪里”和“看什么”，减少FLOPs达34倍、时间6倍，提升稀疏识别（如交通标志）及全局/局部标准任务（如ImageNet分类、ADE20K分割）的效率与精度。**

- **链接: [http://arxiv.org/pdf/2505.18051v1](http://arxiv.org/pdf/2505.18051v1)**

> **作者:** Anthony Fuller; Yousef Yassin; Junfeng Wen; Daniel G. Kyrollos; Tarek Ibrahim; James R. Green; Evan Shelhamer
>
> **摘要:** Vision transformers are ever larger, more accurate, and more expensive to compute. The expense is even more extreme at high resolution as the number of tokens grows quadratically with the image size. We turn to adaptive computation to cope with this cost by learning to predict where to compute. Our LookWhere method divides the computation between a low-resolution selector and a high-resolution extractor without ever processing the full high-resolution input. We jointly pretrain the selector and extractor without task supervision by distillation from a self-supervised teacher, in effect, learning where and what to compute simultaneously. Unlike prior token reduction methods, which pay to save by pruning already-computed tokens, and prior token selection methods, which require complex and expensive per-task optimization, LookWhere economically and accurately selects and extracts transferrable representations of images. We show that LookWhere excels at sparse recognition on high-resolution inputs (Traffic Signs), maintaining accuracy while reducing FLOPs by up to 34x and time by 6x. It also excels at standard recognition tasks that are global (ImageNet classification) or local (ADE20K segmentation), improving accuracy while reducing time by 1.36x.
>
---
#### [new 066] CAMA: Enhancing Multimodal In-Context Learning with Context-Aware Modulated Attention
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对多模态in-context学习（ICL）的不稳定性问题，分析了标准注意力机制的三大缺陷，提出无需训练的CAMA方法，通过调制注意力logits优化模型内部机制，在多个模型和基准上验证了其有效性，属多模态ICL优化任务。**

- **链接: [http://arxiv.org/pdf/2505.17097v1](http://arxiv.org/pdf/2505.17097v1)**

> **作者:** Yanshu Li; JianJiang Yang; Bozheng Li; Ruixiang Tang
>
> **备注:** 10 pages, 2 figures, 6 tables
>
> **摘要:** Multimodal in-context learning (ICL) enables large vision-language models (LVLMs) to efficiently adapt to novel tasks, supporting a wide array of real-world applications. However, multimodal ICL remains unstable, and current research largely focuses on optimizing sequence configuration while overlooking the internal mechanisms of LVLMs. In this work, we first provide a theoretical analysis of attentional dynamics in multimodal ICL and identify three core limitations of standard attention that ICL impair performance. To address these challenges, we propose Context-Aware Modulated Attention (CAMA), a simple yet effective plug-and-play method for directly calibrating LVLM attention logits. CAMA is training-free and can be seamlessly applied to various open-source LVLMs. We evaluate CAMA on four LVLMs across six benchmarks, demonstrating its effectiveness and generality. CAMA opens new opportunities for deeper exploration and targeted utilization of LVLM attention dynamics to advance multimodal reasoning.
>
---
#### [new 067] Object-level Cross-view Geo-localization with Location Enhancement and Multi-Head Cross Attention
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出OCGNet，针对跨视角地理定位任务，解决无人机/地面图像与卫星图像间物体级精准匹配难题。通过整合用户点击的高斯核位置信息、位置增强模块及多头交叉注意力，提升物体特征与上下文关联，实现高精度定位与少样本学习，在CVOGL数据集达最优性能。**

- **链接: [http://arxiv.org/pdf/2505.17911v1](http://arxiv.org/pdf/2505.17911v1)**

> **作者:** Zheyang Huang; Jagannath Aryal; Saeid Nahavandi; Xuequan Lu; Chee Peng Lim; Lei Wei; Hailing Zhou
>
> **摘要:** Cross-view geo-localization determines the location of a query image, captured by a drone or ground-based camera, by matching it to a geo-referenced satellite image. While traditional approaches focus on image-level localization, many applications, such as search-and-rescue, infrastructure inspection, and precision delivery, demand object-level accuracy. This enables users to prompt a specific object with a single click on a drone image to retrieve precise geo-tagged information of the object. However, variations in viewpoints, timing, and imaging conditions pose significant challenges, especially when identifying visually similar objects in extensive satellite imagery. To address these challenges, we propose an Object-level Cross-view Geo-localization Network (OCGNet). It integrates user-specified click locations using Gaussian Kernel Transfer (GKT) to preserve location information throughout the network. This cue is dually embedded into the feature encoder and feature matching blocks, ensuring robust object-specific localization. Additionally, OCGNet incorporates a Location Enhancement (LE) module and a Multi-Head Cross Attention (MHCA) module to adaptively emphasize object-specific features or expand focus to relevant contextual regions when necessary. OCGNet achieves state-of-the-art performance on a public dataset, CVOGL. It also demonstrates few-shot learning capabilities, effectively generalizing from limited examples, making it suitable for diverse applications (https://github.com/ZheyangH/OCGNet).
>
---
#### [new 068] VEAttack: Downstream-agnostic Vision Encoder Attack against Large Vision Language Models
- **分类: cs.CV**

- **简介: 该论文提出VEAttack，一种针对大型视觉语言模型（LVLMs）视觉编码器的对抗攻击方法。解决现有攻击依赖任务、计算成本高的问题。通过优化图像token最小化视觉特征余弦相似性，无需任务/标签信息，降低计算开销。在图像描述和视觉问答任务中分别降效94.5%和75.7%，并分析攻击泛化机制。**

- **链接: [http://arxiv.org/pdf/2505.17440v1](http://arxiv.org/pdf/2505.17440v1)**

> **作者:** Hefei Mei; Zirui Wang; Shen You; Minjing Dong; Chang Xu
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities in multimodal understanding and generation, yet their vulnerability to adversarial attacks raises significant robustness concerns. While existing effective attacks always focus on task-specific white-box settings, these approaches are limited in the context of LVLMs, which are designed for diverse downstream tasks and require expensive full-model gradient computations. Motivated by the pivotal role and wide adoption of the vision encoder in LVLMs, we propose a simple yet effective Vision Encoder Attack (VEAttack), which targets the vision encoder of LVLMs only. Specifically, we propose to generate adversarial examples by minimizing the cosine similarity between the clean and perturbed visual features, without accessing the following large language models, task information, and labels. It significantly reduces the computational overhead while eliminating the task and label dependence of traditional white-box attacks in LVLMs. To make this simple attack effective, we propose to perturb images by optimizing image tokens instead of the classification token. We provide both empirical and theoretical evidence that VEAttack can easily generalize to various tasks. VEAttack has achieved a performance degradation of 94.5% on image caption task and 75.7% on visual question answering task. We also reveal some key observations to provide insights into LVLM attack/defense: 1) hidden layer variations of LLM, 2) token attention differential, 3) M\"obius band in transfer attack, 4) low sensitivity to attack steps. The code is available at https://github.com/hfmei/VEAttack-LVLM
>
---
#### [new 069] R-Genie: Reasoning-Guided Generative Image Editing
- **分类: cs.CV; F.2.2, I.2.7; F.2.2; I.2.7**

- **简介: 该论文提出R-Genie，一种推理引导的图像编辑方法，解决现有技术依赖显式指令、缺乏隐含意图理解的问题。通过构建含1000+图像-指令-编辑三元组的数据集，结合扩散模型与多模态语言模型，采用推理注意力机制，实现复杂语义及常识推理驱动的智能图像合成。**

- **链接: [http://arxiv.org/pdf/2505.17768v1](http://arxiv.org/pdf/2505.17768v1)**

> **作者:** Dong Zhang; Lingfeng He; Rui Yan; Fei Shen; Jinhui Tang
>
> **备注:** https://dongzhang89.github.io/RGenie.github.io/
>
> **摘要:** While recent advances in image editing have enabled impressive visual synthesis capabilities, current methods remain constrained by explicit textual instructions and limited editing operations, lacking deep comprehension of implicit user intentions and contextual reasoning. In this work, we introduce a new image editing paradigm: reasoning-guided generative editing, which synthesizes images based on complex, multi-faceted textual queries accepting world knowledge and intention inference. To facilitate this task, we first construct a comprehensive dataset featuring over 1,000 image-instruction-edit triples that incorporate rich reasoning contexts and real-world knowledge. We then propose R-Genie: a reasoning-guided generative image editor, which synergizes the generation power of diffusion models with advanced reasoning capabilities of multimodal large language models. R-Genie incorporates a reasoning-attention mechanism to bridge linguistic understanding with visual synthesis, enabling it to handle intricate editing requests involving abstract user intentions and contextual reasoning relations. Extensive experimental results validate that R-Genie can equip diffusion models with advanced reasoning-based editing capabilities, unlocking new potentials for intelligent image synthesis.
>
---
#### [new 070] RemoteSAM: Towards Segment Anything for Earth Observation
- **分类: cs.CV**

- **简介: 该论文提出RemoteSAM，用于地球观测的视觉基础模型。针对现有模型任务专用、数据狭窄的问题，其构建自动数据引擎生成27万图像-文本-掩码数据集，并提出基于指代分割的统一任务范式，单模型高效处理分类、检测等多任务，性能超现有模型。**

- **链接: [http://arxiv.org/pdf/2505.18022v1](http://arxiv.org/pdf/2505.18022v1)**

> **作者:** Liang Yao; Fan Liu; Delong Chen; Chuanyi Zhang; Yijun Wang; Ziyun Chen; Wei Xu; Shimin Di; Yuhui Zheng
>
> **摘要:** We aim to develop a robust yet flexible visual foundation model for Earth observation. It should possess strong capabilities in recognizing and localizing diverse visual targets while providing compatibility with various input-output interfaces required across different task scenarios. Current systems cannot meet these requirements, as they typically utilize task-specific architecture trained on narrow data domains with limited semantic coverage. Our study addresses these limitations from two aspects: data and modeling. We first introduce an automatic data engine that enjoys significantly better scalability compared to previous human annotation or rule-based approaches. It has enabled us to create the largest dataset of its kind to date, comprising 270K image-text-mask triplets covering an unprecedented range of diverse semantic categories and attribute specifications. Based on this data foundation, we further propose a task unification paradigm that centers around referring expression segmentation. It effectively handles a wide range of vision-centric perception tasks, including classification, detection, segmentation, grounding, etc, using a single model without any task-specific heads. Combining these innovations on data and modeling, we present RemoteSAM, a foundation model that establishes new SoTA on several earth observation perception benchmarks, outperforming other foundation models such as Falcon, GeoChat, and LHRS-Bot with significantly higher efficiency. Models and data are publicly available at https://github.com/1e12Leon/RemoteSAM.
>
---
#### [new 071] SeaLion: Semantic Part-Aware Latent Point Diffusion Models for 3D Generation
- **分类: cs.CV**

- **简介: 该论文提出SeaLion模型，通过语义部件感知的潜扩散技术生成带精细分割标签的高质量3D点云，解决现有方法在生成带标签点云及评估指标上的不足。其联合预测潜点噪声与分割标签，并引入部件感知Chamfer距离(p-CD)提升评估，在ShapeNet和IntrA数据集上超越现有方法，支持半监督训练，适用于数据增强与3D编辑。**

- **链接: [http://arxiv.org/pdf/2505.17721v1](http://arxiv.org/pdf/2505.17721v1)**

> **作者:** Dekai Zhu; Yan Di; Stefan Gavranovic; Slobodan Ilic
>
> **摘要:** Denoising diffusion probabilistic models have achieved significant success in point cloud generation, enabling numerous downstream applications, such as generative data augmentation and 3D model editing. However, little attention has been given to generating point clouds with point-wise segmentation labels, as well as to developing evaluation metrics for this task. Therefore, in this paper, we present SeaLion, a novel diffusion model designed to generate high-quality and diverse point clouds with fine-grained segmentation labels. Specifically, we introduce the semantic part-aware latent point diffusion technique, which leverages the intermediate features of the generative models to jointly predict the noise for perturbed latent points and associated part segmentation labels during the denoising process, and subsequently decodes the latent points to point clouds conditioned on part segmentation labels. To effectively evaluate the quality of generated point clouds, we introduce a novel point cloud pairwise distance calculation method named part-aware Chamfer distance (p-CD). This method enables existing metrics, such as 1-NNA, to measure both the local structural quality and inter-part coherence of generated point clouds. Experiments on the large-scale synthetic dataset ShapeNet and real-world medical dataset IntrA demonstrate that SeaLion achieves remarkable performance in generation quality and diversity, outperforming the existing state-of-the-art model, DiffFacto, by 13.33% and 6.52% on 1-NNA (p-CD) across the two datasets. Experimental analysis shows that SeaLion can be trained semi-supervised, thereby reducing the demand for labeling efforts. Lastly, we validate the applicability of SeaLion in generative data augmentation for training segmentation models and the capability of SeaLion to serve as a tool for part-aware 3D shape editing.
>
---
#### [new 072] FDBPL: Faster Distillation-Based Prompt Learning for Region-Aware Vision-Language Models Adaptation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型（VLM）适配下游任务，解决传统提示学习效率低、泛化差的问题。提出FDBPL方法，通过共享软监督上下文加速I/O、引入区域感知正负提示空间及相似-差异互学习机制，提升零样本性能与跨领域泛化能力，同时保持参数高效性，训练速度提升2.2倍。**

- **链接: [http://arxiv.org/pdf/2505.18053v1](http://arxiv.org/pdf/2505.18053v1)**

> **作者:** Zherui Zhang; Jiaxin Wu; Changwei Wang; Rongtao Xu; Longzhao Huang; Wenhao Xu; Wenbo Xu; Li Guo; Shibiao Xu
>
> **摘要:** Prompt learning as a parameter-efficient method that has been widely adopted to adapt Vision-Language Models (VLMs) to downstream tasks. While hard-prompt design requires domain expertise and iterative optimization, soft-prompt methods rely heavily on task-specific hard labels, limiting their generalization to unseen categories. Recent popular distillation-based prompt learning methods improve generalization by exploiting larger teacher VLMs and unsupervised knowledge transfer, yet their repetitive teacher model online inference sacrifices the inherent training efficiency advantage of prompt learning. In this paper, we propose {{\large {\textbf{F}}}}aster {{\large {\textbf{D}}}}istillation-{{\large {\textbf{B}}}}ased {{\large {\textbf{P}}}}rompt {{\large {\textbf{L}}}}earning (\textbf{FDBPL}), which addresses these issues by sharing soft supervision contexts across multiple training stages and implementing accelerated I/O. Furthermore, FDBPL introduces a region-aware prompt learning paradigm with dual positive-negative prompt spaces to fully exploit randomly cropped regions that containing multi-level information. We propose a positive-negative space mutual learning mechanism based on similarity-difference learning, enabling student CLIP models to recognize correct semantics while learning to reject weakly related concepts, thereby improving zero-shot performance. Unlike existing distillation-based prompt learning methods that sacrifice parameter efficiency for generalization, FDBPL maintains dual advantages of parameter efficiency and strong downstream generalization. Comprehensive evaluations across 11 datasets demonstrate superior performance in base-to-new generalization, cross-dataset transfer, and robustness tests, achieving $2.2\times$ faster training speed.
>
---
#### [new 073] Extending Dataset Pruning to Object Detection: A Variance-based Approach
- **分类: cs.CV; cs.LG**

- **简介: 该论文属目标检测数据集剪枝任务，解决其扩展挑战：对象归属、评分策略及图像聚合问题。提出VPS方法，结合IoU与置信度筛选样本，实验显示优于现有方法，强调信息量选择比数据规模更重要。（99字）**

- **链接: [http://arxiv.org/pdf/2505.17245v1](http://arxiv.org/pdf/2505.17245v1)**

> **作者:** Ryota Yagi
>
> **摘要:** Dataset pruning -- selecting a small yet informative subset of training data -- has emerged as a promising strategy for efficient machine learning, offering significant reductions in computational cost and storage compared to alternatives like dataset distillation. While pruning methods have shown strong performance in image classification, their extension to more complex computer vision tasks, particularly object detection, remains relatively underexplored. In this paper, we present the first principled extension of classification pruning techniques to the object detection domain, to the best of our knowledge. We identify and address three key challenges that hinder this transition: the Object-Level Attribution Problem, the Scoring Strategy Problem, and the Image-Level Aggregation Problem. To overcome these, we propose tailored solutions, including a novel scoring method called Variance-based Prediction Score (VPS). VPS leverages both Intersection over Union (IoU) and confidence scores to effectively identify informative training samples specific to detection tasks. Extensive experiments on PASCAL VOC and MS COCO demonstrate that our approach consistently outperforms prior dataset pruning methods in terms of mean Average Precision (mAP). We also show that annotation count and class distribution shift can influence detection performance, but selecting informative examples is a more critical factor than dataset size or balance. Our work bridges dataset pruning and object detection, paving the way for dataset pruning in complex vision tasks.
>
---
#### [new 074] TopoPoint: Enhance Topology Reasoning via Endpoint Detection in Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的拓扑推理任务，旨在解决现有方法因车道端点检测偏差导致的拓扑错误问题。提出TopoPoint框架，通过显式检测车道端点并联合车道与端点推理，采用自注意力机制、图卷积网络及几何匹配算法优化端点定位，显著提升拓扑推理精度（48.8 OLS）和端点检测指标（DET_p达52.6）。**

- **链接: [http://arxiv.org/pdf/2505.17771v1](http://arxiv.org/pdf/2505.17771v1)**

> **作者:** Yanping Fu; Xinyuan Liu; Tianyu Li; Yike Ma; Yucheng Zhang; Feng Dai
>
> **摘要:** Topology reasoning, which unifies perception and structured reasoning, plays a vital role in understanding intersections for autonomous driving. However, its performance heavily relies on the accuracy of lane detection, particularly at connected lane endpoints. Existing methods often suffer from lane endpoints deviation, leading to incorrect topology construction. To address this issue, we propose TopoPoint, a novel framework that explicitly detects lane endpoints and jointly reasons over endpoints and lanes for robust topology reasoning. During training, we independently initialize point and lane query, and proposed Point-Lane Merge Self-Attention to enhance global context sharing through incorporating geometric distances between points and lanes as an attention mask . We further design Point-Lane Graph Convolutional Network to enable mutual feature aggregation between point and lane query. During inference, we introduce Point-Lane Geometry Matching algorithm that computes distances between detected points and lanes to refine lane endpoints, effectively mitigating endpoint deviation. Extensive experiments on the OpenLane-V2 benchmark demonstrate that TopoPoint achieves state-of-the-art performance in topology reasoning (48.8 on OLS). Additionally, we propose DET$_p$ to evaluate endpoint detection, under which our method significantly outperforms existing approaches (52.6 v.s. 45.2 on DET$_p$). The code is released at https://github.com/Franpin/TopoPoint.
>
---
#### [new 075] Pixels to Prognosis: Harmonized Multi-Region CT-Radiomics and Foundation-Model Signatures Across Multicentre NSCLC Data
- **分类: cs.CV**

- **简介: 该论文属于非小细胞肺癌（NSCLC）患者生存预测任务，旨在解决多中心CT影像数据差异及多区域特征整合的挑战。研究提取肿瘤、纵隔淋巴结等区域的放射组学与基础模型特征，通过谐波处理（如ComBat）消除中心偏差，结合临床数据构建预测模型，并采用共识模型提升风险分层效果，最终验证多模态整合对预后评估的优化作用。**

- **链接: [http://arxiv.org/pdf/2505.17893v1](http://arxiv.org/pdf/2505.17893v1)**

> **作者:** Shruti Atul Mali; Zohaib Salahuddin; Danial Khan; Yumeng Zhang; Henry C. Woodruff; Eduardo Ibor-Crespo; Ana Jimenez-Pastor; Luis Marti-Bonmati; Philippe Lambin
>
> **摘要:** Purpose: To evaluate the impact of harmonization and multi-region CT image feature integration on survival prediction in non-small cell lung cancer (NSCLC) patients, using handcrafted radiomics, pretrained foundation model (FM) features, and clinical data from a multicenter dataset. Methods: We analyzed CT scans and clinical data from 876 NSCLC patients (604 training, 272 test) across five centers. Features were extracted from the whole lung, tumor, mediastinal nodes, coronary arteries, and coronary artery calcium (CAC). Handcrafted radiomics and FM deep features were harmonized using ComBat, reconstruction kernel normalization (RKN), and RKN+ComBat. Regularized Cox models predicted overall survival; performance was assessed using the concordance index (C-index), 5-year time-dependent area under the curve (t-AUC), and hazard ratio (HR). SHapley Additive exPlanations (SHAP) values explained feature contributions. A consensus model used agreement across top region of interest (ROI) models to stratify patient risk. Results: TNM staging showed prognostic utility (C-index = 0.67; HR = 2.70; t-AUC = 0.85). The clinical + tumor radiomics model with ComBat achieved a C-index of 0.7552 and t-AUC of 0.8820. FM features (50-voxel cubes) combined with clinical data yielded the highest performance (C-index = 0.7616; t-AUC = 0.8866). An ensemble of all ROIs and FM features reached a C-index of 0.7142 and t-AUC of 0.7885. The consensus model, covering 78% of valid test cases, achieved a t-AUC of 0.92, sensitivity of 97.6%, and specificity of 66.7%. Conclusion: Harmonization and multi-region feature integration improve survival prediction in multicenter NSCLC data. Combining interpretable radiomics, FM features, and consensus modeling enables robust risk stratification across imaging centers.
>
---
#### [new 076] PoseBH: Prototypical Multi-Dataset Training Beyond Human Pose Estimation
- **分类: cs.CV**

- **简介: 论文提出PoseBH框架，针对多数据集训练中骨骼异质性和监督不足问题，通过关键点原型与跨类型自监督方法，提升人体、动物及全身姿态估计的泛化能力，同时保持在标准数据集上的性能，并扩展至手部和身体形状估计。**

- **链接: [http://arxiv.org/pdf/2505.17475v1](http://arxiv.org/pdf/2505.17475v1)**

> **作者:** Uyoung Jeong; Jonathan Freer; Seungryul Baek; Hyung Jin Chang; Kwang In Kim
>
> **备注:** accepted to CVPR 2025
>
> **摘要:** We study multi-dataset training (MDT) for pose estimation, where skeletal heterogeneity presents a unique challenge that existing methods have yet to address. In traditional domains, \eg regression and classification, MDT typically relies on dataset merging or multi-head supervision. However, the diversity of skeleton types and limited cross-dataset supervision complicate integration in pose estimation. To address these challenges, we introduce PoseBH, a new MDT framework that tackles keypoint heterogeneity and limited supervision through two key techniques. First, we propose nonparametric keypoint prototypes that learn within a unified embedding space, enabling seamless integration across skeleton types. Second, we develop a cross-type self-supervision mechanism that aligns keypoint predictions with keypoint embedding prototypes, providing supervision without relying on teacher-student models or additional augmentations. PoseBH substantially improves generalization across whole-body and animal pose datasets, including COCO-WholeBody, AP-10K, and APT-36K, while preserving performance on standard human pose benchmarks (COCO, MPII, and AIC). Furthermore, our learned keypoint embeddings transfer effectively to hand shape estimation (InterHand2.6M) and human body shape estimation (3DPW). The code for PoseBH is available at: https://github.com/uyoung-jeong/PoseBH.
>
---
#### [new 077] RoHyDR: Robust Hybrid Diffusion Recovery for Incomplete Multimodal Emotion Recognition
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对不完整多模态情感识别（IMER）任务，解决数据缺失或损坏导致的性能下降问题。提出RoHyDR框架，通过混合扩散模型与对抗学习，在单模态、多模态、特征及语义层面恢复缺失信息，提升训练稳定性和识别鲁棒性，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.17501v1](http://arxiv.org/pdf/2505.17501v1)**

> **作者:** Yuehan Jin; Xiaoqing Liu; Yiyuan Yang; Zhiwen Yu; Tong Zhang; Kaixiang Yang
>
> **摘要:** Multimodal emotion recognition analyzes emotions by combining data from multiple sources. However, real-world noise or sensor failures often cause missing or corrupted data, creating the Incomplete Multimodal Emotion Recognition (IMER) challenge. In this paper, we propose Robust Hybrid Diffusion Recovery (RoHyDR), a novel framework that performs missing-modality recovery at unimodal, multimodal, feature, and semantic levels. For unimodal representation recovery of missing modalities, RoHyDR exploits a diffusion-based generator to generate distribution-consistent and semantically aligned representations from Gaussian noise, using available modalities as conditioning. For multimodal fusion recovery, we introduce adversarial learning to produce a realistic fused multimodal representation and recover missing semantic content. We further propose a multi-stage optimization strategy that enhances training stability and efficiency. In contrast to previous work, the hybrid diffusion and adversarial learning-based recovery mechanism in RoHyDR allows recovery of missing information in both unimodal representation and multimodal fusion, at both feature and semantic levels, effectively mitigating performance degradation caused by suboptimal optimization. Comprehensive experiments conducted on two widely used multimodal emotion recognition benchmarks demonstrate that our proposed method outperforms state-of-the-art IMER methods, achieving robust recognition performance under various missing-modality scenarios. Our code will be made publicly available upon acceptance.
>
---
#### [new 078] 3D Face Reconstruction Error Decomposed: A Modular Benchmark for Fair and Fast Method Evaluation
- **分类: cs.CV**

- **简介: 该论文属于3D人脸重建评估任务，针对现有误差计算方法步骤固化、缺乏共识的问题，提出模块化基准工具M3DFB，分解并替换误差计算组件（如新增拓扑修正模块），测试16种误差估计算法与10种重建方法，揭示ICP算法评估偏差大（相关性低至0.41），证明非刚性对齐与新方案能提升准确性且更快，推动公平高效的基准评测。**

- **链接: [http://arxiv.org/pdf/2505.18025v1](http://arxiv.org/pdf/2505.18025v1)**

> **作者:** Evangelos Sariyanidi; Claudio Ferrari; Federico Nocentini; Stefano Berretti; Andrea Cavallaro; Birkan Tunc
>
> **备注:** To be published in IEEE International Conference on Automatic Face and Gesture Recognition, 2025
>
> **摘要:** Computing the standard benchmark metric for 3D face reconstruction, namely geometric error, requires a number of steps, such as mesh cropping, rigid alignment, or point correspondence. Current benchmark tools are monolithic (they implement a specific combination of these steps), even though there is no consensus on the best way to measure error. We present a toolkit for a Modularized 3D Face reconstruction Benchmark (M3DFB), where the fundamental components of error computation are segregated and interchangeable, allowing one to quantify the effect of each. Furthermore, we propose a new component, namely correction, and present a computationally efficient approach that penalizes for mesh topology inconsistency. Using this toolkit, we test 16 error estimators with 10 reconstruction methods on two real and two synthetic datasets. Critically, the widely used ICP-based estimator provides the worst benchmarking performance, as it significantly alters the true ranking of the top-5 reconstruction methods. Notably, the correlation of ICP with the true error can be as low as 0.41. Moreover, non-rigid alignment leads to significant improvement (correlation larger than 0.90), highlighting the importance of annotating 3D landmarks on datasets. Finally, the proposed correction scheme, together with non-rigid warping, leads to an accuracy on a par with the best non-rigid ICP-based estimators, but runs an order of magnitude faster. Our open-source codebase is designed for researchers to easily compare alternatives for each component, thus helping accelerating progress in benchmarking for 3D face reconstruction and, furthermore, supporting the improvement of learned reconstruction methods, which depend on accurate error estimation for effective training.
>
---
#### [new 079] Clinical Validation of Deep Learning for Real-Time Tissue Oxygenation Estimation Using Spectral Imaging
- **分类: cs.CV**

- **简介: 该论文属于临床验证任务，旨在通过深度学习改进实时组织氧合估计，解决传统线性方法假设局限及模拟-真实数据差异问题。研究提出基于蒙特卡洛模拟光谱训练FCN和CNN，并采用领域对抗训练缩小领域差距，结果表明模型与临床乳酸测量相关性更高，优于传统方法。（99字）**

- **链接: [http://arxiv.org/pdf/2505.18010v1](http://arxiv.org/pdf/2505.18010v1)**

> **作者:** Jens De Winne; Siri Willems; Siri Luthman; Danilo Babin; Hiep Luong; Wim Ceelen
>
> **备注:** Provisionally accepted to the MICCAI 2025 conference
>
> **摘要:** Accurate, real-time monitoring of tissue ischemia is crucial to understand tissue health and guide surgery. Spectral imaging shows great potential for contactless and intraoperative monitoring of tissue oxygenation. Due to the difficulty of obtaining direct reference oxygenation values, conventional methods are based on linear unmixing techniques. These are prone to assumptions and these linear relations may not always hold in practice. In this work, we present deep learning approaches for real-time tissue oxygenation estimation using Monte-Carlo simulated spectra. We train a fully connected neural network (FCN) and a convolutional neural network (CNN) for this task and propose a domain-adversarial training approach to bridge the gap between simulated and real clinical spectral data. Results demonstrate that these deep learning models achieve a higher correlation with capillary lactate measurements, a well-known marker of hypoxia, obtained during spectral imaging in surgery, compared to traditional linear unmixing. Notably, domain-adversarial training effectively reduces the domain gap, optimizing performance in real clinical settings.
>
---
#### [new 080] DiffusionReward: Enhancing Blind Face Restoration through Reward Feedback Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于盲面部修复任务，针对扩散模型生成细节不真实、身份一致性差的问题，提出DiffusionReward框架。通过训练Face Reward Model（FRM）提供反馈信号，结合梯度优化、正则化及结构约束，动态引导模型生成更真实且保持身份的面部图像，实验显示显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.17910v1](http://arxiv.org/pdf/2505.17910v1)**

> **作者:** Bin Wu; Wei Wang; Yahui Liu; Zixiang Li; Yao Zhao
>
> **备注:** 22 pages, 13 figures, 5 tables
>
> **摘要:** Reward Feedback Learning (ReFL) has recently shown great potential in aligning model outputs with human preferences across various generative tasks. In this work, we introduce a ReFL framework, named DiffusionReward, to the Blind Face Restoration task for the first time. DiffusionReward effectively overcomes the limitations of diffusion-based methods, which often fail to generate realistic facial details and exhibit poor identity consistency. The core of our framework is the Face Reward Model (FRM), which is trained using carefully annotated data. It provides feedback signals that play a pivotal role in steering the optimization process of the restoration network. In particular, our ReFL framework incorporates a gradient flow into the denoising process of off-the-shelf face restoration methods to guide the update of model parameters. The guiding gradient is collaboratively determined by three aspects: (i) the FRM to ensure the perceptual quality of the restored faces; (ii) a regularization term that functions as a safeguard to preserve generative diversity; and (iii) a structural consistency constraint to maintain facial fidelity. Furthermore, the FRM undergoes dynamic optimization throughout the process. It not only ensures that the restoration network stays precisely aligned with the real face manifold, but also effectively prevents reward hacking. Experiments on synthetic and wild datasets demonstrate that our method outperforms state-of-the-art methods, significantly improving identity consistency and facial details. The source codes, data, and models are available at: https://github.com/01NeuralNinja/DiffusionReward.
>
---
#### [new 081] An Attention Infused Deep Learning System with Grad-CAM Visualization for Early Screening of Glaucoma
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出结合CNN与Vision Transformer的混合模型，融入交叉注意力机制，用于青光眼早期筛查。利用ACRIMA和Drishti数据集，通过Grad-CAM可视化提升可解释性，旨在提高检测精度并解决传统方法局限。**

- **链接: [http://arxiv.org/pdf/2505.17808v1](http://arxiv.org/pdf/2505.17808v1)**

> **作者:** Ramanathan Swaminathan
>
> **备注:** 6 pages in general IEEE format, 8 figures, 4 tables, pdflatex
>
> **摘要:** This research work reveals the eye opening wisdom of the hybrid labyrinthine deep learning models synergy born out of combining a trailblazing convolutional neural network with a disruptive Vision Transformer, both intertwined together with a radical Cross Attention module. Here, two high yielding datasets for artificial intelligence models in detecting glaucoma, namely ACRIMA and Drishti, are utilized.
>
---
#### [new 082] Generative Data Augmentation for Object Point Cloud Segmentation
- **分类: cs.CV**

- **简介: 该论文属于点云分割任务，旨在解决传统数据增强数据多样性不足及生成模型缺乏语义标签的问题。工作包括扩展3D扩散模型Lion为部分感知模型，生成带分割掩码的高质量点云，并提出三步生成数据增强（GDA）流程，结合生成样本与伪标签过滤方法提升模型性能，实验显示优于传统及半/自监督方法。**

- **链接: [http://arxiv.org/pdf/2505.17783v1](http://arxiv.org/pdf/2505.17783v1)**

> **作者:** Dekai Zhu; Stefan Gavranovic; Flavien Boussuge; Benjamin Busam; Slobodan Ilic
>
> **摘要:** Data augmentation is widely used to train deep learning models to address data scarcity. However, traditional data augmentation (TDA) typically relies on simple geometric transformation, such as random rotation and rescaling, resulting in minimal data diversity enrichment and limited model performance improvement. State-of-the-art generative models for 3D shape generation rely on the denoising diffusion probabilistic models and manage to generate realistic novel point clouds for 3D content creation and manipulation. Nevertheless, the generated 3D shapes lack associated point-wise semantic labels, restricting their usage in enlarging the training data for point cloud segmentation tasks. To bridge the gap between data augmentation techniques and the advanced diffusion models, we extend the state-of-the-art 3D diffusion model, Lion, to a part-aware generative model that can generate high-quality point clouds conditioned on given segmentation masks. Leveraging the novel generative model, we introduce a 3-step generative data augmentation (GDA) pipeline for point cloud segmentation training. Our GDA approach requires only a small amount of labeled samples but enriches the training data with generated variants and pseudo-labeled samples, which are validated by a novel diffusion-based pseudo-label filtering method. Extensive experiments on two large-scale synthetic datasets and a real-world medical dataset demonstrate that our GDA method outperforms TDA approach and related semi-supervised and self-supervised methods.
>
---
#### [new 083] Synthetic History: Evaluating Visual Representations of the Past in Diffusion Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文评估文本到图像扩散模型的历史视觉表现，解决其历史准确性不足的问题。通过构建含3万合成图像的HistVis数据集，从风格关联、历史一致性和人口统计三方面评估模型，发现存在刻板印象、时代错误及人口偏差，提出可扩展的评估方法以改进模型历史准确性。**

- **链接: [http://arxiv.org/pdf/2505.17064v1](http://arxiv.org/pdf/2505.17064v1)**

> **作者:** Maria-Teresa De Rosa Palmini; Eva Cetinic
>
> **摘要:** As Text-to-Image (TTI) diffusion models become increasingly influential in content creation, growing attention is being directed toward their societal and cultural implications. While prior research has primarily examined demographic and cultural biases, the ability of these models to accurately represent historical contexts remains largely underexplored. In this work, we present a systematic and reproducible methodology for evaluating how TTI systems depict different historical periods. For this purpose, we introduce the HistVis dataset, a curated collection of 30,000 synthetic images generated by three state-of-the-art diffusion models using carefully designed prompts depicting universal human activities across different historical periods. We evaluate generated imagery across three key aspects: (1) Implicit Stylistic Associations: examining default visual styles associated with specific eras; (2) Historical Consistency: identifying anachronisms such as modern artifacts in pre-modern contexts; and (3) Demographic Representation: comparing generated racial and gender distributions against historically plausible baselines. Our findings reveal systematic inaccuracies in historically themed generated imagery, as TTI models frequently stereotype past eras by incorporating unstated stylistic cues, introduce anachronisms, and fail to reflect plausible demographic patterns. By offering a scalable methodology and benchmark for assessing historical representation in generated imagery, this work provides an initial step toward building more historically accurate and culturally aligned TTI models.
>
---
#### [new 084] REN: Fast and Efficient Region Encodings from Patch-Based Image Encoders
- **分类: cs.CV**

- **简介: 该论文提出REN模型，旨在通过点提示快速生成区域图像表示。针对现有方法结合分割模型与补丁编码器计算成本高的问题，REN采用轻量模块直接生成区域token，无需分割步骤，实现60倍加速与35倍内存减少，提升表示质量。实验显示其在分割、检索任务中优于原有编码器及SAM基线，并在Ego4D等数据集达SOTA。**

- **链接: [http://arxiv.org/pdf/2505.18153v1](http://arxiv.org/pdf/2505.18153v1)**

> **作者:** Savya Khosla; Sethuraman TV; Barnett Lee; Alexander Schwing; Derek Hoiem
>
> **摘要:** We introduce the Region Encoder Network (REN), a fast and effective model for generating region-based image representations using point prompts. Recent methods combine class-agnostic segmenters (e.g., SAM) with patch-based image encoders (e.g., DINO) to produce compact and effective region representations, but they suffer from high computational cost due to the segmentation step. REN bypasses this bottleneck using a lightweight module that directly generates region tokens, enabling 60x faster token generation with 35x less memory, while also improving token quality. It uses a few cross-attention blocks that take point prompts as queries and features from a patch-based image encoder as keys and values to produce region tokens that correspond to the prompted objects. We train REN with three popular encoders-DINO, DINOv2, and OpenCLIP-and show that it can be extended to other encoders without dedicated training. We evaluate REN on semantic segmentation and retrieval tasks, where it consistently outperforms the original encoders in both performance and compactness, and matches or exceeds SAM-based region methods while being significantly faster. Notably, REN achieves state-of-the-art results on the challenging Ego4D VQ2D benchmark and outperforms proprietary LMMs on Visual Haystacks' single-needle challenge. Code and models are available at: https://github.com/savya08/REN.
>
---
#### [new 085] CHAOS: Chart Analysis with Outlier Samples
- **分类: cs.CV; cs.CL**

- **简介: 论文提出CHAOS基准，评估多模态大语言模型处理带噪声图表的鲁棒性，解决其在异常图表中表现不佳的问题。设计5类文本、10类视觉扰动及三难度等级，测试13种模型，分析ChartQA和Chart-to-Text任务，揭示模型弱点并指导研究。（99字）**

- **链接: [http://arxiv.org/pdf/2505.17235v1](http://arxiv.org/pdf/2505.17235v1)**

> **作者:** Omar Moured; Yufan Chen; Ruiping Liu; Simon Reiß; Philip Torr; Jiaming Zhang; Rainer Stiefelhagen
>
> **备注:** Data and code are publicly available at: http://huggingface.co/datasets/omoured/CHAOS
>
> **摘要:** Charts play a critical role in data analysis and visualization, yet real-world applications often present charts with challenging or noisy features. However, "outlier charts" pose a substantial challenge even for Multimodal Large Language Models (MLLMs), which can struggle to interpret perturbed charts. In this work, we introduce CHAOS (CHart Analysis with Outlier Samples), a robustness benchmark to systematically evaluate MLLMs against chart perturbations. CHAOS encompasses five types of textual and ten types of visual perturbations, each presented at three levels of severity (easy, mid, hard) inspired by the study result of human evaluation. The benchmark includes 13 state-of-the-art MLLMs divided into three groups (i.e., general-, document-, and chart-specific models) according to the training scope and data. Comprehensive analysis involves two downstream tasks (ChartQA and Chart-to-Text). Extensive experiments and case studies highlight critical insights into robustness of models across chart perturbations, aiming to guide future research in chart understanding domain. Data and code are publicly available at: http://huggingface.co/datasets/omoured/CHAOS.
>
---
#### [new 086] Repurposing Marigold for Zero-Shot Metric Depth Estimation via Defocus Blur Cues
- **分类: cs.CV**

- **简介: 论文针对零样本单目度量深度估计任务，解决现有方法在分布外数据性能下降的问题。通过在推理阶段利用散焦模糊线索优化预训练扩散模型Marigold的参数和噪声潜空间，实现无需训练的度量深度预测，在自建真实数据集上取得提升。**

- **链接: [http://arxiv.org/pdf/2505.17358v1](http://arxiv.org/pdf/2505.17358v1)**

> **作者:** Chinmay Talegaonkar; Nikhil Gandudi Suresh; Zachary Novack; Yash Belhe; Priyanka Nagasamudra; Nicholas Antipa
>
> **摘要:** Recent monocular metric depth estimation (MMDE) methods have made notable progress towards zero-shot generalization. However, they still exhibit a significant performance drop on out-of-distribution datasets. We address this limitation by injecting defocus blur cues at inference time into Marigold, a \textit{pre-trained} diffusion model for zero-shot, scale-invariant monocular depth estimation (MDE). Our method effectively turns Marigold into a metric depth predictor in a training-free manner. To incorporate defocus cues, we capture two images with a small and a large aperture from the same viewpoint. To recover metric depth, we then optimize the metric depth scaling parameters and the noise latents of Marigold at inference time using gradients from a loss function based on the defocus-blur image formation model. We compare our method against existing state-of-the-art zero-shot MMDE methods on a self-collected real dataset, showing quantitative and qualitative improvements.
>
---
#### [new 087] SplatCo: Structure-View Collaborative Gaussian Splatting for Detail-Preserving Rendering of Large-Scale Unbounded Scenes
- **分类: cs.CV**

- **简介: 该论文属于大规模无界场景高保真渲染任务，解决现有方法在细节保留与全局一致性中的不足。提出SplatCo框架：1）通过跨结构模块融合全局tri-plane与局部网格特征，实现层次化细节补偿；2）采用跨视图训练策略同步多视角优化，提升几何纹理重建质量。实验显示其在多个数据集上超越现有方法（PSNR+1-2dB，SSIM+0.1-0.2）。**

- **链接: [http://arxiv.org/pdf/2505.17951v1](http://arxiv.org/pdf/2505.17951v1)**

> **作者:** Haihong Xiao; Jianan Zou; Yuxin Zhou; Ying He; Wenxiong Kang
>
> **摘要:** We present SplatCo, a structure-view collaborative Gaussian splatting framework for high-fidelity rendering of complex outdoor environments. SplatCo builds upon two novel components: (1) a cross-structure collaboration module that combines global tri-plane representations, which capture coarse scene layouts, with local context grid features that represent fine surface details. This fusion is achieved through a novel hierarchical compensation strategy, ensuring both global consistency and local detail preservation; and (2) a cross-view assisted training strategy that enhances multi-view consistency by synchronizing gradient updates across viewpoints, applying visibility-aware densification, and pruning overfitted or inaccurate Gaussians based on structural consistency. Through joint optimization of structural representation and multi-view coherence, SplatCo effectively reconstructs fine-grained geometric structures and complex textures in large-scale scenes. Comprehensive evaluations on 13 diverse large-scale scenes, including Mill19, MatrixCity, Tanks & Temples, WHU, and custom aerial captures, demonstrate that SplatCo consistently achieves higher reconstruction quality than state-of-the-art methods, with PSNR improvements of 1-2 dB and SSIM gains of 0.1 to 0.2. These results establish a new benchmark for high-fidelity rendering of large-scale unbounded scenes. Code and additional information are available at https://github.com/SCUT-BIP-Lab/SplatCo.
>
---
#### [new 088] Wildfire Detection Using Vision Transformer with the Wildfire Dataset
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉中的野火检测任务，旨在利用Vision Transformer（ViT）模型提升早期火灾识别精度，解决数据不足、环境干扰及模型优化问题。研究使用10.74GB高分辨率图像数据集，通过标准化预处理训练ViT模型，以提高野火检测准确性和实时性。**

- **链接: [http://arxiv.org/pdf/2505.17395v1](http://arxiv.org/pdf/2505.17395v1)**

> **作者:** Gowtham Raj Vuppari; Navarun Gupta; Ahmed El-Sayed; Xingguo Xiong
>
> **备注:** Published at ASEE NE 2025
>
> **摘要:** The critical need for sophisticated detection techniques has been highlighted by the rising frequency and intensity of wildfires in the US, especially in California. In 2023, wildfires caused 130 deaths nationwide, the highest since 1990. In January 2025, Los Angeles wildfires which included the Palisades and Eaton fires burnt approximately 40,000 acres and 12,000 buildings, and caused loss of human lives. The devastation underscores the urgent need for effective detection and prevention strategies. Deep learning models, such as Vision Transformers (ViTs), can enhance early detection by processing complex image data with high accuracy. However, wildfire detection faces challenges, including the availability of high-quality, real-time data. Wildfires often occur in remote areas with limited sensor coverage, and environmental factors like smoke and cloud cover can hinder detection. Additionally, training deep learning models is computationally expensive, and issues like false positives/negatives and scaling remain concerns. Integrating detection systems with real-time alert mechanisms also poses difficulties. In this work, we used the wildfire dataset consisting of 10.74 GB high-resolution images categorized into 'fire' and 'nofire' classes is used for training the ViT model. To prepare the data, images are resized to 224 x 224 pixels, converted into tensor format, and normalized using ImageNet statistics.
>
---
#### [new 089] TextFlux: An OCR-Free DiT Model for High-Fidelity Multilingual Scene Text Synthesis
- **分类: cs.CV**

- **简介: 该论文提出TextFlux，一种无OCR的DiT模型，用于高保真多语言场景文本合成。解决现有方法依赖视觉模块和大量标注数据的问题，通过简化架构与扩散模型的上下文推理，实现低资源多语言支持、少数据训练及灵活多行控制，提升合成质量和效率。**

- **链接: [http://arxiv.org/pdf/2505.17778v1](http://arxiv.org/pdf/2505.17778v1)**

> **作者:** Yu Xie; Jielei Zhang; Pengyu Chen; Ziyue Wang; Weihang Wang; Longwen Gao; Peiyi Li; Huyang Sun; Qiang Zhang; Qian Qiao; Jiaqing Fan; Zhouhui Lian
>
> **摘要:** Diffusion-based scene text synthesis has progressed rapidly, yet existing methods commonly rely on additional visual conditioning modules and require large-scale annotated data to support multilingual generation. In this work, we revisit the necessity of complex auxiliary modules and further explore an approach that simultaneously ensures glyph accuracy and achieves high-fidelity scene integration, by leveraging diffusion models' inherent capabilities for contextual reasoning. To this end, we introduce TextFlux, a DiT-based framework that enables multilingual scene text synthesis. The advantages of TextFlux can be summarized as follows: (1) OCR-free model architecture. TextFlux eliminates the need for OCR encoders (additional visual conditioning modules) that are specifically used to extract visual text-related features. (2) Strong multilingual scalability. TextFlux is effective in low-resource multilingual settings, and achieves strong performance in newly added languages with fewer than 1,000 samples. (3) Streamlined training setup. TextFlux is trained with only 1% of the training data required by competing methods. (4) Controllable multi-line text generation. TextFlux offers flexible multi-line synthesis with precise line-level control, outperforming methods restricted to single-line or rigid layouts. Extensive experiments and visualizations demonstrate that TextFlux outperforms previous methods in both qualitative and quantitative evaluations.
>
---
#### [new 090] VLM Models and Automated Grading of Atopic Dermatitis
- **分类: cs.CV**

- **简介: 该论文研究利用视觉语言模型（VLM）自动化评估特应性皮炎（AD）严重程度。针对皮肤科医生难以精准分级的临床难题，实验测试了七种VLM模型在AD图像分级任务中的表现，探索多模态模型在可解释性医疗影像分析中的潜力。**

- **链接: [http://arxiv.org/pdf/2505.17835v1](http://arxiv.org/pdf/2505.17835v1)**

> **作者:** Marc Lalonde; Hamed Ghodrati
>
> **备注:** 10 pages
>
> **摘要:** The task of grading atopic dermatitis (or AD, a form of eczema) from patient images is difficult even for trained dermatologists. Research on automating this task has progressed in recent years with the development of deep learning solutions; however, the rapid evolution of multimodal models and more specifically vision-language models (VLMs) opens the door to new possibilities in terms of explainable assessment of medical images, including dermatology. This report describes experiments carried out to evaluate the ability of seven VLMs to assess the severity of AD on a set of test images.
>
---
#### [new 091] Instructify: Demystifying Metadata to Visual Instruction Tuning Data Conversion
- **分类: cs.CV**

- **简介: 该论文提出Instructify框架，解决现有视觉指令调优（VisIT）数据生成依赖闭源模型、成本高且难以扩展的问题。通过开放方法将图像元数据转化为VisIT指令，提升数据质量与可扩展性，支持开源模型并分析关键因素，代码已开源。**

- **链接: [http://arxiv.org/pdf/2505.18115v1](http://arxiv.org/pdf/2505.18115v1)**

> **作者:** Jacob Hansen; Wei Lin; Junmo Kang; Muhammad Jehanzeb Mirza; Hongyin Luo; Rogerio Feris; Alan Ritter; James Glass; Leonid Karlinsky
>
> **摘要:** Visual Instruction Tuning (VisIT) data, commonly available as human-assistant conversations with images interleaved in the human turns, are currently the most widespread vehicle for aligning strong LLMs to understand visual inputs, converting them to strong LMMs. While many VisIT datasets are available, most are constructed using ad-hoc techniques developed independently by different groups. They are often poorly documented, lack reproducible code, and rely on paid, closed-source model APIs such as GPT-4, Gemini, or Claude to convert image metadata (labels) into VisIT instructions. This leads to high costs and makes it challenging to scale, enhance quality, or generate VisIT data for new datasets. In this work, we address these challenges and propose an open and unified recipe and approach,~\textbf{\method}, for converting available metadata to VisIT instructions using open LLMs. Our multi-stage \method features an efficient framework for metadata grouping, quality control, data and prompt organization, and conversation sampling. We show that our approach can reproduce or enhance the data quality of available VisIT datasets when applied to the same image data and metadata sources, improving GPT-4 generated VisIT instructions by ~3\% on average and up to 12\% on individual benchmarks using open models, such as Gemma 2 27B and LLaMa 3.1 70B. Additionally, our approach enables effective performance scaling - both in quantity and quality - by enhancing the resulting LMM performance across a wide range of benchmarks. We also analyze the impact of various factors, including conversation format, base model selection, and resampling strategies. Our code, which supports the reproduction of equal or higher-quality VisIT datasets and facilities future metadata-to-VisIT data conversion for niche domains, is released at https://github.com/jacob-hansen/Instructify.
>
---
#### [new 092] Learning Generalized and Flexible Trajectory Models from Omni-Semantic Supervision
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于轨迹检索任务，旨在解决现有方法在大规模数据处理、条件查询支持不足及依赖轨迹相似度的问题。提出OmniTraj框架，整合原始轨迹、拓扑、路段和区域四种语义模态，通过专用编码器融合到共享空间，支持灵活多模态查询，实验验证其有效性和效率。**

- **链接: [http://arxiv.org/pdf/2505.17437v1](http://arxiv.org/pdf/2505.17437v1)**

> **作者:** Yuanshao Zhu; James Jianqiao Yu; Xiangyu Zhao; Xiao Han; Qidong Liu; Xuetao Wei; Yuxuan Liang
>
> **备注:** Accepted as a full paper by KDD'25 - Research Track
>
> **摘要:** The widespread adoption of mobile devices and data collection technologies has led to an exponential increase in trajectory data, presenting significant challenges in spatio-temporal data mining, particularly for efficient and accurate trajectory retrieval. However, existing methods for trajectory retrieval face notable limitations, including inefficiencies in large-scale data, lack of support for condition-based queries, and reliance on trajectory similarity measures. To address the above challenges, we propose OmniTraj, a generalized and flexible omni-semantic trajectory retrieval framework that integrates four complementary modalities or semantics -- raw trajectories, topology, road segments, and regions -- into a unified system. Unlike traditional approaches that are limited to computing and processing trajectories as a single modality, OmniTraj designs dedicated encoders for each modality, which are embedded and fused into a shared representation space. This design enables OmniTraj to support accurate and flexible queries based on any individual modality or combination thereof, overcoming the rigidity of traditional similarity-based methods. Extensive experiments on two real-world datasets demonstrate the effectiveness of OmniTraj in handling large-scale data, providing flexible, multi-modality queries, and supporting downstream tasks and applications.
>
---
#### [new 093] Semi-Supervised Medical Image Segmentation via Dual Networks
- **分类: cs.CV**

- **简介: 论文提出基于双网络的半监督3D医学图像分割方法，解决标注数据不足及伪标签噪声问题。采用双网络架构优化上下文利用和伪标签可靠性，结合自监督对比学习提升表征能力，减少预测不确定性。**

- **链接: [http://arxiv.org/pdf/2505.17690v1](http://arxiv.org/pdf/2505.17690v1)**

> **作者:** Yunyao Lu; Yihang Wu; Reem Kateb; Ahmad Chaddad
>
> **备注:** Accepted in ISBI2025
>
> **摘要:** Traditional supervised medical image segmentation models require large amounts of labeled data for training; however, obtaining such large-scale labeled datasets in the real world is extremely challenging. Recent semi-supervised segmentation models also suffer from noisy pseudo-label issue and limited supervision in feature space. To solve these challenges, we propose an innovative semi-supervised 3D medical image segmentation method to reduce the dependency on large, expert-labeled datasets. Furthermore, we introduce a dual-network architecture to address the limitations of existing methods in using contextual information and generating reliable pseudo-labels. In addition, a self-supervised contrastive learning strategy is used to enhance the representation of the network and reduce prediction uncertainty by distinguishing between reliable and unreliable predictions. Experiments on clinical magnetic resonance imaging demonstrate that our approach outperforms state-of-the-art techniques. Our code is available at https://github.com/AIPMLab/Semi-supervised-Segmentation.
>
---
#### [new 094] U2-BENCH: Benchmarking Large Vision-Language Models on Ultrasound Understanding
- **分类: cs.CV; cs.LG**

- **简介: 该论文提出U2-BENCH，首个全面评估大视觉语言模型（LVLMs）在超声图像理解的基准测试。针对超声图像解读困难及模型在空间推理、临床文本生成上的不足，构建含7241例病例、覆盖15个解剖区域和8类临床任务的评测集，评估20种LVLMs，发现其分类任务表现佳但空间推理与临床文本生成存在挑战。**

- **链接: [http://arxiv.org/pdf/2505.17779v1](http://arxiv.org/pdf/2505.17779v1)**

> **作者:** Anjie Le; Henan Liu; Yue Wang; Zhenyu Liu; Rongkun Zhu; Taohan Weng; Jinze Yu; Boyang Wang; Yalun Wu; Kaiwen Yan; Quanlin Sun; Meirui Jiang; Jialun Pei; Siya Liu; Haoyun Zheng; Zhoujun Li; Alison Noble; Jacques Souquet; Xiaoqing Guo; Manxi Lin; Hongcheng Guo
>
> **摘要:** Ultrasound is a widely-used imaging modality critical to global healthcare, yet its interpretation remains challenging due to its varying image quality on operators, noises, and anatomical structures. Although large vision-language models (LVLMs) have demonstrated impressive multimodal capabilities across natural and medical domains, their performance on ultrasound remains largely unexplored. We introduce U2-BENCH, the first comprehensive benchmark to evaluate LVLMs on ultrasound understanding across classification, detection, regression, and text generation tasks. U2-BENCH aggregates 7,241 cases spanning 15 anatomical regions and defines 8 clinically inspired tasks, such as diagnosis, view recognition, lesion localization, clinical value estimation, and report generation, across 50 ultrasound application scenarios. We evaluate 20 state-of-the-art LVLMs, both open- and closed-source, general-purpose and medical-specific. Our results reveal strong performance on image-level classification, but persistent challenges in spatial reasoning and clinical language generation. U2-BENCH establishes a rigorous and unified testbed to assess and accelerate LVLM research in the uniquely multimodal domain of medical ultrasound imaging.
>
---
#### [new 095] VIBE: Video-to-Text Information Bottleneck Evaluation for TL;DR
- **分类: cs.CV; cs.HC; cs.IT; math.IT**

- **简介: 该论文提出VIBE方法，用于视频到文本摘要的自动化评估。针对现有视觉语言模型输出冗长低效、评估依赖人工且忽视任务实用性的痛点，通过"grounding"（视觉内容匹配度）和"utility"（任务信息量）双指标评分，筛选优质摘要。实验显示其提升任务准确率61.23%，缩短响应时间75.77%。**

- **链接: [http://arxiv.org/pdf/2505.17423v1](http://arxiv.org/pdf/2505.17423v1)**

> **作者:** Shenghui Chen; Po-han Li; Sandeep Chichali; Ufuk Topcu
>
> **摘要:** Many decision-making tasks, where both accuracy and efficiency matter, still require human supervision. For example, tasks like traffic officers reviewing hour-long dashcam footage or researchers screening conference videos can benefit from concise summaries that reduce cognitive load and save time. Yet current vision-language models (VLMs) often produce verbose, redundant outputs that hinder task performance. Existing video caption evaluation depends on costly human annotations and overlooks the summaries' utility in downstream tasks. We address these gaps with Video-to-text Information Bottleneck Evaluation (VIBE), an annotation-free method that scores VLM outputs using two metrics: grounding (how well the summary aligns with visual content) and utility (how informative it is for the task). VIBE selects from randomly sampled VLM outputs by ranking them according to the two scores to support effective human decision-making. Human studies on LearningPaper24, SUTD-TrafficQA, and LongVideoBench show that summaries selected by VIBE consistently improve performance-boosting task accuracy by up to 61.23% and reducing response time by 75.77% compared to naive VLM summaries or raw video.
>
---
#### [new 096] CAMME: Adaptive Deepfake Image Detection with Multi-Modal Cross-Attention
- **分类: cs.CV; F.2.2; I.2.7**

- **简介: 该论文属于深度伪造图像检测任务，旨在解决现有方法在未知生成模型上性能下降的问题。提出CAMME框架，通过多头交叉注意力融合视觉、文本及频域特征，提升跨生成架构的泛化能力，实验显示其在自然场景和人脸检测中显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.18035v1](http://arxiv.org/pdf/2505.18035v1)**

> **作者:** Naseem Khan; Tuan Nguyen; Amine Bermak; Issa Khalil
>
> **备注:** 20 pages, 8 figures, 12 Tables
>
> **摘要:** The proliferation of sophisticated AI-generated deepfakes poses critical challenges for digital media authentication and societal security. While existing detection methods perform well within specific generative domains, they exhibit significant performance degradation when applied to manipulations produced by unseen architectures--a fundamental limitation as generative technologies rapidly evolve. We propose CAMME (Cross-Attention Multi-Modal Embeddings), a framework that dynamically integrates visual, textual, and frequency-domain features through a multi-head cross-attention mechanism to establish robust cross-domain generalization. Extensive experiments demonstrate CAMME's superiority over state-of-the-art methods, yielding improvements of 12.56% on natural scenes and 13.25% on facial deepfakes. The framework demonstrates exceptional resilience, maintaining (over 91%) accuracy under natural image perturbations and achieving 89.01% and 96.14% accuracy against PGD and FGSM adversarial attacks, respectively. Our findings validate that integrating complementary modalities through cross-attention enables more effective decision boundary realignment for reliable deepfake detection across heterogeneous generative architectures.
>
---
#### [new 097] Adapting SAM 2 for Visual Object Tracking: 1st Place Solution for MMVPR Challenge Multi-Modal Tracking
- **分类: cs.CV**

- **简介: 该论文属于视觉目标跟踪（VOT）任务，针对Segment Anything Model 2（SAM2）在多模态跟踪中的适应性问题，提出优化方法提升其跟踪性能。通过改进模型结构并结合特定技术，在2024 ICPR多模态跟踪挑战赛获AUC 89.4分冠军，验证了方案有效性。**

- **链接: [http://arxiv.org/pdf/2505.18111v1](http://arxiv.org/pdf/2505.18111v1)**

> **作者:** Cheng-Yen Yang; Hsiang-Wei Huang; Pyong-Kun Kim; Chien-Kai Kuo; Jui-Wei Chang; Kwang-Ju Kim; Chung-I Huang; Jenq-Neng Hwang
>
> **备注:** Accepted by ICPR Multi-Modal Visual Pattern Recognition Workshop
>
> **摘要:** We present an effective approach for adapting the Segment Anything Model 2 (SAM2) to the Visual Object Tracking (VOT) task. Our method leverages the powerful pre-trained capabilities of SAM2 and incorporates several key techniques to enhance its performance in VOT applications. By combining SAM2 with our proposed optimizations, we achieved a first place AUC score of 89.4 on the 2024 ICPR Multi-modal Object Tracking challenge, demonstrating the effectiveness of our approach. This paper details our methodology, the specific enhancements made to SAM2, and a comprehensive analysis of our results in the context of VOT solutions along with the multi-modality aspect of the dataset.
>
---
#### [new 098] Instruct2See: Learning to Remove Any Obstructions Across Distributions
- **分类: cs.CV**

- **简介: 该论文属于图像去遮挡任务，旨在解决现有方法仅针对特定遮挡且难以泛化的问题。提出Instruct2See框架，通过多模态提示（视觉语义+文本指令）与动态软掩码机制，实现零样本跨分布遮挡去除，提升对已知/未知遮挡的泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.17649v1](http://arxiv.org/pdf/2505.17649v1)**

> **作者:** Junhang Li; Yu Guo; Chuhua Xian; Shengfeng He
>
> **摘要:** Images are often obstructed by various obstacles due to capture limitations, hindering the observation of objects of interest. Most existing methods address occlusions from specific elements like fences or raindrops, but are constrained by the wide range of real-world obstructions, making comprehensive data collection impractical. To overcome these challenges, we propose Instruct2See, a novel zero-shot framework capable of handling both seen and unseen obstacles. The core idea of our approach is to unify obstruction removal by treating it as a soft-hard mask restoration problem, where any obstruction can be represented using multi-modal prompts, such as visual semantics and textual instructions, processed through a cross-attention unit to enhance contextual understanding and improve mode control. Additionally, a tunable mask adapter allows for dynamic soft masking, enabling real-time adjustment of inaccurate masks. Extensive experiments on both in-distribution and out-of-distribution obstacles show that Instruct2See consistently achieves strong performance and generalization in obstruction removal, regardless of whether the obstacles were present during the training phase. Code and dataset are available at https://jhscut.github.io/Instruct2See.
>
---
#### [new 099] Model Already Knows the Best Noise: Bayesian Active Noise Selection via Attention in Video Diffusion Model
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成优化任务，旨在解决初始噪声选择对生成视频质量及时间连贯性的影响问题。提出ANSE框架，利用注意力机制量化模型内部不确定性（BANSA），通过熵不一致衡量噪声种子优劣，结合伯努利掩码加速推理，提升视频质量与时序一致性，仅增加8%-13%推理时间。**

- **链接: [http://arxiv.org/pdf/2505.17561v1](http://arxiv.org/pdf/2505.17561v1)**

> **作者:** Kwanyoung Kim; Sanghyun Kim
>
> **备注:** 19 pages, 10 figures
>
> **摘要:** The choice of initial noise significantly affects the quality and prompt alignment of video diffusion models, where different noise seeds for the same prompt can lead to drastically different generations. While recent methods rely on externally designed priors such as frequency filters or inter-frame smoothing, they often overlook internal model signals that indicate which noise seeds are inherently preferable. To address this, we propose ANSE (Active Noise Selection for Generation), a model-aware framework that selects high-quality noise seeds by quantifying attention-based uncertainty. At its core is BANSA (Bayesian Active Noise Selection via Attention), an acquisition function that measures entropy disagreement across multiple stochastic attention samples to estimate model confidence and consistency. For efficient inference-time deployment, we introduce a Bernoulli-masked approximation of BANSA that enables score estimation using a single diffusion step and a subset of attention layers. Experiments on CogVideoX-2B and 5B demonstrate that ANSE improves video quality and temporal coherence with only an 8% and 13% increase in inference time, respectively, providing a principled and generalizable approach to noise selection in video diffusion. See our project page: https://anse-project.github.io/anse-project/
>
---
#### [new 100] Enhancing Fourier-based Doppler Resolution with Diffusion Models
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于雷达信号处理任务，旨在提升多普勒分辨率以更好检测慢速目标。针对传统FFT受硬件限制导致分辨率不足的问题，提出结合零填充FFT与扩散模型的生成式神经网络，优化雷达成像中的range-Doppler图，实现对近距离目标的有效分离。**

- **链接: [http://arxiv.org/pdf/2505.17567v1](http://arxiv.org/pdf/2505.17567v1)**

> **作者:** Denisa Qosja; Kilian Barth; Simon Wagner
>
> **备注:** Published at International Radar Symposium (IRS) 2025
>
> **摘要:** In radar systems, high resolution in the Doppler dimension is important for detecting slow-moving targets as it allows for more distinct separation between these targets and clutter, or stationary objects. However, achieving sufficient resolution is constrained by hardware capabilities and physical factors, leading to the development of processing techniques to enhance the resolution after acquisition. In this work, we leverage artificial intelligence to increase the Doppler resolution in range-Doppler maps. Based on a zero-padded FFT, a refinement via the generative neural networks of diffusion models is achieved. We demonstrate that our method overcomes the limitations of traditional FFT, generating data where closely spaced targets are effectively separated.
>
---
#### [new 101] DetailFusion: A Dual-branch Framework with Detail Enhancement for Composed Image Retrieval
- **分类: cs.CV; cs.AI; cs.IR**

- **简介: 该论文提出DetailFusion框架，针对合成图像检索（CIR）中现有方法忽视细节导致难以处理细微视觉变化或复杂文本指令的问题，设计双分支结构：通过原子细节先验构建细节增强分支，并采用自适应特征融合模块动态整合全局与局部信息，在CIRR和FashionIQ数据集上实现最优性能。**

- **链接: [http://arxiv.org/pdf/2505.17796v1](http://arxiv.org/pdf/2505.17796v1)**

> **作者:** Yuxin Yang; Yinan Zhou; Yuxin Chen; Ziqi Zhang; Zongyang Ma; Chunfeng Yuan; Bing Li; Lin Song; Jun Gao; Peng Li; Weiming Hu
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Composed Image Retrieval (CIR) aims to retrieve target images from a gallery based on a reference image and modification text as a combined query. Recent approaches focus on balancing global information from two modalities and encode the query into a unified feature for retrieval. However, due to insufficient attention to fine-grained details, these coarse fusion methods often struggle with handling subtle visual alterations or intricate textual instructions. In this work, we propose DetailFusion, a novel dual-branch framework that effectively coordinates information across global and detailed granularities, thereby enabling detail-enhanced CIR. Our approach leverages atomic detail variation priors derived from an image editing dataset, supplemented by a detail-oriented optimization strategy to develop a Detail-oriented Inference Branch. Furthermore, we design an Adaptive Feature Compositor that dynamically fuses global and detailed features based on fine-grained information of each unique multimodal query. Extensive experiments and ablation analyses not only demonstrate that our method achieves state-of-the-art performance on both CIRR and FashionIQ datasets but also validate the effectiveness and cross-domain adaptability of detail enhancement for CIR.
>
---
#### [new 102] REACT 2025: the Third Multiple Appropriate Facial Reaction Generation Challenge
- **分类: cs.CV; 68T40**

- **简介: 该论文提出REACT 2025挑战，属多适宜面部反应生成（MAFRG）任务。旨在开发ML模型生成多样、真实且同步的面部反应回应视听输入。工作包括发布大规模多模态数据集MARS（含137组互动、2856个会话），提出离线与在线子挑战，并提供基线代码。**

- **链接: [http://arxiv.org/pdf/2505.17223v1](http://arxiv.org/pdf/2505.17223v1)**

> **作者:** Siyang Song; Micol Spitale; Xiangyu Kong; Hengde Zhu; Cheng Luo; Cristina Palmero; German Barquero; Sergio Escalera; Michel Valstar; Mohamed Daoudi; Tobias Baur; Fabien Ringeval; Andrew Howes; Elisabeth Andre; Hatice Gunes
>
> **摘要:** In dyadic interactions, a broad spectrum of human facial reactions might be appropriate for responding to each human speaker behaviour. Following the successful organisation of the REACT 2023 and REACT 2024 challenges, we are proposing the REACT 2025 challenge encouraging the development and benchmarking of Machine Learning (ML) models that can be used to generate multiple appropriate, diverse, realistic and synchronised human-style facial reactions expressed by human listeners in response to an input stimulus (i.e., audio-visual behaviours expressed by their corresponding speakers). As a key of the challenge, we provide challenge participants with the first natural and large-scale multi-modal MAFRG dataset (called MARS) recording 137 human-human dyadic interactions containing a total of 2856 interaction sessions covering five different topics. In addition, this paper also presents the challenge guidelines and the performance of our baselines on the two proposed sub-challenges: Offline MAFRG and Online MAFRG, respectively. The challenge baseline code is publicly available at https://github.com/reactmultimodalchallenge/baseline_react2025
>
---
#### [new 103] One RL to See Them All: Visual Triple Unified Reinforcement Learning
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出V-Triune系统，通过统一强化学习框架使视觉语言模型（VLMs）同时学习视觉推理与感知任务。针对RL在目标检测等感知任务中应用不足的问题，设计三组件架构（数据格式、奖励计算、指标监测）及动态IoU奖励，训练出Orsta模型，在MEGA-Bench等数据集显著提升多任务性能。**

- **链接: [http://arxiv.org/pdf/2505.18129v1](http://arxiv.org/pdf/2505.18129v1)**

> **作者:** Yan Ma; Linge Du; Xuyang Shen; Shaoxiang Chen; Pengfei Li; Qibing Ren; Lizhuang Ma; Yuchao Dai; Pengfei Liu; Junjie Yan
>
> **备注:** Technical Report
>
> **摘要:** Reinforcement learning (RL) has significantly advanced the reasoning capabilities of vision-language models (VLMs). However, the use of RL beyond reasoning tasks remains largely unexplored, especially for perceptionintensive tasks like object detection and grounding. We propose V-Triune, a Visual Triple Unified Reinforcement Learning system that enables VLMs to jointly learn visual reasoning and perception tasks within a single training pipeline. V-Triune comprises triple complementary components: Sample-Level Data Formatting (to unify diverse task inputs), Verifier-Level Reward Computation (to deliver custom rewards via specialized verifiers) , and Source-Level Metric Monitoring (to diagnose problems at the data-source level). We further introduce a novel Dynamic IoU reward, which provides adaptive, progressive, and definite feedback for perception tasks handled by V-Triune. Our approach is instantiated within off-the-shelf RL training framework using open-source 7B and 32B backbone models. The resulting model, dubbed Orsta (One RL to See Them All), demonstrates consistent improvements across both reasoning and perception tasks. This broad capability is significantly shaped by its training on a diverse dataset, constructed around four representative visual reasoning tasks (Math, Puzzle, Chart, and Science) and four visual perception tasks (Grounding, Detection, Counting, and OCR). Subsequently, Orsta achieves substantial gains on MEGA-Bench Core, with improvements ranging from +2.1 to an impressive +14.1 across its various 7B and 32B model variants, with performance benefits extending to a wide range of downstream tasks. These results highlight the effectiveness and scalability of our unified RL approach for VLMs. The V-Triune system, along with the Orsta models, is publicly available at https://github.com/MiniMax-AI.
>
---
#### [new 104] MR-EEGWaveNet: Multiresolutional EEGWaveNet for Seizure Detection from Long EEG Recordings
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于癫痫发作检测任务，旨在解决现有模型区分癫痫与伪影/噪声效果差的问题。提出多分辨率EEGWaveNet（MR-EEGWaveNet），通过卷积、特征提取和预测模块捕捉时空关系，并引入异常评分后处理降低误报。实验显示其在Siena和Juntendo数据集的F1值显著提升。**

- **链接: [http://arxiv.org/pdf/2505.17972v1](http://arxiv.org/pdf/2505.17972v1)**

> **作者:** Kazi Mahmudul Hassan; Xuyang Zhao; Hidenori Sugano; Toshihisa Tanaka
>
> **备注:** 26 pages, 6 figures, 12 tables
>
> **摘要:** Feature engineering for generalized seizure detection models remains a significant challenge. Recently proposed models show variable performance depending on the training data and remain ineffective at accurately distinguishing artifacts from seizure data. In this study, we propose a novel end-to-end model, ''Multiresolutional EEGWaveNet (MR-EEGWaveNet),'' which efficiently distinguishes seizure events from background electroencephalogram (EEG) and artifacts/noise by capturing both temporal dependencies across different time frames and spatial relationships between channels. The model has three modules: convolution, feature extraction, and predictor. The convolution module extracts features through depth-wise and spatio-temporal convolution. The feature extraction module individually reduces the feature dimension extracted from EEG segments and their sub-segments. Subsequently, the extracted features are concatenated into a single vector for classification using a fully connected classifier called the predictor module. In addition, an anomaly score-based post-classification processing technique was introduced to reduce the false-positive rates of the model. Experimental results were reported and analyzed using different parameter settings and datasets (Siena (public) and Juntendo (private)). The proposed MR-EEGWaveNet significantly outperformed the conventional non-multiresolution approach, improving the F1 scores from 0.177 to 0.336 on Siena and 0.327 to 0.488 on Juntendo, with precision gains of 15.9% and 20.62%, respectively.
>
---
#### [new 105] SVL: Spike-based Vision-language Pretraining for Efficient 3D Open-world Understanding
- **分类: cs.CV**

- **简介: 该论文属于3D开放世界理解任务，旨在解决SNNs性能不足的问题。提出SVL框架，通过MTA实现多模态无监督对齐，及Rep-VLI轻量化整合视觉语言，提升SNNs在3D分类、问答等任务的准确率与效率，首次实现SNNs在复杂场景超越ANNs。**

- **链接: [http://arxiv.org/pdf/2505.17674v1](http://arxiv.org/pdf/2505.17674v1)**

> **作者:** Xuerui Qiu; Peixi Wu; Yaozhi Wen; Shaowei Gu; Yuqi Pan; Xinhao Luo; Bo XU; Guoqi Li
>
> **摘要:** Spiking Neural Networks (SNNs) provide an energy-efficient way to extract 3D spatio-temporal features. However, existing SNNs still exhibit a significant performance gap compared to Artificial Neural Networks (ANNs) due to inadequate pre-training strategies. These limitations manifest as restricted generalization ability, task specificity, and a lack of multimodal understanding, particularly in challenging tasks such as multimodal question answering and zero-shot 3D classification. To overcome these challenges, we propose a Spike-based Vision-Language (SVL) pretraining framework that empowers SNNs with open-world 3D understanding while maintaining spike-driven efficiency. SVL introduces two key components: (i) Multi-scale Triple Alignment (MTA) for label-free triplet-based contrastive learning across 3D, image, and text modalities, and (ii) Re-parameterizable Vision-Language Integration (Rep-VLI) to enable lightweight inference without relying on large text encoders. Extensive experiments show that SVL achieves a top-1 accuracy of 85.4% in zero-shot 3D classification, surpassing advanced ANN models, and consistently outperforms prior SNNs on downstream tasks, including 3D classification (+6.1%), DVS action recognition (+2.1%), 3D detection (+1.1%), and 3D segmentation (+2.1%) with remarkable efficiency. Moreover, SVL enables SNNs to perform open-world 3D question answering, sometimes outperforming ANNs. To the best of our knowledge, SVL represents the first scalable, generalizable, and hardware-friendly paradigm for 3D open-world understanding, effectively bridging the gap between SNNs and ANNs in complex open-world understanding tasks. Code is available https://github.com/bollossom/SVL.
>
---
#### [new 106] Research on Defect Detection Method of Motor Control Board Based on Image Processing
- **分类: cs.CV**

- **简介: 该论文提出基于图像处理的电机控制板缺陷检测方法，针对色差、插件错位、焊点短路等问题，通过图像降噪、特征提取模型建立、搜索算法优化，实现99%以上检测准确率，适用于生产线在线检测及行业电路板缺陷处理。**

- **链接: [http://arxiv.org/pdf/2505.17493v1](http://arxiv.org/pdf/2505.17493v1)**

> **作者:** Jingde Huang; Zhangyu Huang; Chenyu Li; Jiantong Liu
>
> **摘要:** The motor control board has various defects such as inconsistent color differences, incorrect plug-in positions, solder short circuits, and more. These defects directly affect the performance and stability of the motor control board, thereby having a negative impact on product quality. Therefore, studying the defect detection technology of the motor control board is an important means to improve the quality control level of the motor control board. Firstly, the processing methods of digital images about the motor control board were studied, and the noise suppression methods that affect image feature extraction were analyzed. Secondly, a specific model for defect feature extraction and color difference recognition of the tested motor control board was established, and qualified or defective products were determined based on feature thresholds. Thirdly, the search algorithm for defective images was optimized. Finally, comparative experiments were conducted on the typical motor control board, and the experimental results demonstrate that the accuracy of the motor control board defect detection model-based on image processing established in this paper reached over 99%. It is suitable for timely image processing of large quantities of motor control boards on the production line, and achieved efficient defect detection. The defect detection method can not only be used for online detection of the motor control board defects, but also provide solutions for the integrated circuit board defect processing for the industry.
>
---
#### [new 107] Optimizing YOLOv8 for Parking Space Detection: Comparative Analysis of Custom YOLOv8 Architecture
- **分类: cs.CV**

- **简介: 该论文属于停车空间占用检测任务，旨在优化YOLOv8以解决其在部分遮挡车辆、小型车（如摩托车）及低光照条件下的检测不足问题。通过比较ResNet-18、VGG16、EfficientNetV2、Ghost等backbone架构在PKLot数据集上的准确率与效率，分析各模型优缺点，为实际应用提供选型依据。**

- **链接: [http://arxiv.org/pdf/2505.17364v1](http://arxiv.org/pdf/2505.17364v1)**

> **作者:** Apar Pokhrel; Gia Dao
>
> **备注:** 9 pages
>
> **摘要:** Parking space occupancy detection is a critical component in the development of intelligent parking management systems. Traditional object detection approaches, such as YOLOv8, provide fast and accurate vehicle detection across parking lots but can struggle with borderline cases, such as partially visible vehicles, small vehicles (e.g., motorcycles), and poor lighting conditions. In this work, we perform a comprehensive comparative analysis of customized backbone architectures integrated with YOLOv8. Specifically, we evaluate various backbones -- ResNet-18, VGG16, EfficientNetV2, Ghost -- on the PKLot dataset in terms of detection accuracy and computational efficiency. Experimental results highlight each architecture's strengths and trade-offs, providing insight into selecting suitable models for parking occupancy.
>
---
#### [new 108] FS-DAG: Few Shot Domain Adapting Graph Networks for Visually Rich Document Understanding
- **分类: cs.CV; cs.AI; cs.CL; cs.IR; cs.LG; I.2.7; I.5.4; I.7**

- **简介: 该论文提出FS-DAG模型，针对小样本视觉丰富文档理解任务，解决跨领域适应、数据稀缺及OCR错误等问题。通过模块化设计融合领域与模态专用backbone，在信息提取中实现高效收敛与高性能，参数仅90M。**

- **链接: [http://arxiv.org/pdf/2505.17330v1](http://arxiv.org/pdf/2505.17330v1)**

> **作者:** Amit Agarwal; Srikant Panda; Kulbhushan Pachauri
>
> **备注:** Published in the Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025), Industry Track, pages 100-114
>
> **摘要:** In this work, we propose Few Shot Domain Adapting Graph (FS-DAG), a scalable and efficient model architecture for visually rich document understanding (VRDU) in few-shot settings. FS-DAG leverages domain-specific and language/vision specific backbones within a modular framework to adapt to diverse document types with minimal data. The model is robust to practical challenges such as handling OCR errors, misspellings, and domain shifts, which are critical in real-world deployments. FS-DAG is highly performant with less than 90M parameters, making it well-suited for complex real-world applications for Information Extraction (IE) tasks where computational resources are limited. We demonstrate FS-DAG's capability through extensive experiments for information extraction task, showing significant improvements in convergence speed and performance compared to state-of-the-art methods. Additionally, this work highlights the ongoing progress in developing smaller, more efficient models that do not compromise on performance. Code : https://github.com/oracle-samples/fs-dag
>
---
#### [new 109] CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学影像诊断推理评估任务，旨在解决现有基准无法评估模型中间推理步骤的问题。提出CheXStruct pipeline自动生成胸片诊断的结构化中间步骤（如解剖分割、测量计算），并构建CXReasonBench基准，包含18,988个QA对及12项任务，用于测试模型的结构化推理能力和泛化效果，揭示当前LVLMs在关联知识与视觉解释上存在不足。**

- **链接: [http://arxiv.org/pdf/2505.18087v1](http://arxiv.org/pdf/2505.18087v1)**

> **作者:** Hyungyung Lee; Geon Choi; Jung-Oh Lee; Hangyul Yoon; Hyuk Gi Hong; Edward Choi
>
> **摘要:** Recent progress in Large Vision-Language Models (LVLMs) has enabled promising applications in medical tasks, such as report generation and visual question answering. However, existing benchmarks focus mainly on the final diagnostic answer, offering limited insight into whether models engage in clinically meaningful reasoning. To address this, we present CheXStruct and CXReasonBench, a structured pipeline and benchmark built on the publicly available MIMIC-CXR-JPG dataset. CheXStruct automatically derives a sequence of intermediate reasoning steps directly from chest X-rays, such as segmenting anatomical regions, deriving anatomical landmarks and diagnostic measurements, computing diagnostic indices, and applying clinical thresholds. CXReasonBench leverages this pipeline to evaluate whether models can perform clinically valid reasoning steps and to what extent they can learn from structured guidance, enabling fine-grained and transparent assessment of diagnostic reasoning. The benchmark comprises 18,988 QA pairs across 12 diagnostic tasks and 1,200 cases, each paired with up to 4 visual inputs, and supports multi-path, multi-stage evaluation including visual grounding via anatomical region selection and diagnostic measurements. Even the strongest of 10 evaluated LVLMs struggle with structured reasoning and generalization, often failing to link abstract knowledge with anatomically grounded visual interpretation. The code is available at https://github.com/ttumyche/CXReasonBench
>
---
#### [new 110] BOTM: Echocardiography Segmentation via Bi-directional Optimal Token Matching
- **分类: cs.CV**

- **简介: 该论文提出BOTM框架，解决超声心动图分割中的解剖结构不一致问题。通过双向最优标记匹配和跨传输注意力机制，保持心脏动态变形中的解剖一致性，提升低信噪比下的分割精度，在CAMUS2H和TED数据集上取得更优结果。（99字）**

- **链接: [http://arxiv.org/pdf/2505.18052v1](http://arxiv.org/pdf/2505.18052v1)**

> **作者:** Zhihua Liu; Lei Tong; Xilin He; Che Liu; Rossella Arcucci; Chen Jin; Huiyu Zhou
>
> **摘要:** Existed echocardiography segmentation methods often suffer from anatomical inconsistency challenge caused by shape variation, partial observation and region ambiguity with similar intensity across 2D echocardiographic sequences, resulting in false positive segmentation with anatomical defeated structures in challenging low signal-to-noise ratio conditions. To provide a strong anatomical guarantee across different echocardiographic frames, we propose a novel segmentation framework named BOTM (Bi-directional Optimal Token Matching) that performs echocardiography segmentation and optimal anatomy transportation simultaneously. Given paired echocardiographic images, BOTM learns to match two sets of discrete image tokens by finding optimal correspondences from a novel anatomical transportation perspective. We further extend the token matching into a bi-directional cross-transport attention proxy to regulate the preserved anatomical consistency within the cardiac cyclic deformation in temporal domain. Extensive experimental results show that BOTM can generate stable and accurate segmentation outcomes (e.g. -1.917 HD on CAMUS2H LV, +1.9% Dice on TED), and provide a better matching interpretation with anatomical consistency guarantee.
>
---
#### [new 111] Graph Mamba for Efficient Whole Slide Image Understanding
- **分类: cs.CV; cs.AI**

- **简介: 该论文针对高分辨率病理全切片图像（WSI）分析中现有方法扩展性差、计算成本高的问题，提出WSI-GMamba框架。结合图神经网络（GNN）与状态空间模型Mamba，通过GMamba模块实现消息传递、图扫描及双向SSM特征聚合，在滑片分类任务中达到Transformer性能但计算量减少7倍，兼顾精度与效率。**

- **链接: [http://arxiv.org/pdf/2505.17457v1](http://arxiv.org/pdf/2505.17457v1)**

> **作者:** Jiaxuan Lu; Junyan Shi; Yuhui Lin; Fang Yan; Yue Gao; Shaoting Zhang; Xiaosong Wang
>
> **摘要:** Whole Slide Images (WSIs) in histopathology present a significant challenge for large-scale medical image analysis due to their high resolution, large size, and complex tile relationships. Existing Multiple Instance Learning (MIL) methods, such as Graph Neural Networks (GNNs) and Transformer-based models, face limitations in scalability and computational cost. To bridge this gap, we propose the WSI-GMamba framework, which synergistically combines the relational modeling strengths of GNNs with the efficiency of Mamba, the State Space Model designed for sequence learning. The proposed GMamba block integrates Message Passing, Graph Scanning & Flattening, and feature aggregation via a Bidirectional State Space Model (Bi-SSM), achieving Transformer-level performance with 7* fewer FLOPs. By leveraging the complementary strengths of lightweight GNNs and Mamba, the WSI-GMamba framework delivers a scalable solution for large-scale WSI analysis, offering both high accuracy and computational efficiency for slide-level classification.
>
---
#### [new 112] Clip4Retrofit: Enabling Real-Time Image Labeling on Edge Devices via Cross-Architecture CLIP Distillation
- **分类: cs.CV**

- **简介: 该论文提出Clip4Retrofit框架，通过跨架构知识蒸馏将CLIP模型压缩为轻量级EfficientNet-B3+MLP结构，解决边缘设备（如车载摄像头）资源受限下实时图像标注难题，实现低算力场景下的跨模态视觉任务。**

- **链接: [http://arxiv.org/pdf/2505.18039v1](http://arxiv.org/pdf/2505.18039v1)**

> **作者:** Li Zhong; Ahmed Ghazal; Jun-Jun Wan; Frederik Zilly; Patrick Mackens; Joachim E. Vollrath; Bogdan Sorin Coseriu
>
> **摘要:** Foundation models like CLIP (Contrastive Language-Image Pretraining) have revolutionized vision-language tasks by enabling zero-shot and few-shot learning through cross-modal alignment. However, their computational complexity and large memory footprint make them unsuitable for deployment on resource-constrained edge devices, such as in-car cameras used for image collection and real-time processing. To address this challenge, we propose Clip4Retrofit, an efficient model distillation framework that enables real-time image labeling on edge devices. The framework is deployed on the Retrofit camera, a cost-effective edge device retrofitted into thousands of vehicles, despite strict limitations on compute performance and memory. Our approach distills the knowledge of the CLIP model into a lightweight student model, combining EfficientNet-B3 with multi-layer perceptron (MLP) projection heads to preserve cross-modal alignment while significantly reducing computational requirements. We demonstrate that our distilled model achieves a balance between efficiency and performance, making it ideal for deployment in real-world scenarios. Experimental results show that Clip4Retrofit can perform real-time image labeling and object identification on edge devices with limited resources, offering a practical solution for applications such as autonomous driving and retrofitting existing systems. This work bridges the gap between state-of-the-art vision-language models and their deployment in resource-constrained environments, paving the way for broader adoption of foundation models in edge computing.
>
---
#### [new 113] The Coherence Trap: When MLLM-Crafted Narratives Exploit Manipulated Visual Contexts
- **分类: cs.CV**

- **简介: 该论文属于多媒体操纵检测任务，针对多模态大语言模型（MLLM）生成的高风险虚假图文内容检测问题。现有方法存在低估MLLM欺骗风险及依赖不自然不一致内容的缺陷，论文构建MDSM数据集（含编辑图像与MLLM生成的连贯虚假文本），提出AMD框架（含Artifact Pre-perception Encoding和Manipulation-Oriented Reasoning模块），提升检测MLLM驱动的多模态欺骗的泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.17476v1](http://arxiv.org/pdf/2505.17476v1)**

> **作者:** Yuchen Zhang; Yaxiong Wang; Yujiao Wu; Lianwei Wu; Li Zhu
>
> **摘要:** The detection and grounding of multimedia manipulation has emerged as a critical challenge in combating AI-generated disinformation. While existing methods have made progress in recent years, we identify two fundamental limitations in current approaches: (1) Underestimation of MLLM-driven deception risk: prevailing techniques primarily address rule-based text manipulations, yet fail to account for sophisticated misinformation synthesized by multimodal large language models (MLLMs) that can dynamically generate semantically coherent, contextually plausible yet deceptive narratives conditioned on manipulated images; (2) Unrealistic misalignment artifacts: currently focused scenarios rely on artificially misaligned content that lacks semantic coherence, rendering them easily detectable. To address these gaps holistically, we propose a new adversarial pipeline that leverages MLLMs to generate high-risk disinformation. Our approach begins with constructing the MLLM-Driven Synthetic Multimodal (MDSM) dataset, where images are first altered using state-of-the-art editing techniques and then paired with MLLM-generated deceptive texts that maintain semantic consistency with the visual manipulations. Building upon this foundation, we present the Artifact-aware Manipulation Diagnosis via MLLM (AMD) framework featuring two key innovations: Artifact Pre-perception Encoding strategy and Manipulation-Oriented Reasoning, to tame MLLMs for the MDSM problem. Comprehensive experiments validate our framework's superior generalization capabilities as a unified architecture for detecting MLLM-powered multimodal deceptions.
>
---
#### [new 114] AutoMiSeg: Automatic Medical Image Segmentation via Test-Time Adaptation of Foundation Models
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文提出AutoMiSeg，一种零样本自动医学图像分割方法，解决传统方法依赖大量标注或交互提示的问题。通过结合视觉语言模型与分割基础模型，利用初始定位、提示增强及测试时适配框架（含可学习适配器和贝叶斯优化），实现跨任务高效自动化分割。**

- **链接: [http://arxiv.org/pdf/2505.17931v1](http://arxiv.org/pdf/2505.17931v1)**

> **作者:** Xingjian Li; Qifeng Wu; Colleen Que; Yiran Ding; Adithya S. Ubaradka; Jianhua Xing; Tianyang Wang; Min Xu
>
> **摘要:** Medical image segmentation is vital for clinical diagnosis, yet current deep learning methods often demand extensive expert effort, i.e., either through annotating large training datasets or providing prompts at inference time for each new case. This paper introduces a zero-shot and automatic segmentation pipeline that combines off-the-shelf vision-language and segmentation foundation models. Given a medical image and a task definition (e.g., "segment the optic disc in an eye fundus image"), our method uses a grounding model to generate an initial bounding box, followed by a visual prompt boosting module that enhance the prompts, which are then processed by a promptable segmentation model to produce the final mask. To address the challenges of domain gap and result verification, we introduce a test-time adaptation framework featuring a set of learnable adaptors that align the medical inputs with foundation model representations. Its hyperparameters are optimized via Bayesian Optimization, guided by a proxy validation model without requiring ground-truth labels. Our pipeline offers an annotation-efficient and scalable solution for zero-shot medical image segmentation across diverse tasks. Our pipeline is evaluated on seven diverse medical imaging datasets and shows promising results. By proper decomposition and test-time adaptation, our fully automatic pipeline performs competitively with weakly-prompted interactive foundation models.
>
---
#### [new 115] Analyzing Fine-Grained Alignment and Enhancing Vision Understanding in Multimodal Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态模型视觉-语言对齐任务，旨在解决现有投影器对视觉 token 与语义词的对齐不足问题。通过分析投影器对视觉信息的压缩机制，提出"多语义对齐假设"及"patch 对齐训练"方法，增强视觉-语言细粒度对齐，提升图像描述生成、目标定位等任务性能。**

- **链接: [http://arxiv.org/pdf/2505.17316v1](http://arxiv.org/pdf/2505.17316v1)**

> **作者:** Jiachen Jiang; Jinxin Zhou; Bo Peng; Xia Ning; Zhihui Zhu
>
> **摘要:** Achieving better alignment between vision embeddings and Large Language Models (LLMs) is crucial for enhancing the abilities of Multimodal LLMs (MLLMs), particularly for recent models that rely on powerful pretrained vision encoders and LLMs. A common approach to connect the pretrained vision encoder and LLM is through a projector applied after the vision encoder. However, the projector is often trained to enable the LLM to generate captions, and hence the mechanism by which LLMs understand each vision token remains unclear. In this work, we first investigate the role of the projector in compressing vision embeddings and aligning them with word embeddings. We show that the projector significantly compresses visual information, removing redundant details while preserving essential elements necessary for the LLM to understand visual content. We then examine patch-level alignment -- the alignment between each vision patch and its corresponding semantic words -- and propose a *multi-semantic alignment hypothesis*. Our analysis indicates that the projector trained by caption loss improves patch-level alignment but only to a limited extent, resulting in weak and coarse alignment. To address this issue, we propose *patch-aligned training* to efficiently enhance patch-level alignment. Our experiments show that patch-aligned training (1) achieves stronger compression capability and improved patch-level alignment, enabling the MLLM to generate higher-quality captions, (2) improves the MLLM's performance by 16% on referring expression grounding tasks, 4% on question-answering tasks, and 3% on modern instruction-following benchmarks when using the same supervised fine-tuning (SFT) setting. The proposed method can be easily extended to other multimodal models.
>
---
#### [new 116] InfLVG: Reinforce Inference-Time Consistent Long Video Generation with GRPO
- **分类: cs.CV**

- **简介: 该论文提出InfLVG框架，解决长视频生成中推理时计算成本高和跨场景不一致的问题。通过GRPO优化的上下文选择策略，动态保留关键上下文，减少计算并提升一致性。引入混合奖励函数及新评估基准，实验显示可生成9倍长且连贯的视频。**

- **链接: [http://arxiv.org/pdf/2505.17574v1](http://arxiv.org/pdf/2505.17574v1)**

> **作者:** Xueji Fang; Liyuan Ma; Zhiyang Chen; Mingyuan Zhou; Guo-jun Qi
>
> **备注:** Preprint. Under review
>
> **摘要:** Recent advances in text-to-video generation, particularly with autoregressive models, have enabled the synthesis of high-quality videos depicting individual scenes. However, extending these models to generate long, cross-scene videos remains a significant challenge. As the context length grows during autoregressive decoding, computational costs rise sharply, and the model's ability to maintain consistency and adhere to evolving textual prompts deteriorates. We introduce InfLVG, an inference-time framework that enables coherent long video generation without requiring additional long-form video data. InfLVG leverages a learnable context selection policy, optimized via Group Relative Policy Optimization (GRPO), to dynamically identify and retain the most semantically relevant context throughout the generation process. Instead of accumulating the entire generation history, the policy ranks and selects the top-$K$ most contextually relevant tokens, allowing the model to maintain a fixed computational budget while preserving content consistency and prompt alignment. To optimize the policy, we design a hybrid reward function that jointly captures semantic alignment, cross-scene consistency, and artifact reduction. To benchmark performance, we introduce the Cross-scene Video Benchmark (CsVBench) along with an Event Prompt Set (EPS) that simulates complex multi-scene transitions involving shared subjects and varied actions/backgrounds. Experimental results show that InfLVG can extend video length by up to 9$\times$, achieving strong consistency and semantic fidelity across scenes. Our code is available at https://github.com/MAPLE-AIGC/InfLVG.
>
---
#### [new 117] Boosting Open Set Recognition Performance through Modulated Representation Learning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于开放集合识别（OSR）任务，旨在解决现有方法因固定温度参数限制模型探索实例级与语义级特征的问题。提出动态负余弦调度方案，通过逐步调整温度参数，使模型先形成粗略决策边界，再优化细节，提升表示学习的丰富性和泛化性。该方案可无损嵌入现有方法，显著提升OSR及闭集分类性能。**

- **链接: [http://arxiv.org/pdf/2505.18137v1](http://arxiv.org/pdf/2505.18137v1)**

> **作者:** Amit Kumar Kundu; Vaishnavi Patil; Joseph Jaja
>
> **摘要:** The open set recognition (OSR) problem aims to identify test samples from novel semantic classes that are not part of the training classes, a task that is crucial in many practical scenarios. However, existing OSR methods use a constant scaling factor (the temperature) to the logits before applying a loss function, which hinders the model from exploring both ends of the spectrum in representation learning -- from instance-level to semantic-level features. In this paper, we address this problem by enabling temperature-modulated representation learning using our novel negative cosine scheduling scheme. Our scheduling lets the model form a coarse decision boundary at the beginning of training by focusing on fewer neighbors, and gradually prioritizes more neighbors to smooth out rough edges. This gradual task switching leads to a richer and more generalizable representation space. While other OSR methods benefit by including regularization or auxiliary negative samples, such as with mix-up, thereby adding a significant computational overhead, our scheme can be folded into any existing OSR method with no overhead. We implement the proposed scheme on top of a number of baselines, using both cross-entropy and contrastive loss functions as well as a few other OSR methods, and find that our scheme boosts both the OSR performance and the closed set performance in most cases, especially on the tougher semantic shift benchmarks.
>
---
#### [new 118] TokBench: Evaluating Your Visual Tokenizer before Visual Generation
- **分类: cs.CV; cs.DB**

- **简介: 该论文属于视觉生成与压缩评估任务。针对视觉分词器和VAE在文本/人脸等细粒度特征保留上的不足，提出TokBench基准。通过OCR评估文本重建准确率，人脸特征相似性量化保真度，轻量级框架分析多尺度下不同模型性能，并扩展至视频，揭示传统指标局限性。**

- **链接: [http://arxiv.org/pdf/2505.18142v1](http://arxiv.org/pdf/2505.18142v1)**

> **作者:** Junfeng Wu; Dongliang Luo; Weizhi Zhao; Zhihao Xie; Yuanhao Wang; Junyi Li; Xudong Xie; Yuliang Liu; Xiang Bai
>
> **备注:** Benchmark, homepagee: https://wjf5203.github.io/TokBench
>
> **摘要:** In this work, we reveal the limitations of visual tokenizers and VAEs in preserving fine-grained features, and propose a benchmark to evaluate reconstruction performance for two challenging visual contents: text and face. Image tokenization has significantly advanced visual generation and multimodal modeling, particularly with autoregressive models due to the modeling simplicity of discrete tokens. Autoregressive models typically rely on image tokenizers to compress images into discrete tokens for sequential prediction, whereas diffusion models often operate on continuous latent space to reduce computational costs. However, both visual compression approaches inevitably lose visual information, thereby limiting the upper bound of visual generation quality. To evaluate how these compression losses affect text and faces, the most human-sensitive visual elements, we first collect and curate a collection of text and faces images from existing datasets, ensuring clarity and diversity. For text reconstruction, we employ OCR models to assess the recognition accuracy of the reconstructed text, and then we measure feature similarity between original and reconstructed faces thereby quantifying faces reconstruction fidelity. Our method is highly lightweight, requiring just 2GB memory and 4 minutes to complete evaluations. With our benchmark, we analyze the reconstruction quality of text and faces at various scales across different image tokenizers and VAEs. Our results demonstrate that modern visual tokenizers still struggle to preserve fine-grained features, particularly at smaller scales. Furthermore, we extend this evaluation framework to the video, conducting a comprehensive analysis of video tokenizers. Additionally, we find that traditional metrics fail to accurately reflect the reconstruction performance for faces and text, while our proposed metrics serve as an effective complement.
>
---
#### [new 119] Alignment and Safety of Diffusion Models via Reinforcement Learning and Reward Modeling: A Survey
- **分类: cs.CV**

- **简介: 该论文属于技术综述任务，旨在解决扩散模型生成内容与人类偏好及安全标准对齐的挑战。工作包括分类现有方法（按反馈类型、微调技术等）、比较关键算法、提出五项未来研究方向（多目标对齐、高效反馈利用、鲁棒安全、持续对齐及可解释奖励建模）。**

- **链接: [http://arxiv.org/pdf/2505.17352v1](http://arxiv.org/pdf/2505.17352v1)**

> **作者:** Preeti Lamba; Kiran Ravish; Ankita Kushwaha; Pawan Kumar
>
> **摘要:** Diffusion models have emerged as leading generative models for images and other modalities, but aligning their outputs with human preferences and safety constraints remains a critical challenge. This thesis proposal investigates methods to align diffusion models using reinforcement learning (RL) and reward modeling. We survey recent advances in fine-tuning text-to-image diffusion models with human feedback, including reinforcement learning from human and AI feedback, direct preference optimization, and differentiable reward approaches. We classify these methods based on the type of feedback (human, automated, binary or ranked preferences), the fine-tuning technique (policy gradient, reward-weighted likelihood, direct backpropagation, etc.), and their efficiency and safety outcomes. We compare key algorithms and frameworks, highlighting how they improve alignment with user intent or safety standards, and discuss inter-relationships such as how newer methods build on or diverge from earlier ones. Based on the survey, we identify five promising research directions for the next two years: (1) multi-objective alignment with combined rewards, (2) efficient human feedback usage and active learning, (3) robust safety alignment against adversarial inputs, (4) continual and online alignment of diffusion models, and (5) interpretable and trustworthy reward modeling for generative images. Each direction is elaborated with its problem statement, challenges, related work, and a proposed research plan. The proposal is organized as a comprehensive document with literature review, comparative tables of methods, and detailed research plans, aiming to contribute new insights and techniques for safer and value-aligned diffusion-based generative AI.
>
---
#### [new 120] Temporal Differential Fields for 4D Motion Modeling via Image-to-Video Synthesis
- **分类: cs.CV**

- **简介: 该论文提出基于图像到视频合成的4D运动建模方法，解决呼吸诱导运动模拟中因患者微动导致的帧间背景偏差问题。通过设计时序差分扩散模型生成相邻帧的差分场，并结合提示注意力与场增强层，提升合成视频的时间一致性，实现在医学影像数据集上的高质量4D运动模拟。**

- **链接: [http://arxiv.org/pdf/2505.17333v1](http://arxiv.org/pdf/2505.17333v1)**

> **作者:** Xin You; Minghui Zhang; Hanxiao Zhang; Jie Yang; Nassir Navab
>
> **备注:** early accepted by MICCAI
>
> **摘要:** Temporal modeling on regular respiration-induced motions is crucial to image-guided clinical applications. Existing methods cannot simulate temporal motions unless high-dose imaging scans including starting and ending frames exist simultaneously. However, in the preoperative data acquisition stage, the slight movement of patients may result in dynamic backgrounds between the first and last frames in a respiratory period. This additional deviation can hardly be removed by image registration, thus affecting the temporal modeling. To address that limitation, we pioneeringly simulate the regular motion process via the image-to-video (I2V) synthesis framework, which animates with the first frame to forecast future frames of a given length. Besides, to promote the temporal consistency of animated videos, we devise the Temporal Differential Diffusion Model to generate temporal differential fields, which measure the relative differential representations between adjacent frames. The prompt attention layer is devised for fine-grained differential fields, and the field augmented layer is adopted to better interact these fields with the I2V framework, promoting more accurate temporal variation of synthesized videos. Extensive results on ACDC cardiac and 4D Lung datasets reveal that our approach simulates 4D videos along the intrinsic motion trajectory, rivaling other competitive methods on perceptual similarity and temporal consistency. Codes will be available soon.
>
---
#### [new 121] A Wavelet-based Stereo Matching Framework for Solving Frequency Convergence Inconsistency
- **分类: cs.CV**

- **简介: 该论文属于立体匹配任务，旨在解决现有方法在高频区域（如边缘、细小物体）迭代优化时精度下降的问题。提出Wavelet-Stereo框架，通过小波分解分离高低频成分，分别用不同特征提取器处理，并设计LSTM-based更新器自适应优化高频细节，提升边缘与平滑区域的同步精度，在KITTI数据集上取得最优性能。**

- **链接: [http://arxiv.org/pdf/2505.18024v1](http://arxiv.org/pdf/2505.18024v1)**

> **作者:** Xiaobao Wei; Jiawei Liu; Dongbo Yang; Junda Cheng; Changyong Shu; Wei Wang
>
> **摘要:** We find that the EPE evaluation metrics of RAFT-stereo converge inconsistently in the low and high frequency regions, resulting high frequency degradation (e.g., edges and thin objects) during the iterative process. The underlying reason for the limited performance of current iterative methods is that it optimizes all frequency components together without distinguishing between high and low frequencies. We propose a wavelet-based stereo matching framework (Wavelet-Stereo) for solving frequency convergence inconsistency. Specifically, we first explicitly decompose an image into high and low frequency components using discrete wavelet transform. Then, the high-frequency and low-frequency components are fed into two different multi-scale frequency feature extractors. Finally, we propose a novel LSTM-based high-frequency preservation update operator containing an iterative frequency adapter to provide adaptive refined high-frequency features at different iteration steps by fine-tuning the initial high-frequency features. By processing high and low frequency components separately, our framework can simultaneously refine high-frequency information in edges and low-frequency information in smooth regions, which is especially suitable for challenging scenes with fine details and textures in the distance. Extensive experiments demonstrate that our Wavelet-Stereo outperforms the state-of-the-art methods and ranks 1st on both the KITTI 2015 and KITTI 2012 leaderboards for almost all metrics. We will provide code and pre-trained models to encourage further exploration, application, and development of our innovative framework (https://github.com/SIA-IDE/Wavelet-Stereo).
>
---
#### [new 122] T2VUnlearning: A Concept Erasing Method for Text-to-Video Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于文本到视频扩散模型的"unlearning"任务，旨在解决模型生成有害内容的风险。提出T2VUnlearning方法，通过负向引导速度预测微调、提示增强及定位/保持正则化，精准擦除特定概念同时保留其他生成能力，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.17550v1](http://arxiv.org/pdf/2505.17550v1)**

> **作者:** Xiaoyu Ye; Songjie Cheng; Yongtao Wang; Yajiao Xiong; Yishen Li
>
> **摘要:** Recent advances in text-to-video (T2V) diffusion models have significantly enhanced the quality of generated videos. However, their ability to produce explicit or harmful content raises concerns about misuse and potential rights violations. Inspired by the success of unlearning techniques in erasing undesirable concepts from text-to-image (T2I) models, we extend unlearning to T2V models and propose a robust and precise unlearning method. Specifically, we adopt negatively-guided velocity prediction fine-tuning and enhance it with prompt augmentation to ensure robustness against LLM-refined prompts. To achieve precise unlearning, we incorporate a localization and a preservation regularization to preserve the model's ability to generate non-target concepts. Extensive experiments demonstrate that our method effectively erases a specific concept while preserving the model's generation capability for all other concepts, outperforming existing methods. We provide the unlearned models in \href{https://github.com/VDIGPKU/T2VUnlearning.git}{https://github.com/VDIGPKU/T2VUnlearning.git}.
>
---
#### [new 123] Locality-Sensitive Hashing for Efficient Hard Negative Sampling in Contrastive Learning
- **分类: cs.CV**

- **简介: 该论文针对对比学习中高效硬负例采样任务，提出GPU友好的局部敏感哈希（LSH）方案，通过将特征向量二值化实现近似最近邻搜索，解决大规模高维数据中硬负例（特征相似但类别不同）的快速定位难题，理论分析与跨模态实验验证了其计算效率与性能优势。**

- **链接: [http://arxiv.org/pdf/2505.17844v1](http://arxiv.org/pdf/2505.17844v1)**

> **作者:** Fabian Deuser; Philipp Hausenblas; Hannah Schieber; Daniel Roth; Martin Werner; Norbert Oswald
>
> **摘要:** Contrastive learning is a representational learning paradigm in which a neural network maps data elements to feature vectors. It improves the feature space by forming lots with an anchor and examples that are either positive or negative based on class similarity. Hard negative examples, which are close to the anchor in the feature space but from a different class, improve learning performance. Finding such examples of high quality efficiently in large, high-dimensional datasets is computationally challenging. In this paper, we propose a GPU-friendly Locality-Sensitive Hashing (LSH) scheme that quantizes real-valued feature vectors into binary representations for approximate nearest neighbor search. We investigate its theoretical properties and evaluate it on several datasets from textual and visual domain. Our approach achieves comparable or better performance while requiring significantly less computation than existing hard negative mining strategies.
>
---
#### [new 124] MODEM: A Morton-Order Degradation Estimation Mechanism for Adverse Weather Image Recovery
- **分类: cs.CV**

- **简介: 该论文属于恶劣天气图像恢复任务，针对天气导致的非均匀、空间异质退化（如雨条纹、雾霾）问题，提出MODEM机制。其包含MOS2D模块（Morton编码+选择性状态空间模型捕捉长程依赖并保持局部结构）和DDEM模块（分离全局/局部退化先验，动态优化恢复），实现自适应处理，达到SOTA效果。**

- **链接: [http://arxiv.org/pdf/2505.17581v1](http://arxiv.org/pdf/2505.17581v1)**

> **作者:** Hainuo Wang; Qiming Hu; Xiaojie Guo
>
> **摘要:** Restoring images degraded by adverse weather remains a significant challenge due to the highly non-uniform and spatially heterogeneous nature of weather-induced artifacts, e.g., fine-grained rain streaks versus widespread haze. Accurately estimating the underlying degradation can intuitively provide restoration models with more targeted and effective guidance, enabling adaptive processing strategies. To this end, we propose a Morton-Order Degradation Estimation Mechanism (MODEM) for adverse weather image restoration. Central to MODEM is the Morton-Order 2D-Selective-Scan Module (MOS2D), which integrates Morton-coded spatial ordering with selective state-space models to capture long-range dependencies while preserving local structural coherence. Complementing MOS2D, we introduce a Dual Degradation Estimation Module (DDEM) that disentangles and estimates both global and local degradation priors. These priors dynamically condition the MOS2D modules, facilitating adaptive and context-aware restoration. Extensive experiments and ablation studies demonstrate that MODEM achieves state-of-the-art results across multiple benchmarks and weather types, highlighting its effectiveness in modeling complex degradation dynamics. Our code will be released at https://github.com/hainuo-wang/MODEM.git.
>
---
#### [new 125] BiggerGait: Unlocking Gait Recognition with Layer-wise Representations from Large Vision Models
- **分类: cs.CV**

- **简介: 该论文属于步态识别任务，旨在解决大型视觉模型（LVM）未充分利用层级特征的问题。通过分析发现LVM中间层具有互补性，提出BiggerGait方法整合多层表示，无需复杂步态先验即提升性能，多数据集验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2505.18132v1](http://arxiv.org/pdf/2505.18132v1)**

> **作者:** Dingqing Ye; Chao Fan; Zhanbo Huang; Chengwen Luo; Jianqiang Li; Shiqi Yu; Xiaoming Liu
>
> **摘要:** Large vision models (LVM) based gait recognition has achieved impressive performance. However, existing LVM-based approaches may overemphasize gait priors while neglecting the intrinsic value of LVM itself, particularly the rich, distinct representations across its multi-layers. To adequately unlock LVM's potential, this work investigates the impact of layer-wise representations on downstream recognition tasks. Our analysis reveals that LVM's intermediate layers offer complementary properties across tasks, integrating them yields an impressive improvement even without rich well-designed gait priors. Building on this insight, we propose a simple and universal baseline for LVM-based gait recognition, termed BiggerGait. Comprehensive evaluations on CCPG, CAISA-B*, SUSTech1K, and CCGR\_MINI validate the superiority of BiggerGait across both within- and cross-domain tasks, establishing it as a simple yet practical baseline for gait representation learning. All the models and code will be publicly available.
>
---
#### [new 126] Canonical Pose Reconstruction from Single Depth Image for 3D Non-rigid Pose Recovery on Limited Datasets
- **分类: cs.CV**

- **简介: 该论文属于3D非刚性姿态重建任务，旨在解决传统方法依赖大量数据且难以处理变形物体的问题。提出通过单深度图像重构规范姿态模型，将非刚性问题转化为刚性重建，支持小规模数据（约300样本）下的高精度3D姿态恢复，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.17992v1](http://arxiv.org/pdf/2505.17992v1)**

> **作者:** Fahd Alhamazani; Yu-Kun Lai; Paul L. Rosin
>
> **摘要:** 3D reconstruction from 2D inputs, especially for non-rigid objects like humans, presents unique challenges due to the significant range of possible deformations. Traditional methods often struggle with non-rigid shapes, which require extensive training data to cover the entire deformation space. This study addresses these limitations by proposing a canonical pose reconstruction model that transforms single-view depth images of deformable shapes into a canonical form. This alignment facilitates shape reconstruction by enabling the application of rigid object reconstruction techniques, and supports recovering the input pose in voxel representation as part of the reconstruction task, utilizing both the original and deformed depth images. Notably, our model achieves effective results with only a small dataset of approximately 300 samples. Experimental results on animal and human datasets demonstrate that our model outperforms other state-of-the-art methods.
>
---
#### [new 127] Segment Anyword: Mask Prompt Inversion for Open-Set Grounded Segmentation
- **分类: cs.CV**

- **简介: 该论文针对开放集语言引导分割任务，解决现有方法依赖大量训练且难以处理多样文本表达导致分割不一致的问题。提出Segment Anyword方法，利用冻结扩散模型生成mask提示，并通过语言引导正则化优化，提升分割精度，取得SOTA结果。**

- **链接: [http://arxiv.org/pdf/2505.17994v1](http://arxiv.org/pdf/2505.17994v1)**

> **作者:** Zhihua Liu; Amrutha Saseendran; Lei Tong; Xilin He; Fariba Yousefi; Nikolay Burlutskiy; Dino Oglic; Tom Diethe; Philip Teare; Huiyu Zhou; Chen Jin
>
> **摘要:** Open-set image segmentation poses a significant challenge because existing methods often demand extensive training or fine-tuning and generally struggle to segment unified objects consistently across diverse text reference expressions. Motivated by this, we propose Segment Anyword, a novel training-free visual concept prompt learning approach for open-set language grounded segmentation that relies on token-level cross-attention maps from a frozen diffusion model to produce segmentation surrogates or mask prompts, which are then refined into targeted object masks. Initial prompts typically lack coherence and consistency as the complexity of the image-text increases, resulting in suboptimal mask fragments. To tackle this issue, we further introduce a novel linguistic-guided visual prompt regularization that binds and clusters visual prompts based on sentence dependency and syntactic structural information, enabling the extraction of robust, noise-tolerant mask prompts, and significant improvements in segmentation accuracy. The proposed approach is effective, generalizes across different open-set segmentation tasks, and achieves state-of-the-art results of 52.5 (+6.8 relative) mIoU on Pascal Context 59, 67.73 (+25.73 relative) cIoU on gRefCOCO, and 67.4 (+1.1 relative to fine-tuned methods) mIoU on GranDf, which is the most complex open-set grounded segmentation task in the field.
>
---
#### [new 128] Robustifying Vision-Language Models via Dynamic Token Reweighting
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型（VLM）安全防御任务，旨在解决对抗性视觉文本交互（jailbreak攻击）突破安全限制的问题。提出DTR方法，通过动态调整视觉标记权重优化KV缓存，减少对抗输入影响，同时保持模型性能与效率，无需额外安全数据或图像转文本转换。**

- **链接: [http://arxiv.org/pdf/2505.17132v1](http://arxiv.org/pdf/2505.17132v1)**

> **作者:** Tanqiu Jiang; Jiacheng Liang; Rongyi Zhu; Jiawei Zhou; Fenglong Ma; Ting Wang
>
> **摘要:** Large vision-language models (VLMs) are highly vulnerable to jailbreak attacks that exploit visual-textual interactions to bypass safety guardrails. In this paper, we present DTR, a novel inference-time defense that mitigates multimodal jailbreak attacks through optimizing the model's key-value (KV) caches. Rather than relying on curated safety-specific data or costly image-to-text conversion, we introduce a new formulation of the safety-relevant distributional shift induced by the visual modality. This formulation enables DTR to dynamically adjust visual token weights, minimizing the impact of adversarial visual inputs while preserving the model's general capabilities and inference efficiency. Extensive evaluation across diverse VLMs and attack benchmarks demonstrates that \sys outperforms existing defenses in both attack robustness and benign task performance, marking the first successful application of KV cache optimization for safety enhancement in multimodal foundation models. The code for replicating DTR is available: https://anonymous.4open.science/r/DTR-2755 (warning: this paper contains potentially harmful content generated by VLMs.)
>
---
#### [new 129] DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 该论文提出DualTalk框架，解决3D虚拟对话中自然切换说话与倾听的问题。通过整合双角色动态行为，生成连贯互动，创建50小时多轮对话数据集，实验验证提升自然度和表现力。**

- **链接: [http://arxiv.org/pdf/2505.18096v1](http://arxiv.org/pdf/2505.18096v1)**

> **作者:** Ziqiao Peng; Yanbo Fan; Haoyu Wu; Xuan Wang; Hongyan Liu; Jun He; Zhaoxin Fan
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** In face-to-face conversations, individuals need to switch between speaking and listening roles seamlessly. Existing 3D talking head generation models focus solely on speaking or listening, neglecting the natural dynamics of interactive conversation, which leads to unnatural interactions and awkward transitions. To address this issue, we propose a new task -- multi-round dual-speaker interaction for 3D talking head generation -- which requires models to handle and generate both speaking and listening behaviors in continuous conversation. To solve this task, we introduce DualTalk, a novel unified framework that integrates the dynamic behaviors of speakers and listeners to simulate realistic and coherent dialogue interactions. This framework not only synthesizes lifelike talking heads when speaking but also generates continuous and vivid non-verbal feedback when listening, effectively capturing the interplay between the roles. We also create a new dataset featuring 50 hours of multi-round conversations with over 1,000 characters, where participants continuously switch between speaking and listening roles. Extensive experiments demonstrate that our method significantly enhances the naturalness and expressiveness of 3D talking heads in dual-speaker conversations. We recommend watching the supplementary video: https://ziqiaopeng.github.io/dualtalk.
>
---
#### [new 130] ProTAL: A Drag-and-Link Video Programming Framework for Temporal Action Localization
- **分类: cs.HC; cs.CV**

- **简介: 该论文属于时间动作定位（TAL）任务。针对传统方法依赖大量人工标注及复杂动作定义困难的问题，提出ProTAL框架：通过拖拽身体部位/物体节点并链接定义关键事件，自动生成标签，结合半监督训练模型。实验通过用例和用户研究验证了有效性。（99字）**

- **链接: [http://arxiv.org/pdf/2505.17555v1](http://arxiv.org/pdf/2505.17555v1)**

> **作者:** Yuchen He; Jianbing Lv; Liqi Cheng; Lingyu Meng; Dazhen Deng; Yingcai Wu
>
> **备注:** Accepted at CHI'25
>
> **摘要:** Temporal Action Localization (TAL) aims to detect the start and end timestamps of actions in a video. However, the training of TAL models requires a substantial amount of manually annotated data. Data programming is an efficient method to create training labels with a series of human-defined labeling functions. However, its application in TAL faces difficulties of defining complex actions in the context of temporal video frames. In this paper, we propose ProTAL, a drag-and-link video programming framework for TAL. ProTAL enables users to define \textbf{key events} by dragging nodes representing body parts and objects and linking them to constrain the relations (direction, distance, etc.). These definitions are used to generate action labels for large-scale unlabelled videos. A semi-supervised method is then employed to train TAL models with such labels. We demonstrate the effectiveness of ProTAL through a usage scenario and a user study, providing insights into designing video programming framework.
>
---
#### [new 131] Distance Estimation in Outdoor Driving Environments Using Phase-only Correlation Method with Event Cameras
- **分类: eess.IV; cs.CV; cs.RO; I.4.8; I.2.10; I.5.4**

- **简介: 该论文提出基于事件相机与 roadside LED 的单目距离估计算法，解决多传感器融合成本高的问题。通过相位相关法检测 LED 光源空间偏移，实现三角测距，实验显示20-60米距离误差<0.5米，成功率达90%以上。**

- **链接: [http://arxiv.org/pdf/2505.17582v1](http://arxiv.org/pdf/2505.17582v1)**

> **作者:** Masataka Kobayashi; Shintaro Shiba; Quan Kong; Norimasa Kobori; Tsukasa Shimizu; Shan Lu; Takaya Yamazato
>
> **备注:** 6 pages, 7 figures. To appear in IEEE Intelligent Vehicles Symposium (IV) 2025
>
> **摘要:** With the growing adoption of autonomous driving, the advancement of sensor technology is crucial for ensuring safety and reliable operation. Sensor fusion techniques that combine multiple sensors such as LiDAR, radar, and cameras have proven effective, but the integration of multiple devices increases both hardware complexity and cost. Therefore, developing a single sensor capable of performing multiple roles is highly desirable for cost-efficient and scalable autonomous driving systems. Event cameras have emerged as a promising solution due to their unique characteristics, including high dynamic range, low latency, and high temporal resolution. These features enable them to perform well in challenging lighting conditions, such as low-light or backlit environments. Moreover, their ability to detect fine-grained motion events makes them suitable for applications like pedestrian detection and vehicle-to-infrastructure communication via visible light. In this study, we present a method for distance estimation using a monocular event camera and a roadside LED bar. By applying a phase-only correlation technique to the event data, we achieve sub-pixel precision in detecting the spatial shift between two light sources. This enables accurate triangulation-based distance estimation without requiring stereo vision. Field experiments conducted in outdoor driving scenarios demonstrated that the proposed approach achieves over 90% success rate with less than 0.5-meter error for distances ranging from 20 to 60 meters. Future work includes extending this method to full position estimation by leveraging infrastructure such as smart poles equipped with LEDs, enabling event-camera-based vehicles to determine their own position in real time. This advancement could significantly enhance navigation accuracy, route optimization, and integration into intelligent transportation systems.
>
---
#### [new 132] Multi-Person Interaction Generation from Two-Person Motion Priors
- **分类: cs.GR; cs.CV; cs.LG; I.3.7**

- **简介: 该论文提出Graph-driven Interaction Sampling方法，通过将多人群体互动分解为两两交互的图结构（Pairwise Interaction Graph），利用现有双人动作扩散模型生成高质量多人交互，引入空间时间指导项减少穿模，无需训练新模型，有效生成多样动作且优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.17860v1](http://arxiv.org/pdf/2505.17860v1)**

> **作者:** Wenning Xu; Shiyu Fan; Paul Henderson; Edmond S. L. Ho
>
> **备注:** SIGGRAPH 2025 Conference Papers
>
> **摘要:** Generating realistic human motion with high-level controls is a crucial task for social understanding, robotics, and animation. With high-quality MOCAP data becoming more available recently, a wide range of data-driven approaches have been presented. However, modelling multi-person interactions still remains a less explored area. In this paper, we present Graph-driven Interaction Sampling, a method that can generate realistic and diverse multi-person interactions by leveraging existing two-person motion diffusion models as motion priors. Instead of training a new model specific to multi-person interaction synthesis, our key insight is to spatially and temporally separate complex multi-person interactions into a graph structure of two-person interactions, which we name the Pairwise Interaction Graph. We thus decompose the generation task into simultaneous single-person motion generation conditioned on one other's motion. In addition, to reduce artifacts such as interpenetrations of body parts in generated multi-person interactions, we introduce two graph-dependent guidance terms into the diffusion sampling scheme. Unlike previous work, our method can produce various high-quality multi-person interactions without having repetitive individual motions. Extensive experiments demonstrate that our approach consistently outperforms existing methods in reducing artifacts when generating a wide range of two-person and multi-person interactions.
>
---
#### [new 133] Is Single-View Mesh Reconstruction Ready for Robotics?
- **分类: cs.RO; cs.CV; I.4.5; I.4.8; I.2.9; I.2.10**

- **简介: 该论文评估单视图网格重建在机器人数字孪生中的适用性。任务为验证其能否满足物理模拟、碰撞检测及计算约束等机器人需求。建立基准标准，通过真实数据集验证发现现有方法不足，揭示计算机视觉技术与机器人应用间的差距，指导未来研究。**

- **链接: [http://arxiv.org/pdf/2505.17966v1](http://arxiv.org/pdf/2505.17966v1)**

> **作者:** Frederik Nolte; Bernhard Schölkopf; Ingmar Posner
>
> **备注:** 20 pages, 17 figures
>
> **摘要:** This paper evaluates single-view mesh reconstruction models for creating digital twin environments in robot manipulation. Recent advances in computer vision for 3D reconstruction from single viewpoints present a potential breakthrough for efficiently creating virtual replicas of physical environments for robotics contexts. However, their suitability for physics simulations and robotics applications remains unexplored. We establish benchmarking criteria for 3D reconstruction in robotics contexts, including handling typical inputs, producing collision-free and stable reconstructions, managing occlusions, and meeting computational constraints. Our empirical evaluation using realistic robotics datasets shows that despite success on computer vision benchmarks, existing approaches fail to meet robotics-specific requirements. We quantitively examine limitations of single-view reconstruction for practical robotics implementation, in contrast to prior work that focuses on multi-view approaches. Our findings highlight critical gaps between computer vision advances and robotics needs, guiding future research at this intersection.
>
---
#### [new 134] FreqU-FNet: Frequency-Aware U-Net for Imbalanced Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出FreqU-FNet，解决医学图像分割中类别不平衡与频率分布问题。通过频率域U型架构，结合低通卷积、小波下采样提取多尺度频谱特征，自适应多分支上采样重建细节，并设计频率感知损失函数，提升少数类学习效果。实验显示其优于CNN和Transformer基线。**

- **链接: [http://arxiv.org/pdf/2505.17544v1](http://arxiv.org/pdf/2505.17544v1)**

> **作者:** Ruiqi Xing
>
> **备注:** 15 pages, 1 figure
>
> **摘要:** Medical image segmentation faces persistent challenges due to severe class imbalance and the frequency-specific distribution of anatomical structures. Most conventional CNN-based methods operate in the spatial domain and struggle to capture minority class signals, often affected by frequency aliasing and limited spectral selectivity. Transformer-based models, while powerful in modeling global dependencies, tend to overlook critical local details necessary for fine-grained segmentation. To overcome these limitations, we propose FreqU-FNet, a novel U-shaped segmentation architecture operating in the frequency domain. Our framework incorporates a Frequency Encoder that leverages Low-Pass Frequency Convolution and Daubechies wavelet-based downsampling to extract multi-scale spectral features. To reconstruct fine spatial details, we introduce a Spatial Learnable Decoder (SLD) equipped with an adaptive multi-branch upsampling strategy. Furthermore, we design a frequency-aware loss (FAL) function to enhance minority class learning. Extensive experiments on multiple medical segmentation benchmarks demonstrate that FreqU-FNet consistently outperforms both CNN and Transformer baselines, particularly in handling under-represented classes, by effectively exploiting discriminative frequency bands.
>
---
#### [new 135] Explainable Anatomy-Guided AI for Prostate MRI: Foundation Models and In Silico Clinical Trials for Virtual Biopsy-based Risk Assessment
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出基于解剖学引导的AI系统，用于前列腺癌风险分层。任务是通过MRI实现自动化、可解释的癌症风险评估。解决传统诊断效率低、可解释性差的问题。工作包括：1）整合nnU-Net分割前列腺区域；2）用UMedPT Swin Transformer分类模型结合解剖先验和临床数据；3）通过VAE-GAN生成反事实热图解释决策；4）模拟临床试验证明AI提升诊断准确率（0.77）和效率（节省40%时间），优于2024竞赛结果。**

- **链接: [http://arxiv.org/pdf/2505.17971v1](http://arxiv.org/pdf/2505.17971v1)**

> **作者:** Danial Khan; Zohaib Salahuddin; Yumeng Zhang; Sheng Kuang; Shruti Atul Mali; Henry C. Woodruff; Sina Amirrajab; Rachel Cavill; Eduardo Ibor-Crespo; Ana Jimenez-Pastor; Adrian Galiana-Bordera; Paula Jimenez Gomez; Luis Marti-Bonmati; Philippe Lambin
>
> **摘要:** We present a fully automated, anatomically guided deep learning pipeline for prostate cancer (PCa) risk stratification using routine MRI. The pipeline integrates three key components: an nnU-Net module for segmenting the prostate gland and its zones on axial T2-weighted MRI; a classification module based on the UMedPT Swin Transformer foundation model, fine-tuned on 3D patches with optional anatomical priors and clinical data; and a VAE-GAN framework for generating counterfactual heatmaps that localize decision-driving image regions. The system was developed using 1,500 PI-CAI cases for segmentation and 617 biparametric MRIs with metadata from the CHAIMELEON challenge for classification (split into 70% training, 10% validation, and 20% testing). Segmentation achieved mean Dice scores of 0.95 (gland), 0.94 (peripheral zone), and 0.92 (transition zone). Incorporating gland priors improved AUC from 0.69 to 0.72, with a three-scale ensemble achieving top performance (AUC = 0.79, composite score = 0.76), outperforming the 2024 CHAIMELEON challenge winners. Counterfactual heatmaps reliably highlighted lesions within segmented regions, enhancing model interpretability. In a prospective multi-center in-silico trial with 20 clinicians, AI assistance increased diagnostic accuracy from 0.72 to 0.77 and Cohen's kappa from 0.43 to 0.53, while reducing review time per case by 40%. These results demonstrate that anatomy-aware foundation models with counterfactual explainability can enable accurate, interpretable, and efficient PCa risk assessment, supporting their potential use as virtual biopsies in clinical practice.
>
---
#### [new 136] Assessing the generalization performance of SAM for ureteroscopy scene understanding
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文评估了SAM模型在肾结石分割任务中的泛化性能，旨在解决传统模型（如U-Net系列）在新数据集上表现不佳的问题。通过对比实验发现，SAM在分布内数据与U-Net相当，但分布外数据表现显著更优（最高提升23%），证明其更强适应性。**

- **链接: [http://arxiv.org/pdf/2505.17210v1](http://arxiv.org/pdf/2505.17210v1)**

> **作者:** Martin Villagrana; Francisco Lopez-Tiro; Clement Larose; Gilberto Ochoa-Ruiz; Christian Daul
>
> **备注:** 15 pages, 4 figures, 2 tables, conference, MIUA25
>
> **摘要:** The segmentation of kidney stones is regarded as a critical preliminary step to enable the identification of urinary stone types through machine- or deep-learning-based approaches. In urology, manual segmentation is considered tedious and impractical due to the typically large scale of image databases and the continuous generation of new data. In this study, the potential of the Segment Anything Model (SAM) -- a state-of-the-art deep learning framework -- is investigated for the automation of kidney stone segmentation. The performance of SAM is evaluated in comparison to traditional models, including U-Net, Residual U-Net, and Attention U-Net, which, despite their efficiency, frequently exhibit limitations in generalizing to unseen datasets. The findings highlight SAM's superior adaptability and efficiency. While SAM achieves comparable performance to U-Net on in-distribution data (Accuracy: 97.68 + 3.04; Dice: 97.78 + 2.47; IoU: 95.76 + 4.18), it demonstrates significantly enhanced generalization capabilities on out-of-distribution data, surpassing all U-Net variants by margins of up to 23 percent.
>
---
#### [new 137] MMMG: a Comprehensive and Reliable Evaluation Suite for Multitask Multimodal Generation
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 论文提出MMMG基准，解决多模态生成自动评估与人类评价不一致的问题。涵盖图像、音频等4种模态组合的49项任务（含29新任务），通过模型与程序结合实现可靠评估，与人类评价一致率达94.3%。测试24个模型显示，现有模型在多模态推理和音频生成上仍有较大提升空间。（99字）**

- **链接: [http://arxiv.org/pdf/2505.17613v1](http://arxiv.org/pdf/2505.17613v1)**

> **作者:** Jihan Yao; Yushi Hu; Yujie Yi; Bin Han; Shangbin Feng; Guang Yang; Bingbing Wen; Ranjay Krishna; Lucy Lu Wang; Yulia Tsvetkov; Noah A. Smith; Banghua Zhu
>
> **摘要:** Automatically evaluating multimodal generation presents a significant challenge, as automated metrics often struggle to align reliably with human evaluation, especially for complex tasks that involve multiple modalities. To address this, we present MMMG, a comprehensive and human-aligned benchmark for multimodal generation across 4 modality combinations (image, audio, interleaved text and image, interleaved text and audio), with a focus on tasks that present significant challenges for generation models, while still enabling reliable automatic evaluation through a combination of models and programs. MMMG encompasses 49 tasks (including 29 newly developed ones), each with a carefully designed evaluation pipeline, and 937 instructions to systematically assess reasoning, controllability, and other key capabilities of multimodal generation models. Extensive validation demonstrates that MMMG is highly aligned with human evaluation, achieving an average agreement of 94.3%. Benchmarking results on 24 multimodal generation models reveal that even though the state-of-the-art model, GPT Image, achieves 78.3% accuracy for image generation, it falls short on multimodal reasoning and interleaved generation. Furthermore, results suggest considerable headroom for improvement in audio generation, highlighting an important direction for future research.
>
---
#### [new 138] WonderPlay: Dynamic 3D Scene Generation from a Single Image and Actions
- **分类: cs.GR; cs.AI; cs.CV**

- **简介: 论文提出WonderPlay框架，任务为通过单张图像和用户动作生成动态3D场景。解决现有方法仅支持刚体或简单弹性动态的问题。其工作是开发混合生成模拟器：先用物理引擎模拟粗粒度动态，再以扩散模型生成精细视频，并通过闭环更新实现多材质（布料、流体等）的交互式动态生成。**

- **链接: [http://arxiv.org/pdf/2505.18151v1](http://arxiv.org/pdf/2505.18151v1)**

> **作者:** Zizhang Li; Hong-Xing Yu; Wei Liu; Yin Yang; Charles Herrmann; Gordon Wetzstein; Jiajun Wu
>
> **备注:** The first two authors contributed equally. Project website: https://kyleleey.github.io/WonderPlay/
>
> **摘要:** WonderPlay is a novel framework integrating physics simulation with video generation for generating action-conditioned dynamic 3D scenes from a single image. While prior works are restricted to rigid body or simple elastic dynamics, WonderPlay features a hybrid generative simulator to synthesize a wide range of 3D dynamics. The hybrid generative simulator first uses a physics solver to simulate coarse 3D dynamics, which subsequently conditions a video generator to produce a video with finer, more realistic motion. The generated video is then used to update the simulated dynamic 3D scene, closing the loop between the physics solver and the video generator. This approach enables intuitive user control to be combined with the accurate dynamics of physics-based simulators and the expressivity of diffusion-based video generators. Experimental results demonstrate that WonderPlay enables users to interact with various scenes of diverse content, including cloth, sand, snow, liquid, smoke, elastic, and rigid bodies -- all using a single image input. Code will be made public. Project website: https://kyleleey.github.io/WonderPlay/
>
---
#### [new 139] Anatomy-Guided Multitask Learning for MRI-Based Classification of Placenta Accreta Spectrum and its Subtypes
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出基于MRI的PAS及其亚型（PA/PI/PP）分类任务，解决现有方法效率低且亚型识别不足的问题。通过设计双分支CNN架构（主分类分支与解剖特征分支）及多任务学习策略，实现高效单阶段多分类，在临床数据中达最优效果。**

- **链接: [http://arxiv.org/pdf/2505.17484v1](http://arxiv.org/pdf/2505.17484v1)**

> **作者:** Hai Jiang; Qiongting Liu; Yuanpin Zhou; Jiawei Pan; Ting Song; Yao Lu
>
> **摘要:** Placenta Accreta Spectrum Disorders (PAS) pose significant risks during pregnancy, frequently leading to postpartum hemorrhage during cesarean deliveries and other severe clinical complications, with bleeding severity correlating to the degree of placental invasion. Consequently, accurate prenatal diagnosis of PAS and its subtypes-placenta accreta (PA), placenta increta (PI), and placenta percreta (PP)-is crucial. However, existing guidelines and methodologies predominantly focus on the presence of PAS, with limited research addressing subtype recognition. Additionally, previous multi-class diagnostic efforts have primarily relied on inefficient two-stage cascaded binary classification tasks. In this study, we propose a novel convolutional neural network (CNN) architecture designed for efficient one-stage multiclass diagnosis of PAS and its subtypes, based on 4,140 magnetic resonance imaging (MRI) slices. Our model features two branches: the main classification branch utilizes a residual block architecture comprising multiple residual blocks, while the second branch integrates anatomical features of the uteroplacental area and the adjacent uterine serous layer to enhance the model's attention during classification. Furthermore, we implement a multitask learning strategy to leverage both branches effectively. Experiments conducted on a real clinical dataset demonstrate that our model achieves state-of-the-art performance.
>
---
#### [new 140] Variational Autoencoding Discrete Diffusion with Enhanced Dimensional Correlations Modeling
- **分类: cs.LG; cs.CV; stat.ML**

- **简介: 该论文属于离散数据生成任务，旨在解决masked扩散模型（MDMs）因维度间依赖建模不足导致的少步数去噪性能下降问题。提出VADD框架，结合变分自编码器，通过辅助识别模型隐式建模维度相关性，稳定训练并提升生成质量，尤其在少步数场景下优于传统MDMs。**

- **链接: [http://arxiv.org/pdf/2505.17384v1](http://arxiv.org/pdf/2505.17384v1)**

> **作者:** Tianyu Xie; Shuchen Xue; Zijin Feng; Tianyang Hu; Jiacheng Sun; Zhenguo Li; Cheng Zhang
>
> **备注:** 23 pages, 14 figures
>
> **摘要:** Discrete diffusion models have recently shown great promise for modeling complex discrete data, with masked diffusion models (MDMs) offering a compelling trade-off between quality and generation speed. MDMs denoise by progressively unmasking multiple dimensions from an all-masked input, but their performance can degrade when using few denoising steps due to limited modeling of inter-dimensional dependencies. In this paper, we propose Variational Autoencoding Discrete Diffusion (VADD), a novel framework that enhances discrete diffusion with latent variable modeling to implicitly capture correlations among dimensions. By introducing an auxiliary recognition model, VADD enables stable training via variational lower bounds maximization and amortized inference over the training set. Our approach retains the efficiency of traditional MDMs while significantly improving sample quality, especially when the number of denoising steps is small. Empirical results on 2D toy data, pixel-level image generation, and text generation demonstrate that VADD consistently outperforms MDM baselines.
>
---
#### [new 141] OCR-Reasoning Benchmark: Unveiling the True Capabilities of MLLMs in Complex Text-Rich Image Reasoning
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出OCR-Reasoning基准，评估多模态大模型在文本密集图像推理中的能力。针对现有方法缺乏系统评测的问题，构建含1,069个标注样本的基准，覆盖6类推理能力及18项任务，并首次标注推理过程。实验显示现有模型准确率不足50%，凸显该任务挑战性。**

- **链接: [http://arxiv.org/pdf/2505.17163v1](http://arxiv.org/pdf/2505.17163v1)**

> **作者:** Mingxin Huang; Yongxin Shi; Dezhi Peng; Songxuan Lai; Zecheng Xie; Lianwen Jin
>
> **摘要:** Recent advancements in multimodal slow-thinking systems have demonstrated remarkable performance across diverse visual reasoning tasks. However, their capabilities in text-rich image reasoning tasks remain understudied due to the lack of a systematic benchmark. To address this gap, we propose OCR-Reasoning, a comprehensive benchmark designed to systematically assess Multimodal Large Language Models on text-rich image reasoning tasks. The benchmark comprises 1,069 human-annotated examples spanning 6 core reasoning abilities and 18 practical reasoning tasks in text-rich visual scenarios. Furthermore, unlike other text-rich image understanding benchmarks that only annotate the final answers, OCR-Reasoning also annotates the reasoning process simultaneously. With the annotated reasoning process and the final answers, OCR-Reasoning evaluates not only the final answers generated by models but also their reasoning processes, enabling a holistic analysis of their problem-solving abilities. Leveraging this benchmark, we conducted a comprehensive evaluation of state-of-the-art MLLMs. Our results demonstrate the limitations of existing methodologies. Notably, even state-of-the-art MLLMs exhibit substantial difficulties, with none achieving accuracy surpassing 50\% across OCR-Reasoning, indicating that the challenges of text-rich image reasoning are an urgent issue to be addressed. The benchmark and evaluation scripts are available at https://github.com/SCUT-DLVCLab/OCR-Reasoning.
>
---
#### [new 142] VideoGameBench: Can Vision-Language Models complete popular video games?
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出VideoGameBench，评估视觉语言模型（VLMs）在实时游戏任务中的能力，解决其在感知、空间导航等人类强项任务上的不足。通过设计包含10款90年代游戏的基准（3款隐藏），要求模型仅用视觉输入和目标描述完成游戏，发现前沿模型因推理延迟表现差（Gemini 2.5 Pro仅完成0.48%），故推出暂停等待的Lite版本（1.6%）。旨在推动VLMs相关研究。**

- **链接: [http://arxiv.org/pdf/2505.18134v1](http://arxiv.org/pdf/2505.18134v1)**

> **作者:** Alex L. Zhang; Thomas L. Griffiths; Karthik R. Narasimhan; Ofir Press
>
> **备注:** 9 pages, 33 pages including supplementary
>
> **摘要:** Vision-language models (VLMs) have achieved strong results on coding and math benchmarks that are challenging for humans, yet their ability to perform tasks that come naturally to humans--such as perception, spatial navigation, and memory management--remains understudied. Real video games are crafted to be intuitive for humans to learn and master by leveraging innate inductive biases, making them an ideal testbed for evaluating such capabilities in VLMs. To this end, we introduce VideoGameBench, a benchmark consisting of 10 popular video games from the 1990s that VLMs directly interact with in real-time. VideoGameBench challenges models to complete entire games with access to only raw visual inputs and a high-level description of objectives and controls, a significant departure from existing setups that rely on game-specific scaffolding and auxiliary information. We keep three of the games secret to encourage solutions that generalize to unseen environments. Our experiments show that frontier vision-language models struggle to progress beyond the beginning of each game. We find inference latency to be a major limitation of frontier models in the real-time setting; therefore, we introduce VideoGameBench Lite, a setting where the game pauses while waiting for the LM's next action. The best performing model, Gemini 2.5 Pro, completes only 0.48% of VideoGameBench and 1.6% of VideoGameBench Lite. We hope that the formalization of the human skills mentioned above into this benchmark motivates progress in these research directions.
>
---
#### [new 143] Soft-CAM: Making black box models self-explainable for high-stakes decisions
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于可解释人工智能任务，旨在解决黑箱模型（如CNN）在高风险领域（如医疗）的解释可靠性问题。提出Soft-CAM方法：通过移除全局平均池化层并替换分类层为卷积证据层，使模型直接生成类激活图，保留空间信息并提升解释可信度，同时保持分类性能。**

- **链接: [http://arxiv.org/pdf/2505.17748v1](http://arxiv.org/pdf/2505.17748v1)**

> **作者:** Kerol Djoumessi; Philipp Berens
>
> **摘要:** Convolutional neural networks (CNNs) are widely used for high-stakes applications like medicine, often surpassing human performance. However, most explanation methods rely on post-hoc attribution, approximating the decision-making process of already trained black-box models. These methods are often sensitive, unreliable, and fail to reflect true model reasoning, limiting their trustworthiness in critical applications. In this work, we introduce SoftCAM, a straightforward yet effective approach that makes standard CNN architectures inherently interpretable. By removing the global average pooling layer and replacing the fully connected classification layer with a convolution-based class evidence layer, SoftCAM preserves spatial information and produces explicit class activation maps that form the basis of the model's predictions. Evaluated on three medical datasets, SoftCAM maintains classification performance while significantly improving both the qualitative and quantitative explanation compared to existing post-hoc methods. Our results demonstrate that CNNs can be inherently interpretable without compromising performance, advancing the development of self-explainable deep learning for high-stakes decision-making.
>
---
#### [new 144] TACO: Enhancing Multimodal In-context Learning via Task Mapping-Guided Sequence Configuration
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态in-context learning（ICL）任务，旨在解决复杂任务中输入序列质量敏感及模型推理机制不明确的问题。提出TACO模型，通过任务映射分析演示序列的局部/全局关系，并动态配置上下文序列，实现序列构建与任务推理的协同优化，实验显示其性能优于基线方法。**

- **链接: [http://arxiv.org/pdf/2505.17098v1](http://arxiv.org/pdf/2505.17098v1)**

> **作者:** Yanshu Li; Tian Yun; Jianjiang Yang; Pinyuan Feng; Jinfa Huang; Ruixiang Tang
>
> **备注:** 29 pages, 11 figures, 19 tables. arXiv admin note: substantial text overlap with arXiv:2503.04839
>
> **摘要:** Multimodal in-context learning (ICL) has emerged as a key mechanism for harnessing the capabilities of large vision-language models (LVLMs). However, its effectiveness remains highly sensitive to the quality of input in-context sequences, particularly for tasks involving complex reasoning or open-ended generation. A major limitation is our limited understanding of how LVLMs actually exploit these sequences during inference. To bridge this gap, we systematically interpret multimodal ICL through the lens of task mapping, which reveals how local and global relationships within and among demonstrations guide model reasoning. Building on this insight, we present TACO, a lightweight transformer-based model equipped with task-aware attention that dynamically configures in-context sequences. By injecting task-mapping signals into the autoregressive decoding process, TACO creates a bidirectional synergy between sequence construction and task reasoning. Experiments on five LVLMs and nine datasets demonstrate that TACO consistently surpasses baselines across diverse ICL tasks. These results position task mapping as a valuable perspective for interpreting and improving multimodal ICL.
>
---
#### [new 145] DECT-based Space-Squeeze Method for Multi-Class Classification of Metastatic Lymph Nodes in Breast Cancer
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出基于DECT的空间挤压方法，用于乳腺癌淋巴结转移的三分类（N0、低/高转移）。针对传统影像难以区分转移程度的问题，结合通道注意力压缩光谱-空间特征及虚拟类注入优化分类边界，提升多分类AUC至0.86，优于传统CNN，有效辅助治疗决策。**

- **链接: [http://arxiv.org/pdf/2505.17528v1](http://arxiv.org/pdf/2505.17528v1)**

> **作者:** Hai Jiang; Chushan Zheng; Jiawei Pan; Yuanpin Zhou; Qiongting Liu; Xiang Zhang; Jun Shen; Yao Lu
>
> **摘要:** Background: Accurate assessment of metastatic burden in axillary lymph nodes is crucial for guiding breast cancer treatment decisions, yet conventional imaging modalities struggle to differentiate metastatic burden levels and capture comprehensive lymph node characteristics. This study leverages dual-energy computed tomography (DECT) to exploit spectral-spatial information for improved multi-class classification. Purpose: To develop a noninvasive DECT-based model classifying sentinel lymph nodes into three categories: no metastasis ($N_0$), low metastatic burden ($N_{+(1-2)}$), and heavy metastatic burden ($N_{+(\geq3)}$), thereby aiding therapeutic planning. Methods: We propose a novel space-squeeze method combining two innovations: (1) a channel-wise attention mechanism to compress and recalibrate spectral-spatial features across 11 energy levels, and (2) virtual class injection to sharpen inter-class boundaries and compact intra-class variations in the representation space. Results: Evaluated on 227 biopsy-confirmed cases, our method achieved an average test AUC of 0.86 (95% CI: 0.80-0.91) across three cross-validation folds, outperforming established CNNs (VGG, ResNet, etc). The channel-wise attention and virtual class components individually improved AUC by 5.01% and 5.87%, respectively, demonstrating complementary benefits. Conclusions: The proposed framework enhances diagnostic AUC by effectively integrating DECT's spectral-spatial data and mitigating class ambiguity, offering a promising tool for noninvasive metastatic burden assessment in clinical practice.
>
---
#### [new 146] Baitradar: A Multi-Model Clickbait Detection Algorithm Using Deep Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文提出BaitRadar算法，针对YouTube平台clickbait问题，通过融合视频标题、评论、封面、标签、统计信息及音频文本的六模态深度学习模型，实现高精度检测。算法采用多模型平均决策，支持缺失数据场景，测试集准确率达98%，推理时间小于2秒。**

- **链接: [http://arxiv.org/pdf/2505.17448v1](http://arxiv.org/pdf/2505.17448v1)**

> **作者:** Bhanuka Gamage; Adnan Labib; Aisha Joomun; Chern Hong Lim; KokSheik Wong
>
> **备注:** Appear in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP'21), Toronto, ON, Canada
>
> **摘要:** Following the rising popularity of YouTube, there is an emerging problem on this platform called clickbait, which provokes users to click on videos using attractive titles and thumbnails. As a result, users ended up watching a video that does not have the content as publicized in the title. This issue is addressed in this study by proposing an algorithm called BaitRadar, which uses a deep learning technique where six inference models are jointly consulted to make the final classification decision. These models focus on different attributes of the video, including title, comments, thumbnail, tags, video statistics and audio transcript. The final classification is attained by computing the average of multiple models to provide a robust and accurate output even in situation where there is missing data. The proposed method is tested on 1,400 YouTube videos. On average, a test accuracy of 98% is achieved with an inference time of less than 2s.
>
---
#### [new 147] SUFFICIENT: A scan-specific unsupervised deep learning framework for high-resolution 3D isotropic fetal brain MRI reconstruction
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出无监督深度学习框架SUFFICIENT，解决胎儿MRI因运动伪影导致的3D高分辨率重建难题。针对临床数据稀缺问题，通过迭代结合切片配准(SVR)与超分辨率重建(SRR)，利用CNN优化刚体变换及降级模型，实现高质量各向同性体积重建，临床实验验证其优越性。**

- **链接: [http://arxiv.org/pdf/2505.17472v1](http://arxiv.org/pdf/2505.17472v1)**

> **作者:** Jiangjie Wu; Lixuan Chen; Zhenghao Li; Xin Li; Saban Ozturk; Lihui Wang; Rongpin Wang; Hongjiang Wei; Yuyao Zhang
>
> **摘要:** High-quality 3D fetal brain MRI reconstruction from motion-corrupted 2D slices is crucial for clinical diagnosis. Reliable slice-to-volume registration (SVR)-based motion correction and super-resolution reconstruction (SRR) methods are essential. Deep learning (DL) has demonstrated potential in enhancing SVR and SRR when compared to conventional methods. However, it requires large-scale external training datasets, which are difficult to obtain for clinical fetal MRI. To address this issue, we propose an unsupervised iterative SVR-SRR framework for isotropic HR volume reconstruction. Specifically, SVR is formulated as a function mapping a 2D slice and a 3D target volume to a rigid transformation matrix, which aligns the slice to the underlying location in the target volume. The function is parameterized by a convolutional neural network, which is trained by minimizing the difference between the volume slicing at the predicted position and the input slice. In SRR, a decoding network embedded within a deep image prior framework is incorporated with a comprehensive image degradation model to produce the high-resolution (HR) volume. The deep image prior framework offers a local consistency prior to guide the reconstruction of HR volumes. By performing a forward degradation model, the HR volume is optimized by minimizing loss between predicted slices and the observed slices. Comprehensive experiments conducted on large-magnitude motion-corrupted simulation data and clinical data demonstrate the superior performance of the proposed framework over state-of-the-art fetal brain reconstruction frameworks.
>
---
#### [new 148] Plan-R1: Safe and Feasible Trajectory Planning as Language Modeling
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶轨迹规划任务，旨在解决现有方法依赖专家数据导致的安全隐患（如超速）。提出Plan-R1框架：先用自回归模型预测轨迹，再通过规则奖励（避撞、限速）和强化学习优化，提升安全性和可行性，在nuPlan基准中达SOTA。**

- **链接: [http://arxiv.org/pdf/2505.17659v1](http://arxiv.org/pdf/2505.17659v1)**

> **作者:** Xiaolong Tang; Meina Kan; Shiguang Shan; Xilin Chen
>
> **摘要:** Safe and feasible trajectory planning is essential for real-world autonomous driving systems. However, existing learning-based planning methods often rely on expert demonstrations, which not only lack explicit safety awareness but also risk inheriting unsafe behaviors such as speeding from suboptimal human driving data. Inspired by the success of large language models, we propose Plan-R1, a novel two-stage trajectory planning framework that formulates trajectory planning as a sequential prediction task, guided by explicit planning principles such as safety, comfort, and traffic rule compliance. In the first stage, we train an autoregressive trajectory predictor via next motion token prediction on expert data. In the second stage, we design rule-based rewards (e.g., collision avoidance, speed limits) and fine-tune the model using Group Relative Policy Optimization (GRPO), a reinforcement learning strategy, to align its predictions with these planning principles. Experiments on the nuPlan benchmark demonstrate that our Plan-R1 significantly improves planning safety and feasibility, achieving state-of-the-art performance.
>
---
#### [new 149] A Foundation Model Framework for Multi-View MRI Classification of Extramural Vascular Invasion and Mesorectal Fascia Invasion in Rectal Cancer
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出基于多视图MRI的直肠癌EVI（血管外侵犯）和MFI（直肠筋膜侵犯）自动分类框架，解决临床视觉评估主观性及设备差异问题。通过频率域调谐消除扫描仪对比度差异，结合UMedPT预训练模型与多视角特征融合，实现优于现有方法的分类性能（EVI AUC 0.82，MFI AUC 0.77）。**

- **链接: [http://arxiv.org/pdf/2505.18058v1](http://arxiv.org/pdf/2505.18058v1)**

> **作者:** Yumeng Zhang; Zohaib Salahuddin; Danial Khan; Shruti Atul Mali; Henry C. Woodruff; Sina Amirrajab; Eduardo Ibor-Crespo; Ana Jimenez-Pastor; Luis Marti-Bonmati; Philippe Lambin
>
> **备注:** 22 pages, 8 figures
>
> **摘要:** Background: Accurate MRI-based identification of extramural vascular invasion (EVI) and mesorectal fascia invasion (MFI) is pivotal for risk-stratified management of rectal cancer, yet visual assessment is subjective and vulnerable to inter-institutional variability. Purpose: To develop and externally evaluate a multicenter, foundation-model-driven framework that automatically classifies EVI and MFI on axial and sagittal T2-weighted MRI. Methods: This retrospective study used 331 pre-treatment rectal cancer MRI examinations from three European hospitals. After TotalSegmentator-guided rectal patch extraction, a self-supervised frequency-domain harmonization pipeline was trained to minimize scanner-related contrast shifts. Four classifiers were compared: ResNet50, SeResNet, the universal biomedical pretrained transformer (UMedPT) with a lightweight MLP head, and a logistic-regression variant using frozen UMedPT features (UMedPT_LR). Results: UMedPT_LR achieved the best EVI detection when axial and sagittal features were fused (AUC = 0.82; sensitivity = 0.75; F1 score = 0.73), surpassing the Chaimeleon Grand-Challenge winner (AUC = 0.74). The highest MFI performance was attained by UMedPT on axial harmonized images (AUC = 0.77), surpassing the Chaimeleon Grand-Challenge winner (AUC = 0.75). Frequency-domain harmonization improved MFI classification but variably affected EVI performance. Conventional CNNs (ResNet50, SeResNet) underperformed, especially in F1 score and balanced accuracy. Conclusion: These findings demonstrate that combining foundation model features, harmonization, and multi-view fusion significantly enhances diagnostic performance in rectal MRI.
>
---
#### [new 150] MinkUNeXt-SI: Improving point cloud-based place recognition including spherical coordinates and LiDAR intensity
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出MinkUNeXt-SI方法，解决点云位置识别在场景变化（如季节/天气）下的鲁棒性和泛化性问题。通过融合球面坐标与LiDAR强度数据，结合Minkowski卷积和U-Net架构构建深度学习模型，生成稳健描述符。实验表明其超越现有方法，并通过自建数据集验证，代码和数据已公开。**

- **链接: [http://arxiv.org/pdf/2505.17591v1](http://arxiv.org/pdf/2505.17591v1)**

> **作者:** Judith Vilella-Cantos; Juan José Cabrera; Luis Payá; Mónica Ballesta; David Valiente
>
> **摘要:** In autonomous navigation systems, the solution of the place recognition problem is crucial for their safe functioning. But this is not a trivial solution, since it must be accurate regardless of any changes in the scene, such as seasonal changes and different weather conditions, and it must be generalizable to other environments. This paper presents our method, MinkUNeXt-SI, which, starting from a LiDAR point cloud, preprocesses the input data to obtain its spherical coordinates and intensity values normalized within a range of 0 to 1 for each point, and it produces a robust place recognition descriptor. To that end, a deep learning approach that combines Minkowski convolutions and a U-net architecture with skip connections is used. The results of MinkUNeXt-SI demonstrate that this method reaches and surpasses state-of-the-art performance while it also generalizes satisfactorily to other datasets. Additionally, we showcase the capture of a custom dataset and its use in evaluating our solution, which also achieves outstanding results. Both the code of our solution and the runs of our dataset are publicly available for reproducibility purposes.
>
---
#### [new 151] Dual Attention Residual U-Net for Accurate Brain Ultrasound Segmentation in IVH Detection
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文提出结合CBAM与SAL的双注意力残差U-Net，用于早产儿脑室内出血（IVH）的脑超声图像分割。任务是准确分割脑室区域，解决局部细节与全局上下文依赖的捕捉难题。通过双注意力机制优化特征提取，实验达Dice 89.04%、IoU 81.84%，提升分割鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.17683v1](http://arxiv.org/pdf/2505.17683v1)**

> **作者:** Dan Yuan; Yi Feng; Ziyun Tang
>
> **备注:** 10 pages,6 figures and 3 tables
>
> **摘要:** Intraventricular hemorrhage (IVH) is a severe neurological complication among premature infants, necessitating early and accurate detection from brain ultrasound (US) images to improve clinical outcomes. While recent deep learning methods offer promise for computer-aided diagnosis, challenges remain in capturing both local spatial details and global contextual dependencies critical for segmenting brain anatomies. In this work, we propose an enhanced Residual U-Net architecture incorporating two complementary attention mechanisms: the Convolutional Block Attention Module (CBAM) and a Sparse Attention Layer (SAL). The CBAM improves the model's ability to refine spatial and channel-wise features, while the SAL introduces a dual-branch design, sparse attention filters out low-confidence query-key pairs to suppress noise, and dense attention ensures comprehensive information propagation. Extensive experiments on the Brain US dataset demonstrate that our method achieves state-of-the-art segmentation performance, with a Dice score of 89.04% and IoU of 81.84% for ventricle region segmentation. These results highlight the effectiveness of integrating spatial refinement and attention sparsity for robust brain anatomy detection. Code is available at: https://github.com/DanYuan001/BrainImgSegment.
>
---
#### [new 152] TAGS: 3D Tumor-Adaptive Guidance for SAM
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出TAGS框架，解决2D预训练模型在3D医学肿瘤分割中的领域差距问题。通过融合CLIP的语义提示与SAM的特征提取，增强3D解剖结构理解，在肿瘤分割任务中超越现有方法（如nnUNet提升46.88%）。**

- **链接: [http://arxiv.org/pdf/2505.17096v1](http://arxiv.org/pdf/2505.17096v1)**

> **作者:** Sirui Li; Linkai Peng; Zheyuan Zhang; Gorkem Durak; Ulas Bagci
>
> **摘要:** Foundation models (FMs) such as CLIP and SAM have recently shown great promise in image segmentation tasks, yet their adaptation to 3D medical imaging-particularly for pathology detection and segmentation-remains underexplored. A critical challenge arises from the domain gap between natural images and medical volumes: existing FMs, pre-trained on 2D data, struggle to capture 3D anatomical context, limiting their utility in clinical applications like tumor segmentation. To address this, we propose an adaptation framework called TAGS: Tumor Adaptive Guidance for SAM, which unlocks 2D FMs for 3D medical tasks through multi-prompt fusion. By preserving most of the pre-trained weights, our approach enhances SAM's spatial feature extraction using CLIP's semantic insights and anatomy-specific prompts. Extensive experiments on three open-source tumor segmentation datasets prove that our model surpasses the state-of-the-art medical image segmentation models (+46.88% over nnUNet), interactive segmentation frameworks, and other established medical FMs, including SAM-Med2D, SAM-Med3D, SegVol, Universal, 3D-Adapter, and SAM-B (at least +13% over them). This highlights the robustness and adaptability of our proposed framework across diverse medical segmentation tasks.
>
---
#### [new 153] Enhancing Large Vision-Language Models with Layout Modality for Table Question Answering on Japanese Annual Securities Reports
- **分类: cs.CL; cs.CV; 68T50; I.2**

- **简介: 该论文针对日文证券报告表格问答任务，提出通过融合表格文本与布局信息增强视觉语言模型，解决现有模型在解析表格字符及空间关系上的不足，实验表明该方法有效提升复杂布局下的理解能力。**

- **链接: [http://arxiv.org/pdf/2505.17625v1](http://arxiv.org/pdf/2505.17625v1)**

> **作者:** Hayato Aida; Kosuke Takahashi; Takahiro Omi
>
> **备注:** Accepted at IIAI AAI 2025, the 3rd International Conference on Computational and Data Sciences in Economics and Finance
>
> **摘要:** With recent advancements in Large Language Models (LLMs) and growing interest in retrieval-augmented generation (RAG), the ability to understand table structures has become increasingly important. This is especially critical in financial domains such as securities reports, where highly accurate question answering (QA) over tables is required. However, tables exist in various formats-including HTML, images, and plain text-making it difficult to preserve and extract structural information. Therefore, multimodal LLMs are essential for robust and general-purpose table understanding. Despite their promise, current Large Vision-Language Models (LVLMs), which are major representatives of multimodal LLMs, still face challenges in accurately understanding characters and their spatial relationships within documents. In this study, we propose a method to enhance LVLM-based table understanding by incorporating in-table textual content and layout features. Experimental results demonstrate that these auxiliary modalities significantly improve performance, enabling robust interpretation of complex document layouts without relying on explicitly structured input formats.
>
---
#### [new 154] Knot So Simple: A Minimalistic Environment for Spatial Reasoning
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出KnotGym环境，用于解决基于图像的复杂绳结操作任务，通过量化绳结交叉数构建分层挑战。旨在整合视觉感知、空间推理与操作控制，测试模型基础强化学习、预测控制及链式推理方法，提供可扩展的通用测试平台。**

- **链接: [http://arxiv.org/pdf/2505.18028v1](http://arxiv.org/pdf/2505.18028v1)**

> **作者:** Zizhao Chen; Yoav Artzi
>
> **摘要:** We propose KnotGym, an interactive environment for complex, spatial reasoning and manipulation. KnotGym includes goal-oriented rope manipulation tasks with varying levels of complexity, all requiring acting from pure image observations. Tasks are defined along a clear and quantifiable axis of complexity based on the number of knot crossings, creating a natural generalization test. KnotGym has a simple observation space, allowing for scalable development, yet it highlights core challenges in integrating acute perception, spatial reasoning, and grounded manipulation. We evaluate methods of different classes, including model-based RL, model-predictive control, and chain-of-thought reasoning, and illustrate the challenges KnotGym presents. KnotGym is available at https://github.com/lil-lab/knotgym.
>
---
#### [new 155] Graph Attention Neural Network for Botnet Detection: Evaluating Autoencoder, VAE and PCA-Based Dimension Reduction
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于物联网僵尸网络检测任务，旨在解决高维IoT数据构建图结构的计算开销问题。研究提出先通过AE、VAE、PCA降维NetFlow数据，再输入图注意力网络（GAT）提升检测效果，对比了三种降维方法对模型性能的影响。**

- **链接: [http://arxiv.org/pdf/2505.17357v1](http://arxiv.org/pdf/2505.17357v1)**

> **作者:** Hassan Wasswa; Hussein Abbass; Timothy Lynar
>
> **摘要:** With the rise of IoT-based botnet attacks, researchers have explored various learning models for detection, including traditional machine learning, deep learning, and hybrid approaches. A key advancement involves deploying attention mechanisms to capture long-term dependencies among features, significantly improving detection accuracy. However, most models treat attack instances independently, overlooking inter-instance relationships. Graph Neural Networks (GNNs) address this limitation by learning an embedding space via iterative message passing where similar instances are placed closer based on node features and relationships, enhancing classification performance. To further improve detection, attention mechanisms have been embedded within GNNs, leveraging both long-range dependencies and inter-instance connections. However, transforming the high dimensional IoT attack datasets into a graph structured dataset poses challenges, such as large graph structures leading computational overhead. To mitigate this, this paper proposes a framework that first reduces dimensionality of the NetFlow-based IoT attack dataset before transforming it into a graph dataset. We evaluate three dimension reduction techniques--Variational Autoencoder (VAE-encoder), classical autoencoder (AE-encoder), and Principal Component Analysis (PCA)--and compare their effects on a Graph Attention neural network (GAT) model for botnet attack detection
>
---
#### [new 156] Mixture of Decoding: An Attention-Inspired Adaptive Decoding Strategy to Mitigate Hallucinations in Large Vision-Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 论文提出MoD方法，通过评估大视觉语言模型（LVLMs）对图像token注意力的一致性，动态采用互补或对比策略调整解码，减少模型幻觉问题，在多基准测试中效果显著。**

- **链接: [http://arxiv.org/pdf/2505.17061v1](http://arxiv.org/pdf/2505.17061v1)**

> **作者:** Xinlong Chen; Yuanxing Zhang; Qiang Liu; Junfei Wu; Fuzheng Zhang; Tieniu Tan
>
> **备注:** Accepted to Findings of ACL 2025
>
> **摘要:** Large Vision-Language Models (LVLMs) have exhibited impressive capabilities across various visual tasks, yet they remain hindered by the persistent challenge of hallucinations. To address this critical issue, we propose Mixture of Decoding (MoD), a novel approach for hallucination mitigation that dynamically adapts decoding strategies by evaluating the correctness of the model's attention on image tokens. Specifically, MoD measures the consistency between outputs generated from the original image tokens and those derived from the model's attended image tokens, to distinguish the correctness aforementioned. If the outputs are consistent, indicating correct attention, MoD employs a complementary strategy to amplify critical information. Conversely, if the outputs are inconsistent, suggesting erroneous attention, MoD utilizes a contrastive strategy to suppress misleading information. Extensive experiments demonstrate that MoD significantly outperforms existing decoding methods across multiple mainstream benchmarks, effectively mitigating hallucinations in LVLMs. The code is available at https://github.com/xlchen0205/MoD.
>
---
#### [new 157] Promptable cancer segmentation using minimal expert-curated data
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，针对高标注成本和数据需求问题，提出结合弱/全监督分类器的提示式方法，仅需24张全标注+8张弱标注图像，通过单点提示引导分割，实现前列腺癌分割，性能超现有方法且标注数据减少100倍。**

- **链接: [http://arxiv.org/pdf/2505.17915v1](http://arxiv.org/pdf/2505.17915v1)**

> **作者:** Lynn Karam; Yipei Wang; Veeru Kasivisvanathan; Mirabela Rusu; Yipeng Hu; Shaheer U. Saeed
>
> **备注:** Accepted at Medical Image Understanding and Analysis (MIUA) 2025
>
> **摘要:** Automated segmentation of cancer on medical images can aid targeted diagnostic and therapeutic procedures. However, its adoption is limited by the high cost of expert annotations required for training and inter-observer variability in datasets. While weakly-supervised methods mitigate some challenges, using binary histology labels for training as opposed to requiring full segmentation, they require large paired datasets of histology and images, which are difficult to curate. Similarly, promptable segmentation aims to allow segmentation with no re-training for new tasks at inference, however, existing models perform poorly on pathological regions, again necessitating large datasets for training. In this work we propose a novel approach for promptable segmentation requiring only 24 fully-segmented images, supplemented by 8 weakly-labelled images, for training. Curating this minimal data to a high standard is relatively feasible and thus issues with the cost and variability of obtaining labels can be mitigated. By leveraging two classifiers, one weakly-supervised and one fully-supervised, our method refines segmentation through a guided search process initiated by a single-point prompt. Our approach outperforms existing promptable segmentation methods, and performs comparably with fully-supervised methods, for the task of prostate cancer segmentation, while using substantially less annotated data (up to 100X less). This enables promptable segmentation with very minimal labelled data, such that the labels can be curated to a very high standard.
>
---
#### [new 158] Towards more transferable adversarial attack in black-box manner
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于黑盒对抗攻击任务，旨在提升对抗样本的迁移性。针对现有方法依赖白盒模型架构且计算成本高的问题，提出结合时间依赖分类器得分的新型损失函数与代理模型，融入自然数据分布知识优化对抗样本生成，在降低计算开销的同时提升跨模型架构的攻击迁移性。**

- **链接: [http://arxiv.org/pdf/2505.18097v1](http://arxiv.org/pdf/2505.18097v1)**

> **作者:** Chun Tong Lei; Zhongliang Guo; Hon Chung Lee; Minh Quoc Duong; Chun Pong Lau
>
> **摘要:** Adversarial attacks have become a well-explored domain, frequently serving as evaluation baselines for model robustness. Among these, black-box attacks based on transferability have received significant attention due to their practical applicability in real-world scenarios. Traditional black-box methods have generally focused on improving the optimization framework (e.g., utilizing momentum in MI-FGSM) to enhance transferability, rather than examining the dependency on surrogate white-box model architectures. Recent state-of-the-art approach DiffPGD has demonstrated enhanced transferability by employing diffusion-based adversarial purification models for adaptive attacks. The inductive bias of diffusion-based adversarial purification aligns naturally with the adversarial attack process, where both involving noise addition, reducing dependency on surrogate white-box model selection. However, the denoising process of diffusion models incurs substantial computational costs through chain rule derivation, manifested in excessive VRAM consumption and extended runtime. This progression prompts us to question whether introducing diffusion models is necessary. We hypothesize that a model sharing similar inductive bias to diffusion-based adversarial purification, combined with an appropriate loss function, could achieve comparable or superior transferability while dramatically reducing computational overhead. In this paper, we propose a novel loss function coupled with a unique surrogate model to validate our hypothesis. Our approach leverages the score of the time-dependent classifier from classifier-guided diffusion models, effectively incorporating natural data distribution knowledge into the adversarial optimization process. Experimental results demonstrate significantly improved transferability across diverse model architectures while maintaining robustness against diffusion-based defenses.
>
---
#### [new 159] CHART-6: Human-Centered Evaluation of Data Visualization Understanding in Vision-Language Models
- **分类: cs.HC; cs.CL; cs.CV**

- **简介: 该论文评估视觉语言模型对数据可视化理解的类人能力。针对现有模型与人类评估标准脱节的问题，使用六个人类设计的测评任务对比八种模型与人类的表现，发现模型平均得分更低且错误模式显著不同，揭示了模型在认知模拟上的不足。任务属模型与人类行为对比，旨在改进数据可视化推理的AI系统开发。**

- **链接: [http://arxiv.org/pdf/2505.17202v1](http://arxiv.org/pdf/2505.17202v1)**

> **作者:** Arnav Verma; Kushin Mukherjee; Christopher Potts; Elisa Kreiss; Judith E. Fan
>
> **摘要:** Data visualizations are powerful tools for communicating patterns in quantitative data. Yet understanding any data visualization is no small feat -- succeeding requires jointly making sense of visual, numerical, and linguistic inputs arranged in a conventionalized format one has previously learned to parse. Recently developed vision-language models are, in principle, promising candidates for developing computational models of these cognitive operations. However, it is currently unclear to what degree these models emulate human behavior on tasks that involve reasoning about data visualizations. This gap reflects limitations in prior work that has evaluated data visualization understanding in artificial systems using measures that differ from those typically used to assess these abilities in humans. Here we evaluated eight vision-language models on six data visualization literacy assessments designed for humans and compared model responses to those of human participants. We found that these models performed worse than human participants on average, and this performance gap persisted even when using relatively lenient criteria to assess model performance. Moreover, while relative performance across items was somewhat correlated between models and humans, all models produced patterns of errors that were reliably distinct from those produced by human participants. Taken together, these findings suggest significant opportunities for further development of artificial systems that might serve as useful models of how humans reason about data visualizations. All code and data needed to reproduce these results are available at: https://osf.io/e25mu/?view_only=399daff5a14d4b16b09473cf19043f18.
>
---
#### [new 160] SynRES: Towards Referring Expression Segmentation in the Wild via Synthetic Data
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于指代表达分割（RES）任务，针对现有基准无法评估复杂推理的问题，提出WildRES基准（含长描述、多目标及跨领域场景）和SynRES合成数据生成方法，通过三创新（密集描述合成、语义对齐、领域增强）提升模型性能，在WildRES上gIoU提升2.0%-3.8%。**

- **链接: [http://arxiv.org/pdf/2505.17695v1](http://arxiv.org/pdf/2505.17695v1)**

> **作者:** Dong-Hee Kim; Hyunjee Song; Donghyun Kim
>
> **摘要:** Despite the advances in Referring Expression Segmentation (RES) benchmarks, their evaluation protocols remain constrained, primarily focusing on either single targets with short queries (containing minimal attributes) or multiple targets from distinctly different queries on a single domain. This limitation significantly hinders the assessment of more complex reasoning capabilities in RES models. We introduce WildRES, a novel benchmark that incorporates long queries with diverse attributes and non-distinctive queries for multiple targets. This benchmark spans diverse application domains, including autonomous driving environments and robotic manipulation scenarios, thus enabling more rigorous evaluation of complex reasoning capabilities in real-world settings. Our analysis reveals that current RES models demonstrate substantial performance deterioration when evaluated on WildRES. To address this challenge, we introduce SynRES, an automated pipeline generating densely paired compositional synthetic training data through three innovations: (1) a dense caption-driven synthesis for attribute-rich image-mask-expression triplets, (2) reliable semantic alignment mechanisms rectifying caption-pseudo mask inconsistencies via Image-Text Aligned Grouping, and (3) domain-aware augmentations incorporating mosaic composition and superclass replacement to emphasize generalization ability and distinguishing attributes over object categories. Experimental results demonstrate that models trained with SynRES achieve state-of-the-art performance, improving gIoU by 2.0% on WildRES-ID and 3.8% on WildRES-DS. Code and datasets are available at https://github.com/UTLLab/SynRES.
>
---
#### [new 161] A Coreset Selection of Coreset Selection Literature: Introduction and Recent Advances
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于coreset选择方法的综述任务，旨在解决现有文献覆盖不全的问题。通过整合训练无关、训练导向及无标签三类方法，建立统一分类体系，涵盖子模函数、双层优化等被忽视方向，分析剪枝策略对泛化的影响，并对比方法性能，提出面向基础模型等开放挑战。**

- **链接: [http://arxiv.org/pdf/2505.17799v1](http://arxiv.org/pdf/2505.17799v1)**

> **作者:** Brian B. Moser; Arundhati S. Shanbhag; Stanislav Frolov; Federico Raue; Joachim Folz; Andreas Dengel
>
> **摘要:** Coreset selection targets the challenge of finding a small, representative subset of a large dataset that preserves essential patterns for effective machine learning. Although several surveys have examined data reduction strategies before, most focus narrowly on either classical geometry-based methods or active learning techniques. In contrast, this survey presents a more comprehensive view by unifying three major lines of coreset research, namely, training-free, training-oriented, and label-free approaches, into a single taxonomy. We present subfields often overlooked by existing work, including submodular formulations, bilevel optimization, and recent progress in pseudo-labeling for unlabeled datasets. Additionally, we examine how pruning strategies influence generalization and neural scaling laws, offering new insights that are absent from prior reviews. Finally, we compare these methods under varying computational, robustness, and performance demands and highlight open challenges, such as robustness, outlier filtering, and adapting coreset selection to foundation models, for future research.
>
---
#### [new 162] VLM-KG: Multimodal Radiology Knowledge Graph Generation
- **分类: cs.CL; cs.CV; cs.IR; cs.LG**

- **简介: 该论文提出VLM-KG框架，属于多模态医学影像知识图谱生成任务。针对现有方法仅利用文本报告、忽略影像信息且难以处理长文本的问题，通过融合视觉-语言模型与影像-报告数据，构建首个多模态放射学知识图谱生成方案，提升生成效果。**

- **链接: [http://arxiv.org/pdf/2505.17042v1](http://arxiv.org/pdf/2505.17042v1)**

> **作者:** Abdullah Abdullah; Seong Tae Kim
>
> **备注:** 10 pages, 2 figures
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated remarkable success in natural language generation, excelling at instruction following and structured output generation. Knowledge graphs play a crucial role in radiology, serving as valuable sources of factual information and enhancing various downstream tasks. However, generating radiology-specific knowledge graphs presents significant challenges due to the specialized language of radiology reports and the limited availability of domain-specific data. Existing solutions are predominantly unimodal, meaning they generate knowledge graphs only from radiology reports while excluding radiographic images. Additionally, they struggle with long-form radiology data due to limited context length. To address these limitations, we propose a novel multimodal VLM-based framework for knowledge graph generation in radiology. Our approach outperforms previous methods and introduces the first multimodal solution for radiology knowledge graph generation.
>
---
#### [new 163] Accelerating Learned Image Compression Through Modeling Neural Training Dynamics
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于学习图像压缩（LIC）任务，旨在解决其训练效率低下的问题。提出STDET机制通过参数聚类和嵌入减少可训练参数，并结合SMA技术稳定训练过程，显著加速模型收敛且不损性能，理论分析验证了方法优势。**

- **链接: [http://arxiv.org/pdf/2505.18107v1](http://arxiv.org/pdf/2505.18107v1)**

> **作者:** Yichi Zhang; Zhihao Duan; Yuning Huang; Fengqing Zhu
>
> **备注:** Accepted to TMLR
>
> **摘要:** As learned image compression (LIC) methods become increasingly computationally demanding, enhancing their training efficiency is crucial. This paper takes a step forward in accelerating the training of LIC methods by modeling the neural training dynamics. We first propose a Sensitivity-aware True and Dummy Embedding Training mechanism (STDET) that clusters LIC model parameters into few separate modes where parameters are expressed as affine transformations of reference parameters within the same mode. By further utilizing the stable intra-mode correlations throughout training and parameter sensitivities, we gradually embed non-reference parameters, reducing the number of trainable parameters. Additionally, we incorporate a Sampling-then-Moving Average (SMA) technique, interpolating sampled weights from stochastic gradient descent (SGD) training to obtain the moving average weights, ensuring smooth temporal behavior and minimizing training state variances. Overall, our method significantly reduces training space dimensions and the number of trainable parameters without sacrificing model performance, thus accelerating model convergence. We also provide a theoretical analysis on the Noisy quadratic model, showing that the proposed method achieves a lower training variance than standard SGD. Our approach offers valuable insights for further developing efficient training methods for LICs.
>
---
#### [new 164] ComfyMind: Toward General-Purpose Generation via Tree-Based Planning and Reactive Feedback
- **分类: cs.AI; cs.CV**

- **简介: 论文提出ComfyMind，针对现有开源框架在复杂生成任务中的脆弱性及缺乏结构规划和反馈问题，通过语义工作流接口（SWI）抽象低级节点为自然语言模块，并结合搜索树规划与局部反馈机制，提升通用生成任务的稳定性和灵活性，性能接近GPT-Image-1。**

- **链接: [http://arxiv.org/pdf/2505.17908v1](http://arxiv.org/pdf/2505.17908v1)**

> **作者:** Litao Guo; Xinli Xu; Luozhou Wang; Jiantao Lin; Jinsong Zhou; Zixin Zhang; Bolan Su; Ying-Cong Chen
>
> **备注:** Project page: https://github.com/LitaoGuo/ComfyMind
>
> **摘要:** With the rapid advancement of generative models, general-purpose generation has gained increasing attention as a promising approach to unify diverse tasks across modalities within a single system. Despite this progress, existing open-source frameworks often remain fragile and struggle to support complex real-world applications due to the lack of structured workflow planning and execution-level feedback. To address these limitations, we present ComfyMind, a collaborative AI system designed to enable robust and scalable general-purpose generation, built on the ComfyUI platform. ComfyMind introduces two core innovations: Semantic Workflow Interface (SWI) that abstracts low-level node graphs into callable functional modules described in natural language, enabling high-level composition and reducing structural errors; Search Tree Planning mechanism with localized feedback execution, which models generation as a hierarchical decision process and allows adaptive correction at each stage. Together, these components improve the stability and flexibility of complex generative workflows. We evaluate ComfyMind on three public benchmarks: ComfyBench, GenEval, and Reason-Edit, which span generation, editing, and reasoning tasks. Results show that ComfyMind consistently outperforms existing open-source baselines and achieves performance comparable to GPT-Image-1. ComfyMind paves a promising path for the development of open-source general-purpose generative AI systems. Project page: https://github.com/LitaoGuo/ComfyMind
>
---
#### [new 165] Mahalanobis++: Improving OOD Detection via Feature Normalization
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于出分布检测（OOD detection）任务，旨在解决基于Mahalanobis距离的OOD方法在不同模型间性能波动大的问题。因原始特征范数差异大导致高斯假设失效，作者提出通过ℓ₂归一化特征使其符合正态分布假设，实验表明该方法显著提升OOD检测效果，超越其他现有方法。**

- **链接: [http://arxiv.org/pdf/2505.18032v1](http://arxiv.org/pdf/2505.18032v1)**

> **作者:** Maximilian Mueller; Matthias Hein
>
> **摘要:** Detecting out-of-distribution (OOD) examples is an important task for deploying reliable machine learning models in safety-critial applications. While post-hoc methods based on the Mahalanobis distance applied to pre-logit features are among the most effective for ImageNet-scale OOD detection, their performance varies significantly across models. We connect this inconsistency to strong variations in feature norms, indicating severe violations of the Gaussian assumption underlying the Mahalanobis distance estimation. We show that simple $\ell_2$-normalization of the features mitigates this problem effectively, aligning better with the premise of normally distributed data with shared covariance matrix. Extensive experiments on 44 models across diverse architectures and pretraining schemes show that $\ell_2$-normalization improves the conventional Mahalanobis distance-based approaches significantly and consistently, and outperforms other recently proposed OOD detection methods.
>
---
#### [new 166] Towards Prospective Medical Image Reconstruction via Knowledge-Informed Dynamic Optimal Transport
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像重建任务，旨在解决模拟训练数据与真实前瞻性数据间的性能差距。提出KIDOT框架，通过知识引导的动态最优传输，利用未配对数据学习符合物理规律的连续演化路径，提升重建鲁棒性和泛化能力。（99字）**

- **链接: [http://arxiv.org/pdf/2505.17644v1](http://arxiv.org/pdf/2505.17644v1)**

> **作者:** Taoran Zheng; Xing Li; Yan Yang; Xiang Gu; Zongben Xu; Jian Sun
>
> **摘要:** Medical image reconstruction from measurement data is a vital but challenging inverse problem. Deep learning approaches have achieved promising results, but often requires paired measurement and high-quality images, which is typically simulated through a forward model, i.e., retrospective reconstruction. However, training on simulated pairs commonly leads to performance degradation on real prospective data due to the retrospective-to-prospective gap caused by incomplete imaging knowledge in simulation. To address this challenge, this paper introduces imaging Knowledge-Informed Dynamic Optimal Transport (KIDOT), a novel dynamic optimal transport framework with optimality in the sense of preserving consistency with imaging physics in transport, that conceptualizes reconstruction as finding a dynamic transport path. KIDOT learns from unpaired data by modeling reconstruction as a continuous evolution path from measurements to images, guided by an imaging knowledge-informed cost function and transport equation. This dynamic and knowledge-aware approach enhances robustness and better leverages unpaired data while respecting acquisition physics. Theoretically, we demonstrate that KIDOT naturally generalizes dynamic optimal transport, ensuring its mathematical rationale and solution existence. Extensive experiments on MRI and CT reconstruction demonstrate KIDOT's superior performance.
>
---
#### [new 167] Wildfire spread forecasting with Deep Learning
- **分类: cs.LG; cs.CV; I.2.7**

- **简介: 该论文提出基于深度学习的野火蔓延预测框架，利用多源时空数据（遥感、气象、植被等）提升烧毁范围预测精度。通过消融实验验证多日数据对模型性能的提升作用，最佳模型整合四天前至五天后的观测数据，较基线模型F1和IoU提升近5%，并公开数据与模型。**

- **链接: [http://arxiv.org/pdf/2505.17556v1](http://arxiv.org/pdf/2505.17556v1)**

> **作者:** Nikolaos Anastasiou; Spyros Kondylatos; Ioannis Papoutsis
>
> **备注:** 10 pages, 9 figures
>
> **摘要:** Accurate prediction of wildfire spread is crucial for effective risk management, emergency response, and strategic resource allocation. In this study, we present a deep learning (DL)-based framework for forecasting the final extent of burned areas, using data available at the time of ignition. We leverage a spatio-temporal dataset that covers the Mediterranean region from 2006 to 2022, incorporating remote sensing data, meteorological observations, vegetation maps, land cover classifications, anthropogenic factors, topography data, and thermal anomalies. To evaluate the influence of temporal context, we conduct an ablation study examining how the inclusion of pre- and post-ignition data affects model performance, benchmarking the temporal-aware DL models against a baseline trained exclusively on ignition-day inputs. Our results indicate that multi-day observational data substantially improve predictive accuracy. Particularly, the best-performing model, incorporating a temporal window of four days before to five days after ignition, improves both the F1 score and the Intersection over Union by almost 5% in comparison to the baseline on the test dataset. We publicly release our dataset and models to enhance research into data-driven approaches for wildfire modeling and response.
>
---
#### [new 168] From Flight to Insight: Semantic 3D Reconstruction for Aerial Inspection via Gaussian Splatting and Language-Guided Segmentation
- **分类: cs.GR; cs.CV; eess.IV**

- **简介: 该论文提出一种基于无人机的语义3D重建方法，结合高斯散射与语言引导分割，解决传统三维重建缺乏语义理解的问题。通过融合CLIP嵌入的特征场生成语义热图，并利用SAM模型优化分割，实现语言驱动的精细场景解析，提升空中检测任务（如基建监测）的自动化能力。**

- **链接: [http://arxiv.org/pdf/2505.17402v1](http://arxiv.org/pdf/2505.17402v1)**

> **作者:** Mahmoud Chick Zaouali; Todd Charter; Homayoun Najjaran
>
> **摘要:** High-fidelity 3D reconstruction is critical for aerial inspection tasks such as infrastructure monitoring, structural assessment, and environmental surveying. While traditional photogrammetry techniques enable geometric modeling, they lack semantic interpretability, limiting their effectiveness for automated inspection workflows. Recent advances in neural rendering and 3D Gaussian Splatting (3DGS) offer efficient, photorealistic reconstructions but similarly lack scene-level understanding. In this work, we present a UAV-based pipeline that extends Feature-3DGS for language-guided 3D segmentation. We leverage LSeg-based feature fields with CLIP embeddings to generate heatmaps in response to language prompts. These are thresholded to produce rough segmentations, and the highest-scoring point is then used as a prompt to SAM or SAM2 for refined 2D segmentation on novel view renderings. Our results highlight the strengths and limitations of various feature field backbones (CLIP-LSeg, SAM, SAM2) in capturing meaningful structure in large-scale outdoor environments. We demonstrate that this hybrid approach enables flexible, language-driven interaction with photorealistic 3D reconstructions, opening new possibilities for semantic aerial inspection and scene understanding.
>
---
#### [new 169] FastCAV: Efficient Computation of Concept Activation Vectors for Explaining Deep Neural Networks
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 论文提出FastCAV方法，解决传统CAV计算效率低的问题，通过理论优化与算法加速（平均提速46.4倍），保持性能同时实现高维模型概念分析的高效性，支持模型训练中概念演化的可解释性研究。**

- **链接: [http://arxiv.org/pdf/2505.17883v1](http://arxiv.org/pdf/2505.17883v1)**

> **作者:** Laines Schmalwasser; Niklas Penzel; Joachim Denzler; Julia Niebling
>
> **备注:** Accepted at ICML 2025, 27 pages, 20 figures, 9 tables
>
> **摘要:** Concepts such as objects, patterns, and shapes are how humans understand the world. Building on this intuition, concept-based explainability methods aim to study representations learned by deep neural networks in relation to human-understandable concepts. Here, Concept Activation Vectors (CAVs) are an important tool and can identify whether a model learned a concept or not. However, the computational cost and time requirements of existing CAV computation pose a significant challenge, particularly in large-scale, high-dimensional architectures. To address this limitation, we introduce FastCAV, a novel approach that accelerates the extraction of CAVs by up to 63.6x (on average 46.4x). We provide a theoretical foundation for our approach and give concrete assumptions under which it is equivalent to established SVM-based methods. Our empirical results demonstrate that CAVs calculated with FastCAV maintain similar performance while being more efficient and stable. In downstream applications, i.e., concept-based explanation methods, we show that FastCAV can act as a replacement leading to equivalent insights. Hence, our approach enables previously infeasible investigations of deep models, which we demonstrate by tracking the evolution of concepts during model training.
>
---
#### [new 170] RAVEN: Query-Guided Representation Alignment for Question Answering over Audio, Video, Embedded Sensors, and Natural Language
- **分类: cs.CL; cs.CV; cs.LG; cs.MM**

- **简介: 该论文属于多模态问答任务，解决模态间干扰导致模型误判的问题。提出RAVEN模型，通过查询引导的跨模态门控（QuART）动态分配模态token相关性分数，结合三阶段训练策略，提升问答准确性与抗干扰能力，并发布AVS-QA数据集，实验显示显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.17114v1](http://arxiv.org/pdf/2505.17114v1)**

> **作者:** Subrata Biswas; Mohammad Nur Hossain Khan; Bashima Islam
>
> **摘要:** Multimodal question answering (QA) often requires identifying which video, audio, or sensor tokens are relevant to the question. Yet modality disagreements are common: off-camera speech, background noise, or motion outside the field of view often mislead fusion models that weight all streams equally. We present RAVEN, a unified QA architecture whose core is QuART, a query-conditioned cross-modal gating module that assigns scalar relevance scores to each token across modalities, enabling the model to amplify informative signals and suppress distractors before fusion. RAVEN is trained through a three-stage pipeline comprising unimodal pretraining, query-aligned fusion, and disagreement-oriented fine-tuning -- each stage targeting a distinct challenge in multi-modal reasoning: representation quality, cross-modal relevance, and robustness to modality mismatch. To support training and evaluation, we release AVS-QA, a dataset of 300K synchronized Audio--Video-Sensor streams paired with automatically generated question-answer pairs. Experimental results on seven multi-modal QA benchmarks -- including egocentric and exocentric tasks -- show that RAVEN achieves up to 14.5\% and 8.0\% gains in accuracy compared to state-of-the-art multi-modal large language models, respectively. Incorporating sensor data provides an additional 16.4\% boost, and the model remains robust under modality corruption, outperforming SOTA baselines by 50.23\%. Our code and dataset are available at https://github.com/BASHLab/RAVEN.
>
---
#### [new 171] UltraBoneUDF: Self-supervised Bone Surface Reconstruction from Ultrasound Based on Neural Unsigned Distance Functions
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出UltraBoneUDF框架，解决超声骨表面重建中数据不完整导致的误差问题。通过自监督神经无符号距离函数，设计融合超声特征的全局提取器及局部切线面优化损失函数，显著提升开放与闭合骨表面重建精度，在四数据集上误差降低39.6%-70.2%。**

- **链接: [http://arxiv.org/pdf/2505.17912v1](http://arxiv.org/pdf/2505.17912v1)**

> **作者:** Luohong Wu; Matthias Seibold; Nicola A. Cavalcanti; Giuseppe Loggia; Lisa Reissner; Bastian Sigrist; Jonas Hein; Lilian Calvet; Arnd Viehöfer; Philipp Fürnstahl
>
> **摘要:** Background: Bone surface reconstruction plays a critical role in computer-assisted orthopedic surgery. Compared to traditional imaging modalities such as CT and MRI, ultrasound offers a radiation-free, cost-effective, and portable alternative. Continuous bone surface reconstruction can be employed for many clinical applications. However, due to the inherent limitations of ultrasound imaging, B-mode ultrasound typically capture only partial bone surfaces. Existing reconstruction methods struggle with such incomplete data, leading to artifacts and increased reconstruction errors. Effective techniques for accurately reconstructing thin and open bone surfaces from real-world 3D ultrasound volumes remain lacking. Methods: We propose UltraBoneUDF, a self-supervised framework designed for reconstructing open bone surfaces from ultrasound using neural Unsigned Distance Functions. To enhance reconstruction quality, we introduce a novel global feature extractor that effectively fuses ultrasound-specific image characteristics. Additionally, we present a novel loss function based on local tangent plane optimization that substantially improves surface reconstruction quality. UltraBoneUDF and baseline models are extensively evaluated on four open-source datasets. Results: Qualitative results highlight the limitations of the state-of-the-art methods for open bone surface reconstruction and demonstrate the effectiveness of UltraBoneUDF. Quantitatively, UltraBoneUDF significantly outperforms competing methods across all evaluated datasets for both open and closed bone surface reconstruction in terms of mean Chamfer distance error: 1.10 mm on the UltraBones100k dataset (39.6\% improvement compared to the SOTA), 0.23 mm on the OpenBoneCT dataset (69.3\% improvement), 0.18 mm on the ClosedBoneCT dataset (70.2\% improvement), and 0.05 mm on the Prostate dataset (55.3\% improvement).
>
---
#### [new 172] CRG Score: A Distribution-Aware Clinical Metric for Radiology Report Generation
- **分类: cs.CL; cs.CV**

- **简介: 该论文针对放射报告生成评估任务，解决现有指标无法准确衡量临床正确性及受类别不平衡影响的问题。提出CRG Score，通过关注参考报告中明确的临床异常、结合分布平衡罚则及结构化标签，实现更公平、鲁棒的临床对齐评估。**

- **链接: [http://arxiv.org/pdf/2505.17167v1](http://arxiv.org/pdf/2505.17167v1)**

> **作者:** Ibrahim Ethem Hamamci; Sezgin Er; Suprosanna Shit; Hadrien Reynaud; Bernhard Kainz; Bjoern Menze
>
> **摘要:** Evaluating long-context radiology report generation is challenging. NLG metrics fail to capture clinical correctness, while LLM-based metrics often lack generalizability. Clinical accuracy metrics are more relevant but are sensitive to class imbalance, frequently favoring trivial predictions. We propose the CRG Score, a distribution-aware and adaptable metric that evaluates only clinically relevant abnormalities explicitly described in reference reports. CRG supports both binary and structured labels (e.g., type, location) and can be paired with any LLM for feature extraction. By balancing penalties based on label distribution, it enables fairer, more robust evaluation and serves as a clinically aligned reward function.
>
---
#### [new 173] Large Language Models Implicitly Learn to See and Hear Just By Reading
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于多模态理解任务，探索文本LLM是否能通过预训练直接处理图像/音频。研究发现仅用文本训练的自回归LLM可内在学习跨模态理解能力，输入图像块或音频波形即可输出分类结果，在FSD-50K、CIFAR-10等数据集验证，证明文本模型内部形成通用感知模块，无需针对新任务从头训练。**

- **链接: [http://arxiv.org/pdf/2505.17091v1](http://arxiv.org/pdf/2505.17091v1)**

> **作者:** Prateek Verma; Mert Pilanci
>
> **备注:** 6 pages, 3 figures, 4 tables. Under Review WASPAA 2025
>
> **摘要:** This paper presents a fascinating find: By training an auto-regressive LLM model on text tokens, the text model inherently develops internally an ability to understand images and audio, thereby developing the ability to see and hear just by reading. Popular audio and visual LLM models fine-tune text LLM models to give text output conditioned on images and audio embeddings. On the other hand, our architecture takes in patches of images, audio waveforms or tokens as input. It gives us the embeddings or category labels typical of a classification pipeline. We show the generality of text weights in aiding audio classification for datasets FSD-50K and GTZAN. Further, we show this working for image classification on CIFAR-10 and Fashion-MNIST, as well on image patches. This pushes the notion of text-LLMs learning powerful internal circuits that can be utilized by activating necessary connections for various applications rather than training models from scratch every single time.
>
---
## 更新

#### [replaced 001] LED: LLM Enhanced Open-Vocabulary Object Detection without Human Curated Data Generation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.13794v3](http://arxiv.org/pdf/2503.13794v3)**

> **作者:** Yang Zhou; Shiyu Zhao; Yuxiao Chen; Zhenting Wang; Can Jin; Dimitris N. Metaxas
>
> **摘要:** Large foundation models trained on large-scale vision-language data can boost Open-Vocabulary Object Detection (OVD) via synthetic training data, yet the hand-crafted pipelines often introduce bias and overfit to specific prompts. We sidestep this issue by directly fusing hidden states from Large Language Models (LLMs) into detectors-an avenue surprisingly under-explored. This paper presents a systematic method to enhance visual grounding by utilizing decoder layers of the LLM of an MLLM. We introduce a zero-initialized cross-attention adapter to enable efficient knowledge fusion from LLMs to object detectors, a new approach called LED (LLM Enhanced Open-Vocabulary Object Detection). We find that intermediate LLM layers already encode rich spatial semantics; adapting only the early layers yields most of the gain. With Swin-T as the vision encoder, Qwen2-0.5B + LED lifts GroundingDINO by 3.82 % on OmniLabel at just 8.7 % extra GFLOPs, and a larger vision backbone pushes the improvement to 6.22 %. Extensive ablations on adapter variants, LLM scales and fusion depths further corroborate our design.
>
---
#### [replaced 002] Panoptic Captioning: Seeking An Equivalency Bridge for Image and Text
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16334v2](http://arxiv.org/pdf/2505.16334v2)**

> **作者:** Kun-Yu Lin; Hongjun Wang; Weining Ren; Kai Han
>
> **备注:** Project page: https://visual-ai.github.io/pancap/
>
> **摘要:** This work introduces panoptic captioning, a novel task striving to seek the minimum text equivalence of images. We take the first step towards panoptic captioning by formulating it as a task of generating a comprehensive textual description for an image, which encapsulates all entities, their respective locations and attributes, relationships among entities, as well as global image state. Through an extensive evaluation, our work reveals that state-of-the-art Multi-modal Large Language Models (MLLMs) have limited performance in solving panoptic captioning. To address this, we propose an effective data engine named PancapEngine to produce high-quality data and a novel method named PancapChain to improve panoptic captioning. Specifically, our PancapEngine first detects diverse categories of entities in images by an elaborate detection suite, and then generates required panoptic captions using entity-aware prompts. Additionally, our PancapChain explicitly decouples the challenging panoptic captioning task into multiple stages and generates panoptic captions step by step. More importantly, we contribute a comprehensive metric named PancapScore and a human-curated test set for reliable model evaluation. Experiments show that our PancapChain-13B model can beat state-of-the-art open-source MLLMs like InternVL-2.5-78B and even surpass proprietary models like GPT-4o and Gemini-2.0-Pro, demonstrating the effectiveness of our data engine and method. Project page: https://visual-ai.github.io/pancap/
>
---
#### [replaced 003] MMInference: Accelerating Pre-filling for Long-Context VLMs via Modality-Aware Permutation Sparse Attention
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.16083v2](http://arxiv.org/pdf/2504.16083v2)**

> **作者:** Yucheng Li; Huiqiang Jiang; Chengruidong Zhang; Qianhui Wu; Xufang Luo; Surin Ahn; Amir H. Abdi; Dongsheng Li; Jianfeng Gao; Yuqing Yang; Lili Qiu
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** The integration of long-context capabilities with visual understanding unlocks unprecedented potential for Vision Language Models (VLMs). However, the quadratic attention complexity during the pre-filling phase remains a significant obstacle to real-world deployment. To overcome this limitation, we introduce MMInference (Multimodality Million tokens Inference), a dynamic sparse attention method that accelerates the prefilling stage for long-context multi-modal inputs. First, our analysis reveals that the temporal and spatial locality of video input leads to a unique sparse pattern, the Grid pattern. Simultaneously, VLMs exhibit markedly different sparse distributions across different modalities. We introduce a permutation-based method to leverage the unique Grid pattern and handle modality boundary issues. By offline search the optimal sparse patterns for each head, MMInference constructs the sparse distribution dynamically based on the input. We also provide optimized GPU kernels for efficient sparse computations. Notably, MMInference integrates seamlessly into existing VLM pipelines without any model modifications or fine-tuning. Experiments on multi-modal benchmarks-including Video QA, Captioning, VisionNIAH, and Mixed-Modality NIAH-with state-of-the-art long-context VLMs (LongVila, LlavaVideo, VideoChat-Flash, Qwen2.5-VL) show that MMInference accelerates the pre-filling stage by up to 8.3x at 1M tokens while maintaining accuracy. Our code is available at https://aka.ms/MMInference.
>
---
#### [replaced 004] Challenger: Affordable Adversarial Driving Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15880v2](http://arxiv.org/pdf/2505.15880v2)**

> **作者:** Zhiyuan Xu; Bohan Li; Huan-ang Gao; Mingju Gao; Yong Chen; Ming Liu; Chenxu Yan; Hang Zhao; Shuo Feng; Hao Zhao
>
> **备注:** Project page: https://pixtella.github.io/Challenger/
>
> **摘要:** Generating photorealistic driving videos has seen significant progress recently, but current methods largely focus on ordinary, non-adversarial scenarios. Meanwhile, efforts to generate adversarial driving scenarios often operate on abstract trajectory or BEV representations, falling short of delivering realistic sensor data that can truly stress-test autonomous driving (AD) systems. In this work, we introduce Challenger, a framework that produces physically plausible yet photorealistic adversarial driving videos. Generating such videos poses a fundamental challenge: it requires jointly optimizing over the space of traffic interactions and high-fidelity sensor observations. Challenger makes this affordable through two techniques: (1) a physics-aware multi-round trajectory refinement process that narrows down candidate adversarial maneuvers, and (2) a tailored trajectory scoring function that encourages realistic yet adversarial behavior while maintaining compatibility with downstream video synthesis. As tested on the nuScenes dataset, Challenger generates a diverse range of aggressive driving scenarios-including cut-ins, sudden lane changes, tailgating, and blind spot intrusions-and renders them into multiview photorealistic videos. Extensive evaluations show that these scenarios significantly increase the collision rate of state-of-the-art end-to-end AD models (UniAD, VAD, SparseDrive, and DiffusionDrive), and importantly, adversarial behaviors discovered for one model often transfer to others.
>
---
#### [replaced 005] Forensics Adapter: Unleashing CLIP for Generalizable Face Forgery Detection
- **分类: cs.CV; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.19715v3](http://arxiv.org/pdf/2411.19715v3)**

> **作者:** Xinjie Cui; Yuezun Li; Delong Zhu; Jiaran Zhou; Junyu Dong; Siwei Lyu
>
> **备注:** Extension of CVPR 2025
>
> **摘要:** We describe Forensics Adapter, an adapter network designed to transform CLIP into an effective and generalizable face forgery detector. Although CLIP is highly versatile, adapting it for face forgery detection is non-trivial as forgery-related knowledge is entangled with a wide range of unrelated knowledge. Existing methods treat CLIP merely as a feature extractor, lacking task-specific adaptation, which limits their effectiveness. To address this, we introduce an adapter to learn face forgery traces -- the blending boundaries unique to forged faces, guided by task-specific objectives. Then we enhance the CLIP visual tokens with a dedicated interaction strategy that communicates knowledge across CLIP and the adapter. Since the adapter is alongside CLIP, its versatility is highly retained, naturally ensuring strong generalizability in face forgery detection. With only 5.7M trainable parameters, our method achieves a significant performance boost, improving by approximately 7% on average across five standard datasets. Additionally, we describe Forensics Adapter++, an extended method that incorporates textual modality via a newly proposed forgery-aware prompt learning strategy. This extension leads to a further 1.3% performance boost over the original Forensics Adapter. We believe the proposed methods can serve as a baseline for future CLIP-based face forgery detection methods. The codes have been released at https://github.com/OUC-VAS/ForensicsAdapter.
>
---
#### [replaced 006] Explaining Black-box Model Predictions via Two-level Nested Feature Attributions with Consistency Property
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2405.14522v2](http://arxiv.org/pdf/2405.14522v2)**

> **作者:** Yuya Yoshikawa; Masanari Kimura; Ryotaro Shimizu; Yuki Saito
>
> **备注:** This manuscript is an extended version of our paper accepted at IJCAI2025, with detailed proofs and additional experimental results
>
> **摘要:** Techniques that explain the predictions of black-box machine learning models are crucial to make the models transparent, thereby increasing trust in AI systems. The input features to the models often have a nested structure that consists of high- and low-level features, and each high-level feature is decomposed into multiple low-level features. For such inputs, both high-level feature attributions (HiFAs) and low-level feature attributions (LoFAs) are important for better understanding the model's decision. In this paper, we propose a model-agnostic local explanation method that effectively exploits the nested structure of the input to estimate the two-level feature attributions simultaneously. A key idea of the proposed method is to introduce the consistency property that should exist between the HiFAs and LoFAs, thereby bridging the separate optimization problems for estimating them. Thanks to this consistency property, the proposed method can produce HiFAs and LoFAs that are both faithful to the black-box models and consistent with each other, using a smaller number of queries to the models. In experiments on image classification in multiple instance learning and text classification using language models, we demonstrate that the HiFAs and LoFAs estimated by the proposed method are accurate, faithful to the behaviors of the black-box models, and provide consistent explanations.
>
---
#### [replaced 007] MedCFVQA: A Causal Approach to Mitigate Modality Preference Bias in Medical Visual Question Answering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16209v2](http://arxiv.org/pdf/2505.16209v2)**

> **作者:** Shuchang Ye; Usman Naseem; Mingyuan Meng; Dagan Feng; Jinman Kim
>
> **摘要:** Medical Visual Question Answering (MedVQA) is crucial for enhancing the efficiency of clinical diagnosis by providing accurate and timely responses to clinicians' inquiries regarding medical images. Existing MedVQA models suffered from modality preference bias, where predictions are heavily dominated by one modality while overlooking the other (in MedVQA, usually questions dominate the answer but images are overlooked), thereby failing to learn multimodal knowledge. To overcome the modality preference bias, we proposed a Medical CounterFactual VQA (MedCFVQA) model, which trains with bias and leverages causal graphs to eliminate the modality preference bias during inference. Existing MedVQA datasets exhibit substantial prior dependencies between questions and answers, which results in acceptable performance even if the model significantly suffers from the modality preference bias. To address this issue, we reconstructed new datasets by leveraging existing MedVQA datasets and Changed their P3rior dependencies (CP) between questions and their answers in the training and test set. Extensive experiments demonstrate that MedCFVQA significantly outperforms its non-causal counterpart on both SLAKE, RadVQA and SLAKE-CP, RadVQA-CP datasets.
>
---
#### [replaced 008] LaViDa: A Large Diffusion Language Model for Multimodal Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16839v2](http://arxiv.org/pdf/2505.16839v2)**

> **作者:** Shufan Li; Konstantinos Kallidromitis; Hritik Bansal; Akash Gokul; Yusuke Kato; Kazuki Kozuka; Jason Kuen; Zhe Lin; Kai-Wei Chang; Aditya Grover
>
> **备注:** 25 pages, 8 figures
>
> **摘要:** Modern Vision-Language Models (VLMs) can solve a wide range of tasks requiring visual reasoning. In real-world scenarios, desirable properties for VLMs include fast inference and controllable generation (e.g., constraining outputs to adhere to a desired format). However, existing autoregressive (AR) VLMs like LLaVA struggle in these aspects. Discrete diffusion models (DMs) offer a promising alternative, enabling parallel decoding for faster inference and bidirectional context for controllable generation through text-infilling. While effective in language-only settings, DMs' potential for multimodal tasks is underexplored. We introduce LaViDa, a family of VLMs built on DMs. We build LaViDa by equipping DMs with a vision encoder and jointly fine-tune the combined parts for multimodal instruction following. To address challenges encountered, LaViDa incorporates novel techniques such as complementary masking for effective training, prefix KV cache for efficient inference, and timestep shifting for high-quality sampling. Experiments show that LaViDa achieves competitive or superior performance to AR VLMs on multi-modal benchmarks such as MMMU, while offering unique advantages of DMs, including flexible speed-quality tradeoff, controllability, and bidirectional reasoning. On COCO captioning, LaViDa surpasses Open-LLaVa-Next-8B by +4.1 CIDEr with 1.92x speedup. On bidirectional tasks, it achieves +59% improvement on Constrained Poem Completion. These results demonstrate LaViDa as a strong alternative to AR VLMs. Code and models will be released in the camera-ready version.
>
---
#### [replaced 009] ReactDiff: Latent Diffusion for Facial Reaction Generation
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.14151v2](http://arxiv.org/pdf/2505.14151v2)**

> **作者:** Jiaming Li; Sheng Wang; Xin Wang; Yitao Zhu; Honglin Xiong; Zixu Zhuang; Qian Wang
>
> **备注:** Neural Networks
>
> **摘要:** Given the audio-visual clip of the speaker, facial reaction generation aims to predict the listener's facial reactions. The challenge lies in capturing the relevance between video and audio while balancing appropriateness, realism, and diversity. While prior works have mostly focused on uni-modal inputs or simplified reaction mappings, recent approaches such as PerFRDiff have explored multi-modal inputs and the one-to-many nature of appropriate reaction mappings. In this work, we propose the Facial Reaction Diffusion (ReactDiff) framework that uniquely integrates a Multi-Modality Transformer with conditional diffusion in the latent space for enhanced reaction generation. Unlike existing methods, ReactDiff leverages intra- and inter-class attention for fine-grained multi-modal interaction, while the latent diffusion process between the encoder and decoder enables diverse yet contextually appropriate outputs. Experimental results demonstrate that ReactDiff significantly outperforms existing approaches, achieving a facial reaction correlation of 0.26 and diversity score of 0.094 while maintaining competitive realism. The code is open-sourced at \href{https://github.com/Hunan-Tiger/ReactDiff}{github}.
>
---
#### [replaced 010] ViFOR: A Fourier-Enhanced Vision Transformer for Multi-Image Super-Resolution in Earth System
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.12427v2](http://arxiv.org/pdf/2502.12427v2)**

> **作者:** Ehsan Zeraatkar; Salah A Faroughi; Jelena Tešić
>
> **摘要:** Super-resolution (SR) techniques are essential for improving Earth System Model (ESM) data's spatial resolution, which helps better understand complex environmental processes. This paper presents a new algorithm, ViFOR, which combines Vision Transformers (ViT) and Fourier-based Implicit Neural Representation Networks (INRs) to generate High-Resolution (HR) images from Low-Resolution (LR) inputs. ViFOR introduces a novel integration of Fourier-based activation functions within the Vision Transformer architecture, enabling it to effectively capture global context and high-frequency details critical for accurate SR reconstruction. The results show that ViFOR outperforms state-of-the-art methods such as ViT, Sinusoidal Representation Networks (SIREN), and SR Generative Adversarial Networks (SRGANs) based on metrics like Peak Signal-to-Noise Ratio (PSNR) and Mean Squared Error (MSE) both for global as well as the local imagery. ViFOR improves PSNR of up to 4.18 dB, 1.56 dB, and 1.73 dB over ViT for full images in the Source Temperature, Shortwave, and Longwave Flux.
>
---
#### [replaced 011] Decoupled Geometric Parameterization and its Application in Deep Homography Estimation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16599v2](http://arxiv.org/pdf/2505.16599v2)**

> **作者:** Yao Huang; Si-Yuan Cao; Yaqing Ding; Hao Yin; Shibin Xie; Shuting Wang; Zhijun Fang; Jiachun Wang; Shen Cai; Junchi Yan; Shuhan Shen
>
> **摘要:** Planar homography, with eight degrees of freedom (DOFs), is fundamental in numerous computer vision tasks. While the positional offsets of four corners are widely adopted (especially in neural network predictions), this parameterization lacks geometric interpretability and typically requires solving a linear system to compute the homography matrix. This paper presents a novel geometric parameterization of homographies, leveraging the similarity-kernel-similarity (SKS) decomposition for projective transformations. Two independent sets of four geometric parameters are decoupled: one for a similarity transformation and the other for the kernel transformation. Additionally, the geometric interpretation linearly relating the four kernel transformation parameters to angular offsets is derived. Our proposed parameterization allows for direct homography estimation through matrix multiplication, eliminating the need for solving a linear system, and achieves performance comparable to the four-corner positional offsets in deep homography estimation.
>
---
#### [replaced 012] A Clinician-Friendly Platform for Ophthalmic Image Analysis Without Technical Barriers
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.15928v2](http://arxiv.org/pdf/2504.15928v2)**

> **作者:** Meng Wang; Tian Lin; Qingshan Hou; Aidi Lin; Jingcheng Wang; Qingsheng Peng; Truong X. Nguyen; Danqi Fang; Ke Zou; Ting Xu; Cancan Xue; Ten Cheer Quek; Qinkai Yu; Minxin Liu; Hui Zhou; Zixuan Xiao; Guiqin He; Huiyu Liang; Tingkun Shi; Man Chen; Linna Liu; Yuanyuan Peng; Lianyu Wang; Qiuming Hu; Junhong Chen; Zhenhua Zhang; Cheng Chen; Yitian Zhao; Dianbo Liu; Jianhua Wu; Xinjian Chen; Changqing Zhang; Triet Thanh Nguyen; Yanda Meng; Yalin Zheng; Yih Chung Tham; Carol Y. Cheung; Huazhu Fu; Haoyu Chen; Ching-Yu Cheng
>
> **摘要:** Artificial intelligence (AI) shows remarkable potential in medical imaging diagnostics, yet most current models require retraining when applied across different clinical settings, limiting their scalability. We introduce GlobeReady, a clinician-friendly AI platform that enables fundus disease diagnosis that operates without retraining, fine-tuning, or the needs for technical expertise. GlobeReady demonstrates high accuracy across imaging modalities: 93.9-98.5% for 11 fundus diseases using color fundus photographs (CPFs) and 87.2-92.7% for 15 fundus diseases using optic coherence tomography (OCT) scans. By leveraging training-free local feature augmentation, GlobeReady platform effectively mitigates domain shifts across centers and populations, achieving accuracies of 88.9-97.4% across five centers on average in China, 86.3-96.9% in Vietnam, and 73.4-91.0% in Singapore, and 90.2-98.9% in the UK. Incorporating a bulit-in confidence-quantifiable diagnostic mechanism further enhances the platform's accuracy to 94.9-99.4% with CFPs and 88.2-96.2% with OCT, while enabling identification of out-of-distribution cases with 86.3% accuracy across 49 common and rare fundus diseases using CFPs, and 90.6% accuracy across 13 diseases using OCT. Clinicians from countries rated GlobeReady highly for usability and clinical relevance (average score 4.6/5). These findings demonstrate GlobeReady's robustness, generalizability and potential to support global ophthalmic care without technical barriers.
>
---
#### [replaced 013] ChatStitch: Visualizing Through Structures via Surround-View Unsupervised Deep Image Stitching with Collaborative LLM-Agents
- **分类: cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2503.14948v2](http://arxiv.org/pdf/2503.14948v2)**

> **作者:** Hao Liang; Zhipeng Dong; Kaixin Chen; Jiyuan Guo; Yufeng Yue; Yi Yang; Mengyin Fu
>
> **摘要:** Surround-view perception has garnered significant attention for its ability to enhance the perception capabilities of autonomous driving vehicles through the exchange of information with surrounding cameras. However, existing surround-view perception systems are limited by inefficiencies in unidirectional interaction pattern with human and distortions in overlapping regions exponentially propagating into non-overlapping areas. To address these challenges, this paper introduces ChatStitch, a surround-view human-machine co-perception system capable of unveiling obscured blind spot information through natural language commands integrated with external digital assets. To dismantle the unidirectional interaction bottleneck, ChatStitch implements a cognitively grounded closed-loop interaction multi-agent framework based on Large Language Models. To suppress distortion propagation across overlapping boundaries, ChatStitch proposes SV-UDIS, a surround-view unsupervised deep image stitching method under the non-global-overlapping condition. We conducted extensive experiments on the UDIS-D, MCOV-SLAM open datasets, and our real-world dataset. Specifically, our SV-UDIS method achieves state-of-the-art performance on the UDIS-D dataset for 3, 4, and 5 image stitching tasks, with PSNR improvements of 9\%, 17\%, and 21\%, and SSIM improvements of 8\%, 18\%, and 26\%, respectively.
>
---
#### [replaced 014] HumaniBench: A Human-Centric Framework for Large Multimodal Models Evaluation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.11454v2](http://arxiv.org/pdf/2505.11454v2)**

> **作者:** Shaina Raza; Aravind Narayanan; Vahid Reza Khazaie; Ashmal Vayani; Mukund S. Chettiar; Amandeep Singh; Mubarak Shah; Deval Pandya
>
> **摘要:** Large multimodal models (LMMs) now excel on many vision language benchmarks, however, they still struggle with human centered criteria such as fairness, ethics, empathy, and inclusivity, key to aligning with human values. We introduce HumaniBench, a holistic benchmark of 32K real-world image question pairs, annotated via a scalable GPT4o assisted pipeline and exhaustively verified by domain experts. HumaniBench evaluates seven Human Centered AI (HCAI) principles: fairness, ethics, understanding, reasoning, language inclusivity, empathy, and robustness, across seven diverse tasks, including open and closed ended visual question answering (VQA), multilingual QA, visual grounding, empathetic captioning, and robustness tests. Benchmarking 15 state of the art LMMs (open and closed source) reveals that proprietary models generally lead, though robustness and visual grounding remain weak points. Some open-source models also struggle to balance accuracy with adherence to human-aligned principles. HumaniBench is the first benchmark purpose built around HCAI principles. It provides a rigorous testbed for diagnosing alignment gaps and guiding LMMs toward behavior that is both accurate and socially responsible. Dataset, annotation prompts, and evaluation code are available at: https://vectorinstitute.github.io/HumaniBench
>
---
#### [replaced 015] Q-Insight: Understanding Image Quality via Visual Reinforcement Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.22679v2](http://arxiv.org/pdf/2503.22679v2)**

> **作者:** Weiqi Li; Xuanyu Zhang; Shijie Zhao; Yabin Zhang; Junlin Li; Li Zhang; Jian Zhang
>
> **摘要:** Image quality assessment (IQA) focuses on the perceptual visual quality of images, playing a crucial role in downstream tasks such as image reconstruction, compression, and generation. The rapid advancement of multi-modal large language models (MLLMs) has significantly broadened the scope of IQA, moving toward comprehensive image quality understanding that incorporates content analysis, degradation perception, and comparison reasoning beyond mere numerical scoring. Previous MLLM-based methods typically either generate numerical scores lacking interpretability or heavily rely on supervised fine-tuning (SFT) using large-scale annotated datasets to provide descriptive assessments, limiting their flexibility and applicability. In this paper, we propose Q-Insight, a reinforcement learning-based model built upon group relative policy optimization (GRPO), which demonstrates strong visual reasoning capability for image quality understanding while requiring only a limited amount of rating scores and degradation labels. By jointly optimizing score regression and degradation perception tasks with carefully designed reward functions, our approach effectively exploits their mutual benefits for enhanced performance. Extensive experiments demonstrate that Q-Insight substantially outperforms existing state-of-the-art methods in both score regression and degradation perception tasks, while exhibiting impressive zero-shot generalization to comparison reasoning tasks. Code will be available at https://github.com/lwq20020127/Q-Insight.
>
---
#### [replaced 016] Exploring Generalized Gait Recognition: Reducing Redundancy and Noise within Indoor and Outdoor Datasets
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15176v2](http://arxiv.org/pdf/2505.15176v2)**

> **作者:** Qian Zhou; Xianda Guo; Jilong Wang; Chuanfu Shen; Zhongyuan Wang; Hua Zou; Qin Zou; Chao Liang; Long Chen; Gang Wu
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Generalized gait recognition, which aims to achieve robust performance across diverse domains, remains a challenging problem due to severe domain shifts in viewpoints, appearances, and environments. While mixed-dataset training is widely used to enhance generalization, it introduces new obstacles including inter-dataset optimization conflicts and redundant or noisy samples, both of which hinder effective representation learning. To address these challenges, we propose a unified framework that systematically improves cross-domain gait recognition. First, we design a disentangled triplet loss that isolates supervision signals across datasets, mitigating gradient conflicts during optimization. Second, we introduce a targeted dataset distillation strategy that filters out the least informative 20\% of training samples based on feature redundancy and prediction uncertainty, enhancing data efficiency. Extensive experiments on CASIA-B, OU-MVLP, Gait3D, and GREW demonstrate that our method significantly improves cross-dataset recognition for both GaitBase and DeepGaitV2 backbones, without sacrificing source-domain accuracy. Code will be released at https://github.com/li1er3/Generalized_Gait.
>
---
#### [replaced 017] NBM: an Open Dataset for the Acoustic Monitoring of Nocturnal Migratory Birds in Europe
- **分类: cs.SD; cs.CV; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.03633v4](http://arxiv.org/pdf/2412.03633v4)**

> **作者:** Louis Airale; Adrien Pajot; Juliette Linossier
>
> **摘要:** The persisting threats on migratory bird populations highlight the urgent need for effective monitoring techniques that could assist in their conservation. Among these, passive acoustic monitoring is an essential tool, particularly for nocturnal migratory species that are difficult to track otherwise. This work presents the Nocturnal Bird Migration (NBM) dataset, a collection of 13,359 annotated vocalizations from 117 species of the Western Palearctic. The dataset includes precise time and frequency annotations, gathered by dozens of bird enthusiasts across France, enabling novel downstream acoustic analysis. In particular, we prove the utility of this database by training an original two-stage deep object detection model tailored for the processing of audio data. While allowing the precise localization of bird calls in spectrograms, this model shows competitive accuracy on the 45 main species of the dataset with state-of-the-art systems trained on much larger audio collections. These results highlight the interest of fostering similar open-science initiatives to acquire costly but valuable fine-grained annotations of audio files. All data and code are made openly available.
>
---
#### [replaced 018] QVGen: Pushing the Limit of Quantized Video Generative Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11497v2](http://arxiv.org/pdf/2505.11497v2)**

> **作者:** Yushi Huang; Ruihao Gong; Jing Liu; Yifu Ding; Chengtao Lv; Haotong Qin; Jun Zhang
>
> **备注:** Our code will be released upon acceptance
>
> **摘要:** Video diffusion models (DMs) have enabled high-quality video synthesis. Yet, their substantial computational and memory demands pose serious challenges to real-world deployment, even on high-end GPUs. As a commonly adopted solution, quantization has proven notable success in reducing cost for image DMs, while its direct application to video DMs remains ineffective. In this paper, we present QVGen, a novel quantization-aware training (QAT) framework tailored for high-performance and inference-efficient video DMs under extremely low-bit quantization (e.g., 4-bit or below). We begin with a theoretical analysis demonstrating that reducing the gradient norm is essential to facilitate convergence for QAT. To this end, we introduce auxiliary modules ($\Phi$) to mitigate large quantization errors, leading to significantly enhanced convergence. To eliminate the inference overhead of $\Phi$, we propose a rank-decay strategy that progressively eliminates $\Phi$. Specifically, we repeatedly employ singular value decomposition (SVD) and a proposed rank-based regularization $\mathbf{\gamma}$ to identify and decay low-contributing components. This strategy retains performance while zeroing out inference overhead. Extensive experiments across $4$ state-of-the-art (SOTA) video DMs, with parameter sizes ranging from $1.3$B $\sim14$B, show that QVGen is the first to reach full-precision comparable quality under 4-bit settings. Moreover, it significantly outperforms existing methods. For instance, our 3-bit CogVideoX-2B achieves improvements of $+25.28$ in Dynamic Degree and $+8.43$ in Scene Consistency on VBench.
>
---
#### [replaced 019] Tracing Representation Progression: Analyzing and Enhancing Layer-Wise Similarity
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.14479v3](http://arxiv.org/pdf/2406.14479v3)**

> **作者:** Jiachen Jiang; Jinxin Zhou; Zhihui Zhu
>
> **摘要:** Analyzing the similarity of internal representations has been an important technique for understanding the behavior of deep neural networks. Most existing methods for analyzing the similarity between representations of high dimensions, such as those based on Centered Kernel Alignment (CKA), rely on statistical properties of the representations for a set of data points. In this paper, we focus on transformer models and study the similarity of representations between the hidden layers of individual transformers. In this context, we show that a simple sample-wise cosine similarity metric is capable of capturing the similarity and aligns with the complicated CKA. Our experimental results on common transformers reveal that representations across layers are positively correlated, with similarity increasing when layers get closer. We provide a theoretical justification for this phenomenon under the geodesic curve assumption for the learned transformer. We then show that an increase in representation similarity implies an increase in predicted probability when directly applying the last-layer classifier to any hidden layer representation. We then propose an aligned training method to improve the effectiveness of shallow layer by enhancing the similarity between internal representations, with trained models that enjoy the following properties: (1) more early saturation events, (2) layer-wise accuracies monotonically increase and reveal the minimal depth needed for the given task, (3) when served as multi-exit models, they achieve on-par performance with standard multi-exit architectures which consist of additional classifiers designed for early exiting in shallow layers. To our knowledge, our work is the first to show that one common classifier is sufficient for multi-exit models. We conduct experiments on both vision and NLP tasks to demonstrate the performance of the proposed aligned training.
>
---
#### [replaced 020] Ctrl-Room: Controllable Text-to-3D Room Meshes Generation with Layout Constraints
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2310.03602v4](http://arxiv.org/pdf/2310.03602v4)**

> **作者:** Chuan Fang; Yuan Dong; Kunming Luo; Xiaotao Hu; Rakesh Shrestha; Ping Tan
>
> **摘要:** Text-driven 3D indoor scene generation is useful for gaming, the film industry, and AR/VR applications. However, existing methods cannot faithfully capture the room layout, nor do they allow flexible editing of individual objects in the room. To address these problems, we present Ctrl-Room, which can generate convincing 3D rooms with designer-style layouts and high-fidelity textures from just a text prompt. Moreover, Ctrl-Room enables versatile interactive editing operations such as resizing or moving individual furniture items. Our key insight is to separate the modeling of layouts and appearance. Our proposed method consists of two stages: a Layout Generation Stage and an Appearance Generation Stage. The Layout Generation Stage trains a text-conditional diffusion model to learn the layout distribution with our holistic scene code parameterization. Next, the Appearance Generation Stage employs a fine-tuned ControlNet to produce a vivid panoramic image of the room guided by the 3D scene layout and text prompt. We thus achieve a high-quality 3D room generation with convincing layouts and lively textures. Benefiting from the scene code parameterization, we can easily edit the generated room model through our mask-guided editing module, without expensive edit-specific training. Extensive experiments on the Structured3D dataset demonstrate that our method outperforms existing methods in producing more reasonable, view-consistent, and editable 3D rooms from natural language prompts.
>
---
#### [replaced 021] RCR: Robust Crowd Reconstruction with Upright Space from a Single Large-scene Image
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.06232v2](http://arxiv.org/pdf/2411.06232v2)**

> **作者:** Jing Huang; Hao Wen; Tianyi Zhou; Haozhe Lin; Yu-kun Lai; Kun Li
>
> **摘要:** This paper focuses on spatially consistent hundreds of human pose and shape reconstruction from a single large-scene image with various human scales under arbitrary camera FoVs (Fields of View). Due to the small and highly varying 2D human scales, depth ambiguity, and perspective distortion, no existing methods can achieve globally consistent reconstruction with correct reprojection. To address these challenges, we first propose a new concept, Human-scene Virtual Interaction Point (HVIP), to convert the complex 3D human localization into 2D-pixel localization. We then extend it to RCR (Robust Crowd Reconstruction), which achieves globally consistent reconstruction and stable generalization on different camera FoVs without test-time optimization. To perceive humans in varying pixel sizes, we propose an Iterative Ground-aware Cropping to automatically crop the image and then merge the results. To eliminate the influence of the camera and cropping process during the reconstruction, we introduce a canonical Upright 3D Space and the corresponding Upright 2D Space. To link the canonical space and the camera space, we propose the Upright Normalization, which transforms the local crop input into the Upright 2D Space, and transforms the output from the Upright 3D Space into the unified camera space. Besides, we contribute two benchmark datasets, LargeCrowd and SynCrowd, for evaluating crowd reconstruction in large scenes. Experimental results demonstrate the effectiveness of the proposed method. The source code and data will be publicly available for research purposes.
>
---
#### [replaced 022] Simpler Fast Vision Transformers with a Jumbo CLS Token
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.15021v2](http://arxiv.org/pdf/2502.15021v2)**

> **作者:** Anthony Fuller; Yousef Yassin; Daniel G. Kyrollos; Evan Shelhamer; James R. Green
>
> **摘要:** We introduce a simple enhancement of vision transformers (ViTs) to improve accuracy while maintaining throughput. Our approach, Jumbo, creates a wider CLS token, which is split to match the patch token width before attention, processed with self-attention, and reassembled. After attention, Jumbo applies a dedicated, wider FFN to this token. Since there is only one Jumbo token, its cost is minimal, and because we share this FFN across layers, its parameter count is controlled. Jumbo significantly improves over ViT+Registers on ImageNet-1K and ImageNet-21K. These gains are largest at small sizes / high speeds, e.g., ViT-nano+Jumbo outperforms ViT-nano+Registers by 13%. In fact, our Jumbo models are so efficient that they outperform specialized compute-efficient models while preserving the architectural advantages of plain ViTs, such as support for token dropping and other modalities. Accordingly, we demonstrate that Jumbo excels in these two settings via masked autoencoding and on a suite of time series benchmarks. Code and weights available: https://github.com/antofuller/jumbo
>
---
#### [replaced 023] Self-supervised Multi-future Occupancy Forecasting for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2407.21126v2](http://arxiv.org/pdf/2407.21126v2)**

> **作者:** Bernard Lange; Masha Itkina; Jiachen Li; Mykel J. Kochenderfer
>
> **摘要:** Environment prediction frameworks are critical for the safe navigation of autonomous vehicles (AVs) in dynamic settings. LiDAR-generated occupancy grid maps (L-OGMs) offer a robust bird's-eye view for the scene representation, enabling self-supervised joint scene predictions while exhibiting resilience to partial observability and perception detection failures. Prior approaches have focused on deterministic L-OGM prediction architectures within the grid cell space. While these methods have seen some success, they frequently produce unrealistic predictions and fail to capture the stochastic nature of the environment. Additionally, they do not effectively integrate additional sensor modalities present in AVs. Our proposed framework, Latent Occupancy Prediction (LOPR), performs stochastic L-OGM prediction in the latent space of a generative architecture and allows for conditioning on RGB cameras, maps, and planned trajectories. We decode predictions using either a single-step decoder, which provides high-quality predictions in real-time, or a diffusion-based batch decoder, which can further refine the decoded frames to address temporal consistency issues and reduce compression losses. Our experiments on the nuScenes and Waymo Open datasets show that all variants of our approach qualitatively and quantitatively outperform prior approaches.
>
---
#### [replaced 024] X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.05528v2](http://arxiv.org/pdf/2505.05528v2)**

> **作者:** Hanxun Huang; Sarah Erfani; Yige Li; Xingjun Ma; James Bailey
>
> **备注:** ICML 2025
>
> **摘要:** As Contrastive Language-Image Pre-training (CLIP) models are increasingly adopted for diverse downstream tasks and integrated into large vision-language models (VLMs), their susceptibility to adversarial perturbations has emerged as a critical concern. In this work, we introduce \textbf{X-Transfer}, a novel attack method that exposes a universal adversarial vulnerability in CLIP. X-Transfer generates a Universal Adversarial Perturbation (UAP) capable of deceiving various CLIP encoders and downstream VLMs across different samples, tasks, and domains. We refer to this property as \textbf{super transferability}--a single perturbation achieving cross-data, cross-domain, cross-model, and cross-task adversarial transferability simultaneously. This is achieved through \textbf{surrogate scaling}, a key innovation of our approach. Unlike existing methods that rely on fixed surrogate models, which are computationally intensive to scale, X-Transfer employs an efficient surrogate scaling strategy that dynamically selects a small subset of suitable surrogates from a large search space. Extensive evaluations demonstrate that X-Transfer significantly outperforms previous state-of-the-art UAP methods, establishing a new benchmark for adversarial transferability across CLIP models. The code is publicly available in our \href{https://github.com/HanxunH/XTransferBench}{GitHub repository}.
>
---
#### [replaced 025] SAGI: Semantically Aligned and Uncertainty Guided AI Image Inpainting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.06593v2](http://arxiv.org/pdf/2502.06593v2)**

> **作者:** Paschalis Giakoumoglou; Dimitrios Karageorgiou; Symeon Papadopoulos; Panagiotis C. Petrantonakis
>
> **摘要:** Recent advancements in generative AI have made text-guided image inpainting -- adding, removing, or altering image regions using textual prompts -- widely accessible. However, generating semantically correct photorealistic imagery, typically requires carefully-crafted prompts and iterative refinement by evaluating the realism of the generated content - tasks commonly performed by humans. To automate the generative process, we propose Semantically Aligned and Uncertainty Guided AI Image Inpainting (SAGI), a model-agnostic pipeline, to sample prompts from a distribution that closely aligns with human perception and to evaluate the generated content and discard one that deviates from such a distribution, which we approximate using pretrained Large Language Models and Vision-Language Models. By applying this pipeline on multiple state-of-the-art inpainting models, we create the SAGI Dataset (SAGI-D), currently the largest and most diverse dataset of AI-generated inpaintings, comprising over 95k inpainted images and a human-evaluated subset. Our experiments show that semantic alignment significantly improves image quality and aesthetics, while uncertainty guidance effectively identifies realistic manipulations - human ability to distinguish inpainted images from real ones drops from 74% to 35% in terms of accuracy, after applying our pipeline. Moreover, using SAGI-D for training several image forensic approaches increases in-domain detection performance on average by 37.4% and out-of-domain generalization by 26.1% in terms of IoU, also demonstrating its utility in countering malicious exploitation of generative AI. Code and dataset are available at https://github.com/mever-team/SAGI
>
---
#### [replaced 026] Beyond the Destination: A Novel Benchmark for Exploration-Aware Embodied Question Answering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11117v3](http://arxiv.org/pdf/2503.11117v3)**

> **作者:** Kaixuan Jiang; Yang Liu; Weixing Chen; Jingzhou Luo; Ziliang Chen; Ling Pan; Guanbin Li; Liang Lin
>
> **摘要:** Embodied Question Answering (EQA) is a challenging task in embodied intelligence that requires agents to dynamically explore 3D environments, actively gather visual information, and perform multi-step reasoning to answer questions. However, current EQA approaches suffer from critical limitations in exploration efficiency, dataset design, and evaluation metrics. Moreover, existing datasets often introduce biases or prior knowledge, leading to disembodied reasoning, while frontier-based exploration strategies struggle in cluttered environments and fail to ensure fine-grained exploration of task-relevant areas. To address these challenges, we construct the EXPloration-awaRe Embodied queStion anSwering Benchmark (EXPRESS-Bench), the largest dataset designed specifically to evaluate both exploration and reasoning capabilities. EXPRESS-Bench consists of 777 exploration trajectories and 2,044 question-trajectory pairs. To improve exploration efficiency, we propose Fine-EQA, a hybrid exploration model that integrates frontier-based and goal-oriented navigation to guide agents toward task-relevant regions more effectively. Additionally, we introduce a novel evaluation metric, Exploration-Answer Consistency (EAC), which ensures faithful assessment by measuring the alignment between answer grounding and exploration reliability. Extensive experimental comparisons with state-of-the-art EQA models demonstrate the effectiveness of our EXPRESS-Bench in advancing embodied exploration and question reasoning.
>
---
#### [replaced 027] Rethinking Bottlenecks in Safety Fine-Tuning of Vision Language Models
- **分类: cs.CV; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2501.18533v2](http://arxiv.org/pdf/2501.18533v2)**

> **作者:** Yi Ding; Lijun Li; Bing Cao; Jing Shao
>
> **摘要:** Large Vision-Language Models (VLMs) have achieved remarkable performance across a wide range of tasks. However, their deployment in safety-critical domains poses significant challenges. Existing safety fine-tuning methods, which focus on textual or multimodal content, fall short in addressing challenging cases or disrupt the balance between helpfulness and harmlessness. Our evaluation highlights a safety reasoning gap: these methods lack safety visual reasoning ability, leading to such bottlenecks. To address this limitation and enhance both visual perception and reasoning in safety-critical contexts, we propose a novel dataset that integrates multi-image inputs with safety Chain-of-Thought (CoT) labels as fine-grained reasoning logic to improve model performance. Specifically, we introduce the Multi-Image Safety (MIS) dataset, an instruction-following dataset tailored for multi-image safety scenarios, consisting of training and test splits. Our experiments demonstrate that fine-tuning InternVL2.5-8B with MIS significantly outperforms both powerful open-source models and API-based models in challenging multi-image tasks requiring safety-related visual reasoning. This approach not only delivers exceptional safety performance but also preserves general capabilities without any trade-offs. Specifically, fine-tuning with MIS increases average accuracy by 0.83% across five general benchmarks and reduces the Attack Success Rate (ASR) on multiple safety benchmarks by a large margin.
>
---
#### [replaced 028] AnimeDL-2M: Million-Scale AI-Generated Anime Image Detection and Localization in Diffusion Era
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.11015v2](http://arxiv.org/pdf/2504.11015v2)**

> **作者:** Chenyang Zhu; Xing Zhang; Yuyang Sun; Ching-Chun Chang; Isao Echizen
>
> **备注:** 8+2 pages; update figure 3,4,5 as adding real images into detection task tests
>
> **摘要:** Recent advances in image generation, particularly diffusion models, have significantly lowered the barrier for creating sophisticated forgeries, making image manipulation detection and localization (IMDL) increasingly challenging. While prior work in IMDL has focused largely on natural images, the anime domain remains underexplored-despite its growing vulnerability to AI-generated forgeries. Misrepresentations of AI-generated images as hand-drawn artwork, copyright violations, and inappropriate content modifications pose serious threats to the anime community and industry. To address this gap, we propose AnimeDL-2M, the first large-scale benchmark for anime IMDL with comprehensive annotations. It comprises over two million images including real, partially manipulated, and fully AI-generated samples. Experiments indicate that models trained on existing IMDL datasets of natural images perform poorly when applied to anime images, highlighting a clear domain gap between anime and natural images. To better handle IMDL tasks in anime domain, we further propose AniXplore, a novel model tailored to the visual characteristics of anime imagery. Extensive evaluations demonstrate that AniXplore achieves superior performance compared to existing methods. Dataset and code can be found in https://flytweety.github.io/AnimeDL2M/.
>
---
#### [replaced 029] ViCTr: Vital Consistency Transfer for Pathology Aware Image Synthesis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.04963v2](http://arxiv.org/pdf/2505.04963v2)**

> **作者:** Onkar Susladkar; Gayatri Deshmukh; Yalcin Tur; Gorkhem Durak; Ulas Bagci
>
> **摘要:** Synthesizing medical images remains challenging due to limited annotated pathological data, modality domain gaps, and the complexity of representing diffuse pathologies such as liver cirrhosis. Existing methods often struggle to maintain anatomical fidelity while accurately modeling pathological features, frequently relying on priors derived from natural images or inefficient multi-step sampling. In this work, we introduce ViCTr (Vital Consistency Transfer), a novel two-stage framework that combines a rectified flow trajectory with a Tweedie-corrected diffusion process to achieve high-fidelity, pathology-aware image synthesis. First, we pretrain ViCTr on the ATLAS-8k dataset using Elastic Weight Consolidation (EWC) to preserve critical anatomical structures. We then fine-tune the model adversarially with Low-Rank Adaptation (LoRA) modules for precise control over pathology severity. By reformulating Tweedie's formula within a linear trajectory framework, ViCTr supports one-step sampling, reducing inference from 50 steps to just 4, without sacrificing anatomical realism. We evaluate ViCTr on BTCV (CT), AMOS (MRI), and CirrMRI600+ (cirrhosis) datasets. Results demonstrate state-of-the-art performance, achieving a Medical Frechet Inception Distance (MFID) of 17.01 for cirrhosis synthesis 28% lower than existing approaches and improving nnUNet segmentation by +3.8% mDSC when used for data augmentation. Radiologist reviews indicate that ViCTr-generated liver cirrhosis MRIs are clinically indistinguishable from real scans. To our knowledge, ViCTr is the first method to provide fine-grained, pathology-aware MRI synthesis with graded severity control, closing a critical gap in AI-driven medical imaging research.
>
---
#### [replaced 030] Multi-Faceted Multimodal Monosemanticity
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.14888v3](http://arxiv.org/pdf/2502.14888v3)**

> **作者:** Hanqi Yan; Xiangxiang Cui; Lu Yin; Paul Pu Liang; Yulan He; Yifei Wang
>
> **摘要:** Humans experience the world through multiple modalities, such as, vision, language, and speech, making it natural to explore the commonality and distinctions among them. In this work, we take a data-driven approach to address this question by analyzing interpretable, monosemantic features extracted from deep multimodal models. Specifically, we investigate CLIP, a prominent visual-language representation model trained on massive image-text pairs. Building on prior research in single-modal interpretability, we develop a set of multi-modal interpretability tools and measures designed to disentangle and analyze features learned from CLIP. Specifically, we introduce the Modality Dominance Score (MDS) to attribute each CLIP feature to a specific modality. We then map CLIP features into a more interpretable space, enabling us to categorize them into three distinct classes: vision features (single-modal), language features (single-modal), and visual-language features (cross-modal). Interestingly, this data-driven categorization closely aligns with human intuitive understandings of different modalities. We further show that this modality decomposition can benefit multiple downstream tasks, including reducing bias in gender detection, generating cross-modal adversarial examples, and enabling modal-specific feature control in text-to-image generation. These results indicate that large-scale multimodal models, when equipped with task-agnostic interpretability tools, can offer valuable insights into the relationships between different data modalities.
>
---
#### [replaced 031] STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.23765v4](http://arxiv.org/pdf/2503.23765v4)**

> **作者:** Yun Li; Yiming Zhang; Tao Lin; XiangRui Liu; Wenxiao Cai; Zheng Liu; Bo Zhao
>
> **摘要:** The use of Multimodal Large Language Models (MLLMs) as an end-to-end solution for Embodied AI and Autonomous Driving has become a prevailing trend. While MLLMs have been extensively studied for visual semantic understanding tasks, their ability to perform precise and quantitative spatial-temporal understanding in real-world applications remains largely unexamined, leading to uncertain prospects. To evaluate models' Spatial-Temporal Intelligence, we introduce STI-Bench, a benchmark designed to evaluate MLLMs' spatial-temporal understanding through challenging tasks such as estimating and predicting the appearance, pose, displacement, and motion of objects. Our benchmark encompasses a wide range of robot and vehicle operations across desktop, indoor, and outdoor scenarios. The extensive experiments reveals that the state-of-the-art MLLMs still struggle in real-world spatial-temporal understanding, especially in tasks requiring precise distance estimation and motion analysis.
>
---
#### [replaced 032] Enhanced 3D Object Detection via Diverse Feature Representations of 4D Radar Tensor
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.06114v3](http://arxiv.org/pdf/2502.06114v3)**

> **作者:** Seung-Hyun Song; Dong-Hee Paek; Minh-Quan Dao; Ezio Malis; Seung-Hyun Kong
>
> **备注:** Arxiv preprint version
>
> **摘要:** Recent advances in automotive four-dimensional (4D) Radar have enabled access to raw 4D Radar Tensor (4DRT), offering richer spatial and Doppler information than conventional point clouds. While most existing methods rely on heavily pre-processed, sparse Radar data, recent attempts to leverage raw 4DRT face high computational costs and limited scalability. To address these limitations, we propose a novel three-dimensional (3D) object detection framework that maximizes the utility of 4DRT while preserving efficiency. Our method introduces a multi-teacher knowledge distillation (KD), where multiple teacher models are trained on point clouds derived from diverse 4DRT pre-processing techniques, each capturing complementary signal characteristics. These teacher representations are fused via a dedicated aggregation module and distilled into a lightweight student model that operates solely on a sparse Radar input. Experimental results on the K-Radar dataset demonstrate that our framework achieves improvements of 7.3% in AP_3D and 9.5% in AP_BEV over the baseline RTNH model when using extremely sparse inputs. Furthermore, it attains comparable performance to denser-input baselines while significantly reducing the input data size by about 90 times, confirming the scalability and efficiency of our approach.
>
---
#### [replaced 033] On the Robustness of Medical Vision-Language Models: Are they Truly Generalizable?
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15425v2](http://arxiv.org/pdf/2505.15425v2)**

> **作者:** Raza Imam; Rufael Marew; Mohammad Yaqub
>
> **备注:** Dataset and Code is available at https://github.com/BioMedIA-MBZUAI/RobustMedCLIP Accepted at: Medical Image Understanding and Analysis (MIUA) 2025
>
> **摘要:** Medical Vision-Language Models (MVLMs) have achieved par excellence generalization in medical image analysis, yet their performance under noisy, corrupted conditions remains largely untested. Clinical imaging is inherently susceptible to acquisition artifacts and noise; however, existing evaluations predominantly assess generally clean datasets, overlooking robustness -- i.e., the model's ability to perform under real-world distortions. To address this gap, we first introduce MediMeta-C, a corruption benchmark that systematically applies several perturbations across multiple medical imaging datasets. Combined with MedMNIST-C, this establishes a comprehensive robustness evaluation framework for MVLMs. We further propose RobustMedCLIP, a visual encoder adaptation of a pretrained MVLM that incorporates few-shot tuning to enhance resilience against corruptions. Through extensive experiments, we benchmark 5 major MVLMs across 5 medical imaging modalities, revealing that existing models exhibit severe degradation under corruption and struggle with domain-modality tradeoffs. Our findings highlight the necessity of diverse training and robust adaptation strategies, demonstrating that efficient low-rank adaptation when paired with few-shot tuning, improves robustness while preserving generalization across modalities.
>
---
#### [replaced 034] Flashback: Memory-Driven Zero-shot, Real-time Video Anomaly Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15205v2](http://arxiv.org/pdf/2505.15205v2)**

> **作者:** Hyogun Lee; Haksub Kim; Ig-Jae Kim; Yonghun Choi
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Video Anomaly Detection (VAD) automatically identifies anomalous events from video, mitigating the need for human operators in large-scale surveillance deployments. However, two fundamental obstacles hinder real-world adoption: domain dependency and real-time constraints -- requiring near-instantaneous processing of incoming video. To this end, we propose Flashback, a zero-shot and real-time video anomaly detection paradigm. Inspired by the human cognitive mechanism of instantly judging anomalies and reasoning in current scenes based on past experience, Flashback operates in two stages: Recall and Respond. In the offline recall stage, an off-the-shelf LLM builds a pseudo-scene memory of both normal and anomalous captions without any reliance on real anomaly data. In the online respond stage, incoming video segments are embedded and matched against this memory via similarity search. By eliminating all LLM calls at inference time, Flashback delivers real-time VAD even on a consumer-grade GPU. On two large datasets from real-world surveillance scenarios, UCF-Crime and XD-Violence, we achieve 87.3 AUC (+7.0 pp) and 75.1 AP (+13.1 pp), respectively, outperforming prior zero-shot VAD methods by large margins.
>
---
#### [replaced 035] ARFC-WAHNet: Adaptive Receptive Field Convolution and Wavelet-Attentive Hierarchical Network for Infrared Small Target Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10595v2](http://arxiv.org/pdf/2505.10595v2)**

> **作者:** Xingye Cui; Junhai Luo; Jiakun Deng; Kexuan Li; Xiangyu Qiu; Zhenming Peng
>
> **摘要:** Infrared small target detection (ISTD) is critical in both civilian and military applications. However, the limited texture and structural information in infrared images makes accurate detection particularly challenging. Although recent deep learning-based methods have improved performance, their use of conventional convolution kernels limits adaptability to complex scenes and diverse targets. Moreover, pooling operations often cause feature loss and insufficient exploitation of image information. To address these issues, we propose an adaptive receptive field convolution and wavelet-attentive hierarchical network for infrared small target detection (ARFC-WAHNet). This network incorporates a multi-receptive field feature interaction convolution (MRFFIConv) module to adaptively extract discriminative features by integrating multiple convolutional branches with a gated unit. A wavelet frequency enhancement downsampling (WFED) module leverages Haar wavelet transform and frequency-domain reconstruction to enhance target features and suppress background noise. Additionally, we introduce a high-low feature fusion (HLFF) module for integrating low-level details with high-level semantics, and a global median enhancement attention (GMEA) module to improve feature diversity and expressiveness via global attention. Experiments on public datasets SIRST, NUDT-SIRST, and IRSTD-1k demonstrate that ARFC-WAHNet outperforms recent state-of-the-art methods in both detection accuracy and robustness, particularly under complex backgrounds. The code is available at https://github.com/Leaf2001/ARFC-WAHNet.
>
---
#### [replaced 036] MMXU: A Multi-Modal and Multi-X-ray Understanding Dataset for Disease Progression
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.11651v2](http://arxiv.org/pdf/2502.11651v2)**

> **作者:** Linjie Mu; Zhongzhen Huang; Shengqian Qin; Yakun Zhu; Shaoting Zhang; Xiaofan Zhang
>
> **摘要:** Large vision-language models (LVLMs) have shown great promise in medical applications, particularly in visual question answering (MedVQA) and diagnosis from medical images. However, existing datasets and models often fail to consider critical aspects of medical diagnostics, such as the integration of historical records and the analysis of disease progression over time. In this paper, we introduce MMXU (Multimodal and MultiX-ray Understanding), a novel dataset for MedVQA that focuses on identifying changes in specific regions between two patient visits. Unlike previous datasets that primarily address single-image questions, MMXU enables multi-image questions, incorporating both current and historical patient data. We demonstrate the limitations of current LVLMs in identifying disease progression on MMXU-\textit{test}, even those that perform well on traditional benchmarks. To address this, we propose a MedRecord-Augmented Generation (MAG) approach, incorporating both global and regional historical records. Our experiments show that integrating historical records significantly enhances diagnostic accuracy by at least 20\%, bridging the gap between current LVLMs and human expert performance. Additionally, we fine-tune models with MAG on MMXU-\textit{dev}, which demonstrates notable improvements. We hope this work could illuminate the avenue of advancing the use of LVLMs in medical diagnostics by emphasizing the importance of historical context in interpreting medical images. Our dataset is released at github: https://github.com/linjiemu/MMXU.
>
---
#### [replaced 037] DiffBreak: Is Diffusion-Based Purification Robust?
- **分类: cs.CR; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.16598v3](http://arxiv.org/pdf/2411.16598v3)**

> **作者:** Andre Kassis; Urs Hengartner; Yaoliang Yu
>
> **摘要:** Diffusion-based purification (DBP) has become a cornerstone defense against adversarial examples (AEs), regarded as robust due to its use of diffusion models (DMs) that project AEs onto the natural data manifold. We refute this core claim, theoretically proving that gradient-based attacks effectively target the DM rather than the classifier, causing DBP's outputs to align with adversarial distributions. This prompts a reassessment of DBP's robustness, attributing it to two critical flaws: incorrect gradients and inappropriate evaluation protocols that test only a single random purification of the AE. We show that with proper accounting for stochasticity and resubmission risk, DBP collapses. To support this, we introduce DiffBreak, the first reliable toolkit for differentiation through DBP, eliminating gradient flaws that previously further inflated robustness estimates. We also analyze the current defense scheme used for DBP where classification relies on a single purification, pinpointing its inherent invalidity. We provide a statistically grounded majority-vote (MV) alternative that aggregates predictions across multiple purified copies, showing partial but meaningful robustness gain. We then propose a novel adaptation of an optimization method against deepfake watermarking, crafting systemic perturbations that defeat DBP even under MV, challenging DBP's viability.
>
---
#### [replaced 038] WildLive: Near Real-time Visual Wildlife Tracking onboard UAVs
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.10165v3](http://arxiv.org/pdf/2504.10165v3)**

> **作者:** Nguyen Ngoc Dat; Tom Richardson; Matthew Watson; Kilian Meier; Jenna Kline; Sid Reid; Guy Maalouf; Duncan Hine; Majid Mirmehdi; Tilo Burghardt
>
> **摘要:** Live tracking of wildlife via high-resolution video processing directly onboard drones is widely unexplored and most existing solutions rely on streaming video to ground stations to support navigation. Yet, both autonomous animal-reactive flight control beyond visual line of sight and/or mission-specific individual and behaviour recognition tasks rely to some degree on this capability. In response, we introduce WildLive - a near real-time animal detection and tracking framework for high-resolution imagery running directly onboard uncrewed aerial vehicles (UAVs). The system performs multi-animal detection and tracking at 17.81fps for HD and 7.53fps on 4K video streams suitable for operation during higher altitude flights to minimise animal disturbance. Our system is optimised for Jetson Orin AGX onboard hardware. It integrates the efficiency of sparse optical flow tracking and mission-specific sampling with device-optimised and proven YOLO-driven object detection and segmentation techniques. Essentially, computational resource is focused onto spatio-temporal regions of high uncertainty to significantly improve UAV processing speeds. Alongside, we introduce our WildLive dataset, which comprises 200K+ annotated animal instances across 19K+ frames from 4K UAV videos collected at the Ol Pejeta Conservancy in Kenya. All frames contain ground truth bounding boxes, segmentation masks, as well as individual tracklets and tracking point trajectories. We compare our system against current object tracking approaches including OC-SORT, ByteTrack, and SORT. Our multi-animal tracking experiments with onboard hardware confirm that near real-time high-resolution wildlife tracking is possible on UAVs whilst maintaining high accuracy levels as needed for future navigational and mission-specific animal-centric operational autonomy. Our materials are available at: https://dat-nguyenvn.github.io/WildLive/
>
---
#### [replaced 039] DiverseNet: Decision Diversified Semi-supervised Semantic Segmentation Networks for Remote Sensing Imagery
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2311.13716v3](http://arxiv.org/pdf/2311.13716v3)**

> **作者:** Wanli Ma; Oktay Karakus; Paul L. Rosin
>
> **摘要:** Semi-supervised learning (SSL) aims to help reduce the cost of the manual labelling process by leveraging a substantial pool of unlabelled data alongside a limited set of labelled data during the training phase. Since pixel-level manual labelling in large-scale remote sensing imagery is expensive and time-consuming, semi-supervised learning has become a widely used solution to deal with this. However, the majority of existing SSL frameworks, especially various teacher-student frameworks, are too bulky to run efficiently on a GPU with limited memory. There is still a lack of lightweight SSL frameworks and efficient perturbation methods to promote the diversity of training samples and enhance the precision of pseudo labels during training. In order to fill this gap, we proposed a simple, lightweight, and efficient SSL architecture named \textit{DiverseHead}, which promotes the utilisation of multiple decision heads instead of multiple whole networks. Another limitation of most existing SSL frameworks is the insufficient diversity of pseudo labels, as they rely on the same network architecture and fail to explore different structures for generating pseudo labels. To solve this issue, we propose \textit{DiverseModel} to explore and analyse different networks in parallel for SSL to increase the diversity of pseudo labels. The two proposed methods, namely \textit{DiverseHead} and \textit{DiverseModel}, both achieve competitive semantic segmentation performance in four widely used remote sensing imagery datasets compared to state-of-the-art semi-supervised learning methods. Meanwhile, the proposed lightweight DiverseHead architecture can be easily applied to various state-of-the-art SSL methods while further improving their performance. The code is available at https://github.com/WANLIMA-CARDIFF/DiverseNet.
>
---
#### [replaced 040] Rapid Whole Brain Motion-robust Mesoscale In-vivo MR Imaging using Multi-scale Implicit Neural Representation
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.08634v2](http://arxiv.org/pdf/2502.08634v2)**

> **作者:** Jun Lyu; Lipeng Ning; William Consagra; Qiang Liu; Richard J. Rushmore; Berkin Bilgic; Yogesh Rathi
>
> **摘要:** High-resolution whole-brain in vivo MR imaging at mesoscale resolutions remains challenging due to long scan durations, motion artifacts, and limited signal-to-noise ratio (SNR). This study proposes Rotating-view super-resolution (ROVER)-MRI, an unsupervised framework based on multi-scale implicit neural representations (INR), enabling efficient recovery of fine anatomical details from multi-view thick-slice acquisitions. ROVER-MRI employs coordinate-based neural networks to implicitly and continuously encode image structures at multiple spatial scales, simultaneously modeling anatomical continuity and correcting inter-view motion through an integrated registration mechanism. Validation on ex-vivo monkey brain data and multiple in-vivo human datasets demonstrates substantially improved reconstruction performance compared to bicubic interpolation and state-of-the-art regularized least-squares super-resolution reconstruction (LS-SRR) with 2-fold reduction in scan time. Notably, ROVER-MRI achieves an unprecedented whole-brain in-vivo T2-weighted imaging at 180 micron isotropic resolution in only 17 minutes of scan time on a 7T scanner with 22.4% lower relative error compared to LS-SRR. We also demonstrate improved SNR using ROVER-MRI compared to a time-matched 3D GRE acquisition. Quantitative results on several datasets demonstrate better sharpness of the reconstructed images with ROVER-MRI for different super-resolution factors (5 to 11). These findings highlight ROVER-MRI's potential as a rapid, accurate, and motion-resilient mesoscale imaging solution, promising substantial advantages for neuroimaging studies.
>
---
#### [replaced 041] Quantifying Statistical Significance in Diffusion-Based Anomaly Localization via Selective Inference
- **分类: stat.ML; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.11789v4](http://arxiv.org/pdf/2402.11789v4)**

> **作者:** Teruyuki Katsuoka; Tomohiro Shiraishi; Daiki Miwa; Vo Nguyen Le Duy; Ichiro Takeuchi
>
> **备注:** 38 pages, 10 figures
>
> **摘要:** Anomaly localization in images (identifying regions that deviate from expected patterns) is vital in applications such as medical diagnosis and industrial inspection. A recent trend is the use of image generation models in anomaly localization, where these models generate normal-looking counterparts of anomalous images, thereby allowing flexible and adaptive anomaly localization. However, these methods inherit the uncertainty and bias implicitly embedded in the employed generative model, raising concerns about the reliability. To address this, we propose a statistical framework based on selective inference to quantify the significance of detected anomalous regions. Our method provides $p$-values to assess the false positive detection rates, providing a principled measure of reliability. As a proof of concept, we consider anomaly localization using a diffusion model and its applications to medical diagnoses and industrial inspections. The results indicate that the proposed method effectively controls the risk of false positive detection, supporting its use in high-stakes decision-making tasks.
>
---
#### [replaced 042] Camera Movement Estimation and Path Correction using the Combination of Modified A-SIFT and Stereo System for 3D Modelling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.17668v2](http://arxiv.org/pdf/2503.17668v2)**

> **作者:** Usha Kumari; Shuvendu Rana
>
> **摘要:** Creating accurate and efficient 3D models poses significant challenges, particularly in addressing large viewpoint variations, computational complexity, and alignment discrepancies. Efficient camera path generation can help resolve these issues. In this context, a modified version of the Affine Scale-Invariant Feature Transform (ASIFT) is proposed to extract more matching points with reduced computational overhead, ensuring an adequate number of inliers for precise camera rotation angle estimation. Additionally, a novel two-camera-based rotation correction model is introduced to mitigate small rotational errors, further enhancing accuracy. Furthermore, a stereo camera-based translation estimation and correction model is implemented to determine camera movement in 3D space by altering the Structure From Motion (SFM) model. Finally, the novel combination of ASIFT and two camera-based SFM models provides an accurate camera movement trajectory in 3D space. Experimental results show that the proposed camera movement approach achieves 99.9% accuracy compared to the actual camera movement path and outperforms state-of-the-art camera path estimation methods. By leveraging this accurate camera path, the system facilitates the creation of precise 3D models, making it a robust solution for applications requiring high fidelity and efficiency in 3D reconstruction.
>
---
#### [replaced 043] Selective Structured State Space for Multispectral-fused Small Target Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.14043v3](http://arxiv.org/pdf/2505.14043v3)**

> **作者:** Qianqian Zhang; WeiJun Wang; Yunxing Liu; Li Zhou; Hao Zhao; Junshe An; Zihan Wang
>
> **备注:** This work was submitted to CVPR 2025, but was rejected after being reviewed by 7 reviewers. After revision, it is currently under review
>
> **摘要:** Target detection in high-resolution remote sensing imagery faces challenges due to the low recognition accuracy of small targets and high computational costs. The computational complexity of the Transformer architecture increases quadratically with image resolution, while Convolutional Neural Networks (CNN) architectures are forced to stack deeper convolutional layers to expand their receptive fields, leading to an explosive growth in computational demands. To address these computational constraints, we leverage Mamba's linear complexity for efficiency. However, Mamba's performance declines for small targets, primarily because small targets occupy a limited area in the image and have limited semantic information. Accurate identification of these small targets necessitates not only Mamba's global attention capabilities but also the precise capture of fine local details. To this end, we enhance Mamba by developing the Enhanced Small Target Detection (ESTD) module and the Convolutional Attention Residual Gate (CARG) module. The ESTD module bolsters local attention to capture fine-grained details, while the CARG module, built upon Mamba, emphasizes spatial and channel-wise information, collectively improving the model's ability to capture distinctive representations of small targets. Additionally, to highlight the semantic representation of small targets, we design a Mask Enhanced Pixel-level Fusion (MEPF) module for multispectral fusion, which enhances target features by effectively fusing visible and infrared multimodal information.
>
---
#### [replaced 044] D3C2-Net: Dual-Domain Deep Convolutional Coding Network for Compressive Sensing
- **分类: cs.CV; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2207.13560v2](http://arxiv.org/pdf/2207.13560v2)**

> **作者:** Weiqi Li; Bin Chen; Shuai Liu; Shijie Zhao; Bowen Du; Yongbing Zhang; Jian Zhang
>
> **备注:** accepted by IEEE TCSVT
>
> **摘要:** By mapping iterative optimization algorithms into neural networks (NNs), deep unfolding networks (DUNs) exhibit well-defined and interpretable structures and achieve remarkable success in the field of compressive sensing (CS). However, most existing DUNs solely rely on the image-domain unfolding, which restricts the information transmission capacity and reconstruction flexibility, leading to their loss of image details and unsatisfactory performance. To overcome these limitations, this paper develops a dual-domain optimization framework that combines the priors of (1) image- and (2) convolutional-coding-domains and offers generality to CS and other inverse imaging tasks. By converting this optimization framework into deep NN structures, we present a Dual-Domain Deep Convolutional Coding Network (D3C2-Net), which enjoys the ability to efficiently transmit high-capacity self-adaptive convolutional features across all its unfolded stages. Our theoretical analyses and experiments on simulated and real captured data, covering 2D and 3D natural, medical, and scientific signals, demonstrate the effectiveness, practicality, superior performance, and generalization ability of our method over other competing approaches and its significant potential in achieving a balance among accuracy, complexity, and interpretability. Code is available at https://github.com/lwq20020127/D3C2-Net.
>
---
#### [replaced 045] Defending Multimodal Backdoored Models by Repulsive Visual Prompt Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.20392v3](http://arxiv.org/pdf/2412.20392v3)**

> **作者:** Zhifang Zhang; Shuo He; Haobo Wang; Bingquan Shen; Lei Feng
>
> **摘要:** Multimodal contrastive learning models (e.g., CLIP) can learn high-quality representations from large-scale image-text datasets, while they exhibit significant vulnerabilities to backdoor attacks, raising serious safety concerns. In this paper, we reveal that CLIP's vulnerabilities primarily stem from its tendency to encode features beyond in-dataset predictive patterns, compromising its visual feature resistivity to input perturbations. This makes its encoded features highly susceptible to being reshaped by backdoor triggers. To address this challenge, we propose Repulsive Visual Prompt Tuning (RVPT), a novel defense approach that employs deep visual prompt tuning with a specially designed feature-repelling loss. Specifically, RVPT adversarially repels the encoded features from deeper layers while optimizing the standard cross-entropy loss, ensuring that only predictive features in downstream tasks are encoded, thereby enhancing CLIP's visual feature resistivity against input perturbations and mitigating its susceptibility to backdoor attacks. Unlike existing multimodal backdoor defense methods that typically require the availability of poisoned data or involve fine-tuning the entire model, RVPT leverages few-shot downstream clean samples and only tunes a small number of parameters. Empirical results demonstrate that RVPT tunes only 0.27\% of the parameters in CLIP, yet it significantly outperforms state-of-the-art defense methods, reducing the attack success rate from 89.70\% to 2.76\% against the most advanced multimodal attacks on ImageNet and effectively generalizes its defensive capabilities across multiple datasets.
>
---
#### [replaced 046] Synergistic Bleeding Region and Point Detection in Laparoscopic Surgical Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.22174v2](http://arxiv.org/pdf/2503.22174v2)**

> **作者:** Jialun Pei; Zhangjun Zhou; Diandian Guo; Zhixi Li; Jing Qin; Bo Du; Pheng-Ann Heng
>
> **摘要:** Intraoperative bleeding in laparoscopic surgery causes rapid obscuration of the operative field to hinder the surgical process and increases the risk of postoperative complications. Intelligent detection of bleeding areas can quantify the blood loss to assist decision-making, while locating bleeding points helps surgeons quickly identify the source of bleeding and achieve hemostasis in time to improve surgical success rates. In this study, we first construct a real-world laparoscopic surgical bleeding detection dataset, named SurgBlood, comprising 5,330 frames from 95 surgical video clips with bleeding region and point annotations. Accordingly, we develop a dual-task synergistic online detector called BlooDet, designed to perform simultaneous detection of bleeding regions and points in laparoscopic surgery. Our framework embraces a dual-branch bidirectional guidance design based on Segment Anything Model 2 (SAM 2). The mask branch detects bleeding regions through adaptive edge and point prompt embeddings, and the point branch leverages mask memory to induce bleeding point memory modeling and capture the direction of bleed point movement via inter-frame optical flow. By bidirectional guidance, the two branches explore potential spatial-temporal relationships while leveraging memory modeling to infer the current bleeding condition. Extensive experiments demonstrate that our baseline outperforms 12 counterparts on SurgBlood in both bleeding region and point detection.
>
---
#### [replaced 047] VGGT-SLAM: Dense RGB SLAM Optimized on the SL(4) Manifold
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12549v2](http://arxiv.org/pdf/2505.12549v2)**

> **作者:** Dominic Maggio; Hyungtae Lim; Luca Carlone
>
> **摘要:** We present VGGT-SLAM, a dense RGB SLAM system constructed by incrementally and globally aligning submaps created from the feed-forward scene reconstruction approach VGGT using only uncalibrated monocular cameras. While related works align submaps using similarity transforms (i.e., translation, rotation, and scale), we show that such approaches are inadequate in the case of uncalibrated cameras. In particular, we revisit the idea of reconstruction ambiguity, where given a set of uncalibrated cameras with no assumption on the camera motion or scene structure, the scene can only be reconstructed up to a 15-degrees-of-freedom projective transformation of the true geometry. This inspires us to recover a consistent scene reconstruction across submaps by optimizing over the SL(4) manifold, thus estimating 15-degrees-of-freedom homography transforms between sequential submaps while accounting for potential loop closure constraints. As verified by extensive experiments, we demonstrate that VGGT-SLAM achieves improved map quality using long video sequences that are infeasible for VGGT due to its high GPU requirements.
>
---
#### [replaced 048] Boosting Edge Detection with Pixel-wise Feature Selection: The Extractor-Selector Paradigm
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.02534v2](http://arxiv.org/pdf/2501.02534v2)**

> **作者:** Hao Shu
>
> **备注:** 17 pages
>
> **摘要:** Deep learning has significantly advanced image edge detection (ED), primarily through improved feature extraction. However, most existing ED models apply uniform feature fusion across all pixels, ignoring critical differences between regions such as edges and textures. To address this limitation, we propose the Extractor-Selector (E-S) paradigm, a novel framework that introduces pixel-wise feature selection for more adaptive and precise fusion. Unlike conventional image-level fusion that applies the same convolutional kernel to all pixels, our approach dynamically selects relevant features at each pixel, enabling more refined edge predictions. The E-S framework can be seamlessly integrated with existing ED models without architectural changes, delivering substantial performance gains. It can also be combined with enhanced feature extractors for further accuracy improvements. Extensive experiments across multiple benchmarks confirm that our method consistently outperforms baseline ED models. For instance, on the BIPED2 dataset, the proposed framework can achieve over 7$\%$ improvements in ODS and OIS, and 22$\%$ improvements in AP, demonstrating its effectiveness and superiority.
>
---
#### [replaced 049] Erasing Undesirable Concepts in Diffusion Models with Adversarial Preservation
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.15618v4](http://arxiv.org/pdf/2410.15618v4)**

> **作者:** Anh Bui; Long Vuong; Khanh Doan; Trung Le; Paul Montague; Tamas Abraham; Dinh Phung
>
> **备注:** Erasing Concepts, Generative Unlearning, NeurIPS 2024. arXiv admin note: text overlap with arXiv:2403.12326
>
> **摘要:** Diffusion models excel at generating visually striking content from text but can inadvertently produce undesirable or harmful content when trained on unfiltered internet data. A practical solution is to selectively removing target concepts from the model, but this may impact the remaining concepts. Prior approaches have tried to balance this by introducing a loss term to preserve neutral content or a regularization term to minimize changes in the model parameters, yet resolving this trade-off remains challenging. In this work, we propose to identify and preserving concepts most affected by parameter changes, termed as \textit{adversarial concepts}. This approach ensures stable erasure with minimal impact on the other concepts. We demonstrate the effectiveness of our method using the Stable Diffusion model, showing that it outperforms state-of-the-art erasure methods in eliminating unwanted content while maintaining the integrity of other unrelated elements. Our code is available at https://github.com/tuananhbui89/Erasing-Adversarial-Preservation.
>
---
#### [replaced 050] Fantastic Targets for Concept Erasure in Diffusion Models and Where To Find Them
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.18950v3](http://arxiv.org/pdf/2501.18950v3)**

> **作者:** Anh Bui; Trang Vu; Long Vuong; Trung Le; Paul Montague; Tamas Abraham; Junae Kim; Dinh Phung
>
> **摘要:** Concept erasure has emerged as a promising technique for mitigating the risk of harmful content generation in diffusion models by selectively unlearning undesirable concepts. The common principle of previous works to remove a specific concept is to map it to a fixed generic concept, such as a neutral concept or just an empty text prompt. In this paper, we demonstrate that this fixed-target strategy is suboptimal, as it fails to account for the impact of erasing one concept on the others. To address this limitation, we model the concept space as a graph and empirically analyze the effects of erasing one concept on the remaining concepts. Our analysis uncovers intriguing geometric properties of the concept space, where the influence of erasing a concept is confined to a local region. Building on this insight, we propose the Adaptive Guided Erasure (AGE) method, which \emph{dynamically} selects optimal target concepts tailored to each undesirable concept, minimizing unintended side effects. Experimental results show that AGE significantly outperforms state-of-the-art erasure methods on preserving unrelated concepts while maintaining effective erasure performance. Our code is published at {https://github.com/tuananhbui89/Adaptive-Guided-Erasure}.
>
---
#### [replaced 051] CostFilter-AD: Enhancing Anomaly Detection through Matching Cost Filtering
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.01476v2](http://arxiv.org/pdf/2505.01476v2)**

> **作者:** Zhe Zhang; Mingxiu Cai; Hanxiao Wang; Gaochang Wu; Tianyou Chai; Xiatian Zhu
>
> **备注:** 25 pages, 12 figures, 20 tables, accepted by Forty-Second International Conference on Machine Learning ( ICML 2025 ), link: https://icml.cc/virtual/2025/poster/46359
>
> **摘要:** Unsupervised anomaly detection (UAD) seeks to localize the anomaly mask of an input image with respect to normal samples. Either by reconstructing normal counterparts (reconstruction-based) or by learning an image feature embedding space (embedding-based), existing approaches fundamentally rely on image-level or feature-level matching to derive anomaly scores. Often, such a matching process is inaccurate yet overlooked, leading to sub-optimal detection. To address this issue, we introduce the concept of cost filtering, borrowed from classical matching tasks, such as depth and flow estimation, into the UAD problem. We call this approach {\em CostFilter-AD}. Specifically, we first construct a matching cost volume between the input and normal samples, comprising two spatial dimensions and one matching dimension that encodes potential matches. To refine this, we propose a cost volume filtering network, guided by the input observation as an attention query across multiple feature layers, which effectively suppresses matching noise while preserving edge structures and capturing subtle anomalies. Designed as a generic post-processing plug-in, CostFilter-AD can be integrated with either reconstruction-based or embedding-based methods. Extensive experiments on MVTec-AD and VisA benchmarks validate the generic benefits of CostFilter-AD for both single- and multi-class UAD tasks. Code and models will be released at https://github.com/ZHE-SAPI/CostFilter-AD.
>
---
#### [replaced 052] MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.00871v3](http://arxiv.org/pdf/2410.00871v3)**

> **作者:** Yunze Liu; Li Yi
>
> **摘要:** Hybrid Mamba-Transformer networks have recently garnered broad attention. These networks can leverage the scalability of Transformers while capitalizing on Mamba's strengths in long-context modeling and computational efficiency. However, the challenge of effectively pretraining such hybrid networks remains an open question. Existing methods, such as Masked Autoencoders (MAE) or autoregressive (AR) pretraining, primarily focus on single-type network architectures. In contrast, pretraining strategies for hybrid architectures must be effective for both Mamba and Transformer components. Based on this, we propose Masked Autoregressive Pretraining (MAP) to pretrain a hybrid Mamba-Transformer vision backbone network. This strategy combines the strengths of both MAE and Autoregressive pretraining, improving the performance of Mamba and Transformer modules within a unified paradigm. Experimental results show that the hybrid Mamba-Transformer vision backbone network pretrained with MAP significantly outperforms other pretraining strategies, achieving state-of-the-art performance. We validate the method's effectiveness on both 2D and 3D datasets and provide detailed ablation studies to support the design choices for each component. The code and checkpoints are available at https://github.com/yunzeliu/MAP
>
---
#### [replaced 053] Open-Set Gait Recognition from Sparse mmWave Radar Point Clouds
- **分类: cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2503.07435v3](http://arxiv.org/pdf/2503.07435v3)**

> **作者:** Riccardo Mazzieri; Jacopo Pegoraro; Michele Rossi
>
> **摘要:** The adoption of Millimeter-Wave (mmWave) radar devices for human sensing, particularly gait recognition, has recently gathered significant attention due to their efficiency, resilience to environmental conditions, and privacy-preserving nature. In this work, we tackle the challenging problem of Open-set Gait Recognition (OSGR) from sparse mmWave radar point clouds. Unlike most existing research, which assumes a closed-set scenario, our work considers the more realistic open-set case, where unknown subjects might be present at inference time, and should be correctly recognized by the system. Point clouds are well-suited for edge computing applications with resource constraints, but are more significantly affected by noise and random fluctuations than other representations, like the more common micro-Doppler signature. This is the first work addressing open-set gait recognition with sparse point cloud data. To do so, we propose a novel neural network architecture that combines supervised classification with unsupervised reconstruction of the point clouds, creating a robust, rich, and highly regularized latent space of gait features. To detect unknown subjects at inference time, we introduce a probabilistic novelty detection algorithm that leverages the structured latent space and offers a tunable trade-off between inference speed and prediction accuracy. Along with this paper, we release mmGait10, an original human gait dataset featuring over five hours of measurements from ten subjects, under varied walking modalities. Extensive experimental results show that our solution attains F1-Score improvements by 24% over state-of-the-art methods, on average, and across multiple openness levels.
>
---
#### [replaced 054] SceneTracker: Long-term Scene Flow Estimation Network
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2403.19924v4](http://arxiv.org/pdf/2403.19924v4)**

> **作者:** Bo Wang; Jian Li; Yang Yu; Li Liu; Zhenping Sun; Dewen Hu
>
> **摘要:** Considering that scene flow estimation has the capability of the spatial domain to focus but lacks the coherence of the temporal domain, this study proposes long-term scene flow estimation (LSFE), a comprehensive task that can simultaneously capture the fine-grained and long-term 3D motion in an online manner. We introduce SceneTracker, the first LSFE network that adopts an iterative approach to approximate the optimal 3D trajectory. The network dynamically and simultaneously indexes and constructs appearance correlation and depth residual features. Transformers are then employed to explore and utilize long-range connections within and between trajectories. With detailed experiments, SceneTracker shows superior capabilities in addressing 3D spatial occlusion and depth noise interference, highly tailored to the needs of the LSFE task. We build a real-world evaluation dataset, LSFDriving, for the LSFE field and use it in experiments to further demonstrate the advantage of SceneTracker in generalization abilities. The code and data are available at https://github.com/wwsource/SceneTracker.
>
---
#### [replaced 055] TDFormer: A Top-Down Attention-Controlled Spiking Transformer
- **分类: cs.NE; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15840v2](http://arxiv.org/pdf/2505.15840v2)**

> **作者:** Zizheng Zhu; Yingchao Yu; Zeqi Zheng; Zhaofei Yu; Yaochu Jin
>
> **备注:** 28 pages
>
> **摘要:** Traditional spiking neural networks (SNNs) can be viewed as a combination of multiple subnetworks with each running for one time step, where the parameters are shared, and the membrane potential serves as the only information link between them. However, the implicit nature of the membrane potential limits its ability to effectively represent temporal information. As a result, each time step cannot fully leverage information from previous time steps, seriously limiting the model's performance. Inspired by the top-down mechanism in the brain, we introduce TDFormer, a novel model with a top-down feedback structure that functions hierarchically and leverages high-order representations from earlier time steps to modulate the processing of low-order information at later stages. The feedback structure plays a role from two perspectives: 1) During forward propagation, our model increases the mutual information across time steps, indicating that richer temporal information is being transmitted and integrated in different time steps. 2) During backward propagation, we theoretically prove that the feedback structure alleviates the problem of vanishing gradients along the time dimension. We find that these mechanisms together significantly and consistently improve the model performance on multiple datasets. In particular, our model achieves state-of-the-art performance on ImageNet with an accuracy of 86.83%.
>
---
#### [replaced 056] A Comprehensive Assessment Benchmark for Rigorously Evaluating Deep Learning Image Classifiers
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2308.04137v3](http://arxiv.org/pdf/2308.04137v3)**

> **作者:** Michael W. Spratling
>
> **摘要:** Reliable and robust evaluation methods are a necessary first step towards developing machine learning models that are themselves robust and reliable. Unfortunately, current evaluation protocols typically used to assess classifiers fail to comprehensively evaluate performance as they tend to rely on limited types of test data, and ignore others. For example, using the standard test data fails to evaluate the predictions made by the classifier to samples from classes it was not trained on. On the other hand, testing with data containing samples from unknown classes fails to evaluate how well the classifier can predict the labels for known classes. This article advocates benchmarking performance using a wide range of different types of data and using a single metric that can be applied to all such data types to produce a consistent evaluation of performance. Using the proposed benchmark it is found that current deep neural networks, including those trained with methods that are believed to produce state-of-the-art robustness, are vulnerable to making mistakes on certain types of data. This means that such models will be unreliable in real-world scenarios where they may encounter data from many different domains, and that they are insecure as they can be easily fooled into making the wrong decisions. It is hoped that these results will motivate the wider adoption of more comprehensive testing methods that will, in turn, lead to the development of more robust machine learning methods in the future. Code is available at: https://codeberg.org/mwspratling/RobustnessEvaluation
>
---
#### [replaced 057] Think or Not? Selective Reasoning via Reinforcement Learning for Vision-Language Models
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16854v2](http://arxiv.org/pdf/2505.16854v2)**

> **作者:** Jiaqi Wang; Kevin Qinghong Lin; James Cheng; Mike Zheng Shou
>
> **备注:** update more examples in appendix
>
> **摘要:** Reinforcement Learning (RL) has proven to be an effective post-training strategy for enhancing reasoning in vision-language models (VLMs). Group Relative Policy Optimization (GRPO) is a recent prominent method that encourages models to generate complete reasoning traces before answering, leading to increased token usage and computational cost. Inspired by the human-like thinking process-where people skip reasoning for easy questions but think carefully when needed-we explore how to enable VLMs to first decide when reasoning is necessary. To realize this, we propose TON, a two-stage training strategy: (i) a supervised fine-tuning (SFT) stage with a simple yet effective 'thought dropout' operation, where reasoning traces are randomly replaced with empty thoughts. This introduces a think-or-not format that serves as a cold start for selective reasoning; (ii) a GRPO stage that enables the model to freely explore when to think or not, while maximizing task-aware outcome rewards. Experimental results show that TON can reduce the completion length by up to 90% compared to vanilla GRPO, without sacrificing performance or even improving it. Further evaluations across diverse vision-language tasks-covering a range of reasoning difficulties under both 3B and 7B models-consistently reveal that the model progressively learns to bypass unnecessary reasoning steps as training advances. These findings shed light on the path toward human-like reasoning patterns in reinforcement learning approaches. Our code is available at https://github.com/kokolerk/TON.
>
---
#### [replaced 058] REG: Rectified Gradient Guidance for Conditional Diffusion Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.18865v2](http://arxiv.org/pdf/2501.18865v2)**

> **作者:** Zhengqi Gao; Kaiwen Zha; Tianyuan Zhang; Zihui Xue; Duane S. Boning
>
> **备注:** 20 pages, 10 figures; accepted by ICML'25
>
> **摘要:** Guidance techniques are simple yet effective for improving conditional generation in diffusion models. Albeit their empirical success, the practical implementation of guidance diverges significantly from its theoretical motivation. In this paper, we reconcile this discrepancy by replacing the scaled marginal distribution target, which we prove theoretically invalid, with a valid scaled joint distribution objective. Additionally, we show that the established guidance implementations are approximations to the intractable optimal solution under no future foresight constraint. Building on these theoretical insights, we propose rectified gradient guidance (REG), a versatile enhancement designed to boost the performance of existing guidance methods. Experiments on 1D and 2D demonstrate that REG provides a better approximation to the optimal solution than prior guidance techniques, validating the proposed theoretical framework. Extensive experiments on class-conditional ImageNet and text-to-image generation tasks show that incorporating REG consistently improves FID and Inception/CLIP scores across various settings compared to its absence.
>
---
#### [replaced 059] VR-FuseNet: A Fusion of Heterogeneous Fundus Data and Explainable Deep Network for Diabetic Retinopathy Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.21464v2](http://arxiv.org/pdf/2504.21464v2)**

> **作者:** Shamim Rahim Refat; Ziyan Shirin Raha; Shuvashis Sarker; Faika Fairuj Preotee; MD. Musfikur Rahman; Tashreef Muhammad; Mohammad Shafiul Alam
>
> **备注:** 33 pages, 49 figures
>
> **摘要:** Diabetic retinopathy is a severe eye condition caused by diabetes where the retinal blood vessels get damaged and can lead to vision loss and blindness if not treated. Early and accurate detection is key to intervention and stopping the disease progressing. For addressing this disease properly, this paper presents a comprehensive approach for automated diabetic retinopathy detection by proposing a new hybrid deep learning model called VR-FuseNet. Diabetic retinopathy is a major eye disease and leading cause of blindness especially among diabetic patients so accurate and efficient automated detection methods are required. To address the limitations of existing methods including dataset imbalance, diversity and generalization issues this paper presents a hybrid dataset created from five publicly available diabetic retinopathy datasets. Essential preprocessing techniques such as SMOTE for class balancing and CLAHE for image enhancement are applied systematically to the dataset to improve the robustness and generalizability of the dataset. The proposed VR-FuseNet model combines the strengths of two state-of-the-art convolutional neural networks, VGG19 which captures fine-grained spatial features and ResNet50V2 which is known for its deep hierarchical feature extraction. This fusion improves the diagnostic performance and achieves an accuracy of 91.824%. The model outperforms individual architectures on all performance metrics demonstrating the effectiveness of hybrid feature extraction in Diabetic Retinopathy classification tasks. To make the proposed model more clinically useful and interpretable this paper incorporates multiple XAI techniques. These techniques generate visual explanations that clearly indicate the retinal features affecting the model's prediction such as microaneurysms, hemorrhages and exudates so that clinicians can interpret and validate.
>
---
#### [replaced 060] Exploring Implicit Visual Misunderstandings in Multimodal Large Language Models through Attention Analysis
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.10541v2](http://arxiv.org/pdf/2505.10541v2)**

> **作者:** Pengfei Wang; Guohai Xu; Weinong Wang; Junjie Yang; Jie Lou; Yunhua Xue
>
> **摘要:** Recent advancements have enhanced the capability of Multimodal Large Language Models (MLLMs) to comprehend multi-image information. However, existing benchmarks primarily evaluate answer correctness, overlooking whether models genuinely comprehend the visual input. To address this, we define implicit visual misunderstanding (IVM), where MLLMs provide correct answers without fully comprehending the visual input. Through our analysis, we decouple the visual and textual modalities within the causal attention module, revealing that attention distribution increasingly converges on the image associated with the correct answer as the network layers deepen. This insight leads to the introduction of a scale-agnostic metric, \textit{attention accuracy}, and a novel benchmark for quantifying IVMs. Attention accuracy directly evaluates the model's visual understanding via internal mechanisms, remaining robust to positional biases for more reliable assessments. Furthermore, we extend our approach to finer granularities and demonstrate its effectiveness in unimodal scenarios, underscoring its versatility and generalizability.
>
---
#### [replaced 061] EMT: A Visual Multi-Task Benchmark Dataset for Autonomous Driving
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.19260v4](http://arxiv.org/pdf/2502.19260v4)**

> **作者:** Nadya Abdel Madjid; Murad Mebrahtu; Abdulrahman Ahmad; Abdelmoamen Nasser; Bilal Hassan; Naoufel Werghi; Jorge Dias; Majid Khonji
>
> **备注:** 19 pages, 6 figures
>
> **摘要:** This paper introduces the Emirates Multi-Task (EMT) dataset, designed to support multi-task benchmarking within a unified framework. It comprises over 30,000 frames from a dash-camera perspective and 570,000 annotated bounding boxes, covering approximately 150 kilometers of driving routes that reflect the distinctive road topology, congestion patterns, and driving behavior of Gulf region traffic. The dataset supports three primary tasks: tracking, trajectory forecasting, and intention prediction. Each benchmark is accompanied by corresponding evaluations: (1) multi-agent tracking experiments addressing multi-class scenarios and occlusion handling; (2) trajectory forecasting evaluation using deep sequential and interaction-aware models; and (3) intention prediction experiments based on observed trajectories. The dataset is publicly available at https://avlab.io/emt-dataset, with pre-processing scripts and evaluation models at https://github.com/AV-Lab/emt-dataset.
>
---
#### [replaced 062] Preconditioners for the Stochastic Training of Neural Fields
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.08784v2](http://arxiv.org/pdf/2402.08784v2)**

> **作者:** Shin-Fang Chng; Hemanth Saratchandran; Simon Lucey
>
> **备注:** The first two authors contributed equally. CVPR 2025
>
> **摘要:** Neural fields encode continuous multidimensional signals as neural networks, enabling diverse applications in computer vision, robotics, and geometry. While Adam is effective for stochastic optimization, it often requires long training times. To address this, we explore alternative optimization techniques to accelerate training without sacrificing accuracy. Traditional second-order methods like L-BFGS are unsuitable for stochastic settings. We propose a theoretical framework for training neural fields with curvature-aware diagonal preconditioners, demonstrating their effectiveness across tasks such as image reconstruction, shape modeling, and Neural Radiance Fields (NeRF).
>
---
#### [replaced 063] Autoregressive Sequence Modeling for 3D Medical Image Representation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.08691v2](http://arxiv.org/pdf/2409.08691v2)**

> **作者:** Siwen Wang; Churan Wang; Fei Gao; Lixian Su; Fandong Zhang; Yizhou Wang; Yizhou Yu
>
> **备注:** Accepted by AAAI 2025
>
> **摘要:** Three-dimensional (3D) medical images, such as Computed Tomography (CT) and Magnetic Resonance Imaging (MRI), are essential for clinical applications. However, the need for diverse and comprehensive representations is particularly pronounced when considering the variability across different organs, diagnostic tasks, and imaging modalities. How to effectively interpret the intricate contextual information and extract meaningful insights from these images remains an open challenge to the community. While current self-supervised learning methods have shown potential, they often consider an image as a whole thereby overlooking the extensive, complex relationships among local regions from one or multiple images. In this work, we introduce a pioneering method for learning 3D medical image representations through an autoregressive pre-training framework. Our approach sequences various 3D medical images based on spatial, contrast, and semantic correlations, treating them as interconnected visual tokens within a token sequence. By employing an autoregressive sequence modeling task, we predict the next visual token in the sequence, which allows our model to deeply understand and integrate the contextual information inherent in 3D medical images. Additionally, we implement a random startup strategy to avoid overestimating token relationships and to enhance the robustness of learning. The effectiveness of our approach is demonstrated by the superior performance over others on nine downstream tasks in public datasets.
>
---
#### [replaced 064] RBench-V: A Primary Assessment for Visual Reasoning Models with Multi-modal Outputs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16770v2](http://arxiv.org/pdf/2505.16770v2)**

> **作者:** Meng-Hao Guo; Xuanyu Chu; Qianrui Yang; Zhe-Han Mo; Yiqing Shen; Pei-lin Li; Xinjie Lin; Jinnian Zhang; Xin-Sheng Chen; Yi Zhang; Kiyohiro Nakayama; Zhengyang Geng; Houwen Peng; Han Hu; Shi-Min Hu
>
> **备注:** 12 pages
>
> **摘要:** The rapid advancement of native multi-modal models and omni-models, exemplified by GPT-4o, Gemini, and o3, with their capability to process and generate content across modalities such as text and images, marks a significant milestone in the evolution of intelligence. Systematic evaluation of their multi-modal output capabilities in visual thinking processes (also known as multi-modal chain of thought, M-CoT) becomes critically important. However, existing benchmarks for evaluating multi-modal models primarily focus on assessing multi-modal inputs and text-only reasoning while neglecting the importance of reasoning through multi-modal outputs. In this paper, we present a benchmark, dubbed RBench-V, designed to assess models' vision-indispensable reasoning abilities. To construct RBench-V, we carefully hand-pick 803 questions covering math, physics, counting, and games. Unlike previous benchmarks that typically specify certain input modalities, RBench-V presents problems centered on multi-modal outputs, which require image manipulation such as generating novel images and constructing auxiliary lines to support the reasoning process. We evaluate numerous open- and closed-source models on RBench-V, including o3, Gemini 2.5 Pro, Qwen2.5-VL, etc. Even the best-performing model, o3, achieves only 25.8% accuracy on RBench-V, far below the human score of 82.3%, highlighting that current models struggle to leverage multi-modal reasoning. Data and code are available at https://evalmodels.github.io/rbenchv
>
---
#### [replaced 065] Hypergraph Tversky-Aware Domain Incremental Learning for Brain Tumor Segmentation with Missing Modalities
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2505.16809v2](http://arxiv.org/pdf/2505.16809v2)**

> **作者:** Junze Wang; Lei Fan; Weipeng Jing; Donglin Di; Yang Song; Sidong Liu; Cong Cong
>
> **备注:** MICCAI 2025 Early Accept. The code is available at https://github.com/reeive/ReHyDIL
>
> **摘要:** Existing methods for multimodal MRI segmentation with missing modalities typically assume that all MRI modalities are available during training. However, in clinical practice, some modalities may be missing due to the sequential nature of MRI acquisition, leading to performance degradation. Furthermore, retraining models to accommodate newly available modalities can be inefficient and may cause overfitting, potentially compromising previously learned knowledge. To address these challenges, we propose Replay-based Hypergraph Domain Incremental Learning (ReHyDIL) for brain tumor segmentation with missing modalities. ReHyDIL leverages Domain Incremental Learning (DIL) to enable the segmentation model to learn from newly acquired MRI modalities without forgetting previously learned information. To enhance segmentation performance across diverse patient scenarios, we introduce the Cross-Patient Hypergraph Segmentation Network (CHSNet), which utilizes hypergraphs to capture high-order associations between patients. Additionally, we incorporate Tversky-Aware Contrastive (TAC) loss to effectively mitigate information imbalance both across and within different modalities. Extensive experiments on the BraTS2019 dataset demonstrate that ReHyDIL outperforms state-of-the-art methods, achieving an improvement of over 2% in the Dice Similarity Coefficient across various tumor regions.
>
---
