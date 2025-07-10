# 计算机视觉 cs.CV

- **最新发布 105 篇**

- **更新 77 篇**

## 最新发布

#### [new 001] EXAONE Path 2.0: Pathology Foundation Model with End-to-End Supervision
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于数字病理学任务，解决WSI处理效率与特征提取不足的问题。通过直接滑片级监督学习，提升数据效率与 biomarker 预测性能。**

- **链接: [http://arxiv.org/pdf/2507.06639v1](http://arxiv.org/pdf/2507.06639v1)**

> **作者:** Myungjang Pyeon; Janghyeon Lee; Minsoo Lee; Juseung Yun; Hwanil Choi; Jonghyun Kim; Jiwon Kim; Yi Hu; Jongseong Jang; Soonyoung Lee
>
> **备注:** EXAONE Path 2.0 technical report
>
> **摘要:** In digital pathology, whole-slide images (WSIs) are often difficult to handle due to their gigapixel scale, so most approaches train patch encoders via self-supervised learning (SSL) and then aggregate the patch-level embeddings via multiple instance learning (MIL) or slide encoders for downstream tasks. However, patch-level SSL may overlook complex domain-specific features that are essential for biomarker prediction, such as mutation status and molecular characteristics, as SSL methods rely only on basic augmentations selected for natural image domains on small patch-level area. Moreover, SSL methods remain less data efficient than fully supervised approaches, requiring extensive computational resources and datasets to achieve competitive performance. To address these limitations, we present EXAONE Path 2.0, a pathology foundation model that learns patch-level representations under direct slide-level supervision. Using only 37k WSIs for training, EXAONE Path 2.0 achieves state-of-the-art average performance across 10 biomarker prediction tasks, demonstrating remarkable data efficiency.
>
---
#### [new 002] Evaluating Large Multimodal Models for Nutrition Analysis: A Benchmark Enriched with Contextual Metadata
- **分类: cs.CV**

- **简介: 该论文属于营养分析任务，旨在提升大型多模态模型对食物图像的营养估算准确性。通过引入上下文元数据和推理修饰方法，优化模型性能。**

- **链接: [http://arxiv.org/pdf/2507.07048v1](http://arxiv.org/pdf/2507.07048v1)**

> **作者:** Bruce Coburn; Jiangpeng He; Megan E. Rollo; Satvinder S. Dhaliwal; Deborah A. Kerr; Fengqing Zhu
>
> **摘要:** Large Multimodal Models (LMMs) are increasingly applied to meal images for nutrition analysis. However, existing work primarily evaluates proprietary models, such as GPT-4. This leaves the broad range of LLMs underexplored. Additionally, the influence of integrating contextual metadata and its interaction with various reasoning modifiers remains largely uncharted. This work investigates how interpreting contextual metadata derived from GPS coordinates (converted to location/venue type), timestamps (transformed into meal/day type), and the food items present can enhance LMM performance in estimating key nutritional values. These values include calories, macronutrients (protein, carbohydrates, fat), and portion sizes. We also introduce ACETADA, a new food-image dataset slated for public release. This open dataset provides nutrition information verified by the dietitian and serves as the foundation for our analysis. Our evaluation across eight LMMs (four open-weight and four closed-weight) first establishes the benefit of contextual metadata integration over straightforward prompting with images alone. We then demonstrate how this incorporation of contextual information enhances the efficacy of reasoning modifiers, such as Chain-of-Thought, Multimodal Chain-of-Thought, Scale Hint, Few-Shot, and Expert Persona. Empirical results show that integrating metadata intelligently, when applied through straightforward prompting strategies, can significantly reduce the Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE) in predicted nutritional values. This work highlights the potential of context-aware LMMs for improved nutrition analysis.
>
---
#### [new 003] IAP: Invisible Adversarial Patch Attack through Perceptibility-Aware Localization and Perturbation Optimization
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于对抗攻击任务，旨在生成隐蔽的对抗补丁。通过感知意识定位和扰动优化，提高补丁的不可见性并增强攻击效果。**

- **链接: [http://arxiv.org/pdf/2507.06856v1](http://arxiv.org/pdf/2507.06856v1)**

> **作者:** Subrat Kishore Dutta; Xiao Zhang
>
> **备注:** Published in ICCV 2025
>
> **摘要:** Despite modifying only a small localized input region, adversarial patches can drastically change the prediction of computer vision models. However, prior methods either cannot perform satisfactorily under targeted attack scenarios or fail to produce contextually coherent adversarial patches, causing them to be easily noticeable by human examiners and insufficiently stealthy against automatic patch defenses. In this paper, we introduce IAP, a novel attack framework that generates highly invisible adversarial patches based on perceptibility-aware localization and perturbation optimization schemes. Specifically, IAP first searches for a proper location to place the patch by leveraging classwise localization and sensitivity maps, balancing the susceptibility of patch location to both victim model prediction and human visual system, then employs a perceptibility-regularized adversarial loss and a gradient update rule that prioritizes color constancy for optimizing invisible perturbations. Comprehensive experiments across various image benchmarks and model architectures demonstrate that IAP consistently achieves competitive attack success rates in targeted settings with significantly improved patch invisibility compared to existing baselines. In addition to being highly imperceptible to humans, IAP is shown to be stealthy enough to render several state-of-the-art patch defenses ineffective.
>
---
#### [new 004] Democratizing High-Fidelity Co-Speech Gesture Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于语音同步手势视频生成任务，旨在解决音频与视觉内容映射困难及数据稀缺问题。提出轻量框架，利用骨骼信息提升生成质量与同步性。**

- **链接: [http://arxiv.org/pdf/2507.06812v1](http://arxiv.org/pdf/2507.06812v1)**

> **作者:** Xu Yang; Shaoli Huang; Shenbo Xie; Xuelin Chen; Yifei Liu; Changxing Ding
>
> **备注:** ICCV 2025
>
> **摘要:** Co-speech gesture video generation aims to synthesize realistic, audio-aligned videos of speakers, complete with synchronized facial expressions and body gestures. This task presents challenges due to the significant one-to-many mapping between audio and visual content, further complicated by the scarcity of large-scale public datasets and high computational demands. We propose a lightweight framework that utilizes 2D full-body skeletons as an efficient auxiliary condition to bridge audio signals with visual outputs. Our approach introduces a diffusion model conditioned on fine-grained audio segments and a skeleton extracted from the speaker's reference image, predicting skeletal motions through skeleton-audio feature fusion to ensure strict audio coordination and body shape consistency. The generated skeletons are then fed into an off-the-shelf human video generation model with the speaker's reference image to synthesize high-fidelity videos. To democratize research, we present CSG-405-the first public dataset with 405 hours of high-resolution videos across 71 speech types, annotated with 2D skeletons and diverse speaker demographics. Experiments show that our method exceeds state-of-the-art approaches in visual quality and synchronization while generalizing across speakers and contexts.
>
---
#### [new 005] MK-Pose: Category-Level Object Pose Estimation via Multimodal-Based Keypoint Learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于类别级目标位姿估计任务，解决物体遮挡和跨实例泛化问题。提出MK-Pose框架，融合RGB、点云和文本信息，提升位姿估计精度。**

- **链接: [http://arxiv.org/pdf/2507.06662v1](http://arxiv.org/pdf/2507.06662v1)**

> **作者:** Yifan Yang; Peili Song; Enfan Lan; Dong Liu; Jingtai Liu
>
> **摘要:** Category-level object pose estimation, which predicts the pose of objects within a known category without prior knowledge of individual instances, is essential in applications like warehouse automation and manufacturing. Existing methods relying on RGB images or point cloud data often struggle with object occlusion and generalization across different instances and categories. This paper proposes a multimodal-based keypoint learning framework (MK-Pose) that integrates RGB images, point clouds, and category-level textual descriptions. The model uses a self-supervised keypoint detection module enhanced with attention-based query generation, soft heatmap matching and graph-based relational modeling. Additionally, a graph-enhanced feature fusion module is designed to integrate local geometric information and global context. MK-Pose is evaluated on CAMERA25 and REAL275 dataset, and is further tested for cross-dataset capability on HouseCat6D dataset. The results demonstrate that MK-Pose outperforms existing state-of-the-art methods in both IoU and average precision without shape priors. Codes will be released at \href{https://github.com/yangyifanYYF/MK-Pose}{https://github.com/yangyifanYYF/MK-Pose}.
>
---
#### [new 006] LIRA: Inferring Segmentation in Large Multi-modal Models with Local Interleaved Region Assistance
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多模态分割任务，旨在解决大模型分割不准确和理解错误的问题。提出LIRA框架，通过语义增强和局部耦合提升分割精度与理解能力。**

- **链接: [http://arxiv.org/pdf/2507.06272v1](http://arxiv.org/pdf/2507.06272v1)**

> **作者:** Zhang Li; Biao Yang; Qiang Liu; Shuo Zhang; Zhiyin Ma; Shuo Zhang; Liang Yin; Linger Deng; Yabo Sun; Yuliang Liu; Xiang Bai
>
> **摘要:** While large multi-modal models (LMMs) demonstrate promising capabilities in segmentation and comprehension, they still struggle with two limitations: inaccurate segmentation and hallucinated comprehension. These challenges stem primarily from constraints in weak visual comprehension and a lack of fine-grained perception. To alleviate these limitations, we propose LIRA, a framework that capitalizes on the complementary relationship between visual comprehension and segmentation via two key components: (1) Semantic-Enhanced Feature Extractor (SEFE) improves object attribute inference by fusing semantic and pixel-level features, leading to more accurate segmentation; (2) Interleaved Local Visual Coupling (ILVC) autoregressively generates local descriptions after extracting local features based on segmentation masks, offering fine-grained supervision to mitigate hallucinations. Furthermore, we find that the precision of object segmentation is positively correlated with the latent related semantics of the <seg> token. To quantify this relationship and the model's potential semantic inferring ability, we introduce the Attributes Evaluation (AttrEval) dataset. Our experiments show that LIRA achieves state-of-the-art performance in both segmentation and comprehension tasks. Code will be available at https://github.com/echo840/LIRA.
>
---
#### [new 007] Diff$^2$I2P: Differentiable Image-to-Point Cloud Registration with Diffusion Prior
- **分类: cs.CV**

- **简介: 该论文属于图像到点云配准任务，旨在解决跨模态对应不准确的问题。通过引入扩散先验和可微模块，提升配准精度。**

- **链接: [http://arxiv.org/pdf/2507.06651v1](http://arxiv.org/pdf/2507.06651v1)**

> **作者:** Juncheng Mu; Chengwei Ren; Weixiang Zhang; Liang Pan; Xiao-Ping Zhang; Yue Gao
>
> **备注:** ICCV 2025
>
> **摘要:** Learning cross-modal correspondences is essential for image-to-point cloud (I2P) registration. Existing methods achieve this mostly by utilizing metric learning to enforce feature alignment across modalities, disregarding the inherent modality gap between image and point data. Consequently, this paradigm struggles to ensure accurate cross-modal correspondences. To this end, inspired by the cross-modal generation success of recent large diffusion models, we propose Diff$^2$I2P, a fully Differentiable I2P registration framework, leveraging a novel and effective Diffusion prior for bridging the modality gap. Specifically, we propose a Control-Side Score Distillation (CSD) technique to distill knowledge from a depth-conditioned diffusion model to directly optimize the predicted transformation. However, the gradients on the transformation fail to backpropagate onto the cross-modal features due to the non-differentiability of correspondence retrieval and PnP solver. To this end, we further propose a Deformable Correspondence Tuning (DCT) module to estimate the correspondences in a differentiable way, followed by the transformation estimation using a differentiable PnP solver. With these two designs, the Diffusion model serves as a strong prior to guide the cross-modal feature learning of image and point cloud for forming robust correspondences, which significantly improves the registration. Extensive experimental results demonstrate that Diff$^2$I2P consistently outperforms SoTA I2P registration methods, achieving over 7% improvement in registration recall on the 7-Scenes benchmark.
>
---
#### [new 008] MCCD: A Multi-Attribute Chinese Calligraphy Character Dataset Annotated with Script Styles, Dynasties, and Calligraphers
- **分类: cs.CV**

- **简介: 该论文提出MCCD数据集，解决中文书法字符多属性识别问题，涵盖风格、朝代和书家信息，用于书法识别与研究。**

- **链接: [http://arxiv.org/pdf/2507.06948v1](http://arxiv.org/pdf/2507.06948v1)**

> **作者:** Yixin Zhao; Yuyi Zhang; Lianwen Jin
>
> **备注:** 17 pages, 8 figures, 9 tables, accepted by the 19th International Conference on Document Analysis and Recognition (ICDAR 2025)
>
> **摘要:** Research on the attribute information of calligraphy, such as styles, dynasties, and calligraphers, holds significant cultural and historical value. However, the styles of Chinese calligraphy characters have evolved dramatically through different dynasties and the unique touches of calligraphers, making it highly challenging to accurately recognize these different characters and their attributes. Furthermore, existing calligraphic datasets are extremely scarce, and most provide only character-level annotations without additional attribute information. This limitation has significantly hindered the in-depth study of Chinese calligraphy. To fill this gap, we present a novel Multi-Attribute Chinese Calligraphy Character Dataset (MCCD). The dataset encompasses 7,765 categories with a total of 329,715 isolated image samples of Chinese calligraphy characters, and three additional subsets were extracted based on the attribute labeling of the three types of script styles (10 types), dynasties (15 periods) and calligraphers (142 individuals). The rich multi-attribute annotations render MCCD well-suited diverse research tasks, including calligraphic character recognition, writer identification, and evolutionary studies of Chinese characters. We establish benchmark performance through single-task and multi-task recognition experiments across MCCD and all of its subsets. The experimental results demonstrate that the complexity of the stroke structure of the calligraphic characters, and the interplay between their different attributes, leading to a substantial increase in the difficulty of accurate recognition. MCCD not only fills a void in the availability of detailed calligraphy datasets but also provides valuable resources for advancing research in Chinese calligraphy and fostering advancements in multiple fields. The dataset is available at https://github.com/SCUT-DLVCLab/MCCD.
>
---
#### [new 009] Spatial-Temporal Graph Mamba for Music-Guided Dance Video Synthesis
- **分类: cs.CV**

- **简介: 该论文属于音乐引导舞蹈视频生成任务，旨在将音乐转换为舞蹈视频。提出STG-Mamba模型，通过音乐生成骨架序列并转化为视频，提升生成质量。**

- **链接: [http://arxiv.org/pdf/2507.06689v1](http://arxiv.org/pdf/2507.06689v1)**

> **作者:** Hao Tang; Ling Shao; Zhenyu Zhang; Luc Van Gool; Nicu Sebe
>
> **备注:** Accepted to TPAMI 2025
>
> **摘要:** We propose a novel spatial-temporal graph Mamba (STG-Mamba) for the music-guided dance video synthesis task, i.e., to translate the input music to a dance video. STG-Mamba consists of two translation mappings: music-to-skeleton translation and skeleton-to-video translation. In the music-to-skeleton translation, we introduce a novel spatial-temporal graph Mamba (STGM) block to effectively construct skeleton sequences from the input music, capturing dependencies between joints in both the spatial and temporal dimensions. For the skeleton-to-video translation, we propose a novel self-supervised regularization network to translate the generated skeletons, along with a conditional image, into a dance video. Lastly, we collect a new skeleton-to-video translation dataset from the Internet, containing 54,944 video clips. Extensive experiments demonstrate that STG-Mamba achieves significantly better results than existing methods.
>
---
#### [new 010] Hallucinating 360°: Panoramic Street-View Generation via Local Scenes Diffusion and Probabilistic Prompting
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文属于自动驾驶中的全景图像生成任务，旨在解决数据获取困难与生成可控性不足的问题。提出Percep360方法，结合局部场景扩散和概率提示技术，提升生成质量与可控性。**

- **链接: [http://arxiv.org/pdf/2507.06971v1](http://arxiv.org/pdf/2507.06971v1)**

> **作者:** Fei Teng; Kai Luo; Sheng Wu; Siyu Li; Pujun Guo; Jiale Wei; Kunyu Peng; Jiaming Zhang; Kailun Yang
>
> **备注:** The source code will be publicly available at https://github.com/Bryant-Teng/Percep360
>
> **摘要:** Panoramic perception holds significant potential for autonomous driving, enabling vehicles to acquire a comprehensive 360{\deg} surround view in a single shot. However, autonomous driving is a data-driven task. Complete panoramic data acquisition requires complex sampling systems and annotation pipelines, which are time-consuming and labor-intensive. Although existing street view generation models have demonstrated strong data regeneration capabilities, they can only learn from the fixed data distribution of existing datasets and cannot achieve high-quality, controllable panoramic generation. In this paper, we propose the first panoramic generation method Percep360 for autonomous driving. Percep360 enables coherent generation of panoramic data with control signals based on the stitched panoramic data. Percep360 focuses on two key aspects: coherence and controllability. Specifically, to overcome the inherent information loss caused by the pinhole sampling process, we propose the Local Scenes Diffusion Method (LSDM). LSDM reformulates the panorama generation as a spatially continuous diffusion process, bridging the gaps between different data distributions. Additionally, to achieve the controllable generation of panoramic images, we propose a Probabilistic Prompting Method (PPM). PPM dynamically selects the most relevant control cues, enabling controllable panoramic image generation. We evaluate the effectiveness of the generated images from three perspectives: image quality assessment (i.e., no-reference and with reference), controllability, and their utility in real-world Bird's Eye View (BEV) segmentation. Notably, the generated data consistently outperforms the original stitched images in no-reference quality metrics and enhances downstream perception models. The source code will be publicly available at https://github.com/Bryant-Teng/Percep360.
>
---
#### [new 011] Vision-Language-Vision Auto-Encoder: Scalable Knowledge Distillation from Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型任务，旨在减少训练数据和成本。通过知识蒸馏方法，利用预训练模型构建高效图像描述生成系统。**

- **链接: [http://arxiv.org/pdf/2507.07104v1](http://arxiv.org/pdf/2507.07104v1)**

> **作者:** Tiezheng Zhang; Yitong Li; Yu-cheng Chou; Jieneng Chen; Alan Yuille; Chen Wei; Junfei Xiao
>
> **摘要:** Building state-of-the-art Vision-Language Models (VLMs) with strong captioning capabilities typically necessitates training on billions of high-quality image-text pairs, requiring millions of GPU hours. This paper introduces the Vision-Language-Vision (VLV) auto-encoder framework, which strategically leverages key pretrained components: a vision encoder, the decoder of a Text-to-Image (T2I) diffusion model, and subsequently, a Large Language Model (LLM). Specifically, we establish an information bottleneck by regularizing the language representation space, achieved through freezing the pretrained T2I diffusion decoder. Our VLV pipeline effectively distills knowledge from the text-conditioned diffusion model using continuous embeddings, demonstrating comprehensive semantic understanding via high-quality reconstructions. Furthermore, by fine-tuning a pretrained LLM to decode the intermediate language representations into detailed descriptions, we construct a state-of-the-art (SoTA) captioner comparable to leading models like GPT-4o and Gemini 2.0 Flash. Our method demonstrates exceptional cost-efficiency and significantly reduces data requirements; by primarily utilizing single-modal images for training and maximizing the utility of existing pretrained models (image encoder, T2I diffusion model, and LLM), it circumvents the need for massive paired image-text datasets, keeping the total training expenditure under $1,000 USD.
>
---
#### [new 012] FOLC-Net: A Federated-Optimized Lightweight Architecture for Enhanced MRI Disease Diagnosis across Axial, Coronal, and Sagittal Views
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像诊断任务，旨在解决MRI多视角分析中模型性能下降的问题。提出FOLC-Net框架，结合优化算法与轻量结构，提升各视角诊断准确性。**

- **链接: [http://arxiv.org/pdf/2507.06763v1](http://arxiv.org/pdf/2507.06763v1)**

> **作者:** Saif Ur Rehman Khan; Muhammad Nabeel Asim; Sebastian Vollmer; Andreas Dengel
>
> **摘要:** The framework is designed to improve performance in the analysis of combined as well as single anatomical perspectives for MRI disease diagnosis. It specifically addresses the performance degradation observed in state-of-the-art (SOTA) models, particularly when processing axial, coronal, and sagittal anatomical planes. The paper introduces the FOLC-Net framework, which incorporates a novel federated-optimized lightweight architecture with approximately 1.217 million parameters and a storage requirement of only 0.9 MB. FOLC-Net integrates Manta-ray foraging optimization (MRFO) mechanisms for efficient model structure generation, global model cloning for scalable training, and ConvNeXt for enhanced client adaptability. The model was evaluated on combined multi-view data as well as individual views, such as axial, coronal, and sagittal, to assess its robustness in various medical imaging scenarios. Moreover, FOLC-Net tests a ShallowFed model on different data to evaluate its ability to generalize beyond the training dataset. The results show that FOLC-Net outperforms existing models, particularly in the challenging sagittal view. For instance, FOLC-Net achieved an accuracy of 92.44% on the sagittal view, significantly higher than the 88.37% accuracy of study method (DL + Residual Learning) and 88.95% of DL models. Additionally, FOLC-Net demonstrated improved accuracy across all individual views, providing a more reliable and robust solution for medical image analysis in decentralized environments. FOLC-Net addresses the limitations of existing SOTA models by providing a framework that ensures better adaptability to individual views while maintaining strong performance in multi-view settings. The incorporation of MRFO, global model cloning, and ConvNeXt ensures that FOLC-Net performs better in real-world medical applications.
>
---
#### [new 013] GreenHyperSpectra: A multi-source hyperspectral dataset for global vegetation trait prediction
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于植物性状预测任务，解决标签稀缺和领域迁移问题。构建了GreenHyperSpectra数据集，用于半监督和自监督学习，提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.06806v1](http://arxiv.org/pdf/2507.06806v1)**

> **作者:** Eya Cherif; Arthur Ouaknine; Luke A. Brown; Phuong D. Dao; Kyle R. Kovach; Bing Lu; Daniel Mederer; Hannes Feilhauer; Teja Kattenborn; David Rolnick
>
> **摘要:** Plant traits such as leaf carbon content and leaf mass are essential variables in the study of biodiversity and climate change. However, conventional field sampling cannot feasibly cover trait variation at ecologically meaningful spatial scales. Machine learning represents a valuable solution for plant trait prediction across ecosystems, leveraging hyperspectral data from remote sensing. Nevertheless, trait prediction from hyperspectral data is challenged by label scarcity and substantial domain shifts (\eg across sensors, ecological distributions), requiring robust cross-domain methods. Here, we present GreenHyperSpectra, a pretraining dataset encompassing real-world cross-sensor and cross-ecosystem samples designed to benchmark trait prediction with semi- and self-supervised methods. We adopt an evaluation framework encompassing in-distribution and out-of-distribution scenarios. We successfully leverage GreenHyperSpectra to pretrain label-efficient multi-output regression models that outperform the state-of-the-art supervised baseline. Our empirical analyses demonstrate substantial improvements in learning spectral representations for trait prediction, establishing a comprehensive methodological framework to catalyze research at the intersection of representation learning and plant functional traits assessment. All code and data are available at: https://github.com/echerif18/HyspectraSSL.
>
---
#### [new 014] Bilateral Collaboration with Large Vision-Language Models for Open Vocabulary Human-Object Interaction Detection
- **分类: cs.CV**

- **简介: 该论文属于开放词汇人-物交互检测任务，解决VLM生成的粗粒度特征与检测任务不匹配的问题。提出BC-HOI框架，结合注意力引导和语言模型监督，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2507.06510v1](http://arxiv.org/pdf/2507.06510v1)**

> **作者:** Yupeng Hu; Changxing Ding; Chang Sun; Shaoli Huang; Xiangmin Xu
>
> **备注:** ICCV 2025
>
> **摘要:** Open vocabulary Human-Object Interaction (HOI) detection is a challenging task that detects all <human, verb, object> triplets of interest in an image, even those that are not pre-defined in the training set. Existing approaches typically rely on output features generated by large Vision-Language Models (VLMs) to enhance the generalization ability of interaction representations. However, the visual features produced by VLMs are holistic and coarse-grained, which contradicts the nature of detection tasks. To address this issue, we propose a novel Bilateral Collaboration framework for open vocabulary HOI detection (BC-HOI). This framework includes an Attention Bias Guidance (ABG) component, which guides the VLM to produce fine-grained instance-level interaction features according to the attention bias provided by the HOI detector. It also includes a Large Language Model (LLM)-based Supervision Guidance (LSG) component, which provides fine-grained token-level supervision for the HOI detector by the LLM component of the VLM. LSG enhances the ability of ABG to generate high-quality attention bias. We conduct extensive experiments on two popular benchmarks: HICO-DET and V-COCO, consistently achieving superior performance in the open vocabulary and closed settings. The code will be released in Github.
>
---
#### [new 015] FlexGaussian: Flexible and Cost-Effective Training-Free Compression for 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于3D场景压缩任务，解决大模型在资源受限设备上的部署问题。提出FlexGaussian方法，结合量化与剪枝实现无需训练的高效压缩。**

- **链接: [http://arxiv.org/pdf/2507.06671v1](http://arxiv.org/pdf/2507.06671v1)**

> **作者:** Boyuan Tian; Qizhe Gao; Siran Xianyu; Xiaotong Cui; Minjia Zhang
>
> **备注:** To appear at ACM MM 2025
>
> **摘要:** 3D Gaussian splatting has become a prominent technique for representing and rendering complex 3D scenes, due to its high fidelity and speed advantages. However, the growing demand for large-scale models calls for effective compression to reduce memory and computation costs, especially on mobile and edge devices with limited resources. Existing compression methods effectively reduce 3D Gaussian parameters but often require extensive retraining or fine-tuning, lacking flexibility under varying compression constraints. In this paper, we introduce FlexGaussian, a flexible and cost-effective method that combines mixed-precision quantization with attribute-discriminative pruning for training-free 3D Gaussian compression. FlexGaussian eliminates the need for retraining and adapts easily to diverse compression targets. Evaluation results show that FlexGaussian achieves up to 96.4% compression while maintaining high rendering quality (<1 dB drop in PSNR), and is deployable on mobile devices. FlexGaussian delivers high compression ratios within seconds, being 1.7-2.1x faster than state-of-the-art training-free methods and 10-100x faster than training-involved approaches. The code is being prepared and will be released soon at: https://github.com/Supercomputing-System-AI-Lab/FlexGaussian
>
---
#### [new 016] Physics-Grounded Motion Forecasting via Equation Discovery for Trajectory-Guided Image-to-Video Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频生成任务，旨在解决现有模型物理对齐不足的问题。通过结合符号回归与轨迹引导，提取并预测物理方程，提升视频生成的物理准确性。**

- **链接: [http://arxiv.org/pdf/2507.06830v1](http://arxiv.org/pdf/2507.06830v1)**

> **作者:** Tao Feng; Xianbing Zhao; Zhenhua Chen; Tien Tsin Wong; Hamid Rezatofighi; Gholamreza Haffari; Lizhen Qu
>
> **摘要:** Recent advances in diffusion-based and autoregressive video generation models have achieved remarkable visual realism. However, these models typically lack accurate physical alignment, failing to replicate real-world dynamics in object motion. This limitation arises primarily from their reliance on learned statistical correlations rather than capturing mechanisms adhering to physical laws. To address this issue, we introduce a novel framework that integrates symbolic regression (SR) and trajectory-guided image-to-video (I2V) models for physics-grounded video forecasting. Our approach extracts motion trajectories from input videos, uses a retrieval-based pre-training mechanism to enhance symbolic regression, and discovers equations of motion to forecast physically accurate future trajectories. These trajectories then guide video generation without requiring fine-tuning of existing models. Evaluated on scenarios in Classical Mechanics, including spring-mass, pendulums, and projectile motions, our method successfully recovers ground-truth analytical equations and improves the physical alignment of generated videos over baseline methods.
>
---
#### [new 017] Mask6D: Masked Pose Priors For 6D Object Pose Estimation
- **分类: cs.CV**

- **简介: 该论文属于6D物体位姿估计任务，解决遮挡和杂乱场景下位姿估计困难的问题。通过引入掩码和2D-3D对应图进行预训练，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.06486v1](http://arxiv.org/pdf/2507.06486v1)**

> **作者:** Yuechen Xie; Haobo Jiang; Jin Xie
>
> **备注:** Accepted at ICASSP 2024. 4 figures, 3 tables
>
> **摘要:** Robust 6D object pose estimation in cluttered or occluded conditions using monocular RGB images remains a challenging task. One reason is that current pose estimation networks struggle to extract discriminative, pose-aware features using 2D feature backbones, especially when the available RGB information is limited due to target occlusion in cluttered scenes. To mitigate this, we propose a novel pose estimation-specific pre-training strategy named Mask6D. Our approach incorporates pose-aware 2D-3D correspondence maps and visible mask maps as additional modal information, which is combined with RGB images for the reconstruction-based model pre-training. Essentially, this 2D-3D correspondence maps a transformed 3D object model to 2D pixels, reflecting the pose information of the target in camera coordinate system. Meanwhile, the integrated visible mask map can effectively guide our model to disregard cluttered background information. In addition, an object-focused pre-training loss function is designed to further facilitate our network to remove the background interference. Finally, we fine-tune our pre-trained pose prior-aware network via conventional pose training strategy to realize the reliable pose prediction. Extensive experiments verify that our method outperforms previous end-to-end pose estimation methods.
>
---
#### [new 018] Learning Deliberately, Acting Intuitively: Unlocking Test-Time Reasoning in Multimodal LLMs
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于多模态大语言模型任务，旨在解决模态对齐与训练成本高的问题。提出D2I框架，在无需额外标注的情况下提升模型推理能力。**

- **链接: [http://arxiv.org/pdf/2507.06999v1](http://arxiv.org/pdf/2507.06999v1)**

> **作者:** Yahan Yu; Yuyang Dong; Masafumi Oyamada
>
> **备注:** Work in progress
>
> **摘要:** Reasoning is a key capability for large language models (LLMs), particularly when applied to complex tasks such as mathematical problem solving. However, multimodal reasoning research still requires further exploration of modality alignment and training costs. Many of these approaches rely on additional data annotation and relevant rule-based rewards to enhance the understanding and reasoning ability, which significantly increases training costs and limits scalability. To address these challenges, we propose the Deliberate-to-Intuitive reasoning framework (D2I) that improves the understanding and reasoning ability of multimodal LLMs (MLLMs) without extra annotations and complex rewards. Specifically, our method sets deliberate reasoning strategies to enhance modality alignment only through the rule-based format reward during training. While evaluating, the reasoning style shifts to intuitive, which removes deliberate reasoning strategies during training and implicitly reflects the model's acquired abilities in the response. D2I outperforms baselines across both in-domain and out-of-domain benchmarks. Our findings highlight the role of format reward in fostering transferable reasoning skills in MLLMs, and inspire directions for decoupling training-time reasoning depth from test-time response flexibility.
>
---
#### [new 019] DenoiseCP-Net: Efficient Collective Perception in Adverse Weather via Joint LiDAR-Based 3D Object Detection and Denoising
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的多任务学习任务，旨在解决恶劣天气下LiDAR感知与通信效率问题。提出DenoiseCP-Net模型，实现降噪与目标检测的联合优化。**

- **链接: [http://arxiv.org/pdf/2507.06976v1](http://arxiv.org/pdf/2507.06976v1)**

> **作者:** Sven Teufel; Dominique Mayer; Jörg Gamerdinger; Oliver Bringmann
>
> **摘要:** While automated vehicles hold the potential to significantly reduce traffic accidents, their perception systems remain vulnerable to sensor degradation caused by adverse weather and environmental occlusions. Collective perception, which enables vehicles to share information, offers a promising approach to overcoming these limitations. However, to this date collective perception in adverse weather is mostly unstudied. Therefore, we conduct the first study of LiDAR-based collective perception under diverse weather conditions and present a novel multi-task architecture for LiDAR-based collective perception under adverse weather. Adverse weather conditions can not only degrade perception capabilities, but also negatively affect bandwidth requirements and latency due to the introduced noise that is also transmitted and processed. Denoising prior to communication can effectively mitigate these issues. Therefore, we propose DenoiseCP-Net, a novel multi-task architecture for LiDAR-based collective perception under adverse weather conditions. DenoiseCP-Net integrates voxel-level noise filtering and object detection into a unified sparse convolution backbone, eliminating redundant computations associated with two-stage pipelines. This design not only reduces inference latency and computational cost but also minimizes communication overhead by removing non-informative noise. We extended the well-known OPV2V dataset by simulating rain, snow, and fog using our realistic weather simulation models. We demonstrate that DenoiseCP-Net achieves near-perfect denoising accuracy in adverse weather, reduces the bandwidth requirements by up to 23.6% while maintaining the same detection accuracy and reducing the inference latency for cooperative vehicles.
>
---
#### [new 020] What Demands Attention in Urban Street Scenes? From Scene Understanding towards Road Safety: A Survey of Vision-driven Datasets and Studies
- **分类: cs.CV**

- **简介: 该论文属于交通场景理解任务，旨在提升道路安全。通过分类关键交通实体、分析35项视觉任务和73个数据集，提出统一框架，解决领域分散与标准不一问题。**

- **链接: [http://arxiv.org/pdf/2507.06513v1](http://arxiv.org/pdf/2507.06513v1)**

> **作者:** Yaoqi Huang; Julie Stephany Berrio; Mao Shan; Stewart Worrall
>
> **备注:** 45 pages, 52 figures, 2 large tables (divided into 5), 73 datatsets, 35 tasks
>
> **摘要:** Advances in vision-based sensors and computer vision algorithms have significantly improved the analysis and understanding of traffic scenarios. To facilitate the use of these improvements for road safety, this survey systematically categorizes the critical elements that demand attention in traffic scenarios and comprehensively analyzes available vision-driven tasks and datasets. Compared to existing surveys that focus on isolated domains, our taxonomy categorizes attention-worthy traffic entities into two main groups that are anomalies and normal but critical entities, integrating ten categories and twenty subclasses. It establishes connections between inherently related fields and provides a unified analytical framework. Our survey highlights the analysis of 35 vision-driven tasks and comprehensive examinations and visualizations of 73 available datasets based on the proposed taxonomy. The cross-domain investigation covers the pros and cons of each benchmark with the aim of providing information on standards unification and resource optimization. Our article concludes with a systematic discussion of the existing weaknesses, underlining the potential effects and promising solutions from various perspectives. The integrated taxonomy, comprehensive analysis, and recapitulatory tables serve as valuable contributions to this rapidly evolving field by providing researchers with a holistic overview, guiding strategic resource selection, and highlighting critical research gaps.
>
---
#### [new 021] Learning from Sparse Point Labels for Dense Carcinosis Localization in Advanced Ovarian Cancer Assessment
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分析任务，旨在从少量点标注中学习密集的癌变定位。通过提出新的损失函数，解决稀疏标签下的密集预测问题。**

- **链接: [http://arxiv.org/pdf/2507.06643v1](http://arxiv.org/pdf/2507.06643v1)**

> **作者:** Farahdiba Zarin; Riccardo Oliva; Vinkle Srivastav; Armine Vardazaryan; Andrea Rosati; Alice Zampolini Faustini; Giovanni Scambia; Anna Fagotti; Pietro Mascagni; Nicolas Padoy
>
> **摘要:** Learning from sparse labels is a challenge commonplace in the medical domain. This is due to numerous factors, such as annotation cost, and is especially true for newly introduced tasks. When dense pixel-level annotations are needed, this becomes even more unfeasible. However, being able to learn from just a few annotations at the pixel-level, while extremely difficult and underutilized, can drive progress in studies where perfect annotations are not immediately available. This work tackles the challenge of learning the dense prediction task of keypoint localization from a few point annotations in the context of 2d carcinosis keypoint localization from laparoscopic video frames for diagnostic planning of advanced ovarian cancer patients. To enable this, we formulate the problem as a sparse heatmap regression from a few point annotations per image and propose a new loss function, called Crag and Tail loss, for efficient learning. Our proposed loss function effectively leverages positive sparse labels while minimizing the impact of false negatives or missed annotations. Through an extensive ablation study, we demonstrate the effectiveness of our approach in achieving accurate dense localization of carcinosis keypoints, highlighting its potential to advance research in scenarios where dense annotations are challenging to obtain.
>
---
#### [new 022] Dual-Granularity Cross-Modal Identity Association for Weakly-Supervised Text-to-Person Image Matching
- **分类: cs.CV; cs.LG; cs.MM**

- **简介: 该论文属于弱监督文本到人像图像匹配任务，旨在解决复杂的一对多身份关系问题。提出双粒度关联机制与信息不对称样本构造方法，提升匹配准确率。**

- **链接: [http://arxiv.org/pdf/2507.06744v1](http://arxiv.org/pdf/2507.06744v1)**

> **作者:** Yafei Zhang; Yongle Shang; Huafeng Li
>
> **摘要:** Weakly supervised text-to-person image matching, as a crucial approach to reducing models' reliance on large-scale manually labeled samples, holds significant research value. However, existing methods struggle to predict complex one-to-many identity relationships, severely limiting performance improvements. To address this challenge, we propose a local-and-global dual-granularity identity association mechanism. Specifically, at the local level, we explicitly establish cross-modal identity relationships within a batch, reinforcing identity constraints across different modalities and enabling the model to better capture subtle differences and correlations. At the global level, we construct a dynamic cross-modal identity association network with the visual modality as the anchor and introduce a confidence-based dynamic adjustment mechanism, effectively enhancing the model's ability to identify weakly associated samples while improving overall sensitivity. Additionally, we propose an information-asymmetric sample pair construction method combined with consistency learning to tackle hard sample mining and enhance model robustness. Experimental results demonstrate that the proposed method substantially boosts cross-modal matching accuracy, providing an efficient and practical solution for text-to-person image matching.
>
---
#### [new 023] When Trackers Date Fish: A Benchmark and Framework for Underwater Multiple Fish Tracking
- **分类: cs.CV**

- **简介: 该论文属于水下多鱼跟踪任务，旨在解决水下环境中的目标跟踪难题。工作包括构建MFT25数据集和提出SU-T跟踪框架。**

- **链接: [http://arxiv.org/pdf/2507.06400v1](http://arxiv.org/pdf/2507.06400v1)**

> **作者:** Weiran Li; Yeqiang Liu; Qiannan Guo; Yijie Wei; Hwa Liang Leo; Zhenbo Li
>
> **摘要:** Multiple object tracking (MOT) technology has made significant progress in terrestrial applications, but underwater tracking scenarios remain underexplored despite their importance to marine ecology and aquaculture. We present Multiple Fish Tracking Dataset 2025 (MFT25), the first comprehensive dataset specifically designed for underwater multiple fish tracking, featuring 15 diverse video sequences with 408,578 meticulously annotated bounding boxes across 48,066 frames. Our dataset captures various underwater environments, fish species, and challenging conditions including occlusions, similar appearances, and erratic motion patterns. Additionally, we introduce Scale-aware and Unscented Tracker (SU-T), a specialized tracking framework featuring an Unscented Kalman Filter (UKF) optimized for non-linear fish swimming patterns and a novel Fish-Intersection-over-Union (FishIoU) matching that accounts for the unique morphological characteristics of aquatic species. Extensive experiments demonstrate that our SU-T baseline achieves state-of-the-art performance on MFT25, with 34.1 HOTA and 44.6 IDF1, while revealing fundamental differences between fish tracking and terrestrial object tracking scenarios. MFT25 establishes a robust foundation for advancing research in underwater tracking systems with important applications in marine biology, aquaculture monitoring, and ecological conservation. The dataset and codes are released at https://vranlee.github.io/SU-T/.
>
---
#### [new 024] SImpHAR: Advancing impedance-based human activity recognition using 3D simulation and text-to-motion models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于人体活动识别任务，解决生物阻抗数据不足的问题。通过3D模拟和文本生成技术生成数据，并采用分阶段训练策略提升识别效果。**

- **链接: [http://arxiv.org/pdf/2507.06405v1](http://arxiv.org/pdf/2507.06405v1)**

> **作者:** Lala Shakti Swarup Ray; Mengxi Liu; Deepika Gurung; Bo Zhou; Sungho Suh; Paul Lukowicz
>
> **摘要:** Human Activity Recognition (HAR) with wearable sensors is essential for applications in healthcare, fitness, and human-computer interaction. Bio-impedance sensing offers unique advantages for fine-grained motion capture but remains underutilized due to the scarcity of labeled data. We introduce SImpHAR, a novel framework addressing this limitation through two core contributions. First, we propose a simulation pipeline that generates realistic bio-impedance signals from 3D human meshes using shortest-path estimation, soft-body physics, and text-to-motion generation serving as a digital twin for data augmentation. Second, we design a two-stage training strategy with decoupled approach that enables broader activity coverage without requiring label-aligned synthetic data. We evaluate SImpHAR on our collected ImpAct dataset and two public benchmarks, showing consistent improvements over state-of-the-art methods, with gains of up to 22.3% and 21.8%, in terms of accuracy and macro F1 score, respectively. Our results highlight the promise of simulation-driven augmentation and modular training for impedance-based HAR.
>
---
#### [new 025] Enhancing Diffusion Model Stability for Image Restoration via Gradient Management
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像修复任务，解决扩散模型生成不稳定问题。通过分析梯度动态，提出SPGD方法增强稳定性与修复效果。**

- **链接: [http://arxiv.org/pdf/2507.06656v1](http://arxiv.org/pdf/2507.06656v1)**

> **作者:** Hongjie Wu; Mingqin Zhang; Linchao He; Ji-Zhe Zhou; Jiancheng Lv
>
> **备注:** Accepted to ACM Multimedia 2025. Preprint version
>
> **摘要:** Diffusion models have shown remarkable promise for image restoration by leveraging powerful priors. Prominent methods typically frame the restoration problem within a Bayesian inference framework, which iteratively combines a denoising step with a likelihood guidance step. However, the interactions between these two components in the generation process remain underexplored. In this paper, we analyze the underlying gradient dynamics of these components and identify significant instabilities. Specifically, we demonstrate conflicts between the prior and likelihood gradient directions, alongside temporal fluctuations in the likelihood gradient itself. We show that these instabilities disrupt the generative process and compromise restoration performance. To address these issues, we propose Stabilized Progressive Gradient Diffusion (SPGD), a novel gradient management technique. SPGD integrates two synergistic components: (1) a progressive likelihood warm-up strategy to mitigate gradient conflicts; and (2) adaptive directional momentum (ADM) smoothing to reduce fluctuations in the likelihood gradient. Extensive experiments across diverse restoration tasks demonstrate that SPGD significantly enhances generation stability, leading to state-of-the-art performance in quantitative metrics and visually superior results. Code is available at \href{https://github.com/74587887/SPGD}{here}.
>
---
#### [new 026] Token Bottleneck: One Token to Remember Dynamics
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，解决动态场景表示学习问题。提出ToBo方法，通过压缩场景为瓶颈令牌并预测后续场景，增强模型对时间动态的感知能力。**

- **链接: [http://arxiv.org/pdf/2507.06543v1](http://arxiv.org/pdf/2507.06543v1)**

> **作者:** Taekyung Kim; Dongyoon Han; Byeongho Heo; Jeongeun Park; Sangdoo Yun
>
> **备注:** 17 pages, 9 figures, 8 tables, project page: https://token-bottleneck.github.io, code: https://github.com/naver-ai/tobo
>
> **摘要:** Deriving compact and temporally aware visual representations from dynamic scenes is essential for successful execution of sequential scene understanding tasks such as visual tracking and robotic manipulation. In this paper, we introduce Token Bottleneck (ToBo), a simple yet intuitive self-supervised learning pipeline that squeezes a scene into a bottleneck token and predicts the subsequent scene using minimal patches as hints. The ToBo pipeline facilitates the learning of sequential scene representations by conservatively encoding the reference scene into a compact bottleneck token during the squeeze step. In the expansion step, we guide the model to capture temporal dynamics by predicting the target scene using the bottleneck token along with few target patches as hints. This design encourages the vision backbone to embed temporal dependencies, thereby enabling understanding of dynamic transitions across scenes. Extensive experiments in diverse sequential tasks, including video label propagation and robot manipulation in simulated environments demonstrate the superiority of ToBo over baselines. Moreover, deploying our pre-trained model on physical robots confirms its robustness and effectiveness in real-world environments. We further validate the scalability of ToBo across different model scales.
>
---
#### [new 027] Design and Implementation of an OCR-Powered Pipeline for Table Extraction from Invoices
- **分类: cs.CV; cs.AI; I.2.10; I.4.9; H.3.1**

- **简介: 该论文属于表格提取任务，解决从发票中准确提取结构化数据的问题。工作包括设计OCR驱动的处理流程，实现表格边界检测与行列映射。**

- **链接: [http://arxiv.org/pdf/2507.07029v1](http://arxiv.org/pdf/2507.07029v1)**

> **作者:** Parshva Dhilankumar Patel
>
> **备注:** 17 pages, 23 figures, submitted to arXiv in July 2025
>
> **摘要:** This paper presents the design and development of an OCR-powered pipeline for efficient table extraction from invoices. The system leverages Tesseract OCR for text recognition and custom post-processing logic to detect, align, and extract structured tabular data from scanned invoice documents. Our approach includes dynamic preprocessing, table boundary detection, and row-column mapping, optimized for noisy and non-standard invoice formats. The resulting pipeline significantly improves data extraction accuracy and consistency, supporting real-world use cases such as automated financial workflows and digital archiving.
>
---
#### [new 028] Longitudinal Study of Facial Biometrics at the BEZ: Temporal Variance Analysis
- **分类: cs.CV**

- **简介: 该论文属于生物特征识别任务，研究面部生物特征随时间的变化，旨在分析长期测试中的波动情况，为未来数据研究提供基础。**

- **链接: [http://arxiv.org/pdf/2507.06858v1](http://arxiv.org/pdf/2507.06858v1)**

> **作者:** Mathias Schulz; Alexander Spenke; Pia Funk; Florian Blümel; Markus Rohde; Ralph Breithaupt; Gerd Nolden; Norbert Jung; Robert Lange
>
> **备注:** 11 pages, 10 figures, 8 tables
>
> **摘要:** This study presents findings from long-term biometric evaluations conducted at the Biometric Evaluation Center (bez). Over the course of two and a half years, our ongoing research with over 400 participants representing diverse ethnicities, genders, and age groups were regularly assessed using a variety of biometric tools and techniques at the controlled testing facilities. Our findings are based on the General Data Protection Regulation-compliant local bez database with more than 238.000 biometric data sets categorized into multiple biometric modalities such as face and finger. We used state-of-the-art face recognition algorithms to analyze long-term comparison scores. Our results show that these scores fluctuate more significantly between individual days than over the entire measurement period. These findings highlight the importance of testing biometric characteristics of the same individuals over a longer period of time in a controlled measurement environment and lays the groundwork for future advancements in biometric data analysis.
>
---
#### [new 029] Unlocking Thermal Aerial Imaging: Synthetic Enhancement of UAV Datasets
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，解决热成像数据不足的问题。通过生成合成热成像数据，增强现有数据集，提升目标检测性能。**

- **链接: [http://arxiv.org/pdf/2507.06797v1](http://arxiv.org/pdf/2507.06797v1)**

> **作者:** Antonella Barisic Kulas; Andreja Jurasovic; Stjepan Bogdan
>
> **备注:** Preprint. Accepted at ECMR 2025
>
> **摘要:** Thermal imaging from unmanned aerial vehicles (UAVs) holds significant potential for applications in search and rescue, wildlife monitoring, and emergency response, especially under low-light or obscured conditions. However, the scarcity of large-scale, diverse thermal aerial datasets limits the advancement of deep learning models in this domain, primarily due to the high cost and logistical challenges of collecting thermal data. In this work, we introduce a novel procedural pipeline for generating synthetic thermal images from an aerial perspective. Our method integrates arbitrary object classes into existing thermal backgrounds by providing control over the position, scale, and orientation of the new objects, while aligning them with the viewpoints of the background. We enhance existing thermal datasets by introducing new object categories, specifically adding a drone class in urban environments to the HIT-UAV dataset and an animal category to the MONET dataset. In evaluating these datasets for object detection task, we showcase strong performance across both new and existing classes, validating the successful expansion into new applications. Through comparative analysis, we show that thermal detectors outperform their visible-light-trained counterparts and highlight the importance of replicating aerial viewing angles. Project page: https://github.com/larics/thermal_aerial_synthetic.
>
---
#### [new 030] DIFFUMA: High-Fidelity Spatio-Temporal Video Prediction via Dual-Path Mamba and Diffusion Enhancement
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频预测任务，解决工业场景中缺乏专用数据集的问题。构建了CHDL数据集，并提出DIFFUMA模型，提升预测精度。**

- **链接: [http://arxiv.org/pdf/2507.06738v1](http://arxiv.org/pdf/2507.06738v1)**

> **作者:** Xinyu Xie; Weifeng Cao; Jun Shi; Yangyang Hu; Hui Liang; Wanyong Liang; Xiaoliang Qian
>
> **摘要:** Spatio-temporal video prediction plays a pivotal role in critical domains, ranging from weather forecasting to industrial automation. However, in high-precision industrial scenarios such as semiconductor manufacturing, the absence of specialized benchmark datasets severely hampers research on modeling and predicting complex processes. To address this challenge, we make a twofold contribution.First, we construct and release the Chip Dicing Lane Dataset (CHDL), the first public temporal image dataset dedicated to the semiconductor wafer dicing process. Captured via an industrial-grade vision system, CHDL provides a much-needed and challenging benchmark for high-fidelity process modeling, defect detection, and digital twin development.Second, we propose DIFFUMA, an innovative dual-path prediction architecture specifically designed for such fine-grained dynamics. The model captures global long-range temporal context through a parallel Mamba module, while simultaneously leveraging a diffusion module, guided by temporal features, to restore and enhance fine-grained spatial details, effectively combating feature degradation. Experiments demonstrate that on our CHDL benchmark, DIFFUMA significantly outperforms existing methods, reducing the Mean Squared Error (MSE) by 39% and improving the Structural Similarity (SSIM) from 0.926 to a near-perfect 0.988. This superior performance also generalizes to natural phenomena datasets. Our work not only delivers a new state-of-the-art (SOTA) model but, more importantly, provides the community with an invaluable data resource to drive future research in industrial AI.
>
---
#### [new 031] Concept-TRAK: Understanding how diffusion models learn concepts through concept-level attribution
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像生成模型的可解释性任务，旨在解决扩散模型中特定概念贡献难以识别的问题。提出Concept-TRAK方法，实现更精确的概念级归因分析。**

- **链接: [http://arxiv.org/pdf/2507.06547v1](http://arxiv.org/pdf/2507.06547v1)**

> **作者:** Yonghyun Park; Chieh-Hsin Lai; Satoshi Hayakawa; Yuhta Takida; Naoki Murata; Wei-Hsiang Liao; Woosung Choi; Kin Wai Cheuk; Junghyun Koo; Yuki Mitsufuji
>
> **备注:** Preprint
>
> **摘要:** While diffusion models excel at image generation, their growing adoption raises critical concerns around copyright issues and model transparency. Existing attribution methods identify training examples influencing an entire image, but fall short in isolating contributions to specific elements, such as styles or objects, that matter most to stakeholders. To bridge this gap, we introduce \emph{concept-level attribution} via a novel method called \emph{Concept-TRAK}. Concept-TRAK extends influence functions with two key innovations: (1) a reformulated diffusion training loss based on diffusion posterior sampling, enabling robust, sample-specific attribution; and (2) a concept-aware reward function that emphasizes semantic relevance. We evaluate Concept-TRAK on the AbC benchmark, showing substantial improvements over prior methods. Through diverse case studies--ranging from identifying IP-protected and unsafe content to analyzing prompt engineering and compositional learning--we demonstrate how concept-level attribution yields actionable insights for responsible generative AI development and governance.
>
---
#### [new 032] 4KAgent: Agentic Any Image to 4K Super-Resolution
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于图像超分辨率任务，旨在将任意低分辨率图像提升至4K分辨率。通过构建agentic系统，结合感知与修复代理，实现高质量图像重建。**

- **链接: [http://arxiv.org/pdf/2507.07105v1](http://arxiv.org/pdf/2507.07105v1)**

> **作者:** Yushen Zuo; Qi Zheng; Mingyang Wu; Xinrui Jiang; Renjie Li; Jian Wang; Yide Zhang; Gengchen Mai; Lihong V. Wang; James Zou; Xiaoyu Wang; Ming-Hsuan Yang; Zhengzhong Tu
>
> **备注:** Project page: https://4kagent.github.io
>
> **摘要:** We present 4KAgent, a unified agentic super-resolution generalist system designed to universally upscale any image to 4K resolution (and even higher, if applied iteratively). Our system can transform images from extremely low resolutions with severe degradations, for example, highly distorted inputs at 256x256, into crystal-clear, photorealistic 4K outputs. 4KAgent comprises three core components: (1) Profiling, a module that customizes the 4KAgent pipeline based on bespoke use cases; (2) A Perception Agent, which leverages vision-language models alongside image quality assessment experts to analyze the input image and make a tailored restoration plan; and (3) A Restoration Agent, which executes the plan, following a recursive execution-reflection paradigm, guided by a quality-driven mixture-of-expert policy to select the optimal output for each step. Additionally, 4KAgent embeds a specialized face restoration pipeline, significantly enhancing facial details in portrait and selfie photos. We rigorously evaluate our 4KAgent across 11 distinct task categories encompassing a total of 26 diverse benchmarks, setting new state-of-the-art on a broad spectrum of imaging domains. Our evaluations cover natural images, portrait photos, AI-generated content, satellite imagery, fluorescence microscopy, and medical imaging like fundoscopy, ultrasound, and X-ray, demonstrating superior performance in terms of both perceptual (e.g., NIQE, MUSIQ) and fidelity (e.g., PSNR) metrics. By establishing a novel agentic paradigm for low-level vision tasks, we aim to catalyze broader interest and innovation within vision-centric autonomous agents across diverse research communities. We will release all the code, models, and results at: https://4kagent.github.io.
>
---
#### [new 033] FIFA: Unified Faithfulness Evaluation Framework for Text-to-Video and Video-to-Text Generation
- **分类: cs.CV; cs.CL; cs.GR**

- **简介: 该论文属于视频多模态生成任务，旨在解决生成内容与视觉输入不符的幻觉问题。提出FIFA评估框架和Post-Correction修正方法，提升生成内容的真实性。**

- **链接: [http://arxiv.org/pdf/2507.06523v1](http://arxiv.org/pdf/2507.06523v1)**

> **作者:** Liqiang Jing; Viet Lai; Seunghyun Yoon; Trung Bui; Xinya Du
>
> **摘要:** Video Multimodal Large Language Models (VideoMLLMs) have achieved remarkable progress in both Video-to-Text and Text-to-Video tasks. However, they often suffer fro hallucinations, generating content that contradicts the visual input. Existing evaluation methods are limited to one task (e.g., V2T) and also fail to assess hallucinations in open-ended, free-form responses. To address this gap, we propose FIFA, a unified FaIthFulness evAluation framework that extracts comprehensive descriptive facts, models their semantic dependencies via a Spatio-Temporal Semantic Dependency Graph, and verifies them using VideoQA models. We further introduce Post-Correction, a tool-based correction framework that revises hallucinated content. Extensive experiments demonstrate that FIFA aligns more closely with human judgment than existing evaluation methods, and that Post-Correction effectively improves factual consistency in both text and video generation.
>
---
#### [new 034] Hierarchical Multi-Stage Transformer Architecture for Context-Aware Temporal Action Localization
- **分类: cs.CV**

- **简介: 该论文属于视频动作定位任务，解决时序动作精确定位问题。提出PCL-Former架构，分阶段处理候选片段生成、分类和边界预测，提升定位精度。**

- **链接: [http://arxiv.org/pdf/2507.06411v1](http://arxiv.org/pdf/2507.06411v1)**

> **作者:** Hayat Ullah; Arslan Munir; Oliver Nina
>
> **备注:** 17 pages, 6 figures,
>
> **摘要:** Inspired by the recent success of transformers and multi-stage architectures in video recognition and object detection domains. We thoroughly explore the rich spatio-temporal properties of transformers within a multi-stage architecture paradigm for the temporal action localization (TAL) task. This exploration led to the development of a hierarchical multi-stage transformer architecture called PCL-Former, where each subtask is handled by a dedicated transformer module with a specialized loss function. Specifically, the Proposal-Former identifies candidate segments in an untrimmed video that may contain actions, the Classification-Former classifies the action categories within those segments, and the Localization-Former precisely predicts the temporal boundaries (i.e., start and end) of the action instances. To evaluate the performance of our method, we have conducted extensive experiments on three challenging benchmark datasets: THUMOS-14, ActivityNet-1.3, and HACS Segments. We also conducted detailed ablation experiments to assess the impact of each individual module of our PCL-Former. The obtained quantitative results validate the effectiveness of the proposed PCL-Former, outperforming state-of-the-art TAL approaches by 2.8%, 1.2%, and 4.8% on THUMOS14, ActivityNet-1.3, and HACS datasets, respectively.
>
---
#### [new 035] Capturing Stable HDR Videos Using a Dual-Camera System
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于HDR视频重建任务，旨在解决参考图像曝光波动导致的闪烁问题。通过双相机系统和自适应融合网络提升视频质量。**

- **链接: [http://arxiv.org/pdf/2507.06593v1](http://arxiv.org/pdf/2507.06593v1)**

> **作者:** Qianyu Zhang; Bolun Zheng; Hangjia Pan; Lingyu Zhu; Zunjie Zhu; Zongpeng Li; Shiqi Wang
>
> **摘要:** In HDR video reconstruction, exposure fluctuations in reference images from alternating exposure methods often result in flickering. To address this issue, we propose a dual-camera system (DCS) for HDR video acquisition, where one camera is assigned to capture consistent reference sequences, while the other is assigned to capture non-reference sequences for information supplementation. To tackle the challenges posed by video data, we introduce an exposure-adaptive fusion network (EAFNet) to achieve more robust results. EAFNet introduced a pre-alignment subnetwork to explore the influence of exposure, selectively emphasizing the valuable features across different exposure levels. Then, the enhanced features are fused by the asymmetric cross-feature fusion subnetwork, which explores reference-dominated attention maps to improve image fusion by aligning cross-scale features and performing cross-feature fusion. Finally, the reconstruction subnetwork adopts a DWT-based multiscale architecture to reduce ghosting artifacts and refine features at different resolutions. Extensive experimental evaluations demonstrate that the proposed method achieves state-of-the-art performance on different datasets, validating the great potential of the DCS in HDR video reconstruction. The codes and data captured by DCS will be available at https://github.com/zqqqyu/DCS.
>
---
#### [new 036] GNN-ViTCap: GNN-Enhanced Multiple Instance Learning with Vision Transformers for Whole Slide Image Classification and Captioning
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于病理图像分类与描述任务，解决WSI中冗余和位置未知的问题，通过GNN-ViTCap框架提升诊断准确性与自动描述能力。**

- **链接: [http://arxiv.org/pdf/2507.07006v1](http://arxiv.org/pdf/2507.07006v1)**

> **作者:** S M Taslim Uddin Raju; Md. Milon Islam; Md Rezwanul Haque; Hamdi Altaheri; Fakhri Karray
>
> **摘要:** Microscopic assessment of histopathology images is vital for accurate cancer diagnosis and treatment. Whole Slide Image (WSI) classification and captioning have become crucial tasks in computer-aided pathology. However, microscopic WSI face challenges such as redundant patches and unknown patch positions due to subjective pathologist captures. Moreover, generating automatic pathology captions remains a significant challenge. To address these issues, we introduce a novel GNN-ViTCap framework for classification and caption generation from histopathological microscopic images. First, a visual feature extractor generates patch embeddings. Redundant patches are then removed by dynamically clustering these embeddings using deep embedded clustering and selecting representative patches via a scalar dot attention mechanism. We build a graph by connecting each node to its nearest neighbors in the similarity matrix and apply a graph neural network to capture both local and global context. The aggregated image embeddings are projected into the language model's input space through a linear layer and combined with caption tokens to fine-tune a large language model. We validate our method on the BreakHis and PatchGastric datasets. GNN-ViTCap achieves an F1 score of 0.934 and an AUC of 0.963 for classification, along with a BLEU-4 score of 0.811 and a METEOR score of 0.569 for captioning. Experimental results demonstrate that GNN-ViTCap outperforms state of the art approaches, offering a reliable and efficient solution for microscopy based patient diagnosis.
>
---
#### [new 037] MADPOT: Medical Anomaly Detection with CLIP Adaptation and Partial Optimal Transport
- **分类: cs.CV**

- **简介: 该论文属于医学异常检测任务，旨在解决医疗图像中异常识别难题。通过结合CLIP适配、部分最优传输和对比学习，提升模型在少量或无标注数据下的性能。**

- **链接: [http://arxiv.org/pdf/2507.06733v1](http://arxiv.org/pdf/2507.06733v1)**

> **作者:** Mahshid Shiri; Cigdem Beyan; Vittorio Murino
>
> **备注:** Accepted to ICIAP 2025 (this version is not peer-reviewed; it is the submitted version). ICIAP 2025 proceedings DOI will appear here
>
> **摘要:** Medical anomaly detection (AD) is challenging due to diverse imaging modalities, anatomical variations, and limited labeled data. We propose a novel approach combining visual adapters and prompt learning with Partial Optimal Transport (POT) and contrastive learning (CL) to improve CLIP's adaptability to medical images, particularly for AD. Unlike standard prompt learning, which often yields a single representation, our method employs multiple prompts aligned with local features via POT to capture subtle abnormalities. CL further enforces intra-class cohesion and inter-class separation. Our method achieves state-of-the-art results in few-shot, zero-shot, and cross-dataset scenarios without synthetic data or memory banks. The code is available at https://github.com/mahshid1998/MADPOT.
>
---
#### [new 038] Ambiguity-aware Point Cloud Segmentation by Adaptive Margin Contrastive Learning
- **分类: cs.CV**

- **简介: 该论文属于3D点云语义分割任务，解决因点云模糊性导致的标注不可靠问题。通过自适应对比学习方法，提升模型对模糊点的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.06592v1](http://arxiv.org/pdf/2507.06592v1)**

> **作者:** Yang Chen; Yueqi Duan; Haowen Sun; Jiwen Lu; Yap-Peng Tan
>
> **备注:** This article has been accepted for publication in IEEE Transactions on Multimedia. arXiv admin note: text overlap with arXiv:2502.04111
>
> **摘要:** This paper proposes an adaptive margin contrastive learning method for 3D semantic segmentation on point clouds. Most existing methods use equally penalized objectives, which ignore the per-point ambiguities and less discriminated features stemming from transition regions. However, as highly ambiguous points may be indistinguishable even for humans, their manually annotated labels are less reliable, and hard constraints over these points would lead to sub-optimal models. To address this, we first design AMContrast3D, a method comprising contrastive learning into an ambiguity estimation framework, tailored to adaptive objectives for individual points based on ambiguity levels. As a result, our method promotes model training, which ensures the correctness of low-ambiguity points while allowing mistakes for high-ambiguity points. As ambiguities are formulated based on position discrepancies across labels, optimization during inference is constrained by the assumption that all unlabeled points are uniformly unambiguous, lacking ambiguity awareness. Inspired by the insight of joint training, we further propose AMContrast3D++ integrating with two branches trained in parallel, where a novel ambiguity prediction module concurrently learns point ambiguities from generated embeddings. To this end, we design a masked refinement mechanism that leverages predicted ambiguities to enable the ambiguous embeddings to be more reliable, thereby boosting segmentation performance and enhancing robustness. Experimental results on 3D indoor scene datasets, S3DIS and ScanNet, demonstrate the effectiveness of the proposed method. Code is available at https://github.com/YangChenApril/AMContrast3D.
>
---
#### [new 039] Towards Multimodal Understanding via Stable Diffusion as a Task-Aware Feature Extractor
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多模态理解任务，旨在解决CLIP模型在细节捕捉上的不足。通过使用扩散模型作为视觉编码器，提升图像-文本对齐与细粒度特征提取能力。**

- **链接: [http://arxiv.org/pdf/2507.07106v1](http://arxiv.org/pdf/2507.07106v1)**

> **作者:** Vatsal Agarwal; Matthew Gwilliam; Gefen Kohavi; Eshan Verma; Daniel Ulbricht; Abhinav Shrivastava
>
> **备注:** Website: see https://vatsalag99.github.io/mustafar/
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have enabled image-based question-answering capabilities. However, a key limitation is the use of CLIP as the visual encoder; while it can capture coarse global information, it often can miss fine-grained details that are relevant to the input query. To address these shortcomings, this work studies whether pre-trained text-to-image diffusion models can serve as instruction-aware visual encoders. Through an analysis of their internal representations, we find diffusion features are both rich in semantics and can encode strong image-text alignment. Moreover, we find that we can leverage text conditioning to focus the model on regions relevant to the input question. We then investigate how to align these features with large language models and uncover a leakage phenomenon, where the LLM can inadvertently recover information from the original diffusion prompt. We analyze the causes of this leakage and propose a mitigation strategy. Based on these insights, we explore a simple fusion strategy that utilizes both CLIP and conditional diffusion features. We evaluate our approach on both general VQA and specialized MLLM benchmarks, demonstrating the promise of diffusion models for visual understanding, particularly in vision-centric tasks that require spatial and compositional reasoning. Our project page can be found https://vatsalag99.github.io/mustafar/.
>
---
#### [new 040] ClipGS: Clippable Gaussian Splatting for Interactive Cinematic Visualization of Volumetric Medical Data
- **分类: cs.CV**

- **简介: 该论文属于医学影像可视化任务，解决动态交互中渲染效率与质量的问题。提出ClipGS框架，支持剪裁平面，提升实时渲染性能。**

- **链接: [http://arxiv.org/pdf/2507.06647v1](http://arxiv.org/pdf/2507.06647v1)**

> **作者:** Chengkun Li; Yuqi Tong; Kai Chen; Zhenya Yang; Ruiyang Li; Shi Qiu; Jason Ying-Kuen Chan; Pheng-Ann Heng; Qi Dou
>
> **备注:** Early accepted by MICCAI 2025. Project is available at: https://med-air.github.io/ClipGS
>
> **摘要:** The visualization of volumetric medical data is crucial for enhancing diagnostic accuracy and improving surgical planning and education. Cinematic rendering techniques significantly enrich this process by providing high-quality visualizations that convey intricate anatomical details, thereby facilitating better understanding and decision-making in medical contexts. However, the high computing cost and low rendering speed limit the requirement of interactive visualization in practical applications. In this paper, we introduce ClipGS, an innovative Gaussian splatting framework with the clipping plane supported, for interactive cinematic visualization of volumetric medical data. To address the challenges posed by dynamic interactions, we propose a learnable truncation scheme that automatically adjusts the visibility of Gaussian primitives in response to the clipping plane. Besides, we also design an adaptive adjustment model to dynamically adjust the deformation of Gaussians and refine the rendering performance. We validate our method on five volumetric medical data (including CT and anatomical slice data), and reach an average 36.635 PSNR rendering quality with 156 FPS and 16.1 MB model size, outperforming state-of-the-art methods in rendering quality and efficiency.
>
---
#### [new 041] PromptTea: Let Prompts Tell TeaCache the Optimal Threshold
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决缓存机制在复杂场景下质量下降的问题。提出PCA缓存和DynCFGCache，动态调整重用阈值以提升速度与质量。**

- **链接: [http://arxiv.org/pdf/2507.06739v1](http://arxiv.org/pdf/2507.06739v1)**

> **作者:** Zishen Huang; Chunyu Yang; Mengyuan Ren
>
> **摘要:** Despite recent progress in video generation, inference speed remains a major bottleneck. A common acceleration strategy involves reusing model outputs via caching mechanisms at fixed intervals. However, we find that such fixed-frequency reuse significantly degrades quality in complex scenes, while manually tuning reuse thresholds is inefficient and lacks robustness. To address this, we propose Prompt-Complexity-Aware (PCA) caching, a method that automatically adjusts reuse thresholds based on scene complexity estimated directly from the input prompt. By incorporating prompt-derived semantic cues, PCA enables more adaptive and informed reuse decisions than conventional caching methods. We also revisit the assumptions behind TeaCache and identify a key limitation: it suffers from poor input-output relationship modeling due to an oversimplified prior. To overcome this, we decouple the noisy input, enhance the contribution of meaningful textual information, and improve the model's predictive accuracy through multivariate polynomial feature expansion. To further reduce computational cost, we replace the static CFGCache with DynCFGCache, a dynamic mechanism that selectively reuses classifier-free guidance (CFG) outputs based on estimated output variations. This allows for more flexible reuse without compromising output quality. Extensive experiments demonstrate that our approach achieves significant acceleration-for example, 2.79x speedup on the Wan2.1 model-while maintaining high visual fidelity across a range of scenes.
>
---
#### [new 042] An AI Approach for Learning the Spectrum of the Laplace-Beltrami Operator
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于几何深度学习任务，旨在高效预测Laplace-Beltrami算子的谱。针对传统FEM方法计算效率低的问题，提出一种图神经网络框架，显著提升计算速度并保持精度。**

- **链接: [http://arxiv.org/pdf/2507.07073v1](http://arxiv.org/pdf/2507.07073v1)**

> **作者:** Yulin An; Enrique del Castillo
>
> **备注:** 18 pages, 9 figures, submitted for publication
>
> **摘要:** The spectrum of the Laplace-Beltrami (LB) operator is central in geometric deep learning tasks, capturing intrinsic properties of the shape of the object under consideration. The best established method for its estimation, from a triangulated mesh of the object, is based on the Finite Element Method (FEM), and computes the top k LB eigenvalues with a complexity of O(Nk), where N is the number of points. This can render the FEM method inefficient when repeatedly applied to databases of CAD mechanical parts, or in quality control applications where part metrology is acquired as large meshes and decisions about the quality of each part are needed quickly and frequently. As a solution to this problem, we present a geometric deep learning framework to predict the LB spectrum efficiently given the CAD mesh of a part, achieving significant computational savings without sacrificing accuracy, demonstrating that the LB spectrum is learnable. The proposed Graph Neural Network architecture uses a rich set of part mesh features - including Gaussian curvature, mean curvature, and principal curvatures. In addition to our trained network, we make available, for repeatability, a large curated dataset of real-world mechanical CAD models derived from the publicly available ABC dataset used for training and testing. Experimental results show that our method reduces computation time of the LB spectrum by approximately 5 times over linear FEM while delivering competitive accuracy.
>
---
#### [new 043] Residual Prior-driven Frequency-aware Network for Image Fusion
- **分类: cs.CV; cs.LG; cs.MM**

- **简介: 该论文属于图像融合任务，旨在解决多模态信息整合中的计算成本高和互补特征提取难的问题。提出RPFNet网络，结合残差先验和频域融合模块，提升融合效果。**

- **链接: [http://arxiv.org/pdf/2507.06735v1](http://arxiv.org/pdf/2507.06735v1)**

> **作者:** Guan Zheng; Xue Wang; Wenhua Qian; Peng Liu; Runzhuo Ma
>
> **摘要:** Image fusion aims to integrate complementary information across modalities to generate high-quality fused images, thereby enhancing the performance of high-level vision tasks. While global spatial modeling mechanisms show promising results, constructing long-range feature dependencies in the spatial domain incurs substantial computational costs. Additionally, the absence of ground-truth exacerbates the difficulty of capturing complementary features effectively. To tackle these challenges, we propose a Residual Prior-driven Frequency-aware Network, termed as RPFNet. Specifically, RPFNet employs a dual-branch feature extraction framework: the Residual Prior Module (RPM) extracts modality-specific difference information from residual maps, thereby providing complementary priors for fusion; the Frequency Domain Fusion Module (FDFM) achieves efficient global feature modeling and integration through frequency-domain convolution. Additionally, the Cross Promotion Module (CPM) enhances the synergistic perception of local details and global structures through bidirectional feature interaction. During training, we incorporate an auxiliary decoder and saliency structure loss to strengthen the model's sensitivity to modality-specific differences. Furthermore, a combination of adaptive weight-based frequency contrastive loss and SSIM loss effectively constrains the solution space, facilitating the joint capture of local details and global features while ensuring the retention of complementary information. Extensive experiments validate the fusion performance of RPFNet, which effectively integrates discriminative features, enhances texture details and salient objects, and can effectively facilitate the deployment of the high-level vision task.
>
---
#### [new 044] A model-agnostic active learning approach for animal detection from camera traps
- **分类: cs.CV**

- **简介: 该论文属于动物检测任务，解决野生动物数据标注成本高的问题。提出一种模型无关的主动学习方法，通过结合样本不确定性和多样性提升检测效果。**

- **链接: [http://arxiv.org/pdf/2507.06537v1](http://arxiv.org/pdf/2507.06537v1)**

> **作者:** Thi Thu Thuy Nguyen; Duc Thanh Nguyen
>
> **摘要:** Smart data selection is becoming increasingly important in data-driven machine learning. Active learning offers a promising solution by allowing machine learning models to be effectively trained with optimal data including the most informative samples from large datasets. Wildlife data captured by camera traps are excessive in volume, requiring tremendous effort in data labelling and animal detection models training. Therefore, applying active learning to optimise the amount of labelled data would be a great aid in enabling automated wildlife monitoring and conservation. However, existing active learning techniques require that a machine learning model (i.e., an object detector) be fully accessible, limiting the applicability of the techniques. In this paper, we propose a model-agnostic active learning approach for detection of animals captured by camera traps. Our approach integrates uncertainty and diversity quantities of samples at both the object-based and image-based levels into the active learning sample selection process. We validate our approach in a benchmark animal dataset. Experimental results demonstrate that, using only 30% of the training data selected by our approach, a state-of-the-art animal detector can achieve a performance of equal or greater than that with the use of the complete training dataset.
>
---
#### [new 045] AR2: Attention-Guided Repair for the Robustness of CNNs Against Common Corruptions
- **分类: cs.CV; cs.LG; cs.SE**

- **简介: 该论文属于图像分类任务，解决CNN在常见噪声等干扰下的鲁棒性问题。提出AR2方法，通过注意力引导修复提升模型可靠性。**

- **链接: [http://arxiv.org/pdf/2507.06332v1](http://arxiv.org/pdf/2507.06332v1)**

> **作者:** Fuyuan Zhang; Qichen Wang; Jianjun Zhao
>
> **摘要:** Deep neural networks suffer from significant performance degradation when exposed to common corruptions such as noise, blur, weather, and digital distortions, limiting their reliability in real-world applications. In this paper, we propose AR2 (Attention-Guided Repair for Robustness), a simple yet effective method to enhance the corruption robustness of pretrained CNNs. AR2 operates by explicitly aligning the class activation maps (CAMs) between clean and corrupted images, encouraging the model to maintain consistent attention even under input perturbations. Our approach follows an iterative repair strategy that alternates between CAM-guided refinement and standard fine-tuning, without requiring architectural changes. Extensive experiments show that AR2 consistently outperforms existing state-of-the-art methods in restoring robustness on standard corruption benchmarks (CIFAR-10-C, CIFAR-100-C and ImageNet-C), achieving a favorable balance between accuracy on clean data and corruption robustness. These results demonstrate that AR2 provides a robust and scalable solution for enhancing model reliability in real-world environments with diverse corruptions.
>
---
#### [new 046] Text-promptable Object Counting via Quantity Awareness Enhancement
- **分类: cs.CV**

- **简介: 该论文属于文本提示的物体计数任务，旨在提升模型对物体数量的感知能力。通过引入量级导向提示和双流解码器，增强模型在零样本场景下的计数性能。**

- **链接: [http://arxiv.org/pdf/2507.06679v1](http://arxiv.org/pdf/2507.06679v1)**

> **作者:** Miaojing Shi; Xiaowen Zhang; Zijie Yue; Yong Luo; Cairong Zhao; Li Li
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Recent advances in large vision-language models (VLMs) have shown remarkable progress in solving the text-promptable object counting problem. Representative methods typically specify text prompts with object category information in images. This however is insufficient for training the model to accurately distinguish the number of objects in the counting task. To this end, we propose QUANet, which introduces novel quantity-oriented text prompts with a vision-text quantity alignment loss to enhance the model's quantity awareness. Moreover, we propose a dual-stream adaptive counting decoder consisting of a Transformer stream, a CNN stream, and a number of Transformer-to-CNN enhancement adapters (T2C-adapters) for density map prediction. The T2C-adapters facilitate the effective knowledge communication and aggregation between the Transformer and CNN streams. A cross-stream quantity ranking loss is proposed in the end to optimize the ranking orders of predictions from the two streams. Extensive experiments on standard benchmarks such as FSC-147, CARPK, PUCPR+, and ShanghaiTech demonstrate our model's strong generalizability for zero-shot class-agnostic counting. Code is available at https://github.com/viscom-tongji/QUANet
>
---
#### [new 047] Know Your Attention Maps: Class-specific Token Masking for Weakly Supervised Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文属于弱监督语义分割任务，旨在减少对精细标注数据的依赖。通过利用ViT的注意力图和多[CLS]令牌进行分类，生成准确的伪分割掩码。**

- **链接: [http://arxiv.org/pdf/2507.06848v1](http://arxiv.org/pdf/2507.06848v1)**

> **作者:** Joelle Hanna; Damian Borth
>
> **摘要:** Weakly Supervised Semantic Segmentation (WSSS) is a challenging problem that has been extensively studied in recent years. Traditional approaches often rely on external modules like Class Activation Maps to highlight regions of interest and generate pseudo segmentation masks. In this work, we propose an end-to-end method that directly utilizes the attention maps learned by a Vision Transformer (ViT) for WSSS. We propose training a sparse ViT with multiple [CLS] tokens (one for each class), using a random masking strategy to promote [CLS] token - class assignment. At inference time, we aggregate the different self-attention maps of each [CLS] token corresponding to the predicted labels to generate pseudo segmentation masks. Our proposed approach enhances the interpretability of self-attention maps and ensures accurate class assignments. Extensive experiments on two standard benchmarks and three specialized datasets demonstrate that our method generates accurate pseudo-masks, outperforming related works. Those pseudo-masks can be used to train a segmentation model which achieves results comparable to fully-supervised models, significantly reducing the need for fine-grained labeled data.
>
---
#### [new 048] EA: An Event Autoencoder for High-Speed Vision Sensing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于事件相机的图像处理任务，旨在解决传统视觉系统在高速场景中的延迟和冗余问题。提出事件自编码器模型，高效压缩与重建事件数据，提升识别速度与精度。**

- **链接: [http://arxiv.org/pdf/2507.06459v1](http://arxiv.org/pdf/2507.06459v1)**

> **作者:** Riadul Islam; Joey Mulé; Dhandeep Challagundla; Shahmir Rizvi; Sean Carson
>
> **摘要:** High-speed vision sensing is essential for real-time perception in applications such as robotics, autonomous vehicles, and industrial automation. Traditional frame-based vision systems suffer from motion blur, high latency, and redundant data processing, limiting their performance in dynamic environments. Event cameras, which capture asynchronous brightness changes at the pixel level, offer a promising alternative but pose challenges in object detection due to sparse and noisy event streams. To address this, we propose an event autoencoder architecture that efficiently compresses and reconstructs event data while preserving critical spatial and temporal features. The proposed model employs convolutional encoding and incorporates adaptive threshold selection and a lightweight classifier to enhance recognition accuracy while reducing computational complexity. Experimental results on the existing Smart Event Face Dataset (SEFD) demonstrate that our approach achieves comparable accuracy to the YOLO-v4 model while utilizing up to $35.5\times$ fewer parameters. Implementations on embedded platforms, including Raspberry Pi 4B and NVIDIA Jetson Nano, show high frame rates ranging from 8 FPS up to 44.8 FPS. The proposed classifier exhibits up to 87.84x better FPS than the state-of-the-art and significantly improves event-based vision performance, making it ideal for low-power, high-speed applications in real-time edge computing.
>
---
#### [new 049] Speak2Sign3D: A Multi-modal Pipeline for English Speech to American Sign Language Animation
- **分类: cs.CV**

- **简介: 该论文属于语音到手语翻译任务，解决将英语语音转换为逼真3D手语动画的问题。工作包括构建多模态管道，融合语音识别、机器翻译和运动生成。**

- **链接: [http://arxiv.org/pdf/2507.06530v1](http://arxiv.org/pdf/2507.06530v1)**

> **作者:** Kazi Mahathir Rahman; Naveed Imtiaz Nafis; Md. Farhan Sadik; Mohammad Al Rafi; Mehedi Hasan Shahed
>
> **备注:** 11 pages, 12 figures
>
> **摘要:** Helping deaf and hard-of-hearing people communicate more easily is the main goal of Automatic Sign Language Translation. Although most past research has focused on turning sign language into text, doing the reverse, turning spoken English into sign language animations, has been largely overlooked. That's because it involves multiple steps, such as understanding speech, translating it into sign-friendly grammar, and generating natural human motion. In this work, we introduce a complete pipeline that converts English speech into smooth, realistic 3D sign language animations. Our system starts with Whisper to translate spoken English into text. Then, we use a MarianMT machine translation model to translate that text into American Sign Language (ASL) gloss, a simplified version of sign language that captures meaning without grammar. This model performs well, reaching BLEU scores of 0.7714 and 0.8923. To make the gloss translation more accurate, we also use word embeddings such as Word2Vec and FastText to understand word meanings. Finally, we animate the translated gloss using a 3D keypoint-based motion system trained on Sign3D-WLASL, a dataset we created by extracting body, hand, and face key points from real ASL videos in the WLASL dataset. To support the gloss translation stage, we also built a new dataset called BookGlossCorpus-CG, which turns everyday English sentences from the BookCorpus dataset into ASL gloss using grammar rules. Our system stitches everything together by smoothly interpolating between signs to create natural, continuous animations. Unlike previous works like How2Sign and Phoenix-2014T that focus on recognition or use only one type of data, our pipeline brings together audio, text, and motion in a single framework that goes all the way from spoken English to lifelike 3D sign language animation.
>
---
#### [new 050] Adaptive Part Learning for Fine-Grained Generalized Category Discovery: A Plug-and-Play Enhancement
- **分类: cs.CV**

- **简介: 该论文属于细粒度广义类别发现任务，旨在提升模型区分相似类别和迁移知识的能力。提出APL方法，通过自适应部件学习增强表示性能。**

- **链接: [http://arxiv.org/pdf/2507.06928v1](http://arxiv.org/pdf/2507.06928v1)**

> **作者:** Qiyuan Dai; Hanzhuo Huang; Yu Wu; Sibei Yang
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** Generalized Category Discovery (GCD) aims to recognize unlabeled images from known and novel classes by distinguishing novel classes from known ones, while also transferring knowledge from another set of labeled images with known classes. Existing GCD methods rely on self-supervised vision transformers such as DINO for representation learning. However, focusing solely on the global representation of the DINO CLS token introduces an inherent trade-off between discriminability and generalization. In this paper, we introduce an adaptive part discovery and learning method, called APL, which generates consistent object parts and their correspondences across different similar images using a set of shared learnable part queries and DINO part priors, without requiring any additional annotations. More importantly, we propose a novel all-min contrastive loss to learn discriminative yet generalizable part representation, which adaptively highlights discriminative object parts to distinguish similar categories for enhanced discriminability while simultaneously sharing other parts to facilitate knowledge transfer for improved generalization. Our APL can easily be incorporated into different GCD frameworks by replacing their CLS token feature with our part representations, showing significant enhancements on fine-grained datasets.
>
---
#### [new 051] Pre-Columbian Settlements Shaped Palm Clusters in the Sierra Nevada de Santa Marta, Colombia
- **分类: cs.CV**

- **简介: 该论文属于生态与考古交叉研究，旨在通过AI识别植被模式，揭示古代人类活动对环境的影响。**

- **链接: [http://arxiv.org/pdf/2507.06949v1](http://arxiv.org/pdf/2507.06949v1)**

> **作者:** Sebastian Fajardo; Sina Mohammadi; Jonas Gregorio de Souza; César Ardila; Alan Tapscott Baltar; Shaddai Heidgen; Maria Isabel Mayorga Hernández; Sylvia Mota de Oliveira; Fernando Montejo; Marco Moderato; Vinicius Peripato; Katy Puche; Carlos Reina; Juan Carlos Vargas; Frank W. Takes; Marco Madella
>
> **摘要:** Ancient populations markedly transformed Neotropical forests, yet understanding the long-term effects of ancient human management, particularly at high-resolution scales, remains challenging. In this work we propose a new approach to investigate archaeological areas of influence based on vegetation signatures. It consists of a deep learning model trained on satellite imagery to identify palm trees, followed by a clustering algorithm to identify palm clusters, which are then used to estimate ancient management areas. To assess the palm distribution in relation to past human activity, we applied the proposed approach to unique high-resolution satellite imagery data covering 765 km2 of the Sierra Nevada de Santa Marta, Colombia. With this work, we also release a manually annotated palm tree dataset along with estimated locations of archaeological sites from ground-surveys and legacy records. Results demonstrate how palms were significantly more abundant near archaeological sites showing large infrastructure investment. The extent of the largest palm cluster indicates that ancient human-managed areas linked to major infrastructure sites may be up to two orders of magnitude bigger than indicated by archaeological evidence alone. Our findings suggest that pre-Columbian populations influenced local vegetation fostering conditions conducive to palm proliferation, leaving a lasting ecological footprint. This may have lowered the logistical costs of establishing infrastructure-heavy settlements in otherwise less accessible locations. Overall, this study demonstrates the potential of integrating artificial intelligence approaches with new ecological and archaeological data to identify archaeological areas of interest through vegetation patterns, revealing fine-scale human-environment interactions.
>
---
#### [new 052] Edge-Boundary-Texture Loss: A Tri-Class Generalization of Weighted Binary Cross-Entropy for Enhanced Edge Detection
- **分类: cs.CV**

- **简介: 该论文属于边缘检测任务，针对非边缘像素模糊问题，提出EBT损失函数，将像素分为三类并赋予不同权重，提升检测精度与结构一致性。**

- **链接: [http://arxiv.org/pdf/2507.06569v1](http://arxiv.org/pdf/2507.06569v1)**

> **作者:** Hao Shu
>
> **备注:** 10 pages
>
> **摘要:** Edge detection (ED) remains a fundamental task in computer vision, yet its performance is often hindered by the ambiguous nature of non-edge pixels near object boundaries. The widely adopted Weighted Binary Cross-Entropy (WBCE) loss treats all non-edge pixels uniformly, overlooking the structural nuances around edges and often resulting in blurred predictions. In this paper, we propose the Edge-Boundary-Texture (EBT) loss, a novel objective that explicitly divides pixels into three categories, edge, boundary, and texture, and assigns each a distinct supervisory weight. This tri-class formulation enables more structured learning by guiding the model to focus on both edge precision and contextual boundary localization. We theoretically show that the EBT loss generalizes the WBCE loss, with the latter becoming a limit case. Extensive experiments across multiple benchmarks demonstrate the superiority of the EBT loss both quantitatively and perceptually. Furthermore, the consistent use of unified hyperparameters across all models and datasets, along with robustness to their moderate variations, indicates that the EBT loss requires minimal fine-tuning and is easily deployable in practice.
>
---
#### [new 053] Cross-Modality Masked Learning for Survival Prediction in ICI Treated NSCLC Patients
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于生存预测任务，旨在解决NSCLC患者免疫治疗预后准确预测的问题。通过构建多模态数据集并提出跨模态掩码学习方法，提升特征融合效果。**

- **链接: [http://arxiv.org/pdf/2507.06994v1](http://arxiv.org/pdf/2507.06994v1)**

> **作者:** Qilong Xing; Zikai Song; Bingxin Gong; Lian Yang; Junqing Yu; Wei Yang
>
> **备注:** MICCAI 2025
>
> **摘要:** Accurate prognosis of non-small cell lung cancer (NSCLC) patients undergoing immunotherapy is essential for personalized treatment planning, enabling informed patient decisions, and improving both treatment outcomes and quality of life. However, the lack of large, relevant datasets and effective multi-modal feature fusion strategies pose significant challenges in this domain. To address these challenges, we present a large-scale dataset and introduce a novel framework for multi-modal feature fusion aimed at enhancing the accuracy of survival prediction. The dataset comprises 3D CT images and corresponding clinical records from NSCLC patients treated with immune checkpoint inhibitors (ICI), along with progression-free survival (PFS) and overall survival (OS) data. We further propose a cross-modality masked learning approach for medical feature fusion, consisting of two distinct branches, each tailored to its respective modality: a Slice-Depth Transformer for extracting 3D features from CT images and a graph-based Transformer for learning node features and relationships among clinical variables in tabular data. The fusion process is guided by a masked modality learning strategy, wherein the model utilizes the intact modality to reconstruct missing components. This mechanism improves the integration of modality-specific features, fostering more effective inter-modality relationships and feature interactions. Our approach demonstrates superior performance in multi-modal integration for NSCLC survival prediction, surpassing existing methods and setting a new benchmark for prognostic models in this context.
>
---
#### [new 054] Cross-Modal Dual-Causal Learning for Long-Term Action Recognition
- **分类: cs.CV**

- **简介: 该论文属于长期动作识别任务，解决视频与文本间复杂因果关系问题。提出CMDCL模型，通过跨模态因果干预提升动作表示鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.06603v1](http://arxiv.org/pdf/2507.06603v1)**

> **作者:** Xu Shaowu; Jia Xibin; Gao Junyu; Sun Qianmei; Chang Jing; Fan Chao
>
> **摘要:** Long-term action recognition (LTAR) is challenging due to extended temporal spans with complex atomic action correlations and visual confounders. Although vision-language models (VLMs) have shown promise, they often rely on statistical correlations instead of causal mechanisms. Moreover, existing causality-based methods address modal-specific biases but lack cross-modal causal modeling, limiting their utility in VLM-based LTAR. This paper proposes \textbf{C}ross-\textbf{M}odal \textbf{D}ual-\textbf{C}ausal \textbf{L}earning (CMDCL), which introduces a structural causal model to uncover causal relationships between videos and label texts. CMDCL addresses cross-modal biases in text embeddings via textual causal intervention and removes confounders inherent in the visual modality through visual causal intervention guided by the debiased text. These dual-causal interventions enable robust action representations to address LTAR challenges. Experimental results on three benchmarks including Charades, Breakfast and COIN, demonstrate the effectiveness of the proposed model. Our code is available at https://github.com/xushaowu/CMDCL.
>
---
#### [new 055] Divergence-Based Similarity Function for Multi-View Contrastive Learning
- **分类: cs.CV; cs.LG; 68T07, 62H12; I.2.6; I.4.8; I.5.1**

- **简介: 该论文属于对比学习任务，旨在解决多视图表示学习中缺乏联合结构建模的问题。提出基于散度的相似性函数（DSF），通过分布间散度衡量相似性，提升模型性能与效率。**

- **链接: [http://arxiv.org/pdf/2507.06560v1](http://arxiv.org/pdf/2507.06560v1)**

> **作者:** Jae Hyoung Jeon; Cheolsu Lim; Myungjoo Kang
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Recent success in contrastive learning has sparked growing interest in more effectively leveraging multiple augmented views of an instance. While prior methods incorporate multiple views at the loss or feature level, they primarily capture pairwise relationships and fail to model the joint structure across all views. In this work, we propose a divergence-based similarity function (DSF) that explicitly captures the joint structure by representing each set of augmented views as a distribution and measuring similarity as the divergence between distributions. Extensive experiments demonstrate that DSF consistently improves performance across various tasks, including kNN classification and linear evaluation, while also offering greater efficiency compared to other multi-view methods. Furthermore, we establish a theoretical connection between DSF and cosine similarity, and show that, unlike cosine similarity, DSF operates effectively without requiring a temperature hyperparameter.
>
---
#### [new 056] Concept Unlearning by Modeling Key Steps of Diffusion Process
- **分类: cs.CV**

- **简介: 该论文属于文本生成图像任务，解决扩散模型生成不当内容的问题。提出KSCU方法，通过关键步骤微调提升概念移除效果并保留生成能力。**

- **链接: [http://arxiv.org/pdf/2507.06526v1](http://arxiv.org/pdf/2507.06526v1)**

> **作者:** Chaoshuo Zhang; Chenhao Lin; Zhengyu Zhao; Le Yang; Qian Wang; Chao Shen
>
> **摘要:** Text-to-image diffusion models (T2I DMs), represented by Stable Diffusion, which generate highly realistic images based on textual input, have been widely used. However, their misuse poses serious security risks. While existing concept unlearning methods aim to mitigate these risks, they struggle to balance unlearning effectiveness with generative retainability.To overcome this limitation, we innovatively propose the Key Step Concept Unlearning (KSCU) method, which ingeniously capitalizes on the unique stepwise sampling characteristic inherent in diffusion models during the image generation process. Unlike conventional approaches that treat all denoising steps equally, KSCU strategically focuses on pivotal steps with the most influence over the final outcome by dividing key steps for different concept unlearning tasks and fine-tuning the model only at those steps. This targeted approach reduces the number of parameter updates needed for effective unlearning, while maximizing the retention of the model's generative capabilities.Through extensive benchmark experiments, we demonstrate that KSCU effectively prevents T2I DMs from generating undesirable images while better retaining the model's generative capabilities.Our code will be released.
>
---
#### [new 057] MS-DPPs: Multi-Source Determinantal Point Processes for Contextual Diversity Refinement of Composite Attributes in Text to Image Retrieval
- **分类: cs.CV; cs.AI; cs.IR**

- **简介: 该论文属于文本到图像检索任务，解决传统方法在多样性评估上的不足。提出MS-DPP模型，通过多源DPP和上下文感知机制提升复合属性的多样性。**

- **链接: [http://arxiv.org/pdf/2507.06654v1](http://arxiv.org/pdf/2507.06654v1)**

> **作者:** Naoya Sogi; Takashi Shibata; Makoto Terao; Masanori Suganuma; Takayuki Okatani
>
> **备注:** IJCAI 2025. Code: https://github.com/NEC-N-SOGI/msdpp
>
> **摘要:** Result diversification (RD) is a crucial technique in Text-to-Image Retrieval for enhancing the efficiency of a practical application. Conventional methods focus solely on increasing the diversity metric of image appearances. However, the diversity metric and its desired value vary depending on the application, which limits the applications of RD. This paper proposes a novel task called CDR-CA (Contextual Diversity Refinement of Composite Attributes). CDR-CA aims to refine the diversities of multiple attributes, according to the application's context. To address this task, we propose Multi-Source DPPs, a simple yet strong baseline that extends the Determinantal Point Process (DPP) to multi-sources. We model MS-DPP as a single DPP model with a unified similarity matrix based on a manifold representation. We also introduce Tangent Normalization to reflect contexts. Extensive experiments demonstrate the effectiveness of the proposed method. Our code is publicly available at https://github.com/NEC-N-SOGI/msdpp.
>
---
#### [new 058] Finetuning Vision-Language Models as OCR Systems for Low-Resource Languages: A Case Study of Manchu
- **分类: cs.CV**

- **简介: 该论文属于OCR任务，解决低资源语言Manchu的文档识别问题。通过微调视觉语言模型，在合成数据上训练并实现高准确率，有效迁移至真实手写文档。**

- **链接: [http://arxiv.org/pdf/2507.06761v1](http://arxiv.org/pdf/2507.06761v1)**

> **作者:** Yan Hon Michael Chung; Donghyeok Choi
>
> **摘要:** Manchu, a critically endangered language essential for understanding early modern Eastern Eurasian history, lacks effective OCR systems that can handle real-world historical documents. This study develops high-performing OCR systems by fine-tuning three open-source vision-language models (LLaMA-3.2-11B, Qwen2.5-VL-7B, Qwen2.5-VL-3B) on 60,000 synthetic Manchu word images using parameter-efficient training. LLaMA-3.2-11B achieved exceptional performance with 98.3\% word accuracy and 0.0024 character error rate on synthetic data, while crucially maintaining 93.1\% accuracy on real-world handwritten documents. Comparative evaluation reveals substantial advantages over traditional approaches: while a CRNN baseline achieved 99.8\% synthetic accuracy, it suffered severe degradation to 72.5\% on real documents. Our approach demonstrates effective synthetic-to-real domain transfer, providing a cost-effective solution deployable on accessible infrastructure. This work establishes a transferable framework for endangered language OCR that removes technical and financial barriers in digital humanities, enabling historians and linguists to process historical archives without specialized computing resources. Code and model weights are available at https://github.com/mic7ch1/ManchuAI-OCR.
>
---
#### [new 059] Centralized Copy-Paste: Enhanced Data Augmentation Strategy for Wildland Fire Semantic Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像分割任务，针对野火分类数据少的问题，提出CCPDA增强方法，通过复制粘贴优化火区分割效果。**

- **链接: [http://arxiv.org/pdf/2507.06321v1](http://arxiv.org/pdf/2507.06321v1)**

> **作者:** Joon Tai Kim; Tianle Chen; Ziyu Dong; Nishanth Kunchala; Alexander Guller; Daniel Ospina Acero; Roger Williams; Mrinal Kumar
>
> **备注:** 21 pages, 5 figures, and under review for AIAA SciTech 2026
>
> **摘要:** Collecting and annotating images for the purpose of training segmentation models is often cost prohibitive. In the domain of wildland fire science, this challenge is further compounded by the scarcity of reliable public datasets with labeled ground truth. This paper presents the Centralized Copy-Paste Data Augmentation (CCPDA) method, for the purpose of assisting with the training of deep-learning multiclass segmentation models, with special focus on improving segmentation outcomes for the fire-class. CCPDA has three main steps: (i) identify fire clusters in the source image, (ii) apply a centralization technique to focus on the core of the fire area, and (iii) paste the refined fire clusters onto a target image. This method increases dataset diversity while preserving the essential characteristics of the fire class. The effectiveness of this augmentation technique is demonstrated via numerical analysis and comparison against various other augmentation methods using a weighted sum-based multi-objective optimization approach. This approach helps elevate segmentation performance metrics specific to the fire class, which carries significantly more operational significance than other classes (fuel, ash, or background). Numerical performance assessment validates the efficacy of the presented CCPDA method in alleviating the difficulties associated with small, manually labeled training datasets. It also illustrates that CCPDA outperforms other augmentation strategies in the application scenario considered, particularly in improving fire-class segmentation performance.
>
---
#### [new 060] StixelNExT++: Lightweight Monocular Scene Segmentation and Representation for Collective Perception
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于单目场景分割任务，旨在提升自动驾驶中的集体感知。通过改进Stixel表示，实现高效3D场景建模与分割，具有实时性与轻量化特点。**

- **链接: [http://arxiv.org/pdf/2507.06687v1](http://arxiv.org/pdf/2507.06687v1)**

> **作者:** Marcel Vosshans; Omar Ait-Aider; Youcef Mezouar; Markus Enzweiler
>
> **摘要:** This paper presents StixelNExT++, a novel approach to scene representation for monocular perception systems. Building on the established Stixel representation, our method infers 3D Stixels and enhances object segmentation by clustering smaller 3D Stixel units. The approach achieves high compression of scene information while remaining adaptable to point cloud and bird's-eye-view representations. Our lightweight neural network, trained on automatically generated LiDAR-based ground truth, achieves real-time performance with computation times as low as 10 ms per frame. Experimental results on the Waymo dataset demonstrate competitive performance within a 30-meter range, highlighting the potential of StixelNExT++ for collective perception in autonomous systems.
>
---
#### [new 061] Free on the Fly: Enhancing Flexibility in Test-Time Adaptation with Online EM
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型的测试时适应任务，解决领域偏移导致的性能下降问题。提出FreeTTA方法，无需训练，通过在线EM算法提升预测灵活性。**

- **链接: [http://arxiv.org/pdf/2507.06973v1](http://arxiv.org/pdf/2507.06973v1)**

> **作者:** Qiyuan Dai; Sibei Yang
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** Vision-Language Models (VLMs) have become prominent in open-world image recognition for their strong generalization abilities. Yet, their effectiveness in practical applications is compromised by domain shifts and distributional changes, especially when test data distributions diverge from training data. Therefore, the paradigm of test-time adaptation (TTA) has emerged, enabling the use of online off-the-shelf data at test time, supporting independent sample predictions, and eliminating reliance on test annotations. Traditional TTA methods, however, often rely on costly training or optimization processes, or make unrealistic assumptions about accessing or storing historical training and test data. Instead, this study proposes FreeTTA, a training-free and universally available method that makes no assumptions, to enhance the flexibility of TTA. More importantly, FreeTTA is the first to explicitly model the test data distribution, enabling the use of intrinsic relationships among test samples to enhance predictions of individual samples without simultaneous access--a direction not previously explored. FreeTTA achieves these advantages by introducing an online EM algorithm that utilizes zero-shot predictions from VLMs as priors to iteratively compute the posterior probabilities of each online test sample and update parameters. Experiments demonstrate that FreeTTA achieves stable and significant improvements compared to state-of-the-art methods across 15 datasets in both cross-domain and out-of-distribution settings.
>
---
#### [new 062] SemRaFiner: Panoptic Segmentation in Sparse and Noisy Radar Point Clouds
- **分类: cs.CV**

- **简介: 该论文属于雷达点云语义分割任务，解决稀疏噪声雷达数据下的全景分割问题。提出SemRaFiner方法提升分割精度与实例分配效果。**

- **链接: [http://arxiv.org/pdf/2507.06906v1](http://arxiv.org/pdf/2507.06906v1)**

> **作者:** Matthias Zeller; Daniel Casado Herraez; Bengisu Ayan; Jens Behley; Michael Heidingsfeld; Cyrill Stachniss
>
> **备注:** Accepted for publication in IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Semantic scene understanding, including the perception and classification of moving agents, is essential to enabling safe and robust driving behaviours of autonomous vehicles. Cameras and LiDARs are commonly used for semantic scene understanding. However, both sensor modalities face limitations in adverse weather and usually do not provide motion information. Radar sensors overcome these limitations and directly offer information about moving agents by measuring the Doppler velocity, but the measurements are comparably sparse and noisy. In this paper, we address the problem of panoptic segmentation in sparse radar point clouds to enhance scene understanding. Our approach, called SemRaFiner, accounts for changing density in sparse radar point clouds and optimizes the feature extraction to improve accuracy. Furthermore, we propose an optimized training procedure to refine instance assignments by incorporating a dedicated data augmentation. Our experiments suggest that our approach outperforms state-of-the-art methods for radar-based panoptic segmentation.
>
---
#### [new 063] A Neural Representation Framework with LLM-Driven Spatial Reasoning for Open-Vocabulary 3D Visual Grounding
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D视觉定位任务，解决开放词汇下基于语言查询的3D目标定位问题。提出SpatialReasoner框架，结合LLM和视觉属性增强空间推理能力。**

- **链接: [http://arxiv.org/pdf/2507.06719v1](http://arxiv.org/pdf/2507.06719v1)**

> **作者:** Zhenyang Liu; Sixiao Zheng; Siyu Chen; Cairong Zhao; Longfei Liang; Xiangyang Xue; Yanwei Fu
>
> **摘要:** Open-vocabulary 3D visual grounding aims to localize target objects based on free-form language queries, which is crucial for embodied AI applications such as autonomous navigation, robotics, and augmented reality. Learning 3D language fields through neural representations enables accurate understanding of 3D scenes from limited viewpoints and facilitates the localization of target objects in complex environments. However, existing language field methods struggle to accurately localize instances using spatial relations in language queries, such as ``the book on the chair.'' This limitation mainly arises from inadequate reasoning about spatial relations in both language queries and 3D scenes. In this work, we propose SpatialReasoner, a novel neural representation-based framework with large language model (LLM)-driven spatial reasoning that constructs a visual properties-enhanced hierarchical feature field for open-vocabulary 3D visual grounding. To enable spatial reasoning in language queries, SpatialReasoner fine-tunes an LLM to capture spatial relations and explicitly infer instructions for the target, anchor, and spatial relation. To enable spatial reasoning in 3D scenes, SpatialReasoner incorporates visual properties (opacity and color) to construct a hierarchical feature field. This field represents language and instance features using distilled CLIP features and masks extracted via the Segment Anything Model (SAM). The field is then queried using the inferred instructions in a hierarchical manner to localize the target 3D instance based on the spatial relation in the language query. Extensive experiments show that our framework can be seamlessly integrated into different neural representations, outperforming baseline models in 3D visual grounding while empowering their spatial reasoning capability.
>
---
#### [new 064] MCA-RG: Enhancing LLMs with Medical Concept Alignment for Radiology Report Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于放射学报告生成任务，旨在解决医学概念与视觉特征对齐困难的问题。提出MCA-RG框架，通过概念对齐和增强提升报告准确性。**

- **链接: [http://arxiv.org/pdf/2507.06992v1](http://arxiv.org/pdf/2507.06992v1)**

> **作者:** Qilong Xing; Zikai Song; Youjia Zhang; Na Feng; Junqing Yu; Wei Yang
>
> **备注:** MICCAI 2025
>
> **摘要:** Despite significant advancements in adapting Large Language Models (LLMs) for radiology report generation (RRG), clinical adoption remains challenging due to difficulties in accurately mapping pathological and anatomical features to their corresponding text descriptions. Additionally, semantic agnostic feature extraction further hampers the generation of accurate diagnostic reports. To address these challenges, we introduce Medical Concept Aligned Radiology Report Generation (MCA-RG), a knowledge-driven framework that explicitly aligns visual features with distinct medical concepts to enhance the report generation process. MCA-RG utilizes two curated concept banks: a pathology bank containing lesion-related knowledge, and an anatomy bank with anatomical descriptions. The visual features are aligned with these medical concepts and undergo tailored enhancement. We further propose an anatomy-based contrastive learning procedure to improve the generalization of anatomical features, coupled with a matching loss for pathological features to prioritize clinically relevant regions. Additionally, a feature gating mechanism is employed to filter out low-quality concept features. Finally, the visual features are corresponding to individual medical concepts, and are leveraged to guide the report generation process. Experiments on two public benchmarks (MIMIC-CXR and CheXpert Plus) demonstrate that MCA-RG achieves superior performance, highlighting its effectiveness in radiology report generation.
>
---
#### [new 065] Evaluating Attribute Confusion in Fashion Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成的评估任务，旨在解决属性混淆问题。通过引入局部化VQA方法，提出L-VQAScore指标，提升对实体-属性关联的评估准确性。**

- **链接: [http://arxiv.org/pdf/2507.07079v1](http://arxiv.org/pdf/2507.07079v1)**

> **作者:** Ziyue Liu; Federico Girella; Yiming Wang; Davide Talon
>
> **备注:** Accepted to ICIAP25. Project page: site [https://intelligolabs.github.io/L-VQAScore/\
>
> **摘要:** Despite the rapid advances in Text-to-Image (T2I) generation models, their evaluation remains challenging in domains like fashion, involving complex compositional generation. Recent automated T2I evaluation methods leverage pre-trained vision-language models to measure cross-modal alignment. However, our preliminary study reveals that they are still limited in assessing rich entity-attribute semantics, facing challenges in attribute confusion, i.e., when attributes are correctly depicted but associated to the wrong entities. To address this, we build on a Visual Question Answering (VQA) localization strategy targeting one single entity at a time across both visual and textual modalities. We propose a localized human evaluation protocol and introduce a novel automatic metric, Localized VQAScore (L-VQAScore), that combines visual localization with VQA probing both correct (reflection) and miss-localized (leakage) attribute generation. On a newly curated dataset featuring challenging compositional alignment scenarios, L-VQAScore outperforms state-of-the-art T2I evaluation methods in terms of correlation with human judgments, demonstrating its strength in capturing fine-grained entity-attribute associations. We believe L-VQAScore can be a reliable and scalable alternative to subjective evaluations.
>
---
#### [new 066] MOST: Motion Diffusion Model for Rare Text via Temporal Clip Banzhaf Interaction
- **分类: cs.CV**

- **简介: 该论文属于文本到动作生成任务，解决稀有语言提示下动作生成的问题。通过引入时间片段Banzhaf交互机制，提升文本与动作的细粒度匹配效果。**

- **链接: [http://arxiv.org/pdf/2507.06590v1](http://arxiv.org/pdf/2507.06590v1)**

> **作者:** Yin Wang; Mu li; Zhiying Leng; Frederick W. B. Li; Xiaohui Liang
>
> **摘要:** We introduce MOST, a novel motion diffusion model via temporal clip Banzhaf interaction, aimed at addressing the persistent challenge of generating human motion from rare language prompts. While previous approaches struggle with coarse-grained matching and overlook important semantic cues due to motion redundancy, our key insight lies in leveraging fine-grained clip relationships to mitigate these issues. MOST's retrieval stage presents the first formulation of its kind - temporal clip Banzhaf interaction - which precisely quantifies textual-motion coherence at the clip level. This facilitates direct, fine-grained text-to-motion clip matching and eliminates prevalent redundancy. In the generation stage, a motion prompt module effectively utilizes retrieved motion clips to produce semantically consistent movements. Extensive evaluations confirm that MOST achieves state-of-the-art text-to-motion retrieval and generation performance by comprehensively addressing previous challenges, as demonstrated through quantitative and qualitative results highlighting its effectiveness, especially for rare prompts.
>
---
#### [new 067] ILNet: Trajectory Prediction with Inverse Learning Attention for Enhancing Intention Capture
- **分类: cs.CV**

- **简介: 该论文属于多智能体轨迹预测任务，旨在解决交互场景中行为意图捕捉不足和环境适应性差的问题。提出ILNet模型，结合逆学习注意力和动态锚点选择模块，提升预测精度与多模态性。**

- **链接: [http://arxiv.org/pdf/2507.06531v1](http://arxiv.org/pdf/2507.06531v1)**

> **作者:** Mingjin Zeng; Nan Ouyang; Wenkang Wan; Lei Ao; Qing Cai; Kai Sheng
>
> **摘要:** Trajectory prediction for multi-agent interaction scenarios is a crucial challenge. Most advanced methods model agent interactions by efficiently factorized attention based on the temporal and agent axes. However, this static and foward modeling lacks explicit interactive spatio-temporal coordination, capturing only obvious and immediate behavioral intentions. Alternatively, the modern trajectory prediction framework refines the successive predictions by a fixed-anchor selection strategy, which is difficult to adapt in different future environments. It is acknowledged that human drivers dynamically adjust initial driving decisions based on further assumptions about the intentions of surrounding vehicles. Motivated by human driving behaviors, this paper proposes ILNet, a multi-agent trajectory prediction method with Inverse Learning (IL) attention and Dynamic Anchor Selection (DAS) module. IL Attention employs an inverse learning paradigm to model interactions at neighboring moments, introducing proposed intentions to dynamically encode the spatio-temporal coordination of interactions, thereby enhancing the model's ability to capture complex interaction patterns. Then, the learnable DAS module is proposed to extract multiple trajectory change keypoints as anchors in parallel with almost no increase in parameters. Experimental results show that the ILNet achieves state-of-the-art performance on the INTERACTION and Argoverse motion forecasting datasets. Particularly, in challenged interaction scenarios, ILNet achieves higher accuracy and more multimodal distributions of trajectories over fewer parameters. Our codes are available at https://github.com/mjZeng11/ILNet.
>
---
#### [new 068] Advancing Offline Handwritten Text Recognition: A Systematic Review of Data Augmentation and Generation Techniques
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于手写文本识别任务，旨在解决数据不足问题。通过综述数据增强与生成技术，提升HTR系统的准确性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.06275v1](http://arxiv.org/pdf/2507.06275v1)**

> **作者:** Yassin Hussein Rassul; Aram M. Ahmed; Polla Fattah; Bryar A. Hassan; Arwaa W. Abdulkareem; Tarik A. Rashid; Joan Lu
>
> **摘要:** Offline Handwritten Text Recognition (HTR) systems play a crucial role in applications such as historical document digitization, automatic form processing, and biometric authentication. However, their performance is often hindered by the limited availability of annotated training data, particularly for low-resource languages and complex scripts. This paper presents a comprehensive survey of offline handwritten data augmentation and generation techniques designed to improve the accuracy and robustness of HTR systems. We systematically examine traditional augmentation methods alongside recent advances in deep learning, including Generative Adversarial Networks (GANs), diffusion models, and transformer-based approaches. Furthermore, we explore the challenges associated with generating diverse and realistic handwriting samples, particularly in preserving script authenticity and addressing data scarcity. This survey follows the PRISMA methodology, ensuring a structured and rigorous selection process. Our analysis began with 1,302 primary studies, which were filtered down to 848 after removing duplicates, drawing from key academic sources such as IEEE Digital Library, Springer Link, Science Direct, and ACM Digital Library. By evaluating existing datasets, assessment metrics, and state-of-the-art methodologies, this survey identifies key research gaps and proposes future directions to advance the field of handwritten text generation across diverse linguistic and stylistic landscapes.
>
---
#### [new 069] CheXPO: Preference Optimization for Chest X-ray VLMs with Counterfactual Rationale
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学视觉语言模型任务，旨在解决医疗场景中的幻觉问题。通过偏好优化和反事实推理，提升模型可靠性与性能。**

- **链接: [http://arxiv.org/pdf/2507.06959v1](http://arxiv.org/pdf/2507.06959v1)**

> **作者:** Xiao Liang; Jiawei Hu; Di Wang; Zhi Ma; Lin Zhao; Ronghan Li; Bo Wan; Quan Wang
>
> **摘要:** Vision-language models (VLMs) are prone to hallucinations that critically compromise reliability in medical applications. While preference optimization can mitigate these hallucinations through clinical feedback, its implementation faces challenges such as clinically irrelevant training samples, imbalanced data distributions, and prohibitive expert annotation costs. To address these challenges, we introduce CheXPO, a Chest X-ray Preference Optimization strategy that combines confidence-similarity joint mining with counterfactual rationale. Our approach begins by synthesizing a unified, fine-grained multi-task chest X-ray visual instruction dataset across different question types for supervised fine-tuning (SFT). We then identify hard examples through token-level confidence analysis of SFT failures and use similarity-based retrieval to expand hard examples for balancing preference sample distributions, while synthetic counterfactual rationales provide fine-grained clinical preferences, eliminating the need for additional expert input. Experiments show that CheXPO achieves 8.93% relative performance gain using only 5% of SFT samples, reaching state-of-the-art performance across diverse clinical tasks and providing a scalable, interpretable solution for real-world radiology applications.
>
---
#### [new 070] A multi-modal dataset for insect biodiversity with imagery and DNA at the trap and individual level
- **分类: cs.CV**

- **简介: 该论文属于昆虫多样性研究任务，旨在解决批量样本自动分类问题。通过构建结合图像与DNA数据的多模态数据集，提升昆虫群落快速识别能力。**

- **链接: [http://arxiv.org/pdf/2507.06972v1](http://arxiv.org/pdf/2507.06972v1)**

> **作者:** Johanna Orsholm; John Quinto; Hannu Autto; Gaia Banelyte; Nicolas Chazot; Jeremy deWaard; Stephanie deWaard; Arielle Farrell; Brendan Furneaux; Bess Hardwick; Nao Ito; Amlan Kar; Oula Kalttopää; Deirdre Kerdraon; Erik Kristensen; Jaclyn McKeown; Tommi Mononen; Ellen Nein; Hanna Rogers; Tomas Roslin; Paula Schmitz; Jayme Sones; Maija Sujala; Amy Thompson; Evgeny V. Zakharov; Iuliia Zarubiieva; Akshita Gupta; Scott C. Lowe; Graham W. Taylor
>
> **备注:** 13 pages, 6 figures, submitted to Scientific Data
>
> **摘要:** Insects comprise millions of species, many experiencing severe population declines under environmental and habitat changes. High-throughput approaches are crucial for accelerating our understanding of insect diversity, with DNA barcoding and high-resolution imaging showing strong potential for automatic taxonomic classification. However, most image-based approaches rely on individual specimen data, unlike the unsorted bulk samples collected in large-scale ecological surveys. We present the Mixed Arthropod Sample Segmentation and Identification (MassID45) dataset for training automatic classifiers of bulk insect samples. It uniquely combines molecular and imaging data at both the unsorted sample level and the full set of individual specimens. Human annotators, supported by an AI-assisted tool, performed two tasks on bulk images: creating segmentation masks around each individual arthropod and assigning taxonomic labels to over 17 000 specimens. Combining the taxonomic resolution of DNA barcodes with precise abundance estimates of bulk images holds great potential for rapid, large-scale characterization of insect communities. This dataset pushes the boundaries of tiny object detection and instance segmentation, fostering innovation in both ecological and machine learning research.
>
---
#### [new 071] Segmentation Regularized Training for Multi-Domain Deep Learning Registration applied to MR-Guided Prostate Cancer Radiotherapy
- **分类: cs.CV; physics.med-ph**

- **简介: 该论文属于医学图像配准任务，旨在解决多域MR图像配准问题，通过深度学习方法提升前列腺癌放疗中的轮廓传播和剂量积累准确性。**

- **链接: [http://arxiv.org/pdf/2507.06966v1](http://arxiv.org/pdf/2507.06966v1)**

> **作者:** Sudharsan Madhavan; Chengcheng Gui; Lando Bosma; Josiah Simeth; Jue Jiang; Nicolas Cote; Nima Hassan Rezaeian; Himanshu Nagar; Victoria Brennan; Neelam Tyagi; Harini Veeraraghavan
>
> **备注:** Preprint in preparation for submission
>
> **摘要:** Background: Accurate deformable image registration (DIR) is required for contour propagation and dose accumulation in MR-guided adaptive radiotherapy (MRgART). This study trained and evaluated a deep learning DIR method for domain invariant MR-MR registration. Methods: A progressively refined registration and segmentation (ProRSeg) method was trained with 262 pairs of 3T MR simulation scans from prostate cancer patients using weighted segmentation consistency loss. ProRSeg was tested on same- (58 pairs), cross- (72 1.5T MR Linac pairs), and mixed-domain (42 MRSim-MRL pairs) datasets for contour propagation accuracy of clinical target volume (CTV), bladder, and rectum. Dose accumulation was performed for 42 patients undergoing 5-fraction MRgART. Results: ProRSeg demonstrated generalization for bladder with similar Dice Similarity Coefficients across domains (0.88, 0.87, 0.86). For rectum and CTV, performance was domain-dependent with higher accuracy on cross-domain MRL dataset (DSCs 0.89) versus same-domain data. The model's strong cross-domain performance prompted us to study the feasibility of using it for dose accumulation. Dose accumulation showed 83.3% of patients met CTV coverage (D95 >= 40.0 Gy) and bladder sparing (D50 <= 20.0 Gy) constraints. All patients achieved minimum mean target dose (>40.4 Gy), but only 9.5% remained under upper limit (<42.0 Gy). Conclusions: ProRSeg showed reasonable multi-domain MR-MR registration performance for prostate cancer patients with preliminary feasibility for evaluating treatment compliance to clinical constraints.
>
---
#### [new 072] Unveiling the Underwater World: CLIP Perception Model-Guided Underwater Image Enhancement
- **分类: cs.CV**

- **简介: 该论文属于 underwater image enhancement 任务，解决增强图像感知质量与内容恢复问题。通过引入 CLIP 模型构建感知损失模块，提升图像质量与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.06234v1](http://arxiv.org/pdf/2507.06234v1)**

> **作者:** Jiangzhong Cao; Zekai Zeng; Xu Zhang; Huan Zhang; Chunling Fan; Gangyi Jiang; Weisi Lin
>
> **备注:** 10 pages, 7 figures;Accepted to PR 2025;The source code is available at https://github.com/Ave001025/UIE_CLIP
>
> **摘要:** High-quality underwater images are essential for both machine vision tasks and viewers with their aesthetic appeal.However, the quality of underwater images is severely affected by light absorption and scattering. Deep learning-based methods for Underwater Image Enhancement (UIE) have achieved good performance. However, these methods often overlook considering human perception and lack sufficient constraints within the solution space. Consequently, the enhanced images often suffer from diminished perceptual quality or poor content restoration.To address these issues, we propose a UIE method with a Contrastive Language-Image Pre-Training (CLIP) perception loss module and curriculum contrastive regularization. Above all, to develop a perception model for underwater images that more aligns with human visual perception, the visual semantic feature extraction capability of the CLIP model is leveraged to learn an appropriate prompt pair to map and evaluate the quality of underwater images. This CLIP perception model is then incorporated as a perception loss module into the enhancement network to improve the perceptual quality of enhanced images. Furthermore, the CLIP perception model is integrated with the curriculum contrastive regularization to enhance the constraints imposed on the enhanced images within the CLIP perceptual space, mitigating the risk of both under-enhancement and over-enhancement. Specifically, the CLIP perception model is employed to assess and categorize the learning difficulty level of negatives in the regularization process, ensuring comprehensive and nuanced utilization of distorted images and negatives with varied quality levels. Extensive experiments demonstrate that our method outperforms state-of-the-art methods in terms of visual quality and generalization ability.
>
---
#### [new 073] Integrating Pathology Foundation Models and Spatial Transcriptomics for Cellular Decomposition from Histology Images
- **分类: cs.CV**

- **简介: 该论文属于病理图像分析任务，旨在无需空间转录组技术即可预测细胞组成。通过结合病理基础模型与轻量回归器，实现高效准确的细胞分解。**

- **链接: [http://arxiv.org/pdf/2507.07013v1](http://arxiv.org/pdf/2507.07013v1)**

> **作者:** Yutong Sun; Sichen Zhu; Peng Qiu
>
> **摘要:** The rapid development of digital pathology and modern deep learning has facilitated the emergence of pathology foundation models that are expected to solve general pathology problems under various disease conditions in one unified model, with or without fine-tuning. In parallel, spatial transcriptomics has emerged as a transformative technology that enables the profiling of gene expression on hematoxylin and eosin (H&E) stained histology images. Spatial transcriptomics unlocks the unprecedented opportunity to dive into existing histology images at a more granular, cellular level. In this work, we propose a lightweight and training-efficient approach to predict cellular composition directly from H&E-stained histology images by leveraging information-enriched feature embeddings extracted from pre-trained pathology foundation models. By training a lightweight multi-layer perceptron (MLP) regressor on cell-type abundances derived via cell2location, our method efficiently distills knowledge from pathology foundation models and demonstrates the ability to accurately predict cell-type compositions from histology images, without physically performing the costly spatial transcriptomics. Our method demonstrates competitive performance compared to existing methods such as Hist2Cell, while significantly reducing computational complexity.
>
---
#### [new 074] Omni-Fusion of Spatial and Spectral for Hyperspectral Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于高光谱图像分割任务，旨在解决空间与光谱信息融合困难的问题。提出Omni-Fuse网络，通过跨维度特征融合提升分割性能。**

- **链接: [http://arxiv.org/pdf/2507.06606v1](http://arxiv.org/pdf/2507.06606v1)**

> **作者:** Qing Zhang; Guoquan Pei; Yan Wang
>
> **摘要:** Medical Hyperspectral Imaging (MHSI) has emerged as a promising tool for enhanced disease diagnosis, particularly in computational pathology, offering rich spectral information that aids in identifying subtle biochemical properties of tissues. Despite these advantages, effectively fusing both spatial-dimensional and spectral-dimensional information from MHSIs remains challenging due to its high dimensionality and spectral redundancy inherent characteristics. To solve the above challenges, we propose a novel spatial-spectral omni-fusion network for hyperspectral image segmentation, named as Omni-Fuse. Here, we introduce abundant cross-dimensional feature fusion operations, including a cross-dimensional enhancement module that refines both spatial and spectral features through bidirectional attention mechanisms, a spectral-guided spatial query selection to select the most spectral-related spatial feature as the query, and a two-stage cross-dimensional decoder which dynamically guide the model to focus on the selected spatial query. Despite of numerous attention blocks, Omni-Fuse remains efficient in execution. Experiments on two microscopic hyperspectral image datasets show that our approach can significantly improve the segmentation performance compared with the state-of-the-art methods, with over 5.73 percent improvement in DSC. Code available at: https://github.com/DeepMed-Lab-ECNU/Omni-Fuse.
>
---
#### [new 075] A Probabilistic Approach to Uncertainty Quantification Leveraging 3D Geometry
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D场景重建任务，旨在解决神经隐式SDF模型中的不确定性量化问题。提出BayesSDF框架，通过Hessian度量实现高效、几何感知的不确定性估计。**

- **链接: [http://arxiv.org/pdf/2507.06269v1](http://arxiv.org/pdf/2507.06269v1)**

> **作者:** Rushil Desai; Frederik Warburg; Trevor Darrell; Marissa Ramirez de Chanlatte
>
> **备注:** ICCV 2025 Workshops (8 Pages, 6 Figures, 2 Tables)
>
> **摘要:** Quantifying uncertainty in neural implicit 3D representations, particularly those utilizing Signed Distance Functions (SDFs), remains a substantial challenge due to computational inefficiencies, scalability issues, and geometric inconsistencies. Existing methods typically neglect direct geometric integration, leading to poorly calibrated uncertainty maps. We introduce BayesSDF, a novel probabilistic framework for uncertainty quantification in neural implicit SDF models, motivated by scientific simulation applications with 3D environments (e.g., forests) such as modeling fluid flow through forests, where precise surface geometry and awareness of fidelity surface geometric uncertainty are essential. Unlike radiance-based models such as NeRF or 3D Gaussian splatting, which lack explicit surface formulations, SDFs define continuous and differentiable geometry, making them better suited for physical modeling and analysis. BayesSDF leverages a Laplace approximation to quantify local surface instability via Hessian-based metrics, enabling computationally efficient, surface-aware uncertainty estimation. Our method shows that uncertainty predictions correspond closely with poorly reconstructed geometry, providing actionable confidence measures for downstream use. Extensive evaluations on synthetic and real-world datasets demonstrate that BayesSDF outperforms existing methods in both calibration and geometric consistency, establishing a strong foundation for uncertainty-aware 3D scene reconstruction, simulation, and robotic decision-making.
>
---
#### [new 076] SPARC: Concept-Aligned Sparse Autoencoders for Cross-Model and Cross-Modal Interpretability
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于跨模型和跨模态解释性任务，旨在解决不同AI模型间概念表示不一致的问题。通过SPARC框架，实现统一的稀疏潜在空间，提升概念对齐与可比性。**

- **链接: [http://arxiv.org/pdf/2507.06265v1](http://arxiv.org/pdf/2507.06265v1)**

> **作者:** Ali Nasiri-Sarvi; Hassan Rivaz; Mahdi S. Hosseini
>
> **摘要:** Understanding how different AI models encode the same high-level concepts, such as objects or attributes, remains challenging because each model typically produces its own isolated representation. Existing interpretability methods like Sparse Autoencoders (SAEs) produce latent concepts individually for each model, resulting in incompatible concept spaces and limiting cross-model interpretability. To address this, we introduce SPARC (Sparse Autoencoders for Aligned Representation of Concepts), a new framework that learns a single, unified latent space shared across diverse architectures and modalities (e.g., vision models like DINO, and multimodal models like CLIP). SPARC's alignment is enforced through two key innovations: (1) a Global TopK sparsity mechanism, ensuring all input streams activate identical latent dimensions for a given concept; and (2) a Cross-Reconstruction Loss, which explicitly encourages semantic consistency between models. On Open Images, SPARC dramatically improves concept alignment, achieving a Jaccard similarity of 0.80, more than tripling the alignment compared to previous methods. SPARC creates a shared sparse latent space where individual dimensions often correspond to similar high-level concepts across models and modalities, enabling direct comparison of how different architectures represent identical concepts without requiring manual alignment or model-specific analysis. As a consequence of this aligned representation, SPARC also enables practical applications such as text-guided spatial localization in vision-only models and cross-model/cross-modal retrieval. Code and models are available at https://github.com/AtlasAnalyticsLab/SPARC.
>
---
#### [new 077] Reading a Ruler in the Wild
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉任务，解决在复杂环境下准确测量的问题。提出RulerNet和DeepGP，实现鲁棒的尺子读数与实时尺度估计。**

- **链接: [http://arxiv.org/pdf/2507.07077v1](http://arxiv.org/pdf/2507.07077v1)**

> **作者:** Yimu Pan; Manas Mehta; Gwen Sincerbeaux; Jeffery A. Goldstein; Alison D. Gernand; James Z. Wang
>
> **摘要:** Accurately converting pixel measurements into absolute real-world dimensions remains a fundamental challenge in computer vision and limits progress in key applications such as biomedicine, forensics, nutritional analysis, and e-commerce. We introduce RulerNet, a deep learning framework that robustly infers scale "in the wild" by reformulating ruler reading as a unified keypoint-detection problem and by representing the ruler with geometric-progression parameters that are invariant to perspective transformations. Unlike traditional methods that rely on handcrafted thresholds or rigid, ruler-specific pipelines, RulerNet directly localizes centimeter marks using a distortion-invariant annotation and training strategy, enabling strong generalization across diverse ruler types and imaging conditions while mitigating data scarcity. We also present a scalable synthetic-data pipeline that combines graphics-based ruler generation with ControlNet to add photorealistic context, greatly increasing training diversity and improving performance. To further enhance robustness and efficiency, we propose DeepGP, a lightweight feed-forward network that regresses geometric-progression parameters from noisy marks and eliminates iterative optimization, enabling real-time scale estimation on mobile or edge devices. Experiments show that RulerNet delivers accurate, consistent, and efficient scale estimates under challenging real-world conditions. These results underscore its utility as a generalizable measurement tool and its potential for integration with other vision components for automated, scale-aware analysis in high-impact domains. A live demo is available at https://huggingface.co/spaces/ymp5078/RulerNet-Demo.
>
---
#### [new 078] Hierarchical Feature Alignment for Gloss-Free Sign Language Translation
- **分类: cs.CV**

- **简介: 该论文属于手语翻译任务，解决视觉与文本表示不匹配问题。提出分层特征对齐方法，提升翻译效果。**

- **链接: [http://arxiv.org/pdf/2507.06732v1](http://arxiv.org/pdf/2507.06732v1)**

> **作者:** Sobhan Asasi; Mohamed Ilyes Lakhal; Richard Bowden
>
> **备注:** Accepted in SLTAT
>
> **摘要:** Sign Language Translation (SLT) attempts to convert sign language videos into spoken sentences. However, many existing methods struggle with the disparity between visual and textual representations during end-to-end learning. Gloss-based approaches help to bridge this gap by leveraging structured linguistic information. While, gloss-free methods offer greater flexibility and remove the burden of annotation, they require effective alignment strategies. Recent advances in Large Language Models (LLMs) have enabled gloss-free SLT by generating text-like representations from sign videos. In this work, we introduce a novel hierarchical pre-training strategy inspired by the structure of sign language, incorporating pseudo-glosses and contrastive video-language alignment. Our method hierarchically extracts features at frame, segment, and video levels, aligning them with pseudo-glosses and the spoken sentence to enhance translation quality. Experiments demonstrate that our approach improves BLEU-4 and ROUGE scores while maintaining efficiency.
>
---
#### [new 079] Video-RTS: Rethinking Reinforcement Learning and Test-Time Scaling for Efficient and Enhanced Video Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频推理任务，旨在解决数据收集和微调成本高的问题。通过结合高效强化学习与视频自适应测试时缩放策略，提升推理效率与性能。**

- **链接: [http://arxiv.org/pdf/2507.06485v1](http://arxiv.org/pdf/2507.06485v1)**

> **作者:** Ziyang Wang; Jaehong Yoon; Shoubin Yu; Md Mohaiminul Islam; Gedas Bertasius; Mohit Bansal
>
> **备注:** The first two authors contributed equally. Project page: https://sites.google.com/cs.unc.edu/videorts2025/
>
> **摘要:** Despite advances in reinforcement learning (RL)-based video reasoning with large language models (LLMs), data collection and finetuning remain significant challenges. These methods often rely on large-scale supervised fine-tuning (SFT) with extensive video data and long Chain-of-Thought (CoT) annotations, making them costly and hard to scale. To address this, we present Video-RTS, a new approach to improve video reasoning capability with drastically improved data efficiency by combining data-efficient RL with a video-adaptive test-time scaling (TTS) strategy. Based on observations about the data scaling of RL samples, we skip the resource-intensive SFT step and employ efficient pure-RL training with output-based rewards, requiring no additional annotations or extensive fine-tuning. Furthermore, to utilize computational resources more efficiently, we introduce a sparse-to-dense video TTS strategy that improves inference by iteratively adding frames based on output consistency. We validate our approach on multiple video reasoning benchmarks, showing that Video-RTS surpasses existing video reasoning models by an average of 2.4% in accuracy using only 3.6% training samples. For example, Video-RTS achieves a 4.2% improvement on Video-Holmes, a recent and challenging video reasoning benchmark, and a 2.6% improvement on MMVU. Notably, our pure RL training and adaptive video TTS offer complementary strengths, enabling Video-RTS's strong reasoning performance.
>
---
#### [new 080] PointVDP: Learning View-Dependent Projection by Fireworks Rays for 3D Point Cloud Segmentation
- **分类: cs.CV**

- **简介: 该论文属于3D点云分割任务，解决传统投影方法依赖人工设定、缺乏多样性问题。提出View-Dependent Projection（VDP）框架，通过数据驱动生成高效投影，提升分割性能。**

- **链接: [http://arxiv.org/pdf/2507.06618v1](http://arxiv.org/pdf/2507.06618v1)**

> **作者:** Yang Chen; Yueqi Duan; Haowen Sun; Ziwei Wang; Jiwen Lu; Yap-Peng Tan
>
> **摘要:** In this paper, we propose view-dependent projection (VDP) to facilitate point cloud segmentation, designing efficient 3D-to-2D mapping that dynamically adapts to the spatial geometry from view variations. Existing projection-based methods leverage view-independent projection in complex scenes, relying on straight lines to generate direct rays or upward curves to reduce occlusions. However, their view independence provides projection rays that are limited to pre-defined parameters by human settings, restricting point awareness and failing to capture sufficient projection diversity across different view planes. Although multiple projections per view plane are commonly used to enhance spatial variety, the projected redundancy leads to excessive computational overhead and inefficiency in image processing. To address these limitations, we design a framework of VDP to generate data-driven projections from 3D point distributions, producing highly informative single-image inputs by predicting rays inspired by the adaptive behavior of fireworks. In addition, we construct color regularization to optimize the framework, which emphasizes essential features within semantic pixels and suppresses the non-semantic features within black pixels, thereby maximizing 2D space utilization in a projected image. As a result, our approach, PointVDP, develops lightweight projections in marginal computation costs. Experiments on S3DIS and ScanNet benchmarks show that our approach achieves competitive results, offering a resource-efficient solution for semantic understanding.
>
---
#### [new 081] THOR: Thermal-guided Hand-Object Reasoning via Adaptive Vision Sampling
- **分类: cs.CV**

- **简介: 该论文属于实时行为识别任务，解决穿戴式摄像头能耗高、数据量大问题，通过热感引导的自适应采样方法减少RGB数据处理量，提升效率与隐私保护。**

- **链接: [http://arxiv.org/pdf/2507.06442v1](http://arxiv.org/pdf/2507.06442v1)**

> **作者:** Soroush Shahi; Farzad Shahabi; Rama Nabulsi; Glenn Fernandes; Aggelos Katsaggelos; Nabil Alshurafa
>
> **摘要:** Wearable cameras are increasingly used as an observational and interventional tool for human behaviors by providing detailed visual data of hand-related activities. This data can be leveraged to facilitate memory recall for logging of behavior or timely interventions aimed at improving health. However, continuous processing of RGB images from these cameras consumes significant power impacting battery lifetime, generates a large volume of unnecessary video data for post-processing, raises privacy concerns, and requires substantial computational resources for real-time analysis. We introduce THOR, a real-time adaptive spatio-temporal RGB frame sampling method that leverages thermal sensing to capture hand-object patches and classify them in real-time. We use low-resolution thermal camera data to identify moments when a person switches from one hand-related activity to another, and adjust the RGB frame sampling rate by increasing it during activity transitions and reducing it during periods of sustained activity. Additionally, we use the thermal cues from the hand to localize the region of interest (i.e., the hand-object interaction) in each RGB frame, allowing the system to crop and process only the necessary part of the image for activity recognition. We develop a wearable device to validate our method through an in-the-wild study with 14 participants and over 30 activities, and further evaluate it on Ego4D (923 participants across 9 countries, totaling 3,670 hours of video). Our results show that using only 3% of the original RGB video data, our method captures all the activity segments, and achieves hand-related activity recognition F1-score (95%) comparable to using the entire RGB video (94%). Our work provides a more practical path for the longitudinal use of wearable cameras to monitor hand-related activities and health-risk behaviors in real time.
>
---
#### [new 082] MST-Distill: Mixture of Specialized Teachers for Cross-Modal Knowledge Distillation
- **分类: cs.CV; cs.LG; cs.MM**

- **简介: 该论文属于跨模态知识蒸馏任务，解决传统方法在跨模态场景中的数据异构性和知识漂移问题，提出MST-Distill框架提升蒸馏效果。**

- **链接: [http://arxiv.org/pdf/2507.07015v1](http://arxiv.org/pdf/2507.07015v1)**

> **作者:** Hui Li; Pengfei Yang; Juanyang Chen; Le Dong; Yanxin Chen; Quan Wang
>
> **备注:** Accepted to ACM MM 2025 (The 33rd ACM International Conference on Multimedia)
>
> **摘要:** Knowledge distillation as an efficient knowledge transfer technique, has achieved remarkable success in unimodal scenarios. However, in cross-modal settings, conventional distillation methods encounter significant challenges due to data and statistical heterogeneities, failing to leverage the complementary prior knowledge embedded in cross-modal teacher models. This paper empirically reveals two critical issues in existing approaches: distillation path selection and knowledge drift. To address these limitations, we propose MST-Distill, a novel cross-modal knowledge distillation framework featuring a mixture of specialized teachers. Our approach employs a diverse ensemble of teacher models across both cross-modal and multimodal configurations, integrated with an instance-level routing network that facilitates adaptive and dynamic distillation. This architecture effectively transcends the constraints of traditional methods that rely on monotonous and static teacher models. Additionally, we introduce a plug-in masking module, independently trained to suppress modality-specific discrepancies and reconstruct teacher representations, thereby mitigating knowledge drift and enhancing transfer effectiveness. Extensive experiments across five diverse multimodal datasets, spanning visual, audio, and text, demonstrate that our method significantly outperforms existing state-of-the-art knowledge distillation methods in cross-modal distillation tasks. The source code is available at https://github.com/Gray-OREO/MST-Distill.
>
---
#### [new 083] HVI-CIDNet+: Beyond Extreme Darkness for Low-Light Image Enhancement
- **分类: cs.CV**

- **简介: 该论文属于低光图像增强任务，旨在解决现有方法在极端黑暗下产生的颜色偏差和噪声问题。提出HVI-CIDNet+网络，结合新型HVI色彩空间与先验引导注意力机制，提升图像质量。**

- **链接: [http://arxiv.org/pdf/2507.06814v1](http://arxiv.org/pdf/2507.06814v1)**

> **作者:** Qingsen Yan; Kangbiao Shi; Yixu Feng; Tao Hu; Peng Wu; Guansong Pang; Yanning Zhang
>
> **备注:** 14 pages
>
> **摘要:** Low-Light Image Enhancement (LLIE) aims to restore vivid content and details from corrupted low-light images. However, existing standard RGB (sRGB) color space-based LLIE methods often produce color bias and brightness artifacts due to the inherent high color sensitivity. While Hue, Saturation, and Value (HSV) color space can decouple brightness and color, it introduces significant red and black noise artifacts. To address this problem, we propose a new color space for LLIE, namely Horizontal/Vertical-Intensity (HVI), defined by the HV color map and learnable intensity. The HV color map enforces small distances for the red coordinates to remove red noise artifacts, while the learnable intensity compresses the low-light regions to remove black noise artifacts. Additionally, we introduce the Color and Intensity Decoupling Network+ (HVI-CIDNet+), built upon the HVI color space, to restore damaged content and mitigate color distortion in extremely dark regions. Specifically, HVI-CIDNet+ leverages abundant contextual and degraded knowledge extracted from low-light images using pre-trained vision-language models, integrated via a novel Prior-guided Attention Block (PAB). Within the PAB, latent semantic priors can promote content restoration, while degraded representations guide precise color correction, both particularly in extremely dark regions through the meticulously designed cross-attention fusion mechanism. Furthermore, we construct a Region Refinement Block that employs convolution for information-rich regions and self-attention for information-scarce regions, ensuring accurate brightness adjustments. Comprehensive results from benchmark experiments demonstrate that the proposed HVI-CIDNet+ outperforms the state-of-the-art methods on 10 datasets.
>
---
#### [new 084] Go to Zero: Towards Zero-shot Motion Generation with Million-scale Data
- **分类: cs.CV**

- **简介: 该论文属于文本到动作生成任务，旨在解决零样本泛化问题。通过构建大规模数据集和评估框架，提升模型在未见动作上的生成能力。**

- **链接: [http://arxiv.org/pdf/2507.07095v1](http://arxiv.org/pdf/2507.07095v1)**

> **作者:** Ke Fan; Shunlin Lu; Minyue Dai; Runyi Yu; Lixing Xiao; Zhiyang Dou; Junting Dong; Lizhuang Ma; Jingbo Wang
>
> **备注:** Project Page: https://vankouf.github.io/MotionMillion/
>
> **摘要:** Generating diverse and natural human motion sequences based on textual descriptions constitutes a fundamental and challenging research area within the domains of computer vision, graphics, and robotics. Despite significant advancements in this field, current methodologies often face challenges regarding zero-shot generalization capabilities, largely attributable to the limited size of training datasets. Moreover, the lack of a comprehensive evaluation framework impedes the advancement of this task by failing to identify directions for improvement. In this work, we aim to push text-to-motion into a new era, that is, to achieve the generalization ability of zero-shot. To this end, firstly, we develop an efficient annotation pipeline and introduce MotionMillion-the largest human motion dataset to date, featuring over 2,000 hours and 2 million high-quality motion sequences. Additionally, we propose MotionMillion-Eval, the most comprehensive benchmark for evaluating zero-shot motion generation. Leveraging a scalable architecture, we scale our model to 7B parameters and validate its performance on MotionMillion-Eval. Our results demonstrate strong generalization to out-of-domain and complex compositional motions, marking a significant step toward zero-shot human motion generation. The code is available at https://github.com/VankouF/MotionMillion-Codes.
>
---
#### [new 085] X-ray transferable polyrepresentation learning
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于特征学习任务，旨在提升模型在未见数据上的泛化能力。通过整合多源表示形成多模态表示，提高性能并验证其在X-ray图像中的可迁移性。**

- **链接: [http://arxiv.org/pdf/2507.06264v1](http://arxiv.org/pdf/2507.06264v1)**

> **作者:** Weronika Hryniewska-Guzik; Przemyslaw Biecek
>
> **备注:** part of Weronika's PhD thesis
>
> **摘要:** The success of machine learning algorithms is inherently related to the extraction of meaningful features, as they play a pivotal role in the performance of these algorithms. Central to this challenge is the quality of data representation. However, the ability to generalize and extract these features effectively from unseen datasets is also crucial. In light of this, we introduce a novel concept: the polyrepresentation. Polyrepresentation integrates multiple representations of the same modality extracted from distinct sources, for example, vector embeddings from the Siamese Network, self-supervised models, and interpretable radiomic features. This approach yields better performance metrics compared to relying on a single representation. Additionally, in the context of X-ray images, we demonstrate the transferability of the created polyrepresentation to a smaller dataset, underscoring its potential as a pragmatic and resource-efficient approach in various image-related solutions. It is worth noting that the concept of polyprepresentation on the example of medical data can also be applied to other domains, showcasing its versatility and broad potential impact.
>
---
#### [new 086] SimCortex: Collision-free Simultaneous Cortical Surfaces Reconstruction
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于脑图像处理任务，旨在解决 cortical 表面重建中的重叠与自相交问题。通过深度学习框架 SimCortex 实现无碰撞的多尺度表面重建。**

- **链接: [http://arxiv.org/pdf/2507.06955v1](http://arxiv.org/pdf/2507.06955v1)**

> **作者:** Kaveh Moradkhani; R Jarrett Rushmore; Sylvain Bouix
>
> **摘要:** Accurate cortical surface reconstruction from magnetic resonance imaging (MRI) data is crucial for reliable neuroanatomical analyses. Current methods have to contend with complex cortical geometries, strict topological requirements, and often produce surfaces with overlaps, self-intersections, and topological defects. To overcome these shortcomings, we introduce SimCortex, a deep learning framework that simultaneously reconstructs all brain surfaces (left/right white-matter and pial) from T1-weighted(T1w) MRI volumes while preserving topological properties. Our method first segments the T1w image into a nine-class tissue label map. From these segmentations, we generate subject-specific, collision-free initial surface meshes. These surfaces serve as precise initializations for subsequent multiscale diffeomorphic deformations. Employing stationary velocity fields (SVFs) integrated via scaling-and-squaring, our approach ensures smooth, topology-preserving transformations with significantly reduced surface collisions and self-intersections. Evaluations on standard datasets demonstrate that SimCortex dramatically reduces surface overlaps and self-intersections, surpassing current methods while maintaining state-of-the-art geometric accuracy.
>
---
#### [new 087] Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于3D医学图像分割任务，旨在解决跨模态处理和数据变异性的挑战。提出HoME模型，通过分层软专家路由提升分割性能。**

- **链接: [http://arxiv.org/pdf/2507.06363v1](http://arxiv.org/pdf/2507.06363v1)**

> **作者:** Szymon Płotka; Maciej Chrabaszcz; Gizem Mert; Ewa Szczurek; Arkadiusz Sitek
>
> **摘要:** In recent years, artificial intelligence has significantly advanced medical image segmentation. However, challenges remain, including efficient 3D medical image processing across diverse modalities and handling data variability. In this work, we introduce Hierarchical Soft Mixture-of-Experts (HoME), a two-level token-routing layer for efficient long-context modeling, specifically designed for 3D medical image segmentation. Built on the Mamba state-space model (SSM) backbone, HoME enhances sequential modeling through sparse, adaptive expert routing. The first stage employs a Soft Mixture-of-Experts (SMoE) layer to partition input sequences into local groups, routing tokens to specialized per-group experts for localized feature extraction. The second stage aggregates these outputs via a global SMoE layer, enabling cross-group information fusion and global context refinement. This hierarchical design, combining local expert routing with global expert refinement improves generalizability and segmentation performance, surpassing state-of-the-art results across datasets from the three most commonly used 3D medical imaging modalities and data quality.
>
---
#### [new 088] Fast Equivariant Imaging: Acceleration for Unsupervised Learning via Augmented Lagrangian and Auxiliary PnP Denoisers
- **分类: eess.IV; cs.CV; cs.LG; math.OC**

- **简介: 该论文属于图像重建任务，解决无监督学习中训练深度网络的问题。通过改进的拉格朗日方法和辅助去噪器，实现更高效的训练。**

- **链接: [http://arxiv.org/pdf/2507.06764v1](http://arxiv.org/pdf/2507.06764v1)**

> **作者:** Guixian Xu; Jinglai Li; Junqi Tang
>
> **摘要:** We propose Fast Equivariant Imaging (FEI), a novel unsupervised learning framework to efficiently train deep imaging networks without ground-truth data. From the perspective of reformulating the Equivariant Imaging based optimization problem via the method of Lagrange multipliers and utilizing plug-and-play denoisers, this novel unsupervised scheme shows superior efficiency and performance compared to vanilla Equivariant Imaging paradigm. In particular, our PnP-FEI scheme achieves an order-of-magnitude (10x) acceleration over standard EI on training U-Net with CT100 dataset for X-ray CT reconstruction, with improved generalization performance.
>
---
#### [new 089] Airway Segmentation Network for Enhanced Tubular Feature Extraction
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决空气道结构自动分割中的精度与连续性问题。提出TfeNet网络，通过方向感知卷积和特征融合模块提升细小空气道的识别能力。**

- **链接: [http://arxiv.org/pdf/2507.06581v1](http://arxiv.org/pdf/2507.06581v1)**

> **作者:** Qibiao Wu; Yagang Wang; Qian Zhang
>
> **摘要:** Manual annotation of airway regions in computed tomography images is a time-consuming and expertise-dependent task. Automatic airway segmentation is therefore a prerequisite for enabling rapid bronchoscopic navigation and the clinical deployment of bronchoscopic robotic systems. Although convolutional neural network methods have gained considerable attention in airway segmentation, the unique tree-like structure of airways poses challenges for conventional and deformable convolutions, which often fail to focus on fine airway structures, leading to missed segments and discontinuities. To address this issue, this study proposes a novel tubular feature extraction network, named TfeNet. TfeNet introduces a novel direction-aware convolution operation that first applies spatial rotation transformations to adjust the sampling positions of linear convolution kernels. The deformed kernels are then represented as line segments or polylines in 3D space. Furthermore, a tubular feature fusion module (TFFM) is designed based on asymmetric convolution and residual connection strategies, enhancing the network's focus on subtle airway structures. Extensive experiments conducted on one public dataset and two datasets used in airway segmentation challenges demonstrate that the proposed TfeNet achieves more accuracy and continuous airway structure predictions compared with existing methods. In particular, TfeNet achieves the highest overall score of 94.95% on the current largest airway segmentation dataset, Airway Tree Modeling(ATM22), and demonstrates advanced performance on the lung fibrosis dataset(AIIB23). The code is available at https://github.com/QibiaoWu/TfeNet.
>
---
#### [new 090] Conformal Prediction for Long-Tailed Classification
- **分类: stat.ML; cs.CV; cs.LG; stat.ME**

- **简介: 该论文属于长尾分类任务，解决预测集在长尾分布下覆盖不足或过大的问题。提出改进的置信度函数和加权方法，平衡集大小与类条件覆盖。**

- **链接: [http://arxiv.org/pdf/2507.06867v1](http://arxiv.org/pdf/2507.06867v1)**

> **作者:** Tiffany Ding; Jean-Baptiste Fermanian; Joseph Salmon
>
> **摘要:** Many real-world classification problems, such as plant identification, have extremely long-tailed class distributions. In order for prediction sets to be useful in such settings, they should (i) provide good class-conditional coverage, ensuring that rare classes are not systematically omitted from the prediction sets, and (ii) be a reasonable size, allowing users to easily verify candidate labels. Unfortunately, existing conformal prediction methods, when applied to the long-tailed setting, force practitioners to make a binary choice between small sets with poor class-conditional coverage or sets with very good class-conditional coverage but that are extremely large. We propose methods with guaranteed marginal coverage that smoothly trade off between set size and class-conditional coverage. First, we propose a conformal score function, prevalence-adjusted softmax, that targets a relaxed notion of class-conditional coverage called macro-coverage. Second, we propose a label-weighted conformal prediction method that allows us to interpolate between marginal and class-conditional conformal prediction. We demonstrate our methods on Pl@ntNet and iNaturalist, two long-tailed image datasets with 1,081 and 8,142 classes, respectively.
>
---
#### [new 091] Capsule-ConvKAN: A Hybrid Neural Approach to Medical Image Classification
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在提升传统卷积模型的性能。通过融合胶囊网络与卷积KAN，提出Capsule-ConvKAN模型，有效捕捉空间特征并提高分类准确率。**

- **链接: [http://arxiv.org/pdf/2507.06417v1](http://arxiv.org/pdf/2507.06417v1)**

> **作者:** Laura Pituková; Peter Sinčák; László József Kovács
>
> **备注:** Preprint version. Accepted to IEEE SMC 2025
>
> **摘要:** This study conducts a comprehensive comparison of four neural network architectures: Convolutional Neural Network, Capsule Network, Convolutional Kolmogorov--Arnold Network, and the newly proposed Capsule--Convolutional Kolmogorov--Arnold Network. The proposed Capsule-ConvKAN architecture combines the dynamic routing and spatial hierarchy capabilities of Capsule Network with the flexible and interpretable function approximation of Convolutional Kolmogorov--Arnold Networks. This novel hybrid model was developed to improve feature representation and classification accuracy, particularly in challenging real-world biomedical image data. The architectures were evaluated on a histopathological image dataset, where Capsule-ConvKAN achieved the highest classification performance with an accuracy of 91.21\%. The results demonstrate the potential of the newly introduced Capsule-ConvKAN in capturing spatial patterns, managing complex features, and addressing the limitations of traditional convolutional models in medical image classification.
>
---
#### [new 092] LOVON: Legged Open-Vocabulary Object Navigator
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人长距离目标导航任务，解决开放词汇下物体检测与任务规划整合难题，提出LOVON框架提升导航能力。**

- **链接: [http://arxiv.org/pdf/2507.06747v1](http://arxiv.org/pdf/2507.06747v1)**

> **作者:** Daojie Peng; Jiahang Cao; Qiang Zhang; Jun Ma
>
> **备注:** 9 pages, 10 figures; Project Page: https://daojiepeng.github.io/LOVON/
>
> **摘要:** Object navigation in open-world environments remains a formidable and pervasive challenge for robotic systems, particularly when it comes to executing long-horizon tasks that require both open-world object detection and high-level task planning. Traditional methods often struggle to integrate these components effectively, and this limits their capability to deal with complex, long-range navigation missions. In this paper, we propose LOVON, a novel framework that integrates large language models (LLMs) for hierarchical task planning with open-vocabulary visual detection models, tailored for effective long-range object navigation in dynamic, unstructured environments. To tackle real-world challenges including visual jittering, blind zones, and temporary target loss, we design dedicated solutions such as Laplacian Variance Filtering for visual stabilization. We also develop a functional execution logic for the robot that guarantees LOVON's capabilities in autonomous navigation, task adaptation, and robust task completion. Extensive evaluations demonstrate the successful completion of long-sequence tasks involving real-time detection, search, and navigation toward open-vocabulary dynamic targets. Furthermore, real-world experiments across different legged robots (Unitree Go2, B2, and H1-2) showcase the compatibility and appealing plug-and-play feature of LOVON.
>
---
#### [new 093] Denoising Multi-Beta VAE: Representation Learning for Disentanglement and Generation
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于生成模型任务，旨在解决 disentanglement 与生成质量的权衡问题。通过多 β 值 VAE 和扩散模型，学习多组潜在表示并实现高质量生成。**

- **链接: [http://arxiv.org/pdf/2507.06613v1](http://arxiv.org/pdf/2507.06613v1)**

> **作者:** Anshuk Uppal; Yuhta Takida; Chieh-Hsin Lai; Yuki Mitsufuji
>
> **备注:** 24 pages, 8 figures and 7 tables
>
> **摘要:** Disentangled and interpretable latent representations in generative models typically come at the cost of generation quality. The $\beta$-VAE framework introduces a hyperparameter $\beta$ to balance disentanglement and reconstruction quality, where setting $\beta > 1$ introduces an information bottleneck that favors disentanglement over sharp, accurate reconstructions. To address this trade-off, we propose a novel generative modeling framework that leverages a range of $\beta$ values to learn multiple corresponding latent representations. First, we obtain a slew of representations by training a single variational autoencoder (VAE), with a new loss function that controls the information retained in each latent representation such that the higher $\beta$ value prioritize disentanglement over reconstruction fidelity. We then, introduce a non-linear diffusion model that smoothly transitions latent representations corresponding to different $\beta$ values. This model denoises towards less disentangled and more informative representations, ultimately leading to (almost) lossless representations, enabling sharp reconstructions. Furthermore, our model supports sample generation without input images, functioning as a standalone generative model. We evaluate our framework in terms of both disentanglement and generation quality. Additionally, we observe smooth transitions in the latent spaces with respect to changes in $\beta$, facilitating consistent manipulation of generated outputs.
>
---
#### [new 094] PAST: A multimodal single-cell foundation model for histopathology and spatial transcriptomics in cancer
- **分类: q-bio.QM; cs.CV; stat.AP**

- **简介: 该论文属于癌症病理分析任务，旨在解决单细胞分子数据与影像整合不足的问题。工作是构建PAST模型，联合编码细胞形态与基因表达，实现精准预测与分析。**

- **链接: [http://arxiv.org/pdf/2507.06418v1](http://arxiv.org/pdf/2507.06418v1)**

> **作者:** Changchun Yang; Haoyang Li; Yushuai Wu; Yilan Zhang; Yifeng Jiao; Yu Zhang; Rihan Huang; Yuan Cheng; Yuan Qi; Xin Guo; Xin Gao
>
> **摘要:** While pathology foundation models have transformed cancer image analysis, they often lack integration with molecular data at single-cell resolution, limiting their utility for precision oncology. Here, we present PAST, a pan-cancer single-cell foundation model trained on 20 million paired histopathology images and single-cell transcriptomes spanning multiple tumor types and tissue contexts. By jointly encoding cellular morphology and gene expression, PAST learns unified cross-modal representations that capture both spatial and molecular heterogeneity at the cellular level. This approach enables accurate prediction of single-cell gene expression, virtual molecular staining, and multimodal survival analysis directly from routine pathology slides. Across diverse cancers and downstream tasks, PAST consistently exceeds the performance of existing approaches, demonstrating robust generalizability and scalability. Our work establishes a new paradigm for pathology foundation models, providing a versatile tool for high-resolution spatial omics, mechanistic discovery, and precision cancer research.
>
---
#### [new 095] Mitigating Multi-Sequence 3D Prostate MRI Data Scarcity through Domain Adaptation using Locally-Trained Latent Diffusion Models for Prostate Cancer Detection
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像处理任务，旨在解决3D前列腺MRI数据不足的问题。通过改进的扩散模型生成多序列MRI数据，提升分类器性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.06384v1](http://arxiv.org/pdf/2507.06384v1)**

> **作者:** Emerson P. Grabke; Babak Taati; Masoom A. Haider
>
> **备注:** BT and MAH are co-senior authors on the work. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Objective: Latent diffusion models (LDMs) could mitigate data scarcity challenges affecting machine learning development for medical image interpretation. The recent CCELLA LDM improved prostate cancer detection performance using synthetic MRI for classifier training but was limited to the axial T2-weighted (AxT2) sequence, did not investigate inter-institutional domain shift, and prioritized radiology over histopathology outcomes. We propose CCELLA++ to address these limitations and improve clinical utility. Methods: CCELLA++ expands CCELLA for simultaneous biparametric prostate MRI (bpMRI) generation, including the AxT2, high b-value diffusion series (HighB) and apparent diffusion coefficient map (ADC). Domain adaptation was investigated by pretraining classifiers on real or LDM-generated synthetic data from an internal institution, followed with fine-tuning on progressively smaller fractions of an out-of-distribution, external dataset. Results: CCELLA++ improved 3D FID for HighB and ADC but not AxT2 (0.013, 0.012, 0.063 respectively) sequences compared to CCELLA (0.060). Classifier pretraining with CCELLA++ bpMRI outperformed real bpMRI in AP and AUC for all domain adaptation scenarios. CCELLA++ pretraining achieved highest classifier performance below 50% (n=665) external dataset volume. Conclusion: Synthetic bpMRI generated by our method can improve downstream classifier generalization and performance beyond real bpMRI or CCELLA-generated AxT2-only images. Future work should seek to quantify medical image sample quality, balance multi-sequence LDM training, and condition the LDM with additional information. Significance: The proposed CCELLA++ LDM can generate synthetic bpMRI that outperforms real data for domain adaptation with a limited target institution dataset. Our code is available at https://github.com/grabkeem/CCELLA-plus-plus
>
---
#### [new 096] Addressing Imbalanced Domain-Incremental Learning through Dual-Balance Collaborative Experts
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于领域增量学习任务，解决数据不平衡带来的类内分布不均和跨域分布变化问题，提出DCE框架提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.07100v1](http://arxiv.org/pdf/2507.07100v1)**

> **作者:** Lan Li; Da-Wei Zhou; Han-Jia Ye; De-Chuan Zhan
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Domain-Incremental Learning (DIL) focuses on continual learning in non-stationary environments, requiring models to adjust to evolving domains while preserving historical knowledge. DIL faces two critical challenges in the context of imbalanced data: intra-domain class imbalance and cross-domain class distribution shifts. These challenges significantly hinder model performance, as intra-domain imbalance leads to underfitting of few-shot classes, while cross-domain shifts require maintaining well-learned many-shot classes and transferring knowledge to improve few-shot class performance in old domains. To overcome these challenges, we introduce the Dual-Balance Collaborative Experts (DCE) framework. DCE employs a frequency-aware expert group, where each expert is guided by specialized loss functions to learn features for specific frequency groups, effectively addressing intra-domain class imbalance. Subsequently, a dynamic expert selector is learned by synthesizing pseudo-features through balanced Gaussian sampling from historical class statistics. This mechanism navigates the trade-off between preserving many-shot knowledge of previous domains and leveraging new data to improve few-shot class performance in earlier tasks. Extensive experimental results on four benchmark datasets demonstrate DCE's state-of-the-art performance.
>
---
#### [new 097] 3D-Generalist: Self-Improving Vision-Language-Action Models for Crafting 3D Worlds
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于多模态AI任务，旨在解决3D环境生成难题。通过VLMs自动生成高质量3D世界，提升模型空间推理能力。**

- **链接: [http://arxiv.org/pdf/2507.06484v1](http://arxiv.org/pdf/2507.06484v1)**

> **作者:** Fan-Yun Sun; Shengguang Wu; Christian Jacobsen; Thomas Yim; Haoming Zou; Alex Zook; Shangru Li; Yu-Hsin Chou; Ethem Can; Xunlei Wu; Clemens Eppner; Valts Blukis; Jonathan Tremblay; Jiajun Wu; Stan Birchfield; Nick Haber
>
> **备注:** project website: https://ai.stanford.edu/~sunfanyun/3d-generalist/
>
> **摘要:** Despite large-scale pretraining endowing models with language and vision reasoning capabilities, improving their spatial reasoning capability remains challenging due to the lack of data grounded in the 3D world. While it is possible for humans to manually create immersive and interactive worlds through 3D graphics, as seen in applications such as VR, gaming, and robotics, this process remains highly labor-intensive. In this paper, we propose a scalable method for generating high-quality 3D environments that can serve as training data for foundation models. We recast 3D environment building as a sequential decision-making problem, employing Vision-Language-Models (VLMs) as policies that output actions to jointly craft a 3D environment's layout, materials, lighting, and assets. Our proposed framework, 3D-Generalist, trains VLMs to generate more prompt-aligned 3D environments via self-improvement fine-tuning. We demonstrate the effectiveness of 3D-Generalist and the proposed training strategy in generating simulation-ready 3D environments. Furthermore, we demonstrate its quality and scalability in synthetic data generation by pretraining a vision foundation model on the generated data. After fine-tuning the pre-trained model on downstream tasks, we show that it surpasses models pre-trained on meticulously human-crafted synthetic data and approaches results achieved with real data orders of magnitude larger.
>
---
#### [new 098] The User-Centric Geo-Experience: An LLM-Powered Framework for Enhanced Planning, Navigation, and Dynamic Adaptation
- **分类: cs.AI; cs.CV**

- **简介: 该论文属于旅行规划任务，解决传统系统在动态适应、精准导航和智能规划方面的不足。提出三个协作代理提升行程规划、最后一公里导航和应对突发情况的能力。**

- **链接: [http://arxiv.org/pdf/2507.06993v1](http://arxiv.org/pdf/2507.06993v1)**

> **作者:** Jieren Deng; Aleksandar Cvetkovic; Pak Kiu Chung; Dragomir Yankov; Chiqun Zhang
>
> **摘要:** Traditional travel-planning systems are often static and fragmented, leaving them ill-equipped to handle real-world complexities such as evolving environmental conditions and unexpected itinerary disruptions. In this paper, we identify three gaps between existing service providers causing frustrating user experience: intelligent trip planning, precision "last-100-meter" navigation, and dynamic itinerary adaptation. We propose three cooperative agents: a Travel Planning Agent that employs grid-based spatial grounding and map analysis to help resolve complex multi-modal user queries; a Destination Assistant Agent that provides fine-grained guidance for the final navigation leg of each journey; and a Local Discovery Agent that leverages image embeddings and Retrieval-Augmented Generation (RAG) to detect and respond to trip plan disruptions. With evaluations and experiments, our system demonstrates substantial improvements in query interpretation, navigation accuracy, and disruption resilience, underscoring its promise for applications from urban exploration to emergency response.
>
---
#### [new 099] Deep Brain Net: An Optimized Deep Learning Model for Brain tumor Detection in MRI Images Using EfficientNetB0 and ResNet50 with Transfer Learning
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在提高脑肿瘤MRI检测的准确性和效率。通过融合EfficientNetB0和ResNet50，并应用迁移学习，提出Deep Brain Net模型，实现高精度分类。**

- **链接: [http://arxiv.org/pdf/2507.07011v1](http://arxiv.org/pdf/2507.07011v1)**

> **作者:** Daniel Onah; Ravish Desai
>
> **备注:** 9 pages, 14 figures, 4 tables. To be submitted to a conference
>
> **摘要:** In recent years, deep learning has shown great promise in the automated detection and classification of brain tumors from MRI images. However, achieving high accuracy and computational efficiency remains a challenge. In this research, we propose Deep Brain Net, a novel deep learning system designed to optimize performance in the detection of brain tumors. The model integrates the strengths of two advanced neural network architectures which are EfficientNetB0 and ResNet50, combined with transfer learning to improve generalization and reduce training time. The EfficientNetB0 architecture enhances model efficiency by utilizing mobile inverted bottleneck blocks, which incorporate depth wise separable convolutions. This design significantly reduces the number of parameters and computational cost while preserving the ability of models to learn complex feature representations. The ResNet50 architecture, pre trained on large scale datasets like ImageNet, is fine tuned for brain tumor classification. Its use of residual connections allows for training deeper networks by mitigating the vanishing gradient problem and avoiding performance degradation. The integration of these components ensures that the proposed system is both computationally efficient and highly accurate. Extensive experiments performed on publicly available MRI datasets demonstrate that Deep Brain Net consistently outperforms existing state of the art methods in terms of classification accuracy, precision, recall, and computational efficiency. The result is an accuracy of 88 percent, a weighted F1 score of 88.75 percent, and a macro AUC ROC score of 98.17 percent which demonstrates the robustness and clinical potential of Deep Brain Net in assisting radiologists with brain tumor diagnosis.
>
---
#### [new 100] Enhancing non-Rigid 3D Model Deformations Using Mesh-based Gaussian Splatting
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于3D模型变形任务，解决传统Gaussian splatting在非刚性变形和编辑上的不足。通过将高斯核嵌入网格表面，实现更灵活的变形操作。**

- **链接: [http://arxiv.org/pdf/2507.07000v1](http://arxiv.org/pdf/2507.07000v1)**

> **作者:** Wijayathunga W. M. R. D. B
>
> **摘要:** We propose a novel framework that enhances non-rigid 3D model deformations by bridging mesh representations with 3D Gaussian splatting. While traditional Gaussian splatting delivers fast, real-time radiance-field rendering, its post-editing capabilities and support for large-scale, non-rigid deformations remain limited. Our method addresses these challenges by embedding Gaussian kernels directly onto explicit mesh surfaces. This allows the mesh's inherent topological and geometric priors to guide intuitive editing operations -- such as moving, scaling, and rotating individual 3D components -- and enables complex deformations like bending and stretching. This work paves the way for more flexible 3D content-creation workflows in applications spanning virtual reality, character animation, and interactive design.
>
---
#### [new 101] Secure and Storage-Efficient Deep Learning Models for Edge AI Using Automatic Weight Generation
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文属于边缘AI任务，解决深度学习模型存储效率问题。通过动态生成权重和压缩敏感层，显著降低内存占用并保持精度。**

- **链接: [http://arxiv.org/pdf/2507.06380v1](http://arxiv.org/pdf/2507.06380v1)**

> **作者:** Habibur Rahaman; Atri Chatterjee; Swarup Bhunia
>
> **备注:** 7 pages, 7 figures
>
> **摘要:** Complex neural networks require substantial memory to store a large number of synaptic weights. This work introduces WINGs (Automatic Weight Generator for Secure and Storage-Efficient Deep Learning Models), a novel framework that dynamically generates layer weights in a fully connected neural network (FC) and compresses the weights in convolutional neural networks (CNNs) during inference, significantly reducing memory requirements without sacrificing accuracy. WINGs framework uses principal component analysis (PCA) for dimensionality reduction and lightweight support vector regression (SVR) models to predict layer weights in the FC networks, removing the need for storing full-weight matrices and achieving substantial memory savings. It also preferentially compresses the weights in low-sensitivity layers of CNNs using PCA and SVR with sensitivity analysis. The sensitivity-aware design also offers an added level of security, as any bit-flip attack with weights in compressed layers has an amplified and readily detectable effect on accuracy. WINGs achieves 53x compression for the FC layers and 28x for AlexNet with MNIST dataset, and 18x for Alexnet with CIFAR-10 dataset with 1-2% accuracy loss. This significant reduction in memory results in higher throughput and lower energy for DNN inference, making it attractive for resource-constrained edge applications.
>
---
#### [new 102] Attention-Enhanced Deep Learning Ensemble for Breast Density Classification in Mammography
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于乳腺密度分类任务，旨在解决高密度乳腺癌风险评估与检测难题。通过集成深度学习模型和改进损失函数，提升分类准确率。**

- **链接: [http://arxiv.org/pdf/2507.06410v1](http://arxiv.org/pdf/2507.06410v1)**

> **作者:** Peyman Sharifian; Xiaotong Hong; Alireza Karimian; Mehdi Amini; Hossein Arabi
>
> **备注:** 2025 IEEE Nuclear Science Symposium, Medical Imaging Conference and Room Temperature Semiconductor Detector Conference
>
> **摘要:** Breast density assessment is a crucial component of mammographic interpretation, with high breast density (BI-RADS categories C and D) representing both a significant risk factor for developing breast cancer and a technical challenge for tumor detection. This study proposes an automated deep learning system for robust binary classification of breast density (low: A/B vs. high: C/D) using the VinDr-Mammo dataset. We implemented and compared four advanced convolutional neural networks: ResNet18, ResNet50, EfficientNet-B0, and DenseNet121, each enhanced with channel attention mechanisms. To address the inherent class imbalance, we developed a novel Combined Focal Label Smoothing Loss function that integrates focal loss, label smoothing, and class-balanced weighting. Our preprocessing pipeline incorporated advanced techniques, including contrast-limited adaptive histogram equalization (CLAHE) and comprehensive data augmentation. The individual models were combined through an optimized ensemble voting approach, achieving superior performance (AUC: 0.963, F1-score: 0.952) compared to any single model. This system demonstrates significant potential to standardize density assessments in clinical practice, potentially improving screening efficiency and early cancer detection rates while reducing inter-observer variability among radiologists.
>
---
#### [new 103] Learning to Evaluate Autonomous Behaviour in Human-Robot Interaction
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于人机交互任务，解决自主机器人评估难题。提出NeME模型，通过轨迹分析比较模仿学习方法性能，实现无需人工参与的自动化评估。**

- **链接: [http://arxiv.org/pdf/2507.06404v1](http://arxiv.org/pdf/2507.06404v1)**

> **作者:** Matteo Tiezzi; Tommaso Apicella; Carlos Cardenas-Perez; Giovanni Fregonese; Stefano Dafarra; Pietro Morerio; Daniele Pucci; Alessio Del Bue
>
> **摘要:** Evaluating and comparing the performance of autonomous Humanoid Robots is challenging, as success rate metrics are difficult to reproduce and fail to capture the complexity of robot movement trajectories, critical in Human-Robot Interaction and Collaboration (HRIC). To address these challenges, we propose a general evaluation framework that measures the quality of Imitation Learning (IL) methods by focusing on trajectory performance. We devise the Neural Meta Evaluator (NeME), a deep learning model trained to classify actions from robot joint trajectories. NeME serves as a meta-evaluator to compare the performance of robot control policies, enabling policy evaluation without requiring human involvement in the loop. We validate our framework on ergoCub, a humanoid robot, using teleoperation data and comparing IL methods tailored to the available platform. The experimental results indicate that our method is more aligned with the success rate obtained on the robot than baselines, offering a reproducible, systematic, and insightful means for comparing the performance of multimodal imitation learning approaches in complex HRI tasks.
>
---
#### [new 104] Speckle2Self: Self-Supervised Ultrasound Speckle Reduction Without Clean Data
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于医学图像去噪任务，解决超声图像中组织依赖的散斑噪声问题。通过自监督方法，仅用单张噪声图像实现有效降噪。**

- **链接: [http://arxiv.org/pdf/2507.06828v1](http://arxiv.org/pdf/2507.06828v1)**

> **作者:** Xuesong Li; Nassir Navab; Zhongliang Jiang
>
> **摘要:** Image denoising is a fundamental task in computer vision, particularly in medical ultrasound (US) imaging, where speckle noise significantly degrades image quality. Although recent advancements in deep neural networks have led to substantial improvements in denoising for natural images, these methods cannot be directly applied to US speckle noise, as it is not purely random. Instead, US speckle arises from complex wave interference within the body microstructure, making it tissue-dependent. This dependency means that obtaining two independent noisy observations of the same scene, as required by pioneering Noise2Noise, is not feasible. Additionally, blind-spot networks also cannot handle US speckle noise due to its high spatial dependency. To address this challenge, we introduce Speckle2Self, a novel self-supervised algorithm for speckle reduction using only single noisy observations. The key insight is that applying a multi-scale perturbation (MSP) operation introduces tissue-dependent variations in the speckle pattern across different scales, while preserving the shared anatomical structure. This enables effective speckle suppression by modeling the clean image as a low-rank signal and isolating the sparse noise component. To demonstrate its effectiveness, Speckle2Self is comprehensively compared with conventional filter-based denoising algorithms and SOTA learning-based methods, using both realistic simulated US images and human carotid US images. Additionally, data from multiple US machines are employed to evaluate model generalization and adaptability to images from unseen domains. \textit{Code and datasets will be released upon acceptance.
>
---
#### [new 105] A Principled Framework for Multi-View Contrastive Learning
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于自监督学习任务，解决多视图对比学习中的优化冲突与交互不足问题，提出两种新损失函数提升模型性能与扩展性。**

- **链接: [http://arxiv.org/pdf/2507.06979v1](http://arxiv.org/pdf/2507.06979v1)**

> **作者:** Panagiotis Koromilas; Efthymios Georgiou; Giorgos Bouritsas; Theodoros Giannakopoulos; Mihalis A. Nicolaou; Yannis Panagakis
>
> **摘要:** Contrastive Learning (CL), a leading paradigm in Self-Supervised Learning (SSL), typically relies on pairs of data views generated through augmentation. While multiple augmentations per instance (more than two) improve generalization in supervised learning, current CL methods handle additional views suboptimally by simply aggregating different pairwise objectives. This approach suffers from four critical limitations: (L1) it utilizes multiple optimization terms per data point resulting to conflicting objectives, (L2) it fails to model all interactions across views and data points, (L3) it inherits fundamental limitations (e.g. alignment-uniformity coupling) from pairwise CL losses, and (L4) it prevents fully realizing the benefits of increased view multiplicity observed in supervised settings. We address these limitations through two novel loss functions: MV-InfoNCE, which extends InfoNCE to incorporate all possible view interactions simultaneously in one term per data point, and MV-DHEL, which decouples alignment from uniformity across views while scaling interaction complexity with view multiplicity. Both approaches are theoretically grounded - we prove they asymptotically optimize for alignment of all views and uniformity, providing principled extensions to multi-view contrastive learning. Our empirical results on ImageNet1K and three other datasets demonstrate that our methods consistently outperform existing multi-view approaches and effectively scale with increasing view multiplicity. We also apply our objectives to multimodal data and show that, in contrast to other contrastive objectives, they can scale beyond just two modalities. Most significantly, ablation studies reveal that MV-DHEL with five or more views effectively mitigates dimensionality collapse by fully utilizing the embedding space, thereby delivering multi-view benefits observed in supervised learning.
>
---
## 更新

#### [replaced 001] Integrated Structural Prompt Learning for Vision-Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.05677v2](http://arxiv.org/pdf/2507.05677v2)**

> **作者:** Jiahui Wang; Qin Xu; Bo Jiang; Bin Luo
>
> **摘要:** Prompt learning methods have significantly extended the transferability of pre-trained Vision-Language Models (VLMs) like CLIP for various downstream tasks. These methods adopt handcraft templates or learnable vectors to provide text or image instructions in fine-tuning VLMs. However, most existing works ignore the structural relationships between learnable prompts and tokens within and between modalities. Moreover, balancing the performance of base and new classes remains a significant challenge. In this paper, we propose an Integrated Structural Prompt (ISP) for VLMs to enhance the interaction of information representations between the text and image branches. ISP introduces self-structural and cross-structural prompt modules to model the structural relationships between learnable prompts and frozen tokens within and across modalities. This enables efficient information transfer while preserving feature stability. Additionally, we propose a sample probing module that dynamically adjusts loss coefficients based on sample difficulty, preventing the mode from overfitting to simple samples and improving generalization ability to new classes. Extensive experiments on three widely used settings: base-to-new generalization, cross-dataset evaluation, and domain generalization demonstrate that the proposed ISP achieves competitive performance against state-of-the-art methods.
>
---
#### [replaced 002] ROVER: A Multi-Season Dataset for Visual SLAM
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.02506v3](http://arxiv.org/pdf/2412.02506v3)**

> **作者:** Fabian Schmidt; Julian Daubermann; Marcel Mitschke; Constantin Blessing; Stefan Meyer; Markus Enzweiler; Abhinav Valada
>
> **备注:** Copyright 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Robust SLAM is a crucial enabler for autonomous navigation in natural, semi-structured environments such as parks and gardens. However, these environments present unique challenges for SLAM due to frequent seasonal changes, varying light conditions, and dense vegetation. These factors often degrade the performance of visual SLAM algorithms originally developed for structured urban environments. To address this gap, we present ROVER, a comprehensive benchmark dataset tailored for evaluating visual SLAM algorithms under diverse environmental conditions and spatial configurations. We captured the dataset with a robotic platform equipped with monocular, stereo, and RGBD cameras, as well as inertial sensors. It covers 39 recordings across five outdoor locations, collected through all seasons and various lighting scenarios, i.e., day, dusk, and night with and without external lighting. With this novel dataset, we evaluate several traditional and deep learning-based SLAM methods and study their performance in diverse challenging conditions. The results demonstrate that while stereo-inertial and RGBD configurations generally perform better under favorable lighting and moderate vegetation, most SLAM systems perform poorly in low-light and high-vegetation scenarios, particularly during summer and autumn. Our analysis highlights the need for improved adaptability in visual SLAM algorithms for outdoor applications, as current systems struggle with dynamic environmental factors affecting scale, feature extraction, and trajectory consistency. This dataset provides a solid foundation for advancing visual SLAM research in real-world, semi-structured environments, fostering the development of more resilient SLAM systems for long-term outdoor localization and mapping. The dataset and the code of the benchmark are available under https://iis-esslingen.github.io/rover.
>
---
#### [replaced 003] Signal-SGN: A Spiking Graph Convolutional Network for Skeletal Action Recognition via Learning Temporal-Frequency Dynamics
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.01701v5](http://arxiv.org/pdf/2408.01701v5)**

> **作者:** Naichuan Zheng; Yuchen Du; Hailun Xia; Zeyu Liang
>
> **摘要:** For multimodal skeleton-based action recognition, Graph Convolutional Networks (GCNs) are effective models. Still, their reliance on floating-point computations leads to high energy consumption, limiting their applicability in battery-powered devices. While energy-efficient, Spiking Neural Networks (SNNs) struggle to model skeleton dynamics, leading to suboptimal solutions. We propose Signal-SGN (Spiking Graph Convolutional Network), which utilizes the temporal dimension of skeleton sequences as the spike time steps and represents features as multi-dimensional discrete stochastic signals for temporal-frequency domain feature extraction. It combines the 1D Spiking Graph Convolution (1D-SGC) module and the Frequency Spiking Convolution (FSC) module to extract features from the skeleton represented as spiking form. Additionally, the Multi-Scale Wavelet Transform Feature Fusion (MWTF) module is proposed to extract dynamic spiking features and capture frequency-specific characteristics, enhancing classification performance. Experiments across three large-scale datasets reveal Signal-SGN exceeding state-of-the-art SNN-based methods in accuracy and computational efficiency while attaining comparable performance with GCN methods and significantly reducing theoretical energy consumption.
>
---
#### [replaced 004] GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.12981v2](http://arxiv.org/pdf/2411.12981v2)**

> **作者:** Xiaobao Wei; Peng Chen; Guangyu Li; Ming Lu; Hui Chen; Feng Tian
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Gaze estimation encounters generalization challenges when dealing with out-of-distribution data. To address this problem, recent methods use neural radiance fields (NeRF) to generate augmented data. However, existing methods based on NeRF are computationally expensive and lack facial details. 3D Gaussian Splatting (3DGS) has become the prevailing representation of neural fields. While 3DGS has been extensively examined in head avatars, it faces challenges with accurate gaze control and generalization across different subjects. In this work, we propose GazeGaussian, the first high-fidelity gaze redirection method that uses a two-stream 3DGS model to represent the face and eye regions separately. Leveraging the unstructured nature of 3DGS, we develop a novel representation of the eye for rigid eye rotation based on the target gaze direction. To enable synthesis generalization across various subjects, we integrate an expression-guided module to inject subject-specific information into the neural renderer. Comprehensive experiments show that GazeGaussian outperforms existing methods in rendering speed, gaze redirection accuracy, and facial synthesis across multiple datasets. The code is available at: https://ucwxb.github.io/GazeGaussian.
>
---
#### [replaced 005] Leveraging Local Patch Alignment to Seam-cutting for Large Parallax Image Stitching
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2311.18564v3](http://arxiv.org/pdf/2311.18564v3)**

> **作者:** Tianli Liao; Chenyang Zhao; Lei Li; Heling Cao
>
> **备注:** ICCV 2025
>
> **摘要:** Seam cutting has shown significant effectiveness in the composition phase of image stitching, particularly for scenarios involving parallax. However, conventional implementations typically position seam-cutting as a downstream process contingent upon successful image alignment. This approach inherently assumes the existence of locally aligned regions where visually plausible seams can be established. Current alignment methods frequently fail to satisfy this prerequisite in large parallax scenarios despite considerable research efforts dedicated to improving alignment accuracy. In this paper, we propose an alignment-compensation paradigm that dissociates seam quality from initial alignment accuracy by integrating a Local Patch Alignment Module (LPAM) into the seam-cutting pipeline. Concretely, given the aligned images with an estimated initial seam, our method first identifies low-quality pixels along the seam through a seam quality assessment, then performs localized SIFT-flow alignment on the critical patches enclosing these pixels. Finally, we recomposite the aligned patches using adaptive seam-cutting and merge them into the original aligned images to generate the final mosaic. Comprehensive experiments on large parallax stitching datasets demonstrate that LPAM significantly enhances stitching quality while maintaining computational efficiency. The code is available at https://github.com/tlliao/LPAM_seam-cutting.
>
---
#### [replaced 006] Tora2: Motion and Appearance Customized Diffusion Transformer for Multi-Entity Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.05963v2](http://arxiv.org/pdf/2507.05963v2)**

> **作者:** Zhenghao Zhang; Junchao Liao; Xiangyu Meng; Long Qin; Weizhi Wang
>
> **备注:** ACM MM25 Conference Proceedings
>
> **摘要:** Recent advances in diffusion transformer models for motion-guided video generation, such as Tora, have shown significant progress. In this paper, we present Tora2, an enhanced version of Tora, which introduces several design improvements to expand its capabilities in both appearance and motion customization. Specifically, we introduce a decoupled personalization extractor that generates comprehensive personalization embeddings for multiple open-set entities, better preserving fine-grained visual details compared to previous methods. Building on this, we design a gated self-attention mechanism to integrate trajectory, textual description, and visual information for each entity. This innovation significantly reduces misalignment in multimodal conditioning during training. Moreover, we introduce a contrastive loss that jointly optimizes trajectory dynamics and entity consistency through explicit mapping between motion and personalization embeddings. Tora2 is, to our best knowledge, the first method to achieve simultaneous multi-entity customization of appearance and motion for video generation. Experimental results demonstrate that Tora2 achieves competitive performance with state-of-the-art customization methods while providing advanced motion control capabilities, which marks a critical advancement in multi-condition video generation. Project page: https://ali-videoai.github.io/Tora2_page/.
>
---
#### [replaced 007] Self-Calibrated Variance-Stabilizing Transformations for Real-World Image Denoising
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2407.17399v2](http://arxiv.org/pdf/2407.17399v2)**

> **作者:** Sébastien Herbreteau; Michael Unser
>
> **备注:** Accepted at IEEE/CVF International Conference on Computer Vision (ICCV) 2025
>
> **摘要:** Supervised deep learning has become the method of choice for image denoising. It involves the training of neural networks on large datasets composed of pairs of noisy and clean images. However, the necessity of training data that are specific to the targeted application constrains the widespread use of denoising networks. Recently, several approaches have been developed to overcome this difficulty by whether artificially generating realistic clean/noisy image pairs, or training exclusively on noisy images. In this paper, we show that, contrary to popular belief, denoising networks specialized in the removal of Gaussian noise can be efficiently leveraged in favor of real-world image denoising, even without additional training. For this to happen, an appropriate variance-stabilizing transform (VST) has to be applied beforehand. We propose an algorithm termed Noise2VST for the learning of such a model-free VST. Our approach requires only the input noisy image and an off-the-shelf Gaussian denoiser. We demonstrate through extensive experiments the efficiency and superiority of Noise2VST in comparison to existing methods trained in the absence of specific clean/noisy pairs.
>
---
#### [replaced 008] Rethinking Diffusion for Text-Driven Human Motion Generation: Redundant Representations, Evaluation, and Masked Autoregression
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.16575v2](http://arxiv.org/pdf/2411.16575v2)**

> **作者:** Zichong Meng; Yiming Xie; Xiaogang Peng; Zeyu Han; Huaizu Jiang
>
> **备注:** CVPR 2025
>
> **摘要:** Since 2023, Vector Quantization (VQ)-based discrete generation methods have rapidly dominated human motion generation, primarily surpassing diffusion-based continuous generation methods in standard performance metrics. However, VQ-based methods have inherent limitations. Representing continuous motion data as limited discrete tokens leads to inevitable information loss, reduces the diversity of generated motions, and restricts their ability to function effectively as motion priors or generation guidance. In contrast, the continuous space generation nature of diffusion-based methods makes them well-suited to address these limitations and with even potential for model scalability. In this work, we systematically investigate why current VQ-based methods perform well and explore the limitations of existing diffusion-based methods from the perspective of motion data representation and distribution. Drawing on these insights, we preserve the inherent strengths of a diffusion-based human motion generation model and gradually optimize it with inspiration from VQ-based approaches. Our approach introduces a human motion diffusion model enabled to perform masked autoregression, optimized with a reformed data representation and distribution. Additionally, we propose a more robust evaluation method to assess different approaches. Extensive experiments on various datasets demonstrate our method outperforms previous methods and achieves state-of-the-art performances.
>
---
#### [replaced 009] CAVIS: Context-Aware Video Instance Segmentation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.03010v2](http://arxiv.org/pdf/2407.03010v2)**

> **作者:** Seunghun Lee; Jiwan Seo; Kiljoon Han; Minwoo Choi; Sunghoon Im
>
> **备注:** ICCV 2025. Code: https://github.com/Seung-Hun-Lee/CAVIS
>
> **摘要:** In this paper, we introduce the Context-Aware Video Instance Segmentation (CAVIS), a novel framework designed to enhance instance association by integrating contextual information adjacent to each object. To efficiently extract and leverage this information, we propose the Context-Aware Instance Tracker (CAIT), which merges contextual data surrounding the instances with the core instance features to improve tracking accuracy. Additionally, we design the Prototypical Cross-frame Contrastive (PCC) loss, which ensures consistency in object-level features across frames, thereby significantly enhancing matching accuracy. CAVIS demonstrates superior performance over state-of-the-art methods on all benchmark datasets in video instance segmentation (VIS) and video panoptic segmentation (VPS). Notably, our method excels on the OVIS dataset, known for its particularly challenging videos. Project page: https://seung-hun-lee.github.io/projects/CAVIS/
>
---
#### [replaced 010] Counting Stacked Objects
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.19149v3](http://arxiv.org/pdf/2411.19149v3)**

> **作者:** Corentin Dumery; Noa Etté; Aoxiang Fan; Ren Li; Jingyi Xu; Hieu Le; Pascal Fua
>
> **备注:** ICCV25. Datasets and code can be found at https://corentindumery.github.io/projects/stacks.html
>
> **摘要:** Visual object counting is a fundamental computer vision task underpinning numerous real-world applications, from cell counting in biomedicine to traffic and wildlife monitoring. However, existing methods struggle to handle the challenge of stacked 3D objects in which most objects are hidden by those above them. To address this important yet underexplored problem, we propose a novel 3D counting approach that decomposes the task into two complementary subproblems - estimating the 3D geometry of the object stack and the occupancy ratio from multi-view images. By combining geometric reconstruction and deep learning-based depth analysis, our method can accurately count identical objects within containers, even when they are irregularly stacked. We validate our 3D Counting pipeline on diverse real-world and large-scale synthetic datasets, which we will release publicly to facilitate further research.
>
---
#### [replaced 011] Many-Task Federated Fine-Tuning via Unified Task Vectors
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.06376v3](http://arxiv.org/pdf/2502.06376v3)**

> **作者:** Vasileios Tsouvalas; Tanir Ozcelebi; Nirvana Meratnia
>
> **备注:** 10 pages, 6 figures, accepted in FedGenAI-IJCAI 2025
>
> **摘要:** Federated Learning (FL) traditionally assumes homogeneous client tasks; however, in real-world scenarios, clients often specialize in diverse tasks, introducing task heterogeneity. To address this challenge, Many-Task FL (MaT-FL) has emerged, enabling clients to collaborate effectively despite task diversity. Existing MaT-FL approaches rely on client grouping or personalized layers, requiring the server to manage individual models and failing to account for clients handling multiple tasks. We propose MaTU, a MaT-FL approach that enables joint learning of task vectors across clients, eliminating the need for clustering or client-specific weight storage at the server. Our method introduces a novel aggregation mechanism that determines task similarity based on the direction of clients task vectors and constructs a unified task vector encapsulating all tasks. To address task-specific requirements, we augment the unified task vector with lightweight modulators that facilitate knowledge transfer among related tasks while disentangling dissimilar ones. Evaluated across 30 datasets, MaTU achieves superior performance over state-of-the-art MaT-FL approaches, with results comparable to per-task fine-tuning, while delivering significant communication savings.
>
---
#### [replaced 012] Modality-agnostic, patient-specific digital twins modeling temporally varying digestive motion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.01909v3](http://arxiv.org/pdf/2507.01909v3)**

> **作者:** Jorge Tapias Gomez; Nishant Nadkarni; Lando S. Bosma; Jue Jiang; Ergys D. Subashi; William P. Segars; James M. Balter; Mert R Sabuncu; Neelam Tyagi; Harini Veeraraghavan
>
> **备注:** This work is still review, it contains 7 Pages, 6 figures, and 4 tables
>
> **摘要:** Objective: Clinical implementation of deformable image registration (DIR) requires voxel-based spatial accuracy metrics such as manually identified landmarks, which are challenging to implement for highly mobile gastrointestinal (GI) organs. To address this, patient-specific digital twins (DT) modeling temporally varying motion were created to assess the accuracy of DIR methods. Approach: 21 motion phases simulating digestive GI motion as 4D sequences were generated from static 3D patient scans using published analytical GI motion models through a semi-automated pipeline. Eleven datasets, including six T2w FSE MRI (T2w MRI), two T1w 4D golden-angle stack-of-stars, and three contrast-enhanced CT scans. The motion amplitudes of the DTs were assessed against real patient stomach motion amplitudes extracted from independent 4D MRI datasets. The generated DTs were then used to assess six different DIR methods using target registration error, Dice similarity coefficient, and the 95th percentile Hausdorff distance using summary metrics and voxel-level granular visualizations. Finally, for a subset of T2w MRI scans from patients treated with MR-guided radiation therapy, dose distributions were warped and accumulated to assess dose warping errors, including evaluations of DIR performance in both low- and high-dose regions for patient-specific error estimation. Main results: Our proposed pipeline synthesized DTs modeling realistic GI motion, achieving mean and maximum motion amplitudes and a mean log Jacobian determinant within 0.8 mm and 0.01, respectively, similar to published real-patient gastric motion data. It also enables the extraction of detailed quantitative DIR performance metrics and rigorous validation of dose mapping accuracy. Significance: The pipeline enables rigorously testing DIR tools for dynamic, anatomically complex regions enabling granular spatial and dosimetric accuracies.
>
---
#### [replaced 013] Omni-Video: Democratizing Unified Video Understanding and Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.06119v2](http://arxiv.org/pdf/2507.06119v2)**

> **作者:** Zhiyu Tan; Hao Yang; Luozheng Qin; Jia Gong; Mengping Yang; Hao Li
>
> **备注:** Technical report, project page: https://howellyoung-s.github.io/OmniVideo_project/
>
> **摘要:** Notable breakthroughs in unified understanding and generation modeling have led to remarkable advancements in image understanding, reasoning, production and editing, yet current foundational models predominantly focus on processing images, creating a gap in the development of unified models for video understanding and generation. This report presents Omni-Video, an efficient and effective unified framework for video understanding, generation, as well as instruction-based editing. Our key insight is to teach existing multimodal large language models (MLLMs) to produce continuous visual clues that are used as the input of diffusion decoders, which produce high-quality videos conditioned on these visual clues. To fully unlock the potential of our system for unified video modeling, we integrate several technical improvements: 1) a lightweight architectural design that respectively attaches a vision head on the top of MLLMs and a adapter before the input of diffusion decoders, the former produce visual tokens for the latter, which adapts these visual tokens to the conditional space of diffusion decoders; and 2) an efficient multi-stage training scheme that facilitates a fast connection between MLLMs and diffusion decoders with limited data and computational resources. We empirically demonstrate that our model exhibits satisfactory generalization abilities across video generation, editing and understanding tasks.
>
---
#### [replaced 014] DriveMRP: Enhancing Vision-Language Models with Synthetic Motion Data for Motion Risk Prediction
- **分类: cs.CV; cs.AI; cs.RO; I.4.8; I.2.7; I.2.10**

- **链接: [http://arxiv.org/pdf/2507.02948v2](http://arxiv.org/pdf/2507.02948v2)**

> **作者:** Zhiyi Hou; Enhui Ma; Fang Li; Zhiyi Lai; Kalok Ho; Zhanqian Wu; Lijun Zhou; Long Chen; Chitian Sun; Haiyang Sun; Bing Wang; Guang Chen; Hangjun Ye; Kaicheng Yu
>
> **备注:** 12 pages, 4 figures. Code available at https://github.com/hzy138/DriveMRP
>
> **摘要:** Autonomous driving has seen significant progress, driven by extensive real-world data. However, in long-tail scenarios, accurately predicting the safety of the ego vehicle's future motion remains a major challenge due to uncertainties in dynamic environments and limitations in data coverage. In this work, we aim to explore whether it is possible to enhance the motion risk prediction capabilities of Vision-Language Models (VLM) by synthesizing high-risk motion data. Specifically, we introduce a Bird's-Eye View (BEV) based motion simulation method to model risks from three aspects: the ego-vehicle, other vehicles, and the environment. This allows us to synthesize plug-and-play, high-risk motion data suitable for VLM training, which we call DriveMRP-10K. Furthermore, we design a VLM-agnostic motion risk estimation framework, named DriveMRP-Agent. This framework incorporates a novel information injection strategy for global context, ego-vehicle perspective, and trajectory projection, enabling VLMs to effectively reason about the spatial relationships between motion waypoints and the environment. Extensive experiments demonstrate that by fine-tuning with DriveMRP-10K, our DriveMRP-Agent framework can significantly improve the motion risk prediction performance of multiple VLM baselines, with the accident recognition accuracy soaring from 27.13% to 88.03%. Moreover, when tested via zero-shot evaluation on an in-house real-world high-risk motion dataset, DriveMRP-Agent achieves a significant performance leap, boosting the accuracy from base_model's 29.42% to 68.50%, which showcases the strong generalization capabilities of our method in real-world scenarios.
>
---
#### [replaced 015] EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.15582v2](http://arxiv.org/pdf/2411.15582v2)**

> **作者:** Xiaobao Wei; Qingpo Wuwu; Zhongyu Zhao; Zhuangzhe Wu; Nan Huang; Ming Lu; Ningning MA; Shanghang Zhang
>
> **备注:** Acccpeted by ICCV2025
>
> **摘要:** Photorealistic reconstruction of street scenes is essential for developing real-world simulators in autonomous driving. While recent methods based on 3D/4D Gaussian Splatting (GS) have demonstrated promising results, they still encounter challenges in complex street scenes due to the unpredictable motion of dynamic objects. Current methods typically decompose street scenes into static and dynamic objects, learning the Gaussians in either a supervised manner (e.g., w/ 3D bounding-box) or a self-supervised manner (e.g., w/o 3D bounding-box). However, these approaches do not effectively model the motions of dynamic objects (e.g., the motion speed of pedestrians is clearly different from that of vehicles), resulting in suboptimal scene decomposition. To address this, we propose Explicit Motion Decomposition (EMD), which models the motions of dynamic objects by introducing learnable motion embeddings to the Gaussians, enhancing the decomposition in street scenes. The proposed plug-and-play EMD module compensates for the lack of motion modeling in self-supervised street Gaussian splatting methods. We also introduce tailored training strategies to extend EMD to supervised approaches. Comprehensive experiments demonstrate the effectiveness of our method, achieving state-of-the-art novel view synthesis performance in self-supervised settings. The code is available at: https://qingpowuwu.github.io/emd.
>
---
#### [replaced 016] ReCamMaster: Camera-Controlled Generative Rendering from A Single Video
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.11647v2](http://arxiv.org/pdf/2503.11647v2)**

> **作者:** Jianhong Bai; Menghan Xia; Xiao Fu; Xintao Wang; Lianrui Mu; Jinwen Cao; Zuozhu Liu; Haoji Hu; Xiang Bai; Pengfei Wan; Di Zhang
>
> **备注:** Project page: https://jianhongbai.github.io/ReCamMaster/
>
> **摘要:** Camera control has been actively studied in text or image conditioned video generation tasks. However, altering camera trajectories of a given video remains under-explored, despite its importance in the field of video creation. It is non-trivial due to the extra constraints of maintaining multiple-frame appearance and dynamic synchronization. To address this, we present ReCamMaster, a camera-controlled generative video re-rendering framework that reproduces the dynamic scene of an input video at novel camera trajectories. The core innovation lies in harnessing the generative capabilities of pre-trained text-to-video models through a simple yet powerful video conditioning mechanism--its capability is often overlooked in current research. To overcome the scarcity of qualified training data, we construct a comprehensive multi-camera synchronized video dataset using Unreal Engine 5, which is carefully curated to follow real-world filming characteristics, covering diverse scenes and camera movements. It helps the model generalize to in-the-wild videos. Lastly, we further improve the robustness to diverse inputs through a meticulously designed training strategy. Extensive experiments show that our method substantially outperforms existing state-of-the-art approaches. Our method also finds promising applications in video stabilization, super-resolution, and outpainting. Our code and dataset are publicly available at: https://github.com/KwaiVGI/ReCamMaster.
>
---
#### [replaced 017] Multi-Modality Conditioned Variational U-Net for Field-of-View Extension in Brain Diffusion MRI
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.13846v2](http://arxiv.org/pdf/2409.13846v2)**

> **作者:** Zhiyuan Li; Chenyu Gao; Praitayini Kanakaraj; Shunxing Bao; Lianrui Zuo; Michael E. Kim; Nancy R. Newlin; Gaurav Rudravaram; Nazirah M. Khairi; Yuankai Huo; Kurt G. Schilling; Walter A. Kukull; Arthur W. Toga; Derek B. Archer; Timothy J. Hohman; Bennett A. Landman
>
> **摘要:** An incomplete field-of-view (FOV) in diffusion magnetic resonance imaging (dMRI) can severely hinder the volumetric and bundle analyses of whole-brain white matter connectivity. Although existing works have investigated imputing the missing regions using deep generative models, it remains unclear how to specifically utilize additional information from paired multi-modality data and whether this can enhance the imputation quality and be useful for downstream tractography. To fill this gap, we propose a novel framework for imputing dMRI scans in the incomplete part of the FOV by integrating the learned diffusion features in the acquired part of the FOV to the complete brain anatomical structure. We hypothesize that by this design the proposed framework can enhance the imputation performance of the dMRI scans and therefore be useful for repairing whole-brain tractography in corrupted dMRI scans with incomplete FOV. We tested our framework on two cohorts from different sites with a total of 96 subjects and compared it with a baseline imputation method that treats the information from T1w and dMRI scans equally. The proposed framework achieved significant improvements in imputation performance, as demonstrated by angular correlation coefficient (p < 1E-5), and in downstream tractography accuracy, as demonstrated by Dice score (p < 0.01). Results suggest that the proposed framework improved imputation performance in dMRI scans by specifically utilizing additional information from paired multi-modality data, compared with the baseline method. The imputation achieved by the proposed framework enhances whole brain tractography, and therefore reduces the uncertainty when analyzing bundles associated with neurodegenerative.
>
---
#### [replaced 018] Beyond Accuracy: Metrics that Uncover What Makes a 'Good' Visual Descriptor
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.03542v2](http://arxiv.org/pdf/2507.03542v2)**

> **作者:** Ethan Lin; Linxi Zhao; Atharva Sehgal; Jennifer J. Sun
>
> **备注:** VisCon @ CVPR 2025
>
> **摘要:** Text-based visual descriptors--ranging from simple class names to more descriptive phrases--are widely used in visual concept discovery and image classification with vision-language models (VLMs). Their effectiveness, however, depends on a complex interplay of factors, including semantic clarity, presence in the VLM's pre-training data, and how well the descriptors serve as a meaningful representation space. In this work, we systematically analyze descriptor quality along two key dimensions: (1) representational capacity, and (2) relationship with VLM pre-training data. We evaluate a spectrum of descriptor generation methods, from zero-shot LLM-generated prompts to iteratively refined descriptors. Motivated by ideas from representation alignment and language understanding, we introduce two alignment-based metrics--Global Alignment and CLIP Similarity--that move beyond accuracy. These metrics shed light on how different descriptor generation strategies interact with foundation model properties, offering new ways to study descriptor effectiveness beyond accuracy evaluations.
>
---
#### [replaced 019] Language-Unlocked ViT (LUViT): Empowering Self-Supervised Vision Transformers with LLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.00754v2](http://arxiv.org/pdf/2507.00754v2)**

> **作者:** Selim Kuzucu; Muhammad Ferjad Naeem; Anna Kukleva; Federico Tombari; Bernt Schiele
>
> **备注:** 26 pages, 6 figures
>
> **摘要:** The integration of Large Language Model (LLMs) blocks with Vision Transformers (ViTs) holds immense promise for vision-only tasks by leveraging the rich semantic knowledge and reasoning capabilities of LLMs. However, a fundamental challenge lies in the inherent modality mismatch between text-centric pretraining of LLMs and vision-centric training of ViTs. Direct fusion often fails to fully exploit the LLM's potential and suffers from unstable finetuning. As a result, LLM blocks are kept frozen while only the vision components are learned. As a remedy to these challenges, we introduce Language-Unlocked Vision Transformers (LUViT), a novel approach that bridges this modality mismatch through a synergistic pre-training strategy. LUViT co-adapts a ViT backbone and an LLM fusion block by (1) employing Masked Auto-Encoding (MAE) to pre-train the ViT for richer visual representations, and (2) concurrently training Low-Rank Adaptation (LoRA) layers within the LLM block using the MAE objective. This joint optimization guides the ViT to produce LLM-aligned features and the LLM to effectively interpret visual information. We demonstrate through extensive experiments that LUViT significantly improves performance on various downstream vision tasks, showcasing a more effective and efficient pathway to harness LLM knowledge for visual understanding.
>
---
#### [replaced 020] Dynamic Reconstruction of Hand-Object Interaction with Distributed Force-aware Contact Representation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.09572v2](http://arxiv.org/pdf/2411.09572v2)**

> **作者:** Zhenjun Yu; Wenqiang Xu; Pengfei Xie; Yutong Li; Brian W. Anthony; Zhuorui Zhang; Cewu Lu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** We present ViTaM-D, a novel visual-tactile framework for reconstructing dynamic hand-object interaction with distributed tactile sensing to enhance contact modeling. Existing methods, relying solely on visual inputs, often fail to capture occluded interactions and object deformation. To address this, we introduce DF-Field, a distributed force-aware contact representation leveraging kinetic and potential energy in hand-object interactions. ViTaM-D first reconstructs interactions using a visual network with contact constraint, then refines contact details through force-aware optimization, improving object deformation modeling. To evaluate deformable object reconstruction, we introduce the HOT dataset, featuring 600 hand-object interaction sequences in a high-precision simulation environment. Experiments on DexYCB and HOT datasets show that ViTaM-D outperforms state-of-the-art methods in reconstruction accuracy for both rigid and deformable objects. DF-Field also proves more effective in refining hand poses and enhancing contact modeling than previous refinement methods. The code, models, and datasets are available at https://sites.google.com/view/vitam-d/.
>
---
#### [replaced 021] Federated Breast Cancer Detection Enhanced by Synthetic Ultrasound Image Augmentation
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.23334v2](http://arxiv.org/pdf/2506.23334v2)**

> **作者:** Hongyi Pan; Ziliang Hong; Gorkem Durak; Ziyue Xu; Ulas Bagci
>
> **摘要:** Federated learning (FL) has emerged as a promising paradigm for collaboratively training deep learning models across institutions without exchanging sensitive medical data. However, its effectiveness is often hindered by limited data availability and non-independent, identically distributed data across participating clients, which can degrade model performance and generalization. To address these challenges, we propose a generative AI based data augmentation framework that integrates synthetic image sharing into the federated training process for breast cancer diagnosis via ultrasound images. Specifically, we train two simple class-specific Deep Convolutional Generative Adversarial Networks: one for benign and one for malignant lesions. We then simulate a realistic FL setting using three publicly available breast ultrasound image datasets: BUSI, BUS-BRA, and UDIAT. FedAvg and FedProx are adopted as baseline FL algorithms. Experimental results show that incorporating a suitable number of synthetic images improved the average AUC from 0.9206 to 0.9237 for FedAvg and from 0.9429 to 0.9538 for FedProx. We also note that excessive use of synthetic data reduced performance, underscoring the importance of maintaining a balanced ratio of real and synthetic samples. Our findings highlight the potential of generative AI based data augmentation to enhance FL results in the breast ultrasound image classification task.
>
---
#### [replaced 022] Label-Efficient LiDAR Panoptic Segmentation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.02372v2](http://arxiv.org/pdf/2503.02372v2)**

> **作者:** Ahmet Selim Çanakçı; Niclas Vödisch; Kürsat Petek; Wolfram Burgard; Abhinav Valada
>
> **备注:** Accepted for the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025
>
> **摘要:** A main bottleneck of learning-based robotic scene understanding methods is the heavy reliance on extensive annotated training data, which often limits their generalization ability. In LiDAR panoptic segmentation, this challenge becomes even more pronounced due to the need to simultaneously address both semantic and instance segmentation from complex, high-dimensional point cloud data. In this work, we address the challenge of LiDAR panoptic segmentation with very few labeled samples by leveraging recent advances in label-efficient vision panoptic segmentation. To this end, we propose a novel method, Limited-Label LiDAR Panoptic Segmentation (L3PS), which requires only a minimal amount of labeled data. Our approach first utilizes a label-efficient 2D network to generate panoptic pseudo-labels from a small set of annotated images, which are subsequently projected onto point clouds. We then introduce a novel 3D refinement module that capitalizes on the geometric properties of point clouds. By incorporating clustering techniques, sequential scan accumulation, and ground point separation, this module significantly enhances the accuracy of the pseudo-labels, improving segmentation quality by up to +10.6 PQ and +7.9 mIoU. We demonstrate that these refined pseudo-labels can be used to effectively train off-the-shelf LiDAR segmentation networks. Through extensive experiments, we show that L3PS not only outperforms existing methods but also substantially reduces the annotation burden. We release the code of our work at https://l3ps.cs.uni-freiburg.de.
>
---
#### [replaced 023] Transformer-Driven Active Transfer Learning for Cross-Hyperspectral Image Classification
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.18115v2](http://arxiv.org/pdf/2411.18115v2)**

> **作者:** Muhammad Ahmad; Francesco Mauro; Manuel Mazzara; Salvatore Distefano; Adil Mehmood Khan; Silvia Liberata Ullo
>
> **摘要:** Hyperspectral image (HSI) classification presents inherent challenges due to high spectral dimensionality, significant domain shifts, and limited availability of labeled data. To address these issues, we propose a novel Active Transfer Learning (ATL) framework built upon a Spatial-Spectral Transformer (SST) backbone. The framework integrates multistage transfer learning with an uncertainty-diversity-driven active learning mechanism that strategically selects highly informative and diverse samples for annotation, thereby significantly reducing labeling costs and mitigating sample redundancy. A dynamic layer freezing strategy is introduced to enhance transferability and computational efficiency, enabling selective adaptation of model layers based on domain shift characteristics. Furthermore, we incorporate a self-calibrated attention mechanism that dynamically refines spatial and spectral weights during adaptation, guided by uncertainty-aware feedback. A diversity-promoting sampling strategy ensures broad spectral coverage among selected samples, preventing overfitting to specific classes. Extensive experiments on benchmark cross-domain HSI datasets demonstrate that the proposed SST-ATL framework achieves superior classification performance compared to conventional approaches. The source code is publicly available at https://github.com/mahmad000/ATL-SST.
>
---
#### [replaced 024] LongAnimation: Long Animation Generation with Dynamic Global-Local Memory
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2507.01945v2](http://arxiv.org/pdf/2507.01945v2)**

> **作者:** Nan Chen; Mengqi Huang; Yihao Meng; Zhendong Mao
>
> **摘要:** Animation colorization is a crucial part of real animation industry production. Long animation colorization has high labor costs. Therefore, automated long animation colorization based on the video generation model has significant research value. Existing studies are limited to short-term colorization. These studies adopt a local paradigm, fusing overlapping features to achieve smooth transitions between local segments. However, the local paradigm neglects global information, failing to maintain long-term color consistency. In this study, we argue that ideal long-term color consistency can be achieved through a dynamic global-local paradigm, i.e., dynamically extracting global color-consistent features relevant to the current generation. Specifically, we propose LongAnimation, a novel framework, which mainly includes a SketchDiT, a Dynamic Global-Local Memory (DGLM), and a Color Consistency Reward. The SketchDiT captures hybrid reference features to support the DGLM module. The DGLM module employs a long video understanding model to dynamically compress global historical features and adaptively fuse them with the current generation features. To refine the color consistency, we introduce a Color Consistency Reward. During inference, we propose a color consistency fusion to smooth the video segment transition. Extensive experiments on both short-term (14 frames) and long-term (average 500 frames) animations show the effectiveness of LongAnimation in maintaining short-term and long-term color consistency for open-domain animation colorization task. The code can be found at https://cn-makers.github.io/long_animation_web/.
>
---
#### [replaced 025] Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models
- **分类: cs.AI; cs.CV; 68T01**

- **链接: [http://arxiv.org/pdf/2507.03916v2](http://arxiv.org/pdf/2507.03916v2)**

> **作者:** Yifan Jiang; Yibo Xue; Yukun Kang; Pin Zheng; Jian Peng; Feiran Wu; Changliang Xu
>
> **备注:** Appendix at: https://github.com/PAMPAS-Lab/ANA-PPT-Anamation/blob/main/Appendix.pdf
>
> **摘要:** Slide animations, such as fade-in, fly-in, and wipe, are critical for audience engagement, efficient information delivery, and vivid visual expression. However, most AI-driven slide-generation tools still lack native animation support, and existing vision-language models (VLMs) struggle with animation tasks due to the absence of public datasets and limited temporal-reasoning capabilities. To address this gap, we release the first public dataset for slide-animation modeling: 12,000 triplets of natural-language descriptions, animation JSON files, and rendered videos, collectively covering every built-in PowerPoint effect. Using this resource, we fine-tune Qwen-2.5-VL-7B with Low-Rank Adaptation (LoRA) and achieve consistent improvements over GPT-4.1 and Gemini-2.5-Pro in BLEU-4, ROUGE-L, SPICE, and our Coverage-Order-Detail Assessment (CODA) metric, which evaluates action coverage, temporal order, and detail fidelity. On a manually created test set of slides, the LoRA model increases BLEU-4 by around 60%, ROUGE-L by 30%, and shows significant improvements in CODA-detail. This demonstrates that low-rank adaptation enables reliable temporal reasoning and generalization beyond synthetic data. Overall, our dataset, LoRA-enhanced model, and CODA metric provide a rigorous benchmark and foundation for future research on VLM-based dynamic slide generation.
>
---
#### [replaced 026] CaO$_2$: Rectifying Inconsistencies in Diffusion-Based Dataset Distillation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.22637v2](http://arxiv.org/pdf/2506.22637v2)**

> **作者:** Haoxuan Wang; Zhenghao Zhao; Junyi Wu; Yuzhang Shang; Gaowen Liu; Yan Yan
>
> **备注:** ICCV 2025. Code is available at https://github.com/hatchetProject/CaO2
>
> **摘要:** The recent introduction of diffusion models in dataset distillation has shown promising potential in creating compact surrogate datasets for large, high-resolution target datasets, offering improved efficiency and performance over traditional bi-level/uni-level optimization methods. However, current diffusion-based dataset distillation approaches overlook the evaluation process and exhibit two critical inconsistencies in the distillation process: (1) Objective Inconsistency, where the distillation process diverges from the evaluation objective, and (2) Condition Inconsistency, leading to mismatches between generated images and their corresponding conditions. To resolve these issues, we introduce Condition-aware Optimization with Objective-guided Sampling (CaO$_2$), a two-stage diffusion-based framework that aligns the distillation process with the evaluation objective. The first stage employs a probability-informed sample selection pipeline, while the second stage refines the corresponding latent representations to improve conditional likelihood. CaO$_2$ achieves state-of-the-art performance on ImageNet and its subsets, surpassing the best-performing baselines by an average of 2.3% accuracy.
>
---
#### [replaced 027] Medical Image Segmentation Using Advanced Unet: VMSE-Unet and VM-Unet CBAM+
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.00511v2](http://arxiv.org/pdf/2507.00511v2)**

> **作者:** Sayandeep Kanrar; Raja Piyush; Qaiser Razi; Debanshi Chakraborty; Vikas Hassija; GSS Chalapathi
>
> **摘要:** In this paper, we present the VMSE U-Net and VM-Unet CBAM+ model, two cutting-edge deep learning architectures designed to enhance medical image segmentation. Our approach integrates Squeeze-and-Excitation (SE) and Convolutional Block Attention Module (CBAM) techniques into the traditional VM U-Net framework, significantly improving segmentation accuracy, feature localization, and computational efficiency. Both models show superior performance compared to the baseline VM-Unet across multiple datasets. Notably, VMSEUnet achieves the highest accuracy, IoU, precision, and recall while maintaining low loss values. It also exhibits exceptional computational efficiency with faster inference times and lower memory usage on both GPU and CPU. Overall, the study suggests that the enhanced architecture VMSE-Unet is a valuable tool for medical image analysis. These findings highlight its potential for real-world clinical applications, emphasizing the importance of further research to optimize accuracy, robustness, and computational efficiency.
>
---
#### [replaced 028] DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.06505v2](http://arxiv.org/pdf/2503.06505v2)**

> **作者:** Xirui Hu; Jiahao Wang; Hao Chen; Weizhan Zhang; Benqi Wang; Yikun Li; Haishun Nan
>
> **备注:** ICCV 2025
>
> **摘要:** Recent advancements in text-to-image generation have spurred interest in personalized human image generation, which aims to create novel images featuring specific human identities as reference images indicate. Although existing methods achieve high-fidelity identity preservation, they often struggle with limited multi-ID usability and inadequate facial editability. We present DynamicID, a tuning-free framework supported by a dual-stage training paradigm that inherently facilitates both single-ID and multi-ID personalized generation with high fidelity and flexible facial editability. Our key innovations include: 1) Semantic-Activated Attention (SAA), which employs query-level activation gating to minimize disruption to the original model when injecting ID features and achieve multi-ID personalization without requiring multi-ID samples during training. 2) Identity-Motion Reconfigurator (IMR), which leverages contrastive learning to effectively disentangle and re-entangle facial motion and identity features, thereby enabling flexible facial editing. Additionally, we have developed a curated VariFace-10k facial dataset, comprising 10k unique individuals, each represented by 35 distinct facial images. Experimental results demonstrate that DynamicID outperforms state-of-the-art methods in identity fidelity, facial editability, and multi-ID personalization capability.
>
---
#### [replaced 029] PR-ENDO: Physically Based Relightable Gaussian Splatting for Endoscopy
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.12510v2](http://arxiv.org/pdf/2411.12510v2)**

> **作者:** Joanna Kaleta; Weronika Smolak-Dyżewska; Dawid Malarz; Diego Dall'Alba; Przemysław Korzeniowski; Przemysław Spurek
>
> **摘要:** Endoluminal endoscopic procedures are essential for diagnosing colorectal cancer and other severe conditions in the digestive tract, urogenital system, and airways. 3D reconstruction and novel-view synthesis from endoscopic images are promising tools for enhancing diagnosis. Moreover, integrating physiological deformations and interaction with the endoscope enables the development of simulation tools from real video data. However, constrained camera trajectories and view-dependent lighting create artifacts, leading to inaccurate or overfitted reconstructions. We present PR-ENDO, a novel 3D reconstruction framework leveraging the unique property of endoscopic imaging, where a single light source is closely aligned with the camera. Our method separates light effects from tissue properties. PR-ENDO enhances 3D Gaussian Splatting with a physically based relightable model. We boost the traditional light transport formulation with a specialized MLP capturing complex light-related effects while ensuring reduced artifacts and better generalization across novel views. PR-ENDO achieves superior reconstruction quality compared to baseline methods on both public and in-house datasets. Unlike existing approaches, PR-ENDO enables tissue modifications while preserving a physically accurate response to light, making it closer to real-world clinical use.
>
---
#### [replaced 030] Non-Negative Reduced Biquaternion Matrix Factorization with Applications in Color Face Recognition
- **分类: cs.CV; cs.NA; math.NA**

- **链接: [http://arxiv.org/pdf/2408.05582v2](http://arxiv.org/pdf/2408.05582v2)**

> **作者:** Jifei Miao; Junjun Pan; Michael K. Ng
>
> **摘要:** Reduced biquaternion (RB), as a four-dimensional algebra highly suitable for representing color pixels, has recently garnered significant attention from numerous scholars. In this paper, for color image processing problems, we introduce a concept of the non-negative RB matrix and then use the multiplication properties of RB to propose a non-negative RB matrix factorization (NRBMF) model. The NRBMF model is introduced to address the challenge of reasonably establishing a non-negative quaternion matrix factorization model, which is primarily hindered by the multiplication properties of traditional quaternions. Furthermore, this paper transforms the problem of solving the NRBMF model into an RB alternating non-negative least squares (RB-ANNLS) problem. Then, by introducing a method to compute the gradient of the real function with RB matrix variables, we solve the RB-ANNLS optimization problem using the RB projected gradient algorithm and conduct a convergence analysis of the algorithm. Finally, we validate the effectiveness and superiority of the proposed NRBMF model in color face recognition.
>
---
#### [replaced 031] Geo-Registration of Terrestrial LiDAR Point Clouds with Satellite Images without GNSS
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.05999v2](http://arxiv.org/pdf/2507.05999v2)**

> **作者:** Xinyu Wang; Muhammad Ibrahim; Haitian Wang; Atif Mansoor; Ajmal Mian
>
> **备注:** Submitted to IEEE Transactions on Geoscience & Remote Sensing. Under reviewing now
>
> **摘要:** Accurate geo-registration of LiDAR point clouds presents significant challenges in GNSS signal denied urban areas with high-rise buildings and bridges. Existing methods typically rely on real-time GNSS and IMU data, that require pre-calibration and assume stable positioning during data collection. However, this assumption often fails in dense urban areas, resulting in localization errors. To address this, we propose a structured geo-registration and spatial correction method that aligns 3D point clouds with satellite images, enabling frame-wise recovery of GNSS information and reconstruction of city scale 3D maps without relying on prior localization. The proposed approach employs a pre-trained Point Transformer model to segment the road points and then extracts the road skeleton and intersection points from the point cloud as well as the target map for alignment. Global rigid alignment of the two is performed using the intersection points, followed by local refinement using radial basis function (RBF) interpolation. Elevation correction is then applied to the point cloud based on terrain information from SRTM dataset to resolve vertical discrepancies. The proposed method was tested on the popular KITTI benchmark and a locally collected Perth (Western Australia) CBD dataset. On the KITTI dataset, our method achieved an average planimetric alignment standard deviation (STD) of 0.84~m across sequences with intersections, representing a 55.3\% improvement over the original dataset. On the Perth dataset, which lacks GNSS information, our method achieved an average STD of 0.96~m compared to the GPS data extracted from Google Maps API. This corresponds to a 77.4\% improvement from the initial alignment. Our method also resulted in elevation correlation gains of 30.5\% on the KITTI dataset and 50.4\% on the Perth dataset.
>
---
#### [replaced 032] Refining Skewed Perceptions in Vision-Language Contrastive Models through Visual Representations
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2405.14030v3](http://arxiv.org/pdf/2405.14030v3)**

> **作者:** Haocheng Dai; Sarang Joshi
>
> **备注:** 10 pages, 8 figures
>
> **摘要:** Large vision-language contrastive models (VLCMs), such as CLIP, have become foundational, demonstrating remarkable success across a variety of downstream tasks. Despite their advantages, these models, akin to other foundational systems, inherit biases from the disproportionate distribution of real-world data, leading to misconceptions about the actual environment. Prevalent datasets like ImageNet are often riddled with non-causal, spurious correlations that can diminish VLCM performance in scenarios where these contextual elements are absent. This study presents an investigation into how a simple linear probe can effectively distill task-specific core features from CLIP's embedding for downstream applications. Our analysis reveals that the CLIP text representations are often tainted by spurious correlations, inherited in the biased pre-training dataset. Empirical evidence suggests that relying on visual representations from CLIP, as opposed to text embedding, is more effective to refine the skewed perceptions in VLCMs, emphasizing the superior utility of visual representations in overcoming embedded biases. Our code can be found here.
>
---
#### [replaced 033] EgoVIS@CVPR: PAIR-Net: Enhancing Egocentric Speaker Detection via Pretrained Audio-Visual Fusion and Alignment Loss
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.02247v2](http://arxiv.org/pdf/2506.02247v2)**

> **作者:** Yu Wang; Juhyung Ha; David J. Crandall
>
> **备注:** 4 pages, 1 figure, and 1 table
>
> **摘要:** Active speaker detection (ASD) in egocentric videos presents unique challenges due to unstable viewpoints, motion blur, and off-screen speech sources - conditions under which traditional visual-centric methods degrade significantly. We introduce PAIR-Net (Pretrained Audio-Visual Integration with Regularization Network), an effective model that integrates a partially frozen Whisper audio encoder with a fine-tuned AV-HuBERT visual backbone to robustly fuse cross-modal cues. To counteract modality imbalance, we introduce an inter-modal alignment loss that synchronizes audio and visual representations, enabling more consistent convergence across modalities. Without relying on multi-speaker context or ideal frontal views, PAIR-Net achieves state-of-the-art performance on the Ego4D ASD benchmark with 76.6% mAP, surpassing LoCoNet and STHG by 8.2% and 12.9% mAP, respectively. Our results highlight the value of pretrained audio priors and alignment-based fusion for robust ASD under real-world egocentric conditions.
>
---
#### [replaced 034] CULTURE3D: A Large-Scale and Diverse Dataset of Cultural Landmarks and Terrains for Gaussian-Based Scene Rendering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.06927v3](http://arxiv.org/pdf/2501.06927v3)**

> **作者:** Xinyi Zheng; Steve Zhang; Weizhe Lin; Aaron Zhang; Walterio W. Mayol-Cuevas; Yunze Liu; Junxiao Shen
>
> **摘要:** Current state-of-the-art 3D reconstruction models face limitations in building extra-large scale outdoor scenes, primarily due to the lack of sufficiently large-scale and detailed datasets. In this paper, we present a extra-large fine-grained dataset with 10 billion points composed of 41,006 drone-captured high-resolution aerial images, covering 20 diverse and culturally significant scenes from worldwide locations such as Cambridge Uni main buildings, the Pyramids, and the Forbidden City Palace. Compared to existing datasets, ours offers significantly larger scale and higher detail, uniquely suited for fine-grained 3D applications. Each scene contains an accurate spatial layout and comprehensive structural information, supporting detailed 3D reconstruction tasks. By reconstructing environments using these detailed images, our dataset supports multiple applications, including outputs in the widely adopted COLMAP format, establishing a novel benchmark for evaluating state-of-the-art large-scale Gaussian Splatting methods.The dataset's flexibility encourages innovations and supports model plug-ins, paving the way for future 3D breakthroughs. All datasets and code will be open-sourced for community use.
>
---
#### [replaced 035] OpenThinkIMG: Learning to Think with Images via Visual Tool Reinforcement Learning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08617v2](http://arxiv.org/pdf/2505.08617v2)**

> **作者:** Zhaochen Su; Linjie Li; Mingyang Song; Yunzhuo Hao; Zhengyuan Yang; Jun Zhang; Guanjie Chen; Jiawei Gu; Juntao Li; Xiaoye Qu; Yu Cheng
>
> **备注:** Work in progress
>
> **摘要:** While humans can flexibly leverage interactive visual cognition for complex problem-solving, enabling Large Vision-Language Models (LVLMs) to learn similarly adaptive behaviors with visual tools remains challenging. A significant hurdle is the current lack of standardized infrastructure, which hinders integrating diverse tools, generating rich interaction data, and training robust agents effectively. To address these gaps, we introduce OpenThinkIMG, the first open-source, comprehensive end-to-end framework for tool-augmented LVLMs. It features standardized vision tool interfaces, scalable trajectory generation for policy initialization, and a flexible training environment. Furthermore, considering supervised fine-tuning (SFT) on static demonstrations offers limited policy generalization for dynamic tool invocation, we propose a novel reinforcement learning (RL) framework V-ToolRL to train LVLMs to learn adaptive policies for invoking external vision tools. V-ToolRL enables LVLMs to autonomously discover optimal tool-usage strategies by directly optimizing for task success using feedback from tool interactions. We empirically validate V-ToolRL on challenging chart reasoning tasks. Our RL-trained agent, built upon a Qwen2-VL-2B, significantly outperforms its SFT-initialized counterpart (+28.83 points) and surpasses established supervised tool-learning baselines like Taco and CogCom by an average of +12.7 points. Notably, it also surpasses prominent closed-source models like GPT-4.1 by +8.68 accuracy points. We hope OpenThinkIMG can serve as a foundational framework for advancing dynamic, tool-augmented visual reasoning, helping the community develop AI agents that can genuinely "think with images".
>
---
#### [replaced 036] Oscillation-Reduced MXFP4 Training for Vision Transformers
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.20853v2](http://arxiv.org/pdf/2502.20853v2)**

> **作者:** Yuxiang Chen; Haocheng Xi; Jun Zhu; Jianfei Chen
>
> **摘要:** Pre-training Transformers in FP4 precision is becoming a promising approach to gain substantial speedup, but it comes with a considerable loss of accuracy. Microscaling (MX) data format provides a fine-grained per-group quantization method to improve the representation ability of the FP4 format and is supported by the next-generation Blackwell GPU architecture. However, training with MXFP4 data format still results in significant degradation and there is a lack of systematic research on the reason. In this work, we propose a novel training method TetraJet for a more accurate FP4 training. We comprehensively evaluate all of the quantizers involved in the training, and identify the weight oscillation problem in the forward pass as the main source of the degradation in MXFP4 training. Therefore, we introduce two novel methods, EMA Quantizer (Q-EMA) and Adaptive Ramping Optimizer (Q-Ramping), to resolve the oscillation problem. Extensive experiments on Vision Transformers demonstrate that TetraJet consistently outperforms the existing 4-bit training methods, and Q-EMA & Q-Ramping can provide additional enhancement by effectively reducing oscillation. We decreased the accuracy degradation by more than $50\%$ compared to the baseline, and can even achieve competitive performance compared to full precision training. The codes are available at https://github.com/thu-ml/TetraJet-MXFP4Training
>
---
#### [replaced 037] VQ-SGen: A Vector Quantized Stroke Representation for Creative Sketch Generation
- **分类: cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2411.16446v3](http://arxiv.org/pdf/2411.16446v3)**

> **作者:** Jiawei Wang; Zhiming Cui; Changjian Li
>
> **备注:** Project Page: https://enigma-li.github.io/projects/VQ-SGen/VQ-SGen.html
>
> **摘要:** This paper presents VQ-SGen, a novel algorithm for high-quality creative sketch generation. Recent approaches have framed the task as pixel-based generation either as a whole or part-by-part, neglecting the intrinsic and contextual relationships among individual strokes, such as the shape and spatial positioning of both proximal and distant strokes. To overcome these limitations, we propose treating each stroke within a sketch as an entity and introducing a vector-quantized (VQ) stroke representation for fine-grained sketch generation. Our method follows a two-stage framework - in stage one, we decouple each stroke's shape and location information to ensure the VQ representation prioritizes stroke shape learning. In stage two, we feed the precise and compact representation into an auto-decoding Transformer to incorporate stroke semantics, positions, and shapes into the generation process. By utilizing tokenized stroke representation, our approach generates strokes with high fidelity and facilitates novel applications, such as text or class label conditioned generation and sketch completion. Comprehensive experiments demonstrate our method surpasses existing state-of-the-art techniques on the CreativeSketch dataset, underscoring its effectiveness.
>
---
#### [replaced 038] Sequential Attention-based Sampling for Histopathological Analysis
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.05077v2](http://arxiv.org/pdf/2507.05077v2)**

> **作者:** Tarun G; Naman Malpani; Gugan Thoppe; Sridharan Devarajan
>
> **摘要:** Deep neural networks are increasingly applied for automated histopathology. Yet, whole-slide images (WSIs) are often acquired at gigapixel sizes, rendering it computationally infeasible to analyze them entirely at high resolution. Diagnostic labels are largely available only at the slide-level, because expert annotation of images at a finer (patch) level is both laborious and expensive. Moreover, regions with diagnostic information typically occupy only a small fraction of the WSI, making it inefficient to examine the entire slide at full resolution. Here, we propose SASHA -- {\it S}equential {\it A}ttention-based {\it S}ampling for {\it H}istopathological {\it A}nalysis -- a deep reinforcement learning approach for efficient analysis of histopathological images. First, SASHA learns informative features with a lightweight hierarchical, attention-based multiple instance learning (MIL) model. Second, SASHA samples intelligently and zooms selectively into a small fraction (10-20\%) of high-resolution patches, to achieve reliable diagnosis. We show that SASHA matches state-of-the-art methods that analyze the WSI fully at high-resolution, albeit at a fraction of their computational and memory costs. In addition, it significantly outperforms competing, sparse sampling methods. We propose SASHA as an intelligent sampling model for medical imaging challenges that involve automated diagnosis with exceptionally large images containing sparsely informative features.
>
---
#### [replaced 039] Beyond Complete Shapes: A Quantitative Evaluation of 3D Shape Matching Algorithms
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.03511v2](http://arxiv.org/pdf/2411.03511v2)**

> **作者:** Viktoria Ehm; Nafie El Amrani; Yizheng Xie; Lennart Bastian; Maolin Gao; Weikang Wang; Lu Sang; Dongliang Cao; Tobias Weißberg; Zorah Lähner; Daniel Cremers; Florian Bernard
>
> **摘要:** Finding correspondences between 3D shapes is an important and long-standing problem in computer vision, graphics and beyond. While approaches based on machine learning dominate modern 3D shape matching, almost all existing (learning-based) methods require that at least one of the involved shapes is complete. In contrast, the most challenging and arguably most practically relevant setting of matching partially observed shapes, is currently underexplored. One important factor is that existing datasets contain only a small number of shapes (typically below 100), which are unable to serve data-hungry machine learning approaches, particularly in the unsupervised regime. In addition, the type of partiality present in existing datasets is often artificial and far from realistic. To address these limitations and to encourage research on these relevant settings, we provide a generic and flexible framework for the procedural generation of challenging partial shape matching scenarios. Our framework allows for a virtually infinite generation of partial shape matching instances from a finite set of shapes with complete geometry. Further, we manually create cross-dataset correspondences between seven existing (complete geometry) shape matching datasets, leading to a total of 2543 shapes. Based on this, we propose several challenging partial benchmark settings, for which we evaluate respective state-of-the-art methods as baselines.
>
---
#### [replaced 040] Bayesian Multi-Scale Neural Network for Crowd Counting
- **分类: cs.CV; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2007.14245v4](http://arxiv.org/pdf/2007.14245v4)**

> **作者:** Abhinav Sagar
>
> **摘要:** Crowd counting is a challenging yet critical task in computer vision with applications ranging from public safety to urban planning. Recent advances using Convolutional Neural Networks (CNNs) that estimate density maps have shown significant success. However, accurately counting individuals in highly congested scenes remains an open problem due to severe occlusions, scale variations, and perspective distortions, where people appear at drastically different sizes across the image. In this work, we propose a novel deep learning architecture that effectively addresses these challenges. Our network integrates a ResNet-based feature extractor for capturing rich hierarchical representations, followed by a downsampling block employing dilated convolutions to preserve spatial resolution while expanding the receptive field. An upsampling block using transposed convolutions reconstructs the high-resolution density map. Central to our architecture is a novel Perspective-aware Aggregation Module (PAM) designed to enhance robustness to scale and perspective variations by adaptively aggregating multi-scale contextual information. We detail the training procedure, including the loss functions and optimization strategies used. Our method is evaluated on three widely used benchmark datasets using Mean Absolute Error (MAE) and Mean Squared Error (MSE) as evaluation metrics. Experimental results demonstrate that our model achieves superior performance compared to existing state-of-the-art methods. Additionally, we incorporate principled Bayesian inference techniques to provide uncertainty estimates along with the crowd count predictions, offering a measure of confidence in the model's outputs.
>
---
#### [replaced 041] Skywork-R1V3 Technical Report
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.06167v2](http://arxiv.org/pdf/2507.06167v2)**

> **作者:** Wei Shen; Jiangbo Pei; Yi Peng; Xuchen Song; Yang Liu; Jian Peng; Haofeng Sun; Yunzhuo Hao; Peiyu Wang; Jianhao Zhang; Yahui Zhou
>
> **摘要:** We introduce Skywork-R1V3, an advanced, open-source vision-language model (VLM) that pioneers a new approach to visual reasoning. Its key innovation lies in effectively transferring reasoning skills from text-only Large Language Models (LLMs) to visual tasks. The strong performance of Skywork-R1V3 primarily stems from our elaborate post-training RL framework, which effectively activates and enhances the model's reasoning ability, without the need for additional continue pre-training. Through this framework, we further uncover the fundamental role of the connector module in achieving robust cross-modal alignment for multimodal reasoning models. In addition, we introduce a unique indicator of reasoning capability, the entropy of critical reasoning tokens, which has proven highly effective for checkpoint selection during RL training. Skywork-R1V3 achieves state-of-the-art results on MMMU, significantly improving from 64.3% to 76.0%. This performance matches entry-level human capabilities. Remarkably, our RL-powered post-training approach enables even the 38B parameter model to rival top closed-source VLMs. The implementation successfully transfers mathematical reasoning to other subject-related reasoning tasks. We also include an analysis of curriculum learning and reinforcement finetuning strategies, along with a broader discussion on multimodal reasoning. Skywork-R1V3 represents a significant leap in multimodal reasoning, showcasing RL as a powerful engine for advancing open-source VLM capabilities.
>
---
#### [replaced 042] CLIPDraw++: Text-to-Sketch Synthesis with Simple Primitives
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2312.02345v2](http://arxiv.org/pdf/2312.02345v2)**

> **作者:** Nityanand Mathur; Shyam Marjit; Abhra Chaudhuri; Anjan Dutta
>
> **备注:** Accepted at CVPRW-25. Project Page: https://clipdrawx.github.io/
>
> **摘要:** With the goal of understanding the visual concepts that CLIP associates with text prompts, we show that the latent space of CLIP can be visualized solely in terms of linear transformations on simple geometric primitives like straight lines and circles. Although existing approaches achieve this by sketch-synthesis-through-optimization, they do so on the space of higher order B\'ezier curves, which exhibit a wastefully large set of structures that they can evolve into, as most of them are non-essential for generating meaningful sketches. We present CLIPDraw++, an algorithm that provides significantly better visualizations for CLIP text embeddings, using only simple primitive shapes like straight lines and circles. This constrains the set of possible outputs to linear transformations on these primitives, thereby exhibiting an inherently simpler mathematical form. The synthesis process of CLIPDraw++ can be tracked end-to-end, with each visual concept being expressed exclusively in terms of primitives. Project Page: https://clipdrawx.github.io/.
>
---
#### [replaced 043] Are They the Same? Exploring Visual Correspondence Shortcomings of Multimodal LLMs
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.04670v3](http://arxiv.org/pdf/2501.04670v3)**

> **作者:** Yikang Zhou; Tao Zhang; Shilin Xu; Shihao Chen; Qianyu Zhou; Yunhai Tong; Shunping Ji; Jiangning Zhang; Lu Qi; Xiangtai Li
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Recent advancements in multimodal large language models (MLLM) have shown a strong ability in visual perception, reasoning abilities, and vision-language understanding. However, the visual matching ability of MLLMs is rarely studied, despite finding the visual correspondence of objects is essential in computer vision. Our research reveals that the matching capabilities in recent MLLMs still exhibit systematic shortcomings, even with current strong MLLMs models, GPT-4o. In particular, we construct a Multimodal Visual Matching (MMVM) benchmark to fairly benchmark over 30 different MLLMs. The MMVM benchmark is built from 15 open-source datasets and Internet videos with manual annotation. We categorize the data samples of MMVM benchmark into eight aspects based on the required cues and capabilities to more comprehensively evaluate and analyze current MLLMs. In addition, we have designed an automatic annotation pipeline to generate the MMVM SFT dataset, including 220K visual matching data with reasoning annotation. To our knowledge, this is the first visual corresponding dataset and benchmark for the MLLM community. Finally, we present CoLVA, a novel contrastive MLLM with two novel technical designs: fine-grained vision expert with object-level contrastive learning and instruction augmentation strategy. The former learns instance discriminative tokens, while the latter further improves instruction following ability. CoLVA-InternVL2-4B achieves an overall accuracy (OA) of 49.80\% on the MMVM benchmark, surpassing GPT-4o and the best open-source MLLM, Qwen2VL-72B, by 7.15\% and 11.72\% OA, respectively. These results demonstrate the effectiveness of our MMVM SFT dataset and our novel technical designs. Code, benchmark, dataset, and models will be released.
>
---
#### [replaced 044] On the Natural Robustness of Vision-Language Models Against Visual Perception Attacks in Autonomous Driving
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.11472v2](http://arxiv.org/pdf/2506.11472v2)**

> **作者:** Pedram MohajerAnsari; Amir Salarpour; Michael Kühr; Siyu Huang; Mohammad Hamad; Sebastian Steinhorst; Habeeb Olufowobi; Mert D. Pesé
>
> **摘要:** Autonomous vehicles (AVs) rely on deep neural networks (DNNs) for critical tasks such as traffic sign recognition (TSR), automated lane centering (ALC), and vehicle detection (VD). However, these models are vulnerable to attacks that can cause misclassifications and compromise safety. Traditional defense mechanisms, including adversarial training, often degrade benign accuracy and fail to generalize against unseen attacks. In this work, we introduce Vehicle Vision Language Models (V2LMs), fine-tuned vision-language models specialized for AV perception. Our findings demonstrate that V2LMs inherently exhibit superior robustness against unseen attacks without requiring adversarial training, maintaining significantly higher accuracy than conventional DNNs under adversarial conditions. We evaluate two deployment strategies: Solo Mode, where individual V2LMs handle specific perception tasks, and Tandem Mode, where a single unified V2LM is fine-tuned for multiple tasks simultaneously. Experimental results reveal that DNNs suffer performance drops of 33% to 46% under attacks, whereas V2LMs maintain adversarial accuracy with reductions of less than 8% on average. The Tandem Mode further offers a memory-efficient alternative while achieving comparable robustness to Solo Mode. We also explore integrating V2LMs as parallel components to AV perception to enhance resilience against adversarial threats. Our results suggest that V2LMs offer a promising path toward more secure and resilient AV perception systems.
>
---
#### [replaced 045] Sparse Autoencoder as a Zero-Shot Classifier for Concept Erasing in Text-to-Image Diffusion Models
- **分类: cs.CV; cs.AI; cs.CR**

- **链接: [http://arxiv.org/pdf/2503.09446v3](http://arxiv.org/pdf/2503.09446v3)**

> **作者:** Zhihua Tian; Sirun Nan; Ming Xu; Shengfang Zhai; Wenjie Qu; Jian Liu; Ruoxi Jia; Jiaheng Zhang
>
> **备注:** 25 pages
>
> **摘要:** Text-to-image (T2I) diffusion models have achieved remarkable progress in generating high-quality images but also raise people's concerns about generating harmful or misleading content. While extensive approaches have been proposed to erase unwanted concepts without requiring retraining from scratch, they inadvertently degrade performance on normal generation tasks. In this work, we propose Interpret then Deactivate (ItD), a novel framework to enable precise concept removal in T2I diffusion models while preserving overall performance. ItD first employs a sparse autoencoder (SAE) to interpret each concept as a combination of multiple features. By permanently deactivating the specific features associated with target concepts, we repurpose SAE as a zero-shot classifier that identifies whether the input prompt includes target concepts, allowing selective concept erasure in diffusion models. Moreover, we demonstrate that ItD can be easily extended to erase multiple concepts without requiring further training. Comprehensive experiments across celebrity identities, artistic styles, and explicit content demonstrate ItD's effectiveness in eliminating targeted concepts without interfering with normal concept generation. Additionally, ItD is also robust against adversarial prompts designed to circumvent content filters. Code is available at: https://github.com/NANSirun/Interpret-then-deactivate.
>
---
#### [replaced 046] Geometric Constraints in Deep Learning Frameworks: A Survey
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2403.12431v2](http://arxiv.org/pdf/2403.12431v2)**

> **作者:** Vibhas K Vats; David J Crandall
>
> **备注:** Published at ACM Surveys
>
> **摘要:** Stereophotogrammetry is an established technique for scene understanding. Its origins go back to at least the 1800s when people first started to investigate using photographs to measure the physical properties of the world. Since then, thousands of approaches have been explored. The classic geometric technique of Shape from Stereo is built on using geometry to define constraints on scene and camera deep learning without any attempt to explicitly model the geometry. In this survey, we explore geometry-inspired deep learning-based frameworks. We compare and contrast geometry enforcing constraints integrated into deep learning frameworks for depth estimation and other closely related vision tasks. We present a new taxonomy for prevalent geometry enforcing constraints used in modern deep learning frameworks. We also present insightful observations and potential future research directions.
>
---
#### [replaced 047] AI-GenBench: A New Ongoing Benchmark for AI-Generated Image Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.20865v2](http://arxiv.org/pdf/2504.20865v2)**

> **作者:** Lorenzo Pellegrini; Davide Cozzolino; Serafino Pandolfini; Davide Maltoni; Matteo Ferrara; Luisa Verdoliva; Marco Prati; Marco Ramilli
>
> **备注:** Accepted at Verimedia workshop, IJCNN 2025. 9 pages, 6 figures, 4 tables, code available: https://github.com/MI-BioLab/AI-GenBench
>
> **摘要:** The rapid advancement of generative AI has revolutionized image creation, enabling high-quality synthesis from text prompts while raising critical challenges for media authenticity. We present Ai-GenBench, a novel benchmark designed to address the urgent need for robust detection of AI-generated images in real-world scenarios. Unlike existing solutions that evaluate models on static datasets, Ai-GenBench introduces a temporal evaluation framework where detection methods are incrementally trained on synthetic images, historically ordered by their generative models, to test their ability to generalize to new generative models, such as the transition from GANs to diffusion models. Our benchmark focuses on high-quality, diverse visual content and overcomes key limitations of current approaches, including arbitrary dataset splits, unfair comparisons, and excessive computational demands. Ai-GenBench provides a comprehensive dataset, a standardized evaluation protocol, and accessible tools for both researchers and non-experts (e.g., journalists, fact-checkers), ensuring reproducibility while maintaining practical training requirements. By establishing clear evaluation rules and controlled augmentation strategies, Ai-GenBench enables meaningful comparison of detection methods and scalable solutions. Code and data are publicly available to ensure reproducibility and to support the development of robust forensic detectors to keep pace with the rise of new synthetic generators.
>
---
#### [replaced 048] ROCKET-2: Steering Visuomotor Policy via Cross-View Goal Alignment
- **分类: cs.AI; cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.02505v2](http://arxiv.org/pdf/2503.02505v2)**

> **作者:** Shaofei Cai; Zhancun Mu; Anji Liu; Yitao Liang
>
> **摘要:** We aim to develop a goal specification method that is semantically clear, spatially sensitive, domain-agnostic, and intuitive for human users to guide agent interactions in 3D environments. Specifically, we propose a novel cross-view goal alignment framework that allows users to specify target objects using segmentation masks from their camera views rather than the agent's observations. We highlight that behavior cloning alone fails to align the agent's behavior with human intent when the human and agent camera views differ significantly. To address this, we introduce two auxiliary objectives: cross-view consistency loss and target visibility loss, which explicitly enhance the agent's spatial reasoning ability. According to this, we develop ROCKET-2, a state-of-the-art agent trained in Minecraft, achieving an improvement in the efficiency of inference 3x to 6x compared to ROCKET-1. We show that ROCKET-2 can directly interpret goals from human camera views, enabling better human-agent interaction. Remarkably, ROCKET-2 demonstrates zero-shot generalization capabilities: despite being trained exclusively on the Minecraft dataset, it can adapt and generalize to other 3D environments like Doom, DMLab, and Unreal through a simple action space mapping.
>
---
#### [replaced 049] Revisiting Likelihood-Based Out-of-Distribution Detection by Modeling Representations
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.07793v2](http://arxiv.org/pdf/2504.07793v2)**

> **作者:** Yifan Ding; Arturas Aleksandraus; Amirhossein Ahmadian; Jonas Unger; Fredrik Lindsten; Gabriel Eilertsen
>
> **备注:** Scandinavian Conference on Image Analysis 2025 (oral)
>
> **摘要:** Out-of-distribution (OOD) detection is critical for ensuring the reliability of deep learning systems, particularly in safety-critical applications. Likelihood-based deep generative models have historically faced criticism for their unsatisfactory performance in OOD detection, often assigning higher likelihood to OOD data than in-distribution samples when applied to image data. In this work, we demonstrate that likelihood is not inherently flawed. Rather, several properties in the images space prohibit likelihood as a valid detection score. Given a sufficiently good likelihood estimator, specifically using the probability flow formulation of a diffusion model, we show that likelihood-based methods can still perform on par with state-of-the-art methods when applied in the representation space of pre-trained encoders. The code of our work can be found at $\href{https://github.com/limchaos/Likelihood-OOD.git}{\texttt{https://github.com/limchaos/Likelihood-OOD.git}}$.
>
---
#### [replaced 050] DArFace: Deformation Aware Robustness for Low Quality Face Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08423v2](http://arxiv.org/pdf/2505.08423v2)**

> **作者:** Sadaf Gulshad; Abdullah Aldahlawi Thakaa
>
> **摘要:** Facial recognition systems have achieved remarkable success by leveraging deep neural networks, advanced loss functions, and large-scale datasets. However, their performance often deteriorates in real-world scenarios involving low-quality facial images. Such degradations, common in surveillance footage or standoff imaging include low resolution, motion blur, and various distortions, resulting in a substantial domain gap from the high-quality data typically used during training. While existing approaches attempt to address robustness by modifying network architectures or modeling global spatial transformations, they frequently overlook local, non-rigid deformations that are inherently present in real-world settings. In this work, we introduce DArFace, a Deformation-Aware robust Face recognition framework that enhances robustness to such degradations without requiring paired high- and low-quality training samples. Our method adversarially integrates both global transformations (e.g., rotation, translation) and local elastic deformations during training to simulate realistic low-quality conditions. Moreover, we introduce a contrastive objective to enforce identity consistency across different deformed views. Extensive evaluations on low-quality benchmarks including TinyFace, IJB-B, and IJB-C demonstrate that DArFace surpasses state-of-the-art methods, with significant gains attributed to the inclusion of local deformation modeling.The code is available at the following https://github.com/sadafgulshad1/DArFace
>
---
#### [replaced 051] Empowering Bridge Digital Twins by Bridging the Data Gap with a Unified Synthesis Framework
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.05814v2](http://arxiv.org/pdf/2507.05814v2)**

> **作者:** Wang Wang; Mingyu Shi; Jun Jiang; Wenqian Ma; Chong Liu; Yasutaka Narazaki; Xuguang Wang
>
> **备注:** 18 pages, 10 figures
>
> **摘要:** As critical transportation infrastructure, bridges face escalating challenges from aging and deterioration, while traditional manual inspection methods suffer from low efficiency. Although 3D point cloud technology provides a new data-driven paradigm, its application potential is often constrained by the incompleteness of real-world data, which results from missing labels and scanning occlusions. To overcome the bottleneck of insufficient generalization in existing synthetic data methods, this paper proposes a systematic framework for generating 3D bridge data. This framework can automatically generate complete point clouds featuring component-level instance annotations, high-fidelity color, and precise normal vectors. It can be further extended to simulate the creation of diverse and physically realistic incomplete point clouds, designed to support the training of segmentation and completion networks, respectively. Experiments demonstrate that a PointNet++ model trained with our synthetic data achieves a mean Intersection over Union (mIoU) of 84.2% in real-world bridge semantic segmentation. Concurrently, a fine-tuned KT-Net exhibits superior performance on the component completion task. This research offers an innovative methodology and a foundational dataset for the 3D visual analysis of bridge structures, holding significant implications for advancing the automated management and maintenance of infrastructure.
>
---
#### [replaced 052] Hespi: A pipeline for automatically detecting information from hebarium specimen sheets
- **分类: cs.CV; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2410.08740v2](http://arxiv.org/pdf/2410.08740v2)**

> **作者:** Robert Turnbull; Emily Fitzgerald; Karen Thompson; Joanne L. Birch
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** Specimen-associated biodiversity data are crucial for biological, environmental, and conservation sciences. A rate shift is needed to extract data from specimen images efficiently, moving beyond human-mediated transcription. We developed `Hespi' (HErbarium Specimen sheet PIpeline) using advanced computer vision techniques to extract pre-catalogue data from primary specimen labels on herbarium specimens. Hespi integrates two object detection models: one for detecting the components of the sheet and another for fields on the primary primary specimen label. It classifies labels as printed, typed, handwritten, or mixed and uses Optical Character Recognition (OCR) and Handwritten Text Recognition (HTR) for extraction. The text is then corrected against authoritative taxon databases and refined using a multimodal Large Language Model (LLM). Hespi accurately detects and extracts text from specimen sheets across international herbaria, and its modular design allows users to train and integrate custom models.
>
---
#### [replaced 053] Revisiting Automatic Data Curation for Vision Foundation Models in Digital Pathology
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.18709v2](http://arxiv.org/pdf/2503.18709v2)**

> **作者:** Boqi Chen; Cédric Vincent-Cuaz; Lydia A. Schoenpflug; Manuel Madeira; Lisa Fournier; Vaishnavi Subramanian; Sonali Andani; Samuel Ruiperez-Campillo; Julia E. Vogt; Raphaëlle Luisier; Dorina Thanou; Viktor H. Koelzer; Pascal Frossard; Gabriele Campanella; Gunnar Rätsch
>
> **备注:** MICCAI 2025
>
> **摘要:** Vision foundation models (FMs) are accelerating the development of digital pathology algorithms and transforming biomedical research. These models learn, in a self-supervised manner, to represent histological features in highly heterogeneous tiles extracted from whole-slide images (WSIs) of real-world patient samples. The performance of these FMs is significantly influenced by the size, diversity, and balance of the pre-training data. However, data selection has been primarily guided by expert knowledge at the WSI level, focusing on factors such as disease classification and tissue types, while largely overlooking the granular details available at the tile level. In this paper, we investigate the potential of unsupervised automatic data curation at the tile-level, taking into account 350 million tiles. Specifically, we apply hierarchical clustering trees to pre-extracted tile embeddings, allowing us to sample balanced datasets uniformly across the embedding space of the pretrained FM. We further identify these datasets are subject to a trade-off between size and balance, potentially compromising the quality of representations learned by FMs, and propose tailored batch sampling strategies to mitigate this effect. We demonstrate the effectiveness of our method through improved performance on a diverse range of clinically relevant downstream tasks.
>
---
#### [replaced 054] Reconstructing Satellites in 3D from Amateur Telescope Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2404.18394v4](http://arxiv.org/pdf/2404.18394v4)**

> **作者:** Zhiming Chang; Boyang Liu; Yifei Xia; Youming Guo; Boxin Shi; He Sun
>
> **摘要:** Monitoring space objects is crucial for space situational awareness, yet reconstructing 3D satellite models from ground-based telescope images is challenging due to atmospheric turbulence, long observation distances, limited viewpoints, and low signal-to-noise ratios. In this paper, we propose a novel computational imaging framework that overcomes these obstacles by integrating a hybrid image pre-processing pipeline with a joint pose estimation and 3D reconstruction module based on controlled Gaussian Splatting (GS) and Branch-and-Bound (BnB) search. We validate our approach on both synthetic satellite datasets and on-sky observations of China's Tiangong Space Station and the International Space Station, achieving robust 3D reconstructions of low-Earth orbit satellites from ground-based data. Quantitative evaluations using SSIM, PSNR, LPIPS, and Chamfer Distance demonstrate that our method outperforms state-of-the-art NeRF-based approaches, and ablation studies confirm the critical role of each component. Our framework enables high-fidelity 3D satellite monitoring from Earth, offering a cost-effective alternative for space situational awareness. Project page: https://ai4scientificimaging.org/ReconstructingSatellites
>
---
#### [replaced 055] Tissue Concepts v2: A Supervised Foundation Model For Whole Slide Images
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.05742v2](http://arxiv.org/pdf/2507.05742v2)**

> **作者:** Till Nicke; Daniela Schacherer; Jan Raphael Schäfer; Natalia Artysh; Antje Prasse; André Homeyer; Andrea Schenk; Henning Höfener; Johannes Lotz
>
> **摘要:** Foundation models (FMs) are transforming the field of computational pathology by offering new approaches to analyzing histopathology images. Typically relying on weeks of training on large databases, the creation of FMs is a resource-intensive process in many ways. In this paper, we introduce the extension of our supervised foundation model, Tissue Concepts, to whole slide images, called Tissue Concepts v2 (TCv2), a supervised foundation model for whole slide images to address the issue above. TCv2 uses supervised, end-to-end multitask learning on slide-level labels. Training TCv2 uses a fraction of the training resources compared to self-supervised training. The presented model shows superior performance compared to SSL-trained models in cancer subtyping benchmarks and is fully trained on freely available data. Furthermore, a shared trained attention module provides an additional layer of explainability across different tasks.
>
---
#### [replaced 056] TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.04709v2](http://arxiv.org/pdf/2411.04709v2)**

> **作者:** Wenhao Wang; Yi Yang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Video generation models are revolutionizing content creation, with image-to-video models drawing increasing attention due to their enhanced controllability, visual consistency, and practical applications. However, despite their popularity, these models rely on user-provided text and image prompts, and there is currently no dedicated dataset for studying these prompts. In this paper, we introduce TIP-I2V, the first large-scale dataset of over 1.70 million unique user-provided Text and Image Prompts specifically for Image-to-Video generation. Additionally, we provide the corresponding generated videos from five state-of-the-art image-to-video models. We begin by outlining the time-consuming and costly process of curating this large-scale dataset. Next, we compare TIP-I2V to two popular prompt datasets, VidProM (text-to-video) and DiffusionDB (text-to-image), highlighting differences in both basic and semantic information. This dataset enables advancements in image-to-video research. For instance, to develop better models, researchers can use the prompts in TIP-I2V to analyze user preferences and evaluate the multi-dimensional performance of their trained models; and to enhance model safety, they may focus on addressing the misinformation issue caused by image-to-video models. The new research inspired by TIP-I2V and the differences with existing datasets emphasize the importance of a specialized image-to-video prompt dataset. The project is available at https://tip-i2v.github.io.
>
---
#### [replaced 057] DilateQuant: Accurate and Efficient Diffusion Quantization via Weight Dilation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.14307v3](http://arxiv.org/pdf/2409.14307v3)**

> **作者:** Xuewen Liu; Zhikai Li; Minhao Jiang; Mengjuan Chen; Jianquan Li; Qingyi Gu
>
> **备注:** ACMMM 2025
>
> **摘要:** Model quantization is a promising method for accelerating and compressing diffusion models. Nevertheless, since post-training quantization (PTQ) fails catastrophically at low-bit cases, quantization-aware training (QAT) is essential. Unfortunately, the wide range and time-varying activations in diffusion models sharply increase the complexity of quantization, making existing QAT methods inefficient. Equivalent scaling can effectively reduce activation range, but previous methods remain the overall quantization error unchanged. More critically, these methods significantly disrupt the original weight distribution, resulting in poor weight initialization and challenging convergence during QAT training. In this paper, we propose a novel QAT framework for diffusion models, called DilateQuant. Specifically, we propose Weight Dilation (WD) that maximally dilates the unsaturated in-channel weights to a constrained range through equivalent scaling. WD decreases the activation range while preserving the original weight range, which steadily reduces the quantization error and ensures model convergence. To further enhance accuracy and efficiency, we design a Temporal Parallel Quantizer (TPQ) to address the time-varying activations and introduce a Block-wise Knowledge Distillation (BKD) to reduce resource consumption in training. Extensive experiments demonstrate that DilateQuant significantly outperforms existing methods in terms of accuracy and efficiency. Code is available at http://github.com/BienLuky/DilateQuant .
>
---
#### [replaced 058] StixelNExT: Toward Monocular Low-Weight Perception for Object Segmentation and Free Space Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.08277v2](http://arxiv.org/pdf/2407.08277v2)**

> **作者:** Marcel Vosshans; Omar Ait-Aider; Youcef Mezouar; Markus Enzweiler
>
> **备注:** Accepted Conference Paper, IEEE IV 2024
>
> **摘要:** In this work, we present a novel approach for general object segmentation from a monocular image, eliminating the need for manually labeled training data and enabling rapid, straightforward training and adaptation with minimal data. Our model initially learns from LiDAR during the training process, which is subsequently removed from the system, allowing it to function solely on monocular imagery. This study leverages the concept of the Stixel-World to recognize a medium level representation of its surroundings. Our network directly predicts a 2D multi-layer Stixel-World and is capable of recognizing and locating multiple, superimposed objects within an image. Due to the scarcity of comparable works, we have divided the capabilities into modules and present a free space detection in our experiments section. Furthermore, we introduce an improved method for generating Stixels from LiDAR data, which we use as ground truth for our network.
>
---
#### [replaced 059] ADPv2: A Hierarchical Histological Tissue Type-Annotated Dataset for Potential Biomarker Discovery of Colorectal Disease
- **分类: eess.IV; cs.CV; cs.LG; q-bio.QM; I.2.10; I.2.1**

- **链接: [http://arxiv.org/pdf/2507.05656v2](http://arxiv.org/pdf/2507.05656v2)**

> **作者:** Zhiyuan Yang; Kai Li; Sophia Ghamoshi Ramandi; Patricia Brassard; Hakim Khellaf; Vincent Quoc-Huy Trinh; Jennifer Zhang; Lina Chen; Corwyn Rowsell; Sonal Varma; Kostas Plataniotis; Mahdi S. Hosseini
>
> **摘要:** Computational pathology (CoPath) leverages histopathology images to enhance diagnostic precision and reproducibility in clinical pathology. However, publicly available datasets for CoPath that are annotated with extensive histological tissue type (HTT) taxonomies at a granular level remain scarce due to the significant expertise and high annotation costs required. Existing datasets, such as the Atlas of Digital Pathology (ADP), address this by offering diverse HTT annotations generalized to multiple organs, but limit the capability for in-depth studies on specific organ diseases. Building upon this foundation, we introduce ADPv2, a novel dataset focused on gastrointestinal histopathology. Our dataset comprises 20,004 image patches derived from healthy colon biopsy slides, annotated according to a hierarchical taxonomy of 32 distinct HTTs of 3 levels. Furthermore, we train a multilabel representation learning model following a two-stage training procedure on our ADPv2 dataset. We leverage the VMamba architecture and achieving a mean average precision (mAP) of 0.88 in multilabel classification of colon HTTs. Finally, we show that our dataset is capable of an organ-specific in-depth study for potential biomarker discovery by analyzing the model's prediction behavior on tissues affected by different colon diseases, which reveals statistical patterns that confirm the two pathological pathways of colon cancer development. Our dataset is publicly available at https://zenodo.org/records/15307021
>
---
#### [replaced 060] Infrared and visible Image Fusion with Language-driven Loss in CLIP Embedding Space
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2402.16267v2](http://arxiv.org/pdf/2402.16267v2)**

> **作者:** Yuhao Wang; Lingjuan Miao; Zhiqiang Zhou; Lei Zhang; Yajun Qiao
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Infrared-visible image fusion (IVIF) has attracted much attention owing to the highly-complementary properties of the two image modalities. Due to the lack of ground-truth fused images, the fusion output of current deep-learning based methods heavily depends on the loss functions defined mathematically. As it is hard to well mathematically define the fused image without ground truth, the performance of existing fusion methods is limited. In this paper, we first propose to use natural language to express the objective of IVIF, which can avoid the explicit mathematical modeling of fusion output in current losses, and make full use of the advantage of language expression to improve the fusion performance. For this purpose, we present a comprehensive language-expressed fusion objective, and encode relevant texts into the multi-modal embedding space using CLIP. A language-driven fusion model is then constructed in the embedding space, by establishing the relationship among the embedded vectors to represent the fusion objective and input image modalities. Finally, a language-driven loss is derived to make the actual IVIF aligned with the embedded language-driven fusion model via supervised training. Experiments show that our method can obtain much better fusion results than existing techniques.
>
---
#### [replaced 061] 3DPortraitGAN: Learning One-Quarter Headshot 3D GANs from a Single-View Portrait Dataset with Diverse Body Poses
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2307.14770v3](http://arxiv.org/pdf/2307.14770v3)**

> **作者:** Yiqian Wu; Hao Xu; Xiangjun Tang; Yue Shangguan; Hongbo Fu; Xiaogang Jin
>
> **备注:** Accepted for publication in IEEE Transactions on Circuits and Systems for Video Technology
>
> **摘要:** 3D-aware face generators are typically trained on 2D real-life face image datasets that primarily consist of near-frontal face data, and as such, they are unable to construct one-quarter headshot 3D portraits with complete head, neck, and shoulder geometry. Two reasons account for this issue: First, existing facial recognition methods struggle with extracting facial data captured from large camera angles or back views. Second, it is challenging to learn a distribution of 3D portraits covering the one-quarter headshot region from single-view data due to significant geometric deformation caused by diverse body poses. To this end, we first create the dataset 360{\deg}-Portrait-HQ (360{\deg}PHQ for short) which consists of high-quality single-view real portraits annotated with a variety of camera parameters (the yaw angles span the entire 360{\deg} range) and body poses. We then propose 3DPortraitGAN, the first 3D-aware one-quarter headshot portrait generator that learns a canonical 3D avatar distribution from the 360{\deg}PHQ dataset with body pose self-learning. Our model can generate view-consistent portrait images from all camera angles with a canonical one-quarter headshot 3D representation. Our experiments show that the proposed framework can accurately predict portrait body poses and generate view-consistent, realistic portrait images with complete geometry from all camera angles.
>
---
#### [replaced 062] From Blurry to Brilliant Detection: YOLO-Based Aerial Object Detection with Super Resolution
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2401.14661v2](http://arxiv.org/pdf/2401.14661v2)**

> **作者:** Ragib Amin Nihal; Benjamin Yen; Takeshi Ashizawa; Katsutoshi Itoyama; Kazuhiro Nakadai
>
> **摘要:** Aerial object detection presents challenges from small object sizes, high density clustering, and image quality degradation from distance and motion blur. These factors create an information bottleneck where limited pixel representation cannot encode sufficient discriminative features. B2BDet addresses this with a two-stage framework that applies domain-specific super-resolution during inference, followed by detection using an enhanced YOLOv5 architecture. Unlike training-time super-resolution approaches that enhance learned representations, our method recovers visual information from each input image. The approach combines aerial-optimized SRGAN fine-tuning with architectural innovations including an Efficient Attention Module (EAM) and Cross-Layer Feature Pyramid Network (CLFPN). Evaluation across four aerial datasets shows performance gains, with VisDrone achieving 52.5% mAP using only 27.7M parameters. Ablation studies show that super-resolution preprocessing contributes +2.6% mAP improvement while architectural enhancements add +2.9%, yielding +5.5% total improvement over baseline YOLOv5. The method achieves computational efficiency with 53.8% parameter reduction compared to recent approaches while achieving strong small object detection performance.
>
---
#### [replaced 063] AHCPTQ: Accurate and Hardware-Compatible Post-Training Quantization for Segment Anything Model
- **分类: cs.CV; cs.AR; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.03088v2](http://arxiv.org/pdf/2503.03088v2)**

> **作者:** Wenlun Zhang; Yunshan Zhong; Shimpei Ando; Kentaro Yoshioka
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** The Segment Anything Model (SAM) has demonstrated strong versatility across various visual tasks. However, its large storage requirements and high computational cost pose challenges for practical deployment. Post-training quantization (PTQ) has emerged as an effective strategy for efficient deployment, but we identify two key challenges in SAM that hinder the effectiveness of existing PTQ methods: the heavy-tailed and skewed distribution of post-GELU activations, and significant inter-channel variation in linear projection activations. To address these challenges, we propose AHCPTQ, an accurate and hardware-efficient PTQ method for SAM. AHCPTQ introduces hardware-compatible Hybrid Log-Uniform Quantization (HLUQ) to manage post-GELU activations, employing log2 quantization for dense small values and uniform quantization for sparse large values to enhance quantization resolution. Additionally, AHCPTQ incorporates Channel-Aware Grouping (CAG) to mitigate inter-channel variation by progressively clustering activation channels with similar distributions, enabling them to share quantization parameters and improving hardware efficiency. The combination of HLUQ and CAG not only enhances quantization effectiveness but also ensures compatibility with efficient hardware execution. For instance, under the W4A4 configuration on the SAM-L model, AHCPTQ achieves 36.6% mAP on instance segmentation with the DINO detector, while achieving a 7.89x speedup and 8.64x energy efficiency over its floating-point counterpart in FPGA implementation.
>
---
#### [replaced 064] Mask-Guided Attention U-Net for Enhanced Neonatal Brain Extraction and Image Preprocessing
- **分类: eess.IV; cs.CV; stat.CO**

- **链接: [http://arxiv.org/pdf/2406.17709v2](http://arxiv.org/pdf/2406.17709v2)**

> **作者:** Bahram Jafrasteh; Simon Pedro Lubian-Lopez; Emiliano Trimarco; Macarena Roman Ruiz; Carmen Rodriguez Barrios; Yolanda Marin Almagro; Isabel Benavente-Fernandez
>
> **摘要:** In this study, we introduce MGA-Net, a novel mask-guided attention neural network, which extends the U-net model for precision neonatal brain imaging. MGA-Net is designed to extract the brain from other structures and reconstruct high-quality brain images. The network employs a common encoder and two decoders: one for brain mask extraction and the other for brain region reconstruction. A key feature of MGA-Net is its high-level mask-guided attention module, which leverages features from the brain mask decoder to enhance image reconstruction. To enable the same encoder and decoder to process both MRI and ultrasound (US) images, MGA-Net integrates sinusoidal positional encoding. This encoding assigns distinct positional values to MRI and US images, allowing the model to effectively learn from both modalities. Consequently, features learned from a single modality can aid in learning a modality with less available data, such as US. We extensively validated the proposed MGA-Net on diverse datasets from varied clinical settings and neonatal age groups. The metrics used for assessment included the DICE similarity coefficient, recall, and accuracy for image segmentation; structural similarity for image reconstruction; and root mean squared error for total brain volume estimation from 3D ultrasound images. Our results demonstrate that MGA-Net significantly outperforms traditional methods, offering superior performance in brain extraction and segmentation while achieving high precision in image reconstruction and volumetric analysis. Thus, MGA-Net represents a robust and effective preprocessing tool for MRI and 3D ultrasound images, marking a significant advance in neuroimaging that enhances both research and clinical diagnostics in the neonatal period and beyond.
>
---
#### [replaced 065] UWarp: A Whole Slide Image Registration Pipeline to Characterize Scanner-Induced Local Domain Shift
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.20653v2](http://arxiv.org/pdf/2503.20653v2)**

> **作者:** Antoine Schieb; Bilal Hadjadji; Natalia Fernanda Valderrama; Daniel Tshokola Mweze; Valentin Derangère; Laurent Arnould; Sylvain Ladoire; Alain Lalande; Alessio Fiorin; Carlos López Pablo; Noèlia Gallardo Borràs; Shrief Abdelazeez; Vincenzo Della Mea; Anna Korzynska; Nathan Vinçon; Louis-Oscar Morel
>
> **备注:** preprint
>
> **摘要:** Histopathology slide digitization introduces scanner-induced domain shift that can significantly impact computational pathology models based on deep learning methods. In the state-of-the-art, this shift is often characterized at a broad scale (slide-level or dataset-level) but not patch-level, which limits our comprehension of the impact of localized tissue characteristics on the accuracy of the deep learning models. To address this challenge, we present a domain shift analysis framework based on UWarp, a novel registration tool designed to accurately align histological slides scanned under varying conditions. UWarp employs a hierarchical registration approach, combining global affine transformations with fine-grained local corrections to achieve robust tissue patch alignment. We evaluate UWarp using two private datasets, CypathLung and BosomShieldBreast, containing whole slide images scanned by multiple devices. Our experiments demonstrate that UWarp outperforms existing open-source registration methods, achieving a median target registration error (TRE) of less than 4 pixels (<1 micrometer at 40x magnification) while significantly reducing computational time. Additionally, we apply UWarp to characterize scanner-induced local domain shift in the predictions of Breast-NEOprAIdict, a deep learning model for breast cancer pathological response prediction. We find that prediction variability is strongly correlated with tissue density on a given patch. Our findings highlight the importance of localized domain shift analysis and suggest that UWarp can serve as a valuable tool for improving model robustness and domain adaptation strategies in computational pathology.
>
---
#### [replaced 066] QuEST: Low-bit Diffusion Model Quantization via Efficient Selective Finetuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2402.03666v5](http://arxiv.org/pdf/2402.03666v5)**

> **作者:** Haoxuan Wang; Yuzhang Shang; Zhihang Yuan; Junyi Wu; Junchi Yan; Yan Yan
>
> **备注:** ICCV 2025. Code is available at https://github.com/hatchetProject/QuEST
>
> **摘要:** The practical deployment of diffusion models is still hindered by the high memory and computational overhead. Although quantization paves a way for model compression and acceleration, existing methods face challenges in achieving low-bit quantization efficiently. In this paper, we identify imbalanced activation distributions as a primary source of quantization difficulty, and propose to adjust these distributions through weight finetuning to be more quantization-friendly. We provide both theoretical and empirical evidence supporting finetuning as a practical and reliable solution. Building on this approach, we further distinguish two critical types of quantized layers: those responsible for retaining essential temporal information and those particularly sensitive to bit-width reduction. By selectively finetuning these layers under both local and global supervision, we mitigate performance degradation while enhancing quantization efficiency. Our method demonstrates its efficacy across three high-resolution image generation tasks, obtaining state-of-the-art performance across multiple bit-width settings.
>
---
#### [replaced 067] HyperGCT: A Dynamic Hyper-GNN-Learned Geometric Constraint for 3D Registration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.02195v2](http://arxiv.org/pdf/2503.02195v2)**

> **作者:** Xiyu Zhang; Jiayi Ma; Jianwei Guo; Wei Hu; Zhaoshuai Qi; Fei Hui; Jiaqi Yang; Yanning Zhang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Geometric constraints between feature matches are critical in 3D point cloud registration problems. Existing approaches typically model unordered matches as a consistency graph and sample consistent matches to generate hypotheses. However, explicit graph construction introduces noise, posing great challenges for handcrafted geometric constraints to render consistency. To overcome this, we propose HyperGCT, a flexible dynamic Hyper-GNN-learned geometric ConstrainT that leverages high-order consistency among 3D correspondences. To our knowledge, HyperGCT is the first method that mines robust geometric constraints from dynamic hypergraphs for 3D registration. By dynamically optimizing the hypergraph through vertex and edge feature aggregation, HyperGCT effectively captures the correlations among correspondences, leading to accurate hypothesis generation. Extensive experiments on 3DMatch, 3DLoMatch, KITTI-LC, and ETH show that HyperGCT achieves state-of-the-art performance. Furthermore, HyperGCT is robust to graph noise, demonstrating a significant advantage in terms of generalization.
>
---
#### [replaced 068] Correlative and Discriminative Label Grouping for Multi-Label Visual Prompt Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.09990v2](http://arxiv.org/pdf/2504.09990v2)**

> **作者:** LeiLei Ma; Shuo Xu; MingKun Xie; Lei Wang; Dengdi Sun; Haifeng Zhao
>
> **备注:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2025
>
> **摘要:** Modeling label correlations has always played a pivotal role in multi-label image classification (MLC), attracting significant attention from researchers. However, recent studies have overemphasized co-occurrence relationships among labels, which can lead to overfitting risk on this overemphasis, resulting in suboptimal models. To tackle this problem, we advocate for balancing correlative and discriminative relationships among labels to mitigate the risk of overfitting and enhance model performance. To this end, we propose the Multi-Label Visual Prompt Tuning framework, a novel and parameter-efficient method that groups classes into multiple class subsets according to label co-occurrence and mutual exclusivity relationships, and then models them respectively to balance the two relationships. In this work, since each group contains multiple classes, multiple prompt tokens are adopted within Vision Transformer (ViT) to capture the correlation or discriminative label relationship within each group, and effectively learn correlation or discriminative representations for class subsets. On the other hand, each group contains multiple group-aware visual representations that may correspond to multiple classes, and the mixture of experts (MoE) model can cleverly assign them from the group-aware to the label-aware, adaptively obtaining label-aware representation, which is more conducive to classification. Experiments on multiple benchmark datasets show that our proposed approach achieves competitive results and outperforms SOTA methods on multiple pre-trained models.
>
---
#### [replaced 069] Enhancing Plasticity for First Session Adaptation Continual Learning
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2310.11482v3](http://arxiv.org/pdf/2310.11482v3)**

> **作者:** Imad Eddine Marouf; Subhankar Roy; Stéphane Lathuilière; Enzo Tartaglione
>
> **备注:** Accepted at CoLLAs 2025, 9 pages, 4 figures
>
> **摘要:** The integration of large pre-trained models (PTMs) into Class-Incremental Learning (CIL) has facilitated the development of computationally efficient strategies such as First-Session Adaptation (FSA), which fine-tunes the model solely on the first task while keeping it frozen for subsequent tasks. Although effective in homogeneous task sequences, these approaches struggle when faced with the heterogeneity of real-world task distributions. We introduce Plasticity-Enhanced Test-Time Adaptation in Class-Incremental Learning (PLASTIC), a method that reinstates plasticity in CIL while preserving model stability. PLASTIC leverages Test-Time Adaptation (TTA) by dynamically fine-tuning LayerNorm parameters on unlabeled test data, enabling adaptability to evolving tasks and improving robustness against data corruption. To prevent TTA-induced model divergence and maintain stable learning across tasks, we introduce a teacher-student distillation framework, ensuring that adaptation remains controlled and generalizable. Extensive experiments across multiple benchmarks demonstrate that PLASTIC consistently outperforms both conventional and state-of-the-art PTM-based CIL approaches, while also exhibiting inherent robustness to data corruptions. Code is available at: https://github.com/IemProg/PLASTIC.
>
---
#### [replaced 070] Batch Normalization in Cytometry Data by kNN-Graph Preservation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2304.00050v4](http://arxiv.org/pdf/2304.00050v4)**

> **作者:** Muhammad S. Battikh; Artem Lensky
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Batch effects in high-dimensional Cytometry by Time-of-Flight (CyTOF) data pose a challenge for comparative analysis across different experimental conditions or time points. Traditional batch normalization methods may fail to preserve the complex topological structures inherent in cellular populations. In this paper, we present a residual neural network-based method for point set registration specifically tailored to address batch normalization in CyTOF data while preserving the topological structure of cellular populations. By viewing the alignment problem as the movement of cells sampled from a target distribution along a regularized displacement vector field, similar to coherent point drift (CPD), our approach introduces a Jacobian-based cost function and geometry-aware statistical distances to ensure local topology preservation. We provide justification for the k-Nearest Neighbour (kNN) graph preservation of the target data when the Jacobian cost is applied, which is crucial for maintaining biological relationships between cells. Furthermore, we introduce a stochastic approximation for high-dimensional registration, making alignment feasible for the high-dimensional space of CyTOF data. Our method is demonstrated on high-dimensional CyTOF dataset, effectively aligning distributions of cells while preserving the kNN-graph structure. This enables accurate batch normalization, facilitating reliable comparative analysis in biomedical research.
>
---
#### [replaced 071] UniF$^2$ace: Fine-grained Face Understanding and Generation with Unified Multimodal Models
- **分类: cs.CV; cs.AI; cs.LG; cs.MM**

- **链接: [http://arxiv.org/pdf/2503.08120v3](http://arxiv.org/pdf/2503.08120v3)**

> **作者:** Junzhe Li; Xuerui Qiu; Linrui Xu; Liya Guo; Delin Qu; Tingting Long; Chun Fan; Ming Li
>
> **摘要:** Unified multimodal models (UMMs) have emerged as a powerful paradigm in foundational computer vision research, demonstrating significant potential in both image understanding and generation. However, existing research in the face domain primarily focuses on $\textbf{coarse}$ facial attribute understanding, with limited capacity to handle $\textbf{fine-grained}$ facial attributes and without addressing generation capabilities. To overcome these limitations, we propose UniF$^2$ace, the first UMM tailored specifically for fine-grained face understanding and generation. In general, we train UniF$^2$ace on a self-constructed, specialized dataset utilizing two mutually beneficial diffusion techniques and a two-level mixture-of-experts architecture. Specifically, we first build a large-scale facial dataset, UniF$^2$ace-130K, which contains 130K image-text pairs with one million question-answering pairs that span a wide range of facial attributes. Second, we establish a theoretical connection between discrete diffusion score matching and masked generative models, optimizing both evidence lower bounds simultaneously, which significantly improves the model's ability to synthesize facial details. Finally, we introduce both token-level and sequence-level mixture-of-experts, enabling efficient fine-grained representation learning for both understanding and generation tasks. Extensive experiments on UniF$^2$ace-130K demonstrate that UniF$^2$ace outperforms existing UMMs and generative models, achieving superior performance across both understanding and generation tasks.
>
---
#### [replaced 072] Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2408.06079v2](http://arxiv.org/pdf/2408.06079v2)**

> **作者:** Kejia Zhang; Juanjuan Weng; Shaozi Li; Zhiming Luo
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Despite the remarkable progress of deep neural networks (DNNs) in various visual tasks, their vulnerability to adversarial examples raises significant security concerns. Recent adversarial training methods leverage inverse adversarial attacks to generate high-confidence examples, aiming to align adversarial distributions with high-confidence class regions. However, our investigation reveals that under inverse adversarial attacks, high-confidence outputs are influenced by biased feature activations, causing models to rely on background features that lack a causal relationship with the labels. This spurious correlation bias leads to overfitting irrelevant background features during adversarial training, thereby degrading the model's robust performance and generalization capabilities. To address this issue, we propose Debiased High-Confidence Adversarial Training (DHAT), a novel approach that aligns adversarial logits with debiased high-confidence logits and restores proper attention by enhancing foreground logit orthogonality. Extensive experiments demonstrate that DHAT achieves state-of-the-art robustness on both CIFAR and ImageNet-1K benchmarks, while significantly improving generalization by mitigating the feature bias inherent in inverse adversarial training approaches. Code is available at https://github.com/KejiaZhang-Robust/DHAT.
>
---
#### [replaced 073] Semantic Augmentation in Images using Language
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2404.02353v2](http://arxiv.org/pdf/2404.02353v2)**

> **作者:** Sahiti Yerramilli; Jayant Sravan Tamarapalli; Tanmay Girish Kulkarni; Jonathan Francis; Eric Nyberg
>
> **摘要:** Deep Learning models are incredibly data-hungry and require very large labeled datasets for supervised learning. As a consequence, these models often suffer from overfitting, limiting their ability to generalize to real-world examples. Recent advancements in diffusion models have enabled the generation of photorealistic images based on textual inputs. Leveraging the substantial datasets used to train these diffusion models, we propose a technique to utilize generated images to augment existing datasets. This paper explores various strategies for effective data augmentation to improve the out-of-domain generalization capabilities of deep learning models.
>
---
#### [replaced 074] PBCAT: Patch-based composite adversarial training against physically realizable attacks on object detection
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.23581v2](http://arxiv.org/pdf/2506.23581v2)**

> **作者:** Xiao Li; Yiming Zhu; Yifan Huang; Wei Zhang; Yingzhe He; Jie Shi; Xiaolin Hu
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Object detection plays a crucial role in many security-sensitive applications. However, several recent studies have shown that object detectors can be easily fooled by physically realizable attacks, \eg, adversarial patches and recent adversarial textures, which pose realistic and urgent threats. Adversarial Training (AT) has been recognized as the most effective defense against adversarial attacks. While AT has been extensively studied in the $l_\infty$ attack settings on classification models, AT against physically realizable attacks on object detectors has received limited exploration. Early attempts are only performed to defend against adversarial patches, leaving AT against a wider range of physically realizable attacks under-explored. In this work, we consider defending against various physically realizable attacks with a unified AT method. We propose PBCAT, a novel Patch-Based Composite Adversarial Training strategy. PBCAT optimizes the model by incorporating the combination of small-area gradient-guided adversarial patches and imperceptible global adversarial perturbations covering the entire image. With these designs, PBCAT has the potential to defend against not only adversarial patches but also unseen physically realizable attacks such as adversarial textures. Extensive experiments in multiple settings demonstrated that PBCAT significantly improved robustness against various physically realizable attacks over state-of-the-art defense methods. Notably, it improved the detection accuracy by 29.7\% over previous defense methods under one recent adversarial texture attack.
>
---
#### [replaced 075] RapidPoseTriangulation: Multi-view Multi-person Whole-body Human Pose Triangulation in a Millisecond
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.21692v3](http://arxiv.org/pdf/2503.21692v3)**

> **作者:** Daniel Bermuth; Alexander Poeppel; Wolfgang Reif
>
> **摘要:** The integration of multi-view imaging and pose estimation represents a significant advance in computer vision applications, offering new possibilities for understanding human movement and interactions. This work presents a new algorithm that improves multi-view multi-person pose estimation, focusing on fast triangulation speeds and good generalization capabilities. The approach extends to whole-body pose estimation, capturing details from facial expressions to finger movements across multiple individuals and viewpoints. Adaptability to different settings is demonstrated through strong performance across unseen datasets and configurations. To support further progress in this field, all of this work is publicly accessible.
>
---
#### [replaced 076] Scaling 4D Representations
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.15212v2](http://arxiv.org/pdf/2412.15212v2)**

> **作者:** João Carreira; Dilara Gokay; Michael King; Chuhan Zhang; Ignacio Rocco; Aravindh Mahendran; Thomas Albert Keck; Joseph Heyward; Skanda Koppula; Etienne Pot; Goker Erdogan; Yana Hasson; Yi Yang; Klaus Greff; Guillaume Le Moing; Sjoerd van Steenkiste; Daniel Zoran; Drew A. Hudson; Pedro Vélez; Luisa Polanía; Luke Friedman; Chris Duvarney; Ross Goroshin; Kelsey Allen; Jacob Walker; Rishabh Kabra; Eric Aboussouan; Jennifer Sun; Thomas Kipf; Carl Doersch; Viorica Pătrăucean; Dima Damen; Pauline Luc; Mehdi S. M. Sajjadi; Andrew Zisserman
>
> **摘要:** Scaling has not yet been convincingly demonstrated for pure self-supervised learning from video. However, prior work has focused evaluations on semantic-related tasks $\unicode{x2013}$ action classification, ImageNet classification, etc. In this paper we focus on evaluating self-supervised learning on non-semantic vision tasks that are more spatial (3D) and temporal (+1D = 4D), such as camera pose estimation, point and object tracking, and depth estimation. We show that by learning from very large video datasets, masked auto-encoding (MAE) with transformer video models actually scales, consistently improving performance on these 4D tasks, as model size increases from 20M all the way to the largest by far reported self-supervised video model $\unicode{x2013}$ 22B parameters. Rigorous apples-to-apples comparison with many recent image and video models demonstrates the benefits of scaling 4D representations. Pretrained models are available at https://github.com/google-deepmind/representations4d .
>
---
#### [replaced 077] From Video to EEG: Adapting Joint Embedding Predictive Architecture to Uncover Visual Concepts in Brain Signal Analysis
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.03633v3](http://arxiv.org/pdf/2507.03633v3)**

> **作者:** Amirabbas Hojjati; Lu Li; Ibrahim Hameed; Anis Yazidi; Pedro G. Lind; Rabindra Khadka
>
> **摘要:** EEG signals capture brain activity with high temporal and low spatial resolution, supporting applications such as neurological diagnosis, cognitive monitoring, and brain-computer interfaces. However, effective analysis is hindered by limited labeled data, high dimensionality, and the absence of scalable models that fully capture spatiotemporal dependencies. Existing self-supervised learning (SSL) methods often focus on either spatial or temporal features, leading to suboptimal representations. To this end, we propose EEG-VJEPA, a novel adaptation of the Video Joint Embedding Predictive Architecture (V-JEPA) for EEG classification. By treating EEG as video-like sequences, EEG-VJEPA learns semantically meaningful spatiotemporal representations using joint embeddings and adaptive masking. To our knowledge, this is the first work that exploits V-JEPA for EEG classification and explores the visual concepts learned by the model. Evaluations on the publicly available Temple University Hospital (TUH) Abnormal EEG dataset show that EEG-VJEPA outperforms existing state-of-the-art models in classification accuracy. Beyond classification accuracy, EEG-VJEPA captures physiologically relevant spatial and temporal signal patterns, offering interpretable embeddings that may support human-AI collaboration in diagnostic workflows. These findings position EEG-VJEPA as a promising framework for scalable, trustworthy EEG analysis in real-world clinical settings.
>
---
