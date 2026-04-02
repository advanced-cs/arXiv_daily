# 计算机视觉 cs.CV

- **最新发布 136 篇**

- **更新 97 篇**

## 最新发布

#### [new 001] Hierarchical Pre-Training of Vision Encoders with Large Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉-语言模型任务，旨在解决视觉特征与语言模型整合不足的问题。提出HIVE框架，通过层次化交叉注意力提升多模态对齐与特征融合效果。**

- **链接: [https://arxiv.org/pdf/2604.00086](https://arxiv.org/pdf/2604.00086)**

> **作者:** Eugene Lee; Ting-Yu Chang; Jui-Huang Tsai; Jiajie Diao; Chen-Yi Lee
>
> **备注:** 17 pages, 14 figures, accepted to Computer Vision and Pattern Recognition Conference (CVPR) Workshops 2026. 5th MMFM Workshop: What is Next in Multimodal Foundation Models?
>
> **摘要:** The field of computer vision has experienced significant advancements through scalable vision encoders and multimodal pre-training frameworks. However, existing approaches often treat vision encoders and large language models (LLMs) as independent modules, limiting the integration of hierarchical visual features. In this work, we propose HIVE (Hierarchical Pre-Training of Vision Encoders), a novel framework that enhances vision-language alignment by introducing hierarchical cross-attention between the vision encoder and LLM. Unlike conventional methods that flatten image embeddings, HIVE enables structured feature fusion across multiple layers, improving gradient flow and representation learning. To optimize this interaction, we introduce a three-stage training strategy that progressively aligns the vision encoder with the LLM, ensuring stable optimization and effective multimodal fusion. Empirical evaluations demonstrate that HIVE achieves superior performance not only in image classification but also on various vision-language tasks, outperforming self-attention-based methods in benchmarks such as MME, GQA, OK-VQA, and ScienceQA. Our results highlight the benefits of hierarchical feature integration, paving the way for more efficient and expressive vision-language models.
>
---
#### [new 002] SANA I2I: A Text Free Flow Matching Framework for Paired Image to Image Translation with a Case Study in Fetal MRI Artifact Reduction
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出SANA-I2I，一种无需文本的图像到图像生成框架，用于解决胎儿MRI运动伪影去除问题。通过配对图像学习条件流匹配模型，实现高效监督图像翻译。**

- **链接: [https://arxiv.org/pdf/2604.00298](https://arxiv.org/pdf/2604.00298)**

> **作者:** Italo Felix Santos; Gilson Antonio Giraldi; Heron Werner Junior
>
> **摘要:** We propose SANA-I2I, a text-free high-resolution image-to-image generation framework that extends the SANA family by removing textual conditioning entirely. In contrast to SanaControlNet, which combines text and image-based control, SANA-I2I relies exclusively on paired source-target images to learn a conditional flow-matching model in latent space. The model learns a conditional velocity field that maps a target image distribution to another one, enabling supervised image translation without reliance on language prompts. We evaluate the proposed approach on the challenging task of fetal MRI motion artifact reduction. To enable paired training in this application, where real paired data are difficult to acquire, we adopt a synthetic data generation strategy based on the method proposed by Duffy et al., which simulates realistic motion artifacts in fetal magnetic resonance imaging (MRI). Experimental results demonstrate that SANA-I2I effectively suppresses motion artifacts while preserving anatomical structure, achieving competitive performance few inference steps. These results highlight the efficiency and suitability of our proposed flow-based, text-free generative models for supervised image-to-image tasks in medical imaging.
>
---
#### [new 003] TF-SSD: A Strong Pipeline via Synergic Mask Filter for Training-free Co-salient Object Detection
- **分类: cs.CV**

- **简介: 该论文属于Co-salient Object Detection任务，解决训练依赖和泛化能力不足的问题。提出TF-SSD方法，结合SAM与DINO，实现无需训练的高效检测。**

- **链接: [https://arxiv.org/pdf/2604.00549](https://arxiv.org/pdf/2604.00549)**

> **作者:** Zhijin He; Shuo Jin; Siyue Yu; Shuwei Wu; Bingfeng Zhang; Li Yu; Jimin Xiao
>
> **备注:** Accepted by CVPR26
>
> **摘要:** Co-salient Object Detection (CoSOD) aims to segment salient objects that consistently appear across a group of related images. Despite the notable progress achieved by recent training-based approaches, they still remain constrained by the closed-set datasets and exhibit limited generalization. However, few studies explore the potential of Vision Foundation Models (VFMs) to address CoSOD, which demonstrate a strong generalized ability and robust saliency understanding. In this paper, we investigate and leverage VFMs for CoSOD, and further propose a novel training-free method, TF-SSD, through the synergy between SAM and DINO. Specifically, we first utilize SAM to generate comprehensive raw proposals, which serve as a candidate mask pool. Then, we introduce a quality mask generator to filter out redundant masks, thereby acquiring a refined mask set. Since this generator is built upon SAM, it inherently lacks semantic understanding of saliency. To this end, we adopt an intra-image saliency filter that employs DINO's attention maps to identify visually salient masks within individual images. Moreover, to extend saliency understanding across group images, we propose an inter-image prototype selector, which computes similarity scores among cross-image prototypes to select masks with the highest score. These selected masks serve as final predictions for CoSOD. Extensive experiments show that our TF-SSD outperforms existing methods (e.g., 13.7\% gains over the recent training-free method). Codes are available at this https URL.
>
---
#### [new 004] ACT Now: Preempting LVLM Hallucinations via Adaptive Context Integration
- **分类: cs.CV**

- **简介: 该论文属于视觉语言模型任务，解决LVLM的幻觉问题。提出ACT方法，通过自适应整合上下文信息减少幻觉，提升模型准确性。**

- **链接: [https://arxiv.org/pdf/2604.00983](https://arxiv.org/pdf/2604.00983)**

> **作者:** Bei Yan; Yuecong Min; Jie Zhang; Shiguang Shan; Xilin Chen
>
> **摘要:** Large Vision-Language Models (LVLMs) frequently suffer from severe hallucination issues. Existing mitigation strategies predominantly rely on isolated, single-step states to enhance visual focus or suppress strong linguistic priors. However, these static approaches neglect dynamic context changes across the generation process and struggles to correct inherited information loss. To address this limitation, we propose Adaptive Context inTegration (ACT), a training-free inference intervention method that mitigates hallucination through the adaptive integration of contextual information. Specifically, we first propose visual context exploration, which leverages spatio-temporal profiling to adaptively amplify attention heads responsible for visual exploration. To further facilitate vision-language alignment, we propose semantic context aggregation that marginalizes potential semantic queries to effectively aggregate visual evidence, thereby resolving the information loss caused by the discrete nature of token prediction. Extensive experiments across diverse LVLMs demonstrate that ACT significantly reduces hallucinations and achieves competitive results on both discriminative and generative benchmarks, acting as a robust and highly adaptable solution without compromising fundamental generation capabilities.
>
---
#### [new 005] RawGen: Learning Camera Raw Image Generation
- **分类: cs.CV**

- **简介: 该论文提出RawGen，解决raw图像生成与逆ISP问题，通过扩散模型实现文本到raw的生成及sRGB到raw的逆向重建。**

- **链接: [https://arxiv.org/pdf/2604.00093](https://arxiv.org/pdf/2604.00093)**

> **作者:** Dongyoung Kim; Junyong Lee; Abhijith Punnappurath; Mahmoud Afifi; Sangmin Han; Alex Levinshtein; Michael S. Brown
>
> **摘要:** Cameras capture scene-referred linear raw images, which are processed by onboard image signal processors (ISPs) into display-referred 8-bit sRGB outputs. Although raw data is more faithful for low-level vision tasks, collecting large-scale raw datasets remains a major bottleneck, as existing datasets are limited and tied to specific camera hardware. Generative models offer a promising way to address this scarcity -- however, existing diffusion frameworks are designed to synthesize photo-finished sRGB images rather than physically consistent linear representations. This paper presents RawGen, to our knowledge the first diffusion-based framework enabling text-to-raw generation for arbitrary target cameras, alongside sRGB-to-raw inversion. RawGen leverages the generative priors of large-scale sRGB diffusion models to synthesize physically meaningful linear outputs, such as CIE XYZ or camera-specific raw representations, via specialized processing in latent and pixel spaces. To handle unknown and diverse ISP pipelines and photo-finishing effects in diffusion-model training data, we build a many-to-one inverse-ISP dataset where multiple sRGB renditions of the same scene generated using diverse ISP parameters are anchored to a common scene-referred target. Fine-tuning a conditional denoiser and specialized decoder on this dataset allows RawGen to obtain camera-centric linear reconstructions that effectively invert the rendering pipeline. We demonstrate RawGen's superior performance over traditional inverse-ISP methods that assume a fixed ISP. Furthermore, we show that augmenting training pipelines with RawGen's scalable, text-driven synthetic data can benefit downstream low-level vision tasks.
>
---
#### [new 006] Lightweight Prompt-Guided CLIP Adaptation for Monocular Depth Estimation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于单目深度估计任务，旨在利用CLIP的语义特征提升深度估计精度。通过轻量级适配器模块和提示引导方法，实现高效参数适应与几何约束优化。**

- **链接: [https://arxiv.org/pdf/2604.01118](https://arxiv.org/pdf/2604.01118)**

> **作者:** Reyhaneh Ahani Manghotay; Jie Liang
>
> **备注:** 14 pages, 2 figures
>
> **摘要:** Leveraging the rich semantic features of vision-language models (VLMs) like CLIP for monocular depth estimation tasks is a promising direction, yet often requires extensive fine-tuning or lacks geometric precision. We present a parameter-efficient framework, named MoA-DepthCLIP, that adapts pretrained CLIP representations for monocular depth estimation with minimal supervision. Our method integrates a lightweight Mixture-of-Adapters (MoA) module into the pretrained Vision Transformer (ViT-B/32) backbone combined with selective fine-tuning of the final layers. This design enables spatially-aware adaptation, guided by a global semantic context vector and a hybrid prediction architecture that synergizes depth bin classification with direct regression. To enhance structural accuracy, we employ a composite loss function that enforces geometric constraints. On the NYU Depth V2 benchmark, MoA-DepthCLIP achieves competitive results, significantly outperforming the DepthCLIP baseline by improving the $\delta_1$ accuracy from 0.390 to 0.745 and reducing the RMSE from 1.176 to 0.520. These results are achieved while requiring substantially few trainable parameters, demonstrating that lightweight, prompt-guided MoA is a highly effective strategy for transferring VLM knowledge to fine-grained monocular depth estimation tasks.
>
---
#### [new 007] LinguDistill: Recovering Linguistic Ability in Vision- Language Models via Selective Cross-Modal Distillation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态学习任务，解决视觉-语言模型在适应过程中损失语言能力的问题。通过无适配器的蒸馏方法，恢复语言模型的原有能力，同时保持视觉理解性能。**

- **链接: [https://arxiv.org/pdf/2604.00829](https://arxiv.org/pdf/2604.00829)**

> **作者:** Patrick Amadeus Irawan; Erland Hilman Fuadi; Shanu Kumar; Alham Fikri Aji; Yova Kementchedjhieva
>
> **摘要:** Adapting pretrained language models (LMs) into vision-language models (VLMs) can degrade their native linguistic capability due to representation shift and cross-modal interference introduced during multimodal adaptation. Such loss is difficult to recover, even with targeted task-specific fine-tuning using standard objectives. Prior recovery approaches typically introduce additional modules that act as intermediate alignment layers to maintain or isolate modality-specific subspaces, which increases architectural complexity, adds parameters at inference time, and limits flexibility across models and settings. We propose LinguDistill, an adapter-free distillation method that restores linguistic capability by utilizing the original frozen LM as a teacher. We overcome the key challenge of enabling vision-conditioned teacher supervision by introducing layer-wise KV-cache sharing, which exposes the teacher to the student's multimodal representations without modifying the architecture of either model. We then selectively distill the teacher's strong linguistic signal on language-intensive data to recover language capability, while preserving the student's visual grounding on multimodal tasks. As a result, LinguDistill recovers $\sim$10% of the performance lost on language and knowledge benchmarks, while maintaining comparable performance on vision-heavy tasks. Our findings demonstrate that linguistic capability can be recovered without additional modules, providing an efficient and practical solution to modality-specific degradation in multimodal models.
>
---
#### [new 008] HICT: High-precision 3D CBCT reconstruction from a single X-ray
- **分类: cs.CV**

- **简介: 该论文属于3D CBCT重建任务，旨在从单张低剂量X光片高精度重建三维图像。通过两阶段方法解决几何不一致和精度不足问题。**

- **链接: [https://arxiv.org/pdf/2604.00792](https://arxiv.org/pdf/2604.00792)**

> **作者:** Wen Ma; Jiaxiang Liu; Zikai Xiao; Ziyang Wang; Feng Yang; Zuozhu Liu
>
> **摘要:** Accurate 3D dental imaging is vital for diagnosis and treatment planning, yet CBCT's high radiation dose and cost limit its accessibility. Reconstructing 3D volumes from a single low-dose panoramic X-ray is a promising alternative but remains challenging due to geometric inconsistencies and limited accuracy. We propose HiCT, a two-stage framework that first generates geometrically consistent multi-view projections from a single panoramic image using a video diffusion model, and then reconstructs high-fidelity CBCT from the projections using a ray-based dynamic attention network and an X-ray sampling strategy. To support this, we built XCT, a large-scale dataset combining public CBCT data with 500 paired PX-CBCT cases. Extensive experiments show that HiCT achieves state-of-the-art performance, delivering accurate and geometrically consistent reconstructions for clinical use.
>
---
#### [new 009] DirectFisheye-GS: Enabling Native Fisheye Input in Gaussian Splatting with Cross-View Joint Optimization
- **分类: cs.CV**

- **简介: 该论文属于3D重建任务，解决 fisheye 图像在 3DGS 中的适配问题。通过直接输入 fisheye 图像并引入跨视角优化，提升重建质量。**

- **链接: [https://arxiv.org/pdf/2604.00648](https://arxiv.org/pdf/2604.00648)**

> **作者:** Zhengxian Yang; Fei Xie; Xutao Xue; Rui Zhang; Taicheng Huang; Yang Liu; Mengqi Ji; Tao Yu
>
> **备注:** CVPR 2026
>
> **摘要:** 3D Gaussian Splatting (3DGS) has enabled efficient 3D scene reconstruction from everyday images with real-time, high-fidelity rendering, greatly advancing VR/AR applications. Fisheye cameras, with their wider field of view (FOV), promise high-quality reconstructions from fewer inputs and have recently attracted much attention. However, since 3DGS relies on rasterization, most subsequent works involving fisheye camera inputs first undistort images before training, which introduces two problems: 1) Black borders at image edges cause information loss and negate the fisheye's large FOV advantage; 2) Undistortion's stretch-and-interpolate resampling spreads each pixel's value over a larger area, diluting detail density -- causes 3DGS overfitting these low-frequency zones, producing blur and floating artifacts. In this work, we integrate fisheye camera model into the original 3DGS framework, enabling native fisheye image input for training without preprocessing. Despite correct modeling, we observed that the reconstructed scenes still exhibit floaters at image edges: Distortion increases toward the periphery, and 3DGS's original per-iteration random-selecting-view optimization ignores the cross-view correlations of a Gaussian, leading to extreme shapes (e.g., oversized or elongated) that degrade reconstruction quality. To address this, we introduce a feature-overlap-driven cross-view joint optimization strategy that establishes consistent geometric and photometric constraints across views-a technique equally applicable to existing pinhole-camera-based pipelines. Our DirectFisheye-GS matches or surpasses state-of-the-art performance on public datasets.
>
---
#### [new 010] VLM-in-the-Loop: A Plug-In Quality Assurance Module for ECG Digitization Pipelines
- **分类: cs.CV**

- **简介: 该论文属于医疗图像质量保障任务，解决ECG digitization中真实场景性能下降的问题。提出VLM-in-the-Loop模块，通过工具接地提升评估一致性与精度。**

- **链接: [https://arxiv.org/pdf/2604.00396](https://arxiv.org/pdf/2604.00396)**

> **作者:** Jiachen Li; Shihao Li; Soovadeep Bakshi; Wei Li; Dongmei Chen
>
> **摘要:** ECG digitization could unlock billions of archived clinical records, yet existing methods collapse on real-world images despite strong benchmark numbers. We introduce \textbf{VLM-in-the-Loop}, a plug-in quality assurance module that wraps any digitization backend with closed-loop VLM feedback via a standardized interface, requiring no modification to the underlying digitizer. The core mechanism is \textbf{tool grounding}: anchoring VLM assessment in quantitative evidence from domain-specific signal analysis tools. In a controlled ablation on 200 records with paired ground truth, tool grounding raises verdict consistency from 71\% to 89\% and doubles fidelity separation ($\Delta$PCC 0.03 $\rightarrow$ 0.08), with the effect replicating across three VLMs (Claude Opus~4, GPT-4o, Gemini~2.5 Pro), confirming a pattern-level rather than model-specific gain. Deployed across four backends, the module improves every one: 29.4\% of borderline leads improved on our pipeline; 41.2\% of failed limb leads recovered on ECG-Digitiser; valid leads per image doubled on Open-ECG-Digitizer (2.5 $\rightarrow$ 5.8). On 428 real clinical HCM images, the integrated system reaches 98.0\% Excellent quality. Both the plug-in architecture and tool-grounding mechanism are domain-parametric, suggesting broader applicability wherever quality criteria are objectively measurable.
>
---
#### [new 011] Learning Quantised Structure-Preserving Motion Representations for Dance Fingerprinting
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出DANCEMATCH，解决舞蹈检索任务中的运动表示问题，通过量化结构保留的运动特征实现高效检索。**

- **链接: [https://arxiv.org/pdf/2604.00927](https://arxiv.org/pdf/2604.00927)**

> **作者:** Arina Kharlamova; Bowei He; Chen Ma; Xue Liu
>
> **摘要:** We present DANCEMATCH, an end-to-end framework for motion-based dance retrieval, the task of identifying semantically similar choreographies directly from raw video, defined as DANCE FINGERPRINTING. While existing motion analysis and retrieval methods can compare pose sequences, they rely on continuous embeddings that are difficult to index, interpret, or scale. In contrast, DANCEMATCH constructs compact, discrete motion signatures that capture the spatio-temporal structure of dance while enabling efficient large-scale retrieval. Our system integrates Skeleton Motion Quantisation (SMQ) with Spatio-Temporal Transformers (STT) to encode human poses, extracted via Apple CoMotion, into a structured motion vocabulary. We further design DANCE RETRIEVAL ENGINE (DRE), which performs sub-linear retrieval using a histogram-based index followed by re-ranking for refined matching. To facilitate reproducible research, we release DANCETYPESBENCHMARK, a pose-aligned dataset annotated with quantised motion tokens. Experiments demonstrate robust retrieval across diverse dance styles and strong generalisation to unseen choreographies, establishing a foundation for scalable motion fingerprinting and quantitative choreographic analysis.
>
---
#### [new 012] Neuropsychiatric Deviations From Normative Profiles: An MRI-Derived Marker for Early Alzheimer's Disease Detection
- **分类: cs.CV**

- **简介: 该论文属于早期阿尔茨海默病检测任务，旨在通过MRI识别异常神经精神症状，以区分正常衰老与AD早期迹象。**

- **链接: [https://arxiv.org/pdf/2604.00545](https://arxiv.org/pdf/2604.00545)**

> **作者:** Synne Hjertager Osenbroch; Lisa Ramona Rosvold; Yao Lu; Alvaro Fernandez-Quilez
>
> **备注:** Accepted and to be presented (ORAL) in ISBI 2026
>
> **摘要:** Neuropsychiatric symptoms (NPS) such as depression and apathy are common in Alzheimer's disease (AD) and often precede cognitive decline. NPS assessments hold promise as early detection markers due to their correlation with disease progression and their non-invasive nature. Yet current tools cannot distinguish whether NPS are part of aging or early signs of AD, limiting their utility. We present a deep learning-based normative modelling framework to identify atypical NPS burden from structural MRI. A 3D convolutional neural network was trained on cognitively stable participants from the Alzheimer's Disease Neuroimaging Initiative, learning the mapping between brain anatomy and Neuropsychiatric Inventory Questionnaire (NPIQ) scores. Deviations between predicted and observed scores defined the Divergence from NPIQ scores (DNPI). Higher DNPI was associated with future AD conversion (adjusted OR=2.5; p < 0.01) and achieved predictive accuracy comparable to cerebrospinal fluid AB42 (AUC=0.74 vs 0.75). Our approach supports scalable, non-invasive strategies for early AD detection.
>
---
#### [new 013] Label-efficient underwater species classification with semi-supervised learning on frozen foundation model embeddings
- **分类: cs.CV**

- **简介: 该论文属于海洋生物分类任务，解决标注成本高和模型泛化能力差的问题。通过半监督学习在冻结的预训练特征上进行自训练，实现高效分类。**

- **链接: [https://arxiv.org/pdf/2604.00313](https://arxiv.org/pdf/2604.00313)**

> **作者:** Thomas Manuel Rost
>
> **摘要:** Automated species classification from underwater imagery is bottlenecked by the cost of expert annotation, and supervised models trained on one dataset rarely transfer to new conditions. We investigate whether semi-supervised methods operating on frozen foundation model embeddings can close this annotation gap with minimal labeling effort. Using DINOv3 ViT-B embeddings with no fine-tuning, we propagate a small set of labeled seeds through unlabeled data via nearest-neighbor-based self-training and evaluate on the AQUA20 benchmark (20 marine species). With fewer than 5% of the training labels, self-training on frozen embeddings closes much of the gap to a fully supervised ConvNeXt baseline trained on the entire labeled dataset; at full supervision, the gap narrows to a few percentage points, with several species exceeding the supervised baseline. Class separability in the embedding space, measured by ROC-AUC, is high even at extreme label scarcity, indicating that the frozen representations capture discriminative structure well before decision boundaries can be reliably estimated. Our approach requires no training, no domain-specific data engineering, and no underwater-adapted models, establishing a practical, immediately deployable baseline for label-efficient marine species recognition. All results are reported on the held-out test set over 100 random seed initializations.
>
---
#### [new 014] Representation Selection via Cross-Model Agreement using Canonical Correlation Analysis
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像表示学习任务，旨在解决预训练模型表示冗余和效率低的问题。通过CCA方法，实现跨模型表示选择与降维，提升下游任务性能。**

- **链接: [https://arxiv.org/pdf/2604.00921](https://arxiv.org/pdf/2604.00921)**

> **作者:** Dylan B. Lewis; Jens Gregor; Hector Santos-Villalobos
>
> **备注:** 9 pages, 5 figures, 6 tables
>
> **摘要:** Modern vision pipelines increasingly rely on pretrained image encoders whose representations are reused across tasks and models, yet these representations are often overcomplete and model-specific. We propose a simple, training-free method to improve the efficiency of image representations via a post-hoc canonical correlation analysis (CCA) operator. By leveraging the shared structure between representations produced by two pre-trained image encoders, our method finds linear projections that serve as a principled form of representation selection and dimensionality reduction, retaining shared semantic content while discarding redundant dimensions. Unlike standard dimensionality reduction techniques such as PCA, which operate on a single embedding space, our approach leverages cross-model agreement to guide representation distillation and refinement. The technique allows representations to be reduced by more than 75% in dimensionality with improved downstream performance, or enhanced at fixed dimensionality via post-hoc representation transfer from larger or fine-tuned models. Empirical results on ImageNet-1k, CIFAR-100, MNIST, and additional benchmarks show consistent improvements over both baseline and PCA-projected representations, with accuracy gains of up to 12.6%.
>
---
#### [new 015] MotionGrounder: Grounded Multi-Object Motion Transfer via Diffusion Transformer
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，解决多物体运动迁移问题。提出MotionGrounder框架，实现多物体精细控制，提升生成视频的准确性和一致性。**

- **链接: [https://arxiv.org/pdf/2604.00853](https://arxiv.org/pdf/2604.00853)**

> **作者:** Samuel Teodoro; Yun Chen; Agus Gunawan; Soo Ye Kim; Jihyong Oh; Munchurl Kim
>
> **备注:** Please visit our project page at this https URL
>
> **摘要:** Motion transfer enables controllable video generation by transferring temporal dynamics from a reference video to synthesize a new video conditioned on a target caption. However, existing Diffusion Transformer (DiT)-based methods are limited to single-object videos, restricting fine-grained control in real-world scenes with multiple objects. In this work, we introduce MotionGrounder, a DiT-based framework that firstly handles motion transfer with multi-object controllability. Our Flow-based Motion Signal (FMS) in MotionGrounder provides a stable motion prior for target video generation, while our Object-Caption Alignment Loss (OCAL) grounds object captions to their corresponding spatial regions. We further propose a new Object Grounding Score (OGS), which jointly evaluates (i) spatial alignment between source video objects and their generated counterparts and (ii) semantic consistency between each generated object and its target caption. Our experiments show that MotionGrounder consistently outperforms recent baselines across quantitative, qualitative, and human evaluations.
>
---
#### [new 016] When AI and Experts Agree on Error: Intrinsic Ambiguity in Dermatoscopic Images
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，探讨AI与专家在皮肤镜图像诊断中出现分歧的原因。研究发现部分图像存在固有歧义，导致AI和专家均易出错，揭示了图像质量对诊断准确性的影响。**

- **链接: [https://arxiv.org/pdf/2604.00651](https://arxiv.org/pdf/2604.00651)**

> **作者:** Loris Cino; Pier Luigi Mazzeo; Alessandro Martella; Giulia Radi; Renato Rossi; Cosimo Distante
>
> **摘要:** The integration of artificial intelligence (AI), particularly Convolutional Neural Networks (CNNs), into dermatological diagnosis demonstrates substantial clinical potential. While existing literature predominantly benchmarks algorithmic performance against human experts, our study adopts a novel perspective by investigating the intrinsic complexity of dermatoscopic images. Through rigorous experimentation with multiple CNN architectures, we isolated a subset of images systematically misclassified across all models-a phenomenon statistically proven to exceed random chance. To determine if these failures stem from algorithmic biases or inherent visual ambiguity, expert dermatologists independently evaluated these challenging cases alongside a control group. The results revealed a collapse in human diagnostic performance on the AI-misclassified images. First, agreement with ground-truth labels plummeted, with Cohen's kappa dropping to a mere 0.08 for the difficult images, compared to a 0.61 for the control group. Second, we observed a severe deterioration in expert consensus; inter-rater reliability among physicians fell from moderate concordance (Fleiss kappa = 0.456) on control images to only modest agreement (Fleiss kappa = 0.275) on difficult cases. We identified image quality as a primary driver of these dual systematic failures. To promote transparency and reproducibility, all data, code, and trained models have been made publicly available
>
---
#### [new 017] Maximizing T2-Only Prostate Cancer Localization from Expected Diffusion Weighted Imaging
- **分类: cs.CV**

- **简介: 该论文属于前列腺癌定位任务，旨在仅使用T2加权图像实现更准确的癌症定位。通过结合扩散加权图像作为潜在模态，提升诊断性能。**

- **链接: [https://arxiv.org/pdf/2604.00985](https://arxiv.org/pdf/2604.00985)**

> **作者:** Weixi Yi; Yipei Wang; Wen Yan; Hanyuan Zhang; Natasha Thorley; Alexander Ng; Shonit Punwani; Fernando Bianco; Mark Emberton; Veeru Kasivisvanathan; Dean C. Barratt; Shaheer U. Saeed; Yipeng Hu
>
> **摘要:** Multiparametric MRI is increasingly recommended as a first-line noninvasive approach to detect and localize prostate cancer, requiring at minimum diffusion-weighted (DWI) and T2-weighted (T2w) MR sequences. Early machine learning attempts using only T2w images have shown promising diagnostic performance in segmenting radiologist-annotated lesions. Such uni-modal T2-only approaches deliver substantial clinical benefits by reducing costs and expertise required to acquire other sequences. This work investigates an arguably more challenging application using only T2w at inference, but to localize individual cancers based on independent histopathology labels. We formulate DWI images as a latent modality (readily available during training) to classify cancer presence at local Barzell zones, given only T2w images as input. In the resulting expectation-maximization algorithm, a latent modality generator (implemented using a flow matching-based generative model) approximates the latent DWI image posterior distribution in the E-steps, while in M-steps a cancer localizer is simultaneously optimized with the generative model to maximize the expected likelihood of cancer presence. The proposed approach provides a novel theoretical framework for learning from a privileged DWI modality, yielding superior cancer localization performance compared to approaches that lack training DWI images or existing frameworks for privileged learning and incomplete modalities. The proposed T2-only methods perform competitively or better than baseline methods using multiple input sequences (e.g., improving the patient-level F1 score by 14.4\% and zone-level QWK by 5.3\% over the T2w+DWI baseline). We present quantitative evaluations using internal and external datasets from 4,133 prostate cancer patients with histopathology-verified labels.
>
---
#### [new 018] An Approach to Enriching Surgical Video Datasets for Fine-Grained Spatial-Temporal Understanding of Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型在手术视频中的应用任务，旨在解决现有数据集无法准确捕捉精细时空关系的问题。提出SurgSTU-Pipeline生成高质量数据集，提升模型的时空理解能力。**

- **链接: [https://arxiv.org/pdf/2604.00784](https://arxiv.org/pdf/2604.00784)**

> **作者:** Lennart Maack; Alexander Schlaefer
>
> **摘要:** Surgical video understanding is a crucial prerequisite for advancing Computer-Assisted Surgery. While vision-language models (VLMs) have recently been applied to the surgical domain, existing surgical vision-language datasets lack in capturing and evaluating complex, interleaved spatial-temporal dynamics. Creating large scale datasets that accurately represent fine-grained spatial-temporal relationships in surgical videos is challenging due to costly manual annotations or error-prone generation using large language models. To address this gap, we introduce the SurgSTU-Pipeline, a deterministic generation pipeline featuring temporal and spatial continuity filtering to reliably create surgical datasets for fine-grained spatial-temporal multimodal understanding. Applying this pipeline to publicly available surgical datasets, we create the SurgSTU dataset, comprising 7515 video clips densely extended with 150k fine-grained spatial-temporal question-answer samples. Our comprehensive evaluation shows that while state-of-the-art generalist VLMs struggle in zero-shot settings, their spatial-temporal capabilities can be improved through in-context learning. A fine-tuned VLM on the SurgSTU training dataset achieves highest performance among all spatial-temporal tasks, validating the dataset's efficacy to improve spatial-temporal understanding of VLMs in surgical videos. Code will be made publicly available.
>
---
#### [new 019] AceTone: Bridging Words and Colors for Conditional Image Grading
- **分类: cs.CV**

- **简介: 该论文提出AceTone，解决颜色分级任务中缺乏语义控制和美学对齐的问题。通过文本或参考图像生成3D-LUT，实现风格化颜色调整。**

- **链接: [https://arxiv.org/pdf/2604.00530](https://arxiv.org/pdf/2604.00530)**

> **作者:** Tianren Ma; Mingxiang Liao; Xijin Zhang; Qixiang Ye
>
> **备注:** Accepted by CVPR 2026. Project Page: this http URL
>
> **摘要:** Color affects how we interpret image style and emotion. Previous color grading methods rely on patch-wise recoloring or fixed filter banks, struggling to generalize across creative intents or align with human aesthetic preferences. In this study, we propose AceTone, the first approach that supports multimodal conditioned color grading within a unified framework. AceTone formulates grading as a generative color transformation task, where a model directly produces 3D-LUTs conditioned on text prompts or reference images. We develop a VQ-VAE based tokenizer which compresses a $3\times32^3$ LUT vector to 64 discrete tokens with $\Delta E<2$ fidelity. We further build a large-scale dataset, AceTone-800K, and train a vision-language model to predict LUT tokens, followed by reinforcement learning to align outputs with perceptual fidelity and aesthetics. Experiments show that AceTone achieves state-of-the-art performance on both text-guided and reference-guided grading tasks, improving LPIPS by up to 50% over existing methods. Human evaluations confirm that AceTone's results are visually pleasing and stylistically coherent, demonstrating a new pathway toward language-driven, aesthetic-aligned color grading.
>
---
#### [new 020] Suppressing Non-Semantic Noise in Masked Image Modeling Representations
- **分类: cs.CV**

- **简介: 该论文属于视觉自监督学习任务，旨在解决MIM模型中保留非语义信息的问题。通过引入PCA评分和SOAP方法，有效抑制非语义噪声，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.00172](https://arxiv.org/pdf/2604.00172)**

> **作者:** Martine Hjelkrem-Tan; Marius Aasan; Rwiddhi Chakraborty; Gabriel Y. Arteaga; Changkyu Choi; Adín Ramírez Rivera
>
> **备注:** Published in CVPR 2026
>
> **摘要:** Masked Image Modeling (MIM) has become a ubiquitous self-supervised vision paradigm. In this work, we show that MIM objectives cause the learned representations to retain non-semantic information, which ultimately hurts performance during inference. We introduce a model-agnostic score for semantic invariance using Principal Component Analysis (PCA) on real and synthetic non-semantic images. Based on this score, we propose a simple method, Semantically Orthogonal Artifact Projection (SOAP), to directly suppress non-semantic information in patch representations, leading to consistent improvements in zero-shot performance across various MIM-based models. SOAP is a post-hoc suppression method, requires zero training, and can be attached to any model as a single linear head.
>
---
#### [new 021] YieldSAT: A Multimodal Benchmark Dataset for High-Resolution Crop Yield Prediction
- **分类: cs.CV**

- **简介: 该论文提出YieldSAT数据集，用于高分辨率作物产量预测任务。解决数据稀缺、质量低及区域限制问题，包含多模态数据和大量样本，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.00940](https://arxiv.org/pdf/2604.00940)**

> **作者:** Miro Miranda; Deepak Pathak; Patrick Helber; Benjamin Bischke; Hiba Najjar; Francisco Mena; Cristhian Sanchez; Akshay Pai; Diego Arenas; Matias Valdenegro-Toro; Marcela Charfuelan; Marlon Nuske; Andreas Dengel
>
> **摘要:** Crop yield prediction requires substantial data to train scalable models. However, creating yield prediction datasets is constrained by high acquisition costs, heterogeneous data quality, and data privacy regulations. Consequently, existing datasets are scarce, low in quality, or limited to regional levels or single crop types, hindering the development of scalable data-driven solutions. In this work, we release YieldSAT, a large, high-quality, and multimodal dataset for high-resolution crop yield prediction. YieldSAT spans various climate zones across multiple countries, including Argentina, Brazil, Uruguay, and Germany, and includes major crop types, including corn, rapeseed, soybeans, and wheat, across 2,173 expert-curated fields. In total, over 12.2 million yield samples are available, each with a spatial resolution of 10 m. Each field is paired with multispectral satellite imagery, resulting in 113,555 labeled satellite images, complemented by auxiliary environmental data. We demonstrate the potential of large-scale and high-resolution crop yield prediction as a pixel regression task by comparing various deep learning models and data fusion architectures. Furthermore, we highlight open challenges arising from severe distribution shifts in the ground truth data under real-world conditions. To mitigate this, we explore a domain-informed Deep Ensemble approach that exhibits significant performance gains. The dataset is available at this https URL.
>
---
#### [new 022] COTTA: Context-Aware Transfer Adaptation for Trajectory Prediction in Autonomous Driving
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于轨迹预测任务，解决模型在不同地理区域性能下降的问题。通过对比多种迁移学习策略，提出在韩国数据上微调解码器的高效方法，显著提升预测精度。**

- **链接: [https://arxiv.org/pdf/2604.00402](https://arxiv.org/pdf/2604.00402)**

> **作者:** Seohyoung Park; Jaeyeol Lim; Seoyoung Ju; Kyeonghun Kim; Nam-Joon Kim; Hyuk-Jae Lee
>
> **备注:** 4 pages, 2 figures. Accepted at ICEIC 2026
>
> **摘要:** Developing robust models to accurately predict the trajectories of surrounding agents is fundamental to autonomous driving safety. However, most public datasets, such as the Waymo Open Motion Dataset and Argoverse, are collected in Western road environments and do not reflect the unique traffic patterns, infrastructure, and driving behaviors of other regions, including South Korea. This domain discrepancy leads to performance degradation when state-of-the-art models trained on Western data are deployed in different geographic contexts. In this work, we investigate the adaptability of Query-Centric Trajectory Prediction (QCNet) when transferred from U.S.-based data to Korean road environments. Using a Korean autonomous driving dataset, we compare four training strategies: zero-shot transfer, training from scratch, full fine-tuning, and encoder freezing. Experimental results demonstrate that leveraging pretrained knowledge significantly improves prediction performance. Specifically, selectively fine-tuning the decoder while freezing the encoder yields the best trade-off between accuracy and training efficiency, reducing prediction error by over 66% compared to training from scratch. This study provides practical insights into effective transfer learning strategies for deploying trajectory prediction models in new geographic domains.
>
---
#### [new 023] Omni-MMSI: Toward Identity-attributed Social Interaction Understanding
- **分类: cs.CV**

- **简介: 该论文提出Omni-MMSI任务，解决AI助手在真实场景中理解社会互动的问题。通过构建参考引导的管道，提升身份属性和社会推理能力。**

- **链接: [https://arxiv.org/pdf/2604.00267](https://arxiv.org/pdf/2604.00267)**

> **作者:** Xinpeng Li; Bolin Lai; Hardy Chen; Shijian Deng; Cihang Xie; Yuyin Zhou; James Matthew Rehg; Yapeng Tian
>
> **备注:** Accepted to CVPR 2026. Project page: this https URL
>
> **摘要:** We introduce Omni-MMSI, a new task that requires comprehensive social interaction understanding from raw audio, vision, and speech input. The task involves perceiving identity-attributed social cues (e.g., who is speaking what) and reasoning about the social interaction (e.g., whom the speaker refers to). This task is essential for developing AI assistants that can perceive and respond to human interactions. Unlike prior studies that operate on oracle-preprocessed social cues, Omni-MMSI reflects realistic scenarios where AI assistants must perceive and reason from raw data. However, existing pipelines and multi-modal LLMs perform poorly on Omni-MMSI because they lack reliable identity attribution capabilities, which leads to inaccurate social interaction understanding. To address this challenge, we propose Omni-MMSI-R, a reference-guided pipeline that produces identity-attributed social cues with tools and conducts chain-of-thought social reasoning. To facilitate this pipeline, we construct participant-level reference pairs and curate reasoning annotations on top of the existing datasets. Experiments demonstrate that Omni-MMSI-R outperforms advanced LLMs and counterparts on Omni-MMSI. Project page: this https URL.
>
---
#### [new 024] Shape Representation using Gaussian Process mixture models
- **分类: cs.CV**

- **简介: 该论文属于3D形状表示任务，旨在解决传统表示方法存储大、索引复杂的问题。提出使用高斯过程混合模型构建轻量级、连续的形状表示。**

- **链接: [https://arxiv.org/pdf/2604.00862](https://arxiv.org/pdf/2604.00862)**

> **作者:** Panagiotis Sapoutzoglou; George Terzakis; Georgios Floros; Maria Pateraki
>
> **备注:** To appear in ISPRS 2026
>
> **摘要:** Traditional explicit 3D representations, such as point clouds and meshes, demand significant storage to capture fine geometric details and require complex indexing systems for surface lookups, making functional representations an efficient, compact, and continuous alternative. In this work, we propose a novel, object-specific functional shape representation that models surface geometry with Gaussian Process (GP) mixture models. Rather than relying on computationally heavy neural architectures, our method is lightweight, leveraging GPs to learn continuous directional distance fields from sparsely sampled point clouds. We capture complex topologies by anchoring local GP priors at strategic reference points, which can be flexibly extracted using any structural decomposition method (e.g. skeletonization, distance-based clustering). Extensive evaluations on the ShapeNetCore and IndustryShapes datasets demonstrate that our method can efficiently and accurately represent complex geometries.
>
---
#### [new 025] PDA: Text-Augmented Defense Framework for Robust Vision-Language Models against Adversarial Image Attacks
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于视觉语言模型安全任务，旨在解决对抗图像攻击导致的模型脆弱性问题。提出PDA框架，通过文本增强提升模型鲁棒性，无需修改模型即可有效防御多种攻击。**

- **链接: [https://arxiv.org/pdf/2604.01010](https://arxiv.org/pdf/2604.01010)**

> **作者:** Jingning Xu; Haochen Luo; Chen Liu
>
> **摘要:** Vision-language models (VLMs) are vulnerable to adversarial image perturbations. Existing works based on adversarial training against task-specific adversarial examples are computationally expensive and often fail to generalize to unseen attack types. To address these limitations, we introduce Paraphrase-Decomposition-Aggregation (PDA), a training-free defense framework that leverages text augmentation to enhance VLM robustness under diverse adversarial image attacks. PDA performs prompt paraphrasing, question decomposition, and consistency aggregation entirely at test time, thus requiring no modification on the underlying models. To balance robustness and efficiency, we instantiate PDA as invariants that reduce the inference cost while retaining most of its robustness gains. Experiments on multiple VLM architectures and benchmarks for visual question answering, classification, and captioning show that PDA achieves consistent robustness gains against various adversarial perturbations while maintaining competitive clean accuracy, establishing a generic, strong and practical defense framework for VLMs during inference.
>
---
#### [new 026] Open-Set Supervised 3D Anomaly Detection: An Industrial Dataset and a Generalisable Framework for Unknown Defects
- **分类: cs.CV**

- **简介: 该论文属于3D异常检测任务，解决工业场景中未知缺陷识别问题。通过构建数据集和提出Open3D-AD方法，利用正常样本和少量异常样本进行训练，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.01171](https://arxiv.org/pdf/2604.01171)**

> **作者:** Hanzhe Liang; Luocheng Zhang; Junyang Xia; HanLiang Zhou; Bingyang Guo; Yingxi Xie; Can Gao; Ruiyun Yu; Jinbao Wang; Pan Li
>
> **备注:** Resources: this https URL
>
> **摘要:** Although self-supervised 3D anomaly detection assumes that acquiring high-precision point clouds is computationally expensive, in real manufacturing scenarios it is often feasible to collect a limited number of anomalous samples. Therefore, we study open-set supervised 3D anomaly detection, where the model is trained with only normal samples and a small number of known anomalous samples, aiming to identify unknown anomalies at test time. We present Open-Industry, a high-quality industrial dataset containing 15 categories, each with five real anomaly types collected from production lines. We first adapt general open-set anomaly detection methods to accommodate 3D point cloud inputs better. Building upon this, we propose Open3D-AD, a point-cloud-oriented approach that leverages normal samples, simulated anomalies, and partially observed real anomalies to model the probability density distributions of normal and anomalous data. Then, we introduce a simple Correspondence Distributions Subsampling to reduce the overlap between normal and non-normal distributions, enabling stronger dual distributions modeling. Based on these contributions, we establish a comprehensive benchmark and evaluate the proposed method extensively on Open-Industry as well as established datasets including Real3D-AD and Anomaly-ShapeNet. Benchmark results and ablation studies demonstrate the effectiveness of Open3D-AD and further reveal the potential of open-set supervised 3D anomaly detection.
>
---
#### [new 027] Automated Detection of Multiple Sclerosis Lesions on 7-tesla MRI Using U-net and Transformer-based Segmentation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于多发性硬化症病灶分割任务，旨在解决7T MRI下传统方法检测效果不佳的问题。通过训练U-net和Transformer模型，提升小病灶的检测能力。**

- **链接: [https://arxiv.org/pdf/2604.00469](https://arxiv.org/pdf/2604.00469)**

> **作者:** Michael Maynord; Minghui Liu; Cornelia Fermüller; Seongjin Choi; Yuxin Zeng; Shishir Dahal; Daniel M. Harrison
>
> **备注:** 31 pages, 3 figures, 3 tables. Inference code and model weights available at this https URL
>
> **摘要:** Ultra-high field 7-tesla (7T) MRI improves visualization of multiple sclerosis (MS) white matter lesions (WML) but differs sufficiently in contrast and artifacts from 1.5-3T imaging - suggesting that widely used automated segmentation tools may not translate directly. We analyzed 7T FLAIR scans and generated reference WML masks from Lesion Segmentation Tool (LST) outputs followed by expert manual revision. As external comparators, we applied LST-LPA and the more recent LST-AI ensemble, both originally developed on lower-field data. We then trained 3D UNETR and SegFormer transformer-based models on 7T FLAIR at multiple resolutions (0.5x0.5x0.5^3, 1.0x1.0x1.0^3, and 1.5x1.5x2.0^3) and evaluated all methods using voxel-wise and lesion-wise metrics from the BraTS 2023 framework. On the held-out test set at native 0.5x0.5x0.5^3 resolution, 7T-trained transformers achieved competitive overlap with LST-AI while recovering additional small lesions that were missed by classical methods, at the cost of some boundary variability and occasional artifact-related false positives. On a held-out 7 T test set, our best transformer model (SegFormer) achieved a voxel-wise Dice of 0.61 and lesion-wise Dice of 0.20, improving on the classical LST-LPA tool (Dice 0.39, lesion-wise Dice 0.02). Performance decreased for models trained on downsampled images, underscoring the value of native 7T resolution for small-lesion detection. By releasing our 7T-trained models, we aim to provide a reproducible, ready-to-use resource for automated lesion quantification in ultra-high field MS research (this https URL).
>
---
#### [new 028] ProOOD: Prototype-Guided Out-of-Distribution 3D Occupancy Prediction
- **分类: cs.CV; cs.LG; cs.RO; eess.IV**

- **简介: 该论文属于3D语义占用预测任务，解决长尾类别偏差和分布外输入问题。提出ProOOD方法，通过原型引导优化提升模型对罕见类别的识别和分布外检测能力。**

- **链接: [https://arxiv.org/pdf/2604.01081](https://arxiv.org/pdf/2604.01081)**

> **作者:** Yuheng Zhang; Mengfei Duan; Kunyu Peng; Yuhang Wang; Di Wen; Danda Pani Paudel; Luc Van Gool; Kailun Yang
>
> **备注:** Accepted to CVPR 2026. The source code is publicly available at this https URL
>
> **摘要:** 3D semantic occupancy prediction is central to autonomous driving, yet current methods are vulnerable to long-tailed class bias and out-of-distribution (OOD) inputs, often overconfidently assigning anomalies to rare classes. We present ProOOD, a lightweight, plug-and-play method that couples prototype-guided refinement with training-free OOD scoring. ProOOD comprises (i) prototype-guided semantic imputation that fills occluded regions with class-consistent features, (ii) prototype-guided tail mining that strengthens rare-class representations to curb OOD absorption, and (iii) EchoOOD, which fuses local logit coherence with local and global prototype matching to produce reliable voxel-level OOD scores. Extensive experiments on five datasets demonstrate that ProOOD achieves state-of-the-art performance on both in-distribution 3D occupancy prediction and OOD detection. On SemanticKITTI, it surpasses baselines by +3.57% mIoU overall and +24.80% tail-class mIoU; on VAA-KITTI, it improves AuPRCr by +19.34 points, with consistent gains across benchmarks. These improvements yield more calibrated occupancy estimates and more reliable OOD detection in safety-critical urban driving. The source code is publicly available at this https URL.
>
---
#### [new 029] UCMNet: Uncertainty-Aware Context Memory Network for Under-Display Camera Image Restoration
- **分类: cs.CV**

- **简介: 该论文属于图像修复任务，解决UDC成像中因光衍射和散射导致的细节丢失问题。提出UCMNet模型，通过不确定性感知机制提升细节恢复效果。**

- **链接: [https://arxiv.org/pdf/2604.00381](https://arxiv.org/pdf/2604.00381)**

> **作者:** Daehyun Kim; Youngmin Kim; Yoon Ju Oh; Tae Hyun Kim
>
> **备注:** We propose UCMNet, an uncertainty-aware adaptive framework that restores high-frequency details in regions with varying levels of degradation in under-display camera images
>
> **摘要:** Under-display cameras (UDCs) allow for full-screen designs by positioning the imaging sensor underneath the display. Nonetheless, light diffraction and scattering through the various display layers result in spatially varying and complex degradations, which significantly reduce high-frequency details. Current PSF-based physical modeling techniques and frequency-separation networks are effective at reconstructing low-frequency structures and maintaining overall color consistency. However, they still face challenges in recovering fine details when dealing with complex, spatially varying degradation. To solve this problem, we propose a lightweight \textbf{U}ncertainty-aware \textbf{C}ontext-\textbf{M}emory \textbf{Network} (\textbf{UCMNet}), for UDC image restoration. Unlike previous methods that apply uniform restoration, UCMNet performs uncertainty-aware adaptive processing to restore high-frequency details in regions with varying degradations. The estimated uncertainty maps, learned through an uncertainty-driven loss, quantify spatial uncertainty induced by diffraction and scattering, and guide the Memory Bank to retrieve region-adaptive context from the Context Bank. This process enables effective modeling of the non-uniform degradation characteristics inherent to UDC imaging. Leveraging this uncertainty as a prior, UCMNet achieves state-of-the-art performance on multiple benchmarks with 30\% fewer parameters than previous models. Project page: \href{this https URL}{this https URL}.
>
---
#### [new 030] ProCap: Projection-Aware Captioning for Spatial Augmented Reality
- **分类: cs.CV; cs.MM**

- **简介: 该论文提出ProCap，解决SAR中虚拟与物理内容混淆问题，通过分割和检索实现语义区分。属于视觉语言模型任务。**

- **链接: [https://arxiv.org/pdf/2604.00912](https://arxiv.org/pdf/2604.00912)**

> **作者:** Zimo Cao; Yuchen Deng; Haibin Ling; Bingyao Huang
>
> **备注:** 16 pages, 7 figures
>
> **摘要:** Spatial augmented reality (SAR) directly projects digital content onto physical scenes using projectors, creating immersive experience without head-mounted displays. However, for SAR to support intelligent interaction, such as reasoning about the scene or answering user queries, it must semantically distinguish between the physical scene and the projected content. Standard Vision Language Models (VLMs) struggle with this virtual-physical ambiguity, often confusing the two contexts. To address this issue, we introduce ProCap, a novel framework that explicitly decouples projected content from physical scenes. ProCap employs a two-stage pipeline: first it visually isolates virtual and physical layers via automated segmentation; then it uses region-aware retrieval to avoid ambiguous semantic context due to projection distortion. To support this, we present RGBP (RGB + Projections), the first large-scale SAR semantic benchmark dataset, featuring 65 diverse physical scenes and over 180,000 projections with dense, decoupled annotations. Finally, we establish a dual-captioning evaluation protocol using task-specific tokens to assess physical scene and projection descriptions independently. Our experiments show that ProCap provides a robust semantic foundation for future SAR research. The source code, pre-trained models and the RGBP dataset are available on the project page: this https URL.
>
---
#### [new 031] Revisiting Human-in-the-Loop Object Retrieval with Pre-Trained Vision Transformers
- **分类: cs.CV; cs.HC; cs.IR**

- **简介: 该论文研究人机协同目标检索任务，旨在通过用户反馈和预训练ViT模型，从无标签数据中高效识别目标类别图像，解决多对象场景下的细粒度特征捕捉问题。**

- **链接: [https://arxiv.org/pdf/2604.00809](https://arxiv.org/pdf/2604.00809)**

> **作者:** Kawtar Zaher; Olivier Buisson; Alexis Joly
>
> **摘要:** Building on existing approaches, we revisit Human-in-the-Loop Object Retrieval, a task that consists of iteratively retrieving images containing objects of a class-of-interest, specified by a user-provided query. Starting from a large unlabeled image collection, the aim is to rapidly identify diverse instances of an object category relying solely on the initial query and the user's Relevance Feedback, with no prior labels. The retrieval process is formulated as a binary classification task, where the system continuously learns to distinguish between relevant and non-relevant images to the query, through iterative user interaction. This interaction is guided by an Active Learning loop: at each iteration, the system selects informative samples for user annotation, thereby refining the retrieval performance. This task is particularly challenging in multi-object datasets, where the object of interest may occupy only a small region of the image within a complex, cluttered scene. Unlike object-centered settings where global descriptors often suffice, multi-object images require more adapted, localized descriptors. In this work, we formulate and revisit the Human-in-the-Loop Object Retrieval task by leveraging pre-trained ViT representations, and addressing key design questions, including which object instances to consider in an image, what form the annotations should take, how Active Selection should be applied, and which representation strategies best capture the object's features. We compare several representation strategies across multi-object datasets highlighting trade-offs between capturing the global context and focusing on fine-grained local object details. Our results offer practical insights for the design of effective interactive retrieval pipelines based on Active Learning for object class retrieval.
>
---
#### [new 032] Video Patch Pruning: Efficient Video Instance Segmentation via Early Token Reduction
- **分类: cs.CV**

- **简介: 该论文属于视频实例分割任务，旨在降低ViT的计算成本。通过引入时间先验知识，在早期层进行高效补丁剪枝，实现高达60%的补丁减少，提升模型效率。**

- **链接: [https://arxiv.org/pdf/2604.00827](https://arxiv.org/pdf/2604.00827)**

> **作者:** Patrick Glandorf; Thomas Norrenbrock; Bodo Rosenhahn
>
> **备注:** CVPR'26 Workshops
>
> **摘要:** Vision Transformers (ViTs) have demonstrated state-ofthe-art performance in several benchmarks, yet their high computational costs hinders their practical deployment. Patch Pruning offers significant savings, but existing approaches restrict token reduction to deeper layers, leaving early-stage compression unexplored. This limits their potential for holistic efficiency. In this work, we present a novel Video Patch Pruning framework (VPP) that integrates temporal prior knowledge to enable efficient sparsity within early ViT layers. Our approach is motivated by the observation that prior features extracted from deeper layers exhibit strong foreground selectivity. Therefore we propose a fully differentiable module for temporal mapping to accurately select the most relevant patches in early network stages. Notably, the proposed method enables a patch reduction of up to 60% in dense prediction tasks, exceeding the capabilities of conventional image-based patch pruning, which typically operate around a 30% patch sparsity. VPP excels the high-sparsity regime, sustaining remarkable performance even when patch usage is reduced below 55%. Specifically, it preserves stable results with a maximal performance drop of 0.6% on the Youtube-VIS 2021 dataset.
>
---
#### [new 033] TRiGS: Temporal Rigid-Body Motion for Scalable 4D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属于4D重建任务，解决动态场景中时间碎片化和内存膨胀问题。提出TRiGS模型，通过连续几何变换实现高效、稳定的长视频序列重建。**

- **链接: [https://arxiv.org/pdf/2604.00538](https://arxiv.org/pdf/2604.00538)**

> **作者:** Suwoong Yeom; Joonsik Nam; Seunggyu Choi; Lucas Yunkyu Lee; Sangmin Kim; Jaesik Park; Joonsoo Kim; Kugjin Yun; Kyeongbo Kong; Sukju Kang
>
> **备注:** Project page: this https URL
>
> **摘要:** Recent 4D Gaussian Splatting (4DGS) methods achieve impressive dynamic scene reconstruction but often rely on piecewise linear velocity approximations and short temporal windows. This disjointed modeling leads to severe temporal fragmentation, forcing primitives to be repeatedly eliminated and regenerated to track complex nonlinear dynamics. This makeshift approximation eliminates the long-term temporal identity of objects and causes an inevitable proliferation of Gaussians, hindering scalability to extended video sequences. To address this, we propose TRiGS, a novel 4D representation that utilizes unified, continuous geometric transformations. By integrating $SE(3)$ transformations, hierarchical Bezier residuals, and learnable local anchors, TRiGS models geometrically consistent rigid motions for individual primitives. This continuous formulation preserves temporal identity and effectively mitigates unbounded memory growth. Extensive experiments demonstrate that TRiGS achieves high fidelity rendering on standard benchmarks while uniquely scaling to extended video sequences (e.g., 600 to 1200 frames) without severe memory bottlenecks, significantly outperforming prior works in temporal stability.
>
---
#### [new 034] Enhancing Gradient Inversion Attacks in Federated Learning via Hierarchical Feature Optimization
- **分类: cs.CV**

- **简介: 该论文属于隐私泄露防护任务，针对联邦学习中梯度反演攻击问题，提出GIFD方法，通过优化特征域提升攻击效果。**

- **链接: [https://arxiv.org/pdf/2604.00955](https://arxiv.org/pdf/2604.00955)**

> **作者:** Hao Fang; Wenbo Yu; Bin Chen; Xuan Wang; Shu-Tao Xia; Qing Liao; Ke Xu
>
> **摘要:** Federated Learning (FL) has emerged as a compelling paradigm for privacy-preserving distributed machine learning, allowing multiple clients to collaboratively train a global model by transmitting locally computed gradients to a central server without exposing their private data. Nonetheless, recent studies find that the gradients exchanged in the FL system are also vulnerable to privacy leakage, e.g., an attacker can invert shared gradients to reconstruct sensitive data by leveraging pre-trained generative adversarial networks (GAN) as prior knowledge. However, existing attacks simply perform gradient inversion in the latent space of the GAN model, which limits their expression ability and generalizability. To tackle these challenges, we propose \textbf{G}radient \textbf{I}nversion over \textbf{F}eature \textbf{D}omains (GIFD), which disassembles the GAN model and searches the hierarchical features of the intermediate layers. Instead of optimizing only over the initial latent code, we progressively change the optimized layer, from the initial latent space to intermediate layers closer to the output images. In addition, we design a regularizer to avoid unreal image generation by adding a small ${l_1}$ ball constraint to the searching range. We also extend GIFD to the out-of-distribution (OOD) setting, which weakens the assumption that the training sets of GANs and FL tasks obey the same data distribution. Furthermore, we consider the challenging OOD scenario of label inconsistency and propose a label mapping technique as an effective solution. Extensive experiments demonstrate that our method can achieve pixel-level reconstruction and outperform competitive baselines across a variety of FL scenarios.
>
---
#### [new 035] A global dataset of continuous urban dashcam driving
- **分类: cs.CV**

- **简介: 该论文提出CROWD数据集，用于城市驾驶行为分析，解决跨域鲁棒性与交互研究问题，包含大量未编辑的行车视频及标注信息。**

- **链接: [https://arxiv.org/pdf/2604.01044](https://arxiv.org/pdf/2604.01044)**

> **作者:** Md Shadab Alam; Olena Bazilinska; Pavlo Bazilinskyy
>
> **摘要:** We introduce CROWD (City Road Observations With Dashcams), a manually curated dataset of ordinary, minute scale, temporally contiguous, unedited, front facing urban dashcam segments screened and segmented from publicly available YouTube videos. CROWD is designed to support cross-domain robustness and interaction analysis by prioritising routine driving and explicitly excluding crashes, crash aftermath, and other edited or incident-focused content. The release contains 51,753 segment records spanning 20,275.56 hours (42,032 videos), covering 7,103 named inhabited places in 238 countries and territories across all six inhabited continents (Africa, Asia, Europe, North America, South America and Oceania), with segment level manual labels for time of day (day or night) and vehicle type. To lower the barrier for benchmarking, we provide per-segment CSV files of machine-generated detections for all 80 MS-COCO classes produced with YOLOv11x, together with segment-local multi-object tracks (BoT-SORT); e.g. person, bicycle, motorcycle, car, bus, truck, traffic light, stop sign, etc. CROWD is distributed as video identifiers with segment boundaries and derived annotations, enabling reproducible research without redistributing the underlying videos.
>
---
#### [new 036] ReMoGen: Real-time Human Interaction-to-Reaction Generation via Modular Learning from Diverse Data
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出ReMoGen，解决实时人机交互生成问题，通过模块化学习框架实现高效、高质量的运动反应生成。**

- **链接: [https://arxiv.org/pdf/2604.01082](https://arxiv.org/pdf/2604.01082)**

> **作者:** Yaoqin Ye; Yiteng Xu; Qin Sun; Xinge Zhu; Yujing Sun; Yuexin Ma
>
> **备注:** accepted by CVPR 2026, project page: this https URL
>
> **摘要:** Human behaviors in real-world environments are inherently interactive, with an individual's motion shaped by surrounding agents and the scene. Such capabilities are essential for applications in virtual avatars, interactive animation, and human-robot collaboration. We target real-time human interaction-to-reaction generation, which generates the ego's future motion from dynamic multi-source cues, including others' actions, scene geometry, and optional high-level semantic inputs. This task is fundamentally challenging due to (i) limited and fragmented interaction data distributed across heterogeneous single-person, human-human, and human-scene domains, and (ii) the need to produce low-latency yet high-fidelity motion responses during continuous online interaction. To address these challenges, we propose ReMoGen (Reaction Motion Generation), a modular learning framework for real-time interaction-to-reaction generation. ReMoGen leverages a universal motion prior learned from large-scale single-person motion datasets and adapts it to target interaction domains through independently trained Meta-Interaction modules, enabling robust generalization under data-scarce and heterogeneous supervision. To support responsive online interaction, ReMoGen performs segment-level generation together with a lightweight Frame-wise Segment Refinement module that incorporates newly observed cues at the frame level, improving both responsiveness and temporal coherence without expensive full-sequence inference. Extensive experiments across human-human, human-scene, and mixed-modality interaction settings show that ReMoGen produces high-quality, coherent, and responsive reactions, while generalizing effectively across diverse interaction scenarios.
>
---
#### [new 037] DVGT-2: Vision-Geometry-Action Model for Autonomous Driving at Scale
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出DVGT-2模型，解决自动驾驶中实时几何重建与轨迹规划问题，通过在线处理和滑动窗口策略提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2604.00813](https://arxiv.org/pdf/2604.00813)**

> **作者:** Sicheng Zuo; Zixun Xie; Wenzhao Zheng; Shaoqing Xu; Fang Li; Hanbing Li; Long Chen; Zhi-Xin Yang; Jiwen Lu
>
> **备注:** Code is available at \href{this https URL}
>
> **摘要:** End-to-end autonomous driving has evolved from the conventional paradigm based on sparse perception into vision-language-action (VLA) models, which focus on learning language descriptions as an auxiliary task to facilitate planning. In this paper, we propose an alternative Vision-Geometry-Action (VGA) paradigm that advocates dense 3D geometry as the critical cue for autonomous driving. As vehicles operate in a 3D world, we think dense 3D geometry provides the most comprehensive information for decision-making. However, most existing geometry reconstruction methods (e.g., DVGT) rely on computationally expensive batch processing of multi-frame inputs and cannot be applied to online planning. To address this, we introduce a streaming Driving Visual Geometry Transformer (DVGT-2), which processes inputs in an online manner and jointly outputs dense geometry and trajectory planning for the current frame. We employ temporal causal attention and cache historical features to support on-the-fly inference. To further enhance efficiency, we propose a sliding-window streaming strategy and use historical caches within a certain interval to avoid repetitive computations. Despite the faster speed, DVGT-2 achieves superior geometry reconstruction performance on various datasets. The same trained DVGT-2 can be directly applied to planning across diverse camera configurations without fine-tuning, including closed-loop NAVSIM and open-loop nuScenes benchmarks.
>
---
#### [new 038] The 1st Winner for 5th PVUW MeViS-Text Challenge: Strong MLLMs Meet SAM3 for Referring Video Object Segmentation
- **分类: cs.CV**

- **简介: 该论文属于视频目标分割任务，解决运动中心语言描述下的视频对象分割问题。通过结合大模型与SAM3，实现无需微调的精准分割。**

- **链接: [https://arxiv.org/pdf/2604.00404](https://arxiv.org/pdf/2604.00404)**

> **作者:** Xusheng He; Canyang Wu; Jinrong Zhang; Weili Guan; Jianlong Wu; Liqiang Nie
>
> **备注:** 1st Place Solution for the 5th PVUW MeViS-Text Challenge (CVPR 2026 Workshop)
>
> **摘要:** This report presents our winning solution to the 5th PVUW MeViS-Text Challenge. The track studies referring video object segmentation under motion-centric language expressions, where the model must jointly understand appearance, temporal behavior, and object interactions. To address this problem, we build a fully training-free pipeline that combines strong multimodal large language models with SAM3. Our method contains three stages. First, Gemini-3.1 Pro decomposes each target event into instance-level grounding targets, selects the frame where the target is most clearly visible, and generates a discriminative description. Second, SAM3-agent produces a precise seed mask on the selected frame, and the official SAM3 tracker propagates the mask through the whole video. Third, a refinement stage uses Qwen3.5-Plus and behavior-level verification to correct ambiguous or semantically inconsistent predictions. Without task-specific fine-tuning, our method ranks first on the PVUW 2026 MeViS-Text test set, achieving a Final score of 0.909064 and a J&F score of 0.7897. The code is available at this https URL.
>
---
#### [new 039] UCell: rethinking generalizability and scaling of bio-medical vision models
- **分类: cs.CV; q-bio.QM**

- **简介: 该论文提出UCell模型，解决生物医学图像分割任务中模型泛化与规模受限的问题。通过递归结构提升小模型性能，实现与大模型相当的效果，且无需依赖外部数据。**

- **链接: [https://arxiv.org/pdf/2604.00243](https://arxiv.org/pdf/2604.00243)**

> **作者:** Nicholas Kuang; Vanessa Scalon; Ji Yu
>
> **摘要:** The modern deep learning field is a scale-centric one. Larger models have been shown to consistently perform better than smaller models of similar architecture. In many sub-domains of biomedical research, however, the model scaling is bottlenecked by the amount of available training data, and the high cost associated with generating and validating additional high quality data. Despite the practical hurdle, the majority of the ongoing research still focuses on building bigger foundation models, whereas the alternative of improving the ability of small models has been under-explored. Here we experiment with building models with 10-30M parameters, tiny by modern standards, to perform the single-cell segmentation task. An important design choice is the incorporation of a recursive structure into the model's forward computation graph, leading to a more parameter-efficient architecture. We found that for the single-cell segmentation, on multiple benchmarks, our small model, UCell, matches the performance of models 10-20 times its size, and with a similar generalizability to unseen out-of-domain data. More importantly, we found that ucell can be trained from scratch using only a set of microscopy imaging data, without relying on massive pretraining on natural images, and therefore decouples the model building from any external commercial interests. Finally, we examined and confirmed the adaptability of ucell by performing a wide range of one-shot and few-shot fine tuning experiments on a diverse set of small datasets. Implementation is available at this https URL
>
---
#### [new 040] MoonAnything: A Vision Benchmark with Large-Scale Lunar Supervised Data
- **分类: cs.CV**

- **简介: 该论文提出MoonAnything基准，解决月面感知数据不足问题，包含几何与光度监督的大型数据集，支持3D重建和光照鲁棒感知。**

- **链接: [https://arxiv.org/pdf/2604.00682](https://arxiv.org/pdf/2604.00682)**

> **作者:** Clémentine Grethen; Yuang Shi; Simone Gasparini; Géraldine Morin
>
> **备注:** Accepted to ACM MMSys 2026
>
> **摘要:** Accurate perception of lunar surfaces is critical for modern lunar exploration missions. However, developing robust learning-based perception systems is hindered by the lack of datasets that provide both geometric and photometric supervision. Existing lunar datasets typically lack either geometric ground truth, photometric realism, illumination diversity, or large-scale coverage. In this paper, we introduce MoonAnything, a unified benchmark built on real lunar topography with physically-based rendering, providing the first comprehensive geometric and photometric supervision under diverse illumination with large scale. The benchmark comprises two complementary sub-datasets : i) LunarGeo provides stereo images with corresponding dense depth maps and camera calibration enabling 3D reconstruction and pose estimation; ii) LunarPhoto provides photorealistic images using a spatially-varying BRDF model, along with multi-illumination renderings under real solar configurations, enabling reflectance estimation and illumination-robust perception. Together, these datasets offer over 130K samples with comprehensive supervision. Beyond lunar applications, MoonAnything offers a unique setting and challenging testbed for algorithms under low-textured, high-contrast conditions and applies to other airless celestial bodies and could generalize beyond. We establish baselines using state-of-the-art methods and release the complete dataset along with generation tools to support community extension: this https URL.
>
---
#### [new 041] HarassGuard: Detecting Harassment Behaviors in Social Virtual Reality with Vision-Language Models
- **分类: cs.CV; cs.HC**

- **简介: 该论文属于内容安全检测任务，旨在解决社会虚拟现实中骚扰行为的主动识别问题。提出HarassGuard系统，利用视觉语言模型仅通过视觉输入检测骚扰行为，兼顾隐私与效果。**

- **链接: [https://arxiv.org/pdf/2604.00592](https://arxiv.org/pdf/2604.00592)**

> **作者:** Junhee Lee; Minseok Kim; Hwanjo Heo; Seungwon Woo; Jinwoo Kim
>
> **备注:** To appear in the 2026 TVCG Special Issue on the 2026 IEEE Conference on Virtual Reality and 3D User Interfaces (VR)
>
> **摘要:** Social Virtual Reality (VR) platforms provide immersive social experiences but also expose users to serious risks of online harassment. Existing safety measures are largely reactive, while proactive solutions that detect harassment behavior during an incident often depend on sensitive biometric data, raising privacy concerns. In this paper, we present HarassGuard, a vision-language model (VLM) based system that detects physical harassment in social VR using only visual input. We construct an IRB-approved harassment vision dataset, apply prompt engineering, and fine-tune VLMs to detect harassment behavior by considering contextual information in social VR. Experimental results demonstrate that HarassGuard achieves competitive performance compared to state-of-the-art baselines (i.e., LSTM/CNN, Transformer), reaching an accuracy of up to 88.09% in binary classification and 68.85% in multi-class classification. Notably, HarassGuard matches these baselines while using significantly fewer fine-tuning samples (200 vs. 1,115), offering unique advantages in contextual reasoning and privacy-preserving detection.
>
---
#### [new 042] ProTPS: Prototype-Guided Text Prompt Selection for Continual Learning
- **分类: cs.CV**

- **简介: 该论文属于持续学习任务，旨在解决新类别语义特征与旧类别重叠导致的灾难性遗忘问题。提出ProTPS方法，通过原型引导选择文本提示，提升模型适应新类别的能力。**

- **链接: [https://arxiv.org/pdf/2604.01116](https://arxiv.org/pdf/2604.01116)**

> **作者:** Jie Mei; Li-Leng Peng; Keith Fuller; Jenq-Neng Hwang
>
> **摘要:** For continual learning, text-prompt-based methods leverage text encoders and learnable prompts to encode semantic features for sequentially arrived classes over time. A common challenge encountered by existing works is how to learn unique text prompts, which implicitly carry semantic information of new classes, so that the semantic features of newly arrived classes do not overlap with those of trained classes, thereby mitigating the catastrophic forgetting problem. To address this challenge, we propose a novel approach Prototype-guided Text Prompt Selection (ProTPS)'' to intentionally increase the training flexibility thus encouraging the learning of unique text prompts. Specifically, our ProTPS learns class-specific vision prototypes and text prompts. Vision prototypes guide the selection and learning of text prompts for each class. We first evaluate our ProTPS in both class incremental (CI) setting and cross-datasets continual (CDC) learning setting. Because our ProTPS achieves performance close to the upper bounds, we further collect a real-world dataset with 112 marine species collected over a span of six years, named Marine112, to bring new challenges to the community. Marine112 is authentically suited for the class and domain incremental (CDI) learning setting and is under natural long-tail distribution. The results under three settings show that our ProTPS performs favorably against the recent state-of-the-art methods. The implementation code and Marine112 dataset will be released upon the acceptance of our paper.
>
---
#### [new 043] STAR: Mitigating Cascading Errors in Spatial Reasoning via Turn-point Alignment and Segment-level DPO
- **分类: cs.CV**

- **简介: 该论文属于空间推理任务，旨在解决复杂拓扑中的级联错误问题。提出STAR框架，通过分阶段优化提升导航准确性。**

- **链接: [https://arxiv.org/pdf/2604.00558](https://arxiv.org/pdf/2604.00558)**

> **作者:** Pukun Zhao; Longxiang Wang; Chen Chen; Peicheng Wang; Fanqing Zhou; Runze Li; Haojian Huang
>
> **备注:** 9 pages, 6 figures, 4 tables, Accepted by ICME 2026
>
> **摘要:** Structured spatial navigation is a core benchmark for Large Language Models (LLMs) spatial reasoning. Existing paradigms like Visualization-of-Thought (VoT) are prone to cascading errors in complex topologies. To solve this, we propose STAR, a two-stage framework grounded on topological anchors, and introduce the RedMaze-23K dataset with human-inspired turnpoint annotations. The first stage uses supervised fine-tuning to help models internalize spatial semantics and prune redundant paths. The second adopts Spatial-aware Segment-level Direct Preference Optimization (SDPO) to refine self-correction in long-horizon navigation. Experiments show STAR achieves state-of-the-art performance among open-source models: its 32B variant outperforms DeepSeek-V3 (29.27% vs. 25.00%) and reaches 82.4% of GPT-4's performance.
>
---
#### [new 044] PC-SAM: Patch-Constrained Fine-Grained Interactive Road Segmentation in High-Resolution Remote Sensing Images
- **分类: cs.CV**

- **简介: 该论文属于遥感图像道路分割任务，解决自动分割不足和缺乏交互细化的问题。提出PC-SAM框架，结合自动与交互分割，实现精准局部修正。**

- **链接: [https://arxiv.org/pdf/2604.00495](https://arxiv.org/pdf/2604.00495)**

> **作者:** Chengcheng Lv; Rushi Li; Mincheng Wu; Xiufang Shi; Zhenyu Wen; Shibo He
>
> **摘要:** Road masks obtained from remote sensing images effectively support a wide range of downstream tasks. In recent years, most studies have focused on improving the performance of fully automatic segmentation models for this task, achieving significant gains. However, current fully automatic methods are still insufficient for identifying certain challenging road segments and often produce false positive and false negative regions. Moreover, fully automatic segmentation does not support local segmentation of regions of interest or refinement of existing masks. Although the SAM model is widely used as an interactive segmentation model and performs well on natural images, it shows poor performance in remote sensing road segmentation and cannot support fine-grained local refinement. To address these limitations, we propose PC-SAM, which integrates fully automatic road segmentation and interactive segmentation within a unified framework. By carefully designing a fine-tuning strategy, the influence of point prompts is constrained to their corresponding patches, overcoming the inability of the original SAM to perform fine local corrections and enabling fine-grained interactive mask refinement. Extensive experiments on several representative remote sensing road segmentation datasets demonstrate that, when combined with point prompts, PC-SAM significantly outperforms state-of-the-art fully automatic models in road mask segmentation, while also providing flexible local mask refinement and local road segmentation. The code will be available at this https URL.
>
---
#### [new 045] All Roads Lead to Rome: Incentivizing Divergent Thinking in Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于视觉-语言模型研究，旨在解决RL导致的思维单一化问题。通过分析GRPO的局限性，提出MUPO方法以促进多样化思考。**

- **链接: [https://arxiv.org/pdf/2604.00479](https://arxiv.org/pdf/2604.00479)**

> **作者:** Xinyu Tian; Shu Zou; Zhaoyuan Yang; Mengqi He; Peter Tu; Jing Zhang
>
> **备注:** Accepted to CVPR2026
>
> **摘要:** Recent studies have demonstrated that Reinforcement Learning (RL), notably Group Relative Policy Optimization (GRPO), can intrinsically elicit and enhance the reasoning capabilities of Vision-Language Models (VLMs). However, despite the promise, the underlying mechanisms that drive the effectiveness of RL models as well as their limitations remain underexplored. In this paper, we highlight a fundamental behavioral distinction between RL and base models, where the former engages in deeper yet narrow reasoning, while base models, despite less refined along individual path, exhibit broader and more diverse thinking patterns. Through further analysis of training dynamics, we show that GRPO is prone to diversity collapse, causing models to prematurely converge to a limited subset of reasoning strategies while discarding the majority of potential alternatives, leading to local optima and poor scalability. To address this, we propose Multi-Group Policy Optimization (MUPO), a simple yet effective approach designed to incentivize divergent thinking across multiple solutions, and demonstrate its effectiveness on established benchmarks. Project page: this https URL
>
---
#### [new 046] TALENT: Target-aware Efficient Tuning for Referring Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于 referring image segmentation 任务，解决 PET 方法中视觉特征误激活非目标对象的问题。提出 TALENT 框架，通过 RCA 和 TLM 提升目标定位准确性。**

- **链接: [https://arxiv.org/pdf/2604.00609](https://arxiv.org/pdf/2604.00609)**

> **作者:** Shuo Jin; Siyue Yu; Bingfeng Zhang; Chao Yao; Meiqin Liu; Jimin Xiao
>
> **备注:** Accepted by CVPR26 Findings
>
> **摘要:** Referring image segmentation aims to segment specific targets based on a natural text expression. Recently, parameter-efficient tuning (PET) has emerged as a promising paradigm. However, existing PET-based methods often suffer from the fact that visual features can't emphasize the text-referred target instance but activate co-category yet unrelated objects. We analyze and quantify this problem, terming it the `non-target activation' (NTA) issue. To address this, we propose a novel framework, TALENT, which utilizes target-aware efficient tuning for PET-based RIS. Specifically, we first propose a Rectified Cost Aggregator (RCA) to efficiently aggregate text-referred features. Then, to calibrate `NTA' into accurate target activation, we adopt a Target-aware Learning Mechanism (TLM), including contextual pairwise consistency learning and target-centric contrastive learning. The former uses the sentence-level text feature to achieve a holistic understanding of the referent and constructs a text-referred affinity map to optimize the semantic association of visual features. The latter further enhances target localization to discover the distinct instance while suppressing associations with other unrelated ones. The two objectives work in concert and address `NTA' effectively. Extensive evaluations show that TALENT outperforms existing methods across various metrics (e.g., 2.5\% mIoU gains on G-Ref val set). Our codes will be released at: this https URL.
>
---
#### [new 047] JAMMEval: A Refined Collection of Japanese Benchmarks for Reliable VLM Evaluation
- **分类: cs.CV**

- **简介: 该论文属于视觉问答任务，旨在解决日本语VQA基准数据质量低的问题。通过两轮人工标注优化数据，提升评估可靠性。**

- **链接: [https://arxiv.org/pdf/2604.00909](https://arxiv.org/pdf/2604.00909)**

> **作者:** Issa Sugiura; Koki Maeda; Shuhei Kurita; Yusuke Oda; Daisuke Kawahara; Naoaki Okazaki
>
> **备注:** 16 pages, 11 figures
>
> **摘要:** Reliable evaluation is essential for the development of vision-language models (VLMs). However, Japanese VQA benchmarks have undergone far less iterative refinement than their English counterparts. As a result, many existing benchmarks contain issues such as ambiguous questions, incorrect answers, and instances that can be solved without visual grounding, undermining evaluation reliability and leading to misleading conclusions in model comparisons. To address these limitations, we introduce JAMMEval, a refined collection of Japanese benchmarks for reliable VLM evaluation. It is constructed by systematically refining seven existing Japanese benchmark datasets through two rounds of human annotation, improving both data quality and evaluation reliability. In our experiments, we evaluate open-weight and proprietary VLMs on JAMMEval and analyze the capabilities of recent models on Japanese VQA. We further demonstrate the effectiveness of our refinement by showing that the resulting benchmarks yield evaluation scores that better reflect model capability, exhibit lower run-to-run variance, and improve the ability to distinguish between models of different capability levels. We release our dataset and code to advance reliable evaluation of VLMs.
>
---
#### [new 048] TTA-Vid: Generalized Test-Time Adaptation for Video Reasoning
- **分类: cs.CV**

- **简介: 该论文提出TTA-Vid，解决视频推理中的领域适应问题。通过测试时强化学习，无需标注数据即可适应新视频数据，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.00696](https://arxiv.org/pdf/2604.00696)**

> **作者:** Soumya Shamarao Jahagirdar; Edson Araujo; Anna Kukleva; M. Jehanzeb Mirza; Saurabhchand Bhati; Samuel Thomas; Brian Kingsbury; Rogerio Feris; James R. Glass; Hilde Kuehne
>
> **摘要:** Recent video reasoning models have shown strong results on temporal and multimodal understanding, yet they depend on large-scale supervised data and multi-stage training pipelines, making them costly to train and difficult to adapt to new domains. In this work, we leverage the paradigm of Test-Time Reinforcement Learning on video-language data to allow for adapting a pretrained model to incoming video samples at test-time without explicit labels. The proposed test-time adaptation for video approach (TTA-Vid) combines two components that work simultaneously: (1) a test-time adaptation that performs step-by-step reasoning at inference time on multiple frame subsets. We then use a batch-aware frequency-based reward computed across different frame subsets as pseudo ground truth to update the model. It shows that the resulting model trained on a single batch or even a single sample from a dataset, is able to generalize at test-time to the whole dataset and even across datasets. Because the adaptation occurs entirely at test time, our method requires no ground-truth annotations or dedicated training splits. Additionally, we propose a multi-armed bandit strategy for adaptive frame selection that learns to prioritize informative frames, guided by the same reward formulation. Our evaluation shows that TTA-Vid yields consistent improvements across various video reasoning tasks and is able to outperform current state-of-the-art methods trained on large-scale data. This highlights the potential of test-time reinforcement learning for temporal multimodal understanding.
>
---
#### [new 049] Improving Generalization of Deep Learning for Brain Metastases Segmentation Across Institutions
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，旨在解决深度学习模型在不同机构间泛化能力差的问题。通过VAE-MMD方法提升脑转移瘤分割的跨机构适应性。**

- **链接: [https://arxiv.org/pdf/2604.00397](https://arxiv.org/pdf/2604.00397)**

> **作者:** Yuchen Yang; Shuangyang Zhong; Haijun Yu; Langcuomu Suo; Hongbin Han; Florian Putz; Yixing Huang
>
> **备注:** 5 figures and 1 table
>
> **摘要:** Background: Deep learning has demonstrated significant potential for automated brain metastases (BM) segmentation; however, models trained at a singular institution often exhibit suboptimal performance at various sites due to disparities in scanner hardware, imaging protocols, and patient demographics. The goal of this work is to create a domain adaptation framework that will allow for BM segmentation to be used across multiple institutions. Methods: We propose a VAE-MMD preprocessing pipeline that combines variational autoencoders (VAE) with maximum mean discrepancy (MMD) loss, incorporating skip connections and self-attention mechanisms alongside nnU-Net segmentation. The method was tested on 740 patients from four public databases: Stanford, UCSF, UCLM, and PKG, evaluated by domain classifier's accuracy, sensitivity, precision, F1/F2 scores, surface Dice (sDice), and 95th percentile Hausdorff distance (HD95). Results: VAE-MMD reduced domain classifier accuracy from 0.91 to 0.50, indicating successful feature alignment across institutions. Reconstructed volumes attained a PSNR greater than 36 dB, maintaining anatomical accuracy. The combined method raised the mean F1 by 11.1% (0.700 to 0.778), the mean sDice by 7.93% (0.7121 to 0.7686), and reduced the mean HD95 by 65.5% (11.33 to 3.91 mm) across all four centers compared to the baseline nnU-Net. Conclusions: VAE-MMD effectively diminishes cross-institutional data heterogeneity and enhances BM segmentation generalization across volumetric, detection, and boundary-level metrics without necessitating target-domain labels, thereby overcoming a significant obstacle to the clinical implementation of AI-assisted segmentation.
>
---
#### [new 050] Continual Vision-Language Learning for Remote Sensing: Benchmarking and Analysis
- **分类: cs.CV**

- **简介: 该论文属于遥感视觉语言学习任务，旨在解决模型持续适应新模态和任务时的灾难性遗忘问题。提出CLeaRS基准，评估模型在多种场景下的持续学习能力。**

- **链接: [https://arxiv.org/pdf/2604.00820](https://arxiv.org/pdf/2604.00820)**

> **作者:** Xingxing Weng; Ruifeng Ni; Chao Pang; XiangYu Hao; Yishan Wang; Xiaokang Zhang; Wei Xu; Gui-Song Xia
>
> **备注:** 23 pages, 7 figures, 9 tables
>
> **摘要:** Current remote sensing vision-language models (RS VLMs) demonstrate impressive performance in image interpretation but rely on static training data, limiting their ability to accommodate continuously emerging sensing modalities and downstream tasks. This exposes a fundamental challenge: enabling RS VLMs to continually adapt without catastrophic forgetting. Despite its practical importance, the continual learning capability of RS VLMs remains underexplored, and no dedicated benchmark currently exists. In this work, we present CLeaRS, a comprehensive benchmark for continual vision-language learning in remote sensing. CLeaRS comprises 10 curated subsets with over 207k image-text pairs, spanning diverse interpretation tasks, sensing modalities, and application scenarios. We further define three evaluation protocols: long-horizon, modality-incremental, and task-incremental settings, to systematically assess continual adaptation. Extensive benchmarking of diverse vision-language models reveals catastrophic forgetting across all settings. Moreover, representative continual learning methods, when adapted to RS VLMs, exhibit limited effectiveness in handling task, instruction, and modality transitions. Our findings underscore the need for developing continual learning methods tailored to RS VLMs.
>
---
#### [new 051] Customizing Large Vision Model-Guided Low-Rank Approximation for Ground-Roll Denoise
- **分类: cs.CV**

- **简介: 该论文属于地震数据降噪任务，旨在解决地面波噪声干扰反射信号的问题。通过引入大视觉模型引导的低秩近似方法，实现无需训练的噪声抑制与反射保留。**

- **链接: [https://arxiv.org/pdf/2604.00998](https://arxiv.org/pdf/2604.00998)**

> **作者:** Jiacheng Liao; Feng Qian; Ziyin Fan; Yongjian Guo
>
> **摘要:** Ground-roll is a dominant source of coherent noise in land and vertical seismic profiling (VSP) data, severely masking reflection events and degrading subsequent imaging and interpretation. Conventional attenuation methods, including transform-domain filtering, sparse representation, and deep learning, often suffer from limited adaptability, signal leakage, or dependence on labeled training data, especially under strong signal-noise overlap. To address these challenges, we propose a training-free framework that reformulates ground-roll attenuation as a semantic-guided signal separation problem. Specifically, a promptable large vision model is employed to extract high-level semantic priors by converting seismic gathers into visual representations and localizing ground-roll-dominant regions via text or image prompts. The resulting semantic response is transformed into a continuous soft mask, which is embedded into a mask-conditioned low-rank inverse formulation to enable spatially adaptive suppression and reflection-preserving reconstruction. An efficient alternating direction method of multipliers (ADMM)-based solver is further developed to solve the proposed inverse problem, enabling stable and physically consistent signal recovery without requiring task-specific training or manual annotation. Extensive experiments on both synthetic and field VSP datasets demonstrate that the proposed method achieves superior ground-roll attenuation while preserving reflection continuity and waveform fidelity, consistently outperforming representative transform-domain filtering and implicit neural representation methods.
>
---
#### [new 052] Benchmarking Interaction, Beyond Policy: a Reproducible Benchmark for Collaborative Instance Object Navigation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出QAsk-Nav，一个用于协作实例物体导航（CoIN）的可复现基准，解决导航与协作问答分离评估的问题。通过改进导航协议和引入问答协议，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.00265](https://arxiv.org/pdf/2604.00265)**

> **作者:** Edoardo Zorzi; Francesco Taioli; Yiming Wang; Marco Cristani; Alessandro Farinelli; Alberto Castellini; Loris Bazzani
>
> **摘要:** We propose Question-Asking Navigation (QAsk-Nav), the first reproducible benchmark for Collaborative Instance Object Navigation (CoIN) that enables an explicit, separate assessment of embodied navigation and collaborative question asking. CoIN tasks an embodied agent with reaching a target specified in free-form natural language under partial observability, using only egocentric visual observations and interactive natural-language dialogue with a human, where the dialogue can help to resolve ambiguity among visually similar object instances. Existing CoIN benchmarks are primarily focused on navigation success and offer no support for consistent evaluation of collaborative interaction. To address this limitation, QAsk-Nav provides (i) a lightweight question-asking protocol scored independently of navigation, (ii) an enhanced navigation protocol with realistic, diverse, high-quality target descriptions, and (iii) an open-source dataset, that includes 28,000 quality-checked reasoning and question-asking traces for training and analysis of interactive capabilities of CoIN models. Using the proposed QAsk-Nav benchmark, we develop Light-CoNav, a lightweight unified model for collaborative navigation that is 3x smaller and 70x faster than existing modular methods, while outperforming state-of-the-art CoIN approaches in generalization to unseen objects and environments. Project page at this https URL
>
---
#### [new 053] Neural Harmonic Textures for High-Quality Primitive Based Neural Reconstruction
- **分类: cs.CV; cs.AI; cs.GR; cs.LG**

- **简介: 该论文属于三维重建任务，旨在解决高频率细节建模难题。提出Neural Harmonic Textures方法，通过周期性激活提升渲染质量，降低计算成本。**

- **链接: [https://arxiv.org/pdf/2604.01204](https://arxiv.org/pdf/2604.01204)**

> **作者:** Jorge Condor; Nicolas Moenne-Loccoz; Merlin Nimier-David; Piotr Didyk; Zan Gojcic; Qi Wu
>
> **摘要:** Primitive-based methods such as 3D Gaussian Splatting have recently become the state-of-the-art for novel-view synthesis and related reconstruction tasks. Compared to neural fields, these representations are more flexible, adaptive, and scale better to large scenes. However, the limited expressivity of individual primitives makes modeling high-frequency detail challenging. We introduce Neural Harmonic Textures, a neural representation approach that anchors latent feature vectors on a virtual scaffold surrounding each primitive. These features are interpolated within the primitive at ray intersection points. Inspired by Fourier analysis, we apply periodic activations to the interpolated features, turning alpha blending into a weighted sum of harmonic components. The resulting signal is then decoded in a single deferred pass using a small neural network, significantly reducing computational cost. Neural Harmonic Textures yield state-of-the-art results in real-time novel view synthesis while bridging the gap between primitive- and neural-field-based reconstruction. Our method integrates seamlessly into existing primitive-based pipelines such as 3DGUT, Triangle Splatting, and 2DGS. We further demonstrate its generality with applications to 2D image fitting and semantic reconstruction.
>
---
#### [new 054] EmoScene: A Dual-space Dataset for Controllable Affective Image Generation
- **分类: cs.CV**

- **简介: 该论文提出EmoScene数据集，用于可控情感图像生成任务。解决现有模型对场景语义和情感细节控制不足的问题，通过双空间标注提升情感可控性。**

- **链接: [https://arxiv.org/pdf/2604.00933](https://arxiv.org/pdf/2604.00933)**

> **作者:** Li He; Longtai Zhang; Wenqiang Zhang; Yan Wang; Lizhe Qi
>
> **摘要:** Text-to-image diffusion models have achieved high visual fidelity, yet precise control over scene semantics and fine-grained affective tone remains challenging. Human visual affect arises from the rapid integration of contextual meaning, including valence, arousal, and dominance, with perceptual cues such as color harmony, luminance contrast, texture variation, curvature, and spatial layout. However, current text-to-image models rarely represent affective and perceptual factors within a unified representation, which limits their ability to synthesize scenes with coherent and nuanced emotional intent. To address this gap, we construct EmoScene, a large-scale dual-space emotion dataset that jointly encodes affective dimensions and perceptual attributes, with contextual semantics provided as supporting annotations. EmoScene contains 1.2M images across more than three hundred real-world scene categories, each annotated with discrete emotion labels, continuous VAD values, perceptual descriptors and textual captions. Multi-space analyses reveal how discrete emotions occupy the VAD space and how affect systematically correlates with scene-level perceptual factors. To benchmark EmoScene, we provide a lightweight reference baseline that injects dual-space controls into a frozen diffusion backbone via shallow cross-attention modulation, serving as a reproducible probe of affect controllability enabled by dual-space supervision.
>
---
#### [new 055] Out of Sight, Out of Track: Adversarial Attacks on Propagation-based Multi-Object Trackers via Query State Manipulation
- **分类: cs.CV**

- **简介: 该论文属于多目标跟踪任务，针对基于查询传播的跟踪器提出对抗攻击方法FADE，解决其在查询状态上的脆弱性问题。**

- **链接: [https://arxiv.org/pdf/2604.00452](https://arxiv.org/pdf/2604.00452)**

> **作者:** Halima Bouzidi; Haoyu Liu; Yonatan Gizachew Achamyeleh; Praneetsai Vasu Iddamsetty; Mohammad Abdullah Al Faruque
>
> **备注:** Accepted for presentation at CVPR 2026 (main track)
>
> **摘要:** Recent Tracking-by-Query-Propagation (TBP) methods have advanced Multi-Object Tracking (MOT) by enabling end-to-end (E2E) pipelines with long-range temporal modeling. However, this reliance on query propagation introduces unexplored architectural vulnerabilities to adversarial attacks. We present FADE, a novel attack framework designed to exploit these specific vulnerabilities. FADE employs two attack strategies targeting core TBP mechanisms: (i) Temporal Query Flooding: Generates spurious temporally consistent track queries to exhaust the tracker's limited query budget, forcing it to terminate valid tracks. (ii) Temporal Memory Corruption: Directly attacks the query updater's memory by severing temporal links via state de-correlation and erasing the learned feature identity of matched tracks. Furthermore, we introduce a differentiable pipeline to optimize these attacks for physical-world realizability by leveraging simulations of advanced perception sensor spoofing. Experiments on MOT17 and MOT20 benchmarks demonstrate that FADE is highly effective against state-of-the-art TBP trackers, causing significant identity switches and track terminations.
>
---
#### [new 056] Neural Reconstruction of LiDAR Point Clouds under Jamming Attacks via Full-Waveform Representation and Simultaneous Laser Sensing
- **分类: cs.CV**

- **简介: 该论文属于LiDAR安全任务，解决 jamming 攻击导致的点云失效问题。通过全波形重建和同步激光感知，提出PULSAR-Net实现攻击下点云恢复。**

- **链接: [https://arxiv.org/pdf/2604.00371](https://arxiv.org/pdf/2604.00371)**

> **作者:** Ryo Yoshida; Takami Sato; Wenlun Zhang; Yuki Hayakawa; Shota Nagai; Takahiro Kado; Taro Beppu; Ibuki Fujioka; Yunshan Zhong; Kentaro Yoshioka
>
> **摘要:** LiDAR sensors are critical for autonomous driving perception, yet remain vulnerable to spoofing attacks. Jamming attacks inject high-frequency laser pulses that completely blind LiDAR sensors by overwhelming authentic returns with malicious signals. We discover that while point clouds become randomized, the underlying full-waveform data retains distinguishable signatures between attack and legitimate signals. In this work, we propose PULSAR-Net, capable of reconstructing authentic point clouds under jamming attacks by leveraging previously underutilized intermediate full-waveform representations and simultaneous laser sensing in modern LiDAR systems. PULSAR-Net adopts a novel U-Net architecture with axial spatial attention mechanisms specifically designed to identify attack-induced signals from authentic object returns in the full-waveform representation. To address the lack of full-waveform representations in existing LiDAR datasets under jamming attacks, we introduce a physics-aware dataset generation pipeline that synthesizes realistic full-waveform representations under jamming attacks. Despite being trained exclusively on synthetic data, PULSAR-Net achieves reconstruction rates of 92% and 73% for vehicles obscured by jamming attacks in real-world static and driving scenarios, respectively.
>
---
#### [new 057] KG-CMI: Knowledge graph enhanced cross-Mamba interaction for medical visual question answering
- **分类: cs.CV**

- **简介: 该论文属于医学视觉问答任务，旨在解决医学知识利用不足和答案形式单一的问题。提出KG-CMI框架，融合知识图谱与跨模态交互，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.00601](https://arxiv.org/pdf/2604.00601)**

> **作者:** Xianyao Zheng; Hong Yu; Hui Cui; Changming Sun; Xiangyu Li; Ran Su; Leyi Wei; Jia Zhou; Junbo Wang; Qiangguo Jin
>
> **摘要:** Medical visual question answering (Med-VQA) is a crucial multimodal task in clinical decision support and telemedicine. Recent methods fail to fully leverage domain-specific medical knowledge, making it difficult to accurately associate lesion features in medical images with key diagnostic criteria. Additionally, classification-based approaches typically rely on predefined answer sets. Treating Med-VQA as a simple classification problem limits its ability to adapt to the diversity of free-form answers and may overlook detailed semantic information in those answers. To address these challenges, we propose a knowledge graph enhanced cross-Mamba interaction (KG-CMI) framework, which consists of a fine-grained cross-modal feature alignment (FCFA) module, a knowledge graph embedding (KGE) module, a cross-modal interaction representation (CMIR) module, and a free-form answer enhanced multi-task learning (FAMT) module. The KG-CMI learns cross-modal feature representations for images and texts by effectively integrating professional medical knowledge through a graph, establishing associations between lesion features and disease knowledge. Moreover, FAMT leverages auxiliary knowledge from open-ended questions, improving the model's capability for open-ended Med-VQA. Experimental results demonstrate that KG-CMI outperforms existing state-of-the-art methods on three Med-VQA datasets, i.e., VQA-RAD, SLAKE, and OVQA. Additionally, we conduct interpretability experiments to further validate the framework's effectiveness.
>
---
#### [new 058] PrivHAR-Bench: A Graduated Privacy Benchmark Dataset for Video-Based Action Recognition
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于视频动作识别任务，旨在解决隐私与识别性能之间的平衡问题。提出PrivHAR-Bench数据集，通过多层级隐私变换评估隐私-效用权衡。**

- **链接: [https://arxiv.org/pdf/2604.00761](https://arxiv.org/pdf/2604.00761)**

> **作者:** Samar Ansari
>
> **摘要:** Existing research on privacy-preserving Human Activity Recognition (HAR) typically evaluates methods against a binary paradigm: clear video versus a single privacy transformation. This limits cross-method comparability and obscures the nuanced relationship between privacy strength and recognition utility. We introduce \textit{PrivHAR-Bench}, a multi-tier benchmark dataset designed to standardize the evaluation of the \textit{Privacy-Utility Trade-off} in video-based action recognition. PrivHAR-Bench applies a graduated spectrum of visual privacy transformations: from lightweight spatial obfuscation to cryptographic block permutation, to a curated subset of 15 activity classes selected for human articulation diversity. Each of the 1,932 source videos is distributed across 9 parallel tiers of increasing privacy strength, with additional background-removed variants to isolate the contribution of human motion features from contextual scene bias. We provide lossless frame sequences, per-frame bounding boxes, estimated pose keypoints with joint-level confidence scores, standardized group-based train/test splits, and an evaluation toolkit computing recognition accuracy and privacy metrics. Empirical validation using R3D-18 demonstrates a measurable and interpretable degradation curve across tiers, with within-tier accuracy declining from 88.8\% (clear) to 53.5\% (encrypted, background-removed) and cross-domain accuracy collapsing to 4.8\%, establishing PrivHAR-Bench as a controlled benchmark for comparing privacy-preserving HAR methods under standardized conditions. The dataset, generation pipeline, and evaluation code are publicly available.
>
---
#### [new 059] Fluently Lying: Adversarial Robustness Can Be Substrate-Dependent
- **分类: cs.CV**

- **简介: 论文研究对抗攻击下目标检测器的鲁棒性，发现某些模型在准确率下降时仍保持检测数量，称为质量退化（QC）。该工作属于目标检测的对抗鲁棒性研究，揭示了防御机制可能依赖特定模型结构。**

- **链接: [https://arxiv.org/pdf/2604.00605](https://arxiv.org/pdf/2604.00605)**

> **作者:** Daye Kang; Hyeongboo Baek
>
> **备注:** 14 pages, 4 figures, 3 tables
>
> **摘要:** The primary tools used to monitor and defend object detectors under adversarial attack assume that when accuracy degrades, detection count drops in tandem. This coupling was assumed, not measured. We report a counterexample observed on a single model: under standard PGD, EMS-YOLO, a spiking neural network (SNN) object detector, retains more than 70% of its detections while mAP collapses from 0.528 to 0.042. We term this count-preserving accuracy collapse Quality Corruption (QC), to distinguish it from the suppression that dominates untargeted evaluation. Across four SNN architectures and two threat models (l-infinity and l-2), QC appears only in one of the four detectors tested (EMS-YOLO). On this model, all five standard defense components fail to detect or mitigate QC, suggesting the defense ecosystem may rely on a shared assumption calibrated on a single substrate. These results provide, to our knowledge, the first evidence that adversarial failure modes can be substrate-dependent.
>
---
#### [new 060] VADMamba++: Efficient Video Anomaly Detection via Hybrid Modeling in Grayscale Space
- **分类: cs.CV**

- **简介: 该论文属于视频异常检测任务，解决传统方法依赖辅助输入的问题。提出VADMamba++，通过灰度到RGB的单任务建模，提升检测效果与效率。**

- **链接: [https://arxiv.org/pdf/2604.00360](https://arxiv.org/pdf/2604.00360)**

> **作者:** Jihao Lyu; Minghua Zhao; Jing Hu; Yifei Chen; Shuangli Du; Cheng Shi
>
> **摘要:** VADMamba pioneered the introduction of Mamba to Video Anomaly Detection (VAD), achieving high accuracy and fast inference through hybrid proxy tasks. Nevertheless, its heavy reliance on optical flow as auxiliary input and inter-task fusion scoring constrains its applicability to a single proxy task. In this paper, we introduce VADMamba++, an efficient VAD method based on the Gray-to-RGB paradigm that enforces a Single-Channel to Three-Channel reconstruction mapping, designed for a single proxy task and operating without auxiliary inputs. This paradigm compels inferring color appearances from grayscale structures, allowing anomalies to be more effectively revealed through dual inconsistencies between structure and chromatic cues. Specifically, VADMamba++ reconstructs grayscale frames into the RGB space to simultaneously discriminate structural geometry and chromatic fidelity, thereby enhancing sensitivity to explicit visual anomalies. We further design a hybrid modeling backbone that integrates Mamba, CNN, and Transformer modules to capture diverse normal patterns while suppressing the appearance of anomalies. Furthermore, an intra-task fusion scoring strategy integrates explicit future-frame prediction errors with implicit quantized feature errors, further improving accuracy under a single task setting. Extensive experiments on three benchmark datasets demonstrate that VADMamba++ outperforms state-of-the-art methods while meeting performance and efficiency, especially under a strict single-task setting with only frame-level inputs.
>
---
#### [new 061] FecalFed: Privacy-Preserving Poultry Disease Detection via Federated Learning
- **分类: cs.CV**

- **简介: 该论文属于禽类疾病检测任务，解决数据隐私和数据孤岛问题。提出FecalFed框架，通过联邦学习实现隐私保护的疾病分类，提升检测准确性。**

- **链接: [https://arxiv.org/pdf/2604.00559](https://arxiv.org/pdf/2604.00559)**

> **作者:** Tien-Yu Chi
>
> **备注:** Accepted to the CVPR 2026 Workshop on Vision for Agriculture
>
> **摘要:** Early detection of highly pathogenic avian influenza (HPAI) and endemic poultry diseases is critical for global food security. While computer vision models excel at classifying diseases from fecal imaging, deploying these systems at scale is bottlenecked by farm data privacy concerns and institutional data silos. Furthermore, existing open-source agricultural datasets frequently suffer from severe, undocumented data contamination. In this paper, we introduce $\textbf{FecalFed}$, a privacy-preserving federated learning framework for poultry disease classification. We first curate and release $\texttt{poultry-fecal-fl}$, a rigorously deduplicated dataset of 8,770 unique images across four disease classes, revealing and eliminating a 46.89$\%$ duplication rate in popular public repositories. To simulate realistic agricultural environments, we evaluate FecalFed under highly heterogeneous, non-IID conditions (Dirichlet $\alpha=0.5$). While isolated single-farm training collapses under this data heterogeneity, yielding only 64.86$\%$ accuracy, our federated approach recovers performance without centralizing sensitive data. Specifically, utilizing server-side adaptive optimization (FedAdam) with a Swin-Small architecture achieves 90.31$\%$ accuracy, closely approaching the centralized upper bound of 95.10\%. Furthermore, we demonstrate that an edge-optimized Swin-Tiny model maintains highly competitive performance at 89.74$\%$, establishing a highly efficient, privacy-first blueprint for on-farm avian disease monitoring.
>
---
#### [new 062] Excite, Attend and Segment (EASe): Domain-Agnostic Fine-Grained Mask Discovery with Feature Calibration and Self-Supervised Upsampling
- **分类: cs.CV**

- **简介: 该论文提出EASe框架，解决复杂场景下的细粒度语义分割问题。通过特征校准和自监督上采样，提升分割精度。**

- **链接: [https://arxiv.org/pdf/2604.00276](https://arxiv.org/pdf/2604.00276)**

> **作者:** Deepank Singh; Anurag Nihal; Vedhus Hoskere
>
> **摘要:** Unsupervised segmentation approaches have increasingly leveraged foundation models (FM) to improve salient object discovery. However, these methods often falter in scenes with complex, multi-component morphologies, where fine-grained structural detail is indispensable. Many state-of-the-art unsupervised segmentation pipelines rely on mask discovery approaches that utilize coarse, patch-level representations. These coarse representations inherently suppress the fine-grained detail required to resolve such complex morphologies. To overcome this limitation, we propose Excite, Attend and Segment (EASe), an unsupervised domain-agnostic semantic segmentation framework for easy fine-grained mask discovery across challenging real-world scenes. EASe utilizes novel Semantic-Aware Upsampling with Channel Excitation (SAUCE) to excite low-resolution FM feature channels for selective calibration and attends across spatially-encoded image and FM features to recover full-resolution semantic representations. Finally, EASe segments the aggregated features into multi-granularity masks using a novel training-free Cue-Attentive Feature Aggregator (CAFE) which leverages SAUCE attention scores as a semantic grouping signal. EASe, together with SAUCE and CAFE, operate directly at pixel-level feature representations to enable accurate fine-grained dense semantic mask discovery. Our evaluation demonstrates superior performance of EASe over previous state-of-the-arts (SOTAs) across major standard benchmarks and diverse datasets with complex morphologies. Code is available at this https URL
>
---
#### [new 063] A 4D Representation for Training-Free Agentic Reasoning from Monocular Laparoscopic Video
- **分类: cs.CV**

- **简介: 该论文属于医疗AI任务，旨在提升手术视频中的时空推理能力。通过构建4D表示，结合2D和3D模型，实现无需训练的智能体推理。**

- **链接: [https://arxiv.org/pdf/2604.00867](https://arxiv.org/pdf/2604.00867)**

> **作者:** Maximilian Fehrentz; Nicolas Stellwag; Robert Wiebe; Nicole Thorisch; Fabian Grob; Patrick Remerscheid; Ken-Joel Simmoteit; Benjamin D. Killeen; Christian Heiliger; Nassir Navab
>
> **摘要:** Spatiotemporal reasoning is a fundamental capability for artificial intelligence (AI) in soft tissue surgery, paving the way for intelligent assistive systems and autonomous robotics. While 2D vision-language models show increasing promise at understanding surgical video, the spatial complexity of surgical scenes suggests that reasoning systems may benefit from explicit 4D representations. Here, we propose a framework for equipping surgical agents with spatiotemporal tools based on an explicit 4D representation, enabling AI systems to ground their natural language reasoning in both time and 3D space. Leveraging models for point tracking, depth, and segmentation, we develop a coherent 4D model with spatiotemporally consistent tool and tissue semantics. A Multimodal Large Language Model (MLLM) then acts as an agent on tools derived from the explicit 4D representation (e.g., trajectories) without any fine-tuning. We evaluate our method on a new dataset of 134 clinically relevant questions and find that the combination of a general purpose reasoning backbone and our 4D representation significantly improves spatiotemporal understanding and allows for 4D grounding. We demonstrate that spatiotemporal intelligence can be "assembled" from 2D MLLMs and 3D computer vision models without additional training. Code, data, and examples are available at this https URL
>
---
#### [new 064] FreqPhys: Repurposing Implicit Physiological Frequency Prior for Robust Remote Photoplethysmography
- **分类: cs.CV**

- **简介: 该论文属于远程光体积描记（rPPG）任务，旨在解决运动伪影和光照变化导致的信号不稳定问题。提出FreqPhys框架，通过生理频率先验提升信号恢复的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.00534](https://arxiv.org/pdf/2604.00534)**

> **作者:** Wei Qian; Dan Guo; Jinxing Zhou; Bochao Zou; Zitong Yu; Meng Wang
>
> **摘要:** Remote photoplethysmography (rPPG) enables contactless physiological monitoring by capturing subtle skin-color variations from facial videos. However, most existing methods predominantly rely on time-domain modeling, making them vulnerable to motion artifacts and illumination fluctuations, where weak physiological clues are easily overwhelmed by noise. To address these challenges, we propose FreqPhys, a frequency-guided rPPG framework that explicitly leverages physiological frequency priors for robust signal recovery. Specifically, FreqPhys first applies a Physiological Bandpass Filtering module to suppress out-of-band interference, and then performs Physiological Spectrum Modulation together with adaptive spectral selection to emphasize pulse-related frequency components while suppress residual in-band noise. A Cross-domain Representation Learning module further fuses these spectral priors with deep time-domain features to capture informative spatial--temporal dependencies. Finally, a frequency-aware conditional diffusion process progressively reconstructs high-fidelity rPPG signals. Extensive experiments on six benchmarks demonstrate that FreqPhys yields significant improvements over state-of-the-art approaches, particularly under challenging motion conditions. It highlights the importance of explicitly modeling physiological frequency priors. The source code will be released.
>
---
#### [new 065] TP-Seg: Task-Prototype Framework for Unified Medical Lesion Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学病灶分割任务，旨在解决统一模型在多种病灶类型和模态下的分割难题。提出TP-Seg框架，通过任务适配器和原型引导解码器提升分割性能。**

- **链接: [https://arxiv.org/pdf/2604.00684](https://arxiv.org/pdf/2604.00684)**

> **作者:** Jiawei Xu; Qiangqiang Zhou; Dandan Zhu; Yong Chen; Yugen Yi; Xiaoqi Zhao
>
> **摘要:** Building a unified model with a single set of parameters to efficiently handle diverse types of medical lesion segmentation has become a crucial objective for AI-assisted diagnosis. Existing unified segmentation approaches typically rely on shared encoders across heterogeneous tasks and modalities, which often leads to feature entanglement, gradient interference, and suboptimal lesion discrimination. In this work, we propose TP-Seg, a task-prototype framework for unified medical lesion segmentation. On one hand, the task-conditioned adapter effectively balances shared and task-specific representations through a dual-path expert structure, enabling adaptive feature extraction across diverse medical imaging modalities and lesion types. On the other hand, the prototype-guided task decoder introduces learnable task prototypes as semantic anchors and employs a cross-attention mechanism to achieve fine-grained modeling of task-specific foreground and background semantics. Without bells and whistles, TP-Seg consistently outperforms specialized, general and unified segmentation methods across 8 different medical lesion segmentation tasks covering multiple imaging modalities, demonstrating strong generalization, scalability and clinical applicability.
>
---
#### [new 066] Looking into a Pixel by Nonlinear Unmixing -- A Generative Approach
- **分类: cs.CV; cs.AI; eess.IV**

- **简介: 该论文属于高光谱图像非线性混叠任务，旨在无需先验混合模型的情况下实现有效解混。通过构建双向GAN框架，结合循环一致性和线性关联约束，提出LCGU网络进行非线性解混。**

- **链接: [https://arxiv.org/pdf/2604.01141](https://arxiv.org/pdf/2604.01141)**

> **作者:** Maofeng Tang; Hairong Qi
>
> **摘要:** Due to the large footprint of pixels in remote sensing imagery, hyperspectral unmixing (HU) has become an important and necessary procedure in hyperspectral image analysis. Traditional HU methods rely on a prior spectral mixing model, especially for nonlinear mixtures, which has largely limited the performance and generalization capacity of the unmixing approach. In this paper, we address the challenging problem of hyperspectral nonlinear unmixing (HNU) without explicit knowledge of the mixing model. Inspired by the principle of generative models, where images of the same distribution can be generated as that of the training images without knowing the exact probability distribution function of the image, we develop an invertible mixing-unmixing process via a bi-directional GAN framework, constrained by both the cycle consistency and the linkage between linear and nonlinear mixtures. The combination of cycle consistency and linear linkage provides powerful constraints without requiring an explicit mixing model. We refer to the proposed approach as the linearly-constrained CycleGAN unmixing net, or LCGU net. Experimental results indicate that the proposed LCGU net exhibits stable and competitive performance across different datasets compared with other state-of-the-art model-based HNU methods.
>
---
#### [new 067] Perturb-and-Restore: Simulation-driven Structural Augmentation Framework for Imbalance Chromosomal Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文属于染色体异常检测任务，解决数据不平衡和稀缺问题。提出Perturb-and-Restore框架，通过模拟生成合成异常数据并优化采样策略，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2604.00854](https://arxiv.org/pdf/2604.00854)**

> **作者:** Yilan Zhang; Hanbiao Chen; Changchun Yang; Yuetan Chu; Siyuan Chen; Jing Wu; Jingdong Hu; Na Li; Junkai Su; Yuxuan Chen; Ao Xu; Xin Gao; Aihua Yin
>
> **备注:** This preprint version of the manuscript has been submitted to the IEEE Journal of Biomedical and Health Informatics (JBHI) for review
>
> **摘要:** Detecting structural chromosomal abnormalities is crucial for accurate diagnosis and management of genetic disorders. However, collecting sufficient structural abnormality data is extremely challenging and costly in clinical practice, and not all abnormal types can be readily collected. As a result, deep learning approaches face significant performance degradation due to the severe imbalance and scarcity of abnormal chromosome data. To address this challenge, we propose a Perturb-and-Restore (P&R), a simulation-driven structural augmentation framework that effectively alleviates data imbalance in chromosome anomaly detection. The P&R framework comprises two key components: (1) Structure Perturbation and Restoration Simulation, which generates synthetic abnormal chromosomes by perturbing chromosomal banding patterns of normal chromosomes followed by a restoration diffusion network that reconstructs continuous chromosome content and edges, thus eliminating reliance on rare abnormal samples; and (2) Energy-guided Adaptive Sampling, an energy score-based online selection strategy that dynamically prioritizes high-quality synthetic samples by referencing the energy distribution of real samples. To evaluate our method, we construct a comprehensive structural anomaly dataset consisting of over 260,000 chromosome images, including 4,242 abnormal samples spanning 24 categories. Experimental results demonstrate that the P&R framework achieves state-of-the-art (SOTA) performance, surpassing existing methods with an average improvement of 8.92% in sensitivity, 8.89% in precision, and 13.79% in F1-score across all categories.
>
---
#### [new 068] IWP: Token Pruning as Implicit Weight Pruning in Large Vision Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉语言模型优化任务，旨在解决计算成本高的问题。通过隐式权重剪枝方法，提升模型效率与性能平衡。**

- **链接: [https://arxiv.org/pdf/2604.00757](https://arxiv.org/pdf/2604.00757)**

> **作者:** Dong-Jae Lee; Sunghyun Baek; Junmo Kim
>
> **摘要:** Large Vision Language Models show impressive performance across image and video understanding tasks, yet their computational cost grows rapidly with the number of visual tokens. Existing token pruning methods mitigate this issue through empirical approaches while overlooking the internal mechanism of attention. In this paper, we propose a novel training free token pruning framework grounded in the dual form perspective of attention. We reformulate attention as an implicit linear layer whose weight matrix is the sum of rank 1 outer products, each generated by a single token's key value pair. Token pruning thus reduces to selecting an optimal subset of these rank 1 updates that best approximates the original dual weight matrix. Extending this perspective to standard softmax attention in LVLMs, we derive a novel metric quantifying both a token's information magnitude and information duplication. To efficiently select the subset with the proposed metric, we introduce Progressive Chunked Maximal Marginal Relevance. Extensive experiments demonstrate that our method achieves a better trade off between performance and efficiency, while providing another perspective on existing pruning approaches.
>
---
#### [new 069] Sparkle: A Robust and Versatile Representation for Point Cloud based Human Motion Capture
- **分类: cs.CV**

- **简介: 该论文属于人体动作捕捉任务，解决点云数据中表达与鲁棒性难以平衡的问题。提出Sparkle框架，融合骨骼与表面结构，提升准确性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.00857](https://arxiv.org/pdf/2604.00857)**

> **作者:** Yiming Ren; Yujing Sun; Aoru Xue; Kwok-Yan Lam; Yuexin Ma
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Point cloud-based motion capture leverages rich spatial geometry and privacy-preserving sensing, but learning robust representations from noisy, unstructured point clouds remains challenging. Existing approaches face a struggle trade-off between point-based methods (geometrically detailed but noisy) and skeleton-based ones (robust but oversimplified). We address the fundamental challenge: how to construct an effective representation for human motion capture that can balance expressiveness and robustness. In this paper, we propose Sparkle, a structured representation unifying skeletal joints and surface anchors with explicit kinematic-geometric factorization. Our framework, SparkleMotion, learns this representation through hierarchical modules embedding geometric continuity and kinematic constraints. By explicitly disentangling internal kinematic structure from external surface geometry, SparkleMotion achieves state-of-the-art performance not only in accuracy but crucially in robustness and generalization under severe domain shifts, noise, and occlusion. Extensive experiments demonstrate our superiority across diverse sensor types and challenging real-world scenarios.
>
---
#### [new 070] PixelPrune: Pixel-Level Adaptive Visual Token Reduction via Predictive Coding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出PixelPrune，解决视觉语言模型中高分辨率输入带来的计算负担问题，通过像素级冗余压缩提升推理效率。**

- **链接: [https://arxiv.org/pdf/2604.00886](https://arxiv.org/pdf/2604.00886)**

> **作者:** Nan Wang; Zhiwei Jin; Chen Chen; Haonan Lu
>
> **摘要:** Document understanding and GUI interaction are among the highest-value applications of Vision-Language Models (VLMs), yet they impose exceptionally heavy computational burden: fine-grained text and small UI elements demand high-resolution inputs that produce tens of thousands of visual tokens. We observe that this cost is largely wasteful -- across document and GUI benchmarks, only 22--71\% of image patches are pixel-unique, the rest being exact duplicates of another patch in the same image. We propose \textbf{PixelPrune}, which exploits this pixel-level redundancy through predictive-coding-based compression, pruning redundant patches \emph{before} the Vision Transformer (ViT) encoder. Because it operates in pixel space prior to any neural computation, PixelPrune accelerates both the ViT encoder and the downstream LLM, covering the full inference pipeline. The method is training-free, requires no learnable parameters, and supports pixel-lossless compression ($\tau{=}0$) as well as controlled lossy compression ($\tau{>}0$). Experiments across three model scales and document and GUI benchmarks show that PixelPrune maintains competitive task accuracy while delivering up to 4.2$\times$ inference speedup and 1.9$\times$ training acceleration. Code is available at this https URL.
>
---
#### [new 071] EgoSim: Egocentric World Simulator for Embodied Interaction Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出EgoSim，解决egocentric场景模拟中的空间一致性与动态更新问题，通过生成交互视频和更新3D场景状态，实现连续模拟。**

- **链接: [https://arxiv.org/pdf/2604.01001](https://arxiv.org/pdf/2604.01001)**

> **作者:** Jinkun Hao; Mingda Jia; Ruiyan Wang; Xihui Liu; Ran Yi; Lizhuang Ma; Jiangmiao Pang; Xudong Xu
>
> **备注:** Project Page: this http URL
>
> **摘要:** We introduce EgoSim, a closed-loop egocentric world simulator that generates spatially consistent interaction videos and persistently updates the underlying 3D scene state for continuous simulation. Existing egocentric simulators either lack explicit 3D grounding, causing structural drift under viewpoint changes, or treat the scene as static, failing to update world states across multi-stage interactions. EgoSim addresses both limitations by modeling 3D scenes as updatable world states. We generate embodiment interactions via a Geometry-action-aware Observation Simulation model, with spatial consistency from an Interaction-aware State Updating module. To overcome the critical data bottleneck posed by the difficulty in acquiring densely aligned scene-interaction training pairs, we design a scalable pipeline that extracts static point clouds, camera trajectories, and embodiment actions from in-the-wild large-scale monocular egocentric videos. We further introduce EgoCap, a capture system that enables low-cost real-world data collection with uncalibrated smartphones. Extensive experiments demonstrate that EgoSim significantly outperforms existing methods in terms of visual quality, spatial consistency, and generalization to complex scenes and in-the-wild dexterous interactions, while supporting cross-embodiment transfer to robotic manipulation. Codes and datasets will be open soon. The project page is at this http URL.
>
---
#### [new 072] CL-VISTA: Benchmarking Continual Learning in Video Large Language Models
- **分类: cs.CV**

- **简介: 该论文属于持续学习任务，旨在解决Video-LLMs在非平稳数据中的适应问题。提出CL-VISTA基准，涵盖8个任务，评估模型的性能、效率和内存占用，揭示方法间的权衡。**

- **链接: [https://arxiv.org/pdf/2604.00677](https://arxiv.org/pdf/2604.00677)**

> **作者:** Haiyang Guo; Yichen Shi; Fei Zhu; Wenzhuo Liu; Hongbo Zhao; Fanhu Zeng; Shijie Ma; Da-Han Wang; Xu-Yao Zhang
>
> **备注:** Preprint
>
> **摘要:** Video Large Language Models (Video-LLMs) require continual learning to adapt to non-stationary real-world data. However, existing benchmarks fall short of evaluating modern foundation models: many still rely on models without large-scale pre-training, and prevailing benchmarks typically partition a single dataset into sub-tasks, resulting in high task redundancy and negligible forgetting on pre-trained Video-LLMs. To address these limitations, we propose CL-VISTA, a benchmark tailored for continual video understanding of Video-LLMs. By curating 8 diverse tasks spanning perception, understanding, and reasoning, CL-VISTA induces substantial distribution shifts that effectively expose catastrophic forgetting. To systematically assess CL methods, we establish a comprehensive evaluation framework comprising 6 distinct protocols across 3 critical dimensions: performance, computational efficiency, and memory footprint. Notably, the performance dimension incorporates a general video understanding assessment to assess whether CL methods genuinely enhance foundational intelligence or merely induce task-specific overfitting. Extensive benchmarking of 10 mainstream CL methods reveals a fundamental trade-off: no single approach achieves universal superiority across all dimensions. Methods that successfully mitigate catastrophic forgetting tend to compromise generalization or incur prohibitive computational and memory overheads. We hope CL-VISTA provides critical insights for advancing continual learning in multimodal foundation models.
>
---
#### [new 073] Multicentric thrombus segmentation using an attention-based recurrent network with gradual modality dropout
- **分类: cs.CV; math.OC**

- **简介: 该论文属于医学图像分割任务，旨在解决3D脑扫描中微小血栓的检测与分割问题，通过引入注意力机制和渐进式模态丢弃提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.00817](https://arxiv.org/pdf/2604.00817)**

> **作者:** Sofia Vargas-Ibarra; Vincent Vigneron; Hichem Maaref; Sonia Garcia-Salicetti
>
> **摘要:** Detecting and delineating tiny targets in 3D brain scans is a central yet under-addressed challenge in medical this http URL ischemic stroke, for instance, the culprit thrombus is small, low-contrast, and variably expressed across modalities(e.g., susceptibility-weighted T2 blooming, diffusion restriction on DWI/ADC), while real-world multi-center dataintroduce domain shifts, anisotropy, and frequent missing sequences. We introduce a methodology that couples an attention-based recurrent segmentation network (UpAttLLSTM), a training schedule that progressively increases the difficulty of hetero-modal learning, with gradual modality dropout, UpAttLLSTM aggregates context across slices via recurrent units (2.5D) and uses attention gates to fuse complementary cues across available sequences, making it robust to anisotropy and class imbalance. Gradual modality dropout systematically simulates site heterogeneity,noise, and missing modalities during training, acting as both augmentation and regularization to improve multi-center generalization. On a monocentric cohort, our approach detects thrombi in >90% of cases with a Dice score of 0.65. In a multi-center setting with missing modalities, it achieves-80% detection with a Dice score around 0.35. Beyond stroke, the proposed methodology directly transfers to other small-lesion tasks in 3D medical imaging where targets are scarce, subtle, and modality-dependent
>
---
#### [new 074] ReinDriveGen: Reinforcement Post-Training for Out-of-Distribution Driving Scene Generation
- **分类: cs.CV**

- **简介: 该论文提出ReinDriveGen，用于生成可控的驾驶场景，解决分布外场景生成问题。通过强化学习提升视频质量，实现安全关键场景的合成。**

- **链接: [https://arxiv.org/pdf/2604.01129](https://arxiv.org/pdf/2604.01129)**

> **作者:** Hao Zhang; Lue Fan; Weikang Bian; Zehuan Wu; Lewei Lu; Zhaoxiang Zhang; Hongsheng Li
>
> **备注:** Project page: this https URL
>
> **摘要:** We present ReinDriveGen, a framework that enables full controllability over dynamic driving scenes, allowing users to freely edit actor trajectories to simulate safety-critical corner cases such as front-vehicle collisions, drifting cars, vehicles spinning out of control, pedestrians jaywalking, and cyclists cutting across lanes. Our approach constructs a dynamic 3D point cloud scene from multi-frame LiDAR data, introduces a vehicle completion module to reconstruct full 360° geometry from partial observations, and renders the edited scene into 2D condition images that guide a video diffusion model to synthesize realistic driving videos. Since such edited scenarios inevitably fall outside the training distribution, we further propose an RL-based post-training strategy with a pairwise preference model and a pairwise reward mechanism, enabling robust quality improvement under out-of-distribution conditions without ground-truth supervision. Extensive experiments demonstrate that ReinDriveGen outperforms existing approaches on edited driving scenarios and achieves state-of-the-art results on novel ego viewpoint synthesis.
>
---
#### [new 075] Autoregressive Appearance Prediction for 3D Gaussian Avatars
- **分类: cs.CV; cs.GR**

- **简介: 该论文属于3D人脸生成任务，旨在解决姿态与外观混淆导致的不稳定问题。通过引入条件隐空间和自回归预测，提升生成质量与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.00928](https://arxiv.org/pdf/2604.00928)**

> **作者:** Michael Steiner; Zhang Chen; Alexander Richard; Vasu Agrawal; Markus Steinberger; Michael Zollhöfer
>
> **备注:** Project Page: this https URL
>
> **摘要:** A photorealistic and immersive human avatar experience demands capturing fine, person-specific details such as cloth and hair dynamics, subtle facial expressions, and characteristic motion patterns. Achieving this requires large, high-quality datasets, which often introduce ambiguities and spurious correlations when very similar poses correspond to different appearances. Models that fit these details during training can overfit and produce unstable, abrupt appearance changes for novel poses. We propose a 3D Gaussian Splatting avatar model with a spatial MLP backbone that is conditioned on both pose and an appearance latent. The latent is learned during training by an encoder, yielding a compact representation that improves reconstruction quality and helps disambiguate pose-driven renderings. At driving time, our predictor autoregressively infers the latent, producing temporally smooth appearance evolution and improved stability. Overall, our method delivers a robust and practical path to high-fidelity, stable avatar driving.
>
---
#### [new 076] Learnability-Guided Diffusion for Dataset Distillation
- **分类: cs.CV**

- **简介: 该论文属于数据集蒸馏任务，旨在减少冗余并提升合成数据质量。通过引入可学习性引导的扩散方法，提高训练效率与性能。**

- **链接: [https://arxiv.org/pdf/2604.00519](https://arxiv.org/pdf/2604.00519)**

> **作者:** Jeffrey A. Chan-Santiago; Mubarak Shah
>
> **备注:** This paper has been accepted to CVPR 2026
>
> **摘要:** Training machine learning models on massive datasets is expensive and time-consuming. Dataset distillation addresses this by creating a small synthetic dataset that achieves the same performance as the full dataset. Recent methods use diffusion models to generate distilled data, either by promoting diversity or matching training gradients. However, existing approaches produce redundant training signals, where samples convey overlapping information. Empirically, disjoint subsets of distilled datasets capture 80-90% overlapping signals. This redundancy stems from optimizing visual diversity or average training dynamics without accounting for similarity across samples, leading to datasets where multiple samples share similar information rather than complementary knowledge. We propose learnability-driven dataset distillation, which constructs synthetic datasets incrementally through successive stages. Starting from a small set, we train a model and generate new samples guided by learnability scores that identify what the current model can learn from, creating an adaptive curriculum. We introduce Learnability-Guided Diffusion (LGD), which balances training utility for the current model with validity under a reference model to generate curriculum-aligned samples. Our approach reduces redundancy by 39.1%, promotes specialization across training stages, and achieves state-of-the-art results on ImageNet-1K (60.1%), ImageNette (87.2%), and ImageWoof (72.9%). Our code is available on our project page this https URL.
>
---
#### [new 077] First Logit Boosting: Visual Grounding Method to Mitigate Object Hallucination in Large Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型任务，旨在解决对象幻觉问题。提出FLB方法，在不增加训练成本的情况下，通过增强视觉信息稳定性来减少幻觉生成。**

- **链接: [https://arxiv.org/pdf/2604.00455](https://arxiv.org/pdf/2604.00455)**

> **作者:** Jiwoo Ha; Jongwoo Baek; Jinhyun So
>
> **备注:** 19 pages, 13 figures
>
> **摘要:** Recent Large Vision-Language Models (LVLMs) have demonstrated remarkable performance across various multimodal tasks that require understanding both visual and linguistic inputs. However, object hallucination -- the generation of nonexistent objects in answers -- remains a persistent challenge. Although several approaches such as retraining and external grounding methods have been proposed to mitigate this issue, they still suffer from high data costs or structural complexity. Training-free methods such as Contrastive Decoding (CD) are more cost-effective, avoiding additional training or external models, but still suffer from long-term decay, where visual grounding weakens and language priors dominate as the generation progresses. In this paper, we propose First Logit Boosting (FLB), a simple yet effective training-free technique designed to alleviate long-term decay in LVLMs. FLB stores the logit of the first generated token and adds it to subsequent token predictions, effectively mitigating long-term decay of visual information. We observe that FLB (1) sustains the visual information embedded in the first token throughout generation, and (2) suppresses hallucinated words through the stabilizing effect of the ``The'' token. Experimental results show that FLB significantly reduces object hallucination across various tasks, benchmarks, and backbone models. Notably, it causes negligible inference overhead, making it highly applicable to real-time multimodal systems. Code is available at this https URL
>
---
#### [new 078] Think, Act, Build: An Agentic Framework with Vision Language Models for Zero-Shot 3D Visual Grounding
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D视觉定位任务，旨在通过自然语言描述定位3D场景中的物体。针对现有方法依赖预处理点云的问题，提出动态框架TAB，结合2D视觉语言模型与多视角几何，实现从2D到3D的生成式重建。**

- **链接: [https://arxiv.org/pdf/2604.00528](https://arxiv.org/pdf/2604.00528)**

> **作者:** Haibo Wang; Zihao Lin; Zhiyang Xu; Lifu Huang
>
> **摘要:** 3D Visual Grounding (3D-VG) aims to localize objects in 3D scenes via natural language descriptions. While recent advancements leveraging Vision-Language Models (VLMs) have explored zero-shot possibilities, they typically suffer from a static workflow relying on preprocessed 3D point clouds, essentially degrading grounding into proposal matching. To bypass this reliance, our core motivation is to decouple the task: leveraging 2D VLMs to resolve complex spatial semantics, while relying on deterministic multi-view geometry to instantiate the 3D structure. Driven by this insight, we propose "Think, Act, Build (TAB)", a dynamic agentic framework that reformulates 3D-VG tasks as a generative 2D-to-3D reconstruction paradigm operating directly on raw RGB-D streams. Specifically, guided by a specialized 3D-VG skill, our VLM agent dynamically invokes visual tools to track and reconstruct the target across 2D frames. Crucially, to overcome the multi-view coverage deficit caused by strict VLM semantic tracking, we introduce the Semantic-Anchored Geometric Expansion, a mechanism that first anchors the target in a reference video clip and then leverages multi-view geometry to propagate its spatial location across unobserved frames. This enables the agent to "Build" the target's 3D representation by aggregating these multi-view features via camera parameters, directly mapping 2D visual cues to 3D coordinates. Furthermore, to ensure rigorous assessment, we identify flaws such as reference ambiguity and category errors in existing benchmarks and manually refine the incorrect queries. Extensive experiments on ScanRefer and Nr3D demonstrate that our framework, relying entirely on open-source models, significantly outperforms previous zero-shot methods and even surpasses fully supervised baselines.
>
---
#### [new 079] Towards Viewpoint-Robust End-to-End Autonomous Driving with 3D Foundation Model Priors
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决相机视角变化下的轨迹规划问题。通过引入3D基础模型的几何先验，提升模型对视角变化的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.00597](https://arxiv.org/pdf/2604.00597)**

> **作者:** Hiroki Hashimoto; Hiromichi Goto; Hiroyuki Sugai; Hiroshi Kera; Kazuhiko Kawamoto
>
> **备注:** Accepted at CVPR Workshop on Simulation for Autonomous Driving 2026
>
> **摘要:** Robust trajectory planning under camera viewpoint changes is important for scalable end-to-end autonomous driving. However, existing models often depend heavily on the camera viewpoints seen during training. We investigate an augmentation-free approach that leverages geometric priors from a 3D foundation model. The method injects per-pixel 3D positions derived from depth estimates as positional embeddings and fuses intermediate geometric features through cross-attention. Experiments on the VR-Drive camera viewpoint perturbation benchmark show reduced performance degradation under most perturbation conditions, with clear improvements under pitch and height perturbations. Gains under longitudinal translation are smaller, suggesting that more viewpoint-agnostic integration is needed for robustness to camera viewpoint changes.
>
---
#### [new 080] Toward Optimal Sampling Rate Selection and Unbiased Classification for Precise Animal Activity Recognition
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于动物行为识别任务，旨在解决分类准确率不均和采样率优化问题。提出IBA-Net模型，结合多采样率特征定制和分类器校准，提升所有行为的识别精度。**

- **链接: [https://arxiv.org/pdf/2604.00517](https://arxiv.org/pdf/2604.00517)**

> **作者:** Axiu Mao; Meilu Zhu; Lei Shen; Xiaoshuai Wang; Tomas Norton; Kai Liu
>
> **备注:** 26 pages, 14 figures
>
> **摘要:** With the rapid advancements in deep learning techniques, wearable sensor-aided animal activity recognition (AAR) has demonstrated promising performance, thereby improving livestock management efficiency as well as animal health and welfare monitoring. However, existing research often prioritizes overall performance, overlooking the fact that classification accuracies for specific animal behavioral categories may remain unsatisfactory. This issue typically stems from suboptimal sampling rates or class imbalance problems. To address these challenges and achieve high classification accuracy across all individual behaviors in farm animals, we propose a novel Individual-Behavior-Aware Network (IBA-Net). This network enhances the recognition of each specific behavior by simultaneously customizing features and calibrating the classifier. Specifically, considering that different behaviors require varying sampling rates to achieve optimal performance, we design a Mixture-of-Experts (MoE)-based Feature Customization (MFC) module. This module adaptively fuses data from multiple sampling rates, capturing customized features tailored to various animal behaviors. Additionally, to mitigate classifier bias toward majority classes caused by class imbalance, we develop a Neural Collapse-driven Classifier Calibration (NC3) module. This module introduces a fixed equiangular tight frame (ETF) classifier during the classification stage, maximizing the angles between pair-wise classifier vectors and thereby improving the classification performance for minority classes. To validate the effectiveness of IBA-Net, we conducted experiments on three public datasets covering goat, cattle, and horse activity recognition. The results demonstrate that our method consistently outperforms existing approaches across all datasets.
>
---
#### [new 081] ONE-SHOT: Compositional Human-Environment Video Synthesis via Spatial-Decoupled Motion Injection and Hybrid Context Integration
- **分类: cs.CV**

- **简介: 该论文属于视频生成任务，旨在解决人-环境视频合成中精细控制与生成灵活性的矛盾。提出ONE-SHOT框架，通过空间解耦和混合上下文整合实现高效可控的视频生成。**

- **链接: [https://arxiv.org/pdf/2604.01043](https://arxiv.org/pdf/2604.01043)**

> **作者:** Fengyuan Yang; Luying Huang; Jiazhi Guan; Quanwei Yang; Dongwei Pan; Jianglin Fu; Haocheng Feng; Wei He; Kaisiyuan Wang; Hang Zhou; Angela Yao
>
> **备注:** 23 pages, 7 figures
>
> **摘要:** Recent advances in Video Foundation Models (VFMs) have revolutionized human-centric video synthesis, yet fine-grained and independent editing of subjects and scenes remains a critical challenge. Recent attempts to incorporate richer environment control through rigid 3D geometric compositions often encounter a stark trade-off between precise control and generative flexibility. Furthermore, the heavy 3D pre-processing still limits practical scalability. In this paper, we propose ONE-SHOT, a parameter-efficient framework for compositional human-environment video generation. Our key insight is to factorize the generative process into disentangled signals. Specifically, we introduce a canonical-space injection mechanism that decouples human dynamics from environmental cues via cross-attention. We also propose Dynamic-Grounded-RoPE, a novel positional embedding strategy that establishes spatial correspondences between disparate spatial domains without any heuristic 3D alignments. To support long-horizon synthesis, we introduce a Hybrid Context Integration mechanism to maintain subject and scene consistency across minute-level generations. Experiments demonstrate that our method significantly outperforms state-of-the-art methods, offering superior structural control and creative diversity for video synthesis. Our project has been available on: this https URL.
>
---
#### [new 082] Mine-JEPA: In-Domain Self-Supervised Learning for Mine-Like Object Classification in Side-Scan Sonar
- **分类: cs.CV**

- **简介: 该论文属于水下目标分类任务，解决侧扫声呐图像中矿类物体识别问题。提出Mine-JEPA方法，在少量数据下实现高精度分类。**

- **链接: [https://arxiv.org/pdf/2604.00383](https://arxiv.org/pdf/2604.00383)**

> **作者:** Taeyoun Kwon; Youngwon Choi; Hyeonyu Kim; Myeongkyun Cho; Junhyeok Choi; Moon Hwan Kim
>
> **备注:** 9 pages, 3 figures, 6 tables. Accepted at CVPR 2026 MACVi Workshop
>
> **摘要:** Side-scan sonar (SSS) mine classification is a challenging maritime vision problem characterized by extreme data scarcity and a large domain gap from natural images. While self-supervised learning (SSL) and general-purpose vision foundation models have shown strong performance in general vision and several specialized domains, their use in SSS remains largely unexplored. We present Mine-JEPA, the first in-domain SSL pipeline for SSS mine classification, using SIGReg, a regularization-based SSL loss, to pretrain on only 1,170 unlabeled sonar images. In the binary mine vs. non-mine setting, Mine-JEPA achieves an F1 score of 0.935, outperforming fine-tuned DINOv3 (0.922), a foundation model pretrained on 1.7B images. For 3-class mine-like object classification, Mine-JEPA reaches 0.820 with synthetic data augmentation, again outperforming fine-tuned DINOv3 (0.810). We further observe that applying in-domain SSL to foundation models degrades performance by 10--13 percentage points, suggesting that stronger pretrained models do not always benefit from additional domain adaptation. In addition, Mine-JEPA with a compact ViT-Tiny backbone achieves competitive performance while using 4x fewer parameters than DINOv3. These results suggest that carefully designed in-domain self-supervised learning is a viable alternative to much larger foundation models in data-scarce maritime sonar imagery.
>
---
#### [new 083] Q-Mask: Query-driven Causal Masks for Text Anchoring in OCR-Oriented Vision-Language Models
- **分类: cs.CV**

- **简介: 该论文属于OCR任务，旨在解决文本锚定问题。提出Q-Mask框架，通过因果查询驱动的掩码解码器提升文本与图像区域的精准对应。**

- **链接: [https://arxiv.org/pdf/2604.00161](https://arxiv.org/pdf/2604.00161)**

> **作者:** Longwei Xu; Feng Feng; Shaojie Zhang; Xin Chen; Hang Li; Anan Du; Hailong Yu; Pei Fu; Zhenbo Luo; Jian Luan
>
> **摘要:** Optical Character Recognition (OCR) is increasingly regarded as a foundational capability for modern vision-language models (VLMs), enabling them not only to read text in images but also to support downstream reasoning in real-world visual question answering (VQA). However, practical applications further require reliable text anchors, i.e., accurately grounding queried text to its corresponding spatial region. To systematically evaluate this capability, we introduce TextAnchor-Bench (TABench), a benchmark for fine-grained text-region grounding, which reveals that both general-purpose and OCR-specific VLMs still struggle to establish accurate and stable text anchors. To address this limitation, we propose Q-Mask, a precise OCR framework built upon a causal query-driven mask decoder (CQMD). Inspired by chain-of-thought reasoning, Q-Mask performs causal visual decoding that sequentially generates query-conditioned visual masks before producing the final OCR output. This visual CoT paradigm disentangles where the text is from what the text is, enforcing grounded evidence acquisition prior to recognition and enabling explicit text anchor construction during inference. To train CQMD, we construct TextAnchor-26M, a large-scale dataset of image-text pairs annotated with fine-grained masks corresponding to specific textual elements, encouraging stable text-region correspondences and injecting strong spatial priors into VLM training. Extensive experiments demonstrate that Q-Mask substantially improves text anchoring and understanding across diverse visual scenes.
>
---
#### [new 084] OmniSch: A Multimodal PCB Schematic Benchmark For Structured Diagram Visual Reasoning
- **分类: cs.CV**

- **简介: 该论文提出OmniSch基准，用于评估大模型在电路图理解与图结构生成的能力。解决电路图到机器可读图的转换问题，涵盖实体定位、拓扑推理、几何权重构建及工具辅助搜索。**

- **链接: [https://arxiv.org/pdf/2604.00270](https://arxiv.org/pdf/2604.00270)**

> **作者:** Taiting Lu; Kaiyuan Lin; Yuxin Tian; Yubo Wang; Muchuan Wang; Sharique Khatri; Akshit Kartik; Yixi Wang; Amey Santosh Rane; Yida Wang; Yifan Yang; Yi-Chao Chen; Yincheng Jin; Mahanth Gowda
>
> **摘要:** Recent large multimodal models (LMMs) have made rapid progress in visual grounding, document understanding, and diagram reasoning tasks. However, their ability to convert Printed Circuit Board (PCB) schematic diagrams into machine-readable spatially weighted netlist graphs, jointly capturing component attributes, connectivity, and geometry, remains largely underexplored, despite such graph representations are the backbone of practical electronic design automation (EDA) workflows. To bridge this gap, we introduce OmniSch, the first comprehensive benchmark designed to assess LMMs on schematic understanding and spatial netlist graph construction. OmniSch contains 1,854 real-world schematic diagrams and includes four tasks: (1) visual grounding for schematic entities, with 109.9K grounded instances aligning 423.4K diagram semantic labels to their visual regions; (2) diagram-to-graph reasoning, understanding topological relationship among diagram elements; (3) geometric reasoning, constructing layout-dependent weights for each connection; and (4) tool-augmented agentic reasoning for visual search, invoking external tools to accomplish (1)-(3). Our results reveal substantial gaps of current LMMs in interpreting schematic engineering artifacts, including unreliable fine-grained grounding, brittle layout-to-graph parsing, inconsistent global connectivity reasoning and inefficient visual exploration.
>
---
#### [new 085] Adversarial Attenuation Patch Attack for SAR Object Detection
- **分类: cs.CV; cs.CR**

- **简介: 该论文属于SAR目标检测任务，旨在解决对抗攻击中扰动明显、难以物理实现的问题。提出AAP方法，在能量约束下实现高效且隐蔽的攻击。**

- **链接: [https://arxiv.org/pdf/2604.00887](https://arxiv.org/pdf/2604.00887)**

> **作者:** Yiming Zhang; Weibo Qin; Feng Wang
>
> **备注:** 5 pages, 4 figures. Source code is available at this https URL
>
> **摘要:** Deep neural networks have demonstrated excellent performance in SAR target detection tasks but remain susceptible to adversarial attacks. Existing SAR-specific attack methods can effectively deceive detectors; however, they often introduce noticeable perturbations and are largely confined to digital domain, neglecting physical implementation constrains for attacking SAR systems. In this paper, a novel Adversarial Attenuation Patch (AAP) method is proposed that employs energy-constrained optimization strategy coupled with an attenuation-based deployment framework to achieve a seamless balance between attack effectiveness and stealthiness. More importantly, AAP exhibits strong potential for physical realization by aligning with signal-level electronic jamming mechanisms. Experimental results show that AAP effectively degrades detection performance while preserving high imperceptibility, and shows favorable transferability across different models. This study provides a physical grounded perspective for adversarial attacks on SAR target detection systems and facilitates the design of more covert and practically deployable attack strategies. The source code is made available at this https URL.
>
---
#### [new 086] ARGS: Auto-Regressive Gaussian Splatting via Parallel Progressive Next-Scale Prediction
- **分类: cs.CV**

- **简介: 该论文提出ARGS框架，用于3D对象生成中的多尺度预测，解决传统方法在效率和细节控制上的不足。通过高斯点云和树结构实现高效生成。**

- **链接: [https://arxiv.org/pdf/2604.00494](https://arxiv.org/pdf/2604.00494)**

> **作者:** Quanyuan Ruan; Kewei Shi; Jiabao Lei; Xifeng Gao; Xiaoguang Han
>
> **摘要:** Auto-regressive frameworks for next-scale prediction of 2D images have demonstrated strong potential for producing diverse and sophisticated content by progressively refining a coarse input. However, extending this paradigm to 3D object generation remains largely unexplored. In this paper, we introduce auto-regressive Gaussian splatting (ARGS), a framework for making next-scale predictions in parallel for generation according to levels of detail. We propose a Gaussian simplification strategy and reverse the simplification to guide next-scale generation. Benefiting from the use of hierarchical trees, the generation process requires only \(\mathcal{O}(\log n)\) steps, where \(n\) is the number of points. Furthermore, we propose a tree-based transformer to predict the tree structure auto-regressively, allowing leaf nodes to attend to their internal ancestors to enhance structural consistency. Extensive experiments demonstrate that our approach effectively generates multi-scale Gaussian representations with controllable levels of detail, visual fidelity, and a manageable time consumption budget.
>
---
#### [new 087] RegFormer: Transferable Relational Grounding for Efficient Weakly-Supervised Human-Object Interaction Detection
- **分类: cs.CV**

- **简介: 该论文属于弱监督人体-物体交互检测任务，旨在解决依赖图像级标注的交互识别问题。提出RegFormer模型，通过空间关系引导实现高效准确的实例级交互推理。**

- **链接: [https://arxiv.org/pdf/2604.00507](https://arxiv.org/pdf/2604.00507)**

> **作者:** Jihwan Park; Chanhyeong Yang; Jinyoung Park; Taehoon Song; Hyunwoo J. Kim
>
> **备注:** Accepted at CVPR2026
>
> **摘要:** Weakly-supervised Human-Object Interaction (HOI) detection is essential for scalable scene understanding, as it learns interactions from only image-level annotations. Due to the lack of localization signals, prior works typically rely on an external object detector to generate candidate pairs and then infer their interactions through pairwise reasoning. However, this framework often struggles to scale due to the substantial computational cost incurred by enumerating numerous instance pairs. In addition, it suffers from false positives arising from non-interactive combinations, which hinder accurate instance-level HOI reasoning. To address these issues, we introduce Relational Grounding Transformer (RegFormer), a versatile interaction recognition module for efficient and accurate HOI reasoning. Under image-level supervision, RegFormer leverages spatially grounded signals as guidance for the reasoning process and promotes locality-aware interaction learning. By learning localized interaction cues, our module distinguishes humans, objects, and their interactions, enabling direct transfer from image-level interaction reasoning to precise and efficient instance-level reasoning without additional training. Our extensive experiments and analyses demonstrate that RegFormer effectively learns spatial cues for instance-level interaction reasoning, operates with high efficiency, and even achieves performance comparable to fully supervised models. Our code is available at this https URL.
>
---
#### [new 088] Disentangling to Re-couple: Resolving the Similarity-Controllability Paradox in Subject-Driven Text-to-Image Generation
- **分类: cs.CV**

- **简介: 该论文属于文本到图像生成任务，旨在解决“相似性与可控性悖论”。通过分离文本与视觉信息并重新耦合，提升主体保真度与文本控制精度。**

- **链接: [https://arxiv.org/pdf/2604.00849](https://arxiv.org/pdf/2604.00849)**

> **作者:** Shuang Li; Chao Deng; Hang Chen; Liqun Liu; Zhenyu Hu; Te Cao; Mengge Xue; Yuan Chen; Peng Shu; Huan Yu; Jie Jiang
>
> **备注:** Accepted by CVPR 2026 (Main)
>
> **摘要:** Subject-Driven Text-to-Image (T2I) Generation aims to preserve a subject's identity while editing its context based on a text prompt. A core challenge in this task is the "similarity-controllability paradox", where enhancing textual control often degrades the subject's fidelity, and vice-versa. We argue this paradox stems from the ambiguous role of text prompts, which are often tasked with describing both the subject and the desired modifications, leading to conflicting signals for the model. To resolve this, we propose DisCo, a novel framework that first Disntangles and then re-Couples visual and textual information. First, our textual-visual decoupling module isolates the sources of information: subject identity is extracted exclusively from the reference image with the entity word of the subject, while the text prompt is simplified to contain only the modification command, where the subject refers to general pronouns, eliminating descriptive ambiguity. However, this strict separation can lead to unnatural compositions between the subject and its contexts. We address this by designing a dedicated reward signal and using reinforcement learning to seamlessly recouple the visually-defined subject and the textually-generated context. Our approach effectively resolves the paradox, enabling simultaneous high-fidelity subject preservation and precise textual control. Extensive experiments demonstrate that our method achieves state-of-the-art performance, producing highly realistic and coherent images.
>
---
#### [new 089] Sub-metre Lunar DEM Generation and Validation from Chandrayaan-2 OHRC Multi-View Imagery Using Open-Source Photogrammetry
- **分类: cs.CV**

- **简介: 该论文属于月球地形建模任务，旨在生成高精度亚米级数字高程模型（DEM）。通过开源摄影测量方法处理 Chandrayaan-2 OHRC 多视角影像，解决月面高程精确建模问题。**

- **链接: [https://arxiv.org/pdf/2604.01032](https://arxiv.org/pdf/2604.01032)**

> **作者:** Aaranay Aadi; Jai Singla; Nitant Dube; Oleg Alexandrov
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** High-resolution digital elevation models (DEMs) of the lunar surface are essential for surface mobility planning, landing site characterization, and planetary science. The Orbiter High Resolution Camera (OHRC) on board Chandrayaan-2 has the best ground sampling capabilities of any lunar orbital imaging currently in use by acquiring panchromatic imagery at a resolution of roughly 20-30 cm per pixel. This work presents, for the first time, the generation of sub-metre DEMs from OHRC multi-view imagery using an exclusively open-source pipeline. Candidate stereo pairs are identified from non-paired OHRC archives through geometric analysis of image metadata, employing baseline-to-height (B/H) ratio computation and convergence angle estimation. Dense stereo correspondence and ray triangulation are then applied to generate point clouds, which are gridded into DEMs at effective spatial resolutions between approximately 24 and 54 cm across five geographically distributed lunar sites. Absolute elevation consistency is established through Iterative Closest Point (ICP) alignment against Lunar Reconnaissance Orbiter Narrow Angle Camera (NAC) Digital Terrain Models, followed by constant-bias offset correction. Validation against NAC reference terrain yields a vertical RMSE of 5.85 m (at native OHRC resolution), and a horizontal accuracy of less than 30 cm assessed by planimetric feature matching.
>
---
#### [new 090] MATHENA: Mamba-based Architectural Tooth Hierarchical Estimator and Holistic Evaluation Network for Anatomy
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出MATHENA，用于牙科诊断中的牙齿检测、龋齿分割、异常检测和发育阶段分类，解决多任务协同问题。**

- **链接: [https://arxiv.org/pdf/2604.00537](https://arxiv.org/pdf/2604.00537)**

> **作者:** Kyeonghun Kim; Jaehyung Park; Youngung Han; Anna Jung; Seongbin Park; Sumin Lee; Jiwon Yang; Jiyoon Han; Subeen Lee; Junsu Lim; Hyunsu Go; Eunseob Choi; Hyeonseok Jung; Soo Yong Kim; Woo Kyoung Jeong; Won Jae Lee; Pa Hong; Hyuk-Jae Lee; Ken Ying-Kai Liao; Nam-Joon Kim
>
> **备注:** 10 pages, 3 figures, 4 tables
>
> **摘要:** Dental diagnosis from Orthopantomograms (OPGs) requires coordination of tooth detection, caries segmentation (CarSeg), anomaly detection (AD), and dental developmental staging (DDS). We propose Mamba-based Architectural Tooth Hierarchical Estimator and Holistic Evaluation Network for Anatomy (MATHENA), a unified framework leveraging Mamba's linear-complexity State Space Models (SSM) to address all four tasks. MATHENA integrates MATHE, a multi-resolution SSM-driven detector with four-directional Vision State Space (VSS) blocks for O(N) global context modeling, generating per-tooth crops. These crops are processed by HENA, a lightweight Mamba-UNet with a triple-head architecture and Global Context State Token (GCST). In the triple-head architecture, CarSeg is first trained as an upstream task to establish shared representations, which are then frozen and reused for downstream AD fine-tuning and DDS classification via linear probing, enabling stable, efficient learning. We also curate PARTHENON, a benchmark comprising 15,062 annotated instances from ten datasets. MATHENA achieves 93.78% mAP@50 in tooth detection, 90.11% Dice for CarSeg, 88.35% for AD, and 72.40% ACC for DDS.
>
---
#### [new 091] Reliev3R: Relieving Feed-forward Reconstruction from Multi-View Geometric Annotations
- **分类: cs.CV**

- **简介: 该论文提出Reliev3R，解决FFRMs依赖多视角几何标注的问题，通过弱监督训练实现低成本3D重建。**

- **链接: [https://arxiv.org/pdf/2604.00548](https://arxiv.org/pdf/2604.00548)**

> **作者:** Youyu Chen; Junjun Jiang; Yueru Luo; Kui Jiang; Xianming Liu; Xu Yan; Dave Zhenyu Chen
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** With recent advances, Feed-forward Reconstruction Models (FFRMs) have demonstrated great potential in reconstruction quality and adaptiveness to multiple downstream tasks. However, the excessive reliance on multi-view geometric annotations, e.g. 3D point maps and camera poses, makes the fully-supervised training scheme of FFRMs difficult to scale up. In this paper, we propose Reliev3R, a weakly-supervised paradigm for training FFRMs from scratch without cost-prohibitive multi-view geometric annotations. Relieving the reliance on geometric sensory data and compute-exhaustive structure-from-motion preprocessing, our method draws 3D knowledge directly from monocular relative depths and image sparse correspondences given by zero-shot predictions of pretrained models. At the core of Reliev3R, we design an ambiguity-aware relative depth loss and a trigonometry-based reprojection loss to facilitate supervision for multi-view geometric consistency. Training from scratch with the less data, Reliev3R catches up with its fully-supervised sibling models, taking a step towards low-cost 3D reconstruction supervisions and scalable FFRMs.
>
---
#### [new 092] TRACE: High-Fidelity 3D Scene Editing via Tangible Reconstruction and Geometry-Aligned Contextual Video Masking
- **分类: cs.CV**

- **简介: 该论文提出TRACE，用于高保真3D场景编辑。解决传统方法在结构完整性和细节操作上的不足，通过三阶段流程实现精准的3D物体添加与修改。**

- **链接: [https://arxiv.org/pdf/2604.01207](https://arxiv.org/pdf/2604.01207)**

> **作者:** Jiyuan Hu; Zechuan Zhang; Zongxin Yang; Yi Yang
>
> **备注:** 22 pages, 9 figures
>
> **摘要:** We present TRACE, a mesh-guided 3DGS editing framework that achieves automated, high-fidelity scene transformation. By anchoring video diffusion with explicit 3D geometry, TRACE uniquely enables fine-grained, part-level manipulatio--such as local pose shifting or component replacemen--while preserving the structural integrity of the central subject, a capability largely absent in existing editing methods. Our approach comprises three key stages: (1) Multi-view 3D-Anchor Synthesis, which leverages a sparse-view editor trained on our MV-TRACE datase--the first multi-view consistent dataset dedicated to scene-coherent object addition and modificatio--to generate spatially consistent 3D-anchors; (2) Tangible Geometry Anchoring (TGA), which ensures precise spatial synchronization between inserted meshes and the 3DGS scene via two-phase registration; and (3) Contextual Video Masking (CVM), which integrates 3D projections into an autoregressive video pipeline to achieve temporally stable, physically-grounded rendering. Extensive experiments demonstrate that TRACE consistently outperforms existing methods especially in editing versatility and structural integrity.
>
---
#### [new 093] MAESIL: Masked Autoencoder for Enhanced Self-supervised Medical Image Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D医学图像预训练任务，旨在解决标注数据不足和忽略3D结构的问题。提出MAESIL框架，通过3D超块和双掩码策略提升重建效果。**

- **链接: [https://arxiv.org/pdf/2604.00514](https://arxiv.org/pdf/2604.00514)**

> **作者:** Kyeonghun Kim; Hyeonseok Jung; Youngung Han; Junsu Lim; YeonJu Jean; Seongbin Park; Eunseob Choi; Hyunsu Go; SeoYoung Ju; Seohyoung Park; Gyeongmin Kim; MinJu Kwon; KyungSeok Yuh; Soo Yong Kim; Ken Ying-Kai Liao; Nam-Joon Kim; Hyuk-Jae Lee
>
> **备注:** 5 pages, 3 figures. Accepted at ICEIC 2026
>
> **摘要:** Training deep learning models for three-dimensional (3D) medical imaging, such as Computed Tomography (CT), is fundamentally challenged by the scarcity of labeled data. While pre-training on natural images is common, it results in a significant domain shift, limiting performance. Self-Supervised Learning (SSL) on unlabeled medical data has emerged as a powerful solution, but prominent frameworks often fail to exploit the inherent 3D nature of CT scans. These methods typically process 3D scans as a collection of independent 2D slices, an approach that fundamentally discards critical axial coherence and the 3D structural context. To address this limitation, we propose the autoencoder for enhanced self-supervised medical image learning(MAESIL), a novel self-supervised learning framework designed to capture 3D structural information efficiently. The core innovation is the 'superpatch', a 3D chunk-based input unit that balances 3D context preservation with computational efficiency. Our framework partitions the volume into superpatches and employs a 3D masked autoencoder strategy with a dual-masking strategy to learn comprehensive spatial representations. We validated our approach on three diverse large-scale public CT datasets. Our experimental results show that MAESIL demonstrates significant improvements over existing methods such as AE, VAE and VQ-VAE in key reconstruction metrics such as PSNR and SSIM. This establishes MAESIL as a robust and practical pre-training solution for 3D medical imaging tasks.
>
---
#### [new 094] Query-Conditioned Evidential Keyframe Sampling for MLLM-Based Long-Form Video Understanding
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于长视频理解任务，解决MLLM在处理长视频时因上下文限制和计算成本高的问题。提出一种基于信息瓶颈理论的关键帧采样方法，通过最大化查询与帧的互信息提升采样效率和效果。**

- **链接: [https://arxiv.org/pdf/2604.01002](https://arxiv.org/pdf/2604.01002)**

> **作者:** Yiheng Wang; Lichen Zhu; Yueqian Lin; Yudong Liu; Jingyang Zhang; Hai "Helen" Li; Yiran Chen
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown strong performance on video question answering, but their application to long-form videos is constrained by limited context length and computational cost, making keyframe sampling essential. Existing approaches typically rely on semantic relevance or reinforcement learning, which either fail to capture evidential clues or suffer from inefficient combinatorial optimization. In this work, we propose an evidence-driven keyframe sampling framework grounded in information bottleneck theory. We formulate keyframe selection as maximizing the conditional mutual information between selected frames and the query, providing a principled objective that reflects each frame's contribution to answering the question. To make this objective tractable, we exploit its structure to derive a decomposed optimization that reduces subset selection to independent frame-level scoring. We further introduce a query-conditioned evidence scoring network trained with a contrastive objective to estimate evidential importance efficiently. Experiments on long-form video understanding benchmarks show that our method consistently outperforms prior sampling strategies under strict token budgets, while significantly improving training efficiency.
>
---
#### [new 095] Semantic Audio-Visual Navigation in Continuous Environments
- **分类: cs.CV; cs.SD**

- **简介: 该论文属于音频-视觉导航任务，解决连续环境中目标失声导致的导航问题。提出MAGNet模型，结合多模态信息与历史上下文，提升导航成功率。**

- **链接: [https://arxiv.org/pdf/2603.19660](https://arxiv.org/pdf/2603.19660)**

> **作者:** Yichen Zeng; Hebaixu Wang; Meng Liu; Yu Zhou; Chen Gao; Kehan Chen; Gongping Huang
>
> **备注:** This paper has been accepted to CVPR 2026
>
> **摘要:** Audio-visual navigation enables embodied agents to navigate toward sound-emitting targets by leveraging both auditory and visual cues. However, most existing approaches rely on precomputed room impulse responses (RIRs) for binaural audio rendering, restricting agents to discrete grid positions and leading to spatially discontinuous observations. To establish a more realistic setting, we introduce Semantic Audio-Visual Navigation in Continuous Environments (SAVN-CE), where agents can move freely in 3D spaces and perceive temporally and spatially coherent audio-visual streams. In this setting, targets may intermittently become silent or stop emitting sound entirely, causing agents to lose goal information. To tackle this challenge, we propose MAGNet, a multimodal transformer-based model that jointly encodes spatial and semantic goal representations and integrates historical context with self-motion cues to enable memory-augmented goal reasoning. Comprehensive experiments demonstrate that MAGNet significantly outperforms state-of-the-art methods, achieving up to a 12.1\% absolute improvement in success rate. These results also highlight its robustness to short-duration sounds and long-distance navigation scenarios. The code is available at this https URL.
>
---
#### [new 096] Benchmarking and Mechanistic Analysis of Vision-Language Models for Cross-Depiction Assembly Instruction Alignment
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉-语言对齐任务，旨在解决装配图与视频间的跨表述匹配问题。通过构建基准数据集并分析19个模型，发现视觉编码是提升对齐性能的关键。**

- **链接: [https://arxiv.org/pdf/2604.00913](https://arxiv.org/pdf/2604.00913)**

> **作者:** Zhuchenyang Liu; Yao Zhang; Yu Xiao
>
> **摘要:** 2D assembly diagrams are often abstract and hard to follow, creating a need for intelligent assistants that can monitor progress, detect errors, and provide step-by-step guidance. In mixed reality settings, such systems must recognize completed and ongoing steps from the camera feed and align them with the diagram instructions. Vision Language Models (VLMs) show promise for this task, but face a depiction gap because assembly diagrams and video frames share few visual features. To systematically assess this gap, we construct IKEA-Bench, a benchmark of 1,623 questions across 6 task types on 29 IKEA furniture products, and evaluate 19 VLMs (2B-38B) under three alignment strategies. Our key findings: (1) assembly instruction understanding is recoverable via text, but text simultaneously degrades diagram-to-video alignment; (2) architecture family predicts alignment accuracy more strongly than parameter count; (3) video understanding remains a hard bottleneck unaffected by strategy. A three-level mechanistic analysis further reveals that diagrams and video occupy disjoint ViT subspaces, and that adding text shifts models from visual to text-driven reasoning. These results identify visual encoding as the primary target for improving cross-depiction robustness. Project page: this https URL
>
---
#### [new 097] A Benchmark of State-Space Models vs. Transformers and BiLSTM-based Models for Historical Newspaper OCR
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于历史报纸OCR任务，旨在解决长文本、低质量印刷和复杂布局带来的识别难题。对比了SSM、Transformer和BiLSTM模型，验证了SSM在效率和准确性的优势。**

- **链接: [https://arxiv.org/pdf/2604.00725](https://arxiv.org/pdf/2604.00725)**

> **作者:** Merveilles Agbeti-messan; Thierry Paquet; Clément Chatelain; Pierrick Tranouez; Stéphane Nicolas
>
> **摘要:** End-to-end OCR for historical newspapers remains challenging, as models must handle long text sequences, degraded print quality, and complex layouts. While Transformer-based recognizers dominate current research, their quadratic complexity limits efficient paragraph-level transcription and large-scale deployment. We investigate linear-time State-Space Models (SSMs), specifically Mamba, as a scalable alternative to Transformer-based sequence modeling for OCR. We present to our knowledge, the first OCR architecture based on SSMs, combining a CNN visual encoder with bi-directional and autoregressive Mamba sequence modeling, and conduct a large-scale benchmark comparing SSMs with Transformer- and BiLSTM-based recognizers. Multiple decoding strategies (CTC, autoregressive, and non-autoregressive) are evaluated under identical training conditions alongside strong neural baselines (VAN, DAN, DANIEL) and widely used off-the-shelf OCR engines (PERO-OCR, Tesseract OCR, TrOCR, Gemini). Experiments on historical newspapers from the Bibliothèque nationale du Luxembourg, with newly released >99% verified gold-standard annotations, and cross-dataset tests on Fraktur and Antiqua lines, show that all neural models achieve low error rates (~2% CER), making computational efficiency the main differentiator. Mamba-based models maintain competitive accuracy while halving inference time and exhibiting superior memory scaling (1.26x vs 2.30x growth at 1000 chars), reaching 6.07% CER at the severely degraded paragraph level compared to 5.24% for DAN, while remaining 2.05x faster. We release code, trained models, and standardized evaluation protocols to enable reproducible research and guide practitioners in large-scale cultural heritage OCR.
>
---
#### [new 098] Foundation Model-guided Iteratively Prompting and Pseudo-Labeling for Partially Labeled Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，解决部分标注数据导致的性能下降问题。提出IPnP框架，通过迭代提示和伪标签生成提升分割效果。**

- **链接: [https://arxiv.org/pdf/2604.01038](https://arxiv.org/pdf/2604.01038)**

> **作者:** Qiaochu Zhao; Wei Wei; David Horowitz; Richard Bakst; Yading Yuan
>
> **备注:** 5 pages, 5 figures. Accepted for presentation at IEEE International Symposium on Biomedical Imaging (ISBI) 2026
>
> **摘要:** Automated medical image segmentation has achieved remarkable progress with fully labeled data. However, site-specific clinical priorities and the high cost of manual annotation often yield scans with only a subset of organs labeled, leading to the partially labeled problem that degrades performance. To address this issue, we propose IPnP, an Iteratively Prompting and Pseudo-labeling framework, for partially labeled medical image segmentation. IPnP iteratively generates and refines pseudo-labels for unlabeled organs through collaboration between a trainable segmentation network (specialist) and a frozen foundation model (generalist), progressively recovering full-organ supervision. On the public dataset AMOS with the simulated partial-label setting, IPnP consistently improves segmentation performance over prior methods and approaches the performance of the fully labeled reference. We further evaluate on a private, partially labeled dataset of 210 head-and-neck cancer patients and demonstrate our effectiveness in real-world clinical settings.
>
---
#### [new 099] IDDM: Identity-Decoupled Personalized Diffusion Models with a Tunable Privacy-Utility Trade-off
- **分类: cs.CV**

- **简介: 该论文属于隐私保护任务，旨在解决个性化生成模型泄露用户身份的问题。提出IDDM模型，在保证生成质量的同时，降低身份可关联性，实现隐私与效用的平衡。**

- **链接: [https://arxiv.org/pdf/2604.00903](https://arxiv.org/pdf/2604.00903)**

> **作者:** Linyan Dai; Xinwei Zhang; Haoyang Li; Qingqing Ye; Haibo Hu
>
> **摘要:** Personalized text-to-image diffusion models (e.g., DreamBooth, LoRA) enable users to synthesize high-fidelity avatars from a few reference photos for social expression. However, once these generations are shared on social media platforms (e.g., Instagram, Facebook), they can be linked to the real user via face recognition systems, enabling identity tracking and profiling. Existing defenses mainly follow an anti-personalization strategy that protects publicly released reference photos by disrupting model fine-tuning. While effective against unauthorized personalization, they do not address another practical setting in which personalization is authorized, but the resulting public outputs still leak identity information. To address this problem, we introduce a new defense setting, termed model-side output immunization, whose goal is to produce a personalized model that supports authorized personalization while reducing the identity linkability of public generations, with tunable control over the privacy-utility trade-off to accommodate diverse privacy needs. To this end, we propose Identity-Decoupled personalized Diffusion Models (IDDM), a model-side defense that integrates identity decoupling into the personalization pipeline. Concretely, IDDM follows an alternating procedure that interleaves short personalization updates with identity-decoupled data optimization, using a two-stage schedule to balance identity linkability suppression and generation utility. Extensive experiments across multiple datasets, diverse prompts, and state-of-the-art face recognition systems show that IDDM consistently reduces identity linkability while preserving high-quality personalized generation.
>
---
#### [new 100] DLWM: Dual Latent World Models enable Holistic Gaussian-centric Pre-training in Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文提出DLWM，用于自动驾驶中的3D语义高斯表示预训练。解决高斯中心表示在占用感知、预测和规划任务中的性能提升问题，通过双潜在世界模型实现多阶段学习。**

- **链接: [https://arxiv.org/pdf/2604.00969](https://arxiv.org/pdf/2604.00969)**

> **作者:** Yiyao Zhu; Ying Xue; Haiming Zhang; Guangfeng Jiang; Wending Zhou; Xu Yan; Jiantao Gao; Yingjie Cai; Bingbing Liu; Zhen Li; Shaojie Shen
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Vision-based autonomous driving has gained much attention due to its low costs and excellent performance. Compared with dense BEV (Bird's Eye View) or sparse query models, Gaussian-centric method is a comprehensive yet sparse representation by describing scene with 3D semantic Gaussians. In this paper, we introduce DLWM, a novel paradigm with Dual Latent World Models specifically designed to enable holistic gaussian-centric pre-training in autonomous driving using two stages. In the first stage, DLWM predicts 3D Gaussians from queries by self-supervised reconstructing multi-view semantic and depth images. Equipped with fine-grained contextual features, in the second stage, two latent world models are trained separately for temporal feature learning, including Gaussian-flow-guided latent prediction for downstream occupancy perception and forecasting tasks, and ego-planning-guided latent prediction for motion planning. Extensive experiments in SurroundOcc and nuScenes benchmarks demonstrate that DLWM shows significant performance gains across Gaussian-centric 3D occupancy perception, 4D occupancy forecasting and motion planning tasks.
>
---
#### [new 101] mmAnomaly: Leveraging Visual Context for Robust Anomaly Detection in the Non-Visual World with mmWave Radar
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于异常检测任务，解决毫米波雷达在非视觉场景中因环境干扰导致的误报问题。通过结合视觉信息与雷达数据，提升异常检测的准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.00382](https://arxiv.org/pdf/2604.00382)**

> **作者:** Tarik Reza Toha; Shao-Jung; Mahathir Monjur; Shahriar Nirjon
>
> **备注:** Accepted at the 24th ACM/IEEE International Conference on Embedded Artificial Intelligence and Sensing Systems (SenSys 2026)
>
> **摘要:** mmWave radar enables human sensing in non-visual scenarios-e.g., through clothing or certain types of walls-where traditional cameras fail due to occlusion or privacy limitations. However, robust anomaly detection with mmWave remains challenging, as signal reflections are influenced by material properties, clutter, and multipath interference, producing complex, non-Gaussian distortions. Existing methods lack contextual awareness and misclassify benign signal variations as anomalies. We present mmAnomaly, a multi-modal anomaly detection framework that combines mmWave radar with RGBD input to incorporate visual context. Our system extracts semantic cues-such as scene geometry and material properties-using a fast ResNet-based classifier, and uses a conditional latent diffusion model to synthesize the expected mmWave spectrum for the given visual context. A dual-input comparison module then identifies spatial deviations between real and generated spectra to localize anomalies. We evaluate mmAnomaly on two multi-modal datasets across three applications: concealed weapon localization, through-wall intruder localization, and through-wall fall localization. The system achieves up to 94% F1 score and sub-meter localization error, demonstrating robust generalization across clothing, occlusions, and cluttered environments. These results establish mmAnomaly as an accurate and interpretable framework for context-aware anomaly detection in mmWave sensing.
>
---
#### [new 102] PHASOR: Anatomy- and Phase-Consistent Volumetric Diffusion for CT Virtual Contrast Enhancement
- **分类: cs.CV**

- **简介: 该论文提出PHASOR方法，用于CT虚拟对比增强任务，解决现有方法在解剖一致性和空间对齐上的不足，提升合成图像质量与增强准确性。**

- **链接: [https://arxiv.org/pdf/2604.01053](https://arxiv.org/pdf/2604.01053)**

> **作者:** Zilong Li; Dongyang Li; Chenglong Ma; Zhan Feng; Dakai Jin; Junping Zhang; Hao Luo; Fan Wang; Hongming Shan
>
> **摘要:** Contrast-enhanced computed tomography (CECT) is pivotal for highlighting tissue perfusion and vascularity, yet its clinical ubiquity is impeded by the invasive nature of contrast agents and radiation risks. While virtual contrast enhancement (VCE) offers an alternative to synthesizing CECT from non-contrast CT (NCCT), existing methods struggle with anatomical heterogeneity and spatial misalignment, leading to inconsistent enhancement patterns and incorrect details. This paper introduces PHASOR, a volumetric diffusion framework for high-fidelity CT VCE. By treating CT volumes as coherent sequences, we leverage a video diffusion model to enhance structural coherence and volumetric accuracy. To ensure anatomy-phase consistent synthesis, we introduce two complementary modules. First, anatomy-routed mixture-of-experts (AR-MoE) anchors distinct enhancement patterns to anatomical semantics, with organ-specific memory to capture salient details. Second, intensity-phase aware representation alignment (IP-REPA) highlights intricate contrast signals while mitigating the impact of imperfect spatial alignment. Extensive experiments across three datasets demonstrate that PHASOR significantly outperforms state-of-the-art methods in both synthesis quality and enhancement accuracy.
>
---
#### [new 103] The Geometry of Compromise: Unlocking Generative Capabilities via Controllable Modality Alignment
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型任务，旨在解决跨模态对齐问题。通过分解模态差距并提出优化框架，显著提升跨模态任务性能。**

- **链接: [https://arxiv.org/pdf/2604.00279](https://arxiv.org/pdf/2604.00279)**

> **作者:** Hongyuan Liu; Qinli Yang; Wen Li; Zhong Zhang; Jiaming Liu; Wei Han; Zhili Qin; Jinxia Guo; Junming Shao
>
> **摘要:** Vision-Language Models (VLMs) such as CLIP learn a shared embedding space for images and text, yet their representations remain geometrically separated, a phenomenon known as the modality gap. This gap limits tasks requiring cross-modal interchangeability, such as captioning and joint clustering. Existing post-processing approaches can partially improve cross-modal compatibility; however, we show through geometric analysis that they primarily reduce the global centroid offset while leaving the underlying distributional mismatch intact. We decompose the modality gap into a Centroid Gap and a Distribution Gap, and demonstrate that the Distribution Gap is the true predictor of cross-modal task quality ($R^2 = 0.986$), whereas the commonly used Raw Gap is misleading ($R^2 = 0.691$). Motivated by this observation, we propose TPC-CMA (Three-Phase Curriculum for Cross-Modal Alignment), a fine-tuning framework that explicitly reduces both components. The proposed CMA jointly mitigates centroid offsets and reshapes the distributional structure, while a three-phase curriculum with gradient-aware scheduling progressively introduces alignment during training to enable stable optimization. Experiments demonstrate that our method significantly improves cross-modal alignment. With $\alpha_{\text{target}}{=}0.05$, the modality gap is reduced by 66.6\% with only 4.84\% accuracy drop. Under stronger alignment ($\alpha_{\text{target}}{=}0.5$), the gap is reduced by 82.3\%, clustering ARI improves from 0.318 to 0.516, and captioning CIDEr increases by 57.1\% over the original model. Our code and pre-trained models will be made publicly available upon acceptance.
>
---
#### [new 104] Multimodal Language Models Cannot Spot Spatial Inconsistencies
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于多模态语言模型任务，旨在解决模型在跨视角3D几何推理中的空间不一致识别问题。研究提出生成不一致图像对的方法，验证模型表现，发现其显著低于人类。**

- **链接: [https://arxiv.org/pdf/2604.00799](https://arxiv.org/pdf/2604.00799)**

> **作者:** Om Khangaonkar; Hadi J. Rad; Hamed Pirsiavash
>
> **摘要:** Spatial consistency is a fundamental property of the visual world and a key requirement for models that aim to understand physical reality. Despite recent advances, multimodal large language models (MLLMs) often struggle to reason about 3D geometry across multiple views. Rather than asking models to describe scene attributes, we introduce a more challenging task: given two views of the same scene, identify the object that violates 3D motion consistency. We propose a simple and scalable method for generating realistic, spatially inconsistent image pairs from multi-view scenes, enabling systematic evaluation of this capability. Our results show that state-of-the-art MLLMs significantly underperform human observers and exhibit substantial variability across different scene attributes, revealing a fragile and incomplete understanding of 3D structure. We hope our findings underscore the need for approaches that develop a more deeply grounded understanding of the physical world.
>
---
#### [new 105] Advancing Complex Video Object Segmentation via Tracking-Enhanced Prompt: The 1st Winner for 5th PVUW MOSE Challenge
- **分类: cs.CV**

- **简介: 该论文属于复杂视频目标分割任务，旨在解决SAM3在小目标和语义主导目标上的性能不足问题。通过引入跟踪增强提示，提升分割效果。**

- **链接: [https://arxiv.org/pdf/2604.00395](https://arxiv.org/pdf/2604.00395)**

> **作者:** Jinrong Zhang; Canyang Wu; Xusheng He; Weili Guan; Jianlong Wu; Liqiang Nie
>
> **备注:** 1st Place Solution for the 5th PVUW MOSE Challenge (CVPR 2026 Workshop)
>
> **摘要:** In the Complex Video Object Segmentation task, researchers are required to track and segment specific targets within cluttered environments, which rigorously tests a method's capability for target comprehension and environmental adaptability. Although SAM3, the current state-of-the-art solution, exhibits unparalleled segmentation performance and robustness on conventional targets, it underperforms on tiny and semantic-dominated objects. The root cause of this limitation lies in SAM3's insufficient comprehension of these specific target types. To address this issue, we propose TEP: Advancing Complex Video Object Segmentation via Tracking-Enhanced Prompts. As a training-free approach, TEP leverages external tracking models and Multimodal Large Language Models to introduce tracking-enhanced prompts, thereby alleviating the difficulty SAM3 faces in understanding these challenging targets. Our method achieved first place (56.91%) on the test set of the PVUW Challenge 2026: Complex Video Object Segmentation Track.
>
---
#### [new 106] Forecasting Motion in the Wild
- **分类: cs.CV**

- **简介: 该论文属于行为预测任务，旨在解决视觉系统难以通用表示运动的问题。提出密集点轨迹作为行为表征，构建扩散Transformer模型，实现复杂运动的准确预测。**

- **链接: [https://arxiv.org/pdf/2604.01015](https://arxiv.org/pdf/2604.01015)**

> **作者:** Neerja Thakkar; Shiry Ginosar; Jacob Walker; Jitendra Malik; Joao Carreira; Carl Doersch
>
> **备注:** project page: this https URL
>
> **摘要:** Visual intelligence requires anticipating the future behavior of agents, yet vision systems lack a general representation for motion and behavior. We propose dense point trajectories as visual tokens for behavior, a structured mid-level representation that disentangles motion from appearance and generalizes across diverse non-rigid agents, such as animals in-the-wild. Building on this abstraction, we design a diffusion transformer that models unordered sets of trajectories and explicitly reasons about occlusion, enabling coherent forecasts of complex motion patterns. To evaluate at scale, we curate 300 hours of unconstrained animal video with robust shot detection and camera-motion compensation. Experiments show that forecasting trajectory tokens achieves category-agnostic, data-efficient prediction, outperforms state-of-the-art baselines, and generalizes to rare species and morphologies, providing a foundation for predictive visual intelligence in the wild.
>
---
#### [new 107] PET-DINO: Unifying Visual Cues into Grounding DINO with Prompt-Enriched Training
- **分类: cs.CV**

- **简介: 该论文属于开放集目标检测任务，解决文本与视觉对齐难及数据稀缺问题。提出PET-DINO模型，结合视觉和文本提示，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2604.00503](https://arxiv.org/pdf/2604.00503)**

> **作者:** Weifu Fu; Jinyang Li; Bin-Bin Gao; Jialin Li; Yuhuan Lin; Hanqiu Deng; Wenbing Tao; Yong Liu; Chengjie Wang
>
> **摘要:** Open-Set Object Detection (OSOD) enables recognition of novel categories beyond fixed classes but faces challenges in aligning text representations with complex visual concepts and the scarcity of image-text pairs for rare categories. This results in suboptimal performance in specialized domains or with complex objects. Recent visual-prompted methods partially address these issues but often involve complex multi-modal designs and multi-stage optimizations, prolonging the development cycle. Additionally, effective training strategies for data-driven OSOD models remain largely unexplored. To address these challenges, we propose PET-DINO, a universal detector supporting both text and visual prompts. Our Alignment-Friendly Visual Prompt Generation (AFVPG) module builds upon an advanced text-prompted detector, addressing the limitations of text representation guidance and reducing the development cycle. We introduce two prompt-enriched training strategies: Intra-Batch Parallel Prompting (IBP) at the iteration level and Dynamic Memory-Driven Prompting (DMD) at the overall training level. These strategies enable simultaneous modeling of multiple prompt routes, facilitating parallel alignment with diverse real-world usage scenarios. Comprehensive experiments demonstrate that PET-DINO exhibits competitive zero-shot object detection capabilities across various prompt-based detection protocols. These strengths can be attributed to inheritance-based philosophy and prompt-enriched training strategies, which play a critical role in building an effective generic object detector. Project page: this https URL.
>
---
#### [new 108] PRISM: Differentiable Analysis-by-Synthesis for Fixel Recovery in Diffusion MRI
- **分类: cs.CV**

- **简介: 该论文提出PRISM框架，用于解决扩散MRI中的纤维峰恢复问题，通过端到端优化多组分模型提升窄交叉纤维的识别精度。**

- **链接: [https://arxiv.org/pdf/2604.00250](https://arxiv.org/pdf/2604.00250)**

> **作者:** Mohamed Abouagour; Atharva Shah; Eleftherios Garyfallidis
>
> **备注:** 10 pages, 1 figure, 2 tables
>
> **摘要:** Diffusion MRI microstructure fitting is nonconvex and often performed voxelwise, which limits fiber peak recovery in narrow crossings. This work introduces PRISM, a differentiable analysis-by-synthesis framework that fits an explicit multi-compartment forward model end-to-end over spatial patches. The model combines cerebrospinal fluid (CSF), gray matter, up to K white-matter fiber compartments (stick-and-zeppelin), and a restricted compartment, with explicit fiber directions and soft model selection via repulsion and sparsity priors. PRISM supports a fast MSE objective and a Rician negative log-likelihood (NLL) that jointly learns sigma without oracle information. A lightweight nuisance calibration module (smooth bias field and per-measurement scale/offset) is included for robustness and regularized to identity in clean-data tests. On synthetic crossing-fiber data (SNR=30; five methods, 16 crossing angles), PRISM achieves 3.5 degrees best-match angular error with 95% recall, which is 1.9x lower than the best baseline (MSMT-CSD, 6.8 degrees, 83% recall); in NLL mode with learned sigma, error drops to 2.3 degrees with 99% recall, resolving crossings down to 20 degrees. On the DiSCo1 phantom (NLL mode), PRISM improves connectivity correlation over CSD baselines at all four tracking angles (best r=.934 at 25 degrees vs. .920 for MSMT-CSD). Whole-brain HCP fitting (~741k voxels, MSE mode) completes in ~12 min on a single GPU with near-identical results across random seeds.
>
---
#### [new 109] A Reasoning-Enabled Vision-Language Foundation Model for Chest X-ray Interpretation
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于医学影像分析任务，旨在解决CXR解读中AI缺乏解释性的问题。提出CheXOne模型，同时生成诊断预测和可解释的推理过程。**

- **链接: [https://arxiv.org/pdf/2604.00493](https://arxiv.org/pdf/2604.00493)**

> **作者:** Yabin Zhang; Chong Wang; Yunhe Gao; Jiaming Liu; Maya Varma; Justin Xu; Sophie Ostmeier; Jin Long; Sergios Gatidis; Seena Dehkharghani; Arne Michalson; Eun Kyoung Hong; Christian Bluethgen; Haiwei Henry Guo; Alexander Victor Ortiz; Stephan Altmayer; Sandhya Bodapati; Joseph David Janizek; Ken Chang; Jean-Benoit Delbrouck; Akshay S. Chaudhari; Curtis P. Langlotz
>
> **备注:** Codes: this https URL Models: this https URL
>
> **摘要:** Chest X-rays (CXRs) are among the most frequently performed imaging examinations worldwide, yet rising imaging volumes increase radiologist workload and the risk of diagnostic errors. Although artificial intelligence (AI) systems have shown promise for CXR interpretation, most generate only final predictions, without making explicit how visual evidence is translated into radiographic findings and diagnostic predictions. We present CheXOne, a reasoning-enabled vision-language model for CXR interpretation. CheXOne jointly generates diagnostic predictions and explicit, clinically grounded reasoning traces that connect visual evidence, radiographic findings, and these predictions. The model is trained on 14.7 million instruction and reasoning samples curated from 30 public datasets spanning 36 CXR interpretation tasks, using a two-stage framework that combines instruction tuning with reinforcement learning to improve reasoning quality. We evaluate CheXOne in zero-shot settings across visual question answering, report generation, visual grounding and reasoning assessment, covering 17 evaluation settings. CheXOne outperforms existing medical and general-domain foundation models and achieves strong performance on independent public benchmarks. A clinical reader study demonstrates that CheXOne-drafted reports are comparable to or better than resident-written reports in 55% of cases, while effectively addressing clinical indications and enhancing both report writing and CXR interpretation efficiency. Further analyses involving radiologists reveal that the generated reasoning traces show high clinical factuality and provide causal support for the final predictions, offering a plausible explanation for the performance gains. These results suggest that explicit reasoning can improve model performance, interpretability and clinical utility in AI-assisted CXR interpretation.
>
---
#### [new 110] Diff3R: Feed-forward 3D Gaussian Splatting with Uncertainty-aware Differentiable Optimization
- **分类: cs.CV**

- **简介: 该论文提出Diff3R，解决3D高斯溅射中快速推理与高质量渲染的平衡问题。通过将可微优化层融入训练，提升测试时优化效果。**

- **链接: [https://arxiv.org/pdf/2604.01030](https://arxiv.org/pdf/2604.01030)**

> **作者:** Yueh-Cheng Liu; Jozef Hladký; Matthias Nießner; Angela Dai
>
> **备注:** Project page: this https URL, Video: this https URL
>
> **摘要:** Recent advances in 3D Gaussian Splatting (3DGS) present two main directions: feed-forward models offer fast inference in sparse-view settings, while per-scene optimization yields high-quality renderings but is computationally expensive. To combine the benefits of both, we introduce Diff3R, a novel framework that explicitly bridges feed-forward prediction and test-time optimization. By incorporating a differentiable 3DGS optimization layer directly into the training loop, our network learns to predict an optimal initialization for test-time optimization rather than a conventional zero-shot result. To overcome the computational cost of backpropagating through the optimization steps, we propose computing gradients via the Implicit Function Theorem and a scalable, matrix-free PCG solver tailored for 3DGS optimization. Additionally, we incorporate a data-driven uncertainty model into the optimization process by adaptively controlling how much the parameters are allowed to change during optimization. This approach effectively mitigates overfitting in under-constrained regions and increases robustness against input outliers. Since our proposed optimization layer is model-agnostic, we show that it can be seamlessly integrated into existing feed-forward 3DGS architectures for both pose-given and pose-free methods, providing improvements for test-time optimization.
>
---
#### [new 111] Dynamic Graph Neural Network with Adaptive Features Selection for RGB-D Based Indoor Scene Recognition
- **分类: cs.CV**

- **简介: 该论文属于室内场景识别任务，旨在解决RGB-D数据中关键局部特征的自适应选择与利用问题。提出一种动态图模型，融合RGB和深度模态特征以提升识别性能。**

- **链接: [https://arxiv.org/pdf/2604.00372](https://arxiv.org/pdf/2604.00372)**

> **作者:** Qiong Liu; Ruofei Xiong; Xingzhen Chen; Muyao Peng; You Yang
>
> **摘要:** Multi-modality of color and depth, i.e., RGB-D, is of great importance in recent research of indoor scene recognition. In this kind of data representation, depth map is able to describe the 3D structure of scenes and geometric relations among objects. Previous works showed that local features of both modalities are vital for promotion of recognition accuracy. However, the problem of adaptive selection and effective exploitation on these key local features remains open in this field. In this paper, a dynamic graph model is proposed with adaptive node selection mechanism to solve the above problem. In this model, a dynamic graph is built up to model the relations among objects and scene, and a method of adaptive node selection is proposed to take key local features from both modalities of RGB and depth for graph modeling. After that, these nodes are grouped by three different levels, representing near or far relations among objects. Moreover, the graph model is updated dynamically according to attention weights. Finally, the updated and optimized features of RGB and depth modalities are fused together for indoor scene recognition. Experiments are performed on public datasets including SUN RGB-D and NYU Depth v2. Extensive results demonstrate that our method has superior performance when comparing to state-of-the-arts methods, and show that the proposed method is able to exploit crucial local features from both modalities of RGB and depth.
>
---
#### [new 112] A ROS 2 Wrapper for Florence-2: Multi-Mode Local Vision-Language Inference for Robotic Systems
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉-语言任务，解决机器人中基础模型集成问题。设计ROS 2封装，支持多种交互模式，实现本地部署与高效推理。**

- **链接: [https://arxiv.org/pdf/2604.01179](https://arxiv.org/pdf/2604.01179)**

> **作者:** J. E. Domínguez-Vidal
>
> **备注:** 5 pages, 1 figure
>
> **摘要:** Foundation vision-language models are becoming increasingly relevant to robotics because they can provide richer semantic perception than narrow task-specific pipelines. However, their practical adoption in robot software stacks still depends on reproducible middleware integrations rather than on model quality alone. Florence-2 is especially attractive in this regard because it unifies captioning, optical character recognition, open-vocabulary detection, grounding and related vision-language tasks within a comparatively manageable model size. This article presents a ROS 2 wrapper for Florence-2 that exposes the model through three complementary interaction modes: continuous topic-driven processing, synchronous service calls and asynchronous actions. The wrapper is designed for local execution and supports both native installation and Docker container deployment. It also combines generic JSON outputs with standard ROS 2 message bindings for detection-oriented tasks. A functional validation is reported together with a throughput study on several GPUs, showing that local deployment is feasible with consumer grade hardware. The repository is publicly available here: this https URL
>
---
#### [new 113] Using predefined vector systems to speed up neural network multimillion class classification
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于分类任务，旨在解决神经网络多类分类中标签预测效率低的问题。通过利用预定义向量系统，将标签预测复杂度从O(n)降至O(1)，提升推理速度。**

- **链接: [https://arxiv.org/pdf/2604.00779](https://arxiv.org/pdf/2604.00779)**

> **作者:** Nikita Gabdullin; Ilya Androsov
>
> **备注:** 12 pages, 2 figures, 3 tables, 2 algorithms, 1 theorem, 1 lemma
>
> **摘要:** Label prediction in neural networks (NNs) has O(n) complexity proportional to the number of classes. This holds true for classification using fully connected layers and cosine similarity with some set of class prototypes. In this paper we show that if NN latent space (LS) geometry is known and possesses specific properties, label prediction complexity can be significantly reduced. This is achieved by associating label prediction with the O(1) complexity closest cluster center search in a vector system used as target for latent space configuration (LSC). The proposed method only requires finding indexes of several largest and lowest values in the embedding vector making it extremely computationally efficient. We show that the proposed method does not change NN training accuracy computational results. We also measure the time required by different computational stages of NN inference and label prediction on multiple datasets. The experiments show that the proposed method allows to achieve up to 11.6 times overall acceleration over conventional methods. Furthermore, the proposed method has unique properties which allow to predict the existence of new classes.
>
---
#### [new 114] Sit-to-Stand Transitions Detection and Duration Measurement Using Smart Lacelock Sensor
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于动作检测任务，旨在解决老年人跌倒风险评估问题。通过Smart Lacelock传感器检测并测量坐-站转换过程，提升老年人 mobility 监测精度。**

- **链接: [https://arxiv.org/pdf/2604.00175](https://arxiv.org/pdf/2604.00175)**

> **作者:** Md Rafi Islam; Md Rejwanul Haque; Elizabeth Choma; Shannon Hayes; Siobhan McMahon; Xiangrong Shen; Edward Sazonov
>
> **备注:** 10 pages, 11 figures
>
> **摘要:** Postural stability during movement is fundamental to independent living, fall prevention, and overall health, particularly among older adults who experience age-related declines in balance, muscle strength, and mobility. Among daily functional activities, the Sit-to-Stand (SiSt) transition is a critical indicator of lower-limb strength, musculoskeletal health, and fall risk, making it an essential parameter for assessing functional capacity and monitoring physical decline in aging populations. This study presents a methodology SiSt transition detection and duration measurement using the Smart Lacelock sensor, a lightweight, shoe-mounted device that integrates a load cell, accelerometer, and gyroscope for motion analysis. The methodology was evaluated in 16 older adults (age: mean: 76.84, SD: 3.45 years) performing SiSt tasks within the Short Physical Performance Battery (SPPB) protocol. Features extracted from multimodal signals were used to train and evaluate four machine learning classifiers using a 4-fold participant-independent cross-validation to classify SiSt transitions and measure their duration. The bagged tree classifier achieved an accuracy of 0.98 and an F1 score of 0.8 in classifying SiSt transition. The mean absolute error in duration measurement of the correctly classified transitions was 0.047, and the SD was 0.07 seconds. These findings highlight the potential of the Smart Lacelock sensor for real-world fall-risk assessment and mobility monitoring in older adults.
>
---
#### [new 115] Learning Humanoid Navigation from Human Data
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于机器人导航任务，解决 humanoid 机器人在未知环境中自主导航的问题。通过学习人类行走数据，构建 EgoNav 系统，实现无需机器人数据的零样本部署。**

- **链接: [https://arxiv.org/pdf/2604.00416](https://arxiv.org/pdf/2604.00416)**

> **作者:** Weizhuo Wang; Yanjie Ze; C. Karen Liu; Monroe Kennedy III
>
> **备注:** 8 pages 8 figures
>
> **摘要:** We present EgoNav, a system that enables a humanoid robot to traverse diverse, unseen environments by learning entirely from 5 hours of human walking data, with no robot data or finetuning. A diffusion model predicts distributions of plausible future trajectories conditioned on past trajectory, a 360 deg visual memory fusing color, depth, and semantics, and video features from a frozen DINOv3 backbone that capture appearance cues invisible to depth sensors. A hybrid sampling scheme achieves real-time inference in 10 denoising steps, and a receding-horizon controller selects paths from the predicted distribution. We validate EgoNav through offline evaluations, where it outperforms baselines in collision avoidance and multi-modal coverage, and through zero-shot deployment on a Unitree G1 humanoid across unseen indoor and outdoor environments. Behaviors such as waiting for doors to open, navigating around crowds, and avoiding glass walls emerge naturally from the learned prior. We will release the dataset and trained models. Our website: this https URL
>
---
#### [new 116] QUEST: A robust attention formulation using query-modulated spherical attention
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出QUEST，一种改进的注意力机制，解决Transformer训练不稳定问题。适用于视觉任务，提升模型性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.00199](https://arxiv.org/pdf/2604.00199)**

> **作者:** Hariprasath Govindarajan; Per Sidén; Jacob Roll; Fredrik Lindsten
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** The Transformer model architecture has become one of the most widely used in deep learning and the attention mechanism is at its core. The standard attention formulation uses a softmax operation applied to a scaled dot product between query and key vectors. We explore the role played by norms of the queries and keys, which can cause training instabilities when they arbitrarily increase. We demonstrate how this can happen even in simple Transformer models, in the presence of easy-to-learn spurious patterns in the data. We propose a new attention formulation, QUEry-modulated Spherical aTtention (QUEST), that constrains the keys to a hyperspherical latent space, while still allowing individual tokens to flexibly control the sharpness of the attention distribution. QUEST can be easily used as a drop-in replacement for standard attention. We focus on vision applications while also exploring other domains to highlight the method's generality. We show that (1) QUEST trains without instabilities and (2) produces models with improved performance (3) that are robust to data corruptions and adversarial attacks.
>
---
#### [new 117] Feature-level Site Leakage Reduction for Cross-Hospital Chest X-ray Transfer via Self-Supervised Learning
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于跨医院医学图像迁移任务，旨在解决领域偏移导致的模型失效问题。通过自监督学习和特征级对抗混淆减少站点泄露，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.00263](https://arxiv.org/pdf/2604.00263)**

> **作者:** Ayoub Louaye Bouaziz; Lokmane Chebouba
>
> **备注:** Accepted at The 7th International Conference on Computing Systems and Applications [Algiers,2026]
>
> **摘要:** Cross-hospital failure in chest X-ray models is often attributed to domain shift, yet most work assumes invariance without measuring it. This paper studies how to measure site leakage directly and how that measurement changes conclusions about transfer methods. We study multi-site self-supervised learning (SSL) and feature-level adversarial site confusion for cross-hospital transfer. We pretrain a ResNet-18 on NIH and CheXpert without pathology labels. We then freeze the encoder and train a linear pneumonia classifier on NIH only, evaluating transfer to RSNA. We quantify site leakage using a post hoc linear probe that predicts acquisition site from frozen backbone features $f$ and projection features $z$. Across 3 random seeds, multi-site SSL improves RSNA AUC from 0.6736 $\pm$ 0.0148 (ImageNet initialization) to 0.7804 $\pm$ 0.0197. Adding adversarial site confusion on $f$ reduces measured leakage but does not reliably improve AUC and increases variance. On $f$, site probe accuracy drops from 0.9890 $\pm$ 0.0021 (SSL-only) to 0.8504 $\pm$ 0.0051 (CanonicalF), where chance is 0.50. On $z$, probe accuracy drops from 0.8912 $\pm$ 0.0092 to 0.7810 $\pm$ 0.0250. These results show that measuring leakage changes how transfer methods should be interpreted: multi-site SSL drives transfer, while adversarial confusion exposes the limits of invariance assumptions.
>
---
#### [new 118] Super-Resolving Coarse-Resolution Weather Forecasts With Flow Matching
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于天气预报任务，旨在解决高分辨率预报计算成本高的问题。通过生成式超分辨率技术，在粗分辨率预测后处理中提升分辨率，保持大尺度结构并引入合理小尺度变化。**

- **链接: [https://arxiv.org/pdf/2604.00897](https://arxiv.org/pdf/2604.00897)**

> **作者:** Aymeric Delefosse; Anastase Charantonis; Dominique Béréziat
>
> **备注:** Accepted to Climate Informatics 2026
>
> **摘要:** Machine learning-based weather forecasting models now surpass state-of-the-art numerical weather prediction systems, but training and operating these models at high spatial resolution remains computationally expensive. We present a modular framework that decouples forecasting from spatial resolution by applying learned generative super-resolution as a post-processing step to coarse-resolution forecast trajectories. We formulate super-resolution as a stochastic inverse problem, using a residual formulation to preserve large-scale structure while reconstructing unresolved variability. The model is trained with flow matching exclusively on reanalysis data and is applied to global medium-range forecasts. We evaluate (i) design consistency by re-coarsening super-resolved forecasts and comparing them to the original coarse trajectories, and (ii) high-resolution forecast quality using standard ensemble verification metrics and spectral diagnostics. Results show that super-resolution preserves large-scale structure and variance after re-coarsening, introduces physically consistent small-scale variability, and achieves competitive probabilistic forecast skill at 0.25° resolution relative to an operational ensemble baseline, while requiring only a modest additional training cost compared with end-to-end high-resolution forecasting.
>
---
#### [new 119] True (VIS) Lies: Analyzing How Generative AI Recognizes Intentionality, Rhetoric, and Misleadingness in Visualization Lies
- **分类: cs.HC; cs.CL; cs.CV**

- **简介: 该论文属于视觉误导识别任务，旨在分析生成式AI在识别可视化谎言中的意图、修辞和误导性方面的能力。研究通过实验和用户测试评估了多种大语言模型的表现。**

- **链接: [https://arxiv.org/pdf/2604.01181](https://arxiv.org/pdf/2604.01181)**

> **作者:** Graziano Blasilli; Marco Angelini
>
> **摘要:** This study investigates the ability of multimodal Large Language Models (LLMs) to identify and interpret misleading visualizations, and recognize these observations along with their underlying causes and potential intentionality. Our analysis leverages concepts from visualization rhetoric and a newly developed taxonomy of authorial intents as explanatory lenses. We formulated three research questions and addressed them experimentally using a dataset of 2,336 COVID-19-related tweets, half of which contain misleading visualizations, and supplemented it with real-world examples of perceptual, cognitive, and conceptual errors drawn from VisLies, the IEEE VIS community event dedicated to showcasing deceptive and misleading visualizations. To ensure broad coverage of the current LLM landscape, we evaluated 16 state-of-the-art models. Among them, 15 are open-weight models, spanning a wide range of model sizes, architectural families, and reasoning capabilities. The selection comprises small models, namely Nemotron-Nano-V2-VL (12B parameters), Mistral-Small-3.2 (24B), DeepSeek-VL2 (27B), Gemma3 (27B), and GTA1 (32B); medium-sized models, namely Qianfan-VL (70B), Molmo (72B), GLM-4.5V (108B), LLaVA-NeXT (110B), and Pixtral-Large (124B); and large models, namely Qwen3-VL (235B), InternVL3.5 (241B), Step3 (321B), Llama-4-Maverick (400B), and Kimi-K2.5 (1000B). In addition, we employed OpenAI GPT-5.4, a frontier proprietary model. To establish a human perspective on these tasks, we also conducted a user study with visualization experts to assess how people perceive rhetorical techniques and the authorial intentions behind the same misleading visualizations. This allows comparison between model and expert behavior, revealing similarities and differences that provide insights into where LLMs align with human judgment and where they diverge.
>
---
#### [new 120] AI-assisted Human-in-the-Loop Web Platform for Structural Characterization in Hard drive design
- **分类: cond-mat.mtrl-sci; cs.CV**

- **简介: 该论文属于半导体材料表征任务，旨在解决STEM图像分析中自动化与灵活性的平衡问题。提出人机协同工作流框架，实现多层薄膜厚度与界面粗糙度的精准量化。**

- **链接: [https://arxiv.org/pdf/2604.00359](https://arxiv.org/pdf/2604.00359)**

> **作者:** Utkarsh Pratiush; Huaixun Huyan; Maryam Zahiri Azar; Esmeralda Yitamben; Allen Bourez; Sergei V Kalinin; Vasfi Burak Ozdol
>
> **摘要:** Scanning transmission electron microscopy (STEM) has become a cornerstone instrument for semiconductor materials metrology, enabling nanoscale analysis of complex multilayer structures that define device performance. Developing effective metrology workflows for such systems requires balancing automation with flexibility; rigid pipelines are brittle to sample variability, while purely manual approaches are slow and subjective. Here, we present a tunable human-AI-assisted workflow framework that enables modular and adaptive analysis of STEM images for device characterization. As an illustrative example, we demonstrate a workflow for automated layer thickness and interface roughness quantification in multilayer thin films. The system integrates gradient-based peak detection with interactive correction modules, allowing human input at the design stage while maintaining fully automated execution across samples. Implemented as a web-based interface, it processes TEM/EMD files directly, applies noise reduction and interface tracking algorithms, and outputs statistical roughness and thickness metrics with nanometer precision. This architecture exemplifies a general approach toward adaptive, reusable metrology workflows - bridging human insight and machine precision for scalable, standardized analysis in semiconductor manufacturing. The code is made available at this https URL
>
---
#### [new 121] Beyond Symbolic Solving: Multi Chain-of-Thought Voting for Geometric Reasoning in Large Language Models
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于几何推理任务，旨在解决大语言模型中逻辑推理不足的问题。提出MARS-GPS方法，通过多链式思考投票提升准确性。**

- **链接: [https://arxiv.org/pdf/2604.00890](https://arxiv.org/pdf/2604.00890)**

> **作者:** Md. Abu Bakor Siddique; Shahrin Hossain; Sadman Ahmed Siam; Syed Rifat Raiyan; Hasan Mahmud; Md Kamrul Hasan
>
> **备注:** Under review, 4 figures, 7 tables
>
> **摘要:** Geometric Problem Solving (GPS) remains at the heart of enhancing mathematical reasoning in large language models because it requires the combination of diagrammatic understanding, symbolic manipulation and logical inference. In existing literature, researchers have chiefly focused on synchronising the diagram descriptions with text literals and solving the problem. In this vein, they have either taken a neural, symbolic or neuro-symbolic approach. But this solves only the first two of the requirements, namely diagrammatic understanding and symbolic manipulation, while leaving logical inference underdeveloped. The logical inference is often limited to one chain-of-thought (CoT). To address this weakness in hitherto existing models, this paper proposes MARS-GPS, that generates multiple parallel reasoning rollouts augmented with Python code execution for numerical verification, ranks them using token-level entropy as a confidence signal, and aggregates answers through a multi-stage voting and self-verification pipeline. Empirical results show that MARS-GPS with 8 parallel rollouts achieves 88.8% on Geometry3K, a nearly +11% improvement over the prior state-of-the-art, with accuracy scaling consistently as the number of rollouts increases from 1 to 16 (+6.0% on ablation subset). We provide our code and data in an anonymous repository: this https URL.
>
---
#### [new 122] RT-GS: Gaussian Splatting with Reflection and Transmittance Primitives
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于三维重建任务，旨在解决Gaussian Splatting无法同时建模镜面反射和透射的问题。通过引入反射和透射的高斯基元及可微渲染，实现更真实的视角合成。**

- **链接: [https://arxiv.org/pdf/2604.00509](https://arxiv.org/pdf/2604.00509)**

> **作者:** Kunnong Zeng; Chensheng Peng; Yichen Xie; Masayoshi Tomizuka; Cem Yuksel
>
> **摘要:** Gaussian Splatting is a powerful tool for reconstructing diffuse scenes, but it struggles to simultaneously model specular reflections and the appearance of objects behind semi-transparent surfaces. These specular reflections and transmittance are essential for realistic novel view synthesis, and existing methods do not properly incorporate the underlying physical processes to simulate them. To address this issue, we propose RT-GS, a unified framework that integrates a microfacet material model and ray tracing to jointly model specular reflection and transmittance in Gaussian Splatting. We accomplish this by using separate Gaussian primitives for reflections and transmittance, which allow modeling distant reflections and reconstructing objects behind transparent surfaces concurrently. We utilize a differentiable ray tracing framework to obtain the specular reflection and transmittance appearance. Our experiments demonstrate that our method successfully produces reflections and recovers objects behind transparent surfaces in complex environments, achieving significant qualitative improvements over prior methods where these specular light interactions are prominent.
>
---
#### [new 123] AutoMIA: Improved Baselines for Membership Inference Attack via Agentic Self-Exploration
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于隐私安全任务，旨在解决MIAs性能不足问题。提出AutoMIA框架，通过自探索和策略进化提升攻击效果，无需人工特征工程。**

- **链接: [https://arxiv.org/pdf/2604.01014](https://arxiv.org/pdf/2604.01014)**

> **作者:** Ruhao Liu; Weiqi Huang; Qi Li; Xinchao Wang
>
> **摘要:** Membership Inference Attacks (MIAs) serve as a fundamental auditing tool for evaluating training data leakage in machine learning models. However, existing methodologies predominantly rely on static, handcrafted heuristics that lack adaptability, often leading to suboptimal performance when transferred across different large models. In this work, we propose AutoMIA, an agentic framework that reformulates membership inference as an automated process of self-exploration and strategy evolution. Given high-level scenario specifications, AutoMIA self-explores the attack space by generating executable logits-level strategies and progressively refining them through closed-loop evaluation feedback. By decoupling abstract strategy reasoning from low-level execution, our framework enables a systematic, model-agnostic traversal of the attack search space. Extensive experiments demonstrate that AutoMIA consistently matches or outperforms state-of-the-art baselines while eliminating the need for manual feature engineering.
>
---
#### [new 124] Multi-Camera View Scaling for Data-Efficient Robot Imitation Learning
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人模仿学习任务，旨在提升策略泛化能力。通过多视角摄像机扩展演示数据，生成伪示范以增强训练多样性，提高视觉表示的视角不变性。**

- **链接: [https://arxiv.org/pdf/2604.00557](https://arxiv.org/pdf/2604.00557)**

> **作者:** Yichen Xie; Yixiao Wang; Shuqi Zhao; Cheng-En Wu; Masayoshi Tomizuka; Jianwen Xie; Hao-Shu Fang
>
> **摘要:** The generalization ability of imitation learning policies for robotic manipulation is fundamentally constrained by the diversity of expert demonstrations, while collecting demonstrations across varied environments is costly and difficult in practice. In this paper, we propose a practical framework that exploits inherent scene diversity without additional human effort by scaling camera views during demonstration collection. Instead of acquiring more trajectories, multiple synchronized camera perspectives are used to generate pseudo-demonstrations from each expert trajectory, which enriches the training distribution and improves viewpoint invariance in visual representations. We analyze how different action spaces interact with view scaling and show that camera-space representations further enhance diversity. In addition, we introduce a multiview action aggregation method that allows single-view policies to benefit from multiple cameras during deployment. Extensive experiments in simulation and real-world manipulation tasks demonstrate significant gains in data efficiency and generalization compared to single-view baselines. Our results suggest that scaling camera views provides a practical and scalable solution for imitation learning, which requires minimal additional hardware setup and integrates seamlessly with existing imitation learning algorithms. The website of our project is this https URL.
>
---
#### [new 125] Brain MR Image Synthesis with Multi-contrast Self-attention GAN
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于多模态MRI图像合成任务，旨在解决获取所有影像模态不实际的问题。通过提出3D-MC-SAGAN模型，从单个T2图像生成其他模态，同时保留肿瘤特征。**

- **链接: [https://arxiv.org/pdf/2604.00070](https://arxiv.org/pdf/2604.00070)**

> **作者:** Zaid A. Abod; Furqan Aziz
>
> **备注:** Note: This work has been submitted to the IEEE for possible publication
>
> **摘要:** Accurate and complete multi-modal Magnetic Resonance Imaging (MRI) is essential for neuro-oncological assessment, as each contrast provides complementary anatomical and pathological information. However, acquiring all modalities (e.g., T1c, T1n, T2, T2f) for every patient is often impractical due to time, cost, and patient discomfort, potentially limiting comprehensive tumour evaluation. We propose 3D-MC-SAGAN (3D Multi-Contrast Self-Attention generative adversarial network), a unified 3D multi-contrast synthesis framework that generates high-fidelity missing modalities from a single T2 input while explicitly preserving tumour characteristics. The model employs a multi-scale 3D encoder-decoder generator with residual connections and a novel Memory-Bounded Hybrid Attention (MBHA) block to capture long-range dependencies efficiently, and is trained with a WGAN-GP critic and an auxiliary contrast-conditioning branch to produce T2f, T1n, and T1c volumes within a single unified network. A frozen 3D U-Net-based segmentation module introduces a segmentation-consistency constraint to preserve lesion morphology. The composite objective integrates adversarial, reconstruction, perceptual, structural similarity, contrast-classification, and segmentation-guided losses to align global realism with tumour-preserving structure. Extensive evaluation on 3D brain MRI datasets demonstrates that 3D-MC-SAGAN achieves state-of-the-art quantitative performance and generates visually coherent, anatomically plausible contrasts with improved distribution-level realism. Moreover, it maintains tumour segmentation accuracy comparable to fully acquired multi-modal inputs, highlighting its potential to reduce acquisition burden while preserving clinically meaningful information.
>
---
#### [new 126] HippoCamp: Benchmarking Contextual Agents on Personal Computers
- **分类: cs.AI; cs.CV**

- **简介: 该论文提出HippoCamp基准，用于评估智能体在个人电脑多模态文件管理中的能力，解决用户个性化环境下的文件搜索与推理问题。**

- **链接: [https://arxiv.org/pdf/2604.01221](https://arxiv.org/pdf/2604.01221)**

> **作者:** Zhe Yang; Shulin Tian; Kairui Hu; Shuai Liu; Hoang-Nhat Nguyen; Yichi Zhang; Zujin Guo; Mengying Yu; Zinan Zhang; Jingkang Yang; Chen Change Loy; Ziwei Liu
>
> **备注:** Project Page: this https URL
>
> **摘要:** We present HippoCamp, a new benchmark designed to evaluate agents' capabilities on multimodal file management. Unlike existing agent benchmarks that focus on tasks like web interaction, tool use, or software automation in generic settings, HippoCamp evaluates agents in user-centric environments to model individual user profiles and search massive personal files for context-aware reasoning. Our benchmark instantiates device-scale file systems over real-world profiles spanning diverse modalities, comprising 42.4 GB of data across over 2K real-world files. Building upon the raw files, we construct 581 QA pairs to assess agents' capabilities in search, evidence perception, and multi-step reasoning. To facilitate fine-grained analysis, we provide 46.1K densely annotated structured trajectories for step-wise failure diagnosis. We evaluate a wide range of state-of-the-art multimodal large language models (MLLMs) and agentic methods on HippoCamp. Our comprehensive experiments reveal a significant performance gap: even the most advanced commercial models achieve only 48.3% accuracy in user profiling, struggling particularly with long-horizon retrieval and cross-modal reasoning within dense personal file systems. Furthermore, our step-wise failure diagnosis identifies multimodal perception and evidence grounding as the primary bottlenecks. Ultimately, HippoCamp exposes the critical limitations of current agents in realistic, user-centric environments and provides a robust foundation for developing next-generation personal AI assistants.
>
---
#### [new 127] LAtent Phase Inference from Short time sequences using SHallow REcurrent Decoders (LAPIS-SHRED)
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文提出LAPIS-SHRED，用于从稀疏观测中重建或预测完整的时空动态，解决复杂系统中数据不足的问题。**

- **链接: [https://arxiv.org/pdf/2604.01216](https://arxiv.org/pdf/2604.01216)**

> **作者:** Yuxuan Bao; Xingyue Zhang; J. Nathan Kutz
>
> **摘要:** Reconstructing full spatio-temporal dynamics from sparse observations in both space and time remains a central challenge in complex systems, as measurements can be spatially incomplete and can be also limited to narrow temporal windows. Yet approximating the complete spatio-temporal trajectory is essential for mechanistic insight and understanding, model calibration, and operational decision-making. We introduce LAPIS-SHRED (LAtent Phase Inference from Short time sequence using SHallow REcurrent Decoders), a modular architecture that reconstructs and/or forecasts complete spatiotemporal dynamics from sparse sensor observations confined to short temporal windows. LAPIS-SHRED operates through a three-stage pipeline: (i) a SHRED model is pre-trained entirely on simulation data to map sensor time-histories into a structured latent space, (ii) a temporal sequence model, trained on simulation-derived latent trajectories, learns to propagate latent states forward or backward in time to span unobserved temporal regions from short observational time windows, and (iii) at deployment, only a short observation window of hyper-sparse sensor measurements from the true system is provided, from which the frozen SHRED model and the temporal model jointly reconstruct or forecast the complete spatiotemporal trajectory. The framework supports bidirectional inference, inherits data assimilation and multiscale reconstruction capabilities from its modular structure, and accommodates extreme observational constraints including single-frame terminal inputs. We evaluate LAPIS-SHRED on six experiments spanning complex spatio-temporal physics: turbulent flows, multiscale propulsion physics, volatile combustion transients, and satellite-derived environmental fields, highlighting a lightweight, modular architecture suited for operational settings where observation is constrained by physical or logistical limitations.
>
---
#### [new 128] TRACE: Training-Free Partial Audio Deepfake Detection via Embedding Trajectory Analysis of Speech Foundation Models
- **分类: cs.SD; cs.AI; cs.CV**

- **简介: 该论文属于音频伪造检测任务，解决部分音频深度伪造的检测问题。提出TRACE方法，通过分析语音基础模型的嵌入轨迹动态，实现无需训练的检测。**

- **链接: [https://arxiv.org/pdf/2604.01083](https://arxiv.org/pdf/2604.01083)**

> **作者:** Awais Khan; Muhammad Umar Farooq; Kutub Uddin; Khalid Malik
>
> **摘要:** Partial audio deepfakes, where synthesized segments are spliced into genuine recordings, are particularly deceptive because most of the audio remains authentic. Existing detectors are supervised: they require frame-level annotations, overfit to specific synthesis pipelines, and must be retrained as new generative models emerge. We argue that this supervision is unnecessary. We hypothesize that speech foundation models implicitly encode a forensic signal: genuine speech forms smooth, slowly varying embedding trajectories, while splice boundaries introduce abrupt disruptions in frame-level transitions. Building on this, we propose TRACE (Training-free Representation-based Audio Countermeasure via Embedding dynamics), a training-free framework that detects partial audio deepfakes by analyzing the first-order dynamics of frozen speech foundation model representations without any training, labeled data, or architectural modification. We evaluate TRACE on four benchmarks that span two languages using six speech foundation models. In PartialSpoof, TRACE achieves 8.08% EER, competitive with fine-tuned supervised baselines. In LlamaPartialSpoof, the most challenging benchmark featuring LLM-driven commercial synthesis, TRACE surpasses a supervised baseline outright (24.12% vs. 24.49% EER) without any target-domain data. These results show that temporal dynamics in speech foundation models provide an effective, generalize signal for training-free audio forensics.
>
---
#### [new 129] Pupil Design for Computational Wavefront Estimation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于波前估计任务，旨在提升单次强度测量下的波前恢复能力。通过引入定量不对称性指标，研究瞳孔设计对波前恢复的影响，优化设计以提高恢复效果。**

- **链接: [https://arxiv.org/pdf/2604.00225](https://arxiv.org/pdf/2604.00225)**

> **作者:** Ali Almuallem; Nicholas Chimitt; Bole Ma; Qi Guo; Stanley H. Chan
>
> **摘要:** Establishing a precise connection between imaged intensity and the incident wavefront is essential for emerging applications in adaptive optics, holography, computational microscopy, and non-line-of-sight imaging. While prior work has shown that breaking symmetries in pupil design enables wavefront recovery from a single intensity measurement, there is little guidance on how to design a pupil that improves wavefront estimation. In this work we introduce a quantitative asymmetry metric to bridge this gap and, through an extensive empirical study and supporting analysis, demonstrate that increasing asymmetry enhances wavefront recoverability. We analyze the trade-offs in pupil design, and the impact on light throughput along with performance in noise. Both large-scale simulations and optical bench experiments are carried out to support our findings.
>
---
#### [new 130] MOON3.0: Reasoning-aware Multimodal Representation Learning for E-commerce Product Understanding
- **分类: cs.LG; cs.AI; cs.CV; cs.IR**

- **简介: 该论文提出MOON3.0，解决电商产品理解中的细粒度属性建模问题，通过多模态融合、对比与强化学习、细粒度增强模块提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2604.00513](https://arxiv.org/pdf/2604.00513)**

> **作者:** Junxian Wu; Chenghan Fu; Zhanheng Nie; Daoze Zhang; Bowen Wan; Wanxian Guan; Chuan Yu; Jian Xu; Bo Zheng
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** With the rapid growth of e-commerce, exploring general representations rather than task-specific ones has attracted increasing attention. Although recent multimodal large language models (MLLMs) have driven significant progress in product understanding, they are typically employed as feature extractors that implicitly encode product information into global embeddings, thereby limiting their ability to capture fine-grained attributes. Therefore, we argue that leveraging the reasoning capabilities of MLLMs to explicitly model fine-grained product attributes holds significant potential. Nevertheless, achieving this goal remains non-trivial due to several key challenges: (i) long-context reasoning tends to dilute the model's attention to salient information in the raw input; (ii) supervised fine-tuning (SFT) primarily encourages rigid imitation, limiting the exploration of effective reasoning strategies; and (iii) fine-grained details are progressively attenuated during forward propagation. To address these issues, we propose MOON3.0, the first reasoning-aware MLLM-based model for product representation learning. Our method (1) employs a multi-head modality fusion module to adaptively integrate raw signals; (2) incorporates a joint contrastive and reinforcement learning framework to autonomously explore more effective reasoning strategies; and (3) introduces a fine-grained residual enhancement module to progressively preserve local details throughout the network. Additionally, we release a large-scale multimodal e-commerce benchmark MBE3.0. Experimentally, our model demonstrates state-of-the-art zero-shot performance across various downstream tasks on both our benchmark and public datasets.
>
---
#### [new 131] LiPS: Lightweight Panoptic Segmentation for Resource-Constrained Robotics
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉感知任务，旨在解决资源受限机器人中高效全景分割的问题。提出轻量级方法LiPS，在保持性能的同时降低计算需求。**

- **链接: [https://arxiv.org/pdf/2604.00634](https://arxiv.org/pdf/2604.00634)**

> **作者:** Calvin Galagain; Martyna Poreba; François Goulette; Cyrill Stachniss
>
> **备注:** Submitted to IEEE ICIP 2026. Under review
>
> **摘要:** Panoptic segmentation is a key enabler for robotic perception, as it unifies semantic understanding with object-level reasoning. However, the increasing complexity of state-of-the-art models makes them unsuitable for deployment on resource-constrained platforms such as mobile robots. We propose a novel approach called LiPS that addresses the challenge of efficient-to-compute panoptic segmentation with a lightweight design that retains query-based decoding while introducing a streamlined feature extraction and fusion pathway. It aims at providing a strong panoptic segmentation performance while substantially lowering the computational demands. Evaluations on standard benchmarks demonstrate that LiPS attains accuracy comparable to much heavier baselines, while providing up to 4.5 higher throughput, measured in frames per second, and requiring nearly 6.8 times fewer computations. This efficiency makes LiPS a highly relevant bridge between modern panoptic models and real-world robotic applications.
>
---
#### [new 132] AdaLoRA-QAT: Adaptive Low-Rank and Quantization-Aware Segmentation
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文针对医学图像分割任务，提出AdaLoRA-QAT框架，解决大模型部署效率与精度平衡问题，通过低秩适配和量化训练实现参数压缩与结构保真。**

- **链接: [https://arxiv.org/pdf/2604.01167](https://arxiv.org/pdf/2604.01167)**

> **作者:** Prantik Deb; Srimanth Dhondy; N. Ramakrishna; Anu Kapoor; Raju S. Bapi; Tapabrata Chakraborti
>
> **备注:** Accepted to ISBI 2026(Oral Presentation)
>
> **摘要:** Chest X-ray (CXR) segmentation is an important step in computer-aided diagnosis, yet deploying large foundation models in clinical settings remains challenging due to computational constraints. We propose AdaLoRA-QAT, a two-stage fine-tuning framework that combines adaptive low-rank encoder adaptation with full quantization-aware training. Adaptive rank allocation improves parameter efficiency, while selective mixed-precision INT8 quantization preserves structural fidelity crucial for clinical reliability. Evaluated across large-scale CXR datasets, AdaLoRA-QAT achieves 95.6% Dice, matching full-precision SAM decoder fine-tuning while reducing trainable parameters by 16.6\times and yielding 2.24\times model compression. A Wilcoxon signed-rank test confirms that quantization does not significantly degrade segmentation accuracy. These results demonstrate that AdaLoRA-QAT effectively balances accuracy, efficiency, and structural trust-worthiness, enabling compact and deployable foundation models for medical image segmentation. Code and pretrained models are available at: this https URL
>
---
#### [new 133] Generalizable Dense Reward for Long-Horizon Robotic Tasks
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文针对长周期机器人任务中的性能问题，提出VLLR框架，结合外部奖励与内在自信心奖励，提升任务完成效率和成功率，无需人工设计奖励。**

- **链接: [https://arxiv.org/pdf/2604.00055](https://arxiv.org/pdf/2604.00055)**

> **作者:** Silong Yong; Stephen Sheng; Carl Qi; Xiaojie Wang; Evan Sheehan; Anurag Shivaprasad; Yaqi Xie; Katia Sycara; Yesh Dattatreya
>
> **备注:** Project page: this https URL
>
> **摘要:** Existing robotic foundation policies are trained primarily via large-scale imitation learning. While such models demonstrate strong capabilities, they often struggle with long-horizon tasks due to distribution shift and error accumulation. While reinforcement learning (RL) can finetune these models, it cannot work well across diverse tasks without manual reward engineering. We propose VLLR, a dense reward framework combining (1) an extrinsic reward from Large Language Models (LLMs) and Vision-Language Models (VLMs) for task progress recognition, and (2) an intrinsic reward based on policy self-certainty. VLLR uses LLMs to decompose tasks into verifiable subtasks and then VLMs to estimate progress to initialize the value function for a brief warm-up phase, avoiding prohibitive inference cost during full training; and self-certainty provides per-step intrinsic guidance throughout PPO finetuning. Ablation studies reveal complementary benefits: VLM-based value initialization primarily improves task completion efficiency, while self-certainty primarily enhances success rates, particularly on out-of-distribution tasks. On the CHORES benchmark covering mobile manipulation and navigation, VLLR achieves up to 56% absolute success rate gains over the pretrained policy, up to 5% gains over state-of-the-art RL finetuning methods on in-distribution tasks, and up to $10\%$ gains on out-of-distribution tasks, all without manual reward engineering. Additional visualizations can be found in this https URL
>
---
#### [new 134] Toward Personalized Darts Training: A Data-Driven Framework Based on Skeleton-Based Biomechanical Analysis and Motion Modeling
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于运动分析任务，旨在解决传统 dart 训练方法的不足。通过数据驱动框架，提取运动特征并提供个性化反馈，提升训练精准度。**

- **链接: [https://arxiv.org/pdf/2604.01130](https://arxiv.org/pdf/2604.01130)**

> **作者:** Zhantao Chen; Dongyi He; Jin Fang; Xi Chen; Yisuo Liu; Xiaozhen Zhong; Xuejun Hu
>
> **摘要:** As sports training becomes more data-driven, traditional dart coaching based mainly on experience and visual observation is increasingly inadequate for high-precision, goal-oriented movements. Although prior studies have highlighted the importance of release parameters, joint motion, and coordination in dart throwing, most quantitative methods still focus on local variables, single-release metrics, or static template matching. These approaches offer limited support for personalized training and often overlook useful movement variability. This paper presents a data-driven dart training assistance system. The system creates a closed-loop framework spanning motion capture, feature modeling, and personalized feedback. Dart-throwing data were collected in markerless conditions using a Kinect 2.0 depth sensor and an optical camera. Eighteen kinematic features were extracted from four biomechanical dimensions: three-link coordination, release velocity, multi-joint angular configuration, and postural stability. Two modules were developed: a personalized optimal throwing trajectory model that combines historical high-quality samples with the minimum jerk criterion, and a motion deviation diagnosis and recommendation model based on z-scores and hierarchical logic. A total of 2,396 throwing samples from professional and non-professional athletes were collected. Results show that the system generates smooth personalized reference trajectories consistent with natural human movement. Case studies indicate that it can detect poor trunk stability, abnormal elbow displacement, and imbalanced velocity control, then provide targeted recommendations. The framework shifts dart evaluation from deviation from a uniform standard to deviation from an individual's optimal control range, improving personalization and interpretability for darts training and other high-precision target sports.
>
---
#### [new 135] A Dual-Stream Transformer Architecture for Illumination-Invariant TIR-LiDAR Person Tracking
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于目标跟踪任务，旨在解决光照变化下人体跟踪性能下降的问题。通过融合热红外与深度信息，提出一种双流Transformer架构，提升机器人在复杂环境中的跟踪能力。**

- **链接: [https://arxiv.org/pdf/2604.00363](https://arxiv.org/pdf/2604.00363)**

> **作者:** Yuki Minase; Kanji Tanaka
>
> **备注:** 6 pages, 4 figures, technical report
>
> **摘要:** Robust person tracking is a critical capability for autonomous mobile robots operating in diverse and unpredictable environments. While RGB-D tracking has shown high precision, its performance severely degrades under challenging illumination conditions, such as total darkness or intense backlighting. To achieve all-weather robustness, this paper proposes a novel Thermal-Infrared and Depth (TIR-D) tracking architecture that leverages the standard sensor suite of SLAM-capable robots, namely LiDAR and TIR cameras. A major challenge in TIR-D tracking is the scarcity of annotated multi-modal datasets. To address this, we introduce a sequential knowledge transfer strategy that evolves structural priors from a large-scale thermal-trained model into the TIR-D domain. By employing a differential learning rate strategy -- referred to as ``Fine-grained Differential Learning Rate Strategy'' -- we effectively preserve pre-trained feature extraction capabilities while enabling rapid adaptation to geometric depth cues. Experimental results demonstrate that our proposed TIR-D tracker achieves superior performance, with an Average Overlap (AO) of 0.700 and a Success Rate (SR) of 58.7\%, significantly outperforming conventional RGB-transfer and single-modality baselines. Our approach provides a practical and resource-efficient solution for robust human-following in all-weather robotics applications.
>
---
#### [new 136] Compact Keyframe-Optimized Multi-Agent Gaussian Splatting SLAM
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于多智能体SLAM任务，解决通信带宽受限下的高效3D地图构建问题。通过压缩冗余信息和优化数据传输，显著降低通信负载。**

- **链接: [https://arxiv.org/pdf/2604.00804](https://arxiv.org/pdf/2604.00804)**

> **作者:** Monica M.Q. Li; Pierre-Yves Lajoie; Jialiang Liu; Giovanni Beltrame
>
> **摘要:** Efficient multi-agent 3D mapping is essential for robotic teams operating in unknown environments, but dense representations hinder real-time exchange over constrained communication links. In multi-agent Simultaneous Localization and Mapping (SLAM), systems typically rely on a centralized server to merge and optimize the local maps produced by individual agents. However, sharing these large map representations, particularly those generated by recent methods such as Gaussian Splatting, becomes a bottleneck in real-world scenarios with limited bandwidth. We present an improved multi-agent RGB-D Gaussian Splatting SLAM framework that reduces communication load while preserving map fidelity. First, we incorporate a compaction step into our SLAM system to remove redundant 3D Gaussians, without degrading the rendering quality. Second, our approach performs centralized loop closure computation without initial guess, operating in two modes: a pure rendered-depth mode that requires no data beyond the 3D Gaussians, and a camera-depth mode that includes lightweight depth images for improved registration accuracy and additional Gaussian pruning. Evaluation on both synthetic and real-world datasets shows up to 85-95\% reduction in transmitted data compared to state-of-the-art approaches in both modes, bringing 3D Gaussian multi-agent SLAM closer to practical deployment in real-world scenarios. Code: this https URL
>
---
## 更新

#### [replaced 001] Beyond the Golden Data: Resolving the Motion-Vision Quality Dilemma via Timestep Selective Training
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.25527](https://arxiv.org/pdf/2603.25527)**

> **作者:** Xiangyang Luo; Qingyu Li; Yuming Li; Guanbo Huang; Yongjie Zhu; Wenyu Qin; Meng Wang; Pengfei Wan; Shao-Lun Huang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Recent advances in video generation models have achieved impressive results. However, these models heavily rely on the use of high-quality data that combines both high visual quality and high motion quality. In this paper, we identify a key challenge in video data curation: the Motion-Vision Quality Dilemma. We discovered that visual quality and motion intensity inherently exhibit a negative correlation, making it hard to obtain golden data that excels in both aspects. To address this challenge, we first examine the hierarchical learning dynamics of video diffusion models and conduct gradient-based analysis on quality-degraded samples. We discover that quality-imbalanced data can produce gradients similar to golden data at appropriate timesteps. Based on this, we introduce the novel concept of Timestep selection in Training Process. We propose Timestep-aware Quality Decoupling (TQD), which modifies the data sampling distribution to better match the model's learning process. For certain types of data, the sampling distribution is skewed toward higher timesteps for motion-rich data, while high visual quality data is more likely to be sampled during lower timesteps. Through extensive experiments, we demonstrate that TQD enables training exclusively on separated imbalanced data to achieve performance surpassing conventional training with better data, challenging the necessity of perfect data in video generation. Moreover, our method also boosts model performance when trained on high-quality data, showcasing its effectiveness across different data scenarios.
>
---
#### [replaced 002] Seeing Beyond the Image: ECG and Anatomical Knowledge-Guided Myocardial Scar Segmentation from Late Gadolinium-Enhanced Images
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.14702](https://arxiv.org/pdf/2511.14702)**

> **作者:** Farheen Ramzan; Yusuf Kiberu; Nikesh Jathanna; Meryem Jabrane; Vicente Grau; Shahnaz Jamil-Copley; Richard H. Clayton; Chen; Chen
>
> **备注:** oral presentation at International Symposium on Biomedical Imaging (ISBI 2026)
>
> **摘要:** Accurate segmentation of myocardial scar from late gadolinium enhanced (LGE) cardiac MRI is essential for evaluating tissue viability, yet remains challenging due to variable contrast and imaging artifacts. Electrocardiogram (ECG) signals provide complementary physiological information, as conduction abnormalities can help localize or suggest scarred myocardial regions. In this work, we propose a novel multimodal framework that integrates ECG-derived electrophysiological information with anatomical priors from the AHA-17 atlas for physiologically consistent LGE-based scar segmentation. As ECGs and LGE-MRIs are not acquired simultaneously, we introduce a Temporal Aware Feature Fusion (TAFF) mechanism that dynamically weights and fuses features based on their acquisition time difference. Our method was evaluated on a clinical dataset and achieved substantial gains over the state-of-the-art image-only baseline (nnU-Net), increasing the average Dice score for scars from 0.6149 to 0.8463 and achieving high performance in both precision (0.9115) and sensitivity (0.9043). These results show that integrating physiological and anatomical knowledge allows the model to "see beyond the image", setting a new direction for robust and physiologically grounded cardiac scar segmentation.
>
---
#### [replaced 003] TempoControl: Temporal Attention Guidance for Text-to-Video Models
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.02226](https://arxiv.org/pdf/2510.02226)**

> **作者:** Shira Schiber; Ofir Lindenbaum; Idan Schwartz
>
> **备注:** Accepted CVPR'26
>
> **摘要:** Recent advances in generative video models have enabled the creation of high-quality videos based on natural language prompts. However, these models frequently lack fine-grained temporal control, meaning they do not allow users to specify when particular visual elements should appear within a generated sequence. In this work, we introduce TempoControl, a method that allows for temporal alignment of visual concepts during inference, without requiring retraining or additional supervision. TempoControl utilizes cross-attention maps, a key component of text-to-video diffusion models, to guide the timing of concepts through a novel optimization approach. Our method steers attention using three complementary principles: aligning its temporal pattern with a control signal (correlation), adjusting its strength where visibility is required (magnitude), and preserving semantic consistency (entropy). TempoControl provides precise temporal control while maintaining high video quality and diversity. We demonstrate its effectiveness across various applications, including temporal reordering of single and multiple objects, action timing, and audio-aligned video generation. Project page: this https URL.
>
---
#### [replaced 004] Erased, But Not Forgotten: Erased Rectified Flow Transformers Still Remain Unsafe Under Concept Attack
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.00635](https://arxiv.org/pdf/2510.00635)**

> **作者:** Nanxiang Jiang; Zhaoxin Fan; Enhan Kang; Daiheng Gao; Yun Zhou; Yanxia Chang; Zheng Zhu; Yeying Jin; Wenjun Wu
>
> **摘要:** Recent advances in text-to-image (T2I) diffusion models have enabled impressive generative capabilities, but they also raise significant safety concerns due to the potential to produce harmful or undesirable content. While concept erasure has been explored as a mitigation strategy, most existing approaches and corresponding attack evaluations are tailored to Stable Diffusion (SD) and exhibit limited effectiveness when transferred to next-generation rectified flow transformers such as Flux. In this work, we present ReFlux, the first concept attack method specifically designed to assess the robustness of concept erasure in the latest rectified flow-based T2I framework. Our approach is motivated by the observation that existing concept erasure techniques, when applied to Flux, fundamentally rely on a phenomenon known as attention localization. Building on this insight, we propose a simple yet effective attack strategy that specifically targets this property. At its core, a reverse-attention optimization strategy is introduced to effectively reactivate suppressed signals while stabilizing attention. This is further reinforced by a velocity-guided dynamic that enhances the robustness of concept reactivation by steering the flow matching process, and a consistency-preserving objective that maintains the global layout and preserves unrelated content. Extensive experiments consistently demonstrate the effectiveness and efficiency of the proposed attack method, establishing a reliable benchmark for evaluating the robustness of concept erasure strategies in rectified flow transformers.
>
---
#### [replaced 005] Robust Residual Finite Scalar Quantization for Neural Compression
- **分类: eess.IV; cs.CV; eess.AS**

- **简介: 该论文属于神经压缩任务，解决多阶段FSQ中残差幅度衰减问题，提出RFSQ方法，通过自适应缩放和层归一化提升性能。**

- **链接: [https://arxiv.org/pdf/2508.15860](https://arxiv.org/pdf/2508.15860)**

> **作者:** Xiaoxu Zhu; Xiaojie Yu; Guangchao Yao; Yiming Ren; Baoxiang Li
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Finite Scalar Quantization (FSQ) offers simplified training but suffers from residual magnitude decay in multi-stage settings, where subsequent stages receive exponentially weaker signals. We propose Robust Residual Finite Scalar Quantization (RFSQ), addressing this fundamental limitation through two novel conditioning strategies: learnable scaling factors and invertible layer normalization. Our experiments across audio and image modalities demonstrate RFSQ's effectiveness and generalizability. In audio reconstruction at 24 bits/frame, RFSQ-LayerNorm achieves 3.646 DNSMOS, a 3.6% improvement over state-of-the-art RVQ (3.518). On ImageNet, RFSQ achieves 0.102 L1 loss and 0.100 perceptual loss, with LayerNorm providing 9.7% L1 improvement and 17.4% perceptual improvement over unconditioned variants. The LayerNorm strategy consistently outperforms alternatives by maintaining normalized input statistics across stages, effectively preventing exponential magnitude decay that limits naive residual approaches. RFSQ combines FSQ's simplicity with multi-stage quantization's representational power, establishing a new standard for neural compression across diverse modalities.
>
---
#### [replaced 006] OTPrune: Distribution-Aligned Visual Token Pruning via Optimal Transport
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.20205](https://arxiv.org/pdf/2602.20205)**

> **作者:** Xiwen Chen; Wenhui Zhu; Gen Li; Xuanzhao Dong; Yujian Xiong; Hao Wang; Peijie Qiu; Qingquan Song; Zhipeng Wang; Shao Tang; Yalin Wang; Abolfazl Razi
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Multi-modal large language models (MLLMs) achieve strong visual-language reasoning but suffer from high inference cost due to redundant visual tokens. Recent work explores visual token pruning to accelerate inference, while existing pruning methods overlook the underlying distributional structure of visual representations. We propose OTPrune, a training-free framework that formulates pruning as distribution alignment via optimal transport (OT). By minimizing the 2-Wasserstein distance between the full and pruned token distributions, OTPrune preserves both local diversity and global representativeness while reducing inference cost. Moreover, we derive a tractable submodular objective that enables efficient optimization, and theoretically prove its monotonicity and submodularity, providing a principled foundation for stable and efficient pruning. We further provide a comprehensive analysis that explains how distributional alignment contributes to stable and semantically faithful pruning. Comprehensive experiments on wider benchmarks demonstrate that OTPrune achieves superior performance-efficiency tradeoffs compared to state-of-the-art methods. The code is available at this https URL.
>
---
#### [replaced 007] ActionMesh: Animated 3D Mesh Generation with Temporal 3D Diffusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.16148](https://arxiv.org/pdf/2601.16148)**

> **作者:** Remy Sabathier; David Novotny; Niloy J. Mitra; Tom Monnier
>
> **备注:** CVPR 2026. Project webpage with code and videos: this https URL . V2 update includes more baseline models with a larger evaluation set on our new publicly released benchmark ActionBench, and {3D+video}-to-animated-mesh qualitative comparison in supplemental
>
> **摘要:** Generating animated 3D objects is at the heart of many applications, yet most advanced works are typically difficult to apply in practice because of their limited setup, their long runtime, or their limited quality. We introduce ActionMesh, a generative model that predicts production-ready 3D meshes "in action" in a feed-forward manner. Drawing inspiration from early video models, our key insight is to modify existing 3D diffusion models to include a temporal axis, resulting in a framework we dubbed "temporal 3D diffusion". Specifically, we first adapt the 3D diffusion stage to generate a sequence of synchronized latents representing time-varying and independent 3D shapes. Second, we design a temporal 3D autoencoder that translates a sequence of independent shapes into the corresponding deformations of a pre-defined reference shape, allowing us to build an animation. Combining these two components, ActionMesh generates animated 3D meshes from different inputs like a monocular video, a text description, or even a 3D mesh with a text prompt describing its animation. Besides, compared to previous approaches, our method is fast and produces results that are rig-free and topology consistent, hence enabling rapid iteration and seamless applications like texturing and retargeting. We evaluate our model on standard video-to-4D benchmarks (Consistent4D, Objaverse) and report state-of-the-art performances on both geometric accuracy and temporal consistency, demonstrating that our model can deliver animated 3D meshes with unprecedented speed and quality.
>
---
#### [replaced 008] Learning to Infer Parameterized Representations of Plants from 3D Scans
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.22337](https://arxiv.org/pdf/2505.22337)**

> **作者:** Samara Ghrer; Christophe Godin; Stefanie Wuhrer
>
> **摘要:** Plants frequently contain numerous organs, organized in 3D branching systems defining the plant's architecture. Reconstructing the architecture of plants from unstructured observations is challenging because of self-occlusion and spatial proximity between organs, which are often thin structures. To achieve the challenging task, we propose an approach that allows to infer a parameterized representation of the plant's architecture from a given 3D scan of a plant. In addition to the plant's branching structure, this representation contains parametric information for each plant organ, and can therefore be used directly in a variety of tasks. In this data-driven approach, we train a recursive neural network with virtual plants generated using a procedural model. After training, the network allows to infer a parametric tree-like representation based on an input 3D point cloud. Our method is applicable to any plant that can be represented as binary axial tree. We quantitatively evaluate our approach on Chenopodium Album plants on reconstruction, segmentation and skeletonization, which are important problems in plant phenotyping. In addition to carrying out several tasks at once, our method achieves results on-par with strong baselines for each task. We apply our method, trained exclusively on synthetic data, to 3D scans and show that it generalizes well.
>
---
#### [replaced 009] MOOZY: A Patient-First Foundation Model for Computational Pathology
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.27048](https://arxiv.org/pdf/2603.27048)**

> **作者:** Yousef Kotp; Vincent Quoc-Huy Trinh; Christopher Pal; Mahdi S. Hosseini
>
> **摘要:** Computational pathology needs whole-slide image (WSI) foundation models that transfer across diverse clinical tasks, yet current approaches remain largely slide-centric, often depend on private data and expensive paired-report supervision, and do not explicitly model relationships among multiple slides from the same patient. We present MOOZY, a patient-first pathology foundation model in which the patient case, not the individual slide, is the core unit of representation. MOOZY explicitly models dependencies across all slides from the same patient via a case transformer during pretraining, combining multi-stage open self-supervision with scaled low-cost task supervision. In Stage 1, we pretrain a vision-only slide encoder on 77,134 public slide feature grids using masked self-distillation. In Stage 2, we align these representations with clinical semantics using a case transformer and multi-task supervision over 333 tasks from 56 public datasets, including 205 classification and 128 survival tasks across four endpoints. Across eight held-out tasks with five-fold frozen-feature probe evaluation, MOOZY achieves best or tied-best performance on most metrics and improves macro averages over TITAN by +7.37%, +5.50%, and +7.83% and over PRISM by +8.83%, +10.70%, and +9.78% for weighted F1, weighted ROC-AUC, and balanced accuracy, respectively. MOOZY is also parameter efficient with 85.77M parameters, 14x smaller than GigaPath. These results demonstrate that open, reproducible patient-level pretraining yields transferable embeddings, providing a practical path toward scalable patient-first histopathology foundation models.
>
---
#### [replaced 010] Learning by Neighbor-Aware Semantics, Deciding by Open-form Flows: Towards Robust Zero-Shot Skeleton Action Recognition
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.09388](https://arxiv.org/pdf/2511.09388)**

> **作者:** Yang Chen; Miaoge Li; Zhijie Rao; Deze Zeng; Song Guo; Jingcai Guo
>
> **备注:** Accepted by CVPR 2026 Findings; Project Code: this https URL
>
> **摘要:** Recognizing unseen skeleton action categories remains highly challenging due to the absence of corresponding skeletal priors. Existing approaches generally follow an ``align-then-classify'' paradigm but face two fundamental issues, \textit{i.e.}, (i) fragile point-to-point alignment arising from imperfect semantics, and (ii) rigid classifiers restricted by static decision boundaries and coarse-grained anchors. To address these issues, we propose a novel method for zero-shot skeleton action recognition, termed \texttt{\textbf{Flora}}, which builds upon \textbf{F}lexib\textbf{L}e neighb\textbf{O}r-aware semantic attunement and open-form dist\textbf{R}ibution-aware flow cl\textbf{A}ssifier. Specifically, we flexibly attune textual semantics by incorporating neighboring inter-class contextual cues to form direction-aware regional semantics, coupled with a cross-modal geometric consistency objective that ensures stable and robust point-to-region alignment. Furthermore, we employ noise-free flow matching to bridge the modality distribution gap between semantic and skeleton latent embeddings, while a condition-free contrastive regularization enhances discriminability, leading to a distribution-aware classifier with fine-grained decision boundaries achieved through token-level velocity predictions. Extensive experiments on three benchmark datasets validate the effectiveness of our method, showing particularly impressive performance even when trained with only 10% of the seen data. Code is available at this https URL.
>
---
#### [replaced 011] Video2LoRA: Unified Semantic-Controlled Video Generation via Per-Reference-Video LoRA
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.08210](https://arxiv.org/pdf/2603.08210)**

> **作者:** Zexi Wu; Baolu Li; Jing Dai; Yiming Zhang; Yue Ma; Qinghe Wang; Xu Jia; Hongming Xu
>
> **备注:** 10 pages
>
> **摘要:** Achieving semantic alignment across diverse video generation conditions remains a significant challenge. Methods that rely on explicit structural guidance often enforce rigid spatial constraints that limit semantic flexibility, whereas models tailored for individual control types lack interoperability and adaptability. These design bottlenecks hinder progress toward flexible and efficient semantic video generation. To address this, we propose Video2LoRA, a scalable and generalizable framework for semantic-controlled video generation that conditions on a reference video. Video2LoRA employs a lightweight hypernetwork to predict personalized LoRA weights for each semantic input, which are combined with auxiliary matrices to form adaptive LoRA modules integrated into a frozen diffusion backbone. This design enables the model to generate videos consistent with the reference semantics while preserving key style and content variations, eliminating the need for any per-condition training. Notably, the final model weights less than 150MB, making it highly efficient for storage and deployment. Video2LoRA achieves coherent, semantically aligned generation across diverse conditions and exhibits strong zero-shot generalization to unseen semantics.
>
---
#### [replaced 012] DW-DGAT: Dynamically Weighted Dual Graph Attention Network for Neurodegenerative Disease Diagnosis
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.10001](https://arxiv.org/pdf/2601.10001)**

> **作者:** Chengjia Liang; Zhenjiong Wang; Chao Chen; Ruizhi Zhang; Songxi Liang; Hai Xie; Haijun Lei; Zhongwei Huang
>
> **备注:** The exended version of an AAAI-2026 accepted poster paper
>
> **摘要:** Parkinson's disease (PD) and Alzheimer's disease (AD) are the two most prevalent and incurable neurodegenerative diseases (NDs) worldwide, for which early diagnosis is critical to delay their progression. However, the high dimensionality of multi-metric data with diverse structural forms, the heterogeneity of neuroimaging and phenotypic data, and class imbalance collectively pose significant challenges to early ND diagnosis. To address these challenges, we propose a dynamically weighted dual graph attention network (DW-DGAT) that integrates: (1) a general-purpose data fusion strategy to merge three structural forms of multi-metric data; (2) a dual graph attention architecture based on brain regions and inter-sample relationships to extract both micro- and macro-level features; and (3) a class weight generation mechanism combined with two stable and effective loss functions to mitigate class imbalance. Rigorous experiments, based on the Parkinson Progression Marker Initiative (PPMI) and Alzheimer's Disease Neuroimaging Initiative (ADNI) studies, demonstrate the state-of-the-art performance of our approach.
>
---
#### [replaced 013] Unified Medical Image Tokenizer for Autoregressive Synthesis and Understanding
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.19225](https://arxiv.org/pdf/2505.19225)**

> **作者:** Chenglong Ma; Yuanfeng Ji; Jin Ye; Zilong Li; Chenhui Wang; Junzhi Ning; Wei Li; Lihao Liu; Qiushan Guo; Tianbin Li; Junjun He; Hongming Shan
>
> **摘要:** Autoregressive modeling has driven major advances in multimodal AI, yet its application to medical imaging remains constrained by the absence of a unified image tokenizer that simultaneously preserves fine-grained anatomical structures and rich clinical semantics across heterogeneous modalities. Existing approaches jointly optimize image reconstruction and textual semantic objectives, relying on large-scale image-caption pairs and are prone to gradient interference. This is ill-suited for the medical domain where paired data are scarce and abundant unpaired images remain unexploited. This work identifies these issues in building unified medical image tokenizers, and introduces a principled two-stage training framework using visual representation as a bridge to address them. The propose visual representation alignment stage enables the utilization of large-scale unpaired medical images to ensure reconstruction fidelity and establish foundational semantics, alleviating the interference and better preparing for the second stage where fine-grained textual semantics are injected using image-text pairs. The resulting tokenizer, MedITok, is trained on over 33 million medical images spanning 9 modalities and 2 million image-text pairs. MedITok achieves state-of-the-art performance on 30+ benchmarks spanning 9 imaging modalities and 4 task families. It further enables autoregressive modeling for diagnostic and generative applications, serving as a scalable component for future multimodal models with unified synthesis and understanding capabilities in the medical domain. Project page: this https URL
>
---
#### [replaced 014] Ar2Can: An Architect and an Artist Leveraging a Canvas for Multi-Human Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22690](https://arxiv.org/pdf/2511.22690)**

> **作者:** Shubhankar Borse; Phuc Pham; Farzad Farhadzadeh; Seokeon Choi; Phong Ha Nguyen; Anh Tuan Tran; Sungrack Yun; Munawar Hayat; Fatih Porikli
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Despite recent advances in personalized image generation, existing models consistently fail to produce reliable multi-human scenes, often merging or losing facial identity. We present Ar2Can, a novel two-stage framework that disentangles spatial planning from identity rendering for multi-human generation. The Architect predicts structured layouts, specifying where each person should appear. The Artist then synthesizes photorealistic images, guided by a spatially-grounded face matching reward that combines Hungarian spatial alignment with identity similarity. This approach ensures faces are rendered at correct locations and faithfully preserve reference identities. We develop two Architect variants, seamlessly integrated with our diffusion-based Artist model. This is optimized via Group Relative Policy Optimization (GRPO) using compositional rewards for count accuracy, image quality, and identity matching. Evaluated on the MultiHuman-Testbench, Ar2Can achieves substantial improvements in both count accuracy and identity preservation, while maintaining high perceptual quality. Notably, our method achieves these results using primarily synthetic data, without requiring real multi-human images. Project page: this https URL.
>
---
#### [replaced 015] Cross-modal Proxy Evolving for OOD Detection with Vision-Language Models
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2601.08476](https://arxiv.org/pdf/2601.08476)**

> **作者:** Hao Tang; Yu Liu; Shuanglin Yan; Fei Shen; Shengfeng He; Jing Qin
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Reliable zero-shot detection of out-of-distribution (OOD) inputs is critical for deploying vision-language models in open-world settings. However, the lack of labeled negatives in zero-shot OOD detection necessitates proxy signals that remain effective under distribution shift. Existing negative-label methods rely on a fixed set of textual proxies, which (i) sparsely sample the semantic space beyond in-distribution (ID) classes and (ii) remain static while only visual features drift, leading to cross-modal misalignment and unstable predictions. In this paper, we propose CoEvo, a training- and annotation-free test-time framework that performs bidirectional, sample-conditioned adaptation of both textual and visual proxies. Specifically, CoEvo introduces a proxy-aligned co-evolution mechanism to maintain two evolving proxy caches, which dynamically mines contextual textual negatives guided by test images and iteratively refines visual proxies, progressively realigning cross-modal similarities and enlarging local OOD margins. Finally, we dynamically re-weight the contributions of dual-modal proxies to obtain a calibrated OOD score that is robust to distribution shift. Extensive experiments on standard benchmarks demonstrate that CoEvo achieves state-of-the-art performance, improving AUROC by 1.33% and reducing FPR95 by 45.98% on ImageNet-1K compared to strong negative-label baselines.
>
---
#### [replaced 016] Monocular Models are Strong Learners for Multi-View Human Mesh Recovery
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.20391](https://arxiv.org/pdf/2603.20391)**

> **作者:** Haoyu Xie; Shengkai Xu; Cheng Guo; Muhammad Usama Saleem; Wenhan Wu; Chen Chen; Ahmed Helmy; Pu Wang; Hongfei Xue
>
> **摘要:** Multi-view human mesh recovery (HMR) is broadly deployed in diverse domains where high accuracy and strong generalization are essential. Existing approaches can be broadly grouped into geometry-based and learning-based methods. However, geometry-based methods (e.g., triangulation) rely on cumbersome camera calibration, while learning-based approaches often generalize poorly to unseen camera configurations due to the lack of multi-view training data, limiting their performance in real-world scenarios. To enable calibration-free reconstruction that generalizes to arbitrary camera setups, we propose a training-free framework that leverages pretrained single-view HMR models as strong priors, eliminating the need for multi-view training data. Our method first constructs a robust and consistent multi-view initialization from single-view predictions, and then refines it via test-time optimization guided by multi-view consistency and anatomical constraints. Extensive experiments demonstrate state-of-the-art performance on standard benchmarks, surpassing multi-view models trained with explicit multi-view supervision.
>
---
#### [replaced 017] Unregistered Spectral Image Fusion: Unmixing, Adversarial Learning, and Recoverability
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.21510](https://arxiv.org/pdf/2603.21510)**

> **作者:** Jiahui Song; Sagar Shrestha; Xiao Fu
>
> **摘要:** This paper addresses the fusion of a pair of spatially unregistered hyperspectral image (HSI) and multispectral image (MSI) covering roughly overlapping regions. HSIs offer high spectral but low spatial resolution, while MSIs provide the opposite. The goal is to integrate their complementary information to enhance both HSI spatial resolution and MSI spectral resolution. While hyperspectral-multispectral fusion (HMF) has been widely studied, the unregistered setting remains challenging. Many existing methods focus solely on MSI super-resolution, leaving HSI unchanged. Supervised deep learning approaches were proposed for HSI super-resolution, but rely on accurate training data, which is often unavailable. Moreover, theoretical analyses largely address the co-registered case, leaving unregistered HMF poorly understood. In this work, an unsupervised framework is proposed to simultaneously super-resolve both MSI and HSI. The method integrates coupled spectral unmixing for MSI super-resolution with latent-space adversarial learning for HSI super-resolution. Theoretical guarantees on the recoverability of the super-resolution MSI and HSI are established under reasonable generative models -- providing, to our best knowledge, the first such insights for unregistered HMF. The approach is validated on semi-real and real HSI-MSI pairs across diverse conditions.
>
---
#### [replaced 018] EoS-FM: Can an Ensemble of Specialist Models act as a Generalist Feature Extractor?
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.21523](https://arxiv.org/pdf/2511.21523)**

> **作者:** Pierre Adorni; Minh-Tan Pham; Stéphane May; Sébastien Lefèvre
>
> **摘要:** Recent advances in foundation models have shown great promise in domains such as natural language processing and computer vision, and similar efforts are now emerging in the Earth Observation community. These models aim to generalize across tasks with limited supervision, reducing the need for training separate models for each task. However, current strategies, which largely focus on scaling model size and dataset volume, require prohibitive computational and data resources, limiting accessibility to only a few large institutions. Moreover, this paradigm of ever-larger models stands in stark contrast with the principles of sustainable and environmentally responsible AI, as it leads to immense carbon footprints and resource inefficiency. In this work, we present a novel and efficient alternative: an Ensemble-of-Specialists framework for building Remote Sensing Foundation Models (RSFMs). Our method decomposes the training process into lightweight, task-specific ConvNeXtV2 specialists that can be frozen and reused. This modular approach offers strong advantages in efficiency, interpretability, and extensibility. Moreover, it naturally supports federated training, pruning, and continuous specialist integration, making it particularly well-suited for collaborative and resource-constrained settings. Our framework sets a new direction for building scalable and efficient RSFMs. All codes and pretrained models are available on the public repo at this https URL .
>
---
#### [replaced 019] Conditional Polarization Guidance for Camouflaged Object Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.30008](https://arxiv.org/pdf/2603.30008)**

> **作者:** QIfan Zhang; Hao Wang; Xiangrong Qin; Ruijie Li
>
> **备注:** 11 pages, 10 figures, 4 tables
>
> **摘要:** Camouflaged object detection (COD) aims to identify targets that are highly blended with their backgrounds. Recent works have shown that the optical characteristics of polarization cues play a significant role in improving camouflaged object detection. However, most existing polarization-based approaches depend on complex visual encoders and fusion mechanisms, leading to increased model complexity and computational overhead, while failing to fully explore how polarization can explicitly guide hierarchical RGB representation learning. To address these limitations, we propose CPGNet, an asymmetric RGB-polarization framework that introduces a conditional polarization guidance mechanism to explicitly regulate RGB feature learning for camouflaged object detection. Specifically, we design a lightweight polarization interaction module that jointly models these complementary cues and generates reliable polarization guidance in a unified manner. Unlike conventional feature fusion strategies, the proposed conditional guidance mechanism dynamically modulates RGB features using polarization priors, enabling the network to focus on subtle discrepancies between camouflaged objects and their backgrounds. Furthermore, we introduce a polarization edge-guided frequency refinement strategy that enhances high-frequency components under polarization constraints, effectively breaking camouflage patterns. Finally, we develop an iterative feedback decoder to perform coarse-to-fine feature calibration and progressively refine camouflage prediction. Extensive experiments on polarization datasets across multiple tasks, along with evaluations on non-polarization datasets, demonstrate that CPGNet consistently outperforms state-of-the-art methods.
>
---
#### [replaced 020] Object Affordance Recognition and Grounding via Multi-scale Cross-modal Representation Learning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.01184](https://arxiv.org/pdf/2508.01184)**

> **作者:** Xinhang Wan; Dongqiang Gou; Xinwang Liu; En Zhu; Xuming He
>
> **摘要:** A core problem of Embodied AI is to learn object manipulation from observation, as humans do. To achieve this, it is important to localize 3D object affordance areas through observation such as images (3D affordance grounding) and understand their functionalities (affordance classification). Previous attempts usually tackle these two tasks separately, leading to inconsistent predictions due to lacking proper modeling of their dependency. In addition, these methods typically only ground the incomplete affordance areas depicted in images, failing to predict the full potential affordance areas, and operate at a fixed scale, resulting in difficulty in coping with affordances significantly varying in scale with respect to the whole object. To address these issues, we propose a novel approach that learns an affordance-aware 3D representation and employs a stage-wise inference strategy leveraging the dependency between grounding and classification tasks. Specifically, we first develop a cross-modal 3D representation through efficient fusion and multi-scale geometric feature propagation, enabling inference of full potential affordance areas at a suitable regional scale. Moreover, we adopt a simple two-stage prediction mechanism, effectively coupling grounding and classification for better affordance understanding. Experiments demonstrate the effectiveness of our method, showing improved performance in both affordance grounding and classification.
>
---
#### [replaced 021] Towards Online Multi-Modal Social Interaction Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2503.19851](https://arxiv.org/pdf/2503.19851)**

> **作者:** Xinpeng Li; Shijian Deng; Bolin Lai; Weiguo Pian; James M. Rehg; Yapeng Tian
>
> **备注:** Accepted to Transactions on Machine Learning Research (TMLR). Project page: this https URL
>
> **摘要:** In this paper, we introduce a new problem, Online-MMSI, where the model must perform multimodal social interaction understanding (MMSI) using only historical information. Given a recorded video and a multi-party dialogue, the AI assistant is required to immediately identify the speaker's referent, which is critical for real-world human-AI interaction. Without access to future conversational context, both humans and models experience substantial performance degradation when moving from offline to online settings. To tackle the challenges, we propose Online-MMSI-VLM, a novel framework based on multimodal large language models. The core innovations of our approach lie in two components: (1) multi-party conversation forecasting, which predicts upcoming speaker turns and utterances in a coarse-to-fine manner; and (2) socially-aware visual prompting, which highlights salient social cues in each video frame using bounding boxes and body keypoints. Our model achieves state-of-the-art results on three tasks across two datasets, significantly outperforming the baseline and demonstrating the effectiveness of Online-MMSI-VLM. Project page: this https URL.
>
---
#### [replaced 022] CHEEM: Continual Learning by Reuse, New, Adapt and Skip -- A Hierarchical Exploration-Exploitation Approach
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2303.08250](https://arxiv.org/pdf/2303.08250)**

> **作者:** Chinmay Savadikar; Michelle Dai; Tianfu Wu
>
> **备注:** CVPR 2026
>
> **摘要:** To effectively manage the complexities of real-world dynamic environments, continual learning must incrementally acquire, update, and accumulate knowledge from a stream of tasks of different nature without suffering from catastrophic forgetting of prior knowledge. While this capability is innate to human cognition, it remains a significant challenge for modern deep learning systems. At the heart of this challenge lies the stability-plasticity dilemma: the need to balance leveraging prior knowledge, integrating novel information, and allocating model capacity adaptively based on task complexity and synergy. In this paper, we propose a novel exemplar-free class-incremental continual learning (ExfCCL) framework that addresses these issues through a Hierarchical Exploration-Exploitation (HEE) approach. The core of our method is a HEE-guided efficient neural architecture search (HEE-NAS) that enables a learning-to-adapt backbone via four primitive operations - reuse, new, adapt, and skip - thereby serving as an internal memory that dynamically updates selected components across streaming tasks. To address the task ID inference problem in ExfCCL, we exploit an external memory of task centroids proposed in the prior art. We term our method CHEEM (Continual Hierarchical-Exploration-Exploitation Memory). CHEEM is evaluated on the challenging MTIL and VDD benchmarks using both Tiny and Base Vision Transformers and a proposed holistic Figure-of-Merit (FoM) metric. It significantly outperforms state-of-the-art prompting-based continual learning methods, closely approaching full fine-tuning upper bounds. Furthermore, it learns adaptive model structures tailored to individual tasks in a semantically meaningful way. Our code is available at this https URL .
>
---
#### [replaced 023] TeFlow: Enabling Multi-frame Supervision for Self-Supervised Feed-forward Scene Flow Estimation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于场景流估计任务，解决自监督方法在遮挡下监督不稳定的问题。提出TeFlow，通过多帧时序一致性监督提升性能。**

- **链接: [https://arxiv.org/pdf/2602.19053](https://arxiv.org/pdf/2602.19053)**

> **作者:** Qingwen Zhang; Chenhan Jiang; Xiaomeng Zhu; Yunqi Miao; Yushan Zhang; Olov Andersson; Patric Jensfelt
>
> **备注:** CVPR 2026; 16 pages, 8 figures
>
> **摘要:** Self-supervised feed-forward methods for scene flow estimation offer real-time efficiency, but their supervision from two-frame point correspondences is unreliable and often breaks down under occlusions. Multi-frame supervision has the potential to provide more stable guidance by incorporating motion cues from past frames, yet naive extensions of two-frame objectives are ineffective because point correspondences vary abruptly across frames, producing inconsistent signals. In the paper, we present TeFlow, enabling multi-frame supervision for feed-forward models by mining temporally consistent supervision. TeFlow introduces a temporal ensembling strategy that forms reliable supervisory signals by aggregating the most temporally consistent motion cues from a candidate pool built across multiple frames. Extensive evaluations demonstrate that TeFlow establishes a new state-of-the-art for self-supervised feed-forward methods, achieving performance gains of up to 33\% on the challenging Argoverse 2 and nuScenes datasets. Our method performs on par with leading optimization-based methods, yet speeds up 150 times. The code is open-sourced at this https URL along with trained model weights.
>
---
#### [replaced 024] RANGER: A Monocular Zero-Shot Semantic Navigation Framework through Visual Contextual Adaptation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉导航任务，解决无监督环境下的目标定位与导航问题。提出RANGER框架，通过单目相机和视觉上下文学习，提升导航效率与适应性。**

- **链接: [https://arxiv.org/pdf/2512.24212](https://arxiv.org/pdf/2512.24212)**

> **作者:** Ming-Ming Yu; Yi Chen; Börje F. Karlsson; Wenjun Wu
>
> **备注:** Accepted at ICRA 2026
>
> **摘要:** Efficient target localization and autonomous navigation in complex environments are fundamental to real-world embodied applications. While recent advances in multimodal foundation models have enabled zero-shot object goal navigation, allowing robots to search for arbitrary objects without fine-tuning, existing methods face two key limitations: (1) heavy reliance on ground-truth depth and pose information, which restricts applicability in real-world scenarios; and (2) lack of visual in-context learning (VICL) capability to extract geometric and semantic priors from environmental context, as in a short traversal video. To address these challenges, we propose RANGER, a novel zero-shot, open-vocabulary semantic navigation framework that operates using only a monocular camera. Leveraging powerful 3D foundation models, RANGER eliminates the dependency on depth and pose while exhibiting strong VICL capability. By simply observing a short video of the target environment, the system can also significantly improve task efficiency without requiring architectural modifications or task-specific retraining. The framework integrates several key components: keyframe-based 3D reconstruction, semantic point cloud generation, vision-language model (VLM)-driven exploration value estimation, high-level adaptive waypoint selection, and low-level action execution. Experiments on the HM3D benchmark and real-world environments demonstrate that RANGER achieves competitive performance in terms of navigation success rate and exploration efficiency, while showing superior VICL adaptability, with no previous 3D mapping of the environment required.
>
---
#### [replaced 025] MoRe-3DGSMR: Motion-resolved reconstruction framework for free-breathing pulmonary MRI based on 3D Gaussian representation
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.04959](https://arxiv.org/pdf/2505.04959)**

> **作者:** Tengya Peng; Ruyi Zha; Qing Zou
>
> **摘要:** This study presents an unsupervised, motion-resolved reconstruction framework for high-resolution, free-breathing pulmonary magnetic resonance imaging (MRI), utilizing a three-dimensional Gaussian representation (3DGS). The proposed method leverages 3DGS to address the challenges of motion-resolved 3D isotropic pulmonary MRI reconstruction by enabling data smoothing between voxels for continuous spatial representation. Pulmonary MRI data acquisition is performed using a golden-angle radial sampling trajectory, with respiratory motion signals extracted from the center of k-space in each radial spoke. Based on the estimated motion signal, the k-space data is sorted into multiple respiratory phases. A 3DGS framework is then applied to reconstruct a reference image volume from the first motion state. Subsequently, a patient-specific convolutional neural network is trained to estimate the deformation vector fields (DVFs), which are used to generate the remaining motion states through spatial transformation of the reference volume. The proposed reconstruction pipeline is evaluated on six datasets from six subjects and bench-marked against three state-of-the-art reconstruction methods. The experimental findings demonstrate that the proposed reconstruction framework effectively reconstructs high-resolution, motion-resolved pulmonary MR images. Compared with existing approaches, it achieves superior image quality, reflected by higher signal-to-noise ratio and contrast-to-noise ratio. The proposed unsupervised 3DGS-based reconstruction method enables accurate motion-resolved pulmonary MRI with isotropic spatial resolution. Its superior performance in image quality metrics over state-of-the-art methods highlights its potential as a robust solution for clinical pulmonary MR imaging.
>
---
#### [replaced 026] IMTBench: A Multi-Scenario Cross-Modal Collaborative Evaluation Benchmark for In-Image Machine Translation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.10495](https://arxiv.org/pdf/2603.10495)**

> **作者:** Jiahao Lyu; Pei Fu; Zhenhang Li; Weichao Zeng; Shaojie Zhang; Jiahui Yang; Can Ma; Yu Zhou; Zhenbo Luo; Jian Luan
>
> **摘要:** End-to-end In-Image Machine Translation (IIMT) aims to convert text embedded within an image into a target language while preserving the original visual context, layout, and rendering style. However, existing IIMT benchmarks are largely synthetic and thus fail to reflect real-world complexity, while current evaluation protocols focus on single-modality metrics and overlook cross-modal faithfulness between rendered text and model outputs. To address these shortcomings, we present In-image Machine Translation Benchmark (IMTBench), a new benchmark of 2,500 image translation samples covering four practical scenarios and nine languages. IMTBench supports multi-aspect evaluation, including translation quality, background preservation, overall image quality, and a cross-modal alignment score that measures consistency between the translated text produced by the model and the text rendered in the translated image. We benchmark strong commercial cascade systems, and both closed- and open-source unified multi-modal models, and observe large performance gaps across scenarios and languages, especially on natural scenes and resource-limited languages, highlighting substantial headroom for end-to-end image text translation. We hope IMTBench establishes a standardized benchmark to accelerate progress in this emerging task.
>
---
#### [replaced 027] Communicating about Space: Language-Mediated Spatial Integration Across Partial Views
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.27183](https://arxiv.org/pdf/2603.27183)**

> **作者:** Ankur Sikarwar; Debangan Mishra; Sudarshan Nikhil; Ponnurangam Kumaraguru; Aishwarya Agrawal
>
> **摘要:** Humans build shared spatial understanding by communicating partial, viewpoint-dependent observations. We ask whether Multimodal Large Language Models (MLLMs) can do the same, aligning distinct egocentric views through dialogue to form a coherent, allocentric mental model of a shared environment. To study this systematically, we introduce COSMIC, a benchmark for Collaborative Spatial Communication. In this setting, two static MLLM agents observe a 3D indoor environment from different viewpoints and exchange natural-language messages to solve spatial queries. COSMIC contains 899 diverse scenes and 1250 question-answer pairs spanning five tasks. We find a capability hierarchy, MLLMs are most reliable at identifying shared anchor objects across views, perform worse on relational reasoning, and largely fail at building globally consistent maps, performing near chance, even for frontier models. Moreover, we find thinking capability yields gains in anchor grounding, but is insufficient for higher-level spatial communication. To contextualize model behavior, we collect 250 human-human dialogues. Humans achieve 95% aggregate accuracy, while the best model, Gemini-3-Pro-Thinking, reaches 72%, leaving substantial room for improvement. Moreover, human conversations grow more precise as partners align on a shared spatial understanding, whereas MLLMs keep exploring without converging, suggesting limited capacity to form and sustain a robust shared mental model throughout the dialogue. Our code and data is available at this https URL.
>
---
#### [replaced 028] Grow, Assess, Compress: Adaptive Backbone Scaling for Memory-Efficient Class Incremental Learning
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.08426](https://arxiv.org/pdf/2603.08426)**

> **作者:** Adrian Garcia-Castañeda; Jon Irureta; Jon Imaz; Aizea Lojo
>
> **摘要:** Class Incremental Learning (CIL) poses a fundamental challenge: maintaining a balance between the plasticity required to learn new tasks and the stability needed to prevent catastrophic forgetting. While expansion-based methods effectively mitigate forgetting by adding task-specific parameters, they suffer from uncontrolled architectural growth and memory overhead. In this paper, we propose a novel dynamic scaling framework that adaptively manages model capacity through a cyclic "GRow, Assess, ComprEss" (GRACE) strategy. Crucially, we supplement backbone expansion with a novel saturation assessment phase that evaluates the utilization of the model's capacity. This assessment allows the framework to make informed decisions to either expand the architecture or compress the backbones into a streamlined representation, preventing parameter explosion. Experimental results demonstrate that our approach achieves state-of-the-art performance across multiple CIL benchmarks, while reducing memory footprint by up to a 73% compared to purely expansionist models.
>
---
#### [replaced 029] Q-REAL: Towards Realism and Plausibility Evaluation for AI-Generated Content
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.16908](https://arxiv.org/pdf/2511.16908)**

> **作者:** Shushi Wang; Zicheng Zhang; Chunyi Li; Wei Wang; Liya Ma; Fengjiao Chen; Xiaoyu Li; Xuezhi Cao; Guangtao Zhai; Xiaohong Liu
>
> **摘要:** Quality assessment of AI-generated content is crucial for evaluating model capability and guiding model optimization. However, most existing quality assessment datasets and models provide only a single quality score, which is too coarse to offer targeted guidance for improving generative models. In current applications of AI-generated images, realism and plausibility are two critical dimensions, and with the emergence of unified generation-understanding models, fine-grained evaluation along these dimensions becomes especially effective for improving generative performance. Therefore, we introduce Q-Real, a novel dataset for fine-grained evaluation of realism and plausibility in AI-generated images. Q-Real consists of 3,088 images generated by popular text-to-image models. For each image, we annotate the locations of major entities and provide a set of judgment questions and attribution descriptions for these along the dimensions of realism and plausibility. Considering that recent advances in multi-modal large language models (MLLMs) enable fine-grained evaluation of AI-generated images, we construct Q-Real Bench to evaluate them on two tasks: judgment and grounding with reasoning. Finally, to enhance MLLM capabilities, we design a fine-tuning framework and conduct experiments on multiple MLLMs using our dataset. Experimental results demonstrate the high quality and significance of our dataset and the comprehensiveness of the benchmark. Dataset and code will be released upon publication.
>
---
#### [replaced 030] WaveGuard: Robust Deepfake Detection and Source Tracing via Dual-Tree Complex Wavelet and Graph Neural Networks
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.08614](https://arxiv.org/pdf/2505.08614)**

> **作者:** Ziyuan He; Zhiqing Guo; Liejun Wang; Gaobo Yang; Yunfeng Diao; Dan Ma
>
> **备注:** 14 pages, 6 figures, 7 tables
>
> **摘要:** Deepfake technology poses increasing risks such as privacy invasion and identity theft. To address these threats, we propose WaveGuard, a proactive watermarking framework that enhances robustness and imperceptibility via frequency-domain embedding and graph-based structural consistency. Specifically, we embed watermarks into high-frequency sub-bands using Dual-Tree Complex Wavelet Transform (DT-CWT) and employ a Structural Consistency Graph Neural Network (SC-GNN) to preserve visual quality. We also design an attention module to refine embedding precision. Experimental results on face swap and reenactment tasks demonstrate that WaveGuard outperforms state-of-the-art methods in both robustness and visual quality. Code is available at this https URL.
>
---
#### [replaced 031] Representation Learning with Semantic-aware Instance and Sparse Token Alignments
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.08165](https://arxiv.org/pdf/2601.08165)**

> **作者:** Phuoc-Nguyen Bui; Toan Duc Nguyen; Junghyun Bum; Duc-Tai Le; Hyunseung Choo
>
> **备注:** Accepted to ICPR 2026
>
> **摘要:** Medical contrastive vision-language pre-training (VLP) has demonstrated significant potential in improving performance on downstream tasks. Traditional approaches typically employ contrastive learning, treating paired image-report samples as positives and unpaired ones as negatives. However, in medical datasets, there can be substantial similarities between images or reports from different patients. Rigidly treating all unpaired samples as negatives, can disrupt the underlying semantic structure and negatively impact the quality of the learned representations. In this paper, we propose a multi-level alignment framework, Representation Learning with Semantic-aware Instance and Sparse Token Alignments (SISTA) by exploiting the semantic correspondence between medical image and radiology reports at two levels, i.e., image-report and patch-word levels. Specifically, we improve the conventional contrastive learning by incorporating inter-report similarity to eliminate the false negatives and introduce a method to effectively align image patches with relevant word tokens. Experimental results demonstrate the effectiveness of the proposed framework in improving transfer performance across different datasets on three downstream tasks: image classification, image segmentation, and object detection. Notably, our framework achieves significant improvements in fine-grained tasks even with limited labeled data. Codes and pre-trained models will be made available.
>
---
#### [replaced 032] Error Propagation Mechanisms and Compensation Strategies for Quantized Diffusion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.12094](https://arxiv.org/pdf/2508.12094)**

> **作者:** Songwei Liu; Chao Zeng; Chenqian Yan; Xurui Peng; Xing Wang; Fangmin Chen; Xing Mei
>
> **摘要:** Diffusion models have transformed image synthesis by establishing unprecedented quality and creativity benchmarks. Nevertheless, their large-scale deployment faces challenges due to computationally intensive iterative denoising processes. Although post-training quantization (PTQ) provides an effective pathway for accelerating sampling, the iterative nature of diffusion models causes stepwise quantization errors to accumulate progressively during generation, inevitably compromising output fidelity. To address this challenge, we develop a theoretical framework that mathematically formulates error propagation in Diffusion Models (DMs), deriving per-step quantization error propagation equations and establishing the first closed-form solution for cumulative error. Building on this theoretical foundation, we propose a timestep-aware cumulative error compensation scheme. Extensive experiments on multiple image datasets demonstrate that our compensation strategy effectively mitigates error propagation, significantly enhancing existing PTQ methods. Specifically, it achieves a 1.2 PSNR improvement over SVDQuant on SDXL W4A4, while incurring only an additional $<$ 0.5\% time overhead.
>
---
#### [replaced 033] The Prism Hypothesis: Harmonizing Semantic and Pixel Representations via Unified Autoencoding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.19693](https://arxiv.org/pdf/2512.19693)**

> **作者:** Weichen Fan; Haiwen Diao; Quan Wang; Dahua Lin; Ziwei Liu
>
> **备注:** Code link: this https URL
>
> **摘要:** Deep representations across modalities are inherently intertwined. In this paper, we systematically analyze the spectral characteristics of various semantic and pixel encoders. Interestingly, our study uncovers a highly inspiring and rarely explored correspondence between an encoder's feature spectrum and its functional role: semantic encoders primarily capture low-frequency components that encode abstract meaning, whereas pixel encoders additionally retain high-frequency information that conveys fine-grained detail. This heuristic finding offers a unifying perspective that ties encoder behavior to its underlying spectral structure. We define it as the Prism Hypothesis, where each data modality can be viewed as a projection of the natural world onto a shared feature spectrum, just like the prism. Building on this insight, we propose Unified Autoencoding (UAE), a model that harmonizes semantic structure and pixel details via an innovative frequency-band modulator, enabling their seamless coexistence. Extensive experiments demonstrate that UAE effectively unifies semantic abstraction and pixel-level fidelity within a single latent space, achieving state-of-the-art performance. Moreover, we show that UAE can be directly applied to pixel-space modeling, significantly improving both FID and IS over the vanilla JIT baseline. Our code is avaliable at: this https URL.
>
---
#### [replaced 034] OmniEgoCap: Camera-Agnostic Sequence-Level Egocentric Motion Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.19283](https://arxiv.org/pdf/2512.19283)**

> **作者:** Kyungwon Cho; Hanbyul Joo
>
> **备注:** Project Page: this https URL
>
> **摘要:** The proliferation of commercial egocentric devices offers a unique lens into human behavior, yet reconstructing full-body 3D motion remains difficult due to frequent self-occlusion and the 'out-of-sight' nature of the wearer's limbs. While head and hand trajectories provide sparse anchor points, current methods often overfit to specific hardware optics or rely on expensive, post-hoc optimizations that compromise motion naturalness. In this paper, we present OmniEgoCap, a unified diffusion framework that scales egocentric reconstruction to diverse capture setups. By shifting from short-term windowed estimation to sequence-level inference, our method captures a global perspective and recovers invariant physical attributes, such as height and body proportions, that provide critical constraints for disambiguating head-only cues. To ensure hardware-agnostic generalization, we introduce a geometry-aware visibility augmentation strategy that treats intermittent hand appearances as principled geometric constraints rather than missing data. Our architecture jointly predicts temporally coherent motion and consistent body shape, establishing a new state-of-the-art on public benchmarks and demonstrating robust performance across diverse, in-the-wild environments.
>
---
#### [replaced 035] Geometric-Photometric Event-based 3D Gaussian Ray Tracing
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于3D重建任务，解决事件相机在3D高斯泼溅中利用时间信息的问题。提出GPERT框架，通过分离几何与辐射渲染，提升重建精度与速度。**

- **链接: [https://arxiv.org/pdf/2512.18640](https://arxiv.org/pdf/2512.18640)**

> **作者:** Kai Kohyama; Yoshimitsu Aoki; Guillermo Gallego; Shintaro Shiba
>
> **备注:** 15 pages, 12 figures, 5 tables
>
> **摘要:** Event cameras offer a high temporal resolution over traditional frame-based cameras, which makes them suitable for motion and structure estimation. However, it has been unclear how event-based 3D Gaussian Splatting (3DGS) approaches could leverage fine-grained temporal information of sparse events. This work proposes GPERT, a framework to address the trade-off between accuracy and temporal resolution in event-based 3DGS. Our key idea is to decouple the rendering into two branches: event-by-event geometry (depth) rendering and snapshot-based radiance (intensity) rendering, by using ray-tracing and the image of warped events. The extensive evaluation shows that our method achieves state-of-the-art performance on the real-world datasets and competitive performance on the synthetic dataset. Also, the proposed method works without prior information (e.g., pretrained image reconstruction models) or COLMAP-based initialization, is more flexible in the event selection number, and achieves sharp reconstruction on scene edges with fast training time. We hope that this work deepens our understanding of the sparse nature of events for 3D reconstruction. this https URL
>
---
#### [replaced 036] EditCtrl: Disentangled Local and Global Control for Real-Time Generative Video Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.15031](https://arxiv.org/pdf/2602.15031)**

> **作者:** Yehonathan Litman; Shikun Liu; Dario Seyb; Nicholas Milef; Yang Zhou; Carl Marshall; Shubham Tulsiani; Caleb Leak
>
> **备注:** Project page: this https URL
>
> **摘要:** High-fidelity generative video editing has seen significant quality improvements by leveraging pre-trained video foundation models. However, their computational cost is a major bottleneck, as they are often designed to inefficiently process the full video context regardless of the inpainting mask's size, even for sparse, localized edits. In this paper, we introduce EditCtrl, an efficient video inpainting control framework that focuses computation only where it is needed. Our approach features a novel local video context module that operates solely on masked tokens, yielding a computational cost proportional to the edit size. This local-first generation is then guided by a lightweight temporal global context embedder that ensures video-wide context consistency with minimal overhead. Not only is EditCtrl 10 times more compute efficient than state-of-the-art generative editing methods, it even improves editing quality compared to methods designed with full-attention. Finally, we showcase how EditCtrl unlocks new capabilities, including multi-region editing with text prompts and autoregressive content propagation.
>
---
#### [replaced 037] Science-T2I: Addressing Scientific Illusions in Image Synthesis
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2504.13129](https://arxiv.org/pdf/2504.13129)**

> **作者:** Jialuo Li; Wenhao Chai; Xingyu Fu; Haiyang Xu; Saining Xie
>
> **备注:** Accepted to CVPR 2025. Code, docs, weight, benchmark and training data are all avaliable at this https URL
>
> **摘要:** Current image generation models produce visually compelling but scientifically implausible images, exposing a fundamental gap between visual fidelity and physical realism. In this work, we introduce ScienceT2I, an expert-annotated dataset comprising a training set of over 20k adversarial image pairs and 9k prompts across 16 scientific domains and an isolated test set of 454 challenging prompts. Using this benchmark, we evaluate 18 recent image generation models and find that none scores above 50 out of 100 under implicit scientific prompts, while explicit prompts that directly describe the intended outcome yield scores roughly 35 points higher, confirming that current models can render correct scenes when told what to depict but cannot reason from scientific cues to the correct visual outcome. To address this, we develop SciScore, a reward model fine-tuned from CLIP-H that captures fine-grained scientific phenomena without relying on language-guided inference, surpassing GPT-4o and experienced human evaluators by roughly 5 points. We further propose a two-stage alignment framework combining supervised fine-tuning with masked online fine-tuning to inject scientific knowledge into generative models. Applying this framework to FLUX.1[dev] yields a relative improvement exceeding 50% on SciScore, demonstrating that scientific reasoning in image generation can be substantially improved through targeted data and alignment.
>
---
#### [replaced 038] Q-DiT4SR: Exploration of Detail-Preserving Diffusion Transformer Quantization for Real-World Image Super-Resolution
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.01273](https://arxiv.org/pdf/2602.01273)**

> **作者:** Xun Zhang; Kaicheng Yang; Hongliang Lu; Haotong Qin; Yong Guo; Yulun Zhang
>
> **备注:** Our code and models will be available at this https URL
>
> **摘要:** Recently, Diffusion Transformers (DiTs) have emerged in Real-World Image Super-Resolution (Real-ISR) to generate high-quality textures, yet their heavy inference burden hinders real-world deployment. While Post-Training Quantization (PTQ) is a promising solution for acceleration, existing methods in super-resolution mostly focus on U-Net architectures, whereas generic DiT quantization is typically designed for text-to-image tasks. Directly applying these methods to DiT-based super-resolution models leads to severe degradation of local textures. Therefore, we propose Q-DiT4SR, the first PTQ framework specifically tailored for DiT-based Real-ISR. We propose H-SVD, a hierarchical SVD that integrates a global low-rank branch with a local block-wise rank-1 branch under a matched parameter budget. We further propose Variance-aware Spatio-Temporal Mixed Precision: VaSMP allocates cross-layer weight bit-widths in a data-free manner based on rate-distortion theory, while VaTMP schedules intra-layer activation precision across diffusion timesteps via dynamic programming (DP) with minimal calibration. Experiments on multiple real-world datasets demonstrate that our Q-DiT4SR achieves SOTA performance under both W4A6 and W4A4 settings. Notably, the W4A4 quantization configuration reduces model size by 5.8$\times$ and computational operations by 6.14$\times$. Our code and models will be available at this https URL.
>
---
#### [replaced 039] Visual Neural Decoding via Improved Visual-EEG Semantic Consistency
- **分类: cs.CV; cs.HC**

- **链接: [https://arxiv.org/pdf/2408.06788](https://arxiv.org/pdf/2408.06788)**

> **作者:** Hongzhou Chen; Lianghua He; Yihang Liu; Longzhen Yang; Shaohua Shang; MengChu Zhou
>
> **摘要:** Visual neural decoding aims to extract and interpret original visual experiences directly from human brain activity. Recent studies have demonstrated the feasibility of decoding visual semantic categories from electroencephalography (EEG) signals, among which metric learning-based approaches have delivered promising results. However, these methods that directly map EEG features into a pre-trained embedding space inevitably introduce mapping bias, resulting in a modality gap and semantic inconsistency that impair cross-modal alignment. To address these issues, this work constructs a Visual-EEG Joint Semantic Space to bridge the gap between visual images and neural signals. Building upon this space, we propose two novel approaches to improve semantic consistency between cross-modal representations and facilitate optimal alignment. Specifically, (1) we introduce a Visual-EEG Semantic Decoupling Network (VE-SDN) to explicitly disentangle semantic components from modality representations, thereby achieving purely semantic-level cross-modal alignment. (2) We introduce a Neural-Guided Intra-Class Consistency (NGIC) objective, an asymmetric representation alignment strategy designed to effectively enhance the robustness of visual representations and further boost decoding performance. Extensive experiments on a large-scale Visual-EEG dataset validate the effectiveness of the proposed method. Compared to the strongest baseline, our approach demonstrates superior decoding performance, yielding relative Top-1/Top-5 accuracy improvements of 38.9%/17.9% in intra-subject and 16.1%/11.3% in inter-subject settings. The code is available at this https URL
>
---
#### [replaced 040] Variance-Based Pruning for Accelerating and Compressing Trained Networks
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2507.12988](https://arxiv.org/pdf/2507.12988)**

> **作者:** Uranik Berisha; Jens Mehnert; Alexandru Paul Condurache
>
> **备注:** Accepted as Oral at ICCV'25 (IEEE/CVF International Conference on Computer Vision)
>
> **摘要:** Increasingly expensive training of ever larger models such as Vision Transfomers motivate reusing the vast library of already trained state-of-the-art networks. However, their latency, high computational costs and memory demands pose significant challenges for deployment, especially on resource-constrained hardware. While structured pruning methods can reduce these factors, they often require costly retraining, sometimes for up to hundreds of epochs, or even training from scratch to recover the lost accuracy resulting from the structural modifications. Maintaining the provided performance of trained models after structured pruning and thereby avoiding extensive retraining remains a challenge. To solve this, we introduce Variance-Based Pruning, a simple and structured one-shot pruning technique for efficiently compressing networks, with minimal finetuning. Our approach first gathers activation statistics, which are used to select neurons for pruning. Simultaneously the mean activations are integrated back into the model to preserve a high degree of performance. On ImageNet-1k recognition tasks, we demonstrate that directly after pruning DeiT-Base retains over 70% of its original performance and requires only 10 epochs of fine-tuning to regain 99% of the original accuracy while simultaneously reducing MACs by 35% and model size by 36%, thus speeding up the model by 1.44x. The code is available at: this https URL
>
---
#### [replaced 041] EagleNet: Energy-Aware Fine-Grained Relationship Learning Network for Text-Video Retrieval
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.25267](https://arxiv.org/pdf/2603.25267)**

> **作者:** Yuhan Chen; Pengwen Dai; Chuan Wang; Dayan Wu; Xiaochun Cao
>
> **备注:** Accepted at CVPR 2026
>
> **摘要:** Text-video retrieval tasks have seen significant improvements due to the recent development of large-scale vision-language pre-trained models. Traditional methods primarily focus on video representations or cross-modal alignment, while recent works shift toward enriching text expressiveness to better match the rich semantics in videos. However, these methods use only interactions between text and frames/video, and ignore rich interactions among the internal frames within a video, so the final expanded text cannot capture frame contextual information, leading to disparities between text and video. In response, we introduce Energy-Aware Fine-Grained Relationship Learning Network (EagleNet) to generate accurate and context-aware enriched text embeddings. Specifically, the proposed Fine-Grained Relationship Learning mechanism (FRL) first constructs a text-frame graph by the generated text candidates and frames, then learns relationships among texts and frames, which are finally used to aggregate text candidates into an enriched text embedding that incorporates frame contextual information. To further improve fine-grained relationship learning in FRL, we design Energy-Aware Matching (EAM) to model the energy of text-frame interactions and thus accurately capture the distribution of real text-video pairs. Moreover, for more effective cross-modal alignment and stable training, we replace the conventional softmax-based contrastive loss with the sigmoid loss. Extensive experiments have demonstrated the superiority of EagleNet across MSRVTT, DiDeMo, MSVD, and VATEX. Codes are available at this https URL.
>
---
#### [replaced 042] HUMOF: Human Motion Forecasting in Interactive Social Scenes
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.03753](https://arxiv.org/pdf/2506.03753)**

> **作者:** Caiyi Sun; Yujing Sun; Xiao Han; Zemin Yang; Jiawei Liu; Xinge Zhu; Siu Ming Yiu; Yuexin Ma
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Complex scenes present significant challenges for predicting human behaviour due to the abundance of interaction information, such as human-human and humanenvironment interactions. These factors complicate the analysis and understanding of human behaviour, thereby increasing the uncertainty in forecasting human motions. Existing motion prediction methods thus struggle in these complex scenarios. In this paper, we propose an effective method for human motion forecasting in interactive scenes. To achieve a comprehensive representation of interactions, we design a hierarchical interaction feature representation so that high-level features capture the overall context of the interactions, while low-level features focus on fine-grained details. Besides, we propose a coarse-to-fine interaction reasoning module that leverages both spatial and frequency perspectives to efficiently utilize hierarchical features, thereby enhancing the accuracy of motion predictions. Our method achieves state-of-the-art performance across four public datasets. The source code will be available at this https URL.
>
---
#### [replaced 043] Are Large Vision-Language Models Ready to Guide Blind and Low-Vision Individuals?
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.00766](https://arxiv.org/pdf/2510.00766)**

> **作者:** Eunki Kim; Na Min An; Wan Ju Kang; Sangryul Kim; James Thorne; Hyunjung Shim
>
> **备注:** 42 pages, 14 figures, 28 tables
>
> **摘要:** Large Vision-Language Models (LVLMs) demonstrate a promising direction for assisting individuals with blindness or low-vision (BLV). Yet, measuring their true utility in real-world scenarios is challenging because evaluating whether their descriptions are BLV-informative requires a fundamentally different approach from assessing standard scene descriptions. While the "VLM-as-a-metric" or "LVLM-as-a-judge" paradigm has emerged, existing evaluators still fall short of capturing the unique requirements of BLV-centric evaluation, lacking at least one of the following key properties: (1) High correlation with human judgments, (2) Long instruction understanding, (3) Score generation efficiency, and (4) Multi-dimensional assessment. To this end, we propose a unified framework to bridge the gap between automated evaluation and actual BLV needs. First, we conduct an in-depth user study with BLV participants to understand and quantify their navigational preferences, curating VL-GUIDEDATA, a large-scale BLV user-simulated preference dataset containing image-request-response-score pairs. We then leverage the dataset to develop an accessibility-aware evaluator, VL-GUIDE-S, which outperforms existing (L)VLM judges in both human alignment and inference efficiency. Notably, its effectiveness extends beyond a single domain, demonstrating strong performance across multiple fine-grained, BLV-critical dimensions. We hope our work lays as a foundation for automatic AI judges that advance safe, barrier-free navigation for BLV users.
>
---
#### [replaced 044] A 3D Cross-modal Keypoint Descriptor for MR-US Matching and Registration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.18551](https://arxiv.org/pdf/2507.18551)**

> **作者:** Daniil Morozov; Reuben Dorent; Nazim Haouchine
>
> **备注:** Accepted in IEEE Transactions on Medical Imaging
>
> **摘要:** Intraoperative registration of real-time ultrasound (iUS) to preoperative Magnetic Resonance Imaging (MRI) remains an unsolved problem due to severe modality-specific differences in appearance, resolution, and field-of-view. To address this, we propose a novel 3D cross-modal keypoint descriptor for MRI-iUS matching and registration. Our approach employs a patient-specific matching-by-synthesis approach, generating synthetic iUS volumes from preoperative MRI. This enables supervised contrastive training to learn a shared descriptor space. A probabilistic keypoint detection strategy is then employed to identify anatomically salient and modality-consistent locations. During training, a curriculum-based triplet loss with dynamic hard negative mining is used to learn descriptors that are i) robust to iUS artifacts such as speckle noise and limited coverage, and ii) rotation-invariant. At inference, the method detects keypoints in MR and real iUS images and identifies sparse matches, which are then used to perform rigid registration. Our approach is evaluated using 3D MRI-iUS pairs from the ReMIND dataset. Experiments show that our approach outperforms state-of-the-art keypoint matching methods across 11 patients, with an average precision of 69.8%. For image registration, our method achieves a competitive mean Target Registration Error of 2.39 mm on the ReMIND2Reg benchmark. Compared to existing iUS-MR registration approaches, our framework is interpretable, requires no manual initialization, and shows robustness to iUS field-of-view variation. Code, data and model weights are available at this https URL.
>
---
#### [replaced 045] WAON: Large-Scale Japanese Image-Text Pair Dataset for Improving Model Performance on Japanese Cultural Tasks
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出WAON数据集，解决日本文化任务中高质量图像-文本对数据不足的问题。通过构建大规模、高质量的数据集及评估基准，提升模型在日语文化任务上的性能。**

- **链接: [https://arxiv.org/pdf/2510.22276](https://arxiv.org/pdf/2510.22276)**

> **作者:** Issa Sugiura; Shuhei Kurita; Yusuke Oda; Daisuke Kawahara; Yasuo Okabe; Naoaki Okazaki
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** Contrastive pre-training on large-scale image-text pair datasets has driven major advances in vision-language representation learning. Recent work shows that pretraining on global data followed by language or culture specific fine-tuning is effective for improving performance in target domains. With the availability of strong open-weight multilingual models such as SigLIP2, this paradigm has become increasingly practical. However, for Japanese, the scarcity of large-scale, high-quality image-text pair datasets tailored to Japanese language and cultural content remains a key limitation. To address this gap, we introduce WAON, the largest Japanese image-text pair dataset constructed from Japanese web content in Common Crawl, containing approximately 155 million examples. Our dataset construction pipeline employs filtering and deduplication to improve dataset quality. To improve the quality and reliability of evaluation on Japanese cultural tasks, we also construct WAON-Bench, a manually curated benchmark for Japanese cultural image classification comprising 374 classes, which addresses issues in the existing benchmark such as category imbalance and label-image mismatches. Our experiments demonstrate that fine-tuning on WAON improves model performance on Japanese cultural benchmarks more efficiently than existing datasets, achieving state-of-the-art results among publicly available models of comparable architecture. We release our dataset, model, and code.
>
---
#### [replaced 046] SurgTEMP: Temporal-Aware Surgical Video Question Answering with Text-guided Visual Memory for Laparoscopic Cholecystectomy
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.29962](https://arxiv.org/pdf/2603.29962)**

> **作者:** Shi Li; Vinkle Srivastav; Nicolas Chanel; Saurav Sharma; Nabani Banik; Lorenzo Arboit; Kun Yuan; Pietro Mascagni; Nicolas Padoy
>
> **备注:** 29 pages, 14 figures, 9 tables
>
> **摘要:** Surgical procedures are inherently complex and risky, requiring extensive expertise and constant focus to well navigate evolving intraoperative scenes. Computer-assisted systems such as surgical visual question answering (VQA) offer promises for education and intraoperative support. Current surgical VQA research largely focuses on static frame analysis, overlooking rich temporal semantics. Surgical video question answering is further challenged by low visual contrast, its highly knowledge-driven nature, diverse analytical needs spanning scattered temporal windows, and the hierarchy from basic perception to high-level intraoperative assessment. To address these challenges, we propose SurgTEMP, a multimodal LLM framework featuring (i) a query-guided token selection module that builds hierarchical visual memory (spatial and temporal memory banks) and (ii) a Surgical Competency Progression (SCP) training scheme. Together, these components enable effective modeling of variable-length surgical videos while preserving procedure-relevant cues and temporal coherence, and better support diverse downstream assessment tasks. To support model development, we introduce CholeVidQA-32K, a surgical video question answering dataset comprising 32K open-ended QA pairs and 3,855 video segments (approximately 128 h total) from laparoscopic cholecystectomy. The dataset is organized into a three-level hierarchy -- Perception, Assessment, and Reasoning -- spanning 11 tasks from instrument/action/anatomy perception to Critical View of Safety (CVS), intraoperative difficulty, skill proficiency, and adverse event assessment. In comprehensive evaluations against state-of-the-art open-source multimodal and video LLMs (fine-tuned and zero-shot), SurgTEMP achieves substantial performance improvements, advancing the state of video-based surgical VQA.
>
---
#### [replaced 047] MemeMind: A Large-Scale Multimodal Dataset with Chain-of-Thought Reasoning for Harmful Meme Detection
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于有害表情包检测任务，旨在解决隐含有害内容识别困难的问题。构建了MemeMind数据集并提出MemeGuard框架，提升检测准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2506.18919](https://arxiv.org/pdf/2506.18919)**

> **作者:** Hexiang Gu; Qifan Yu; Yuan Liu; Zikang Li; Saihui Hou; Jian Zhao; Zhaofeng He
>
> **摘要:** As a multimodal medium combining images and text, memes frequently convey implicit harmful content through metaphors and humor, rendering the detection of harmful memes a complex and challenging task. Although recent studies have made progress in detection accuracy and interpretability, large-scale, high-quality datasets for harmful memes remain scarce, and current methods still struggle to capture implicit risks and nuanced semantics. Thus, we construct MemeMind, a large-scale harmful meme dataset. Aligned with the international standards and the context of internet, MemeMind provides detailed Chain-of-Thought (CoT) reasoning annotations to support fine-grained analysis of implicit intentions in memes. Based on this dataset, we further propose MemeGuard, a reasoning-oriented multimodal detection framework that significantly improves both the accuracy of harmful meme detection and the interpretability of model decisions. Extensive experimental results demonstrate that MemeGuard outperforms existing state-of-the-art methods on the MemeMind dataset, establishing a solid foundation for future research in harmful meme detection. The complete dataset and code will be released upon acceptance.
>
---
#### [replaced 048] Equilibrium contrastive learning for imbalanced image classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2602.09506](https://arxiv.org/pdf/2602.09506)**

> **作者:** Sumin Roh; Harim Kim; Ho Yun Lee; Il Yong Chun
>
> **备注:** 18 pages, 8 figures
>
> **摘要:** Contrastive learning (CL) is a predominant technique in image classification, but they showed limited performance with an imbalanced dataset. Recently, several supervised CL methods have been proposed to promote an ideal regular simplex geometric configuration in the representation space-characterized by intra-class feature collapse and uniform inter-class mean spacing, especially for imbalanced datasets. In particular, existing prototype-based methods include class prototypes, as additional samples to consider all classes. However, the existing CL methods suffer from two limitations. First, they do not consider the alignment between the class means/prototypes and classifiers, which could lead to poor generalization. Second, existing prototype-based methods treat prototypes as only one additional sample per class, making their influence depend on the number of class instances in a batch and causing unbalanced contributions across classes. To address these limitations, we propose Equilibrium Contrastive Learning (ECL), a supervised CL framework designed to promote geometric equilibrium, where class features, means, and classifiers are harmoniously balanced under data imbalance. The proposed ECL framework uses two main components. First, ECL promotes the representation geometric equilibrium (i.e., a regular simplex geometry characterized by collapsed class samples and uniformly distributed class means), while balancing the contributions of class-average features and class prototypes. Second, ECL establishes a classifier-class center geometric equilibrium by aligning classifier weights and class prototypes. We ran experiments with three long-tailed datasets, the CIFAR-10(0)-LT, ImageNet-LT, and the two imbalanced medical datasets, the ISIC 2019 and our constructed LCCT dataset. Results show that ECL outperforms existing SOTA supervised CL methods designed for imbalanced classification.
>
---
#### [replaced 049] Attention-guided reference point shifting for Gaussian-mixture-based partial point set registration
- **分类: cs.CV; cs.GR**

- **链接: [https://arxiv.org/pdf/2512.02496](https://arxiv.org/pdf/2512.02496)**

> **作者:** Mizuki Kikkawa; Tatsuya Yatagawa; Yutaka Ohtake; Hiromasa Suzuki
>
> **备注:** 16 pages, 9 figures, 7 tables
>
> **摘要:** This study investigates the impact of the invariance of feature vectors for partial-to-partial point set registration under translation and rotation of input point sets, particularly in the realm of techniques based on deep learning and Gaussian mixture models (GMMs). We reveal both theoretical and practical problems associated with such deep-learning-based registration methods using GMMs, with a particular focus on the limitations of DeepGMR, a pioneering study in this line, to the partial-to-partial point set registration. Our primary goal is to uncover the causes behind such methods and propose a comprehensible solution for that. To address this, we introduce an attention-based reference point shifting (ARPS) layer, which robustly identifies a common reference point of two partial point sets, thereby acquiring transformation-invariant features. The ARPS layer employs a well-studied attention module to find a common reference point rather than the overlap region. Owing to this, it significantly enhances the performance of DeepGMR and its recent variant, UGMMReg. Furthermore, these extension models outperform even prior deep learning methods using attention blocks and Transformer to extract the overlap region or common reference points. We believe these findings provide deeper insights into registration methods using deep learning and GMMs.
>
---
#### [replaced 050] Missing No More: Dictionary-Guided Cross-Modal Image Fusion under Missing Infrared
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.08018](https://arxiv.org/pdf/2603.08018)**

> **作者:** Yafei Zhang; Meng Ma; Huafeng Li; Yu Liu
>
> **备注:** This paper has been accepted by CVPR 2026
>
> **摘要:** Infrared-visible (IR-VIS) image fusion is vital for perception and security, yet most methods rely on the availability of both modalities during training and inference. When the infrared modality is absent, pixel-space generative substitutes become hard to control and inherently lack interpretability. We address missing-IR fusion by proposing a dictionary-guided, coefficient-domain framework built upon a shared convolutional dictionary. The pipeline comprises three key components: (1) Joint Shared-dictionary Representation Learning (JSRL) learns a unified and interpretable atom space shared by both IR and VIS modalities; (2) VIS-Guided IR Inference (VGII) transfers VIS coefficients to pseudo-IR coefficients in the coefficient domain and performs a one-step closed-loop refinement guided by a frozen large language model as a weak semantic prior; and (3) Adaptive Fusion via Representation Inference (AFRI) merges VIS structures and inferred IR cues at the atom level through window attention and convolutional mixing, followed by reconstruction with the shared dictionary. This encode-transfer-fuse-reconstruct pipeline avoids uncontrolled pixel-space generation while ensuring prior preservation within interpretable dictionary-coefficient representation. Experiments under missing-IR settings demonstrate consistent improvements in perceptual quality and downstream detection performance. To our knowledge, this represents the first framework that jointly learns a shared dictionary and performs coefficient-domain inference-fusion to tackle missing-IR fusion. The source code is publicly available at this https URL.
>
---
#### [replaced 051] EvalBlocks: A Modular Pipeline for Rapidly Evaluating Foundation Models in Medical Imaging
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2601.03811](https://arxiv.org/pdf/2601.03811)**

> **作者:** Jan Tagscherer; Sarah de Boer; Lena Philipp; Fennie van der Graaf; Dré Peeters; Joeran Bosma; Lars Leijten; Bogdan Obreja; Ewoud Smit; Alessa Hering
>
> **备注:** Accepted and published in BVM 2026 proceedings (Springer)
>
> **摘要:** Developing foundation models in medical imaging requires continuous monitoring of downstream performance. Researchers are burdened with tracking numerous experiments, design choices, and their effects on performance, often relying on ad-hoc, manual workflows that are inherently slow and error-prone. We introduce EvalBlocks, a modular, plug-and-play framework for efficient evaluation of foundation models during development. Built on Snakemake, EvalBlocks supports seamless integration of new datasets, foundation models, aggregation methods, and evaluation strategies. All experiments and results are tracked centrally and are reproducible with a single command, while efficient caching and parallel execution enable scalable use on shared compute infrastructure. Demonstrated on five state-of-the-art foundation models and three medical imaging classification tasks, EvalBlocks streamlines model evaluation, enabling researchers to iterate faster and focus on model innovation rather than evaluation logistics. The framework is released as open source software at this https URL.
>
---
#### [replaced 052] CoCoDiff: Correspondence-Consistent Diffusion Model for Fine-grained Style Transfer
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2602.14464](https://arxiv.org/pdf/2602.14464)**

> **作者:** Wenbo Nie; Zixiang Li; Renshuai Tao; Bin Wu; Yunchao Wei; Yao Zhao
>
> **摘要:** Transferring visual style between images while preserving semantic correspondence between similar objects remains a central challenge in computer vision. While existing methods have made great strides, most of them operate at global level but overlook region-wise and even pixel-wise semantic correspondence. To address this, we propose CoCoDiff, a novel training-free and low-cost style transfer framework that leverages pretrained latent diffusion models to achieve fine-grained, semantically consistent stylization. We identify that correspondence cues within generative diffusion models are under-explored and that content consistency across semantically matched regions is often neglected. CoCoDiff introduces a pixel-wise semantic correspondence module that mines intermediate diffusion features to construct a dense alignment map between content and style images. Furthermore, a cycle-consistency module then enforces structural and perceptual alignment across iterations, yielding object and region level stylization that preserves geometry and detail. Despite requiring no additional training or supervision, CoCoDiff delivers state-of-the-art visual quality and strong quantitative results, outperforming methods that rely on extra training or annotations.
>
---
#### [replaced 053] Moving Light Adaptive Colonoscopy Reconstruction via Illumination-Attenuation-Aware 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.18739](https://arxiv.org/pdf/2510.18739)**

> **作者:** Hao Wang; Ying Zhou; Haoyu Zhao; Rui Wang; Qiang Hu; Xing Zhang; Qiang Li; Zhiwei Wang
>
> **备注:** Accepted by ICME2026
>
> **摘要:** 3D Gaussian Splatting (3DGS) enables real-time view synthesis in colonoscopy but assumes static illumination, making it incompatible with the strong photometric variations caused by the moving light source and camera. This mismatch leads existing methods to compensate for illumination attenuation with structure-violating Gaussians, degrading geometric fidelity. Prior work considers only distance-based attenuation and overlooks the physical characteristics of colonscopic lighting. In this paper, we propose ColIAGS, an improved 3DGS framework for colonoscopy. To mimic realistic appearance under varying illumination, we introduce a lighting model with two types of illumination attenuation factors. To satisfy this lighting model's approximation and effectively integrate it into the 3DGS framework, we design Improved Geometry Modeling to strengthen geometry details and Improved Appearance Modeling to implicitly optimize illumination attenuation solutions. Experimental results on standard benchmarks demonstrate that ColIAGS supports both high-quality novel-view synthesis and accurate geometry reconstruction, outperforming state-of-the-art methods in rendering fidelity and Depth MSE. Our code is available at this https URL.
>
---
#### [replaced 054] Lossy Common Information in a Learnable Gray-Wyner Network
- **分类: cs.LG; cs.CV; cs.IT**

- **链接: [https://arxiv.org/pdf/2601.21424](https://arxiv.org/pdf/2601.21424)**

> **作者:** Anderson de Andrade; Alon Harell; Ivan V. Bajić
>
> **摘要:** Many computer vision tasks share substantial overlapping information, yet conventional codecs tend to ignore this, leading to redundant and inefficient representations. The Gray-Wyner network, a classical concept from information theory, offers a principled framework for separating common and task-specific information. Inspired by this idea, we develop a learnable three-channel codec that disentangles shared information from task-specific details across multiple vision tasks. We characterize the limits of this approach through the notion of lossy common information, and propose an optimization objective that balances inherent tradeoffs in learning such representations. Through comparisons of three codec architectures on two-task scenarios spanning six vision benchmarks, we demonstrate that our approach substantially reduces redundancy and consistently outperforms independent coding. These results highlight the practical value of revisiting Gray-Wyner theory in modern machine learning contexts, bridging classic information theory with task-driven representation learning.
>
---
#### [replaced 055] How Blind and Low-Vision Individuals Prefer Large Vision-Language Model-Generated Scene Descriptions
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2502.14883](https://arxiv.org/pdf/2502.14883)**

> **作者:** Na Min An; Eunki Kim; Wan Ju Kang; Sangryul Kim; James Thorne; Hyunjung Shim
>
> **备注:** This paper has been superseded by version 2 of arXiv:2510.00766
>
> **摘要:** For individuals with blindness or low vision (BLV), navigating complex environments can pose serious risks. Large Vision-Language Models (LVLMs) show promise for generating scene descriptions, but their effectiveness for BLV users remains underexplored. To address this gap, we conducted a user study with eight BLV participants to systematically evaluate preferences for six types of LVLM descriptions. While they helped to reduce fear and improve actionability, user ratings showed wide variation in sufficiency and conciseness. Furthermore, GPT-4o--despite its strong potential to refine descriptions--was not consistently preferred by participants. We use the insights obtained from the user study to build training data for building our new automatic evaluation metric that can capture BLV preferences effectively. Our findings underscore the urgent need for BLV-centered evaluation metrics and human-in-the-loop feedback to advance LVLM description quality for accessibility.
>
---
#### [replaced 056] FedKLPR: KL-Guided Pruning-Aware Federated Learning for Person Re-Identification
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2508.17431](https://arxiv.org/pdf/2508.17431)**

> **作者:** Po-Hsien Yu; Yu-Syuan Tseng; Shao-Yi Chien
>
> **备注:** 13 pages, 3 figures, submitted to IEEE Transactions on Circuits and Systems for Video Technology
>
> **摘要:** Person re-identification (re-ID) is a fundamental task in intelligent surveillance and public safety. Federated learning (FL) provides a privacy-preserving paradigm by enabling collaborative model training without centralized data collection. However, applying FL to real-world re-ID systems remains challenging due to two major issues: statistical heterogeneity across clients caused by non-IID data distributions and substantial communication overhead resulting from the frequent transmission of large-scale models. To address these challenges, we propose FedKLPR, a lightweight and communication-efficient federated learning framework for person re-ID. FedKLPR consists of three key components. First, the KL-Divergence Regularization Loss (KLL) constrains local updates by reducing the discrepancy between local and global feature distributions, thereby alleviating the effects of statistical heterogeneity and improving convergence stability under non-IID settings. Second, KL-Divergence-Prune Weighted Aggregation (KLPWA) incorporates both pruning ratio and distributional similarity into the aggregation process, enabling more effective aggregation of pruned local models under non-IID data distributions and enhancing the robustness of the global model. Third, Cross-Round Recovery (CRR) employs a dynamic pruning control mechanism to prevent excessive pruning and preserve model accuracy during iterative compression. Experimental results on eight benchmark datasets demonstrate that FedKLPR achieves substantial communication savings while maintaining competitive accuracy. Compared with state-of-the-art methods, FedKLPR reduces communication cost by 40\%--42\% on ResNet-50 while achieving superior overall performance.
>
---
#### [replaced 057] MOLM: Mixture of LoRA Markers
- **分类: cs.CV; cs.CR; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.00293](https://arxiv.org/pdf/2510.00293)**

> **作者:** Samar Fares; Nurbek Tastan; Noor Hussein; Karthik Nandakumar
>
> **备注:** ICLR 2026
>
> **摘要:** Generative models can generate photorealistic images at scale. This raises urgent concerns about the ability to detect synthetically generated images and attribute these images to specific sources. While watermarking has emerged as a possible solution, existing methods remain fragile to realistic distortions, susceptible to adaptive removal, and expensive to update when the underlying watermarking key changes. We propose a general watermarking framework that formulates the encoding problem as key-dependent perturbation of the parameters of a generative model. Within this framework, we introduce Mixture of LoRA Markers (MOLM), a routing-based instantiation in which binary keys activate lightweight LoRA adapters inside residual and attention blocks. This design avoids key-specific re-training and achieves the desired properties such as imperceptibility, fidelity, verifiability, and robustness. Experiments on Stable Diffusion and FLUX show that MOLM preserves image quality while achieving robust key recovery against distortions, compression and regeneration, averaging attacks, and black-box adversarial attacks on the extractor.
>
---
#### [replaced 058] Coupled Reconstruction of 2D Blood Flow and Vessel Geometry from Noisy Images via Physics-Informed Neural Networks and Quasi-Conformal Mapping
- **分类: math.NA; cs.CV**

- **链接: [https://arxiv.org/pdf/2508.11216](https://arxiv.org/pdf/2508.11216)**

> **作者:** Han Zhang; Xue-Cheng Tai; Jean-Michel Morel; Raymond H. Chan
>
> **摘要:** Blood flow imaging provides important information for hemodynamic behavior within the vascular system and plays an essential role in medical diagnosis and treatment planning. However, obtaining high-quality flow images remains a significant challenge. In this work, we address the problem of denoising flow images that may suffer from artifacts due to short acquisition times or device-induced errors. We formulate this task as an optimization problem, where the objective is to minimize the discrepancy between the modeled velocity field, constrained to satisfy the Navier-Stokes equations, and the observed noisy velocity data. To solve this problem, we decompose it into two subproblems: a fluid subproblem and a geometry subproblem. The fluid subproblem leverages a Physics-Informed Neural Network to reconstruct the velocity field from noisy observations, assuming a fixed domain. The geometry subproblem aims to infer the underlying flow region by optimizing a quasi-conformal mapping that deforms a reference domain. These two subproblems are solved in an alternating Gauss-Seidel fashion, iteratively refining both the velocity field and the domain. Upon convergence, the framework yields a high-quality reconstruction of the flow image. We validate the proposed method through experiments on synthetic flow data in a converging channel geometry under varying levels of Gaussian noise, and on real-like flow data in an aortic geometry with signal-dependent noise. The results demonstrate the effectiveness and robustness of the approach. Additionally, ablation studies are conducted to assess the influence of key hyperparameters.
>
---
#### [replaced 059] SA-CycleGAN-2.5D: Self-Attention CycleGAN with Tri-Planar Context for Multi-Site MRI Harmonization
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2603.17219](https://arxiv.org/pdf/2603.17219)**

> **作者:** Ishrith Gowda; Chunwei Liu
>
> **备注:** 12 pages, 5 figures, 5 tables. Submitted to MICCAI 2026
>
> **摘要:** Multi-site neuroimaging analysis is fundamentally confounded by scanner-induced covariate shifts, where the marginal distribution of voxel intensities $P(\mathbf{x})$ varies non-linearly across acquisition protocols while the conditional anatomy $P(\mathbf{y}|\mathbf{x})$ remains constant. This is particularly detrimental to radiomic reproducibility, where acquisition variance often exceeds biological pathology variance. Existing statistical harmonization methods (e.g., ComBat) operate in feature space, precluding spatial downstream tasks, while standard deep learning approaches are theoretically bounded by local effective receptive fields (ERF), failing to model the global intensity correlations characteristic of field-strength bias. We propose SA-CycleGAN-2.5D, a domain adaptation framework motivated by the $H\Delta H$-divergence bound of Ben-David et al., integrating three architectural innovations: (1) A 2.5D tri-planar manifold injection preserving through-plane gradients $\nabla_z$ at $O(HW)$ complexity; (2) A U-ResNet generator with dense voxel-to-voxel self-attention, surpassing the $O(\sqrt{L})$ receptive field limit of CNNs to model global scanner field biases; and (3) A spectrally-normalized discriminator constraining the Lipschitz constant ($K_D \le 1$) for stable adversarial optimization. Evaluated on 654 glioma patients across two institutional domains (BraTS and UPenn-GBM), our method reduces Maximum Mean Discrepancy (MMD) by 99.1% ($1.729 \to 0.015$) and degrades domain classifier accuracy to near-chance (59.7%). Ablation confirms that global attention is statistically essential (Cohen's $d = 1.32$, $p < 0.001$) for the harder heterogeneous-to-homogeneous translation direction. By bridging 2D efficiency and 3D consistency, our framework yields voxel-level harmonized images that preserve tumor pathophysiology, enabling reproducible multi-center radiomic analysis.
>
---
#### [replaced 060] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.06328](https://arxiv.org/pdf/2511.06328)**

> **作者:** Dingkang Yang; Mingcheng Li; Xuecheng Wu; Zhaoyu Chen; Kaixun Jiang; Keliang Liu; Peng Zhai; Lihua Zhang
>
> **摘要:** Multimodal Sentiment Analysis (MSA) aims to predict sentiment from language, acoustic, and visual data in videos. However, imbalanced unimodal performance often leads to suboptimal fused representations. Existing approaches typically adopt fixed primary modality strategies to maximize dominant modality advantages, yet fail to adapt to dynamic variations in modality importance across different samples. Moreover, non-language modalities suffer from sequential redundancy and noise, degrading model performance when they serve as primary inputs. To address these issues, this paper proposes a modality optimization and dynamic primary modality selection framework (MODS). First, a Graph-based Dynamic Sequence Compressor (GDC) is constructed, which employs capsule networks and graph convolution to reduce sequential redundancy in acoustic/visual modalities. Then, we develop a sample-adaptive Primary Modality Selector (MSelector) for dynamic dominance determination. Finally, a Primary-modality-Centric Cross-Attention (PCCA) module is designed to enhance dominant modalities while facilitating cross-modal interaction. Extensive experiments on four benchmark datasets demonstrate that MODS outperforms state-of-the-art methods, achieving superior performance by effectively balancing modality contributions and eliminating redundant noise.
>
---
#### [replaced 061] CLoD-GS: Continuous Level-of-Detail via 3D Gaussian Splatting
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.09997](https://arxiv.org/pdf/2510.09997)**

> **作者:** Zhigang Cheng; Mingchao Sun; Yu Liu; Zengye Ge; Luyang Tang; Mu Xu; Yangyan Li; Peng Pan
>
> **备注:** Accepted by ICLR 2026 poster
>
> **摘要:** Level of Detail (LoD) is a fundamental technique in real-time computer graphics for managing the rendering costs of complex scenes while preserving visual fidelity. Traditionally, LoD is implemented using discrete levels (DLoD), where multiple, distinct versions of a model are swapped out at different distances. This long-standing paradigm, however, suffers from two major drawbacks: it requires significant storage for multiple model copies and causes jarring visual ``popping" artifacts during transitions, degrading the user experience. We argue that the explicit, primitive-based nature of the emerging 3D Gaussian Splatting (3DGS) technique enables a more ideal paradigm: Continuous LoD (CLoD). A CLoD approach facilitates smooth, seamless quality scaling within a single, unified model, thereby circumventing the core problems of DLOD. To this end, we introduce CLoD-GS, a framework that integrates a continuous LoD mechanism directly into a 3DGS representation. Our method introduces a learnable, distance-dependent decay parameter for each Gaussian primitive, which dynamically adjusts its opacity based on viewpoint proximity. This allows for the progressive and smooth filtering of less significant primitives, effectively creating a continuous spectrum of detail within one model. To train this model to be robust across all distances, we introduce a virtual distance scaling mechanism and a novel coarse-to-fine training strategy with rendered point count regularization. Our approach not only eliminates the storage overhead and visual artifacts of discrete methods but also reduces the primitive count and memory footprint of the final model. Extensive experiments demonstrate that CLoD-GS achieves smooth, quality-scalable rendering from a single model, delivering high-fidelity results across a wide range of performance targets.
>
---
#### [replaced 062] Vision-Language-Model-Guided Differentiable Ray Tracing for Fast and Accurate Multi-Material RF Parameter Estimation
- **分类: cs.CV; cs.NI**

- **链接: [https://arxiv.org/pdf/2601.18242](https://arxiv.org/pdf/2601.18242)**

> **作者:** Zerui Kang; Yishen Lim; Zhouyou Gu; Seung-Woo Ko; Tony Q.S. Quek; Jihong Park
>
> **摘要:** Accurate radio-frequency (RF) material parameters are essential for electromagnetic digital twins in 6G systems, yet gradient-based inverse ray tracing (RT) remains sensitive to initialization and costly under limited measurements. This paper proposes a vision-language-model (VLM) guided framework that accelerates and stabilizes multi-material parameter estimation in a differentiable RT (DRT) engine. A VLM parses scene images to infer material categories and maps them to quantitative priors via an ITU-R material table, yielding informed conductivity initializations. The VLM further selects informative transmitter/receiver placements that promote diverse, material-discriminative paths. Starting from these priors, the DRT performs gradient-based refinement using measured received signal strengths. Experiments in NVIDIA Sionna on indoor scenes show 2-4$\times$ faster convergence and 10-100$\times$ lower final parameter error compared with uniform or random initialization and random placement baselines, achieving sub-0.1\% mean relative error with only a few receivers. Complexity analyses indicate per-iteration time scales near-linearly with the number of materials and measurement setups, while VLM-guided placement reduces the measurements required for accurate recovery. Ablations over RT depth and ray counts confirm further accuracy gains without significant per-iteration overhead. Results demonstrate that semantic priors from VLMs effectively guide physics-based optimization for fast and reliable RF material estimation.
>
---
#### [replaced 063] Next-Scale Prediction: A Self-Supervised Approach for Real-World Image Denoising
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.21038](https://arxiv.org/pdf/2512.21038)**

> **作者:** Yiwen Shan; Haiyu Zhao; Peng Hu; Xi Peng; Yuanbiao Gou
>
> **摘要:** Self-supervised real-world image denoising remains a fundamental challenge, arising from the antagonistic trade-off between decorrelating spatially structured noise and preserving high-frequency details. Existing blind-spot network (BSN) methods rely on pixel-shuffle downsampling (PD) to decorrelate noise, but aggressive downsampling fragments fine structures, while milder downsampling fails to remove correlated noise. To address this, we introduce Next-Scale Prediction (NSP), a novel self-supervised paradigm that decouples noise decorrelation from detail preservation. NSP constructs cross-scale training pairs, where BSN takes low-resolution, fully decorrelated sub-images as input to predict high-resolution targets that retain fine details. As a by-product, NSP naturally supports super-resolution of noisy images without retraining or modification. Extensive experiments demonstrate that NSP achieves state-of-the-art self-supervised denoising performance on real-world benchmarks, significantly alleviating the long-standing conflict between noise decorrelation and detail preservation. The code is available at this https URL.
>
---
#### [replaced 064] LG-HCC: Local Geometry-Aware Hierarchical Context Compression for 3D Gaussian Splatting
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2603.28431](https://arxiv.org/pdf/2603.28431)**

> **作者:** Xuan Deng; Xiandong Meng; Hengyu Man; Qiang Zhu; Tiange Zhang; Debin Zhao; Xiaopeng Fan
>
> **备注:** 10
>
> **摘要:** Although 3D Gaussian Splatting (3DGS) enables high-fidelity real-time rendering, its prohibitive storage overhead severely hinders practical deployment. Recent anchor-based 3DGS compression schemes reduce gaussian redundancy through some advanced context models. However, they overlook explicit geometric dependencies, leading to structural degradation and suboptimal ratedistortion performance. In this paper, we propose a Local Geometry-aware Hierarchical Context Compression framework for 3DGS(LG-HCC) that incorporates inter-anchor geometric correlations into anchor pruning and entropy coding for compact representation. Specifically, we introduce an Neighborhood-Aware Anchor Pruning (NAAP) strategy, which evaluates anchor importance via weighted neighborhood feature aggregation and then merges low-contribution anchors into salient neighbors, yielding a compact yet geometry-consistent anchor set. Moreover, we further develop a hierarchical entropy coding scheme, in which coarse-to-fine priors are exploited through a lightweight Geometry-Guided Convolution(GG-Conv) operator to enable spatially adaptive context modeling and rate-distortion optimization. Extensive experiments show that LG-HCC effectively alleviates structural preservation issues,achieving superior geometric integrity and rendering fidelity while reducing storage by up to 30.85x compared to the Scaffold-GS baseline on the Mip-NeRF360 dataset
>
---
#### [replaced 065] Resolving the Identity Crisis in Text-to-Image Generation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.01399](https://arxiv.org/pdf/2510.01399)**

> **作者:** Shubhankar Borse; Farzad Farhadzadeh; Munawar Hayat; Fatih Porikli
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** State-of-the-art text-to-image models suffer from a persistent identity crisis when generating scenes with multiple humans: producing duplicate faces, merging identities, and miscounting individuals. We present DisCo (Reinforcement with Diversity Constraints), a reinforcement learning framework that directly optimizes identity diversity both within images and across groups of generated samples. DisCo fine-tunes flow-matching models using Group-Relative Policy Optimization (GRPO), guided by a compositional reward that: (i) penalizes facial similarity within images, (ii) discourages identity repetition across samples, (iii) enforces accurate person counts, and (iv) preserves visual fidelity and prompt alignment via human preference scores. A single-stage curriculum stabilizes training as prompt complexity increases. Importantly, this method does not require any real data. On the DiverseHumans Testset, DisCo achieves 98.6% Unique Face Accuracy and near-perfect Global Identity Spread, outperforming open-source and proprietary models (e.g., Gemini, GPT-Image) while maintaining perceptual quality. Our results establish cross-sample diversity as a critical axis for resolving identity collapse, positioning DisCo as a scalable, annotation-free solution for multi-human image synthesis. Project page: this https URL
>
---
#### [replaced 066] Cross-Camera Distracted Driver Classification through Feature Disentanglement and Contrastive Learning
- **分类: cs.CV; cs.AI; cs.CY**

- **链接: [https://arxiv.org/pdf/2411.13181](https://arxiv.org/pdf/2411.13181)**

> **作者:** Luigi Celona; Simone Bianco; Paolo Napoletano
>
> **摘要:** The classification of distracted drivers is pivotal for ensuring safe driving. Previous studies demonstrated the effectiveness of neural networks in automatically predicting driver distraction, fatigue, and potential hazards. However, recent research has uncovered a significant loss of accuracy in these models when applied to samples acquired under conditions that differ from the training data. In this paper, we introduce a robust model designed to withstand changes in camera position within the vehicle. Our Driver Behavior Monitoring Network (DBMNet) relies on a lightweight backbone and integrates a disentanglement module to discard camera view information from features, coupled with contrastive learning to enhance the encoding of various driver actions. Experiments conducted using a leave-one-camera-out protocol on the daytime and nighttime subsets of the 100-Driver dataset validate the effectiveness of our approach. Cross-dataset and cross-camera experiments conducted on three benchmark datasets, namely AUCDD-V1, EZZ2021 and SFD, demonstrate the superior generalization capabilities of the proposed method. Overall DBMNet achieves an improvement of 7% in Top-1 accuracy compared to existing efficient approaches. Moreover, a quantized version of the DBMNet and all considered methods has been deployed on a Coral Dev Board board. In this deployment scenario, DBMNet outperforms alternatives, achieving the lowest average error while maintaining a compact model size, low memory footprint, fast inference time, and minimal power consumption.
>
---
#### [replaced 067] From Hindsight to Foresight: Self-Encouraged Hindsight Distillation for Knowledge-based Visual Question Answering
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.11132](https://arxiv.org/pdf/2511.11132)**

> **作者:** Yu Zhao; Ying Zhang; Xuhui Sui; Baohang Zhou; Li Shen; Dacheng Tao
>
> **摘要:** Knowledge-based Visual Question Answering (KBVQA) necessitates external knowledge incorporation beyond cross-modal understanding. Existing KBVQA methods either utilize implicit knowledge in multimodal large language models (MLLMs) via in-context learning or explicit knowledge via retrieval augmented generation. However, their reasoning processes remain implicit, without explicit multi-step trajectories from MLLMs. To address this gap, we provide a Hindsight Distilled Reasoning (HinD) framework with Knowledge Encouragement Preference Optimization, aiming at self-encouraging the knowledge reasoning ability inside the MLLM. First, we construct the Hindsight Teacher by prompting the MLLM to complete the reasoning process with knowing the right answer, obtaining Hindsight-Zero training data. Then, the Foresight Student, without knowing the answer, learns the golden trajectories from Hindsight: (1) Hindsight Distillation Fine-Tuning (HDFT) to self-distill the Hindsight-Zero into a modularized Chain-of-Thought (CoT) Generator and a Knowledge Generator for sequential steps and discrete facts generation, respectively; (2) Knowledge Encouragement Preference Optimization (KEPO) to encourage the under-confident but relevant knowledge inside the MLLM and suppress the over-confident but irrelevant one. Experiments on OK-VQA and A-OKVQA validate the effectiveness of HinD, showing that HinD with 7-8B MLLM achieves superior performance without commercial model APIs or retrieved knowledge.
>
---
#### [replaced 068] D4C: Data-Free Quantization for Contrastive Language-Image Pre-training Models
- **分类: cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.15411](https://arxiv.org/pdf/2511.15411)**

> **作者:** Wenlun Zhang; Yunshan Zhong; Zihao Ding; Xinyu Li; Kentaro Yoshioka
>
> **备注:** Accepted to CVPRF 2026
>
> **摘要:** Data-Free Quantization (DFQ) offers a practical solution for model compression without requiring access to real data, making it particularly attractive in privacy-sensitive scenarios. While DFQ has shown promise for unimodal models, its extension to Vision-Language Models such as Contrastive Language-Image Pre-training (CLIP) models remains underexplored. In this work, we reveal that directly applying existing DFQ techniques to CLIP results in substantial performance degradation due to two key limitations: insufficient semantic content and low intra-image diversity in synthesized samples. To tackle these challenges, we propose D4C, the first DFQ framework tailored for CLIP. D4C synthesizes semantically rich and structurally diverse pseudo images through three key components: 1) Prompt-Guided Semantic Injection aligns generated images with real-world semantics using text prompts; 2) Structural Contrastive Generation reproduces compositional structures of natural images by leveraging foreground-background contrastive synthesis; and 3) Perturbation-Aware Enhancement applies controlled perturbations to improve sample diversity and robustness. These components jointly empower D4C to synthesize images that are both semantically informative and structurally diverse, effectively bridging the performance gap of DFQ on CLIP. Extensive experiments validate the effectiveness of D4C, showing significant performance improvements on various bit-widths and models.
>
---
#### [replaced 069] ANVIL: Accelerator-Native Video Interpolation via Codec Motion Vector Priors
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2603.26835](https://arxiv.org/pdf/2603.26835)**

> **作者:** Shibo Liu
>
> **备注:** 12 pages, 4 figures, 10 tables. Submitted to IEEE TCSVT. v3: Fixed architecture diagram and caption to accurately reflect the 4-level U-Net implementation
>
> **摘要:** Real-time 30-to-60 fps video frame interpolation on mobile neural processing units (NPUs) requires each synthesized frame within 33.3 ms. We show that mainstream flow-based video frame interpolation faces three structural deployment barriers on mobile NPUs: spatial sampling operators exceed the frame budget or lack hardware support, iterative flow refinement collapses under 8-bit integer post-training quantization, and memory-bound operators dominate the inference graph. ANVIL addresses these barriers by reusing motion vectors from the H.264/AVC decoder to prealign input frames, removing learned optical flow, spatial sampling, and iterative accumulation from the accelerator graph. The remaining residual is refined by a convolution-dominated network composed almost entirely of compute-bound operators. On a Snapdragon 8 Gen 3 device, ANVIL achieves 12.8 ms 1080p inference at 8-bit integer precision; an open-source Android player sustains 28.4 ms median end-to-end latency over 30-minute continuous playback. Per-operator causal analysis identifies quantized accumulation on recurrent flow states as a key mechanism behind integer quantization failure in iterative methods. The current design targets H.264/AVC playback with decoder-exposed motion vectors.
>
---
#### [replaced 070] VMAD: Visual-enhanced Multimodal Large Language Model for Zero-Shot Anomaly Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2409.20146](https://arxiv.org/pdf/2409.20146)**

> **作者:** Huilin Deng; Hongchen Luo; Wei Zhai; Yang Cao; Yu Kang
>
> **摘要:** Zero-shot anomaly detection (ZSAD) recognizes and localizes anomalies in previously unseen objects by establishing feature mapping between textual prompts and inspection images, demonstrating excellent research value in flexible industrial manufacturing. However, existing ZSAD methods are limited by closed-world settings, struggling to unseen defects with predefined prompts. Recently, adapting Multimodal Large Language Models (MLLMs) for Industrial Anomaly Detection (IAD) presents a viable solution. Unlike fixed-prompt methods, MLLMs exhibit a generative paradigm with open-ended text interpretation, enabling more adaptive anomaly analysis. However, this adaption faces inherent challenges as anomalies often manifest in fine-grained regions and exhibit minimal visual discrepancies from normal samples. To address these challenges, we propose a novel framework VMAD (Visual-enhanced MLLM Anomaly Detection) that enhances MLLM with visual-based IAD knowledge and fine-grained perception, simultaneously providing precise detection and comprehensive analysis of anomalies. Specifically, we design a Defect-Sensitive Structure Learning scheme that transfers patch-similarities cues from visual branch to our MLLM for improved anomaly discrimination. Besides, we introduce a novel visual projector, Locality-enhanced Token Compression, which mines multi-level features in local contexts to enhance fine-grained detection. Furthermore, we introduce the Real Industrial Anomaly Detection (RIAD), a comprehensive IAD dataset with detailed anomaly descriptions and analyses, offering a valuable resource for MLLM-based IAD development. Extensive experiments on zero-shot benchmarks, including MVTec-AD, Visa, WFDD, and RIAD datasets, demonstrate our superior performance over state-of-the-art methods. The code and dataset will be available soon.
>
---
#### [replaced 071] SPDMark: Selective Parameter Displacement for Robust Video Watermarking
- **分类: cs.CV; cs.CR; cs.LG**

- **链接: [https://arxiv.org/pdf/2512.12090](https://arxiv.org/pdf/2512.12090)**

> **作者:** Samar Fares; Nurbek Tastan; Karthik Nandakumar
>
> **备注:** CVPR 2026
>
> **摘要:** The advent of high-quality video generation models has amplified the need for robust watermarking schemes that can be used to reliably detect and track the provenance of generated videos. Existing video watermarking methods based on both post-hoc and in-generation approaches fail to simultaneously achieve imperceptibility, robustness, and computational efficiency. This work introduces a novel framework for in-generation video watermarking called SPDMark (pronounced `SpeedMark') based on selective parameter displacement of a video diffusion model. Watermarks are embedded into the generated videos by modifying a subset of parameters in the generative model. To make the problem tractable, the displacement is modeled as an additive composition of layer-wise basis shifts, where the final composition is indexed by the watermarking key. For parameter efficiency, this work specifically leverages low-rank adaptation (LoRA) to implement the basis shifts. During the training phase, the basis shifts and the watermark extractor are jointly learned by minimizing a combination of message recovery, perceptual similarity, and temporal consistency losses. To detect and localize temporal modifications in the watermarked videos, we use a cryptographic hashing function to derive frame-specific watermark messages from the given base watermarking key. During watermark extraction, maximum bipartite matching is applied to recover the correct frame order, even from temporally tampered videos. Evaluations on both text-to-video and image-to-video generation models demonstrate the ability of SPDMark to generate imperceptible watermarks that can be recovered with high accuracy and also establish its robustness against a variety of common video modifications.
>
---
#### [replaced 072] Processing and acquisition traces in visual encoders: What does CLIP know about your camera?
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.10637](https://arxiv.org/pdf/2508.10637)**

> **作者:** Ryan Ramos; Vladan Stojnić; Giorgos Kordopatis-Zilos; Yuta Nakashima; Giorgos Tolias; Noa Garcia
>
> **备注:** 8 main pages, supplementary attached, ICCV 2025 highlight
>
> **摘要:** Prior work has analyzed the robustness of visual encoders to image transformations and corruptions, particularly in cases where such alterations are not seen during training. When this occurs, they introduce a form of distribution shift at test time, often leading to performance degradation. The primary focus has been on severe corruptions that, when applied aggressively, distort useful signals necessary for accurate semantic predictions. We take a different perspective by analyzing parameters of the image acquisition process and transformations that may be subtle or even imperceptible to the human eye. We find that such parameters are systematically encoded in the learned visual representations and can be easily recovered. More strikingly, their presence can have a profound impact, either positively or negatively, on semantic predictions. This effect depends on whether there is a strong correlation or anti-correlation between semantic labels and these acquisition-based or processing-based labels. Our code and data are available at: this https URL
>
---
#### [replaced 073] Le MuMo JEPA: Multi-Modal Self-Supervised Representation Learning with Learnable Fusion Tokens
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.24327](https://arxiv.org/pdf/2603.24327)**

> **作者:** Ciem Cornelissen; Sam Leroux; Pieter Simoens
>
> **备注:** 14 pages, 4 figures, supplementary material. Accepted at the CVPR 2026 Workshop on Unified Robotic Vision with Cross-Modal Sensing and Alignment (URVIS)
>
> **摘要:** Self-supervised learning has emerged as a powerful paradigm for learning visual representations without manual annotations, yet most methods still operate on a single modality and therefore miss the complementary structure available from heterogeneous sensors. We present Le MuMo JEPA, a self-supervised framework that learns unified representations from RGB images and aligned companion modalities. In our driving experiments, the second modality is camera-aligned LiDAR depth; we also evaluate RGB-thermal training and transfer on the Teledyne FLIR ADAS benchmark. Our approach extends LeJEPA to the multi-modal setting by learning fusion tokens that act as a latent bottleneck between modality-specific patch stems inside a shared transformer. Our default model employs a pruned fusion strategy: after an initial cross-modal attention layer, modality-specific tokens are dropped, forcing cross-modal information into the shared fusion-token grid as an efficient latent bottleneck before Sketched Isotropic Gaussian Regularization (SIGReg) is applied to the joint multimodal CLS embedding. On Waymo, Le MuMo JEPA gives the strongest performance-efficiency trade-off on downstream patch probes among the from-scratch multimodal baselines, improving CenterNet detection and dense depth while remaining competitive on segmentation. Under from-scratch training on nuScenes, Le MuMo JEPA remains the strongest model, and it also gives the best FLIR results, especially after Waymo-initialized fine-tuning. It also retains the best overall accuracy-efficiency balance in our study at substantially lower compute, memory, and estimated training time.
>
---
#### [replaced 074] Toward Physically Consistent Driving Video World Models under Challenging Trajectories
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.24506](https://arxiv.org/pdf/2603.24506)**

> **作者:** Jiawei Zhou; Zhenxin Zhu; Lingyi Du; Linye Lyu; Lijun Zhou; Zhanqian Wu; Hongcheng Luo; Zhuotao Tian; Bing Wang; Guang Chen; Hangjun Ye; Haiyang Sun; Yu Li
>
> **摘要:** Video generation models have shown strong potential as world models for autonomous driving simulation. However, existing approaches are primarily trained on real-world driving datasets, which mostly contain natural and safe driving scenarios. As a result, current models often fail when conditioned on challenging or counterfactual trajectories-such as imperfect trajectories generated by simulators or planning systems-producing videos with severe physical inconsistencies and artifacts. To address this limitation, we propose PhyGenesis, a world model designed to generate driving videos with high visual fidelity and strong physical consistency. Our framework consists of two key components: (1) a physical condition generator that transforms potentially invalid trajectory inputs into physically plausible conditions, and (2) a physics-enhanced video generator that produces high-fidelity multi-view driving videos under these conditions. To effectively train these components, we construct a large-scale, physics-rich heterogeneous dataset. Specifically, in addition to real-world driving videos, we generate diverse challenging driving scenarios using the CARLA simulator, from which we derive supervision signals that guide the model to learn physically grounded dynamics under extreme conditions. This challenging-trajectory learning strategy enables trajectory correction and promotes physically consistent video generation. Extensive experiments demonstrate that PhyGenesis consistently outperforms state-of-the-art methods, especially on challenging trajectories. Our project page is available at: this https URL.
>
---
#### [replaced 075] ForgeDreamer: Industrial Text-to-3D Generation with Multi-Expert LoRA and Cross-View Hypergraph
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.09266](https://arxiv.org/pdf/2603.09266)**

> **作者:** Junhao Cai; Deyu Zeng; Junhao Pang; Lini Li; Zongze Wu; Xiaopin Zhong
>
> **备注:** Accepted to CVPR 2026 Findings!
>
> **摘要:** Current text-to-3D generation methods excel in natural scenes but struggle with industrial applications due to two critical limitations: domain adaptation challenges where conventional LoRA fusion causes knowledge interference across categories, and geometric reasoning deficiencies where pairwise consistency constraints fail to capture higher-order structural dependencies essential for precision manufacturing. We propose a novel framework named ForgeDreamer addressing both challenges through two key innovations. First, we introduce a Multi-Expert LoRA Ensemble mechanism that consolidates multiple category-specific LoRA models into a unified representation, achieving superior cross-category generalization while eliminating knowledge interference. Second, building on enhanced semantic understanding, we develop a Cross-View Hypergraph Geometric Enhancement approach that captures structural dependencies spanning multiple viewpoints simultaneously. These components work synergistically improved semantic understanding, enables more effective geometric reasoning, while hypergraph modeling ensures manufacturing-level consistency. Extensive experiments on a custom industrial dataset demonstrate superior semantic generalization and enhanced geometric fidelity compared to state-of-the-art approaches. Code is available at this https URL
>
---
#### [replaced 076] RefTon: Reference person shot assist virtual Try-on
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.00956](https://arxiv.org/pdf/2511.00956)**

> **作者:** Liuzhuozheng Li; Yue Gong; Shanyuan Liu; Bo Cheng; Yuhang Ma; Leibucha Wu; Dengyang Jiang; Zanyi Wang; Dawei Leng; Yuhui Yin
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** We introduce RefTon, a flux-based person-to-person virtual try-on framework that enhances garment realism through unpaired visual references. Unlike conventional approaches that rely on complex auxiliary inputs such as body parsing and warped mask or require finely designed extract branches to process various input conditions, RefTon streamlines the process by directly generating try-on results from a source image and a target garment, without the need for structural guidance or auxiliary components to handle diverse inputs. Moreover, inspired by human clothing selection behavior, RefTon leverages additional reference images (the target garment worn on different individuals) to provide powerful guidance for refining texture alignment and maintaining the garment details. To enable this capability, we built a dataset containing unpaired reference images for training. Extensive experiments on public benchmarks demonstrate that RefTon achieves competitive or superior performance compared to state-of-the-art methods, while maintaining a simple and efficient person-to-person design.
>
---
#### [replaced 077] ActErase: A Training-Free Paradigm for Precise Concept Erasure via Activation Redirection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2601.00267](https://arxiv.org/pdf/2601.00267)**

> **作者:** Yi Sun; Xinhao Zhong; Hongyan Li; Yimin Zhou; Junhao Li; Bin Chen; Xuan Wang
>
> **摘要:** Recent advances in text-to-image diffusion models have demonstrated remarkable generation capabilities, yet they raise significant concerns regarding safety, copyright, and ethical implications. Existing concept erasure methods address these risks by removing sensitive concepts from pre-trained models, but most of them rely on data-intensive and computationally expensive fine-tuning, which poses a critical limitation. To overcome these challenges, inspired by the observation that the model's activations are predominantly composed of generic concepts, with only a minimal component can represent the target concept, we propose a novel training-free method (ActErase) for efficient concept erasure. Specifically, the proposed method operates by identifying activation difference regions via prompt-pair analysis, extracting target activations and dynamically replacing input activations during forward passes. Comprehensive evaluations across three critical erasure tasks (nudity, artistic style, and object removal) demonstrates that our training-free method achieves state-of-the-art (SOTA) erasure performance, while effectively preserving the model's overall generative capability. Our approach also exhibits strong robustness against adversarial attacks, establishing a new plug-and-play paradigm for lightweight yet effective concept manipulation in diffusion models.
>
---
#### [replaced 078] CDH-Bench: A Commonsense-Driven Hallucination Benchmark for Evaluating Visual Fidelity in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型评估任务，旨在解决视觉证据与常识冲突下的幻觉问题。通过构建CDH-Bench基准，评估模型在不同异常类型下的表现。**

- **链接: [https://arxiv.org/pdf/2603.27982](https://arxiv.org/pdf/2603.27982)**

> **作者:** Kesheng Chen; Yamin Hu; Qi Zhou; Zhenqian Zhu; Wenjian Luo
>
> **摘要:** Vision-language models (VLMs) achieve strong performance on many benchmarks, yet a basic reliability question remains underexplored: when visual evidence conflicts with commonsense, do models follow what is shown or what commonsense suggests? A characteristic failure in this setting is that the model overrides visual evidence and outputs the commonsense alternative. We term this phenomenon \textbf{commonsense-driven hallucination} (CDH). To evaluate it, we introduce \textbf{CDH-Bench}, a benchmark designed to create explicit \textbf{visual evidence--commonsense conflicts}. CDH-Bench covers three dimensions: \textit{counting anomalies}, \textit{relational anomalies}, and \textit{attribute anomalies}. We evaluate frontier VLMs under \textit{binary Question Answering (QA)} and \textit{multiple-choice QA}, and report metrics including \textit{Counterfactual Accuracy} (CF-Acc), \textit{Commonsense Accuracy} (CS-Acc), \textit{Counterfactual Accuracy Drop} (CFAD), \textit{Commonsense Collapse Rate} (CCR), and \textit{Relative Prior Dependency} (RPD). Results show that even strong models remain vulnerable to prior-driven normalization under visual evidence--commonsense conflict. CDH-Bench provides a controlled diagnostic of visual fidelity under visual evidence--commonsense conflict.
>
---
#### [replaced 079] Low-Resolution Editing is All You Need for High-Resolution Editing
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.19945](https://arxiv.org/pdf/2511.19945)**

> **作者:** Junsung Lee; Hyunsoo Lee; Yong Jae Lee; Bohyung Han
>
> **备注:** CVPR 2026
>
> **摘要:** High-resolution content creation is rapidly emerging as a central challenge in both the vision and graphics communities. Images serve as the most fundamental modality for visual expression, and content generation that aligns with the user intent requires effective, controllable high-resolution image manipulation mechanisms. However, existing approaches remain limited to low-resolution settings, typically supporting only up to 1K resolution. In this work, we introduce the task of high-resolution image editing and propose a test-time optimization framework to address it. Our method performs patch-wise optimization on high-resolution source images, followed by a fine-grained detail transfer module and a novel synchronization strategy to maintain consistency across patches. Extensive experiments show that our method produces high-quality edits, facilitating high-resolution content creation.
>
---
#### [replaced 080] Can We Go Beyond Visual Features? Neural Tissue Relation Modeling for Relational Graph Analysis in Non-Melanoma Skin Histology
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.06949](https://arxiv.org/pdf/2512.06949)**

> **作者:** Shravan Venkatraman; Muthu Subash Kavitha; Joe Dhanith P R; V Manikandarajan; Jia Wu
>
> **备注:** CVPR 2026 Workshops
>
> **摘要:** Histopathology image segmentation is essential for delineating tissue structures in skin cancer diagnostics, but modeling spatial context and inter-tissue relationships remains a challenge, especially in regions with overlapping or morphologically similar tissues. Current convolutional neural network (CNN)-based approaches operate primarily on visual texture, often treating tissues as independent regions and failing to encode biological context. To this end, we introduce Neural Tissue Relation Modeling (NTRM), a novel segmentation framework that augments CNNs with a tissue-level graph neural network to model spatial and functional relationships across tissue types. NTRM constructs a graph over predicted regions, propagates contextual information via message passing, and refines segmentation through spatial projection. Unlike prior methods, NTRM explicitly encodes inter-tissue dependencies, enabling structurally coherent predictions in boundary-dense zones. On the benchmark Histopathology Non-Melanoma Skin Cancer Segmentation Dataset, NTRM outperforms state-of-the-art methods, achieving a robust Dice similarity coefficient that is 4.9\% to 31.25\% higher than the best-performing models among the evaluated approaches. Our experiments indicate that relational modeling offers a principled path toward more context-aware and interpretable histological segmentation, compared to local receptive-field architectures that lack tissue-level structural awareness. Our code is available at this https URL.
>
---
#### [replaced 081] Unify-Agent: A Unified Multimodal Agent for World-Grounded Image Synthesis
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2603.29620](https://arxiv.org/pdf/2603.29620)**

> **作者:** Shuang Chen; Quanxin Shou; Hangting Chen; Yucheng Zhou; Kaituo Feng; Wenbo Hu; Yi-Fan Zhang; Yunlong Lin; Wenxuan Huang; Mingyang Song; Dasen Dai; Bolin Jiang; Manyuan Zhang; Shi-Xue Zhang; Zhengkai Jiang; Lucas Wang; Zhao Zhong; Yu Cheng; Nanyun Peng
>
> **备注:** Project Page: this https URL
>
> **摘要:** Unified multimodal models provide a natural and promising architecture for understanding diverse and complex real-world knowledge while generating high-quality images. However, they still rely primarily on frozen parametric knowledge, which makes them struggle with real-world image generation involving long-tail and knowledge-intensive concepts. Inspired by the broad success of agents on real-world tasks, we explore agentic modeling to address this limitation. Specifically, we present Unify-Agent, a unified multimodal agent for world-grounded image synthesis, which reframes image generation as an agentic pipeline consisting of prompt understanding, multimodal evidence searching, grounded recaptioning, and final synthesis. To train our model, we construct a tailored multimodal data pipeline and curate 143K high-quality agent trajectories for world-grounded image synthesis, enabling effective supervision over the full agentic generation process. We further introduce FactIP, a benchmark covering 12 categories of culturally significant and long-tail factual concepts that explicitly requires external knowledge grounding. Extensive experiments show that our proposed Unify-Agent substantially improves over its base unified model across diverse benchmarks and real world generation tasks, while approaching the world knowledge capabilities of the strongest closed-source models. As an early exploration of agent-based modeling for world-grounded image synthesis, our work highlights the value of tightly coupling reasoning, searching, and generation for reliable open-world agentic image synthesis.
>
---
#### [replaced 082] Not All Birds Look The Same: Identity-Preserving Generation For Birds
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.04485](https://arxiv.org/pdf/2512.04485)**

> **作者:** Aaron Sun; Oindrila Saha; Subhransu Maji
>
> **摘要:** Since the advent of controllable image generation, increasingly rich modes of control have enabled greater customization and accessibility for everyday users. Zero-shot, identity-preserving models such as Insert Anything and OminiControl now support applications like virtual try-on without requiring additional fine-tuning. While these models may be fitting for humans and rigid everyday objects, they still have limitations for non-rigid or fine-grained categories. These domains often lack accessible, high-quality data -- especially videos or multi-view observations of the same subject -- making them difficult both to evaluate and to improve upon. Yet, such domains are essential for moving beyond content creation toward applications that demand accuracy and fine detail. Birds are an excellent domain for this task: they exhibit high diversity, require fine-grained cues for identification, and come in a wide variety of poses. We introduce the NABirds Look-Alikes (NABLA) dataset, consisting of 4,759 expert-curated image pairs. Together with 1,073 pairs collected from multi-image observations on iNaturalist and a small set of videos, this forms a benchmark for evaluating identity-preserving generation of birds. We show that state-of-the-art baselines fail to maintain identity on this dataset, and we demonstrate that training on images grouped by species, age, and sex -- used as a proxy for identity -- substantially improves performance on both seen and unseen species.
>
---
#### [replaced 083] Harmonization in Magnetic Resonance Imaging: A Survey of Acquisition, Image-level, and Feature-level Methods
- **分类: eess.IV; cs.CV; physics.med-ph**

- **链接: [https://arxiv.org/pdf/2507.16962](https://arxiv.org/pdf/2507.16962)**

> **作者:** Qinqin Yang; Firoozeh Shomal-Zadeh; Ali Gholipour
>
> **备注:** 27 pages, 6 figures, 3 tables
>
> **摘要:** Magnetic resonance imaging (MRI) has greatly advanced neuroscience research and clinical diagnostics. However, imaging data collected across different scanners, acquisition protocols, or imaging sites often exhibit substantial heterogeneity, known as batch effects or site effects. These non-biological sources of variability can obscure true biological signals, reduce reproducibility and statistical power, and severely impair the generalizability of learning-based models across datasets. Image harmonization is grounded in the central hypothesis that site-related biases can be eliminated or mitigated while preserving meaningful biological information, thereby improving data comparability and consistency. This review provides a comprehensive overview of key concepts, methodological advances, publicly available datasets, and evaluation metrics in the field of MRI harmonization. We systematically cover the full imaging pipeline and categorize harmonization approaches into prospective acquisition and reconstruction, retrospective image-level and feature-level methods, and traveling-subject-based techniques. By synthesizing existing methods and evidence, we revisit the central hypothesis of image harmonization and show that, although site invariance can be achieved with current techniques, further evaluation is required to verify the preservation of biological information. To this end, we summarize the remaining challenges and highlight key directions for future research, including the need for standardized validation benchmarks, improved evaluation strategies, and tighter integration of harmonization methods across the imaging pipeline.
>
---
#### [replaced 084] CodeDance: A Dynamic Tool-integrated MLLM for Executable Visual Reasoning
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.17312](https://arxiv.org/pdf/2512.17312)**

> **作者:** Qi Song; Honglin Li; Yingchen Yu; Haoyi Zhou; Lin Yang; Song Bai; Qi She; Zilong Huang; Yunqing Zhao
>
> **备注:** CVPR 2026. Project page: this https URL
>
> **摘要:** Recent releases such as o3 highlight human-like "thinking with images" reasoning that combines tool use with stepwise verification, yet most open-source approaches still rely on text-only chains, rigid visual schemas, or single-step pipelines, limiting flexibility, interpretability, and transferability on complex tasks. We introduce CodeDance, which explores executable code as a general solver for visual reasoning. Unlike fixed-schema calls (e.g., only predicting bounding-box coordinates), CodeDance defines, composes, and executes code to orchestrate multiple tools, compute intermediate results, and render visual artifacts (e.g., boxes, lines, plots) that support transparent, self-checkable reasoning. To guide this process, we introduce a reward for balanced and adaptive tool calling, which balances exploration with efficiency and mitigates tool overuse. Interestingly, beyond the expected capabilities taught by atomic supervision, we empirically observe novel emergent behaviors during RL training: CodeDance demonstrates novel tool invocations, unseen compositions, and cross-task transfer. These behaviors arise without task-specific fine-tuning, suggesting a general and scalable mechanism for executable visual reasoning. Extensive experiments across reasoning benchmarks (e.g., visual search, math, chart QA) show that CodeDance not only consistently outperforms schema-driven and text-only baselines, but also surpasses closed models such as GPT-4o and larger open-source models.
>
---
#### [replaced 085] Spatial Reasoning is Not a Free Lunch: A Controlled Study on LLaVA
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.12545](https://arxiv.org/pdf/2603.12545)**

> **作者:** Nahid Alam; Leema Krishna Murali; Siddhant Bharadwaj; Patrick Liu; Timothy Chung; Drishti Sharma; Akshata A.; Kranthi Kiran; Wesley Tam; Bala Krishna S Vegesna
>
> **备注:** Accepted as a poster at ICLR 2026 workshop ICBINB, typo fixed
>
> **摘要:** Vision-language models (VLMs) have advanced rapidly, yet they still struggle with basic spatial reasoning. Despite strong performance on general benchmarks, modern VLMs remain brittle at understanding 2D spatial relationships such as relative position, layout, and counting. We argue that this failure is not merely a data problem, but is closely tied to dominant design choices in current VLM pipelines: reliance on CLIP-style image encoders and the flattening of images into 1D token sequences with 1D positional encoding. We present a controlled diagnostic study within the LLaVA framework to isolate how these choices affect spatial grounding. We evaluate frontier models and LLaVA variants on a suite of spatial benchmarks, comparing CLIP-based encoders against alternatives trained with denser or generative objectives, as well as variants augmented with 2D positional encoding. Our results show consistent spatial performance gaps across models, and indicate that encoder objectives and positional structure shape spatial behavior, but do not fully resolve it.
>
---
#### [replaced 086] Beyond the Ground Truth: Enhanced Supervision for Image Restoration
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.03932](https://arxiv.org/pdf/2512.03932)**

> **作者:** Donghun Ryou; Inju Ha; Sanghyeok Chu; Bohyung Han
>
> **备注:** Project page: this https URL Accepted to CVPR 2026
>
> **摘要:** Deep learning-based image restoration has achieved significant success. However, when addressing real-world degradations, model performance is limited by the quality of groundtruth images in datasets due to practical constraints in data acquisition. To address this limitation, we propose a novel framework that enhances existing ground truth images to provide higher-quality supervision for real-world restoration. Our framework generates perceptually enhanced ground truth images using super-resolution by incorporating adaptive frequency masks, which are learned by a conditional frequency mask generator. These masks guide the optimal fusion of frequency components from the original ground truth and its super-resolved variants, yielding enhanced ground truth images. This frequency-domain mixup preserves the semantic consistency of the original content while selectively enriching perceptual details, preventing hallucinated artifacts that could compromise fidelity. The enhanced ground truth images are used to train a lightweight output refinement network that can be seamlessly integrated with existing restoration models. Extensive experiments demonstrate that our approach improves the quality of restored images. We further validate the effectiveness of both supervision enhancement and output refinement through user studies.
>
---
#### [replaced 087] SHIFT: Stochastic Hidden-Trajectory Deflection for Removing Diffusion-based Watermark
- **分类: cs.CV; cs.CR**

- **链接: [https://arxiv.org/pdf/2603.29742](https://arxiv.org/pdf/2603.29742)**

> **作者:** Rui Bao; Zheng Gao; Xiaoyu Li; Xiaoyan Feng; Yang Song; Jiaojiao Jiang
>
> **摘要:** Diffusion-based watermarking methods embed verifiable marks by manipulating the initial noise or the reverse diffusion trajectory. However, these methods share a critical assumption: verification can succeed only if the diffusion trajectory can be faithfully reconstructed. This reliance on trajectory recovery constitutes a fundamental and exploitable vulnerability. We propose $\underline{\mathbf{S}}$tochastic $\underline{\mathbf{Hi}}$dden-Trajectory De$\underline{\mathbf{f}}$lec$\underline{\mathbf{t}}$ion ($\mathbf{SHIFT}$), a training-free attack that exploits this common weakness across diverse watermarking paradigms. SHIFT leverages stochastic diffusion resampling to deflect the generative trajectory in latent space, making the reconstructed image statistically decoupled from the original watermark-embedded trajectory while preserving strong visual quality and semantic consistency. Extensive experiments on nine representative watermarking methods spanning noise-space, frequency-domain, and optimization-based paradigms show that SHIFT achieves 95%--100% attack success rates with nearly no loss in semantic quality, without requiring any watermark-specific knowledge or model retraining.
>
---
#### [replaced 088] Sketch It Out: Exploring Label-Free Structural Cues for Multimodal Gait Recognition
- **分类: cs.CV; eess.IV**

- **链接: [https://arxiv.org/pdf/2603.05537](https://arxiv.org/pdf/2603.05537)**

> **作者:** Chao Zhang; Zhuang Zheng; Ruixin Li; Zhanyong Mei
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Gait recognition is a non-intrusive biometric technique for security applications, yet existing studies are dominated by silhouette- and parsing-based representations. Silhouettes are sparse and miss internal structural details, limiting discriminability. Parsing enriches silhouettes with part-level structures, but relies heavily on upstream human parsers (e.g., label granularity and boundary precision), leading to unstable performance across datasets and sometimes even inferior results to silhouettes. We revisit gait representations from a structural perspective and describe a design space defined by edge density and supervision form: silhouettes use sparse boundary edges with weak single-label supervision, while parsing uses denser cues with strong semantic priors. In this space, we identify an underexplored paradigm: dense part-level structure without explicit semantic labels, and introduce SKETCH as a new visual modality for gait recognition. Sketch extracts high-frequency structural cues (e.g., limb articulations and self-occlusion contours) directly from RGB images via edge-based detectors in a label-free manner. We further show that label-guided parsing and label-free sketch are semantically decoupled and structurally complementary. Based on this, we propose SKETCHGAIT, a hierarchically disentangled multi-modal framework with two independent streams for modality-specific learning and a lightweight early-stage fusion branch to capture structural complementarity. Extensive experiments on SUSTech1K and CCPG validate the proposed modality and framework: SketchGait achieves 92.9% Rank-1 on SUSTech1K and 93.1% mean Rank-1 on CCPG.
>
---
#### [replaced 089] Exploring Self-Supervised Learning with U-Net Masked Autoencoders and EfficientNet-B7 for Improved Gastrointestinal Abnormality Classification in Video Capsule Endoscopy
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2410.19899](https://arxiv.org/pdf/2410.19899)**

> **作者:** Vamshi Krishna Kancharla; Pavan Kumar Kaveti; Dasari Naga Raju
>
> **备注:** Capsule Vision 2024 Challenge
>
> **摘要:** Video Capsule Endoscopy (VCE) has become an indispensable diagnostic tool for gastrointestinal (GI) disorders due to its non-invasive nature and ability to capture high-resolution images of the small intestine. However, the enormous volume of data generated during a single procedure makes manual inspection labor-intensive, time-consuming, and prone to inter-observer variability. Automated analysis using deep learning offers a promising solution, but its effectiveness is often limited by data imbalance and the high cost of labeled medical data. In this work, we propose a novel framework that combines self-supervised learning through a U-Net-based masked autoencoder with supervised feature extraction using EfficientNet-B7 for multi-class abnormality classification in VCE images. The U-Net model is first trained in a self-supervised manner using Gaussian noise removal and masked reconstruction to learn robust visual representations without requiring annotations. The learned encoder features are then fused with EfficientNet-B7 features to form a rich, discriminative representation for classification. We evaluate our approach on the Capsule Vision 2024 Challenge dataset consisting of ten abnormality classes and a dominant normal class. Experimental results demonstrate that the proposed fusion framework achieves a validation accuracy of 94\%, outperforming standalone architectures and attention-based fusion variants. The study highlights the effectiveness of self-supervised representation learning and feature fusion in addressing class imbalance and improving diagnostic accuracy in real-world medical imaging scenarios.
>
---
#### [replaced 090] Octree Diffusion for Semantic Scene Generation and Completion
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.16483](https://arxiv.org/pdf/2509.16483)**

> **作者:** Xujia Zhang; Brendan Crowe; Christoffer Heckman
>
> **备注:** Accepted to ICRA 2026. Revised version with updated paragraphs
>
> **摘要:** The completion, extension, and generation of 3D semantic scenes are an interrelated set of capabilities that are useful for robotic navigation and exploration. Existing approaches seek to decouple these problems and solve them one-off. Additionally, these approaches are often domain-specific, requiring separate models for different data distributions, e.g.\ indoor vs.\ outdoor scenes. To unify these techniques and provide cross-domain compatibility, we develop a single framework that can perform scene completion, extension, and generation in both indoor and outdoor scenes, which we term Octree Latent Semantic Diffusion. Our approach operates directly on an efficient dual octree graph latent representation: a hierarchical, sparse, and memory-efficient occupancy structure. This technique disentangles synthesis into two stages: (i) structure diffusion, which predicts binary split signals to construct a coarse occupancy octree, and (ii) latent semantic diffusion, which generates semantic embeddings decoded by a graph VAE into voxel-level semantic labels. To perform semantic scene completion or extension, our model leverages inference-time latent inpainting, or outpainting respectively. These inference-time methods use partial LiDAR scans or maps to condition generation, without the need for retraining or finetuning. We demonstrate high-quality structure, coherent semantics, and robust completion from single LiDAR scans, as well as zero-shot generalization to out-of-distribution LiDAR data. These results indicate that completion-through-generation in a dual octree graph latent space is a practical and scalable alternative to regression-based pipelines for real-world robotic perception tasks.
>
---
#### [replaced 091] Enhancing Floor Plan Recognition: A Hybrid Mix-Transformer and U-Net Approach for Precise Wall Segmentation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.02413](https://arxiv.org/pdf/2512.02413)**

> **作者:** Dmitriy Parashchuk; Alexey Kaspshitskiy; Yuriy Karyakin
>
> **备注:** 11 pages, 5 figures, 3 tables
>
> **摘要:** Automatic 3D reconstruction of indoor spaces from 2D floor plans necessitates high-precision semantic segmentation of structural elements, particularly walls. However, existing methods often struggle with detecting thin structures and maintaining geometric precision. To address this, we introduce MitUNet, a hybrid neural network designed to bridge the gap between global semantic context and fine-grained structural details. Our architecture combines a Mix-Transformer encoder with a U-Net decoder enhanced with spatial and channel attention blocks. Optimized with the Tversky loss function, this approach achieves a balance between precision and recall, ensuring accurate boundary recovery. Experiments on the CubiCasa5k dataset and the regional dataset demonstrate MitUNet's superiority in generating structurally correct masks with high boundary accuracy, outperforming standard models. This tool provides a robust foundation for automated 3D reconstruction pipelines. To ensure reproducibility and facilitate future research, the source code and the regional dataset are publicly available at this https URL and this https URL, respectively.
>
---
#### [replaced 092] Vision Tiny Recursion Model (ViTRM): Parameter-Efficient Image Classification via Recursive State Refinement
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.19503](https://arxiv.org/pdf/2603.19503)**

> **作者:** Ange-Clément Akazan; Abdoulaye Koroko; Verlon Roel Mbingui; Choukouriyah Arinloye; Hassan Fifen; Rose Bandolo
>
> **摘要:** The success of deep learning in computer vision has been driven by models of increasing scale, from deep Convolutional Neural Networks (CNN) to large Vision Transformers (ViT). While effective, these architectures are parameter-intensive and demand significant computational resources, limiting deployment in resource-constrained environments. Inspired by Tiny Recursive Models (TRM), which show that small recursive networks can solve complex reasoning tasks through iterative state refinement, we introduce the \textbf{Vision Tiny Recursion Model (ViTRM)}: a parameter-efficient architecture that replaces the $L$-layer ViT encoder with a single tiny $k$-layer block ($k{=}3$) applied recursively $N$ times. Despite using up to $6 \times $ and $84 \times$ fewer parameters than CNN based models and ViT respectively, ViTRM maintains competitive performance on CIFAR-10 and CIFAR-100. This demonstrates that recursive computation is a viable, parameter-efficient alternative to architectural depth in vision.
>
---
#### [replaced 093] Harnessing the Power of Local Representations for Few-Shot Classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2407.01967](https://arxiv.org/pdf/2407.01967)**

> **作者:** Shi Tang; Guiming Luo; Xinchen Ye; Zhiyi Xia
>
> **摘要:** Generalizing to novel classes unseen during training is a key challenge of few-shot classification. Recent metric-based methods try to address this by local representations. However, they are unable to take full advantage of them due to (i) improper supervision for pretraining the feature extractor, and (ii) lack of adaptability in the metric for handling various possible compositions of local feature sets. In this work, we harness the power of local representations in improving novel-class generalization. For the feature extractor, we design a novel pretraining paradigm that learns randomly cropped patches by soft labels. It utilizes the class-level diversity of patches while diminishing the impact of their semantic misalignments to hard labels. To align network output with soft labels, we also propose a UniCon KL-Divergence that emphasizes the equal contribution of each base class in describing "non-base" patches. For the metric, we formulate measuring local feature sets as an entropy-regularized optimal transport problem to introduce the ability to handle sets consisting of homogeneous elements. Furthermore, we design a Modulate Module to endow the metric with the necessary adaptability. Our method achieves new state-of-the-art performance on three popular benchmarks. Moreover, it exceeds state-of-the-art transductive and cross-modal methods in the fine-grained scenario.
>
---
#### [replaced 094] Two-stage Vision Transformers and Hard Masking offer Robust Object Representations
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.08915](https://arxiv.org/pdf/2506.08915)**

> **作者:** Ananthu Aniraj; Cassio F. Dantas; Dino Ienco; Diego Marcos
>
> **备注:** Accepted at ICPR 2026
>
> **摘要:** Context can strongly affect object representations, sometimes leading to undesired biases, particularly when objects appear in out-of-distribution backgrounds at inference. At the same time, many object-centric tasks require to leverage the context for identifying the relevant image regions. We posit that this conundrum, in which context is simultaneously needed and a potential nuisance, can be addressed by an attention-based approach that uses learned binary attention masks to ensure that only attended image regions influence the prediction. To test this hypothesis, we evaluate a two-stage framework: stage 1 processes the full image to discover object parts and identify task-relevant regions, for which context cues are likely to be needed, while stage 2 leverages input attention masking to restrict its receptive field to these regions, enabling a focused analysis while filtering out potentially spurious information. Both stages are trained jointly, allowing stage 2 to refine stage 1. The explicit nature of the semantic masks also makes the model's reasoning auditable, enabling powerful test-time interventions to further enhance robustness. Extensive experiments across diverse benchmarks demonstrate that this approach significantly improves robustness against spurious correlations and out-of-distribution backgrounds. Code: this https URL
>
---
#### [replaced 095] Refracting Reality: Generating Images with Realistic Transparent Objects
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.17340](https://arxiv.org/pdf/2511.17340)**

> **作者:** Yue Yin; Enze Tao; Dylan Campbell
>
> **备注:** this https URL
>
> **摘要:** Generative image models can produce convincingly real images, with plausible shapes, textures, layouts and lighting. However, one domain in which they perform notably poorly is in the synthesis of transparent objects, which exhibit refraction, reflection, absorption and scattering. Refraction is a particular challenge, because refracted pixel rays often intersect with surfaces observed in other parts of the image, providing a constraint on the color. It is clear from inspection that generative models have not distilled the laws of optics sufficiently well to accurately render refractive objects. In this work, we consider the problem of generating images with accurate refraction, given a text prompt. We synchronize the pixels within the object's boundary with those outside by warping and merging the pixels using Snell's Law of Refraction, at each step of the generation trajectory. For those surfaces that are not directly observed in the image, but are visible via refraction or reflection, we recover their appearance by synchronizing the image with a second generated image -- a panorama centered at the object -- using the same warping and merging procedure. We demonstrate that our approach generates much more optically-plausible images that respect the physical constraints.
>
---
#### [replaced 096] Pulp Motion: Framing-aware multimodal camera and human motion generation
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.05097](https://arxiv.org/pdf/2510.05097)**

> **作者:** Robin Courant; Xi Wang; David Loiseaux; Marc Christie; Vicky Kalogeiton
>
> **备注:** Project page: this https URL
>
> **摘要:** Treating human motion and camera trajectory generation separately overlooks a core principle of cinematography: the tight interplay between actor performance and camera work in the screen space. In this paper, we are the first to cast this task as a text-conditioned joint generation, aiming to maintain consistent on-screen framing while producing two heterogeneous, yet intrinsically linked, modalities: human motion and camera trajectories. We propose a simple, model-agnostic framework that enforces multimodal coherence via an auxiliary modality: the on-screen framing induced by projecting human joints onto the camera. This on-screen framing provides a natural and effective bridge between modalities, promoting consistency and leading to more precise joint distribution. We first design a joint autoencoder that learns a shared latent space, together with a lightweight linear transform from the human and camera latents to a framing latent. We then introduce auxiliary sampling, which exploits this linear transform to steer generation toward a coherent framing modality. To support this task, we also introduce the PulpMotion dataset, a human-motion and camera-trajectory dataset with rich captions, and high-quality human motions. Extensive experiments across DiT- and MAR-based architectures show the generality and effectiveness of our method in generating on-frame coherent human-camera motions, while also achieving gains on textual alignment for both modalities. Our qualitative results yield more cinematographically meaningful framings setting the new state of the art for this task. Code, models and data are available in our \href{this https URL}{project page}.
>
---
#### [replaced 097] BigEarthNet.txt: A Large-Scale Multi-Sensor Image-Text Dataset and Benchmark for Earth Observation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2603.29630](https://arxiv.org/pdf/2603.29630)**

> **作者:** Johann-Ludwig Herzog; Mathis Jürgen Adler; Leonard Hackel; Yan Shu; Angelos Zavras; Ioannis Papoutsis; Paolo Rota; Begüm Demir
>
> **备注:** For details, see this https URL
>
> **摘要:** Vision-langugage models (VLMs) have shown strong performance in computer vision (CV), yet their performance on remote sensing (RS) data remains limited due to the lack of large-scale, multi-sensor RS image-text datasets with diverse textual annotations. Existing datasets predominantly include aerial Red-Green-Blue imagery, with short or weakly grounded captions, and provide limited diversity in annotation types. To address this limitation, we introduce BigEarthNet$.$txt, a large-scale, multi-sensor image-text dataset designed to advance instruction-driven image-text learning in Earth observation across multiple tasks. BigEarthNet$.$txt contains 464044 co-registered Sentinel-1 synthetic aperture radar and Sentinel-2 multispectral images with 9.6M text annotations, including: i) geographically anchored captions describing land-use/land-cover (LULC) classes, their spatial relations, and environmental context; ii) visual question answering pairs relevant for different tasks; and iii) referring expression detection instructions for bounding box prediction. Through a comparative statistical analysis, we demonstrate that BigEarthNet$.$txt surpasses existing RS image-text datasets in textual richness and annotation type variety. We further establish a manually-verified benchmark split to evaluate VLMs in RS and CV. The results show the limitations of these models on tasks that involve complex LULC classes, whereas fine-tuning using BigEarthNet$.$txt results in consistent performance gains across all considered tasks.
>
---
