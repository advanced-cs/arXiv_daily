# 计算机视觉 cs.CV

- **最新发布 97 篇**

- **更新 73 篇**

## 最新发布

#### [new 001] Midway Network: Learning Representations for Recognition and Motion from Latent Dynamics
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉表征学习任务，旨在解决从无标签自然视频中同时学习物体识别与运动理解的问题。作者提出了Midway Network，通过扩展潜在动态建模，结合中层路径、前向预测目标和分层结构，实现对复杂多物体场景的有效建模，并在语义分割与光流估计任务上取得优异表现。**

- **链接: [http://arxiv.org/pdf/2510.05558v1](http://arxiv.org/pdf/2510.05558v1)**

> **作者:** Christopher Hoang; Mengye Ren
>
> **备注:** Project page: https://agenticlearning.ai/midway-network/
>
> **摘要:** Object recognition and motion understanding are key components of perception that complement each other. While self-supervised learning methods have shown promise in their ability to learn from unlabeled data, they have primarily focused on obtaining rich representations for either recognition or motion rather than both in tandem. On the other hand, latent dynamics modeling has been used in decision making to learn latent representations of observations and their transformations over time for control and planning tasks. In this work, we present Midway Network, a new self-supervised learning architecture that is the first to learn strong visual representations for both object recognition and motion understanding solely from natural videos, by extending latent dynamics modeling to this domain. Midway Network leverages a midway top-down path to infer motion latents between video frames, as well as a dense forward prediction objective and hierarchical structure to tackle the complex, multi-object scenes of natural videos. We demonstrate that after pretraining on two large-scale natural video datasets, Midway Network achieves strong performance on both semantic segmentation and optical flow tasks relative to prior self-supervised learning methods. We also show that Midway Network's learned dynamics can capture high-level correspondence via a novel analysis method based on forward feature perturbation.
>
---
#### [new 002] Beyond Spectral Peaks: Interpreting the Cues Behind Synthetic Image Detection
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文属于图像取证任务，旨在分析深度学习模型在检测合成图像时是否依赖频域峰值。论文通过去除频域峰值并测试其对检测结果的影响，发现多数模型不依赖这些峰值。此外，作者构建了一个仅依赖频域峰值的线性检测器，以提供可解释的对比方法。**

- **链接: [http://arxiv.org/pdf/2510.05633v1](http://arxiv.org/pdf/2510.05633v1)**

> **作者:** Sara Mandelli; Diego Vila-Portela; David Vázquez-Padín; Paolo Bestagini; Fernando Pérez-González
>
> **摘要:** Over the years, the forensics community has proposed several deep learning-based detectors to mitigate the risks of generative AI. Recently, frequency-domain artifacts (particularly periodic peaks in the magnitude spectrum), have received significant attention, as they have been often considered a strong indicator of synthetic image generation. However, state-of-the-art detectors are typically used as black-boxes, and it still remains unclear whether they truly rely on these peaks. This limits their interpretability and trust. In this work, we conduct a systematic study to address this question. We propose a strategy to remove spectral peaks from images and analyze the impact of this operation on several detectors. In addition, we introduce a simple linear detector that relies exclusively on frequency peaks, providing a fully interpretable baseline free from the confounding influence of deep learning. Our findings reveal that most detectors are not fundamentally dependent on spectral peaks, challenging a widespread assumption in the field and paving the way for more transparent and reliable forensic tools.
>
---
#### [new 003] Attention-Enhanced Prototypical Learning for Few-Shot Infrastructure Defect Segmentation
- **分类: cs.CV**

- **简介: 该论文属于基础设施缺陷分割任务，旨在解决小样本下新缺陷类别难以学习的问题。作者提出了E-FPN框架，结合原型学习与注意力机制，实现高效多尺度特征提取与强大原型生成。实验表明其在小样本设置下性能优越，提升了基础设施检测系统的效率与适应性。**

- **链接: [http://arxiv.org/pdf/2510.05266v1](http://arxiv.org/pdf/2510.05266v1)**

> **作者:** Christina Thrainer; Md Meftahul Ferdaus; Mahdi Abdelguerfi; Christian Guetl; Steven Sloan; Kendall N. Niles; Ken Pathak
>
> **摘要:** Few-shot semantic segmentation is vital for deep learning-based infrastructure inspection applications, where labeled training examples are scarce and expensive. Although existing deep learning frameworks perform well, the need for extensive labeled datasets and the inability to learn new defect categories with little data are problematic. We present our Enhanced Feature Pyramid Network (E-FPN) framework for few-shot semantic segmentation of culvert and sewer defect categories using a prototypical learning framework. Our approach has three main contributions: (1) adaptive E-FPN encoder using InceptionSepConv blocks and depth-wise separable convolutions for efficient multi-scale feature extraction; (2) prototypical learning with masked average pooling for powerful prototype generation from small support examples; and (3) attention-based feature representation through global self-attention, local self-attention and cross-attention. Comprehensive experimentation on challenging infrastructure inspection datasets illustrates that the method achieves excellent few-shot performance, with the best configuration being 8-way 5-shot training configuration at 82.55% F1-score and 72.26% mIoU in 2-way classification testing. The self-attention method had the most significant performance improvements, providing 2.57% F1-score and 2.9% mIoU gain over baselines. Our framework addresses the critical need to rapidly respond to new defect types in infrastructure inspection systems with limited new training data that lead to more efficient and economical maintenance plans for critical infrastructure systems.
>
---
#### [new 004] SD-MVSum: Script-Driven Multimodal Video Summarization Method and Datasets
- **分类: cs.CV**

- **简介: 该论文属于视频摘要任务，旨在解决如何结合用户提供的脚本与视频的视觉和语音内容生成更相关的视频摘要。作者提出了SD-MVSum方法，通过加权跨模态注意力机制建模脚本与视频、脚本与转录文本的关系，并扩展了两个数据集以支持该任务的研究。**

- **链接: [http://arxiv.org/pdf/2510.05652v1](http://arxiv.org/pdf/2510.05652v1)**

> **作者:** Manolis Mylonas; Charalampia Zerva; Evlampios Apostolidis; Vasileios Mezaris
>
> **备注:** Under review
>
> **摘要:** In this work, we extend a recent method for script-driven video summarization, originally considering just the visual content of the video, to take into account the relevance of the user-provided script also with the video's spoken content. In the proposed method, SD-MVSum, the dependence between each considered pair of data modalities, i.e., script-video and script-transcript, is modeled using a new weighted cross-modal attention mechanism. This explicitly exploits the semantic similarity between the paired modalities in order to promote the parts of the full-length video with the highest relevance to the user-provided script. Furthermore, we extend two large-scale datasets for video summarization (S-VideoXum, MrHiSum), to make them suitable for training and evaluation of script-driven multimodal video summarization methods. Experimental comparisons document the competitiveness of our SD-MVSum method against other SOTA approaches for script-driven and generic video summarization. Our new method and extended datasets are available at: https://github.com/IDT-ITI/SD-MVSum.
>
---
#### [new 005] acia-workflows: Automated Single-cell Imaging Analysis for Scalable and Deep Learning-based Live-cell Imaging Analysis Workflows
- **分类: cs.CV; q-bio.QM**

- **简介: 该论文属于生物成像分析任务，旨在解决高通量活细胞成像数据难以高效处理的问题。作者开发了acia-workflows平台，集成深度学习模型与分析流程，实现自动化、可扩展的单细胞动态研究，提升生物实验的可重复性与实用性。**

- **链接: [http://arxiv.org/pdf/2510.05886v1](http://arxiv.org/pdf/2510.05886v1)**

> **作者:** Johannes Seiffarth; Keitaro Kasahara; Michelle Bund; Benita Lückel; Richard D. Paul; Mathias Pesch; Lennart Witting; Michael Bott; Dietrich Kohlheyer; Katharina Nöh
>
> **摘要:** Live-cell imaging (LCI) technology enables the detailed spatio-temporal characterization of living cells at the single-cell level, which is critical for advancing research in the life sciences, from biomedical applications to bioprocessing. High-throughput setups with tens to hundreds of parallel cell cultivations offer the potential for robust and reproducible insights. However, these insights are obscured by the large amount of LCI data recorded per experiment. Recent advances in state-of-the-art deep learning methods for cell segmentation and tracking now enable the automated analysis of such large data volumes, offering unprecedented opportunities to systematically study single-cell dynamics. The next key challenge lies in integrating these powerful tools into accessible, flexible, and user-friendly workflows that support routine application in biological research. In this work, we present acia-workflows, a platform that combines three key components: (1) the Automated live-Cell Imaging Analysis (acia) Python library, which supports the modular design of image analysis pipelines offering eight deep learning segmentation and tracking approaches; (2) workflows that assemble the image analysis pipeline, its software dependencies, documentation, and visualizations into a single Jupyter Notebook, leading to accessible, reproducible and scalable analysis workflows; and (3) a collection of application workflows showcasing the analysis and customization capabilities in real-world applications. Specifically, we present three workflows to investigate various types of microfluidic LCI experiments ranging from growth rate comparisons to precise, minute-resolution quantitative analyses of individual dynamic cells responses to changing oxygen conditions. Our collection of more than ten application workflows is open source and publicly available at https://github.com/JuBiotech/acia-workflows.
>
---
#### [new 006] ShapeGen4D: Towards High Quality 4D Shape Generation from Videos
- **分类: cs.CV**

- **简介: 该论文属于视频驱动的4D形状生成任务，旨在从输入视频中直接生成随时间变化的高质量3D几何和外观。为解决时间一致性差、细节模糊等问题，论文提出ShapeGen4D框架，包含时态注意力、时序点采样与噪声共享三大技术，实现端到端动态3D表示生成，有效捕捉非刚性运动、体积变化和拓扑转换。**

- **链接: [http://arxiv.org/pdf/2510.06208v1](http://arxiv.org/pdf/2510.06208v1)**

> **作者:** Jiraphon Yenphraphai; Ashkan Mirzaei; Jianqi Chen; Jiaxu Zou; Sergey Tulyakov; Raymond A. Yeh; Peter Wonka; Chaoyang Wang
>
> **备注:** Project page: https://shapegen4d.github.io/
>
> **摘要:** Video-conditioned 4D shape generation aims to recover time-varying 3D geometry and view-consistent appearance directly from an input video. In this work, we introduce a native video-to-4D shape generation framework that synthesizes a single dynamic 3D representation end-to-end from the video. Our framework introduces three key components based on large-scale pre-trained 3D models: (i) a temporal attention that conditions generation on all frames while producing a time-indexed dynamic representation; (ii) a time-aware point sampling and 4D latent anchoring that promote temporally consistent geometry and texture; and (iii) noise sharing across frames to enhance temporal stability. Our method accurately captures non-rigid motion, volume changes, and even topological transitions without per-frame optimization. Across diverse in-the-wild videos, our method improves robustness and perceptual fidelity and reduces failure modes compared with the baselines.
>
---
#### [new 007] AgeBooth: Controllable Facial Aging and Rejuvenation via Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决基于单张参考图进行可控人脸老化与年轻化生成的问题。现有方法难以精准控制年龄且依赖成对数据，论文提出AgeBooth，通过年龄条件提示混合与LoRA融合策略，实现高质量、身份一致的年龄编辑，无需大量跨年龄配对数据。**

- **链接: [http://arxiv.org/pdf/2510.05715v1](http://arxiv.org/pdf/2510.05715v1)**

> **作者:** Shihao Zhu; Bohan Cao; Ziheng Ouyang; Zhen Li; Peng-Tao Jiang; Qibin Hou
>
> **摘要:** Recent diffusion model research focuses on generating identity-consistent images from a reference photo, but they struggle to accurately control age while preserving identity, and fine-tuning such models often requires costly paired images across ages. In this paper, we propose AgeBooth, a novel age-specific finetuning approach that can effectively enhance the age control capability of adapterbased identity personalization models without the need for expensive age-varied datasets. To reduce dependence on a large amount of age-labeled data, we exploit the linear nature of aging by introducing age-conditioned prompt blending and an age-specific LoRA fusion strategy that leverages SVDMix, a matrix fusion technique. These techniques enable high-quality generation of intermediate-age portraits. Our AgeBooth produces realistic and identity-consistent face images across different ages from a single reference image. Experiments show that AgeBooth achieves superior age control and visual quality compared to previous state-of-the-art editing-based methods.
>
---
#### [new 008] Bimanual 3D Hand Motion and Articulation Forecasting in Everyday Images
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于计算机视觉与动作预测任务，旨在从单张日常图像中预测双手的3D运动与关节变化。为解决缺乏多样场景的3D标注数据问题，作者设计了基于扩散模型的标注流程，将2D关键点序列提升至4D手部运动，并构建了适用于多模态手部动作预测的模型，提升了预测准确性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.06145v1](http://arxiv.org/pdf/2510.06145v1)**

> **作者:** Aditya Prakash; David Forsyth; Saurabh Gupta
>
> **备注:** Project page: https://ap229997.github.io/projects/forehand4d
>
> **摘要:** We tackle the problem of forecasting bimanual 3D hand motion & articulation from a single image in everyday settings. To address the lack of 3D hand annotations in diverse settings, we design an annotation pipeline consisting of a diffusion model to lift 2D hand keypoint sequences to 4D hand motion. For the forecasting model, we adopt a diffusion loss to account for the multimodality in hand motion distribution. Extensive experiments across 6 datasets show the benefits of training on diverse data with imputed labels (14% improvement) and effectiveness of our lifting (42% better) & forecasting (16.4% gain) models, over the best baselines, especially in zero-shot generalization to everyday images.
>
---
#### [new 009] Discrete Diffusion Models with MLLMs for Unified Medical Multimodal Generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学多模态生成任务，旨在解决现有模型因模态分离导致的生成限制。论文提出MeDiM，首个基于离散扩散模型与多模态大语言模型（MLLM）的统一医学生成框架，实现图像与文本间的双向生成及联合生成。通过去除因果注意力掩码和引入时间步嵌入，提升跨模态推理与生成质量，实验证明其在医学图像与报告生成中的有效性。**

- **链接: [http://arxiv.org/pdf/2510.06131v1](http://arxiv.org/pdf/2510.06131v1)**

> **作者:** Jiawei Mao; Yuhan Wang; Lifeng Chen; Can Zhao; Yucheng Tang; Dong Yang; Liangqiong Qu; Daguang Xu; Yuyin Zhou
>
> **备注:** 16 pages,6 figures
>
> **摘要:** Recent advances in generative medical models are constrained by modality-specific scenarios that hinder the integration of complementary evidence from imaging, pathology, and clinical notes. This fragmentation limits their evolution into foundation models that can learn and reason across the full spectrum of biomedical data. We propose MeDiM, the first medical discrete diffusion model that learns shared distributions across modalities without modality-specific components. MeDiM unifies multiple generative tasks: translating between images and text, and jointly producing image-report pairs across domains in response to prompts. Built on a discrete diffusion framework, MeDiM bridges vision and language representations through a shared probabilistic space. To enable unified and flexible medical generation, we employ a multimodal large language model (MLLM) as the diffusion backbone, leveraging its prior knowledge and cross-modal reasoning. Two key designs are introduced: (1) removing the causal attention mask for bidirectional context, and (2) injecting continuous timestep embeddings for diffusion awareness. Experiments demonstrate high-fidelity medical generation (FID 16.60 on MIMIC-CXR and FID 24.19 on PathGen) and accurate report generation (METEOR 0.2650 and 0.2580). Jointly generated image-report pairs further enhance downstream performance (plus6.43 percent BLEU-1, plus18.57 percent BLEU-2, plus31.58 percent BLEU-3, plus4.80 percent METEOR), showing that MeDiM supports coherent and clinically grounded multimodal outputs.
>
---
#### [new 010] Diffusion-Based Image Editing for Breaking Robust Watermarks
- **分类: cs.CV**

- **简介: 该论文研究扩散模型对鲁棒图像水印的攻击。任务是分析现有水印方案在生成模型下的安全性。作者提出扩散驱动的图像再生和引导攻击方法，以去除水印并验证其效果，揭示当前水印技术的不足。**

- **链接: [http://arxiv.org/pdf/2510.05978v1](http://arxiv.org/pdf/2510.05978v1)**

> **作者:** Yunyi Ni; Finn Carter; Ze Niu; Emily Davis; Bo Zhang
>
> **备注:** Preprint
>
> **摘要:** Robust invisible watermarking aims to embed hidden information into images such that the watermark can survive various image manipulations. However, the rise of powerful diffusion-based image generation and editing techniques poses a new threat to these watermarking schemes. In this paper, we present a theoretical study and method demonstrating that diffusion models can effectively break robust image watermarks that were designed to resist conventional perturbations. We show that a diffusion-driven ``image regeneration'' process can erase embedded watermarks while preserving perceptual image content. We further introduce a novel guided diffusion attack that explicitly targets the watermark signal during generation, significantly degrading watermark detectability. Theoretically, we prove that as an image undergoes sufficient diffusion-based transformation, the mutual information between the watermarked image and the embedded watermark payload vanishes, resulting in decoding failure. Experimentally, we evaluate our approach on multiple state-of-the-art watermarking schemes (including the deep learning-based methods StegaStamp, TrustMark, and VINE) and demonstrate near-zero watermark recovery rates after attack, while maintaining high visual fidelity of the regenerated images. Our findings highlight a fundamental vulnerability in current robust watermarking techniques against generative model-based attacks, underscoring the need for new watermarking strategies in the era of generative AI.
>
---
#### [new 011] There is More to Attention: Statistical Filtering Enhances Explanations in Vision Transformers
- **分类: cs.CV**

- **简介: 该论文属于可解释人工智能（XAI）任务，旨在提升视觉Transformer（ViT）的解释性。现有方法依赖注意力权重，但解释结果常含噪声。论文提出结合统计过滤的注意力解释方法，去除无关模式，生成更清晰、符合人类感知的解释图，并在多个数据集上验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2510.06070v1](http://arxiv.org/pdf/2510.06070v1)**

> **作者:** Meghna P Ayyar; Jenny Benois-Pineau; Akka Zemmari
>
> **摘要:** Explainable AI (XAI) has become increasingly important with the rise of large transformer models, yet many explanation methods designed for CNNs transfer poorly to Vision Transformers (ViTs). Existing ViT explanations often rely on attention weights, which tend to yield noisy maps as they capture token-to-token interactions within each layer.While attribution methods incorporating MLP blocks have been proposed, we argue that attention remains a valuable and interpretable signal when properly filtered. We propose a method that combines attention maps with a statistical filtering, initially proposed for CNNs, to remove noisy or uninformative patterns and produce more faithful explanations. We further extend our approach with a class-specific variant that yields discriminative explanations. Evaluation against popular state-of-the-art methods demonstrates that our approach produces sharper and more interpretable maps. In addition to perturbation-based faithfulness metrics, we incorporate human gaze data to assess alignment with human perception, arguing that human interpretability remains essential for XAI. Across multiple datasets, our approach consistently outperforms or is comparable to the SOTA methods while remaining efficient and human plausible.
>
---
#### [new 012] Shaken or Stirred? An Analysis of MetaFormer's Token Mixing for Medical Imaging
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决MetaFormer架构中不同token mixer在医学图像分类与分割任务中的适用性问题。作者系统评估了池化、卷积和注意力机制等token mixer，并分析了预训练权重的迁移效果。结果指出低复杂度mixer适合分类，卷积mixer更适合分割，且预训练权重在域差异下仍有效。**

- **链接: [http://arxiv.org/pdf/2510.05971v1](http://arxiv.org/pdf/2510.05971v1)**

> **作者:** Ron Keuth; Paul Kaftan; Mattias P. Heinrich
>
> **备注:** Code and data: https://github.com/multimodallearning/MetaFormerMedImaging/tree/clean_code
>
> **摘要:** The generalization of the Transformer architecture via MetaFormer has reshaped our understanding of its success in computer vision. By replacing self-attention with simpler token mixers, MetaFormer provides strong baselines for vision tasks. However, while extensively studied on natural image datasets, its use in medical imaging remains scarce, and existing works rarely compare different token mixers, potentially overlooking more suitable designs choices. In this work, we present the first comprehensive study of token mixers for medical imaging. We systematically analyze pooling-, convolution-, and attention-based token mixers within the MetaFormer architecture on image classification (global prediction task) and semantic segmentation (dense prediction task). Our evaluation spans eight datasets covering diverse modalities and common challenges in the medical domain. Given the prevalence of pretraining from natural images to mitigate medical data scarcity, we also examine transferring pretrained weights to new token mixers. Our results show that, for classification, low-complexity token mixers (e.g. grouped convolution or pooling) are sufficient, aligning with findings on natural images. Pretrained weights remain useful despite the domain gap introduced by the new token mixer. For segmentation, we find that the local inductive bias of convolutional token mixers is essential. Grouped convolutions emerge as the preferred choice, as they reduce runtime and parameter count compared to standard convolutions, while the MetaFormer's channel-MLPs already provide the necessary cross-channel interactions. Our code is available on GitHub.
>
---
#### [new 013] Drive&Gen: Co-Evaluating End-to-End Driving and Video Generation Models
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶与生成模型交叉任务，旨在解决如何有效评估端到端（E2E）自动驾驶模型在生成虚拟环境中的表现。论文提出Drive&Gen，利用E2E驾驶模型评估视频生成模型的现实性，并通过合成数据提升E2E模型在新场景中的泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.06209v1](http://arxiv.org/pdf/2510.06209v1)**

> **作者:** Jiahao Wang; Zhenpei Yang; Yijing Bai; Yingwei Li; Yuliang Zou; Bo Sun; Abhijit Kundu; Jose Lezama; Luna Yue Huang; Zehao Zhu; Jyh-Jing Hwang; Dragomir Anguelov; Mingxing Tan; Chiyu Max Jiang
>
> **备注:** Accepted by IROS 2025
>
> **摘要:** Recent advances in generative models have sparked exciting new possibilities in the field of autonomous vehicles. Specifically, video generation models are now being explored as controllable virtual testing environments. Simultaneously, end-to-end (E2E) driving models have emerged as a streamlined alternative to conventional modular autonomous driving systems, gaining popularity for their simplicity and scalability. However, the application of these techniques to simulation and planning raises important questions. First, while video generation models can generate increasingly realistic videos, can these videos faithfully adhere to the specified conditions and be realistic enough for E2E autonomous planner evaluation? Second, given that data is crucial for understanding and controlling E2E planners, how can we gain deeper insights into their biases and improve their ability to generalize to out-of-distribution scenarios? In this work, we bridge the gap between the driving models and generative world models (Drive&Gen) to address these questions. We propose novel statistical measures leveraging E2E drivers to evaluate the realism of generated videos. By exploiting the controllability of the video generation model, we conduct targeted experiments to investigate distribution gaps affecting E2E planner performance. Finally, we show that synthetic data produced by the video generation model offers a cost-effective alternative to real-world data collection. This synthetic data effectively improves E2E model generalization beyond existing Operational Design Domains, facilitating the expansion of autonomous vehicle services into new operational contexts.
>
---
#### [new 014] A Dynamic Mode Decomposition Approach to Morphological Component Analysis
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于信号处理任务，旨在解决视频和图像中不同形态成分的分离问题。作者提出动态形态成分分析（DMCA）方法，通过动态模态分解与特征值聚类，实现自适应字典学习，提升了去噪与目标增强效果。**

- **链接: [http://arxiv.org/pdf/2510.05977v1](http://arxiv.org/pdf/2510.05977v1)**

> **作者:** Owen T. Huber; Raghu G. Raj; Tianyu Chen; Zacharie I. Idriss
>
> **摘要:** This paper introduces a novel methodology of adapting the representation of videos based on the dynamics of their scene content variation. In particular, we demonstrate how the clustering of dynamic mode decomposition eigenvalues can be leveraged to learn an adaptive video representation for separating structurally distinct morphologies of a video. We extend the morphological component analysis (MCA) algorithm, which uses multiple predefined incoherent dictionaries and a sparsity prior to separate distinct sources in signals, by introducing our novel eigenspace clustering technique to obtain data-driven MCA dictionaries, which we call dynamic morphological component analysis (DMCA). After deriving our novel algorithm, we offer a motivational example of DMCA applied to a still image, then demonstrate DMCA's effectiveness in denoising applications on videos from the Adobe 240fps dataset. Afterwards, we provide an example of DMCA enhancing the signal-to-noise ratio of a faint target summed with a sea state, and conclude the paper by applying DMCA to separate a bicycle from wind clutter in inverse synthetic aperture radar images.
>
---
#### [new 015] Reasoning under Vision: Understanding Visual-Spatial Cognition in Vision-Language Models for CAPTCHA
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉-语言模型的视觉空间推理任务，旨在解决当前模型在CAPTCHA验证码识别中推理能力不足的问题。论文提出CAPTCHA-X基准和推理指标，结合逐步推理框架，显著提升模型准确率至83.9%。**

- **链接: [http://arxiv.org/pdf/2510.06067v1](http://arxiv.org/pdf/2510.06067v1)**

> **作者:** Python Song; Luke Tenyi Chang; Yun-Yun Tsai; Penghui Li; Junfeng Yang
>
> **备注:** 14pages, 11figures
>
> **摘要:** CAPTCHA, originally designed to distinguish humans from robots, has evolved into a real-world benchmark for assessing the spatial reasoning capabilities of vision-language models. In this work, we first show that step-by-step reasoning is crucial for vision-language models (VLMs) to solve CAPTCHAs, which represent high-difficulty spatial reasoning tasks, and that current commercial vision-language models still struggle with such reasoning. In particular, we observe that most commercial VLMs (e.g., Gemini, Claude, GPT, etc.) fail to effectively solve CAPTCHAs and thus achieve low accuracy (around 21.9 percent). However, our findings indicate that requiring the model to perform step-by-step reasoning before generating the final coordinates can significantly enhance its solving accuracy, underscoring the severity of the gap. To systematically study this issue, we introduce CAPTCHA-X, the first real-world CAPTCHA benchmark with reasoning, covering seven categories of CAPTCHAs (such as Gobang, hCaptcha, etc.) with step-by-step action solutions and grounding annotations. We further define five reasoning-oriented metrics that enable a comprehensive evaluation of models reasoning capabilities. To validate the effectiveness of reasoning, we also propose a general agentic VLM-based framework that incorporates the models inherent reasoning abilities. Our method achieves state-of-the-art performance across five high-difficulty CAPTCHA types, with an average solving accuracy of 83.9 percent, substantially surpassing existing baselines. These results reveal the limitations of current models and highlight the importance of reasoning in advancing visual-spatial challenges in the future.
>
---
#### [new 016] Kaputt: A Large-Scale Dataset for Visual Defect Detection
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于视觉缺陷检测任务，旨在解决零售物流中因物体姿态和外观变化大而导致的异常检测难题。现有数据集如MVTec-AD已无法满足需求，因其实验结果显示顶尖方法在此新场景下表现不佳。论文作者构建了一个大规模数据集Kaputt，包含超23万张图像、48,000个不同物体，用于推动相关研究。**

- **链接: [http://arxiv.org/pdf/2510.05903v1](http://arxiv.org/pdf/2510.05903v1)**

> **作者:** Sebastian Höfer; Dorian Henning; Artemij Amiranashvili; Douglas Morrison; Mariliza Tzes; Ingmar Posner; Marc Matvienko; Alessandro Rennola; Anton Milan
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** We present a novel large-scale dataset for defect detection in a logistics setting. Recent work on industrial anomaly detection has primarily focused on manufacturing scenarios with highly controlled poses and a limited number of object categories. Existing benchmarks like MVTec-AD [6] and VisA [33] have reached saturation, with state-of-the-art methods achieving up to 99.9% AUROC scores. In contrast to manufacturing, anomaly detection in retail logistics faces new challenges, particularly in the diversity and variability of object pose and appearance. Leading anomaly detection methods fall short when applied to this new setting. To bridge this gap, we introduce a new benchmark that overcomes the current limitations of existing datasets. With over 230,000 images (and more than 29,000 defective instances), it is 40 times larger than MVTec-AD and contains more than 48,000 distinct objects. To validate the difficulty of the problem, we conduct an extensive evaluation of multiple state-of-the-art anomaly detection methods, demonstrating that they do not surpass 56.96% AUROC on our dataset. Further qualitative analysis confirms that existing methods struggle to leverage normal samples under heavy pose and appearance variation. With our large-scale dataset, we set a new benchmark and encourage future research towards solving this challenging problem in retail logistics anomaly detection. The dataset is available for download under https://www.kaputt-dataset.com.
>
---
#### [new 017] PointNSP: Autoregressive 3D Point Cloud Generation with Next-Scale Level-of-Detail Prediction
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于3D点云生成任务，旨在解决自回归模型在生成质量上落后于扩散模型的问题。论文提出PointNSP，采用由粗到细的生成框架，通过下一级细节层次预测来保留全局结构并逐步细化几何细节，从而提升生成质量和效率。**

- **链接: [http://arxiv.org/pdf/2510.05613v1](http://arxiv.org/pdf/2510.05613v1)**

> **作者:** Ziqiao Meng; Qichao Wang; Zhiyang Dou; Zixing Song; Zhipeng Zhou; Irwin King; Peilin Zhao
>
> **摘要:** Autoregressive point cloud generation has long lagged behind diffusion-based approaches in quality. The performance gap stems from the fact that autoregressive models impose an artificial ordering on inherently unordered point sets, forcing shape generation to proceed as a sequence of local predictions. This sequential bias emphasizes short-range continuity but undermines the model's capacity to capture long-range dependencies, hindering its ability to enforce global structural properties such as symmetry, consistent topology, and large-scale geometric regularities. Inspired by the level-of-detail (LOD) principle in shape modeling, we propose PointNSP, a coarse-to-fine generative framework that preserves global shape structure at low resolutions and progressively refines fine-grained geometry at higher scales through a next-scale prediction paradigm. This multi-scale factorization aligns the autoregressive objective with the permutation-invariant nature of point sets, enabling rich intra-scale interactions while avoiding brittle fixed orderings. Experiments on ShapeNet show that PointNSP establishes state-of-the-art (SOTA) generation quality for the first time within the autoregressive paradigm. In addition, it surpasses strong diffusion-based baselines in parameter, training, and inference efficiency. Finally, in dense generation with 8,192 points, PointNSP's advantages become even more pronounced, underscoring its scalability potential.
>
---
#### [new 018] GLVD: Guided Learned Vertex Descent
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于3D人脸重建任务，旨在解决现有方法依赖固定形状先验或计算昂贵的问题。论文提出GLVD方法，结合神经场优化与全局关键点引导，实现高效、高质量的少样本3D人脸重建，无需密集3D监督。**

- **链接: [http://arxiv.org/pdf/2510.06046v1](http://arxiv.org/pdf/2510.06046v1)**

> **作者:** Pol Caselles Rico; Francesc Moreno Noguer
>
> **摘要:** Existing 3D face modeling methods usually depend on 3D Morphable Models, which inherently constrain the representation capacity to fixed shape priors. Optimization-based approaches offer high-quality reconstructions but tend to be computationally expensive. In this work, we introduce GLVD, a hybrid method for 3D face reconstruction from few-shot images that extends Learned Vertex Descent (LVD) by integrating per-vertex neural field optimization with global structural guidance from dynamically predicted 3D keypoints. By incorporating relative spatial encoding, GLVD iteratively refines mesh vertices without requiring dense 3D supervision. This enables expressive and adaptable geometry reconstruction while maintaining computational efficiency. GLVD achieves state-of-the-art performance in single-view settings and remains highly competitive in multi-view scenarios, all while substantially reducing inference time.
>
---
#### [new 019] Efficient Conditional Generation on Scale-based Visual Autoregressive Models
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决基于条件的图像生成中训练成本高、控制不够高效的问题。作者提出了Efficient Control Model（ECM），通过引入轻量级控制模块和分布式架构，结合早期采样策略与推理阶段的温度调度，在保证生成质量的同时显著提升了训练和推理效率。**

- **链接: [http://arxiv.org/pdf/2510.05610v1](http://arxiv.org/pdf/2510.05610v1)**

> **作者:** Jiaqi Liu; Tao Huang; Chang Xu
>
> **摘要:** Recent advances in autoregressive (AR) models have demonstrated their potential to rival diffusion models in image synthesis. However, for complex spatially-conditioned generation, current AR approaches rely on fine-tuning the pre-trained model, leading to significant training costs. In this paper, we propose the Efficient Control Model (ECM), a plug-and-play framework featuring a lightweight control module that introduces control signals via a distributed architecture. This architecture consists of context-aware attention layers that refine conditional features using real-time generated tokens, and a shared gated feed-forward network (FFN) designed to maximize the utilization of its limited capacity and ensure coherent control feature learning. Furthermore, recognizing the critical role of early-stage generation in determining semantic structure, we introduce an early-centric sampling strategy that prioritizes learning early control sequences. This approach reduces computational cost by lowering the number of training tokens per iteration, while a complementary temperature scheduling during inference compensates for the resulting insufficient training of late-stage tokens. Extensive experiments on scale-based AR models validate that our method achieves high-fidelity and diverse control over image generation, surpassing existing baselines while significantly improving both training and inference efficiency.
>
---
#### [new 020] Rasterized Steered Mixture of Experts for Efficient 2D Image Regression
- **分类: cs.CV**

- **简介: 论文属于图像处理任务，旨在解决Steered Mixture of Experts方法因计算成本高而难以应用的问题。工作提出基于光栅化的优化策略，结合高斯核渲染效率与边缘感知机制，加速图像回归，同时保持稀疏性与重建质量，支持超分辨率和去噪等应用。**

- **链接: [http://arxiv.org/pdf/2510.05814v1](http://arxiv.org/pdf/2510.05814v1)**

> **作者:** Yi-Hsin Li; Thomas Sikora; Sebastian Knorr; Mårten Sjöström
>
> **摘要:** The Steered Mixture of Experts regression framework has demonstrated strong performance in image reconstruction, compression, denoising, and super-resolution. However, its high computational cost limits practical applications. This work introduces a rasterization-based optimization strategy that combines the efficiency of rasterized Gaussian kernel rendering with the edge-aware gating mechanism of the Steered Mixture of Experts. The proposed method is designed to accelerate two-dimensional image regression while maintaining the model's inherent sparsity and reconstruction quality. By replacing global iterative optimization with a rasterized formulation, the method achieves significantly faster parameter updates and more memory-efficient model representations. In addition, the proposed framework supports applications such as native super-resolution and image denoising, which are not directly achievable with standard rasterized Gaussian kernel approaches. The combination of fast rasterized optimization with the edge-aware structure of the Steered Mixture of Experts provides a new balance between computational efficiency and reconstruction fidelity for two-dimensional image processing tasks.
>
---
#### [new 021] TFM Dataset: A Novel Multi-task Dataset and Integrated Pipeline for Automated Tear Film Break-Up Segmentation
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决干眼症诊断中泪膜破裂（TFBU）自动分割的难题。论文构建了首个多任务泪膜分析数据集TFM，包含15个高分辨率视频及多任务标注，并提出了高效分割模型TF-Net和集成分析流程TF-Collab，实现了TFBU的自动化分析。**

- **链接: [http://arxiv.org/pdf/2510.05615v1](http://arxiv.org/pdf/2510.05615v1)**

> **作者:** Guangrong Wan; Jun liu; Tang tang; Lianghao Shi; Wenjun Luo; TingTing Xu
>
> **摘要:** Tear film break-up (TFBU) analysis is critical for diagnosing dry eye syndrome, but automated TFBU segmentation remains challenging due to the lack of annotated datasets and integrated solutions. This paper introduces the Tear Film Multi-task (TFM) Dataset, the first comprehensive dataset for multi-task tear film analysis, comprising 15 high-resolution videos (totaling 6,247 frames) annotated with three vision tasks: frame-level classification ('clear', 'closed', 'broken', 'blur'), Placido Ring detection, and pixel-wise TFBU area segmentation. Leveraging this dataset, we first propose TF-Net, a novel and efficient baseline segmentation model. TF-Net incorporates a MobileOne-mini backbone with re-parameterization techniques and an enhanced feature pyramid network to achieve a favorable balance between accuracy and computational efficiency for real-time clinical applications. We further establish benchmark performance on the TFM segmentation subset by comparing TF-Net against several state-of-the-art medical image segmentation models. Furthermore, we design TF-Collab, a novel integrated real-time pipeline that synergistically leverages models trained on all three tasks of the TFM dataset. By sequentially orchestrating frame classification for BUT determination, pupil region localization for input standardization, and TFBU segmentation, TF-Collab fully automates the analysis. Experimental results demonstrate the effectiveness of the proposed TF-Net and TF-Collab, providing a foundation for future research in ocular surface diagnostics. Our code and the TFM datasets are available at https://github.com/glory-wan/TF-Net
>
---
#### [new 022] Data Factory with Minimal Human Effort Using VLMs
- **分类: cs.CV**

- **简介: 该论文属于图像数据增强任务，旨在解决传统方法在语义属性操作上的不足及扩散模型的高计算成本问题。论文提出了一种无需训练的合成图像生成方法，结合ControlNet和VLMs，并引入多模块优化，实现高质量带标签图像生成，提升下游任务表现。**

- **链接: [http://arxiv.org/pdf/2510.05722v1](http://arxiv.org/pdf/2510.05722v1)**

> **作者:** Jiaojiao Ye; Jiaxing Zhong; Qian Xie; Yuzhou Zhou; Niki Trigoni; Andrew Markham
>
> **备注:** Tech report
>
> **摘要:** Generating enough and diverse data through augmentation offers an efficient solution to the time-consuming and labour-intensive process of collecting and annotating pixel-wise images. Traditional data augmentation techniques often face challenges in manipulating high-level semantic attributes, such as materials and textures. In contrast, diffusion models offer a robust alternative, by effectively utilizing text-to-image or image-to-image transformation. However, existing diffusion-based methods are either computationally expensive or compromise on performance. To address this issue, we introduce a novel training-free pipeline that integrates pretrained ControlNet and Vision-Language Models (VLMs) to generate synthetic images paired with pixel-level labels. This approach eliminates the need for manual annotations and significantly improves downstream tasks. To improve the fidelity and diversity, we add a Multi-way Prompt Generator, Mask Generator and High-quality Image Selection module. Our results on PASCAL-5i and COCO-20i present promising performance and outperform concurrent work for one-shot semantic segmentation.
>
---
#### [new 023] InstaGeo: Compute-Efficient Geospatial Machine Learning from Data to Deployment
- **分类: cs.CV; cs.CY; cs.LG**

- **简介: 该论文属于遥感图像处理任务，旨在解决现有地理空间基础模型部署困难的问题。论文提出InstaGeo框架，整合自动化数据处理、模型压缩和部署功能，实现高效、低能耗的地理空间机器学习。**

- **链接: [http://arxiv.org/pdf/2510.05617v1](http://arxiv.org/pdf/2510.05617v1)**

> **作者:** Ibrahim Salihu Yusuf; Iffanice Houndayi; Rym Oualha; Mohamed Aziz Cherif; Kobby Panford-Quainoo; Arnu Pretorius
>
> **摘要:** Open-access multispectral imagery from missions like Landsat 8-9 and Sentinel-2 has fueled the development of geospatial foundation models (GFMs) for humanitarian and environmental applications. Yet, their deployment remains limited by (i) the absence of automated geospatial data pipelines and (ii) the large size of fine-tuned models. Existing GFMs lack workflows for processing raw satellite imagery, and downstream adaptations often retain the full complexity of the original encoder. We present InstaGeo, an open-source, end-to-end framework that addresses these challenges by integrating: (1) automated data curation to transform raw imagery into model-ready datasets; (2) task-specific model distillation to derive compact, compute-efficient models; and (3) seamless deployment as interactive web-map applications. Using InstaGeo, we reproduced datasets from three published studies and trained models with marginal mIoU differences of -0.73 pp for flood mapping, -0.20 pp for crop segmentation, and +1.79 pp for desert locust prediction. The distilled models are up to 8x smaller than standard fine-tuned counterparts, reducing FLOPs and CO2 emissions with minimal accuracy loss. Leveraging InstaGeo's streamlined data pipeline, we also curated a larger crop segmentation dataset, achieving a state-of-the-art mIoU of 60.65%, a 12 pp improvement over prior baselines. Moreover, InstaGeo enables users to progress from raw data to model deployment within a single working day. By unifying data preparation, model compression, and deployment, InstaGeo transforms research-grade GFMs into practical, low-carbon tools for real-time, large-scale Earth observation. This approach shifts geospatial AI toward data quality and application-driven innovation. Source code, datasets, and model checkpoints are available at: https://github.com/instadeepai/InstaGeo-E2E-Geospatial-ML.git
>
---
#### [new 024] Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow
- **分类: cs.CV**

- **简介: 该论文属于视频理解任务，旨在解决长视频中的时空冗余问题。提出Flow4Agent框架，利用光流运动先验，通过时间粒度优化和运动标记剪枝，提升多模态大模型的长视频理解能力。**

- **链接: [http://arxiv.org/pdf/2510.05836v1](http://arxiv.org/pdf/2510.05836v1)**

> **作者:** Ruyang Liu; Shangkun Sun; Haoran Tang; Ge Li; Wei Gao
>
> **备注:** Accepted to ICCV' 2025
>
> **摘要:** Long-form video understanding has always been a challenging problem due to the significant redundancy in both temporal and spatial contents. This challenge is further exacerbated by the limited context length of Multimodal Large Language Models (MLLMs). To address this issue, many previous works have attempted to extract key video information, where the "key" is typically semantic-aware and heavily dependent on the CLIP model as prior. In this paper, we propose Flow4Agent, a novel framework that pioneeringly incorporates motion priors from optical flow to facilitate LLM-based long video understanding. Flow4Agent mitigates the redundancy in long videos at both temporal and spatial levels through two core modules: Temporal Granularity Optimization (TGO) adaptively refines framelevel hierarchies, which first leverages coarse flow priors to group similar visual contents and then applies semantic priors to filter out highly irrelevant scene information. Motion Token Pruning (MTP) further refines the intra-frame visual representations, pruning high-redundancy video tokens using fine-grained optical flow information. Extensive experiments demonstrate that our Flow4Agent outperforms existing methods across a wide range of video MLLM benchmarks, especially for hour-level video understanding tasks, achieving 64.7% on Video-MME, 71.4% on MLVU and 60.4% on LongVideoBench.
>
---
#### [new 025] Mitigating Diffusion Model Hallucinations with Dynamic Guidance
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于图像生成任务，旨在解决扩散模型生成图像时出现的结构不一致和虚假内容问题。通过引入动态引导方法，在生成过程中选择性增强关键方向的细节，减少虚假内容，同时保留合理的语义变化，有效缓解生成结果的失真现象。**

- **链接: [http://arxiv.org/pdf/2510.05356v1](http://arxiv.org/pdf/2510.05356v1)**

> **作者:** Kostas Triaridis; Alexandros Graikos; Aggelina Chatziagapi; Grigorios G. Chrysos; Dimitris Samaras
>
> **摘要:** Diffusion models, despite their impressive demos, often produce hallucinatory samples with structural inconsistencies that lie outside of the support of the true data distribution. Such hallucinations can be attributed to excessive smoothing between modes of the data distribution. However, semantic interpolations are often desirable and can lead to generation diversity, thus we believe a more nuanced solution is required. In this work, we introduce Dynamic Guidance, which tackles this issue. Dynamic Guidance mitigates hallucinations by selectively sharpening the score function only along the pre-determined directions known to cause artifacts, while preserving valid semantic variations. To our knowledge, this is the first approach that addresses hallucinations at generation time rather than through post-hoc filtering. Dynamic Guidance substantially reduces hallucinations on both controlled and natural image datasets, significantly outperforming baselines.
>
---
#### [new 026] Redefining Generalization in Visual Domains: A Two-Axis Framework for Fake Image Detection with FusionDetect
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像检测任务，旨在解决生成图像检测中的跨生成器与跨视觉域泛化问题。作者提出了FusionDetect方法，结合CLIP与Dinov2模型提取互补特征，并构建了OmniGen基准数据集用于评估。实验表明其方法在检测准确率和鲁棒性方面均优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.05740v1](http://arxiv.org/pdf/2510.05740v1)**

> **作者:** Amirtaha Amanzadi; Zahra Dehghanian; Hamid Beigy; Hamid R. Rabiee
>
> **备注:** Project code: http://github.com/amir-aman/FusionDetect
>
> **摘要:** The rapid development of generative models has made it increasingly crucial to develop detectors that can reliably detect synthetic images. Although most of the work has now focused on cross-generator generalization, we argue that this viewpoint is too limited. Detecting synthetic images involves another equally important challenge: generalization across visual domains. To bridge this gap,we present the OmniGen Benchmark. This comprehensive evaluation dataset incorporates 12 state-of-the-art generators, providing a more realistic way of evaluating detector performance under realistic conditions. In addition, we introduce a new method, FusionDetect, aimed at addressing both vectors of generalization. FusionDetect draws on the benefits of two frozen foundation models: CLIP & Dinov2. By deriving features from both complementary models,we develop a cohesive feature space that naturally adapts to changes in both thecontent and design of the generator. Our extensive experiments demonstrate that FusionDetect delivers not only a new state-of-the-art, which is 3.87% more accurate than its closest competitor and 6.13% more precise on average on established benchmarks, but also achieves a 4.48% increase in accuracy on OmniGen,along with exceptional robustness to common image perturbations. We introduce not only a top-performing detector, but also a new benchmark and framework for furthering universal AI image detection. The code and dataset are available at http://github.com/amir-aman/FusionDetect
>
---
#### [new 027] Deforming Videos to Masks: Flow Matching for Referring Video Segmentation
- **分类: cs.CV**

- **简介: 该论文属于视频语义分割任务，旨在解决基于自然语言描述的视频对象连续分割问题。现有方法因分阶段处理导致语义简化和时序不连贯。作者提出FlowRVS，将任务建模为语言引导的视频到掩码的连续变形过程，实现语义对齐与时序一致性，取得最优性能。**

- **链接: [http://arxiv.org/pdf/2510.06139v1](http://arxiv.org/pdf/2510.06139v1)**

> **作者:** Zanyi Wang; Dengyang Jiang; Liuzhuozheng Li; Sizhe Dang; Chengzu Li; Harry Yang; Guang Dai; Mengmeng Wang; Jingdong Wang
>
> **摘要:** Referring Video Object Segmentation (RVOS) requires segmenting specific objects in a video guided by a natural language description. The core challenge of RVOS is to anchor abstract linguistic concepts onto a specific set of pixels and continuously segment them through the complex dynamics of a video. Faced with this difficulty, prior work has often decomposed the task into a pragmatic `locate-then-segment' pipeline. However, this cascaded design creates an information bottleneck by simplifying semantics into coarse geometric prompts (e.g, point), and struggles to maintain temporal consistency as the segmenting process is often decoupled from the initial language grounding. To overcome these fundamental limitations, we propose FlowRVS, a novel framework that reconceptualizes RVOS as a conditional continuous flow problem. This allows us to harness the inherent strengths of pretrained T2V models, fine-grained pixel control, text-video semantic alignment, and temporal coherence. Instead of conventional generating from noise to mask or directly predicting mask, we reformulate the task by learning a direct, language-guided deformation from a video's holistic representation to its target mask. Our one-stage, generative approach achieves new state-of-the-art results across all major RVOS benchmarks. Specifically, achieving a $\mathcal{J}\&\mathcal{F}$ of 51.1 in MeViS (+1.6 over prior SOTA) and 73.3 in the zero shot Ref-DAVIS17 (+2.7), demonstrating the significant potential of modeling video understanding tasks as continuous deformation processes.
>
---
#### [new 028] ArchitectHead: Continuous Level of Detail Control for 3D Gaussian Head Avatars
- **分类: cs.CV**

- **简介: 该论文属于3D头像建模与渲染任务，旨在解决现有3D高斯点渲染方法在细节层次控制上的不足。作者提出ArchitectHead框架，通过参数化2D UV特征空间和多级特征图，实现无需重训练的连续细节层次控制，兼顾渲染效率与视觉质量。**

- **链接: [http://arxiv.org/pdf/2510.05488v1](http://arxiv.org/pdf/2510.05488v1)**

> **作者:** Peizhi Yan; Rabab Ward; Qiang Tang; Shan Du
>
> **摘要:** 3D Gaussian Splatting (3DGS) has enabled photorealistic and real-time rendering of 3D head avatars. Existing 3DGS-based avatars typically rely on tens of thousands of 3D Gaussian points (Gaussians), with the number of Gaussians fixed after training. However, many practical applications require adjustable levels of detail (LOD) to balance rendering efficiency and visual quality. In this work, we propose "ArchitectHead", the first framework for creating 3D Gaussian head avatars that support continuous control over LOD. Our key idea is to parameterize the Gaussians in a 2D UV feature space and propose a UV feature field composed of multi-level learnable feature maps to encode their latent features. A lightweight neural network-based decoder then transforms these latent features into 3D Gaussian attributes for rendering. ArchitectHead controls the number of Gaussians by dynamically resampling feature maps from the UV feature field at the desired resolutions. This method enables efficient and continuous control of LOD without retraining. Experimental results show that ArchitectHead achieves state-of-the-art (SOTA) quality in self and cross-identity reenactment tasks at the highest LOD, while maintaining near SOTA performance at lower LODs. At the lowest LOD, our method uses only 6.2\% of the Gaussians while the quality degrades moderately (L1 Loss +7.9\%, PSNR --0.97\%, SSIM --0.6\%, LPIPS Loss +24.1\%), and the rendering speed nearly doubles.
>
---
#### [new 029] Universal Neural Architecture Space: Covering ConvNets, Transformers and Everything in Between
- **分类: cs.CV**

- **简介: 该论文属于神经架构搜索（NAS）任务，旨在解决统一不同神经网络架构搜索空间的问题。作者提出了UniNAS，一个统一的搜索空间，涵盖卷积网络、Transformer及其混合架构，并设计了新搜索算法以探索该空间，发现了性能更优的新型架构。此外，还提供了统一的工具包以提升研究的可复现性和公平比较。**

- **链接: [http://arxiv.org/pdf/2510.06035v1](http://arxiv.org/pdf/2510.06035v1)**

> **作者:** Ondřej Týbl; Lukáš Neumann
>
> **摘要:** We introduce Universal Neural Architecture Space (UniNAS), a generic search space for neural architecture search (NAS) which unifies convolutional networks, transformers, and their hybrid architectures under a single, flexible framework. Our approach enables discovery of novel architectures as well as analyzing existing architectures in a common framework. We also propose a new search algorithm that allows traversing the proposed search space, and demonstrate that the space contains interesting architectures, which, when using identical training setup, outperform state-of-the-art hand-crafted architectures. Finally, a unified toolkit including a standardized training and evaluation protocol is introduced to foster reproducibility and enable fair comparison in NAS research. Overall, this work opens a pathway towards systematically exploring the full spectrum of neural architectures with a unified graph-based NAS perspective.
>
---
#### [new 030] Diffusion Models for Low-Light Image Enhancement: A Multi-Perspective Taxonomy and Performance Analysis
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于低光图像增强任务，旨在解决低光环境下图像质量下降影响后续应用的问题。论文系统分析了扩散模型在此领域的应用，提出了多视角分类体系，并对比了不同方法的性能、效率及部署挑战，探讨了未来研究方向。**

- **链接: [http://arxiv.org/pdf/2510.05976v1](http://arxiv.org/pdf/2510.05976v1)**

> **作者:** Eashan Adhikarla; Yixin Liu; Brian D. Davison
>
> **摘要:** Low-light image enhancement (LLIE) is vital for safety-critical applications such as surveillance, autonomous navigation, and medical imaging, where visibility degradation can impair downstream task performance. Recently, diffusion models have emerged as a promising generative paradigm for LLIE due to their capacity to model complex image distributions via iterative denoising. This survey provides an up-to-date critical analysis of diffusion models for LLIE, distinctively featuring an in-depth comparative performance evaluation against Generative Adversarial Network and Transformer-based state-of-the-art methods, a thorough examination of practical deployment challenges, and a forward-looking perspective on the role of emerging paradigms like foundation models. We propose a multi-perspective taxonomy encompassing six categories: Intrinsic Decomposition, Spectral & Latent, Accelerated, Guided, Multimodal, and Autonomous; that map enhancement methods across physical priors, conditioning schemes, and computational efficiency. Our taxonomy is grounded in a hybrid view of both the model mechanism and the conditioning signals. We evaluate qualitative failure modes, benchmark inconsistencies, and trade-offs between interpretability, generalization, and inference efficiency. We also discuss real-world deployment constraints (e.g., memory, energy use) and ethical considerations. This survey aims to guide the next generation of diffusion-based LLIE research by highlighting trends and surfacing open research questions, including novel conditioning, real-time adaptation, and the potential of foundation models.
>
---
#### [new 031] DeepAf: One-Shot Spatiospectral Auto-Focus Model for Digital Pathology
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像处理任务，旨在解决病理切片数字化中的自动对焦问题。现有方法成本高或效率低，且泛化能力差。论文提出DeepAf，一种结合空间与光谱特征的单次自动对焦模型，实现快速、准确的对焦，提升低资源环境下的数字病理诊断效果。**

- **链接: [http://arxiv.org/pdf/2510.05315v1](http://arxiv.org/pdf/2510.05315v1)**

> **作者:** Yousef Yeganeh; Maximilian Frantzen; Michael Lee; Kun-Hsing Yu; Nassir Navab; Azade Farshad
>
> **摘要:** While Whole Slide Imaging (WSI) scanners remain the gold standard for digitizing pathology samples, their high cost limits accessibility in many healthcare settings. Other low-cost solutions also face critical limitations: automated microscopes struggle with consistent focus across varying tissue morphology, traditional auto-focus methods require time-consuming focal stacks, and existing deep-learning approaches either need multiple input images or lack generalization capability across tissue types and staining protocols. We introduce a novel automated microscopic system powered by DeepAf, a novel auto-focus framework that uniquely combines spatial and spectral features through a hybrid architecture for single-shot focus prediction. The proposed network automatically regresses the distance to the optimal focal point using the extracted spatiospectral features and adjusts the control parameters for optimal image outcomes. Our system transforms conventional microscopes into efficient slide scanners, reducing focusing time by 80% compared to stack-based methods while achieving focus accuracy of 0.18 {\mu}m on the same-lab samples, matching the performance of dual-image methods (0.19 {\mu}m) with half the input requirements. DeepAf demonstrates robust cross-lab generalization with only 0.72% false focus predictions and 90% of predictions within the depth of field. Through an extensive clinical study of 536 brain tissue samples, our system achieves 0.90 AUC in cancer classification at 4x magnification, a significant achievement at lower magnification than typical 20x WSI scans. This results in a comprehensive hardware-software design enabling accessible, real-time digital pathology in resource-constrained settings while maintaining diagnostic accuracy.
>
---
#### [new 032] Continual Learning for Image Captioning through Improved Image-Text Alignment
- **分类: cs.CV**

- **简介: 该论文属于图像描述生成任务，旨在解决持续学习中的灾难性遗忘及视觉-文本对齐问题。作者提出了一种结合语义引导与对比对齐的多损失框架，在不增加推理开销的情况下，有效提升了模型在持续学习场景下的表现。**

- **链接: [http://arxiv.org/pdf/2510.06009v1](http://arxiv.org/pdf/2510.06009v1)**

> **作者:** Bertram Taetz; Gal Bordelius
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Generating accurate and coherent image captions in a continual learning setting remains a major challenge due to catastrophic forgetting and the difficulty of aligning evolving visual concepts with language over time. In this work, we propose a novel multi-loss framework for continual image captioning that integrates semantic guidance through prompt-based continual learning and contrastive alignment. Built upon a pretrained ViT-GPT-2 backbone, our approach combines standard cross-entropy loss with three additional components: (1) a prompt-based cosine similarity loss that aligns image embeddings with synthetically constructed prompts encoding objects, attributes, and actions; (2) a CLIP-style loss that promotes alignment between image embeddings and target caption embedding; and (3) a language-guided contrastive loss that employs a triplet loss to enhance class-level discriminability between tasks. Notably, our approach introduces no additional overhead at inference time and requires no prompts during caption generation. We find that this approach mitigates catastrophic forgetting, while achieving better semantic caption alignment compared to state-of-the-art methods. The code can be found via the following link https://github.com/ Gepardius/Taetz_Bordelius_Continual_ImageCaptioning.
>
---
#### [new 033] Towards Data-Efficient Medical Imaging: A Generative and Semi-Supervised Framework
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决标注数据稀缺且不平衡的问题。论文提出了SSGNet框架，结合生成模型与半监督伪标签方法，通过生成高质量图像并迭代优化标签，提升分类与分割性能，缓解医学图像标注瓶颈。**

- **链接: [http://arxiv.org/pdf/2510.06123v1](http://arxiv.org/pdf/2510.06123v1)**

> **作者:** Mosong Ma; Tania Stathaki; Michalis Lazarou
>
> **备注:** Accepted at BMVC2025
>
> **摘要:** Deep learning in medical imaging is often limited by scarce and imbalanced annotated data. We present SSGNet, a unified framework that combines class specific generative modeling with iterative semisupervised pseudo labeling to enhance both classification and segmentation. Rather than functioning as a standalone model, SSGNet augments existing baselines by expanding training data with StyleGAN3 generated images and refining labels through iterative pseudo labeling. Experiments across multiple medical imaging benchmarks demonstrate consistent gains in classification and segmentation performance, while Frechet Inception Distance analysis confirms the high quality of generated samples. These results highlight SSGNet as a practical strategy to mitigate annotation bottlenecks and improve robustness in medical image analysis.
>
---
#### [new 034] EduVerse: A User-Defined Multi-Agent Simulation Space for Education Scenario
- **分类: cs.CV; cs.CY**

- **简介: 论文提出EduVerse，一个用户定义的多智能体教育模拟空间，旨在解决虚拟课堂中认知发展、群体互动与长期演化的综合建模难题。它支持环境、智能体与会话定制，并通过人机协同接口实现真实用户参与，验证了其在中文课堂中的教学真实性与长期适应性。**

- **链接: [http://arxiv.org/pdf/2510.05650v1](http://arxiv.org/pdf/2510.05650v1)**

> **作者:** Yiping Ma; Shiyu Hu; Buyuan Zhu; Yipei Wang; Yaxuan Kang; Shiqing Liu; Kang Hao Cheong
>
> **备注:** Preprint, Under review
>
> **摘要:** Reproducing cognitive development, group interaction, and long-term evolution in virtual classrooms remains a core challenge for educational AI, as real classrooms integrate open-ended cognition, dynamic social interaction, affective factors, and multi-session development rarely captured together. Existing approaches mostly focus on short-term or single-agent settings, limiting systematic study of classroom complexity and cross-task reuse. We present EduVerse, the first user-defined multi-agent simulation space that supports environment, agent, and session customization. A distinctive human-in-the-loop interface further allows real users to join the space. Built on a layered CIE (Cognition-Interaction-Evolution) architecture, EduVerse ensures individual consistency, authentic interaction, and longitudinal adaptation in cognition, emotion, and behavior-reproducing realistic classroom dynamics with seamless human-agent integration. We validate EduVerse in middle-school Chinese classes across three text genres, environments, and multiple sessions. Results show: (1) Instructional alignment: simulated IRF rates (0.28-0.64) closely match real classrooms (0.37-0.49), indicating pedagogical realism; (2) Group interaction and role differentiation: network density (0.27-0.40) with about one-third of peer links realized, while human-agent tasks indicate a balance between individual variability and instructional stability; (3) Cross-session evolution: the positive transition rate R+ increase by 11.7% on average, capturing longitudinal shifts in behavior, emotion, and cognition and revealing structured learning trajectories. Overall, EduVerse balances realism, reproducibility, and interpretability, providing a scalable platform for educational AI. The system will be open-sourced to foster cross-disciplinary research.
>
---
#### [new 035] Combined Hyperbolic and Euclidean Soft Triple Loss Beyond the Single Space Deep Metric Learning
- **分类: cs.CV; 68T10,; I.2.10; I.4.7**

- **简介: 该论文属于深度度量学习任务，旨在解决超球面空间中代理损失难以应用的问题。通过结合超球面与欧氏空间的代理损失及超球面正则化，提出CHEST损失，提升模型准确性和稳定性，达到更优的语义相似度学习效果。**

- **链接: [http://arxiv.org/pdf/2510.05643v1](http://arxiv.org/pdf/2510.05643v1)**

> **作者:** Shozo Saeki; Minoru Kawahara; Hirohisa Aman
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Deep metric learning (DML) aims to learn a neural network mapping data to an embedding space, which can represent semantic similarity between data points. Hyperbolic space is attractive for DML since it can represent richer structures, such as tree structures. DML in hyperbolic space is based on pair-based loss or unsupervised regularization loss. On the other hand, supervised proxy-based losses in hyperbolic space have not been reported yet due to some issues in applying proxy-based losses in a hyperbolic space. However, proxy-based losses are attractive for large-scale datasets since they have less training complexity. To address these, this paper proposes the Combined Hyperbolic and Euclidean Soft Triple (CHEST) loss. CHEST loss is composed of the proxy-based losses in hyperbolic and Euclidean spaces and the regularization loss based on hyperbolic hierarchical clustering. We find that the combination of hyperbolic and Euclidean spaces improves DML accuracy and learning stability for both spaces. Finally, we evaluate the CHEST loss on four benchmark datasets, achieving a new state-of-the-art performance.
>
---
#### [new 036] CalibCLIP: Contextual Calibration of Dominant Semantics for Text-Driven Image Retrieval
- **分类: cs.CV**

- **简介: 论文属于文本驱动图像检索任务，旨在解决现有视觉语言模型中低贡献文本标记压制判别特征的问题。作者提出CalibCLIP方法，通过对比视觉增强器和判别概念校准器，分别在视觉和文本空间中优化特征表示，提升检索效果。实验验证了该方法在多个基准上的有效性。**

- **链接: [http://arxiv.org/pdf/2510.05586v1](http://arxiv.org/pdf/2510.05586v1)**

> **作者:** Bin Kang; Bin Chen; Junjie Wang; Yulin Li; Junzhi Zhao; Zhuotao Tian
>
> **备注:** ACMMM2025(oral)
>
> **摘要:** Existing Visual Language Models (VLMs) suffer structural limitations where a few low contribution tokens may excessively capture global semantics, dominating the information aggregation process and suppressing the discriminative features in text-driven image retrieval tasks. To address this, we introduce \textbf{CalibCLIP}, a training-free method designed to calibrate the suppressive effect of dominant tokens. Specifically, in the visual space, we propose the Contrastive Visual Enhancer (CVE), which decouples visual features into target and low information regions. Subsequently, it identifies dominant tokens and dynamically suppresses their representations.In the textual space, we introduce the Discriminative Concept Calibrator (DCC), which aims to differentiate between general and discriminative concepts within the text query. By mitigating the challenges posed by generic concepts and improving the representations of discriminative concepts, DCC strengthens the differentiation among similar samples. Finally, extensive experiments demonstrate consistent improvements across seven benchmarks spanning three image retrieval tasks, underscoring the effectiveness of CalibCLIP. Code is available at: https://github.com/kangbin98/CalibCLIP
>
---
#### [new 037] Dropping the D: RGB-D SLAM Without the Depth Sensor
- **分类: cs.CV; cs.RO**

- **简介: 论文提出DropD-SLAM，一种无需深度传感器的实时单目SLAM系统。它属于SLAM任务，旨在解决单目SLAM缺乏度量尺度、依赖深度传感器的问题。通过使用预训练视觉模块估计深度、检测关键点和分割实例，实现RGB-D级精度，取得了良好效果。**

- **链接: [http://arxiv.org/pdf/2510.06216v1](http://arxiv.org/pdf/2510.06216v1)**

> **作者:** Mert Kiray; Alican Karaomer; Benjamin Busam
>
> **摘要:** We present DropD-SLAM, a real-time monocular SLAM system that achieves RGB-D-level accuracy without relying on depth sensors. The system replaces active depth input with three pretrained vision modules: a monocular metric depth estimator, a learned keypoint detector, and an instance segmentation network. Dynamic objects are suppressed using dilated instance masks, while static keypoints are assigned predicted depth values and backprojected into 3D to form metrically scaled features. These are processed by an unmodified RGB-D SLAM back end for tracking and mapping. On the TUM RGB-D benchmark, DropD-SLAM attains 7.4 cm mean ATE on static sequences and 1.8 cm on dynamic sequences, matching or surpassing state-of-the-art RGB-D methods while operating at 22 FPS on a single GPU. These results suggest that modern pretrained vision models can replace active depth sensors as reliable, real-time sources of metric scale, marking a step toward simpler and more cost-effective SLAM systems.
>
---
#### [new 038] Development and Validation of a Low-Cost Imaging System for Seedling Germination Kinetics through Time-Cumulative Analysis
- **分类: cs.CV**

- **简介: 该论文属于图像分析与植物表型任务，旨在解决病原菌对种子萌发影响的准确监测问题。作者开发了一种低成本成像系统及相应图像分析算法，通过时间累积分析提高萌发动态和幼苗活力的量化精度，尤其在幼苗密集或重叠情况下仍能保持高准确性。**

- **链接: [http://arxiv.org/pdf/2510.05668v1](http://arxiv.org/pdf/2510.05668v1)**

> **作者:** M. Torrente; A. Follador; A. Calcante; P. Casati; R. Oberti
>
> **摘要:** The study investigates the effects of R. solani inoculation on the germination and early development of Lactuca sativa L. seeds using a low-cost, image-based monitoring system. Multiple cameras were deployed to continuously capture images of the germination process in both infected and control groups. The objective was to assess the impact of the pathogen by analyzing germination dynamics and growth over time. To achieve this, a novel image analysis pipeline was developed. The algorithm integrates both morphological and spatial features to identify and quantify individual seedlings, even under complex conditions where traditional image analyses fails. A key innovation of the method lies in its temporal integration: each analysis step considers not only the current status but also their developmental across prior time points. This approach enables robust discrimination of individual seedlings, especially when overlapping leaves significantly hinder object separation. The method demonstrated high accuracy in seedling counting and vigor assessment, even in challenging scenarios characterized by dense and intertwined growth. Results confirm that R. solani infection significantly reduces germination rates and early seedling vigor. The study also validates the feasibility of combining low-cost imaging hardware with advanced computational tools to obtain phenotyping data in a non-destructive and scalable manner. The temporal integration enabled accurate quantification of germinated seeds and precise determination of seedling emergence timing. This approach proved particularly effective in later stages of the experiment, where conventional segmentation techniques failed due to overlapping or intertwined seedlings, making accurate counting. The method achieved a coefficient of determination of 0.98 and a root mean square error (RMSE) of 1.12, demonstrating its robustness and reliability.
>
---
#### [new 039] Deformable Image Registration for Self-supervised Cardiac Phase Detection in Multi-View Multi-Disease Cardiac Magnetic Resonance Images
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析任务，旨在解决心脏磁共振图像中自动检测心脏周期关键帧的问题。现有方法仅依赖左心室容积曲线检测关键帧，无法深入分析心肌运动。论文提出一种自监督深度学习方法，结合可变形配准和运动描述符，准确检测多个关键帧，提升了检测精度，并验证了其在多数据集上的重复性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.05819v1](http://arxiv.org/pdf/2510.05819v1)**

> **作者:** Sven Koehler; Sarah Kaye Mueller; Jonathan Kiekenap; Gerald Greil; Tarique Hussain; Samir Sarikouch; Florian André; Norbert Frey; Sandy Engelhardt
>
> **备注:** Main 30 pages, 6 figures
>
> **摘要:** Cardiovascular magnetic resonance (CMR) is the gold standard for assessing cardiac function, but individual cardiac cycles complicate automatic temporal comparison or sub-phase analysis. Accurate cardiac keyframe detection can eliminate this problem. However, automatic methods solely derive end-systole (ES) and end-diastole (ED) frames from left ventricular volume curves, which do not provide a deeper insight into myocardial motion. We propose a self-supervised deep learning method detecting five keyframes in short-axis (SAX) and four-chamber long-axis (4CH) cine CMR. Initially, dense deformable registration fields are derived from the images and used to compute a 1D motion descriptor, which provides valuable insights into global cardiac contraction and relaxation patterns. From these characteristic curves, keyframes are determined using a simple set of rules. The method was independently evaluated for both views using three public, multicentre, multidisease datasets. M&Ms-2 (n=360) dataset was used for training and evaluation, and M&Ms (n=345) and ACDC (n=100) datasets for repeatability control. Furthermore, generalisability to patients with rare congenital heart defects was tested using the German Competence Network (GCN) dataset. Our self-supervised approach achieved improved detection accuracy by 30% - 51% for SAX and 11% - 47% for 4CH in ED and ES, as measured by cyclic frame difference (cFD), compared with the volume-based approach. We can detect ED and ES, as well as three additional keyframes throughout the cardiac cycle with a mean cFD below 1.31 frames for SAX and 1.73 for LAX. Our approach enables temporally aligned inter- and intra-patient analysis of cardiac dynamics, irrespective of cycle or phase lengths. GitHub repository: https://github.com/Cardio-AI/cmr-multi-view-phase-detection.git
>
---
#### [new 040] Compact Multi-level-prior Tensor Representation for Hyperspectral Image Super-resolution
- **分类: cs.CV**

- **简介: 该论文属于图像超分辨率任务，旨在解决高光谱图像与多光谱图像融合中多级先验建模困难的问题。作者提出一种新的张量表示方法，通过块分解解耦光谱低秩性和空间先验，并利用非凸张量全变分建模高阶空间结构，最终设计高效算法优化模型，提升了融合效果。**

- **链接: [http://arxiv.org/pdf/2510.06098v1](http://arxiv.org/pdf/2510.06098v1)**

> **作者:** Yinjian Wang; Wei Li; Yuanyuan Gui; Gemine Vivone
>
> **摘要:** Fusing a hyperspectral image with a multispectral image acquired over the same scene, \textit{i.e.}, hyperspectral image super-resolution, has become a popular computational way to access the latent high-spatial-spectral-resolution image. To date, a variety of fusion methods have been proposed, among which the tensor-based ones have testified that multiple priors, such as multidimensional low-rankness and spatial total variation at multiple levels, effectively drive the fusion process. However, existing tensor-based models can only effectively leverage one or two priors at one or two levels, since simultaneously incorporating multi-level priors inevitably increases model complexity. This introduces challenges in both balancing the weights of different priors and optimizing multi-block structures. Concerning this, we present a novel hyperspectral super-resolution model compactly characterizing these multi-level priors of hyperspectral images within the tensor framework. Firstly, the proposed model decouples the spectral low-rankness and spatial priors by casting the latent high-spatial-spectral-resolution image into spectral subspace and spatial maps via block term decomposition. Secondly, these spatial maps are stacked as the spatial tensor encoding the high-order spatial low-rankness and smoothness priors, which are co-modeled via the proposed non-convex mode-shuffled tensor correlated total variation. Finally, we draw inspiration from the linearized alternating direction method of multipliers to design an efficient algorithm to optimize the resulting model, theoretically proving its Karush-Kuhn-Tucker convergence under mild conditions. Experiments on multiple datasets demonstrate the effectiveness of the proposed algorithm. The code implementation will be available from https://github.com/WongYinJ.
>
---
#### [new 041] Efficient Universal Models for Medical Image Segmentation via Weakly Supervised In-Context Learning
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决现有通用模型依赖大量精细标注数据的问题。作者提出了一种弱监督上下文学习（WS-ICL）方法，利用弱标签（如边界框或点）代替精细标注，显著降低了标注成本，同时保持了与传统方法相当的性能。**

- **链接: [http://arxiv.org/pdf/2510.05899v1](http://arxiv.org/pdf/2510.05899v1)**

> **作者:** Jiesi Hu; Yanwu Yang; Zhiyu Ye; Jinyan Zhou; Jianfeng Cao; Hanyang Peng; Ting Ma
>
> **摘要:** Universal models for medical image segmentation, such as interactive and in-context learning (ICL) models, offer strong generalization but require extensive annotations. Interactive models need repeated user prompts for each image, while ICL relies on dense, pixel-level labels. To address this, we propose Weakly Supervised In-Context Learning (WS-ICL), a new ICL paradigm that leverages weak prompts (e.g., bounding boxes or points) instead of dense labels for context. This approach significantly reduces annotation effort by eliminating the need for fine-grained masks and repeated user prompting for all images. We evaluated the proposed WS-ICL model on three held-out benchmarks. Experimental results demonstrate that WS-ICL achieves performance comparable to regular ICL models at a significantly lower annotation cost. In addition, WS-ICL is highly competitive even under the interactive paradigm. These findings establish WS-ICL as a promising step toward more efficient and unified universal models for medical image segmentation. Our code and model are publicly available at https://github.com/jiesihu/Weak-ICL.
>
---
#### [new 042] Personalizing Retrieval using Joint Embeddings or "the Return of Fluffy"
- **分类: cs.CV**

- **简介: 该论文属于图像检索任务，旨在解决结合图像与文本描述的个性化检索问题。作者提出pi-map方法，通过映射网络将局部图像嵌入翻译为文本标记，与自然语言查询结合，提升检索效果。实验证明其方法在两个基准上达到先进水平。**

- **链接: [http://arxiv.org/pdf/2510.05411v1](http://arxiv.org/pdf/2510.05411v1)**

> **作者:** Bruno Korbar; Andrew Zisserman
>
> **备注:** Published as an oral in CBMI2025
>
> **摘要:** The goal of this paper is to be able to retrieve images using a compound query that combines object instance information from an image, with a natural text description of what that object is doing or where it is. For example, to retrieve an image of "Fluffy the unicorn (specified by an image) on someone's head". To achieve this we design a mapping network that can "translate" from a local image embedding (of the object instance) to a text token, such that the combination of the token and a natural language query is suitable for CLIP style text encoding, and image retrieval. Generating a text token in this manner involves a simple training procedure, that only needs to be performed once for each object instance. We show that our approach of using a trainable mapping network, termed pi-map, together with frozen CLIP text and image encoders, improves the state of the art on two benchmarks designed to assess personalized retrieval.
>
---
#### [new 043] Multimodal Feature Prototype Learning for Interpretable and Discriminative Cancer Survival Prediction
- **分类: cs.CV**

- **简介: 该论文属于癌症生存预测任务，旨在解决现有模型缺乏可解释性及多模态数据融合不足的问题。论文提出FeatProto框架，结合病理图像与基因组数据，通过创新的原型学习策略提升预测准确性与解释性。**

- **链接: [http://arxiv.org/pdf/2510.06113v1](http://arxiv.org/pdf/2510.06113v1)**

> **作者:** Shuo Jiang; Zhuwen Chen; Liaoman Xu; Yanming Zhu; Changmiao Wang; Jiong Zhang; Feiwei Qin; Yifei Chen; Zhu Zhu
>
> **备注:** 12 pages, 10 figures
>
> **摘要:** Survival analysis plays a vital role in making clinical decisions. However, the models currently in use are often difficult to interpret, which reduces their usefulness in clinical settings. Prototype learning presents a potential solution, yet traditional methods focus on local similarities and static matching, neglecting the broader tumor context and lacking strong semantic alignment with genomic data. To overcome these issues, we introduce an innovative prototype-based multimodal framework, FeatProto, aimed at enhancing cancer survival prediction by addressing significant limitations in current prototype learning methodologies within pathology. Our framework establishes a unified feature prototype space that integrates both global and local features of whole slide images (WSI) with genomic profiles. This integration facilitates traceable and interpretable decision-making processes. Our approach includes three main innovations: (1) A robust phenotype representation that merges critical patches with global context, harmonized with genomic data to minimize local bias. (2) An Exponential Prototype Update Strategy (EMA ProtoUp) that sustains stable cross-modal associations and employs a wandering mechanism to adapt prototypes flexibly to tumor heterogeneity. (3) A hierarchical prototype matching scheme designed to capture global centrality, local typicality, and cohort-level trends, thereby refining prototype inference. Comprehensive evaluations on four publicly available cancer datasets indicate that our method surpasses current leading unimodal and multimodal survival prediction techniques in both accuracy and interoperability, providing a new perspective on prototype learning for critical medical applications. Our source code is available at https://github.com/JSLiam94/FeatProto.
>
---
#### [new 044] Fine-grained Defocus Blur Control for Generative Image Models
- **分类: cs.CV**

- **简介: 该论文属于图像生成任务，旨在解决当前文本到图像扩散模型难以精细控制镜头模糊的问题。论文提出了一种结合相机元数据（如EXIF）的新框架，通过模拟物理成像过程，实现对景深模糊的精细控制，提升了生成图像的真实感与可控性。**

- **链接: [http://arxiv.org/pdf/2510.06215v1](http://arxiv.org/pdf/2510.06215v1)**

> **作者:** Ayush Shrivastava; Connelly Barnes; Xuaner Zhang; Lingzhi Zhang; Andrew Owens; Sohrab Amirghodsi; Eli Shechtman
>
> **备注:** Project link: https://www.ayshrv.com/defocus-blur-gen
>
> **摘要:** Current text-to-image diffusion models excel at generating diverse, high-quality images, yet they struggle to incorporate fine-grained camera metadata such as precise aperture settings. In this work, we introduce a novel text-to-image diffusion framework that leverages camera metadata, or EXIF data, which is often embedded in image files, with an emphasis on generating controllable lens blur. Our method mimics the physical image formation process by first generating an all-in-focus image, estimating its monocular depth, predicting a plausible focus distance with a novel focus distance transformer, and then forming a defocused image with an existing differentiable lens blur model. Gradients flow backwards through this whole process, allowing us to learn without explicit supervision to generate defocus effects based on content elements and the provided EXIF data. At inference time, this enables precise interactive user control over defocus effects while preserving scene contents, which is not achievable with existing diffusion models. Experimental results demonstrate that our model enables superior fine-grained control without altering the depicted scene.
>
---
#### [new 045] When Thinking Drifts: Evidential Grounding for Robust Video Reasoning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视频推理任务，旨在解决多步骤逻辑推理中视觉细节失真的问题。作者提出“视觉思维漂移”现象，分析其成因，并引入基于视觉证据的强化学习框架VER，提升视频推理的准确性与可靠性。**

- **链接: [http://arxiv.org/pdf/2510.06077v1](http://arxiv.org/pdf/2510.06077v1)**

> **作者:** Mi Luo; Zihui Xue; Alex Dimakis; Kristen Grauman
>
> **备注:** Accepted by NeurIPS 2025, Project page: https://vision.cs.utexas.edu/projects/video-ver/
>
> **摘要:** Video reasoning, the task of enabling machines to infer from dynamic visual content through multi-step logic, is crucial for advanced AI. While the Chain-of-Thought (CoT) mechanism has enhanced reasoning in text-based tasks, its application to video understanding remains underexplored. This paper presents a systematic analysis revealing that CoT often degrades performance in video reasoning, generating verbose but misleading internal monologues, and leading to hallucinated visual details and overridden correct intuitions - a phenomenon we term "visual thinking drift". We explain this drift through a Bayesian lens, positing that CoT traces often diverge from actual visual evidence, instead amplifying internal biases or language priors, causing models to storytell rather than engage in grounded reasoning. To counteract this, we introduce Visual Evidence Reward (VER), a novel reinforcement learning framework that explicitly rewards the generation of reasoning traces that are verifiably grounded in visual evidence. Comprehensive evaluation across 10 diverse video understanding benchmarks demonstrates that our Video-VER consistently achieves top performance. Our work sheds light on the distinct challenges of video-centric reasoning and encourages the development of AI that robustly grounds its inferences in visual evidence - for large multimodal models that not only "think before answering", but also "see while thinking".
>
---
#### [new 046] Human3R: Everyone Everywhere All at Once
- **分类: cs.CV**

- **简介: 该论文提出Human3R，用于在线4D多人与场景联合重建的任务，旨在从单目视频中一次性重建多人SMPL-X模型、3D场景及相机轨迹。相比传统多阶段方法，Human3R采用单次前向推理，无需依赖检测、深度估计或SLAM预处理，实现高效实时重建。**

- **链接: [http://arxiv.org/pdf/2510.06219v1](http://arxiv.org/pdf/2510.06219v1)**

> **作者:** Yue Chen; Xingyu Chen; Yuxuan Xue; Anpei Chen; Yuliang Xiu; Gerard Pons-Moll
>
> **备注:** Page: https://fanegg.github.io/Human3R Code: https://github.com/fanegg/Human3R
>
> **摘要:** We present Human3R, a unified, feed-forward framework for online 4D human-scene reconstruction, in the world frame, from casually captured monocular videos. Unlike previous approaches that rely on multi-stage pipelines, iterative contact-aware refinement between humans and scenes, and heavy dependencies, e.g., human detection, depth estimation, and SLAM pre-processing, Human3R jointly recovers global multi-person SMPL-X bodies ("everyone"), dense 3D scene ("everywhere"), and camera trajectories in a single forward pass ("all-at-once"). Our method builds upon the 4D online reconstruction model CUT3R, and uses parameter-efficient visual prompt tuning, to strive to preserve CUT3R's rich spatiotemporal priors, while enabling direct readout of multiple SMPL-X bodies. Human3R is a unified model that eliminates heavy dependencies and iterative refinement. After being trained on the relatively small-scale synthetic dataset BEDLAM for just one day on one GPU, it achieves superior performance with remarkable efficiency: it reconstructs multiple humans in a one-shot manner, along with 3D scenes, in one stage, at real-time speed (15 FPS) with a low memory footprint (8 GB). Extensive experiments demonstrate that Human3R delivers state-of-the-art or competitive performance across tasks, including global human motion estimation, local human mesh recovery, video depth estimation, and camera pose estimation, with a single unified model. We hope that Human3R will serve as a simple yet strong baseline, be easily extended for downstream applications.Code available in https://fanegg.github.io/Human3R
>
---
#### [new 047] A Novel Technique for Robust Training of Deep Networks With Multisource Weak Labeled Remote Sensing Data
- **分类: cs.CV**

- **简介: 该论文属于遥感图像场景分类任务，旨在解决深度学习模型依赖大量高质量标注数据的问题。利用多源弱标注数据与少量可靠数据结合，并提出一种基于标签误差统计的加权训练方法，提升模型在不可靠标签下的鲁棒性与性能。**

- **链接: [http://arxiv.org/pdf/2510.05760v1](http://arxiv.org/pdf/2510.05760v1)**

> **作者:** Gianmarco Perantoni; Lorenzo Bruzzone
>
> **备注:** 16 pages, 9 figures, accepted article
>
> **摘要:** Deep learning has gained broad interest in remote sensing image scene classification thanks to the effectiveness of deep neural networks in extracting the semantics from complex data. However, deep networks require large amounts of training samples to obtain good generalization capabilities and are sensitive to errors in the training labels. This is a problem in remote sensing since highly reliable labels can be obtained at high costs and in limited amount. However, many sources of less reliable labeled data are available, e.g., obsolete digital maps. In order to train deep networks with larger datasets, we propose both the combination of single or multiple weak sources of labeled data with a small but reliable dataset to generate multisource labeled datasets and a novel training strategy where the reliability of each source is taken in consideration. This is done by exploiting the transition matrices describing the statistics of the errors of each source. The transition matrices are embedded into the labels and used during the training process to weigh each label according to the related source. The proposed method acts as a weighting scheme at gradient level, where each instance contributes with different weights to the optimization of different classes. The effectiveness of the proposed method is validated by experiments on different datasets. The results proved the robustness and capability of leveraging on unreliable source of labels of the proposed method.
>
---
#### [new 048] EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于视觉理解任务，旨在解决夜间第一视角视觉理解的不足。作者构建了首个全面的夜间基准EgoNight，包含合成与真实对齐视频，并提出EgoNight-VQA数据集，用于评估多模态模型在低光条件下的表现，揭示了现有模型在夜间场景中的性能下降问题。**

- **链接: [http://arxiv.org/pdf/2510.06218v1](http://arxiv.org/pdf/2510.06218v1)**

> **作者:** Deheng Zhang; Yuqian Fu; Runyi Yang; Yang Miao; Tianwen Qian; Xu Zheng; Guolei Sun; Ajad Chhatkuli; Xuanjing Huang; Yu-Gang Jiang; Luc Van Gool; Danda Pani Paudel
>
> **摘要:** Most existing benchmarks for egocentric vision understanding focus primarily on daytime scenarios, overlooking the low-light conditions that are inevitable in real-world applications. To investigate this gap, we present EgoNight, the first comprehensive benchmark for nighttime egocentric vision, with visual question answering (VQA) as the core task. A key feature of EgoNight is the introduction of day-night aligned videos, which enhance night annotation quality using the daytime data and reveal clear performance gaps between lighting conditions. To achieve this, we collect both synthetic videos rendered by Blender and real-world recordings, ensuring that scenes and actions are visually and temporally aligned. Leveraging these paired videos, we construct EgoNight-VQA, supported by a novel day-augmented night auto-labeling engine and refinement through extensive human verification. Each QA pair is double-checked by annotators for reliability. In total, EgoNight-VQA contains 3658 QA pairs across 90 videos, spanning 12 diverse QA types, with more than 300 hours of human work. Evaluations of state-of-the-art multimodal large language models (MLLMs) reveal substantial performance drops when transferring from day to night, underscoring the challenges of reasoning under low-light conditions. Beyond VQA, EgoNight also introduces two auxiliary tasks, day-night correspondence retrieval and egocentric depth estimation at night, that further explore the boundaries of existing models. We believe EgoNight-VQA provides a strong foundation for advancing application-driven egocentric vision research and for developing models that generalize across illumination domains. All the data and code will be made available upon acceptance.
>
---
#### [new 049] When and How to Cut Classical Concerts? A Multimodal Automated Video Editing Approach
- **分类: cs.CV; cs.MM**

- **简介: 该论文属于多媒体与计算机视觉任务，旨在解决古典音乐会多机位视频的自动化剪辑问题，具体研究“何时剪”和“如何剪”。作者提出了一种多模态架构，融合音频、图像和时间特征进行剪辑点检测，并改进了画面选择方法。通过伪标签构建数据集，模型在剪辑点检测和画面选择上优于基线方法，推动了自动化视频剪辑的发展。**

- **链接: [http://arxiv.org/pdf/2510.05661v1](http://arxiv.org/pdf/2510.05661v1)**

> **作者:** Daniel Gonzálbez-Biosca; Josep Cabacas-Maso; Carles Ventura; Ismael Benito-Altamirano
>
> **摘要:** Automated video editing remains an underexplored task in the computer vision and multimedia domains, especially when contrasted with the growing interest in video generation and scene understanding. In this work, we address the specific challenge of editing multicamera recordings of classical music concerts by decomposing the problem into two key sub-tasks: when to cut and how to cut. Building on recent literature, we propose a novel multimodal architecture for the temporal segmentation task (when to cut), which integrates log-mel spectrograms from the audio signals, plus an optional image embedding, and scalar temporal features through a lightweight convolutional-transformer pipeline. For the spatial selection task (how to cut), we improve the literature by updating from old backbones, e.g. ResNet, with a CLIP-based encoder and constraining distractor selection to segments from the same concert. Our dataset was constructed following a pseudo-labeling approach, in which raw video data was automatically clustered into coherent shot segments. We show that our models outperformed previous baselines in detecting cut points and provide competitive visual shot selection, advancing the state of the art in multimodal automated video editing.
>
---
#### [new 050] Improving Chain-of-Thought Efficiency for Autoregressive Image Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于图像生成任务，旨在解决链式思维（CoT）在生成图像时冗余导致的效率低下问题。作者提出ShortCoTI框架，通过强化学习优化CoT提示，使其更简洁且保持图像质量，从而提升计算效率。**

- **链接: [http://arxiv.org/pdf/2510.05593v1](http://arxiv.org/pdf/2510.05593v1)**

> **作者:** Zeqi Gu; Markos Georgopoulos; Xiaoliang Dai; Marjan Ghazvininejad; Chu Wang; Felix Juefei-Xu; Kunpeng Li; Yujun Shi; Zecheng He; Zijian He; Jiawei Zhou; Abe Davis; Jialiang Wang
>
> **摘要:** Autoregressive multimodal large language models have recently gained popularity for image generation, driven by advances in foundation models. To enhance alignment and detail, newer approaches employ chain-of-thought (CoT) reasoning, expanding user inputs into elaborated prompts prior to image synthesis. However, this strategy can introduce unnecessary redundancy -- a phenomenon we call visual overthinking -- which increases computational costs and can introduce details that contradict the original prompt. In this work, we explore how to generate more concise CoT sequences for more efficient image generation. We introduce ShortCoTI, a lightweight optimization framework that encourages more concise CoT while preserving output image quality. ShortCoTI rewards more concise prompts with an adaptive function that scales according to an estimated difficulty for each task. Incorporating this reward into a reinforcement learning paradigm reduces prompt reasoning length by 54% while maintaining or slightly improving quality metrics across multiple benchmarks (T2I-CompBench, GenEval). Qualitative analysis shows that our method eliminates verbose explanations and repetitive refinements, producing reasoning prompts that are both concise and semantically rich. As a result, ShortCoTI improves computational efficiency without compromising the fidelity or visual appeal of generated images.
>
---
#### [new 051] A Hierarchical Geometry-guided Transformer for Histological Subtyping of Primary Liver Cancer
- **分类: cs.CV**

- **简介: 该论文属于医学图像分析任务，旨在解决肝癌组织亚型分类问题。针对现有方法未充分利用组织切片图像中的层次化特征，作者提出了ARGUS模型，通过构建微观几何特征、建模多尺度视野关系，并融合几何先验信息，实现了更精准的肝癌亚型分类，提升了诊断性能。**

- **链接: [http://arxiv.org/pdf/2510.05657v1](http://arxiv.org/pdf/2510.05657v1)**

> **作者:** Anwen Lu; Mingxin Liu; Yiping Jiao; Hongyi Gong; Geyang Xu; Jun Chen; Jun Xu
>
> **备注:** 7 pages, 2 figures, accepted by IEEE BIBM 2025
>
> **摘要:** Primary liver malignancies are widely recognized as the most heterogeneous and prognostically diverse cancers of the digestive system. Among these, hepatocellular carcinoma (HCC) and intrahepatic cholangiocarcinoma (ICC) emerge as the two principal histological subtypes, demonstrating significantly greater complexity in tissue morphology and cellular architecture than other common tumors. The intricate representation of features in Whole Slide Images (WSIs) encompasses abundant crucial information for liver cancer histological subtyping, regarding hierarchical pyramid structure, tumor microenvironment (TME), and geometric representation. However, recent approaches have not adequately exploited these indispensable effective descriptors, resulting in a limited understanding of histological representation and suboptimal subtyping performance. To mitigate these limitations, ARGUS is proposed to advance histological subtyping in liver cancer by capturing the macro-meso-micro hierarchical information within the TME. Specifically, we first construct a micro-geometry feature to represent fine-grained cell-level pattern via a geometric structure across nuclei, thereby providing a more refined and precise perspective for delineating pathological images. Then, a Hierarchical Field-of-Views (FoVs) Alignment module is designed to model macro- and meso-level hierarchical interactions inherent in WSIs. Finally, the augmented micro-geometry and FoVs features are fused into a joint representation via present Geometry Prior Guided Fusion strategy for modeling holistic phenotype interactions. Extensive experiments on public and private cohorts demonstrate that our ARGUS achieves state-of-the-art (SOTA) performance in histological subtyping of liver cancer, which provide an effective diagnostic tool for primary liver malignancies in clinical practice.
>
---
#### [new 052] VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于长视频理解任务，旨在解决现有方法在冗余信息干扰和复杂结构适应上的不足。论文提出VideoMiner框架，通过迭代分割、描述和聚类视频，结合树状结构与T-GRPO强化学习方法，实现精准关键帧定位与高效视频理解。**

- **链接: [http://arxiv.org/pdf/2510.06040v1](http://arxiv.org/pdf/2510.06040v1)**

> **作者:** Xinye Cao; Hongcan Guo; Jiawen Qian; Guoshun Nan; Chao Wang; Yuqi Pan; Tianhao Hou; Xiaojuan Wang; Yutong Gao
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Understanding hour-long videos with multi-modal large language models (MM-LLMs) enriches the landscape of human-centered AI applications. However, for end-to-end video understanding with LLMs, uniformly sampling video frames results in LLMs being overwhelmed by a vast amount of irrelevant information as video length increases. Existing hierarchical key frame extraction methods improve the accuracy of video understanding but still face two critical challenges. 1) How can the interference of extensive redundant information in long videos be mitigated? 2) How can a model dynamically adapt to complex hierarchical structures while accurately identifying key frames? To address these issues, we propose VideoMiner, which iteratively segments, captions, and clusters long videos, forming a hierarchical tree structure. The proposed VideoMiner progresses from long videos to events to frames while preserving temporal coherence, effectively addressing the first challenge. To precisely locate key frames, we introduce T-GRPO, a tree-based group relative policy optimization in reinforcement learning method that guides the exploration of the VideoMiner. The proposed T-GRPO is specifically designed for tree structures, integrating spatiotemporal information at the event level while being guided by the question, thus solving the second challenge. We achieve superior performance in all long-video understanding tasks and uncover several interesting insights. Our proposed T-GRPO surprisingly incentivizes the model to spontaneously generate a reasoning chain. Additionally, the designed tree growth auxin dynamically adjusts the expansion depth, obtaining accuracy and efficiency gains. The code is publicly available at https://github.com/caoxinye/VideoMiner.
>
---
#### [new 053] Medical Vision Language Models as Policies for Robotic Surgery
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医疗机器人手术任务，旨在解决视觉输入下策略优化效果差的问题。作者将医学视觉语言模型MedFlamingo与PPO结合，提升手术策略的生成效果。实验表明其方法优于基线，在五个手术任务中成功率均超70%。**

- **链接: [http://arxiv.org/pdf/2510.06064v1](http://arxiv.org/pdf/2510.06064v1)**

> **作者:** Akshay Muppidi; Martin Radfar
>
> **备注:** IEEE CAI 2025
>
> **摘要:** Vision-based Proximal Policy Optimization (PPO) struggles with visual observation-based robotic laparoscopic surgical tasks due to the high-dimensional nature of visual input, the sparsity of rewards in surgical environments, and the difficulty of extracting task-relevant features from raw visual data. We introduce a simple approach integrating MedFlamingo, a medical domain-specific Vision-Language Model, with PPO. Our method is evaluated on five diverse laparoscopic surgery task environments in LapGym, using only endoscopic visual observations. MedFlamingo PPO outperforms and converges faster compared to both standard vision-based PPO and OpenFlamingo PPO baselines, achieving task success rates exceeding 70% across all environments, with improvements ranging from 66.67% to 1114.29% compared to baseline. By processing task observations and instructions once per episode to generate high-level planning tokens, our method efficiently combines medical expertise with real-time visual feedback. Our results highlight the value of specialized medical knowledge in robotic surgical planning and decision-making.
>
---
#### [new 054] ALISE: Annotation-Free LiDAR Instance Segmentation for Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文属于自动驾驶中的激光雷达实例分割任务，旨在解决无需人工标注的全无监督实例分割问题。论文提出ALISE框架，利用视觉基础模型生成伪标签，并通过时空投票模块优化，结合2D与3D语义监督学习，实现高性能无监督3D实例分割，超越部分有监督方法。**

- **链接: [http://arxiv.org/pdf/2510.05752v1](http://arxiv.org/pdf/2510.05752v1)**

> **作者:** Yongxuan Lyu; Guangfeng Jiang; Hongsi Liu; Jun Liu
>
> **摘要:** The manual annotation of outdoor LiDAR point clouds for instance segmentation is extremely costly and time-consuming. Current methods attempt to reduce this burden but still rely on some form of human labeling. To completely eliminate this dependency, we introduce ALISE, a novel framework that performs LiDAR instance segmentation without any annotations. The central challenge is to generate high-quality pseudo-labels in a fully unsupervised manner. Our approach starts by employing Vision Foundation Models (VFMs), guided by text and images, to produce initial pseudo-labels. We then refine these labels through a dedicated spatio-temporal voting module, which combines 2D and 3D semantics for both offline and online optimization. To achieve superior feature learning, we further introduce two forms of semantic supervision: a set of 2D prior-based losses that inject visual knowledge into the 3D network, and a novel prototype-based contrastive loss that builds a discriminative feature space by exploiting 3D semantic consistency. This comprehensive design results in significant performance gains, establishing a new state-of-the-art for unsupervised 3D instance segmentation. Remarkably, our approach even outperforms MWSIS, a method that operates with supervision from ground-truth (GT) 2D bounding boxes by a margin of 2.53% in mAP (50.95% vs. 48.42%).
>
---
#### [new 055] See the past: Time-Reversed Scene Reconstruction from Thermal Traces Using Visual Language Models
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像重建任务，旨在通过热成像与视觉语言模型，反向重构场景的过去状态。利用人类活动留下的热痕迹，结合RGB和热图像，提出时间反演框架，实现最多120秒前的场景恢复，提升热成像在事件回溯中的应用。**

- **链接: [http://arxiv.org/pdf/2510.05408v1](http://arxiv.org/pdf/2510.05408v1)**

> **作者:** Kebin Contreras; Luis Toscano-Palomino; Mauro Dalla Mura; Jorge Bacca
>
> **摘要:** Recovering the past from present observations is an intriguing challenge with potential applications in forensics and scene analysis. Thermal imaging, operating in the infrared range, provides access to otherwise invisible information. Since humans are typically warmer (37 C -98.6 F) than their surroundings, interactions such as sitting, touching, or leaning leave residual heat traces. These fading imprints serve as passive temporal codes, allowing for the inference of recent events that exceed the capabilities of RGB cameras. This work proposes a time-reversed reconstruction framework that uses paired RGB and thermal images to recover scene states from a few seconds earlier. The proposed approach couples Visual-Language Models (VLMs) with a constrained diffusion process, where one VLM generates scene descriptions and another guides image reconstruction, ensuring semantic and structural consistency. The method is evaluated in three controlled scenarios, demonstrating the feasibility of reconstructing plausible past frames up to 120 seconds earlier, providing a first step toward time-reversed imaging from thermal traces.
>
---
#### [new 056] Teleportraits: Training-Free People Insertion into Any Scene
- **分类: cs.CV**

- **简介: 该论文属于图像合成任务，旨在解决将参考图像中的人物真实地插入到任意背景场景中的问题。现有方法通常需专门训练且效果有限。本文提出一种无需训练的新方法，利用预训练扩散模型，结合反转技术和注意力机制，实现高质量人物插入与个性化，保持身份和场景一致性。**

- **链接: [http://arxiv.org/pdf/2510.05660v1](http://arxiv.org/pdf/2510.05660v1)**

> **作者:** Jialu Gao; K J Joseph; Fernando De La Torre
>
> **摘要:** The task of realistically inserting a human from a reference image into a background scene is highly challenging, requiring the model to (1) determine the correct location and poses of the person and (2) perform high-quality personalization conditioned on the background. Previous approaches often treat them as separate problems, overlooking their interconnections, and typically rely on training to achieve high performance. In this work, we introduce a unified training-free pipeline that leverages pre-trained text-to-image diffusion models. We show that diffusion models inherently possess the knowledge to place people in complex scenes without requiring task-specific training. By combining inversion techniques with classifier-free guidance, our method achieves affordance-aware global editing, seamlessly inserting people into scenes. Furthermore, our proposed mask-guided self-attention mechanism ensures high-quality personalization, preserving the subject's identity, clothing, and body features from just a single reference image. To the best of our knowledge, we are the first to perform realistic human insertions into scenes in a training-free manner and achieve state-of-the-art results in diverse composite scene images with excellent identity preservation in backgrounds and subjects.
>
---
#### [new 057] $\bf{D^3}$QE: Learning Discrete Distribution Discrepancy-aware Quantization Error for Autoregressive-Generated Image Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像检测任务，旨在解决视觉自回归（AR）模型生成图像的检测问题。针对AR模型生成图像的离散token预测特性，论文提出D³QE方法，利用真实与生成图像在向量量化表示上的频率分布差异，通过结合动态码本统计与语义特征，提升检测准确性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.05891v1](http://arxiv.org/pdf/2510.05891v1)**

> **作者:** Yanran Zhang; Bingyao Yu; Yu Zheng; Wenzhao Zheng; Yueqi Duan; Lei Chen; Jie Zhou; Jiwen Lu
>
> **备注:** 10 pages, 5 figures, published to ICCV2025
>
> **摘要:** The emergence of visual autoregressive (AR) models has revolutionized image generation while presenting new challenges for synthetic image detection. Unlike previous GAN or diffusion-based methods, AR models generate images through discrete token prediction, exhibiting both marked improvements in image synthesis quality and unique characteristics in their vector-quantized representations. In this paper, we propose to leverage Discrete Distribution Discrepancy-aware Quantization Error (D$^3$QE) for autoregressive-generated image detection that exploits the distinctive patterns and the frequency distribution bias of the codebook existing in real and fake images. We introduce a discrete distribution discrepancy-aware transformer that integrates dynamic codebook frequency statistics into its attention mechanism, fusing semantic features and quantization error latent. To evaluate our method, we construct a comprehensive dataset termed ARForensics covering 7 mainstream visual AR models. Experiments demonstrate superior detection accuracy and strong generalization of D$^3$QE across different AR models, with robustness to real-world perturbations. Code is available at \href{https://github.com/Zhangyr2022/D3QE}{https://github.com/Zhangyr2022/D3QE}.
>
---
#### [new 058] Detection and Measurement of Hailstones with Multimodal Large Language Models
- **分类: cs.CV; cs.AI; 68T07, 68T45, 86A10; I.4; I.2**

- **简介: 该论文属于图像检测与测量任务，旨在利用多模态大语言模型从社交媒体图片中检测并测量冰雹直径。研究使用474张奥地利冰雹事件图片，比较不同模型与提示策略的效果，探索无需微调的现成模型在冰雹测量中的潜力。**

- **链接: [http://arxiv.org/pdf/2510.06008v1](http://arxiv.org/pdf/2510.06008v1)**

> **作者:** Moritz Alker; David C. Schedl; Andreas Stöckl
>
> **备注:** 6 pages, 5 figures, accepted at The 2nd International Conference on Electrical and Computer Engineering Researches
>
> **摘要:** This study examines the use of social media and news images to detect and measure hailstones, utilizing pre-trained multimodal large language models. The dataset for this study comprises 474 crowdsourced images of hailstones from documented hail events in Austria, which occurred between January 2022 and September 2024. These hailstones have maximum diameters ranging from 2 to 11cm. We estimate the hail diameters and compare four different models utilizing one-stage and two-stage prompting strategies. The latter utilizes additional size cues from reference objects, such as human hands, within the image. Our results show that pretrained models already have the potential to measure hailstone diameters from images with an average mean absolute error of 1.12cm for the best model. In comparison to a single-stage prompt, two-stage prompting improves the reliability of most models. Our study suggests that these off-the-shelf models, even without fine-tuning, can complement traditional hail sensors by extracting meaningful and spatially dense information from social media imagery, enabling faster and more detailed assessments of severe weather events. The automated real-time image harvesting from social media and other sources remains an open task, but it will make our approach directly applicable to future hail events.
>
---
#### [new 059] LightCache: Memory-Efficient, Training-Free Acceleration for Video Generation
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视频生成任务，旨在解决扩散模型推理过程中内存消耗大、速度慢的问题。通过分析推理阶段特性，提出三种阶段特定策略：异步缓存交换、特征块处理和切片解码，在降低内存使用的同时提升推理速度，且保证生成质量。**

- **链接: [http://arxiv.org/pdf/2510.05367v1](http://arxiv.org/pdf/2510.05367v1)**

> **作者:** Yang Xiao; Gen Li; Kaiyuan Deng; Yushu Wu; Zheng Zhan; Yanzhi Wang; Xiaolong Ma; Bo Hui
>
> **摘要:** Training-free acceleration has emerged as an advanced research area in video generation based on diffusion models. The redundancy of latents in diffusion model inference provides a natural entry point for acceleration. In this paper, we decompose the inference process into the encoding, denoising, and decoding stages, and observe that cache-based acceleration methods often lead to substantial memory surges in the latter two stages. To address this problem, we analyze the characteristics of inference across different stages and propose stage-specific strategies for reducing memory consumption: 1) Asynchronous Cache Swapping. 2) Feature chunk. 3) Slicing latents to decode. At the same time, we ensure that the time overhead introduced by these three strategies remains lower than the acceleration gains themselves. Compared with the baseline, our approach achieves faster inference speed and lower memory usage, while maintaining quality degradation within an acceptable range. The Code is available at https://github.com/NKUShaw/LightCache .
>
---
#### [new 060] A public cardiac CT dataset featuring the left atrial appendage
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分割任务，旨在解决左心耳、冠状动脉和肺静脉的精确分割难题。论文贡献了一个开源高质量心脏CT数据集，包含精细分割标注，并优化了原有标注缺陷，以推动相关研究发展。**

- **链接: [http://arxiv.org/pdf/2510.06090v1](http://arxiv.org/pdf/2510.06090v1)**

> **作者:** Bjoern Hansen; Jonas Pedersen; Klaus F. Kofoed; Oscar Camara; Rasmus R. Paulsen; Kristine Soerensen
>
> **备注:** 8 pages, 5 figures, published at STACOM2025
>
> **摘要:** Despite the success of advanced segmentation frameworks such as TotalSegmentator (TS), accurate segmentations of the left atrial appendage (LAA), coronary arteries (CAs), and pulmonary veins (PVs) remain a significant challenge in medical imaging. In this work, we present the first open-source, anatomically coherent dataset of curated, high-resolution segmentations for these structures, supplemented with whole-heart labels produced by TS on the publicly available ImageCAS dataset consisting of 1000 cardiac computed tomography angiography (CCTA) scans. One purpose of the data set is to foster novel approaches to the analysis of LAA morphology. LAA segmentations on ImageCAS were generated using a state-of-the-art segmentation framework developed specifically for high resolution LAA segmentation. We trained the network on a large private dataset with manual annotations provided by medical readers guided by a trained cardiologist and transferred the model to ImageCAS data. CA labels were improved from the original ImageCAS annotations, while PV segmentations were refined from TS outputs. In addition, we provide a list of scans from ImageCAS that contains common data flaws such as step artefacts, LAAs extending beyond the scanner's field of view, and other types of data defects.
>
---
#### [new 061] Context Matters: Learning Global Semantics for Visual Reasoning and Comprehension
- **分类: cs.CV**

- **简介: 该论文属于视觉语义建模任务，旨在解决视觉模型在推理和上下文学习方面落后于语言模型的问题。论文提出了一种基于对象级语义的视觉建模方法，通过掩码图像建模框架验证对象级表示对学习真实世界分布的重要性，并展示了其在多模态问答任务中的优异表现。**

- **链接: [http://arxiv.org/pdf/2510.05674v1](http://arxiv.org/pdf/2510.05674v1)**

> **作者:** Jike Zhong; Yuxiang Lai; Xiaofeng Yang; Konstantinos Psounis
>
> **摘要:** Recent advances in language modeling have witnessed the rise of highly desirable emergent capabilities, such as reasoning and in-context learning. However, vision models have yet to exhibit comparable progress in these areas. In this paper, we argue that this gap could stem from the lack of semantic and contextual guidance in current vision transformer (ViT) training schemes, and such a gap can be narrowed through the design of a semantic-grounded objective. Specifically, we notice that individual words in natural language are inherently semantic, and modeling directly on word tokens naturally learns a realistic distribution. In contrast, ViTs rely on spatial patchification, which inevitably lacks semantic information. To bridge this gap, we propose to directly model "object" as the visual equivalence of "word," pushing the model to learn the global context and semantics among visual elements. We investigate our hypotheses via masked image modeling (MIM), a framework where our approach can be readily tested by applying masks to visual objects rather than random patches. Considerable evidence from qualitative and quantitative evaluations reveals a key finding: object-level representation alone helps to learn a real-world distribution, whereas pixel-averaging shortcuts are often learned without it. Moreover, further evaluations with multimodal LLMs (MLLM) on visual question answering (VQA, GQA, ScienceQA) tasks demonstrate the strong reasoning and contextual understanding gained with this simple objective. We hope our study highlights the effectiveness of object-level encoding and provides a plausible direction for developing stronger vision encoders and tokenizers. Code and model will be publicly released. Keywords: Semantic Visual Tokenizer, Vision Reasoning, In-context Learning, Multimodal Reasoning
>
---
#### [new 062] Mysteries of the Deep: Role of Intermediate Representations in Out of Distribution Detection
- **分类: cs.CV**

- **简介: 该论文属于机器学习中的分布外（OOD）检测任务，旨在提升模型在未知数据上的可靠性。论文提出利用预训练模型中间层表示，通过熵准则自动选择最具互补信息的层，以提升OOD检测性能。实验表明方法在不同模型架构和训练目标下均有效。**

- **链接: [http://arxiv.org/pdf/2510.05782v1](http://arxiv.org/pdf/2510.05782v1)**

> **作者:** I. M. De la Jara; C. Rodriguez-Opazo; D. Teney; D. Ranasinghe; E. Abbasnejad
>
> **备注:** 28
>
> **摘要:** Out-of-distribution (OOD) detection is essential for reliably deploying machine learning models in the wild. Yet, most methods treat large pre-trained models as monolithic encoders and rely solely on their final-layer representations for detection. We challenge this wisdom. We reveal the \textit{intermediate layers} of pre-trained models, shaped by residual connections that subtly transform input projections, \textit{can} encode \textit{surprisingly rich and diverse signals} for detecting distributional shifts. Importantly, to exploit latent representation diversity across layers, we introduce an entropy-based criterion to \textit{automatically} identify layers offering the most complementary information in a training-free setting -- \textit{without access to OOD data}. We show that selectively incorporating these intermediate representations can increase the accuracy of OOD detection by up to \textbf{$10\%$} in far-OOD and over \textbf{$7\%$} in near-OOD benchmarks compared to state-of-the-art training-free methods across various model architectures and training objectives. Our findings reveal a new avenue for OOD detection research and uncover the impact of various training objectives and model architectures on confidence-based OOD detection methods.
>
---
#### [new 063] Ocular-Induced Abnormal Head Posture: Diagnosis and Missing Data Imputation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于医学图像分析与临床数据处理任务，旨在解决眼部异常头位的自动诊断与缺失数据填补问题。研究提出了AHP-CADNet用于多特征融合诊断，并设计基于课程学习的框架提升缺失数据恢复能力，从而提高临床诊断的准确性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.05649v1](http://arxiv.org/pdf/2510.05649v1)**

> **作者:** Saja Al-Dabet; Sherzod Turaev; Nazar Zaki; Arif O. Khan; Luai Eldweik
>
> **摘要:** Ocular-induced abnormal head posture (AHP) is a compensatory mechanism that arises from ocular misalignment conditions, such as strabismus, enabling patients to reduce diplopia and preserve binocular vision. Early diagnosis minimizes morbidity and secondary complications such as facial asymmetry; however, current clinical assessments remain largely subjective and are further complicated by incomplete medical records. This study addresses both challenges through two complementary deep learning frameworks. First, AHP-CADNet is a multi-level attention fusion framework for automated diagnosis that integrates ocular landmarks, head pose features, and structured clinical attributes to generate interpretable predictions. Second, a curriculum learning-based imputation framework is designed to mitigate missing data by progressively leveraging structured variables and unstructured clinical notes to enhance diagnostic robustness under realistic data conditions. Evaluation on the PoseGaze-AHP dataset demonstrates robust diagnostic performance. AHP-CADNet achieves 96.9-99.0 percent accuracy across classification tasks and low prediction errors for continuous variables, with MAE ranging from 0.103 to 0.199 and R2 exceeding 0.93. The imputation framework maintains high accuracy across all clinical variables (93.46-99.78 percent with PubMedBERT), with clinical dependency modeling yielding significant improvements (p < 0.001). These findings confirm the effectiveness of both frameworks for automated diagnosis and recovery from missing data in clinical settings.
>
---
#### [new 064] HoloScene: Simulation-Ready Interactive 3D Worlds from a Single Video
- **分类: cs.CV**

- **简介: 该论文提出HoloScene，旨在解决从单个视频生成具备几何完整性、物理合理性和逼真渲染的交互式3D虚拟环境的问题。它属于3D重建与场景理解任务。通过引入包含几何、外观和物理属性的场景图表示，并结合能量优化与混合优化方法，实现了高质量、可交互的数字孪生重建。**

- **链接: [http://arxiv.org/pdf/2510.05560v1](http://arxiv.org/pdf/2510.05560v1)**

> **作者:** Hongchi Xia; Chih-Hao Lin; Hao-Yu Hsu; Quentin Leboutet; Katelyn Gao; Michael Paulitsch; Benjamin Ummenhofer; Shenlong Wang
>
> **备注:** Project page: https://xiahongchi.github.io/HoloScene
>
> **摘要:** Digitizing the physical world into accurate simulation-ready virtual environments offers significant opportunities in a variety of fields such as augmented and virtual reality, gaming, and robotics. However, current 3D reconstruction and scene-understanding methods commonly fall short in one or more critical aspects, such as geometry completeness, object interactivity, physical plausibility, photorealistic rendering, or realistic physical properties for reliable dynamic simulation. To address these limitations, we introduce HoloScene, a novel interactive 3D reconstruction framework that simultaneously achieves these requirements. HoloScene leverages a comprehensive interactive scene-graph representation, encoding object geometry, appearance, and physical properties alongside hierarchical and inter-object relationships. Reconstruction is formulated as an energy-based optimization problem, integrating observational data, physical constraints, and generative priors into a unified, coherent objective. Optimization is efficiently performed via a hybrid approach combining sampling-based exploration with gradient-based refinement. The resulting digital twins exhibit complete and precise geometry, physical stability, and realistic rendering from novel viewpoints. Evaluations conducted on multiple benchmark datasets demonstrate superior performance, while practical use-cases in interactive gaming and real-time digital-twin manipulation illustrate HoloScene's broad applicability and effectiveness. Project page: https://xiahongchi.github.io/HoloScene.
>
---
#### [new 065] Emergent AI Surveillance: Overlearned Person Re-Identification and Its Mitigation in Law Enforcement Context
- **分类: cs.CV; cs.AI; cs.CY; cs.LG**

- **简介: 论文研究AI模型在无意识情况下通过过度学习实现个体重识别，带来隐私风险。任务是分析其问题并提出缓解方法。工作包括评估“索引排除”和“混淆损失”两种技术，以降低识别准确率，同时保持非人物体检索性能。**

- **链接: [http://arxiv.org/pdf/2510.06026v1](http://arxiv.org/pdf/2510.06026v1)**

> **作者:** An Thi Nguyen; Radina Stoykova; Eric Arazo
>
> **备注:** 10 pages, accepted to AIES 2025
>
> **摘要:** Generic instance search models can dramatically reduce the manual effort required to analyze vast surveillance footage during criminal investigations by retrieving specific objects of interest to law enforcement. However, our research reveals an unintended emergent capability: through overlearning, these models can single out specific individuals even when trained on datasets without human subjects. This capability raises concerns regarding identification and profiling of individuals based on their personal data, while there is currently no clear standard on how de-identification can be achieved. We evaluate two technical safeguards to curtail a model's person re-identification capacity: index exclusion and confusion loss. Our experiments demonstrate that combining these approaches can reduce person re-identification accuracy to below 2% while maintaining 82% of retrieval performance for non-person objects. However, we identify critical vulnerabilities in these mitigations, including potential circumvention using partial person images. These findings highlight urgent regulatory questions at the intersection of AI governance and data protection: How should we classify and regulate systems with emergent identification capabilities? And what technical standards should be required to prevent identification capabilities from developing in seemingly benign applications?
>
---
#### [new 066] Be Tangential to Manifold: Discovering Riemannian Metric for Diffusion Models
- **分类: cs.CV**

- **简介: 该论文属于生成模型任务，旨在解决扩散模型缺乏显式低维流形空间的问题。通过提出一种基于得分函数雅可比矩阵的黎曼度量，使噪声空间中的测地线更贴合数据流形，从而提升图像插值的自然性和保真度。**

- **链接: [http://arxiv.org/pdf/2510.05509v1](http://arxiv.org/pdf/2510.05509v1)**

> **作者:** Shinnosuke Saito; Takashi Matsubara
>
> **摘要:** Diffusion models are powerful deep generative models (DGMs) that generate high-fidelity, diverse content. However, unlike classical DGMs, they lack an explicit, tractable low-dimensional latent space that parameterizes the data manifold. This absence limits manifold-aware analysis and operations, such as interpolation and editing. Existing interpolation methods for diffusion models typically follow paths through high-density regions, which are not necessarily aligned with the data manifold and can yield perceptually unnatural transitions. To exploit the data manifold learned by diffusion models, we propose a novel Riemannian metric on the noise space, inspired by recent findings that the Jacobian of the score function captures the tangent spaces to the local data manifold. This metric encourages geodesics in the noise space to stay within or run parallel to the learned data manifold. Experiments on image interpolation show that our metric produces perceptually more natural and faithful transitions than existing density-based and naive baselines.
>
---
#### [new 067] SkinMap: Weighted Full-Body Skin Segmentation for Robust Remote Photoplethysmography
- **分类: cs.CV; eess.IV**

- **简介: 该论文属于远程光体积描记（rPPG）任务，旨在通过摄像头监测心率等生命体征。为提升rPPG在复杂场景下的准确性，论文提出了一种加权全身体肤分割方法SkinMap，优先选取高质量皮肤区域，排除口、眼、头发等干扰区域。论文还提出了一个新的数据集SYNC-rPPG，并验证了该方法在运动和不同肤色场景下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.05296v1](http://arxiv.org/pdf/2510.05296v1)**

> **作者:** Zahra Maleki; Amirhossein Akbari; Amirhossein Binesh; Babak Khalaj
>
> **摘要:** Remote photoplethysmography (rPPG) is an innovative method for monitoring heart rate and vital signs by using a simple camera to record a person, as long as any part of their skin is visible. This low-cost, contactless approach helps in remote patient monitoring, emotion analysis, smart vehicle utilization, and more. Over the years, various techniques have been proposed to improve the accuracy of this technology, especially given its sensitivity to lighting and movement. In the unsupervised pipeline, it is necessary to first select skin regions from the video to extract the rPPG signal from the skin color changes. We introduce a novel skin segmentation technique that prioritizes skin regions to enhance the quality of the extracted signal. It can detect areas of skin all over the body, making it more resistant to movement, while removing areas such as the mouth, eyes, and hair that may cause interference. Our model is evaluated on publicly available datasets, and we also present a new dataset, called SYNC-rPPG, to better represent real-world conditions. The results indicate that our model demonstrates a prior ability to capture heartbeats in challenging conditions, such as talking and head rotation, and maintain the mean absolute error (MAE) between predicted and actual heart rates, while other methods fail to do so. In addition, we demonstrate high accuracy in detecting a diverse range of skin tones, making this technique a promising option for real-world applications.
>
---
#### [new 068] Human Action Recognition from Point Clouds over Time
- **分类: cs.CV**

- **简介: 该论文属于人类动作识别（HAR）任务，旨在从时序点云数据中识别动作。为解决传统方法依赖骨骼或视频的问题，论文提出一种结合点云处理与稀疏卷积网络的新框架，并引入辅助特征提升精度，最终在NTU RGB-D 120数据集上取得优异表现。**

- **链接: [http://arxiv.org/pdf/2510.05506v1](http://arxiv.org/pdf/2510.05506v1)**

> **作者:** James Dickens
>
> **摘要:** Recent research into human action recognition (HAR) has focused predominantly on skeletal action recognition and video-based methods. With the increasing availability of consumer-grade depth sensors and Lidar instruments, there is a growing opportunity to leverage dense 3D data for action recognition, to develop a third way. This paper presents a novel approach for recognizing actions from 3D videos by introducing a pipeline that segments human point clouds from the background of a scene, tracks individuals over time, and performs body part segmentation. The method supports point clouds from both depth sensors and monocular depth estimation. At the core of the proposed HAR framework is a novel backbone for 3D action recognition, which combines point-based techniques with sparse convolutional networks applied to voxel-mapped point cloud sequences. Experiments incorporate auxiliary point features including surface normals, color, infrared intensity, and body part parsing labels, to enhance recognition accuracy. Evaluation on the NTU RGB- D 120 dataset demonstrates that the method is competitive with existing skeletal action recognition algorithms. Moreover, combining both sensor-based and estimated depth inputs in an ensemble setup, this approach achieves 89.3% accuracy when different human subjects are considered for training and testing, outperforming previous point cloud action recognition methods.
>
---
#### [new 069] Seeing the Big Picture: Evaluating Multimodal LLMs' Ability to Interpret and Grade Handwritten Student Work
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于教育评估任务，旨在评估多模态大语言模型（MLLMs）解读和评分学生手写作业的能力。研究通过两个实验探讨MLLM在数学手写作业评分中的应用，实验A评估算术题答案，模型表现接近人类；实验B评估学生数学绘图，直接分析效果不佳，但结合人工描述后显著提升。论文展示了MLLM在教育场景中的潜力与局限。**

- **链接: [http://arxiv.org/pdf/2510.05538v1](http://arxiv.org/pdf/2510.05538v1)**

> **作者:** Owen Henkel; Bill Roberts; Doug Jaffe; Laurence Holt
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) raise the question of their potential for grading, analyzing, and offering feedback on handwritten student classwork. This capability would be particularly beneficial in elementary and middle-school mathematics education, where most work remains handwritten, because seeing students' full working of a problem provides valuable insights into their learning processes, but is extremely time-consuming to grade. We present two experiments investigating MLLM performance on handwritten student mathematics classwork. Experiment A examines 288 handwritten responses from Ghanaian middle school students solving arithmetic problems with objective answers. In this context, models achieved near-human accuracy (95%, k = 0.90) but exhibited occasional errors that human educators would be unlikely to make. Experiment B evaluates 150 mathematical illustrations from American elementary students, where the drawings are the answer to the question. These tasks lack single objective answers and require sophisticated visual interpretation as well as pedagogical judgment in order to analyze and evaluate them. We attempted to separate MLLMs' visual capabilities from their pedagogical abilities by first asking them to grade the student illustrations directly, and then by augmenting the image with a detailed human description of the illustration. We found that when the models had to analyze the student illustrations directly, they struggled, achieving only k = 0.20 with ground truth scores, but when given human descriptions, their agreement levels improved dramatically to k = 0.47, which was in line with human-to-human agreement levels. This gap suggests MLLMs can "see" and interpret arithmetic work relatively well, but still struggle to "see" student mathematical illustrations.
>
---
#### [new 070] HOI-R1: Exploring the Potential of Multimodal Large Language Models for Human-Object Interaction Detection
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于人-物交互检测（HOID）任务，旨在解决现有方法依赖视觉语言模型（VLMs）和复杂框架的问题。作者提出HOI-R1，首次探索仅用多模态大语言模型（MLLM）和强化学习（RL）进行纯文本交互检测，无需额外检测模块。实验表明其准确率是基线的2倍，且具备良好泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.05609v1](http://arxiv.org/pdf/2510.05609v1)**

> **作者:** Junwen Chen; Peilin Xiong; Keiji Yanai
>
> **摘要:** Recent Human-object interaction detection (HOID) methods highly require prior knowledge from VLMs to enhance the interaction recognition capabilities. The training strategies and model architectures for connecting the knowledge from VLMs to the HOI instance representations from the object detector are challenging, and the whole framework is complex for further development or application. On the other hand, the inherent reasoning abilities of MLLMs on human-object interaction detection are under-explored. Inspired by the recent success of training MLLMs with reinforcement learning (RL) methods, we propose HOI-R1 and first explore the potential of the language model on the HOID task without any additional detection modules. We introduce an HOI reasoning process and HOID reward functions to solve the HOID task by pure text. The results on the HICO-DET dataset show that HOI-R1 achieves 2x the accuracy of the baseline with great generalization ability. The source code is available at https://github.com/cjw2021/HOI-R1.
>
---
#### [new 071] Fine-Tuned CNN-Based Approach for Multi-Class Mango Leaf Disease Detection
- **分类: cs.CV**

- **简介: 该论文属于图像分类任务，旨在解决芒果叶病害的多类别自动识别问题。作者采用迁移学习与微调策略，对比了五种预训练CNN模型的性能。实验表明，DenseNet201在八类病害识别中达到最高准确率99.33%，适用于智能农业中的精准病害检测。**

- **链接: [http://arxiv.org/pdf/2510.05326v1](http://arxiv.org/pdf/2510.05326v1)**

> **作者:** Jalal Ahmmed; Faruk Ahmed; Rashedul Hasan Shohan; Md. Mahabub Rana; Mahdi Hasan
>
> **备注:** Double column 6 pages, 10 figures, ieee conference style
>
> **摘要:** Mango is an important fruit crop in South Asia, but its cultivation is frequently hampered by leaf diseases that greatly impact yield and quality. This research examines the performance of five pre-trained convolutional neural networks, DenseNet201, InceptionV3, ResNet152V2, SeResNet152, and Xception, for multi-class identification of mango leaf diseases across eight classes using a transfer learning strategy with fine-tuning. The models were assessed through standard evaluation metrics, such as accuracy, precision, recall, F1-score, and confusion matrices. Among the architectures tested, DenseNet201 delivered the best results, achieving 99.33% accuracy with consistently strong metrics for individual classes, particularly excelling in identifying Cutting Weevil and Bacterial Canker. Moreover, ResNet152V2 and SeResNet152 provided strong outcomes, whereas InceptionV3 and Xception exhibited lower performance in visually similar categories like Sooty Mould and Powdery Mildew. The training and validation plots demonstrated stable convergence for the highest-performing models. The capability of fine-tuned transfer learning models, for precise and dependable multi-class mango leaf disease detection in intelligent agricultural applications.
>
---
#### [new 072] Teamwork: Collaborative Diffusion with Low-rank Coordination and Adaptation
- **分类: cs.CV; cs.GR; cs.LG**

- **简介: 该论文属于图像生成与逆向图形任务，旨在解决扩散模型在多通道输入输出及任务适配中的局限性。论文提出Teamwork方法，通过低秩协调与适配多个预训练模型，实现灵活高效的通道扩展和跨任务适配。**

- **链接: [http://arxiv.org/pdf/2510.05532v1](http://arxiv.org/pdf/2510.05532v1)**

> **作者:** Sam Sartor; Pieter Peers
>
> **摘要:** Large pretrained diffusion models can provide strong priors beneficial for many graphics applications. However, generative applications such as neural rendering and inverse methods such as SVBRDF estimation and intrinsic image decomposition require additional input or output channels. Current solutions for channel expansion are often application specific and these solutions can be difficult to adapt to different diffusion models or new tasks. This paper introduces Teamwork: a flexible and efficient unified solution for jointly increasing the number of input and output channels as well as adapting a pretrained diffusion model to new tasks. Teamwork achieves channel expansion without altering the pretrained diffusion model architecture by coordinating and adapting multiple instances of the base diffusion model (\ie, teammates). We employ a novel variation of Low Rank-Adaptation (LoRA) to jointly address both adaptation and coordination between the different teammates. Furthermore Teamwork supports dynamic (de)activation of teammates. We demonstrate the flexibility and efficiency of Teamwork on a variety of generative and inverse graphics tasks such as inpainting, single image SVBRDF estimation, intrinsic decomposition, neural shading, and intrinsic image synthesis.
>
---
#### [new 073] OneVision: An End-to-End Generative Framework for Multi-view E-commerce Vision Search
- **分类: cs.CV**

- **简介: 该论文属于电商视觉搜索任务，旨在解决多阶段级联架构中多视角表征差异影响用户体验与转化的问题。论文提出OneVision框架，通过视觉对齐残差量化编码与多阶段语义对齐，实现端到端生成式搜索，在提升效率的同时优化点击率、转化率与订单量。**

- **链接: [http://arxiv.org/pdf/2510.05759v1](http://arxiv.org/pdf/2510.05759v1)**

> **作者:** Zexin Zheng; Huangyu Dai; Lingtao Mao; Xinyu Sun; Zihan Liang; Ben Chen; Yuqing Ding; Chenyi Lei; Wenwu Ou; Han Li; Kun Gai
>
> **摘要:** Traditional vision search, similar to search and recommendation systems, follows the multi-stage cascading architecture (MCA) paradigm to balance efficiency and conversion. Specifically, the query image undergoes feature extraction, recall, pre-ranking, and ranking stages, ultimately presenting the user with semantically similar products that meet their preferences. This multi-view representation discrepancy of the same object in the query and the optimization objective collide across these stages, making it difficult to achieve Pareto optimality in both user experience and conversion. In this paper, an end-to-end generative framework, OneVision, is proposed to address these problems. OneVision builds on VRQ, a vision-aligned residual quantization encoding, which can align the vastly different representations of an object across multiple viewpoints while preserving the distinctive features of each product as much as possible. Then a multi-stage semantic alignment scheme is adopted to maintain strong visual similarity priors while effectively incorporating user-specific information for personalized preference generation. In offline evaluations, OneVision performs on par with online MCA, while improving inference efficiency by 21% through dynamic pruning. In A/B tests, it achieves significant online improvements: +2.15% item CTR, +2.27% CVR, and +3.12% order volume. These results demonstrate that a semantic ID centric, generative architecture can unify retrieval and personalization while simplifying the serving pathway.
>
---
#### [new 074] BioAutoML-NAS: An End-to-End AutoML Framework for Multimodal Insect Classification via Neural Architecture Search on Large-Scale Biodiversity Data
- **分类: cs.CV**

- **简介: 该论文属于昆虫分类任务，旨在解决农业与生态研究中昆虫分类的挑战。论文提出了BioAutoML-NAS，一种基于神经架构搜索的多模态自动机器学习框架，融合图像与元数据，提升分类性能。实验表明其在大规模数据集上表现优异，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.05888v1](http://arxiv.org/pdf/2510.05888v1)**

> **作者:** Arefin Ittesafun Abian; Debopom Sutradhar; Md Rafi Ur Rashid; Reem E. Mohamed; Md Rafiqul Islam; Asif Karim; Kheng Cher Yeo; Sami Azam
>
> **摘要:** Insect classification is important for agricultural management and ecological research, as it directly affects crop health and production. However, this task remains challenging due to the complex characteristics of insects, class imbalance, and large-scale datasets. To address these issues, we propose BioAutoML-NAS, the first BioAutoML model using multimodal data, including images, and metadata, which applies neural architecture search (NAS) for images to automatically learn the best operations for each connection within each cell. Multiple cells are stacked to form the full network, each extracting detailed image feature representations. A multimodal fusion module combines image embeddings with metadata, allowing the model to use both visual and categorical biological information to classify insects. An alternating bi-level optimization training strategy jointly updates network weights and architecture parameters, while zero operations remove less important connections, producing sparse, efficient, and high-performing architectures. Extensive evaluation on the BIOSCAN-5M dataset demonstrates that BioAutoML-NAS achieves 96.81% accuracy, 97.46% precision, 96.81% recall, and a 97.05% F1 score, outperforming state-of-the-art transfer learning, transformer, AutoML, and NAS methods by approximately 16%, 10%, and 8% respectively. Further validation on the Insects-1M dataset obtains 93.25% accuracy, 93.71% precision, 92.74% recall, and a 93.22% F1 score. These results demonstrate that BioAutoML-NAS provides accurate, confident insect classification that supports modern sustainable farming.
>
---
#### [new 075] Beyond Monolithic Rewards: A Hybrid and Multi-Aspect Reward Optimization for MLLM Alignment
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态大语言模型（MLLM）对齐任务，旨在解决单一奖励信号在对齐人类偏好时的局限性。论文提出了一种混合奖励建模框架，结合模型基础奖励与规则基础奖励，并引入多方面奖励和长度惩罚机制，以提升模型在多模态任务上的表现。实验表明该方法在通用和数学推理任务上均有显著改进。**

- **链接: [http://arxiv.org/pdf/2510.05283v1](http://arxiv.org/pdf/2510.05283v1)**

> **作者:** Radha Gulhane; Sathish Reddy Indurthi
>
> **摘要:** Aligning multimodal large language models (MLLMs) with human preferences often relies on single-signal, model-based reward methods. Such monolithic rewards often lack confidence calibration across domain-specific tasks, fail to capture diverse aspects of human preferences, and require extensive data annotation and reward model training. In this work, we propose a hybrid reward modeling framework that integrates complementary reward paradigms: (i) model-based rewards, where a learned reward model predicts scalar or vector scores from synthetic and human feedback, and (ii) rule-based rewards, where domain-specific heuristics provide explicit correctness signals with confidence. Beyond accuracy, we further incorporate multi-aspect rewards to enforce instruction adherence and introduce a generalized length-penalty reward to stabilize training and improve performance. The proposed framework provides a flexible and effective approach to aligning MLLMs through reinforcement learning policy optimization. Our experiments show consistent improvements across different multimodal benchmarks when applying hybrid and multi-aspect reward modeling. Our best performing model in the 3B family achieves an overall average improvement of ~9.5% across general and math reasoning tasks. Focusing specifically on mathematical benchmarks, the model achieves a significant average improvement of ~16%, highlighting its effectiveness in mathematical reasoning and problem solving.
>
---
#### [new 076] RegMix: Adversarial Mutual and Generalization Regularization for Enhancing DNN Robustness
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于对抗训练任务，旨在提升深度神经网络对抗攻击的鲁棒性。现有方法使用均方误差（MSE）作为正则项，但其优化过于均匀，限制了鲁棒性。论文提出RegMix，包含两种新策略：加权对抗互信息正则化和对抗泛化正则化，通过改进损失函数提升模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.05317v1](http://arxiv.org/pdf/2510.05317v1)**

> **作者:** Zhenyu Liu; Varun Ojha
>
> **摘要:** Adversarial training is the most effective defense against adversarial attacks. The effectiveness of the adversarial attacks has been on the design of its loss function and regularization term. The most widely used loss function in adversarial training is cross-entropy and mean squared error (MSE) as its regularization objective. However, MSE enforces overly uniform optimization between two output distributions during training, which limits its robustness in adversarial training scenarios. To address this issue, we revisit the idea of mutual learning (originally designed for knowledge distillation) and propose two novel regularization strategies tailored for adversarial training: (i) weighted adversarial mutual regularization and (ii) adversarial generalization regularization. In the former, we formulate a decomposed adversarial mutual Kullback-Leibler divergence (KL-divergence) loss, which allows flexible control over the optimization process by assigning unequal weights to the main and auxiliary objectives. In the latter, we introduce an additional clean target distribution into the adversarial training objective, improving generalization and enhancing model robustness. Extensive experiments demonstrate that our proposed methods significantly improve adversarial robustness compared to existing regularization-based approaches.
>
---
#### [new 077] Leveraging Vision Transformers for Enhanced Classification of Emotions using ECG Signals
- **分类: eess.SP; cs.CV**

- **简介: 该论文属于情感分类任务，旨在通过心电图（ECG）信号提升情绪识别效果。论文提出一种改进的视觉Transformer（ViT）模型，融合CNN和SE模块，并采用小波变换和谱分析预处理ECG信号。在YAAD和DREAMER数据集上验证，该方法在情绪分类表现上超越现有技术。**

- **链接: [http://arxiv.org/pdf/2510.05826v1](http://arxiv.org/pdf/2510.05826v1)**

> **作者:** Pubudu L. Indrasiri; Bipasha Kashyap; Pubudu N. Pathirana
>
> **备注:** 14pages, 2 figures
>
> **摘要:** Biomedical signals provide insights into various conditions affecting the human body. Beyond diagnostic capabilities, these signals offer a deeper understanding of how specific organs respond to an individual's emotions and feelings. For instance, ECG data can reveal changes in heart rate variability linked to emotional arousal, stress levels, and autonomic nervous system activity. This data offers a window into the physiological basis of our emotional states. Recent advancements in the field diverge from conventional approaches by leveraging the power of advanced transformer architectures, which surpass traditional machine learning and deep learning methods. We begin by assessing the effectiveness of the Vision Transformer (ViT), a forefront model in image classification, for identifying emotions in imaged ECGs. Following this, we present and evaluate an improved version of ViT, integrating both CNN and SE blocks, aiming to bolster performance on imaged ECGs associated with emotion detection. Our method unfolds in two critical phases: first, we apply advanced preprocessing techniques for signal purification and converting signals into interpretable images using continuous wavelet transform and power spectral density analysis; second, we unveil a performance-boosted vision transformer architecture, cleverly enhanced with convolutional neural network components, to adeptly tackle the challenges of emotion recognition. Our methodology's robustness and innovation were thoroughly tested using ECG data from the YAAD and DREAMER datasets, leading to remarkable outcomes. For the YAAD dataset, our approach outperformed existing state-of-the-art methods in classifying seven unique emotional states, as well as in valence and arousal classification. Similarly, in the DREAMER dataset, our method excelled in distinguishing between valence, arousal and dominance, surpassing current leading techniques.
>
---
#### [new 078] From Neural Activity to Computation: Biological Reservoirs for Pattern Recognition in Digit Classification
- **分类: cs.NE; cs.AI; cs.CV**

- **简介: 该论文属于模式识别任务，旨在探索生物神经元网络作为计算基质的潜力。研究人员利用培养的生物神经元构建储层计算系统，通过电刺激输入图像并记录神经活动，训练线性分类器进行数字分类。同时与人工储层模型对比，验证生物系统的有效性，推动生物启发式机器学习的发展。**

- **链接: [http://arxiv.org/pdf/2510.05637v1](http://arxiv.org/pdf/2510.05637v1)**

> **作者:** Ludovico Iannello; Luca Ciampi; Fabrizio Tonelli; Gabriele Lagani; Lucio Maria Calcagnile; Federico Cremisi; Angelo Di Garbo; Giuseppe Amato
>
> **备注:** Accepted at HiCV@ICCV2025
>
> **摘要:** In this paper, we present a biologically grounded approach to reservoir computing (RC), in which a network of cultured biological neurons serves as the reservoir substrate. This system, referred to as biological reservoir computing (BRC), replaces artificial recurrent units with the spontaneous and evoked activity of living neurons. A multi-electrode array (MEA) enables simultaneous stimulation and readout across multiple sites: inputs are delivered through a subset of electrodes, while the remaining ones capture the resulting neural responses, mapping input patterns into a high-dimensional biological feature space. We evaluate the system through a case study on digit classification using a custom dataset. Input images are encoded and delivered to the biological reservoir via electrical stimulation, and the corresponding neural activity is used to train a simple linear classifier. To contextualize the performance of the biological system, we also include a comparison with a standard artificial reservoir trained on the same task. The results indicate that the biological reservoir can effectively support classification, highlighting its potential as a viable and interpretable computational substrate. We believe this work contributes to the broader effort of integrating biological principles into machine learning and aligns with the goals of human-inspired vision by exploring how living neural systems can inform the design of efficient and biologically plausible models.
>
---
#### [new 079] Advancing Automated Spatio-Semantic Analysis in Picture Description Using Language Models
- **分类: cs.CL; cs.CV; eess.AS**

- **简介: 该论文属于自然语言处理与认知评估交叉任务，旨在解决图片描述中认知语言障碍自动分析问题。现有方法忽略视觉叙事路径，该研究利用微调BERT模型自动提取并排序内容信息单元（CIU），实现高效评估认知障碍，相关模型与工具已开源。**

- **链接: [http://arxiv.org/pdf/2510.05128v1](http://arxiv.org/pdf/2510.05128v1)**

> **作者:** Si-Ioi Ng; Pranav S. Ambadi; Kimberly D. Mueller; Julie Liss; Visar Berisha
>
> **摘要:** Current methods for automated assessment of cognitive-linguistic impairment via picture description often neglect the visual narrative path - the sequence and locations of elements a speaker described in the picture. Analyses of spatio-semantic features capture this path using content information units (CIUs), but manual tagging or dictionary-based mapping is labor-intensive. This study proposes a BERT-based pipeline, fine tuned with binary cross-entropy and pairwise ranking loss, for automated CIU extraction and ordering from the Cookie Theft picture description. Evaluated by 5-fold cross-validation, it achieves 93% median precision, 96% median recall in CIU detection, and 24% sequence error rates. The proposed method extracts features that exhibit strong Pearson correlations with ground truth, surpassing the dictionary-based baseline in external validation. These features also perform comparably to those derived from manual annotations in evaluating group differences via ANCOVA. The pipeline is shown to effectively characterize visual narrative paths for cognitive impairment assessment, with the implementation and models open-sourced to public.
>
---
#### [new 080] UNIDOC-BENCH: A Unified Benchmark for Document-Centric Multimodal RAG
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态检索增强生成（MM-RAG）任务，旨在解决当前评估方法割裂、无法反映真实文档场景的问题。论文构建了UniDoc-Bench，首个大规模、现实文档基准，包含70k PDF页面和1,600多模态问答对，支持统一比较四种范式，揭示多模态融合的优势与当前嵌入方法的不足。**

- **链接: [http://arxiv.org/pdf/2510.03663v1](http://arxiv.org/pdf/2510.03663v1)**

> **作者:** Xiangyu Peng; Cab Qin; Zeyuan Chen; Ran Xu; Caiming Xiong; Chien-Sheng Wu
>
> **摘要:** Multimodal retrieval-augmented generation (MM-RAG) is a key approach for applying large language models (LLMs) and agents to real-world knowledge bases, yet current evaluations are fragmented, focusing on either text or images in isolation or on simplified multimodal setups that fail to capture document-centric multimodal use cases. In this paper, we introduce UniDoc-Bench, the first large-scale, realistic benchmark for MM-RAG built from 70k real-world PDF pages across eight domains. Our pipeline extracts and links evidence from text, tables, and figures, then generates 1,600 multimodal QA pairs spanning factual retrieval, comparison, summarization, and logical reasoning queries. To ensure reliability, 20% of QA pairs are validated by multiple annotators and expert adjudication. UniDoc-Bench supports apples-to-apples comparison across four paradigms: (1) text-only, (2) image-only, (3) multimodal text-image fusion, and (4) multimodal joint retrieval -- under a unified protocol with standardized candidate pools, prompts, and evaluation metrics. Our experiments show that multimodal text-image fusion RAG systems consistently outperform both unimodal and jointly multimodal embedding-based retrieval, indicating that neither text nor images alone are sufficient and that current multimodal embeddings remain inadequate. Beyond benchmarking, our analysis reveals when and how visual context complements textual evidence, uncovers systematic failure modes, and offers actionable guidance for developing more robust MM-RAG pipelines.
>
---
#### [new 081] A Warm-basis Method for Bridging Learning and Iteration: a Case Study in Fluorescence Molecular Tomography
- **分类: math.NA; cs.CV; cs.NA**

- **简介: 该论文属于医学成像任务，旨在解决荧光分子断层扫描（FMT）中深度重建精度低的问题。作者提出了一种新的温基迭代投影方法（WB-IPM），结合了学习与迭代方法的优点，提升了重建准确性和稳定性，并降低了训练成本。**

- **链接: [http://arxiv.org/pdf/2510.05926v1](http://arxiv.org/pdf/2510.05926v1)**

> **作者:** Ruchi Guo; Jiahua Jiang; Bangti Jin; Wuwei Ren; Jianru Zhang
>
> **摘要:** Fluorescence Molecular Tomography (FMT) is a widely used non-invasive optical imaging technology in biomedical research. It usually faces significant accuracy challenges in depth reconstruction, and conventional iterative methods struggle with poor $z$-resolution even with advanced regularization. Supervised learning approaches can improve recovery accuracy but rely on large, high-quality paired training dataset that is often impractical to acquire in practice. This naturally raises the question of how learning-based approaches can be effectively combined with iterative schemes to yield more accurate and stable algorithms. In this work, we present a novel warm-basis iterative projection method (WB-IPM) and establish its theoretical underpinnings. The method is able to achieve significantly more accurate reconstructions than the learning-based and iterative-based methods. In addition, it allows a weaker loss function depending solely on the directional component of the difference between ground truth and neural network output, thereby substantially reducing the training effort. These features are justified by our error analysis as well as simulated and real-data experiments.
>
---
#### [new 082] Controllable Audio-Visual Viewpoint Generation from 360° Spatial Information
- **分类: cs.MM; cs.AI; cs.CV**

- **简介: 该论文属于音频-视频生成任务，旨在解决从360°空间信息中生成可控视角内容的问题。现有方法难以精细控制视角生成，限制了沉浸式体验。论文提出一种扩散模型，引入全景显著图、带边界框的距离图和场景描述作为条件信号，实现对视角视频和音频的可控生成，提升了生成内容的空间感知与沉浸感。**

- **链接: [http://arxiv.org/pdf/2510.06060v1](http://arxiv.org/pdf/2510.06060v1)**

> **作者:** Christian Marinoni; Riccardo Fosco Gramaccioni; Eleonora Grassucci; Danilo Comminiello
>
> **摘要:** The generation of sounding videos has seen significant advancements with the advent of diffusion models. However, existing methods often lack the fine-grained control needed to generate viewpoint-specific content from larger, immersive 360-degree environments. This limitation restricts the creation of audio-visual experiences that are aware of off-camera events. To the best of our knowledge, this is the first work to introduce a framework for controllable audio-visual generation, addressing this unexplored gap. Specifically, we propose a diffusion model by introducing a set of powerful conditioning signals derived from the full 360-degree space: a panoramic saliency map to identify regions of interest, a bounding-box-aware signed distance map to define the target viewpoint, and a descriptive caption of the entire scene. By integrating these controls, our model generates spatially-aware viewpoint videos and audios that are coherently influenced by the broader, unseen environmental context, introducing a strong controllability that is essential for realistic and immersive audio-visual generation. We show audiovisual examples proving the effectiveness of our framework.
>
---
#### [new 083] NEO: No-Optimization Test-Time Adaptation through Latent Re-Centering
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于测试时自适应（TTA）任务，旨在解决模型在分布偏移数据上表现差、计算开销大及对超参数敏感的问题。作者提出NEO方法，通过潜在空间中心化实现无需优化、低计算成本的自适应，提升ViT模型在多个数据集上的分类准确率与校准性能。**

- **链接: [http://arxiv.org/pdf/2510.05635v1](http://arxiv.org/pdf/2510.05635v1)**

> **作者:** Alexander Murphy; Michal Danilowski; Soumyajit Chatterjee; Abhirup Ghosh
>
> **摘要:** Test-Time Adaptation (TTA) methods are often computationally expensive, require a large amount of data for effective adaptation, or are brittle to hyperparameters. Based on a theoretical foundation of the geometry of the latent space, we are able to significantly improve the alignment between source and distribution-shifted samples by re-centering target data embeddings at the origin. This insight motivates NEO -- a hyperparameter-free fully TTA method, that adds no significant compute compared to vanilla inference. NEO is able to improve the classification accuracy of ViT-Base on ImageNet-C from 55.6% to 59.2% after adapting on just one batch of 64 samples. When adapting on 512 samples NEO beats all 7 TTA methods we compare against on ImageNet-C, ImageNet-R and ImageNet-S and beats 6/7 on CIFAR-10-C, while using the least amount of compute. NEO performs well on model calibration metrics and additionally is able to adapt from 1 class to improve accuracy on 999 other classes in ImageNet-C. On Raspberry Pi and Jetson Orin Nano devices, NEO reduces inference time by 63% and memory usage by 9% compared to baselines. Our results based on 3 ViT architectures and 4 datasets show that NEO can be used efficiently and effectively for TTA.
>
---
#### [new 084] Overlap-aware segmentation for topological reconstruction of obscured objects
- **分类: hep-ex; astro-ph.IM; cs.CV**

- **简介: 该论文属于图像分割任务，旨在解决科学成像中重叠物体分离的难题。现有方法未优先处理重叠区域，导致模糊区域重建效果差。作者提出OASIS框架，通过加权损失函数在训练中重点关注重叠区域，提升分割与拓扑重建效果。在MIGDAL实验数据中验证，有效恢复被核反冲掩盖的电子轨迹信号。**

- **链接: [http://arxiv.org/pdf/2510.06194v1](http://arxiv.org/pdf/2510.06194v1)**

> **作者:** J. Schueler; H. M. Araújo; S. N. Balashov; J. E. Borg; C. Brew; F. M. Brunbauer; C. Cazzaniga; A. Cottle; D. Edgeman; C. D. Frost; F. Garcia; D. Hunt; M. Kastriotou; P. Knights; H. Kraus; A. Lindote; M. Lisowska; D. Loomba; E. Lopez Asamar; P. A. Majewski; T. Marley; C. McCabe; L. Millins; R. Nandakumar; T. Neep; F. Neves; K. Nikolopoulos; E. Oliveri; A. Roy; T. J. Sumner; E. Tilly; W. Thompson; M. A. Vogiatzi
>
> **摘要:** The separation of overlapping objects presents a significant challenge in scientific imaging. While deep learning segmentation-regression algorithms can predict pixel-wise intensities, they typically treat all regions equally rather than prioritizing overlap regions where attribution is most ambiguous. Recent advances in instance segmentation show that weighting regions of pixel overlap in training can improve segmentation boundary predictions in regions of overlap, but this idea has not yet been extended to segmentation regression. We address this with Overlap-Aware Segmentation of ImageS (OASIS): a new segmentation-regression framework with a weighted loss function designed to prioritize regions of object-overlap during training, enabling extraction of pixel intensities and topological features from heavily obscured objects. We demonstrate OASIS in the context of the MIGDAL experiment, which aims to directly image the Migdal effect--a rare process where electron emission is induced by nuclear scattering--in a low-pressure optical time projection chamber. This setting poses an extreme test case, as the target for reconstruction is a faint electron recoil track which is often heavily-buried within the orders-of-magnitude brighter nuclear recoil track. Compared to unweighted training, OASIS improves median intensity reconstruction errors from -32% to -14% for low-energy electron tracks (4-5 keV) and improves topological intersection-over-union scores from 0.828 to 0.855. These performance gains demonstrate OASIS's ability to recover obscured signals in overlap-dominated regions. The framework provides a generalizable methodology for scientific imaging where pixels represent physical quantities and overlap obscures features of interest. All code is openly available to facilitate cross-domain adoption.
>
---
#### [new 085] Neighborhood-Adaptive Generalized Linear Graph Embedding with Latent Pattern Mining
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于图嵌入任务，旨在解决现有方法需预定义邻域大小且线性投影依赖单一模式的问题。作者提出NGLGE模型，通过自适应邻域图学习和低秩重构，结合$\ell_{2,0}$范数约束，挖掘潜在模式并提升模型适应性。**

- **链接: [http://arxiv.org/pdf/2510.05719v1](http://arxiv.org/pdf/2510.05719v1)**

> **作者:** S. Peng; L. Hu; W. Zhang; B. Jie; Y. Luo
>
> **摘要:** Graph embedding has been widely applied in areas such as network analysis, social network mining, recommendation systems, and bioinformatics. However, current graph construction methods often require the prior definition of neighborhood size, limiting the effective revelation of potential structural correlations in the data. Additionally, graph embedding methods using linear projection heavily rely on a singular pattern mining approach, resulting in relative weaknesses in adapting to different scenarios. To address these challenges, we propose a novel model, Neighborhood-Adaptive Generalized Linear Graph Embedding (NGLGE), grounded in latent pattern mining. This model introduces an adaptive graph learning method tailored to the neighborhood, effectively revealing intrinsic data correlations. Simultaneously, leveraging a reconstructed low-rank representation and imposing $\ell_{2,0}$ norm constraint on the projection matrix allows for flexible exploration of additional pattern information. Besides, an efficient iterative solving algorithm is derived for the proposed model. Comparative evaluations on datasets from diverse scenarios demonstrate the superior performance of our model compared to state-of-the-art methods.
>
---
#### [new 086] FoleyGRAM: Video-to-Audio Generation with GRAM-Aligned Multimodal Encoders
- **分类: cs.SD; cs.CV; cs.LG; cs.MM; eess.AS**

- **简介: 该论文属于视频到音频生成任务，旨在解决生成音频与视频内容语义对齐的问题。作者提出FoleyGRAM，利用GRAM对齐多模态编码器，结合扩散模型生成音频，提升了生成音频的语义准确性和与视频的同步性。**

- **链接: [http://arxiv.org/pdf/2510.05829v1](http://arxiv.org/pdf/2510.05829v1)**

> **作者:** Riccardo Fosco Gramaccioni; Christian Marinoni; Eleonora Grassucci; Giordano Cicchetti; Aurelio Uncini; Danilo Comminiello
>
> **备注:** Acepted at IJCNN 2025
>
> **摘要:** In this work, we present FoleyGRAM, a novel approach to video-to-audio generation that emphasizes semantic conditioning through the use of aligned multimodal encoders. Building on prior advancements in video-to-audio generation, FoleyGRAM leverages the Gramian Representation Alignment Measure (GRAM) to align embeddings across video, text, and audio modalities, enabling precise semantic control over the audio generation process. The core of FoleyGRAM is a diffusion-based audio synthesis model conditioned on GRAM-aligned embeddings and waveform envelopes, ensuring both semantic richness and temporal alignment with the corresponding input video. We evaluate FoleyGRAM on the Greatest Hits dataset, a standard benchmark for video-to-audio models. Our experiments demonstrate that aligning multimodal encoders using GRAM enhances the system's ability to semantically align generated audio with video content, advancing the state of the art in video-to-audio synthesis.
>
---
#### [new 087] DeLTa: Demonstration and Language-Guided Novel Transparent Object Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人透明物体操作任务，旨在解决现有方法在长视野、精确操作透明物体上的局限性。作者提出了DeLTa框架，结合深度估计、6D姿态估计与视觉语言规划，实现基于自然指令的单次示范泛化，提升对新透明物体的操作能力。**

- **链接: [http://arxiv.org/pdf/2510.05662v1](http://arxiv.org/pdf/2510.05662v1)**

> **作者:** Taeyeop Lee; Gyuree Kang; Bowen Wen; Youngho Kim; Seunghyeok Back; In So Kweon; David Hyunchul Shim; Kuk-Jin Yoon
>
> **备注:** Project page: https://sites.google.com/view/DeLTa25/
>
> **摘要:** Despite the prevalence of transparent object interactions in human everyday life, transparent robotic manipulation research remains limited to short-horizon tasks and basic grasping capabilities.Although some methods have partially addressed these issues, most of them have limitations in generalizability to novel objects and are insufficient for precise long-horizon robot manipulation. To address this limitation, we propose DeLTa (Demonstration and Language-Guided Novel Transparent Object Manipulation), a novel framework that integrates depth estimation, 6D pose estimation, and vision-language planning for precise long-horizon manipulation of transparent objects guided by natural task instructions. A key advantage of our method is its single-demonstration approach, which generalizes 6D trajectories to novel transparent objects without requiring category-level priors or additional training. Additionally, we present a task planner that refines the VLM-generated plan to account for the constraints of a single-arm, eye-in-hand robot for long-horizon object manipulation tasks. Through comprehensive evaluation, we demonstrate that our method significantly outperforms existing transparent object manipulation approaches, particularly in long-horizon scenarios requiring precise manipulation capabilities. Project page: https://sites.google.com/view/DeLTa25/
>
---
#### [new 088] SafeGuider: Robust and Practical Content Safety Control for Text-to-Image Models
- **分类: cs.CR; cs.AI; cs.CV; I.2**

- **简介: 该论文属于文本到图像生成模型的安全控制任务，旨在解决对抗性提示绕过安全措施的问题。作者提出SafeGuider框架，通过识别嵌入空间中的[EOS]标记分布差异，并结合安全感知的特征擦除算法，在保持生成质量的同时有效防御攻击。**

- **链接: [http://arxiv.org/pdf/2510.05173v1](http://arxiv.org/pdf/2510.05173v1)**

> **作者:** Peigui Qi; Kunsheng Tang; Wenbo Zhou; Weiming Zhang; Nenghai Yu; Tianwei Zhang; Qing Guo; Jie Zhang
>
> **备注:** Accepted by ACM CCS 2025
>
> **摘要:** Text-to-image models have shown remarkable capabilities in generating high-quality images from natural language descriptions. However, these models are highly vulnerable to adversarial prompts, which can bypass safety measures and produce harmful content. Despite various defensive strategies, achieving robustness against attacks while maintaining practical utility in real-world applications remains a significant challenge. To address this issue, we first conduct an empirical study of the text encoder in the Stable Diffusion (SD) model, which is a widely used and representative text-to-image model. Our findings reveal that the [EOS] token acts as a semantic aggregator, exhibiting distinct distributional patterns between benign and adversarial prompts in its embedding space. Building on this insight, we introduce \textbf{SafeGuider}, a two-step framework designed for robust safety control without compromising generation quality. SafeGuider combines an embedding-level recognition model with a safety-aware feature erasure beam search algorithm. This integration enables the framework to maintain high-quality image generation for benign prompts while ensuring robust defense against both in-domain and out-of-domain attacks. SafeGuider demonstrates exceptional effectiveness in minimizing attack success rates, achieving a maximum rate of only 5.48\% across various attack scenarios. Moreover, instead of refusing to generate or producing black images for unsafe prompts, \textbf{SafeGuider} generates safe and meaningful images, enhancing its practical utility. In addition, SafeGuider is not limited to the SD model and can be effectively applied to other text-to-image models, such as the Flux model, demonstrating its versatility and adaptability across different architectures. We hope that SafeGuider can shed some light on the practical deployment of secure text-to-image systems.
>
---
#### [new 089] D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于视觉-动作预训练任务，旨在解决实体AI缺乏大规模数据的问题。论文提出了D2E框架，利用桌面环境（如游戏）进行预训练，并通过标准化工具、事件预测模型和迁移方法，将桌面数据迁移到实体AI任务中，取得了良好效果。**

- **链接: [http://arxiv.org/pdf/2510.05684v1](http://arxiv.org/pdf/2510.05684v1)**

> **作者:** Suwhan Choi; Jaeyoon Jung; Haebin Seong; Minchan Kim; Minyeong Kim; Yongjun Cho; Yoonshik Kim; Yubeen Park; Youngjae Yu; Yunsung Lee
>
> **摘要:** Large language models leverage internet-scale text data, yet embodied AI remains constrained by the prohibitive costs of physical trajectory collection. Desktop environments -- particularly gaming -- offer a compelling alternative: they provide rich sensorimotor interactions at scale while maintaining the structured observation-action coupling essential for embodied learning. We present D2E (Desktop to Embodied AI), a framework that demonstrates desktop interactions can serve as an effective pretraining substrate for robotics embodied AI tasks. Unlike prior work that remained domain-specific (e.g., VPT for Minecraft) or kept data proprietary (e.g., SIMA), D2E establishes a complete pipeline from scalable desktop data collection to verified transfer in embodied domains. Our framework comprises three components: (1) the OWA Toolkit that unifies diverse desktop interactions into a standardized format with 152x compression, (2) the Generalist-IDM that achieves strong zero-shot generalization across unseen games through timestamp-based event prediction, enabling internet-scale pseudo-labeling, and (3) VAPT that transfers desktop-pretrained representations to physical manipulation and navigation. Using 1.3K+ hours of data (259 hours of human demonstrations, and 1K+ hours of pseudo-labeled gameplay), we achieve a total of 96.6% success rate on LIBERO manipulation and 83.3% on CANVAS navigation benchmarks. This validates that sensorimotor primitives in digital interactions exhibit sufficient invariance to transfer meaningfully to physical embodied tasks, establishing desktop pretraining as a practical paradigm for robotics. We will make all our work public, including the OWA toolkit, datasets of human-collected and pseudo-labeled, and VAPT-trained models available at https://worv-ai.github.io/d2e/
>
---
#### [new 090] Gaussian Embeddings: How JEPAs Secretly Learn Your Data Density
- **分类: cs.LG; cs.AI; cs.CV; stat.ML**

- **简介: 该论文属于自监督学习任务，旨在解决如何从JEPA类模型中提取数据密度估计。论文发现JEPA中的反坍塌项可估计数据密度，提出JEPA-SCORE方法，通过模型雅可比矩阵计算样本概率，适用于数据筛选、异常检测等任务，并在多种数据集和模型上验证有效性。**

- **链接: [http://arxiv.org/pdf/2510.05949v1](http://arxiv.org/pdf/2510.05949v1)**

> **作者:** Randall Balestriero; Nicolas Ballas; Mike Rabbat; Yann LeCun
>
> **摘要:** Joint Embedding Predictive Architectures (JEPAs) learn representations able to solve numerous downstream tasks out-of-the-box. JEPAs combine two objectives: (i) a latent-space prediction term, i.e., the representation of a slightly perturbed sample must be predictable from the original sample's representation, and (ii) an anti-collapse term, i.e., not all samples should have the same representation. While (ii) is often considered as an obvious remedy to representation collapse, we uncover that JEPAs' anti-collapse term does much more--it provably estimates the data density. In short, any successfully trained JEPA can be used to get sample probabilities, e.g., for data curation, outlier detection, or simply for density estimation. Our theoretical finding is agnostic of the dataset and architecture used--in any case one can compute the learned probabilities of sample $x$ efficiently and in closed-form using the model's Jacobian matrix at $x$. Our findings are empirically validated across datasets (synthetic, controlled, and Imagenet) and across different Self Supervised Learning methods falling under the JEPA family (I-JEPA and DINOv2) and on multimodal models, such as MetaCLIP. We denote the method extracting the JEPA learned density as {\bf JEPA-SCORE}.
>
---
#### [new 091] StereoSync: Spatially-Aware Stereo Audio Generation from Video
- **分类: cs.SD; cs.CV; cs.LG; cs.MM; eess.AS**

- **简介: 该论文属于视频到音频生成任务，旨在解决视频与音频在时间和空间上的对齐问题。论文提出StereoSync模型，利用深度图和边界框提取空间线索，通过扩散模型生成时空同步的立体音频，提升音频生成的沉浸感与真实感。**

- **链接: [http://arxiv.org/pdf/2510.05828v1](http://arxiv.org/pdf/2510.05828v1)**

> **作者:** Christian Marinoni; Riccardo Fosco Gramaccioni; Kazuki Shimada; Takashi Shibuya; Yuki Mitsufuji; Danilo Comminiello
>
> **备注:** Accepted at IJCNN 2025
>
> **摘要:** Although audio generation has been widely studied over recent years, video-aligned audio generation still remains a relatively unexplored frontier. To address this gap, we introduce StereoSync, a novel and efficient model designed to generate audio that is both temporally synchronized with a reference video and spatially aligned with its visual context. Moreover, StereoSync also achieves efficiency by leveraging pretrained foundation models, reducing the need for extensive training while maintaining high-quality synthesis. Unlike existing methods that primarily focus on temporal synchronization, StereoSync introduces a significant advancement by incorporating spatial awareness into video-aligned audio generation. Indeed, given an input video, our approach extracts spatial cues from depth maps and bounding boxes, using them as cross-attention conditioning in a diffusion-based audio generation model. Such an approach allows StereoSync to go beyond simple synchronization, producing stereo audio that dynamically adapts to the spatial structure and movement of a video scene. We evaluate StereoSync on Walking The Maps, a curated dataset comprising videos from video games that feature animated characters walking through diverse environments. Experimental results demonstrate the ability of StereoSync to achieve both temporal and spatial alignment, advancing the state of the art in video-to-audio generation and resulting in a significantly more immersive and realistic audio experience.
>
---
#### [new 092] Improving Clinical Dataset Condensation with Mode Connectivity-based Trajectory Surrogates
- **分类: cs.LG; cs.CV; cs.DB**

- **简介: 该论文属于数据集压缩任务，旨在解决现有方法中使用完整SGD轨迹导致的噪声大、存储高、收敛慢等问题。工作提出用二次贝塞尔曲线替代完整轨迹，提供平滑监督信号，提升压缩数据集效果，实验证明在五个临床数据集上表现更优。**

- **链接: [http://arxiv.org/pdf/2510.05805v1](http://arxiv.org/pdf/2510.05805v1)**

> **作者:** Pafue Christy Nganjimi; Andrew Soltan; Danielle Belgrave; Lei Clifton; David A. Clifton; Anshul Thakur
>
> **备注:** 20 pages, 4 figures, Submitted to AISTATS 2026
>
> **摘要:** Dataset condensation (DC) enables the creation of compact, privacy-preserving synthetic datasets that can match the utility of real patient records, supporting democratised access to highly regulated clinical data for developing downstream clinical models. State-of-the-art DC methods supervise synthetic data by aligning the training dynamics of models trained on real and those trained on synthetic data, typically using full stochastic gradient descent (SGD) trajectories as alignment targets; however, these trajectories are often noisy, high-curvature, and storage-intensive, leading to unstable gradients, slow convergence, and substantial memory overhead. We address these limitations by replacing full SGD trajectories with smooth, low-loss parametric surrogates, specifically quadratic B\'ezier curves that connect the initial and final model states from real training trajectories. These mode-connected paths provide noise-free, low-curvature supervision signals that stabilise gradients, accelerate convergence, and eliminate the need for dense trajectory storage. We theoretically justify B\'ezier-mode connections as effective surrogates for SGD paths and empirically show that the proposed method outperforms state-of-the-art condensation approaches across five clinical datasets, yielding condensed datasets that enable clinically effective model development.
>
---
#### [new 093] nnSAM2: nnUNet-Enhanced One-Prompt SAM2 for Few-shot Multi-Modality Segmentation and Composition Analysis of Lumbar Paraspinal Muscles
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决仅用单个标注切片进行多模态腰椎旁肌分割的问题。作者提出了nnSAM2方法，结合SAM2与nnU-Net，通过伪标签生成与模型迭代优化，实现了高精度分割与成分分析，验证了其与专家标注的统计等效性。**

- **链接: [http://arxiv.org/pdf/2510.05555v1](http://arxiv.org/pdf/2510.05555v1)**

> **作者:** Zhongyi Zhang; Julie A. Hides; Enrico De Martino; Abdul Joseph Fofanah; Gervase Tuxworth
>
> **摘要:** Purpose: To develop and validate No-New SAM2 (nnsam2) for few-shot segmentation of lumbar paraspinal muscles using only a single annotated slice per dataset, and to assess its statistical comparability with expert measurements across multi-sequence MRI and multi-protocol CT. Methods: We retrospectively analyzed 1,219 scans (19,439 slices) from 762 participants across six datasets. Six slices (one per dataset) served as labeled examples, while the remaining 19,433 slices were used for testing. In this minimal-supervision setting, nnsam2 used single-slice SAM2 prompts to generate pseudo-labels, which were pooled across datasets and refined through three sequential, independent nnU-Net models. Segmentation performance was evaluated using the Dice similarity coefficient (DSC), and automated measurements-including muscle volume, fat ratio, and CT attenuation-were assessed with two one-sided tests (TOST) and intraclass correlation coefficients (ICC). Results: nnsam2 outperformed vanilla SAM2, its medical variants, TotalSegmentator, and the leading few-shot method, achieving DSCs of 0.94-0.96 on MR images and 0.92-0.93 on CT. Automated and expert measurements were statistically equivalent for muscle volume (MRI/CT), CT attenuation, and Dixon fat ratio (TOST, P < 0.05), with consistently high ICCs (0.86-1.00). Conclusion: We developed nnsam2, a state-of-the-art few-shot framework for multi-modality LPM segmentation, producing muscle volume (MRI/CT), attenuation (CT), and fat ratio (Dixon MRI) measurements that were statistically comparable to expert references. Validated across multimodal, multicenter, and multinational cohorts, and released with open code and data, nnsam2 demonstrated high annotation efficiency, robust generalizability, and reproducibility.
>
---
#### [new 094] Discretized Quadratic Integrate-and-Fire Neuron Model for Deep Spiking Neural Networks
- **分类: cs.LG; cs.CV; I.2**

- **简介: 该论文属于深度脉冲神经网络（SNN）研究任务，旨在解决传统LIF神经元模型表达能力受限的问题。作者提出了一种适用于深度SNN的离散化QIF神经元模型，并通过分析梯度窗口提升训练稳定性。实验表明，该模型在多个数据集上优于现有LIF方法，兼具非线性动态和可扩展性。**

- **链接: [http://arxiv.org/pdf/2510.05168v1](http://arxiv.org/pdf/2510.05168v1)**

> **作者:** Eric Jahns; Davi Moreno; Milan Stojkov; Michel A. Kinsy
>
> **备注:** 18 pages, 2 figures
>
> **摘要:** Spiking Neural Networks (SNNs) have emerged as energy-efficient alternatives to traditional artificial neural networks, leveraging asynchronous and biologically inspired neuron dynamics. Among existing neuron models, the Leaky Integrate-and-Fire (LIF) neuron has become widely adopted in deep SNNs due to its simplicity and computational efficiency. However, this efficiency comes at the expense of expressiveness, as LIF dynamics are constrained to linear decay at each timestep. In contrast, more complex models, such as the Quadratic Integrate-and-Fire (QIF) neuron, exhibit richer, nonlinear dynamics but have seen limited adoption due to their training instability. On that note, we propose the first discretization of the QIF neuron model tailored for high-performance deep spiking neural networks and provide an in-depth analysis of its dynamics. To ensure training stability, we derive an analytical formulation for surrogate gradient windows directly from our discretizations' parameter set, minimizing gradient mismatch. We evaluate our method on CIFAR-10, CIFAR-100, ImageNet, and CIFAR-10 DVS, demonstrating its ability to outperform state-of-the-art LIF-based methods. These results establish our discretization of the QIF neuron as a compelling alternative to LIF neurons for deep SNNs, combining richer dynamics with practical scalability.
>
---
#### [new 095] Smartphone-based iris recognition through high-quality visible-spectrum iris image capture.V2
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属于生物特征识别任务，旨在解决智能手机可见光虹膜识别中的准确性问题。通过构建符合ISO标准的高质量虹膜图像采集流程、开发轻量级分割网络LightIrisNet和适配可见光域的Transformer匹配模型IrisFormer，实验证明在智能手机上可实现高精度虹膜识别。**

- **链接: [http://arxiv.org/pdf/2510.06170v1](http://arxiv.org/pdf/2510.06170v1)**

> **作者:** Naveenkumar G Venkataswamy; Yu Liu; Soumyabrata Dey; Stephanie Schuckers; Masudul H Imtiaz
>
> **备注:** We build upon our earlier work, arXiv:2412.13063
>
> **摘要:** Smartphone-based iris recognition in the visible spectrum (VIS) remains difficult due to illumination variability, pigmentation differences, and the absence of standardized capture controls. This work presents a compact end-to-end pipeline that enforces ISO/IEC 29794-6 quality compliance at acquisition and demonstrates that accurate VIS iris recognition is feasible on commodity devices. Using a custom Android application performing real-time framing, sharpness evaluation, and feedback, we introduce the CUVIRIS dataset of 752 compliant images from 47 subjects. A lightweight MobileNetV3-based multi-task segmentation network (LightIrisNet) is developed for efficient on-device processing, and a transformer matcher (IrisFormer) is adapted to the VIS domain. Under a standardized protocol and comparative benchmarking against prior CNN baselines, OSIRIS attains a TAR of 97.9% at FAR=0.01 (EER=0.76%), while IrisFormer, trained only on UBIRIS.v2, achieves an EER of 0.057% on CUVIRIS. The acquisition app, trained models, and a public subset of the dataset are released to support reproducibility. These results confirm that standardized capture and VIS-adapted lightweight models enable accurate and practical iris recognition on smartphones.
>
---
#### [new 096] Towards Robust and Realible Multimodal Fake News Detection with Incomplete Modality
- **分类: cs.MM; cs.CV**

- **简介: 该论文属于多模态虚假新闻检测任务，旨在解决实际应用中多模态信息缺失导致检测性能下降的问题。作者提出MMLNet方法，通过多专家协作推理、不完整模态适配器和模态缺失学习三个步骤，提升模型在模态不完整情况下的鲁棒性和准确性。**

- **链接: [http://arxiv.org/pdf/2510.05839v1](http://arxiv.org/pdf/2510.05839v1)**

> **作者:** Hengyang Zhou; Yiwei Wei; Jian Yang; Zhenyu Zhang
>
> **摘要:** Multimodal fake news detection (MFND) has become an urgent task with the emergence of huge multimodal fake content on social media platforms. Previous studies mainly focus on complex feature extraction and fusion to learn discriminative information from multimodal content. However, in real-world applications, multimedia news may naturally lose some information during dissemination, resulting in modality incompleteness, which is detrimental to the generalization and robustness of existing models. To this end, we propose a novel generic and robust multimodal fusion strategy, termed Multi-expert Modality-incomplete Learning Network (MMLNet), which is simple yet effective. It consists of three key steps: (1) Multi-Expert Collaborative Reasoning to compensate for missing modalities by dynamically leveraging complementary information through multiple experts. (2) Incomplete Modality Adapters compensates for the missing information by leveraging the new feature distribution. (3) Modality Missing Learning leveraging an label-aware adaptive weighting strategy to learn a robust representation with contrastive learning. We evaluate MMLNet on three real-world benchmarks across two languages, demonstrating superior performance compared to state-of-the-art methods while maintaining relative simplicity. By ensuring the accuracy of fake news detection in incomplete modality scenarios caused by information propagation, MMLNet effectively curbs the spread of malicious misinformation. Code is publicly available at https://github.com/zhyhome/MMLNet.
>
---
#### [new 097] The Safety Challenge of World Models for Embodied AI Agents: A Review
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文综述了具身智能中世界模型的安全挑战，重点分析自动驾驶和机器人领域。任务是评估模型在场景与控制生成中的安全性，识别常见错误并量化分析，以提升预测安全性。**

- **链接: [http://arxiv.org/pdf/2510.05865v1](http://arxiv.org/pdf/2510.05865v1)**

> **作者:** Lorenzo Baraldi; Zifan Zeng; Chongzhe Zhang; Aradhana Nayak; Hongbo Zhu; Feng Liu; Qunli Zhang; Peng Wang; Shiming Liu; Zheng Hu; Angelo Cangelosi; Lorenzo Baraldi
>
> **摘要:** The rapid progress in embodied artificial intelligence has highlighted the necessity for more advanced and integrated models that can perceive, interpret, and predict environmental dynamics. In this context, World Models (WMs) have been introduced to provide embodied agents with the abilities to anticipate future environmental states and fill in knowledge gaps, thereby enhancing agents' ability to plan and execute actions. However, when dealing with embodied agents it is fundamental to ensure that predictions are safe for both the agent and the environment. In this article, we conduct a comprehensive literature review of World Models in the domains of autonomous driving and robotics, with a specific focus on the safety implications of scene and control generation tasks. Our review is complemented by an empirical analysis, wherein we collect and examine predictions from state-of-the-art models, identify and categorize common faults (herein referred to as pathologies), and provide a quantitative evaluation of the results.
>
---
## 更新

#### [replaced 001] RimSet: Quantitatively Identifying and Characterizing Chronic Active Multiple Sclerosis Lesion on Quantitative Susceptibility Maps
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2312.16835v2](http://arxiv.org/pdf/2312.16835v2)**

> **作者:** Jinwei Zhang; Thanh D. Nguyen; Renjiu Hu; Susan A. Gauthier; Yi Wang; Hang Zhang
>
> **备注:** 13 pages, 7 figures, 4 tables
>
> **摘要:** Background: Rim+ lesions in multiple sclerosis (MS), detectable via Quantitative Susceptibility Mapping (QSM), correlate with increased disability. Existing literature lacks quantitative analysis of these lesions. We introduce RimSet for quantitative identification and characterization of rim+ lesions on QSM. Methods: RimSet combines RimSeg, an unsupervised segmentation method using level-set methodology, and radiomic measurements with Local Binary Pattern texture descriptors. We validated RimSet using simulated QSM images and an in vivo dataset of 172 MS subjects with 177 rim+ and 3986 rim-lesions. Results: RimSeg achieved a 78.7% Dice score against the ground truth, with challenges in partial rim lesions. RimSet detected rim+ lesions with a partial ROC AUC of 0.808 and PR AUC of 0.737, surpassing existing methods. QSMRim-Net showed the lowest mean square error (0.85) and high correlation (0.91; 95% CI: 0.88, 0.93) with expert annotations at the subject level.
>
---
#### [replaced 002] Background Semantics Matter: Cross-Task Feature Exchange Network for Clustered Infrared Small Target Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2407.20078v3](http://arxiv.org/pdf/2407.20078v3)**

> **作者:** Mengxuan Xiao; Yinfei Zhu; Yiming Zhu; Boyang Li; Feifei Zhang; Huan Wang; Meng Cai; Yimian Dai
>
> **摘要:** Infrared small target detection presents significant challenges due to the limited intrinsic features of the target and the overwhelming presence of visually similar background distractors. We contend that background semantics are critical for distinguishing between objects that appear visually similar in this context. To address this challenge, we propose a task, clustered infrared small target detection, and introduce DenseSIRST, a benchmark dataset that provides per-pixel semantic annotations for background regions. This dataset facilitates the shift from sparse to dense target detection. This dataset facilitates the shift from sparse to dense target detection. Building on this resource, we propose the Background-Aware Feature Exchange Network (BAFE-Net), a multi-task architecture that jointly tackles target detection and background semantic segmentation. BAFE-Net incorporates a dynamic cross-task feature hard-exchange mechanism, enabling the effective exchange of target and background semantics between the two tasks. Comprehensive experiments demonstrate that BAFE-Net significantly enhances target detection accuracy while mitigating false alarms. The DenseSIRST dataset, along with the code and trained models, is publicly available at https://github.com/GrokCV/BAFE-Net.
>
---
#### [replaced 003] LaB-RAG: Label Boosted Retrieval Augmented Generation for Radiology Report Generation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.16523v2](http://arxiv.org/pdf/2411.16523v2)**

> **作者:** Steven Song; Anirudh Subramanyam; Irene Madejski; Robert L. Grossman
>
> **摘要:** In the current paradigm of image captioning, deep learning models are trained to generate text from image embeddings of latent features. We challenge the assumption that fine-tuning of large, bespoke models is required to improve model generation accuracy. Here we propose Label Boosted Retrieval Augmented Generation (LaB-RAG), a small-model-based approach to image captioning that leverages image descriptors in the form of categorical labels to boost standard retrieval augmented generation (RAG) with pretrained large language models (LLMs). We study our method in the context of radiology report generation (RRG) over MIMIC-CXR and CheXpert Plus. We argue that simple classification models combined with zero-shot embeddings can effectively transform X-rays into text-space as radiology-specific labels. In combination with standard RAG, we show that these derived text labels can be used with general-domain LLMs to generate radiology reports. Without ever training our generative language model or image embedding models specifically for the task, and without ever directly "showing" the LLM an X-ray, we demonstrate that LaB-RAG achieves better results across natural language and radiology language metrics compared with other retrieval-based RRG methods, while attaining competitive results compared to other fine-tuned vision-language RRG models. We further conduct extensive ablation experiments to better understand the components of LaB-RAG. Our results suggest broader compatibility and synergy with fine-tuned methods to further enhance RRG performance.
>
---
#### [replaced 004] Video-in-the-Loop: Span-Grounded Long Video QA with Interleaved Reasoning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.04022v2](http://arxiv.org/pdf/2510.04022v2)**

> **作者:** Chendong Wang; Donglin Bai; Yifan Yang; Xiao Jin; Anlan Zhang; Rui Wang; Shiqi Jiang; Yuqing Yang; Hao Wu; Qi Dai; Chong Luo; Ting Cao; Lili Qiu; Suman Banerjee
>
> **摘要:** We present \emph{Video-in-the-Loop} (ViTL), a two-stage long-video QA framework that preserves a fixed token budget by first \emph{localizing} question-relevant interval(s) with a low-fps skim and then \emph{answering} via span-aware reallocation of visual tokens at higher effective frame rate, emitting an interleaved output with both spans and the final option for direct attribution. We also introduce \dataname{}, which converts description based event graphs into \emph{span-grounded} multiple-choice QA by pairing each question with \emph{ground-truth} time span(s) and related reasoning. ViTL is trained end-to-end with an interleaved group-relative objective that couples temporal IoU for localization with answer correctness, allowing credit to flow from answers back to spans without increasing compute. Under fixed token budgets, ViTL attains up to 8.6% with 50% less frame input on long-video QA and temporal grounding (e.g., Charades-STA, ActivityNet-Captions) and ablations show that span-aware token reallocation consistently surpasses uniform sampling. Together, \dataname{} and ViTL provide an interpretable, compute-efficient recipe for scalable long-video QA.
>
---
#### [replaced 005] GeoRemover: Removing Objects and Their Causal Visual Artifacts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.18538v2](http://arxiv.org/pdf/2509.18538v2)**

> **作者:** Zixin Zhu; Haoxiang Li; Xuelu Feng; He Wu; Chunming Qiao; Junsong Yuan
>
> **备注:** Accepted as Spotlight at NeurIPS 2025
>
> **摘要:** Towards intelligent image editing, object removal should eliminate both the target object and its causal visual artifacts, such as shadows and reflections. However, existing image appearance-based methods either follow strictly mask-aligned training and fail to remove these causal effects which are not explicitly masked, or adopt loosely mask-aligned strategies that lack controllability and may unintentionally over-erase other objects. We identify that these limitations stem from ignoring the causal relationship between an object's geometry presence and its visual effects. To address this limitation, we propose a geometry-aware two-stage framework that decouples object removal into (1) geometry removal and (2) appearance rendering. In the first stage, we remove the object directly from the geometry (e.g., depth) using strictly mask-aligned supervision, enabling structure-aware editing with strong geometric constraints. In the second stage, we render a photorealistic RGB image conditioned on the updated geometry, where causal visual effects are considered implicitly as a result of the modified 3D geometry. To guide learning in the geometry removal stage, we introduce a preference-driven objective based on positive and negative sample pairs, encouraging the model to remove objects as well as their causal visual artifacts while avoiding new structural insertions. Extensive experiments demonstrate that our method achieves state-of-the-art performance in removing both objects and their associated artifacts on two popular benchmarks. The code is available at https://github.com/buxiangzhiren/GeoRemover.
>
---
#### [replaced 006] SBP-YOLO:A Lightweight Real-Time Model for Detecting Speed Bumps and Potholes toward Intelligent Vehicle Suspension Systems
- **分类: cs.CV; cs.AI; 68T45; I.4.8; C.3**

- **链接: [http://arxiv.org/pdf/2508.01339v4](http://arxiv.org/pdf/2508.01339v4)**

> **作者:** Chuanqi Liang; Jie Fu; Miao Yu; Lei Luo
>
> **备注:** 14pages,11figures
>
> **摘要:** Speed bumps and potholes are the most common road anomalies, significantly affecting ride comfort and vehicle stability. Preview-based suspension control mitigates their impact by detecting such irregularities in advance and adjusting suspension parameters proactively. Accurate and real-time detection is essential, but embedded deployment is constrained by limited computational resources and the small size of targets in input images.To address these challenges, this paper proposes SBP-YOLO, an efficient detection framework for speed bumps and potholes in embedded systems. Built upon YOLOv11n, it integrates GhostConv and VoVGSCSPC modules in the backbone and neck to reduce computation while enhancing multi-scale semantic features. A P2-level branch improves small-object detection, and a lightweight and efficient detection head (LEDH) maintains accuracy with minimal overhead. A hybrid training strategy further enhances robustness under varying road and environmental conditions, combining NWD loss, BCKD knowledge distillation, and Albumentations-based augmentation. Experiments show that SBP-YOLO achieves 87.0% mAP, outperforming the YOLOv11n baseline by 5.8%. After TensorRT FP16 quantization, it runs at 139.5 FPS on Jetson AGX Xavier, yielding a 12.4% speedup over the P2-enhanced YOLOv11. These results demonstrate the framework's suitability for fast, low-latency road condition perception in embedded suspension control systems.
>
---
#### [replaced 007] Can Video Large Multimodal Models Think Like Doubters-or Double-Down: A Study on Defeasible Video Entailment
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.22385v2](http://arxiv.org/pdf/2506.22385v2)**

> **作者:** Yue Zhang; Jilei Sun; Yunhui Guo; Vibhav Gogate
>
> **摘要:** Video Large Multimodal Models (VLMMs) have made impressive strides in understanding video content, but they often struggle with abstract and adaptive reasoning-the ability to revise their interpretations when new information emerges. In reality, conclusions are rarely set in stone; additional context can strengthen or weaken an initial inference. To address this, we introduce Defeasible Video Entailment (DVidE), a new task that challenges models to think like doubters, constantly updating their reasoning based on evolving evidence. In DVidE, given a video premise and a textual hypothesis, models must determine whether a new update strengthens or weakens the hypothesis (classification version) or generate a coherent update that modifies the entailment relationship (generation version). For solving the classification task, we propose the Chain of Counterfactual Thought framework, utilizing counterfactual reasoning, ASR-enhanced video content, and rationale refinement to reduce inference bias. For the generation task, we develop a framework that combines ASR output with a Large Language Model (LLM) to produce coherent, contextually relevant updates aligned with the intended strengthener or weakener goals. Additionally, we introduce a novel benchmark dataset, with strengthener/weakener annotations and an LLM-based evaluation metric specifically designed for assessing generative performance. Experimental results demonstrate significant improvements, highlighting our proposed method in enhancing dynamic reasoning capabilities of VLMMs.
>
---
#### [replaced 008] Electromagnetic Inverse Scattering from a Single Transmitter
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2506.21349v5](http://arxiv.org/pdf/2506.21349v5)**

> **作者:** Yizhe Cheng; Chunxun Tian; Haoru Wang; Wentao Zhu; Xiaoxuan Ma; Yizhou Wang
>
> **摘要:** Solving Electromagnetic Inverse Scattering Problems (EISP) is fundamental in applications such as medical imaging, where the goal is to reconstruct the relative permittivity from scattered electromagnetic field. This inverse process is inherently ill-posed and highly nonlinear, making it particularly challenging, especially under sparse transmitter setups, e.g., with only one transmitter. A recent machine learning-based approach, Img-Interiors, shows promising results by leveraging continuous implicit functions. However, it requires time-consuming case-specific optimization and fails under sparse transmitter setups. To address these limitations, we revisit EISP from a data-driven perspective. The scarcity of transmitters leads to an insufficient amount of measured data, which fails to capture adequate physical information for stable inversion. Built on this insight, we propose a fully end-to-end and data-driven framework that predicts the relative permittivity of scatterers from measured fields, leveraging data distribution priors to compensate for the lack of physical information. This design enables data-driven training and feed-forward prediction of relative permittivity while maintaining strong robustness to transmitter sparsity. Extensive experiments show that our method outperforms state-of-the-art approaches in reconstruction accuracy and robustness. Notably, it achieves high-quality results even with a single transmitter, a setting where previous methods consistently fail. This work offers a fundamentally new perspective on electromagnetic inverse scattering and represents a major step toward cost-effective practical solutions for electromagnetic imaging.
>
---
#### [replaced 009] Deep Reinforcement Learning for Urban Air Quality Management: Multi-Objective Optimization of Pollution Mitigation Booth Placement in Metropolitan Environments
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.00668v2](http://arxiv.org/pdf/2505.00668v2)**

> **作者:** Kirtan Rajesh; Suvidha Rupesh Kumar
>
> **备注:** This is the preprint version of the article published in IEEE Access vol. 13, pp. 146503--146526, 2025, doi:10.1109/ACCESS.2025.3599541. Please cite the published version
>
> **摘要:** This is the preprint version of the article published in IEEE Access vol. 13, pp. 146503--146526, 2025, doi:10.1109/ACCESS.2025.3599541. Please cite the published version. Urban air pollution remains a pressing global concern, particularly in densely populated and traffic-intensive metropolitan areas like Delhi, where exposure to harmful pollutants severely impacts public health. Delhi, being one of the most polluted cities globally, experiences chronic air quality issues due to vehicular emissions, industrial activities, and construction dust, which exacerbate its already fragile atmospheric conditions. Traditional pollution mitigation strategies, such as static air purifying installations, often fail to maximize their impact due to suboptimal placement and limited adaptability to dynamic urban environments. This study presents a novel deep reinforcement learning (DRL) framework to optimize the placement of air purification booths to improve the air quality index (AQI) in the city of Delhi. We employ Proximal Policy Optimization (PPO), a state-of-the-art reinforcement learning algorithm, to iteratively learn and identify high-impact locations based on multiple spatial and environmental factors, including population density, traffic patterns, industrial influence, and green space constraints. Our approach is benchmarked against conventional placement strategies, including random and greedy AQI-based methods, using multi-dimensional performance evaluation metrics such as AQI improvement, spatial coverage, population and traffic impact, and spatial entropy.
>
---
#### [replaced 010] DiffCom: Decoupled Sparse Priors Guided Diffusion Compression for Point Clouds
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2411.13860v3](http://arxiv.org/pdf/2411.13860v3)**

> **作者:** Xiaoge Zhang; Zijie Wu; Mehwish Nasim; Mingtao Feng; Saeed Anwar; Ajmal Mian
>
> **摘要:** Lossy compression relies on an autoencoder to transform a point cloud into latent points for storage, leaving the inherent redundancy of latent representations unexplored. To reduce redundancy in latent points, we propose a diffusion-based framework guided by sparse priors that achieves high reconstruction quality, especially at low bitrates. Our approach features an efficient dual-density data flow that relaxes size constraints on latent points. It hybridizes a probabilistic conditional diffusion model to encapsulate essential details for reconstruction within sparse priors, which are decoupled hierarchically into intra- and inter-point priors. Specifically, our DiffCom encodes the original point cloud into latent points and decoupled sparse priors through separate encoders. To dynamically attend to geometric and semantic cues from the priors at each encoding and decoding layer, we employ an attention-guided latent denoiser conditioned on the decoupled priors. Additionally, we integrate the local distribution into the arithmetic encoder and decoder to enhance local context modeling of the sparse points. The original point cloud is reconstructed through a point decoder. Compared to state-of-the-art methods, our approach achieves a superior rate-distortion trade-off, as evidenced by extensive evaluations on the ShapeNet dataset and standard test datasets from the MPEG PCC Group.
>
---
#### [replaced 011] LV-MAE: Learning Long Video Representations through Masked-Embedding Autoencoders
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.03501v2](http://arxiv.org/pdf/2504.03501v2)**

> **作者:** Ilan Naiman; Emanuel Ben-Baruch; Oron Anschel; Alon Shoshan; Igor Kviatkovsky; Manoj Aggarwal; Gerard Medioni
>
> **备注:** Accepted to the International Conference on Computer Vision, ICCV 2025
>
> **摘要:** In this work, we introduce long-video masked-embedding autoencoders (LV-MAE), a self-supervised learning framework for long video representation. Our approach treats short- and long-span dependencies as two separate tasks. Such decoupling allows for a more intuitive video processing where short-span spatiotemporal primitives are first encoded and are then used to capture long-range dependencies across consecutive video segments. To achieve this, we leverage advanced off-the-shelf multimodal encoders to extract representations from short segments within the long video, followed by pre-training a masked-embedding autoencoder capturing high-level interactions across segments. LV-MAE is highly efficient to train and enables the processing of much longer videos by alleviating the constraint on the number of input frames. Furthermore, unlike existing methods that typically pre-train on short-video datasets, our approach offers self-supervised pre-training using long video samples (e.g., 20+ minutes video clips) at scale. Using LV-MAE representations, we achieve state-of-the-art results on three long-video benchmarks -- LVU, COIN, and Breakfast -- employing only a simple classification head for either attentive or linear probing. Finally, to assess LV-MAE pre-training and visualize its reconstruction quality, we leverage the video-language aligned space of short video representations to monitor LV-MAE through video-text retrieval. Code is available at https://github.com/amazon-science/lv-mae.
>
---
#### [replaced 012] Sparse VideoGen2: Accelerate Video Generation with Sparse Attention via Semantic-Aware Permutation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18875v3](http://arxiv.org/pdf/2505.18875v3)**

> **作者:** Shuo Yang; Haocheng Xi; Yilong Zhao; Muyang Li; Jintao Zhang; Han Cai; Yujun Lin; Xiuyu Li; Chenfeng Xu; Kelly Peng; Jianfei Chen; Song Han; Kurt Keutzer; Ion Stoica
>
> **摘要:** Diffusion Transformers (DiTs) are essential for video generation but suffer from significant latency due to the quadratic complexity of attention. By computing only critical tokens, sparse attention reduces computational costs and offers a promising acceleration approach. However, we identify that existing methods fail to approach optimal generation quality under the same computation budget for two reasons: (1) Inaccurate critical token identification: current methods cluster tokens based on position rather than semantics, leading to imprecise aggregated representations. (2) Excessive computation waste: critical tokens are scattered among non-critical ones, leading to wasted computation on GPUs, which are optimized for processing contiguous tokens. In this paper, we propose SVG2, a training-free framework that maximizes identification accuracy and minimizes computation waste, achieving a Pareto frontier trade-off between generation quality and efficiency. The core of SVG2 is semantic-aware permutation, which clusters and reorders tokens based on semantic similarity using k-means. This approach ensures both a precise cluster representation, improving identification accuracy, and a densified layout of critical tokens, enabling efficient computation without padding. Additionally, SVG2 integrates top-p dynamic budget control and customized kernel implementations, achieving up to 2.30x and 1.89x speedup while maintaining a PSNR of up to 30 and 26 on HunyuanVideo and Wan 2.1, respectively. Our code is open-sourced at \href{https://github.com/svg-project/Sparse-VideoGen}{https://github.com/svg-project/Sparse-VideoGen}.
>
---
#### [replaced 013] Safe-LLaVA: A Privacy-Preserving Vision-Language Dataset and Benchmark for Biometric Safety
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.00192v2](http://arxiv.org/pdf/2509.00192v2)**

> **作者:** Younggun Kim; Sirnam Swetha; Fazil Kagdi; Mubarak Shah
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in vision-language tasks. However, these models often infer and reveal sensitive biometric attributes such as race, gender, age, body weight, and eye color; even when such information is not explicitly requested. This raises critical concerns, particularly in real-world applications and socially-sensitive domains. Despite increasing awareness, no publicly available dataset or benchmark exists to comprehensively evaluate or mitigate biometric leakage in MLLMs. To address this gap, we introduce PRISM (Privacy-aware Evaluation of Responses in Sensitive Modalities), a new benchmark designed to assess MLLMs on two fronts: (1) refuse biometric-related queries and (2) implicit biometric leakage in general responses while maintaining semantic faithfulness. Further, we conduct a detailed audit of the widely used LLaVA datasets and uncover extensive biometric leakage across pretraining and instruction data. To address this, we present Safe-LLaVA dataset, the first privacy-preserving MLLM training dataset constructed by systematically removing explicit and implicit biometric information from LLaVA dataset. Our evaluations on PRISM reveal biometric leakages across MLLMs for different attributes, highlighting the detailed privacy-violations. We also fine-tune a model on Safe-LLaVA dataset and show that it substantially reduces the biometric leakages. Together, Safe-LLaVA and PRISM set a new standard for privacy-aligned development and evaluation of MLLMs.
>
---
#### [replaced 014] Step-by-Step Video-to-Audio Synthesis via Negative Audio Guidance
- **分类: cs.CV; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.20995v3](http://arxiv.org/pdf/2506.20995v3)**

> **作者:** Akio Hayakawa; Masato Ishii; Takashi Shibuya; Yuki Mitsufuji
>
> **摘要:** We propose a step-by-step video-to-audio (V2A) generation method for finer controllability over the generation process and more realistic audio synthesis. Inspired by traditional Foley workflows, our approach aims to comprehensively capture all sound events induced by a video through the incremental generation of missing sound events. To avoid the need for costly multi-reference video-audio datasets, each generation step is formulated as a negatively guided V2A process that discourages duplication of existing sounds. The guidance model is trained by finetuning a pre-trained V2A model on audio pairs from adjacent segments of the same video, allowing training with standard single-reference audiovisual datasets that are easily accessible. Objective and subjective evaluations demonstrate that our method enhances the separability of generated sounds at each step and improves the overall quality of the final composite audio, outperforming existing baselines.
>
---
#### [replaced 015] Unified Cross-Modal Medical Image Synthesis with Hierarchical Mixture of Product-of-Experts
- **分类: cs.CV; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2410.19378v3](http://arxiv.org/pdf/2410.19378v3)**

> **作者:** Reuben Dorent; Nazim Haouchine; Alexandra Golby; Sarah Frisken; Tina Kapur; William Wells
>
> **备注:** Accepted in IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
>
> **摘要:** We propose a deep mixture of multimodal hierarchical variational auto-encoders called MMHVAE that synthesizes missing images from observed images in different modalities. MMHVAE's design focuses on tackling four challenges: (i) creating a complex latent representation of multimodal data to generate high-resolution images; (ii) encouraging the variational distributions to estimate the missing information needed for cross-modal image synthesis; (iii) learning to fuse multimodal information in the context of missing data; (iv) leveraging dataset-level information to handle incomplete data sets at training time. Extensive experiments are performed on the challenging problem of pre-operative brain multi-parametric magnetic resonance and intra-operative ultrasound imaging.
>
---
#### [replaced 016] Enhancing Fitness Movement Recognition with Attention Mechanism and Pre-Trained Feature Extractors
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.02511v2](http://arxiv.org/pdf/2509.02511v2)**

> **作者:** Shanjid Hasan Nishat; Srabonti Deb; Mohiuddin Ahmed
>
> **备注:** 6 pages,9 figures, 2025 28th International Conference on Computer and Information Technology (ICCIT)
>
> **摘要:** Fitness movement recognition, a focused subdomain of human activity recognition (HAR), plays a vital role in health monitoring, rehabilitation, and personalized fitness training by enabling automated exercise classification from video data. However, many existing deep learning approaches rely on computationally intensive 3D models, limiting their feasibility in real-time or resource-constrained settings. In this paper, we present a lightweight and effective framework that integrates pre-trained 2D Convolutional Neural Networks (CNNs) such as ResNet50, EfficientNet, and Vision Transformers (ViT) with a Long Short-Term Memory (LSTM) network enhanced by spatial attention. These models efficiently extract spatial features while the LSTM captures temporal dependencies, and the attention mechanism emphasizes informative segments. We evaluate the framework on a curated subset of the UCF101 dataset, achieving a peak accuracy of 93.34\% with the ResNet50-based configuration. Comparative results demonstrate the superiority of our approach over several state-of-the-art HAR systems. The proposed method offers a scalable and real-time-capable solution for fitness activity recognition with broader applications in vision-based health and activity monitoring.
>
---
#### [replaced 017] AuxDet: Auxiliary Metadata Matters for Omni-Domain Infrared Small Target Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15184v2](http://arxiv.org/pdf/2505.15184v2)**

> **作者:** Yangting Shi; Yinfei Zhu; Renjie He; Le Hui; Meng Cai; Ming-Ming Cheng; Yimian Dai
>
> **摘要:** Omni-domain infrared small target detection (Omni-IRSTD) poses formidable challenges, as a single model must seamlessly adapt to diverse imaging systems, varying resolutions, and multiple spectral bands simultaneously. Current approaches predominantly rely on visual-only modeling paradigms that not only struggle with complex background interference and inherently scarce target features, but also exhibit limited generalization capabilities across complex omni-scene environments where significant domain shifts and appearance variations occur. In this work, we reveal a critical oversight in existing paradigms: the neglect of readily available auxiliary metadata describing imaging parameters and acquisition conditions, such as spectral bands, sensor platforms, resolution, and observation perspectives. To address this limitation, we propose the Auxiliary Metadata Driven Infrared Small Target Detector (AuxDet), a novel multimodal framework that is the first to incorporate metadata into the IRSTD paradigm for scene-aware optimization. Through a high-dimensional fusion module based on multi-layer perceptrons (MLPs), AuxDet dynamically integrates metadata semantics with visual features, guiding adaptive representation learning for each individual sample. Additionally, we design a lightweight prior-initialized enhancement module using 1D convolutional blocks to further refine fused features and recover fine-grained target cues. Extensive experiments on the challenging WideIRSTD-Full benchmark demonstrate that AuxDet consistently outperforms state-of-the-art methods, validating the critical role of auxiliary information in improving robustness and accuracy in omni-domain IRSTD tasks. Code is available at https://github.com/GrokCV/AuxDet.
>
---
#### [replaced 018] Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.03993v2](http://arxiv.org/pdf/2510.03993v2)**

> **作者:** Yaxin Hou; Bo Han; Yuheng Jia; Hui Liu; Junhui Hou
>
> **备注:** The paper is accepted by NeurIPS 2025
>
> **摘要:** Current long-tailed semi-supervised learning methods assume that labeled data exhibit a long-tailed distribution, and unlabeled data adhere to a typical predefined distribution (i.e., long-tailed, uniform, or inverse long-tailed). However, the distribution of the unlabeled data is generally unknown and may follow an arbitrary distribution. To tackle this challenge, we propose a Controllable Pseudo-label Generation (CPG) framework, expanding the labeled dataset with the progressively identified reliable pseudo-labels from the unlabeled dataset and training the model on the updated labeled dataset with a known distribution, making it unaffected by the unlabeled data distribution. Specifically, CPG operates through a controllable self-reinforcing optimization cycle: (i) at each training step, our dynamic controllable filtering mechanism selectively incorporates reliable pseudo-labels from the unlabeled dataset into the labeled dataset, ensuring that the updated labeled dataset follows a known distribution; (ii) we then construct a Bayes-optimal classifier using logit adjustment based on the updated labeled data distribution; (iii) this improved classifier subsequently helps identify more reliable pseudo-labels in the next training step. We further theoretically prove that this optimization cycle can significantly reduce the generalization error under some conditions. Additionally, we propose a class-aware adaptive augmentation module to further improve the representation of minority classes, and an auxiliary branch to maximize data utilization by leveraging all labeled and unlabeled samples. Comprehensive evaluations on various commonly used benchmark datasets show that CPG achieves consistent improvements, surpassing state-of-the-art methods by up to $\textbf{15.97\%}$ in accuracy. The code is available at https://github.com/yaxinhou/CPG.
>
---
#### [replaced 019] A Graph-Based Framework for Interpretable Whole Slide Image Analysis
- **分类: eess.IV; cs.AI; cs.CV; cs.LG; q-bio.QM**

- **链接: [http://arxiv.org/pdf/2503.11846v2](http://arxiv.org/pdf/2503.11846v2)**

> **作者:** Alexander Weers; Alexander H. Berger; Laurin Lux; Peter Schüffler; Daniel Rueckert; Johannes C. Paetzold
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** The histopathological analysis of whole-slide images (WSIs) is fundamental to cancer diagnosis but is a time-consuming and expert-driven process. While deep learning methods show promising results, dominant patch-based methods artificially fragment tissue, ignore biological boundaries, and produce black-box predictions. We overcome these limitations with a novel framework that transforms gigapixel WSIs into biologically-informed graph representations and is interpretable by design. Our approach builds graph nodes from tissue regions that respect natural structures, not arbitrary grids. We introduce an adaptive graph coarsening technique, guided by learned embeddings, to efficiently merge homogeneous regions while preserving diagnostically critical details in heterogeneous areas. Each node is enriched with a compact, interpretable feature set capturing clinically-motivated priors. A graph attention network then performs diagnosis on this compact representation. We demonstrate strong performance on challenging cancer staging and survival prediction tasks. Crucially, our resource-efficient model ($>$13x fewer parameters and $>$300x less data) achieves results competitive with a massive foundation model, while offering full interpretability through feature attribution. Our code is publicly available at https://github.com/HistoGraph31/pix2pathology.
>
---
#### [replaced 020] MoSA: Motion-Coherent Human Video Generation via Structure-Appearance Decoupling
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.17404v2](http://arxiv.org/pdf/2508.17404v2)**

> **作者:** Haoyu Wang; Hao Tang; Donglin Di; Zhilu Zhang; Wangmeng Zuo; Feng Gao; Siwei Ma; Shiliang Zhang
>
> **备注:** Project: https://hywang2002.github.io/MoSA
>
> **摘要:** Existing video generation models predominantly emphasize appearance fidelity while exhibiting limited ability to synthesize complex human motions, such as whole-body movements, long-range dynamics, and fine-grained human-environment interactions. This often leads to unrealistic or physically implausible movements with inadequate structural coherence. To conquer these challenges, we propose MoSA, which decouples the process of human video generation into two components, i.e., structure generation and appearance generation. MoSA first employs a 3D structure transformer to generate a human motion sequence from the text prompt. The remaining video appearance is then synthesized under the guidance of this structural sequence. We achieve fine-grained control over the sparse human structures by introducing Human-Aware Dynamic Control modules with a dense tracking constraint during training. The modeling of human-environment interactions is improved through the proposed contact constraint. Those two components work comprehensively to ensure the structural and appearance fidelity across the generated videos. This paper also contributes a large-scale human video dataset, which features more complex and diverse motions than existing human video datasets. We conduct comprehensive comparisons between MoSA and a variety of approaches, including general video generation models, human video generation models, and human animation models. Experiments demonstrate that MoSA substantially outperforms existing approaches across the majority of evaluation metrics.
>
---
#### [replaced 021] Robust Object Detection for Autonomous Driving via Curriculum-Guided Group Relative Policy Optimization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.22688v2](http://arxiv.org/pdf/2509.22688v2)**

> **作者:** Xu Jia
>
> **摘要:** Multimodal Large Language Models (MLLMs) excel in vision-language reasoning but often struggle with structured perception tasks requiring precise localization and robustness. We propose a reinforcement learning framework that augments Group Relative Policy Optimization (GRPO) with curriculum-based data scheduling and difficulty-aware filtering. This approach stabilizes optimization under sparse, noisy rewards and enables progressive adaptation to complex samples. Evaluations on autonomous driving benchmarks demonstrate substantial improvements in detection accuracy and robustness. Ablation studies confirm the importance of reward design, KL regularization, and curriculum pacing for convergence stability and generalization. Our findings highlight reinforcement-driven optimization with structured data curricula as a scalable path toward robust and interpretable multimodal detection.
>
---
#### [replaced 022] Adapting Large Language Models to Mitigate Skin Tone Biases in Clinical Dermatology Tasks: A Mixed-Methods Study
- **分类: eess.IV; cs.CV; cs.CY**

- **链接: [http://arxiv.org/pdf/2510.00055v2](http://arxiv.org/pdf/2510.00055v2)**

> **作者:** Kiran Nijjer; Ryan Bui; Derek Jiu; Adnan Ahmed; Peter Wang; Kevin Zhu; Lilly Zhu
>
> **备注:** Accepted to EADV (European Academy of Dermatology) and SID (Society for Investigative Dermatology)
>
> **摘要:** SkinGPT-4, a large vision-language model, leverages annotated skin disease images to augment clinical workflows in underserved communities. However, its training dataset predominantly represents lighter skin tones, limiting diagnostic accuracy for darker tones. Here, we evaluated performance biases in SkinGPT-4 across skin tones on common skin diseases, including eczema, allergic-contact dermatitis, and psoriasis using the open-sourced SCIN dataset. We leveraged the SkinGPT-4 backbone to develop finetuned models for custom skin disease classification tasks and explored bias mitigation strategies. Clinical evaluation by board-certified dermatologists on six relevant skin diseases from 300 SCIN cases assessed images for diagnostic accuracy, informativity, physician utility, and patient utility. Model fairness metrics, including demographic parity and equalized odds, were calculated across skin tones. SkinGPT-4 achieved an average demographic parity of 0.10 across Fitzpatrick types, with notable differences of 0.10-0.15 between lightest and darkest tones across evaluation metrics. Model hallucinations in artifacts and anatomy occurred at a rate of 17.8. Our customized models achieved average F1, precision, and AUROC of 0.75, 0.78, and 0.78 across visually similar disease pairs. Fairness analysis showed an average demographic parity of 0.75, with a maximum disparity of 0.21 across skin tones. The best model achieved parity scores of 0.83, 0.83, 0.76, 0.89, 0.90, and 0.90 for Fitzpatrick I-VI, indicating robust fairness. Large language models such as SkinGPT-4 showed weaker performance on darker tones. Model biases exist across evaluation criteria, and hallucinations may affect diagnostic efficacy. These findings demonstrate the efficacy of training accurate, fair models using existing backbones for custom skin disease classification.
>
---
#### [replaced 023] Low-Rank Tensor Recovery via Variational Schatten-p Quasi-Norm and Jacobian Regularization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.22134v2](http://arxiv.org/pdf/2506.22134v2)**

> **作者:** Zhengyun Cheng; Ruizhe Zhang; Guanwen Zhang; Yi Xu; Xiangyang Ji; Wei Zhou
>
> **摘要:** Higher-order tensors are well-suited for representing multi-dimensional data, such as images and videos, which typically characterize low-rank structures. Low-rank tensor decomposition has become essential in machine learning and computer vision, but existing methods like Tucker decomposition offer flexibility at the expense of interpretability. The CANDECOMP/PARAFAC (CP) decomposition provides a natural and interpretable structure, while obtaining a sparse solutions remains challenging. Leveraging the rich properties of CP decomposition, we propose a CP-based low-rank tensor function parameterized by neural networks (NN) for implicit neural representation. This approach can model the tensor both on-grid and beyond grid, fully utilizing the non-linearity of NN with theoretical guarantees on excess risk bounds. To achieve sparser CP decomposition, we introduce a variational Schatten-p quasi-norm to prune redundant rank-1 components and prove that it serves as a common upper bound for the Schatten-p quasi-norms of arbitrary unfolding matrices. For smoothness, we propose a regularization term based on the spectral norm of the Jacobian and Hutchinson's trace estimator. The proposed smoothness regularization is SVD-free and avoids explicit chain rule derivations. It can serve as an alternative to Total Variation (TV) regularization in image denoising tasks and is naturally applicable to implicit neural representation. Extensive experiments on multi-dimensional data recovery tasks, including image inpainting, denoising, and point cloud upsampling, demonstrate the superiority and versatility of our method compared to state-of-the-art approaches. The code is available at https://github.com/CZY-Code/CP-Pruner.
>
---
#### [replaced 024] Noise2Score3D: Tweedie's Approach for Unsupervised Point Cloud Denoising
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.09283v3](http://arxiv.org/pdf/2503.09283v3)**

> **作者:** Xiangbin Wei; Yuanfeng Wang; Ao XU; Lingyu Zhu; Dongyong Sun; Keren Li; Yang Li; Qi Qin
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2502.16826
>
> **摘要:** Building on recent advances in Bayesian statistics and image denoising, we propose Noise2Score3D, a fully unsupervised framework for point cloud denoising. Noise2Score3D learns the score function of the underlying point cloud distribution directly from noisy data, eliminating the need for clean data during training. Using Tweedie's formula, our method performs denoising in a single step, avoiding the iterative processes used in existing unsupervised methods, thus improving both accuracy and efficiency. Additionally, we introduce Total Variation for Point Clouds as a denoising quality metric, which allows for the estimation of unknown noise parameters. Experimental results demonstrate that Noise2Score3D achieves state-of-the-art performance on standard benchmarks among unsupervised learning methods in Chamfer distance and point-to-mesh metrics. Noise2Score3D also demonstrates strong generalization ability beyond training datasets. Our method, by addressing the generalization issue and challenge of the absence of clean data in learning-based methods, paves the way for learning-based point cloud denoising methods in real-world applications.
>
---
#### [replaced 025] Accuracy-Robustness Trade Off via Spiking Neural Network Gradient Sparsity Trail
- **分类: cs.NE; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.23762v2](http://arxiv.org/pdf/2509.23762v2)**

> **作者:** Nhan T. Luu
>
> **备注:** Work under peer-review
>
> **摘要:** Spiking Neural Networks (SNNs) have attracted growing interest in both computational neuroscience and artificial intelligence, primarily due to their inherent energy efficiency and compact memory footprint. However, achieving adversarial robustness in SNNs, particularly for vision-related tasks, remains a nascent and underexplored challenge. Recent studies have proposed leveraging sparse gradients as a form of regularization to enhance robustness against adversarial perturbations. In this work, we present a surprising finding: under specific architectural configurations, SNNs exhibit natural gradient sparsity and can achieve state-of-the-art adversarial defense performance without the need for any explicit regularization. Further analysis reveals a trade-off between robustness and generalization: while sparse gradients contribute to improved adversarial resilience, they can impair the model's ability to generalize; conversely, denser gradients support better generalization but increase vulnerability to attacks.
>
---
#### [replaced 026] Bridging Semantic Logic Gaps: A Cognition Inspired Multimodal Boundary Preserving Network for Image Manipulation Localization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.07216v3](http://arxiv.org/pdf/2508.07216v3)**

> **作者:** Songlin Li; Zhiqing Guo; Yuanman Li; Zeyu Li; Yunfeng Diao; Gaobo Yang; Liejun Wang
>
> **摘要:** The existing image manipulation localization (IML) models mainly relies on visual cues, but ignores the semantic logical relationships between content features. In fact, the content semantics conveyed by real images often conform to human cognitive laws. However, image manipulation technology usually destroys the internal relationship between content features, thus leaving semantic clues for IML. In this paper, we propose a cognition inspired multimodal boundary preserving network (CMB-Net). Specifically, CMB-Net utilizes large language models (LLMs) to analyze manipulated regions within images and generate prompt-based textual information to compensate for the lack of semantic relationships in the visual information. Considering that the erroneous texts induced by hallucination from LLMs will damage the accuracy of IML, we propose an image-text central ambiguity module (ITCAM). It assigns weights to the text features by quantifying the ambiguity between text and image features, thereby ensuring the beneficial impact of textual information. We also propose an image-text interaction module (ITIM) that aligns visual and text features using a correlation matrix for fine-grained interaction. Finally, inspired by invertible neural networks, we propose a restoration edge decoder (RED) that mutually generates input and output features to preserve boundary information in manipulated regions without loss. Extensive experiments show that CMB-Net outperforms most existing IML models. Our code is available on https://github.com/vpsg-research/CMB-Net.
>
---
#### [replaced 027] Evaluation of Deformable Image Registration under Alignment-Regularity Trade-of
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.07185v3](http://arxiv.org/pdf/2503.07185v3)**

> **作者:** Vasiliki Sideri-Lampretsa; Daniel Rueckert; Huaqi Qiu
>
> **摘要:** Evaluating deformable image registration (DIR) is challenging due to the inherent trade-off between achieving high alignment accuracy and maintaining deformation regularity. However, most existing DIR works either address this trade-off inadequately or overlook it altogether. In this paper, we highlight the issues with existing practices and propose an evaluation scheme that captures the trade-off continuously to holistically evaluate DIR methods. We first introduce the alignment regularity characteristic (ARC) curves, which describe the performance of a given registration method as a spectrum under various degrees of regularity. We demonstrate that the ARC curves reveal unique insights that are not evident from existing evaluation practices, using experiments on representative deep learning DIR methods with various network architectures and transformation models. We further adopt a HyperNetwork based approach that learns to continuously interpolate across the full regularization range, accelerating the construction and improving the sample density of ARC curves. Finally, we provide general guidelines for a nuanced model evaluation and selection using our evaluation scheme for both practitioners and registration researchers.
>
---
#### [replaced 028] Towards Unified Image Deblurring using a Mixture-of-Experts Decoder
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.06228v2](http://arxiv.org/pdf/2508.06228v2)**

> **作者:** Daniel Feijoo; Paula Garrido-Mellado; Jaesung Rim; Alvaro Garcia; Marcos V. Conde
>
> **摘要:** Image deblurring, removing blurring artifacts from images, is a fundamental task in computational photography and low-level computer vision. Existing approaches focus on specialized solutions tailored to particular blur types, thus, these solutions lack generalization. This limitation in current methods implies requiring multiple models to cover several blur types, which is not practical in many real scenarios. In this paper, we introduce the first all-in-one deblurring method capable of efficiently restoring images affected by diverse blur degradations, including global motion, local motion, blur in low-light conditions, and defocus blur. We propose a mixture-of-experts (MoE) decoding module, which dynamically routes image features based on the recognized blur degradation, enabling precise and efficient restoration in an end-to-end manner. Our unified approach not only achieves performance comparable to dedicated task-specific models, but also shows promising generalization to unseen blur scenarios, particularly when leveraging appropriate expert selection. Code available at https://github.com/cidautai/DeMoE.
>
---
#### [replaced 029] A discussion about violin reduction: geometric analysis of contour lines and channel of minima
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2404.01995v2](http://arxiv.org/pdf/2404.01995v2)**

> **作者:** Philémon Beghin; Anne-Emmanuelle Ceulemans; François Glineur
>
> **备注:** Paper accepted for the Florence Heri-Tech 2024 Conference
>
> **摘要:** Some early violins have been reduced during their history to fit imposed morphological standards, while more recent ones have been built directly to these standards. We can observe differences between reduced and unreduced instruments, particularly in their contour lines and channel of minima. In a recent preliminary work, we computed and highlighted those two features for two instruments using triangular 3D meshes acquired by photogrammetry, whose fidelity has been assessed and validated with sub-millimetre accuracy. We propose here an extension to a corpus of 38 violins, violas and cellos, and introduce improved procedures, leading to a stronger discussion of the geometric analysis. We first recall the material we are working with. We then discuss how to derive the best reference plane for the violin alignment, which is crucial for the computation of contour lines and channel of minima. Finally, we show how to compute efficiently both characteristics and we illustrate our results with a few examples.
>
---
#### [replaced 030] VisioMath: Benchmarking Figure-based Mathematical Reasoning in LMMs
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.06727v3](http://arxiv.org/pdf/2506.06727v3)**

> **作者:** Can Li; Ying Liu; Ting Zhang; Mei Wang; Hua Huang
>
> **摘要:** Large Multimodal Models have achieved remarkable progress in integrating vision and language, enabling strong performance across perception, reasoning, and domain-specific tasks. However, their capacity to reason over multiple, visually similar inputs remains insufficiently explored. Such fine-grained comparative reasoning is central to real-world tasks, especially in mathematics and education, where learners must often distinguish between nearly identical diagrams to identify correct solutions. To address this gap, we present VisioMath, a curated benchmark of 1,800 high-quality K-12 mathematics problems in which all candidate answers are diagrams with subtle visual similarities. A comprehensive evaluation of state-of-the-art LMMs, covering both leading closed-source systems and widely adopted open-source models, reveals a consistent decline in accuracy as inter-image similarity increases. Analysis indicates that the dominant failure mode stems from image-text misalignment: rather than grounding reasoning in textual cues, models often resort to shallow positional heuristics, resulting in systematic errors. We further explore three alignment-oriented strategies, spanning training-free approaches and finetuning, and achieve substantial accuracy gains. We hope that VisioMath will serve as a rigorous benchmark and catalyst for developing LMMs toward deeper diagram understanding, precise comparative reasoning, and grounded multi-image-text integration.
>
---
#### [replaced 031] Ouroboros: Single-step Diffusion Models for Cycle-consistent Forward and Inverse Rendering
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.14461v2](http://arxiv.org/pdf/2508.14461v2)**

> **作者:** Shanlin Sun; Yifan Wang; Hanwen Zhang; Yifeng Xiong; Qin Ren; Ruogu Fang; Xiaohui Xie; Chenyu You
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** While multi-step diffusion models have advanced both forward and inverse rendering, existing approaches often treat these problems independently, leading to cycle inconsistency and slow inference speed. In this work, we present Ouroboros, a framework composed of two single-step diffusion models that handle forward and inverse rendering with mutual reinforcement. Our approach extends intrinsic decomposition to both indoor and outdoor scenes and introduces a cycle consistency mechanism that ensures coherence between forward and inverse rendering outputs. Experimental results demonstrate state-of-the-art performance across diverse scenes while achieving substantially faster inference speed compared to other diffusion-based methods. We also demonstrate that Ouroboros can transfer to video decomposition in a training-free manner, reducing temporal inconsistency in video sequences while maintaining high-quality per-frame inverse rendering.
>
---
#### [replaced 032] Teaching Metric Distance to Discrete Autoregressive Language Models
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.02379v4](http://arxiv.org/pdf/2503.02379v4)**

> **作者:** Jiwan Chung; Saejin Kim; Yongrae Jo; Jaewoo Park; Dongjun Min; Youngjae Yu
>
> **摘要:** As large language models expand beyond natural language to domains such as mathematics, multimodal understanding, and embodied agents, tokens increasingly reflect metric relationships rather than purely linguistic meaning. We introduce DIST2Loss, a distance-aware framework designed to train autoregressive discrete models by leveraging predefined distance relationships among output tokens. At its core, DIST2Loss transforms continuous exponential family distributions derived from inherent distance metrics into discrete, categorical optimization targets compatible with the models' architectures. This approach enables the models to learn and preserve meaningful distance relationships during token generation while maintaining compatibility with existing architectures. Empirical evaluations show consistent performance gains in diverse multimodal applications, including visual grounding, robotic manipulation, generative reward modeling, and image generation using vector-quantized features. These improvements are most notable in low-data regimes, demonstrating DIST2Loss's strength under resource constraints.
>
---
#### [replaced 033] MoME: Estimating Psychological Traits from Gait with Multi-Stage Mixture of Movement Experts
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2510.04654v2](http://arxiv.org/pdf/2510.04654v2)**

> **作者:** Andy Cǎtrunǎ; Adrian Cosma; Emilian Rǎdoi
>
> **备注:** 4 Figures, 4 Tables
>
> **摘要:** Gait encodes rich biometric and behavioural information, yet leveraging the manner of walking to infer psychological traits remains a challenging and underexplored problem. We introduce a hierarchical Multi-Stage Mixture of Movement Experts (MoME) architecture for multi-task prediction of psychological attributes from gait sequences represented as 2D poses. MoME processes the walking cycle in four stages of movement complexity, employing lightweight expert models to extract spatio-temporal features and task-specific gating modules to adaptively weight experts across traits and stages. Evaluated on the PsyMo benchmark covering 17 psychological traits, our method outperforms state-of-the-art gait analysis models, achieving a 37.47% weighted F1 score at the run level and 44.6% at the subject level. Our experiments show that integrating auxiliary tasks such as identity recognition, gender prediction, and BMI estimation further improves psychological trait estimation. Our findings demonstrate the viability of multi-task gait-based learning for psychological trait estimation and provide a foundation for future research on movement-informed psychological inference.
>
---
#### [replaced 034] Sparse Representations Improve Adversarial Robustness of Neural Network Classifiers
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.21130v2](http://arxiv.org/pdf/2509.21130v2)**

> **作者:** Killian Steunou; Théo Druilhe; Sigurd Saue
>
> **备注:** Killian Steunou is the main contributor and corresponding author of this work
>
> **摘要:** Deep neural networks perform remarkably well on image classification tasks but remain vulnerable to carefully crafted adversarial perturbations. This work revisits linear dimensionality reduction as a simple, data-adapted defense. We empirically compare standard Principal Component Analysis (PCA) with its sparse variant (SPCA) as front-end feature extractors for downstream classifiers, and we complement these experiments with a theoretical analysis. On the theory side, we derive exact robustness certificates for linear heads applied to SPCA features: for both $\ell_\infty$ and $\ell_2$ threat models (binary and multiclass), the certified radius grows as the dual norms of $W^\top u$ shrink, where $W$ is the projection and $u$ the head weights. We further show that for general (non-linear) heads, sparsity reduces operator-norm bounds through a Lipschitz composition argument, predicting lower input sensitivity. Empirically, with a small non-linear network after the projection, SPCA consistently degrades more gracefully than PCA under strong white-box and black-box attacks while maintaining competitive clean accuracy. Taken together, the theory identifies the mechanism (sparser projections reduce adversarial leverage) and the experiments verify that this benefit persists beyond the linear setting. Our code is available at https://github.com/killian31/SPCARobustness.
>
---
#### [replaced 035] SAMCIRT: A Simultaneous Reconstruction and Affine Motion Compensation Technique for Four Dimensional Computed Tomography (4DCT)
- **分类: eess.IV; cs.CV; math.OC; 65K10, 68U10, 68W01, 92C55, 94A08**

- **链接: [http://arxiv.org/pdf/2402.04480v2](http://arxiv.org/pdf/2402.04480v2)**

> **作者:** Anh-Tuan Nguyen; Jens Renders; Khoi-Nguyen Nguyen; Tat-Dat To; Domenico Iuso; Yves Maris
>
> **备注:** 25 pages, revised version submitted to the SIAM Journal on Imaging Sciences (SIIMS)
>
> **摘要:** The majority of the recent iterative approaches in 4DCT not only rely on nested iterations, thereby increasing computational complexity and constraining potential acceleration, but also fail to provide a theoretical proof of convergence for their proposed iterative schemes. On the other hand, the latest MATLAB and Python image processing toolboxes lack the implementation of analytic adjoints of affine motion operators for 3D object volumes, which does not allow gradient methods using exact derivatives towards affine motion parameters. In this work, we propose the Simultaneous Affine Motion-Compensated Image Reconstruction Technique (SAMCIRT)- an efficient iterative reconstruction scheme that combines image reconstruction and affine motion estimation in a single update step, based on the analytic adjoints of the motion operators then exact partial derivatives with respect to both the reconstruction and the affine motion parameters. Moreover, we prove the separated Lipschitz continuity of the objective function and its associated functions, including the gradient, which supports the convergence of our proposed iterative scheme, despite the non-convexity of the objective function with respect to the affine motion parameters. Results from simulation and real experiments show that our method outperforms the state-of-the-art CT reconstruction with affine motion correction methods in computational feasibility and projection distance. In particular, this allows accurate reconstruction for a real, nonstationary diamond, showing a novel application of 4DCT.
>
---
#### [replaced 036] SpaCE-10: A Comprehensive Benchmark for Multimodal Large Language Models in Compositional Spatial Intelligence
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.07966v4](http://arxiv.org/pdf/2506.07966v4)**

> **作者:** Ziyang Gong; Wenhao Li; Oliver Ma; Songyuan Li; Zhaokai Wang; Songyuan Li; Jiayi Ji; Xue Yang; Gen Luo; Junchi Yan; Rongrong Ji
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved remarkable progress in various multimodal tasks. To pursue higher intelligence in space, MLLMs require integrating multiple spatial capabilities, even for handling simple and normal tasks. However, existing benchmarks struggle to comprehensively evaluate the spatial intelligence of common MLLMs from the atomic level to the compositional level. To fill this gap, we present SpaCE-10, a comprehensive benchmark for compositional spatial evaluations. In SpaCE-10, we define 10 atomic spatial capabilities, which are combined to form 8 compositional capabilities. Based on these definitions, we propose a novel hierarchical annotation pipeline to generate high-quality and diverse question-answer (QA) pairs. With over 150+ hours of human expert effort, we obtain over 5k QA pairs for 811 real indoor scenes in SpaCE-10, which covers various evaluation settings like point cloud input and multi-choice QA. We conduct an extensive evaluation of common MLLMs on SpaCE-10 and find that even the most advanced MLLM still lags behind humans by large margins. Through our careful study, we also draw several significant findings that benefit the MLLM community. For example, we reveal that the shortcoming of counting capability greatly limits the compositional spatial capabilities of existing MLLMs.
>
---
#### [replaced 037] PartSDF: Part-Based Implicit Neural Representation for Composite 3D Shape Parametrization and Optimization
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.12985v2](http://arxiv.org/pdf/2502.12985v2)**

> **作者:** Nicolas Talabot; Olivier Clerc; Arda Cinar Demirtas; Hieu Le; Doruk Oner; Pascal Fua
>
> **备注:** Accepted to TMLR (27 pages, 14 figures)
>
> **摘要:** Accurate 3D shape representation is essential in engineering applications such as design, optimization, and simulation. In practice, engineering workflows require structured, part-based representations, as objects are inherently designed as assemblies of distinct components. However, most existing methods either model shapes holistically or decompose them without predefined part structures, limiting their applicability in real-world design tasks. We propose PartSDF, a supervised implicit representation framework that explicitly models composite shapes with independent, controllable parts while maintaining shape consistency. Thanks to its simple but innovative architecture, PartSDF outperforms both supervised and unsupervised baselines in reconstruction and generation tasks. We further demonstrate its effectiveness as a structured shape prior for engineering applications, enabling precise control over individual components while preserving overall coherence. Code available at https://github.com/cvlab-epfl/PartSDF.
>
---
#### [replaced 038] Robust Concept Erasure in Diffusion Models: A Theoretical Perspective on Security and Robustness
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.12024v2](http://arxiv.org/pdf/2509.12024v2)**

> **作者:** Zixuan Fu; Yan Ren; Finn Carter; Chenyue Wen; Le Ku; Daheng Yu; Emily Davis; Bo Zhang
>
> **备注:** updated version
>
> **摘要:** Diffusion models have achieved unprecedented success in image generation but pose increasing risks in terms of privacy, fairness, and security. A growing demand exists to \emph{erase} sensitive or harmful concepts (e.g., NSFW content, private individuals, artistic styles) from these models while preserving their overall generative capabilities. We introduce \textbf{SCORE} (Secure and Concept-Oriented Robust Erasure), a novel framework for robust concept removal in diffusion models. SCORE formulates concept erasure as an \emph{adversarial independence} problem, theoretically guaranteeing that the model's outputs become statistically independent of the erased concept. Unlike prior heuristic methods, SCORE minimizes the mutual information between a target concept and generated outputs, yielding provable erasure guarantees. We provide formal proofs establishing convergence properties and derive upper bounds on residual concept leakage. Empirically, we evaluate SCORE on Stable Diffusion and FLUX across four challenging benchmarks: object erasure, NSFW removal, celebrity face suppression, and artistic style unlearning. SCORE consistently outperforms state-of-the-art methods including EraseAnything, ANT, MACE, ESD, and UCE, achieving up to \textbf{12.5\%} higher erasure efficacy while maintaining comparable or superior image quality. By integrating adversarial optimization, trajectory consistency, and saliency-driven fine-tuning, SCORE sets a new standard for secure and robust concept erasure in diffusion models.
>
---
#### [replaced 039] Robust Neural Rendering in the Wild with Asymmetric Dual 3D Gaussian Splatting
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.03538v2](http://arxiv.org/pdf/2506.03538v2)**

> **作者:** Chengqi Li; Zhihao Shi; Yangdi Lu; Wenbo He; Xiangyu Xu
>
> **备注:** NeurIPS 2025 Spotlight; Project page: https://steveli88.github.io/AsymGS/
>
> **摘要:** 3D reconstruction from in-the-wild images remains a challenging task due to inconsistent lighting conditions and transient distractors. Existing methods typically rely on heuristic strategies to handle the low-quality training data, which often struggle to produce stable and consistent reconstructions, frequently resulting in visual artifacts.In this work, we propose \modelname{}, a novel framework that leverages the stochastic nature of these artifacts: they tend to vary across different training runs due to minor randomness. Specifically, our method trains two 3D Gaussian Splatting (3DGS) models in parallel, enforcing a consistency constraint that encourages convergence on reliable scene geometry while suppressing inconsistent artifacts. To prevent the two models from collapsing into similar failure modes due to confirmation bias, we introduce a divergent masking strategy that applies two complementary masks: a multi-cue adaptive mask and a self-supervised soft mask, which leads to an asymmetric training process of the two models, reducing shared error modes. In addition, to improve the efficiency of model training, we introduce a lightweight variant called Dynamic EMA Proxy, which replaces one of the two models with a dynamically updated Exponential Moving Average (EMA) proxy, and employs an alternating masking strategy to preserve divergence. Extensive experiments on challenging real-world datasets demonstrate that our method consistently outperforms existing approaches while achieving high efficiency. See the project website at https://steveli88.github.io/AsymGS.
>
---
#### [replaced 040] OpenFake: An Open Dataset and Platform Toward Real-World Deepfake Detection
- **分类: cs.CV; cs.AI; cs.LG; I.4.9; I.5.4; I.2.10**

- **链接: [http://arxiv.org/pdf/2509.09495v2](http://arxiv.org/pdf/2509.09495v2)**

> **作者:** Victor Livernoche; Akshatha Arodi; Andreea Musulan; Zachary Yang; Adam Salvail; Gaétan Marceau Caron; Jean-François Godbout; Reihaneh Rabbany
>
> **备注:** 26 pages, 12 figures
>
> **摘要:** Deepfakes, synthetic media created using advanced AI techniques, pose a growing threat to information integrity, particularly in politically sensitive contexts. This challenge is amplified by the increasing realism of modern generative models, which our human perception study confirms are often indistinguishable from real images. Yet, existing deepfake detection benchmarks rely on outdated generators or narrowly scoped datasets (e.g., single-face imagery), limiting their utility for real-world detection. To address these gaps, we present OpenFake, a large politically grounded dataset specifically crafted for benchmarking against modern generative models with high realism, and designed to remain extensible through an innovative crowdsourced adversarial platform that continually integrates new hard examples. OpenFake comprises nearly four million total images: three million real images paired with descriptive captions and almost one million synthetic counterparts from state-of-the-art proprietary and open-source models. Detectors trained on OpenFake achieve near-perfect in-distribution performance, strong generalization to unseen generators, and high accuracy on a curated in-the-wild social media test set, significantly outperforming models trained on existing datasets. Overall, we demonstrate that with high-quality and continually updated benchmarks, automatic deepfake detection is both feasible and effective in real-world settings.
>
---
#### [replaced 041] OneCAT: Decoder-Only Auto-Regressive Model for Unified Understanding and Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.03498v3](http://arxiv.org/pdf/2509.03498v3)**

> **作者:** Han Li; Xinyu Peng; Yaoming Wang; Zelin Peng; Xin Chen; Rongxiang Weng; Jingang Wang; Xunliang Cai; Wenrui Dai; Hongkai Xiong
>
> **备注:** technical report, project url:https://onecat-ai.github.io/
>
> **摘要:** We introduce OneCAT, a unified multimodal model that seamlessly integrates understanding, generation, and editing within a novel, pure decoder-only transformer architecture. Our framework uniquely eliminates the need for external components such as Vision Transformers (ViT) or vision tokenizer during inference, leading to significant efficiency gains, especially for high-resolution inputs. This is achieved through a modality-specific Mixture-of-Experts (MoE) structure trained with a single autoregressive (AR) objective, which also natively supports dynamic resolutions. Furthermore, we pioneer a multi-scale visual autoregressive mechanism within the Large Language Model (LLM) that drastically reduces decoding steps compared to diffusion-based methods while maintaining state-of-the-art performance. Our findings demonstrate the powerful potential of pure autoregressive modeling as a sufficient and elegant foundation for unified multimodal intelligence. As a result, OneCAT sets a new performance standard, outperforming existing open-source unified multimodal models across benchmarks for multimodal generation, editing, and understanding.
>
---
#### [replaced 042] Incremental Object Detection with Prompt-based Methods
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.14599v2](http://arxiv.org/pdf/2508.14599v2)**

> **作者:** Matthias Neuwirth-Trapp; Maarten Bieshaar; Danda Pani Paudel; Luc Van Gool
>
> **备注:** Accepted to ICCV Workshops 2025: v2 update affiliation
>
> **摘要:** Visual prompt-based methods have seen growing interest in incremental learning (IL) for image classification. These approaches learn additional embedding vectors while keeping the model frozen, making them efficient to train. However, no prior work has applied such methods to incremental object detection (IOD), leaving their generalizability unclear. In this paper, we analyze three different prompt-based methods under a complex domain-incremental learning setting. We additionally provide a wide range of reference baselines for comparison. Empirically, we show that the prompt-based approaches we tested underperform in this setting. However, a strong yet practical method, combining visual prompts with replaying a small portion of previous data, achieves the best results. Together with additional experiments on prompt length and initialization, our findings offer valuable insights for advancing prompt-based IL in IOD.
>
---
#### [replaced 043] ExGS: Extreme 3D Gaussian Compression with Diffusion Priors
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.24758v4](http://arxiv.org/pdf/2509.24758v4)**

> **作者:** Jiaqi Chen; Xinhao Ji; Yuanyuan Gao; Hao Li; Yuning Gong; Yifei Liu; Dan Xu; Zhihang Zhong; Dingwen Zhang; Xiao Sun
>
> **摘要:** Neural scene representations, such as 3D Gaussian Splatting (3DGS), have enabled high-quality neural rendering; however, their large storage and transmission costs hinder deployment in resource-constrained environments. Existing compression methods either rely on costly optimization, which is slow and scene-specific, or adopt training-free pruning and quantization, which degrade rendering quality under high compression ratios. In contrast, recent data-driven approaches provide a promising direction to overcome this trade-off, enabling efficient compression while preserving high rendering quality. We introduce ExGS, a novel feed-forward framework that unifies Universal Gaussian Compression (UGC) with GaussPainter for Extreme 3DGS compression. UGC performs re-optimization-free pruning to aggressively reduce Gaussian primitives while retaining only essential information, whereas GaussPainter leverages powerful diffusion priors with mask-guided refinement to restore high-quality renderings from heavily pruned Gaussian scenes. Unlike conventional inpainting, GaussPainter not only fills in missing regions but also enhances visible pixels, yielding substantial improvements in degraded renderings. To ensure practicality, it adopts a lightweight VAE and a one-step diffusion design, enabling real-time restoration. Our framework can even achieve over 100X compression (reducing a typical 354.77 MB model to about 3.31 MB) while preserving fidelity and significantly improving image quality under challenging conditions. These results highlight the central role of diffusion priors in bridging the gap between extreme compression and high-quality neural rendering. Our code repository will be released at: https://github.com/chenttt2001/ExGS
>
---
#### [replaced 044] Trajectory Prediction Meets Large Language Models: A Survey
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.03408v2](http://arxiv.org/pdf/2506.03408v2)**

> **作者:** Yi Xu; Ruining Yang; Yitian Zhang; Jianglin Lu; Mingyuan Zhang; Yizhou Wang; Lili Su; Yun Fu
>
> **备注:** 16 pages, GitHub: https://github.com/colorfulfuture/Awesome-Trajectory-Motion-Prediction-Papers
>
> **摘要:** Recent advances in large language models (LLMs) have sparked growing interest in integrating language-driven techniques into trajectory prediction. By leveraging their semantic and reasoning capabilities, LLMs are reshaping how autonomous systems perceive, model, and predict trajectories. This survey provides a comprehensive overview of this emerging field, categorizing recent work into five directions: (1) Trajectory prediction via language modeling paradigms, (2) Direct trajectory prediction with pretrained language models, (3) Language-guided scene understanding for trajectory prediction, (4) Language-driven data generation for trajectory prediction, (5) Language-based reasoning and interpretability for trajectory prediction. For each, we analyze representative methods, highlight core design choices, and identify open challenges. This survey bridges natural language processing and trajectory prediction, offering a unified perspective on how language can enrich trajectory prediction.
>
---
#### [replaced 045] A Comprehensive Survey of Mamba Architectures for Medical Image Analysis: Classification, Segmentation, Restoration and Beyond
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.02362v2](http://arxiv.org/pdf/2410.02362v2)**

> **作者:** Shubhi Bansal; Sreeharish A; Madhava Prasath J; Manikandan S; Sreekanth Madisetty; Mohammad Zia Ur Rehman; Chandravardhan Singh Raghaw; Gaurav Duggal; Nagendra Kumar
>
> **摘要:** Mamba, a special case of the State Space Model, is gaining popularity as an alternative to template-based deep learning approaches in medical image analysis. While transformers are powerful architectures, they have drawbacks, including quadratic computational complexity and an inability to address long-range dependencies efficiently. This limitation affects the analysis of large and complex datasets in medical imaging, where there are many spatial and temporal relationships. In contrast, Mamba offers benefits that make it well-suited for medical image analysis. It has linear time complexity, which is a significant improvement over transformers. Mamba processes longer sequences without attention mechanisms, enabling faster inference and requiring less memory. Mamba also demonstrates strong performance in merging multimodal data, improving diagnosis accuracy and patient outcomes. The organization of this paper allows readers to appreciate the capabilities of Mamba in medical imaging step by step. We begin by defining core concepts of SSMs and models, including S4, S5, and S6, followed by an exploration of Mamba architectures such as pure Mamba, U-Net variants, and hybrid models with convolutional neural networks, transformers, and Graph Neural Networks. We also cover Mamba optimizations, techniques and adaptations, scanning, datasets, applications, experimental results, and conclude with its challenges and future directions in medical imaging. This review aims to demonstrate the transformative potential of Mamba in overcoming existing barriers within medical imaging while paving the way for innovative advancements in the field. A comprehensive list of Mamba architectures applied in the medical field, reviewed in this work, is available at Github.
>
---
#### [replaced 046] Training Vision-Language Process Reward Models for Test-Time Scaling in Multimodal Reasoning: Key Insights and Lessons Learned
- **分类: cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.23250v3](http://arxiv.org/pdf/2509.23250v3)**

> **作者:** Brandon Ong; Tej Deep Pala; Vernon Toh; William Chandra Tjhi; Soujanya Poria
>
> **摘要:** Process Reward Models (PRMs) provide step-level supervision that improves the reliability of reasoning in large language models. While PRMs have been extensively studied in text-based domains, their extension to Vision Language Models (VLMs) remains limited. Existing Vision-Language PRMs (VL-PRMs) rely on Monte Carlo Tree Search (MCTS) for data construction, which can often produce noisy supervision signals and limit generalization across tasks. In this work, we aim to elucidate the design space of VL-PRMs by exploring diverse strategies for dataset construction, training, and test-time scaling. First, we introduce a hybrid data synthesis framework that combines MCTS with judgments from a strong VLM, producing more accurate step-level labels. Second, we propose perception-focused supervision, enabling our PRM to explicitly detect errors at the visual grounding stage of reasoning. Third, we systematically evaluate multiple test-time scaling strategies, showing that our PRMs can reliably guide VLMs toward more accurate solutions. Our experiments covering five diverse multimodal benchmarks (MMMU, PuzzleVQA, AlgoPuzzleVQA, MathVista, and MathVision) reveal several key insights: (i) VL-PRMs when used as Outcome Reward Models (ORMs) during test-time scaling (TTS) can outperform VL-PRM guided process step selection, (ii) smaller VL-PRMs can match or even surpass larger ones in detecting process errors, (iii) VL-PRMs uncover latent reasoning abilities in stronger VLM backbones, (iv) perception-level supervision leads to significant gains in test-time scaling, and (v) TTS performance of different policies improve on advanced math reasoning datasets despite not training VL-PRMs on such datasets. We hope our work will motivate further research and support the advancement of VLMs.
>
---
#### [replaced 047] Human + AI for Accelerating Ad Localization Evaluation
- **分类: cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.12543v3](http://arxiv.org/pdf/2509.12543v3)**

> **作者:** Harshit Rajgarhia; Shivali Dalmia; Mengyang Zhao; Mukherji Abhishek; Kiran Ganesh
>
> **摘要:** Adapting advertisements for multilingual audiences requires more than simple text translation; it demands preservation of visual consistency, spatial alignment, and stylistic integrity across diverse languages and formats. We introduce a structured framework that combines automated components with human oversight to address the complexities of advertisement localization. To the best of our knowledge, this is the first work to integrate scene text detection, inpainting, machine translation (MT), and text reimposition specifically for accelerating ad localization evaluation workflows. Qualitative results across six locales demonstrate that our approach produces semantically accurate and visually coherent localized advertisements, suitable for deployment in real-world workflows.
>
---
#### [replaced 048] The Mirage of Performance Gains: Why Contrastive Decoding Fails to Mitigate Object Hallucinations in MLLMs?
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.10020v3](http://arxiv.org/pdf/2504.10020v3)**

> **作者:** Hao Yin; Guangzong Si; Zilei Wang
>
> **摘要:** Contrastive decoding strategies are widely used to reduce object hallucinations in multimodal large language models (MLLMs). These methods work by constructing contrastive samples to induce hallucinations and then suppressing them in the output distribution. However, this paper demonstrates that such approaches fail to effectively mitigate the hallucination problem. The performance improvements observed on POPE Benchmark are largely driven by two misleading factors: (1) crude, unidirectional adjustments to the model's output distribution and (2) the adaptive plausibility constraint, which reduces the sampling strategy to greedy search. To further illustrate these issues, we introduce a series of spurious improvement methods and evaluate their performance against contrastive decoding techniques. Experimental results reveal that the observed performance gains in contrastive decoding are entirely unrelated to its intended goal of mitigating hallucinations. Our findings challenge common assumptions about the effectiveness of contrastive decoding strategies and pave the way for developing genuinely effective solutions to hallucinations in MLLMs.
>
---
#### [replaced 049] Self-Evolving Vision-Language Models for Image Quality Assessment via Voting and Ranking
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.25787v3](http://arxiv.org/pdf/2509.25787v3)**

> **作者:** Wen Wen; Tianwu Zhi; Kanglong Fan; Yang Li; Xinge Peng; Yabin Zhang; Yiting Liao; Junlin Li; Li Zhang
>
> **备注:** Technical Report
>
> **摘要:** Improving vision-language models (VLMs) in the post-training stage typically relies on supervised fine-tuning or reinforcement learning, methods that necessitate costly, human-annotated data. While self-supervised techniques such as self-consistency have proven effective for enhancing reasoning capabilities, their application to perceptual domains such as image quality assessment (IQA) remains largely unexplored. In this work, we introduce EvoQuality, a novel framework that enables a VLM to autonomously refine its quality perception capabilities without any ground-truth labels. EvoQuality adapts the principle of self-consistency to the ranking-based nature of IQA. It generates pseudo-labels by performing pairwise majority voting on the VLM's own outputs to establish a consensus on relative quality. These pseudo-rankings are then formulated into a fidelity reward that guides the model's iterative evolution through group relative policy optimization (GRPO). By iteratively leveraging its own predictions, EvoQuality progressively refines the VLM's perceptual capability. Extensive experiments show that EvoQuality boosts the base VLM's zero-shot performance by 31.8\% on PLCC across diverse IQA benchmarks. Remarkably, despite being entirely self-supervised, EvoQuality achieves performance that is competitive with, or even surpasses, state-of-the-art supervised VLM-based IQA models, outperforming these models on 5 out of 7 IQA benchmarks.
>
---
#### [replaced 050] Anchors Aweigh! Sail for Optimal Unified Multi-Modal Representations
- **分类: cs.LG; cs.CV; stat.ML**

- **链接: [http://arxiv.org/pdf/2410.02086v3](http://arxiv.org/pdf/2410.02086v3)**

> **作者:** Minoh Jeong; Zae Myung Kim; Min Namgung; Dongyeop Kang; Yao-Yi Chiang; Alfred Hero
>
> **摘要:** A unified representation space in multi-modal learning is essential for effectively integrating diverse data sources, such as text, images, and audio, to enhance efficiency and performance across various downstream tasks. Recent binding methods, such as ImageBind, typically rely on a single, fixed anchor modality for aligning multi-modal data. We mathematically analyze these fixed anchor binding methods and uncover significant limitations: (1) over-reliance on the choice of the anchor modality, (2) inadequate capture of intra-modal information, and (3) failure to account for cross-modal correlation among non-anchored modalities. To address these issues, we propose the need for adaptive anchor binding methods, exemplified by our framework CentroBind. The proposed method uses adaptively adjustable centroid-based anchors generated from all available modalities, leading to a balanced and rich representation space. We theoretically demonstrate that our approach captures three critical properties of multi-modal learning -- intra-modal learning, inter-modal learning, and multi-modal alignment -- while constructing a unified representation that spans all modalities. Experiments on both synthetic and real-world datasets show that adaptive anchor methods such as CentroBind consistently outperform fixed anchor binding methods, verifying our analysis.
>
---
#### [replaced 051] Submillimeter-Accurate 3D Lumbar Spine Reconstruction from Biplanar X-Ray Images: Incorporating a Multi-Task Network and Landmark-Weighted Loss
- **分类: eess.IV; cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2503.14573v3](http://arxiv.org/pdf/2503.14573v3)**

> **作者:** Wanxin Yu; Zhemin Zhu; Cong Wang; Yihang Bao; Chunjie Xia; Rongshan Cheng; Yan Yu; Tsung-Yuan Tsai
>
> **备注:** 27 pages, 16 figures, 9 tables
>
> **摘要:** To meet the clinical demand for accurate 3D lumbar spine assessment in a weight-bearing position, this study presents a novel, fully automatic framework for high-precision 3D reconstruction from biplanar X-ray images, overcoming the limitations of existing methods. The core of this method involves a novel multi-task deep learning network that simultaneously performs lumbar decomposition and landmark detection on the original biplanar radiographs. The decomposition effectively eliminates interference from surrounding tissues, simplifying subsequent image registration, while the landmark detection provides an initial pose estimation for the Statistical Shape Model (SSM), enhancing the efficiency and robustness of the registration process. Building on this, we introduce a landmark-weighted 2D-3D registration strategy. By assigning higher weights to complex posterior structures like the transverse and spinous processes during optimization, this strategy significantly enhances the reconstruction accuracy of the posterior arch. Our method was validated against a gold standard derived from registering CT segmentations to the biplanar X-rays. It sets a new benchmark by achieving sub-millimeter accuracy and completes the full reconstruction and measurement workflow in under 20 seconds, establishing a state-of-the-art combination of precision and speed. This fast and low-dose pipeline provides a powerful automated tool for diagnosing lumbar conditions such as spondylolisthesis and scoliosis in their functional, weight-bearing state.
>
---
#### [replaced 052] AutoEdit: Automatic Hyperparameter Tuning for Image Editing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.15031v2](http://arxiv.org/pdf/2509.15031v2)**

> **作者:** Chau Pham; Quan Dao; Mahesh Bhosale; Yunjie Tian; Dimitris Metaxas; David Doermann
>
> **备注:** Provided code link
>
> **摘要:** Recent advances in diffusion models have revolutionized text-guided image editing, yet existing editing methods face critical challenges in hyperparameter identification. To get the reasonable editing performance, these methods often require the user to brute-force tune multiple interdependent hyperparameters, such as inversion timesteps and attention modification. This process incurs high computational costs due to the huge hyperparameter search space. We consider searching optimal editing's hyperparameters as a sequential decision-making task within the diffusion denoising process. Specifically, we propose a reinforcement learning framework, which establishes a Markov Decision Process that dynamically adjusts hyperparameters across denoising steps, integrating editing objectives into a reward function. The method achieves time efficiency through proximal policy optimization while maintaining optimal hyperparameter configurations. Experiments demonstrate significant reduction in search time and computational overhead compared to existing brute-force approaches, advancing the practical deployment of a diffusion-based image editing framework in the real world. Codes can be found at https://github.com/chaupham1709/AutoEdit.git.
>
---
#### [replaced 053] Imagining the Unseen: Generative Location Modeling for Object Placement
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.13564v2](http://arxiv.org/pdf/2410.13564v2)**

> **作者:** Jooyeol Yun; Davide Abati; Mohamed Omran; Jaegul Choo; Amirhossein Habibian; Auke Wiggers
>
> **备注:** Accepted by ICCV 2025 DRL4Real Workshop
>
> **摘要:** Location modeling, or determining where non-existing objects could feasibly appear in a scene, has the potential to benefit numerous computer vision tasks, from automatic object insertion to scene creation in virtual reality. Yet, this capability remains largely unexplored to date. In this paper, we develop a generative location model that, given an object class and an image, learns to predict plausible bounding boxes for such an object. Our approach first tokenizes the image and target object class, then decodes bounding box coordinates through an autoregressive transformer. This formulation effectively addresses two core challenges in locatio modeling: the inherent one-to-many nature of plausible locations, and the sparsity of existing location modeling datasets, where fewer than 1% of valid placements are labeled. Furthermore, we incorporate Direct Preference Optimization to leverage negative labels, refining the spatial predictions. Empirical evaluations reveal that our generative location model achieves superior placement accuracy on the OPA dataset as compared to discriminative baselines and image composition approaches. We further test our model in the context of object insertion, where it proposes locations for an off-the-shelf inpainting model to render objects. In this respect, our proposal exhibits improved visual coherence relative to state-of-the-art instruction-tuned editing methods, demonstrating a high-performing location model's utility in a downstream application.
>
---
#### [replaced 054] Integrating Feature Selection and Machine Learning for Nitrogen Assessment in Grapevine Leaves using In-Field Hyperspectral Imaging
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.17869v2](http://arxiv.org/pdf/2507.17869v2)**

> **作者:** Atif Bilal Asad; Achyut Paudel; Safal Kshetri; Chenchen Kang; Salik Ram Khanal; Nataliya Shcherbatyuk; Pierre Davadant; R. Paul Schreiner; Santosh Kalauni; Manoj Karkee; Markus Keller
>
> **备注:** Major Revision
>
> **摘要:** Nitrogen (N) is one of the most crucial nutrients in vineyards, affecting plant growth and subsequent products such as wine and juice. Because soil N has high spatial and temporal variability, it is desirable to accurately estimate the N concentration of grapevine leaves and manage fertilization at the individual plant level to optimally meet plant needs. In this study, we used in-field hyperspectral images with wavelengths ranging from $400 to 1000nm of four different grapevine cultivars collected from distinct vineyards and over two growth stages during two growing seasons to develop models for predicting N concentration at the leaf-level and canopy-level. After image processing, two feature selection methods were employed to identify the optimal set of spectral bands that were responsive to leaf N concentrations. The selected spectral bands were used to train and test two different Machine Learning (ML) models, Gradient Boosting and XGBoost, for predicting nitrogen concentrations. The comparison of selected bands for both leaf-level and canopy-level datasets showed that most of the spectral regions identified by the feature selection methods were across both methods and the dataset types (leaf- and canopy-level datasets), particularly in the key regions, 500-525nm, 650-690nm, 750-800nm, and 900-950nm. These findings indicated the robustness of these spectral regions for predicting nitrogen content. The results for N prediction demonstrated that the ML model achieved an R square of 0.49 for canopy-level data and an R square of 0.57 for leaf-level data, despite using different sets of selected spectral bands for each analysis level. The study demonstrated the potential of using in-field hyperspectral imaging and the use of spectral data in integrated feature selection and ML techniques to monitor N status in vineyards.
>
---
#### [replaced 055] RICO: Two Realistic Benchmarks and an In-Depth Analysis for Incremental Learning in Object Detection
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2508.13878v2](http://arxiv.org/pdf/2508.13878v2)**

> **作者:** Matthias Neuwirth-Trapp; Maarten Bieshaar; Danda Pani Paudel; Luc Van Gool
>
> **备注:** Accepted to ICCV Workshops 2025; v2: add GitHub link and update affiliation
>
> **摘要:** Incremental Learning (IL) trains models sequentially on new data without full retraining, offering privacy, efficiency, and scalability. IL must balance adaptability to new data with retention of old knowledge. However, evaluations often rely on synthetic, simplified benchmarks, obscuring real-world IL performance. To address this, we introduce two Realistic Incremental Object Detection Benchmarks (RICO): Domain RICO (D-RICO) features domain shifts with a fixed class set, and Expanding-Classes RICO (EC-RICO) integrates new domains and classes per IL step. Built from 14 diverse datasets covering real and synthetic domains, varying conditions (e.g., weather, time of day), camera sensors, perspectives, and labeling policies, both benchmarks capture challenges absent in existing evaluations. Our experiments show that all IL methods underperform in adaptability and retention, while replaying a small amount of previous data already outperforms all methods. However, individual training on the data remains superior. We heuristically attribute this gap to weak teachers in distillation, single models' inability to manage diverse tasks, and insufficient plasticity. Our code will be made publicly available.
>
---
#### [replaced 056] ImageNet-trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.20234v2](http://arxiv.org/pdf/2509.20234v2)**

> **作者:** Tom Burgert; Oliver Stoll; Paolo Rota; Begüm Demir
>
> **备注:** Accepted at NeurIPS 2025 (oral)
>
> **摘要:** The hypothesis that Convolutional Neural Networks (CNNs) are inherently texture-biased has shaped much of the discourse on feature use in deep learning. We revisit this hypothesis by examining limitations in the cue-conflict experiment by Geirhos et al. To address these limitations, we propose a domain-agnostic framework that quantifies feature reliance through systematic suppression of shape, texture, and color cues, avoiding the confounds of forced-choice conflicts. By evaluating humans and neural networks under controlled suppression conditions, we find that CNNs are not inherently texture-biased but predominantly rely on local shape features. Nonetheless, this reliance can be substantially mitigated through modern training strategies or architectures (ConvNeXt, ViTs). We further extend the analysis across computer vision, medical imaging, and remote sensing, revealing that reliance patterns differ systematically: computer vision models prioritize shape, medical imaging models emphasize color, and remote sensing models exhibit a stronger reliance on texture. Code is available at https://github.com/tomburgert/feature-reliance.
>
---
#### [replaced 057] HBSplat: Robust Sparse-View Gaussian Reconstruction with Hybrid-Loss Guided Depth and Bidirectional Warping
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.24893v2](http://arxiv.org/pdf/2509.24893v2)**

> **作者:** Yu Ma; Guoliang Wei; Yue Cheng
>
> **备注:** 14 pages, 21 figures
>
> **摘要:** Novel View Synthesis (NVS) from sparse views presents a formidable challenge in 3D reconstruction, where limited multi-view constraints lead to severe overfitting, geometric distortion, and fragmented scenes. While 3D Gaussian Splatting (3DGS) delivers real-time, high-fidelity rendering, its performance drastically deteriorates under sparse inputs, plagued by floating artifacts and structural failures. To address these challenges, we introduce HBSplat, a unified framework that elevates 3DGS by seamlessly integrating robust structural cues, virtual view constraints, and occluded region completion. Our core contributions are threefold: a Hybrid-Loss Depth Estimation module that ensures multi-view consistency by leveraging dense matching priors and integrating reprojection, point propagation, and smoothness constraints; a Bidirectional Warping Virtual View Synthesis method that enforces substantially stronger constraints by creating high-fidelity virtual views through bidirectional depth-image warping and multi-view fusion; and an Occlusion-Aware Reconstruction component that recovers occluded areas using a depth-difference mask and a learning-based inpainting model. Extensive evaluations on LLFF, Blender, and DTU benchmarks validate that HBSplat sets a new state-of-the-art, achieving up to 21.13 dB PSNR and 0.189 LPIPS, while maintaining real-time inference. Code is available at: https://github.com/eternalland/HBSplat.
>
---
#### [replaced 058] Think Before You Diffuse: Infusing Physical Rules into Video Diffusion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.21653v3](http://arxiv.org/pdf/2505.21653v3)**

> **作者:** Ke Zhang; Cihan Xiao; Jiacong Xu; Yiqun Mei; Vishal M. Patel
>
> **备注:** 19 pages, 8 figures
>
> **摘要:** Recent video diffusion models have demonstrated their great capability in generating visually-pleasing results, while synthesizing the correct physical effects in generated videos remains challenging. The complexity of real-world motions, interactions, and dynamics introduce great difficulties when learning physics from data. In this work, we propose DiffPhy, a generic framework that enables physically-correct and photo-realistic video generation by fine-tuning a pre-trained video diffusion model. Our method leverages large language models (LLMs) to infer rich physical context from the text prompt. To incorporate this context into the video diffusion model, we use a multimodal large language model (MLLM) to verify intermediate latent variables against the inferred physical rules, guiding the gradient updates of model accordingly. Textual output of LLM is transformed into continuous signals. We then formulate a set of training objectives that jointly ensure physical accuracy and semantic alignment with the input text. Additionally, failure facts of physical phenomena are corrected via attention injection. We also establish a high-quality physical video dataset containing diverse phyiscal actions and events to facilitate effective finetuning. Extensive experiments on public benchmarks demonstrate that DiffPhy is able to produce state-of-the-art results across diverse physics-related scenarios. Our project page is available at https://bwgzk-keke.github.io/DiffPhy/.
>
---
#### [replaced 059] A weakly-supervised deep learning model for fast localisation and delineation of the skeleton, internal organs, and spinal canal on Whole-Body Diffusion-Weighted MRI (WB-DWI)
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.20722v2](http://arxiv.org/pdf/2503.20722v2)**

> **作者:** A. Candito; A. Dragan; R. Holbrey; A. Ribeiro; R. Donners; C. Messiou; N. Tunariu; D. -M. Koh; M. D. Blackledge
>
> **摘要:** Background: Apparent Diffusion Coefficient (ADC) values and Total Diffusion Volume (TDV) from Whole-body diffusion-weighted MRI (WB-DWI) are recognized cancer imaging biomarkers. However, manual disease delineation for ADC and TDV measurements is unfeasible in clinical practice, demanding automation. As a first step, we propose an algorithm to generate fast and reproducible probability maps of the skeleton, adjacent internal organs (liver, spleen, urinary bladder, and kidneys), and spinal canal. Methods: We developed an automated deep-learning pipeline based on a 3D patch-based Residual U-Net architecture that localises and delineates these anatomical structures on WB-DWI. The algorithm was trained using "soft labels" (non-binary segmentations) derived from a computationally intensive atlas-based approach. For training and validation, we employed a multi-centre WB-DWI dataset comprising 532 scans from patients with Advanced Prostate Cancer (APC) or Multiple Myeloma (MM), with testing on 45 patients. Results: Our weakly-supervised deep learning model achieved an average dice score of 0.67 for whole skeletal delineation, 0.76 when excluding ribcage, 0.83 for internal organs, and 0.86 for spinal canal, with average surface distances below 3mm. Relative median ADC differences between automated and manual full-body delineations were below 10%. The model was 12x faster than the atlas-based registration algorithm (25 sec vs. 5 min). Two experienced radiologists rated the model's outputs as either "good" or "excellent" on test scans, with inter-reader agreement from fair to substantial (Gwet's AC1 = 0.27-0.72). Conclusion: The model offers fast, reproducible probability maps for localising and delineating body regions on WB-DWI, potentially enabling non-invasive imaging biomarker quantification to support disease staging and treatment response assessment.
>
---
#### [replaced 060] Uni-Instruct: One-step Diffusion Model through Unified Diffusion Divergence Instruction
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.20755v3](http://arxiv.org/pdf/2505.20755v3)**

> **作者:** Yifei Wang; Weimin Bai; Colin Zhang; Debing Zhang; Weijian Luo; He Sun
>
> **摘要:** In this paper, we unify more than 10 existing one-step diffusion distillation approaches, such as Diff-Instruct, DMD, SIM, SiD, $f$-distill, etc, inside a theory-driven framework which we name the \textbf{\emph{Uni-Instruct}}. Uni-Instruct is motivated by our proposed diffusion expansion theory of the $f$-divergence family. Then we introduce key theories that overcome the intractability issue of the original expanded $f$-divergence, resulting in an equivalent yet tractable loss that effectively trains one-step diffusion models by minimizing the expanded $f$-divergence family. The novel unification introduced by Uni-Instruct not only offers new theoretical contributions that help understand existing approaches from a high-level perspective but also leads to state-of-the-art one-step diffusion generation performances. On the CIFAR10 generation benchmark, Uni-Instruct achieves record-breaking Frechet Inception Distance (FID) values of \textbf{\emph{1.46}} for unconditional generation and \textbf{\emph{1.38}} for conditional generation. On the ImageNet-$64\times 64$ generation benchmark, Uni-Instruct achieves a new SoTA one-step generation FID of \textbf{\emph{1.02}}, which outperforms its 79-step teacher diffusion with a significant improvement margin of 1.33 (1.02 vs 2.35). We also apply Uni-Instruct on broader tasks like text-to-3D generation. For text-to-3D generation, Uni-Instruct gives decent results, which slightly outperforms previous methods, such as SDS and VSD, in terms of both generation quality and diversity. Both the solid theoretical and empirical contributions of Uni-Instruct will potentially help future studies on one-step diffusion distillation and knowledge transferring of diffusion models.
>
---
#### [replaced 061] VisRet: Visualization Improves Knowledge-Intensive Text-to-Image Retrieval
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20291v2](http://arxiv.org/pdf/2505.20291v2)**

> **作者:** Di Wu; Yixin Wan; Kai-Wei Chang
>
> **摘要:** Text-to-image retrieval (T2I retrieval) remains challenging because cross-modal embeddings often behave as bags of concepts and underrepresent structured visual relationships such as pose and viewpoint. We propose Visualize-then-Retrieve (VisRet), a new paradigm for T2I retrieval that mitigates this limitation of cross-modal similarity alignment. VisRet first projects textual queries into the image modality via T2I generation. Then, it performs retrieval within the image modality to bypass the weaknesses of cross-modal retrievers in recognizing subtle visual-spatial features. Across four benchmarks (Visual-RAG, INQUIRE-Rerank, Microsoft COCO, and our new Visual-RAG-ME featuring multi-entity comparisons), VisRet substantially outperforms cross-modal similarity matching and baselines that recast T2I retrieval as text-to-text similarity matching, improving nDCG@30 by 0.125 on average with CLIP as the retriever and by 0.121 with E5-V. For downstream question answering, VisRet increases accuracy on Visual-RAG and Visual-RAG-ME by 3.8% and 15.7% in top-1 retrieval, and by 3.9% and 11.1% in top-10 retrieval. Ablation studies show compatibility with different T2I instruction LLMs, T2I generation models, and downstream LLMs. VisRet provides a practical and principled path that energizes further advances in vision-language retrieval. Our code and the Visual-RAG-ME benchmark will be publicly released.
>
---
#### [replaced 062] When Semantics Mislead Vision: Mitigating Large Multimodal Models Hallucinations in Scene Text Spotting and Understanding
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2506.05551v2](http://arxiv.org/pdf/2506.05551v2)**

> **作者:** Yan Shu; Hangui Lin; Yexin Liu; Yan Zhang; Gangyan Zeng; Yan Li; Yu Zhou; Ser-Nam Lim; Harry Yang; Nicu Sebe
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Large Multimodal Models (LMMs) have achieved impressive progress in visual perception and reasoning. However, when confronted with visually ambiguous or non-semantic scene text, they often struggle to accurately spot and understand the content, frequently generating semantically plausible yet visually incorrect answers, which we refer to as semantic hallucination. In this work, we investigate the underlying causes of semantic hallucination and identify a key finding: Transformer layers in LLM with stronger attention focus on scene text regions are less prone to producing semantic hallucinations. Thus, we propose a training-free semantic hallucination mitigation framework comprising two key components: (1) ZoomText, a coarse-to-fine strategy that identifies potential text regions without external detectors; and (2) Grounded Layer Correction, which adaptively leverages the internal representations from layers less prone to hallucination to guide decoding, correcting hallucinated outputs for non-semantic samples while preserving the semantics of meaningful ones. To enable rigorous evaluation, we introduce TextHalu-Bench, a benchmark of 1,740 samples spanning both semantic and non-semantic cases, with manually curated question answer pairs designed to probe model hallucinations. Extensive experiments demonstrate that our method not only effectively mitigates semantic hallucination but also achieves strong performance on public benchmarks for scene text spotting and understanding.
>
---
#### [replaced 063] Tables Guide Vision: Learning to See the Heart through Tabular Data
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.14998v2](http://arxiv.org/pdf/2503.14998v2)**

> **作者:** Marta Hasny; Maxime Di Folco; Keno Bressem; Julia Schnabel
>
> **摘要:** Contrastive learning methods in computer vision typically rely on augmented views of the same image or multimodal pretraining strategies that align paired modalities. However, these approaches often overlook semantic relationships between distinct instances, leading to false negatives when semantically similar samples are treated as negatives. This limitation is especially critical in medical imaging domains such as cardiology, where demographic and clinical attributes play a critical role in assessing disease risk and patient outcomes. We introduce a tabular-guided contrastive learning framework that leverages clinically relevant tabular data to identify patient-level similarities and construct more meaningful pairs, enabling semantically aligned representation learning without requiring joint embeddings across modalities. Additionally, we adapt the k-NN algorithm for zero-shot prediction to overcome the lack of zero-shot capability in unimodal representations. We demonstrate the strength of our methods using a large cohort of short-axis cardiac MR images and clinical attributes, where tabular data helps to more effectively distinguish between patient subgroups. Evaluation on downstream tasks, including fine-tuning, linear probing, and zero-shot prediction of cardiovascular artery diseases and cardiac phenotypes, shows that incorporating tabular data guidance yields stronger visual representations than conventional methods that rely solely on image augmentation or combined image-tabular embeddings. Further, we show that our method can generalize to natural images by evaluating it on a car advertisement dataset. The code will be available on GitHub upon acceptance.
>
---
#### [replaced 064] ECORE: Energy-Conscious Optimized Routing for Deep Learning Models at the Edge
- **分类: cs.DC; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.06011v3](http://arxiv.org/pdf/2507.06011v3)**

> **作者:** Daghash K. Alqahtani; Maria A. Rodriguez; Muhammad Aamir Cheema; Hamid Rezatofighi; Adel N. Toosi
>
> **摘要:** Edge computing enables data processing closer to the source, significantly reducing latency, an essential requirement for real-time vision-based analytics such as object detection in surveillance and smart city environments. However, these tasks place substantial demands on resource-constrained edge devices, making the joint optimization of energy consumption and detection accuracy critical. To address this challenge, we propose ECORE, a framework that integrates multiple dynamic routing strategies, including a novel estimation-based techniques and an innovative greedy selection algorithm, to direct image processing requests to the most suitable edge device-model pair. ECORE dynamically balances energy efficiency and detection performance based on object characteristics. We evaluate our framework through extensive experiments on real-world datasets, comparing against widely used baseline techniques. The evaluation leverages established object detection models (YOLO, SSD, EfficientDet) and diverse edge platforms, including Jetson Orin Nano, Raspberry Pi 4 and 5, and TPU accelerators. Results demonstrate that our proposed context-aware routing strategies can reduce energy consumption and latency by 35% and 49%, respectively, while incurring only a 2% loss in detection accuracy compared to accuracy-centric methods.
>
---
#### [replaced 065] Self-Supervised Representation Learning with Joint Embedding Predictive Architecture for Automotive LiDAR Object Detection
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.04969v2](http://arxiv.org/pdf/2501.04969v2)**

> **作者:** Haoran Zhu; Zhenyuan Dong; Kristi Topollai; Beiyao Sha; Anna Choromanska
>
> **摘要:** Recently, self-supervised representation learning relying on vast amounts of unlabeled data has been explored as a pre-training method for autonomous driving. However, directly applying popular contrastive or generative methods to this problem is insufficient and may even lead to negative transfer. In this paper, we present AD-L-JEPA, a novel self-supervised pre-training framework with a joint embedding predictive architecture (JEPA) for automotive LiDAR object detection. Unlike existing methods, AD-L-JEPA is neither generative nor contrastive. Instead of explicitly generating masked regions, our method predicts Bird's-Eye-View embeddings to capture the diverse nature of driving scenes. Furthermore, our approach eliminates the need to manually form contrastive pairs by employing explicit variance regularization to avoid representation collapse. Experimental results demonstrate consistent improvements on the LiDAR 3D object detection downstream task across the KITTI3D, Waymo, and ONCE datasets, while reducing GPU hours by 1.9x-2.7x and GPU memory by 2.8x-4x compared with the state-of-the-art method Occupancy-MAE. Notably, on the largest ONCE dataset, pre-training on 100K frames yields a 1.61 mAP gain, better than all other methods pre-trained on either 100K or 500K frames, and pre-training on 500K frames yields a 2.98 mAP gain, better than all other methods pre-trained on either 500K or 1M frames. AD-L-JEPA constitutes the first JEPA-based pre-training method for autonomous driving. It offers better quality, faster, and more GPU-memory-efficient self-supervised representation learning. The source code of AD-L-JEPA is ready to be released.
>
---
#### [replaced 066] Leveraging Foundation Models for Multimodal Graph-Based Action Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15192v2](http://arxiv.org/pdf/2505.15192v2)**

> **作者:** Fatemeh Ziaeetabar; Florentin Wörgötter
>
> **摘要:** Foundation models have ushered in a new era for multimodal video understanding by enabling the extraction of rich spatiotemporal and semantic representations. In this work, we introduce a novel graph-based framework that integrates a vision-language foundation, leveraging VideoMAE for dynamic visual encoding and BERT for contextual textual embedding, to address the challenge of recognizing fine-grained bimanual manipulation actions. Departing from conventional static graph architectures, our approach constructs an adaptive multimodal graph where nodes represent frames, objects, and textual annotations, and edges encode spatial, temporal, and semantic relationships. These graph structures evolve dynamically based on learned interactions, allowing for flexible and context-aware reasoning. A task-specific attention mechanism within a Graph Attention Network further enhances this reasoning by modulating edge importance based on action semantics. Through extensive evaluations on diverse benchmark datasets, we demonstrate that our method consistently outperforms state-of-the-art baselines, underscoring the strength of combining foundation models with dynamic graph-based reasoning for robust and generalizable action recognition.
>
---
#### [replaced 067] HiMat: DiT-based Ultra-High Resolution SVBRDF Generation
- **分类: cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2508.07011v4](http://arxiv.org/pdf/2508.07011v4)**

> **作者:** Zixiong Wang; Jian Yang; Yiwei Hu; Milos Hasan; Beibei Wang
>
> **摘要:** Creating ultra-high-resolution spatially varying bidirectional reflectance functions (SVBRDFs) is critical for photorealistic 3D content creation, to faithfully represent fine-scale surface details required for close-up rendering. However, achieving 4K generation faces two key challenges: (1) the need to synthesize multiple reflectance maps at full resolution, which multiplies the pixel budget and imposes prohibitive memory and computational cost, and (2) the requirement to maintain strong pixel-level alignment across maps at 4K, which is particularly difficult when adapting pretrained models designed for the RGB image domain. We introduce HiMat, a diffusion-based framework tailored for efficient and diverse 4K SVBRDF generation. To address the first challenge, HiMat performs generation in a high-compression latent space via DC-AE, and employs a pretrained diffusion transformer with linear attention to improve per-map efficiency. To address the second challenge, we propose CrossStitch, a lightweight convolutional module that enforces cross-map consistency without incurring the cost of global attention. Our experiments show that HiMat achieves high-fidelity 4K SVBRDF generation with superior efficiency, structural consistency, and diversity compared to prior methods. Beyond materials, our framework also generalizes to related applications such as intrinsic decomposition.
>
---
#### [replaced 068] v1: Learning to Point Visual Tokens for Multimodal Grounded Reasoning
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.18842v4](http://arxiv.org/pdf/2505.18842v4)**

> **作者:** Jiwan Chung; Junhyeok Kim; Siyeol Kim; Jaeyoung Lee; Min Soo Kim; Youngjae Yu
>
> **摘要:** When thinking with images, humans rarely rely on a single glance: they revisit visual information repeatedly during reasoning. However, existing models typically process images only once and thereafter generate reasoning entirely in text, lacking mechanisms to re-access or ground inference in visual representations. We empirically confirm this: as reasoning chains lengthen, models progressively lose focus on relevant regions. In response, we introduce v1, a lightweight extension that enables active visual referencing through a simple point-and-copy approach. This allows the model to identify relevant image patches and copy their embeddings back into the reasoning stream, ensuring that evolving hypotheses remain grounded in perceptual evidence. Crucially, our pointing strategy lets the MLLM directly select image patches using their semantic representations as keys, keeping perceptual evidence embedded in the same space as the model's reasoning. To train this capability, we construct v1g, a dataset of 300K multimodal reasoning traces with interleaved visual grounding annotations. Across various multimodal mathematical reasoning benchmarks, v1 consistently outperforms comparable baselines, establishing point-and-copy as a practical mechanism for grounded reasoning. The model checkpoint and dataset are available at github.com/jun297/v1.
>
---
#### [replaced 069] Deep Learning Approaches with Explainable AI for Differentiating Alzheimer Disease and Mild Cognitive Impairment
- **分类: eess.IV; cs.AI; cs.CV; cs.LG; stat.AP; stat.ML**

- **链接: [http://arxiv.org/pdf/2510.00048v2](http://arxiv.org/pdf/2510.00048v2)**

> **作者:** Fahad Mostafa; Kannon Hossain; Hafiz Khan
>
> **备注:** 18 pages, 4 figures
>
> **摘要:** Early and accurate diagnosis of Alzheimer Disease is critical for effective clinical intervention, particularly in distinguishing it from Mild Cognitive Impairment, a prodromal stage marked by subtle structural changes. In this study, we propose a hybrid deep learning ensemble framework for Alzheimer Disease classification using structural magnetic resonance imaging. Gray and white matter slices are used as inputs to three pretrained convolutional neural networks such as ResNet50, NASNet, and MobileNet, each fine tuned through an end to end process. To further enhance performance, we incorporate a stacked ensemble learning strategy with a meta learner and weighted averaging to optimally combine the base models. Evaluated on the Alzheimer Disease Neuroimaging Initiative dataset, the proposed method achieves state of the art accuracy of 99.21% for Alzheimer Disease vs. Mild Cognitive Impairment and 91.0% for Mild Cognitive Impairment vs. Normal Controls, outperforming conventional transfer learning and baseline ensemble methods. To improve interpretability in image based diagnostics, we integrate Explainable AI techniques by Gradient weighted Class Activation, which generates heatmaps and attribution maps that highlight critical regions in gray and white matter slices, revealing structural biomarkers that influence model decisions. These results highlight the frameworks potential for robust and scalable clinical decision support in neurodegenerative disease diagnostics.
>
---
#### [replaced 070] Inference-Time Text-to-Video Alignment with Diffusion Latent Beam Search
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.19252v3](http://arxiv.org/pdf/2501.19252v3)**

> **作者:** Yuta Oshima; Masahiro Suzuki; Yutaka Matsuo; Hiroki Furuta
>
> **备注:** Accepted to NeurIPS2025. Website: https://sites.google.com/view/t2v-dlbs and Code: https://github.com/shim0114/T2V-Diffusion-Search
>
> **摘要:** The remarkable progress in text-to-video diffusion models enables the generation of photorealistic videos, although the content of these generated videos often includes unnatural movement or deformation, reverse playback, and motionless scenes. Recently, an alignment problem has attracted huge attention, where we steer the output of diffusion models based on some measure of the content's goodness. Because there is a large room for improvement of perceptual quality along the frame direction, we should address which metrics we should optimize and how we can optimize them in the video generation. In this paper, we propose diffusion latent beam search with lookahead estimator, which can select a better diffusion latent to maximize a given alignment reward at inference time. We then point out that improving perceptual video quality with respect to alignment to prompts requires reward calibration by weighting existing metrics. This is because when humans or vision language models evaluate outputs, many previous metrics to quantify the naturalness of video do not always correlate with the evaluation. We demonstrate that our method improves the perceptual quality evaluated on the calibrated reward, VLMs, and human assessment, without model parameter update, and outputs the best generation compared to greedy search and best-of-N sampling under much more efficient computational cost. The experiments highlight that our method is beneficial to many capable generative models, and provide a practical guideline: we should prioritize the inference-time compute allocation into enabling the lookahead estimator and increasing the search budget, rather than expanding the denoising steps.
>
---
#### [replaced 071] Optimal Transport for Brain-Image Alignment: Unveiling Redundancy and Synergy in Neural Information Processing
- **分类: q-bio.NC; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.10663v2](http://arxiv.org/pdf/2503.10663v2)**

> **作者:** Yang Xiao; Wang Lu; Jie Ji; Ruimeng Ye; Gen Li; Xiaolong Ma; Bo Hui
>
> **备注:** 14pages
>
> **摘要:** The design of artificial neural networks (ANNs) is inspired by the structure of the human brain, and in turn, ANNs offer a potential means to interpret and understand brain signals. Existing methods primarily align brain signals with stimulus signals using Mean Squared Error (MSE), which focuses only on local point-wise alignment and ignores global matching, leading to coarse interpretations and inaccuracies in brain signal decoding. In this paper, we address these issues through optimal transport (OT) and theoretically demonstrate why OT provides a more effective alignment strategy than MSE. Specifically, we construct a transport plan between brain voxel embeddings and image embeddings, enabling more precise matching. By controlling the amount of transport, we mitigate the influence of redundant information. We apply our alignment model directly to the Brain Captioning task by feeding brain signals into a large language model (LLM) instead of images. Our approach achieves state-of-the-art performance across ten evaluation metrics, surpassing the previous best method by an average of 6.11\% in single-subject training and 3.81\% in cross-subject training. Additionally, we have uncovered several insightful conclusions that align with existing brain research. We unveil the redundancy and synergy of brain information processing through region masking and data dimensionality reduction visualization experiments. We believe our approach paves the way for a more precise understanding of brain signals in the future. The code is available at https://github.com/NKUShaw/OT-Alignment4brain-to-image.
>
---
#### [replaced 072] High-pass filtered fidelity-imposed network edit (HP-FINE) for robust quantitative susceptibility mapping from high-pass filtered phase
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2305.03844v2](http://arxiv.org/pdf/2305.03844v2)**

> **作者:** Jinwei Zhang; Alexey Dimov; Chao Li; Hang Zhang; Thanh D. Nguyen; Pascal Spincemaille; Yi Wang
>
> **摘要:** Purpose: To improve the generalization ability of deep learning based predictions of quantitative susceptibility mapping (QSM) from high-pass filtered phase (HPFP) data. Methods: A network fine-tuning step called HP-FINE is proposed, which is based on the high-pass filtering forward model with low-frequency preservation regularization. Several comparisons were conducted: 1. HP-FINE with and without low-frequency regularization, 2. three 3D network architectures (Unet, Progressive Unet, and Big Unet), 3. two types of network output (recovered field and susceptibility), and 4. pre-training with and without the filtering augmentation. HPFP datasets with diverse high-pass filters, another acquisition voxel size, and prospective acquisition were used to assess the accuracy of QSM predictions. In the retrospective datasets, quantitative metrics (PSNR, SSIM, RMSE and HFEN) were used for evaluation. In the prospective dataset, statistics of ROI linear regression and Bland-Altman analysis were used for evaluation. Results: In the retrospective datasets, adding low-frequency regularization in HP-FINE substantially improved prediction accuracy compared to the pre-trained results, especially when combined with the filtering augmentation and recovered field output. In the prospective datasets, HP-FINE with low-frequency regularization and recovered field output demonstrated the preservation of ROI values, a result that was not achieved when using susceptibility as the output. Furthermore, Progressive Unet pre-trained with a combination of multiple losses outperformed both Unet and Progressive Unet pre-trained with a single loss in terms of preserving ROI values.
>
---
#### [replaced 073] Cat: Post-Training Quantization Error Reduction via Cluster-based Affine Transformation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2509.26277v2](http://arxiv.org/pdf/2509.26277v2)**

> **作者:** Ali Zoljodi; Radu Timofte; Masoud Daneshtalab
>
> **备注:** 29 pages, 20 figures
>
> **摘要:** Post-Training Quantization (PTQ) reduces the memory footprint and computational overhead of deep neural networks by converting full-precision (FP) values into quantized and compressed data types. While PTQ is more cost-efficient than Quantization-Aware Training (QAT), it is highly susceptible to accuracy degradation under a low-bit quantization (LQ) regime (e.g., 2-bit). Affine transformation is a classical technique used to reduce the discrepancy between the information processed by a quantized model and that processed by its full-precision counterpart; however, we find that using plain affine transformation, which applies a uniform affine parameter set for all outputs, worsens the results in low-bit PTQ. To address this, we propose Cluster-based Affine Transformation (CAT), an error-reduction framework that employs cluster-specific parameters to align LQ outputs with FP counterparts. CAT refines LQ outputs with only a negligible number of additional parameters, without requiring fine-tuning of the model or quantization parameters. We further introduce a novel PTQ framework integrated with CAT. Experiments on ImageNet-1K show that this framework consistently outperforms prior PTQ methods across diverse architectures and LQ settings, achieving up to 53.18% Top-1 accuracy on W2A2 ResNet-18. Moreover, CAT enhances existing PTQ baselines by more than 3% when used as a plug-in. We plan to release our implementation alongside the publication of this paper.
>
---
