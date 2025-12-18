# 计算机视觉 cs.CV

- **最新发布 109 篇**

- **更新 58 篇**

## 最新发布

#### [new 001] IMKD: Intensity-Aware Multi-Level Knowledge Distillation for Camera-Radar Fusion
- **分类: cs.CV; cs.LG**

- **简介: 该论文面向雷达-相机融合的3D目标检测任务，解决现有知识蒸馏方法破坏模态特性的缺陷。提出IMKD框架，通过三阶段强度感知的多级知识蒸馏，在不依赖LiDAR推理的前提下，保留各传感器特性并增强互补性，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.15581v1](https://arxiv.org/pdf/2512.15581v1)**

> **作者:** Shashank Mishra; Karan Patil; Didier Stricker; Jason Rambach
>
> **备注:** Accepted at IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026. 22 pages, 8 figures. Includes supplementary material
>
> **摘要:** High-performance Radar-Camera 3D object detection can be achieved by leveraging knowledge distillation without using LiDAR at inference time. However, existing distillation methods typically transfer modality-specific features directly to each sensor, which can distort their unique characteristics and degrade their individual strengths. To address this, we introduce IMKD, a radar-camera fusion framework based on multi-level knowledge distillation that preserves each sensor's intrinsic characteristics while amplifying their complementary strengths. IMKD applies a three-stage, intensity-aware distillation strategy to enrich the fused representation across the architecture: (1) LiDAR-to-Radar intensity-aware feature distillation to enhance radar representations with fine-grained structural cues, (2) LiDAR-to-Fused feature intensity-guided distillation to selectively highlight useful geometry and depth information at the fusion level, fostering complementarity between the modalities rather than forcing them to align, and (3) Camera-Radar intensity-guided fusion mechanism that facilitates effective feature alignment and calibration. Extensive experiments on the nuScenes benchmark show that IMKD reaches 67.0% NDS and 61.0% mAP, outperforming all prior distillation-based radar-camera fusion methods. Our code and models are available at https://github.com/dfki-av/IMKD/.
>
---
#### [new 002] IC-Effect: Precise and Efficient Video Effects Editing via In-Context Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出IC-Effect，面向视频VFX编辑任务，解决少样本下特效注入难、背景保真差、时序不一致等问题。工作包括：基于DiT的上下文学习框架、两阶段训练（通用适配+Effect-LoRA）、时空稀疏令牌化，并发布15种风格的配对数据集。**

- **链接: [https://arxiv.org/pdf/2512.15635v1](https://arxiv.org/pdf/2512.15635v1)**

> **作者:** Yuanhang Li; Yiren Song; Junzhe Bai; Xinran Liang; Hu Yang; Libiao Jin; Qi Mao
>
> **摘要:** We propose \textbf{IC-Effect}, an instruction-guided, DiT-based framework for few-shot video VFX editing that synthesizes complex effects (\eg flames, particles and cartoon characters) while strictly preserving spatial and temporal consistency. Video VFX editing is highly challenging because injected effects must blend seamlessly with the background, the background must remain entirely unchanged, and effect patterns must be learned efficiently from limited paired data. However, existing video editing models fail to satisfy these requirements. IC-Effect leverages the source video as clean contextual conditions, exploiting the contextual learning capability of DiT models to achieve precise background preservation and natural effect injection. A two-stage training strategy, consisting of general editing adaptation followed by effect-specific learning via Effect-LoRA, ensures strong instruction following and robust effect modeling. To further improve efficiency, we introduce spatiotemporal sparse tokenization, enabling high fidelity with substantially reduced computation. We also release a paired VFX editing dataset spanning $15$ high-quality visual styles. Extensive experiments show that IC-Effect delivers high-quality, controllable, and temporally consistent VFX editing, opening new possibilities for video creation.
>
---
#### [new 003] Improving VQA Reliability: A Dual-Assessment Approach with Self-Reflection and Cross-Model Verification
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向视觉问答（VQA）任务，旨在解决视觉语言模型（VLM）易幻觉、答案不可靠的问题。提出DAVR框架，融合自省评估与跨模型验证双路径，实现更准确实时的不确定性估计，显著提升VQA答案可信度。**

- **链接: [https://arxiv.org/pdf/2512.14770v1](https://arxiv.org/pdf/2512.14770v1)**

> **作者:** Xixian Wu; Yang Ou; Pengchao Tian; Zian Yang; Jielei Zhang; Peiyi Li; Longwen Gao
>
> **摘要:** Vision-language models (VLMs) have demonstrated significant potential in Visual Question Answering (VQA). However, the susceptibility of VLMs to hallucinations can lead to overconfident yet incorrect answers, severely undermining answer reliability. To address this, we propose Dual-Assessment for VLM Reliability (DAVR), a novel framework that integrates Self-Reflection and Cross-Model Verification for comprehensive uncertainty estimation. The DAVR framework features a dual-pathway architecture: one pathway leverages dual selector modules to assess response reliability by fusing VLM latent features with QA embeddings, while the other deploys external reference models for factual cross-checking to mitigate hallucinations. Evaluated in the Reliable VQA Challenge at ICCV-CLVL 2025, DAVR achieves a leading $Φ_{100}$ score of 39.64 and a 100-AUC of 97.22, securing first place and demonstrating its effectiveness in enhancing the trustworthiness of VLM responses.
>
---
#### [new 004] The LUMirage: An independent evaluation of zero-shot performance in the LUMIR challenge
- **分类: cs.CV; eess.IV**

- **简介: 该论文属医学图像配准任务，旨在检验深度学习方法在LUMIR挑战中宣称的零样本泛化能力。作者独立复现评估，发现模型在T1w数据上表现良好，但在跨模态（T2等）、高分辨率及不同预处理下性能显著下降，揭示其零样本能力被高估，强调需更贴近临床的实际评估协议。**

- **链接: [https://arxiv.org/pdf/2512.15505v1](https://arxiv.org/pdf/2512.15505v1)**

> **作者:** Rohit Jena; Pratik Chaudhari; James C. Gee
>
> **摘要:** The LUMIR challenge represents an important benchmark for evaluating deformable image registration methods on large-scale neuroimaging data. While the challenge demonstrates that modern deep learning methods achieve competitive accuracy on T1-weighted MRI, it also claims exceptional zero-shot generalization to unseen contrasts and resolutions, assertions that contradict established understanding of domain shift in deep learning. In this paper, we perform an independent re-evaluation of these zero-shot claims using rigorous evaluation protocols while addressing potential sources of instrumentation bias. Our findings reveal a more nuanced picture: (1) deep learning methods perform comparably to iterative optimization on in-distribution T1w images and even on human-adjacent species (macaque), demonstrating improved task understanding; (2) however, performance degrades significantly on out-of-distribution contrasts (T2, T2*, FLAIR), with Cohen's d scores ranging from 0.7-1.5, indicating substantial practical impact on downstream clinical workflows; (3) deep learning methods face scalability limitations on high-resolution data, failing to run on 0.6 mm isotropic images, while iterative methods benefit from increased resolution; and (4) deep methods exhibit high sensitivity to preprocessing choices. These results align with the well-established literature on domain shift and suggest that claims of universal zero-shot superiority require careful scrutiny. We advocate for evaluation protocols that reflect practical clinical and research workflows rather than conditions that may inadvertently favor particular method classes.
>
---
#### [new 005] In Pursuit of Pixel Supervision for Visual Pre-training
- **分类: cs.CV**

- **简介: 该论文属视觉自监督预训练任务，旨在探索像素级监督的有效性。针对现有方法多依赖latent空间的问题，提出Pixio模型——增强型掩码自编码器，通过更难的重建任务和强架构，在20亿图像上自监督训练，显著提升深度估计、3D重建等下游性能，证明像素空间学习是latent方法的有力替代与补充。**

- **链接: [https://arxiv.org/pdf/2512.15715v1](https://arxiv.org/pdf/2512.15715v1)**

> **作者:** Lihe Yang; Shang-Wen Li; Yang Li; Xinjie Lei; Dong Wang; Abdelrahman Mohamed; Hengshuang Zhao; Hu Xu
>
> **备注:** Project page: https://github.com/facebookresearch/pixio
>
> **摘要:** At the most basic level, pixels are the source of the visual information through which we perceive the world. Pixels contain information at all levels, ranging from low-level attributes to high-level concepts. Autoencoders represent a classical and long-standing paradigm for learning representations from pixels or other raw inputs. In this work, we demonstrate that autoencoder-based self-supervised learning remains competitive today and can produce strong representations for downstream tasks, while remaining simple, stable, and efficient. Our model, codenamed "Pixio", is an enhanced masked autoencoder (MAE) with more challenging pre-training tasks and more capable architectures. The model is trained on 2B web-crawled images with a self-curation strategy with minimal human curation. Pixio performs competitively across a wide range of downstream tasks in the wild, including monocular depth estimation (e.g., Depth Anything), feed-forward 3D reconstruction (i.e., MapAnything), semantic segmentation, and robot learning, outperforming or matching DINOv3 trained at similar scales. Our results suggest that pixel-space self-supervised learning can serve as a promising alternative and a complement to latent-space approaches.
>
---
#### [new 006] Towards Physically-Based Sky-Modeling For Image Based Lighting
- **分类: cs.CV; cs.GR**

- **简介: 该论文属图像生成与物理光照建模任务，旨在解决现有DNN天空模型无法兼顾光度真实性与全动态范围（22 f-stops）的问题。作者提出AllSky——首个直接从实拍HDR数据学习的全天气物理天空模型，支持用户控太阳/云位置，并建立新评估标准验证其不可替代性。**

- **链接: [https://arxiv.org/pdf/2512.15632v1](https://arxiv.org/pdf/2512.15632v1)**

> **作者:** Ian J. Maquignaz
>
> **摘要:** Accurate environment maps are a key component for rendering photorealistic outdoor scenes with coherent illumination. They enable captivating visual arts, immersive virtual reality, and a wide range of engineering and scientific applications. Recent works have extended sky-models to be more comprehensive and inclusive of cloud formations but, as we demonstrate, existing methods fall short in faithfully recreating natural skies. Though in recent years the visual quality of DNN-generated High Dynamic Range Imagery (HDRI) has greatly improved, the environment maps generated by DNN sky-models do not re-light scenes with the same tones, shadows, and illumination as physically captured HDR imagery. In this work, we demonstrate progress in HDR literature to be tangential to sky-modelling as current works cannot support both photorealism and the 22 f-stops required for the Full Dynamic Range (FDR) of outdoor illumination. We achieve this by proposing AllSky, a flexible all-weather sky-model learned directly from physically captured HDRI which we leverage to study the input modalities, tonemapping, conditioning, and evaluation of sky-models. Per user-controlled positioning of the sun and cloud formations, AllSky expands on current functionality by allowing for intuitive user control over environment maps and achieves state-of-the-art sky-model performance. Through our proposed evaluation, we demonstrate existing DNN sky-models are not interchangeable with physically captured HDRI or parametric sky-models, with current limitations being prohibitive of scalability and accurate illumination in downstream applications
>
---
#### [new 007] PMMD: A pose-guided multi-view multi-modal diffusion for person generation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属人物图像生成任务，旨在解决多视角下姿态控制难、服饰失真与遮挡等问题。提出PMMD扩散模型，融合多视角图像、姿态图和文本提示，设计多模态编码器、ResCVA模块及跨模态融合机制，提升身份一致性、细节与可控性。**

- **链接: [https://arxiv.org/pdf/2512.15069v1](https://arxiv.org/pdf/2512.15069v1)**

> **作者:** Ziyu Shang; Haoran Liu; Rongchao Zhang; Zhiqian Wei; Tongtong Feng
>
> **摘要:** Generating consistent human images with controllable pose and appearance is essential for applications in virtual try on, image editing, and digital human creation. Current methods often suffer from occlusions, garment style drift, and pose misalignment. We propose Pose-guided Multi-view Multimodal Diffusion (PMMD), a diffusion framework that synthesizes photorealistic person images conditioned on multi-view references, pose maps, and text prompts. A multimodal encoder jointly models visual views, pose features, and semantic descriptions, which reduces cross modal discrepancy and improves identity fidelity. We further design a ResCVA module to enhance local detail while preserving global structure, and a cross modal fusion module that integrates image semantics with text throughout the denoising pipeline. Experiments on the DeepFashion MultiModal dataset show that PMMD outperforms representative baselines in consistency, detail preservation, and controllability. Project page and code are available at https://github.com/ZANMANGLOOPYE/PMMD.
>
---
#### [new 008] Uni-Parser Technical Report
- **分类: cs.CV**

- **简介: 该论文提出Uni-Parser，一种面向科技文献与专利的工业级多模态文档解析引擎。旨在解决传统流水线方法跨模态对齐差、扩展难、吞吐低等问题，通过多专家模块化架构、动态调度与GPU优化，实现高精度、高吞吐、低成本的大规模PDF解析。**

- **链接: [https://arxiv.org/pdf/2512.15098v1](https://arxiv.org/pdf/2512.15098v1)**

> **作者:** Xi Fang; Haoyi Tao; Shuwen Yang; Suyang Zhong; Haocheng Lu; Han Lyu; Chaozheng Huang; Xinyu Li; Linfeng Zhang; Guolin Ke
>
> **摘要:** This technical report introduces Uni-Parser, an industrial-grade document parsing engine tailored for scientific literature and patents, delivering high throughput, robust accuracy, and cost efficiency. Unlike pipeline-based document parsing methods, Uni-Parser employs a modular, loosely coupled multi-expert architecture that preserves fine-grained cross-modal alignments across text, equations, tables, figures, and chemical structures, while remaining easily extensible to emerging modalities. The system incorporates adaptive GPU load balancing, distributed inference, dynamic module orchestration, and configurable modes that support either holistic or modality-specific parsing. Optimized for large-scale cloud deployment, Uni-Parser achieves a processing rate of up to 20 PDF pages per second on 8 x NVIDIA RTX 4090D GPUs, enabling cost-efficient inference across billions of pages. This level of scalability facilitates a broad spectrum of downstream applications, ranging from literature retrieval and summarization to the extraction of chemical structures, reaction schemes, and bioactivity data, as well as the curation of large-scale corpora for training next-generation large language models and AI4Science models.
>
---
#### [new 009] VAAS: Vision-Attention Anomaly Scoring for Image Manipulation Detection in Digital Forensics
- **分类: cs.CV; cs.MM**

- **简介: 该论文面向数字取证中的图像篡改检测任务，旨在解决AI生成伪造图像难识别、缺乏可量化异常强度的问题。提出VAAS框架：融合ViT全局注意力异常估计与SegFormer补丁级自一致性评分，输出连续、可解释的定位与强度联合异常分。**

- **链接: [https://arxiv.org/pdf/2512.15512v1](https://arxiv.org/pdf/2512.15512v1)**

> **作者:** Opeyemi Bamigbade; Mark Scanlon; John Sheppard
>
> **摘要:** Recent advances in AI-driven image generation have introduced new challenges for verifying the authenticity of digital evidence in forensic investigations. Modern generative models can produce visually consistent forgeries that evade traditional detectors based on pixel or compression artefacts. Most existing approaches also lack an explicit measure of anomaly intensity, which limits their ability to quantify the severity of manipulation. This paper introduces Vision-Attention Anomaly Scoring (VAAS), a novel dual-module framework that integrates global attention-based anomaly estimation using Vision Transformers (ViT) with patch-level self-consistency scoring derived from SegFormer embeddings. The hybrid formulation provides a continuous and interpretable anomaly score that reflects both the location and degree of manipulation. Evaluations on the DF2023 and CASIA v2.0 datasets demonstrate that VAAS achieves competitive F1 and IoU performance, while enhancing visual explainability through attention-guided anomaly maps. The framework bridges quantitative detection with human-understandable reasoning, supporting transparent and reliable image integrity assessment. The source code for all experiments and corresponding materials for reproducing the results are available open source.
>
---
#### [new 010] Expand and Prune: Maximizing Trajectory Diversity for Effective GRPO in Generative Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文面向生成模型对齐任务，解决GRPO中大组采样与高计算成本的矛盾。提出Pro-GRPO框架：通过潜空间轨迹动态剪枝（早停奖励聚集轨迹），结合“扩展—剪枝”策略，在提升轨迹多样性的同时显著降低计算开销。**

- **链接: [https://arxiv.org/pdf/2512.15347v1](https://arxiv.org/pdf/2512.15347v1)**

> **作者:** Shiran Ge; Chenyi Huang; Yuang Ai; Qihang Fan; Huaibo Huang; Ran He
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Group Relative Policy Optimization (GRPO) is a powerful technique for aligning generative models, but its effectiveness is bottlenecked by the conflict between large group sizes and prohibitive computational costs. In this work, we investigate the trade-off through empirical studies, yielding two key observations. First, we discover the reward clustering phenomenon in which many trajectories collapse toward the group-mean reward, offering limited optimization value. Second, we design a heuristic strategy named Optimal Variance Filtering (OVF), and verify that a high-variance subset of trajectories, selected by OVF can outperform the larger, unfiltered group. However, this static, post-sampling OVF approach still necessitates critical computational overhead, as it performs unnecessary sampling for trajectories that are ultimately discarded. To resolve this, we propose Pro-GRPO (Proactive GRPO), a novel dynamic framework that integrates latent feature-based trajectory pruning into the sampling process. Through the early termination of reward-clustered trajectories, Pro-GRPO reduces computational overhead. Leveraging its efficiency, Pro-GRPO employs an "Expand-and-Prune" strategy. This strategy first expands the size of initial sampling group to maximize trajectory diversity, then it applies multi-step OVF to the latents, avoiding prohibitive computational costs. Extensive experiments on both diffusion-based and flow-based models demonstrate the generality and effectiveness of our Pro-GRPO framework.
>
---
#### [new 011] DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models
- **分类: cs.CV**

- **简介: 该论文提出DiffusionVL，将现有自回归（AR）多模态模型转化为扩散式视觉语言模型（dVLM）。旨在解决dVLM性能落后于AR模型的问题，通过轻量微调实现范式迁移，支持任意长度生成与KV缓存复用，显著提升性能与推理速度。**

- **链接: [https://arxiv.org/pdf/2512.15713v1](https://arxiv.org/pdf/2512.15713v1)**

> **作者:** Lunbin Zeng; Jingfeng Yao; Bencheng Liao; Hongyuan Tao; Wenyu Liu; Xinggang Wang
>
> **备注:** 11 pages, 5 figures, conference or other essential info
>
> **摘要:** In recent multimodal research, the diffusion paradigm has emerged as a promising alternative to the autoregressive paradigm (AR), owing to its unique decoding advantages. However, due to the capability limitations of the base diffusion language model, the performance of the diffusion vision language model (dVLM) still lags significantly behind that of mainstream models. This leads to a simple yet fundamental question: Is it possible to construct dVLMs based on existing powerful AR models? In response, we propose DiffusionVL, a dVLM family that could be translated from any powerful AR models. Through simple fine-tuning, we successfully adapt AR pre-trained models into the diffusion paradigm. This approach yields two key observations: (1) The paradigm shift from AR-based multimodal models to diffusion is remarkably effective. (2) Direct conversion of an AR language model to a dVLM is also feasible, achieving performance competitive with LLaVA-style visual-instruction-tuning. Further, we introduce a block-decoding design into dVLMs that supports arbitrary-length generation and KV cache reuse, achieving a significant inference speedup. We conduct a large number of experiments. Despite training with less than 5% of the data required by prior methods, DiffusionVL achieves a comprehensive performance improvement-a 34.4% gain on the MMMU-Pro (vision) bench and 37.5% gain on the MME (Cog.) bench-alongside a 2x inference speedup. The model and code are released at https://github.com/hustvl/DiffusionVL.
>
---
#### [new 012] MMMamba: A Versatile Cross-Modal In Context Fusion Framework for Pan-Sharpening and Zero-Shot Image Enhancement
- **分类: cs.CV**

- **简介: 该论文面向遥感图像融合任务，解决传统方法在泛化性与计算效率上的不足。提出MMMamba框架，基于Mamba架构与多模态交错扫描机制，实现跨模态上下文融合，支持全色锐化与零样本图像增强。**

- **链接: [https://arxiv.org/pdf/2512.15261v1](https://arxiv.org/pdf/2512.15261v1)**

> **作者:** Yingying Wang; Xuanhua He; Chen Wu; Jialing Huang; Suiyun Zhang; Rui Liu; Xinghao Ding; Haoxuan Che
>
> **备注:** \link{Code}{https://github.com/Gracewangyy/MMMamba}
>
> **摘要:** Pan-sharpening aims to generate high-resolution multispectral (HRMS) images by integrating a high-resolution panchromatic (PAN) image with its corresponding low-resolution multispectral (MS) image. To achieve effective fusion, it is crucial to fully exploit the complementary information between the two modalities. Traditional CNN-based methods typically rely on channel-wise concatenation with fixed convolutional operators, which limits their adaptability to diverse spatial and spectral variations. While cross-attention mechanisms enable global interactions, they are computationally inefficient and may dilute fine-grained correspondences, making it difficult to capture complex semantic relationships. Recent advances in the Multimodal Diffusion Transformer (MMDiT) architecture have demonstrated impressive success in image generation and editing tasks. Unlike cross-attention, MMDiT employs in-context conditioning to facilitate more direct and efficient cross-modal information exchange. In this paper, we propose MMMamba, a cross-modal in-context fusion framework for pan-sharpening, with the flexibility to support image super-resolution in a zero-shot manner. Built upon the Mamba architecture, our design ensures linear computational complexity while maintaining strong cross-modal interaction capacity. Furthermore, we introduce a novel multimodal interleaved (MI) scanning mechanism that facilitates effective information exchange between the PAN and MS modalities. Extensive experiments demonstrate the superior performance of our method compared to existing state-of-the-art (SOTA) techniques across multiple tasks and benchmarks.
>
---
#### [new 013] TalkVerse: Democratizing Minute-Long Audio-Driven Video Generation
- **分类: cs.CV; cs.AI; cs.MM; cs.SD**

- **简介: 该论文面向音频驱动的长时 talking-video 生成任务，旨在解决现有方法依赖闭源数据、计算成本高、难以复现的问题。作者构建了开源大规模数据集 TalkVerse（230 万片段），并提出轻量级 5B DiT 基线模型，支持分钟级生成、零-shot 拍摄与语音重配音，显著降低推理开销。**

- **链接: [https://arxiv.org/pdf/2512.14938v1](https://arxiv.org/pdf/2512.14938v1)**

> **作者:** Zhenzhi Wang; Jian Wang; Ke Ma; Dahua Lin; Bing Zhou
>
> **备注:** open-sourced single-person full-body talking video generation dataset, training code and checkpoints
>
> **摘要:** We introduce TalkVerse, a large-scale, open corpus for single-person, audio-driven talking video generation designed to enable fair, reproducible comparison across methods. While current state-of-the-art systems rely on closed data or compute-heavy models, TalkVerse offers 2.3 million high-resolution (720p/1080p) audio-video synchronized clips totaling 6.3k hours. These are curated from over 60k hours of video via a transparent pipeline that includes scene-cut detection, aesthetic assessment, strict audio-visual synchronization checks, and comprehensive annotations including 2D skeletons and structured visual/audio-style captions. Leveraging TalkVerse, we present a reproducible 5B DiT baseline built on Wan2.2-5B. By utilizing a video VAE with a high downsampling ratio and a sliding window mechanism with motion-frame context, our model achieves minute-long generation with low drift. It delivers comparable lip-sync and visual quality to the 14B Wan-S2V model but with 10$\times$ lower inference cost. To enhance storytelling in long videos, we integrate an MLLM director to rewrite prompts based on audio and visual cues. Furthermore, our model supports zero-shot video dubbing via controlled latent noise injection. We open-source the dataset, training recipes, and 5B checkpoints to lower barriers for research in audio-driven human video generation. Project Page: https://zhenzhiwang.github.io/talkverse/
>
---
#### [new 014] Where is the Watermark? Interpretable Watermark Detection at the Block Level
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于图像水印检测任务，旨在解决现有水印方法缺乏可解释性、无法定位水印位置的问题。作者提出一种后处理式块级可解释水印方法，基于小波域统计嵌入，生成可视化检测图，兼顾鲁棒性、不可感知性与区域级可解释性。**

- **链接: [https://arxiv.org/pdf/2512.14994v1](https://arxiv.org/pdf/2512.14994v1)**

> **作者:** Maria Bulychev; Neil G. Marchant; Benjamin I. P. Rubinstein
>
> **备注:** 20 pages, 14 figures. Camera-ready for WACV 2026
>
> **摘要:** Recent advances in generative AI have enabled the creation of highly realistic digital content, raising concerns around authenticity, ownership, and misuse. While watermarking has become an increasingly important mechanism to trace and protect digital media, most existing image watermarking schemes operate as black boxes, producing global detection scores without offering any insight into how or where the watermark is present. This lack of transparency impacts user trust and makes it difficult to interpret the impact of tampering. In this paper, we present a post-hoc image watermarking method that combines localised embedding with region-level interpretability. Our approach embeds watermark signals in the discrete wavelet transform domain using a statistical block-wise strategy. This allows us to generate detection maps that reveal which regions of an image are likely watermarked or altered. We show that our method achieves strong robustness against common image transformations while remaining sensitive to semantic manipulations. At the same time, the watermark remains highly imperceptible. Compared to prior post-hoc methods, our approach offers more interpretable detection while retaining competitive robustness. For example, our watermarks are robust to cropping up to half the image.
>
---
#### [new 015] PANDA-PLUS-Bench: A Clinical Benchmark for Evaluating Robustness of AI Foundation Models in Prostate Cancer Diagnosis
- **分类: cs.CV**

- **简介: 该论文面向前列腺癌Gleason分级任务，旨在解决AI模型依赖幻觉性切片伪影而非真实生物特征导致临床鲁棒性差的问题。作者构建了PANDA-PLUS-Bench基准数据集，评估7个基础模型对滑片混淆因素的鲁棒性，并开源评测工具。**

- **链接: [https://arxiv.org/pdf/2512.14922v1](https://arxiv.org/pdf/2512.14922v1)**

> **作者:** Joshua L. Ebbert; Dennis Della Corte
>
> **备注:** 21 pages, 5 figures, 6 Tables
>
> **摘要:** Artificial intelligence foundation models are increasingly deployed for prostate cancer Gleason grading, where GP3/GP4 distinction directly impacts treatment decisions. However, these models may achieve high validation accuracy by learning specimen-specific artifacts rather than generalizable biological features, limiting real-world clinical utility. We introduce PANDA-PLUS-Bench, a curated benchmark dataset derived from expert-annotated prostate biopsies designed specifically to quantify this failure mode. The benchmark comprises nine carefully selected whole slide images from nine unique patients containing diverse Gleason patterns, with non-overlapping tissue patches extracted at both 512x512 and 224x224 pixel resolutions across eight augmentation conditions. Using this benchmark, we evaluate seven foundation models on their ability to separate biological signal from slide-level confounders. Our results reveal substantial variation in robustness across models: Virchow2 achieved the lowest slide-level encoding among large-scale models (81.0%) yet exhibited the second-lowest cross-slide accuracy (47.2%). HistoEncoder, trained specifically on prostate tissue, demonstrated the highest cross-slide accuracy (59.7%) and the strongest slide-level encoding (90.3%), suggesting tissue-specific training may enhance both biological feature capture and slide-specific signatures. All models exhibited measurable within-slide vs. cross-slide accuracy gaps, though the magnitude varied from 19.9 percentage points to 26.9 percentage points. We provide an open-source Google Colab notebook enabling researchers to evaluate additional foundation models against our benchmark using standardized metrics. PANDA-PLUS-Bench addresses a critical gap in foundation model evaluation by providing a purpose-built resource for robustness assessment in the clinically important context of Gleason grading.
>
---
#### [new 016] MECAD: A multi-expert architecture for continual anomaly detection
- **分类: cs.CV**

- **简介: 该论文提出MECAD，一种面向持续异常检测的多专家架构。旨在解决工业场景中产品类别动态增加导致的知识遗忘问题，通过特征相似性动态分配专家、优化核心集选择与重放缓冲机制，实现增量学习，避免全模型重训练，兼顾效率与知识保留。**

- **链接: [https://arxiv.org/pdf/2512.15323v1](https://arxiv.org/pdf/2512.15323v1)**

> **作者:** Malihe Dahmardeh; Francesco Setti
>
> **备注:** Accepted to ICIAP 2025
>
> **摘要:** In this paper we propose MECAD, a novel approach for continual anomaly detection using a multi-expert architecture. Our system dynamically assigns experts to object classes based on feature similarity and employs efficient memory management to preserve the knowledge of previously seen classes. By leveraging an optimized coreset selection and a specialized replay buffer mechanism, we enable incremental learning without requiring full model retraining. Our experimental evaluation on the MVTec AD dataset demonstrates that the optimal 5-expert configuration achieves an average AUROC of 0.8259 across 15 diverse object categories while significantly reducing knowledge degradation compared to single-expert approaches. This framework balances computational efficiency, specialized knowledge retention, and adaptability, making it well-suited for industrial environments with evolving product types.
>
---
#### [new 017] Prototypical Learning Guided Context-Aware Segmentation Network for Few-Shot Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文面向少样本异常检测（FSAD）任务，旨在解决预训练特征与目标场景间域差距导致的检测性能下降问题。提出PCSNet模型，包含原型特征适配（PFA）和上下文感知分割（CAS）子网络，通过原型引导、像素级差异损失和伪异常训练提升特征判别力与定位精度。**

- **链接: [https://arxiv.org/pdf/2512.15319v1](https://arxiv.org/pdf/2512.15319v1)**

> **作者:** Yuxin Jiang; Yunkang Cao; Weiming Shen
>
> **摘要:** Few-shot anomaly detection (FSAD) denotes the identification of anomalies within a target category with a limited number of normal samples. Existing FSAD methods largely rely on pre-trained feature representations to detect anomalies, but the inherent domain gap between pre-trained representations and target FSAD scenarios is often overlooked. This study proposes a Prototypical Learning Guided Context-Aware Segmentation Network (PCSNet) to address the domain gap, thereby improving feature descriptiveness in target scenarios and enhancing FSAD performance. In particular, PCSNet comprises a prototypical feature adaption (PFA) sub-network and a context-aware segmentation (CAS) sub-network. PFA extracts prototypical features as guidance to ensure better feature compactness for normal data while distinct separation from anomalies. A pixel-level disparity classification loss is also designed to make subtle anomalies more distinguishable. Then a CAS sub-network is introduced for pixel-level anomaly localization, where pseudo anomalies are exploited to facilitate the training process. Experimental results on MVTec and MPDD demonstrate the superior FSAD performance of PCSNet, with 94.9% and 80.2% image-level AUROC in an 8-shot scenario, respectively. Real-world applications on automotive plastic part inspection further demonstrate that PCSNet can achieve promising results with limited training samples. Code is available at https://github.com/yuxin-jiang/PCSNet.
>
---
#### [new 018] Explainable Action Form Assessment by Exploiting Multimodal Chain-of-Thoughts Reasoning
- **分类: cs.CV**

- **简介: 该论文提出人类动作形态评估（AFA）新任务，旨在解决动作标准化程度评估与可解释反馈缺失问题。构建多模态链式思维数据集CoT-AFA，提出Explainable Fitness Assessor框架，融合视觉与语义信息，实现动作判别、原因解释与改进建议一体化。**

- **链接: [https://arxiv.org/pdf/2512.15153v1](https://arxiv.org/pdf/2512.15153v1)**

> **作者:** Mengshi Qi; Yeteng Wu; Xianlin Zhang; Huadong Ma
>
> **摘要:** Evaluating whether human action is standard or not and providing reasonable feedback to improve action standardization is very crucial but challenging in real-world scenarios. However, current video understanding methods are mainly concerned with what and where the action is, which is unable to meet the requirements. Meanwhile, most of the existing datasets lack the labels indicating the degree of action standardization, and the action quality assessment datasets lack explainability and detailed feedback. Therefore, we define a new Human Action Form Assessment (AFA) task, and introduce a new diverse dataset CoT-AFA, which contains a large scale of fitness and martial arts videos with multi-level annotations for comprehensive video analysis. We enrich the CoT-AFA dataset with a novel Chain-of-Thought explanation paradigm. Instead of offering isolated feedback, our explanations provide a complete reasoning process--from identifying an action step to analyzing its outcome and proposing a concrete solution. Furthermore, we propose a framework named Explainable Fitness Assessor, which can not only judge an action but also explain why and provide a solution. This framework employs two parallel processing streams and a dynamic gating mechanism to fuse visual and semantic information, thereby boosting its analytical capabilities. The experimental results demonstrate that our method has achieved improvements in explanation generation (e.g., +16.0% in CIDEr), action classification (+2.7% in accuracy) and quality assessment (+2.1% in accuracy), revealing great potential of CoT-AFA for future studies. Our dataset and source code is available at https://github.com/MICLAB-BUPT/EFA.
>
---
#### [new 019] Vibe Spaces for Creatively Connecting and Expressing Visual Concepts
- **分类: cs.CV**

- **简介: 该论文提出“Vibe Blending”新任务，旨在生成视觉概念的创意融合体。针对现有方法难以在非线性潜空间中识别并遍历远距概念间共享语义（即“vibe”）的问题，作者构建了分层图流形“Vibe Space”，在CLIP等特征空间中学习低维测地线路径，并设计融合人类判断、LLM推理与几何难度评分的评估框架。**

- **链接: [https://arxiv.org/pdf/2512.14884v1](https://arxiv.org/pdf/2512.14884v1)**

> **作者:** Huzheng Yang; Katherine Xu; Andrew Lu; Michael D. Grossberg; Yutong Bai; Jianbo Shi
>
> **备注:** Project page: https://huzeyann.github.io/VibeSpace-webpage/
>
> **摘要:** Creating new visual concepts often requires connecting distinct ideas through their most relevant shared attributes -- their vibe. We introduce Vibe Blending, a novel task for generating coherent and meaningful hybrids that reveals these shared attributes between images. Achieving such blends is challenging for current methods, which struggle to identify and traverse nonlinear paths linking distant concepts in latent space. We propose Vibe Space, a hierarchical graph manifold that learns low-dimensional geodesics in feature spaces like CLIP, enabling smooth and semantically consistent transitions between concepts. To evaluate creative quality, we design a cognitively inspired framework combining human judgments, LLM reasoning, and a geometric path-based difficulty score. We find that Vibe Space produces blends that humans consistently rate as more creative and coherent than current methods.
>
---
#### [new 020] Is Nano Banana Pro a Low-Level Vision All-Rounder? A Comprehensive Evaluation on 14 Tasks and 40 Datasets
- **分类: cs.CV**

- **简介: 该论文评估商用文生图模型Nano Banana Pro能否作为低-level视觉通用求解器。针对14类任务、40个数据集开展零样本评测，发现其主观视觉质量优于专用模型，但定量指标落后，主因生成随机性导致像素一致性不足。**

- **链接: [https://arxiv.org/pdf/2512.15110v1](https://arxiv.org/pdf/2512.15110v1)**

> **作者:** Jialong Zuo; Haoyou Deng; Hanyu Zhou; Jiaxin Zhu; Yicheng Zhang; Yiwei Zhang; Yongxin Yan; Kaixing Huang; Weisen Chen; Yongtai Deng; Rui Jin; Nong Sang; Changxin Gao
>
> **备注:** Technical Report; 65 Pages, 36 Figures, 17 Tables; Poject Page: https://lowlevelbanana.github.io/
>
> **摘要:** The rapid evolution of text-to-image generation models has revolutionized visual content creation. While commercial products like Nano Banana Pro have garnered significant attention, their potential as generalist solvers for traditional low-level vision challenges remains largely underexplored. In this study, we investigate the critical question: Is Nano Banana Pro a Low-Level Vision All-Rounder? We conducted a comprehensive zero-shot evaluation across 14 distinct low-level tasks spanning 40 diverse datasets. By utilizing simple textual prompts without fine-tuning, we benchmarked Nano Banana Pro against state-of-the-art specialist models. Our extensive analysis reveals a distinct performance dichotomy: while \textbf{Nano Banana Pro demonstrates superior subjective visual quality}, often hallucinating plausible high-frequency details that surpass specialist models, it lags behind in traditional reference-based quantitative metrics. We attribute this discrepancy to the inherent stochasticity of generative models, which struggle to maintain the strict pixel-level consistency required by conventional metrics. This report identifies Nano Banana Pro as a capable zero-shot contender for low-level vision tasks, while highlighting that achieving the high fidelity of domain specialists remains a significant hurdle.
>
---
#### [new 021] Multi-View Foundation Models
- **分类: cs.CV**

- **简介: 该论文提出多视图基础模型，解决单图基础模型在多视图下特征不一致问题。通过在Transformer模型（如DINO、SAM、CLIP）中引入3D感知注意力层，实现跨视图对应点的特征一致性，无需显式3D重建，直接在图像空间优化，提升表面法向估计与多视图分割性能。**

- **链接: [https://arxiv.org/pdf/2512.15708v1](https://arxiv.org/pdf/2512.15708v1)**

> **作者:** Leo Segre; Or Hirschorn; Shai Avidan
>
> **摘要:** Foundation models are vital tools in various Computer Vision applications. They take as input a single RGB image and output a deep feature representation that is useful for various applications. However, in case we have multiple views of the same 3D scene, they operate on each image independently and do not always produce consistent features for the same 3D point. We propose a way to convert a Foundation Model into a Multi-View Foundation Model. Such a model takes as input a set of images and outputs a feature map for each image such that the features of corresponding points are as consistent as possible. This approach bypasses the need to build a consistent 3D model of the features and allows direct manipulation in the image space. Specifically, we show how to augment Transformers-based foundation models (i.e., DINO, SAM, CLIP) with intermediate 3D-aware attention layers that help match features across different views. As leading examples, we show surface normal estimation and multi-view segmentation tasks. Quantitative experiments show that our method improves feature matching considerably compared to current foundation models.
>
---
#### [new 022] Evaluation of deep learning architectures for wildlife object detection: A comparative study of ResNet and Inception
- **分类: cs.CV**

- **简介: 该论文属野生动物目标检测任务，旨在解决环境多变、物种相似及类内差异大导致的检测难题。研究对比ResNet-101与Inception v3在统一预处理和数据划分下的性能，评估其准确率与mAP，并分析优劣与局限。**

- **链接: [https://arxiv.org/pdf/2512.15480v1](https://arxiv.org/pdf/2512.15480v1)**

> **作者:** Malach Obisa Amonga; Benard Osero; Edna Too
>
> **摘要:** Wildlife object detection plays a vital role in biodiversity conservation, ecological monitoring, and habitat protection. However, this task is often challenged by environmental variability, visual similarities among species, and intra-class diversity. This study investigates the effectiveness of two individual deep learning architectures ResNet-101 and Inception v3 for wildlife object detection under such complex conditions. The models were trained and evaluated on a wildlife image dataset using a standardized preprocessing approach, which included resizing images to a maximum dimension of 800 pixels, converting them to RGB format, and transforming them into PyTorch tensors. A ratio of 70:30 training and validation split was used for model development. The ResNet-101 model achieved a classification accuracy of 94% and a mean Average Precision (mAP) of 0.91, showing strong performance in extracting deep hierarchical features. The Inception v3 model performed slightly better, attaining a classification accuracy of 95% and a mAP of 0.92, attributed to its efficient multi-scale feature extraction through parallel convolutions. Despite the strong results, both models exhibited challenges when detecting species with similar visual characteristics or those captured under poor lighting and occlusion. Nonetheless, the findings confirm that both ResNet-101 and Inception v3 are effective models for wildlife object detection tasks and provide a reliable foundation for conservation-focused computer vision applications.
>
---
#### [new 023] Intersectional Fairness in Vision-Language Models for Medical Image Disease Classification
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向医学图像疾病分类任务，解决视觉-语言模型在交叉人口亚组（如年龄、性别、种族组合）中存在的诊断置信度偏差问题。提出无需敏感属性推理的CMAC-MMD训练框架，提升交叉公平性与整体诊断精度。**

- **链接: [https://arxiv.org/pdf/2512.15249v1](https://arxiv.org/pdf/2512.15249v1)**

> **作者:** Yupeng Zhang; Adam G. Dunn; Usman Naseem; Jinman Kim
>
> **摘要:** Medical artificial intelligence (AI) systems, particularly multimodal vision-language models (VLM), often exhibit intersectional biases where models are systematically less confident in diagnosing marginalised patient subgroups. Such bias can lead to higher rates of inaccurate and missed diagnoses due to demographically skewed data and divergent distributions of diagnostic certainty. Current fairness interventions frequently fail to address these gaps or compromise overall diagnostic performance to achieve statistical parity among the subgroups. In this study, we developed Cross-Modal Alignment Consistency (CMAC-MMD), a training framework that standardises diagnostic certainty across intersectional patient subgroups. Unlike traditional debiasing methods, this approach equalises the model's decision confidence without requiring sensitive demographic data during clinical inference. We evaluated this approach using 10,015 skin lesion images (HAM10000) with external validation on 12,000 images (BCN20000), and 10,000 fundus images for glaucoma detection (Harvard-FairVLMed), stratifying performance by intersectional age, gender, and race attributes. In the dermatology cohort, the proposed method reduced the overall intersectional missed diagnosis gap (difference in True Positive Rate, $Δ$TPR) from 0.50 to 0.26 while improving the overall Area Under the Curve (AUC) from 0.94 to 0.97 compared to standard training. Similarly, for glaucoma screening, the method reduced $Δ$TPR from 0.41 to 0.31, achieving a better AUC of 0.72 (vs. 0.71 baseline). This establishes a scalable framework for developing high-stakes clinical decision support systems that are both accurate and can perform equitably across diverse patient subgroups, ensuring reliable performance without increasing privacy risks.
>
---
#### [new 024] Towards Seamless Interaction: Causal Turn-Level Modeling of Interactive 3D Conversational Head Dynamics
- **分类: cs.CV**

- **简介: 该论文属3D对话头动生成任务，旨在解决现有方法忽略对话轮次因果性、导致时序不连贯的问题。提出TIMAR框架：基于因果的轮次级掩码自回归建模，融合音视频上下文，用轻量扩散头预测连续3D头动，显著提升生成质量与泛化性。**

- **链接: [https://arxiv.org/pdf/2512.15340v1](https://arxiv.org/pdf/2512.15340v1)**

> **作者:** Junjie Chen; Fei Wang; Zhihao Huang; Qing Zhou; Kun Li; Dan Guo; Linfeng Zhang; Xun Yang
>
> **摘要:** Human conversation involves continuous exchanges of speech and nonverbal cues such as head nods, gaze shifts, and facial expressions that convey attention and emotion. Modeling these bidirectional dynamics in 3D is essential for building expressive avatars and interactive robots. However, existing frameworks often treat talking and listening as independent processes or rely on non-causal full-sequence modeling, hindering temporal coherence across turns. We present TIMAR (Turn-level Interleaved Masked AutoRegression), a causal framework for 3D conversational head generation that models dialogue as interleaved audio-visual contexts. It fuses multimodal information within each turn and applies turn-level causal attention to accumulate conversational history, while a lightweight diffusion head predicts continuous 3D head dynamics that captures both coordination and expressive variability. Experiments on the DualTalk benchmark show that TIMAR reduces Fréchet Distance and MSE by 15-30% on the test set, and achieves similar gains on out-of-distribution data. The source code will be released in the GitHub repository https://github.com/CoderChen01/towards-seamleass-interaction.
>
---
#### [new 025] Isolated Sign Language Recognition with Segmentation and Pose Estimation
- **分类: cs.CV**

- **简介: 该论文面向孤立手语识别（ISLR）任务，旨在解决ASL识别中数据稀缺、 signer差异大、计算成本高的问题。提出融合姿态估计、关键区域分割与ResNet-Transformer联合建模的方法，在降低计算开销的同时提升对 signer 变异的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.14876v1](https://arxiv.org/pdf/2512.14876v1)**

> **作者:** Daniel Perkins; Davis Hunter; Dhrumil Patel; Galen Flanagan
>
> **备注:** 5 pages, 3 Figures
>
> **摘要:** The recent surge in large language models has automated translations of spoken and written languages. However, these advances remain largely inaccessible to American Sign Language (ASL) users, whose language relies on complex visual cues. Isolated sign language recognition (ISLR) - the task of classifying videos of individual signs - can help bridge this gap but is currently limited by scarce per-sign data, high signer variability, and substantial computational costs. We propose a model for ISLR that reduces computational requirements while maintaining robustness to signer variation. Our approach integrates (i) a pose estimation pipeline to extract hand and face joint coordinates, (ii) a segmentation module that isolates relevant information, and (iii) a ResNet-Transformer backbone to jointly model spatial and temporal dependencies.
>
---
#### [new 026] CLIP-FTI: Fine-Grained Face Template Inversion via CLIP-Driven Attribute Conditioning
- **分类: cs.CV**

- **简介: 该论文属人脸模板逆向攻击任务，旨在从泄露的识别模板重建高保真、细粒度人脸图像。针对现有方法面部部件模糊、迁移性差的问题，提出CLIP-FTI：利用CLIP提取面部属性语义嵌入，通过跨模态交互融合模板并驱动StyleGAN生成更真实、可迁移的重建结果。**

- **链接: [https://arxiv.org/pdf/2512.15433v1](https://arxiv.org/pdf/2512.15433v1)**

> **作者:** Longchen Dai; Zixuan Shen; Zhiheng Zhou; Peipeng Yu; Zhihua Xia
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Face recognition systems store face templates for efficient matching. Once leaked, these templates pose a threat: inverting them can yield photorealistic surrogates that compromise privacy and enable impersonation. Although existing research has achieved relatively realistic face template inversion, the reconstructed facial images exhibit over-smoothed facial-part attributes (eyes, nose, mouth) and limited transferability. To address this problem, we present CLIP-FTI, a CLIP-driven fine-grained attribute conditioning framework for face template inversion. Our core idea is to use the CLIP model to obtain the semantic embeddings of facial features, in order to realize the reconstruction of specific facial feature attributes. Specifically, facial feature attribute embeddings extracted from CLIP are fused with the leaked template via a cross-modal feature interaction network and projected into the intermediate latent space of a pretrained StyleGAN. The StyleGAN generator then synthesizes face images with the same identity as the templates but with more fine-grained facial feature attributes. Experiments across multiple face recognition backbones and datasets show that our reconstructions (i) achieve higher identification accuracy and attribute similarity, (ii) recover sharper component-level attribute semantics, and (iii) improve cross-model attack transferability compared to prior reconstruction attacks. To the best of our knowledge, ours is the first method to use additional information besides the face template attack to realize face template inversion and obtains SOTA results.
>
---
#### [new 027] Skyra: AI-Generated Video Detection via Grounded Artifact Reasoning
- **分类: cs.CV**

- **简介: 该论文面向AI生成视频检测任务，旨在解决现有方法缺乏可解释性的问题。提出Skyra模型，通过识别视觉伪影实现检测与解释；构建ViF-CoT-4K数据集和ViF-Bench基准；采用两阶段训练提升时空感知与推理能力。**

- **链接: [https://arxiv.org/pdf/2512.15693v1](https://arxiv.org/pdf/2512.15693v1)**

> **作者:** Yifei Li; Wenzhao Zheng; Yanran Zhang; Runze Sun; Yu Zheng; Lei Chen; Jie Zhou; Jiwen Lu
>
> **备注:** Project Page: https://github.com/JoeLeelyf/Skyra
>
> **摘要:** The misuse of AI-driven video generation technologies has raised serious social concerns, highlighting the urgent need for reliable AI-generated video detectors. However, most existing methods are limited to binary classification and lack the necessary explanations for human interpretation. In this paper, we present Skyra, a specialized multimodal large language model (MLLM) that identifies human-perceivable visual artifacts in AI-generated videos and leverages them as grounded evidence for both detection and explanation. To support this objective, we construct ViF-CoT-4K for Supervised Fine-Tuning (SFT), which represents the first large-scale AI-generated video artifact dataset with fine-grained human annotations. We then develop a two-stage training strategy that systematically enhances our model's spatio-temporal artifact perception, explanation capability, and detection accuracy. To comprehensively evaluate Skyra, we introduce ViF-Bench, a benchmark comprising 3K high-quality samples generated by over ten state-of-the-art video generators. Extensive experiments demonstrate that Skyra surpasses existing methods across multiple benchmarks, while our evaluation yields valuable insights for advancing explainable AI-generated video detection.
>
---
#### [new 028] Step-GUI Technical Report
- **分类: cs.CV**

- **简介: 该论文面向GUI自动化任务，解决高质量训练数据获取成本高、标注可靠性低的问题。提出自进化训练流水线（含校准步奖励系统）和Step-GUI模型，构建隐私优先的GUI-MCP协议，并发布真实场景基准AndroidDaily。**

- **链接: [https://arxiv.org/pdf/2512.15431v1](https://arxiv.org/pdf/2512.15431v1)**

> **作者:** Haolong Yan; Jia Wang; Xin Huang; Yeqing Shen; Ziyang Meng; Zhimin Fan; Kaijun Tan; Jin Gao; Lieyu Shi; Mi Yang; Shiliang Yang; Zhirui Wang; Brian Li; Kang An; Chenyang Li; Lei Lei; Mengmeng Duan; Danxun Liang; Guodong Liu; Hang Cheng; Hao Wu; Jie Dong; Junhao Huang; Mei Chen; Renjie Yu; Shunshan Li; Xu Zhou; Yiting Dai; Yineng Deng; Yingdan Liang; Zelin Chen; Wen Sun; Chengxu Yan; Chunqin Xu; Dong Li; Fengqiong Xiao; Guanghao Fan; Guopeng Li; Guozhen Peng; Hongbing Li; Hang Li; Hongming Chen; Jingjing Xie; Jianyong Li; Jingyang Zhang; Jiaju Ren; Jiayu Yuan; Jianpeng Yin; Kai Cao; Liang Zhao; Liguo Tan; Liying Shi; Mengqiang Ren; Min Xu; Manjiao Liu; Mao Luo; Mingxin Wan; Na Wang; Nan Wu; Ning Wang; Peiyao Ma; Qingzhou Zhang; Qiao Wang; Qinlin Zeng; Qiong Gao; Qiongyao Li; Shangwu Zhong; Shuli Gao; Shaofan Liu; Shisi Gao; Shuang Luo; Xingbin Liu; Xiaojia Liu; Xiaojie Hou; Xin Liu; Xuanti Feng; Xuedan Cai; Xuan Wen; Xianwei Zhu; Xin Liang; Xin Liu; Xin Zhou; Yingxiu Zhao; Yukang Shi; Yunfang Xu; Yuqing Zeng; Yixun Zhang; Zejia Weng; Zhonghao Yan; Zhiguo Huang; Zhuoyu Wang; Zheng Ge; Jing Li; Yibo Zhu; Binxing Jiao; Xiangyu Zhang; Daxin Jiang
>
> **备注:** 41 pages, 26 figures
>
> **摘要:** Recent advances in multimodal large language models unlock unprecedented opportunities for GUI automation. However, a fundamental challenge remains: how to efficiently acquire high-quality training data while maintaining annotation reliability? We introduce a self-evolving training pipeline powered by the Calibrated Step Reward System, which converts model-generated trajectories into reliable training signals through trajectory-level calibration, achieving >90% annotation accuracy with 10-100x lower cost. Leveraging this pipeline, we introduce Step-GUI, a family of models (4B/8B) that achieves state-of-the-art GUI performance (8B: 80.2% AndroidWorld, 48.5% OSWorld, 62.6% ScreenShot-Pro) while maintaining robust general capabilities. As GUI agent capabilities improve, practical deployment demands standardized interfaces across heterogeneous devices while protecting user privacy. To this end, we propose GUI-MCP, the first Model Context Protocol for GUI automation with hierarchical architecture that combines low-level atomic operations and high-level task delegation to local specialist models, enabling high-privacy execution where sensitive data stays on-device. Finally, to assess whether agents can handle authentic everyday usage, we introduce AndroidDaily, a benchmark grounded in real-world mobile usage patterns with 3146 static actions and 235 end-to-end tasks across high-frequency daily scenarios (8B: static 89.91%, end-to-end 52.50%). Our work advances the development of practical GUI agents and demonstrates strong potential for real-world deployment in everyday digital interactions.
>
---
#### [new 029] Emotion Recognition in Signers
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属多模态情感识别任务，旨在解决手语者情感识别中语法与情感面部表达重叠、数据稀缺两大挑战。作者构建日语手语数据集eJSL，结合英手语数据集BOBSL，提出跨语言方法，验证文本情绪识别迁移、时序片段选择及手部运动建模的有效性，并建立强基线。**

- **链接: [https://arxiv.org/pdf/2512.15376v1](https://arxiv.org/pdf/2512.15376v1)**

> **作者:** Kotaro Funakoshi; Yaoxiong Zhu
>
> **摘要:** Recognition of signers' emotions suffers from one theoretical challenge and one practical challenge, namely, the overlap between grammatical and affective facial expressions and the scarcity of data for model training. This paper addresses these two challenges in a cross-lingual setting using our eJSL dataset, a new benchmark dataset for emotion recognition in Japanese Sign Language signers, and BOBSL, a large British Sign Language dataset with subtitles. In eJSL, two signers expressed 78 distinct utterances with each of seven different emotional states, resulting in 1,092 video clips. We empirically demonstrate that 1) textual emotion recognition in spoken language mitigates data scarcity in sign language, 2) temporal segment selection has a significant impact, and 3) incorporating hand motion enhances emotion recognition in signers. Finally we establish a stronger baseline than spoken language LLMs.
>
---
#### [new 030] MVGSR: Multi-View Consistent 3D Gaussian Super-Resolution via Epipolar Guidance
- **分类: cs.CV**

- **简介: 该论文面向3D高斯泼溅（3DGS）超分辨率任务，解决低分辨率输入导致高分辨率渲染质量差、跨视图不一致的问题。提出MVGSR框架：基于相机位姿的辅助视图选择方法，以及首创的极线约束多视图注意力机制，提升几何一致性与细节保真度。**

- **链接: [https://arxiv.org/pdf/2512.15048v1](https://arxiv.org/pdf/2512.15048v1)**

> **作者:** Kaizhe Zhang; Shinan Chen; Qian Zhao; Weizhan Zhang; Caixia Yan; Yudeng Xin
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Scenes reconstructed by 3D Gaussian Splatting (3DGS) trained on low-resolution (LR) images are unsuitable for high-resolution (HR) rendering. Consequently, a 3DGS super-resolution (SR) method is needed to bridge LR inputs and HR rendering. Early 3DGS SR methods rely on single-image SR networks, which lack cross-view consistency and fail to fuse complementary information across views. More recent video-based SR approaches attempt to address this limitation but require strictly sequential frames, limiting their applicability to unstructured multi-view datasets. In this work, we introduce Multi-View Consistent 3D Gaussian Splatting Super-Resolution (MVGSR), a framework that focuses on integrating multi-view information for 3DGS rendering with high-frequency details and enhanced consistency. We first propose an Auxiliary View Selection Method based on camera poses, making our method adaptable for arbitrarily organized multi-view datasets without the need of temporal continuity or data reordering. Furthermore, we introduce, for the first time, an epipolar-constrained multi-view attention mechanism into 3DGS SR, which serves as the core of our proposed multi-view SR network. This design enables the model to selectively aggregate consistent information from auxiliary views, enhancing the geometric consistency and detail fidelity of 3DGS representations. Extensive experiments demonstrate that our method achieves state-of-the-art performance on both object-centric and scene-level 3DGS SR benchmarks.
>
---
#### [new 031] ERIENet: An Efficient RAW Image Enhancement Network under Low-Light Environment
- **分类: cs.CV**

- **简介: 该论文属低光RAW图像增强任务，旨在解决现有方法串行处理多尺度信息导致模型重、速度慢，且忽视绿色通道优势的问题。提出ERIENet：采用全并行多尺度架构与绿通道引导分支，提升效率与重建质量，达146+ FPS（4K）。**

- **链接: [https://arxiv.org/pdf/2512.15186v1](https://arxiv.org/pdf/2512.15186v1)**

> **作者:** Jianan Wang; Yang Hong; Hesong Li; Tao Wang; Songrong Liu; Ying Fu
>
> **备注:** 5 pages, 4 figures, conference ICVISP
>
> **摘要:** RAW images have shown superior performance than sRGB images in many image processing tasks, especially for low-light image enhancement. However, most existing methods for RAW-based low-light enhancement usually sequentially process multi-scale information, which makes it difficult to achieve lightweight models and high processing speeds. Besides, they usually ignore the green channel superiority of RAW images, and fail to achieve better reconstruction performance with good use of green channel information. In this work, we propose an efficient RAW Image Enhancement Network (ERIENet), which parallelly processes multi-scale information with efficient convolution modules, and takes advantage of rich information in green channels to guide the reconstruction of images. Firstly, we introduce an efficient multi-scale fully-parallel architecture with a novel channel-aware residual dense block to extract feature maps, which reduces computational costs and achieves real-time processing speed. Secondly, we introduce a green channel guidance branch to exploit the rich information within the green channels of the input RAW image. It increases the quality of reconstruction results with few parameters and computations. Experiments on commonly used low-light image enhancement datasets show that ERIENet outperforms state-of-the-art methods in enhancing low-light RAW images with higher effiency. It also achieves an optimal speed of over 146 frame-per-second (FPS) for 4K-resolution images on a single NVIDIA GeForce RTX 3090 with 24G memory.
>
---
#### [new 032] DeX-Portrait: Disentangled and Expressive Portrait Animation via Explicit and Latent Motion Representations
- **分类: cs.CV**

- **简介: 该论文属人脸动画任务，旨在解决单图+驱动视频生成中头姿与表情难以解耦控制的问题。提出DeX-Portrait方法：显式建模姿态（全局变换）、隐式编码表情（latent code），通过双分支条件注入与交叉注意力实现解耦驱动，并引入渐进式混合CFG提升身份一致性。**

- **链接: [https://arxiv.org/pdf/2512.15524v1](https://arxiv.org/pdf/2512.15524v1)**

> **作者:** Yuxiang Shi; Zhe Li; Yanwen Wang; Hao Zhu; Xun Cao; Ligang Liu
>
> **备注:** Projectpage: https://syx132.github.io/DeX-Portrait/
>
> **摘要:** Portrait animation from a single source image and a driving video is a long-standing problem. Recent approaches tend to adopt diffusion-based image/video generation models for realistic and expressive animation. However, none of these diffusion models realizes high-fidelity disentangled control between the head pose and facial expression, hindering applications like expression-only or pose-only editing and animation. To address this, we propose DeX-Portrait, a novel approach capable of generating expressive portrait animation driven by disentangled pose and expression signals. Specifically, we represent the pose as an explicit global transformation and the expression as an implicit latent code. First, we design a powerful motion trainer to learn both pose and expression encoders for extracting precise and decomposed driving signals. Then we propose to inject the pose transformation into the diffusion model through a dual-branch conditioning mechanism, and the expression latent through cross attention. Finally, we design a progressive hybrid classifier-free guidance for more faithful identity consistency. Experiments show that our method outperforms state-of-the-art baselines on both animation quality and disentangled controllability.
>
---
#### [new 033] 3DProxyImg: Controllable 3D-Aware Animation Synthesis from Single Image via 2D-3D Aligned Proxy Embedding
- **分类: cs.CV**

- **简介: 该论文属单图像3D动画生成任务，旨在解决现有方法在渲染质量与3D可控性间的权衡困境。提出3DProxyImg框架，通过2D-3D对齐的代理表示解耦几何控制与外观合成，实现轻量、可控、高保真的3D-aware动画生成。**

- **链接: [https://arxiv.org/pdf/2512.15126v1](https://arxiv.org/pdf/2512.15126v1)**

> **作者:** Yupeng Zhu; Xiongzhen Zhang; Ye Chen; Bingbing Ni
>
> **摘要:** 3D animation is central to modern visual media, yet traditional production pipelines remain labor-intensive, expertise-demanding, and computationally expensive. Recent AIGC-based approaches partially automate asset creation and rigging, but they either inherit the heavy costs of full 3D pipelines or rely on video-synthesis paradigms that sacrifice 3D controllability and interactivity. We focus on single-image 3D animation generation and argue that progress is fundamentally constrained by a trade-off between rendering quality and 3D control. To address this limitation, we propose a lightweight 3D animation framework that decouples geometric control from appearance synthesis. The core idea is a 2D-3D aligned proxy representation that uses a coarse 3D estimate as a structural carrier, while delegating high-fidelity appearance and view synthesis to learned image-space generative priors. This proxy formulation enables 3D-aware motion control and interaction comparable to classical pipelines, without requiring accurate geometry or expensive optimization, and naturally extends to coherent background animation. Extensive experiments demonstrate that our method achieves efficient animation generation on low-power platforms and outperforms video-based 3D animation generation in identity preservation, geometric and textural consistency, and the level of precise, interactive control it offers to users.
>
---
#### [new 034] A Masked Reverse Knowledge Distillation Method Incorporating Global and Local Information for Image Anomaly Detection
- **分类: cs.CV**

- **简介: 该论文面向图像异常检测任务，旨在解决知识蒸馏方法易过泛化的问题。提出掩码反向知识蒸馏（MRKD），通过图像级和特征级掩码，将重建转为恢复，融合全局与局部信息，提升上下文建模能力并抑制过泛化。**

- **链接: [https://arxiv.org/pdf/2512.15326v1](https://arxiv.org/pdf/2512.15326v1)**

> **作者:** Yuxin Jiang; Yunkang Can; Weiming Shen
>
> **摘要:** Knowledge distillation is an effective image anomaly detection and localization scheme. However, a major drawback of this scheme is its tendency to overly generalize, primarily due to the similarities between input and supervisory signals. In order to address this issue, this paper introduces a novel technique called masked reverse knowledge distillation (MRKD). By employing image-level masking (ILM) and feature-level masking (FLM), MRKD transforms the task of image reconstruction into image restoration. Specifically, ILM helps to capture global information by differentiating input signals from supervisory signals. On the other hand, FLM incorporates synthetic feature-level anomalies to ensure that the learned representations contain sufficient local information. With these two strategies, MRKD is endowed with stronger image context capture capacity and is less likely to be overgeneralized. Experiments on the widely-used MVTec anomaly detection dataset demonstrate that MRKD achieves impressive performance: image-level 98.9% AU-ROC, pixel-level 98.4% AU-ROC, and 95.3% AU-PRO. In addition, extensive ablation experiments have validated the superiority of MRKD in mitigating the overgeneralization problem.
>
---
#### [new 035] SMART: Semantic Matching Contrastive Learning for Partially View-Aligned Clustering
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文面向部分视图对齐聚类（PVC）任务，解决未对齐数据语义利用不足及跨视图分布偏移导致匹配不准的问题。提出SMART模型，通过语义匹配对比学习缓解分布偏移，统一建模对齐与未对齐样本，提升多视图聚类性能。**

- **链接: [https://arxiv.org/pdf/2512.15396v1](https://arxiv.org/pdf/2512.15396v1)**

> **作者:** Liang Peng; Yixuan Ye; Cheng Liu; Hangjun Che; Fei Wang; Zhiwen Yu; Si Wu; Hau-San Wong
>
> **摘要:** Multi-view clustering has been empirically shown to improve learning performance by leveraging the inherent complementary information across multiple views of data. However, in real-world scenarios, collecting strictly aligned views is challenging, and learning from both aligned and unaligned data becomes a more practical solution. Partially View-aligned Clustering aims to learn correspondences between misaligned view samples to better exploit the potential consistency and complementarity across views, including both aligned and unaligned data. However, most existing PVC methods fail to leverage unaligned data to capture the shared semantics among samples from the same cluster. Moreover, the inherent heterogeneity of multi-view data induces distributional shifts in representations, leading to inaccuracies in establishing meaningful correspondences between cross-view latent features and, consequently, impairing learning effectiveness. To address these challenges, we propose a Semantic MAtching contRasTive learning model (SMART) for PVC. The main idea of our approach is to alleviate the influence of cross-view distributional shifts, thereby facilitating semantic matching contrastive learning to fully exploit semantic relationships in both aligned and unaligned data. Extensive experiments on eight benchmark datasets demonstrate that our method consistently outperforms existing approaches on the PVC problem.
>
---
#### [new 036] BLANKET: Anonymizing Faces in Infant Video Recordings
- **分类: cs.CV**

- **简介: 该论文提出BLANKET方法，解决婴儿视频中人脸匿名化任务：在保护隐私（去标识化）的同时，保留面部关键属性与表情时序一致性。工作包括基于扩散模型的随机脸生成与时空一致的换脸融合，并在婴儿视频数据集上验证其优于DeepPrivacy2。**

- **链接: [https://arxiv.org/pdf/2512.15542v1](https://arxiv.org/pdf/2512.15542v1)**

> **作者:** Ditmar Hadera; Jan Cech; Miroslav Purkrabek; Matej Hoffmann
>
> **备注:** Project website: https://github.com/ctu-vras/blanket-infant-face-anonym
>
> **摘要:** Ensuring the ethical use of video data involving human subjects, particularly infants, requires robust anonymization methods. We propose BLANKET (Baby-face Landmark-preserving ANonymization with Keypoint dEtection consisTency), a novel approach designed to anonymize infant faces in video recordings while preserving essential facial attributes. Our method comprises two stages. First, a new random face, compatible with the original identity, is generated via inpainting using a diffusion model. Second, the new identity is seamlessly incorporated into each video frame through temporally consistent face swapping with authentic expression transfer. The method is evaluated on a dataset of short video recordings of babies and is compared to the popular anonymization method, DeepPrivacy2. Key metrics assessed include the level of de-identification, preservation of facial attributes, impact on human pose estimation (as an example of a downstream task), and presence of artifacts. Both methods alter the identity, and our method outperforms DeepPrivacy2 in all other respects. The code is available as an easy-to-use anonymization demo at https://github.com/ctu-vras/blanket-infant-face-anonym.
>
---
#### [new 037] EmoCaliber: Advancing Reliable Visual Emotion Comprehension via Confidence Verbalization and Calibration
- **分类: cs.CV**

- **简介: 该论文面向视觉情感理解（VEC）任务，解决现有MLLMs将情感预测视为确定性问题、忽视情感主观性与多义性的缺陷。提出EmoCaliber模型，通过三阶段训练实现信心口头化与校准，提升预测可靠性与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.15528v1](https://arxiv.org/pdf/2512.15528v1)**

> **作者:** Daiqing Wu; Dongbao Yang; Can Ma. Yu Zhou
>
> **摘要:** Visual Emotion Comprehension (VEC) aims to infer sentiment polarities or emotion categories from affective cues embedded in images. In recent years, Multimodal Large Language Models (MLLMs) have established a popular paradigm in VEC, leveraging their generalizability to unify VEC tasks defined under diverse emotion taxonomies. While this paradigm achieves notable success, it typically formulates VEC as a deterministic task, requiring the model to output a single, definitive emotion label for each image. Such a formulation insufficiently accounts for the inherent subjectivity of emotion perception, overlooking alternative interpretations that may be equally plausible to different viewers. To address this limitation, we propose equipping MLLMs with capabilities to verbalize their confidence in emotion predictions. This additional signal provides users with an estimate of both the plausibility of alternative interpretations and the MLLMs' self-assessed competence, thereby enhancing reliability in practice. Building on this insight, we introduce a three-stage training framework that progressively endows with structured reasoning, teaches to verbalize confidence, and calibrates confidence expression, culminating in EmoCaliber, a confidence-aware MLLM for VEC. Through fair and comprehensive evaluations on the unified benchmark VECBench, EmoCaliber demonstrates overall superiority against existing methods in both emotion prediction and confidence estimation. These results validate the effectiveness of our approach and mark a feasible step toward more reliable VEC systems. Project page: https://github.com/wdqqdw/EmoCaliber.
>
---
#### [new 038] See It Before You Grab It: Deep Learning-based Action Anticipation in Basketball
- **分类: cs.CV**

- **简介: 该论文提出篮球视频中的动作预判任务，聚焦于投篮后哪队将获得球权（即篮板归属）的提前预测。作者构建了含10万片段、300小时视频及2000+标注篮板事件的新数据集，并基于深度学习方法开展基准实验，同时拓展至篮板分类与定位任务。**

- **链接: [https://arxiv.org/pdf/2512.15386v1](https://arxiv.org/pdf/2512.15386v1)**

> **作者:** Arnau Barrera Roy; Albert Clapés Sintes
>
> **摘要:** Computer vision and video understanding have transformed sports analytics by enabling large-scale, automated analysis of game dynamics from broadcast footage. Despite significant advances in player and ball tracking, pose estimation, action localization, and automatic foul recognition, anticipating actions before they occur in sports videos has received comparatively little attention. This work introduces the task of action anticipation in basketball broadcast videos, focusing on predicting which team will gain possession of the ball following a shot attempt. To benchmark this task, a new self-curated dataset comprising 100,000 basketball video clips, over 300 hours of footage, and more than 2,000 manually annotated rebound events is presented. Comprehensive baseline results are reported using state-of-the-art action anticipation methods, representing the first application of deep learning techniques to basketball rebound prediction. Additionally, two complementary tasks, rebound classification and rebound spotting, are explored, demonstrating that this dataset supports a wide range of video understanding applications in basketball, for which no comparable datasets currently exist. Experimental results highlight both the feasibility and inherent challenges of anticipating rebounds, providing valuable insights into predictive modeling for dynamic multi-agent sports scenarios. By forecasting team possession before rebounds occur, this work enables applications in real-time automated broadcasting and post-game analysis tools to support decision-making.
>
---
#### [new 039] Null-LoRA: Low-Rank Adaptation on Null Space
- **分类: cs.CV**

- **简介: 该论文属参数高效微调任务，旨在解决大模型微调冗余高、参数效率低的问题。提出Null-LoRA方法，将低秩增量更新约束于预训练模型的非平凡零空间内，冻结部分低秩矩阵以减少冗余、提升有效秩，在图像-文本检索和视觉问答任务上以更少参数超越现有方法。**

- **链接: [https://arxiv.org/pdf/2512.15233v1](https://arxiv.org/pdf/2512.15233v1)**

> **作者:** Yi Zhang; Yulei Kang; Haoxuan Chen; Jinxuan Li; ian-Fang Hu
>
> **摘要:** Parameter-efficient fine-tuning methods have gained considerable popularity for adapting large-scale models to downstream tasks, particularly LoRA and its variants. Existing methods perform low-rank adaptation over the full parameter space. However, fine-tuning within a subspace can achieve comparable effectiveness. Inspired by the observation that pre-trained models possess non-trivial null spaces, we propose Null-space based Low-Rank Adaptation (Null-LoRA). Null-LoRA effectively reduces redundancy and enhances effective rank by freezing portions of the low-rank matrices. To further improve parameter efficiency, Null-LoRA constrains the entire incremental update within the null space, maximizing the utilization of incremental updates to adapt to new task paradigms. Null-LoRA surpasses the state of the art with fewer parameters in extensive experiments across image-text retrieval and visual question answering tasks.
>
---
#### [new 040] The Renaissance of Expert Systems: Optical Recognition of Printed Chinese Jianpu Musical Scores with Lyrics
- **分类: cs.CV**

- **简介: 该论文属光学音乐识别（OMR）任务，旨在解决中文简谱（含歌词）的自动识别难题。提出一种无需大量标注数据的模块化专家系统，融合传统计算机视觉与无监督深度学习，实现简谱到MusicXML/MIDI的高精度转换。**

- **链接: [https://arxiv.org/pdf/2512.14758v1](https://arxiv.org/pdf/2512.14758v1)**

> **作者:** Fan Bu; Rongfeng Li; Zijin Li; Ya Li; Linfeng Fan; Pei Huang
>
> **备注:** 13 pages, 12 figures
>
> **摘要:** Large-scale optical music recognition (OMR) research has focused mainly on Western staff notation, leaving Chinese Jianpu (numbered notation) and its rich lyric resources underexplored. We present a modular expert-system pipeline that converts printed Jianpu scores with lyrics into machine-readable MusicXML and MIDI, without requiring massive annotated training data. Our approach adopts a top-down expert-system design, leveraging traditional computer-vision techniques (e.g., phrase correlation, skeleton analysis) to capitalize on prior knowledge, while integrating unsupervised deep-learning modules for image feature embeddings. This hybrid strategy strikes a balance between interpretability and accuracy. Evaluated on The Anthology of Chinese Folk Songs, our system massively digitizes (i) a melody-only collection of more than 5,000 songs (> 300,000 notes) and (ii) a curated subset with lyrics comprising over 1,400 songs (> 100,000 notes). The system achieves high-precision recognition on both melody (note-wise F1 = 0.951) and aligned lyrics (character-wise F1 = 0.931).
>
---
#### [new 041] End-to-End Training for Autoregressive Video Diffusion via Self-Resampling
- **分类: cs.CV**

- **简介: 该论文面向自回归视频扩散建模任务，旨在解决训练与推理不匹配导致的暴露偏差问题。提出教师无关的端到端训练框架Resampling Forcing，通过自采样模拟历史帧错误、稀疏因果掩码保持时序因果性，并引入无参历史路由机制提升长视频生成一致性。**

- **链接: [https://arxiv.org/pdf/2512.15702v1](https://arxiv.org/pdf/2512.15702v1)**

> **作者:** Yuwei Guo; Ceyuan Yang; Hao He; Yang Zhao; Meng Wei; Zhenheng Yang; Weilin Huang; Dahua Lin
>
> **备注:** Project Page: https://guoyww.github.io/projects/resampling-forcing/
>
> **摘要:** Autoregressive video diffusion models hold promise for world simulation but are vulnerable to exposure bias arising from the train-test mismatch. While recent works address this via post-training, they typically rely on a bidirectional teacher model or online discriminator. To achieve an end-to-end solution, we introduce Resampling Forcing, a teacher-free framework that enables training autoregressive video models from scratch and at scale. Central to our approach is a self-resampling scheme that simulates inference-time model errors on history frames during training. Conditioned on these degraded histories, a sparse causal mask enforces temporal causality while enabling parallel training with frame-level diffusion loss. To facilitate efficient long-horizon generation, we further introduce history routing, a parameter-free mechanism that dynamically retrieves the top-k most relevant history frames for each query. Experiments demonstrate that our approach achieves performance comparable to distillation-based baselines while exhibiting superior temporal consistency on longer videos owing to native-length training.
>
---
#### [new 042] Preserving Marker Specificity with Lightweight Channel-Independent Representation Learning
- **分类: cs.CV**

- **简介: 该论文属多路组织成像的自监督表征学习任务，旨在解决早期通道融合模型丢失标记特异性信息的问题。作者提出轻量级通道独立模型CIM-S，通过分离通道与浅层架构保留标记特异性，在Hodgkin淋巴瘤数据上验证其在罕见细胞判别等任务中优于大模型。**

- **链接: [https://arxiv.org/pdf/2512.15410v1](https://arxiv.org/pdf/2512.15410v1)**

> **作者:** Simon Gutwein; Arthur Longuefosse; Jun Seita; Sabine Taschner-Mandl; Roxane Licandro
>
> **备注:** 16 pages, 9 figures, MIDL 2026 conference
>
> **摘要:** Multiplexed tissue imaging measures dozens of protein markers per cell, yet most deep learning models still apply early channel fusion, assuming shared structure across markers. We investigate whether preserving marker independence, combined with deliberately shallow architectures, provides a more suitable inductive bias for self-supervised representation learning in multiplex data than increasing model scale. Using a Hodgkin lymphoma CODEX dataset with 145,000 cells and 49 markers, we compare standard early-fusion CNNs with channel-separated architectures, including a marker-aware baseline and our novel shallow Channel-Independent Model (CIM-S) with 5.5K parameters. After contrastive pretraining and linear evaluation, early-fusion models show limited ability to retain marker-specific information and struggle particularly with rare-cell discrimination. Channel-independent architectures, and CIM-S in particular, achieve substantially stronger representations despite their compact size. These findings are consistent across multiple self-supervised frameworks, remain stable across augmentation settings, and are reproducible across both the 49-marker and reduced 18-marker settings. These results show that lightweight, channel-independent architectures can match or surpass deep early-fusion CNNs and foundation models for multiplex representation learning. Code is available at https://github.com/SimonBon/CIM-S.
>
---
#### [new 043] Adaptive Multimodal Person Recognition: A Robust Framework for Handling Missing Modalities
- **分类: cs.CV; cs.SD; eess.AS; eess.IV**

- **简介: 该论文面向多模态人物识别任务，解决现实场景中模态缺失或退化导致性能下降的问题。提出自适应三模态框架，融合语音、人脸、手势，采用多任务学习、跨模态注意力与置信度加权融合，显著提升缺失模态下的鲁棒性与准确率。**

- **链接: [https://arxiv.org/pdf/2512.14961v1](https://arxiv.org/pdf/2512.14961v1)**

> **作者:** Aref Farhadipour; Teodora Vukovic; Volker Dellwo; Petr Motlicek; Srikanth Madikeri
>
> **备注:** 10 pages and 8 tables
>
> **摘要:** Person recognition systems often rely on audio, visual, or behavioral cues, but real-world conditions frequently result in missing or degraded modalities. To address this challenge, we propose a Trimodal person identification framework that integrates voice, face, and gesture modalities, while remaining robust to modality loss. Our approach leverages multi-task learning to process each modality independently, followed by a cross-attention and gated fusion mechanisms to facilitate interaction across modalities. Moreover, a confidence-weighted fusion strategy dynamically adapts to missing and low-quality data, ensuring optimal classification even in Unimodal or Bimodal scenarios. We evaluate our method on CANDOR, a newly introduced interview-based multimodal dataset, which we benchmark for the first time. Our results demonstrate that the proposed Trimodal system achieves 99.18% Top-1 accuracy on person identification tasks, outperforming conventional Unimodal and late-fusion approaches. In addition, we evaluate our model on the VoxCeleb1 dataset as a benchmark and reach 99.92% accuracy in Bimodal mode. Moreover, we show that our system maintains high accuracy even when one or two modalities are unavailable, making it a robust solution for real-world person recognition applications. The code and data for this work are publicly available.
>
---
#### [new 044] Borrowing from anything: A generalizable framework for reference-guided instance editing
- **分类: cs.CV**

- **简介: 该论文面向参考引导的实例编辑任务，旨在解决语义纠缠问题——即参考图像的内在外观与外在属性难以分离。提出GENIE框架，含空间对齐、自适应残差缩放和渐进注意力融合模块，实现显式解耦与精准外观迁移。**

- **链接: [https://arxiv.org/pdf/2512.15138v1](https://arxiv.org/pdf/2512.15138v1)**

> **作者:** Shengxiao Zhou; Chenghua Li; Jianhao Huang; Qinghao Hu; Yifan Zhang
>
> **备注:** 5 pages
>
> **摘要:** Reference-guided instance editing is fundamentally limited by semantic entanglement, where a reference's intrinsic appearance is intertwined with its extrinsic attributes. The key challenge lies in disentangling what information should be borrowed from the reference, and determining how to apply it appropriately to the target. To tackle this challenge, we propose GENIE, a Generalizable Instance Editing framework capable of achieving explicit disentanglement. GENIE first corrects spatial misalignments with a Spatial Alignment Module (SAM). Then, an Adaptive Residual Scaling Module (ARSM) learns what to borrow by amplifying salient intrinsic cues while suppressing extrinsic attributes, while a Progressive Attention Fusion (PAF) mechanism learns how to render this appearance onto the target, preserving its structure. Extensive experiments on the challenging AnyInsertion dataset demonstrate that GENIE achieves state-of-the-art fidelity and robustness, setting a new standard for disentanglement-based instance editing.
>
---
#### [new 045] VTCBench: Can Vision-Language Models Understand Long Context with Vision-Text Compression?
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出VTCBench基准，评估视觉语言模型（VLMs）在视觉-文本压缩（VTC）下的长上下文理解能力。针对VTC虽提升效率但损害长程依赖建模的问题，设计三大任务：检索、推理与记忆，并评测主流模型，揭示其在VTC下长上下文理解显著退化。**

- **链接: [https://arxiv.org/pdf/2512.15649v1](https://arxiv.org/pdf/2512.15649v1)**

> **作者:** Hongbo Zhao; Meng Wang; Fei Zhu; Wenzhuo Liu; Bolin Ni; Fanhu Zeng; Gaofeng Meng; Zhaoxiang Zhang
>
> **摘要:** The computational and memory overheads associated with expanding the context window of LLMs severely limit their scalability. A noteworthy solution is vision-text compression (VTC), exemplified by frameworks like DeepSeek-OCR and Glyph, which convert long texts into dense 2D visual representations, thereby achieving token compression ratios of 3x-20x. However, the impact of this high information density on the core long-context capabilities of vision-language models (VLMs) remains under-investigated. To address this gap, we introduce the first benchmark for VTC and systematically assess the performance of VLMs across three long-context understanding settings: VTC-Retrieval, which evaluates the model's ability to retrieve and aggregate information; VTC-Reasoning, which requires models to infer latent associations to locate facts with minimal lexical overlap; and VTC-Memory, which measures comprehensive question answering within long-term dialogue memory. Furthermore, we establish the VTCBench-Wild to simulate diverse input scenarios.We comprehensively evaluate leading open-source and proprietary models on our benchmarks. The results indicate that, despite being able to decode textual information (e.g., OCR) well, most VLMs exhibit a surprisingly poor long-context understanding ability with VTC-compressed information, failing to capture long associations or dependencies in the context.This study provides a deep understanding of VTC and serves as a foundation for designing more efficient and scalable VLMs.
>
---
#### [new 046] Beyond Proximity: A Keypoint-Trajectory Framework for Classifying Affiliative and Agonistic Social Networks in Dairy Cattle
- **分类: cs.CV; cs.AI**

- **简介: 该论文属行为识别任务，旨在解决奶牛社交行为（亲和/对抗）自动分类难题。提出基于关键点轨迹的框架，用姿态动态特征替代静态距离阈值，结合YOLOv11、ByteTrack、ZebraPose等实现端到端分析，在商业牧场数据上达77.51%准确率。**

- **链接: [https://arxiv.org/pdf/2512.14998v1](https://arxiv.org/pdf/2512.14998v1)**

> **作者:** Sibi Parivendan; Kashfia Sailunaz; Suresh Neethirajan
>
> **备注:** 36 pages, 12 figures, 8 tables
>
> **摘要:** Precision livestock farming requires objective assessment of social behavior to support herd welfare monitoring, yet most existing approaches infer interactions using static proximity thresholds that cannot distinguish affiliative from agonistic behaviors in complex barn environments. This limitation constrains the interpretability of automated social network analysis in commercial settings. We present a pose-based computational framework for interaction classification that moves beyond proximity heuristics by modeling the spatiotemporal geometry of anatomical keypoints. Rather than relying on pixel-level appearance or simple distance measures, the proposed method encodes interaction-specific motion signatures from keypoint trajectories, enabling differentiation of social interaction valence. The framework is implemented as an end-to-end computer vision pipeline integrating YOLOv11 for object detection (mAP@0.50: 96.24%), supervised individual identification (98.24% accuracy), ByteTrack for multi-object tracking (81.96% accuracy), ZebraPose for 27-point anatomical keypoint estimation, and a support vector machine classifier trained on pose-derived distance dynamics. On annotated interaction clips collected from a commercial dairy barn, the classifier achieved 77.51% accuracy in distinguishing affiliative and agonistic behaviors using pose information alone. Comparative evaluation against a proximity-only baseline shows substantial gains in behavioral discrimination, particularly for affiliative interactions. The results establish a proof-of-concept for automated, vision-based inference of social interactions suitable for constructing interaction-aware social networks, with near-real-time performance on commodity hardware.
>
---
#### [new 047] Model Agnostic Preference Optimization for Medical Image Segmentation
- **分类: cs.CV**

- **简介: 该论文面向医学图像分割任务，解决现有偏好优化方法模型依赖性强、预测多样性不足的问题。提出模型无关的偏好优化框架MAPO，利用Dropout生成随机分割假设构建无真值监督的偏好一致梯度，支持2D/3D CNN与Transformer架构。**

- **链接: [https://arxiv.org/pdf/2512.15009v1](https://arxiv.org/pdf/2512.15009v1)**

> **作者:** Yunseong Nam; Jiwon Jang; Dongkyu Won; Sang Hyun Park; Soopil Kim
>
> **摘要:** Preference optimization offers a scalable supervision paradigm based on relative preference signals, yet prior attempts in medical image segmentation remain model-specific and rely on low-diversity prediction sampling. In this paper, we propose MAPO (Model-Agnostic Preference Optimization), a training framework that utilizes Dropout-driven stochastic segmentation hypotheses to construct preference-consistent gradients without direct ground-truth supervision. MAPO is fully architecture- and dimensionality-agnostic, supporting 2D/3D CNN and Transformer-based segmentation pipelines. Comprehensive evaluations across diverse medical datasets reveal that MAPO consistently enhances boundary adherence, reduces overfitting, and yields more stable optimization dynamics compared to conventional supervised training.
>
---
#### [new 048] Spatia: Video Generation with Updatable Spatial Memory
- **分类: cs.CV; cs.AI**

- **简介: 该论文属视频生成任务，旨在解决长时空间-时间一致性差的问题。提出Spatia框架，以可更新的3D点云作为空间记忆，结合视觉SLAM动态更新，实现动态-静态解耦，提升一致性，并支持相机控制与3D交互编辑。**

- **链接: [https://arxiv.org/pdf/2512.15716v1](https://arxiv.org/pdf/2512.15716v1)**

> **作者:** Jinjing Zhao; Fangyun Wei; Zhening Liu; Hongyang Zhang; Chang Xu; Yan Lu
>
> **备注:** Project page: https://zhaojingjing713.github.io/Spatia/
>
> **摘要:** Existing video generation models struggle to maintain long-term spatial and temporal consistency due to the dense, high-dimensional nature of video signals. To overcome this limitation, we propose Spatia, a spatial memory-aware video generation framework that explicitly preserves a 3D scene point cloud as persistent spatial memory. Spatia iteratively generates video clips conditioned on this spatial memory and continuously updates it through visual SLAM. This dynamic-static disentanglement design enhances spatial consistency throughout the generation process while preserving the model's ability to produce realistic dynamic entities. Furthermore, Spatia enables applications such as explicit camera control and 3D-aware interactive editing, providing a geometrically grounded framework for scalable, memory-driven video generation.
>
---
#### [new 049] An Efficient and Effective Encoder Model for Vision and Language Tasks in the Remote Sensing Domain
- **分类: cs.CV**

- **简介: 该论文面向遥感领域的视觉-语言任务，旨在解决大模型参数多、计算成本高的问题。提出轻量级编码器-only模型GeoMELT，统一支持图像到文本生成与跨模态检索，兼顾高效性与多任务性能。**

- **链接: [https://arxiv.org/pdf/2512.15531v1](https://arxiv.org/pdf/2512.15531v1)**

> **作者:** João Daniel Silva; Joao Magalhaes; Devis Tuia; Bruno Martins
>
> **摘要:** The remote sensing community has recently seen the emergence of methods based on Large Vision and Language Models (LVLMs) that can address multiple tasks at the intersection of computer vision and natural language processing. To fully exploit the potential of such models, a significant focus has been given to the collection of large amounts of training data that cover multiple remote sensing-specific tasks, such as image captioning or visual question answering. However, the cost of using and training LVLMs is high, due to the large number of parameters. While multiple parameter-efficient adaptation techniques have been explored, the computational costs of training and inference with these models can remain prohibitive for most institutions. In this work, we explore the use of encoder-only architectures and propose a model that can effectively address multi-task learning while remaining compact in terms of the number of parameters. In particular, our model tackles combinations of tasks that are not typically explored in a unified model: the generation of text from remote sensing images and cross-modal retrieval. The results of our GeoMELT model - named from Multi-task Efficient Learning Transformer - in established benchmarks confirm the efficacy and efficiency of the proposed approach.
>
---
#### [new 050] From Camera to World: A Plug-and-Play Module for Human Mesh Transformation
- **分类: cs.CV**

- **简介: 该论文属3D人体网格重建任务，旨在解决从单目图像恢复世界坐标系下准确人体网格时因相机旋转未知导致的误差问题。提出Mesh-Plug插件模块，利用RGB与深度图估计相机俯仰角，并联合优化根关节朝向与姿态，实现相机到世界坐标的精准转换。**

- **链接: [https://arxiv.org/pdf/2512.15212v1](https://arxiv.org/pdf/2512.15212v1)**

> **作者:** Changhai Ma; Ziyu Wu; Yunkang Zhang; Qijun Ying; Boyan Liu; Xiaohui Cai
>
> **摘要:** Reconstructing accurate 3D human meshes in the world coordinate system from in-the-wild images remains challenging due to the lack of camera rotation information. While existing methods achieve promising results in the camera coordinate system by assuming zero camera rotation, this simplification leads to significant errors when transforming the reconstructed mesh to the world coordinate system. To address this challenge, we propose Mesh-Plug, a plug-and-play module that accurately transforms human meshes from camera coordinates to world coordinates. Our key innovation lies in a human-centered approach that leverages both RGB images and depth maps rendered from the initial mesh to estimate camera rotation parameters, eliminating the dependency on environmental cues. Specifically, we first train a camera rotation prediction module that focuses on the human body's spatial configuration to estimate camera pitch angle. Then, by integrating the predicted camera parameters with the initial mesh, we design a mesh adjustment module that simultaneously refines the root joint orientation and body pose. Extensive experiments demonstrate that our framework outperforms state-of-the-art methods on the benchmark datasets SPEC-SYN and SPEC-MTP.
>
---
#### [new 051] TBC: A Target-Background Contrast Metric for Low-Altitude Infrared and Visible Image Fusion
- **分类: cs.CV**

- **简介: 该论文面向红外与可见光图像融合任务，解决低空无人机侦察中传统无参考评价指标（如熵、平均梯度）易将噪声误判为细节、导致“噪声陷阱”的问题；提出基于韦伯定律的Target-Background Contrast（TBC）指标，聚焦目标与背景相对对比度，抑制背景噪声、突出目标可见性。**

- **链接: [https://arxiv.org/pdf/2512.15211v1](https://arxiv.org/pdf/2512.15211v1)**

> **作者:** Yufeng Xie
>
> **摘要:** Infrared and visible image fusion is a pivotal technology in low-altitude UAV reconnaissance missions, providing high-quality data support for downstream tasks such as target detection and tracking by integrating thermal saliency with background texture details.However, traditional no-reference metrics fail(Specifically,like Entropy (EN) and Average Gradient (AG)) in complex low-light environments. They often misinterpret high-frequency sensor noise as valid detail. This creates a "Noise Trap," paradoxically assigning higher scores to noisy images and misguiding fusion algorithms.To address this, we propose the Target-Background Contrast (TBC) metric. Inspired by Weber's Law, TBC focuses on the relative contrast of salient targets rather than global statistics. Unlike traditional metrics, TBC penalizes background noise and rewards target visibility. Experiments on the DroneVehicle dataset demonstrate that TBC aligns better with human perception and provides a reliable standard for low-altitude scenarios.
>
---
#### [new 052] Cross-modal ultra-scale learning with tri-modalities of renal biopsy images for glomerular multi-disease auxiliary diagnosis
- **分类: cs.CV**

- **简介: 该论文属医学图像多模态分类任务，旨在解决肾活检三模态（TEM、OM、IM）图像因纳米/微米尺度差异导致的特征融合难问题。提出CMUS-Net网络，含稀疏多实例学习、跨模态尺度注意力模块及多损失加权机制，实现IgAN、MN、LN等多病种精准辅助诊断。**

- **链接: [https://arxiv.org/pdf/2512.15171v1](https://arxiv.org/pdf/2512.15171v1)**

> **作者:** Kaixing Long; Danyi Weng; Yun Mi; Zhentai Zhang; Yanmeng Lu; Jian Geng; Zhitao Zhou; Liming Zhong; Qianjin Feng; Wei Yang; Lei Cao
>
> **摘要:** Constructing a multi-modal automatic classification model based on three types of renal biopsy images can assist pathologists in glomerular multi-disease identification. However, the substantial scale difference between transmission electron microscopy (TEM) image features at the nanoscale and optical microscopy (OM) or immunofluorescence microscopy (IM) images at the microscale poses a challenge for existing multi-modal and multi-scale models in achieving effective feature fusion and improving classification accuracy. To address this issue, we propose a cross-modal ultra-scale learning network (CMUS-Net) for the auxiliary diagnosis of multiple glomerular diseases. CMUS-Net utilizes multiple ultrastructural information to bridge the scale difference between nanometer and micrometer images. Specifically, we introduce a sparse multi-instance learning module to aggregate features from TEM images. Furthermore, we design a cross-modal scale attention module to facilitate feature interaction, enhancing pathological semantic information. Finally, multiple loss functions are combined, allowing the model to weigh the importance among different modalities and achieve precise classification of glomerular diseases. Our method follows the conventional process of renal biopsy pathology diagnosis and, for the first time, performs automatic classification of multiple glomerular diseases including IgA nephropathy (IgAN), membranous nephropathy (MN), and lupus nephritis (LN) based on images from three modalities and two scales. On an in-house dataset, CMUS-Net achieves an ACC of 95.37+/-2.41%, an AUC of 99.05+/-0.53%, and an F1-score of 95.32+/-2.41%. Extensive experiments demonstrate that CMUS-Net outperforms other well-known multi-modal or multi-scale methods and show its generalization capability in staging MN. Code is available at https://github.com/SMU-GL-Group/MultiModal_lkx/tree/main.
>
---
#### [new 053] Robust and Calibrated Detection of Authentic Multimedia Content
- **分类: cs.CV**

- **简介: 该论文属多媒体真实性检测任务，旨在解决现有深伪检测方法假阳性率高、易被对抗攻击的缺陷。作者提出校准式重合成框架，通过判断内容是否可被合理否认，实现高精度、低假阳、抗高效对抗攻击的真实性验证。**

- **链接: [https://arxiv.org/pdf/2512.15182v1](https://arxiv.org/pdf/2512.15182v1)**

> **作者:** Sarim Hashmi; Abdelrahman Elsayed; Mohammed Talha Alam; Samuele Poppi; Nils Lukas
>
> **摘要:** Generative models can synthesize highly realistic content, so-called deepfakes, that are already being misused at scale to undermine digital media authenticity. Current deepfake detection methods are unreliable for two reasons: (i) distinguishing inauthentic content post-hoc is often impossible (e.g., with memorized samples), leading to an unbounded false positive rate (FPR); and (ii) detection lacks robustness, as adversaries can adapt to known detectors with near-perfect accuracy using minimal computational resources. To address these limitations, we propose a resynthesis framework to determine if a sample is authentic or if its authenticity can be plausibly denied. We make two key contributions focusing on the high-precision, low-recall setting against efficient (i.e., compute-restricted) adversaries. First, we demonstrate that our calibrated resynthesis method is the most reliable approach for verifying authentic samples while maintaining controllable, low FPRs. Second, we show that our method achieves adversarial robustness against efficient adversaries, whereas prior methods are easily evaded under identical compute budgets. Our approach supports multiple modalities and leverages state-of-the-art inversion techniques.
>
---
#### [new 054] Robust Multi-view Camera Calibration from Dense Matches
- **分类: cs.CV**

- **简介: 该论文属多视图几何任务，旨在提升强径向畸变相机的鲁棒标定与位姿估计。针对SfM流程，提出两项改进：（1）优化密集匹配点的子采样策略；（2）设计增量式视图添加准则。实验表明其显著提升精度（79.9% vs 40.4%）。**

- **链接: [https://arxiv.org/pdf/2512.15608v1](https://arxiv.org/pdf/2512.15608v1)**

> **作者:** Johannes Hägerlind; Bao-Long Tran; Urs Waldmann; Per-Erik Forssén
>
> **备注:** This paper has been accepted for publication at the 21st International Conference on Computer Vision Theory and Applications (VISAPP 2026). Conference website: https://visapp.scitevents.org
>
> **摘要:** Estimating camera intrinsics and extrinsics is a fundamental problem in computer vision, and while advances in structure-from-motion (SfM) have improved accuracy and robustness, open challenges remain. In this paper, we introduce a robust method for pose estimation and calibration. We consider a set of rigid cameras, each observing the scene from a different perspective, which is a typical camera setup in animal behavior studies and forensic analysis of surveillance footage. Specifically, we analyse the individual components in a structure-from-motion (SfM) pipeline, and identify design choices that improve accuracy. Our main contributions are: (1) we investigate how to best subsample the predicted correspondences from a dense matcher to leverage them in the estimation process. (2) We investigate selection criteria for how to add the views incrementally. In a rigorous quantitative evaluation, we show the effectiveness of our changes, especially for cameras with strong radial distortion (79.9% ours vs. 40.4 vanilla VGGT). Finally, we demonstrate our correspondence subsampling in a global SfM setting where we initialize the poses using VGGT. The proposed pipeline generalizes across a wide range of camera setups, and could thus become a useful tool for animal behavior and forensic analysis.
>
---
#### [new 055] EagleVision: A Dual-Stage Framework with BEV-grounding-based Chain-of-Thought for Spatial Intelligence
- **分类: cs.CV**

- **简介: 该论文提出EagleVision框架，解决空间智能中三维一致性弱、视角单一、推理链不可追溯等问题。它分两阶段：宏观感知（SPF-DPP选关键帧）和微观验证（BEV姿态查询+强化学习），实现可解释、可验证的空间链式推理。**

- **链接: [https://arxiv.org/pdf/2512.15160v1](https://arxiv.org/pdf/2512.15160v1)**

> **作者:** Jiaxu Wan; Xu Wang; Mengwei Xie; Hang Zhang; Mu Xu; Yang Han; Hong Zhang; Ding Yuan; Yifan Yang
>
> **备注:** 13 pages, 7 figures, 6 tables
>
> **摘要:** Recent spatial intelligence approaches typically attach 3D cues to 2D reasoning pipelines or couple MLLMs with black-box reconstruction modules, leading to weak spatial consistency, limited viewpoint diversity, and evidence chains that cannot be traced back to supporting views. Frameworks for "thinking with images" (e.g., ChatGPT-o3 and DeepEyes) show that stepwise multimodal reasoning can emerge by interleaving hypothesis formation with active acquisition of visual evidence, but they do not address three key challenges in spatial Chain-of-Thought (CoT): building global space perception under strict token budgets, explicitly associating 3D hypotheses with video frames for verification, and designing spatially grounded rewards for reinforcement learning. To address these issues, we present EagleVision, a dual-stage framework for progressive spatial cognition through macro perception and micro verification. In the macro perception stage, EagleVision employs a semantics-perspective-fusion determinantal point process (SPF-DPP) to select a compact set of geometry- and semantics-aware keyframes from long videos under a fixed token budget. In the micro verification stage, we formalize spatial CoT as BEV-grounded pose querying: the agent iteratively predicts poses on a BEV plane, retrieves the nearest real frames, and is trained purely by reinforcement learning with a spatial grounding reward that scores the consistency between predicted poses and observed views. On VSI-Bench, EagleVision achieves state-of-the-art performance among open-source vision-language models, demonstrating strong and generalizable spatial understanding.
>
---
#### [new 056] Puzzle Curriculum GRPO for Vision-Centric Reasoning
- **分类: cs.CV**

- **简介: 该论文属视觉语言模型（VLM）的强化学习后训练任务，旨在解决GRPO中依赖人工标注、奖励稀疏平坦、推理与答案不一致三大问题。提出无监督的Puzzle Curriculum GRPO（PC-GRPO），引入自监督拼图环境、难度感知课程学习和推理-答案一致性奖励机制，提升视觉推理质量与稳定性。**

- **链接: [https://arxiv.org/pdf/2512.14944v1](https://arxiv.org/pdf/2512.14944v1)**

> **作者:** Ahmadreza Jeddi; Hakki Can Karaimer; Hue Nguyen; Zhongling Wang; Ke Zhao; Javad Rajabi; Ran Zhang; Raghav Goyal; Babak Taati; Radek Grzeszczuk
>
> **备注:** Project page: https://pcgrpo.github.io
>
> **摘要:** Recent reinforcement learning (RL) approaches like outcome-supervised GRPO have advanced chain-of-thought reasoning in Vision Language Models (VLMs), yet key issues linger: (i) reliance on costly and noisy hand-curated annotations or external verifiers; (ii) flat and sparse reward schemes in GRPO; and (iii) logical inconsistency between a chain's reasoning and its final answer. We present Puzzle Curriculum GRPO (PC-GRPO), a supervision-free recipe for RL with Verifiable Rewards (RLVR) that strengthens visual reasoning in VLMs without annotations or external verifiers. PC-GRPO replaces labels with three self-supervised puzzle environments: PatchFit, Rotation (with binary rewards) and Jigsaw (with graded partial credit mitigating reward sparsity). To counter flat rewards and vanishing group-relative advantages, we introduce a difficulty-aware curriculum that dynamically weights samples and peaks at medium difficulty. We further monitor Reasoning-Answer Consistency (RAC) during post-training: mirroring reports for vanilla GRPO in LLMs, RAC typically rises early then degrades; our curriculum delays this decline, and consistency-enforcing reward schemes further boost RAC. RAC correlates with downstream accuracy. Across diverse benchmarks and on Qwen-7B and Qwen-3B backbones, PC-GRPO improves reasoning quality, training stability, and end-task accuracy, offering a practical path to scalable, verifiable, and interpretable RL post-training for VLMs.
>
---
#### [new 057] Evaluating the Capability of Video Question Generation for Expert Knowledge Elicitation
- **分类: cs.CV; cs.AI**

- **简介: 该论文属视频问答中的视频问题生成（VQG）任务，旨在评估模型生成问题以有效激发专家隐性知识的能力。作者构建新数据集EgoExoAsk，提出基于问答检索的量化评估协议，验证模型上下文利用能力与问题启发性的正相关。**

- **链接: [https://arxiv.org/pdf/2512.15006v1](https://arxiv.org/pdf/2512.15006v1)**

> **作者:** Huaying Zhang; Atsushi Hashimoto; Tosho Hirasawa
>
> **备注:** WACV 2026 accepted
>
> **摘要:** Skilled human interviewers can extract valuable information from experts. This raises a fundamental question: what makes some questions more effective than others? To address this, a quantitative evaluation of question-generation models is essential. Video question generation (VQG) is a topic for video question answering (VideoQA), where questions are generated for given answers. Their evaluation typically focuses on the ability to answer questions, rather than the quality of generated questions. In contrast, we focus on the question quality in eliciting unseen knowledge from human experts. For a continuous improvement of VQG models, we propose a protocol that evaluates the ability by simulating question-answering communication with experts using a question-to-answer retrieval. We obtain the retriever by constructing a novel dataset, EgoExoAsk, which comprises 27,666 QA pairs generated from Ego-Exo4D's expert commentary annotation. The EgoExoAsk training set is used to obtain the retriever, and the benchmark is constructed on the validation set with Ego-Exo4D video segments. Experimental results demonstrate our metric reasonably aligns with question generation settings: models accessing richer context are evaluated better, supporting that our protocol works as intended. The EgoExoAsk dataset is available in https://github.com/omron-sinicx/VQG4ExpertKnowledge .
>
---
#### [new 058] InpaintDPO: Mitigating Spatial Relationship Hallucinations in Foreground-conditioned Inpainting via Diverse Preference Optimization
- **分类: cs.CV**

- **简介: 该论文针对前景条件图像修复中前景与背景空间关系幻觉（如尺度、位置、视角不合理）问题，提出InpaintDPO框架。首创将DPO引入该任务，通过MaskDPO、条件非对称偏好优化和共享共性偏好优化，提升空间合理性与边界一致性。**

- **链接: [https://arxiv.org/pdf/2512.15644v1](https://arxiv.org/pdf/2512.15644v1)**

> **作者:** Qirui Li; Yizhe Tang; Ran Yi; Guangben Lu; Fangyuan Zou; Peng Shu; Huan Yu; Jie Jiang
>
> **摘要:** Foreground-conditioned inpainting, which aims at generating a harmonious background for a given foreground subject based on the text prompt, is an important subfield in controllable image generation. A common challenge in current methods, however, is the occurrence of Spatial Relationship Hallucinations between the foreground subject and the generated background, including inappropriate scale, positional relationships, and viewpoints. Critically, the subjective nature of spatial rationality makes it challenging to quantify, hindering the use of traditional reward-based RLHF methods. To address this issue, we propose InpaintDPO, the first Direct Preference Optimization (DPO) based framework dedicated to spatial rationality in foreground-conditioned inpainting, ensuring plausible spatial relationships between foreground and background elements. To resolve the gradient conflicts in standard DPO caused by identical foreground in win-lose pairs, we propose MaskDPO, which confines preference optimization exclusively to the background to enhance background spatial relationships, while retaining the inpainting loss in the foreground region for robust foreground preservation. To enhance coherence at the foreground-background boundary, we propose Conditional Asymmetric Preference Optimization, which samples pairs with differentiated cropping operations and applies global preference optimization to promote contextual awareness and enhance boundary coherence. Finally, based on the observation that winning samples share a commonality in plausible spatial relationships, we propose Shared Commonality Preference Optimization to enhance the model's understanding of spatial commonality across high-quality winning samples, further promoting shared spatial rationality.
>
---
#### [new 059] Assessing the Visual Enumeration Abilities of Specialized Counting Architectures and Vision-Language Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于视觉计数任务，旨在评估专用计数模型与多模态大模型（VLMs）在开放场景下的物体数量估计能力。研究对比二者在多个数据集上的性能，发现VLMs经中间表征提示后可媲美甚至超越专用模型，但仍难以应对复杂场景。**

- **链接: [https://arxiv.org/pdf/2512.15254v1](https://arxiv.org/pdf/2512.15254v1)**

> **作者:** Kuinan Hou; Jing Mi; Marco Zorzi; Lamberto Ballan; Alberto Testolin
>
> **摘要:** Counting the number of items in a visual scene remains a fundamental yet challenging task in computer vision. Traditional approaches to solving this problem rely on domain-specific counting architectures, which are trained using datasets annotated with a predefined set of object categories. However, recent progress in creating large-scale multimodal vision-language models (VLMs) suggests that these domain-general architectures may offer a flexible alternative for open-set object counting. In this study, we therefore systematically compare the performance of state-of-the-art specialized counting architectures against VLMs on two popular counting datasets, as well as on a novel benchmark specifically created to have a finer-grained control over the visual properties of test images. Our findings show that most VLMs can approximately enumerate the number of items in a visual scene, matching or even surpassing the performance of specialized computer vision architectures. Notably, enumeration accuracy significantly improves when VLMs are prompted to generate intermediate representations (i.e., locations and verbal labels) of each object to be counted. Nevertheless, none of the models can reliably count the number of objects in complex visual scenes, showing that further research is still needed to create AI systems that can reliably deploy counting procedures in realistic environments.
>
---
#### [new 060] Stylized Synthetic Augmentation further improves Corruption Robustness
- **分类: cs.CV; cs.LG**

- **简介: 该论文属图像分类任务，旨在提升模型对常见图像腐蚀的鲁棒性。提出“风格化合成增强”方法：将神经风格迁移应用于合成图像，并系统分析其与规则增强（如TrivialAugment）的协同效应，在CIFAR-10-C等基准上达SOTA鲁棒精度。**

- **链接: [https://arxiv.org/pdf/2512.15675v1](https://arxiv.org/pdf/2512.15675v1)**

> **作者:** Georg Siedel; Rojan Regmi; Abhirami Anand; Weijia Shao; Silvia Vock; Andrey Morozov
>
> **备注:** Accepted at VISAPP 2026 conference
>
> **摘要:** This paper proposes a training data augmentation pipeline that combines synthetic image data with neural style transfer in order to address the vulnerability of deep vision models to common corruptions. We show that although applying style transfer on synthetic images degrades their quality with respect to the common FID metric, these images are surprisingly beneficial for model training. We conduct a systematic empirical analysis of the effects of both augmentations and their key hyperparameters on the performance of image classifiers. Our results demonstrate that stylization and synthetic data complement each other well and can be combined with popular rule-based data augmentation techniques such as TrivialAugment, while not working with others. Our method achieves state-of-the-art corruption robustness on several small-scale image classification benchmarks, reaching 93.54%, 74.9% and 50.86% robust accuracy on CIFAR-10-C, CIFAR-100-C and TinyImageNet-C, respectively
>
---
#### [new 061] Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting
- **分类: cs.CV**

- **简介: 该论文属3D场景重建任务，旨在解决传统前馈式3D高斯泼溅中像素对齐的刚性网格导致的原始体素分布粗糙、效率低、质量差问题。提出“离网”架构，通过多分辨率关键点式检测器自适应定位子像素级高斯原语，端到端自监督训练，显著提升渲染质量与效率。**

- **链接: [https://arxiv.org/pdf/2512.15508v1](https://arxiv.org/pdf/2512.15508v1)**

> **作者:** Arthur Moreau; Richard Shaw; Michal Nazarczuk; Jisu Shin; Thomas Tanay; Zhensong Zhang; Songcen Xu; Eduardo Pérez-Pellitero
>
> **摘要:** Feed-forward 3D Gaussian Splatting (3DGS) models enable real-time scene generation but are hindered by suboptimal pixel-aligned primitive placement, which relies on a dense, rigid grid and limits both quality and efficiency. We introduce a new feed-forward architecture that detects 3D Gaussian primitives at a sub-pixel level, replacing the pixel grid with an adaptive, "Off The Grid" distribution. Inspired by keypoint detection, our multi-resolution decoder learns to distribute primitives across image patches. This module is trained end-to-end with a 3D reconstruction backbone using self-supervised learning. Our resulting pose-free model generates photorealistic scenes in seconds, achieving state-of-the-art novel view synthesis for feed-forward models. It outperforms competitors while using far fewer primitives, demonstrating a more accurate and efficient allocation that captures fine details and reduces artifacts. Moreover, we observe that by learning to render 3D Gaussians, our 3D reconstruction backbone improves camera pose estimation, suggesting opportunities to train these foundational models without labels.
>
---
#### [new 062] GRAN-TED: Generating Robust, Aligned, and Nuanced Text Embedding for Diffusion Models
- **分类: cs.CV**

- **简介: 该论文面向文本到图像/视频扩散模型，解决文本编码器评估难、适配差的问题。提出TED-6K轻量文本基准用于高效评估，并设计两阶段训练范式（MLLM微调+层加权）构建更鲁棒、对齐、细腻的GRAN-TED文本编码器。**

- **链接: [https://arxiv.org/pdf/2512.15560v1](https://arxiv.org/pdf/2512.15560v1)**

> **作者:** Bozhou Li; Sihan Yang; Yushuo Guan; Ruichuan An; Xinlong Chen; Yang Shi; Pengfei Wan; Wentao Zhang; Yuanxing zhang
>
> **摘要:** The text encoder is a critical component of text-to-image and text-to-video diffusion models, fundamentally determining the semantic fidelity of the generated content. However, its development has been hindered by two major challenges: the lack of an efficient evaluation framework that reliably predicts downstream generation performance, and the difficulty of effectively adapting pretrained language models for visual synthesis. To address these issues, we introduce GRAN-TED, a paradigm to Generate Robust, Aligned, and Nuanced Text Embeddings for Diffusion models. Our contribution is twofold. First, we propose TED-6K, a novel text-only benchmark that enables efficient and robust assessment of an encoder's representational quality without requiring costly end-to-end model training. We demonstrate that performance on TED-6K, standardized via a lightweight, unified adapter, strongly correlates with an encoder's effectiveness in downstream generation tasks. Second, guided by this validated framework, we develop a superior text encoder using a novel two-stage training paradigm. This process involves an initial fine-tuning stage on a Multimodal Large Language Model for better visual representation, followed by a layer-wise weighting method to extract more nuanced and potent text features. Our experiments show that the resulting GRAN-TED encoder not only achieves state-of-the-art performance on TED-6K but also leads to demonstrable performance gains in text-to-image and text-to-video generation. Our code is available at the following link: https://anonymous.4open.science/r/GRAN-TED-4FCC/.
>
---
#### [new 063] SynthSeg-Agents: Multi-Agent Synthetic Data Generation for Zero-Shot Weakly Supervised Semantic Segmentation
- **分类: cs.CV**

- **简介: 该论文提出SynthSeg-Agents，面向零样本弱监督语义分割（ZSWSSS）任务，旨在仅用图像级标签、无需任何真实图像即实现像素级分割。其通过LLM驱动的双智能体框架——自优化提示生成与VLM图像合成——全自动构建高质量合成训练数据，并经CLIP筛选与ViT重标注，实现无真实图像监督的端到端训练。**

- **链接: [https://arxiv.org/pdf/2512.15310v1](https://arxiv.org/pdf/2512.15310v1)**

> **作者:** Wangyu Wu; Zhenhong Chen; Xiaowei Huang; Fei Ma; Jimin Xiao
>
> **摘要:** Weakly Supervised Semantic Segmentation (WSSS) with image level labels aims to produce pixel level predictions without requiring dense annotations. While recent approaches have leveraged generative models to augment existing data, they remain dependent on real world training samples. In this paper, we introduce a novel direction, Zero Shot Weakly Supervised Semantic Segmentation (ZSWSSS), and propose SynthSeg Agents, a multi agent framework driven by Large Language Models (LLMs) to generate synthetic training data entirely without real images. SynthSeg Agents comprises two key modules, a Self Refine Prompt Agent and an Image Generation Agent. The Self Refine Prompt Agent autonomously crafts diverse and semantically rich image prompts via iterative refinement, memory mechanisms, and prompt space exploration, guided by CLIP based similarity and nearest neighbor diversity filtering. These prompts are then passed to the Image Generation Agent, which leverages Vision Language Models (VLMs) to synthesize candidate images. A frozen CLIP scoring model is employed to select high quality samples, and a ViT based classifier is further trained to relabel the entire synthetic dataset with improved semantic precision. Our framework produces high quality training data without any real image supervision. Experiments on PASCAL VOC 2012 and COCO 2014 show that SynthSeg Agents achieves competitive performance without using real training images. This highlights the potential of LLM driven agents in enabling cost efficient and scalable semantic segmentation.
>
---
#### [new 064] GateFusion: Hierarchical Gated Cross-Modal Fusion for Active Speaker Detection
- **分类: cs.CV**

- **简介: 该论文面向主动说话人检测（ASD）任务，旨在精准定位视频中每帧的说话人。针对 late fusion 忽视细粒度跨模态交互的问题，提出 GateFusion：基于分层门控跨模态融合（HiGate）的架构，并引入掩码对齐损失与过正惩罚两项辅助目标，显著提升多基准性能。**

- **链接: [https://arxiv.org/pdf/2512.15707v1](https://arxiv.org/pdf/2512.15707v1)**

> **作者:** Yu Wang; Juhyung Ha; Frangil M. Ramirez; Yuchen Wang; David J. Crandall
>
> **备注:** accepted by WACV 2026
>
> **摘要:** Active Speaker Detection (ASD) aims to identify who is currently speaking in each frame of a video. Most state-of-the-art approaches rely on late fusion to combine visual and audio features, but late fusion often fails to capture fine-grained cross-modal interactions, which can be critical for robust performance in unconstrained scenarios. In this paper, we introduce GateFusion, a novel architecture that combines strong pretrained unimodal encoders with a Hierarchical Gated Fusion Decoder (HiGate). HiGate enables progressive, multi-depth fusion by adaptively injecting contextual features from one modality into the other at multiple layers of the Transformer backbone, guided by learnable, bimodally-conditioned gates. To further strengthen multimodal learning, we propose two auxiliary objectives: Masked Alignment Loss (MAL) to align unimodal outputs with multimodal predictions, and Over-Positive Penalty (OPP) to suppress spurious video-only activations. GateFusion establishes new state-of-the-art results on several challenging ASD benchmarks, achieving 77.8% mAP (+9.4%), 86.1% mAP (+2.9%), and 96.1% mAP (+0.5%) on Ego4D-ASD, UniTalk, and WASD benchmarks, respectively, and delivering competitive performance on AVA-ActiveSpeaker. Out-of-domain experiments demonstrate the generalization of our model, while comprehensive ablations show the complementary benefits of each component.
>
---
#### [new 065] SkyCap: Bitemporal VHR Optical-SAR Quartets for Amplitude Change Detection and Foundation-Model Evaluation
- **分类: cs.CV**

- **简介: 该论文面向VHR SAR幅度变化检测（ACD）任务，解决SAR标注难、光学易受云干扰问题；构建首个光学-SAR四元组数据集SkyCap，提出光学到SAR标签迁移方法，并评估基础模型在ACD上的性能，发现光学模型经适配预处理可优于SAR专用模型。**

- **链接: [https://arxiv.org/pdf/2512.14755v1](https://arxiv.org/pdf/2512.14755v1)**

> **作者:** Paul Weinmann; Ferdinand Schenck; Martin Šiklar
>
> **备注:** 8 pages, 0 figures. Accepted at Advances in Representation Learning for Earth Observation (REO) at EurIPS 2025
>
> **摘要:** Change detection for linear infrastructure monitoring requires reliable high-resolution data and regular acquisition cadence. Optical very-high-resolution (VHR) imagery is interpretable and straightforward to label, but clouds break this cadence. Synthetic Aperture Radar (SAR) enables all-weather acquisitions, yet is difficult to annotate. We introduce SkyCap, a bitemporal VHR optical-SAR dataset constructed by archive matching and co-registration of (optical) SkySat and Capella Space (SAR) scenes. We utilize optical-to-SAR label transfer to obtain SAR amplitude change detection (ACD) labels without requiring SAR-expert annotations. We perform continued pretraining of SARATR-X on our SAR data and benchmark the resulting SAR-specific foundation models (FMs) together with SARATR-X against optical FMs on SkyCap under different preprocessing choices. Among evaluated models, MTP(ViT-B+RVSA), an optical FM, with dB+Z-score preprocessing attains the best result (F1$_c$ = 45.06), outperforming SAR-specific FMs further pretrained directly on Capella data. We observe strong sensitivity to preprocessing alignment with pretraining statistics, and the ranking of optical models on optical change detection does not transfer one-to-one to SAR ACD. To our knowledge, this is the first evaluation of foundation models on VHR SAR ACD.
>
---
#### [new 066] RUMPL: Ray-Based Transformers for Universal Multi-View 2D to 3D Human Pose Lifting
- **分类: cs.CV**

- **简介: 该论文属2D到3D人体姿态估计任务，旨在解决多视角下因遮挡、投影歧义及真实数据稀缺导致的泛化性差问题。提出RUMPL：基于射线的Transformer模型，用相机无关的3D射线表征2D关键点，支持任意多视角配置；引入视图融合Transformer提升一致性。**

- **链接: [https://arxiv.org/pdf/2512.15488v1](https://arxiv.org/pdf/2512.15488v1)**

> **作者:** Seyed Abolfazl Ghasemzadeh; Alexandre Alahi; Christophe De Vleeschouwer
>
> **摘要:** Estimating 3D human poses from 2D images remains challenging due to occlusions and projective ambiguity. Multi-view learning-based approaches mitigate these issues but often fail to generalize to real-world scenarios, as large-scale multi-view datasets with 3D ground truth are scarce and captured under constrained conditions. To overcome this limitation, recent methods rely on 2D pose estimation combined with 2D-to-3D pose lifting trained on synthetic data. Building on our previous MPL framework, we propose RUMPL, a transformer-based 3D pose lifter that introduces a 3D ray-based representation of 2D keypoints. This formulation makes the model independent of camera calibration and the number of views, enabling universal deployment across arbitrary multi-view configurations without retraining or fine-tuning. A new View Fusion Transformer leverages learned fused-ray tokens to aggregate information along rays, further improving multi-view consistency. Extensive experiments demonstrate that RUMPL reduces MPJPE by up to 53% compared to triangulation and over 60% compared to transformer-based image-representation baselines. Results on new benchmarks, including in-the-wild multi-view and multi-person datasets, confirm its robustness and scalability. The framework's source code is available at https://github.com/aghasemzadeh/OpenRUMPL
>
---
#### [new 067] Persistent feature reconstruction of resident space objects (RSOs) within inverse synthetic aperture radar (ISAR) images
- **分类: cs.CV; eess.SP**

- **简介: 该论文属空间目标识别任务，旨在提升近地轨道空间物体（RSO）的态势感知能力。针对ISAR图像中结构特征易受噪声干扰、难以稳定检测的问题，提出基于梯度比边缘检测与双权重Hough变换的序列特征检测与跟踪方法，实现高精度线性结构重建与阴影等鲁棒特征识别。**

- **链接: [https://arxiv.org/pdf/2512.15618v1](https://arxiv.org/pdf/2512.15618v1)**

> **作者:** Morgan Coe; Gruffudd Jones; Leah-Nani Alconcel; Marina Gashinova
>
> **摘要:** With the rapidly growing population of resident space objects (RSOs) in the near-Earth space environment, detailed information about their condition and capabilities is needed to provide Space Domain Awareness (SDA). Space-based sensing will enable inspection of RSOs at shorter ranges, independent of atmospheric effects, and from all aspects. The use of a sub-THz inverse synthetic aperture radar (ISAR) imaging and sensing system for SDA has been proposed in previous work, demonstrating the achievement of sub-cm image resolution at ranges of up to 100 km. This work focuses on recognition of external structures by use of sequential feature detection and tracking throughout the aligned ISAR images of the satellites. The Hough transform is employed to detect linear features, which are tracked throughout the sequence. ISAR imagery is generated via a metaheuristic simulator capable of modelling encounters for a variety of deployment scenarios. Initial frame-to-frame alignment is achieved through a series of affine transformations to facilitate later association between image features. A gradient-by-ratio method is used for edge detection within individual ISAR images, and edge magnitude and direction are subsequently used to inform a double-weighted Hough transform to detect features with high accuracy. Feature evolution during sequences of frames is analysed. It is shown that the use of feature tracking within sequences with the proposed approach will increase confidence in feature detection and classification, and an example use-case of robust detection of shadowing as a feature is presented.
>
---
#### [new 068] AquaDiff: Diffusion-Based Underwater Image Enhancement for Addressing Color Distortion
- **分类: cs.CV**

- **简介: 该论文属于水下图像增强任务，旨在解决水下图像因光吸收与散射导致的色偏、低对比度和细节丢失问题。提出基于扩散模型的AquaDiff框架，融合色度先验引导的色彩补偿、跨注意力条件扩散及多分辨率特征提取，并设计跨域一致性损失优化增强效果。**

- **链接: [https://arxiv.org/pdf/2512.14760v1](https://arxiv.org/pdf/2512.14760v1)**

> **作者:** Afrah Shaahid; Muzammil Behzad
>
> **摘要:** Underwater images are severely degraded by wavelength-dependent light absorption and scattering, resulting in color distortion, low contrast, and loss of fine details that hinder vision-based underwater applications. To address these challenges, we propose AquaDiff, a diffusion-based underwater image enhancement framework designed to correct chromatic distortions while preserving structural and perceptual fidelity. AquaDiff integrates a chromatic prior-guided color compensation strategy with a conditional diffusion process, where cross-attention dynamically fuses degraded inputs and noisy latent states at each denoising step. An enhanced denoising backbone with residual dense blocks and multi-resolution attention captures both global color context and local details. Furthermore, a novel cross-domain consistency loss jointly enforces pixel-level accuracy, perceptual similarity, structural integrity, and frequency-domain fidelity. Extensive experiments on multiple challenging underwater benchmarks demonstrate that AquaDiff provides good results as compared to the state-of-the-art traditional, CNN-, GAN-, and diffusion-based methods, achieving superior color correction and competitive overall image quality across diverse underwater conditions.
>
---
#### [new 069] Tracking spatial temporal details in ultrasound long video via wavelet analysis and memory bank
- **分类: cs.CV; cs.AI**

- **简介: 该论文面向超声长视频的器官/病灶分割与跟踪任务，旨在解决低对比度、噪声干扰导致的小目标丢失和边界模糊问题。提出基于小波分析与记忆库的网络MWNet，融合高频细节、多尺度频域特征，并利用跨注意力记忆机制实现长时序目标跟踪。**

- **链接: [https://arxiv.org/pdf/2512.15066v1](https://arxiv.org/pdf/2512.15066v1)**

> **作者:** Chenxiao Zhang; Runshi Zhang; Junchen Wang
>
> **备注:** Chenxiao Zhang and Runshi Zhang contributed equally to this work. 14 pages, 11 figures
>
> **摘要:** Medical ultrasound videos are widely used for medical inspections, disease diagnosis and surgical planning. High-fidelity lesion area and target organ segmentation constitutes a key component of the computer-assisted surgery workflow. The low contrast levels and noisy backgrounds of ultrasound videos cause missegmentation of organ boundary, which may lead to small object losses and increase boundary segmentation errors. Object tracking in long videos also remains a significant research challenge. To overcome these challenges, we propose a memory bank-based wavelet filtering and fusion network, which adopts an encoder-decoder structure to effectively extract fine-grained detailed spatial features and integrate high-frequency (HF) information. Specifically, memory-based wavelet convolution is presented to simultaneously capture category, detailed information and utilize adjacent information in the encoder. Cascaded wavelet compression is used to fuse multiscale frequency-domain features and expand the receptive field within each convolutional layer. A long short-term memory bank using cross-attention and memory compression mechanisms is designed to track objects in long video. To fully utilize the boundary-sensitive HF details of feature maps, an HF-aware feature fusion module is designed via adaptive wavelet filters in the decoder. In extensive benchmark tests conducted on four ultrasound video datasets (two thyroid nodule, the thyroid gland, the heart datasets) compared with the state-of-the-art methods, our method demonstrates marked improvements in segmentation metrics. In particular, our method can more accurately segment small thyroid nodules, demonstrating its effectiveness for cases involving small ultrasound objects in long video. The code is available at https://github.com/XiAooZ/MWNet.
>
---
#### [new 070] VLIC: Vision-Language Models As Perceptual Judges for Human-Aligned Image Compression
- **分类: cs.CV**

- **简介: 该论文提出VLIC，一种基于视觉-语言模型（VLM）零样本二元判断的扩散式图像压缩方法，旨在解决传统失真度量（如MSE）与人类感知不一致的问题。通过直接利用VLM作为感知裁判进行后训练，提升压缩结果的人类偏好对齐性。**

- **链接: [https://arxiv.org/pdf/2512.15701v1](https://arxiv.org/pdf/2512.15701v1)**

> **作者:** Kyle Sargent; Ruiqi Gao; Philipp Henzler; Charles Herrmann; Aleksander Holynski; Li Fei-Fei; Jiajun Wu; Jason Zhang
>
> **备注:** 14 pages, 8 figures
>
> **摘要:** Evaluations of image compression performance which include human preferences have generally found that naive distortion functions such as MSE are insufficiently aligned to human perception. In order to align compression models to human perception, prior work has employed differentiable perceptual losses consisting of neural networks calibrated on large-scale datasets of human psycho-visual judgments. We show that, surprisingly, state-of-the-art vision-language models (VLMs) can replicate binary human two-alternative forced choice (2AFC) judgments zero-shot when asked to reason about the differences between pairs of images. Motivated to exploit the powerful zero-shot visual reasoning capabilities of VLMs, we propose Vision-Language Models for Image Compression (VLIC), a diffusion-based image compression system designed to be post-trained with binary VLM judgments. VLIC leverages existing techniques for diffusion model post-training with preferences, rather than distilling the VLM judgments into a separate perceptual loss network. We show that calibrating this system on VLM judgments produces competitive or state-of-the-art performance on human-aligned visual compression depending on the dataset, according to perceptual metrics and large-scale user studies. We additionally conduct an extensive analysis of the VLM-based reward design and training procedure and share important insights. More visuals are available at https://kylesargent.github.io/vlic
>
---
#### [new 071] SemanticBridge -- A Dataset for 3D Semantic Segmentation of Bridges and Domain Gap Analysis
- **分类: cs.CV**

- **简介: 该论文面向桥梁3D语义分割任务，旨在解决因传感器差异导致的域间性能下降问题。作者构建了SemanticBridge数据集，涵盖多国桥梁的高分辨率3D扫描与精细语义标注，并评估了三种SOTA模型，量化了最大11.4% mIoU的域差距。**

- **链接: [https://arxiv.org/pdf/2512.15369v1](https://arxiv.org/pdf/2512.15369v1)**

> **作者:** Maximilian Kellner; Mariana Ferrandon Cervantes; Yuandong Pan; Ruodan Lu; Ioannis Brilakis; Alexander Reiterer
>
> **摘要:** We propose a novel dataset that has been specifically designed for 3D semantic segmentation of bridges and the domain gap analysis caused by varying sensors. This addresses a critical need in the field of infrastructure inspection and maintenance, which is essential for modern society. The dataset comprises high-resolution 3D scans of a diverse range of bridge structures from various countries, with detailed semantic labels provided for each. Our initial objective is to facilitate accurate and automated segmentation of bridge components, thereby advancing the structural health monitoring practice. To evaluate the effectiveness of existing 3D deep learning models on this novel dataset, we conduct a comprehensive analysis of three distinct state-of-the-art architectures. Furthermore, we present data acquired through diverse sensors to quantify the domain gap resulting from sensor variations. Our findings indicate that all architectures demonstrate robust performance on the specified task. However, the domain gap can potentially lead to a decline in the performance of up to 11.4% mIoU.
>
---
#### [new 072] On the Effectiveness of Textual Prompting with Lightweight Fine-Tuning for SAM3 Remote Sensing Segmentation
- **分类: cs.CV**

- **简介: 该论文研究遥感图像分割任务，旨在解决标注数据少、基础模型（SAM3）与遥感影像语义对齐差的问题。作者评估文本、几何及混合提示策略，结合轻量微调，在四类目标上验证效果，发现几何引导的混合提示最优，且少量几何标注即可显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.15564v1](https://arxiv.org/pdf/2512.15564v1)**

> **作者:** Roni Blushtein-Livnon; Osher Rafaeli; David Ioffe; Amir Boger; Karen Sandberg Esquenazi; Tal Svoray
>
> **摘要:** Remote sensing (RS) image segmentation is constrained by the limited availability of annotated data and a gap between overhead imagery and natural images used to train foundational models. This motivates effective adaptation under limited supervision. SAM3 concept-driven framework generates masks from textual prompts without requiring task-specific modifications, which may enable this adaptation. We evaluate SAM3 for RS imagery across four target types, comparing textual, geometric, and hybrid prompting strategies, under lightweight fine-tuning scales with increasing supervision, alongside zero-shot inference. Results show that combining semantic and geometric cues yields the highest performance across targets and metrics. Text-only prompting exhibits the lowest performance, with marked score gaps for irregularly shaped targets, reflecting limited semantic alignment between SAM3 textual representations and their overhead appearances. Nevertheless, textual prompting with light fine-tuning offers a practical performance-effort trade-off for geometrically regular and visually salient targets. Across targets, performance improves between zero-shot inference and fine-tuning, followed by diminishing returns as the supervision scale increases. Namely, a modest geometric annotation effort is sufficient for effective adaptation. A persistent gap between Precision and IoU further indicates that under-segmentation and boundary inaccuracies remain prevalent error patterns in RS tasks, particularly for irregular and less prevalent targets.
>
---
#### [new 073] Gaussian Pixel Codec Avatars: A Hybrid Representation for Efficient Rendering
- **分类: cs.CV; cs.GR**

- **简介: 该论文提出GPiCA，一种用于高效渲染的光真实感头像生成方法。旨在解决移动端实时渲染与高保真度难以兼顾的问题。工作包括：设计三角网格与各向异性3D高斯的混合表征，构建统一可微渲染管线，并通过多视角图像监督训练网络解码表情为网格、纹理和高斯集。**

- **链接: [https://arxiv.org/pdf/2512.15711v1](https://arxiv.org/pdf/2512.15711v1)**

> **作者:** Divam Gupta; Anuj Pahuja; Nemanja Bartolovic; Tomas Simon; Forrest Iandola; Giljoo Nam
>
> **备注:** Tech report
>
> **摘要:** We present Gaussian Pixel Codec Avatars (GPiCA), photorealistic head avatars that can be generated from multi-view images and efficiently rendered on mobile devices. GPiCA utilizes a unique hybrid representation that combines a triangle mesh and anisotropic 3D Gaussians. This combination maximizes memory and rendering efficiency while maintaining a photorealistic appearance. The triangle mesh is highly efficient in representing surface areas like facial skin, while the 3D Gaussians effectively handle non-surface areas such as hair and beard. To this end, we develop a unified differentiable rendering pipeline that treats the mesh as a semi-transparent layer within the volumetric rendering paradigm of 3D Gaussian Splatting. We train neural networks to decode a facial expression code into three components: a 3D face mesh, an RGBA texture, and a set of 3D Gaussians. These components are rendered simultaneously in a unified rendering engine. The networks are trained using multi-view image supervision. Our results demonstrate that GPiCA achieves the realism of purely Gaussian-based avatars while matching the rendering performance of mesh-based avatars.
>
---
#### [new 074] Visual-textual Dermatoglyphic Animal Biometrics: A First Case Study on Panthera tigris
- **分类: cs.CV**

- **简介: 该论文提出视觉-文本联合的皮肤纹路（dermatoglyphic）动物个体识别新任务，解决传统纯图像Re-ID可解释性差、数据稀缺问题。工作包括：构建虎纹 minutiae 标注数据集，设计文本-图像协同合成 pipeline 生成虚拟个体，实现可解释、跨模态的文本到视觉身份检索。**

- **链接: [https://arxiv.org/pdf/2512.14878v1](https://arxiv.org/pdf/2512.14878v1)**

> **作者:** Wenshuo Li; Majid Mirmehdi; Tilo Burghardt
>
> **摘要:** Biologists have long combined visuals with textual field notes to re-identify (Re-ID) animals. Contemporary AI tools automate this for species with distinctive morphological features but remain largely image-based. Here, we extend Re-ID methodologies by incorporating precise dermatoglyphic textual descriptors-an approach used in forensics but new to ecology. We demonstrate that these specialist semantics abstract and encode animal coat topology using human-interpretable language tags. Drawing on 84,264 manually labelled minutiae across 3,355 images of 185 tigers (Panthera tigris), we evaluate this visual-textual methodology, revealing novel capabilities for cross-modal identity retrieval. To optimise performance, we developed a text-image co-synthesis pipeline to generate 'virtual individuals', each comprising dozens of life-like visuals paired with dermatoglyphic text. Benchmarking against real-world scenarios shows this augmentation significantly boosts AI accuracy in cross-modal retrieval while alleviating data scarcity. We conclude that dermatoglyphic language-guided biometrics can overcome vision-only limitations, enabling textual-to-visual identity recovery underpinned by human-verifiable matchings. This represents a significant advance towards explainability in Re-ID and a language-driven unification of descriptive modalities in ecological monitoring.
>
---
#### [new 075] OccSTeP: Benchmarking 4D Occupancy Spatio-Temporal Persistence
- **分类: cs.CV**

- **简介: 该论文提出4D Occupancy Spatio-Temporal Persistence（OccSTeP）新任务，聚焦自动驾驶中鲁棒的时空场景理解，涵盖反应式与前瞻性预测。构建首个OccSTeP基准（含噪声/丢帧挑战），并设计无tokenizer的世界模型OccSTeP-WM，实现高效、鲁棒的在线4D占用预测。**

- **链接: [https://arxiv.org/pdf/2512.15621v1](https://arxiv.org/pdf/2512.15621v1)**

> **作者:** Yu Zheng; Jie Hu; Kailun Yang; Jiaming Zhang
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** Autonomous driving requires a persistent understanding of 3D scenes that is robust to temporal disturbances and accounts for potential future actions. We introduce a new concept of 4D Occupancy Spatio-Temporal Persistence (OccSTeP), which aims to address two tasks: (1) reactive forecasting: ''what will happen next'' and (2) proactive forecasting: "what would happen given a specific future action". For the first time, we create a new OccSTeP benchmark with challenging scenarios (e.g., erroneous semantic labels and dropped frames). To address this task, we propose OccSTeP-WM, a tokenizer-free world model that maintains a dense voxel-based scene state and incrementally fuses spatio-temporal context over time. OccSTeP-WM leverages a linear-complexity attention backbone and a recurrent state-space module to capture long-range spatial dependencies while continually updating the scene memory with ego-motion compensation. This design enables online inference and robust performance even when historical sensor input is missing or noisy. Extensive experiments prove the effectiveness of the OccSTeP concept and our OccSTeP-WM, yielding an average semantic mIoU of 23.70% (+6.56% gain) and occupancy IoU of 35.89% (+9.26% gain). The data and code will be open source at https://github.com/FaterYU/OccSTeP.
>
---
#### [new 076] ST-DETrack: Identity-Preserving Branch Tracking in Entangled Plant Canopies via Dual Spatiotemporal Evidence
- **分类: cs.CV**

- **简介: 该论文属植物表型分析中的分支跟踪任务，旨在解决密集缠绕冠层下分支身份碎片化问题。提出ST-DETrack网络，融合空间（几何先验）与时间（运动一致性）双解码器，并引入自适应门控与负向地性生物约束，实现从萌芽到开花的长期身份保持。**

- **链接: [https://arxiv.org/pdf/2512.15445v1](https://arxiv.org/pdf/2512.15445v1)**

> **作者:** Yueqianji Chen; Kevin Williams; John H. Doonan; Paolo Remagnino; Jo Hepworth
>
> **备注:** Under Review at IEEE Transactions on Image Processing
>
> **摘要:** Automated extraction of individual plant branches from time-series imagery is essential for high-throughput phenotyping, yet it remains computationally challenging due to non-rigid growth dynamics and severe identity fragmentation within entangled canopies. To overcome these stage-dependent ambiguities, we propose ST-DETrack, a spatiotemporal-fusion dual-decoder network designed to preserve branch identity from budding to flowering. Our architecture integrates a spatial decoder, which leverages geometric priors such as position and angle for early-stage tracking, with a temporal decoder that exploits motion consistency to resolve late-stage occlusions. Crucially, an adaptive gating mechanism dynamically shifts reliance between these spatial and temporal cues, while a biological constraint based on negative gravitropism mitigates vertical growth ambiguities. Validated on a Brassica napus dataset, ST-DETrack achieves a Branch Matching Accuracy (BMA) of 93.6%, significantly outperforming spatial and temporal baselines by 28.9 and 3.3 percentage points, respectively. These results demonstrate the method's robustness in maintaining long-term identity consistency amidst complex, dynamic plant architectures.
>
---
#### [new 077] Criticality Metrics for Relevance Classification in Safety Evaluation of Object Detection in Automated Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属自动驾驶安全评估任务，旨在解决物体检测中相关性（关键性）分类不准确的问题。作者系统综述并实证评估现有关键性指标，提出双向关键性评分与多指标聚合新策略，在DeepAccident数据集上将关键性分类准确率提升达100%。**

- **链接: [https://arxiv.org/pdf/2512.15181v1](https://arxiv.org/pdf/2512.15181v1)**

> **作者:** Jörg Gamerdinger; Sven Teufel; Stephan Amann; Oliver Bringmann
>
> **备注:** Accepted at IEEE ICVES 2025
>
> **摘要:** Ensuring safety is the primary objective of automated driving, which necessitates a comprehensive and accurate perception of the environment. While numerous performance evaluation metrics exist for assessing perception capabilities, incorporating safety-specific metrics is essential to reliably evaluate object detection systems. A key component for safety evaluation is the ability to distinguish between relevant and non-relevant objects - a challenge addressed by criticality or relevance metrics. This paper presents the first in-depth analysis of criticality metrics for safety evaluation of object detection systems. Through a comprehensive review of existing literature, we identify and assess a range of applicable metrics. Their effectiveness is empirically validated using the DeepAccident dataset, which features a variety of safety-critical scenarios. To enhance evaluation accuracy, we propose two novel application strategies: bidirectional criticality rating and multi-metric aggregation. Our approach demonstrates up to a 100% improvement in terms of criticality classification accuracy, highlighting its potential to significantly advance the safety evaluation of object detection systems in automated vehicles.
>
---
#### [new 078] Automated Motion Artifact Check for MRI (AutoMAC-MRI): An Interpretable Framework for Motion Artifact Detection and Severity Assessment
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出AutoMAC-MRI，解决MRI中运动伪影的自动检测与严重程度分级问题。属医学图像质量评估任务，采用监督对比学习构建可解释的特征空间，通过等级特异性亲和度分数实现细粒度、透明化的伪影分级，在5000+专家标注脑MRI上验证了准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2512.15315v1](https://arxiv.org/pdf/2512.15315v1)**

> **作者:** Antony Jerald; Dattesh Shanbhag; Sudhanya Chatterjee
>
> **摘要:** Motion artifacts degrade MRI image quality and increase patient recalls. Existing automated quality assessment methods are largely limited to binary decisions and provide little interpretability. We introduce AutoMAC-MRI, an explainable framework for grading motion artifacts across heterogeneous MR contrasts and orientations. The approach uses supervised contrastive learning to learn a discriminative representation of motion severity. Within this feature space, we compute grade-specific affinity scores that quantify an image's proximity to each motion grade, thereby making grade assignments transparent and interpretable. We evaluate AutoMAC-MRI on more than 5000 expert-annotated brain MRI slices spanning multiple contrasts and views. Experiments assessing affinity scores against expert labels show that the scores align well with expert judgment, supporting their use as an interpretable measure of motion severity. By coupling accurate grade detection with per-grade affinity scoring, AutoMAC-MRI enables inline MRI quality control, with the potential to reduce unnecessary rescans and improve workflow efficiency.
>
---
#### [new 079] Improving Pre-trained Segmentation Models using Post-Processing
- **分类: cs.CV; cs.AI**

- **简介: 该论文属医学图像分割任务，旨在解决预训练大模型在胶质瘤MRI分割中泛化差、误分割（如假阳性、标签错换、层间不连续）及计算资源消耗大等问题。提出自适应后处理技术，在BraTS 2025挑战中显著提升性能，推动轻量、公平、可持续的临床分割范式。**

- **链接: [https://arxiv.org/pdf/2512.14937v1](https://arxiv.org/pdf/2512.14937v1)**

> **作者:** Abhijeet Parida; Daniel Capellán-Martín; Zhifan Jiang; Nishad Kulkarni; Krithika Iyer; Austin Tapp; Syed Muhammad Anwar; María J. Ledesma-Carbayo; Marius George Linguraru
>
> **摘要:** Gliomas are the most common malignant brain tumors in adults and are among the most lethal. Despite aggressive treatment, the median survival rate is less than 15 months. Accurate multiparametric MRI (mpMRI) tumor segmentation is critical for surgical planning, radiotherapy, and disease monitoring. While deep learning models have improved the accuracy of automated segmentation, large-scale pre-trained models generalize poorly and often underperform, producing systematic errors such as false positives, label swaps, and slice discontinuities in slices. These limitations are further compounded by unequal access to GPU resources and the growing environmental cost of large-scale model training. In this work, we propose adaptive post-processing techniques to refine the quality of glioma segmentations produced by large-scale pretrained models developed for various types of tumors. We demonstrated the techniques in multiple BraTS 2025 segmentation challenge tasks, with the ranking metric improving by 14.9 % for the sub-Saharan Africa challenge and 0.9% for the adult glioma challenge. This approach promotes a shift in brain tumor segmentation research from increasingly complex model architectures to efficient, clinically aligned post-processing strategies that are precise, computationally fair, and sustainable.
>
---
#### [new 080] MoonSeg3R: Monocular Online Zero-Shot Segment Anything in 3D with Reconstructive Foundation Priors
- **分类: cs.CV**

- **简介: 该论文提出MoonSeg3R，解决在线零样本单目3D实例分割任务——即仅用单路RGB视频流实时生成3D实例分割，无需预训练类别或深度图。它利用CUT3R几何先验，设计查询提炼、时序记忆与状态令牌融合三大模块，在ScanNet200等数据集上首次实现单目在线3D分割，性能媲美RGB-D方法。**

- **链接: [https://arxiv.org/pdf/2512.15577v1](https://arxiv.org/pdf/2512.15577v1)**

> **作者:** Zhipeng Du; Duolikun Danier; Jan Eric Lenssen; Hakan Bilen
>
> **摘要:** In this paper, we focus on online zero-shot monocular 3D instance segmentation, a novel practical setting where existing approaches fail to perform because they rely on posed RGB-D sequences. To overcome this limitation, we leverage CUT3R, a recent Reconstructive Foundation Model (RFM), to provide reliable geometric priors from a single RGB stream. We propose MoonSeg3R, which introduces three key components: (1) a self-supervised query refinement module with spatial-semantic distillation that transforms segmentation masks from 2D visual foundation models (VFMs) into discriminative 3D queries; (2) a 3D query index memory that provides temporal consistency by retrieving contextual queries; and (3) a state-distribution token from CUT3R that acts as a mask identity descriptor to strengthen cross-frame fusion. Experiments on ScanNet200 and SceneNN show that MoonSeg3R is the first method to enable online monocular 3D segmentation and achieves performance competitive with state-of-the-art RGB-D-based systems. Code and models will be released.
>
---
#### [new 081] Asynchronous Event Stream Noise Filtering for High-frequency Structure Deformation Measurement
- **分类: cs.CV**

- **简介: 该论文属高频率结构变形测量任务，旨在解决传统高速相机受光照与成本限制难以实时监测大型结构高频变形的问题。提出基于事件相机与LED标记的异步事件流去噪方法，利用闪烁特性与时空相关性滤除噪声，分离运动与闪烁事件，实现单目事件相机下的高频平面变形测量。**

- **链接: [https://arxiv.org/pdf/2512.15055v1](https://arxiv.org/pdf/2512.15055v1)**

> **作者:** Yifei Bian; Banglei Guan; Zibin Liu; Ang Su; Shiyao Zhu; Yang Shang; Qifeng Yu
>
> **备注:** 13 pages, 12 figures
>
> **摘要:** Large-scale structures suffer high-frequency deformations due to complex loads. However, harsh lighting conditions and high equipment costs limit measurement methods based on traditional high-speed cameras. This paper proposes a method to measure high-frequency deformations by exploiting an event camera and LED markers. Firstly, observation noise is filtered based on the characteristics of the event stream generated by LED markers blinking and spatiotemporal correlation. Then, LED markers are extracted from the event stream after differentiating between motion-induced events and events from LED blinking, which enables the extraction of high-speed moving LED markers. Ultimately, high-frequency planar deformations are measured by a monocular event camera. Experimental results confirm the accuracy of our method in measuring high-frequency planar deformations.
>
---
#### [new 082] Qwen-Image-Layered: Towards Inherent Editability via Layer Decomposition
- **分类: cs.CV**

- **简介: 该论文提出Qwen-Image-Layered，属图像分层分解任务，旨在解决生成式图像编辑中因像素纠缠导致的不一致问题。通过RGBA-VAE、可变层数MMDiT架构和多阶段训练，将单图分解为语义解耦的RGBA图层，实现固有可编辑性，并构建PSD数据集支撑训练。**

- **链接: [https://arxiv.org/pdf/2512.15603v1](https://arxiv.org/pdf/2512.15603v1)**

> **作者:** Shengming Yin; Zekai Zhang; Zecheng Tang; Kaiyuan Gao; Xiao Xu; Kun Yan; Jiahao Li; Yilei Chen; Yuxiang Chen; Heung-Yeung Shum; Lionel M. Ni; Jingren Zhou; Junyang Lin; Chenfei Wu
>
> **备注:** 12 pages, 8 figures
>
> **摘要:** Recent visual generative models often struggle with consistency during image editing due to the entangled nature of raster images, where all visual content is fused into a single canvas. In contrast, professional design tools employ layered representations, allowing isolated edits while preserving consistency. Motivated by this, we propose \textbf{Qwen-Image-Layered}, an end-to-end diffusion model that decomposes a single RGB image into multiple semantically disentangled RGBA layers, enabling \textbf{inherent editability}, where each RGBA layer can be independently manipulated without affecting other content. To support variable-length decomposition, we introduce three key components: (1) an RGBA-VAE to unify the latent representations of RGB and RGBA images; (2) a VLD-MMDiT (Variable Layers Decomposition MMDiT) architecture capable of decomposing a variable number of image layers; and (3) a Multi-stage Training strategy to adapt a pretrained image generation model into a multilayer image decomposer. Furthermore, to address the scarcity of high-quality multilayer training images, we build a pipeline to extract and annotate multilayer images from Photoshop documents (PSD). Experiments demonstrate that our method significantly surpasses existing approaches in decomposition quality and establishes a new paradigm for consistent image editing. Our code and models are released on \href{https://github.com/QwenLM/Qwen-Image-Layered}{https://github.com/QwenLM/Qwen-Image-Layered}
>
---
#### [new 083] SocialNav-MoE: A Mixture-of-Experts Vision Language Model for Socially Compliant Navigation with Reinforcement Fine-Tuning
- **分类: cs.CV; cs.RO**

- **简介: 该论文面向机器人社会合规导航任务，解决现有方法重安全轻社交、大模型难实时部署的问题。提出轻量级MoE架构SocialNav-MoE，结合强化微调与语义相似性奖励（SSR），并系统评估小语言模型、路由策略及视觉编码器组合，在SNEI数据集上实现精度与效率的平衡。**

- **链接: [https://arxiv.org/pdf/2512.14757v1](https://arxiv.org/pdf/2512.14757v1)**

> **作者:** Tomohito Kawabata; Xinyu Zhang; Ling Xiao
>
> **摘要:** For robots navigating in human-populated environments, safety and social compliance are equally critical, yet prior work has mostly emphasized safety. Socially compliant navigation that accounts for human comfort, social norms, and contextual appropriateness remains underexplored. Vision language models (VLMs) show promise for this task; however, large-scale models incur substantial computational overhead, leading to higher inference latency and energy consumption, which makes them unsuitable for real-time deployment on resource-constrained robotic platforms. To address this issue, we investigate the effectiveness of small VLM and propose SocialNav-MoE, an efficient Mixture-of-Experts vision language model for socially compliant navigation with reinforcement fine-tuning (RFT). We further introduce a semantic similarity reward (SSR) to effectively leverage RFT for enhancing the decision-making capabilities. Additionally, we study the effectiveness of different small language model types (Phi, Qwen, and StableLM), routing strategies, and vision encoders (CLIP vs. SigLIP, frozen vs. fine-tuned). Experiments on the SNEI dataset demonstrate that SocialNav-MoE achieves an excellent balance between navigation accuracy and efficiency. The proposed SSR function is more effective than hard-level and character-level rewards. Source code will be released upon acceptance.
>
---
#### [new 084] Photorealistic Phantom Roads in Real Scenes: Disentangling 3D Hallucinations from Physical Geometry
- **分类: cs.CV; cs.RO**

- **简介: 该论文聚焦单目深度估计任务，解决深度模型因语义先验导致的“3D幻境”（在平面区域误估非平面结构）这一安全风险。工作包括：构建首个真实幻觉基准3D-Mirage；提出拉普拉斯评估指标（DCS/CCS）；设计参数高效方法“接地自蒸馏”强制幻觉区域平面性。**

- **链接: [https://arxiv.org/pdf/2512.15423v1](https://arxiv.org/pdf/2512.15423v1)**

> **作者:** Hoang Nguyen; Xiaohao Xu; Xiaonan Huang
>
> **摘要:** Monocular depth foundation models achieve remarkable generalization by learning large-scale semantic priors, but this creates a critical vulnerability: they hallucinate illusory 3D structures from geometrically planar but perceptually ambiguous inputs. We term this failure the 3D Mirage. This paper introduces the first end-to-end framework to probe, quantify, and tame this unquantified safety risk. To probe, we present 3D-Mirage, the first benchmark of real-world illusions (e.g., street art) with precise planar-region annotations and context-restricted crops. To quantify, we propose a Laplacian-based evaluation framework with two metrics: the Deviation Composite Score (DCS) for spurious non-planarity and the Confusion Composite Score (CCS) for contextual instability. To tame this failure, we introduce Grounded Self-Distillation, a parameter-efficient strategy that surgically enforces planarity on illusion ROIs while using a frozen teacher to preserve background knowledge, thus avoiding catastrophic forgetting. Our work provides the essential tools to diagnose and mitigate this phenomenon, urging a necessary shift in MDE evaluation from pixel-wise accuracy to structural and contextual robustness. Our code and benchmark will be publicly available to foster this exciting research direction.
>
---
#### [new 085] KD360-VoxelBEV: LiDAR and 360-degree Camera Cross Modality Knowledge Distillation for Bird's-Eye-View Segmentation
- **分类: cs.CV**

- **简介: 该论文面向BEV语义分割任务，解决单全景相机方案精度低的问题。提出KD360-VoxelBEV框架，利用LiDAR-相机融合教师模型蒸馏知识至轻量级纯相机学生模型，在Dur360BEV上提升IoU达8.5%，达31.2 FPS。**

- **链接: [https://arxiv.org/pdf/2512.15311v1](https://arxiv.org/pdf/2512.15311v1)**

> **作者:** Wenke E; Yixin Sun; Jiaxu Liu; Hubert P. H. Shum; Amir Atapour-Abarghouei; Toby P. Breckon
>
> **摘要:** We present the first cross-modality distillation framework specifically tailored for single-panoramic-camera Bird's-Eye-View (BEV) segmentation. Our approach leverages a novel LiDAR image representation fused from range, intensity and ambient channels, together with a voxel-aligned view transformer that preserves spatial fidelity while enabling efficient BEV processing. During training, a high-capacity LiDAR and camera fusion Teacher network extracts both rich spatial and semantic features for cross-modality knowledge distillation into a lightweight Student network that relies solely on a single 360-degree panoramic camera image. Extensive experiments on the Dur360BEV dataset demonstrate that our teacher model significantly outperforms existing camera-based BEV segmentation methods, achieving a 25.6\% IoU improvement. Meanwhile, the distilled Student network attains competitive performance with an 8.5\% IoU gain and state-of-the-art inference speed of 31.2 FPS. Moreover, evaluations on KITTI-360 (two fisheye cameras) confirm that our distillation framework generalises to diverse camera setups, underscoring its feasibility and robustness. This approach reduces sensor complexity and deployment costs while providing a practical solution for efficient, low-cost BEV segmentation in real-world autonomous driving.
>
---
#### [new 086] SLCFormer: Spectral-Local Context Transformer with Physics-Grounded Flare Synthesis for Nighttime Flare Removal
- **分类: cs.CV**

- **简介: 该论文属计算机视觉中的夜间镜头眩光去除任务，旨在解决非均匀散射眩光去除难的问题。提出SLCFormer模型：含频域建模的FFEM模块和空域方向增强的DESM模块，并设计ZernikeVAE物理驱动的眩光合成方法，提升真实场景泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.15221v1](https://arxiv.org/pdf/2512.15221v1)**

> **作者:** Xiyu Zhu; Wei Wang; Xin Yuan; Xiao Wang
>
> **摘要:** Lens flare is a common nighttime artifact caused by strong light sources scattering within camera lenses, leading to hazy streaks, halos, and glare that degrade visual quality. However, existing methods usually fail to effectively address nonuniform scattered flares, which severely reduces their applicability to complex real-world scenarios with diverse lighting conditions. To address this issue, we propose SLCFormer, a novel spectral-local context transformer framework for effective nighttime lens flare removal. SLCFormer integrates two key modules: the Frequency Fourier and Excitation Module (FFEM), which captures efficient global contextual representations in the frequency domain to model flare characteristics, and the Directionally-Enhanced Spatial Module (DESM) for local structural enhancement and directional features in the spatial domain for precise flare removal. Furthermore, we introduce a ZernikeVAE-based scatter flare generation pipeline to synthesize physically realistic scatter flares with spatially varying PSFs, bridging optical physics and data-driven training. Extensive experiments on the Flare7K++ dataset demonstrate that our method achieves state-of-the-art performance, outperforming existing approaches in both quantitative metrics and perceptual visual quality, and generalizing robustly to real nighttime scenes with complex flare artifacts.
>
---
#### [new 087] Hard Labels In! Rethinking the Role of Hard Labels in Mitigating Local Semantic Drift
- **分类: cs.CV**

- **简介: 该论文属知识蒸馏/数据集蒸馏任务，旨在解决软标签在少裁剪场景下引发的局部语义漂移问题。作者发现硬标签可作为语义锚点校准漂移，提出HALD新范式，融合软硬标签以提升泛化性，在ImageNet等任务上显著超越SOTA。**

- **链接: [https://arxiv.org/pdf/2512.15647v1](https://arxiv.org/pdf/2512.15647v1)**

> **作者:** Jiacheng Cui; Bingkui Tong; Xinyue Bi; Xiaohan Zhao; Jiacheng Liu; Zhiqiang shen
>
> **备注:** Code at: https://github.com/Jiacheng8/HALD
>
> **摘要:** Soft labels generated by teacher models have become a dominant paradigm for knowledge transfer and recent large-scale dataset distillation such as SRe2L, RDED, LPLD, offering richer supervision than conventional hard labels. However, we observe that when only a limited number of crops per image are used, soft labels are prone to local semantic drift: a crop may visually resemble another class, causing its soft embedding to deviate from the ground-truth semantics of the original image. This mismatch between local visual content and global semantic meaning introduces systematic errors and distribution misalignment between training and testing. In this work, we revisit the overlooked role of hard labels and show that, when appropriately integrated, they provide a powerful content-agnostic anchor to calibrate semantic drift. We theoretically characterize the emergence of drift under few soft-label supervision and demonstrate that hybridizing soft and hard labels restores alignment between visual content and semantic supervision. Building on this insight, we propose a new training paradigm, Hard Label for Alleviating Local Semantic Drift (HALD), which leverages hard labels as intermediate corrective signals while retaining the fine-grained advantages of soft labels. Extensive experiments on dataset distillation and large-scale conventional classification benchmarks validate our approach, showing consistent improvements in generalization. On ImageNet-1K, we achieve 42.7% with only 285M storage for soft labels, outperforming prior state-of-the-art LPLD by 9.0%. Our findings re-establish the importance of hard labels as a complementary tool, and call for a rethinking of their role in soft-label-dominated training.
>
---
#### [new 088] FlexAvatar: Learning Complete 3D Head Avatars with Partial Supervision
- **分类: cs.CV**

- **简介: 该论文属3D头像重建与动画任务，旨在解决单图输入下3D头像不完整、视角外推差的问题。提出FlexAvatar方法：基于Transformer的模型，引入可学习“bias sink”令牌统一融合单目与多视角数据，实现高质量、完整、可动画的3D头像生成，并支持身份插值与灵活拟合。**

- **链接: [https://arxiv.org/pdf/2512.15599v1](https://arxiv.org/pdf/2512.15599v1)**

> **作者:** Tobias Kirschstein; Simon Giebenhain; Matthias Nießner
>
> **备注:** Project website: https://tobias-kirschstein.github.io/flexavatar/ , Video: https://youtu.be/g8wxqYBlRGY
>
> **摘要:** We introduce FlexAvatar, a method for creating high-quality and complete 3D head avatars from a single image. A core challenge lies in the limited availability of multi-view data and the tendency of monocular training to yield incomplete 3D head reconstructions. We identify the root cause of this issue as the entanglement between driving signal and target viewpoint when learning from monocular videos. To address this, we propose a transformer-based 3D portrait animation model with learnable data source tokens, so-called bias sinks, which enables unified training across monocular and multi-view datasets. This design leverages the strengths of both data sources during inference: strong generalization from monocular data and full 3D completeness from multi-view supervision. Furthermore, our training procedure yields a smooth latent avatar space that facilitates identity interpolation and flexible fitting to an arbitrary number of input observations. In extensive evaluations on single-view, few-shot, and monocular avatar creation tasks, we verify the efficacy of FlexAvatar. Many existing methods struggle with view extrapolation while FlexAvatar generates complete 3D head avatars with realistic facial animations. Website: https://tobias-kirschstein.github.io/flexavatar/
>
---
#### [new 089] HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering
- **分类: cs.CV; eess.IV**

- **简介: 该论文提出HERBench——一个专为评估视频问答（VideoQA）中跨时间多证据整合能力而设计的基准。针对现有基准易被单线索或语言先验“作弊”的问题，它强制要求融合至少三个时序分离的视觉证据，并引入MRFS量化证据需求，揭示当前Video-LLMs在此任务上表现极差（31–42%），主因是证据检索与融合双重缺陷。**

- **链接: [https://arxiv.org/pdf/2512.14870v1](https://arxiv.org/pdf/2512.14870v1)**

> **作者:** Dan Ben-Ami; Gabriele Serussi; Kobi Cohen; Chaim Baskin
>
> **摘要:** Video Large Language Models (Video-LLMs) are rapidly improving, yet current Video Question Answering (VideoQA) benchmarks often allow questions to be answered from a single salient cue, under-testing reasoning that must aggregate multiple, temporally separated visual evidence. We present HERBench, a VideoQA benchmark purpose-built to assess multi-evidence integration across time. Each question requires aggregating at least three non-overlapping evidential cues across distinct video segments, so neither language priors nor a single snapshot can suffice. HERBench comprises 26K five-way multiple-choice questions organized into twelve compositional tasks that probe identity binding, cross-entity relations, temporal ordering, co-occurrence verification, and counting. To make evidential demand measurable, we introduce the Minimum Required Frame-Set (MRFS), the smallest number of frames a model must fuse to answer correctly, and show that HERBench imposes substantially higher demand than prior datasets (mean MRFS 5.5 vs. 2.6-4.2). Evaluating 13 state-of-the-art Video-LLMs on HERBench reveals pervasive failures: accuracies of 31-42% are only slightly above the 20% random-guess baseline. We disentangle this failure into two critical bottlenecks: (1) a retrieval deficit, where frame selectors overlook key evidence, and (2) a fusion deficit, where models fail to integrate information even when all necessary evidence is provided. By making cross-time evidence both unavoidable and quantifiable, HERBench establishes a principled target for advancing robust, compositional video understanding.
>
---
#### [new 090] Vision-based module for accurately reading linear scales in a laboratory
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉中的精确计量任务，旨在让机器人自主读取实验室线性刻度（如注射器、量筒）的数值。工作包括：图像定向校正、感兴趣区域裁剪、关键特征（刻度线、数字、液面指示器）提取与定位，最终实现高精度读数，并验证其与人工读数高度一致。**

- **链接: [https://arxiv.org/pdf/2512.15327v1](https://arxiv.org/pdf/2512.15327v1)**

> **作者:** Parvesh Saini; Soumyadipta Maiti; Beena Rai
>
> **备注:** 10 pages, 16 figures
>
> **摘要:** Capabilities and the number of vision-based models are increasing rapidly. And these vision models are now able to do more tasks like object detection, image classification, instance segmentation etc. with great accuracy. But models which can take accurate quantitative measurements form an image, as a human can do by just looking at it, are rare. For a robot to work with complete autonomy in a Laboratory environment, it needs to have some basic skills like navigation, handling objects, preparing samples etc. to match human-like capabilities in an unstructured environment. Another important capability is to read measurements from instruments and apparatus. Here, we tried to mimic a human inspired approach to read measurements from a linear scale. As a test case we have picked reading level from a syringe and a measuring cylinder. For a randomly oriented syringe we carry out transformations to correct the orientation. To make the system efficient and robust, the area of interest is reduced to just the linear scale containing part of the image. After that, a series of features were extracted like the major makers, the corresponding digits, and the level indicator location, from which the final reading was calculated. Readings obtained using this system were also compared against human read values of the same instances and an accurate correspondence was observed.
>
---
#### [new 091] SoFlow: Solution Flow Models for One-Step Generative Modeling
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于生成式建模任务，旨在解决扩散与Flow Matching模型多步采样导致的效率低问题。作者提出SoFlow框架，实现端到端单步生成；设计Flow Matching损失（支持CFG）和无需JVP的解一致性损失；在ImageNet上用DiT架构验证，FID优于MeanFlow。**

- **链接: [https://arxiv.org/pdf/2512.15657v1](https://arxiv.org/pdf/2512.15657v1)**

> **作者:** Tianze Luo; Haotian Yuan; Zhuang Liu
>
> **备注:** Our code is available at https://github.com/zlab-princeton/SoFlow
>
> **摘要:** The multi-step denoising process in diffusion and Flow Matching models causes major efficiency issues, which motivates research on few-step generation. We present Solution Flow Models (SoFlow), a framework for one-step generation from scratch. By analyzing the relationship between the velocity function and the solution function of the velocity ordinary differential equation (ODE), we propose a Flow Matching loss and a solution consistency loss to train our models. The Flow Matching loss allows our models to provide estimated velocity fields for Classifier-Free Guidance (CFG) during training, which improves generation performance. Notably, our consistency loss does not require the calculation of the Jacobian-vector product (JVP), a common requirement in recent works that is not well-optimized in deep learning frameworks like PyTorch. Experimental results indicate that, when trained from scratch using the same Diffusion Transformer (DiT) architecture and an equal number of training epochs, our models achieve better FID-50K scores than MeanFlow models on the ImageNet 256x256 dataset.
>
---
#### [new 092] EPSM: A Novel Metric to Evaluate the Safety of Environmental Perception in Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出EPSM——一种面向自动驾驶环境感知安全性的新型评估指标。针对传统精度类指标忽视安全风险的问题，它联合建模目标与车道检测任务，设计轻量级对象安全度量和考虑任务关联的车道安全度量，生成统一可解释的安全评分，并在DeepAccident数据集上验证其有效性。**

- **链接: [https://arxiv.org/pdf/2512.15195v1](https://arxiv.org/pdf/2512.15195v1)**

> **作者:** Jörg Gamerdinger; Sven Teufel; Stephan Amann; Lukas Marc Listl; Oliver Bringmann
>
> **备注:** Submitted at IEEE IV 2026
>
> **摘要:** Extensive evaluation of perception systems is crucial for ensuring the safety of intelligent vehicles in complex driving scenarios. Conventional performance metrics such as precision, recall and the F1-score assess the overall detection accuracy, but they do not consider the safety-relevant aspects of perception. Consequently, perception systems that achieve high scores in these metrics may still cause misdetections that could lead to severe accidents. Therefore, it is important to evaluate not only the overall performance of perception systems, but also their safety. We therefore introduce a novel safety metric for jointly evaluating the most critical perception tasks, object and lane detection. Our proposed framework integrates a new, lightweight object safety metric that quantifies the potential risk associated with object detection errors, as well as an lane safety metric including the interdependence between both tasks that can occur in safety evaluation. The resulting combined safety score provides a unified, interpretable measure of perception safety performance. Using the DeepAccident dataset, we demonstrate that our approach identifies safety critical perception errors that conventional performance metrics fail to capture. Our findings emphasize the importance of safety-centric evaluation methods for perception systems in autonomous driving.
>
---
#### [new 093] MiVLA: Towards Generalizable Vision-Language-Action Model with Human-Robot Mutual Imitation Pre-training
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MiVLA模型，属视觉-语言-动作（VLA）任务，旨在解决现有VLAs因视角、外观和形态差异导致的跨人机泛化能力弱问题。通过人类与机器人双向行为模仿预训练，利用手/臂运动学对齐，融合真实人类视频与仿真机器人数据，显著提升跨平台泛化性能。**

- **链接: [https://arxiv.org/pdf/2512.15411v1](https://arxiv.org/pdf/2512.15411v1)**

> **作者:** Zhenhan Yin; Xuanhan Wang; Jiahao Jiang; Kaiyuan Deng; Pengqi Chen; Shuangle Li; Chong Liu; Xing Xu; ingkuan Song; Lianli Gao; Heng Tao Shen
>
> **摘要:** While leveraging abundant human videos and simulated robot data poses a scalable solution to the scarcity of real-world robot data, the generalization capability of existing vision-language-action models (VLAs) remains limited by mismatches in camera views, visual appearance, and embodiment morphologies. To overcome this limitation, we propose MiVLA, a generalizable VLA empowered by human-robot mutual imitation pre-training, which leverages inherent behavioral similarity between human hands and robotic arms to build a foundation of strong behavioral priors for both human actions and robotic control. Specifically, our method utilizes kinematic rules with left/right hand coordinate systems for bidirectional alignment between human and robot action spaces. Given human or simulated robot demonstrations, MiVLA is trained to forecast behavior trajectories for one embodiment, and imitate behaviors for another one unseen in the demonstration. Based on this mutual imitation, it integrates the behavioral fidelity of real-world human data with the manipulative diversity of simulated robot data into a unified model, thereby enhancing the generalization capability for downstream tasks. Extensive experiments conducted on both simulation and real-world platforms with three robots (ARX, PiPer and LocoMan), demonstrate that MiVLA achieves strong improved generalization capability, outperforming state-of-the-art VLAs (e.g., $\boldsymbolπ_{0}$, $\boldsymbolπ_{0.5}$ and H-RDT) by 25% in simulation, and 14% in real-world robot control tasks.
>
---
#### [new 094] Image Complexity-Aware Adaptive Retrieval for Efficient Vision-Language Models
- **分类: cs.IR; cs.AI; cs.CV; cs.LG; cs.MM**

- **简介: 该论文属视觉-语言模型推理优化任务，旨在解决图像统一计算导致的效率低下问题。提出ICAR方法：用ConvNeXt-IC评估图像复杂度，自适应调整ViT计算深度；通过双路径训练保证不同深度图像嵌入与文本嵌入语义兼容，实现免重排序的高效跨模态匹配。**

- **链接: [https://arxiv.org/pdf/2512.15372v1](https://arxiv.org/pdf/2512.15372v1)**

> **作者:** Mikel Williams-Lekuona; Georgina Cosma
>
> **备注:** Accepted paper for ECIR 2026
>
> **摘要:** Vision transformers in vision-language models apply uniform computational effort across all images, expending 175.33 GFLOPs (ViT-L/14) whether analysing a straightforward product photograph or a complex street scene. We propose ICAR (Image Complexity-Aware Retrieval), which enables vision transformers to use less compute for simple images whilst processing complex images through their full network depth. The key challenge is maintaining cross-modal alignment: embeddings from different processing depths must remain compatible for text matching. ICAR solves this through dual-path training that produces compatible embeddings from both reduced-compute and full-compute processing. This maintains compatibility between image representations and text embeddings in the same semantic space, whether an image exits early or processes fully. Unlike existing two-stage approaches that require expensive reranking, ICAR enables direct image-text matching without additional overhead. To determine how much compute to use, we develop ConvNeXt-IC, which treats image complexity assessment as a classification task. By applying modern classifier backbones rather than specialised architectures, ConvNeXt-IC achieves state-of-the-art performance with 0.959 correlation with human judgement (Pearson) and 4.4x speedup. Evaluated on standard benchmarks augmented with real-world web data, ICAR achieves 20% practical speedup while maintaining category-level performance and 95% of instance-level performance, enabling sustainable scaling of vision-language systems.
>
---
#### [new 095] BEV-Patch-PF: Particle Filtering with BEV-Aerial Feature Matching for Off-Road Geo-Localization
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出BEV-Patch-PF，一种无GPS的越野场景顺序地理定位方法。针对密集树冠和阴影下定位失效问题，融合车载BEV特征图与航拍特征图，通过粒子滤波匹配实现鲁棒实时定位，在真实数据集上显著降低轨迹误差。**

- **链接: [https://arxiv.org/pdf/2512.15111v1](https://arxiv.org/pdf/2512.15111v1)**

> **作者:** Dongmyeong Lee; Jesse Quattrociocchi; Christian Ellis; Rwik Rana; Amanda Adkins; Adam Uccello; Garrett Warnell; Joydeep Biswas
>
> **摘要:** We propose BEV-Patch-PF, a GPS-free sequential geo-localization system that integrates a particle filter with learned bird's-eye-view (BEV) and aerial feature maps. From onboard RGB and depth images, we construct a BEV feature map. For each 3-DoF particle pose hypothesis, we crop the corresponding patch from an aerial feature map computed from a local aerial image queried around the approximate location. BEV-Patch-PF computes a per-particle log-likelihood by matching the BEV feature to the aerial patch feature. On two real-world off-road datasets, our method achieves 7.5x lower absolute trajectory error (ATE) on seen routes and 7.0x lower ATE on unseen routes than a retrieval-based baseline, while maintaining accuracy under dense canopy and shadow. The system runs in real time at 10 Hz on an NVIDIA Tesla T4, enabling practical robot deployment.
>
---
#### [new 096] Meta-learners for few-shot weakly-supervised optic disc and cup segmentation on fundus images
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文面向眼底图像中视杯/视盘分割任务，解决标注数据稀缺下的弱监督分割难题。提出Omni元训练、高效架构及稀疏标注生成技术，构建轻量级元学习器EO-ProtoSeg，在仅需1张稀疏标注图时即达SOTA性能。**

- **链接: [https://arxiv.org/pdf/2512.15061v1](https://arxiv.org/pdf/2512.15061v1)**

> **作者:** Pandega Abyan Zumarsyah; Igi Ardiyanto; Hanung Adi Nugroho
>
> **备注:** Submitted to Computers in Biology and Medicine
>
> **摘要:** This study develops meta-learners for few-shot weakly-supervised segmentation (FWS) to address the challenge of optic disc (OD) and optic cup (OC) segmentation for glaucoma diagnosis with limited labeled fundus images. We significantly improve existing meta-learners by introducing Omni meta-training which balances data usage and diversifies the number of shots. We also develop their efficient versions that reduce computational costs. In addition, we develop sparsification techniques that generate more customizable and representative scribbles and other sparse labels. After evaluating multiple datasets, we find that Omni and efficient versions outperform the original versions, with the best meta-learner being Efficient Omni ProtoSeg (EO-ProtoSeg). It achieves intersection over union (IoU) scores of 88.15% for OD and 71.17% for OC on the REFUGE dataset using just one sparsely labeled image, outperforming few-shot and semi-supervised methods which require more labeled images. Its best performance reaches 86.80% for OD and 71.78%for OC on DRISHTIGS, 88.21% for OD and 73.70% for OC on REFUGE, 80.39% for OD and 52.65% for OC on REFUGE. EO-ProtoSeg is comparable to unsupervised domain adaptation methods yet much lighter with less than two million parameters and does not require any retraining.
>
---
#### [new 097] PyFi: Toward Pyramid-like Financial Image Understanding for VLMs via Adversarial Agents
- **分类: q-fin.CP; cs.AI; cs.CV**

- **简介: 论文提出PyFi框架，解决VLM在金融图像理解中缺乏渐进式推理能力的问题。构建无标注的600K金字塔式金融QA数据集（PyFi-600K），通过多智能体对抗机制（PyFi-adv）自动生成分层问题链，并微调Qwen2.5-VL模型，显著提升复杂金融图像问答准确率。**

- **链接: [https://arxiv.org/pdf/2512.14735v1](https://arxiv.org/pdf/2512.14735v1)**

> **作者:** Yuqun Zhang; Yuxuan Zhao; Sijia Chen
>
> **摘要:** This paper proposes PyFi, a novel framework for pyramid-like financial image understanding that enables vision language models (VLMs) to reason through question chains in a progressive, simple-to-complex manner. At the core of PyFi is PyFi-600K, a dataset comprising 600K financial question-answer pairs organized into a reasoning pyramid: questions at the base require only basic perception, while those toward the apex demand increasing levels of capability in financial visual understanding and expertise. This data is scalable because it is synthesized without human annotations, using PyFi-adv, a multi-agent adversarial mechanism under the Monte Carlo Tree Search (MCTS) paradigm, in which, for each image, a challenger agent competes with a solver agent by generating question chains that progressively probe deeper capability levels in financial visual reasoning. Leveraging this dataset, we present fine-grained, hierarchical, and comprehensive evaluations of advanced VLMs in the financial domain. Moreover, fine-tuning Qwen2.5-VL-3B and Qwen2.5-VL-7B on the pyramid-structured question chains enables these models to answer complex financial questions by decomposing them into sub-questions with gradually increasing reasoning demands, yielding average accuracy improvements of 19.52% and 8.06%, respectively, on the dataset. All resources of code, dataset and models are available at: https://github.com/AgenticFinLab/PyFi .
>
---
#### [new 098] HERO: Hierarchical Traversable 3D Scene Graphs for Embodied Navigation Among Movable Obstacles
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属具身导航任务，解决静态场景图无法处理可移动障碍物导致的可达性差问题。提出 HERO 框架，构建分层可通行3D场景图，将可操作障碍物建模为可穿越路径，提升导航效率与可达性。**

- **链接: [https://arxiv.org/pdf/2512.15047v1](https://arxiv.org/pdf/2512.15047v1)**

> **作者:** Yunheng Wang; Yixiao Feng; Yuetong Fang; Shuning Zhang; Tan Jing; Jian Li; Xiangrui Jiang; Renjing Xu
>
> **摘要:** 3D Scene Graphs (3DSGs) constitute a powerful representation of the physical world, distinguished by their abilities to explicitly model the complex spatial, semantic, and functional relationships between entities, rendering a foundational understanding that enables agents to interact intelligently with their environment and execute versatile behaviors. Embodied navigation, as a crucial component of such capabilities, leverages the compact and expressive nature of 3DSGs to enable long-horizon reasoning and planning in complex, large-scale environments. However, prior works rely on a static-world assumption, defining traversable space solely based on static spatial layouts and thereby treating interactable obstacles as non-traversable. This fundamental limitation severely undermines their effectiveness in real-world scenarios, leading to limited reachability, low efficiency, and inferior extensibility. To address these issues, we propose HERO, a novel framework for constructing Hierarchical Traversable 3DSGs, that redefines traversability by modeling operable obstacles as pathways, capturing their physical interactivity, functional semantics, and the scene's relational hierarchy. The results show that, relative to its baseline, HERO reduces PL by 35.1% in partially obstructed environments and increases SR by 79.4% in fully obstructed ones, demonstrating substantially higher efficiency and reachability.
>
---
#### [new 099] Generative Preprocessing for Image Compression with Pre-trained Diffusion Models
- **分类: eess.IV; cs.CV; cs.MM**

- **简介: 该论文属图像压缩预处理任务，旨在解决传统R-D优化方法忽视感知质量的问题。首次将预训练扩散模型用于R-P优化：先蒸馏Stable Diffusion为单步模型，再轻量微调其注意力模块，结合R-P损失与可微编解码器，提升纹理与主观质量，兼容标准编码器。**

- **链接: [https://arxiv.org/pdf/2512.15270v1](https://arxiv.org/pdf/2512.15270v1)**

> **作者:** Mengxi Guo; Shijie Zhao; Junlin Li; Li Zhang
>
> **备注:** Accepted as a PAPER and for publication in the DCC 2026 proceedings
>
> **摘要:** Preprocessing is a well-established technique for optimizing compression, yet existing methods are predominantly Rate-Distortion (R-D) optimized and constrained by pixel-level fidelity. This work pioneers a shift towards Rate-Perception (R-P) optimization by, for the first time, adapting a large-scale pre-trained diffusion model for compression preprocessing. We propose a two-stage framework: first, we distill the multi-step Stable Diffusion 2.1 into a compact, one-step image-to-image model using Consistent Score Identity Distillation (CiD). Second, we perform a parameter-efficient fine-tuning of the distilled model's attention modules, guided by a Rate-Perception loss and a differentiable codec surrogate. Our method seamlessly integrates with standard codecs without any modification and leverages the model's powerful generative priors to enhance texture and mitigate artifacts. Experiments show substantial R-P gains, achieving up to a 30.13% BD-rate reduction in DISTS on the Kodak dataset and delivering superior subjective visual quality.
>
---
#### [new 100] mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出mimic-video模型，属机器人控制任务，旨在解决VLAs缺乏物理因果理解、依赖大量专家数据的问题。它用预训练视频模型联合建模语义与动态，并设计流匹配动作解码器作为逆动力学模型，提升样本效率与收敛速度。**

- **链接: [https://arxiv.org/pdf/2512.15692v1](https://arxiv.org/pdf/2512.15692v1)**

> **作者:** Jonas Pai; Liam Achenbach; Victoriano Montesinos; Benedek Forrai; Oier Mees; Elvis Nava
>
> **摘要:** Prevailing Vision-Language-Action Models (VLAs) for robotic manipulation are built upon vision-language backbones pretrained on large-scale, but disconnected static web data. As a result, despite improved semantic generalization, the policy must implicitly infer complex physical dynamics and temporal dependencies solely from robot trajectories. This reliance creates an unsustainable data burden, necessitating continuous, large-scale expert data collection to compensate for the lack of innate physical understanding. We contend that while vision-language pretraining effectively captures semantic priors, it remains blind to physical causality. A more effective paradigm leverages video to jointly capture semantics and visual dynamics during pretraining, thereby isolating the remaining task of low-level control. To this end, we introduce \model, a novel Video-Action Model (VAM) that pairs a pretrained Internet-scale video model with a flow matching-based action decoder conditioned on its latent representations. The decoder serves as an Inverse Dynamics Model (IDM), generating low-level robot actions from the latent representation of video-space action plans. Our extensive evaluation shows that our approach achieves state-of-the-art performance on simulated and real-world robotic manipulation tasks, improving sample efficiency by 10x and convergence speed by 2x compared to traditional VLA architectures.
>
---
#### [new 101] LLM as a Neural Architect: Controlled Generation of Image Captioning Models Under Strict API Contracts
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出NN-Caption，用LLM指导神经架构搜索（NAS），在严格API约束下自动生成可运行的图像描述模型。它组合CNN编码器与序列解码器，生成、训练并评估数十种新模型，提升AutoML与可复现基准研究。**

- **链接: [https://arxiv.org/pdf/2512.14706v1](https://arxiv.org/pdf/2512.14706v1)**

> **作者:** Krunal Jesani; Dmitry Ignatov; Radu Timofte
>
> **摘要:** Neural architecture search (NAS) traditionally requires significant human expertise or automated trial-and-error to design deep learning models. We present NN-Caption, an LLM-guided neural architecture search pipeline that generates runnable image-captioning models by composing CNN encoders from LEMUR's classification backbones with sequence decoders (LSTM/GRU/Transformer) under a strict Net API. Using DeepSeek-R1-0528-Qwen3-8B as the primary generator, we present the prompt template and examples of generated architectures. We evaluate on MS COCO with BLEU-4. The LLM generated dozens of captioning models, with over half successfully trained and producing meaningful captions. We analyse the outcomes of using different numbers of input model snippets (5 vs. 10) in the prompt, finding a slight drop in success rate when providing more candidate components. We also report training dynamics (caption accuracy vs. epochs) and the highest BLEU-4 attained. Our results highlight the promise of LLM-guided NAS: the LLM not only proposes architectures but also suggests hyperparameters and training practices. We identify the challenges encountered (e.g., code hallucinations or API compliance issues) and detail how prompt rules and iterative code fixes addressed them. This work presents a pipeline that integrates prompt-based code generation with automatic evaluation, and adds dozens of novel captioning models to the open LEMUR dataset to facilitate reproducible benchmarking and downstream AutoML research.
>
---
#### [new 102] Magnification-Aware Distillation (MAD): A Self-Supervised Framework for Unified Representation Learning in Gigapixel Whole-Slide Images
- **分类: eess.IV; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属医学图像自监督学习任务，旨在解决WSI多尺度表征不一致问题。提出Magnification-Aware Distillation（MAD）框架，通过跨尺度知识蒸馏，使模型学习分辨率不变的统一嵌入表示，无需标注即可实现跨倍率（如10x→40x）鲁棒分类与分割。**

- **链接: [https://arxiv.org/pdf/2512.14796v1](https://arxiv.org/pdf/2512.14796v1)**

> **作者:** Mahmut S. Gokmen; Mitchell A. Klusty; Peter T. Nelson; Allison M. Neltner; Sen-Ching Samson Cheung; Thomas M. Pearce; David A Gutman; Brittany N. Dugger; Devavrat S. Bisht; Margaret E. Flanagan; V. K. Cody Bumgardner
>
> **备注:** 10 pages, 4 figures, 5 tables, submitted to AMIA 2026 Informatics Summit
>
> **摘要:** Whole-slide images (WSIs) contain tissue information distributed across multiple magnification levels, yet most self-supervised methods treat these scales as independent views. This separation prevents models from learning representations that remain stable when resolution changes, a key requirement for practical neuropathology workflows. This study introduces Magnification-Aware Distillation (MAD), a self-supervised strategy that links low-magnification context with spatially aligned high-magnification detail, enabling the model to learn how coarse tissue structure relates to fine cellular patterns. The resulting foundation model, MAD-NP, is trained entirely through this cross-scale correspondence without annotations. A linear classifier trained only on 10x embeddings maintains 96.7% of its performance when applied to unseen 40x tiles, demonstrating strong resolution-invariant representation learning. Segmentation outputs remain consistent across magnifications, preserving anatomical boundaries and minimizing noise. These results highlight the feasibility of scalable, magnification-robust WSI analysis using a unified embedding space
>
---
#### [new 103] Artificial Intelligence for the Assessment of Peritoneal Carcinosis during Diagnostic Laparoscopy for Advanced Ovarian Cancer
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文属医学AI任务，旨在解决腹腔镜下腹膜癌病（PC）评估主观性强、可重复性差的问题。研究基于诊断性腹腔镜视频，构建深度学习模型，自动识别关键帧、分割解剖结构与PC，并预测Fagotti评分及手术指征，提升卵巢癌术前评估标准化与可靠性。**

- **链接: [https://arxiv.org/pdf/2512.14797v1](https://arxiv.org/pdf/2512.14797v1)**

> **作者:** Riccardo Oliva; Farahdiba Zarin; Alice Zampolini Faustini; Armine Vardazaryan; Andrea Rosati; Vinkle Srivastav; Nunzia Del Villano; Jacques Marescaux; Giovanni Scambia; Pietro Mascagni; Nicolas Padoy; Anna Fagotti
>
> **摘要:** Advanced Ovarian Cancer (AOC) is often diagnosed at an advanced stage with peritoneal carcinosis (PC). Fagotti score (FS) assessment at diagnostic laparoscopy (DL) guides treatment planning by estimating surgical resectability, but its subjective and operator-dependent nature limits reproducibility and widespread use. Videos of patients undergoing DL with concomitant FS assessments at a referral center were retrospectively collected and divided into a development dataset, for data annotation, AI training and evaluation, and an independent test dataset, for internal validation. In the development dataset, FS-relevant frames were manually annotated for anatomical structures and PC. Deep learning models were trained to automatically identify FS-relevant frames, segment structures and PC, and predict video-level FS and indication to surgery (ItS). AI performance was evaluated using Dice score for segmentation, F1-scores for anatomical stations (AS) and ItS prediction, and root mean square error (RMSE) for final FS estimation. In the development dataset, the segmentation model trained on 7,311 frames, achieved Dice scores of 70$\pm$3% for anatomical structures and 56$\pm$3% for PC. Video-level AS classification achieved F1-scores of 74$\pm$3% and 73$\pm$4%, FS prediction showed normalized RMSE values of 1.39$\pm$0.18 and 1.15$\pm$0.08, and ItS reached F1-scores of 80$\pm$8% and 80$\pm$2% in the development (n=101) and independent test datasets (n=50), respectively. This is the first AI model to predict the feasibility of cytoreductive surgery providing automated FS estimation from DL videos. Its reproducible and reliable performance across datasets suggests that AI can support surgeons through standardized intraoperative tumor burden assessment and clinical decision-making in AOC.
>
---
#### [new 104] Evaluating Large Language Models on Multimodal Chemistry Olympiad Exams
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态科学推理任务，旨在评估大语言模型在化学奥赛题上的图文联合推理能力。作者构建USNCO试题基准，系统评测40个MLLMs，发现其模态融合能力薄弱，甚至图文共存时性能下降；验证思维链提示可提升准确率与视觉定位能力，并提出改进方向。**

- **链接: [https://arxiv.org/pdf/2512.14989v1](https://arxiv.org/pdf/2512.14989v1)**

> **作者:** Yiming Cui; Xin Yao; Yuxuan Qin; Xin Li; Shijin Wang; Guoping Hu
>
> **备注:** Published at Communications Chemistry
>
> **摘要:** Multimodal scientific reasoning remains a significant challenge for large language models (LLMs), particularly in chemistry, where problem-solving relies on symbolic diagrams, molecular structures, and structured visual data. Here, we systematically evaluate 40 proprietary and open-source multimodal LLMs, including GPT-5, o3, Gemini-2.5-Pro, and Qwen2.5-VL, on a curated benchmark of Olympiad-style chemistry questions drawn from over two decades of U.S. National Chemistry Olympiad (USNCO) exams. These questions require integrated visual and textual reasoning across diverse modalities. We find that many models struggle with modality fusion, where in some cases, removing the image even improves accuracy, indicating misalignment in vision-language integration. Chain-of-Thought prompting consistently enhances both accuracy and visual grounding, as demonstrated through ablation studies and occlusion-based interpretability. Our results reveal critical limitations in the scientific reasoning abilities of current MLLMs, providing actionable strategies for developing more robust and interpretable multimodal systems in chemistry. This work provides a timely benchmark for measuring progress in domain-specific multimodal AI and underscores the need for further advances at the intersection of artificial intelligence and scientific reasoning.
>
---
#### [new 105] A Preprocessing Framework for Video Machine Vision under Compression
- **分类: cs.MM; cs.CV**

- **简介: 该论文属视频压缩与机器视觉交叉任务，旨在解决标准视频编码（面向人眼）导致机器视觉性能下降的问题。提出一种神经预处理框架，含可微虚拟编解码器辅助训练，兼容真实标准编解码器，显著提升率-精度性能。**

- **链接: [https://arxiv.org/pdf/2512.15331v1](https://arxiv.org/pdf/2512.15331v1)**

> **作者:** Fei Zhao; Mengxi Guo; Shijie Zhao; Junlin Li; Li Zhang; Xiaodong Xie
>
> **备注:** Accepted as a POSTER and for publication in the DCC 2024 proceedings
>
> **摘要:** There has been a growing trend in compressing and transmitting videos from terminals for machine vision tasks. Nevertheless, most video coding optimization method focus on minimizing distortion according to human perceptual metrics, overlooking the heightened demands posed by machine vision systems. In this paper, we propose a video preprocessing framework tailored for machine vision tasks to address this challenge. The proposed method incorporates a neural preprocessor which retaining crucial information for subsequent tasks, resulting in the boosting of rate-accuracy performance. We further introduce a differentiable virtual codec to provide constraints on rate and distortion during the training stage. We directly apply widely used standard codecs for testing. Therefore, our solution can be easily applied to real-world scenarios. We conducted extensive experiments evaluating our compression method on two typical downstream tasks with various backbone networks. The experimental results indicate that our approach can save over 15% of bitrate compared to using only the standard codec anchor version.
>
---
#### [new 106] SepsisSuite: Beyond Risk Stratification -- A Comparative Analysis of Deep Fusion vs. Expert Stacking for Prescriptive Sepsis AI
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.CY**

- **简介: 该论文聚焦脓毒症的预测与抗生素处方决策任务，旨在解决多模态数据融合效果差、模型过拟合及临床实用性不足问题。提出SepsisLateFusion（上下文感知专家堆叠）和Quad-Modal Ensemble等新架构，在MIMIC-IV上实现SOTA预测性能（0.915 AUC）与处方支持，并开源部署框架SepsisSuite。**

- **链接: [https://arxiv.org/pdf/2512.14712v1](https://arxiv.org/pdf/2512.14712v1)**

> **作者:** Ryan Cartularo
>
> **备注:** 7 Pages, 4 Tables, 9 Figures
>
> **摘要:** Sepsis accounts for nearly 20% of global ICU admissions, yet conventional prediction models often fail to effectively integrate heterogeneous data streams, remaining either siloed by modality or reliant on brittle early fusion. In this work, we present a rigorous architectural comparison between End-to-End Deep Fusion and Context-Aware Stacking for sepsis tasks. We initially hypothesized that a novel Quad-Modal Hierarchical Gated Attention Network -- termed SepsisFusionFormer -- would resolve complex cross-modal interactions between vitals, text, and imaging. However, experiments on MIMIC-IV revealed that SepsisFusionFormer suffered from "attention starvation" in the small antibiotic cohort ($N \approx 2,100$), resulting in overfitting (AUC 0.66). This counterintuitive result informed the design of SepsisLateFusion, a "leaner" Context-Aware Mixture-of-Experts (MoE) architecture. By treating modalities as orthogonal experts -- the "Historian" (Static), the "Monitor" (Temporal), and the "Reader" (NLP) -- and dynamically gating them via a CatBoost meta-learner, we achieved State-of-the-Art (SOTA) performance: 0.915 AUC for prediction 4 hours prior to clinical onset. By calibrating the decision threshold for clinical safety, we reduced missed cases by 48% relative to the default operating point, thus opening a true preventative window for timely intervention over reactive alerts. Furthermore, for the novel prescriptive task of multi-class antibiotic selection, we demonstrate that a Quad-Modal Ensemble achieved the highest performance (0.72 AUC). These models are integrated into SepsisSuite, a deployment-ready Python framework for clinical decision support. SepsisSuite is available for free at: https://github.com/RyanCartularo/SepsisSuite-Info
>
---
#### [new 107] INFORM-CT: INtegrating LLMs and VLMs FOR Incidental Findings Management in Abdominal CT
- **分类: cs.LG; cs.AI; cs.CV; eess.IV**

- **简介: 该论文提出INFORM-CT框架，解决腹部CT中偶然发现（incidental findings）的自动检测、分类与报告问题。它融合LLM（作规划器生成Python脚本）与VLM/分割模型（作执行器完成视觉分析），实现端到端自动化管理，提升精度与效率。**

- **链接: [https://arxiv.org/pdf/2512.14732v1](https://arxiv.org/pdf/2512.14732v1)**

> **作者:** Idan Tankel; Nir Mazor; Rafi Brada; Christina LeBedis; Guy ben-Yosef
>
> **摘要:** Incidental findings in CT scans, though often benign, can have significant clinical implications and should be reported following established guidelines. Traditional manual inspection by radiologists is time-consuming and variable. This paper proposes a novel framework that leverages large language models (LLMs) and foundational vision-language models (VLMs) in a plan-and-execute agentic approach to improve the efficiency and precision of incidental findings detection, classification, and reporting for abdominal CT scans. Given medical guidelines for abdominal organs, the process of managing incidental findings is automated through a planner-executor framework. The planner, based on LLM, generates Python scripts using predefined base functions, while the executor runs these scripts to perform the necessary checks and detections, via VLMs, segmentation models, and image processing subroutines. We demonstrate the effectiveness of our approach through experiments on a CT abdominal benchmark for three organs, in a fully automatic end-to-end manner. Our results show that the proposed framework outperforms existing pure VLM-based approaches in terms of accuracy and efficiency.
>
---
#### [new 108] Task Matrices: Linear Maps for Cross-Model Finetuning Transfer
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文提出“任务矩阵”，即从预训练到微调模型的线性映射，旨在验证跨模型微调中存在通用线性编码。它在视觉与语言模型、10个数据集上验证：仅添加任务矩阵即可逼近全量微调性能，优于线性探针，支持高效可迁移的适配。**

- **链接: [https://arxiv.org/pdf/2512.14880v1](https://arxiv.org/pdf/2512.14880v1)**

> **作者:** Darrin O' Brien; Dhikshith Gajulapalli; Eric Xia
>
> **备注:** NeurIPS Unireps 2025
>
> **摘要:** Results in interpretability suggest that large vision and language models learn implicit linear encodings when models are biased by in-context prompting. However, the existence of similar linear representations in more general adaptation regimes has not yet been demonstrated. In this work, we develop the concept of a task matrix, a linear transformation from a base to finetuned embedding state. We demonstrate that for vision and text models and ten different datasets, a base model augmented with a task matrix achieves results surpassing linear probes, sometimes approaching finetuned levels. Our results validate the existence of cross-layer linear encodings between pretrained and finetuned architectures. Moreover, we show that a data-based approximation for such encodings is both efficient and generalizable to multiple domains. We make our implementation publicly available.
>
---
#### [new 109] A Gaussian Parameterization for Direct Atomic Structure Identification in Electron Tomography
- **分类: eess.IV; cs.CV**

- **简介: 该论文属原子电子断层成像（AET）任务，旨在直接识别3D原子结构。针对传统方法需先重建体素再后处理的缺陷，提出高斯参数化方法，将原子建模为可学习位置与属性的高斯分布，引入物理先验以提升抗噪鲁棒性，并在仿真与实测数据上验证有效性。**

- **链接: [https://arxiv.org/pdf/2512.15034v1](https://arxiv.org/pdf/2512.15034v1)**

> **作者:** Nalini M. Singh; Tiffany Chien; Arthur R. C. McCray; Colin Ophus; Laura Waller
>
> **备注:** Published in ICCP 2025. 14 pages, 10 figures. Keywords: Atomic electron tomography, Gaussian splatting
>
> **摘要:** Atomic electron tomography (AET) enables the determination of 3D atomic structures by acquiring a sequence of 2D tomographic projection measurements of a particle and then computationally solving for its underlying 3D representation. Classical tomography algorithms solve for an intermediate volumetric representation that is post-processed into the atomic structure of interest. In this paper, we reformulate the tomographic inverse problem to solve directly for the locations and properties of individual atoms. We parameterize an atomic structure as a collection of Gaussians, whose positions and properties are learnable. This representation imparts a strong physical prior on the learned structure, which we show yields improved robustness to real-world imaging artifacts. Simulated experiments and a proof-of-concept result on experimentally-acquired data confirm our method's potential for practical applications in materials characterization and analysis with Transmission Electron Microscopy (TEM). Our code is available at https://github.com/nalinimsingh/gaussian-atoms.
>
---
## 更新

#### [replaced 001] Binarization-Aware Adjuster: A Theoretical Framework for Bridging Continuous Optimization and Discrete Inference with Application to Edge Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2506.12460v2](https://arxiv.org/pdf/2506.12460v2)**

> **作者:** Hao Shu
>
> **备注:** 30 pages
>
> **摘要:** In machine learning, discrete decision-making tasks exhibit a fundamental inconsistency between training and inference: models are optimized using continuous-valued outputs, yet evaluated through discrete predictions. This discrepancy arises from the non-differentiability of discretization operations, weakening the alignment between optimization objectives and practical decision outcomes. To address this, we present a theoretical framework for constructing a Binarization-Aware Adjuster (BAA) that integrates binarization behavior directly into gradient-based learning. Central to the approach is a Distance Weight Function (DWF) that dynamically modulates pixel-wise loss contributions based on prediction correctness and proximity to the decision boundary, thereby emphasizing decision-critical regions while de-emphasizing confidently correct samples. Furthermore, a self-adaptive threshold estimation procedure is introduced to better match optimization dynamics with inference conditions. As one of its applications, we implement experiments on the edge detection (ED) task, which also demonstrate the effectiveness of the proposed method experimentally. Beyond binary decision tasks and ED, the proposed framework provides a general strategy for aligning continuous optimization with discrete evaluation and can be extended to multi-valued decision processes in broader structured prediction problems.
>
---
#### [replaced 002] Efficiency vs. Efficacy: Assessing the Compression Ratio-Dice Score Relationship through a Simple Benchmarking Framework for Cerebrovascular 3D Segmentation
- **分类: cs.CV; eess.SP**

- **链接: [https://arxiv.org/pdf/2510.03769v3](https://arxiv.org/pdf/2510.03769v3)**

> **作者:** Shimaa Elbana; Ahmad Kamal; Shahd Ahmed Ali; Ahmad Al-Kabbany
>
> **摘要:** The increasing size and complexity of medical imaging datasets, particularly in 3D formats, present significant barriers to collaborative research and transferability. This study investigates whether the ZFP compression technique can mitigate these challenges without compromising the performance of automated cerebrovascular segmentation, a critical first step in intracranial aneurysm detection. We apply ZFP in both its error tolerance and fixed-rate modes to a large scale, and one of the most recent, datasets in the literature, 3D medical dataset containing ground-truth vascular segmentations. The segmentation quality on the compressed volumes is rigorously compared to the uncompressed baseline (Dice approximately equals 0.8774). Our findings reveal that ZFP can achieve substantial data reduction--up to a 22.89:1 ratio in error tolerance mode--while maintaining a high degree of fidelity, with the mean Dice coefficient remaining high at 0.87656. These results demonstrate that ZFP is a viable and powerful tool for enabling more efficient and accessible research on large-scale medical datasets, fostering broader collaboration across the community.
>
---
#### [replaced 003] Learning 3D Texture-Aware Representations for Parsing Diverse Human Clothing and Body Parts
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2508.06032v2](https://arxiv.org/pdf/2508.06032v2)**

> **作者:** Kiran Chhatre; Christopher Peters; Srikrishna Karanam
>
> **备注:** Association for the Advancement of Artificial Intelligence (AAAI) 2026, 14 pages, 11 figures. Webpage: https://s-pectrum.github.io/
>
> **摘要:** Existing methods for human parsing into body parts and clothing often use fixed mask categories with broad labels that obscure fine-grained clothing types. Recent open-vocabulary segmentation approaches leverage pretrained text-to-image (T2I) diffusion model features for strong zero-shot transfer, but typically group entire humans into a single person category, failing to distinguish diverse clothing or detailed body parts. To address this, we propose Spectrum, a unified network for part-level pixel parsing (body parts and clothing) and instance-level grouping. While diffusion-based open-vocabulary models generalize well across tasks, their internal representations are not specialized for detailed human parsing. We observe that, unlike diffusion models with broad representations, image-driven 3D texture generators maintain faithful correspondence to input images, enabling stronger representations for parsing diverse clothing and body parts. Spectrum introduces a novel repurposing of an Image-to-Texture (I2Tx) diffusion model (obtained by fine-tuning a T2I model on 3D human texture maps) for improved alignment with body parts and clothing. From an input image, we extract human-part internal features via the I2Tx diffusion model and generate semantically valid masks aligned to diverse clothing categories through prompt-guided grounding. Once trained, Spectrum produces semantic segmentation maps for every visible body part and clothing category, ignoring standalone garments or irrelevant objects, for any number of humans in the scene. We conduct extensive cross-dataset experiments, separately assessing body parts, clothing parts, unseen clothing categories, and full-body masks, and demonstrate that Spectrum consistently outperforms baseline methods in prompt-based segmentation.
>
---
#### [replaced 004] DiffEM: Learning from Corrupted Data with Diffusion Models via Expectation Maximization
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.12691v2](https://arxiv.org/pdf/2510.12691v2)**

> **作者:** Danial Hosseintabar; Fan Chen; Giannis Daras; Antonio Torralba; Constantinos Daskalakis
>
> **摘要:** Diffusion models have emerged as powerful generative priors for high-dimensional inverse problems, yet learning them when only corrupted or noisy observations are available remains challenging. In this work, we propose a new method for training diffusion models with Expectation-Maximization (EM) from corrupted data. Our proposed method, DiffEM, utilizes conditional diffusion models to reconstruct clean data from observations in the E-step, and then uses the reconstructed data to refine the conditional diffusion model in the M-step. Theoretically, we provide monotonic convergence guarantees for the DiffEM iteration, assuming appropriate statistical conditions. We demonstrate the effectiveness of our approach through experiments on various image reconstruction tasks.
>
---
#### [replaced 005] Registering the 4D Millimeter Wave Radar Point Clouds Via Generalized Method of Moments
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对4D毫米波雷达点云稀疏、噪声大导致配准困难的问题，提出基于广义矩估计（GMM）的无对应点配准方法，无需显式点对匹配，具有理论一致性。实验表明其精度与鲁棒性优于基准，媲美激光雷达方案。**

- **链接: [https://arxiv.org/pdf/2508.02187v2](https://arxiv.org/pdf/2508.02187v2)**

> **作者:** Xingyi Li; Han Zhang; Ziliang Wang; Yukai Yang; Weidong Chen
>
> **摘要:** 4D millimeter wave radars (4D radars) are new emerging sensors that provide point clouds of objects with both position and radial velocity measurements. Compared to LiDARs, they are more affordable and reliable sensors for robots' perception under extreme weather conditions. On the other hand, point cloud registration is an essential perception module that provides robot's pose feedback information in applications such as Simultaneous Localization and Mapping (SLAM). Nevertheless, the 4D radar point clouds are sparse and noisy compared to those of LiDAR, and hence we shall confront great challenges in registering the radar point clouds. To address this issue, we propose a point cloud registration framework for 4D radars based on Generalized Method of Moments. The method does not require explicit point-to-point correspondences between the source and target point clouds, which is difficult to compute for sparse 4D radar point clouds. Moreover, we show the consistency of the proposed method. Experiments on both synthetic and real-world datasets show that our approach achieves higher accuracy and robustness than benchmarks, and the accuracy is even comparable to LiDAR-based frameworks.
>
---
#### [replaced 006] M4Human: A Large-Scale Multimodal mmWave Radar Benchmark for Human Mesh Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.12378v2](https://arxiv.org/pdf/2512.12378v2)**

> **作者:** Junqiao Fan; Yunjiao Zhou; Yizhuo Yang; Xinyuan Cui; Jiarui Zhang; Lihua Xie; Jianfei Yang; Chris Xiaoxuan Lu; Fangqiang Ding
>
> **摘要:** Human mesh reconstruction (HMR) provides direct insights into body-environment interaction, which enables various immersive applications. While existing large-scale HMR datasets rely heavily on line-of-sight RGB input, vision-based sensing is limited by occlusion, lighting variation, and privacy concerns. To overcome these limitations, recent efforts have explored radio-frequency (RF) mmWave radar for privacy-preserving indoor human sensing. However, current radar datasets are constrained by sparse skeleton labels, limited scale, and simple in-place actions. To advance the HMR research community, we introduce M4Human, the current largest-scale (661K-frame) ($9\times$ prior largest) multimodal benchmark, featuring high-resolution mmWave radar, RGB, and depth data. M4Human provides both raw radar tensors (RT) and processed radar point clouds (RPC) to enable research across different levels of RF signal granularity. M4Human includes high-quality motion capture (MoCap) annotations with 3D meshes and global trajectories, and spans 20 subjects and 50 diverse actions, including in-place, sit-in-place, and free-space sports or rehabilitation movements. We establish benchmarks on both RT and RPC modalities, as well as multimodal fusion with RGB-D modalities. Extensive results highlight the significance of M4Human for radar-based human modeling while revealing persistent challenges under fast, unconstrained motion. The dataset and code will be released after the paper publication.
>
---
#### [replaced 007] Event Camera Meets Mobile Embodied Perception: Abstraction, Algorithm, Acceleration, Application
- **分类: cs.RO; cs.CV**

- **简介: 该论文是一篇综述，旨在解决事件相机在资源受限移动设备上高精度、低延迟感知的挑战。工作包括梳理2014–2025年文献，系统总结事件抽象、算法、软硬件加速及应用（如VO、跟踪、光流、3D重建），并指出未来方向与开源资源。**

- **链接: [https://arxiv.org/pdf/2503.22943v4](https://arxiv.org/pdf/2503.22943v4)**

> **作者:** Haoyang Wang; Ruishan Guo; Pengtao Ma; Ciyu Ruan; Xinyu Luo; Wenhua Ding; Tianyang Zhong; Jingao Xu; Yunhao Liu; Xinlei Chen
>
> **备注:** Accepted by ACM CSUR,35 pages
>
> **摘要:** With the increasing complexity of mobile device applications, these devices are evolving toward high agility. This shift imposes new demands on mobile sensing, particularly in achieving high-accuracy and low-latency. Event-based vision has emerged as a disruptive paradigm, offering high temporal resolution and low latency, making it well-suited for high-accuracy and low-latency sensing tasks on high-agility platforms. However, the presence of substantial noisy events, lack of stable, persistent semantic information, and large data volume pose challenges for event-based data processing on resource-constrained mobile devices. This paper surveys the literature from 2014 to 2025 and presents a comprehensive overview of event-based mobile sensing, encompassing its fundamental principles, event \textit{abstraction} methods, \textit{algorithm} advancements, and both hardware and software \textit{acceleration} strategies. We discuss key \textit{applications} of event cameras in mobile sensing, including visual odometry, object tracking, optical flow, and 3D reconstruction, while highlighting challenges associated with event data processing, sensor fusion, and real-time deployment. Furthermore, we outline future research directions, such as improving the event camera with advanced optics, leveraging neuromorphic computing for efficient processing, and integrating bio-inspired algorithms. To support ongoing research, we provide an open-source \textit{Online Sheet} with recent developments. We hope this survey serves as a reference, facilitating the adoption of event-based vision across diverse applications.
>
---
#### [replaced 008] PerTouch: VLM-Driven Agent for Personalized and Semantic Image Retouching
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.12998v2](https://arxiv.org/pdf/2511.12998v2)**

> **作者:** Zewei Chang; Zheng-Peng Duan; Jianxing Zhang; Chun-Le Guo; Siyu Liu; Hyungju Chun; Hyunhee Park; Zikun Liu; Chongyi Li
>
> **备注:** To appear at AAAI 2026
>
> **摘要:** Image retouching aims to enhance visual quality while aligning with users' personalized aesthetic preferences. To address the challenge of balancing controllability and subjectivity, we propose a unified diffusion-based image retouching framework called PerTouch. Our method supports semantic-level image retouching while maintaining global aesthetics. Using parameter maps containing attribute values in specific semantic regions as input, PerTouch constructs an explicit parameter-to-image mapping for fine-grained image retouching. To improve semantic boundary perception, we introduce semantic replacement and parameter perturbation mechanisms in the training process. To connect natural language instructions with visual control, we develop a VLM-driven agent that can handle both strong and weak user instructions. Equipped with mechanisms of feedback-driven rethinking and scene-aware memory, PerTouch better aligns with user intent and captures long-term preferences. Extensive experiments demonstrate each component's effectiveness and the superior performance of PerTouch in personalized image retouching. Code is available at: https://github.com/Auroral703/PerTouch.
>
---
#### [replaced 009] Human-Centric Open-Future Task Discovery: Formulation, Benchmark, and Scalable Tree-Based Search
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18929v3](https://arxiv.org/pdf/2511.18929v3)**

> **作者:** Zijian Song; Xiaoxin Lin; Tao Pu; Zhenlong Yuan; Guangrun Wang; Liang Lin
>
> **备注:** accepted to AAAI 2026, 10 pages, 9 figures
>
> **摘要:** Recent progress in robotics and embodied AI is largely driven by Large Multimodal Models (LMMs). However, a key challenge remains underexplored: how can we advance LMMs to discover tasks that assist humans in open-future scenarios, where human intentions are highly concurrent and dynamic. In this work, we formalize the problem of Human-centric Open-future Task Discovery (HOTD), focusing particularly on identifying tasks that reduce human effort across plausible futures. To facilitate this study, we propose HOTD-Bench, which features over 2K real-world videos, a semi-automated annotation pipeline, and a simulation-based protocol tailored for open-set future evaluation. Additionally, we propose the Collaborative Multi-Agent Search Tree (CMAST) framework, which decomposes complex reasoning through a multi-agent system and structures the reasoning process through a scalable search tree module. In our experiments, CMAST achieves the best performance on the HOTD-Bench, significantly surpassing existing LMMs. It also integrates well with existing LMMs, consistently improving performance.
>
---
#### [replaced 010] If you can describe it, they can see it: Cross-Modal Learning of Visual Concepts from Textual Descriptions
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.15611v2](https://arxiv.org/pdf/2411.15611v2)**

> **作者:** Carlo Alberto Barbano; Luca Molinaro; Massimiliano Ciranni; Emanuele Aiello; Vito Paolo Pastore; Marco Grangetto
>
> **备注:** 27 pages. Under review
>
> **摘要:** Humans can visualize new and unknown concepts from their natural language description, based on their experience and previous knowledge. Insipired by this, we present a way to extend this ability to Vision-Language Models (VLMs), teaching them novel concepts by only using a textual description. We refer to this approach as Knowledge Transfer (KT). Our hypothesis is that the knowledge of a pre-trained VLM can be re-used to represent previously unknown concepts. Provided with a textual description of the novel concept, KT works by aligning relevant features of the visual encoder, obtained through model inversion, to its text representation. Differently from approaches relying on visual examples or external generative models, KT transfers knowledge within the same VLM by injecting visual knowledge directly from the text. Through an extensive evaluation on several VLM tasks, including classification, segmentation, image-text retrieval, and captioning, we show that: 1) KT can efficiently introduce new visual concepts from a single textual description; 2) the same principle can be used to refine the representation of existing concepts; and 3) KT significantly improves the performance of zero-shot VLMs.
>
---
#### [replaced 011] Scaling Instruction-Based Video Editing with a High-Quality Synthetic Dataset
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.15742v2](https://arxiv.org/pdf/2510.15742v2)**

> **作者:** Qingyan Bai; Qiuyu Wang; Hao Ouyang; Yue Yu; Hanlin Wang; Wen Wang; Ka Leong Cheng; Shuailei Ma; Yanhong Zeng; Zichen Liu; Yinghao Xu; Yujun Shen; Qifeng Chen
>
> **备注:** Project page: https://ezioby.github.io/Ditto_page Code: https://github.com/EzioBy/Ditto
>
> **摘要:** Instruction-based video editing promises to democratize content creation, yet its progress is severely hampered by the scarcity of large-scale, high-quality training data. We introduce Ditto, a holistic framework designed to tackle this fundamental challenge. At its heart, Ditto features a novel data generation pipeline that fuses the creative diversity of a leading image editor with an in-context video generator, overcoming the limited scope of existing models. To make this process viable, our framework resolves the prohibitive cost-quality trade-off by employing an efficient, distilled model architecture augmented by a temporal enhancer, which simultaneously reduces computational overhead and improves temporal coherence. Finally, to achieve full scalability, this entire pipeline is driven by an intelligent agent that crafts diverse instructions and rigorously filters the output, ensuring quality control at scale. Using this framework, we invested over 12,000 GPU-days to build Ditto-1M, a new dataset of one million high-fidelity video editing examples. We trained our model, Editto, on Ditto-1M with a curriculum learning strategy. The results demonstrate superior instruction-following ability and establish a new state-of-the-art in instruction-based video editing.
>
---
#### [replaced 012] From Segments to Scenes: Temporal Understanding in Autonomous Driving via Vision-Language Model
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.05277v2](https://arxiv.org/pdf/2512.05277v2)**

> **作者:** Kevin Cannons; Saeed Ranjbar Alvar; Mohammad Asiful Hossain; Ahmad Rezaei; Mohsen Gholami; Alireza Heidarikhazaei; Zhou Weimin; Yong Zhang; Mohammad Akbari
>
> **摘要:** Temporal understanding in autonomous driving (AD) remains a significant challenge, even for recent state-of-the-art (SoTA) Vision-Language Models (VLMs). Prior work has introduced datasets and benchmarks aimed at improving temporal reasoning, but these have emphasized other video content, including sports, cooking, and movies. No existing benchmark focuses exclusively on the unique challenges of temporal understanding in ego-centric AD footage. To fill this gap, the Temporal Understanding in Autonomous Driving (TAD) benchmark is presented, which evaluates VLMs' ability to capture the dynamic relationships between actions in AD. TAD comprises nearly 6,000 question-answer (QA) pairs, spanning 7 human-designed tasks. In addition, an evaluation is performed that consists of 9 closed- and open-source generalist models as well as SoTA AD specialist models. When applied to TAD, current SoTA models demonstrated substandard accuracies, largely due to imperfect fine-grained motion understanding. To improve motion understanding and overall accuracy on TAD, two novel training-free solutions are proposed: Scene-CoT, that leverages Chain-of-Thought (CoT) and TCogMap, which incorporates an ego-centric temporal cognitive map. The proposed approaches are integrated with existing VLMs and improve average accuracy on TAD by up to 17.72%. By introducing TAD, benchmarking multiple SoTA models, and proposing effective enhancements, this work aims to catalyze future research on temporal understanding in AD. The benchmark and evaluation code are available at \href{https://huggingface.co/datasets/vbdai/TAD}{Hugging Face} and \href{https://github.com/vbdi/tad_bench}{Github}, respectively.
>
---
#### [replaced 013] ChronoSelect: Robust Learning with Noisy Labels via Dynamics Temporal Memory
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2507.18183v2](https://arxiv.org/pdf/2507.18183v2)**

> **作者:** Jianchao Wang; Qingfeng Li; Pengcheng Zheng; Xiaorong Pu; Yazhou Ren
>
> **摘要:** Training deep neural networks on real-world datasets is often hampered by the presence of noisy labels, which can be memorized by over-parameterized models, leading to significant degradation in generalization performance. While existing methods for learning with noisy labels (LNL) have made considerable progress, they fundamentally suffer from static snapshot evaluations and fail to leverage the rich temporal dynamics of learning evolution. In this paper, we propose ChronoSelect (chrono denoting its temporal nature), a novel framework featuring an innovative four-stage memory architecture that compresses prediction history into compact temporal distributions. Our unique sliding update mechanism with controlled decay maintains only four dynamic memory units per sample, progressively emphasizing recent patterns while retaining essential historical knowledge. This enables precise three-way sample partitioning into clean, boundary, and noisy subsets through temporal trajectory analysis and dual-branch consistency. Theoretical guarantees prove the mechanism's convergence and stability under noisy conditions. Extensive experiments demonstrate ChronoSelect's state-of-the-art performance across synthetic and real-world benchmarks.
>
---
#### [replaced 014] MMGR: Multi-Modal Generative Reasoning
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出MMGR评估框架，旨在解决视频/图像生成模型缺乏推理能力的问题。它从物理、逻辑、3D/2D空间、时间五方面评测生成式模型在抽象推理、具身导航、物理常识三类任务中的多模态生成推理能力，并揭示当前模型重感知轻因果、弱全局一致等缺陷。**

- **链接: [https://arxiv.org/pdf/2512.14691v2](https://arxiv.org/pdf/2512.14691v2)**

> **作者:** Zefan Cai; Haoyi Qiu; Tianyi Ma; Haozhe Zhao; Gengze Zhou; Kung-Hsiang Huang; Parisa Kordjamshidi; Minjia Zhang; Wen Xiao; Jiuxiang Gu; Nanyun Peng; Junjie Hu
>
> **备注:** work in progress
>
> **摘要:** Video foundation models generate visually realistic and temporally coherent content, but their reliability as world simulators depends on whether they capture physical, logical, and spatial constraints. Existing metrics such as Frechet Video Distance (FVD) emphasize perceptual quality and overlook reasoning failures, including violations of causality, physics, and global consistency. We introduce MMGR (Multi-Modal Generative Reasoning Evaluation and Benchmark), a principled evaluation framework based on five reasoning abilities: Physical, Logical, 3D Spatial, 2D Spatial, and Temporal. MMGR evaluates generative reasoning across three domains: Abstract Reasoning (ARC-AGI, Sudoku), Embodied Navigation (real-world 3D navigation and localization), and Physical Commonsense (sports and compositional interactions). MMGR applies fine-grained metrics that require holistic correctness across both video and image generation. We benchmark leading video models (Veo-3, Sora-2, Wan-2.2) and image models (Nano-banana, Nano-banana Pro, GPT-4o-image, Qwen-image), revealing strong performance gaps across domains. Models show moderate success on Physical Commonsense tasks but perform poorly on Abstract Reasoning (below 10 percent accuracy on ARC-AGI) and struggle with long-horizon spatial planning in embodied settings. Our analysis highlights key limitations in current models, including overreliance on perceptual data, weak global state consistency, and objectives that reward visual plausibility over causal correctness. MMGR offers a unified diagnostic benchmark and a path toward reasoning-aware generative world models.
>
---
#### [replaced 015] dots.ocr: Multilingual Document Layout Parsing in a Single Vision-Language Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.02498v4](https://arxiv.org/pdf/2512.02498v4)**

> **作者:** Yumeng Li; Guang Yang; Hao Liu; Bowen Wang; Colin Zhang
>
> **摘要:** Document Layout Parsing serves as a critical gateway for Artificial Intelligence (AI) to access and interpret the world's vast stores of structured knowledge. This process,which encompasses layout detection, text recognition, and relational understanding, is particularly crucial for empowering next-generation Vision-Language Models. Current methods, however, rely on fragmented, multi-stage pipelines that suffer from error propagation and fail to leverage the synergies of joint training. In this paper, we introduce dots_ocr, a single Vision-Language Model that, for the first time, demonstrates the advantages of jointly learning three core tasks within a unified, end-to-end framework. This is made possible by a highly scalable data engine that synthesizes a vast multilingual corpus, empowering the model to deliver robust performance across a wide array of tasks, encompassing diverse languages, layouts, and domains. The efficacy of our unified paradigm is validated by state-of-the-art performance on the comprehensive OmniDocBench. Furthermore, to catalyze research in global document intelligence, we introduce XDocParse, a challenging new benchmark spanning 126 languages. On this benchmark, dots_ocr achieves state-of-the-art performance, delivering an approximately 10% relative improvement and demonstrating strong multilingual capability.
>
---
#### [replaced 016] SynJAC: Synthetic-data-driven Joint-granular Adaptation and Calibration for Domain Specific Scanned Document Key Information Extraction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2410.01609v2](https://arxiv.org/pdf/2410.01609v2)**

> **作者:** Yihao Ding; Soyeon Caren Han; Zechuan Li; Hyunsuk Chung
>
> **备注:** Accepted for publication in Information Fusion
>
> **摘要:** Visually Rich Documents (VRDs), comprising elements such as charts, tables, and paragraphs, convey complex information across diverse domains. However, extracting key information from these documents remains labour-intensive, particularly for scanned formats with inconsistent layouts and domain-specific requirements. Despite advances in pretrained models for VRD understanding, their dependence on large annotated datasets for fine-tuning hinders scalability. This paper proposes \textbf{SynJAC} (Synthetic-data-driven Joint-granular Adaptation and Calibration), a method for key information extraction in scanned documents. SynJAC leverages synthetic, machine-generated data for domain adaptation and employs calibration on a small, manually annotated dataset to mitigate noise. By integrating fine-grained and coarse-grained document representation learning, SynJAC significantly reduces the need for extensive manual labelling while achieving competitive performance. Extensive experiments demonstrate its effectiveness in domain-specific and scanned VRD scenarios.
>
---
#### [replaced 017] AdSum: Two-stream Audio-visual Summarization for Automated Video Advertisement Clipping
- **分类: cs.CV; cs.IR; cs.MM**

- **链接: [https://arxiv.org/pdf/2510.26569v2](https://arxiv.org/pdf/2510.26569v2)**

> **作者:** Wen Xie; Yanjun Zhu; Gijs Overgoor; Yakov Bart; Agata Lapedriza Garcia; Sarah Ostadabbas
>
> **备注:** Accepted at 32nd International Conference on MultiMedia Modeling
>
> **摘要:** Advertisers commonly need multiple versions of the same advertisement (ad) at varying durations for a single campaign. The traditional approach involves manually selecting and re-editing shots from longer video ads to create shorter versions, which is labor-intensive and time-consuming. In this paper, we introduce a framework for automated video ad clipping using video summarization techniques. We are the first to frame video clipping as a shot selection problem, tailored specifically for advertising. Unlike existing general video summarization methods that primarily focus on visual content, our approach emphasizes the critical role of audio in advertising. To achieve this, we develop a two-stream audio-visual fusion model that predicts the importance of video frames, where importance is defined as the likelihood of a frame being selected in the firm-produced short ad. To address the lack of ad-specific datasets, we present AdSum204, a novel dataset comprising 102 pairs of 30-second and 15-second ads from real advertising campaigns. Extensive experiments demonstrate that our model outperforms state-of-the-art methods across various metrics, including Average Precision, Area Under Curve, Spearman, and Kendall. The dataset and code are available at https://github.com/ostadabbas/AdSum204.
>
---
#### [replaced 018] Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.24431v2](https://arxiv.org/pdf/2505.24431v2)**

> **作者:** Bozhong Zheng; Jinye Gan; Xiaohao Xu; Xintao Chen; Wenqiao Li; Xiaonan Huang; Na Ni; Yingna Wu
>
> **摘要:** 3D point cloud anomaly detection is essential for robust vision systems but is challenged by pose variations and complex geometric anomalies. Existing patch-based methods often suffer from geometric fidelity issues due to discrete voxelization or projection-based representations, limiting fine-grained anomaly localization. We introduce Pose-Aware Signed Distance Field (PASDF), a novel framework that integrates 3D anomaly detection and repair by learning a continuous, pose-invariant shape representation. PASDF leverages a Pose Alignment Module for canonicalization and a SDF Network to dynamically incorporate pose, enabling implicit learning of high-fidelity anomaly repair templates from the continuous SDF. This facilitates precise pixel-level anomaly localization through an Anomaly-Aware Scoring Module. Crucially, the continuous 3D representation in PASDF extends beyond detection, facilitating in-situ anomaly repair. Experiments on Real3D-AD and Anomaly-ShapeNet demonstrate state-of-the-art performance, achieving high object-level AUROC scores of 80.2% and 90.0%, respectively. These results highlight the effectiveness of continuous geometric representations in advancing 3D anomaly detection and facilitating practical anomaly region repair. The code is available at https://github.com/ZZZBBBZZZ/PASDF to support further research.
>
---
#### [replaced 019] ViRC: Enhancing Visual Interleaved Mathematical CoT with Reason Chunking
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.14654v2](https://arxiv.org/pdf/2512.14654v2)**

> **作者:** Lihong Wang; Liangqi Li; Weiwei Feng; Jiamin Wu; Changtao Miao; Tieru Wu; Rui Ma; Bo Zhang; Zhe Li
>
> **备注:** Code is available at https://github.com/Leon-LihongWang/ViRC
>
> **摘要:** CoT has significantly enhanced the reasoning ability of LLMs while it faces challenges when extended to multimodal domains, particularly in mathematical tasks. Existing MLLMs typically perform textual reasoning solely from a single static mathematical image, overlooking dynamic visual acquisition during reasoning. In contrast, humans repeatedly examine visual image and employ step-by-step reasoning to prove intermediate propositions. This strategy of decomposing the problem-solving process into key logical nodes adheres to Miller's Law in cognitive science. Inspired by this insight, we propose a ViRC framework for multimodal mathematical tasks, introducing a Reason Chunking mechanism that structures multimodal mathematical CoT into consecutive Critical Reasoning Units (CRUs) to simulate human expert problem-solving patterns. CRUs ensure intra-unit textual coherence for intermediate proposition verification while integrating visual information across units to generate subsequent propositions and support structured reasoning. To this end, we present CRUX dataset by using three visual tools and four reasoning patterns to provide explicitly annotated CRUs across multiple reasoning paths for each mathematical problem. Leveraging the CRUX dataset, we propose a progressive training strategy inspired by human cognitive learning, which includes Instructional SFT, Practice SFT, and Strategic RL, aimed at further strengthening the Reason Chunking ability of the model. The resulting ViRC-7B model achieves a 18.8% average improvement over baselines across multiple mathematical benchmarks. Code is available at https://github.com/Leon-LihongWang/ViRC.
>
---
#### [replaced 020] MedChat: A Multi-Agent Framework for Multimodal Diagnosis with Large Language Models
- **分类: cs.MA; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.07400v3](https://arxiv.org/pdf/2506.07400v3)**

> **作者:** Philip R. Liu; Sparsh Bansal; Jimmy Dinh; Aditya Pawar; Ramani Satishkumar; Shail Desai; Neeraj Gupta; Xin Wang; Shu Hu
>
> **摘要:** The integration of deep learning-based glaucoma detection with large language models (LLMs) presents an automated strategy to mitigate ophthalmologist shortages and improve clinical reporting efficiency. However, applying general LLMs to medical imaging remains challenging due to hallucinations, limited interpretability, and insufficient domain-specific medical knowledge, which can potentially reduce clinical accuracy. Although recent approaches combining imaging models with LLM reasoning have improved reporting, they typically rely on a single generalist agent, restricting their capacity to emulate the diverse and complex reasoning found in multidisciplinary medical teams. To address these limitations, we propose MedChat, a multi-agent diagnostic framework and platform that combines specialized vision models with multiple role-specific LLM agents, all coordinated by a director agent. This design enhances reliability, reduces hallucination risk, and enables interactive diagnostic reporting through an interface tailored for clinical review and educational use. Code available at https://github.com/Purdue-M2/MedChat.
>
---
#### [replaced 021] Few-Shot Multimodal Medical Imaging: A Theoretical Framework
- **分类: stat.ML; cs.AI; cs.CV; cs.LG; eess.IV**

- **链接: [https://arxiv.org/pdf/2511.01140v2](https://arxiv.org/pdf/2511.01140v2)**

> **作者:** Md Talha Mohsin; Ismail Abdulrashid
>
> **备注:** 6 Pages
>
> **摘要:** Medical imaging often operates under limited labeled data, especially in rare disease and low resource clinical environments. Existing multimodal and meta learning approaches improve performance in these settings but lack a theoretical explanation of why or when they succeed. This paper presents a unified theoretical framework for few shot multimodal medical imaging that jointly characterizes sample complexity, uncertainty quantification, and interpretability. Using PAC learning, VC theory, and PAC Bayesian analysis, we derive bounds that describe the minimum number of labeled samples required for reliable performance and show how complementary modalities reduce effective capacity through an information gain term. We further introduce a formal metric for explanation stability, proving that explanation variance decreases at an inverse n rate. A sequential Bayesian interpretation of Chain of Thought reasoning is also developed to show stepwise posterior contraction. To illustrate these ideas, we implement a controlled multimodal dataset and evaluate an additive CNN MLP fusion model under few shot regimes, confirming predicted multimodal gains, modality interference at larger sample sizes, and shrinking predictive uncertainty. Together, the framework provides a principled foundation for designing data efficient, uncertainty aware, and interpretable diagnostic models in low resource settings.
>
---
#### [replaced 022] Omni-Effects: Unified and Spatially-Controllable Visual Effects Generation
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.07981v4](https://arxiv.org/pdf/2508.07981v4)**

> **作者:** Fangyuan Mao; Aiming Hao; Jintao Chen; Dongxia Liu; Xiaokun Feng; Jiashu Zhu; Meiqi Wu; Chubin Chen; Jiahong Wu; Xiangxiang Chu
>
> **备注:** Accepted to AAAI2026
>
> **摘要:** Visual effects (VFX) are essential visual enhancements fundamental to modern cinematic production. Although video generation models offer cost-efficient solutions for VFX production, current methods are constrained by per-effect LoRA training, which limits generation to single effects. This fundamental limitation impedes applications that require spatially controllable composite effects, i.e., the concurrent generation of multiple effects at designated locations. However, integrating diverse effects into a unified framework faces major challenges: interference from effect variations and spatial uncontrollability during multi-VFX joint training. To tackle these challenges, we propose Omni-Effects, a first unified framework capable of generating prompt-guided effects and spatially controllable composite effects. The core of our framework comprises two key innovations: (1) LoRA-based Mixture of Experts (LoRA-MoE), which employs a group of expert LoRAs, integrating diverse effects within a unified model while effectively mitigating cross-task interference. (2) Spatial-Aware Prompt (SAP) incorporates spatial mask information into the text token, enabling precise spatial control. Furthermore, we introduce an Independent-Information Flow (IIF) module integrated within the SAP, isolating the control signals corresponding to individual effects to prevent any unwanted blending. To facilitate this research, we construct a comprehensive VFX dataset Omni-VFX via a novel data collection pipeline combining image editing and First-Last Frame-to-Video (FLF2V) synthesis, and introduce a dedicated VFX evaluation framework for validating model performance. Extensive experiments demonstrate that Omni-Effects achieves precise spatial control and diverse effect generation, enabling users to specify both the category and location of desired effects.
>
---
#### [replaced 023] FitPro: A Zero-Shot Framework for Interactive Text-based Pedestrian Retrieval in Open World
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2509.16674v3](https://arxiv.org/pdf/2509.16674v3)**

> **作者:** Zengli Luo; Canlong Zhang; Xiaochun Lu; Zhixin Li
>
> **备注:** 12pages,6 figures
>
> **摘要:** Text-based Pedestrian Retrieval (TPR) deals with retrieving specific target pedestrians in visual scenes according to natural language descriptions. Although existing methods have achieved progress under constrained settings, interactive retrieval in the open-world scenario still suffers from limited model generalization and insufficient semantic understanding. To address these challenges, we propose FitPro, an open-world interactive zero-shot TPR framework with enhanced semantic comprehension and cross-scene adaptability. FitPro has three innovative components: Feature Contrastive Decoding (FCD), Incremental Semantic Mining (ISM), and Query-aware Hierarchical Retrieval (QHR). The FCD integrates prompt-guided contrastive decoding to generate high-quality structured pedestrian descriptions from denoised images, effectively alleviating semantic drift in zero-shot scenarios. The ISM constructs holistic pedestrian representations from multi-view observations to achieve global semantic modeling in multi-turn interactions, thereby improving robustness against viewpoint shifts and fine-grained variations in descriptions. The QHR dynamically optimizes the retrieval pipeline according to query types, enabling efficient adaptation to multi-modal and multi-view inputs. Extensive experiments on five public datasets and two evaluation protocols demonstrate that FitPro significantly overcomes the generalization limitations and semantic modeling constraints of existing methods in interactive retrieval, paving the way for practical deployment.
>
---
#### [replaced 024] One-Cycle Structured Pruning via Stability-Driven Subnetwork Search
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2501.13439v2](https://arxiv.org/pdf/2501.13439v2)**

> **作者:** Deepak Ghimire; Dayoung Kil; Seonghwan Jeong; Jaesik Park; Seong-heum Kim
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Existing structured pruning methods typically rely on multi-stage training procedures that incur high computational costs. Pruning at initialization aims to reduce this burden but often suffers from degraded performance. To address these limitations, we propose an efficient one-cycle structured pruning framework that integrates pre-training, pruning, and fine-tuning into a single training cycle without sacrificing accuracy. The key idea is to identify an optimal sub-network during the early stages of training, guided by norm-based group saliency criteria and structured sparsity regularization. We introduce a novel pruning indicator that detects a stable pruning epoch by measuring the similarity between pruning sub-networks across consecutive training epochs. In addition, group sparsity regularization accelerates convergence, further reducing overall training time. Extensive experiments on CIFAR-10, CIFAR-100, and ImageNet using VGG, ResNet, and MobileNet architectures demonstrate that the proposed method achieves state-of-the-art accuracy while being among the most efficient structured pruning frameworks in terms of training cost. Code is available at https://github.com/ghimiredhikura/OCSPruner.
>
---
#### [replaced 025] GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2505.15208v3](https://arxiv.org/pdf/2505.15208v3)**

> **作者:** Wenjie Liu; Zhongliang Liu; Junwei Shu; Changbo Wang; Yang Li
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Transferring 2D textures onto complex 3D scenes plays a vital role in enhancing the efficiency and controllability of 3D multimedia content creation. However, existing 3D style transfer methods primarily focus on transferring abstract artistic styles to 3D scenes. These methods often overlook the geometric information of the scene, which makes it challenging to achieve high-quality 3D texture transfer results. In this paper, we present GT2-GS, a geometry-aware texture transfer framework for gaussian splatting. First, we propose a geometry-aware texture transfer loss that enables view-consistent texture transfer by leveraging prior view-dependent feature information and texture features augmented with additional geometric parameters. Moreover, an adaptive fine-grained control module is proposed to address the degradation of scene information caused by low-granularity texture features. Finally, a geometry preservation branch is introduced. This branch refines the geometric parameters using additionally bound Gaussian color priors, thereby decoupling the optimization objectives of appearance and geometry. Extensive experiments demonstrate the effectiveness and controllability of our method. Through geometric awareness, our approach achieves texture transfer results that better align with human visual perception. Our homepage is available at https://vpx-ecnu.github.io/GT2-GS-website.
>
---
#### [replaced 026] TerraFusion: Joint Generation of Terrain Geometry and Texture Using Latent Diffusion Models
- **分类: cs.GR; cs.CV**

- **链接: [https://arxiv.org/pdf/2505.04050v3](https://arxiv.org/pdf/2505.04050v3)**

> **作者:** Kazuki Higo; Toshiki Kanai; Yuki Endo; Yoshihiro Kanamori
>
> **摘要:** 3D terrain models are essential in fields such as video game development and film production. Since surface color often correlates with terrain geometry, capturing this relationship is crucial to achieving realism. However, most existing methods generate either a heightmap or a texture, without sufficiently accounting for the inherent correlation. In this paper, we propose a method that jointly generates terrain heightmaps and textures using a latent diffusion model. First, we train the model in an unsupervised manner to randomly generate paired heightmaps and textures. Then, we perform supervised learning of an external adapter to enable user control via hand-drawn sketches. Experiments show that our approach allows intuitive terrain generation while preserving the correlation between heightmaps and textures.
>
---
#### [replaced 027] Benchmarking and Mitigating Sycophancy in Medical Vision Language Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.21979v3](https://arxiv.org/pdf/2509.21979v3)**

> **作者:** Zikun Guo; Jingwei Lv; Xinyue Xu; Shu Yang; Jun Wen; Di Wang; Lijie Hu
>
> **备注:** 19figures, 61pages
>
> **摘要:** Visual language models (VLMs) have the potential to transform medical workflows. However, the deployment is limited by sycophancy. Despite this serious threat to patient safety, a systematic benchmark remains lacking. This paper addresses this gap by introducing a Medical benchmark that applies multiple templates to VLMs in a hierarchical medical visual question answering task. We find that current VLMs are highly susceptible to visual cues, with failure rates showing a correlation to model size or overall accuracy. we discover that perceived authority and user mimicry are powerful triggers, suggesting a bias mechanism independent of visual data. To overcome this, we propose a Visual Information Purification for Evidence based Responses (VIPER) strategy that proactively filters out non-evidence-based social cues, thereby reinforcing evidence based reasoning. VIPER reduces sycophancy while maintaining interpretability and consistently outperforms baseline methods, laying the necessary foundation for the robust and secure integration of VLMs.
>
---
#### [replaced 028] PP-Motion: Physical-Perceptual Fidelity Evaluation for Human Motion Generation
- **分类: cs.CV; cs.MM**

- **链接: [https://arxiv.org/pdf/2508.08179v2](https://arxiv.org/pdf/2508.08179v2)**

> **作者:** Sihan Zhao; Zixuan Wang; Tianyu Luan; Jia Jia; Wentao Zhu; Jiebo Luo; Junsong Yuan; Nan Xi
>
> **备注:** Accepted by ACM Multimedia 2025
>
> **摘要:** Human motion generation has found widespread applications in AR/VR, film, sports, and medical rehabilitation, offering a cost-effective alternative to traditional motion capture systems. However, evaluating the fidelity of such generated motions is a crucial, multifaceted task. Although previous approaches have attempted at motion fidelity evaluation using human perception or physical constraints, there remains an inherent gap between human-perceived fidelity and physical feasibility. Moreover, the subjective and coarse binary labeling of human perception further undermines the development of a robust data-driven metric. We address these issues by introducing a physical labeling method. This method evaluates motion fidelity by calculating the minimum modifications needed for a motion to align with physical laws. With this approach, we are able to produce fine-grained, continuous physical alignment annotations that serve as objective ground truth. With these annotations, we propose PP-Motion, a novel data-driven metric to evaluate both physical and perceptual fidelity of human motion. To effectively capture underlying physical priors, we employ Pearson's correlation loss for the training of our metric. Additionally, by incorporating a human-based perceptual fidelity loss, our metric can capture fidelity that simultaneously considers both human perception and physical alignment. Experimental results demonstrate that our metric, PP-Motion, not only aligns with physical laws but also aligns better with human perception of motion fidelity than previous work.
>
---
#### [replaced 029] Chain-of-Evidence Multimodal Reasoning for Few-shot Temporal Action Localization
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2504.13460v4](https://arxiv.org/pdf/2504.13460v4)**

> **作者:** Mengshi Qi; Hongwei Ji; Wulian Yun; Xianlin Zhang; Huadong Ma
>
> **摘要:** Traditional temporal action localization (TAL) methods rely on large amounts of detailed annotated data, whereas few-shot TAL reduces this dependence by using only a few training samples to identify unseen action categories. However, existing few-shot TAL methods typically focus solely on video-level information, neglecting textual information, which can provide valuable semantic support for the action localization task. To address these issues, in this work, we propose a new few-shot temporal action localization method by Chain-of-Evidence multimodal reasoning to improve localization performance. Specifically, we design a novel few-shot learning framework to capture action commonalities and variations, which includes a semantic-aware text-visual alignment module designed to align the query and support videos at different levels. Meanwhile, to better express the temporal dependencies and causal relationships between actions at the textual level, we design a Chain-of-Evidence (CoE) reasoning method that progressively guides the Vision Language Model (VLM) and Large Language Model (LLM) to generate CoE text descriptions for videos. The generated texts can capture more variance of action than visual features. We conduct extensive experiments on the publicly available ActivityNet1.3, THUMOS14 and our newly collected Human-related Anomaly Localization Dataset. The experimental results demonstrate that our proposed method significantly outperforms existing methods in single-instance and multi-instance scenarios. Our source code and data are available at https://github.com/MICLAB-BUPT/VAL-VLM.
>
---
#### [replaced 030] 3DLLM-Mem: Long-Term Spatial-Temporal Memory for Embodied 3D Large Language Model
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文面向具身智能任务，解决LLM在动态3D环境中长期时空记忆建模不足的问题。提出3DMem-Bench基准和3DLLM-Mem模型，通过工作记忆查询与选择性融合 episodic 记忆，提升长程空间-时间推理与动作规划能力。**

- **链接: [https://arxiv.org/pdf/2505.22657v2](https://arxiv.org/pdf/2505.22657v2)**

> **作者:** Wenbo Hu; Yining Hong; Yanjun Wang; Leison Gao; Zibu Wei; Xingcheng Yao; Nanyun Peng; Yonatan Bitton; Idan Szpektor; Kai-Wei Chang
>
> **备注:** demos at: https://3dllm-mem.github.io
>
> **摘要:** Humans excel at performing complex tasks by leveraging long-term memory across temporal and spatial experiences. In contrast, current Large Language Models (LLMs) struggle to effectively plan and act in dynamic, multi-room 3D environments. We posit that part of this limitation is due to the lack of proper 3D spatial-temporal memory modeling in LLMs. To address this, we first introduce 3DMem-Bench, a comprehensive benchmark comprising over 26,000 trajectories and 2,892 embodied tasks, question-answering and captioning, designed to evaluate an agent's ability to reason over long-term memory in 3D environments. Second, we propose 3DLLM-Mem, a novel dynamic memory management and fusion model for embodied spatial-temporal reasoning and actions in LLMs. Our model uses working memory tokens, which represents current observations, as queries to selectively attend to and fuse the most useful spatial and temporal features from episodic memory, which stores past observations and interactions. Our approach allows the agent to focus on task-relevant information while maintaining memory efficiency in complex, long-horizon environments. Experimental results demonstrate that 3DLLM-Mem achieves state-of-the-art performance across various tasks, outperforming the strongest baselines by 16.5% in success rate on 3DMem-Bench's most challenging in-the-wild embodied tasks.
>
---
#### [replaced 031] Weakly Supervised Pneumonia Localization from Chest X-Rays Using Deep Neural Network and Grad-CAM Explanations
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.00456v5](https://arxiv.org/pdf/2511.00456v5)**

> **作者:** Kiran Shahi; Anup Bagale
>
> **备注:** https://github.com/kiranshahi/pneumonia-analysis
>
> **摘要:** Chest X-ray imaging is commonly used to diagnose pneumonia, but accurately localizing the pneumonia-affected regions typically requires detailed pixel-level annotations, which are costly and time consuming to obtain. To address this limitation, this study proposes a weakly supervised deep learning framework for pneumonia classification and localization using Gradient-weighted Class Activation Mapping (Grad-CAM). Instead of relying on costly pixel-level annotations, the proposed method utilizes image-level labels to generate clinically meaningful heatmaps that highlight pneumonia-affected regions. Furthermore, we evaluate seven pre-trained deep learning models, including a Vision Transformer, under identical training conditions, using focal loss and patient-wise splits to prevent data leakage. Experimental results suggest that all models achieved high classification accuracy (96--98\%), with ResNet-18 and EfficientNet-B0 showing the best overall performance and MobileNet-V3 providing an efficient lightweight alternative. Grad-CAM heatmap visualizations confirm that the proposed methods focus on clinically relevant lung regions, supporting the use of explainable AI for radiological diagnostics. Overall, this work highlights the potential of weakly supervised, explainable models that enhance transparency and clinical trust in AI-assisted pneumonia screening.
>
---
#### [replaced 032] Abstract 3D Perception for Spatial Intelligence in Vision-Language Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.10946v2](https://arxiv.org/pdf/2511.10946v2)**

> **作者:** Yifan Liu; Fangneng Zhan; Kaichen Zhou; Yilun Du; Paul Pu Liang; Hanspeter Pfister
>
> **摘要:** Vision-language models (VLMs) struggle with 3D-related tasks such as spatial cognition and physical understanding, which are crucial for real-world applications like robotics and embodied agents. We attribute this to a modality gap between the 3D tasks and the 2D training of VLM, which led to inefficient retrieval of 3D information from 2D input. To bridge this gap, we introduce SandboxVLM, a simple yet effective framework that leverages abstract bounding boxes to encode geometric structure and physical kinematics for VLM. Specifically, we design a 3D Sandbox reconstruction and perception pipeline comprising four stages: generating multi-view priors with abstract control, proxy elevation, multi-view voting and clustering, and 3D-aware reasoning. Evaluated in zero-shot settings across multiple benchmarks and VLM backbones, our approach consistently improves spatial intelligence, achieving an 8.3\% gain on SAT Real compared with baseline methods for instance. These results demonstrate that equipping VLMs with a 3D abstraction substantially enhances their 3D reasoning ability without additional training, suggesting new possibilities for general-purpose embodied intelligence.
>
---
#### [replaced 033] DriveMLM: Aligning Multi-Modal Large Language Models with Behavioral Planning States for Autonomous Driving
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2312.09245v3](https://arxiv.org/pdf/2312.09245v3)**

> **作者:** Erfei Cui; Wenhai Wang; Zhiqi Li; Jiangwei Xie; Haoming Zou; Hanming Deng; Gen Luo; Lewei Lu; Xizhou Zhu; Jifeng Dai
>
> **备注:** Accepted to Visual Intelligence
>
> **摘要:** Large language models (LLMs) have opened up new possibilities for intelligent agents, endowing them with human-like thinking and cognitive abilities. In this work, we delve into the potential of large language models (LLMs) in autonomous driving (AD). We introduce DriveMLM, an LLM-based AD framework that can perform close-loop autonomous driving in realistic simulators. To this end, (1) we bridge the gap between the language decisions and the vehicle control commands by standardizing the decision states according to the off-the-shelf motion planning module. (2) We employ a multimodal LLM (MLLM) to model the behavior planning module of a module AD system, which uses driving rules, user commands, and inputs from various sensors (e.g., camera, lidar) as input and makes driving decisions and provide explanations; This model can plug-and-play in existing AD systems such as Autopilot and Apollo for close-loop driving. (3) We design an effective data engine to collect a dataset that includes decision state and corresponding explanation annotation for model training and evaluation. We conduct extensive experiments and show that replacing the decision-making modules of the Autopilot and Apollo with DriveMLM resulted in significant improvements of 3.2 and 4.7 points on the CARLA Town05 Long respectively, demonstrating the effectiveness of our model. We hope this work can serve as a baseline for autonomous driving with LLMs.
>
---
#### [replaced 034] MedicoSAM: Robust Improvement of SAM for Medical Imaging
- **分类: eess.IV; cs.CV**

- **链接: [https://arxiv.org/pdf/2501.11734v2](https://arxiv.org/pdf/2501.11734v2)**

> **作者:** Anwai Archit; Luca Freckmann; Constantin Pape
>
> **摘要:** Medical image segmentation is an important analysis task in clinical practice and research. Deep learning has massively advanced the field, but current approaches are mostly based on models trained for a specific task. Training such models or adapting them to a new condition is costly due to the need for (manually) labeled data. The emergence of vision foundation models, especially Segment Anything, offers a path to universal segmentation for medical images, overcoming these issues. Here, we study how to improve Segment Anything for medical images by comparing different finetuning strategies on a large and diverse dataset. We evaluate the finetuned models on a wide range of interactive and (automatic) semantic segmentation tasks. We find that the performance can be clearly improved for interactive segmentation. However, semantic segmentation does not benefit from pretraining on medical images. Our best model, MedicoSAM, is publicly available at https://github.com/computational-cell-analytics/medico-sam. We show that it is compatible with existing tools for data annotation and believe that it will be of great practical value.
>
---
#### [replaced 035] Online Navigation Refinement: Achieving Lane-Level Guidance by Associating Standard-Definition and Online Perception Maps
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2507.07487v3](https://arxiv.org/pdf/2507.07487v3)**

> **作者:** Jiaxu Wan; Xu Wang; Mengwei Xie; Xinyuan Chang; Xinran Liu; Zheng Pan; Mu Xu; Hong Zhang; Ding Yuan; Yifan Yang
>
> **备注:** 35 pages, 17 figures, 23 tables
>
> **摘要:** Lane-level navigation is critical for geographic information systems and navigation-based tasks, offering finer-grained guidance than road-level navigation by standard definition (SD) maps. However, it currently relies on expansive global HD maps that cannot adapt to dynamic road conditions. Recently, online perception (OP) maps have become research hotspots, providing real-time geometry as an alternative, but lack the global topology needed for navigation. To address these issues, Online Navigation Refinement (ONR), a new mission is introduced that refines SD-map-based road-level routes into accurate lane-level navigation by associating SD maps with OP maps. The map-to-map association to handle many-to-one lane-to-road mappings under two key challenges: (1) no public dataset provides lane-to-road correspondences; (2) severe misalignment from spatial fluctuations, semantic disparities, and OP map noise invalidates traditional map matching. For these challenges, We contribute: (1) Online map association dataset (OMA), the first ONR benchmark with 30K scenarios and 2.6M annotated lane vectors; (2) MAT, a transformer with path-aware attention to aligns topology despite spatial fluctuations and semantic disparities and spatial attention for integrates noisy OP features via global context; and (3) NR P-R, a metric evaluating geometric and semantic alignment. Experiments show that MAT outperforms existing methods at 34 ms latency, enabling low-cost and up-to-date lane-level navigation.
>
---
#### [replaced 036] Prompt-Based Continual Compositional Zero-Shot Learning
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2512.09172v2](https://arxiv.org/pdf/2512.09172v2)**

> **作者:** Sauda Maryam; Sara Nadeem; Faisal Qureshi; Mohsen Ali
>
> **摘要:** We tackle continual adaptation of vision-language models to new attributes, objects, and their compositions in Compositional Zero-Shot Learning (CZSL), while preventing forgetting of prior knowledge. Unlike classical continual learning where classes are disjoint, CCZSL is more complex as attributes and objects may reoccur across sessions while compositions remain unique. Built on a frozen VLM backbone, we propose the first Prompt-based Continual Compositional Zero-Shot Learning (PromptCCZSL) framework that retains prior knowledge through recency-weighted multi-teacher distillation. It employs session-aware compositional prompts to fuse multimodal features for new compositions, while attribute and object prompts are learned through session-agnostic fusion to maintain global semantic consistency, which is further stabilized by a Cosine Anchor Loss (CAL) to preserve prior knowledge. To enhance adaptation in the current session, an Orthogonal Projection Loss (OPL) ensures that new attribute and object embeddings remain distinct from previous ones, preventing overlap, while an Intra-Session Diversity Loss (IDL) promotes variation among current-session embeddings for richer, more discriminative representations. We also introduce a comprehensive protocol that jointly measures catastrophic forgetting and compositional generalization. Extensive experiments on UT-Zappos and C-GQA benchmarks demonstrate that PromptCCZSL achieves substantial improvements over prior VLM-based and non-VLM baselines, setting a new benchmark for CCZSL in closed-world settings.
>
---
#### [replaced 037] 3D Software Synthesis Guided by Constraint-Expressive Intermediate Representation
- **分类: cs.CV; cs.AI; cs.MM; cs.SE**

- **链接: [https://arxiv.org/pdf/2507.18625v2](https://arxiv.org/pdf/2507.18625v2)**

> **作者:** Shuqing Li; Anson Y. Lam; Yun Peng; Wenxuan Wang; Michael R. Lyu
>
> **备注:** Accepted by the IEEE/ACM International Conference on Software Engineering (ICSE) 2026, Rio de Janeiro, Brazil
>
> **摘要:** Graphical user interface (UI) software has undergone a fundamental transformation from traditional two-dimensional (2D) desktop/web/mobile interfaces to spatial three-dimensional (3D) environments. While existing work has made remarkable success in automated 2D software generation, such as HTML/CSS and mobile app interface code synthesis, the generation of 3D software still remains under-explored. Current methods for 3D software generation usually generate the 3D environments as a whole and cannot modify or control specific elements in the software. Furthermore, these methods struggle to handle the complex spatial and semantic constraints inherent in the real world. To address the challenges, we present Scenethesis, a novel requirement-sensitive 3D software synthesis approach that maintains formal traceability between user specifications and generated 3D software. Scenethesis is built upon ScenethesisLang, a domain-specific language that serves as a granular constraint-aware intermediate representation (IR) to bridge natural language requirements and executable 3D software. It serves both as a comprehensive scene description language enabling fine-grained modification of 3D software elements and as a formal constraint-expressive specification language capable of expressing complex spatial constraints. By decomposing 3D software synthesis into stages operating on ScenethesisLang, Scenethesis enables independent verification, targeted modification, and systematic constraint satisfaction. Our evaluation demonstrates that Scenethesis accurately captures over 80% of user requirements and satisfies more than 90% of hard constraints while handling over 100 constraints simultaneously. Furthermore, Scenethesis achieves a 42.8% improvement in BLIP-2 visual evaluation scores compared to the state-of-the-art method.
>
---
#### [replaced 038] The Devil is in Attention Sharing: Improving Complex Non-rigid Image Editing Faithfulness via Attention Synergy
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.14423v2](https://arxiv.org/pdf/2512.14423v2)**

> **作者:** Zhuo Chen; Fanyue Wei; Runze Xu; Jingjing Li; Lixin Duan; Angela Yao; Wen Li
>
> **备注:** Project page:https://synps26.github.io/
>
> **摘要:** Training-free image editing with large diffusion models has become practical, yet faithfully performing complex non-rigid edits (e.g., pose or shape changes) remains highly challenging. We identify a key underlying cause: attention collapse in existing attention sharing mechanisms, where either positional embeddings or semantic features dominate visual content retrieval, leading to over-editing or under-editing. To address this issue, we introduce SynPS, a method that Synergistically leverages Positional embeddings and Semantic information for faithful non-rigid image editing. We first propose an editing measurement that quantifies the required editing magnitude at each denoising step. Based on this measurement, we design an attention synergy pipeline that dynamically modulates the influence of positional embeddings, enabling SynPS to balance semantic modifications and fidelity preservation. By adaptively integrating positional and semantic cues, SynPS effectively avoids both over- and under-editing. Extensive experiments on public and newly curated benchmarks demonstrate the superior performance and faithfulness of our approach.
>
---
#### [replaced 039] MovSemCL: Movement-Semantics Contrastive Learning for Trajectory Similarity (Extension)
- **分类: cs.CV; cs.AI; cs.DB**

- **链接: [https://arxiv.org/pdf/2511.12061v2](https://arxiv.org/pdf/2511.12061v2)**

> **作者:** Zhichen Lai; Hua Lu; Huan Li; Jialiang Li; Christian S. Jensen
>
> **备注:** 8 pages, 6 figures; accepted by AAAI 2026 as an Oral paper
>
> **摘要:** Trajectory similarity computation is fundamental functionality that is used for, e.g., clustering, prediction, and anomaly detection. However, existing learning-based methods exhibit three key limitations: (1) insufficient modeling of trajectory semantics and hierarchy, lacking both movement dynamics extraction and multi-scale structural representation; (2) high computational costs due to point-wise encoding; and (3) use of physically implausible augmentations that distort trajectory semantics. To address these issues, we propose MovSemCL, a movement-semantics contrastive learning framework for trajectory similarity computation. MovSemCL first transforms raw GPS trajectories into movement-semantics features and then segments them into patches. Next, MovSemCL employs intra- and inter-patch attentions to encode local as well as global trajectory patterns, enabling efficient hierarchical representation and reducing computational costs. Moreover, MovSemCL includes a curvature-guided augmentation strategy that preserves informative segments (e.g., turns and intersections) and masks redundant ones, generating physically plausible augmented views. Experiments on real-world datasets show that MovSemCL is capable of outperforming state-of-the-art methods, achieving mean ranks close to the ideal value of 1 at similarity search tasks and improvements by up to 20.3% at heuristic approximation, while reducing inference latency by up to 43.4%.
>
---
#### [replaced 040] Control-Augmented Autoregressive Diffusion for Data Assimilation
- **分类: cs.LG; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2510.06637v2](https://arxiv.org/pdf/2510.06637v2)**

> **作者:** Prakhar Srivastava; Farrin Marouf Sofian; Francesco Immorlano; Kushagra Pandey; Stephan Mandt
>
> **摘要:** Despite recent advances in test-time scaling and finetuning of diffusion models, guidance in Auto-Regressive Diffusion Models (ARDMs) remains underexplored. We introduce an amortized framework that augments a pretrained ARDM with a lightweight controller network, trained offline by previewing future rollouts to output stepwise controls that anticipate upcoming observations under a terminal-cost objective. Our approach is motivated by viewing guided generation as an entropy-regularized stochastic optimal control problem over ARDM trajectories: we learn a reusable policy that injects small control corrections inside each denoising sub-step while remaining anchored to the pretrained dynamics. We evaluate this framework in the context of data assimilation (DA) for chaotic spatiotemporal partial differential equations (PDEs), where existing methods can be computationally prohibitive and prone to forecast drift under sparse observations. At inference, DA reduces to a single causal forward rollout with on-the-fly corrections, requiring neither adjoint computations nor gradient-based optimization, and yields an order-of-magnitude speedup over strong diffusion-based DA baselines. Across two canonical PDEs and six observation regimes, our method consistently improves stability, accuracy, and physics-aware fidelity over state-of-the-art baselines. We will release code and checkpoints publicly.
>
---
#### [replaced 041] DM3D: Deformable Mamba via Offset-Guided Gaussian Sequencing for Point Cloud Understanding
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.03424v2](https://arxiv.org/pdf/2512.03424v2)**

> **作者:** Bin Liu; Chunyang Wang; Xuelian Liu
>
> **摘要:** State Space Models (SSMs) demonstrate significant potential for long-sequence modeling, but their reliance on input order conflicts with the irregular nature of point clouds. Existing approaches often rely on predefined serialization strategies, which cannot adjust based on diverse geometric structures. To overcome this limitation, we propose \textbf{DM3D}, a deformable Mamba architecture for point cloud understanding. Specifically, DM3D introduces an offset-guided Gaussian sequencing mechanism that unifies local resampling and global reordering within a deformable scan. The Gaussian-based KNN Resampling (GKR) enhances structural awareness by adaptively reorganizing neighboring points, while the Gaussian-based Differentiable Reordering (GDR) enables end-to-end optimization of serialization order. Furthermore, a Tri-Path Frequency Fusion module enhances feature complementarity and reduces aliasing. Together, these components enable structure-adaptive serialization of point clouds. Extensive experiments on benchmark datasets show that DM3D achieves state-of-the-art performance in classification, few-shot learning, and part segmentation, demonstrating that adaptive serialization effectively unlocks the potential of SSMs for point cloud understanding. The code will be released at https://github.com/L1277471578/DM3D.
>
---
#### [replaced 042] MS-Temba: Multi-Scale Temporal Mamba for Understanding Long Untrimmed Videos
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2501.06138v3](https://arxiv.org/pdf/2501.06138v3)**

> **作者:** Arkaprava Sinha; Monish Soundar Raj; Pu Wang; Ahmed Helmy; Hieu Le; Srijan Das
>
> **摘要:** Temporal Action Detection (TAD) in untrimmed videos poses significant challenges, particularly for Activities of Daily Living (ADL) requiring models to (1) process long-duration videos, (2) capture temporal variations in actions, and (3) simultaneously detect dense overlapping actions. Existing CNN and Transformer-based approaches, struggle to jointly capture fine-grained detail and long-range structure at scale. State-space Model (SSM) based Mamba offers powerful long-range modeling, but naive application to TAD collapses fine-grained temporal structure and fails to account for the challenges inherent to TAD. To this end, we propose Multi-Scale Temporal Mamba (MS-Temba), which extends Mamba to TAD with newly introduced dilated SSMs. Each Temba block, comprising dilated SSMs coupled with our proposed additional losses, enables the learning of discriminative representations across temporal scales. A lightweight Multi-scale Mamba Fuser then unifies these multi-scale features via SSM-based aggregation, yielding precise action-boundary localization. With only 17M parameters, MS-Temba achieves state-of-the-art performance on densely labeled ADL benchmarks TSU & Charades, and further generalizes to long-form video summarization, setting new state-of-the-art results on TVSum & SumMe.
>
---
#### [replaced 043] MUSE: Multi-Scale Dense Self-Distillation for Nucleus Detection and Classification
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.05170v2](https://arxiv.org/pdf/2511.05170v2)**

> **作者:** Zijiang Yang; Hanqing Chao; Bokai Zhao; Yelin Yang; Yunshuo Zhang; Dongmei Fu; Junping Zhang; Le Lu; Ke Yan; Dakai Jin; Minfeng Xu; Yun Bian; Hui Jiang
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** Nucleus detection and classification (NDC) in histopathology analysis is a fundamental task that underpins a wide range of high-level pathology applications. However, existing methods heavily rely on labor-intensive nucleus-level annotations and struggle to fully exploit large-scale unlabeled data for learning discriminative nucleus representations. In this work, we propose MUSE (MUlti-scale denSE self-distillation), a novel self-supervised learning method tailored for NDC. At its core is NuLo (Nucleus-based Local self-distillation), a coordinate-guided mechanism that enables flexible local self-distillation based on predicted nucleus positions. By removing the need for strict spatial alignment between augmented views, NuLo allows critical cross-scale alignment, thus unlocking the capacity of models for fine-grained nucleus-level representation. To support MUSE, we design a simple yet effective encoder-decoder architecture and a large field-of-view semi-supervised fine-tuning strategy that together maximize the value of unlabeled pathology images. Extensive experiments on three widely used benchmarks demonstrate that MUSE effectively addresses the core challenges of histopathological NDC. The resulting models not only surpass state-of-the-art supervised baselines but also outperform generic pathology foundation models.
>
---
#### [replaced 044] MAGIC: Few-Shot Mask-Guided Anomaly Inpainting with Prompt Perturbation, Spatially Adaptive Guidance, and Context Awareness
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.02314v3](https://arxiv.org/pdf/2507.02314v3)**

> **作者:** JaeHyuck Choi; MinJun Kim; JeHyeong Hong
>
> **备注:** 46 pages, 47 figures. Code: https://github.com/Jaeihk/MAGIC-Anomaly-generation
>
> **摘要:** Few-shot anomaly generation is a key challenge in industrial quality control. Although diffusion models are promising, existing methods struggle: global prompt-guided approaches corrupt normal regions, and existing inpainting-based methods often lack the in-distribution diversity essential for robust downstream models. We propose MAGIC, a fine-tuned inpainting framework that generates high-fidelity anomalies that strictly adhere to the mask while maximizing this diversity. MAGIC introduces three complementary components: (i) Gaussian prompt perturbation, which prevents model overfitting in the few-shot setting by learning and sampling from a smooth manifold of realistic anomalies, (ii) spatially adaptive guidance that applies distinct guidance strengths to the anomaly and background regions, and (iii) context-aware mask alignment to relocate masks for plausible placement within the host object. Under consistent identical evaluation protocol, MAGIC outperforms state-of-the-art methods on diverse anomaly datasets in downstream tasks.
>
---
#### [replaced 045] TPG-INR: Target Prior-Guided Implicit 3D CT Reconstruction for Enhanced Sparse-view Imaging
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.18806v2](https://arxiv.org/pdf/2511.18806v2)**

> **作者:** Qinglei Cao; Ziyao Tang; Xiaoqin Tang
>
> **备注:** We are withdrawing to restructure and refine the research plan to enhance its systematic rigor, completeness, and overall depth
>
> **摘要:** X-ray imaging, based on penetration, enables detailed visualization of internal structures. Building on this capability, existing implicit 3D reconstruction methods have adapted the NeRF model and its variants for internal CT reconstruction. However, these approaches often neglect the significance of objects' anatomical priors for implicit learning, limiting both reconstruction precision and learning efficiency, particularly in ultra-sparse view scenarios. To address these challenges, we propose a novel 3D CT reconstruction framework that employs a 'target prior' derived from the object's projection data to enhance implicit learning. Our approach integrates positional and structural encoding to facilitate voxel-wise implicit reconstruction, utilizing the target prior to guide voxel sampling and enrich structural encoding. This dual strategy significantly boosts both learning efficiency and reconstruction quality. Additionally, we introduce a CUDA-based algorithm for rapid estimation of high-quality 3D target priors from sparse-view projections. Experiments utilizing projection data from a complex abdominal dataset demonstrate that the proposed model substantially enhances learning efficiency, outperforming the current leading model, NAF, by a factor of ten. In terms of reconstruction quality, it also exceeds the most accurate model, NeRP, achieving PSNR improvements of 3.57 dB, 5.42 dB, and 5.70 dB with 10, 20, and 30 projections, respectively. The code is available at https://github.com/qlcao171/TPG-INR.
>
---
#### [replaced 046] Deep Learning for Retinal Degeneration Assessment: A Comprehensive Analysis of the MARIO Challenge
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.02976v3](https://arxiv.org/pdf/2506.02976v3)**

> **作者:** Rachid Zeghlache; Ikram Brahim; Pierre-Henri Conze; Mathieu Lamard; Mohammed El Amine Lazouni; Zineb Aziza Elaouaber; Leila Ryma Lazouni; Christopher Nielsen; Ahmad O. Ahsan; Matthias Wilms; Nils D. Forkert; Lovre Antonio Budimir; Ivana Matovinović; Donik Vršnak; Sven Lončarić; Philippe Zhang; Weili Jiang; Yihao Li; Yiding Hao; Markus Frohmann; Patrick Binder; Marcel Huber; Taha Emre; Teresa Finisterra Araújo; Marzieh Oghbaie; Hrvoje Bogunović; Amerens A. Bekkers; Nina M. van Liebergen; Hugo J. Kuijf; Abdul Qayyum; Moona Mazher; Steven A. Niederer; Alberto J. Beltrán-Carrero; Juan J. Gómez-Valverde; Javier Torresano-Rodríquez; Álvaro Caballero-Sastre; María J. Ledesma Carbayo; Yosuke Yamagishi; Yi Ding; Robin Peretzke; Alexandra Ertl; Maximilian Fischer; Jessica Kächele; Sofiane Zehar; Karim Boukli Hacene; Thomas Monfort; Béatrice Cochener; Mostafa El Habib Daho; Anas-Alexis Benyoussef; Gwenolé Quellec
>
> **备注:** MARIO-MICCAI-CHALLENGE 2024
>
> **摘要:** The MARIO challenge, held at MICCAI 2024, focused on advancing the automated detection and monitoring of age-related macular degeneration (AMD) through the analysis of optical coherence tomography (OCT) images. Designed to evaluate algorithmic performance in detecting neovascular activity changes within AMD, the challenge incorporated unique multi-modal datasets. The primary dataset, sourced from Brest, France, was used by participating teams to train and test their models. The final ranking was determined based on performance on this dataset. An auxiliary dataset from Algeria was used post-challenge to evaluate population and device shifts from submitted solutions. Two tasks were involved in the MARIO challenge. The first one was the classification of evolution between two consecutive 2D OCT B-scans. The second one was the prediction of future AMD evolution over three months for patients undergoing anti-vascular endothelial growth factor (VEGF) therapy. Thirty-five teams participated, with the top 12 finalists presenting their methods. This paper outlines the challenge's structure, tasks, data characteristics, and winning methodologies, setting a benchmark for AMD monitoring using OCT, infrared imaging, and clinical data (such as the number of visits, age, gender, etc.). The results of this challenge indicate that artificial intelligence (AI) performs as well as a physician in measuring AMD progression (Task 1) but is not yet able of predicting future evolution (Task 2).
>
---
#### [replaced 047] Cascaded Dual Vision Transformer for Accurate Facial Landmark Detection
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.07167v2](https://arxiv.org/pdf/2411.07167v2)**

> **作者:** Ziqiang Dang; Jianfang Li; Lin Liu
>
> **备注:** Accepted by WACV 2025. The code can be found at https://github.com/Human3DAIGC/AccurateFacialLandmarkDetection . Supplementary material is included at the end of the main paper (3 pages, 5 figures, 2 tables)
>
> **摘要:** Facial landmark detection is a fundamental problem in computer vision for many downstream applications. This paper introduces a new facial landmark detector based on vision transformers, which consists of two unique designs: Dual Vision Transformer (D-ViT) and Long Skip Connections (LSC). Based on the observation that the channel dimension of feature maps essentially represents the linear bases of the heatmap space, we propose learning the interconnections between these linear bases to model the inherent geometric relations among landmarks via Channel-split ViT. We integrate such channel-split ViT into the standard vision transformer (i.e., spatial-split ViT), forming our Dual Vision Transformer to constitute the prediction blocks. We also suggest using long skip connections to deliver low-level image features to all prediction blocks, thereby preventing useful information from being discarded by intermediate supervision. Extensive experiments are conducted to evaluate the performance of our proposal on the widely used benchmarks, i.e., WFLW, COFW, and 300W, demonstrating that our model outperforms the previous SOTAs across all three benchmarks.
>
---
#### [replaced 048] SparseWorld-TC: Trajectory-Conditioned Sparse Occupancy World Model
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.22039v2](https://arxiv.org/pdf/2511.22039v2)**

> **作者:** Jiayuan Du; Yiming Zhao; Zhenglong Guo; Yong Pan; Wenbo Hou; Zhihui Hao; Kun Zhan; Qijun Chen
>
> **摘要:** This paper introduces a novel architecture for trajectory-conditioned forecasting of future 3D scene occupancy. In contrast to methods that rely on variational autoencoders (VAEs) to generate discrete occupancy tokens, which inherently limit representational capacity, our approach predicts multi-frame future occupancy in an end-to-end manner directly from raw image features. Inspired by the success of attention-based transformer architectures in foundational vision and language models such as GPT and VGGT, we employ a sparse occupancy representation that bypasses the intermediate bird's eye view (BEV) projection and its explicit geometric priors. This design allows the transformer to capture spatiotemporal dependencies more effectively. By avoiding both the finite-capacity constraint of discrete tokenization and the structural limitations of BEV representations, our method achieves state-of-the-art performance on the nuScenes benchmark for 1-3 second occupancy forecasting, outperforming existing approaches by a significant margin. Furthermore, it demonstrates robust scene dynamics understanding, consistently delivering high accuracy under arbitrary future trajectory conditioning.
>
---
#### [replaced 049] History-Augmented Contrastive Learning With Soft Mixture of Experts for Blind Super-Resolution of Planetary Remote Sensing Images
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2511.20045v2](https://arxiv.org/pdf/2511.20045v2)**

> **作者:** Hui-Jia Zhao; Jie Lu; Yunqing Jiang; Xiao-Ping Lu; Kaichang Di
>
> **备注:** 12pages
>
> **摘要:** Blind Super-Resolution (BSR) in planetary remote sensing constitutes a highly ill-posed inverse problem, characterized by unknown degradation patterns and a complete absence of ground-truth supervision. Existing unsupervised approaches often struggle with optimization instability and distribution shifts, relying on greedy strategies or generic priors that fail to preserve distinct morphological semantics. To address these challenges, we propose History-Augmented Contrastive Mixture of Experts (HAC-MoE), a novel unsupervised framework that decouples kernel estimation from image reconstruction without external kernel priors. The framework is founded on three key innovations: (1) A Contrastive Kernel Sampling mechanism that mitigates the distribution bias inherent in random Gaussian sampling, ensuring the generation of plausible kernel priors via similarity constraints; (2) A History-Augmented Contrastive Learning strategy that leverages historical model states as negative self-priors. We provide a theoretical analysis demonstrating that this mechanism induces strong convexity in the feature space, thereby stabilizing the unsupervised optimization trajectory and preventing overfitting; and (3) A Morphology-Aware Soft Mixture-of-Experts (MA-MoE) estimator that dynamically modulates spectral-spatial features to adaptively reconstruct diverse planetary topographies. To facilitate rigorous evaluation, we introduce Ceres-50, a benchmark dataset encapsulating diverse geological features under realistic degradation simulations. Extensive experiments demonstrate that HAC-MoE achieves state-of-the-art performance in reconstruction quality and kernel estimation accuracy, offering a solution for scientific observation in data-sparse extraterrestrial environments.
>
---
#### [replaced 050] REAL: Representation Enhanced Analytic Learning for Exemplar-free Class-incremental Learning
- **分类: cs.LG; cs.CV**

- **链接: [https://arxiv.org/pdf/2403.13522v3](https://arxiv.org/pdf/2403.13522v3)**

> **作者:** Run He; Di Fang; Yizhu Chen; Kai Tong; Cen Chen; Yi Wang; Lap-pui Chau; Huiping Zhuang
>
> **备注:** 13 pages, 7 figures. This paper is published in Knowledge-based System
>
> **摘要:** Exemplar-free class-incremental learning (EFCIL) aims to mitigate catastrophic forgetting in class-incremental learning (CIL) without available historical training samples as exemplars. Compared with its exemplar-based CIL counterpart that stores exemplars, EFCIL suffers more from forgetting issues. Recently, a new EFCIL branch named Analytic Continual Learning (ACL) introduces a gradient-free paradigm via Recursive Least-Square, achieving a forgetting-resistant classifier training with a frozen backbone during CIL. However, existing ACL suffers from ineffective representations and insufficient utilization of backbone knowledge. In this paper, we propose a representation-enhanced analytic learning (REAL) to address these problems. To enhance the representation, REAL constructs a dual-stream base pretraining followed by representation enhancing distillation process. The dual-stream base pretraining combines self-supervised contrastive learning for general features and supervised learning for class-specific knowledge, followed by the representation enhancing distillation to merge both streams, enhancing representations for subsequent CIL paradigm. To utilize more knowledge from the backbone, REAL presents a feature fusion buffer to multi-layer backbone features, providing informative features for the subsequent classifier training. Our method can be incorporated into existing ACL techniques and provides more competitive performance. Empirical results demonstrate that, REAL achieves state-of-the-art performance on CIFAR-100, ImageNet-100 and ImageNet-1k benchmarks, outperforming exemplar-free methods and rivaling exemplar-based approaches.
>
---
#### [replaced 051] Benchmarking Gaslighting Negation Attacks Against Reasoning Models
- **分类: cs.CV; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.09677v2](https://arxiv.org/pdf/2506.09677v2)**

> **作者:** Bin Zhu; Hailong Yin; Jingjing Chen; Yu-Gang Jiang
>
> **摘要:** Recent advances in reasoning-centric models promise improved robustness through mechanisms such as chain-of-thought prompting and test-time scaling. However, their ability to withstand gaslighting negation attacks-adversarial prompts that confidently deny correct answers-remains underexplored. In this paper, we conduct a systematic evaluation of three state-of-the-art reasoning models, i.e., OpenAI's o4-mini, Claude-3.7-Sonnet and Gemini-2.5-Flash, across three multimodal benchmarks: MMMU, MathVista, and CharXiv. Our evaluation reveals significant accuracy drops (25-29% on average) following gaslighting negation attacks, indicating that even top-tier reasoning models struggle to preserve correct answers under manipulative user feedback. Built upon the insights of the evaluation and to further probe this vulnerability, we introduce GaslightingBench-R, a new diagnostic benchmark specifically designed to evaluate reasoning models' susceptibility to defend their belief under gaslighting negation attacks. Constructed by filtering and curating 1,025 challenging samples from the existing benchmarks, GaslightingBench-R induces even more dramatic failures, with accuracy drops exceeding 53% on average. Our findings highlight a fundamental gap between step-by-step reasoning and resistance to adversarial manipulation, calling for new robustness strategies that safeguard reasoning models against gaslighting negation attacks.
>
---
#### [replaced 052] From Pretraining to Privacy: Federated Ultrasound Foundation Model with Self-Supervised Learning
- **分类: eess.IV; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2411.16380v2](https://arxiv.org/pdf/2411.16380v2)**

> **作者:** Yuncheng Jiang; Chun-Mei Feng; Jinke Ren; Jun Wei; Zixun Zhang; Yiwen Hu; Yunbi Liu; Rui Sun; Xuemei Tang; Juan Du; Xiang Wan; Yong Xu; Bo Du; Xin Gao; Guangyu Wang; Shaohua Zhou; Shuguang Cui; Zhen Li
>
> **备注:** npj digital medicine(2025)
>
> **摘要:** Ultrasound imaging is widely used in clinical diagnosis due to its non-invasive nature and real-time capabilities. However, traditional ultrasound diagnostics relies heavily on physician expertise and is often hampered by suboptimal image quality, leading to potential diagnostic errors. While artificial intelligence (AI) offers a promising solution to enhance clinical diagnosis by detecting abnormalities across various imaging modalities, existing AI methods for ultrasound face two major challenges. First, they typically require vast amounts of labeled medical data, raising serious concerns regarding patient privacy. Second, most models are designed for specific tasks, which restricts their broader clinical utility. To overcome these challenges, we present UltraFedFM, an innovative privacy-preserving ultrasound foundation model. UltraFedFM is collaboratively pre-trained using federated learning across 16 distributed medical institutions in 9 countries, leveraging a dataset of over 1 million ultrasound images covering 19 organs and 10 ultrasound modalities. This extensive and diverse data, combined with a secure training framework, enables UltraFedFM to exhibit strong generalization and diagnostic capabilities. It achieves an average area under the receiver operating characteristic curve (AUROC) of 0.927 for disease diagnosis and a dice similarity coefficient (DSC) of 0.878 for lesion segmentation. Notably, UltraFedFM surpasses the diagnostic accuracy of mid-level ultrasonographers (4-8 years of experience) and matches the performance of expert-level sonographers (10+ years of experience) in the joint diagnosis of 8 common systemic diseases.c These findings indicate that UltraFedFM can significantly enhance clinical diagnostics while safeguarding patient privacy, marking a significant advancement in AI-driven ultrasound imaging for future clinical applications.
>
---
#### [replaced 053] History-Enhanced Two-Stage Transformer for Aerial Vision-and-Language Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文面向空中视觉-语言导航（AVLN）任务，解决无人机在大尺度城市中依语言指令准确定位目标时，难以兼顾全局环境推理与局部场景理解的问题。提出历史增强的两阶段Transformer（HETT），通过粗粒度定位与细粒度动作优化的级联框架，并引入历史网格地图增强空间记忆，显著提升导航性能。**

- **链接: [https://arxiv.org/pdf/2512.14222v2](https://arxiv.org/pdf/2512.14222v2)**

> **作者:** Xichen Ding; Jianzhe Gao; Cong Pan; Wenguan Wang; Jie Qin
>
> **摘要:** Aerial Vision-and-Language Navigation (AVLN) requires Unmanned Aerial Vehicle (UAV) agents to localize targets in large-scale urban environments based on linguistic instructions. While successful navigation demands both global environmental reasoning and local scene comprehension, existing UAV agents typically adopt mono-granularity frameworks that struggle to balance these two aspects. To address this limitation, this work proposes a History-Enhanced Two-Stage Transformer (HETT) framework, which integrates the two aspects through a coarse-to-fine navigation pipeline. Specifically, HETT first predicts coarse-grained target positions by fusing spatial landmarks and historical context, then refines actions via fine-grained visual analysis. In addition, a historical grid map is designed to dynamically aggregate visual features into a structured spatial memory, enhancing comprehensive scene awareness. Additionally, the CityNav dataset annotations are manually refined to enhance data quality. Experiments on the refined CityNav dataset show that HETT delivers significant performance gains, while extensive ablation studies further verify the effectiveness of each component.
>
---
#### [replaced 054] Do MLLMs Exhibit Human-like Perceptual Behaviors? HVSBench: A Benchmark for MLLM Alignment with Human Perceptual Behavior
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2412.09603v3](https://arxiv.org/pdf/2412.09603v3)**

> **作者:** Jiaying Lin; Shuquan Ye; Dan Xu; Wanli Ouyang; Rynson W. H. Lau
>
> **备注:** Project page: https://jiaying.link/HVSBench/
>
> **摘要:** While Multimodal Large Language Models (MLLMs) excel at many vision tasks, it is unknown if they exhibit human-like perceptual behaviors. To evaluate this, we introduce HVSBench, the first large-scale benchmark with over 85,000 samples designed to test MLLM alignment with the human visual system (HVS). The benchmark covers 13 categories across 5 key fields: Prominence, Subitizing, Prioritizing, Free-Viewing, and Searching. Our comprehensive evaluation reveals a significant perceptual gap: even state-of-the-art MLLMs achieve only moderate results. In contrast, human participants demonstrate strong performance, significantly outperforming all models. This underscores the high quality of HVSBench and the need for more human-aligned AI. We believe our benchmark will be a critical tool for developing the next generation of explainable MLLMs.
>
---
#### [replaced 055] ASSR-NeRF: Arbitrary-Scale Super-Resolution on Voxel Grid for High-Quality Radiance Fields Reconstruction
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2406.20066v2](https://arxiv.org/pdf/2406.20066v2)**

> **作者:** Ding-Jiun Huang; Zi-Ting Chou; Yu-Chiang Frank Wang; Cheng Sun
>
> **摘要:** NeRF-based methods reconstruct 3D scenes by building a radiance field with implicit or explicit representations. While NeRF-based methods can perform novel view synthesis (NVS) at arbitrary scale, the performance in high-resolution novel view synthesis (HRNVS) with low-resolution (LR) optimization often results in oversmoothing. On the other hand, single-image super-resolution (SR) aims to enhance LR images to HR counterparts but lacks multi-view consistency. To address these challenges, we propose Arbitrary-Scale Super-Resolution NeRF (ASSR-NeRF), a novel framework for super-resolution novel view synthesis (SRNVS). We propose an attention-based VoxelGridSR model to directly perform 3D super-resolution (SR) on the optimized volume. Our model is trained on diverse scenes to ensure generalizability. For unseen scenes trained with LR views, we then can directly apply our VoxelGridSR to further refine the volume and achieve multi-view consistent SR. We demonstrate quantitative and qualitatively that the proposed method achieves significant performance in SRNVS.
>
---
#### [replaced 056] LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2510.15022v2](https://arxiv.org/pdf/2510.15022v2)**

> **作者:** Mert Sonmezer; Matthew Zheng; Pinar Yanardag
>
> **摘要:** Low-rank Adaptation (LoRA) models have revolutionized the personalization of pre-trained diffusion models by enabling fine-tuning through low-rank, factorized weight matrices specifically optimized for attention layers. These models facilitate the generation of highly customized content across a variety of objects, individuals, and artistic styles without the need for extensive retraining. Despite the availability of over 100K LoRA adapters on platforms like Civit.ai, users often face challenges in navigating, selecting, and effectively utilizing the most suitable adapters due to their sheer volume, diversity, and lack of structured organization. This paper addresses the problem of selecting the most relevant and diverse LoRA models from this vast database by framing the task as a combinatorial optimization problem and proposing a novel submodular framework. Our quantitative and qualitative experiments demonstrate that our method generates diverse outputs across a wide range of domains.
>
---
#### [replaced 057] RecTok: Reconstruction Distillation along Rectified Flow
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2512.13421v2](https://arxiv.org/pdf/2512.13421v2)**

> **作者:** Qingyu Shi; Size Wu; Jinbin Bai; Kaidong Yu; Yujing Wang; Yunhai Tong; Xiangtai Li; Xuelong Li
>
> **摘要:** Visual tokenizers play a crucial role in diffusion models. The dimensionality of latent space governs both reconstruction fidelity and the semantic expressiveness of the latent feature. However, a fundamental trade-off is inherent between dimensionality and generation quality, constraining existing methods to low-dimensional latent spaces. Although recent works have leveraged vision foundation models to enrich the semantics of visual tokenizers and accelerate convergence, high-dimensional tokenizers still underperform their low-dimensional counterparts. In this work, we propose RecTok, which overcomes the limitations of high-dimensional visual tokenizers through two key innovations: flow semantic distillation and reconstruction--alignment distillation. Our key insight is to make the forward flow in flow matching semantically rich, which serves as the training space of diffusion transformers, rather than focusing on the latent space as in previous works. Specifically, our method distills the semantic information in VFMs into the forward flow trajectories in flow matching. And we further enhance the semantics by introducing a masked feature reconstruction loss. Our RecTok achieves superior image reconstruction, generation quality, and discriminative performance. It achieves state-of-the-art results on the gFID-50K under both with and without classifier-free guidance settings, while maintaining a semantically rich latent space structure. Furthermore, as the latent dimensionality increases, we observe consistent improvements. Code and model are available at https://shi-qingyu.github.io/rectok.github.io.
>
---
#### [replaced 058] Toward Robust and Accurate Adversarial Camouflage Generation against Vehicle Detectors
- **分类: cs.CV**

- **链接: [https://arxiv.org/pdf/2411.10029v2](https://arxiv.org/pdf/2411.10029v2)**

> **作者:** Jiawei Zhou; Linye Lyu; Daojing He; Yu Li
>
> **备注:** 14 pages. arXiv admin note: substantial text overlap with arXiv:2402.15853
>
> **摘要:** Adversarial camouflage is a widely used physical attack against vehicle detectors for its superiority in multi-view attack performance. One promising approach involves using differentiable neural renderers to facilitate adversarial camouflage optimization through gradient back-propagation. However, existing methods often struggle to capture environmental characteristics during the rendering process or produce adversarial textures that can precisely map to the target vehicle. Moreover, these approaches neglect diverse weather conditions, reducing the efficacy of generated camouflage across varying weather scenarios. To tackle these challenges, we propose a robust and accurate camouflage generation method, namely RAUCA. The core of RAUCA is a novel neural rendering component, End-to-End Neural Renderer Plus (E2E-NRP), which can accurately optimize and project vehicle textures and render images with environmental characteristics such as lighting and weather. In addition, we integrate a multi-weather dataset for camouflage generation, leveraging the E2E-NRP to enhance the attack robustness. Experimental results on six popular object detectors show that RAUCA-final outperforms existing methods in both simulation and real-world settings.
>
---
