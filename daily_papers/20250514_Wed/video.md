# 计算机视觉 cs.CV

- **最新发布 98 篇**

- **更新 69 篇**

## 最新发布

#### [new 001] Advancing Food Nutrition Estimation via Visual-Ingredient Feature Fusion
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于营养估计任务，旨在解决因标注数据不足导致的精度受限问题。研究者构建了FastFood数据集（8.4万张带营养标注的快餐图像），并提出模型无关的视觉-成分特征融合方法VIF²，通过增强成分鲁棒性、多模态数据优化及特征融合提升预测精度，验证了成分信息对营养估算的关键作用。**

- **链接: [http://arxiv.org/pdf/2505.08747v1](http://arxiv.org/pdf/2505.08747v1)**

> **作者:** Huiyan Qi; Bin Zhu; Chong-Wah Ngo; Jingjing Chen; Ee-Peng Lim
>
> **备注:** Accepted for publication in ACM International Conference on Multimedia Retrieval 2025
>
> **摘要:** Nutrition estimation is an important component of promoting healthy eating and mitigating diet-related health risks. Despite advances in tasks such as food classification and ingredient recognition, progress in nutrition estimation is limited due to the lack of datasets with nutritional annotations. To address this issue, we introduce FastFood, a dataset with 84,446 images across 908 fast food categories, featuring ingredient and nutritional annotations. In addition, we propose a new model-agnostic Visual-Ingredient Feature Fusion (VIF$^2$) method to enhance nutrition estimation by integrating visual and ingredient features. Ingredient robustness is improved through synonym replacement and resampling strategies during training. The ingredient-aware visual feature fusion module combines ingredient features and visual representation to achieve accurate nutritional prediction. During testing, ingredient predictions are refined using large multimodal models by data augmentation and majority voting. Our experiments on both FastFood and Nutrition5k datasets validate the effectiveness of our proposed method built in different backbones (e.g., Resnet, InceptionV3 and ViT), which demonstrates the importance of ingredient information in nutrition estimation. https://huiyanqi.github.io/fastfood-nutrition-estimation/.
>
---
#### [new 002] A computer vision-based model for occupancy detection using low-resolution thermal images
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于计算机视觉的占用检测任务，旨在解决传统HVAC系统忽视实时占用及RGB监测隐私问题。研究者通过迁移学习微调YOLOv5模型，利用低分辨率热成像技术实现非侵入式人员检测，在保证高精度（关键指标接近1.000）的同时兼顾隐私保护和计算效率。**

- **链接: [http://arxiv.org/pdf/2505.08336v1](http://arxiv.org/pdf/2505.08336v1)**

> **作者:** Xue Cui; Vincent Gbouna Zakka; Minhyun Lee
>
> **摘要:** Occupancy plays an essential role in influencing the energy consumption and operation of heating, ventilation, and air conditioning (HVAC) systems. Traditional HVAC typically operate on fixed schedules without considering occupancy. Advanced occupant-centric control (OCC) adopted occupancy status in regulating HVAC operations. RGB images combined with computer vision (CV) techniques are widely used for occupancy detection, however, the detailed facial and body features they capture raise significant privacy concerns. Low-resolution thermal images offer a non-invasive solution that mitigates privacy issues. The study developed an occupancy detection model utilizing low-resolution thermal images and CV techniques, where transfer learning was applied to fine-tune the You Only Look Once version 5 (YOLOv5) model. The developed model ultimately achieved satisfactory performance, with precision, recall, mAP50, and mAP50 values approaching 1.000. The contributions of this model lie not only in mitigating privacy concerns but also in reducing computing resource demands.
>
---
#### [new 003] Calibration and Uncertainty for multiRater Volume Assessment in multiorgan Segmentation (CURVAS) challenge results
- **分类: cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决深度学习模型的可靠性问题，如标注变异性、校准不足及不确定性估计。研究者发起CURVAS挑战赛，利用多标注者共识/分歧数据评估模型性能，结合DSC、ECE等指标分析校准效果，发现校准良好的模型表现更优，并验证数据多样性与预训练对模型鲁棒性的提升。**

- **链接: [http://arxiv.org/pdf/2505.08685v1](http://arxiv.org/pdf/2505.08685v1)**

> **作者:** Meritxell Riera-Marin; Sikha O K; Julia Rodriguez-Comas; Matthias Stefan May; Zhaohong Pan; Xiang Zhou; Xiaokun Liang; Franciskus Xaverius Erick; Andrea Prenner; Cedric Hemon; Valentin Boussot; Jean-Louis Dillenseger; Jean-Claude Nunes; Abdul Qayyum; Moona Mazher; Steven A Niederer; Kaisar Kushibar; Carlos Martin-Isla; Petia Radeva; Karim Lekadir; Theodore Barfoot; Luis C. Garcia Peraza Herrera; Ben Glocker; Tom Vercauteren; Lucas Gago; Justin Englemann; Joy-Marie Kleiss; Anton Aubanell; Andreu Antolin; Javier Garcia-Lopez; Miguel A. Gonzalez Ballester; Adrian Galdran
>
> **备注:** This challenge was hosted in MICCAI 2024
>
> **摘要:** Deep learning (DL) has become the dominant approach for medical image segmentation, yet ensuring the reliability and clinical applicability of these models requires addressing key challenges such as annotation variability, calibration, and uncertainty estimation. This is why we created the Calibration and Uncertainty for multiRater Volume Assessment in multiorgan Segmentation (CURVAS), which highlights the critical role of multiple annotators in establishing a more comprehensive ground truth, emphasizing that segmentation is inherently subjective and that leveraging inter-annotator variability is essential for robust model evaluation. Seven teams participated in the challenge, submitting a variety of DL models evaluated using metrics such as Dice Similarity Coefficient (DSC), Expected Calibration Error (ECE), and Continuous Ranked Probability Score (CRPS). By incorporating consensus and dissensus ground truth, we assess how DL models handle uncertainty and whether their confidence estimates align with true segmentation performance. Our findings reinforce the importance of well-calibrated models, as better calibration is strongly correlated with the quality of the results. Furthermore, we demonstrate that segmentation models trained on diverse datasets and enriched with pre-trained knowledge exhibit greater robustness, particularly in cases deviating from standard anatomical structures. Notably, the best-performing models achieved high DSC and well-calibrated uncertainty estimates. This work underscores the need for multi-annotator ground truth, thorough calibration assessments, and uncertainty-aware evaluations to develop trustworthy and clinically reliable DL-based medical image segmentation models.
>
---
#### [new 004] Knowledge-Informed Deep Learning for Irrigation Type Mapping from Remote Sensing
- **分类: cs.CV**

- **简介: 该论文属于遥感图像分类任务，旨在解决复杂农业环境中灌溉类型识别困难的问题。提出基于Swin-Transformer的知识融合模型KIIM，整合作物概率编码、空间注意力、多模态交叉注意力及加权集成策略，通过迁移学习减少数据依赖，在灌溉分类精度和跨区域适应性上显著优于基线。**

- **链接: [http://arxiv.org/pdf/2505.08302v1](http://arxiv.org/pdf/2505.08302v1)**

> **作者:** Oishee Bintey Hoque; Nibir Chandra Mandal; Abhijin Adiga; Samarth Swarup; Sayjro Kossi Nouwakpo; Amanda Wilson; Madhav Marathe
>
> **备注:** Full version of the paper will be appearing at the Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI-25), Special Track on AI for Good
>
> **摘要:** Accurate mapping of irrigation methods is crucial for sustainable agricultural practices and food systems. However, existing models that rely solely on spectral features from satellite imagery are ineffective due to the complexity of agricultural landscapes and limited training data, making this a challenging problem. We present Knowledge-Informed Irrigation Mapping (KIIM), a novel Swin-Transformer based approach that uses (i) a specialized projection matrix to encode crop to irrigation probability, (ii) a spatial attention map to identify agricultural lands from non-agricultural lands, (iii) bi-directional cross-attention to focus complementary information from different modalities, and (iv) a weighted ensemble for combining predictions from images and crop information. Our experimentation on five states in the US shows up to 22.9\% (IoU) improvement over baseline with a 71.4% (IoU) improvement for hard-to-classify drip irrigation. In addition, we propose a two-phase transfer learning approach to enhance cross-state irrigation mapping, achieving a 51% IoU boost in a state with limited labeled data. The ability to achieve baseline performance with only 40% of the training data highlights its efficiency, reducing the dependency on extensive manual labeling efforts and making large-scale, automated irrigation mapping more feasible and cost-effective.
>
---
#### [new 005] A Survey of 3D Reconstruction with Event Cameras: From Event-based Geometry to Neural 3D Rendering
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于综述研究，系统梳理事件相机在3D重建中的技术发展。针对事件相机数据特性（异步高动态）与传统方法的适配问题，将现有方法按输入模态（立体/单目/多模态）和重建技术（几何/深度学习/神经渲染）分类，总结数据集并指出数据稀缺、动态场景处理等挑战，为领域提供技术框架和研究方向。**

- **链接: [http://arxiv.org/pdf/2505.08438v1](http://arxiv.org/pdf/2505.08438v1)**

> **作者:** Chuanzhi Xu; Haoxian Zhou; Langyi Chen; Haodong Chen; Ying Zhou; Vera Chung; Qiang Qu
>
> **备注:** 35 pages, 12 figures, 11 tables
>
> **摘要:** Event cameras have emerged as promising sensors for 3D reconstruction due to their ability to capture per-pixel brightness changes asynchronously. Unlike conventional frame-based cameras, they produce sparse and temporally rich data streams, which enable more accurate 3D reconstruction and open up the possibility of performing reconstruction in extreme environments such as high-speed motion, low light, or high dynamic range scenes. In this survey, we provide the first comprehensive review focused exclusively on 3D reconstruction using event cameras. The survey categorises existing works into three major types based on input modality - stereo, monocular, and multimodal systems, and further classifies them by reconstruction approach, including geometry-based, deep learning-based, and recent neural rendering techniques such as Neural Radiance Fields and 3D Gaussian Splatting. Methods with a similar research focus were organised chronologically into the most subdivided groups. We also summarise public datasets relevant to event-based 3D reconstruction. Finally, we highlight current research limitations in data availability, evaluation, representation, and dynamic scene handling, and outline promising future research directions. This survey aims to serve as a comprehensive reference and a roadmap for future developments in event-driven 3D reconstruction.
>
---
#### [new 006] Sleep Position Classification using Transfer Learning for Bed-based Pressure Sensors
- **分类: cs.CV**

- **简介: 该论文属于睡眠姿势分类任务，旨在解决临床环境下标记数据不足导致的深度学习模型训练难题。通过迁移学习（ViTMAE/ViTPose预训练模型），在低分辨率床基压力传感器数据上实现四类睡姿分类，性能优于传统特征工程方法和时序卷积网络，并验证了临床实用性。**

- **链接: [http://arxiv.org/pdf/2505.08111v1](http://arxiv.org/pdf/2505.08111v1)**

> **作者:** Olivier Papillon; Rafik Goubran; James Green; Julien Larivière-Chartier; Caitlin Higginson; Frank Knoefel; Rébecca Robillard
>
> **备注:** Conference publication submitted to IEEE I2MTC 2025
>
> **摘要:** Bed-based pressure-sensitive mats (PSMs) offer a non-intrusive way of monitoring patients during sleep. We focus on four-way sleep position classification using data collected from a PSM placed under a mattress in a sleep clinic. Sleep positions can affect sleep quality and the prevalence of sleep disorders, such as apnea. Measurements were performed on patients with suspected sleep disorders referred for assessments at a sleep clinic. Training deep learning models can be challenging in clinical settings due to the need for large amounts of labeled data. To overcome the shortage of labeled training data, we utilize transfer learning to adapt pre-trained deep learning models to accurately estimate sleep positions from a low-resolution PSM dataset collected in a polysomnography sleep lab. Our approach leverages Vision Transformer models pre-trained on ImageNet using masked autoencoding (ViTMAE) and a pre-trained model for human pose estimation (ViTPose). These approaches outperform previous work from PSM-based sleep pose classification using deep learning (TCN) as well as traditional machine learning models (SVM, XGBoost, Random Forest) that use engineered features. We evaluate the performance of sleep position classification from 112 nights of patient recordings and validate it on a higher resolution 13-patient dataset. Despite the challenges of differentiating between sleep positions from low-resolution PSM data, our approach shows promise for real-world deployment in clinical settings
>
---
#### [new 007] Now you see it, Now you don't: Damage Label Agreement in Drone & Satellite Post-Disaster Imagery
- **分类: cs.CV**

- **简介: 该论文研究灾害后无人机与卫星图像中建筑损伤标签的一致性，属数据验证任务。针对两者标签差异（29.02%分歧）导致机器学习模型失准的问题，通过三场飓风15,814栋建筑的对比分析，发现卫星标签低估损伤超20%，证实数据源分布差异显著。提出四项建议以减少CV/ML灾害评估系统的误判风险。**

- **链接: [http://arxiv.org/pdf/2505.08117v1](http://arxiv.org/pdf/2505.08117v1)**

> **作者:** Thomas Manzini; Priyankari Perali; Jayesh Tripathi; Robin Murphy
>
> **备注:** 11 pages, 5 figures, 3 tables. Appearing at ACM FAccT'25
>
> **摘要:** This paper audits damage labels derived from coincident satellite and drone aerial imagery for 15,814 buildings across Hurricanes Ian, Michael, and Harvey, finding 29.02% label disagreement and significantly different distributions between the two sources, which presents risks and potential harms during the deployment of machine learning damage assessment systems. Currently, there is no known study of label agreement between drone and satellite imagery for building damage assessment. The only prior work that could be used to infer if such imagery-derived labels agree is limited by differing damage label schemas, misaligned building locations, and low data quantities. This work overcomes these limitations by comparing damage labels using the same damage label schemas and building locations from three hurricanes, with the 15,814 buildings representing 19.05 times more buildings considered than the most relevant prior work. The analysis finds satellite-derived labels significantly under-report damage by at least 20.43% compared to drone-derived labels (p<1.2x10^-117), and satellite- and drone-derived labels represent significantly different distributions (p<5.1x10^-175). This indicates that computer vision and machine learning (CV/ML) models trained on at least one of these distributions will misrepresent actual conditions, as the differing satellite and drone-derived distributions cannot simultaneously represent the distribution of actual conditions in a scene. This potential misrepresentation poses ethical risks and potential societal harm if not managed. To reduce the risk of future societal harms, this paper offers four recommendations to improve reliability and transparency to decisio-makers when deploying CV/ML damage assessment systems in practice
>
---
#### [new 008] Identifying Memorization of Diffusion Models through p-Laplace Analysis
- **分类: cs.CV; cs.NA; math.NA**

- **简介: 该论文研究扩散模型的记忆化检测，属于生成模型安全任务。提出基于p-Laplace算子分析估计得分函数的高阶微分，识别训练数据记忆现象。通过数值近似方法验证其在混合高斯模型和图像生成中的有效性，首次实现基于p-Laplace的生成模型记忆检测。**

- **链接: [http://arxiv.org/pdf/2505.08246v1](http://arxiv.org/pdf/2505.08246v1)**

> **作者:** Jonathan Brokman; Amit Giloni; Omer Hofman; Roman Vainshtein; Hisashi Kojima; Guy Gilboa
>
> **备注:** To be published in SSVM 2025 (proceedings of the 10th International Conference on Scale Space and Variational Methods in Computer Vision)
>
> **摘要:** Diffusion models, today's leading image generative models, estimate the score function, i.e. the gradient of the log probability of (perturbed) data samples, without direct access to the underlying probability distribution. This work investigates whether the estimated score function can be leveraged to compute higher-order differentials, namely p-Laplace operators. We show here these operators can be employed to identify memorized training data. We propose a numerical p-Laplace approximation based on the learned score functions, showing its effectiveness in identifying key features of the probability landscape. We analyze the structured case of Gaussian mixture models, and demonstrate the results carry-over to image generative models, where memorization identification based on the p-Laplace operator is performed for the first time.
>
---
#### [new 009] Congenital Heart Disease recognition using Deep Learning/Transformer models
- **分类: cs.CV**

- **简介: 该论文属于医学影像/信号的多模态分类任务，旨在通过深度学习提升先天性心脏病筛查精度。针对传统非侵入检测假阴性率高的问题，研究者构建了融合心音与胸部X光的双模态模型，在ZCHSound(73.9%)和DICOM(80.72%)数据集上验证了诊断有效性。**

- **链接: [http://arxiv.org/pdf/2505.08242v1](http://arxiv.org/pdf/2505.08242v1)**

> **作者:** Aidar Amangeldi; Vladislav Yarovenko; Angsar Taigonyrov
>
> **摘要:** Congenital Heart Disease (CHD) remains a leading cause of infant morbidity and mortality, yet non-invasive screening methods often yield false negatives. Deep learning models, with their ability to automatically extract features, can assist doctors in detecting CHD more effectively. In this work, we investigate the use of dual-modality (sound and image) deep learning methods for CHD diagnosis. We achieve 73.9% accuracy on the ZCHSound dataset and 80.72% accuracy on the DICOM Chest X-ray dataset.
>
---
#### [new 010] PrePrompt: Predictive prompting for class incremental learning
- **分类: cs.CV; I.5.4**

- **简介: 该论文研究基于预训练模型的类增量学习（CIL），解决传统相关性提示方法难以用少量提示拟合多任务特征空间的问题。提出PrePrompt框架，通过预训练模型的分类能力预测任务专属提示，分解为任务提示预测和标签预测两阶段，并引入特征翻译动态平衡新旧类偏差，实验验证其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.08586v1](http://arxiv.org/pdf/2505.08586v1)**

> **作者:** Libo Huang; Zhulin An; Chuanguang Yang; Boyu Diao; Fei Wang; Yan Zeng; Zhifeng Hao; Yongjun Xu
>
> **备注:** 16 pages, 29 figures, conference
>
> **摘要:** Class Incremental Learning (CIL) based on pre-trained models offers a promising direction for open-world continual learning. Existing methods typically rely on correlation-based strategies, where an image's classification feature is used as a query to retrieve the most related key prompts and select the corresponding value prompts for training. However, these approaches face an inherent limitation: fitting the entire feature space of all tasks with only a few trainable prompts is fundamentally challenging. We propose Predictive Prompting (PrePrompt), a novel CIL framework that circumvents correlation-based limitations by leveraging pre-trained models' natural classification ability to predict task-specific prompts. Specifically, PrePrompt decomposes CIL into a two-stage prediction framework: task-specific prompt prediction followed by label prediction. While theoretically appealing, this framework risks bias toward recent classes due to missing historical data for older classifier calibration. PrePrompt then mitigates this by incorporating feature translation, dynamically balancing stability and plasticity. Experiments across multiple benchmarks demonstrate PrePrompt's superiority over state-of-the-art prompt-based CIL methods. The code will be released upon acceptance.
>
---
#### [new 011] EventDiff: A Unified and Efficient Diffusion Model Framework for Event-based Video Frame Interpolation
- **分类: cs.CV**

- **简介: 该论文针对视频帧插值任务，解决复杂运动场景下传统事件相机方法依赖显式建模导致图像重建质量低的问题。提出EventDiff框架，结合事件-帧混合自编码器和时空交叉注意力模块，通过隐空间扩散去噪实现高效插值，两阶段训练策略在多个数据集取得最优性能，PSNR提升显著且推理更快。**

- **链接: [http://arxiv.org/pdf/2505.08235v1](http://arxiv.org/pdf/2505.08235v1)**

> **作者:** Hanle Zheng; Xujie Han; Zegang Peng; Shangbin Zhang; Guangxun Du; Zhuo Zou; Xilin Wang; Jibin Wu; Hao Guo; Lei Deng
>
> **摘要:** Video Frame Interpolation (VFI) is a fundamental yet challenging task in computer vision, particularly under conditions involving large motion, occlusion, and lighting variation. Recent advancements in event cameras have opened up new opportunities for addressing these challenges. While existing event-based VFI methods have succeeded in recovering large and complex motions by leveraging handcrafted intermediate representations such as optical flow, these designs often compromise high-fidelity image reconstruction under subtle motion scenarios due to their reliance on explicit motion modeling. Meanwhile, diffusion models provide a promising alternative for VFI by reconstructing frames through a denoising process, eliminating the need for explicit motion estimation or warping operations. In this work, we propose EventDiff, a unified and efficient event-based diffusion model framework for VFI. EventDiff features a novel Event-Frame Hybrid AutoEncoder (HAE) equipped with a lightweight Spatial-Temporal Cross Attention (STCA) module that effectively fuses dynamic event streams with static frames. Unlike previous event-based VFI methods, EventDiff performs interpolation directly in the latent space via a denoising diffusion process, making it more robust across diverse and challenging VFI scenarios. Through a two-stage training strategy that first pretrains the HAE and then jointly optimizes it with the diffusion model, our method achieves state-of-the-art performance across multiple synthetic and real-world event VFI datasets. The proposed method outperforms existing state-of-the-art event-based VFI methods by up to 1.98dB in PSNR on Vimeo90K-Triplet and shows superior performance in SNU-FILM tasks with multiple difficulty levels. Compared to the emerging diffusion-based VFI approach, our method achieves up to 5.72dB PSNR gain on Vimeo90K-Triplet and 4.24X faster inference.
>
---
#### [new 012] Ultra Lowrate Image Compression with Semantic Residual Coding and Compression-aware Diffusion
- **分类: cs.CV; eess.IV**

- **简介: 该论文研究超低码率图像压缩任务，解决现有方法重建质量与效率不足的问题。提出ResULIC框架，结合语义残差编码(SRC)捕获原始图像与压缩语义的差异，并通过压缩感知扩散模型(CDM)优化码率与扩散步长的对齐，提升重建效果。实验表明其在保真度和码率上显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.08281v1](http://arxiv.org/pdf/2505.08281v1)**

> **作者:** Anle Ke; Xu Zhang; Tong Chen; Ming Lu; Chao Zhou; Jiawen Gu; Zhan Ma
>
> **摘要:** Existing multimodal large model-based image compression frameworks often rely on a fragmented integration of semantic retrieval, latent compression, and generative models, resulting in suboptimal performance in both reconstruction fidelity and coding efficiency. To address these challenges, we propose a residual-guided ultra lowrate image compression named ResULIC, which incorporates residual signals into both semantic retrieval and the diffusion-based generation process. Specifically, we introduce Semantic Residual Coding (SRC) to capture the semantic disparity between the original image and its compressed latent representation. A perceptual fidelity optimizer is further applied for superior reconstruction quality. Additionally, we present the Compression-aware Diffusion Model (CDM), which establishes an optimal alignment between bitrates and diffusion time steps, improving compression-reconstruction synergy. Extensive experiments demonstrate the effectiveness of ResULIC, achieving superior objective and subjective performance compared to state-of-the-art diffusion-based methods with - 80.7%, -66.3% BD-rate saving in terms of LPIPS and FID. Project page is available at https: //njuvision.github.io/ResULIC/.
>
---
#### [new 013] SLAG: Scalable Language-Augmented Gaussian Splatting
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出SLAG框架，用于提升语言增强高斯溅射的实时性与可扩展性，解决大规模机器人应用中快速、高效构建3D场景语义模型的难题。通过整合2D视觉-语言模型特征至3D高斯参数，采用并行化计算和向量数据库，实现18倍加速（16GPU），保持ScanNet/LERF数据集性能。**

- **链接: [http://arxiv.org/pdf/2505.08124v1](http://arxiv.org/pdf/2505.08124v1)**

> **作者:** Laszlo Szilagyi; Francis Engelmann; Jeannette Bohg
>
> **摘要:** Language-augmented scene representations hold great promise for large-scale robotics applications such as search-and-rescue, smart cities, and mining. Many of these scenarios are time-sensitive, requiring rapid scene encoding while also being data-intensive, necessitating scalable solutions. Deploying these representations on robots with limited computational resources further adds to the challenge. To address this, we introduce SLAG, a multi-GPU framework for language-augmented Gaussian splatting that enhances the speed and scalability of embedding large scenes. Our method integrates 2D visual-language model features into 3D scenes using SAM and CLIP. Unlike prior approaches, SLAG eliminates the need for a loss function to compute per-Gaussian language embeddings. Instead, it derives embeddings from 3D Gaussian scene parameters via a normalized weighted average, enabling highly parallelized scene encoding. Additionally, we introduce a vector database for efficient embedding storage and retrieval. Our experiments show that SLAG achieves an 18 times speedup in embedding computation on a 16-GPU setup compared to OpenGaussian, while preserving embedding quality on the ScanNet and LERF datasets. For more details, visit our project website: https://slag-project.github.io/.
>
---
#### [new 014] Visually Interpretable Subtask Reasoning for Visual Question Answering
- **分类: cs.CV**

- **简介: 该论文针对视觉问答（VQA）中复杂问题的多步推理任务，解决现有方法计算成本高、数据适应性差导致精度低的问题。提出VISTAR框架，通过子任务驱动的训练机制，在MLLMs内部生成结构化文本/视觉解释，提升推理准确性和可解释性，无需依赖外部模型。实验验证其在基准测试中的有效性。**

- **链接: [http://arxiv.org/pdf/2505.08084v1](http://arxiv.org/pdf/2505.08084v1)**

> **作者:** Yu Cheng; Arushi Goel; Hakan Bilen
>
> **摘要:** Answering complex visual questions like `Which red furniture can be used for sitting?' requires multi-step reasoning, including object recognition, attribute filtering, and relational understanding. Recent work improves interpretability in multimodal large language models (MLLMs) by decomposing tasks into sub-task programs, but these methods are computationally expensive and less accurate due to poor adaptation to target data. To address this, we introduce VISTAR (Visually Interpretable Subtask-Aware Reasoning Model), a subtask-driven training framework that enhances both interpretability and reasoning by generating textual and visual explanations within MLLMs. Instead of relying on external models, VISTAR fine-tunes MLLMs to produce structured Subtask-of-Thought rationales (step-by-step reasoning sequences). Experiments on two benchmarks show that VISTAR consistently improves reasoning accuracy while maintaining interpretability. Our code and dataset will be available at https://github.com/ChengJade/VISTAR.
>
---
#### [new 015] Extending Large Vision-Language Model for Diverse Interactive Tasks in Autonomous Driving
- **分类: cs.CV**

- **简介: 该论文研究自动驾驶场景下多任务交互的视觉语言模型。针对现有模型缺乏3D感知、视角单一及指令理解不足的问题，提出多视角数据集NuInteract(150万图文对)和框架DriveMonkey，通过空间处理器融合3D定位能力，在3D视觉定位任务中提升9.86%。**

- **链接: [http://arxiv.org/pdf/2505.08725v1](http://arxiv.org/pdf/2505.08725v1)**

> **作者:** Zongchuang Zhao; Haoyu Fu; Dingkang Liang; Xin Zhou; Dingyuan Zhang; Hongwei Xie; Bing Wang; Xiang Bai
>
> **备注:** The dataset and code will be released at https://github.com/zc-zhao/DriveMonkey
>
> **摘要:** The Large Visual-Language Models (LVLMs) have significantly advanced image understanding. Their comprehension and reasoning capabilities enable promising applications in autonomous driving scenarios. However, existing research typically focuses on front-view perspectives and partial objects within scenes, struggling to achieve comprehensive scene understanding. Meanwhile, existing LVLMs suffer from the lack of mapping relationship between 2D and 3D and insufficient integration of 3D object localization and instruction understanding. To tackle these limitations, we first introduce NuInteract, a large-scale dataset with over 1.5M multi-view image language pairs spanning dense scene captions and diverse interactive tasks. Furthermore, we propose DriveMonkey, a simple yet effective framework that seamlessly integrates LVLMs with a spatial processor using a series of learnable queries. The spatial processor, designed as a plug-and-play component, can be initialized with pre-trained 3D detectors to improve 3D perception. Our experiments show that DriveMonkey outperforms general LVLMs, especially achieving a 9.86% notable improvement on the 3D visual grounding task. The dataset and code will be released at https://github.com/zc-zhao/DriveMonkey.
>
---
#### [new 016] DArFace: Deformation Aware Robustness for Low Quality Face Recognition
- **分类: cs.CV**

- **简介: 该论文针对低质量人脸识别任务，解决现有方法忽略局部非刚性形变导致性能下降的问题。提出DArFace框架，通过对抗训练模拟全局变换与局部弹性形变，并采用对比学习保持身份一致性，无需配对训练数据。实验表明其在主流低质量数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.08423v1](http://arxiv.org/pdf/2505.08423v1)**

> **作者:** Sadaf Gulshad; Abdullah Aldahlawi Thakaa
>
> **摘要:** Facial recognition systems have achieved remarkable success by leveraging deep neural networks, advanced loss functions, and large-scale datasets. However, their performance often deteriorates in real-world scenarios involving low-quality facial images. Such degradations, common in surveillance footage or standoff imaging include low resolution, motion blur, and various distortions, resulting in a substantial domain gap from the high-quality data typically used during training. While existing approaches attempt to address robustness by modifying network architectures or modeling global spatial transformations, they frequently overlook local, non-rigid deformations that are inherently present in real-world settings. In this work, we introduce DArFace, a Deformation-Aware robust Face recognition framework that enhances robustness to such degradations without requiring paired high- and low-quality training samples. Our method adversarially integrates both global transformations (e.g., rotation, translation) and local elastic deformations during training to simulate realistic low-quality conditions. Moreover, we introduce a contrastive objective to enforce identity consistency across different deformed views. Extensive evaluations on low-quality benchmarks including TinyFace, IJB-B, and IJB-C demonstrate that DArFace surpasses state-of-the-art methods, with significant gains attributed to the inclusion of local deformation modeling.
>
---
#### [new 017] Leveraging Segment Anything Model for Source-Free Domain Adaptation via Dual Feature Guided Auto-Prompting
- **分类: cs.CV**

- **简介: 该论文研究无源域适应分割任务，解决现有方法因领域差异生成错误边界框的问题。提出双特征引导自动提示方法：先聚合特征适应目标域，再结合目标模型和SAM特征逐步扩展边界框，并通过连通性分析优化SAM生成的伪标签，提升跨域分割性能。**

- **链接: [http://arxiv.org/pdf/2505.08527v1](http://arxiv.org/pdf/2505.08527v1)**

> **作者:** Zheang Huai; Hui Tang; Yi Li; Zhuangzhuang Chen; Xiaomeng Li
>
> **摘要:** Source-free domain adaptation (SFDA) for segmentation aims at adapting a model trained in the source domain to perform well in the target domain with only the source model and unlabeled target data.Inspired by the recent success of Segment Anything Model (SAM) which exhibits the generality of segmenting images of various modalities and in different domains given human-annotated prompts like bounding boxes or points, we for the first time explore the potentials of Segment Anything Model for SFDA via automatedly finding an accurate bounding box prompt. We find that the bounding boxes directly generated with existing SFDA approaches are defective due to the domain gap.To tackle this issue, we propose a novel Dual Feature Guided (DFG) auto-prompting approach to search for the box prompt. Specifically, the source model is first trained in a feature aggregation phase, which not only preliminarily adapts the source model to the target domain but also builds a feature distribution well-prepared for box prompt search. In the second phase, based on two feature distribution observations, we gradually expand the box prompt with the guidance of the target model feature and the SAM feature to handle the class-wise clustered target features and the class-wise dispersed target features, respectively. To remove the potentially enlarged false positive regions caused by the over-confident prediction of the target model, the refined pseudo-labels produced by SAM are further postprocessed based on connectivity analysis. Experiments on 3D and 2D datasets indicate that our approach yields superior performance compared to conventional methods. Code is available at https://github.com/zheangh/DFG.
>
---
#### [new 018] Controllable Image Colorization with Instance-aware Texts and Masks
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究基于扩散模型的实例级可控图像上色任务，旨在解决颜色渗透和绑定错误问题。提出MT-Color方法，通过像素级掩码注意力机制防止实例间颜色干扰，结合实例掩码和文本引导模块实现精准区域控制，并采用多实例采样策略。构建GPT-color数据集，实验证明模型和数据集性能优于现有方案。**

- **链接: [http://arxiv.org/pdf/2505.08705v1](http://arxiv.org/pdf/2505.08705v1)**

> **作者:** Yanru An; Ling Gui; Qiang Hu; Chunlei Cai; Tianxiao Ye; Xiaoyun Zhang; Yanfeng Wang
>
> **摘要:** Recently, the application of deep learning in image colorization has received widespread attention. The maturation of diffusion models has further advanced the development of image colorization models. However, current mainstream image colorization models still face issues such as color bleeding and color binding errors, and cannot colorize images at the instance level. In this paper, we propose a diffusion-based colorization method MT-Color to achieve precise instance-aware colorization with use-provided guidance. To tackle color bleeding issue, we design a pixel-level mask attention mechanism that integrates latent features and conditional gray image features through cross-attention. We use segmentation masks to construct cross-attention masks, preventing pixel information from exchanging between different instances. We also introduce an instance mask and text guidance module that extracts instance masks and text representations of each instance, which are then fused with latent features through self-attention, utilizing instance masks to form self-attention masks to prevent instance texts from guiding the colorization of other areas, thus mitigating color binding errors. Furthermore, we apply a multi-instance sampling strategy, which involves sampling each instance region separately and then fusing the results. Additionally, we have created a specialized dataset for instance-level colorization tasks, GPT-color, by leveraging large visual language models on existing image datasets. Qualitative and quantitative experiments show that our model and dataset outperform previous methods and datasets.
>
---
#### [new 019] Rejoining fragmented ancient bamboo slips with physics-driven deep learning
- **分类: cs.CV; cond-mat.mtrl-sci**

- **简介: 该论文属于文物复原任务，解决古代竹简碎片拼接难题。通过物理驱动深度学习框架WisePanda，模拟竹简断裂与老化特性生成合成数据，训练无监督匹配网络，提升拼接准确率（Top-50达52%）与效率（提速20倍），为考古修复提供新范式。**

- **链接: [http://arxiv.org/pdf/2505.08601v1](http://arxiv.org/pdf/2505.08601v1)**

> **作者:** Jinchi Zhu; Zhou Zhao; Hailong Lei; Xiaoguang Wang; Jialiang Lu; Jing Li; Qianqian Tang; Jiachen Shen; Gui-Song Xia; Bo Du; Yongchao Xu
>
> **摘要:** Bamboo slips are a crucial medium for recording ancient civilizations in East Asia, and offers invaluable archaeological insights for reconstructing the Silk Road, studying material culture exchanges, and global history. However, many excavated bamboo slips have been fragmented into thousands of irregular pieces, making their rejoining a vital yet challenging step for understanding their content. Here we introduce WisePanda, a physics-driven deep learning framework designed to rejoin fragmented bamboo slips. Based on the physics of fracture and material deterioration, WisePanda automatically generates synthetic training data that captures the physical properties of bamboo fragmentations. This approach enables the training of a matching network without requiring manually paired samples, providing ranked suggestions to facilitate the rejoining process. Compared to the leading curve matching method, WisePanda increases Top-50 matching accuracy from 36\% to 52\%. Archaeologists using WisePanda have experienced substantial efficiency improvements (approximately 20 times faster) when rejoining fragmented bamboo slips. This research demonstrates that incorporating physical principles into deep learning models can significantly enhance their performance, transforming how archaeologists restore and study fragmented artifacts. WisePanda provides a new paradigm for addressing data scarcity in ancient artifact restoration through physics-driven machine learning.
>
---
#### [new 020] SPAST: Arbitrary Style Transfer with Style Priors via Pre-trained Large-scale Model
- **分类: cs.CV**

- **简介: 该论文研究任意风格迁移任务，旨在解决现有方法生成质量低、内容结构失真及推理耗时长的问题。提出SPAST框架，结合预训练大模型，设计局部-全局特征融合模块和风格先验损失，在提升图像质量的同时缩短生成时间，兼顾内容保持与高效推理。**

- **链接: [http://arxiv.org/pdf/2505.08695v1](http://arxiv.org/pdf/2505.08695v1)**

> **作者:** Zhanjie Zhang; Quanwei Zhang; Junsheng Luan; Mengyuan Yang; Yun Wang; Lei Zhao
>
> **备注:** Accepted by Neural Networks
>
> **摘要:** Given an arbitrary content and style image, arbitrary style transfer aims to render a new stylized image which preserves the content image's structure and possesses the style image's style. Existing arbitrary style transfer methods are based on either small models or pre-trained large-scale models. The small model-based methods fail to generate high-quality stylized images, bringing artifacts and disharmonious patterns. The pre-trained large-scale model-based methods can generate high-quality stylized images but struggle to preserve the content structure and cost long inference time. To this end, we propose a new framework, called SPAST, to generate high-quality stylized images with less inference time. Specifically, we design a novel Local-global Window Size Stylization Module (LGWSSM)tofuse style features into content features. Besides, we introduce a novel style prior loss, which can dig out the style priors from a pre-trained large-scale model into the SPAST and motivate the SPAST to generate high-quality stylized images with short inference time.We conduct abundant experiments to verify that our proposed method can generate high-quality stylized images and less inference time compared with the SOTA arbitrary style transfer methods.
>
---
#### [new 021] Asynchronous Multi-Object Tracking with an Event Camera
- **分类: cs.CV**

- **简介: 该论文属于多目标跟踪任务，旨在解决动态场景下事件相机高效追踪多个小物体（如蜜蜂）的难题。提出异步事件多目标跟踪算法AEMOT，通过光流一致性检测特征，结合异步事件块跟踪和基于学习的验证机制，在自建数据集上实现比现有方法高37%的精度，并开源代码及数据集。**

- **链接: [http://arxiv.org/pdf/2505.08126v1](http://arxiv.org/pdf/2505.08126v1)**

> **作者:** Angus Apps; Ziwei Wang; Vladimir Perejogin; Timothy Molloy; Robert Mahony
>
> **备注:** 7 pages, 5 figures, published in IEEE International Conference on Robotics and Automation (ICRA), 2025
>
> **摘要:** Events cameras are ideal sensors for enabling robots to detect and track objects in highly dynamic environments due to their low latency output, high temporal resolution, and high dynamic range. In this paper, we present the Asynchronous Event Multi-Object Tracking (AEMOT) algorithm for detecting and tracking multiple objects by processing individual raw events asynchronously. AEMOT detects salient event blob features by identifying regions of consistent optical flow using a novel Field of Active Flow Directions built from the Surface of Active Events. Detected features are tracked as candidate objects using the recently proposed Asynchronous Event Blob (AEB) tracker in order to construct small intensity patches of each candidate object. A novel learnt validation stage promotes or discards candidate objects based on classification of their intensity patches, with promoted objects having their position, velocity, size, and orientation estimated at their event rate. We evaluate AEMOT on a new Bee Swarm Dataset, where it tracks dozens of small bees with precision and recall performance exceeding that of alternative event-based detection and tracking algorithms by over 37%. Source code and the labelled event Bee Swarm Dataset will be open sourced
>
---
#### [new 022] Topology-Guided Knowledge Distillation for Efficient Point Cloud Processing
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于点云高效处理任务，旨在解决高计算需求模型难以在资源受限场景部署的问题。提出基于拓扑感知和梯度引导的知识蒸馏框架，通过几何结构捕捉与特征对齐，将大教师模型知识迁移至轻量学生模型，实现模型尺寸缩减16倍、推理加速1.9倍，并在分割任务中达到SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.08101v1](http://arxiv.org/pdf/2505.08101v1)**

> **作者:** Luu Tung Hai; Thinh D. Le; Zhicheng Ding; Qing Tian; Truong-Son Hy
>
> **摘要:** Point cloud processing has gained significant attention due to its critical role in applications such as autonomous driving and 3D object recognition. However, deploying high-performance models like Point Transformer V3 in resource-constrained environments remains challenging due to their high computational and memory demands. This work introduces a novel distillation framework that leverages topology-aware representations and gradient-guided knowledge distillation to effectively transfer knowledge from a high-capacity teacher to a lightweight student model. Our approach captures the underlying geometric structures of point clouds while selectively guiding the student model's learning process through gradient-based feature alignment. Experimental results in the Nuscenes, SemanticKITTI, and Waymo datasets demonstrate that the proposed method achieves competitive performance, with an approximately 16x reduction in model size and a nearly 1.9x decrease in inference time compared to its teacher model. Notably, on NuScenes, our method achieves state-of-the-art performance among knowledge distillation techniques trained solely on LiDAR data, surpassing prior knowledge distillation baselines in segmentation performance. Our implementation is available publicly at: https://github.com/HySonLab/PointDistill
>
---
#### [new 023] ReSurgSAM2: Referring Segment Anything in Surgical Video via Credible Long-term Tracking
- **分类: cs.CV; eess.IV; q-bio.TO**

- **简介: 该论文属于手术场景的交互式目标分割任务，旨在解决现有方法效率低、无法长期跟踪的问题。提出ReSurgSAM2双阶段框架：先通过跨模态时空Mamba模型实现文本引导检测，再结合可信初始帧选择与多样性记忆机制完成长时跟踪，在保持61.2 FPS实时性的同时显著提升精度。**

- **链接: [http://arxiv.org/pdf/2505.08581v1](http://arxiv.org/pdf/2505.08581v1)**

> **作者:** Haofeng Liu; Mingqi Gao; Xuxiao Luo; Ziyue Wang; Guanyi Qin; Junde Wu; Yueming Jin
>
> **备注:** Early accepted by MICCAI 2025
>
> **摘要:** Surgical scene segmentation is critical in computer-assisted surgery and is vital for enhancing surgical quality and patient outcomes. Recently, referring surgical segmentation is emerging, given its advantage of providing surgeons with an interactive experience to segment the target object. However, existing methods are limited by low efficiency and short-term tracking, hindering their applicability in complex real-world surgical scenarios. In this paper, we introduce ReSurgSAM2, a two-stage surgical referring segmentation framework that leverages Segment Anything Model 2 to perform text-referred target detection, followed by tracking with reliable initial frame identification and diversity-driven long-term memory. For the detection stage, we propose a cross-modal spatial-temporal Mamba to generate precise detection and segmentation results. Based on these results, our credible initial frame selection strategy identifies the reliable frame for the subsequent tracking. Upon selecting the initial frame, our method transitions to the tracking stage, where it incorporates a diversity-driven memory mechanism that maintains a credible and diverse memory bank, ensuring consistent long-term tracking. Extensive experiments demonstrate that ReSurgSAM2 achieves substantial improvements in accuracy and efficiency compared to existing methods, operating in real-time at 61.2 FPS. Our code and datasets will be available at https://github.com/jinlab-imvr/ReSurgSAM2.
>
---
#### [new 024] Visual Image Reconstruction from Brain Activity via Latent Representation
- **分类: cs.CV; q-bio.NC**

- **简介: 该论文属于视觉图像重建任务，旨在从脑活动中解码并生成感知图像。通过回顾DNN和生成模型的进展，分析层次化潜在表征、组合策略等，解决重建主观视觉细节、实现零样本泛化等挑战。提出需优化数据集、评估指标及组合表征以增强泛化，并强调伦理风险。**

- **链接: [http://arxiv.org/pdf/2505.08429v1](http://arxiv.org/pdf/2505.08429v1)**

> **作者:** Yukiyasu Kamitani; Misato Tanaka; Ken Shirakawa
>
> **摘要:** Visual image reconstruction, the decoding of perceptual content from brain activity into images, has advanced significantly with the integration of deep neural networks (DNNs) and generative models. This review traces the field's evolution from early classification approaches to sophisticated reconstructions that capture detailed, subjective visual experiences, emphasizing the roles of hierarchical latent representations, compositional strategies, and modular architectures. Despite notable progress, challenges remain, such as achieving true zero-shot generalization for unseen images and accurately modeling the complex, subjective aspects of perception. We discuss the need for diverse datasets, refined evaluation metrics aligned with human perceptual judgments, and compositional representations that strengthen model robustness and generalizability. Ethical issues, including privacy, consent, and potential misuse, are underscored as critical considerations for responsible development. Visual image reconstruction offers promising insights into neural coding and enables new psychological measurements of visual experiences, with applications spanning clinical diagnostics and brain-machine interfaces.
>
---
#### [new 025] A Large-scale Benchmark on Geological Fault Delineation Models: Domain Shift, Training Dynamics, Generalizability, Evaluation and Inferential Behavior
- **分类: cs.CV**

- **简介: 该论文属于地质断层描绘模型的基准测试研究，旨在解决机器学习模型在地震数据中因领域偏移、评估不一致导致的泛化性不足问题。通过构建涵盖3个异构数据集的大规模基准（200+模型），系统评估预训练、微调等策略，揭示现有方法的脆弱性（如灾难性遗忘），为提升模型鲁棒性与可解释性提供实践指南。**

- **链接: [http://arxiv.org/pdf/2505.08585v1](http://arxiv.org/pdf/2505.08585v1)**

> **作者:** Jorge Quesada; Chen Zhou; Prithwijit Chowdhury; Mohammad Alotaibi; Ahmad Mustafa; Yusufjon Kumamnov; Mohit Prabhushankar; Ghassan AlRegib
>
> **摘要:** Machine learning has taken a critical role in seismic interpretation workflows, especially in fault delineation tasks. However, despite the recent proliferation of pretrained models and synthetic datasets, the field still lacks a systematic understanding of the generalizability limits of these models across seismic data representing a variety of geologic, acquisition and processing settings. Distributional shifts between different data sources, limitations in fine-tuning strategies and labeled data accessibility, and inconsistent evaluation protocols all represent major roadblocks in the deployment of reliable and robust models in real-world exploration settings. In this paper, we present the first large-scale benchmarking study explicitly designed to provide answers and guidelines for domain shift strategies in seismic interpretation. Our benchmark encompasses over $200$ models trained and evaluated on three heterogeneous datasets (synthetic and real data) including FaultSeg3D, CRACKS, and Thebe. We systematically assess pretraining, fine-tuning, and joint training strategies under varying degrees of domain shift. Our analysis highlights the fragility of current fine-tuning practices, the emergence of catastrophic forgetting, and the challenges of interpreting performance in a systematic manner. We establish a robust experimental baseline to provide insights into the tradeoffs inherent to current fault delineation workflows, and shed light on directions for developing more generalizable, interpretable and effective machine learning models for seismic interpretation. The insights and analyses reported provide a set of guidelines on the deployment of fault delineation models within seismic interpretation workflows.
>
---
#### [new 026] CNN and ViT Efficiency Study on Tiny ImageNet and DermaMNIST Datasets
- **分类: cs.CV**

- **简介: 该论文研究图像分类任务，旨在解决卷积和Transformer模型在资源受限环境中的效率权衡问题。通过微调ViT变体并对比ResNet-18，证明ViT可在保持或提升准确率的同时降低延迟和参数量，适合部署。实验基于DermaMNIST和TinyImageNet进行系统性超参数优化。**

- **链接: [http://arxiv.org/pdf/2505.08259v1](http://arxiv.org/pdf/2505.08259v1)**

> **作者:** Aidar Amangeldi; Angsar Taigonyrov; Muhammad Huzaid Jawad; Chinedu Emmanuel Mbonu
>
> **摘要:** This study evaluates the trade-offs between convolutional and transformer-based architectures on both medical and general-purpose image classification benchmarks. We use ResNet-18 as our baseline and introduce a fine-tuning strategy applied to four Vision Transformer variants (Tiny, Small, Base, Large) on DermatologyMNIST and TinyImageNet. Our goal is to reduce inference latency and model complexity with acceptable accuracy degradation. Through systematic hyperparameter variations, we demonstrate that appropriately fine-tuned Vision Transformers can match or exceed the baseline's performance, achieve faster inference, and operate with fewer parameters, highlighting their viability for deployment in resource-constrained environments.
>
---
#### [new 027] JSover: Joint Spectrum Estimation and Multi-Material Decomposition from Single-Energy CT Projections
- **分类: cs.CV; cs.AI**

- **简介: 该论文提出JSover框架，用于单能CT投影的联合光谱估计和多材料分解任务，解决传统两步法因忽略能量依赖衰减导致的伪影和噪声问题。通过物理先验建模和隐式神经表示，实现一步式重建，提升分解精度与效率。**

- **链接: [http://arxiv.org/pdf/2505.08123v1](http://arxiv.org/pdf/2505.08123v1)**

> **作者:** Qing Wu; Hongjiang Wei; Jingyi Yu; S. Kevin Zhou; Yuyao Zhang
>
> **备注:** 11 pages
>
> **摘要:** Multi-material decomposition (MMD) enables quantitative reconstruction of tissue compositions in the human body, supporting a wide range of clinical applications. However, traditional MMD typically requires spectral CT scanners and pre-measured X-ray energy spectra, significantly limiting clinical applicability. To this end, various methods have been developed to perform MMD using conventional (i.e., single-energy, SE) CT systems, commonly referred to as SEMMD. Despite promising progress, most SEMMD methods follow a two-step image decomposition pipeline, which first reconstructs monochromatic CT images using algorithms such as FBP, and then performs decomposition on these images. The initial reconstruction step, however, neglects the energy-dependent attenuation of human tissues, introducing severe nonlinear beam hardening artifacts and noise into the subsequent decomposition. This paper proposes JSover, a fundamentally reformulated one-step SEMMD framework that jointly reconstructs multi-material compositions and estimates the energy spectrum directly from SECT projections. By explicitly incorporating physics-informed spectral priors into the SEMMD process, JSover accurately simulates a virtual spectral CT system from SE acquisitions, thereby improving the reliability and accuracy of decomposition. Furthermore, we introduce implicit neural representation (INR) as an unsupervised deep learning solver for representing the underlying material maps. The inductive bias of INR toward continuous image patterns constrains the solution space and further enhances estimation quality. Extensive experiments on both simulated and real CT datasets show that JSover outperforms state-of-the-art SEMMD methods in accuracy and computational efficiency.
>
---
#### [new 028] Disruptive Transformation of Artworks in Master-Disciple Relationships: The Case of Ukiyo-e Artworks
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于艺术量化分析任务，旨在解决东方绘画缺乏大规模定量研究的问题。通过机器学习分析11,000幅浮世绘图像，计算艺术作品的网络创造力特征，揭示其整体创造力随文化成熟下降，但风格细分保持创新，为东方艺术演变提供数据驱动的文化历史视角。**

- **链接: [http://arxiv.org/pdf/2505.08284v1](http://arxiv.org/pdf/2505.08284v1)**

> **作者:** Honna Shinichi; Akira Matsui
>
> **摘要:** Artwork research has long relied on human sensibility and subjective judgment, but recent developments in machine learning have enabled the quantitative assessment of features that humans could not discover. In Western paintings, comprehensive analyses have been conducted from various perspectives in conjunction with large databases, but such extensive analysis has not been sufficiently conducted for Eastern paintings. Then, we focus on Ukiyo-e, a traditional Japanese art form, as a case study of Eastern paintings, and conduct a quantitative analysis of creativity in works of art using 11,000 high-resolution images. This involves using the concept of calculating creativity from networks to analyze both the creativity of the artwork and that of the artists. As a result, In terms of Ukiyo-e as a whole, it was found that the creativity of its appearance has declined with the maturation of culture, but in terms of style, it has become more segmented with the maturation of culture and has maintained a high level of creativity. This not only provides new insights into the study of Ukiyo-e but also shows how Ukiyo-e has evolved within the ongoing cultural history, playing a culturally significant role in the analysis of Eastern art.
>
---
#### [new 029] Multi-modal wound classification using wound image and location by Xception and Gaussian Mixture Recurrent Neural Network (GMRNN)
- **分类: cs.CV**

- **简介: 该论文属于多模态医学图像分类任务，旨在提升糖尿病足、压疮等四类常见伤口的诊断准确性。研究者结合Xception网络提取图像特征与GMRNN处理位置信息，通过迁移学习构建融合模型。实验证明该方法在四分类任务中准确率达78.77-100%，优于传统深度神经网络，解决了伤口类型自动识别精度不足的问题。**

- **链接: [http://arxiv.org/pdf/2505.08086v1](http://arxiv.org/pdf/2505.08086v1)**

> **作者:** Ramin Mousa; Ehsan Matbooe; Hakimeh Khojasteh; Amirali Bengari; Mohammadmahdi Vahediahmar
>
> **摘要:** The effective diagnosis of acute and hard-to-heal wounds is crucial for wound care practitioners to provide effective patient care. Poor clinical outcomes are often linked to infection, peripheral vascular disease, and increasing wound depth, which collectively exacerbate these comorbidities. However, diagnostic tools based on Artificial Intelligence (AI) speed up the interpretation of medical images and improve early detection of disease. In this article, we propose a multi-modal AI model based on transfer learning (TL), which combines two state-of-the-art architectures, Xception and GMRNN, for wound classification. The multi-modal network is developed by concatenating the features extracted by a transfer learning algorithm and location features to classify the wound types of diabetic, pressure, surgical, and venous ulcers. The proposed method is comprehensively compared with deep neural networks (DNN) for medical image analysis. The experimental results demonstrate a notable wound-class classifications (containing only diabetic, pressure, surgical, and venous) vary from 78.77 to 100\% in various experiments. The results presented in this study showcase the exceptional accuracy of the proposed methodology in accurately classifying the most commonly occurring wound types using wound images and their corresponding locations.
>
---
#### [new 030] HMPNet: A Feature Aggregation Architecture for Maritime Object Detection from a Shipborne Perspective
- **分类: cs.CV**

- **简介: 该论文聚焦船载视角的海上目标检测任务，解决因海事数据稀缺导致现有视觉检测模型性能受限的问题。通过提出Navigation12多环境数据集，并设计轻量级HMPNet模型（含动态特征聚合结构和多尺度模块），在精度和效率上超越主流方法，参数量减少23%，mAP提升3.3%。**

- **链接: [http://arxiv.org/pdf/2505.08231v1](http://arxiv.org/pdf/2505.08231v1)**

> **作者:** Yu Zhang; Fengyuan Liu; Juan Lyu; Yi Wei; Changdong Yu
>
> **备注:** This paper has been accepted to ICME 2025
>
> **摘要:** In the realm of intelligent maritime navigation, object detection from a shipborne perspective is paramount. Despite the criticality, the paucity of maritime-specific data impedes the deployment of sophisticated visual perception techniques, akin to those utilized in autonomous vehicular systems, within the maritime context. To bridge this gap, we introduce Navigation12, a novel dataset annotated for 12 object categories under diverse maritime environments and weather conditions. Based upon this dataset, we propose HMPNet, a lightweight architecture tailored for shipborne object detection. HMPNet incorporates a hierarchical dynamic modulation backbone to bolster feature aggregation and expression, complemented by a matrix cascading poly-scale neck and a polymerization weight sharing detector, facilitating efficient multi-scale feature aggregation. Empirical evaluations indicate that HMPNet surpasses current state-of-the-art methods in terms of both accuracy and computational efficiency, realizing a 3.3% improvement in mean Average Precision over YOLOv11n, the prevailing model, and reducing parameters by 23%.
>
---
#### [new 031] A Deep Learning-Driven Framework for Inhalation Injury Grading Using Bronchoscopy Images
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于医学图像分类任务，旨在解决吸入性损伤传统分级方法（如AIS）主观性强、数据不足的问题。通过提出改进的StarGAN模型（整合Patch Loss和SSIM Loss）生成高质量支气管镜图像，并利用Swin Transformer分类器实现77.78%的准确率，较基线提升11.11%，经临床验证图像真实有效。**

- **链接: [http://arxiv.org/pdf/2505.08517v1](http://arxiv.org/pdf/2505.08517v1)**

> **作者:** Yifan Li; Alan W Pang; Jo Woon Chong
>
> **摘要:** Inhalation injuries face a challenge in clinical diagnosis and grading due to the limitations of traditional methods, such as Abbreviated Injury Score (AIS), which rely on subjective assessments and show weak correlations with clinical outcomes. This study introduces a novel deep learning-based framework for grading inhalation injuries using bronchoscopy images with the duration of mechanical ventilation as an objective metric. To address the scarcity of medical imaging data, we propose enhanced StarGAN, a generative model that integrates Patch Loss and SSIM Loss to improve synthetic images' quality and clinical relevance. The augmented dataset generated by enhanced StarGAN significantly improved classification performance when evaluated using the Swin Transformer, achieving an accuracy of 77.78%, an 11.11% improvement over the original dataset. Image quality was assessed using the Fr\'echet Inception Distance (FID), where Enhanced StarGAN achieved the lowest FID of 30.06, outperforming baseline models. Burn surgeons confirmed the realism and clinical relevance of the generated images, particularly the preservation of bronchial structures and color distribution. These results highlight the potential of enhanced StarGAN in addressing data limitations and improving classification accuracy for inhalation injury grading.
>
---
#### [new 032] Vision Foundation Model Embedding-Based Semantic Anomaly Detection
- **分类: cs.CV; cs.LG**

- **简介: 该论文研究语义异常检测任务，解决自主系统中因上下文无效视觉组合引发的故障问题。提出基于视觉基础模型嵌入的框架，通过对比实时图像与安全场景数据库定位异常，采用网格嵌入和实例分割两种方法，并设计过滤机制降低误报。实验表明实例分割方法结合过滤在仿真环境中性能接近GPT-4o，实现精准异常定位。**

- **链接: [http://arxiv.org/pdf/2505.07998v1](http://arxiv.org/pdf/2505.07998v1)**

> **作者:** Max Peter Ronecker; Matthew Foutter; Amine Elhafsi; Daniele Gammelli; Ihor Barakaiev; Marco Pavone; Daniel Watzenig
>
> **备注:** Accepted for the Workshop "Safely Leveraging Vision-Language Foundation Models in Robotics: Challenges and Opportunities" at ICRA 2025
>
> **摘要:** Semantic anomalies are contextually invalid or unusual combinations of familiar visual elements that can cause undefined behavior and failures in system-level reasoning for autonomous systems. This work explores semantic anomaly detection by leveraging the semantic priors of state-of-the-art vision foundation models, operating directly on the image. We propose a framework that compares local vision embeddings from runtime images to a database of nominal scenarios in which the autonomous system is deemed safe and performant. In this work, we consider two variants of the proposed framework: one using raw grid-based embeddings, and another leveraging instance segmentation for object-centric representations. To further improve robustness, we introduce a simple filtering mechanism to suppress false positives. Our evaluations on CARLA-simulated anomalies show that the instance-based method with filtering achieves performance comparable to GPT-4o, while providing precise anomaly localization. These results highlight the potential utility of vision embeddings from foundation models for real-time anomaly detection in autonomous systems.
>
---
#### [new 033] Thermal Detection of People with Mobility Restrictions for Barrier Reduction at Traffic Lights Controlled Intersections
- **分类: cs.CV**

- **简介: 该论文属于智能交通系统的计算机视觉任务，旨在解决现有RGB交通灯系统忽视行动不便者需求、环境适应性差及隐私问题。通过热成像技术构建TD4PWMR数据集，开发改进的YOLO-Thermal检测模型，实现恶劣环境下精准识别，动态调整信号时长并触发辅助功能，促进无障碍路口建设。**

- **链接: [http://arxiv.org/pdf/2505.08568v1](http://arxiv.org/pdf/2505.08568v1)**

> **作者:** Xiao Ni; Carsten Kuehnel; Xiaoyi Jiang
>
> **摘要:** Rapid advances in deep learning for computer vision have driven the adoption of RGB camera-based adaptive traffic light systems to improve traffic safety and pedestrian comfort. However, these systems often overlook the needs of people with mobility restrictions. Moreover, the use of RGB cameras presents significant challenges, including limited detection performance under adverse weather or low-visibility conditions, as well as heightened privacy concerns. To address these issues, we propose a fully automated, thermal detector-based traffic light system that dynamically adjusts signal durations for individuals with walking impairments or mobility burden and triggers the auditory signal for visually impaired individuals, thereby advancing towards barrier-free intersection for all users. To this end, we build the thermal dataset for people with mobility restrictions (TD4PWMR), designed to capture diverse pedestrian scenarios, particularly focusing on individuals with mobility aids or mobility burden under varying environmental conditions, such as different lighting, weather, and crowded urban settings. While thermal imaging offers advantages in terms of privacy and robustness to adverse conditions, it also introduces inherent hurdles for object detection due to its lack of color and fine texture details and generally lower resolution of thermal images. To overcome these limitations, we develop YOLO-Thermal, a novel variant of the YOLO architecture that integrates advanced feature extraction and attention mechanisms for enhanced detection accuracy and robustness in thermal imaging. Experiments demonstrate that the proposed thermal detector outperforms existing detectors, while the proposed traffic light system effectively enhances barrier-free intersection. The source codes and dataset are available at https://github.com/leon2014dresden/YOLO-THERMAL.
>
---
#### [new 034] TT-DF: A Large-Scale Diffusion-Based Dataset and Benchmark for Human Body Forgery Detection
- **分类: cs.CV**

- **简介: 该论文针对人体伪造检测任务，解决现有数据及方法匮乏问题。提出了大规模扩散模型数据集TT-DF（含6,120伪造视频），涵盖多种生成方法和压缩版本，并开发时态光流网络TOF-Net，通过时空不一致性检测伪造内容，性能优于现有面部伪造检测模型。**

- **链接: [http://arxiv.org/pdf/2505.08437v1](http://arxiv.org/pdf/2505.08437v1)**

> **作者:** Wenkui Yang; Zhida Zhang; Xiaoqiang Zhou; Junxian Duan; Jie Cao
>
> **备注:** Accepted to PRCV 2024
>
> **摘要:** The emergence and popularity of facial deepfake methods spur the vigorous development of deepfake datasets and facial forgery detection, which to some extent alleviates the security concerns about facial-related artificial intelligence technologies. However, when it comes to human body forgery, there has been a persistent lack of datasets and detection methods, due to the later inception and complexity of human body generation methods. To mitigate this issue, we introduce TikTok-DeepFake (TT-DF), a novel large-scale diffusion-based dataset containing 6,120 forged videos with 1,378,857 synthetic frames, specifically tailored for body forgery detection. TT-DF offers a wide variety of forgery methods, involving multiple advanced human image animation models utilized for manipulation, two generative configurations based on the disentanglement of identity and pose information, as well as different compressed versions. The aim is to simulate any potential unseen forged data in the wild as comprehensively as possible, and we also furnish a benchmark on TT-DF. Additionally, we propose an adapted body forgery detection model, Temporal Optical Flow Network (TOF-Net), which exploits the spatiotemporal inconsistencies and optical flow distribution differences between natural data and forged data. Our experiments demonstrate that TOF-Net achieves favorable performance on TT-DF, outperforming current state-of-the-art extendable facial forgery detection models. For our TT-DF dataset, please refer to https://github.com/HashTAG00002/TT-DF.
>
---
#### [new 035] VCRBench: Exploring Long-form Causal Reasoning Capabilities of Large Video Language Models
- **分类: cs.CV**

- **简介: 该论文属于视频理解与因果推理任务，旨在解决大型视频语言模型（LVLM）在长序列视觉因果推理评估缺失的问题。通过构建VCRBench基准测试（包含步骤打乱的日常活动视频），验证模型对事件因果关系的识别与排序能力，并提出分解视觉识别与推理的RRD方法，将准确率提升25.2%，揭示模型过度依赖语言知识的局限性。**

- **链接: [http://arxiv.org/pdf/2505.08455v1](http://arxiv.org/pdf/2505.08455v1)**

> **作者:** Pritam Sarkar; Ali Etemad
>
> **摘要:** Despite recent advances in video understanding, the capabilities of Large Video Language Models (LVLMs) to perform video-based causal reasoning remains underexplored, largely due to the absence of relevant and dedicated benchmarks for evaluating causal reasoning in visually grounded and goal-driven settings. To fill this gap, we introduce a novel benchmark named Video-based long-form Causal Reasoning (VCRBench). We create VCRBench using procedural videos of simple everyday activities, where the steps are deliberately shuffled with each clip capturing a key causal event, to test whether LVLMs can identify, reason about, and correctly sequence the events needed to accomplish a specific goal. Moreover, the benchmark is carefully designed to prevent LVLMs from exploiting linguistic shortcuts, as seen in multiple-choice or binary QA formats, while also avoiding the challenges associated with evaluating open-ended QA. Our evaluation of state-of-the-art LVLMs on VCRBench suggests that these models struggle with video-based long-form causal reasoning, primarily due to their difficulty in modeling long-range causal dependencies directly from visual observations. As a simple step toward enabling such capabilities, we propose Recognition-Reasoning Decomposition (RRD), a modular approach that breaks video-based causal reasoning into two sub-tasks of video recognition and causal reasoning. Our experiments on VCRBench show that RRD significantly boosts accuracy on VCRBench, with gains of up to 25.2%. Finally, our thorough analysis reveals interesting insights, for instance, that LVLMs primarily rely on language knowledge for complex video-based long-form causal reasoning tasks.
>
---
#### [new 036] An incremental algorithm for non-convex AI-enhanced medical image processing
- **分类: cs.CV; cs.NA; math.NA**

- **简介: 该论文针对医学成像中的非凸正则化逆问题优化困难，提出incDG框架。通过结合深度学习与增量模型优化，利用神经网络生成初始解，再经变分求解器迭代优化，平衡AI效率与理论稳定性。实验验证其在去模糊、断层重建等任务中优于传统方法和纯深度学习方法，无需真实数据训练仍保持性能。**

- **链接: [http://arxiv.org/pdf/2505.08324v1](http://arxiv.org/pdf/2505.08324v1)**

> **作者:** Elena Morotti
>
> **摘要:** Solving non-convex regularized inverse problems is challenging due to their complex optimization landscapes and multiple local minima. However, these models remain widely studied as they often yield high-quality, task-oriented solutions, particularly in medical imaging, where the goal is to enhance clinically relevant features rather than merely minimizing global error. We propose incDG, a hybrid framework that integrates deep learning with incremental model-based optimization to efficiently approximate the $\ell_0$-optimal solution of imaging inverse problems. Built on the Deep Guess strategy, incDG exploits a deep neural network to generate effective initializations for a non-convex variational solver, which refines the reconstruction through regularized incremental iterations. This design combines the efficiency of Artificial Intelligence (AI) tools with the theoretical guarantees of model-based optimization, ensuring robustness and stability. We validate incDG on TpV-regularized optimization tasks, demonstrating its effectiveness in medical image deblurring and tomographic reconstruction across diverse datasets, including synthetic images, brain CT slices, and chest-abdomen scans. Results show that incDG outperforms both conventional iterative solvers and deep learning-based methods, achieving superior accuracy and stability. Moreover, we confirm that training incDG without ground truth does not significantly degrade performance, making it a practical and powerful tool for solving non-convex inverse problems in imaging and beyond.
>
---
#### [new 037] G-MSGINet: A Grouped Multi-Scale Graph-Involution Network for Contactless Fingerprint Recognition
- **分类: cs.CV**

- **简介: 该论文提出G-MSGINet框架，用于非接触式指纹识别任务，解决现有方法依赖复杂预处理、泛化性差的问题。通过新型GMSGI层整合像素级卷积、多尺度核和图关系建模，联合优化局部细节特征与全局拓扑表示，无需方向标注。实验显示其参数少且性能优于基线，在三个基准数据集上F1达0.83±0.02，识别准确率97-99.1%。**

- **链接: [http://arxiv.org/pdf/2505.08233v1](http://arxiv.org/pdf/2505.08233v1)**

> **作者:** Santhoshkumar Peddi; Soham Bandyopadhyay; Debasis Samanta
>
> **摘要:** This paper presents G-MSGINet, a unified and efficient framework for robust contactless fingerprint recognition that jointly performs minutiae localization and identity embedding directly from raw input images. Existing approaches rely on multi-branch architectures, orientation labels, or complex preprocessing steps, which limit scalability and generalization across real-world acquisition scenarios. In contrast, the proposed architecture introduces the GMSGI layer, a novel computational module that integrates grouped pixel-level involution, dynamic multi-scale kernel generation, and graph-based relational modelling into a single processing unit. Stacked GMSGI layers progressively refine both local minutiae-sensitive features and global topological representations through end-to-end optimization. The architecture eliminates explicit orientation supervision and adapts graph connectivity directly from learned kernel descriptors, thereby capturing meaningful structural relationships among fingerprint regions without fixed heuristics. Extensive experiments on three benchmark datasets, namely PolyU, CFPose, and Benchmark 2D/3D, demonstrate that G-MSGINet consistently achieves minutiae F1-scores in the range of $0.83\pm0.02$ and Rank-1 identification accuracies between 97.0% and 99.1%, while maintaining an Equal Error Rate (EER) as low as 0.5%. These results correspond to improvements of up to 4.8% in F1-score and 1.4% in Rank-1 accuracy when compared to prior methods, using only 0.38 million parameters and 6.63 giga floating-point operations, which represents up to ten times fewer parameters than competitive baselines. This highlights the scalability and effectiveness of G-MSGINet in real-world contactless biometric recognition scenarios.
>
---
#### [new 038] FauForensics: Boosting Audio-Visual Deepfake Detection with Facial Action Units
- **分类: cs.CV**

- **简介: 该论文属于多模态深度伪造检测任务，旨在解决现有方法对跨模态伪造内容检测效果差、泛化不足的问题。提出FauForensics框架，通过引入生物特征（面部动作单元）增强伪造抗性，并设计帧级视听融合模块动态对齐唇部-音频关系，提升跨数据集检测精度。实验表明其性能优于现有方法4.83%。**

- **链接: [http://arxiv.org/pdf/2505.08294v1](http://arxiv.org/pdf/2505.08294v1)**

> **作者:** Jian Wang; Baoyuan Wu; Li Liu; Qingshan Liu
>
> **摘要:** The rapid evolution of generative AI has increased the threat of realistic audio-visual deepfakes, demanding robust detection methods. Existing solutions primarily address unimodal (audio or visual) forgeries but struggle with multimodal manipulations due to inadequate handling of heterogeneous modality features and poor generalization across datasets. To this end, we propose a novel framework called FauForensics by introducing biologically invariant facial action units (FAUs), which is a quantitative descriptor of facial muscle activity linked to emotion physiology. It serves as forgery-resistant representations that reduce domain dependency while capturing subtle dynamics often disrupted in synthetic content. Besides, instead of comparing entire video clips as in prior works, our method computes fine-grained frame-wise audiovisual similarities via a dedicated fusion module augmented with learnable cross-modal queries. It dynamically aligns temporal-spatial lip-audio relationships while mitigating multi-modal feature heterogeneity issues. Experiments on FakeAVCeleb and LAV-DF show state-of-the-art (SOTA) performance and superior cross-dataset generalizability with up to an average of 4.83\% than existing methods.
>
---
#### [new 039] Removing Watermarks with Partial Regeneration using Semantic Information
- **分类: cs.CV; cs.AI; cs.CR**

- **简介: 该论文研究对抗攻击，解决现有语义水印易被去除的问题。提出SemanticRegen方法，通过视觉语言模型生成描述、分割前景后仅修复背景，在保留图像内容的同时消除水印。实验表明其能突破主流水印方案，揭示当前防御机制在语义攻击下的脆弱性。**

- **链接: [http://arxiv.org/pdf/2505.08234v1](http://arxiv.org/pdf/2505.08234v1)**

> **作者:** Krti Tallam; John Kevin Cava; Caleb Geniesse; N. Benjamin Erichson; Michael W. Mahoney
>
> **摘要:** As AI-generated imagery becomes ubiquitous, invisible watermarks have emerged as a primary line of defense for copyright and provenance. The newest watermarking schemes embed semantic signals - content-aware patterns that are designed to survive common image manipulations - yet their true robustness against adaptive adversaries remains under-explored. We expose a previously unreported vulnerability and introduce SemanticRegen, a three-stage, label-free attack that erases state-of-the-art semantic and invisible watermarks while leaving an image's apparent meaning intact. Our pipeline (i) uses a vision-language model to obtain fine-grained captions, (ii) extracts foreground masks with zero-shot segmentation, and (iii) inpaints only the background via an LLM-guided diffusion model, thereby preserving salient objects and style cues. Evaluated on 1,000 prompts across four watermarking systems - TreeRing, StegaStamp, StableSig, and DWT/DCT - SemanticRegen is the only method to defeat the semantic TreeRing watermark (p = 0.10 > 0.05) and reduces bit-accuracy below 0.75 for the remaining schemes, all while maintaining high perceptual quality (masked SSIM = 0.94 +/- 0.01). We further introduce masked SSIM (mSSIM) to quantify fidelity within foreground regions, showing that our attack achieves up to 12 percent higher mSSIM than prior diffusion-based attackers. These results highlight an urgent gap between current watermark defenses and the capabilities of adaptive, semantics-aware adversaries, underscoring the need for watermarking algorithms that are resilient to content-preserving regenerative attacks.
>
---
#### [new 040] Boosting Zero-shot Stereo Matching using Large-scale Mixed Images Sources in the Real World
- **分类: cs.CV**

- **简介: 该论文属于零样本立体匹配任务，旨在解决标注数据稀缺和合成-真实域差距问题。提出BooSTer框架，融合视觉基础模型与多源图像（合成/真实/单目），通过单目深度估计和扩散模型生成立体数据，引入伪标签和动态损失优化监督，并利用基础模型提升特征泛化性，显著提升跨域和少标注场景的匹配精度。**

- **链接: [http://arxiv.org/pdf/2505.08607v1](http://arxiv.org/pdf/2505.08607v1)**

> **作者:** Yuran Wang; Yingping Liang; Ying Fu
>
> **摘要:** Stereo matching methods rely on dense pixel-wise ground truth labels, which are laborious to obtain, especially for real-world datasets. The scarcity of labeled data and domain gaps between synthetic and real-world images also pose notable challenges. In this paper, we propose a novel framework, \textbf{BooSTer}, that leverages both vision foundation models and large-scale mixed image sources, including synthetic, real, and single-view images. First, to fully unleash the potential of large-scale single-view images, we design a data generation strategy combining monocular depth estimation and diffusion models to generate dense stereo matching data from single-view images. Second, to tackle sparse labels in real-world datasets, we transfer knowledge from monocular depth estimation models, using pseudo-mono depth labels and a dynamic scale- and shift-invariant loss for additional supervision. Furthermore, we incorporate vision foundation model as an encoder to extract robust and transferable features, boosting accuracy and generalization. Extensive experiments on benchmark datasets demonstrate the effectiveness of our approach, achieving significant improvements in accuracy over existing methods, particularly in scenarios with limited labeled data and domain shifts.
>
---
#### [new 041] Unsupervised Out-of-Distribution Detection in Medical Imaging Using Multi-Exit Class Activation Maps and Feature Masking
- **分类: cs.CV**

- **简介: 本文提出一种无监督医学影像分布外检测方法MECAM，通过多出口网络提取不同层次类激活图，结合特征掩码技术，利用分布内数据特征扰动敏感性差异实现检测，在多个医学数据集上验证优于现有方法，提升临床模型可靠性。**

- **链接: [http://arxiv.org/pdf/2505.08604v1](http://arxiv.org/pdf/2505.08604v1)**

> **作者:** Yu-Jen Chen; Xueyang Li; Yiyu Shi; Tsung-Yi Ho
>
> **备注:** 10 pages, 2 figures
>
> **摘要:** Out-of-distribution (OOD) detection is essential for ensuring the reliability of deep learning models in medical imaging applications. This work is motivated by the observation that class activation maps (CAMs) for in-distribution (ID) data typically emphasize regions that are highly relevant to the model's predictions, whereas OOD data often lacks such focused activations. By masking input images with inverted CAMs, the feature representations of ID data undergo more substantial changes compared to those of OOD data, offering a robust criterion for differentiation. In this paper, we introduce a novel unsupervised OOD detection framework, Multi-Exit Class Activation Map (MECAM), which leverages multi-exit CAMs and feature masking. By utilizing mult-exit networks that combine CAMs from varying resolutions and depths, our method captures both global and local feature representations, thereby enhancing the robustness of OOD detection. We evaluate MECAM on multiple ID datasets, including ISIC19 and PathMNIST, and test its performance against three medical OOD datasets, RSNA Pneumonia, COVID-19, and HeadCT, and one natural image OOD dataset, iSUN. Comprehensive comparisons with state-of-the-art OOD detection methods validate the effectiveness of our approach. Our findings emphasize the potential of multi-exit networks and feature masking for advancing unsupervised OOD detection in medical imaging, paving the way for more reliable and interpretable models in clinical practice.
>
---
#### [new 042] Attention-based Generative Latent Replay: A Continual Learning Approach for WSI Analysis
- **分类: cs.CV; cs.ET**

- **简介: 该论文属于计算病理学中的持续学习任务，解决全切片图像分类的领域偏移和隐私问题。提出AGLR-CL框架，通过高斯混合模型生成合成样本替代原始数据存储，结合注意力机制筛选关键特征，实现跨器官/机构的隐私保护增量学习，在多个病理数据集验证了知识保留与适应能力。**

- **链接: [http://arxiv.org/pdf/2505.08524v1](http://arxiv.org/pdf/2505.08524v1)**

> **作者:** Pratibha Kumari; Daniel Reisenbüchler; Afshin Bozorgpour; Nadine S. Schaadt; Friedrich Feuerhake; Dorit Merhof
>
> **摘要:** Whole slide image (WSI) classification has emerged as a powerful tool in computational pathology, but remains constrained by domain shifts, e.g., due to different organs, diseases, or institution-specific variations. To address this challenge, we propose an Attention-based Generative Latent Replay Continual Learning framework (AGLR-CL), in a multiple instance learning (MIL) setup for domain incremental WSI classification. Our method employs Gaussian Mixture Models (GMMs) to synthesize WSI representations and patch count distributions, preserving knowledge of past domains without explicitly storing original data. A novel attention-based filtering step focuses on the most salient patch embeddings, ensuring high-quality synthetic samples. This privacy-aware strategy obviates the need for replay buffers and outperforms other buffer-free counterparts while matching the performance of buffer-based solutions. We validate AGLR-CL on clinically relevant biomarker detection and molecular status prediction across multiple public datasets with diverse centers, organs, and patient cohorts. Experimental results confirm its ability to retain prior knowledge and adapt to new domains, offering an effective, privacy-preserving avenue for domain incremental continual learning in WSI classification.
>
---
#### [new 043] Open the Eyes of MPNN: Vision Enhances MPNN in Link Prediction
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文针对图神经网络中的链接预测任务，提出融合视觉感知增强传统MPNN方法的结构理解能力。为解决现有方法忽视视觉信息的问题，作者设计了Graph Vision Network（GVN）及其高效变体E-GVN，通过视觉结构感知机制在七大数据集上取得性能突破，兼容现有技术并刷新SOTA结果，开辟了视觉增强图学习的新方向。**

- **链接: [http://arxiv.org/pdf/2505.08266v1](http://arxiv.org/pdf/2505.08266v1)**

> **作者:** Yanbin Wei; Xuehao Wang; Zhan Zhuang; Yang Chen; Shuhao Chen; Yulong Zhang; Yu Zhang; James Kwok
>
> **备注:** ICML 2025
>
> **摘要:** Message-passing graph neural networks (MPNNs) and structural features (SFs) are cornerstones for the link prediction task. However, as a common and intuitive mode of understanding, the potential of visual perception has been overlooked in the MPNN community. For the first time, we equip MPNNs with vision structural awareness by proposing an effective framework called Graph Vision Network (GVN), along with a more efficient variant (E-GVN). Extensive empirical results demonstrate that with the proposed frameworks, GVN consistently benefits from the vision enhancement across seven link prediction datasets, including challenging large-scale graphs. Such improvements are compatible with existing state-of-the-art (SOTA) methods and GVNs achieve new SOTA results, thereby underscoring a promising novel direction for link prediction.
>
---
#### [new 044] Visual Watermarking in the Era of Diffusion Models: Advances and Challenges
- **分类: cs.CV**

- **简介: 该论文研究扩散模型时代的视觉水印技术，属于数字版权保护领域。针对生成AI滥用导致的侵权问题，分析扩散模型在水印生成中的优势（如高鲁棒性、不可感知特征嵌入），探讨其应对伪造威胁的能力，并提出整合先进模型提升水印安全性的解决方案。**

- **链接: [http://arxiv.org/pdf/2505.08197v1](http://arxiv.org/pdf/2505.08197v1)**

> **作者:** Junxian Duan; Jiyang Guang; Wenkui Yang; Ran He
>
> **摘要:** As generative artificial intelligence technologies like Stable Diffusion advance, visual content becomes more vulnerable to misuse, raising concerns about copyright infringement. Visual watermarks serve as effective protection mechanisms, asserting ownership and deterring unauthorized use. Traditional deepfake detection methods often rely on passive techniques that struggle with sophisticated manipulations. In contrast, diffusion models enhance detection accuracy by allowing for the effective learning of features, enabling the embedding of imperceptible and robust watermarks. We analyze the strengths and challenges of watermark techniques related to diffusion models, focusing on their robustness and application in watermark generation. By exploring the integration of advanced diffusion models and watermarking security, we aim to advance the discourse on preserving watermark robustness against evolving forgery threats. It emphasizes the critical importance of developing innovative solutions to protect digital content and ensure the preservation of ownership rights in the era of generative AI.
>
---
#### [new 045] WaveGuard: Robust Deepfake Detection and Source Tracing via Dual-Tree Complex Wavelet and Graph Neural Networks
- **分类: cs.CV**

- **简介: 该论文属于Deepfake防御任务，旨在解决伪造内容检测及来源追踪问题。提出WaveGuard框架，结合双树复小波变换在频域嵌入抗攻击水印，利用图神经网络保持视觉一致性，通过注意力机制提升精度，实现在人脸替换/重现场景下优于现有方法的鲁棒性和隐蔽性。**

- **链接: [http://arxiv.org/pdf/2505.08614v1](http://arxiv.org/pdf/2505.08614v1)**

> **作者:** Ziyuan He; Zhiqing Guo; Liejun Wang; Gaobo Yang; Yunfeng Diao; Dan Ma
>
> **备注:** 11 pages, 5 figures, 4 tables
>
> **摘要:** Deepfake technology poses increasing risks such as privacy invasion and identity theft. To address these threats, we propose WaveGuard, a proactive watermarking framework that enhances robustness and imperceptibility via frequency-domain embedding and graph-based structural consistency. Specifically, we embed watermarks into high-frequency sub-bands using Dual-Tree Complex Wavelet Transform (DT-CWT) and employ a Structural Consistency Graph Neural Network (SC-GNN) to preserve visual quality. We also design an attention module to refine embedding precision. Experimental results on face swap and reenactment tasks demonstrate that WaveGuard outperforms state-of-the-art methods in both robustness and visual quality. Code is available at https://github.com/vpsg-research/WaveGuard.
>
---
#### [new 046] FAD: Frequency Adaptation and Diversion for Cross-domain Few-shot Learning
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于跨域小样本学习任务，解决分布差异下模型泛化问题。针对现有方法忽视频域特征差异的局限，提出频率适应与转移框架（FAD）：将特征分解为低/中/高频段，通过频域重构和分频段卷积适配器实现针对性调整，在Meta-Dataset上验证了跨域泛化性能提升。**

- **链接: [http://arxiv.org/pdf/2505.08349v1](http://arxiv.org/pdf/2505.08349v1)**

> **作者:** Ruixiao Shi; Fu Feng; Yucheng Xie; Jing Wang; Xin Geng
>
> **摘要:** Cross-domain few-shot learning (CD-FSL) requires models to generalize from limited labeled samples under significant distribution shifts. While recent methods enhance adaptability through lightweight task-specific modules, they operate solely in the spatial domain and overlook frequency-specific variations that are often critical for robust transfer. We observe that spatially similar images across domains can differ substantially in their spectral representations, with low and high frequencies capturing complementary semantic information at coarse and fine levels. This indicates that uniform spatial adaptation may overlook these spectral distinctions, thus constraining generalization. To address this, we introduce Frequency Adaptation and Diversion (FAD), a frequency-aware framework that explicitly models and modulates spectral components. At its core is the Frequency Diversion Adapter, which transforms intermediate features into the frequency domain using the discrete Fourier transform (DFT), partitions them into low, mid, and high-frequency bands via radial masks, and reconstructs each band using inverse DFT (IDFT). Each frequency band is then adapted using a dedicated convolutional branch with a kernel size tailored to its spectral scale, enabling targeted and disentangled adaptation across frequencies. Extensive experiments on the Meta-Dataset benchmark demonstrate that FAD consistently outperforms state-of-the-art methods on both seen and unseen domains, validating the utility of frequency-domain representations and band-wise adaptation for improving generalization in CD-FSL.
>
---
#### [new 047] DFA-CON: A Contrastive Learning Approach for Detecting Copyright Infringement in DeepFake Art
- **分类: cs.CV; cs.AI; cs.LG**

- **简介: 该论文属于AI生成内容检测任务，旨在解决生成模型可能侵犯版权的问题。通过提出DFA-CON对比学习框架，区分原创作品与伪造的AI生成艺术，在多种攻击类型（如修复、风格迁移）中实现高效检测，性能优于现有模型。**

- **链接: [http://arxiv.org/pdf/2505.08552v1](http://arxiv.org/pdf/2505.08552v1)**

> **作者:** Haroon Wahab; Hassan Ugail; Irfan Mehmood
>
> **摘要:** Recent proliferation of generative AI tools for visual content creation-particularly in the context of visual artworks-has raised serious concerns about copyright infringement and forgery. The large-scale datasets used to train these models often contain a mixture of copyrighted and non-copyrighted artworks. Given the tendency of generative models to memorize training patterns, they are susceptible to varying degrees of copyright violation. Building on the recently proposed DeepfakeArt Challenge benchmark, this work introduces DFA-CON, a contrastive learning framework designed to detect copyright-infringing or forged AI-generated art. DFA-CON learns a discriminative representation space, posing affinity among original artworks and their forged counterparts within a contrastive learning framework. The model is trained across multiple attack types, including inpainting, style transfer, adversarial perturbation, and cutmix. Evaluation results demonstrate robust detection performance across most attack types, outperforming recent pretrained foundation models. Code and model checkpoints will be released publicly upon acceptance.
>
---
#### [new 048] MilChat: Introducing Chain of Thought Reasoning and GRPO to a Multimodal Small Language Model for Remote Sensing
- **分类: cs.CV**

- **简介: 该论文提出轻量级多模态模型MilChat，针对遥感图像分析任务，解决大模型在军事领域效率低、适应性差的问题。通过构建MilData数据集，结合思维链推理和GRPO强化学习优化2B参数模型，提升对军事设施检测的准确性与解释性，在开放标注任务中超越通用模型。**

- **链接: [http://arxiv.org/pdf/2505.07984v1](http://arxiv.org/pdf/2505.07984v1)**

> **作者:** Aybora Koksal; A. Aydin Alatan
>
> **备注:** Submitted to JSTARS on April 2, 2025. Code and dataset will be available upon acceptance
>
> **摘要:** Remarkable capabilities in understanding and generating text-image content have been demonstrated by recent advancements in multimodal large language models (MLLMs). However, their effectiveness in specialized domains-particularly those requiring resource-efficient and domain-specific adaptations-has remained limited. In this work, a lightweight multimodal language model termed MilChat is introduced, specifically adapted to analyze remote sensing imagery in secluded areas, including challenging missile launch sites. A new dataset, MilData, was compiled by verifying hundreds of aerial images through expert review, and subtle military installations were highlighted via detailed captions. Supervised fine-tuning on a 2B-parameter open-source MLLM with chain-of-thought (CoT) reasoning annotations was performed, enabling more accurate and interpretable explanations. Additionally, Group Relative Policy Optimization (GRPO) was leveraged to enhance the model's ability to detect critical domain-specific cues-such as defensive layouts and key military structures-while minimizing false positives on civilian scenes. Through empirical evaluations, it has been shown that MilChat significantly outperforms both larger, general-purpose multimodal models and existing remote sensing-adapted approaches on open-ended captioning and classification metrics. Over 80% recall and 98% precision were achieved on the newly proposed MilData benchmark, underscoring the potency of targeted fine-tuning and reinforcement learning in specialized real-world applications.
>
---
#### [new 049] Monocular Depth Guided Occlusion-Aware Disparity Refinement via Semi-supervised Learning in Laparoscopic Images
- **分类: cs.CV**

- **简介: 该论文属于立体视觉中的视差估计任务，针对腹腔镜图像中遮挡和标注数据稀缺的问题，提出DGORNet模型。通过单目深度信息引导视差优化，结合位置嵌入模块增强空间感知，并设计光流差异损失利用未标注视频数据提升动态场景鲁棒性。实验表明该方法在遮挡/弱纹理区域优于现有技术。**

- **链接: [http://arxiv.org/pdf/2505.08178v1](http://arxiv.org/pdf/2505.08178v1)**

> **作者:** Ziteng Liu; Dongdong He; Chenghong Zhang; Wenpeng Gao; Yili Fu
>
> **摘要:** Occlusion and the scarcity of labeled surgical data are significant challenges in disparity estimation for stereo laparoscopic images. To address these issues, this study proposes a Depth Guided Occlusion-Aware Disparity Refinement Network (DGORNet), which refines disparity maps by leveraging monocular depth information unaffected by occlusion. A Position Embedding (PE) module is introduced to provide explicit spatial context, enhancing the network's ability to localize and refine features. Furthermore, we introduce an Optical Flow Difference Loss (OFDLoss) for unlabeled data, leveraging temporal continuity across video frames to improve robustness in dynamic surgical scenes. Experiments on the SCARED dataset demonstrate that DGORNet outperforms state-of-the-art methods in terms of End-Point Error (EPE) and Root Mean Squared Error (RMSE), particularly in occlusion and texture-less regions. Ablation studies confirm the contributions of the Position Embedding and Optical Flow Difference Loss, highlighting their roles in improving spatial and temporal consistency. These results underscore DGORNet's effectiveness in enhancing disparity estimation for laparoscopic surgery, offering a practical solution to challenges in disparity estimation and data limitations.
>
---
#### [new 050] MESSI: A Multi-Elevation Semantic Segmentation Image Dataset of an Urban Environment
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出MESSI多高度城市语义分割数据集，解决无人机在密集城市场景中因视角和高度变化导致语义分割性能不稳定的问题。通过采集2525张多海拔、多区域的无人机图像，提供位置、相机参数等标注数据，并验证了多种神经网络模型性能，旨在为三维飞行环境下的语义分割及导航任务建立公共基准。**

- **链接: [http://arxiv.org/pdf/2505.08589v1](http://arxiv.org/pdf/2505.08589v1)**

> **作者:** Barak Pinkovich; Boaz Matalon; Ehud Rivlin; Hector Rotstein
>
> **摘要:** This paper presents a Multi-Elevation Semantic Segmentation Image (MESSI) dataset comprising 2525 images taken by a drone flying over dense urban environments. MESSI is unique in two main features. First, it contains images from various altitudes, allowing us to investigate the effect of depth on semantic segmentation. Second, it includes images taken from several different urban regions (at different altitudes). This is important since the variety covers the visual richness captured by a drone's 3D flight, performing horizontal and vertical maneuvers. MESSI contains images annotated with location, orientation, and the camera's intrinsic parameters and can be used to train a deep neural network for semantic segmentation or other applications of interest (e.g., localization, navigation, and tracking). This paper describes the dataset and provides annotation details. It also explains how semantic segmentation was performed using several neural network models and shows several relevant statistics. MESSI will be published in the public domain to serve as an evaluation benchmark for semantic segmentation using images captured by a drone or similar vehicle flying over a dense urban environment.
>
---
#### [new 051] STORYANCHORS: Generating Consistent Multi-Scene Story Frames for Long-Form Narratives
- **分类: cs.CV; cs.AI**

- **简介: 该论文属于多场景故事生成任务，旨在解决长叙事中时间一致性与场景多样性不足的问题。提出StoryAnchors框架，通过双向生成器融合上下文保持角色/场景连贯性，结合多事件标注与渐进训练增强叙事丰富度，支持可编辑扩展的序列生成，实验证明其在一致性、多样性方面优于现有模型，达到GPT-4o的叙事水平。**

- **链接: [http://arxiv.org/pdf/2505.08350v1](http://arxiv.org/pdf/2505.08350v1)**

> **作者:** Bo Wang; Haoyang Huang; Zhiyin Lu; Fengyuan Liu; Guoqing Ma; Jianlong Yuan; Yuan Zhang; Nan Duan
>
> **摘要:** This paper introduces StoryAnchors, a unified framework for generating high-quality, multi-scene story frames with strong temporal consistency. The framework employs a bidirectional story generator that integrates both past and future contexts to ensure temporal consistency, character continuity, and smooth scene transitions throughout the narrative. Specific conditions are introduced to distinguish story frame generation from standard video synthesis, facilitating greater scene diversity and enhancing narrative richness. To further improve generation quality, StoryAnchors integrates Multi-Event Story Frame Labeling and Progressive Story Frame Training, enabling the model to capture both overarching narrative flow and event-level dynamics. This approach supports the creation of editable and expandable story frames, allowing for manual modifications and the generation of longer, more complex sequences. Extensive experiments show that StoryAnchors outperforms existing open-source models in key areas such as consistency, narrative coherence, and scene diversity. Its performance in narrative consistency and story richness is also on par with GPT-4o. Ultimately, StoryAnchors pushes the boundaries of story-driven frame generation, offering a scalable, flexible, and highly editable foundation for future research.
>
---
#### [new 052] MoKD: Multi-Task Optimization for Knowledge Distillation
- **分类: cs.CV**

- **简介: 该论文属于知识蒸馏领域，解决任务目标与教师指导间的梯度冲突及不平衡问题。提出MoKD方法，将知识蒸馏转化为多目标优化问题，平衡梯度冲突与主导，并设计子空间学习框架提升跨模型知识迁移效果。实验验证其在图像分类和检测任务中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.08170v1](http://arxiv.org/pdf/2505.08170v1)**

> **作者:** Zeeshan Hayder; Ali Cheraghian; Lars Petersson; Mehrtash Harandi
>
> **摘要:** Compact models can be effectively trained through Knowledge Distillation (KD), a technique that transfers knowledge from larger, high-performing teacher models. Two key challenges in Knowledge Distillation (KD) are: 1) balancing learning from the teacher's guidance and the task objective, and 2) handling the disparity in knowledge representation between teacher and student models. To address these, we propose Multi-Task Optimization for Knowledge Distillation (MoKD). MoKD tackles two main gradient issues: a) Gradient Conflicts, where task-specific and distillation gradients are misaligned, and b) Gradient Dominance, where one objective's gradient dominates, causing imbalance. MoKD reformulates KD as a multi-objective optimization problem, enabling better balance between objectives. Additionally, it introduces a subspace learning framework to project feature representations into a high-dimensional space, improving knowledge transfer. Our MoKD is demonstrated to outperform existing methods through extensive experiments on image classification using the ImageNet-1K dataset and object detection using the COCO dataset, achieving state-of-the-art performance with greater efficiency. To the best of our knowledge, MoKD models also achieve state-of-the-art performance compared to models trained from scratch.
>
---
#### [new 053] Empowering Vision Transformers with Multi-Scale Causal Intervention for Long-Tailed Image Classification
- **分类: cs.CV**

- **简介: 该论文研究长尾图像分类任务，解决视觉变换器（ViT）在因果推理方法中因全局特征导致尾部细粒度类别分类困难的问题。提出TSCNet模型，通过两阶段多尺度因果干预：HCRL阶段解耦背景与对象特征增强因果关联，CLBC阶段校准决策边界消除数据分布偏差，实验表明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.08173v1](http://arxiv.org/pdf/2505.08173v1)**

> **作者:** Xiaoshuo Yan; Zhaochuan Li; Lei Meng; Zhuang Qi; Wei Wu; Zixuan Li; Xiangxu Meng
>
> **摘要:** Causal inference has emerged as a promising approach to mitigate long-tail classification by handling the biases introduced by class imbalance. However, along with the change of advanced backbone models from Convolutional Neural Networks (CNNs) to Visual Transformers (ViT), existing causal models may not achieve an expected performance gain. This paper investigates the influence of existing causal models on CNNs and ViT variants, highlighting that ViT's global feature representation makes it hard for causal methods to model associations between fine-grained features and predictions, which leads to difficulties in classifying tail classes with similar visual appearance. To address these issues, this paper proposes TSCNet, a two-stage causal modeling method to discover fine-grained causal associations through multi-scale causal interventions. Specifically, in the hierarchical causal representation learning stage (HCRL), it decouples the background and objects, applying backdoor interventions at both the patch and feature level to prevent model from using class-irrelevant areas to infer labels which enhances fine-grained causal representation. In the counterfactual logits bias calibration stage (CLBC), it refines the optimization of model's decision boundary by adaptive constructing counterfactual balanced data distribution to remove the spurious associations in the logits caused by data distribution. Extensive experiments conducted on various long-tail benchmarks demonstrate that the proposed TSCNet can eliminate multiple biases introduced by data imbalance, which outperforms existing methods.
>
---
#### [new 054] IrrMap: A Large-Scale Comprehensive Dataset for Irrigation Method Mapping
- **分类: cs.CV**

- **简介: 该论文属于农业遥感与灌溉分类任务，旨在解决缺乏大规模灌溉方法数据集的问题。提出了首个覆盖美国西部（2013-2023年）的百万级多模态数据集IrrMap，整合卫星影像、作物类型等数据，提供标准化ML训练框架及扩展工具，分析灌溉分布规律，并开源数据与模型以促进农业地理分析研究。**

- **链接: [http://arxiv.org/pdf/2505.08273v1](http://arxiv.org/pdf/2505.08273v1)**

> **作者:** Nibir Chandra Mandal; Oishee Bintey Hoque; Abhijin Adiga; Samarth Swarup; Mandy Wilson; Lu Feng; Yangfeng Ji; Miaomiao Zhang; Geoffrey Fox; Madhav Marathe
>
> **摘要:** We introduce IrrMap, the first large-scale dataset (1.1 million patches) for irrigation method mapping across regions. IrrMap consists of multi-resolution satellite imagery from LandSat and Sentinel, along with key auxiliary data such as crop type, land use, and vegetation indices. The dataset spans 1,687,899 farms and 14,117,330 acres across multiple western U.S. states from 2013 to 2023, providing a rich and diverse foundation for irrigation analysis and ensuring geospatial alignment and quality control. The dataset is ML-ready, with standardized 224x224 GeoTIFF patches, the multiple input modalities, carefully chosen train-test-split data, and accompanying dataloaders for seamless deep learning model training andbenchmarking in irrigation mapping. The dataset is also accompanied by a complete pipeline for dataset generation, enabling researchers to extend IrrMap to new regions for irrigation data collection or adapt it with minimal effort for other similar applications in agricultural and geospatial analysis. We also analyze the irrigation method distribution across crop groups, spatial irrigation patterns (using Shannon diversity indices), and irrigated area variations for both LandSat and Sentinel, providing insights into regional and resolution-based differences. To promote further exploration, we openly release IrrMap, along with the derived datasets, benchmark models, and pipeline code, through a GitHub repository: https://github.com/Nibir088/IrrMap and Data repository: https://huggingface.co/Nibir/IrrMap, providing comprehensive documentation and implementation details.
>
---
#### [new 055] The RaspGrade Dataset: Towards Automatic Raspberry Ripeness Grading with Deep Learning
- **分类: cs.CV**

- **简介: 该论文研究基于深度学习的树莓成熟度自动分级，属于实例分割与多类分类任务。针对工业流水线环境下果实颜色相似、遮挡导致的实时分级难题，构建了RaspGrade数据集并进行实验，发现部分成熟度等级易区分而部分存在挑战，最终公开了非侵入式食品质量检测数据集。**

- **链接: [http://arxiv.org/pdf/2505.08537v1](http://arxiv.org/pdf/2505.08537v1)**

> **作者:** Mohamed Lamine Mekhalfi; Paul Chippendale; Fabio Poiesi; Samuele Bonecher; Gilberto Osler; Nicola Zancanella
>
> **摘要:** This research investigates the application of computer vision for rapid, accurate, and non-invasive food quality assessment, focusing on the novel challenge of real-time raspberry grading into five distinct classes within an industrial environment as the fruits move along a conveyor belt. To address this, a dedicated dataset of raspberries, namely RaspGrade, was acquired and meticulously annotated. Instance segmentation experiments revealed that accurate fruit-level masks can be obtained; however, the classification of certain raspberry grades presents challenges due to color similarities and occlusion, while others are more readily distinguishable based on color. The acquired and annotated RaspGrade dataset is accessible on HuggingFace at: https://huggingface.co/datasets/FBK-TeV/RaspGrade.
>
---
#### [new 056] RDD: Robust Feature Detector and Descriptor using Deformable Transformer
- **分类: cs.CV**

- **简介: 该论文属于计算机视觉中的特征检测与描述任务，旨在解决大视角/尺度变化下特征匹配的鲁棒性问题。提出RDD模型，利用可变形Transformer的注意力机制捕捉全局上下文和几何不变性，通过聚焦关键位置降低搜索复杂度。结合Air-to-Ground与MegaDepth数据集训练，并构建新评测基准，在稀疏/半稠密匹配任务中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.08013v1](http://arxiv.org/pdf/2505.08013v1)**

> **作者:** Gonglin Chen; Tianwen Fu; Haiwei Chen; Wenbin Teng; Hanyuan Xiao; Yajie Zhao
>
> **摘要:** As a core step in structure-from-motion and SLAM, robust feature detection and description under challenging scenarios such as significant viewpoint changes remain unresolved despite their ubiquity. While recent works have identified the importance of local features in modeling geometric transformations, these methods fail to learn the visual cues present in long-range relationships. We present Robust Deformable Detector (RDD), a novel and robust keypoint detector/descriptor leveraging the deformable transformer, which captures global context and geometric invariance through deformable self-attention mechanisms. Specifically, we observed that deformable attention focuses on key locations, effectively reducing the search space complexity and modeling the geometric invariance. Furthermore, we collected an Air-to-Ground dataset for training in addition to the standard MegaDepth dataset. Our proposed method outperforms all state-of-the-art keypoint detection/description methods in sparse matching tasks and is also capable of semi-dense matching. To ensure comprehensive evaluation, we introduce two challenging benchmarks: one emphasizing large viewpoint and scale variations, and the other being an Air-to-Ground benchmark -- an evaluation setting that has recently gaining popularity for 3D reconstruction across different altitudes.
>
---
#### [new 057] ADC-GS: Anchor-Driven Deformable and Compressed Gaussian Splatting for Dynamic Scene Reconstruction
- **分类: cs.CV**

- **简介: 该论文提出ADC-GS方法，针对动态场景重建任务，解决传统4D高斯泼溅方法中相邻高斯基元冗余导致的效率低下问题。通过锚点驱动的规范空间结构、分层运动捕捉和速率-失真优化，实现高效存储与实时渲染，速度提升3-8倍，保持渲染质量。**

- **链接: [http://arxiv.org/pdf/2505.08196v1](http://arxiv.org/pdf/2505.08196v1)**

> **作者:** He Huang; Qi Yang; Mufan Liu; Yiling Xu; Zhu Li
>
> **摘要:** Existing 4D Gaussian Splatting methods rely on per-Gaussian deformation from a canonical space to target frames, which overlooks redundancy among adjacent Gaussian primitives and results in suboptimal performance. To address this limitation, we propose Anchor-Driven Deformable and Compressed Gaussian Splatting (ADC-GS), a compact and efficient representation for dynamic scene reconstruction. Specifically, ADC-GS organizes Gaussian primitives into an anchor-based structure within the canonical space, enhanced by a temporal significance-based anchor refinement strategy. To reduce deformation redundancy, ADC-GS introduces a hierarchical coarse-to-fine pipeline that captures motions at varying granularities. Moreover, a rate-distortion optimization is adopted to achieve an optimal balance between bitrate consumption and representation fidelity. Experimental results demonstrate that ADC-GS outperforms the per-Gaussian deformation approaches in rendering speed by 300%-800% while achieving state-of-the-art storage efficiency without compromising rendering quality. The code is released at https://github.com/H-Huang774/ADC-GS.git.
>
---
#### [new 058] TiMo: Spatiotemporal Foundation Model for Satellite Image Time Series
- **分类: cs.CV**

- **简介: 该论文属于卫星图像时序分析的时空基础模型任务，旨在解决现有模型无法有效捕捉多尺度时空关系的问题。提出层次化视觉Transformer模型TiMo，引入动态时空陀螺注意力机制，构建百万级数据集MillionST进行掩码建模预训练，在森林监测、洪灾检测等下游任务中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.08723v1](http://arxiv.org/pdf/2505.08723v1)**

> **作者:** Xiaolei Qin; Di Wang; Jing Zhang; Fengxiang Wang; Xin Su; Bo Du; Liangpei Zhang
>
> **摘要:** Satellite image time series (SITS) provide continuous observations of the Earth's surface, making them essential for applications such as environmental management and disaster assessment. However, existing spatiotemporal foundation models rely on plain vision transformers, which encode entire temporal sequences without explicitly capturing multiscale spatiotemporal relationships between land objects. This limitation hinders their effectiveness in downstream tasks. To overcome this challenge, we propose TiMo, a novel hierarchical vision transformer foundation model tailored for SITS analysis. At its core, we introduce a spatiotemporal gyroscope attention mechanism that dynamically captures evolving multiscale patterns across both time and space. For pre-training, we curate MillionST, a large-scale dataset of one million images from 100,000 geographic locations, each captured across 10 temporal phases over five years, encompassing diverse geospatial changes and seasonal variations. Leveraging this dataset, we adapt masked image modeling to pre-train TiMo, enabling it to effectively learn and encode generalizable spatiotemporal representations.Extensive experiments across multiple spatiotemporal tasks-including deforestation monitoring, land cover segmentation, crop type classification, and flood detection-demonstrate TiMo's superiority over state-of-the-art methods. Code, model, and dataset will be released at https://github.com/MiliLab/TiMo.
>
---
#### [new 059] Leveraging Multi-Modal Information to Enhance Dataset Distillation
- **分类: cs.CV**

- **简介: 该论文聚焦数据集蒸馏任务，旨在通过多模态信息提升合成数据集质量。针对现有方法仅优化视觉特征的局限，提出结合文本信息（特征拼接与标题匹配）及对象分割掩码，设计四种损失函数增强语义对齐与目标聚焦。实验表明新方法有效提升下游任务性能。**

- **链接: [http://arxiv.org/pdf/2505.08605v1](http://arxiv.org/pdf/2505.08605v1)**

> **作者:** Zhe Li; Hadrien Reynaud; Bernhard Kainz
>
> **备注:** 10 pages
>
> **摘要:** Dataset distillation aims to create a compact and highly representative synthetic dataset that preserves the knowledge of a larger real dataset. While existing methods primarily focus on optimizing visual representations, incorporating additional modalities and refining object-level information can significantly improve the quality of distilled datasets. In this work, we introduce two key enhancements to dataset distillation: caption-guided supervision and object-centric masking. To integrate textual information, we propose two strategies for leveraging caption features: the feature concatenation, where caption embeddings are fused with visual features at the classification stage, and caption matching, which introduces a caption-based alignment loss during training to ensure semantic coherence between real and synthetic data. Additionally, we apply segmentation masks to isolate target objects and remove background distractions, introducing two loss functions designed for object-centric learning: masked feature alignment loss and masked gradient matching loss. Comprehensive evaluations demonstrate that integrating caption-based guidance and object-centric masking enhances dataset distillation, leading to synthetic datasets that achieve superior performance on downstream tasks.
>
---
#### [new 060] Object detection in adverse weather conditions for autonomous vehicles using Instruct Pix2Pix
- **分类: cs.CV; cs.AI; I.2.6; I.2.10; I.4.8; I.5.1**

- **简介: 该论文属于自动驾驶中的目标检测任务，旨在解决恶劣天气下模型性能下降问题。研究利用Instruct Pix2Pix生成天气增强数据集，提升Faster R-CNN和YOLOv10的鲁棒性。通过CARLA仿真和真实数据集（BDD100K、ACDC）验证方法有效性，核心贡献包括量化天气对检测的影响差异及提出针对性数据增强策略，为自动驾驶感知系统可靠性改进奠定基础。**

- **链接: [http://arxiv.org/pdf/2505.08228v1](http://arxiv.org/pdf/2505.08228v1)**

> **作者:** Unai Gurbindo; Axel Brando; Jaume Abella; Caroline König
>
> **备注:** 8 pages, 5 figures. Accepted at the International Joint Conference on Neural Networks (IJCNN) 2025 (to appear)
>
> **摘要:** Enhancing the robustness of object detection systems under adverse weather conditions is crucial for the advancement of autonomous driving technology. This study presents a novel approach leveraging the diffusion model Instruct Pix2Pix to develop prompting methodologies that generate realistic datasets with weather-based augmentations aiming to mitigate the impact of adverse weather on the perception capabilities of state-of-the-art object detection models, including Faster R-CNN and YOLOv10. Experiments were conducted in two environments, in the CARLA simulator where an initial evaluation of the proposed data augmentation was provided, and then on the real-world image data sets BDD100K and ACDC demonstrating the effectiveness of the approach in real environments. The key contributions of this work are twofold: (1) identifying and quantifying the performance gap in object detection models under challenging weather conditions, and (2) demonstrating how tailored data augmentation strategies can significantly enhance the robustness of these models. This research establishes a solid foundation for improving the reliability of perception systems in demanding environmental scenarios, and provides a pathway for future advancements in autonomous driving.
>
---
#### [new 061] DHECA-SuperGaze: Dual Head-Eye Cross-Attention and Super-Resolution for Unconstrained Gaze Estimation
- **分类: cs.CV**

- **简介: 该论文研究无约束视线估计任务，解决现实场景中图像分辨率低、头眼交互建模不足的问题。提出DHECA-SuperGaze方法，结合超分辨率重建与双头眼交叉注意力模块，优化特征交互，并修正Gaze360数据集标注错误。实验显示其在静态/动态场景下均超越现有方法，泛化能力显著提升。**

- **链接: [http://arxiv.org/pdf/2505.08426v1](http://arxiv.org/pdf/2505.08426v1)**

> **作者:** Franko Šikić; Donik Vršnak; Sven Lončarić
>
> **摘要:** Unconstrained gaze estimation is the process of determining where a subject is directing their visual attention in uncontrolled environments. Gaze estimation systems are important for a myriad of tasks such as driver distraction monitoring, exam proctoring, accessibility features in modern software, etc. However, these systems face challenges in real-world scenarios, partially due to the low resolution of in-the-wild images and partially due to insufficient modeling of head-eye interactions in current state-of-the-art (SOTA) methods. This paper introduces DHECA-SuperGaze, a deep learning-based method that advances gaze prediction through super-resolution (SR) and a dual head-eye cross-attention (DHECA) module. Our dual-branch convolutional backbone processes eye and multiscale SR head images, while the proposed DHECA module enables bidirectional feature refinement between the extracted visual features through cross-attention mechanisms. Furthermore, we identified critical annotation errors in one of the most diverse and widely used gaze estimation datasets, Gaze360, and rectified the mislabeled data. Performance evaluation on Gaze360 and GFIE datasets demonstrates superior within-dataset performance of the proposed method, reducing angular error (AE) by 0.48{\deg} (Gaze360) and 2.95{\deg} (GFIE) in static configurations, and 0.59{\deg} (Gaze360) and 3.00{\deg} (GFIE) in temporal settings compared to prior SOTA methods. Cross-dataset testing shows improvements in AE of more than 1.53{\deg} (Gaze360) and 3.99{\deg} (GFIE) in both static and temporal settings, validating the robust generalization properties of our approach.
>
---
#### [new 062] Towards Autonomous UAV Visual Object Search in City Space: Benchmark and Agentic Methodology
- **分类: cs.CV; cs.AI**

- **简介: 该论文研究无人机城市视觉目标搜索（AVOS），解决复杂环境中冗余处理、相似物区分及探索-利用平衡问题。提出首个基准数据集CityAVOS（含2420任务）和PRPSearcher方法，利用多模态大模型构建三层认知地图，通过去噪和自适应规划提升搜索效率，实验指标显著优于基线。**

- **链接: [http://arxiv.org/pdf/2505.08765v1](http://arxiv.org/pdf/2505.08765v1)**

> **作者:** Yatai Ji; Zhengqiu Zhu; Yong Zhao; Beidan Liu; Chen Gao; Yihao Zhao; Sihang Qiu; Yue Hu; Quanjun Yin; Yong Li
>
> **摘要:** Aerial Visual Object Search (AVOS) tasks in urban environments require Unmanned Aerial Vehicles (UAVs) to autonomously search for and identify target objects using visual and textual cues without external guidance. Existing approaches struggle in complex urban environments due to redundant semantic processing, similar object distinction, and the exploration-exploitation dilemma. To bridge this gap and support the AVOS task, we introduce CityAVOS, the first benchmark dataset for autonomous search of common urban objects. This dataset comprises 2,420 tasks across six object categories with varying difficulty levels, enabling comprehensive evaluation of UAV agents' search capabilities. To solve the AVOS tasks, we also propose PRPSearcher (Perception-Reasoning-Planning Searcher), a novel agentic method powered by multi-modal large language models (MLLMs) that mimics human three-tier cognition. Specifically, PRPSearcher constructs three specialized maps: an object-centric dynamic semantic map enhancing spatial perception, a 3D cognitive map based on semantic attraction values for target reasoning, and a 3D uncertainty map for balanced exploration-exploitation search. Also, our approach incorporates a denoising mechanism to mitigate interference from similar objects and utilizes an Inspiration Promote Thought (IPT) prompting mechanism for adaptive action planning. Experimental results on CityAVOS demonstrate that PRPSearcher surpasses existing baselines in both success rate and search efficiency (on average: +37.69% SR, +28.96% SPL, -30.69% MSS, and -46.40% NE). While promising, the performance gap compared to humans highlights the need for better semantic reasoning and spatial exploration capabilities in AVOS tasks. This work establishes a foundation for future advances in embodied target search. Dataset and source code are available at https://anonymous.4open.science/r/CityAVOS-3DF8.
>
---
#### [new 063] DLO-Splatting: Tracking Deformable Linear Objects Using 3D Gaussian Splatting
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D物体跟踪任务，旨在解决仅视觉方法在复杂形变下（如打结）对可变形线性物体（DLO）的3D形状估计问题。提出DLO-Splatting算法，结合动力学模型预测形状，并通过3D高斯溅射的渲染损失优化多视角图像与夹爪数据，实现预测迭代修正。**

- **链接: [http://arxiv.org/pdf/2505.08644v1](http://arxiv.org/pdf/2505.08644v1)**

> **作者:** Holly Dinkel; Marcel Büsching; Alberta Longhini; Brian Coltin; Trey Smith; Danica Kragic; Mårten Björkman; Timothy Bretl
>
> **备注:** 5 pages, 2 figures, presented at the 2025 5th Workshop: Reflections on Representations and Manipulating Deformable Objects at the IEEE International Conference on Robotics and Automation. RMDO workshop (https://deformable-workshop.github.io/icra2025/)
>
> **摘要:** This work presents DLO-Splatting, an algorithm for estimating the 3D shape of Deformable Linear Objects (DLOs) from multi-view RGB images and gripper state information through prediction-update filtering. The DLO-Splatting algorithm uses a position-based dynamics model with shape smoothness and rigidity dampening corrections to predict the object shape. Optimization with a 3D Gaussian Splatting-based rendering loss iteratively renders and refines the prediction to align it with the visual observations in the update step. Initial experiments demonstrate promising results in a knot tying scenario, which is challenging for existing vision-only methods.
>
---
#### [new 064] OpenThinkIMG: Learning to Think with Images via Visual Tool Reinforcement Learning
- **分类: cs.CV**

- **简介: 该论文属于工具增强的视觉推理任务，旨在解决大型视觉语言模型(LVLMs)动态调用视觉工具能力不足的问题。通过构建开源框架OpenThinkIMG，整合标准化工具接口和强化学习算法V-ToolRL，使模型能自主优化视觉工具使用策略，在图表推理任务中超越监督学习基线和GPT-4等模型。**

- **链接: [http://arxiv.org/pdf/2505.08617v1](http://arxiv.org/pdf/2505.08617v1)**

> **作者:** Zhaochen Su; Linjie Li; Mingyang Song; Yunzhuo Hao; Zhengyuan Yang; Jun Zhang; Guanjie Chen; Jiawei Gu; Juntao Li; Xiaoye Qu; Yu Cheng
>
> **备注:** Work in progress
>
> **摘要:** While humans can flexibly leverage interactive visual cognition for complex problem-solving, enabling Large Vision-Language Models (LVLMs) to learn similarly adaptive behaviors with visual tools remains challenging. A significant hurdle is the current lack of standardized infrastructure, which hinders integrating diverse tools, generating rich interaction data, and training robust agents effectively. To address these gaps, we introduce OpenThinkIMG, the first open-source, comprehensive end-to-end framework for tool-augmented LVLMs. It features standardized vision tool interfaces, scalable trajectory generation for policy initialization, and a flexible training environment. Furthermore, considering supervised fine-tuning (SFT) on static demonstrations offers limited policy generalization for dynamic tool invocation, we propose a novel reinforcement learning (RL) framework V-ToolRL to train LVLMs to learn adaptive policies for invoking external vision tools. V-ToolRL enables LVLMs to autonomously discover optimal tool-usage strategies by directly optimizing for task success using feedback from tool interactions. We empirically validate V-ToolRL on challenging chart reasoning tasks. Our RL-trained agent, built upon a Qwen2-VL-2B, significantly outperforms its SFT-initialized counterpart (+28.83 points) and surpasses established supervised tool-learning baselines like Taco and CogCom by an average of +12.7 points. Notably, it also surpasses prominent closed-source models like GPT-4.1 by +8.68 accuracy points. We hope OpenThinkIMG can serve as a foundational framework for advancing dynamic, tool-augmented visual reasoning, helping the community develop AI agents that can genuinely "think with images".
>
---
#### [new 065] Reinforcement Learning meets Masked Video Modeling : Trajectory-Guided Adaptive Token Selection
- **分类: cs.CV**

- **简介: 该论文属于视频预训练任务，旨在解决传统掩码视频建模中掩码策略选择困难的问题。提出轨迹感知自适应令牌选择器(TATS)与强化学习联合训练框架，通过动态建模运动轨迹选择关键令牌，在高效掩码的同时保持动作识别性能。**

- **链接: [http://arxiv.org/pdf/2505.08561v1](http://arxiv.org/pdf/2505.08561v1)**

> **作者:** Ayush K. Rai; Kyle Min; Tarun Krishna; Feiyan Hu; Alan F. Smeaton; Noel E. O'Connor
>
> **摘要:** Masked video modeling~(MVM) has emerged as a highly effective pre-training strategy for visual foundation models, whereby the model reconstructs masked spatiotemporal tokens using information from visible tokens. However, a key challenge in such approaches lies in selecting an appropriate masking strategy. Previous studies have explored predefined masking techniques, including random and tube-based masking, as well as approaches that leverage key motion priors, optical flow and semantic cues from externally pre-trained models. In this work, we introduce a novel and generalizable Trajectory-Aware Adaptive Token Sampler (TATS), which models the motion dynamics of tokens and can be seamlessly integrated into the masked autoencoder (MAE) framework to select motion-centric tokens in videos. Additionally, we propose a unified training strategy that enables joint optimization of both MAE and TATS from scratch using Proximal Policy Optimization (PPO). We show that our model allows for aggressive masking without compromising performance on the downstream task of action recognition while also ensuring that the pre-training remains memory efficient. Extensive experiments of the proposed approach across four benchmarks, including Something-Something v2, Kinetics-400, UCF101, and HMDB51, demonstrate the effectiveness, transferability, generalization, and efficiency of our work compared to other state-of-the-art methods.
>
---
#### [new 066] Dynamic Snake Upsampling Operater and Boundary-Skeleton Weighted Loss for Tubular Structure Segmentation
- **分类: cs.CV**

- **简介: 该论文针对管状结构（如血管）分割任务，解决传统上采样无法处理纤细形态和弯曲结构的问题。提出动态蛇形上采样算子，通过自适应步长沿蛇形路径恢复亚像素特征；设计边界-骨架加权损失函数，平衡主体与边界权重。实验表明该方法提升了分割精度与拓扑一致性。**

- **链接: [http://arxiv.org/pdf/2505.08525v1](http://arxiv.org/pdf/2505.08525v1)**

> **作者:** Yiqi Chen; Ganghai Huang; Sheng Zhang; Jianglin Dai
>
> **摘要:** Accurate segmentation of tubular topological structures (e.g., fissures and vasculature) is critical in various fields to guarantee dependable downstream quantitative analysis and modeling. However, in dense prediction tasks such as semantic segmentation and super-resolution, conventional upsampling operators cannot accommodate the slenderness of tubular structures and the curvature of morphology. This paper introduces a dynamic snake upsampling operators and a boundary-skeleton weighted loss tailored for topological tubular structures. Specifically, we design a snake upsampling operators based on an adaptive sampling domain, which dynamically adjusts the sampling stride according to the feature map and selects a set of subpixel sampling points along the serpentine path, enabling more accurate subpixel-level feature recovery for tubular structures. Meanwhile, we propose a skeleton-to-boundary increasing weighted loss that trades off main body and boundary weight allocation based on mask class ratio and distance field, preserving main body overlap while enhancing focus on target topological continuity and boundary alignment precision. Experiments across various domain datasets and backbone networks show that this plug-and-play dynamic snake upsampling operator and boundary-skeleton weighted loss boost both pixel-wise segmentation accuracy and topological consistency of results.
>
---
#### [new 067] SkillFormer: Unified Multi-View Video Understanding for Proficiency Estimation
- **分类: cs.CV**

- **简介: 该论文属于多视角视频理解任务，旨在解决复杂活动中人类技能评估的难题。提出SkillFormer模型，基于TimeSformer框架设计跨视角融合模块（结合多头注意力与自适应校准），通过低秩适配微调少量参数，在EgoExo4D数据集上实现高效多视角技能评估，参数减少4.5倍且训练速度提升3.75倍。**

- **链接: [http://arxiv.org/pdf/2505.08665v1](http://arxiv.org/pdf/2505.08665v1)**

> **作者:** Edoardo Bianchi; Antonio Liotta
>
> **摘要:** Assessing human skill levels in complex activities is a challenging problem with applications in sports, rehabilitation, and training. In this work, we present SkillFormer, a parameter-efficient architecture for unified multi-view proficiency estimation from egocentric and exocentric videos. Building on the TimeSformer backbone, SkillFormer introduces a CrossViewFusion module that fuses view-specific features using multi-head cross-attention, learnable gating, and adaptive self-calibration. We leverage Low-Rank Adaptation to fine-tune only a small subset of parameters, significantly reducing training costs. In fact, when evaluated on the EgoExo4D dataset, SkillFormer achieves state-of-the-art accuracy in multi-view settings while demonstrating remarkable computational efficiency, using 4.5x fewer parameters and requiring 3.75x fewer training epochs than prior baselines. It excels in multiple structured tasks, confirming the value of multi-view integration for fine-grained skill assessment.
>
---
#### [new 068] Few-shot Novel Category Discovery
- **分类: cs.CV**

- **简介: 该论文研究小样本新类别发现任务，旨在解决现有方法依赖转导式学习、限制实际应用的问题。通过引入少量新类别标注数据，提出模型可动态切换已知类识别与未标记类聚类的框架，开发了SHC和UKC算法提升模型推理能力，并在多数据集验证有效性。**

- **链接: [http://arxiv.org/pdf/2505.08260v1](http://arxiv.org/pdf/2505.08260v1)**

> **作者:** Chunming Li; Shidong Wang; Haofeng Zhang
>
> **摘要:** The recently proposed Novel Category Discovery (NCD) adapt paradigm of transductive learning hinders its application in more real-world scenarios. In fact, few labeled data in part of new categories can well alleviate this burden, which coincides with the ease that people can label few of new category data. Therefore, this paper presents a new setting in which a trained agent is able to flexibly switch between the tasks of identifying examples of known (labelled) classes and clustering novel (completely unlabeled) classes as the number of query examples increases by leveraging knowledge learned from only a few (handful) support examples. Drawing inspiration from the discovery of novel categories using prior-based clustering algorithms, we introduce a novel framework that further relaxes its assumptions to the real-world open set level by unifying the concept of model adaptability in few-shot learning. We refer to this setting as Few-Shot Novel Category Discovery (FSNCD) and propose Semi-supervised Hierarchical Clustering (SHC) and Uncertainty-aware K-means Clustering (UKC) to examine the model's reasoning capabilities. Extensive experiments and detailed analysis on five commonly used datasets demonstrate that our methods can achieve leading performance levels across different task settings and scenarios.
>
---
#### [new 069] Unsupervised Raindrop Removal from a Single Image using Conditional Diffusion Models
- **分类: cs.CV; cs.LG**

- **简介: 该论文属于单图像雨滴去除任务，旨在解决无监督条件下单一图像中雨滴干扰的消除难题。传统方法依赖GAN进行背景修复，本文提出基于条件扩散模型的图像修复技术，通过扩散模型实现更优的雨滴区域检测与背景重建，取代了传统生成对抗网络方案。**

- **链接: [http://arxiv.org/pdf/2505.08190v1](http://arxiv.org/pdf/2505.08190v1)**

> **作者:** Lhuqita Fazry; Valentino Vito
>
> **摘要:** Raindrop removal is a challenging task in image processing. Removing raindrops while relying solely on a single image further increases the difficulty of the task. Common approaches include the detection of raindrop regions in the image, followed by performing a background restoration process conditioned on those regions. While various methods can be applied for the detection step, the most common architecture used for background restoration is the Generative Adversarial Network (GAN). Recent advances in the use of diffusion models have led to state-of-the-art image inpainting techniques. In this paper, we introduce a novel technique for raindrop removal from a single image using diffusion-based image inpainting.
>
---
#### [new 070] Fréchet Power-Scenario Distance: A Metric for Evaluating Generative AI Models across Multiple Time-Scales in Smart Grids
- **分类: cs.LG; cs.AI; cs.CV; eess.SP**

- **简介: 该论文属于生成模型评估任务，旨在解决智能电网中生成AI模型产生的合成数据质量评估难题。针对传统欧氏距离无法衡量群体数据差异的缺陷，提出基于特征空间Fréchet距离的新指标，从分布层面评估生成质量。实验验证了该方法在多时间尺度和模型中的优越性。**

- **链接: [http://arxiv.org/pdf/2505.08082v1](http://arxiv.org/pdf/2505.08082v1)**

> **作者:** Yuting Cai; Shaohuai Liu; Chao Tian; Le Xie
>
> **摘要:** Generative artificial intelligence (AI) models in smart grids have advanced significantly in recent years due to their ability to generate large amounts of synthetic data, which would otherwise be difficult to obtain in the real world due to confidentiality constraints. A key challenge in utilizing such synthetic data is how to assess the data quality produced from such generative models. Traditional Euclidean distance-based metrics only reflect pair-wise relations between two individual samples, and could fail in evaluating quality differences between groups of synthetic datasets. In this work, we propose a novel metric based on the Fr\'{e}chet Distance (FD) estimated between two datasets in a learned feature space. The proposed method evaluates the quality of generation from a distributional perspective. Empirical results demonstrate the superiority of the proposed metric across timescales and models, enhancing the reliability of data-driven decision-making in smart grid operations.
>
---
#### [new 071] Decoding Neighborhood Environments with Large Language Models
- **分类: cs.AI; cs.CV**

- **简介: 该论文研究利用大语言模型（LLMs）自动化评估邻里环境的任务，解决传统方法（如GIS）资源消耗大、机器学习标注成本高的问题。通过训练YOLOv11模型（99.13%准确率）检测六类环境要素，并测试四种LLMs的识别能力，结合多数投票策略使LLMs准确率达88%，证明其无需训练即可高效分析环境指标的潜力。**

- **链接: [http://arxiv.org/pdf/2505.08163v1](http://arxiv.org/pdf/2505.08163v1)**

> **作者:** Andrew Cart; Shaohu Zhang; Melanie Escue; Xugui Zhou; Haitao Zhao; Prashanth BusiReddyGari; Beiyu Lin; Shuang Li
>
> **备注:** 8 pages
>
> **摘要:** Neighborhood environments include physical and environmental conditions such as housing quality, roads, and sidewalks, which significantly influence human health and well-being. Traditional methods for assessing these environments, including field surveys and geographic information systems (GIS), are resource-intensive and challenging to evaluate neighborhood environments at scale. Although machine learning offers potential for automated analysis, the laborious process of labeling training data and the lack of accessible models hinder scalability. This study explores the feasibility of large language models (LLMs) such as ChatGPT and Gemini as tools for decoding neighborhood environments (e.g., sidewalk and powerline) at scale. We train a robust YOLOv11-based model, which achieves an average accuracy of 99.13% in detecting six environmental indicators, including streetlight, sidewalk, powerline, apartment, single-lane road, and multilane road. We then evaluate four LLMs, including ChatGPT, Gemini, Claude, and Grok, to assess their feasibility, robustness, and limitations in identifying these indicators, with a focus on the impact of prompting strategies and fine-tuning. We apply majority voting with the top three LLMs to achieve over 88% accuracy, which demonstrates LLMs could be a useful tool to decode the neighborhood environment without any training effort.
>
---
#### [new 072] Visually Guided Decoding: Gradient-Free Hard Prompt Inversion with Language Models
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究文本到图像模型的提示生成任务，解决现有方法生成不连贯、可解释性差的问题。提出VGD方法，通过无梯度优化结合大语言模型生成能力与CLIP视觉对齐，无需训练即可生成语义一致且易理解的提示词，提升交互可控性。**

- **链接: [http://arxiv.org/pdf/2505.08622v1](http://arxiv.org/pdf/2505.08622v1)**

> **作者:** Donghoon Kim; Minji Bae; Kyuhong Shim; Byonghyo Shim
>
> **备注:** ICLR 2025
>
> **摘要:** Text-to-image generative models like DALL-E and Stable Diffusion have revolutionized visual content creation across various applications, including advertising, personalized media, and design prototyping. However, crafting effective textual prompts to guide these models remains challenging, often requiring extensive trial and error. Existing prompt inversion approaches, such as soft and hard prompt techniques, are not so effective due to the limited interpretability and incoherent prompt generation. To address these issues, we propose Visually Guided Decoding (VGD), a gradient-free approach that leverages large language models (LLMs) and CLIP-based guidance to generate coherent and semantically aligned prompts. In essence, VGD utilizes the robust text generation capabilities of LLMs to produce human-readable prompts. Further, by employing CLIP scores to ensure alignment with user-specified visual concepts, VGD enhances the interpretability, generalization, and flexibility of prompt generation without the need for additional training. Our experiments demonstrate that VGD outperforms existing prompt inversion techniques in generating understandable and contextually relevant prompts, facilitating more intuitive and controllable interactions with text-to-image models.
>
---
#### [new 073] Computationally Efficient Diffusion Models in Medical Imaging: A Comprehensive Review
- **分类: eess.IV; cs.AI; cs.CV**

- **简介: 该论文为综述类研究，聚焦扩散模型在医学影像中的计算效率问题。针对训练及生成过程的高计算成本，系统分析了DDPM、LDM、WDM三类模型框架，探讨其在自然/医学影像中填补计算复杂度缺口的作用，同时指出模型局限性与未来研究方向。**

- **链接: [http://arxiv.org/pdf/2505.07866v1](http://arxiv.org/pdf/2505.07866v1)**

> **作者:** Abdullah; Tao Huang; Ickjai Lee; Euijoon Ahn
>
> **备注:** pages 36, 6 figures
>
> **摘要:** The diffusion model has recently emerged as a potent approach in computer vision, demonstrating remarkable performances in the field of generative artificial intelligence. Capable of producing high-quality synthetic images, diffusion models have been successfully applied across a range of applications. However, a significant challenge remains with the high computational cost associated with training and generating these models. This study focuses on the efficiency and inference time of diffusion-based generative models, highlighting their applications in both natural and medical imaging. We present the most recent advances in diffusion models by categorizing them into three key models: the Denoising Diffusion Probabilistic Model (DDPM), the Latent Diffusion Model (LDM), and the Wavelet Diffusion Model (WDM). These models play a crucial role in medical imaging, where producing fast, reliable, and high-quality medical images is essential for accurate analysis of abnormalities and disease diagnosis. We first investigate the general framework of DDPM, LDM, and WDM and discuss the computational complexity gap filled by these models in natural and medical imaging. We then discuss the current limitations of these models as well as the opportunities and future research directions in medical imaging.
>
---
#### [new 074] A portable diagnosis model for Keratoconus using a smartphone
- **分类: eess.IV; cs.CV; cs.LG**

- **简介: 该论文属于医疗诊断任务，旨在解决传统圆锥角膜（KC）检测依赖专业设备的问题。提出智能手机便携方案，通过屏幕显示Placido盘并分析角膜反射，采用两阶段检测：WSVM分类KC阶段（准确率超90%），结合色图可视化病变区域。使用3D打印眼球模型验证，多机型测试并通过统计检验证明特征有效性。**

- **链接: [http://arxiv.org/pdf/2505.08616v1](http://arxiv.org/pdf/2505.08616v1)**

> **作者:** Yifan Li; Myeongjun Kim; Yanjing Jin; Peter Ho; Jo Woon Chong
>
> **摘要:** Keratoconus (KC) is a progressive corneal disorder characterized by localized thinning and protrusion, leading to visual distortion. While Placido disc-based topography remains a standard in clinical diagnostics, its dependence on specialized equipment limits accessibility. In this paper, we propose a portable, smartphone-based diagnostic framework that captures corneal reflections of a Placido disc displayed on a phone screen and applies a two-stage detection pipeline, then validate on 3D-printed emulated eyeball models that simulate normal, moderate, and severe KC stages based on anterior chamber depth (ACD). The first step of the two-stage detection pipeline is classifying different stages of KC with features including height and width of extracted reflections using weighted support vector machine (WSVM). It achieves a maximum accuracy of 92.93%, and maintains over 90% accuracy across multiple smartphone models, including the Galaxy Z Flip 3, iPhone 15 Pro, and iPhone 16 Pro. For the second step, we visualize the KC-affected protrusion regions on the corneas with color maps based on inter-disc distance, that provides an intuitive representation of disease severity and localization. Moreover, we validate the ability of the extracted features to differentiate between KC stages with ANOVA and Omega Squared, with significant p-values (e.g., $p < 10^{-6}$) and large effect sizes ($\\omega^2$ up to 0.8398) among classes.
>
---
#### [new 075] GNCAF: A GNN-based Neighboring Context Aggregation Framework for Tertiary Lymphoid Structures Semantic Segmentation in WSI
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像语义分割任务，旨在端到端分割全切片图像中的三级淋巴结构区域及成熟阶段。针对现有方法依赖后处理、难以有效整合邻域上下文的问题，提出基于图神经网络的GNCAF框架，通过多跳邻域聚合和自注意力机制增强上下文感知，并构建公开数据集验证性能优势（mF1提升22.08%）。**

- **链接: [http://arxiv.org/pdf/2505.08430v1](http://arxiv.org/pdf/2505.08430v1)**

> **作者:** Lei Su
>
> **摘要:** Tertiary lymphoid structures (TLS) are organized clusters of immune cells, whose maturity and area can be quantified in whole slide image (WSI) for various prognostic tasks. Existing methods for assessing these characteristics typically rely on cell proxy tasks and require additional post-processing steps. In this work, We focus on a novel task-TLS Semantic Segmentation (TLS-SS)-which segments both the regions and maturation stages of TLS in WSI in an end-to-end manner. Due to the extensive scale of WSI and patch-based segmentation strategies, TLS-SS necessitates integrating from neighboring patches to guide target patch (target) segmentation. Previous techniques often employ on multi-resolution approaches, constraining the capacity to leverage the broader neighboring context while tend to preserve coarse-grained information. To address this, we propose a GNN-based Neighboring Context Aggregation Framework (GNCAF), which progressively aggregates multi-hop neighboring context from the target and employs a self-attention mechanism to guide the segmentation of the target. GNCAF can be integrated with various segmentation models to enhance their ability to perceive contextual information outside of the patch. We build two TLS-SS datasets, called TCGA-COAD and INHOUSE-PAAD, and make the former (comprising 225 WSIs and 5041 TLSs) publicly available. Experiments on these datasets demonstrate the superiority of GNCAF, achieving a maximum of 22.08% and 26.57% improvement in mF1 and mIoU, respectively. Additionally, we also validate the task scalability of GNCAF on segmentation of lymph node metastases.
>
---
#### [new 076] M3G: Multi-Granular Gesture Generator for Audio-Driven Full-Body Human Motion Synthesis
- **分类: cs.GR; cs.AI; cs.CV; cs.SD; eess.AS; I.3.6**

- **简介: 该论文研究音频驱动全身人体动作生成（含面部、肢体和全局运动），属于虚拟形象合成任务。针对现有方法固定粒度建模导致动作不自然的问题，提出M3G框架：通过多粒度VQ-VAE编码不同时长的动作模式，并设计音频特征提取器预测多粒度动作令牌，最终重建自然动作，实验证明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.08293v1](http://arxiv.org/pdf/2505.08293v1)**

> **作者:** Zhizhuo Yin; Yuk Hang Tsui; Pan Hui
>
> **备注:** 9 Pages, 4 figures, submitted to NIPS 2025
>
> **摘要:** Generating full-body human gestures encompassing face, body, hands, and global movements from audio is a valuable yet challenging task in virtual avatar creation. Previous systems focused on tokenizing the human gestures framewisely and predicting the tokens of each frame from the input audio. However, one observation is that the number of frames required for a complete expressive human gesture, defined as granularity, varies among different human gesture patterns. Existing systems fail to model these gesture patterns due to the fixed granularity of their gesture tokens. To solve this problem, we propose a novel framework named Multi-Granular Gesture Generator (M3G) for audio-driven holistic gesture generation. In M3G, we propose a novel Multi-Granular VQ-VAE (MGVQ-VAE) to tokenize motion patterns and reconstruct motion sequences from different temporal granularities. Subsequently, we proposed a multi-granular token predictor that extracts multi-granular information from audio and predicts the corresponding motion tokens. Then M3G reconstructs the human gestures from the predicted tokens using the MGVQ-VAE. Both objective and subjective experiments demonstrate that our proposed M3G framework outperforms the state-of-the-art methods in terms of generating natural and expressive full-body human gestures.
>
---
#### [new 077] An integrated language-vision foundation model for conversational diagnostics and triaging in primary eye care
- **分类: eess.IV; cs.CV**

- **简介: 该论文提出多模态基础模型Meta-EyeFM，整合语言与视觉模型实现眼科疾病诊断。解决现有模型任务单一、交互不便的问题，通过路由机制分配眼底图像至专用视觉模型，使用低秩适配技术提升疾病检测（≥82.2%）、严重度区分（≥89%）和体征识别（≥76%），准确率超越主流模型，接近眼科医生水平，为初级诊疗提供智能决策支持。**

- **链接: [http://arxiv.org/pdf/2505.08414v1](http://arxiv.org/pdf/2505.08414v1)**

> **作者:** Zhi Da Soh; Yang Bai; Kai Yu; Yang Zhou; Xiaofeng Lei; Sahil Thakur; Zann Lee; Lee Ching Linette Phang; Qingsheng Peng; Can Can Xue; Rachel Shujuan Chong; Quan V. Hoang; Lavanya Raghavan; Yih Chung Tham; Charumathi Sabanayagam; Wei-Chi Wu; Ming-Chih Ho; Jiangnan He; Preeti Gupta; Ecosse Lamoureux; Seang Mei Saw; Vinay Nangia; Songhomitra Panda-Jonas; Jie Xu; Ya Xing Wang; Xinxing Xu; Jost B. Jonas; Tien Yin Wong; Rick Siow Mong Goh; Yong Liu; Ching-Yu Cheng
>
> **摘要:** Current deep learning models are mostly task specific and lack a user-friendly interface to operate. We present Meta-EyeFM, a multi-function foundation model that integrates a large language model (LLM) with vision foundation models (VFMs) for ocular disease assessment. Meta-EyeFM leverages a routing mechanism to enable accurate task-specific analysis based on text queries. Using Low Rank Adaptation, we fine-tuned our VFMs to detect ocular and systemic diseases, differentiate ocular disease severity, and identify common ocular signs. The model achieved 100% accuracy in routing fundus images to appropriate VFMs, which achieved $\ge$ 82.2% accuracy in disease detection, $\ge$ 89% in severity differentiation, $\ge$ 76% in sign identification. Meta-EyeFM was 11% to 43% more accurate than Gemini-1.5-flash and ChatGPT-4o LMMs in detecting various eye diseases and comparable to an ophthalmologist. This system offers enhanced usability and diagnostic performance, making it a valuable decision support tool for primary eye care or an online LLM for fundus evaluation.
>
---
#### [new 078] Image-Guided Microstructure Optimization using Diffusion Models: Validated with Li-Mn-rich Cathode Precursors
- **分类: cond-mat.mtrl-sci; cs.CV; cs.LG**

- **简介: 该论文属于AI驱动的材料设计优化任务，旨在解决微结构难以作为可控设计变量的问题。通过结合扩散模型生成SEM图像、图像分析提取形态特征及粒子群优化算法，建立了正/逆向预测合成条件与微结构（如颗粒尺寸、形貌）的闭环框架，并以锂锰氧化物阴极前驱体实验验证了预测与实测结构的高度一致性，实现数据驱动的自主微结构调控。**

- **链接: [http://arxiv.org/pdf/2505.07906v1](http://arxiv.org/pdf/2505.07906v1)**

> **作者:** Geunho Choi; Changhwan Lee; Jieun Kim; Insoo Ye; Keeyoung Jung; Inchul Park
>
> **备注:** 37 pages, 10 figures
>
> **摘要:** Microstructure often dictates materials performance, yet it is rarely treated as an explicit design variable because microstructure is hard to quantify, predict, and optimize. Here, we introduce an image centric, closed-loop framework that makes microstructural morphology into a controllable objective and demonstrate its use case with Li- and Mn-rich layered oxide cathode precursors. This work presents an integrated, AI driven framework for the predictive design and optimization of lithium-ion battery cathode precursor synthesis. This framework integrates a diffusion-based image generation model, a quantitative image analysis pipeline, and a particle swarm optimization (PSO) algorithm. By extracting key morphological descriptors such as texture, sphericity, and median particle size (D50) from SEM images, the platform accurately predicts SEM like morphologies resulting from specific coprecipitation conditions, including reaction time-, solution concentration-, and pH-dependent structural changes. Optimization then pinpoints synthesis parameters that yield user defined target morphologies, as experimentally validated by the close agreement between predicted and synthesized structures. This framework offers a practical strategy for data driven materials design, enabling both forward prediction and inverse design of synthesis conditions and paving the way toward autonomous, image guided microstructure engineering.
>
---
#### [new 079] Judging the Judges: Can Large Vision-Language Models Fairly Evaluate Chart Comprehension and Reasoning?
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究大型视觉语言模型(LVLM)作为自动评估工具在图表理解任务中的可行性，属于模型评估任务。针对现有评估方法成本高、封闭性强的问题，系统测试了13个开源LVLM的评判能力，设计多维度评估标准并分析偏差，发现部分模型能达到GPT-4水平但存在位置偏好等局限性。**

- **链接: [http://arxiv.org/pdf/2505.08468v1](http://arxiv.org/pdf/2505.08468v1)**

> **作者:** Md Tahmid Rahman Laskar; Mohammed Saidul Islam; Ridwan Mahbub; Ahmed Masry; Mizanur Rahman; Amran Bhuiyan; Mir Tafseer Nayeem; Shafiq Joty; Enamul Hoque; Jimmy Huang
>
> **备注:** Accepted at ACL 2025 Industry Track
>
> **摘要:** Charts are ubiquitous as they help people understand and reason with data. Recently, various downstream tasks, such as chart question answering, chart2text, and fact-checking, have emerged. Large Vision-Language Models (LVLMs) show promise in tackling these tasks, but their evaluation is costly and time-consuming, limiting real-world deployment. While using LVLMs as judges to assess the chart comprehension capabilities of other LVLMs could streamline evaluation processes, challenges like proprietary datasets, restricted access to powerful models, and evaluation costs hinder their adoption in industrial settings. To this end, we present a comprehensive evaluation of 13 open-source LVLMs as judges for diverse chart comprehension and reasoning tasks. We design both pairwise and pointwise evaluation tasks covering criteria like factual correctness, informativeness, and relevancy. Additionally, we analyze LVLM judges based on format adherence, positional consistency, length bias, and instruction-following. We focus on cost-effective LVLMs (<10B parameters) suitable for both research and commercial use, following a standardized evaluation protocol and rubric to measure the LVLM judge's accuracy. Experimental results reveal notable variability: while some open LVLM judges achieve GPT-4-level evaluation performance (about 80% agreement with GPT-4 judgments), others struggle (below ~10% agreement). Our findings highlight that state-of-the-art open-source LVLMs can serve as cost-effective automatic evaluators for chart-related tasks, though biases such as positional preference and length bias persist.
>
---
#### [new 080] Aya Vision: Advancing the Frontier of Multilingual Multimodality
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文针对多语言多模态模型构建中的数据稀缺、翻译失真及灾难性遗忘问题，提出合成数据框架和跨模态模型融合技术，开发了Aya Vision系列模型，在保持文本能力的同时提升多模态生成性能，显著优于更大规模的基准模型。属于多模态自然语言处理任务。**

- **链接: [http://arxiv.org/pdf/2505.08751v1](http://arxiv.org/pdf/2505.08751v1)**

> **作者:** Saurabh Dash; Yiyang Nan; John Dang; Arash Ahmadian; Shivalika Singh; Madeline Smith; Bharat Venkitesh; Vlad Shmyhlo; Viraat Aryabumi; Walter Beller-Morales; Jeremy Pekmez; Jason Ozuzu; Pierre Richemond; Acyr Locatelli; Nick Frosst; Phil Blunsom; Aidan Gomez; Ivan Zhang; Marzieh Fadaee; Manoj Govindassamy; Sudip Roy; Matthias Gallé; Beyza Ermis; Ahmet Üstün; Sara Hooker
>
> **摘要:** Building multimodal language models is fundamentally challenging: it requires aligning vision and language modalities, curating high-quality instruction data, and avoiding the degradation of existing text-only capabilities once vision is introduced. These difficulties are further magnified in the multilingual setting, where the need for multimodal data in different languages exacerbates existing data scarcity, machine translation often distorts meaning, and catastrophic forgetting is more pronounced. To address the aforementioned challenges, we introduce novel techniques spanning both data and modeling. First, we develop a synthetic annotation framework that curates high-quality, diverse multilingual multimodal instruction data, enabling Aya Vision models to produce natural, human-preferred responses to multimodal inputs across many languages. Complementing this, we propose a cross-modal model merging technique that mitigates catastrophic forgetting, effectively preserving text-only capabilities while simultaneously enhancing multimodal generative performance. Aya-Vision-8B achieves best-in-class performance compared to strong multimodal models such as Qwen-2.5-VL-7B, Pixtral-12B, and even much larger Llama-3.2-90B-Vision. We further scale this approach with Aya-Vision-32B, which outperforms models more than twice its size, such as Molmo-72B and LLaMA-3.2-90B-Vision. Our work advances multilingual progress on the multi-modal frontier, and provides insights into techniques that effectively bend the need for compute while delivering extremely high performance.
>
---
#### [new 081] CAD-Coder:Text-Guided CAD Files Code Generation
- **分类: cs.GR; cs.CV; cs.LG**

- **简介: 该论文属于文本到代码生成任务，旨在解决传统CAD生成方法缺乏交互可编辑性和几何标注的问题。提出CAD-Coder框架，通过自然语言生成可执行CAD脚本代码，构建包含2.9万样本的数据集，实现可编辑DXF文件及标注信息生成，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.08686v1](http://arxiv.org/pdf/2505.08686v1)**

> **作者:** Changqi He; Shuhan Zhang; Liguo Zhang; Jiajun Miao
>
> **摘要:** Computer-aided design (CAD) is a way to digitally create 2D drawings and 3D models of real-world products. Traditional CAD typically relies on hand-drawing by experts or modifications of existing library files, which doesn't allow for rapid personalization. With the emergence of generative artificial intelligence, convenient and efficient personalized CAD generation has become possible. However, existing generative methods typically produce outputs that lack interactive editability and geometric annotations, limiting their practical applications in manufacturing. To enable interactive generative CAD, we propose CAD-Coder, a framework that transforms natural language instructions into CAD script codes, which can be executed in Python environments to generate human-editable CAD files (.Dxf). To facilitate the generation of editable CAD sketches with annotation information, we construct a comprehensive dataset comprising 29,130 Dxf files with their corresponding script codes, where each sketch preserves both editability and geometric annotations. We evaluate CAD-Coder on various 2D/3D CAD generation tasks against existing methods, demonstrating superior interactive capabilities while uniquely providing editable sketches with geometric annotations.
>
---
#### [new 082] GradMix: Gradient-based Selective Mixup for Robust Data Augmentation in Class-Incremental Learning
- **分类: cs.LG; cs.AI; cs.CV**

- **简介: 该论文针对持续学习中的类增量学习任务，解决传统经验回放方法随机混合样本导致灾难性遗忘的问题。提出GradMix方法，通过梯度分析选择性混合有益类别对的样本进行数据增强，避免损害已有知识。实验验证其能有效减少遗忘并提升准确率。**

- **链接: [http://arxiv.org/pdf/2505.08528v1](http://arxiv.org/pdf/2505.08528v1)**

> **作者:** Minsu Kim; Seong-Hyeon Hwang; Steven Euijong Whang
>
> **摘要:** In the context of continual learning, acquiring new knowledge while maintaining previous knowledge presents a significant challenge. Existing methods often use experience replay techniques that store a small portion of previous task data for training. In experience replay approaches, data augmentation has emerged as a promising strategy to further improve the model performance by mixing limited previous task data with sufficient current task data. However, we theoretically and empirically analyze that training with mixed samples from random sample pairs may harm the knowledge of previous tasks and cause greater catastrophic forgetting. We then propose GradMix, a robust data augmentation method specifically designed for mitigating catastrophic forgetting in class-incremental learning. GradMix performs gradient-based selective mixup using a class-based criterion that mixes only samples from helpful class pairs and not from detrimental class pairs for reducing catastrophic forgetting. Our experiments on various real datasets show that GradMix outperforms data augmentation baselines in accuracy by minimizing the forgetting of previous knowledge.
>
---
#### [new 083] Claycode: Stylable and Deformable 2D Scannable Codes
- **分类: cs.GR; cs.CG; cs.CV; cs.HC; I.3.0; I.3.5; I.3.6; E.4**

- **简介: 该论文提出Claycode，一种支持高度风格化和形变的2D可扫描码，属于信息编码与识别任务。针对传统矩阵码（如QR码）在变形和设计限制下的易失效问题，其将数据编码为树结构，通过嵌套颜色区域实现形状和样式自由，并构建端到端编解码系统。实验证明其在大幅变形下仍保持高识别率，优于传统方案。**

- **链接: [http://arxiv.org/pdf/2505.08666v1](http://arxiv.org/pdf/2505.08666v1)**

> **作者:** Marco Maida; Alberto Crescini; Marco Perronet; Elena Camuffo
>
> **摘要:** This paper introduces Claycode, a novel 2D scannable code designed for extensive stylization and deformation. Unlike traditional matrix-based codes (e.g., QR codes), Claycodes encode their message in a tree structure. During the encoding process, bits are mapped into a topology tree, which is then depicted as a nesting of color regions drawn within the boundaries of a target polygon shape. When decoding, Claycodes are extracted and interpreted in real-time from a camera stream. We detail the end-to-end pipeline and show that Claycodes allow for extensive stylization without compromising their functionality. We then empirically demonstrate Claycode's high tolerance to heavy deformations, outperforming traditional 2D scannable codes in scenarios where they typically fail.
>
---
#### [new 084] Skeleton-Guided Diffusion Model for Accurate Foot X-ray Synthesis in Hallux Valgus Diagnosis
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像合成任务，旨在解决足部X光生成中骨骼一致性与图像质量不足的问题。针对拇外翻诊断需求，提出骨骼约束条件扩散模型（SCCDM），融合多尺度特征和注意力机制，并设计KCC骨骼评估方法，显著提升SSIM和PSNR指标，临床评分达0.85，代码已开源。**

- **链接: [http://arxiv.org/pdf/2505.08247v1](http://arxiv.org/pdf/2505.08247v1)**

> **作者:** Midi Wan; Pengfei Li; Yizhuo Liang; Di Wu; Yushan Pan; Guangzhen Zhu; Hao Wang
>
> **摘要:** Medical image synthesis plays a crucial role in providing anatomically accurate images for diagnosis and treatment. Hallux valgus, which affects approximately 19% of the global population, requires frequent weight-bearing X-rays for assessment, placing additional strain on both patients and healthcare providers. Existing X-ray models often struggle to balance image fidelity, skeletal consistency, and physical constraints, particularly in diffusion-based methods that lack skeletal guidance. We propose the Skeletal-Constrained Conditional Diffusion Model (SCCDM) and introduce KCC, a foot evaluation method utilizing skeletal landmarks. SCCDM incorporates multi-scale feature extraction and attention mechanisms, improving the Structural Similarity Index (SSIM) by 5.72% (0.794) and Peak Signal-to-Noise Ratio (PSNR) by 18.34% (21.40 dB). When combined with KCC, the model achieves an average score of 0.85, demonstrating strong clinical applicability. The code is available at https://github.com/midisec/SCCDM.
>
---
#### [new 085] OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval
- **分类: cs.IR; cs.AI; cs.CV**

- **简介: 该论文针对知识型视觉问答任务，解决多模态检索中模态与知识粒度协调不足的问题，提出分层多步检索系统：先粗粒度跨模态对齐，再融合多模态信息重排序，最后筛选细粒度文本增强生成，在主流基准达到最优性能。**

- **链接: [http://arxiv.org/pdf/2505.07879v1](http://arxiv.org/pdf/2505.07879v1)**

> **作者:** Wei Yang; Jingjing Fu; Rui Wang; Jinyu Wang; Lei Song; Jiang Bian
>
> **备注:** 19 pages, 6 figures, 17 tables
>
> **摘要:** Vision-language retrieval-augmented generation (RAG) has become an effective approach for tackling Knowledge-Based Visual Question Answering (KB-VQA), which requires external knowledge beyond the visual content presented in images. The effectiveness of Vision-language RAG systems hinges on multimodal retrieval, which is inherently challenging due to the diverse modalities and knowledge granularities in both queries and knowledge bases. Existing methods have not fully tapped into the potential interplay between these elements. We propose a multimodal RAG system featuring a coarse-to-fine, multi-step retrieval that harmonizes multiple granularities and modalities to enhance efficacy. Our system begins with a broad initial search aligning knowledge granularity for cross-modal retrieval, followed by a multimodal fusion reranking to capture the nuanced multimodal information for top entity selection. A text reranker then filters out the most relevant fine-grained section for augmented generation. Extensive experiments on the InfoSeek and Encyclopedic-VQA benchmarks show our method achieves state-of-the-art retrieval performance and highly competitive answering results, underscoring its effectiveness in advancing KB-VQA systems.
>
---
#### [new 086] Decoupled Multimodal Prototypes for Visual Recognition with Missing Modalities
- **分类: cs.LG; cs.CV**

- **简介: 该论文针对多模态视觉识别中模态缺失问题，提出解耦的原型输出头。现有方法假设所有模态可用，但实际常缺失导致性能下降。作者设计模态专属类别原型，动态适应不同缺失场景，兼容现有提示方法，显著提升各类缺失情况下的识别效果。**

- **链接: [http://arxiv.org/pdf/2505.08283v1](http://arxiv.org/pdf/2505.08283v1)**

> **作者:** Jueqing Lu; Yuanyuan Qi; Xiaohao Yang; Shujie Zhou; Lan Du
>
> **摘要:** Multimodal learning enhances deep learning models by enabling them to perceive and understand information from multiple data modalities, such as visual and textual inputs. However, most existing approaches assume the availability of all modalities, an assumption that often fails in real-world applications. Recent works have introduced learnable missing-case-aware prompts to mitigate performance degradation caused by missing modalities while reducing the need for extensive model fine-tuning. Building upon the effectiveness of missing-case-aware handling for missing modalities, we propose a novel decoupled prototype-based output head, which leverages missing-case-aware class-wise prototypes tailored for each individual modality. This approach dynamically adapts to different missing modality scenarios and can be seamlessly integrated with existing prompt-based methods. Extensive experiments demonstrate that our proposed output head significantly improves performance across a wide range of missing-modality scenarios and varying missing rates.
>
---
#### [new 087] UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representations
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人模仿学习任务，旨在解决人类与机器人形态差异导致的技能迁移难题。提出UniSkill框架，通过无监督学习从跨形态视频中提取通用技能表示，使人类视频中的技能能直接迁移到机器人策略，无需对齐数据。实验验证了其在仿真和真实环境中的有效性。**

- **链接: [http://arxiv.org/pdf/2505.08787v1](http://arxiv.org/pdf/2505.08787v1)**

> **作者:** Hanjung Kim; Jaehyun Kang; Hyolim Kang; Meedeum Cho; Seon Joo Kim; Youngwoon Lee
>
> **备注:** Project Page: https://kimhanjung.github.io/UniSkill/
>
> **摘要:** Mimicry is a fundamental learning mechanism in humans, enabling individuals to learn new tasks by observing and imitating experts. However, applying this ability to robots presents significant challenges due to the inherent differences between human and robot embodiments in both their visual appearance and physical capabilities. While previous methods bridge this gap using cross-embodiment datasets with shared scenes and tasks, collecting such aligned data between humans and robots at scale is not trivial. In this paper, we propose UniSkill, a novel framework that learns embodiment-agnostic skill representations from large-scale cross-embodiment video data without any labels, enabling skills extracted from human video prompts to effectively transfer to robot policies trained only on robot data. Our experiments in both simulation and real-world environments show that our cross-embodiment skills successfully guide robots in selecting appropriate actions, even with unseen video prompts. The project website can be found at: https://kimhanjung.github.io/UniSkill.
>
---
#### [new 088] Pose Estimation for Intra-cardiac Echocardiography Catheter via AI-Based Anatomical Understanding
- **分类: eess.IV; cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于医学图像姿态估计任务，旨在解决心脏超声导管术中依赖外部跟踪或人工调整的问题。研究者提出基于Vision Transformer的深度学习模型，通过分析超声图像直接预测导管位置和方向，使用851例临床数据训练，实现平均9.48mm位置误差及16度内方向误差，提升介入手术精度与效率。**

- **链接: [http://arxiv.org/pdf/2505.07851v1](http://arxiv.org/pdf/2505.07851v1)**

> **作者:** Jaeyoung Huh; Ankur Kapoor; Young-Ho Kim
>
> **摘要:** Intra-cardiac Echocardiography (ICE) plays a crucial role in Electrophysiology (EP) and Structural Heart Disease (SHD) interventions by providing high-resolution, real-time imaging of cardiac structures. However, existing navigation methods rely on electromagnetic (EM) tracking, which is susceptible to interference and position drift, or require manual adjustments based on operator expertise. To overcome these limitations, we propose a novel anatomy-aware pose estimation system that determines the ICE catheter position and orientation solely from ICE images, eliminating the need for external tracking sensors. Our approach leverages a Vision Transformer (ViT)-based deep learning model, which captures spatial relationships between ICE images and anatomical structures. The model is trained on a clinically acquired dataset of 851 subjects, including ICE images paired with position and orientation labels normalized to the left atrium (LA) mesh. ICE images are patchified into 16x16 embeddings and processed through a transformer network, where a [CLS] token independently predicts position and orientation via separate linear layers. The model is optimized using a Mean Squared Error (MSE) loss function, balancing positional and orientational accuracy. Experimental results demonstrate an average positional error of 9.48 mm and orientation errors of (16.13 deg, 8.98 deg, 10.47 deg) across x, y, and z axes, confirming the model accuracy. Qualitative assessments further validate alignment between predicted and target views within 3D cardiac meshes. This AI-driven system enhances procedural efficiency, reduces operator workload, and enables real-time ICE catheter localization for tracking-free procedures. The proposed method can function independently or complement existing mapping systems like CARTO, offering a transformative approach to ICE-guided interventions.
>
---
#### [new 089] Improving Unsupervised Task-driven Models of Ventral Visual Stream via Relative Position Predictivity
- **分类: cs.CE; cs.CV**

- **简介: 论文提出改进腹侧视觉流(VVS)的无监督任务驱动模型，解决现有模型仅关注物体识别、忽略位置感知的问题。通过理论分析指出对比学习的局限，将相对位置预测任务与对比学习结合，提出新方法。实验证明该方法提升了物体识别性能和脑相似性，验证了VVS参与位置感知的计算机制。**

- **链接: [http://arxiv.org/pdf/2505.08316v1](http://arxiv.org/pdf/2505.08316v1)**

> **作者:** Dazhong Rong; Hao Dong; Xing Gao; Jiyu Wei; Di Hong; Yaoyao Hao; Qinming He; Yueming Wang
>
> **备注:** This paper has been accepted for full publication at CogSci 2025 (https://cognitivesciencesociety.org/cogsci-2025/)
>
> **摘要:** Based on the concept that ventral visual stream (VVS) mainly functions for object recognition, current unsupervised task-driven methods model VVS by contrastive learning, and have achieved good brain similarity. However, we believe functions of VVS extend beyond just object recognition. In this paper, we introduce an additional function involving VVS, named relative position (RP) prediction. We first theoretically explain contrastive learning may be unable to yield the model capability of RP prediction. Motivated by this, we subsequently integrate RP learning with contrastive learning, and propose a new unsupervised task-driven method to model VVS, which is more inline with biological reality. We conduct extensive experiments, demonstrating that: (i) our method significantly improves downstream performance of object recognition while enhancing RP predictivity; (ii) RP predictivity generally improves the model brain similarity. Our results provide strong evidence for the involvement of VVS in location perception (especially RP prediction) from a computational perspective.
>
---
#### [new 090] Monocular Online Reconstruction with Enhanced Detail Preservation
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于单目在线三维重建任务，解决无深度图的高斯分布及局部与全局一致性问题。提出分层高斯管理模块、全局优化模块和多级哈希体素结构（MOHV），通过动态调节高斯粒度和跨尺度约束，实现高效高精度的细节重建，在纹理与几何还原上超越现有RGB/D方法，兼具计算效率和系统兼容性。**

- **链接: [http://arxiv.org/pdf/2505.07887v1](http://arxiv.org/pdf/2505.07887v1)**

> **作者:** Songyin Wu; Zhaoyang Lv; Yufeng Zhu; Duncan Frost; Zhengqin Li; Ling-Qi Yan; Carl Ren; Richard Newcombe; Zhao Dong
>
> **摘要:** We propose an online 3D Gaussian-based dense mapping framework for photorealistic details reconstruction from a monocular image stream. Our approach addresses two key challenges in monocular online reconstruction: distributing Gaussians without relying on depth maps and ensuring both local and global consistency in the reconstructed maps. To achieve this, we introduce two key modules: the Hierarchical Gaussian Management Module for effective Gaussian distribution and the Global Consistency Optimization Module for maintaining alignment and coherence at all scales. In addition, we present the Multi-level Occupancy Hash Voxels (MOHV), a structure that regularizes Gaussians for capturing details across multiple levels of granularity. MOHV ensures accurate reconstruction of both fine and coarse geometries and textures, preserving intricate details while maintaining overall structural integrity. Compared to state-of-the-art RGB-only and even RGB-D methods, our framework achieves superior reconstruction quality with high computational efficiency. Moreover, it integrates seamlessly with various tracking systems, ensuring generality and scalability.
>
---
#### [new 091] Efficient Unstructured Pruning of Mamba State-Space Models for Resource-Constrained Environments
- **分类: cs.LG; cs.CV**

- **简介: 该论文属于模型压缩任务，旨在解决Mamba状态空间模型参数量大、难以部署在资源受限环境的问题。通过提出结合梯度感知剪枝、迭代稀疏化调度和全局优化的非结构化剪枝框架，实现了70%参数削减且保持95%以上性能，验证了模型冗余分布规律并提升实用价值。**

- **链接: [http://arxiv.org/pdf/2505.08299v1](http://arxiv.org/pdf/2505.08299v1)**

> **作者:** Ibne Farabi Shihab; Sanjeda Akter; Anuj Sharma
>
> **摘要:** State-space models (SSMs), particularly the Mamba architecture, have emerged as powerful alternatives to Transformers for sequence modeling, offering linear-time complexity and competitive performance across diverse tasks. However, their large parameter counts pose significant challenges for deployment in resource-constrained environments. We propose a novel unstructured pruning framework tailored for Mamba models that achieves up to 70\% parameter reduction while retaining over 95\% of the original performance. Our approach integrates three key innovations: (1) a gradient-aware magnitude pruning technique that combines weight magnitude and gradient information to identify less critical parameters, (2) an iterative pruning schedule that gradually increases sparsity to maintain model stability, and (3) a global pruning strategy that optimizes parameter allocation across the entire model. Through extensive experiments on WikiText-103, Long Range Arena, and ETT time-series benchmarks, we demonstrate significant efficiency gains with minimal performance degradation. Our analysis of pruning effects on Mamba's components reveals critical insights into the architecture's redundancy and robustness, enabling practical deployment in resource-constrained settings while broadening Mamba's applicability.
>
---
#### [new 092] Where the Devil Hides: Deepfake Detectors Can No Longer Be Trusted
- **分类: cs.CR; cs.CV**

- **简介: 该论文属于对抗攻击任务，针对第三方数据投毒导致Deepfake检测器存在后门漏洞的问题，提出一种隐蔽触发器生成方法，通过脏标签/干净标签投毒注入后门，实验验证了攻击的隐蔽性和有效性。**

- **链接: [http://arxiv.org/pdf/2505.08255v1](http://arxiv.org/pdf/2505.08255v1)**

> **作者:** Shuaiwei Yuan; Junyu Dong; Yuezun Li
>
> **备注:** CVPR 2025
>
> **摘要:** With the advancement of AI generative techniques, Deepfake faces have become incredibly realistic and nearly indistinguishable to the human eye. To counter this, Deepfake detectors have been developed as reliable tools for assessing face authenticity. These detectors are typically developed on Deep Neural Networks (DNNs) and trained using third-party datasets. However, this protocol raises a new security risk that can seriously undermine the trustfulness of Deepfake detectors: Once the third-party data providers insert poisoned (corrupted) data maliciously, Deepfake detectors trained on these datasets will be injected ``backdoors'' that cause abnormal behavior when presented with samples containing specific triggers. This is a practical concern, as third-party providers may distribute or sell these triggers to malicious users, allowing them to manipulate detector performance and escape accountability. This paper investigates this risk in depth and describes a solution to stealthily infect Deepfake detectors. Specifically, we develop a trigger generator, that can synthesize passcode-controlled, semantic-suppression, adaptive, and invisible trigger patterns, ensuring both the stealthiness and effectiveness of these triggers. Then we discuss two poisoning scenarios, dirty-label poisoning and clean-label poisoning, to accomplish the injection of backdoors. Extensive experiments demonstrate the effectiveness, stealthiness, and practicality of our method compared to several baselines.
>
---
#### [new 093] VIViT: Variable-Input Vision Transformer Framework for 3D MR Image Segmentation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决多对比度MR数据因输入不一致导致的自监督预训练与下游任务适配问题。提出VIViT框架，基于Transformer支持可变模态输入，通过自监督预训练最大化数据利用率，并在脑梗死（Dice 0.624）和肿瘤分割（0.883）任务中超越现有模型。**

- **链接: [http://arxiv.org/pdf/2505.08693v1](http://arxiv.org/pdf/2505.08693v1)**

> **作者:** Badhan Kumar Das; Ajay Singh; Gengyan Zhao; Han Liu; Thomas J. Re; Dorin Comaniciu; Eli Gibson; Andreas Maier
>
> **备注:** 9 pages
>
> **摘要:** Self-supervised pretrain techniques have been widely used to improve the downstream tasks' performance. However, real-world magnetic resonance (MR) studies usually consist of different sets of contrasts due to different acquisition protocols, which poses challenges for the current deep learning methods on large-scale pretrain and different downstream tasks with different input requirements, since these methods typically require a fixed set of input modalities or, contrasts. To address this challenge, we propose variable-input ViT (VIViT), a transformer-based framework designed for self-supervised pretraining and segmentation finetuning for variable contrasts in each study. With this ability, our approach can maximize the data availability in pretrain, and can transfer the learned knowledge from pretrain to downstream tasks despite variations in input requirements. We validate our method on brain infarct and brain tumor segmentation, where our method outperforms current CNN and ViT-based models with a mean Dice score of 0.624 and 0.883 respectively. These results highlight the efficacy of our design for better adaptability and performance on tasks with real-world heterogeneous MR data.
>
---
#### [new 094] SpNeRF: Memory Efficient Sparse Volumetric Neural Rendering Accelerator for Edge Devices
- **分类: cs.AR; cs.CV**

- **简介: 该论文提出SpNeRF，针对边缘设备稀疏体素神经渲染的存储效率优化任务，解决传统方法因大网格数据导致的频繁片外存储访问和高能耗问题。通过哈希映射预处理和位图掩码在线解码压缩内存，结合专用硬件设计，实现内存缩减21倍且保持画质，相比主流方案提速最高95倍，能效提升达625倍。**

- **链接: [http://arxiv.org/pdf/2505.08191v1](http://arxiv.org/pdf/2505.08191v1)**

> **作者:** Yipu Zhang; Jiawei Liang; Jian Peng; Jiang Xu; Wei Zhang
>
> **备注:** Accepted by DATE 2025
>
> **摘要:** Neural rendering has gained prominence for its high-quality output, which is crucial for AR/VR applications. However, its large voxel grid data size and irregular access patterns challenge real-time processing on edge devices. While previous works have focused on improving data locality, they have not adequately addressed the issue of large voxel grid sizes, which necessitate frequent off-chip memory access and substantial on-chip memory. This paper introduces SpNeRF, a software-hardware co-design solution tailored for sparse volumetric neural rendering. We first identify memory-bound rendering inefficiencies and analyze the inherent sparsity in the voxel grid data of neural rendering. To enhance efficiency, we propose novel preprocessing and online decoding steps, reducing the memory size for voxel grid. The preprocessing step employs hash mapping to support irregular data access while maintaining a minimal memory size. The online decoding step enables efficient on-chip sparse voxel grid processing, incorporating bitmap masking to mitigate PSNR loss caused by hash collisions. To further optimize performance, we design a dedicated hardware architecture supporting our sparse voxel grid processing technique. Experimental results demonstrate that SpNeRF achieves an average 21.07$\times$ reduction in memory size while maintaining comparable PSNR levels. When benchmarked against Jetson XNX, Jetson ONX, RT-NeRF.Edge and NeuRex.Edge, our design achieves speedups of 95.1$\times$, 63.5$\times$, 1.5$\times$ and 10.3$\times$, and improves energy efficiency by 625.6$\times$, 529.1$\times$, 4$\times$, and 4.4$\times$, respectively.
>
---
#### [new 095] Evaluation of UAV-Based RGB and Multispectral Vegetation Indices for Precision Agriculture in Palm Tree Cultivation
- **分类: eess.IV; cs.CV**

- **简介: 该论文属于农业监测技术评估任务，旨在解决精准农业中多光谱传感器成本高的问题。通过对比无人机搭载的多光谱和RGB相机在棕榈树健康监测中的表现，证明RGB植被指数（如VARI）能达到与多光谱指数（如NDVI）相当的分类精度，为大规模农田提供低成本监测方案。**

- **链接: [http://arxiv.org/pdf/2505.07840v1](http://arxiv.org/pdf/2505.07840v1)**

> **作者:** Alavikunhu Panthakkan; S M Anzar; K. Sherin; Saeed Al Mansoori; Hussain Al-Ahmad
>
> **摘要:** Precision farming relies on accurate vegetation monitoring to enhance crop productivity and promote sustainable agricultural practices. This study presents a comprehensive evaluation of UAV-based imaging for vegetation health assessment in a palm tree cultivation region in Dubai. By comparing multispectral and RGB image data, we demonstrate that RGBbased vegetation indices offer performance comparable to more expensive multispectral indices, providing a cost-effective alternative for large-scale agricultural monitoring. Using UAVs equipped with multispectral sensors, indices such as NDVI and SAVI were computed to categorize vegetation into healthy, moderate, and stressed conditions. Simultaneously, RGB-based indices like VARI and MGRVI delivered similar results in vegetation classification and stress detection. Our findings highlight the practical benefits of integrating RGB imagery into precision farming, reducing operational costs while maintaining accuracy in plant health monitoring. This research underscores the potential of UAVbased RGB imaging as a powerful tool for precision agriculture, enabling broader adoption of data-driven decision-making in crop management. By leveraging the strengths of both multispectral and RGB imaging, this work advances the state of UAV applications in agriculture, paving the way for more efficient and scalable farming solutions.
>
---
#### [new 096] ACT-R: Adaptive Camera Trajectories for 3D Reconstruction from Single Image
- **分类: cs.GR; cs.CV**

- **简介: 该论文属于单图像3D重建任务，旨在解决多视角合成中遮挡区域揭示不足和3D一致性问题。提出自适应相机轨迹规划方法，通过分析遮挡动态生成最优环绕视点序列，并利用视频扩散模型合成新视角，结合预训练模型实现高效重建。实验表明其在GSO数据集上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.08239v1](http://arxiv.org/pdf/2505.08239v1)**

> **作者:** Yizhi Wang; Mingrui Zhao; Ali Mahdavi-Amiri; Hao Zhang
>
> **摘要:** We introduce adaptive view planning to multi-view synthesis, aiming to improve both occlusion revelation and 3D consistency for single-view 3D reconstruction. Instead of generating an unordered set of views independently or simultaneously, we generate a sequence of views, leveraging temporal consistency to enhance 3D coherence. Most importantly, our view sequence is not determined by a pre-determined camera setup. Instead, we compute an adaptive camera trajectory (ACT), specifically, an orbit of camera views, which maximizes the visibility of occluded regions of the 3D object to be reconstructed. Once the best orbit is found, we feed it to a video diffusion model to generate novel views around the orbit, which in turn, are passed to a multi-view 3D reconstruction model to obtain the final reconstruction. Our multi-view synthesis pipeline is quite efficient since it involves no run-time training/optimization, only forward inferences by applying the pre-trained models for occlusion analysis and multi-view synthesis. Our method predicts camera trajectories that reveal occlusions effectively and produce consistent novel views, significantly improving 3D reconstruction over SOTA on the unseen GSO dataset, both quantitatively and qualitatively.
>
---
#### [new 097] Arrow-Guided VLM: Enhancing Flowchart Understanding via Arrow Direction Encoding
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究流程图理解的视觉语言模型任务，解决现有模型误判箭头方向及拓扑结构的问题。提出七阶段流程，整合箭头检测、OCR文本提取和结构化提示生成，无需微调将流程图QA准确率从80%提升至89%，显著改善下一步推理（100%准确率）。方法依赖显式箭头编码，但多入边节点仍存挑战，未来将扩展数据集并验证BPMN/UML适用性。（100字）**

- **链接: [http://arxiv.org/pdf/2505.07864v1](http://arxiv.org/pdf/2505.07864v1)**

> **作者:** Takamitsu Omasa; Ryo Koshihara; Masumi Morishige
>
> **备注:** 11 pages, 1 figures,
>
> **摘要:** Flowcharts are indispensable tools in software design and business-process analysis, yet current vision-language models (VLMs) frequently misinterpret the directional arrows and graph topology that set these diagrams apart from natural images. We introduce a seven-stage pipeline grouped into three broader processes: (1) arrow-aware detection of nodes and arrow endpoints; (2) optical character recognition (OCR) to extract node text; and (3) construction of a structured prompt that guides the VLMs. Tested on a 90-question benchmark distilled from 30 annotated flowcharts, the method raises overall accuracy from 80 % to 89 % (+9 percentage points) without any task-specific fine-tuning. The gain is most pronounced for next-step queries (25/30 -> 30/30; 100 %, +17 pp); branch-result questions improve more modestly, and before-step questions remain difficult. A parallel evaluation with an LLM-as-a-Judge protocol shows the same trends, reinforcing the advantage of explicit arrow encoding. Limitations include dependence on detector and OCR precision, the small evaluation set, and residual errors at nodes with multiple incoming edges. Future work will enlarge the benchmark with synthetic and handwritten flowcharts and assess the approach on Business Process Model and Notation (BPMN) and Unified Modeling Language (UML).
>
---
#### [new 098] A Reproduction Study: The Kernel PCA Interpretation of Self-Attention Fails Under Scrutiny
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于验证性研究，针对自注意力机制是否实现核主成分分析(KPCA)的论点进行复现检验。通过分析10种Transformer架构，发现原主张的三个核心论据（值向量与Gram矩阵特征向量对齐、投影误差优化、Gram矩阵特征值统计）均存在矛盾，表明自注意力与KPCA缺乏实证关联。**

- **链接: [http://arxiv.org/pdf/2505.07908v1](http://arxiv.org/pdf/2505.07908v1)**

> **作者:** Karahan Sarıtaş; Çağatay Yıldız
>
> **摘要:** In this reproduction study, we revisit recent claims that self-attention implements kernel principal component analysis (KPCA) (Teo et al., 2024), positing that (i) value vectors $V$ capture the eigenvectors of the Gram matrix of the keys, and (ii) that self-attention projects queries onto the principal component axes of the key matrix $K$ in a feature space. Our analysis reveals three critical inconsistencies: (1) No alignment exists between learned self-attention value vectors and what is proposed in the KPCA perspective, with average similarity metrics (optimal cosine similarity $\leq 0.32$, linear CKA (Centered Kernel Alignment) $\leq 0.11$, kernel CKA $\leq 0.32$) indicating negligible correspondence; (2) Reported decreases in reconstruction loss $J_\text{proj}$, arguably justifying the claim that the self-attention minimizes the projection error of KPCA, are misinterpreted, as the quantities involved differ by orders of magnitude ($\sim\!10^3$); (3) Gram matrix eigenvalue statistics, introduced to justify that $V$ captures the eigenvector of the gram matrix, are irreproducible without undocumented implementation-specific adjustments. Across 10 transformer architectures, we conclude that the KPCA interpretation of self-attention lacks empirical support.
>
---
## 更新

#### [replaced 001] Web2Grasp: Learning Functional Grasps from Web Images of Hand-Object Interactions
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.05517v2](http://arxiv.org/pdf/2505.05517v2)**

> **作者:** Hongyi Chen; Yunchao Yao; Yufei Ye; Zhixuan Xu; Homanga Bharadhwaj; Jiashun Wang; Shubham Tulsiani; Zackory Erickson; Jeffrey Ichnowski
>
> **摘要:** Functional grasp is essential for enabling dexterous multi-finger robot hands to manipulate objects effectively. However, most prior work either focuses on power grasping, which simply involves holding an object still, or relies on costly teleoperated robot demonstrations to teach robots how to grasp each object functionally. Instead, we propose extracting human grasp information from web images since they depict natural and functional object interactions, thereby bypassing the need for curated demonstrations. We reconstruct human hand-object interaction (HOI) 3D meshes from RGB images, retarget the human hand to multi-finger robot hands, and align the noisy object mesh with its accurate 3D shape. We show that these relatively low-quality HOI data from inexpensive web sources can effectively train a functional grasping model. To further expand the grasp dataset for seen and unseen objects, we use the initially-trained grasping policy with web data in the IsaacGym simulator to generate physically feasible grasps while preserving functionality. We train the grasping model on 10 object categories and evaluate it on 9 unseen objects, including challenging items such as syringes, pens, spray bottles, and tongs, which are underrepresented in existing datasets. The model trained on the web HOI dataset, achieving a 75.8% success rate on seen objects and 61.8% across all objects in simulation, with a 6.7% improvement in success rate and a 1.8x increase in functionality ratings over baselines. Simulator-augmented data further boosts performance from 61.8% to 83.4%. The sim-to-real transfer to the LEAP Hand achieves a 85% success rate. Project website is at: https://web2grasp.github.io/.
>
---
#### [replaced 002] Semantic Shift Estimation via Dual-Projection and Classifier Reconstruction for Exemplar-Free Class-Incremental Learning
- **分类: cs.CV; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.05423v3](http://arxiv.org/pdf/2503.05423v3)**

> **作者:** Run He; Di Fang; Yicheng Xu; Yawen Cui; Ming Li; Cen Chen; Ziqian Zeng; Huiping Zhuang
>
> **备注:** Accepted by ICML 2025; Camera ready version
>
> **摘要:** Exemplar-Free Class-Incremental Learning (EFCIL) aims to sequentially learn from distinct categories without retaining exemplars but easily suffers from catastrophic forgetting of learned knowledge. While existing EFCIL methods leverage knowledge distillation to alleviate forgetting, they still face two critical challenges: semantic shift and decision bias. Specifically, the embeddings of old tasks shift in the embedding space after learning new tasks, and the classifier becomes biased towards new tasks due to training solely with new data, hindering the balance between old and new knowledge. To address these issues, we propose the Dual-Projection Shift Estimation and Classifier Reconstruction (DPCR) approach for EFCIL. DPCR effectively estimates semantic shift through a dual-projection, which combines a learnable transformation with a row-space projection to capture both task-wise and category-wise shifts. Furthermore, to mitigate decision bias, DPCR employs ridge regression to reformulate a classifier reconstruction process. This reconstruction exploits previous in covariance and prototype of each class after calibration with estimated shift, thereby reducing decision bias. Extensive experiments demonstrate that, on various datasets, DPCR effectively balances old and new tasks, outperforming state-of-the-art EFCIL methods. Our codes are available at https://github.com/RHe502/ICML25-DPCR.
>
---
#### [replaced 003] Visual Imitation Enables Contextual Humanoid Control
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.03729v3](http://arxiv.org/pdf/2505.03729v3)**

> **作者:** Arthur Allshire; Hongsuk Choi; Junyi Zhang; David McAllister; Anthony Zhang; Chung Min Kim; Trevor Darrell; Pieter Abbeel; Jitendra Malik; Angjoo Kanazawa
>
> **备注:** Project website: https://www.videomimic.net/
>
> **摘要:** How can we teach humanoids to climb staircases and sit on chairs using the surrounding environment context? Arguably, the simplest way is to just show them-casually capture a human motion video and feed it to humanoids. We introduce VIDEOMIMIC, a real-to-sim-to-real pipeline that mines everyday videos, jointly reconstructs the humans and the environment, and produces whole-body control policies for humanoid robots that perform the corresponding skills. We demonstrate the results of our pipeline on real humanoid robots, showing robust, repeatable contextual control such as staircase ascents and descents, sitting and standing from chairs and benches, as well as other dynamic whole-body skills-all from a single policy, conditioned on the environment and global root commands. VIDEOMIMIC offers a scalable path towards teaching humanoids to operate in diverse real-world environments.
>
---
#### [replaced 004] Clinically inspired enhance Explainability and Interpretability of an AI-Tool for BCC diagnosis based on expert annotation
- **分类: cs.LG; cs.AI; cs.CV; cs.IR; eess.IV**

- **链接: [http://arxiv.org/pdf/2407.00104v2](http://arxiv.org/pdf/2407.00104v2)**

> **作者:** Iván Matas; Carmen Serrano; Francisca Silva; Amalia Serrano; Tomás Toledo-Pastrana; Begoña Acha
>
> **备注:** 8 pages, 4 figures, 4 tables, under review
>
> **摘要:** An AI tool has been developed to provide interpretable support for the diagnosis of BCC via teledermatology, thus speeding up referrals and optimizing resource utilization. The interpretability is provided in two ways: on the one hand, the main BCC dermoscopic patterns are found in the image to justify the BCC/Non BCC classification. Secondly, based on the common visual XAI Grad-CAM, a clinically inspired visual explanation is developed where the relevant features for diagnosis are located. Since there is no established ground truth for BCC dermoscopic features, a standard reference is inferred from the diagnosis of four dermatologists using an Expectation Maximization (EM) based algorithm. The results demonstrate significant improvements in classification accuracy and interpretability, positioning this approach as a valuable tool for early BCC detection and referral to dermatologists. The BCC/non-BCC classification achieved an accuracy rate of 90%. For Clinically-inspired XAI results, the detection of BCC patterns useful to clinicians reaches 99% accuracy. As for the Clinically-inspired Visual XAI results, the mean of the Grad-CAM normalized value within the manually segmented clinical features is 0.57, while outside this region it is 0.16. This indicates that the model struggles to accurately identify the regions of the BCC patterns. These results prove the ability of the AI tool to provide a useful explanation.
>
---
#### [replaced 005] Decadal analysis of sea surface temperature patterns, climatology, and anomalies in temperate coastal waters with Landsat-8 TIRS observations
- **分类: physics.ao-ph; cs.CV; eess.IV; eess.SP; physics.geo-ph**

- **链接: [http://arxiv.org/pdf/2503.05843v2](http://arxiv.org/pdf/2503.05843v2)**

> **作者:** Yiqing Guo; Nagur Cherukuru; Eric Lehmann; Xiubin Qi; Mark Doubelld; S. L. Kesav Unnithan; Ming Feng
>
> **备注:** Submitted to GIScience & Remote Sensing
>
> **摘要:** Sea surface temperature (SST) is a fundamental physical parameter characterising the thermal state of sea surface. Due to the intricate thermal interactions between land, sea, and atmosphere, the spatial gradients of SST in coastal waters often appear at finer spatial scales than those in open ocean waters. The Thermal Infrared Sensor (TIRS) onboard Landsat-8, with its 100-meter spatial resolution, offers a unique opportunity to uncover fine-scale coastal SST patterns that would otherwise be overlooked by coarser-resolution thermal sensors. In this study, we first analysed the spatiotemporal patterns of SST in South Australia's temperate coastal waters from 2014 to 2023 by developing an operational approach for SST retrieval from the Landsat-8 TIRS sensor. A buoy was deployed off the coast of Port Lincoln, South Australia, to validate the quality of SST retrievals. Then the daily baseline climatology of SST with 100 m resolution was constructed, which allowed for the detection and analysis of anomalous SST events. Our results suggest the following: (1) the satellite-derived SST data aligned well with the in-situ measured SST values; (2) the semi-enclosed, shallow regions of Upper Spencer Gulf and Upper St Vincent Gulf showed higher temperatures during summer and cooler temperatures during winter than waters closer to the open ocean, resulting in a higher seasonal variation in SST; (3) the near-shore shallow areas in Spencer Gulf and St Vincent Gulf, and regions surrounding Kangaroo Island, were identified to have a higher probability of SST anomalies compared to the rest of the study area; and (4) anomalous SST events were more likely to happen during the warm months than the cool months. We hope these findings would be helpful in supporting the fishing and aquaculture industries in the coastal waters of South Australia.
>
---
#### [replaced 006] An Analysis of Data Transformation Effects on Segment Anything 2
- **分类: eess.IV; cs.AI; cs.CV; 68T45; I.4.6; I.2.10**

- **链接: [http://arxiv.org/pdf/2503.00042v2](http://arxiv.org/pdf/2503.00042v2)**

> **作者:** Clayton Bromley; Alexander Moore; Amar Saini; Doug Poland; Carmen Carrano
>
> **备注:** 11 pages, 30 figures
>
> **摘要:** Video object segmentation (VOS) is a critical task in the development of video perception and understanding. The Segment-Anything Model 2 (SAM 2), released by Meta AI, is the current state-of-the-art architecture for end-to-end VOS. SAM 2 performs very well on both clean video data and augmented data, and completely intelligent video perception requires an understanding of how this architecture is capable of achieving such quality results. To better understand how each step within the SAM 2 architecture permits high-quality video segmentation, a variety of complex video transformations are passed through the architecture, and the impact at each stage of the process is measured. It is observed that each progressive stage enables the filtering of complex transformation noise and the emphasis of the object of interest. Contributions include the creation of complex transformation video datasets, an analysis of how each stage of the SAM 2 architecture interprets these transformations, and visualizations of segmented objects through each stage. By better understanding how each model structure impacts overall video understanding, VOS development can work to improve real-world applicability and performance tracking, localizing, and segmenting objects despite complex cluttered scenes and obscurations.
>
---
#### [replaced 007] Motion Blender Gaussian Splatting for Dynamic Scene Reconstruction
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.09040v2](http://arxiv.org/pdf/2503.09040v2)**

> **作者:** Xinyu Zhang; Haonan Chang; Yuhan Liu; Abdeslam Boularias
>
> **摘要:** Gaussian splatting has emerged as a powerful tool for high-fidelity reconstruction of dynamic scenes. However, existing methods primarily rely on implicit motion representations, such as encoding motions into neural networks or per-Gaussian parameters, which makes it difficult to further manipulate the reconstructed motions. This lack of explicit controllability limits existing methods to replaying recorded motions only, which hinders a wider application in robotics. To address this, we propose Motion Blender Gaussian Splatting (MBGS), a novel framework that uses motion graphs as an explicit and sparse motion representation. The motion of a graph's links is propagated to individual Gaussians via dual quaternion skinning, with learnable weight painting functions that determine the influence of each link. The motion graphs and 3D Gaussians are jointly optimized from input videos via differentiable rendering. Experiments show that MBGS achieves state-of-the-art performance on the highly challenging iPhone dataset while being competitive on HyperNeRF. We demonstrate the application potential of our method in animating novel object poses, synthesizing real robot demonstrations, and predicting robot actions through visual planning. The source code, models, video demonstrations can be found at http://mlzxy.github.io/motion-blender-gs.
>
---
#### [replaced 008] Adaptive Integrated Layered Attention (AILA)
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.IR; cs.NE**

- **链接: [http://arxiv.org/pdf/2503.22742v2](http://arxiv.org/pdf/2503.22742v2)**

> **作者:** William Claster; Suhas KM; Dhairya Gundechia
>
> **摘要:** We propose Adaptive Integrated Layered Attention (AILA), a neural network architecture that combines dense skip connections with different mechanisms for adaptive feature reuse across network layers. We evaluate AILA on three challenging tasks: price forecasting for various commodities and indices (S&P 500, Gold, US dollar Futures, Coffee, Wheat), image recognition using the CIFAR-10 dataset, and sentiment analysis on the IMDB movie review dataset. In all cases, AILA matches strong deep learning baselines (LSTMs, Transformers, and ResNets), achieving it at a fraction of the training and inference time. Notably, we implement and test two versions of the model - AILA-Architecture 1, which uses simple linear layers as the connection mechanism between layers, and AILA-Architecture 2, which implements an attention mechanism to selectively focus on outputs from previous layers. Both architectures are applied in a single-task learning setting, with each model trained separately for individual tasks. Results confirm that AILA's adaptive inter-layer connections yield robust gains by flexibly reusing pertinent features at multiple network depths. The AILA approach thus presents an extension to existing architectures, improving long-range sequence modeling, image recognition with optimised computational speed, and SOTA classification performance in practice.
>
---
#### [replaced 009] FLUXSynID: A Framework for Identity-Controlled Synthetic Face Generation with Document and Live Images
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.07530v2](http://arxiv.org/pdf/2505.07530v2)**

> **作者:** Raul Ismayilov; Dzemila Sero; Luuk Spreeuwers
>
> **摘要:** Synthetic face datasets are increasingly used to overcome the limitations of real-world biometric data, including privacy concerns, demographic imbalance, and high collection costs. However, many existing methods lack fine-grained control over identity attributes and fail to produce paired, identity-consistent images under structured capture conditions. We introduce FLUXSynID, a framework for generating high-resolution synthetic face datasets with user-defined identity attribute distributions and paired document-style and trusted live capture images. The dataset generated using the FLUXSynID framework shows improved alignment with real-world identity distributions and greater inter-set diversity compared to prior work. The FLUXSynID framework for generating custom datasets, along with a dataset of 14,889 synthetic identities, is publicly released to support biometric research, including face recognition and morphing attack detection.
>
---
#### [replaced 010] UAV-VLA: Vision-Language-Action System for Large Scale Aerial Mission Generation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.05014v2](http://arxiv.org/pdf/2501.05014v2)**

> **作者:** Oleg Sautenkov; Yasheerah Yaqoot; Artem Lykov; Muhammad Ahsan Mustafa; Grik Tadevosyan; Aibek Akhmetkazy; Miguel Altamirano Cabrera; Mikhail Martynov; Sausar Karaf; Dzmitry Tsetserukou
>
> **备注:** HRI 2025
>
> **摘要:** The UAV-VLA (Visual-Language-Action) system is a tool designed to facilitate communication with aerial robots. By integrating satellite imagery processing with the Visual Language Model (VLM) and the powerful capabilities of GPT, UAV-VLA enables users to generate general flight paths-and-action plans through simple text requests. This system leverages the rich contextual information provided by satellite images, allowing for enhanced decision-making and mission planning. The combination of visual analysis by VLM and natural language processing by GPT can provide the user with the path-and-action set, making aerial operations more efficient and accessible. The newly developed method showed the difference in the length of the created trajectory in 22% and the mean error in finding the objects of interest on a map in 34.22 m by Euclidean distance in the K-Nearest Neighbors (KNN) approach.
>
---
#### [replaced 011] Automatic quality control in multi-centric fetal brain MRI super-resolution reconstruction
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10156v3](http://arxiv.org/pdf/2503.10156v3)**

> **作者:** Thomas Sanchez; Vladyslav Zalevskyi; Angeline Mihailov; Gerard Martí-Juan; Elisenda Eixarch; Andras Jakab; Vincent Dunet; Mériam Koob; Guillaume Auzias; Meritxell Bach Cuadra
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Quality control (QC) has long been considered essential to guarantee the reliability of neuroimaging studies. It is particularly important for fetal brain MRI, where acquisitions and image processing techniques are less standardized than in adult imaging. In this work, we focus on automated quality control of super-resolution reconstruction (SRR) volumes of fetal brain MRI, an important processing step where multiple stacks of thick 2D slices are registered together and combined to build a single, isotropic and artifact-free T2 weighted volume. We propose FetMRQC$_{SR}$, a machine-learning method that extracts more than 100 image quality metrics to predict image quality scores using a random forest model. This approach is well suited to a problem that is high dimensional, with highly heterogeneous data and small datasets. We validate FetMRQC$_{SR}$ in an out-of-domain (OOD) setting and report high performance (ROC AUC = 0.89), even when faced with data from an unknown site or SRR method. We also investigate failure cases and show that they occur in $45\%$ of the images due to ambiguous configurations for which the rating from the expert is arguable. These results are encouraging and illustrate how a non deep learning-based method like FetMRQC$_{SR}$ is well suited to this multifaceted problem. Our tool, along with all the code used to generate, train and evaluate the model are available at https://github.com/Medical-Image-Analysis-Laboratory/fetmrqc_sr/ .
>
---
#### [replaced 012] Nonlinearity Enhanced Adaptive Activation Functions
- **分类: cs.LG; cs.CV; cs.NE**

- **链接: [http://arxiv.org/pdf/2403.19896v2](http://arxiv.org/pdf/2403.19896v2)**

> **作者:** David Yevick
>
> **摘要:** A general procedure for introducing parametric, learned, nonlinearity into activation functions is found to enhance the accuracy of representative neural networks without requiring significant additional computational resources. Examples are given based on the standard rectified linear unit (ReLU) as well as several other frequently employed activation functions. The associated accuracy improvement is quantified both in the context of the MNIST digit data set and a convolutional neural network (CNN) benchmark example.
>
---
#### [replaced 013] Schrödinger Diffusion Driven Signal Recovery in 3T BOLD fMRI Using Unmatched 7T Observations
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.01004v2](http://arxiv.org/pdf/2504.01004v2)**

> **作者:** Yujian Xiong; Xuanzhao Dong; Sebastian Waz; Wenhui Zhu; Negar Mallak; Zhong-lin Lu; Yalin Wang
>
> **摘要:** Ultra-high-field (7 Tesla) BOLD fMRI offers exceptional detail in both spatial and temporal domains, along with robust signal-to-noise characteristics, making it a powerful modality for studying visual information processing in the brain. However, due to the limited accessibility of 7T scanners, the majority of neuroimaging studies are still conducted using 3T systems, which inherently suffer from reduced fidelity in both resolution and SNR. To mitigate this limitation, we introduce a new computational approach designed to enhance the quality of 3T BOLD fMRI acquisitions. Specifically, we project both 3T and 7T datasets, sourced from different individuals and experimental setups, into a shared low-dimensional representation space. Within this space, we employ a lightweight, unsupervised Schr\"odinger Bridge framework to infer a high-SNR, high-resolution counterpart of the 3T data, without relying on paired supervision. This methodology is evaluated across multiple fMRI retinotopy datasets, including synthetically generated samples, and demonstrates a marked improvement in the reliability and fit of population receptive field (pRF) models applied to the enhanced 3T outputs. Our findings suggest that it is feasible to computationally approximate 7T-level quality from standard 3T acquisitions.
>
---
#### [replaced 014] MGPATH: Vision-Language Model with Multi-Granular Prompt Learning for Few-Shot WSI Classification
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.07409v2](http://arxiv.org/pdf/2502.07409v2)**

> **作者:** Anh-Tien Nguyen; Duy Minh Ho Nguyen; Nghiem Tuong Diep; Trung Quoc Nguyen; Nhat Ho; Jacqueline Michelle Metsch; Miriam Cindy Maurer; Daniel Sonntag; Hanibal Bohnenberger; Anne-Christin Hauschild
>
> **摘要:** Whole slide pathology image classification presents challenges due to gigapixel image sizes and limited annotation labels, hindering model generalization. This paper introduces a prompt learning method to adapt large vision-language models for few-shot pathology classification. We first extend the Prov-GigaPath vision foundation model, pre-trained on 1.3 billion pathology image tiles, into a vision-language model by adding adaptors and aligning it with medical text encoders via contrastive learning on 923K image-text pairs. The model is then used to extract visual features and text embeddings from few-shot annotations and fine-tunes with learnable prompt embeddings. Unlike prior methods that combine prompts with frozen features using prefix embeddings or self-attention, we propose multi-granular attention that compares interactions between learnable prompts with individual image patches and groups of them. This approach improves the model's ability to capture both fine-grained details and broader context, enhancing its recognition of complex patterns across sub-regions. To further improve accuracy, we leverage (unbalanced) optimal transport-based visual-text distance to secure model robustness by mitigating perturbations that might occur during the data augmentation process. Empirical experiments on lung, kidney, and breast pathology modalities validate the effectiveness of our approach; thereby, we surpass several of the latest competitors and consistently improve performance across diverse architectures, including CLIP, PLIP, and Prov-GigaPath integrated PLIP. We release our implementations and pre-trained models at this MGPATH.
>
---
#### [replaced 015] Diffusion-VLA: Scaling Robot Foundation Models via Unified Diffusion and Autoregression
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.03293v2](http://arxiv.org/pdf/2412.03293v2)**

> **作者:** Junjie Wen; Minjie Zhu; Yichen Zhu; Zhibin Tang; Jinming Li; Zhongyi Zhou; Chengmeng Li; Xiaoyu Liu; Yaxin Peng; Chaomin Shen; Feifei Feng
>
> **备注:** The project page is available at: http://diffusion-vla.github.io
>
> **摘要:** In this paper, we present DiffusionVLA, a novel framework that seamlessly combines the autoregression model with the diffusion model for learning visuomotor policy. Central to our approach is a next-token prediction objective, enabling the model to reason effectively over the user's query in the context of current observations. Subsequently, a diffusion model is attached to generate robust action outputs. To enhance policy learning through self-reasoning, we introduce a novel reasoning injection module that integrates reasoning phrases directly into the policy learning process. The whole framework is simple and flexible, making it easy to deploy and upgrade. We conduct extensive experiments using multiple real robots to validate the effectiveness of DiffusionVLA. Our tests include a challenging factory sorting task, where DiffusionVLA successfully categorizes objects, including those not seen during training. We observe that the reasoning module makes the model interpretable. It allows observers to understand the model thought process and identify potential causes of policy failures. Additionally, we test DiffusionVLA on a zero-shot bin-picking task, achieving 63.7\% accuracy on 102 previously unseen objects. Our method demonstrates robustness to visual changes, such as distractors and new backgrounds, and easily adapts to new embodiments. Furthermore, DiffusionVLA can follow novel instructions and retain conversational ability. Notably, DiffusionVLA is data-efficient and fast at inference; our smallest DiffusionVLA-2B runs 82Hz on a single A6000 GPU and can train from scratch on less than 50 demonstrations for a complex task. Finally, we scale the model from 2B to 72B parameters, showcasing improved generalization capabilities with increased model size.
>
---
#### [replaced 016] DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.05855v2](http://arxiv.org/pdf/2502.05855v2)**

> **作者:** Junjie Wen; Yichen Zhu; Jinming Li; Zhibin Tang; Chaomin Shen; Feifei Feng
>
> **备注:** The webpage is at https://dex-vla.github.io/
>
> **摘要:** Enabling robots to perform diverse tasks across varied environments is a central challenge in robot learning. While vision-language-action (VLA) models have shown promise for generalizable robot skills, realizing their full potential requires addressing limitations in action representation and efficient training. Current VLA models often focus on scaling the vision-language model (VLM) component, while the action space representation remains a critical bottleneck. This paper introduces DexVLA, a novel framework designed to enhance the efficiency and generalization capabilities of VLAs for complex, long-horizon tasks across diverse robot embodiments. DexVLA features a novel diffusion-based action expert, scaled to one billion parameters, designed for cross-embodiment learning. A novel embodiment curriculum learning strategy facilitates efficient training: (1) pre-training the diffusion expert that is separable from the VLA on cross-embodiment data, (2) aligning the VLA model to specific embodiments, and (3) post-training for rapid adaptation to new tasks. We conduct comprehensive experiments across multiple embodiments, including single-arm, bimanual, and dexterous hand, demonstrating DexVLA's adaptability to challenging tasks without task-specific adaptation, its ability to learn dexterous skills on novel embodiments with limited data, and its capacity to complete complex, long-horizon tasks using only direct language prompting, such as laundry folding. In all settings, our method demonstrates superior performance compared to state-of-the-art models like Octo, OpenVLA, and Diffusion Policy.
>
---
#### [replaced 017] Benchmarking Multimodal Mathematical Reasoning with Explicit Visual Dependency
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.18589v4](http://arxiv.org/pdf/2504.18589v4)**

> **作者:** Zhikai Wang; Jiashuo Sun; Wenqi Zhang; Zhiqiang Hu; Xin Li; Fan Wang; Deli Zhao
>
> **备注:** Home page: https://alibaba-damo-academy.github.io/VCBench/
>
> **摘要:** Recent advancements in Large Vision-Language Models (LVLMs) have significantly enhanced their ability to integrate visual and linguistic information, achieving near-human proficiency in tasks like object recognition, captioning, and visual question answering. However, current benchmarks typically focus on knowledge-centric evaluations that assess domain-specific expertise, often neglecting the core ability to reason about fundamental mathematical elements and visual concepts. We identify a gap in evaluating elementary-level math problems, which rely on explicit visual dependencies-requiring models to discern, integrate, and reason across multiple images while incorporating commonsense knowledge, all of which are crucial for advancing toward broader AGI capabilities. To address this gap, we introduce VCBENCH, a comprehensive benchmark for multimodal mathematical reasoning with explicit visual dependencies. VCBENCH includes 1,720 problems across six cognitive domains, featuring 6,697 images (averaging 3.9 per question) to ensure multi-image reasoning. We evaluate 26 state-of-the-art LVLMs on VCBENCH, revealing substantial performance disparities, with even the top models unable to exceed 50% accuracy. Our findings highlight the ongoing challenges in visual-mathematical integration and suggest avenues for future LVLM advancements. The project can be found at https://alibaba-damo-academy.github.io/VCBench/.
>
---
#### [replaced 018] Inter-event Interval Microscopy for Event Cameras
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2504.04924v3](http://arxiv.org/pdf/2504.04924v3)**

> **作者:** Changqing Su; Yanqin Chen; Zihan Lin; Zhen Cheng; You Zhou; Bo Xiong; Zhaofei Yu; Tiejun Huang
>
> **摘要:** Event cameras, an innovative bio-inspired sensor, differ from traditional cameras by sensing changes in intensity rather than directly perceiving intensity and recording these variations as a continuous stream of "events". The intensity reconstruction from these sparse events has long been a challenging problem. Previous approaches mainly focused on transforming motion-induced events into videos or achieving intensity imaging for static scenes by integrating modulation devices at the event camera acquisition end. In this paper, for the first time, we achieve event-to-intensity conversion using a static event camera for both static and dynamic scenes in fluorescence microscopy. Unlike conventional methods that primarily rely on event integration, the proposed Inter-event Interval Microscopy (IEIM) quantifies the time interval between consecutive events at each pixel. With a fixed threshold in the event camera, the time interval can precisely represent the intensity. At the hardware level, the proposed IEIM integrates a pulse light modulation device within a microscope equipped with an event camera, termed Pulse Modulation-based Event-driven Fluorescence Microscopy. Additionally, we have collected IEIMat dataset under various scenes including high dynamic range and high-speed scenarios. Experimental results on the IEIMat dataset demonstrate that the proposed IEIM achieves superior spatial and temporal resolution, as well as a higher dynamic range, with lower bandwidth compared to other methods. The code and the IEIMat dataset will be made publicly available.
>
---
#### [replaced 019] FG-CLIP: Fine-Grained Visual and Textual Alignment
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.05071v2](http://arxiv.org/pdf/2505.05071v2)**

> **作者:** Chunyu Xie; Bin Wang; Fanjing Kong; Jincheng Li; Dawei Liang; Gengshen Zhang; Dawei Leng; Yuhui Yin
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** Contrastive Language-Image Pre-training (CLIP) excels in multimodal tasks such as image-text retrieval and zero-shot classification but struggles with fine-grained understanding due to its focus on coarse-grained short captions. To address this, we propose Fine-Grained CLIP (FG-CLIP), which enhances fine-grained understanding through three key innovations. First, we leverage large multimodal models to generate 1.6 billion long caption-image pairs for capturing global-level semantic details. Second, a high-quality dataset is constructed with 12 million images and 40 million region-specific bounding boxes aligned with detailed captions to ensure precise, context-rich representations. Third, 10 million hard fine-grained negative samples are incorporated to improve the model's ability to distinguish subtle semantic differences. We construct a comprehensive dataset, termed FgGRN, by integrating high-quality region-specific annotations with challenging fine-grained negative samples. Corresponding training methods are meticulously designed for these data. Extensive experiments demonstrate that FG-CLIP outperforms the original CLIP and other state-of-the-art methods across various downstream tasks, including fine-grained understanding, open-vocabulary object detection, image-text retrieval, and general multimodal benchmarks. These results highlight FG-CLIP's effectiveness in capturing fine-grained image details and improving overall model performance. The related data, code, and models are available at https://github.com/360CVGroup/FG-CLIP.
>
---
#### [replaced 020] Unsupervised Urban Land Use Mapping with Street View Contrastive Clustering and a Geographical Prior
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.17551v2](http://arxiv.org/pdf/2504.17551v2)**

> **作者:** Lin Che; Yizi Chen; Tanhua Jin; Martin Raubal; Konrad Schindler; Peter Kiefer
>
> **备注:** 11 pages, 7 figures, preprint version
>
> **摘要:** Urban land use classification and mapping are critical for urban planning, resource management, and environmental monitoring. Existing remote sensing techniques often lack precision in complex urban environments due to the absence of ground-level details. Unlike aerial perspectives, street view images provide a ground-level view that captures more human and social activities relevant to land use in complex urban scenes. Existing street view-based methods primarily rely on supervised classification, which is challenged by the scarcity of high-quality labeled data and the difficulty of generalizing across diverse urban landscapes. This study introduces an unsupervised contrastive clustering model for street view images with a built-in geographical prior, to enhance clustering performance. When combined with a simple visual assignment of the clusters, our approach offers a flexible and customizable solution to land use mapping, tailored to the specific needs of urban planners. We experimentally show that our method can generate land use maps from geotagged street view image datasets of two cities. As our methodology relies on the universal spatial coherence of geospatial data ("Tobler's law"), it can be adapted to various settings where street view images are available, to enable scalable, unsupervised land use mapping and updating. The code will be available at https://github.com/lin102/CCGP.
>
---
#### [replaced 021] TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.12514v5](http://arxiv.org/pdf/2409.12514v5)**

> **作者:** Junjie Wen; Yichen Zhu; Jinming Li; Minjie Zhu; Kun Wu; Zhiyuan Xu; Ning Liu; Ran Cheng; Chaomin Shen; Yaxin Peng; Feifei Feng; Jian Tang
>
> **备注:** add more citations
>
> **摘要:** Vision-Language-Action (VLA) models have shown remarkable potential in visuomotor control and instruction comprehension through end-to-end learning processes. However, current VLA models face significant challenges: they are slow during inference and require extensive pre-training on large amounts of robotic data, making real-world deployment difficult. In this paper, we introduce a new family of compact vision-language-action models, called TinyVLA, which offers two key advantages over existing VLA models: (1) faster inference speeds, and (2) improved data efficiency, eliminating the need for pre-training stage. Our framework incorporates two essential components to build TinyVLA: (1) initializing the policy backbone with robust, high-speed multimodal models, and (2) integrating a diffusion policy decoder during fine-tuning to enable precise robot actions. We conducted extensive evaluations of TinyVLA in both simulation and on real robots, demonstrating that our approach significantly outperforms the state-of-the-art VLA model, OpenVLA, in terms of speed and data efficiency, while delivering comparable or superior performance. Additionally, TinyVLA exhibits strong generalization capabilities across various dimensions, including language instructions, novel objects, unseen positions, changes in object appearance, background variations, and environmental shifts, often matching or exceeding the performance of OpenVLA. We believe that \methodname offers an interesting perspective on utilizing pre-trained multimodal models for policy learning. Our project is at https://tiny-vla.github.io.
>
---
#### [replaced 022] HoloTime: Taming Video Diffusion Models for Panoramic 4D Scene Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.21650v2](http://arxiv.org/pdf/2504.21650v2)**

> **作者:** Haiyang Zhou; Wangbo Yu; Jiawen Guan; Xinhua Cheng; Yonghong Tian; Li Yuan
>
> **备注:** Project Homepage: https://zhouhyocean.github.io/holotime/ Code: https://github.com/PKU-YuanGroup/HoloTime
>
> **摘要:** The rapid advancement of diffusion models holds the promise of revolutionizing the application of VR and AR technologies, which typically require scene-level 4D assets for user experience. Nonetheless, existing diffusion models predominantly concentrate on modeling static 3D scenes or object-level dynamics, constraining their capacity to provide truly immersive experiences. To address this issue, we propose HoloTime, a framework that integrates video diffusion models to generate panoramic videos from a single prompt or reference image, along with a 360-degree 4D scene reconstruction method that seamlessly transforms the generated panoramic video into 4D assets, enabling a fully immersive 4D experience for users. Specifically, to tame video diffusion models for generating high-fidelity panoramic videos, we introduce the 360World dataset, the first comprehensive collection of panoramic videos suitable for downstream 4D scene reconstruction tasks. With this curated dataset, we propose Panoramic Animator, a two-stage image-to-video diffusion model that can convert panoramic images into high-quality panoramic videos. Following this, we present Panoramic Space-Time Reconstruction, which leverages a space-time depth estimation method to transform the generated panoramic videos into 4D point clouds, enabling the optimization of a holistic 4D Gaussian Splatting representation to reconstruct spatially and temporally consistent 4D scenes. To validate the efficacy of our method, we conducted a comparative analysis with existing approaches, revealing its superiority in both panoramic video generation and 4D scene reconstruction. This demonstrates our method's capability to create more engaging and realistic immersive environments, thereby enhancing user experiences in VR and AR applications.
>
---
#### [replaced 023] GBT-SAM: Adapting a Foundational Deep Learning Model for Generalizable Brain Tumor Segmentation via Efficient Integration of Multi-Parametric MRI Data
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.04325v3](http://arxiv.org/pdf/2503.04325v3)**

> **作者:** Cecilia Diana-Albelda; Roberto Alcover-Couso; Álvaro García-Martín; Jesus Bescos; Marcos Escudero-Viñolo
>
> **摘要:** Gliomas are aggressive brain tumors that require accurate imaging-based diagnosis, with segmentation playing a critical role in evaluating morphology and treatment decisions. Manual delineation of gliomas is time-consuming and prone to variability, motivating the use of deep learning to improve consistency and alleviate clinical workload. However, existing methods often fail to fully exploit the information available in multi-parametric MRI (mp-MRI), particularly inter-slice contextual features, and typically require considerable computational resources while lacking robustness across tumor type variations. We present GBT-SAM, a parameter-efficient deep learning framework that adapts the Segment Anything Model (SAM), a large-scale vision model, to volumetric mp-MRI data. GBT-SAM reduces input complexity by selecting fewer than 2.6\% of slices per scan while incorporating all four MRI modalities, preserving essential tumor-related information with minimal cost. Furthermore, our model is trained by a two-step fine-tuning strategy that incorporates a depth-aware module to capture inter-slice correlations and lightweight adaptation layers, resulting in just 6.5M trainable parameters, which is the lowest among SAM-based approaches. GBT-SAM achieves a Dice Score of 93.54 on the BraTS Adult Glioma dataset and demonstrates robust performance on Meningioma, Pediatric Glioma, and Sub-Saharan Glioma datasets. These results highlight GBT-SAM's potential as a computationally efficient and domain-robust framework for brain tumor segmentation using mp-MRI. Our code and models are available at https://github.com/vpulab/med-sam-brain .
>
---
#### [replaced 024] Equipping Sketch Patches with Context-Aware Positional Encoding for Graphic Sketch Representation
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2403.17525v2](http://arxiv.org/pdf/2403.17525v2)**

> **作者:** Sicong Zang; Zhijun Fang
>
> **摘要:** When benefiting graphic sketch representation with sketch drawing orders, recent studies have linked sketch patches as graph edges by drawing orders in accordance to a temporal-based nearest neighboring strategy. However, such constructed graph edges may be unreliable, since the contextual relationships between patches may be inconsistent with the sequential positions in drawing orders, due to variants of sketch drawings. In this paper, we propose a variant-drawing-protected method by equipping sketch patches with context-aware positional encoding (PE) to make better use of drawing orders for sketch learning. We introduce a sinusoidal absolute PE to embed the sequential positions in drawing orders, and a learnable relative PE to encode the unseen contextual relationships between patches. Both types of PEs never attend the construction of graph edges, but are injected into graph nodes to cooperate with the visual patterns captured from patches. After linking nodes by semantic proximity, during message aggregation via graph convolutional networks, each node receives both semantic features from patches and contextual information from PEs from its neighbors, which equips local patch patterns with global contextual information, further obtaining drawing-order-enhanced sketch representations. Experimental results indicate that our method significantly improves sketch healing and controllable sketch synthesis. The source codes could be found at https://github.com/SCZang/DC-gra2seq.
>
---
#### [replaced 025] Efficient Adaptation For Remote Sensing Visual Grounding
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.23083v2](http://arxiv.org/pdf/2503.23083v2)**

> **作者:** Hasan Moughnieh; Mohamad Chalhoub; Hasan Nasrallah; Cristiano Nattero; Paolo Campanella; Giovanni Nico; Ali J. Ghandour
>
> **摘要:** Adapting pre-trained models has become an effective strategy in artificial intelligence, offering a scalable and efficient alternative to training models from scratch. In the context of remote sensing (RS), where visual grounding(VG) remains underexplored, this approach enables the deployment of powerful vision-language models to achieve robust cross-modal understanding while significantly reducing computational overhead. To address this, we applied Parameter Efficient Fine Tuning (PEFT) techniques to adapt these models for RS-specific VG tasks. Specifically, we evaluated LoRA placement across different modules in Grounding DINO and used BitFit and adapters to fine-tune the OFA foundation model pre-trained on general-purpose VG datasets. This approach achieved performance comparable to or surpassing current State Of The Art (SOTA) models while significantly reducing computational costs. This study highlights the potential of PEFT techniques to advance efficient and precise multi-modal analysis in RS, offering a practical and cost-effective alternative to full model training.
>
---
#### [replaced 026] Brain Hematoma Marker Recognition Using Multitask Learning: SwinTransformer and Swin-Unet
- **分类: cs.LG; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.06185v2](http://arxiv.org/pdf/2505.06185v2)**

> **作者:** Kodai Hirata; Tsuyoshi Okita
>
> **备注:** 8 pages,4 figures
>
> **摘要:** This paper proposes a method MTL-Swin-Unet which is multi-task learning using transformers for classification and semantic segmentation. For spurious-correlation problems, this method allows us to enhance the image representation with two other image representations: representation obtained by semantic segmentation and representation obtained by image reconstruction. In our experiments, the proposed method outperformed in F-value measure than other classifiers when the test data included slices from the same patient (no covariate shift). Similarly, when the test data did not include slices from the same patient (covariate shift setting), the proposed method outperformed in AUC measure.
>
---
#### [replaced 027] Geometry-Aware Feature Matching for Large-Scale Structure from Motion
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2409.02310v4](http://arxiv.org/pdf/2409.02310v4)**

> **作者:** Gonglin Chen; Jinsen Wu; Haiwei Chen; Wenbin Teng; Zhiyuan Gao; Andrew Feng; Rongjun Qin; Yajie Zhao
>
> **摘要:** Establishing consistent and dense correspondences across multiple images is crucial for Structure from Motion (SfM) systems. Significant view changes, such as air-to-ground with very sparse view overlap, pose an even greater challenge to the correspondence solvers. We present a novel optimization-based approach that significantly enhances existing feature matching methods by introducing geometry cues in addition to color cues. This helps fill gaps when there is less overlap in large-scale scenarios. Our method formulates geometric verification as an optimization problem, guiding feature matching within detector-free methods and using sparse correspondences from detector-based methods as anchor points. By enforcing geometric constraints via the Sampson Distance, our approach ensures that the denser correspondences from detector-free methods are geometrically consistent and more accurate. This hybrid strategy significantly improves correspondence density and accuracy, mitigates multi-view inconsistencies, and leads to notable advancements in camera pose accuracy and point cloud density. It outperforms state-of-the-art feature matching methods on benchmark datasets and enables feature matching in challenging extreme large-scale settings.
>
---
#### [replaced 028] TUM2TWIN: Introducing the Large-Scale Multimodal Urban Digital Twin Benchmark Dataset
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.07396v2](http://arxiv.org/pdf/2505.07396v2)**

> **作者:** Olaf Wysocki; Benedikt Schwab; Manoj Kumar Biswanath; Michael Greza; Qilin Zhang; Jingwei Zhu; Thomas Froech; Medhini Heeramaglore; Ihab Hijazi; Khaoula Kanna; Mathias Pechinger; Zhaiyu Chen; Yao Sun; Alejandro Rueda Segura; Ziyang Xu; Omar AbdelGafar; Mansour Mehranfar; Chandan Yeshwanth; Yueh-Cheng Liu; Hadi Yazdi; Jiapan Wang; Stefan Auer; Katharina Anders; Klaus Bogenberger; Andre Borrmann; Angela Dai; Ludwig Hoegner; Christoph Holst; Thomas H. Kolbe; Ferdinand Ludwig; Matthias Nießner; Frank Petzold; Xiao Xiang Zhu; Boris Jutzi
>
> **备注:** Submitted to the ISPRS Journal of Photogrammetry and Remote Sensing
>
> **摘要:** Urban Digital Twins (UDTs) have become essential for managing cities and integrating complex, heterogeneous data from diverse sources. Creating UDTs involves challenges at multiple process stages, including acquiring accurate 3D source data, reconstructing high-fidelity 3D models, maintaining models' updates, and ensuring seamless interoperability to downstream tasks. Current datasets are usually limited to one part of the processing chain, hampering comprehensive UDTs validation. To address these challenges, we introduce the first comprehensive multimodal Urban Digital Twin benchmark dataset: TUM2TWIN. This dataset includes georeferenced, semantically aligned 3D models and networks along with various terrestrial, mobile, aerial, and satellite observations boasting 32 data subsets over roughly 100,000 $m^2$ and currently 767 GB of data. By ensuring georeferenced indoor-outdoor acquisition, high accuracy, and multimodal data integration, the benchmark supports robust analysis of sensors and the development of advanced reconstruction methods. Additionally, we explore downstream tasks demonstrating the potential of TUM2TWIN, including novel view synthesis of NeRF and Gaussian Splatting, solar potential analysis, point cloud semantic segmentation, and LoD3 building reconstruction. We are convinced this contribution lays a foundation for overcoming current limitations in UDT creation, fostering new research directions and practical solutions for smarter, data-driven urban environments. The project is available under: https://tum2t.win
>
---
#### [replaced 029] 2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.00958v4](http://arxiv.org/pdf/2501.00958v4)**

> **作者:** Wenqi Zhang; Hang Zhang; Xin Li; Jiashuo Sun; Yongliang Shen; Weiming Lu; Deli Zhao; Yueting Zhuang; Lidong Bing
>
> **备注:** Under review
>
> **摘要:** Compared to image-text pair data, interleaved corpora enable Vision-Language Models (VLMs) to understand the world more naturally like humans. However, such existing datasets are crawled from webpage, facing challenges like low knowledge density, loose image-text relations, and poor logical coherence between images. On the other hand, the internet hosts vast instructional videos (e.g., online geometry courses) that are widely used by humans to learn foundational subjects, yet these valuable resources remain underexplored in VLM training. In this paper, we introduce a high-quality \textbf{multimodal textbook} corpus with richer foundational knowledge for VLM pretraining. It collects over 2.5 years of instructional videos, totaling 22,000 class hours. We first use an LLM-proposed taxonomy to systematically gather instructional videos. Then we progressively extract and refine visual (keyframes), audio (ASR), and textual knowledge (OCR) from the videos, and organize as an image-text interleaved corpus based on temporal order. Compared to its counterparts, our video-centric textbook offers more coherent context, richer knowledge, and better image-text alignment. Experiments demonstrate its superb pretraining performance, particularly in knowledge- and reasoning-intensive tasks like ScienceQA and MathVista. Moreover, VLMs pre-trained on our textbook exhibit outstanding interleaved context awareness, leveraging visual and textual cues in their few-shot context for task solving. Our code are available at https://github.com/DAMO-NLP-SG/multimodal_textbook.
>
---
#### [replaced 030] HLV-1K: A Large-scale Hour-Long Video Benchmark for Time-Specific Long Video Understanding
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.01645v3](http://arxiv.org/pdf/2501.01645v3)**

> **作者:** Heqing Zou; Tianze Luo; Guiyang Xie; Victor Xiao Jie Zhang; Fengmao Lv; Guangcong Wang; Junyang Chen; Zhuochen Wang; Hansheng Zhang; Huaijian Zhang
>
> **备注:** Accepted to ICME 2025
>
> **摘要:** Multimodal large language models have become a popular topic in deep visual understanding due to many promising real-world applications. However, hour-long video understanding, spanning over one hour and containing tens of thousands of visual frames, remains under-explored because of 1) challenging long-term video analyses, 2) inefficient large-model approaches, and 3) lack of large-scale benchmark datasets. Among them, in this paper, we focus on building a large-scale hour-long long video benchmark, HLV-1K, designed to evaluate long video understanding models. HLV-1K comprises 1009 hour-long videos with 14,847 high-quality question answering (QA) and multi-choice question asnwering (MCQA) pairs with time-aware query and diverse annotations, covering frame-level, within-event-level, cross-event-level, and long-term reasoning tasks. We evaluate our benchmark using existing state-of-the-art methods and demonstrate its value for testing deep long video understanding capabilities at different levels and for various tasks. This includes promoting future long video understanding tasks at a granular level, such as deep understanding of long live videos, meeting recordings, and movies.
>
---
#### [replaced 031] Calibrated and Efficient Sampling-Free Confidence Estimation for LiDAR Scene Semantic Segmentation
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.11935v2](http://arxiv.org/pdf/2411.11935v2)**

> **作者:** Hanieh Shojaei Miandashti; Qianqian Zou; Claus Brenner
>
> **摘要:** Reliable deep learning models require not only accurate predictions but also well-calibrated confidence estimates to ensure dependable uncertainty estimation. This is crucial in safety-critical applications like autonomous driving, which depend on rapid and precise semantic segmentation of LiDAR point clouds for real-time 3D scene understanding. In this work, we introduce a sampling-free approach for estimating well-calibrated confidence values for classification tasks, achieving alignment with true classification accuracy and significantly reducing inference time compared to sampling-based methods. Our evaluation using the Adaptive Calibration Error (ACE) metric for LiDAR semantic segmentation shows that our approach maintains well-calibrated confidence values while achieving increased processing speed compared to a sampling baseline. Additionally, reliability diagrams reveal that our method produces underconfidence rather than overconfident predictions, an advantage for safety-critical applications. Our sampling-free approach offers well-calibrated and time-efficient predictions for LiDAR scene semantic segmentation.
>
---
#### [replaced 032] CHOICE: Benchmarking the Remote Sensing Capabilities of Large Vision-Language Models
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2411.18145v3](http://arxiv.org/pdf/2411.18145v3)**

> **作者:** Xiao An; Jiaxing Sun; Zihan Gui; Wei He
>
> **备注:** 32 pages, 15 figures
>
> **摘要:** The rapid advancement of Large Vision-Language Models (VLMs), both general-domain models and those specifically tailored for remote sensing, has demonstrated exceptional perception and reasoning capabilities in Earth observation tasks. However, a benchmark for systematically evaluating their capabilities in this domain is still lacking. To bridge this gap, we propose CHOICE, an extensive benchmark designed to objectively evaluate the hierarchical remote sensing capabilities of VLMs. Focusing on 2 primary capability dimensions essential to remote sensing: perception and reasoning, we further categorize 6 secondary dimensions and 23 leaf tasks to ensure a well-rounded assessment coverage. CHOICE guarantees the quality of all 10,507 problems through a rigorous process of data collection from 50 globally distributed cities, question construction and quality control. The newly curated data and the format of multiple-choice questions with definitive answers allow for an objective and straightforward performance assessment. Our evaluation of 3 proprietary and 21 open-source VLMs highlights their critical limitations within this specialized context. We hope that CHOICE will serve as a valuable resource and offer deeper insights into the challenges and potential of VLMs in the field of remote sensing. We will release CHOICE at https://github.com/ShawnAn-WHU/CHOICE.
>
---
#### [replaced 033] Deep Representation Learning for Unsupervised Clustering of Myocardial Fiber Trajectories in Cardiac Diffusion Tensor Imaging
- **分类: eess.IV; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.01953v2](http://arxiv.org/pdf/2504.01953v2)**

> **作者:** Mohini Anand; Xavier Tricoche
>
> **备注:** 10 pages, 5 figures. An extended journal manuscript is in preparation
>
> **摘要:** Understanding the complex myocardial architecture is critical for diagnosing and treating heart disease. However, existing methods often struggle to accurately capture this intricate structure from Diffusion Tensor Imaging (DTI) data, particularly due to the lack of ground truth labels and the ambiguous, intertwined nature of fiber trajectories. We present a novel deep learning framework for unsupervised clustering of myocardial fibers, providing a data-driven approach to identifying distinct fiber bundles. We uniquely combine a Bidirectional Long Short-Term Memory network to capture local sequential information along fibers, with a Transformer autoencoder to learn global shape features, with pointwise incorporation of essential anatomical context. Clustering these representations using a density-based algorithm identifies 33 to 62 robust clusters, successfully capturing the subtle distinctions in fiber trajectories with varying levels of granularity. Our framework offers a new, flexible, and quantitative way to analyze myocardial structure, achieving a level of delineation that, to our knowledge, has not been previously achieved, with potential applications in improving surgical planning, characterizing disease-related remodeling, and ultimately, advancing personalized cardiac care.
>
---
#### [replaced 034] Deep learning-based interactive segmentation in remote sensing
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2308.13174v3](http://arxiv.org/pdf/2308.13174v3)**

> **作者:** Zhe Wang; Shoukun Sun; Xiang Que; Xiaogang Ma; Carmen Galaz Garcia
>
> **摘要:** Interactive segmentation, a computer vision technique where a user provides guidance to help an algorithm segment a feature of interest in an image, has achieved outstanding accuracy and efficient human-computer interaction. However, few studies have discussed its application to remote sensing imagery, where click-based interactive segmentation could greatly facilitate the analysis of complicated landscapes. This study aims to bridge the gap between click-based interactive segmentation and remote sensing image analysis by conducting a benchmark study on various click-based interactive segmentation models. We assessed the performance of five state-of-the-art interactive segmentation methods (Reviving Iterative Training with Mask Guidance for Interactive Segmentation (RITM), FocalClick, SimpleClick, Iterative Click Loss (ICL), and Segment Anything (SAM)) on two high-resolution aerial imagery datasets. The Cascade-Forward Refinement (CFR) approach, an innovative inference strategy for interactive segmentation, was also introduced to enhance the segmentation results without requiring manual efforts. We further integrated CFR into all models for comparison. The performance of these methods on various land cover types, different object sizes, and multiple band combinations in the datasets was evaluated. The SimpleClick-CFR model consistently outperformed the other methods in our experiments. Building upon these findings, we developed a dedicated online tool called SegMap for interactive segmentation of remote sensing data. SegMap incorporates a well-performing interactive model that is fine-tuned with remote sensing data. Unlike existing interactive segmentation tools, SegMap offers robust interactivity, modifiability, and adaptability to analyze remote sensing imagery.
>
---
#### [replaced 035] DiTPainter: Efficient Video Inpainting with Diffusion Transformers
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.15661v2](http://arxiv.org/pdf/2504.15661v2)**

> **作者:** Xian Wu; Chang Liu
>
> **摘要:** Many existing video inpainting algorithms utilize optical flows to construct the corresponding maps and then propagate pixels from adjacent frames to missing areas by mapping. Despite the effectiveness of the propagation mechanism, they might encounter blurry and inconsistencies when dealing with inaccurate optical flows or large masks. Recently, Diffusion Transformer (DiT) has emerged as a revolutionary technique for video generation tasks. However, pretrained DiT models for video generation all contain a large amount of parameters, which makes it very time consuming to apply to video inpainting tasks. In this paper, we present DiTPainter, an end-to-end video inpainting model based on Diffusion Transformer (DiT). DiTPainter uses an efficient transformer network designed for video inpainting, which is trained from scratch instead of initializing from any large pretrained models. DiTPainter can address videos with arbitrary lengths and can be applied to video decaptioning and video completion tasks with an acceptable time cost. Experiments show that DiTPainter outperforms existing video inpainting algorithms with higher quality and better spatial-temporal consistency.
>
---
#### [replaced 036] Vision-Language Models Do Not Understand Negation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.09425v2](http://arxiv.org/pdf/2501.09425v2)**

> **作者:** Kumail Alhamoud; Shaden Alshammari; Yonglong Tian; Guohao Li; Philip Torr; Yoon Kim; Marzyeh Ghassemi
>
> **备注:** CVPR 2025; project page: https://negbench.github.io
>
> **摘要:** Many practical vision-language applications require models that understand negation, e.g., when using natural language to retrieve images which contain certain objects but not others. Despite advancements in vision-language models (VLMs) through large-scale training, their ability to comprehend negation remains underexplored. This study addresses the question: how well do current VLMs understand negation? We introduce NegBench, a new benchmark designed to evaluate negation understanding across 18 task variations and $79$k examples spanning image, video, and medical datasets. The benchmark consists of two core tasks designed to evaluate negation understanding in diverse multimodal settings: Retrieval with Negation and Multiple Choice Questions with Negated Captions. Our evaluation reveals that modern VLMs struggle significantly with negation, often performing at chance level. To address these shortcomings, we explore a data-centric approach wherein we finetune CLIP models on large-scale synthetic datasets containing millions of negated captions. We show that this approach can result in a 10% increase in recall on negated queries and a 28% boost in accuracy on multiple-choice questions with negated captions.
>
---
#### [replaced 037] Using Few-Shot Learning to Classify Primary Lung Cancer and Other Malignancy with Lung Metastasis in Cytological Imaging via Endobronchial Ultrasound Procedures
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2404.06080v3](http://arxiv.org/pdf/2404.06080v3)**

> **作者:** Ching-Kai Lin; Di-Chun Wei; Yun-Chien Cheng
>
> **摘要:** This study presents a computer-aided diagnosis (CAD) system to assist early detection of lung metastases during endobronchial ultrasound (EBUS) procedures, significantly reducing follow-up time and enabling timely treatment. Due to limited cytology images and morphological similarities among cells, classifying lung metastases is challenging, and existing research rarely targets this issue directly.To overcome data scarcity and improve classification, the authors propose a few-shot learning model using a hybrid pretrained backbone with fine-grained classification and contrastive learning. Parameter-efficient fine-tuning on augmented support sets enhances generalization and transferability. The model achieved 49.59% accuracy, outperforming existing methods. With 20 image samples, accuracy improved to 55.48%, showing strong potential for identifying rare or novel cancer types in low-data clinical environments.
>
---
#### [replaced 038] FANeRV: Frequency Separation and Augmentation based Neural Representation for Video
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.06755v4](http://arxiv.org/pdf/2504.06755v4)**

> **作者:** Li Yu; Zhihui Li; Chao Yao; Jimin Xiao; Moncef Gabbouj
>
> **摘要:** Neural representations for video (NeRV) have gained considerable attention for their strong performance across various video tasks. However, existing NeRV methods often struggle to capture fine spatial details, resulting in vague reconstructions. In this paper, we present a Frequency Separation and Augmentation based Neural Representation for video (FANeRV), which addresses these limitations with its core Wavelet Frequency Upgrade Block. This block explicitly separates input frames into high and low-frequency components using discrete wavelet transform, followed by targeted enhancement using specialized modules. Finally, a specially designed gated network effectively fuses these frequency components for optimal reconstruction. Additionally, convolutional residual enhancement blocks are integrated into the later stages of the network to balance parameter distribution and improve the restoration of high-frequency details. Experimental results demonstrate that FANeRV significantly improves reconstruction performance and excels in multiple tasks, including video compression, inpainting, and interpolation, outperforming existing NeRV methods.
>
---
#### [replaced 039] Towards Anytime Optical Flow Estimation with Event Cameras
- **分类: cs.CV; cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2307.05033v3](http://arxiv.org/pdf/2307.05033v3)**

> **作者:** Yaozu Ye; Hao Shi; Kailun Yang; Ze Wang; Xiaoting Yin; Lei Sun; Yaonan Wang; Kaiwei Wang
>
> **备注:** Accepted to Sensors. Our code will be available at https://github.com/Yaozhuwa/EVA-Flow
>
> **摘要:** Event cameras respond to changes in log-brightness at the millisecond level, making them ideal for optical flow estimation. However, existing datasets from event cameras provide only low frame rate ground truth for optical flow, limiting the research potential of event-driven optical flow. To address this challenge, we introduce a low-latency event representation, Unified Voxel Grid, and propose EVA-Flow, an EVent-based Anytime Flow estimation network to produce high-frame-rate event optical flow with only low-frame-rate optical flow ground truth for supervision. Furthermore, we propose the Rectified Flow Warp Loss (RFWL) for the unsupervised assessment of intermediate optical flow. A comprehensive variety of experiments on MVSEC, DESC, and our EVA-FlowSet demonstrates that EVA-Flow achieves competitive performance, super-low-latency (5ms), time-dense motion estimation (200Hz), and strong generalization. Our code will be available at https://github.com/Yaozhuwa/EVA-Flow.
>
---
#### [replaced 040] Training Strategies for Isolated Sign Language Recognition
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2412.11553v2](http://arxiv.org/pdf/2412.11553v2)**

> **作者:** Karina Kvanchiani; Roman Kraynov; Elizaveta Petrova; Petr Surovcev; Aleksandr Nagaev; Alexander Kapitanov
>
> **备注:** sign language recognition, training strategies, computer vision, isolated sign language recognition
>
> **摘要:** Accurate recognition and interpretation of sign language are crucial for enhancing communication accessibility for deaf and hard of hearing individuals. However, current approaches of Isolated Sign Language Recognition (ISLR) often face challenges such as low data quality and variability in gesturing speed. This paper introduces a comprehensive model training pipeline for ISLR designed to accommodate the distinctive characteristics and constraints of the Sign Language (SL) domain. The constructed pipeline incorporates carefully selected image and video augmentations to tackle the challenges of low data quality and varying sign speeds. Including an additional regression head combined with IoU-balanced classification loss enhances the model's awareness of the gesture and simplifies capturing temporal information. Extensive experiments demonstrate that the developed training pipeline easily adapts to different datasets and architectures. Additionally, the ablation study shows that each proposed component expands the potential to consider ISLR task specifics. The presented strategies enhance recognition performance across various ISLR benchmarks and achieve state-of-the-art results on the WLASL and Slovo datasets.
>
---
#### [replaced 041] Optimized View and Geometry Distillation from Multi-view Diffuser
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2312.06198v4](http://arxiv.org/pdf/2312.06198v4)**

> **作者:** Youjia Zhang; Zikai Song; Junqing Yu; Yawei Luo; Wei Yang
>
> **备注:** IJCAI 2025. Project page: https://youjiazhang.github.io/USD/
>
> **摘要:** Generating multi-view images from a single input view using image-conditioned diffusion models is a recent advancement and has shown considerable potential. However, issues such as the lack of consistency in synthesized views and over-smoothing in extracted geometry persist. Previous methods integrate multi-view consistency modules or impose additional supervisory to enhance view consistency while compromising on the flexibility of camera positioning and limiting the versatility of view synthesis. In this study, we consider the radiance field optimized during geometry extraction as a more rigid consistency prior, compared to volume and ray aggregation used in previous works. We further identify and rectify a critical bias in the traditional radiance field optimization process through score distillation from a multi-view diffuser. We introduce an Unbiased Score Distillation (USD) that utilizes unconditioned noises from a 2D diffusion model, greatly refining the radiance field fidelity. We leverage the rendered views from the optimized radiance field as the basis and develop a two-step specialization process of a 2D diffusion model, which is adept at conducting object-specific denoising and generating high-quality multi-view images. Finally, we recover faithful geometry and texture directly from the refined multi-view images. Empirical evaluations demonstrate that our optimized geometry and view distillation technique generates comparable results to the state-of-the-art models trained on extensive datasets, all while maintaining freedom in camera positioning. Please see our project page at https://youjiazhang.github.io/USD/.
>
---
#### [replaced 042] High-Quality Spatial Reconstruction and Orthoimage Generation Using Efficient 2D Gaussian Splatting
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2503.19703v2](http://arxiv.org/pdf/2503.19703v2)**

> **作者:** Qian Wang; Zhihao Zhan; Jialei He; Zhituo Tu; Xiang Zhu; Jie Yuan
>
> **摘要:** Highly accurate geometric precision and dense image features characterize True Digital Orthophoto Maps (TDOMs), which are in great demand for applications such as urban planning, infrastructure management, and environmental monitoring.Traditional TDOM generation methods need sophisticated processes, such as Digital Surface Models (DSM) and occlusion detection, which are computationally expensive and prone to errors.This work presents an alternative technique rooted in 2D Gaussian Splatting (2DGS), free of explicit DSM and occlusion detection. With depth map generation, spatial information for every pixel within the TDOM is retrieved and can reconstruct the scene with high precision. Divide-and-conquer strategy achieves excellent GS training and rendering with high-resolution TDOMs at a lower resource cost, which preserves higher quality of rendering on complex terrain and thin structure without a decrease in efficiency. Experimental results demonstrate the efficiency of large-scale scene reconstruction and high-precision terrain modeling. This approach provides accurate spatial data, which assists users in better planning and decision-making based on maps.
>
---
#### [replaced 043] Self-Supervised Learning for Robotic Leaf Manipulation: A Hybrid Geometric-Neural Approach
- **分类: cs.RO; cs.CV; cs.LG; I.2.10**

- **链接: [http://arxiv.org/pdf/2505.03702v2](http://arxiv.org/pdf/2505.03702v2)**

> **作者:** Srecharan Selvam
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** Automating leaf manipulation in agricultural settings faces significant challenges, including the variability of plant morphologies and deformable leaves. We propose a novel hybrid geometric-neural approach for autonomous leaf grasping that combines traditional computer vision with neural networks through self-supervised learning. Our method integrates YOLOv8 for instance segmentation and RAFT-Stereo for 3D depth estimation to build rich leaf representations, which feed into both a geometric feature scoring pipeline and a neural refinement module (GraspPointCNN). The key innovation is our confidence-weighted fusion mechanism that dynamically balances the contribution of each approach based on prediction certainty. Our self-supervised framework uses the geometric pipeline as an expert teacher to automatically generate training data. Experiments demonstrate that our approach achieves an 88.0% success rate in controlled environments and 84.7% in real greenhouse conditions, significantly outperforming both purely geometric (75.3%) and neural (60.2%) methods. This work establishes a new paradigm for agricultural robotics where domain expertise is seamlessly integrated with machine learning capabilities, providing a foundation for fully automated crop monitoring systems.
>
---
#### [replaced 044] EMPERROR: A Flexible Generative Perception Error Model for Probing Self-Driving Planners
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.07719v2](http://arxiv.org/pdf/2411.07719v2)**

> **作者:** Niklas Hanselmann; Simon Doll; Marius Cordts; Hendrik P. A. Lensch; Andreas Geiger
>
> **备注:** Project page: https://lasnik.github.io/emperror/
>
> **摘要:** To handle the complexities of real-world traffic, learning planners for self-driving from data is a promising direction. While recent approaches have shown great progress, they typically assume a setting in which the ground-truth world state is available as input. However, when deployed, planning needs to be robust to the long-tail of errors incurred by a noisy perception system, which is often neglected in evaluation. To address this, previous work has proposed drawing adversarial samples from a perception error model (PEM) mimicking the noise characteristics of a target object detector. However, these methods use simple PEMs that fail to accurately capture all failure modes of detection. In this paper, we present EMPERROR, a novel transformer-based generative PEM, apply it to stress-test an imitation learning (IL)-based planner and show that it imitates modern detectors more faithfully than previous work. Furthermore, it is able to produce realistic noisy inputs that increase the planner's collision rate by up to 85%, demonstrating its utility as a valuable tool for a more complete evaluation of self-driving planners.
>
---
#### [replaced 045] Gaussian Shading++: Rethinking the Realistic Deployment Challenge of Performance-Lossless Image Watermark for Diffusion Models
- **分类: cs.CV; cs.CR**

- **链接: [http://arxiv.org/pdf/2504.15026v2](http://arxiv.org/pdf/2504.15026v2)**

> **作者:** Zijin Yang; Xin Zhang; Kejiang Chen; Kai Zeng; Qiyi Yao; Han Fang; Weiming Zhang; Nenghai Yu
>
> **备注:** 18 pages, 8 figures
>
> **摘要:** Ethical concerns surrounding copyright protection and inappropriate content generation pose challenges for the practical implementation of diffusion models. One effective solution involves watermarking the generated images. Existing methods primarily focus on ensuring that watermark embedding does not degrade the model performance. However, they often overlook critical challenges in real-world deployment scenarios, such as the complexity of watermark key management, user-defined generation parameters, and the difficulty of verification by arbitrary third parties. To address this issue, we propose Gaussian Shading++, a diffusion model watermarking method tailored for real-world deployment. We propose a double-channel design that leverages pseudorandom error-correcting codes to encode the random seed required for watermark pseudorandomization, achieving performance-lossless watermarking under a fixed watermark key and overcoming key management challenges. Additionally, we model the distortions introduced during generation and inversion as an additive white Gaussian noise channel and employ a novel soft decision decoding strategy during extraction, ensuring strong robustness even when generation parameters vary. To enable third-party verification, we incorporate public key signatures, which provide a certain level of resistance against forgery attacks even when model inversion capabilities are fully disclosed. Extensive experiments demonstrate that Gaussian Shading++ not only maintains performance losslessness but also outperforms existing methods in terms of robustness, making it a more practical solution for real-world deployment.
>
---
#### [replaced 046] Ophora: A Large-Scale Data-Driven Text-Guided Ophthalmic Surgical Video Generation Model
- **分类: eess.IV; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.07449v2](http://arxiv.org/pdf/2505.07449v2)**

> **作者:** Wei Li; Ming Hu; Guoan Wang; Lihao Liu; Kaijin Zhou; Junzhi Ning; Xin Guo; Zongyuan Ge; Lixu Gu; Junjun He
>
> **备注:** Early accepted in MICCAI25
>
> **摘要:** In ophthalmic surgery, developing an AI system capable of interpreting surgical videos and predicting subsequent operations requires numerous ophthalmic surgical videos with high-quality annotations, which are difficult to collect due to privacy concerns and labor consumption. Text-guided video generation (T2V) emerges as a promising solution to overcome this issue by generating ophthalmic surgical videos based on surgeon instructions. In this paper, we present Ophora, a pioneering model that can generate ophthalmic surgical videos following natural language instructions. To construct Ophora, we first propose a Comprehensive Data Curation pipeline to convert narrative ophthalmic surgical videos into a large-scale, high-quality dataset comprising over 160K video-instruction pairs, Ophora-160K. Then, we propose a Progressive Video-Instruction Tuning scheme to transfer rich spatial-temporal knowledge from a T2V model pre-trained on natural video-text datasets for privacy-preserved ophthalmic surgical video generation based on Ophora-160K. Experiments on video quality evaluation via quantitative analysis and ophthalmologist feedback demonstrate that Ophora can generate realistic and reliable ophthalmic surgical videos based on surgeon instructions. We also validate the capability of Ophora for empowering downstream tasks of ophthalmic surgical workflow understanding. Code is available at https://github.com/mar-cry/Ophora.
>
---
#### [replaced 047] InstanceGen: Image Generation with Instance-level Instructions
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.05678v2](http://arxiv.org/pdf/2505.05678v2)**

> **作者:** Etai Sella; Yanir Kleiman; Hadar Averbuch-Elor
>
> **备注:** Project page: https://tau-vailab.github.io/InstanceGen/
>
> **摘要:** Despite rapid advancements in the capabilities of generative models, pretrained text-to-image models still struggle in capturing the semantics conveyed by complex prompts that compound multiple objects and instance-level attributes. Consequently, we are witnessing growing interests in integrating additional structural constraints, typically in the form of coarse bounding boxes, to better guide the generation process in such challenging cases. In this work, we take the idea of structural guidance a step further by making the observation that contemporary image generation models can directly provide a plausible fine-grained structural initialization. We propose a technique that couples this image-based structural guidance with LLM-based instance-level instructions, yielding output images that adhere to all parts of the text prompt, including object counts, instance-level attributes, and spatial relations between instances.
>
---
#### [replaced 048] HarmoniCa: Harmonizing Training and Inference for Better Feature Caching in Diffusion Transformer Acceleration
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2410.01723v5](http://arxiv.org/pdf/2410.01723v5)**

> **作者:** Yushi Huang; Zining Wang; Ruihao Gong; Jing Liu; Xinjie Zhang; Jinyang Guo; Xianglong Liu; Jun Zhang
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Diffusion Transformers (DiTs) excel in generative tasks but face practical deployment challenges due to high inference costs. Feature caching, which stores and retrieves redundant computations, offers the potential for acceleration. Existing learning-based caching, though adaptive, overlooks the impact of the prior timestep. It also suffers from misaligned objectives--aligned predicted noise vs. high-quality images--between training and inference. These two discrepancies compromise both performance and efficiency. To this end, we harmonize training and inference with a novel learning-based caching framework dubbed HarmoniCa. It first incorporates Step-Wise Denoising Training (SDT) to ensure the continuity of the denoising process, where prior steps can be leveraged. In addition, an Image Error Proxy-Guided Objective (IEPO) is applied to balance image quality against cache utilization through an efficient proxy to approximate the image error. Extensive experiments across $8$ models, $4$ samplers, and resolutions from $256\times256$ to $2K$ demonstrate superior performance and speedup of our framework. For instance, it achieves over $40\%$ latency reduction (i.e., $2.07\times$ theoretical speedup) and improved performance on PixArt-$\alpha$. Remarkably, our image-free approach reduces training time by $25\%$ compared with the previous method. Our code is available at https://github.com/ModelTC/HarmoniCa.
>
---
#### [replaced 049] GP-GS: Gaussian Processes for Enhanced Gaussian Splatting
- **分类: cs.CV; cs.AI; 68T45**

- **链接: [http://arxiv.org/pdf/2502.02283v5](http://arxiv.org/pdf/2502.02283v5)**

> **作者:** Zhihao Guo; Jingxuan Su; Shenglin Wang; Jinlong Fan; Jing Zhang; Wei Zhou; Hadi Amirpour; Yunlong Zhao; Liangxiu Han; Peng Wang
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** 3D Gaussian Splatting has emerged as an efficient photorealistic novel view synthesis method. However, its reliance on sparse Structure-from-Motion (SfM) point clouds often limits scene reconstruction quality. To address the limitation, this paper proposes a novel 3D reconstruction framework, Gaussian Processes enhanced Gaussian Splatting (GP-GS), in which a multi-output Gaussian Process model is developed to enable adaptive and uncertainty-guided densification of sparse SfM point clouds. Specifically, we propose a dynamic sampling and filtering pipeline that adaptively expands the SfM point clouds by leveraging GP-based predictions to infer new candidate points from the input 2D pixels and depth maps. The pipeline utilizes uncertainty estimates to guide the pruning of high-variance predictions, ensuring geometric consistency and enabling the generation of dense point clouds. These densified point clouds provide high-quality initial 3D Gaussians, enhancing reconstruction performance. Extensive experiments conducted on synthetic and real-world datasets across various scales validate the effectiveness and practicality of the proposed framework.
>
---
#### [replaced 050] Assessing the Feasibility of Internet-Sourced Video for Automatic Cattle Lameness Detection
- **分类: cs.CV; cs.AI; cs.LG; eess.IV**

- **链接: [http://arxiv.org/pdf/2504.16404v2](http://arxiv.org/pdf/2504.16404v2)**

> **作者:** Md Fahimuzzman Sohan; A. H. Abdul Hafez; Raid Alzubi
>
> **摘要:** Cattle lameness is often caused by hoof injuries or interdigital dermatitis, leads to pain and significantly impacts essential physiological activities such as walking, feeding, and drinking. This study presents a deep learning-based model for detecting cattle lameness, sickness, or gait abnormalities using publicly available video data. The dataset consists of 50 unique videos from 40 individual cattle, recorded from various angles in both indoor and outdoor environments. Half of the dataset represents naturally walking (normal/non-lame) cattle, while the other half consists of cattle exhibiting gait abnormalities (lame). To enhance model robustness and generalizability, data augmentation was applied to the training data. The pre-processed videos were then classified using two deep learning models: ConvLSTM2D and 3D CNN. A comparative analysis of the results demonstrates strong classification performance. Specifically, the 3D CNN model achieved a video-level classification accuracy of 90%, with precision, recall, and f1-score of 90.9%, 90.9%, and 90.91% respectively. The ConvLSTM2D model exhibited a slightly lower accuracy of 85%. This study highlights the effectiveness of directly applying classification models to learn spatiotemporal features from video data, offering an alternative to traditional multi-stage approaches that typically involve object detection, pose estimation, and feature extraction. Besides, the findings demonstrate that the proposed deep learning models, particularly the 3D CNN, effectively classify and detect lameness in cattle while simplifying the processing pipeline.
>
---
#### [replaced 051] LP-DETR: Layer-wise Progressive Relations for Object Detection
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.05147v3](http://arxiv.org/pdf/2502.05147v3)**

> **作者:** Zhengjian Kang; Ye Zhang; Xiaoyu Deng; Xintao Li; Yongzhe Zhang
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** This paper presents LP-DETR (Layer-wise Progressive DETR), a novel approach that enhances DETR-based object detection through multi-scale relation modeling. Our method introduces learnable spatial relationships between object queries through a relation-aware self-attention mechanism, which adaptively learns to balance different scales of relations (local, medium and global) across decoder layers. This progressive design enables the model to effectively capture evolving spatial dependencies throughout the detection pipeline. Extensive experiments on COCO 2017 dataset demonstrate that our method improves both convergence speed and detection accuracy compared to standard self-attention module. The proposed method achieves competitive results, reaching 52.3\% AP with 12 epochs and 52.5\% AP with 24 epochs using ResNet-50 backbone, and further improving to 58.0\% AP with Swin-L backbone. Furthermore, our analysis reveals an interesting pattern: the model naturally learns to prioritize local spatial relations in early decoder layers while gradually shifting attention to broader contexts in deeper layers, providing valuable insights for future research in object detection.
>
---
#### [replaced 052] DreamO: A Unified Framework for Image Customization
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2504.16915v3](http://arxiv.org/pdf/2504.16915v3)**

> **作者:** Chong Mou; Yanze Wu; Wenxu Wu; Zinan Guo; Pengze Zhang; Yufeng Cheng; Yiming Luo; Fei Ding; Shiwen Zhang; Xinghui Li; Mengtian Li; Mingcong Liu; Yi Zhang; Shaojin Wu; Songtao Zhao; Jian Zhang; Qian He; Xinglong Wu
>
> **摘要:** Recently, extensive research on image customization (e.g., identity, subject, style, background, etc.) demonstrates strong customization capabilities in large-scale generative models. However, most approaches are designed for specific tasks, restricting their generalizability to combine different types of condition. Developing a unified framework for image customization remains an open challenge. In this paper, we present DreamO, an image customization framework designed to support a wide range of tasks while facilitating seamless integration of multiple conditions. Specifically, DreamO utilizes a diffusion transformer (DiT) framework to uniformly process input of different types. During training, we construct a large-scale training dataset that includes various customization tasks, and we introduce a feature routing constraint to facilitate the precise querying of relevant information from reference images. Additionally, we design a placeholder strategy that associates specific placeholders with conditions at particular positions, enabling control over the placement of conditions in the generated results. Moreover, we employ a progressive training strategy consisting of three stages: an initial stage focused on simple tasks with limited data to establish baseline consistency, a full-scale training stage to comprehensively enhance the customization capabilities, and a final quality alignment stage to correct quality biases introduced by low-quality data. Extensive experiments demonstrate that the proposed DreamO can effectively perform various image customization tasks with high quality and flexibly integrate different types of control conditions.
>
---
#### [replaced 053] FMNV: A Dataset of Media-Published News Videos for Fake News Detection
- **分类: cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2504.07687v3](http://arxiv.org/pdf/2504.07687v3)**

> **作者:** Yihao Wang; Zhong Qian; Peifeng Li
>
> **摘要:** News media, particularly video-based platforms, have become deeply embed-ded in daily life, concurrently amplifying the risks of misinformation dissem-ination. Consequently, multimodal fake news detection has garnered signifi-cant research attention. However, existing datasets predominantly comprise user-generated videos characterized by crude editing and limited public en-gagement, whereas professionally crafted fake news videos disseminated by media outlets-often politically or virally motivated-pose substantially greater societal harm. To address this gap, we construct FMNV, a novel da-taset exclusively composed of news videos published by media organizations. Through empirical analysis of existing datasets and our curated collection, we categorize fake news videos into four distinct types. Building upon this taxonomy, we employ Large Language Models (LLMs) to automatically generate deceptive content by manipulating authentic media-published news videos. Furthermore, we propose FMNVD, a baseline model featuring a dual-stream architecture that integrates spatio-temporal motion features from a 3D ResNeXt-101 backbone and static visual semantics from CLIP. The two streams are fused via an attention-based mechanism, while co-attention modules refine the visual, textual, and audio features for effective multi-modal aggregation. Comparative experiments demonstrate both the generali-zation capability of FMNV across multiple baselines and the superior detec-tion efficacy of FMNVD. This work establishes critical benchmarks for de-tecting high-impact fake news in media ecosystems while advancing meth-odologies for cross-modal inconsistency analysis. Our dataset is available in https://github.com/DennisIW/FMNV.
>
---
#### [replaced 054] TextCenGen: Attention-Guided Text-Centric Background Adaptation for Text-to-Image Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2404.11824v5](http://arxiv.org/pdf/2404.11824v5)**

> **作者:** Tianyi Liang; Jiangqi Liu; Yifei Huang; Shiqi Jiang; Jianshen Shi; Changbo Wang; Chenhui Li
>
> **备注:** 7 pages, 7 figures
>
> **摘要:** Text-to-image (T2I) generation has made remarkable progress in producing high-quality images, but a fundamental challenge remains: creating backgrounds that naturally accommodate text placement without compromising image quality. This capability is non-trivial for real-world applications like graphic design, where clear visual hierarchy between content and text is essential. Prior work has primarily focused on arranging layouts within existing static images, leaving unexplored the potential of T2I models for generating text-friendly backgrounds. We present TextCenGen, a training-free dynamic background adaptation in the blank region for text-friendly image generation. Instead of directly reducing attention in text areas, which degrades image quality, we relocate conflicting objects before background optimization. Our method analyzes cross-attention maps to identify conflicting objects overlapping with text regions and uses a force-directed graph approach to guide their relocation, followed by attention excluding constraints to ensure smooth backgrounds. Our method is plug-and-play, requiring no additional training while well balancing both semantic fidelity and visual quality. Evaluated on our proposed text-friendly T2I benchmark of 27,000 images across four seed datasets, TextCenGen outperforms existing methods by achieving 23% lower saliency overlap in text regions while maintaining 98% of the semantic fidelity measured by CLIP score and our proposed Visual-Textual Concordance Metric (VTCM).
>
---
#### [replaced 055] Hierarchical and Multimodal Data for Daily Activity Understanding
- **分类: cs.CV; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.17696v3](http://arxiv.org/pdf/2504.17696v3)**

> **作者:** Ghazal Kaviani; Yavuz Yarici; Seulgi Kim; Mohit Prabhushankar; Ghassan AlRegib; Mashhour Solh; Ameya Patil
>
> **摘要:** Daily Activity Recordings for Artificial Intelligence (DARai, pronounced "Dahr-ree") is a multimodal, hierarchically annotated dataset constructed to understand human activities in real-world settings. DARai consists of continuous scripted and unscripted recordings of 50 participants in 10 different environments, totaling over 200 hours of data from 20 sensors including multiple camera views, depth and radar sensors, wearable inertial measurement units (IMUs), electromyography (EMG), insole pressure sensors, biomonitor sensors, and gaze tracker. To capture the complexity in human activities, DARai is annotated at three levels of hierarchy: (i) high-level activities (L1) that are independent tasks, (ii) lower-level actions (L2) that are patterns shared between activities, and (iii) fine-grained procedures (L3) that detail the exact execution steps for actions. The dataset annotations and recordings are designed so that 22.7% of L2 actions are shared between L1 activities and 14.2% of L3 procedures are shared between L2 actions. The overlap and unscripted nature of DARai allows counterfactual activities in the dataset. Experiments with various machine learning models showcase the value of DARai in uncovering important challenges in human-centered applications. Specifically, we conduct unimodal and multimodal sensor fusion experiments for recognition, temporal localization, and future action anticipation across all hierarchical annotation levels. To highlight the limitations of individual sensors, we also conduct domain-variant experiments that are enabled by DARai's multi-sensor and counterfactual activity design setup. The code, documentation, and dataset are available at the dedicated DARai website: https://alregib.ece.gatech.edu/software-and-datasets/darai-daily-activity-recordings-for-artificial-intelligence-and-machine-learning/
>
---
#### [replaced 056] Transforming Hyperspectral Images Into Chemical Maps: An End-to-End Deep Learning Approach
- **分类: cs.CV; cs.LG; q-bio.QM**

- **链接: [http://arxiv.org/pdf/2504.14131v3](http://arxiv.org/pdf/2504.14131v3)**

> **作者:** Ole-Christian Galbo Engstrøm; Michela Albano-Gaglio; Erik Schou Dreier; Yamine Bouzembrak; Maria Font-i-Furnols; Puneet Mishra; Kim Steenstrup Pedersen
>
> **摘要:** Current approaches to chemical map generation from hyperspectral images are based on models such as partial least squares (PLS) regression, generating pixel-wise predictions that do not consider spatial context and suffer from a high degree of noise. This study proposes an end-to-end deep learning approach using a modified version of U-Net and a custom loss function to directly obtain chemical maps from hyperspectral images, skipping all intermediate steps required for traditional pixel-wise analysis. We compare the U-Net with the traditional PLS regression on a real dataset of pork belly samples with associated mean fat reference values. The U-Net obtains a test set root mean squared error of between 9% and 13% lower than that of PLS regression on the task of mean fat prediction. At the same time, U-Net generates fine detail chemical maps where 99.91% of the variance is spatially correlated. Conversely, only 2.53% of the variance in the PLS-generated chemical maps is spatially correlated, indicating that each pixel-wise prediction is largely independent of neighboring pixels. Additionally, while the PLS-generated chemical maps contain predictions far beyond the physically possible range of 0-100%, U-Net learns to stay inside this range. Thus, the findings of this study indicate that U-Net is superior to PLS for chemical map generation.
>
---
#### [replaced 057] No Other Representation Component Is Needed: Diffusion Transformers Can Provide Representation Guidance by Themselves
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.02831v3](http://arxiv.org/pdf/2505.02831v3)**

> **作者:** Dengyang Jiang; Mengmeng Wang; Liuzhuozheng Li; Lei Zhang; Haoyu Wang; Wei Wei; Guang Dai; Yanning Zhang; Jingdong Wang
>
> **备注:** Self-Representation Alignment for Diffusion Transformers. Code: https://github.com/vvvvvjdy/SRA
>
> **摘要:** Recent studies have demonstrated that learning a meaningful internal representation can both accelerate generative training and enhance the generation quality of diffusion transformers. However, existing approaches necessitate to either introduce an external and complex representation training framework or rely on a large-scale, pre-trained representation foundation model to provide representation guidance during the original generative training process. In this study, we posit that the unique discriminative process inherent to diffusion transformers enables them to offer such guidance without requiring external representation components. We therefore propose Self-Representation Alignment (SRA), a simple yet straightforward method that obtains representation guidance through a self-distillation manner. Specifically, SRA aligns the output latent representation of the diffusion transformer in the earlier layer with higher noise to that in the later layer with lower noise to progressively enhance the overall representation learning during only the generative training process. Experimental results indicate that applying SRA to DiTs and SiTs yields consistent performance improvements. Moreover, SRA not only significantly outperforms approaches relying on auxiliary, complex representation training frameworks but also achieves performance comparable to methods that are heavily dependent on powerful external representation priors.
>
---
#### [replaced 058] Building Age Estimation: A New Multi-Modal Benchmark Dataset and Community Challenge
- **分类: cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.13818v2](http://arxiv.org/pdf/2502.13818v2)**

> **作者:** Nikolaos Dionelis; Nicolas Longépé; Alessandra Feliciotti; Mattia Marconcini; Devis Peressutti; Nika Oman Kadunc; JaeWan Park; Hagai Raja Sinulingga; Steve Andreas Immanuel; Ba Tran; Caroline Arnold
>
> **备注:** 13 pages, 22 figures, Submitted
>
> **摘要:** Estimating the construction year of buildings is of great importance for sustainability. Sustainable buildings minimize energy consumption and are a key part of responsible and sustainable urban planning and development to effectively combat climate change. By using Artificial Intelligence (AI) and recently proposed powerful Transformer models, we are able to estimate the construction epoch of buildings from a multi-modal dataset. In this paper, we introduce a new benchmark multi-modal dataset, i.e. the Map your City Dataset (MyCD), containing top-view Very High Resolution (VHR) images, Earth Observation (EO) multi-spectral data from the Copernicus Sentinel-2 satellite constellation, and street-view images in many different cities in Europe that are co-localized with respect to the building under study and labelled with the construction epoch. We assess EO generalization performance on new/ previously unseen cities that have been held-out from training and appear only during inference. In this work, we present the community-based data challenge we organized based on MyCD. The AI4EO Challenge ESA MapYourCity was opened in 2024 for 4 months. In this paper, we present the Top-4 performing models of the challenge, and the evaluation results. During inference, the performance of the models using: i) both all three input modalities, and ii) only the two top-view modalities, i.e. without the street-view ground images, is examined. The evaluation results in this work show that the models to estimate the construction year of buildings are effective and can achieve good performance on this difficult important real-world task, even when inference is on previously unseen cities, as well as even when using only the two top-view modalities (i.e. VHR and Sentinel-2) during inference.
>
---
#### [replaced 059] RT-GAN: Recurrent Temporal GAN for Adding Lightweight Temporal Consistency to Frame-Based Domain Translation Approaches
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2310.00868v2](http://arxiv.org/pdf/2310.00868v2)**

> **作者:** Shawn Mathew; Saad Nadeem; Alvin C. Goh; Arie Kaufman
>
> **备注:** MICCAI 2025 Early Accept. First two authors contributed equally
>
> **摘要:** Fourteen million colonoscopies are performed annually just in the U.S. However, the videos from these colonoscopies are not saved due to storage constraints (each video from a high-definition colonoscope camera can be in tens of gigabytes). Instead, a few relevant individual frames are saved for documentation/reporting purposes and these are the frames on which most current colonoscopy AI models are trained on. While developing new unsupervised domain translation methods for colonoscopy (e.g. to translate between real optical and virtual/CT colonoscopy), it is thus typical to start with approaches that initially work for individual frames without temporal consistency. Once an individual-frame model has been finalized, additional contiguous frames are added with a modified deep learning architecture to train a new model from scratch for temporal consistency. This transition to temporally-consistent deep learning models, however, requires significantly more computational and memory resources for training. In this paper, we present a lightweight solution with a tunable temporal parameter, RT-GAN (Recurrent Temporal GAN), for adding temporal consistency to individual frame-based approaches that reduces training requirements by a factor of 5. We demonstrate the effectiveness of our approach on two challenging use cases in colonoscopy: haustral fold segmentation (indicative of missed surface) and realistic colonoscopy simulator video generation. We also release a first-of-its kind temporal dataset for colonoscopy for the above use cases. The datasets, accompanying code, and pretrained models will be made available on our Computational Endoscopy Platform GitHub (https://github.com/nadeemlab/CEP). The supplementary video is available at https://youtu.be/UMVP-uIXwWk.
>
---
#### [replaced 060] CAST: Component-Aligned 3D Scene Reconstruction from an RGB Image
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2502.12894v2](http://arxiv.org/pdf/2502.12894v2)**

> **作者:** Kaixin Yao; Longwen Zhang; Xinhao Yan; Yan Zeng; Qixuan Zhang; Wei Yang; Lan Xu; Jiayuan Gu; Jingyi Yu
>
> **备注:** Project Page: https://sites.google.com/view/cast4
>
> **摘要:** Recovering high-quality 3D scenes from a single RGB image is a challenging task in computer graphics. Current methods often struggle with domain-specific limitations or low-quality object generation. To address these, we propose CAST (Component-Aligned 3D Scene Reconstruction from a Single RGB Image), a novel method for 3D scene reconstruction and recovery. CAST starts by extracting object-level 2D segmentation and relative depth information from the input image, followed by using a GPT-based model to analyze inter-object spatial relationships. This enables the understanding of how objects relate to each other within the scene, ensuring more coherent reconstruction. CAST then employs an occlusion-aware large-scale 3D generation model to independently generate each object's full geometry, using MAE and point cloud conditioning to mitigate the effects of occlusions and partial object information, ensuring accurate alignment with the source image's geometry and texture. To align each object with the scene, the alignment generation model computes the necessary transformations, allowing the generated meshes to be accurately placed and integrated into the scene's point cloud. Finally, CAST incorporates a physics-aware correction step that leverages a fine-grained relation graph to generate a constraint graph. This graph guides the optimization of object poses, ensuring physical consistency and spatial coherence. By utilizing Signed Distance Fields (SDF), the model effectively addresses issues such as occlusions, object penetration, and floating objects, ensuring that the generated scene accurately reflects real-world physical interactions. CAST can be leveraged in robotics, enabling efficient real-to-simulation workflows and providing realistic, scalable simulation environments for robotic systems.
>
---
#### [replaced 061] SeriesBench: A Benchmark for Narrative-Driven Drama Series Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.21435v3](http://arxiv.org/pdf/2504.21435v3)**

> **作者:** Chenkai Zhang; Yiming Lei; Zeming Liu; Haitao Leng; Shaoguo Liu; Tingting Gao; Qingjie Liu; Yunhong Wang
>
> **备注:** 29 pages, 15 figures, CVPR 2025
>
> **摘要:** With the rapid development of Multi-modal Large Language Models (MLLMs), an increasing number of benchmarks have been established to evaluate the video understanding capabilities of these models. However, these benchmarks focus on standalone videos and mainly assess "visual elements" like human actions and object states. In reality, contemporary videos often encompass complex and continuous narratives, typically presented as a series. To address this challenge, we propose SeriesBench, a benchmark consisting of 105 carefully curated narrative-driven series, covering 28 specialized tasks that require deep narrative understanding. Specifically, we first select a diverse set of drama series spanning various genres. Then, we introduce a novel long-span narrative annotation method, combined with a full-information transformation approach to convert manual annotations into diverse task formats. To further enhance model capacity for detailed analysis of plot structures and character relationships within series, we propose a novel narrative reasoning framework, PC-DCoT. Extensive results on SeriesBench indicate that existing MLLMs still face significant challenges in understanding narrative-driven series, while PC-DCoT enables these MLLMs to achieve performance improvements. Overall, our SeriesBench and PC-DCoT highlight the critical necessity of advancing model capabilities to understand narrative-driven series, guiding the future development of MLLMs. SeriesBench is publicly available at https://github.com/zackhxn/SeriesBench-CVPR2025.
>
---
#### [replaced 062] Multi-Modal Language Models as Text-to-Image Model Evaluators
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.00759v2](http://arxiv.org/pdf/2505.00759v2)**

> **作者:** Jiahui Chen; Candace Ross; Reyhane Askari-Hemmat; Koustuv Sinha; Melissa Hall; Michal Drozdzal; Adriana Romero-Soriano
>
> **摘要:** The steady improvements of text-to-image (T2I) generative models lead to slow deprecation of automatic evaluation benchmarks that rely on static datasets, motivating researchers to seek alternative ways to evaluate the T2I progress. In this paper, we explore the potential of multi-modal large language models (MLLMs) as evaluator agents that interact with a T2I model, with the objective of assessing prompt-generation consistency and image aesthetics. We present Multimodal Text-to-Image Eval (MT2IE), an evaluation framework that iteratively generates prompts for evaluation, scores generated images and matches T2I evaluation of existing benchmarks with a fraction of the prompts used in existing static benchmarks. Moreover, we show that MT2IE's prompt-generation consistency scores have higher correlation with human judgment than scores previously introduced in the literature. MT2IE generates prompts that are efficient at probing T2I model performance, producing the same relative T2I model rankings as existing benchmarks while using only 1/80th the number of prompts for evaluation.
>
---
#### [replaced 063] Discriminative and Consistent Representation Distillation
- **分类: cs.CV; cs.AI; 68T07; I.4; I.2**

- **链接: [http://arxiv.org/pdf/2407.11802v5](http://arxiv.org/pdf/2407.11802v5)**

> **作者:** Nikolaos Giakoumoglou; Tania Stathaki
>
> **备注:** Preprint. Code: https://github.com/giakoumoglou/distillers, Supplementary: https://giakoumoglou.com/src/dcd_suppl.pdf
>
> **摘要:** Knowledge Distillation (KD) aims to transfer knowledge from a large teacher model to a smaller student model. While contrastive learning has shown promise in self-supervised learning by creating discriminative representations, its application in knowledge distillation remains limited and focuses primarily on discrimination, neglecting the structural relationships captured by the teacher model. To address this limitation, we propose Discriminative and Consistent Distillation (DCD), which employs a contrastive loss along with a consistency regularization to minimize the discrepancy between the distributions of teacher and student representations. Our method introduces learnable temperature and bias parameters that adapt during training to balance these complementary objectives, replacing the fixed hyperparameters commonly used in contrastive learning approaches. Through extensive experiments on CIFAR-100 and ImageNet ILSVRC-2012, we demonstrate that DCD achieves state-of-the-art performance, with the student model sometimes surpassing the teacher's accuracy. Furthermore, we show that DCD's learned representations exhibit superior cross-dataset generalization when transferred to Tiny ImageNet and STL-10.
>
---
#### [replaced 064] VideoUFO: A Million-Scale User-Focused Dataset for Text-to-Video Generation
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2503.01739v2](http://arxiv.org/pdf/2503.01739v2)**

> **作者:** Wenhao Wang; Yi Yang
>
> **摘要:** Text-to-video generative models convert textual prompts into dynamic visual content, offering wide-ranging applications in film production, gaming, and education. However, their real-world performance often falls short of user expectations. One key reason is that these models have not been trained on videos related to some topics users want to create. In this paper, we propose VideoUFO, the first Video dataset specifically curated to align with Users' FOcus in real-world scenarios. Beyond this, our VideoUFO also features: (1) minimal (0.29%) overlap with existing video datasets, and (2) videos searched exclusively via YouTube's official API under the Creative Commons license. These two attributes provide future researchers with greater freedom to broaden their training sources. The VideoUFO comprises over 1.09 million video clips, each paired with both a brief and a detailed caption (description). Specifically, through clustering, we first identify 1,291 user-focused topics from the million-scale real text-to-video prompt dataset, VidProM. Then, we use these topics to retrieve videos from YouTube, split the retrieved videos into clips, and generate both brief and detailed captions for each clip. After verifying the clips with specified topics, we are left with about 1.09 million video clips. Our experiments reveal that (1) current 16 text-to-video models do not achieve consistent performance across all user-focused topics; and (2) a simple model trained on VideoUFO outperforms others on worst-performing topics. The dataset and code are publicly available at https://huggingface.co/datasets/WenhaoWang/VideoUFO and https://github.com/WangWenhao0716/BenchUFO under the CC BY 4.0 License.
>
---
#### [replaced 065] Enhancing Scene Coordinate Regression with Efficient Keypoint Detection and Sequential Information
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.06488v2](http://arxiv.org/pdf/2412.06488v2)**

> **作者:** Kuan Xu; Zeyu Jiang; Haozhi Cao; Shenghai Yuan; Chen Wang; Lihua Xie
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Scene Coordinate Regression (SCR) is a visual localization technique that utilizes deep neural networks (DNN) to directly regress 2D-3D correspondences for camera pose estimation. However, current SCR methods often face challenges in handling repetitive textures and meaningless areas due to their reliance on implicit triangulation. In this paper, we propose an efficient and accurate SCR system. Compared to existing SCR methods, we propose a unified architecture for both scene encoding and salient keypoint detection, allowing our system to prioritize the encoding of informative regions. This design significantly improves computational efficiency. Additionally, we introduce a mechanism that utilizes sequential information during both mapping and relocalization. The proposed method enhances the implicit triangulation, especially in environments with repetitive textures. Comprehensive experiments conducted across indoor and outdoor datasets demonstrate that the proposed system outperforms state-of-the-art (SOTA) SCR methods. Our single-frame relocalization mode improves the recall rate of our baseline by 6.4% and increases the running speed from 56Hz to 90Hz. Furthermore, our sequence-based mode increases the recall rate by 11% while maintaining the original efficiency.
>
---
#### [replaced 066] Learning Phase Distortion with Selective State Space Models for Video Turbulence Mitigation
- **分类: cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2504.02697v2](http://arxiv.org/pdf/2504.02697v2)**

> **作者:** Xingguang Zhang; Nicholas Chimitt; Xijun Wang; Yu Yuan; Stanley H. Chan
>
> **备注:** CVPR 2025 Highlight (extended), project page: https://xg416.github.io/MambaTM/
>
> **摘要:** Atmospheric turbulence is a major source of image degradation in long-range imaging systems. Although numerous deep learning-based turbulence mitigation (TM) methods have been proposed, many are slow, memory-hungry, and do not generalize well. In the spatial domain, methods based on convolutional operators have a limited receptive field, so they cannot handle a large spatial dependency required by turbulence. In the temporal domain, methods relying on self-attention can, in theory, leverage the lucky effects of turbulence, but their quadratic complexity makes it difficult to scale to many frames. Traditional recurrent aggregation methods face parallelization challenges. In this paper, we present a new TM method based on two concepts: (1) A turbulence mitigation network based on the Selective State Space Model (MambaTM). MambaTM provides a global receptive field in each layer across spatial and temporal dimensions while maintaining linear computational complexity. (2) Learned Latent Phase Distortion (LPD). LPD guides the state space model. Unlike classical Zernike-based representations of phase distortion, the new LPD map uniquely captures the actual effects of turbulence, significantly improving the model's capability to estimate degradation by reducing the ill-posedness. Our proposed method exceeds current state-of-the-art networks on various synthetic and real-world TM benchmarks with significantly faster inference speed.
>
---
#### [replaced 067] Relational Representation Distillation
- **分类: cs.CV; cs.AI; 68T07; I.4; I.2**

- **链接: [http://arxiv.org/pdf/2407.12073v5](http://arxiv.org/pdf/2407.12073v5)**

> **作者:** Nikolaos Giakoumoglou; Tania Stathaki
>
> **备注:** Preprint. Code: https://github.com/giakoumoglou/distillers, Supplementary: https://giakoumoglou.com/src/rrd_suppl.pdf
>
> **摘要:** Knowledge distillation involves transferring knowledge from large, cumbersome teacher models to more compact student models. The standard approach minimizes the Kullback-Leibler (KL) divergence between the probabilistic outputs of a teacher and student network. However, this approach fails to capture important structural relationships in the teacher's internal representations. Recent advances have turned to contrastive learning objectives, but these methods impose overly strict constraints through instance-discrimination, forcing apart semantically similar samples even when they should maintain similarity. This motivates an alternative objective by which we preserve relative relationships between instances. Our method employs separate temperature parameters for teacher and student distributions, with sharper student outputs, enabling precise learning of primary relationships while preserving secondary similarities. We show theoretical connections between our objective and both InfoNCE loss and KL divergence. Experiments demonstrate that our method significantly outperforms existing knowledge distillation methods across diverse knowledge transfer tasks, achieving better alignment with teacher models, and sometimes even outperforms the teacher network.
>
---
#### [replaced 068] Human Motion Prediction via Test-domain-aware Adaptation with Easily-available Human Motions Estimated from Videos
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2505.07301v2](http://arxiv.org/pdf/2505.07301v2)**

> **作者:** Katsuki Shimbo; Hiromu Taketsugu; Norimichi Ukita
>
> **备注:** 5 pages, 4 figures
>
> **摘要:** In 3D Human Motion Prediction (HMP), conventional methods train HMP models with expensive motion capture data. However, the data collection cost of such motion capture data limits the data diversity, which leads to poor generalizability to unseen motions or subjects. To address this issue, this paper proposes to enhance HMP with additional learning using estimated poses from easily available videos. The 2D poses estimated from the monocular videos are carefully transformed into motion capture-style 3D motions through our pipeline. By additional learning with the obtained motions, the HMP model is adapted to the test domain. The experimental results demonstrate the quantitative and qualitative impact of our method.
>
---
#### [replaced 069] ConceptMaster: Multi-Concept Video Customization on Diffusion Transformer Models Without Test-Time Tuning
- **分类: cs.CV**

- **链接: [http://arxiv.org/pdf/2501.04698v2](http://arxiv.org/pdf/2501.04698v2)**

> **作者:** Yuzhou Huang; Ziyang Yuan; Quande Liu; Qiulin Wang; Xintao Wang; Ruimao Zhang; Pengfei Wan; Di Zhang; Kun Gai
>
> **备注:** Project Page: https://yuzhou914.github.io/ConceptMaster/. Update and release MCVC Evaluation Set
>
> **摘要:** Text-to-video generation has made remarkable advancements through diffusion models. However, Multi-Concept Video Customization (MCVC) remains a significant challenge. We identify two key challenges for this task: 1) the identity decoupling issue, where directly adopting existing customization methods inevitably mix identity attributes when handling multiple concepts simultaneously, and 2) the scarcity of high-quality video-entity pairs, which is crucial for training a model that can well represent and decouple various customized concepts in video generation. To address these challenges, we introduce ConceptMaster, a novel framework that effectively addresses the identity decoupling issues while maintaining concept fidelity in video customization. Specifically, we propose to learn decoupled multi-concept embeddings and inject them into diffusion models in a standalone manner, which effectively guarantees the quality of customized videos with multiple identities, even for highly similar visual concepts. To overcome the scarcity of high-quality MCVC data, we establish a data construction pipeline, which enables collection of high-quality multi-concept video-entity data pairs across diverse scenarios. A multi-concept video evaluation set is further devised to comprehensively validate our method from three dimensions, including concept fidelity, identity decoupling ability, and video generation quality, across six different concept composition scenarios. Extensive experiments demonstrate that ConceptMaster significantly outperforms previous methods for video customization tasks, showing great potential to generate personalized and semantically accurate content for video diffusion models.
>
---
