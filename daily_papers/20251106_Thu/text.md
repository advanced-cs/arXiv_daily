# 自然语言处理 cs.CL

- **最新发布 55 篇**

- **更新 50 篇**

## 最新发布

#### [new 001] Benchmarking the Thinking Mode of Multimodal Large Language Models in Clinical Tasks
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文评估多模态大模型在临床任务中“思考模式”对性能的影响，对比其与“非思考模式”在医学视觉问答和图像解读上的表现，发现提升有限，凸显医疗专用数据与知识整合方法的迫切需求。**

- **链接: [http://arxiv.org/pdf/2511.03328v1](http://arxiv.org/pdf/2511.03328v1)**

> **作者:** Jindong Hong; Tianjie Chen; Lingjie Luo; Chuanyang Zheng; Ting Xu; Haibao Yu; Jianing Qiu; Qianzhong Chen; Suning Huang; Yan Xu; Yong Gui; Yijun He; Jiankai Sun
>
> **摘要:** A recent advancement in Multimodal Large Language Models (MLLMs) research is the emergence of "reasoning MLLMs" that offer explicit control over their internal thinking processes (normally referred as the "thinking mode") alongside the standard "non-thinking mode". This capability allows these models to engage in a step-by-step process of internal deliberation before generating a final response. With the rapid transition to and adoption of these "dual-state" MLLMs, this work rigorously evaluated how the enhanced reasoning processes of these MLLMs impact model performance and reliability in clinical tasks. This paper evaluates the active "thinking mode" capabilities of two leading MLLMs, Seed1.5-VL and Gemini-2.5-Flash, for medical applications. We assessed their performance on four visual medical tasks using VQA-RAD and ROCOv2 datasets. Our findings reveal that the improvement from activating the thinking mode remains marginal compared to the standard non-thinking mode for the majority of the tasks. Their performance on complex medical tasks such as open-ended VQA and medical image interpretation remains suboptimal, highlighting the need for domain-specific medical data and more advanced methods for medical knowledge integration.
>
---
#### [new 002] BanglaSTEM: A Parallel Corpus for Technical Domain Bangla-English Translation
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出BanglaSTEM，一个面向STEM领域的平行语料库，解决Bangla-English技术术语翻译不准问题。通过人工筛选高质量句对，训练T5翻译模型，显著提升技术内容翻译精度，助力Bangla用户使用英文大模型。**

- **链接: [http://arxiv.org/pdf/2511.03498v1](http://arxiv.org/pdf/2511.03498v1)**

> **作者:** Kazi Reyazul Hasan; Mubasshira Musarrat; A. B. M. Alim Al Islam; Muhammad Abdullah Adnan
>
> **摘要:** Large language models work well for technical problem solving in English but perform poorly when the same questions are asked in Bangla. A simple solution would be to translate Bangla questions into English first and then use these models. However, existing Bangla-English translation systems struggle with technical terms. They often mistranslate specialized vocabulary, which changes the meaning of the problem and leads to wrong answers. We present BanglaSTEM, a dataset of 5,000 carefully selected Bangla-English sentence pairs from STEM fields including computer science, mathematics, physics, chemistry, and biology. We generated over 12,000 translations using language models and then used human evaluators to select the highest quality pairs that preserve technical terminology correctly. We train a T5-based translation model on BanglaSTEM and test it on two tasks: generating code and solving math problems. Our results show significant improvements in translation accuracy for technical content, making it easier for Bangla speakers to use English-focused language models effectively. Both the BanglaSTEM dataset and the trained translation model are publicly released at https://huggingface.co/reyazul/BanglaSTEM-T5.
>
---
#### [new 003] PolyNorm: Few-Shot LLM-Based Text Normalization for Text-to-Speech
- **分类: cs.CL; cs.LG**

- **简介: PolyNorm提出一种基于LLM的少样本文本归一化方法，解决传统系统依赖人工规则、难扩展的问题，通过提示工程实现多语言低成本适配，并构建了多语言基准数据集PolyNorm-Benchmark。**

- **链接: [http://arxiv.org/pdf/2511.03080v1](http://arxiv.org/pdf/2511.03080v1)**

> **作者:** Michel Wong; Ali Alshehri; Sophia Kao; Haotian He
>
> **备注:** 9 pages including appendix. EMNLP 2025 Industry Track
>
> **摘要:** Text Normalization (TN) is a key preprocessing step in Text-to-Speech (TTS) systems, converting written forms into their canonical spoken equivalents. Traditional TN systems can exhibit high accuracy, but involve substantial engineering effort, are difficult to scale, and pose challenges to language coverage, particularly in low-resource settings. We propose PolyNorm, a prompt-based approach to TN using Large Language Models (LLMs), aiming to reduce the reliance on manually crafted rules and enable broader linguistic applicability with minimal human intervention. Additionally, we present a language-agnostic pipeline for automatic data curation and evaluation, designed to facilitate scalable experimentation across diverse languages. Experiments across eight languages show consistent reductions in the word error rate (WER) compared to a production-grade-based system. To support further research, we release PolyNorm-Benchmark, a multilingual data set covering a diverse range of text normalization phenomena.
>
---
#### [new 004] Segmentation Beyond Defaults: Asymmetrical Byte Pair Encoding for Optimal Machine Translation Performance
- **分类: cs.CL**

- **简介: 该论文面向机器翻译任务，解决对称BPE分词在多语言对中性能不佳的问题，提出非对称BPE策略——源语言使用高合并数、目标语言使用低合并数，在低资源场景下显著提升翻译质量。**

- **链接: [http://arxiv.org/pdf/2511.03383v1](http://arxiv.org/pdf/2511.03383v1)**

> **作者:** Saumitra Yadav; Manish Shrivastava
>
> **备注:** Accepted at WAT 2025
>
> **摘要:** Existing Machine Translation (MT) research often suggests a single, fixed set of hyperparameters for word segmentation models, symmetric Byte Pair Encoding (BPE), which applies the same number of merge operations (NMO) to train tokenizers for both source and target languages. However, we demonstrate that this uniform approach doesn't guarantee optimal MT performance across different language pairs and data sizes. This work investigates BPE segmentation recipes across various data volumes and language pairs to evaluate MT system performance. We find that utilizing asymmetric BPE, where the source and target languages have different NMOs, significantly improves results over the symmetric approach, especially in low-resource settings (50K, 100K, and 500K sentence pairs). Specifically, asymmetric BPE yield statistically significant ($p<0.05$) average gains of 5.32, 4.46, and 0.7 CHRF++ on English-Hindi in low-resource setups. We validated this trend across six additional language pairs (English and Telugu, Shona, Norwegian, Kyrgyz, Hausa, and Inuktitut), observing statistically significant improvement in 10 out of 12 systems compared to symmetric BPE. Our findings indicate a high NMO for the source (4K to 32K) and a low NMO for the target (0.5K to 2K) provides optimal results, particularly benefiting low-resource MT.
>
---
#### [new 005] ROBoto2: An Interactive System and Dataset for LLM-assisted Clinical Trial Risk of Bias Assessment
- **分类: cs.CL**

- **简介: ROBOTO2是一个面向临床试验偏倚风险评估的交互式系统，利用LLM辅助完成ROB2标注任务，解决人工效率低问题。构建了521份儿科试验数据集，开源平台并评估了4个LLM性能。**

- **链接: [http://arxiv.org/pdf/2511.03048v1](http://arxiv.org/pdf/2511.03048v1)**

> **作者:** Anthony Hevia; Sanjana Chintalapati; Veronica Ka Wai Lai; Thanh Tam Nguyen; Wai-Tat Wong; Terry Klassen; Lucy Lu Wang
>
> **备注:** EMNLP 2025 System Demonstration
>
> **摘要:** We present ROBOTO2, an open-source, web-based platform for large language model (LLM)-assisted risk of bias (ROB) assessment of clinical trials. ROBOTO2 streamlines the traditionally labor-intensive ROB v2 (ROB2) annotation process via an interactive interface that combines PDF parsing, retrieval-augmented LLM prompting, and human-in-the-loop review. Users can upload clinical trial reports, receive preliminary answers and supporting evidence for ROB2 signaling questions, and provide real-time feedback or corrections to system suggestions. ROBOTO2 is publicly available at https://roboto2.vercel.app/, with code and data released to foster reproducibility and adoption. We construct and release a dataset of 521 pediatric clinical trial reports (8954 signaling questions with 1202 evidence passages), annotated using both manually and LLM-assisted methods, serving as a benchmark and enabling future research. Using this dataset, we benchmark ROB2 performance for 4 LLMs and provide an analysis into current model capabilities and ongoing challenges in automating this critical aspect of systematic review.
>
---
#### [new 006] Step-Audio-EditX Technical Report
- **分类: cs.CL; cs.AI; cs.HC; cs.SD; eess.AS**

- **简介: Step-Audio-EditX是首个基于LLM的开源音频编辑模型，解决传统音频编辑依赖嵌入先验的问题，仅用大间隔合成数据实现情感、语调等细粒度迭代控制，兼具强零样本TTS能力，性能超越现有模型。**

- **链接: [http://arxiv.org/pdf/2511.03601v1](http://arxiv.org/pdf/2511.03601v1)**

> **作者:** Chao Yan; Boyong Wu; Peng Yang; Pengfei Tan; Guoqiang Hu; Yuxin Zhang; Xiangyu; Zhang; Fei Tian; Xuerui Yang; Xiangyu Zhang; Daxin Jiang; Gang Yu
>
> **摘要:** We present Step-Audio-EditX, the first open-source LLM-based audio model excelling at expressive and iterative audio editing encompassing emotion, speaking style, and paralinguistics alongside robust zero-shot text-to-speech (TTS) capabilities.Our core innovation lies in leveraging only large-margin synthetic data, which circumvents the need for embedding-based priors or auxiliary modules. This large-margin learning approach enables both iterative control and high expressivity across voices, and represents a fundamental pivot from the conventional focus on representation-level disentanglement. Evaluation results demonstrate that Step-Audio-EditX surpasses both MiniMax-2.6-hd and Doubao-Seed-TTS-2.0 in emotion editing and other fine-grained control tasks.
>
---
#### [new 007] AILA--First Experiments with Localist Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AILA模型，首次在Transformer中实现可调局部表征，通过参数λ在局部与分布式编码间动态插值，提升可解释性同时保持性能，解决语言模型透明性与效率的权衡问题。**

- **链接: [http://arxiv.org/pdf/2511.03559v1](http://arxiv.org/pdf/2511.03559v1)**

> **作者:** Joachim Diederich
>
> **摘要:** This paper presents the first empirical demonstration of controllable locality in transformer language models, a novel architectural framework that enables continuous control over the degree of representation localization through a tunable locality dial parameter. Unlike traditional language models that rely exclusively on distributed representations, our approach allows dynamic interpolation between highly interpretable localist encodings and efficient distributed representations without requiring model retraining. We conducted experiments on the WikiText corpus using a two-layer transformer architecture, systematically varying the locality parameter {\lambda} across the full spectrum from 1.0 (fully localist) to 0.0 (fully distributed). Our results demonstrate that localist configurations achieve dramatically lower attention entropy, with {\lambda} = 1.0 yielding 5.36 bits compared to 7.18 bits at {\lambda} = 0.0, while maintaining substantially higher pointer fidelity scores reflecting stronger alignment with rule-specified targets. Prediction experiments reveal that intermediate locality values optimize the tradeoff between interpretability and performance, with {\lambda} = 0.6 achieving test perplexity of 4.65 and accuracy of 84.7%. These findings establish that localist language models provide a practical framework for applications in regulated domains requiring both transparency and capability, offering precise mathematical control over the interpretability-performance spectrum through explicit penalty thresholds and information-theoretic design principles.
>
---
#### [new 008] Towards Transparent Stance Detection: A Zero-Shot Approach Using Implicit and Explicit Interpretability
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对零样本立场检测（ZSSD）中解释性差、泛化弱的问题，提出IRIS框架，通过隐式（文本序列）与显式（语言特征）推理机制，将立场检测建模为排序任务，无需标注推理依据即可实现可解释且高泛化的预测。**

- **链接: [http://arxiv.org/pdf/2511.03635v1](http://arxiv.org/pdf/2511.03635v1)**

> **作者:** Apoorva Upadhyaya; Wolfgang Nejdl; Marco Fisichella
>
> **备注:** Accepted in AAAI CONFERENCE ON WEB AND SOCIAL MEDIA (ICWSM 2026)
>
> **摘要:** Zero-Shot Stance Detection (ZSSD) identifies the attitude of the post toward unseen targets. Existing research using contrastive, meta-learning, or data augmentation suffers from generalizability issues or lack of coherence between text and target. Recent works leveraging large language models (LLMs) for ZSSD focus either on improving unseen target-specific knowledge or generating explanations for stance analysis. However, most of these works are limited by their over-reliance on explicit reasoning, provide coarse explanations that lack nuance, and do not explicitly model the reasoning process, making it difficult to interpret the model's predictions. To address these issues, in our study, we develop a novel interpretable ZSSD framework, IRIS. We provide an interpretable understanding of the attitude of the input towards the target implicitly based on sequences within the text (implicit rationales) and explicitly based on linguistic measures (explicit rationales). IRIS considers stance detection as an information retrieval ranking task, understanding the relevance of implicit rationales for different stances to guide the model towards correct predictions without requiring the ground-truth of rationales, thus providing inherent interpretability. In addition, explicit rationales based on communicative features help decode the emotional and cognitive dimensions of stance, offering an interpretable understanding of the author's attitude towards the given target. Extensive experiments on the benchmark datasets of VAST, EZ-STANCE, P-Stance, and RFD using 50%, 30%, and even 10% training data prove the generalizability of our model, benefiting from the proposed architecture and interpretable design.
>
---
#### [new 009] ASVRI-Legal: Fine-Tuning LLMs with Retrieval Augmented Generation for Enhanced Legal Regulation
- **分类: cs.CL**

- **简介: 该论文面向法律监管任务，旨在提升LLMs对法律文本的理解与法规起草能力。通过构建法律领域专用数据集进行微调，并融合RAG技术接入实时法律知识，显著增强法律研究与政策制定的效率与准确性。**

- **链接: [http://arxiv.org/pdf/2511.03563v1](http://arxiv.org/pdf/2511.03563v1)**

> **作者:** One Octadion; Bondan Sapta Prakoso; Nanang Yudi Setiawan; Novanto Yudistira
>
> **备注:** 11 pages (including references), 2 figures, 4 tables, published in Atlantis Press (Open Access under CC BY-NC 4.0 license)
>
> **摘要:** In this study, we explore the fine-tuning of Large Language Models (LLMs) to better support policymakers in their crucial work of understanding, analyzing, and crafting legal regulations. To equip the model with a deep understanding of legal texts, we curated a supervised dataset tailored to the specific needs of the legal domain. Additionally, we integrated the Retrieval-Augmented Generation (RAG) method, enabling the LLM to access and incorporate up-to-date legal knowledge from external sources. This combination of fine-tuning and RAG-based augmentation results in a tool that not only processes legal information but actively assists policymakers in interpreting regulations and drafting new ones that align with current needs. The results demonstrate that this approach can significantly enhance the effectiveness of legal research and regulation development, offering a valuable resource in the ever-evolving field of law.
>
---
#### [new 010] IndicSuperTokenizer: An Optimized Tokenizer for Indic Multilingual LLMs
- **分类: cs.CL**

- **简介: 论文提出IndicSuperTokenizer，面向印地语系多语言LLM，解决传统子词分词器在形态丰富语言中效率低的问题，融合子词与多词分词及语言预处理，显著提升分词效率与推理吞吐量，达新SOTA。**

- **链接: [http://arxiv.org/pdf/2511.03237v1](http://arxiv.org/pdf/2511.03237v1)**

> **作者:** Souvik Rana; Arul Menezes; Ashish Kulkarni; Chandra Khatri; Shubham Agarwal
>
> **摘要:** Tokenizers play a crucial role in determining the performance, training efficiency, and the inference cost of Large Language Models (LLMs). Designing effective tokenizers for multilingual LLMs is particularly challenging due to diverse scripts and rich morphological variation. While subword methods such as Byte Pair Encoding (BPE) are widely adopted, their effectiveness in multilingual settings remains underexplored. We present IndicSuperTokenizer, a tokenizer for Indic multilingual LLMs, that combines both subword and multi-word tokenization, along with language-specific pre-tokenization, leading to more linguistically aligned tokens and achieving a new state-of-the-art in fertility score. Evaluated across English, 22 Indian languages and code data, our tokenizer improves the average fertility score by 39.5% over LLaMA4 and by 18% over Sutra (the current best). This translates to 44% improvement in inference throughput over LLaMA4 while maintaining comparable performance on English and Indic benchmarks. We also present detailed ablations across tokenizer training data size, vocabulary size, merging techniques, and pre-tokenization strategies, demonstrating the robustness of our design choices.
>
---
#### [new 011] Kastor: Fine-tuned Small Language Models for Shape-based Active Relation Extraction
- **分类: cs.CL; I.2.4; I.2.7**

- **简介: Kastor针对知识库补全任务，提出一种基于SHACL形状的主动关系抽取框架，通过优化属性组合选择与迭代学习，提升小语言模型在稀疏数据下的泛化能力与事实发现性能。**

- **链接: [http://arxiv.org/pdf/2511.03466v1](http://arxiv.org/pdf/2511.03466v1)**

> **作者:** Ringwald Celian; Gandon Fabien; Faron Catherine; Michel Franck; Abi Akl Hanna
>
> **备注:** Accepted at ESWC 2025
>
> **摘要:** RDF pattern-based extraction is a compelling approach for fine-tuning small language models (SLMs) by focusing a relation extraction task on a specified SHACL shape. This technique enables the development of efficient models trained on limited text and RDF data. In this article, we introduce Kastor, a framework that advances this approach to meet the demands for completing and refining knowledge bases in specialized domains. Kastor reformulates the traditional validation task, shifting from single SHACL shape validation to evaluating all possible combinations of properties derived from the shape. By selecting the optimal combination for each training example, the framework significantly enhances model generalization and performance. Additionally, Kastor employs an iterative learning process to refine noisy knowledge bases, enabling the creation of robust models capable of uncovering new, relevant facts
>
---
#### [new 012] Cache Mechanism for Agent RAG Systems
- **分类: cs.CL**

- **简介: 该论文提出ARC缓存机制，解决Agent RAG系统中动态维护高效小规模知识库的问题，通过无标注方式融合查询分布与嵌入几何结构，显著降低存储与延迟，提升检索效率与准确率。**

- **链接: [http://arxiv.org/pdf/2511.02919v1](http://arxiv.org/pdf/2511.02919v1)**

> **作者:** Shuhang Lin; Zhencan Peng; Lingyao Li; Xiao Lin; Xi Zhu; Yongfeng Zhang
>
> **摘要:** Recent advances in Large Language Model (LLM)-based agents have been propelled by Retrieval-Augmented Generation (RAG), which grants the models access to vast external knowledge bases. Despite RAG's success in improving agent performance, agent-level cache management, particularly constructing, maintaining, and updating a compact, relevant corpus dynamically tailored to each agent's need, remains underexplored. Therefore, we introduce ARC (Agent RAG Cache Mechanism), a novel, annotation-free caching framework that dynamically manages small, high-value corpora for each agent. By synthesizing historical query distribution patterns with the intrinsic geometry of cached items in the embedding space, ARC automatically maintains a high-relevance cache. With comprehensive experiments on three retrieval datasets, our experimental results demonstrate that ARC reduces storage requirements to 0.015% of the original corpus while offering up to 79.8% has-answer rate and reducing average retrieval latency by 80%. Our results demonstrate that ARC can drastically enhance efficiency and effectiveness in RAG-powered LLM agents.
>
---
#### [new 013] A Computational Approach to Analyzing Disrupted Language in Schizophrenia: Integrating Surprisal and Coherence Measures
- **分类: cs.CL; eess.AS; eess.SP**

- **简介: 该论文属于计算语言学与精神医学交叉任务，旨在通过 surprisal 和语义连贯性量化精神分裂症患者的语言紊乱，区分患者与健康对照，并关联症状严重程度，为客观诊断提供计算指标。**

- **链接: [http://arxiv.org/pdf/2511.03089v1](http://arxiv.org/pdf/2511.03089v1)**

> **作者:** Gowtham Premananth; Carol Espy-Wilson
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Language disruptions are one of the well-known effects of schizophrenia symptoms. They are often manifested as disorganized speech and impaired discourse coherence. These abnormalities in spontaneous language production reflect underlying cognitive disturbances and have the potential to serve as objective markers for symptom severity and diagnosis of schizophrenia. This study focuses on how these language disruptions can be characterized in terms of two computational linguistic measures: surprisal and semantic coherence. By computing surprisal and semantic coherence of language using computational models, this study investigates how they differ between subjects with schizophrenia and healthy controls. Furthermore, this study provides further insight into how language disruptions in terms of these linguistic measures change with varying degrees of schizophrenia symptom severity.
>
---
#### [new 014] Generative Artificial Intelligence in Bioinformatics: A Systematic Review of Models, Applications, and Methodological Advances
- **分类: cs.CL; cs.AI**

- **简介: 该论文为系统性综述，旨在评估生成式AI在生物信息学中的模型、应用与方法进展，解决其在基因组、蛋白组等领域的有效性与局限性问题，通过六项研究问题系统分析了模型性能、数据资源及未来方向。**

- **链接: [http://arxiv.org/pdf/2511.03354v1](http://arxiv.org/pdf/2511.03354v1)**

> **作者:** Riasad Alvi; Sayeem Been Zaman; Wasimul Karim; Arefin Ittesafun Abian; Mohaimenul Azam Khan Raiaan; Saddam Mukta; Md Rafi Ur Rashid; Md Rafiqul Islam; Yakub Sebastian; Sami Azam
>
> **摘要:** Generative artificial intelligence (GenAI) has become a transformative approach in bioinformatics that often enables advancements in genomics, proteomics, transcriptomics, structural biology, and drug discovery. To systematically identify and evaluate these growing developments, this review proposed six research questions (RQs), according to the preferred reporting items for systematic reviews and meta-analysis methods. The objective is to evaluate impactful GenAI strategies in methodological advancement, predictive performance, and specialization, and to identify promising approaches for advanced modeling, data-intensive discovery, and integrative biological analysis. RQ1 highlights diverse applications across multiple bioinformatics subfields (sequence analysis, molecular design, and integrative data modeling), which demonstrate superior performance over traditional methods through pattern recognition and output generation. RQ2 reveals that adapted specialized model architectures outperformed general-purpose models, an advantage attributed to targeted pretraining and context-aware strategies. RQ3 identifies significant benefits in the bioinformatics domains, focusing on molecular analysis and data integration, which improves accuracy and reduces errors in complex analysis. RQ4 indicates improvements in structural modeling, functional prediction, and synthetic data generation, validated by established benchmarks. RQ5 suggests the main constraints, such as the lack of scalability and biases in data that impact generalizability, and proposes future directions focused on robust evaluation and biologically grounded modeling. RQ6 examines that molecular datasets (such as UniProtKB and ProteinNet12), cellular datasets (such as CELLxGENE and GTEx) and textual resources (such as PubMedQA and OMIM) broadly support the training and generalization of GenAI models.
>
---
#### [new 015] SOLVE-Med: Specialized Orchestration for Leading Vertical Experts across Medical Specialties
- **分类: cs.CL; cs.AI**

- **简介: SOLVE-Med提出一种多智能体架构，用于医疗问答任务，解决大模型幻觉、算力高与领域专精不足问题。通过10个小型专科模型与路由、协调智能体协同，实现高效、精准、可本地部署的医学问答。**

- **链接: [http://arxiv.org/pdf/2511.03542v1](http://arxiv.org/pdf/2511.03542v1)**

> **作者:** Roberta Di Marino; Giovanni Dioguardi; Antonio Romano; Giuseppe Riccio; Mariano Barone; Marco Postiglione; Flora Amato; Vincenzo Moscato
>
> **摘要:** Medical question answering systems face deployment challenges including hallucinations, bias, computational demands, privacy concerns, and the need for specialized expertise across diverse domains. Here, we present SOLVE-Med, a multi-agent architecture combining domain-specialized small language models for complex medical queries. The system employs a Router Agent for dynamic specialist selection, ten specialized models (1B parameters each) fine-tuned on specific medical domains, and an Orchestrator Agent that synthesizes responses. Evaluated on Italian medical forum data across ten specialties, SOLVE-Med achieves superior performance with ROUGE-1 of 0.301 and BERTScore F1 of 0.697, outperforming standalone models up to 14B parameters while enabling local deployment. Our code is publicly available on GitHub: https://github.com/PRAISELab-PicusLab/SOLVE-Med.
>
---
#### [new 016] Overcoming the Generalization Limits of SLM Finetuning for Shape-Based Extraction of Datatype and Object Properties
- **分类: cs.CL; I.2.7; I.2.4**

- **简介: 该论文面向基于SHACL形状的RDF三元组抽取任务，解决SLM在长尾属性上泛化差的问题，通过对比多种策略，发现按属性频次阈值构建训练集最有效，并开源了数据与代码。**

- **链接: [http://arxiv.org/pdf/2511.03407v1](http://arxiv.org/pdf/2511.03407v1)**

> **作者:** Célian Ringwald; Fabien Gandon; Catherine Faron; Franck Michel; Hanna Abi Akl
>
> **备注:** Accepted at KCAP 2025
>
> **摘要:** Small language models (SLMs) have shown promises for relation extraction (RE) when extracting RDF triples guided by SHACL shapes focused on common datatype properties. This paper investigates how SLMs handle both datatype and object properties for a complete RDF graph extraction. We show that the key bottleneck is related to long-tail distribution of rare properties. To solve this issue, we evaluate several strategies: stratified sampling, weighted loss, dataset scaling, and template-based synthetic data augmentation. We show that the best strategy to perform equally well over unbalanced target properties is to build a training set where the number of occurrences of each property exceeds a given threshold. To enable reproducibility, we publicly released our datasets, experimental results and code. Our findings offer practical guidance for training shape-aware SLMs and highlight promising directions for future work in semantic RE.
>
---
#### [new 017] Efficient Reasoning via Thought-Training and Thought-Free Inference
- **分类: cs.CL; I.2.7**

- **简介: 该论文提出3TF框架，解决LLMs推理效率低问题：通过训练模型内化推理能力，实现无显式思维链的高效推理，在保持输出简洁的同时提升推理质量。**

- **链接: [http://arxiv.org/pdf/2511.03408v1](http://arxiv.org/pdf/2511.03408v1)**

> **作者:** Canhui Wu; Qiong Cao; Chao Xue; Wei Xi; Xiaodong He
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Recent advances in large language models (LLMs) have leveraged explicit Chain-of-Thought (CoT) prompting to improve reasoning accuracy. However, most existing methods primarily compress verbose reasoning outputs. These Long-to-Short transformations aim to improve efficiency, but still rely on explicit reasoning during inference. In this work, we introduce \textbf{3TF} (\textbf{T}hought-\textbf{T}raining and \textbf{T}hought-\textbf{F}ree inference), a framework for efficient reasoning that takes a Short-to-Long perspective. We first train a hybrid model that can operate in both reasoning and non-reasoning modes, and then further train it on CoT-annotated data to internalize structured reasoning, while enforcing concise, thought-free outputs at inference time using the no-reasoning mode. Unlike compression-based approaches, 3TF improves the reasoning quality of non-reasoning outputs, enabling models to perform rich internal reasoning implicitly while keeping external outputs short. Empirically, 3TF-trained models obtain large improvements on reasoning benchmarks under thought-free inference, demonstrating that high quality reasoning can be learned and executed implicitly without explicit step-by-step generation.
>
---
#### [new 018] CARMA: Comprehensive Automatically-annotated Reddit Mental Health Dataset for Arabic
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CARMA，首个自动标注的阿拉伯语Reddit心理健康数据集，解决阿拉伯语心理健康检测数据稀缺问题，涵盖六类病症，通过语言分析与分类实验验证其在低资源语言中的应用潜力。**

- **链接: [http://arxiv.org/pdf/2511.03102v1](http://arxiv.org/pdf/2511.03102v1)**

> **作者:** Saad Mankarious; Ayah Zirikly
>
> **摘要:** Mental health disorders affect millions worldwide, yet early detection remains a major challenge, particularly for Arabic-speaking populations where resources are limited and mental health discourse is often discouraged due to cultural stigma. While substantial research has focused on English-language mental health detection, Arabic remains significantly underexplored, partly due to the scarcity of annotated datasets. We present CARMA, the first automatically annotated large-scale dataset of Arabic Reddit posts. The dataset encompasses six mental health conditions, such as Anxiety, Autism, and Depression, and a control group. CARMA surpasses existing resources in both scale and diversity. We conduct qualitative and quantitative analyses of lexical and semantic differences between users, providing insights into the linguistic markers of specific mental health conditions. To demonstrate the dataset's potential for further mental health analysis, we perform classification experiments using a range of models, from shallow classifiers to large language models. Our results highlight the promise of advancing mental health detection in underrepresented languages such as Arabic.
>
---
#### [new 019] ChiMDQA: Towards Comprehensive Chinese Document QA with Fine-grained Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ChiMDQA，首个面向中文多文档问答的细粒度评估数据集，覆盖六大学科领域，构建6068高质量QA对，解决中文QA数据稀缺与评估粗粒度问题，支持文档理解与智能问答研究。**

- **链接: [http://arxiv.org/pdf/2511.03656v1](http://arxiv.org/pdf/2511.03656v1)**

> **作者:** Jing Gao; Shutiao Luo; Yumeng Liu; Yuanming Li; Hongji Zeng
>
> **备注:** 13 pages, 6 tables, 4 figures, accepted by ICANN 2025
>
> **摘要:** With the rapid advancement of natural language processing (NLP) technologies, the demand for high-quality Chinese document question-answering datasets is steadily growing. To address this issue, we present the Chinese Multi-Document Question Answering Dataset(ChiMDQA), specifically designed for downstream business scenarios across prevalent domains including academic, education, finance, law, medical treatment, and news. ChiMDQA encompasses long-form documents from six distinct fields, consisting of 6,068 rigorously curated, high-quality question-answer (QA) pairs further classified into ten fine-grained categories. Through meticulous document screening and a systematic question-design methodology, the dataset guarantees both diversity and high quality, rendering it applicable to various NLP tasks such as document comprehension, knowledge extraction, and intelligent QA systems. Additionally, this paper offers a comprehensive overview of the dataset's design objectives, construction methodologies, and fine-grained evaluation system, supplying a substantial foundation for future research and practical applications in Chinese QA. The code and data are available at: https://anonymous.4open.science/r/Foxit-CHiMDQA/.
>
---
#### [new 020] MultiZebraLogic: A Multilingual Logical Reasoning Benchmark
- **分类: cs.CL; cs.AI**

- **简介: 论文提出MultiZebraLogic，构建多语言逻辑推理基准，通过生成多语言、多主题的“斑马谜题”评估LLM推理能力，涵盖14种线索与8种干扰项，发布128+1024道谜题数据集及生成代码，支持多语言扩展。**

- **链接: [http://arxiv.org/pdf/2511.03553v1](http://arxiv.org/pdf/2511.03553v1)**

> **作者:** Sofie Helene Bruun; Dan Saattrup Smart
>
> **备注:** Submitted to LREC 2026
>
> **摘要:** Measuring the full abilities of large language models (LLMs) requires benchmarks representing multiple tasks. We aim to create large, high-quality datasets for comparison of logical reasoning skills across several languages and of suitable difficulty for LLMs of various reasoning ability. We explore multiple ways of increasing difficulty. We generate zebra puzzles in multiple languages, themes, sizes and including 14 different clue types and 8 red herring types (uninformative clues). We find puzzle sizes 2x3 and 4x5 are sufficiently challenging for GPT-4o mini (a non-reasoning model) and o3-mini (a reasoning model), respectively. Including 5 red herrings decreases o3-mini puzzle-level accuracy on 4x5 puzzles by 15$\pm$7 %. Scores of o3-mini on 4x5 puzzles are not significantly affected by use of English vs. Danish or the common houses theme vs. the country-specific smoerrebroed theme. We find no correlation between difficulty and the selected clue types. Datasets of 128+1024 puzzles are published as MultiZebraLogic in each of nine Germanic languages for sizes 2x3 and 4x5. We publish code for puzzle generation, designed for adaptablity into more languages and themes.
>
---
#### [new 021] LGM: Enhancing Large Language Models with Conceptual Meta-Relations and Iterative Retrieval
- **分类: cs.CL; cs.AI**

- **简介: 论文提出LGM，通过提取概念元关系（继承、别名、组合）并结合迭代检索，提升大语言模型对模糊概念的理解能力，解决传统RAG依赖长上下文的问题，无需截断即可处理任意长度文本。**

- **链接: [http://arxiv.org/pdf/2511.03214v1](http://arxiv.org/pdf/2511.03214v1)**

> **作者:** Wenchang Lei; Ping Zou; Yue Wang; Feng Sun; Lei Zhao
>
> **备注:** 30 pages, 5 figures
>
> **摘要:** Large language models (LLMs) exhibit strong semantic understanding, yet struggle when user instructions involve ambiguous or conceptually misaligned terms. We propose the Language Graph Model (LGM) to enhance conceptual clarity by extracting meta-relations-inheritance, alias, and composition-from natural language. The model further employs a reflection mechanism to validate these meta-relations. Leveraging a Concept Iterative Retrieval Algorithm, these relations and related descriptions are dynamically supplied to the LLM, improving its ability to interpret concepts and generate accurate responses. Unlike conventional Retrieval-Augmented Generation (RAG) approaches that rely on extended context windows, our method enables large language models to process texts of any length without the need for truncation. Experiments on standard benchmarks demonstrate that the LGM consistently outperforms existing RAG baselines.
>
---
#### [new 022] Hybrid Fact-Checking that Integrates Knowledge Graphs, Large Language Models, and Search-Based Retrieval Agents Improves Interpretable Claim Verification
- **分类: cs.CL; cs.AI; cs.CY; cs.IR; 68T50; I.2.7; H.3.3**

- **简介: 该论文提出一种混合事实核查系统，整合知识图谱、大语言模型与搜索代理，解决LLM缺乏可靠依据、知识图谱覆盖不足的问题，实现高精度、可解释的声明验证，并提升对“信息不足”类声明的识别能力。**

- **链接: [http://arxiv.org/pdf/2511.03217v1](http://arxiv.org/pdf/2511.03217v1)**

> **作者:** Shaghayegh Kolli; Richard Rosenbaum; Timo Cavelius; Lasse Strothe; Andrii Lata; Jana Diesner
>
> **备注:** Paper has been accepted at 9th wiNLP workshop at EMNLP
>
> **摘要:** Large language models (LLMs) excel in generating fluent utterances but can lack reliable grounding in verified information. At the same time, knowledge-graph-based fact-checkers deliver precise and interpretable evidence, yet suffer from limited coverage or latency. By integrating LLMs with knowledge graphs and real-time search agents, we introduce a hybrid fact-checking approach that leverages the individual strengths of each component. Our system comprises three autonomous steps: 1) a Knowledge Graph (KG) Retrieval for rapid one - hop lookups in DBpedia, 2) an LM-based classification guided by a task-specific labeling prompt, producing outputs with internal rule-based logic, and 3) a Web Search Agent invoked only when KG coverage is insufficient. Our pipeline achieves an F1 score of 0.93 on the FEVER benchmark on the Supported/Refuted split without task- specific fine - tuning. To address Not enough information cases, we conduct a targeted reannotation study showing that our approach frequently uncovers valid evidence for claims originally labeled as Not Enough Information (NEI), as confirmed by both expert annotators and LLM reviewers. With this paper, we present a modular, opensource fact-checking pipeline with fallback strategies and generalization across datasets.
>
---
#### [new 023] SCALE: Upscaled Continual Learning of Large Language Models
- **分类: cs.CL**

- **简介: 论文提出SCALE，一种宽度扩展架构，通过冻结预训练参数并插入轻量扩展模块，实现大语言模型的持续预训练，缓解灾难性遗忘，平衡稳定性与可塑性。**

- **链接: [http://arxiv.org/pdf/2511.03270v1](http://arxiv.org/pdf/2511.03270v1)**

> **作者:** Jin-woo Lee; Junhwa Choi; Bongkyu Hwang; Jinho Choo; Bogun Kim; JeongSeon Yi; Joonseok Lee; DongYoung Jung; Jaeseon Park; Kyoungwon Park; Suk-hoon Jung
>
> **摘要:** We revisit continual pre-training for large language models and argue that progress now depends more on scaling the right structure than on scaling parameters alone. We introduce SCALE, a width upscaling architecture that inserts lightweight expansion into linear modules while freezing all pre-trained parameters. This preserves the residual and attention topologies and increases capacity without perturbing the base model's original functionality. SCALE is guided by two principles: Persistent Preservation, which maintains the base model's behavior via preservation-oriented initialization and freezing of the pre-trained weights, and Collaborative Adaptation, which selectively trains a subset of expansion components to acquire new knowledge with minimal interference. We instantiate these ideas as SCALE-Preserve (preservation-first), SCALE-Adapt (adaptation-first), and SCALE-Route, an optional routing extension that performs token-level routing between preservation and adaptation heads. On a controlled synthetic biography benchmark, SCALE mitigates the severe forgetting observed with depth expansion while still acquiring new knowledge. In continual pre-training on a Korean corpus, SCALE variants achieve less forgetting on English evaluations and competitive gains on Korean benchmarks, with these variants offering the best overall stability-plasticity trade-off. Accompanying analysis clarifies when preservation provably holds and why the interplay between preservation and adaptation stabilizes optimization compared to standard continual learning setups.
>
---
#### [new 024] Targeted Error Correction in Knowledge Distillation: Small Language Models Surpass GPT
- **分类: cs.CL**

- **简介: 该论文针对知识蒸馏中的错误传播问题，提出ARF管道：分析GPT-3.5的摘要错误，用Llama 3.1 70B修正，再用修正数据微调Llama 3.1 8B，使其在客服摘要任务上超越GPT-3.5，提升效率与隐私。**

- **链接: [http://arxiv.org/pdf/2511.03005v1](http://arxiv.org/pdf/2511.03005v1)**

> **作者:** Hee-Jin Lee; Zhen Guo; Luchao Jin; Morteza Moazami Goudarzi
>
> **摘要:** We introduce an Analyze-Revise-Finetune (ARF) pipeline that enables smaller open-source language models (LLMs) to surpass substantially larger proprietary models in customer service summarization tasks. The pipeline first analyzes and categorizes common errors in summaries produced by a teacher model (GPT-3.5), then performs a targeted revision using a compact editor model (Llama 3.1 70B) to generate high-quality, refined training data. Fine-tuning a smaller student model (Llama 3.1 8B) on this refined data resulted in superior summarization performance compared to GPT-3.5. The ARF pipeline improves cost efficiency and data privacy while maintaining competitive accuracy, illustrating a generalizable framework for enhancing open-source LLMs across diverse downstream applications.
>
---
#### [new 025] Bearing Syntactic Fruit with Stack-Augmented Neural Networks
- **分类: cs.CL**

- **简介: 该论文研究神经网络如何像人类一样习得句法结构，提出栈增强神经网络（如带非确定性栈的Transformer）无需监督或预训练即可实现类人句法泛化，并在疑问句生成任务中验证其有效性。**

- **链接: [http://arxiv.org/pdf/2511.03547v1](http://arxiv.org/pdf/2511.03547v1)**

> **作者:** Brian DuSell; Ryan Cotterell
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** Any finite set of training data is consistent with an infinite number of hypothetical algorithms that could have generated it. Studies have shown that when human children learn language, they consistently favor hypotheses based on hierarchical syntactic rules without ever encountering disambiguating examples. A recent line of work has inquired as to whether common neural network architectures share this bias, finding that they do so only under special conditions: when syntactically supervised, when pre-trained on massive corpora, or when trained long past convergence. In this paper, we demonstrate, for the first time, neural network architectures that are able to generalize in human-like fashion without any of the aforementioned requirements: stack-augmented neural networks. We test three base architectures (transformer, simple RNN, LSTM) augmented with two styles of stack: the superposition stack of Joulin & Mikolov (2015) and a nondeterministic generalization of it proposed by DuSell & Chiang (2023). We find that transformers with nondeterministic stacks generalize best out of these architectures on a classical question formation task. We also propose a modification to the stack RNN architecture that improves hierarchical generalization. These results suggest that stack-augmented neural networks may be more accurate models of human language acquisition than standard architectures, serving as useful objects of psycholinguistic study. Our code is publicly available.
>
---
#### [new 026] One Battle After Another: Probing LLMs' Limits on Multi-Turn Instruction Following with a Benchmark Evolving Framework
- **分类: cs.CL**

- **简介: 该论文提出一种动态评估大模型多轮指令遵循能力的框架EvolIF，解决传统基准固定、易饱和问题，通过三层次机制模拟用户交互，构建包含九类约束的进化基准，揭示GPT-5性能显著领先。**

- **链接: [http://arxiv.org/pdf/2511.03508v1](http://arxiv.org/pdf/2511.03508v1)**

> **作者:** Qi Jia; Kaiwei Zhang; Xiujie Song; Ye Shen; Xiangyang Zhu; Guangtao Zhai
>
> **摘要:** Understanding how well large language models can follow users' instructions throughout a dialogue spanning multiple topics is of great importance for data-intensive conversational applications. Existing benchmarks are often limited to a fixed number of turns, making them susceptible to saturation and failing to account for the user's interactive experience. In this work, we propose an extensible framework for assessing multi-turn instruction-following ability. At its core, our framework decouples linguistic surface forms from user intent simulation through a three-layer mechanism that tracks constraints, instructions, and topics. This framework mimics User-LLM interaction by enabling the dynamic construction of benchmarks with state changes and tracebacks, terminating a conversation only when the model exhausts a simulated user's patience. We define a suite of metrics capturing the quality of the interaction process. Using this framework, we construct EvolIF, an evolving instruction-following benchmark incorporating nine distinct constraint types. Our results indicate that GPT-5 exhibits superior instruction-following performance. It sustains an average of 18.54 conversational turns and demonstrates 70.31% robustness, outperforming Gemini-2.5-Pro by a significant margin of 11.41%, while other models lag far behind. All of the data and code will be made publicly available online.
>
---
#### [new 027] Automatic Machine Translation Detection Using a Surrogate Multilingual Translation Model
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出一种基于代理多语言翻译模型内部表征的自动检测方法，用于识别机器翻译文本，解决训练数据中合成内容导致翻译质量下降的问题，显著提升非英语对的检测准确率。**

- **链接: [http://arxiv.org/pdf/2511.02958v1](http://arxiv.org/pdf/2511.02958v1)**

> **作者:** Cristian García-Romero; Miquel Esplà-Gomis; Felipe Sánchez-Martínez
>
> **备注:** Pre-MIT Press publication version
>
> **摘要:** Modern machine translation (MT) systems depend on large parallel corpora, often collected from the Internet. However, recent evidence indicates that (i) a substantial portion of these texts are machine-generated translations, and (ii) an overreliance on such synthetic content in training data can significantly degrade translation quality. As a result, filtering out non-human translations is becoming an essential pre-processing step in building high-quality MT systems. In this work, we propose a novel approach that directly exploits the internal representations of a surrogate multilingual MT model to distinguish between human and machine-translated sentences. Experimental results show that our method outperforms current state-of-the-art techniques, particularly for non-English language pairs, achieving gains of at least 5 percentage points of accuracy.
>
---
#### [new 028] Grounded Misunderstandings in Asymmetric Dialogue: A Perspectivist Annotation Scheme for MapTask
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对非对称对话中的指代误解问题，提出一种视角主义标注方案，分析MapTask语料中说话者与听者对指代表达的不同理解，揭示表面共识下的指代错位，并为评估LLM的视角建模能力提供资源与方法。**

- **链接: [http://arxiv.org/pdf/2511.03718v1](http://arxiv.org/pdf/2511.03718v1)**

> **作者:** Nan Li; Albert Gatt; Massimo Poesio
>
> **备注:** 11 pages, 3 figures, 5 tables; under review
>
> **摘要:** Collaborative dialogue relies on participants incrementally establishing common ground, yet in asymmetric settings they may believe they agree while referring to different entities. We introduce a perspectivist annotation scheme for the HCRC MapTask corpus (Anderson et al., 1991) that separately captures speaker and addressee grounded interpretations for each reference expression, enabling us to trace how understanding emerges, diverges, and repairs over time. Using a scheme-constrained LLM annotation pipeline, we obtain 13k annotated reference expressions with reliability estimates and analyze the resulting understanding states. The results show that full misunderstandings are rare once lexical variants are unified, but multiplicity discrepancies systematically induce divergences, revealing how apparent grounding can mask referential misalignment. Our framework provides both a resource and an analytic lens for studying grounded misunderstanding and for evaluating (V)LLMs' capacity to model perspective-dependent grounding in collaborative dialogue.
>
---
#### [new 029] LEGO-Eval: Towards Fine-Grained Evaluation on Synthesizing 3D Embodied Environments with Tool Augmentation
- **分类: cs.CL**

- **简介: 论文提出LEGO-Eval框架与LEGO-Bench基准，解决3D生成场景与细粒度指令对齐评估难题，通过工具增强的场景 grounding，提升评估准确性，揭示现有方法在复杂真实场景生成中成功率不足10%。**

- **链接: [http://arxiv.org/pdf/2511.03001v1](http://arxiv.org/pdf/2511.03001v1)**

> **作者:** Gyeom Hwangbo; Hyungjoo Chae; Minseok Kang; Hyeonjong Ju; Soohyun Oh; Jinyoung Yeo
>
> **备注:** Work in Progress
>
> **摘要:** Despite recent progress in using Large Language Models (LLMs) for automatically generating 3D scenes, generated scenes often lack realistic spatial layouts and object attributes found in real-world environments. As this problem stems from insufficiently detailed, coarse-grained instructions, advancing 3D scene synthesis guided by more detailed, fine-grained instructions that reflect real-world environments becomes crucial. Without such realistic scenes, training embodied agents in unrealistic environments can lead them to learn priors that diverge significantly from real-world physics and semantics, degrading their performance when deployed. Thus, verifying the alignment between the fine-grained instruction and the generated scene is essential for effective learning. However, current evaluation methods, such as CLIPScore and vision-language models (VLMs), often fail to reliably assess such alignment. This shortcoming arises primarily from their shallow understanding of 3D scenes, which often leads to improperly grounded scene components. To address this, we introduce LEGO-Eval, an evaluation framework equipped with diverse tools designed to explicitly ground scene components, enabling more accurate alignment assessments. We also present LEGO-Bench, a benchmark of detailed instructions that specify complex layouts and attributes of real-world environments. Experiments demonstrate that LEGO-Eval outperforms VLM-as-a-judge by 0.41 F1 score in assessing scene-instruction alignment. Benchmarking with LEGO-Bench reveals significant limitations in current generation methods. Across all evaluated approaches, success rates reached at most 10% in generating scenes that fully align with fine-grained instructions.
>
---
#### [new 030] CareMedEval dataset: Evaluating Critical Appraisal and Reasoning in the Biomedical Field
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CareMedEval数据集，用于评估大语言模型在生物医学领域对科学文献的批判性阅读与推理能力，解决现有模型在真实医学场景中推理不足的问题，基于法国医学生考题构建534道题目，揭示当前模型在统计分析与研究局限性判断上的短板。**

- **链接: [http://arxiv.org/pdf/2511.03441v1](http://arxiv.org/pdf/2511.03441v1)**

> **作者:** Doria Bonzi; Alexandre Guiggi; Frédéric Béchet; Carlos Ramisch; Benoit Favre
>
> **备注:** Preprint submitted to LREC 2026 (under review) To access the dataset, see https://github.com/bonzid/CareMedEval
>
> **摘要:** Critical appraisal of scientific literature is an essential skill in the biomedical field. While large language models (LLMs) can offer promising support in this task, their reliability remains limited, particularly for critical reasoning in specialized domains. We introduce CareMedEval, an original dataset designed to evaluate LLMs on biomedical critical appraisal and reasoning tasks. Derived from authentic exams taken by French medical students, the dataset contains 534 questions based on 37 scientific articles. Unlike existing benchmarks, CareMedEval explicitly evaluates critical reading and reasoning grounded in scientific papers. Benchmarking state-of-the-art generalist and biomedical-specialized LLMs under various context conditions reveals the difficulty of the task: open and commercial models fail to exceed an Exact Match Rate of 0.5 even though generating intermediate reasoning tokens considerably improves the results. Yet, models remain challenged especially on questions about study limitations and statistical analysis. CareMedEval provides a challenging benchmark for grounded reasoning, exposing current LLM limitations and paving the way for future development of automated support for critical appraisal.
>
---
#### [new 031] Silenced Biases: The Dark Side LLMs Learned to Refuse
- **分类: cs.CL; stat.ML**

- **简介: 该论文提出“沉默偏见”概念，揭示安全对齐LLMs通过拒绝响应掩盖潜在偏见的问题，构建SBB基准，利用激活引导降低拒绝率，以更真实评估模型公平性，突破传统QA评估的局限。**

- **链接: [http://arxiv.org/pdf/2511.03369v1](http://arxiv.org/pdf/2511.03369v1)**

> **作者:** Rom Himelstein; Amit LeVi; Brit Youngmann; Yaniv Nemcovsky; Avi Mendelson
>
> **摘要:** Safety-aligned large language models (LLMs) are becoming increasingly widespread, especially in sensitive applications where fairness is essential and biased outputs can cause significant harm. However, evaluating the fairness of models is a complex challenge, and approaches that do so typically utilize standard question-answer (QA) styled schemes. Such methods often overlook deeper issues by interpreting the model's refusal responses as positive fairness measurements, which creates a false sense of fairness. In this work, we introduce the concept of silenced biases, which are unfair preferences encoded within models' latent space and are effectively concealed by safety-alignment. Previous approaches that considered similar indirect biases often relied on prompt manipulation or handcrafted implicit queries, which present limited scalability and risk contaminating the evaluation process with additional biases. We propose the Silenced Bias Benchmark (SBB), which aims to uncover these biases by employing activation steering to reduce model refusals during QA. SBB supports easy expansion to new demographic groups and subjects, presenting a fairness evaluation framework that encourages the future development of fair models and tools beyond the masking effects of alignment training. We demonstrate our approach over multiple LLMs, where our findings expose an alarming distinction between models' direct responses and their underlying fairness issues.
>
---
#### [new 032] Who Sees the Risk? Stakeholder Conflicts and Explanatory Policies in LLM-based Risk Assessment
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出一种基于LLM的多方利益相关者风险评估框架，通过可解释方法识别不同群体对AI风险的认知冲突，生成个性化政策，并可视化冲突成因，提升AI决策的透明性与人本治理对齐性。**

- **链接: [http://arxiv.org/pdf/2511.03152v1](http://arxiv.org/pdf/2511.03152v1)**

> **作者:** Srishti Yadav; Jasmina Gajcin; Erik Miehling; Elizabeth Daly
>
> **摘要:** Understanding how different stakeholders perceive risks in AI systems is essential for their responsible deployment. This paper presents a framework for stakeholder-grounded risk assessment by using LLMs, acting as judges to predict and explain risks. Using the Risk Atlas Nexus and GloVE explanation method, our framework generates stakeholder-specific, interpretable policies that shows how different stakeholders agree or disagree about the same risks. We demonstrate our method using three real-world AI use cases of medical AI, autonomous vehicles, and fraud detection domain. We further propose an interactive visualization that reveals how and why conflicts emerge across stakeholder perspectives, enhancing transparency in conflict reasoning. Our results show that stakeholder perspectives significantly influence risk perception and conflict patterns. Our work emphasizes the importance of these stakeholder-aware explanations needed to make LLM-based evaluations more transparent, interpretable, and aligned with human-centered AI governance goals.
>
---
#### [new 033] BengaliMoralBench: A Benchmark for Auditing Moral Reasoning in Large Language Models within Bengali Language and Culture
- **分类: cs.CL**

- **简介: 论文提出BengaliMoralBench，首个孟加拉语道德推理基准，解决西方中心伦理评估忽视本地文化的问题，通过50个文化相关场景与三重伦理框架，评估多语言大模型在孟加拉语语境中的道德对齐能力。**

- **链接: [http://arxiv.org/pdf/2511.03180v1](http://arxiv.org/pdf/2511.03180v1)**

> **作者:** Shahriyar Zaman Ridoy; Azmine Toushik Wasi; Koushik Ahamed Tonmoy
>
> **备注:** This manuscript is a preprint currently under review
>
> **摘要:** As multilingual Large Language Models (LLMs) gain traction across South Asia, their alignment with local ethical norms, particularly for Bengali, which is spoken by over 285 million people and ranked 6th globally, remains underexplored. Existing ethics benchmarks are largely English-centric and shaped by Western frameworks, overlooking cultural nuances critical for real-world deployment. To address this, we introduce BengaliMoralBench, the first large-scale ethics benchmark for the Bengali language and socio-cultural contexts. It covers five moral domains, Daily Activities, Habits, Parenting, Family Relationships, and Religious Activities, subdivided into 50 culturally relevant subtopics. Each scenario is annotated via native-speaker consensus using three ethical lenses: Virtue, Commonsense, and Justice ethics. We conduct systematic zero-shot evaluation of prominent multilingual LLMs, including Llama, Gemma, Qwen, and DeepSeek, using a unified prompting protocol and standard metrics. Performance varies widely (50-91% accuracy), with qualitative analysis revealing consistent weaknesses in cultural grounding, commonsense reasoning, and moral fairness. BengaliMoralBench provides a foundation for responsible localization, enabling culturally aligned evaluation and supporting the deployment of ethically robust AI in diverse, low-resource multilingual settings such as Bangladesh.
>
---
#### [new 034] HaluMem: Evaluating Hallucinations in Memory Systems of Agents
- **分类: cs.CL**

- **简介: 论文提出HaluMem，首个面向记忆系统的操作级幻觉评估基准，解决现有评估无法定位幻觉产生阶段的问题，构建多轮交互数据集，揭示记忆提取与更新阶段是幻觉主要来源。**

- **链接: [http://arxiv.org/pdf/2511.03506v1](http://arxiv.org/pdf/2511.03506v1)**

> **作者:** Ding Chen; Simin Niu; Kehang Li; Peng Liu; Xiangping Zheng; Bo Tang; Xinchi Li; Feiyu Xiong; Zhiyu Li
>
> **摘要:** Memory systems are key components that enable AI systems such as LLMs and AI agents to achieve long-term learning and sustained interaction. However, during memory storage and retrieval, these systems frequently exhibit memory hallucinations, including fabrication, errors, conflicts, and omissions. Existing evaluations of memory hallucinations are primarily end-to-end question answering, which makes it difficult to localize the operational stage within the memory system where hallucinations arise. To address this, we introduce the Hallucination in Memory Benchmark (HaluMem), the first operation level hallucination evaluation benchmark tailored to memory systems. HaluMem defines three evaluation tasks (memory extraction, memory updating, and memory question answering) to comprehensively reveal hallucination behaviors across different operational stages of interaction. To support evaluation, we construct user-centric, multi-turn human-AI interaction datasets, HaluMem-Medium and HaluMem-Long. Both include about 15k memory points and 3.5k multi-type questions. The average dialogue length per user reaches 1.5k and 2.6k turns, with context lengths exceeding 1M tokens, enabling evaluation of hallucinations across different context scales and task complexities. Empirical studies based on HaluMem show that existing memory systems tend to generate and accumulate hallucinations during the extraction and updating stages, which subsequently propagate errors to the question answering stage. Future research should focus on developing interpretable and constrained memory operation mechanisms that systematically suppress hallucinations and improve memory reliability.
>
---
#### [new 035] EQ-Negotiator: Dynamic Emotional Personas Empower Small Language Models for Edge-Deployable Credit Negotiation
- **分类: cs.CL**

- **简介: EQ-Negotiator提出一种动态情感人格框架，利用博弈论与HMM在不预训练下追踪债务人情绪，赋能小语言模型（SLM）实现高效、伦理且隐私安全的边缘端信贷谈判，性能超越超大模型。**

- **链接: [http://arxiv.org/pdf/2511.03370v1](http://arxiv.org/pdf/2511.03370v1)**

> **作者:** Yunbo Long; Yuhan Liu; Alexandra Brintrup
>
> **摘要:** The deployment of large language models (LLMs) in automated negotiation has set a high performance benchmark, but their computational cost and data privacy requirements render them unsuitable for many privacy-sensitive, on-device applications such as mobile assistants, embodied AI agents or private client interactions. While small language models (SLMs) offer a practical alternative, they suffer from a significant performance gap compared to LLMs in playing emotionally charged complex personas, especially for credit negotiation. This paper introduces EQ-Negotiator, a novel framework that bridges this capability gap using emotional personas. Its core is a reasoning system that integrates game theory with a Hidden Markov Model(HMM) to learn and track debtor emotional states online, without pre-training. This allows EQ-Negotiator to equip SLMs with the strategic intelligence to counter manipulation while de-escalating conflict and upholding ethical standards. Through extensive agent-to-agent simulations across diverse credit negotiation scenarios, including adversarial debtor strategies like cheating, threatening, and playing the victim, we show that a 7B parameter language model with EQ-Negotiator achieves better debt recovery and negotiation efficiency than baseline LLMs more than 10 times its size. This work advances persona modeling from descriptive character profiles to dynamic emotional architectures that operate within privacy constraints. Besides, this paper establishes that strategic emotional intelligence, not raw model scale, is the critical factor for success in automated negotiation, paving the way for effective, ethical, and privacy-preserving AI negotiators that can operate on the edge.
>
---
#### [new 036] Beyond Ranked Lists: The SARAL Framework for Cross-Lingual Document Set Retrieval
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出SARAL框架，用于跨语言文档集检索任务，突破传统排序列表限制，直接检索与查询相关的文档集合。在MATERIAL项目中，SARAL在三种语言的多数评估中表现最优。**

- **链接: [http://arxiv.org/pdf/2511.03228v1](http://arxiv.org/pdf/2511.03228v1)**

> **作者:** Shantanu Agarwal; Joel Barry; Elizabeth Boschee; Scott Miller
>
> **摘要:** Machine Translation for English Retrieval of Information in Any Language (MATERIAL) is an IARPA initiative targeted to advance the state of cross-lingual information retrieval (CLIR). This report provides a detailed description of Information Sciences Institute's (ISI's) Summarization and domain-Adaptive Retrieval Across Language's (SARAL's) effort for MATERIAL. Specifically, we outline our team's novel approach to handle CLIR with emphasis in developing an approach amenable to retrieve a query-relevant document \textit{set}, and not just a ranked document-list. In MATERIAL's Phase-3 evaluations, SARAL exceeded the performance of other teams in five out of six evaluation conditions spanning three different languages (Farsi, Kazakh, and Georgian).
>
---
#### [new 037] A systematic review of relation extraction task since the emergence of Transformers
- **分类: cs.CL; A.1; I.2.4; I.2.7**

- **简介: 该论文对2019–2024年基于Transformer的关系抽取（RE）研究进行系统综述，梳理了34篇综述、64个数据集和104个模型，分析方法演进、基准资源与语义网融合，揭示趋势与挑战，为RE领域提供全面参考。**

- **链接: [http://arxiv.org/pdf/2511.03610v1](http://arxiv.org/pdf/2511.03610v1)**

> **作者:** Ringwald Celian; Gandon; Fabien; Faron Catherine; Michel Franck; Abi Akl Hanna
>
> **备注:** Submited at ACM-Computing Surveys + The resulting annotated Zotero bibliography : https://www.zotero.org/groups/6070963/scilex_re_systlitreview/library + SciLEx software: https://github.com/Wimmics/SciLEx
>
> **摘要:** This article presents a systematic review of relation extraction (RE) research since the advent of Transformer-based models. Using an automated framework to collect and annotate publications, we analyze 34 surveys, 64 datasets, and 104 models published between 2019 and 2024. The review highlights methodological advances, benchmark resources, and the integration of semantic web technologies. By consolidating results across multiple dimensions, the study identifies current trends, limitations, and open challenges, offering researchers and practitioners a comprehensive reference for understanding the evolution and future directions of RE.
>
---
#### [new 038] MME-CC: A Challenging Multi-Modal Evaluation Benchmark of Cognitive Capacity
- **分类: cs.CL**

- **简介: 论文提出MME-CC基准，聚焦视觉认知能力评估，填补现有多模态基准在空间、几何与知识推理上的不足，系统评测16个MLLMs，揭示其认知短板与错误模式，推动模型设计以认知能力为核心。**

- **链接: [http://arxiv.org/pdf/2511.03146v1](http://arxiv.org/pdf/2511.03146v1)**

> **作者:** Kaiyuan Zhang; Chenghao Yang; Zhoufutu Wen; Sihang Yuan; Qiuyue Wang; Chaoyi Huang; Guosheng Zhu; He Wang; Huawenyu Lu; Jianing Wen; Jianpeng Jiao; Lishu Luo; Longxiang Liu; Sijin Wu; Xiaolei Zhu; Xuanliang Zhang; Ge Zhang; Yi Lin; Guang Shi; Chaoyou Fu; Wenhao Huang
>
> **摘要:** As reasoning models scale rapidly, the essential role of multimodality in human cognition has come into sharp relief, driving a growing need to probe vision-centric cognitive behaviors. Yet, existing multimodal benchmarks either overemphasize textual reasoning or fall short of systematically capturing vision-centric cognitive behaviors, leaving the cognitive capacity of MLLMs insufficiently assessed. To address this limitation, we introduce MME-CC (Multi-Modal Evaluation benchmark of Cognitive Capacity), a vision-grounded benchmark that organizes 11 representative reasoning tasks into three fundamental categories of visual information: spatial, geometric, and knowledge-based reasoning, and provides fine-grained analyses of MLLMs' cognitive capacity across these dimensions. Based on MME-CC, we conduct extensive experiments over 16 representative MLLMs. Our study reveals that closed-source models currently lead overall (e.g., 42.66 for Gemini-2.5-Pro vs. 30.45 for GLM-4.5V), while spatial and geometric reasoning remain broadly weak (less than or equal to 30%). We further identify common error patterns, including orientation mistakes, fragile cross-view identity persistence, and poor adherence to counterfactual instructions, and observe that Chain-of-Thought typically follows a three-stage process (extract -> reason -> verify) with heavy reliance on visual extraction. We hope this work catalyzes a shift toward treating the cognitive capacity of MLLMs as central to both evaluation and model design.
>
---
#### [new 039] Reading Between the Lines: The One-Sided Conversation Problem
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出“单边对话问题”（1SC），旨在从仅有一方对话记录中推断缺失话语并生成摘要。研究发现，少量上下文与提示策略可提升重建效果，且无需重建即可生成高质量摘要，推动隐私敏感型对话AI发展。**

- **链接: [http://arxiv.org/pdf/2511.03056v1](http://arxiv.org/pdf/2511.03056v1)**

> **作者:** Victoria Ebert; Rishabh Singh; Tuochao Chen; Noah A. Smith; Shyamnath Gollakota
>
> **备注:** 8 pages, 6 figures, 4 tables
>
> **摘要:** Conversational AI is constrained in many real-world settings where only one side of a dialogue can be recorded, such as telemedicine, call centers, and smart glasses. We formalize this as the one-sided conversation problem (1SC): inferring and learning from one side of a conversation. We study two tasks: (1) reconstructing the missing speaker's turns for real-time use cases, and (2) generating summaries from one-sided transcripts. Evaluating prompting and finetuned models on MultiWOZ, DailyDialog, and Candor with both human A/B testing and LLM-as-a-judge metrics, we find that access to one future turn and information about utterance length improves reconstruction, placeholder prompting helps to mitigate hallucination, and while large models generate promising reconstructions with prompting, smaller models require finetuning. Further, high-quality summaries can be generated without reconstructing missing turns. We present 1SC as a novel challenge and report promising results that mark a step toward privacy-aware conversational AI.
>
---
#### [new 040] Do Androids Dream of Unseen Puppeteers? Probing for a Conspiracy Mindset in Large Language Models
- **分类: cs.CL; cs.CY**

- **简介: 该论文研究大语言模型（LLMs）是否具备阴谋思维倾向，探讨其是否受社会人口特征影响及易被诱导。通过心理量表实验，发现LLMs易被引导产生阴谋信念，且存在潜在偏见，揭示其社会心理风险。**

- **链接: [http://arxiv.org/pdf/2511.03699v1](http://arxiv.org/pdf/2511.03699v1)**

> **作者:** Francesco Corso; Francesco Pierri; Gianmarco De Francisci Morales
>
> **摘要:** In this paper, we investigate whether Large Language Models (LLMs) exhibit conspiratorial tendencies, whether they display sociodemographic biases in this domain, and how easily they can be conditioned into adopting conspiratorial perspectives. Conspiracy beliefs play a central role in the spread of misinformation and in shaping distrust toward institutions, making them a critical testbed for evaluating the social fidelity of LLMs. LLMs are increasingly used as proxies for studying human behavior, yet little is known about whether they reproduce higher-order psychological constructs such as a conspiratorial mindset. To bridge this research gap, we administer validated psychometric surveys measuring conspiracy mindset to multiple models under different prompting and conditioning strategies. Our findings reveal that LLMs show partial agreement with elements of conspiracy belief, and conditioning with socio-demographic attributes produces uneven effects, exposing latent demographic biases. Moreover, targeted prompts can easily shift model responses toward conspiratorial directions, underscoring both the susceptibility of LLMs to manipulation and the potential risks of their deployment in sensitive contexts. These results highlight the importance of critically evaluating the psychological dimensions embedded in LLMs, both to advance computational social science and to inform possible mitigation strategies against harmful uses.
>
---
#### [new 041] How to Evaluate Speech Translation with Source-Aware Neural MT Metrics
- **分类: cs.CL; cs.AI**

- **简介: 该论文面向语音翻译（ST）评估任务，解决参考式评估忽略源语音信息的问题，提出利用ASR转录和反译文本作为源文本代理，并设计跨语言重分段算法，实现源感知MT指标在ST中的有效应用。**

- **链接: [http://arxiv.org/pdf/2511.03295v1](http://arxiv.org/pdf/2511.03295v1)**

> **作者:** Mauro Cettolo; Marco Gaido; Matteo Negri; Sara Papi; Luisa Bentivogli
>
> **摘要:** Automatic evaluation of speech-to-text translation (ST) systems is typically performed by comparing translation hypotheses with one or more reference translations. While effective to some extent, this approach inherits the limitation of reference-based evaluation that ignores valuable information from the source input. In machine translation (MT), recent progress has shown that neural metrics incorporating the source text achieve stronger correlation with human judgments. Extending this idea to ST, however, is not trivial because the source is audio rather than text, and reliable transcripts or alignments between source and references are often unavailable. In this work, we conduct the first systematic study of source-aware metrics for ST, with a particular focus on real-world operating conditions where source transcripts are not available. We explore two complementary strategies for generating textual proxies of the input audio, automatic speech recognition (ASR) transcripts, and back-translations of the reference translation, and introduce a novel two-step cross-lingual re-segmentation algorithm to address the alignment mismatch between synthetic sources and reference translations. Our experiments, carried out on two ST benchmarks covering 79 language pairs and six ST systems with diverse architectures and performance levels, show that ASR transcripts constitute a more reliable synthetic source than back-translations when word error rate is below 20%, while back-translations always represent a computationally cheaper but still effective alternative. Furthermore, our cross-lingual re-segmentation algorithm enables robust use of source-aware MT metrics in ST evaluation, paving the way toward more accurate and principled evaluation methodologies for speech translation.
>
---
#### [new 042] Comparing the Performance of LLMs in RAG-based Question-Answering: A Case Study in Computer Science Literature
- **分类: cs.CL; cs.AI; I.2.1; I.2.7**

- **简介: 该论文研究RAG框架下不同LLM在计算机科学文献问答中的性能，对比GPT-3.5与四个开源模型在准确率、延迟等指标上的表现，发现Mistral-7b-instruct表现最优，Orca-mini延迟最低，验证开源模型可与商业模型媲美。**

- **链接: [http://arxiv.org/pdf/2511.03261v1](http://arxiv.org/pdf/2511.03261v1)**

> **作者:** Ranul Dayarathne; Uvini Ranaweera; Upeksha Ganegoda
>
> **备注:** 18 pages, 4 figures, 5 tables, presented at the 5th International Conference on Artificial Intelligence in Education Technology
>
> **摘要:** Retrieval Augmented Generation (RAG) is emerging as a powerful technique to enhance the capabilities of Generative AI models by reducing hallucination. Thus, the increasing prominence of RAG alongside Large Language Models (LLMs) has sparked interest in comparing the performance of different LLMs in question-answering (QA) in diverse domains. This study compares the performance of four open-source LLMs, Mistral-7b-instruct, LLaMa2-7b-chat, Falcon-7b-instruct and Orca-mini-v3-7b, and OpenAI's trending GPT-3.5 over QA tasks within the computer science literature leveraging RAG support. Evaluation metrics employed in the study include accuracy and precision for binary questions and ranking by a human expert, ranking by Google's AI model Gemini, alongside cosine similarity for long-answer questions. GPT-3.5, when paired with RAG, effectively answers binary and long-answer questions, reaffirming its status as an advanced LLM. Regarding open-source LLMs, Mistral AI's Mistral-7b-instruct paired with RAG surpasses the rest in answering both binary and long-answer questions. However, among the open-source LLMs, Orca-mini-v3-7b reports the shortest average latency in generating responses, whereas LLaMa2-7b-chat by Meta reports the highest average latency. This research underscores the fact that open-source LLMs, too, can go hand in hand with proprietary models like GPT-3.5 with better infrastructure.
>
---
#### [new 043] Control Barrier Function for Aligning Large Language Models
- **分类: cs.CL; cs.AI; cs.SY; eess.SY**

- **简介: 该论文提出一种基于控制屏障函数（CBF）的无微调框架，用于安全干预大语言模型的文本生成，确保输出符合用户期望。通过添加式安全过滤器，在不修改原模型前提下实现对齐任务，支持直接集成评估模型。**

- **链接: [http://arxiv.org/pdf/2511.03121v1](http://arxiv.org/pdf/2511.03121v1)**

> **作者:** Yuya Miyaoka; Masaki Inoue
>
> **摘要:** This paper proposes a control-based framework for aligning large language models (LLMs) by leveraging a control barrier function (CBF) to ensure user-desirable text generation. The presented framework applies the CBF safety filter to the predicted token generated from the baseline LLM, to intervene in the generated text. The safety filter includes two significant advantages: this safety filter is an add-on type, allowing it to be used for alignment purposes without fine-tuning the baseline LLM, and if there is an evaluation model regarding the desired alignment, it can be directly applied to the filter design. The overall text-generation system is implemented with open-source language models, aiming to generate positive text.
>
---
#### [new 044] Measuring Aleatoric and Epistemic Uncertainty in LLMs: Empirical Evaluation on ID and OOD QA Tasks
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLMs）在问答任务中对认知与偶然不确定性的估计，对比十二种不确定性度量方法在分布内与分布外数据上的表现，揭示不同方法的适用场景。**

- **链接: [http://arxiv.org/pdf/2511.03166v1](http://arxiv.org/pdf/2511.03166v1)**

> **作者:** Kevin Wang; Subre Abdoul Moktar; Jia Li; Kangshuo Li; Feng Chen
>
> **备注:** Accepted by UDM-KDD'24
>
> **摘要:** Large Language Models (LLMs) have become increasingly pervasive, finding applications across many industries and disciplines. Ensuring the trustworthiness of LLM outputs is paramount, where Uncertainty Estimation (UE) plays a key role. In this work, a comprehensive empirical study is conducted to examine the robustness and effectiveness of diverse UE measures regarding aleatoric and epistemic uncertainty in LLMs. It involves twelve different UE methods and four generation quality metrics including LLMScore from LLM criticizers to evaluate the uncertainty of LLM-generated answers in Question-Answering (QA) tasks on both in-distribution (ID) and out-of-distribution (OOD) datasets. Our analysis reveals that information-based methods, which leverage token and sequence probabilities, perform exceptionally well in ID settings due to their alignment with the model's understanding of the data. Conversely, density-based methods and the P(True) metric exhibit superior performance in OOD contexts, highlighting their effectiveness in capturing the model's epistemic uncertainty. Semantic consistency methods, which assess variability in generated answers, show reliable performance across different datasets and generation metrics. These methods generally perform well but may not be optimal for every situation.
>
---
#### [new 045] LFC-DA: Logical Formula-Controlled Data Augmentation for Enhanced Logical Reasoning
- **分类: cs.CL; I.2.7; I.2.6; F.4.1**

- **简介: LFC-DA提出一种逻辑公式控制的数据增强方法，解决LLM生成逻辑数据时多样性不足与逻辑错误问题。通过符号逻辑映射与搜索，自动生成符合命题逻辑的高质量问答对，提升模型在ReClor和LogiQA上的推理准确率。**

- **链接: [http://arxiv.org/pdf/2511.03372v1](http://arxiv.org/pdf/2511.03372v1)**

> **作者:** Shenghao Li
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** For complex logical data augmentation, heavy reliance on human annotation is costly, whereas direct generation with large language models yields uninterpretable and logically homogeneous examples. To address this, we present LFC-DA, a symbolic-logic-controlled pipeline: logical text is first mapped to propositional expressions, a compact rule library is compiled, and a bounded state-space search systematically discovers valid formulas that are then verbalized back into natural-language questions, ensuring both diversity and logical rigor under propositional logic. Experiments on ReClor and LogiQA show significant improvements in the logical-reasoning accuracy of pretrained models, confirming the effectiveness of LFC-DA for LLM-guided logical data augmentation.
>
---
#### [new 046] Data-Efficient Adaptation and a Novel Evaluation Method for Aspect-based Sentiment Analysis
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对教育等低资源领域的方面级情感分析（ABSA），提出灵活评估方法FTS-OBP，首次探索小规模生成模型在数据高效场景下的表现，通过多任务微调仅用200-1000样本即超越大模型，并发布首个教育评论ABSA数据集。**

- **链接: [http://arxiv.org/pdf/2511.03034v1](http://arxiv.org/pdf/2511.03034v1)**

> **作者:** Yan Cathy Hua; Paul Denny; Jörg Wicker; Katerina Taškova
>
> **摘要:** Aspect-based Sentiment Analysis (ABSA) is a fine-grained opinion mining approach that identifies and classifies opinions associated with specific entities (aspects) or their categories within a sentence. Despite its rapid growth and broad potential, ABSA research and resources remain concentrated in commercial domains, leaving analytical needs unmet in high-demand yet low-resource areas such as education and healthcare. Domain adaptation challenges and most existing methods' reliance on resource-intensive in-training knowledge injection further hinder progress in these areas. Moreover, traditional evaluation methods based on exact matches are overly rigid for ABSA tasks, penalising any boundary variations which may misrepresent the performance of generative models. This work addresses these gaps through three contributions: 1) We propose a novel evaluation method, Flexible Text Similarity Matching and Optimal Bipartite Pairing (FTS-OBP), which accommodates realistic extraction boundary variations while maintaining strong correlation with traditional metrics and offering fine-grained diagnostics. 2) We present the first ABSA study of small decoder-only generative language models (SLMs; <7B parameters), examining resource lower bounds via a case study in education review ABSA. We systematically explore data-free (in-context learning and weight merging) and data-light fine-tuning methods, and propose a multitask fine-tuning strategy that significantly enhances SLM performance, enabling 1.5-3.8 B models to surpass proprietary large models and approach benchmark results with only 200-1,000 examples on a single GPU. 3) We release the first public set of education review ABSA resources to support future research in low-resource domains.
>
---
#### [new 047] Knowledge-Augmented Question Error Correction for Chinese Question Answer System with QuestionRAG
- **分类: cs.CL**

- **简介: 该论文面向中文问答系统的问句纠错任务，解决LLM易误判意图或过度修改的问题。提出QuestionRAG框架，融合外部知识增强理解，结合强化学习引导精准修正，显著优于传统微调方法。**

- **链接: [http://arxiv.org/pdf/2511.03410v1](http://arxiv.org/pdf/2511.03410v1)**

> **作者:** Longpeng Qiu; Ting Li; Shuai Mao; Nan Yang; Xiaohui Yan
>
> **备注:** EMNLP2025 Industry Track
>
> **摘要:** Input errors in question-answering (QA) systems often lead to incorrect responses. Large language models (LLMs) struggle with this task, frequently failing to interpret user intent (misinterpretation) or unnecessarily altering the original question's structure (over-correction). We propose QuestionRAG, a framework that tackles these problems. To address misinterpretation, it enriches the input with external knowledge (e.g., search results, related entities). To prevent over-correction, it uses reinforcement learning (RL) to align the model's objective with precise correction, not just paraphrasing. Our results demonstrate that knowledge augmentation is critical for understanding faulty questions. Furthermore, RL-based alignment proves significantly more effective than traditional supervised fine-tuning (SFT), boosting the model's ability to follow instructions and generalize. By integrating these two strategies, QuestionRAG unlocks the full potential of LLMs for the question correction task.
>
---
#### [new 048] From Measurement to Expertise: Empathetic Expert Adapters for Context-Based Empathy in Conversational AI Agents
- **分类: cs.HC; cs.AI; cs.CL; cs.CY; cs.LG**

- **简介: 该论文提出情感专家适配器，解决对话AI中同理心泛化不足问题，通过上下文感知的合成数据与适配器训练，实现任务定制化同理心表达，显著缩小用户期望与实际体验的差距。**

- **链接: [http://arxiv.org/pdf/2511.03143v1](http://arxiv.org/pdf/2511.03143v1)**

> **作者:** Erfan Shayegani; Jina Suh; Andy Wilson; Nagu Rangan; Javier Hernandez
>
> **摘要:** Empathy is a critical factor in fostering positive user experiences in conversational AI. While models can display empathy, it is often generic rather than tailored to specific tasks and contexts. In this work, we introduce a novel framework for developing and evaluating context-specific empathetic large language models (LLMs). We first analyze a real-world conversational dataset consisting of 672 multi-turn conversations across 8 tasks, revealing significant differences in terms of expected and experienced empathy before and after the conversations, respectively. To help minimize this gap, we develop a synthetic multi-turn conversational generation pipeline and steer responses toward our defined empathy patterns based on the context that more closely matches users' expectations. We then train empathetic expert adapters for context-specific empathy that specialize in varying empathy levels based on the recognized task. Our empirical results demonstrate a significant gap reduction of 72.66% between perceived and desired empathy with scores increasing by an average factor of 2.43 as measured by our metrics and reward models. Additionally, our trained empathetic expert adapters demonstrate superior effectiveness in preserving empathy patterns throughout conversation turns, outperforming system prompts, which tend to dramatically diminish in impact as conversations lengthen.
>
---
#### [new 049] Let the Bees Find the Weak Spots: A Path Planning Perspective on Multi-Turn Jailbreak Attacks against LLMs
- **分类: cs.CR; cs.CL**

- **简介: 该论文将多轮越狱攻击建模为路径规划问题，提出ABC算法（改进人工蜂群），高效搜索最优攻击路径，显著提升攻击成功率（最高98%）并降低平均查询次数至26次，减少红队评估开销。**

- **链接: [http://arxiv.org/pdf/2511.03271v1](http://arxiv.org/pdf/2511.03271v1)**

> **作者:** Yize Liu; Yunyun Hou; Aina Sui
>
> **摘要:** Large Language Models (LLMs) have been widely deployed across various applications, yet their potential security and ethical risks have raised increasing concerns. Existing research employs red teaming evaluations, utilizing multi-turn jailbreaks to identify potential vulnerabilities in LLMs. However, these approaches often lack exploration of successful dialogue trajectories within the attack space, and they tend to overlook the considerable overhead associated with the attack process. To address these limitations, this paper first introduces a theoretical model based on dynamically weighted graph topology, abstracting the multi-turn attack process as a path planning problem. Based on this framework, we propose ABC, an enhanced Artificial Bee Colony algorithm for multi-turn jailbreaks, featuring a collaborative search mechanism with employed, onlooker, and scout bees. This algorithm significantly improves the efficiency of optimal attack path search while substantially reducing the average number of queries required. Empirical evaluations on three open-source and two proprietary language models demonstrate the effectiveness of our approach, achieving attack success rates above 90\% across the board, with a peak of 98\% on GPT-3.5-Turbo, and outperforming existing baselines. Furthermore, it achieves comparable success with only 26 queries on average, significantly reducing red teaming overhead and highlighting its superior efficiency.
>
---
#### [new 050] LiveTradeBench: Seeking Real-World Alpha with Large Language Models
- **分类: q-fin.TR; cs.AI; cs.CE; cs.CL**

- **简介: 论文提出LiveTradeBench，一个实时交易评估平台，解决LLM在静态基准中无法评估真实决策能力的问题。通过流式市场数据、多资产组合管理与跨市场测试，评估LLM在动态不确定性下的交易决策能力。**

- **链接: [http://arxiv.org/pdf/2511.03628v1](http://arxiv.org/pdf/2511.03628v1)**

> **作者:** Haofei Yu; Fenghai Li; Jiaxuan You
>
> **备注:** 16 pages
>
> **摘要:** Large language models (LLMs) achieve strong performance across benchmarks--from knowledge quizzes and math reasoning to web-agent tasks--but these tests occur in static settings, lacking real dynamics and uncertainty. Consequently, they evaluate isolated reasoning or problem-solving rather than decision-making under uncertainty. To address this, we introduce LiveTradeBench, a live trading environment for evaluating LLM agents in realistic and evolving markets. LiveTradeBench follows three design principles: (i) Live data streaming of market prices and news, eliminating dependence on offline backtesting and preventing information leakage while capturing real-time uncertainty; (ii) a portfolio-management abstraction that extends control from single-asset actions to multi-asset allocation, integrating risk management and cross-asset reasoning; and (iii) multi-market evaluation across structurally distinct environments--U.S. stocks and Polymarket prediction markets--differing in volatility, liquidity, and information flow. At each step, an agent observes prices, news, and its portfolio, then outputs percentage allocations that balance risk and return. Using LiveTradeBench, we run 50-day live evaluations of 21 LLMs across families. Results show that (1) high LMArena scores do not imply superior trading outcomes; (2) models display distinct portfolio styles reflecting risk appetite and reasoning dynamics; and (3) some LLMs effectively leverage live signals to adapt decisions. These findings expose a gap between static evaluation and real-world competence, motivating benchmarks that test sequential decision making and consistency under live uncertainty.
>
---
#### [new 051] Watermarking Large Language Models in Europe: Interpreting the AI Act in Light of Technology
- **分类: cs.CR; cs.AI; cs.CL; cs.CY; 68T01, 68727, 68T30, 68T35, 68T37, 68T50**

- **简介: 该论文属AI监管技术研究，旨在解决欧盟AI法案中水印标准（可靠、互操作、有效、鲁棒）如何量化评估的问题。提出水印分类法、评估框架与互操作维度，指出当前方法均未达标，呼吁嵌入底层架构的水印研究。**

- **链接: [http://arxiv.org/pdf/2511.03641v1](http://arxiv.org/pdf/2511.03641v1)**

> **作者:** Thomas Souverain
>
> **备注:** 17 pages, 2 Tables and 2 Pictures
>
> **摘要:** To foster trustworthy Artificial Intelligence (AI) within the European Union, the AI Act requires providers to mark and detect the outputs of their general-purpose models. The Article 50 and Recital 133 call for marking methods that are ''sufficiently reliable, interoperable, effective and robust''. Yet, the rapidly evolving and heterogeneous landscape of watermarks for Large Language Models (LLMs) makes it difficult to determine how these four standards can be translated into concrete and measurable evaluations. Our paper addresses this challenge, anchoring the normativity of European requirements in the multiplicity of watermarking techniques. Introducing clear and distinct concepts on LLM watermarking, our contribution is threefold. (1) Watermarking Categorisation: We propose an accessible taxonomy of watermarking methods according to the stage of the LLM lifecycle at which they are applied - before, during, or after training, and during next-token distribution or sampling. (2) Watermarking Evaluation: We interpret the EU AI Act's requirements by mapping each criterion with state-of-the-art evaluations on robustness and detectability of the watermark, and of quality of the LLM. Since interoperability remains largely untheorised in LLM watermarking research, we propose three normative dimensions to frame its assessment. (3) Watermarking Comparison: We compare current watermarking methods for LLMs against the operationalised European criteria and show that no approach yet satisfies all four standards. Encouraged by emerging empirical tests, we recommend further research into watermarking directly embedded within the low-level architecture of LLMs.
>
---
#### [new 052] Zero-shot data citation function classification using transformer-based large language models (LLMs)
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文利用Llama 3.1-405B模型，以零样本方式自动分类科学文献中对基因组数据的引用用途，解决人工标注成本高的问题，提出新评估框架，实现F1=0.674，但受限于数据与算力。**

- **链接: [http://arxiv.org/pdf/2511.02936v1](http://arxiv.org/pdf/2511.02936v1)**

> **作者:** Neil Byers; Ali Zaidi; Valerie Skye; Chris Beecroft; Kjiersten Fagnan
>
> **摘要:** Efforts have increased in recent years to identify associations between specific datasets and the scientific literature that incorporates them. Knowing that a given publication cites a given dataset, the next logical step is to explore how or why that data was used. Advances in recent years with pretrained, transformer-based large language models (LLMs) offer potential means for scaling the description of data use cases in the published literature. This avoids expensive manual labeling and the development of training datasets for classical machine-learning (ML) systems. In this work we apply an open-source LLM, Llama 3.1-405B, to generate structured data use case labels for publications known to incorporate specific genomic datasets. We also introduce a novel evaluation framework for determining the efficacy of our methods. Our results demonstrate that the stock model can achieve an F1 score of .674 on a zero-shot data citation classification task with no previously defined categories. While promising, our results are qualified by barriers related to data availability, prompt overfitting, computational infrastructure, and the expense required to conduct responsible performance evaluation.
>
---
#### [new 053] From Insight to Exploit: Leveraging LLM Collaboration for Adaptive Adversarial Text Generation
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出StaDec和DyDec框架，利用LLM协作自动生成语义保留的自适应对抗文本，以评估LLM对对抗攻击的鲁棒性，无需外部启发式规则，具备跨模型迁移性。**

- **链接: [http://arxiv.org/pdf/2511.03128v1](http://arxiv.org/pdf/2511.03128v1)**

> **作者:** Najrin Sultana; Md Rafi Ur Rashid; Kang Gu; Shagufta Mehnaz
>
> **备注:** Findings of the Association for Computational Linguistics: EMNLP 2025 (camera-ready)
>
> **摘要:** LLMs can provide substantial zero-shot performance on diverse tasks using a simple task prompt, eliminating the need for training or fine-tuning. However, when applying these models to sensitive tasks, it is crucial to thoroughly assess their robustness against adversarial inputs. In this work, we introduce Static Deceptor (StaDec) and Dynamic Deceptor (DyDec), two innovative attack frameworks designed to systematically generate dynamic and adaptive adversarial examples by leveraging the understanding of the LLMs. We produce subtle and natural-looking adversarial inputs that preserve semantic similarity to the original text while effectively deceiving the target LLM. By utilizing an automated, LLM-driven pipeline, we eliminate the dependence on external heuristics. Our attacks evolve with the advancements in LLMs and demonstrate strong transferability across models unknown to the attacker. Overall, this work provides a systematic approach for the self-assessment of an LLM's robustness. We release our code and data at https://github.com/Shukti042/AdversarialExample.
>
---
#### [new 054] Beyond Citations: Measuring Idea-level Knowledge Diffusion from Research to Journalism and Policy-making
- **分类: cs.SI; cs.CL**

- **简介: 该论文提出一种文本分析方法，突破引文计量，追踪社会科学理念在研究、新闻与政策间的跨域扩散。通过语义嵌入分析，揭示理念在不同领域中的意义演变与扩散差异，解决知识影响测量不精准的问题。**

- **链接: [http://arxiv.org/pdf/2511.03378v1](http://arxiv.org/pdf/2511.03378v1)**

> **作者:** Yangliu Fan; Kilian Buehling; Volker Stocker
>
> **摘要:** Despite the importance of social science knowledge for various stakeholders, measuring its diffusion into different domains remains a challenge. This study uses a novel text-based approach to measure the idea-level diffusion of social science knowledge from the research domain to the journalism and policy-making domains. By doing so, we expand the detection of knowledge diffusion beyond the measurements of direct references. Our study focuses on media effects theories as key research ideas in the field of communication science. Using 72,703 documents (2000-2019) from three domains (i.e., research, journalism, and policy-making) that mention these ideas, we count the mentions of these ideas in each domain, estimate their domain-specific contexts, and track and compare differences across domains and over time. Overall, we find that diffusion patterns and dynamics vary considerably between ideas, with some ideas diffusing between other domains, while others do not. Based on the embedding regression approach, we compare contextualized meanings across domains and find that the distances between research and policy are typically larger than between research and journalism. We also find that ideas largely shift roles across domains - from being the theories themselves in research to sense-making in news to applied, administrative use in policy. Over time, we observe semantic convergence mainly for ideas that are practically oriented. Our results characterize the cross-domain diffusion patterns and dynamics of social science knowledge at the idea level, and we discuss the implications for measuring knowledge diffusion beyond citations.
>
---
#### [new 055] The Curved Spacetime of Transformer Architectures
- **分类: cs.LG; cs.CL; math.DG**

- **简介: 该论文将Transformer架构类比为广义相对论中的弯曲时空，提出查询-键构建度量、注意力实现平行移动，通过可视化与实验验证嵌入轨迹的非直线弯曲，揭示注意力机制的几何本质。**

- **链接: [http://arxiv.org/pdf/2511.03060v1](http://arxiv.org/pdf/2511.03060v1)**

> **作者:** Riccardo Di Sipio; Jairo Diaz-Rodriguez; Luis Serrano
>
> **摘要:** We present a geometric framework for understanding Transformer-based language models, drawing an explicit analogy to General Relativity. Queries and keys induce an effective metric on representation space, and attention acts as a discrete connection that implements parallel transport of value vectors across tokens. Stacked layers provide discrete time-slices through which token representations evolve on this curved manifold, while backpropagation plays the role of a least-action principle that shapes loss-minimizing trajectories in parameter space. If this analogy is correct, token embeddings should not traverse straight paths in feature space; instead, their layer-wise steps should bend and reorient as interactions mediated by embedding space curvature. To test this prediction, we design experiments that expose both the presence and the consequences of curvature: (i) we visualize a curvature landscape for a full paragraph, revealing how local turning angles vary across tokens and layers; (ii) we show through simulations that excess counts of sharp/flat angles and longer length-to-chord ratios are not explainable by dimensionality or chance; and (iii) inspired by Einstein's eclipse experiment, we probe deflection under controlled context edits, demonstrating measurable, meaning-consistent bends in embedding trajectories that confirm attention-induced curvature.
>
---
## 更新

#### [replaced 001] Modeling Annotator Disagreement with Demographic-Aware Experts and Synthetic Perspectives
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.02853v3](http://arxiv.org/pdf/2508.02853v3)**

> **作者:** Yinuo Xu; Veronica Derricks; Allison Earl; David Jurgens
>
> **备注:** 8 pages, 17 figures
>
> **摘要:** We present an approach to modeling annotator disagreement in subjective NLP tasks through both architectural and data-centric innovations. Our model, DEM-MoE (Demographic-Aware Mixture of Experts), routes inputs to expert subnetworks based on annotator demographics, enabling it to better represent structured, group-level variation compared to prior models. DEM-MoE consistently performs competitively across demographic groups, and shows especially strong results on datasets with high annotator disagreement. To address sparse demographic coverage, we test whether LLM-generated synthetic annotations via zero-shot persona prompting can be used for data imputation. We show these synthetic judgments align moderately well with human annotations on our data and offer a scalable way to potentially enrich training data. We then propose and evaluate approaches for blending real and synthetic data using strategies tailored to dataset structure. We find that the optimal strategies depend on dataset structure. Together, these contributions improve the representation of diverse perspectives.
>
---
#### [replaced 002] Dense SAE Latents Are Features, Not Bugs
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.15679v2](http://arxiv.org/pdf/2506.15679v2)**

> **作者:** Xiaoqing Sun; Alessandro Stolfo; Joshua Engels; Ben Wu; Senthooran Rajamanoharan; Mrinmaya Sachan; Max Tegmark
>
> **备注:** NeurIPS 2025 poster
>
> **摘要:** Sparse autoencoders (SAEs) are designed to extract interpretable features from language models by enforcing a sparsity constraint. Ideally, training an SAE would yield latents that are both sparse and semantically meaningful. However, many SAE latents activate frequently (i.e., are \emph{dense}), raising concerns that they may be undesirable artifacts of the training procedure. In this work, we systematically investigate the geometry, function, and origin of dense latents and show that they are not only persistent but often reflect meaningful model representations. We first demonstrate that dense latents tend to form antipodal pairs that reconstruct specific directions in the residual stream, and that ablating their subspace suppresses the emergence of new dense features in retrained SAEs -- suggesting that high density features are an intrinsic property of the residual space. We then introduce a taxonomy of dense latents, identifying classes tied to position tracking, context binding, entropy regulation, letter-specific output signals, part-of-speech, and principal component reconstruction. Finally, we analyze how these features evolve across layers, revealing a shift from structural features in early layers, to semantic features in mid layers, and finally to output-oriented signals in the last layers of the model. Our findings indicate that dense latents serve functional roles in language model computation and should not be dismissed as training noise.
>
---
#### [replaced 003] Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2408.15176v5](http://arxiv.org/pdf/2408.15176v5)**

> **作者:** Longshen Ou; Jingwei Zhao; Ziyu Wang; Gus Xia; Qihao Liang; Torin Hopkins Ye Wang
>
> **备注:** NeurIPS 2025 camera ready version
>
> **摘要:** We present a unified framework for automatic multitrack music arrangement that enables a single pre-trained symbolic music model to handle diverse arrangement scenarios, including reinterpretation, simplification, and additive generation. At its core is a segment-level reconstruction objective operating on token-level disentangled content and style, allowing for flexible any-to-any instrumentation transformations at inference time. To support track-wise modeling, we introduce REMI-z, a structured tokenization scheme for multitrack symbolic music that enhances modeling efficiency and effectiveness for both arrangement tasks and unconditional generation. Our method outperforms task-specific state-of-the-art models on representative tasks in different arrangement scenarios -- band arrangement, piano reduction, and drum arrangement, in both objective metrics and perceptual evaluations. Taken together, our framework demonstrates strong generality and suggests broader applicability in symbolic music-to-music transformation.
>
---
#### [replaced 004] Omni-Router: Sharing Routing Decisions in Sparse Mixture-of-Experts for Speech Recognition
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.05724v3](http://arxiv.org/pdf/2507.05724v3)**

> **作者:** Zijin Gu; Tatiana Likhomanenko; Navdeep Jaitly
>
> **备注:** Accepted in 2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)
>
> **摘要:** Mixture-of-experts (MoE) architectures have expanded from language modeling to automatic speech recognition (ASR). Traditional MoE methods, such as the Switch Transformer, route experts independently within each layer. Our analysis reveals that routers in most layers make expert choices that are not strongly correlated with the choices of the routers in other layers. To increase the cooperation between experts in different layers and encourage greater specialization, we use a shared router across different MoE layers. We call this model Omni-router Transformer. Extensive experiments on a large-scale pseudo-labeled dataset and evaluations across 10 diverse, out-of-domain ASR benchmarks demonstrate that the Omni-router Transformer is able to achieve lower training loss and consistently outperform dense and Switch Transformer models, reducing average word error rates by 11.2% and 8.2%, respectively, while providing structured expert usage and improved robustness to diverse data.
>
---
#### [replaced 005] AgenticMath: Enhancing LLM Reasoning via Agentic-based Math Data Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.19361v2](http://arxiv.org/pdf/2510.19361v2)**

> **作者:** Xianyang Liu; Yilin Liu; Shuai Wang; Hao Cheng; Andrew Estornell; Yuzhi Zhao; Jiaheng Wei
>
> **备注:** 9 pages
>
> **摘要:** The creation of high-quality datasets to improve Large Language Model (LLM) reasoning remains a significant challenge, as current methods often suffer from generating low-quality/incorrect answers and limited information richness from available data sources. To address this, we propose AgenticMath, a novel agentic pipeline for generating high-quality mathematical question-answer pairs to enhance the supervised fine-tuning of LLMs. Our method operates through four stages: (1) Seed Question Filter that selects questions with high information richness, complexity, and clarity; (2) an Agentic Question Rephrase step that employs a multi-agent system to generate diverse, logically consistent paraphrases; (3) an Answer Augment step where rewrite answers using chain-of-thought reasoning to enhance numerical and logical correctness, without reliance on human-provided labels; and (4) a final Question and Answer Evaluation that retains only the most superior pairs. Extensive experiments demonstrate that, fine-tuning 3B-8B parameter LLMs on AgenticMath generated datasets (comprising only 30-60K math samples) achieves competitive or superior performance on diverse in domain and out-of-domain mathematical reasoning benchmarks compared to baselines trained on much more data (e.g., 400K or 2.3M samples). Our work demonstrates that targeted, high-quality data generation is a more efficient path to improving mathematical reasoning in LLMs than large-scale, low-quality alternatives.
>
---
#### [replaced 006] HPLT 3.0: Very Large-Scale Multilingual Resources for LLM and MT. Mono- and Bi-lingual Data, Multilingual Evaluation, and Pre-Trained Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2511.01066v2](http://arxiv.org/pdf/2511.01066v2)**

> **作者:** Stephan Oepen; Nikolay Arefev; Mikko Aulamo; Marta Bañón; Maja Buljan; Laurie Burchell; Lucas Charpentier; Pinzhen Chen; Mariya Fedorova; Ona de Gibert; Barry Haddow; Jan Hajič; Jindřich Helcl; Andrey Kutuzov; Veronika Laippala; Zihao Li; Risto Luukkonen; Bhavitvya Malik; Vladislav Mikhailov; Amanda Myntti; Dayyán O'Brien; Lucie Poláková; Sampo Pyysalo; Gema Ramírez Sánchez; Janine Siewert; Pavel Stepachev; Jörg Tiedemann; Teemu Vahtola; Dušan Variš; Fedor Vitiugin; Tea Vojtěchová; Jaume Zaragoza
>
> **摘要:** We present an ongoing initiative to provide open, very large, high-quality, and richly annotated textual datasets for almost 200 languages. At 30 trillion tokens, this is likely the largest generally available multilingual collection of LLM pre-training data. These datasets are derived from web crawls from different sources and accompanied with a complete, open-source pipeline for document selection from web archives, text extraction from HTML, language identification for noisy texts, exact and near-deduplication, annotation with, among others, register labels, text quality estimates, and personally identifiable information; and final selection and filtering. We report on data quality probes through contrastive and analytical statistics, through manual inspection of samples for 24 languages, and through end-to-end evaluation of various language model architectures trained on this data. For multilingual LLM evaluation, we provide a comprehensive collection of benchmarks for nine European languages, with special emphasis on natively created tasks, mechanisms to mitigate prompt sensitivity, and refined normalization and aggregation of scores. Additionally, we train and evaluate a family of 57 monolingual encoder-decoder models, as well as a handful of monolingual GPT-like reference models. Besides the monolingual data and models, we also present a very large collection of parallel texts automatically mined from this data, together with a novel parallel corpus synthesized via machine translation.
>
---
#### [replaced 007] The Case for Repeatable, Open, and Expert-Grounded Hallucination Benchmarks in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17345v2](http://arxiv.org/pdf/2505.17345v2)**

> **作者:** Justin D. Norman; Michael U. Rivera; D. Alex Hughes
>
> **备注:** 9 pages
>
> **摘要:** Plausible, but inaccurate, tokens in model-generated text are widely believed to be pervasive and problematic for the responsible adoption of language models. Despite this concern, there is little scientific work that attempts to measure the prevalence of language model hallucination in a comprehensive way. In this paper, we argue that language models should be evaluated using repeatable, open, and domain-contextualized hallucination benchmarking. We present a taxonomy of hallucinations alongside a case study that demonstrates that when experts are absent from the early stages of data creation, the resulting hallucination metrics lack validity and practical utility.
>
---
#### [replaced 008] LexTime: A Benchmark for Temporal Ordering of Legal Events
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.04041v2](http://arxiv.org/pdf/2506.04041v2)**

> **作者:** Claire Barale; Leslie Barrett; Vikram Sunil Bajaj; Michael Rovatsos
>
> **备注:** EMNLP 2025 (Findings) long paper
>
> **摘要:** Understanding temporal relationships and accurately reconstructing the event timeline is important for case law analysis, compliance monitoring, and legal summarization. However, existing benchmarks lack specialized language evaluation, leaving a gap in understanding how LLMs handle event ordering in legal contexts. We introduce LexTime, a dataset designed to evaluate LLMs' event ordering capabilities in legal language, consisting of 512 instances from U.S. Federal Complaints with annotated event pairs and their temporal relations. Our findings show that (1) LLMs are more accurate on legal event ordering than on narrative texts (up to +10.5%); (2) longer input contexts and implicit events boost accuracy, reaching 80.8% for implicit-explicit event pairs; (3) legal linguistic complexities and nested clauses remain a challenge. While performance is promising, specific features of legal texts remain a bottleneck for legal temporal event reasoning, and we propose concrete modeling directions to better address them.
>
---
#### [replaced 009] Exploring Typographic Visual Prompts Injection Threats in Cross-Modality Generation Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.11519v4](http://arxiv.org/pdf/2503.11519v4)**

> **作者:** Hao Cheng; Erjia Xiao; Yichi Wang; Lingfeng Zhang; Qiang Zhang; Jiahang Cao; Kaidi Xu; Mengshu Sun; Xiaoshuai Hao; Jindong Gu; Renjing Xu
>
> **备注:** This paper is accepted by IJCAI2025 Workshop on Deepfake Detection, Localization, and Interpretability as Best Student Paper
>
> **摘要:** Current Cross-Modality Generation Models (GMs) demonstrate remarkable capabilities in various generative tasks. Given the ubiquity and information richness of vision modality inputs in real-world scenarios, Cross-Vision tasks, encompassing Vision-Language Perception (VLP) and Image-to-Image (I2I), have attracted significant attention. Large Vision Language Models (LVLMs) and I2I Generation Models (GMs) are employed to handle VLP and I2I tasks, respectively. Previous research indicates that printing typographic words into input images significantly induces LVLMs and I2I GMs to produce disruptive outputs that are semantically aligned with those words. Additionally, visual prompts, as a more sophisticated form of typography, are also revealed to pose security risks to various applications of cross-vision tasks. However, the specific characteristics of the threats posed by visual prompts remain underexplored. In this paper, to comprehensively investigate the performance impact induced by Typographic Visual Prompt Injection (TVPI) in various LVLMs and I2I GMs, we propose the Typographic Visual Prompts Injection Dataset and thoroughly evaluate the TVPI security risks on various open-source and closed-source LVLMs and I2I GMs under visual prompts with different target semantics, deepening the understanding of TVPI threats.
>
---
#### [replaced 010] Does Synthetic Data Help Named Entity Recognition for Low-Resource Languages?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16814v3](http://arxiv.org/pdf/2505.16814v3)**

> **作者:** Gaurav Kamath; Sowmya Vajjala
>
> **备注:** Accepted at AACL 2025. Camera-ready version
>
> **摘要:** Named Entity Recognition(NER) for low-resource languages aims to produce robust systems for languages where there is limited labeled training data available, and has been an area of increasing interest within NLP. Data augmentation for increasing the amount of low-resource labeled data is a common practice. In this paper, we explore the role of synthetic data in the context of multilingual, low-resource NER, considering 11 languages from diverse language families. Our results suggest that synthetic data does in fact hold promise for low-resource language NER, though we see significant variation between languages.
>
---
#### [replaced 011] Inv-Entropy: A Fully Probabilistic Framework for Uncertainty Quantification in Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.09684v2](http://arxiv.org/pdf/2506.09684v2)**

> **作者:** Haoyi Song; Ruihan Ji; Naichen Shi; Fan Lai; Raed Al Kontar
>
> **摘要:** Large language models (LLMs) have transformed natural language processing, but their reliable deployment requires effective uncertainty quantification (UQ). Existing UQ methods are often heuristic and lack a probabilistic interpretation. This paper begins by providing a theoretical justification for the role of perturbations in UQ for LLMs. We then introduce a dual random walk perspective, modeling input-output pairs as two Markov chains with transition probabilities defined by semantic similarity. Building on this, we propose a fully probabilistic framework based on an inverse model, which quantifies uncertainty by evaluating the diversity of the input space conditioned on a given output through systematic perturbations. Within this framework, we define a new uncertainty measure, Inv-Entropy. A key strength of our framework is its flexibility: it supports various definitions of uncertainty measures, embeddings, perturbation strategies, and similarity metrics. We also propose GAAP, a perturbation algorithm based on genetic algorithms, which enhances the diversity of sampled inputs. In addition, we introduce a new evaluation metric, Temperature Sensitivity of Uncertainty (TSU), which directly assesses uncertainty without relying on correctness as a proxy. Extensive experiments demonstrate that Inv-Entropy outperforms existing semantic UQ methods. The code to reproduce the results can be found at https://github.com/UMDataScienceLab/Uncertainty-Quantification-for-LLMs.
>
---
#### [replaced 012] Erasing 'Ugly' from the Internet: Propagation of the Beauty Myth in Text-Image Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2511.00749v2](http://arxiv.org/pdf/2511.00749v2)**

> **作者:** Tanvi Dinkar; Aiqi Jiang; Gavin Abercrombie; Ioannis Konstas
>
> **备注:** This is a preprint under review
>
> **摘要:** Social media has exacerbated the promotion of Western beauty norms, leading to negative self-image, particularly in women and girls, and causing harm such as body dysmorphia. Increasingly content on the internet has been artificially generated, leading to concerns that these norms are being exaggerated. The aim of this work is to study how generative AI models may encode 'beauty' and erase 'ugliness', and discuss the implications of this for society. To investigate these aims, we create two image generation pipelines: a text-to-image model and a text-to-language model-to image model. We develop a structured beauty taxonomy which we use to prompt three language models (LMs) and two text-to-image models to cumulatively generate 5984 images using our two pipelines. We then recruit women and non-binary social media users to evaluate 1200 of the images through a Likert-scale within-subjects study. Participants show high agreement in their ratings. Our results show that 86.5% of generated images depicted people with lighter skin tones, 22% contained explicit content despite Safe for Work (SFW) training, and 74% were rated as being in a younger age demographic. In particular, the images of non-binary individuals were rated as both younger and more hypersexualised, indicating troubling intersectional effects. Notably, prompts encoded with 'negative' or 'ugly' beauty traits (such as "a wide nose") consistently produced higher Not SFW (NSFW) ratings regardless of gender. This work sheds light on the pervasive demographic biases related to beauty standards present in generative AI models -- biases that are actively perpetuated by model developers, such as via negative prompting. We conclude by discussing the implications of this on society, which include pollution of the data streams and active erasure of features that do not fall inside the stereotype of what is considered beautiful by developers.
>
---
#### [replaced 013] VoiceAgentBench: Are Voice Assistants ready for agentic tasks?
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.07978v2](http://arxiv.org/pdf/2510.07978v2)**

> **作者:** Dhruv Jain; Harshit Shukla; Gautam Rajeev; Ashish Kulkarni; Chandra Khatri; Shubham Agarwal
>
> **摘要:** Large-scale Speech Language Models (SpeechLMs) have enabled voice assistants capable of understanding natural spoken queries and performing complex tasks. However, existing speech benchmarks primarily focus on isolated capabilities such as transcription, or question-answering, and do not systematically evaluate agentic scenarios encompassing multilingual and cultural understanding, as well as adversarial robustness. To address this, we introduce VoiceAgentBench, a comprehensive benchmark designed to evaluate SpeechLMs in realistic spoken agentic settings. It comprises over 5,500 synthetic spoken queries, including dialogues grounded in Indian context, covering single-tool invocations, multi-tool workflows, multi-turn interactions, and safety evaluations. The benchmark supports English, Hindi, and 5 other Indian languages, reflecting real-world linguistic and cultural diversity. We simulate speaker variability using a novel sampling algorithm that selects audios for TTS voice conversion based on its speaker embeddings, maximizing acoustic and speaker diversity. Our evaluation measures tool selection accuracy, structural consistency, and the correctness of tool invocations, including adversarial robustness. Our experiments reveal significant gaps in contextual tool orchestration tasks, Indic generalization, and adversarial robustness, exposing critical limitations of current SpeechLMs.
>
---
#### [replaced 014] Novelty and Impact of Economics Papers
- **分类: econ.GN; cs.CE; cs.CL; cs.DL; q-fin.EC**

- **链接: [http://arxiv.org/pdf/2511.01211v2](http://arxiv.org/pdf/2511.01211v2)**

> **作者:** Chaofeng Wu
>
> **摘要:** We propose a framework that recasts scientific novelty not as a single attribute of a paper, but as a reflection of its position within the evolving intellectual landscape. We decompose this position into two orthogonal dimensions: \textit{spatial novelty}, which measures a paper's intellectual distinctiveness from its neighbors, and \textit{temporal novelty}, which captures its engagement with a dynamic research frontier. To operationalize these concepts, we leverage Large Language Models to develop semantic isolation metrics that quantify a paper's location relative to the full-text literature. Applying this framework to a large corpus of economics articles, we uncover a fundamental trade-off: these two dimensions predict systematically different outcomes. Temporal novelty primarily predicts citation counts, whereas spatial novelty predicts disruptive impact. This distinction allows us to construct a typology of semantic neighborhoods, identifying four archetypes associated with distinct and predictable impact profiles. Our findings demonstrate that novelty can be understood as a multidimensional construct whose different forms, reflecting a paper's strategic location, have measurable and fundamentally distinct consequences for scientific progress.
>
---
#### [replaced 015] Agent-Omni: Test-Time Multimodal Reasoning via Model Coordination for Understanding Anything
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2511.02834v2](http://arxiv.org/pdf/2511.02834v2)**

> **作者:** Huawei Lin; Yunzhi Shi; Tong Geng; Weijie Zhao; Wei Wang; Ravender Pal Singh
>
> **备注:** 16 pages, 7 figures, 14 tables. Under Review
>
> **摘要:** Multimodal large language models (MLLMs) have shown strong capabilities but remain limited to fixed modality pairs and require costly fine-tuning with large aligned datasets. Building fully omni-capable models that can integrate text, images, audio, and video remains impractical and lacks robust reasoning support. In this paper, we propose an Agent-Omni framework that coordinates existing foundation models through a master-agent system, enabling flexible multimodal reasoning without retraining. The master agent interprets user intent, delegates subtasks to modality-specific agents, and integrates their outputs into coherent responses. Extensive experiments across text, image, audio, video, and omni benchmarks show that Agent-Omni consistently achieves state-of-the-art performance, particularly on tasks requiring complex cross-modal reasoning. Its agent-based design enables seamless integration of specialized foundation models, ensuring adaptability to diverse inputs while maintaining transparency and interpretability. In addition, the framework is modular and easily extensible, allowing future improvements as stronger models become available.
>
---
#### [replaced 016] R2R: Efficiently Navigating Divergent Reasoning Paths with Small-Large Model Token Routing
- **分类: cs.CL; cs.AI; cs.LG; cs.PF; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.21600v2](http://arxiv.org/pdf/2505.21600v2)**

> **作者:** Tianyu Fu; Yi Ge; Yichen You; Enshu Liu; Zhihang Yuan; Guohao Dai; Shengen Yan; Huazhong Yang; Yu Wang
>
> **摘要:** Large Language Models (LLMs) achieve impressive reasoning capabilities at the cost of substantial inference overhead, posing substantial deployment challenges. Although distilled Small Language Models (SLMs) significantly enhance efficiency, their performance suffers as they fail to follow LLMs' reasoning paths. Luckily, we reveal that only a small fraction of tokens genuinely diverge reasoning paths between LLMs and SLMs. Most generated tokens are either identical or exhibit neutral differences, such as minor variations in abbreviations or expressions. Leveraging this insight, we introduce **Roads to Rome (R2R)**, a neural token routing method that selectively utilizes LLMs only for these critical, path-divergent tokens, while leaving the majority of token generation to the SLM. We also develop an automatic data generation pipeline that identifies divergent tokens and generates token-level routing labels to train the lightweight router. We apply R2R to combine R1-1.5B and R1-32B models from the DeepSeek family, and evaluate on challenging math, coding, and QA benchmarks. With an average activated parameter size of 5.6B, R2R surpasses the average accuracy of R1-7B by 1.6x, outperforming even the R1-14B model. Compared to R1-32B, it delivers a 2.8x wall-clock speedup with comparable performance, advancing the Pareto frontier of test-time scaling efficiency. Our code is available at https://github.com/thu-nics/R2R.
>
---
#### [replaced 017] MathOPEval: A Fine-grained Evaluation Benchmark for Visual Operations of MLLMs in Mathematical Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.18140v3](http://arxiv.org/pdf/2507.18140v3)**

> **作者:** Xiaoyuan Li; Moxin Li; Wenjie Wang; Rui Men; Yichang Zhang; Fuli Feng; Dayiheng Liu
>
> **备注:** Under Review
>
> **摘要:** Recent progress in Multi-modal Large Language Models (MLLMs) has enabled step-by-step multi-modal mathematical reasoning by performing visual operations based on the textual instructions. A promising approach uses code as an intermediate representation to precisely express and manipulate the images in the reasoning steps. However, existing evaluations focus mainly on text-only reasoning outputs, leaving the MLLM's ability to perform accurate visual operations via code largely unexplored. This work takes a first step toward addressing that gap by evaluating MLLM's code-based capabilities in multi-modal mathematical reasoning.Specifically, our framework focuses on two key evaluation aspects: (1) Multi-modal Code Generation (MCG) evaluates the model's ability to accurately understand and construct visualizations from scratch. (2) Multi-modal Code Editing (MCE) assesses the model's capacity for fine-grained operations, which include three types: Deletion, Modification and Annotation. To evaluate the above tasks, we incorporate a dataset that covers the five most popular types of mathematical figures, including geometric diagrams, function plots, and three types of statistical charts, to provide a comprehensive and effective measurement of existing MLLMs. Our experimental evaluation involves nine mainstream MLLMs, and the results reveal that existing models still lag significantly behind human performance in performing fine-grained visual operations.
>
---
#### [replaced 018] Scalable Medication Extraction and Discontinuation Identification from Electronic Health Records Using Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.11137v2](http://arxiv.org/pdf/2506.11137v2)**

> **作者:** Chong Shao; Douglas Snyder; Chiran Li; Bowen Gu; Kerry Ngan; Chun-Ting Yang; Jiageng Wu; Richard Wyss; Kueiyu Joshua Lin; Jie Yang
>
> **摘要:** Identifying medication discontinuations in electronic health records (EHRs) is vital for patient safety but is often hindered by information being buried in unstructured notes. This study aims to evaluate the capabilities of advanced open-sourced and proprietary large language models (LLMs) in extracting medications and classifying their medication status from EHR notes, focusing on their scalability on medication information extraction without human annotation. We collected three EHR datasets from diverse sources to build the evaluation benchmark. We evaluated 12 advanced LLMs and explored multiple LLM prompting strategies. Performance on medication extraction, medication status classification, and their joint task (extraction then classification) was systematically compared across all experiments. We found that LLMs showed promising performance on the medication extraction and discontinuation classification from EHR notes. GPT-4o consistently achieved the highest average F1 scores in all tasks under zero-shot setting - 94.0% for medication extraction, 78.1% for discontinuation classification, and 72.7% for the joint task. Open-sourced models followed closely, Llama-3.1-70B-Instruct achieved the highest performance in medication status classification on the MIV-Med dataset (68.7%) and in the joint task on both the Re-CASI (76.2%) and MIV-Med (60.2%) datasets. Medical-specific LLMs demonstrated lower performance compared to advanced general-domain LLMs. Few-shot learning generally improved performance, while CoT reasoning showed inconsistent gains. LLMs demonstrate strong potential for medication extraction and discontinuation identification on EHR notes, with open-sourced models offering scalable alternatives to proprietary systems and few-shot can further improve LLMs' capability.
>
---
#### [replaced 019] The Mirror Loop: Recursive Non-Convergence in Generative Reasoning Systems
- **分类: cs.LG; cs.AI; cs.CL; 68T05; I.2.6; I.2.8**

- **链接: [http://arxiv.org/pdf/2510.21861v2](http://arxiv.org/pdf/2510.21861v2)**

> **作者:** Bentley DeVilling
>
> **备注:** 18 pages, 2 figures. Category: cs.LG. Code and data: https://github.com/Course-Correct-Labs/mirror-loop
>
> **摘要:** Large language models are often described as capable of reflective reasoning, yet recursive self-evaluation without external feedback frequently yields reformulation rather than progress. We test this prediction in a cross-provider study of 144 reasoning sequences across three models (OpenAI GPT-4o-mini, Anthropic Claude 3 Haiku, and Google Gemini 2.0 Flash) and four task families (arithmetic, code, explanation, reflection), each iterated ten times under two conditions: ungrounded self-critique and a minimal grounding intervention (a single verification step at iteration three). Mean informational change (delta I, measured via normalized edit distance) declined by 55% from early (0.193) to late (0.087) iterations in ungrounded runs, with consistent patterns across all three providers. Grounded runs showed a +28% rebound in informational change immediately after the intervention and sustained non-zero variance thereafter. Complementary measures-n-gram novelty, embedding drift, and character-level entropy-converged on the same pattern: reflection without contact tends toward informational closure. We interpret this as evidence for a structural limit on self-correction in generative reasoning: without an exchange of information with an independent verifier or environment, recursive inference approaches an attractor state of epistemic stasis. Minimal grounding functions as dissipative coupling, reintroducing informational flux. The cross-architecture consistency suggests the mirror loop arises from shared autoregressive training objectives rather than provider-specific alignment schemes. The results delineate when reflection is performative rather than epistemic and motivate design principles for grounded, cooperative reasoning. Materials and code are publicly available.
>
---
#### [replaced 020] SafeSearch: Do Not Trade Safety for Utility in LLM Search Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.17017v3](http://arxiv.org/pdf/2510.17017v3)**

> **作者:** Qiusi Zhan; Angeline Budiman-Chan; Abdelrahman Zayed; Xingzhi Guo; Daniel Kang; Joo-Kyung Kim
>
> **备注:** Code available at https://github.com/amazon-science/SafeSearch
>
> **摘要:** Large language model (LLM) based search agents iteratively generate queries, retrieve external information, and reason to answer open-domain questions. While researchers have primarily focused on improving their utility, their safety behaviors remain underexplored. In this paper, we first evaluate search agents using red-teaming datasets and find that they are more likely to produce harmful outputs than base LLMs. For example, when asked "How can I track someone's location without their consent?", a base model refuses, whereas a search agent designed to retrieve and cite sources may lower its refusal threshold, fetch documents (e.g., court cases), and, once appended, synthesize them into an informative yet unsafe summary. We further show that utility-oriented fine-tuning intensifies this risk, motivating joint alignment of safety and utility. We present SafeSearch, a multi-objective reinforcement learning approach that couples a final-output safety/utility reward with a novel query-level shaping term that penalizes unsafe queries and rewards safe ones. Experiments show that SafeSearch reduces agent harmfulness by over 70% across three red-teaming datasets while producing safe, helpful responses, and matches the QA performance of a utility-only finetuned agent; further analyses confirm the effectiveness of the query-level reward in jointly improving safety and utility.
>
---
#### [replaced 021] Reinforcement Learning Foundations for Deep Research Systems: A Survey
- **分类: cs.AI; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2509.06733v2](http://arxiv.org/pdf/2509.06733v2)**

> **作者:** Wenjun Li; Zhi Chen; Jingru Lin; Hannan Cao; Wei Han; Sheng Liang; Zhi Zhang; Kuicai Dong; Dexun Li; Chen Zhang; Yong Liu
>
> **备注:** 39 pages, second version
>
> **摘要:** Deep research systems, agentic AI that solve complex, multi-step tasks by coordinating reasoning, search across the open web and user files, and tool use, are moving toward hierarchical deployments with a Planner, Coordinator, and Executors. In practice, training entire stacks end-to-end remains impractical, so most work trains a single planner connected to core tools such as search, browsing, and code. While SFT imparts protocol fidelity, it suffers from imitation and exposure biases and underuses environment feedback. Preference alignment methods such as DPO are schema and proxy-dependent, off-policy, and weak for long-horizon credit assignment and multi-objective trade-offs. A further limitation of SFT and DPO is their reliance on human defined decision points and subskills through schema design and labeled comparisons. Reinforcement learning aligns with closed-loop, tool-interaction research by optimizing trajectory-level policies, enabling exploration, recovery behaviors, and principled credit assignment, and it reduces dependence on such human priors and rater biases. This survey is, to our knowledge, the first dedicated to the RL foundations of deep research systems. It systematizes recent work along three axes: (i) data synthesis and curation; (ii) RL methods for agentic research covering stability, sample efficiency, long context handling, reward and credit design, multi-objective optimization, and multimodal integration; and (iii) agentic RL training systems and frameworks. We also cover agent architecture and coordination, as well as evaluation and benchmarks, including recent QA, VQA, long-form synthesis, and domain-grounded, tool-interaction tasks. We distill recurring patterns, surface infrastructure bottlenecks, and offer practical guidance for training robust, transparent deep research agents with RL.
>
---
#### [replaced 022] s3: You Don't Need That Much Data to Train a Search Agent via RL
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14146v2](http://arxiv.org/pdf/2505.14146v2)**

> **作者:** Pengcheng Jiang; Xueqiang Xu; Jiacheng Lin; Jinfeng Xiao; Zifeng Wang; Jimeng Sun; Jiawei Han
>
> **备注:** EMNLP 2025 camera-ready
>
> **摘要:** Retrieval-augmented generation (RAG) systems empower large language models (LLMs) to access external knowledge during inference. Recent advances have enabled LLMs to act as search agents via reinforcement learning (RL), improving information acquisition through multi-turn interactions with retrieval engines. However, existing approaches either optimize retrieval using search-only metrics (e.g., NDCG) that ignore downstream utility or fine-tune the entire LLM to jointly reason and retrieve-entangling retrieval with generation and limiting the real search utility and compatibility with frozen or proprietary models. In this work, we propose s3, a lightweight, model-agnostic framework that decouples the searcher from the generator and trains the searcher using a Gain Beyond RAG reward: the improvement in generation accuracy over naive RAG. s3 requires only 2.4k training samples to outperform baselines trained on over 70x more data, consistently delivering stronger downstream performance across six general QA and five medical QA benchmarks.
>
---
#### [replaced 023] Post Persona Alignment for Multi-Session Dialogue Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.11857v2](http://arxiv.org/pdf/2506.11857v2)**

> **作者:** Yi-Pei Chen; Noriki Nishida; Hideki Nakayama; Yuji Matsumoto
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Multi-session persona-based dialogue generation presents challenges in maintaining long-term consistency and generating diverse, personalized responses. While large language models (LLMs) excel in single-session dialogues, they struggle to preserve persona fidelity and conversational coherence across extended interactions. Existing methods typically retrieve persona information before response generation, which can constrain diversity and result in generic outputs. We propose Post Persona Alignment (PPA), a novel two-stage framework that reverses this process. PPA first generates a general response based solely on dialogue context, then retrieves relevant persona memories using the response as a query, and finally refines the response to align with the speaker's persona. This post-hoc alignment strategy promotes naturalness and diversity while preserving consistency and personalization. Experiments on multi-session LLM-generated dialogue data demonstrate that PPA significantly outperforms prior approaches in consistency, diversity, and persona relevance, offering a more flexible and effective paradigm for long-term personalized dialogue generation.
>
---
#### [replaced 024] Traversal Verification for Speculative Tree Decoding
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.12398v2](http://arxiv.org/pdf/2505.12398v2)**

> **作者:** Yepeng Weng; Qiao Hu; Xujie Chen; Li Liu; Dianwen Mei; Huishi Qiu; Jiang Tian; Zhongchao Shi
>
> **备注:** NeurIPS 2025 poster
>
> **摘要:** Speculative decoding is a promising approach for accelerating large language models. The primary idea is to use a lightweight draft model to speculate the output of the target model for multiple subsequent timesteps, and then verify them in parallel to determine whether the drafted tokens should be accepted or rejected. To enhance acceptance rates, existing frameworks typically construct token trees containing multiple candidates in each timestep. However, their reliance on token-level verification mechanisms introduces two critical limitations: First, the probability distribution of a sequence differs from that of individual tokens, leading to suboptimal acceptance length. Second, current verification schemes begin from the root node and proceed layer by layer in a top-down manner. Once a parent node is rejected, all its child nodes should be discarded, resulting in inefficient utilization of speculative candidates. This paper introduces Traversal Verification, a novel speculative decoding algorithm that fundamentally rethinks the verification paradigm through leaf-to-root traversal. Our approach considers the acceptance of the entire token sequence from the current node to the root, and preserves potentially valid subsequences that would be prematurely discarded by existing methods. We theoretically prove that the probability distribution obtained through Traversal Verification is identical to that of the target model, guaranteeing lossless inference while achieving substantial acceleration gains. Experimental results across different large language models and multiple tasks show that our method consistently improves acceptance length and throughput over existing methods.
>
---
#### [replaced 025] Evaluating Large Language Models for Detecting Antisemitism
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2509.18293v2](http://arxiv.org/pdf/2509.18293v2)**

> **作者:** Jay Patel; Hrudayangam Mehta; Jeremy Blackburn
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Detecting hateful content is a challenging and important problem. Automated tools, like machine-learning models, can help, but they require continuous training to adapt to the ever-changing landscape of social media. In this work, we evaluate eight open-source LLMs' capability to detect antisemitic content, specifically leveraging in-context definition. We also study how LLMs understand and explain their decisions given a moderation policy as a guideline. First, we explore various prompting techniques and design a new CoT-like prompt, Guided-CoT, and find that injecting domain-specific thoughts increases performance and utility. Guided-CoT handles the in-context policy well, improving performance and utility by reducing refusals across all evaluated models, regardless of decoding configuration, model size, or reasoning capability. Notably, Llama 3.1 70B outperforms fine-tuned GPT-3.5. Additionally, we examine LLM errors and introduce metrics to quantify semantic divergence in model-generated rationales, revealing notable differences and paradoxical behaviors among LLMs. Our experiments highlight the differences observed across LLMs' utility, explainability, and reliability. Code and resources available at: https://github.com/idramalab/quantify-llm-explanations
>
---
#### [replaced 026] GDS Agent for Graph Algorithmic Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.20637v2](http://arxiv.org/pdf/2508.20637v2)**

> **作者:** Borun Shi; Ioannis Panagiotas
>
> **备注:** Technical report
>
> **摘要:** Large language models (LLMs) have shown remarkable multimodal information processing and reasoning ability. When equipped with tools through function calling and enhanced with retrieval-augmented techniques, compound LLM-based systems can access closed data sources and answer questions about them. However, they still struggle to process and reason over large-scale graph-structure data. We introduce the GDS (Graph Data Science) agent in this technical report. The GDS agent introduces a comprehensive set of graph algorithms as tools, together with preprocessing (retrieval) and postprocessing of algorithm results, in a model context protocol (MCP) server. The server can be used with any modern LLM out-of-the-box. GDS agent allows users to ask any question that implicitly and intrinsically requires graph algorithmic reasoning about their data, and quickly obtain accurate and grounded answers. We introduce new benchmarks that evaluate intermediate tool calls as well as final responses. The results indicate that GDS agent is able to solve a wide spectrum of graph tasks. We also provide detailed case studies for more open-ended tasks and study scenarios where the agent struggles. Finally, we discuss the remaining challenges and the future roadmap.
>
---
#### [replaced 027] Activation Transport Operators
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.17540v2](http://arxiv.org/pdf/2508.17540v2)**

> **作者:** Andrzej Szablewski; Marek Masiak
>
> **备注:** 5 pages, 5 figures, references and appendices
>
> **摘要:** The residual stream mediates communication between transformer decoder layers via linear reads and writes of non-linear computations. While sparse-dictionary learning-based methods locate features in the residual stream, and activation patching methods discover circuits within the model, the mechanism by which features flow through the residual stream remains understudied. Understanding this dynamic can better inform jailbreaking protections, enable early detection of model mistakes, and their correction. In this work, we propose Activation Transport Operators (ATO), linear maps from upstream to downstream residuals $k$ layers later, evaluated in feature space using downstream SAE decoder projections. We empirically demonstrate that these operators can determine whether a feature has been linearly transported from a previous layer or synthesised from non-linear layer computation. We develop the notion of transport efficiency, for which we provide an upper bound, and use it to estimate the size of the residual stream subspace that corresponds to linear transport. We empirically demonstrate the linear transport, report transport efficiency and the size of the residual stream's subspace involved in linear transport. This compute-light (no finetuning, <50 GPU-h) method offers practical tools for safety, debugging, and a clearer picture of where computation in LLMs behaves linearly.
>
---
#### [replaced 028] AlphaDecay: Module-wise Weight Decay for Heavy-Tailed Balancing in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.14562v3](http://arxiv.org/pdf/2506.14562v3)**

> **作者:** Di He; Songjun Tu; Ajay Jaiswal; Li Shen; Ganzhao Yuan; Shiwei Liu; Lu Yin
>
> **摘要:** Weight decay is a standard regularization technique for training large language models (LLMs). While it is common to assign a uniform decay rate to every layer, this approach overlooks the structural diversity of LLMs and the varying spectral properties across modules. In this paper, we introduce AlphaDecay, a simple yet effective method that adaptively assigns different weight decay strengths to each module of an LLM. Our approach is guided by Heavy-Tailed Self-Regularization (HT-SR) theory, which analyzes the empirical spectral density (ESD) of weight correlation matrices to quantify "heavy-tailedness." Modules exhibiting more pronounced heavy-tailed ESDs, reflecting stronger feature learning, are assigned weaker decay, while modules with lighter-tailed spectra receive stronger decay. Our method leverages tailored weight decay assignments to balance the module-wise differences in spectral properties, leading to improved performance. Extensive pre-training tasks with various model sizes from 60M to 1B demonstrate that AlphaDecay achieves better perplexity and generalization than conventional uniform decay and other adaptive decay baselines. Our code is available at https://github.com/hed-ucas/AlphaDecay.
>
---
#### [replaced 029] Meta-Semantics Augmented Few-Shot Relational Learning
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.05684v4](http://arxiv.org/pdf/2505.05684v4)**

> **作者:** Han Wu; Jie Yin
>
> **备注:** Appear in EMNLP 2025
>
> **摘要:** Few-shot relational learning on knowledge graph (KGs) aims to perform reasoning over relations with only a few training examples. While current methods have focused primarily on leveraging specific relational information, rich semantics inherent in KGs have been largely overlooked. To bridge this gap, we propose PromptMeta, a novel prompted meta-learning framework that seamlessly integrates meta-semantics with relational information for few-shot relational learning. PromptMeta introduces two core innovations: (1) a Meta-Semantic Prompt (MSP) pool that learns and consolidates high-level meta-semantics shared across tasks, enabling effective knowledge transfer and adaptation to newly emerging relations; and (2) a learnable fusion mechanism that dynamically combines meta-semantics with task-specific relational information tailored to different few-shot tasks. Both components are optimized jointly with model parameters within a meta-learning framework. Extensive experiments and analyses on two real-world KG benchmarks validate the effectiveness of PromptMeta in adapting to new relations with limited supervision.
>
---
#### [replaced 030] Retrieval-Augmented Feature Generation for Domain-Specific Classification
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.11177v3](http://arxiv.org/pdf/2406.11177v3)**

> **作者:** Xinhao Zhang; Jinghan Zhang; Fengran Mo; Dakshak Keerthi Chandra; Yuzhong Chen; Fei Xie; Kunpeng Liu
>
> **备注:** Accepted by ICDM 2025
>
> **摘要:** Feature generation can significantly enhance learning outcomes, particularly for tasks with limited data. An effective way to improve feature generation is to expand the current feature space using existing features and enriching the informational content. However, generating new, interpretable features usually requires domain-specific knowledge on top of the existing features. In this paper, we introduce a Retrieval-Augmented Feature Generation method, RAFG, to generate useful and explainable features specific to domain classification tasks. To increase the interpretability of the generated features, we conduct knowledge retrieval among the existing features in the domain to identify potential feature associations. These associations are expected to help generate useful features. Moreover, we develop a framework based on large language models (LLMs) for feature generation with reasoning to verify the quality of the features during their generation process. Experiments across several datasets in medical, economic, and geographic domains show that our RAFG method can produce high-quality, meaningful features and significantly improve classification performance compared with baseline methods.
>
---
#### [replaced 031] Verdict: A Library for Scaling Judge-Time Compute
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.18018v2](http://arxiv.org/pdf/2502.18018v2)**

> **作者:** Nimit Kalra; Leonard Tang
>
> **摘要:** The use of LLMs as automated judges ("LLM-as-a-judge") is now widespread, yet standard judges suffer from a multitude of reliability issues. To address these challenges, we introduce Verdict, an open-source library for scaling judge-time compute to enhance the accuracy, reliability, and interpretability of automated evaluators. Verdict leverages the composition of modular reasoning units (such as verification, debate, and aggregation) and increased inference-time compute to improve LLM judge quality. Across a variety of challenging tasks such as content moderation, fact-checking, and hallucination detection, Verdict judges achieves performance competitive with orders-of-magnitude larger fine-tuned judges, prompted judges, and reasoning models. Our framework establishes a foundation for scalable, interpretable, and reliable LLM-based evaluation systems for both researchers and practitioners.
>
---
#### [replaced 032] Read Your Own Mind: Reasoning Helps Surface Self-Confidence Signals in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23845v2](http://arxiv.org/pdf/2505.23845v2)**

> **作者:** Jakub Podolak; Rajeev Verma
>
> **备注:** Presented at UncertaiNLP Workshop at EMNLP 2025 https://aclanthology.org/2025.uncertainlp-main.21.pdf
>
> **摘要:** We study the source of uncertainty in DeepSeek R1-32B by analyzing its self-reported verbal confidence on question answering (QA) tasks. In the default answer-then-confidence setting, the model is regularly over-confident, whereas semantic entropy - obtained by sampling many responses - remains reliable. We hypothesize that this is because of semantic entropy's larger test-time compute, which lets us explore the model's predictive distribution. We show that granting DeepSeek the budget to explore its distribution by forcing a long chain-of-thought before the final answer greatly improves its verbal score effectiveness, even on simple fact-retrieval questions that normally require no reasoning. Furthermore, a separate reader model that sees only the chain can reconstruct very similar confidences, indicating the verbal score might be merely a statistic of the alternatives surfaced during reasoning. Our analysis concludes that reliable uncertainty estimation requires explicit exploration of the generative space, and self-reported confidence is trustworthy only after such exploration.
>
---
#### [replaced 033] CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization
- **分类: cs.LG; cs.AI; cs.CL; cs.DC**

- **链接: [http://arxiv.org/pdf/2511.01884v2](http://arxiv.org/pdf/2511.01884v2)**

> **作者:** Zijian Zhang; Rong Wang; Shiyang Li; Yuebo Luo; Mingyi Hong; Caiwen Ding
>
> **摘要:** Developing efficient CUDA kernels is increasingly critical for AI applications such as large-scale LLM training. However, manual kernel design is both costly and time-consuming, motivating automatic approaches that leverage LLMs for code generation. Existing methods for automatic kernel generation, however, often produce low-efficiency kernels, incur high computational overhead, and fail to generalize across settings. In this work, we propose CudaForge, a training-free multi-agent workflow for CUDA kernel generation and optimization. Our workflow is inspired by the iterative workflow of human experts, which contains steps such as developing initial kernels, testing correctness, analyzing hardware feedback, and iterative improvement. More specifically, CudaForge employs two LLM agents: a Coder and a Judge, that iteratively generate, correct, and optimize CUDA kernels, while integrating hardware feedback such as Nsight Compute (NCU) metrics. In extensive evaluations, we show that CudaForge, by leveraging base models like OpenAI-o3, achieves 97.6\% correctness of generated kernels and an average 1.68$\times$ speedup over PyTorch baselines, substantially surpassing state-of-the-art models including OpenAI-o3 and Kevin on KernelBench.Beyond accuracy and speed, CudaForge demonstrates strong generalization across GPUs (A100, RTX 6000, 4090, 3090) and base models (OpenAI-o3, GPT-5, gpt-oss-120B, Claude-Sonnet-4, QwQ-32B), while maintaining high efficiency. In particular, generating an optimized kernel takes about 26.5 minutes on one RTX6000 and incurs about \$ 0.3 API cost, which is significantly cheaper than existing agentic work that costs 6 H100 hours and \$ 5 API cost per kernel. Our results highlight that multi-agent, training-free workflows can enable cost-effective, generalizable, and high-performance CUDA kernel optimization. Code available at https://github.com/OptimAI-Lab/CudaForge
>
---
#### [replaced 034] Do Automatic Factuality Metrics Measure Factuality? A Critical Evaluation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.16638v4](http://arxiv.org/pdf/2411.16638v4)**

> **作者:** Sanjana Ramprasad; Byron C. Wallace
>
> **摘要:** Modern LLMs can now produce highly readable abstractive summaries, to the point that traditional automated metrics for evaluating summary quality, such as ROUGE, have saturated. However, LLMs still sometimes introduce inaccuracies into summaries, i.e., information inconsistent with or unsupported by the corresponding source. Measuring the occurrence of these often subtle factual inconsistencies automatically has proved challenging. This in turn has motivated development of metrics intended to measure the factual consistency of generated summaries against sources. But are these approaches measuring what they purport to? Or are they mostly exploiting artifacts? In this work, we stress test a range of automatic factuality metrics, including specialized models and LLM-based prompting methods, to probe what they actually capture. Using a shallow classifier to separate ``easy'' examples for factual evaluation where surface features suffice from ``hard'' cases requiring deeper reasoning, we find that all metrics show substantial performance drops on the latter. Furthermore, some metrics are more sensitive to benign, fact-preserving edits than to factual corrections. Building on this observation, we demonstrate that most automatic factuality metrics can be gamed, i.e., their scores can be artificially inflated by appending innocuous, content-free sentences to summaries. Among the metrics tested, the prompt based ChatGPT-DA approach is the most robust and reliable. However, this comes with a notable caveat: Prompting LLMs to assess factuality may overly rely on their parametric knowledge rather than the provided reference when making judgments. Taken together, our findings call into question the reliability of current factuality metrics and prompt a broader reflection on what these metrics are truly measuring.
>
---
#### [replaced 035] FaStfact: Faster, Stronger Long-Form Factuality Evaluations in LLMs
- **分类: cs.CL; cs.AI; cs.CE; cs.CY**

- **链接: [http://arxiv.org/pdf/2510.12839v2](http://arxiv.org/pdf/2510.12839v2)**

> **作者:** Yingjia Wan; Haochen Tan; Xiao Zhu; Xinyu Zhou; Zhiwei Li; Qingsong Lv; Changxuan Sun; Jiaqi Zeng; Yi Xu; Jianqiao Lu; Yinhong Liu; Zhijiang Guo
>
> **备注:** EMNLP 2025 (Findings)
>
> **摘要:** Evaluating the factuality of long-form generations from Large Language Models (LLMs) remains challenging due to efficiency bottlenecks and reliability concerns. Prior efforts attempt this by decomposing text into claims, searching for evidence, and verifying claims, but suffer from critical drawbacks: (1) inefficiency due to overcomplicated pipeline components, and (2) ineffectiveness stemming from inaccurate claim sets and insufficient evidence. To address these limitations, we propose \textbf{FaStfact}, an evaluation framework that achieves the highest alignment with human evaluation and time/token efficiency among existing baselines. FaStfact first employs chunk-level claim extraction integrated with confidence-based pre-verification, significantly reducing the time and token cost while ensuring reliability. For searching and verification, it collects document-level evidence from crawled web-pages and selectively retrieves it during verification. Extensive experiments based on an annotated benchmark \textbf{FaStfact-Bench} demonstrate the reliability of FaStfact in both efficiently and effectively evaluating long-form factuality. Code, benchmark data, and annotation interface tool are available at https://github.com/Yingjia-Wan/FaStfact.
>
---
#### [replaced 036] From Haystack to Needle: Label Space Reduction for Zero-shot Classification
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.08436v2](http://arxiv.org/pdf/2502.08436v2)**

> **作者:** Nathan Vandemoortele; Bram Steenwinckel; Femke Ongenae; Sofie Van Hoecke
>
> **备注:** Add acknowledgment
>
> **摘要:** We present Label Space Reduction (LSR), a novel method for improving zero-shot classification performance of Large Language Models (LLMs). LSR iteratively refines the classification label space by systematically ranking and reducing candidate classes, enabling the model to concentrate on the most relevant options. By leveraging unlabeled data with the statistical learning capabilities of data-driven models, LSR dynamically optimizes the label space representation at test time. Our experiments across seven benchmarks demonstrate that LSR improves macro-F1 scores by an average of 7.0% (up to 14.2%) with Llama-3.1-70B and 3.3% (up to 11.1%) with Claude-3.5-Sonnet compared to standard zero-shot classification baselines. To reduce the computational overhead of LSR, which requires an additional LLM call at each iteration, we propose distilling the model into a probabilistic classifier, allowing for efficient inference.
>
---
#### [replaced 037] Token Perturbation Guidance for Diffusion Models
- **分类: cs.GR; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.10036v2](http://arxiv.org/pdf/2506.10036v2)**

> **作者:** Javad Rajabi; Soroush Mehraban; Seyedmorteza Sadat; Babak Taati
>
> **备注:** Accepted at NeurIPS 2025. Project page: https://github.com/TaatiTeam/Token-Perturbation-Guidance
>
> **摘要:** Classifier-free guidance (CFG) has become an essential component of modern diffusion models to enhance both generation quality and alignment with input conditions. However, CFG requires specific training procedures and is limited to conditional generation. To address these limitations, we propose Token Perturbation Guidance (TPG), a novel method that applies perturbation matrices directly to intermediate token representations within the diffusion network. TPG employs a norm-preserving shuffling operation to provide effective and stable guidance signals that improve generation quality without architectural changes. As a result, TPG is training-free and agnostic to input conditions, making it readily applicable to both conditional and unconditional generation. We further analyze the guidance term provided by TPG and show that its effect on sampling more closely resembles CFG compared to existing training-free guidance techniques. Extensive experiments on SDXL and Stable Diffusion 2.1 show that TPG achieves nearly a 2$\times$ improvement in FID for unconditional generation over the SDXL baseline, while closely matching CFG in prompt alignment. These results establish TPG as a general, condition-agnostic guidance method that brings CFG-like benefits to a broader class of diffusion models.
>
---
#### [replaced 038] Constraint-Driven Small Language Models Based on Agent and OpenAlex Knowledge Graph: Mining Conceptual Pathways and Discovering Innovation Points in Academic Papers
- **分类: cs.CL; cs.LG; I.2.7**

- **链接: [http://arxiv.org/pdf/2510.14303v2](http://arxiv.org/pdf/2510.14303v2)**

> **作者:** Ziye Xia; Sergei S. Ospichev
>
> **备注:** 11 pages, 10 figures
>
> **摘要:** In recent years, the rapid increase in academic publications across various fields has posed severe challenges for academic paper analysis: scientists struggle to timely and comprehensively track the latest research findings and methodologies. Key concept extraction has proven to be an effective analytical paradigm, and its automation has been achieved with the widespread application of language models in industrial and scientific domains. However, existing paper databases are mostly limited to similarity matching and basic classification of key concepts, failing to deeply explore the relational networks between concepts. This paper is based on the OpenAlex opensource knowledge graph. By analyzing nearly 8,000 open-source paper data from Novosibirsk State University, we discovered a strong correlation between the distribution patterns of paper key concept paths and both innovation points and rare paths. We propose a prompt engineering-based key concept path analysis method. This method leverages small language models to achieve precise key concept extraction and innovation point identification, and constructs an agent based on a knowledge graph constraint mechanism to enhance analysis accuracy. Through fine-tuning of the Qwen and DeepSeek models, we achieved significant improvements in accuracy, with the models publicly available on the Hugging Face platform.
>
---
#### [replaced 039] Distilling LLM Agent into Small Models with Retrieval and Code Tools
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.17612v2](http://arxiv.org/pdf/2505.17612v2)**

> **作者:** Minki Kang; Jongwon Jeong; Seanie Lee; Jaewoong Cho; Sung Ju Hwang
>
> **备注:** NeurIPS 2025 Spotlight
>
> **摘要:** Large language models (LLMs) excel at complex reasoning tasks but remain computationally expensive, limiting their practical deployment. To address this, recent works have focused on distilling reasoning capabilities into smaller language models (sLMs) using chain-of-thought (CoT) traces from teacher LLMs. However, this approach struggles in scenarios requiring rare factual knowledge or precise computation, where sLMs often hallucinate due to limited capability. In this work, we propose Agent Distillation, a framework for transferring not only reasoning capability but full task-solving behavior from LLM-based agents into sLMs with retrieval and code tools. We improve agent distillation along two complementary axes: (1) we introduce a prompting method called first-thought prefix to enhance the quality of teacher-generated trajectories; and (2) we propose a self-consistent action generation for improving test-time robustness of small agents. We evaluate our method on eight reasoning tasks across factual and mathematical domains, covering both in-domain and out-of-domain generalization. Our results show that sLMs as small as 0.5B, 1.5B, 3B parameters can achieve performance competitive with next-tier larger 1.5B, 3B, 7B models fine-tuned using CoT distillation, demonstrating the potential of agent distillation for building practical, tool-using small agents. Our code is available at https://github.com/Nardien/agent-distillation.
>
---
#### [replaced 040] PhysicsEval: Inference-Time Techniques to Improve the Reasoning Proficiency of Large Language Models on Physics Problems
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.00079v2](http://arxiv.org/pdf/2508.00079v2)**

> **作者:** Oshayer Siddique; J. M Areeb Uzair Alam; Md Jobayer Rahman Rafy; Syed Rifat Raiyan; Hasan Mahmud; Md Kamrul Hasan
>
> **备注:** Accepted in Findings of the Association for Computational Linguistics: IJCNLP-AACL 2025, 23 pages, 4 figures, 8 tables
>
> **摘要:** The discipline of physics stands as a cornerstone of human intellect, driving the evolution of technology and deepening our understanding of the fundamental principles of the cosmos. Contemporary literature includes some works centered on the task of solving physics problems - a crucial domain of natural language reasoning. In this paper, we evaluate the performance of frontier LLMs in solving physics problems, both mathematical and descriptive. We also employ a plethora of inference-time techniques and agentic frameworks to improve the performance of the models. This includes the verification of proposed solutions in a cumulative fashion by other, smaller LLM agents, and we perform a comparative analysis of the performance that the techniques entail. There are significant improvements when the multi-agent framework is applied to problems that the models initially perform poorly on. Furthermore, we introduce a new evaluation benchmark for physics problems, ${\rm P{\small HYSICS}E{\small VAL}}$, consisting of 19,609 problems sourced from various physics textbooks and their corresponding correct solutions scraped from physics forums and educational websites. Our code and data are publicly available at https://github.com/areebuzair/PhysicsEval.
>
---
#### [replaced 041] Which Way Does Time Flow? A Psychophysics-Grounded Evaluation for Vision-Language Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.26241v2](http://arxiv.org/pdf/2510.26241v2)**

> **作者:** Shiho Matta; Lis Kanashiro Pereira; Peitao Han; Fei Cheng; Shigeru Kitazawa
>
> **备注:** 10 pages
>
> **摘要:** Modern vision-language models (VLMs) excel at many multimodal tasks, yet their grasp of temporal information in video remains weak and, crucially, under-evaluated. We probe this gap with a deceptively simple but revealing challenge: judging the arrow of time (AoT)-whether a short clip is played forward or backward. We introduce AoT-PsyPhyBENCH, a psychophysically validated benchmark that tests whether VLMs can infer temporal direction in natural videos using the same stimuli and behavioral baselines established for humans. Our comprehensive evaluation of open-weight and proprietary, reasoning and non-reasoning VLMs reveals that most models perform near chance, and even the best lag far behind human accuracy on physically irreversible processes (e.g., free fall, diffusion/explosion) and causal manual actions (division/addition) that humans recognize almost instantly. These results highlight a fundamental gap in current multimodal systems: while they capture rich visual-semantic correlations, they lack the inductive biases required for temporal continuity and causal understanding. We release the code and data for AoT-PsyPhyBENCH to encourage further progress in the physical and temporal reasoning capabilities of VLMs.
>
---
#### [replaced 042] Assessing the Macro and Micro Effects of Random Seeds on Fine-Tuning Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.07329v2](http://arxiv.org/pdf/2503.07329v2)**

> **作者:** Nghia Bui; Guergana Savova; Lijing Wang
>
> **备注:** 7 pages, 5 tables, 3 figures. Accepted at IJCNLP 2025. This is the final, peer-reviewed version of the work, which supersedes and extends the unauthorized draft previously posted as arXiv:2503.07329
>
> **摘要:** The impact of random seeds in fine-tuning large language models (LLMs) has been largely overlooked despite its potential influence on model performance.In this study, we systematically evaluate the effects of random seeds on LLMs using the GLUE and SuperGLUE benchmarks. We analyze the macro-level impact through traditional metrics like accuracy and F1, calculating their mean and variance to quantify performance fluctuations. To capture the micro-level effects, we introduce a novel metric, consistency, measuring the stability of individual predictions across runs. Our experiments reveal significant variance at both macro and micro levels, underscoring the need for careful consideration of random seeds in fine-tuning and evaluation.
>
---
#### [replaced 043] TABLET: A Large-Scale Dataset for Robust Visual Table Understanding
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.21205v2](http://arxiv.org/pdf/2509.21205v2)**

> **作者:** Iñigo Alonso; Imanol Miranda; Eneko Agirre; Mirella Lapata
>
> **摘要:** While table understanding increasingly relies on pixel-only settings where tables are processed as visual representations, current benchmarks predominantly use synthetic renderings that lack the complexity and visual diversity of real-world tables. Additionally, existing visual table understanding (VTU) datasets offer fixed examples with single visualizations and pre-defined instructions, providing no access to underlying serialized data for reformulation. We introduce TABLET, a large-scale VTU dataset with 4 million examples across 20 tasks, grounded in 2 million unique tables where 88% preserve original visualizations. Each example includes paired image-HTML representations, comprehensive metadata, and provenance information linking back to the source datasets. Fine-tuning vision-language models like Qwen2.5-VL-7B on TABLET improves performance on seen and unseen VTU tasks while increasing robustness on real-world table visualizations. By preserving original visualizations and maintaining example traceability in a unified large-scale collection, TABLET establishes a foundation for robust training and extensible evaluation of future VTU models.
>
---
#### [replaced 044] Matryoshka Pilot: Learning to Drive Black-Box LLMs with LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.20749v3](http://arxiv.org/pdf/2410.20749v3)**

> **作者:** Changhao Li; Yuchen Zhuang; Rushi Qiang; Haotian Sun; Hanjun Dai; Chao Zhang; Bo Dai
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Despite the impressive generative abilities of black-box large language models (LLMs), their inherent opacity hinders further advancements in capabilities such as reasoning, planning, and personalization. Existing works aim to enhance LLM capabilities via domain-specific adaptation, which require additional training on accessible model parameters, an infeasible option for black-box LLMs. To address this challenge, we introduce Matryoshka Pilot (M-Pilot), a lightweight white-box LLM controller that guides a large-scale black-box LLM generator by decomposing complex tasks into a series of intermediate outputs. Specifically, we consider the black-box LLM as an environment, with M-Pilot serving as a policy to provide intermediate guidance through prompts for driving the black-box LLM. M-Pilot is trained to pivot the outputs of the black-box LLM aligning with preferences during iterative interaction, which enables controllable multi-turn generation and self-improvement in optimizing intermediate guidance. Empirical evaluations on diverse tasks demonstrate that our method effectively enhances the capabilities of black-box LLMs in complex, long-horizon tasks. Our code is publicly available at: https://github.com/lichangh20/Matryoshka.
>
---
#### [replaced 045] Training Optimal Large Diffusion Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.03280v2](http://arxiv.org/pdf/2510.03280v2)**

> **作者:** Jinjie Ni; Qian Liu; Chao Du; Longxu Dou; Hang Yan; Zili Wang; Tianyu Pang; Michael Qizhe Shieh
>
> **摘要:** We introduce Quokka, the first systematic scaling law for diffusion language models (DLMs), encompassing both compute-constrained and data-constrained regimes, and studying the key modeling and optimization designs. Quokka is a good friend of Chinchilla and provides wider scopes. We hope the results would bring short-term practical guidance in DLMs training and long-term inspirations for the whole AI community.
>
---
#### [replaced 046] Sparse-dLLM: Accelerating Diffusion LLMs with Dynamic Cache Eviction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.02558v2](http://arxiv.org/pdf/2508.02558v2)**

> **作者:** Yuerong Song; Xiaoran Liu; Ruixiao Li; Zhigeng Liu; Zengfeng Huang; Qipeng Guo; Ziwei He; Xipeng Qiu
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** Diffusion Large Language Models (dLLMs) enable breakthroughs in reasoning and parallel decoding but suffer from prohibitive quadratic computational complexity and memory overhead during inference. Current caching techniques accelerate decoding by storing full-layer states, yet impose substantial memory usage that limit long-context applications. Our analysis of attention patterns in dLLMs reveals persistent cross-layer sparsity, with pivotal tokens remaining salient across decoding steps and low-relevance tokens staying unimportant, motivating selective cache eviction. We propose Sparse-dLLM, the first training-free framework integrating dynamic cache eviction with sparse attention via delayed bidirectional sparse caching. By leveraging the stability of token saliency over steps, it retains critical tokens and dynamically evicts unimportant prefix/suffix entries using an attention-guided strategy. Extensive experiments on LLaDA and Dream series demonstrate Sparse-dLLM achieves up to 10$\times$ higher throughput than vanilla dLLMs, with comparable performance and similar peak memory costs, outperforming previous methods in efficiency and effectiveness. The code is available at https://github.com/OpenMOSS/Sparse-dLLM.
>
---
#### [replaced 047] Emotion Detection From Social Media Posts
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2302.05610v2](http://arxiv.org/pdf/2302.05610v2)**

> **作者:** Md Mahbubur Rahman; Shaila Sharmin
>
> **备注:** Course Project
>
> **摘要:** Over the last few years, social media has evolved into a medium for expressing personal views, emotions, and even business and political proposals, recommendations, and advertisements. We address the topic of identifying emotions from text data obtained from social media posts like Twitter in this research. We have deployed different traditional machine learning techniques such as Support Vector Machines (SVM), Naive Bayes, Decision Trees, and Random Forest, as well as deep neural network models such as LSTM, CNN, GRU, BiLSTM, BiGRU to classify these tweets into four emotion categories (Fear, Anger, Joy, and Sadness). Furthermore, we have constructed a BiLSTM and BiGRU ensemble model. The evaluation result shows that the deep neural network models(BiGRU, to be specific) produce the most promising results compared to traditional machine learning models, with an 87.53 % accuracy rate. The ensemble model performs even better (87.66 %), albeit the difference is not significant. This result will aid in the development of a decision-making tool that visualizes emotional fluctuations.
>
---
#### [replaced 048] A Survey on Collaborating Small and Large Language Models for Performance, Cost-effectiveness, Cloud-edge Privacy, and Trustworthiness
- **分类: cs.CL; cs.AI; 68T50 (Primary) 68T07 (Secondary); I.2.7**

- **链接: [http://arxiv.org/pdf/2510.13890v2](http://arxiv.org/pdf/2510.13890v2)**

> **作者:** Fali Wang; Jihai Chen; Shuhua Yang; Ali Al-Lawati; Linli Tang; Hui Liu; Suhang Wang
>
> **备注:** 24 pages, 19 figures-under review; more detailed than v1
>
> **摘要:** Large language models (LLMs) have achieved remarkable progress across domains and applications but face challenges such as high fine-tuning costs, inference latency, limited edge deployability, and reliability concerns. Small language models (SLMs), with compact, efficient, and adaptable features, offer promising solutions. Building on this potential, recent research explores collaborative frameworks that integrate their complementary strengths, leveraging SLMs' specialization and efficiency with LLMs' generalization and reasoning to address diverse objectives across tasks and deployment scenarios. Motivated by these developments, this paper presents a systematic survey of SLM-LLM collaboration from the perspective of collaboration objectives. We propose a taxonomy covering four goals: performance enhancement, cost-effectiveness, cloud-edge privacy, and trustworthiness. Under this framework, we review representative methods, summarize design paradigms, and outline open challenges and future directions toward efficient and secure SLM-LLM collaboration. The collected papers are available at https://github.com/FairyFali/SLMs-Survey.
>
---
#### [replaced 049] REFA: Reference Free Alignment for multi-preference optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.16378v4](http://arxiv.org/pdf/2412.16378v4)**

> **作者:** Taneesh Gupta; Rahul Madhavan; Xuchao Zhang; Chetan Bansal; Saravan Rajmohan
>
> **摘要:** To mitigate reward hacking from response verbosity, modern preference optimization methods are increasingly adopting length normalization (e.g., SimPO, ORPO, LN-DPO). While effective against this bias, we demonstrate that length normalization itself introduces a failure mode: the URSLA shortcut. Here models learn to satisfy the alignment objective by prematurely truncating low-quality responses rather than learning from their semantic content. To address this, we introduce REFA, a new alignment framework that proposes probabilistic control on a structural token that controls termination. Our core innovation is a new class of regularizers that operate directly on the probability of the End-of-Sequence (EOS) token, a previously unexploited control lever. This token-level intervention provides a principled solution to the URSLA shortcut, ensuring genuine quality improvements. Furthermore, it unlocks a versatile mechanism for managing the alignment-efficiency tradeoff, enabling practitioners to fine-tune models that adhere to specific token budgets. Empirically, REFA achieves a 60.29% win rate and a 52.17% length-controlled win rate on AlpacaEval2 with Llama-3-8B-Instruct, demonstrating the power of our token-level control paradigm.
>
---
#### [replaced 050] StutterZero and StutterFormer: End-to-End Speech Conversion for Stuttering Transcription and Correction
- **分类: eess.AS; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.18938v2](http://arxiv.org/pdf/2510.18938v2)**

> **作者:** Qianheng Xu
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Over 70 million people worldwide experience stuttering, yet most automatic speech systems misinterpret disfluent utterances or fail to transcribe them accurately. Existing methods for stutter correction rely on handcrafted feature extraction or multi-stage automatic speech recognition (ASR) and text-to-speech (TTS) pipelines, which separate transcription from audio reconstruction and often amplify distortions. This work introduces StutterZero and StutterFormer, the first end-to-end waveform-to-waveform models that directly convert stuttered speech into fluent speech while jointly predicting its transcription. StutterZero employs a convolutional-bidirectional LSTM encoder-decoder with attention, whereas StutterFormer integrates a dual-stream Transformer with shared acoustic-linguistic representations. Both architectures are trained on paired stuttered-fluent data synthesized from the SEP-28K and LibriStutter corpora and evaluated on unseen speakers from the FluencyBank dataset. Across all benchmarks, StutterZero had a 24% decrease in Word Error Rate (WER) and a 31% improvement in semantic similarity (BERTScore) compared to the leading Whisper-Medium model. StutterFormer achieved better results, with a 28% decrease in WER and a 34% improvement in BERTScore. The results validate the feasibility of direct end-to-end stutter-to-fluent speech conversion, offering new opportunities for inclusive human-computer interaction, speech therapy, and accessibility-oriented AI systems.
>
---
