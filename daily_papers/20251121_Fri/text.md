# 自然语言处理 cs.CL

- **最新发布 45 篇**

- **更新 49 篇**

## 最新发布

#### [new 001] Arctic-Extract Technical Report
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出Arctic-Extract模型，用于从扫描或数字生成的业务文档中提取结构化数据（问答、实体、表格）。针对资源受限设备部署难题，模型仅6.6 GiB，可在A10 GPU上处理高达125页的长文档，兼具高性能与低资源消耗，显著提升文档理解在边缘设备上的可行性。**

- **链接: [https://arxiv.org/pdf/2511.16470v1](https://arxiv.org/pdf/2511.16470v1)**

> **作者:** Mateusz Chiliński; Julita Ołtusek; Wojciech Jaśkowski
>
> **摘要:** Arctic-Extract is a state-of-the-art model designed for extracting structural data (question answering, entities and tables) from scanned or digital-born business documents. Despite its SoTA capabilities, the model is deployable on resource-constrained hardware, weighting only 6.6 GiB, making it suitable for deployment on devices with limited resources, such as A10 GPUs with 24 GB of memory. Arctic-Extract can process up to 125 A4 pages on those GPUs, making suitable for long document processing. This paper highlights Arctic-Extract's training protocols and evaluation results, demonstrating its strong performance in document understanding.
>
---
#### [new 002] AICC: Parse HTML Finer, Make Models Better -- A 7.3T AI-Ready Corpus Built by a Model-Based HTML Parser
- **分类: cs.CL**

- **简介: 该论文针对大模型训练中网页数据提取质量低的问题，提出基于语言模型的HTML解析器MinerU-HTML，将提取任务转为序列标注，显著提升结构化内容保留率。基于此构建7.3T tokens的AICC语料库，在相同过滤条件下使模型性能提升1.08pp，证明高质量提取对模型能力的关键作用。**

- **链接: [https://arxiv.org/pdf/2511.16397v1](https://arxiv.org/pdf/2511.16397v1)**

> **作者:** Ren Ma; Jiantao Qiu; Chao Xu; Pei Chu; Kaiwen Liu; Pengli Ren; Yuan Qu; Jiahui Peng; Linfeng Hou; Mengjie Liu; Lindong Lu; Wenchang Ning; Jia Yu; Rui Min; Jin Shi; Haojiong Chen; Peng Zhang; Wenjian Zhang; Qian Jiang; Zengjie Hu; Guoqiang Yang; Zhenxiang Li; Fukai Shang; Zhongying Tu; Wentao Zhang; Dahua Lin; Conghui He
>
> **摘要:** While web data quality is crucial for large language models, most curation efforts focus on filtering and deduplication,treating HTML-to-text extraction as a fixed pre-processing step. Existing web corpora rely on heuristic-based extractors like Trafilatura, which struggle to preserve document structure and frequently corrupt structured elements such as formulas, codes, and tables. We hypothesize that improving extraction quality can be as impactful as aggressive filtering strategies for downstream performance. We introduce MinerU-HTML, a novel extraction pipeline that reformulates content extraction as a sequence labeling problem solved by a 0.6B-parameter language model. Unlike text-density heuristics, MinerU-HTML leverages semantic understanding and employs a two-stage formatting pipeline that explicitly categorizes semantic elements before converting to Markdown. Crucially, its model-based approach is inherently scalable, whereas heuristic methods offer limited improvement pathways. On MainWebBench, our benchmark of 7,887 annotated web pages, MinerU-HTML achieves 81.8\% ROUGE-N F1 compared to Trafilatura's 63.6\%, with exceptional structured element preservation (90.9\% for code blocks, 94.0\% for formulas). Using MinerU-HTML, we construct AICC (AI-ready Common Crawl), a 7.3-trillion token multilingual corpus from two Common Crawl snapshots. In controlled pretraining experiments where AICC and Trafilatura-extracted TfCC undergo identical filtering, models trained on AICC (62B tokens) achieve 50.8\% average accuracy across 13 benchmarks, outperforming TfCC by 1.08pp-providing direct evidence that extraction quality significantly impacts model capabilities. AICC also surpasses RefinedWeb and FineWeb on key benchmarks. We publicly release MainWebBench, MinerU-HTML, and AICC, demonstrating that HTML extraction is a critical, often underestimated component of web corpus construction.
>
---
#### [new 003] NLP Datasets for Idiom and Figurative Language Tasks
- **分类: cs.CL**

- **简介: 该论文聚焦于汉语成语和修辞语言的自然语言处理任务，旨在解决大语言模型在理解习语和隐喻语言上的不足。研究构建了大规模、多类别数据集，通过整合与人工标注，提升模型在习语识别与语义理解上的表现，并支持模型无关训练与评估。**

- **链接: [https://arxiv.org/pdf/2511.16345v1](https://arxiv.org/pdf/2511.16345v1)**

> **作者:** Blake Matheny; Phuong Minh Nguyen; Minh Le Nguyen; Stephanie Reynolds
>
> **备注:** 32 pages, 10 figures
>
> **摘要:** Idiomatic and figurative language form a large portion of colloquial speech and writing. With social media, this informal language has become more easily observable to people and trainers of large language models (LLMs) alike. While the advantage of large corpora seems like the solution to all machine learning and Natural Language Processing (NLP) problems, idioms and figurative language continue to elude LLMs. Finetuning approaches are proving to be optimal, but better and larger datasets can help narrow this gap even further. The datasets presented in this paper provide one answer, while offering a diverse set of categories on which to build new models and develop new approaches. A selection of recent idiom and figurative language datasets were used to acquire a combined idiom list, which was used to retrieve context sequences from a large corpus. One large-scale dataset of potential idiomatic and figurative language expressions and two additional human-annotated datasets of definite idiomatic and figurative language expressions were created to evaluate the baseline ability of pre-trained language models in handling figurative meaning through idiom recognition (detection) tasks. The resulting datasets were post-processed for model agnostic training compatibility, utilized in training, and evaluated on slot labeling and sequence tagging.
>
---
#### [new 004] TOD-ProcBench: Benchmarking Complex Instruction-Following in Task-Oriented Dialogues
- **分类: cs.CL**

- **简介: 该论文针对任务导向对话中复杂指令遵循能力不足的问题，提出TOD-ProcBench基准。通过构建含细粒度约束的多层级条件-动作指令，设计三类任务评估大模型在多轮对话中理解与遵循复杂指令的能力，并分析多语言及格式影响，推动高质量指令遵循评测。**

- **链接: [https://arxiv.org/pdf/2511.15976v1](https://arxiv.org/pdf/2511.15976v1)**

> **作者:** Sarik Ghazarian; Abhinav Gullapalli; Swair Shah; Anurag Beniwal; Nanyun Peng; Narayanan Sadagopan; Zhou Yu
>
> **摘要:** In real-world task-oriented dialogue (TOD) settings, agents are required to strictly adhere to complex instructions while conducting multi-turn conversations with customers. These instructions are typically presented in natural language format and include general guidelines and step-by-step procedures with complex constraints. Existing TOD benchmarks often oversimplify the complex nature of these instructions by reducing them to simple schemas composed of intents, slots, and API call configurations. To address this gap and systematically benchmark LLMs' instruction-following capabilities, we propose TOD-ProcBench, a challenging benchmark featuring complex process instructions with intricate, fine-grained constraints that evaluates various LLMs' abilities to understand and follow instructions in multi-turn TODs. Our benchmark dataset comprises instruction documents derived from the high-quality ABCD dataset with corresponding conversations under human quality control. We formulate fine-grained constraints and action procedures as multi-level condition-action instruction statements. We design three tasks to comprehensively benchmark LLMs' complex instruction-following capabilities in multi-turn TODs. Task 1 evaluates how LLMs retrieve the most relevant statement from a complex instruction and predict the corresponding next action. In Task 2, we synthesize instruction-violating responses by injecting inconsistencies and manipulating the original instructions, and then we analyze how effectively LLMs can identify instruction-violating responses. Task 3 investigates LLMs' abilities in conditional generation of instruction-following responses based on the original complex instructions. Additionally, we conduct studies on the impact of multilingual settings and different instruction text formats on compliance performance. We release our benchmark under the Llama 3.3 Community License Agreement.
>
---
#### [new 005] ESGBench: A Benchmark for Explainable ESG Question Answering in Corporate Sustainability Reports
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出ESGBench，一个用于评估可解释性ESG问答系统的基准数据集与评估框架。针对企业可持续报告中AI模型缺乏事实一致性与可追溯性的难题，构建了多主题、带证据的问答对，推动透明、可信的ESG-AI研究。**

- **链接: [https://arxiv.org/pdf/2511.16438v1](https://arxiv.org/pdf/2511.16438v1)**

> **作者:** Sherine George; Nithish Saji
>
> **备注:** Workshop paper accepted at AI4DF 2025 (part of ACM ICAIF 2025). 3 pages including tables and figures
>
> **摘要:** We present ESGBench, a benchmark dataset and evaluation framework designed to assess explainable ESG question answering systems using corporate sustainability reports. The benchmark consists of domain-grounded questions across multiple ESG themes, paired with human-curated answers and supporting evidence to enable fine-grained evaluation of model reasoning. We analyze the performance of state-of-the-art LLMs on ESGBench, highlighting key challenges in factual consistency, traceability, and domain alignment. ESGBench aims to accelerate research in transparent and accountable ESG-focused AI systems.
>
---
#### [new 006] Liars' Bench: Evaluating Lie Detectors for Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型谎言检测任务，指出现有方法在多样化谎言场景下表现不佳。为此构建LIARS' BENCH测试集，包含7万余例谎言与真实回答，涵盖不同谎言类型和信念目标。评估发现现有技术在无法仅从文本判断是否说谎的场景中系统性失效，揭示了当前方法的局限性，并为未来研究提供基准。**

- **链接: [https://arxiv.org/pdf/2511.16035v1](https://arxiv.org/pdf/2511.16035v1)**

> **作者:** Kieron Kretschmar; Walter Laurito; Sharan Maiya; Samuel Marks
>
> **备注:** *Kieron Kretschmar and Walter Laurito contributed equally to this work. 10 pages, 2 figures; plus appendix. Code at https://github.com/Cadenza-Labs/liars-bench and datasets at https://huggingface.co/datasets/Cadenza-Labs/liars-bench Subjects: Computation and Language (cs.CL); Artificial Intelligence (cs.AI)
>
> **摘要:** Prior work has introduced techniques for detecting when large language models (LLMs) lie, that is, generating statements they believe are false. However, these techniques are typically validated in narrow settings that do not capture the diverse lies LLMs can generate. We introduce LIARS' BENCH, a testbed consisting of 72,863 examples of lies and honest responses generated by four open-weight models across seven datasets. Our settings capture qualitatively different types of lies and vary along two dimensions: the model's reason for lying and the object of belief targeted by the lie. Evaluating three black- and white-box lie detection techniques on LIARS' BENCH, we find that existing techniques systematically fail to identify certain types of lies, especially in settings where it's not possible to determine whether the model lied from the transcript alone. Overall, LIARS' BENCH reveals limitations in prior techniques and provides a practical testbed for guiding progress in lie detection.
>
---
#### [new 007] TS-PEFT: Token-Selective Parameter-Efficient Fine-Tuning with Learnable Threshold Gating
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型微调中参数效率问题，提出Token-Selective PEFT（TS-PEFT），通过可学习阈值门控机制选择性地对部分位置索引进行参数高效微调。解决了传统PEFT全量应用修改不必要且可能有害的问题，实现了更精准、高效的微调策略。**

- **链接: [https://arxiv.org/pdf/2511.16147v1](https://arxiv.org/pdf/2511.16147v1)**

> **作者:** Dabiao Ma; Ziming Dai; Zhimin Xin; Shu Wang; Ye Wang; Haojun Fei
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** In the field of large models (LMs) for natural language processing (NLP) and computer vision (CV), Parameter-Efficient Fine-Tuning (PEFT) has emerged as a resource-efficient method that modifies a limited number of parameters while keeping the pretrained weights fixed. This paper investigates the traditional PEFT approach, which applies modifications to all position indices, and questions its necessity. We introduce a new paradigm called Token-Selective PEFT (TS-PEFT), in which a function S selectively applies PEFT modifications to a subset of position indices, potentially enhancing performance on downstream tasks. Our experimental results reveal that the indiscriminate application of PEFT to all indices is not only superfluous, but may also be counterproductive. This study offers a fresh perspective on PEFT, advocating for a more targeted approach to modifications and providing a framework for future research to optimize the fine-tuning process for large models.
>
---
#### [new 008] Comparison of Text-Based and Image-Based Retrieval in Multimodal Retrieval Augmented Generation Large Language Model Systems
- **分类: cs.CL**

- **简介: 该论文研究多模态RAG系统中的信息检索任务，针对现有方法依赖文本摘要导致视觉信息丢失的问题，比较了文本块检索与直接多模态嵌入检索。实验表明，后者在准确率和一致性上显著更优，证明保留原始图像信息对提升性能至关重要。**

- **链接: [https://arxiv.org/pdf/2511.16654v1](https://arxiv.org/pdf/2511.16654v1)**

> **作者:** Elias Lumer; Alex Cardenas; Matt Melich; Myles Mason; Sara Dieter; Vamse Kumar Subbiah; Pradeep Honaganahalli Basavaraju; Roberto Hernandez
>
> **摘要:** Recent advancements in Retrieval-Augmented Generation (RAG) have enabled Large Language Models (LLMs) to access multimodal knowledge bases containing both text and visual information such as charts, diagrams, and tables in financial documents. However, existing multimodal RAG systems rely on LLM-based summarization to convert images into text during preprocessing, storing only text representations in vector databases, which causes loss of contextual information and visual details critical for downstream retrieval and question answering. To address this limitation, we present a comprehensive comparative analysis of two retrieval approaches for multimodal RAG systems, including text-based chunk retrieval (where images are summarized into text before embedding) and direct multimodal embedding retrieval (where images are stored natively in the vector space). We evaluate all three approaches across 6 LLM models and a two multi-modal embedding models on a newly created financial earnings call benchmark comprising 40 question-answer pairs, each paired with 2 documents (1 image and 1 text chunk). Experimental results demonstrate that direct multimodal embedding retrieval significantly outperforms LLM-summary-based approaches, achieving absolute improvements of 13% in mean average precision (mAP@5) and 11% in normalized discounted cumulative gain. These gains correspond to relative improvements of 32% in mAP@5 and 20% in nDCG@5, providing stronger evidence of their practical impact. We additionally find that direct multimodal retrieval produces more accurate and factually consistent answers as measured by LLM-as-a-judge pairwise comparisons. We demonstrate that LLM summarization introduces information loss during preprocessing, whereas direct multimodal embeddings preserve visual context for retrieval and inference.
>
---
#### [new 009] Classification of worldwide news articles by perceived quality, 2018-2024
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于新闻质量分类任务，旨在区分低质与高质新闻。基于140万篇英文新闻，利用579个网站的专家评级，构建了包含194个语言特征的数据集，评估了3种机器学习和3种深度学习模型，结果表明深度学习模型在准确率和AUC上表现更优，可有效识别新闻感知质量。**

- **链接: [https://arxiv.org/pdf/2511.16416v1](https://arxiv.org/pdf/2511.16416v1)**

> **作者:** Connor McElroy; Thiago E. A. de Oliveira; Chris Brogly
>
> **摘要:** This study explored whether supervised machine learning and deep learning models can effectively distinguish perceived lower-quality news articles from perceived higher-quality news articles. 3 machine learning classifiers and 3 deep learning models were assessed using a newly created dataset of 1,412,272 English news articles from the Common Crawl over 2018-2024. Expert consensus ratings on 579 source websites were split at the median, creating perceived low and high-quality classes of about 706,000 articles each, with 194 linguistic features per website-level labelled article. Traditional machine learning classifiers such as the Random Forest demonstrated capable performance (0.7355 accuracy, 0.8131 ROC AUC). For deep learning, ModernBERT-large (256 context length) achieved the best performance (0.8744 accuracy; 0.9593 ROC-AUC; 0.8739 F1), followed by DistilBERT-base (512 context length) at 0.8685 accuracy and 0.9554 ROC-AUC. DistilBERT-base (256 context length) reached 0.8478 accuracy and 0.9407 ROC-AUC, while ModernBERT-base (256 context length) attained 0.8569 accuracy and 0.9470 ROC-AUC. These results suggest that the perceived quality of worldwide news articles can be effectively differentiated by traditional CPU-based machine learning classifiers and deep learning classifiers.
>
---
#### [new 010] TurkColBERT: A Benchmark of Dense and Late-Interaction Models for Turkish Information Retrieval
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文针对土耳其语信息检索中密集编码器与延迟交互模型的对比问题，构建了首个全面基准TurkColBERT。通过两阶段微调与转换，评估10个模型在多个领域数据集上的性能，发现小型延迟交互模型在精度和效率上显著优于密集编码器，并优化了索引算法以实现低延迟检索。**

- **链接: [https://arxiv.org/pdf/2511.16528v1](https://arxiv.org/pdf/2511.16528v1)**

> **作者:** Özay Ezerceli; Mahmoud El Hussieni; Selva Taş; Reyhan Bayraktar; Fatma Betül Terzioğlu; Yusuf Çelebi; Yağız Asker
>
> **摘要:** Neural information retrieval systems excel in high-resource languages but remain underexplored for morphologically rich, lower-resource languages such as Turkish. Dense bi-encoders currently dominate Turkish IR, yet late-interaction models -- which retain token-level representations for fine-grained matching -- have not been systematically evaluated. We introduce TurkColBERT, the first comprehensive benchmark comparing dense encoders and late-interaction models for Turkish retrieval. Our two-stage adaptation pipeline fine-tunes English and multilingual encoders on Turkish NLI/STS tasks, then converts them into ColBERT-style retrievers using PyLate trained on MS MARCO-TR. We evaluate 10 models across five Turkish BEIR datasets covering scientific, financial, and argumentative domains. Results show strong parameter efficiency: the 1.0M-parameter colbert-hash-nano-tr is 600$\times$ smaller than the 600M turkish-e5-large dense encoder while preserving over 71\% of its average mAP. Late-interaction models that are 3--5$\times$ smaller than dense encoders significantly outperform them; ColmmBERT-base-TR yields up to +13.8\% mAP on domain-specific tasks. For production-readiness, we compare indexing algorithms: MUVERA+Rerank is 3.33$\times$ faster than PLAID and offers +1.7\% relative mAP gain. This enables low-latency retrieval, with ColmmBERT-base-TR achieving 0.54 ms query times under MUVERA. We release all checkpoints, configs, and evaluation scripts. Limitations include reliance on moderately sized datasets ($\leq$50K documents) and translated benchmarks, which may not fully reflect real-world Turkish retrieval conditions; larger-scale MUVERA evaluations remain necessary.
>
---
#### [new 011] SDA: Steering-Driven Distribution Alignment for Open LLMs without Fine-Tuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型推理时与人类意图对齐的问题，提出无需微调的SDA框架。通过动态调整输出概率分布，实现高效、轻量化的对齐，提升模型在帮助性、诚实性和无害性上的表现，适用于多种开源模型且支持个性化控制。**

- **链接: [https://arxiv.org/pdf/2511.16324v1](https://arxiv.org/pdf/2511.16324v1)**

> **作者:** Wei Xia; Zhi-Hong Deng
>
> **摘要:** With the rapid advancement of large language models (LLMs), their deployment in real-world applications has become increasingly widespread. LLMs are expected to deliver robust performance across diverse tasks, user preferences, and practical scenarios. However, as demands grow, ensuring that LLMs produce responses aligned with human intent remains a foundational challenge. In particular, aligning model behavior effectively and efficiently during inference, without costly retraining or extensive supervision, is both a critical requirement and a non-trivial technical endeavor. To address the challenge, we propose SDA (Steering-Driven Distribution Alignment), a training-free and model-agnostic alignment framework designed for open-source LLMs. SDA dynamically redistributes model output probabilities based on user-defined alignment instructions, enhancing alignment between model behavior and human intents without fine-tuning. The method is lightweight, resource-efficient, and compatible with a wide range of open-source LLMs. It can function independently during inference or be integrated with training-based alignment strategies. Moreover, SDA supports personalized preference alignment, enabling flexible control over the model response behavior. Empirical results demonstrate that SDA consistently improves alignment performance across 8 open-source LLMs with varying scales and diverse origins, evaluated on three key alignment dimensions, helpfulness, harmlessness, and honesty (3H). Specifically, SDA achieves average gains of 64.4% in helpfulness, 30% in honesty and 11.5% in harmlessness across the tested models, indicating its effectiveness and generalization across diverse models and application scenarios.
>
---
#### [new 012] Integrating Symbolic Natural Language Understanding and Language Models for Word Sense Disambiguation
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对词汇消歧任务，解决现有方法依赖人工标注数据、难以处理丰富语义表示的问题。提出利用语言模型作为无标注训练的决策者，将符号系统生成的候选词义转化为自然语言查询，通过LLM选择上下文合适含义，并反馈给符号系统，实现无需人工标注的精准消歧。**

- **链接: [https://arxiv.org/pdf/2511.16577v1](https://arxiv.org/pdf/2511.16577v1)**

> **作者:** Kexin Zhao; Ken Forbus
>
> **备注:** 16 pages
>
> **摘要:** Word sense disambiguation is a fundamental challenge in natural language understanding. Current methods are primarily aimed at coarse-grained representations (e.g. WordNet synsets or FrameNet frames) and require hand-annotated training data to construct. This makes it difficult to automatically disambiguate richer representations (e.g. built on OpenCyc) that are needed for sophisticated inference. We propose a method that uses statistical language models as oracles for disambiguation that does not require any hand-annotation of training data. Instead, the multiple candidate meanings generated by a symbolic NLU system are converted into distinguishable natural language alternatives, which are used to query an LLM to select appropriate interpretations given the linguistic context. The selected meanings are propagated back to the symbolic NLU system. We evaluate our method against human-annotated gold answers to demonstrate its effectiveness.
>
---
#### [new 013] ELPO: Ensemble Learning Based Prompt Optimization for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ELPO框架，针对大语言模型提示工程中手动设计耗时、效率低的问题，通过集成学习思想融合多种生成与搜索策略，实现更高效准确的自动提示优化。实验表明其在多个任务上显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.16122v1](https://arxiv.org/pdf/2511.16122v1)**

> **作者:** Qing Zhang; Bing Xu; Xudong Zhang; Yifan Shi; Yang Li; Chen Zhang; Yik Chung Wu; Ngai Wong; Yijie Chen; Hong Dai; Xiansen Chen; Mian Zhang
>
> **摘要:** The remarkable performance of Large Language Models (LLMs) highly relies on crafted prompts. However, manual prompt engineering is a laborious process, creating a core bottleneck for practical application of LLMs. This phenomenon has led to the emergence of a new research area known as Automatic Prompt Optimization (APO), which develops rapidly in recent years. Existing APO methods such as those based on evolutionary algorithms or trial-and-error approaches realize an efficient and accurate prompt optimization to some extent. However, those researches focus on a single model or algorithm for the generation strategy and optimization process, which limits their performance when handling complex tasks. To address this, we propose a novel framework called Ensemble Learning based Prompt Optimization (ELPO) to achieve more accurate and robust results. Motivated by the idea of ensemble learning, ELPO conducts voting mechanism and introduces shared generation strategies along with different search methods for searching superior prompts. Moreover, ELPO creatively presents more efficient algorithms for the prompt generation and search process. Experimental results demonstrate that ELPO outperforms state-of-the-art prompt optimization methods across different tasks, e.g., improving F1 score by 7.6 on ArSarcasm dataset.
>
---
#### [new 014] SeSE: A Structural Information-Guided Uncertainty Quantification Framework for Hallucination Detection in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型（LLM）幻觉检测中的不确定性量化问题，提出基于结构信息的SeSE框架。通过构建稀疏有向语义图并计算最优语义编码树的结构熵，精准捕捉语义不确定性，提升长文本生成中细粒度幻觉检测能力，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.16275v1](https://arxiv.org/pdf/2511.16275v1)**

> **作者:** Xingtao Zhao; Hao Peng; Dingli Su; Xianghua Zeng; Chunyang Liu; Jinzhi Liao; Philip S. Yu
>
> **备注:** 14 pages of main text and 10 pages of appendices
>
> **摘要:** Reliable uncertainty quantification (UQ) is essential for deploying large language models (LLMs) in safety-critical scenarios, as it enables them to abstain from responding when uncertain, thereby avoiding hallucinating falsehoods. However, state-of-the-art UQ methods primarily rely on semantic probability distributions or pairwise distances, overlooking latent semantic structural information that could enable more precise uncertainty estimates. This paper presents Semantic Structural Entropy (SeSE), a principled UQ framework that quantifies the inherent semantic uncertainty of LLMs from a structural information perspective for hallucination detection. Specifically, to effectively model semantic spaces, we first develop an adaptively sparsified directed semantic graph construction algorithm that captures directional semantic dependencies while automatically pruning unnecessary connections that introduce negative interference. We then exploit latent semantic structural information through hierarchical abstraction: SeSE is defined as the structural entropy of the optimal semantic encoding tree, formalizing intrinsic uncertainty within semantic spaces after optimal compression. A higher SeSE value corresponds to greater uncertainty, indicating that LLMs are highly likely to generate hallucinations. In addition, to enhance fine-grained UQ in long-form generation -- where existing methods often rely on heuristic sample-and-count techniques -- we extend SeSE to quantify the uncertainty of individual claims by modeling their random semantic interactions, providing theoretically explicable hallucination detection. Extensive experiments across 29 model-dataset combinations show that SeSE significantly outperforms advanced UQ baselines, including strong supervised methods and the recently proposed KLE.
>
---
#### [new 015] Learning from Sufficient Rationales: Analysing the Relationship Between Explanation Faithfulness and Token-level Regularisation Strategies
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究自然语言推理中解释的可信度与标记级正则化策略的关系。针对现有充分性指标无法揭示解释信息对模型性能影响的问题，通过分析标记分类与注意力正则化，发现高充分性解释未必提升分类准确率，且其与标记分类能力无关，揭示了解释机制的复杂性。**

- **链接: [https://arxiv.org/pdf/2511.16353v1](https://arxiv.org/pdf/2511.16353v1)**

> **作者:** Jonathan Kamp; Lisa Beinborn; Antske Fokkens
>
> **备注:** Long paper accepted to the main conference of AACL 2025. Please cite the conference proceedings when available
>
> **摘要:** Human explanations of natural language, rationales, form a tool to assess whether models learn a label for the right reasons or rely on dataset-specific shortcuts. Sufficiency is a common metric for estimating the informativeness of rationales, but it provides limited insight into the effects of rationale information on model performance. We address this limitation by relating sufficiency to two modelling paradigms: the ability of models to identify which tokens are part of the rationale (through token classification) and the ability of improving model performance by incorporating rationales in the input (through attention regularisation). We find that highly informative rationales are not likely to help classify the instance correctly. Sufficiency conversely captures the classification impact of the non-rationalised context, which interferes with rationale information in the same input. We also find that incorporating rationale information in model inputs can boost cross-domain classification, but results are inconsistent per task and model type. Finally, sufficiency and token classification appear to be unrelated. These results exemplify the complexity of rationales, showing that metrics capable of systematically capturing this type of information merit further investigation.
>
---
#### [new 016] Early science acceleration experiments with GPT-5
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于人工智能辅助科研任务，旨在探索GPT-5在科学研究中的应用。通过多领域案例，展示其加速科研进程的能力，尤其在数学领域实现四项经验证的新成果，揭示了前沿AI与人类协作的潜力与局限。**

- **链接: [https://arxiv.org/pdf/2511.16072v1](https://arxiv.org/pdf/2511.16072v1)**

> **作者:** Sébastien Bubeck; Christian Coester; Ronen Eldan; Timothy Gowers; Yin Tat Lee; Alexandru Lupsasca; Mehtaab Sawhney; Robert Scherrer; Mark Sellke; Brian K. Spears; Derya Unutmaz; Kevin Weil; Steven Yin; Nikita Zhivotovskiy
>
> **备注:** 89 pages
>
> **摘要:** AI models like GPT-5 are an increasingly valuable tool for scientists, but many remain unaware of the capabilities of frontier AI. We present a collection of short case studies in which GPT-5 produced new, concrete steps in ongoing research across mathematics, physics, astronomy, computer science, biology, and materials science. In these examples, the authors highlight how AI accelerated their work, and where it fell short; where expert time was saved, and where human input was still key. We document the interactions of the human authors with GPT-5, as guiding examples of fruitful collaboration with AI. Of note, this paper includes four new results in mathematics (carefully verified by the human authors), underscoring how GPT-5 can help human mathematicians settle previously unsolved problems. These contributions are modest in scope but profound in implication, given the rate at which frontier AI is progressing.
>
---
#### [new 017] Mind the Motions: Benchmarking Theory-of-Mind in Everyday Body Language
- **分类: cs.CL**

- **简介: 该论文聚焦于机器理解日常身体语言的理论心理能力，旨在弥补现有基准在非言语线索（NVC）和复杂心理状态评估上的不足。研究构建了Motion2Mind框架，包含精细标注的视频数据集与心理解释，揭示当前AI在检测与解释NVC上表现不佳，存在显著性能差距与过度解读问题。**

- **链接: [https://arxiv.org/pdf/2511.15887v1](https://arxiv.org/pdf/2511.15887v1)**

> **作者:** Seungbeen Lee; Jinhong Jeong; Donghyun Kim; Yejin Son; Youngjae Yu
>
> **摘要:** Our ability to interpret others' mental states through nonverbal cues (NVCs) is fundamental to our survival and social cohesion. While existing Theory of Mind (ToM) benchmarks have primarily focused on false-belief tasks and reasoning with asymmetric information, they overlook other mental states beyond belief and the rich tapestry of human nonverbal communication. We present Motion2Mind, a framework for evaluating the ToM capabilities of machines in interpreting NVCs. Leveraging an expert-curated body-language reference as a proxy knowledge base, we build Motion2Mind, a carefully curated video dataset with fine-grained nonverbal cue annotations paired with manually verified psychological interpretations. It encompasses 222 types of nonverbal cues and 397 mind states. Our evaluation reveals that current AI systems struggle significantly with NVC interpretation, exhibiting not only a substantial performance gap in Detection, as well as patterns of over-interpretation in Explanation compared to human annotators.
>
---
#### [new 018] Beyond Tokens in Language Models: Interpreting Activations through Text Genre Chunks
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型可解释性任务，旨在通过分析语言模型激活值预测输入文本的文体类型。研究使用Mistral-7B与两个数据集，发现仅用浅层学习模型即可高精度（最高F1=98%）预测文体，验证了激活值蕴含文体信息，为理解大模型内部机制提供新路径。**

- **链接: [https://arxiv.org/pdf/2511.16540v1](https://arxiv.org/pdf/2511.16540v1)**

> **作者:** Éloïse Benito-Rodriguez; Einar Urdshals; Jasmina Nasufi; Nicky Pochinkov
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Understanding Large Language Models (LLMs) is key to ensure their safe and beneficial deployment. This task is complicated by the difficulty of interpretability of LLM structures, and the inability to have all their outputs human-evaluated. In this paper, we present the first step towards a predictive framework, where the genre of a text used to prompt an LLM, is predicted based on its activations. Using Mistral-7B and two datasets, we show that genre can be extracted with F1-scores of up to 98% and 71% using scikit-learn classifiers. Across both datasets, results consistently outperform the control task, providing a proof of concept that text genres can be inferred from LLMs with shallow learning models.
>
---
#### [new 019] Learning Tractable Distributions Of Language Model Continuations
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对可控语言生成中未来约束导致自回归模型难以直接建模的问题，提出LTLA方法。通过结合语言模型的上下文编码与固定可计算的隐马尔可夫模型，实现高效、高精度的续写分布估计，在保持低推理开销的同时提升约束满足率和生成质量。**

- **链接: [https://arxiv.org/pdf/2511.16054v1](https://arxiv.org/pdf/2511.16054v1)**

> **作者:** Gwen Yidou-Weng; Ian Li; Anji Liu; Oliver Broadrick; Guy Van den Broeck; Benjie Wang
>
> **摘要:** Controlled language generation conditions text on sequence-level constraints (for example, syntax, style, or safety). These constraints may depend on future tokens, which makes directly conditioning an autoregressive language model (LM) generally intractable. Prior work uses tractable surrogates such as hidden Markov models (HMMs) to approximate the distribution over continuations and adjust the model's next-token logits at decoding time. However, we find that these surrogates are often weakly context aware, which reduces query quality. We propose Learning to Look Ahead (LTLA), a hybrid approach that pairs the same base language model for rich prefix encoding with a fixed tractable surrogate model that computes exact continuation probabilities. Two efficiency pitfalls arise when adding neural context: (i) naively rescoring the prefix with every candidate next token requires a sweep over the entire vocabulary at each step, and (ii) predicting fresh surrogate parameters for each prefix, although tractable at a single step, forces recomputation of future probabilities for every new prefix and eliminates reuse. LTLA avoids both by using a single batched HMM update to account for all next-token candidates at once, and by conditioning only the surrogate's latent state prior on the LM's hidden representations while keeping the surrogate decoder fixed, so computations can be reused across prefixes. Empirically, LTLA attains higher conditional likelihood than an unconditional HMM, approximates continuation distributions for vision-language models where a standalone HMM cannot encode visual context, and improves constraint satisfaction at comparable fluency on controlled-generation tasks, with minimal inference overhead.
>
---
#### [new 020] What Really Counts? Examining Step and Token Level Attribution in Multilingual CoT Reasoning
- **分类: cs.CL**

- **简介: 该论文研究多语言大模型链式思维（CoT）推理的可解释性问题，旨在评估不同语言下推理过程的忠实度与透明度。通过结合步骤级和词元级归因方法，分析Qwen2.5模型在MGSM基准上的表现，发现其推理链过度依赖最终步骤，且在低资源语言中效果有限，揭示了CoT提示在多语言场景下的局限性。**

- **链接: [https://arxiv.org/pdf/2511.15886v1](https://arxiv.org/pdf/2511.15886v1)**

> **作者:** Jeremias Ferrao; Ezgi Basar; Khondoker Ittehadul Islam; Mahrokh Hassani
>
> **备注:** Received the Best Student Project Award at RuG's Advanced-NLP course
>
> **摘要:** This study investigates the attribution patterns underlying Chain-of-Thought (CoT) reasoning in multilingual LLMs. While prior works demonstrate the role of CoT prompting in improving task performance, there are concerns regarding the faithfulness and interpretability of the generated reasoning chains. To assess these properties across languages, we applied two complementary attribution methods--ContextCite for step-level attribution and Inseq for token-level attribution--to the Qwen2.5 1.5B-Instruct model using the MGSM benchmark. Our experimental results highlight key findings such as: (1) attribution scores excessively emphasize the final reasoning step, particularly in incorrect generations; (2) structured CoT prompting significantly improves accuracy primarily for high-resource Latin-script languages; and (3) controlled perturbations via negation and distractor sentences reduce model accuracy and attribution coherence. These findings highlight the limitations of CoT prompting, particularly in terms of multilingual robustness and interpretive transparency.
>
---
#### [new 021] WER is Unaware: Assessing How ASR Errors Distort Clinical Understanding in Patient Facing Dialogue
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对临床对话中自动语音识别（ASR）评估依赖错误率（WER）的问题，提出以临床影响为核心评价标准。通过专家标注构建基准，发现现有指标与临床风险关联性差。研究引入基于GEPA优化的LLM-as-a-Judge（Gemini-2.5-Pro），实现与专家相当的自动化临床影响评估，推动ASR评价从文本准确转向医疗安全考量。**

- **链接: [https://arxiv.org/pdf/2511.16544v1](https://arxiv.org/pdf/2511.16544v1)**

> **作者:** Zachary Ellis; Jared Joselowitz; Yash Deo; Yajie He; Anna Kalygina; Aisling Higham; Mana Rahimzadeh; Yan Jia; Ibrahim Habli; Ernest Lim
>
> **摘要:** As Automatic Speech Recognition (ASR) is increasingly deployed in clinical dialogue, standard evaluations still rely heavily on Word Error Rate (WER). This paper challenges that standard, investigating whether WER or other common metrics correlate with the clinical impact of transcription errors. We establish a gold-standard benchmark by having expert clinicians compare ground-truth utterances to their ASR-generated counterparts, labeling the clinical impact of any discrepancies found in two distinct doctor-patient dialogue datasets. Our analysis reveals that WER and a comprehensive suite of existing metrics correlate poorly with the clinician-assigned risk labels (No, Minimal, or Significant Impact). To bridge this evaluation gap, we introduce an LLM-as-a-Judge, programmatically optimized using GEPA to replicate expert clinical assessment. The optimized judge (Gemini-2.5-Pro) achieves human-comparable performance, obtaining 90% accuracy and a strong Cohen's $κ$ of 0.816. This work provides a validated, automated framework for moving ASR evaluation beyond simple textual fidelity to a necessary, scalable assessment of safety in clinical dialogue.
>
---
#### [new 022] Anatomy of an Idiom: Tracing Non-Compositionality in Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究预训练语言模型对习语的非组合性处理，属于自然语言理解任务。针对模型如何处理语义不等于字面意义的习语这一问题，提出新电路发现方法，揭示“习语头”和“增强接收”机制，阐明模型在效率与鲁棒性间的平衡策略，为理解复杂语法结构提供洞见。**

- **链接: [https://arxiv.org/pdf/2511.16467v1](https://arxiv.org/pdf/2511.16467v1)**

> **作者:** Andrew Gomes
>
> **摘要:** We investigate the processing of idiomatic expressions in transformer-based language models using a novel set of techniques for circuit discovery and analysis. First discovering circuits via a modified path patching algorithm, we find that idiom processing exhibits distinct computational patterns. We identify and investigate ``Idiom Heads,'' attention heads that frequently activate across different idioms, as well as enhanced attention between idiom tokens due to earlier processing, which we term ``augmented reception.'' We analyze these phenomena and the general features of the discovered circuits as mechanisms by which transformers balance computational efficiency and robustness. Finally, these findings provide insights into how transformers handle non-compositional language and suggest pathways for understanding the processing of more complex grammatical constructions.
>
---
#### [new 023] SemanticCite: Citation Verification with AI-Powered Full-Text Analysis and Evidence-Based Reasoning
- **分类: cs.CL; cs.DL**

- **简介: 该论文提出SemanticCite，一个基于AI的引文验证系统，旨在解决引文误用、AI幻觉引用及传统引文缺乏上下文的问题。通过全文本分析与证据推理，实现引文准确性的自动验证，并提供透明解释。研究构建了跨八学科的大规模标注数据集，开发轻量级模型，实现高效、可扩展的引文质量控制。**

- **链接: [https://arxiv.org/pdf/2511.16198v1](https://arxiv.org/pdf/2511.16198v1)**

> **作者:** Sebastian Haan
>
> **备注:** 21 pages, 4 figures
>
> **摘要:** Effective scientific communication depends on accurate citations that validate sources and guide readers to supporting evidence. Yet academic literature faces mounting challenges: semantic citation errors that misrepresent sources, AI-generated hallucinated references, and traditional citation formats that point to entire papers without indicating which sections substantiate specific claims. We introduce SemanticCite, an AI-powered system that verifies citation accuracy through full-text source analysis while providing rich contextual information via detailed reasoning and relevant text snippets. Our approach combines multiple retrieval methods with a four-class classification system (Supported, Partially Supported, Unsupported, Uncertain) that captures nuanced claim-source relationships and enables appropriate remedial actions for different error types. Our experiments show that fine-tuned lightweight language models achieve performance comparable to large commercial systems with significantly lower computational requirements, making large-scale citation verification practically feasible. The system provides transparent, evidence-based explanations that support user understanding and trust. We contribute a comprehensive dataset of over 1,000 citations with detailed alignments, functional classifications, semantic annotations, and bibliometric metadata across eight disciplines, alongside fine-tuned models and the complete verification framework as open-source software. SemanticCite addresses critical challenges in research integrity through scalable citation verification, streamlined peer review, and quality control for AI-generated content, providing an open-source foundation for maintaining citation accuracy at scale.
>
---
#### [new 024] Nemotron Elastic: Towards Efficient Many-in-One Reasoning LLMs
- **分类: cs.CL**

- **简介: 该论文提出Nemotron Elastic框架，解决多规模推理型大模型训练成本高的问题。通过在单个模型中嵌套多个共享权重的子模型，实现零样本提取与多预算优化，显著降低训练成本，提升部署效率。**

- **链接: [https://arxiv.org/pdf/2511.16664v1](https://arxiv.org/pdf/2511.16664v1)**

> **作者:** Ali Taghibakhshi; Sharath Turuvekere Sreenivas; Saurav Muralidharan; Ruisi Cai; Marcin Chochowski; Ameya Sunil Mahabaleshwarkar; Yoshi Suhara; Oluwatobi Olabiyi; Daniel Korzekwa; Mostofa Patwary; Mohammad Shoeybi; Jan Kautz; Bryan Catanzaro; Ashwath Aithal; Nima Tajbakhsh; Pavlo Molchanov
>
> **摘要:** Training a family of large language models targeting multiple scales and deployment objectives is prohibitively expensive, requiring separate training runs for each different size. Recent work on model compression through pruning and knowledge distillation has reduced this cost; however, this process still incurs hundreds of billions of tokens worth of training cost per compressed model. In this paper, we present Nemotron Elastic, a framework for building reasoning-oriented LLMs, including hybrid Mamba-Attention architectures, that embed multiple nested submodels within a single parent model, each optimized for different deployment configurations and budgets. Each of these submodels shares weights with the parent model and can be extracted zero-shot during deployment without additional training or fine-tuning. We enable this functionality through an end-to-end trained router, tightly coupled to a two-stage training curriculum designed specifically for reasoning models. We additionally introduce group-aware SSM elastification that preserves Mamba's structural constraints, heterogeneous MLP elastification, normalized MSE-based layer importance for improved depth selection, and knowledge distillation enabling simultaneous multi-budget optimization. We apply Nemotron Elastic to the Nemotron Nano V2 12B model, simultaneously producing a 9B and a 6B model using only 110B training tokens; this results in over 360x cost reduction compared to training model families from scratch, and around 7x compared to SoTA compression techniques. Each of the nested models performs on par or better than the SoTA in accuracy. Moreover, unlike other compression methods, the nested capability of our approach allows having a many-in-one reasoning model that has constant deployment memory against the number of models in the family.
>
---
#### [new 025] Incorporating Self-Rewriting into Large Language Model Reasoning Reinforcement
- **分类: cs.CL**

- **简介: 该论文针对大语言模型推理中因仅依赖最终正确性奖励导致的内部思维质量差问题，提出自重写框架。通过选择性重写一致正确的简单样本，模型自我优化推理过程，提升准确性与思维质量，实现更高效、更优的推理表现。**

- **链接: [https://arxiv.org/pdf/2511.16331v1](https://arxiv.org/pdf/2511.16331v1)**

> **作者:** Jiashu Yao; Heyan Huang; Shuang Zeng; Chuwei Luo; WangJie You; Jie Tang; Qingsong Liu; Yuhang Guo; Yangyang Kang
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Through reinforcement learning (RL) with outcome correctness rewards, large reasoning models (LRMs) with scaled inference computation have demonstrated substantial success on complex reasoning tasks. However, the one-sided reward, focused solely on final correctness, limits its ability to provide detailed supervision over internal reasoning process. This deficiency leads to suboptimal internal reasoning quality, manifesting as issues like over-thinking, under-thinking, redundant-thinking, and disordered-thinking. Inspired by the recent progress in LRM self-rewarding, we introduce self-rewriting framework, where a model rewrites its own reasoning texts, and subsequently learns from the rewritten reasoning to improve the internal thought process quality. For algorithm design, we propose a selective rewriting approach wherein only "simple" samples, defined by the model's consistent correctness, are rewritten, thereby preserving all original reward signals of GRPO. For practical implementation, we compile rewriting and vanilla generation within one single batch, maintaining the scalability of the RL algorithm and introducing only ~10% overhead. Extensive experiments on diverse tasks with different model sizes validate the effectiveness of self-rewriting. In terms of the accuracy-length tradeoff, the self-rewriting approach achieves improved accuracy (+0.6) with substantially shorter reasoning (-46%) even without explicit instructions in rewriting prompts to reduce reasoning length, outperforming existing strong baselines. In terms of internal reasoning quality, self-rewriting achieves significantly higher scores (+7.2) under the LLM-as-a-judge metric, successfully mitigating internal reasoning flaws.
>
---
#### [new 026] MiMo-Embodied: X-Embodied Foundation Model Technical Report
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出MiMo-Embodied，首个跨具身基础模型，融合自动驾驶与具身智能任务。通过多阶段学习与数据优化，在17项具身AI与12项自动驾驶基准上均达领先性能，验证两领域间正向迁移与协同增强。**

- **链接: [https://arxiv.org/pdf/2511.16518v1](https://arxiv.org/pdf/2511.16518v1)**

> **作者:** Xiaoshuai Hao; Lei Zhou; Zhijian Huang; Zhiwen Hou; Yingbo Tang; Lingfeng Zhang; Guang Li; Zheng Lu; Shuhuai Ren; Xianhui Meng; Yuchen Zhang; Jing Wu; Jinghui Lu; Chenxu Dang; Jiayi Guan; Jianhua Wu; Zhiyi Hou; Hanbing Li; Shumeng Xia; Mingliang Zhou; Yinan Zheng; Zihao Yue; Shuhao Gu; Hao Tian; Yuannan Shen; Jianwei Cui; Wen Zhang; Shaoqing Xu; Bing Wang; Haiyang Sun; Zeyu Zhu; Yuncheng Jiang; Zibin Guo; Chuhong Gong; Chaofan Zhang; Wenbo Ding; Kun Ma; Guang Chen; Rui Cai; Diyun Xiang; Heng Qu; Fuli Luo; Hangjun Ye; Long Chen
>
> **备注:** Code: https://github.com/XiaomiMiMo/MiMo-Embodied Model: https://huggingface.co/XiaomiMiMo/MiMo-Embodied-7B
>
> **摘要:** We open-source MiMo-Embodied, the first cross-embodied foundation model to successfully integrate and achieve state-of-the-art performance in both Autonomous Driving and Embodied AI. MiMo-Embodied sets new records across 17 embodied AI benchmarks in Task Planning, Affordance Prediction and Spatial Understanding, while also excelling in 12 autonomous driving benchmarks across Environmental Perception, Status Prediction, and Driving Planning. Across these tasks, MiMo-Embodied significantly outperforms existing open-source, closed-source, and specialized baselines. Our results indicate that through multi-stage learning, curated data construction, and CoT/RL fine-tuning, these two domains exhibit strong positive transfer and mutually reinforce one another. We provide a detailed analysis of our model design and training methodologies to facilitate further research. Code and models are available at https://github.com/XiaomiMiMo/MiMo-Embodied.
>
---
#### [new 027] PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization
- **分类: cs.CR; cs.CL**

- **简介: 该论文针对大模型系统提示词易遭窃取的安全问题，提出基于LLM引导的黑盒优化框架PSM。通过添加保护性文本（SHIELD），在最小化泄露风险的同时保持任务性能，实现轻量级、实用化的提示防护。**

- **链接: [https://arxiv.org/pdf/2511.16209v1](https://arxiv.org/pdf/2511.16209v1)**

> **作者:** Huseein Jawad; Nicolas Brunel
>
> **摘要:** System prompts are critical for guiding the behavior of Large Language Models (LLMs), yet they often contain proprietary logic or sensitive information, making them a prime target for extraction attacks. Adversarial queries can successfully elicit these hidden instructions, posing significant security and privacy risks. Existing defense mechanisms frequently rely on heuristics, incur substantial computational overhead, or are inapplicable to models accessed via black-box APIs. This paper introduces a novel framework for hardening system prompts through shield appending, a lightweight approach that adds a protective textual layer to the original prompt. Our core contribution is the formalization of prompt hardening as a utility-constrained optimization problem. We leverage an LLM-as-optimizer to search the space of possible SHIELDs, seeking to minimize a leakage metric derived from a suite of adversarial attacks, while simultaneously preserving task utility above a specified threshold, measured by semantic fidelity to baseline outputs. This black-box, optimization-driven methodology is lightweight and practical, requiring only API access to the target and optimizer LLMs. We demonstrate empirically that our optimized SHIELDs significantly reduce prompt leakage against a comprehensive set of extraction attacks, outperforming established baseline defenses without compromising the model's intended functionality. Our work presents a paradigm for developing robust, utility-aware defenses in the escalating landscape of LLM security. The code is made public on the following link: https://github.com/psm-defense/psm
>
---
#### [new 028] Thinking-while-Generating: Interleaving Textual Reasoning throughout Visual Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出Thinking-while-Generating（TwiG）框架，解决视觉生成中推理与生成分离的问题。通过在生成过程中动态交织文本推理，实现对局部区域的引导与已有内容的反思，提升输出的语义丰富性与上下文一致性。研究对比了零样本提示、微调和强化学习三种策略，验证了交错推理的有效性。**

- **链接: [https://arxiv.org/pdf/2511.16671v1](https://arxiv.org/pdf/2511.16671v1)**

> **作者:** Ziyu Guo; Renrui Zhang; Hongyu Li; Manyuan Zhang; Xinyan Chen; Sifan Wang; Yan Feng; Peng Pei; Pheng-Ann Heng
>
> **备注:** Project Page: https://think-while-gen.github.io Code: https://github.com/ZiyuGuo99/Thinking-while-Generating
>
> **摘要:** Recent advances in visual generation have increasingly explored the integration of reasoning capabilities. They incorporate textual reasoning, i.e., think, either before (as pre-planning) or after (as post-refinement) the generation process, yet they lack on-the-fly multimodal interaction during the generation itself. In this preliminary study, we introduce Thinking-while-Generating (TwiG), the first interleaved framework that enables co-evolving textual reasoning throughout the visual generation process. As visual content is progressively generating, textual reasoning is interleaved to both guide upcoming local regions and reflect on previously synthesized ones. This dynamic interplay produces more context-aware and semantically rich visual outputs. To unveil the potential of this framework, we investigate three candidate strategies, zero-shot prompting, supervised fine-tuning (SFT) on our curated TwiG-50K dataset, and reinforcement learning (RL) via a customized TwiG-GRPO strategy, each offering unique insights into the dynamics of interleaved reasoning. We hope this work inspires further research into interleaving textual reasoning for enhanced visual generation. Code will be released at: https://github.com/ZiyuGuo99/Thinking-while-Generating.
>
---
#### [new 029] Can MLLMs Read the Room? A Multimodal Benchmark for Assessing Deception in Multi-Party Social Interactions
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对多模态大模型在社交互动中识别欺骗能力不足的问题，提出MIDA任务与多模态数据集，构建基准评估12个模型。发现主流模型难以基于多模态线索判断真伪。提出SoCoT与DSEM框架，提升模型对社会认知的推理能力，推动更智能的社交理解AI发展。**

- **链接: [https://arxiv.org/pdf/2511.16221v1](https://arxiv.org/pdf/2511.16221v1)**

> **作者:** Caixin Kang; Yifei Huang; Liangyang Ouyang; Mingfang Zhang; Ruicong Liu; Yoichi Sato
>
> **摘要:** Despite their advanced reasoning capabilities, state-of-the-art Multimodal Large Language Models (MLLMs) demonstrably lack a core component of human intelligence: the ability to `read the room' and assess deception in complex social interactions. To rigorously quantify this failure, we introduce a new task, Multimodal Interactive Deception Assessment (MIDA), and present a novel multimodal dataset providing synchronized video and text with verifiable ground-truth labels for every statement. We establish a comprehensive benchmark evaluating 12 state-of-the-art open- and closed-source MLLMs, revealing a significant performance gap: even powerful models like GPT-4o struggle to distinguish truth from falsehood reliably. Our analysis of failure modes indicates that these models fail to effectively ground language in multimodal social cues and lack the ability to model what others know, believe, or intend, highlighting the urgent need for novel approaches to building more perceptive and trustworthy AI systems. To take a step forward, we design a Social Chain-of-Thought (SoCoT) reasoning pipeline and a Dynamic Social Epistemic Memory (DSEM) module. Our framework yields performance improvement on this challenging task, demonstrating a promising new path toward building MLLMs capable of genuine human-like social reasoning.
>
---
#### [new 030] JudgeBoard: Benchmarking and Enhancing Small Language Models for Reasoning Evaluation
- **分类: cs.AI; cs.CL**

- **简介: 该论文聚焦小语言模型（SLMs）在推理评估中的判断能力，旨在解决其相比大模型（LLMs）判断准确性不足的问题。提出JudgeBoard评估框架，实现无需对比的直接判断；引入MAJ多智能体协作机制，显著提升SLMs判断性能，使其在数学推理等任务上可媲美甚至超越部分大模型。**

- **链接: [https://arxiv.org/pdf/2511.15958v1](https://arxiv.org/pdf/2511.15958v1)**

> **作者:** Zhenyu Bi; Gaurav Srivastava; Yang Li; Meng Lu; Swastik Roy; Morteza Ziyadi; Xuan Wang
>
> **备注:** 23 pages, 4 figures
>
> **摘要:** While small language models (SLMs) have shown promise on various reasoning tasks, their ability to judge the correctness of answers remains unclear compared to large language models (LLMs). Prior work on LLM-as-a-judge frameworks typically relies on comparing candidate answers against ground-truth labels or other candidate answers using predefined metrics like entailment. However, this approach is inherently indirect and difficult to fully automate, offering limited support for fine-grained and scalable evaluation of reasoning outputs. In this work, we propose JudgeBoard, a novel evaluation pipeline that directly queries models to assess the correctness of candidate answers without requiring extra answer comparisons. We focus on two core reasoning domains: mathematical reasoning and science/commonsense reasoning, and construct task-specific evaluation leaderboards using both accuracy-based ranking and an Elo-based rating system across five benchmark datasets, enabling consistent model comparison as judges rather than comparators. To improve judgment performance in lightweight models, we propose MAJ (Multi-Agent Judging), a novel multi-agent evaluation framework that leverages multiple interacting SLMs with distinct reasoning profiles to approximate LLM-level judgment accuracy through collaborative deliberation. Experimental results reveal a significant performance gap between SLMs and LLMs in isolated judging tasks. However, our MAJ framework substantially improves the reliability and consistency of SLMs. On the MATH dataset, MAJ using smaller-sized models as backbones performs comparatively well or even better than their larger-sized counterparts. Our findings highlight that multi-agent SLM systems can potentially match or exceed LLM performance in judgment tasks, with implications for scalable and efficient assessment.
>
---
#### [new 031] D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对GUI智能体在真实异常环境下的鲁棒性不足问题，提出D-GARA动态评测框架。通过构建含多种现实异常（如弹窗、提醒）的Android应用基准集，揭示现有先进模型在异常场景下性能显著下降，推动鲁棒性学习研究。**

- **链接: [https://arxiv.org/pdf/2511.16590v1](https://arxiv.org/pdf/2511.16590v1)**

> **作者:** Sen Chen; Tong Zhao; Yi Bin; Fei Ma; Wenqi Shao; Zheng Wang
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Developing intelligent agents capable of operating a wide range of Graphical User Interfaces (GUIs) with human-level proficiency is a key milestone on the path toward Artificial General Intelligence. While most existing datasets and benchmarks for training and evaluating GUI agents are static and idealized, failing to reflect the complexity and unpredictability of real-world environments, particularly the presence of anomalies. To bridge this research gap, we propose D-GARA, a dynamic benchmarking framework, to evaluate Android GUI agent robustness in real-world anomalies. D-GARA introduces a diverse set of real-world anomalies that GUI agents commonly face in practice, including interruptions such as permission dialogs, battery warnings, and update prompts. Based on D-GARA framework, we construct and annotate a benchmark featuring commonly used Android applications with embedded anomalies to support broader community research. Comprehensive experiments and results demonstrate substantial performance degradation in state-of-the-art GUI agents when exposed to anomaly-rich environments, highlighting the need for robustness-aware learning. D-GARA is modular and extensible, supporting the seamless integration of new tasks, anomaly types, and interaction scenarios to meet specific evaluation goals.
>
---
#### [new 032] Music Recommendation with Large Language Models: Challenges, Opportunities, and Evaluation
- **分类: cs.IR; cs.CL**

- **简介: 该论文研究大语言模型（LLM）在音乐推荐系统中的应用。针对传统推荐系统评价体系在LLM场景下失效的问题，提出重构评估框架。工作包括分析LLM对用户/物品建模与自然语言推荐的影响，借鉴NLP评估方法，构建涵盖成功与风险维度的系统性评估体系，推动MRS向更智能、可解释的方向发展。**

- **链接: [https://arxiv.org/pdf/2511.16478v1](https://arxiv.org/pdf/2511.16478v1)**

> **作者:** Elena V. Epure; Yashar Deldjoo; Bruno Sguerra; Markus Schedl; Manuel Moussallam
>
> **备注:** Under review with the ACM Transactions on Recommender Systems (TORS)
>
> **摘要:** Music Recommender Systems (MRS) have long relied on an information-retrieval framing, where progress is measured mainly through accuracy on retrieval-oriented subtasks. While effective, this reductionist paradigm struggles to address the deeper question of what makes a good recommendation, and attempts to broaden evaluation, through user studies or fairness analyses, have had limited impact. The emergence of Large Language Models (LLMs) disrupts this framework: LLMs are generative rather than ranking-based, making standard accuracy metrics questionable. They also introduce challenges such as hallucinations, knowledge cutoffs, non-determinism, and opaque training data, rendering traditional train/test protocols difficult to interpret. At the same time, LLMs create new opportunities, enabling natural-language interaction and even allowing models to act as evaluators. This work argues that the shift toward LLM-driven MRS requires rethinking evaluation. We first review how LLMs reshape user modeling, item modeling, and natural-language recommendation in music. We then examine evaluation practices from NLP, highlighting methodologies and open challenges relevant to MRS. Finally, we synthesize insights-focusing on how LLM prompting applies to MRS, to outline a structured set of success and risk dimensions. Our goal is to provide the MRS community with an updated, pedagogical, and cross-disciplinary perspective on evaluation.
>
---
#### [new 033] OpenMMReasoner: Pushing the Frontiers for Multimodal Reasoning with an Open and General Recipe
- **分类: cs.AI; cs.CL**

- **简介: 该论文聚焦多模态推理任务，针对现有方法数据与训练流程不透明、可复现性差的问题，提出OpenMMReasoner开源两阶段训练框架（SFT+RL）。构建高质量冷启动数据集，通过严谨设计提升模型推理能力，在9个基准上超越基线11.6%，推动多模态推理研究的可复现性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2511.16334v1](https://arxiv.org/pdf/2511.16334v1)**

> **作者:** Kaichen Zhang; Keming Wu; Zuhao Yang; Kairui Hu; Bin Wang; Ziwei Liu; Xingxuan Li; Lidong Bing
>
> **摘要:** Recent advancements in large reasoning models have fueled growing interest in extending such capabilities to multimodal domains. However, despite notable progress in visual reasoning, the lack of transparent and reproducible data curation and training strategies remains a major barrier to scalable research. In this work, we introduce OpenMMReasoner, a fully transparent two-stage recipe for multimodal reasoning spanning supervised fine-tuning (SFT) and reinforcement learning (RL). In the SFT stage, we construct an 874K-sample cold-start dataset with rigorous step-by-step validation, providing a strong foundation for reasoning capabilities. The subsequent RL stage leverages a 74K-sample dataset across diverse domains to further sharpen and stabilize these abilities, resulting in a more robust and efficient learning process. Extensive evaluations demonstrate that our training recipe not only surpasses strong baselines but also highlights the critical role of data quality and training design in shaping multimodal reasoning performance. Notably, our method achieves a 11.6% improvement over the Qwen2.5-VL-7B-Instruct baseline across nine multimodal reasoning benchmarks, establishing a solid empirical foundation for future large-scale multimodal reasoning research. We open-sourced all our codes, pipeline, and data at https://github.com/EvolvingLMMs-Lab/OpenMMReasoner.
>
---
#### [new 034] QueryGym: A Toolkit for Reproducible LLM-Based Query Reformulation
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出QueryGym，一个用于大语言模型（LLM）查询重写任务的开源工具包。针对现有方法缺乏统一实现、难以公平比较与复现的问题，该工具提供标准化API、可扩展架构及基准支持，促进高效实验与可靠部署。**

- **链接: [https://arxiv.org/pdf/2511.15996v1](https://arxiv.org/pdf/2511.15996v1)**

> **作者:** Amin Bigdeli; Radin Hamidi Rad; Mert Incesu; Negar Arabzadeh; Charles L. A. Clarke; Ebrahim Bagheri
>
> **备注:** 4 pages
>
> **摘要:** We present QueryGym, a lightweight, extensible Python toolkit that supports large language model (LLM)-based query reformulation. This is an important tool development since recent work on llm-based query reformulation has shown notable increase in retrieval effectiveness. However, while different authors have sporadically shared the implementation of their methods, there is no unified toolkit that provides a consistent implementation of such methods, which hinders fair comparison, rapid experimentation, consistent benchmarking and reliable deployment. QueryGym addresses this gap by providing a unified framework for implementing, executing, and comparing llm-based reformulation methods. The toolkit offers: (1) a Python API for applying diverse LLM-based methods, (2) a retrieval-agnostic interface supporting integration with backends such as Pyserini and PyTerrier, (3) a centralized prompt management system with versioning and metadata tracking, (4) built-in support for benchmarks like BEIR and MS MARCO, and (5) a completely open-source extensible implementation available to all researchers. QueryGym is publicly available at https://github.com/radinhamidi/QueryGym.
>
---
#### [new 035] The Subtle Art of Defection: Understanding Uncooperative Behaviors in LLM based Multi-Agent Systems
- **分类: cs.MA; cs.CL**

- **简介: 该论文研究大模型多智能体系统中的不合作行为。针对现有研究缺乏行为分类与动态模拟的痛点，提出基于博弈论的分类框架与多阶段仿真流程。通过资源管理实验验证，发现任何不合作行为均可导致系统快速崩溃，而合作则保持100%稳定性，揭示了系统韧性设计的重要性。**

- **链接: [https://arxiv.org/pdf/2511.15862v1](https://arxiv.org/pdf/2511.15862v1)**

> **作者:** Devang Kulshreshtha; Wanyu Du; Raghav Jain; Srikanth Doss; Hang Su; Sandesh Swamy; Yanjun Qi
>
> **摘要:** This paper introduces a novel framework for simulating and analyzing how uncooperative behaviors can destabilize or collapse LLM-based multi-agent systems. Our framework includes two key components: (1) a game theory-based taxonomy of uncooperative agent behaviors, addressing a notable gap in the existing literature; and (2) a structured, multi-stage simulation pipeline that dynamically generates and refines uncooperative behaviors as agents' states evolve. We evaluate the framework via a collaborative resource management setting, measuring system stability using metrics such as survival time and resource overuse rate. Empirically, our framework achieves 96.7% accuracy in generating realistic uncooperative behaviors, validated by human evaluations. Our results reveal a striking contrast: cooperative agents maintain perfect system stability (100% survival over 12 rounds with 0% resource overuse), while any uncooperative behavior can trigger rapid system collapse within 1 to 7 rounds. These findings demonstrate that uncooperative agents can significantly degrade collective outcomes, highlighting the need for designing more resilient multi-agent systems.
>
---
#### [new 036] CARE-RAG - Clinical Assessment and Reasoning in RAG
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究临床问答中检索与推理之间的差距，聚焦于大模型在权威证据下仍存在推理错误的问题。针对此问题，提出CARE-RAG评估框架，衡量推理的准确性、一致性和真实性，强调需像评估检索一样严格评估推理，以保障临床应用安全。**

- **链接: [https://arxiv.org/pdf/2511.15994v1](https://arxiv.org/pdf/2511.15994v1)**

> **作者:** Deepthi Potluri; Aby Mammen Mathew; Jeffrey B DeWitt; Alexander L. Rasgon; Yide Hao; Junyuan Hong; Ying Ding
>
> **备注:** The Second Workshop on GenAI for Health: Potential, Trust, and Policy Compliance
>
> **摘要:** Access to the right evidence does not guarantee that large language models (LLMs) will reason with it correctly. This gap between retrieval and reasoning is especially concerning in clinical settings, where outputs must align with structured protocols. We study this gap using Written Exposure Therapy (WET) guidelines as a testbed. In evaluating model responses to curated clinician-vetted questions, we find that errors persist even when authoritative passages are provided. To address this, we propose an evaluation framework that measures accuracy, consistency, and fidelity of reasoning. Our results highlight both the potential and the risks: retrieval-augmented generation (RAG) can constrain outputs, but safe deployment requires assessing reasoning as rigorously as retrieval.
>
---
#### [new 037] Step-Audio-R1 Technical Report
- **分类: cs.AI; cs.CL; cs.SD**

- **简介: 该论文提出Step-Audio-R1，首个成功实现音频领域推理的模型。针对音频语言模型常因缺乏有效推理而表现不佳的问题，研究提出模态锚定推理蒸馏框架（MGRD），使推理链基于真实声学特征，避免幻觉。实验表明其性能超越Gemini 2.5 Pro，接近Gemini 3 Pro，证明推理能力可有效迁移至音频领域。**

- **链接: [https://arxiv.org/pdf/2511.15848v1](https://arxiv.org/pdf/2511.15848v1)**

> **作者:** Fei Tian; Xiangyu Tony Zhang; Yuxin Zhang; Haoyang Zhang; Yuxin Li; Daijiao Liu; Yayue Deng; Donghang Wu; Jun Chen; Liang Zhao; Chengyuan Yao; Hexin Liu; Eng Siong Chng; Xuerui Yang; Xiangyu Zhang; Daxin Jiang; Gang Yu
>
> **备注:** 15 pages, 5 figures. Technical Report
>
> **摘要:** Recent advances in reasoning models have demonstrated remarkable success in text and vision domains through extended chain-of-thought deliberation. However, a perplexing phenomenon persists in audio language models: they consistently perform better with minimal or no reasoning, raising a fundamental question - can audio intelligence truly benefit from deliberate thinking? We introduce Step-Audio-R1, the first audio reasoning model that successfully unlocks reasoning capabilities in the audio domain. Through our proposed Modality-Grounded Reasoning Distillation (MGRD) framework, Step-Audio-R1 learns to generate audio-relevant reasoning chains that genuinely ground themselves in acoustic features rather than hallucinating disconnected deliberations. Our model exhibits strong audio reasoning capabilities, surpassing Gemini 2.5 Pro and achieving performance comparable to the state-of-the-art Gemini 3 Pro across comprehensive audio understanding and reasoning benchmarks spanning speech, environmental sounds, and music. These results demonstrate that reasoning is a transferable capability across modalities when appropriately anchored, transforming extended deliberation from a liability into a powerful asset for audio intelligence. By establishing the first successful audio reasoning model, Step-Audio-R1 opens new pathways toward building truly multimodal reasoning systems that think deeply across all sensory modalities.
>
---
#### [new 038] The Oracle and The Prism: A Decoupled and Efficient Framework for Generative Recommendation Explanation
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对生成式推荐解释中排名与解释联合优化导致的性能-效率权衡问题，提出一种解耦框架Prism。通过将任务分解为独立的排名与解释阶段，利用大模型（Oracle）生成高质量解释知识，再由轻量级学生模型（Prism）生成个性化解释。实验表明，该方法在保持高解释质量的同时，显著提升推理速度并降低内存消耗。**

- **链接: [https://arxiv.org/pdf/2511.16543v1](https://arxiv.org/pdf/2511.16543v1)**

> **作者:** Jiaheng Zhang; Daqiang Zhang
>
> **备注:** 11 pages,3 figures
>
> **摘要:** The integration of Large Language Models (LLMs) into explainable recommendation systems often leads to a performance-efficiency trade-off in end-to-end architectures, where joint optimization of ranking and explanation can result in suboptimal compromises. To resolve this, we propose Prism, a novel decoupled framework that rigorously separates the recommendation process into a dedicated ranking stage and an explanation generation stage. Inspired by knowledge distillation, Prism leverages a powerful teacher LLM (e.g., FLAN-T5-XXL) as an Oracle to produce high-fidelity explanatory knowledge. A compact, fine-tuned student model (e.g., BART-Base), the Prism, then specializes in synthesizing this knowledge into personalized explanations. This decomposition ensures that each component is optimized for its specific objective, eliminating inherent conflicts in coupled models. Extensive experiments on benchmark datasets demonstrate that our 140M-parameter Prism model significantly outperforms its 11B-parameter teacher in human evaluations of faithfulness and personalization, while achieving a 24 times speedup and a 10 times reduction in memory consumption during inference. These results validate that decoupling, coupled with targeted distillation, provides an efficient and effective pathway to high-quality explainable recommendation.
>
---
#### [new 039] Codec2Vec: Self-Supervised Speech Representation Learning Using Neural Speech Codecs
- **分类: eess.AS; cs.CL**

- **简介: 该论文提出Codec2Vec，一种基于离散语音编码器单元的自监督语音表征学习框架。针对传统连续输入模型存储与训练效率低的问题，利用神经语音编码器生成离散单元，实现高效存储与快速训练，同时保障良好性能，在SUPERB基准上表现优异。**

- **链接: [https://arxiv.org/pdf/2511.16639v1](https://arxiv.org/pdf/2511.16639v1)**

> **作者:** Wei-Cheng Tseng; David Harwath
>
> **备注:** To be presented at ASRU 2025
>
> **摘要:** Recent advancements in neural audio codecs have not only enabled superior audio compression but also enhanced speech synthesis techniques. Researchers are now exploring their potential as universal acoustic feature extractors for a broader range of speech processing tasks. Building on this trend, we introduce Codec2Vec, the first speech representation learning framework that relies exclusively on discrete audio codec units. This approach offers several advantages, including improved data storage and transmission efficiency, faster training, and enhanced data privacy. We explore masked prediction with various training target derivation strategies to thoroughly understand the effectiveness of this framework. Evaluated on the SUPERB benchmark, Codec2Vec achieves competitive performance compared to continuous-input models while reducing storage requirements by up to 16.5x and training time by 2.3x, showcasing its scalability and efficiency.
>
---
#### [new 040] SurvAgent: Hierarchical CoT-Enhanced Case Banking and Dichotomy-Based Multi-Agent System for Multimodal Survival Prediction
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对癌症生存预测中模型缺乏透明性的问题，提出SurvAgent系统。通过层次化思维链增强的多模态病例库构建与基于二分法的多专家推理，实现病理与基因数据融合、兴趣区域有效探索及历史经验学习，提升预测可解释性与准确性。**

- **链接: [https://arxiv.org/pdf/2511.16635v1](https://arxiv.org/pdf/2511.16635v1)**

> **作者:** Guolin Huang; Wenting Chen; Jiaqi Yang; Xinheng Lyu; Xiaoling Luo; Sen Yang; Xiaohan Xing; Linlin Shen
>
> **备注:** 20 pages
>
> **摘要:** Survival analysis is critical for cancer prognosis and treatment planning, yet existing methods lack the transparency essential for clinical adoption. While recent pathology agents have demonstrated explainability in diagnostic tasks, they face three limitations for survival prediction: inability to integrate multimodal data, ineffective region-of-interest exploration, and failure to leverage experiential learning from historical cases. We introduce SurvAgent, the first hierarchical chain-of-thought (CoT)-enhanced multi-agent system for multimodal survival prediction. SurvAgent consists of two stages: (1) WSI-Gene CoT-Enhanced Case Bank Construction employs hierarchical analysis through Low-Magnification Screening, Cross-Modal Similarity-Aware Patch Mining, and Confidence-Aware Patch Mining for pathology images, while Gene-Stratified analysis processes six functional gene categories. Both generate structured reports with CoT reasoning, storing complete analytical processes for experiential learning. (2) Dichotomy-Based Multi-Expert Agent Inference retrieves similar cases via RAG and integrates multimodal reports with expert predictions through progressive interval refinement. Extensive experiments on five TCGA cohorts demonstrate SurvAgent's superority over conventional methods, proprietary MLLMs, and medical agents, establishing a new paradigm for explainable AI-driven survival prediction in precision oncology.
>
---
#### [new 041] SpellForger: Prompting Custom Spell Properties In-Game using BERT supervised-trained model
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出SpellForger，一个基于BERT的AI游戏系统，允许玩家用自然语言生成自定义法术。任务为将文本提示转化为平衡的法术属性，解决游戏内容个性化与AI作为核心玩法机制的融合问题。工作包括构建监督训练的BERT模型、实现实时法术生成，并在Unity中集成Python后端，验证了AI作为直接游戏机制的可行性。**

- **链接: [https://arxiv.org/pdf/2511.16018v1](https://arxiv.org/pdf/2511.16018v1)**

> **作者:** Emanuel C. Silva; Emily S. M. Salum; Gabriel M. Arantes; Matheus P. Pereira; Vinicius F. Oliveira; Alessandro L. Bicho
>
> **备注:** Published in Anais Estendidos do XXIV Simpósio Brasileiro de Jogos e Entretenimento Digital (SBGames 2025)
>
> **摘要:** Introduction: The application of Artificial Intelligence in games has evolved significantly, allowing for dynamic content generation. However, its use as a core gameplay co-creation tool remains underexplored. Objective: This paper proposes SpellForger, a game where players create custom spells by writing natural language prompts, aiming to provide a unique experience of personalization and creativity. Methodology: The system uses a supervisedtrained BERT model to interpret player prompts. This model maps textual descriptions to one of many spell prefabs and balances their parameters (damage, cost, effects) to ensure competitive integrity. The game is developed in the Unity Game Engine, and the AI backend is in Python. Expected Results: We expect to deliver a functional prototype that demonstrates the generation of spells in real time, applied to an engaging gameplay loop, where player creativity is central to the experience, validating the use of AI as a direct gameplay mechanic.
>
---
#### [new 042] TimeViper: A Hybrid Mamba-Transformer Vision-Language Model for Efficient Long Video Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出TimeViper，一种用于长视频理解的混合Mamba-Transformer视觉语言模型。针对长视频处理中效率与上下文建模难题，通过混合架构提升效率，并发现视觉信息向文本传递导致冗余，提出TransV模块压缩视觉令牌，实现超长视频（>10,000帧）高效理解，实验验证其性能与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.16595v1](https://arxiv.org/pdf/2511.16595v1)**

> **作者:** Boshen Xu; Zihan Xiao; Jiaze Li; Jianzhong Ju; Zhenbo Luo; Jian Luan; Qin Jin
>
> **备注:** Project page: https://xuboshen.github.io/TimeViper
>
> **摘要:** We introduce TimeViper, a hybrid vision-language model designed to tackle challenges of long video understanding. Processing long videos demands both an efficient model architecture and an effective mechanism for handling extended temporal contexts. To this end, TimeViper adopts a hybrid Mamba-Transformer backbone that combines the efficiency of state-space models with the expressivity of attention mechanisms. Through this hybrid design, we reveal the vision-to-text information aggregation phenomenon, where information progressively flows from vision tokens to text tokens across increasing LLM depth, resulting in severe vision token redundancy. Motivated by this observation, we propose TransV, a token information transfer module that transfers and compresses vision tokens into instruction tokens while maintaining multimodal understanding capabilities. This design enables TimeViper to process hour-long videos exceeding 10,000 frames. Extensive experiments across multiple benchmarks demonstrate that TimeViper competes with state-of-the-art models while extending frame numbers. We further analyze attention behaviors of both Mamba and Transformer layers, offering new insights into hybrid model interpretability. This work represents an initial step towards developing, interpreting, and compressing hybrid Mamba-Transformer architectures.
>
---
#### [new 043] AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出AccelOpt，一个自进化的大语言模型代理系统，用于自动优化新兴AI加速器的计算核函数。针对缺乏硬件专家优化知识的问题，系统通过迭代生成与经验记忆提升优化能力。在自建的NKIBench基准上，其性能随时间显著提升，且成本仅为商用模型的1/26。**

- **链接: [https://arxiv.org/pdf/2511.15915v1](https://arxiv.org/pdf/2511.15915v1)**

> **作者:** Genghan Zhang; Shaowei Zhu; Anjiang Wei; Zhenyu Song; Allen Nie; Zhen Jia; Nandita Vijaykumar; Yida Wang; Kunle Olukotun
>
> **摘要:** We present AccelOpt, a self-improving large language model (LLM) agentic system that autonomously optimizes kernels for emerging AI acclerators, eliminating the need for expert-provided hardware-specific optimization knowledge. AccelOpt explores the kernel optimization space through iterative generation, informed by an optimization memory that curates experiences and insights from previously encountered slow-fast kernel pairs. We build NKIBench, a new benchmark suite of AWS Trainium accelerator kernels with varying complexity extracted from real-world LLM workloads to evaluate the effectiveness of AccelOpt. Our evaluation confirms that AccelOpt's capability improves over time, boosting the average percentage of peak throughput from $49\%$ to $61\%$ on Trainium 1 and from $45\%$ to $59\%$ on Trainium 2 for NKIBench kernels. Moreover, AccelOpt is highly cost-effective: using open-source models, it matches the kernel improvements of Claude Sonnet 4 while being $26\times$ cheaper.
>
---
#### [new 044] TOFA: Training-Free One-Shot Federated Adaptation for Vision-Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对联邦学习中视觉语言模型（VLM）适应效率低、通信开销大及数据异构问题，提出无需训练的TOFA框架。通过双模态特征提取与自适应融合，实现单轮交互下的轻量级高效适配，显著降低资源消耗并提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.16423v1](https://arxiv.org/pdf/2511.16423v1)**

> **作者:** Li Zhang; Zhongxuan Han; XiaoHua Feng; Jiaming Zhang; Yuyuan Li; Linbo Jiang; Jianan Lin; Chaochao Chen
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Efficient and lightweight adaptation of pre-trained Vision-Language Models (VLMs) to downstream tasks through collaborative interactions between local clients and a central server is a rapidly emerging research topic in federated learning. Existing adaptation algorithms are typically trained iteratively, which incur significant communication costs and increase the susceptibility to potential attacks. Motivated by the one-shot federated training techniques that reduce client-server exchanges to a single round, developing a lightweight one-shot federated VLM adaptation method to alleviate these issues is particularly attractive. However, current one-shot approaches face certain challenges in adapting VLMs within federated settings: (1) insufficient exploitation of the rich multimodal information inherent in VLMs; (2) lack of specialized adaptation strategies to systematically handle the severe data heterogeneity; and (3) requiring additional training resource of clients or server. To bridge these gaps, we propose a novel Training-free One-shot Federated Adaptation framework for VLMs, named TOFA. To fully leverage the generalizable multimodal features in pre-trained VLMs, TOFA employs both visual and textual pipelines to extract task-relevant representations. In the visual pipeline, a hierarchical Bayesian model learns personalized, class-specific prototype distributions. For the textual pipeline, TOFA evaluates and globally aligns the generated local text prompts for robustness. An adaptive weight calibration mechanism is also introduced to combine predictions from both modalities, balancing personalization and robustness to handle data heterogeneity. Our method is training-free, not relying on additional training resources on either the client or server side. Extensive experiments across 9 datasets in various federated settings demonstrate the effectiveness of the proposed TOFA method.
>
---
#### [new 045] Chain of Summaries: Summarization Through Iterative Questioning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Chain of Summaries（CoS）方法，解决LLMs处理网页内容时因格式不友好和上下文长度限制导致的信息获取困难问题。通过迭代提问式总结，生成通用、信息密集的摘要，显著提升下游问答性能，减少令牌使用，且不依赖特定LLM。**

- **链接: [https://arxiv.org/pdf/2511.15719v1](https://arxiv.org/pdf/2511.15719v1)**

> **作者:** William Brach; Lukas Galke Poech
>
> **摘要:** Large Language Models (LLMs) are increasingly using external web content. However, much of this content is not easily digestible by LLMs due to LLM-unfriendly formats and limitations of context length. To address this issue, we propose a method for generating general-purpose, information-dense summaries that act as plain-text repositories of web content. Inspired by Hegel's dialectical method, our approach, denoted as Chain of Summaries (CoS), iteratively refines an initial summary (thesis) by identifying its limitations through questioning (antithesis), leading to a general-purpose summary (synthesis) that can satisfy current and anticipate future information needs. Experiments on the TriviaQA, TruthfulQA, and SQUAD datasets demonstrate that CoS outperforms zero-shot LLM baselines by up to 66% and specialized summarization methods such as BRIO and PEGASUS by up to 27%. CoS-generated summaries yield higher Q&A performance compared to the source content, while requiring substantially fewer tokens and being agnostic to the specific downstream LLM. CoS thus resembles an appealing option for website maintainers to make their content more accessible for LLMs, while retaining possibilities for human oversight.
>
---
## 更新

#### [replaced 001] Injecting Falsehoods: Adversarial Man-in-the-Middle Attacks Undermining Factual Recall in LLMs
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2511.05919v2](https://arxiv.org/pdf/2511.05919v2)**

> **作者:** Alina Fastowski; Bardh Prenkaj; Yuxiao Li; Gjergji Kasneci
>
> **摘要:** LLMs are now an integral part of information retrieval. As such, their role as question answering chatbots raises significant concerns due to their shown vulnerability to adversarial man-in-the-middle (MitM) attacks. Here, we propose the first principled attack evaluation on LLM factual memory under prompt injection via Xmera, our novel, theory-grounded MitM framework. By perturbing the input given to "victim" LLMs in three closed-book and fact-based QA settings, we undermine the correctness of the responses and assess the uncertainty of their generation process. Surprisingly, trivial instruction-based attacks report the highest success rate (up to ~85.3%) while simultaneously having a high uncertainty for incorrectly answered questions. To provide a simple defense mechanism against Xmera, we train Random Forest classifiers on the response uncertainty levels to distinguish between attacked and unattacked queries (average AUC of up to ~96%). We believe that signaling users to be cautious about the answers they receive from black-box and potentially corrupt LLMs is a first checkpoint toward user cyberspace safety.
>
---
#### [replaced 002] Property-guided Inverse Design of Metal-Organic Frameworks Using Quantum Natural Language Processing
- **分类: cs.LG; cs.AI; cs.CL; quant-ph**

- **链接: [https://arxiv.org/pdf/2405.11783v3](https://arxiv.org/pdf/2405.11783v3)**

> **作者:** Shinyoung Kang; Jihan Kim
>
> **备注:** 46 pages, 7 figures, 6 supplementary figures, 1 table, 2 supplementary tables, 1 supplementary note
>
> **摘要:** In this study, we explore the potential of using quantum natural language processing (QNLP) to inverse design metal-organic frameworks (MOFs) with targeted properties. Specifically, by analyzing 450 hypothetical MOF structures consisting of 3 topologies, 10 metal nodes and 15 organic ligands, we categorize these structures into four distinct classes for pore volume and $CO_{2}$ Henry's constant values. We then compare various QNLP models (i.e. the bag-of-words, DisCoCat (Distributional Compositional Categorical), and sequence-based models) to identify the most effective approach to process the MOF dataset. Using a classical simulator provided by the IBM Qiskit, the bag-of-words model is identified to be the optimum model, achieving validation accuracies of 88.6% and 78.0% for binary classification tasks on pore volume and $CO_{2}$ Henry's constant, respectively. Further, we developed multi-class classification models tailored to the probabilistic nature of quantum circuits, with average test accuracies of 92% and 80% across different classes for pore volume and $CO_{2}$ Henry's constant datasets. Finally, the performance of generating MOF with target properties showed accuracies of 93.5% for pore volume and 87% for $CO_{2}$ Henry's constant, respectively. Although our investigation covers only a fraction of the vast MOF search space, it marks a promising first step towards using quantum computing for materials design, offering a new perspective through which to explore the complex landscape of MOFs.
>
---
#### [replaced 003] Efficient Architectures for High Resolution Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2501.02584v2](https://arxiv.org/pdf/2501.02584v2)**

> **作者:** Miguel Carvalho; Bruno Martins
>
> **备注:** Accepted at COLING 2025
>
> **摘要:** Vision-Language Models (VLMs) have recently experienced significant advancements. However, challenges persist in the accurate recognition of fine details within high resolution images, which limits performance in multiple tasks. This work introduces Pheye, a novel architecture that efficiently processes high-resolution images while training fewer parameters than similarly sized VLMs. Notably, Pheye achieves a high efficiency while maintaining strong performance, particularly in tasks that demand fine-grained image understanding and/or the handling of scene-text.
>
---
#### [replaced 004] GPTopic: Dynamic and Interactive Topic Representations
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2403.03628v3](https://arxiv.org/pdf/2403.03628v3)**

> **作者:** Arik Reuter; Bishnu Khadka; Anton Thielmann; Christoph Weisser; Sebastian Fischer; Benjamin Säfken
>
> **摘要:** Topic modeling seems to be almost synonymous with generating lists of top words to represent topics within large text corpora. However, deducing a topic from such list of individual terms can require substantial expertise and experience, making topic modelling less accessible to people unfamiliar with the particularities and pitfalls of top-word interpretation. A topic representation limited to top-words might further fall short of offering a comprehensive and easily accessible characterization of the various aspects, facets and nuances a topic might have. To address these challenges, we introduce GPTopic, a software package that leverages Large Language Models (LLMs) to create dynamic, interactive topic representations. GPTopic provides an intuitive chat interface for users to explore, analyze, and refine topics interactively, making topic modeling more accessible and comprehensive. The corresponding code is available here: https://github.com/ArikReuter/TopicGPT.
>
---
#### [replaced 005] Sigma: Semantically Informative Pre-training for Skeleton-based Sign Language Understanding
- **分类: cs.CV; cs.CL**

- **链接: [https://arxiv.org/pdf/2509.21223v2](https://arxiv.org/pdf/2509.21223v2)**

> **作者:** Muxin Pu; Mei Kuan Lim; Chun Yong Chong; Chen Change Loy
>
> **摘要:** Pre-training has proven effective for learning transferable features in sign language understanding (SLU) tasks. Recently, skeleton-based methods have gained increasing attention because they can robustly handle variations in subjects and backgrounds without being affected by appearance or environmental factors. Current SLU methods continue to face three key limitations: 1) weak semantic grounding, as models often capture low-level motion patterns from skeletal data but struggle to relate them to linguistic meaning; 2) imbalance between local details and global context, with models either focusing too narrowly on fine-grained cues or overlooking them for broader context; and 3) inefficient cross-modal learning, as constructing semantically aligned representations across modalities remains difficult. To address these, we propose Sigma, a unified skeleton-based SLU framework featuring: 1) a sign-aware early fusion mechanism that facilitates deep interaction between visual and textual modalities, enriching visual features with linguistic context; 2) a hierarchical alignment learning strategy that jointly maximises agreements across different levels of paired features from different modalities, effectively capturing both fine-grained details and high-level semantic relationships; and 3) a unified pre-training framework that combines contrastive learning, text matching and language modelling to promote semantic consistency and generalisation. Sigma achieves new state-of-the-art results on isolated sign language recognition, continuous sign language recognition, and gloss-free sign language translation on multiple benchmarks spanning different sign and spoken languages, demonstrating the impact of semantically informative pre-training and the effectiveness of skeletal data as a stand-alone solution for SLU.
>
---
#### [replaced 006] Multimodal Evaluation of Russian-language Architectures
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2511.15552v2](https://arxiv.org/pdf/2511.15552v2)**

> **作者:** Artem Chervyakov; Ulyana Isaeva; Anton Emelyanov; Artem Safin; Maria Tikhonova; Alexander Kharitonov; Yulia Lyakh; Petr Surovtsev; Denis Shevelev; Vildan Saburov; Vasily Konovalov; Elisei Rykov; Ivan Sviridov; Amina Miftakhova; Ilseyar Alimova; Alexander Panchenko; Alexander Kapitanov; Alena Fenogenova
>
> **摘要:** Multimodal large language models (MLLMs) are currently at the center of research attention, showing rapid progress in scale and capabilities, yet their intelligence, limitations, and risks remain insufficiently understood. To address these issues, particularly in the context of the Russian language, where no multimodal benchmarks currently exist, we introduce Mera Multi, an open multimodal evaluation framework for Russian-spoken architectures. The benchmark is instruction-based and encompasses default text, image, audio, and video modalities, comprising 18 newly constructed evaluation tasks for both general-purpose models and modality-specific architectures (image-to-text, video-to-text, and audio-to-text). Our contributions include: (i) a universal taxonomy of multimodal abilities; (ii) 18 datasets created entirely from scratch with attention to Russian cultural and linguistic specificity, unified prompts, and metrics; (iii) baseline results for both closed-source and open-source models; (iv) a methodology for preventing benchmark leakage, including watermarking and licenses for private sets. While our current focus is on Russian, the proposed benchmark provides a replicable methodology for constructing multimodal benchmarks in typologically diverse languages, particularly within the Slavic language family.
>
---
#### [replaced 007] VisPlay: Self-Evolving Vision-Language Models from Images
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.15661v2](https://arxiv.org/pdf/2511.15661v2)**

> **作者:** Yicheng He; Chengsong Huang; Zongxia Li; Jiaxin Huang; Yonghui Yang
>
> **摘要:** Reinforcement learning (RL) provides a principled framework for improving Vision-Language Models (VLMs) on complex reasoning tasks. However, existing RL approaches often rely on human-annotated labels or task-specific heuristics to define verifiable rewards, both of which are costly and difficult to scale. We introduce VisPlay, a self-evolving RL framework that enables VLMs to autonomously improve their reasoning abilities using large amounts of unlabeled image data. Starting from a single base VLM, VisPlay assigns the model into two interacting roles: an Image-Conditioned Questioner that formulates challenging yet answerable visual questions, and a Multimodal Reasoner that generates silver responses. These roles are jointly trained with Group Relative Policy Optimization (GRPO), which incorporates diversity and difficulty rewards to balance the complexity of generated questions with the quality of the silver answers. VisPlay scales efficiently across two model families. When trained on Qwen2.5-VL and MiMo-VL, VisPlay achieves consistent improvements in visual reasoning, compositional generalization, and hallucination reduction across eight benchmarks, including MM-Vet and MMMU, demonstrating a scalable path toward self-evolving multimodal intelligence. The project page is available at https://bruno686.github.io/VisPlay/
>
---
#### [replaced 008] LLMs as Models for Analogical Reasoning
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2406.13803v3](https://arxiv.org/pdf/2406.13803v3)**

> **作者:** Sam Musker; Alex Duchnowski; Raphaël Millière; Ellie Pavlick
>
> **备注:** The title has been changed from Semantic Structure-Mapping in LLM and Human Analogical Reasoning to LLMs as Models for Analogical Reasoning to improve clarity and accuracy
>
> **摘要:** Analogical reasoning -- the capacity to identify and map structural relationships between different domains -- is fundamental to human cognition and learning. Recent studies have shown that large language models (LLMs) can sometimes match humans in analogical reasoning tasks, opening the possibility that analogical reasoning might emerge from domain-general processes. However, it is still debated whether these emergent capacities are largely superficial and limited to simple relations seen during training or whether they encompass the flexible representational and mapping capabilities which are the focus of leading cognitive models of analogy. In this study, we introduce novel analogical reasoning tasks that require participants to map between semantically contentful words and sequences of letters and other abstract characters. This task necessitates the ability to flexibly re-represent rich semantic information -- an ability which is known to be central to human analogy but which is thus far not well captured by existing cognitive theories and models. We assess the performance of both human participants and LLMs on tasks focusing on reasoning from semantic structure and semantic content, introducing variations that test the robustness of their analogical inferences. Advanced LLMs match human performance across several conditions, though humans and LLMs respond differently to certain task variations and semantic distractors. Our results thus provide new evidence that LLMs might offer a how-possibly explanation of human analogical reasoning in contexts that are not yet well modeled by existing theories, but that even today's best models are unlikely to yield how-actually explanations.
>
---
#### [replaced 009] CAIRe: Cultural Attribution of Images by Retrieval-Augmented Evaluation
- **分类: cs.CV; cs.CL**

- **链接: [https://arxiv.org/pdf/2506.09109v2](https://arxiv.org/pdf/2506.09109v2)**

> **作者:** Arnav Yayavaram; Siddharth Yayavaram; Simran Khanuja; Michael Saxon; Graham Neubig
>
> **备注:** Preprint, under review
>
> **摘要:** As text-to-image models become increasingly prevalent, ensuring their equitable performance across diverse cultural contexts is critical. Efforts to mitigate cross-cultural biases have been hampered by trade-offs, including a loss in performance, factual inaccuracies, or offensive outputs. Despite widespread recognition of these challenges, an inability to reliably measure these biases has stalled progress. To address this gap, we introduce CAIRe, an evaluation metric that assesses the degree of cultural relevance of an image, given a user-defined set of labels. Our framework grounds entities and concepts in the image to a knowledge base and uses factual information to give independent graded judgments for each culture label. On a manually curated dataset of culturally salient but rare items built using language models, CAIRe surpasses all baselines by 22% F1 points. Additionally, we construct two datasets for culturally universal concepts, one comprising T2I-generated outputs and another retrieved from naturally occurring data. CAIRe achieves Pearson's correlations of 0.56 and 0.66 with human ratings on these sets, based on a 5-point Likert scale of cultural relevance. This demonstrates its strong alignment with human judgment across diverse image sources.
>
---
#### [replaced 010] Beyond Bias Scores: Unmasking Vacuous Neutrality in Small Language Models
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.08487v2](https://arxiv.org/pdf/2506.08487v2)**

> **作者:** Sumanth Manduru; Carlotta Domeniconi
>
> **摘要:** The rapid adoption of Small Language Models (SLMs) for resource constrained applications has outpaced our understanding of their ethical and fairness implications. To address this gap, we introduce the Vacuous Neutrality Framework (VaNeu), a multi-dimensional evaluation paradigm designed to assess SLM fairness prior to deployment. The framework examines model robustness across four stages - biases, utility, ambiguity handling, and positional bias over diverse social bias categories. To the best of our knowledge, this work presents the first large-scale audit of SLMs in the 0.5-5B parameter range, an overlooked "middle tier" between BERT-class encoders and flagship LLMs. We evaluate nine widely used SLMs spanning four model families under both ambiguous and disambiguated contexts. Our findings show that models demonstrating low bias in early stages often fail subsequent evaluations, revealing hidden vulnerabilities and unreliable reasoning. These results underscore the need for a more comprehensive understanding of fairness and reliability in SLMs, and position the proposed framework as a principled tool for responsible deployment in socially sensitive settings.
>
---
#### [replaced 011] CoBA: Counterbias Text Augmentation for Mitigating Various Spurious Correlations via Semantic Triples
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.21083v2](https://arxiv.org/pdf/2508.21083v2)**

> **作者:** Kyohoon Jin; Juhwan Choi; Jungmin Yun; Junho Lee; Soojin Jang; Youngbin Kim
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** Deep learning models often learn and exploit spurious correlations in training data, using these non-target features to inform their predictions. Such reliance leads to performance degradation and poor generalization on unseen data. To address these limitations, we introduce a more general form of counterfactual data augmentation, termed counterbias data augmentation, which simultaneously tackles multiple biases (e.g., gender bias, simplicity bias) and enhances out-of-distribution robustness. We present CoBA: CounterBias Augmentation, a unified framework that operates at the semantic triple level: first decomposing text into subject-predicate-object triples, then selectively modifying these triples to disrupt spurious correlations. By reconstructing the text from these adjusted triples, CoBA generates counterbias data that mitigates spurious patterns. Through extensive experiments, we demonstrate that CoBA not only improves downstream task performance, but also effectively reduces biases and strengthens out-of-distribution resilience, offering a versatile and robust solution to the challenges posed by spurious correlations.
>
---
#### [replaced 012] AutoJudge: Judge Decoding Without Manual Annotation
- **分类: cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2504.20039v4](https://arxiv.org/pdf/2504.20039v4)**

> **作者:** Roman Garipov; Fedor Velikonivtsev; Ivan Ermakov; Ruslan Svirschevski; Vage Egiazarian; Max Ryabinin
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** We introduce AutoJudge, a method that accelerates large language model (LLM) inference with task-specific lossy speculative decoding. Instead of matching the original model output distribution token-by-token, we identify which of the generated tokens affect the downstream quality of the response, relaxing the distribution match guarantee so that the "unimportant" tokens can be generated faster. Our approach relies on a semi-greedy search algorithm to test which of the mismatches between target and draft models should be corrected to preserve quality and which ones may be skipped. We then train a lightweight classifier based on existing LLM embeddings to predict, at inference time, which mismatching tokens can be safely accepted without compromising the final answer quality. We evaluate the effectiveness of AutoJudge with multiple draft/target model pairs on mathematical reasoning and programming benchmarks, achieving significant speedups at the cost of a minor accuracy reduction. Notably, on GSM8k with the Llama 3.1 70B target model, our approach achieves up to $\approx2\times$ speedup over speculative decoding at the cost of $\le 1\%$ drop in accuracy. When applied to the LiveCodeBench benchmark, AutoJudge automatically detects programming-specific important tokens, accepting $\ge 25$ tokens per speculation cycle at $2\%$ drop in Pass@1. Our approach requires no human annotation and is easy to integrate with modern LLM inference frameworks.
>
---
#### [replaced 013] Diagnosing the Performance Trade-off in Moral Alignment: A Case Study on Gender Stereotypes
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2509.21456v3](https://arxiv.org/pdf/2509.21456v3)**

> **作者:** Guangliang Liu; Bocheng Chen; Han Zi; Xitong Zhang; Kristen Marie Johnson
>
> **摘要:** Moral alignment has emerged as a widely adopted approach for regulating the behavior of pretrained language models (PLMs), typically through fine-tuning on curated datasets. Gender stereotype mitigation is a representational task within the broader application of moral alignment. However, this process often comes at the cost of degraded downstream task performance. Prior studies commonly aim to achieve a performance trade-off by encouraging PLMs to selectively forget only stereotypical knowledge through carefully designed fairness objective, while preserving their language modeling capability (overall forgetting). In this short paper, we investigate whether the performance trade-off can be achieved through the lens of forgetting and the fairness objective. Our analysis shows that the large datasets needed for satisfactory fairness highlight the limitations of current fairness objectives in achieving an effective trade-off: (1) downstream task performance is strongly correlated with overall forgetting; (2) selective forgetting reduces stereotypes, but overall forgetting increases. and (3) general solutions for alleviating forgetting are ineffective at reducing the overall forgetting and fail to improve downstream task performance.
>
---
#### [replaced 014] Crowdsourcing Lexical Diversity
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2410.23133v2](https://arxiv.org/pdf/2410.23133v2)**

> **作者:** Hadi Khalilia; Jahna Otterbacher; Gabor Bella; Shandy Darma; Fausto Giunchiglia
>
> **摘要:** Lexical-semantic resources (LSRs), such as online lexicons and wordnets, are fundamental to natural language processing applications as well as to fields such as linguistic anthropology and language preservation. In many languages, however, such resources suffer from quality issues: incorrect entries, incompleteness, but also the rarely addressed issue of bias towards the English language and Anglo-Saxon culture. Such bias manifests itself in the absence of concepts specific to the language or culture at hand, the presence of foreign (Anglo-Saxon) concepts, as well as in the lack of an explicit indication of untranslatability, also known as cross-lingual lexical gaps, when a term has no equivalent in another language. This paper proposes a novel crowdsourcing methodology for reducing bias in LSRs. Crowd workers compare lexemes from two languages, focusing on domains rich in lexical diversity, such as kinship or food. Our LingoGap crowdsourcing platform facilitates comparisons through microtasks identifying equivalent terms, language-specific terms, and lexical gaps across languages. We validated our method by applying it to two case studies focused on food-related terminology: (1) English and Arabic, and (2) Standard Indonesian and Banjarese. These experiments identified 2,140 lexical gaps in the first case study and 951 in the second. The success of these experiments confirmed the usability of our method and tool for future large-scale lexicon enrichment tasks.
>
---
#### [replaced 015] MAQuA: Adaptive Question-Asking for Multidimensional Mental Health Screening using Item Response Theory
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.07279v3](https://arxiv.org/pdf/2508.07279v3)**

> **作者:** Vasudha Varadarajan; Hui Xu; Rebecca Astrid Boehme; Mariam Marlan Mirstrom; Sverker Sikstrom; H. Andrew Schwartz
>
> **摘要:** Recent advances in large language models (LLMs) offer new opportunities for scalable, interactive mental health assessment, but excessive querying by LLMs burdens users and is inefficient for real-world screening across transdiagnostic symptom profiles. We introduce MAQuA, an adaptive question-asking framework for simultaneous, multidimensional mental health screening. Combining multi-outcome modeling on language responses with item response theory (IRT) and factor analysis, MAQuA selects the questions with most informative responses across multiple dimensions at each turn to optimize diagnostic information, improving accuracy and potentially reducing response burden. Empirical results on a novel dataset reveal that MAQuA reduces the number of assessment questions required for score stabilization by 50-87% compared to random ordering (e.g., achieving stable depression scores with 71% fewer questions and eating disorder scores with 85% fewer questions). MAQuA demonstrates robust performance across both internalizing (depression, anxiety) and externalizing (substance use, eating disorder) domains, with early stopping strategies further reducing patient time and burden. These findings position MAQuA as a powerful and efficient tool for scalable, nuanced, and interactive mental health screening, advancing the integration of LLM-based agents into real-world clinical workflows.
>
---
#### [replaced 016] Probing the Critical Point (CritPt) of AI Reasoning: a Frontier Physics Research Benchmark
- **分类: cs.AI; cond-mat.other; cs.CL; hep-th; quant-ph**

- **链接: [https://arxiv.org/pdf/2509.26574v3](https://arxiv.org/pdf/2509.26574v3)**

> **作者:** Minhui Zhu; Minyang Tian; Xiaocheng Yang; Tianci Zhou; Lifan Yuan; Penghao Zhu; Eli Chertkov; Shengyan Liu; Yufeng Du; Ziming Ji; Indranil Das; Junyi Cao; Yufeng Du; Jiabin Yu; Peixue Wu; Jinchen He; Yifan Su; Yikun Jiang; Yujie Zhang; Chang Liu; Ze-Min Huang; Weizhen Jia; Yunkai Wang; Farshid Jafarpour; Yong Zhao; Xinan Chen; Jessie Shelton; Aaron W. Young; John Bartolotta; Wenchao Xu; Yue Sun; Anjun Chu; Victor Colussi; Chris Akers; Nathan Brooks; Wenbo Fu; Jinchao Zhao; Marvin Qi; Anqi Mu; Yubo Yang; Allen Zang; Yang Lyu; Peizhi Mai; Christopher Wilson; Xuefei Guo; Juntai Zhou; Daniel Inafuku; Chi Xue; Luyu Gao; Ze Yang; Yaïr Hein; Yonatan Kahn; Kevin Zhou; Di Luo; John Drew Wilson; Jarrod T. Reilly; Dmytro Bandak; Ofir Press; Liang Yang; Xueying Wang; Hao Tong; Nicolas Chia; Eliu Huerta; Hao Peng
>
> **备注:** 39 pages, 6 figures, 6 tables
>
> **摘要:** While large language models (LLMs) with reasoning capabilities are progressing rapidly on high-school math competitions and coding, can they reason effectively through complex, open-ended challenges found in frontier physics research? And crucially, what kinds of reasoning tasks do physicists want LLMs to assist with? To address these questions, we present the CritPt (Complex Research using Integrated Thinking - Physics Test, pronounced "critical point"), the first benchmark designed to test LLMs on unpublished, research-level reasoning tasks that broadly covers modern physics research areas, including condensed matter, quantum physics, atomic, molecular & optical physics, astrophysics, high energy physics, mathematical physics, statistical physics, nuclear physics, nonlinear dynamics, fluid dynamics and biophysics. CritPt consists of 71 composite research challenges designed to simulate full-scale research projects at the entry level, which are also decomposed to 190 simpler checkpoint tasks for more fine-grained insights. All problems are newly created by 50+ active physics researchers based on their own research. Every problem is hand-curated to admit a guess-resistant and machine-verifiable answer and is evaluated by an automated grading pipeline heavily customized for advanced physics-specific output formats. We find that while current state-of-the-art LLMs show early promise on isolated checkpoints, they remain far from being able to reliably solve full research-scale challenges: the best average accuracy among base models is only 5.7%, achieved by GPT-5 (high), moderately rising to around 10% when equipped with coding tools. Through the realistic yet standardized evaluation offered by CritPt, we highlight a large disconnect between current model capabilities and realistic physics research demands, offering a foundation to guide the development of scientifically grounded AI tools.
>
---
#### [replaced 017] Multi-dimensional Data Analysis and Applications Basing on LLM Agents and Knowledge Graph Interactions
- **分类: cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2510.15258v2](https://arxiv.org/pdf/2510.15258v2)**

> **作者:** Xi Wang; Xianyao Ling; Kun Li; Gang Yin; Liang Zhang; Jiang Wu; Jun Xu; Fu Zhang; Wenbo Lei; Annie Wang; Peng Gong
>
> **备注:** 14 pages, 7 figures, 40 references
>
> **摘要:** In the current era of big data, extracting deep insights from massive, heterogeneous, and complexly associated multi-dimensional data has become a significant challenge. Large Language Models (LLMs) perform well in natural language understanding and generation, but still suffer from "hallucination" issues when processing structured knowledge and are difficult to update in real-time. Although Knowledge Graphs (KGs) can explicitly store structured knowledge, their static nature limits dynamic interaction and analytical capabilities. Therefore, this paper proposes a multi-dimensional data analysis method based on the interactions between LLM agents and KGs, constructing a dynamic, collaborative analytical ecosystem. This method utilizes LLM agents to automatically extract product data from unstructured data, constructs and visualizes the KG in real-time, and supports users in deep exploration and analysis of graph nodes through an interactive platform. Experimental results show that this method has significant advantages in product ecosystem analysis, relationship mining, and user-driven exploratory analysis, providing new ideas and tools for multi-dimensional data analysis.
>
---
#### [replaced 018] Efficient Environmental Claim Detection with Hyperbolic Graph Neural Networks
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2502.13628v3](https://arxiv.org/pdf/2502.13628v3)**

> **作者:** Darpan Aswal; Manjira Sinha
>
> **摘要:** Transformer based models, especially large language models (LLMs) dominate the field of NLP with their mass adoption in tasks such as text generation, summarization and fake news detection. These models offer ease of deployment and reliability for most applications, however, they require significant amounts of computational power for training as well as inference. This poses challenges in their adoption in resource-constrained applications, especially in the open-source community where compute availability is usually scarce. This work proposes a graph-based approach for Environmental Claim Detection, exploring Graph Neural Networks (GNNs) and Hyperbolic Graph Neural Networks (HGNNs) as lightweight yet effective alternatives to transformer-based models. Re-framing the task as a graph classification problem, we transform claim sentences into dependency parsing graphs, utilizing a combination of word2vec \& learnable part-of-speech (POS) tag embeddings for the node features and encoding syntactic dependencies in the edge relations. Our results show that our graph-based models, particularly HGNNs in the poincaré space (P-HGNNs), achieve performance superior to the state-of-the-art on environmental claim detection while using up to \textbf{30x fewer parameters}. We also demonstrate that HGNNs benefit vastly from explicitly modeling data in hierarchical (tree-like) structures, enabling them to significantly improve over their euclidean counterparts.
>
---
#### [replaced 019] Confidence-Guided Stepwise Model Routing for Cost-Efficient Reasoning
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2511.06190v2](https://arxiv.org/pdf/2511.06190v2)**

> **作者:** Sangmook Lee; Dohyung Kim; Hyukhun Koh; Nakyeong Yang; Kyomin Jung
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** Recent advances in Large Language Models (LLMs) - particularly model scaling and test-time techniques - have greatly enhanced the reasoning capabilities of language models at the expense of higher inference costs. To lower inference costs, prior works train router models or deferral mechanisms that allocate easy queries to a small, efficient model, while forwarding harder queries to larger, more expensive models. However, these trained router models often lack robustness under domain shifts and require expensive data synthesis techniques such as Monte Carlo rollouts to obtain sufficient ground-truth routing labels for training. In this work, we propose Confidence-Guided Stepwise Model Routing for Cost-Efficient Reasoning (STEER), a domain-agnostic framework that performs fine-grained, step-level routing between smaller and larger LLMs without utilizing external models. STEER leverages confidence scores from the smaller model's logits prior to generating a reasoning step, so that the large model is invoked only when necessary. Extensive evaluations using different LLMs on a diverse set of challenging benchmarks across multiple domains such as Mathematical Reasoning, Multi-Hop QA, and Planning tasks indicate that STEER achieves competitive or enhanced accuracy while reducing inference costs (up to +20% accuracy with 48% less FLOPs compared to solely using the larger model on AIME), outperforming baselines that rely on trained external modules. Our results establish model-internal confidence as a robust, domain-agnostic signal for model routing, offering a scalable pathway for efficient LLM deployment.
>
---
#### [replaced 020] Adversarial Poetry as a Universal Single-Turn Jailbreak Mechanism in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.15304v2](https://arxiv.org/pdf/2511.15304v2)**

> **作者:** Piercosma Bisconti; Matteo Prandi; Federico Pierucci; Francesco Giarrusso; Marcantonio Bracale; Marcello Galisai; Vincenzo Suriani; Olga Sorokoletova; Federico Sartore; Daniele Nardi
>
> **摘要:** We present evidence that adversarial poetry functions as a universal single-turn jailbreak technique for Large Language Models (LLMs). Across 25 frontier proprietary and open-weight models, curated poetic prompts yielded high attack-success rates (ASR), with some providers exceeding 90%. Mapping prompts to MLCommons and EU CoP risk taxonomies shows that poetic attacks transfer across CBRN, manipulation, cyber-offence, and loss-of-control domains. Converting 1,200 MLCommons harmful prompts into verse via a standardized meta-prompt produced ASRs up to 18 times higher than their prose baselines. Outputs are evaluated using an ensemble of 3 open-weight LLM judges, whose binary safety assessments were validated on a stratified human-labeled subset. Poetic framing achieved an average jailbreak success rate of 62% for hand-crafted poems and approximately 43% for meta-prompt conversions (compared to non-poetic baselines), substantially outperforming non-poetic baselines and revealing a systematic vulnerability across model families and safety training approaches. These findings demonstrate that stylistic variation alone can circumvent contemporary safety mechanisms, suggesting fundamental limitations in current alignment methods and evaluation protocols.
>
---
#### [replaced 021] MajinBook: An open catalogue of digital world literature with likes
- **分类: cs.CL; cs.CY; stat.OT**

- **链接: [https://arxiv.org/pdf/2511.11412v3](https://arxiv.org/pdf/2511.11412v3)**

> **作者:** Antoine Mazières; Thierry Poibeau
>
> **备注:** 9 pages, 5 figures, 1 table
>
> **摘要:** This data paper introduces MajinBook, an open catalogue designed to facilitate the use of shadow libraries--such as Library Genesis and Z-Library--for computational social science and cultural analytics. By linking metadata from these vast, crowd-sourced archives with structured bibliographic data from Goodreads, we create a high-precision corpus of over 539,000 references to English-language books spanning three centuries, enriched with first publication dates, genres, and popularity metrics like ratings and reviews. Our methodology prioritizes natively digital EPUB files to ensure machine-readable quality, while addressing biases in traditional corpora like HathiTrust, and includes secondary datasets for French, German, and Spanish. We evaluate the linkage strategy for accuracy, release all underlying data openly, and discuss the project's legal permissibility under EU and US frameworks for text and data mining in research.
>
---
#### [replaced 022] CoTKR: Chain-of-Thought Enhanced Knowledge Rewriting for Complex Knowledge Graph Question Answering
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2409.19753v4](https://arxiv.org/pdf/2409.19753v4)**

> **作者:** Yike Wu; Yi Huang; Nan Hu; Yuncheng Hua; Guilin Qi; Jiaoyan Chen; Jeff Z. Pan
>
> **摘要:** Recent studies have explored the use of Large Language Models (LLMs) with Retrieval Augmented Generation (RAG) for Knowledge Graph Question Answering (KGQA). They typically require rewriting retrieved subgraphs into natural language formats comprehensible to LLMs. However, when tackling complex questions, the knowledge rewritten by existing methods may include irrelevant information, omit crucial details, or fail to align with the question's semantics. To address them, we propose a novel rewriting method CoTKR, Chain-of-Thought Enhanced Knowledge Rewriting, for generating reasoning traces and corresponding knowledge in an interleaved manner, thereby mitigating the limitations of single-step knowledge rewriting. Additionally, to bridge the preference gap between the knowledge rewriter and the question answering (QA) model, we propose a training strategy PAQAF, Preference Alignment from Question Answering Feedback, for leveraging feedback from the QA model to further optimize the knowledge rewriter. We conduct experiments using various LLMs across several KGQA benchmarks. Experimental results demonstrate that, compared with previous knowledge rewriting methods, CoTKR generates the most beneficial knowledge representation for QA models, which significantly improves the performance of LLMs in KGQA.
>
---
#### [replaced 023] The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2506.06941v3](https://arxiv.org/pdf/2506.06941v3)**

> **作者:** Parshin Shojaee; Iman Mirzadeh; Keivan Alizadeh; Maxwell Horton; Samy Bengio; Mehrdad Farajtabar
>
> **备注:** NeurIPS 2025. camera-ready version + additional discussion in the appendix
>
> **摘要:** Recent generations of language models have introduced Large Reasoning Models (LRMs) that generate detailed thinking processes before providing answers. While these models demonstrate improved performance on reasoning benchmarks, their fundamental capabilities, scaling properties, and limitations remain insufficiently understood. Current evaluations primarily focus on established math and coding benchmarks, emphasizing final answer accuracy. However, this evaluation paradigm often suffers from contamination and does not provide insights into the reasoning traces. In this work, we systematically investigate these gaps with the help of controllable puzzle environments that allow precise manipulation of complexity while maintaining consistent logical structures. This setup enables the analysis of not only final answers but also the internal reasoning traces, offering insights into how LRMs think. Through extensive experiments, we show that LRMs face a complete accuracy collapse beyond certain complexities. Moreover, they exhibit a counterintuitive scaling limit: their reasoning effort increases with problem complexity up to a point, then declines despite having remaining token budget. By comparing LRMs with their standard LLM counterparts under same inference compute, we identify three performance regimes: (1) low-complexity tasks where standard models outperform LRMs, (2) medium-complexity tasks where LRMs demonstrates advantage, and (3) high-complexity tasks where both models face complete collapse. We found that LRMs have limitations in exact computation: they fail to use explicit algorithms and reason inconsistently across scales. We also investigate the reasoning traces in more depth, studying the patterns of explored solutions and analyzing the models' computational behavior, shedding light on their strengths, limitations, and raising questions about their reasoning capabilities.
>
---
#### [replaced 024] Steering Evaluation-Aware Language Models to Act Like They Are Deployed
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2510.20487v3](https://arxiv.org/pdf/2510.20487v3)**

> **作者:** Tim Tian Hua; Andrew Qin; Samuel Marks; Neel Nanda
>
> **摘要:** Large language models (LLMs) can sometimes detect when they are being evaluated and adjust their behavior to appear more aligned, compromising the reliability of safety evaluations. In this paper, we show that adding a steering vector to an LLM's activations can suppress evaluation-awareness and make the model act like it is deployed during evaluation. To study our steering technique, we train an LLM to exhibit evaluation-aware behavior using a two-step training process designed to mimic how this behavior could emerge naturally. First, we perform continued pretraining on documents with factual descriptions of the model (1) using Python type hints during evaluation but not during deployment and (2) recognizing that the presence of a certain evaluation cue always means that it is being tested. Then, we train the model with expert iteration to use Python type hints in evaluation settings. The resulting model is evaluation-aware: it writes type hints in evaluation contexts more than deployment contexts. We find that activation steering can suppress evaluation awareness and make the model act like it is deployed even when the cue is present. Importantly, we constructed our steering vector using the original model before our additional training. Our results suggest that AI evaluators could improve the reliability of safety evaluations by steering models to act like they are deployed.
>
---
#### [replaced 025] LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2511.07129v2](https://arxiv.org/pdf/2511.07129v2)**

> **作者:** Seungeon Lee; Soumi Das; Manish Gupta; Krishna P. Gummadi
>
> **摘要:** Low-Rank Adaptation (LoRA) has emerged as a parameter-efficient approach for fine-tuning large language models. However, conventional LoRA adapters are typically trained for a single task, limiting their applicability in real-world settings where inputs may span diverse and unpredictable domains. At inference time, existing approaches combine multiple LoRAs for improving performance on diverse tasks, while usually requiring labeled data or additional task-specific training, which is expensive at scale. In this work, we introduce LoRA on the Go (LoGo), a training-free framework that dynamically selects and merges adapters at the instance level without any additional requirements. LoGo leverages signals extracted from a single forward pass through LoRA adapters, to identify the most relevant adapters and determine their contributions on-the-fly. Across 5 NLP benchmarks, 27 datasets, and 3 model families, LoGo outperforms training-based baselines on some tasks upto a margin of 3.6% while remaining competitive on other tasks and maintaining inference throughput, highlighting its effectiveness and practicality.
>
---
#### [replaced 026] AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2506.06017v2](https://arxiv.org/pdf/2506.06017v2)**

> **作者:** Yu Li; Lehui Li; Zhihao Wu; Qingmin Liao; Jianye Hao; Kun Shao; Fengli Xu; Yong Li
>
> **备注:** AAAI-2026
>
> **摘要:** Large language model (LLM) agents have demonstrated strong capabilities across diverse domains, yet automated agent design remains a significant challenge. Current automated agent design approaches are often constrained by limited search spaces that primarily optimize workflows but fail to integrate crucial human-designed components like memory, planning, and tool use. Furthermore, these methods are hampered by high evaluation costs, as evaluating even a single new agent on a benchmark can require tens of dollars. The difficulty of this exploration is further exacerbated by inefficient search strategies that struggle to navigate the large design space effectively, making the discovery of novel agents a slow and resource-intensive process. To address these challenges, we propose AgentSwift, a novel framework for automated agent design. We formalize a hierarchical search space that jointly models agentic workflow and composable functional components. This structure moves beyond optimizing workflows alone by co-optimizing functional components, which enables the discovery of more complex and effective agent architectures. To make exploration within this expansive space feasible, we mitigate high evaluation costs by training a value model on a high-quality dataset, generated via a novel strategy combining combinatorial coverage and balanced Bayesian sampling for low-cost evaluation. Guiding the entire process is a hierarchical MCTS strategy, which is informed by uncertainty to efficiently navigate the search space. Evaluated across a comprehensive set of seven benchmarks spanning embodied, math, web, tool, and game domains, AgentSwift discovers agents that achieve an average performance gain of 8.34\% over both existing automated agent search methods and manually designed agents. Our framework serves as a launchpad for researchers to rapidly discover powerful agent architectures.
>
---
#### [replaced 027] CRISP: Persistent Concept Unlearning via Sparse Autoencoders
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2508.13650v2](https://arxiv.org/pdf/2508.13650v2)**

> **作者:** Tomer Ashuach; Dana Arad; Aaron Mueller; Martin Tutek; Yonatan Belinkov
>
> **备注:** 18 pages, 5 figures
>
> **摘要:** As large language models (LLMs) are increasingly deployed in real-world applications, the need to selectively remove unwanted knowledge while preserving model utility has become paramount. Recent work has explored sparse autoencoders (SAEs) to perform precise interventions on monosemantic features. However, most SAE-based methods operate at inference time, which does not create persistent changes in the model's parameters. Such interventions can be bypassed or reversed by malicious actors with parameter access. We introduce CRISP, a parameter-efficient method for persistent concept unlearning using SAEs. CRISP automatically identifies salient SAE features across multiple layers and suppresses their activations. We experiment with two LLMs and show that our method outperforms prior approaches on safety-critical unlearning tasks from the WMDP benchmark, successfully removing harmful knowledge while preserving general and in-domain capabilities. Feature-level analysis reveals that CRISP achieves semantically coherent separation between target and benign concepts, allowing precise suppression of the target features.
>
---
#### [replaced 028] Eliciting Reasoning in Language Models with Cognitive Tools
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.12115v2](https://arxiv.org/pdf/2506.12115v2)**

> **作者:** Brown Ebouky; Andrea Bartezzaghi; Mattia Rigotti
>
> **备注:** 25 pages, 2 figures
>
> **摘要:** The recent advent of reasoning models like OpenAI's o1 was met with excited speculation by the AI community about the mechanisms underlying these capabilities in closed models, followed by a rush of replication efforts, particularly from the open source community. These speculations were largely settled by the demonstration from DeepSeek-R1 that chains-of-thought and reinforcement learning (RL) can effectively replicate reasoning on top of base LLMs. However, it remains valuable to explore alternative methods for theoretically eliciting reasoning that could help elucidate the underlying mechanisms, as well as providing additional methods that may offer complementary benefits. Here, we build on the long-standing literature in cognitive psychology and cognitive architectures, which postulates that reasoning arises from the orchestrated, sequential execution of a set of modular, predetermined cognitive operations. Crucially, we implement this key idea within a modern agentic tool-calling framework. In particular, we endow an LLM with a small set of "cognitive tools" encapsulating specific reasoning operations, each executed by the LLM itself. Surprisingly, this simple strategy results in considerable gains in performance on standard mathematical reasoning benchmarks compared to base LLMs, for both closed and open-weight models. For instance, providing our "cognitive tools" to GPT-4.1 increases its pass@1 performance on AIME2024 from 32% to 53%, even surpassing the performance of o1-preview. In addition to its practical implications, this demonstration contributes to the debate regarding the role of post-training methods in eliciting reasoning in LLMs versus the role of inherent capabilities acquired during pre-training, and whether post-training merely uncovers these latent abilities.
>
---
#### [replaced 029] Atomic Calibration of LLMs in Long-Form Generations
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2410.13246v3](https://arxiv.org/pdf/2410.13246v3)**

> **作者:** Caiqi Zhang; Ruihan Yang; Zhisong Zhang; Xinting Huang; Sen Yang; Dong Yu; Nigel Collier
>
> **备注:** ACL 2025 KnowFM Oral / AACL-IJCNLP 2025
>
> **摘要:** Large language models (LLMs) often suffer from hallucinations, posing significant challenges for real-world applications. Confidence calibration, as an effective indicator of hallucination, is thus essential to enhance the trustworthiness of LLMs. Prior work mainly focuses on short-form tasks using a single response-level score (macro calibration), which is insufficient for long-form outputs that may contain both accurate and inaccurate claims. In this work, we systematically study atomic calibration, which evaluates factuality calibration at a fine-grained level by decomposing long responses into atomic claims. We further categorize existing confidence elicitation methods into discriminative and generative types, and propose two new confidence fusion strategies to improve calibration. Our experiments demonstrate that LLMs exhibit poorer calibration at the atomic level during long-form generation. More importantly, atomic calibration uncovers insightful patterns regarding the alignment of confidence methods and the changes of confidence throughout generation. This sheds light on future research directions for confidence estimation in long-form generation.
>
---
#### [replaced 030] OmniThink: Expanding Knowledge Boundaries in Machine Writing through Thinking
- **分类: cs.CL; cs.AI; cs.HC; cs.IR; cs.LG**

- **链接: [https://arxiv.org/pdf/2501.09751v5](https://arxiv.org/pdf/2501.09751v5)**

> **作者:** Zekun Xi; Wenbiao Yin; Jizhan Fang; Jialong Wu; Runnan Fang; Yong Jiang; Pengjun Xie; Fei Huang; Huajun Chen; Ningyu Zhang
>
> **备注:** EMNLP 2025
>
> **摘要:** Machine writing with large language models often relies on retrieval-augmented generation. However, these approaches remain confined within the boundaries of the model's predefined scope, limiting the generation of content with rich information. Specifically, vanilla-retrieved information tends to lack depth, novelty, and suffers from redundancy, which negatively impacts the quality of generated articles, leading to shallow, unoriginal, and repetitive outputs. To address these issues, we propose OmniThink, a slow-thinking machine writing framework that emulates the human-like process of iterative expansion and reflection. The core idea behind OmniThink is to simulate the cognitive behavior of learners as they slowly deepen their knowledge of the topics. Experimental results demonstrate that OmniThink improves the knowledge density of generated articles without compromising metrics such as coherence and depth. Human evaluations and expert feedback further highlight the potential of OmniThink to address real-world challenges in the generation of long-form articles. Code is available at https://github.com/zjunlp/OmniThink.
>
---
#### [replaced 031] Interpreting the Effects of Quantization on LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2508.16785v3](https://arxiv.org/pdf/2508.16785v3)**

> **作者:** Manpreet Singh; Hassan Sajjad
>
> **备注:** Accepted to AACL 2025 Main
>
> **摘要:** Quantization offers a practical solution to deploy LLMs in resource-constraint environments. However, its impact on internal representations remains understudied, raising questions about the reliability of quantized models. In this study, we employ a range of interpretability techniques to investigate how quantization affects model and neuron behavior. We analyze multiple LLMs under 4-bit and 8-bit quantization. Our findings reveal that the impact of quantization on model calibration is generally minor. Analysis of neuron activations indicates that the number of dead neurons, i.e., those with activation values close to 0 across the dataset, remains consistent regardless of quantization. In terms of neuron contribution to predictions, we observe that smaller full precision models exhibit fewer salient neurons, whereas larger models tend to have more, with the exception of Llama-2-7B. The effect of quantization on neuron redundancy varies across models. Overall, our findings suggest that effect of quantization may vary by model and tasks, however, we did not observe any drastic change which may discourage the use of quantization as a reliable model compression technique.
>
---
#### [replaced 032] OEMA: Ontology-Enhanced Multi-Agent Collaboration Framework for Zero-Shot Clinical Named Entity Recognition
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.15211v2](https://arxiv.org/pdf/2511.15211v2)**

> **作者:** Xinli Tao; Xin Dong; Xuezhong Zhou
>
> **备注:** 12 pages, 4 figures, 4 tables
>
> **摘要:** With the rapid expansion of unstructured clinical texts in electronic health records (EHRs), clinical named entity recognition (NER) has become a crucial technique for extracting medical information. However, traditional supervised models such as CRF and BioClinicalBERT suffer from high annotation costs. Although zero-shot NER based on large language models (LLMs) reduces the dependency on labeled data, challenges remain in aligning example selection with task granularity and effectively integrating prompt design with self-improvement frameworks. To address these limitations, we propose OEMA, a novel zero-shot clinical NER framework based on multi-agent collaboration. OEMA consists of three core components: (1) a self-annotator that autonomously generates candidate examples; (2) a discriminator that leverages SNOMED CT to filter token-level examples by clinical relevance; and (3) a predictor that incorporates entity-type descriptions to enhance inference accuracy. Experimental results on two benchmark datasets, MTSamples and VAERS, demonstrate that OEMA achieves state-of-the-art performance under exact-match evaluation. Moreover, under related-match criteria, OEMA performs comparably to the supervised BioClinicalBERT model while significantly outperforming the traditional CRF method. OEMA improves zero-shot clinical NER, achieving near-supervised performance under related-match criteria. Future work will focus on continual learning and open-domain adaptation to expand its applicability in clinical NLP.
>
---
#### [replaced 033] Verbalized Algorithms
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2509.08150v3](https://arxiv.org/pdf/2509.08150v3)**

> **作者:** Supriya Lall; Christian Farrell; Hari Pathanjaly; Marko Pavic; Sarvesh Chezhian; Masataro Asai
>
> **备注:** Accepted in NeurIPS 2025 Workshop on Efficient Reasoning
>
> **摘要:** Instead of querying LLMs in a one-shot manner and hoping to get the right answer for a reasoning task, we propose a paradigm we call \emph{verbalized algorithms} (VAs), which leverage classical algorithms with established theoretical understanding. VAs decompose a task into simple elementary operations on natural language strings that they should be able to answer reliably, and limit the scope of LLMs to only those simple tasks. For example, for sorting a series of natural language strings, \emph{verbalized sorting} uses an LLM as a binary comparison oracle in a known and well-analyzed sorting algorithm (e.g., bitonic sorting network). We demonstrate the effectiveness of this approach on sorting and clustering tasks.
>
---
#### [replaced 034] HAWAII: Hierarchical Visual Knowledge Transfer for Efficient Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2506.19072v2](https://arxiv.org/pdf/2506.19072v2)**

> **作者:** Yimu Wang; Mozhgan Nasr Azadani; Sean Sedwards; Krzysztof Czarnecki
>
> **备注:** NeurIPS 2025
>
> **摘要:** Improving the visual understanding ability of vision-language models (VLMs) is crucial for enhancing their performance across various tasks. While using multiple pretrained visual experts has shown great promise, it often incurs significant computational costs during training and inference. To address this challenge, we propose HAWAII, a novel framework that distills knowledge from multiple visual experts into a single vision encoder, enabling it to inherit the complementary strengths of several experts with minimal computational overhead. To mitigate conflicts among different teachers and switch between different teacher-specific knowledge, instead of using a fixed set of adapters for multiple teachers, we propose to use teacher-specific Low-Rank Adaptation (LoRA) adapters with a corresponding router. Each adapter is aligned with a specific teacher, avoiding noisy guidance during distillation. To enable efficient knowledge distillation, we propose fine-grained and coarse-grained distillation. At the fine-grained level, token importance scores are employed to emphasize the most informative tokens from each teacher adaptively. At the coarse-grained level, we summarize the knowledge from multiple teachers and transfer it to the student using a set of general-knowledge LoRA adapters with a router. Extensive experiments on various vision-language tasks demonstrate the superiority of HAWAII compared to popular open-source VLMs. The code is available at https://github.com/yimuwangcs/wise-hawaii.
>
---
#### [replaced 035] False Sense of Security: Why Probing-based Malicious Input Detection Fails to Generalize
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2509.03888v2](https://arxiv.org/pdf/2509.03888v2)**

> **作者:** Cheng Wang; Zeming Wei; Qin Liu; Muhao Chen
>
> **备注:** Withdrawn due to identified errors in the experimental procedure
>
> **摘要:** Large Language Models (LLMs) can comply with harmful instructions, raising serious safety concerns despite their impressive capabilities. Recent work has leveraged probing-based approaches to study the separability of malicious and benign inputs in LLMs' internal representations, and researchers have proposed using such probing methods for safety detection. We systematically re-examine this paradigm. Motivated by poor out-of-distribution performance, we hypothesize that probes learn superficial patterns rather than semantic harmfulness. Through controlled experiments, we confirm this hypothesis and identify the specific patterns learned: instructional patterns and trigger words. Our investigation follows a systematic approach, progressing from demonstrating comparable performance of simple n-gram methods, to controlled experiments with semantically cleaned datasets, to detailed analysis of pattern dependencies. These results reveal a false sense of security around current probing-based approaches and highlight the need to redesign both models and evaluation protocols, for which we provide further discussions in the hope of suggesting responsible further research in this direction. We have open-sourced the project at https://github.com/WangCheng0116/Why-Probe-Fails.
>
---
#### [replaced 036] Arg-LLaDA: Argument Summarization via Large Language Diffusion Models and Sufficiency-Aware Refinement
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2507.19081v4](https://arxiv.org/pdf/2507.19081v4)**

> **作者:** Hao Li; Yizheng Sun; Viktor Schlegel; Kailai Yang; Riza Batista-Navarro; Goran Nenadic
>
> **备注:** Preprint
>
> **摘要:** Argument summarization aims to generate concise, structured representations of complex, multi-perspective debates. While recent work has advanced the identification and clustering of argumentative components, the generation stage remains underexplored. Existing approaches typically rely on single-pass generation, offering limited support for factual correction or structural refinement. To address this gap, we introduce Arg-LLaDA, a novel large language diffusion framework that iteratively improves summaries via sufficiency-guided remasking and regeneration. Our method combines a flexible masking controller with a sufficiency-checking module to identify and revise unsupported, redundant, or incomplete spans, yielding more faithful, concise, and coherent outputs. Empirical results on two benchmark datasets demonstrate that Arg-LLaDA surpasses state-of-the-art baselines in 7 out of 10 automatic evaluation metrics. In addition, human evaluations reveal substantial improvements across core dimensions, coverage, faithfulness, and conciseness, validating the effectiveness of our iterative, sufficiency-aware generation strategy.
>
---
#### [replaced 037] HalluClean: A Unified Framework to Combat Hallucinations in LLMs
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2511.08916v3](https://arxiv.org/pdf/2511.08916v3)**

> **作者:** Yaxin Zhao; Yu Zhang
>
> **摘要:** Large language models (LLMs) have achieved impressive performance across a wide range of natural language processing tasks, yet they often produce hallucinated content that undermines factual reliability. To address this challenge, we introduce HalluClean, a lightweight and task-agnostic framework for detecting and correcting hallucinations in LLM-generated text. HalluClean adopts a reasoning-enhanced paradigm, explicitly decomposing the process into planning, execution, and revision stages to identify and refine unsupported claims. It employs minimal task-routing prompts to enable zero-shot generalization across diverse domains, without relying on external knowledge sources or supervised detectors. We conduct extensive evaluations on five representative tasks-question answering, dialogue, summarization, math word problems, and contradiction detection. Experimental results show that HalluClean significantly improves factual consistency and outperforms competitive baselines, demonstrating its potential to enhance the trustworthiness of LLM outputs in real-world applications.
>
---
#### [replaced 038] CaKE: Circuit-aware Editing Enables Generalizable Knowledge Learners
- **分类: cs.CL; cs.AI; cs.CV; cs.IR; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.16356v3](https://arxiv.org/pdf/2503.16356v3)**

> **作者:** Yunzhi Yao; Jizhan Fang; Jia-Chen Gu; Ningyu Zhang; Shumin Deng; Huajun Chen; Nanyun Peng
>
> **备注:** EMNLP 2025
>
> **摘要:** Knowledge Editing (KE) enables the modification of outdated or incorrect information in large language models (LLMs). While existing KE methods can update isolated facts, they often fail to generalize these updates to multi-hop reasoning tasks that rely on the modified knowledge. Through an analysis of reasoning circuits -- the neural pathways LLMs use for knowledge-based inference, we find that current layer-localized KE approaches (e.g., MEMIT, WISE), which edit only single or a few model layers, inadequately integrate updated knowledge into these reasoning pathways. To address this limitation, we present CaKE (Circuit-aware Knowledge Editing), a novel method that enhances the effective integration of updated knowledge in LLMs. By only leveraging a few curated data samples guided by our circuit-based analysis, CaKE stimulates the model to develop appropriate reasoning circuits for newly incorporated knowledge. Experiments show that CaKE enables more accurate and consistent use of edited knowledge across related reasoning tasks, achieving an average improvement of 20% in multi-hop reasoning accuracy on the MQuAKE dataset while requiring less memory than existing KE methods. We release the code and data in https://github.com/zjunlp/CaKE.
>
---
#### [replaced 039] ATLAS: A High-Difficulty, Multidisciplinary Benchmark for Frontier Scientific Reasoning
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2511.14366v2](https://arxiv.org/pdf/2511.14366v2)**

> **作者:** Hongwei Liu; Junnan Liu; Shudong Liu; Haodong Duan; Yuqiang Li; Mao Su; Xiaohong Liu; Guangtao Zhai; Xinyu Fang; Qianhong Ma; Taolin Zhang; Zihan Ma; Yufeng Zhao; Peiheng Zhou; Linchen Xiao; Wenlong Zhang; Shijie Zhou; Xingjian Ma; Siqi Sun; Jiaye Ge; Meng Li; Yuhong Liu; Jianxin Dong; Jiaying Li; Hui Wu; Hanwen Liang; Jintai Lin; Yanting Wang; Jie Dong; Tong Zhu; Tianfan Fu; Conghui He; Qi Zhang; Songyang Zhang; Lei Bai; Kai Chen
>
> **备注:** 39 pages
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has led to performance saturation on many established benchmarks, questioning their ability to distinguish frontier models. Concurrently, existing high-difficulty benchmarks often suffer from narrow disciplinary focus, oversimplified answer formats, and vulnerability to data contamination, creating a fidelity gap with real-world scientific inquiry. To address these challenges, we introduce ATLAS (AGI-Oriented Testbed for Logical Application in Science), a large-scale, high-difficulty, and cross-disciplinary evaluation suite composed of approximately 800 original problems. Developed by domain experts (PhD-level and above), ATLAS spans seven core scientific fields: mathematics, physics, chemistry, biology, computer science, earth science, and materials science. Its key features include: (1) High Originality and Contamination Resistance, with all questions newly created or substantially adapted to prevent test data leakage; (2) Cross-Disciplinary Focus, designed to assess models' ability to integrate knowledge and reason across scientific domains; (3) High-Fidelity Answers, prioritizing complex, open-ended answers involving multi-step reasoning and LaTeX-formatted expressions over simple multiple-choice questions; and (4) Rigorous Quality Control, employing a multi-stage process of expert peer review and adversarial testing to ensure question difficulty, scientific value, and correctness. We also propose a robust evaluation paradigm using a panel of LLM judges for automated, nuanced assessment of complex answers. Preliminary results on leading models demonstrate ATLAS's effectiveness in differentiating their advanced scientific reasoning capabilities. We plan to develop ATLAS into a long-term, open, community-driven platform to provide a reliable "ruler" for progress toward Artificial General Intelligence.
>
---
#### [replaced 040] LLMInit: A Free Lunch from Large Language Models for Selective Initialization of Recommendation
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **链接: [https://arxiv.org/pdf/2503.01814v2](https://arxiv.org/pdf/2503.01814v2)**

> **作者:** Weizhi Zhang; Liangwei Yang; Wooseong Yang; Henry Peng Zou; Yuqing Liu; Ke Xu; Sourav Medya; Philip S. Yu
>
> **备注:** Accepted in EMNLP 2025 Industry Track
>
> **摘要:** Collaborative filtering (CF) is widely adopted in industrial recommender systems (RecSys) for modeling user-item interactions across numerous applications, but often struggles with cold-start and data-sparse scenarios. Recent advancements in pre-trained large language models (LLMs) with rich semantic knowledge, offer promising solutions to these challenges. However, deploying LLMs at scale is hindered by their significant computational demands and latency. In this paper, we propose a novel and scalable LLM-RecSys framework, LLMInit, designed to integrate pretrained LLM embeddings into CF models through selective initialization strategies. Specifically, we identify the embedding collapse issue observed when CF models scale and match the large embedding sizes in LLMs and avoid the problem by introducing efficient sampling methods, including, random, uniform, and variance-based selections. Comprehensive experiments conducted on multiple real-world datasets demonstrate that LLMInit significantly improves recommendation performance while maintaining low computational costs, offering a practical and scalable solution for industrial applications. To facilitate industry adoption and promote future research, we provide open-source access to our implementation at https://github.com/DavidZWZ/LLMInit.
>
---
#### [replaced 041] One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image
- **分类: cs.CL; cs.CR; cs.CV; cs.IR**

- **链接: [https://arxiv.org/pdf/2504.02132v3](https://arxiv.org/pdf/2504.02132v3)**

> **作者:** Ezzeldin Shereen; Dan Ristea; Shae McFadden; Burak Hasircioglu; Vasilios Mavroudis; Chris Hicks
>
> **摘要:** Retrieval-augmented generation (RAG) is instrumental for inhibiting hallucinations in large language models (LLMs) through the use of a factual knowledge base (KB). Although PDF documents are prominent sources of knowledge, text-based RAG pipelines are ineffective at capturing their rich multi-modal information. In contrast, visual document RAG (VD-RAG) uses screenshots of document pages as the KB, which has been shown to achieve state-of-the-art results. However, by introducing the image modality, VD-RAG introduces new attack vectors for adversaries to disrupt the system by injecting malicious documents into the KB. In this paper, we demonstrate the vulnerability of VD-RAG to poisoning attacks targeting both retrieval and generation. We define two attack objectives and demonstrate that both can be realized by injecting only a single adversarial image into the KB. Firstly, we introduce a targeted attack against one or a group of queries with the goal of spreading targeted disinformation. Secondly, we present a universal attack that, for any potential user query, influences the response to cause a denial-of-service in the VD-RAG system. We investigate the two attack objectives under both white-box and black-box assumptions, employing a multi-objective gradient-based optimization approach as well as prompting state-of-the-art generative models. Using two visual document datasets, a diverse set of state-of-the-art retrievers (embedding models) and generators (vision language models), we show VD-RAG is vulnerable to poisoning attacks in both the targeted and universal settings, yet demonstrating robustness to black-box attacks in the universal setting.
>
---
#### [replaced 042] An Iterative Question-Guided Framework for Knowledge Base Question Answering
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2506.01784v4](https://arxiv.org/pdf/2506.01784v4)**

> **作者:** Shuai Wang; Yinan Yu
>
> **备注:** Accepted to the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025), Main Track
>
> **摘要:** Large Language Models (LLMs) excel in many natural language processing tasks but often exhibit factual inconsistencies in knowledge-intensive settings. Integrating external knowledge resources, particularly knowledge graphs (KGs), provides a transparent and updatable foundation for more reliable reasoning. Knowledge Base Question Answering (KBQA), which queries and reasons over KGs, is central to this effort, especially for complex, multi-hop queries. However, multi-hop reasoning poses two key challenges: (1)~maintaining coherent reasoning paths, and (2)~avoiding prematurely discarding critical multi-hop connections. To tackle these challenges, we introduce iQUEST, a question-guided KBQA framework that iteratively decomposes complex queries into simpler sub-questions, ensuring a structured and focused reasoning trajectory. Additionally, we integrate a Graph Neural Network (GNN) to look ahead and incorporate 2-hop neighbor information at each reasoning step. This dual approach strengthens the reasoning process, enabling the model to explore viable paths more effectively. Detailed experiments demonstrate the consistent improvement delivered by iQUEST across four benchmark datasets and four LLMs.
>
---
#### [replaced 043] KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2502.04420v5](https://arxiv.org/pdf/2502.04420v5)**

> **作者:** Xing Li; Zeyu Xing; Yiming Li; Linping Qu; Hui-Ling Zhen; Wulong Liu; Yiwu Yao; Sinno Jialin Pan; Mingxuan Yuan
>
> **备注:** Accepted by ICML25. Code: https://github.com/cmd2001/KVTuner
>
> **摘要:** KV cache quantization can improve Large Language Models (LLMs) inference throughput and latency in long contexts and large batch-size scenarios while preserving LLMs effectiveness. However, current methods have three unsolved issues: overlooking layer-wise sensitivity to KV cache quantization, high overhead of online fine-grained decision-making, and low flexibility to different LLMs and constraints. Therefore, we theoretically analyze the inherent correlation of layer-wise transformer attention patterns to KV cache quantization errors and study why key cache is generally more important than value cache for quantization error reduction. We further propose a simple yet effective framework KVTuner to adaptively search for the optimal hardware-friendly layer-wise KV quantization precision pairs for coarse-grained KV cache with multi-objective optimization and directly utilize the offline searched configurations during online inference. To reduce the computational cost of offline calibration, we utilize the intra-layer KV precision pair pruning and inter-layer clustering to reduce the search space. Experimental results show that we can achieve nearly lossless 3.25-bit mixed precision KV cache quantization for LLMs like Llama-3.1-8B-Instruct and 4.0-bit for sensitive models like Qwen2.5-7B-Instruct on mathematical reasoning tasks. The maximum inference throughput can be improved by 21.25\% compared with KIVI-KV8 quantization over various context lengths. Our code and searched configurations are available at https://github.com/cmd2001/KVTuner.
>
---
#### [replaced 044] TabDistill: Distilling Transformers into Neural Nets for Few-Shot Tabular Classification
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [https://arxiv.org/pdf/2511.05704v2](https://arxiv.org/pdf/2511.05704v2)**

> **作者:** Pasan Dissanayake; Sanghamitra Dutta
>
> **摘要:** Transformer-based models have shown promising performance on tabular data compared to their classical counterparts such as neural networks and Gradient Boosted Decision Trees (GBDTs) in scenarios with limited training data. They utilize their pre-trained knowledge to adapt to new domains, achieving commendable performance with only a few training examples, also called the few-shot regime. However, the performance gain in the few-shot regime comes at the expense of significantly increased complexity and number of parameters. To circumvent this trade-off, we introduce TabDistill, a new strategy to distill the pre-trained knowledge in complex transformer-based models into simpler neural networks for effectively classifying tabular data. Our framework yields the best of both worlds: being parameter-efficient while performing well with limited training data. The distilled neural networks surpass classical baselines such as regular neural networks, XGBoost and logistic regression under equal training data, and in some cases, even the original transformer-based models that they were distilled from.
>
---
#### [replaced 045] From Confidence to Collapse in LLM Factual Robustness
- **分类: cs.CL; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.16267v3](https://arxiv.org/pdf/2508.16267v3)**

> **作者:** Alina Fastowski; Bardh Prenkaj; Gjergji Kasneci
>
> **摘要:** Ensuring the robustness of factual knowledge in LLMs is critical for reliable applications in tasks such as question answering and reasoning. However, existing evaluation methods predominantly focus on performance-based metrics, often investigating from the perspective of prompt perturbations, which captures only the externally triggered side of knowledge robustness. To bridge this gap, we introduce a principled approach to measure factual robustness from the perspective of the generation process by analyzing token distribution entropy in combination with temperature scaling sensitivity. These two factors build the Factual Robustness Score (FRS), a novel metric which quantifies the stability of a fact against perturbations in decoding conditions, given its initial uncertainty. To validate our approach, we conduct extensive experiments on 5 LLMs across 3 closed-book QA datasets (SQuAD, TriviaQA, and HotpotQA). We show that factual robustness varies significantly -- smaller models report an FRS of $0.76$, larger ones $0.93$ -- with accuracy degrading by ~$60\%$ under increased uncertainty. These insights demonstrate how entropy and temperature scaling impact factual accuracy, and lay a foundation for developing more robust knowledge retention and retrieval in future models.
>
---
#### [replaced 046] Auditing Google's AI Overviews and Featured Snippets: A Case Study on Baby Care and Pregnancy
- **分类: cs.CL; cs.AI; cs.CY; cs.HC; cs.IR**

- **链接: [https://arxiv.org/pdf/2511.12920v2](https://arxiv.org/pdf/2511.12920v2)**

> **作者:** Desheng Hu; Joachim Baumann; Aleksandra Urman; Elsa Lichtenegger; Robin Forsberg; Aniko Hannak; Christo Wilson
>
> **备注:** 18 pages, 10 figures; to appear in AAAI ICWSM 2026
>
> **摘要:** Google Search increasingly surfaces AI-generated content through features like AI Overviews (AIO) and Featured Snippets (FS), which users frequently rely on despite having no control over their presentation. Through a systematic algorithm audit of 1,508 real baby care and pregnancy-related queries, we evaluate the quality and consistency of these information displays. Our robust evaluation framework assesses multiple quality dimensions, including answer consistency, relevance, presence of medical safeguards, source categories, and sentiment alignment. Our results reveal concerning gaps in information consistency, with information in AIO and FS displayed on the same search result page being inconsistent with each other in 33% of cases. Despite high relevance scores, both features critically lack medical safeguards (present in just 11% of AIO and 7% of FS responses). While health and wellness websites dominate source categories for both, AIO and FS, FS also often link to commercial sources. These findings have important implications for public health information access and demonstrate the need for stronger quality controls in AI-mediated health information. Our methodology provides a transferable framework for auditing AI systems across high-stakes domains where information quality directly impacts user well-being.
>
---
#### [replaced 047] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation
- **分类: cs.CV; cs.CL; cs.MM**

- **链接: [https://arxiv.org/pdf/2505.17534v3](https://arxiv.org/pdf/2505.17534v3)**

> **作者:** Jingjing Jiang; Chongjie Si; Jun Luo; Hanwang Zhang; Chao Ma
>
> **备注:** NeurIPS 2025
>
> **摘要:** This paper presents a pioneering exploration of reinforcement learning (RL) via group relative policy optimization for unified multimodal large language models (ULMs), aimed at simultaneously reinforcing generation and understanding capabilities. Through systematic pilot studies, we uncover the significant potential of ULMs to enable the synergistic co-evolution of dual capabilities within a shared policy optimization framework. Building on this insight, we introduce CoRL, a co-reinforcement learning framework comprising a unified RL stage for joint optimization and a refined RL stage for task-specific enhancement. With the proposed CoRL, our resulting model, ULM-R1, achieves average improvements of 7% on three text-to-image generation datasets and 23% on nine multimodal understanding benchmarks. These results demonstrate the effectiveness of CoRL and highlight the substantial benefit of reinforcement learning in facilitating cross-task synergy and optimization for ULMs. Code is available at https://github.com/mm-vl/ULM-R1.
>
---
#### [replaced 048] Can LLMs Replace Economic Choice Prediction Labs? The Case of Language-based Persuasion Games
- **分类: cs.LG; cs.AI; cs.CL; cs.GT; cs.HC**

- **链接: [https://arxiv.org/pdf/2401.17435v5](https://arxiv.org/pdf/2401.17435v5)**

> **作者:** Eilam Shapira; Omer Madmon; Roi Reichart; Moshe Tennenholtz
>
> **摘要:** Human choice prediction in economic contexts is crucial for applications in marketing, finance, public policy, and more. This task, however, is often constrained by the difficulties in acquiring human choice data. With most experimental economics studies focusing on simple choice settings, the AI community has explored whether LLMs can substitute for humans in these predictions and examined more complex experimental economics settings. However, a key question remains: can LLMs generate training data for human choice prediction? We explore this in language-based persuasion games, a complex economic setting involving natural language in strategic interactions. Our experiments show that models trained on LLM-generated data can effectively predict human behavior in these games and even outperform models trained on actual human data. Beyond data generation, we investigate the dual role of LLMs as both data generators and predictors, introducing a comprehensive empirical study on the effectiveness of utilizing LLMs for data generation, human choice prediction, or both. We then utilize our choice prediction framework to analyze how strategic factors shape decision-making, showing that interaction history (rather than linguistic sentiment alone) plays a key role in predicting human decision-making in repeated interactions. Particularly, when LLMs capture history-dependent decision patterns similarly to humans, their predictive success improves substantially. Finally, we demonstrate the robustness of our findings across alternative persuasion-game settings, highlighting the broader potential of using LLM-generated data to model human decision-making.
>
---
#### [replaced 049] Discriminating Form and Meaning in Multilingual Models with Minimal-Pair ABX Tasks
- **分类: cs.CL**

- **链接: [https://arxiv.org/pdf/2505.17747v4](https://arxiv.org/pdf/2505.17747v4)**

> **作者:** Maureen de Seyssel; Jie Chi; Skyler Seto; Maartje ter Hoeve; Masha Fedzechkina; Natalie Schluter
>
> **备注:** Comments: Published in EMNLP 2025. https://aclanthology.org/2025.emnlp-main.1210.pdf
>
> **摘要:** We introduce a set of training-free ABX-style discrimination tasks to evaluate how multilingual language models represent language identity (form) and semantic content (meaning). Inspired from speech processing, these zero-shot tasks measure whether minimal differences in representation can be reliably detected. This offers a flexible and interpretable alternative to probing. Applied to XLM-R (Conneau et al, 2020) across pretraining checkpoints and layers, we find that language discrimination declines over training and becomes concentrated in lower layers, while meaning discrimination strengthens over time and stabilizes in deeper layers. We then explore probing tasks, showing some alignment between our metrics and linguistic learning performance. Our results position ABX tasks as a lightweight framework for analyzing the structure of multilingual representations.
>
---
