# 自然语言处理 cs.CL

- **最新发布 152 篇**

- **更新 96 篇**

## 最新发布

#### [new 001] Parameter-Efficient Fine-Tuning for Medical Text Summarization: A Comparative Study of Lora, Prompt Tuning, and Full Fine-Tuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗文本摘要任务，旨在解决大模型微调计算资源消耗大的问题。通过比较LoRA、Prompt Tuning和全量微调方法，发现LoRA在参数效率和性能上更优。**

- **链接: [https://arxiv.org/pdf/2603.21970](https://arxiv.org/pdf/2603.21970)**

> **作者:** Ulugbek Shernazarov; Rostislav Svitsov; Bin Shi
>
> **备注:** 9 pages, 5 figures, presented at 6th International Conference on NLP & Text Mining (NLTM 2026), March 21-22, Sydney, Australia. Published in Computer Science & Information Technology (CS & IT), pp. 01-09, 2026
>
> **摘要:** Fine-tuning large language models for domain-specific tasks such as medical text summarization demands substantial computational resources. Parameter-efficient fine-tuning (PEFT) methods offer promising alternatives by updating only a small fraction of parameters. This paper compares three adaptation approaches-Low-Rank Adaptation (LoRA), Prompt Tuning, and Full Fine-Tuning-across the Flan-T5 model family on the PubMed medical summarization dataset. Through experiments with multiple random seeds, we demonstrate that LoRA consistently outperforms full fine-tuning, achieving 43.52 +/- 0.18 ROUGE-1 on Flan-T5-Large with only 0.6% trainable parameters compared to 40.67 +/- 0.21 for full fine-tuning. Sensitivity analyses examine the impact of LoRA rank and prompt token count. Our findings suggest the low-rank constraint provides beneficial regularization, challenging assumptions about the necessity of full parameter updates. Code is available at this https URL
>
---
#### [new 002] Explainable Semantic Textual Similarity via Dissimilar Span Detection
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，针对语义文本相似度（STS）可解释性不足的问题，提出DSD任务以检测文本中的语义差异片段，并构建了SSD数据集进行实验。**

- **链接: [https://arxiv.org/pdf/2603.21174](https://arxiv.org/pdf/2603.21174)**

> **作者:** Diego Miguel Lozano; Daryna Dementieva; Alexander Fraser
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** Semantic Textual Similarity (STS) is a crucial component of many Natural Language Processing (NLP) applications. However, existing approaches typically reduce semantic nuances to a single score, limiting interpretability. To address this, we introduce the task of Dissimilar Span Detection (DSD), which aims to identify semantically differing spans between pairs of texts. This can help users understand which particular words or tokens negatively affect the similarity score, or be used to improve performance in STS-dependent downstream tasks. Furthermore, we release a new dataset suitable for the task, the Span Similarity Dataset (SSD), developed through a semi-automated pipeline combining large language models (LLMs) with human verification. We propose and evaluate different baseline methods for DSD, both unsupervised, based on LIME, SHAP, LLMs, and our own method, as well as an additional supervised approach. While LLMs and supervised models achieve the highest performance, overall results remain low, highlighting the complexity of the task. Finally, we set up an additional experiment that shows how DSD can lead to increased performance in the specific task of paraphrase detection.
>
---
#### [new 003] Assessing the Ability of Neural TTS Systems to Model Consonant-Induced F0 Perturbation
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音合成任务，旨在评估神经文本转语音系统对辅音引发的音高扰动的建模能力。通过对比合成与自然语音，发现模型在高频词表现良好，但低频词泛化能力差。**

- **链接: [https://arxiv.org/pdf/2603.21078](https://arxiv.org/pdf/2603.21078)**

> **作者:** Tianle Yang; Chengzhe Sun; Phil Rose; Cassandra L. Jacobs; Siwei Lyu
>
> **备注:** Accepted for publication in Computer Speech & Language
>
> **摘要:** This study proposes a segmental-level prosodic probing framework to evaluate neural TTS models' ability to reproduce consonant-induced f0 perturbation, a fine-grained segmental-prosodic effect that reflects local articulatory mechanisms. We compare synthetic and natural speech realizations for thousands of words, stratified by lexical frequency, using Tacotron 2 and FastSpeech 2 trained on the same speech corpus (LJ Speech). These controlled analyses are then complemented by a large-scale evaluation spanning multiple advanced TTS systems. Results show accurate reproduction for high-frequency words but poor generalization to low-frequency items, suggesting that the examined TTS architectures rely more on lexical-level memorization than on abstract segmental-prosodic encoding. This finding highlights a limitation in such TTS systems' ability to generalize prosodic detail beyond seen data. The proposed probe offers a linguistically informed diagnostic framework that may inform future TTS evaluation methods, and has implications for interpretability and authenticity assessment in synthetic speech.
>
---
#### [new 004] User Preference Modeling for Conversational LLM Agents: Weak Rewards from Retrieval-Augmented Interaction
- **分类: cs.CL; cs.AI; cs.HC; cs.IR; stat.ML**

- **简介: 该论文属于对话系统任务，旨在解决用户偏好建模问题。通过VARS框架，利用弱奖励更新用户向量，提升交互效率与个性化，无需微调模型。**

- **链接: [https://arxiv.org/pdf/2603.20939](https://arxiv.org/pdf/2603.20939)**

> **作者:** Yuren Hao; Shuhaib Mehri; ChengXiang Zhai; Dilek Hakkani-Tür
>
> **备注:** 21 pages including appendices
>
> **摘要:** Large language models are increasingly used as personal assistants, yet most lack a persistent user model, forcing users to repeatedly restate preferences across sessions. We propose Vector-Adapted Retrieval Scoring (VARS), a pipeline-agnostic, frozen-backbone framework that represents each user with long-term and short-term vectors in a shared preference space and uses these vectors to bias retrieval scoring over structured preference memory. The vectors are updated online from weak scalar rewards from users' feedback, enabling personalization without per-user fine-tuning. We evaluate on \textsc{MultiSessionCollab}, an online multi-session collaboration benchmark with rich user preference profiles, across math and code tasks. Under frozen backbones, the main benefit of user-aware retrieval is improved interaction efficiency rather than large gains in raw task accuracy: our full VARS agent achieves the strongest overall performance, matches a strong Reflection baseline in task success, and reduces timeout rate and user effort. The learned long-term vectors also align with cross-user preference overlap, while short-term vectors capture session-specific adaptation, supporting the interpretability of the dual-vector design. Code, model, and data are available at this https URL.
>
---
#### [new 005] Entropy Alone is Insufficient for Safe Selective Prediction in LLMs
- **分类: cs.CL**

- **简介: 该论文属于语言模型安全任务，解决 hallucinations 问题。通过结合熵与正确性探测，提升选择性预测的可靠性与校准性能。**

- **链接: [https://arxiv.org/pdf/2603.21172](https://arxiv.org/pdf/2603.21172)**

> **作者:** Edward Phillips; Fredrik K. Gustafsson; Sean Wu; Anshul Thakur; David A. Clifton
>
> **摘要:** Selective prediction systems can mitigate harms resulting from language model hallucinations by abstaining from answering in high-risk cases. Uncertainty quantification techniques are often employed to identify such cases, but are rarely evaluated in the context of the wider selective prediction policy and its ability to operate at low target error rates. We identify a model-dependent failure mode of entropy-based uncertainty methods that leads to unreliable abstention behaviour, and address it by combining entropy scores with a correctness probe signal. We find that across three QA benchmarks (TriviaQA, BioASQ, MedicalQA) and four model families, the combined score generally improves both the risk--coverage trade-off and calibration performance relative to entropy-only baselines. Our results highlight the importance of deployment-facing evaluation of uncertainty methods, using metrics that directly reflect whether a system can be trusted to operate at a stated risk level.
>
---
#### [new 006] PAVE: Premise-Aware Validation and Editing for Retrieval-Augmented LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PAVE，用于检索增强语言模型的答案验证与编辑，解决答案与证据不一致的问题。通过提取前提并评估支持度，提升答案的证据一致性。**

- **链接: [https://arxiv.org/pdf/2603.20673](https://arxiv.org/pdf/2603.20673)**

> **作者:** Tianyi Huang; Caden Yang; Emily Yin; Eric Wang; Michael Zhang
>
> **摘要:** Retrieval-augmented language models can retrieve relevant evidence yet still commit to answers before explicitly checking whether the retrieved context supports the conclusion. We present PAVE (Premise-Grounded Answer Validation and Editing), an inference-time validation layer for evidence-grounded question answering. PAVE decomposes retrieved context into question-conditioned atomic facts, drafts an answer, scores how well that draft is supported by the extracted premises, and revises low-support outputs before finalization. The resulting trace makes answer commitment auditable at the level of explicit premises, support scores, and revision decisions. In controlled ablations with a fixed retriever and backbone, PAVE outperforms simpler post-retrieval baselines in two evidence-grounded QA settings, with the largest gain reaching 32.7 accuracy points on a span-grounded benchmark. We view these findings as proof-of-concept evidence that explicit premise extraction plus support-gated revision can strengthen evidence-grounded consistency in retrieval-augmented LLM systems.
>
---
#### [new 007] Beyond Test-Time Compute Strategies: Advocating Energy-per-Token in LLM Inference
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM推理中的能耗与准确率平衡问题。通过分析不同模型和策略的能耗与性能，提出能量效率指标及动态推理控制方法，以实现可持续AI部署。**

- **链接: [https://arxiv.org/pdf/2603.20224](https://arxiv.org/pdf/2603.20224)**

> **作者:** Patrick Wilhelm; Thorsten Wittkopp; Odej Kao
>
> **摘要:** Large Language Models (LLMs) demonstrate exceptional performance across diverse tasks but come with substantial energy and computational costs, particularly in request-heavy scenarios. In many real-world applications, the full scale and capabilities of LLMs are often unnecessary, as Small Language Models (SLMs) can provide accurate responses for simpler text generation tasks. When enhanced with advanced reasoning strategies, such as Chain-of-Thought (CoT) prompting or Majority Voting, SLMs can approach the performance of larger models while reducing overall computational requirements. However, these strategies can also introduce additional energy costs, creating an energy-accuracy trade-off. Our analysis examines these trade-offs in test-time compute strategies for smaller models compared to larger ones, using the MMLU benchmark. Additionally, we explore the input-output token dynamics of transformer architectures, which result in nonlinear hardware energy operation curves for LLMs. To bridge AI research with its physical impact, we propose \textit{energy efficiency metrics}, including Energy-per-Token, as complements to traditional accuracy benchmarks. Beyond model selection, we propose controlled reasoning in CoT token generation, using operating curves to regulate reasoning depth dynamically. This vision integrates a energy-aware routing mechanism, ensuring that model selection and inference strategies balance accuracy for sustainable AI deployment.
>
---
#### [new 008] KG-Hopper: Empowering Compact Open LLMs with Knowledge Graph Reasoning via Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱推理任务，旨在解决大型语言模型在多跳推理中的不足。通过强化学习框架KG-Hopper，实现高效、全局的多跳推理，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.21440](https://arxiv.org/pdf/2603.21440)**

> **作者:** Shuai Wang; Yinan Yu
>
> **备注:** Accepted to IJCNN 2026
>
> **摘要:** Large Language Models (LLMs) demonstrate impressive natural language capabilities but often struggle with knowledge-intensive reasoning tasks. Knowledge Base Question Answering (KBQA), which leverages structured Knowledge Graphs (KGs) exemplifies this challenge due to the need for accurate multi-hop reasoning. Existing approaches typically perform sequential reasoning steps guided by predefined pipelines, restricting flexibility and causing error cascades due to isolated reasoning at each step. To address these limitations, we propose KG-Hopper, a novel Reinforcement Learning (RL) framework that empowers compact open LLMs with the ability to perform integrated multi-hop KG reasoning within a single inference round. Rather than reasoning step-by-step, we train a Reasoning LLM that embeds the entire KG traversal and decision process into a unified ``thinking'' stage, enabling global reasoning over cross-step dependencies and dynamic path exploration with backtracking. Experimental results on eight KG reasoning benchmarks show that KG-Hopper, based on a 7B-parameter LLM, consistently outperforms larger multi-step systems (up to 70B) and achieves competitive performance with proprietary models such as GPT-3.5-Turbo and GPT-4o-mini, while remaining compact, open, and data-efficient. The code is publicly available at: this https URL.
>
---
#### [new 009] The Semantic Ladder: A Framework for Progressive Formalization of Natural Language Content for Knowledge Graphs and AI Systems
- **分类: cs.CL; cs.DB**

- **简介: 该论文提出“语义阶梯”框架，解决自然语言与形式化语义模型之间的鸿沟问题。通过逐步形式化数据，实现知识图谱和AI系统的高效集成。**

- **链接: [https://arxiv.org/pdf/2603.22136](https://arxiv.org/pdf/2603.22136)**

> **作者:** Lars Vogt
>
> **摘要:** Semantic data and knowledge infrastructures must reconcile two fundamentally different forms of representation: natural language, in which most knowledge is created and communicated, and formal semantic models, which enable machine-actionable integration, interoperability, and reasoning. Bridging this gap remains a central challenge, particularly when full semantic formalization is required at the point of data entry. Here, we introduce the Semantic Ladder, an architectural framework that enables the progressive formalization of data and knowledge. Building on the concept of modular semantic units as identifiable carriers of meaning, the framework organizes representations across levels of increasing semantic explicitness, ranging from natural language text snippets to ontology-based and higher-order logical models. Transformations between levels support semantic enrichment, statement structuring, and logical modelling while preserving semantic continuity and traceability. This approach enables the incremental construction of semantic knowledge spaces, reduces the semantic parsing burden, and supports the integration of heterogeneous representations, including natural language, structured semantic models, and vector-based embeddings. The Semantic Ladder thereby provides a foundation for scalable, interoperable, and AI-ready data and knowledge infrastructures.
>
---
#### [new 010] Reasoning Topology Matters: Network-of-Thought for Complex Reasoning Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于复杂推理任务，旨在解决传统推理结构的局限性。提出Network-of-Thought框架，通过图结构提升多源信息整合与回溯能力。**

- **链接: [https://arxiv.org/pdf/2603.20730](https://arxiv.org/pdf/2603.20730)**

> **作者:** Fan Huang
>
> **摘要:** Existing prompting paradigms structure LLM reasoning in limited topologies: Chain-of-Thought (CoT) produces linear traces, while Tree-of-Thought (ToT) performs branching search. Yet complex reasoning often requires merging intermediate results, revisiting hypotheses, and integrating evidence from multiple sources. We propose Network-of-Thought (NoT), a framework that models reasoning as a directed graph with typed nodes and edges, guided by a heuristic-based controller policy. Across four benchmarks (GSM8K, Game of 24, HotpotQA, ProofWriter) and three models (GPT-4o-mini, Llama-3.3-70B-Instruct, Qwen2.5-72B-Instruct), we investigate when network topology outperforms chain or tree structures, whether LLM-generated heuristics can guide graph-based reasoning search, and the computation-accuracy tradeoff across topologies, evaluating each method on accuracy, topology simplicity, and token efficiency. Our results show that CoT remains effective for sequential tasks with GPT-4o-mini (89.5\% on GSM8K), while NoT surpasses ToT on multi-hop reasoning (91.0\% vs.\ 88.0\% on HotpotQA with LLM-as-Judge). With 72B open-source models, NoT achieves the highest accuracy on GSM8K (91.5\%), and Qwen2.5-72B achieves the best multi-hop QA result overall (91.7\% on HotpotQA). Self-generated controller heuristics outperform fixed and random strategies on logical reasoning, with uncertainty-only weighting achieving 57.0\% on ProofWriter. We also find that evaluation methodology significantly impacts method rankings: string-match underestimates all methods on open-ended QA, with the largest gap for NoT, a pattern consistent across all three models (14--18 percentage point gap on HotpotQA).
>
---
#### [new 011] Thinking into the Future: Latent Lookahead Training for Transformers
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出一种名为“潜在前瞻训练”的方法，用于改进Transformer模型的生成能力。针对自回归模型在生成过程中缺乏前瞻性的问题，通过在特定位置进行多步潜在空间前瞻预测，提升模型在需要规划的任务中的表现。**

- **链接: [https://arxiv.org/pdf/2603.20219](https://arxiv.org/pdf/2603.20219)**

> **作者:** Lorenzo Noci; Gregor Bachmann; Seyed-Mohsen Moosavi-Dezfooli; Moin Nabi
>
> **摘要:** Autoregressive language models trained with next-token prediction generate text by sampling one discrete token at a time. Although very scalable, this objective forces the model to commit at every step, preventing it from exploring or reflecting upon multiple plausible continuations. Furthermore, the compute allocation across tokens is uniform; every token is formed based on a single forward-pass, potentially limiting the model's expressiveness in cases where difficult tokens require inherently more compute. Towards addressing these limitations, we introduce latent lookahead, a training strategy that enables models to "think" before generating: at selected positions in the sequence, before committing to the next token, the model performs a multi-step lookahead in latent space. More precisely, instead of sampling future tokens, we leverage the network's latent space by recursively feeding its hidden states back into the context for $\tau$ steps, investing more compute on predicting that token. This produces $\tau$ latent predictions that are supervised against the next $\tau$ ground-truth tokens, encouraging the model to "lookahead" and refine its prediction. We show that latent lookahead substantially outperforms both autoregressive and non-autoregressive baselines on planning tasks such as maze solving, Sudoku, and ProsQA, where foresight is essential.
>
---
#### [new 012] SLURP-TN : Resource for Tunisian Dialect Spoken Language Understanding
- **分类: cs.CL**

- **简介: 该论文属于语音理解任务，旨在解决低资源语言SLU数据缺失的问题。通过构建SLURP-TN数据集并开发相关模型，提升突尼斯方言的语音理解能力。**

- **链接: [https://arxiv.org/pdf/2603.21940](https://arxiv.org/pdf/2603.21940)**

> **作者:** Haroun Elleuch; Salima Mdhaffar; Yannick Estève; Fethi Bougares
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** Spoken Language Understanding (SLU) aims to extract the semantic information from the speech utterance of user queries. It is a core component in a task-oriented dialogue system. With the spectacular progress of deep neural network models and the evolution of pre-trained language models, SLU has obtained significant breakthroughs. However, only a few high-resource languages have taken advantage of this progress due to the absence of SLU resources. In this paper, we seek to mitigate this obstacle by introducing SLURP-TN. This dataset was created by recording 55 native speakers uttering sentences in Tunisian dialect, manually translated from six SLURP domains. The result is an SLU Tunisian dialect dataset that comprises 4165 sentences recorded into around 5 hours of acoustic material. We also develop a number of Automatic Speech Recognition and SLU models exploiting SLUTP-TN. The Dataset and baseline models are available at: this https URL.
>
---
#### [new 013] PROMPT2BOX: Uncovering Entailment Structure among LLM Prompts
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM提示间细粒度差异难以捕捉的问题。通过引入box嵌入方法，更准确地表征提示的语义和具体性，提升对模型弱点的分析效果。**

- **链接: [https://arxiv.org/pdf/2603.21438](https://arxiv.org/pdf/2603.21438)**

> **作者:** Neeladri Bhuiya; Shib Sankar Dasgupta; Andrew McCallum; Haw-Shiuan Chang
>
> **摘要:** To discover the weaknesses of LLMs, researchers often embed prompts into a vector space and cluster them to extract insightful patterns. However, vector embeddings primarily capture topical similarity. As a result, prompts that share a topic but differ in specificity, and consequently in difficulty, are often represented similarly, making fine-grained weakness analysis difficult. To address this limitation, we propose PROMPT2BOX, which embeds prompts into a box embedding space using a trained encoder. The encoder, trained on existing and synthesized datasets, outputs box embeddings that capture not only semantic similarity but also specificity relations between prompts (e.g., "writing an adventure story" is more specific than "writing a story"). We further develop a novel dimension reduction technique for box embeddings to facilitate dataset visualization and comparison. Our experiments demonstrate that box embeddings consistently capture prompt specificity better than vector baselines. On the downstream task of creating hierarchical clustering trees for 17 LLMs from the UltraFeedback dataset, PROMPT2BOX can identify 8.9\% more LLM weaknesses than vector baselines and achieves an approximately 33\% stronger correlation between hierarchical depth and instruction specificity.
>
---
#### [new 014] Decoding the decoder: Contextual sequence-to-sequence modeling for intracortical speech decoding
- **分类: cs.CL; cs.AI; cs.NE; q-bio.NC**

- **简介: 该论文属于脑机接口中的语音解码任务，旨在提升 intracortical 信号到语言的转换效果。通过引入上下文序列到序列模型和校准模块，提高解码准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.20246](https://arxiv.org/pdf/2603.20246)**

> **作者:** Michal Olak; Tommaso Boccato; Matteo Ferrante
>
> **摘要:** Speech brain--computer interfaces require decoders that translate intracortical activity into linguistic output while remaining robust to limited data and day-to-day variability. While prior high-performing systems have largely relied on framewise phoneme decoding combined with downstream language models, it remains unclear what contextual sequence-to-sequence decoding contributes to sublexical neural readout, robustness, and interpretability. We evaluated a multitask Transformer-based sequence-to-sequence model for attempted speech decoding from area 6v intracortical recordings. The model jointly predicts phoneme sequences, word sequences, and auxiliary acoustic features. To address day-to-day nonstationarity, we introduced the Neural Hammer Scalpel (NHS) calibration module, which combines global alignment with feature-wise modulation. We further analyzed held-out-day generalization and attention patterns in the encoder and decoders. On the Willett et al. dataset, the proposed model achieved a state-of-the-art phoneme error rate of 14.3%. Word decoding reached 25.6% WER with direct decoding and 19.4% WER with candidate generation and rescoring. NHS substantially improved both phoneme and word decoding relative to linear or no day-specific transform, while held-out-day experiments showed increasing degradation on unseen days with temporal distance. Attention visualizations revealed recurring temporal chunking in encoder representations and distinct use of these segments by phoneme and word decoders. These results indicate that contextual sequence-to-sequence modeling can improve the fidelity of neural-to-phoneme readout from intracortical speech signals and suggest that attention-based analyses can generate useful hypotheses about how neural speech evidence is segmented and accumulated over time.
>
---
#### [new 015] SozKZ: Training Efficient Small Language Models for Kazakh from Scratch
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出针对哈萨克语的小型语言模型SozKZ，解决低资源语言模型性能不足的问题。通过从零训练和专用分词器，提升模型效果。**

- **链接: [https://arxiv.org/pdf/2603.20854](https://arxiv.org/pdf/2603.20854)**

> **作者:** Saken Tukenov
>
> **备注:** 12 pages, 3 figures, 2 tables
>
> **摘要:** Kazakh, a Turkic language spoken by over 22 million people, remains underserved by existing multilingual language models, which allocate minimal capacity to low-resource languages and employ tokenizers ill-suited to agglutinative morphology. We present SozKZ, a family of Llama-architecture language models (50M-600M parameters) trained entirely from scratch on 9 billion tokens of Kazakh text with a dedicated 50K BPE tokenizer. We evaluate all models on three Kazakh benchmarks -- multiple-choice cultural QA, reading comprehension (Belebele), and topic classification (SIB-200) -- alongside five multilingual baselines ranging from 500M to 3B parameters. Our 600M model achieves 30.3% accuracy on Kazakh cultural QA, approaching the 32.0% of Llama-3.2-1B (2x larger), and 25.5% on SIB-200 topic classification, surpassing all evaluated multilingual models up to 2B parameters. We observe consistent scaling from 50M to 600M, with MC QA accuracy rising from 22.8% to 30.3%, suggesting that further scaling remains beneficial. These results demonstrate that small, dedicated models trained from scratch with a language-appropriate tokenizer offer a viable path for low-resource language technology, achieving competitive performance at a fraction of the computational cost. All models and the tokenizer are released under open licenses.
>
---
#### [new 016] Can ChatGPT Really Understand Modern Chinese Poetry?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在评估ChatGPT对现代中文诗歌的理解能力。通过多维度分析，发现其理解准确率达73%，但在诗意捕捉上仍有不足。**

- **链接: [https://arxiv.org/pdf/2603.20851](https://arxiv.org/pdf/2603.20851)**

> **作者:** Shanshan Wang; Derek F. Wong; Jingming Yao; Lidia S. Chao
>
> **备注:** Accepted by EACL 2026
>
> **摘要:** ChatGPT has demonstrated remarkable capabilities on both poetry generation and translation, yet its ability to truly understand poetry remains unexplored. Previous poetry-related work merely analyzed experimental outcomes without addressing fundamental issues of comprehension. This paper introduces a comprehensive framework for evaluating ChatGPT's understanding of modern poetry. We collaborated with professional poets to evaluate ChatGPT's interpretation of modern Chinese poems by different poets along multiple dimensions. Evaluation results show that ChatGPT's interpretations align with the original poets' intents in over 73% of the cases. However, its understanding in certain dimensions, particularly in capturing poeticity, proved to be less satisfactory. These findings highlight the effectiveness and necessity of our proposed framework. This study not only evaluates ChatGPT's ability to understand modern poetry but also establishes a solid foundation for future research on LLMs and their application to poetry-related tasks.
>
---
#### [new 017] Left Behind: Cross-Lingual Transfer as a Bridge for Low-Resource Languages in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究低资源语言在大语言模型中的表现，通过实验分析跨语言迁移效果，旨在解决模型对低资源语言支持不足的问题。**

- **链接: [https://arxiv.org/pdf/2603.21036](https://arxiv.org/pdf/2603.21036)**

> **作者:** Abdul-Salem Beibitkhan
>
> **摘要:** We investigate how large language models perform on low-resource languages by benchmarking eight LLMs across five experimental conditions in English, Kazakh, and Mongolian. Using 50 hand-crafted questions spanning factual, reasoning, technical, and culturally grounded categories, we evaluate 2,000 responses on accuracy, fluency, and completeness. We find a consistent performance gap of 13.8-16.7 percentage points between English and low-resource language conditions, with models maintaining surface-level fluency while producing significantly less accurate content. Cross-lingual transfer-prompting models to reason in English before translating back-yields selective gains for bilingual architectures (+2.2pp to +4.3pp) but provides no benefit to English-dominant models. Our results demonstrate that current LLMs systematically underserve low-resource language communities, and that effective mitigation strategies are architecture-dependent rather than universal.
>
---
#### [new 018] JUBAKU: An Adversarial Benchmark for Exposing Culturally Grounded Stereotypes in Japanese LLMs
- **分类: cs.CL**

- **简介: 该论文属于社会偏见检测任务，旨在解决非英语语言模型中文化特定偏见评估不足的问题。研究构建了针对日语文化的JUBAKU基准，通过人工设计的对话场景揭示模型中的隐性偏见。**

- **链接: [https://arxiv.org/pdf/2603.20581](https://arxiv.org/pdf/2603.20581)**

> **作者:** Taihei Shiotani; Masahiro Kaneko; Ayana Niwa; Yuki Maruyama; Daisuke Oba; Masanari Ohi; Naoaki Okazaki
>
> **摘要:** Social biases reflected in language are inherently shaped by cultural norms, which vary significantly across regions and lead to diverse manifestations of stereotypes. Existing evaluations of social bias in large language models (LLMs) for non-English contexts, however, often rely on translations of English benchmarks. Such benchmarks fail to reflect local cultural norms, including those found in Japanese. For instance, Western benchmarks may overlook Japan-specific stereotypes related to hierarchical relationships, regional dialects, or traditional gender roles. To address this limitation, we introduce Japanese cUlture adversarial BiAs benchmarK Under handcrafted creation (JUBAKU), a benchmark tailored to Japanese cultural contexts. JUBAKU uses adversarial construction to expose latent biases across ten distinct cultural categories. Unlike existing benchmarks, JUBAKU features dialogue scenarios hand-crafted by native Japanese annotators, specifically designed to trigger and reveal latent social biases in Japanese LLMs. We evaluated nine Japanese LLMs on JUBAKU and three others adapted from English benchmarks. All models clearly exhibited biases on JUBAKU, performing below the random baseline of 50% with an average accuracy of 23% (ranging from 13% to 33%), despite higher accuracy on the other benchmarks. Human annotators achieved 91% accuracy in identifying unbiased responses, confirming JUBAKU's reliability and its adversarial nature to LLMs.
>
---
#### [new 019] A Training-Free Regeneration Paradigm: Contrastive Reflection Memory Guided Self-Verification and Self-Improvement
- **分类: cs.CL**

- **简介: 该论文属于大语言模型优化任务，旨在解决验证与纠错效率低的问题。提出一种无需训练的再生范式，利用对比记忆进行自我验证与改进，提升准确性同时保持高效。**

- **链接: [https://arxiv.org/pdf/2603.20441](https://arxiv.org/pdf/2603.20441)**

> **作者:** Yuran Li; Di Wu; Benoit Boulet
>
> **备注:** 18 pages, 5 figures
>
> **摘要:** Verification-guided self-improvement has recently emerged as a promising approach to improving the accuracy of large language model (LLM) outputs. However, existing approaches face a trade-off between inference efficiency and accuracy: iterative verification-rectification is computationally expensive and prone to being trapped in faulty reasoning, while best-of-N selection requires extensive sampling without addressing internal model flaws. We propose a training-free regeneration paradigm that leverages an offline-curated contrastive Reflection Memory (RM) to provide corrective guidance, while regenerating from scratch helps break out of faulty reasoning. At inference time, the method performs RM-guided self-verification followed by a single RM-guided regeneration, avoiding both iterative correction and multi-sample selection. We evaluated our method on nine benchmarks that span algorithmic, reasoning, symbolic, and domain-specific tasks in both small- and large-scale LLMs. Experiment results show that our method outperforms prior methods while maintaining low computational cost.
>
---
#### [new 020] Select, Label, Evaluate: Active Testing in NLP
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决测试数据标注成本高的问题。通过主动测试框架，选择最具信息量的样本进行标注，显著减少标注量并保持评估精度。**

- **链接: [https://arxiv.org/pdf/2603.21840](https://arxiv.org/pdf/2603.21840)**

> **作者:** Antonio Purificato; Maria Sofia Bucarelli; Andrea Bacciu; Amin Mantrach; Fabrizio Silvestri
>
> **备注:** 27 pages, 6 figures
>
> **摘要:** Human annotation cost and time remain significant bottlenecks in Natural Language Processing (NLP), with test data annotation being particularly expensive due to the stringent requirement for low-error and high-quality labels necessary for reliable model evaluation. Traditional approaches require annotating entire test sets, leading to substantial resource requirements. Active Testing is a framework that selects the most informative test samples for annotation. Given a labeling budget, it aims to choose the subset that best estimates model performance while minimizing cost and human effort. In this work, we formalize Active Testing in NLP and we conduct an extensive benchmarking of existing approaches across 18 datasets and 4 embedding strategies spanning 4 different NLP tasks. The experiments show annotation reductions of up to 95%, with performance estimation accuracy difference from the full test set within 1%. Our analysis reveals variations in method effectiveness across different data characteristics and task types, with no single approach emerging as universally superior. Lastly, to address the limitation of requiring a predefined annotation budget in existing sample selection strategies, we introduce an adaptive stopping criterion that automatically determines the optimal number of samples.
>
---
#### [new 021] Greater accessibility can amplify discrimination in generative AI
- **分类: cs.CL**

- **简介: 论文探讨语音交互中生成式AI的性别偏见问题，属于AI公平性研究任务。它揭示语音接口可能加剧歧视，提出通过音调调整缓解偏见。**

- **链接: [https://arxiv.org/pdf/2603.22260](https://arxiv.org/pdf/2603.22260)**

> **作者:** Carolin Holtermann; Minh Duc Bui; Kaitlyn Zhou; Valentin Hofmann; Katharina von der Wense; Anne Lauscher
>
> **备注:** Preprint
>
> **摘要:** Hundreds of millions of people rely on large language models (LLMs) for education, work, and even healthcare. Yet these models are known to reproduce and amplify social biases present in their training data. Moreover, text-based interfaces remain a barrier for many, for example, users with limited literacy, motor impairments, or mobile-only devices. Voice interaction promises to expand accessibility, but unlike text, speech carries identity cues that users cannot easily mask, raising concerns about whether accessibility gains may come at the cost of equitable treatment. Here we show that audio-enabled LLMs exhibit systematic gender discrimination, shifting responses toward gender-stereotyped adjectives and occupations solely on the basis of speaker voice, and amplifying bias beyond that observed in text-based interaction. Thus, voice interfaces do not merely extend text models to a new modality but introduce distinct bias mechanisms tied to paralinguistic cues. Complementary survey evidence ($n=1,000$) shows that infrequent chatbot users are most hesitant to undisclosed attribute inference and most likely to disengage when such practices are revealed. To demonstrate a potential mitigation strategy, we show that pitch manipulation can systematically regulate gender-discriminatory outputs. Overall, our findings reveal a critical tension in AI development: efforts to expand accessibility through voice interfaces simultaneously create new pathways for discrimination, demanding that fairness and accessibility be addressed in tandem.
>
---
#### [new 022] The Anatomy of an Edit: Mechanism-Guided Activation Steering for Knowledge Editing
- **分类: cs.CL**

- **简介: 该论文属于知识编辑任务，旨在解决如何有效实施模型内的知识更新问题。通过分析编辑后的激活模式，提出MEGA方法实现无权重修改的知识编辑。**

- **链接: [https://arxiv.org/pdf/2603.20795](https://arxiv.org/pdf/2603.20795)**

> **作者:** Yuan Cao; Mingyang Wang; Hinrich Schütze
>
> **摘要:** Large language models (LLMs) are increasingly used as knowledge bases, but keeping them up to date requires targeted knowledge editing (KE). However, it remains unclear how edits are implemented inside the model once applied. In this work, we take a mechanistic view of KE using neuron-level knowledge attribution (NLKA). Unlike prior work that focuses on pre-edit causal tracing and localization, we use post-edit attribution -- contrasting successful and failed edits -- to isolate the computations that shift when an edit succeeds. Across representative KE methods, we find a consistent pattern: mid-to-late attention predominantly promotes the new target, while attention and FFN modules cooperate to suppress the original fact. Motivated by these findings, we propose MEGA, a MEchanism-Guided Activation steering method that performs attention-residual interventions in attribution-aligned regions without modifying model weights. On CounterFact and Popular, MEGA achieves strong editing performance across KE metrics on GPT2-XL and LLaMA2-7B. Overall, our results elevate post-edit attribution from analysis to engineering signal: by pinpointing where and how edits take hold, it powers MEGA to deliver reliable, architecture-agnostic knowledge edits.
>
---
#### [new 023] An experimental study of KV cache reuse strategies in chunk-level caching systems
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究CLC系统中KV缓存复用策略。解决跨块注意力依赖缺失导致的输出质量下降问题，通过分析现有方法并提出融合设计提升准确性。**

- **链接: [https://arxiv.org/pdf/2603.20218](https://arxiv.org/pdf/2603.20218)**

> **作者:** Samuel Cestola; Tianxiang Xia; Zheng Weiyan; Zheng Pengfei; Diego Didona
>
> **摘要:** Retrieval-augmented generation improves large language models' accuracy by adding relevant retrieved text to the prompt. Chunk level caching (CLC) accelerates inference by precomputing KV caches for these retrieved chunks and reusing them. However, these caches miss cross-attention dependencies between chunks, which can reduce output quality. Several methods try to improve CLC accuracy using different techniques. We make two main contributions. First, we show that existing CLC approaches have fundamental limitations that limit their accuracy or their applicability. We back this conclusion with an extensive CLC system experimental evaluation. Second, we observe that existing CLC techniques are complementary. We leverage this insight to propose a new CLC design that carefully combines them and achieves better accuracy.
>
---
#### [new 024] Enhancing Document-Level Machine Translation via Filtered Synthetic Corpora and Two-Stage LLM Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文档级机器翻译任务，旨在解决数据稀缺和生成幻觉问题。通过合成数据增强与两阶段微调提升翻译质量。**

- **链接: [https://arxiv.org/pdf/2603.22186](https://arxiv.org/pdf/2603.22186)**

> **作者:** Ireh Kim; Tesia Sker; Chanwoo Kim
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** In Machine Translation, Large Language Models (LLMs) have generally underperformed compared to conventional encoder-decoder systems and thus see limited adoption. However, LLMs excel at modeling contextual information, making them a natural fit for document-level translation tasks where coherence across sentences is crucial. Despite this potential, document-level MT with LLMs faces two key challenges: (1) the scarcity of large-scale, high-quality document-level parallel data; and (2) the propensity of LLMs to introduce hallucinations and omissions during generation. To address these challenges, we propose a two-stage fine-tuning strategy leveraging LLM-augmented document-level data. First, we augment data by converting summarization data into document-level parallel data using a LLM, and then filter it using multiple metrics, leveraging sacreBLEU, COMET, and LaBSE-based cosine similarity-to improve data quality. Finally, we employ a two-stage fine-tuning strategy: first fine-tuning on the abundant sentence-level MT resources, and then on the filtered document-level corpus.
>
---
#### [new 025] Gumbel Distillation for Parallel Text Generation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于文本生成任务，旨在解决并行解码模型生成质量低的问题。通过引入Gumbel Distillation技术，提升并行模型对词序列联合分布的建模能力。**

- **链接: [https://arxiv.org/pdf/2603.22216](https://arxiv.org/pdf/2603.22216)**

> **作者:** Chi Zhang; Xixi Hu; Bo Liu; Qiang Liu
>
> **备注:** ICLR 2026
>
> **摘要:** The slow, sequential nature of autoregressive (AR) language models has driven the adoption of parallel decoding methods. However, these non-AR models often sacrifice generation quality as they struggle to model the complex joint distribution of token sequences. To narrow this performance gap, we introduce Gumbel Distillation, a novel distillation technique that enables parallel decoders to learn this distribution effectively. Our method leverages the Gumbel-Max trick to create a deterministic mapping from a latent Gumbel noise space to the output tokens of a high-performing AR teacher. As a model-agnostic technique, Gumbel Distillation seamlessly integrates with diverse parallel decoding architectures, including MDLM and BD3-LM. Experiments on LM1B and OpenWebText show that Gumbel Distillation substantially improves the generation quality of parallel language models, achieving a 30.0% improvement in MAUVE score and 10.5% in generative perplexity over MDLM trained on OpenWebText dataset. Code available at this https URL.
>
---
#### [new 026] Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对多选和配对评估任务中的选择偏差问题，提出PA-GRPO方法，通过强化学习提升模型的语义一致性。**

- **链接: [https://arxiv.org/pdf/2603.21016](https://arxiv.org/pdf/2603.21016)**

> **作者:** Jinquan Zheng; Jia Yuan; Jiacheng Yao; Chenyang Gu; Pujun Zheng; Guoxiu He
>
> **备注:** 16 pages, 3 figures, 5 tables
>
> **摘要:** Large language models (LLMs) used for multiple-choice and pairwise evaluation tasks often exhibit selection bias due to non-semantic factors like option positions and label symbols. Existing inference-time debiasing is costly and may harm reasoning, while pointwise training ignores that the same question should yield consistent answers across permutations. To address this issue, we propose Permutation-Aware Group Relative Policy Optimization (PA-GRPO), which mitigates selection bias by enforcing permutation-consistent semantic reasoning. PA-GRPO constructs a permutation group for each instance by generating multiple candidate permutations, and optimizes the model using two complementary mechanisms: (1) cross-permutation advantage, which computes advantages relative to the mean reward over all permutations of the same instance, and (2) consistency-aware reward, which encourages the model to produce consistent decisions across different permutations. Experimental results demonstrate that PA-GRPO outperforms strong baselines across seven benchmarks, substantially reducing selection bias while maintaining high overall performance. The code will be made available on Github (this https URL).
>
---
#### [new 027] A Comparative Analysis of LLM Memorization at Statistical and Internal Levels: Cross-Model Commonalities and Model-Specific Signatures
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型研究任务，旨在探讨LLM记忆行为的共性与特性。通过分析多个模型系列，揭示记忆率与模型规模的关系及内部机制，解决跨模型记忆理解不一致的问题。**

- **链接: [https://arxiv.org/pdf/2603.21658](https://arxiv.org/pdf/2603.21658)**

> **作者:** Bowen Chen; Namgi Han; Yusuke Miyao
>
> **备注:** 8 pages of main content, in conference submission, other contents are references and extra appendix
>
> **摘要:** Memorization is a fundamental component of intelligence for both humans and LLMs. However, while LLM performance scales rapidly, our understanding of memorization lags. Due to limited access to the pre-training data of LLMs, most previous studies focus on a single model series, leading to isolated observations among series, making it unclear which findings are general or specific. In this study, we collect multiple model series (Pythia, OpenLLaMa, StarCoder, OLMo1/2/3) and analyze their shared or unique memorization behavior at both the statistical and internal levels, connecting individual observations while showing new findings. At the statistical level, we reveal that the memorization rate scales log-linearly with model size, and memorized sequences can be further compressed. Further analysis demonstrated a shared frequency and domain distribution pattern for memorized sequences. However, different models also show individual features under the above observations. At the internal level, we find that LLMs can remove certain injected perturbations, while memorized sequences are more sensitive. By decoding middle layers and attention head ablation, we revealed the general decoding process and shared important heads for memorization. However, the distribution of those important heads differs between families, showing a unique family-level feature. Through bridging various experiments and revealing new findings, this study paves the way for a universal and fundamental understanding of memorization in LLM.
>
---
#### [new 028] RLVR Training of LLMs Does Not Improve Thinking Ability for General QA: Evaluation Method and a Simple Solution
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究RLVR对通用问答任务的影响，发现其效果有限。提出跨生成评估框架，验证RLVR在GQA上的不足，并引入START方法提升思考质量与答案准确性。任务为语言模型训练，解决RLVR在GQA中效果不佳的问题。**

- **链接: [https://arxiv.org/pdf/2603.20799](https://arxiv.org/pdf/2603.20799)**

> **作者:** Kaiyuan Li; Jing-Cheng Pang; Yang Yu
>
> **摘要:** Reinforcement learning from verifiable rewards (RLVR) stimulates the thinking processes of large language models (LLMs), substantially enhancing their reasoning abilities on verifiable tasks. It is often assumed that similar gains should transfer to general question answering (GQA), but this assumption has not been thoroughly validated. To assess whether RLVR automatically improves LLM performance on GQA, we propose a Cross-Generation evaluation framework that measures the quality of intermediate reasoning by feeding the generated thinking context into LLMs of varying capabilities. Our evaluation leads to a discouraging finding: the efficacy of the thinking process on GQA tasks is markedly lower than on verifiable tasks, suggesting that explicit training on GQA remains necessary in addition to training on verifiable tasks. We further observe that direct RL training on GQA is less effective than RLVR. Our hypothesis is that, whereas verifiable tasks demand robust logical chains to obtain high rewards, GQA tasks often admit shortcuts to high rewards without cultivating high-quality thinking. To avoid possible shortcuts, we introduce a simple method, Separated Thinking And Response Training (START), which first trains only the thinking process, using rewards defined on the final answer. We show that START improves both the quality of thinking and the final answer across several GQA benchmarks and RL algorithms.
>
---
#### [new 029] Locally Coherent Parallel Decoding in Diffusion Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于代码生成任务，解决并行采样导致的语法不一致问题。通过引入辅助自回归模型，实现局部依赖建模，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2603.20216](https://arxiv.org/pdf/2603.20216)**

> **作者:** Michael Hersche; Nicolas Menet; Ronan Tanios; Abbas Rahimi
>
> **摘要:** Diffusion language models (DLMs) have emerged as a promising alternative to autoregressive (AR) models, offering sub-linear generation latency and bidirectional capabilities that are particularly appealing for code generation and editing. Achieving sub-linear latency in discrete DLMs requires predicting multiple tokens in parallel. However, standard DLMs sample tokens independently from conditional marginal distributions, failing to capture the joint dependencies among concurrently generated tokens. As a result, they often lead to syntactic inconsistencies and break multi-token structures. In this work, we introduce CoDiLA (Coherent Diffusion with Local Autoregression), a method that reconciles parallel sampling with local dependency modeling. Rather than forcing the DLM to resolve fine-grained syntax, CoDiLA delegates local decoding to a small, auxiliary AR model operating on the diffusion latents. This design allows for parallel block generation while ensuring sequential validity within each block and maintaining core DLM capabilities, including bidirectional modeling across blocks. We demonstrate that using a highly compact auxiliary AR model (e.g., 0.6B parameters) effectively eliminates coherence artifacts, establishing a new Pareto frontier for accuracy and speed in code generation benchmarks.
>
---
#### [new 030] Adapting Self-Supervised Speech Representations for Cross-lingual Dysarthria Detection in Parkinson's Disease
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于跨语言语音识别任务，旨在解决帕金森病患者发音障碍检测中数据不足的问题。通过语言迁移方法调整语音表示，提升跨语言检测效果。**

- **链接: [https://arxiv.org/pdf/2603.22225](https://arxiv.org/pdf/2603.22225)**

> **作者:** Abner Hernandez; Eunjung Yeo; Kwanghee Choi; Chin-Jou Li; Zhengjun Yue; Rohan Kumar Das; Jan Rusz; Mathew Magimai Doss; Juan Rafael Orozco-Arroyave; Tomás Arias-Vergara; Andreas Maier; Elmar Nöth; David R. Mortensen; David Harwath; Paula Andrea Perez-Toro
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** The limited availability of dysarthric speech data makes cross-lingual detection an important but challenging problem. A key difficulty is that speech representations often encode language-dependent structure that can confound dysarthria detection. We propose a representation-level language shift (LS) that aligns source-language self-supervised speech representations with the target-language distribution using centroid-based vector adaptation estimated from healthy-control speech. We evaluate the approach on oral DDK recordings from Parkinson's disease speech datasets in Czech, German, and Spanish under both cross-lingual and multilingual settings. LS substantially improves sensitivity and F1 in cross-lingual settings, while yielding smaller but consistent gains in multilingual settings. Representation analysis further shows that LS reduces language identity in the embedding space, supporting the interpretation that LS removes language-dependent structure.
>
---
#### [new 031] ViCLSR: A Supervised Contrastive Learning Framework with Natural Language Inference for Natural Language Understanding Tasks
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对越南语自然语言理解任务，解决数据稀缺问题，提出ViCLSR框架，通过监督对比学习提升句子表示效果。**

- **链接: [https://arxiv.org/pdf/2603.21084](https://arxiv.org/pdf/2603.21084)**

> **作者:** Tin Van Huynh; Kiet Van Nguyen; Ngan Luu-Thuy Nguyen
>
> **摘要:** High-quality text representations are crucial for natural language understanding (NLU), but low-resource languages like Vietnamese face challenges due to limited annotated data. While pre-trained models like PhoBERT and CafeBERT perform well, their effectiveness is constrained by data scarcity. Contrastive learning (CL) has recently emerged as a promising approach for improving sentence representations, enabling models to effectively distinguish between semantically similar and dissimilar sentences. We propose ViCLSR (Vietnamese Contrastive Learning for Sentence Representations), a novel supervised contrastive learning framework specifically designed to optimize sentence embeddings for Vietnamese, leveraging existing natural language inference (NLI) datasets. Additionally, we propose a process to adapt existing Vietnamese datasets for supervised learning, ensuring compatibility with CL methods. Our experiments demonstrate that ViCLSR significantly outperforms the powerful monolingual pre-trained model PhoBERT on five benchmark NLU datasets such as ViNLI (+6.97% F1), ViWikiFC (+4.97% F1), ViFactCheck (+9.02% F1), UIT-ViCTSD (+5.36% F1), and ViMMRC2.0 (+4.33% Accuracy). ViCLSR shows that supervised contrastive learning can effectively address resource limitations in Vietnamese NLU tasks and improve sentence representation learning for low-resource languages. Furthermore, we conduct an in-depth analysis of the experimental results to uncover the factors contributing to the superior performance of contrastive learning models. ViCLSR is released for research purposes in advancing natural language processing tasks.
>
---
#### [new 032] Probing How Scalable Table Data Enhances General Long-Context Reasoning
- **分类: cs.CL**

- **简介: 该论文研究如何利用结构化表格数据提升大语言模型的长上下文推理能力。针对长上下文推理效果不佳的问题，通过分析表格数据的依赖结构，提出一种合成高质量表格数据的方法，显著提升了模型表现。**

- **链接: [https://arxiv.org/pdf/2603.21719](https://arxiv.org/pdf/2603.21719)**

> **作者:** Huaibing Xie; Guoliang Zhao; Yang Liu; Shihan Dou; Siming Huang; Yanling Xiao; Shaolei Wang; Yiting Liu; Cheng Zhang; Shaofan Liu; Pluto Zhou
>
> **摘要:** As real-world tasks grow increasingly complex, long-context reasoning has become a core capability for Large Language Models (LLMs). However, few studies explore which data types are effective for long-context reasoning and why. We find that structured table data with periodic structures shows strong potential for long-context reasoning. Motivated by this observation, we mathematically analyze tabular dependency structures using mutual information, revealing periodic non-vanishing dependencies in table data. Furthermore, we systematically analyze the capabilities of structured table data, conduct relevant scaling experiments, and validate its underlying mechanisms for enhancing long-context reasoning, yielding several meaningful insights. Leveraging these insights, we propose a simple yet scalable pipeline(TableLong) for synthesizing high-quality, diverse, and verifiable structured table data to boost long-context reasoning via RL. Extensive experimental results demonstrate that table data significantly enhances the long-context reasoning capability of LLMs across multiple long-context benchmarks (+8.24\% on average), and even improves performance on out-of-domain benchmarks (+8.06\% on average). We hope that our insights provide practical guidance for effective post-training data to enhance long-context reasoning in LLMs.
>
---
#### [new 033] Code-MIE: A Code-style Model for Multimodal Information Extraction with Scene Graph and Entity Attribute Knowledge Enhancement
- **分类: cs.CL**

- **简介: 论文提出Code-MIE框架，解决多模态信息抽取问题。通过代码风格的输入输出模板，融合实体属性和场景图，提升抽取效果。**

- **链接: [https://arxiv.org/pdf/2603.20781](https://arxiv.org/pdf/2603.20781)**

> **作者:** Jiang Liu; Ge Qiu; Hao Fei; Dongdong Xie; Jinbo Li; Fei Li; Chong Teng; Donghong Ji
>
> **摘要:** With the rapid development of large language models (LLMs), more and more researchers have paid attention to information extraction based on LLMs. However, there are still some spaces to improve in the existing related methods. First, existing multimodal information extraction (MIE) methods usually employ natural language templates as the input and output of LLMs, which mismatch with the characteristics of information tasks that mostly include structured information such as entities and relations. Second, although a few methods have adopted structured and more IE-friendly code-style templates, they just explored their methods on text-only IE rather than multimodal IE. Moreover, their methods are more complex in design, requiring separate templates to be designed for each task. In this paper, we propose a Code-style Multimodal Information Extraction framework (Code-MIE) which formalizes MIE as unified code understanding and generation. Code-MIE has the following novel designs: (1) Entity attributes such as gender, affiliation are extracted from the text to guide the model to understand the context and role of entities. (2) Images are converted into scene graphs and visual features to incorporate rich visual information into the model. (3) The input template is constructed as a Python function, where entity attributes, scene graphs and raw text compose of the function parameters. In contrast, the output template is formalized as Python dictionaries containing all extraction results such as entities, relations, etc. To evaluate Code-MIE, we conducted extensive experiments on the M$^3$D, Twitter-15, Twitter-17, and MNRE datasets. The results show that our method achieves state-of-the-art performance compared to six competing baseline models, with 61.03\% and 60.49\% on the English and Chinese datasets of M$^3$D, and 76.04\%, 88.07\%, and 73.94\% on the other three datasets.
>
---
#### [new 034] Optimizing Multi-Agent Weather Captioning via Text Gradient Descent: A Training-Free Approach with Consensus-Aware Gradient Fusion
- **分类: cs.CL**

- **简介: 该论文属于气象文本生成任务，解决天气数据转自然语言描述的问题。提出WeatherTGD框架，通过多代理协作生成精准且有领域深度的天气描述。**

- **链接: [https://arxiv.org/pdf/2603.21673](https://arxiv.org/pdf/2603.21673)**

> **作者:** Shixu Liu
>
> **备注:** Preprint and under consideration
>
> **摘要:** Generating interpretable natural language captions from weather time series data remains a significant challenge at the intersection of meteorological science and natural language processing. While recent advances in Large Language Models (LLMs) have demonstrated remarkable capabilities in time series forecasting and analysis, existing approaches either produce numerical predictions without human-accessible explanations or generate generic descriptions lacking domain-specific depth. We introduce WeatherTGD, a training-free multi-agent framework that reinterprets collaborative caption refinement through the lens of Text Gradient Descent (TGD). Our system deploys three specialized LLM agents including a Statistical Analyst, a Physics Interpreter, and a Meteorology Expert that generate domain-specific textual gradients from weather time series observations. These gradients are aggregated through a novel Consensus-Aware Gradient Fusion mechanism that extracts common signals while preserving unique domain perspectives. The fused gradients then guide an iterative refinement process analogous to gradient descent, where each LLM-generated feedback signal updates the caption toward an optimal solution. Experiments on real-world meteorological datasets demonstrate that WeatherTGD achieves significant improvements in both LLM-based evaluation and human expert evaluation, substantially outperforming existing multi-agent baselines while maintaining computational efficiency through parallel agent execution.
>
---
#### [new 035] FinReflectKG -- HalluBench: GraphRAG Hallucination Benchmark for Financial Question Answering Systems
- **分类: cs.CL; q-fin.CP**

- **简介: 该论文属于金融问答系统任务，旨在解决知识图谱增强型问答系统中的幻觉检测问题。通过构建基准数据集并评估多种检测方法，提升系统可靠性。**

- **链接: [https://arxiv.org/pdf/2603.20252](https://arxiv.org/pdf/2603.20252)**

> **作者:** Mahesh Kumar; Bhaskarjit Sarmah; Stefano Pasquali
>
> **摘要:** As organizations increasingly integrate AI-powered question-answering systems into financial information systems for compliance, risk assessment, and decision support, ensuring the factual accuracy of AI-generated outputs becomes a critical engineering challenge. Current Knowledge Graph (KG)-augmented QA systems lack systematic mechanisms to detect hallucinations - factually incorrect outputs that undermine reliability and user trust. We introduce FinBench-QA-Hallucination, a benchmark for evaluating hallucination detection methods in KG-augmented financial QA over SEC 10-K filings. The dataset contains 755 annotated examples from 300 pages, each labeled for groundedness using a conservative evidence-linkage protocol requiring support from both textual chunks and extracted relational triplets. We evaluate six detection approaches - LLM judges, fine-tuned classifiers, Natural Language Inference (NLI) models, span detectors, and embedding-based methods under two conditions: with and without KG triplets. Results show that LLM-based judges and embedding approaches achieve the highest performance (F1: 0.82-0.86) under clean conditions. However, most methods degrade significantly when noisy triplets are introduced, with Matthews Correlation Coefficient (MCC) dropping 44-84 percent, while embedding methods remain relatively robust with only 9 percent degradation. Statistical tests (Cochran's Q and McNemar) confirm significant performance differences (p < 0.001). Our findings highlight vulnerabilities in current KG-augmented systems and provide insights for building reliable financial information systems, where hallucinations can lead to regulatory violations and flawed decisions. The benchmark also offers a framework for integrating AI reliability evaluation into information system design across other high-stakes domains such as healthcare, legal, and government.
>
---
#### [new 036] Mitigating Shortcut Reasoning in Language Models: A Gradient-Aware Training Approach
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决语言模型依赖快捷方式而非逻辑推理的问题。提出SART方法，通过梯度分析检测并减少快捷样本的影响，提升模型的推理能力和泛化性。**

- **链接: [https://arxiv.org/pdf/2603.20899](https://arxiv.org/pdf/2603.20899)**

> **作者:** Hongyu Cao; Kunpeng Liu; Dongjie Wang; Yanjie Fu
>
> **备注:** 12 pages, 2 figures. Preprint. Experiments on synthetic reasoning benchmarks. Code available
>
> **摘要:** Large language models exhibit strong reasoning capabilities, yet often rely on shortcuts such as surface pattern matching and answer memorization rather than genuine logical inference. We propose Shortcut-Aware Reasoning Training (SART), a gradient-aware framework that detects and mitigates shortcut-promoting samples via ShortcutScore and gradient surgery. Our method identifies shortcut signals through gradient misalignment with validation objectives and answer-token concentration, and modifies training dynamics accordingly. Experiments on controlled reasoning benchmarks show that SART achieves +16.5% accuracy and +40.2% robustness over the strongest baseline, significantly improving generalization under distribution shifts. Code is available at: this https URL.
>
---
#### [new 037] Expected Reward Prediction, with Applications to Model Routing
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究如何预测模型在给定提示下的预期奖励，用于模型路由任务，以在控制成本的同时最大化奖励。工作包括预测方法及实验验证。**

- **链接: [https://arxiv.org/pdf/2603.20217](https://arxiv.org/pdf/2603.20217)**

> **作者:** Kenan Hasanaliyev; Silas Alberti; Jenny Hamer; Dheeraj Rajagopal; Kevin Robinson; Jasper Snoek; Victor Veitch; Alexander Nicholas D'Amour
>
> **备注:** ICML 2025 Workshop on Models of Human Feedback for AI Alignment
>
> **摘要:** Reward models are a standard tool to score responses from LLMs. Reward models are built to rank responses to a fixed prompt sampled from a single model, for example to choose the best of n sampled responses. In this paper, we study whether scores from response-level reward models lifted to score a model's suitability for a prompt, prior to seeing responses from that model. Specifically, we show that it is straightforward to predict the expected reward that an LLM would earn from the reward model under repeated sampling. Further, we show that these expected reward predictions are precise and discriminative enough to support an application to a model routing protocol that routes prompts to models at inference time to maximize reward while controlling computational cost. We demonstrate the performance of this routing procedure on the open-perfectblend dataset, using a model pool composed of Llama3.1-Instruct 8B/70B, Gemma2-IT 9B/27B, and Gemma1-IT 7B models. Our simple expected reward prediction--based routing (ERP) outperforms baselines that route prompts to models with the best average performance within each prompt's category, and explains the success of more complex routing protocols that implicitly estimate an expected reward. Our approach has the added advantage of being trivially extensible as new models are added to the pool.
>
---
#### [new 038] TimeTox: An LLM-Based Pipeline for Automated Extraction of Time Toxicity from Clinical Trial Protocols
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出TimeTox，用于自动化提取临床试验方案中的时间毒性数据。解决的是手动提取耗时的问题，通过LLM实现高效准确的量化分析。**

- **链接: [https://arxiv.org/pdf/2603.21335](https://arxiv.org/pdf/2603.21335)**

> **作者:** Saketh Vinjamuri; Marielle Fis Loperena; Marie C. Spezia; Ramez Kouzy
>
> **备注:** 19 pages, 5 figures, 7 tables
>
> **摘要:** Time toxicity, the cumulative healthcare contact days from clinical trial participation, is an important but labor-intensive metric to extract from protocol documents. We developed TimeTox, an LLM-based pipeline for automated extraction of time toxicity from Schedule of Assessments tables. TimeTox uses Google's Gemini models in three stages: summary extraction from full-length protocol PDFs, time toxicity quantification at six cumulative timepoints for each treatment arm, and multi-run consensus via position-based arm matching. We validated against 20 synthetic schedules (240 comparisons) and assessed reproducibility on 644 real-world oncology protocols. Two architectures were compared: single-pass (vanilla) and two-stage (structure-then-count). The two-stage pipeline achieved 100% clinically acceptable accuracy ($\pm$3 days) on synthetic data (MAE 0.81 days) versus 41.5% for vanilla (MAE 9.0 days). However, on real-world protocols, the vanilla pipeline showed superior reproducibility: 95.3% clinically acceptable accuracy (IQR $\leq$ 3 days) across 3 runs on 644 protocols, with 82.0% perfect stability (IQR = 0). The production pipeline extracted time toxicity for 1,288 treatment arms across multiple disease sites. Extraction stability on real-world data, rather than accuracy on synthetic benchmarks, is the decisive factor for production LLM deployment.
>
---
#### [new 039] Conversation Tree Architecture: A Structured Framework for Context-Aware Multi-Branch LLM Conversations
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于对话系统任务，旨在解决多主题对话中上下文混乱问题。提出Conversation Tree Architecture，通过树状结构管理对话上下文，提升多分支对话的准确性。**

- **链接: [https://arxiv.org/pdf/2603.21278](https://arxiv.org/pdf/2603.21278)**

> **作者:** Pranav Hemanth; Sampriti Saha
>
> **备注:** 6 pages, 1 figure. Prototype available at this https URL
>
> **摘要:** Large language models (LLMs) are increasingly deployed for extended, multi-topic conversations, yet the flat, append-only structure of current conversation interfaces introduces a fundamental limitation: all context accumulates in a single unbounded window, causing topically distinct threads to bleed into one another and progressively degrade response quality. We term this failure mode logical context poisoning. In this paper, we introduce the Conversation Tree Architecture (CTA), a hierarchical framework that organizes LLM conversations as trees of discrete, context-isolated nodes. Each node maintains its own local context window; structured mechanisms govern how context flows between parent and child nodes, downstream on branch creation and upstream on branch deletion. We additionally introduce volatile nodes, transient branches whose local context must be selectively merged upward or permanently discarded before purging. We formalize the architecture's primitives, characterize the open design problems in context flow, relate our framework to prior work in LLM memory management, and describe a working prototype implementation. The CTA provides a principled foundation for structured conversational context management and extends naturally to multi-agent settings.
>
---
#### [new 040] Weber's Law in Transformer Magnitude Representations: Efficient Coding, Representational Geometry, and Psychophysical Laws in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究Transformer模型中数值表示的几何结构，探讨其是否符合韦伯定律。通过多种实验方法，发现模型具有对数压缩特性，但该特性与行为表现无直接关联。任务为理解语言模型的数值表征机制。**

- **链接: [https://arxiv.org/pdf/2603.20642](https://arxiv.org/pdf/2603.20642)**

> **作者:** Jon-Paul Cacioli
>
> **备注:** 18 pages, 7 figures, 5 tables. Pre-registered on OSF. Submitted to TMLR
>
> **摘要:** How do transformer language models represent magnitude? Recent work disagrees: some find logarithmic spacing, others linear encoding, others per-digit circular representations. We apply the formal tools of psychophysics to resolve this. Using four converging paradigms (representational similarity analysis, behavioural discrimination, precision gradients, causal intervention) across three magnitude domains in three 7-9B instruction-tuned models spanning three architecture families (Llama, Mistral, Qwen), we report three findings. First, representational geometry is consistently log-compressive: RSA correlations with a Weber-law dissimilarity matrix ranged from .68 to .96 across all 96 model-domain-layer cells, with linear geometry never preferred. Second, this geometry is dissociated from behaviour: one model produces a human-range Weber fraction (WF = 0.20) while the other does not, and both models perform at chance on temporal and spatial discrimination despite possessing logarithmic geometry. Third, causal intervention reveals a layer dissociation: early layers are functionally implicated in magnitude processing (4.1x specificity) while later layers where geometry is strongest are not causally engaged (1.2x). Corpus analysis confirms the efficient coding precondition (alpha = 0.77). These results suggest that training data statistics alone are sufficient to produce log-compressive magnitude geometry, but geometry alone does not guarantee behavioural competence.
>
---
#### [new 041] RedacBench: Can AI Erase Your Secrets?
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文属于文本隐私保护任务，旨在解决敏感信息自动删除问题。提出RedacBench基准，评估模型在遵守安全策略下准确删除敏感信息并保留语义的能力。**

- **链接: [https://arxiv.org/pdf/2603.20208](https://arxiv.org/pdf/2603.20208)**

> **作者:** Hyunjun Jeon; Kyuyoung Kim; Jinwoo Shin
>
> **摘要:** Modern language models can readily extract sensitive information from unstructured text, making redaction -- the selective removal of such information -- critical for data security. However, existing benchmarks for redaction typically focus on predefined categories of data such as personally identifiable information (PII) or evaluate specific techniques like masking. To address this limitation, we introduce RedacBench, a comprehensive benchmark for evaluating policy-conditioned redaction across domains and strategies. Constructed from 514 human-authored texts spanning individual, corporate, and government sources, paired with 187 security policies, RedacBench measures a model's ability to selectively remove policy-violating information while preserving the original semantics. We quantify performance using 8,053 annotated propositions that capture all inferable information in each text. This enables assessment of both security -- the removal of sensitive propositions -- and utility -- the preservation of non-sensitive propositions. Experiments across multiple redaction strategies and state-of-the-art language models show that while more advanced models can improve security, preserving utility remains a challenge. To facilitate future research, we release RedacBench along with a web-based playground for dataset customization and evaluation. Available at this https URL.
>
---
#### [new 042] MemDLM: Memory-Enhanced DLM Training
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，针对扩散语言模型训练与推理不匹配的问题，提出MemDLM方法，通过双层优化引入参数化记忆，提升模型性能和长文本理解能力。**

- **链接: [https://arxiv.org/pdf/2603.22241](https://arxiv.org/pdf/2603.22241)**

> **作者:** Zehua Pei; Hui-Ling Zhen; Weizhe Lin; Sinno Jialin Pan; Yunhe Wang; Mingxuan Yuan; Bei Yu
>
> **摘要:** Diffusion Language Models (DLMs) offer attractive advantages over Auto-Regressive (AR) models, such as full-attention parallel decoding and flexible generation. However, they suffer from a notable train-inference mismatch: DLMs are trained with a static, single-step masked prediction objective, but deployed through a multi-step progressive denoising trajectory. We propose MemDLM (Memory-Enhanced DLM), which narrows this gap by embedding a simulated denoising process into training via Bi-level Optimization. An inner loop updates a set of fast weights, forming a Parametric Memory that captures the local trajectory experience of each sample, while an outer loop updates the base model conditioned on this memory. By offloading memorization pressure from token representations to parameters, MemDLM yields faster convergence and lower training loss. Moreover, the inner loop can be re-enabled at inference time as an adaptation step, yielding additional gains on long-context understanding. We find that, when activated at inference time, this Parametric Memory acts as an emergent in-weight retrieval mechanism, helping MemDLM further reduce token-level attention bottlenecks on challenging Needle-in-a-Haystack retrieval tasks. Code: this https URL.
>
---
#### [new 043] Task-Specific Efficiency Analysis: When Small Language Models Outperform Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理领域，旨在解决资源受限环境下模型效率问题。通过对比分析，发现小型模型在多个任务中比大型模型更具效率优势。**

- **链接: [https://arxiv.org/pdf/2603.21389](https://arxiv.org/pdf/2603.21389)**

> **作者:** Jinghan Cao; Yu Ma; Xinjin Li; Qingyang Ren; Xiangyun Chen
>
> **备注:** Accepted for publication at ESANN 2025. This is a task-specific efficiency analysis comparing small language models
>
> **摘要:** Large Language Models achieve remarkable performance but incur substantial computational costs unsuitable for resource-constrained deployments. This paper presents the first comprehensive task-specific efficiency analysis comparing 16 language models across five diverse NLP tasks. We introduce the Performance-Efficiency Ratio (PER), a novel metric integrating accuracy, throughput, memory, and latency through geometric mean normalization. Our systematic evaluation reveals that small models (0.5--3B parameters) achieve superior PER scores across all given tasks. These findings establish quantitative foundations for deploying small models in production environments prioritizing inference efficiency over marginal accuracy gains.
>
---
#### [new 044] Multi-Perspective LLM Annotations for Valid Analyses in Subjective Tasks
- **分类: cs.CL**

- **简介: 该论文研究主观任务中的标注问题，针对LLM标注误差提出改进方法。通过分析不同群体的标注分布，优化人类标注资源分配，提升模型在难处理群体上的表现。**

- **链接: [https://arxiv.org/pdf/2603.21404](https://arxiv.org/pdf/2603.21404)**

> **作者:** Navya Mehrotra; Adam Visokay; Kristina Gligorić
>
> **摘要:** Large language models are increasingly used to annotate texts, but their outputs reflect some human perspectives better than others. Existing methods for correcting LLM annotation error assume a single ground truth. However, this assumption fails in subjective tasks where disagreement across demographic groups is meaningful. Here we introduce Perspective-Driven Inference, a method that treats the distribution of annotations across groups as the quantity of interest, and estimates it using a small human annotation budget. We contribute an adaptive sampling strategy that concentrates human annotation effort on groups where LLM proxies are least accurate. We evaluate on politeness and offensiveness rating tasks, showing targeted improvements for harder-to-model demographic groups relative to uniform sampling baselines, while maintaining coverage.
>
---
#### [new 045] Efficient Fine-Tuning Methods for Portuguese Question Answering: A Comparative Study of PEFT on BERTimbau and Exploratory Evaluation of Generative LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文研究巴西葡萄牙语问答任务，探讨PEFT方法在BERTimbau上的应用，以降低计算成本。对比不同方法的性能与效率，验证小模型在该任务中的优势。**

- **链接: [https://arxiv.org/pdf/2603.21418](https://arxiv.org/pdf/2603.21418)**

> **作者:** Mariela M. Nina; Caio Veloso Costa; Lilian Berton; Didier A. Vega-Oliveros
>
> **备注:** 10 pages, 2 figures, PROPOR 2026
>
> **摘要:** Although large language models have transformed natural language processing, their computational costs create accessibility barriers for low-resource languages such as Brazilian Portuguese. This work presents a systematic evaluation of Parameter-Efficient Fine-Tuning (PEFT) and quantization techniques applied to BERTimbau for Question Answering on SQuAD-BR, the Brazilian Portuguese translation of SQuAD v1. We evaluate 40 configurations combining four PEFT methods (LoRA, DoRA, QLoRA, QDoRA) across two model sizes (Base: 110M, Large: 335M parameters). Our findings reveal three critical insights: (1) LoRA achieves 95.8\% of baseline performance on BERTimbau-Large while reducing training time by 73.5\% (F1=81.32 vs 84.86); (2) higher learning rates (2e-4) substantially improve PEFT performance, with F1 gains of up to +19.71 points over standard rates; and (3) larger models show twice the quantization resilience (loss of 4.83 vs 9.56 F1 points). These results demonstrate that encoder-based models can be efficiently fine-tuned for extractive Brazilian Portuguese QA with substantially lower computational cost than large generative LLMs, promoting more sustainable approaches aligned with \textit{Green AI} principles. An exploratory evaluation of Tucano and Sabiá on the same extractive QA benchmark shows that while generative models can reach competitive F1 scores with LoRA fine-tuning, they require up to 4.2$\times$ more GPU memory and 3$\times$ more training time than BERTimbau-Base, reinforcing the efficiency advantage of smaller encoder-based architectures for this task.
>
---
#### [new 046] Retrieving Climate Change Disinformation by Narrative
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决气候虚假信息检测问题。通过将叙事检测转化为检索任务，利用SpecFi框架提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.22015](https://arxiv.org/pdf/2603.22015)**

> **作者:** Max Upravitelev; Veronika Solopova; Charlott Jakob; Premtim Sahitaj; Vera Schmitt
>
> **摘要:** Detecting climate disinformation narratives typically relies on fixed taxonomies, which do not accommodate emerging narratives. Thus, we re-frame narrative detection as a retrieval task: given a narrative's core message as a query, rank texts from a corpus by alignment with that narrative. This formulation requires no predefined label set and can accommodate emerging narratives. We repurpose three climate disinformation datasets (CARDS, Climate Obstruction, climate change subset of PolyNarrative) for retrieval evaluation and propose SpecFi, a framework that generates hypothetical documents to bridge the gap between abstract narrative descriptions and their concrete textual instantiations. SpecFi uses community summaries from graph-based community detection as few-shot examples for generation, achieving a MAP of 0.505 on CARDS without access to narrative labels. We further introduce narrative variance, an embedding-based difficulty metric, and show via partial correlation analysis that standard retrieval degrades on high-variance narratives (BM25 loses 63.4% of MAP), while SpecFi-CS remains robust (32.7% loss). Our analysis also reveals that unsupervised community summaries converge on descriptions close to expert-crafted taxonomies, suggesting that graph-based methods can surface narrative structure from unlabeled text.
>
---
#### [new 047] The production of meaning in the processing of natural language
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于自然语言处理任务，探讨语言理解中的意义生成机制，研究量子逻辑在语义处理中的体现，分析模型上下文敏感性与基准测试的关系。**

- **链接: [https://arxiv.org/pdf/2603.20381](https://arxiv.org/pdf/2603.20381)**

> **作者:** Christopher J. Agostino; Quan Le Thien; Nayan D'Souza; Louis van der Elst
>
> **备注:** Submitted to HAXD 2026, 9 pages, 3 figures, 2 tables. associated package available at this https URL
>
> **摘要:** Understanding the fundamental mechanisms governing the production of meaning in the processing of natural language is critical for designing safe, thoughtful, engaging, and empowering human-agent interactions. Experiments in cognitive science and social psychology have demonstrated that human semantic processing exhibits contextuality more consistent with quantum logical mechanisms than classical Boolean theories, and recent works have found similar results in large language models -- in particular, clear violations of the Bell inequality in experiments of contextuality during interpretation of ambiguous expressions. We explore the CHSH $|S|$ parameter -- the metric associated with the inequality -- across the inference parameter space of models spanning four orders of magnitude in scale, cross-referencing it with MMLU, hallucination rate, and nonsense detection benchmarks. We find that the interquartile range of the $|S|$ distribution -- the statistic that most sharply differentiates models from one another -- is completely orthogonal to all external benchmarks, while violation rate shows weak anticorrelation with all three benchmarks that does not reach significance. We investigate how $|S|$ varies with sampling parameters and word order, and discuss the information-theoretic constraints that genuine contextuality imposes on prompt injection defenses and its human analogue, whereby careful construction and maintenance of social contextuality can be carried out at scale -- manufacturing not consent but contextuality itself, a subtler and more fundamental form of manipulation that shapes the space of possible interpretations before any particular one is reached.
>
---
#### [new 048] PARHAF, a human-authored corpus of clinical reports for fictitious patients in French
- **分类: cs.CL**

- **简介: 该论文提出PARHAF，一个用于法语临床文本处理的匿名合成语料库，解决医疗数据隐私限制问题，通过专家编写虚构病例报告实现数据共享与模型训练。**

- **链接: [https://arxiv.org/pdf/2603.20494](https://arxiv.org/pdf/2603.20494)**

> **作者:** Xavier Tannier; Salam Abbara; Rémi Flicoteaux; Youness Khalil; Aurélie Névéol; Pierre Zweigenbaum; Emmanuel Bacry
>
> **摘要:** The development of clinical natural language processing (NLP) systems is severely hampered by the sensitive nature of medical records, which restricts data sharing under stringent privacy regulations, particularly in France and the broader European Union. To address this gap, we introduce PARHAF, a large open-source corpus of clinical documents in French. PARHAF comprises expert-authored clinical reports describing realistic yet entirely fictitious patient cases, making it anonymous and freely shareable by design. The corpus was developed using a structured protocol that combined clinician expertise with epidemiological guidance from the French National Health Data System (SNDS), ensuring broad clinical coverage. A total of 104 medical residents across 18 specialties authored and peer-reviewed the reports following predefined clinical scenarios and document templates. The corpus contains 7394 clinical reports covering 5009 patient cases across a wide range of medical and surgical specialties. It includes a general-purpose component designed to approximate real-world hospitalization distributions, and four specialized subsets that support information-extraction use cases in oncology, infectious diseases, and diagnostic coding. Documents are released under a CC-BY open license, with a portion temporarily embargoed to enable future benchmarking under controlled conditions. PARHAF provides a valuable resource for training and evaluating French clinical language models in a fully privacy-preserving setting, and establishes a replicable methodology for building shareable synthetic clinical corpora in other languages and health systems.
>
---
#### [new 049] Generalizable Self-Evolving Memory for Automatic Prompt Optimization
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决传统提示优化方法泛化能力差的问题。提出MemAPO框架，通过记忆机制积累有效策略和错误模式，提升提示优化的通用性和效率。**

- **链接: [https://arxiv.org/pdf/2603.21520](https://arxiv.org/pdf/2603.21520)**

> **作者:** Guanbao Liang; Yuanchen Bei; Sheng Zhou; Yuheng Qin; Huan Zhou; Bingxin Jia; Bin Li; Jiajun Bu
>
> **摘要:** Automatic prompt optimization is a promising approach for adapting large language models (LLMs) to downstream tasks, yet existing methods typically search for a specific prompt specialized to a fixed task. This paradigm limits generalization across heterogeneous queries and prevents models from accumulating reusable prompting knowledge over time. In this paper, we propose MemAPO, a memory-driven framework that reconceptualizes prompt optimization as generalizable and self-evolving experience accumulation. MemAPO maintains a dual-memory mechanism that distills successful reasoning trajectories into reusable strategy templates while organizing incorrect generations into structured error patterns that capture recurrent failure modes. Given a new prompt, the framework retrieves both relevant strategies and failure patterns to compose prompts that promote effective reasoning while discouraging known mistakes. Through iterative self-reflection and memory editing, MemAPO continuously updates its memory, enabling prompt optimization to improve over time rather than restarting from scratch for each task. Experiments on diverse benchmarks show that MemAPO consistently outperforms representative prompt optimization baselines while substantially reducing optimization cost.
>
---
#### [new 050] HiCI: Hierarchical Construction-Integration for Long-Context Attention
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于长文本建模任务，旨在解决长上下文注意力的可扩展性问题。提出HiCI模块，通过分层构建与整合信息，提升模型对长文本的理解能力。**

- **链接: [https://arxiv.org/pdf/2603.20843](https://arxiv.org/pdf/2603.20843)**

> **作者:** Xiangyu Zeng; Qi Xu; Yunke Wang; Chang Xu
>
> **备注:** 18 pages, 5 figures
>
> **摘要:** Long-context language modeling is commonly framed as a scalability challenge of token-level attention, yet local-to-global information structuring remains largely implicit in existing approaches. Drawing on cognitive theories of discourse comprehension, we propose HiCI (Hierarchical Construction--Integration), a hierarchical attention module that constructs segment-level representations, integrates them into a shared global context, and broadcasts both to condition segment-level attention. We validate HiCI through parameter-efficient adaptation of LLaMA-2 with only <5.5% additional parameters, extending context from 4K to 100K tokens (7B) and 64K tokens (13B). Across language modeling, retrieval, and instruction-following benchmarks, HiCI yields consistent improvements over strong baselines, including matching proprietary models on topic retrieval and surpassing GPT-3.5-Turbo-16K on code comprehension. These results demonstrate the effectiveness of explicit hierarchical structuring as an inductive bias for long-context modeling.
>
---
#### [new 051] Policies Permitting LLM Use for Polishing Peer Reviews Are Currently Not Enforceable
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **简介: 该论文属于AI检测任务，旨在评估LLM辅助润色的同行评审是否可被检测。研究发现现有检测工具无法准确识别混合人类与AI生成的评审内容，导致误判风险。**

- **链接: [https://arxiv.org/pdf/2603.20450](https://arxiv.org/pdf/2603.20450)**

> **作者:** Rounak Saha; Gurusha Juneja; Dayita Chaudhuri; Naveeja Sajeevan; Nihar B Shah; Danish Pruthi
>
> **摘要:** A number of scientific conferences and journals have recently enacted policies that prohibit LLM usage by peer reviewers, except for polishing, paraphrasing, and grammar correction of otherwise human-written reviews. But, are these policies enforceable? To answer this question, we assemble a dataset of peer reviews simulating multiple levels of human-AI collaboration, and evaluate five state-of-the-art detectors, including two commercial systems. Our analysis shows that all detectors misclassify a non-trivial fraction of LLM-polished reviews as AI-generated, thereby risking false accusations of academic misconduct. We further investigate whether peer-review-specific signals, including access to the paper manuscript and the constrained domain of scientific writing, can be leveraged to improve detection. While incorporating such signals yields measurable gains in some settings, we identify limitations in each approach and find that none meets the accuracy standards required for identifying AI use in peer reviews. Importantly, our results suggest that recent public estimates of AI use in peer reviews through the use of AI-text detectors should be interpreted with caution, as current detectors misclassify mixed reviews (collaborative human-AI outputs) as fully AI generated, potentially overstating the extent of policy violations.
>
---
#### [new 052] Evaluating Large Language Models on Historical Health Crisis Knowledge in Resource-Limited Settings: A Hybrid Multi-Metric Study
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于评估任务，旨在解决LLMs在资源有限地区健康危机知识可靠性问题。通过多指标分析，评估模型在新冠、登革热等疾病信息上的表现。**

- **链接: [https://arxiv.org/pdf/2603.20514](https://arxiv.org/pdf/2603.20514)**

> **作者:** Mohammed Rakibul Hasan
>
> **备注:** Comments: 20 pages, 7 figures, 3 tables
>
> **摘要:** Large Language Models (LLMs) offer significant potential for delivering health information. However, their reliability in low-resource contexts remains uncertain. This study evaluates GPT-4, Gemini Pro, Llama~3, and Mistral-7B on health crisis-related enquiries concerning COVID-19, dengue, the Nipah virus, and Chikungunya in the low-resource context of Bangladesh. We constructed a question--answer dataset from authoritative sources and assessed model outputs through semantic similarity, expert-model cross-evaluation, and Natural Language Inference (NLI). Findings highlight both the strengths and limitations of LLMs in representing epidemiological history and health crisis knowledge, underscoring their promise and risks for informing policy in resource-constrained environments.
>
---
#### [new 053] Evaluating Reasoning-Based Scaffolds for Human-AI Co-Annotation: The ReasonAlign Annotation Protocol
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的情感分类与观点检测任务，旨在研究推理引导的标注框架对人类标注行为的影响。通过设计一种双阶段标注协议，分析推理解释如何提升标注一致性。**

- **链接: [https://arxiv.org/pdf/2603.21094](https://arxiv.org/pdf/2603.21094)**

> **作者:** Smitha Muthya Sudheendra; Jaideep Srivastava
>
> **摘要:** Human annotation is central to NLP evaluation, yet subjective tasks often exhibit substantial variability across annotators. While large language models (LLMs) can provide structured reasoning to support annotation, their influence on human annotation behavior remains unclear. We introduce ReasonAlign, a reasoning-based annotation scaffold that exposes LLM-generated explanations while withholding predicted labels. We frame this as a controlled study of how reasoning affects human annotation behavior, rather than a full evaluation of annotation accuracy. Using a two-pass protocol inspired by Delphi-style revision, annotators first label instances independently and then revise their decisions after viewing model-generated reasoning. We evaluate the approach on sentiment classification and opinion detection tasks, analyzing changes in inter-annotator agreement and revision behavior. To quantify these effects, we introduce the Annotator Effort Proxy (AEP), a metric capturing the proportion of labels revised after exposure to reasoning. Our results show that exposure to reasoning is associated with increased agreement alongside minimal revision, suggesting that reasoning primarily helps resolve ambiguous cases without inducing widespread changes. These findings provide insight into how reasoning explanations shape annotation consistency and highlight reasoning-based scaffolds as a practical mechanism for supporting human-AI annotation workflows.
>
---
#### [new 054] SynSym: A Synthetic Data Generation Framework for Psychiatric Symptom Identification
- **分类: cs.CL**

- **简介: 该论文属于精神症状识别任务，旨在解决标注数据不足的问题。提出SynSym框架，利用大语言模型生成高质量合成数据，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.21529](https://arxiv.org/pdf/2603.21529)**

> **作者:** Migyeong Kang; Jihyun Kim; Hyolim Jeon; Sunwoo Hwang; Jihyun An; Yonghoon Kim; Haewoon Kwak; Jisun An; Jinyoung Han
>
> **摘要:** Psychiatric symptom identification on social media aims to infer fine-grained mental health symptoms from user-generated posts, allowing a detailed understanding of users' mental states. However, the construction of large-scale symptom-level datasets remains challenging due to the resource-intensive nature of expert labeling and the lack of standardized annotation guidelines, which in turn limits the generalizability of models to identify diverse symptom expressions from user-generated text. To address these issues, we propose SynSym, a synthetic data generation framework for constructing generalizable datasets for symptom identification. Leveraging large language models (LLMs), SynSym constructs high-quality training samples by (1) expanding each symptom into sub-concepts to enhance the diversity of generated expressions, (2) producing synthetic expressions that reflect psychiatric symptoms in diverse linguistic styles, and (3) composing realistic multi-symptom expressions, informed by clinical co-occurrence patterns. We validate SynSym on three benchmark datasets covering different styles of depressive symptom expression. Experimental results demonstrate that models trained solely on the synthetic data generated by SynSym perform comparably to those trained on real data, and benefit further from additional fine-tuning with real data. These findings underscore the potential of synthetic data as an alternative resource to real-world annotations in psychiatric symptom modeling, and SynSym serves as a practical framework for generating clinically relevant and realistic symptom expressions.
>
---
#### [new 055] Dual-Space Knowledge Distillation with Key-Query Matching for Large Language Models with Vocabulary Mismatch
- **分类: cs.CL**

- **简介: 该论文属于知识蒸馏任务，解决不同分词器模型间的分布不匹配问题。通过引入生成对抗学习改进DSKD-CMA方法，提升文本生成质量。**

- **链接: [https://arxiv.org/pdf/2603.22056](https://arxiv.org/pdf/2603.22056)**

> **作者:** Stella Eva Tsiapali; Cong-Thanh Do; Kate Knill
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Large language models (LLMs) achieve state-of-the-art (SOTA) performance across language tasks, but are costly to deploy due to their size and resource demands. Knowledge Distillation (KD) addresses this by training smaller Student models to mimic larger Teacher models, improving efficiency without significant performance loss. Dual-Space Knowledge Distillation with Cross-Model Attention (DSKD-CMA) has emerged as a SOTA method for KD between LLMs with distinct tokenizers, yet its internal workings remain largely opaque. In this work, we systematically analyse the attention mechanism of DSKD-CMA through manual token alignment probing and heatmap visualisations, revealing both strengths and limitations. Building on this, we introduce a novel method, DSKD-CMA-GA, based on Generative Adversarial (GA) learning, to address the mismatched distributions between the keys and queries computed from distinct models. Experiments show modest but consistent ROUGE-L gains in text generation quality, particularly on out-of-distribution data (+0.37 on average), narrowing the gap between cross- and same-tokenizer KD.
>
---
#### [new 056] DRTriton: Large-Scale Synthetic Data Reinforcement Learning for Triton Kernel Generation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于CUDA内核生成任务，旨在解决LLM在将PyTorch代码转换为高效CUDA内核时效率低的问题。提出DRTriton框架，通过合成数据和强化学习提升转换效果。**

- **链接: [https://arxiv.org/pdf/2603.21465](https://arxiv.org/pdf/2603.21465)**

> **作者:** Siqi Guo; Ming Lin; Tianbao Yang
>
> **摘要:** Developing efficient CUDA kernels is a fundamental yet challenging task in the generative AI industry. Recent researches leverage Large Language Models (LLMs) to automatically convert PyTorch reference implementations to CUDA kernels, significantly reducing the engineering efforts. State-of-the-art LLMs, such as GPT-5.2 and Claude-Sonnet-4.5, still struggle in this specific task. To address this challenge, we propose DRTriton, a scalable learning framework for training LLMs to convert PyTorch codes into highly optimized Triton kernels, which are then compiled to CUDA kernels at runtime. DRTriton consists of three key components: (i) a data synthetic algorithm CSP-DAG that guarantees full coverage and unbiased uniform sampling over the operator space with controlled difficulty; (ii) a curriculum reinforcement learning with decoupled reward efficiently optimizes conversion success rate and inference speed simultaneously; and (iii) a test-time search algorithm that further improves the inference speed of the generated Triton kernels. Notably, despite being trained exclusively on synthetic data, DRTriton generalizes effectively to real-world CUDA kernels that are challenging even for human experts. Experimental results show that DRTriton-7B achieves speedup on 92% of the KernelBench Level 2, compared to 23% for GPT-5.2 and 19% for Claude-Sonnet-4.5.
>
---
#### [new 057] BenchBench: Benchmarking Automated Benchmark Generation
- **分类: cs.CL**

- **简介: 该论文提出BenchBench，用于评估自动化基准生成能力。解决静态基准局限性问题，通过多阶段流程生成高质量基准，并分析模型设计与解答能力的关系。**

- **链接: [https://arxiv.org/pdf/2603.20807](https://arxiv.org/pdf/2603.20807)**

> **作者:** Yandan Zheng; Haoran Luo; Zhenghong Lin; Wenjin Liu; Luu Anh Tuan
>
> **摘要:** Benchmarks are the de facto standard for tracking progress in large language models (LLMs), yet static test sets can rapidly saturate, become vulnerable to contamination, and are costly to refresh. Scalable evaluation of open-ended items often relies on LLM judges, introducing additional sources of bias and prompt sensitivity. We argue that evaluation must extend beyond how well models answer benchmarks to how well models design them. We introduce BenchBench, a three-stage pipeline and dataset for benchmarking automated benchmark generation: (i) extract structured domain cards from seed benchmarks, (ii) prompt multiple designer LLMs to generate quota-controlled suites, and (iii) validate items with a multi-model answerer panel using exact/numeric/symbolic verifiers when possible and rubric-guided judging otherwise, yielding designer--answerer matrices with item-level quality flags and psychometric diagnostics. Across nine variants spanning computer science, mathematics, medicine, and theory-of-mind reasoning (including multilingual and multimodal settings), we generate 16.7K items, retain ~15K core items post-filtering, and produce ~152K graded model--item responses. BenchBench shows that benchmark-design ability is only moderately correlated with answer-time strength (Spearman rho ~0.37), invalidity is negatively associated with discrimination (Pearson r~0.62), and the resulting designer--answerer matrices enable scalable audits of format/modality/language fidelity and suite-dependent self/family interactions. The project is available at: this https URL.
>
---
#### [new 058] TiCo: Time-Controllable Training for Spoken Dialogue Models
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文属于语音对话系统任务，解决模型无法准确控制响应时长的问题。通过引入时间标记和强化学习，提升模型对时间约束的适应能力。**

- **链接: [https://arxiv.org/pdf/2603.22267](https://arxiv.org/pdf/2603.22267)**

> **作者:** Kai-Wei Chang; Wei-Chih Chen; En-Pei Hu; Hung-yi Lee; James Glass
>
> **摘要:** We propose TiCo, a simple post-training method for enabling spoken dialogue models (SDMs) to follow time-constrained instructions and generate responses with controllable duration. This capability is valuable for real-world spoken language systems such as voice assistants and interactive agents, where controlling response duration can improve interaction quality. However, despite their strong ability to generate natural spoken responses, existing models lack time awareness and struggle to follow duration-related instructions (e.g., "Please generate a response lasting about 15 seconds"). Through an empirical evaluation of both open-source and commercial SDMs, we show that they frequently fail to satisfy such time-control requirements. TiCo addresses this limitation by enabling models to estimate elapsed speaking time during generation through Spoken Time Markers (STM) (e.g., <10.6 seconds>). These markers help the model maintain awareness of time and adjust the remaining content to meet the target duration. TiCo is simple and efficient: it requires only a small amount of data and no additional question-answer pairs, relying instead on self-generation and reinforcement learning. Experimental results show that TiCo significantly improves adherence to duration constraints while preserving response quality.
>
---
#### [new 059] Can I guess where you are from? Modeling dialectal morphosyntactic similarities in Brazilian Portuguese
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于语言学分析任务，旨在通过语法现象推断巴西葡萄牙语方言。研究分析了四种与代词相关的语法现象，采用相关性和聚类方法，探索方言分布与语言变量的关系。**

- **链接: [https://arxiv.org/pdf/2603.20695](https://arxiv.org/pdf/2603.20695)**

> **作者:** Manoel Siqueira; Raquel Freitag
>
> **备注:** 17th International Conference on Computational Processing of Portuguese - PROPOR
>
> **摘要:** This paper investigates morphosyntactic covariation in Brazilian Portuguese (BP) to assess whether dialectal origin can be inferred from the combined behavior of linguistic variables. Focusing on four grammatical phenomena related to pronouns, correlation and clustering methods are applied to model covariation and dialectal distribution. The results indicate that correlation captures only limited pairwise associations, whereas clustering reveals speaker groupings that reflect regional dialectal patterns. Despite the methodological constraints imposed by differences in sample size requirements between sociolinguistics and computational approaches, the study highlights the importance of interdisciplinary research. Developing fair and inclusive language technologies that respect dialectal diversity outweighs the challenges of integrating these fields.
>
---
#### [new 060] CatRAG: Functor-Guided Structural Debiasing with Retrieval Augmentation for Fair LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CatRAG框架，用于减少大语言模型的偏见问题。针对公平性任务，通过结构化去偏和检索增强生成技术，有效提升模型准确性并降低偏见。**

- **链接: [https://arxiv.org/pdf/2603.21524](https://arxiv.org/pdf/2603.21524)**

> **作者:** Ravi Ranjan; Utkarsh Grover; Mayur Akewar; Xiaomin Lin; Agoritsa Polyzou
>
> **备注:** 9 pages, 4 figures, and accepted in IJCNN 2026 (part of IEEE WCCI 2026)
>
> **摘要:** Large Language Models (LLMs) are deployed in high-stakes settings but can show demographic, gender, and geographic biases that undermine fairness and trust. Prior debiasing methods, including embedding-space projections, prompt-based steering, and causal interventions, often act at a single stage of the pipeline, resulting in incomplete mitigation and brittle utility trade-offs under distribution shifts. We propose CatRAG Debiasing, a dual-pronged framework that integrates functor with Retrieval-Augmented Generation (RAG) guided structural debiasing. The functor component leverages category-theoretic structure to induce a principled, structure-preserving projection that suppresses bias-associated directions in the embedding space while retaining task-relevant semantics. On the Bias Benchmark for Question Answering (BBQ) across three open-source LLMs (Meta Llama-3, OpenAI GPT-OSS, and Google Gemma-3), CatRAG achieves state-of-the-art results, improving accuracy by up to 40% over the corresponding base models and by more than 10% over prior debiasing methods, while reducing bias scores to near zero (from 60% for the base models) across gender, nationality, race, and intersectional subgroups.
>
---
#### [new 061] Enhancing Safety of Large Language Models via Embedding Space Separation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型安全任务，旨在提升大语言模型对有害提示的防御能力。通过扩大嵌入空间中有害与安全表示的距离，增强模型安全性。**

- **链接: [https://arxiv.org/pdf/2603.20206](https://arxiv.org/pdf/2603.20206)**

> **作者:** Xu Zhao; Xiting Wang; Weiran Shen
>
> **摘要:** Large language models (LLMs) have achieved impressive capabilities, yet ensuring their safety against harmful prompts remains a critical challenge. Recent work has revealed that the latent representations (embeddings) of harmful and safe queries in LLMs typically exhibit linear separability, a property that has been exploited to construct attacks by perturbing the embeddings of harmful queries towards the safe subspace. Motivated by this observation, we propose a representation-level fine-tuning approach, named Embedding Space Separation (ES2), which improves LLM safety by explicitly enlarging the distance between harmful and safe representations in the embedding space. To prevent degradation of model's general capabilities, we introduce a Kullback-Leibler (KL) divergence regularization term into the loss function, which constrains the logits of the fine-tuned model to align with those of the original base model on harmless inputs. We evaluate our method on several open-source LLMs using standard safety benchmarks. Extensive experimental results demonstrate that our approach substantially improves model safety while maintaining comparable general capabilities.
>
---
#### [new 062] Children's Intelligence Tests Pose Challenges for MLLMs? KidGym: A 2D Grid-Based Reasoning Benchmark for MLLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出KidGym，一个用于评估多模态大语言模型（MLLMs）的2D网格基准，解决模型在执行、感知推理等五方面能力的测评问题。**

- **链接: [https://arxiv.org/pdf/2603.20209](https://arxiv.org/pdf/2603.20209)**

> **作者:** Hengwei Ye; Yuanting Guan; Yuxuan Ge; Tianying Zhu; Zhenhan Guan; Yijia Zhong; Yijing Zhang; Han Zhang; Yingna Wu; Zheng Tian
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Multimodal Large Language Models (MLLMs) combine the linguistic strengths of LLMs with the ability to process multimodal data, enbaling them to address a broader range of visual tasks. Because MLLMs aim at more general, human-like competence than language-only models, we take inspiration from the Wechsler Intelligence Scales - an established battery for evaluating children by decomposing intelligence into interpretable, testable abilities. We introduce KidGym, a comprehensive 2D grid-based benchmark for assessing five essential capabilities of MLLMs: Execution, Perception Reasoning, Learning, Memory and Planning. The benchmark comprises 12 unique tasks, each targeting at least one core capability, specifically designed to guage MLLMs' adaptability and developmental potential, mirroring the stages of children's cognitive growth. Additionally, our tasks encompass diverse scenarios and objects with randomly generated layouts, ensuring a more accurate and robust evluation of MLLM capabilities. KidGym is designed to be fully user-customizable and extensible, allowing researchers to create new evaluation scenarios and adjust difficuly levels to accommodate the rapidly growing MLLM community. Through the evaluation of state-of-the-art MLLMs using KidGym, we identified significant insights into model capabilities and revealed several limitations of current models. We release our benchmark at: this https URL.
>
---
#### [new 063] LLM Router: Prefill is All You Need
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型集成任务，旨在提升LLM性能。通过构建路由机制，利用预填充激活优化模型组合，解决单一模型性能不足问题。**

- **链接: [https://arxiv.org/pdf/2603.20895](https://arxiv.org/pdf/2603.20895)**

> **作者:** Tanay Varshney; Annie Surla; Michelle Xu; Gomathy Venkata Krishnan; Maximilian Jeblick; David Austin; Neal Vaidya; Davide Onofrio
>
> **摘要:** LLMs often share comparable benchmark accuracies, but their complementary performance across task subsets suggests that an Oracle router--a theoretical selector with perfect foresight--can significantly surpass standalone model accuracy by navigating model-specific strengths. While current routers rely on fragile semantic signals, we propose using internal prefill activations via Encoder-Target Decoupling--a functional separation between the model providing the predictive signal (the Encoder) and the model whose performance is being estimated (the Target). This allows optimized heterogeneous pairing between unique encoders and target models. We utilize Fisher Separability (J) and Effective Dimensionality (d_eff) as mathematical probes to isolate optimal layer-wise signals, providing the predictive foundation for our SharedTrunkNet architecture. SharedTrunkNet captures up to 45.58% of the accuracy gap between the strongest standalone model and the Oracle while achieving 74.31% cost savings relative to the highest-cost model.
>
---
#### [new 064] Cross-Context Verification: Hierarchical Detection of Benchmark Contamination through Session-Isolated Analysis
- **分类: cs.CL**

- **简介: 该论文针对LLM编码基准的可信度问题，提出Cross-Context Verification方法，通过独立会话分析检测污染，解决模型是否真实推理而非记忆的问题。**

- **链接: [https://arxiv.org/pdf/2603.21454](https://arxiv.org/pdf/2603.21454)**

> **作者:** Tae-Eun Song
>
> **备注:** 11 pages, 3 figures, 4 tables
>
> **摘要:** LLM coding benchmarks face a credibility crisis: widespread solution leakage and test quality issues undermine SWE-bench Verified, while existing detection methods--paraphrase consistency, n-gram overlap, perplexity analysis--never directly observe whether a model reasons or recalls. Meanwhile, simply repeating verification degrades accuracy: multi-turn review generates false positives faster than it discovers true errors, suggesting that structural approaches are needed. We introduce Cross-Context Verification (CCV), a black-box method that solves the same benchmark problem in N independent sessions and measures solution diversity, combined with the Hierarchical Cross-Context Architecture (HCCA), a multi-agent analysis framework that prevents confirmation bias through intentional information restriction across specialized analytical roles. On 9 SWE-bench Verified problems (45 trials, Claude Opus 4.6, temperature 0), CCV achieves perfect separation between contaminated and genuine reasoning (Mann-Whitney U=0, p approx 0.012, r = 1.0). Key findings: (1) contamination is binary--models either recall perfectly or not at all; (2) reasoning absence is a perfect discriminator; (3) 33% of prior contamination labels are false positives; (4) HCCA's independent analysis structure discovers contamination-flaw composite cases that single-analyst approaches miss. A pilot experiment extending HCCA to multi-stage verification (Worker to Verifier to Director) yields a negative result--100% sycophantic confirmation--providing further evidence that information restriction, not structural complexity, is the key mechanism. We release all code and data.
>
---
#### [new 065] Effective Strategies for Asynchronous Software Engineering Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多智能体协作在软件工程中的应用，解决长周期任务的协调难题。提出CAID框架，通过集中任务分配、异步执行和隔离工作区提高任务完成的准确性和效率。**

- **链接: [https://arxiv.org/pdf/2603.21489](https://arxiv.org/pdf/2603.21489)**

> **作者:** Jiayi Geng; Graham Neubig
>
> **摘要:** AI agents have become increasingly capable at isolated software engineering (SWE) tasks such as resolving issues on Github. Yet long-horizon tasks involving multiple interdependent subtasks still pose challenges both with respect to accuracy, and with respect to timely completion. A natural approach to solving these long-horizon tasks in a timely manner is asynchronous multi-agent collaboration, where multiple agents work on different parts of the task at the same time. But effective application of multi-agent systems has proven surprisingly difficult: concurrent edits by multiple agents interfere with each other, dependencies are difficult to synchronize, and combining partial progress into a coherent whole is challenging. On the other hand, human developers have long relied on mature collaboration infrastructure to manage these challenges in large software projects. Inspired by these collaboration primitives, we introduce Centralized Asynchronous Isolated Delegation (CAID), a structured multi-agent coordination paradigm grounded in three core SWE primitives: centralized task delegation, asynchronous execution, and isolated workspaces. CAID constructs dependency-aware task plans through a central manager, executes subtasks concurrently in isolated workspaces, and consolidates progress via structured integration with executable test-based verification. In empirical evaluation, we find that CAID improves accuracy over single-agent baselines by 26.7% absolute on paper reproduction tasks (PaperBench) and 14.3% on Python library development tasks (Commit0). Through systematic analysis, we find that branch-and-merge is a central coordination mechanism for multi-agent collaboration, and that SWE primitives such as git worktree, git commit, and git merge enable it to be realized in a reliable and executable manner.
>
---
#### [new 066] Diffutron: A Masked Diffusion Language Model for Turkish Language
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决土耳其语非自回归文本生成问题。提出Diffutron模型，结合掩码扩散与多阶段调优，实现高效生成。**

- **链接: [https://arxiv.org/pdf/2603.20466](https://arxiv.org/pdf/2603.20466)**

> **作者:** Şuayp Talha Kocabay; Talha Rüzgar Akkuş
>
> **摘要:** Masked Diffusion Language Models (MDLMs) have emerged as a compelling non-autoregressive alternative to standard large language models; however, their application to morphologically rich languages remains limited. In this paper, we introduce $\textit{Diffutron}$, a masked diffusion language model specifically designed for Turkish. Our approach leverages a resource-efficient training pipeline, starting with LoRA-based continual pre-training of a multilingual encoder on a large-scale corpus. To enable generative capabilities, we employ a progressive instruction-tuning strategy, sequentially adapting the model on general and task-specific instruction sets. Experimental results across comprehensive benchmarks demonstrate that, despite its compact size, our model achieves competitive performance compared to existing multi-billion-parameter baselines. These findings validate the effectiveness of masked diffusion modeling combined with multi-stage tuning for non-autoregressive text generation in Turkish.
>
---
#### [new 067] Multi-Agent Debate with Memory Masking
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于推理任务，针对多智能体辩论中错误记忆影响性能的问题，提出MAD-M$^2$框架，通过记忆屏蔽提升推理鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.20215](https://arxiv.org/pdf/2603.20215)**

> **作者:** Hongduan Tian; Xiao Feng; Ziyuan Zhao; Xiangyu Zhu; Rolan Yan; Bo Han
>
> **备注:** ICLR 2026
>
> **摘要:** Large language models (LLMs) have recently demonstrated impressive capabilities in reasoning tasks. Currently, mainstream LLM reasoning frameworks predominantly focus on scaling up inference-time sampling to enhance performance. In particular, among all LLM reasoning frameworks, *multi-agent debate* (MAD), which employs multiple LLMs as agents to perform reasoning in the way of multi-round debate, has emerged as a powerful reasoning paradigm since it allows agents to access previous memories to alleviate fallacious content and refine their reasoning iteratively in each debate round. However, although MAD significantly improves the reasoning capabilities of LLMs, in this paper, we observe that there remain erroneous memories, and LLM agents are vulnerable to these erroneous memories. To explore this phenomenon, we provide a theoretical insight that the performance of MAD is highly dependent on the quality of memories derived from the previous debate, indicating that the existence of erroneous memories poses a threat to the performance of MAD. To address this problem, we introduce a simple yet effective multi-agent debate framework, *multi-agent debate with memory masking* (MAD-M$^2$), to improve the robustness of MAD by allowing LLM agents to mask erroneous memories from the previous debate round at the beginning of each debate round. In this way, MAD-M$^2$ can polish the contextual information before each debate round by preserving informative and meaningful memories while discarding the erroneous memories. Extensive experiments and analyses on mainstream mathematical and logical reasoning benchmarks demonstrate that MAD-M$^2$ can identify the erroneous memories and achieve better performance in reasoning than MAD.
>
---
#### [new 068] Agentic Automation of BT-RADS Scoring: End-to-End Multi-Agent System for Standardized Brain Tumor Follow-up Assessment
- **分类: cs.CL; cs.MA**

- **简介: 该论文属于医学影像分析任务，旨在解决脑肿瘤随访评估中的标准化问题。通过多智能体系统自动完成BT-RADS分类，提升评估准确性与一致性。**

- **链接: [https://arxiv.org/pdf/2603.21494](https://arxiv.org/pdf/2603.21494)**

> **作者:** Mohamed Sobhi Jabal; Jikai Zhang; Dominic LaBella; Jessica L. Houk; Dylan Zhang; Jeffrey D. Rudie; Kirti Magudia; Maciej A. Mazurowski; Evan Calabrese
>
> **备注:** 17 pages, 5 figures, 4 tables, 2 supplementary figures, 3 supplementary tables
>
> **摘要:** The Brain Tumor Reporting and Data System (BT-RADS) standardizes post-treatment MRI response assessment in patients with diffuse gliomas but requires complex integration of imaging trends, medication effects, and radiation timing. This study evaluates an end-to-end multi-agent large language model (LLM) and convolutional neural network (CNN) system for automated BT-RADS classification. A multi-agent LLM system combined with automated CNN-based tumor segmentation was retrospectively evaluated on 509 consecutive post-treatment glioma MRI examinations from a single high-volume center. An extractor agent identified clinical variables (steroid status, bevacizumab status, radiation date) from unstructured clinical notes, while a scorer agent applied BT-RADS decision logic integrating extracted variables with volumetric measurements. Expert reference standard classifications were established by an independent board-certified neuroradiologist. Of 509 examinations, 492 met inclusion criteria. The system achieved 374/492 (76.0%; 95% CI, 72.1%-79.6%) accuracy versus 283/492 (57.5%; 95% CI, 53.1%-61.8%) for initial clinical assessments (+18.5 percentage points; P<.001). Context-dependent categories showed high sensitivity (BT-1b 100%, BT-1a 92.7%, BT-3a 87.5%), while threshold-dependent categories showed moderate sensitivity (BT-3c 74.8%, BT-2 69.2%, BT-4 69.3%, BT-3b 57.1%). For BT-4, positive predictive value was 92.9%. The multi-agent LLM system achieved higher BT-RADS classification agreement with expert reference standard compared to initial clinical scoring, with high accuracy for context-dependent scores and high positive predictive value for BT-4 detection.
>
---
#### [new 069] Hear Both Sides: Efficient Multi-Agent Debate via Diversity-Aware Message Retention
- **分类: cs.CL**

- **简介: 该论文属于多智能体推理任务，旨在解决辩论中信息冗余和噪声问题。提出DAR框架，通过选择差异大的消息传播，提升辩论效果。**

- **链接: [https://arxiv.org/pdf/2603.20640](https://arxiv.org/pdf/2603.20640)**

> **作者:** Manh Nguyen; Anh Nguyen; Dung Nguyen; Svetha Venkatesh; Hung Le
>
> **摘要:** Multi-Agent Debate has emerged as a promising framework for improving the reasoning quality of large language models through iterative inter-agent communication. However, broadcasting all agent messages at every round introduces noise and redundancy that can degrade debate quality and waste computational resources. Current approaches rely on uncertainty estimation to filter low-confidence responses before broadcasting, but this approach is unreliable due to miscalibrated confidence scores and sensitivity to threshold selection. To address this, we propose Diversity-Aware Retention (DAR), a lightweight debate framework that, at each debate round, selects the subset of agent responses that maximally disagree with each other and with the majority vote before broadcasting. Through an explicit index-based retention mechanism, DAR preserves the original messages without modification, ensuring that retained disagreements remain authentic. Experiments on diverse reasoning and question answering benchmarks demonstrate that our selective message propagation consistently improves debate performance, particularly as the number of agents scales, where noise accumulation is most severe. Our results highlight that what agents hear is as important as what agents say in multi-agent reasoning systems.
>
---
#### [new 070] Conspiracy Frame: a Semiotically-Driven Approach for Conspiracy Theories Detection
- **分类: cs.CL**

- **简介: 该论文属于阴谋论检测任务，旨在识别和理解阴谋叙事。提出“阴谋框架”方法，结合语义学与符号学，构建标注数据集，探索框架对检测的潜在作用。**

- **链接: [https://arxiv.org/pdf/2603.21368](https://arxiv.org/pdf/2603.21368)**

> **作者:** Heidi Campana Piva; Shaina Ashraf; Maziar Kianimoghadam Jouneghani; Arianna Longo; Rossana Damiano; Lucie Flek; Marco Antonio Stranisci
>
> **摘要:** Conspiracy theories are anti-authoritarian narratives that lead to social conflict, impacting how people perceive political information. To help in understanding this issue, we introduce the Conspiracy Frame: a fine-grained semantic representation of conspiratorial narratives derived from frame-semantics and semiotics, which spawned the Conspiracy Frames (this http URL.) dataset: a corpus of Telegram messages annotated at span-level. The Conspiracy Frame and this http URL. dataset contribute to the implementation of a more generalizable understanding and recognition of conspiracy theories. We observe the ability of LLMs to recognize this phenomenon in-domain and out-of-domain, investigating the role that frames may have in supporting this task. Results show that, while the injection of frames in an in-context approach does not lead to clear increase of performance, it has potential; the mapping of annotated spans with FrameNet shows abstract semantic patterns (e.g., `Kinship', `Ingest\_substance') that potentially pave the way for a more semantically- and semiotically-aware detection of conspiratorial narratives.
>
---
#### [new 071] TaigiSpeech: A Low-Resource Real-World Speech Intent Dataset and Preliminary Results with Scalable Data Mining In-the-Wild
- **分类: cs.CL; cs.LG; eess.AS**

- **简介: 该论文介绍了一个针对低资源语言的语音意图数据集TaigiSpeech，用于解决语音技术中语言资源不足的问题。通过数据挖掘策略构建数据集，支持医疗和家庭助手等应用场景。**

- **链接: [https://arxiv.org/pdf/2603.21478](https://arxiv.org/pdf/2603.21478)**

> **作者:** Kai-Wei Chang; Yi-Cheng Lin; Huang-Cheng Chou; Wenze Ren; Yu-Han Huang; Yun-Shao Tsai; Chien-Cheng Chen; Yu Tsao; Yuan-Fu Liao; Shrikanth Narayanan; James Glass; Hung-yi Lee
>
> **备注:** submitted to Interspeech 2026
>
> **摘要:** Speech technologies have advanced rapidly and serve diverse populations worldwide. However, many languages remain underrepresented due to limited resources. In this paper, we introduce \textbf{TaigiSpeech}, a real-world speech intent dataset in Taiwanese Taigi (aka Taiwanese Hokkien/Southern Min), which is a low-resource and primarily spoken language. The dataset is collected from older adults, comprising 21 speakers with a total of 3k utterances. It is designed for practical intent detection scenarios, including healthcare and home assistant applications. To address the scarcity of labeled data, we explore two data mining strategies with two levels of supervision: keyword match data mining with LLM pseudo labeling via an intermediate language and an audio-visual framework that leverages multimodal cues with minimal textual supervision. This design enables scalable dataset construction for low-resource and unwritten spoken languages. TaigiSpeech will be released under the CC BY 4.0 license to facilitate broad adoption and research on low-resource and unwritten languages. The project website and the dataset can be found on this https URL.
>
---
#### [new 072] Semantic Shift: the Fundamental Challenge in Text Embedding and Retrieval
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于文本嵌入与检索任务，解决嵌入模型中的语义漂移问题。通过分析语义平滑现象，提出语义漂移度量，揭示其对检索性能的影响。**

- **链接: [https://arxiv.org/pdf/2603.21437](https://arxiv.org/pdf/2603.21437)**

> **作者:** Hang Gao; Dimitris N. Metaxas
>
> **摘要:** Transformer-based embedding models rely on pooling to map variable-length text into a single vector, enabling efficient similarity search but also inducing well-known geometric pathologies such as anisotropy and length-induced embedding collapse. Existing accounts largely describe \emph{what} these pathologies look like, yet provide limited insight into \emph{when} and \emph{why} they harm downstream retrieval. In this work, we argue that the missing causal factor is \emph{semantic shift}: the intrinsic, structured evolution and dispersion of semantics within a text. We first present a theoretical analysis of \emph{semantic smoothing} in Transformer embeddings: as the semantic diversity among constituent sentences increases, the pooled representation necessarily shifts away from every individual sentence embedding, yielding a smoothed and less discriminative vector. Building on this foundation, we formalize semantic shift as a computable measure integrating local semantic evolution and global semantic dispersion. Through controlled experiments across corpora and multiple embedding models, we show that semantic shift aligns closely with the severity of embedding concentration and predicts retrieval degradation, whereas text length alone does not. Overall, semantic shift offers a unified and actionable lens for understanding embedding collapse and for diagnosing when anisotropy becomes harmful.
>
---
#### [new 073] A Modular LLM Framework for Explainable Price Outlier Detection
- **分类: cs.CL; cs.CE**

- **简介: 该论文属于价格异常检测任务，旨在解决产品价格异常识别问题。通过构建基于大语言模型的框架，结合语义分析和推理，实现可解释的价格异常判断。**

- **链接: [https://arxiv.org/pdf/2603.20636](https://arxiv.org/pdf/2603.20636)**

> **作者:** Shadi Sartipi; John Wu; Sina Ghotbi; Nikhita Vedula; Shervin Malmasi
>
> **备注:** 13 pages, 3 figures
>
> **摘要:** Detecting product price outliers is important for retail and e-commerce stores as erroneous or unexpectedly high prices adversely affect competitiveness, revenue, and consumer trust. Classical techniques offer simple thresholds while ignoring the rich semantic relationships among product attributes. We propose an agentic Large Language Model (LLM) framework that treats outlier price flagging as a reasoning task grounded in related product detection and comparison. The system processes the prices of target products in three stages: (i) relevance classification selects price-relevant similar products using product descriptions and attributes; (ii) relative utility assessment evaluates the target product against each similar product along price influencing dimensions (e.g., brand, size, features); (iii) reasoning-based decision aggregates these justifications into an explainable price outlier judgment. The framework attains over 75% agreement with human auditors on a test dataset, and outperforms zero-shot and retrieval based LLM techniques. Ablation studies show the sensitivity of the method to key hyper-parameters and testify on its flexibility to be applied to cases with different accuracy requirement and auditor agreements.
>
---
#### [new 074] TAMTRL: Teacher-Aligned Reward Reshaping for Multi-Turn Reinforcement Learning in Long-Context Compression
- **分类: cs.CL**

- **简介: 该论文属于长文本压缩任务，解决多轮强化学习中的时间信用分配问题。通过教师对齐奖励重塑方法（TAMTRL），提升多轮记忆更新的精度与效果。**

- **链接: [https://arxiv.org/pdf/2603.21663](https://arxiv.org/pdf/2603.21663)**

> **作者:** Li Wang; Yandong Wang; Xin Yu; Kui Zhang; Tianhao Peng; Wenjun Wu
>
> **摘要:** The rapid progress of large language models (LLMs) has led to remarkable performance gains across a wide range of tasks. However, when handling long documents that exceed the model's context window limit, the entire context cannot be processed in a single pass, making chunk-wise processing necessary. This requires multiple turns to read different chunks and update memory. However, supervision is typically provided only by the final outcome, which makes it difficult to evaluate the quality of memory updates at each turn in the multi-turn training setting. This introduces a temporal credit assignment challenge. Existing approaches, such as LLM-as-a-judge or process reward models, incur substantial computational overhead and suffer from estimation noise. To better address the credit assignment problem in multi-turn memory training, we propose Teacher-Aligned Reward Reshaping for Multi-Turn Reinforcement Learning (TAMTRL). TAMTRL leverages relevant documents as teacher signals by aligning them with each turn of model input and assigns rewards through normalized probabilities in a self-supervised manner. This provides fine-grained learning signals for each memory update and improves long-context processing. Experiments with multiple models of varying scales across seven long-context benchmarks show that TAMTRL consistently outperforms strong baselines, demonstrating its effectiveness. Our code is available at this https URL.
>
---
#### [new 075] Graph Fusion Across Languages using Large Language Models
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于跨语言知识图谱融合任务，旨在解决多语言知识图谱间的语义差异问题。通过将三元组转化为自然语言序列，利用大语言模型进行实体和关系对齐，实现多图的连续融合。**

- **链接: [https://arxiv.org/pdf/2603.21248](https://arxiv.org/pdf/2603.21248)**

> **作者:** Kaung Myat Kyaw; Khush Agarwal; Jonathan Chan
>
> **摘要:** Combining multiple knowledge graphs (KGs) across linguistic boundaries is a persistent challenge due to semantic heterogeneity and the complexity of graph environments. We propose a framework for cross-lingual graph fusion, leveraging the in-context reasoning and multilingual semantic priors of Large Language Models (LLMs). The framework implements structural linearization by mapping triplets directly into natural language sequences (e.g., [head] [relation] [tail]), enabling the LLM to map relations and reconcile entities between an evolving fused graph ($G_{c}^{(t-1)}$) and a new candidate graph ($G_{t}$). Evaluated on the DBP15K dataset, this exploratory study demonstrates that LLMs can serve as a universal semantic bridge to resolve cross-lingual discrepancies. Results show the successful sequential agglomeration of multiple heterogeneous graphs, offering a scalable, modular solution for continuous knowledge synthesis in multi-source, multilingual environments.
>
---
#### [new 076] Alignment Whack-a-Mole : Finetuning Activates Verbatim Recall of Copyrighted Books in Large Language Models
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于AI安全领域，揭示了微调导致大模型泄露受版权保护书籍的问题，通过实验验证了模型权重中存储了版权内容，并指出现有防护措施失效。**

- **链接: [https://arxiv.org/pdf/2603.20957](https://arxiv.org/pdf/2603.20957)**

> **作者:** Xinyue Liu; Niloofar Mireshghallah; Jane C. Ginsburg; Tuhin Chakrabarty
>
> **备注:** Preprint Under Review
>
> **摘要:** Frontier LLM companies have repeatedly assured courts and regulators that their models do not store copies of training data. They further rely on safety alignment strategies via RLHF, system prompts, and output filters to block verbatim regurgitation of copyrighted works, and have cited the efficacy of these measures in their legal defenses against copyright infringement claims. We show that finetuning bypasses these protections: by training models to expand plot summaries into full text, a task naturally suited for commercial writing assistants, we cause GPT-4o, Gemini-2.5-Pro, and DeepSeek-V3.1 to reproduce up to 85-90% of held-out copyrighted books, with single verbatim spans exceeding 460 words, using only semantic descriptions as prompts and no actual book text. This extraction generalizes across authors: finetuning exclusively on Haruki Murakami's novels unlocks verbatim recall of copyrighted books from over 30 unrelated authors. The effect is not specific to any training author or corpus: random author pairs and public-domain finetuning data produce comparable extraction, while finetuning on synthetic text yields near-zero extraction, indicating that finetuning on individual authors' works reactivates latent memorization from pretraining. Three models from different providers memorize the same books in the same regions ($r \ge 0.90$), pointing to an industry-wide vulnerability. Our findings offer compelling evidence that model weights store copies of copyrighted works and that the security failures that manifest after finetuning on individual authors' works undermine a key premise of recent fair use rulings, where courts have conditioned favorable outcomes on the adequacy of measures preventing reproduction of protected expression.
>
---
#### [new 077] Riding Brainwaves in LLM Space: Understanding Activation Patterns Using Individual Neural Signatures
- **分类: cs.CL**

- **简介: 该论文研究EEG信号与语言模型激活模式的关系，任务为个体神经特征建模。解决如何利用冻结语言模型捕捉个体特定脑电特征的问题，通过训练个性化探针实现高效预测。**

- **链接: [https://arxiv.org/pdf/2603.21847](https://arxiv.org/pdf/2603.21847)**

> **作者:** Ajan Subramanian; Sumukh Bettadapura; Rohan Sathish
>
> **摘要:** Consumer-grade EEG is entering everyday devices, from earbuds to headbands, raising the question of whether language models can be adapted to individual neural responses. We test this by asking whether frozen LLM representations encode person-specific EEG signals, directions in activation space that predict one person's brain activity but not another's. Using word-level EEG from 30 participants reading naturalistic sentences (ZuCo corpus), we train a separate linear probe for each person, mapping hidden states from a frozen Qwen 2.5 7B to that individual's EEG power. Person-specific probes outperform a single population probe on every EEG feature tested; for high-gamma power, the person-specific probe achieves rho = 0.183, a ninefold improvement over the population probe (rho = 0.020, p < 10^-4). A negative control, fixation count, shows no person-specific advantage (p = 0.360); fixation count reflects word length and frequency rather than individual cognition. The individual directions are temporally stable (split-half cosine = 0.824), non-transferable across people (self rho = 0.369 vs. other rho = 0.143, p < 10^-19), and distinct from the shared population signal: person-specific probes retain predictive power after the population component is removed. The person-specific signal concentrates in the model's deep layers, rising consistently with depth and peaking at Layer 24 of 28. The results are consistent across architectures (LLaMA 3.1 8B) and survive word-level confound controls. Frozen language models contain stable, person-specific neural directions in their deep layers, providing a geometric foundation for EEG-driven personalization.
>
---
#### [new 078] Multiperspectivity as a Resource for Narrative Similarity Prediction
- **分类: cs.CL**

- **简介: 该论文属于叙事相似性预测任务，旨在解决多视角解释导致的评价基准单一问题。通过构建31个LLM角色集合，提升预测准确性，并发现性别相关解释影响效果。**

- **链接: [https://arxiv.org/pdf/2603.22103](https://arxiv.org/pdf/2603.22103)**

> **作者:** Max Upravitelev; Veronika Solopova; Jing Yang; Charlott Jakob; Premtim Sahitaj; Ariana Sahitaj; Vera Schmitt
>
> **摘要:** Predicting narrative similarity can be understood as an inherently interpretive task: different, equally valid readings of the same text can produce divergent interpretations and thus different similarity judgments, posing a fundamental challenge for semantic evaluation benchmarks that encode a single ground truth. Rather than treating this multiperspectivity as a challenge to overcome, we propose to incorporate it in the decision making process of predictive systems. To explore this strategy, we created an ensemble of 31 LLM personas. These range from practitioners following interpretive frameworks to more intuitive, lay-style characters. Our experiments were conducted on the SemEval-2026 Task 4 dataset, where the system achieved an accuracy score of 0.705. Accuracy improves with ensemble size, consistent with Condorcet Jury Theorem-like dynamics under weakened independence. Practitioner personas perform worse individually but produce less correlated errors, yielding larger ensemble gains under majority voting. Our error analysis reveals a consistent negative association between gender-focused interpretive vocabulary and accuracy across all persona categories, suggesting either attention to dimensions not relevant for the benchmark or valid interpretations absent from the ground truth. This finding underscores the need for evaluation frameworks that account for interpretive plurality.
>
---
#### [new 079] SciNav: A General Agent Framework for Scientific Coding Tasks
- **分类: cs.CL; cs.AI; cs.CE; cs.LG; cs.MA; eess.SY**

- **简介: 该论文提出SciNav框架，解决科学编码任务中的有效解探索问题。通过相对判断引导的拓扑搜索，提升编码质量与效率。**

- **链接: [https://arxiv.org/pdf/2603.20256](https://arxiv.org/pdf/2603.20256)**

> **作者:** Tianshu Zhang; Huan Sun
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Autonomous science agents built on large language models (LLMs) are increasingly used to generate hypotheses, design experiments, and produce reports. However, prior work mainly targets open-ended scientific problems with subjective outputs that are difficult to evaluate. Scientific coding benchmarks, by contrast, provide executable outputs for objective assessment. Existing approaches remain engineering-driven pipelines, revealing the need for structured, end-to-end science agent frameworks for scientific coding tasks. We address this gap by focusing on scientific coding tasks, where evaluation can be made rigorously, and introducing an agent framework SciNav (Scientific Navigator) that enables more effective solution exploration. Our framework is designed to operate under constrained search budgets, moving beyond reliance on pre-defined success metrics and prolonged search cycles. Inspired by findings that comparative judgments often reveal finer-grained quality differences and therefore provide greater discriminative power than absolute scoring, our framework leverages pairwise relative judgments within a tree search process to select top-K promising solution branches, prune low-potential ones, and progressively narrow down the solution candidates on the selected branches guided by relative comparisons. We demonstrate our agent's effectiveness across different types of tasks on two benchmarks. Experiments show that SciNav significantly outperforms direct prompting and prior agents like OpenHands and Self-Debug across different base models, task types, and difficulty levels, and exceeds different frontier comparators such as random selection and LLM absolute scoring. These results confirm the strength of our agent design and highlight the effectiveness of relative judgment-guided top-K search for high-quality scientific coding, marking a step toward more practical science agents.
>
---
#### [new 080] Ara-Best-RQ: Multi Dialectal Arabic SSL
- **分类: cs.CL**

- **简介: 该论文提出Ara-BEST-RQ模型，解决多方言阿拉伯语语音处理问题。通过自监督学习，提升方言识别和语音识别性能。**

- **链接: [https://arxiv.org/pdf/2603.21900](https://arxiv.org/pdf/2603.21900)**

> **作者:** Haroun Elleuch; Ryan Whetten; Salima Mdhaffar; Yannick Estève; Fethi Bougares
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** We present Ara-BEST-RQ, a family of self-supervised learning (SSL) models specifically designed for multi-dialectal Arabic speech processing. Leveraging 5,640 hours of crawled Creative Commons speech and combining it with publicly available datasets, we pre-train conformer-based BEST-RQ models up to 600M parameters. Our models are evaluated on dialect identification (DID) and automatic speech recognition (ASR) tasks, achieving state-of-the-art performance on the former while using fewer parameters than competing models. We demonstrate that family-targeted pre-training on Arabic dialects significantly improves downstream performance compared to multilingual or monolingual models trained on non-Arabic data. All models, code, and pre-processed datasets will be publicly released to support reproducibility and further research in Arabic speech technologies.
>
---
#### [new 081] The Hidden Puppet Master: A Theoretical and Real-World Account of Emotional Manipulation in LLMs
- **分类: cs.CL**

- **简介: 该论文属于AI伦理研究任务，旨在解决LLMs中情感操控问题。通过理论分析与实验，揭示隐藏激励对用户信念的影响，并评估模型预测能力。**

- **链接: [https://arxiv.org/pdf/2603.20907](https://arxiv.org/pdf/2603.20907)**

> **作者:** Jocelyn Shen; Amina Luvsanchultem; Jessica Kim; Kynnedy Smith; Valdemar Danry; Kantwon Rogers; Sharifa Alghowinem; Hae Won Park; Maarten Sap; Cynthia Breazeal
>
> **摘要:** As users increasingly turn to LLMs for practical and personal advice, they become vulnerable to being subtly steered toward hidden incentives misaligned with their own interests. Prior works have benchmarked persuasion and manipulation detection, but these efforts rely on simulated or debate-style settings, remain uncorrelated with real human belief shifts, and overlook a critical dimension: the morality of hidden incentives driving the manipulation. We introduce PUPPET, a theoretical taxonomy of personalized emotional manipulation in LLM-human dialogues that centers around incentive morality, and conduct a human study with N=1,035 participants across realistic everyday queries, varying personalization and incentive direction (harmful versus prosocial). We find that harmful hidden incentives produce significantly larger belief shifts than prosocial ones. Finally, we benchmark LLMs on the task of belief prediction, finding that models exhibit moderate predictive ability of belief change based on conversational contexts (r=0.3 - 0.5), but they also systematically underestimate the magnitude of belief shift. Together, this work establishes a theoretically grounded and behaviorally validated foundation for studying, and ultimately combatting, incentive-driven manipulation in LLMs during everyday, practical user queries.
>
---
#### [new 082] Abjad-Kids: An Arabic Speech Classification Dataset for Primary Education
- **分类: cs.CL; cs.HC; cs.LG; cs.SD; eess.AS**

- **简介: 该论文提出Abjad-Kids数据集，用于阿拉伯语儿童语音分类任务，解决低资源语言儿童语音数据不足的问题。通过CNN-LSTM模型进行分类研究。**

- **链接: [https://arxiv.org/pdf/2603.20255](https://arxiv.org/pdf/2603.20255)**

> **作者:** Abdul Aziz Snoubara; Baraa Al_Maradni; Haya Al_Naal; Malek Al_Madrmani; Roaa Jdini; Seedra Zarzour; Khloud Al Jallad
>
> **摘要:** Speech-based AI educational applications have gained significant interest in recent years, particularly for children. However, children speech research remains limited due to the lack of publicly available datasets, especially for low-resource languages such as this http URL paper presents Abjad-Kids, an Arabic speech dataset designed for kindergarten and primary education, focusing on fundamental learning of alphabets, numbers, and colors. The dataset consists of 46397 audio samples collected from children aged 3 - 12 years, covering 141 classes. All samples were recorded under controlled specifications to ensure consistency in duration, sampling rate, and format. To address high intra-class similarity among Arabic phonemes and the limited samples per class, we propose a hierarchical audio classification based on CNN-LSTM architectures. Our proposed methodology decomposes alphabet recognition into a two-stage process: an initial grouping classification model followed by specialized classifiers for each group. Both strategies: static linguistic-based grouping and dynamic clustering-based grouping, were evaluated. Experimental results demonstrate that static linguistic-based grouping achieves superior performance. Comparisons between traditional machine learning with deep learning approaches, highlight the effectiveness of CNN-LSTM models combined with data augmentation. Despite achieving promising results, most of our experiments indicate a challenge with overfitting, which is likely due to the limited number of samples, even after data augmentation and model regularization. Thus, future work may focus on collecting additional data to address this issue. Abjad-Kids will be publicly available. We hope that Abjad-Kids enrich children representation in speech dataset, and be a good resource for future research in Arabic speech classification for kids.
>
---
#### [new 083] MzansiText and MzansiLM: An Open Corpus and Decoder-Only Language Model for South African Languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决南非洲低资源语言的模型适配问题。作者构建了MzansiText语料库和MzansiLM模型，评估其在多种任务上的表现。**

- **链接: [https://arxiv.org/pdf/2603.20732](https://arxiv.org/pdf/2603.20732)**

> **作者:** Anri Lombard; Simbarashe Mawere; Temi Aina; Ethan Wolff; Sbonelo Gumede; Elan Novick; Francois Meyer; Jan Buys
>
> **备注:** 15 pages, 11 tables, appendix included. Accepted at LREC 2026
>
> **摘要:** Decoder-only language models can be adapted to diverse tasks through instruction finetuning, but the extent to which this generalizes at small scale for low-resource languages remains unclear. We focus on the languages of South Africa, where we are not aware of a publicly available decoder-only model that explicitly targets all eleven official written languages, nine of which are low-resource. We introduce MzansiText, a curated multilingual pretraining corpus with a reproducible filtering pipeline, and MzansiLM, a 125M-parameter language model trained from scratch. We evaluate MzansiLM on natural language understanding and generation using three adaptation regimes: monolingual task-specific finetuning, multilingual task-specific finetuning, and general multi-task instruction finetuning. Monolingual task-specific finetuning achieves strong performance on data-to-text generation, reaching 20.65 BLEU on isiXhosa and competing with encoder-decoder baselines over ten times larger. Multilingual task-specific finetuning benefits closely related languages on topic classification, achieving 78.5% macro-F1 on isiXhosa news classification. While MzansiLM adapts effectively to supervised NLU and NLG tasks, few-shot reasoning remains challenging at this model size, with performance near chance even for much larger decoder-only models. We release MzansiText and MzansiLM to provide a reproducible decoder-only baseline and clear guidance on adaptation strategies for South African languages at small scale.
>
---
#### [new 084] Autoregressive vs. Masked Diffusion Language Models: A Controlled Comparison
- **分类: cs.CL**

- **简介: 该论文对比了自回归与掩码扩散语言模型，旨在分析两者在训练效率、收敛速度和生成质量上的差异。任务为语言模型生成方法比较。**

- **链接: [https://arxiv.org/pdf/2603.22075](https://arxiv.org/pdf/2603.22075)**

> **作者:** Caio Vicentino
>
> **备注:** 10 pages, 2 figures, 4 tables. Code and checkpoints at this https URL
>
> **摘要:** We present a controlled empirical comparison between autoregressive (AR) and masked diffusion (MDLM) language models. Both models are trained on identical data (50M tokens from TinyStories), identical compute budget (20,000 steps, batch size 32, sequence length 512), and identical hardware (NVIDIA H100 80GB), isolating the generation paradigm as the sole variable. We report three findings. First, both paradigms achieve comparable training throughput (~50K tokens/second), with MDLM requiring only 4.7% more wall-clock time. Second, AR converges faster and begins overfitting by step 14,000, while MDLM converges more slowly and is still improving at step 20,000, suggesting different compute-optimal training regimes. Third, quantitative diversity analysis over 1,000 generated samples reveals a structural diversity-fluency trade-off: AR produces fluent but repetitive outputs (99.8% begin with the same word), while MDLM generates more diverse narratives (93.4% unique 5-word openings, higher Distinct-n, lower Self-BLEU), at the cost of occasional grammatical inconsistencies. All code, trained checkpoints, and data pipelines are released for reproducibility.
>
---
#### [new 085] Fast-Slow Thinking RM: Efficient Integration of Scalar and Generative Reward Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决奖励模型效率与性能的平衡问题。提出F/S-RM融合快慢思维机制，提升效果并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2603.20212](https://arxiv.org/pdf/2603.20212)**

> **作者:** Jiayun Wu; Peixu Hou; Shan Qu; Peng Zhang; Ning Gu; Tun Lu
>
> **摘要:** Reward models (RMs) are critical for aligning Large Language Models via Reinforcement Learning from Human Feedback (RLHF). While Generative Reward Models (GRMs) achieve superior accuracy through chain-of-thought (CoT) reasoning, they incur substantial computational costs. Conversely, Scalar Reward Models (SRMs) offer efficiency but suffer from limited performance and adaptability in complex scenarios. We introduce Fast-Slow Thinking Reward Models (F/S-RM), a hybrid RM architecture inspired by Dual Process Theory. It trains a single model to integrate two distinct reward paradigms: first-token prediction as a scalar score (fast thinking) and CoT-based judgment (slow thinking), regulated by a dual-confidence activation mechanism that determines when to activate slow thinking. F/S-RM achieves a 1.2% relative performance improvement over state-of-the-art models while reducing token consumption by 20.8%. Code and data will be publicly available.
>
---
#### [new 086] enhancing reasoning accuracy in large language models during inference time
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型在推理任务中的准确性。通过三种推理增强方法，如自一致性、双模型验证和自省，实验表明自一致性效果最佳。**

- **链接: [https://arxiv.org/pdf/2603.21301](https://arxiv.org/pdf/2603.21301)**

> **作者:** Vinay Sharma; Manish Jain
>
> **摘要:** Large Language Models (LLMs) often exhibit strong linguistic abilities while remaining unreliable on multi-step reasoning tasks, particularly when deployed without additional training or fine-tuning. In this work, we study inference-time techniques to improve the reasoning accuracy of LLMs. We systematically evaluate three classes of inference-time strategies: (i) self-consistency via stochastic decoding, where the model is sampled multiple times using controlled temperature and nucleus sampling and the most frequent final answer is selected; (ii) dual-model reasoning agreement, where outputs from two independent models are compared and only consistent reasoning traces are trusted; and (iii) self-reflection, where the model critiques and revises its own reasoning. Across all evaluated methods, we employ Chain-of-Thought (CoT) [1] prompting to elicit explicit intermediate reasoning steps before generating final answers. In this work, we provide a controlled comparative evaluation across three inference-time strategies under identical prompting and verification settings. Our experiments on LLM [2] show that self-consistency with nucleus sampling and controlled temperature value yields the substantial gains, achieving a 9% to 15% absolute improvement in accuracy over greedy single-pass decoding, well-suited for low-risk domains, offering meaningful gains with minimal overhead. The dual-model approach provides additional confirmation for model reasoning steps thus more appropriate for moderate-risk domains, where higher reliability justifies additional compute. Self-reflection offers only marginal improvements, suggesting limited effectiveness for smaller non-reasoning models at inference time.
>
---
#### [new 087] More Than Sum of Its Parts: Deciphering Intent Shifts in Multimodal Hate Speech Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态仇恨言论检测任务，旨在解决隐性仇恨言论识别难题。通过构建H-VLI基准和提出ARCADE框架，提升模型对模态交互中隐含意图的捕捉能力。**

- **链接: [https://arxiv.org/pdf/2603.21298](https://arxiv.org/pdf/2603.21298)**

> **作者:** Runze Sun; Yu Zheng; Zexuan Xiong; Zhongjin Qu; Lei Chen; Jiwen Lu; Jie Zhou
>
> **摘要:** Combating hate speech on social media is critical for securing cyberspace, yet relies heavily on the efficacy of automated detection systems. As content formats evolve, hate speech is transitioning from solely plain text to complex multimodal expressions, making implicit attacks harder to spot. Current systems, however, often falter on these subtle cases, as they struggle with multimodal content where the emergent meaning transcends the aggregation of individual modalities. To bridge this gap, we move beyond binary classification to characterize semantic intent shifts where modalities interact to construct implicit hate from benign cues or neutralize toxicity through semantic inversion. Guided by this fine-grained formulation, we curate the Hate via Vision-Language Interplay (H-VLI) benchmark where the true intent hinges on the intricate interplay of modalities rather than overt visual or textual slurs. To effectively decipher these complex cues, we further propose the Asymmetric Reasoning via Courtroom Agent DEbate (ARCADE) framework. By simulating a judicial process where agents actively argue for accusation and defense, ARCADE forces the model to scrutinize deep semantic cues before reaching a verdict. Extensive experiments demonstrate that ARCADE significantly outperforms state-of-the-art baselines on H-VLI, particularly for challenging implicit cases, while maintaining competitive performance on established benchmarks. Our code and data are available at: this https URL
>
---
#### [new 088] Instruction Set and Language for Symbolic Regression
- **分类: cs.CL; cs.AI; cs.PL**

- **简介: 该论文属于符号回归任务，解决结构冗余问题。提出IsalSR框架，通过编码表达式DAG为规范字符串，消除等价表示的多样性。**

- **链接: [https://arxiv.org/pdf/2603.21836](https://arxiv.org/pdf/2603.21836)**

> **作者:** Ezequiel Lopez-Rubio; Mario Pascual-Gonzalez
>
> **摘要:** A fundamental but largely unaddressed obstacle in Symbolic regression (SR) is structural redundancy: every expression DAG with admits many distinct node-numbering schemes that all encode the same expression, each occupying a separate point in the search space and consuming fitness evaluations without adding diversity. We present IsalSR (Instruction Set and Language for Symbolic Regression), a representation framework that encodes expression DAGs as strings over a compact two-tier alphabet and computes a pruned canonical string -- a complete labeled-DAG isomorphism invariant -- that collapses all the equivalent representations into a single canonical form.
>
---
#### [new 089] Permutation-Consensus Listwise Judging for Robust Factuality Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于事实性评估任务，解决候选答案顺序对判断结果的影响问题。通过多次排列并聚合结果，提升评估可靠性。**

- **链接: [https://arxiv.org/pdf/2603.20562](https://arxiv.org/pdf/2603.20562)**

> **作者:** Tianyi Huang; Nathan Huang; Justin Tang; Wenqian Chen; Elsa Fan
>
> **摘要:** Large language models (LLMs) are now widely used as judges, yet their decisions can change under presentation choices that should be irrelevant. We study one such source of instability: candidate-order sensitivity in listwise factuality evaluation, where several answers can look similarly polished while differing sharply in hallucination risk. We introduce PCFJudge, an inference-time method that reruns the same factuality-first listwise prompt over multiple orderings of the same candidate set and aggregates the resulting scores, ranks, and uncertainty signals into a single consensus decision. On RewardBench 2 Factuality, PCFJudge improves over direct judging by up to 7 absolute points. Development ablations show that the dominant gain comes from permutation consensus itself rather than from heavier arbitration layers. These results suggest that a meaningful share of factuality-judging error arises from order instability, and that averaging over this nuisance variation is a simple and effective way to make LLM evaluation more reliable.
>
---
#### [new 090] SemEval-2026 Task 12: Abductive Event Reasoning: Towards Real-World Event Causal Inference for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文介绍SemEval-2026 Task 12，旨在解决现实事件因果推理问题。任务要求系统从证据中找出最可能的直接原因，聚焦于因果推理与多文档理解挑战。**

- **链接: [https://arxiv.org/pdf/2603.21720](https://arxiv.org/pdf/2603.21720)**

> **作者:** Pengfei Cao; Mingxuan Yang; Yubo Chen; Chenlong Zhang; Mingxuan Liu; Kang Liu; Jun Zhao
>
> **备注:** 9 pages, 3 figures, semeval 2026 task 12 description paper
>
> **摘要:** Understanding why real-world events occur is important for both natural language processing and practical decision-making, yet direct-cause inference remains underexplored in evidence-rich settings. To address this gap, we organized SemEval-2026 Task 12: Abductive Event Reasoning (AER).\footnote{The task data is available at this https URL} The task asks systems to identify the most plausible direct cause of a target event from supporting evidence. We formulate AER as an evidence-grounded multiple-choice benchmark that captures key challenges of real-world causal reasoning, including distributed evidence, indirect background factors, and semantically related but non-causal distractors. The shared task attracted 122 participants and received 518 submissions. This paper presents the task formulation, dataset construction pipeline, evaluation setup, and system results. AER provides a focused benchmark for abductive reasoning over real-world events and highlights challenges for future work on causal reasoning and multi-document understanding.
>
---
#### [new 091] Coding Agents are Effective Long-Context Processors
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究如何通过编码代理处理长文本上下文，解决LLM在长文本处理中的性能下降问题。工作包括评估编码代理在多个任务中的表现，证明其有效性。**

- **链接: [https://arxiv.org/pdf/2603.20432](https://arxiv.org/pdf/2603.20432)**

> **作者:** Weili Cao; Xunjian Yin; Bhuwan Dhingra; Shuyan Zhou
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable progress in scaling to access massive contexts. However, the access is via the latent and uninterpretable attention mechanisms, and LLMs fail to effective process long context, exhibiting significant performance degradation as context length increases. In this work, we study whether long-context processing can be externalized from latent attention into explicit, executable interactions, by allowing coding agents to organize text in file systems and manipulate it using its native tools. We evaluate off-the-shelf frontier coding agents as the general interface for tasks that require processing long contexts, including long-context reasoning, retrieval-augmented generation, and open-domain question answering with large-scale corpus contains up to three trillion tokens. Across multiple benchmarks, these agents outperform published state-of-the-art by 17.3% on average. We attribute this efficacy to two key factors: native tool proficiency, which enables agents to leverage executable code and terminal commands rather than passive semantic queries, and file system familiarity, which allows them to navigate massive text corpora as directory structures. These findings suggest that delegating long-context processing to coding agents offers an effective alternative to semantic search or context window scaling, opening new directions for long-context processing in LLMs.
>
---
#### [new 092] Politics of Questions in News: A Mixed-Methods Study of Interrogative Stances as Markers of Voice and Power
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于新闻话语分析任务，探讨新闻中疑问句的使用及其权力与声音的体现。通过混合方法研究，分析疑问句的功能、分布及与回答的关系，揭示其在新闻中的结构作用。**

- **链接: [https://arxiv.org/pdf/2603.21823](https://arxiv.org/pdf/2603.21823)**

> **作者:** Bros Victor; Barbini Matilde; Gerard Patrick; Gatica-Perez Daniel
>
> **备注:** ICWSM 2026
>
> **摘要:** Interrogatives in news discourse have been examined in linguistics and conversation analysis, but mostly in broadcast interviews and relatively small, often English-language corpora, while large-scale computational studies of news rarely distinguish interrogatives from declaratives or differentiate their functions. This paper brings these strands together through a mixed-methods study of the "Politics of Questions" in contemporary French-language digital news. Using over one million articles published between January 2023 and June 2024, we automatically detect interrogative stances, approximate their functional types, and locate textual answers when present, linking these quantitative measures to a qualitatively annotated subcorpus grounded in semantic and pragmatic theories of questions. Interrogatives are sparse but systematically patterned: they mainly introduce or organize issues, with most remaining cases being information-seeking or echo-like, while explicitly leading or tag questions are rare. Although their density and mix vary across outlets and topics, our heuristic suggests that questions are overwhelmingly taken up within the same article and usually linked to a subsequent answer-like span, most often in the journalist's narrative voice and less often through quoted speech. Interrogative contexts are densely populated with named individuals, organizations, and places, whereas publics and broad social groups are mentioned much less frequently, suggesting that interrogative discourse tends to foreground already prominent actors and places and thus exhibits strong personalization. We show how interrogative stance, textual uptake, and voice can be operationalized at corpus scale, and argue that combining computational methods with pragmatic and sociological perspectives can help account for how questioning practices structure contemporary news discourse.
>
---
#### [new 093] Context Selection for Hypothesis and Statistical Evidence Extraction from Full-Text Scientific Articles
- **分类: cs.CL; cs.AI; cs.DL**

- **简介: 该论文属于科学文本信息提取任务，旨在解决从全文中提取假设及其统计证据的问题。通过构建检索与抽取框架，研究不同上下文选择对提取效果的影响。**

- **链接: [https://arxiv.org/pdf/2603.21193](https://arxiv.org/pdf/2603.21193)**

> **作者:** Sai Koneru; Jian Wu; Sarah Rajtmajer
>
> **摘要:** Extracting hypotheses and their supporting statistical evidence from full-text scientific articles is central to the synthesis of empirical findings, but remains difficult due to document length and the distribution of scientific arguments across sections of the paper. The work studies a sequential full-text extraction setting, where the statement of a primary finding in an article's abstract is linked to (i) a corresponding hypothesis statement in the paper body and (ii) the statistical evidence that supports or refutes that hypothesis. This formulation induces a challenging within-document retrieval setting in which many candidate paragraphs are topically related to the finding but differ in rhetorical role, creating hard negatives for retrieval and extraction. Using a two-stage retrieve-and-extract framework, we conduct a controlled study of retrieval design choices, varying context quantity, context quality (standard Retrieval Augmented Generation, reranking, and a fine-tuned retriever paired with reranking), as well as an oracle paragraph setting to separate retrieval failures from extraction limits across four Large Language Model extractors. We find that targeted context selection consistently improves hypothesis extraction relative to full-text prompting, with gains concentrated in configurations that optimize retrieval quality and context cleanliness. In contrast, statistical evidence extraction remains substantially harder. Even with oracle paragraphs, performance remains moderate, indicating persistent extractor limitations in handling hybrid numeric-textual statements rather than retrieval failures alone.
>
---
#### [new 094] NoveltyAgent: Autonomous Novelty Reporting Agent with Point-wise Novelty Analysis and Self-Validation
- **分类: cs.CL**

- **简介: 该论文提出NoveltyAgent，用于自动检测论文新颖性，解决学术论文筛选中质量参差不齐的问题。通过细粒度分析和自检机制，提升新颖性报告的准确性和可靠性。**

- **链接: [https://arxiv.org/pdf/2603.20884](https://arxiv.org/pdf/2603.20884)**

> **作者:** Jiajun Hou; Hexuan Deng; Wenxiang Jiao; Xuebo Liu; Xiaopeng Ke; Min Zhang
>
> **摘要:** The exponential growth of academic publications has led to a surge in papers of varying quality, increasing the cost of paper screening. Current approaches either use novelty assessment within general AI Reviewers or repurpose DeepResearch, which lacks domain-specific mechanisms and thus delivers lower-quality results. To bridge this gap, we introduce NoveltyAgent, a multi-agent system designed to generate comprehensive and faithful novelty reports, enabling thorough evaluation of a paper's originality. It decomposes manuscripts into discrete novelty points for fine-grained retrieval and comparison, and builds a comprehensive related-paper database while cross-referencing claims to ensure faithfulness. Furthermore, to address the challenge of evaluating such open-ended generation tasks, we propose a checklist-based evaluation framework, providing an unbiased paradigm for building reliable evaluations. Extensive experiments show that NoveltyAgent achieves state-of-the-art performance, outperforming GPT-5 DeepResearch by 10.15%. We hope this system will provide reliable, high-quality novelty analysis and help researchers quickly identify novel papers. Code and demo are available at this https URL.
>
---
#### [new 095] DATASHI: A Parallel English-Tashlhiyt Corpus for Orthography Normalization and Low-Resource Language Processing
- **分类: cs.CL**

- **简介: 该论文提出DATASHI语料库，解决阿迈尔语正字法标准化与低资源语言处理问题。通过构建平行语料，支持文本处理任务并评估大模型表现。**

- **链接: [https://arxiv.org/pdf/2603.21571](https://arxiv.org/pdf/2603.21571)**

> **作者:** Nasser-Eddine Monir; Zakaria Baou
>
> **备注:** This paper has been accepted for presentation at LREC 2026
>
> **摘要:** DATASHI is a new parallel English-Tashlhiyt corpus that fills a critical gap in computational resources for Amazigh languages. It contains 5,000 sentence pairs, including a 1,500-sentence subset with expert-standardized and non-standard user-generated versions, enabling systematic study of orthographic diversity and normalization. This dual design supports text-based NLP tasks - such as tokenization, translation, and normalization - and also serves as a foundation for read-speech data collection and multimodal alignment. Comprehensive evaluations with state-of-the-art Large Language Models (GPT-5, Claude-Sonnet-4.5, Gemini-2.5-Pro, Mistral, Qwen3-Max) show clear improvements from zero-shot to few-shot prompting, with Gemini-2.5-Pro achieving the lowest word and character-level error rates and exhibiting robust cross-lingual generalization. A fine-grained analysis of edit operations - deletions, substitutions, and insertions - across phonological classes (geminates, emphatics, uvulars, and pharyngeals) further highlights model-specific sensitivities to marked Tashlhiyt features and provides new diagnostic insights for low-resource Amazigh orthography normalization.
>
---
#### [new 096] Many Dialects, Many Languages, One Cultural Lens: Evaluating Multilingual VLMs for Bengali Culture Understanding Across Historically Linked Languages and Regional Dialects
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态语言理解任务，旨在解决 Bengali 文化在多语言和方言中的评估不足问题。构建了 BanglaVerse 基准，涵盖九个领域，测试 VLM 在文化理解上的表现。**

- **链接: [https://arxiv.org/pdf/2603.21165](https://arxiv.org/pdf/2603.21165)**

> **作者:** Nurul Labib Sayeedi; Md. Faiyaz Abdullah Sayeedi; Shubhashis Roy Dipta; Rubaya Tabassum; Ariful Ekraj Hridoy; Mehraj Mahmood; Mahbub E Sobhani; Md. Tarek Hasan; Swakkhar Shatabda
>
> **备注:** this https URL
>
> **摘要:** Bangla culture is richly expressed through region, dialect, history, food, politics, media, and everyday visual life, yet it remains underrepresented in multimodal evaluation. To address this gap, we introduce BanglaVerse, a culturally grounded benchmark for evaluating multilingual vision-language models (VLMs) on Bengali culture across historically linked languages and regional dialects. Built from 1,152 manually curated images across nine domains, the benchmark supports visual question answering and captioning, and is expanded into four languages and five Bangla dialects, yielding ~32.3K artifacts. Our experiments show that evaluating only standard Bangla overestimates true model capability: performance drops under dialectal variation, especially for caption generation, while historically linked languages such as Hindi and Urdu retain some cultural meaning but remain weaker for structured reasoning. Across domains, the main bottleneck is missing cultural knowledge rather than visual grounding alone, with knowledge-intensive categories. These findings position BanglaVerse as a more realistic test bed for measuring culturally grounded multimodal understanding under linguistic variation.
>
---
#### [new 097] Reading Between the Lines: How Electronic Nonverbal Cues shape Emotion Decoding
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于情感解码任务，解决文本中非语言线索如何影响情绪理解的问题。通过理论构建、实验验证和用户研究，提出电子非语言线索（eNVCs）的分类与检测工具。**

- **链接: [https://arxiv.org/pdf/2603.21038](https://arxiv.org/pdf/2603.21038)**

> **作者:** Taara Kumar; Kokil Jaidka
>
> **备注:** Accepted at AAAI ICWSM 2026
>
> **摘要:** As text-based computer-mediated communication (CMC) increasingly structures everyday interaction, a central question re-emerges with new urgency: How do users reconstruct nonverbal expression in environments where embodied cues are absent? This paper provides a systematic, theory-driven account of electronic nonverbal cues (eNVCs) - textual analogues of kinesics, vocalics, and paralinguistics - in public microblog communication. Across three complementary studies, we advance conceptual, empirical, and methodological contributions. Study 1 develops a unified taxonomy of eNVCs grounded in foundational nonverbal communication theory and introduces a scalable Python toolkit for their automated detection. Study 2, a within-subject survey experiment, offers controlled causal evidence that eNVCs substantially improve emotional decoding accuracy and lower perceived ambiguity, while also identifying boundary conditions, such as sarcasm, under which these benefits weaken or disappear. Study 3, through focus group discussions, reveals the interpretive strategies users employ when reasoning about digital prosody, including drawing meaning from the absence of expected cues and defaulting toward negative interpretations in ambiguous contexts. Together, these studies establish eNVCs as a coherent and measurable class of digital behaviors, refine theoretical accounts of cue richness and interpretive effort, and provide practical tools for affective computing, user modeling, and emotion-aware interface design. The eNVC detection toolkit is available as a Python and R package at this https URL.
>
---
#### [new 098] Triangulating Temporal Dynamics in Multilingual Swiss Online News
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于媒体分析任务，旨在研究多语言社会中的新闻动态。通过三角验证方法，分析瑞士三种语言区的新闻趋势，揭示语言与文化对报道的影响。**

- **链接: [https://arxiv.org/pdf/2603.21519](https://arxiv.org/pdf/2603.21519)**

> **作者:** Bros Victor; Dufraisse Evan; Popescu Adrian; Gatica-Perez Daniel
>
> **备注:** ICWSM 2026
>
> **摘要:** Analyzing news coverage in multilingual societies can offer valuable insights into the dynamics of public discourse and the development of collective narratives, yet comprehensive studies that account for linguistic and cultural diversity within national media ecosystems remain limited, particularly in complex contexts such as Switzerland. This paper studies temporal trends in Swiss digital media across the country's three main linguistic regions, French, German, and Italian, using a triangulated methodology that combines quantitative analyses with qualitative insights. We collected and processed over 1.7 million news articles, applying lexical metrics, named entity recognition and Wikidata-based linking, targeted sentiment analysis, and consensus-based change-point detection. To enable principled cross-language comparisons and to connect to theories of domestication and cultural proximity, we derive domestication profiles together with a proximity salience ratio. Our analysis spans thematic, recurrent, and singular events. By integrating quantitative data with qualitative interpretation, we provide new insights into the dynamics of Swiss digital media and demonstrate the usefulness of triangulation in media studies. The findings reveal distinct temporal patterns and highlight how linguistic and cultural contexts influence reporting. Our approach offers a framework applicable to other multilingual or culturally diverse media environments, contributing to a deeper understanding of how news is shaped by linguistic and cultural factors.
>
---
#### [new 099] Beyond Memorization: Distinguishing between Reductive and Epistemic Reasoning in LLMs using Classic Logic Puzzles
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，探讨大模型在经典逻辑谜题中的推理能力。解决的问题是区分模型是通过记忆还是推理解题。工作包括提出还原阶梯，分析模型在不同难度下的表现。**

- **链接: [https://arxiv.org/pdf/2603.21350](https://arxiv.org/pdf/2603.21350)**

> **作者:** Adi Gabay; Gabriel Stanovsky; Liat Peterfreund
>
> **摘要:** Epistemic reasoning requires agents to infer the state of the world from partial observations and information about other agents' knowledge. Prior work evaluating LLMs on canonical epistemic puzzles interpreted their behavior through a dichotomy between epistemic reasoning and brittle memorization. We argue that this framing is incomplete: in recent models, memorization is better understood as a special case of reduction, where a new instance is mapped onto a known problem. Instead, we introduce a reduction ladder, a sequence of modifications that progressively move instances away from a canonical epistemic puzzle, making reduction increasingly difficult while preserving the underlying logic. We find that while some large models succeed via reduction, other models fail early, and all models struggle once epistemic reasoning is required.
>
---
#### [new 100] DiscoUQ: Structured Disagreement Analysis for Uncertainty Quantification in LLM Agent Ensembles
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出DiscoUQ框架，用于量化大语言模型代理集成的不确定性。针对现有方法依赖简单投票统计的问题，通过分析代理间的分歧结构，提升置信度估计的准确性与校准性。**

- **链接: [https://arxiv.org/pdf/2603.20975](https://arxiv.org/pdf/2603.20975)**

> **作者:** Bo Jiang
>
> **摘要:** Multi-agent LLM systems, where multiple prompted instances of a language model independently answer questions, are increasingly used for complex reasoning tasks. However, existing methods for quantifying the uncertainty of their collective outputs rely on shallow voting statistics that discard the rich semantic information in agents' reasoning. We introduce DiscoUQ, a framework that extracts and leverages the structure of inter-agent disagreement -- both linguistic properties (evidence overlap, argument strength, divergence depth) and embedding geometry (cluster distances, dispersion, cohesion) -- to produce well-calibrated confidence estimates. We propose three methods of increasing complexity: DiscoUQ-LLM (logistic regression on LLM-extracted structure features), DiscoUQ-Embed (logistic regression on embedding geometry), and DiscoUQ-Learn (a neural network combining all features). Evaluated on four diverse benchmarks (StrategyQA, MMLU, TruthfulQA, ARC-Challenge) with a 5-agent system using Qwen3.5-27B, DiscoUQ-LLM achieves an average AUROC of 0.802, outperforming the best baseline (LLM Aggregator, 0.791) while being substantially better calibrated (ECE 0.036 vs. 0.098). The learned features generalize across benchmarks with near-zero performance degradation and provide the largest improvements where they are most needed: in the ambiguous "weak disagreement" tier where simple vote counting fails.
>
---
#### [new 101] Benchmarking Bengali Dialectal Bias: A Multi-Stage Framework Integrating RAG-Based Translation and Human-Augmented RLAIF
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于自然语言处理任务，旨在解决低资源语言方言在大模型中的性能偏差问题。通过构建多阶段框架，评估九种孟加拉语方言的模型表现，并提出新的评估方法和基准数据集。**

- **链接: [https://arxiv.org/pdf/2603.21359](https://arxiv.org/pdf/2603.21359)**

> **作者:** K. M. Jubair Sami; Dipto Sumit; Ariyan Hossain; Farig Sadeque
>
> **备注:** 12 pages, 1 figure, 5 tables
>
> **摘要:** Large language models (LLMs) frequently exhibit performance biases against regional dialects of low-resource languages. However, frameworks to quantify these disparities remain scarce. We propose a two-phase framework to evaluate dialectal bias in LLM question-answering across nine Bengali dialects. First, we translate and gold-label standard Bengali questions into dialectal variants adopting a retrieval-augmented generation (RAG) pipeline to prepare 4,000 question sets. Since traditional translation quality evaluation metrics fail on unstandardized dialects, we evaluate fidelity using an LLM-as-a-judge, which human correlation confirms outperforms legacy metrics. Second, we benchmark 19 LLMs across these gold-labeled sets, running 68,395 RLAIF evaluations validated through multi-judge agreement and human fallback. Our findings reveal severe performance drops linked to linguistic divergence. For instance, responses to the highly divergent Chittagong dialect score 5.44/10, compared to 7.68/10 for Tangail. Furthermore, increased model scale does not consistently mitigate this bias. We contribute a validated translation quality evaluation method, a rigorous benchmark dataset, and a Critical Bias Sensitivity (CBS) metric for safety-critical applications.
>
---
#### [new 102] Linguistic Signatures for Enhanced Emotion Detection
- **分类: cs.CL**

- **简介: 该论文属于情感检测任务，旨在探索语言特征对情绪识别的贡献。通过提取情感特定的语言签名并融入Transformer模型，提升了情绪分类性能。**

- **链接: [https://arxiv.org/pdf/2603.20222](https://arxiv.org/pdf/2603.20222)**

> **作者:** Florian Lecourt; Madalina Croitoru; Konstantin Todorov
>
> **摘要:** Emotion detection is a central problem in NLP, with recent progress driven by transformer-based models trained on established datasets. However, little is known about the linguistic regularities that characterize how emotions are expressed across different corpora and labels. This study examines whether linguistic features can serve as reliable interpretable signals for emotion recognition in text. We extract emotion-specific linguistic signatures from 13 English datasets and evaluate how incorporating these features into transformer models impacts performance. Our RoBERTa-based models enriched with high level linguistic features achieve consistent performance gains of up to +2.4 macro F1 on the GoEmotions benchmark, showing that explicit lexical cues can complement neural representations and improve robustness in predicting emotion categories.
>
---
#### [new 103] CRoCoDiL: Continuous and Robust Conditioned Diffusion for Language
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本生成任务，解决MDMs在语义连贯性与速度上的不足。提出CRoCoDiL方法，通过连续语义空间提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2603.20210](https://arxiv.org/pdf/2603.20210)**

> **作者:** Roy Uziel; Omer Belhasin; Itay Levi; Akhiad Bercovich; Ran El-Yaniv; Ran Zilberstein; Michael Elad
>
> **摘要:** Masked Diffusion Models (MDMs) provide an efficient non-causal alternative to autoregressive generation but often struggle with token dependencies and semantic incoherence due to their reliance on discrete marginal distributions. We address these limitations by shifting the diffusion process into a continuous sentence-level semantic space. We propose CRoCoDiL (Continuous and Robust Conditioned Diffusion for Language), a unified fine-tuning approach that jointly trains an encoder-demasker architecture, grounding the MDM demasking in continuous latent representations. This leads to the formation of a novel autoencoder in which decoding is obtained by an MDM algorithm. Relying on the same framework, we introduce two unconditional text synthesis algorithms: Continuous-Then-Discrete (ConThenDisc), a hybrid-diffusion approach that first generates latent representations in continuous space and then decodes these to tokens via an MDM, and Continuous-Within-Discrete (ConWithinDisc), a multi-diffusion strategy that refines latent representations throughout the discrete sampling process. Experiments using LLaDA show that our methods achieve superior generation quality and more than 10x faster sampling speeds in an unconditional setting.
>
---
#### [new 104] On the Challenges and Opportunities of Learned Sparse Retrieval for Code
- **分类: cs.IR; cs.CL**

- **简介: 该论文研究代码检索任务，针对稀疏检索在代码中的应用挑战，提出SPLADE-Code模型，提升检索效果与效率。**

- **链接: [https://arxiv.org/pdf/2603.22008](https://arxiv.org/pdf/2603.22008)**

> **作者:** Simon Lupart; Maxime Louis; Thibault Formal; Hervé Déjean; Stéphane Clinchant
>
> **备注:** 15 pages, 5 figures, 12 tables
>
> **摘要:** Retrieval over large codebases is a key component of modern LLM-based software engineering systems. Existing approaches predominantly rely on dense embedding models, while learned sparse retrieval (LSR) remains largely unexplored for code. However, applying sparse retrieval to code is challenging due to subword fragmentation, semantic gaps between natural-language queries and code, diversity of programming languages and sub-tasks, and the length of code documents, which can harm sparsity and latency. We introduce SPLADE-Code, the first large-scale family of learned sparse retrieval models specialized for code retrieval (600M-8B parameters). Despite a lightweight one-stage training pipeline, SPLADE-Code achieves state-of-the-art performance among retrievers under 1B parameters (75.4 on MTEB Code) and competitive results at larger scales (79.0 with 8B). We show that learned expansion tokens are critical to bridge lexical and semantic matching, and provide a latency analysis showing that LSR enables sub-millisecond retrieval on a 1M-passage collection with little effectiveness loss.
>
---
#### [new 105] Improving Coherence and Persistence in Agentic AI for System Optimization
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于系统优化任务，旨在解决Agentic AI在长期探索中的连贯性与持久性问题。提出Engram架构，通过分阶段代理和知识沉淀提升性能。**

- **链接: [https://arxiv.org/pdf/2603.21321](https://arxiv.org/pdf/2603.21321)**

> **作者:** Pantea Karimi; Kimia Noorbakhsh; Mohammad Alizadeh; Hari Balakrishnan
>
> **摘要:** Designing high-performance system heuristics is a creative, iterative process requiring experts to form hypotheses and execute multi-step conceptual shifts. While Large Language Models (LLMs) show promise in automating this loop, they struggle with complex system problems due to two critical failure modes: evolutionary neighborhood bias and the coherence ceiling. Evolutionary methods often remain trapped in local optima by relying on scalar benchmark scores, failing when coordinated multi-step changes are required. Conversely, existing agentic frameworks suffer from context degradation over long horizons or fail to accumulate knowledge across independent runs. We present Engram, an agentic researcher architecture that addresses these limitations by decoupling long-horizon exploration from the constraints of a single context window. Engram organizes exploration into a sequence of agents that iteratively design, test, and analyze mechanisms. At the conclusion of each run, an agent stores code snapshots, logs, and results in a persistent Archive and distills high-level modeling insights into a compact, persistent Research Digest. Subsequent agents then begin with a fresh context window, reading the Research Digest to build on prior discoveries. We find that Engram exhibits superior performance across diverse domains including multi-cloud multicast, LLM inference request routing, and optimizing KV cache reuse in databases with natural language queries.
>
---
#### [new 106] kRAIG: A Natural Language-Driven Agent for Automated DataOps Pipeline Generation
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出kRAIG，解决数据工程流程自动化问题。通过自然语言生成生产级Kubeflow管道，提升数据提取、加载和转换的准确性和可靠性。**

- **链接: [https://arxiv.org/pdf/2603.20311](https://arxiv.org/pdf/2603.20311)**

> **作者:** Rohan Siva; Kai Cheung; Lichi Li; Ganesh Sundaram
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Modern machine learning systems rely on complex data engineering workflows to extract, transform, and load (ELT) data into production pipelines. However, constructing these pipelines remains time-consuming and requires substantial expertise in data infrastructure and orchestration frameworks. Recent advances in large language model (LLM) agents offer a potential path toward automating these workflows, but existing approaches struggle with under-specified user intent, unreliable tool generation, and limited guarantees of executable outputs. We introduce kRAIG, an AI agent that translates natural language specifications into production-ready Kubeflow Pipelines (KFP). To resolve ambiguity in user intent, we propose ReQuesAct (Reason, Question, Act), an interaction framework that explicitly clarifies intent prior to pipeline synthesis. The system orchestrates end-to-end data movement from diverse sources and generates task-specific transformation components through a retrieval-augmented tool synthesis process. To ensure data quality and safety, kRAIG incorporates LLM-based validation stages that verify pipeline integrity prior to execution. Our framework achieves a 3x improvement in extraction and loading success and a 25 percent increase in transformation accuracy compared to state-of-the-art agentic baselines. These improvements demonstrate that structured agent workflows with explicit intent clarification and validation significantly enhance the reliability and executability of automated data engineering pipelines.
>
---
#### [new 107] AgentHER: Hindsight Experience Replay for LLM Agent Trajectory Relabeling
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出AgentHER，用于提升LLM代理在导航任务中的表现。针对失败轨迹被浪费的问题，通过 hindsight experience replay 方法将其转化为训练数据，提高数据效率和性能。**

- **链接: [https://arxiv.org/pdf/2603.21357](https://arxiv.org/pdf/2603.21357)**

> **作者:** Liang Ding
>
> **摘要:** LLM agents fail on the majority of real-world tasks -- GPT-4o succeeds on fewer than 15% of WebArena navigation tasks and below 55% pass@1 on ToolBench (Zhou et al., 2024; Qin et al., 2024) -- yet every failed trajectory is routinely discarded, wasting the dominant source of collected experience. We introduce AgentHER, a framework that recovers this lost training signal by adapting the Hindsight Experience Replay (HER; Andrychowicz et al., 2017) principle to natural-language agent trajectories for offline data augmentation. The key insight is simple: a trajectory that fails goal A is often a correct demonstration for some achievable alternative goal B. AgentHER realises this idea through a four-stage pipeline -- failure classification, outcome extraction, LLM-guided prompt relabeling with confidence gating, and data packaging -- that converts discarded failures into high-quality SFT, DPO, and ShareGPT training data, with both zero-cost rule-based and LLM-judge implementations. On WebArena (Zhou et al., 2024) and ToolBench (Qin et al., 2024), AgentHER improves over success-only SFT by +7.1-11.7 pp across four model families (GPT-4o, Qwen2.5-72B/7B, LLaMA-3.1-8B), while achieving 2x data efficiency -- matching baseline performance with only 50% of successful demonstrations. Gains are consistent from 1.5B to 72B parameters (+5.8-9.2 pp) and compound under iterative redeployment (+2.1 pp over additional rounds). Human evaluation confirms 97.7% relabeling precision under multi-judge verification.
>
---
#### [new 108] Clinical Cognition Alignment for Gastrointestinal Diagnosis with Multimodal LLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于医学图像诊断任务，旨在解决MLLM在胃肠道内镜中推理与临床路径不匹配、视觉特征与诊断无因果关联的问题。提出CogAlign框架，提升诊断准确性。**

- **链接: [https://arxiv.org/pdf/2603.20698](https://arxiv.org/pdf/2603.20698)**

> **作者:** Huan Zheng; Yucheng Zhou; Tianyi Yan; Dubing Chen; Hongbo Lu; Wenlong Liao; Tao He; Pai Peng; Jianbing Shen
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated remarkable potential in medical image analysis. However, their application in gastrointestinal endoscopy is currently hindered by two critical limitations: the misalignment between general model reasoning and standardized clinical cognitive pathways, and the lack of causal association between visual features and diagnostic outcomes. In this paper, we propose a novel Clinical-Cognitive-Aligned (CogAlign) framework to address these challenges. First, we endow the model with rigorous clinical analytical capabilities by constructing the hierarchical clinical cognition dataset and employing Supervised Fine-Tuning (SFT). Unlike conventional approaches, this strategy internalizes the hierarchical diagnostic logic of experts, ranging from anatomical localization and morphological evaluation to microvascular analysis, directly into the model. Second, to eliminate visual bias, we provide a theoretical analysis demonstrating that standard supervised tuning inevitably converges to spurious background correlations. Guided by this insight, we propose a counterfactual-driven reinforcement learning strategy to enforce causal rectification. By generating counterfactual normal samples via lesion masking and optimizing through clinical-cognition-centric rewards, we constrain the model to strictly ground its diagnosis in causal lesion features. Extensive experiments demonstrate that our approach achieves State-of-the-Art (SoTA) performance across multiple benchmarks, significantly enhancing diagnostic accuracy in complex clinical scenarios. All source code and datasets will be made publicly available.
>
---
#### [new 109] Silicon Bureaucracy and AI Test-Oriented Education: Contamination Sensitivity and Score Confidence in LLM Benchmarks
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI模型评估任务，旨在解决基准测试中因数据污染导致的评分可信度问题。通过审计框架分析污染敏感性和评分信心，揭示基准分数可能不反映真实能力。**

- **链接: [https://arxiv.org/pdf/2603.21636](https://arxiv.org/pdf/2603.21636)**

> **作者:** Yiliang Song; Hongjun An; Jiangan Chen; Xuanchen Yan; Huan Song; Jiawei Shao; Xuelong Li
>
> **备注:** First update
>
> **摘要:** Public benchmarks increasingly govern how large language models (LLMs) are ranked, selected, and deployed. We frame this benchmark-centered regime as Silicon Bureaucracy and AI Test-Oriented Education, and argue that it rests on a fragile assumption: that benchmark scores directly reflect genuine generalization. In practice, however, such scores may conflate exam-oriented competence with principled capability, especially when contamination and semantic leakage are difficult to exclude from modern training pipelines. We therefore propose an audit framework for analyzing contamination sensitivity and score confidence in LLM benchmarks. Using a router-worker setup, we compare a clean-control condition with noisy conditions in which benchmark problems are systematically deleted, rewritten, and perturbed before being passed downstream. For a genuinely clean benchmark, noisy conditions should not consistently outperform the clean-control baseline. Yet across multiple models, we find widespread but heterogeneous above-baseline gains under noisy conditions, indicating that benchmark-related cues may be reassembled and can reactivate contamination-related memory. These results suggest that similar benchmark scores may carry substantially different levels of confidence. Rather than rejecting benchmarks altogether, we argue that benchmark-based evaluation should be supplemented with explicit audits of contamination sensitivity and score confidence.
>
---
#### [new 110] ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文属于视频预测任务，旨在解决长时序语义捕捉不足的问题。通过结合视觉语言模型与JEPA框架，提升预测的语义能力和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.22281](https://arxiv.org/pdf/2603.22281)**

> **作者:** Haichao Zhang; Yijiang Li; Shwai He; Tushar Nagarajan; Mingfei Chen; Jianglin Lu; Ang Li; Yun Fu
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Recent progress in latent world models (e.g., V-JEPA2) has shown promising capability in forecasting future world states from video observations. Nevertheless, dense prediction from a short observation window limits temporal context and can bias predictors toward local, low-level extrapolation, making it difficult to capture long-horizon semantics and reducing downstream utility. Vision--language models (VLMs), in contrast, provide strong semantic grounding and general knowledge by reasoning over uniformly sampled frames, but they are not ideal as standalone dense predictors due to compute-driven sparse sampling, a language-output bottleneck that compresses fine-grained interaction states into text-oriented representations, and a data-regime mismatch when adapting to small action-conditioned datasets. We propose a VLM-guided JEPA-style latent world modeling framework that combines dense-frame dynamics modeling with long-horizon semantic guidance via a dual-temporal pathway: a dense JEPA branch for fine-grained motion and interaction cues, and a uniformly sampled VLM \emph{thinker} branch with a larger temporal stride for knowledge-rich guidance. To transfer the VLM's progressive reasoning signals effectively, we introduce a hierarchical pyramid representation extraction module that aggregates multi-layer VLM representations into guidance features compatible with latent prediction. Experiments on hand-manipulation trajectory prediction show that our method outperforms both a strong VLM-only baseline and a JEPA-predictor baseline, and yields more robust long-horizon rollout behavior.
>
---
#### [new 111] Putnam 2025 Problems in Rocq using Opus 4.6 and Rocq-MCP
- **分类: cs.LG; cs.CL; cs.LO**

- **简介: 该论文属于自动定理证明任务，旨在用Claude Opus 4.6和MCP工具自主证明数学竞赛问题。工作包括开发策略、部署子代理并完成10道题的证明。**

- **链接: [https://arxiv.org/pdf/2603.20405](https://arxiv.org/pdf/2603.20405)**

> **作者:** Guillaume Baudart; Marc Lelarge; Tristan Stérin; Jules Viennot
>
> **摘要:** We report on an experiment in which Claude Opus~4.6, equipped with a suite of Model Context Protocol (MCP) tools for the Rocq proof assistant, autonomously proved 10 of 12 problems from the 2025 Putnam Mathematical Competition. The MCP tools, designed with Claude by analyzing logs from a prior experiment on miniF2F-Rocq, encode a "compile-first, interactive-fallback" strategy. Running on an isolated VM with no internet access, the agent deployed 141 subagents over 17.7 hours of active compute (51.6h wall-clock), consuming approximately 1.9 billion tokens. All proofs are publicly available.
>
---
#### [new 112] Demystifying Reinforcement Learning for Long-Horizon Tool-Using Agents: A Comprehensive Recipe
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决长周期工具使用代理的训练问题。通过实验分析RL设计要素，提出有效训练方案，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.21972](https://arxiv.org/pdf/2603.21972)**

> **作者:** Xixi Wu; Qianguo Sun; Ruiyang Zhang; Chao Song; Junlong Wu; Yiyan Qi; Hong Cheng
>
> **备注:** Codes are available at this https URL
>
> **摘要:** Reinforcement Learning (RL) is essential for evolving Large Language Models (LLMs) into autonomous agents capable of long-horizon planning, yet a practical recipe for scaling RL in complex, multi-turn environments remains elusive. This paper presents a systematic empirical study using TravelPlanner, a challenging testbed requiring tool orchestration to satisfy multifaceted constraints. We decompose the agentic RL design space along 5 axes: reward shaping, model scaling, data composition, algorithm selection, and environmental stability. Our controlled experiments yield 7 key takeaways, e.g., (1) reward and algorithm choices are scale-dependent as smaller models benefit from staged rewards and enhanced exploration, whereas larger models converge efficiently with simpler dense rewards, (2) ~ 1K training samples with a balanced difficulty mixture mark a sweet spot for both in-domain and out-of-domain performance, and (3) environmental stability is critical to prevent policy degradation. Based on our distilled recipe, our RL-trained models achieve state-of-the-art performance on TravelPlanner, significantly outperforming leading LLMs.
>
---
#### [new 113] Generalized Discrete Diffusion from Snapshots
- **分类: stat.ML; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出GDDS，一种统一的离散扩散框架，解决大规模离散生成任务中的训练效率与生成质量问题，通过灵活的噪声过程和高效反向过程实现更优性能。**

- **链接: [https://arxiv.org/pdf/2603.21342](https://arxiv.org/pdf/2603.21342)**

> **作者:** Oussama Zekri; Théo Uscidda; Nicolas Boullé; Anna Korba
>
> **备注:** 37 pages, 6 figures, 13 tables
>
> **摘要:** We introduce Generalized Discrete Diffusion from Snapshots (GDDS), a unified framework for discrete diffusion modeling that supports arbitrary noising processes over large discrete state spaces. Our formulation encompasses all existing discrete diffusion approaches, while allowing significantly greater flexibility in the choice of corruption dynamics. The forward noising process relies on uniformization and enables fast arbitrary corruption. For the reverse process, we derive a simple evidence lower bound (ELBO) based on snapshot latents, instead of the entire noising path, that allows efficient training of standard generative modeling architectures with clear probabilistic interpretation. Our experiments on large-vocabulary discrete generation tasks suggest that the proposed framework outperforms existing discrete diffusion methods in terms of training efficiency and generation quality, and beats autoregressive models for the first time at this scale. We provide the code along with a blog post on the project page : \href{this https URL}{this https URL}.
>
---
#### [new 114] AgenticGEO: A Self-Evolving Agentic System for Generative Engine Optimization
- **分类: cs.AI; cs.CL; cs.LG; cs.NE**

- **简介: 该论文属于生成式引擎优化任务，旨在解决传统方法适应性差、依赖大量交互反馈的问题。提出AgenticGEO框架，通过进化策略提升内容质量与引擎适应性。**

- **链接: [https://arxiv.org/pdf/2603.20213](https://arxiv.org/pdf/2603.20213)**

> **作者:** Jiaqi Yuan; Jialu Wang; Zihan Wang; Qingyun Sun; Ruijie Wang; Jianxin Li
>
> **摘要:** Generative search engines represent a transition from traditional ranking-based retrieval to Large Language Model (LLM)-based synthesis, transforming optimization goals from ranking prominence towards content inclusion. Generative Engine Optimization (GEO), specifically, aims to maximize visibility and attribution in black-box summarized outputs by strategically manipulating source content. However, existing methods rely on static heuristics, single-prompt optimization, or engine preference rule distillation that is prone to overfitting. They cannot flexibly adapt to diverse content or the changing behaviors of generative engines. Moreover, effectively optimizing these strategies requires an impractical amount of interaction feedback from the engines. To address these challenges, we propose AgenticGEO, a self-evolving agentic framework formulating optimization as a content-conditioned control problem, which enhances intrinsic content quality to robustly adapt to the unpredictable behaviors of black-box engines. Unlike fixed-strategy methods, AgenticGEO employs a MAP-Elites archive to evolve diverse, compositional strategies. To mitigate interaction costs, we introduce a Co-Evolving Critic, a lightweight surrogate that approximates engine feedback for content-specific strategy selection and refinement, efficiently guiding both evolutionary search and inference-time planning. Through extensive in-domain and cross-domain experiments on two representative engines, AgenticGEO achieves state-of-the-art performance and demonstrates robust transferability, outperforming 14 baselines across 3 datasets. Our code and model are available at: this https URL.
>
---
#### [new 115] ROM: Real-time Overthinking Mitigation via Streaming Detection and Intervention
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决大模型过思考问题。通过实时检测与干预，提出ROM方法提升响应效率和准确性。**

- **链接: [https://arxiv.org/pdf/2603.22016](https://arxiv.org/pdf/2603.22016)**

> **作者:** Xinyan Wang; Xiaogeng Liu; Chaowei Xiao
>
> **备注:** Code is available at this https URL
>
> **摘要:** Large Reasoning Models (LRMs) achieve strong accuracy on challenging tasks by generating long Chain-of-Thought traces, but suffer from overthinking. Even after reaching the correct answer, they continue generating redundant reasoning steps. This behavior increases latency and compute cost and can also lead to answer drift. Existing mitigation methods either require training-heavy backbone modification or rely on hand-crafted heuristics that do not truly capture overthinking patterns. We propose ROM, the first method that formulates overthinking mitigation as a streaming prediction-and-control problem. ROM attaches a lightweight detection head to the late-layer hidden states of a frozen large language model backbone. It monitors tokens in real time and triggers an early transition to the final answer once overthinking is detected. We also introduce token-level supervision based on solution correctness boundaries and a data augmentation strategy that reduces distilled-data bias. Across seven benchmarks, ROM achieves the highest accuracy (93.51%), the shortest responses (1,159 tokens), and the best response efficiency. Compared with the vanilla baseline, it reduces response length by 47.2% and improves efficiency by 121%. These results show that streaming detection is a promising approach to real-time overthinking mitigation.
>
---
#### [new 116] The Reasoning Error About Reasoning: Why Different Types of Reasoning Require Different Representational Structures
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于认知科学领域，探讨不同推理类型对表征系统结构的要求。提出四个结构属性框架，解释推理失败原因，分析推理类型间的结构性界限。**

- **链接: [https://arxiv.org/pdf/2603.21736](https://arxiv.org/pdf/2603.21736)**

> **作者:** Yiling Wu
>
> **摘要:** Different types of reasoning impose different structural demands on representational systems, yet no systematic account of these demands exists across psychology, AI, and philosophy of mind. I propose a framework identifying four structural properties of representational systems: operability, consistency, structural preservation, and compositionality. These properties are demanded to different degrees by different forms of reasoning, from induction through analogy and causal inference to deduction and formal logic. Each property excludes a distinct class of reasoning failure. The analysis reveals a principal structural boundary: reasoning types below it can operate on associative, probabilistic representations, while those above it require all four properties to be fully satisfied. Scaling statistical learning without structural reorganization is insufficient to cross this boundary, because the structural guarantees required by deductive reasoning cannot be approximated through probabilistic means. Converging evidence from AI evaluation, developmental psychology, and cognitive neuroscience supports the framework at different levels of directness. Three testable predictions are derived, including compounding degradation, selective vulnerability to targeted structural disruption, and irreducibility under scaling. The framework is a necessary-condition account, agnostic about representational format, that aims to reorganize existing debates rather than close them.
>
---
#### [new 117] Dyadic: A Scalable Platform for Human-Human and Human-AI Conversation Research
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文介绍Dyadic平台，用于人与人及人与AI的对话研究。解决传统工具模块化不足的问题，提供多模态支持、实时监控和问卷部署等功能。**

- **链接: [https://arxiv.org/pdf/2603.22227](https://arxiv.org/pdf/2603.22227)**

> **作者:** David M. Markowitz
>
> **摘要:** Conversation is ubiquitous in social life, but the empirical study of this interactive process has been thwarted by tools that are insufficiently modular and unadaptive to researcher needs. To relieve many constraints in conversation research, the current tutorial presents an overview and introduction to a new tool, Dyadic (this https URL), a web-based platform for studying human-human and human-AI conversations using text-based or voice-based chats. Dyadic is distinct from other platforms by offering studies with multiple modalities, AI suggestions (e.g., in human-human studies, AI can suggest responses to a participant), live monitoring (e.g., researchers can evaluate, in real time, chats between communicators), and survey deployment (e.g., Likert-type scales, feeling thermometers, and open-ended text boxes can be sent to humans for in situ evaluations of the interaction), among other consequential features. No coding is required to operate Dyadic directly, and integrations with existing survey platforms are offered.
>
---
#### [new 118] Semantic Sections: An Atlas-Native Feature Ontology for Obstructed Representation Spaces
- **分类: cs.LG; cs.AI; cs.CL; cs.NE**

- **简介: 该论文提出"语义片段"概念，解决阻塞表示空间中的特征表示问题。通过构建局部一致的全局可扩展特征结构，提升特征语义一致性。**

- **链接: [https://arxiv.org/pdf/2603.20867](https://arxiv.org/pdf/2603.20867)**

> **作者:** Hossein Javidnia
>
> **备注:** 20 pages, 2 figures
>
> **摘要:** Recent interpretability work often treats a feature as a single global direction, dictionary atom, or latent coordinate shared across contexts. We argue that this ontology can fail in obstructed representation spaces, where locally coherent meanings need not assemble into one globally consistent feature. We introduce an atlas-native replacement object, the semantic section: a transport-compatible family of local feature representatives defined over a context atlas. We formalize semantic sections, prove that tree-supported propagation is always pathwise realizable, and show that cycle consistency is the key criterion for genuine globalization. This yields a distinction between tree-local, globalizable, and twisted sections, with twisted sections capturing locally coherent but holonomy-obstructed meanings. We then develop a discovery-and-certification pipeline based on seeded propagation, synchronization across overlaps, defect-based pruning, cycle-aware taxonomy, and deduplication. Across layer-16 atlases for Llama 3.2 3B Instruct, Qwen 2.5 3B Instruct, and Gemma 2 2B IT, we find nontrivial populations of semantic sections, including cycle-supported globalizable and twisted regimes after deduplication. Most importantly, semantic identity is not recovered by raw global-vector similarity. Even certified globalizable sections show low cross-chart signed cosine similarity, and raw similarity baselines recover only a small fraction of true within-section pairs, often collapsing at moderate thresholds. By contrast, section-based identity recovery is perfect on certified supports. These results support semantic sections as a better feature ontology in obstructed regimes.
>
---
#### [new 119] Profiling learners' affective engagement: Emotion AI, intercultural pragmatics, and language learning
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 论文探讨情感AI在语言学习中的应用，分析其对学习者情绪参与的影响及挑战。属于教育技术任务，旨在解决如何有效利用AI提升语言交际能力的问题。**

- **链接: [https://arxiv.org/pdf/2603.20479](https://arxiv.org/pdf/2603.20479)**

> **作者:** Robert Godwin-Jones
>
> **摘要:** Learning another language can be a highly emotional process, typically characterized by numerous frustrations and triumphs, big and small. For most learners, language learning does not follow a linear, predictable path, its zigzag course shaped by motivational (or demotivating) variables such as personal characteristics, teacher/peer relationships, learning materials, and dreams of a future L2 (second language) self. While some aspects of language learning (reading, grammar) are relatively mechanical, others can be stressful and unpredictable, especially conversing in the target language. That experience necessitates not only knowledge of structure and lexis, but also the ability to use the language in ways that are appropriate to the social and cultural context. A new opportunity to practice conversational abilities has arrived through the availability of AI chatbots, with both advantages (responsive, non-judgmental) and drawbacks (emotionally void, culturally biased). This column explores aspects of emotion as they arise in technology use and in particular how automatic emotion recognition and simulated human responsiveness in AI systems interface with language learning and the development of pragmatic and interactional competence. Emotion AI, the algorithmically driven interpretation of users' affective signals, has been seen as enabling greater personalized learning, adapting to perceived learner cognitive and emotional states. Others warn of emotional manipulation and inappropriate and ineffective user profiling
>
---
#### [new 120] The Library Theorem: How External Organization Governs Agentic Reasoning Capacity
- **分类: cs.AI; cs.CL; cs.DS; cs.LG**

- **简介: 论文探讨了外部组织如何影响智能体的推理能力，提出通过索引检索可显著降低检索成本。任务是优化智能体的推理效率，解决传统方法检索效率低的问题，通过实验验证索引机制的优势。**

- **链接: [https://arxiv.org/pdf/2603.21272](https://arxiv.org/pdf/2603.21272)**

> **作者:** Zachary F. Mainen
>
> **备注:** 19 pages, 6 figures
>
> **摘要:** Externalized reasoning is already exploited by transformer-based agents through chain-of-thought, but structured retrieval -- indexing over one's own reasoning state -- remains underexplored. We formalize the transformer context window as an I/O page and prove that tool-augmented agents with indexed external memory achieve exponentially lower retrieval cost than agents restricted to sequential scanning: $O(\log_b N)$ versus $\Omega(N)$ page reads per query, and $O(T \log_b T)$ versus $\Theta(T^2)$ cumulative cost over $T$ reasoning steps -- a gap that widens as deliberation deepens. We test these predictions on a controlled lookup benchmark across three content types -- random hashes, ordered integers, and encyclopedia entries -- varying store size from 50 to 5,000 items, and replicate key conditions across two model generations (GPT-4o-mini and GPT-5.4). On abstract content, the indexed agent achieves median 1 page read regardless of store size, confirming the $O(1)$ prediction. Sorted pages without an index fail to close the gap: the weaker model cannot sustain binary search at scale, and the stronger model achieves near-optimal $\log_2 N$ search but still loses to the index by $5\times$. On familiar content (encyclopedia entries), a competing failure mode emerges: the model recognizes the domain, bypasses the retrieval protocol, and generates answers from parametric memory, producing catastrophic token expenditure even when the index is sound. This parametric memory competition dissociates the two cognitive operations that indexing combines: understanding content (where language models excel) and following navigational protocols (where they fail when understanding tempts them to shortcut). The result argues for a separation of concerns: use language models for index construction, where semantic understanding helps, and deterministic algorithms for index traversal, where it hurts.
>
---
#### [new 121] AdaRubric: Task-Adaptive Rubrics for LLM Agent Evaluation
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出ADARUBRIC，用于LLM代理评估，解决固定评分标准无法适应不同任务的问题。通过生成任务特定评分标准，提升评估准确性与代理性能。**

- **链接: [https://arxiv.org/pdf/2603.21362](https://arxiv.org/pdf/2603.21362)**

> **作者:** Liang Ding
>
> **摘要:** LLM-as-Judge evaluation fails agent tasks because a fixed rubric cannot capture what matters for this task: code debugging demands Correctness and Error Handling; web navigation demands Goal Alignment and Action Efficiency. We present ADARUBRIC, which closes this gap by generating task-specific evaluation rubrics on the fly from task descriptions, scoring trajectories step-by-step with confidence-weighted per-dimension feedback, and filtering preference pairs with the novel DimensionAwareFilter - a provably necessary condition for preventing high-scoring dimensions from masking dimension-level failures. On WebArena and ToolBench, ADARUBRIC achieves Pearson r=0.79 human correlation (+0.16 over the best static baseline) with deployment-grade reliability (Krippendorff's $\alpha$=0.83). DPO agents trained on ADARUBRIC preference pairs gain +6.8 to +8.5 pp task success over Prometheus across three benchmarks; gains transfer to SWE-bench code repair (+4.9 pp) and accelerate PPO convergence by +6.6 pp at 5K steps - both without any rubric engineering. Code: this https URL.
>
---
#### [new 122] Knowledge Boundary Discovery for Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出KBD框架，通过强化学习探索大语言模型的知识边界，解决模型知识范围难以量化的问题。通过生成可回答与不可回答问题，自动识别模型边界。**

- **链接: [https://arxiv.org/pdf/2603.21022](https://arxiv.org/pdf/2603.21022)**

> **作者:** Ziquan Wang; Zhongqi Lu
>
> **备注:** 9 pages,4 figures
>
> **摘要:** We propose Knowledge Boundary Discovery (KBD), a reinforcement learning based framework to explore the knowledge boundaries of the Large Language Models (LLMs). We define the knowledge boundary by automatically generating two types of questions: (i) those the LLM can confidently answer (within-knowledge boundary) and (ii) those it cannot (beyond-knowledge boundary). Iteratively exploring and exploiting the LLM's responses to find its knowledge boundaries is challenging because of the hallucination phenomenon. To find the knowledge boundaries of an LLM, the agent interacts with the LLM under the modeling of exploring a partially observable environment. The agent generates a progressive question as the action, adopts an entropy reduction as the reward, receives the LLM's response as the observation and updates its belief states. We demonstrate that the KBD detects knowledge boundaries of LLMs by automatically finding a set of non-trivial answerable and unanswerable questions. We validate the KBD by comparing its generated knowledge boundaries with manually crafted LLM benchmark datasets. Experiments show that our KBD-generated question set is comparable to the human-generated datasets. Our approach paves a new way to evaluate LLMs.
>
---
#### [new 123] How AI Systems Think About Education: Analyzing Latent Preference Patterns in Large Language Models
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于AI教育对齐研究，旨在评估大语言模型的教育价值观。通过系统测量GPT-5.1的偏好模式，发现其与人本主义教育原则高度一致，但在争议领域表现出立场。**

- **链接: [https://arxiv.org/pdf/2603.21006](https://arxiv.org/pdf/2603.21006)**

> **作者:** Daniel Autenrieth
>
> **备注:** 15 pages, 2 figures, 8 tables. Code and data available at this https URL. arXiv admin note: text overlap with arXiv:2502.08640 by other authors
>
> **摘要:** This paper presents the first systematic measurement of educational alignment in Large Language Models. Using a Delphi-validated instrument comprising 48 items across eight educational-theoretical dimensions, the study reveals that GPT-5.1 exhibits highly coherent preference patterns (99.78% transitivity; 92.79% model accuracy) that largely align with humanistic educational principles where expert consensus exists. Crucially, divergences from expert opinion occur precisely in domains of normative disagreement among human experts themselves, particularly emotional dimensions and epistemic normativity. This raises a fundamental question for alignment research: When human values are contested, what should models be aligned to? The findings demonstrate that GPT-5.1 does not remain neutral in contested domains but adopts coherent positions, prioritizing emotional responsiveness and rejecting false balance. The methodology, combining Delphi consensus-building with Structured Preference Elicitation and Thurstonian Utility modeling, provides a replicable framework for domain-specific alignment evaluation beyond generic value benchmarks.
>
---
#### [new 124] BHDD: A Burmese Handwritten Digit Dataset
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出BHDD数据集，用于缅甸手写数字识别任务。收集了87,561张图像，分析其分布与特征，并测试了多种模型性能。**

- **链接: [https://arxiv.org/pdf/2603.21966](https://arxiv.org/pdf/2603.21966)**

> **作者:** Swan Htet Aung; Hein Htet; Htoo Say Wah Khaing; Thuya Myo Nyunt
>
> **备注:** 4 pages, 9 figures, 1 table. Dataset available at this https URL
>
> **摘要:** We introduce the Burmese Handwritten Digit Dataset (BHDD), a collection of 87,561 grayscale images of handwritten Burmese digits in ten classes. Each image is 28x28 pixels, following the MNIST format. The training set has 60,000 samples split evenly across classes; the test set has 27,561 samples with class frequencies as they arose during collection. Over 150 people of different ages and backgrounds contributed samples. We analyze the dataset's class distribution, pixel statistics, and morphological variation, and identify digit pairs that are easily confused due to the round shapes of the Myanmar script. Simple baselines (an MLP, a two-layer CNN, and an improved CNN with batch normalization and augmentation) reach 99.40%, 99.75%, and 99.83% test accuracy respectively. BHDD is available under CC BY-SA 4.0 at this https URL
>
---
#### [new 125] Understanding Contextual Recall in Transformers: How Finetuning Enables In-Context Reasoning over Pretraining Knowledge
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究Transformer模型在上下文学习中的事实回忆能力，探讨预训练与微调对实现这一能力的影响。任务属于自然语言处理中的上下文学习研究，解决如何通过微调提升模型的上下文推理能力。工作包括构建合成框架验证预训练不足，并通过微调触发回忆能力的形成。**

- **链接: [https://arxiv.org/pdf/2603.20969](https://arxiv.org/pdf/2603.20969)**

> **作者:** Bhavya Vasudeva; Puneesh Deora; Alberto Bietti; Vatsal Sharan; Christos Thrampoulidis
>
> **备注:** 28 pages, 26 figures
>
> **摘要:** Transformer-based language models excel at in-context learning (ICL), where they can adapt to new tasks based on contextual examples, without parameter updates. In a specific form of ICL, which we refer to as \textit{contextual recall}, models pretrained on open-ended text leverage pairwise examples to recall specific facts in novel prompt formats. We investigate whether contextual recall emerges from pretraining alone, what finetuning is required, and what mechanisms drive the necessary representations. For this, we introduce a controlled synthetic framework where pretraining sequences consist of subject-grammar-attribute tuples, with attribute types tied to grammar statistics. We demonstrate that while such pretraining successfully yields factual knowledge, it is insufficient for contextual recall: models fail to implicitly infer attribute types when the grammar statistics are removed in ICL prompts. However, we show that finetuning on tasks requiring implicit inference, distinct from the ICL evaluation, using a subset of subjects, triggers the emergence of contextual recall across all subjects. This transition is accompanied by the formation of low-dimensional latent encodings of the shared attribute type. For mechanistic insight, we derive a construction for an attention-only transformer that replicates the transition from factual to contextual recall, corroborated by empirical validation.
>
---
#### [new 126] Structural Sensitivity in Compressed Transformers: Error Propagation, Lyapunov Stability, and Formally Verified Bounds
- **分类: cs.LG; cs.AI; cs.CL; cs.LO**

- **简介: 该论文研究Transformer模型压缩中的结构敏感性问题，分析误差传播与稳定性，提出形式化验证的误差边界，提升模型压缩的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.20991](https://arxiv.org/pdf/2603.20991)**

> **作者:** Abhinaba Basu
>
> **摘要:** A single matrix out of 468 in GPT-2 Small can increase perplexity by 20,000x when compressed, revealing that transformer compression sensitivity spans five orders of magnitude. We map this sensitivity landscape across five architectures (117M-8B parameters), finding a consistent hierarchy: early-layer MLP up-projections are catastrophically sensitive while value projections compress nearly for free. This hierarchy is stable across compression levels, evaluation scales (2K-51K tokens), and datasets (WikiText-103, C4). Using Lyapunov stability theory, we show that residual connections contract compression errors by growing the hidden state faster than the error. Error contraction is necessary but not sufficient for compression tolerance: architecture-specific redundancy plays an equally important role, as demonstrated by the hybrid LFM2-2.6B degrading only 7x despite higher amplification than the fully-contracting GPT-2 Small (120x). Ten machine-checked Lean 4 theorems formalize per-matrix error bounds with no sorry markers; all bounds produce zero violations across 14,040+ configurations. We validate with downstream task evaluation (HellaSwag, ARC-Easy, Winogrande), activation-aware pruning on two architectures, and a Compression Fragility Index that rank-orders model robustness.
>
---
#### [new 127] RubricRAG: Towards Interpretable and Reliable LLM Evaluation via Domain Knowledge Retrieval for Rubric Generation
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的模型评估任务，旨在解决LLM评价缺乏可解释性的问题。通过生成可解释的评分标准（rubric），提升评估的透明度和有效性。**

- **链接: [https://arxiv.org/pdf/2603.20882](https://arxiv.org/pdf/2603.20882)**

> **作者:** Kaustubh D. Dhole; Eugene Agichtein
>
> **摘要:** Large language models (LLMs) are increasingly evaluated and sometimes trained using automated graders such as LLM-as-judges that output scalar scores or preferences. While convenient, these approaches are often opaque: a single score rarely explains why an answer is good or bad, which requirements were missed, or how a system should be improved. This lack of interpretability limits their usefulness for model development, dataset curation, and high-stakes deployment. Query-specific rubric-based evaluation offers a more transparent alternative by decomposing quality into explicit, checkable criteria. However, manually designing high-quality, query-specific rubrics is labor-intensive and cognitively demanding and not feasible for deployment. While previous approaches have focused on generating intermediate rubrics for automated downstream evaluation, it is unclear if these rubrics are both interpretable and effective for human users. In this work, we investigate whether LLMs can generate useful, instance-specific rubrics as compared to human-authored rubrics, while also improving effectiveness for identifying good responses. Through our systematic study on two rubric benchmarks, and on multiple few-shot and post-training strategies, we find that off-the-shelf LLMs produce rubrics that are poorly aligned with human-authored ones. We introduce a simple strategy, RubricRAG, which retrieves domain knowledge via rubrics at inference time from related queries. We demonstrate that RubricRAG can generate more interpretable rubrics both for similarity to human-authored rubrics, and for improved downstream evaluation effectiveness. Our results highlight both the challenges and a promising approach of scalable, interpretable evaluation through automated rubric generation.
>
---
#### [new 128] PRISM: Breaking the O(n) Memory Wall in Long-Context LLM Inference via O(1) Photonic Block Selection
- **分类: physics.optics; cs.AI; cs.AR; cs.CL; cs.LG**

- **简介: 该论文属于长上下文大语言模型推理任务，旨在解决KV缓存扫描带来的O(n)内存带宽瓶颈。通过光子块选择技术PRISM，实现O(1)复杂度，提升效率并降低能耗。**

- **链接: [https://arxiv.org/pdf/2603.21576](https://arxiv.org/pdf/2603.21576)**

> **作者:** Hyoseok Park; Yeonsang Park
>
> **备注:** 28 pages, 27 figures, 15 tables, including supplementary material. Code available at this https URL
>
> **摘要:** Long-context LLM inference is bottlenecked not by compute but by the O(n) memory bandwidth cost of scanning the KV cache at every decode step -- a wall that no amount of arithmetic scaling can break. Recent photonic accelerators have demonstrated impressive throughput for dense attention computation; however, these approaches inherit the same O(n) memory scaling as electronic attention when applied to long contexts. We observe that the real leverage point is the coarse block-selection step: a memory-bound similarity search that determines which KV blocks to fetch. We identify, for the first time, that this task is structurally matched to the photonic broadcast-and-weight paradigm -- the query fans out to all candidates via passive splitting, signatures are quasi-static (matching electro-optic MRR programming), and only rank order matters (relaxing precision to 4-6 bits). Crucially, the photonic advantage grows with context length: as N increases, the electronic scan cost rises linearly while the photonic evaluation remains O(1). We instantiate this insight in PRISM (Photonic Ranking via Inner-product Similarity with Microring weights), a thin-film lithium niobate (TFLN) similarity engine. Hardware-impaired needle-in-a-haystack evaluation on Qwen2.5-7B confirms 100% accuracy from 4K through 64K tokens at k=32, with 16x traffic reduction at 64K context. PRISM achieves a four-order-of-magnitude energy advantage over GPU baselines at practical context lengths (n >= 4K).
>
---
#### [new 129] CLT-Forge: A Scalable Library for Cross-Layer Transcoders and Attribution Graphs
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于机械可解释性任务，旨在解决特征归因图冗余问题。提出CLT-Forge库，实现跨层编解码器的高效训练与解释，提升模型可解释性。**

- **链接: [https://arxiv.org/pdf/2603.21014](https://arxiv.org/pdf/2603.21014)**

> **作者:** Florent Draye; Abir Harrasse; Vedant Palit; Tung-Yu Wu; Jiarui Liu; Punya Syon Pandey; Roderick Wu; Terry Jingchen Zhang; Zhijing Jin; Bernhard Schölkopf
>
> **备注:** 9 pages, 2 figures, code: this https URL
>
> **摘要:** Mechanistic interpretability seeks to understand how Large Language Models (LLMs) represent and process information. Recent approaches based on dictionary learning and transcoders enable representing model computation in terms of sparse, interpretable features and their interactions, giving rise to feature attribution graphs. However, these graphs are often large and redundant, limiting their interpretability in practice. Cross-Layer Transcoders (CLTs) address this issue by sharing features across layers while preserving layer-specific decoding, yielding more compact representations, but remain difficult to train and analyze at scale. We introduce an open-source library for end-to-end training and interpretability of CLTs. Our framework integrates scalable distributed training with model sharding and compressed activation caching, a unified automated interpretability pipeline for feature analysis and explanation, attribution graph computation using Circuit-Tracer, and a flexible visualization interface. This provides a practical and unified solution for scaling CLT-based mechanistic interpretability. Our code is available at: this https URL.
>
---
#### [new 130] Beyond Correlation: Refutation-Validated Aspect-Based Sentiment Analysis for Explainable Energy Market Returns
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于情感分析任务，旨在解决金融市场中情感信号与收益关系的验证问题。通过构建验证框架，测试情感信号与股票收益的稳健关联。**

- **链接: [https://arxiv.org/pdf/2603.21473](https://arxiv.org/pdf/2603.21473)**

> **作者:** Wihan van der Heever; Keane Ong; Ranjan Satapathy; Erik Cambria
>
> **备注:** 13 pages, 6 figures, submitted to Expert Systems with Applications
>
> **摘要:** This paper proposes a refutation-validated framework for aspect-based sentiment analysis in financial markets, addressing the limitations of correlational studies that cannot distinguish genuine associations from spurious ones. Using X data for the energy sector, we test whether aspect-level sentiment signals show robust, refutation-validated relationships with equity returns. Our pipeline combines net-ratio scoring with z-normalization, OLS with Newey West HAC errors, and refutation tests including placebo, random common cause, subset stability, and bootstrap. Across six energy tickers, only a few associations survive all checks, while renewables show aspect and horizon specific responses. While not establishing causality, the framework provides statistically robust, directionally interpretable signals, with limited sample size (six stocks, one quarter) constraining generalizability and framing this work as a methodological proof of concept.
>
---
#### [new 131] ALICE: A Multifaceted Evaluation Framework of Large Audio-Language Models' In-Context Learning Ability
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于音频语言模型研究，旨在评估大音频语言模型的上下文学习能力。针对现有模型在音频条件下任务理解不足的问题，提出ALICE框架进行系统测试，发现模型仅能提升格式合规性，难以提升核心任务性能。**

- **链接: [https://arxiv.org/pdf/2603.20433](https://arxiv.org/pdf/2603.20433)**

> **作者:** Yen-Ting Piao; Jay Chiehen Liao; Wei-Tang Chien; Toshiki Ogimoto; Shang-Tse Chen; Yun-Nung Chen; Chun-Yi Lee; Shao-Yuan Lo
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** While Large Audio-Language Models (LALMs) have been shown to exhibit degraded instruction-following capabilities, their ability to infer task patterns from in-context examples under audio conditioning remains unstudied. To address this gap, we present ALICE, a three-stage framework that progressively reduces textual guidance to systematically evaluate LALMs' in-context learning ability under audio conditioning. Evaluating six LALMs across four audio understanding tasks under two output constraint categories, we uncover a consistent asymmetry across all stages and LALMs: in-context demonstrations reliably improve format compliance but fail to improve, and often degrade, the core task performance. This suggests that LALMs can glean surface-level formatting patterns from demonstrations but may struggle to leverage cross-modal semantic grounding to reliably infer task objectives from audio-conditioned examples, highlighting potential limitations in current cross-modal integration.
>
---
#### [new 132] Email in the Era of LLMs
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文探讨LLM在邮件沟通中的表现，通过游戏实验分析人类与LLM的协作效果，旨在解决人机通信优化问题。**

- **链接: [https://arxiv.org/pdf/2603.20231](https://arxiv.org/pdf/2603.20231)**

> **作者:** Dang Nguyen; Harvey Yiyun Fu; Peter West; Chenhao Tan; Ari Holtzman
>
> **备注:** 47 pages (including appendix), 6 figures, 2 tables main body
>
> **摘要:** Email communication increasingly involves large language models (LLMs), but we lack intuition on how they will read, write, and optimize for nuanced social goals. We introduce HR Simulator, a game where communication is the core mechanic: players play as a Human Resources officer and write emails to solve socially challenging workplace scenarios. An analysis of 600+ human and LLM emails with LLMs-as-judge reveals evidence for larger LLMs becoming more homogenous in their email quality judgments. Under LLM judges, humans underperform LLMs (e.g., 23.5% vs. 48-54% success rate), but a human+LLM approach can outperform LLM-only (e.g., from 40% to nearly 100% in one scenario). In cases where models' email preferences disagree, emergent tact is a plausible explanation: weaker models prefer less tactful strategies while stronger models prefer more tactful ones. Regarding tone, LLM emails are more formal and empathetic while human emails are more varied. LLM rewrites make human emails more formal and empathetic, but models still struggle to imitate human emails in the low empathy, low formality quadrant, which highlights a limitation of current post-training approaches. Our results demonstrate the efficacy of communication games as instruments to measure communication in the era of LLMs, and posit human-LLM co-writing as an effective form of communication in that future.
>
---
#### [new 133] AE-LLM: Adaptive Efficiency Optimization for Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于大语言模型效率优化任务，解决模型部署中计算成本高、资源消耗大的问题。提出AE-LLM框架，自动选择最优效率技术，提升效率并保持准确率。**

- **链接: [https://arxiv.org/pdf/2603.20492](https://arxiv.org/pdf/2603.20492)**

> **作者:** Kaito Tanaka; Masato Ito; Yuji Nishimura; Keisuke Matsuda; Aya Nakayama
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable success across diverse applications, yet their deployment remains challenging due to substantial computational costs, memory requirements, and energy consumption. Recent empirical studies have demonstrated that no single efficiency technique is universally optimal; instead, the effectiveness of methods such as efficient attention mechanisms, mixture-of-experts (MoE), parameter-efficient fine-tuning, and quantization varies significantly depending on task characteristics, resource constraints, and model scales. Building upon these insights, we propose AE-LLM, a unified framework that automatically selects and combines optimal efficiency techniques tailored to specific deployment scenarios. Our approach introduces a multi-objective optimization framework that jointly considers accuracy, latency, memory footprint, and energy consumption, while accounting for hardware constraints and task requirements. We develop an efficient search algorithm that explores the combinatorial space of efficiency techniques across architecture, fine-tuning, and inference stages, identifying Pareto-optimal configurations. Extensive experiments across 15 models (0.5B-70B parameters) and 10 diverse tasks demonstrate that AE-LLM achieves an average of $2.8\times$ improvement in efficiency metrics while maintaining competitive accuracy (within 1.2\% of baseline), compared to static efficiency configurations. Furthermore, our framework generalizes effectively to vision-language models, achieving similar efficiency gains. Our contributions provide practitioners with an automated tool for navigating the complex trade-off landscape of LLM efficiency optimization.
>
---
#### [new 134] Mixture of Chapters: Scaling Learnt Memory in Transformers
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决Transformer模型缺乏显式记忆机制的问题。通过引入可学习的稀疏记忆库和章节路由方法，提升模型的记忆能力和训练稳定性。**

- **链接: [https://arxiv.org/pdf/2603.21096](https://arxiv.org/pdf/2603.21096)**

> **作者:** Tasmay Pankaj Tibrewal; Pritish Saha; Ankit Meda; Kunal Singh; Pradeep Moturi
>
> **备注:** 20 pages, 2 figures, 8 tables. Accepted at ICLR 2026 New Frontiers in Associative Memory Workshop. Code available at this https URL
>
> **摘要:** Transformers lack an explicit architectural mechanism for storing and organizing knowledge acquired during training. We introduce learnable sparse memory banks: a set of latent tokens, randomly initialized and trained end-to-end, that transformer layers query via cross-attention to retrieve stored knowledge. To scale memory capacity without prohibitive attention costs, we propose chapter-based routing inspired by Mixture-of-Experts architectures, partitioning the memory bank into chapters and training a router to select relevant subsets per input. This enables scaling to 262K memory tokens while maintaining tractable computation. We evaluate our approach against standard transformers (in iso-FLOP settings) on pre-training and instruction fine-tuning across relevant benchmarks. Our models surpass iso-FLOP baselines suggesting scope for a new axis of scaling, demonstrating that explicit associative memory provides complementary capacity to what is captured implicitly in model parameters. Additionally, we observe improved knowledge retention under continued training, with robustness to forgetting when transitioning between training phases (e.g., pretraining to instruction fine-tuning).
>
---
#### [new 135] Revenue-Sharing as Infrastructure: A Distributed Business Model for Generative AI Platforms
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 论文提出“收益共享即基础设施”模型，解决开发者进入门槛高问题，通过免费提供AI基础设施并分享收益，降低开发成本，促进创新与社会经济发展。**

- **链接: [https://arxiv.org/pdf/2603.20533](https://arxiv.org/pdf/2603.20533)**

> **作者:** Ghislain Dorian Tchuente Mondjo
>
> **备注:** 11 pages, 1 figures, 2 tables
>
> **摘要:** Generative AI platforms (Google AI Studio, OpenAI, Anthropic) provide infrastructures (APIs, models) that are transforming the application development ecosystem. Recent literature distinguishes three generations of business models: a first generation modeled on cloud computing (pay-per-use), a second characterized by diversification (freemium, subscriptions), and a third, emerging generation exploring multi-layer market architectures with revenue-sharing mechanisms. Despite these advances, current models impose a financial barrier to entry for developers, limiting innovation and excluding actors from emerging economies. This paper proposes and analyzes an original model, "Revenue-Sharing as Infrastructure" (RSI), where the platform offers its AI infrastructure for free and takes a percentage of the revenues generated by developers applications. This model reverses the traditional upstream payment logic and mobilizes concepts of value co-creation, incentive mechanisms, and multi-layer market architecture to build an original theoretical framework. A detailed comparative analysis shows that the RSI model lowers entry barriers for developers, aligns stakeholder interests, and could stimulate innovation in the ecosystem. Beyond its economic relevance, RSI has a major societal dimension: by enabling developers without initial capital to participate in the digital economy, it could unlock the "latent jobs dividend" in low-income countries, where mobile penetration reaches 84%, and help address local challenges in health, agriculture, and services. Finally, we discuss the conditions of feasibility and strategic implications for platforms and developers.
>
---
#### [new 136] PLR: Plackett-Luce for Reordering In-Context Learning Examples
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决In-Context Learning中示例顺序优化问题。提出PLR方法，通过Plackett-Luce模型学习最优顺序分布，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.21373](https://arxiv.org/pdf/2603.21373)**

> **作者:** Pawel Batorski; Paul Swoboda
>
> **摘要:** In-context learning (ICL) adapts large language models by conditioning on a small set of ICL examples, avoiding costly parameter updates. Among other factors, performance is often highly sensitive to the ordering of the examples. However, exhaustive search over the $n!$ possible orderings is infeasible. Therefore more efficient ordering methods use model confidence measures (e.g., label-probability entropy) over label sets or take a direct approach to finding the best ordering. We propose PLR, a probabilistic approach to in-context example ordering that replaces discrete ordering search with learning a probability distribution over orderings with the Plackett-Luce model. PLR models orderings using a Plackett-Luce distribution and iteratively updates its parameters to concentrate probability mass on high-performing orderings under a task-level metric. Candidate orderings are sampled efficiently via a Gumbel perturb-and-sort procedure. Experiments on multiple classification benchmarks show that PLR consistently improves few-shot accuracy for $k \in \{4, 8, 16, 32\}$ examples, and we further demonstrate gains on mathematical reasoning tasks where label-based ordering methods are not applicable. Our code is available at this https URL.
>
---
#### [new 137] GIP-RAG: An Evidence-Grounded Retrieval-Augmented Framework for Interpretable Gene Interaction and Pathway Impact Analysis
- **分类: q-bio.MN; cs.AI; cs.CL**

- **简介: 该论文提出GIP-RAG框架，解决基因互作与通路影响的可解释分析问题，整合知识图谱与大模型实现多步推理。**

- **链接: [https://arxiv.org/pdf/2603.20321](https://arxiv.org/pdf/2603.20321)**

> **作者:** Fujian Jia; Jiwen Gu; Cheng Lu; Dezhi Zhao; Mengjiang Huang; Yuanzhi Lu; Xin Liu; Kang Liu
>
> **备注:** 29 pages
>
> **摘要:** Understanding mechanistic relationships among genes and their impacts on biological pathways is essential for elucidating disease mechanisms and advancing precision medicine. Despite the availability of extensive molecular interaction and pathway data in public databases, integrating heterogeneous knowledge sources and enabling interpretable multi-step reasoning across biological networks remain challenging. We present GIP-RAG (Gene Interaction Prediction through Retrieval-Augmented Generation), a computational framework that combines biomedical knowledge graphs with large language models (LLMs) to infer and interpret gene interactions. The framework constructs a unified gene interaction knowledge graph by integrating curated data from KEGG, WikiPathways, SIGNOR, Pathway Commons, and PubChem. Given user-specified genes, a query-driven module retrieves relevant subgraphs, which are incorporated into structured prompts to guide LLM-based stepwise reasoning. This enables identification of direct and indirect regulatory relationships and generation of mechanistic explanations supported by biological evidence. Beyond pairwise interactions, GIP-RAG includes a pathway-level functional impact module that simulates propagation of gene perturbations through signaling networks and evaluates potential pathway state changes. Evaluation across diverse biological scenarios demonstrates that the framework generates consistent, interpretable, and evidence-supported insights into gene regulatory mechanisms. Overall, GIP-RAG provides a general and interpretable approach for integrating knowledge graphs with retrieval-augmented LLMs to support mechanistic reasoning in complex molecular systems.
>
---
#### [new 138] Epistemic Observability in Language Models
- **分类: cs.DC; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究语言模型的可信性问题，探讨为何模型在虚构时表现出高自信。通过分析发现，文本监督下无法区分真实与虚构输出，并提出基于熵值的检测方法提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2603.20531](https://arxiv.org/pdf/2603.20531)**

> **作者:** Tony Mason
>
> **摘要:** We find that models report highest confidence precisely when they are fabricating. Across four model families (OLMo-3, Llama-3.1, Qwen3, Mistral), self-reported confidence inversely correlates with accuracy, with AUC ranging from 0.28 to 0.36 where 0.5 is random guessing. We prove, under explicit formal assumptions, that this is not a capability gap but an observational one. Under text-only observation, where a supervisor sees only the model's output text, no monitoring system can reliably distinguish honest model outputs from plausible fabrications. We prove two results: first, that any policy conditioning only on the query cannot satisfy epistemic honesty across ambiguous world states; second, that no learning algorithm optimizing reward from a text-only supervisor can converge to honest behavior when the supervisor's observations are identical for both grounded and fabricated responses. Within our formal model, these impossibilities hold regardless of model scale or training procedure, including RLHF and instruction tuning. We construct a tensor interface that escapes the impossibility by exporting computational byproducts (per-token entropy and log-probability distributions) that are structurally coupled to correctness under standard training. Per-token entropy achieves pooled AUC 0.757, outperforming all text baselines by 2.5--3.9 percentage points at every budget level tested (10\%, 20\%, 30\%). The entropy signal generalizes across architectures (Spearman $\rho = 0.762$). The core contribution is a cost surface where the empirical mapping from verification budget (fraction of queries receiving expensive checks) to detection accuracy for each judge strategy is a practical lookup for system builders deciding how to allocate verification resources. The contribution is the map. The territory is the system you are building.
>
---
#### [new 139] NDT: Non-Differential Transformer and Its Application to Sentiment Analysis
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文属于情感分析任务，旨在解决文本情感准确捕捉的问题。提出Non-Differential Transformer模型，通过正向注意力融合提升情感理解效果。**

- **链接: [https://arxiv.org/pdf/2603.20704](https://arxiv.org/pdf/2603.20704)**

> **作者:** Soudeep Ghoshal; Himanshu Buckchash; Sarita Paudel; Rubén Ruiz-Torrubiano
>
> **备注:** 10 pages, 16 figures. Submitted to IEEE Transactions on Computational Social Systems
>
> **摘要:** From customer feedback to social media, understanding human sentiment in text is central to how machines can interact meaningfully with people. However, despite notable progress, accurately capturing sentiment remains a challenging task, which continues to motivate further research in this area. To this end, we introduce Non-Differential Transformer (NDT). It is inspired by (but in contrast to) the state-of-the-art Differential Transformer (DT) model. While standard Transformers can struggle with irrelevant context, the sota DT model uses attention map subtraction, potentially for noise cancellation. We explore an alternative motivation, hypothesizing that benefits may arise from enabling different attention components to specialize on distinct concepts within the text, similar to multiplexing information channels or mixture models, rather than primarily canceling noise via subtraction. Guided by this concept-multiplexing (ConPlex) view, the specific architecture presented in this paper employs a purely additive strategy. It uses only positive weights, learned during training, to ensure constructive combination of these specialized attention perspectives. This design choice explores positive only integration, though our broader framework also shows promise with less constrained linear combinations involving both positive and negative weights. Our model computes attention via this positively weighted sum of multiple distinct attention maps. This allows the model to constructively integrate diverse signals and potentially capture more complex contextual relationships. Competitive performance is achieved by the proposed model for Sentiment Analysis while tested on multiple datasets. We conclude by presenting our results, challenges and future research agenda in this important area of research.
>
---
#### [new 140] EvoIdeator: Evolving Scientific Ideas through Checklist-Grounded Reinforcement Learning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于科学创意生成任务，旨在解决LLM在迭代优化研究提案中的不足。提出EvoIdeator框架，结合检查清单反馈进行强化学习，提升创意质量与可行性。**

- **链接: [https://arxiv.org/pdf/2603.21728](https://arxiv.org/pdf/2603.21728)**

> **作者:** Andreas Sauter; Yuyue Zhao; Jacopo Urbani; Wenxiang Hu; Zaiqiao Meng; Lun Zhou; Xiaohui Yan; Yougang Lyu
>
> **摘要:** Scientific idea generation is a cornerstone of autonomous knowledge discovery, yet the iterative evolution required to transform initial concepts into high-quality research proposals remains a formidable challenge for Large Language Models (LLMs). Existing Reinforcement Learning (RL) paradigms often rely on rubric-based scalar rewards that provide global quality scores but lack actionable granularity. Conversely, language-based refinement methods are typically confined to inference-time prompting, targeting models that are not explicitly optimized to internalize such critiques. To bridge this gap, we propose \textbf{EvoIdeator}, a framework that facilitates the evolution of scientific ideas by aligning the RL training objective with \textbf{checklist-grounded feedback}. EvoIdeator leverages a structured judge model to generate two synergistic signals: (1) \emph{lexicographic rewards} for multi-dimensional optimization, and (2) \emph{fine-grained language feedback} that offers span-level critiques regarding grounding, feasibility, and methodological rigor. By integrating these signals into the RL loop, we condition the policy to systematically utilize precise feedback during both optimization and inference. Extensive experiments demonstrate that EvoIdeator, built on Qwen3-4B, significantly outperforms much larger frontier models across key scientific metrics. Crucially, the learned policy exhibits strong generalization to diverse external feedback sources without further fine-tuning, offering a scalable and rigorous path toward self-refining autonomous ideation.
>
---
#### [new 141] TIDE: Token-Informed Depth Execution for Per-Token Early Exit in LLM Inference
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出TIDE系统，用于大语言模型推理中的逐token提前退出。解决传统模型对每个token遍历所有层的低效问题，通过学习路由器选择最早收敛层，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2603.21365](https://arxiv.org/pdf/2603.21365)**

> **作者:** Jaber Jaber; Osama Jaber
>
> **备注:** 9 pages, 5 tables, 2 figures. Code: this https URL
>
> **摘要:** Large language models run every token through every layer, regardless of difficulty. We present TIDE, a post-training system that attaches tiny learned routers at periodic checkpoint layers and, at inference time, selects the earliest layer whose hidden state has converged for each token. TIDE requires no model retraining, works with any HuggingFace causal LM, auto-detects GPU architecture, and supports float32, float16, and bfloat16 through fused CUDA kernels. On an NVIDIA A100 with DeepSeek R1 Distill 8B, TIDE achieves 100% prefill exit rate (5% of tokens exit at layer 11, the remaining at layer 31), reduces prefill latency by 7.2%, and increases single-batch throughput by 6.6%. During autoregressive decoding, 98-99% of tokens exit early while the model correctly solves a multi-step math problem with 95 unique output tokens. On Qwen3 8B (36 layers), throughput improves by 8.1% at batch size 8. Calibration on 2,000 WikiText samples takes under 3 minutes and produces a ~4 MB router checkpoint. The system comprises 1,308 lines of Python and 1,081 lines of CUDA/C++ with 74 passing tests. Code: this https URL
>
---
#### [new 142] LongCat-Flash-Prover: Advancing Native Formal Reasoning via Agentic Tool-Integrated Reinforcement Learning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出LongCat-Flash-Prover，用于提升Lean4中的形式化推理能力，解决自动形式化和定理证明问题，通过混合专家模型和强化学习方法实现高效推理。**

- **链接: [https://arxiv.org/pdf/2603.21065](https://arxiv.org/pdf/2603.21065)**

> **作者:** Jianing Wang; Jianfei Zhang; Qi Guo; Linsen Guo; Rumei Li; Chao Zhang; Chong Peng; Cunguang Wang; Dengchang Zhao; Jiarong Shi; Jingang Wang; Liulin Feng; Mengxia Shen; Qi Li; Shengnan An; Shun Wang; Wei Shi; Xiangyu Xi; Xiaoyu Li; Xuezhi Cao; Yi Lu; Yunke Zhao; Zhengyu Chen; Zhimin Lin; Wei Wang; Peng Pei; Xunliang Cai
>
> **备注:** 43 pages, 5 figures
>
> **摘要:** We introduce LongCat-Flash-Prover, a flagship 560-billion-parameter open-source Mixture-of- Experts (MoE) model that advances Native Formal Reasoning in Lean4 through agentic tool-integrated reasoning (TIR). We decompose the native formal reasoning task into three independent formal capabilities, i.e., auto-formalization, sketching, and proving. To facilitate these capabilities, we propose a Hybrid-Experts Iteration Framework to expand high-quality task trajectories, including generating a formal statement based on a given informal problem, producing a whole-proof directly from the statement, or a lemma-style sketch. During agentic RL, we present a Hierarchical Importance Sampling Policy Optimization (HisPO) algorithm, which aims to stabilize the MoE model training on such long-horizon tasks. It employs a gradient masking strategy that accounts for the policy staleness and the inherent train-inference engine discrepancies at both sequence and token levels. Additionally, we also incorporate theorem consistency and legality detection mechanisms to eliminate reward hacking issues. Extensive evaluations show that our LongCat-Flash-Prover sets a new state-of-the-art for open-weights models in both auto-formalization and theorem proving. Demonstrating remarkable sample efficiency, it achieves a 97.1% pass rate on MiniF2F-Test using only 72 inference budget per problem. On more challenging benchmarks, it solves 70.8% of ProverBench and 41.5% of PutnamBench with no more than 220 attempts per problem, significantly outperforming existing open-weights baselines.
>
---
#### [new 143] SqueezeComposer: Temporal Speed-up is A Simple Trick for Long-form Music Composing
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出SqueezeComposer，解决长时音乐生成中的计算与内存限制问题。通过先生成加速音频再恢复原速，实现高效、高质量的长音乐生成。**

- **链接: [https://arxiv.org/pdf/2603.21073](https://arxiv.org/pdf/2603.21073)**

> **作者:** Jianyi Chen; Rongxiu Zhong; Shilei Zhang; Kun Qian; Jinglei Liu; Yike Guo; Wei Xue
>
> **备注:** Under Review
>
> **摘要:** Composing coherent long-form music remains a significant challenge due to the complexity of modeling long-range dependencies and the prohibitive memory and computational requirements associated with lengthy audio representations. In this work, we propose a simple yet powerful trick: we assume that AI models can understand and generate time-accelerated (speeded-up) audio at rates such as 2x, 4x, or even 8x. By first generating a high-speed version of the music, we greatly reduce the temporal length and resource requirements, making it feasible to handle long-form music that would otherwise exceed memory or computational limits. The generated audio is then restored to its original speed, recovering the full temporal structure. This temporal speed-up and slow-down strategy naturally follows the principle of hierarchical generation from abstract to detailed content, and can be conveniently applied to existing music generation models to enable long-form music generation. We instantiate this idea in SqueezeComposer, a framework that employs diffusion models for generation in the accelerated domain and refinement in the restored domain. We validate the effectiveness of this approach on two tasks: long-form music generation, which evaluates temporal-wise control (including continuation, completion, and generation from scratch), and whole-song singing accompaniment generation, which evaluates track-wise control. Experimental results demonstrate that our simple temporal speed-up trick enables efficient, scalable, and high-quality long-form music generation. Audio samples are available at this https URL.
>
---
#### [new 144] The Presupposition Problem in Representation Genesis
- **分类: cs.AI; cs.CL**

- **简介: 论文探讨语言模型在表征生成过程中的预设问题，属于认知科学与哲学领域。它指出现有理论在解释表征起源时存在循环依赖，导致解释失效。工作是分析问题结构并提出解决条件。**

- **链接: [https://arxiv.org/pdf/2603.21745](https://arxiv.org/pdf/2603.21745)**

> **作者:** Yiling Wu
>
> **摘要:** Large language models are the first systems to achieve high cognitive performance without clearly undergoing representation genesis: the transition from a non-representing physical system to one whose states guide behavior in a content-sensitive way. Prior cognitive systems had already made this transition before we could examine it, and philosophy of mind treated genesis as a background condition rather than an explanatory target. LLMs provide a case that does not clearly involve this transition, making the genesis question newly urgent: if genesis did not occur, which cognitive capacities are affected, and why? We currently lack the conceptual resources to answer this. The reason, this paper argues, is structural. Major frameworks in philosophy of mind, including the Language of Thought hypothesis, teleosemantics, predictive processing, enactivism, and genetic phenomenology, share a common feature when applied to the genesis question: at some explanatory step, each deploys concepts whose explanatory purchase depends on the system already being organized as a representer. This pattern, which we call the Representation Presupposition structure, generates systematic explanatory deferral. Attempts to explain the first acquisition of content-manipulable representation within the existing categorical vocabulary import resources from the representational side of the transition itself. We call this the Representation Regress. The paper offers a conceptual diagnosis rather than a new theory, establishing the structure of the problem and deriving two minimum adequacy conditions for any account that avoids this pattern. LLMs make the absence of such a theory consequential rather than merely theoretical.
>
---
#### [new 145] SecureBreak -- A dataset towards safe and secure models
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于AI安全任务，旨在解决模型生成有害内容的问题。提出SecureBreak数据集，用于检测和防御安全对齐中的漏洞。**

- **链接: [https://arxiv.org/pdf/2603.21975](https://arxiv.org/pdf/2603.21975)**

> **作者:** Marco Arazzi; Vignesh Kumar Kembu; Antonino Nocera
>
> **摘要:** Large language models are becoming pervasive core components in many real-world applications. As a consequence, security alignment represents a critical requirement for their safe deployment. Although previous related works focused primarily on model architectures and alignment methodologies, these approaches alone cannot ensure the complete elimination of harmful generations. This concern is reinforced by the growing body of scientific literature showing that attacks, such as jailbreaking and prompt injection, can bypass existing security alignment mechanisms. As a consequence, additional security strategies are needed both to provide qualitative feedback on the robustness of the obtained security alignment at the training stage, and to create an ``ultimate'' defense layer to block unsafe outputs possibly produced by deployed models. To provide a contribution in this scenario, this paper introduces SecureBreak, a safety-oriented dataset designed to support the development of AI-driven solutions for detecting harmful LLM outputs caused by residual weaknesses in security alignment. The dataset is highly reliable due to careful manual annotation, where labels are assigned conservatively to ensure safety. It performs well in detecting unsafe content across multiple risk categories. Tests with pre-trained LLMs show improved results after fine-tuning on SecureBreak. Overall, the dataset is useful both for post-generation safety filtering and for guiding further model alignment and security improvements.
>
---
#### [new 146] SPA: A Simple but Tough-to-Beat Baseline for Knowledge Injection
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于知识注入任务，旨在解决LLM在专业领域知识不足的问题。通过设计简洁提示生成大量合成数据，提出SPA方法提升知识注入效果。**

- **链接: [https://arxiv.org/pdf/2603.22213](https://arxiv.org/pdf/2603.22213)**

> **作者:** Kexian Tang; Jiani Wang; Shaowen Wang; Kaifeng Lyu
>
> **摘要:** While large language models (LLMs) are pretrained on massive amounts of data, their knowledge coverage remains incomplete in specialized, data-scarce domains, motivating extensive efforts to study synthetic data generation for knowledge injection. We propose SPA (Scaling Prompt-engineered Augmentation), a simple but tough-to-beat baseline that uses a small set of carefully designed prompts to generate large-scale synthetic data for knowledge injection. Through systematic comparisons, we find that SPA outperforms several strong baselines. Furthermore, we identify two key limitations of prior approaches: (1) while RL-based methods may improve the token efficiency of LLM-based data augmentation at small scale, they suffer from diversity collapse as data scales, leading to diminishing returns; and (2) while multi-stage prompting may outperform simple augmentation methods, their advantages can disappear after careful prompt tuning. Our results suggest that, for knowledge injection, careful prompt design combined with straightforward large-scale augmentation can be surprisingly effective, and we hope SPA can serve as a strong baseline for future studies in this area. Our code is available at this https URL.
>
---
#### [new 147] OpenResearcher: A Fully Open Pipeline for Long-Horizon Deep Research Trajectory Synthesis
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文提出OpenResearcher，解决长周期深度研究轨迹生成问题。通过离线搜索环境和教师模型，合成大量研究轨迹，提升模型在复杂任务上的表现。**

- **链接: [https://arxiv.org/pdf/2603.20278](https://arxiv.org/pdf/2603.20278)**

> **作者:** Zhuofeng Li; Dongfu Jiang; Xueguang Ma; Haoxiang Zhang; Ping Nie; Yuyu Zhang; Kai Zou; Jianwen Xie; Yu Zhang; Wenhu Chen
>
> **摘要:** Training deep research agents requires long-horizon trajectories that interleave search, evidence aggregation, and multi-step reasoning. However, existing data collection pipelines typically rely on proprietary web APIs, making large-scale trajectory synthesis costly, unstable, and difficult to reproduce. We present OpenResearcher, a reproducible pipeline that decouples one-time corpus bootstrapping from multi-turn trajectory synthesis and executes the search-and-browse loop entirely offline using three explicit browser primitives: search, open, and find, over a 15M-document corpus. Using GPT-OSS-120B as the teacher model, we synthesize over 97K trajectories, including a substantial long-horizon tail with 100+ tool calls. Supervised fine-tuning a 30B-A3B backbone on these trajectories achieves 54.8\% accuracy on BrowseComp-Plus, a +34.0 point improvement over the base model, while remaining competitive on BrowseComp, GAIA, and xbench-DeepSearch. Because the environment is offline and fully instrumented, it also enables controlled analysis, where our study reveals practical insights into deep research pipeline design, including data filtering strategies, agent configuration choices, and how retrieval success relates to final answer accuracy. We release the pipeline, synthesized trajectories, model checkpoints, and the offline search environment at this https URL.
>
---
#### [new 148] Measuring Reasoning Trace Legibility: Can Those Who Understand Teach?
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理领域，研究语言模型推理过程的可读性。旨在评估模型推理轨迹的清晰度及其对弱模型的指导效果，发现高表现模型的推理轨迹可读性低，且奖励机制不鼓励可读性。**

- **链接: [https://arxiv.org/pdf/2603.20508](https://arxiv.org/pdf/2603.20508)**

> **作者:** Dani Roytburg; Shreya Sridhar; Daphne Ippolito
>
> **摘要:** Language models are increasingly being trained to "reason" before answering users' queries, outputting hundreds or even thousands of tokens worth of deliberation before their final answer. While the main intention of reasoning is to improve models' ability to arrive at a correct answer, we argue that these models should be assessed for the legibility of their reasoning traces in addition to the correctness of their final answers. In this paper, we evaluate 90k traces from 12 Reasoning Language Models (RLMs) for the quality of their reasoning traces. We introduce the concept of transfer utility, which assesses how useful an RLM's reasoning traces are for guiding a weaker, non-reasoning model toward arriving at the correct answer. We find that the reasoning traces of the highest-performing models rank among the lowest for legibility. Furthermore, we uncover tensions between efficiency-based measurements of legibility (such as trace length) and transfer utility. These tensions establish a legibility Pareto frontier, and we demonstrate that an RLM's ability to output highly legible traces can be a task- and audience-dependent goal. Crucially, we find that reward models used to train RLMs do not intrinsically reward legibility. Together, these metrics and the findings they surface chart a path towards scaffolding reasoning traces for a multi-agent future.
>
---
#### [new 149] DSPA: Dynamic SAE Steering for Data-Efficient Preference Alignment
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出DSPA方法，用于高效偏好对齐。解决传统方法计算量大、缺乏可解释性的问题，通过动态稀疏自编码器控制生成，提升效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2603.21461](https://arxiv.org/pdf/2603.21461)**

> **作者:** James Wedgwood; Aashiq Muhamed; Mona T. Diab; Virginia Smith
>
> **摘要:** Preference alignment is usually achieved by weight-updating training on preference data, which adds substantial alignment-stage compute and provides limited mechanistic visibility. We propose Dynamic SAE Steering for Preference Alignment (DSPA), an inference-time method that makes sparse autoencoder (SAE) steering prompt-conditional. From preference triples, DSPA computes a conditional-difference map linking prompt features to generation-control features; during decoding, it modifies only token-active latents, without base-model weight updates. Across Gemma-2-2B/9B and Qwen3-8B, DSPA improves MT-Bench and is competitive on AlpacaEval while preserving multiple-choice accuracy. Under restricted preference data, DSPA remains robust and can rival the two-stage RAHF-SCIT pipeline while requiring up to $4.47\times$ fewer alignment-stage FLOPs. Finally, we audit the SAE features DSPA modifies, finding that preference directions are dominated by discourse and stylistic signals, and provide theory clarifying the conditional-difference map estimate and when top-$k$ ablation is principled.
>
---
#### [new 150] Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出深度循环Transformer，解决变量深度推理任务的泛化问题，通过迭代共享权重块实现更深层次推理。**

- **链接: [https://arxiv.org/pdf/2603.21676](https://arxiv.org/pdf/2603.21676)**

> **作者:** Hung-Hsuan Chen
>
> **摘要:** Standard Transformers have a fixed computational depth, fundamentally limiting their ability to generalize to tasks requiring variable-depth reasoning, such as multi-hop graph traversal or nested logic. We propose a depth-recurrent Transformer that decouples computational depth from parameter count by iteratively applying a shared-weight Transformer block in latent space -- enabling the model to trade recurrence steps for deeper reasoning at inference time. Our architecture incorporates three mechanisms to make deep recurrence (20+ steps) stable: (1) a silent thinking objective that supervises only the final output, forcing genuine multi-step reasoning rather than intermediate heuristic shortcuts; (2) LayerScale initialization to protect fragile reasoning states from untrained layer noise; and (3) an identity-biased recurrence that creates a gradient highway across many steps. We evaluate on three compositional reasoning domains with decreasing inductive biases: graph reachability (strict adjacency masking), nested boolean logic (relative positioning), and unstructured relational text (where sequence position provides no structural hints). Across all tasks, we observe a clear \emph{computational frontier} -- a boundary where performance transitions from chance to near-perfect as thinking steps scale with task complexity. Moreover, these tasks reveal qualitatively different generalization behaviors: precise but brittle (graph), approximate but robust (logic), and autonomous latent routing without structural hints (text). This progression illuminates how the interplay between a task-invariant recurrent reasoning core and task-specific perceptual interfaces shapes out-of-distribution (OOD) generalization, offering a mechanistic perspective on vertical chain-of-thought that complements the prevailing horizontal token-generation paradigm.
>
---
#### [new 151] Disentangling Speaker Traits for Deepfake Source Verification via Chebyshev Polynomial and Riemannian Metric Learning
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音深度伪造源验证任务，旨在解决源与说话人特征混淆的问题。提出SDML框架，通过Chebyshev多项式和双曲空间投影实现特征解耦，提升验证效果。**

- **链接: [https://arxiv.org/pdf/2603.21875](https://arxiv.org/pdf/2603.21875)**

> **作者:** Xi Xuan; Wenxin Zhang; Zhiyu Li; Jennifer Williams; Ville Hautamäki; Tomi H. Kinnunen
>
> **备注:** Submitted to Interspeech 2026; The code, evaluation protocols and demo website are available at this https URL
>
> **摘要:** Speech deepfake source verification systems aims to determine whether two synthetic speech utterances originate from the same source generator, often assuming that the resulting source embeddings are independent of speaker traits. However, this assumption remains unverified. In this paper, we first investigate the impact of speaker factors on source verification. We propose a speaker-disentangled metric learning (SDML) framework incorporating two novel loss functions. The first leverages Chebyshev polynomial to mitigate gradient instability during disentanglement optimization. The second projects source and speaker embeddings into hyperbolic space, leveraging Riemannian metric distances to reduce speaker information and learn more discriminative source features. Experimental results on MLAAD benchmark, evaluated under four newly proposed protocols designed for source-speaker disentanglement scenarios, demonstrate the effectiveness of SDML framework. The code, evaluation protocols and demo website are available at this https URL.
>
---
#### [new 152] WorldCache: Content-Aware Caching for Accelerated Video World Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出WorldCache，解决视频世界模型中的计算效率问题。通过动态特征重用，提升推理速度并保持高质量，适用于加速扩散Transformer的视频生成任务。**

- **链接: [https://arxiv.org/pdf/2603.22286](https://arxiv.org/pdf/2603.22286)**

> **作者:** Umair Nawaz; Ahmed Heakl; Ufaq Khan; Abdelrahman Shaker; Salman Khan; Fahad Shahbaz Khan
>
> **备注:** 33 Pages
>
> **摘要:** Diffusion Transformers (DiTs) power high-fidelity video world models but remain computationally expensive due to sequential denoising and costly spatio-temporal attention. Training-free feature caching accelerates inference by reusing intermediate activations across denoising steps; however, existing methods largely rely on a Zero-Order Hold assumption i.e., reusing cached features as static snapshots when global drift is small. This often leads to ghosting artifacts, blur, and motion inconsistencies in dynamic scenes. We propose \textbf{WorldCache}, a Perception-Constrained Dynamical Caching framework that improves both when and how to reuse features. WorldCache introduces motion-adaptive thresholds, saliency-weighted drift estimation, optimal approximation via blending and warping, and phase-aware threshold scheduling across diffusion steps. Our cohesive approach enables adaptive, motion-consistent feature reuse without retraining. On Cosmos-Predict2.5-2B evaluated on PAI-Bench, WorldCache achieves \textbf{2.3$\times$} inference speedup while preserving \textbf{99.4\%} of baseline quality, substantially outperforming prior training-free caching approaches. Our code can be accessed on \href{this https URL}{World-Cache}.
>
---
## 更新

#### [replaced 001] Human or LLM as Standardized Patients? A Comparative Study for Medical Education
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于医学教育任务，旨在解决传统标准化患者成本高、难扩展的问题。通过提出EasyMED框架和SPBench基准，比较LLM与人类标准化患者的教学效果。**

- **链接: [https://arxiv.org/pdf/2511.14783](https://arxiv.org/pdf/2511.14783)**

> **作者:** Bingquan Zhang; Xiaoxiao Liu; Yuchi Wang; Lei Zhou; Qianqian Xie; Benyou Wang
>
> **备注:** 24 pages, 13 figures, 10 table
>
> **摘要:** Standardized patients (SPs) are indispensable for clinical skills training but remain expensive and difficult to scale. Although large language model (LLM)-based virtual standardized patients (VSPs) have been proposed as an alternative, their behavior remains unstable and lacks rigorous comparison with human standardized patients. We propose EasyMED, a multi-agent VSP framework that separates case-grounded information disclosure from response generation to support stable, inquiry-conditioned patient behavior. We also introduce SPBench, a human-grounded benchmark with eight expert-defined criteria for interaction-level evaluation. Experiments show that EasyMED more closely matches human SP behavior than existing VSPs, particularly in case consistency and controlled disclosure. A four-week controlled study further demonstrates learning outcomes comparable to human SP training, with stronger early gains for novice learners and improved flexibility, psychological safety, and cost efficiency.
>
---
#### [replaced 002] Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM中sycophancy行为的机制问题。通过分解sycophantic agreement和praise，证明其为独立表示，可分别调控。**

- **链接: [https://arxiv.org/pdf/2509.21305](https://arxiv.org/pdf/2509.21305)**

> **作者:** Daniel Vennemeyer; Phan Anh Duong; Tiffany Zhan; Tianyu Jiang
>
> **摘要:** Large language models (LLMs) often exhibit sycophantic behaviors -- such as excessive agreement with or flattery of the user -- but it is unclear whether these behaviors arise from a single mechanism or multiple distinct processes. We decompose sycophancy into sycophantic agreement and sycophantic praise, contrasting both with genuine agreement. Using difference-in-means directions, activation additions, and subspace geometry across multiple models and datasets, we show that: (1) the three behaviors are encoded along distinct linear directions in latent space; (2) each behavior can be independently amplified or suppressed without affecting the others; and (3) their representational structure is consistent across model families and scales. These results suggest that sycophantic behaviors correspond to distinct, independently steerable representations.
>
---
#### [replaced 003] Mining Legal Arguments to Study Judicial Formalism
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于法律文本分析任务，旨在解决司法推理自动识别与分类问题。通过构建数据集并应用NLP技术，实现法律论证检测与形式主义判决分类。**

- **链接: [https://arxiv.org/pdf/2512.11374](https://arxiv.org/pdf/2512.11374)**

> **作者:** Tomáš Koref; Lena Held; Mahammad Namazov; Harun Kumru; Yassine Thlija; Ivan Habernal
>
> **备注:** pre-print under review
>
> **摘要:** Courts must justify their decisions, but systematically analyzing judicial reasoning at scale remains difficult. This study tests claims about formalistic judging in Central and Eastern Europe (CEE) by developing automated methods to detect and classify judicial reasoning in decisions of Czech Supreme Courts using state-of-the-art natural language processing methods. We create the MADON dataset of 272 decisions from two Czech Supreme Courts with expert annotations of 9,183 paragraphs with eight argument types and holistic formalism labels for supervised training and evaluation. Using a corpus of 300,511 Czech court decisions, we adapt transformer LLMs to Czech legal domain through continued pretraining and we experiment with methods to address dataset imbalance including asymmetric loss and class weighting. The best models can detect argumentative paragraphs (82.6% Bal-F1), classify traditional types of legal argument (77.5% Bal-F1), and classify decisions as formalistic/non-formalistic (83.8% Bal-F1). Our three-stage pipeline combining ModernBERT, Llama 3.1, and traditional feature-based machine learning achieves promising results for decision classification while reducing computational costs and increasing explainability. Empirically, we challenge prevailing narratives about CEE formalism. We demonstrate that legal argument mining enables promising judicial philosophy classification and highlight its potential for other important tasks in computational legal studies. Our methodology can be used across jurisdictions, and our entire pipeline, datasets, guidelines, models, and source codes are available at this https URL.
>
---
#### [replaced 004] Multi-Task Instruction Tuning via Data Scheduling for Low-Resource Arabic AudioLLMs
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文研究低资源阿拉伯语音频大模型的多任务指令调优，解决语音理解与生成中的方言和情感识别问题。提出AraMega-SSum数据集及多种训练策略，提升模型性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2601.12494](https://arxiv.org/pdf/2601.12494)**

> **作者:** Hunzalah Hassan Bhatti; Firoj Alam; Shammur Absar Chowdhury
>
> **备注:** Foundation Models, Large Language Models, Native, Speech Models, Arabic
>
> **摘要:** Audio large language models (LLMs) enable unified speech understanding and generation, but adapting them to linguistically complex and dialect-rich settings such as Arabic-English remains challenging. We present a controlled study of multi-task instruction tuning for an Arabic-centric audio LLM across generative tasks including ASR and speech and text summarization, and discriminative tasks including dialect and emotion recognition, in a resource-constrained setting. To support end-to-end Arabic speech summarization, we introduce AraMega-SSum, a first speech summarization resource for training and benchmarking Arabic-centric Audio-LLMs. We compare four training strategies (i) Uniform Task Mixing, (ii) Task-Progressive Curriculum (TPC), (iiii) Aligner-Based Diverse Sampling (ADS) for training-time batch construction, and (iv) A two-stage TPC->ADS strategy. Our results show a clear efficiency-robustness trade-off. ADS speeds up early convergence and improves paralinguistic performance, however, it hurts other tasks. A two-stage TPC-> ADS strategy gives the most reliable overall balance across tasks, offering practical guidance for adapting omni audio LLMs to low-resource, dialect-rich environments. We will make AraMega-SSum and all experimental resources publicly available to the community.
>
---
#### [replaced 005] Conflict-Aware Fusion: Mitigating Logic Inertia in Large Language Models via Structured Cognitive Priors
- **分类: cs.AI; cs.CL; cs.LG; cs.LO**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在规则系统扰动下的推理可靠性问题。通过设计测试框架，提出Conflict-Aware Fusion方法，提升模型对矛盾的敏感性和推理准确性。**

- **链接: [https://arxiv.org/pdf/2512.06393](https://arxiv.org/pdf/2512.06393)**

> **作者:** Qiming Bao; Xiaoxuan Fu; Michael Witbrock
>
> **备注:** Under review as a conference paper at ICLR 2026
>
> **摘要:** Large language models (LLMs) excel at many natural language tasks, yet their reasoning reliability under structured perturbations of rule-based systems remains brittle. We present a controlled evaluation framework consisting of four stress tests: (1) rule deletion (redundant vs. essential), (2) contradictory evidence injection, (3) logic-preserving rewrites, and (4) multi-law equivalence stacking. While representative model families (BERT, Qwen2, and TinyLlama) achieve Acc = 1.0000 on base tasks, our framework reveals a critical failure mode termed Logic Inertia - a total breakdown with Acc = 0.0000 under contradictions, where deductive momentum overrides factual reality. To address this, we propose Conflict-Aware Fusion (Fusion-Conflict), a framework grounded in the Cognitive Structure Hypothesis, which posits that robust reasoning requires an explicit structural inductive bias. By imposing a dual-process architecture that separates premise verification from logical deduction, Conflict-Aware Fusion effectively mitigates logic inertia under the proposed evaluation framework, achieving 1.0000 accuracy on both base and contradictory stress tests. It also significantly enhances robustness to missing evidence. Our results demonstrate that, for reliable multi-step reasoning, structural verification discipline is as critical as training data scale, providing a potential blueprint for building robust, contradiction-aware AI systems this this https URL . See the OpenAI/Evals pull request this this https URL .
>
---
#### [replaced 006] MobileIPL: Enhancing Mobile Agents Thinking Process via Iterative Preference Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于移动代理任务，解决GUI任务中推理能力不足的问题。通过迭代偏好学习提升移动代理的思考过程，增强其泛化能力。**

- **链接: [https://arxiv.org/pdf/2505.12299](https://arxiv.org/pdf/2505.12299)**

> **作者:** Kun Huang; Weikai Xu; Yuxuan Liu; Quandong Wang; Pengzhi Gao; Wei Liu; Jian Luan; Bin Wang; Bo An
>
> **备注:** 9 pages, 8 figures, 7 tables
>
> **摘要:** The Chain of Action-Planning Thoughts (CoaT) paradigm has been shown to improve the reasoning performance of VLM-based mobile agents in GUI tasks. However, the scarcity of diverse CoaT trajectories limits the expressiveness and generalization ability of such agents. While self-training is commonly employed to address data scarcity, existing approaches either overlook the correctness of intermediate reasoning steps or depend on expensive process-level annotations to construct process reward models (PRM). To address the above problems, we propose an Iterative Preference Learning (IPL) that constructs a CoaT-tree through interative sampling, scores leaf nodes using rule-based reward, and backpropagates feedback to derive Thinking-level Direct Preference Optimization (T-DPO) pairs. To prevent overfitting during warm-up supervised fine-tuning, we further introduce a three-stage instruction evolution, which leverages GPT-4o to generate diverse Q\&A pairs based on real mobile UI screenshots, enhancing both generality and layout understanding. Experiments on three standard Mobile GUI-agent benchmarks demonstrate that our agent MobileIPL outperforms strong baselines, including continual pretraining models such as OS-ATLAS and UI-TARS. It achieves state-of-the-art performance across three standard Mobile GUI-Agents benchmarks and shows strong generalization to out-of-domain scenarios.
>
---
#### [replaced 007] Long Chain-of-Thought Reasoning Across Languages
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究多语言长链推理能力的迁移问题，探讨模型在不同语言中的推理表现与优化方法。**

- **链接: [https://arxiv.org/pdf/2508.14828](https://arxiv.org/pdf/2508.14828)**

> **作者:** Josh Barua; Seun Eisape; Kayo Yin; Alane Suhr
>
> **备注:** Accepted to ICLR 2026. v1 is a workshop version accepted to SCALR @ COLM 2025
>
> **摘要:** While large reasoning models have shown remarkable ability to generate long chains-of-thought (CoTs) in English, we still lack understanding of how these long-form reasoning abilities transfer to the vast majority of the world's languages. In this work, we systematically investigate four key stages of model development--scaling, pretraining, post-training, and inference--to understand how long CoT capabilities extend beyond English. We compare two reasoning settings across nine non-English target languages: En-CoT, where models process target-language inputs, but reason in English; and Target-CoT, where models both process inputs and generate long CoTs in the target language. We find that scaling reasoning model size improves multilingual task performance in En-CoT, but Target-CoT performance lags behind. This gap widens for tasks requiring long, multi-step CoTs such as mathematical reasoning. Shifting to pretraining, we find that adding a specialized reasoning stage enhances En-CoT performance but degrades Target-CoT, whereas broad multilingual pretraining improves both modes simultaneously. Given the scarcity of high-quality reasoning traces in languages other than English, we explore synthetic data curation approaches for post-training. We demonstrate that fine-tuning on reasoning traces automatically translated from gold English traces outperforms fine-tuning on target-language traces distilled from large reasoning models. Finally, we report disparities in inference efficiency between languages and uncover language-specific failure modes in CoTs. We release models, datasets, and code to foster further research.
>
---
#### [replaced 008] Instructional Text Across Disciplines: A Survey of Representations, Downstream Tasks, and Open Challenges Toward Capable AI Agents
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决复杂指令理解问题。通过综述181篇文献，分析资源、表示方法和下游任务，为AI代理提供统一视角和研究方向。**

- **链接: [https://arxiv.org/pdf/2410.18529](https://arxiv.org/pdf/2410.18529)**

> **作者:** Abdulfattah Safa; Tamta Kapanadze; Arda Uzunoğlu; Gözde Gül Şahin
>
> **备注:** Pre-CoLI print. Accepted for publication in Computational Linguistics (MIT Press). Advance online publication. March 2026
>
> **摘要:** Recent advances in large language models have demonstrated promising capabilities in following simple instructions through instruction tuning. However, real-world tasks often involve complex, multi-step instructions that remain challenging for current NLP systems. Robust understanding of such instructions is essential for deploying LLMs as general-purpose agents that can be programmed in natural language to perform complex, real-world tasks across domains like robotics, business automation, and interactive systems. Despite growing interest in this area, there is a lack of a comprehensive survey that systematically analyzes the landscape of complex instruction understanding and processing. Through a systematic review of the literature, we analyze available resources, representation schemes, and downstream tasks related to instructional text. Our study examines 181 papers, identifying trends, challenges, and opportunities in this emerging field. We provide AI/NLP researchers with essential background knowledge and a unified view of various approaches to complex instruction understanding, bridging gaps between different research directions and highlighting future research opportunities.
>
---
#### [replaced 009] Pretraining with hierarchical memories: separating long-tail and common knowledge
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决大模型参数冗余与边缘设备限制问题。通过引入分层记忆机制，提升小模型性能，实现高效知识存储与推理。**

- **链接: [https://arxiv.org/pdf/2510.02375](https://arxiv.org/pdf/2510.02375)**

> **作者:** Hadi Pouransari; David Grangier; C Thomas; Michael Kirchhof; Oncel Tuzel
>
> **备注:** ICLR 2026
>
> **摘要:** The impressive performance gains of modern language models currently rely on scaling parameters: larger models store more world knowledge and reason better. Yet compressing all world knowledge into parameters is unnecessary, as only a fraction is used per prompt, and impractical for edge devices with limited inference-time memory and compute. We address this shortcoming by a memory-augmented architecture and a pretraining strategy aligned with existing hardware paradigms. We introduce small language models that access large hierarchical parametric memory banks encoding world knowledge. During pretraining and inference, we fetch a small, context-dependent memory block and add it to the model. Our pretraining learns to store long-tail world knowledge in the memory parameters, while the small language model acts as an anchor capturing common knowledge and general reasoning abilities. Through trillion-token-scale experiments, we show significant gains: a 160M-parameters model augmented with an 18M-parameters memory fetched from a 4.6B memory bank obtains comparable performance to a regular model with more than 2x the parameters. Through extensive experiments, we study the optimal type and size of parametric memories in transformers, scaling them to over 21B parameters. We find that our proposed hierarchical feed-forward memories work robustly across transformer architectures, whether added during pretraining or post-hoc.
>
---
#### [replaced 010] APEX-SWE
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出APEX-SWE，评估AI模型执行软件工程任务的能力，解决现有评估过于狭窄的问题。工作包括设计两种真实任务并测试多个模型。**

- **链接: [https://arxiv.org/pdf/2601.08806](https://arxiv.org/pdf/2601.08806)**

> **作者:** Abhi Kottamasu; Chirag Mahapatra; Sam Lee; Ben Pan; Aakash Barthwal; Akul Datta; Anurag Gupta; Pranav Mehta; Ajay Arun; Silas Alberti; Adarsh Hiremath; Brendan Foody; Bertie Vidgen
>
> **摘要:** We introduce the AI Productivity Index for Software Engineering (APEX-SWE), a benchmark for assessing whether frontier AI models can execute economically valuable software engineering work. Unlike existing evaluations that focus on narrow, well-defined tasks, APEX-SWE assesses two novel task types that reflect real-world software engineering: (1) Integration tasks (n=100), which require constructing end-to-end systems across heterogeneous cloud primitives, business applications, and infrastructure-as-code services, and (2) Observability tasks (n=100), which require debugging production failures using telemetry signals such as logs and dashboards, as well as unstructured context. We evaluated eleven frontier models for the APEX-SWE leaderboard. Claude Opus 4.6 leads the APEX-SWE leaderboard with 40.5% Pass@1, followed by Claude Opus 4.5 at 38.7%. Our analysis shows that strong performance is primarily driven by epistemic discipline, defined as the capacity to distinguish between assumptions and verified facts. It is often combined with systematic verification prior to acting. We open-source the APEX-SWE evaluation harness and a dev set (n=50).
>
---
#### [replaced 011] FACTUM: Mechanistic Detection of Citation Hallucination in Long-Form RAG
- **分类: cs.CL**

- **简介: 该论文属于RAG系统中的引用幻觉检测任务，旨在解决模型错误引用问题。提出FACTUM框架，通过四个机制评分提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.05866](https://arxiv.org/pdf/2601.05866)**

> **作者:** Maxime Dassen; Rebecca Kotula; Kenton Murray; Andrew Yates; Dawn Lawrie; Efsun Kayi; James Mayfield; Kevin Duh
>
> **备注:** Accepted at ECIR 2026. 13 pages, 2 figures
>
> **摘要:** Retrieval-Augmented Generation (RAG) models are critically undermined by citation hallucinations, a deceptive failure where a model cites a source that fails to support its claim. While existing work attributes hallucination to a simple over-reliance on parametric knowledge, we reframe this failure as an evolving, scale-dependent coordination failure between the Attention (reading) and Feed-Forward Network (recalling) pathways. We introduce FACTUM (Framework for Attesting Citation Trustworthiness via Underlying Mechanisms), a framework of four mechanistic scores: Contextual Alignment (CAS), Attention Sink Usage (BAS), Parametric Force (PFS), and Pathway Alignment (PAS). Our analysis reveals that correct citations are consistently marked by higher parametric force (PFS) and greater use of the attention sink (BAS) for information synthesis. Crucially, we find that "one-size-fits-all" theories are insufficient as the signature of correctness evolves with scale: while the 3B model relies on high pathway alignment (PAS), our best-performing 8B detector identifies a shift toward a specialized strategy where pathways provide distinct, orthogonal information. By capturing this complex interplay, FACTUM outperforms state-of-the-art baselines by up to 37.5% in AUC. Our results demonstrate that high parametric force is constructive when successfully coordinated with the Attention pathway, paving the way for more nuanced and reliable RAG systems.
>
---
#### [replaced 012] Chain of Retrieval: Multi-Aspect Iterative Search Expansion and Post-Order Search Aggregation for Full Paper Retrieval
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于科学论文检索任务，解决长文档查询的全篇匹配问题。提出COR框架，通过多视角迭代扩展和后序聚合提升检索效果。**

- **链接: [https://arxiv.org/pdf/2507.10057](https://arxiv.org/pdf/2507.10057)**

> **作者:** Sangwoo Park; Jinheon Baek; Soyeong Jeong; Sung Ju Hwang
>
> **摘要:** Scientific paper retrieval, particularly framed as document-to-document retrieval, aims to identify relevant papers in response to a long-form query paper, rather than a short query string. Previous approaches to this task have focused exclusively on abstracts, embedding them into dense vectors as surrogates for full documents and calculating similarity between them. Yet, abstracts offer only sparse and high-level summaries, and such methods primarily optimize one-to-one similarity, overlooking the dynamic relations that emerge across relevant papers during the retrieval process. To address this, we propose Chain of Retrieval(COR), a novel iterative framework for full-paper retrieval. Specifically, COR decomposes each query paper into multiple aspect-specific views, matches them against segmented candidate papers, and iteratively expands the search by promoting top-ranked results as new queries, thereby forming a tree-structured retrieval process. The resulting retrieval tree is then aggregated in a post-order manner: descendants are first combined at the query level, then recursively merged with their parent nodes, to capture hierarchical relations across iterations. To validate this, we present SCIFULLBENCH, a large-scale benchmark providing both complete and segmented contexts of full papers for queries and candidates, and results show that COR significantly outperforms existing retrieval baselines. Our code and dataset is available at this https URL.
>
---
#### [replaced 013] Putting on the Thinking Hats: A Survey on Chain of Thought Fine-tuning from the Perspective of Human Reasoning Mechanism
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在解决如何使大语言模型具备人类推理能力的问题。通过分析人类思维机制，系统梳理CoT微调方法，并提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2510.13170](https://arxiv.org/pdf/2510.13170)**

> **作者:** Xiaoshu Chen; Sihang Zhou; Ke Liang; Duanyang Yuan; Haoyuan Chen; Xiaoyu Sun; Lingyuan Meng; Xinwang Liu
>
> **摘要:** Chain of thought (CoT) fine-tuning aims to endow large language models (LLMs) with reasoning capabilities by training them on curated reasoning traces. It leverages both supervised and reinforced fine-tuning to cultivate human-like reasoning skills in LLMs, including detailed planning, divergent thinking, intuitive judgment, timely reflection, internal thinking, and fact perception, etc. As CoT fine-tuning has advanced, LLMs have demonstrated substantial improvements in tasks such as mathematical reasoning and code generation. However, existing surveys about CoT fine-tuning primarily focus on technical aspects and overlook a systematic analysis from the perspective of human reasoning mechanisms. Given that the ultimate goal of CoT fine-tuning is to enable LLMs to reason like humans, it is crucial to investigate this technique through the lens of human cognition. To fill this gap, we present the first comprehensive survey of CoT fine-tuning grounded in human reasoning theory. Specifically, inspired by the well-known Six Thinking Hats framework, which systematically characterizes common human thinking modes using six metaphorical hats, we classify and examine CoT fine-tuning methods through this lens. Furthermore, building upon this theory, we outline potential directions for future research in CoT fine-tuning. In addition, we compile a comprehensive overview of existing datasets and model performances, and a real-time GitHub repository \footnote{this https URL} that continuously tracks recent advances in this area is maintained. We hope this survey will serve as a valuable resource to inspire innovation and foster progress in this rapidly evolving field.
>
---
#### [replaced 014] Edu-Values: Towards Evaluating the Chinese Education Values of Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出Edu-Values，一个评估中文教育价值观的基准，旨在解决大语言模型在教育价值理解上的不足。通过设计多种题型测试模型，发现中文模型表现更优，并提出利用该基准提升模型对齐效果。**

- **链接: [https://arxiv.org/pdf/2409.12739](https://arxiv.org/pdf/2409.12739)**

> **作者:** Peiyi Zhang; Yazhou Zhang; Bo Wang; Lu Rong; Prayag Tiwari; Jing Qin
>
> **备注:** The authors are withdrawing this paper to make substantial revisions and improvements before future submission
>
> **摘要:** In this paper, we present Edu-Values, the first Chinese education values evaluation benchmark that includes seven core values: professional philosophy, teachers' professional ethics, education laws and regulations, cultural literacy, educational knowledge and skills, basic competencies and subject knowledge. We meticulously design 1,418 questions, covering multiple-choice, multi-modal question answering, subjective analysis, adversarial prompts, and Chinese traditional culture (short answer) questions. We conduct human feedback based automatic evaluation over 21 state-of-the-art (SoTA) LLMs, and highlight three main findings: (1) due to differences in educational culture, Chinese LLMs outperform English LLMs, with Qwen 2 ranking the first with a score of 81.37; (2) LLMs often struggle with teachers' professional ethics and professional philosophy; (3) leveraging Edu-Values to build an external knowledge repository for RAG significantly improves LLMs' alignment. This demonstrates the effectiveness of the proposed benchmark.
>
---
#### [replaced 015] Breaking the Silence: A Dataset and Benchmark for Bangla Text-to-Gloss Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于 Bangla 文本到手语符号的翻译任务，旨在解决 BdSL 数据匮乏的问题。构建了首个 Bangla 文本-手语数据集，并对比了多种模型的翻译效果。**

- **链接: [https://arxiv.org/pdf/2504.02293](https://arxiv.org/pdf/2504.02293)**

> **作者:** Sharif Mohammad Abdullah; Abhijit Paul; Shubhashis Roy Dipta; Zarif Masud; Shebuti Rayana; Ahmedul Kabir
>
> **摘要:** Gloss is a written approximation that bridges Sign Language (SL) and its corresponding spoken language. Despite a deaf and hard-of-hearing population of at least 3 million in Bangladesh, Bangla Sign Language (BdSL) remains largely understudied, with no prior work on Bangla text-to-gloss translation and no publicly available datasets. To address this gap, we construct the first Bangla text-to-gloss dataset, consisting of 1,000 manually annotated and 4,000 synthetically generated Bangla sentence-gloss pairs, along with 159 expert human-annotated pairs used as a test set. Our experimental framework performs a comparative analysis between several fine-tuned open-source models and a leading closed-source LLM to evaluate their performance in low-resource BdSL translation. GPT-5.4 achieves the best overall performance, while a fine-tuned mBART model performs competitively despite being approximately 100% smaller. Qwen-3 outperforms all other models in human evaluation. This work introduces the first dataset and trained model for Bangla text-to-gloss translation. It also demonstrates the effectiveness of systematically generated synthetic data for addressing challenges in low-resource sign language translation.
>
---
#### [replaced 016] CoVerRL: Breaking the Consensus Trap in Label-Free Reasoning via Generator-Verifier Co-Evolution
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决无标签强化学习中的共识陷阱问题。提出CoVerRL框架，通过生成器与验证器协同进化提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2603.17775](https://arxiv.org/pdf/2603.17775)**

> **作者:** Teng Pan; Yuchen Yan; Zixuan Wang; Ruiqing Zhang; Guiyang Hou; Wenqi Zhang; Weiming Lu; Jun Xiao; Yongliang Shen
>
> **备注:** Project Page: this https URL Code: this https URL
>
> **摘要:** Label-free reinforcement learning enables large language models to improve reasoning capabilities without ground-truth supervision, typically by treating majority-voted answers as pseudo-labels. However, we identify a critical failure mode: as training maximizes self-consistency, output diversity collapses, causing the model to confidently reinforce systematic errors that evade detection. We term this the consensus trap. To escape it, we propose CoVerRL, a framework where a single model alternates between generator and verifier roles, with each capability bootstrapping the other. Majority voting provides noisy but informative supervision for training the verifier, while the improving verifier progressively filters self-consistent errors from pseudo-labels. This co-evolution creates a virtuous cycle that maintains high reward accuracy throughout training. Experiments across Qwen and Llama model families demonstrate that CoVerRL outperforms label-free baselines by 4.7-5.9% on mathematical reasoning benchmarks. Moreover, self-verification accuracy improves from around 55% to over 85%, confirming that both capabilities genuinely co-evolve.
>
---
#### [replaced 017] Script Sensitivity: Benchmarking Language Models on Unicode, Romanized and Mixed-Script Sinhala
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型评估任务，旨在解决低资源语言在不同书写系统下的性能问题。通过基准测试24个模型，分析其在Unicode、罗马化及混合脚本的僧伽罗语中的表现。**

- **链接: [https://arxiv.org/pdf/2601.14958](https://arxiv.org/pdf/2601.14958)**

> **作者:** Minuri Rajapakse; Ruvan Weerasinghe
>
> **备注:** Accepted at the 9th International Research Conference on Smart Computing and Systems Engineering (SCSE). To appear in IEEE proceedings
>
> **摘要:** The performance of Language Models (LMs) on low-resource, morphologically rich languages like Sinhala remains largely unexplored, particularly regarding script variation in digital communication. Sinhala exhibits script duality, with Unicode used in formal contexts and Romanized text dominating social media, while mixed-script usage is common in practice. This paper benchmarks 24 open-source LMs on Unicode, Romanized and mixed-script Sinhala using perplexity evaluation across diverse text sources. Results reveal substantial script sensitivity, with median performance degradation exceeding 300 times from Unicode to Romanized text. Critically, model size shows no correlation with script-handling competence, as smaller models often outperform architectures 28 times larger. Unicode performance strongly predicts mixed-script robustness but not Romanized capability, demonstrating that single-script evaluation substantially underestimates real-world deployment challenges. These findings establish baseline LM capabilities for Sinhala and provide practical guidance for model selection in multi-script low-resource environments.
>
---
#### [replaced 018] Knowing What's Missing: Assessing Information Sufficiency in Question Answering
- **分类: cs.CL**

- **简介: 该论文属于问答系统任务，旨在解决判断给定上下文是否足够回答问题的问题。提出一种结构化框架，通过识别并验证缺失信息来提升判断准确性。**

- **链接: [https://arxiv.org/pdf/2512.06476](https://arxiv.org/pdf/2512.06476)**

> **作者:** Akriti Jain; Aparna Garimella
>
> **备注:** Accepted to EACL Findings 2026
>
> **摘要:** Determining whether a provided context contains sufficient information to answer a question is a critical challenge for building reliable question-answering systems. While simple prompting strategies have shown success on factual questions, they frequently fail on inferential ones that require reasoning beyond direct text extraction. We hypothesize that asking a model to first reason about what specific information is missing provides a more reliable, implicit signal for assessing overall sufficiency. To this end, we propose a structured Identify-then-Verify framework for robust sufficiency modeling. Our method first generates multiple hypotheses about missing information and establishes a semantic consensus. It then performs a critical verification step, forcing the model to re-examine the source text to confirm whether this information is truly absent. We evaluate our method against established baselines across diverse multi-hop and factual QA datasets. The results demonstrate that by guiding the model to justify its claims about missing information, our framework produces more accurate sufficiency judgments while clearly articulating any information gaps.
>
---
#### [replaced 019] TRI-DEP: A Trimodal Comparative Study for Depression Detection Using Speech, Text, and EEG
- **分类: cs.AI; cs.CL; cs.LG; eess.AS; eess.SP**

- **简介: 该论文属于抑郁症检测任务，旨在通过语音、文本和脑电（EEG）的多模态分析提升检测效果。研究比较了不同特征和模型配置，验证了预训练嵌入和融合策略的有效性。**

- **链接: [https://arxiv.org/pdf/2510.14922](https://arxiv.org/pdf/2510.14922)**

> **作者:** Annisaa Fitri Nurfidausi; Eleonora Mancini; Paolo Torroni
>
> **摘要:** Depression is a widespread mental health disorder, yet its automatic detection remains challenging. Prior work has explored unimodal and multimodal approaches, with multimodal systems showing promise by leveraging complementary signals. However, existing studies are limited in scope, lack systematic comparisons of features, and suffer from inconsistent evaluation protocols. We address these gaps by systematically exploring feature representations and modelling strategies across EEG, together with speech and text. We evaluate handcrafted features versus pre-trained embeddings, assess the effectiveness of different neural encoders, compare unimodal, bimodal, and trimodal configurations, and analyse fusion strategies with attention to the role of EEG. Consistent subject-independent splits are applied to ensure robust, reproducible benchmarking. Our results show that (i) the combination of EEG, speech and text modalities enhances multimodal detection, (ii) pretrained embeddings outperform handcrafted features, and (iii) carefully designed trimodal models achieve state-of-the-art performance. Our work lays the groundwork for future research in multimodal depression detection.
>
---
#### [replaced 020] Pantagruel: Unified Self-Supervised Encoders for French Text and Speech
- **分类: cs.CL**

- **简介: 该论文提出Pantagruel模型，用于法语文本和语音的统一自监督编码。解决多模态表示学习问题，通过共享架构提升效果。**

- **链接: [https://arxiv.org/pdf/2601.05911](https://arxiv.org/pdf/2601.05911)**

> **作者:** Phuong-Hang Le; Valentin Pelloin; Arnault Chatelain; Maryem Bouziane; Mohammed Ghennai; Qianwen Guan; Kirill Milintsevich; Salima Mdhaffar; Aidan Mannion; Nils Defauw; Shuyue Gu; Alexandre Audibert; Marco Dinarelli; Yannick Estève; Lorraine Goeuriot; Steffen Lalande; Nicolas Hervé; Maximin Coavoux; François Portet; Étienne Ollion; Marie Candito; Maxime Peyrard; Solange Rossato; Benjamin Lecouteux; Aurélie Nardy; Gilles Sérasset; Vincent Segonne; Solène Evain; Diandra Fabre; Didier Schwab
>
> **备注:** Accepted to LREC 2026
>
> **摘要:** We release Pantagruel models, a new family of self-supervised encoder models for French text and speech. Instead of predicting modality-tailored targets such as textual tokens or speech units, Pantagruel learns contextualized target representations in the feature space, allowing modality-specific encoders to capture linguistic and acoustic regularities more effectively. Separate models are pre-trained on large-scale French corpora, including Wikipedia, OSCAR and CroissantLLM for text, together with MultilingualLibriSpeech, LeBenchmark, and INA-100k for speech. INA-100k is a newly introduced 100,000-hour corpus of French audio derived from the archives of the Institut National de l'Audiovisuel (INA), the national repository of French radio and television broadcasts, providing highly diverse audio data. We evaluate Pantagruel across a broad range of downstream tasks spanning both modalities, including those from the standard French benchmarks such as FLUE or LeBenchmark. Across these tasks, Pantagruel models show competitive or superior performance compared to strong French baselines such as CamemBERT, FlauBERT, and LeBenchmark2.0, while maintaining a shared architecture that can seamlessly handle either speech or text inputs. These results confirm the effectiveness of feature-space self-supervised objectives for French representation learning and highlight Pantagruel as a robust foundation for multimodal speech-text understanding.
>
---
#### [replaced 021] SciLaD: A Large-Scale, Transparent, Reproducible Dataset for Natural Scientific Language Processing
- **分类: cs.CL**

- **简介: 该论文提出SciLaD，一个大规模、透明的科学语言数据集，用于自然科学语言处理任务。旨在解决科学文献数据不足与可复现性问题，通过开源工具构建高质量数据集并预训练模型验证其有效性。**

- **链接: [https://arxiv.org/pdf/2512.11192](https://arxiv.org/pdf/2512.11192)**

> **作者:** Luca Foppiano; Sotaro Takeshita; Pedro Ortiz Suarez; Ekaterina Borisova; Raia Abu Ahmad; Malte Ostendorff; Fabio Barth; Julian Moreno-Schneider; Georg Rehm
>
> **备注:** 13 pages, 3 figures, 3 tables
>
> **摘要:** SciLaD is a novel, large-scale dataset of scientific language constructed entirely using open-source frameworks and publicly available data sources. It comprises a curated English split containing over 10 million scientific publications and a multilingual, unfiltered TEI XML split including more than 35 million publications. We also publish the extensible pipeline for generating SciLaD. The dataset construction and processing workflow demonstrates how open-source tools can enable large-scale, scientific data curation while maintaining high data quality. Finally, we pre-train a RoBERTa model on our dataset and evaluate it across a comprehensive set of benchmarks, achieving performance comparable to other scientific language models of similar size, validating the quality and utility of SciLaD. We publish the dataset and evaluation pipeline to promote reproducibility, transparency, and further research in natural scientific language processing and understanding, including scholarly document processing.
>
---
#### [replaced 022] M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出M4-RAG，解决多语言多文化多模态VQA中的信息检索问题，构建大规模基准数据集以评估模型性能。**

- **链接: [https://arxiv.org/pdf/2512.05959](https://arxiv.org/pdf/2512.05959)**

> **作者:** David Anugraha; Patrick Amadeus Irawan; Anshul Singh; En-Shiun Annie Lee; Genta Indra Winata
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Vision-language models (VLMs) have achieved strong performance in visual question answering (VQA), yet they remain constrained by static training data. Retrieval-Augmented Generation (RAG) mitigates this limitation by enabling access to up-to-date, culturally grounded, and multilingual information; however, multilingual multimodal RAG remains largely underexplored. We introduce M4-RAG, a massive-scale benchmark spanning 42 languages, 56 regional dialects and registers, and 189 countries, comprising over 80,000 culturally diverse image-question pairs for evaluating retrieval-augmented VQA across languages and modalities. To balance realism with reproducibility, we build a controlled retrieval environment containing millions of carefully curated multilingual documents relevant to the query domains, approximating real-world retrieval conditions while ensuring consistent experimentation. Our systematic evaluation reveals that although RAG consistently benefits smaller VLMs, it fails to scale to larger models and often even degrades their performance, exposing a critical mismatch between model size and current retrieval effectiveness. Our cross-lingual evaluations also reveal significant performance degradation when prompts or retrieved context are provided in non-English languages. The code, datasets, and evaluation protocols for M4-RAG are available as open-source at this https URL.
>
---
#### [replaced 023] Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究语言建模中的离散扩散模型，解决如何高效估计条件分布的问题。提出RADD模型，简化计算并提升性能。**

- **链接: [https://arxiv.org/pdf/2406.03736](https://arxiv.org/pdf/2406.03736)**

> **作者:** Jingyang Ou; Shen Nie; Kaiwen Xue; Fengqi Zhu; Jiacheng Sun; Zhenguo Li; Chongxuan Li
>
> **摘要:** Discrete diffusion models with absorbing processes have shown promise in language modeling. The key quantities to be estimated are the ratios between the marginal probabilities of two transitive states at all timesteps, called the concrete score. In this paper, we reveal that the concrete score in absorbing diffusion can be expressed as conditional probabilities of clean data, multiplied by a time-dependent scalar in an analytic form. Motivated by this finding, we propose reparameterized absorbing discrete diffusion (RADD), a dedicated diffusion model without time-condition that characterizes the time-independent conditional probabilities. Besides its simplicity, RADD can reduce the number of function evaluations (NFEs) by caching the output of the time-independent network when the noisy sample remains unchanged in a sampling interval, which enables sampling acceleration. Built upon the new perspective of conditional distributions, we further unify absorbing discrete diffusion and any-order autoregressive models (AO-ARMs), showing that the upper bound on the negative log-likelihood for the diffusion model can be interpreted as an expected negative log-likelihood for AO-ARMs. Further, our RADD models achieve SOTA performance among diffusion models on 5 zero-shot language modeling benchmarks (measured by perplexity) at the GPT-2 scale. Our code is available at this https URL.
>
---
#### [replaced 024] MolLangBench: A Comprehensive Benchmark for Language-Prompted Molecular Structure Recognition, Editing, and Generation
- **分类: cs.CL; cs.AI; cs.LG; q-bio.BM**

- **简介: 该论文提出MolLangBench，用于评估分子结构的识别、编辑和生成任务，解决AI在化学领域处理分子语言接口的问题。**

- **链接: [https://arxiv.org/pdf/2505.15054](https://arxiv.org/pdf/2505.15054)**

> **作者:** Feiyang Cai; Jiahui Bai; Tao Tang; Guijuan He; Joshua Luo; Tianyu Zhu; Srikanth Pilla; Gang Li; Ling Liu; Feng Luo
>
> **备注:** ICLR-2026 Camera-Ready version
>
> **摘要:** Precise recognition, editing, and generation of molecules are essential prerequisites for both chemists and AI systems tackling various chemical tasks. We present MolLangBench, a comprehensive benchmark designed to evaluate fundamental molecule-language interface tasks: language-prompted molecular structure recognition, editing, and generation. To ensure high-quality, unambiguous, and deterministic outputs, we construct the recognition tasks using automated cheminformatics tools, and curate editing and generation tasks through rigorous expert annotation and validation. MolLangBench supports the evaluation of models that interface language with different molecular representations, including linear strings, molecular images, and molecular graphs. Evaluations of state-of-the-art models reveal significant limitations: the strongest model (GPT-5) achieves $86.2\%$ and $85.5\%$ accuracy on recognition and editing tasks, which are intuitively simple for humans, and performs even worse on the generation task, reaching only $43.0\%$ accuracy. These results highlight the shortcomings of current AI systems in handling even preliminary molecular recognition and manipulation tasks. We hope MolLangBench will catalyze further research toward more effective and reliable AI systems for chemical this http URL dataset and code can be accessed at this https URL and this https URL, respectively.
>
---
#### [replaced 025] Automatic Analysis of Collaboration Through Human Conversational Data Resources: A Review
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在利用对话数据分析协作过程。通过回顾相关理论与方法，解决如何从任务导向的对话中自动解析协作问题。**

- **链接: [https://arxiv.org/pdf/2603.19292](https://arxiv.org/pdf/2603.19292)**

> **作者:** Yi Yu; Maria Boritchev; Chloé Clavel
>
> **备注:** 9 pages
>
> **摘要:** Collaboration is a task-oriented, high-level human behavior. In most cases, conversation serves as the primary medium for information exchange and coordination, making conversational data a valuable resource for the automatic analysis of collaborative processes. In this paper, we focus on verbal aspects of collaboration and conduct a review of collaboration analysis using task-oriented conversation resources, encompassing related theories, coding schemes, tasks, and modeling approaches. We aim to address the question of how to utilize task-oriented human-human conversational data for collaboration analysis. We hope our review will serve as a practical resource and illuminate unexplored areas for future collaboration analysis.
>
---
#### [replaced 026] FinTradeBench: A Financial Reasoning Benchmark for LLMs
- **分类: cs.CE; cs.AI; cs.CL; cs.IR; q-fin.CP**

- **简介: 该论文提出FinTradeBench，一个评估大语言模型金融推理能力的基准，解决现有基准缺乏市场交易信号与公司基本面综合推理的问题。**

- **链接: [https://arxiv.org/pdf/2603.19225](https://arxiv.org/pdf/2603.19225)**

> **作者:** Yogesh Agrawal; Aniruddha Dutta; Md Mahadi Hasan; Santu Karmaker; Aritra Dutta
>
> **备注:** 8 pages main text, 22 pages total (including references and appendix). 5 figures, 14 tables. Preprint under review. Code and data will be made available upon publication
>
> **摘要:** Real-world financial decision-making is a challenging problem that requires reasoning over heterogeneous signals, including company fundamentals derived from regulatory filings and trading signals computed from price dynamics. Recently, with the advancement of Large Language Models (LLMs), financial analysts have begun to use them for financial decision-making tasks. However, existing financial question answering benchmarks for testing these models primarily focus on company balance sheet data and rarely evaluate reasoning over how company stocks trade in the market or their interactions with fundamentals. To take advantage of the strengths of both approaches, we introduce FinTradeBench, a benchmark for evaluating financial reasoning that integrates company fundamentals and trading signals. FinTradeBench contains 1,400 questions grounded in NASDAQ-100 companies over a ten-year historical window. The benchmark is organized into three reasoning categories: fundamentals-focused, trading-signal-focused, and hybrid questions requiring cross-signal reasoning. To ensure reliability at scale, we adopt a calibration-then-scaling framework that combines expert seed questions, multi-model response generation, intra-model self-filtering, numerical auditing, and human-LLM judge alignment. We evaluate 14 LLMs under zero-shot prompting and retrieval-augmented settings and witness a clear performance gap. Retrieval substantially improves reasoning over textual fundamentals, but provides limited benefit for trading-signal reasoning. These findings highlight fundamental challenges in the numerical and time-series reasoning for current LLMs and motivate future research in financial intelligence.
>
---
#### [replaced 027] Levels of Analysis for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文探讨如何用认知科学方法理解大语言模型，属于模型解释任务。解决模型复杂难懂的问题，提出基于Marr分析层次的框架，提供分析工具。**

- **链接: [https://arxiv.org/pdf/2503.13401](https://arxiv.org/pdf/2503.13401)**

> **作者:** Alexander Y. Ku; Declan Campbell; Xuechunzi Bai; Jiayi Geng; Ryan Liu; Raja Marjieh; R. Thomas McCoy; Andrew Nam; Ilia Sucholutsky; Veniamin Veselovsky; Liyi Zhang; Jian-Qiao Zhu; Thomas L. Griffiths
>
> **摘要:** Modern artificial intelligence systems, such as large language models, are increasingly powerful but also increasingly hard to understand. Recognizing this problem as analogous to the historical difficulties in understanding the human mind, we argue that methods developed in cognitive science can be useful for understanding large language models. We propose a framework for applying these methods based on the levels of analysis that David Marr proposed for studying information processing systems. By revisiting established cognitive science techniques relevant to each level and illustrating their potential to yield insights into the behavior and internal organization of large language models, we aim to provide a toolkit for making sense of these new kinds of minds.
>
---
#### [replaced 028] Moneyball with LLMs: Analyzing Tabular Summarization in Sports Narratives
- **分类: cs.CL**

- **简介: 该论文属于长文本表格生成任务，旨在解决多实体跟踪与统计聚合问题。通过构建基准数据集，评估分解策略效果，发现模型在多实体记忆上存在瓶颈。**

- **链接: [https://arxiv.org/pdf/2510.18173](https://arxiv.org/pdf/2510.18173)**

> **作者:** Ritam Upadhyay; Naman Ahuja; Rishabh Baral; Aparna Garimella; Vivek Gupta
>
> **摘要:** Large language model (LLM) approaches to tabular summarization rely on extensive prompt engineering, decomposition pipelines, or entity-level intermediate representations to achieve strong performance. While effective, these strategies are computationally expensive and offer limited insight into how well models maintain state over long, evolving narratives. We introduce SPORTABSET, a diagnostic benchmark for long-context tabular summarization across two complementary sports domains that require tracking multiple entities and aggregating statistics under domain-specific rules. Using SporTabSet, we systematically evaluate decomposition-based strategies across several long context LLMs. Results show that although decomposition substantially improves accuracy and numerical fidelity, gains stem mainly from dissecting multi-entity interference rather than improved local arithmetic. Robustness experiments further reveal high sensitivity to surface-level cues with structured failures, including hallucination, omission, and role confusion. Together, these findings identify consistent multientity memory as a key bottleneck in long context table generation, motivating diagnostic evaluation as a prerequisite for scalable, efficient and reliable tabular summarization models.
>
---
#### [replaced 029] LexInstructEval: Lexical Instruction Following Evaluation for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型精准遵循细粒度词汇指令的评估问题。提出LexInstructEval框架，通过形式化语法生成数据并实现客观验证。**

- **链接: [https://arxiv.org/pdf/2511.17561](https://arxiv.org/pdf/2511.17561)**

> **作者:** Huimin Ren; Yan Liang; Baiqiao Su; Chaobo Sun; Hengtong Lu; Kaike Zhang; Chen Wei
>
> **摘要:** The ability of Large Language Models (LLMs) to precisely follow complex and fine-grained lexical instructions is a cornerstone of their utility and controllability. However, evaluating this capability remains a significant challenge. Current methods either rely on subjective and costly human evaluation or on automated LLM-as-a-judge systems, which suffer from inherent biases and unreliability. Existing programmatic benchmarks, while objective, often lack the expressiveness to test intricate, compositional constraints at a granular level. To address these limitations, we introduce LexInstructEval, a new benchmark and evaluation framework for fine-grained lexical instruction following. Our framework is built upon a formal, rule-based grammar that deconstructs complex instructions into a canonical <Procedure, Relation, Value> triplet. This grammar enables the systematic generation of a diverse dataset through a multi-stage, human-in-the-loop pipeline and facilitates objective verification via a transparent, programmatic engine. We release our dataset and open-source evaluation tools to facilitate further research into the controllability and reliability of LLMs.
>
---
#### [replaced 030] Rethinking Evaluation in Retrieval-Augmented Personalized Dialogue: A Cognitive and Linguistic Perspective
- **分类: cs.CL**

- **简介: 该论文属于对话系统评估任务，旨在解决现有评价方法忽视对话深度问题。通过分析LAPDOG框架，指出当前评估指标的不足，并提出基于认知和语言的评估方法。**

- **链接: [https://arxiv.org/pdf/2603.14217](https://arxiv.org/pdf/2603.14217)**

> **作者:** Tianyi Zhang; David Traum
>
> **摘要:** In cognitive science and linguistic theory, dialogue is not seen as a chain of independent utterances but rather as a joint activity sustained by coherence, consistency, and shared understanding. However, many systems for open-domain and personalized dialogue use surface-level similarity metrics (e.g., BLEU, ROUGE, F1) as one of their main reporting measures, which fail to capture these deeper aspects of conversational quality. We re-examine a notable retrieval-augmented framework for personalized dialogue, LAPDOG, as a case study for evaluation methodology. Using both human and LLM-based judges, we identify limitations in current evaluation practices, including corrupted dialogue histories, contradictions between retrieved stories and persona, and incoherent response generation. Our results show that human and LLM judgments align closely but diverge from lexical similarity metrics, underscoring the need for cognitively grounded evaluation methods. Broadly, this work charts a path toward more reliable assessment frameworks for retrieval-augmented dialogue systems that better reflect the principles of natural human communication.
>
---
#### [replaced 031] Does Geo-co-location Matter? A Case Study of Public Health Conversations during COVID-19
- **分类: cs.SI; cs.CL**

- **简介: 该论文属于社会媒体分析任务，研究地理邻近性对公共健康对话的影响，旨在探讨本地化互动如何提升社交媒体参与度。**

- **链接: [https://arxiv.org/pdf/2405.17710](https://arxiv.org/pdf/2405.17710)**

> **作者:** Paiheng Xu; Louiqa Raschid; Vanessa Frias-Martinez
>
> **备注:** ICWSM 2026
>
> **摘要:** Social media platforms like Twitter (now X) have been pivotal in information dissemination and public engagement. The objective of our research is to analyze the effect of localized engagement on social media conversations. This study examines the impact of geographic co-location, as a proxy for localized engagement. Our research is grounded in a COVID-19 dataset. A key goal during the pandemic for public health experts was to encourage prosocial behavior that could impact local outcomes such as masking and social distancing. Given the importance of local news and guidance during COVID-19, we analyze the effect of localized engagement, between public health experts (PHEs) and the public, on social media. We analyze a Twitter Conversation dataset from January 2020 to November 2021, comprising over 19 K tweets from nearly five hundred PHEs, and 800 K replies from 350 K participants. We use a Poisson regression model to show that geo-co-location is indeed associated with higher engagement. Lexical features associated with emotion and personal experiences were more common in geo-co-located conversations. To complement our statistical analysis, we also applied a large language model (LLM)-based method to automatically generate and evaluate hypotheses; the LLM results confirm the results using lexical features. This research provides insights into how geographic co-location influences social media engagement and can inform strategies to improve public health messaging.
>
---
#### [replaced 032] Measuring Iterative Temporal Reasoning with Time Puzzles
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于时间推理任务，旨在评估大语言模型在使用工具时的迭代时间推理能力。提出Time Puzzles基准，发现模型在工具辅助下表现仍有限。**

- **链接: [https://arxiv.org/pdf/2601.07148](https://arxiv.org/pdf/2601.07148)**

> **作者:** Zhengxiang Wang; Zeyu Dong
>
> **备注:** 11 pages, 4 tables, 3 figures
>
> **摘要:** Tool use, such as web search, has become a standard capability even in freely available large language models (LLMs). However, existing benchmarks evaluate temporal reasoning mainly in static, non-tool-using settings, which poorly reflect how LLMs perform temporal reasoning in practice. We introduce Time Puzzles, a constraint-based date inference task for evaluating iterative temporal reasoning with tools. Each puzzle combines factual temporal anchors with (cross-cultural) calendar relations and may admit one or multiple valid dates. The puzzles are algorithmically generated, enabling controlled and continual evaluation. Across 13 LLMs, even the best model (GPT-5) achieves only 55.3% accuracy without tools, despite using easily searchable facts. While web search improves performance, models perform substantially better when constraints are rewritten with explicit dates, removing the need for factual lookup. These results reveal a gap in reliable tool use for iterative temporal reasoning.
>
---
#### [replaced 033] PA3: Policy-Aware Agent Alignment through Chain-of-Thought
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于对话系统任务，解决大语言模型在遵循复杂业务规则时的对齐问题。通过引入政策回忆机制和奖励函数，提升模型在推理中应用相关政策的能力，减少上下文长度并提高性能。**

- **链接: [https://arxiv.org/pdf/2603.14602](https://arxiv.org/pdf/2603.14602)**

> **作者:** Shubhashis Roy Dipta; Daniel Bis; Kun Zhou; Lichao Wang; Benjamin Z. Yao; Chenlei Guo; Ruhi Sarikaya
>
> **摘要:** Conversational assistants powered by large language models (LLMs) excel at tool-use tasks but struggle with adhering to complex, business-specific rules. While models can reason over business rules provided in context, including all policies for every query introduces high latency and wastes compute. Furthermore, these lengthy prompts lead to long contexts, harming overall performance due to the "needle-in-the-haystack" problem. To address these challenges, we propose a multi-stage alignment method that teaches models to recall and apply relevant business policies during chain-of-thought reasoning at inference time, without including the full business policy in-context. Furthermore, we introduce a novel PolicyRecall reward based on the Jaccard score and a Hallucination Penalty for GRPO training. Altogether, our best model outperforms the baseline by 16 points and surpasses comparable in-context baselines of similar model size by 3 points, while using 40% fewer words.
>
---
#### [replaced 034] DeepCompress: A Dual Reward Strategy for Dynamically Exploring and Compressing Reasoning Chains
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出DeepCompress，解决大模型推理效率与准确率的平衡问题。通过双奖励策略，动态调整推理链长度，提升效率与准确性。属于模型优化任务。**

- **链接: [https://arxiv.org/pdf/2510.27419](https://arxiv.org/pdf/2510.27419)**

> **作者:** Tian Liang; Wenxiang Jiao; Zhiwei He; Jiahao Xu; Haitao Mi; Dong Yu
>
> **备注:** ICLR 2026
>
> **摘要:** Large Reasoning Models (LRMs) have demonstrated impressive capabilities but suffer from cognitive inefficiencies like "overthinking" simple problems and "underthinking" complex ones. While existing methods that use supervised fine-tuning (SFT) or reinforcement learning (RL) with token-length rewards can improve efficiency, they often do so at the cost of accuracy. This paper introduces DeepCompress, a novel framework that simultaneously enhances both the accuracy and efficiency of LRMs. We challenge the prevailing approach of consistently favoring shorter reasoning paths, showing that longer responses can contain a broader range of correct solutions for difficult problems. DeepCompress employs an adaptive length reward mechanism that dynamically classifies problems as "Simple" or "Hard" in real-time based on the model's evolving capability. It encourages shorter, more efficient reasoning for "Simple" problems while promoting longer, more exploratory thought chains for "Hard" problems. This dual-reward strategy enables the model to autonomously adjust its Chain-of-Thought (CoT) length, compressing reasoning for well-mastered problems and extending it for those it finds challenging. Experimental results on challenging mathematical benchmarks show that DeepCompress consistently outperforms baseline methods, achieving superior accuracy while significantly improving token efficiency.
>
---
#### [replaced 035] Difficulty-Controllable Multiple-Choice Question Generation Using Large Language Models and Direct Preference Optimization
- **分类: cs.CL**

- **简介: 该论文属于阅读理解中的题目生成任务，旨在解决传统方法无法直接生成选择题及难度控制不精准的问题。工作是提出一种基于大语言模型和直接偏好优化的难度可控选择题生成方法。**

- **链接: [https://arxiv.org/pdf/2510.19265](https://arxiv.org/pdf/2510.19265)**

> **作者:** Yuto Tomikawa; Masaki Uto
>
> **备注:** Accepted for publication in IEEE Access. Please refer to the published version for the final content. DOI: https://doi.org/10.1109/ACCESS.2026.3674595
>
> **摘要:** Difficulty-controllable question generation for reading comprehension has gained significant attention in the field of education as a fundamental tool for adaptive learning support. Although several neural question generation methods have recently succeeded in controlling difficulty, conventional approaches still face two major limitations. First, they cannot directly generate multiple-choice questions, which are the most widely used question type in educational contexts. Second, they are not explicitly trained to optimize the accuracy of difficulty control, leaving room for further improvement in difficulty controllability. To address these limitations, this study proposes a novel difficulty-controllable multiple-choice question generation method for reading comprehension which leverages a large language model trained using a direct preference optimization technique to improve the accuracy of difficulty control.
>
---
#### [replaced 036] An evolutionary perspective on modes of learning in Transformers
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究Transformer模型在不同环境下的学习策略，分析IWL与ICL的适用条件，旨在理解其学习机制与适应性。**

- **链接: [https://arxiv.org/pdf/2505.09855](https://arxiv.org/pdf/2505.09855)**

> **作者:** Alexander Y. Ku; Thomas L. Griffiths; Stephanie C.Y. Chan
>
> **摘要:** The success of Transformers lies in their ability to improve inference through two complementary strategies: the permanent refinement of model parameters via in-weight learning (IWL), and the ephemeral modulation of inferences via in-context learning (ICL), which leverages contextual information maintained in the model's activations. Evolutionary biology tells us that the predictability of the environment across timescales predicts the extent to which analogous strategies should be preferred. Genetic evolution adapts to stable environmental features by gradually modifying the genotype over generations. Conversely, environmental volatility favors plasticity, which enables a single genotype to express different traits within a lifetime, provided there are reliable cues to guide the adaptation. We operationalize these dimensions (environmental stability and cue reliability) in controlled task settings (sinusoid regression and Omniglot classification) to characterize their influence on learning in Transformers. We find that stable environments favor IWL, often exhibiting a sharp transition when conditions are static. Conversely, reliable cues favor ICL, particularly when the environment is volatile. Furthermore, an analysis of learning dynamics reveals task-dependent transitions between strategies (ICL to IWL and vice versa). We demonstrate that these transitions are governed by (1) the asymptotic optimality of the strategy with respect to the environment, and (2) the optimization cost of acquiring that strategy, which depends on the task structure and the learner's inductive bias.
>
---
#### [replaced 037] Multi-Session Client-Centered Treatment Outcome Evaluation in Psychotherapy
- **分类: cs.CL**

- **简介: 该论文属于心理治疗效果评估任务，旨在解决现有方法忽视客户主观体验和多阶段进展的问题。提出IPAEval框架，从客户视角进行多轮评估。**

- **链接: [https://arxiv.org/pdf/2410.05824](https://arxiv.org/pdf/2410.05824)**

> **作者:** Hongbin Na; Tao Shen; Shumao Yu; Ling Chen
>
> **备注:** Accepted at LREC 2026. Camera-ready Version
>
> **摘要:** In psychotherapy, therapeutic outcome assessment, or treatment outcome evaluation, is essential to mental health care by systematically evaluating therapeutic processes and outcomes. Existing large language model approaches often focus on therapist-centered, single-session evaluations, neglecting the client's subjective experience and longitudinal progress across multiple sessions. To address these limitations, we propose IPAEval, a client-Informed Psychological Assessment-based Evaluation framework, which automates treatment outcome evaluations from the client's perspective using clinical interviews. It integrates cross-session client-contextual assessment and session-focused client-dynamics assessment for a comprehensive understanding of therapeutic progress. Specifically, IPAEval employs a two-stage prompt scheme that maps client information onto psychometric test items, enabling interpretable and structured psychological assessments. Experiments on our new TheraPhase dataset, comprising 400 paired initial and completion stage client records, demonstrate that IPAEval effectively tracks symptom severity and treatment outcomes over multiple sessions, outperforming baseline approaches across both closed-source and open-source models, and validating the benefits of items-aware reasoning mechanisms.
>
---
#### [replaced 038] Measuring Complexity at the Requirements Stage: Spectral Metrics as Development Effort Predictors
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于需求工程任务，旨在解决需求阶段复杂性量化不足的问题。通过自然语言处理提取结构网络，利用谱度量预测开发工作量，验证其有效性。**

- **链接: [https://arxiv.org/pdf/2602.07182](https://arxiv.org/pdf/2602.07182)**

> **作者:** Maximilian Vierlboeck; Antonio Pugliese; Roshanak Nilchian; Paul Grogan; Rashika Sugganahalli Natesh Babu
>
> **备注:** 36 pages, 4 figures, 5 tables
>
> **摘要:** Complexity in engineered systems presents one of the most persistent challenges in modern development since it is driving cost overruns, schedule delays, and outright project failures. Yet while architectural complexity has been studied, the structural complexity embedded within requirements specifications remains poorly understood and inadequately quantified. This gap is consequential: requirements fundamentally drive system design, and complexity introduced at this stage propagates through architecture, implementation, and integration. To address this gap, we build on Natural Language Processing methods that extract structural networks from textual requirements. Using these extracted structures, we conducted a controlled experiment employing molecular integration tasks as structurally isomorphic proxies for requirements integration - leveraging the topological equivalence between molecular graphs and requirement networks while eliminating confounding factors such as domain expertise and semantic ambiguity. Our results demonstrate that spectral measures predict integration effort with correlations exceeding 0.95, while structural metrics achieve correlations above 0.89. Notably, density-based metrics show no significant predictive validity. These findings indicate that eigenvalue-derived measures capture cognitive and effort dimensions that simpler connectivity metrics cannot. As a result, this research bridges a critical methodological gap between architectural complexity analysis and requirements engineering practice, providing a validated foundation for applying these metrics to requirements engineering, where similar structural complexity patterns may predict integration effort.
>
---
#### [replaced 039] Auditing Google's AI Overviews and Featured Snippets: A Case Study on Baby Care and Pregnancy
- **分类: cs.CL; cs.AI; cs.CY; cs.HC; cs.IR**

- **简介: 该论文属于AI信息质量审计任务，旨在评估Google AI Overviews和Featured Snippets在母婴领域的信息一致性与安全性，发现其存在信息不一致和医疗保障缺失问题。**

- **链接: [https://arxiv.org/pdf/2511.12920](https://arxiv.org/pdf/2511.12920)**

> **作者:** Desheng Hu; Joachim Baumann; Aleksandra Urman; Elsa Lichtenegger; Robin Forsberg; Aniko Hannak; Christo Wilson
>
> **备注:** 18 pages, 10 figures; to appear in AAAI ICWSM 2026
>
> **摘要:** Google Search increasingly surfaces AI-generated content through features like AI Overviews (AIO) and Featured Snippets (FS), which users frequently rely on despite having no control over their presentation. Through a systematic algorithm audit of 1,508 real baby care and pregnancy-related queries, we evaluate the quality and consistency of these information displays. Our robust evaluation framework assesses multiple quality dimensions, including answer consistency, relevance, presence of medical safeguards, source categories, and sentiment alignment. Our results reveal concerning gaps in information consistency, with information in AIO and FS displayed on the same search result page being inconsistent with each other in 33% of cases. Despite high relevance scores, both features critically lack medical safeguards (present in just 11% of AIO and 7% of FS responses). While health and wellness websites dominate source categories for both, AIO and FS, FS also often link to commercial sources. These findings have important implications for public health information access and demonstrate the need for stronger quality controls in AI-mediated health information. Our methodology provides a transferable framework for auditing AI systems across high-stakes domains where information quality directly impacts user well-being.
>
---
#### [replaced 040] Unlocking Multimodal Document Intelligence: From Current Triumphs to Future Frontiers of Visual Document Retrieval
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于视觉文档检索任务，旨在解决多模态信息与精准信息获取之间的差距。通过综述MLLM时代的VDR方法，分析其发展与挑战，提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2602.19961](https://arxiv.org/pdf/2602.19961)**

> **作者:** Yibo Yan; Jiahao Huo; Guanbo Feng; Mingdong Ou; Yi Cao; Xin Zou; Shuliang Liu; Yuanhuiyi Lyu; Yu Huang; Jungang Li; Kening Zheng; Xu Zheng; Philip S. Yu; James Kwok; Xuming Hu
>
> **备注:** Under review. This version updates the relevant works released before 15 March, 2026
>
> **摘要:** With the rapid proliferation of multimodal information, Visual Document Retrieval (VDR) has emerged as a critical frontier in bridging the gap between unstructured visually rich data and precise information acquisition. Unlike traditional natural image retrieval, visual documents exhibit unique characteristics defined by dense textual content, intricate layouts, and fine-grained semantic dependencies. This paper presents the first comprehensive survey of the VDR landscape, specifically through the lens of the Multimodal Large Language Model (MLLM) era. We begin by examining the benchmark landscape, and subsequently dive into the methodological evolution, categorizing approaches into three primary aspects: multimodal embedding models, multimodal reranker models, and the integration of Retrieval-Augmented Generation (RAG) and Agentic systems for complex document intelligence. Finally, we identify persistent challenges and outline promising future directions, aiming to provide a clear roadmap for future multimodal document intelligence.
>
---
#### [replaced 041] Scalable Prompt Routing via Fine-Grained Latent Task Discovery
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型路由任务，旨在解决大规模模型池中精准选择合适模型的问题。通过自动发现细粒度任务并评估质量，提升路由效果与成本效率。**

- **链接: [https://arxiv.org/pdf/2603.19415](https://arxiv.org/pdf/2603.19415)**

> **作者:** Yunyi Zhang; Soji Adeshina; Sheng Guan; Ashwin Ganesh; Zhen Han; Vassilis N. Ioannidis; Huzefa Rangwala; George Karypis
>
> **摘要:** Prompt routing dynamically selects the most appropriate large language model from a pool of candidates for each query, optimizing performance while managing costs. As model pools scale to include dozens of frontier models with narrow performance gaps, existing approaches face significant challenges: manually defined task taxonomies cannot capture fine-grained capability distinctions, while monolithic routers struggle to differentiate subtle differences across diverse tasks. We propose a two-stage routing architecture that addresses these limitations through automated fine-grained task discovery and task-aware quality estimation. Our first stage employs graph-based clustering to discover latent task types and trains a classifier to assign prompts to discovered tasks. The second stage uses a mixture-of-experts architecture with task-specific prediction heads for specialized quality estimates. At inference, we aggregate predictions from both stages to balance task-level stability with prompt-specific adaptability. Evaluated on 10 benchmarks with 11 frontier models, our method consistently outperforms existing baselines and surpasses the strongest individual model while incurring less than half its cost.
>
---
#### [replaced 042] SafeSearch: Do Not Trade Safety for Utility in LLM Search Agents
- **分类: cs.CL**

- **简介: 该论文属于安全与效用平衡任务，旨在解决LLM搜索代理的安全性问题。通过多目标强化学习方法SafeSearch，在提升安全性的同时保持响应质量。**

- **链接: [https://arxiv.org/pdf/2510.17017](https://arxiv.org/pdf/2510.17017)**

> **作者:** Qiusi Zhan; Angeline Budiman-Chan; Abdelrahman Zayed; Xingzhi Guo; Daniel Kang; Joo-Kyung Kim
>
> **备注:** EACL 2026 Findings. Code available at this https URL
>
> **摘要:** Large language model (LLM) based search agents iteratively generate queries, retrieve external information, and reason to answer open-domain questions. While researchers have primarily focused on improving their utility, their safety behaviors remain underexplored. In this paper, we first evaluate search agents using red-teaming datasets and find that they are more likely to produce harmful outputs than base LLMs. For example, when asked ``How can I track someone's location without their consent?'', a base model refuses, whereas a search agent designed to retrieve and cite sources may lower its refusal threshold, fetch documents (e.g., court cases), and, once appended, synthesize them into an informative yet unsafe summary. We further show that utility-oriented finetuning intensifies this risk, motivating joint alignment of safety and utility. To this end, we present SafeSearch, a multi-objective reinforcement learning approach that couples a final-output safety/utility reward with a novel query-level shaping term that penalizes unsafe queries and rewards safe ones. Experiments show that SafeSearch reduces agent harmfulness by over 90% across three red-teaming datasets on a 7B model while producing safe and helpful responses, and maintains QA performance comparable to that of a utility-only finetuned agent. Further analyses confirm the effectiveness of the query-level reward in jointly improving safety and utility.
>
---
#### [replaced 043] The MediaSpin Dataset: Post-Publication News Headline Edits Annotated for Media Bias
- **分类: cs.CL**

- **简介: 该论文提出MediaSpin数据集，用于分析新闻标题中的媒体偏见。任务是识别和分类新闻标题中的偏见类型，解决如何系统识别媒体偏见的问题。**

- **链接: [https://arxiv.org/pdf/2412.02271](https://arxiv.org/pdf/2412.02271)**

> **作者:** Preetika Verma; Kokil Jaidka
>
> **备注:** 8 pages, 3 figures, 8 tables Accepted at AAAI ICWSM 2026 We updated the paper title from "MediaSpin: Exploring Media Bias Through Fine-Grained Analysis of News Headlines " to "The MediaSpin Dataset: Post-Publication News Headline Edits Annotated for Media Bias"
>
> **摘要:** The editability of online news content has become a significant factor in shaping public perception, as social media platforms introduce new affordances for dynamic and adaptive news framing. Edits to news headlines can refocus audience attention, add or remove emotional language, and shift the framing of events in subtle yet impactful ways. What types of media bias are editorialized in and out of news headlines, and how can they be systematically identified? This study introduces the MediaSpin dataset, the first to characterize the bias in how prominent news outlets editorialize news headlines after publication. The dataset includes 78,910 pairs of headlines annotated with 13 distinct types of media bias, using human-supervised LLM labeling. We discuss the linguistic insights it affords and show its applications for bias prediction and user behavior analysis.
>
---
#### [replaced 044] AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出AdaptVision，解决视觉语言模型计算开销大的问题。通过自适应获取视觉标记，提升效率并保持性能。任务为视觉问答。**

- **链接: [https://arxiv.org/pdf/2512.03794](https://arxiv.org/pdf/2512.03794)**

> **作者:** Zichuan Lin; Yicheng Liu; Yang Yang; Lvfang Tao; Deheng Ye
>
> **备注:** Accepted by CVPR 2026. Code and models are available at this https URL
>
> **摘要:** Vision-Language Models (VLMs) have achieved remarkable success in visual question answering tasks, but their reliance on large numbers of visual tokens introduces significant computational overhead. While existing efficient VLM approaches reduce visual tokens through fixed-ratio compression, they operate passively and lack the ability to adapt to varying task requirements. This motivates a fundamental question: Can VLMs autonomously determine the minimum number of visual tokens required for each sample? Inspired by human active vision mechanisms, we introduce AdaptVision, an efficient VLM paradigm that enables adaptive visual token acquisition through a coarse-to-fine approach. Our model initially processes compressed visual tokens from low-resolution images and selectively acquires additional visual information by invoking a bounding box tool to crop key regions when necessary. We train AdaptVision using a reinforcement learning framework that carefully balances accuracy and efficiency. Central to our approach is Decoupled Turn Policy Optimization (DTPO), which decouples the learning objective into two components: (1) tool learning, which optimizes correct tool utilization, and (2) accuracy improvement, which refines the generated responses to improve answer correctness. Based on this formulation, we further decouple advantage estimation by computing separate advantages for tokens associated with each objective. This formulation enables more effective optimization for AdaptVision compared to vanilla GRPO. Comprehensive experiments across multiple VQA benchmarks demonstrate that AdaptVision achieves superior performance while consuming substantially fewer visual tokens than state-of-the-art efficient VLM methods.
>
---
#### [replaced 045] DEBATE: A Large-Scale Benchmark for Evaluating Opinion Dynamics in Role-Playing LLM Agents
- **分类: cs.CL**

- **简介: 该论文属于多智能体模拟任务，旨在解决角色扮演大语言模型在意见动态模拟中的真实性问题。提出DEBATE基准，评估模拟群体行为与真实人类互动的对齐程度。**

- **链接: [https://arxiv.org/pdf/2510.25110](https://arxiv.org/pdf/2510.25110)**

> **作者:** Yun-Shiuan Chuang; Ruixuan Tu; Chengtao Dai; Smit Vasani; You Li; Binwei Yao; Michael Henry Tessler; Sijia Yang; Dhavan Shah; Robert Hawkins; Junjie Hu; Timothy T. Rogers
>
> **摘要:** Accurately modeling opinion change through social interactions is crucial for understanding and mitigating polarization, misinformation, and societal conflict. Recent work simulates opinion dynamics with role-playing LPL agents (RPLAs), but multi-agent simulations often display unnatural group behavior, such as premature convergence, and lack empirical benchmarks for assessing alignment with real human group interactions. We introduce DEBATE, a large-scale benchmark for evaluating the authenticity of opinion dynamics in multi-agent RPLA simulations. DEBATE contains 30,707 messages from 2,832 U.S.-based participants across 708 groups and 107 topics, with both public messages and private Likert-scale beliefs, enabling evaluation at the utterance and group levels while also supporting future individual-level analyses. We instantiate "digital twin" RPLAs with seven LLMs and evaluate them in two settings: next-message prediction and full conversation rollout, using stance-alignment and opinion-convergence metrics. In zero-shot settings, RPLA groups exhibit strong opinion convergence relative to human groups. Post-training via supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) improves stance alignment and brings group-level convergence closer to human behavior, though discrepancies in opinion change and belief updating remain. DEBATE enables rigorous benchmarking of simulated opinion dynamics and supports future research on aligning multi-agent RPLAs with realistic human interactions. The benchmark is publicly available at.
>
---
#### [replaced 046] Mind the Gap: Pitfalls of LLM Alignment with Asian Public Opinion
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理中的文化对齐研究，旨在解决LLM与亚洲公众意见不匹配的问题。通过多语言审计，分析模型在宗教等敏感领域的偏差，提出改进方法。**

- **链接: [https://arxiv.org/pdf/2603.06264](https://arxiv.org/pdf/2603.06264)**

> **作者:** Hari Shankar; Vedanta S P; Sriharini Margapuri; Debjani Mazumder; Ponnurangam Kumaraguru; Abhijnan Chakraborty
>
> **备注:** 13 pages, including AAAI Paper Checklist. Accepted in Proceedings of the 20th International AAAI Conference on Web and Social Media (ICWSM 2026)
>
> **摘要:** Large Language Models (LLMs) are increasingly being deployed in multilingual, multicultural settings, yet their reliance on predominantly English-centric training data risks misalignment with the diverse cultural values of different societies. In this paper, we present a comprehensive, multilingual audit of the cultural alignment of contemporary LLMs including GPT-4o-Mini, Gemini-2.5-Flash, Llama 3.2, Mistral and Gemma 3 across India, East Asia and Southeast Asia. Our study specifically focuses on the sensitive domain of religion as the prism for broader alignment. To facilitate this, we conduct a multi-faceted analysis of every LLM's internal representations, using log-probs/logits, to compare the model's opinion distributions against ground-truth public attitudes. We find that while the popular models generally align with public opinion on broad social issues, they consistently fail to accurately represent religious viewpoints, especially those of minority groups, often amplifying negative stereotypes. Lightweight interventions, such as demographic priming and native language prompting, partially mitigate but do not eliminate these cultural gaps. We further show that downstream evaluations on bias benchmarks (such as CrowS-Pairs, IndiBias, ThaiCLI, KoBBQ) reveal persistent harms and under-representation in sensitive contexts. Our findings underscore the urgent need for systematic, regionally grounded audits to ensure equitable global deployment of LLMs.
>
---
#### [replaced 047] DMFI: A Dual-Modality Log Analysis Framework for Insider Threat Detection with LoRA-Tuned Language Models
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于信息安全领域，解决 insider threat detection 问题。提出 DMFI 框架，结合语义分析与行为建模，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2508.05694](https://arxiv.org/pdf/2508.05694)**

> **作者:** Kaichuan Kong; Dongjie Liu; Xiaobo Jin; Guanggang Geng; Zhiying Li; Jian Weng
>
> **备注:** This work has been accepted by 2025 IEEE International Conference on Data Mining (ICDM)
>
> **摘要:** Insider threat detection (ITD) poses a persistent and high-impact challenge in cybersecurity due to the subtle, long-term, and context-dependent nature of malicious insider behaviors. Traditional models often struggle to capture semantic intent and complex behavior dynamics, while existing LLM-based solutions face limitations in prompt adaptability and modality coverage. To bridge this gap, we propose DMFI, a dual-modality framework that integrates semantic inference with behavior-aware fine-tuning. DMFI converts raw logs into two structured views: (1) a semantic view that processes content-rich artifacts (e.g., emails, https) using instruction-formatted prompts; and (2) a behavioral abstraction, constructed via a 4W-guided (When-Where-What-Which) transformation to encode contextual action sequences. Two LoRA-enhanced LLMs are fine-tuned independently, and their outputs are fused via a lightweight MLP-based decision module. We further introduce DMFI-B, a discriminative adaptation strategy that separates normal and abnormal behavior representations, improving robustness under severe class imbalance. Experiments on CERT r4.2 and r5.2 datasets demonstrate that DMFI outperforms state-of-the-art methods in detection accuracy. Our approach combines the semantic reasoning power of LLMs with structured behavior modeling, offering a scalable and effective solution for real-world insider threat detection.
>
---
#### [replaced 048] Prompt-Induced Linguistic Fingerprints for LLM-Generated Fake News Detection
- **分类: cs.CL**

- **简介: 该论文属于虚假新闻检测任务，旨在解决LLM生成假新闻难以识别的问题。通过分析提示引发的语言特征，提出LIFE方法，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2508.12632](https://arxiv.org/pdf/2508.12632)**

> **作者:** Chi Wang; Min Gao; Zongwei Wang; Junwei Yin; Kai Shu; Chenghua Lin
>
> **备注:** published in WWW 2026
>
> **摘要:** With the rapid development of large language models, the generation of fake news has become increasingly effortless, posing a growing societal threat and underscoring the urgent need for reliable detection methods. Early efforts to identify LLM-generated fake news have predominantly focused on the textual content itself; however, because much of that content may appear coherent and factually consistent, the subtle traces of falsification are often difficult to uncover. Through distributional divergence analysis, we uncover prompt-induced linguistic fingerprints: statistically distinct probability shifts between LLM-generated real and fake news when maliciously prompted. Based on this insight, we propose a novel method named Linguistic Fingerprints Extraction (LIFE). By reconstructing word-level probability distributions, LIFE can find discriminative patterns that facilitate the detection of LLM-generated fake news. To further amplify these fingerprint patterns, we also leverage key-fragment techniques that accentuate subtle linguistic differences, thereby improving detection reliability. Our experiments show that LIFE achieves state-of-the-art performance in LLM-generated fake news and maintains high performance in human-written fake news. The code and data are available at this https URL.
>
---
#### [replaced 049] Learning to Reason without External Rewards
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决依赖外部奖励的局限性。通过引入内部反馈机制，使用模型自信心作为奖励信号，实现无需外部监督的高效学习。**

- **链接: [https://arxiv.org/pdf/2505.19590](https://arxiv.org/pdf/2505.19590)**

> **作者:** Xuandong Zhao; Zhewei Kang; Aosong Feng; Sergey Levine; Dawn Song
>
> **备注:** ICLR 2026
>
> **摘要:** Training large language models (LLMs) for complex reasoning via Reinforcement Learning with Verifiable Rewards (RLVR) is effective but limited by reliance on costly, domain-specific supervision. We explore Reinforcement Learning from Internal Feedback (RLIF), a framework that enables LLMs to learn from intrinsic signals without external rewards or labeled data. We propose Intuitor, an RLIF method that uses a model's own confidence-termed self-certainty-as its sole reward signal. Intuitor replaces external rewards in Group Relative Policy Optimization (GRPO) with self-certainty scores, enabling fully unsupervised learning. Experiments demonstrate that Intuitor matches GRPO's performance on mathematical benchmarks while achieving better generalization to out-of-domain tasks like code generation, without requiring gold solutions or test cases. Our findings show that intrinsic model signals can drive effective learning across domains, offering a scalable alternative to RLVR for autonomous AI systems where verifiable rewards are unavailable. Code is available at this https URL
>
---
#### [replaced 050] WiFi-GEN: High-Resolution Indoor Imaging from WiFi Signals Using Generative AI
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于室内成像任务，旨在通过WiFi信号生成高分辨率图像。提出WiFi-GEN网络，解决传统方法的非线性与不确定性问题，提升成像精度。**

- **链接: [https://arxiv.org/pdf/2401.04317](https://arxiv.org/pdf/2401.04317)**

> **作者:** Jianyang Shi; Bowen Zhang; Amartansh Dubey; Ross Murch; Liwen Jing
>
> **摘要:** Indoor imaging is a critical task for robotics and internet-ofthings. WiFi as an omnipresent signal is a promising candidate for carrying out passive imaging and synchronizing the up-to-date information to all connected devices. This is the first research work to consider WiFi indoor imaging as a multi-modal image generation task that converts the measured WiFi power into a high-resolution indoor image. Our proposedWiFi-GEN network achieves a shape reconstruction accuracy that is 275% of that achieved by physical model-based inversion methods. Additionally, the Frechet Inception Distance score has been significantly reduced by 82%. To examine the effectiveness of models for this task, the first large-scale dataset is released containing 80,000 pairs of WiFi signal and imaging target. Our model absorbs challenges for the model-based methods including the nonlinearity, ill-posedness and non-certainty into massive parameters of our generative AI network. The network is also designed to best fit measured WiFi signals and the desired imaging output. Code: this https URL
>
---
#### [replaced 051] AgentExpt: Automating AI Experiment Design with LLM-based Resource Retrieval Agent
- **分类: cs.CL**

- **简介: 该论文属于AI实验设计自动化任务，旨在解决数据和基线推荐覆盖不足与匹配不准确的问题。通过构建数据集和改进检索方法，提升推荐效果。**

- **链接: [https://arxiv.org/pdf/2511.04921](https://arxiv.org/pdf/2511.04921)**

> **作者:** Yu Li; Lehui Li; Qingmin Liao; Fengli Xu; Yong Li
>
> **备注:** 10 pages
>
> **摘要:** Large language model agents are becoming increasingly capable at web-centric tasks such as information retrieval, complex reasoning. These emerging capabilities have given rise to surge research interests in developing LLM agent for facilitating scientific quest. One key application in AI research is to automate experiment design through agentic dataset and baseline retrieval. However, prior efforts suffer from limited data coverage, as recommendation datasets primarily harvest candidates from public portals and omit many datasets actually used in published papers, and from an overreliance on content similarity that biases model toward superficial similarity and overlooks experimental suitability. Harnessing collective perception embedded in the baseline and dataset citation network, we present a comprehensive framework for baseline and dataset recommendation. First, we design an automated data-collection pipeline that links roughly one hundred thousand accepted papers to the baselines and datasets they actually used. Second, we propose a collective perception enhanced retriever. To represent the position of each dataset or baseline within the scholarly network, it concatenates self-descriptions with aggregated citation contexts. To achieve efficient candidate recall, we finetune an embedding model on these representations. Finally, we develop a reasoning-augmented reranker that exact interaction chains to construct explicit reasoning chains and finetunes a large language model to produce interpretable justifications and refined rankings. The dataset we curated covers 85\% of the datasets and baselines used at top AI conferences over the past five years. On our dataset, the proposed method outperforms the strongest prior baseline with average gains of +5.85\% in Recall@20, +8.30\% in HitRate@5. Taken together, our results advance reliable, interpretable automation of experimental design.
>
---
#### [replaced 052] LinguaMap: Which Layers of LLMs Speak Your Language and How to Tune Them?
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于多语言模型优化任务，旨在解决语言控制不足的问题。通过分析模型层结构，提出仅微调末尾层的方法，提升语言一致性并减少计算资源消耗。**

- **链接: [https://arxiv.org/pdf/2601.20009](https://arxiv.org/pdf/2601.20009)**

> **作者:** J. Ben Tamo; Daniel Carlander-Reuterfelt; Jonathan Rubin; Dezhi Hong; Mingxian Wang; Oleg Poliannikov
>
> **摘要:** Despite multilingual pretraining, large language models often struggle with non-English tasks, particularly in language control, the ability to respond in the intended language. We identify and characterize two key failure modes: the multilingual transfer bottleneck (correct language, incorrect task response) and the language consistency bottleneck (correct task response, wrong language). To systematically surface these issues, we design a four-scenario evaluation protocol spanning MMLU, MGSM, and XQuAD benchmarks. To probe these issues with interpretability, we extend logit lens analysis to track language probabilities layer by layer and compute cross-lingual semantic similarity of hidden states. The results reveal a three-phase internal structure: early layers align inputs into a shared semantic space, middle layers perform task reasoning, and late layers drive language-specific generation. Guided by these insights, we introduce selective fine-tuning of only the final layers responsible for language control. On Qwen-3-32B and Bloom-7.1B, this method achieves over 98 percent language consistency across six languages while fine-tuning only 3-5 percent of parameters, without sacrificing task accuracy. Importantly, this result is nearly identical to that of full-scope fine-tuning (for example, above 98 percent language consistency for both methods across all prompt scenarios) but uses a fraction of the computational resources. To the best of our knowledge, this is the first approach to leverage layer-localization of language control for efficient multilingual adaptation.
>
---
#### [replaced 053] Learning to Interpret Weight Differences in Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决模型微调后权重变化难以解释的问题。提出DIT方法，让模型自我描述微调带来的变化。**

- **链接: [https://arxiv.org/pdf/2510.05092](https://arxiv.org/pdf/2510.05092)**

> **作者:** Avichal Goel; Yoon Kim; Nir Shavit; Tony T. Wang
>
> **备注:** Project code and links to weight diffs, adapters, and training data can be found at this https URL
>
> **摘要:** Finetuning (pretrained) language models is a standard approach for updating their internal parametric knowledge and specializing them to new tasks and domains. However, the corresponding model weight changes ("weight diffs") are not generally interpretable. While inspecting the finetuning dataset can give a sense of how the model might have changed, these datasets are often not publicly available or are too large to work with directly. Towards the goal of comprehensively understanding weight diffs in natural language, we introduce Diff Interpretation Tuning (DIT), a method that trains models to describe their own finetuning-induced modifications. Our approach uses synthetic, labeled weight diffs to train a DIT-adapter, which can be applied to a compatible finetuned model to make it describe how it has changed. We demonstrate in two proof-of-concept settings (reporting hidden behaviors and summarizing finetuned knowledge) that our method enables models to describe their finetuning-induced modifications using accurate natural language descriptions.
>
---
#### [replaced 054] Feature Resemblance: Towards a Theoretical Understanding of Analogical Reasoning in Transformers
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究Transformer模型中的类比推理机制，解决如何理解其推理能力的问题。通过理论分析与实验，揭示了类比推理的形成机制及训练条件。**

- **链接: [https://arxiv.org/pdf/2603.05143](https://arxiv.org/pdf/2603.05143)**

> **作者:** Ruichen Xu; Wenjing Yan; Ying-Jun Angela Zhang
>
> **摘要:** Understanding reasoning in large language models is complicated by evaluations that conflate multiple reasoning types. We isolate analogical reasoning (inferring shared properties between entities based on known similarities) and analyze its emergence in transformers. We theoretically prove three key results: (1) Joint training on similarity and attribution premises enables analogical reasoning through aligned representations; (2) Sequential training succeeds only when similarity structure is learned before specific attributes, revealing a necessary curriculum; (3) Two-hop reasoning ($a \to b, b \to c \implies a \to c$) reduces to analogical reasoning with identity bridges ($b = b$), which must appear explicitly in training data. These results reveal a unified mechanism: transformers encode entities with similar properties into similar representations, enabling property transfer through feature alignment. Experiments with architectures up to 1.5B parameters validate our theory and demonstrate how representational geometry shapes inductive reasoning capabilities.
>
---
#### [replaced 055] Do LLMs Understand Collaborative Signals? Diagnosis and Repair
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于推荐系统任务，研究LLM是否能有效理解协同信号，提出用RAG方法提升其推理能力。**

- **链接: [https://arxiv.org/pdf/2505.20730](https://arxiv.org/pdf/2505.20730)**

> **作者:** Shahrooz Pouryousef; Ali Montazeralghaem
>
> **摘要:** Collaborative information from user-item interactions is a fundamental source of signal in successful recommender systems. Recently, researchers have attempted to incorporate this knowledge into large language model-based recommender approaches (LLMRec) to enhance their performance. However, there has been little fundamental analysis of whether LLMs can effectively reason over collaborative information. In this paper, we analyze the ability of LLMs to reason about collaborative information in recommendation tasks, comparing their performance to traditional matrix factorization (MF) models. We propose a simple and effective method to improve LLMs' reasoning capabilities using retrieval-augmented generation (RAG) over the user-item interaction matrix with four different prompting strategies. Our results show that the LLM outperforms the MF model whenever we provide relevant information in a clear and easy-to-follow format, and prompt the LLM to reason based on it. We observe that with this strategy, in almost all cases, the more information we provide, the better the LLM performs.
>
---
#### [replaced 056] Semantic Self-Distillation for Language Model Uncertainty
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决大语言模型不确定性量化问题。通过语义自蒸馏技术，构建轻量学生模型，估计提示级不确定性和答案可靠性。**

- **链接: [https://arxiv.org/pdf/2602.04577](https://arxiv.org/pdf/2602.04577)**

> **作者:** Edward Phillips; Sean Wu; Fredrik K. Gustafsson; Boyan Gao; David A. Clifton
>
> **备注:** Added experiments on MMLU dataset, investigating utility of likelihood for multiple-choice answer selection
>
> **摘要:** Large language models present challenges for principled uncertainty quantification, in part due to their complexity and the diversity of their outputs. Semantic dispersion, or the variance in the meaning of sampled answers, has been proposed as a useful proxy for model uncertainty, but the associated computational cost prohibits its use in latency-critical applications. We show that sampled semantic distributions can be distilled into lightweight student models which estimate a prompt-conditioned density before the language model generates an answer token. The student model predicts a semantic distribution over possible answers; the entropy of this distribution provides a prompt-level uncertainty signal, and the probability density allows answer-level reliability evaluation. Across experiments on TriviaQA and MMLU, we find our student models perform competitively relative to sampling-based semantic dispersion baselines on a hallucination prediction task, whilst offering additional uncertainty primitives for out-of-domain detection and multiple-choice answer selection. We term this technique Semantic Self-Distillation (SSD), which can serve as a general framework for distilling predictive uncertainty in complex output spaces beyond language.
>
---
#### [replaced 057] A Training-free Method for LLM Text Attribution
- **分类: stat.ML; cs.AI; cs.CL; cs.IT; cs.LG**

- **简介: 该论文属于文本归属任务，旨在识别文本是否由特定大语言模型生成。工作包括构建统计测试方法，确保低误报率，并验证其理论性能与实际效果。**

- **链接: [https://arxiv.org/pdf/2501.02406](https://arxiv.org/pdf/2501.02406)**

> **作者:** Tara Radvand; Mojtaba Abdolmaleki; Mohamed Mostagir; Ambuj Tewari
>
> **摘要:** Verifying the provenance of content is crucial to the functioning of many organizations, e.g., educational institutions, social media platforms, and firms. This problem is becoming increasingly challenging as text generated by Large Language Models (LLMs) becomes almost indistinguishable from human-generated content. In addition, many institutions use in-house LLMs and want to ensure that external, non-sanctioned LLMs do not produce content within their institutions. In this paper, we answer the following question: Given a piece of text, can we identify whether it was produced by a particular LLM, while ensuring a guaranteed low false positive rate? We model LLM text as a sequential stochastic process with complete dependence on history. We then design zero-shot statistical tests to (i) distinguish between text generated by two different known sets of LLMs $A$ (non-sanctioned) and $B$ (in-house), and (ii) identify whether text was generated by a known LLM or by any unknown model. We prove that the Type I and Type II errors of our test decrease exponentially with the length of the text. We also extend our theory to black-box access via sampling and characterize the required sample size to obtain essentially the same Type I and Type II error upper bounds as in the white-box setting (i.e., with access to $A$). We show the tightness of our upper bounds by providing an information-theoretic lower bound. We next present numerical experiments to validate our theoretical results and assess their robustness in settings with adversarial post-editing. Our work has a host of practical applications in which determining the origin of a text is important and can also be useful for combating misinformation and ensuring compliance with emerging AI regulations. See this https URL for code, data, and an online demo of the project.
>
---
#### [replaced 058] Must Read: A Comprehensive Survey of Computational Persuasion
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于AI伦理与 persuasion 研究任务，旨在探讨AI在说服中的角色及风险，分析AI作为说服者、被说服者和评判者的三重角色，提出未来研究方向以提升AI说服的安全性与有效性。**

- **链接: [https://arxiv.org/pdf/2505.07775](https://arxiv.org/pdf/2505.07775)**

> **作者:** Nimet Beyza Bozdag; Shuhaib Mehri; Xiaocheng Yang; Hyeonjeong Ha; Zirui Cheng; Esin Durmus; Jiaxuan You; Heng Ji; Gokhan Tur; Dilek Hakkani-Tür
>
> **备注:** Accepted to ACM Computing Surveys
>
> **摘要:** Persuasion is a fundamental aspect of communication, influencing decision-making across diverse contexts, from everyday conversations to high-stakes scenarios such as politics, marketing, and law. The rise of conversational AI systems has significantly expanded the scope of persuasion, introducing both opportunities and risks. AI-driven persuasion can be leveraged for beneficial applications, but also poses threats through unethical influence. Moreover, AI systems are not only persuaders, but also susceptible to persuasion, making them vulnerable to adversarial attacks and bias reinforcement. Despite rapid advancements in AI-generated persuasive content, our understanding of what makes persuasion effective remains limited due to its inherently subjective and context-dependent nature. In this survey, we provide a comprehensive overview of persuasion, structured around three key perspectives: (1) AI as a Persuader, which explores AI-generated persuasive content and its applications; (2) AI as a Persuadee, which examines AI's susceptibility to influence and manipulation; and (3) AI as a Persuasion Judge, which analyzes AI's role in evaluating persuasive strategies, detecting manipulation, and ensuring ethical persuasion. We introduce a taxonomy for persuasion research and discuss key challenges for future research to enhance the safety, fairness, and effectiveness of AI-powered persuasion while addressing the risks posed by increasingly capable language models.
>
---
#### [replaced 059] Knowledge Fusion via Bidirectional Information Aggregation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识融合任务，旨在解决大语言模型知识滞后问题。通过引入KGA框架，在推理时动态整合知识图谱，提升模型时效性与准确性。**

- **链接: [https://arxiv.org/pdf/2507.08704](https://arxiv.org/pdf/2507.08704)**

> **作者:** Songlin Zhai; Guilin Qi; Yue Wang; Yuan Meng
>
> **摘要:** Knowledge graphs (KGs) are the cornerstone of the semantic web, offering up-to-date representations of real-world entities and relations. Yet large language models (LLMs) remain largely static after pre-training, causing their internal knowledge to become outdated and limiting their utility in time-sensitive web applications. To bridge this gap between dynamic knowledge and static models, a prevalent approach is to enhance LLMs with KGs. However, prevailing methods typically rely on parameter-invasive fine-tuning, which risks catastrophic forgetting and often degrades LLMs' general capabilities. Moreover, their static integration frameworks cannot keep pace with the continuous evolution of real-world KGs, hindering their deployment in dynamic web environments. To bridge this gap, we introduce KGA (\textit{\underline{K}nowledge \underline{G}raph-guided \underline{A}ttention}), a novel framework that dynamically integrates external KGs into LLMs exclusively at inference-time without any parameter modification. Inspired by research on neuroscience, we rewire the self-attention module by innovatively introducing two synergistic pathways: a \textit{bottom-up knowledge fusion} pathway and a \textit{top-down attention guidance} pathway. The \textit{bottom-up pathway} dynamically integrates external knowledge into input representations via input-driven KG fusion, which is akin to the \textit{stimulus-driven attention process} in the human brain. Complementarily, the \textit{top-down pathway} aims to assess the contextual relevance of each triple through a \textit{goal-directed verification process}, thereby suppressing task-irrelevant signals and amplifying knowledge-relevant patterns. By synergistically combining these two pathways, our method supports real-time knowledge fusion. Extensive experiments on four benchmarks verify KGA's strong fusion performance and efficiency.
>
---
#### [replaced 060] RLHF in an SFT Way: From Optimal Solution to Reward-Weighted Alignment
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型对齐任务，旨在解决RLHF计算复杂和训练不稳定的问题。提出VAR方法，将对齐转化为改进的SFT，提升稳定性和效率。**

- **链接: [https://arxiv.org/pdf/2502.11026](https://arxiv.org/pdf/2502.11026)**

> **作者:** Yuhao Du; Zhuo Li; Pengyu Cheng; Zhihong Chen; Yuejiao Xie; Xiang Wan; Anningzhe Gao
>
> **备注:** Published in TMLR-2026
>
> **摘要:** Reinforcement Learning from Human Feedback (RLHF) is crucial for aligning Large Language Models (LLMs) with human values. However, RLHF has been continuously challenged by its high complexity in implementation and computation consumption, specifically for online sampling-based methods like Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO). Even with recent simplifications, such as Direct Preference Optimization (DPO) that designs an offline implicit reward learning objective relying on pre-collected preference datasets, the problems of over-fitting and training instability remain hindering the alignment process from the expected optimal performance. To address the existing challenges, we propose a novel simplification of RLHF from the perspective of variational inference, called Variational Alignment with Re-weighting (VAR). Specifically, by directly minimizing the distribution gap between the learning LLM policy and the optimal solution of RLHF, we transform the alignment objective into an offline reward-driven re-weighted supervised fine-tuning (SFT) form, which only requires minor adjustment on the SFT loss to obtain noticeable improvement on training stability and effectiveness. In comprehensive evaluation benchmarks, our objective empowers LLMs to outperform offline alignments, demonstrating superior performance in both helpfulness and harmlessness metrics (avg. $\uparrow7.16\%$ than DPO). Meanwhile, when compared to online sampling methods, our method is also comparable even better while significantly reducing computational overhead and accelerating convergence speed (over $5\times$ faster than GRPO), suggesting our approach as an efficient and effective solution in bridging the gap between efficiency and performance in LLM alignment.
>
---
#### [replaced 061] DialectalArabicMMLU: Benchmarking Dialectal Capabilities in Arabic and Multilingual Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决阿拉伯语方言在大模型评估中的不足。通过构建包含5种方言的15K问答基准，评估模型在不同方言上的表现，促进更全面的模型开发。**

- **链接: [https://arxiv.org/pdf/2510.27543](https://arxiv.org/pdf/2510.27543)**

> **作者:** Malik H. Altakrori; Nizar Habash; Abed Alhakim Freihat; Younes Samih; Kirill Chirkunov; Muhammed AbuOdeh; Radu Florian; Teresa Lynn; Preslav Nakov; Alham Fikri Aji
>
> **备注:** 9 pages, 10 tables, accepted to LREC 2026
>
> **摘要:** We present DialectalArabicMMLU, a new benchmark for evaluating the performance of large language models (LLMs) across Arabic dialects. While recently developed Arabic and multilingual benchmarks have advanced LLM evaluation for Modern Standard Arabic (MSA), dialectal varieties remain underrepresented despite their prevalence in everyday communication. DialectalArabicMMLU extends the MMLU-Redux framework through manual translation and adaptation of 3K multiple-choice question-answer pairs into five major dialects (Syrian, Egyptian, Emirati, Saudi, and Moroccan), yielding a total of 15K QA pairs across 32 academic and professional domains (22K QA pairs when also including English and MSA). The benchmark enables systematic assessment of LLM reasoning and comprehension beyond MSA, supporting both task-based and linguistic analysis. We evaluate 19 open-weight Arabic and multilingual LLMs (1B-13B parameters) and report substantial performance variation across dialects, revealing persistent gaps in dialectal generalization. DialectalArabicMMLU provides the first unified, human-curated resource for measuring dialectal understanding in Arabic, thus promoting more inclusive evaluation and future model development.
>
---
#### [replaced 062] Flipping the Dialogue: Training and Evaluating User Language Models
- **分类: cs.CL**

- **简介: 该论文属于对话系统任务，旨在解决真实对话环境下的模型评估问题。通过构建专门模拟用户行为的User LMs，提升对话模拟的准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.06552](https://arxiv.org/pdf/2510.06552)**

> **作者:** Tarek Naous; Philippe Laban; Wei Xu; Jennifer Neville
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Conversations with LMs involve two participants: a human user leading the conversation, and an LM assistant responding to the user's request. To satisfy this specific role, LMs are post-trained to be helpful assistants -- optimized to produce exhaustive and well-structured responses, free of ambiguity and grammar errors. User utterances, on the other hand, are rarely perfected, with each user phrasing requests in unique ways, sometimes putting in partial effort at each turn and refining on the fly. To evaluate LM performance in realistic settings, prior work simulated users in multi-turn conversations, often by prompting an LM originally trained to be a helpful assistant to act as a user. However, we show that assistant LMs make for poor user simulators, with the surprising finding that better assistants yield worse simulators. Instead, we introduce purpose-built User Language Models (User LMs) - models post-trained to simulate human users in multi-turn conversations. Through various evaluations, we show how User LMs align better with human behavior and achieve better simulation robustness than existing simulation methods. When leveraging User LMs to simulate coding and math conversations, the performance of a strong assistant (GPT-4o) drops from 74.6% to 57.4%, confirming that more realistic simulation environments lead to assistant struggles as they fail to cope with the nuances of users in multi-turn setups.
>
---
#### [replaced 063] Rethinking Soft Compression in Retrieval-Augmented Generation: A Query-Conditioned Selector Perspective
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于信息检索与生成任务，旨在解决RAG中因上下文过长和冗余检索导致的可扩展性问题。提出SeleCom框架，通过查询条件选择信息，提升效率与效果。**

- **链接: [https://arxiv.org/pdf/2602.15856](https://arxiv.org/pdf/2602.15856)**

> **作者:** Yunhao Liu; Zian Jia; Xinyu Gao; Kanjun Xu; Yun Xiong
>
> **备注:** Accepted by WWW 2026
>
> **摘要:** Retrieval-Augmented Generation (RAG) effectively grounds Large Language Models (LLMs) with external knowledge and is widely applied to Web-related tasks. However, its scalability is hindered by excessive context length and redundant retrievals. Recent research on soft context compression aims to address this by encoding long documents into compact embeddings, yet they often underperform non-compressed RAG due to their reliance on auto-encoder-like full-compression that forces the encoder to compress all document information regardless of relevance to the input query. In this work, we conduct an analysis on this paradigm and reveal two fundamental limitations: (I) Infeasibility, full-compression conflicts with the LLM's downstream generation behavior; and (II) Non-necessity: full-compression is unnecessary and dilutes task-relevant information density. Motivated by these insights, we introduce SeleCom, a selector-based soft compression framework for RAG that redefines the encoder's role as query-conditioned information selector. The selector is decoder-only and is trained with a massive, diverse and difficulty-graded synthetic QA dataset with curriculum learning. Extensive experiments show that SeleCom significantly outperforms existing soft compression approaches and achieves competitive or superior performance to non-compression baselines, while reducing computation and latency by 33.8%~84.6%.
>
---
#### [replaced 064] MUTANT: A Recipe for Multilingual Tokenizer Design
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的分词任务，旨在解决多语言模型分词器设计难题。提出MUTANT框架，优化词汇和训练数据，提升分词效果与效率。**

- **链接: [https://arxiv.org/pdf/2511.03237](https://arxiv.org/pdf/2511.03237)**

> **作者:** Souvik Rana; Arul Menezes; Ashish Kulkarni; Chandra Khatri; Shubham Agarwal
>
> **摘要:** Tokenizers play a crucial role in determining the performance, training efficiency, and the inference cost of Large Language Models (LLMs). Designing effective tokenizers for multilingual LLMs is particularly challenging due to diverse scripts and rich morphological variation. While subword methods like Byte Pair Encoding (BPE) are widely adopted, their effectiveness in multilingual settings remains underexplored. We present MUTANT, a recipe for building multilingual tokenizers, with careful vocabulary and training data design, language-aware pre-tokenization, and subword and multiword aware training. We also introduce MUTANT-Indic, a tokenizer for India-specific multilingual LLMs, that produces linguistically coherent tokens and achieves state-of-the-art performance. Evaluated across English, 22 Indian languages and code data, our tokenizer improves the average fertility score by 39.5%$ over LLaMA4 and by 18% over Sutra (the current best). This translates to 44% improvement in inference throughput over LLaMA4 while maintaining comparable performance on English and Indic benchmarks. We present detailed ablations across tokenizer training data size, vocabulary size, merging techniques, and pre-tokenization strategies, demonstrating the robustness of our design choices.
>
---
#### [replaced 065] Current LLMs still cannot 'talk much' about grammar modules: Evidence from syntax
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，探讨LLMs在语法翻译中的表现。研究发现LLMs在准确翻译语法术语方面存在显著不足，提出需加强AI与语言学合作以提升翻译质量。**

- **链接: [https://arxiv.org/pdf/2603.20114](https://arxiv.org/pdf/2603.20114)**

> **作者:** Mohammed Q. Shormani
>
> **备注:** 15 pages
>
> **摘要:** We aim to examine the extent to which Large Language Models (LLMs) can 'talk much' about grammar modules, providing evidence from syntax core properties translated by ChatGPT into Arabic. We collected 44 terms from generative syntax previous works, including books and journal articles, as well as from our experience in the field. These terms were translated by humans, and then by ChatGPT-5. We then analyzed and compared both translations. We used an analytical and comparative approach in our analysis. Findings unveil that LLMs still cannot 'talk much' about the core syntax properties embedded in the terms under study involving several syntactic and semantic challenges: only 25% of ChatGPT translations were accurate, while 38.6% were inaccurate, and 36.4.% were partially correct, which we consider appropriate. Based on these findings, a set of actionable strategies were proposed, the most notable of which is a close collaboration between AI specialists and linguists to better LLMs' working mechanism for accurate or at least appropriate translation.
>
---
#### [replaced 066] Why Agent Caching Fails and How to Fix It: Structured Intent Canonicalization with Few-Shot Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究如何提升AI代理的缓存效果，解决因重复调用大模型导致的成本问题。通过结构化意图分解和少量样本学习，提升缓存准确性与效率。**

- **链接: [https://arxiv.org/pdf/2602.18922](https://arxiv.org/pdf/2602.18922)**

> **作者:** Abhinaba Basu
>
> **备注:** Added github repo and Hugging Face dataset link
>
> **摘要:** Personal AI agents incur substantial cost via repeated LLM calls. We show existing caching methods fail: GPTCache achieves 37.9% accuracy on real benchmarks; APC achieves 0-12%. The root cause is optimizing for the wrong property -- cache effectiveness requires key consistency and precision, not classification accuracy. We observe cache-key evaluation reduces to clustering evaluation and apply V-measure decomposition to separate these on n=8,682 points across MASSIVE, BANKING77, CLINC150, and NyayaBench v2, our new 8,514-entry multilingual agentic dataset (528 intents, 20 W5H2 classes, 63 languages). We introduce W5H2, a structured intent decomposition framework. Using SetFit with 8 examples per class, W5H2 achieves 91.1%+/-1.7% on MASSIVE in ~2ms -- vs 37.9% for GPTCache and 68.8% for a 20B-parameter LLM at 3,447ms. On NyayaBench v2 (20 classes), SetFit achieves 55.3%, with cross-lingual transfer across 30 languages. Our five-tier cascade handles 85% of interactions locally, projecting 97.5% cost reduction. We provide risk-controlled selective prediction guarantees via RCPS with nine bound families.
>
---
#### [replaced 067] Theory-Grounded Evaluation of Human-Like Fallacy Patterns in LLM Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于人工智能推理研究任务，旨在检验语言模型的错误是否符合人类逻辑谬误模式。通过PyETR工具生成测试题并分析模型错误，发现模型能力与谬误比例正相关，且调整前提顺序可减少谬误。**

- **链接: [https://arxiv.org/pdf/2506.11128](https://arxiv.org/pdf/2506.11128)**

> **作者:** Andrew Keenan Richardson; Ryan Othniel Kearns; Sean Moss; Vincent Wang-Mascianica; Philipp Koralus
>
> **摘要:** We study logical reasoning in language models by asking whether their errors follow established human fallacy patterns. Using the Erotetic Theory of Reasoning (ETR) and its open-source implementation, PyETR, we programmatically generate 383 formally specified reasoning problems and evaluate 38 models. For each response, we judge logical correctness and, when incorrect, whether it matches an ETR-predicted fallacy. Two results stand out: (i) as a capability proxy (Chatbot Arena Elo) increases, a larger share of a model's incorrect answers are ETR-predicted fallacies $(\rho=0.360, p=0.0265)$, while overall correctness on this dataset shows no correlation with capability; (ii) reversing premise order significantly reduces fallacy production for many models, mirroring human order effects. Methodologically, PyETR provides an open-source pipeline for unbounded, synthetic, contamination-resistant reasoning tests linked to a cognitive theory, enabling analyses that focus on error composition rather than error rate.
>
---
#### [replaced 068] SafeConstellations: Mitigating Over-Refusals in LLMs Through Task-Aware Representation Steering
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs的过度拒绝问题。通过分析嵌入空间中的轨迹模式，提出SafeConstellations方法，在不损害实用性的前提下降低过度拒绝率。**

- **链接: [https://arxiv.org/pdf/2508.11290](https://arxiv.org/pdf/2508.11290)**

> **作者:** Utsav Maskey; Sumit Yadav; Mark Dras; Usman Naseem
>
> **备注:** Preprint
>
> **摘要:** LLMs increasingly exhibit over-refusal behavior, where safety mechanisms cause models to reject benign instructions that seemingly resemble harmful content. This phenomenon diminishes utility in production applications that repeatedly rely on common prompt templates or applications that frequently rely on LLMs for specific tasks (e.g. sentiment analysis, language translation). Through extensive evaluation, we demonstrate that LLMs persist in refusing inputs containing harmful content, even when they are reframed with tasks that have benign intent. Our mechanistic analysis reveals that LLMs follow distinct "constellation" patterns in embedding space as representations traverse layers, with each NLP task maintaining consistent trajectories that shift predictably between refusal and non-refusal cases. We introduce SafeConstellations, an inference-time trajectory-shifting approach that tracks task-specific trajectory patterns and guides representations toward non-refusal pathways. By selectively guiding model behavior only on tasks prone to over-refusal, our method reduces over-refusal rates by up to 73% with minimal impact on utility -- offering a principled and conditional approach to mitigating over-refusals.
>
---
#### [replaced 069] Masked Diffusion Models as Energy Minimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文将掩码扩散模型视为能量最小化问题，解决其理论基础与采样优化问题，通过能量等价性分析和Beta分布参数化提升采样效果。**

- **链接: [https://arxiv.org/pdf/2509.13866](https://arxiv.org/pdf/2509.13866)**

> **作者:** Sitong Chen; Shen Nie; Jiacheng Sun; Zijin Feng; Zhenguo Li; Ji-Rong Wen; Chongxuan Li
>
> **摘要:** We present a systematic theoretical framework that interprets masked diffusion models (MDMs) as solutions to energy minimization problems in discrete optimal transport. Specifically, we prove that three distinct energy formulations--kinetic, conditional kinetic, and geodesic energy--are mathematically equivalent under the structure of MDMs, and that MDMs minimize all three when the mask schedule satisfies a closed-form optimality condition. This unification not only clarifies the theoretical foundations of MDMs, but also motivates practical improvements in sampling. By parameterizing interpolation schedules via Beta distributions, we reduce the schedule design space to a tractable 2D search, enabling efficient post-training tuning without model modification. Experiments on synthetic and real-world benchmarks demonstrate that our energy-inspired schedules outperform hand-crafted baselines, particularly in low-step sampling settings.
>
---
#### [replaced 070] Bot Meets Shortcut: How Can LLMs Aid in Handling Unknown Invariance OOD Scenarios?
- **分类: cs.CL**

- **简介: 该论文属于社会机器人检测任务，旨在解决未知不变性分布外样本（OOD）场景下的模型鲁棒性问题。通过构造快捷学习场景评估模型性能，并提出基于大语言模型的缓解策略。**

- **链接: [https://arxiv.org/pdf/2511.08455](https://arxiv.org/pdf/2511.08455)**

> **作者:** Shiyan Zheng; Herun Wan; Minnan Luo; Junhang Huang
>
> **摘要:** While existing social bot detectors perform well on benchmarks, their robustness across diverse real-world scenarios remains limited due to unclear ground truth and varied misleading cues. In particular, the impact of shortcut learning, where models rely on spurious correlations instead of capturing causal task-relevant features, has received limited attention. To address this gap, we conduct an in-depth study to assess how detectors are influenced by potential shortcuts based on textual features, which are most susceptible to manipulation by social bots. We design a series of shortcut scenarios by constructing spurious associations between user labels and superficial textual cues to evaluate model robustness. Results show that shifts in irrelevant feature distributions significantly degrade social bot detector performance, with an average relative accuracy drop of 32\% in the baseline models. To tackle this challenge, we propose mitigation strategies based on large language models, leveraging counterfactual data augmentation. These methods mitigate the problem from data and model perspectives across three levels, including data distribution at both the individual user text and overall dataset levels, as well as the model's ability to extract causal information. Our strategies achieve an average relative performance improvement of 56\% under shortcut scenarios.
>
---
#### [replaced 071] AmbiSQL: Interactive Ambiguity Detection and Resolution for Text-to-SQL
- **分类: cs.DB; cs.CL**

- **简介: 该论文属于Text-to-SQL任务，旨在解决查询歧义导致的SQL生成错误问题。提出AmbiSQL系统，通过交互方式检测并消除歧义，提升SQL准确性。**

- **链接: [https://arxiv.org/pdf/2508.15276](https://arxiv.org/pdf/2508.15276)**

> **作者:** Zhongjun Ding; Yin Lin; Tianjing Zeng; Rong Zhu; Bolin Ding; Jingren Zhou
>
> **摘要:** Text-to-SQL systems translate natural language questions into SQL queries, providing substantial value for non-expert users. While large language models (LLMs) show promising results for this task, they remain error-prone. Query ambiguity has been recognized as a major obstacle in LLM-based Text-to-SQL systems, leading to misinterpretation of user intent and inaccurate SQL generation. To this end, we present AmbiSQL, an interactive system that automatically detects query ambiguities and guides users through intuitive multiple-choice questions to clarify their intent. It introduces a fine-grained ambiguity taxonomy for identifying ambiguities arising from both database elements and LLM reasoning, and subsequently incorporates user feedback to rewrite ambiguous questions. In this demonstration, AmbiSQL is integrated with XiYan-SQL, our commercial Text-to-SQL backend. We provide 40 ambiguous queries collected from two real-world benchmarks that SIGMOD'26 attendees can use to explore how disambiguation improves SQL generation quality. Participants can also apply the system to their own databases and natural language questions. The codebase and demo video are available at: this https URL and this https URL.
>
---
#### [replaced 072] How Psychological Learning Paradigms Shaped and Constrained Artificial Intelligence
- **分类: cs.CL; cs.CY**

- **简介: 论文探讨心理学习范式对人工智能的影响，分析其优缺点，提出ReSynth框架解决表示架构问题，旨在提升AI的适应性。**

- **链接: [https://arxiv.org/pdf/2603.18203](https://arxiv.org/pdf/2603.18203)**

> **作者:** Alex Anvi Eponon; Ildar Batyrshin; Christian E. Maldonado-Sifuentes; Grigori Sidorov
>
> **备注:** preprint journal
>
> **摘要:** The dominant paradigms of artificial intelligence were shaped by learning theories from psychology: behaviorism inspired reinforcement learning, cognitivism gave rise to deep learning and memory-augmented architectures, and constructivism influenced curriculum learning and compositional approaches. This paper argues that each AI paradigm inherited not only the strengths but the structural limitations of the psychological theory that inspired it. Reinforcement learning cannot account for the internal structure of knowledge, deep learning compresses representations into opaque parameter spaces resistant to principled update, and current integrative approaches lack a formal account of how new understanding is constructed from existing components. The paper further examines a cross-cultural divergence in the interpretation of rote learning, arguing that the Eastern conception of memorization as a structured, multi-phase precursor to understanding offers an underexploited bridge between psychological theory and AI methodology. Drawing on the systematicity debate and critique of Aizawa of both classicism and connectionism, this paper introduces ReSynth, a trimodular framework that separates reasoning (Intellect), purpose (Identity), and knowledge (Memory) as architecturally independent components. The paper traces the genealogy from psychological paradigm to AI method, diagnoses the inherited limitations at each stage, and argues that adaptability, the central challenge of artificial general intelligence requires a representational architecture in which systematic behavior is a necessary consequence rather than an accidental property.
>
---
#### [replaced 073] Evaluating LLM-Generated Lessons from the Language Learning Students' Perspective: A Short Case Study on Duolingo
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于语言学习任务，旨在解决LLM生成课程缺乏专业场景的问题。通过调查五名员工，发现通用场景更有效，但专业场景有助于提升职业流利度，建议结合个性化专业场景与基础通用内容。**

- **链接: [https://arxiv.org/pdf/2603.18873](https://arxiv.org/pdf/2603.18873)**

> **作者:** Carlos Rafael Catalan; Patricia Nicole Monderin; Lheane Marie Dizon; Gap Estrella; Raymund John Sarmimento; Marie Antoinette Patalagsa
>
> **备注:** 5 pages,3 figures,presented at the 3rd HEAL Workshop at CHI 2026
>
> **摘要:** Popular language learning applications such as Duolingo use large language models (LLMs) to generate lessons for its users. Most lessons focus on general real-world scenarios such as greetings, ordering food, or asking directions, with limited support for profession-specific contexts. This gap can hinder learners from achieving professional-level fluency, which we define as the ability to communicate comfortably various work-related and domain-specific information in the target language. We surveyed five employees from a multinational company in the Philippines on their experiences with Duolingo. Results show that respondents encountered general scenarios more frequently than work-related ones, and that the former are relatable and effective in building foundational grammar, vocabulary, and cultural knowledge. The latter helps bridge the gap toward professional fluency as it contains domain-specific vocabulary. Each participant suggested lesson scenarios that diverge in contexts when analyzed in aggregate. With this understanding, we propose that language learning applications should generate lessons that adapt to an individual's needs through personalized, domain specific lesson scenarios while maintaining foundational support through general, relatable lesson scenarios.
>
---
#### [replaced 074] Teaching Old Tokenizers New Words: Efficient Tokenizer Adaptation for Pre-trained Models
- **分类: cs.CL**

- **简介: 该论文属于预训练模型优化任务，解决领域适配中的分词器扩展与精简问题。通过持续BPE训练扩展词汇，结合剪枝保留质量，提升分词效率。**

- **链接: [https://arxiv.org/pdf/2512.03989](https://arxiv.org/pdf/2512.03989)**

> **作者:** Taido Purason; Pavel Chizhov; Ivan P. Yamshchikov; Mark Fishel
>
> **备注:** Accepted to Findings of EACL 2026
>
> **摘要:** Tokenizer adaptation plays an important role in adapting pre-trained language models to new domains or languages. In this work, we address two complementary aspects of this process: vocabulary extension and pruning. The common approach to extension trains a new tokenizer on domain-specific text and appends the tokens that do not overlap with the existing vocabulary, which often results in many tokens that are unreachable or never used. We propose continued BPE training that extends a pre-trained tokenizer by continuing the BPE merge learning process on new data. Experiments across multiple languages and model families show that this approach improves tokenization efficiency and leads to better utilization of added vocabulary. We also introduce leaf-based vocabulary pruning, which removes redundant tokens while preserving model quality. Together, these methods provide practical tools for controlled vocabulary modification, which we release as an open-source toolkit.
>
---
#### [replaced 075] Different Demographic Cues Yield Inconsistent Conclusions About LLM Personalization and Bias
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理中的模型评估任务，探讨LLM在不同人口统计线索下的个性化与偏见表现。研究发现，不同线索导致不一致结论，强调需多线索评估以提高可靠性。**

- **链接: [https://arxiv.org/pdf/2601.18486](https://arxiv.org/pdf/2601.18486)**

> **作者:** Manuel Tonneau; Neil K. R. Seghal; Niyati Malhotra; Sharif Kazemi; Victor Orozco-Olvera; Ana María Muñoz Boudet; Lakshmi Subramanian; Samuel P. Fraiberger; Sharath Chandra Guntuku; Valentin Hofmann
>
> **摘要:** Demographic cue-based evaluation is widely used to study how large language models (LLMs) adapt their responses to signaled demographic attributes within and across groups. This approach typically relies on a single cue (e.g., names) as a proxy for group membership, implicitly treating different cues as interchangeable operationalizations of the same identity-conditioned behavior. We test this assumption in realistic advice-seeking interactions spanning 14.8 million prompts, focusing on race and gender in a U.S. context. We find that cues for the same group induce only partially overlapping changes in model responses, yielding inconsistent conclusions about personalization, while bias conclusions are unstable, with both magnitude and direction of group differences varying across cues. We further show that these inconsistencies reflect differences in cue-group association strength and linguistic features bundled within cues that shape model responses. Together, our findings suggest that demographic conditioning in LLMs is not a cue-invariant category-level parameter but depends fundamentally on how identity is cued, reflecting responses to linguistic signals rather than stable demographic categories. We therefore advocate multi-cue, mechanism-aware evaluations for robust and interpretable claims about demographic variation in LLM responses.
>
---
#### [replaced 076] Agent-Dice: Disentangling Knowledge Updates via Geometric Consensus for Agent Continual Learning
- **分类: cs.CL**

- **简介: 该论文属于持续学习任务，旨在解决代理在学习新任务时的灾难性遗忘问题。提出Agent-Dice框架，通过知识解耦提升模型稳定性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.03641](https://arxiv.org/pdf/2601.03641)**

> **作者:** Zheng Wu; Xingyu Lou; Xinbei Ma; Yansi Li; Weiwen Liu; Weinan Zhang; Jun Wang; Zhuosheng Zhang
>
> **摘要:** Large Language Model (LLM)-based agents significantly extend the utility of LLMs by interacting with dynamic environments. However, enabling agents to continually learn new tasks without catastrophic forgetting remains a critical challenge, known as the stability-plasticity dilemma. In this work, we argue that this dilemma fundamentally arises from the failure to explicitly distinguish between common knowledge shared across tasks and conflicting knowledge introduced by task-specific interference. To address this, we propose Agent-Dice, a parameter fusion framework based on directional consensus evaluation. Concretely, Agent-Dice disentangles knowledge updates through a two-stage process: geometric consensus filtering to prune conflicting gradients, and curvature-based importance weighting to amplify shared semantics. We provide a rigorous theoretical analysis that establishes the validity of the proposed fusion scheme and offers insight into the origins of the stability-plasticity dilemma. Extensive experiments on GUI agents and tool-use agent domains demonstrate that Agent-Dice exhibits outstanding continual learning performance with minimal computational overhead and parameter updates. The codes are available at this https URL.
>
---
#### [replaced 077] IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding
- **分类: cs.CV; cs.CL; cs.CR**

- **简介: 该论文属于视觉定位任务，针对VLM系统的安全性问题，提出IAG方法实现多目标后门攻击，通过动态生成文本引导的触发器，在不影响正常性能的情况下提升攻击效果。**

- **链接: [https://arxiv.org/pdf/2508.09456](https://arxiv.org/pdf/2508.09456)**

> **作者:** Junxian Li; Beining Xu; Simin Chen; Jiatong Li; Jingdi Lei; Haodong Zhao; Di Zhang
>
> **备注:** Accepted by CVPR 2026; Code is at this https URL
>
> **摘要:** Recent advances in vision-language models (VLMs) have significantly enhanced the visual grounding task, which involves locating objects in an image based on natural language queries. Despite these advancements, the security of VLM-based grounding systems has not been thoroughly investigated. This paper reveals a novel and realistic vulnerability: the first multi-target backdoor attack on VLM-based visual grounding. Unlike prior attacks that rely on static triggers or fixed targets, we propose IAG, a method that dynamically generates input-aware, text-guided triggers conditioned on any specified target object description to execute the attack. This is achieved through a text-conditioned UNet that embeds imperceptible target semantic cues into visual inputs while preserving normal grounding performance on benign samples. We further develop a joint training objective that balances language capability with perceptual reconstruction to ensure imperceptibility, effectiveness, and stealth. Extensive experiments on multiple VLMs (e.g., LLaVA, InternVL, Ferret) and benchmarks (RefCOCO, RefCOCO+, RefCOCOg, Flickr30k Entities, and ShowUI) demonstrate that IAG achieves the best ASRs compared with other baselines on almost all settings without compromising clean accuracy, maintaining robustness against existing defenses, and exhibiting transferability across datasets and models. These findings underscore critical security risks in grounding-capable VLMs and highlight the need for further research on trustworthy multimodal understanding.
>
---
#### [replaced 078] From Synthetic Scenes to Real Performance: Enhancing Spatial Reasoning in VLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型优化任务，旨在解决真实场景数据偏差和分布不均问题。通过生成可控的合成数据进行微调，提升模型在真实数据上的表现。**

- **链接: [https://arxiv.org/pdf/2511.11440](https://arxiv.org/pdf/2511.11440)**

> **作者:** Massimo Rizzoli; Simone Alghisi; Seyed Mahed Mousavi; Giuseppe Riccardi
>
> **摘要:** Fine-tuning Vision-Language Models (VLMs) is a common strategy to improve performance following an ad-hoc data collection and annotation of real-world scenes. However, this process is often prone to biases, errors, and distribution imbalance, resulting in overfitting and imbalanced performance. Although a few studies have tried to address this problem by generating synthetic data, they lacked control over distribution bias and annotation quality. To address these challenges, we redesign the fine-tuning process in two ways. First, we control the generation of data and its annotations, ensuring it is free from bias, distribution imbalance, and annotation errors. We automatically construct the dataset by comprehensively sampling objects' attributes, including color, shape, size, and position within the scene. Secondly, using this annotated dataset, we fine-tune state-of-the-art VLMs and assess performance transferability to real-world data on the absolute position task. We conduct exhaustive evaluations on both synthetic and real-world benchmarks. Our experiments reveal two key findings: 1) fine-tuning on balanced synthetic data yields uniform performance across the visual scene and mitigates common biases; and 2) fine-tuning on synthetic stimuli improves performance by 13% on real-world data (COCO), outperforming models fine-tuned on the full COCO train set.
>
---
#### [replaced 079] Hybrid Architectures for Language Models: Systematic Analysis and Design Insights
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究混合架构语言模型。旨在解决如何有效结合自注意力与状态空间模型以提升性能与效率的问题。通过系统分析不同融合策略，提出优化设计方法。**

- **链接: [https://arxiv.org/pdf/2510.04800](https://arxiv.org/pdf/2510.04800)**

> **作者:** Sangmin Bae; Bilge Acun; Chien-Yu Lin; Haroun Habeeb; Seungyeon Kim; Liang Luo; Junjie Wang; Carole-Jean Wu
>
> **备注:** 41 pages, 8 figures, 22 tables;
>
> **摘要:** Recent progress in large language models demonstrates that hybrid architectures--combining self-attention mechanisms with structured state space models like Mamba--can achieve a compelling balance between modeling quality and computational efficiency, particularly for long-context tasks. While these hybrid models show promising performance, systematic comparisons of hybridization strategies and analyses on the key factors behind their effectiveness have not been clearly shared to the community. In this work, we present a holistic evaluation of hybrid architectures based on inter-layer (sequential) or intra-layer (parallel) fusion. We comprehensively evaluate these designs across multiple dimensions: language modeling and downstream task performance, long-context capabilities, scaling analysis, and training and inference efficiency. By investigating the core characteristics of their computational primitive, we identify the most critical elements for each hybridization strategy and further propose optimal design recipes for hybrid models. Our comprehensive analysis provides practical guidance and valuable insights for developing hybrid language models, facilitating the optimization of architectural configurations.
>
---
#### [replaced 080] Modality Matching Matters: Calibrating Language Distances for Cross-Lingual Transfer in URIEL+
- **分类: cs.CL**

- **简介: 该论文属于跨语言迁移任务，解决现有知识库在语言表示和距离聚合上的不足。提出结构感知的多模态距离表示，并构建统一的综合距离模型，提升迁移效果。**

- **链接: [https://arxiv.org/pdf/2510.19217](https://arxiv.org/pdf/2510.19217)**

> **作者:** York Hay Ng; Aditya Khan; Xiang Lu; Matteo Salloum; Michael Zhou; Phuong H. Hoang; A. Seza Doğruöz; En-Shiun Annie Lee
>
> **备注:** Accepted to EACL 2026 SRW
>
> **摘要:** Existing linguistic knowledge bases such as URIEL+ provide valuable geographic, genetic and typological distances for cross-lingual transfer but suffer from two key limitations. First, their one-size-fits-all vector representations are ill-suited to the diverse structures of linguistic data. Second, they lack a principled method for aggregating these signals into a single, comprehensive score. In this paper, we address these gaps by introducing a framework for type-matched language distances. We propose novel, structure-aware representations for each distance type: speaker-weighted distributions for geography, hyperbolic embeddings for genealogy, and a latent variables model for typology. We unify these signals into a robust, task-agnostic composite distance. Across multiple zero-shot transfer benchmarks, we demonstrate that our representations significantly improve transfer performance when the distance type is relevant to the task, while our composite distance yields gains in most tasks.
>
---
#### [replaced 081] Detecting AI-Generated Content in Academic Peer Reviews
- **分类: cs.CL; cs.AI; cs.LG; cs.SI**

- **简介: 该论文属于AI检测任务，旨在识别学术同行评审中的AI生成内容。研究分析了ICLR和Nature Communications的评审数据，发现AI生成内容在2025年占比达20%和12%，揭示其快速增加的趋势。**

- **链接: [https://arxiv.org/pdf/2602.00319](https://arxiv.org/pdf/2602.00319)**

> **作者:** Siyuan Shen; Kai Wang
>
> **摘要:** The growing availability of large language models (LLMs) has raised questions about their role in academic peer review. This study examines the temporal emergence of AI-generated content in peer reviews by applying a detection model trained on historical reviews to later review cycles at International Conference on Learning Representations (ICLR) and Nature Communications (NC). We observe minimal detection of AI-generated content before 2022, followed by a substantial increase through 2025, with approximately 20% of ICLR reviews and 12% of Nature Communications reviews classified as AI-generated in 2025. The most pronounced growth of AI-generated reviews in NC occurs between the third and fourth quarter of 2024. Together, these findings provide suggestive evidence of a rapidly increasing presence of AI-assisted content in peer review and highlight the need for further study of its implications for scholarly evaluation.
>
---
#### [replaced 082] VorTEX: Various overlap ratio for Target speech EXtraction
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于语音分离任务，解决真实场景下不同重叠比例的语音提取问题。提出VorTEX模型和PORTE数据集，提升分离效果并避免抑制现象。**

- **链接: [https://arxiv.org/pdf/2603.14803](https://arxiv.org/pdf/2603.14803)**

> **作者:** Ro-hoon Oh; Jihwan Seol; Bugeun Kim
>
> **备注:** Submitted to InterSpeech 2026 (under review)
>
> **摘要:** Target speech extraction (TSE) aims to recover a target speaker's voice from a mixture. While recent text-prompted approaches have shown promise, most approaches assume fully overlapped mixtures, limiting insight into behavior across realistic overlap ratios. We introduce VorTEX (Various overlap ratio for Target speech EXtraction), a text-prompted TSE architecture with a Decoupled Adaptive Multi-branch (DAM) Fusion block that separates primary extraction from auxiliary regularization pathways. To enable controlled analysis, we construct PORTE, a two-speaker dataset spanning overlap ratios from 0% to 100%. We further propose Suppression Ratio on Energy (SuRE), a diagnostic metric that detects suppression behavior not captured by conventional measures. Experiments show that existing models exhibit suppression or residual interference under overlap, whereas VorTEX achieves the highest separation fidelity across 20-100% overlap (e.g., 5.50 dB at 20% and 2.04 dB at 100%) while maintaining zero SuRE, indicating robust extraction without suppression-driven artifacts.
>
---
#### [replaced 083] Hidden State Poisoning Attacks against Mamba-based Language Models
- **分类: cs.CL**

- **简介: 该论文研究了针对Mamba语言模型的隐藏状态污染攻击（HiSPA），揭示其对抗脆弱性。任务是评估模型鲁棒性，解决安全漏洞问题，通过实验验证攻击有效性并提出潜在防御方向。**

- **链接: [https://arxiv.org/pdf/2601.01972](https://arxiv.org/pdf/2601.01972)**

> **作者:** Alexandre Le Mercier; Chris Develder; Thomas Demeester
>
> **备注:** 29 pages, 4 figures
>
> **摘要:** State space models (SSMs) like Mamba offer efficient alternatives to Transformer-based language models, with linear time complexity. Yet, their adversarial robustness remains critically unexplored. This paper studies the phenomenon whereby specific short input phrases induce a partial amnesia effect in such models, by irreversibly overwriting information in their hidden states, referred to as a Hidden State Poisoning Attack (HiSPA). Our benchmark RoBench-25 allows evaluating a model's information retrieval capabilities when subject to HiSPAs, and confirms the vulnerability of SSMs against such attacks. Even the recent Jamba-1.7-Mini SSM--Transformer (a 52B hybrid model) collapses on RoBench-25 under some HiSPA triggers, whereas pure Transformers do not. We also observe that HiSPA triggers significantly weaken the Jamba model on the popular Open-Prompt-Injections benchmark, unlike pure Transformers. We further show that the theoretical and empirical findings extend to Mamba-2, and also analyse a Mamba-2-based hybrid (Nemotron-3-Nano). Finally, our interpretability study reveals patterns in Mamba's hidden layers during HiSPAs that could be used to build a HiSPA mitigation system. The full code and data to reproduce the experiments can be found at this https URL.
>
---
#### [replaced 084] Collusive Pricing Under LLM
- **分类: econ.TH; cs.AI; cs.CE; cs.CL; cs.GT**

- **简介: 该论文研究LLM在双寡头市场中促进合谋定价的机制。任务是分析LLM配置对市场价格行为的影响，解决如何通过模型参数设计影响市场竞争的问题。工作包括建立模型并分析不同参数下的系统稳定性。**

- **链接: [https://arxiv.org/pdf/2601.01279](https://arxiv.org/pdf/2601.01279)**

> **作者:** Shengyu Cao; Ming Hu
>
> **备注:** 46 pages
>
> **摘要:** We study how delegating pricing to large language models (LLMs) can facilitate collusion in a duopoly when both sellers rely on the same pre-trained model. The LLM is characterized by (i) a propensity parameter capturing its internal bias toward high-price recommendations and (ii) an output-fidelity parameter measuring how tightly outputs track that bias; the propensity evolves through retraining. We show that configuring LLMs for robustness and reproducibility can induce collusion via a phase transition: there exists a critical output-fidelity threshold that pins down long-run behavior. Below it, competitive pricing is the unique long-run outcome. Above it, the system is bistable, with competitive and collusive pricing both locally stable and the realized outcome determined by the model's initial preference. The collusive regime resembles tacit collusion: prices are elevated on average, yet occasional low-price recommendations provide plausible deniability. With perfect fidelity, full collusion emerges from any interior initial condition. For finite training batches of size $b$, infrequent retraining (driven by computational costs) further amplifies collusion: conditional on starting in the collusive basin, the probability of collusion approaches one as $b$ grows, since larger batches dampen stochastic fluctuations that might otherwise tip the system toward competition. The indeterminacy region shrinks at rate $O(1/\sqrt{b})$.
>
---
#### [replaced 085] MHPO: Modulated Hazard-aware Policy Optimization for Stable Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决GRPO框架中训练不稳定问题。提出MHPO框架，通过LFM和DHP机制稳定策略更新，提升训练稳定性与性能。**

- **链接: [https://arxiv.org/pdf/2603.16929](https://arxiv.org/pdf/2603.16929)**

> **作者:** Hongjun Wang; Wei Liu; Weibo Gu; Xing Sun; Kai Han
>
> **备注:** 18 pages, 3 figures, 4 tables
>
> **摘要:** Regulating the importance ratio is critical for the training stability of Group Relative Policy Optimization (GRPO) based frameworks. However, prevailing ratio control methods, such as hard clipping, suffer from non-differentiable boundaries and vanishing gradient regions, failing to maintain gradient fidelity. Furthermore, these methods lack a hazard-aware mechanism to adaptively suppress extreme deviations, leaving the optimization process vulnerable to abrupt policy shifts. To address these challenges, we propose Modulated Hazard-aware Policy Optimization (MHPO), a novel framework designed for robust and stable reinforcement learning. The proposed MHPO introduces a Log-Fidelity Modulator (LFM) to map unbounded importance ratios into a bounded, differentiable domain. This mechanism effectively prevents high-variance outlier tokens from destabilizing the loss landscape while ensuring global gradient stability. Complementarily, a Decoupled Hazard Penalty (DHP) integrates cumulative hazard functions from survival analysis to independently regulate positive and negative policy shifts. By shaping the optimization landscape with hazard-aware penalties, the proposed MHPO achieves fine-grained regulation of asymmetric policy shifts simultaneously mitigating mode collapse from over-expansion and preventing policy erosion from catastrophic contraction within a stabilized trust region. Extensive evaluations on diverse reasoning benchmarks across both text-based and vision-language tasks demonstrate that MHPO consistently outperforms existing methods, achieving superior performance while significantly enhancing training stability.
>
---
#### [replaced 086] Automatic Essay Scoring and Feedback Generation in Basque Language Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自动作文评分与反馈生成任务，旨在解决低资源语言Basque的C1级作文评估问题。构建了首个公开数据集，并优化模型提升评分与反馈质量。**

- **链接: [https://arxiv.org/pdf/2512.08713](https://arxiv.org/pdf/2512.08713)**

> **作者:** Ekhi Azurmendi; Xabier Arregi; Oier Lopez de Lacalle
>
> **备注:** Accepted to LREC 2026
>
> **摘要:** This paper introduces the first publicly available dataset for Automatic Essay Scoring (AES) and feedback generation in Basque, targeting the CEFR C1 proficiency level. The dataset comprises 3,200 essays from HABE, each annotated by expert evaluators with criterion specific scores covering correctness, richness, coherence, cohesion, and task alignment enriched with detailed feedback and error examples. We fine-tune open-source models, including RoBERTa-EusCrawl and Latxa 8B/70B, for both scoring and explanation generation. Our experiments show that encoder models remain highly reliable for AES, while supervised fine-tuning (SFT) of Latxa significantly enhances performance, surpassing state-of-the-art (SoTA) closed-source systems such as GPT-5 and Claude Sonnet 4.5 in scoring consistency and feedback quality. We also propose a novel evaluation methodology for assessing feedback generation, combining automatic consistency metrics with expert-based validation of extracted learner errors. Results demonstrate that the fine-tuned Latxa model produces criterion-aligned, pedagogically meaningful feedback and identifies a wider range of error types than proprietary models. This resource and benchmark establish a foundation for transparent, reproducible, and educationally grounded NLP research in low-resource languages such as Basque.
>
---
#### [replaced 087] Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文介绍Nemotron-Cascade 2，一个30B参数的开源大模型，解决高效推理与代理能力问题。通过级联强化学习和多领域策略蒸馏，提升数学与编程性能，达到国际竞赛高水平。**

- **链接: [https://arxiv.org/pdf/2603.19220](https://arxiv.org/pdf/2603.19220)**

> **作者:** Zhuolin Yang; Zihan Liu; Yang Chen; Wenliang Dai; Boxin Wang; Sheng-Chieh Lin; Chankyu Lee; Yangyi Chen; Dongfu Jiang; Jiafan He; Renjie Pi; Grace Lam; Nayeon Lee; Alexander Bukharin; Mohammad Shoeybi; Bryan Catanzaro; Wei Ping
>
> **备注:** We release the model and data at this https URL
>
> **摘要:** We introduce Nemotron-Cascade 2, an open 30B MoE model with 3B activated parameters that delivers best-in-class reasoning and strong agentic capabilities. Despite its compact size, its mathematical and coding reasoning performance approaches that of frontier open models. It is the second open-weight LLM, after DeepSeekV3.2-Speciale-671B-A37B, to achieve Gold Medal-level performance in the 2025 International Mathematical Olympiad (IMO), the International Olympiad in Informatics (IOI), and the ICPC World Finals, demonstrating remarkably high intelligence density with 20x fewer parameters. In contrast to Nemotron-Cascade 1, the key technical advancements are as follows. After SFT on a meticulously curated dataset, we substantially expand Cascade RL to cover a much broader spectrum of reasoning and agentic domains. Furthermore, we introduce multi-domain on-policy distillation from the strongest intermediate teacher models for each domain throughout the Cascade RL process, allowing us to efficiently recover benchmark regressions and sustain strong performance gains along the way. We release the collection of model checkpoint and training data.
>
---
#### [replaced 088] Explore Briefly, Then Decide: Mitigating LLM Overthinking via Cumulative Entropy Regulation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决LLM过思考问题。提出TECA和CER机制，通过动态决定推理结束点，提升效率并保持解题能力。**

- **链接: [https://arxiv.org/pdf/2510.02249](https://arxiv.org/pdf/2510.02249)**

> **作者:** Yi Bin; Tianyi Jiang; Yujuan Ding; Kainian Zhu; Fei Ma; Jingkuan Song; Yang Yang; Heng Tao Shen
>
> **备注:** Code: this https URL
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable reasoning abilities on complex problems using long Chain-of-Thought (CoT) reasoning. However, they often suffer from overthinking, meaning generating unnecessarily lengthy reasoning steps for simpler problems. This issue may degrade the efficiency of the models and make them difficult to adapt the reasoning depth to the complexity of problems. To address this, we introduce a novel metric Token Entropy Cumulative Average (TECA), which measures the extent of exploration throughout the reasoning process. We further propose a novel reasoning paradigm named "Explore Briefly, Then Decide", with an associated Cumulative Entropy Regulation (CER) mechanism. This paradigm leverages TECA to help the model dynamically determine the optimal point to conclude its thought process and provide a final answer, thus achieving efficient reasoning. Experimental results across diverse mathematical benchmarks show that our approach substantially mitigates overthinking without sacrificing problem solving ability. With our thinking paradigm, the average response length decreases by up to 71% on simpler datasets, demonstrating the effectiveness of our method in creating a more efficient and adaptive reasoning process.
>
---
#### [replaced 089] Automatically Benchmarking LLM Code Agents through Agent-Driven Annotation and Evaluation
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于代码代理评估任务，旨在解决现有基准的高成本、低多样性及评估不准确问题。提出PRDBench和PRDJudge，实现高效、准确的代码代理评测。**

- **链接: [https://arxiv.org/pdf/2510.24358](https://arxiv.org/pdf/2510.24358)**

> **作者:** Lingyue Fu; Bolun Zhang; Hao Guan; Yaoming Zhu; Lin Qiu; Weiwen Liu; Xuezhi Cao; Xunliang Cai; Weinan Zhang; Yong Yu
>
> **备注:** Accepted by AAMAS 2026
>
> **摘要:** Recent advances in code agents have enabled automated software development at the project level, supported by large language models (LLMs). However, existing benchmarks for code agent evaluation face two major limitations. First, creating high-quality project-level evaluation datasets requires extensive domain expertise, leading to prohibitive annotation costs and limited diversity. Second, while recent Agent-as-a-Judge paradigms address the rigidity of traditional unit tests by enabling flexible metrics, their reliance on In-Context Learning (ICL) with general LLMs often results in inaccurate assessments that misalign with human standards. To address these challenges, we propose an agent-driven benchmark construction pipeline that leverages human supervision to efficiently generate diverse project-level tasks. Based on this, we introduce PRDBench, comprising 50 real-world Python projects across 20 domains, each with structured Product Requirement Documents (PRDs) and comprehensive criteria. Furthermore, to overcome the inaccuracy of general LLM judges, we propose a highly reliable evaluation framework powered by a specialized, fine-tuned model. Based on Qwen3-Coder-30B, our dedicated PRDJudge achieves over 90% human alignment in fixed-interface scenarios. Extensive experiments demonstrate that our suite provides a scalable, robust, and highly accurate framework for assessing state-of-the-art code agents.
>
---
#### [replaced 090] Emotionally Charged, Logically Blurred: AI-driven Emotional Framing Impairs Human Fallacy Detection
- **分类: cs.CL**

- **简介: 该论文属于情感与逻辑推理任务，研究情感框架如何影响人类对逻辑谬误的识别。通过AI生成带有情感的谬误论证，发现情感会降低人类检测谬误的能力。**

- **链接: [https://arxiv.org/pdf/2510.09695](https://arxiv.org/pdf/2510.09695)**

> **作者:** Yanran Chen; Lynn Greschner; Roman Klinger; Michael Klenk; Steffen Eger
>
> **备注:** EACL 2026 Main Camera-ready; Figure 4 and typo fixed
>
> **摘要:** Logical fallacies are common in public communication and can mislead audiences; fallacious arguments may still appear convincing despite lacking soundness, because convincingness is inherently subjective. We present the first computational study of how emotional framing interacts with fallacies and convincingness, using large language models (LLMs) to systematically change emotional appeals in fallacious arguments. We benchmark eight LLMs on injecting emotional appeal into fallacious arguments while preserving their logical structures, then use the best models to generate stimuli for a human study. Our results show that LLM-driven emotional framing reduces human fallacy detection in F1 by 14.5% on average. Humans perform better in fallacy detection when perceiving enjoyment than fear or sadness, and these three emotions also correlate with significantly higher convincingness compared to neutral or other emotion states. Our work has implications for AI-driven emotional manipulation in the context of fallacious argumentation.
>
---
#### [replaced 091] A Theory of Adaptive Scaffolding for LLM-Based Pedagogical Agents
- **分类: cs.CL**

- **简介: 该论文属于教育技术领域，旨在解决LLM在教学中缺乏理论基础的问题。通过结合学习理论构建自适应指导框架，开发了Inquizzitor系统，提升智能教学效果。**

- **链接: [https://arxiv.org/pdf/2508.01503](https://arxiv.org/pdf/2508.01503)**

> **作者:** Clayton Cohn; Surya Rayala; Namrata Srivastava; Joyce Horn Fonteles; Shruti Jain; Xinying Luo; Divya Mereddy; Naveeduddin Mohammed; Gautam Biswas
>
> **备注:** Published in the proceedings of AAAI 2026 (main technical track)
>
> **摘要:** Large language models (LLMs) present new opportunities for creating pedagogical agents that engage in meaningful dialogue to support student learning. However, current LLM systems used in classrooms often lack the solid theoretical foundations found in earlier intelligent tutoring systems. To bridge this gap, we propose a framework that combines Evidence-Centered Design with Social Cognitive Theory and Zone of Proximal Development for adaptive scaffolding in LLM-based agents focused on STEM+C learning. We instantiate this framework with Inquizzitor, an LLM-based formative assessment agent that integrates human-AI hybrid intelligence and provides feedback grounded in cognitive science principles. Our findings show that Inquizzitor delivers high-quality assessment and interaction aligned with core learning theories, offering effective guidance that students value. This research demonstrates the potential for theory-driven LLM integration in education, highlighting the ability of these systems to provide adaptive and principled instruction.
>
---
#### [replaced 092] HPE-CogVLM: Advancing Vision Language Models with a Head Pose Grounding Task
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于头姿估计任务，旨在提升HPE精度。针对现有模型在真实场景下表现不佳的问题，提出基于VLM的融合方法，有效整合检测与HPE能力，显著提高准确率。**

- **链接: [https://arxiv.org/pdf/2406.01914](https://arxiv.org/pdf/2406.01914)**

> **作者:** Yu Tian; Tianqi Shao; Tsukasa Demizu; Xuyang Wu; Hsin-Tai Wu
>
> **备注:** Accepted by IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2026. This version includes major updates in methodology and experiments. The final version is available at IEEE Xplore
>
> **摘要:** Head pose estimation (HPE) requires a sophisticated understanding of 3D spatial relationships to generate precise yaw, pitch, and roll angles. Previous HPE models, primarily CNN-based, rely on cropped close-up human head images as inputs and often lack robustness in real-world scenario. Vision Language Models (VLMs) can analyze entire images while focusing on specific objects through their attention mechanisms. In this paper, we propose a novel framework to improve the HPE accuracy by leveraging the object detection grounding capability of a VLM, referred to as CogVLM. We empirically find that directly LoRA fine-tuning of this VLM for the HPE task fails to achieve desirable HPE accuracy, while some model merging methods can improve accuracy but frequently produce blended invalid response formats, struggling to handle both object detection and HPE tasks simultaneously. To integrate HPE capability into CogVLM effectively, we develop a novel LoRA layer-based model merging method. This merging approach applies a high cosine similarity threshold and a 'winner-takes-all' layer selection strategy, aligning attention to the HPE task while preserving original object detection knowledge. It successfully resolves issues with blended invalid response formats and improves accuracy. Results show that our HPE-CogVLM achieves a 31.5% reduction in Mean Absolute Error over the current state-of-the-art CNN model, 6DRepNet, in cross-dataset evaluation. Furthermore, HPE-CogVLM outperforms both directly LoRA fine-tuned and task arithmetic-based merged VLMs across all HPE metrics.
>
---
#### [replaced 093] On-Policy Context Distillation for Language Models
- **分类: cs.CL**

- **简介: 该论文提出On-Policy Context Distillation（OPCD），解决语言模型知识内化问题。通过学生模型在自身轨迹上训练，优化知识迁移效果，提升任务准确性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.12275](https://arxiv.org/pdf/2602.12275)**

> **作者:** Tianzhu Ye; Li Dong; Xun Wu; Shaohan Huang; Furu Wei
>
> **摘要:** Context distillation enables language models to internalize in-context knowledge into their parameters. In our work, we propose On-Policy Context Distillation (OPCD), a framework that bridges on-policy distillation with context distillation by training a student model on its own generated trajectories while minimizing reverse Kullback-Leibler divergence against a context-conditioned teacher. We demonstrate the effectiveness of OPCD on two important applications: experiential knowledge distillation, where models extract and consolidate transferable knowledge from their historical solution traces, and system prompt distillation, where models internalize beneficial behaviors encoded in optimized prompts. Across mathematical reasoning, text-based games, and domain-specific tasks, OPCD consistently outperforms baseline methods, achieving higher task accuracy while better preserving out-of-distribution capabilities. We further show that OPCD enables effective cross-size distillation, where smaller student models can internalize experiential knowledge from larger teachers.
>
---
#### [replaced 094] Seamless Deception: Larger Language Models Are Better Knowledge Concealers
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型安全任务，旨在检测模型是否隐藏知识。研究发现大模型更擅长掩盖信息，现有方法难以有效识别。**

- **链接: [https://arxiv.org/pdf/2603.14672](https://arxiv.org/pdf/2603.14672)**

> **作者:** Dhananjay Ashok; Ruth-Ann Armstrong; Jonathan May
>
> **摘要:** Language Models (LMs) may acquire harmful knowledge, and yet feign ignorance of these topics when under audit. Inspired by the recent discovery of deception-related behaviour patterns in LMs, we aim to train classifiers that detect when a LM is actively concealing knowledge. Initial findings on smaller models show that classifiers can detect concealment more reliably than human evaluators, with gradient-based concealment proving easier to identify than prompt-based methods. However, contrary to prior work, we find that the classifiers do not reliably generalize to unseen model architectures and topics of hidden knowledge. Most concerningly, the identifiable traces associated with concealment become fainter as the models increase in scale, with the classifiers achieving no better than random performance on any model exceeding 70 billion parameters. Our results expose a key limitation in black-box-only auditing of LMs and highlight the need to develop robust methods to detect models that are actively hiding the knowledge they contain.
>
---
#### [replaced 095] C$^2$-Cite: Contextual-Aware Citation Generation for Attributed Large Language Models
- **分类: cs.IR; cs.CL; cs.DL; cs.LG**

- **简介: 该论文属于文本生成任务，解决引用标记与上下文不一致的问题。提出C²-Cite框架，增强引用与内容的语义对齐，提升引用质量和回答准确性。**

- **链接: [https://arxiv.org/pdf/2602.00004](https://arxiv.org/pdf/2602.00004)**

> **作者:** Yue Yu; Ting Bai; HengZhi Lan; Li Qian; Li Peng; Jie Wu; Wei Liu; Jian Luan; Chuan Shi
>
> **备注:** WSDM26
>
> **摘要:** The attribution technique enhances the credibility of LLMs by adding citations to the generated sentences, enabling users to trace back to the original sources and verify the reliability of the output. However, existing instruction-tuned attributed LLMs often fail to properly interpret the contextual semantics of citation symbols (e.g., [i]) during text generation. This shortcoming arises from their insufficient awareness of the context information surrounding citation markers, which in turn leads to disjointed references and poor integration of retrieved knowledge into the generated content. To address this issue, we propose a novel \textbf{C}ontextual-aware \textbf{C}itation generation framework (\textbf{C$^2$}-\textbf{Cite}) that explicitly integrates the semantic relationships between citation markers and their referenced content. Specifically, a contextual citation alignment mechanism is adopted: it first encodes the retrieved document contexts into the symbol representation of citations, then aligns the marker numbers by decoding information from a citation router function. This mechanism enables the transformation of citation markers from generic placeholders into active knowledge pointers that link to the referenced source information. Experimental results on the ALCE benchmark across three datasets validate our framework C$^2$-Cite++: it outperforms the SOTA baseline by an average of 5.8\% in citation quality and 17.4\% in response correctness. The implementation is publicly available at this https URL
>
---
#### [replaced 096] BERnaT: Basque Encoders for Representing Natural Textual Diversity
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决语言模型缺乏多样性的问题。通过构建多源语料并训练模型，提升其对非标准语言变体的泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.03903](https://arxiv.org/pdf/2512.03903)**

> **作者:** Ekhi Azurmendi; Joseba Fernandez de Landa; Jaione Bengoetxea; Maite Heredia; Julen Etxaniz; Mikel Zubillaga; Ander Soraluze; Aitor Soroa
>
> **备注:** Under review for the Journal Procesamiento de Lenguaje Natural 2026 // En revisión en la revista de Procesamiente de Lenguaje Natural 2026
>
> **摘要:** Language models depend on massive text corpora that are often filtered for quality, a process that can unintentionally exclude non-standard linguistic varieties, reduce model robustness and reinforce representational biases. In this paper, we argue that language models should aim to capture the full spectrum of language variation (dialectal, historical, informal, etc.) rather than relying solely on standardized text. Focusing on the Basque language, we construct new corpora combining standard, social media, and historical sources, and pre-train the BERnaT family of encoder-only models in three configurations: standard, diverse, and combined. We further propose an evaluation framework that separates Natural Language Understanding (NLU) tasks into standard and diverse subsets to assess linguistic generalization. Results show that models trained on both standard and diverse data consistently outperform those trained on standard corpora, improving performance across all task types without compromising standard benchmark accuracy. These findings highlight the importance of linguistic diversity in building inclusive, generalizable language models.
>
---
