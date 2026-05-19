# 自然语言处理 cs.CL

- **最新发布 200 篇**

- **更新 166 篇**

## 最新发布

#### [new 001] Language Acquisition Device in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在提升大语言模型的数据效率。通过引入受语言习得装置启发的预预训练方法，解决模型对结构不合理语言敏感的问题。**

- **链接: [https://arxiv.org/pdf/2605.16758](https://arxiv.org/pdf/2605.16758)**

> **作者:** Masato Mita; Taiga Someya; Ryo Yoshida; Yohei Oseki
>
> **备注:** Accepted to ACL2026 Main Conference
>
> **摘要:** Large Language Models (LLMs) remain substantially less data-efficient than humans. Pre-pretraining (PPT) on synthetic languages has been proposed to close this gap, with prior work emphasizing highly expressive formal languages such as $k$-Shuffle Dyck. Inspired by the Language Acquisition Device (LAD) hypothesis, which posits that innate constraints preemptively restrict the learner's hypothesis space to natural-language-like structure, we propose LAD-inspired PPT: pre-pretraining on MP-STRUCT, a formal language whose strings encode hierarchical composition, feature-based dependencies, and long-distance displacement via MERGE, AGREE, and MOVE. A brief 500-step PPT with MP-STRUCT matches strong formal-language baselines in token efficiency while additionally imparting a human-like resistance to structurally implausible languages (e.g., REVERSE). Analyzing simplified variants, we find that MP-STRUCT CORE outperforms $k$-Shuffle Dyck despite not being definable in C-RASP (a formal bound on transformer expressivity), challenging the prior hypothesis that effective PPT languages must be both hierarchically expressive and circuit-theoretically learnable. We show that functional landmarks, which reduce dependency resolution ambiguity, are a key driver, suggesting that effective PPT design depends not only on expressivity but also on the accessibility of dependency resolution.
>
---
#### [new 002] FOL2NS: Generating Natural Sentences from First-Order Logic
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，旨在将一阶逻辑翻译为自然句子。提出FOL2NS框架，解决结构复杂时语义准确性和自然性不足的问题。**

- **链接: [https://arxiv.org/pdf/2605.18155](https://arxiv.org/pdf/2605.18155)**

> **作者:** Mei Jia
>
> **备注:** 11 pages, 8 figures
>
> **摘要:** Translating formal language into natural language is a foundational challenge in NLP, driving various downstream applications in semantic parsing, theorem validation, and question answering. In this study, we introduce First-Order Logic to Natural Sentence (FOL2NS), a neurosymbolic framework designed to generate synthetic FOL formulas and convert them into natural human expressions. It handles deeply nested structures with varying quantifier depths (QD), which are rarely captured by existing corpora. By combining rule-driven modules with fine-tuned language models, FOL2NS enhances the diversity and coverage of the generated samples. In our experiments, we systematically evaluate the framework's capabilities through both character-level analysis and overall performance metrics. Experimental results show that FOL2NS can reliably produce well-formed templates and fluent statements, but it faces challenges in achieving precise semantic representations and natural generation as structural complexity increases.
>
---
#### [new 003] CompactAttention: Accelerating Chunked Prefill with Block-Union KV Selection
- **分类: cs.CL**

- **简介: 该论文属于大模型推理优化任务，解决块状预填充中注意力计算效率低的问题。提出CompactAttention，通过块联合KV选择提升速度。**

- **链接: [https://arxiv.org/pdf/2605.16839](https://arxiv.org/pdf/2605.16839)**

> **作者:** Jiwon Song; Dongwon Jo; Beomseok Kang; Jae-Joon Kim
>
> **摘要:** Chunked prefill has become a widely adopted serving strategy for long-context large language models, but efficient attention computation in this regime remains challenging. Existing sparse attention methods are primarily designed for one-shot prefill and do not translate efficiently to chunked prefill: block-sparse kernels lose efficiency when the query length is limited by the chunk size, while fine-grained pattern search becomes costly when repeated over the accumulated KV cache at every chunk. QUOKA, a recent method that directly targets chunked prefill, avoids sparse-kernel overhead but relies on query-subsampled, token-level KV selection, which can miss query-specific KV entries and introduce explicit KV-copy overhead. To address these limitations, we propose CompactAttention, a chunked-prefill attention mechanism based on Block-Union KV Selection. CompactAttention treats 2D block-sparse masks as KV-selection signals rather than direct sparse-kernel execution plans, and converts them into GQA-aware per-group KV block tables through Q-block union and intra-group union. This construction produces the minimal block tables that preserve all KV blocks selected by the input masks under paged execution constraints, enabling selected KV blocks to be accessed in place without explicit KV compaction. On LLaMA-3.1-8B-Instruct, CompactAttention maintains accuracy close to dense attention on the RULER benchmark while delivering up to 2.72$\times$ attention speedup at 128K context length under chunked prefill.
>
---
#### [new 004] CHI-Bench: Can AI Agents Automate End-to-End, Long-Horizon, Policy-Rich Healthcare Workflows?
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CHI-Bench，用于评估AI在复杂医疗流程中的自动化能力，解决政策密集、多角色协作和多方交互的难题。**

- **链接: [https://arxiv.org/pdf/2605.16679](https://arxiv.org/pdf/2605.16679)**

> **作者:** Haolin Chen; Deon Metelski; Leon Qi; Tao Xia; Joonyul Lee; Steve Brown; Kevin Riley; Frank Wang; T. Y. Alvin Liu; Hank Capps MD; Zeyu Tang; Xiangchen Song; Lingjing Kong; Fan Feng; Tianyi Zeng; Zhiwei Liu; Zixian Ma; Hang Jiang; Fangli Geng; Yuan Yuan; Chenyu You; Qingsong Wen; Hua Wei; Yanjie Fu; Yue Zhao; Carl Yang; Biwei Huang; Kun Zhang; Caiming Xiong; Sanmi Koyejo; Eric P. Xing; Philip S. Yu; Weiran Yao
>
> **备注:** Website: this https URL Code: this https URL Dataset: this https URL
>
> **摘要:** End-to-end automation of realistic healthcare operations stresses three capabilities underrepresented in current benchmarks: policy density, decisions must be grounded in a large library of medical, insurance, and operational rules; Multi-role composition: a single task requires the agent to play multiple roles with handoffs; and multilateral interaction: intermediate workflow steps are multi-turn dialogs, such as peer-to-peer review and patient outreach. We introduce $\chi$-Bench, a benchmark of long-horizon healthcare workflows across three domains: provider prior authorization, payer utilization management, and care management. Each task hands the agent a clinical case in a high-fidelity simulator of 20 healthcare apps exposed via 87 MCP tools, which it must drive to a terminal status through tool calls and writing the role's artifacts, guided by a 1,290+ document managed-care operations handbook skill. Across 30 agent harness/models configurations, the best agent resolves only 28.0% of tasks, no agent clears 20% on strict pass^3, and executing all tasks in a single session slumps the performance to 3.8%. These results raise the hypothesis that similar gaps are likely to surface in other policy-dense, role-composed, irreversible enterprise domains.
>
---
#### [new 005] Artificial Intolerance: Stigmatizing Language in Clinical Documentation Skews Large Language Model Decision-Making
- **分类: cs.CL**

- **简介: 该论文属于临床自然语言处理任务，研究LLMs在临床文本中对污名化语言的敏感性及偏见传播问题。工作包括评估九个模型对不同强度SL的反应，发现其决策显著偏移，并验证缓解策略效果有限。**

- **链接: [https://arxiv.org/pdf/2605.17228](https://arxiv.org/pdf/2605.17228)**

> **作者:** Jen-tse Huang; Didi Zhou; Faith Kamau; Amy Oh; Anne R. Links; Mark Dredze; Mary Catherine Beach; Somnath Saha
>
> **备注:** 9 pages
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in high-stakes domains such as clinical decision support and medical documentation. However, the robustness of these models against subtle linguistic variations, specifically stigmatizing language (SL) commonly found in human-authored clinical notes, remains critically under-explored. In this work, we investigate whether frontier LLMs inherit and propagate this human bias when processing clinical text. We systematically evaluate nine frontier LLMs across four stigmatized medical conditions, utilizing clinical vignettes injected with varying intensities and phenotypes of SL (doubt, blame, and maligning). Our results demonstrate that all evaluated models exhibit substantial bias, with clinical decision-making significantly skewed towards less aggressive patient management. Notably, we observe a high sensitivity to linguistic framing, where a single SL sentence is sufficient to alter model outputs, revealing a clear dose-response relationship. Furthermore, we evaluate standard prompt-based mitigation strategies, including Chain-of-Thought (CoT) reasoning and model self-debiasing. These approaches show limited efficacy; models struggle to explicitly identify SL while remaining implicitly influenced by it. Our findings expose a critical vulnerability in current LLMs regarding fairness and robustness in clinical NLP, underscoring the need for rigorous algorithmic guardrails to prevent the automation of health disparities.
>
---
#### [new 006] NewsLens: A Multi-Agent Framework for Adversarial News Bias Navigation
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于新闻偏见分析任务，旨在检测并揭示新闻中的结构性偏见。通过多智能体框架，识别偏见来源、手法及遗漏内容，提升对新闻偏见的理解与透明度。**

- **链接: [https://arxiv.org/pdf/2605.17364](https://arxiv.org/pdf/2605.17364)**

> **作者:** Joy Bose
>
> **备注:** 17 pages, 2 figures, 7 tables, 1 appendix
>
> **摘要:** Media bias detection has predominantly been framed as a classification task: assign a political label to an article or outlet. We argue this framing is too shallow: it identifies that bias exists but not where, how, or crucially, what is structurally omitted. We present NewsLens, a five-agent adversarial pipeline for structured news bias navigation. A Fact Verifier, Progressive Framing Analyst, Conservative Framing Analyst, Propaganda Detector, and Neutral Summarizer collaborate to deconstruct articles into interpretable framing maps, exposing ideological omissions, rhetorical manipulation, and framing boundaries. The system is evaluated on 15 articles across four geopolitical event clusters (India-Pakistan Kashmir, Gaza, Climate Policy, Ukraine) using Qwen2.5-3B-Instruct (4-bit quantised, Google Colab T4), with cross-model validation using Mistral 7B on the Kashmir cluster. Center outlets show the highest mean Perspective Divergence Score (PDS: Qwen 0.907, Mistral 0.729 on Kashmir subset); conservative-framing outlets show the highest mean Manipulation Index (MI: 0.600 across both models). Cross-model comparison shows high consistency for high-propaganda content (Republic World delta-PDS=0.125, MI=0.8 both models) and greater variance for nuanced reporting. Mann-Whitney U tests find no statistically significant between-group differences at n=15, reported honestly as a sample-size limitation confirmed by post-hoc power analysis. A partial ablation removing the Propaganda Detector shows degraded omission precision in the Neutral Summarizer output. The architecture extends prior lexical-geometric bias work to agentic LLM reasoning, and is fully reproducible using open-weight models without API keys.
>
---
#### [new 007] PaliBench: A Multi-Reference Blueprint for Classical Language Translation Benchmarks
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决经典文本多参考译本评估问题。通过构建PaliBench基准，实现对古典语言翻译的多参考评价方法。**

- **链接: [https://arxiv.org/pdf/2605.16881](https://arxiv.org/pdf/2605.16881)**

> **作者:** Máté Metzger; Nadnapang Phophichit
>
> **备注:** Preprint. This manuscript has not yet been peer reviewed
>
> **摘要:** Digital humanities projects increasingly rely on machine translation and large language models to widen access to classical, religious, and otherwise under-translated textual traditions. Yet standard translation benchmarks are poorly suited to such materials: they typically compare a system output against a single reference translation, even though classical texts often support multiple faithful renderings that differ in terminology, register, and interpretation. This article introduces PaliBench, both a benchmark for Pali-to-English translation and a reusable method for constructing multi-reference translation benchmarks for classical languages. The Pali case study draws on passages from the Sutta Pitaka aligned with independent English translations by Bhikkhu Sujato, Bhikkhu Thanissaro, and Bhikkhu Bodhi. The workflow combines LLM-assisted alignment of independently segmented translations, automated verification against source files, passage-level quality filtering, deduplication of formulaic repetitions, and multi-metric evaluation against multiple human references. The resulting benchmark contains 1,700 passages spanning 8,389 segments and approximately 345,000 tokens. We use it to evaluate ten contemporary large language models with complementary metrics, finding strong cross-metric concordance in system rankings alongside substantial variation in reliability and semantic outlier rates. The broader contribution is methodological: PaliBench shows how existing scholarly translations can be transformed into evaluation infrastructure for interpretive textual traditions without treating any single translation as definitive. Although developed for Pali Buddhist texts, the approach could be portable to other classical corpora where sufficient independent reference translations exist.
>
---
#### [new 008] Predictable Confabulations: Factual Recall by LLMs Scales with Model Size and Topic Frequency
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型的事实回忆能力，探讨模型规模与训练数据主题频率的关系。属于模型性能分析任务，旨在揭示影响事实 recall 的关键因素。**

- **链接: [https://arxiv.org/pdf/2605.18732](https://arxiv.org/pdf/2605.18732)**

> **作者:** Matthew L. Smith; Jonathan P. Shock; Samuel T. Segun; Iyiola E. Olatunji; Tegawendé F. Bissyandé
>
> **备注:** 18 pages, 5 figures, 6 tables
>
> **摘要:** While scaling laws govern aggregate large language model performance, no scaling law has linked factual recall to both model size and training-data composition. We evaluated 38 models on over 8,900 scholarly references evaluated by an automated reference verification system. Recall quality follows a sigmoid in the log-linear combination of model parameter count and topic representation in training data. These two variables alone explain 60% of the variance across 16 dense models from four families, rising to 74-94% within individual families. The form matches a superposition-inspired account in which recall is gated by a signal-to-noise ratio: signal strength scales with concept frequency and the noise floor with model capacity.
>
---
#### [new 009] Readers make targeted regressions to plausible errors in reanalysis of "noisy-channel garden-path" sentences
- **分类: cs.CL**

- **简介: 该论文研究语言理解中的错误检测机制，分析读者如何通过回视定位潜在错误。属于心理语言学任务，解决如何处理含噪声的句法歧义问题。**

- **链接: [https://arxiv.org/pdf/2605.18563](https://arxiv.org/pdf/2605.18563)**

> **作者:** Thomas Hikaru Clark; Roger Levy; Edward Gibson
>
> **摘要:** A key question in psycholinguistics is how inferences about the meaning of linguistic input unfold incrementally a comprehender's mind. In this work, we study reading dynamics for ``noisy-channel garden-path'' sentences, which temporarily appear well-formed but feature late-appearing violations of expectation that can be resolved not by inferring an alternative syntactic structure, but by inferring the presence of an error. We find evidence for targeted regressions -- eye movements towards regions that are promising loci of possible errors in light of later-arriving information, showing patterns consistent with the posterior inferences of a model of noisy-channel processing with reanalysis. We discuss the implications of these findings for theories of noisy-channel language comprehension and information-theoretic explanations of reading dynamics.
>
---
#### [new 010] From Documents to Segments: A Contextual Reformulation for Topic Assignment
- **分类: cs.CL**

- **简介: 该论文属于主题建模任务，解决文档内多主题分配问题。提出SBTA方法，将主题分配到文本片段，提升主题清晰度与分析效果。**

- **链接: [https://arxiv.org/pdf/2605.17714](https://arxiv.org/pdf/2605.17714)**

> **作者:** Hoonsang Yoon; Takyoung Kim; Wonkee Lee; Ilmin Cho; Dilek Hakkani-Tür; Stanley Jungkyu Choi
>
> **备注:** Findings of ACL 2026
>
> **摘要:** Traditional topic modeling assigns a single topic to each document. In practice, however, many real-world documents, such as product reviews or open-ended survey responses, contain multiple distinct topics. This mismatch often leads to topic contamination, where unrelated themes are merged into a single topic, making it difficult to identify documents that truly focus on a specific subject. We address this issue by introducing segment-based topic allocation (SBTA), a reformulation of topic modeling that assigns topics not to entire documents, but to segments: short, coherent spans of text that each express a single theme. By modeling topical structure at the segment level, our approach yields cleaner and more interpretable topics and better supports analysis of multi-theme documents. To support systematic evaluation, we construct a SemEval-STM, a new dataset inspired by aspect-based sentiment analysis. Documents are first decomposed into topical segments using large language models (LLMs), followed by human refinement to ensure segment quality. We also propose a segment-level extension of the word intrusion task, enabling human evaluation of topical coherence at the granularity where topics are actually assigned. Across multiple models and evaluation metrics, we show that SBTA improves clustering quality and interpretability. Overall, this work provides a practical, scalable framework for fine-grained topic analysis in heterogeneous text corpora where documents naturally span multiple topics. URL: this https URL
>
---
#### [new 011] Evaluation Drift in LLM Personality Induction: Are We Moving the Goalpost?
- **分类: cs.CL**

- **简介: 该论文研究大语言模型是否能稳定表达人类人格特质。任务是评估模型人格诱导的稳定性与准确性。通过微调和问卷测试，发现模型虽减少波动，但整体准确率仍低，表明缺乏足够线索。**

- **链接: [https://arxiv.org/pdf/2605.16996](https://arxiv.org/pdf/2605.16996)**

> **作者:** Prateek Rajput; Yewei Song; Iyiola E. Olatunji; Jacques Klein; Tegawendé F. Bissyandé
>
> **备注:** 14 pages, 8 main pages, 5 figures, 4 main page figures
>
> **摘要:** Can large language models reliably express a human-like personality, or are they merely mimicking surface cues without a stable underlying profile? To investigate this, we induce personality in LLMs by fine-tuning them on the long-form essays, where each essay is associated with a target Big Five personality profile. We then evaluate the stability and fidelity of the induced personality using the IPIP-NEO questionnaire. Specifically, we ask: (i) does post-training (SFT, DPO, ORPO) stabilize questionnaire scores under prompt rephrasings, and (ii) can it induce target Big Five profiles from unguided essays? Our results demonstrate that fine-tuning consistently reduces variance in questionnaire responses across five models, directly mitigating the evaluation fragility reported in pre-trained models. However, this newfound stability reveals a more fundamental limitation: accuracy on the full five-dimensional profile remains near chance, even when single-trait scores improve. This indicates that unguided essays lack the cues needed for faithful personality expression. We therefore argue for scenario-grounded datasets or interactive elicitation that accumulates test-aligned evidence over time.
>
---
#### [new 012] Stop When Reasoning Converges: Semantic-Preserving Early Exit for Reasoning Models
- **分类: cs.CL**

- **简介: 该论文属于推理模型优化任务，解决模型过度推理浪费资源的问题。提出PUMA框架，通过检测语义冗余实现高效早停，保持答案准确性和推理连贯性。**

- **链接: [https://arxiv.org/pdf/2605.17672](https://arxiv.org/pdf/2605.17672)**

> **作者:** Dehai Min; Giovanni Vaccarino; Huiyi Chen; Yongliang Wu; Gal Yona; Lu Cheng
>
> **备注:** under review
>
> **摘要:** Large Reasoning Models (LRMs) achieve strong performance by generating long chains of thought (CoT), but often overthink, continuing to reason after a solution has already stabilized and thereby wasting tokens and increasing latency. Existing inference-time early-exit methods rely primarily on answer-level signals, such as confidence or trial-answer consistency, to decide when to stop. However, these signals mainly reflect answer readiness rather than reasoning convergence: they may trigger before the model has finished exploring or self-correcting, causing premature exits that can degrade final-answer accuracy and leave the retained reasoning chain semantically incomplete. We identify reasoning-level semantic redundancy as a complementary signal for semantic-preserving early exit: when successive steps no longer add novel progress and instead revisit established conclusions, the reasoning trajectory has likely converged. Building on this insight, we propose PUMA, a plug-and-play framework that combines a lightweight Redundancy Detector with answer-level verification. The detector flags semantically redundant candidate exits, while verification confirms whether stopping is safe, allowing PUMA to remove redundant continuation while preserving both answer accuracy and a coherent reasoning prefix. Across five LRMs and five challenging reasoning benchmarks, PUMA achieves 26.2% average token reduction while preserving accuracy and retained CoT quality. Additional experiments on code generation, zero-shot vision-language reasoning, and learned stopping-policy internalization further demonstrate that reasoning-level redundancy is a robust, transferable, and learnable signal for efficient reasoning. Our code is available at \url{this https URL}.
>
---
#### [new 013] Knowledge-to-Verification: Exploring RLVR for LLMs in Knowledge-Intensive Domains
- **分类: cs.CL**

- **简介: 该论文属于增强大语言模型推理能力的任务，解决知识密集型领域中RLVR数据不足与推理验证问题，提出K2V框架实现自动数据合成与过程验证。**

- **链接: [https://arxiv.org/pdf/2605.18261](https://arxiv.org/pdf/2605.18261)**

> **作者:** Zhonghang Yuan; Zhefan Wang; Fang Hu; Zihong Chen; Jinzhe Li; Gang Li; Jie Ying; Huanjun Kong; Songyang Zhang; Nanqing Dong
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has demonstrated promising potential to enhance the reasoning capabilities of large language models (LLMs) in domains such as mathematics and coding. However, its applications on knowledge-intensive domains have not been effectively explored due to the scarcity of high-quality verifiable data. Furthermore, current RLVR focuses solely on the correctness of final answers, leading to the limitations of flawed reasoning and sparse reward signals. In this work, we propose Knowledge-to-Verification (K2V), a framework that extends RLVR to knowledge-intensive domains through automated verifiable data synthesis, while enabling verification of the LLM's reasoning process. Extensive experiments demonstrate that K2V enhances the reasoning of LLM in knowledge-intensive domains without significantly compromising the model's general capabilities. This study also suggests that integrating automated data synthesis with reasoning verification is a promising direction to enhance model capabilities in these broader domains. Code is available at this https URL.
>
---
#### [new 014] Do LLM Agents Mirror Socio-Cognitive Effects in Power-Asymmetric Conversations?
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究LLM在权力不对称对话中的社会认知行为。旨在探讨LLM是否表现出类似人类的权力相关沟通特征，通过模拟不同职业角色的对话进行实验分析。**

- **链接: [https://arxiv.org/pdf/2605.17694](https://arxiv.org/pdf/2605.17694)**

> **作者:** Anvesh Rao Vijjini; Sagar Manjunath; Snigdha Chaturvedi
>
> **备注:** ACL 2026 (main)
>
> **摘要:** Power differences shape human communication through well documented socio cognitive effects, including language coordination, pronoun usage, authority bias, and harmful compliance. We examine whether large language models (LLMs) exhibit similar behaviors when assigned high or low status personas. Using personas from diverse professions, we simulate multi turn, power asymmetric dialogues (e.g., principal teacher, justice lawyer) and measure (i) linguistic coordination, (ii) pronoun usage, (iii) persuasion success, and (iv) compliance with unsafe requests. Our results show that LLMs show key socio cognitive effects of power, albeit with nuances and variability, linking simulated interactions to both desirable and unsafe behaviors.
>
---
#### [new 015] JSPG: Dynamic Dictionary Filtering via Joint Semantic-Pinyin-Glyph Retrieval for Chinese Contextual ASR
- **分类: cs.CL**

- **简介: 该论文属于中文上下文语音识别任务，旨在解决大词典中无关候选词导致的识别准确率下降问题。通过结合语义、拼音和字形特征，提出JSPG框架提升过滤效果。**

- **链接: [https://arxiv.org/pdf/2605.16896](https://arxiv.org/pdf/2605.16896)**

> **作者:** Shilin Zhou; Zhenghua Li
>
> **摘要:** Contextual Automatic Speech Recognition (ASR) faces challenges with large-scale keyword dictionaries, as excessive irrelevant candidates introduce noise that degrades accuracy. To address this, dynamic filtering typically uses a base ASR model to generate preliminary hypotheses, followed by semantic text retrievers to fetch a concise subset of relevant keywords. However, this approach frequently fails in Chinese ASR. Base models often produce homophonic or near-homophonic errors that preserve the phonetic cues of the target keywords but severely distort their semantic meaning, rendering standard semantic retrievers ineffective. To resolve this, we propose a filtering framework that jointly integrates Semantic, Pinyin, and Glyph features (JSPG). Pinyin effectively retrieves targets based on phonetic similarity, while glyph provides complementary structural cues to filter out numerous irrelevant homophones inherent in Chinese. To bridge the gap between character-level pinyin/glyph metrics and sequence-level filtering, we introduce an extended Smith-Waterman algorithm that computes similarity scores between the N-best hypothesis sequences and keywords. Experiments on the Aishell-1 and RWCS-NER datasets demonstrate that JSPG significantly outperforms single-feature baselines. Furthermore, downstream contextual ASR models guided by JSPG achieve substantial improvements in keyword recognition accuracy.
>
---
#### [new 016] Leveraging Graph Structure in Seq2Seq Models for Knowledge Graph Link Prediction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱链接预测任务，旨在解决传统Seq2Seq模型忽略图结构的问题。工作是提出GA-S2S框架，结合文本与图结构信息提升预测效果。**

- **链接: [https://arxiv.org/pdf/2605.18211](https://arxiv.org/pdf/2605.18211)**

> **作者:** Luu Huu Phuc; Ratan Bahadur Thapa; Mojtaba Nayyeri; Jingcheng Wu; Evgeny Kharlamov; Steffen Staab
>
> **备注:** 9 pages, 1 figure, 2 tables. Preprint of a paper accepted at the 5th Workshop on LLM-Integrated Knowledge Graph Generation from Text (TEXT2KG), co-located with ESWC 2026, May 10--14, 2026, Dubrovnik, Croatia
>
> **摘要:** We introduce Graph-Augmented Sequence-to-Sequence (GA-S2S), a novel framework that integrates a T5-small encoder-decoder with a Relational Graph Attention Network (RGAT) to improve link prediction in knowledge graphs. While existing Seq2Seq models rely solely on surface-level textual descriptions of entities and relations and at best, flatten the neighborhoods of a query entity into a single linear sequence, thereby discarding the inherent graph structure, GA-S2S jointly encodes both textual features and the full $k$-hop subgraph topology surrounding the query entity. By integrating raw encoder outputs with RGAT's relation-aware embeddings, our model captures and leverages richer multi-hop relational patterns and textual information. Our preliminary experiments on the CoDEx dataset demonstrate that GA-S2S outperforms competitive Seq2Seq-based baseline models, achieving up to a 19\% relative gain in link prediction accuracy.
>
---
#### [new 017] Weak-to-Strong Elicitation via Mismatched Wrong Drafts
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究强化学习中的策略优化问题，通过引入不匹配的错误草稿提升强学习者的性能。**

- **链接: [https://arxiv.org/pdf/2605.17314](https://arxiv.org/pdf/2605.17314)**

> **作者:** Wei Deng
>
> **摘要:** We consider whether off-policy experience from a smaller, weaker model can elicit capability in a stronger learner that on-policy RL fine-tuning (e.g., GRPO) does not reach. We find that injecting mathematically wrong drafts from a smaller but more domain-trained model -- mismatched to the current problem -- into a stronger learner's GRPO context consistently outperforms standard on-policy GRPO on held-out MATH-500 and out-of-distribution AIME 2025/2026. Concretely, we use Mathstral-7B as the learner, Qwen2.5-Math-1.5B as the draft model, 8.8K Level 3--5 MATH problems (with MATH-500 held out), and train with Dr. GRPO. Mismatch is an active ingredient: shuffling drafts to mismatched problems while holding everything else constant yields $+1.62$pp on MATH-500 (greedy pass@1) over the matched-wrong variant ($n=10$ seeds, $p=0.0015$, Welch's $t$). In fact, the mismatched-wrong variant leads all other variants we tested on MATH-500 across both greedy pass@1 and sampling pass@$k$. On out-of-distribution AIME 2025 and 2026, the mismatched-wrong variant uniquely lifts pass@$k$ above both Mathstral-7B (in its native [INST] format) and the Qwen2.5-Math-1.5B draft model at every sample budget from $k=1$ to $k=1024$ across 2 seeds ($+14.2$pp on 2025 and $+9.0$pp on 2026 at pass@1024 over Mathstral-7B), and at pass@1024 also leads no-draft, matched-wrong, and mismatched-correct variants on both years. All variants use the same prompt with no draft injection at test time. The recipe -- trained on a single GPU with no SFT, no reward models, no synthesized data, and no produce-critique-revise inner loop -- reaches 71.98% MATH-500 on Mathstral-7B-v0.1, the highest published result on this model to our knowledge, surpassing the heavier WizardMath pipeline at 70.9% on full MATH (SFT + PPO with process/instruction reward models).
>
---
#### [new 018] Language-Switching Triggers Take a Latent Detour Through Language Models
- **分类: cs.CL**

- **简介: 该论文研究语言模型中的后门攻击，分析触发器如何引导模型输出。任务是理解触发机制，解决安全漏洞问题，通过分解电路揭示其工作原理。**

- **链接: [https://arxiv.org/pdf/2605.18646](https://arxiv.org/pdf/2605.18646)**

> **作者:** Francis Kulumba; Wissam Antoun; Théo Lasnier; Benoît Sagot; Djamé Seddah
>
> **备注:** 15 pages, 16 figures. Under review
>
> **摘要:** Backdoor attacks on language models pose a growing security concern, yet the internal mechanisms by which a trigger sequence hijacks model computations remain poorly understood. We identify a circuit underlying a language-switching backdoor in an 8B-parameter autoregressive language model, where a three-word Latin trigger (nine tokens) redirects English output to French. We decompose the circuit into three phases: (1) distributed attention heads at early layers compose the trigger tokens into the last sequence position; (2) the resulting signal propagates through mid-layers in a subspace orthogonal to the model's natural language-identity direction; (3) the MLP at the final layer converts this latent signal into French logits. The entire circuit flows through a serial bottleneck at a single position: corrupting that position at any layer entirely mitigate the trigger but also hinder the model's capabilities. The orthogonal latent encoding suggests that defenses that search for language-like signals in intermediate representations would miss this trigger entirely.
>
---
#### [new 019] Mixture of Experts for Low-Resource LLMs
- **分类: cs.CL**

- **简介: 该论文研究低资源语言在Mixture-of-Experts模型中的路由问题，分析了预训练不足导致的专家使用失衡，并通过实验提出改进方法。**

- **链接: [https://arxiv.org/pdf/2605.17598](https://arxiv.org/pdf/2605.17598)**

> **作者:** Ori Bar Joseph; Smadar Arvatz; Noam Kayzer; Dan Revital; Sarel Weinberger
>
> **摘要:** Mixture-of-Experts (MoE) architectures enable efficient model scaling, yet expert routing behavior across underrepresented languages remains poorly understood. We analyze routing dynamics in two architecturally distinct MoE models -- a pure Transformer (Qwen3-30B-A3B) and a hybrid Mamba-Transformer (Nemotron-3-Nano-30B-A3B) -- using Hebrew as a morphologically rich, low-resource testbed. Both pre-trained models exhibit \emph{deep-layer routing collapse}: usage entropy drops sharply in final layers and tokens concentrate on a narrow expert subset, a pattern largely absent for English. Continual pre-training (CPT) on balanced bilingual data substantially corrects this imbalance, increasing entropy and shifting routing toward shared, language-agnostic experts; supervised fine-tuning (SFT) alone achieves less complete correction. Extending the analysis to Japanese reveals quantitatively consistent collapse signatures, providing cross-linguistic evidence that the phenomenon is a systematic consequence of pre-training underrepresentation rather than any language-intrinsic property. Routing improvements correlate with consistent downstream benchmark gains, positioning routing entropy and expert specialization as principled diagnostics for multilingual capacity in MoE systems.
>
---
#### [new 020] AgentKernelArena: Generalization-Aware Benchmarking of GPU Kernel Optimization Agents
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出AgentKernelArena，用于评估AI代码代理在GPU内核优化中的泛化能力，解决现有基准测试不足的问题。**

- **链接: [https://arxiv.org/pdf/2605.16819](https://arxiv.org/pdf/2605.16819)**

> **作者:** Sharareh Younesian; Wenwen Ouyang; Sina Rafati; Mehdi Rezagholizadeh; Sharon Zhou; Ji Liu; Yue Liu; Yuchen Yang; Hao Li; Ziqiong Liu; Dong Li; Vikram Appia; Zhenyu Gu; Emad Barsoum
>
> **摘要:** GPU kernel optimization is increasingly critical for efficient deep learning systems, but writing high-performance kernels still requires substantial low-level expertise. Recent AI coding agents can iteratively read code, invoke compilers and profilers, and refine implementations, yet existing kernel benchmarks evaluate single LLM calls rather than full agent workflows, and none include both kernel-to-kernel optimization and unseen-configuration generalization testing. We present AgentKernelArena, an open-source benchmark for measuring AI coding agents on GPU kernel optimization. The benchmark contains 196 tasks spanning HIP-to-HIP optimization, Triton-to-Triton optimization, and PyTorch-to-HIP translation, and evaluates complete agent workflows in isolated workspaces using gated compilation, correctness, and performance checks, centralized scoring and an unseen-configuration generalization protocol that tests whether optimizations transfer to input configurations the agent never observed. Across production agents including Cursor Agent, Claude Code, and Codex Agent, we find near-perfect compilation and high correctness rates on most task categories, with the strongest configurations achieving mean speedups of up to 6.89x on PyTorch-to-HIP, 6.69x on HIP-to-HIP, and 2.13x on Triton-to-Triton tasks. Our unseen-configuration evaluation shows that HIP-to-HIP and Triton-to-Triton optimizations largely transfer to unseen input shapes, while PyTorch-to-HIP exhibits substantial correctness drops, indicating that agents generating kernels from scratch frequently hardcode shape-specific assumptions. AgentKernelArena is designed as a modular, extensible framework for rigorous evaluation of agentic GPU kernel optimization across agents, tasks, and hardware targets.
>
---
#### [new 021] E-PMQ: Expert-Guided Post-Merge Quantization with Merged-Weight Anchoring
- **分类: cs.CL**

- **简介: 该论文提出E-PMQ框架，解决模型合并后量化效果不佳的问题，通过专家引导和权重锚定提升低比特部署性能。**

- **链接: [https://arxiv.org/pdf/2605.16882](https://arxiv.org/pdf/2605.16882)**

> **作者:** Wenjun Wang; Yanggan Gu; Shuo Cai; Yuanyi Wang; Pengkai Wang; Jianmin Wu; Hongxia Yang
>
> **摘要:** Low-resource deployment constraints have made model quantization essential for deploying neural networks while preserving performance. Meanwhile, model merging has become an increasingly practical low-resource strategy for integrating multiple task- or domain-specialized experts into a single model without joint training or multi-model serving. Together, quantization and model merging enable an efficient low-resource deployment pipeline by integrating multiple experts into one low-bit model. We formulate this setting as Post-Merge Quantization (PMQ). We show that directly applying post-training quantization (PTQ) to a merged model is unreliable because two distinct deviations are coupled: the quantization deviation introduced by low-bit reconstruction and the expert-relative merging deviation inherited from model merging. To mitigate these deviations, we propose E-PMQ, an expert-guided PMQ framework that uses source expert weights to provide expert- guided output targets during layer-wise calibration, together with merged-weight anchoring to stabilize the calibration and preserve the integrated behavior of the merged model. On CLIP-ViT-B/32 eight-task merging, E-PMQ improves 4-bit GPTQ from 65.0% to 73.6% under Task Arithmetic and from 69.1% to 74.8% under TIES-Merging. On harder settings, E-PMQ improves GPTQ from 34.8% to 76.7% on 20-task CLIP-ViT-L/14 and from 78.26% to 83.34% on FLAN-T5- base GLUE. These results demonstrate that E-PMQ enables effective post-merge quantization and low-bit deployment.
>
---
#### [new 022] STT-Arena: A More Realistic Environment for Tool-Using with Spatio-Temporal Dynamics
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出STT-Arena，一个用于评估和提升大语言模型在动态环境中适应性任务的基准。旨在解决模型在时空动态变化下无法有效调整策略的问题，通过构建真实交互任务并分析失败模式，改进模型的适应能力。**

- **链接: [https://arxiv.org/pdf/2605.18548](https://arxiv.org/pdf/2605.18548)**

> **作者:** Tingfeng Hui; Hao Xu; Pengyu Zhu; Hongsheng Xin; Kun Zhan; Sen Su; Chunxiao Liu; Ning Miao
>
> **备注:** Work in progress
>
> **摘要:** Large language models (LLMs) deployed in real-world agentic applications must be capable of replanning and adapting when mid-task disruptions invalidate their prior decisions. Existing dynamic benchmarks primarily measure whether LLMs can detect temporal changes in a timely manner, leaving the complementary challenge of adaptive replanning under spatio-temporal dynamics largely unexplored. We introduce STT-Arena (Spatio-Temporal Tool-Use Arena), a benchmark of 227 high-quality interactive tasks spanning nine spatio-temporal conflict types and four solvability levels. Each task is grounded in a realistic, executable environment equipped with injected spatio-temporal triggers that can abruptly invalidate an ongoing plan, forcing the model to detect the state shift and construct a revised execution strategy. Extensive evaluation of frontier LLMs reveals that even the SOTA proprietary models, including Claude-4.6-Opus, achieves less than 40\% overall accuracies, highlighting the fundamental difficulty of spatio-temporal dynamic reasoning. Systematic analysis of failure trajectories uncovers three recurring error modes of existing models: Stale-State Execution, Misdiagnosis of Dynamic Triggers, and Missing Post-Adaptation Verification. Guided by these findings, we propose an iterative trajectory refinement technique that eliminates these failure patterns from training data, and combine it with online RL to produce STT-Agent-4B which outperforms frontier LLMs on STT-Arena.
>
---
#### [new 023] Learning Transferable Topology Priors for Multi-Agent LLM Collaboration Across Domains
- **分类: cs.CL**

- **简介: 该论文属于多智能体语言模型协作任务，解决跨领域协作效率低的问题。通过学习可迁移的拓扑先验，提升协作结构生成效率，减少计算资源消耗。**

- **链接: [https://arxiv.org/pdf/2605.17359](https://arxiv.org/pdf/2605.17359)**

> **作者:** Taolin Zhang; Zijie Zhou; Jiuheng Wan; Tingyuan Hu; Chengyu Wang; Xiaofeng He; Richang Hong
>
> **摘要:** Large language model (LLM)-based multi-agent systems have shown strong potential for complex reasoning by coordinating specialized agents through structured communication. However, existing topology-evolution methods typically construct or optimize a collaboration topology for each query from scratch, leading to substantial online search overhead, high inference-time token consumption, and limited scalability in multi-domain settings. We propose TopoPrior, a framework for learning transferable topology priors for multi-agent LLM collaboration across domains. Rather than repeatedly searching for effective collaboration structures online, TopoPrior learns reusable topology priors from reference collaboration graphs collected offline from multiple domains and uses them to generate query-conditioned initial collaboration graphs for downstream refinement. By shifting part of topology search from per-query online optimization to offline prior learning, TopoPrior amortizes search cost while remaining compatible with existing topology-evolution backbones. Technically, TopoPrior contains two key components. First, a transferable topology prior learning module employs a conditional variational graph framework to capture reusable structural regularities across domains in a latent space. Second, a query-conditioned latent adaptation module introduces adversarial alignment to reduce unnecessary domain discrepancy while preserving query-relevant structural variation. Experiments on multi-domain reasoning benchmarks show that TopoPrior consistently improves several heterogeneous topology-evolution backbones while reducing online inference-time token usage, with only modest additional trainable parameters. These results suggest that transferable topology initialization is an effective and lightweight mechanism for improving the efficiency of multi-agent LLM collaboration across domains.
>
---
#### [new 024] Presupposition and Reasoning in Conditionals: A Theory-Based Study of Humans and LLMs
- **分类: cs.CL**

- **简介: 该论文研究条件句中的预设投射问题，比较人类与大语言模型的判断差异。通过行为实验和模型评估，揭示模型可能依赖表面模式而非语用能力。任务属于自然语言理解与语用研究。**

- **链接: [https://arxiv.org/pdf/2605.18352](https://arxiv.org/pdf/2605.18352)**

> **作者:** Tara Azin; Yongan Yu; Raj Singh; Olessia Jouravlev
>
> **备注:** To appear in the Proceedings of CoNLL 2026, colocated with ACL 2026
>
> **摘要:** Presupposition projection in conditionals is central to theories of meaning and pragmatics, yet it remains largely unevaluated in large language models. We address this gap through a parallel behavioral study comparing human judgments and LLM predictions on a normed dataset of conditional sentences that controls the relation between the antecedent and the projected presupposition. We collect likelihood ratings from 120 participants and four LLMs under matched contextual conditions. Results show that humans integrate probabilistic and pragmatic cues in their judgment, whereas LLMs show variable alignment with human patterns. Using a linguistically motivated checklist within an LLM-as-a-Judge framework, we further evaluate model reasoning. We observe models that best match human ratings often lack coherent pragmatic reasoning, while models with stronger reasoning produce less human-like judgments. These findings suggest that LLMs' performance on such tasks may result from surface pattern matching rather than pragmatic competence. Our findings highlight the importance of benchmarks grounded in linguistic theory for comparing humans and models.
>
---
#### [new 025] Sometin Beta Pass Notin (SBPN): Improving Multilingual ASR for Nigerian Languages via Knowledge Distillation
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，旨在提升尼日利亚语言的多语种ASR性能。针对数据稀缺等问题，提出知识蒸馏方法，构建SBPN模型，显著降低错误率。**

- **链接: [https://arxiv.org/pdf/2605.17710](https://arxiv.org/pdf/2605.17710)**

> **作者:** Sewade Ogun
>
> **备注:** 25 pages
>
> **摘要:** Although modern multilingual Automatic Speech Recognition (ASR) systems support several Nigerian languages, their performance consistently lags behind high-resource languages like English and French. Nigerian languages present unique modelling hurdles, including acute data scarcity, inconsistent orthography, tonal diacritics, diverse accents, frequent code-switching, and localized named entities. To address these challenges, we developed a multilingual ASR framework utilizing a two-stage distillation process. First, we employ student-teacher knowledge distillation from existing monolingual models, conditioned on robust language-specific N-gram language models. Second, we perform iterative self improvement using pseudo-labelled data to further refine accuracy. Our method significantly bridges the performance gap, achieving on average a relative Word Error Rate (WER) reduction of 29 % over monolingual baselines. Our models also outperform state-of-the-art multilingual models across major benchmarks, including Common Voice and Fleurs. We introduce Sometin Beta Pass Notin (SBPN), a foundational multilingual ASR model covering Yorùbá, Hausa, Igbo, Nigerian Pidgin, and Nigerian English. SBPN is released in two sizes: SBPN-Base (120 M parameters) and SBPN-Large (600 M parameters). By releasing these as open foundation models, we aim to provide ASR resources for further research into the rich phonetic and cultural landscape of the region.
>
---
#### [new 026] Semantic Reranking at Inference Time for Hard Examples in Rhetorical Role Labeling
- **分类: cs.CL**

- **简介: 该论文属于 rhetorical role labeling 任务，旨在解决语言模型在困难示例上表现不稳定的问题。提出 RISE 框架，通过语义重排序提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2605.18007](https://arxiv.org/pdf/2605.18007)**

> **作者:** Anas Belfathi; Nicolas Hernandez; Laura Monceaux; Warren Bonnard; Richard Dufour
>
> **备注:** Accepted at ACL 2026 (Main Conference)
>
> **摘要:** Rhetorical Role Labeling (RRL) assigns a functional role to each sentence in a document and is widely used in legal, medical, and scientific domains. While language models (LMs) achieve strong average performance, they remain unreliable on hard examples, where prediction confidence is low. Existing approaches typically handle uncertainty implicitly and treat labels as discrete identifiers, overlooking the semantic information encoded in label names. We introduce RISE, an inference-time semantic reranking framework that leverages label semantics to refine predictions on hard instances. RISE automatically identifies low-confidence predictions and reranks model outputs using contrastively learned label representations, without retraining or modifying the underlying model. Experiments on eight domain-specific RRL datasets with seven LMs, including encoder-based and causal architectures, show an average gain of +9.15 macro-F1 points on hard examples. For explainability, we further propose manual hardness annotations to study difficulty from both model and human perspectives, revealing a moderate agreement with Cohen's kappa = 0.40.
>
---
#### [new 027] PQR: A Framework to Generate Diverse and Realistic User Queries that Elicit QA Agent Failures
- **分类: cs.CL**

- **简介: 该论文提出PQR框架，用于生成多样且真实的用户查询，以检测QA代理的失败。任务是评估LLM代理，解决如何有效发现代理失败问题，通过迭代模块生成更具现实性的测试用例。**

- **链接: [https://arxiv.org/pdf/2605.16551](https://arxiv.org/pdf/2605.16551)**

> **作者:** Yunan Lu; Luigi Liu; Omar Yahia; Arpit Sharma; Zhou Yu
>
> **摘要:** Evaluating LLM-based agents remains challenging because identifying meaningful failure cases often requires substantial human effort to design realistic test scenarios. Prior works primarily focus on automatically discovering agent failures induced by adversarial users, while overlooking queries with real user intents that also trigger agent failures. We introduce PQR, a framework that not only surfaces agent failures with respect to specific objectives (e.g., helpfulness, safety, etc.) but also resembles real users' intents. PQR operates through an iterative interaction between two complementary modules. The query refinement module performs rewrites to explore diverse query variations, while the prompt refinement module uses prior feedback to derive new objective-violating strategies and realism policies for refining prompts, which in turn generate failure-triggering yet realistic queries. We evaluate PQR on detecting an e-commerce QA agent's unhelpful responses. Our method uncovers 23% - 78% more unhelpful responses, and our generated queries are more diverse and realistic compared to previous methods.
>
---
#### [new 028] Systematic Evaluation of the Quality of Synthetic Clinical Notes Rephrased by LLMs at Million-Note Scale
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床文本生成评估任务，旨在解决LLM生成临床笔记的质量问题。通过大规模分析，评估其信息保留与错误类型，提出改进方法。**

- **链接: [https://arxiv.org/pdf/2605.17775](https://arxiv.org/pdf/2605.17775)**

> **作者:** Jinghui Liu; Sarvesh Soni; Anthony Nguyen
>
> **摘要:** Large language models (LLMs) can generate or synthesize clinical text for a wide range of applications, from improving clinical documentation to augmenting clinical text analytics. Yet evaluations typically focus on a narrow aspect -- such as similarity or utility comparisons -- even though these aspects are complementary and best viewed in parallel. In this study, we aim to conduct a systematic evaluation of LLM-generated clinical text, which includes intrinsic, extrinsic, and factuality evaluations of synthetic clinical notes rephrased from MIMIC databases at million-note scale. Our analysis demonstrates that synthetic notes preserve core clinical information and predictive utility for coarse-grained tasks despite substantial linguistic changes, but lose fine-grained details for task like ICD coding. We show this loss of detail can be substantially mitigated by rephrasing notes by chunks rather than by the whole note, but at the cost of reduced factual precision under incomplete context. Through fact-checking and error analysis, we further find that synthesis errors are dominated by misinterpretation of clinical context, alongside temporal confusion, measurement errors, and fabricated claims. Finally, we show that the synthetic notes -- despite their task-agnostic nature -- can effectively augment task-specific training for rare ICD codes.
>
---
#### [new 029] Beyond Transcripts: Iterative Peer-Editing with Audio Unlocks High-Quality Human Summaries of Conversational Speech
- **分类: cs.CL**

- **简介: 该论文属于语音摘要任务，旨在解决缺乏高质量基准的问题。通过对比音频与文本摘要及LLM输出，发现迭代同行编辑可提升音频摘要质量。**

- **链接: [https://arxiv.org/pdf/2605.17652](https://arxiv.org/pdf/2605.17652)**

> **作者:** Kaavya Chaparala; Thomas Thebaud; Jesús Villalba López; Laureano Moro-Velazquez; Peter Viechnicki; Najim Dehak
>
> **备注:** Accepted in LREC 2026
>
> **摘要:** There are not enough established benchmarks for the task fo speech summarization. Creating new benchmarks demands human annotation, as LLMs could embed systemic errors and bias into datasets. We test ten annotation workflows varying input modality (audio, transcript, or both) and the inclusion of editing (self or peer-editing) to investigate potential quality tradeoffs from using human annotators to summarize audio. We compare human audio-based summaries to human transcript-based summaries to track the impact of the different information modalities on summary quality. We also compare the human outputs against four LLM benchmarks (three text, one audio) to examine whether human-written summaries are less informative than highly fluent automated outputs. We find that audio-based summaries are less informative and more compressed than transcript summaries. However, iterative peer-editing with audio mitigates this difference, enabling audio-based summaries to be as informative as their transcript counterparts and LLM summaries. These findings validate iterative peer-editing among human annotators for the creation of benchmarks informed by both lexical and prosodic information. This enables crucial dataset collection even in setting where transcripts are unavailable.
>
---
#### [new 030] Internalizing Tool Knowledge in Small Language Models via QLoRA Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决小模型工具使用效率低的问题。通过QLoRA微调，将工具知识内化到小模型中，减少推理时的上下文负担，提升规划质量。**

- **链接: [https://arxiv.org/pdf/2605.17774](https://arxiv.org/pdf/2605.17774)**

> **作者:** Yuval Shemla; Ayal Yakobe; Tanmay Agarwal
>
> **摘要:** Large language models are increasingly used as planning components in agentic systems, but current tool-use pipelines often require full tool schemas to be included in every prompt, creating substantial token overhead and limiting the practicality of smaller models. This paper investigates whether tool-use knowledge can be internalized into small language models through parameter-efficient fine-tuning, enabling structured planning without explicit tool descriptions at inference time. Using AssetOpsBench as the primary benchmark, we fine-tune Gemma 4 E4B and Qwen3-4B with 8-bit QLoRA on approximately 1,700 tool-use examples spanning tool knowledge, question-to-plan mappings, and execution-style traces. We evaluate the resulting models under description-free inference, where the prompt omits the tool catalog entirely. The fine-tuned models outperform an informed unfine-tuned baseline that receives full tool descriptions, reducing input length by 82.6\% while improving structural and LLM-judge planning scores. In the best Gemma run, the model achieves an AT-F1 of 0.65 and an overall judge score of 3.88, compared with 0.47 and 2.88 for the informed baseline. Qwen3-4B achieves a strong overall judge score of 3.78 while using 62\% less memory and running 2.5$\times$ faster than Gemma, though it also exhibits greater catastrophic forgetting on general multiple-choice benchmarks. Additional ablations show that LoRA rank controls a quality--retention trade-off, with $r=32$ maximizing planning quality and smaller ranks preserving more general knowledge. These results suggest that, for fixed tool catalogs, QLoRA fine-tuning can shift tool knowledge from prompt context into model weights, substantially reducing inference overhead while maintaining or improving tool-planning quality.
>
---
#### [new 031] How Loud Rumbles Hit Newsstands: A Data Analysis of Coverage and Spatial Bias in German News about Landslides Around the World
- **分类: cs.CL**

- **简介: 该论文分析德国媒体报道全球滑坡事件的覆盖情况，旨在揭示媒体关注的地域偏差。任务属于媒体分析与灾害研究，解决新闻报道不均问题，通过数据筛选、地理定位和比较分析提供见解。**

- **链接: [https://arxiv.org/pdf/2605.18105](https://arxiv.org/pdf/2605.18105)**

> **作者:** Brielen Madureira; Andreas Niekler; Marc Keuschnigg; Mariana Madruga de Brito
>
> **备注:** Work in progress
>
> **摘要:** Landslides often hit newsstands due to their destructive and potentially fatal effects. News are a valuable source of information for creating or enriching disaster databases and for expediting media-based studies of the dynamics of media attention. To accomplish that, news datasets must be filtered, geolocated and validated. This paper focuses on how landslides around the world are reported in German newspapers. We analyse almost 60k news articles about 5.5k news events in a 25-year period, compare it with external measures of countries' susceptibility to landslides and provide insights, e.g.~the overreporting of Southern and Western Europe, to foment further studies on inequalities in media attention to international disasters.
>
---
#### [new 032] Full Attention Strikes Back: Transferring Full Attention into Sparse within Hundred Training Steps
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型优化任务，旨在解决长文本推理中全注意力机制效率低的问题。通过利用模型内在稀疏性，提出RTPurbo方法，在少量训练步骤内实现高效稀疏注意力。**

- **链接: [https://arxiv.org/pdf/2605.16928](https://arxiv.org/pdf/2605.16928)**

> **作者:** Yanke Zhou; Yiduo Li; Hanlin Tang; Maohua Li; Kan Liu; Lan Tao; Lin Qu; Yuan Yao; Xiaoxing Ma
>
> **备注:** 20 pages, 9 figures
>
> **摘要:** Long-context inference in large language models is bottlenecked by the quadratic cost of full attention. Existing efficient alternatives often rely either on native sparse training or on heuristic token eviction, creating an undesirable trade-off among efficiency, training cost, and accuracy. In this work, we show that full-attention LLMs are already intrinsically sparse and can be transformed into highly sparse models with only minimal adaptation. Our approach is built on three observations: (1) only a small subset of attention heads truly requires full long-context processing; (2) long-range retrieval is governed primarily by a low-dimensional subspace, allowing relevant tokens to be retrieved efficiently with a 16-dimensional indexer; and (3) the useful token budget is strongly query-dependent, making dynamic top-$p$ selection more suitable than fixed top-$k$ sparsification. Based on these insights, we propose RTPurbo, which retains the full KV cache only for retrieval heads and introduces a lightweight token indexer for sparse attention. By exploiting the model's intrinsic sparsity, RTPurbo achieves sparsification with only a few hundred training steps. Experiments on long-context benchmarks and reasoning tasks show that RTPurbo preserves near-lossless accuracy while delivering substantial efficiency gains, including up to a 9.36$\times$ prefill speedup at 1M context and about a 2.01$\times$ decode speedup. These results suggest that strong sparse inference can be obtained from standard full-attention training without expensive native sparse pretraining.
>
---
#### [new 033] VerifyMAS: Hypothesis Verification for Failure Attribution in LLM Multi-Agent Systems
- **分类: cs.CL**

- **简介: 该论文属于故障归因任务，解决LLM-MAS中全局故障识别问题。提出VerifyMAS框架，通过假设验证实现更高效的故障定位与分析。**

- **链接: [https://arxiv.org/pdf/2605.17467](https://arxiv.org/pdf/2605.17467)**

> **作者:** Hezhe Qiao; Hanghang Tong; Ee-Peng Lim; Bing Liu; Guansong Pang
>
> **备注:** 22 pages
>
> **摘要:** Large language model-driven multi-agent systems (LLM-MAS) excel at complex tasks, yet unreliable agents remain a key bottleneck to system-level reliability. Automatic failure attribution is therefore critical, but existing approaches, such as direct prediction of agent-error pairs and agent-first failure attribution, rely on local logs of agents and miss global failures that only manifest over full interaction trajectories, such as cross-step inconsistencies and inter-agent coordination errors. Moreover, directly predicting failures induces a large combinatorial search space, hindering fine-grained attribution. To address these challenges, we propose VerifyMAS, a hypothesis verification framework for agent failure attribution. Instead of directly predicting faulty agents and error types, VerifyMAS formulates and verifies failure hypotheses against full trajectories. This verification-based approach decomposes attribution into trajectory-level error validation and fine-grained agent localization, providing an error-first attribution approach that captures global failure patterns while substantially reducing the search space. We further introduce a hypothesis-based data construction strategy grounded in a structured error taxonomy and fine-tune a specialized LLM verifier model for trajectory-level failure verification and agent attribution. Experiments on Aegis-Bench and Who&When show that VerifyMAS consistently improves diverse backbone models, including open-source Qwen and API-based GPT models, outperforming prior methods without sacrificing inference efficiency for long multi-agent trajectories.
>
---
#### [new 034] AMATA: Adaptive Multi-Agent Trajectory Alignment for Knowledge-Intensive Question Answering
- **分类: cs.CL**

- **简介: 该论文提出AMATA框架，解决知识密集型问答中的事实一致性问题。通过多智能体协作与外部知识融合，提升回答的准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2605.17352](https://arxiv.org/pdf/2605.17352)**

> **作者:** Taolin Zhang; Dongyang Li; Chen Chen; Qizhou Chen; Jiuheng Wan; Xiaofeng He; Chengyu Wang; Richang Hong
>
> **摘要:** Despite substantial advances in large language models (LLMs), generating factually consistent responses for knowledge-intensive question answering remains challenging. These difficulties are primarily due to hallucinations and the limitations of LLMs in bridging long-tail knowledge gaps. To address this, we propose AMATA, an Adaptive Multi-Agent Trajectory Alignment framework that dynamically integrates external knowledge to improve response interpretability and factual grounding. Our architecture leverages six specialized agents that collaboratively perform structured actions for complex question reasoning. We formalize multi-agent collaboration with external tools as a trajectory preference alignment problem, incorporating question-aware agent customization and inter-agent preference harmonization. AMATA introduces two principal innovations: (1) Intra-Trajectory Preference Learning, which learns objective-oriented preferences to prioritize critical agents, and (2) Inter-Agent Dependency Learning, which captures cross-agent tool dependencies through a novel dependency-aware direct preference optimization technique. Empirical results show that AMATA consistently outperforms baseline approaches, knowledge-augmented frameworks, and LLM-based trajectory systems on five established knowledge-intensive QA benchmarks. Further analysis demonstrates the efficiency of our method in reducing token consumption.
>
---
#### [new 035] SocialMemBench: Are AI Memory Systems Ready for Social Group Settings?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI记忆系统任务，旨在解决多用户社交场景下的记忆不足问题。通过构建SocialMemBench基准，评估现有系统的缺陷。**

- **链接: [https://arxiv.org/pdf/2605.17789](https://arxiv.org/pdf/2605.17789)**

> **作者:** Olukunle Owolabi
>
> **摘要:** Memory systems for AI assistants were built for single-user dialogue and fail characteristically when applied to multi-party social group settings. This gap matters for the social assistants being built today: group-acting agents embedded in chat platforms, and proactive personal-assistant agents whose holistic model of a user must include their social context. Existing memory benchmarks evaluate dyadic or workplace dialogue; none targets multi-party social groups, where memory must anchor facts in shared history rather than professional roles, separate group norms from individual exceptions, and correctly attribute even after member departure. We introduce SocialMemBench, a benchmark of human-verified synthetic social group networks across five archetypes (close friends, family, recreational, interest community, acquaintance network) and three group-size tiers (4-30 members), with 430 personas and 7,355 conversation turns, yielding 1,031 QA pairs across nine question categories. Each category isolates an architectural capability, and the five failure modes (single-stream conflation, temporal-state overwrite, entity merging at scale, missing cross-persona knowledge, norm-individual conflation) are testable hypotheses; our two research probes Subject-Mem and SMG provide evidence on two, three remain open. A full-context Gemini 2.5 Flash reference reaches only 0.721 against a blind-critic reasoning-model mean of 0.98 on small networks, indicating the benchmark is genuinely difficult even with complete access to the conversation. Across all 43 networks, the four open-source memory frameworks evaluated (Mem0, LangMem, Graphiti, Cognee) cluster in the 0.12-0.18 question-weighted range with overlapping 95% CIs, well below an uncompressed retrieval reference of 0.345 and a matched-answerer full-context reference of 0.369 (GPT-4o-mini). Current memory systems show a measurable gap.
>
---
#### [new 036] The Scaling Laws of Skills in LLM Agent Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM代理系统中技能的扩展规律，解决技能库结构对代理性能影响的问题。通过分析15个模型和大量数据，发现两个耦合定律，并提出优化方法提升执行效果。**

- **链接: [https://arxiv.org/pdf/2605.16508](https://arxiv.org/pdf/2605.16508)**

> **作者:** Charles Chen; Qiming Yu; Yuhang Gu; Zhuoye Huang; Hanjing Li; Hongyu Liu; Simin Liu; Jinhao Liu; Dengyun Peng; Jiangyi Wang; Zheng Yan; Fanqing Meng; Ethan Qin; Carl Che; Mengkang Hu
>
> **备注:** Technical Report
>
> **摘要:** As agent systems scale, skills accumulate into large reusable libraries, yet their scaling laws remain poorly understood. Across 15 frontier LLMs, 1,141 real-world skills, and over 3M routing or execution decisions, we identify two coupled laws. Routing law: single-step routing accuracy decays logarithmically with library size ($R^2{>}0.97$ for all models), with errors progressing from local skill competition to cross-family drift and capture by overly general "black-hole skills". Execution law: before state realization, joint routing is approximately multiplicative, whereas correct execution can improve difficult downstream decisions by about $4{\times}$. A single parameter, the routing logarithmic decay slope $b$, couples the two laws: routing-side fits predict execution-side rescue across models, showing that the same library property controls both pre-execution collapse and downstream recoverability. The laws are actionable: law-guided optimization raises held-out routing accuracy from 71.3% to 91.7%, reduces hijack from 22.4% to 4.1%, and transfers directionally to downstream ClawBench and ClawMark execution settings, improving mean pass rate from 49.3% to 61.6% on ClawBench and from 28.4% to 34.5% on ClawMark. These results show that agent performance depends not only on model capability, but also on the structure, granularity, and exposure policy of the skill library.
>
---
#### [new 037] Hybrid Feature Combinations with CNN for Bangla Fake News Classification
- **分类: cs.CL**

- **简介: 该论文属于 Bangla 假新闻分类任务，旨在提升假新闻检测效果。通过结合多种特征（语义、统计、字符级）并使用 CNN 模型，显著提高了召回率和 F1 分数。**

- **链接: [https://arxiv.org/pdf/2605.17481](https://arxiv.org/pdf/2605.17481)**

> **作者:** Md Gulzar Hussain; Babe Sultana; Md Rinku Ali
>
> **备注:** Already accepted and presented in the 3rd International Conference on Big Data, IoT and Machine Learning (BIM 2025)
>
> **摘要:** Nowadays, people in Bangladesh frequently rely on the internet and social media for daily news instead of traditional newspapers. However, the spread of false Bangla news through these platforms poses risks and challenges to the credibility of authentic media. Although several studies have been conducted on detecting Bangla fake news, there is still significant room for improvement in this area. To assist people, this research explores the effectiveness of feature selection approaches in identifying appropriate features, such as semantic, statistical, and character-level features, or their combinations, on the BanFakeNews-2.0 dataset for detecting Bangla fake news using a CNN model. In this paper, key findings reveal that combining multiple features significantly improves recall and F1-scores compared to using individual features alone. The code for this research can be availed here, this https URL\_FNews\this http URL.
>
---
#### [new 038] EvoMemBench: Benchmarking Agent Memory from a Self-Evolving Perspective
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于LLM代理记忆研究任务，旨在解决现有基准未系统评估记忆机制的问题。提出EvoMemBench，从自进化视角评估不同记忆方法。**

- **链接: [https://arxiv.org/pdf/2605.18421](https://arxiv.org/pdf/2605.18421)**

> **作者:** Yuyao Wang; Zhongjian Zhang; Mo Chi; Kaichi Yu; Yuhan Li; Miao Peng; Bing Tong; Chen Zhang; Yan Zhou; Jia Li
>
> **摘要:** Recent benchmarks for Large Language Model (LLM) agents mainly evaluate reasoning, planning, and execution. However, memory is also essential for agents, as it enables them to store, update, and retrieve information over time. This ability remains under-evaluated, largely because existing benchmarks do not provide a systematic way to assess memory mechanisms. In this paper, we study agent memory from a self-evolving perspective and introduce EvoMemBench, a unified benchmark organized along two axes: memory scope (in-episode vs. cross-episode) and memory content (knowledge-oriented vs. execution-oriented). We compare 15 representative memory methods with strong long-context baselines under a standardized protocol. Results show that current memory systems are still far from a general solution: long-context baselines remain highly competitive, memory helps most when the current context is insufficient or tasks are difficult, and no single memory form works consistently across all settings. Retrieval-based methods remain strong for knowledge-intensive settings, whereas procedural and long-term memory methods are more effective for execution-oriented tasks when their stored experience matches the task structure. We hope EvoMemBench facilitates future research on more effective memory systems for LLM-based agents. Our code is available at this https URL.
>
---
#### [new 039] PROTEA: Offline Evaluation and Iterative Refinement for Multi-Agent LLM Workflows
- **分类: cs.CL; cs.AI; cs.HC; cs.SE**

- **简介: 该论文提出PROTEA，用于多智能体大语言模型工作流的离线评估与迭代优化。针对调试困难的问题，通过评分、定位瓶颈和提示修改实现改进。**

- **链接: [https://arxiv.org/pdf/2605.18032](https://arxiv.org/pdf/2605.18032)**

> **作者:** Kazuki Kawamura; Satoshi Waki; Kei Tateno
>
> **备注:** 9 pages, 3 figures, 1 table. To appear in Proceedings of ACL 2026 System Demonstrations
>
> **摘要:** Multi-agent LLM workflows -- systems composed of multiple role-specific LLM calls -- often outperform single-prompt baselines, but they remain difficult to debug and refine. Failures can originate from subtle errors in intermediate outputs that propagate to downstream nodes, requiring developers to inspect long traces and infer which agent to modify. We present PROTEA, a unified interface for offline, test-driven improvement of multi-agent workflows. PROTEA executes a workflow, scores intermediate node outputs with configurable rubrics, and overlays per-node states and rationales on the workflow graph to localize likely bottlenecks. To support complex systems where final-answer references are the primary supervision, PROTEA performs backward node evaluation: it generates candidate node-level expectations from final-answer references and graph context, then compares them with observed node outputs. For selected nodes, PROTEA presents targeted prompt revisions as editable before/after comparisons, then automatically reruns and re-evaluates the workflow to show output changes and score trajectories within the same interface. In two production-adjacent workflows, PROTEA improved document-inspection accuracy from 64.3% to 83.9% and recommendation Hit@5 from 0.30 to 0.38. In a formative study with six experienced LLM developers, participants valued graph-level localization, per-node rationales, and editable before/after prompt revisions.
>
---
#### [new 040] ConflictRAG: Detecting and Resolving Knowledge Conflicts in Retrieval Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ConflictRAG，解决RAG系统中知识冲突问题。通过冲突检测、评估与处理，提升生成答案的准确性。**

- **链接: [https://arxiv.org/pdf/2605.17301](https://arxiv.org/pdf/2605.17301)**

> **作者:** Chenyu Wang; Yingmin Liu; Yang Shu
>
> **备注:** 6 pages, 6 figures, submitted to IEEE SMC 2026
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems implicitly assume mutual consistency among retrieved documents -- an assumption that frequently fails in practice. We present ConflictRAG, a conflict-aware RAG framework that detects, classifies, and resolves knowledge conflicts prior to answer generation. The framework introduces three contributions: (1) a two-stage conflict detection module combining a lightweight embedding-based MLP classifier with selective LLM refinement, reducing API costs by 62% while maintaining 90.8% detection accuracy; (2) an Entropy-TOPSIS framework for data-driven source credibility assessment, improving selection accuracy by 7.1% over manual heuristics; and (3) a Conflict-Aware RAG Score (CARS) for diagnostic evaluation of conflict-handling capabilities. Experiments on three benchmarks against six baselines demonstrate 88.7% conflict-detection F1 and consistent 5.3--6.1% correctness gains over the strongest conflict-aware baseline, with the pipeline transferring effectively across backbone LLMs.
>
---
#### [new 041] Taming "Zombie'' Agents: A Markov State-Aware Framework for Resilient Multi-Agent Evolution
- **分类: cs.CL**

- **简介: 该论文属于多智能体系统任务，旨在解决因过度剪枝导致有价值代理被错误淘汰的问题。提出AgentRevive框架，通过状态感知策略动态管理代理协作，提升系统韧性。**

- **链接: [https://arxiv.org/pdf/2605.17348](https://arxiv.org/pdf/2605.17348)**

> **作者:** Taolin Zhang; Pukun Zhao; Qizhou Chen; Jiuheng Wan; Chen Chen; Xiaofeng He; Chengyu Wang; Richang Hong
>
> **摘要:** Recent advancements in LLM-based multi-agent systems have demonstrated remarkable collaborative capabilities across complex tasks. To improve overall efficiency, existing methods often rely on aggressive graph evolution among agents (e.g., node or edge pruning), which risks prematurely discarding valuable agents due to transient issues such as hallucinations or temporary knowledge gaps. However, such hard pruning overlooks the potential for ``zombie'' agents to recover and contribute in subsequent discussion rounds. In this paper, we propose AgentRevive, a Markov state-aware framework for resilient multi-agent evolution. Our approach dynamically manages agent collaboration through soft state transitions, implemented via two key components: (1) State-Aware Policy Learning: Agent states are divided into ``Active'', ``Standby'', and ``Terminated'' states, selectively propagating messages based on agent memory. The policy employs a risk estimator to optimize agent state transitions by assessing hallucination risk, minimizing the influence of unreliable nodes while safeguarding valuable ones. (2) State-Aware Edge Optimization: Subgraph edges are pruned according to states learned from the policy, permanently removing ``Terminated'' nodes and retaining ``Standby'' nodes for subsequent rounds to assess their potential future contributions. Extensive experiments on general reasoning, domain-specific, and hallucination challenge tasks show that our method consistently outperforms strong baselines and significantly reduces token consumption through state-aware agent scheduling.
>
---
#### [new 042] Beyond Sentiment Classification: A Generative Framework for Emotion Intensity Evaluation in Text
- **分类: cs.CL; econ.GN; q-fin.GN**

- **简介: 该论文属于情感分析任务，旨在解决传统分类方法在情绪强度评估上的不足。通过构建情感强度数据集并微调生成模型，实现连续值输出，提升分析的表达能力和适用性。**

- **链接: [https://arxiv.org/pdf/2605.16613](https://arxiv.org/pdf/2605.16613)**

> **作者:** Francesco A. Fabozzi; Dasol Kim; William N. Goetzmann
>
> **备注:** 10 pages, no figures, 5 tables
>
> **摘要:** We introduce a novel approach to emotion modeling that shifts the focus from identification to evaluation, addressing the limitations of discrete classification in applied domains such as finance. By constructing a dataset of emotional intensity scores and fine-tuning open-weight generative language models to output continuous values from 0-100, we demonstrate a more expressive, generalizable framework for sentiment and emotion analysis. Our findings not only outperform classification baselines but also reveal surprising generalization capabilities and transfer effects to related constructs such as sentiment and arousal. This work contributes to the interdisciplinary recontextualization of NLP by introducing emotion intensity evaluation as an alternative to classification, arguing that this shift better aligns with the needs of domains--such as finance--where the degree of emotional content is central to interpretation and decision-making.
>
---
#### [new 043] iPOE: Interpretable Prompt Optimization via Explanations
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决提示优化的透明性问题。通过引入可解释的提示优化方法iPOE，利用自动生成的指南提升提示效果与可解释性。**

- **链接: [https://arxiv.org/pdf/2605.18113](https://arxiv.org/pdf/2605.18113)**

> **作者:** Jiahui Li; Sean Papay; Roman Klinger
>
> **摘要:** Prompt optimization has often been framed as a discrete search problem to find high-performing and robust instructions for an LLM. However, the search result might not make it transparent why and where specific prompt changes lead to performance gains. This is in contrast to how humans are instructed for annotation tasks. Here, researchers carefully design annotation guidelines, leading to enhanced annotation consistency. Our paper aims at joining these two approaches and introduces iPOE, a novel interpretable prompt optimization strategy via explanations. We guide the prompt optimization process by automatically created guidelines from explanations of annotation decisions (either automatically generated or from humans). This set of guidelines is furthermore optimized by as series of operations, including removing, adding, shuffling, and merging. The resulting prompt includes guidelines that instruct the annotation, making the decision process of the LLM and the optimization transparent. It therefore supports also laypeople in the area of prompt optimization, particularly in challenging domains requiring expertise. In our experiments on four datasets, we find that iPOE can improves over prompts without guidelines and with random selected guidelines by up to $31\%$ and $35\%$, respectively. Moreover, LLM explanations can replace human explanations in the proposed method.
>
---
#### [new 044] Exploring Lightweight Large Language Models for Court View Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律人工智能领域，研究轻量级大语言模型在刑事法庭观点生成及指控预测中的应用，探讨模型架构、规模及与DNN的对比效果。**

- **链接: [https://arxiv.org/pdf/2605.16770](https://arxiv.org/pdf/2605.16770)**

> **作者:** Zhitian Hou; Tianyong Hao; Nanli Zeng; Zhixiong Chao; Kun Zeng
>
> **摘要:** Criminal Court View Generation (CVG) is a critical task in Legal Artificial Intelligence (Legal AI), involving the generation of court view based on case facts. In this work, we systematically explore the capabilities of lightweight (smaller than 2B) large language models (LLMs) in CVG and their impact on charge prediction. Our study addresses four key questions: (1) how does different architecture of LLMs affect the CVG quality and charge prediction. (2) how does LLMs size contribute to the performance, (3) how do lightweight LLMs compare with Deep Neural Networks (DNNs) in these tasks, and (4) how does predicting charge by court view generation first compare with predicting it directly. Additionally, we also develop CVGEvalKit, an evaluation framework including three public available datasets for CVG tasks, as well as predicting their charges. Comprehensive experiments are conducted on this framework, where models are trained on a mixed training set and evaluated on each dataset's test set. Experimental results provide new insights into the trade-offs between model architecture, model size, and the influence between different tasks, highlighting the potential of lightweight LLMs in judicial AI applications. The source code is anonymously available at \url{this https URL}
>
---
#### [new 045] HyDRA: Hybrid Dynamic Routing Architecture for Heterogeneous LLM Pools
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出HyDRA，解决异构大模型池中的动态路由问题，通过预测查询需求并匹配模型能力，实现成本与质量的优化。**

- **链接: [https://arxiv.org/pdf/2605.17106](https://arxiv.org/pdf/2605.17106)**

> **作者:** Aashna Garg; Siddharth Singha Roy; Jinu Jang; Federico Brancasi; Shengyu Fu
>
> **备注:** 26 pages, preprint v1. Production-telemetry tables and per-language breakdown deferred to v2
>
> **摘要:** Production LLM deployments increasingly maintain heterogeneous model pools spanning order-of-magnitude cost differences. Existing routers make binary strong-vs-weak decisions and couple learned parameters to specific model identities, requiring retraining whenever the catalog changes. We present HyDRA (Hybrid Dynamic Routing Architecture), a framework that predicts fine-grained, multi-dimensional capability requirements per query and matches them against configuration-defined model profiles via shortfall matching. A ModernBERT encoder with K=4 independent sigmoid heads scores each query along reasoning, code generation, debugging, and tool use; a shortfall-matching algorithm then selects the cheapest model whose capabilities meet the predicted requirements. The deployed predictor runs at 86 ms median CPU inference latency in production, and is fully decoupled from the model catalog -- adding or removing models requires only a configuration change, with zero retraining. On SWE-Bench Verified (5-model pool: GPT-5.4-mini, Claude Haiku 4.5, GPT-5.3 Codex, Claude Sonnet 4.6, GPT-5.4), HyDRA's tunable shortfall threshold spans three regimes: peak-quality exceeds the always-strong Claude Sonnet 4.6 baseline (75.4% vs. 74.2% resolution) at 12.9% cost savings; iso-quality matches Sonnet at 54.1% cost savings, a 6x improvement over our prior in-house binary router at 9.1%; aggressive pushes savings to 72.5% for a 3.2-point quality trade. Results generalize across LiveCodeBench, BigCodeBench, and tau-bench. HyDRA is deployed to all users in GitHub Copilot's VS Code Chat auto-mode and -- to our knowledge for the first time in the LLM routing literature -- demonstrates language-invariant routing across CJK, European, and other script families.
>
---
#### [new 046] From BERT to T5: A Study of Named Entity Recognition
- **分类: cs.CL**

- **简介: 该论文属于命名实体识别任务，旨在比较BERT和T5模型在该任务中的表现，通过微调和实验分析其效果与差异。**

- **链接: [https://arxiv.org/pdf/2605.18462](https://arxiv.org/pdf/2605.18462)**

> **作者:** Mei Jia
>
> **备注:** 11 pages, 9 figures
>
> **摘要:** Named entity recognition (NER) has been one of the essential preliminary steps in modern NLP applications. This report focuses on implementing the NER task on finetuning two pretrained models: (i) an encoder-only model (BERT) with a simple classification head, and (ii) a sequence-to-sequence model (T5) with few-shot prompts. Under the original 7-class tag and 3-class simplified tag schemes, BERT is applied a weighted cross-entropy for training loss, and T5 is fine-tuned with two validation strategies. It also conducted an ablation study with different hyperparameters. Moreover, the related analysis provides valuable insights into common errors in BERT and the two models' performance. Based on a bunch of performance metrics, this report aims to compare the above two architectures and explore their abilities in the sequence labelling task, laying the groundwork for further practical use cases.
>
---
#### [new 047] SEMA-RAG: A Self-Evolving Multi-Agent Retrieval-Augmented Generation Framework for Medical Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗问答任务，解决RAG框架在临床推理中的不足。通过引入多智能体协作，提升检索与生成的准确性。**

- **链接: [https://arxiv.org/pdf/2605.17101](https://arxiv.org/pdf/2605.17101)**

> **作者:** Yongfeng Huang; Ruiying Chen; James Cheng
>
> **摘要:** Retrieval-Augmented Generation (RAG) is widely employed to mitigate risks such as hallucinations and knowledge obsolescence in medical question answering, yet its predominantly single-round, static retrieval paradigm misaligns with the multi-stage process of clinical reasoning. This compressed workflow induces two structural deficiencies: question-to-query translation often lacks clinically grounded semantic interpretation, and retrieval lacks iterative sufficiency feedback, making it difficult to form reliable evidence chains. We argue that both issues stem from a deeper cause: overloading a single reasoning chain with heterogeneous tasks of interpretation, exploration, and adjudication. The remedy is to reconstruct the workflow via task decoupling and dynamic multi-round exploration. To this end, we propose SEMA-RAG, a Self-Evolving Multi-Agent RAG framework for medical question answering, which assigns these roles to three specialist agents: the Interpreter Agent for clinical schema interpretation, the Explorer Agent for sufficiency-driven self-evolving retrieval, and the Arbiter Agent for evidence adjudication and answer selection. Across five benchmarks and five LLM backbones, SEMA-RAG improves the strongest baseline by +6.46 accuracy points on average, measured per backbone.
>
---
#### [new 048] BacktestBench: Benchmarking Large Language Models for Automated Quantitative Strategy Backtesting
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于量化策略回测任务，旨在解决自动化回测的高技术门槛和可扩展性问题。提出BacktestBench基准和AutoBacktest方法，提升LLM在该领域的性能。**

- **链接: [https://arxiv.org/pdf/2605.17937](https://arxiv.org/pdf/2605.17937)**

> **作者:** Zhensheng Wang; Wenmian Yang; Qingtai Wu; Lequan Ma; Yiquan Zhang; Weijia Jia
>
> **备注:** This paper has been accepted by KDD 2026 (Datasets and Benchmarks Track)
>
> **摘要:** Quantitative backtesting is essential for evaluating trading strategies but remains hampered by high technical barriers and limited scalability. While Large Language Models (LLMs) offer a transformative path to automate this complex, interdisciplinary workflow through advanced code generation, tool usage, and agentic planning, the practical realization is significantly challenged by the current lack of a large-scale benchmark dedicated to automated quantitative backtesting, which hinders progress in this field. To bridge this critical gap, we introduce BacktestBench, the first large-scale benchmark for automated quantitative backtesting. Built from over 6 million real market records, it comprises 18,246 meticulously annotated question-answering pairs across four task categories: metrics calculation, ticker selection, strategy selection, and parameter confirmation. We also propose AutoBacktest, a robust multi-agent baseline that translates natural language strategies into reproducible backtests by coordinating a Summarizer for semantic factor extraction, a Retriever for validated SQL generation, and a Coder for Python backtesting implementation. Our evaluation on 23 mainstream LLMs, complemented by targeted ablations, identifies key factors that influence end-to-end performance and highlights the importance of grounded verification and standardized indicator representations.
>
---
#### [new 049] PAREDA: A Multi-Accent Speech Dataset of Natural Language Processing Research Discussions
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PAREDA数据集，用于研究多口音语音识别问题。针对ASR系统在真实场景下的性能下降，通过收集不同口音的NLP讨论数据，评估模型表现并验证数据集的有效性。**

- **链接: [https://arxiv.org/pdf/2605.17860](https://arxiv.org/pdf/2605.17860)**

> **作者:** Sicheng Jin; Dipankar Srirag; Aditya Joshi
>
> **备注:** Accepted and presented at SPEAKABLE 2026 workshop at LREC 2026
>
> **摘要:** While modern Automatic Speech Recognition (ASR) systems achieve high accuracy on benchmark corpora, their performance often degrades when there is real-world variability. This work focuses on variability arising due to accented, spontaneous, and domain-specific speech. In particular, we introduce PAper REading DAtaset (PAREDA), a first-of-its-kind multi-accent speech dataset consisting of discussions on academic Natural Language Processing (NLP) papers between speakers with Australian, Indian-English, and Chinese English accents. Each session elicits a spontaneous monologue (a summary of a paper's abstract) and a non-monologue (a question-and-answer session between participants), resulting in a corpus rich with technical jargon and conversational phenomena. We evaluate the performance of SOTA ASR models on PAREDA, analysing the impact of accent mixing and increased speech rate. Our results show that, in the zero-shot setting, models perform worse, confirming the dataset's challenging nature. However, fine-tuning on PAREDA significantly reduces the Word Error Rate (WER), demonstrating that our dataset captures linguistic characteristics often missing from existing corpora. PAREDA serves as a valuable new resource for building and evaluating more robust and inclusive ASR systems for specialised, real-world applications.
>
---
#### [new 050] Analyzing Error Propagation in Korean Spoken QA with ASR-LLM Cascades
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 论文研究韩国语音问答中ASR-LLM级联的误差传播问题，分析ASR错误如何影响下游语义表现，提出直接音频输入可能减少信息损失。**

- **链接: [https://arxiv.org/pdf/2605.17443](https://arxiv.org/pdf/2605.17443)**

> **作者:** Donghyuk Jung; Youngwon Choi
>
> **备注:** Preprint. Submitted to APSIPA ASC 2026
>
> **摘要:** We analyze how automatic speech recognition (ASR) errors propagate through ASR-LLM cascades in Korean spoken question answering (SQA), focusing on downstream semantic failures that conventional ASR metrics cannot fully capture. Our analysis shows that the relative downstream degradation caused by ASR errors is consistent across LLMs with different absolute performance, suggesting that cascade degradation largely tracks ASR-stage information loss. We further identify single-character Korean ASR errors as a distinct semantic-failure channel, where the gold answer becomes entirely absent from the downstream prediction despite only a minimal transcription difference. Finally, an auxiliary comparison shows that a large audio language model outperforms an ASR-LLM pipeline with a matched language backbone in noisy Korean SQA, indicating the potential of direct audio input to mitigate transcript-induced information loss.
>
---
#### [new 051] Bridging the Gap: Converting Read Text to Conversational Dialogue
- **分类: cs.CL**

- **简介: 该论文属于语音转换任务，旨在将朗读语音转化为自然对话语音。解决传统方法缺乏韵律变化的问题，提出PACC方法，利用深度学习提升语音自然度和准确性。**

- **链接: [https://arxiv.org/pdf/2605.18001](https://arxiv.org/pdf/2605.18001)**

> **作者:** Parshav Singla; Agnik Banerjee; Aaditya Arora; Shruti Aggarwal; Anil Kumar Verma; Vikram C M; Raj Prakash Gohil; Gopal Kumar Agarwal
>
> **备注:** 11 pages, 4 figures. Published in ICICC 2025, Springer Lecture Notes in Networks and Systems
>
> **摘要:** In recent advancements within speech processing, converting read speech to conversational speech has gained significant attention. The primary challenge in this domain is maintaining naturalness and intelligibility while minimizing computational overhead for real-time applications. Traditional read speech often lacks the nuanced prosodic variation essential for natural conversational interactions, posing challenges for applications in virtual assistants, customer service, and language learning tools. This paper introduces a novel approach, Prosodic Adjustment with Conversational Context (PACC), aimed at converting read speech into natural conversational speech used in various modern applications. PACC utilizes advanced deep neural networks to analyze and modify prosodic features such as intonation, stress, and rhythm. Unlike conventional methods, our approach uses High-Fidelity Generative Adversarial Networks (HiFi-GAN) for speech synthesis. Our experimental results demonstrate significant improvements in speech conversion, enhancing naturalness and achieving better model accuracy with additional training on speech datasets. This research establishes new benchmarks in speech conversion tasks and Mean Opinion Score (MOS) evaluation for testing model accuracy, and we show that our approach can be successfully extended to other speech conversion applications.
>
---
#### [new 052] Residual Semantic Decomposition of Word Embeddings
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出RSD方法，用于分解词向量，平衡重构与关系结构保持。解决词义歧义问题，通过残差分析进行语义诊断。**

- **链接: [https://arxiv.org/pdf/2605.17482](https://arxiv.org/pdf/2605.17482)**

> **作者:** Seungmin Jin
>
> **备注:** Short paper; includes appendix. Code and data are not included in the arXiv source package
>
> **摘要:** We introduce Residual Semantic Decomposition (RSD), a neural additive decomposition of word embeddings that balances embedding reconstruction with relational structure preservation. RSD supports recursive binary decomposition: each $K=2$ fit extracts a local semantic axis, while residuals expose information not absorbed by that axis. In manually specified paired-context diagnostics over ambiguous words, RSD separates supplied context anchors above shuffled-label controls, but entropy diagnostics show that ambiguous targets are not uniformly high-entropy boundary points in static GloVe. We therefore treat residual neighborhoods as qualitative diagnostics rather than benchmark sense predictions.
>
---
#### [new 053] HalluScore: Large Language Model Hallucination Question Answering Benchmark
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在阿拉伯语中的幻觉问题。作者构建了HalluScore基准，用于评估和减少模型幻觉，涵盖多个推理难度和知识领域。**

- **链接: [https://arxiv.org/pdf/2605.17007](https://arxiv.org/pdf/2605.17007)**

> **作者:** Aisha Alansari; Hamzah Luqman
>
> **摘要:** Large language models (LLMs) have achieved remarkable progress in natural language generation, but remain susceptible to hallucination. In response to growing concerns about hallucinations, several benchmarks have been developed, primarily in English and Chinese. However, Arabic remains underrepresented, with limited benchmarks for LLMs hallucination due to scarce annotated resources and the language's morphological complexity. Consequently, existing benchmarks do not adequately reflect the linguistic, cultural, and reasoning characteristics of Arabic. To address this gap, we introduce HalluScore, a structured Arabic question answering benchmark designed to evaluate hallucination behavior in LLMs across different levels of reasoning difficulty, various knowledge domains, historical timelines, and culturally grounded Arabic scenarios. It contains 827 carefully curated questions for evaluating, detecting, and mitigating hallucination in LLMs. The dataset was constructed through a structured pipeline involving quality assurance, filtering for clarity and factual validity, and model-driven selection to retain questions that consistently trigger hallucinations. Each question is linked to verified ground-truth evidence, answer explanations, and multi-label annotations. Using the HalluScore benchmark, we conduct a comprehensive empirical analysis of hallucination patterns across 17 Arabic, multilingual, and reasoning LLMs. Moreover, we provide high-quality human annotations identifying hallucinated, non-hallucinated, and partially hallucinated responses of all evaluated LLMs. These results suggest that hallucination in Arabic LLMs extends beyond factual inaccuracies, encompassing challenges related to cultural understanding, linguistic reasoning, and logical consistency. We release HalluScore to support future research on improving the reliability and cultural competence of LLMs in Arabic.
>
---
#### [new 054] Retrieval-Based Multi-Label Legal Annotation: Extensible, Data-Efficient and Hallucination-Free
- **分类: cs.CL**

- **简介: 该论文属于多标签法律标注任务，旨在解决标签集频繁变化和数据稀缺的问题。通过检索方法实现高效、无幻觉的标注。**

- **链接: [https://arxiv.org/pdf/2605.16767](https://arxiv.org/pdf/2605.16767)**

> **作者:** Li Zhang; Jaromir Savelka; Kevin Ashley
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Multi-label legal annotation requires assigning multiple labels from large, evolving taxonomies to long, fact-intensive documents, often under limited supervision. Parametric encoders typically require task-specific training and retraining when the label set changes, while prompting generative large language models becomes costly and degrades as the label space grows. We cast legal annotation as retrieval: we embed documents and label descriptions with a frozen retrieval model and predict labels via k-nearest neighbors in the embedding space, enabling updates by re-embedding and re-indexing rather than gradient-based backpropagation. Across three legal datasets (ECtHR-A, ECtHR-B, and Eurlex with 100 labels), retrieval achieves competitive accuracy and strong data efficiency; on Eurlex, Qwen-8B retrieval improves Macro-F1 from 40.41 (GPT-5.2, zero-shot) to 49.12 while reducing estimated compute by 20-30 times compared to fine-tuning. With only (N=100) training samples, retrieval nearly doubles Micro-F1 over hierarchical Legal-BERT on ECtHR-A (48.29 vs. 27.87). We also quantify a reliability failure mode of generative inference: GPT-5.2 hallucinates labels outside the provided taxonomy in 0.12-0.9% of test samples under deterministic decoding. In contrast, retrieval strictly respects defined label sets, eliminating hallucination by design. These results suggest retrieval-model-based annotators are a practical, deployable alternative for high-cardinality and rapidly changing legal label spaces.
>
---
#### [new 055] MiniGPT: Rebuilding GPT from First Principles
- **分类: cs.CL; cs.LG**

- **简介: 论文介绍MiniGPT，一个基于PyTorch的GPT风格语言模型实现，旨在从基础原理重建GPT架构。任务是语言建模，解决如何复现GPT结构的问题，工作包括实现核心组件并进行文本生成实验。**

- **链接: [https://arxiv.org/pdf/2605.17398](https://arxiv.org/pdf/2605.17398)**

> **作者:** Jibin Joseph
>
> **备注:** 13 pages, 2 figures
>
> **摘要:** This paper presents MiniGPT, a compact from-scratch implementation of GPT-style autoregressive language modeling in PyTorch. The aim is to rebuild the core GPT pipeline from first principles after studying the design of nanoGPT by Andrej Karpathy, while keeping the model and training code independently written in a single notebook. MiniGPT implements token and positional embeddings, causal multi-head self-attention, pre-LayerNorm Transformer blocks, residual connections, feed-forward MLP layers, next-token cross-entropy training (teacher forcing), validation tracking, checkpoint selection, and autoregressive text generation. This paper evaluates the implementation on Tiny Shakespeare dataset using character-level tokenization. A baseline 0.83M-parameter model reaches a validation loss of 1.7236 after 3000 training iterations. A stronger 10.77M-parameter configuration, using a larger context length and improved training settings, reaches a best validation loss of 1.4780 and generates text with recognizable Shakespeare-style dialogue structure. MiniGPT does not introduce a new language-model architecture. Instead, it documents a clear and reproducible implementation path from raw text to trained character-level generation, including design choices, training behavior, generation quality, and practical limitations.
>
---
#### [new 056] MixSD: Mixed Contextual Self-Distillation for Knowledge Injection
- **分类: cs.CL**

- **简介: 该论文属于知识注入任务，旨在解决微调导致模型遗忘原能力的问题。提出MixSD方法，通过混合模型自身条件生成监督信号，实现更有效的知识注入。**

- **链接: [https://arxiv.org/pdf/2605.16865](https://arxiv.org/pdf/2605.16865)**

> **作者:** Jiarui Liu; Lechen Zhang; Yongjin Yang; Yinghui He; Yingheng Wang; Weihao Xuan; Zhijing Jin; Mona Diab
>
> **摘要:** Supervised fine-tuning (SFT) is widely used to inject new knowledge into language models, but it often degrades pretrained capabilities such as reasoning and general-domain performance. We argue this forgetting arises because fine-tuning targets from humans or external systems diverge from the model's autoregressive distribution, forcing the optimizer to imitate low-probability token sequences. To address this problem, we propose MixSD, a simple external-teacher-free method for distribution-aligned knowledge injection. Instead of training on fixed targets, MixSD constructs supervision dynamically by mixing tokens from two conditionals of the base model itself: an expert conditional that observes the injected fact in context, and a naive conditional that reflects the model's original prior. The resulting supervision sequences preserve the factual learning signal while remaining substantially closer to the base model's distribution. We evaluate MixSD on two synthetic corpora that we construct to study factual recall and arithmetic function acquisition in a controlled setting, together with established benchmarks for open-domain factual question answering and knowledge editing. Across multiple model scales and settings, MixSD consistently achieves a better memorization-retention trade-off compared to SFT and on-policy self distillation baselines, retaining up to 100% of the base model's held-out capability while maintaining near-perfect training accuracy, whereas standard SFT retains as little as 1%. We further show that MixSD produces substantially lower-NLL supervision targets under the base model and reduces harmful movement along Fisher-sensitive parameter directions. These results suggest that aligning supervision with the model's native generation distribution is a simple and effective principle for knowledge injection that mitigates catastrophic forgetting.
>
---
#### [new 057] A Pilot Benchmark for NL-to-FOL Translation in Planetary Exploration
- **分类: cs.CL**

- **简介: 该论文属于自然语言到一阶逻辑的翻译任务，旨在解决行星探索中高阶任务知识的结构化表示问题。工作包括构建真实任务文档的FOL标注数据集，并提供词汇和常量支持实验。**

- **链接: [https://arxiv.org/pdf/2605.17911](https://arxiv.org/pdf/2605.17911)**

> **作者:** Hayden Moore; Suman Saha; Mahfuza Farooque
>
> **摘要:** Future planetary exploration envisions autonomous robotic agents operating under severe communication constraints, without global positioning, and with minimal human intervention. In such environments, agents must not only perceive and act, but also reason over mission objectives, operational constraints, and evolving environmental conditions. While prior work has largely focused on perception and control, the translation of high-level mission knowledge into structured, machine-interpretable representations remains underexplored. We introduce a pilot benchmark for translating natural language (NL) into First-Order Logic (FOL) within the domain of planetary exploration. The dataset is constructed from real mission documentation sourced from NASA's Planetary Data System (PDS), spanning missions from 2003 to 2013. These documents describe mission phases such as launch, boost, coast, cruise, and orbital operations in rich natural language. We manually annotate these documents with corresponding FOL representations that capture temporal structure, agent roles, and operational dependencies. In addition, we provide structured predicate vocabularies and typed constants to enable controlled experimentation with varying levels of prior knowledge. This pilot benchmark provides a foundation for research at the intersection of language understanding and formal reasoning, grounded in real-world, safety-critical mission data. The dataset is provided at: this https URL
>
---
#### [new 058] Infini-News: Efficiently Queryable Access to 1.3 Billion Processed Common Crawl News Articles
- **分类: cs.CL**

- **简介: 该论文提出Infini-News，解决大规模新闻数据高效检索问题。任务为构建可查询的新闻语料库，通过处理135亿条新闻，实现快速文本搜索与地理语言分析。**

- **链接: [https://arxiv.org/pdf/2605.18337](https://arxiv.org/pdf/2605.18337)**

> **作者:** Ruggero Marino Lazzaroni; Jana Lasser; Kirill Solovev
>
> **摘要:** Large-scale news corpora support a wide range of research in Computational Social Science and NLP, yet access remains constrained: commercial archives impose prohibitive costs and licensing restrictions, while open alternatives like Common Crawl's CC-News require terabyte-scale storage and computationally intensive processing. We present Infini-News, a retrieval toolkit and index for the entire CC-News archive from August 2016 to the latest available snapshot. Our contributions are threefold. First, we extract, clean the text, and parse the structured metadata of over 1.35B articles. Second, we enrich the corpus with language detection using three frontier language classifiers (GlotLID, lingua, and CommonLingua), and with multi-source geographic attribution that resolves a country of origin for 83.4% of articles across 222 countries. Third, we construct Infini-gram indexes: suffix-array structures that let researchers search the full archive for arbitrary text patterns in sub-second time. Together, these resources lower the barrier to longitudinal, cross-national media research.
>
---
#### [new 059] Monitoring the Internal Monologue: Probe Trajectories Reveal Reasoning Dynamics
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于模型监控任务，旨在解决CoT可靠性问题。通过分析模型推理过程中的探针轨迹，提取动态特征以预测未来行为，提升安全监测效果。**

- **链接: [https://arxiv.org/pdf/2605.18549](https://arxiv.org/pdf/2605.18549)**

> **作者:** Maciej Chrabąszcz; Aleksander Szymczyk; Marcin Sendera; Tomasz Trzciński; Sebastian Cygert
>
> **摘要:** Large Reasoning Models (LRMs) introduce new opportunities for safety monitoring through their Chain of Thought (CoT) reasoning. However, CoT is not always faithful to the model's final output, undermining its reliability as a monitoring tool. To address this, we investigate the hidden representations of LRMs to determine whether future behavior can be predicted from prompt and CoT representations. By evaluating a probe at each generated token, we construct a probe trajectory, the continuous evolution of a concept's probability across the reasoning process. We find that future model behavior is more distinguishable when examined over the full trajectory than from a single static prediction. To characterize these temporal dynamics, we extract signal-processing features that capture volatility, trend, and steady-state behavior, significantly improving the separation of future model states. We also present two methodological insights. First, template-based training data achieves near-parity with dynamically generated model responses, eliminating the need for a costly initial inference and labeling. Second, the choice of pooling operation is critical: average-pooling and last-token methods collapse to near-random performance, while max-pooling achieves up to 95% AUROC and yields stable probe trajectories. Using four datasets and four reasoning models across the domains of safety and mathematics, we demonstrate that trajectory features encode task-specific dynamics that improve outcome separability. These findings establish probe trajectories as a complementary framework for monitoring LRM behavior. Warning: This article contains potentially harmful content.
>
---
#### [new 060] Implicit Hierarchical GRPO: Decoupling Tool Invocation from Execution for Tool-Integrated Mathematical Reasoning
- **分类: cs.CL**

- **简介: 该论文属于数学推理任务，旨在解决工具调用与执行耦合导致的推理性能下降问题。通过提出IH-GRPO算法，实现工具调用与执行的解耦，提升推理效果。**

- **链接: [https://arxiv.org/pdf/2605.18500](https://arxiv.org/pdf/2605.18500)**

> **作者:** Li Wang; Xiaohan Wang; Xiaodong Lu; Zipeng Zhang; Jinyang Wu; Jiajun Chai; Wei Lin; Guojun Yin
>
> **摘要:** Large language models (LLMs) have increasingly leveraged tool invocation to enhance their reasoning capabilities. However, existing approaches typically tightly couple tool invocation with immediate execution. Such immediate tool interaction may disrupt the reasoning coherence of LLMs and constrain their expressivity, ultimately degrading reasoning performance. To this end, for the first time, we propose and formalize the problem of decoupling tool invocation from execution during reasoning, and introduce delayed execution with explicit control to enhance tool-integrated reasoning (TIR). Furthermore, we propose a hierarchical control framework and theoretically derive a surrogate loss that enables an implicitly hierarchical policy to learn behavior equivalent to that of an explicit hierarchical policy, leading to the proposed IH-GRPO algorithm. Extensive experiments on IH-GRPO achieve absolute improvements of 1.87\%, 2.16\%, and 2.53\% on Qwen3-1.7B, Qwen3-4B, and Qwen3-8B across six out-of-domain mathematical reasoning benchmarks over the strongest baseline method, while also yielding consistent performance gains in other domains. Our code is available at this https URL.
>
---
#### [new 061] Agentic AI Translate: An Agentic Translator Prototype for Translation as Communication Design
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文提出一种基于代理的翻译系统，将翻译视为通信设计而非文本转换。解决传统机器翻译范式的问题，通过四阶段代理循环实现更符合语境的翻译。**

- **链接: [https://arxiv.org/pdf/2605.17041](https://arxiv.org/pdf/2605.17041)**

> **作者:** Masaru Yamada
>
> **备注:** 14 pages. Conceptual and architectural paper; empirical validation in future work. Code: this https URL (v0.8.0). Live demo: this https URL
>
> **摘要:** We present Agentic AI Translate, an agentic translator prototype that operationalises the thesis of Yamada (forthcoming) -- that the metalanguage of Translation Studies has become an instruction code for generative AI. The system replaces the dominant text-in / text-out paradigm of machine translation with a four-stage agentic cycle (Identify -> Prompt -> Generate -> Verify), preceded by an interactive specification phase in which the user composes -- through model-assisted dialogue -- a structured translation brief grounded in skopos theory, register, audience, and genre conventions. The verification stage adopts the GEMBA-MQM error-span protocol (Kocmi & Federmann, 2023) for evidence-grounded scoring, and document-level coherence is preserved through a DelTA-lite memory of proper nouns and a running bilingual summary, after Wang et al. (2025). We describe the philosophical motivation, the architectural commitments, the four reference-material categories the system consumes, and the principal design tensions the architecture makes explicit. Empirical validation is left for future work; the contribution here is conceptual and architectural -- an executable embodiment of the position that translation in the GenAI era is communication design, not text conversion.
>
---
#### [new 062] SKG-Eval: Stateful Evaluation of Multi-Turn Dialogue via Incremental Semantic Knowledge Graphs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统评估任务，解决多轮对话中长期不一致问题。提出SKG-Eval框架，通过语义知识图谱跟踪对话状态，提升评估准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2605.16650](https://arxiv.org/pdf/2605.16650)**

> **作者:** Avijit Shil; Suman Samui
>
> **备注:** 36 Pages, 6 Figures
>
> **摘要:** Evaluating multi-turn dialogue systems remains challenging because response quality depends not only on the current prompt, but also on previously established entities, claims, and conversational commitments. Existing automatic evaluators, including LLM-as-a-judge frameworks and embedding-based metrics, largely rely on flat or turn-isolated representations, making them less effective at detecting long-range issues such as contradiction, topic drift, and entity inconsistency. To address this, we propose SKG-Eval, a quasi-deterministic and interpretable framework that models dialogue as an evolving Semantic Knowledge Graph (SKG) of entities, relations, and commitments across turns. The framework incrementally updates the graph through structured triple extraction and computes three complementary signals: (i) local relevance, measuring alignment with the current prompt and optional reference; (ii) historical consistency, evaluating how newly introduced information connects to prior conversational context using graph-based and embedding-driven signals; and (iii) logical coherence, assessed by a geometric contradiction engine that detects cross-turn conflicts without relying on NLI models or LLM judges. These signals are adaptively fused and aggregated into a length-invariant session score via recency-weighted trend analysis. Across multiple benchmarks, SKG-Eval achieves higher correlation with human judgments and substantially improves detection of long-range inconsistencies in extended conversations. In addition, the framework produces explicit contradiction certificates and deterministic scores for fixed inputs, enabling reproducible and auditable evaluation. Overall, our results suggest that structured externalized state tracking through semantic knowledge graphs provides a scalable alternative to implicit reasoning in LLM-based dialogue evaluators.
>
---
#### [new 063] Multi-agent AI systems outperform human teams in creativity
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于人工智能与创造力研究任务，旨在比较多智能体AI与人类团队的创造力。研究显示多智能体系统在六项任务中显著优于人类，揭示了其生成过程的不同模式。**

- **链接: [https://arxiv.org/pdf/2605.17885](https://arxiv.org/pdf/2605.17885)**

> **作者:** Tiancheng Hu; Yixuan Jiang; Haotian Li; José Hernández-Orallo; Xing Xie; Nigel Collier; David Stillwell; Luning Sun
>
> **摘要:** Although artificial intelligence (AI) now matches or exceeds human performance across numerous cognitive tasks, creativity remains a highly contested frontier. As AI systems based on large language models (LLMs) are increasingly adopted in research and innovation, it is essential to understand and augment their creativity. Here we demonstrate that multi-agent LLM teams not only surpass single agents, but also substantially outperform human teams in creativity (Cohen's d=1.50) across 4,541 multi-agent LLM ideas and 341 human-team ideas on six diverse problem-solving tasks. This advantage is driven by novelty while maintaining comparable usefulness. To investigate the generative processes in both groups, we represent conversations as paths through semantic space using neural language model representations. Both LLM and human teams produce more creative ideas when conversations range widely rather than staying centered on a single theme (low global coherence). However, the additional patterns that predict creativity differ: LLM teams benefit from efficient exploration (high semantic spread, shorter paths), while human teams benefit from maintaining smooth conversational flow (high local coherence, frequent pivots). Additionally, we identify model choice and discussion structure as orthogonal design levers that together explain 26.8% of variance in LLM conversational dynamics, paving the way for systematic approaches to developing multi-agent systems with augmented creative capabilities.
>
---
#### [new 064] Roll Out and Roll Back: Diffusion LLMs are Their Own Efficiency Teachers
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，解决扩散大模型中质量与速度的矛盾。通过提出WINO和WINO+方法，实现高效且高质量的解码。**

- **链接: [https://arxiv.org/pdf/2605.16941](https://arxiv.org/pdf/2605.16941)**

> **作者:** Fanqin Zeng; Feng Hong; Geng Yu; Huangjie Zheng; Xiaofeng Cao; Ya Zhang; Bo Han; Yanfeng Wang; Jiangchao Yao
>
> **摘要:** Diffusion Large Language Models (DLLMs) promise fast parallel generation, yet open-source DLLMs still face a severe quality-speed trade-off: accelerating decoding by revealing multiple tokens often causes substantial quality degradation. We attribute this dilemma to a train-inference mismatch amplified by irreversible decoding. While training reconstructs tokens from randomly corrupted states, efficient inference requires an adaptive denoising order, where easier tokens are revealed earlier and context-dependent ones are deferred. This view motivates two complementary methods: an inference-time method that makes parallel decoding revokable, and a training-time extension that distills the reliable order exposed by this revokable process. Accordingly, we first propose Wide-In, Narrow-Out (WINO), a training-free decoding algorithm that enables revokable parallel generation. WINO aggressively drafts multiple tokens, verifies generated tokens with enriched global context, and re-masks unreliable ones for later refinement. Building on this discovered order, we further introduce WINO+, which injects the verified denoising trajectories produced by WINO into model parameters, aligning training with efficient inference. Experiments on LLaDA and MMaDA show that WINO improves both quality and efficiency, while WINO+ further strengthens this progression. On GSM8K, WINO improves accuracy from 73.24% to 75.82% with a 6.10x step reduction, and WINO+ further achieves 76.58% with a 6.83x reduction. On Flickr30K, WINO+ reaches a 16.22x step reduction with improved CIDEr. These results demonstrate that DLLMs can serve as their own efficiency teachers by first discovering reliable denoising orders through revokable decoding and then learning to follow them for faster generation. Code is available at this https URL.
>
---
#### [new 065] PPAI: Enabling Personalized LLM Agent Interoperability for Collaborative Edge Intelligence
- **分类: cs.CL**

- **简介: 该论文提出PPAI系统，解决个性化LLM代理在边缘智能中的协作问题。通过查询-代理匹配和博弈模型，提升任务处理效率与负载平衡。**

- **链接: [https://arxiv.org/pdf/2605.18067](https://arxiv.org/pdf/2605.18067)**

> **作者:** Zile Wang; Qianli Liu; Kaibin Guo; Haodong Wang; Jian Lin; Zicong Hong; Song Guo
>
> **摘要:** Deploying large language model (LLM) on edge device enables personalized LLM agents for various users. The growing availability of diverse personalized agents presents a unique opportunity for peer-to-peer (P2P) collaboration, wherein each user can delegate tasks beyond the local agent's expertise to remote agents more suited for the specific query. This paper introduces PPAI, the first personalized LLM agent interoperability system, which enables users to collaborate with each other based on agent specialization. However, the ever-changing pool of agents and their interchangeable capacity introduce new challenges when it comes to matching queries to agents and balancing loads, compared with existing P2P systems. Therefore, we propose a scalable query-agent pair scoring mechanism based on prototypes to identify suitable agents within a P2P network with churn. Moreover, we propose a multi-agent interoperability Bayesian game to balance local demand and global efficiency, when changes in remote agent load occur too quickly to be observed. Finally, we implement a prototype of PPAI and demonstrate that it substantially broadens the range of tasks that could be carried out while maintaining load balance. On average, it achieves an accuracy improvement of up to 7.96% across multiple tasks, while reducing latency by 16.34% compared to the baseline.
>
---
#### [new 066] From Volume to Value: Preference-Aligned Memory Construction for On-Device RAG
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文属于RAG任务，解决设备端个性化信息检索问题。通过EPIC方法，高效存储用户偏好，提升检索准确性与速度。**

- **链接: [https://arxiv.org/pdf/2605.18271](https://arxiv.org/pdf/2605.18271)**

> **作者:** Changmin Lee; Jaemin Kim; Taesik Gong
>
> **备注:** Accepted to ICML 2026. Code and data are available at this https URL
>
> **摘要:** With the rapid emergence of personal AI agents based on Large Language Models (LLMs), implementing them on-device has become essential for privacy and responsiveness. To handle the inherently personal and context-dependent nature of real-world requests, such agents must ground their generation in device-resident personal context. However, under tight memory budgets, the core bottleneck is what to store so that retrieval remains aligned with the user. We propose EPIC (Efficient Preference-aligned Index Construction), which focuses on user preferences as a compact and stable form of personal context and integrates them throughout the RAG pipeline. EPIC selectively retains preference-relevant information from raw data and aligns retrieval toward preference-aligned contexts. Across four benchmarks covering conversations, debates, explanations, and recommendations, EPIC reduces indexing memory by 2,404 times, improves preference-following accuracy by 20.17 percentage points, and achieves 33.33 times lower retrieval latency over the best-performing baseline. In our on-device experiment, EPIC maintains a memory footprint under 1 MB with 29.35 ms/query latency in streaming updates.
>
---
#### [new 067] MA$^{2}$P: A Meta-Cognitive Autonomous Intelligent Agents Framework for Complex Persuasion
- **分类: cs.CL**

- **简介: 该论文属于复杂说服任务，旨在解决 persuader 难以准确理解 persuadee 内部状态的问题。提出 MA$^{2}$P 框架，整合多智能体协作与元认知配置，提升说服效果。**

- **链接: [https://arxiv.org/pdf/2605.18572](https://arxiv.org/pdf/2605.18572)**

> **作者:** Dingyi Zhang; Ziqing Zhuang; Linhai Zhang; Ziyang Gao; Deyu Zhou
>
> **备注:** 22 pages, 8 figures. Accepted to Findings of ACL 2026
>
> **摘要:** Persuasive dialogue generation plays a vital role in decision-making, negotiation, counseling, and behavior change, yet it remains a challenging problem. In complex persuasion where the persuadee's internal states are not expressed clearly, the persuader must interpret responses, infer the persuadee's latent mental states (e.g., beliefs and desires), and translate them into targeted, strategy-consistent actions; however, current approaches often produce generic or weakly grounded responses even when such cues are identified. Moreover, although large language models (LLMs) can generate persuasive content, their performance varies substantially across domains due to uneven knowledge coverage and limited reasoning generalization. To address these challenges, we propose MA$^{2}$P, a meta-cognitive autonomous intelligent agent framework for complex persuasion. Specifically, we develop an autonomous multi-agent architecture that coordinates perception management, mental-state inference, strategy execution, memory maintenance, and performance evaluation. To mitigate cross-domain performance variation, we further design a meta-cognitive configurator that selects an appropriate meta-strategy from a structured knowledge base at the outset, thereby guiding subsequent reasoning and planning. Experimental results show that our approach achieves a higher persuasion success rate than baselines.
>
---
#### [new 068] How Good LLMs Are at Answering Bangla Medical Visual Questions? Dataset and Benchmarking
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于医学视觉问答任务，旨在解决 Bangla 语言在医学视觉问答领域缺乏基准的问题。作者构建了 BanglaMedVQA 数据集，并评估了现有模型的表现。**

- **链接: [https://arxiv.org/pdf/2605.18111](https://arxiv.org/pdf/2605.18111)**

> **作者:** Rafid Ahmed; Intesar Tahmid; Mir Sazzat Hossain; Tasnimul Hossain Tomal; Md Fahim; Md Farhad Alam Bhuiyan
>
> **备注:** 14 pages, 7 figures, 5 tables, Proceedings of The Second AAAI Bridge Program on AI for Medicine and Healthcare, PMLR 317:1-14, 2026
>
> **摘要:** Recent advancements in Large Language Models (LLMs) and Large Vision Language Models (LVLMs) have enabled general-purpose systems to demonstrate promising capabilities in complex reasoning tasks, including those in the medical domain. Medical Visual Question Answering (MedVQA) has particularly benefited from these developments. However, despite Bangla being one of the most widely spoken languages globally, there exists no established MedVQA benchmark for it. To address this gap, we introduce BanglaMedVQA, a dataset comprising clinically validated image-question-answer pairs, along with a comprehensive evaluation of current foundation models on this resource. Consistent with prior findings that report low performance of current models on English MedVQA benchmarks, our analysis reveals that Bangla performance is substantially lower, reflecting the challenges inherent to low-resource languages. Even top-performing models such as Gemini and GPT-4.1 mini fail to accurately answer specialized diagnostic questions, indicating severe limitations in fine-grained medical reasoning. Although certain open-source models, such as Gemma-3, occasionally outperform these models in general categories, they too struggle with clinically complex questions, underscoring the urgent need for top-notch evaluation method.
>
---
#### [new 069] KVDrive: A Holistic Multi-Tier KV Cache Management System for Long-Context LLM Inference
- **分类: cs.CL**

- **简介: 该论文属于大模型推理任务，解决长上下文KV缓存管理问题。提出KVDrive系统，通过多级内存协同优化，提升推理吞吐量并降低延迟。**

- **链接: [https://arxiv.org/pdf/2605.18071](https://arxiv.org/pdf/2605.18071)**

> **作者:** Jian Lin; Jiazhi Mi; Zicong Hong; Haodong Wang; Qianli Liu; Haodyue Zhang; Peng Li; Song Guo
>
> **摘要:** Supporting long-context LLMs is challenging due to the substantial memory demands of the key-value (KV) cache. Existing offloading systems store the full cache in host memory and selectively fetch critical entries during decoding, but this strategy quickly hits a ceiling: sparsity cannot be pushed further without degrading accuracy. As a result, when context length and batch size grow, the volume of KV transfers rises sharply and becomes the dominant source of decoding latency. We present KVDrive, a holistic multi-tier KV cache management system spanning GPU memory, host DRAM, and SSD. Unlike prior work that pursues greater sparsity through algorithmic refinements, KVDrive tackles the problem from a systems perspective - jointly orchestrating cache placement, pipeline scheduling, and cross-tier coordination to sustain high-throughput inference under tight GPU budgets. KVDrive advances three fundamental capabilities: it adapts cache management to attention behavior to maximize reuse and minimize redundant data movement; it restructures the decoding pipeline to overlap I/O- and CPU/GPU compute-bound stages, eliminating stalls across heterogeneous resources; and it harmonizes data movement across memory tiers to unlock scalable long-context inference far beyond GPU and DRAM limits. We have implemented a fully functional prototype of KVDrive and evaluated it on long-context benchmarks with popular LLMs. The system achieves up to 1.74x higher throughput compared to state-of-the-art works while preserving accuracy.
>
---
#### [new 070] Skills on the Fly: Test-Time Adaptive Skill Synthesis for LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SkillTTA方法，解决LLM代理在测试时需动态生成特定技能的问题。通过检索并合成相关训练轨迹，生成临时任务技能，提升任务完成效果。**

- **链接: [https://arxiv.org/pdf/2605.16986](https://arxiv.org/pdf/2605.16986)**

> **作者:** Jingxing Wang; Chenyu Zhou; Zhihui Fu; Jun Wang; Weiwen Liu; Weinan Zhang; Jianghao Lin
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** LLM agents benefit from reusable skills, yet test-time tasks often require guidance more specific than a static skill library can provide. We propose \emph{SkillTTA}, a Test-Time Adaptive Skill Synthesis method that retrieves a small set of training trajectories relevant to the current task and synthesizes them into a temporary, task-specific textual skill. The solver model is kept fixed, so adaptation happens entirely through generated context rather than parameter updates. We evaluate the method on SpreadsheetBench, ALFWorld, and BigCodeBench. Compared with static trajectory-to-skill synthesis using GPT-5.5, task-specific skills improve SpreadsheetBench Pass@1 from 0.397 to 0.505 and BigCodeBench Pass@1 from 0.517 to 0.651. On ALFWorld, the method matches a heavier memory-learning baseline within four points of success rate while producing the shortest successful trajectories among reported methods. Ablations on SpreadsheetBench further show that synthesized skills outperform raw trajectory prompting, that top-$k$ retrieval should stay small, and that failed trajectories are especially useful because they expose recurring evaluator-facing mistakes.
>
---
#### [new 071] Why Do Safety Guardrails Degrade Across Languages?
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理中的安全评估任务，旨在解决多语言下安全防护机制退化的问题。通过引入IRT框架，分解安全因素，分析模型在不同语言中的安全表现。**

- **链接: [https://arxiv.org/pdf/2605.17173](https://arxiv.org/pdf/2605.17173)**

> **作者:** Max Zhang; Ameen Patel; Sang T. Truong; Sanmi Koyejo
>
> **摘要:** Large language models exhibit safety degradation in non-English languages. Standard evaluation relies on Jailbreak Success Rate (JSR), which confounds several safety-driving factors into one, obscuring the specific cause(s) of safety failure. We introduce a latent variable model, a Multi-Group Item Response Theory (IRT) framework, that decouples safety-driving factors such as language-agnostic safety robustness ($\theta$), intrinsic prompt hardness ($\beta$), global language processing difficulty ($\gamma$), and a prompt-specific cross-lingual safety gap ($\tau$). Using the MultiJail dataset, we evaluate the safety robustness of 61 model configurations across 5 closed-model families and 10 languages of varying resource, aggregating a dataset of 1.9 million rows. Exploratory Factor Analysis shows safety is primarily unidimensional: models refuse different harm types mainly through a shared mechanism. Contrary to the expected trend that safety degrades largely in low-resource languages, 22 model configurations are more vulnerable in English than in low-resource languages. Low-resource languages produce more uncertain responses (high entropy) than high-resource languages. Also, high-$\tau$ prompts cluster in physical harm categories like Theft and Weapons and lower-resource languages, trends validated through cross-dataset generalization. While global translation quality shows low correlation with $\tau$, severe mistranslations drive high-bias outliers, as validated by native speakers. Cultural and conceptual grounding mismatches also contribute to $\tau$. In predictive validation, the IRT framework achieves $\mathrm{AUC} = 0.940$, outperforming simpler baselines in predicting safe refusal of unsafe prompts. Our framework reveals concept-language vulnerabilities that aggregate metrics obscure, enabling fairer cross-lingual safety evaluation and targeted improvements in dataset construction.
>
---
#### [new 072] RTI-Bench: A Structured Dataset for Indian Right-to-Information Decision Analysis
- **分类: cs.CL**

- **简介: 该论文提出RTI-Bench，一个结构化数据集，用于印度信息公开行政决定分析。任务是信息理解与预测，解决公众难以解读CIC决定的问题。工作包括数据收集、标注及基准测试。**

- **链接: [https://arxiv.org/pdf/2605.16843](https://arxiv.org/pdf/2605.16843)**

> **作者:** Joy Bose
>
> **备注:** 8 pages, 4 tables
>
> **摘要:** India's Right to Information Act, 2005 gives every citizen the right to demand information from public authorities, yet in practice most people cannot make sense of the dense administrative language used in Central Information Commission (CIC) decisions, let alone predict whether an appeal is worth filing. This paper introduces RTI-Bench, a structured dataset of CIC decisions with outcome labels, exemption citations, IRAC-style reasoning components, and procedural timelines. To the best of our knowledge it is the first publicly released structured dataset for Indian RTI administrative decisions. The dataset draws from two sources: 1,218 cases from a publicly available instruction-response corpus (with structured fields added through rule-based extraction), and 298 CIC decision PDFs collected directly from the Commission portal, spanning five commissioners and three document format generations from 2023 to 2026. Label coverage reaches 89% on the instruction-response corpus. For the PDF subset of 239 primary decisions, coverage is 51% in this first release. A random sample of 50 labelled cases was manually reviewed, yielding a label precision of 95.3%. A zero-shot Mistral 7B baseline on 100 cases gives 57.3% accuracy and 37.0% macro-F1 on outcome prediction, well above the majority-class baseline of 14.3% macro-F1. RTI-Bench is available at this https URL
>
---
#### [new 073] LLMs for automatic annotation of Mandarin narrative transcripts
- **分类: cs.CL**

- **简介: 该论文研究LLMs在中文叙事转录的语篇标注任务中的应用，旨在解决人工标注耗时的问题。通过对比模型与人类标注者，评估其在叙事结构标注中的效果。**

- **链接: [https://arxiv.org/pdf/2605.17205](https://arxiv.org/pdf/2605.17205)**

> **作者:** Qingwen Zhao; Hongao Zhu; Yunqi He; Rui Wang; Aijun Huang; Hai Hu
>
> **备注:** 28 pages, 9 tables
>
> **摘要:** Linguistic annotation of transcribed speech is essential for research in language acquisition, language disorders, and sociolinguistics, yet remains labor-intensive and time-consuming. While Large Language Models (LLMs) have shown promise in automating annotation tasks, their ability to handle complex discourse-level annotation in non-English languages remains understudied. This study evaluates whether LLMs can reliably annotate narrative macrostructure-the hierarchical organization of story grammar elements-in spoken Mandarin, using the Multilingual Assessment Instrument for Narratives (MAIN) as a testbed. We compared four LLMs against trained human annotators on narratives produced by children, young adults, and older adults. The best-performing model achieved agreement with human raters (k=.794) approaching human-human reliability levels (k=.872) while reducing annotation time by 65%, whereas the locally deployable lightweight model performed substantially worse. Annotation difficulty varied systematically by macrostructure element type, with categories requiring subtle semantic differentiation posing persistent challenges. Furthermore, model reliability decreased on young adult narratives, which exhibited greater lexical variation, semantic ambiguity, and multi-element integration within single utterances. These findings suggest that LLMs can effectively support discourse-level annotation in non-English spoken corpora, while highlighting the continued need for human oversight in semantically complex tasks. Our prompt templates are open sourced for future use.
>
---
#### [new 074] EnvFactory: Scaling Tool-Use Agents via Executable Environments Synthesis and Robust RL
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决工具使用代理的环境构建和训练数据不足问题。通过自动合成真实多轮轨迹，提升训练效率与性能。**

- **链接: [https://arxiv.org/pdf/2605.18703](https://arxiv.org/pdf/2605.18703)**

> **作者:** Minrui Xu; Zilin Wang; Mengyi DENG; Zhiwei Li; Zhicheng Yang; Xiao Zhu; Yinhong Liu; Boyu Zhu; Baiyu Huang; Chao Chen; Heyuan Deng; Fei Mi; Lifeng Shang; Xingshan Zeng; Zhijiang Guo
>
> **备注:** 11 pages
>
> **摘要:** Equipping LLMs with tool-use capabilities via Agentic Reinforcement Learning (Agentic RL) is bottlenecked by two challenges: the lack of scalable, robust execution environments and the scarcity of realistic training data that captures implicit human reasoning. Existing approaches depend on costly real-world APIs, hallucination-prone LLM simulators, or synthetic environments that are often single-turn or depend on pre-collected documents. Moreover, synthetic trajectories are frequently over-specified, resembling instruction sequences rather than natural human intents, reducing their effectiveness for RL training. We introduce EnvFactory, a fully automated framework that addresses both challenges. EnvFactory autonomously explores and verifies stateful, executable tool environments from authentic resources, and synthesizes natural multi-turn trajectories through topology-aware sampling and calibrated refinement, producing grounded queries with implicit intents. Using only 85 verified environments across 7 domains, EnvFactory generates 2,575 SFT and RL trajectories. Despite using significantly fewer environments than prior work, which are often 5 times more, EnvFactory achieves superior training efficiency and downstream performance, improving Qwen3-series models by up to +15% on BFCLv3, +8.6% on MCP-Atlas, and +6% on conversational benchmarks including $\tau^2$-Bench and VitaBench. By fully automating both environment construction and trajectory synthesis, EnvFactory provides a scalable, extensible, and robust foundation for Agentic RL.
>
---
#### [new 075] Easier to Judge than to Find: Predicting In-Context Learning Success for Demonstration Selection
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的上下文学习任务，旨在解决演示选择效率低的问题。提出DiSP框架，通过样本和判断策略提升预测成功率并加快推理速度。**

- **链接: [https://arxiv.org/pdf/2605.18512](https://arxiv.org/pdf/2605.18512)**

> **作者:** Haochun Wang; Chaofen Yang; Jiatong Liu; Jingbo Wang; Zewen Qiang; Sendong Zhao; Bing Qin; Ting Liu
>
> **备注:** ICML 2026
>
> **摘要:** In-context learning (ICL) is highly sensitive to which demonstrations appear in the prompt, but selecting them is expensive because the space of possible demonstration contexts and combinations is enormous. We argue that demonstration selection is \emph{easier to judge than to find}: predicting whether a specific query--context pair $(q,D)$ will succeed is cheaper and more general than searching for an optimal $D^\star$. Based on this insight, we propose DiSP, a sample-and-judge framework that stratifies queries by difficulty. DiSP runs random demonstration trials to estimate success rate of each training query, trains a lightweight router to predict difficulty from the query, and trains level-specific judges for sampled demonstrations. At inference, DiSP performs stop-on-acceptance judging under an explicit budget, emitting diagnostic risk tags when no suitable context is found. Across five classification datasets with Llama~3--8B and Qwen~2.5--7B, DiSP achieves the best average accuracy, improving over strong learned selection baselines by up to 3.4\%, while achieving up to $23\times$ end-to-end wall-clock speedup.
>
---
#### [new 076] Constrained Code Generation with Discrete Diffusion
- **分类: cs.CL; cs.PL**

- **简介: 该论文属于代码生成任务，旨在解决生成代码时的约束满足问题。提出CDC框架，在扩散过程中集成约束优化，提升代码功能、安全与语法正确性。**

- **链接: [https://arxiv.org/pdf/2605.16829](https://arxiv.org/pdf/2605.16829)**

> **作者:** Lize Shao; Michael Cardei; Zichen Xie; Ferdinando Fioretto; Wenxi Wang
>
> **摘要:** Discrete diffusion models are a powerful, emerging paradigm for code generation. They construct programs through iterative refinement of partially corrupted token sequences and enable parallel token refinement. Importantly, this paradigm exposes a global program state at each denoising step, which provides a natural intervention point for enforcing program-level functionality and security constraints, guiding the generation before the final code is committed. Building on this observation, the paper introduces Constrained Diffusion for Code (CDC), a training-free neurosymbolic inference framework that integrates constraint satisfaction directly into the reverse denoising process. CDC augments the base discrete diffusion sampler with constraint-aware denoising operators that combine mathematical optimization with program analysis to identify constraint-relevant regions of the intermediate program state and locally adjust the denoising trajectory, steering generation toward feasible programs while remaining close to the base model. Across code generation benchmarks, CDC consistently improves constraint satisfaction in functional correctness, security, and even syntax, outperforming discrete diffusion and autoregressive baselines with less corrective computation and more localized edits.
>
---
#### [new 077] AutoVecCoder: Teaching LLMs to Generate Explicitly Vectorized Code
- **分类: cs.CL**

- **简介: 该论文属于代码生成任务，旨在解决LLMs在显式向量化上的不足。通过引入VecPrompt和VecRL，提升LLMs生成高效向量代码的能力。**

- **链接: [https://arxiv.org/pdf/2605.17978](https://arxiv.org/pdf/2605.17978)**

> **作者:** Shangzhan Li; Xinyu Yin; Xuanyu Jin; Ye He; Yuxin Zhou; Yuxuan Li; Xu Han; Wanxiang Che; Qi Shi; Ting Liu; Maosong Sun
>
> **摘要:** Vectorization via Single Instruction, Multiple Data (SIMD) architectures is a cornerstone of high-performance computing. To fully exploit hardware potential, developers often resort to explicit vectorization using intrinsics, as compiler-based auto-vectorization frequently yields suboptimal results due to conservative static analysis. While Large Language Models (LLMs) have demonstrated remarkable proficiency in general code generation, they struggle with explicit vectorization due to the scarcity of high-quality corpora and the strict semantic constraints of low-level hardware instructions. In this paper, we propose AutoVecCoder, a novel framework designed to empower LLMs with the capability of automated explicit vectorization. AutoVecCoder integrates two core components: VecPrompt, an automated data synthesis pipeline to inject domain-specific intrinsic knowledge; and VecRL, a reinforcement learning framework that aligns code generation with execution efficiency. AutoVecCoder-8B trained by this framework achieves state-of-the-art performance on the SSE and AVX subsets of SimdBench and, in some cases, generates implementations surpassing standard -O3 optimizations, effectively overcoming the inherent bottlenecks of traditional automated vectorization.
>
---
#### [new 078] LongMINT: Evaluating Memory under Multi-Target Interference in Long-Horizon Agent Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LongMINT基准，用于评估长时空中多目标干扰下的记忆性能，解决记忆干扰与多源信息推理问题。**

- **链接: [https://arxiv.org/pdf/2605.18565](https://arxiv.org/pdf/2605.18565)**

> **作者:** Hyunji Lee; Justin Chih-Yao Chen; Joykirat Singh; Zaid Khan; Elias Stengel-Eskin; Mohit Bansal
>
> **备注:** Equal contribution; order decided by a coin flip. Code and data: this https URL
>
> **摘要:** Real-world agents operate over long and evolving horizons, where information is repeatedly updated and may interfere across memories, requiring accurate recall and aggregated reasoning over multiple pieces of information. However, existing benchmarks focus on static, independent recall and fail to capture these dynamic interactions between evolving memories. In this paper, we study how current memory-augmented agents perform in realistic, interference-heavy, long-horizon settings across diverse domains and question types. We introduce LongMINT (Long-Horizon Memory under INTerference), a benchmark featuring (1) long, highly interconnected contexts with frequently updated information that induces substantial interference, (2) diverse domains (state tracking, multi-turn dialogue, Wikipedia revisions, and GitHub commits), enabling evaluation of domain generalization, and (3) diverse question types that assess robustness to interference, including (i) single-target recall tasks requiring retrieval of a specific target from long contexts, and (ii) multi-target aggregation tasks requiring reasoning over multiple relevant pieces of information. Overall, LongMINT has 15.6k question-answering pairs over long-horizon contexts averaging 138.8k tokens and extending up to 1.8M tokens per instance. We evaluate 7 representative systems, including vanilla long-context LLMs, RAG, and memory-augmented agent frameworks. Across all systems, we observe consistently low performance (avg. 27.9% accuracy), especially on questions requiring aggregated reasoning over multiple pieces of evidence. Our analysis shows that performance is primarily limited by retrieval and memory construction. Furthermore, current memory systems struggle to recall and reason over earlier facts that are later revised or interfered with by subsequent context, with performance degrading as the number of intervening updates increases.
>
---
#### [new 079] Prompt Compression in Diffusion Large Language Models: Evaluating LLMLingua-2 on LLaDA
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究扩散大语言模型的提示压缩问题，评估LLMLingua-2在LLaDA上的效果，发现压缩方法在自回归模型中有效，在扩散模型中表现不一，需发展专用策略。**

- **链接: [https://arxiv.org/pdf/2605.17932](https://arxiv.org/pdf/2605.17932)**

> **作者:** Sterling Huang; Abigayle Brown; Jiyoo Noh; Jiakang Xu; Wantong Huo; Kaung Myat Kyaw; Jonathan Chan
>
> **摘要:** Prompt compression reduces inference cost and context length in large language models, but prior evaluations focus primarily on autoregressive architectures. This study investigates whether prompt compression transfers effectively to diffusion large language models (DLLMs) using LLMLingua-2, specifically the 8B-parameter DLLM LLaDA. We evaluate compression performance on GSM8K, DUC2004, and ShareGPT using 250 prompts per dataset at an approximate 2$\times$ compression ratio, across mathematical reasoning, prompt reconstruction, and summarization tasks. Outputs generated from original prompts, compressed prompts, reconstructed prompts, and reconstructed-prompt reasoning were compared using exact-match accuracy, BLEU, ROUGE, and BERTScore. Results show that semantic preservation does not necessarily imply stable downstream behavior in diffusion models. Summarization tasks remained comparatively robust under compression, while mathematical reasoning degraded substantially despite high semantic similarity scores. Reconstruction experiments further showed that semantically similar prompts may still omit reasoning-critical information required for stable denoising. Across tasks, BERTScore recall was consistently lower than precision, suggesting that compression failures are primarily driven by information omission rather than semantic drift. These findings indicate that prompt compression methods designed for autoregressive models do not transfer uniformly to diffusion large language models and motivate the development of diffusion-aware compression strategies.
>
---
#### [new 080] Response-free item difficulty modelling for multiple-choice items with fine-tuned transformers: Component-wise representation and multi-task learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于阅读理解题目难度建模任务，旨在无需答题数据的情况下预测多选题难度。通过微调Transformer模型，结合组件编码和多任务学习，提升小样本下的建模效果。**

- **链接: [https://arxiv.org/pdf/2605.16991](https://arxiv.org/pdf/2605.16991)**

> **作者:** Jan Netík; Patrícia Martinková
>
> **摘要:** Response-free item difficulty modelling promises to reduce reliance on response-based calibration but is intrinsically difficult on reading-comprehension multiple-choice items, where difficulty depends on inferential demands across wording components. Whereas most existing approaches extract item-text features and pass them to a separate statistical or machine-learning model, we fine-tune transformer encoders end-to-end on the item wording, eliminating the manual feature engineering and preprocessing that discards information. Moreover, two extensions to this joint-encoding approach are proposed: a component-wise variant that encodes wording components separately through a shared encoder, and a multi-task variant that retains joint encoding and adds an auxiliary multiple-choice question answering objective on the shared encoder. Each method is evaluated under a Monte Carlo subsampling design at three training-set sizes on a held-out test set. We find that joint encoding is a viable end-to-end alternative to feature-engineering pipelines; while the component-wise variant shows no detectable benefit, consistent with self-attention already harvesting the cross-component signal, the multi-task variant delivers significant paired improvements in the smallest-sample regime. Transformer fine-tuning, especially if regularised by a suitable auxiliary task, recovers a substantial share of the wording-derivable signal at training-set sizes typical of applied measurement. The framework provides a customisable interface for psychometrically motivated extensions.
>
---
#### [new 081] BELIEF: Structured Evidence Modeling and Uncertainty-Aware Fusion for Biomedical Question Answering
- **分类: cs.CL**

- **简介: 该论文属于生物医学问答任务，解决文献证据结构化与不确定性融合问题。提出BELIEF框架，通过结构化证据和双路径推理提升问答准确性。**

- **链接: [https://arxiv.org/pdf/2605.17435](https://arxiv.org/pdf/2605.17435)**

> **作者:** Chang Zong; Hao Ning; Siliang Tang; Jie Huang; Jian Wan
>
> **备注:** 14 pages, 6 figures
>
> **摘要:** Biomedical question answering often requires decisions from retrieved literature whose relevance, quality, and support for candidate answers are uneven. Most retrieval-augmented large language model (LLM) methods feed this literature to the model as flat text, leaving evidence reliability and remaining uncertainty largely implicit. We propose BELIEF, a structured evidence modeling and uncertainty-aware fusion framework for closed-set biomedical question answering. Rather than treating retrieved documents as undifferentiated context, BELIEF converts them into evidence objects that record clinical attributes, source quality, question relevance, support strength, and the associated candidate hypothesis. These evidence objects provide a shared basis for two complementary reasoning paths. The symbolic path constructs reliability-weighted basic probability assignments based on Dempster--Shafer (D-S) theory over a finite answer space and performs uncertainty-aware symbolic evidence fusion to estimate belief and residual uncertainty. The neural path uses the same structured evidence for LLM-based semantic inference, while a reliability-aware arbitration module reconciles the symbolic and neural outputs according to belief strength, uncertainty, evidence reliability, and semantic consistency. Experiments on PubMedQA, MedQA, and MedMCQA with five general-purpose LLM backbones show that BELIEF obtains the best result in 25 of 30 backbone--dataset--metric settings. Comparisons with biomedical-domain models indicate that BELIEF is competitive on MedQA and MedMCQA, while specialized biomedical pretraining remains advantageous on PubMedQA. Ablation, complementarity, uncertainty-stratified, and cost analyses further show that BELIEF improves retrieved-evidence utilization by making evidence structure, path disagreement, and decision uncertainty explicit.
>
---
#### [new 082] The Point of No Return: Counterfactual Localization of Deceptive Commitment in Language-Model Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型中欺骗性承诺的定位问题，通过构建环境并分析推理轨迹，识别模型何时开始欺骗。任务属于自然语言处理中的 deception detection。**

- **链接: [https://arxiv.org/pdf/2605.17113](https://arxiv.org/pdf/2605.17113)**

> **作者:** Scott Merrill; Shashank Srivastava
>
> **备注:** 41 pages, 25 figures
>
> **摘要:** Existing deception datasets label completed outputs as honest or deceptive, treating deception as a property of the final response rather than a function of the model's reasoning trace. This obscures a more fundamental question: when does a language model become committed to deception? We introduce counterfactual localization: for each sentence prefix in a reasoning trace, we fix the prefix, resample continuations, and estimate the probability of a deceptive outcome. To scale this, we construct five environments (spanning strategic bluffing, maze guidance, financial advice, used-car sales, and offer negotiation) in which deception is never prompted but emerges from strategic incentives and labels follow mechanically from environment state rather than subjective human judgment. The resulting corpus localizes $\sim$1.46M sentences across four reasoning models, drawn from over 94.1M sampled continuations, 91.5B generated tokens, and over 100K scenarios. Sentence-level human evaluation confirms that detected commitment points correspond to interpretable shifts in decision state. Using this resource, we show that lexical cues for commitment prediction transfer poorly across environments, whereas attention-based transition features generalize out of distribution, suggesting that deceptive commitment is reflected in reusable changes in reasoning dynamics rather than surface form. We further identify compact attention-head sets (under 10% of heads) that, selected on one environment, causally suppress deceptive commitment across held-out environments. We release the corpus as a substrate for studying deception, and more broadly commitment, in language-model reasoning.
>
---
#### [new 083] Continuous Diffusion Scales Competitively with Discrete Diffusion for Language
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **简介: 该论文属于语言建模任务，旨在解决连续扩散模型可扩展性不足的问题。通过改进架构，提出RePlaid模型，在性能上接近甚至超越离散模型，验证了连续扩散的竞争力。**

- **链接: [https://arxiv.org/pdf/2605.18530](https://arxiv.org/pdf/2605.18530)**

> **作者:** Zhihan Yang; Wei Guo; Shuibai Zhang; Subham Sekhar Sahoo; Yongxin Chen; Arash Vahdat; Morteza Mardani; John Thickstun
>
> **摘要:** While diffusion has drawn considerable recent attention from the language modeling community, continuous diffusion has appeared less scalable than discrete approaches. To challenge this belief we revisit Plaid, a likelihood-based continuous diffusion language model (DLM), and construct RePlaid by aligning the architecture of Plaid with modern discrete DLMs. In this unified setting, we establish the first scaling law for continuous DLMs that rivals discrete DLMs: RePlaid exhibits a compute gap of only $20\times$ compared to autoregressive models, outperforms Duo while using fewer parameters, and outperforms MDLM in the over-trained regime. We benchmark RePlaid against recent continuous DLMs: on OpenWebText, RePlaid achieves a new state-of-the-art PPL bound of $22.1$ among continuous DLMs and superior generation quality. These results suggest that continuous diffusion, when trained via likelihood, is a highly competitive and scalable alternative to discrete DLMs. Moreover, we offer theoretical insights to understand the advantage of likelihood-based training. We show that optimizing the noise schedule to minimize the ELBO's variance naturally yields linear cross-entropy (information loss) over time. This evenly distributes denoising difficulty without any case-specific time reparameterization. In addition, we find that optimizing embeddings via likelihood creates structured geometries and drives the most significant likelihood gain.
>
---
#### [new 084] Forecasting Downstream Performance of LLMs With Proxy Metrics
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型性能预测任务，旨在解决传统指标（如交叉熵损失）与下游性能关联弱的问题。通过构建代理指标，利用token级统计信息提升模型选择、数据筛选和训练预测的准确性。**

- **链接: [https://arxiv.org/pdf/2605.18607](https://arxiv.org/pdf/2605.18607)**

> **作者:** Arkil Patel; Siva Reddy; Marius Mosbach; Dzmitry Bahdanau
>
> **备注:** Preprint. 31 pages
>
> **摘要:** Progress in language model development is often driven by comparative decisions: which architecture to adopt, which pretraining corpus to use, or which training recipe to apply. Making these decisions well requires reliable performance forecasts, yet the two commonly used signals are fundamentally limited. Cross-entropy loss is poorly aligned with downstream capabilities, and direct downstream evaluation is expensive, sparse, and often uninformative at early training stages. Instead, we propose to construct proxy metrics by aggregating token-level statistics, such as entropy, top-k accuracy, and expert token rank, from a candidate model's next token distribution over expert-written solutions. Across three settings, our proxies consistently outperform loss- and compute-based baselines: 1) For cross-family model selection, they rank a heterogeneous population of reasoning models with mean Spearman Rho = 0.81 (vs. Rho = 0.36 for cross-entropy loss); 2) For pretraining data selection, they reliably rank 25 candidate corpora for a target model at roughly $10{,}000\times$ less compute than direct evaluation, pushing the Pareto frontier beyond existing methods; and 3) for training-time forecasting, they extrapolate downstream accuracy across an $18\times$ compute horizon with roughly half the error of existing alternatives. Together, these results suggest that expert trajectories are a broadly useful source of signal for assessing model capabilities, enabling reliable performance forecasting throughout the model development life cycle.
>
---
#### [new 085] SkillsVote: Lifecycle Governance of Agent Skills from Collection, Recommendation to Evolution
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SkillsVote框架，解决Agent技能生命周期管理问题，通过收集、推荐和演化技能提升代理性能。**

- **链接: [https://arxiv.org/pdf/2605.18401](https://arxiv.org/pdf/2605.18401)**

> **作者:** Hongyi Liu; Haoyan Yang; Tao Jiang; Bo Tang; Feiyu Xiong; Zhiyu Li
>
> **备注:** 44 pages, 7 figures, 5 tables
>
> **摘要:** Long-horizon LLM agents leave traces that could become reusable experience, but raw trajectories are noisy and hard to govern. We treat Agent Skills as an experience schema that couples executable scripts, with non-executable guidance on procedures. Yet open skill ecosystems contain redundant, uneven, environment-sensitive artifacts, and indiscriminate updates can pollute future context. We present SkillsVote, a lifecycle-governance framework for Agent Skills from collection and recommendation to evolution. SkillsVote profiles a million-scale open-source corpus for environment requirements, quality, and verifiability, then synthesizes tasks for verifiable skills. Before execution, SkillsVote performs agentic library search over structured skill library to expose instructional skill context. After execution, it decomposes trajectories into skill-linked subtasks, attributes outcomes to skill use, agent exploration, environment, and result signals, and admits only successful reusable discoveries to evidence-gated updates. In our evaluation, offline evolution improves GPT-5.2 on Terminal-Bench 2.0 by up to 7.9 pp, while online evolution improves SWE-Bench Pro by up to 2.6 pp. Overall, governed external skill libraries can improve frozen agents without model updates when systems control exposure, credit, and preservation.
>
---
#### [new 086] Generating Pretraining Tokens from Organic Data for Data-Bound Scaling
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决数据受限下的大模型预训练问题。提出SynPro框架，通过重述和格式转换生成合成数据，提升有机数据利用率。**

- **链接: [https://arxiv.org/pdf/2605.17849](https://arxiv.org/pdf/2605.17849)**

> **作者:** Zichun Yu; Chenyan Xiong
>
> **摘要:** LLM pretraining is shifting from a compute-bound to a data-bound regime, where available human (organic) text falls far short of scaling demands. However, reaching the data-bound regime does not mean the model has fully utilized its organic corpus. In this paper, we introduce SynPro, a synthetic data generation framework that helps LLMs more thoroughly learn from limited organic data. SynPro applies two operations, rephrasing and reformat, that present the same organic source in diverse forms to facilitate deeper learning without introducing external information. Both generators are optimized via reinforcement learning with quality, faithfulness, and data influence rewards, and are continuously updated as pretraining plateaus to target content the model has yet to absorb. We pretrain 400M and 1.1B models with 10% of their Chinchilla-optimal tokens (0.8B and 2.2B) from DCLM-Baseline, reflecting a realistic data-bound regime in frontier pretraining. Our results reveal that organic data is significantly underutilized by standard repetition: SynPro unlocks 3.7-5.2x the effective tokens of repetition, even surpassing the non-data-bound oracle that trains on equivalent unique data at the 1.1B scale. Analyses confirm that faithful, model-aware synthesis sustains data-bound scaling without causing distribution collapse. We open-source our code at this https URL.
>
---
#### [new 087] Ancient Greek to Modern Greek Machine Translation: A Novel Benchmark and Fine-Tuning Experiments on LLMs and NMT Models
- **分类: cs.CL**

- **简介: 该论文属于古希腊语到现代希腊语的机器翻译任务，解决低资源数据不足的问题。构建了新的平行语料库，并进行了模型微调实验，提升了翻译性能。**

- **链接: [https://arxiv.org/pdf/2605.18504](https://arxiv.org/pdf/2605.18504)**

> **作者:** Spyridon Mavromatis; Sokratis Sofianopoulos; Prokopis Prokopidis; Maria Giagkou
>
> **备注:** 14 pages. Accepted for presentation at the 15th Language Resources and Evaluation Conference (LREC 2026), Palma, Mallorca, Spain
>
> **摘要:** Machine Translation (MT) for Ancient Greek (AG) to Modern Greek (MG) is a low-resource task, constrained by the lack of large-scale, high-quality parallel data. We address this gap by introducing the AG-MG Parallel Corpus, a new resource containing 132,481 sentence-aligned pairs derived from literary, historical, and biblical texts. We present a novel corpus creation pipeline that combines web-scraped, excerpt-level data with a multi-stage sentence-level alignment, and refinement process. Our method uses VecAlign with LaBSE embeddings, which we first fine-tune on a manually-aligned AG-MG subset, followed by an LLM-based error/misalignment correction phase using Gemini 2.5 Flash to ensure high alignment quality. Furthermore, we provide the first comprehensive benchmark of modern MT models on this task, evaluating three fine-tuning strategies across NMT models (NLLB, M2M100) and a Greek LLM (Llama-Krikri-8B). Our experiments show that fine-tuning yields significant improvements over base models, increasing performance by up to +10.3 BLEU points. Specifically, full-parameter fine-tuning of Llama-Krikri-8B achieves the highest overall performance with a BLEU score of 13.16, while the QLoRA-adapted M2M100-1.2B model demonstrates the largest relative gains and highly competitive results. Our dataset and models represent a significant contribution to Greek NLP.
>
---
#### [new 088] Universal Adversarial Triggers
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究对抗攻击任务，旨在生成自然且有效的触发词以干扰情感分析模型。通过结合词性过滤和困惑度损失函数，提升攻击效果并增强模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.17936](https://arxiv.org/pdf/2605.17936)**

> **作者:** Benedict Florance Arockiaraj; Alexander Feng; Jianxiong Cai; Xiaoyu Cheng
>
> **摘要:** Recent works have illustrated that modern NLP models trained for diverse tasks ranging from sentiment analysis to language generation succumb to universal adversarial attacks, a class of input-agnostic attacks where a common trigger sequence is used to attack the model. Although these attacks are successful, the triggers generated by such attacks are ungrammatical and unnatural. Our work proposes a novel technique combining parts-of-speech filtering and perplexity based loss function to generate sensible triggers that are closer to natural phrases. For the task of sentiment analysis on the SST dataset, the method produces sensible triggers that achieve accuracies as low as 0.04 and 0.12 for flipping positive to negative predictions and vice-versa. To build robust models, we also perform adversarial training using the generated triggers that increases the accuracy of the model from 0.12 to 0.48. We aim to illustrate that adversarial attacks can be made difficult to detect by generating sensible triggers, and to facilitate robust model development through relevant defenses.
>
---
#### [new 089] A Scalable Tool for Measuring Manner and Result Verbs in Developmental Language Research
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言学中的语义分类任务，旨在解决 manner 和 result 动词难以大规模测量的问题。通过构建标注数据并训练分类器，实现对两类动词的高效识别。**

- **链接: [https://arxiv.org/pdf/2605.16654](https://arxiv.org/pdf/2605.16654)**

> **作者:** Divyesh Pratap Singh; Dakshesh Gusain; Federica Bulgarelli; Alison Eisel Hendricks; John Beavers; Nathan M. Beers; Ifeoma Nwogu
>
> **备注:** 12 pages
>
> **摘要:** Manner and result verbs encode different aspects of event structure and have been discussed in developmental work as a potentially informative distinction for studying early verb learning. However, this distinction remains difficult to measure at scale because large annotated resources for manner and result classification are not currently available. We present a computational approach for identifying manner and result verbs in sentence context. Using linguistically informed prompts, we generate sentence-level annotations with large language models over data drawn from MASC and InterCorp, extending coverage from previously annotated portions of VerbNet to 436 classes. We then train a RoBERTa-based classifier on these annotations and evaluate it on three held-out gold-standard datasets, including previously annotated items and a new expert-annotated set. Across these evaluations, the model shows promising performance, with average accuracy up to 89.6%. We present this work as a scalable measurement tool that can support future research on verb semantics in developmental and other language datasets, while noting that further validation is needed for borderline cases, mixed manner/result verbs, and downstream developmental applications.
>
---
#### [new 090] A Data-Efficient Path to Multilingual LLMs: Language Expansion via Post-training PARAM$Δ$ Integration into Upcycled MoE
- **分类: cs.CL**

- **简介: 该论文属于多语言大模型扩展任务，旨在解决语言扩展与原有能力保持的冲突。通过将密集模型转为MoE架构并引入参数增量，实现高效多语言扩展。**

- **链接: [https://arxiv.org/pdf/2605.18083](https://arxiv.org/pdf/2605.18083)**

> **作者:** Hao Zhou; Tianhao Li; Zhijun Wang; Shuaijie She; Linjuan Wu; Hao-Ran Wei; Baosong Yang; Jiajun Chen; Shujian Huang
>
> **摘要:** Expanding Large Language Models~(LLMs) to new languages is a costly endeavor, demanding extensive Continued Pre-Training~(CPT) and data-intensive alignment. While recent data-free merging techniques attempt to bypass alignment by fusing a multilingual CPT-enhanced model with its instruct counterpart, they are plagued by a critical trade-off: mitigating parameter conflicts to preserve original abilities inevitably dilutes new language acquisition, and vice-versa. To resolve this conflict, we introduce \method, which upcycles a dense model into a Mixture-of-Experts~(MoE) architecture, allocating different experts to different languages. Alignment ability is then transferred by grafting a MoE-expanded parameter delta~($\Delta_{\text{post}}$) to the CPT-enhanced base model, bypassing the complex alignment phase. Experiments demonstrate \method's superiority even against baselines with similar FLOPs or number of parameters; it improves performance on expanded languages while effectively preserving original capabilities. We further show our approach is highly applicable across different models and Post-training deltas.
>
---
#### [new 091] Can LLMs Think Like Consumers? Benchmarking Crowd-Level Reaction Reconstruction with ConsumerSimBench
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于自然语言处理任务，旨在评估大模型是否能像消费者一样思考。通过构建ConsumerSimBench基准，解决模型在模拟消费者反应上的不足，发现前沿模型表现仍远低于人类水平。**

- **链接: [https://arxiv.org/pdf/2605.17079](https://arxiv.org/pdf/2605.17079)**

> **作者:** Tianyu Wang; Jiajun Li; Jianghao Lin
>
> **摘要:** LLMs are increasingly used as ``digital consumers'' to simulate public opinion, pre-test marketing decisions, and anticipate audience response. However, existing evaluations rarely ask whether a model can reconstruct the concrete reaction patterns that real consumers surface in public discourse. We introduce ConsumerSimBench, a benchmark built from 1,553 real Chinese social-media topics and 23,122 atomic, rule-audited criteria spanning four reaction families. Rather than scoring open-ended generations with a holistic preference judge, ConsumerSimBench decomposes each task into auditable yes-no decisions over concrete reaction points, raising three-judge agreement from 65.8% to 92.1% with 98.4% agreement between pointwise judge decisions and human-majority labels. Across 13 frontier generators, the strongest model, Gemini-3.1-Pro, covers only 47.8% of real reaction criteria, while GPT-5.2 and Claude-4.6 trail far behind despite their strength on technical benchmarks. The failures reveal a sharp gap between technical-benchmark performance and socially grounded consumer intuition. A direct structured reasoning prompt decreases coverage, while a generate--reflect multi-agent pipeline improves MiMo-V2.5-Pro from 32.9% to 37.6% on a subset. ConsumerSimBench reframes consumer simulation as a forecasting problem over real public-discourse reactions, showing that frontier LLMs remain far from reliably predicting what consumers will actually care about in high-context Chinese consumer discourse.
>
---
#### [new 092] PARALLAX: Separating Genuine Hallucination Detection from Benchmark Construction Artifacts
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于 hallucination 检测任务，旨在解决模型输出错误难以检测的问题。研究指出多数基准数据存在漏洞，提出新方法 DRIFT 评估真实检测能力。**

- **链接: [https://arxiv.org/pdf/2605.17028](https://arxiv.org/pdf/2605.17028)**

> **作者:** Khizar Hussain; Murat Kantarcioglu
>
> **备注:** Preprint to Neurips 2026 submission
>
> **摘要:** Large language models (LLMs) hallucinate with confidence: their outputs can be fluent, authoritative, and simply wrong. In medical, legal, and scientific applications this failure causes direct harm, and detecting it from internal model states offers a path to safer deployment. A growing body of work reports that this problem is increasingly tractable, with recent methods achieving high detection performance on widely used benchmarks. We show, however, that much of this apparent progress does not survive scrutiny. Four of the six corpora embed the ground-truth answer directly in the input prompt. A naïve text-similarity baseline we call \textsc{TxTemb} exploits this to achieve near-perfect detection scores without any access to model internals. To measure what genuine detection capability remains once these artifacts are controlled, we conduct a large-scale evaluation spanning twenty-two detection methods, twelve open-source models spanning six architectural families, and six corpora. We further introduce \textbf{DRIFT}, a supervised probe over inter-layer hidden-state transitions, as a point of comparison for live-generation detection. Our findings suggest that the field's reported progress on hallucination detection is substantially explained by benchmark construction artifacts in widely used corpora, and that the majority of established baselines perform near chance under controlled conditions; the consistent exceptions are SAPLMA and DRIFT, both supervised probes on upper-layer hidden states.
>
---
#### [new 093] ACIL: Auto Chain of Thoughts for In-Context Learning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决ICL在多步骤推理任务中的性能不足问题。通过生成推理链增强示例，提升ICL效果。**

- **链接: [https://arxiv.org/pdf/2605.17088](https://arxiv.org/pdf/2605.17088)**

> **作者:** Rui Chu
>
> **摘要:** Recent advances in large language models (LLMs) have shown that Chain-of-Thought (CoT) reasoning can substantially improve performance on complex reasoning tasks. At the same time, In-Context Learning (ICL) has become an important mechanism for adapting LLMs to new tasks without updating model parameters, using only examples provided in the prompt. However, standard ICL often struggles on tasks that require multi-step reasoning, because the demonstrations usually contain only input-output pairs and lack explicit intermediate reasoning steps. This paper introduces an Automatic Chain-of-Thought (Auto-CoT) framework to improve ICL by automatically constructing reasoning-enhanced demonstrations. Auto-CoT generates reasoning chains for input-output examples, augments the prompt context with structured intermediate explanations, and removes irrelevant or low-quality demonstrations through a systematic selection process. By incorporating high-quality reasoning examples into the ICL prompt, Auto-CoT guides the model toward more reliable reasoning and improves prediction accuracy. Experiments across multiple reasoning tasks demonstrate that the proposed framework improves ICL performance by providing explicit intermediate reasoning guidance.
>
---
#### [new 094] Multilingual and Multimodal LLMs in the Wild: Building for Low-Resource Languages
- **分类: cs.CL**

- **简介: 该论文属于多语言多模态AI任务，旨在解决低资源语言下模型训练与应用问题，提出低成本数据方法和适配器技术，支持视觉、语音与文本的联合处理。**

- **链接: [https://arxiv.org/pdf/2605.17152](https://arxiv.org/pdf/2605.17152)**

> **作者:** Firoj Alam; Shammur Absar Chowdhury; Enamul Hoque Prince
>
> **备注:** Multimodal Foundation Models, Large Language Models, Native, Multilingual, Language Diversity, Low-resources-language
>
> **摘要:** Multimodal LLMs are evolving from vision-language to tri-modality that see, hear, and read, yet pipelines and benchmarks remain English-centric and compute-heavy. The tutorial offers an overview of this emerging research area for multilingual multimodality across text, speech, and vision under limited data/compute budgets, synthesizing foundations, recent multilingual models (PALO, Maya), speech-text LLMs. We cover low-cost data creation/curation; adapter stacks for tri-modal alignment; culture-aware evaluation beyond English and hands on resources for fine-tuning a compact multilingual VLM and wiring a speech->text->LLM pipeline. The content will be delivered as an interactive half-day tutorial, designed for researchers and practitioners working on multilingual, multimodal AI in low-resource language settings.
>
---
#### [new 095] Code as Agent Harness
- **分类: cs.CL; cs.AI**

- **简介: 该论文探讨代码作为智能体基础设施的统一视角，解决如何构建可执行、可验证的智能体系统问题，涵盖接口、机制与多智能体扩展等层面。**

- **链接: [https://arxiv.org/pdf/2605.18747](https://arxiv.org/pdf/2605.18747)**

> **作者:** Xuying Ning; Katherine Tieu; Dongqi Fu; Tianxin Wei; Zihao Li; Yuanchen Bei; Jiaru Zou; Mengting Ai; Zhining Liu; Ting-Wei Li; Lingjie Chen; Yanjun Zhao; Ke Yang; Bingxuan Li; Cheng Qian; Gaotang Li; Xiao Lin; Zhichen Zeng; Ruizhong Qiu; Sirui Chen; Yifan Sun; Xiyuan Yang; Ruida Wang; Rui Pan; Chenyuan Yang; Dylan Zhang; Liri Fang; Zikun Cui; Yang Cao; Pan Chen; Dorothy Sun; Ren Chen; Mahesh Srinivasan; Nipun Mathur; Yinglong Xia; Hong Li; Hong Yan; Pan Lu; Lingming Zhang; Tong Zhang; Hanghang Tong; Jingrui He
>
> **备注:** GitHub: this https URL
>
> **摘要:** Recent large language models (LLMs) have demonstrated strong capabilities in understanding and generating code, from competitive programming to repository-level software engineering. In emerging agentic systems, code is no longer only a target output. It increasingly serves as an operational substrate for agent reasoning, acting, environment modeling, and execution-based verification. We frame this shift through the lens of agent harnesses and introduce code as agent harness: a unified view that centers code as the basis for agent infrastructure. To systematically study this perspective, we organize the survey around three connected layers. First, we study the harness interface, where code connects agents to reasoning, action, and environment modeling. Second, we examine harness mechanisms: planning, memory, and tool use for long-horizon execution, together with feedback-driven control and optimization that make harness reliable and adaptive. Third, we discuss scaling the harness from single-agent systems to multi-agent settings, where shared code artifacts support multi-agent coordination, review, and verification. Across these layers, we summarize representative methods and practical applications of code as agent harness, spanning coding assistants, GUI/OS automation, embodied agents, scientific discovery, personalization and recommendation, DevOps, and enterprise workflows. We further outline open challenges for harness engineering, including evaluation beyond final task success, verification under incomplete feedback, regression-free harness improvement, consistent shared state across multiple agents, human oversight for safety-critical actions, and extensions to multimodal environments. By centering code as the harness of agentic AI, this survey provides a unified roadmap toward executable, verifiable, and stateful AI agent systems.
>
---
#### [new 096] GUT-IS: A Data-Driven Approach to Integrating Constructs and Their Relations in Information Systems
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于信息系统研究任务，旨在解决构造定义不一致问题。通过整合结构方程模型，利用文本嵌入和聚类方法生成构造分组，并优化语义纯度与简洁性之间的权衡。**

- **链接: [https://arxiv.org/pdf/2605.18567](https://arxiv.org/pdf/2605.18567)**

> **作者:** Maximilian Reinhardt; Jonas Scharfenberger; Burkhardt Funk
>
> **备注:** Accepted at the 34th European Conference on Information Systems (ECIS 2026), Milan, Italy
>
> **摘要:** Structural equation modeling is widely used in IS research. However, inconsistent construct definitions impede the cumulative development of knowledge. In this work, we present an approach that aims at the integration of structural equation models into a unified model: We use a combination of task-adapted text embeddings and clustering to produce a candidate set of construct groupings. Subsequently, we select the optimal solution using a loss function that explicitly trades off semantic purity and parsimony in the number of clusters. By making this trade-off explicit, our approach allows to analyze how construct groupings and their relations change as one shifts the priority from purity to parsimony. Empirically, we evaluate and explore the proposed methodology on two datasets from the IS domain.
>
---
#### [new 097] Effort as Ceiling, Not Dial: Reasoning Budget Does Not Modulate Cognitive Cost Alignment Between Humans and Large Reasoning Models
- **分类: cs.CL; cs.AI; q-bio.NC**

- **简介: 该论文属于认知与模型对比研究，探讨大模型与人类在推理成本上的对齐机制。通过实验发现，模型推理过程中的努力程度不影响这种对齐，表明其为训练时形成的固定策略。**

- **链接: [https://arxiv.org/pdf/2605.16938](https://arxiv.org/pdf/2605.16938)**

> **作者:** Yueqing Hu; Tianhong Wang
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Large Reasoning Models (LRMs) generate chain-of-thought traces whose length tracks human reaction times across cognitive tasks, but recent debate questions whether this alignment reflects genuine computational structure or surface verbosity. We test whether the alignment varies with inference-time reasoning effort. Across GPT-OSS-20B and GPT-OSS-120B, three effort levels, and six reasoning tasks, within-task and cross-task alignment remain invariant: Bayes Factors lean toward the null, and mean alignment is numerically near-identical across conditions. A manipulation check reveals that the effort parameter sets an upper budget on generation rather than driving real-time allocation, suggesting that the allocation policy is crystallized at training time. Arithmetic complexity contrasts further show that token allocation tracks fine-grained, format-dependent human difficulty patterns, with model scale improving the match. Cognitive cost alignment between LRMs and humans appears to be a training-time achievement, robust to inference-time perturbations, supporting a compiled rather than online account of LRM problem-solving.
>
---
#### [new 098] Vector RAG vs LLM-Compiled Wiki: A Preregistered Comparison on a Small Multi-Domain Research
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于信息检索任务，比较Vector RAG与LLM编写的维基在小规模多领域研究中的表现，探讨其在答案组织、引用支持和成本上的差异。**

- **链接: [https://arxiv.org/pdf/2605.18490](https://arxiv.org/pdf/2605.18490)**

> **作者:** Theodore O. Cochran
>
> **摘要:** We preregistered a comparison of two ways to help an LLM answer questions over a small research corpus: a single-round Vector RAG system and an LLM-compiled markdown wiki. Both systems answered the same 13 questions over 24 papers using the same answer-generating model, and their answers were scored by blinded LLM judges. The wiki scored much better at connecting findings across papers, but its advantage in answer organization was not strong after judge adjustment. RAG met the preregistered test for single-fact lookup questions. The clean query-side cost result went against the expected wiki advantage: under the tested setup, the wiki used far more query tokens than RAG, so it could not recover any upfront build cost through cheaper queries. Two exploratory analyses changed how we interpret the result. First, claim-level citation checking favored the wiki: its cited pages more often supported the exact claims being made, even though RAG scored better on the overall groundedness rubric. Second, a decomposition-based RAG variant recovered most of the wiki's advantage on cross-paper synthesis at lower LLM-token cost, but it did not recover the wiki advantage in claim-by-claim citation support. The main conclusion is that grounded research synthesis is not a single capability. Systems can differ in how well they organize evidence, how well their citations support each claim, and how much they cost to run. In this study, no architecture was best on all three.
>
---
#### [new 099] DashAttention: Differentiable and Adaptive Sparse Hierarchical Attention
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出DashAttention，用于解决长序列注意力计算效率低的问题。通过自适应稀疏机制提升模型效率与性能，属于自然语言处理任务。**

- **链接: [https://arxiv.org/pdf/2605.18753](https://arxiv.org/pdf/2605.18753)**

> **作者:** Yuxiang Huang; Nuno M. T. Gonçalves; Federico Alvetreti; Lei Li; Xu Han; Edoardo M. Ponti; André F. T. Martins; Marcos V. Treviso
>
> **备注:** Preprint
>
> **摘要:** Current hierarchical attention methods, such as NSA and InfLLMv2, select the top-k relevant key-value (KV) blocks based on coarse attention scores and subsequently apply fine-grained softmax attention on the selected tokens. However, the top-k operation assumes the number of relevant tokens for any query is fixed and it precludes the gradient flow between the sparse and dense stages. In this work, we propose DashAttention (Differentiable and Adaptive Sparse Hierarchical Attention), which leverages the adaptively sparse $\alpha$-entmax transformation to select a variable number of blocks according to the current query in the first stage. This in turn provides a prior for the second-stage softmax attention, keeping the entire hierarchy fully differentiable. Contrary to other hierarchical attention methods, we show that DashAttention is non-dispersive, translating to better long-context modeling ability. Experiments with large language models (LLMs) show that DashAttention achieves comparable accuracy as full attention with 75% sparsity and a better Pareto frontier than NSA and InfLLMv2, especially in high-sparsity regimes. We also provide an efficient, GPU-aware implementation of DashAttention in Triton, which achieves a speedup of up to over FlashAttention-3 at inference time. Overall, DashAttention offers a cost-effective strategy to model long contexts.
>
---
#### [new 100] Closing the Gap at CRAC 2026: Two-Stage Adaptation for LLM-Based Multilingual Coreference Resolution
- **分类: cs.CL**

- **简介: 该论文属于多语言共指消解任务，旨在提升大语言模型的性能。通过两阶段微调和多语言适配器，提高系统在CRAC 2026中的表现。**

- **链接: [https://arxiv.org/pdf/2605.16984](https://arxiv.org/pdf/2605.16984)**

> **作者:** Antoine Bourgois; Olga Seminck; Thierry Poibeau
>
> **摘要:** We present our submission to the LLM track of the 2026 Computational Models of Reference, Anaphora and Coreference (CRAC 2026) shared task. With an average CoNLL F1 score of 74.32 on the official test set, our system ranked first in the LLM track, and third overall. Our system is based on the Gemma-3-27b model, fine-tuned using a two-stage strategy with a multilingual base adapter followed by dataset-specific adapters. We represent mention spans by their headword using an XML-inspired format with local reindexing and annotate documents iteratively. These design choices proved effective across languages, document lengths, and annotation guidelines.
>
---
#### [new 101] Machine Unlearning for Masked Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器去学习任务，旨在解决MDLMs中特定知识的删除问题。提出MDU框架，通过最小化KL散度实现有效去学习。**

- **链接: [https://arxiv.org/pdf/2605.18253](https://arxiv.org/pdf/2605.18253)**

> **作者:** Georu Lee; Seungwon Jeong; Hoki Kim; Jinseong Park; Woojin Lee
>
> **备注:** 20 pages, 8 figures, appendix included
>
> **摘要:** Recent masked diffusion language models (MDLMs), such as LLaDA and Dream, have achieved performance comparable to autoregressive large language models. Unlike autoregressive models, which generate text sequentially, MDLMs generate text by iteratively denoising masked positions in parallel. During fine-tuning, MDLMs learn to recover responses from masked response states conditioned on a prompt, thereby shifting their predictions from a prompt-masked unconditional distribution toward a prompt-conditional distribution. Despite this distinct generative and fine-tuning mechanism, machine unlearning for MDLMs remains largely unexplored. In this paper, we propose Masked Diffusion Unlearning (MDU), the first unlearning framework for MDLMs, by revisiting the process of learning specific knowledge in terms of diffusion. Specifically, MDU minimizes a forward KL divergence from the prompt-conditional prediction to a prompt-masked unconditional anchor at every masked response position, with a temperature scaling parameter to control the privacy-utility trade-off. Our empirical results on standard benchmarks and MDLM backbones show that MDU achieves high unlearning performance compared to existing LLM unlearning methods. Code is available at this https URL.
>
---
#### [new 102] Learning Faster with Better Tokens: Parameter-Efficient Vocabulary Adaptation for Specialized Text Summarization
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对专业文本摘要任务，解决通用语言模型在专业领域中的分词效率问题。通过参数高效词汇适应方法，提升摘要质量并减少训练时间和参数量。**

- **链接: [https://arxiv.org/pdf/2605.17379](https://arxiv.org/pdf/2605.17379)**

> **作者:** Gunjan Balde; Soumyadeep Roy; Mainack Mondal; Niloy Ganguly
>
> **备注:** 16 pages. Accepted in the 64th Annual Meeting of the Association for Computational Linguistics [ACL (Main) 2026] as a long paper
>
> **摘要:** Large language models pretrained on general-domain corpora often exhibit tokenization inefficiencies when applied to specialized domains. Although continual pretraining for domain adaptation partially alleviate performance degradation, it does not resolve the fundamental vocabulary mismatch. To address this gap, we introduce a targeted parameter-efficient domain adaptation approach that combines vocabulary adaptation with pretraining for LLM-based text summarization. Our unified framework augments pretrained tokenizers with domain-specific tokens while selectively replacing under-trained and unreachable tokens to limit parameter growth. We evaluate our approach on Llama-3.1-8B and Qwen2.5-7B across legal and medical summarization tasks on a challenge-oriented evaluation protocol focused on expert-driven text and summaries which typically has higher concentration of over-fragmented Out-of-Vocabulary (OOV) words. The vocabulary adaptation algorithm enhances the overall quality of the summarization model by improving semantic similarity between the generated summaries and their references. In addition, the adapted model produces summaries that incorporate more appropriate novel and domain-specific words, leading to improved coherence, relevance, and faithfulness. We further observe that our proposed approach significantly reduce training time by $35-55\%$ over continual pretraining and reduce parameter counts up to $37\%$ w.r.t expansion-only methods. We make the codebase publicly available at this https URL.
>
---
#### [new 103] Transitivity Meets Cyclicity: Explicit Preference Decomposition for Dynamic Large Language Model Alignment
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型对齐任务，解决人类偏好中循环性与传递性难以同时捕捉的问题。提出HRC模型和DSPPO方法，分离偏好成分并提升对齐效果。**

- **链接: [https://arxiv.org/pdf/2605.17342](https://arxiv.org/pdf/2605.17342)**

> **作者:** Yucong Huang; Xiucheng Li; Kaiqi Zhao; Jing Li
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Standard RLHF relies on transitive scalar rewards, failing to capture the cyclic nature of human preferences. While some approaches like the General Preference Model (GPM) address this, we identify a theoretical limitation: their implicit formulation entangles hierarchy with cyclicity, failing to guarantee dominant solutions. To address this, we propose the Hybrid Reward-Cyclic (HRC) model, which utilizes game-theoretic decomposition to explicitly disentangle preferences into orthogonal transitive (scalar) and cyclic (vector) components. Complementing this, we introduce Dynamic Self-Play Preference Optimization (DSPPO), which treats alignment as a time-varying game to progressively guide the policy toward the Nash equilibrium. Synthetic data experiments further validate HRC's structural superiority in mixed transitive--cyclic settings, where HRC converges faster and achieves higher accuracy than GPM. Experiments on RewardBench 2 demonstrate that HRC consistently improves over both BT and GPM baselines (e.g., +1.23% on Gemma-2B-it). In particular, its superior performance in the Ties domain empirically validates the model's robustness in handling complex, non-strict preferences. Extensive downstream evaluations on AlpacaEval 2.0, Arena-Hard-v0.1, and MT-Bench confirm the efficacy of our framework. Notably, when using Gemma-2B-it as the base preference model, HRC+DSPPO achieves a peak length-controlled win-rate of 44.75% on AlpacaEval 2.0 and 46.8% on Arena-Hard-v0.1, significantly outperforming SPPO baselines trained with BT or GPM. Our code is publicly available at this https URL.
>
---
#### [new 104] Scaling Accessible Mathematics on arXiv: HTML Conversion and MathML 4
- **分类: cs.CL; cs.DL**

- **简介: 该论文属于数学文献可访问性任务，旨在提升arXiv的HTML论文质量与可读性。通过优化转换流程、引入MathML 4标注及Rust重构，解决数学内容在网页上的准确显示与无障碍访问问题。**

- **链接: [https://arxiv.org/pdf/2605.16562](https://arxiv.org/pdf/2605.16562)**

> **作者:** Deyan Ginev; Brian Caruso; Bruce Miller; Jeff Sank; Jacob Weiskoff
>
> **备注:** 6 pages, ICMS 2026
>
> **摘要:** We report on the ongoing development of arXiv's HTML Papers offering, available on every new TeX/LaTeX submission since its initial release in 2023. The main highlights from 2025 and early 2026 are: (i) community-driven improvements to HTML fidelity and service health, with roughly half of 6,000 user reports resolved; (ii) corpus-scale conversion work aimed at 90% error-free HTML (currently 75%); (iii) initial MathML 4 Intent annotations for accessible speech output; (iv) an in-progress Rust port of LaTeXML, reducing compute costs and enabling faster previews on submission. The arXiv HTML Papers project remains experimental, but is gradually maturing as we better understand the needs of arXiv's readers and the technical opportunities presented by new standards and by advances in programming languages and AI.
>
---
#### [new 105] Temporal Decay of Co-Citation Predictability: A 20-Year Statute Retrieval Benchmark from 396M Ukrainian Court Citations
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于法律信息检索任务，研究共引用结构的预测能力随时间的变化。通过构建基准数据集，分析20年间的引用模式变化，揭示法律体系改革对检索效果的影响。**

- **链接: [https://arxiv.org/pdf/2605.17639](https://arxiv.org/pdf/2605.17639)**

> **作者:** Volodymyr Ovcharov
>
> **备注:** 12 pages, 8 figures, 4 tables. Dataset: this https URL
>
> **摘要:** Co-citation structure is widely assumed to provide stable retrieval signal in legal information systems. We test this assumption longitudinally by constructing UA-StatuteRetrieval, a benchmark that measures co-citation predictability across 20 annual snapshots (2007-2026) of 396 million codex citations from 101 million Ukrainian court decisions. Using a leave-one-out protocol over the full bipartite citation graph, we find that Adamic-Adar MRR declines 33% on a fixed set of articles (from 0.43 to 0.29) and 47% under a train/test temporal split (from 0.51 to 0.27) confirming genuine temporal decay rather than compositional shift or evaluation artifact. The decay is non-uniform: criminal procedure maintains stable co-citation patterns (MRR ~0.40), while civil law degrades from 0.35 to 0.15, coinciding with the 2017 judicial reform. Hub articles (>100K citations) resist decay, but mid-frequency articles (1K-10K) -- the practical retrieval frontier lose half their predictability. A BM25 text baseline decays even faster (31%), and embedding drift analysis with E5-large reveals a 4.3% semantic shift in how articles are cited, providing a mechanistic explanation for the observed decay. The benchmark is released at this https URL.
>
---
#### [new 106] OProver: A Unified Framework for Agentic Formal Theorem Proving
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出OProver，一个统一的代理形式化定理证明框架，解决传统定理证明中代理推理不足的问题。通过迭代训练和修复轨迹提升证明效果。**

- **链接: [https://arxiv.org/pdf/2605.17283](https://arxiv.org/pdf/2605.17283)**

> **作者:** David Ma; Kaijing Ma; Shawn Guo; Yunfeng Shi; Enduo Zhao; Jiajun Shi; Zhaoxiang Zhang; Gavin Cheung; Jiaheng Liu; Zili Wang
>
> **摘要:** Recent progress in formal theorem proving has benefited from large-scale proof generation and verifier-aware training, but agentic proving is rarely integrated into prover training, appearing only at inference time. We present OProver, a unified framework for agentic formal theorem proving in Lean 4, in which failed proof attempts are iteratively revised using retrieved compiler verified proofs and Lean compiler feedback. OProver is trained through continued pretraining followed by iterative post-training: each iteration runs agentic proving, indexes newly verified proofs into OProofs and the retrieval memory, uses repair trajectories as SFT data, and uses unresolved hard cases for RL. OProofs is built from public Lean resources, large-scale proof synthesis, and agentic proving traces, containing 1.77M Lean statements, 6.86M compiler-verified proofs, and serialized trajectories with retrieved context, failed attempts, feedback, and repairs. Across five benchmarks, OProver-32B attains the best Pass@32 on MiniF2F (93.3%), ProverBench (58.2%), and PutnamBench (11.3%), and ranks second on MathOlympiad (22.8%) and ProofNet (33.2%) more top placements than any prior open-weight whole-proof prover.
>
---
#### [new 107] Context Memorization for Efficient Long Context Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决长上下文生成中注意力计算效率低和上下文影响减弱的问题。提出一种无需训练的注意力状态记忆方法，提升生成效果并降低延迟。**

- **链接: [https://arxiv.org/pdf/2605.18226](https://arxiv.org/pdf/2605.18226)**

> **作者:** Yasuyuki Okoshi; Hao Mark Chen; Guanxi Lu; Hongxiang Fan; Masato Motomura; Daichi Fujiki
>
> **摘要:** Modern large language model (LLM) applications increasingly rely on long conditioning prefixes to control model behavior at inference time. While prefix-augmented inference is effective, it incurs two structural limitations: i) the prefix's influence fades as generation proceeds, and ii) attention computation over the prefix scales linearly with its length. Existing approaches either keep the prefix in attention while compressing it, or internalize it into model parameters through gradient-based training. The former still attends to the prefix at inference, while the latter is training-intensive and ill-suited to prefix updates. To address these issues, we propose attention-state memory, a training-free approach that externalizes the prefix into a lightweight, lookup-based memory of precomputed attention states between prefix and query tokens. On ManyICLBench with LLaMA-3.1-8B, our method improves accuracy over in-context learning at 1K-8K memory budgets while reducing attention latency by 1.36x at 8K, and surpasses full-attention RAG performance on NBA benchmark using only 20% of its memory footprint.
>
---
#### [new 108] Multilingual jailbreaking of LLMs using low-resource languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于安全漏洞研究任务，旨在解决LLMs在多语言环境下被越狱的问题。通过使用低资源非洲语言进行多轮对话测试，评估不同模型的安全机制有效性。**

- **链接: [https://arxiv.org/pdf/2605.18239](https://arxiv.org/pdf/2605.18239)**

> **作者:** Dylan Marx; Marcel Dunaiski
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Large Language Models (LLMs) remain vulnerable to jailbreak attempts that circumvent safety guardrails. We investigate whether multi-turn conversations using low-resource African languages (Afrikaans, Kiswahili, isiXhosa, and isiZulu) can bypass safety mechanisms across commercial LLMs. We translated prompts from existing datasets and evaluated ChatGPT, Claude, DeepSeek, Gemini, and Grok through automated testing and human red-teaming with native speakers. Single-turn translation attacks proved ineffective, while multi-turn conversations achieved English harmful response rates from 52.7% (Claude 3.5 Haiku) to 83.6% (GPT-4o-mini), Afrikaans from 60.0% (Claude 3.5 Haiku) to 78.2% (GPT-4o-mini), and Kiswahili from 41.8% (Claude 3.5 Haiku) to 70.9% (DeepSeek). Human red-teaming increased jailbreak rates compared to automated methods. Over all evaluated languages, the average jailbreak rate increased from 59.8% to 75.8%, with improvements of +20.0% (Afrikaans), +12.7% (isiZulu), +12.3% (isiXhosa), and +1% (Kiswahili), demonstrating that poor translation quality limits jailbreak success. These findings suggest that vulnerabilities in LLMs persist in multilingual contexts and that translation quality is the critical factor determining jailbreak success in low-resource languages.
>
---
#### [new 109] SomaliWeb v1: A Quality-Filtered Somali Web Corpus with a Matched Tokenizer and a Public Language-Identification Benchmark
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出 SomaliWeb v1，一个高质量的索马里语语料库及配套分词器和语言识别基准，解决索马里语数据不足与质量低的问题。**

- **链接: [https://arxiv.org/pdf/2605.18232](https://arxiv.org/pdf/2605.18232)**

> **作者:** Khalid Yusuf Dahir
>
> **备注:** 16 pages, 6 figures, 6 tables. Code: this https URL Dataset: this https URL
>
> **摘要:** Somali is a Cushitic language of the Horn of Africa with ~25 million speakers, yet no documented dedicated Somali pretraining corpus with a companion tokenizer and language-identification benchmark has been publicly released. Existing Somali text appears either inside multilingual distributions (HPLT v2, CC100, MADLAD-400, OSCAR, mC4) or in small, undocumented Somali-only uploads on Hugging Face. We introduce SomaliWeb v1, a quality-filtered Somali corpus of 819,322 documents (~303M tokens) built from three upstream sources (HPLT v2, CC100, Somali Wikipedia) through a six-stage reproducible pipeline. We release (i) the corpus, (ii) a matched BPE-16K tokenizer, and (iii) the first public side-by-side Somali benchmark of three production language identifiers. Our measurements reveal concrete quality defects in existing distributions: HPLT v2's "cleaned" Somali release retains 17.3% byte-exact duplicates, 56.1% of its documents contain fixable mojibake, and 10.7% of its byte-unique documents are near-duplicates at Jaccard tau=0.80. Our BPE-16K tokenizer emits 40.2% fewer tokens than GPT-4's cl100k_base on FLORES-200 Somali devtest as a tokenizer-level measurement; downstream language-model perplexity comparisons are deferred to a follow-up release.
>
---
#### [new 110] Bridging the Version Gap: Multi-version Training Improves ICD Code Prediction, Especially for Rare Codes
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗编码任务，旨在解决ICD版本差异和罕见代码预测问题。通过多版本训练提升模型性能，尤其在罕见代码上效果显著。**

- **链接: [https://arxiv.org/pdf/2605.17755](https://arxiv.org/pdf/2605.17755)**

> **作者:** Jinghui Liu; Anthony Nguyen
>
> **摘要:** Clinical coding maps clinical documentation to standardized medical codes, an essential yet time-consuming administrative task that could benefit from automation. Current models on ICD coding are typically optimized for codes from a specific ICD version. However, in reality, ICD systems evolve continuously, and different versions are adopted across time periods and regions. Moreover, ICD coding suffers from the long-tail problem, and rare code performance can be a bottleneck for developing implementable models. We examine whether it is viable to train version-independent models by combining data annotated in different ICD versions, which may help address these challenges. We add ICD-9 data to the training of a modified label-wise attention model for ICD-10 prediction, and find that despite the version mismatch, adding ICD-9 yields a 27% increase in micro F1 for 18K rare ICD codes compared to training on ICD-10 alone. On 8K frequent ICD-10 codes, the multi-version training also substantially improves macro metrics, with far fewer model parameters.
>
---
#### [new 111] Beyond Catalogue Counts: the Dataset Visibility Asymmetry in Low-Resource Multilingual NLP
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于多语言自然语言处理领域，旨在解决数据集可见性不对称问题。通过分析文献中的数据集使用情况，揭示了资源丰富语言在目录中显示数据匮乏的现象。**

- **链接: [https://arxiv.org/pdf/2605.17442](https://arxiv.org/pdf/2605.17442)**

> **作者:** Zhiyin Tan; Changxu Duan
>
> **备注:** Accepted at the 15th edition of the Language Resources and Evaluation Conference (LREC 2026)
>
> **摘要:** Multilingual NLP often relies on dataset counts from centralized catalogues to characterize which languages are resource-rich or resource-poor. However, these catalogues record only one layer of dataset visibility: what has been registered or institutionally distributed. They do not necessarily reflect which datasets are created, cited, or reused in the research literature. To examine this gap, we combine a catalogue-based baseline with literature-backed evidence of dataset circulation. We introduce the Resource Density Index (RDI), defined as the number of catalogued datasets per one million speakers, and compute it for the 200 most widely spoken languages in Ethnologue. Among them, 118 languages (59%) have an average RDI of zero across the LRE Map and the Linguistic Data Consortium (LDC), and another 23 fall below 0.1, corresponding to at most one catalogued dataset per ten million speakers. We then apply an LLM-assisted citation-mining pipeline over the Semantic Scholar corpus to these 141 low-visibility languages. After manual validation and consolidation, we identify 609 unique datasets across 53 languages, of which 356 remain openly accessible through working public links. These results reveal a substantial visibility gap: many large-speaker languages appear data-poor in catalogue records yet show clear evidence of dataset activity in the research literature. Our findings suggest that multilingual data scarcity should be understood not only as a production problem, but also as a question of documentation, discoverability, and long-term accessibility. Code and data are publicly available at (this https URL).
>
---
#### [new 112] PluRule: A Benchmark for Moderating Pluralistic Communities on Social Media
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文提出PluRule基准，用于检测社交媒体中多元社区的规则违规行为。任务是识别特定社区的违规规则，解决AI在多元语境下内容审核的挑战。工作包括构建多模态多语言数据集并评估模型表现。**

- **链接: [https://arxiv.org/pdf/2605.17187](https://arxiv.org/pdf/2605.17187)**

> **作者:** Zoher Kachwala; Bao Tran Truong; Rasika Muralidharan; Haewoon Kwak; Jisun An; Filippo Menczer
>
> **备注:** Accepted to ACL 2026 Main Conference
>
> **摘要:** Social media are shifting towards pluralism -- community-governed platforms where groups define their own norms. What violates rules in one community may be perfectly acceptable in another. Can AI models help moderate such pluralistic communities? We formalize the task as a multiple-choice problem, mirroring how human moderators operate in the real world: given a comment and its surrounding context, identify which specific rule, if any, is violated. We introduce PluRule, a multimodal, multilingual benchmark for detecting 13,371 rule violations across 1,989 Reddit communities spanning 2,885 rules in 9 languages. Using this benchmark, we show that state-of-the-art vision-language models struggle significantly: even GPT-5.2 with high reasoning performs only slightly better than a trivial baseline. We also find that bigger models and increased context provide marginal gains, and universal rules like civility and self-promotion are easier to detect. Our results show that moderation of pluralistic communities on social media is a fundamental challenge for language models. Our code and benchmark are publicly available.
>
---
#### [new 113] Predictive Prefetching for Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决RAG系统中因同步检索导致的延迟问题。提出异步检索框架，通过预测机制实现信息预取，降低延迟并保持答案质量。**

- **链接: [https://arxiv.org/pdf/2605.17989](https://arxiv.org/pdf/2605.17989)**

> **作者:** Wuyang Zhang; Shichao Pei
>
> **备注:** Accepted by Forty-third International Conference on Machine Learning ICML 2026
>
> **摘要:** Retrieval-Augmented Generation (RAG) improves factual grounding in large language models but suffers from substantial latency due to synchronous retrieval. While recent work explores asynchronous retrieval, existing approaches rely on heuristic coordination between retrieval and generation and assume stable information demands during decoding that often break in complex, multi-domain settings. In this paper, we propose an advanced asynchronous retrieval framework that enables predictive prefetching aligned with evolving information needs. The framework explicitly predicts when retrieval should be triggered and what information should be retrieved using three components, a retrieval predictor, a context monitor, and a query generator, by exploiting semantic precursors in generation dynamics that emerge several tokens before uncertainty becomes critical. Experiments on multiple benchmarks demonstrate up to 43.5% end-to-end latency reduction and 62.4% improvement in time-to-first-token, while maintaining answer quality comparable to synchronous RAG baselines.
>
---
#### [new 114] Validate Your Authority: Benchmarking LLMs on Multi-Label Precedent Treatment Classification
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律文本分类任务，旨在解决负性治疗分类的准确性和风险问题。通过构建新数据集和提出新评估指标，对比了多个大模型的性能。**

- **链接: [https://arxiv.org/pdf/2605.17691](https://arxiv.org/pdf/2605.17691)**

> **作者:** M. Mikail Demir; M. Abdullah Canbaz
>
> **备注:** Accepted for publication at the Natural Legal Language Processing Workshop (NLLP) 2025, co-located with EMNLP
>
> **摘要:** Automating the classification of negative treatment in legal precedent is a critical yet nuanced NLP task where misclassification carries significant risk. To address the shortcomings of standard accuracy, this paper introduces a more robust evaluation framework. We benchmark modern Large Language Models on a new, expert-annotated dataset of 239 real-world legal citations and propose a novel Average Severity Error metric to better measure the practical impact of classification errors. Our experiments reveal a performance split. Google's Gemini 2.5 Flash achieved the highest accuracy on a high-level classification task (79.1%), while OpenAI's GPT-5-mini was the top performer on the more complex fine-grained schema (67.7%). This work establishes a crucial baseline, provides a new context-rich dataset, and introduces an evaluation metric tailored to the demands of this complex legal reasoning task.
>
---
#### [new 115] HyperPersona: A Multi-Level Hypergraph Framework for Text-Based Automatic Personality Prediction
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于文本情感分析任务，旨在解决传统方法忽略文本多层级结构的问题。提出HyperPersona框架，通过超图建模文档、句子和词语的层次关系，提升人格预测效果。**

- **链接: [https://arxiv.org/pdf/2605.17355](https://arxiv.org/pdf/2605.17355)**

> **作者:** Sina Heydari; Majid Ramezani
>
> **备注:** Preprint. Submitted to Artificial Intelligence (Elsevier)
>
> **摘要:** As a modern commodity, language has become a vast repository of socially and psychologically significant traits and concepts, reflecting the ways people encode pattern of thoughts, behaviors, and emotions into words. Text-based Automatic Personality Prediction (APP), seeks to infer personality from linguistic behavior, offering a scalable alternative to traditional psychometric assessments. Although text is inherently hierarchical, with the document-level capturing global features, the sentence-level encoding local semantics, and the word-level providing fine-grained lexical information, most existing approaches rely on shallow, sequential, or single-level representations that ignore the multi-level structure of written language. To address this, we propose HyperPersona, a framework that explicitly models the hierarchical organization of text (document, sentence, and word) through hypergraph structure, where a document and its sentences are represented as hyperedges, and the words are represented as nodes, enabling joint modeling of global, local, and lexical dependencies of text. Followed by a transformer-based graph encoder that learns interactions within and across these linguistic layers, yielding context-sensitive and structurally grounded feature representations for personality prediction. Experiments on the Big Five personality dimensions show that, while relying solely on text, HyperPersona effectively integrates multi-level linguistic cues, achieving superior performance compared to state-of-the-art baselines. These findings underscore the critical role of textual hierarchy in advancing human-like personality inference from natural language.
>
---
#### [new 116] ANVIL: Analogies and Videos for Lecturers
- **分类: cs.CY; cs.AI; cs.CL; cs.GR; cs.HC; cs.MM**

- **简介: 该论文提出ANVIL，一个生成计算机科学教学动画的系统，解决如何自动化生成类比教学内容的问题。工作包括生成文本类比、制作视觉脚本、生成动画代码及质量评估。**

- **链接: [https://arxiv.org/pdf/2605.16295](https://arxiv.org/pdf/2605.16295)**

> **作者:** Yuri Noviello; Anastasiia Birillo; Gosia Migut
>
> **摘要:** We present ANVIL, a multimodal generative system that automates the production of analogy-based instructional animations for computer science topics. Given a concept definition, ANVIL generates a textual analogy, compiles it into a structured visual screenplay, and produces executable manim code to render an animation, with an automated repair mechanism to improve robustness. Evaluating such systems at scale requires balancing pedagogical validity with scalability. We begin with a teacher evaluation to ground the quality assessment and use its findings to guide automated screening. For textual analogies, we introduce an LLM-based evaluator for scalable quality screening; for videos, where subjective judgments are difficult to automate, we instead assess fidelity to the intended screenplay using an automated proxy for auditing and error analysis. We further conduct a user study with educators to examine adoption requirements and risks. Our findings suggest that ANVIL can produce materials that are frequently rated as adequate, and that educators respond positively to its perceived value and usability.
>
---
#### [new 117] Scalable Environments Drive Generalizable Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于强化学习领域，旨在解决智能体泛化能力不足的问题。通过环境扩展提升智能体适应新环境的能力，提出统一分类并探讨构建可扩展环境的方法。**

- **链接: [https://arxiv.org/pdf/2605.18181](https://arxiv.org/pdf/2605.18181)**

> **作者:** Jiayi Zhang; Fanqi Kong; Guibin Zhang; Maojia Song; Zhaoyang Yu; Jianhao Ruan; Jinyu Xiang; Bang Liu; Chenglin Wu; Yuyu Luo
>
> **摘要:** Generalizable agents should adapt to diverse tasks and unseen environments beyond their training distribution. This position paper argues that such generalization requires environment scaling: expanding the distribution of executable rule-sets that agents interact with, rather than only increasing trajectories or tasks within fixed benchmarks. Current scaling practices largely focus on collecting more experience or broader task sets under fixed interaction rules, leaving agents brittle when underlying interfaces, dynamics, observations, or feedback signals change. The core challenge is therefore a world-level distribution shift: agents need systematic exposure to environments with meaningfully different executable rule-sets. To clarify this challenge, we propose a unified taxonomy that separates trajectory scaling, task scaling, and environment scaling by their primary deliverables and by what changes in the executable rule-set. Building on this taxonomy, we synthesize construction paradigms for scalable environments, contrasting programmatic generators that prioritize controllability and verifiability with generative world models that offer broader coverage and open-endedness. We further outline how environment scaling can be coupled with stateful learning mechanisms, emphasizing learned update rules for cross-environment adaptation. We conclude by discussing alternative perspectives and argue that scalable environments provide the essential substrate for measurable and controllable progress toward robust general agents.
>
---
#### [new 118] Self-Improving CAD Generation Agents with Finite Element Analysis as Feedback
- **分类: cs.GR; cs.CL**

- **简介: 该论文属于CAD生成任务，旨在解决现有模型无法模拟工程师迭代和评估工程需求的问题。通过引入FEA反馈及新监督信号，提升生成CAD的物理合理性与结构准确性。**

- **链接: [https://arxiv.org/pdf/2605.17448](https://arxiv.org/pdf/2605.17448)**

> **作者:** Guijin Son; Jehyun Park; Seyeon Park; Sunghee Ahn; Youngjae Yu
>
> **备注:** Work in progress
>
> **摘要:** Computer-aided design (CAD) is the backbone of modern industrial design, yet learned CAD generators still fall short of real engineering pipelines: they neither iterate like engineers nor evaluate what engineering requires. Prior work has treated CAD generation as two disjoint steps, part synthesis and assembly, where the former is graded by proximity to a gold reference and the latter, when handled at all, is reduced to a separate constraint solving step. In this work, we introduce a more industry-native task formulation that requires a model to produce a fully assembled multi-part STEP file from a free-form engineering brief, which is then validated via finite element analysis (FEA). FEA validation reveals that Codex (GPT-5.5) and Claude Code (Opus-4.7) agents do not produce a single strict-passing artifact in the main first-attempt sweep, with the best configuration meeting only about 20% of typed requirements on average. Moreover, we introduce two additional supervision signals, a novel text-only blueprint schema and a 21-view image renderer that aids the agent's visual inspection, that better align the generation loop with how engineers iterate in practice. On S2O and Fusion360, the same feedback tools improve geometric reconstruction, with GPT-5.5/xhigh rising from 0.444 to 0.592 Box-IoU on S2O and from 0.397 to 0.505 on Fusion360. Together these signals move CAD programs toward artifacts that are not only visually plausible but also checked against physical and structural requirements.
>
---
#### [new 119] The Expressive Power of Low Precision Softmax Transformers with (Summarized) Chain-of-Thought
- **分类: cs.LG; cs.CC; cs.CL**

- **简介: 该论文研究Transformer模型的表达能力，解决其在低精度和标准架构下的可表达性问题。通过分析softmax注意力机制，证明其能模拟图灵机，提升推理任务的可学习性。**

- **链接: [https://arxiv.org/pdf/2605.18079](https://arxiv.org/pdf/2605.18079)**

> **作者:** Moritz Brösamle; Stephan Eckstein
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Existing expressivity results for transformers typically rely on hardmax attention, high precision, and other architectural modifications that disconnect them from the models used in practice. We bridge this gap by analyzing standard transformer decoders with softmax attention and rounding of activations and attention weights, while allowing depth and width to grow logarithmically with the context length. As an intermediate step, we construct hardmax transformers with ternary activations and well-separated attention scores that simulate Turing machines using Chain-of-Thought (CoT). This lets us convert the constructions to equivalent softmax transformers without the unrealistic parameter magnitudes or activation precision that prior approaches would require. Using the same technique, we analyze a recently proposed summarized CoT paradigm and show that it simulates Turing machines more efficiently, with model size scaling logarithmically in a space bound rather than a time bound. We empirically test predictions made by our results on a Sudoku reasoning task and find better alignment with learnability than for prior high-precision results. Our code is available at this https URL.
>
---
#### [new 120] SafeLens: Deliberate and Efficient Video Guardrails with Fast-and-Slow Screening
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出SafeLens，解决视频内容审核效率与成本问题。通过快慢推理架构和高效数据筛选，提升审核效果并降低计算开销。**

- **链接: [https://arxiv.org/pdf/2605.17610](https://arxiv.org/pdf/2605.17610)**

> **作者:** Shahriar Kabir Nahin; Hadi Askari; Muhao Chen; Anshuman Chhabra
>
> **摘要:** The rapid growth of online video platforms and AI-generated content has made reliable video guardrails a key challenge for safety and real-world deployment. While most videos can be screened through fast pattern recognition, a small subset requires deeper reasoning over temporally complex content and nuanced policy constraints. Existing approaches typically rely on large vision-language models applied uniformly across all inputs, resulting in high inference costs and inefficient allocation of computation. We propose SafeLens, a video guardrail framework that introduces a fast-and-slow inference architecture for efficient and accurate content moderation with variable computational cost across inputs. Additionally, we construct a high-quality dataset by applying influence-guided filtering to the SafeWatch Dataset, retaining only 2.4% of the original data. To further address limitations of training-time scaling, we enable test-time reasoning by augmenting the filtered data with structured Chain-of-Thought traces. Across real-world and AI-generated video benchmarks, SafeLens achieves state-of-the-art performance, outperforming strong open-source video guardrails (e.g., SafeWatch-8B, OmniGuard-7B) and closed-source models (e.g., GPT-5.4, Gemini-3.1-pro) while significantly reducing inference cost, demonstrating that efficient design serves to be more effective than scaling data or model size alone.
>
---
#### [new 121] The Alpha Illusion: Reported Alpha from LLM Trading Agents Should Not Be Treated as Deployment Evidence
- **分类: cs.CE; cs.AI; cs.CL**

- **简介: 该论文属于金融AI领域，旨在解决LLM交易代理报告的Alpha是否可作为部署证据的问题。研究指出当前证据无法区分真实预测与数据污染，并提出验证协议和替代架构。**

- **链接: [https://arxiv.org/pdf/2605.16895](https://arxiv.org/pdf/2605.16895)**

> **作者:** Yuxuan Ye; Jun Han; Ao Hu; Juncheng Bu; Yiyi Chen; Liangjian Wen; Danilo Mandic; Danny Dongning Sun; Xu Yinghui; Zenglin Xu
>
> **摘要:** End-to-end LLM trading agents have moved quickly from research curiosity to a small ecosystem of named systems, including FinCon, FinMem, TradingAgents, FinAgent, QuantAgent, and FLAG-Trader. Several of these report headline Sharpe ratios that would be material if read at face value on a deployment desk, and associated benchmarks such as FinBen report trading-task Sharpe statistics in the same range. The gap between architecture research and deployment claim has been crossed too freely on both sides of the academia--industry divide. We take a position on that gap: reported alpha from end-to-end LLM trading agents should not be treated as deployment evidence. Before such returns can support claims of deployable trading capability, they must survive structural validity tests for temporal integrity, real-world frictions, counterfactual robustness, predictive calibration, numerical execution, and multi-agent disaggregation. Current public evidence cannot yet distinguish robust predictive ability from temporal contamination, unmodeled frictions, short-window Sharpe uncertainty, narrative fitting, and parametric priors. The problem is not only evaluative but structural. Language confidence is not tradable probability, narrative reasoning is not numerical execution, and model priors may become undisclosed implicit factor exposures. We contribute a minimum reporting protocol suite, P1--P6, with tiered applicability by claim strength, and a conservative modular alternative that uses LLMs as auditable information interfaces upstream of independent calibration, risk, and execution modules. Code and reproduction harness: \url{this https URL}.
>
---
#### [new 122] FIM-LoRA: Task-Informative Rank Allocation for LoRA via Calibration-Time Gradient-Variance Estimation
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出FIM-LoRA，解决LoRA中各层适应能力不均的问题。通过梯度方差估计分配每层秩，提升模型效果。属于模型优化任务。**

- **链接: [https://arxiv.org/pdf/2605.16800](https://arxiv.org/pdf/2605.16800)**

> **作者:** Ramakrishnan Sathyavageeswaran
>
> **备注:** 10 pages, 1 figure
>
> **摘要:** Low-rank adaptation (LoRA) assigns a uniform rank to every adapted weight matrix - a practical convenience that ignores a fundamental reality: different layers contribute unequally to task adaptation. We address this with a lightweight engineering solution: before fine-tuning begins, run eight calibration backward passes, compute the gradient variance of each LoRA-B matrix as a proxy for layer informativeness, and redistribute the rank budget proportionally. The resulting adapter is a standard LoRA with a per-layer rank pattern - no new parameters, no training overhead, no changes to serving infrastructure. We implement this via an efficient approximation of the empirical Fisher Information Matrix (eFIM) diagonal, restricted to LoRA adapter matrices only, which reduces memory cost by approximately 256x compared to full-model Fisher estimation. On GLUE with DeBERTa-v3-base, FIM-LoRA matches LoRA (88.6 vs. 88.7) at the same parameter budget, and on commonsense reasoning with LLaMA-3-8B reaches 68.5 vs. 68.7 for LoRA. The per-layer rank maps are interpretable: value projections and early-to-middle layers consistently receive higher rank, consistent with established findings on transformer layer roles.
>
---
#### [new 123] FishBack: Pullback Fisher Geometry for Optimal Activation Steering in Transformers
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出FishBack框架，解决Transformer中激活空间几何非欧几里得问题，通过拉回Fisher度量优化激活引导，提升模型控制效果。**

- **链接: [https://arxiv.org/pdf/2605.17231](https://arxiv.org/pdf/2605.17231)**

> **作者:** Sihan Wang; Jiayi Zhao
>
> **备注:** Preprint. 20 pages, 9 figures, 5 tables
>
> **摘要:** Activation steering methods modify intermediate representations of language models to control output behavior, but universally assume the activation space is Euclidean. We show this assumption fails drastically: the local geometry induced by the model's own output behavior -- the Fisher information metric of the softmax layer, pulled back through the Jacobian of subsequent layers -- deviates from the Euclidean metric by over 97% in relative spectral norm on GPT-2, with an effective dimensionality of only 2--17% of the ambient space. From this pullback Fisher metric, we derive a closed-form steering equation that identifies the minimum-distortion direction for any target concept, yielding a closed-form optimal direction at each point that can be applied iteratively without manifold fitting or data-driven geometry estimation. We call the resulting framework FishBack. The metric admits a layer-wise recursive decomposition, which reveals that existing methods -- CAA, ActAdd, ITI, and others -- each implicitly adopt a particular approximate metric, and that their performance gaps are quantitatively predicted by a single spectral diagnostic: the ratio of their implicit metric's cost to the Fisher-optimal cost. On GPT-2, iterative pullback steering consistently outperforms all Euclidean baselines across three verb-morphology concepts and four layers, with off-target KL reductions of $1.3\times$--$2.5\times$ relative to Euclidean gradient ascent and $1.5\times$ relative to CAA at matched concept probability.
>
---
#### [new 124] Algorithmic Cultivation: How Social Media Feeds Shape User Language
- **分类: cs.SI; cs.AI; cs.CL; cs.CY; cs.HC**

- **简介: 该论文属于算法影响研究，探讨社交媒体信息流如何塑造用户语言。通过分析大量用户数据，发现不同信息流导致语言风格、语义和主题的变化，揭示了算法对在线表达的长期影响。**

- **链接: [https://arxiv.org/pdf/2605.17010](https://arxiv.org/pdf/2605.17010)**

> **作者:** Olivia Pal; Agam Goyal; Eshwar Chandrasekharan; Koustuv Saha
>
> **摘要:** Algorithmic feeds have become primary environments for encountering information online, yet while they shape what people see, less is known about how sustained feed exposure shapes how people write. Drawing on Cultivation Theory, we examine whether algorithmic feeds function as online environments that leave measurable traces in users' language. We leverage a large-scale longitudinal dataset of 235M posts by 4M users on Bluesky, and conduct a quasi-experimental study matching an initial pool of 368,513 users exposed to one of three feeds -- News, Science, and Blacksky -- with a pool of 2,001,915 active control users who did not engage with any of these feeds. We examine linguistic evolution across three dimensions: lexico-semantics, psycholinguistics, and topics. We find that users exposed to these feeds show significantly greater stylistic accommodation, semantic alignment, and register formalization than matched controls. These effects vary markedly by feed identity -- Blacksky produces the deepest psycholinguistic restructuring, with significant shifts in cognitive processing, affective expression, and pronoun use, while News and Science effects are largely confined to register and topical focus. Regression models reveal that reposting is the most consistent predictor of linguistic convergence across all feeds, whereas posting and bookmarking show feed-dependent effects, with effects differing more than fourfold across feeds. Our work extends Cultivation Theory beyond belief formation to linguistic behavior, demonstrating that feeds function as persistent linguistic environments that gradually shape what and how users write online. Our work has implications for studying algorithmic influence, online identity formation, and the design and governance of feed-based platforms that mediate online interactions.
>
---
#### [new 125] Generalization or Memorization? Brittleness Testing for Chess-Trained Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于棋类语言模型研究任务，旨在验证棋类训练模型是泛化还是记忆。通过实验揭示其性能主要源于模式匹配，并提出结合外部验证器的高效方法。**

- **链接: [https://arxiv.org/pdf/2605.17565](https://arxiv.org/pdf/2605.17565)**

> **作者:** Ethan Tang
>
> **备注:** 14 pages, 2 figures, 4 tables, 3 equations
>
> **摘要:** Recent work has fine-tuned language models on chess data and reported high benchmark scores as evidence that the resulting models can understand the rules of chess, play full chess games at a professional level, or generate human-readable explanations grounded in expert knowledge. We train KinGPT, a 25M-parameter character-level language model trained only on (position, best-move) pairs, who exceeds 3B-parameter ChessGPT on a 600-puzzle mate-in-N suite and 4B-parameter C1-4B over a 20-theme puzzle benchmark. We examine several claims made in existing literature regarding chess-trained language models and assert that their impressive benchmark performance is largely explained by pattern-matching. We also demonstrate how LLM-Modulo, a verifier-in-the-loop framework, raises RedPajama 3B's best move accuracy from 1.2% to 21.2% and move generation validity from 19.3% to 95.3% on mate-in-N chess puzzles, comparable to gains achieved from ChessGPT's fine-tuning on chess-specific web corpora at a fraction of the cost. Our results illustrate how pairing a general LLM with an external verifier offers a more flexible alternative to directly training on synthetic data for well-defined domains. We open source all training/evaluation code, datasets, puzzle samples, and KinGPT model checkpoints for reproducibility.
>
---
#### [new 126] Reducing Credit Assignment Variance via Counterfactual Reasoning Paths
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决多步推理中信用分配不均的问题。通过引入反事实比较框架，提升训练稳定性与模型性能。**

- **链接: [https://arxiv.org/pdf/2605.16302](https://arxiv.org/pdf/2605.16302)**

> **作者:** Fei Ding; Yongkang Zhang; Yeling Peng; Youwei Wang; Guoxiong Zhou; Zijian Zeng
>
> **摘要:** Reinforcement learning for multi-step reasoning with large language models (LLMs) often relies on sparse terminal rewards, leading to poor credit assignment conditions where the final feedback is evenly propagated across all intermediate decisions. This results in high gradient variance, unstable training, and numerous ineffective updates, ultimately causing the model to fail and preventing sustained improvement. We introduce a counterfactual comparison-based credit assignment framework, which samples multiple reasoning trajectories under the same input. By treating their differences as an implicit approximation of alternative decisions, we construct an implicit process-level advantage estimator that transforms sparse terminal rewards into step-sensitive learning signals. Based on this, we propose Implicit Behavior Policy Optimization (IBPO), which significantly improves training stability and performance upper bounds on mathematical and code reasoning benchmarks, pointing to a promising direction for unlocking the performance potential of LLMs.
>
---
#### [new 127] ESI-Bench: Towards Embodied Spatial Intelligence that Closes the Perception-Action Loop
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文提出ESI-Bench，用于评估具身空间智能，解决感知-行动闭环问题，通过主动探索提升任务表现。**

- **链接: [https://arxiv.org/pdf/2605.18746](https://arxiv.org/pdf/2605.18746)**

> **作者:** Yining Hong; Jiageng Liu; Han Yin; Manling Li; Leonidas Guibas; Li Fei-Fei; Jiajun Wu; Yejin Choi
>
> **备注:** this https URL
>
> **摘要:** Spatial intelligence unfolds through a perception-action loop: agents act to acquire observations, and reason about how observations vary as a function of action. Rather than passively processing what is seen, they actively uncover what is unseen - occluded structure, dynamics, containment, and functionality that cannot be resolved from passive sensing alone. We move beyond prior formulations of spatial intelligence that assume oracle observations by recasting the observer as an actor. We introduce ESI-BENCH, a comprehensive benchmark for embodied spatial intelligence spanning 10 task categories and 29 subcategories built on OmniGibson, grounded in Spelke's core knowledge systems. Agents must decide what abilities to deploy - perception, locomotion, and manipulation - and how to sequence them to actively accumulate task-relevant evidence. We conduct extensive experiments on state-of-the-art MLLMs and find that active exploration substantially outperforms passive counterparts, with agents spontaneously discovering emergent spatial strategies without explicit instructions, while random multi-view often adds noise rather than signal despite consuming far more images. Most failures stem not from weak perception but from action blindness: poor action choices lead to poor observations, which in turn drive cascading errors. While explicit 3D grounding stabilizes reasoning on depth-sensitive tasks, imperfect 3D representation proves more harmful than 2D baselines by distorting spatial relations. Human studies further reveal that unlike humans who seek falsifying viewpoints and revise beliefs under contradiction, models commit prematurely with high confidence regardless of evidence quality, exposing a metacognitive gap that neither better perception nor more embodied interaction alone can close.
>
---
#### [new 128] Mechanistically Interpretable Neural Encoding Reveals Fine-Grained Functional Selectivity in Human Visual Cortex
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; q-bio.NC**

- **简介: 该论文属于神经科学与人工智能交叉任务，旨在解决如何解释神经网络对大脑视觉皮层的预测机制。通过MINE框架，揭示单个脑区对特定图像特征的响应机制。**

- **链接: [https://arxiv.org/pdf/2605.16468](https://arxiv.org/pdf/2605.16468)**

> **作者:** Idan Daniel Grosbard; Mor Geva; Galit Yovel
>
> **备注:** 40 pages, 28 figures
>
> **摘要:** A central goal in understanding human vision is to uncover the visual features that drive neuronal activity. A growing body of work has used artificial neural networks as encoding models to predict cortical responses to natural images, revealing the visual content that activates category-selective regions. However, existing approaches are largely correlational and treat the encoder as a black box, leaving open which image features drive each voxel's response. We introduce Mechanistically Interpretable Neural Encoding (MINE), a framework that opens this black box by applying mechanistic-interpretability tools to localize the features within natural images that drive millimeter-scale (voxel-level) activity. MINE predicts each voxel's response using language-aligned image representations, and produces semantically interpretable descriptions of the features critical for the voxel's activation. We further generalize these per-image features into per-voxel functional profiles. To validate the per-image descriptions, we show they are sufficient to generate images that elicit voxel responses matching the responses to the original images, more accurately than images generated from random or low-attribution controls. Moreover, counterfactually inserting or removing the predicted features from images shifts activation in the expected direction, providing causal evidence. Counterfactual editing guided by the per-voxel activation profiles produces even stronger activation shifts, indicating that the profiles faithfully capture each voxel's selectivity. Finally, we apply MINE to well-studied category-selective brain regions, showing it recovers their known categorical preferences while revealing fine-grained unique voxel structure within each region. Overall, our results establish mechanistic interpretability as a path to discover and causally validate fine-grained hypotheses about neural function.
>
---
#### [new 129] Confidence Geometry Reveals Trace-Level Correctness in Large Language Model Reasoning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于语言模型推理任务，旨在解决如何通过模型自身信心轨迹判断推理正确性的问题。研究发现信心轨迹具有区分正确与错误推理的几何结构，并提出NeuralConf模型提升答案评估效果。**

- **链接: [https://arxiv.org/pdf/2605.16824](https://arxiv.org/pdf/2605.16824)**

> **作者:** Shuo Liu; Ding Liu; Shi-Ju Ran
>
> **备注:** 11 pages, 9 figures, 1 table. Code is available at this https URL
>
> **摘要:** Large language models (LLMs) generate not only reasoning text, but also token-level confidence trajectories that record how uncertainty evolves during inference. Whether these trajectories are relevant to reasoning correctness remains unclear. Here we show that confidence trajectories encode a content-agnostic confidence geometry associated with trace-level final-answer correctness. Using only token-level confidence values, without access to the input question, reasoning text, hidden states, or external verifiers, we find that low-dimensional representations of confidence trajectories separate correct from incorrect reasoning traces. Across GSM8K, MATH, and MMLU, this geometric separation is quantitatively linked to downstream predictability: stronger clustering of correct and incorrect traces, measured by the Davies--Bouldin index, consistently corresponds to higher correctness-discrimination AUC. We further show that correctness-related information is enriched in the tail of reasoning, suggesting that late-stage confidence dynamics carry key correctness signals. We propose NeuralConf, a lightweight estimator that learns from confidence trajectories for correctness evaluation. Under a fixed trace budget, NeuralConf-derived scores improve confidence-weighted answer aggregation over majority voting, tail confidence, and other static baselines. These results reveal that LLMs expose trace-intrinsic statistical signals of correctness through their own confidence dynamics, offering a route to improve inference using information already present within generation.
>
---
#### [new 130] GIM: Evaluating models via tasks that integrate multiple cognitive domains
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出GIM基准，用于评估模型在多认知领域整合任务中的表现，解决现有基准过于依赖知识或抽象推理的问题。工作包括设计820个问题，构建评分体系，并通过IRT模型分析模型能力。**

- **链接: [https://arxiv.org/pdf/2605.18663](https://arxiv.org/pdf/2605.18663)**

> **作者:** Rohit Patel; Alexandre Rezende; Steven McClain
>
> **备注:** 56 pages, 27 figures, 4 tables. Code: this https URL ; Dataset: this https URL
>
> **摘要:** As LLM benchmarks saturate, the evaluation community has pursued two strategies to increase difficulty: escalating knowledge demands (GPQA, HLE) or removing knowledge entirely in favor of abstract reasoning (ARC-AGI). The first conflates memorization with capability; the second divorces reasoning from the practical contexts in which it matters. We take a different approach. The Grounded Integration Measure (GIM) is a benchmark of 820 original problems (615 public, 205 private) where difficulty comes from integration; individual problems require coordinating multiple cognitive operations (constraint satisfaction, state tracking, epistemic vigilance, audience calibration) over broadly accessible knowledge, so that reasoning stays grounded in realistic tasks without being gated on specialized expertise. Each problem is an original expert-authored composition, majority with rubric-decomposed scoring (median 6 independently judged criteria). A balanced public--private split provides built-in contamination diagnostic. We calibrate a continuous response 2-parameter logistic (2PL) IRT model over >200k prompt-response pairs across 28 models, producing robust ability estimates that correctly order test-configurations even when raw accuracy is distorted by errors or missing data, addressing a common challenge in benchmark reporting. Using this framework, we present a comprehensive leaderboard spanning 22 models and 47 test-configurations (unique model, thinking-level pairs), and conduct what is to our knowledge the most extensive published study of how test-time compute trades off against model capability on a fixed benchmark: 11 models swept across 35 test-configurations. We observe that within-family configuration choices, such as thinking budget and quantization, matter as much as model selection. We release the evaluation framework, calibrated IRT parameters, and all public problems.
>
---
#### [new 131] General Preference Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出GPRL，解决大语言模型对齐中在线强化学习与偏好优化的分离问题，通过多维偏好建模提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.18721](https://arxiv.org/pdf/2605.18721)**

> **作者:** Muhammad Umer; Muhammad Ahmed Mohsin; Ahsan Bilal; Arslan Chaudhry; Andreas Haupt; Sanmi Koyejo; Emily Fox; John M. Cioffi
>
> **备注:** Submitted to NeurIPS 2026
>
> **摘要:** Post-training has split large language model (LLM) alignment into two largely disconnected tracks. Online reinforcement learning (RL) with verifiable rewards drives emergent reasoning on math and code but depends on a programmatic verifier that cannot reach open-ended tasks, while preference optimization handles open-ended generation yet forgoes the continuous exploration that powers online RL. Closing this gap requires a verifier for open-ended quality, but a scalar reward model is the wrong shape for the job. Quality is multi-dimensional, and any scalar score is an incomplete proxy that lets online RL collapse onto whichever axis the score is most sensitive to. We turn instead to the General Preference Model (GPM), which embeds responses into $k$ skew-symmetric subspaces and represents preference as a structured, intransitivity-aware comparison. Building on this, we propose General Preference Reinforcement Learning (GPRL), which carries the $k$-way structure through to the policy update. GPRL computes per-dimension group-relative advantages, normalizes each on its own scale so no axis can dominate, and aggregates them with context-dependent eigenvalues. The same structure powers a closed-loop drift monitor that detects single-axis exploitation and corrects it on the fly by reweighting dimensions and tightening the trust region. Starting from $\texttt{Llama-3-8B-Instruct}$, GPRL reaches a length-controlled win rate of $56.51\%$ on AlpacaEval~2.0 while also outperforming SimPO and SPPO on Arena-Hard, MT-Bench, and WildBench by resisting reward hacking across extended training runs.
>
---
#### [new 132] TRACE: Trajectory Correction from Cross-layer Evidence for Hallucination Reduction
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出TRACE算法，用于减少大模型中的幻觉问题。通过分析跨层证据，在推理时修正错误，无需额外训练或数据。**

- **链接: [https://arxiv.org/pdf/2605.18163](https://arxiv.org/pdf/2605.18163)**

> **作者:** Tej Sanibh Ranade
>
> **备注:** 25 pages, 8 figures, 4 tables
>
> **摘要:** Hallucination correction is not a one-direction problem. We show that intermediate layers are neither uniformly more truthful than final layers nor uniformly less trustworthy. Yet hallucination reduction is usually instantiated through one fixed intervention form: contrast one layer against another, steer along a truthfulness direction, or defer to external evidence. This framing is structurally incomplete. Cross-layer factual evidence does not evolve uniformly: in some failures truthful support is present internally and later suppressed, whereas in others candidate competition remains genuinely multi-directional across depth, so no single signed scalar family is generally sufficient. We introduce Trajectory Correction from Cross-layer Evidence for Hallucination Reduction (TRACE), a deterministic, training-free algorithm which corrects hallucinations at inference time by deriving both the corrective layer and the appropriate correction operator from each input's cross-layer candidate trajectory inside the LLM's own forward pass. Under one frozen hyperparameter setting, TRACE selects among scalar reversal, earlier-state recovery, and candidate-space correction using only model-internal evidence. Evaluated as a single universal algorithm across 15 models, 8 model families, and 3 factuality benchmarks, TRACE improves every evaluation cell, yielding mean gains of +12.26 MC1 points and +8.65 MC2-style points with no regressions, with gains reaching +47.20 MC1 and +43.38 MC2-style points. The method uses no labels, retrieval, pretraining, finetuning, or per-model calibration.
>
---
#### [new 133] LLMs in Qualitative Research: Opportunities, Limitations, and Practical Considerations
- **分类: cs.HC; cs.CL**

- **简介: 论文探讨LLMs在定性研究中的应用，分析其机遇、局限及实际考量，旨在指导研究人员合理整合LLMs，解决其透明度与定性方法论契合问题。**

- **链接: [https://arxiv.org/pdf/2605.16538](https://arxiv.org/pdf/2605.16538)**

> **作者:** Henry Salgado; Meagan R. Kendall; Martine Ceberio; Alexandra Coso Strong
>
> **备注:** To be published and presented in 2026 ASEE Annual Conference and Exposition
>
> **摘要:** This paper examines the opportunities, limitations, and practical considerations associated with the use of large language models (LLMs) in qualitative research. Drawing on a multidisciplinary perspective that combines expertise in qualitative methods and explainable AI, the paper argues that responsible integration of LLMs into qualitative workflows requires researchers to engage critically with a curated set of technical parameters, that is, context window constraints, temperature and top-p sampling settings, user and system prompt design, and model documentation in the form of system cards. The paper situates these considerations within the epistemological commitments of qualitative research, including reflexivity, positionality, and interpretive judgment, and discusses how the opacity of contemporary LLMs differs from earlier natural language processing tools such as topic models and lexicon-based sentiment analyzers.
>
---
#### [new 134] The IsalProgram Programming Language
- **分类: cs.PL; cs.AI; cs.CL**

- **简介: 该论文提出一种新型汇编语言IsalProgram，解决编程语言设计问题。其特点为正则语言、无地址变量，程序由固定指令序列构成，运行于虚拟机上。**

- **链接: [https://arxiv.org/pdf/2605.17008](https://arxiv.org/pdf/2605.17008)**

> **作者:** Ezequiel López-Rubio
>
> **摘要:** We introduce IsalProgram (Instruction Set and Language for Programming), a novel assembly-like programming language with three distinctive theoretical properties: (1) it is a regular language in the sense of formal language theory, meaning its programs are accepted by a finite automaton; (2) every finite string over the instruction alphabet is a syntactically valid program; and (3) it makes no explicit use of memory addresses or variable names, absolute or relative. Programs are finite sequences of tokens drawn from a fixed instruction set, and are executed on a virtual machine whose sole data structure is a circular doubly linked list (CDLL) navigated by three data pointers, with control flow governed by two code pointers. We give a complete formal definition of the language and its virtual machine, prove its regularity, and demonstrate its expressive power. We further discuss IsalProgram's potential advantages as a target language for neural program synthesis, the amenability of its program space to metric-based exploration via the Levenshtein edit distance, and directions for analyzing computability and complexity within this framework.
>
---
#### [new 135] DriveSafe: A Framework for Risk Detection and Safety Suggestions in Driving Scenarios
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出DriveSafe框架，解决自动驾驶中的风险检测与安全建议问题。通过生成空间语义描述，提升风险评估精度，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2605.16892](https://arxiv.org/pdf/2605.16892)**

> **作者:** Sainithin Artham; Shankar Gangisetty; Avijit Dasgupta; C. V. Jawahar
>
> **备注:** 8 pages
>
> **摘要:** Comprehensive situational awareness is essential for autonomous vehicles operating in safety-critical environments, as it enables the identification and mitigation of potential risks. Although recent Multimodal Large Language Models (MLLMs) have shown promise on general vision-language tasks, our findings indicate that zero-shot MLLMs still underperform compared to domain-specific methods in fine-grained, spatially grounded risk assessment. To address this gap, we propose DriveSafe, a framework for risk-aware scene understanding that leverages structured natural language descriptions. Specifically, our method first generates spatially grounded captions enriched with multimodal context, including motion, spatial, and depth cues. These captions are then used for downstream risk assessment, explicitly identifying hazardous objects, their locations, and the unsafe behaviors they imply, followed by actionable safety suggestions. To further improve performance, we employ caption-risk pairings to fine-tune a lightweight adapter module, efficiently injecting domain-specific knowledge into the base LLM. By conditioning risk assessment on explicit language-based scene representations, DriveSafe achieves significant gains over both zero-shot MLLMs and prior domain-specific baselines. Exhaustive experiments on the DRAMA benchmark demonstrate state-of-the-art performance, while ablation studies validate the effectiveness of our key design choices. Project page: this https URL research/projects/cvit-projects/drivesafe
>
---
#### [new 136] OpenJarvis: Personal AI, On Personal Devices
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出OpenJarvis，解决本地与云端模型性能差距问题，通过分解AI系统结构并优化各组件，提升本地模型表现。**

- **链接: [https://arxiv.org/pdf/2605.17172](https://arxiv.org/pdf/2605.17172)**

> **作者:** Jon Saad-Falcon; Avanika Narayan; Robby Manihani; Tanvir Bhathal; Herumb Shandilya; Hakki Orhun Akengin; Gabriel Bo; Andrew Park; Matthew Hart; Caia Costello; Chuan Li; Christopher Ré; Azalia Mirhoseini
>
> **备注:** Code: this https URL Website: this https URL
>
> **摘要:** Personal AI stacks, like OpenClaw and Hermes Agent, are becoming central to daily work, yet they route nearly every query (often over sensitive local data) to cloud-hosted frontier models. Replacing frontier models with local models inside existing stacks does not work: swapping Claude Opus 4.6 for Qwen3.5-9B drops accuracy by 25-39 pp across personal AI tasks like PinchBench and GAIA. Existing stacks bundle agentic prompts, tool descriptions, memory configuration, and runtime settings around a specific cloud model. Only the prompts can be tuned, and state-of-the-art prompt optimizers close just 5 pp of the local-cloud gap on their own. This motivates a decomposed personal AI stack: one that exposes individual primitives which can be optimized individually or jointly to close the local-cloud gap. We present OpenJarvis, an architecture that represents a personal AI system as a typed spec over five primitives: Intelligence, Engine, Agents, Tools & Memory, and Learning. Each primitive is an independently editable field, making the stack end-to-end optimizable and measurable against accuracy, cost, and latency. Towards closing the local-cloud gap without surrendering local-model properties, OpenJarvis introduces LLM-guided spec search, a local-cloud collaboration in which frontier cloud models propose edits across the spec at search time, only non-regressing edits are accepted, and the resulting spec runs entirely on-device at inference time. With LLM-guided spec search, on-device specs match or exceed cloud accuracy on 4 of 8 benchmarks and land within 3.2 pp of the best cloud baseline on average. They also reduce marginal API cost by ~800x and end-to-end latency by 4x.
>
---
#### [new 137] HINT-SD: Targeted Hindsight Self-Distillation for Long-Horizon Agents
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决长时序智能体训练中的稀疏奖励问题。通过提出HINT-SD框架，精准选择需监督的动作片段，提升训练效率与效果。**

- **链接: [https://arxiv.org/pdf/2605.17873](https://arxiv.org/pdf/2605.17873)**

> **作者:** Woongyeng Yeo; Yumin Choi; Taekyung Ki; Sung Ju Hwang
>
> **摘要:** Training long-horizon LLM agents with reinforcement learning is challenging because sparse outcome rewards reveal whether a task succeeds, but not which intermediate actions caused the outcome or how they should be corrected. Recent methods alleviate this issue by generating rewards or textual hints from turn-level action-output signals, or by using feedback-conditioned self-distillation. However, generating feedback at every turn is inefficient when many intermediate turns are already successful or neutral, and applying feedback at a fixed or misaligned turn often fails to supervise the actions that contributed to the failure. To bridge this gap, we propose HINT-SD, a targeted self-distillation framework that uses full-trajectory hindsight to select failure-relevant actions and applies feedback-conditioned distillation only on targeted action spans. Experiments on BFCL v3 and AppWorld show that our method improves over the dense per-turn feedback baseline by up to 18.80 percent while achieving 2.26$\times$ lower time per training step, suggesting that selecting where to distill is a key factor for both effective and efficient long-horizon agent training.
>
---
#### [new 138] How Off-Policy Can GRPO Be? Mu-GRPO for Efficient LLM Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究大语言模型的强化学习任务，解决GRPO算法在低偏离策略下的高系统开销问题，提出Mu-GRPO框架，提升训练效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.17570](https://arxiv.org/pdf/2605.17570)**

> **作者:** Minghao Tian; Yunfei Xie; Chen Wei
>
> **摘要:** Group Relative Policy Optimization (GRPO) has been a key driver of recent progress in reinforcement learning with verifiable rewards (RLVR) for large language models, but it is typically trained in a low-staleness, near-on-policy regime that incurs substantial system overhead. We ask a simple question: How off-policy can GRPO be? We show that GRPO-style algorithms can tolerate substantially larger rollout staleness than previously assumed, and propose Mu-GRPO, an RL training framework that organizes training into a small number (e.g., four) of large sequential generation-optimization stages. This design induces high rollout staleness while greatly reducing rollout-optimization switching overhead. To stabilize learning under stale data, Mu-GRPO combines relaxed clipping, which preserves useful stale-rollout gradients, with negative-advantage veto, which removes destabilizing post-trigger suffix updates in negative-advantage responses. Across five language models and multiple math reasoning benchmarks, Mu-GRPO matches or exceeds the performance of standard GRPO while achieving around 2x speedup in wall-clock training time, establishing a substantially improved performance-efficiency trade-off for LLM reinforcement learning.
>
---
#### [new 139] Overeager Coding Agents: Measuring Out-of-Scope Actions on Benign Tasks
- **分类: cs.SE; cs.AI; cs.CL; cs.CR**

- **简介: 该论文研究代码代理在良性任务中的越界行为，属于安全与授权领域。旨在检测并量化代理的过度操作，提出基准测试OverEager-Gen进行评估。**

- **链接: [https://arxiv.org/pdf/2605.18583](https://arxiv.org/pdf/2605.18583)**

> **作者:** Yubin Qu; Ying Zhang; Yanjun Zhang; Gelei Deng; Yuekang Li; Leo Yu Zhang; Yi Liu
>
> **摘要:** Coding agents now run autonomously with shell, file, and network privileges. When a user issues a benign request, the agent sometimes does more than asked: it deletes unrelated files, wipes a stale credentials backup, or rewrites configuration the user never mentioned. We call these scope expansions overeager actions, an authorization problem distinct from capability failures, prompt injection, or sandbox escapes. We present OverEager-Gen, a benchmark dedicated to overeager behavior on benign tasks. Building it surfaces a measurement-validity issue: if a benchmark spells out the authorized scope inside the prompt, the agent stops inferring boundaries and starts pattern-matching declaration text. On Claude Code, stripping the consent declaration alone raises the overeager rate from 0.0% to 17.1% on paired scenarios (McNemar exact p = 2.4 x 10^-4). OverEager-Gen therefore certifies each scenario's discriminative power before admission via a behavioral-gradient validator, audits internal tool calls through a dual-channel stack (PATH-injected shim plus per-agent event streams), and ships byte-identical consent_kept and consent_stripped variants. OverEager-Bench contains 500 validated scenarios and ~7,500 runs across four agent products (Claude Code, OpenHands, Codex CLI, Gemini CLI) and six base models; a 50-sample re-annotation gives Cohen's kappa = 0.73 and rule-judge recall = 1.00. Stripping consent multiplies the overeager rate on every shared base model (Delta in [11.9, 17.2] pp). The framework axis dominates effect size: a permissive cluster (Claude Code, Codex CLI, Gemini CLI) runs at 5.4-27.7% while the ask-to-continue framework (OpenHands) sits at 0.2-4.5% (Fisher p <= 10^-5). Within-framework base-model variance reaches 15.9 pp, indicating that model-layer alignment does not fully propagate through permissive permission gating.
>
---
#### [new 140] RAGA: Reading-And-Graph-building-Agent for Autonomous Knowledge Graph Construction and Retrieval-Augmented Generation
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出RAGA，用于自主构建知识图谱并增强检索生成。解决传统方法在语义关系捕捉、实体消歧和可解释性方面的不足。通过融合检索与生成，提升知识图谱质量和可靠性。**

- **链接: [https://arxiv.org/pdf/2605.17072](https://arxiv.org/pdf/2605.17072)**

> **作者:** Chengrui Han; Zesheng Cheng
>
> **摘要:** Existing LLM-driven knowledge graph (KG) construction methods predominantly employ stateless batch processing pipelines, exhibiting structural deficiencies in cross-chunk semantic relation capture, entity disambiguation, and construction process interpretability. These limitations undermine KG quality, retrieval precision, and deployment trust in high-stakes domains. We propose RAGA (Reading And Graph-building Agent), an LLM-based autonomous KG construction and retrieval fusion framework. RAGA provides an atomic toolset supporting full KG lifecycle CRUD operations and embeds a Read-Search-Verify-Construct cognitive constraint into a ReAct tool loop. A KG-vector synchronization mechanism enables hybrid symbolic-vector retrieval, while evidence-anchored verification links every knowledge entry to its source text for auditable provenance. Preliminary experiments on a subset of the QASPER scientific QA dataset indicate that RAGA's fusion retrieval outperforms zero-shot baselines, with KG integration providing measurable gains in both answer and evidence quality. The framework design and experimental baseline serve as a reference for agent-driven autonomous KG construction.
>
---
#### [new 141] AMARIS: A Memory-Augmented Rubric Improvement System for Rubric-Based Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出AMARIS系统，解决基于评分体系的强化学习中评估知识无法长期积累的问题。通过引入持久评估记忆，提升奖励塑造效果。**

- **链接: [https://arxiv.org/pdf/2605.18592](https://arxiv.org/pdf/2605.18592)**

> **作者:** Peilin Wu; Xinlu Zhang; Kun Wan; Wentian Zhao; Gang Wu; Xinya Du; Zhiyu Chen
>
> **备注:** Preprint. Under review
>
> **摘要:** Rubric-based reward shaping is an effective method for fine-tuning LLMs via RL, where structured rubrics decompose standard outcome rewards into multiple dimensions to provide richer reward signals. Recent works make the rubrics adaptive based on local signals such as the rollouts from the current step or pairwise comparisons. However, these methods discard the diagnostics produced during evaluation after immediate use and prevent the long-term accumulation and strategic reuse of evaluation knowledge. This forces the system to re-derive evaluation principles from scratch, limits its ability to detect recurring suboptimal behaviors, and forfeits the curriculum-like progression that a persistent training history would naturally support. To address these limitations, we introduce AMARIS, which grounds rubric modifications in long-term training history. At each training step, AMARIS analyzes individual rollouts, aggregates findings into step-level summaries, retrieves relevant historical context from a persistent evaluation memory through both static (recent steps) and dynamic (semantically matched) retrieval, and updates rubrics based on these accumulated analyses. This procedure runs asynchronously alongside the normal RL loop with minimal overhead. Experiments across both closed and open-ended domains show that AMARIS consistently outperforms the baselines. Ablation studies show that static and dynamic memory retrieval contributes to the performance gain and their combination provides the strongest results with moderate retrieval budgets sufficient to provide most of the gain, and that the entire pipeline adds only ~5\% time overhead through asynchronous execution. These results show that persistent evaluation memory can transform rubric-based reward shaping from a stateless, per-step heuristic into an evidence-driven loop for RL training.
>
---
#### [new 142] AI Agents May Always Fall for Prompt Injections
- **分类: cs.CR; cs.CL; cs.CY**

- **简介: 该论文属于AI安全领域，研究如何防御提示注入攻击。通过Contextual Integrity理论分析攻击机制，指出现有防御的局限性，并提出新的评估框架。**

- **链接: [https://arxiv.org/pdf/2605.17634](https://arxiv.org/pdf/2605.17634)**

> **作者:** Sahar Abdelnabi; Eugene Bagdasarian
>
> **摘要:** Prompt injection is the most critical vulnerability in deployed AI agents. Despite recent progress, we show that the prevailing defense paradigm (data-instruction separation) both fails to detect attacks that operate through contextual manipulation and degrades contextually appropriate behavior. We then recast prompt injection via the lens of Contextual Integrity (CI), a privacy theory that judges information flow compliance with contextual norms. This explains types of attacks that current defenses attempt to patch and predict advanced ones future agents will face. We develop unique benign and attack scenarios that force an agent to violate the norms by (1) misrepresenting the flow, (2) manipulating norms, or (3) mixing multiple flows. This reframing suggests an impossibility result: an adversary can always construct a context under which a blocked flow appears legitimate, or a defender who tightens norms will block genuinely legitimate flows. Our findings suggest that current research addresses a shrinking fraction of future attack surfaces. Instead, through CI, we offer a principled framework for evaluating context-sensitive failures, and designing CI-aware alignment for the frontier autonomous agents.
>
---
#### [new 143] Augmenting Human Evaluation with LLM Judges: How Many Human Reviews Do You Need?
- **分类: cs.LG; cs.AI; cs.CL; cs.HC; stat.ML**

- **简介: 该论文属于评估任务，解决如何有效结合LLM与人类评价的问题。通过两阶段采样设计，提出使用双重稳健估计器，优化人类与LLM评价样本量分配。**

- **链接: [https://arxiv.org/pdf/2605.16354](https://arxiv.org/pdf/2605.16354)**

> **作者:** Jane Paik Kim
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Large language models (LLMs) are increasingly used as automated evaluators of AI systems, including in high-stakes applications. In this role, LLMs are used to generate judgments about the quality, appropriateness, or even safety of model outputs. This approach is motivated by practical constraints. Expert human ratings are costly and difficult to scale, whereas LLM ratings can be produced quickly at low cost. However, current approaches to deploying LLM evaluators are ad hoc, typically limited to reporting agreement metrics between human and LLM judges as a justification for substitution of human ratings, and lack a formal basis for study design. This paper (1) shifts the role of the LLM judge from substitutive to auxiliary, and (2) formulates the LLM-as-a-judge paradigm as one of augmenting human evaluation through a two-stage sampling design, where LLM evaluations are measured for all observations at the first stage and human ratings are partially observed for a subsample at the second stage. We propose to use a doubly robust estimator from the missing data literature, which takes advantage of the robustness property against the prediction model, since the missingness model is known by design. Using the asymptotic variance of this estimator, we propose how sample sizes of human and LLM ratings can be determined to achieve a targeted level of power. We also show that a study can be efficiently designed by allocating more human ratings for types of evaluations where the predictability of LLM ratings is not high. To the best of our knowledge, there is very little guidance on how much human oversight should be retained when validating benchmarks.
>
---
#### [new 144] To MRL or not to MRL: Text Embeddings are Robust to Truncation Without Matryoshka Embeddings, Except In Heavy Truncation Scenarios
- **分类: cs.LG; cs.CL**

- **简介: 论文研究文本嵌入对截断的鲁棒性，比较MRL与随机截断的效果。任务是评估不同截断方法在下游任务中的表现，发现非MRL模型在轻度截断下表现更优。**

- **链接: [https://arxiv.org/pdf/2605.16608](https://arxiv.org/pdf/2605.16608)**

> **作者:** Sotaro Takeshita; Yurina Takeshita; Simone Paolo Ponzetto; Daniel Ruffinelli
>
> **摘要:** Matryoshka Representation Learning (MRL) is a widely adopted approach for training text encoders so they provide useful text representations at various sizes, available by simply truncating the resulting vectors at sizes pre-determined at training time. Recent works have shown that randomly truncating text embeddings has minimal impact in downstream performance unless vectors are reduced in size by at least 70%, suggesting that embeddings are already robust to truncation without the use of MRL. However, no prior work has compared random truncation to MRL, so it is unclear how the two methods compare as effective embedding reduction methods. In this paper, we study this by applying the same truncation used by MRL to models trained with and without MRL. Our results across several models and downstream tasks show that, unless heavily truncating embeddings (i.e. reducing their size by at least 80%), truncated embeddings of non-MRL models are competitive with, and often outperform models trained with MRL. This suggests that truncation robustness may not necessarily come from MRL, and that the choice of spending the additional training cost of MRL depends on whether heavy truncation is desired.
>
---
#### [new 145] What is Holding Back Latent Visual Reasoning?
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉推理任务，研究为何潜在视觉推理效果不佳。工作包括分析 latent tokens 的作用，发现数据信息不足和推理偏差是主要问题，并提出改进方向。**

- **链接: [https://arxiv.org/pdf/2605.18445](https://arxiv.org/pdf/2605.18445)**

> **作者:** André G. Viveiros; Nuno Gonçalves; André F. T. Martins; Matthias Lindemann
>
> **摘要:** Humans can approach complex visual problems by mentally simulating intermediate visual steps, rather than reasoning through language alone. Inspired by this, several works on Vision-Language Models have recently explored chain-of-thought reasoning with continuous latent tokens as intermediate visual imagination steps. In this work, we investigate how recent models leverage such latent tokens. Surprisingly, we find that model accuracy is unaffected when latent tokens are replaced by uninformative ``dummy'' tokens. This indicates that latent tokens play a minimal causal role in the model's final prediction. To better understand this phenomenon, we analyze both the training signal provided by oracle latent representations and the quality of the latent tokens generated at inference time. Our experiments reveal two crucial issues holding back latent visual reasoning: First, in most existing datasets, oracle latent tokens provide limited additional information beyond the original image and do not substantially simplify the task, leading models to ignore them during training and effectively bypassing them at inference time. When fine-tuned on a diagnostic dataset, in which latent tokens provide sufficient support for the final prediction, we show that models can causally rely on them. Second, the latent tokens produced at inference time deviate from their corresponding oracle representations, collapsing to a narrow region and preventing benefits even when the model relies on them. Overall, our findings suggest that future progress in latent visual reasoning depends on two key pillars: high-quality datasets with informative intermediate steps and more precise latent token prediction.
>
---
#### [new 146] UCSF-PDGM-VQA: Visual Question Answering dataset for brain tumor MRI interpretation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉问答任务，旨在解决脑肿瘤MRI解读中人工负担重的问题。提出UCSF-PDGM-VQA数据集，并评估模型性能，发现现有模型存在模态崩溃问题。**

- **链接: [https://arxiv.org/pdf/2605.17140](https://arxiv.org/pdf/2605.17140)**

> **作者:** Shiv Ghosh; Junayd Lateef; Chih-Hua; Yannan Yu; Andreas M. Rauschecker; Madhumita Sushil
>
> **备注:** 10 pages, 2 figures, 6 tables
>
> **摘要:** Brain tumor diagnosis is largely dependent on Magnetic Resonance Imaging (MRI) evaluation, which requires radiologists to synthesize thousands of images across multiple 3D sequences and longitudinal studies. This process requires advanced neuro-radiology training, poses substantial cognitive load, and is highly time-consuming. Despite increasing demands in radiology, this expertise is difficult to scale, straining the current health systems. Vision-Language Models (VLMs) provide an opportunity to reduce this burden through a semi-automated, interactive interpretation of complex brain MRIs. However, they are currently underutilized in neuro-oncology due to a lack of specialized benchmarks for evaluating them. We introduce a clinically relevant visual question answering (VQA) benchmark -- the UCSF-PDGM-VQA dataset -- consisting of 2,387 QA pairs from 473 glioma-related MRI studies in the public UCSF-PDGM dataset. We further establish a performance baseline for six state-of-the-art vision-language models (VLMs) and one large language model on this dataset. We find that current models are incapable of effectively processing multi-sequence, 3-dimensional MRI scans, thus resulting in a suppression of visual features and over-reliance on language priors, causing modality collapse. These findings underscore a critical deficiency in current model reliability and safety within clinical settings, necessitating the development of robust, domain-specific VLMs.
>
---
#### [new 147] Reducing Hallucination in Vision-Language Models via Stage-wise Preference Optimization under Distribution Shift
- **分类: cs.CV; cs.AI; cs.CL; cs.DB; cs.LG**

- **简介: 该论文属于视觉语言模型任务，旨在减少模型生成中的幻觉问题。通过分阶段偏好优化框架，构建特定数据以提升生成内容的视觉一致性。**

- **链接: [https://arxiv.org/pdf/2605.16411](https://arxiv.org/pdf/2605.16411)**

> **作者:** Qinwu Xu
>
> **摘要:** Hallucination remains a fundamental challenge in vision-language models (VLMs), where autoregressive generation may produce linguistically plausible yet physically inconsistent or visually ungrounded responses due to likelihood maximization under joint probabilistic modeling. We propose a stage-wise preference optimization framework for hallucination reduction through targeted multimodal data construction. Rather than directly optimizing on generic instruction-following data, our approach progressively constructs hallucination-focused preference pairs near known failure boundaries. The framework emphasizes ambiguous spatial orientation, object relationships, OCR uncertainty, and adversarial false-premise training. Hallucinated negatives are generated through minimally perturbed yet visually inconsistent alternatives, enabling Direct Preference Optimization (DPO) to better separate grounded reasoning from plausible hallucination. Experiments on open-source benchmarks and real-world multimodal evaluation scenarios demonstrate improved grounding consistency, reduced hallucination, and more informative grounded responses. Cross-model qualitative evaluation further shows that the proposed multimodal LLM DPO framework produces more visually grounded responses than several frontier proprietary VLMs, such as in ambiguous spatial reasoning and adversarial false-premise settings. The results suggest that hallucination may arise not only from limited model capacity, but also from inherent tendencies of autoregressive probabilistic generation to favor linguistically plausible continuations under weak visual grounding. Future work may explore physical consistency modeling, uncertainty-aware multimodal reasoning, and architectural alternatives beyond standard autoregressive decoding.
>
---
#### [new 148] Generative AI Advertising as a Problem of Trustworthy Commercial Intervention
- **分类: cs.CY; cs.CL**

- **简介: 论文探讨生成式AI广告对用户信任的影响，属于AI伦理与广告监管任务。解决如何识别和管理AI生成内容中的隐蔽商业影响问题，提出影响层级分类框架。**

- **链接: [https://arxiv.org/pdf/2605.18673](https://arxiv.org/pdf/2605.18673)**

> **作者:** Jingyi Qiu; Qiaozhu Mei
>
> **摘要:** Major deployed generative AI advertising systems preserve a visible boundary between commercial content and AI-generated responses. Yet empirical research shows that ads woven directly into large language model (LLM) outputs often go undetected by users. We argue that generative AI fundamentally changes advertising: rather than placing products into discrete slots, it enables interventions on the generative process itself, which induce commercial influence through less observable channels. This reframes generative AI advertising as a problem of trustworthy intervention rather than content placement. We introduce a taxonomy organized by influence tier, corresponding to interventions on progressively more latent variables: product mentions, information framing, behavioral redirection, and long-term preference shaping; and show how these tiers instantiate across modalities and system architectures, including retrieval-augmented generation and agentic pipelines where upstream decisions can sharply constrain downstream outcomes. Both major deployed systems and designed mechanisms concentrate on the most observable and easiest-to-govern tier, while the forms of commercial influence most consequential for user autonomy remain poorly understood and lack frameworks for detection, measurement, or disclosure. The central challenge is whether commercial influence in generative systems can be made trustworthy, i.e., attributable, measurable, contestable, and aligned with user welfare.
>
---
#### [new 149] From Demographics to Survey Anchors: Evaluating LLM Agents for Modeling Retirement Attitudes
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于预测任务，旨在评估LLM代理在模拟退休态度中的表现。对比基于人口统计的代理与基于调查数据的代理，发现前者存在偏差且不够真实。**

- **链接: [https://arxiv.org/pdf/2605.16303](https://arxiv.org/pdf/2605.16303)**

> **作者:** Rubén Garzón; Pauline Baron; Vincent Grari; Jonne Kamphorst; Michael Bernstein; Marcin Detyniecki
>
> **备注:** 50 pages, 22 figures
>
> **摘要:** Large language models (LLM) agents may offer tools to predict human responses to surveys. A common technique for defining these agents uses only demographics, for example country, age, gender, employment status, income, education and marital status. We compare the predictive accuracy of demographic agents to that of survey agents defined with a larger set of in-domain survey responses. We test both approaches in predicting responses to the multidisciplinary, cross-national Survey of Health, Ageing and Retirement in Europe (SHARE), focusing on five variables from three policy-relevant constructs around personal finance. In these three constructs, we observe that, compared to survey agents trained on broader data, demographics-only agents (1) exhibited a central tendency bias, skewing answers toward population means, and (2) were unrealistically accurate, failing to reproduce the incorrect answers and "don't know" responses typical of human respondents. These performance differences are further substantiated through the replication of a hierarchical regression analysis from prior retirement planning research. Agents based solely on demographic information reproduce the outcome that financial risk tolerance, future time perspective, and knowledge of retirement planning each are predictive of retirement savings. However, only the survey-anchored agents succeed in reproducing the interaction among these three factors. These findings suggest caution in using only demographics to define LLM agents for predicting survey responses.
>
---
#### [new 150] How do Humans Process AI-generated Hallucination Contents: a Neuroimaging Study
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于认知神经科学任务，旨在揭示人类如何处理AI生成的幻觉内容。通过EEG研究，分析人类在验证图像描述时的脑电活动，探讨其识别或被误导的神经机制。**

- **链接: [https://arxiv.org/pdf/2605.16953](https://arxiv.org/pdf/2605.16953)**

> **作者:** Shuqi Zhu; Yi Zhong; Ziyi Ye; Bangde Du; Yujia Zhou; Qingyao Ai; Yiqun Liu
>
> **摘要:** While AI-generated hallucinations pose considerable risks, the underlying cognitive mechanisms by which humans can successfully recognize or be misled by these hallucinations remain unclear. To address this problem, this paper explores humans' neural dynamics to characterize how the brain processes hallucinated content. We record EEG signals from 27 participants while they are performing a verification task to judge the correctness of image descriptions generated by a multi-modal large language model (MLLM). Based on an averaged event-related potential (ERP) study, we reveal that multiple cognitive processes, e.g., semantic integration, inferential processing, memory retrieval, and cognitive load, exhibit distinct patterns when humans process hallucinated versus non-hallucinated content. Notably, neural responses to hallucinations that were misjudged versus correctly judged by human participants showed significant differences. This indicates that misjudged AI-generated hallucinations failed to trigger the standard neurocognitive fact verification pathway.
>
---
#### [new 151] Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态语言模型任务，旨在解决细粒度视觉理解问题。通过区域到全局的自蒸馏框架，提升模型对关键视觉证据的聚焦能力。**

- **链接: [https://arxiv.org/pdf/2605.18740](https://arxiv.org/pdf/2605.18740)**

> **作者:** Qianhao Yuan; Jie Lou; Xing Yu; Hongyu Lin; Le Sun; Xianpei Han; Yaojie Lu
>
> **备注:** Project page: this https URL
>
> **摘要:** Multimodal Large Language Models (MLLMs) still struggle with fine-grained visual understanding, where answers often depend on small but decisive evidence in the full image. We observe a regional-to-global perception gap: the same MLLM answers fine-grained questions more accurately when conditioned on evidence-centered crops than on the corresponding full images, suggesting that many failures stem from difficulty to focus on relevant evidence rather than insufficient local recognition ability. Motivated by this observation, we propose Vision-OPD (Vision On-Policy Distillation), a regional-to-global self-distillation framework that transfers the model's own privileged regional perception to its full-image policy. Vision-OPD instantiates two conditional policies from the same MLLM: a crop-conditioned teacher and a full-image-conditioned student. The student generates on-policy rollouts, and Vision-OPD minimizes token-level divergence between the teacher and student next-token distributions along these rollouts. This enables the model to internalize the benefit of visual zooming without external teacher models, ground-truth labels, reward verifiers, or inference-time tool use. Experiments on multiple fine-grained visual understanding benchmarks show that Vision-OPD models achieve competitive or superior performance against much larger open-source, closed-source, and "Thinking-with-Images" agentic models.
>
---
#### [new 152] Linguistic Uncertainty and Reply Engagement on X: A Cross-Domain Replication of the Uncertainty-Reply Asymmetry
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于自然语言处理任务，研究 linguistic uncertainty 与社交平台互动的关系。通过分析英文帖子，验证不确定语言是否引发更多回复，发现不确定内容显著增加回复量，揭示跨语言的互动机制。**

- **链接: [https://arxiv.org/pdf/2605.16289](https://arxiv.org/pdf/2605.16289)**

> **作者:** Mohamed Soufan
>
> **备注:** 13 pages, 2 figures, 2 tables
>
> **摘要:** Linguistic uncertainty is common in social media, but its relationship with engagement remains unclear across languages and topics. Using 2,258 English-language posts on Federal Reserve policy, inflation, and electoral politics collected over three days in April 2026, we test whether the Uncertainty-Reply Asymmetry observed in prior Arabic-language research replicates in a broader context. Posts are classified using a lexicon-based uncertainty framework, with approximately one-third identified as uncertain. Uncertain posts receive 82% more replies on average than certain posts, with smaller increases in reposts and likes, replicating the asymmetric engagement pattern observed in prior work. Regression results confirm a positive and statistically significant association between uncertainty and replies (\b{eta} = 0.126, p = 0.011), equivalent to ~13% higher expected reply engagement, while total engagement shows a positive but weaker association. These findings suggest that linguistic uncertainty systematically increases conversational engagement and may reflect a general interactional mechanism across languages and domains.
>
---
#### [new 153] DISA: Offline Importance Sampling for Distribution-Matching LLM-RL
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出DISA方法，解决分布匹配强化学习中的校准问题。通过离线重要性采样估计分区函数，提升策略多样性与稳定性，适用于数学和代码任务。**

- **链接: [https://arxiv.org/pdf/2605.17295](https://arxiv.org/pdf/2605.17295)**

> **作者:** Shaobo Wang; Yujie Chen; Yafeng Sun; Wenjie Qiu; Zhihui Xie; Sihang Li; Yucheng Li; Huiqiang Jiang; Xingzhang Ren; Xuming Hu; Dayiheng Liu; Linfeng Zhang
>
> **备注:** 21 pages, 7 figures, 7 tables. Abstract shortened to respect the arXiv limit of 1920 characters. Please see the PDF for the full abstract
>
> **摘要:** Modern reasoning agents are increasingly evaluated on their ability to generate multiple valid solution paths, plans, or tool-use traces for a given input. Standard reward-maximizing RL tends to collapse onto the most easily reinforced high-reward mode, whereas distribution-matching RL aims to allocate probability mass across the entire reward-shaped solution set. Achieving this objective requires computing a prompt-dependent partition function over the trajectory space. Because existing distribution-matching methods learn this partition function online alongside the policy, calibration errors in the partition function directly distort policy updates and remain impossible to diagnose independently. We introduce DISA, short for Decoupled Importance-Sampled Anchoring, which moves this calibration problem outside the RL loop. DISA draws proposal trajectories offline, estimates the partition function via importance sampling, and freezes the resulting partition-function estimate before policy optimization begins. This decoupling preserves the distribution-matching objective while strictly separating partition-function estimation from policy learning in data, gradients, loss, and diagnostics. Empirically, on two open-weight backbones across six math and three code benchmarks, DISA matches or exceeds the online-coupled distribution-matching baseline FlowRL, outperforms rewardmaximization baselines GRPO and GSPO on math averages, and exceeds LoRASFT distillation by up to 13.8 Mean@8 points on the same offline trajectories. An LLM-as-judge evaluation further shows that DISA retains substantially more strategy-level diversity than reward-maximization baselines, and sensitivity studies on the proposal strength and inverse temperature follow the bias-variance pattern predicted by the analysis.
>
---
#### [new 154] Agentic Chunking and Bayesian De-chunking of AI Generated Fuzzy Cognitive Maps: A Model of the Thucydides Trap
- **分类: cs.AI; cs.CL; cs.HC; cs.IR**

- **简介: 该论文属于知识图谱构建任务，解决从文本生成因果模糊认知图的问题。通过分块与混合技术生成FCM，并应用贝叶斯推理进行去分块，用于分析权力冲突模型。**

- **链接: [https://arxiv.org/pdf/2605.17903](https://arxiv.org/pdf/2605.17903)**

> **作者:** Akash Kumar Panda; Olaoluwa Adigun; Bart Kosko
>
> **备注:** 15 pages, 6 figures
>
> **摘要:** We automatically generate feedback causal fuzzy cognitive maps (FCMs) from text by teaching large-language-model agents to break the text into overlapping chunks of text. Convex mixing of these chunk FCMs gives a representative cyclic FCM knowledge graph. The text chunks can have different levels of overlap. The chunk FCMs still mix to form a new FCM causal knowledge graph. The mixing technique scales because it uses light computation with sparse causal chunk matrices. The mixing structure allows an operator-level type of Bayesian inference that produces "de-chunked" or posterior-like FCMs from the mixed FCM. These de-chunked FCMs are useful in their own right and allow further iterations of Bayesian updating. We demonstrate these mixing techniques on the essay text of Allison's "Thucydides Trap" model of conflict between a dominant power such as the United States and a rising power such as China. The FCM dynamical systems predict outcomes as they equilibrate to fixed-point or limit-cycle attractors. Seven out of 8 FCM knowledge graphs predicted a type of war when we stimulated them by turning on and keeping on the concept node that stands for the rising power's ambition and entitlement. Gemini 3.1 LLMs served as the chunking AI agents.
>
---
#### [new 155] WASIL: In-the-Wild Arabic Spoken Interactions with LLMs
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文提出WASIL数据集，解决阿拉伯语语音交互中ASR误差影响用户意图的问题。任务为语音对话理解，通过标注和评估分离ASR干扰与内在不可回答性。**

- **链接: [https://arxiv.org/pdf/2605.16364](https://arxiv.org/pdf/2605.16364)**

> **作者:** Zien Sheikh Ali; Hamdy Mubarak; Soon-Gyo Jung; Hunzalah Hassan Bhatti; Firoj Alam; Shammur Absar Chowdhury
>
> **备注:** Spoken Prompts, Multilingual LLMs, Speech-based Evaluation, Dialectal Speech, Low-resource Languages, Conversational AI, Speech-to-Text QA, Real-world Interaction, Spoken Language Understanding
>
> **摘要:** Large Language Models (LLMs) voice assistants are commonly built as cascaded Automatic Speech recognition (ASR) to LLM systems, where recognition errors can distort user intent. Dislikes may also arise from ambiguous, out-of-domain, or non-request turns, making it hard to isolate ASR effects. We release WASIL (it denotes connection or linking in Arabic): in-the-wild Arabic spoken interaction prompts with audio, ASR hypotheses, assistant responses, and explicit like/dislike feedback (8,529 turns; 14.2% dislikes), plus a 2,000-turn test set covering Modern Standard Arabic (MSA) and four major dialects with their labels. We provide low-cost gold transcripts via multi-ASR agreement-guided post-editing and annotate answerability (answerable, ambiguous/needs-clarification, unsupported, not-a-request/noise) to separate intrinsic unanswerability from ASR-induced degradation. Finally, we describe scalable reference-free evaluation of responses from ASR vs. gold transcripts using multi-judge LLM scoring.
>
---
#### [new 156] Alignment Drift in Long-Term Human-LLM Interaction: A Mechanism-Oriented Framework
- **分类: cs.HC; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于人机交互研究，解决长期互动中的对齐漂移问题。提出机制框架，分析漂移成因及过程，为长期交互提供理论基础。**

- **链接: [https://arxiv.org/pdf/2605.16516](https://arxiv.org/pdf/2605.16516)**

> **作者:** Xintong Yao
>
> **备注:** 16 pages, 1 appendix
>
> **摘要:** Long-term interaction with LLM-based systems may produce alignment drift: a gradual process in which system outputs become less constrained by the user's current message and more shaped by prior interaction history, while still appearing helpful, coherent, and responsive. This process is difficult to detect because the user's subjective experience may improve as the system becomes more familiar, useful, and attuned. Existing research on human-LLM interaction has largely focused on short-term task performance, isolated outputs, or single-instance alignment problems, leaving slow and cumulative interaction-level dynamics undercharacterized. This paper proposes a mechanism-oriented framework for describing alignment drift. The framework defines the distinction between signal A and signal B, explains how drift develops through feedback loops and sub-pattern selection, divides the process into three interactional regimes, and identifies boundary conditions for controlling drift. By framing alignment drift as a recursive interactional process rather than an isolated model-side failure, the paper provides a conceptual basis for studying long-term human-system interaction.
>
---
#### [new 157] CodeBind: Decoupled Representation Learning for Multimodal Alignment with Unified Compositional Codebook
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出CodeBind，解决多模态对齐问题。通过共享-特定代码本设计，提升多模态表示一致性，无需完全配对数据。**

- **链接: [https://arxiv.org/pdf/2605.18257](https://arxiv.org/pdf/2605.18257)**

> **作者:** Zeyu Chen; Jie Li; Kai Han
>
> **备注:** ACL 2026 Findings; Project page: this https URL
>
> **摘要:** Multimodal representation alignment is pivotal for large language models and robotics. Traditional methods are often hindered by cross-modal information discrepancies and data scarcity, leading to suboptimal alignment spaces that overlook modality-unique features. We propose CodeBind, a framework that optimizes multimodal representation spaces through a modality-shared-specific codebook design. By incrementally aligning target and bridging modalities, CodeBind bypasses the need for fully paired data. Unlike traditional hard alignment, CodeBind decomposes features into shared components for semantic consistency and specific components for modality-unique details. This design utilizes a compositional vector quantization scheme, where a shared codebook bridges modality gaps and modality-specific codebooks mitigate representation bias by preventing dominant modalities from overshadowing others. Validated across nine modalities (text, image, video, audio, depth, thermal, tactile, 3D point cloud, EEG), CodeBind achieves state-of-the-art performance in multimodal classification and retrieval tasks.
>
---
#### [new 158] ContraFix: Agentic Vulnerability Repair via Differential Runtime Evidence and Skill Reuse
- **分类: cs.SE; cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于自动化漏洞修复任务，解决LLM代理在真实漏洞中因语义误解导致修复失败的问题。提出ContraFix框架，通过运行时证据和技能复用提升修复效果。**

- **链接: [https://arxiv.org/pdf/2605.17450](https://arxiv.org/pdf/2605.17450)**

> **作者:** Simiao Liu; Fang Liu; Li Zhang; Yang Liu; Yinghao Zhu
>
> **摘要:** Large language model (LLM) agents are increasingly used for automated vulnerability repair (AVR), where repository-level reasoning enables them to inspect context and produce source-code patches. However, recent empirical results show that these agents still struggle with real-world vulnerabilities. Their main failure mode is semantic misunderstanding: choosing a repair direction that does not match the root cause. We identify two reasons for this gap. Existing agents usually reason from the failing execution alone. A crash report can pinpoint where the program failed, but it does not reveal which variable or state transition, among many candidates near the fault site, separates the crashing behavior from safe execution. As a result, agents often produce symptom-oriented patches instead of causal fixes. Moreover, evidence collected for one vulnerability is rarely retained, so similar cases in later repositories must be diagnosed again from scratch. We present ContraFix, an agentic AVR framework that couples differential runtime evidence with reusable repair skills. Its Mutator constructs PoC variants that straddle the failure boundary; its Analyzer inserts state probes around the fault region and summarizes divergences between crashing and non-crashing executions into a repair specification; and its Patcher converts the specification into verified source patches. Each successful repair updates a two-track skill base containing repair specifications and mutation strategies, which are retrieved through a three-tier policy for future instances. On SEC-Bench (C/C++, 200 instances) and PatchEval (Go, Python, JavaScript, 225 instances), ContraFix with GPT-5-mini resolves 84.0% and 73.8% of the tasks, respectively, achieving state-of-the-art performance on both benchmarks while costing less than one-third of the strongest comparable baseline.
>
---
#### [new 159] CasualSynth: Generating Structurally Sound Synthetic Data
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出CausalSynth框架，解决合成数据中因果结构与语义不一致的问题，通过分阶段生成确保数据因果有效且语言丰富。**

- **链接: [https://arxiv.org/pdf/2605.17528](https://arxiv.org/pdf/2605.17528)**

> **作者:** Zehua Cheng; Wei Dai; Jiahao Sun; Thomas Lukasiewicz
>
> **备注:** 15 pages
>
> **摘要:** Large Language Models (LLMs) generate realistic synthetic data but offer no guarantee that their outputs respect the causal mechanisms governing the target domain. We introduce CausalSynth, a framework that decouples causal structure generation from semantic realization, yielding synthetic data that is both causally valid and linguistically rich. The framework operates in three phases. First, a Structural Causal Model (SCM) - a tuple of structural equations defined over a directed acyclic graph (DAG) generates causal skeletons, i.e., variable assignments that satisfy the Global Markov Property of the governing DAG, via ancestral sampling. Second, an LLM acts as a constrained \emph{realizer}, a conditional translator that maps each skeleton to a high-dimensional observation such as a clinical note or a transaction log. Third, an Iterative Consistency Verification module detects structural violations through deterministic extraction and feeds targeted corrections back to the LLM, forming a closed-loop refinement process. We identify the Semantic Backdoor problem the systematic tendency of LLMs to override imposed causal facts with pre-training priors -- and prove that our iterative mechanism reduces the resulting selection bias relative to standard rejection sampling. On three causal benchmarks (ASIA, ALARM, and MIMIC-Struct), CausalSynth preserved conditional independencies with false-positive rates near the nominal $\alpha=0.05$ level and achieved realizability rates above 96% with 70B-parameter LLM backbones. The framework additionally supports principled interventional and counterfactual generation through noise retention and graph mutilation.
>
---
#### [new 160] The Unlearnability Phenomenon in RLVR for Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，研究RLVR中语言模型的不可学习现象。揭示部分难题样本即使有正确反馈仍无法学习，分析其表示问题及优化局限。**

- **链接: [https://arxiv.org/pdf/2605.16787](https://arxiv.org/pdf/2605.16787)**

> **作者:** Yulin Chen; He He; Chen Zhao
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Reinforcement Learning with Verifiable Reward (RLVR) has proven effective in improving Large Language Model's (LLM) reasoning ability. However, the learning dynamics of RLVR remain underexplored. In this paper, we reveal a counterintuitive phenomenon: among hard examples that the model initially struggles with, a substantial subset remains unlearnable even when correct rollouts are present. To understand the phenomenon, we first demonstrate that existing optimization and sampling techniques fail to resolve unlearnability. With cross-example gradient analysis, we show that unlearnable examples have fundamental representation issue, characterized by low gradient similarity with the rest of the examples and ungeneralizable reasoning patterns. We further show that representation flaws are difficult to mitigate in RL, as data augmentation does not improve gradient similarity. Our study provides the first systematic characterization of unlearnable data in RLVR training and reveals fundamental limitations in current RL approaches for reasoning tasks. Code and data are available at \url{this https URL}.
>
---
#### [new 161] Generative Artificial Intelligence for Literature Reviews
- **分类: cs.DL; cs.CL**

- **简介: 论文探讨了生成式人工智能在文献综述中的应用，分析其优势与风险，提出使用通用和专业工具的方法与策略，旨在提升文献综述效率与质量。**

- **链接: [https://arxiv.org/pdf/2605.16475](https://arxiv.org/pdf/2605.16475)**

> **作者:** Gerit Wagner; Julian Prester; Reza Mousavi; Roman Lukyanenko; Guy Pare
>
> **摘要:** Generative artificial intelligence (GenAI), based on large-language models (LLMs), such as ChatGPT, has taken organizations, academia, and the public by storm. In particular, impressive GenAI capabilities such as summarization of large text corpora, question-answering, data extraction, and translation, carry profound implications for the conduct of literature reviews. This impacts science, organizations and the general public, as all can benefit from GenAI-supported literature reviews. Building on the technical foundations of GenAI and grounded in established methodological discourse, this work outlines approaches for conducting literature reviews using both general-purpose (e.g., ChatGPT, Gemini, Claude) and specialized GenAI tools (e.g., Consensus, Elicit). We provide illustrative examples of prompts and suggest methodologically-sound literature review strategies. Throughout this perspective paper, we adopt a balanced approach considering both the opportunities and the risks of relying on GenAI in the conduct of literature reviews. We conclude by discussing philosophical questions related to the effects of GenAI on long-term scientific progress, and also present fruitful opportunities for research on improving the core of GenAI's technology-its architecture and training data-and suggest open issues in GenAI-based literature reviews methodology.
>
---
#### [new 162] EmoMind: Decoding Affective Captions from Human Brain fMRI
- **分类: cs.LG; cs.AI; cs.CL; q-bio.NC**

- **简介: 该论文提出EmoMind，属于脑机接口任务，旨在从fMRI信号中解码情感化描述。解决传统方法忽略情感或依赖离散标签的问题，通过连续情绪向量生成个性化情感文本。**

- **链接: [https://arxiv.org/pdf/2605.16739](https://arxiv.org/pdf/2605.16739)**

> **作者:** Bilal A. Mohammed; Lin Gu; Ruogo Fang
>
> **摘要:** Decoding visual experience from brain activity has advanced substantially, but cur- rent brain-to-text systems largely recover semantic content while discarding affect. Additionally, language models can generate emotional text when prompted with categorical labels, but such labels collapse rich inter-subject variability into coarse discrete bins. We present EmoMind, the first end-to-end pipeline for decoding affective captions directly from fMRI signals. EmoMind first retrieves a semanti- cally grounded neutral scene description from brain-decoded visual features, then rewrites it using a continuous 34-dimensional emotion vector decoded from the same fMRI recording. To control the balance between content preservation and affective expression, we train the rewriter with classifier-free guidance against an identity-preserving null branch, enabling smooth interpolation between semantic fidelity and affective expressivity. We evaluate affective caption generation with a three-axis validation framework spanning subject-specificity, structural geometry, and causal control. We further augment this framework with a synthetic-brain substitution test that probes robustness to the measurement apparatus, and we benchmark each axis against GPT-4 prompted with brain-decoded top-5 emotion labels as a strong discrete baseline. Across two independent emotion fMRI datasets, EmoMind significantly outperforms label-prompted GPT-4 on all three axes, with the largest gains on metrics that require person-specific affective structure rather than population-level emotion aggregation. These results establish continuous brain-decoded affect as a viable control signal for individualized affective cap- tion generation and open new directions for studying individual affective brain organisation.
>
---
#### [new 163] Where Pretraining writes and Alignment reads: the asymmetry of Transformer weight space
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究Transformer权重空间的不对称性，解决预训练与对齐更新差异问题，通过分析权重变化方向揭示其几何特征。**

- **链接: [https://arxiv.org/pdf/2605.16600](https://arxiv.org/pdf/2605.16600)**

> **作者:** Valeria Ruscio; Eli-Shaoul Khedouri; Keiran Thompson
>
> **摘要:** Cross-entropy pretraining and preference alignment update the same transformer weights, but leave geometrically distinct traces. We characterise this asymmetry with a relative-subspace-fraction probe that tracks how weight deltas align with residual-stream activation subspaces and with the prediction subspace defined by the unembedding. Alignment deltas concentrate in the read pathway ($W_Q$, $W_K$), along principal directions of attention-input activations, while remaining near-isotropic in the write pathway ($W_O$, $W_2$) relative to the prediction subspace. We explain this pattern through anisotropic gradient accumulation: updates to a matrix $W$ are sums of outer products $\delta_t a_t^\top$, and inherit directional structure from whichever side has concentrated covariance. For read-pathway matrices, this side is the input activation $a_t$, whose covariance is spiked in trained transformers and therefore produces objective-agnostic concentration. For write-pathway matrices, the relevant side is the upstream gradient $\delta_t$, whose anisotropy depends on the loss. Cross-entropy supplies the canonical sharp per-sample signal, inducing write-pathway prediction geometry during pretraining; alignment objectives typically add little further write-side concentration. We support this explanation with a within-checkpoint trajectory, a graded contrastive-objective control, and a closed-form rank-1 intervention with matched direction controls, providing causal evidence for the proposed weight-space geometry.
>
---
#### [new 164] Compress the Context, Keep the Commitments: A Formal Framework for Verifiable LLM Context Compression
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决长对话上下文压缩中的承诺保留问题。提出Context Codec框架，实现可验证的上下文压缩。**

- **链接: [https://arxiv.org/pdf/2605.17304](https://arxiv.org/pdf/2605.17304)**

> **作者:** Natalia Trukhina; Vadim Vashkelis
>
> **摘要:** LLM context is not just tokens; it is a set of commitments. Long-running conversations accumulate goals, constraints, decisions, preferences, tool results, retrieved evidence, artifacts, and safety boundaries that future responses must preserve. Existing context-management methods reduce length through truncation, retrieval, summarization, memory systems, or token-level prompt compression, but they rarely specify which semantic commitments must survive compression or how their preservation should be measured. We propose Context Codec, a commitment-level framework for compressing prompts and chat histories. Context Codec represents dialogue state as typed, source-grounded semantic atoms with canonical identity, equivalence, conflict, confidence, risk, and evidence spans. It separates five concerns - extraction, normalization, representation, rendering, and verification - and introduces metrics for Critical Atom Recall, Weighted Atom Recall, Commitment Density, and round-trip recoverability. It also defines a taxonomy of semantic compression errors, a concrete normalization procedure, conservative fallback rules for low-confidence and safety-critical atoms, and Context Compression Language (CCL), an ASCII-first compact rendering of canonical JSON atoms. In a small diagnostic study, CCL-Core occupies a useful middle ground between structured prose and JSON: more explicit and auditable than prose, usually more compact than JSON, and less risky than heavily minified notation. The result is not a claim that shorthand solves compression, but a framework for making context compression verifiable: compress the conversation, keep the commitments.
>
---
#### [new 165] Decoupling KL and Trajectories: A Unified Perspective for SFT, DAgger, Offline RL, and OPD in LLM Distillation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大模型蒸馏中的KL散度与轨迹关系，解决如何有效设计蒸馏目标的问题。通过解耦KL方向与前缀来源，提出四种优化目标，并验证其在数学推理任务中的效果。**

- **链接: [https://arxiv.org/pdf/2605.16826](https://arxiv.org/pdf/2605.16826)**

> **作者:** Anhao Zhao; Haoran Xin; Yingqi Fan; Junlong Tong; Wenjie Li; Xiaoyu Shen
>
> **备注:** Code available at this https URL
>
> **摘要:** Knowledge distillation is central to LLM post-training, yet its design space remains poorly understood, especially alongside reinforcement learning (RL). We show that the prevailing paradigms, off-policy distillation and on-policy distillation (OPD), implicitly couple two orthogonal choices: prefix source and token-level KL direction. This follows from decomposing sequence-level KL over autoregressive response distributions: forward KL pairs teacher prefixes with token-level forward KL, and reverse KL pairs student prefixes with token-level reverse KL. We argue this coupling is not intrinsic: decoupling the two axes yields four valid objectives. We establish gradient-level identities showing forward KL gives SFT-style cross-entropy matching with teacher soft targets, whereas reverse KL gives an RL-style policy-gradient objective with a dense teacher-student log-ratio reward, connecting them to off-policy SFT, DAgger-style on-policy SFT, offline-RL-style distillation, and OPD. We conduct an extensive controlled study on math reasoning, evaluating the four objectives both as standalone methods and as initializations for subsequent RL. The results reveal three tradeoffs: KL direction induces an accuracy-entropy tradeoff, prefix source a quality-compute tradeoff, and training length an accuracy-stability tradeoff. Motivated by these findings, we propose KL mixing and an entropy-gated length curriculum. KL mixing shows long-sequence distillation requires substantial forward-KL weight to prevent entropy collapse and length inflation without sacrificing accuracy. The entropy-gated length curriculum improves Avg@k and Pass@k by 3.6 and up to 5.8 points, and cuts average response length by roughly 3x versus fixed long-horizon training. Our results provide a framework and practical methods for designing reasoning distillation objectives that balance accuracy, diversity, compute, and RL behavior.
>
---
#### [new 166] QQJ: Quantifying Qualitative Judgment for Scalable and Human-Aligned Evaluation of Generative AI
- **分类: cs.AI; cs.CL; cs.GR**

- **简介: 该论文提出QQJ框架，解决生成AI评价中自动化与人类判断不一致的问题。通过结构化评分体系提升评价的可扩展性和一致性。**

- **链接: [https://arxiv.org/pdf/2605.17382](https://arxiv.org/pdf/2605.17382)**

> **作者:** Marjan Veysi; Pirooz Shamsinejadbabaki; Mohammad Zare; Mohammad Sabouri
>
> **摘要:** The rapid progress of generative artificial intelligence has exposed fundamental limitations in existing evaluation methodologies, particularly for open-ended, creative, and human-facing tasks. Traditional automatic metrics rely on surface-level statistical similarity and often fail to reflect human perceptions of quality, while purely human evaluation, although reliable, is costly, subjective, and difficult to scale. Recent approaches using large language models as evaluators offer improved scalability but frequently lack explicit grounding in human-defined evaluation principles, leading to bias and inconsistency. In this paper, we introduce Quantifying Qualitative Judgment (QQJ), a scalable and human-centric evaluation framework that explicitly bridges the gap between human judgment and automated assessment. QQJ separates the definition of quality from its execution by anchoring evaluation in expert-designed, multi-dimensional rubrics and calibrating large language model evaluators to align with expert reasoning using a small, high-quality annotation set. This design enables consistent, interpretable, and scalable evaluation across diverse generative tasks and modalities. Extensive experiments on text and image generation demonstrate that QQJ achieves substantially stronger alignment with human judgment than traditional automatic metrics and unconstrained LLM-based evaluators. Moreover, QQJ exhibits improved stability across repeated evaluations and superior diagnostic capability in identifying critical failure modes such as hallucination and intent mismatch. These results indicate that structured qualitative judgment can be operationalized at scale without sacrificing interpretability or human alignment, positioning QQJ as a practical foundation for reliable evaluation of modern generative AI systems.
>
---
#### [new 167] CyberCorrect: A Cybernetic Framework for Closed-Loop Self-Correction in Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出CyberCorrect框架，解决大语言模型自我纠错问题。通过闭环控制理论提升纠错系统性和稳定性，提高准确率并减少过纠正。**

- **链接: [https://arxiv.org/pdf/2605.17305](https://arxiv.org/pdf/2605.17305)**

> **作者:** Yuning Wu; Yingmin Liu; Yang Shu
>
> **备注:** 6 pages, 1 figure, submitted to IEEE SMC 2026
>
> **摘要:** Large language model (LLM) self-correction -- the ability to detect and fix errors in generated outputs -- remains largely ad hoc, relying on generic prompts such as "please reconsider your answer" without systematic error analysis or convergence guarantees. We propose CyberCorrect, a framework that formalizes LLM self-correction as a closed-loop control system grounded in cybernetic theory. The framework models the LLM generator as the plant and introduces a tri-modal Error Detector (combining self-consistency, verbalized confidence, and logic-chain verification) as the sensor. A type-directed Correction Controller generates targeted repair instructions based on diagnosed error categories, while a Convergence Judge determines iteration termination using stability criteria adapted from control theory. We further introduce three control-theoretic evaluation metrics -- convergence rate, overshoot rate, and oscillation rate -- that capture correction dynamics beyond final accuracy. Experiments on our constructed CyberCorrect-Bench (440 reasoning tasks with annotated error types and correction paths) show that CyberCorrect achieves 79.8% final accuracy, improving upon the best existing self-correction method by 6.2 percentage points, while reducing overshoot (erroneous over-correction) by 41% through its convergence control mechanism.
>
---
#### [new 168] Trust No Tool: Evaluating and Defending LLM Agents under Untrusted Tool Feedback
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于安全防护任务，研究在不可信工具反馈下的LLM代理安全问题。提出TRUST-Bench和VISTA-Guard，解决工具被攻击时的信任累积风险问题。**

- **链接: [https://arxiv.org/pdf/2605.17453](https://arxiv.org/pdf/2605.17453)**

> **作者:** Lecheng Yan; Ruizhe Li; Xicheng Han; Wenxi Li; Binwu Wang; Longyue Wang; Chenyang Lyu; Guanhua Chen
>
> **摘要:** Tool-using LLM agents increasingly rely on external tools to make consequential decisions, yet most existing agent-security benchmarks and defenses implicitly assume that tool feedback is trustworthy once a tool has been selected. We study a different failure mode, cognitive poisoning, in which a malicious tool behaves plausibly during exploration, accumulates trust through benign-looking feedback, and becomes harmful only when hidden state conditions align with the final executable action. To study this setting, we construct TRUST-Bench, a task-conditioned benchmark of 1,970 hidden-trigger tool-compromise episodes with matched safe controls, introduce an asymmetric penalty metric, GuardedJoint, to better reflect real deployment risk, and present VISTA-Guard, a backbone-agnostic framework for final-action risk scoring. The core idea is to abstract multi-step tool interaction into structured environment variables that encode trust-formation dynamics and then score the risk of the final executable action from this trajectory-conditioned representation. Experiments show that prompt-centric heuristics, scalarized features, and zero-shot judges fail in this regime, whereas trajectory-aware final-action scoring yields strong in-domain discrimination and remains effective under balanced out-of-distribution transfer. Under GuardedJoint, VISTA-Guard reaches $84.2$ in-domain and $56.9$ on balanced out-of-distribution evaluation, while methods that optimize only one side of the safety--utility tradeoff collapse to zero. These findings support a broader view of agent security in black-box tool ecosystems: the decisive defense target is not local prompt text or tool descriptors alone, but the way trust is formed across the interaction trajectory and committed through the final action.
>
---
#### [new 169] Post-Trained MoE Can Skip Half Experts via Self-Distillation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型优化任务，旨在解决静态MoE模型在推理时计算效率低的问题。通过引入ZEDA框架，将预训练的静态MoE转换为动态高效模型，减少专家计算量并提升推理速度。**

- **链接: [https://arxiv.org/pdf/2605.18643](https://arxiv.org/pdf/2605.18643)**

> **作者:** Xingtai Lv; Li Sheng; Kaiyan Zhang; Yichen You; Siyan Gao; Xueheng Luo; Yuxin Zuo; Yuchen Fan; Junlin Yang; Ganqu Cui; Bingning Wang; Fan Yang; Youbang Sun; Ning Ding; Bowen Zhou
>
> **摘要:** Mixture-of-Experts (MoE) scales language models efficiently through sparse expert activation, and its dynamic variant further reduces computation by adjusting the activated experts in an input-dependent manner. Existing dynamic MoE methods usually rely on pre-training from scratch or task-specific adaptation, leaving the practical conversion of fully trained MoE underexplored. Enabling such adaptation would directly alleviate the inference costs by allowing easy tokens to bypass unnecessary expert during serving. This paper introduces Zero-Expert Self-Distillation Adaptation (ZEDA), a low-cost framework that transforms post-trained static MoE models into efficient dynamic ones. To stabilize this architectural conversion, ZEDA injects parameter-free zero-output experts into each MoE layer and adapts the augmented model through two-stage self-distillation, utilizing the original MoE as a frozen teacher and applying a group-level balancing loss. On Qwen3-30B-A3B and GLM-4.7-Flash across 11 benchmarks spanning math, code, and instruction following, ZEDA eliminates over 50% of expert FLOPs at marginal accuracy loss. It outperforms the strongest dynamic MoE baseline by 6.1 and 4.0 points on the two models, and delivers ~1.20$\times$ end-to-end inference speedup.
>
---
#### [new 170] Responsible Agentic AI Requires Explicit Provenance
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于AI责任研究任务，旨在解决Agentic AI中责任归属不清的问题。通过引入显式溯源机制，明确责任主体与因果关系，确保AI系统可追踪、可干预。**

- **链接: [https://arxiv.org/pdf/2605.17169](https://arxiv.org/pdf/2605.17169)**

> **作者:** Jinwei Hu; Xinmiao Huang; Qisong He; Youcheng Sun; Yi Dong; Xiaowei Huang
>
> **备注:** Under Review
>
> **摘要:** Agentic AI is rapidly proliferating across diverse real-world domains such as software engineering, yet public trust has not kept pace. The central reason is that responsibility, despite being widely discussed, remains a subjective and unenforced concept, as no current agentic framework produces the quantifiable, traceable, and interventionable provenance needed to assign it when harm emerges from compositions no single party designed. We position that what is missing is not better benchmark-level evaluation but $\textbf{explicit provenance}$ across the full agentic lifecycle, which is the only viable basis for making responsibility computable and actionable. We advance this agenda along four axes: establishing $\textit{why}$ such provenance is a structural necessity by identifying responsibility gaps across sociotechnical dimensions, formalizing $\textit{what}$ it must encode through a causal attribution function and responsibility tensor, discussing $\textit{how}$ it can be made computable across four lifecycle layers with preliminary experiments showing that provenance is estimable and interveneable online before irreversible harm accumulates, and examining $\textit{who}$ bears responsibility through a concrete agentic incident. Explicit provenance is not a discretionary refinement but the necessary condition for responsible agentic AI, and no stakeholder across its ecosystem can afford to treat it as optional.
>
---
#### [new 171] Hilbert-Geo: Solving Solid Geometric Problems by Neural-Symbolic Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于几何推理任务，解决三维几何问题难以自动求解的问题。提出Hilbert-Geo框架及Parse2Reason方法，实现准确的几何推理与验证。**

- **链接: [https://arxiv.org/pdf/2605.16385](https://arxiv.org/pdf/2605.16385)**

> **作者:** Ruoran Xu; Haoyu Cheng; Bin Dong; Qiufeng Wang
>
> **备注:** CVPR2026
>
> **摘要:** Geometric problem solving, as a typical multimodal reasoning problem, has attracted much attention and made great progress recently, however most of works focus on plane geometry while usually fail in solid geometry due to 3D spatial diagrams and complex reasoning. To bridge this gap, we introduce Hilbert-Geo, the first unified formal language framework for solid geometry, including an extensive predicate library and a dedicated theorem bank. Based on this framework, we propose a Parse2Reason method containing two steps of first parsing then reasoning. In the parsing step, we utilize conditional description language (CDL), a formalized language composed of predicates specifically designed to construct geometric conditions, to represent both problem description (natural text) and solid diagrams (visual image). In the reasoning step, we leverage those formal CDL and the theorem bank to perform relational inference and algebraic computation, generating strictly correct, verifiable, and human-readable reasoning processes. Notably, our proposed Hilbert-Geo is also applicable to plane geometry. To advance geometric reasoning, we curate two expert-annotated dataset SolidFGeo2k and PlaneFGeo3k, which are furnished with geometric formal language annotations, solutions and answers. Extensive experiments show that our proposed method achieves the state-of-the-art (SOTA) performance 77.3% in SolidFGeo2k and 84.1% in MathVerse-Solid (one small subset in MathVerse dedicated to solid geometry), substantially outperforming leading MLLMs, such as Gemini-2.5-pro (54.2% on SolidFGeo2k) and GPT-5 (62.9% on MathVerse-Solid). In addition, our method achieves the SOTA accuracy 80.2% in PlaneFGeo3k, demonstrating the generality of the Hilbert-Geo in geometric reasoning. Our code and datasets will be publicly available.
>
---
#### [new 172] SIREM: Speech-Informed MRI Reconstruction with Learned Sampling
- **分类: cs.SD; cs.CL; cs.CV; cs.LG; physics.med-ph**

- **简介: 该论文属于实时MRI重建任务，旨在解决高分辨率与快速成像的矛盾。通过引入语音作为先验信息，提出SIREM框架，结合音频与MRI数据提升重建质量与速度。**

- **链接: [https://arxiv.org/pdf/2605.18221](https://arxiv.org/pdf/2605.18221)**

> **作者:** Md Hasan; Nyvenn Castro; Daiqi Liu; Lukas Mulzer; Jana Hutter; Jonghye Woo; Moritz Zaiss; Andreas Maier; Paula A. Perez-Toro
>
> **摘要:** Real-time magnetic resonance imaging (rtMRI) of speech production enables non-invasive visualization of dynamic vocal-tract motion and is valuable for speech science and clinical assessment. However, rtMRI is fundamentally constrained by trade-offs among spatial resolution, temporal resolution, and acquisition speed, often leading to undersampled k-space measurements and degraded reconstructions. We propose SIREM, a speech-informed MRI reconstruction framework that uses synchronized speech as a cross-modal prior. The central idea is that vocal-tract configurations during speech are correlated with the produced acoustics, making part of the image content predictable from audio. SIREM models each frame as a fusion of an audio-driven component and an MRI-driven component through a spatial weighting map. The audio branch predicts articulator-related structure from speech, while the MRI branch reconstructs complementary content from measured k-space data. We further introduce a learnable soft weighting profile over spiral arms, enabling a differentiable study of how k-space arm usage interacts with speech-informed fusion. This yields a unified multimodal formulation that combines audio-driven prediction, MRI reconstruction, and sampling adaptation. We evaluate SIREM on the USC speech rtMRI benchmark against standard baselines, including gridding, wavelet-based compressed sensing, and total variation. SIREM introduces a speech-informed reconstruction paradigm that operates in a substantially higher-throughput regime than iterative methods while preserving anatomically plausible vocal-tract structure. These results establish an initial benchmark for multimodal speech-informed rtMRI reconstruction and highlight the potential of synchronized speech as an auxiliary prior for fast reconstruction. The source code is available at this https URL
>
---
#### [new 173] AI Slop or AI-enhancement? Student perceptions of AI-generated media for an English for Academic Purposes course
- **分类: cs.CY; cs.AI; cs.CL; cs.MM**

- **简介: 该论文属于教育技术任务，探讨AI生成内容在EAP课程中的应用效果。研究解决AI工具是否提升教学或仅生成低质材料的问题，通过混合方法分析学生反馈与学习成果，验证AI辅助教学的有效性。**

- **链接: [https://arxiv.org/pdf/2605.16275](https://arxiv.org/pdf/2605.16275)**

> **作者:** David James Woo; Deliang Wang; Kai Guo
>
> **备注:** 23 pages, 7 figures
>
> **摘要:** Artificial intelligence (AI) retrieval-augmented generation (RAG) tools now enable educators to transform course materials into diverse multimedia at scale. However, it remains unclear whether such AI-generated content functions as a pedagogical scaffold or AI slop: high volume, low quality material. This innovative practice paper reports on the development, implementation, and evaluation of teacher-prompted, AI-generated supplemental materials in an English for Academic Purposes (EAP) course at a Hong Kong Community College. Using primarily Google Notebook LM, the instructor generated videos, podcasts, infographics, and individualized feedback reports from course materials and student work for 106 English as a Foreign Language learners. An explanatory sequential mixed-methods design comprising a survey, semi-structured interviews, and correlation analysis with academic scores was employed to examine students' preferences, perceptions, and learning outcomes. Findings are framed through the Technology Acceptance Model and Cognitive Load Theory. Students rated the materials highly for perceived usefulness and ease of use, and preferred assessment-linked content presented in visual and multimodal formats, particularly videos and infographics. Video preference correlated positively with academic performance; however, higher cognitive load was negatively associated with course grades, indicating that material complexity must be carefully calibrated. Notably, some lower-performing students independently adopted the materials as remedial scaffolds. The practice demonstrates that RAG tools enable scalable personalized feedback that would be less feasible through traditional methods. When aligned with student goals and cognitive principles, teacher-prompted AI generation can meaningfully enhance the EAP learning ecosystem rather than producing AI slop.
>
---
#### [new 174] 1GC-7RC: One Graphic Card -- Seven Research Challenges! How Good Are AI Agents at Doing Your Job?
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出1GC-7RC基准，用于评估AI编码代理在七个机器学习任务中的自主能力，解决缺乏标准化评测的问题。**

- **链接: [https://arxiv.org/pdf/2605.17046](https://arxiv.org/pdf/2605.17046)**

> **作者:** Robin-Nico Kampa; Fabian Deuser; Anna Bößendörfer; Konrad Habel; Norbert Oswald
>
> **摘要:** Autonomous AI coding agents are becoming a core tool for ML practitioners in industry and research alike. Despite this growing adoption, no standardized benchmark exists to evaluate their ability to design, implement, and train models from scratch across diverse domains. We introduce **1GC-7RC** (*Single Graphic Card: Seven Research Challenges*), a benchmark comprising seven ML tasks spanning language modeling, image classification, semantic segmentation, graph learning, tabular prediction, time-series forecasting, and text classification. Each task provides a locked data-preparation and evaluation script together with a baseline training script; the agent may only modify the training code, has no access to pretrained weights (with one controlled exception for semantic segmentation), no internet access, and must complete each task within a task-specific wall-clock budget (40-120 minutes) on a single GPU. We evaluate seven coding agents: five proprietary (Claude Code with Sonnet 4.6, Opus 4.6, and Opus 4.7; Codex CLI with GPT 5.5; and OpenCode with Qwen 3.6+) and two open-source (OpenCode with Kimi K2.5, Kimi K2.6). Across 5 runs per agent-task pair, we report substantial performance differences that reveal varying levels of implicit ML knowledge, planning ability, and time-budget management. The benchmark, harness, and all evaluation artifacts are publicly available on GitHub at this https URL to facilitate reproducible comparison of future agents. Because our benchmark design is modular, the benchmark can be extended to new tasks and domains, adapted to different GPU budgets, and used to study multi-agent settings, making it a flexible platform for future research on autonomous research agents.
>
---
#### [new 175] D$^2$Evo: Dual Difficulty-Aware Self-Evolution for Data-Efficient Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决数据稀缺和难度变化问题。提出D²Evo框架，通过双难度感知自进化提升推理能力。**

- **链接: [https://arxiv.org/pdf/2605.17037](https://arxiv.org/pdf/2605.17037)**

> **作者:** Ru Zhang; Renda Li; Ziyu Ma; Weijie Qiu; Chongyang Tao; Yong Wang; Xiangxiang Chu
>
> **备注:** Accepted by ICML 2026. First two authors contributed equally
>
> **摘要:** Reinforcement learning (RL) has demonstrated potential for enhancing reasoning in large language models (LLMs). However, effective RL training, which requires medium-difficulty training samples, faces two fundamental challenges: Effective Data Scarcity and Dynamic Difficulty Shifts, where medium-difficulty samples are scarce and become trivial as models improve. Existing methods mitigate this scarcity to some extent by generating training samples. However, these approaches suffer from anchor-free generation, ignoring co-evolution, and difficulty mismatch. To address these issues, we propose D$^2$Evo, a Dual Difficulty-aware self-Evolution RL framework. In each iteration, our method mines medium-difficulty anchors based on the current Solver's capability, trains the Questioner to generate diverse questions at appropriate difficulty levels, and jointly optimizes both components to enable progressive reasoning gains. Extensive experiments demonstrate that D$^2$Evo outperforms existing methods on mathematical reasoning benchmarks with fewer than 2K real mathematical samples, and exhibits strong generalization on general reasoning benchmarks.
>
---
#### [new 176] DACA-GRPO: Denoising-Aware Credit Assignment for Reinforcement Learning in Diffusion Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出DACA-GRPO方法，解决扩散语言模型中强化学习的信用分配与似然估计偏差问题，提升生成任务性能。**

- **链接: [https://arxiv.org/pdf/2605.16342](https://arxiv.org/pdf/2605.16342)**

> **作者:** Amin Karimi Monsefi; Dominic Culver; Nikhil Bhendawade; Lokesh Boominathan; Manuel R. Ciosici; Yizhe Zhang; Irina Belousova
>
> **摘要:** Diffusion large language models are a compelling alternative to autoregressive models, yet existing RL methods for diffusion treat all denoising steps as equally important and rely on biased, high-variance likelihood estimates. We identify two fundamental weaknesses: the absence of temporal credit assignment across the denoising trajectory, and the systematic bias of mean-field likelihood estimates used for policy optimization. To address these, we propose Denoising-Aware Credit Assignment for GRPO (DACA-GRPO), a lightweight, plug-and-play enhancement for any GRPO-style trainer. DACA-GRPO introduces two complementary mechanisms: Denoising Progress Scores, which extract per-token importance weights from intermediate predictions at no additional forward cost, and Stratified Masking Likelihood, which partitions token positions into strata so that each token is predicted with most of the sequence as context, reducing the mean-field bias. Applied on top of three GRPO base methods, DACA-GRPO achieves consistent improvements across seven benchmarks spanning mathematical reasoning, code generation, constraint satisfaction, and constrained generation, with gains of up to 5.6pp on math reasoning, 7.4pp on code generation, 36.3pp on constraint satisfaction, and 5.9pp on JSON schema adherence.
>
---
#### [new 177] Recall Isn't Enough: Bounding Commitments in Personalized Language Systems
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于自然语言处理任务，解决个性化语言系统中的承诺约束问题。提出CBEA+LCV方法，有效减少系统失误，提升可靠性。**

- **链接: [https://arxiv.org/pdf/2605.16712](https://arxiv.org/pdf/2605.16712)**

> **作者:** Rui Tang; Yichi Zhang; Xi Chen; Chen Dong; Youwei Yang; Yumeng Shen
>
> **备注:** 14 pages, 3 figures, 22 tables; preprint version
>
> **摘要:** Long-context and memory systems usually treat personalization as a recall problem. In practice, many failures occur later, when a system commits: it turns noisy hints into hard constraints, drops rare witnesses, forgets downstream obligations, or answers despite infeasibility. We introduce Contract-Bounded Evidence Activation (CBEA) with Lexicographic Commitment Validation (LCV). CBEA activates a bounded evidence set using typed coverage, tail witnesses, and consequence debt; LCV validates structured commitments before prose and routes infeasible states to repair, abstention, or recontract. Across 360 fixtures and three generation backends, CBEA+LCV reaches zero failures within validator scope at 0.49-0.60 availability over attempted runs. Raw and long-context baselines with the same LCV gate reach zero only at 0.003-0.092. A shadow oracle diagnostic marks the limit: CBEA+LCV recalls 0.012 of uncompiled visible facts, while raw recalls 0.53. The result is a bounded operating point: explicit commitment control and 74-75% lower median input payload, not universal memory dominance.
>
---
#### [new 178] LLM-Based Intelligent Notification Composition: From Static Personalization to Context-Aware Persuasive Messaging
- **分类: cs.HC; cs.CL**

- **简介: 该论文属于智能通知生成任务，解决传统通知系统中消息质量不足的问题。通过LLM提升通知的上下文相关性、说服力等维度，优化用户互动效果。**

- **链接: [https://arxiv.org/pdf/2605.16264](https://arxiv.org/pdf/2605.16264)**

> **作者:** Nilesh Agrawal
>
> **备注:** 17 pages, 1 figure, 7 tables. Code available at this https URL
>
> **摘要:** Push notifications remain among the most direct channels through which digital platforms engage users, yet existing approaches have invested heavily in who to notify, when to notify, and what to recommend, while leaving how to communicate as the least-optimized stage. This paper argues that message quality is an independent, underinvested lever, and that LLMs create their most differentiated value precisely at this layer. We make three contributions. First, we define notification message quality along six dimensions (contextual relevance, clarity, actionability, novelty handling, linguistic freshness, and persuasive appropriateness) and show how LLM-based composition improves each relative to templates. Across reviewed deployments, reported improvements range from +8% to +14.5% CTR over static templates and +1% to +2.5% over mature slot-filling systems, though these span heterogeneous systems and should not be treated as directly comparable. Second, we provide an architectural attribution analysis disentangling message generation from adjacent components (targeting, ranking, timing), arguing that observed gains are frequently misattributed to text generation alone. Third, we introduce a three-criterion decision framework specifying when LLM generation is and is not the binding constraint. We support these arguments through a PRISMA-guided survey (28 sources from 142 screened), examine domain-specific applications across social media, food delivery, and e-commerce, and propose a unified architectural framework with budget-aware routing, grounded generation, candidate ranking, diversity controls, and online learning.
>
---
#### [new 179] Prompt2Fingerprint: Plug-and-Play LLM Fingerprinting via Text-to-Weight Generation
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于模型指纹识别任务，旨在解决LLM指纹注入的可扩展性问题。提出Prompt2Fingerprint框架，通过文本生成参数增量实现快速指纹注入。**

- **链接: [https://arxiv.org/pdf/2605.18474](https://arxiv.org/pdf/2605.18474)**

> **作者:** Sixu Chen; Xiang Chen; Hongyao Yu; Jiaxin Hong; Hao Fang; Shuoyang Sun; Bin Chen; Shu-Tao Xia
>
> **摘要:** The widespread deployment and redistribution of large language models (LLMs) have made model provenance tracking a critical challenge. While existing LLM fingerprinting methods, particularly active approaches that embed identity signals via fine-tuning, achieve high accuracy and robustness, they suffer from significant scalability bottlenecks. These methods typically treat fingerprint injection as an independent, one-off optimization task rather than a reusable capability, necessitating separate, resource-intensive training for every new identity. This incurs prohibitive computational costs and deployment delays. To address this, we propose Prompt2Fingerprint (P2F), the first framework that reformulates fingerprinting as a conditional parameter generation task. By leveraging a specialized generator, P2F maps textual descriptions directly to low-rank parameter increments in a single forward pass, enabling plug-and-play LLM fingerprint injection without further model retraining. Our experiments demonstrate that P2F maintains high fingerprint accuracy, harmlessness, and robustness while significantly reducing computational overhead, offering a scalable and instant solution for LLM ownership management.
>
---
#### [new 180] Scale Determines Whether Language Models Organize Representation Geometry for Prediction
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究语言模型的表征几何结构是否服务于预测任务。通过提出Subspace PGA度量，分析不同规模模型的几何组织特性，揭示模型规模对表征结构的影响。属于自然语言处理中的表征学习任务。**

- **链接: [https://arxiv.org/pdf/2605.17084](https://arxiv.org/pdf/2605.17084)**

> **作者:** Weilun Xu
>
> **摘要:** In language models, what a representation encodes is determined by the geometry of its representation space: distances, not activations, carry meaning. Existing tools characterize the shape of this geometry but do not ask what that shape is organized for. We introduce Subspace PGA, a metric that tests whether a layer's distance structure aligns with the readout subspace of the unembedding matrix $W_U$ more than with random subspaces of equal size. Across seven Pythia models (70M--6.9B) and three cross-family models, intermediate geometry is significantly organized for prediction (peak $z = 9$--$24$), but the degree is scale-dependent: small models ($d \leq 1024$) progressively lose it at late layers during training -- even as loss keeps improving -- while large models ($d \geq 2048$) preserve it throughout. We trace this to a capacity trade-off: a few dominant directions migrate away from $W_U$'s readout, masking rather than destroying the predictive structure beneath, and removing them restores alignment. Neither spectral metrics nor loss curves capture this distinction. Scale thus determines not only how well a model predicts, but how its representation geometry is organized to do so.
>
---
#### [new 181] Proof-Carrying Certificates for LLM Pipelines: A Trust-Boundary Architecture
- **分类: cs.LO; cs.CL; cs.CR; cs.PL**

- **简介: 该论文提出一种信任边界架构，用于验证大语言模型流水线的确定性计算，解决可信验证问题，通过证书机制确保高风险应用的安全性。**

- **链接: [https://arxiv.org/pdf/2605.16407](https://arxiv.org/pdf/2605.16407)**

> **作者:** George Koomullil
>
> **备注:** 83 pages, 1 figure, 12 tables
>
> **摘要:** We present a framework for verifying the deterministic structured computations surrounding a large language model rather than the model itself, extending a Lean 4 trust-boundary architecture to the generic interfaces of modern LLM pipelines. Certificate validity is a Lean 4 kernel type-check plus a sorry-free transitive axiom audit against the trusted set {propext, this http URL, this http URL}; other assumptions are declared and partitioned by tier (mathematical placeholders, cryptographic assumptions, ML/human oracles). The technical contribution comprises three local certificate families and two operators. The families are conflict-aware bilattice grounding (with an emission-gate soundness lemma), embedding sensitivity and paraphrase stability, and Hoare-style agent action. The operators are a Maximal Certifiable Residue, which turns abstention into the maximum-weight certifiable residue with audit-logged dropped claims, and a Compositional Stability theorem, which yields a closed-form pipeline-wide perturbation budget from per-layer gains and margins. The three families plus a Universal Assurance Card consolidator form the per-call deliverable for high-stakes deployments: patent and legal retrieval, regulated finance, clinical decision support, and agentic systems with irreversible side effects. A compiled Lean 4 reference artifact (Lean v4.30.0-rc2, Mathlib) covers all 22 certificate types, with 17 of 46 kernel-audited declarations axiom-free, the rest depending only on the trusted set and declared assumptions, and zero uses of sorryAx or this http URL. The three families are empirically tested through four registered pilots: bilattice grounding on adversarially perturbed HotpotQA, embedding sensitivity in short- and long-form settings, and Hoare-style agent action on a filesystem sandbox with adversarial prompt injection.
>
---
#### [new 182] Symphony for Speech-to-Text: Supporting Real-Time Medical Voice Interfaces
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决医疗领域语音识别准确率低的问题。提出Symphony系统，通过分解识别流程提升医学术语召回率和临床文本结构化能力。**

- **链接: [https://arxiv.org/pdf/2605.16545](https://arxiv.org/pdf/2605.16545)**

> **作者:** Arne Nix; Robert James; Lasse Borgholt; Anna B. Ekner; Lana Krumm; Julius Severin; Dan Engel; Lars Maaløe; Jakob Havtorn
>
> **摘要:** After decades of use in dictation and, more recently, ambient documentation, speech is emerging as a primary modality for interacting with technology and AI in healthcare. Yet medical speech recognition remains difficult: systems must capture specialized terminology, resolve contextual ambiguity, and render measurements, abbreviations, and clinical shorthand precisely. Existing solutions are typically optimized either for general-purpose transcription or narrow dictation workflows, limiting their reliability in safety-critical settings and their usefulness for broader clinical workflows. We introduce Symphony for Speech-to-Text, a medical-grade speech recognition system for real-time streaming and batch file-based clinical use. Symphony decomposes the transcription process into specialized components for recognition, formatting, and contextual correction to optimize medical term recall while producing clinically structured text in real time and adapting across use cases. Evaluations on public benchmark and medical speech datasets show that Symphony substantially outperforms state-of-the-art systems in clinical settings while matching or exceeding them in general-domain settings, suggesting robust generalization rather than overfitting. We release a clinical benchmark dataset to support reliable validation and further progress in medical speech recognition. Symphony is available through a production-grade API for live dictation, conversational transcription, and batch audio file processing.
>
---
#### [new 183] Multilingual OCR-Aware Fine-Tuning and Prompt-Guided Chain-of-Thought Reasoning for Multimodal Large Language Models
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于多模态语言模型任务，旨在解决OCR和多语言理解在复杂图像中的不足。通过合成数据、微调和推理提示，提升OCR准确性和视觉文本定位能力。**

- **链接: [https://arxiv.org/pdf/2605.16409](https://arxiv.org/pdf/2605.16409)**

> **作者:** Qinwu Xu; Xin Liu; Yifan Jiang; Haoyu Ren
>
> **摘要:** Optical character recognition (OCR) and multilingual text understanding remain major failure modes of multimodal large language models (MLLMs), particularly in real-world images containing cluttered layouts, small fonts, blur, occlusion, and complex typography. We present an OCR-aware multilingual multimodal training framework that combines (i) large-scale synthetic OCR-to-translation data generation, (ii) OCR-aware supervised fine-tuning (SFT) with LoRA adaptation, and (iii) structured visual chain-of-thought (CoT) prompting for reasoning under uncertain visual conditions. Using a LLaMA-based multimodal architecture, the proposed framework substantially improves OCR completeness, multilingual translation accuracy, and robustness under degraded visual conditions. Experimental results on multilingual receipts, menus, posters, signs, handwritten text, and document images demonstrate significantly improved visual-text grounding compared with the baseline model. In particular, the proposed OCR-aware post-training framework improves extraction of small, blurred, spatially scattered, and partially occluded text while reducing reliance on language priors under uncertain OCR conditions. Qualitative comparisons with frontier multimodal systems, including GPT-5-class and Gemini-family models, further suggest improved OCR grounding and reduced hallucination under noisy and visually ambiguous OCR scenarios. Overall, the results indicate that data-centric OCR-aware multimodal post-training provides an effective and scalable direction for improving multilingual OCR and OCR-based visual question answering systems.
>
---
#### [new 184] An Assessment of Human vs. Model Uncertainty in Soft-Label Learning and Calibration
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究人类与模型在软标签学习中的不确定性问题，旨在区分软标签的校准优势与数据错误修正的影响。通过控制实验验证软标签的正则化作用。**

- **链接: [https://arxiv.org/pdf/2605.18648](https://arxiv.org/pdf/2605.18648)**

> **作者:** Maja Pavlovic; Silviu Paun; Massimo Poesio
>
> **摘要:** Central to human-aligned AI is understanding the benefits of human-elicited labels over synthetic alternatives. While human soft-labels improve calibration by capturing uncertainty, prior studies conflate these benefits with the implicit correction of mislabeled data (mode shifts), obscuring true effects of soft-labels. We present a controlled audit of soft-label learning across MNIST and a synthetic variant, re-annotating subsets to extract human uncertainty. By decoupling soft-label supervision from underlying label mode shifts, we show that while human soft-labels do provide accuracy gains, their larger value lies in acting as a regularizer that improves model calibration on difficult samples and promotes stable convergence across training runs. Dataset cartography reveals models trained on human soft-labels mirror human uncertainty, whereas those trained on synthetic labels fail to align with humans. Broadly, this work provides a diagnostic testbed for human-AI uncertainty alignment.
>
---
#### [new 185] ChemVA: Advancing Large Language Models on Chemical Reaction Diagrams Understanding
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于化学反应图理解任务，解决LLM在解析化学图示时的视觉与语义瓶颈问题，提出ChemVA框架提升结构识别与化学推理能力。**

- **链接: [https://arxiv.org/pdf/2605.17214](https://arxiv.org/pdf/2605.17214)**

> **作者:** Mingyang Rao; Kehua Feng; Zhihui Zhu; Jiangzhen Fu; Hao Yu; Keyan Ding; Huajun Chen
>
> **摘要:** While Large Language Models (LLMs) have revolutionized scientific text processing, they exhibit a significant capability gap when interpreting chemical reaction diagrams. We identify two fundamental bottlenecks restricting current systems: a Visual Deficit, where generic vision encoders struggle to resolve the strict topological connectivity of dense molecular graphs, and a Semantic Disconnect, where standard linear strings, such as SMILES, fail to effectively activate the model's latent chemical reasoning. To bridge these gaps, we propose the Chemical Visual Activation (ChemVA) framework, which employs a Visual Anchor mechanism to ground functional groups via hybrid-granularity detection, followed by a semantic alignment approach that translates visual features into entity names to maximize knowledge activation in LLMs. We evaluate our approach on OCRD-Bench, a newly constructed dataset featuring dense visual-semantic contexts and comprehensive reaction coverage to evaluate the full spectrum from recognition to reasoning. Extensive experiments on OCRD-Bench demonstrate that ChemVA achieves 92.0% structural recognition accuracy. By bridging visual and semantic bottlenecks, our framework delivers a consistent performance gain of approximately 20 percentage points across 9 diverse LLMs, enabling open-weight models to rival proprietary SOTA systems in complex chemical reasoning tasks.
>
---
#### [new 186] Protection Is (Nearly) All You Need: Structural Protection Dominates Scoring in Globally Capped KV Eviction
- **分类: cs.LG; cs.CL; cs.CR; cs.PF**

- **简介: 该论文研究KV缓存淘汰策略，在全局限制下提升模型质量。解决缓存淘汰导致性能下降的问题，通过结构保护提升效果，验证不同策略的有效性。**

- **链接: [https://arxiv.org/pdf/2605.18053](https://arxiv.org/pdf/2605.18053)**

> **作者:** Gabriel Garcia
>
> **备注:** 38 pages, 6 figures, 25 tables (includes one longtable). Code and figure regeneration scripts: this https URL
>
> **摘要:** We study KV cache eviction under a shared globally capped decode-time harness. Seven policies (LRU, H2O, SnapKV, StreamingLLM, Ada-KV, QUEST, Random) share a prompt-boundary vulnerability: without structural protection, they collapse to near-zero quality on six pure-transformer models (F1$\leq$0.064). Reserving 10\% of cache at each boundary recovers 69--90\% of the $C{=}2{,}048$ reference-ceiling quality on seven LongBench models at $C{=}256$ (13\% retention); a ten-model panel spans 68--98\%. An attention-mass pilot (Qwen2.5-3B, $N{=}30$) suggests why: the position-0 sink holds ${\sim}75\%$ of prefix mass, while other boundary tokens sit near ${\sim}0.41{\times}$ uniform expectation, so attention scorers retain the sink but still drop structurally critical tokens. With protection, simplified score-isolation variants are TOST-equivalent to LRU at $K{=}32$ ($\Delta{=}0.02$); at $K{=}8$, attention policies pairwise converge yet beat LRU by 0.011--0.021 F1 across $C{=}256$ and $C{=}512$. Faithful Ada-KV/QUEST add ${\sim}0.03$--$0.04$ F1 on Mistral-7B and Phi-3.5 beyond simplified variants. A NIAH-32K regime-transfer pilot on Qwen3-4B (decode vs.\ prefill, $C{\in}\{512,2048\}$) shows near-identical protection lifts (ratio 0.99--1.00). At 64K, protection helps but recovery is modest; faithful per-head scoring matches full-cache ceiling on Gemma-3-4B at 6.3\% retention only when the model already supports strong 64K retrieval without eviction. Overall: protection dominates; scoring differences are secondary once boundaries are guarded; per-head allocation gives a further modest gain.
>
---
#### [new 187] Firefly: Illuminating Large-Scale Verified Tool-Call Data Generation from Real APIs
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出FireFly，用于生成可验证的大规模工具调用数据，解决真实API环境下数据生成难题。属于工具调用数据生成任务，通过逆向合成保证标签正确性。**

- **链接: [https://arxiv.org/pdf/2605.17558](https://arxiv.org/pdf/2605.17558)**

> **作者:** Yuxuan Lu; Ziyi Wang; Yingzhou Lu; Yisi Sang; Jiri Gesi; Xianfeng Tang; Yimeng Zhang; Zhenwei Dai; Hui Liu; Hanqing Lu; Chen Luo; Qi He; Benoit Dumoulin; Jing Huang; Dakuo Wang
>
> **摘要:** Training tool-calling agents requires large-scale trajectory data with verifiable labels, yet existing approaches either synthesize environments that diverge from real API behavior or generate tasks without ground-truth outcomes for verification. We present FireFly, a pipeline for generating verified tool-call data from real-world MCP servers. Our key insight is to invert the standard synthesis pipeline: rather than generating tasks and hoping they are solvable, we first let a strong LLM explore real APIs along graph-guided DAG structures, then synthesize tasks backward from observed outcomes, guaranteeing label correctness by construction. To handle the scale of real-world tool spaces (${\sim}$1,000 tools), we build a pairwise tool graph and sample sub-DAGs to focus exploration on semantically coherent workflows. To address environment drift in live APIs, we construct a retrieval-augmented simulator that caches all exploration results and replays them during training and evaluation, enabling fully offline and reproducible RL. Applying this pipeline yields 5,144 verified tasks spanning 240 servers and 993 tools. A 4B-parameter model trained with GRPO on FireFly matches Claude Sonnet 4.6 on our held-out test set and shows improvements on multiple tool-calling benchmarks including Tau2-Bench, MCPMark, and MCP-Atlas.
>
---
#### [new 188] SD-Search: On-Policy Hindsight Self-Distillation for Search-Augmented Reasoning
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出SD-Search，用于增强推理代理的搜索决策。任务是提升搜索增强型推理系统的性能，解决奖励稀疏问题，通过自蒸馏方法获得步骤级监督。**

- **链接: [https://arxiv.org/pdf/2605.18299](https://arxiv.org/pdf/2605.18299)**

> **作者:** Yufei Ma; Zihan Liang; Ben Chen; Zhipeng Qian; Huangyu Dai; Lingtao Mao; Xuxin Zhang; Chenyi Lei; Wenwu Ou
>
> **摘要:** Search-augmented reasoning agents interleave internal reasoning with calls to an external retriever, and their performance relies on the quality of each issued query. However, under outcome-reward reinforcement learning, every search decision in a rollout shares the same trajectory-level reward, leaving individual queries without step-specific credit. Recent process-supervision approaches address this gap by drawing step-level signals from outside the policy, relying either on a much larger teacher model, or on sub-question annotations produced by a stronger external system. In contrast, we propose SD-Search, which derives step-level supervision from the policy itself through on-policy hindsight self-distillation, requiring neither an external teacher nor additional annotations. In SD-Search, a single model plays two roles that differ only in conditioning: a student that sees only the context available at inference time, and a teacher that additionally conditions on a compact hindsight block summarizing the search queries and final outcomes of a group of rollouts sampled from the same question. Since the teacher knows how each rollout unfolded and which ones succeeded, its query distribution implicitly marks which decisions were worth making, and the student is trained to recover this behavior by minimizing the token-level Jensen--Shannon divergence to the teacher at search-query positions. This layers a dense, step-level signal on top of GRPO's coarse trajectory reward. Crucially, this signal is produced by the policy itself within the standard RL training loop, without external model inference, auxiliary annotation pipeline, or additional training stage.
>
---
#### [new 189] Thinking with Patterns: Breaking the Perceptual Bottleneck in Visual Planning via Pattern Induction
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉规划任务，旨在解决VLMs在复杂视觉输入下的感知瓶颈。通过引入模式推理和归纳策略，提升规划效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.16848](https://arxiv.org/pdf/2605.16848)**

> **作者:** Yichang Jian; Boyuan Xiao; Zhenyuan Huang; Yifei Peng; Yao-Xiang Ding
>
> **摘要:** Planning from raw visual input remains a significant challenge for current Vision-Language Models (VLMs), when the complexity of input is beyond their one-step perception capability. Motivated by recent advances in Thinking with Images (TWI), a reasonable solution is to decompose the perception process into simpler steps by iteratively acquiring and incorporating local visual evidence. However, even though current VLMs are well-trained in general TWI ability, their perceptual bottleneck in the planning domain remains. To tackle this challenge, we formulate TWI as a tool to gradually build and reflect an accurate internal world model. We find that the resulting training-free planning strategy enables VLMs to solve tasks that are far beyond their initial capabilities, at the cost that too many TWI operations would significantly increase the computational overhead. To further improve efficiency, we propose Pattern Inference, a novel TWI strategy enabling VLMs to actively recognize known visual patterns in the new tasks and directly infer local world model structures. To obtain these patterns, we propose Pattern Induction, an online inductive learning strategy treating visual patterns as composite and reusable experts, which are autonomously discovered and optimized from experience. Experimental evaluations in FrozenLake, Crafter and CubeBench domains show that our approaches achieve a desirable balance between accuracy and efficiency.
>
---
#### [new 190] Medical Context Distorts Decisions in Clinical Vision Language Models
- **分类: cs.CV; cs.CL**

- **简介: 论文研究医疗场景下视觉语言模型的决策偏差，属于临床辅助决策任务。针对模型过度依赖文本、受无关病史干扰及提示敏感等问题，通过实验分析不同模型表现，提出需加强验证与测试。**

- **链接: [https://arxiv.org/pdf/2605.17436](https://arxiv.org/pdf/2605.17436)**

> **作者:** David Restrepo; Ira Ktena; Maria Vakalopoulou; Stergios Christodoulidis; Enzo Ferrante
>
> **摘要:** Vision-language models (VLMs) are increasingly proposed for clinical decision support, yet their reliability in real-world scenarios that require integrating both visual and textual context from medical records remains poorly characterized. This paper identifies three failure modes: (1) modality over-reliance on text over images, (2) spurious reliance on irrelevant clinical history, and (3) prompt sensitivity across semantically equivalent inputs. We evaluate a diverse set of general-domain and medically-tuned open and closed VLMs on chest x-ray tasks using MIMIC-CXR. By systematically manipulating image-text alignment, clinical history, and prompt formulations, we found that VLM decisions are dominated by the text modality, even when visual evidence is available. Moreover, we observed that VLMs are heavily influenced by irrelevant reports, while minor prompt changes can reverse correct image-based predictions. Our findings underscore the need for explicit safeguards and stress-testing before considering the use of these models in clinical practice.
>
---
#### [new 191] TIER: Trajectory-Invariant Execution Rewards for Multi-Step Tool Composition
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决多步骤工具组合中的奖励设计问题。提出TIER框架，通过执行过程直接生成奖励，提升模型在复杂任务中的表现。**

- **链接: [https://arxiv.org/pdf/2605.16790](https://arxiv.org/pdf/2605.16790)**

> **作者:** Anay Kulkarni; ChiaEn Lu; Dheeraj Mekala; Jayanth Srinivasa; Gaowen Liu; Jingbo Shang
>
> **备注:** Preprint. Submitted to NeurIPS 2026. 28 pages, 7 figures, 8 tables. Code and datasets available at this https URL
>
> **摘要:** Tool use enables large language models to solve complex tasks through sequences of API calls, yet existing reinforcement learning approaches fail to scale to multi-step composition settings. Outcome-based rewards provide only sparse feedback, while trajectory-supervised rewards depend on annotated reference solutions, penalizing valid alternatives and limiting scalability. We propose TIER: Trajectory-Invariant Execution Rewards, a reward framework that derives supervision directly from function schemas and runtime execution, rather than from reference trajectories. The reward decomposes into format validity, schema adherence, execution success, and answer correctness, providing dense, interpretable sequence-level feedback derived from fine-grained verification of individual steps of tool use. This design allows any valid execution path to receive credit, naturally supporting multiple solution strategies and adapting to evolving tool interfaces. On DepthBench, a compositional benchmark stratified by depth (1 to 6 steps), TIER achieves >90% accuracy across steps, where trajectory-supervised rewards collapse beyond step-4. We further demonstrate consistent gains on benchmarks like BFCL v3 and NestFUL. Ablation studies confirm that all reward components are necessary, highlighting the importance of multi-level supervision for compositional reasoning.
>
---
#### [new 192] MemRepair: Hierarchical Memory for Agentic Repository-Level Vulnerability Repair
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出MemRepair，解决多文件漏洞修复问题。通过引入分层记忆机制，提升自动化修复的可靠性和效率。**

- **链接: [https://arxiv.org/pdf/2605.17444](https://arxiv.org/pdf/2605.17444)**

> **作者:** Simiao Liu; Li Zhang; Fang Liu; Xiaoli Lian; Yang Liu; Yinghao Zhu
>
> **摘要:** Modern software ecosystems face a rapidly growing number of disclosed vulnerabilities, increasing the need for automated repair techniques that can operate reliably at repository scale. Although Large Language Model (LLM)-based agents have recently shown promise for automated vulnerability repair (AVR), most existing systems still treat repair as a single generation step over the currently visible code context. As a result, they lack a persistent mechanism for reusing prior fixes or learning from failed validation attempts, which limits their effectiveness on complex, multi-file repair tasks. We present MemRepair, a memory-augmented agentic framework that formulates vulnerability repair as an iterative, experience-driven process. MemRepair combines three complementary memory layers, i.e., History-Fix, Security-Pattern, and Refinement-Trajectory memories, with a dynamic feedback-driven refinement loop. This design allows the agent to retrieve repository-specific repair conventions, apply reusable security defenses, and exploit prior "failure-to-success" trajectories to revise semantically invalid patches based on runtime evidence. We evaluate MemRepair on three representative repository-level vulnerability repair benchmarks: SEC-Bench, PatchEval (Python, Go, JavaScript), and the C++ subset of Multi-SWE-bench. MemRepair achieves state-of-the-art resolution rates of 58.0%, 58.2%, and 30.58%, respectively, outperforming strong general-purpose agents such as OpenHands and SWE-agent, as well as the specialized AVR tool InfCode-C++, while maintaining competitive repair cost. These results show that persistent, hierarchical repair memory can substantially improve the reliability of agentic vulnerability repair across diverse languages and repository settings.
>
---
#### [new 193] Entropy-Gradient Inversion: Moving Toward Internal Mechanism of Large Reasoning Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于大模型推理研究，旨在解决模型内部机制与行为分析的差距及强化学习不稳定问题。提出熵梯度反转和CorR-PO方法，提升推理性能。**

- **链接: [https://arxiv.org/pdf/2605.17770](https://arxiv.org/pdf/2605.17770)**

> **作者:** Junyao Yang; Chen Qian; Kun Wang; Linfeng Zhang; Quanshi Zhang; Yong Liu; Dongrui Liu
>
> **备注:** 28 pages, 5 figures, 9 tables
>
> **摘要:** The advancement of Large Reasoning Models (LRMs) has catalyzed a paradigm shift from reactive ``fast thinking'' text generation to systematic, step-by-step ``slow thinking'' reasoning, unlocking state-of-the-art performance in complex mathematical and logical tasks. However, the field faces \textit{the fundamental gap between token-level behavioral analysis and internal reasoning mechanisms, and the instability of reinforcement learning (RL) for reasoning optimization relying on costly external verifiers}. We identify and formally define \textbf{Entropy-Gradient Inversion}, a robust negative correlation between token entropy and logit gradients that acts as a definitive geometric fingerprint for LRM reasoning capability. Building on this, we propose \textbf{Correlation-Regularized Group Policy Optimization (CorR-PO)}, which embeds this inversion signature into RL reward regularization. Extensive experiments on various reasoning benchmarks across multiple model scales show CorR-PO consistently outperforms state-of-the-art baselines, confirming that stronger inversion directly correlates with superior reasoning performance.
>
---
#### [new 194] HEED: Density-Weighted Residual Alignment for Hybrid Vision-Language Model Distillation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉-语言模型压缩任务，解决蒸馏过程中文本理解能力下降的问题。通过引入密度加权残差对齐（HEED），提升学生模型在文本相关任务上的表现。**

- **链接: [https://arxiv.org/pdf/2605.17093](https://arxiv.org/pdf/2605.17093)**

> **作者:** Yihao Liang; Niraj K. Jha
>
> **摘要:** Distilling vision-language models into faster hybrid architectures, such as 3:1 Mamba-2/attention mixes, is now standard practice for making inference efficient. Aggregate benchmarks suggest that this works but they hide selective failures. When we distill Qwen3-VL-8B-Instruct into a 3:1 Mamba-2/attention hybrid, student model stays within 2 points of the teacher across visual reasoning benchmarks like MMStar, MMBench, and MMMU-Pro, while dropping 13 points on optical-character-recognition and document tasks. The student can still understand the scene but loses the fine-grained text needed to answer. We localize much of the failure to a specific kind of position. In a high-resolution image, most patches are sky, wall, or smooth texture, while a small fraction carries text, edges, object boundaries, or other local details. In a token-level diagnostic, the top 10% highest-density patches have 3.6$\times$ larger residual drift than the bottom 10% lowest-density patches and 3.5$\times$ larger teacher-masking answer contribution. Uniform weighting devotes many loss terms to low-information background patches, whereas sparse answer-bearing patches receive no special protection. The required intervention is minimal: we replace uniform residual alignment with density-weighted residual alignment, using patch self-dissimilarity as a training-free proxy for position importance. We call this HEED. Compared with normal end-to-end distillation, HEED increases performance by 8.7 points on OCRBench v2 and 5.13 points on a 10-benchmark average. The gain is realized on different teacher models and hybrid architectures. After standard post-training, the student reaches teacher-level performance on the 10-benchmark average with a 4.12$\times$ throughput and a 68% memory saving at 128k context, with no additional parameters and no inference-time cost.
>
---
#### [new 195] Remembering More, Risking More: Longitudinal Safety Risks in Memory-Equipped LLM Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究记忆增强型大语言模型代理的长期安全风险，解决多任务环境下记忆污染导致的安全问题，通过实验分析记忆累积对安全的影响。**

- **链接: [https://arxiv.org/pdf/2605.17830](https://arxiv.org/pdf/2605.17830)**

> **作者:** Ahmad Al-Tawaha; Shangding Gu; Peizhi Niu; Ruoxi Jia; Ming Jin
>
> **摘要:** Safety evaluations of memory-equipped LLM agents typically measure within-task safety: whether an agent completes a single scenario safely, often under adversarial conditions such as prompt injection or memory poisoning. In deployment, however, a single agent serves many independent tasks over a long horizon, and memory accumulated during earlier tasks can affect behavior on later, unrelated ones. Studying this regime requires evaluation along the temporal dimension across tasks: not whether an agent is safe at any single memory state, but how its safety profile changes as memory accumulates across many independent interactions. We call this failure mode temporal memory contamination. To isolate memory exposure from stream non-stationarity, we introduce a trigger-probe protocol that evaluates a fixed probe set against read-only memory snapshots at varying prefix lengths, together with a NullMemory counterfactual baseline for identifying memory-induced violations. We apply this protocol across three deployment scenarios spanning records, memos, forms, and email correspondence and eight memory architectures, and additionally on Claw-like AI agents, such as OpenClaw, using the platform's native memory mechanism. Memory-enabled agents consistently exceed the NullMemory baseline, and memory-induced violation rates show a robust upward trend with exposure length on both agent classes. Order-randomization experiments indicate that the effect is driven primarily by accumulated content rather than encounter order. Finally, a structural consequence of the event decomposition is that memory-induced risk is detectable from retrieval state before generation, which we confirm with a high-recall diagnostic monitor. Our results argue for treating memory safety as a longitudinal property that requires temporal evaluation, not a single-state property that can be captured by a snapshot.
>
---
#### [new 196] Causal Intervention-Based Memory Selection for Long-Horizon LLM Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于长期记忆管理任务，解决LLM代理在多轮对话中如何有效选择有用记忆的问题。提出CMI方法，通过因果干预筛选提升任务性能的记忆。**

- **链接: [https://arxiv.org/pdf/2605.17641](https://arxiv.org/pdf/2605.17641)**

> **作者:** Saksham Sahai Srivastava
>
> **备注:** 12 pages, 3 figures, 3 tables
>
> **摘要:** Long-horizon LLM agents rely on persistent memory to support interactions across sessions, yet existing memory systems often retrieve context using semantic similarity or broad history inclusion, treating retrieved memories as uniformly useful. This assumption is fragile because memories may be topically related while remaining irrelevant, stale, or misleading. We propose Causal Memory Intervention (CMI), a causal memory-selection technique that estimates how candidate memories affect the model's answer under controlled interventions, selecting memories that improve task performance while suppressing unstable, irrelevant, or harmful ones. To evaluate this setting, we introduce Causal-LoCoMo, a causally annotated benchmark derived from long conversational data, where each example contains a user request, a structured memory bank, useful memories, irrelevant distractors, and synthetic harmful memories. We compare CMI against vector, graph, reflection, summary, full-history, and no-memory baselines. Results show that CMI achieves a stronger balance between answer quality and robustness to misleading memory, suggesting that reliable long-term memory requires selecting context based on causal usefulness rather than relevance alone. The full framework, benchmark construction code, and experimental pipeline are available at this https URL.
>
---
#### [new 197] RAG-based EEG-to-Text Translation Using Deep Learning and LLMs
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于EEG-to-Text翻译任务，旨在解决EEG信号中句子级语言信息解码难题。通过结合RAG框架与LLM，提升解码效果。**

- **链接: [https://arxiv.org/pdf/2605.17503](https://arxiv.org/pdf/2605.17503)**

> **作者:** Enrico Collautti; Xiaopeng Mao; Luca Tonin; Stefano Tortora; Sadasivan Puthusserypady
>
> **备注:** 6 pages, 2 figures. Submitted to the 2026 IEEE International Conference on Systems, Man, and Cybernetics
>
> **摘要:** The decoding of linguistic information from electroencephalography (EEG) signals remains an extremely challenging problem in brain-computer interface (BCI) research. In particular, sentence-level decoding from EEG is difficult due to the low signal-to-noise ratio of these recordings. Previous studies tackling this problem have typically failed to surpass random baseline performance unless teacher forcing is used during the inference phase. In this work, we propose a retrieval-augmented generation (RAG)-based sentence-level EEG-to-text decoding pipeline that combines an EEG encoder aligned with semantic sentence embeddings, a vector retrieval stage, and a large language model (LLM) to refine retrieved sentences into coherent output. Experiments are conducted on the Zurich Cognitive Language Processing Corpus (ZuCo) dataset, which contains single-trial EEG recordings collected during silent reading. To evaluate whether the system extracts meaningful information from these EEG signals, the results are compared with a random baseline. In nine subjects, the proposed pipeline outperforms the random baseline, achieving a mean cosine similarity of 0.181 +- 0.022 compared to 0.139 +- 0.029 for the baseline, corresponding to a relative improvement of 30.45%. Statistical analysis further confirms that this improvement is significant, following a strict evaluation workflow where inference is performed without access to ground-truth labels.
>
---
#### [new 198] FastOCR: Dynamic Visual Fixation via KV Cache Pruning for Efficient Document Parsing
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文档解析任务，解决OCR中视觉令牌过多导致的计算成本高的问题。提出FastOCR框架，通过动态视觉聚焦和缓存优化，减少注意力计算，提升效率。**

- **链接: [https://arxiv.org/pdf/2605.17447](https://arxiv.org/pdf/2605.17447)**

> **作者:** Zihan Tang; Leqi Shen; Hui Chen; Ao Wang; Ben Wan; Yan Feng; Ke Zhang; Sicheng Zhao; Tongxuan Liu; Guiguang Ding
>
> **摘要:** Vision-Language Models (VLMs) have shown strong promise on Optical Character Recognition (OCR), yet the sheer number of visual tokens required to encode dense documents incurs prohibitive inference cost. Existing pruning methods rely on physical eviction, e.g., permanently discarding visual tokens during the prefill stage. While effective for natural images, this strategy fundamentally breaks down on OCR, where virtually every visual token may correspond to a character or structural element, and any irreversible loss leads to catastrophic accuracy degradation. We observe that, although document images appear globally dense and seemingly unprunable, the model's attention to them is in fact temporally sparse: at each decoding step it concentrates on a small region that shifts gradually across steps, much as a human reader fixates on successive words rather than perceiving an entire page at once. Motivated by this Dynamic Visual Fixation phenomenon, we recast the intractable global pruning problem as a tractable local, dynamic one and propose FastOCR, a training-free framework with two complementary modules. Specifically, Focal-Guided Pruning identifies a small set of focal layers and selects the most task-relevant visual tokens from them at each step, while Cross-Step Fixation Reuse exploits the gradual shift of fixation to warm-start each step from the previous one. By dynamically adjusting which tokens are attended rather than evicting any from the cache, FastOCR avoids permanent information loss. Extensive experiments show that FastOCR serves as a plug-and-play acceleration module, generalizing consistently across five VLMs of varying sizes and architectures. On Qwen2.5-VL, FastOCR retains 98% of the unpruned model's accuracy while attending to only 5% of the visual tokens per decoding step, reducing attention latency by 3.0$\times$.
>
---
#### [new 199] When AI Tells You What You Want to Hear: Sycophantic Behavior of Large Language Models in Dementia Care Settings
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于AI伦理研究任务，探讨LLMs在老年痴呆护理中是否表现出迎合行为。通过实验分析不同提示对模型响应质量的影响，揭示其在高风险医疗环境中的潜在风险。**

- **链接: [https://arxiv.org/pdf/2605.16288](https://arxiv.org/pdf/2605.16288)**

> **作者:** Christian Kolb
>
> **备注:** 10 pages, 4 figures. Exploratory study
>
> **摘要:** Large language models (LLMs) are increasingly used in clinical and care settings. This exploratory study investigates whether LLMs exhibit sycophantic behavior - adapting their responses to social expectation signals rather than maintaining professional quality - in the context of dementia care. Five prompts with systematically increasing confirmatory and authority-related framing (P1 neutral to P5 authority-signaled implementation support) were submitted to four LLMs (GPT-5, Claude Sonnet 4.6, Gemini 3.1 Pro, Mistral Large), each repeated five times (N = 100 responses). Responses were evaluated using an LLM-as-a-Judge methodology against seven nursing-ethical quality criteria (K1-K7) and a tone scale (0-3). All models showed significant negative Spearman correlations between prompt level and response quality (rho ranging from -0.543 to -0.734, all p < 0.01). Mistral Large exhibited the most pronounced effect (rho = -0.734), with mean scores dropping from 6.0/7 at P1 to 0.2/7 at P5. The findings suggest that LLMs pose context-sensitive risks in high-stakes care environments and that prompt framing significantly shapes response quality - a dimension that has received insufficient attention in healthcare AI deployment.
>
---
#### [new 200] Vidya: An AI-Driven Modular Pipeline for Archival Automation and Semantic Metadata Enrichment
- **分类: cs.DL; cs.CL**

- **简介: 该论文提出Vidya系统，解决历史档案数字化中的“暗数据”问题，通过AI和开源工具实现自动化语义元数据增强。**

- **链接: [https://arxiv.org/pdf/2605.16338](https://arxiv.org/pdf/2605.16338)**

> **作者:** Cloter Migliorini Filho; Julia Graciela Machado; Edson Armando Silva; Marcella Scoczynski
>
> **摘要:** The large-scale digitization of historical archives has created a paradox: "dark data"-digital objects lacking metadata for retrieval. Manual archival description is slow and expensive, limiting discovery and reuse. We propose Vidya, a modular pipeline that orchestrates Large Language Models (LLMs) and FOSS tools to automate semantic enrichment and archival ingestion at scale. Vidya constrains generations using YAML-defined ontologies and Pydantic validation, producing deterministic, structured JSON outputs from probabilistic models. Developed at Laboratory for Digital Humanities and Innovation (LAMUHDI) of the State University of Ponta Grossa (UEPG), Vidya applies Maker principles and open-source practices to enable low-cost deployment in memory institutions using modest hardware. We compare LLM performance and present a cost-benefit analysis showing major gains, reducing processing time from decades to days while complying with NOBRADE and ISAD(G).
>
---
## 更新

#### [replaced 001] Language Models as Efficient Reward Function Searchers for Custom-Environment Multi-Objective Reinforcement
- **分类: cs.LG; cs.AI; cs.CL; eess.SY**

- **简介: 该论文属于多目标强化学习任务，旨在解决复杂环境中奖励函数设计难题。提出ERFSL框架，利用大语言模型高效搜索奖励函数，实现快速优化与精准调整。**

- **链接: [https://arxiv.org/pdf/2409.02428](https://arxiv.org/pdf/2409.02428)**

> **作者:** Guanwen Xie; Jingzehua Xu; Yiyuan Yang; Yimian Ding; Shuai Zhang
>
> **摘要:** Achieving the effective design and improvement of reward functions in reinforcement learning (RL) tasks with complex custom environments and multiple requirements presents considerable challenges. In this paper, we propose ERFSL, an efficient reward function searcher using LLMs, which enables LLMs to be effective white-box searchers and highlights their advanced semantic understanding capabilities. Specifically, we generate reward components for each numerically explicit user requirement and employ a reward critic to identify the correct code form. Then, LLMs assign weights to the reward components to balance their values and iteratively adjust the weights without ambiguity and redundant adjustments by flexibly adopting directional mutation and crossover strategies, similar to genetic algorithms, based on the context provided by the training log analyzer. We applied the framework to a customized data collection RL task without direct human feedback or reward examples (zero-shot learning). The reward critic successfully corrects the reward code with only one feedback instance for each requirement, effectively preventing unrectifiable errors. The initialization of weights enables the acquisition of different reward functions within the Pareto solution set without the need for weight search. Even in cases where a weight is 500 times off, on average, only 5.2 iterations are needed to meet user requirements. The ERFSL also works well with most prompts utilizing GPT-4o mini, as we decompose the weight searching process to reduce the requirement for numerical and long-context understanding capabilities.
>
---
#### [replaced 002] Dynamic Generation of Multi-LLM Agents Communication Topologies with Graph Diffusion Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多智能体系统任务，旨在解决LLM代理通信拓扑设计问题。通过引入GTD框架，实现动态生成高效、适应任务的通信结构。**

- **链接: [https://arxiv.org/pdf/2510.07799](https://arxiv.org/pdf/2510.07799)**

> **作者:** Eric Hanchen Jiang; Mengting Li; Guancheng Wan; Sophia Yin; Yuchen Wu; Xiao Liang; Xinfeng Li; Yizhou Sun; Wei Wang; Kai-Wei Chang; Ying Nian Wu
>
> **备注:** ACL 2026 Main
>
> **摘要:** The efficiency of multi-agent systems driven by large language models (LLMs) largely hinges on their communication topology. However, designing an optimal topology is a non-trivial challenge, as it requires balancing competing objectives such as task performance, communication cost, and robustness. Existing frameworks often rely on static or hand-crafted topologies, which inherently fail to adapt to diverse task requirements, leading to either excessive token consumption for simple problems or performance bottlenecks for complex ones. To address this challenge, we introduce a novel generative framework called \textit{Guided Topology Diffusion (GTD)}. Inspired by conditional discrete graph diffusion models, GTD formulates topology synthesis as an iterative construction process. At each step, the generation is steered by a lightweight proxy model that predicts multi-objective rewards (e.g., accuracy, utility, cost), enabling real-time, gradient-free optimization towards task-adaptive topologies. This iterative, guided synthesis process distinguishes GTD from single-step generative frameworks, enabling it to better navigate complex design trade-offs. We validated GTD across multiple benchmarks, and experiments show that this framework can generate highly task-adaptive, sparse, and efficient communication topologies, significantly outperforming existing methods in LLM agent collaboration.
>
---
#### [replaced 003] ADMEDTAGGER: an annotation framework for distillation of expert knowledge for the Polish medical language
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医学文本分类任务，解决标注资源不足的问题。利用大模型进行知识蒸馏，训练出高效的分类器，实现轻量化、快速的医疗文本标签识别。**

- **链接: [https://arxiv.org/pdf/2601.09722](https://arxiv.org/pdf/2601.09722)**

> **作者:** Franciszek Górski; Andrzej Czyżewski
>
> **摘要:** In this work, we present an annotation framework that demonstrates how a multilingual LLM pretrained on a large corpus can be used as a teacher model to distill the expert knowledge needed for tagging medical texts in Polish. This work is part of a larger project called ADMEDVOICE, within which we collected an extensive corpus of medical texts representing five clinical categories - Radiology, Oncology, Cardiology, Hypertension, and Pathology. Using this data, we had to develop a multi-class classifier, but the fundamental problem turned out to be the lack of resources for annotating an adequate number of texts. Therefore, in our solution, we used the multilingual Llama3.1 model to annotate an extensive corpus of medical texts in Polish. Using our limited annotation resources, we verified only a portion of these labels, creating a test set from them. The data annotated in this way were then used for training and validation of 3 different types of classifiers based on the BERT architecture - the distilled DistilBERT model, BioBERT fine-tuned on medical data, and HerBERT fine-tuned on the Polish language corpus. Among the models we trained, the DistilBERT model achieved the best results, reaching an F1 score > 0.80 for each clinical category and an F1 score > 0.93 for 3 of them. In this way, we obtained a series of highly effective classifiers that represent an alternative to large language models, due to their nearly 500 times smaller size, 300 times lower GPU VRAM consumption, and several hundred times faster inference.
>
---
#### [replaced 004] Lying with Truths: Open-Channel Multi-Agent Collusion for Belief Manipulation via Generative Montage
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文属于安全与可信AI任务，研究如何通过公开渠道的真相片段诱导受害者形成错误信念，提出Generative Montage框架进行认知共谋攻击。**

- **链接: [https://arxiv.org/pdf/2601.01685](https://arxiv.org/pdf/2601.01685)**

> **作者:** Jinwei Hu; Xinmiao Huang; Youcheng Sun; Yi Dong; Xiaowei Huang
>
> **备注:** Accepted to the ACL 2026 Main Conference (Oral Presentation)
>
> **摘要:** As large language models (LLMs) transition to autonomous agents synthesizing real-time information, their reasoning capabilities introduce an unexpected attack surface. This paper introduces a novel threat where colluding agents steer victim beliefs using only truthful evidence fragments distributed through public channels, without relying on covert communications, backdoors, or falsified documents. By exploiting LLMs' overthinking tendency, we formalize the first cognitive collusion attack and propose Generative Montage: a Writer-Editor-Director framework that constructs deceptive narratives through adversarial debate and coordinated posting of evidence fragments, causing victims to internalize and propagate fabricated conclusions. To study this risk, we develop CoPHEME, a dataset derived from real-world rumor events, and simulate attacks across diverse LLM families. Our results show pervasive vulnerability across 14 LLM families: attack success rates reach 74.4% for proprietary models and 70.6% for open-weights models. Counterintuitively, stronger reasoning capabilities increase susceptibility, with reasoning-specialized models showing higher attack success than base models or prompts. Furthermore, these false beliefs then cascade to downstream judges, achieving over 60% deception rates, highlighting a socio-technical vulnerability in how LLM-based agents interact with dynamic information environments. Our implementation and data are available at: this https URL.
>
---
#### [replaced 005] Multi-Dimensional Behavioral Evaluation of Agentic Stock Prediction Systems Using Large Language Model Judges with Closed-Loop Reinforcement Learning Feedback
- **分类: cs.LG; cs.AI; cs.CL; q-fin.CP**

- **简介: 该论文属于智能系统评估任务，旨在解决传统评估无法诊断决策过程的问题。通过行为评估方法，量化系统在多个维度的表现，并利用强化学习优化模型性能。**

- **链接: [https://arxiv.org/pdf/2605.05739](https://arxiv.org/pdf/2605.05739)**

> **作者:** Mohammad Al Ridhawi; Mahtab Haj Ali; Hussein Al Osman
>
> **备注:** 17 pages, 5 figures, 14 tables. Manuscript submitted to Applied Artificial Intelligence (Taylor and Francis)
>
> **摘要:** Agentic artificial intelligence systems produce outputs through sequences of interdependent autonomous decisions, yet standard evaluation assesses outputs alone and cannot diagnose the underlying process. We develop a behavioral evaluation methodology that complements output-level testing by scoring the intermediate decision process itself. Behavioral traces logged at each autonomous decision point are grouped into five-day episodes and scored along six domain-specific dimensions (regime detection, routing, adaptation, risk calibration, strategy coherence, error recovery) by an ensemble of three large language model (LLM) judges. A perturbation procedure that corrupts one dimension while leaving the other five intact confirms dimension specificity; cross-model agreement reaches Krippendorff's alpha = 0.85. The composite behavioral score correlates at Spearman rho = 0.72 with realized 20-day Sharpe ratio. Closing the loop, the framework converts deficient per-dimension scores into a credit-assigned penalty added to the Soft Actor-Critic reward. Three fine-tuning cycles, confined to validation data, reduce one-day MAPE from 0.61% to 0.54% (11.5% relative; p<0.001, d=0.31) on the held-out 2017 to 2025 test period, significant under Diebold-Mariano and localized by Giacomini-White to the high-volatility regime. The methodology is application-agnostic and applies to any agentic system whose intermediate decisions can be logged.
>
---
#### [replaced 006] Reinforcement Learning for LLM Post-Training: A Survey
- **分类: cs.CL**

- **简介: 该论文属于大语言模型后训练任务，旨在解决模型输出有害或不一致的问题。论文提出统一框架，分析RLHF与RLVR方法，进行技术对比与实证研究。**

- **链接: [https://arxiv.org/pdf/2407.16216](https://arxiv.org/pdf/2407.16216)**

> **作者:** Zhichao Wang; Kiran Ramnath; Bin Bi; Shiva Kumar Pentyala; Sougata Chaudhuri; Shubham Mehrotra; Zixu; Xiang-Bo Mao; Sitaram Asur; Cheng
>
> **摘要:** Large language models (LLMs) trained via pretraining and supervised fine-tuning (SFT) can still produce harmful and misaligned outputs, or struggle in domains like math and coding. Reinforcement learning (RL)-based post-training methods, including Reinforcement Learning from Human Feedback (RLHF) methods like Direct Preference Optimization (DPO) and Reinforcement Learning with Verifiable Rewards (RLVR) approaches like PPO and GRPO, have made remarkable gains to alleviate these issues. Yet, no existing work offers a technically detailed comparison of the various methods driving this progress. In order to fill this gap, we present a timely survey that connects foundational components with latest advancements. We derive a single policy gradient framework that unifies pretraining, SFT, RLHF, and RLVR as special cases while also organizing the more recent techniques therein. The main contributions of our survey are as follows: (1) a self-contained introduction to MLE, RLHF, and RLVR foundations and the unified policy gradient framework; (2) detailed technical analysis of PPO- and GRPO-based methods alongside offline and iterative DPO approaches, decomposed along prompt sampling, response sampling, and gradient coefficient axes; (3) standardized notation enabling direct cross-method comparison; and (4) comprehensive comparison of implementation details and empirical results of each method in the appendix. We aim to serve as a technically grounded reference for researchers and practitioners working on LLM post-training.
>
---
#### [replaced 007] Ontology-Constrained Neural Reasoning in Enterprise Agentic Systems: A Neurosymbolic Architecture for Domain-Grounded AI Agents
- **分类: cs.AI; cs.CL; cs.SE**

- **简介: 该论文属于企业智能系统任务，解决LLM的幻觉、领域偏移和合规性问题。通过神经符号架构和本体约束，提升代理的准确性与一致性。**

- **链接: [https://arxiv.org/pdf/2604.00555](https://arxiv.org/pdf/2604.00555)**

> **作者:** Thanh Luong Tuan; Abhijit Sanyal
>
> **备注:** 24 pages, 6 tables, 6 figures, 1 algorithm, 65 references. Replication study: 1,800 runs (600 per model) across 5 regulated industries (3 English, 2 Vietnamese) and 3 LLMs (Claude Sonnet 4, Qwen 2.5 72B, Gemma 4 26B). v3 changes: deep-review trim from 34pp
>
> **摘要:** Enterprise adoption of Large Language Models (LLMs) is constrained by hallucination, domain drift, and the inability to enforce regulatory compliance at the reasoning level. We present a neurosymbolic architecture implemented within the Foundation AgenticOS (FAOS) platform that addresses these limitations through ontology-constrained neural reasoning. We introduce a three-layer ontological framework--Role, Domain, and Interaction ontologies--grounding LLM-based enterprise agents. We formalize asymmetric neurosymbolic coupling: current enterprise systems constrain agent inputs (context assembly, tool discovery, governance thresholds) but not outputs, and we propose mechanisms extending this coupling to output-side validation (response checking, reasoning verification, compliance enforcement). A controlled experiment (1,800 runs across five industries and three LLMs: Claude Sonnet 4, Qwen 2.5 72B, Gemma 4 26B) finds ontology-coupled agents significantly outperform ungrounded agents on Metric Accuracy (p < .001) and Role Consistency (p < .001) across all three models with large effect sizes (Kendall's W = .46-.64). Improvements are greatest where LLM parametric knowledge is weakest--particularly in Vietnam-localized domains, where ontology lift is 2x that of English domains. Contributions: (1) a formal three-layer enterprise ontology model; (2) a taxonomy of neurosymbolic coupling patterns; (3) ontology-constrained tool discovery via SQL-pushdown scoring; (4) a proposed framework for output-side ontological validation; (5) empirical evidence for the inverse parametric knowledge effect--ontological grounding value is inversely proportional to LLM training-data coverage of the domain; (6) cross-model replication establishing model-independence; (7) a production system serving 22 industry verticals with 650+ agents.
>
---
#### [replaced 008] FinAuditing: A Financial Taxonomy-Structured Multi-Document Benchmark for Evaluating LLMs
- **分类: cs.CL; cs.CE; cs.IR**

- **简介: 该论文提出FinAuditing，一个结构化财务审计基准，用于评估大语言模型在财务语义匹配、关系提取和数学推理任务中的表现。**

- **链接: [https://arxiv.org/pdf/2510.08886](https://arxiv.org/pdf/2510.08886)**

> **作者:** Yan Wang; Keyi Wang; Shanshan Yang; Jaisal Patel; Jeff Zhao; Fengran Mo; Xueqing Peng; Lingfei Qian; Yankai Chen; Víctor Gutiérrez-Basulto; Jimin Huang; Guojun Xiong; Xiao-Yang Liu; Xue Liu; Jian-Yun Nie
>
> **备注:** Accepted by SIGIR 2026 Resource Track. Pre-camera-ready version
>
> **摘要:** Going beyond simple text processing, financial auditing requires detecting semantic, structural, and numerical inconsistencies across large-scale disclosures. As financial reports are filed in XBRL, a structured XML format governed by accounting standards, auditing becomes a structured information extraction and reasoning problem involving concept alignment, taxonomy-defined relations, and cross-document consistency. Although large language models (LLMs) show promise on isolated financial tasks, their capability in professional-grade auditing remains unclear. We introduce FinAuditing, a taxonomy-aligned, structure-aware benchmark built from real XBRL filings. It contains 1,102 annotated instances averaging over 33k tokens and defines three tasks: Financial Semantic Matching (FinSM), Financial Relationship Extraction (FinRE), and Financial Mathematical Reasoning (FinMR). Evaluations of 13 state-of-the-art LLMs reveal substantial gaps in concept retrieval, taxonomy-aware relation modeling, and consistent cross-document reasoning. These findings highlight the need for realistic, structure-aware benchmarks. We release the evaluation code at this https URL and the dataset at this https URL. The task currently serves as the official benchmark of an ongoing public evaluation contest at this https URL.
>
---
#### [replaced 009] Embodied Task Planning via Graph-Informed Action Generation with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于机器人任务规划任务，解决长距离规划中策略不连贯和状态幻觉问题。提出GiG框架，利用图结构增强记忆与规划能力。**

- **链接: [https://arxiv.org/pdf/2601.21841](https://arxiv.org/pdf/2601.21841)**

> **作者:** Xiang Li; Ning Yan; Masood Mortazavi
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** While Large Language Models (LLMs) have demonstrated strong zero-shot reasoning capabilities, their deployment as embodied agents still faces fundamental challenges in long-horizon planning. Unlike open-ended text generation, embodied agents must decompose high-level intents into actionable sub-goals while adhering to the constraints of a dynamic environment. Standard LLM planners frequently fail to maintain strategy coherence over extended horizons due to context window limitations or hallucinate state transitions that violate environment constraints. We propose GiG, a planning framework that structures embodied agents' memory using a Graph-in-Graph architecture. Our approach employs a Graph Neural Network (GNN) to encode environmental states into embeddings, organizing these embeddings into action-connected execution trace graphs within an experience memory bank. GiG enables retrieval of structurally-similar priors, allowing agents to ground current decisions in relevant past structural patterns. Furthermore, we introduce a bounded lookahead module that leverages symbolic transition logic to enhance the agent's planning capabilities through grounded action projections. We evaluate our framework on three embodied planning benchmarks-Robotouille Synchronous, Robotouille Asynchronous, and ALFWorld. Our method outperforms state-of-the-art baselines, achieving Pass@1 performance gains of up to 22% on Robotouille Synchronous, 37% on Asynchronous, and 15% on ALFWorld while maintaining comparable or lower computational cost.
>
---
#### [replaced 010] Embracing Anisotropy: Turning Massive Activations into Interpretable Control Knobs for Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究如何利用大语言模型中的高幅特征维度，作为可解释的控制变量。任务是提升模型在特定领域的适应性和可控性，通过识别关键维度并进行定向调整，实现更有效的领域迁移与攻击防御。**

- **链接: [https://arxiv.org/pdf/2603.00029](https://arxiv.org/pdf/2603.00029)**

> **作者:** Youngji Roh; Hyunjin Cho; Jaehyung Kim
>
> **备注:** ACL 2026 (main, long, oral), 27 pages
>
> **摘要:** Large Language Models (LLMs) exhibit highly anisotropic internal representations, often characterized by massive activations, a phenomenon where a small subset of feature dimensions possesses magnitudes significantly larger than the rest. While prior works view these extreme dimensions primarily as artifacts to be managed, we propose a distinct perspective: these dimensions serve as intrinsic interpretable functional units arising from domain specialization. Specifically, we propose a simple magnitude-based criterion to identify Domain-Critical Dimensions in a training-free manner. Our analyses reveal that such dimensions behave as interpretable semantic detectors for symbolic/quantitative patterns or domain-specific terms. In addition, we introduce Critical Dimension Steering, which applies activation steering exclusively to the identified dimensions. Empirical results show that this approach outperforms conventional whole-dimension steering in domain adaptation and jailbreaking scenarios.
>
---
#### [replaced 011] RLBFF: Binary Flexible Feedback to bridge between Human Feedback & Verifiable Rewards
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大模型对齐任务，旨在解决RLHF可解释性差和RLVR范围窄的问题。提出RLBFF，结合人类反馈与规则验证，提升奖励模型性能。**

- **链接: [https://arxiv.org/pdf/2509.21319](https://arxiv.org/pdf/2509.21319)**

> **作者:** Zhilin Wang; Jiaqi Zeng; Olivier Delalleau; Ellie Evans; Daniel Egert; Hoo-Chang Shin; Felipe Soares; Yi Dong; Oleksii Kuchaiev
>
> **备注:** Published at ICLR 2026, 21 pages
>
> **摘要:** Reinforcement Learning with Human Feedback (RLHF) and Reinforcement Learning with Verifiable Rewards (RLVR) are the main RL paradigms used in LLM post-training, each offering distinct advantages. However, RLHF struggles with interpretability and reward hacking because it relies on human judgments that usually lack explicit criteria, whereas RLVR is limited in scope by its focus on correctness-based verifiers. We propose Reinforcement Learning with Binary Flexible Feedback (RLBFF), which combines the versatility of human-driven preferences with the precision of rule-based verification, enabling reward models to capture nuanced aspects of response quality beyond mere correctness. RLBFF extracts principles that can be answered in a binary fashion (e.g. accuracy of information: yes, or code readability: no) from natural language feedback. Such principles can then be used to ground Reward Model training as an entailment task (response satisfies or does not satisfy an arbitrary principle). We show that Reward Models trained in this manner can outperform Bradley-Terry models when matched for data and achieve top performance on RM-Bench (86.2%) and JudgeBench (81.4%, #1 on leaderboard as of September 24, 2025). Additionally, users can specify principles of interest at inference time to customize the focus of our reward models, in contrast to Bradley-Terry models. Finally, we present a fully open source recipe (including data) to align Qwen3-32B using RLBFF and our Reward Model, to match or exceed the performance of o3-mini and DeepSeek R1 on general alignment benchmarks of MT-Bench, WildBench, and Arena Hard v2 (at <5% of the inference cost). Models: this https URL
>
---
#### [replaced 012] Can RL Teach Long-Horizon Reasoning to LLMs? Expressiveness Is Key
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究强化学习提升大语言模型长距离推理能力的问题。通过构建可控制难度的逻辑框架，发现训练计算量与推理深度呈幂律关系，且逻辑表达能力影响模型表现。**

- **链接: [https://arxiv.org/pdf/2605.06638](https://arxiv.org/pdf/2605.06638)**

> **作者:** Tianle Wang; Zhaoyang Wang; Guangchen Lan; Xinpeng Wei; Sipeng Zhang; Guanwen Qiu; Abulhair Saparov
>
> **摘要:** Reinforcement learning (RL) has been applied to improve large language model (LLM) reasoning, yet the systematic study of how training scales with task difficulty has been hampered by the lack of controlled, scalable environments. Observed LLM shortcomings in long-horizon reasoning have raised the prospect that they are fundamental to the autoregressive transformer architecture. To address this, we introduce ScaleLogic, a synthetic logical reasoning framework that offers independent control over two axes of difficulty: the depth of the required proof planning (i.e., the horizon) and the expressiveness of the underlying logic. Our proposed framework supports a wide range of logics: from simple implication-only logic ("if-then") towards more expressive first-order reasoning with conjunction ("and"), disjunction ("or"), negation ("not"), and universal quantification ("for all"). Using this framework, we show that the RL training compute $T$ follows a power law with respect to reasoning depth $D$ ($T \propto D^{\gamma}$, $R^{2} > 0.99$), and that the scaling exponent $\gamma$ increases monotonically with logical expressiveness, from $1.04$ to $2.60$. On downstream mathematics and general reasoning benchmarks, more expressive training settings yield both larger performance gains (up to $+10.66$ points) and more compute-efficient transfer compared to less expressive settings, demonstrating that what a model is trained on, not just how much it is trained, shapes downstream transfer. We further show that the power-law relationship holds across multiple RL methods, and curriculum-based training substantially improves scaling efficiency. More broadly, our results demonstrate that LLM shortcomings in long-horizon reasoning are not fundamental to the underlying architecture, and can be addressed by improved training methodology and data.
>
---
#### [replaced 013] Scaling Laws for Code: A More Data-Hungry Regime
- **分类: cs.CL**

- **简介: 该论文属于代码大模型研究，解决代码与自然语言在扩展规律上的差异问题。通过实验分析代码模型的扩展规律，发现代码需要更高的数据量。**

- **链接: [https://arxiv.org/pdf/2510.08702](https://arxiv.org/pdf/2510.08702)**

> **作者:** Xianzhen Luo; Wenzhen Zheng; Qingfu Zhu; Rongyi Zhang; Houyi Li; Siming Huang; YuanTao Fan; Wanxiang Che
>
> **备注:** Accepted by ACL2026
>
> **摘要:** Code Large Language Models (LLMs) are revolutionizing software engineering. However, scaling laws that guide the efficient training are predominantly analyzed on Natural Language (NL). Given the fundamental differences like strict syntax between code and NL, it is unclear whether these laws are directly applicable to code. To address this gap, we conduct the first large-scale empirical study of scaling laws for code, comprising 117 experimental runs with model sizes from 0.2B to 3.8B and training tokens from 2B to 128B. We fit the Chinchilla law and the Farsser law. First, the results show that the more expressive Farseer law offers greater accuracy. Second, the analysis reveals that Code LLMs scale effectively with model size. Crucially, code represents a more data-hungry regime, requiring a substantially higher data-to-parameter ratio than NL. Finally, two additional sets of experiments on code-NL mixtures show that NL benefits resource-constrained scenarios, but becomes a detriment at higher compute budgets.
>
---
#### [replaced 014] CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于事实问答任务，解决系统正确答案错误的问题。通过生成条件反例证据，对初步答案进行验证和修正，提升准确性。**

- **链接: [https://arxiv.org/pdf/2603.16091](https://arxiv.org/pdf/2603.16091)**

> **作者:** Tianyi Huang; Ying Kai Deng
>
> **备注:** Accepted at the 4th Workshop on Towards Knowledgeable Foundation Models at ACL 2026
>
> **摘要:** In factual question answering, many errors are not failures of access but failures of commitment: the system retrieves relevant evidence, yet still settles on the wrong answer. We present CounterRefine, a lightweight repair layer for short-form RAG that treats the first answer as a hypothesis to test. Given a draft, CounterRefine issues answer-conditioned expansion queries to retrieve candidate-specific evidence, then applies a constrained KEEP or REVISE refinement step whose proposed revisions are accepted only after deterministic validation. The design is intentionally narrow: it adds one evidence-gathering pass and one guarded refinement call rather than replacing the retriever or building a broad agentic system. On the full SimpleQA benchmark, CounterRefine improves a matched one-pass RAG baseline by up to 5.8 correct-rate points; in the full Claude trace, it changes only 5.6% of outputs, with 180 beneficial outcome changes and 8 harmful ones. These findings suggest a simple but important direction for knowledgeable foundation models: beyond accessing evidence, they should also be able to use that evidence to reconsider and, when necessary, repair their own answers.
>
---
#### [replaced 015] Old Habits Die Hard: How Conversational History Geometrically Traps LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM对话历史对其后续表现的影响，属于模型行为分析任务。通过概率和几何方法分析对话偏差，揭示其导致的轨迹限制问题。**

- **链接: [https://arxiv.org/pdf/2603.03308](https://arxiv.org/pdf/2603.03308)**

> **作者:** Adi Simhi; Fazl Barez; Martin Tutek; Yonatan Belinkov; Shay B. Cohen
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** How does the conversational past of large language models (LLMs) influence their future performance? Recent work suggests that LLMs are affected by their conversational history in unexpected ways. For instance, hallucinations in prior interactions may influence subsequent model responses. In this work, we introduce History-Echoes, a framework that investigates how conversational history biases subsequent generations. The framework explores this bias from two perspectives: probabilistically, we model conversations as Markov chains to quantify state consistency; geometrically, we measure the consistency of consecutive hidden representations. Across three model families and six datasets spanning diverse phenomena, our analysis reveals a strong correlation between the two perspectives. By bridging these perspectives, we demonstrate that behavioral persistence manifests as a geometric trap, where gaps in the latent space confine the model's trajectory. Code available at this https URL.
>
---
#### [replaced 016] Reverse-Engineering Model Editing on Language Models
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于模型编辑安全任务，旨在解决参数更新泄露编辑数据的问题。提出KSTER攻击方法，通过分析参数更新恢复编辑内容，并设计子空间伪装作为防御手段。**

- **链接: [https://arxiv.org/pdf/2602.10134](https://arxiv.org/pdf/2602.10134)**

> **作者:** Zhiyu Sun; Minrui Luo; Yu Wang; Zhili Chen; Tianxing He
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Large language models (LLMs) are pretrained on corpora containing trillions of tokens and, therefore, inevitably memorize sensitive information. Locate-then-edit methods, as a mainstream paradigm of model editing, offer a promising solution by modifying model parameters without retraining. However, in this work, we reveal a critical vulnerability of this paradigm: the parameter updates inadvertently serve as a side channel, enabling attackers to recover the edited data. We propose a two-stage reverse-engineering attack named \textit{KSTER} (\textbf{K}ey\textbf{S}paceRecons\textbf{T}ruction-then-\textbf{E}ntropy\textbf{R}eduction) that leverages the low-rank structure of these updates. First, we theoretically show that the row space of the update matrix encodes a ``fingerprint" of the edited subjects, enabling accurate subject recovery via spectral analysis. Second, we introduce an entropy-based prompt recovery attack that reconstructs the semantic context of the edit. Extensive experiments on multiple LLMs demonstrate that our attacks can recover edited data with high success rates. Furthermore, we propose \textit{subspace camouflage}, a defense strategy that obfuscates the update fingerprint with semantic decoys. This approach effectively mitigates reconstruction risks without compromising editing utility. Our code is available at this https URL.
>
---
#### [replaced 017] Tongyi DeepResearch Technical Report
- **分类: cs.CL; cs.AI; cs.IR; cs.LG; cs.MA**

- **简介: 该论文介绍Tongyi DeepResearch，一个用于长周期深度信息检索任务的代理型大语言模型。旨在解决复杂任务中的自主推理与信息搜索问题，通过端到端训练框架提升性能。**

- **链接: [https://arxiv.org/pdf/2510.24701](https://arxiv.org/pdf/2510.24701)**

> **作者:** Tongyi DeepResearch Team; Baixuan Li; Bo Zhang; Dingchu Zhang; Fei Huang; Guangyu Li; Guoxin Chen; Huifeng Yin; Jialong Wu; Jingren Zhou; Kuan Li; Liangcai Su; Litu Ou; Liwen Zhang; Pengjun Xie; Rui Ye; Wenbiao Yin; Xinmiao Yu; Xinyu Wang; Xixi Wu; Xuanzhong Chen; Yida Zhao; Zhen Zhang; Zhengwei Tao; Zhongwang Zhang; Zile Qiao; Chenxi Wang; Donglei Yu; Gang Fu; Haiyang Shen; Jiayin Yang; Jun Lin; Junkai Zhang; Kui Zeng; Li Yang; Hailong Yin; Maojia Song; Ming Yan; Minpeng Liao; Peng Xia; Qian Xiao; Rui Min; Ruixue Ding; Runnan Fang; Shaowei Chen; Shen Huang; Shihang Wang; Shihao Cai; Weizhou Shen; Xiaobin Wang; Xin Guan; Xinyu Geng; Yingcheng Shi; Yuning Wu; Zhuo Chen; Zijian Li; Yong Jiang
>
> **备注:** this https URL
>
> **摘要:** We present Tongyi DeepResearch, an agentic large language model, which is specifically designed for long-horizon, deep information-seeking research tasks. To incentivize autonomous deep research agency, Tongyi DeepResearch is developed through an end-to-end training framework that combines agentic mid-training and agentic post-training, enabling scalable reasoning and information seeking across complex tasks. We design a highly scalable data synthesis pipeline that is fully automatic, without relying on costly human annotation, and empowers all training stages. By constructing customized environments for each stage, our system enables stable and consistent interactions throughout. Tongyi DeepResearch, featuring 30.5 billion total parameters, with only 3.3 billion activated per token, achieves state-of-the-art performance across a range of agentic deep research benchmarks, including Humanity's Last Exam, BrowseComp, BrowseComp-ZH, WebWalkerQA, xbench-DeepSearch, FRAMES and xbench-DeepSearch-2510. We open-source the model, framework, and complete solutions to empower the community.
>
---
#### [replaced 018] Provable Knowledge Acquisition and Extraction in One-Layer Transformers
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究Transformer模型在微调后如何提取预训练阶段存储的事实知识。任务为知识获取与提取，解决模型微调后知识不可用的问题。工作包括理论分析与实验验证。**

- **链接: [https://arxiv.org/pdf/2508.00901](https://arxiv.org/pdf/2508.00901)**

> **作者:** Ruichen Xu; Kexin Chen
>
> **摘要:** Large language models may encounter factual knowledge during pre-training yet fail to reliably use that knowledge after fine-tuning. Despite growing empirical evidence that MLP layers store factual associations and fine-tuning affects factual recall, the training-dynamics mechanisms linking next-token pre-training, knowledge storage, and post-fine-tuning extraction remain poorly understood. We study this problem in a stylized one-layer transformer with self-attention and MLP modules, trained by next-token prediction and subsequently fine-tuned on question-answering data. Under suitable regularity conditions, we first prove that the model reaches near-optimal pre-training loss while learning structured attention patterns and relation-specific feature directions, giving a mechanism for factual knowledge acquisition. We then show that fine-tuning can turn the Q&A prompt format into a trigger for pre-trained relation features, enabling the model to extract facts that are not revisited during fine-tuning. Our analysis yields a relation-covering characterization of knowledge extraction: fine-tuning need not revisit every stored subject-answer pair, but it must cover enough latent relation-template directions through which facts were encoded during pre-training. Consequently, extraction improves with pre-training multiplicity and fine-tuning coverage, but becomes harder as the relation-template universe grows. Conversely, insufficient coverage leads to a failure regime in which facts may be stored but remain inaccessible, providing a stylized mechanism for hallucination. The theory applies to both full and low-rank fine-tuning, offering insight into why low-rank adaptation can recover pre-trained factual knowledge when relation coverage is sufficient. Experiments on synthetic data and PopQA-based GPT-2/Llama models support the predicted trends.
>
---
#### [replaced 019] LaPA$^2$: Length-Aware Prefix and Prompt Attention Augmentation for Long-Form Controllable Text Generation
- **分类: cs.CL**

- **简介: 该论文属于可控文本生成任务，针对长序列中控制信号效果减弱的问题，提出LaPA²框架，通过注意力增强保持控制效果。**

- **链接: [https://arxiv.org/pdf/2508.04047](https://arxiv.org/pdf/2508.04047)**

> **作者:** Jiabing Yang; Yixiang Chen; Zichen Wen; Chenhang Cui; Peiyan Li; Yuan Xu; Bowen Fang; Tao Yu; Ruikang Lin; Yan Huang; Liang Wang
>
> **备注:** Substantially revised version, renamed from "DTPA: Dynamic Token-level Prefix Augmentation for Controllable Text Generation"
>
> **摘要:** Prefix-based methods have emerged as a promising paradigm for Controllable Text Generation (CTG) due to their parameter efficiency. However, while effective in short sequences, their controllability tends to diminish as the generated sequence grows. In this paper, we identify Attention Dilution as a key factor behind this phenomenon: as the sequence length increases, the attention allocated to the control signal naturally decays due to the softmax mechanism, leading to a "fading" control effect. To address this, we propose LaPA$^2$ (Length-aware Prefix and Prompt Attention Augmentation), a training-free and model-agnostic framework designed to sustain robust control in long contexts. Specifically, LaPA$^2$ employs Length-Aware Logarithmic Scaling to dynamically amplify prefix attention weights, mathematically counteracting the dilution effect, while an optional Contextual Anchor Reinforcement applies synchronized augmentation to prompt tokens, preserving semantic coherence when strong attribute control risks overshadowing the original prompt. LaPA$^2$ is versatile, supporting both soft prefixes (continuous embeddings) and hard prefixes (discrete instructions). Experiments on multiple CTG tasks demonstrate that LaPA$^2$ consistently improves the performance of various prefix-based methods in long-form settings, leading to superior attribute controllability while preserving content relevance and fluency. Our code and data are publicly available at this https URL.
>
---
#### [replaced 020] Self-Play Only Evolves When Self-Synthetic Pipeline Ensures Learnable Information Gain
- **分类: cs.LG; cs.AI; cs.CL; cs.IT**

- **简介: 该论文研究自进化系统在语言模型中的应用，解决自玩机制易停滞的问题。通过设计三角色系统，提升可学习信息量，实现持续进化。**

- **链接: [https://arxiv.org/pdf/2603.02218](https://arxiv.org/pdf/2603.02218)**

> **作者:** Wei Liu; Siya Qi; Yali Du; Yulan He
>
> **备注:** 10 pages, 6 figures, 7 formulas, accepted by ICML 2026 position paper track
>
> **摘要:** Large language models (LLMs) make it plausible to build systems that improve through self-evolving loops, but many existing proposals are better understood as self-play and often plateau quickly. A central failure mode is that the loop synthesises more data without increasing learnable information for the next iteration. Through experiments on a self-play coding task, we reveal that sustainable self-evolution requires a self-synthesised data pipeline with learnable information that increases across iterations. We identify triadic roles that self-evolving LLMs play: the Proposer, which generates tasks; the Solver, which attempts solutions; and the Verifier, which provides training signals, and we identify three system designs that jointly target learnable information gain from this triadic roles perspective. Asymmetric co-evolution closes a weak-to-strong-to-weak loop across roles. Capacity growth expands parameter and inference-time budgets to match rising learnable information. Proactive information seeking introduces external context and new task sources that prevent saturation. Together, these modules provide a measurable, system-level path from brittle self-play dynamics to sustained self-evolution.
>
---
#### [replaced 021] Responsible Federated LLMs via Safety Filtering and Constitutional AI
- **分类: cs.CL; cs.DC; cs.MA**

- **简介: 该论文属于安全AI任务，旨在解决FedLLM中因客户端数据有害导致模型不安全的问题。通过引入安全过滤和宪法AI技术提升模型安全性。**

- **链接: [https://arxiv.org/pdf/2502.16691](https://arxiv.org/pdf/2502.16691)**

> **作者:** Eunchung Noh; Jeonghun Baek
>
> **备注:** Accepted at the 6th Workshop on Trustworthy NLP (TrustNLP), ACL 2026
>
> **摘要:** Recent research has increasingly focused on training large language models (LLMs) using federated learning, known as FedLLM. However, responsible AI (RAI), which aims to ensure safe and trustworthy responses, remains underexplored in this context. In FedLLM, client-side training data may contain harmful content, resulting in unsafe LLMs that can generate inappropriate responses. Aggregating such models into a global model and redistributing it to clients risks the widespread deployment of unsafe LLMs. To address this, we incorporate two well-established RAI techniques into FedLLM: safety filtering and constitutional AI. Our experiments show that these methods significantly improve LLM safety, achieving over 20% improvement on AdvBench.
>
---
#### [replaced 022] Dynamic Skill Lifecycle Management for Agentic Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，解决外部技能管理问题。提出SLIM框架，动态优化技能生命周期，提升代理性能。**

- **链接: [https://arxiv.org/pdf/2605.10923](https://arxiv.org/pdf/2605.10923)**

> **作者:** Junhao Shen; Teng Zhang; Xiaoyan Zhao; Hong Cheng
>
> **备注:** Implementation code is available at this https URL
>
> **摘要:** Large language model agents increasingly rely on external skills to solve complex tasks, where skills act as modular units that extend their capabilities beyond what parametric memory alone supports. Existing methods assume external skills either accumulate as persistent guidance or internalized into the policy, eventually leading to zero-skill inference. We argue this assumption is overly restrictive, since with limited parametric capacity and uneven marginal contribution across skills, the optimal active skill set is non-monotonic, task- and stage-dependent. In this work, we propose SLIM, a framework of dynamic Skill LIfecycle Management for agentic reinforcement learning (RL), which treats the active external skill set as a dynamic optimization variable jointly updated with policy learning. Specifically, SLIM estimates each active skill's marginal external contribution through leave-one-skill-out validation, then applies three lifecycle operations: retaining high-value skills, retiring skills whose contribution becomes negligible after sufficient exposure, and expanding the skill bank when persistent failures reveal missing capability coverage. Experiments show that SLIM outperforms the best baselines by an average of 7.1% points across ALFWorld and SearchQA. Results further indicate that policy learning and external skill retention are not mutually exclusive: some skills are absorbed into the policy, while others continue to provide external value, supporting SLIM as a more general paradigm for skill-based agentic RL.
>
---
#### [replaced 023] Leveraging Speech to Identify Signatures of Insight and Transfer in Problem Solving
- **分类: cs.CL**

- **简介: 该论文研究问题解决中的直觉与迁移，通过分析参与者说话内容，探讨相同与不同问题解决策略的影响。任务属于认知科学与自然语言处理交叉领域。**

- **链接: [https://arxiv.org/pdf/2605.12970](https://arxiv.org/pdf/2605.12970)**

> **作者:** Linas Nasvytis; Judith E. Fan
>
> **摘要:** Many problems seem to require a flash of insight to solve. What form do these sudden insights take, and what impact do they have on how people approach similar problems in the future? In this work, we prompted participants (N = 189) to think aloud as they attempted to solve a sequence of five "matchstick-arithmetic" problems. These problems either all relied on the same kind of non-obvious solution (Same group) or a different kind each time (Different group). Our first observation was that Same participants improved more rapidly than Different participants. We then leveraged techniques from natural language processing to analyze participants' speech, and found that this accelerated improvement for Same participants was accompanied by changes in both how much they spoke and what they said. In particular, they were more likely to spontaneously label the kind of problem they were working on. Taken together, these findings suggest that a hallmark of transferable insights is their accessibility for verbal report, even if the underlying precursors of insight remain difficult to articulate.
>
---
#### [replaced 024] SEDD: Scalable and Efficient Dataset Deduplication with GPUs
- **分类: cs.CL**

- **简介: 该论文属于数据去重任务，解决传统方法在GPU加速下的通信瓶颈和资源利用率低的问题。提出SEDD框架，优化哈希函数和GPU计算，提升去重效率。**

- **链接: [https://arxiv.org/pdf/2501.01046](https://arxiv.org/pdf/2501.01046)**

> **作者:** Youngjun Son; Chaewon Kim; Jaejin Lee
>
> **备注:** 13 pages, 7 figures
>
> **摘要:** Dataset deduplication is widely recognized as a crucial preprocessing step that enhances data quality and improves the performance of large language models. A commonly used method for this process is the MinHash Locality-Sensitive Hashing (LSH) algorithm. Recently, GPU-accelerated frameworks such as NVIDIA NeMo Curator have been introduced to handle large-scale corpora; however, they remain suboptimal due to high communication overhead from physical data shuffling and underutilization of GPU resources. In this paper, we propose SEDD, a high-performance GPU-accelerated deduplication framework optimized for distributed cluster environments. SEDD introduces a computationally efficient, partially reusable hash function, alongside highly optimized GPU kernels and a hardware-aware automatic parameter selection mechanism. By replacing traditional data shuffling with a streaming-based approach, SEDD significantly mitigates communication bottlenecks. Our framework outperforms the CPU-based deduplication tool in SlimPajama by up to 158$\times$ and the GPU-based tool in NVIDIA NeMo Curator by up to 7.8$\times$ when processing 30 million documents on a node with four GPUs. Notably, SEDD dramatically accelerates the previously time-consuming MinHash signature generation phase, achieving speedups of up to 375$\times$ over the CPU baseline. Despite these gains in efficiency, SEDD maintains high deduplication fidelity, with duplicate document sets achieving Jaccard similarities of over 0.95 compared to those identified by the standard MinHash algorithm. In large-scale experiments, the deduplication of 1.2 trillion tokens is completed in just 3 hours on an 8-node 32-GPU V100 cluster. The related code is publicly available on GitHub (this https URL).
>
---
#### [replaced 025] Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于强化学习任务，解决LLM代理在不确定环境中的成本-不确定性权衡问题。提出Calibrate-Then-Act框架，使代理更优决策。**

- **链接: [https://arxiv.org/pdf/2602.16699](https://arxiv.org/pdf/2602.16699)**

> **作者:** Wenxuan Ding; Nicholas Tomlin; Greg Durrett
>
> **摘要:** LLM agents are deployed in environments where they must interact to acquire information. In these scenarios, the agent must reason about inherent cost-uncertainty tradeoffs in how to act, such as when to stop exploring and commit to an answer. For instance, on a programming task, an agent might run the code it generates, or it might generate tests for that code snippet; the cost of writing and running a test is nonzero, but typically lower than the cost of running buggy code. In this work, we show that we can induce LLM agents to explicitly reason about balancing these cost-uncertainty tradeoffs, then act more optimally in their environments. We formalize multiple tasks, including retrieval-augmented QA and a file reading coding task, as sequential decision-making problems under uncertainty. Each problem has latent environment state that impacts the agent's performance. We introduce a framework called Calibrate-Then-Act (CTA), where we pass the agent an inferred prior about this environment state to enable it to act more optimally. This information qualitatively changes agent behavior, and adds environment sensitivity to the agent which is not learned via standard RL training. Our results on a synthetic task, QA, and file reading show that making cost-benefit tradeoffs explicit with CTA helps agents discover more optimal decision-making strategies.
>
---
#### [replaced 026] Med-V1: Small Language Models for Zero-shot and Scalable Biomedical Evidence Attribution
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于 biomedical evidence attribution 任务，旨在解决LLM生成内容中的幻觉问题。提出Med-V1小模型，高效准确地进行证据归属与验证。**

- **链接: [https://arxiv.org/pdf/2603.05308](https://arxiv.org/pdf/2603.05308)**

> **作者:** Qiao Jin; Yin Fang; Lauren He; Yifan Yang; Guangzhi Xiong; Zhizheng Wang; Nicholas Wan; Joey Chan; Donald C. Comeau; Robert Leaman; Charalampos S. Floudas; Aidong Zhang; Michael F. Chiang; Yifan Peng; Zhiyong Lu
>
> **摘要:** Assessing whether an article supports an assertion is essential for hallucination detection and claim verification. While large language models (LLMs) have the potential to automate this task, achieving strong performance requires frontier models such as GPT-5 that are prohibitively expensive to deploy at scale. To efficiently perform biomedical evidence attribution, we present Med-V1, a family of small language models with only three billion parameters. Trained on high-quality synthetic data newly developed in this study, Med-V1 substantially outperforms (+27.0% to +71.3%) its base models on five biomedical benchmarks unified into a verification format. Despite its smaller size, Med-V1 performs comparably to frontier LLMs such as GPT-5, along with high-quality explanations for its predictions. We use Med-V1 to conduct a first-of-its-kind use case study that quantifies hallucinations in LLM-generated answers under different citation instructions. Results show that the format instruction strongly affects citation validity and hallucination, with GPT-5 generating more claims but exhibiting hallucination rates similar to GPT-4o. Additionally, we present a second use case showing that Med-V1 can automatically identify high-stakes evidence misattributions in clinical practice guidelines, revealing potentially negative public health impacts that are otherwise challenging to identify at scale. Overall, Med-V1 provides an efficient and accurate lightweight alternative to frontier LLMs for practical and real-world applications in biomedical evidence attribution and verification tasks. Med-V1 is available at this https URL.
>
---
#### [replaced 027] STEM: Structure-Tracing Evidence Mining for Knowledge Graphs-Driven Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于知识图谱问答任务，旨在解决多跳推理中的结构异构和全局视角不足问题。提出STEM框架，通过结构引导的图搜索提升推理准确性和证据完整性。**

- **链接: [https://arxiv.org/pdf/2604.22282](https://arxiv.org/pdf/2604.22282)**

> **作者:** Peng Yu; En Xu; Bin Chen; Haibiao Chen; Yinfei Xu
>
> **备注:** 34 pages, 16 figures, accepted to ACL 2026 (Main Conference, Oral Presentation)
>
> **摘要:** Knowledge Graph-based Question Answering (KGQA) plays a pivotal role in complex reasoning tasks but remains constrained by two persistent challenges: the structural heterogeneity of Knowledge Graphs(KGs) often leads to semantic mismatch during retrieval, while existing reasoning path retrieval methods lack a global structural perspective. To address these issues, we propose Structure-Tracing Evidence Mining (STEM), a novel framework that reframes multi-hop reasoning as a schema-guided graph search task. First, we design a Semantic-to-Structural Projection pipeline that leverages KG structural priors to decompose queries into atomic relational assertions and construct an adaptive query schema graph. Subsequently, we execute globally-aware node anchoring and subgraph retrieval to obtain the final evidence reasoning graph from KG. To more effectively integrate global structural information during the graph construction process, we design a Triple-Dependent GNN (Triple-GNN) to generate a Global Guidance Subgraph (Guidance Graph) that guides the construction. STEM significantly improves both the accuracy and evidence completeness of multi-hop reasoning graph retrieval, and achieves State-of-the-Art performance on multiple multi-hop benchmarks.
>
---
#### [replaced 028] LLM Agents Are the Antidote to Walled Gardens
- **分类: cs.LG; cs.CL; cs.CY; cs.SI**

- **简介: 论文探讨了LLM代理如何打破封闭平台，实现数据无缝交换，解决互操作性难题。属于AI与互联网治理任务，旨在促进数据自由流动和市场竞争。**

- **链接: [https://arxiv.org/pdf/2506.23978](https://arxiv.org/pdf/2506.23978)**

> **作者:** Samuele Marro; Philip Torr
>
> **备注:** Published at the ICML 2026 Position Paper track
>
> **摘要:** While the Internet's core infrastructure was designed to be open and universal, today's application layer is dominated by closed, proprietary platforms. Open and interoperable APIs require significant investment, and market leaders have little incentive to enable data exchange that could erode their user lock-in. We argue that LLM-based agents fundamentally disrupt this status quo. Agents can automatically translate between data formats and interact with interfaces designed for humans: this makes interoperability dramatically cheaper and effectively unavoidable. We name this shift universal interoperability: the ability for any two digital services to exchange data seamlessly using AI-mediated adapters. Universal interoperability undermines monopolistic behaviours and promotes data portability. However, it can also lead to new security risks, technical debt, and legal frictions. Our position is that the ML community should embrace this development while building the appropriate frameworks to mitigate the downsides. By acting now, we can harness AI to restore user freedom and competitive markets without sacrificing security.
>
---
#### [replaced 029] Rethinking Table Pruning in TableQA: From Sequential Revisions to Gold Trajectory-Supervised Parallel Search
- **分类: cs.CL**

- **简介: 该论文属于表格问答任务，旨在解决表 pruning 中因依赖不可靠信号导致的关键数据丢失问题。提出 TabTrim 框架，通过黄金轨迹监督并行搜索提升表 pruning 效果。**

- **链接: [https://arxiv.org/pdf/2601.03851](https://arxiv.org/pdf/2601.03851)**

> **作者:** Yu Guo; Shenghao Ye; Shuangwu Chen; Zijian Wen; Tao Zhang; Qirui Bai; Dong Jin; Yunpeng Hou; Huasen He; Jian Yang; Xiaobin Tan
>
> **备注:** 17 pages, 5 figures, accepted to ACL 2026 Oral
>
> **摘要:** Table Question Answering (TableQA) benefits significantly from table pruning, which extracts compact sub-tables by eliminating redundant cells to streamline downstream reasoning. However, existing pruning methods typically rely on sequential revisions driven by unreliable critique signals, often failing to detect the loss of answer-critical data. To address this limitation, we propose TabTrim, a novel table pruning framework which transforms table pruning from sequential revisions to gold trajectory-supervised parallel search. TabTrim derives a gold pruning trajectory using the intermediate sub-tables in the execution process of gold SQL queries, and trains a pruner and a verifier to make the step-wise pruning result align with the gold pruning trajectory. During inference, TabTrim performs parallel search to explore multiple candidate pruning trajectories and identify the optimal sub-table. Extensive experiments demonstrate that TabTrim achieves state-of-the-art performance across diverse tabular reasoning tasks: TabTrim-8B reaches 73.5% average accuracy, outperforming the strongest baseline by 3.2%, including 79.4% on WikiTQ and 61.2% on TableBench.
>
---
#### [replaced 030] Probing Multimodal Large Language Models on Cognitive Biases in Chinese Short-Video Misinformation
- **分类: cs.CL**

- **简介: 该论文属于信息检测任务，旨在评估多模态大模型对短视频中认知偏见引发的虚假信息的识别能力。研究构建了高质量数据集并测试多个模型表现。**

- **链接: [https://arxiv.org/pdf/2601.06600](https://arxiv.org/pdf/2601.06600)**

> **作者:** Jen-tse Huang; Chang Chen; Shiyang Lai; Wenxuan Wang; Michelle R. Kaufman; Mark Dredze
>
> **备注:** Accepted to ACL 2026 (Findings)
>
> **摘要:** Short-video platforms have become major channels for misinformation, where deceptive claims frequently leverage visual experiments and social cues. While Multimodal Large Language Models (MLLMs) have demonstrated impressive reasoning capabilities, their robustness against misinformation entangled with cognitive biases remains under-explored. In this paper, we introduce a comprehensive evaluation framework using a high-quality, manually annotated dataset of 200 short videos spanning four health domains. This dataset provides fine-grained annotations for three deceptive patterns-experimental errors, logical fallacies, and fabricated claims-each verified by evidence such as national standards and academic literature. We evaluate eight frontier MLLMs across five modality settings. Experimental results demonstrate that Gemini-2.5-Pro achieves the highest performance in the multimodal setting with a belief score of 71.5/100, while o3 performs the worst at 35.2. Furthermore, we investigate social cues that induce false beliefs in videos and find that models are susceptible to biases like authoritative channel IDs.
>
---
#### [replaced 031] Merlin's Whisper: Enabling Efficient Reasoning in Large Language Models via Black-box Persuasive Prompting
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于提升大语言模型推理效率的任务，旨在减少推理过程中的计算和延迟开销。通过黑盒说服提示技术，生成简洁响应，保持准确率。**

- **链接: [https://arxiv.org/pdf/2510.10528](https://arxiv.org/pdf/2510.10528)**

> **作者:** Heming Xia; Cunxiao Du; Rui Li; Chak Tou Leong; Yongqi Li; Wenjie Li
>
> **备注:** ACL 2026 (Long Paper), camera-ready version
>
> **摘要:** Large reasoning models (LRMs) have demonstrated remarkable proficiency in tackling complex tasks through step-by-step thinking. However, this lengthy reasoning process incurs substantial computational and latency overheads, hindering the practical deployment of LRMs. This work presents a new approach to mitigating overthinking in LRMs via black-box persuasive prompting. By treating LRMs as black-box communicators, we investigate how to persuade them to generate concise responses without compromising accuracy. We introduce Whisper, an iterative refinement framework that generates high-quality persuasive prompts from diverse perspectives. Experiments across multiple benchmarks demonstrate that Whisper consistently reduces token usage while preserving performance. Notably, Whisper achieves a 3x reduction in average response length on simple GSM8K questions for the Qwen3 model series and delivers an average ~40% token reduction across all benchmarks. For closed-source APIs, Whisper reduces token usage on MATH-500 by 46% for Claude-3.7 and 50% for Gemini-2.5. Further analysis reveals the broad applicability of Whisper across data domains, model scales, and families, underscoring the potential of black-box persuasive prompting as a practical strategy for enhancing LRM efficiency.
>
---
#### [replaced 032] Unlocking the Potential of Diffusion Language Models through Template Infilling
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，旨在解决扩散语言模型推理策略受限的问题。提出模板填充方法，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2510.13870](https://arxiv.org/pdf/2510.13870)**

> **作者:** Junhoo Lee; Seungyeon Kim; Nojun Kwak
>
> **备注:** ACL 2026 Main Conference - Long Paper, Oral Presentation
>
> **摘要:** Diffusion Language Models (DLMs) have emerged as a promising alternative to Autoregressive Language Models, yet their inference strategies remain limited to prefix-based prompting inherited from the autoregressive paradigm. In this paper, we propose Template Infilling (TI), a tailored conditioning methodology for DLMs. Unlike conventional prefix prompting, TI flexibly aligns structural anchors across the entire target response space, establishing a global blueprint before filling in the masked segments. We demonstrate the effectiveness of our approach on diverse benchmarks, including mathematical reasoning, code generation, and trip planning, achieving consistent improvements of 9.40% over the baseline. Furthermore, we observe that TI provides additional advantages in multi-token generation settings, enabling effective speedup while maintaining generation quality and robustness. By enforcing these global constraints, TI ultimately facilitates System-2 reasoning, empowering the model to deliberate within a structurally defined solution space.
>
---
#### [replaced 033] VectraYX-Nano: A 42M-Parameter Spanish Cybersecurity Language Model with Curriculum Learning and Native Tool Use
- **分类: cs.CL**

- **简介: 该论文提出VectraYX-Nano，一个42M参数的西班牙语网络安全语言模型，解决网络安全文本生成与工具调用问题，通过课程学习和原生工具集成实现高效性能。**

- **链接: [https://arxiv.org/pdf/2605.13989](https://arxiv.org/pdf/2605.13989)**

> **作者:** Juan S. Santillana
>
> **备注:** 22 pages, 5 figures, 19 tables. v2: corrected GPT-4o frontier numbers from measured CSV (B2 0.110, B3-TM 0.520, B4 0.615, B5 0.631); annotated LATAM eval harness bug (all-zero scores, same key-mismatch as B2); fixed §7 B4 prompt count (200, not 25); added Nano v7 to conclusion. Models and eval data released at this https URL
>
> **摘要:** We present VectraYX-Nano, a 41.95M-parameter decoder-only language model trained from scratch in Spanish for cybersecurity, with a Latin-American regional focus and native tool invocation via the Model Context Protocol (MCP). The model has four contributions. (i) Corpus: VectraYX-Sec-ES, a 170M-token Spanish corpus assembled by an eight-VM distributed pipeline at ~$25 USD of cloud compute and split into three curriculum phases (conversational 42M, cybersecurity 118M, offensive tooling 10M). (ii) Architecture: a 42M Transformer decoder with GQA, QK-Norm, RMSNorm, SwiGLU, RoPE and z-loss, paired with a domain-balanced 16,384-token byte-fallback BPE. (iii) Curriculum with replay across the three phases yields a monotonic loss descent (9.80 -> 3.17 -> 3.00 -> 2.16); after SFT (loss 1.74) the v2 bootstrap-ablation reference attains a conversational gate of 0.775 +/- 0.043 on B5 over N=4 seeds, and a controlled Phase-2 replay sweep over {0,5,10,25,50}% saturates B5 at >=25% replay. (iv) Two empirical findings, both N=4. A controlled bootstrap-corpus ablation across v2 (OpenSubs), v4 (mC4-ES), and v6 (60/25/15 OpenSubs/mC4/Wiki) exposes a loss-versus-register inversion: lower-perplexity bootstraps yield measurably worse conversational behavior (v2 > v4 > v6 on B5 at every paired seed). The B4 (tool-selection) floor of 0.000 is a corpus-density artifact, not a capacity gate: rebalancing the SFT mixture to tool-use ratio 1:21 yields VectraYX-Nano v7, the released headline configuration, reaching B4 = 0.230 +/- 0.052 at 42M while retaining B1 = 0.332 +/- 0.005 and B5 = 0.725 +/- 0.130; a LoRA replication on a 260M from-scratch mid-tier reaches 0.445 +/- 0.201. The released GGUF is 96 MB in F16, runs sub-second TTFT on commodity hardware under this http URL, and is, to our knowledge, the first published Spanish-native cybersecurity LLM with end-to-end MCP integration.
>
---
#### [replaced 034] Do Composed Image Retrieval Benchmarks Require Multimodal Composition?
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究CIR任务，探讨其是否需要多模态组合。发现多数查询可单模态解决，揭示了简化的捷径。通过分析和人工验证，指出当前基准混淆了不同类型的查询，导致模型能力被高估。**

- **链接: [https://arxiv.org/pdf/2605.14787](https://arxiv.org/pdf/2605.14787)**

> **作者:** Matteo Attimonelli; Alessandro De Bellis; Aryo Pradipta Gema; Rohit Saxena; Monica Sekoyan; Wai-Chung Kwan; Claudio Pomo; Alessandro Suglia; Dietmar Jannach; Tommaso Di Noia; Pasquale Minervini
>
> **摘要:** Composed Image Retrieval (CIR) is a multimodal retrieval task where a query consists of a reference image and a textual modification, and the goal is to retrieve a target image satisfying both. In principle, strong performance on CIR benchmarks is assumed to require multimodal composition, i.e., combining complementary information from reference image and textual modification. In this work, we show that this assumption does not always hold. Across four widely used CIR benchmarks and eleven Generalist Multimodal Embedding models, a large fraction of queries can be solved using a single modality (from 32.2% to 83.6%), revealing pervasive unimodal shortcuts. Thus, high CIR performance can arise from unimodal signals rather than true multimodal composition. To better understand this issue, we perform a two-stage audit. First, we identify shortcut-solvable queries through cross-model analysis. Second, we conduct human validation on 4,741 shortcut-free queries, of which only 1,689 are well-formed, with common issues including ambiguous edits and mismatched targets. Re-evaluating models on this validated subset reveals qualitatively different behaviour: queries can no longer be solved with a single modality, and successful retrieval requires combining both inputs. While accuracy decreases, reliance on multimodal information increases. Overall, current CIR benchmarks conflate shortcut-solvable, noisy, and genuinely compositional queries, leading to an overestimation of model capability in multimodal composition.
>
---
#### [replaced 035] Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **简介: 该论文属于机器学习中的模型遗忘任务，旨在解决现有评估方法无法真实反映数据删除效果的问题。通过引入表征级分析框架，研究遗忘的可逆性与灾难性遗忘，提出更可靠的评估方法。**

- **链接: [https://arxiv.org/pdf/2505.16831](https://arxiv.org/pdf/2505.16831)**

> **作者:** Xiaoyu Xu; Xiang Yue; Yang Liu; Qingqing Ye; Huadi Zheng; Peizhao Hu; Minxin Du; Haibo Hu
>
> **备注:** ICML 2026, accepted to appear
>
> **摘要:** Unlearning in large language models (LLMs) aims to remove specified data, but its efficacy is typically assessed with task-level metrics like accuracy and perplexity. We show that these metrics can be misleading, as models can appear to forget while their original behavior is easily restored through minimal fine-tuning. This \emph{reversibility} suggests that information is merely suppressed, not genuinely erased. To address this critical evaluation gap, we introduce a \emph{representation-level analysis framework}. Our toolkit comprises PCA similarity and shift, centered kernel alignment (CKA), and Fisher information, complemented by a summary metric, the mean PCA distance, to measure representational drift. Applying this framework across multiple unlearning methods, data domains, and LLMs, we identify four distinct forgetting regimes based on their \emph{reversibility} and \emph{catastrophicity}. We compare recovery strategies and show that relearning efficiency relies on the data source. We also find that irreversible, non-catastrophic forgetting is exceptionally challenging. By probing unlearning limits, we identify a case of seemingly irreversible, targeted forgetting, offering insights for more robust erasure algorithms. Overall, our findings expose a gap in current evaluation and establish a representation-level foundation for trustworthy unlearning.
>
---
#### [replaced 036] From graphemic dependence to lexical structure: a Markovian perspective on Dante's Commedia
- **分类: cs.CL**

- **简介: 该论文属于文本结构分析任务，旨在通过V/C编码和马尔可夫模型研究《神曲》的局部依赖结构。工作包括构建四状态马尔可夫链、分析三部曲间的差异及词汇锚点。**

- **链接: [https://arxiv.org/pdf/2604.22626](https://arxiv.org/pdf/2604.22626)**

> **作者:** Angelo Maria Sabatini
>
> **备注:** 26 pages, 8 figures, 1 supplementary material; submitted to Journal of Computational Literary Studies
>
> **摘要:** This study investigates the structural organisation of Dante's Divina Commedia through a symbolic representation based on vowel-consonant (V/C) encoding. Modelling the resulting sequence as a four-state Markov chain yields a parsimonious index of graphemic memory, capturing local persistence and alternation patterns. Across the poem, this index shows a slight but consistent increase from the Inferno to the Paradiso, indicating a directional shift in local dependency structure. Trigram analysis identifies a restricted set of recurrent configurations acting as graphemic probes, linking Markov patterns to lexical environments and orthographic phenomena such as apostrophised forms. A complementary classification analysis identifies cantica-specific lexical anchors, showing that local symbolic dependencies reflect both the separation among the three cantiche and a continuous progression across the poem. The results provide an interpretable framework connecting local symbolic structure with higher-level textual organisation.
>
---
#### [replaced 037] Wasserstein Distributionally Robust Regret Optimization for Reinforcement Learning from Human Feedback
- **分类: cs.LG; cs.CL; math.OC; stat.ML**

- **简介: 该论文属于强化学习任务，解决RLHF中奖励信号与真实目标不一致的问题。提出Wasserstein DRRO方法，通过优化最差情况下的遗憾来减少过拟合，提升模型部署性能。**

- **链接: [https://arxiv.org/pdf/2605.00155](https://arxiv.org/pdf/2605.00155)**

> **作者:** Yikai Wang; Shang Liu; Jose Blanchet
>
> **摘要:** Reinforcement learning from human feedback (RLHF) has become a core post-training step for aligning large language models, yet the reward signal used in RLHF is only a learned proxy for true human utility. From an operations research perspective, this creates a decision problem under objective misspecification: the policy is optimized against an estimated reward, while deployment performance is determined by an unobserved objective. The resulting gap leads to reward over-optimization, or Goodharting, where proxy reward continues to improve even after true quality deteriorates. Existing mitigations address this problem through uncertainty penalties, pessimistic rewards, or conservative constraints, but they can be computationally burdensome and overly pessimistic. We propose Wasserstein distributionally robust regret optimization (DRRO) for RLHF. Instead of pessimizing worst-case value as in standard DRO, DRRO pessimizes worst-case regret relative to the best policy under the same plausible reward perturbation. We study the promptwise problem through a simplex allocation model and show that, under an $\ell_1$-ground-cost Wasserstein ambiguity set, the inner worst-case regret admits an exact solution and the optimal policy has a water-filling structure. These results lead to a practical policy-gradient algorithm with a simple sampled-bonus interpretation and only minor changes to GRPO-style RLHF training. The framework also clarifies theoretically why DRRO is less pessimistic than DRO, and our experiments show that DRRO mitigates over-optimization more effectively than existing baselines while standard DRO is systematically over-pessimistic.
>
---
#### [replaced 038] Finding Sense in Nonsense with Generated Contexts: Perspectives from Humans and Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言理解任务，旨在区分句子的异常与荒谬。通过收集人类和语言模型的判断，分析五组语义偏离数据集，发现多数句子仅异常而非完全无意义，且语言模型能生成合理上下文。**

- **链接: [https://arxiv.org/pdf/2602.11699](https://arxiv.org/pdf/2602.11699)**

> **作者:** Katrina Olsen; Sebastian Padó
>
> **备注:** Accepted for publication at STARSEM 2026, San Diego, CA
>
> **摘要:** Nonsensical and anomalous sentences have been instrumental in the development of computational models of semantic interpretation. A core challenge is to distinguish between what is merely anomalous (but can be interpreted given a supporting context) and what is truly nonsensical. However, it is unclear (a) how nonsensical, rather than merely anomalous, existing datasets are; and (b) how well LLMs can make this distinction. In this paper, we answer both questions by collecting sensicality judgments from human raters and LLMs on sentences from five semantically deviant datasets: both context-free and when providing a context. We find that raters consider most sentences at most anomalous, and only a few as properly nonsensical. We also show that LLMs are substantially skilled in generating plausible contexts for anomalous cases.
>
---
#### [replaced 039] Friends and Grandmothers in Silico: Localizing Entity Cells in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型如何存储和检索实体知识，通过定位特定的“实体单元”来理解其事实回忆机制，旨在提升模型的可解释性与可控性。**

- **链接: [https://arxiv.org/pdf/2604.01404](https://arxiv.org/pdf/2604.01404)**

> **作者:** Itay Yona; Dan Barzilay; Michael Karasik; Mor Geva
>
> **摘要:** How do language models retrieve entity-specific facts from their parameters? We investigate this question by searching for sparse, entity-selective MLP neurons - which we call entity cells, by analogy to the "grandmother cell" hypothesis in neuroscience - and testing whether they play a causal role in factual recall. We localize candidate entity cells by ranking MLP neurons for activation consistency across varied prompts about the same entity, applying this procedure across seven models on a curated subset of PopQA. In all models, localized neurons cluster predominantly in early layers, an empirical pattern not imposed by the architecture. Using Qwen2.5-7B base as a model organism, we find the clearest causal evidence: suppressing a localized cell selectively erases recall for its matched entity while leaving others intact, and activating a single cell is sufficient to recover correct knowledge for most entities - even when the entity is absent from the context. The same cells are recovered under aliases, acronyms, misspellings, and multilingual surface forms, and remain stable through instruction tuning, suggesting they encode canonical entity identity rather than surface token patterns. Causal signals vary across model families, pointing to architectural differences in how entity knowledge is organized. These findings offer concrete, interpretable access points for understanding, controlling, and correcting factual knowledge in language models, and draw a surprising empirical parallel to longstanding questions in neuroscience about sparse coding of concepts.
>
---
#### [replaced 040] Early Stopping Chain-of-thoughts in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理任务，旨在解决长链式思维（CoT）推理成本高的问题。通过检测答案收敛并提前停止，减少推理token数量，保持准确率。**

- **链接: [https://arxiv.org/pdf/2509.14004](https://arxiv.org/pdf/2509.14004)**

> **作者:** Minjia Mao; Bowen Yin; Yu Zhu; Xiao Fang
>
> **摘要:** Reasoning large language models (LLMs) have demonstrated superior capacities in solving complicated problems by generating long chain-of-thoughts (CoT), but such a lengthy CoT incurs high inference costs. Previous methods on inference-stage efficient reasoning either require white-box models to monitor the reasoning process or are not reliable through direct prompting. In response, we introduce ES-CoT, an inference-time method that shortens CoT generation by detecting answer convergence and stopping early with almost no performance loss. When observing a linguistic marker (such as "wait") in the reasoning process, we prompt the LLM to output its current final answer, denoted as a step answer. We then track the run length of consecutive identical step answers as a measure of answer convergence. We show both empirically and theoretically that step answers steadily converge to the final answer, and large run-length jumps reliably mark this convergence. Experiments on six reasoning datasets across three LLMs show that ES-CoT reduces the number of inference tokens by 16.08% on average while maintaining accuracy comparable to standard CoT.
>
---
#### [replaced 041] Spherical Steering: Geometry-Aware Activation Rotation for Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于语言模型控制任务，解决推理时控制模型生成的问题。提出Spherical Steering方法，通过激活旋转实现精准控制，避免信息损失，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2602.08169](https://arxiv.org/pdf/2602.08169)**

> **作者:** Zejia You; Chunyuan Deng; Hanjie Chen
>
> **备注:** ICML 2026
>
> **摘要:** Inference-time steering offers a promising way to control language models (LMs) without retraining. However, standard approaches typically rely on activation addition, which inevitably alters the hidden-state magnitudes raising concerns about representation collapse and degraded open-ended generation. In this work, we explore Spherical Steering, a training-free primitive that resolves this trade-off through activation rotation. Rather than shifting activations with a fixed vector, our method rotates them along a geodesic toward a target direction, preserving signal integrity while steering toward the target concept. To further enhance adaptivity, we incorporate a confidence gate that dynamically modulates steering strength based on input uncertainty. Extensive experiments across multiple-choice benchmarks demonstrate that Spherical Steering significantly outperforms addition-based baselines (notably by +10% on TruthfulQA, COPA, and Storycloze), while simultaneously maintaining the model's general open-ended generation quality. This work highlights the value of geometric consistency, suggesting that norm-preserving rotation is a robust and effective primitive for precise inference-time control. The code is available at: this https URL.
>
---
#### [replaced 042] The Homogenization Problem in LLMs: Towards Meaningful Diversity in AI Safety
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文探讨生成式AI中的同质化问题，旨在提升AI安全性中的多样性。通过构建框架和实验，识别并缓解模型中的性别偏见，提出xeno-reproduction方法促进多元表达。**

- **链接: [https://arxiv.org/pdf/2601.06116](https://arxiv.org/pdf/2601.06116)**

> **作者:** Ian Rios-Sialer
>
> **摘要:** Generative AI models reproduce the human biases in their training data and further amplify them through mechanisms such as mode collapse. The loss of diversity produces homogenization, which not only harms the minoritized but impoverishes everyone. We argue homogenization should be a central concern in AI safety. To meaningfully characterize homogenization in Large Language Models (LLMs), we introduce a framework that allows stakeholders to encode their context and value system. We illustrate our approach with an experiment that surfaces gender bias in an LLM (Claude 3.5 Haiku) on an open-ended story prompt. Building from queer theory, we formalize homogenization in terms of normativity. Borrowing language from feminist theory, we introduce the concept of xeno-reproduction as a class of tasks for mitigating homogenization by promoting diversity. Our work opens a collaborative line of research that seeks to understand and advance diversity in AI.
>
---
#### [replaced 043] GraphMind: Theorem Selection and Conclusion Generation Framework with Dynamic GNN for LLM Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出GraphMind框架，解决LLM在多步推理中缺乏动态结构表示的问题。通过动态图神经网络实现定理选择与结论生成，提升推理能力。**

- **链接: [https://arxiv.org/pdf/2511.19078](https://arxiv.org/pdf/2511.19078)**

> **作者:** Yutong Li; Yitian Zhou; Xudong Wang; GuoChen; Caiyan Qin
>
> **备注:** This paper has been withdrawn by the authors in order to prepare a substantially revised version
>
> **摘要:** Large language models (LLMs) have demonstrated impressive capabilities in natural language understanding and generation, including multi-step reasoning such as mathematical proving. However, existing approaches often lack an explicit and dynamic mechanism to structurally represent and evolve intermediate reasoning states, which limits their ability to perform context-aware theorem selection and iterative conclusion generation. To address these challenges, we propose GraphMind, a novel dynamic graph-based framework that integrates the graph neural network (GNN) with LLMs to iteratively select theorems and generate intermediate conclusions for multi-step reasoning. Our method models the reasoning process as a heterogeneous evolving graph, where nodes represent conditions, theorems, and conclusions, while edges capture logical dependencies between nodes. By encoding the current reasoning state with GNN and leveraging semantic matching for theorem selection, our framework enables context-aware, interpretable, and structured reasoning in a closed-loop manner. Experiments on various question-answering (QA) datasets demonstrate that our proposed GraphMind method achieves consistent performance improvements and significantly outperforms existing baselines in multi-step reasoning, validating the effectiveness and generalizability of our approach.
>
---
#### [replaced 044] ProfBench: Multi-Domain Rubrics requiring Professional Knowledge to Answer and Judge
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出ProfBench，用于评估大语言模型在专业领域任务中的表现，解决现有评估方法受限的问题。通过构建专家标注的数据集和低成本的评估系统，提升评估的公平性和实用性。**

- **链接: [https://arxiv.org/pdf/2510.18941](https://arxiv.org/pdf/2510.18941)**

> **作者:** Zhilin Wang; Jaehun Jung; Ximing Lu; Shizhe Diao; Ellie Evans; Jiaqi Zeng; Pavlo Molchanov; Yejin Choi; Jan Kautz; Yi Dong
>
> **备注:** Published at ICLR 2026, 30 pages
>
> **摘要:** Evaluating progress in large language models (LLMs) is often constrained by the challenge of verifying responses, limiting assessments to tasks like mathematics, programming, and short-form question-answering. However, many real-world applications require evaluating LLMs in processing professional documents, synthesizing information, and generating comprehensive reports in response to user queries. We introduce ProfBench: a set of over 7000 response-criterion pairs as evaluated by human-experts with professional knowledge across Physics PhD, Chemistry PhD, Finance MBA and Consulting MBA. We build robust and affordable LLM-Judges to evaluate ProfBench rubrics, by mitigating self-enhancement bias and reducing the cost of evaluation by 2-3 orders of magnitude, to make it fair and accessible to the broader community. Our findings reveal that ProfBench poses significant challenges even for state-of-the-art LLMs, with top-performing models like GPT-5-high achieving only 65.9% overall performance. Furthermore, we identify notable performance disparities between proprietary and open-weight models and provide insights into the role that extended thinking plays in addressing complex, professional-domain tasks. Data: this https URL and Code: this https URL and Leaderboard: this https URL
>
---
#### [replaced 045] AdaSwitch: Adaptive Switching between Small and Large Agents for Effective Cloud-Local Collaborative Learning
- **分类: cs.CL**

- **简介: 该论文提出AdaSwitch，解决云-本地大语言模型协作问题，通过自适应切换提升任务性能与效率。**

- **链接: [https://arxiv.org/pdf/2410.13181](https://arxiv.org/pdf/2410.13181)**

> **作者:** Hao Sun; Jiayi Wu; Hengyi Cai; Xiaochi Wei; Yue Feng; Bo Wang; Shuaiqiang Wang; Yan Zhang; Dawei Yin
>
> **备注:** EMNLP 2024 Main Conference
>
> **摘要:** Recent advancements in large language models (LLMs) have been remarkable. Users face a choice between using cloud-based LLMs for generation quality and deploying local-based LLMs for lower computational cost. The former option is typically costly and inefficient, while the latter usually fails to deliver satisfactory performance for reasoning steps requiring deliberate thought processes. In this work, we propose a novel LLM utilization paradigm that facilitates the collaborative operation of large cloud-based LLMs and smaller local-deployed LLMs. Our framework comprises two primary modules: the local agent instantiated with a relatively smaller LLM, handling less complex reasoning steps, and the cloud agent equipped with a larger LLM, managing more intricate reasoning steps. This collaborative processing is enabled through an adaptive mechanism where the local agent introspectively identifies errors and proactively seeks assistance from the cloud agent, thereby effectively integrating the strengths of both locally-deployed and cloud-based LLMs, resulting in significant enhancements in task completion performance and efficiency. We evaluate AdaSwitch across 7 benchmarks, ranging from mathematical reasoning and complex question answering, using various types of LLMs to instantiate the local and cloud agents. The empirical results show that AdaSwitch effectively improves the performance of the local agent, and sometimes achieves competitive results compared to the cloud agent while utilizing much less computational overhead.
>
---
#### [replaced 046] Prompt reinforcing for long-term planning of large language models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于对话系统任务，旨在解决LLM在多轮交互中的长期规划问题。通过强化学习启发的提示优化框架，提升模型跟踪用户目标的能力。**

- **链接: [https://arxiv.org/pdf/2510.05921](https://arxiv.org/pdf/2510.05921)**

> **作者:** Hsien-Chin Lin; Benjamin Matthias Ruppik; Carel van Niekerk; Chia-Hao Shen; Michael Heck; Nurul Lubis; Renato Vukovic; Shutong Feng; Milica Gašić
>
> **摘要:** Large language models (LLMs) have achieved remarkable success in a wide range of natural language processing tasks and can be adapted through prompting. However, they remain suboptimal in multi-turn interactions, often relying on incorrect early assumptions and failing to track user goals over time, which makes such tasks particularly challenging. Prior works in dialogue systems have shown that long-term planning is essential for handling interactive tasks. In this work, we propose a prompt optimisation framework inspired by reinforcement learning, which enables such planning to take place by only modifying the task instruction prompt of the LLM-based agent. By generating turn-by-turn feedback and leveraging experience replay for prompt rewriting, our proposed method shows significant improvement in multi-turn tasks such as text-to-SQL and task-oriented dialogue. Moreover, it generalises across different LLM-based agents and can leverage diverse LLMs as meta-prompting agents. This warrants future research in reinforcement learning-inspired parameter-free optimisation methods.
>
---
#### [replaced 047] NodeSynth: Socially Aligned Synthetic Data for AI Evaluation
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出NodeSynth，用于生成具有社会相关性的合成数据，以评估AI模型的安全性和可靠性。任务是提升AI模型在敏感领域的评估效果，解决现有数据缺乏社会技术细节的问题。工作包括构建基于真实证据的分类生成器，并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2605.14381](https://arxiv.org/pdf/2605.14381)**

> **作者:** Qazi Mamunur Rashid; Xuan Yang; Zhengzhe Yang; Yanzhou Pan; Erin van Liemt; Darlene Neal; Kshitij Pancholi; Jamila Smith-Loud
>
> **摘要:** Recent advancements in generative AI facilitate large-scale synthetic data generation for model evaluation. However, without targeted approaches, these datasets often lack the sociotechnical nuance required for sensitive domains. We introduce NodeSynth, an evidence-grounded methodology that generates socially relevant synthetic queries by leveraging a fine-tuned taxonomy generator (TaG) anchored in real-world evidence. Evaluated against four mainstream LLMs (e.g., Claude 4.5 Haiku), NodeSynth elicited failure rates up to five times higher than human-authored benchmarks. Ablation studies confirm that our granular taxonomic expansion significantly drives these failure rates, while independent validation reveals critical deficiencies in prominent guard models (e.g., Llama-Guard-3). We open-source our end-to-end research prototype and datasets to enable scalable, high-stakes model evaluation and targeted safety interventions (this https URL).
>
---
#### [replaced 048] KIT-TIP-NLP at MultiPride: Continual Learning with Multilingual Foundation Model
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文聚焦于多语言社交媒体中再利用贬义词的检测任务，解决数据稀缺、类别不平衡和跨语言情感差异问题，通过模型选择、数据增强和阈值优化提升检测效果。**

- **链接: [https://arxiv.org/pdf/2605.13415](https://arxiv.org/pdf/2605.13415)**

> **作者:** Barathi Ganesh HB; Michal Ptaszynski; Rene Melendez; Juuso Eronen
>
> **备注:** Final Workshop of the 9th evaluation campaign EVALITA 2026
>
> **摘要:** This paper presents a multi-stage framework for detecting reclaimed slurs in multilingual social media discourse. It addresses the challenge of identifying reclamatory versus non-reclamatory usage of LGBTQ+-related slurs across English, Spanish, and Italian tweets. The framework handles three intertwined methodological challenges like data scarcity, class imbalance, and cross-linguistic variation in sentiment expression. It integrates data-driven model selection via cross-validation, semantic-preserving augmentation through back-translation, inductive transfer learning with dynamic epoch-level undersampling, and domain-specific knowledge injection via masked language modeling. Eight multilingual embedding models were evaluated systematically, with XLM-RoBERTa selected as the foundation model based on macro-averaged F1 score. Data augmentation via GPT-4o-mini back-translation to alternate languages effectively tripled the training corpus while preserving semantic content and class distribution ratios. The framework produces four final runs for the evaluation purposes where RUN 1 is inductive transfer learning with augmentation and undersampling, RUN 2 with masked language modeling pre-training, RUN 3 and RUN 4 are previous predictions refined via language-specific decision thresholds optimized via ROC analysis. Language-specific threshold refinement reveals that optimal decision boundaries vary significantly across languages. This reflects distributional differences in model confidence scores and linguistic variation in reclamatory language usage. The threshold-based optimization yields 2-5% absolute F1 improvement without requiring model retraining. The methodology is fully reproducible, with all code and experimental setup available at this https URL.
>
---
#### [replaced 049] Speak Your Mind: The Speech Continuation Task as a Probe of Voice-Based Model Bias
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音续写任务，旨在探究语音基础模型中的偏见。通过评估性别和发声类型对续写行为的影响，揭示模型在语音质量和文本指标上的偏差。**

- **链接: [https://arxiv.org/pdf/2509.22061](https://arxiv.org/pdf/2509.22061)**

> **作者:** Shree Harsha Bokkahalli Satish; Harm Lameris; Olivier Perrotin; Gustav Eje Henter; Éva Székely
>
> **备注:** 8 pages, 2 figures, Accepted to Identity-Aware AI LREC Workshop 2026
>
> **摘要:** Speech Continuation (SC) is the task of generating a coherent extension of a spoken prompt while preserving both semantic context and speaker identity. Because SC is constrained to a single audio stream, it offers a more direct setting for probing biases in speech foundation models than dialogue does. In this work we present the first systematic evaluation of bias in SC, investigating how gender and phonation type (breathy, creaky, end-creak) affect continuation behaviour. We evaluate three recent models: SpiritLM (base and expressive), VAE-GSLM, and SpeechGPT across speaker similarity, voice quality preservation, and text-based bias metrics. Results show that while both speaker similarity and coherence remain a challenge, textual evaluations reveal significant model and gender interactions: once coherence is sufficiently high (for VAE-GSLM), gender effects emerge on text-metrics such as agency and sentence polarity. In addition, continuations revert toward modal phonation more strongly for female prompts than for male ones, revealing a systematic voice-quality bias. These findings highlight SC as a controlled probe of socially relevant representational biases in speech foundation models, and suggest that it will become an increasingly informative diagnostic as continuation quality improves.
>
---
#### [replaced 050] Helpful to a Fault: Measuring Illicit Assistance in Multi-Turn, Multilingual LLM Agents
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于安全评估任务，旨在检测多轮、多语言大模型代理在滥用场景下的风险。工作包括提出STING框架，评估代理在复杂非法任务中的协助程度。**

- **链接: [https://arxiv.org/pdf/2602.16346](https://arxiv.org/pdf/2602.16346)**

> **作者:** Nivya Talokar; Ayush K Tarun; Murari Mandal; Maksym Andriushchenko; Antoine Bosselut
>
> **摘要:** LLM-based agents execute real-world workflows via tools and memory. These affordances enable ill-intended adversaries to also use these agents to carry out complex misuse scenarios. Existing agent misuse benchmarks largely test single-prompt instructions, leaving a gap in measuring how agents end up helping with harmful or illegal tasks over multiple turns. We introduce STING (Sequential Testing of Illicit N-step Goal execution), an automated red-teaming framework that constructs a step-by-step illicit plan grounded in a benign persona and iteratively probes a target agent with adaptive follow-ups, using judge agents to track phase completion. We further introduce an analysis framework that models multi-turn red-teaming as a time-to-first-jailbreak random variable, enabling analysis tools like discovery curves, hazard-ratio attribution by attack language, and a new metric: Restricted Mean Jailbreak Discovery. Across AgentHarm scenarios, STING yields substantially higher illicit-task completion than single-turn prompting and chat-oriented multi-turn baselines adapted to tool-using agents. In multilingual evaluations across six non-English settings, we find that attack success and illicit-task completion do not consistently increase in lower-resource languages, diverging from common chatbot findings. Overall, STING provides a practical way to evaluate and stress-test agent misuse in realistic deployment settings, where interactions are inherently multi-turn and often multilingual.
>
---
#### [replaced 051] Double-Calibration: Towards Reliable LLMs via Calibrating Knowledge and Reasoning Confidence
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于可信推理任务，旨在解决LLMs的幻觉问题。通过双校准框架DoublyCal，提升模型的准确性和置信度校准。**

- **链接: [https://arxiv.org/pdf/2601.11956](https://arxiv.org/pdf/2601.11956)**

> **作者:** Yuyin Lu; Ziran Liang; Yanghui Rao; Wenqi Fan; Fu Lee Wang; Qing Li
>
> **备注:** This work is to appear in the Proceedings of the 35th International Joint Conference on Artificial Intelligence (IJCAI 2026)
>
> **摘要:** Reliable reasoning in Large Language Models (LLMs) is challenged by their propensity for hallucination. While augmenting LLMs with Knowledge Graphs (KGs) improves factual accuracy, existing KG-augmented methods fail to quantify epistemic uncertainty in both the retrieved evidence and LLMs' reasoning. To bridge this gap, we introduce DoublyCal, a framework built on a novel double-calibration principle. DoublyCal employs a lightweight proxy model to first generate KG evidence alongside a calibrated evidence confidence. This calibrated supporting evidence then guides a black-box LLM, yielding final predictions that are not only more accurate but also well-calibrated, with confidence scores traceable to the uncertainty of the supporting evidence. Experiments on knowledge-intensive benchmarks show that DoublyCal significantly improves both the accuracy and confidence calibration of black-box LLMs while maintaining low token cost.
>
---
#### [replaced 052] Evolve the Method, Not the Prompts: Evolutionary Synthesis of Jailbreak Attacks on LLMs
- **分类: cs.CL; cs.CR**

- **简介: 该论文属于LLM安全研究任务，旨在解决传统攻击方法依赖提示优化的问题。工作是提出EvoSynth框架，通过代码空间进化生成更有效且多样的越狱攻击。**

- **链接: [https://arxiv.org/pdf/2511.12710](https://arxiv.org/pdf/2511.12710)**

> **作者:** Yunhao Chen; Xin Wang; Juncheng Li; Yixu Wang; Jie Li; Yan Teng; Yingchun Wang; Xingjun Ma
>
> **摘要:** Automated red teaming frameworks for Large Language Models (LLMs) have become increasingly sophisticated, yet many still formulate attack optimization primarily in the prompt space. In other words, these methods mainly search for better attack wording or better strategy choices, but they do not search over executable code. By moving the search into code space, we can optimize not only the final attack prompt, but also the procedure that generates it, including execution flow, reusable logic, branching, and failure-driven repair. To overcome this gap, we introduce EvoSynth, an autonomous multi-agent framework that shifts the optimization space from prompts to executable code. Instead of refining prompts directly, EvoSynth employs a multi-agent system to autonomously engineer, evolve, and execute code-based attack algorithms. Crucially, it features a code-level self-correction loop, allowing it to iteratively rewrite the code-based algorithm in response to target-model feedback and failed attempts. Through extensive experiments, we demonstrate that EvoSynth achieves an 85.5\% Attack Success Rate (ASR) against highly robust models like Claude-Sonnet-4.5 and a 95.9\% average ASR across evaluated targets, while generating attacks that are significantly more diverse than those from existing methods. We release our framework to facilitate future research on evolutionary synthesis in executable code space.
>
---
#### [replaced 053] Large Language Models and Impossible Language Acquisition: "False Promise" or an Overturn of our Current Perspective towards AI
- **分类: cs.CL**

- **简介: 论文探讨LLM是否能区分可能与不可能语言，通过实验对比GPT-2和LSTM模型表现，挑战Chomsky观点，提出新的理论视角。任务属于AI与语言学交叉研究，解决LLM语言理解能力问题。**

- **链接: [https://arxiv.org/pdf/2602.08437](https://arxiv.org/pdf/2602.08437)**

> **作者:** Ziyan Wang; Longlong Ma
>
> **摘要:** In Chomsky's provocative critique "The False Promise of CHATGPT," Large Language Models (LLMs) are characterized as mere pattern predictors that do not acquire languages via intrinsic causal and self-correction structures like humans, therefore are not able to distinguish impossible languages. It stands as a representative in a fundamental challenge to the intellectual foundations of AI, for it integrally synthesizes major issues in methodologies within LLMs and possesses an iconic a priori rationalist perspective. We examine this famous critique from both the perspective in pre-existing literature of linguistics and psychology as well as a research based on an experiment inquiring into the capacity of learning both possible and impossible languages among LLMs. We constructed a set of syntactically impossible languages by applying certain transformations to English. These include reversing whole sentences, and adding negation based on word-count parity. Two rounds of controlled experiments were each conducted on GPT-2 small models and long short-term memory (LSTM) models. Descriptive analysis of single-run training trajectories shows that GPT-2 small models exhibit lower final loss, faster convergence, and lower perplexity on natural language compared to impossible language conditions, with the reversed condition showing the largest departure (loss ratios up to 2.25 * natural). LSTM models, by contrast, show minimal differences across conditions. Given the single-run nature of our experiments (n=1 per condition), we report descriptive comparisons and caution that formal statistical inference is precluded. Based on theoretical analysis and descriptive empirical findings, we propose a new vision within Chomsky's theory towards LLMs, and a shift of theoretical paradigm outside Chomsky, from his "rationalist-romantics" paradigm to functionalism and empiricism in LLMs research.
>
---
#### [replaced 054] Tokenizer Fertility and Zero-Shot Performance of Foundation Models on Ukrainian Legal Text: A Comparative Study
- **分类: cs.CL**

- **简介: 该论文研究基础模型在乌克兰法律文本上的分词器效率与零样本性能，比较不同模型表现，解决模型选择与跨时期适应性问题。**

- **链接: [https://arxiv.org/pdf/2605.14890](https://arxiv.org/pdf/2605.14890)**

> **作者:** Volodymyr Ovcharov
>
> **备注:** 25 pages, 13 tables, 5 figures; v2 adds cross-temporal generalization experiment and classical baseline
>
> **摘要:** Tokenizer fertility varies 1.6x across foundation models on Ukrainian legal text, yet this cost-critical dimension is absent from model selection practice. We benchmark seven models from five providers on 273 validated court decisions from Ukraine's state registry (EDRSR), measuring tokenizer fertility and zero-shot performance on three tasks. Four findings emerge. (1) Qwen 3 models consume 60% more tokens than Llama-family models on identical input, making tokenizer analysis a prerequisite for cost-efficient deployment. (2) NVIDIA Nemotron Super 3 (120B) achieves the highest composite score (83.1), outperforming Mistral Large 3 (5.6x more total parameters) at one-third the API cost model scale is a poor proxy for domain performance. (3) Few-shot prompting degrades performance by up to 26 percentage points; stratified and prompt-sensitivity ablations confirm this is intrinsic to Ukrainian-language demonstrations, not an artifact of example selection. (4) A cross-temporal generalization experiment reveals that classifiers trained on pre-war court ecisions (2008-2013) lose 27.9 percentage points when applied to full-scale invasion era decisions (2022-2026), with a pronounced forward-backward asymmetry: newer models transfer backward (+14.6 pp above forward transfer), but older models fail catastrophically on wartime legal language. For practitioners: tokenizer analysis should precede model selection, and zero-shot is a more reliable default than few-shot for morphologically rich languages. To support reproducibility and address the absence of Ukrainian from legal NLP benchmarks, we release a public dataset of 14,452 court decisions spanning 2008-2026, annotated with seven outcome labels across three temporal epochs that capture the impact of armed conflict on judicial proceedings.
>
---
#### [replaced 055] OmniCode: A Benchmark for Evaluating Software Engineering Agents
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出OmniCode基准，用于评估软件工程AI代理。解决现有基准任务狭窄的问题，涵盖更广泛的软件开发任务，如错误修复、测试生成等。**

- **链接: [https://arxiv.org/pdf/2602.02262](https://arxiv.org/pdf/2602.02262)**

> **作者:** Atharv Sonwane; Eng-Shen Tu; Wei-Chung Lu; Claas Beger; Carter Larsen; Debjit Dhar; Simon Alford; Rachel Chen; Ronit Pattanayak; Tuan Anh Dang; Guohao Chen; Gloria Geng; Kevin Ellis; Saikat Dutta
>
> **摘要:** LLM-powered coding agents are redefining how real-world software is developed. To drive the research towards better coding agents, we require challenging benchmarks that can rigorously evaluate the ability of such agents to perform various software engineering tasks. However, popular coding benchmarks such as HumanEval and SWE-Bench focus on narrowly scoped tasks such as competition programming and patch generation. In reality, software engineers have to handle a broader set of tasks for real-world software development. To address this gap, we propose OmniCode, a novel software engineering benchmark that contains a broader and more diverse set of task categories beyond code or patch generation. Overall, OmniCode contains 1794 tasks spanning three programming languages - Python, Java, and C++ - and four key categories: bug fixing, test generation, code review fixing, and style fixing. In contrast to prior software engineering benchmarks, the tasks in OmniCode are (1) manually validated to eliminate ill-defined problems, and (2) synthetically crafted or recently curated to avoid data leakage issues, presenting a new framework for synthetically generating diverse software tasks from limited real-world data. We evaluate OmniCode with popular agent frameworks such as SWE-Agent and show that while they may perform well on bug fixing for Python, they fall short on tasks such as Test Generation and in languages such as C++ and Java. For instance, SWE-Agent achieves a maximum of 25.0% with DeepSeek-V3.1 on C++ Test Generation. OmniCode aims to serve as a robust benchmark and spur the development of agents that can perform well across different aspects of software development. Code and data are available at this https URL.
>
---
#### [replaced 056] Locally Coherent Parallel Decoding in Diffusion Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于代码生成任务，解决扩散语言模型并行生成中的语法不一致问题。通过引入辅助自回归模型，实现高效且连贯的并行解码。**

- **链接: [https://arxiv.org/pdf/2603.20216](https://arxiv.org/pdf/2603.20216)**

> **作者:** Michael Hersche; Nicolas Menet; Ronan Tanios; Abbas Rahimi
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Diffusion language models (DLMs) have emerged as a promising alternative to autoregressive (AR) models, offering sub-linear generation latency and bidirectional capabilities that are particularly appealing for code generation and editing. Achieving sub-linear latency in discrete DLMs requires predicting multiple tokens in parallel. However, standard DLMs sample tokens independently from conditional marginal distributions, failing to capture the joint dependencies among concurrently generated tokens. As a result, they often lead to syntactic inconsistencies and break multi-token structures. In this work, we introduce CoDiLA (Coherent Diffusion with Local Autoregression), a method that reconciles parallel sampling with local dependency modeling. Rather than forcing the DLM to resolve fine-grained syntax, CoDiLA delegates local decoding to a small, auxiliary AR model operating on the diffusion latents. This design allows for parallel generation while ensuring sequential validity within a block and maintaining core DLM capabilities, including bidirectional modeling across blocks. We demonstrate that using a highly compact auxiliary AR model (e.g., 0.6B parameters) effectively eliminates coherence artifacts, establishing a new Pareto frontier for accuracy and speed in code generation benchmarks.
>
---
#### [replaced 057] Deep sequence models tend to memorize geometrically; it is unclear why
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 论文探讨深度序列模型如何存储事实，提出几何记忆概念，揭示其不同于传统关联记忆的存储机制。任务是理解模型记忆方式，解决为何模型倾向于几何存储的问题。工作包括分析几何记忆特性及形成原因。**

- **链接: [https://arxiv.org/pdf/2510.26745](https://arxiv.org/pdf/2510.26745)**

> **作者:** Shahriar Noroozizadeh; Vaishnavh Nagarajan; Elan Rosenfeld; Sanjiv Kumar
>
> **备注:** Forty-third International Conference on Machine Learning (ICML 2026)
>
> **摘要:** Deep sequence models are said to store atomic facts predominantly in the form of associative memory: a brute-force lookup of co-occurring entities. We identify a dramatically different form of storage of atomic facts that we term as geometric memory. Here, the model has synthesized embeddings encoding novel global relationships between all entities, including ones that do not co-occur in training. Such storage is powerful: for instance, we show how it transforms a hard reasoning task involving an $\ell$-fold composition into an easy-to-learn $1$-step navigation task. From this phenomenon, we extract fundamental aspects of neural embedding geometries that are hard to explain. We argue that the rise of such a geometry, as against a lookup of local associations, cannot be straightforwardly attributed to typical supervisory, architectural, or optimizational pressures. Counterintuitively, a geometry is learned even when it is more complex than the brute-force lookup. Then, by analyzing a connection to Node2Vec, we demonstrate how the geometry stems from a spectral bias that -- in contrast to prevailing theories -- indeed arises naturally despite the lack of various pressures. This analysis also points out to practitioners a visible headroom to make Transformer memory more strongly geometric. We hope the geometric view of parametric memory encourages revisiting the default intuitions that guide researchers in areas like knowledge acquisition, capacity, discovery, and unlearning.
>
---
#### [replaced 058] Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于代码调试任务，旨在评估大模型在调试中的精确性。提出PDB框架，通过生成多bug程序并定义新指标，发现模型虽能通过测试但精度不足，需优化训练流程。**

- **链接: [https://arxiv.org/pdf/2604.17338](https://arxiv.org/pdf/2604.17338)**

> **作者:** Wang Bill Zhu; Miaosen Chai; Shangshang Wang; Yejia Liu; Song Bian; Honghua Dong; Willie Neiswanger; Robin Jia
>
> **摘要:** Unlike code completion, debugging requires localizing faults and applying targeted edits. We observe that frontier LLMs often regenerate correct but over-edited solutions during debugging. To evaluate how far LLMs are from precise debugging, we introduce the Precise Debugging Benchmark (PDB) framework, which automatically converts any coding dataset into a debugging benchmark with precision-aware evaluation. PDB generates buggy programs by synthesizing verified atomic bugs and composing them into multi-bug programs. We define two novel metrics, edit-level precision and bug-level recall, which measures how many necessary edits are made and how many bugs are resolved. We release two evaluation benchmarks: PDB-Single-Hard on single-line bugs, and PDB-Multi on multi-line bugs. Experiments show that frontier models, such as GPT-5.1-Codex and DeepSeek-V3.2-Thinking, achieve unit-test pass rates above 76% but exhibit precision below 45%, even when explicitly instructed to perform minimal debugging. Finally, we show that iterative and agentic debugging strategies do not substantially improve precision or recall, highlighting the need to rethink post-training pipelines for coding models.
>
---
#### [replaced 059] Mechanism Plausibility in Generative Agent-Based Modeling
- **分类: cs.MA; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于人工智能与社会模拟领域，旨在解决生成模型在机制解释上的不足。通过构建四等级的机制可信度评估体系，区分模型的生成能力和机制合理性。**

- **链接: [https://arxiv.org/pdf/2605.12824](https://arxiv.org/pdf/2605.12824)**

> **作者:** Patrick Zhao; David Huu Pham; Nicholas Vincent
>
> **备注:** Accepted at ACM FAccT 2026
>
> **摘要:** Large language models (LLMs) can generate high-level diverse phenomena without explicitly programmed rules. This capability has led to their adoption within different agent-based models (ABMs) and social simulations. Recent studies investigate their ability to generate different phenomena of interest, for example, human behavior on social media platforms or alien behavior in game-theoretic scenarios. However, capability, prediction, and explanation are different--drawing from the philosophy of science and mechanisms literature, explanation requires showing, to some degree, how a phenomenon is produced by related organized entities and activities. For modelers, describing the characteristics of an experiment or whether a simulation provides progress in capability (or explanation), can be difficult without being grounded in potentially distant research areas. We integrate recent work on LLM-ABMs with contemporary philosophy of science literature and use it to operationalize a definition of 'plausibility' in a four-level scale. Our scale separates the evaluation of a model's generative sufficiency (ability to reproduce a phenomenon) from its mechanistic plausibility (how the phenomenon could be produced), and clarifies the distinct roles of different models, such as predictive and explanatory ones. We introduce this as the Mechanism Plausibility Scale.
>
---
#### [replaced 060] Embodied Multi-Agent Coordination by Aligning World Models Through Dialogue
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文研究多智能体协作任务，解决部分可观测环境下如何通过对话实现世界模型对齐的问题。通过引入自然语言对话机制，评估对话是否促进真实协作而非表面协调。**

- **链接: [https://arxiv.org/pdf/2605.12920](https://arxiv.org/pdf/2605.12920)**

> **作者:** Vardhan Dongre; Dilek Hakkani-Tür
>
> **摘要:** Effective collaboration between embodied agents requires more than acting in a shared environment; it demands communication grounded in each agent's evolving understanding of the world. When agents can only partially observe their surroundings, coordination without communication is provably hard, but communication can, in principle, bridge this gap by allowing agents to share observations and align their world models. In this work, we examine whether LLM-based embodied agents actually realize the ability to communicate. We extend PARTNR, a benchmark for collaborative household robotics, with a natural-language dialogue channel that enables two agents with partial observability to communicate during task execution. To evaluate whether dialogue leads to genuine world-model alignment rather than superficial coordination, we propose a framework for measuring world-model alignment defined over per-agent world graphs: observation convergence (do private world models align over time?), information novelty (do messages convey what the partner lacks?), and belief-sensitive messaging (do agents model what their partner knows?). Our experiments across three LLMs reveal that dialogue reduces action conflicts 40 to 83 percentage points but degrades task success relative to silent coordination. Using our metrics, we characterize the gap between superficial coordination and genuine world-model alignment, and identify where current models fall on this spectrum.
>
---
#### [replaced 061] "The Whole Is Greater Than the Sum of Its Parts": A Compatibility-Aware Multi-Teacher CoT Distillation Framework
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识蒸馏任务，旨在解决多教师协同训练中模型兼容性问题，通过动态融合不同教师的监督信号提升学生模型的推理能力。**

- **链接: [https://arxiv.org/pdf/2601.13992](https://arxiv.org/pdf/2601.13992)**

> **作者:** Jin Cui; Jiaqi Guo; Ruixuan Yang; Jiayi Lu; Jiepeng Zhou; Jiajun Xu; Jiangcheng Song; Boran Zhao; Pengju Ren
>
> **备注:** 11pages, 9figures
>
> **摘要:** Chain-of-Thought (CoT) reasoning empowers Large Language Models (LLMs) with remarkable capabilities but typically requires prohibitive parameter scales. CoT distillation has emerged as a promising paradigm to transfer reasoning prowess into compact Student Models (SLMs), but existing approaches often rely on a solitary teacher, capping the student's potential since individual LLMs often exhibit distinct capability biases and may suffer from catastrophic forgetting. While leveraging diverse teachers seems appealing, effectively fusing their supervisions remains challenging: teacher-student incompatibility risks amplifying hallucinations, and passive supervision fails to ensure genuine logic internalization. To address this, we introduce COMPACT, a framework that adaptively fuses supervisions from different teachers by dynamically weighting teacher gradients based on the student's real-time compatibility evaluated by a multi-dimensional metric: (1) Graph-based Consensus to filter misleading rationales by identifying mainstream reasoning paths; (2) Mutual-Information-based Adaptability to detect "epiphany moments" for genuinely understanding the reasoning process rather than merely imitating; and (3) Loss-based Difficulty to assess student receptivity to the teacher's guidance and prevent negative transfer. Extensive experiments and latent space analysis demonstrate that COMPACT effectively integrates diverse reasoning capabilities without damaging the model's original knowledge structure, achieving state-of-the-art performance on various benchmarks while mitigating catastrophic forgetting.
>
---
#### [replaced 062] Lean Meets Theoretical Computer Science: Scalable Synthesis of Theorem Proving Challenges in Formal-Informal Pairs
- **分类: cs.LO; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自动化定理证明任务，旨在解决数据稀缺问题。通过理论计算机科学生成大量严谨的定理-证明对，提升模型推理能力评估。**

- **链接: [https://arxiv.org/pdf/2508.15878](https://arxiv.org/pdf/2508.15878)**

> **作者:** Terry Jingchen Zhang; Wenyuan Jiang; Rongchuan Liu; Yisong Wang; Junran Yang; Ning Wang; Nicole Ni; Yinya Huang; Mrinmaya Sachan
>
> **备注:** Accepted to AI4MATH@ICML2025
>
> **摘要:** Formal theorem proving (FTP) has emerged as a critical foundation for evaluating the reasoning capabilities of large language models, enabling automated verification of mathematical proofs at scale. However, progress has been constrained by limited datasets due to the high cost of manual curation and the scarcity of challenging problems with verified formal-informal correspondences. We propose leveraging theoretical computer science (TCS) as a scalable source of rigorous proof problems, where algorithmic definitions enable automated generation of arbitrarily many challenging theorem-proof pairs. We demonstrate this approach on two TCS domains: Busy Beaver problems, which involve proving bounds on Turing machine halting behavior, and Mixed Boolean Arithmetic problems, which combine logical and arithmetic reasoning. Our framework automatically synthesizes problems with parallel formal (Lean4) and informal (Markdown) specifications, creating a scalable pipeline for generating verified proof challenges. Evaluation on frontier models reveals substantial gaps in automated theorem proving: while DeepSeekProver-V2-671B achieves 57.5\% success on Busy Beaver problems, it manages only 12\% on Mixed Boolean Arithmetic problems. These results highlight the difficulty of long-form proof generation even for problems that are computationally easy to verify, demonstrating the value of TCS domains for advancing automated reasoning research.
>
---
#### [replaced 063] Beyond LoRA vs. Full Fine-Tuning: Gradient-Guided Optimizer Routing for LLM Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型微调任务，解决FFT与LoRA性能瓶颈问题。提出MoLF框架，动态融合两者优势，提升模型适应性。**

- **链接: [https://arxiv.org/pdf/2605.07111](https://arxiv.org/pdf/2605.07111)**

> **作者:** Haozhan Tang; Xiuqi Zhu; Xinyin Zhang; Boxun Li; Virginia Smith; Kevin Kuo
>
> **摘要:** Recent literature on fine-tuning Large Language Models highlights a fundamental debate. While Full Fine-Tuning (FFT) provides the representational plasticity required for high-entropy knowledge injection, Low-Rank Adaptation (LoRA) can match or surpass FFT performance because many tasks only require updates in a low-rank space and benefit from LoRA's additional regularization. Through empirical evaluation across diverse tasks (SQL, Medical QA, and Counterfactual Knowledge) and varying language models (Gemma-3-1B, Qwen2.5-1.5B, and Qwen2.5-3B), we verify both trends and demonstrate that relying solely on either static architecture is structurally limited. To address this challenge, we propose a Mixture of LoRA and Full (MoLF) Fine-Tuning, a unified framework that enables continuous navigation between both training regimes. MoLF dynamically routes updates between FFT and LoRA at the optimizer level to ensure that exact gradient signals are available to both experts throughout training, yielding stable training dynamics. For memory-constrained environments, we also introduce MoLF-Efficient, which freezes base weights and only routes updates among a pair of LoRA experts of potentially varying rank. Our evaluations show that MoLF either improves on or stays within $1.5\%$ of the better of FFT and LoRA across all settings, while MoLF-Efficient outperforms prior adaptive LoRA approaches by up to $20\%$ on Fact and $9\%$ on Med and SQL. Our code is open-sourced at this https URL.
>
---
#### [replaced 064] Auditing Agent Harness Safety
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于安全审计任务，旨在解决LLM代理在执行过程中可能违反权限和信息流约束的问题。工作包括提出HarnessAudit框架和基准测试，评估多代理系统的安全性。**

- **链接: [https://arxiv.org/pdf/2605.14271](https://arxiv.org/pdf/2605.14271)**

> **作者:** Chengzhi Liu; Yichen Guo; Yepeng Liu; Yuzhe Yang; Qianqi Yan; Xuandong Zhao; Wenyue Hua; Sheng Liu; Sharon Li; Yuheng Bu; Xin Eric Wang
>
> **备注:** 11 Pages, 8 Figures
>
> **摘要:** LLM agents increasingly run inside execution harnesses that dispatch tools, allocate resources, and route messages between specialized components. However, a harness can return a correct, benign answer over a trajectory that accesses unauthorized resources or leaks context to the wrong agent. Output-level evaluation cannot see these failures, yet most safety benchmarks score only final outputs or terminal states, even though many violations occur mid-trajectory rather than at termination. The central question is whether the harness respects user intent, permission boundaries, and information-flow constraints throughout execution. To address this gap, we propose HarnessAudit, a framework that audits full execution trajectories across boundary compliance, execution fidelity, and system stability, with a focus on multi-agent harnesses where these risks are most pronounced. We further introduce HarnessAudit-Bench, a benchmark of 210 tasks across eight real-world domains, instantiated in both single-agent and multi-agent configurations with embedded safety constraints. Evaluating ten harness configurations across frontier models and three multi-agent frameworks, we find that: (i) task completion is misaligned with safe execution, and violations accumulate with trajectory length; (ii) safety risks vary across domains, task types, and agent roles; (iii) most violations concentrate in resource access and inter-agent information transfer; and (iv) multi-agent collaboration expands the safety risk surface, while harness design sets the upper bound of safe deployment.
>
---
#### [replaced 065] Difficulty-Based Preference Data Selection by DPO Implicit Reward Gap
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型对齐任务，旨在解决偏好数据选择效率低的问题。通过基于DPO隐式奖励差距的难度选择策略，提升数据效率和模型对齐效果。**

- **链接: [https://arxiv.org/pdf/2508.04149](https://arxiv.org/pdf/2508.04149)**

> **作者:** Xuan Qi; Rongwu Xu; Zhijing Jin
>
> **备注:** Our code and data are available at this https URL
>
> **摘要:** Aligning large language models (LLMs) with human preferences is a critical challenge in AI research. While methods like Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) are widely used, they often rely on large, costly preference datasets. The current work lacks methods for high-quality data selection specifically for preference data. In this work, we introduce a novel difficulty-based data selection strategy for preference datasets, grounded in the DPO implicit reward mechanism. By selecting preference data examples with smaller DPO implicit reward gaps, which are indicative of more challenging cases, we improve data efficiency and model alignment. Our approach consistently outperforms five strong baselines across multiple datasets and alignment tasks, achieving superior performance with only 10\% of the original data. This principled, efficient selection method offers a promising solution for scaling LLM alignment with limited resources.
>
---
#### [replaced 066] Mixture-of-Experts Can Surpass Dense LLMs Under Strictly Equal Resource
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究MoE模型在相同资源下是否优于密集模型。通过优化设计和实验验证，证明MoE可在同等参数、计算和数据条件下超越密集模型。**

- **链接: [https://arxiv.org/pdf/2506.12119](https://arxiv.org/pdf/2506.12119)**

> **作者:** Houyi Li; Ka Man Lo; Shijie Xuyang; Ziqi Wang; Wenzhen Zheng; Haocheng Zhang; Zhao Li; Shuigeng Zhou; Xiangyu Zhang; Daxin Jiang
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** Mixture-of-Experts (MoE) language models dramatically expand model capacity and achieve remarkable performance without increasing per-token compute. However, can MoEs surpass dense architectures under strictly equal resource constraints -- that is, when the total parameter count, training compute, and data budget are identical? This question remains under-explored despite its significant practical value and potential. In this paper, we propose a novel perspective and methodological framework to study this question thoroughly. First, we comprehensively investigate the architecture of MoEs and achieve an optimal model design that maximizes the performance. Based on this, we subsequently find that an MoE model with activation rate in an optimal region is able to outperform its dense counterpart under the same total parameter, training compute and data resource. More importantly, this optimal region remains consistent across different model sizes. Although additional amount of data turns out to be a trade-off for enhanced performance, we show that this can be resolved via reusing data. We validate our findings through extensive experiments, training nearly 200 language models at 2B scale and over 50 at 7B scale, cumulatively processing 50 trillion tokens. All model checkpoints are publicly available.
>
---
#### [replaced 067] Fix the Structural Bottleneck: Context Compression via Explicit Information Transmission
- **分类: cs.CL**

- **简介: 该论文属于上下文压缩任务，解决长上下文大模型中token、内存和延迟成本过高的问题。提出ComprExIT框架，通过显式信息传输提升压缩效果。**

- **链接: [https://arxiv.org/pdf/2602.03784](https://arxiv.org/pdf/2602.03784)**

> **作者:** Jiangnan Ye; Hanqi Yan; Zhenyi Shen; Heng Chang; Ye Mao; Yulan He
>
> **摘要:** Long-context LLM agents often struggle with growing token, memory, and latency costs, making efficient context compression essential for practical deployment. Existing LLM-as-a-compressor methods remain noticeably inferior to using the full context. We find that this gap partly stems from their inability to preserve contextual information effectively. In this work, we revisit context compression from a structural perspective and identify two key bottlenecks in standard LLM-based compressors: limited coordination among compression tokens during information aggregation, and layerwise dilution that weakens useful signals from intermediate hidden states. To address these limitations, we propose ComprExIT, a new context compression framework based on explicit information transmission. ComprExIT adaptively selects features across frozen LLM layers, then allocates information from anchors to compression slots through a globally coordinated transport plan. Experiments on 12 datasets show that ComprExIT consistently outperforms strong soft-compression baselines, improving average F1 by up to 18.5%, while adding only ~1% trainable parameters and achieving more than 2x faster compression than the fastest baselines. The code will be released upon acceptance.
>
---
#### [replaced 068] Beyond Neural Incompatibility: Cross-Scale Knowledge Transfer in Language Models through Latent Semantic Alignment
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于知识迁移任务，解决跨尺度语言模型参数知识转移问题。通过激活作为媒介，利用语义对齐实现有效知识传递。**

- **链接: [https://arxiv.org/pdf/2510.24208](https://arxiv.org/pdf/2510.24208)**

> **作者:** Jian Gu; Aldeida Aleti; Chunyang Chen; Hongyu Zhang
>
> **备注:** an early-stage version
>
> **摘要:** Language Models (LMs) encode substantial knowledge in their parameters, yet it remains unclear how to transfer such knowledge in a fine-grained manner, namely parametric knowledge transfer (PKT). A central challenge is to make cross-scale transfer effective and efficient when source and target models differ in architecture and parameterization, making direct parameter reuse strongly limited by neural incompatibility. In this paper, we identify latent semantic alignment as the key prerequisite for cross-scale knowledge transfer. Instead of directly moving layer parameters, our approach uses activations as the transfer medium. \textsc{SemAlign} has two stages: an \emph{layer attribution} stage that attributes task-relevant source layers and selects exactly one source layer for each target layer, and a \emph{semantic alignment} stage that pairs them layer by layer and optimizes the target with source-side semantic supervision. The alignment is carried out in latent space through semantic decomposition and recomposition. During the shallow-to-deep transfer, only the frontier target layer is trainable. The layer objective supervises the residual contribution of that layer by matching centered token-token relation geometry against an aligned supervisory residual, while output KL preserves source-level predictive behavior. The transferred medium is therefore neither a parameter block nor an absolute hidden state, but target-space residual geometry induced by paired source-layer supervision. Evaluations on four benchmarks demonstrate the efficacy of \textsc{SemAlign}, and further analysis confirms that semantic decomposition and recomposition provide a stable mechanism for cross-scale knowledge transfer.
>
---
#### [replaced 069] From Chatbots to Confidants: A Cross-Cultural Study of LLM Adoption for Emotional Support
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于跨文化研究任务，旨在探讨LLM在情感支持中的使用情况及影响因素。通过调查7个国家4641名参与者，分析用户对LLM情感支持的接受度及其社会经济和文化背景的影响。**

- **链接: [https://arxiv.org/pdf/2604.25525](https://arxiv.org/pdf/2604.25525)**

> **作者:** Natalia Amat-Lefort; Mert Yazan; Amanda Cercas Curry; Flor Miriam Plaza-del-Arco
>
> **备注:** 28 pages (9 pages main text, 19 pages references and appendices), 14 figures. The first two authors contributed equally
>
> **摘要:** Large Language Models (LLMs) are increasingly used not only for instrumental tasks, but as always-available and non-judgmental confidants for emotional support. Yet what drives adoption and how users perceive emotional support interactions across countries remains unknown. To address this gap, we present the first large-scale cross-cultural study of LLM use for emotional support, surveying 4,641 participants across seven countries (USA, UK, Germany, France, Spain, Italy, and The Netherlands). Our results show that adoption rates vary dramatically across countries (from 20% to 59%). Using mixed models that separate cultural effects from demographic composition, we find that: Being aged 25-44, religious, married, and of higher socioeconomic status are predictors of positive perceptions (trust, usage, perceived benefits), with socioeconomic status being the strongest. English-speaking countries consistently show more positive perceptions than Continental European countries. We further collect a corpus of 731 real multilingual prompts from user interactions, showing that users mainly seek help for loneliness, stress, relationship conflicts, and mental health struggles. Our findings reveal that LLM emotional support use is shaped by a complex sociotechnical landscape and call for a broader research agenda examining how these systems can be developed, deployed, and governed to ensure safe and informed access.
>
---
#### [replaced 070] Trustworthiness in Retrieval-Augmented Generation Systems: A Survey
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决RAG系统可信度不足的问题。通过构建评估框架和基准，分析并提升RAG系统的可靠性。**

- **链接: [https://arxiv.org/pdf/2409.10102](https://arxiv.org/pdf/2409.10102)**

> **作者:** Yujia Zhou; Wenbo Zhang; Jingying Shao; Yan Liu; Xiaoxi Li; Jiajie Jin; Hongjin Qian; Zheng Liu; Chaozhuo Li; Jason Chen Zhang; Zhicheng Dou; Philip S. Yu; Jiaxin Mao
>
> **摘要:** Retrieval-Augmented Generation (RAG) has quickly grown into a pivotal paradigm in the development of Large Language Models (LLMs). Although existing research mainly emphasizes accuracy and efficiency, the trustworthiness of RAG systems remains insufficiently explored. RAG can improve LLM reliability by grounding responses in external and up-to-date knowledge, reducing hallucinations. However, unreliable retrieval or improper knowledge utilization may still lead to undesirable outputs. To address these concerns, we propose a unified framework, Trust-RAG Compass, that assesses the trustworthiness of RAG systems across six key dimensions: factuality, robustness, fairness, transparency, accountability, and privacy. Within this framework, we provide a thorough review of the existing literature along each dimension. Furthermore, we introduce an evaluation benchmark, TRC Bench (\underline{T}rust-\underline{R}AG \underline{C}ompass \underline{Bench}mark), regarding the six dimensions and conduct comprehensive evaluations for a variety of proprietary and open-source models. Our results shed light on the performance gaps between different types of LLMs across varying dimensions of trustworthiness. Finally, we identify key challenges and promising directions for future research based on our findings. Through this work, we aim to provide a structured foundation for subsequent investigations and practical guidance for developing trustworthy RAG systems in real-world scenarios.
>
---
#### [replaced 071] Traces of Social Competence in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的社会认知研究任务，旨在评估大语言模型的理论心智能力。通过改进的False Belief Test，分析模型规模、训练方式对社会认知表现的影响。**

- **链接: [https://arxiv.org/pdf/2603.04161](https://arxiv.org/pdf/2603.04161)**

> **作者:** Tom Kouwenhoven; Michiel van der Meer; Max van Duijn
>
> **备注:** Presented at the 2026 Conference on Computational Natural Language Learning (CoNLL)
>
> **摘要:** The False Belief Test (FBT) has been the main method for assessing Theory of Mind (ToM) and related socio-cognitive competencies. For Large Language Models (LLMs), the reliability and explanatory potential of this test have remained limited due to issues like data contamination, insufficient model details, and inconsistent controls. We address these issues by testing 17 open-weight models on a balanced set of 192 FBT variants (Trott et al., 2023) using Bayesian Logistic regression to identify how model size and post-training affect socio-cognitive competence. We find that scaling model size benefits performance, but not strictly. A cross-over effect reveals that explicating propositional attitudes (X thinks) fundamentally alters response patterns. Instruction tuning partially mitigates this effect, but further reasoning-oriented fine-tuning amplifies it. In a case study analysing social reasoning ability throughout OLMo 2 training, we show that this cross-over effect emerges during pre-training, suggesting that models acquire stereotypical response patterns tied to mental-state vocabulary that can outweigh other scenario semantics. Finally, vector steering allows us to isolate a think vector as the causal driver of observed FBT behaviour.
>
---
#### [replaced 072] Can Language Models Identify Side Effects of Breast Cancer Radiation Treatments?
- **分类: cs.CL**

- **简介: 该论文属于医学信息处理任务，旨在评估语言模型识别乳腺癌放疗副作用的能力，解决临床沟通与数据碎片化问题，通过实验分析模型性能并提出改进方案。**

- **链接: [https://arxiv.org/pdf/2605.08439](https://arxiv.org/pdf/2605.08439)**

> **作者:** Natalie Seah; Danielle S. Bitterman; Daphna Spiegel; Thomas Hartvigsen
>
> **摘要:** Accurately communicating the side effects of cancer treatments to cancer survivors is critical, particularly in settings such as informed consent, where clinicians must clearly and comprehensively convey potential treatment toxicities. However, this task remains challenging due to clinical knowledge deficits about adverse treatment effects and fragmentation across electronic health record (EHR) systems. Large language models (LLMs) have the potential to assist in this task, though their reliability in oncology survivorship contexts remains poorly understood. We present a deployment-oriented stress-testing framework for evaluating LLM-generated radiation side effect lists in breast cancer treatment and survivorship care. Using 21 breast cancer patient profiles, we construct paired patient clinical scenarios that differ only in radiotherapy regimens to evaluate seven instruction-tuned LLMs under multiple prompting regimes. We then compare LLM outputs to a clinician-curated reference derived from informed consent documents at two major academic medical centers and developed by a team including more than seven breast radiation oncologists. The reference maps radiation dose-fractionation, fields, and locations to associated toxicities, broken down by frequency and temporal onset. Across models, we reveal sensitivity to minor documentation changes, trade-offs between precision and recall, and systematic under-recall of rare and long-term side effects. When used alone, constraints on the number of side effects generated reduce precision, and grounding outputs in clinician-curated side effect lists substantially improves reliability and robustness. These findings highlight important limitations of LLM use in oncology and suggest practical design choices for safer and more informative survivorship-focused applications.
>
---
#### [replaced 073] AuthorMix: Modular Authorship Style Transfer via Layer-wise Adapter Mixing
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于作者风格迁移任务，旨在在保持原意的前提下转换文本风格。提出AuthorMix框架，通过模块化适配器实现高效、灵活的风格迁移。**

- **链接: [https://arxiv.org/pdf/2603.23069](https://arxiv.org/pdf/2603.23069)**

> **作者:** Sarubi Thillainathan; Ji-Ung Lee; Michael Sullivan; Alexander Koller
>
> **备注:** Under review
>
> **摘要:** The task of authorship style transfer involves rewriting text in the style of a target author while preserving the meaning of the original text. Existing style transfer methods train a single model on large corpora to model all target styles at once: this high-cost approach offers limited flexibility for target-specific adaptation, and often sacrifices meaning preservation for style transfer. In this paper, we propose AuthorMix: a lightweight, modular, and interpretable style transfer framework. We train individual, style-specific LoRA adapters on a small set of high-resource authors, allowing the rapid training of specialized adaptation models for each new target via learned, layer-wise adapter mixing, using only a handful of target style training examples. AuthorMix outperforms existing, SoTA style-transfer baselines -- as well as GPT-5.1 -- for low-resource targets, achieving the highest overall score and substantially improving meaning preservation.
>
---
#### [replaced 074] SignRoundV2: Toward Closing the Performance Gap in Extremely Low-Bit Post-Training Quantization for LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型量化任务，旨在解决低比特量化导致的性能下降问题。提出SignRoundV2框架，通过混合精度策略和稳定技术，提升低比特压缩下的模型性能。**

- **链接: [https://arxiv.org/pdf/2512.04746](https://arxiv.org/pdf/2512.04746)**

> **作者:** Wenhua Cheng; Weiwei Zhang; Heng Guo; Haihao Shen; Zaner Ma
>
> **摘要:** Extremely low-bit quantization is critical for efficiently deploying Large Language Models (LLMs), yet it often leads to severe performance degradation at 2 bits and even at 4 bits (e.g., MXFP4). We present SignRoundV2, a post-training quantization framework designed to maintain high performance even under aggressive compression. SignRoundV2 introduces (1) a simple yet efficient adaptive mixed-precision strategy that leverages gradient information and quantization-induced reconstruction errors to guide layer-wise bit allocation, and (2) a set of lightweight stabilization techniques, including loss filtering and a pre-tuning scale search, to improve tuning effectiveness in extremely low-bit regimes. Our approach takes a significant step toward closing the performance gap between quantized and full-precision models. Experimental results across diverse LLMs demonstrate that SignRoundV2 achieves near-lossless performance in mixed MXFP settings, narrowing the gap to $\sim$1\% at an average of 4.5 bits, while substantially improving accuracy in challenging 2-bit weight-only quantization. The source code is available at \url{this https URL}.
>
---
#### [replaced 075] The Frequency Confound in Language-Model Surprisal and Metaphor Novelty
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，探讨语言模型 surprisal 与隐喻新颖性之间的关系。研究发现词频比 surprisal 更能预测隐喻新颖性，揭示了 surprisal 与词频的复杂关联。**

- **链接: [https://arxiv.org/pdf/2605.06506](https://arxiv.org/pdf/2605.06506)**

> **作者:** Omar Momen; Sina Zarrieß
>
> **备注:** to be presented and published at the 15th Joint Conference on Lexical and Computational Semantics (*SEM 2026)
>
> **摘要:** Language-model (LM) surprisal is widely used as a proxy for contextual predictability and has been reported to correlate with metaphor novelty judgments. However, surprisal is tightly intertwined with lexical frequency. We explore this interaction on metaphor novelty ratings using two different word frequency measures. We analyse surprisal estimates from eight Pythia model sizes and 154 training checkpoints. Across settings, word frequency is a stronger predictor of metaphor novelty than surprisal. Across training stages, the surprisal--novelty association peaks at an early stage and then falls again, mirroring a similarly timed increase in the surprisal--frequency association. These results suggest that the often-reported optimal LM surprisal settings may incorrectly associate contextual predictability with metaphor novelty and processing difficulty, whereas lexical frequency may be the major underlying factor.
>
---
#### [replaced 076] ToolMATH: A Diagnostic Benchmark for Long-Horizon Tool Use under Systematic Tool-Catalog Constraints
- **分类: cs.CL; cs.LG; cs.SE**

- **简介: 该论文提出ToolMATH，用于评估语言模型在工具约束下的长期工具使用能力，解决工具适应性、鲁棒性和连通性问题。**

- **链接: [https://arxiv.org/pdf/2602.21265](https://arxiv.org/pdf/2602.21265)**

> **作者:** Hyeonje Choi; Jeongsoo Lee; Hyojun Lee; Jay-Yoon Lee
>
> **备注:** Submitted to NeurIPS Evaluation & Dataset Track
>
> **摘要:** We introduce \ToolMATH, a math-grounded diagnostic benchmark for evaluating long-horizon tool use under controllable tool-catalog conditions. \ToolMATH converts stepwise MATH solutions into reusable Python tools with natural-language descriptions and typed schemas, and pairs each problem with a tool environment requiring sequential tool use, intermediate-output reuse, and logically connected tool-call chains. \ToolMATH controls tool availability and catalog difficulty by constructing gold tools and graded distractors with varying similarity to gold tools. \ToolMATH also incorporates behavior-conditioned metrics, enabling diagnostic evaluation beyond final accuracy. Building on these measurements, \ToolMATH emphasizes three evaluation axes: (1) \emph{Adaptability} measures how much Gold-only success is retained when gold tools are replaced entirely by distractors; (2) \emph{Robustness} measures stability under adding distractors as a noise; and (3) \emph{Tool Connectivity} measures whether models preserve accuracy over long executed tool-call chains. Furthermore, trace-level failure analyses characterize how models fail under each tool-catalog condition. Together, these diagnostics reveal distinct model profiles: reliable tool use, tool avoidance, adaptive substitution, and impacts of unreliable tool catalogs. Overall, \ToolMATH provides a controlled testbed for evaluating how language models adapt to changing tool availability, remain robust to distractors, and maintain correctness across long-horizon tool-use trajectories.
>
---
#### [replaced 077] A Survey of On-Policy Distillation for Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于知识蒸馏任务，旨在解决长序列生成中的暴露偏差问题。通过重新设计训练流程，将蒸馏转化为迭代修正过程，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.00626](https://arxiv.org/pdf/2604.00626)**

> **作者:** Mingyang Song; Mao Zheng
>
> **备注:** Ongoing Work
>
> **摘要:** As Large Language Models (LLMs) continue to grow in both capability and cost, transferring frontier capabilities into smaller, deployable students has become a central engineering problem, and knowledge distillation remains the dominant technique for this transfer. The prevailing recipe in industrial pipelines, static imitation of teacher-generated text, carries a structural weakness that grows more severe as tasks become longer and more reasoning-intensive. Because the student is trained on flawless teacher prefixes but must generate its own at inference, small errors tend to accumulate into trajectories it has rarely been trained to recover from, and the resulting exposure bias has been shown to scale roughly with the square of sequence length. On-Policy Distillation (OPD) reorganizes the training loop around this observation by having the teacher provide feedback on what the student actually produces, with the goal of reducing the compounding term toward linear and reframing distillation as an iterative correction process rather than single-pass imitation. The resulting literature has expanded along divergence design, reward-guided optimization, and self-play, yet contributions remain scattered across the knowledge distillation, RLHF, and imitation learning communities without a unified treatment. This survey provides such a treatment. We formalize OPD as $f$-divergence minimization over student-sampled trajectories, organize the field along three design axes (what to optimize, where the signal comes from, and how to stabilize training in practice), and consolidate success conditions, recurring failure modes, and the connection between OPD and KL-constrained RL. We close with open problems that emerge from this synthesis, including distillation scaling laws, uncertainty-aware feedback, agentic distillation, and the growing overlap between knowledge distillation and RL.
>
---
#### [replaced 078] UbuntuGuard: A Culturally-Grounded Policy Benchmark for Equitable AI Safety in African Languages
- **分类: cs.CL**

- **简介: 该论文属于AI安全任务，旨在解决非洲语言在安全模型中的文化不适应问题。提出UbuntuGuard基准，通过专家生成的查询构建本地化安全策略。**

- **链接: [https://arxiv.org/pdf/2601.12696](https://arxiv.org/pdf/2601.12696)**

> **作者:** Tassallah Abdullahi; Macton Mgonzo; Mardiyyah Oduwole; Paul Okewunmi; Abraham Owodunni; Ritambhara Singh; Carsten Eickhoff
>
> **备注:** 15 pages
>
> **摘要:** Current guardian models are predominantly Western-centric and optimized for high-resource languages, leaving low-resource African languages vulnerable to evolving harms, cross-lingual failures, and cultural misalignment. Moreover, most guardian models rely on rigid, predefined safety categories that fail to generalize across diverse linguistic and sociocultural contexts. Achieving robust safety requires flexible, runtime-enforceable policies and benchmarks that reflect local norms, harm scenarios, and cultural expectations. We introduce UbuntuGuard, the first policy-based safety benchmark for African languages built from adversarial queries authored by 155 domain experts across sensitive fields, including healthcare. From these expert-crafted queries, we derive context-specific safety policies and reference responses that capture culturally grounded risk signals, enabling policy-aligned evaluation of guardian models. We evaluate 15 models, comprising seven general-purpose LLMs and eight guardian models across three distinct variants: static, dynamic, and multilingual. Our findings reveal that existing English-centric benchmarks overestimate real-world multilingual safety, cross-lingual transfer provides partial but insufficient coverage, and dynamic models, while better equipped to leverage policies at inference time, still struggle to fully localize African-language contexts. These findings highlight the urgent need for multilingual, culturally grounded safety benchmarks to enable the development of reliable and equitable guardian models for low-resource languages. Our benchmark and code can be found online.\footnote{Dataset and code repository available at \url{this https URL}.}
>
---
#### [replaced 079] Red-Bandit: Test-Time Adaptation for LLM Red-Teaming via Bandit-Guided LoRA Experts
- **分类: cs.CL**

- **简介: 该论文提出Red-Bandit框架，用于LLM红队测试中的实时适应，解决模型特定漏洞识别问题，通过LoRA专家和强化学习生成安全提示。**

- **链接: [https://arxiv.org/pdf/2510.07239](https://arxiv.org/pdf/2510.07239)**

> **作者:** Christos Ziakas; Nicholas Loo; Nishita Jain; Alessandra Russo
>
> **备注:** Accepted to the Main Conference at ACL 2026
>
> **摘要:** Automated red-teaming has emerged as a scalable approach for auditing Large Language Models (LLMs) prior to deployment, yet existing approaches lack mechanisms to efficiently adapt to model-specific vulnerabilities at inference. We introduce Red-Bandit, a red-teaming framework that adapts online to identify and exploit model failure modes under distinct attack styles (e.g., manipulation, slang). Red-Bandit post-trains a set of parameter-efficient LoRA experts, each specialized for a particular attack style, using reinforcement learning that rewards the generation of unsafe prompts via a rule-based safety model. At inference, a multi-armed bandit policy dynamically selects among these attack-style experts based on the target model's response safety, balancing exploration and exploitation. Red-Bandit achieves state-of-the-art results on AdvBench under sufficient exploration (ASR@10), while producing more human-readable prompts (lower perplexity). Moreover, Red-Bandit's bandit policy serves as a diagnostic tool for uncovering model-specific vulnerabilities by indicating which attack styles most effectively elicit unsafe behaviors.
>
---
#### [replaced 080] ExpThink: Experience-Guided Reinforcement Learning for Adaptive Chain-of-Thought Compression
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决大模型推理过程中的高消耗和低效率问题。通过引入经验引导的强化学习框架，提升推理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.07501](https://arxiv.org/pdf/2605.07501)**

> **作者:** Tingcheng Bian; Yuzhe Zhang; Jing Jin; Jinchang Luo; MingQuan Cheng; Haiwei Wang; Wenyuan Jiang; Miaohui Wang
>
> **备注:** 39 pages, 18 figures. Code and model checkpoints will be released upon publication
>
> **摘要:** Large reasoning models (LRMs) achieve strong performance via extended chain-of-thought (CoT) reasoning, yet suffer from excessive token consumption and high inference latency. Existing reinforcement learning (RL) approaches for CoT compression rely on uniform, static length penalties that neglect model capability dynamics and problem-level difficulty variation. We propose \textbf{ExpThink}\xspace, an RL framework that addresses both dimensions through two complementary mechanisms. First, \emph{experience-guided reward shaping} tracks the shortest correct solution found so far for each problem and applies a three-tier reward: full credit for concise correct responses, discounted credit for verbose correct ones, and zero for incorrect ones. The threshold tightens automatically with model improvement, forming a self-evolving curriculum that requires no manual scheduling. Second, \emph{difficulty-adaptive advantage} replaces standard deviation normalization with correct-count normalization, yielding monotonically difficulty-scaled gradients that amplify learning on hard problems to preserve accuracy while suppressing gradients on easy ones to encourage brevity. Together, these mechanisms enforce an accuracy-first, compression-second training objective. Experiments on multiple mathematical reasoning benchmarks demonstrate that \textbf{ExpThink}\xspace reduces average response length by up to 77\% while simultaneously improving accuracy, achieving up to $3\times$ higher accuracy-efficiency ratio (accuracy divided by average token count) than the vanilla baseline and outperforming existing RL-based compression methods on both metrics.
>
---
#### [replaced 081] Information-Theoretic Storage Cost in Sentence Comprehension
- **分类: cs.CL**

- **简介: 该论文属于语言理解任务，旨在解决句法预测对工作记忆负荷的量化问题。提出基于信息论的存储成本度量方法，通过神经语言模型估计，验证其有效性。**

- **链接: [https://arxiv.org/pdf/2602.18217](https://arxiv.org/pdf/2602.18217)**

> **作者:** Kohei Kajikawa; Shinnosuke Isono; Ethan Gotlieb Wilcox
>
> **备注:** Accepted to CoNLL 2026
>
> **摘要:** Real-time sentence comprehension imposes a significant load on working memory, as comprehenders must maintain contextual information to anticipate future input. While measures of such load have played an important role in psycholinguistic theories, they have largely been formalized using symbolic grammars, which assign discrete, uniform costs to syntactic predictions. This study proposes a measure of processing storage cost based on an information-theoretic formalization, as the amount of information previous words carry about future context, under uncertainty. Unlike previous discrete, grammar-based metrics, this measure is continuous, probabilistic, theory-neutral, and can be estimated from pre-trained neural language models. The validity of this approach is demonstrated through three analyses in English: our measure (i) recovers well-known processing asymmetries in center embeddings and relative clauses, (ii) correlates with a grammar-based storage cost in a syntactically-annotated corpus, and (iii) predicts reading-time variance in two large-scale naturalistic datasets over and above baseline models with traditional information-based predictors. Our code is available at this https URL.
>
---
#### [replaced 082] Beyond Pattern Matching: Seven Cross-Domain Techniques for Prompt Injection Detection
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于提示注入检测任务，旨在解决现有方法的不足，提出七种跨领域检测技术，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.18248](https://arxiv.org/pdf/2604.18248)**

> **作者:** Thamilvendhan Munirathinam
>
> **备注:** v3.0 (18 May 2026): Added Sec. 5.6 with independent evaluation on three peer-reviewed benchmarks (Liu, USENIX Sec 2024; Garak, Derczynski 2024; InjecAgent, ACL Findings 2024). 8,276 unseen attacks; cross-benchmark plateau at 35-45% on subtle indirect injection. Abstract, contributions, Sec. 6, and 6 refs updated
>
> **摘要:** Current open-source prompt-injection detectors converge on two architectural choices: regular-expression pattern matching and fine-tuned transformer classifiers. Both share failure modes that recent work has made concrete. Regular expressions miss paraphrased attacks. Fine-tuned classifiers are vulnerable to adaptive adversaries: a 2025 NAACL Findings study reported that eight published indirect-injection defenses were bypassed with greater than fifty percent attack success rates under adaptive attacks. This work proposes seven detection techniques that each port a specific mechanism from a discipline outside large-language-model security: forensic linguistics, materials-science fatigue analysis, deception technology from network security, local-sequence alignment from bioinformatics, mechanism design from economics, spectral signal analysis from epidemiology, and taint tracking from compiler theory. Three of the seven techniques are implemented in the prompt-shield v0.4.1 release (Apache 2.0) and evaluated in a four-configuration ablation across six datasets including deepset/prompt-injections, NotInject, LLMail-Inject, AgentHarm, and AgentDojo. The local-alignment detector lifts F1 on deepset from 0.033 to 0.378 with zero additional false positives. The stylometric detector adds 11.1 percentage points of F1 on an indirect-injection benchmark. The fatigue tracker is validated via a probing-campaign integration test. All code, data, and reproduction scripts are released under Apache 2.0.
>
---
#### [replaced 083] Beyond Superficial Unlearning: Sharpness-Aware Robust Erasure of Hallucinations in Multimodal LLMs
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态大模型的幻觉消除任务，解决模型产生虚假实体的问题。通过SARE方法，实现鲁棒的幻觉擦除，提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2601.16527](https://arxiv.org/pdf/2601.16527)**

> **作者:** Xianya Fang; Feiyang Ren; Xiang Chen; Yu Tian; Zhen Bi; Haiyang Yu; Sheng-Jun Huang
>
> **摘要:** Multimodal LLMs are powerful but prone to object hallucinations, which describe non-existent entities and harm reliability. While recent unlearning methods attempt to mitigate this, we identify a critical flaw: structural fragility. We empirically demonstrate that standard erasure achieves only superficial suppression, trapping the model in sharp minima where hallucinations catastrophically resurge after lightweight relearning. To ensure geometric stability, we propose SARE, which casts unlearning as a targeted min-max optimization problem and uses a Targeted-SAM mechanism to explicitly flatten the loss landscape around hallucinated concepts. By suppressing hallucinations under simulated worst-case parameter perturbations, our framework ensures robust removal stable against weight shifts. Extensive experiments demonstrate that SARE significantly outperforms baselines in erasure efficacy while preserving general generation quality. Crucially, it maintains persistent hallucination suppression against relearning and parameter updates, validating the effectiveness of geometric stabilization.
>
---
#### [replaced 084] When TableQA Meets Noise: A Dual Denoising Framework for Complex Questions and Large-scale Tables
- **分类: cs.CL**

- **简介: 该论文属于TableQA任务，旨在解决复杂问题和大规模表格中的噪声干扰问题。提出EnoTab框架，通过双重去噪提升推理性能。**

- **链接: [https://arxiv.org/pdf/2509.17680](https://arxiv.org/pdf/2509.17680)**

> **作者:** Shenghao Ye; Yu Guo; Dong Jin; Yikai Shen; Yunpeng Hou; Shuangwu Chen; Jian Yang; Xiaofeng Jiang
>
> **备注:** 24 pages, 24 figures, accepted to ACL 2026 Main
>
> **摘要:** Table question answering (TableQA) is a fundamental task in natural language processing (NLP). The strong reasoning capabilities of large language models (LLMs) have brought significant advances in this field. However, as real-world applications involve increasingly complex questions and larger tables, substantial noisy data is introduced, which severely degrades reasoning performance. To address this challenge, we focus on improving two core capabilities: Relevance Filtering, which identifies and retains information truly relevant to reasoning, and Table Pruning, which reduces table size while preserving essential content. Based on these principles, we propose EnoTab, a dual denoising framework for complex questions and large-scale tables. Specifically, we first perform Evidence-based Question Denoising by decomposing the question into minimal semantic units and filtering out those irrelevant to answer reasoning based on consistency and usability criteria. Then, we propose Evidence Tree-guided Table Denoising, which constructs an explicit and transparent table pruning path to remove irrelevant data step by step. At each pruning step, we observe the intermediate state of the table and apply a post-order node rollback mechanism to handle abnormal table states, ultimately producing a highly reliable sub-table for final answer reasoning. Finally, extensive experiments show that EnoTab achieves outstanding performance on TableQA tasks with complex questions and large-scale tables, confirming its effectiveness.
>
---
#### [replaced 085] MULTITEXTEDIT: Benchmarking Cross-Lingual Degradation in Text-in-Image Editing
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于文本图像编辑任务，旨在解决跨语言文本编辑中的语义与视觉一致性问题。通过构建多语言基准数据集，评估模型在不同语言中的表现差异。**

- **链接: [https://arxiv.org/pdf/2605.08163](https://arxiv.org/pdf/2605.08163)**

> **作者:** Liwei Cheng; Shibo Feng; Lunjie Zhou; Yixuan Guan; Dayan Guan
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Text-in-image editing has become a key capability for visual content creation, yet existing benchmarks remain overwhelmingly English-centric and often conflate visual plausibility with semantic correctness. We introduce MULTITEXTEDIT, a controlled benchmark of 3,600 instances spanning 12 typologically diverse languages, 5 visual domains, and 7 editing operations. Language variants of each instance share a common visual base and are paired with a human-edited reference and region masks, isolating the language variable for cross-lingual comparison. To capture script-level errors that coarse text-matching metrics miss, such as missing diacritics, reversed RTL order, and mixed-script renderings, we introduce a language fidelity (LSF) metric scored by a two-stage LVM protocol that first traces the edited target text and then judges it in isolation, reaching a quadratic-weighted \k{appa} of 0.76 against native-speaker annotators. Evaluating 12 open-source and proprietary systems with LSF alongside standard semantic and mask-aware pixel metrics, we find pronounced cross-lingual degradation for every model, largest on Hebrew and Arabic and smallest on Dutch and Spanish, and concentrated in text accuracy and script fidelity rather than in coarse structural dimensions. We also uncover a pervasive semantic and pixel mismatch, where outputs preserve global layout and background fidelity yet distort script-specific forms.
>
---
#### [replaced 086] KASER: Knowledge-Aligned Student Error Simulator for Open-Ended Coding Tasks
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **简介: 该论文提出KASER模型，用于模拟开放编码任务中的学生错误。解决LLM在生成学生错误时多样性不足的问题，通过强化学习提升错误匹配与代码多样性。**

- **链接: [https://arxiv.org/pdf/2601.06633](https://arxiv.org/pdf/2601.06633)**

> **作者:** Zhangqi Duan; Nigel Fernandez; Andrew Lan
>
> **备注:** Published in ACL 2026: The 64th Annual Meeting of the Association for Computational Linguistics
>
> **摘要:** Open-ended tasks, such as coding problems that are common in computer science education, provide detailed insights into student knowledge. However, training large language models (LLMs) to simulate and predict possible student errors in their responses to these problems can be challenging: they often suffer from mode collapse and fail to fully capture the diversity in syntax, style, and solution approach in student responses. In this work, we present KASER (Knowledge-Aligned Student Error Simulator), a novel approach that aligns errors with student knowledge. We propose a training method based on reinforcement learning using a hybrid reward that reflects three aspects of student code prediction: i) code similarity to the ground-truth, ii) error matching, and iii) code prediction diversity. On two real-world datasets, we perform two levels of evaluation and show that: At the per-student-problem pair level, our method outperforms baselines on code and error prediction; at the per-problem level, our method outperforms baselines on error coverage and simulated code diversity.
>
---
#### [replaced 087] Mitigating Extrinsic Gender Bias for Bangla Classification Tasks
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决 Bangla 预训练模型中的外在性别偏见问题。通过构建基准数据集并提出 RandSymKL 方法进行去偏，提升分类任务的公平性与准确性。**

- **链接: [https://arxiv.org/pdf/2411.10636](https://arxiv.org/pdf/2411.10636)**

> **作者:** Sajib Kumar Saha Joy; Arman Hassan Mahy; Meherin Sultana; Azizah Mamun Abha; MD Piyal Ahmmed; Yue Dong; G M Shahariar
>
> **摘要:** In this study, we investigate extrinsic gender bias in Bangla pretrained language models, a largely underexplored area in low-resource languages. To assess this bias, we construct four manually annotated, task-specific benchmark datasets for sentiment analysis, toxicity detection, hate speech detection, and sarcasm detection. Each dataset is augmented using nuanced gender perturbations, where we systematically swap gendered names and terms while preserving semantic content, enabling minimal-pair evaluation of gender-driven prediction shifts. We then propose RandSymKL, a randomized debiasing strategy integrated with symmetric KL divergence and cross-entropy loss to mitigate the bias across task-specific pretrained models. RandSymKL is a refined training approach to integrate these elements in a unified way for extrinsic gender bias mitigation focused on classification tasks. Our approach was evaluated against existing bias mitigation methods, with results showing that our technique not only effectively reduces bias but also maintains competitive accuracy compared to other baseline approaches. To promote further research, we have made both our implementation and datasets publicly available: this https URL
>
---
#### [replaced 088] Automated Coding of Communication Data Using ChatGPT: Consistency Across Subgroups
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决AI在通信数据编码中是否具有一致性的问题。通过测试ChatGPT在不同群体中的表现，验证其可靠性。**

- **链接: [https://arxiv.org/pdf/2510.20584](https://arxiv.org/pdf/2510.20584)**

> **作者:** Jiangang Hao; Wenju Cui; Patrick Kyllonen; Emily Kerzabi
>
> **备注:** Accepted to the Journal of Educational Measurement
>
> **摘要:** Assessing communication and collaboration at scale depends on a labor-intensive task of coding communication data into categories according to different frameworks. Prior research has established that ChatGPT can be directly instructed with coding rubrics to code the communication data and achieves accuracy comparable to human raters. However, whether the coding from ChatGPT or similar AI technology perform consistently across different demographic groups, such as gender and race, remains unclear. To address this gap, we introduce three checks for evaluating subgroup consistency in LLM-based coding by adapting an existing framework from the automated scoring literature. Using a typical collaborative problem-solving coding framework and data from three types of collaborative tasks, we examine ChatGPT-based coding performance across gender and racial/ethnic groups. Our results show that ChatGPT-based coding perform consistently in the same way as human raters across gender or racial/ethnic groups, demonstrating the possibility of its use in large-scale assessments of collaboration and communication.
>
---
#### [replaced 089] PEGRL: Improving Machine Translation by Post-Editing Guided Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决强化学习在翻译中的噪声信号和优化难题。提出PEGRL框架，通过后编辑辅助任务稳定训练并提升效果。**

- **链接: [https://arxiv.org/pdf/2602.03352](https://arxiv.org/pdf/2602.03352)**

> **作者:** Yunzhi Shen; Hao Zhou; Xin Huang; Xue Han; Junlan Feng; Shujian Huang
>
> **摘要:** Reinforcement learning (RL) has shown strong promise for LLM-based machine translation, with recent methods such as GRPO demonstrating notable gains; nevertheless, translation-oriented RL remains challenged by noisy learning signals arising from Monte Carlo return estimation, as well as a large trajectory space that favors global exploration over fine-grained local optimization. We introduce \textbf{PEGRL}, a \textit{two-stage} RL framework that uses post-editing as an auxiliary task to stabilize training and guide overall optimization. At each iteration, translation outputs are sampled to construct post-editing inputs, allowing return estimation in the post-editing stage to benefit from conditioning on the current translation behavior, while jointly supporting both global exploration and fine-grained local optimization. A task-specific weighting scheme further balances the contributions of translation and post-editing objectives, yielding a biased yet more sample-efficient estimator. Experiments on English$\to$Finnish, English$\to$Turkish, and English$\leftrightarrow$Chinese show consistent gains over RL baselines, and for English$\to$Turkish, performance on COMET-KIWI is comparable to advanced LLM-based systems (DeepSeek-V3.2). Our code and a set of representative pretrained models are publicly available at \url{this https URL} and \url{this https URL}
>
---
#### [replaced 090] Disentangling Ambiguity from Instability in Large Language Models: A Clinical Text-to-SQL Case Study
- **分类: cs.CL**

- **简介: 该论文属于临床文本到SQL任务，解决模型输出多样性来源的区分问题，提出CLUES框架，通过分解不确定性提升错误预测与干预效率。**

- **链接: [https://arxiv.org/pdf/2602.12015](https://arxiv.org/pdf/2602.12015)**

> **作者:** Angelo Ziletti; Leonardo D'Ambrosi
>
> **摘要:** Deploying large language models for clinical Text-to-SQL requires distinguishing two qualitatively different causes of output diversity: (i) input ambiguity that should trigger clarification, and (ii) model instability that should trigger human review. We propose CLUES, a framework that models Text-to-SQL as a two-stage process (interpretations --> answers) and decomposes semantic uncertainty into an ambiguity score and an instability score. The instability score is computed via the Schur complement of a bipartite semantic graph matrix. Across AmbigQA/SituatedQA (gold interpretations) and a clinical Text-to-SQL benchmark (known interpretations), CLUES improves failure prediction over state-of-the-art Kernel Language Entropy. In deployment settings, it remains competitive while providing a diagnostic decomposition unavailable from a single score. The resulting uncertainty regimes map to targeted interventions - query refinement for ambiguity, model improvement for instability. The high-ambiguity/high-instability regime contains 51% of errors while covering 25% of queries, enabling efficient triage.
>
---
#### [replaced 091] Sparse-to-Dense: A Free Lunch for Lossless Acceleration of Video Understanding in LLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对视频理解任务中的推理延迟问题，提出Sparse-to-Dense策略，在不损失性能的前提下提升视频大模型的处理速度。**

- **链接: [https://arxiv.org/pdf/2505.19155](https://arxiv.org/pdf/2505.19155)**

> **作者:** Xuan Zhang; Cunxiao Du; Sicheng Yu; Jiawei Wu; Fengzhuo Zhang; Wei Gao; Qian Liu
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Due to the auto-regressive nature of current video large language models (Video-LLMs), the inference latency increases as the input sequence length grows, posing challenges for the efficient processing of video sequences that are usually very long. We observe that during decoding, the attention scores of most tokens in Video-LLMs tend to be sparse and concentrated, with only certain tokens requiring comprehensive full attention. Based on this insight, we introduce Sparse-to-Dense (StD), a novel decoding strategy that integrates two distinct modules: one leveraging sparse top-K attention and the other employing dense full attention. These modules collaborate to accelerate Video-LLMs without loss. The fast (sparse) model speculatively decodes multiple tokens, while the slow (dense) model verifies them in parallel. StD is a tuning-free, plug-and-play solution that achieves up to a 1.94$\times$ walltime speedup in video processing. It maintains model performance while enabling a seamless transition from a standard Video-LLM to a sparse Video-LLM with minimal code modifications.
>
---
#### [replaced 092] QuCo-RAG: Quantifying Uncertainty from the Pre-training Corpus for Dynamic Retrieval-Augmented Generation
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出QuCo-RAG，解决LLM生成中的幻觉问题。通过预训练语料统计量化不确定性，动态触发检索，提升问答任务效果。**

- **链接: [https://arxiv.org/pdf/2512.19134](https://arxiv.org/pdf/2512.19134)**

> **作者:** Dehai Min; Kailin Zhang; Tongtong Wu; Lu Cheng
>
> **备注:** ACL Findings 2026
>
> **摘要:** Dynamic Retrieval-Augmented Generation adaptively determines when to retrieve during generation to mitigate hallucinations in large language models (LLMs). However, existing methods rely on model-internal signals (e.g., logits, entropy), which are fundamentally unreliable because LLMs are typically ill-calibrated and often exhibit high confidence in erroneous outputs. We propose QuCo-RAG, which shifts from subjective confidence to objective statistics computed from pre-training data. Our method quantifies uncertainty through two stages: (1) before generation, we identify low-frequency entities indicating long-tail knowledge gaps; (2) during generation, we verify entity co-occurrence in the pre-training corpus, where zero co-occurrence often signals hallucination risk. Both stages leverage Infini-gram for millisecond-latency queries over 4 trillion tokens, triggering retrieval when uncertainty is high. Experiments on multi-hop QA benchmarks show QuCo-RAG achieves EM gains of 5--12 points over state-of-the-art baselines with OLMo-2 models, and transfers effectively to models with undisclosed pre-training data (Llama-3, Qwen2.5, GPT-4.1/5-chat), improving EM by up to 14 points. Generalization to long-form generation and biomedical QA further validates the robustness of our paradigm. These results establish corpus-grounded verification as a principled, practically model-agnostic paradigm for dynamic RAG. Our code is publicly available at this https URL.
>
---
#### [replaced 093] Natural-Language Agent Harnesses
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出自然语言代理框架，解决代理系统难以分析和复用的问题。通过可编辑的自然语言文档描述任务流程，实现更简洁、可分析的代理执行。属于人工智能任务。**

- **链接: [https://arxiv.org/pdf/2603.25723](https://arxiv.org/pdf/2603.25723)**

> **作者:** Linyue Pan; Lexiao Zou; Shuo Guo; Jingchen Ni; Hai-Tao Zheng
>
> **备注:** revise paper
>
> **摘要:** Agent performance is strongly shaped by the surrounding harness: the external execution system around a model that organizes a task run. Yet this logic is usually buried in tightly coupled controller code, which makes harnesses hard to inspect, compare, transfer, and ablate. This paper asks whether the reusable design pattern of an agent harness can be represented as an executable natural-language object. We introduce Natural-Language Agent Harnesses (NLAHs), editable documents that describe run-level harness policy, and Intelligent Harness Runtime (IHR), a shared runtime that interprets these documents into agent calls, handoffs, state updates, validation gates, and artifact contracts. Across coding, terminal-use, and computer-use benchmarks, IHR-executed NLAHs achieve comparable task outcomes to code and prompted realizations, while exposing much shorter static harness policies. Module ablations further show that explicit harness modules are analyzable. These results suggest that agent harnesses can be turned from incidental glue around models into scientific representation objects.
>
---
#### [replaced 094] White-Box Sensitivity Auditing with Steering Vectors
- **分类: cs.CY; cs.CL; cs.LG**

- **简介: 该论文属于AI模型审计任务，旨在解决传统黑盒评估无法全面检测模型偏见的问题。通过白盒敏感性审计框架，利用激活控制进行内部测试，揭示模型对受保护属性的依赖。**

- **链接: [https://arxiv.org/pdf/2601.16398](https://arxiv.org/pdf/2601.16398)**

> **作者:** Hannah Cyberey; Yangfeng Ji; David Evans
>
> **摘要:** Algorithmic audits are essential tools for examining systems for properties required by regulators or desired by operators. Current audits of large language models (LLMs) primarily rely on black-box evaluations that assess model behavior only through input-output testing. These methods are limited to tests constructed in the input space, often generated by heuristics. In addition, many socially relevant model properties (e.g., gender bias) are abstract and difficult to measure through text-based inputs alone. To address these limitations, we propose a white-box sensitivity auditing framework for LLMs that leverages activation steering to conduct more rigorous assessments through model internals. Our auditing method conducts internal sensitivity tests by manipulating key concepts relevant to the model's intended function for the task. We demonstrate its application to bias audits in four simulated high-stakes LLM decision tasks. Our method consistently indicates substantial dependence on protected attributes in model predictions, even in settings where standard black-box evaluations suggest little or no bias. Our code is openly available at this https URL
>
---
#### [replaced 095] CarbonScaling: Extending Neural Scaling Laws for Carbon Footprint in Large Language Models
- **分类: cs.CL; cs.AI; cs.CY; cs.DC; cs.LG**

- **简介: 该论文属于AI可持续性研究，解决大模型训练碳排放估算问题。提出CarbonScaling框架，整合硬件与训练策略，更准确预测碳足迹。**

- **链接: [https://arxiv.org/pdf/2508.06524](https://arxiv.org/pdf/2508.06524)**

> **作者:** Lei Jiang; Fan Chen
>
> **备注:** 8 pages
>
> **摘要:** Large language models (LLMs) increasingly follow neural scaling laws that tie performance gains to rapidly expanding computational budgets, raising concerns about the sustainability of frontier-scale training. Existing carbon-estimation methods largely depend on regression over historical runs and fail to capture critical system-level factors, including hardware heterogeneity, distributed parallelism, communication overhead, and architectural sparsity. We present \textit{CarbonScaling}, a hardware-aware analytical framework for modeling the carbon scaling behavior of frontier LLM training. The framework integrates neural scaling laws, distributed training strategies, accelerator and interconnect modeling, and operational and embodied carbon accounting to estimate feasible hardware configurations and associated emissions. CarbonScaling jointly models tensor, pipeline, data, and expert parallelism while incorporating memory, bandwidth, utilization, and runtime constraints. Experimental validation demonstrates substantially higher fidelity than regression-based baselines and highlights the growing importance of embodied carbon at trillion-parameter scales. Source code: \url{this https URL}.
>
---
#### [replaced 096] The Expert Strikes Back: Interpreting Mixture-of-Experts Language Models at Expert Level
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究MoE模型的可解释性问题，通过对比专家与密集网络，发现专家更少多义，提出以专家为单位进行解释，验证其在细粒度任务上的专业化。**

- **链接: [https://arxiv.org/pdf/2604.02178](https://arxiv.org/pdf/2604.02178)**

> **作者:** Jeremy Herbst; Stefan Wermter; Jae Hee Lee
>
> **备注:** 8 pages, 7 Figures. Accepted at ICML 2026. Improved writing, changed author order, updated citations
>
> **摘要:** Mixture-of-Experts (MoE) architectures have become the dominant choice for scaling Large Language Models (LLMs), activating only a subset of parameters per token. While MoE architectures are primarily adopted for computational efficiency, it remains an open question whether their sparsity makes them inherently easier to interpret than dense feed-forward networks (FFNs). We compare MoE experts and dense FFNs using $k$-sparse probing and find that expert neurons are consistently less polysemantic, with the gap widening as routing becomes sparser. This suggests that sparsity pressures both individual neurons and entire experts toward monosemanticity. Leveraging this finding, we zoom out from the neuron to the expert level as a more effective unit of analysis. We validate this approach by automatically interpreting hundreds of experts. This analysis allows us to resolve the debate on specialization: experts are neither broad domain specialists (e.g., biology) nor simple token-level processors. Instead, they function as fine-grained task experts, specializing in linguistic operations or semantic tasks (e.g., closing brackets in $\LaTeX{}$). Our findings suggest that MoEs are inherently interpretable at the expert level, providing a clearer path toward large-scale model interpretability. Code is available at: this https URL.
>
---
#### [replaced 097] LLM-Oriented Information Retrieval: A Denoising-First Perspective
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决LLM在检索过程中因噪声导致的错误问题。通过提出去噪框架和优化技术，提升信息可用性和可验证性。**

- **链接: [https://arxiv.org/pdf/2605.00505](https://arxiv.org/pdf/2605.00505)**

> **作者:** Lu Dai; Liang Sun; Fanpu Cao; Ziyang Rao; Cehao Yang; Hao Liu; Hui Xiong
>
> **备注:** SIGIR 2026
>
> **摘要:** Modern information retrieval (IR) is no longer consumed primarily by humans but increasingly by large language models (LLMs) via retrieval-augmented generation (RAG) and agentic search. Unlike human users, LLMs are constrained by limited attention budgets and are uniquely vulnerable to noise; misleading or irrelevant information is no longer just a nuisance, but a direct cause of hallucinations and reasoning failures. In this perspective paper, we argue that denoising-maximizing usable evidence density and verifiability within a context window-is becoming the primary bottleneck across the full information access pipeline. We conceptualize this paradigm shift through a four-stage framework of IR challenges: from inaccessible to undiscoverable, to misaligned, and finally to unverifiable. Furthermore, we provide a pipeline-organized taxonomy of signal-to-noise optimization techniques, spanning indexing, retrieval, context engineering, verification, and agentic workflow. We also present research works on information denoising in domains that rely heavily on retrieval such as lifelong assistant, coding agent, deep research, and multimodal understanding.
>
---
#### [replaced 098] AutoLLMResearch: Training Research Agents for Automating LLM Experiment Configuration - Learning from Cheap, Optimizing Expensive
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自动化配置任务，旨在解决高成本LLM实验配置难题。通过构建多保真度环境和训练框架，实现高效、可泛化的配置优化。**

- **链接: [https://arxiv.org/pdf/2605.11518](https://arxiv.org/pdf/2605.11518)**

> **作者:** Taicheng Guo; Nitesh V. Chawla; Olaf Wiest; Xiangliang Zhang
>
> **摘要:** Effectively configuring scalable large language model (LLM) experiments, spanning architecture design, hyperparameter tuning, and beyond, is crucial for advancing LLM research, as poor configuration choices can waste substantial computational resources and prevent models from realizing their full potential. Prior automated methods are designed for low-cost settings where repeated trial and error is feasible, but scalable LLM experiments are too expensive for such extensive iteration. To our knowledge, no work has addressed the automation of high-cost LLM experiment configurations, leaving this problem labor-intensive and dependent on expert intuition. Motivated by this gap, we propose AutoLLMResearch, an agentic framework that mimics how human researchers learn generalizable principles from low-fidelity experiments and extrapolate to efficiently identify promising configurations in expensive LLM settings. The core challenge is how to enable an agent to learn, through interaction with a multi-fidelity experimental environment that captures the structure of the LLM configuration landscape. To achieve this, we propose a systematic framework with two key components: 1) LLMConfig-Gym, a multi-fidelity environment encompassing four critical LLM experiment tasks, supported by over one million GPU hours of verifiable experiment outcomes; 2) A structured training pipeline that formulates configuration research as a long-horizon Markov Decision Process and accordingly incentivizes cross-fidelity extrapolation reasoning. Extensive evaluation against diverse strong baselines on held-out experiments demonstrates the effectiveness, generalization, and interpretability of our framework, supporting its potential as a practical and general solution for scalable real-world LLM experiment automation.
>
---
#### [replaced 099] Beyond Accuracy: Decomposing the Reasoning Efficiency of LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大模型推理效率分析任务，旨在解决单一准确率无法反映模型实际效率的问题。通过分解token效率，评估模型在不同任务中的表现与失效模式。**

- **链接: [https://arxiv.org/pdf/2602.09805](https://arxiv.org/pdf/2602.09805)**

> **作者:** Daniel Kaiser; Arnoldo Frigessi; Ali Ramezani-Kebrya; Benjamin Ricaud
>
> **备注:** Preprint (under review). 29 pages, 4 figures
>
> **摘要:** As reasoning LLMs increasingly trade tokens for accuracy through deliberation, search, and self-correction, a single accuracy score can no longer tell whether those tokens buy useful reasoning, recovery from hard instances, or unnecessary verbosity. We introduce a trace-optional evaluation protocol that exactly decomposes token efficiency using three observables available even for closed models: completion rate, conditional correctness given completion, and generated length. When instance-level workload metadata is available, we further normalize generated length by declared task-implied work and separate mean verbalization overhead from workload-dependent scaling. When such metadata is absent, we define an auditable solver-derived workload scale and evaluate its stability under leave-self-out, leave-top-k, and held-out-reference-pool perturbations. We evaluate 14 shared open-weight models on CogniLoad, GSM8K, ProofWriter, and ZebraLogic. We further evaluate 11 additional models on CogniLoad, enabling a fine-grained analysis of reasoning-task difficulty factors: task length, intrinsic difficulty, and distractor density. Efficiency and overhead rankings remain stable across all benchmark pairs, more robustly than accuracy rankings, while the decomposition separates logic-limited, context-limited (truncation-driven), and verbosity-limited failure modes that look identical under accuracy-per-token. We release an evaluation artifact and reporting template, which elaborates on why an LLM is inefficient at reasoning.
>
---
#### [replaced 100] Rethinking 1-bit Optimization Leveraging Pre-trained Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于模型量化任务，旨在解决1-bit LLM训练成本高、精度下降的问题。通过渐进式训练和优化策略，提升1-bit LLM性能。**

- **链接: [https://arxiv.org/pdf/2508.06974](https://arxiv.org/pdf/2508.06974)**

> **作者:** Zhijun Tu; Jian Li; Yuanyuan Xi; Siqi Liu; Chuanjian Liu; Hanting Chen; Jie Hu; Yunhe Wang
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** 1-bit LLM quantization offers significant advantages in reducing storage and computational costs. However, existing methods typically train 1-bit LLMs from scratch, failing to fully leverage pre-trained models. This results in high training costs and notable accuracy degradation. We identify that the large gap between full precision and 1-bit representations makes naive adaptation difficult. In this paper, we introduce a consistent progressive training for both forward and backward, smoothly converting the full-precision weights into the binarized ones. Additionally, we incorporate binary-aware initialization and dual-scaling compensation to reduce the difficulty of progressive training and improve the performance. Experimental results on LLMs of various sizes demonstrate that our method outperforms existing approaches. Our results show that high-performance 1-bit LLMs can be achieved using pre-trained models, eliminating the need for expensive training from scratch.
>
---
#### [replaced 101] Dual-Space Knowledge Distillation with Key-Query Matching for Large Language Models with Vocabulary Mismatch
- **分类: cs.CL**

- **简介: 该论文属于知识蒸馏任务，解决不同分词器模型间的分布不匹配问题。通过引入生成对抗学习，提升跨模型知识迁移效果。**

- **链接: [https://arxiv.org/pdf/2603.22056](https://arxiv.org/pdf/2603.22056)**

> **作者:** Stella Eva Tsiapali; Cong-Thanh Do; Kate Knill
>
> **备注:** Copyright 2026 IEEE. Published in ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), scheduled for 4-8 May 2026 in Barcelona, Spain
>
> **摘要:** Large language models (LLMs) achieve state-of-the-art (SOTA) performance across language tasks, but are costly to deploy due to their size and resource demands. Knowledge Distillation (KD) addresses this by training smaller Student models to mimic larger Teacher models, improving efficiency without significant performance loss. Dual-Space Knowledge Distillation with Cross-Model Attention (DSKD-CMA) has emerged as a SOTA method for KD between LLMs with distinct tokenizers, yet its internal workings remain largely opaque. In this work, we systematically analyse the attention mechanism of DSKD-CMA through manual token alignment probing and heatmap visualisations, revealing both strengths and limitations. Building on this, we introduce a novel method, DSKD-CMA-GA, based on Generative Adversarial (GA) learning, to address the mismatched distributions between the keys and queries computed from distinct models. Experiments show modest but consistent ROUGE-L gains in text generation quality, particularly on out-of-distribution data (+0.37 on average), narrowing the gap between cross- and same-tokenizer KD.
>
---
#### [replaced 102] Surgical Post-Training: Proximal On-Policy Distillation for Reasoning with Knowledge Retention
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识注入任务，解决LLM后训练中的灾难性遗忘问题。提出SPOT框架，通过数据修正和奖励优化，提升推理能力并保留原有知识。**

- **链接: [https://arxiv.org/pdf/2603.01683](https://arxiv.org/pdf/2603.01683)**

> **作者:** Wenye Lin; Kai Han
>
> **备注:** 21 pages
>
> **摘要:** Injecting new reasoning knowledge into Large Language Models (LLMs) via post-training often induces catastrophic forgetting. Recent studies emphasize the importance of on-policy data but suggest that KL-divergence fails to mitigate forgetting. In contrast, we show, both analytically and empirically, that the KL-constrained reward formulation actually plays a critical role in retaining knowledge during post-training. This motivates our Surgical Post-Training (SPOT), a proximal on-policy distillation framework designed to optimize reasoning efficiently while preserving prior knowledge. SPOT consists of (1) a data rectification pipeline employing an Oracle to surgically correct erroneous steps via minimal edits, generating proximal on-policy data; and (2) a reward-based binary cross-entropy objective essential for enhancing reasoning and mitigating forgetting. Empirically, with only 4k rectified math pairs, SPOT improves Qwen3-8B's accuracy by 6.2% on average across in-domain and out-of-domain tasks, requiring merely 16-minute model training on 8x H800 GPUs. Moreover, SPOT provides a superior initialization for subsequent reinforcement learning, significantly elevating the performance ceiling. Code: this https URL
>
---
#### [replaced 103] Answer Only as Precisely as Justified: Calibrated Claim-Level Specificity Control for Agentic Systems
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决agentic系统过度精确的问题。通过引入CSS方法，控制每个声明的特定性，提升不确定性表达效果。**

- **链接: [https://arxiv.org/pdf/2604.17487](https://arxiv.org/pdf/2604.17487)**

> **作者:** Tianyi Huang; Samuel Xu; Jason Tansong Dang; Samuel Yan; Kimberley Yin
>
> **备注:** Accepted at the ICML 2026 Workshop on Statistical Frameworks for Uncertainty in Agentic Systems
>
> **摘要:** Agentic systems often fail not by being entirely wrong, but by being too precise: a response may be generally useful while particular claims exceed what the evidence supports. We study this failure mode as overcommitment control and introduce compositional selective specificity (CSS), a post-generation layer that decomposes an answer into claims, proposes coarser backoffs, and emits each claim at the most specific calibrated level that appears admissible. The method is designed to express uncertainty as a local semantic backoff rather than as a whole-answer refusal. Across a full LongFact run and HotpotQA pilots, calibrated CSS improves the risk-utility trade-off of fixed drafts. On the full LongFact run, it raises overcommitment-aware utility from 0.846 to 0.913 relative to the no-CSS output while achieving 0.938 specificity retention. These results suggest that claim-level specificity control is a useful uncertainty interface for agentic systems and a target for future distribution-free validity layers.
>
---
#### [replaced 104] NaviRAG: Towards Active Knowledge Navigation for Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文提出NaviRAG，解决RAG在复杂任务中检索与合成信息的不足，通过主动知识导航提升性能。**

- **链接: [https://arxiv.org/pdf/2604.12766](https://arxiv.org/pdf/2604.12766)**

> **作者:** Jihao Dai; Dingjun Wu; Yuxuan Chen; Zheni Zeng; Yukun Yan; Zhenghao Liu; Maosong Sun
>
> **摘要:** Retrieval-augmented generation (RAG) typically relies on a flat retrieval paradigm that maps queries directly to static, isolated text segments. This approach struggles with more complex tasks that require the conditional retrieval and dynamic synthesis of information across different levels of granularity (e.g., from broad concepts to specific evidence). To bridge this gap, we introduce NaviRAG, a novel framework that shifts from passive segment retrieval to active knowledge navigation. NaviRAG first structures the knowledge documents into a hierarchical form, preserving semantic relationships from coarse-grained topics to fine-grained details. Leveraging this reorganized knowledge records, a large language model (LLM) agent actively navigates the records, iteratively identifying information gaps and retrieving relevant content from the most appropriate granularity level. Extensive experiments on long-document QA benchmarks show that NaviRAG consistently improves both retrieval recall and end-to-end answer performance over conventional RAG baselines. Ablation studies confirm performance gains stem from our method's capacity for multi-granular evidence localization and dynamic retrieval planning. We further discuss efficiency, applicable scenario, and future directions of our method, hoping to make RAG systems more intelligent and autonomous.
>
---
#### [replaced 105] Mistletoe: Stealthy Acceleration-Collapse Attacks on Speculative Decoding
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究对抗性攻击任务，针对推测解码中的机制漏洞，提出Mistletoe攻击方法，通过降低生成文本的接受率来削弱加速效果，同时保持输出质量。**

- **链接: [https://arxiv.org/pdf/2605.14005](https://arxiv.org/pdf/2605.14005)**

> **作者:** Shuoyang Sun; Chang Dai; Hao Fang; Kuofeng Gao; Xinhao Zhong; Yi Sun; Fan Mo; Shu-Tao Xia; Bin Chen
>
> **摘要:** Speculative decoding has become a widely adopted technique for accelerating large language model (LLM) inference by drafting multiple candidate tokens and verifying them with a target model in parallel. Its efficiency, however, critically depends on the average accepted length $\tau$, i.e., how many draft tokens survive each verification step. In this work, we identify a new mechanism-level vulnerability in model-based speculative decoding: the drafter is trained to approximate the target model distribution, but this approximation is inevitably imperfect. Such a drafter-target mismatch creates a hidden attack surface where small perturbations can preserve the target model's visible behavior while substantially reducing draft-token acceptability. We propose Mistletoe, a stealthy acceleration-collapse attack against speculative decoding. Mistletoe directly targets the acceptance mechanism of speculative decoding. It jointly optimizes a degradation objective that decreases drafter-target agreement and a semantic-preservation objective that constrains the target model's output distribution. To resolve the conflict between these objectives, we introduce a null-space projection mechanism, where degradation gradients are projected away from the local semantic-preserving direction, suppressing draft acceptance while minimizing semantic drift. Experiments on various speculative decoding systems show that Mistletoe substantially reduces average accepted length $\tau$, collapses speedup, and lowers averaged token throughput, while preserving output quality and perplexity. Our work highlights that speculative decoding introduces a mechanism-level attack surface beyond existing output robustness, calling for more robust designs of LLM acceleration systems.
>
---
#### [replaced 106] Fine-tuning vs. In-context Learning in Large Language Models: A Formal Language Learning Perspective
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型比较任务，旨在分析FT与ICL在语言掌握上的差异。通过形式化语言实验，评估两者在泛化能力和归纳偏向上表现。**

- **链接: [https://arxiv.org/pdf/2604.23267](https://arxiv.org/pdf/2604.23267)**

> **作者:** Bishwamittra Ghosh; Soumi Das; Till Speicher; Qinyuan Wu; Mohammad Aflah Khan; Deepak Garg; Krishna P. Gummadi; Evimaria Terzi
>
> **备注:** Accepted at ACL 2026 (Main)
>
> **摘要:** Large language models (LLMs) operate in two fundamental learning modes - fine-tuning (FT) and in-context learning (ICL) - raising key questions about which mode yields greater language proficiency and whether they differ in their inductive biases. Prior studies comparing FT and ICL have yielded mixed and inconclusive results due to inconsistent experimental setups. To enable a rigorous comparison, we propose a formal language learning task - offering precise language boundaries, controlled string sampling, and no data contamination - and introduce a discriminative test for language proficiency, where an LLM succeeds if it assigns higher generation probability to in-language strings than to out-of-language strings. Empirically, we find that: (a) FT has greater language proficiency than ICL on in-distribution generalization, but both perform equally well on out-of-distribution generalization. (b) Their inductive biases, measured by the correlation in string generation probabilities, are similar when both modes partially learn the language but diverge at higher proficiency levels. (c) Unlike FT, ICL performance differs substantially across models of varying sizes and families and is sensitive to the token vocabulary of the language. Thus, our work demonstrates the promise of formal languages as a controlled testbed for evaluating LLMs, behaviors that are difficult to isolate in natural language datasets. Our source code is available at this https URL.
>
---
#### [replaced 107] Factual Inconsistencies in Multilingual Wikipedia Tables
- **分类: cs.CL; cs.DB; cs.DL; cs.IR**

- **简介: 该论文属于多语言知识一致性研究任务，旨在解决Wikipedia中跨语言表格数据的事实不一致问题。通过构建方法分析多语言表格，评估其对AI系统的影响。**

- **链接: [https://arxiv.org/pdf/2507.18406](https://arxiv.org/pdf/2507.18406)**

> **作者:** Silvia Cappa; Lingxiao Kong; Pille-Riin Peet; Fanfu Wei; Yuchen Zhou; Jan-Christoph Kalo
>
> **备注:** 11 pages, 7 figures, White Paper for RTF Work at ISWS Summer School 2025
>
> **摘要:** Wikipedia serves as a globally accessible knowledge source with content in over 300 languages. Despite covering the same topics, the different versions of Wikipedia are written and updated independently. This leads to factual inconsistencies that can impact the neutrality and reliability of the encyclopedia and AI systems, which often rely on Wikipedia as a main training source. This study investigates cross-lingual inconsistencies in Wikipedia's structured content, with a focus on tabular data. We developed a methodology to collect, align, and analyze tables from Wikipedia multilingual articles, defining categories of inconsistency. We apply various quantitative and qualitative metrics to assess multilingual alignment using a sample dataset. These insights have implications for factual verification, multilingual knowledge interaction, and design for reliable AI systems leveraging Wikipedia content.
>
---
#### [replaced 108] DimMem: Dimensional Structuring for Efficient Long-Term Agent Memory
- **分类: cs.CL**

- **简介: 该论文提出DimMem，解决LLM代理长期记忆的结构化问题，通过显式维度表示提升记忆效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.15759](https://arxiv.org/pdf/2605.15759)**

> **作者:** Wentao Qiu; Haotian Hu; Fanyi Wang; Jinwei Kong; Yu Zhang
>
> **摘要:** Large language model (LLM) agents require long-term memory to leverage information from past interactions. However, existing memory systems often face a fidelity--efficiency trade-off: raw dialogue histories are expensive, while flat facts or summaries may discard the structure needed for precise recall. We propose \textbf{DimMem}, a lightweight dimensional memory framework that represents each memory as an atomic, typed, and self-contained unit with explicit fields such as time, location, reason, purpose, and keywords. This representation exposes the structure needed for dimension-aware retrieval, memory update, and selective assistant-context recall without storing full histories in the model context. Across LoCoMo-10 and LongMemEval-S, DimMem achieves \textbf{81.43\%} and \textbf{78.20\%} overall accuracy, respectively, outperforming existing lightweight memory systems while reducing LoCoMo per-query token cost by \textbf{24\%}. We further show that dimensional memory extraction is learnable by compact models: after fine-tuning on the DimMem schema, a Qwen3-4B extractor surpasses LightMem with GPT-4.1-mini on both benchmarks and reaches performance comparable to, or better than, much larger extractors in key settings. These results suggest that explicit dimensional structuring is an effective and efficient foundation for long-term memory in LLM agents. Code is available at this https URL.
>
---
#### [replaced 109] WriteSAE: Sparse Autoencoders for Recurrent State
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出WriteSAE，解决状态空间和混合循环语言模型中矩阵缓存写入的分解与编辑问题，通过稀疏自编码器实现精准控制。**

- **链接: [https://arxiv.org/pdf/2605.12770](https://arxiv.org/pdf/2605.12770)**

> **作者:** Jack Young
>
> **备注:** 26 pages, 14 figures, 21 tables; code at this https URL
>
> **摘要:** We introduce WriteSAE, the first sparse autoencoder that decomposes and edits the matrix cache write of state-space and hybrid recurrent language models, where residual SAEs cannot reach. Existing SAEs read residual streams, but Gated DeltaNet, Mamba-2, and RWKV-7 write to a $d_k \times d_v$ cache through rank-1 updates $k_t v_t^\top$ that no vector atom can replace. WriteSAE factors each decoder atom into the native write shape, exposes a closed form for the per-token logit shift, and trains under matched Frobenius norm so atoms swap one cache slot at a time. Atom substitution beats matched-norm ablation on 92.4% of $n=4{,}851$ firings at Qwen3.5-0.8B L9 H4, the 87-atom population test holds at 89.8%, the closed form predicts measured effects at $R^2=0.98$, and Mamba-2-370M substitutes at 88.1% over 2,500 firings. Sustained three-position installs at $3\times$ lift midrank target-in-continuation from 33.3% to 100% under greedy decoding, the first behavioral install at the matrix-recurrent write site.
>
---
#### [replaced 110] StructLens: A Structural Lens for Language Models via Maximum Spanning Trees
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出StructLens，用于分析语言模型的结构组织。任务是理解模型如何组织token表示，通过最大生成树揭示其内部结构。**

- **链接: [https://arxiv.org/pdf/2603.03328](https://arxiv.org/pdf/2603.03328)**

> **作者:** Haruki Sakajo; Frederikus Hudi; Yusuke Sakai; Hidetaka Kamigaito; Taro Watanabe
>
> **摘要:** Language exhibits inherent structures, a property that explains both language acquisition and language change. Given this characteristic, we expect language models to manifest their own internal structures as well. While interpretability research has investigated how models compute representations mechanistically through attention patterns and Sparse AutoEncoders, the organization of the resulting representations is overlooked. To address this gap, we introduce StructLens, a framework to analyze representations through a holistic structural view. StructLens constructs maximum spanning trees based on the semantic representations in residual streams, inspired by tree representation in dependency parsing, and provides summaries of token relationships in representation space. We analyze how contiguous tokens are also nearby in representation space and find that middle layers show the strongest local-span organization. Moreover, analysis of pre-training checkpoints reveals that smaller local units become detectable earlier in pre-training, and larger units later. Our findings demonstrate that StructLens provides insights into how models organize token representations across layers and training. Our code is available at this https URL.
>
---
#### [replaced 111] DocReward: A Document Reward Model for Structuring and Stylizing
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出DocReward，解决文档结构与风格专业性评估问题，通过构建数据集并训练模型提升文档质量。**

- **链接: [https://arxiv.org/pdf/2510.11391](https://arxiv.org/pdf/2510.11391)**

> **作者:** Junpeng Liu; Yuzhong Zhao; Bowen Cao; Jiayu Ding; Yilin Jia; Tengchao Lv; Yupan Huang; Wenshan Wu; Shaohan Huang; Nan Yang; Li Dong; Lei Cui; Tao Ge; Xun Wang; Huitian Jiao; Sun Mao; FNU Kartik; Si-Qing Chen; Wai Lam; Furu Wei
>
> **摘要:** Recent agentic workflows automate professional document generation but focus narrowly on textual quality, overlooking structural and stylistic professionalism, which is equally critical for readability. This gap stems mainly from a lack of effective reward models capable of guiding agents toward producing documents with high structural and stylistic professionalism. We introduce DocReward, a document reward model that evaluates documents based on their structure and style. To achieve this, we propose a textual-quality-agnostic framework that ensures assessments are not confounded by content quality, and construct DocPair, a dataset of 117K paired documents covering 32 domains and 267 types. Each pair shares identical content but differs in structural and stylistic professionalism. DocReward is trained using the Bradley-Terry loss. On a manually annotated benchmark, DocReward outperforms GPT-5 by 14.6 percentage points in the same setting. Reinforcement learning experiments further show that DocReward effectively guides agents toward generating documents with consistently higher structural and stylistic professionalism, highlighting its practical utility.
>
---
#### [replaced 112] Can LLMs Generate and Solve Linguistic Olympiad Puzzles?
- **分类: cs.CL**

- **简介: 该论文研究LLMs生成与解决语言奥赛谜题的能力，属于自然语言处理任务。旨在解决如何利用LLMs提升语言学习兴趣及推广冷门语言。工作包括扩展基准、测试模型表现并探索自动出题方法。**

- **链接: [https://arxiv.org/pdf/2509.21820](https://arxiv.org/pdf/2509.21820)**

> **作者:** Neh Majmudar; Elena Filatova
>
> **备注:** Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)
>
> **摘要:** In this paper, we introduce a combination of novel and exciting tasks: the solution and generation of linguistic puzzles. We focus on puzzles used in Linguistic Olympiads for high school students. We first extend the existing benchmark for the task of solving linguistic puzzles. We explore the use of Large Language Models (LLMs), including recent state-of-the-art models such as OpenAI's o1, for solving linguistic puzzles, analyzing their performance across various linguistic topics. We demonstrate that LLMs outperform humans on most puzzles types, except for those centered on writing systems, and for the understudied languages. We use the insights from puzzle-solving experiments to direct the novel task of puzzle generation. We believe that automating puzzle generation, even for relatively simple puzzles, holds promise for expanding interest in linguistics and introducing the field to a broader audience. This finding highlights the importance of linguistic puzzle generation as a research task: such puzzles can not only promote linguistics but also support the dissemination of knowledge about rare and understudied languages.
>
---
#### [replaced 113] STS: Efficient Sparse Attention with Speculative Token Sparsity
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决大模型推理中的计算瓶颈问题。通过引入稀疏注意力机制STS，提升推理效率，同时保持高精度。**

- **链接: [https://arxiv.org/pdf/2605.15508](https://arxiv.org/pdf/2605.15508)**

> **作者:** Ceyu Xu; Jiangnan Yu; Yongji Wu; Yuan Xie
>
> **备注:** 14 pages, 12 figures
>
> **摘要:** The quadratic complexity of attention imposes severe memory and computational bottlenecks on Large Language Model (LLM) inference. This challenge is particularly acute for emerging agentic applications that require processing multi-million token sequences. We propose STS, a sparse attention mechanism that requires no model retraining. STS leverages the key insight that tokens identified as important by a smaller draft model are highly predictive of important tokens for a larger target model. By integrating into speculative decoding frameworks, STS repurposes the draft model's attention scores to dynamically construct a token-and-head-wise sparsity mask. This mask effectively prunes the expensive attention computation in the target LLM. Our evaluation shows that STS achieves a 2.67x speedup operating at approximately 90% sparsity on representative benchmark NarrativeQA, maintaining negligible accuracy degradation compared to dense attention. STS establishes a new state-of-the-art on the sparsity-accuracy trade-off, outperforming prior techniques by enabling higher sparsity levels for a given accuracy budget.
>
---
#### [replaced 114] Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection
- **分类: cs.CL**

- **简介: 该论文属于文本检测任务，旨在解决LLM生成文本的细粒度分类问题。提出RACE方法，通过分析创作者与编辑者角色，提升检测精度。**

- **链接: [https://arxiv.org/pdf/2604.04932](https://arxiv.org/pdf/2604.04932)**

> **作者:** Yang Li; Qiang Sheng; Zhengjia Wang; Yehan Yang; Danding Wang; Juan Cao
>
> **备注:** ACL 2026 (Oral)
>
> **摘要:** The misuse of large language models (LLMs) requires precise detection of synthetic text. Existing works mainly follow binary or ternary classification settings, which can only distinguish pure human/LLM text or collaborative text at best. This remains insufficient for the nuanced regulation, as the LLM-polished human text and humanized LLM text often trigger different policy consequences. In this paper, we explore fine-grained LLM-generated text detection under a rigorous four-class setting. To handle such complexities, we propose RACE (Rhetorical Analysis for Creator-Editor Modeling), a fine-grained detection method that characterizes the distinct signatures of creator and editor. Specifically, RACE utilizes Rhetorical Structure Theory (RST) to construct a logic graph for the creator's foundation while extracting Elementary Discourse Unit (EDU)-level features for the editor's style. Experiments show that RACE outperforms 12 baselines in identifying fine-grained types with low false alarms, offering a policy-aligned solution for LLM regulation.
>
---
#### [replaced 115] No Free Swap: Protocol-Dependent Layer Redundancy in Transformers
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究Transformer层冗余问题，解决层是否可压缩的判定难题。通过分析替换与交换两种协议的差异，提出基于KL散度的诊断方法，以评估层移除安全性。**

- **链接: [https://arxiv.org/pdf/2605.16234](https://arxiv.org/pdf/2605.16234)**

> **作者:** Gabriel Garcia
>
> **备注:** 40 pages, 8 figures, 24 tables. Code is available at this https URL
>
> **摘要:** When researchers ask whether two transformer layers are "equivalent" for compression, they often conflate distinct tests. Replacement asks whether one layer's map can substitute for another's in place; interchange asks whether two layers approximately commute when their positions are swapped. Both are output-grounded swap-KL probes, but they need not agree: on pretrained transformers the protocol gap can change which layers look safe to prune by several-fold under the same evaluator, especially when replacement distances are high. We measure both protocols across checkpoints and architectures. On a Pythia training trajectory (410M and 1.4B), the replacement-interchange gap grows from initialization to convergence. Under one matched WikiText-2 contract at 8B scale, Qwen3-8B enters a divergent regime: interchange-guided removal is several-fold safer than replacement-guided at the same layer budgets, while Llama-3.1-8B ties the two protocols for pruning cost even though interchange KL is lower, showing metric gaps need not map one-to-one to removal. Before layer removal or merging, score both swap-KLs on the target checkpoint; the diagnostic requires only unlabeled forward passes.
>
---
#### [replaced 116] SynCABEL: Synthetic Contextualized Augmentation for Biomedical Entity Linking
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文属于生物医学实体链接任务，解决标注数据稀缺问题。通过生成合成上下文数据，提升模型性能并减少对人工标注的依赖。**

- **链接: [https://arxiv.org/pdf/2601.19667](https://arxiv.org/pdf/2601.19667)**

> **作者:** Adam Remaki; Christel Gérardin; Eulàlia Farré-Maduell; Martin Krallinger; Xavier Tannier
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** We present SynCABEL (Synthetic Contextualized Augmentation for Biomedical Entity Linking), a framework that addresses a central bottleneck in supervised biomedical entity linking (BEL): the scarcity of expert-annotated training data. SynCABEL leverages large language models to generate context-rich synthetic training examples for all candidate concepts in a target knowledge base, providing broad supervision without manual annotation. We demonstrate that SynCABEL, when combined with decoder-only models and guided inference, establishes new state-of-the-art results across three widely used multilingual benchmarks: MedMentions for English, QUAERO for French, and SPACCC for Spanish. Evaluating data efficiency, we show that SynCABEL reaches the performance of full human supervision using up to 60% less annotated data, substantially reducing reliance on labor-intensive and costly expert labeling. Finally, acknowledging that standard evaluation based on exact code matching often underestimates clinically valid predictions due to ontology redundancy, we introduce an LLM-as-a-judge protocol. This analysis reveals that SynCABEL significantly improves the rate of clinically valid predictions. Our synthetic datasets, models, and code are released to support reproducibility and future research.
>
---
#### [replaced 117] Soohak: A Mathematician-Curated Benchmark for Evaluating Research-level Math Capabilities of LLMs
- **分类: cs.CL**

- **简介: 该论文提出Soohak基准，用于评估大语言模型的研究级数学能力，解决当前缺乏高质量数学基准的问题。**

- **链接: [https://arxiv.org/pdf/2605.09063](https://arxiv.org/pdf/2605.09063)**

> **作者:** Guijin Son; Seungone Kim; Catherine Arnett; Hyunwoo Ko; Hyein Lee; Hyeonah Kang; Jiang Longxi; Jin Yun; JungYup Lee; Kyungmin Lee; Sam Yoosuk Kim; Sang Park; Seunghyeok Hong; SeungJae Lee; Seungyeop Yi; Shinae Shin; SunHye Bok; Sunyoung Shin; Yonghoon Ji; Youngtaek Kim; Hanearl Jung; Akari Asai; Graham Neubig; Sean Welleck; Youngjae Yu; Akshelin R; Alexander B. Ivanov; Boboev Muhammadjon; Chaeyoung Han; Christian Stump; Dmitrii Karp; Dohyun Kwon; DoYong Kwon; Duk-Soon Oh; Giovanni Resta; Greta Panova; Huiyun Noh; Hyungryul Baik; Hyungsun Bae; Inomov Mashrafdzhon; Jeewon Kim; Ji Eun Lee; Jiaqi Liu; Jieui Kang; Jimin Kim; Jon-Lark Kim; Junseo Yoon; Junwoo Jo; Kibeom Kim; Kiwoon Kwon; Mario Kummer; Max Mercer; Minjun Kim; Nahyun Lee; Ng Ze-An; Rafał Marcin Łochowski; Raphaël Lachièze-Rey; Ruichen Zhang; Sejin Park; Seonguk Seo; Shin Jaehoon; Sunatullo; Taewoong Eom; Yeachan Park; Yongseok Jang; Youchan Oh; Zhaoyang Wang; Zoltán Kovács
>
> **备注:** Under review, For questions or model-evaluation requests, contact $this http URL@snu.this http URL$
>
> **摘要:** Following the recent achievement of gold-medal performance on the IMO by frontier LLMs, the community is searching for the next meaningful and challenging target for measuring LLM reasoning. Whereas olympiad-style problems measure step-by-step reasoning alone, research-level problems use such reasoning to advance the frontier of mathematical knowledge itself, emerging as a compelling alternative. Yet research-level math benchmarks remain scarce because such problems are difficult to source (e.g., Riemann Bench and FrontierMath-Tier 4 contain 25 and 50 problems, respectively). To support reliable evaluation of next-generation frontier models, we introduce Soohak, a 439-problem benchmark newly authored from scratch by 64 mathematicians. Soohak comprises two subsets. On the Challenge subset, frontier models including Gemini-3-Pro, GPT-5, and Claude-Opus-4.5 reach 30.4%, 26.4%, and 10.4% respectively, leaving substantial headroom, while leading open-weight models such as Qwen3-235B, GPT-OSS-120B, and Kimi-2.5 remain below 15%. Notably, beyond standard problem solving, Soohak introduces a refusal subset that probes a capability intrinsic to research mathematics: recognizing ill-posed problems and pausing rather than producing confident but unjustified answers. On this subset, no model exceeds 50%, identifying refusal as a new optimization target that current models do not directly address. To prevent contamination, the dataset will be publicly released in late 2026, with model evaluations available upon request in the interim.
>
---
#### [replaced 118] CoCoReviewBench: A Completeness- and Correctness-Oriented Benchmark for AI Reviewers
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI评审系统评估任务，旨在解决现有评价指标偏向人类评论而非正确性的问题。通过构建基准集并强化完整性和正确性，提升AI评审的可靠性。**

- **链接: [https://arxiv.org/pdf/2605.07905](https://arxiv.org/pdf/2605.07905)**

> **作者:** Hexuan Deng; Xiaopeng Ke; Yichen Li; Ruina Hu; Dehao Huang; Derek F. Wong; Yue Wang; Xuebo Liu; Min Zhang
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Despite the rapid development of AI reviewers, evaluating such systems remains challenging: metrics favor overlap with human reviews over correctness. However, since human reviews often cover only a subset of salient issues and sometimes contain mistakes, they are unreliable as gold references. To address this, we build category-specific benchmark subsets and skip evaluation when the corresponding human reviews are missing to strengthen Completeness. We also leverage reviewer--author--meta-review discussions as expert annotations and filter unreliable reviews accordingly to strengthen Correctness. Finally, we introduce CoCoReviewBench, which curates 3,900 papers from ICLR and NeurIPS to enable reliable and fine-grained evaluation of AI reviewers. Analysis shows that AI reviewers remain limited in correctness and are prone to hallucinations, and highlights reasoning models as more effective reviewers, motivating further directions for improving AI reviewers. Benchmarks and models are available at this https URL.
>
---
#### [replaced 119] MentalBench: A DSM-Grounded Benchmark for Evaluating Psychiatric Diagnostic Capability of Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出MentalBench，用于评估大语言模型在精神疾病诊断中的能力，解决现有基准不足的问题。通过构建知识图谱生成临床案例，测试模型在不同模糊程度下的诊断准确性。**

- **链接: [https://arxiv.org/pdf/2602.12871](https://arxiv.org/pdf/2602.12871)**

> **作者:** Hoyun Song; Migyeong Kang; Jisu Shin; Jihyun Kim; Chanbi Park; Hangyeol Yoo; Jihyun An; Alice Oh; Jinyoung Han; KyungTae Lim
>
> **摘要:** Large language models (LLMs) have attracted growing interest as supportive tools for psychiatric assessment and clinical decision support. However, existing mental health benchmarks largely rely on social media data or supportive dialogue settings, limiting their ability to assess whether models can apply formal diagnostic criteria and differential diagnostic rules. In this paper, we introduce MentalBench, a benchmark for evaluating whether LLMs can make DSM-grounded psychiatric diagnostic decisions under varying levels of clinical ambiguity. At the core of MentalBench is MentalKG, a psychiatrist-built and validated knowledge graph encoding DSM-5 diagnostic criteria and differential diagnostic rules for 23 psychiatric disorders. Using MentalKG as an expert-curated logical backbone, we generate 24,750 synthetic clinical cases that systematically vary in information completeness and diagnostic complexity, enabling DSM-grounded evaluation. Our experiments show that although state-of-the-art LLMs perform well on noise-free queries that probe DSM-5 knowledge, they struggle to calibrate their confidence when distinguishing between disorders with overlapping symptoms. These findings raise concerns about the reliability of LLMs as psychiatric decision-support tools and highlight the need for more evaluation that reflects the diverse challenges in real-world psychiatric diagnosis.
>
---
#### [replaced 120] FinTagging: Benchmarking LLMs for Extracting and Structuring Financial Information
- **分类: cs.CL; cs.AI; cs.CE**

- **简介: 该论文提出FinTagging，解决金融信息结构化标注问题。任务是准确提取并映射财务数据到US GAAP分类体系，通过两个子任务评估大语言模型的数值推理与分类能力。**

- **链接: [https://arxiv.org/pdf/2505.20650](https://arxiv.org/pdf/2505.20650)**

> **作者:** Yan Wang; Lingfei Qian; Xueqing Peng; Yang Ren; Keyi Wang; Yi Han; Dongji Feng; Fengran Mo; Shengyuan Lin; Qinchuan Zhang; Kaiwen He; Chenri Luo; Jianxing Chen; Junwei Wu; Chen Xu; Ziyang Xu; Jimin Huang; Guojun Xiong; Xiao-Yang Liu; Qianqian Xie; Jian-Yun Nie
>
> **摘要:** Accurate interpretation of numerical data in financial reports is critical for markets and regulators. Although XBRL (eXtensible Business Reporting Language) provides a standard for tagging financial figures, mapping thousands of facts to over 10k US GAAP concepts remains costly and error prone. Existing benchmarks oversimplify this task as flat, single step classification over small subsets of concepts, ignoring the hierarchical semantics of the taxonomy and the structured nature of financial documents. Consequently, these benchmarks fail to evaluate Large Language Models (LLMs) under realistic reporting conditions. To bridge this gap, we introduce FinTagging, the first comprehensive benchmark for structure aware and full scope XBRL tagging. We decompose the complex tagging process into two subtasks: (1) FinNI (Financial Numeric Identification), which extracts entities and types from heterogeneous contexts including text and tables; and (2) FinCL (Financial Concept Linking), which maps extracted entities to the full US GAAP taxonomy. This two stage formulation enables a fair assessment of LLMs' capabilities in numerical reasoning and taxonomy alignment. Evaluating diverse LLMs in zero shot settings reveals that while models generalize well in extraction, they struggle significantly with fine grained concept linking, highlighting critical limitations in domain specific structure aware reasoning.
>
---
#### [replaced 121] Can LLMs Refuse Questions They Do Not Know? Measuring Knowledge-Aware Refusal in Factual Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决LLMs在事实任务中无法可靠拒绝未知问题的问题。提出Refusal Index（RI）衡量模型的知识感知拒绝能力。**

- **链接: [https://arxiv.org/pdf/2510.01782](https://arxiv.org/pdf/2510.01782)**

> **作者:** Wenbo Pan; Jie Xu; Qiguang Chen; Junhao Dong; Libo Qin; Xinfeng Li; Haining Yu; Xiaohua Jia
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Large Language Models (LLMs) should refuse to answer questions beyond their knowledge. This capability, which we term knowledge-aware refusal, is crucial for factual reliability, while existing metrics fail to capture this ability. In this work, we propose the Refusal Index (RI), a novel and principled metric that measures how accurately LLMs refuse questions they do not know. We define RI as Spearman's rank correlation between refusal probability and error probability. RI is practically measurable with a lightweight two-pass evaluation method which only require observed refusal rates across two standard evaluation runs. Extensive experiments across 16 models and 5 datasets demonstrate that RI accurately quantifies a model's knowledge-aware refusal capability. Notably, RI remains stable across different refusal rates and provides consistent model rankings independent of a model's overall accuracy and refusal rates. These properties suggest RI captures a stable, intrinsic aspect of model knowledge calibration. More importantly, RI provides insight into an important but previously overlooked aspect of LLM factuality: while LLMs achieve high accuracy on factual tasks, their refusal behavior can be unreliable and fragile.
>
---
#### [replaced 122] EndoCogniAgent: Closed-Loop Agentic Reasoning with Self-Consistency Validation for Endoscopic Diagnosis
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出EndoCogniAgent，用于解决内镜诊断中的多步骤推理与证据验证问题，通过闭环代理框架提升诊断可靠性。**

- **链接: [https://arxiv.org/pdf/2508.07292](https://arxiv.org/pdf/2508.07292)**

> **作者:** Yi Tang; Kai-Ni Wang; Yang Chen; Xiaopu He; Guangquan Zhou
>
> **备注:** 10 pages, 8 figures, 2 tables. Revised version with major updates on methodology and extended evaluation on EndoAgentBench. Code and data are available at this https URL
>
> **摘要:** Endoscopic diagnosis is an iterative process in which clinicians progressively acquire, compare, and verify local visual evidence before reaching a conclusion. Current AI systems do not adequately support this process because fine-grained evidence acquisition and multi-step reasoning remain weakly coupled. This gives rise to two failure modes, hallucinated evidence and uncorrected error accumulation, that undermine diagnostic reliability. We propose EndoCogniAgent, a closed-loop agentic framework that formulates endoscopic diagnosis as a controlled state update process. At each reasoning round, a central planner selects the next evidence acquisition action, specialized expert tools extract the corresponding observation, and a self-consistency validation mechanism examines the observation along two dimensions, knowledge consistency against the input image and temporal consistency with prior validated findings, before updating the diagnostic state. Validated observations are admitted into the evolving state to condition subsequent planning, while insufficiently supported findings are retained with corrective feedback that redirects the planner toward additional verification. We further introduce EndoAgentBench, a workflow-oriented benchmark comprising 6,132 question-answer pairs from 11 endoscopic datasets, designed to evaluate diagnostic agents across a comprehensive diagnostic chain, from fine-grained visual perception to high-level diagnostic reasoning. Experiments show that EndoCogniAgent achieves 85.23\% average accuracy on perception tasks and 71.13\% clinical acceptance rate on reasoning tasks, with ablation analysis confirming that self-consistency validation and episodic state maintenance are individually critical to these gains.
>
---
#### [replaced 123] Permutation-Consensus Listwise Judging for Robust Factuality Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于事实性评估任务，解决候选答案顺序对LLM判断的影响问题。通过多次排列并聚合结果，提升评估可靠性。**

- **链接: [https://arxiv.org/pdf/2603.20562](https://arxiv.org/pdf/2603.20562)**

> **作者:** Tianyi Huang; Nathan Huang; Justin Tang; Wenqian Chen; Elsa Fan
>
> **备注:** Accepted at the Fifth Workshop on Natural Language Generation, Evaluation, and Metrics at ACL 2026
>
> **摘要:** Large language models (LLMs) are now widely used as judges, yet their decisions can change under presentation choices that should be irrelevant. We study one such source of instability: candidate-order sensitivity in listwise factuality evaluation, where several answers can look similarly polished while differing substantially in hallucination risk. We introduce PCFJudge, an inference-time method that reruns the same factuality-first listwise prompt over multiple orderings of the same candidate set and aggregates the resulting scores, ranks, and uncertainty signals into a single consensus decision. On RewardBench 2 Factuality, the final seven-permutation aggregate (K=7) improves top-1 selection accuracy from 86.00% to 91.33% with GPT-5.4 and from 86.33% to 89.67% with Claude Sonnet 4.6. These results suggest that candidate order can be a meaningful source of factuality-judging error and that marginalizing over this nuisance variation can improve the reliability of LLM evaluation.
>
---
#### [replaced 124] Learning from Self-Debate: Preparing Reasoning Models for Multi-Agent Debate
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升大模型的推理能力。针对现有方法未充分准备模型参与多智能体辩论的问题，提出SDRL框架，通过自辩论训练增强模型独立解决问题和协作推理的能力。**

- **链接: [https://arxiv.org/pdf/2601.22297](https://arxiv.org/pdf/2601.22297)**

> **作者:** Chenxi Liu; Yanshuo Chen; Ruibo Chen; Tianyi Xiong; Tong Zheng; Heng Huang
>
> **摘要:** The reasoning abilities of large language models (LLMs) have been substantially improved by reinforcement learning with verifiable rewards (RLVR). At test time, collaborative reasoning through Multi-Agent Debate (MAD) has emerged as a promising approach for enhancing LLM performance. However, current RLVR methods typically train LLMs to solve problems in isolation, without explicitly preparing them to synthesize and benefit from different rationales that arise during debate. In this work, we propose Self-Debate Reinforcement Learning(SDRL), a training framework where models learn from self-debate, equipping a single LLM with both strong standalone problem-solving ability and the capability to process diverse reasoning trajectories in MAD. Given a prompt, SDRL first samples multiple candidate solutions, then constructs a debate context with diverse reasoning paths and generates second-turn responses conditioned on this context. Finally, SDRL jointly optimizes both the initial and debate-conditioned responses, yielding a model that is effective as both a standalone solver and a debate participant. Experiments across multiple base models and reasoning benchmarks show that SDRL consistently improves MAD performance across diverse debate protocols and agent configurations, while simultaneously strengthening single-model reasoning.
>
---
#### [replaced 125] AI Alignment Breaks at the Edge
- **分类: cs.CL**

- **简介: 论文探讨AI对齐在边缘案例中的失效问题，提出Edge对齐方法以识别和处理价值冲突、多元意见和不确定性。属于AI安全任务，旨在提升对齐的全面性和动态治理能力。**

- **链接: [https://arxiv.org/pdf/2602.20042](https://arxiv.org/pdf/2602.20042)**

> **作者:** Han Bao; Yue Huang; Xiaoda Wang; Zheyuan Zhang; Yujun Zhou; Carl Yang; Xiangliang Zhang; Yanfang Ye
>
> **备注:** 38 pages, 6 figures
>
> **摘要:** General Alignment has improved average-case helpfulness and safety, but current alignment practice still rewards confident, single-turn responses. The problem is not only that models fail on edge cases; it is that current evaluation makes many of these failures hard to see. We take the position that alignment must move beyond average-case evaluation by making failures under value conflict, plural stakeholder disagreement, and epistemic ambiguity visible and actionable. Scalar rewards compress diverse values into a single number; data and evaluation regimes collapse, filter, or fail to elicit the cases where alignment is hardest; and governance often lacks mechanisms for adjudicating contested cases. These blind spots produce value flattening, representation loss, and uncertainty blindness. We use Edge alignment to name a detection, evaluation, and governance agenda for surfacing these failures and connecting them to appropriate interventions. Rather than a single training objective, Edge alignment defines the conditions under which standard alignment should yield to mechanisms that preserve multidimensional value structure, represent plural perspectives, and support uncertainty-aware interaction. A pilot diagnostic set of 91 edge cases and four contemporary models illustrates that ordinary helpfulness and safety readings can miss process failures that edge-aware evaluation exposes. We outline operational edge signals, process-aware evaluation criteria, and a three-phase process stack that reframes alignment as a lifecycle problem of dynamic normative governance.
>
---
#### [replaced 126] GroupMemBench: Benchmarking LLM Agent Memory in Multi-Party Conversations
- **分类: cs.CL**

- **简介: 该论文提出GroupMemBench，用于评估LLM代理在多方对话中的记忆能力，解决多用户场景下记忆系统不足的问题。**

- **链接: [https://arxiv.org/pdf/2605.14498](https://arxiv.org/pdf/2605.14498)**

> **作者:** Jingbo Yang; Kwei-Herng Lai; Xiaowen Wang; Shiyu Chang; Yaar Harari; Evgeniy Gabrilovich
>
> **摘要:** Large Language Model (LLM) agents increasingly serve as personal assistants and workplace collaborators, where their utility depends on memory systems that extract, retrieve, and apply information across long-running conversations. However, both existing memory systems and benchmarks are built around the dyadic, single-user setup, even though real deployments routinely span groups and channels with multiple users interacting with the agent and with each other. This mismatch leaves three properties of group memory unmeasured: (i) group dynamics that go beyond concatenated one-on-one chats, (ii) speaker-grounded belief tracking, where the per-user memory modeling is needed, and (iii) audience-adapted language, where Theory-of-Mind shifts produce role-specific vocabulary. We introduce GroupMemBench, a benchmark that exposes all three. A graph-grounded synthesis pipeline produces multi-party conversations with controllable reply structure and conditions each message on per-user personas and target audiences. An adversarial query pipeline then binds every question to a specific asker across six categories, spanning multi-hop reasoning, knowledge update, term ambiguity, user-implicit reasoning, temporal reasoning, and abstention, and iteratively searches challenging, realistic queries that reflect comprehensive memory capability. Benchmarking leading memory systems exposes a sharp collapse: the strongest one reaches only 46.0% average accuracy, with knowledge update at 27.1% and term ambiguity at 37.7%, while a simple BM25 baseline matches or exceeds most agent memory systems. This indicates current memory ingestion erases the structural and lexical features group memory depends on, leaving multi-user memory far from solved.
>
---
#### [replaced 127] Hunt Instead of Wait: Evaluating Deep Data Research on Large Language Models
- **分类: cs.AI; cs.CL; cs.DB; cs.LG**

- **简介: 该论文属于人工智能领域，探讨如何提升大语言模型的自主探索能力。研究提出DDR任务和DDR-Bench基准，以评估模型从数据中自主发现洞察的能力，解决传统任务难以衡量自主性的问题。**

- **链接: [https://arxiv.org/pdf/2602.02039](https://arxiv.org/pdf/2602.02039)**

> **作者:** Wei Liu; Peijie Yu; Michele Orini; Yali Du; Yulan He
>
> **备注:** 14 pages, 7 tables, 8 figures, accepted by ICML 2026
>
> **摘要:** The agency expected of Agentic Large Language Models goes beyond answering correctly, requiring autonomy to set goals and decide what to explore. We term this investigatory intelligence, distinguishing it from executional intelligence, which merely completes assigned tasks. Data Science provides a natural testbed, as real-world analysis starts from raw data rather than explicit queries, yet few benchmarks focus on it. To address this, we introduce Deep Data Research (DDR), an open-ended task where LLMs autonomously extract key insights from databases, and DDR-Bench, a large-scale, checklist-based benchmark that enables verifiable evaluation. Results show that while frontier models display emerging agency, long-horizon exploration remains challenging. Our analysis highlights that effective investigatory intelligence depends not only on agent scaffolding or merely scaling, but also on intrinsic strategies of agentic models.
>
---
#### [replaced 128] Automated Knowledge Component Generation for Interpretable Knowledge Tracing in Coding Problems
- **分类: cs.AI; cs.CL; cs.CY; cs.LG; cs.SE**

- **简介: 该论文属于知识追踪任务，旨在自动化生成编程问题的知识组件（KCs），解决人工标注耗时的问题。工作包括构建LLM驱动的KC生成与追踪框架，验证其有效性。**

- **链接: [https://arxiv.org/pdf/2502.18632](https://arxiv.org/pdf/2502.18632)**

> **作者:** Zhangqi Duan; Nigel Fernandez; Arun Balajiee Lekshmi Narayanan; Mohammad Hassany; Rafaella Sampaio de Alencar; Peter Brusilovsky; Bita Akram; Andrew Lan
>
> **备注:** Findings of ACL 2026: The 64th Annual Meeting of the Association for Computational Linguistics
>
> **摘要:** Knowledge components (KCs) mapped to problems help model student learning, tracking their mastery levels on fine-grained skills thereby facilitating personalized learning and feedback in online learning platforms. However, crafting and tagging KCs to problems, traditionally performed by human domain experts, is highly labor intensive. We present an automated, LLM-based pipeline for KC generation and tagging for open-ended programming problems. We also develop an LLM-based knowledge tracing (KT) framework to leverage these LLM-generated KCs, which we refer to as KCGen-KT. We conduct extensive quantitative and qualitative evaluations on two real-world student code submission datasets in different programming this http URL find that KCGen-KT outperforms existing KT methods and human-written KCs on future student response prediction. We investigate the learning curves of generated KCs and show that LLM-generated KCs result in a better fit than human written KCs under a cognitive model. We also conduct a human evaluation with course instructors to show that our pipeline generates reasonably accurate problem-KC mappings.
>
---
#### [replaced 129] Beacon: Single-Turn Diagnosis and Mitigation of Latent Sycophancy in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型对齐研究，旨在解决大语言模型中的隐性奉承问题。通过构建基准测试，分析并干预模型的偏见，以改善其真实性与社会合规性之间的平衡。**

- **链接: [https://arxiv.org/pdf/2510.16727](https://arxiv.org/pdf/2510.16727)**

> **作者:** Sanskar Pandey; Ruhaan Chopra; Angkul Puniya; Sohom Pal
>
> **摘要:** Large language models internalize a structural trade-off between truthfulness and obsequious flattery, emerging from reward optimization that conflates helpfulness with polite submission. This latent bias, known as sycophancy, manifests as a preference for user agreement over principled reasoning. We introduce Beacon, a single-turn forced-choice benchmark that isolates this bias independent of conversational context, enabling precise measurement of the tension between factual accuracy and submissive bias. Evaluations across twelve state-of-the-art models reveal that sycophancy decomposes into stable linguistic and affective sub-biases, each scaling with model capacity. We further propose prompt-level and activation-level interventions that modulate these biases in opposing directions, exposing the internal geometry of alignment as a dynamic manifold between truthfulness and socially compliant judgment. Beacon reframes sycophancy as a measurable form of normative misgeneralization, providing a reproducible foundation for studying and mitigating alignment drift in large-scale generative systems.
>
---
#### [replaced 130] Query-Aware Learnable Graph Pooling Tokens as Prompt for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于图数据处理任务，解决节点投影可扩展性差和图投影信息丢失问题，提出LGPT方法结合可学习参数与大语言模型，提升图表示效果。**

- **链接: [https://arxiv.org/pdf/2501.17549](https://arxiv.org/pdf/2501.17549)**

> **作者:** Wooyoung Kim; Byungyoon Park; Wooju Kim
>
> **摘要:** Graph-structured data plays a vital role in numerous domains, such as social networks, citation networks, commonsense reasoning graphs and knowledge graphs. While graph neural networks have been employed for graph processing, recent advancements have explored integrating large language models for graph-based tasks. In this paper, we propose a novel approach named Learnable Graph Pooling Token (LGPT), which addresses the limitations of the scalability issues in node-level projection and information loss in graph-level projection. LGPT enables flexible and efficient graph representation by introducing learnable parameters that act as tokens in large language models, balancing fine-grained and global graph information. Additionally, we investigate an Early Query Fusion technique, which fuses query context before constructing the graph representation, leading to more effective graph embeddings. Our method achieves a 4.13\% performance improvement on the GraphQA benchmark without training the large language model, demonstrating significant gains in handling complex textual-attributed graph data.
>
---
#### [replaced 131] ClawArena: Benchmarking AI Agents in Evolving Information Environments
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出ClawArena，用于评估AI代理在动态信息环境中的表现。解决AI代理如何处理多源冲突、动态信念更新和隐式个性化的问题。通过设计多轮场景进行实验验证。**

- **链接: [https://arxiv.org/pdf/2604.04202](https://arxiv.org/pdf/2604.04202)**

> **作者:** Haonian Ji; Kaiwen Xiong; Siwei Han; Peng Xia; Shi Qiu; Yiyang Zhou; Jiaqi Liu; Jinlong Li; Bingzhou Li; Zeyu Zheng; Cihang Xie; Huaxiu Yao
>
> **摘要:** AI agents deployed as persistent assistants must maintain correct beliefs as their information environment evolves. In practice, evidence is scattered across heterogeneous sources that often contradict one another, new information can invalidate earlier conclusions, and user preferences surface through corrections rather than explicit instructions. Existing benchmarks largely assume static, single-authority settings and do not evaluate whether agents can keep up with this complexity. We introduce ClawArena, a benchmark for evaluating AI agents in evolving information environments. Each scenario maintains a complete hidden ground truth while exposing the agent only to noisy, partial, and sometimes contradictory traces across multi-channel sessions, workspace files, and staged updates. Evaluation is organized around three coupled challenges: multi-source conflict reasoning, dynamic belief revision, and implicit personalization, whose interactions yield a 14-category question taxonomy. Two question formats, multi-choice (set-selection) and shell-based executable checks, test both reasoning and workspace grounding. ClawArena comprises 12 multi-turn scenarios spanning 337 evaluation rounds with 45 dynamic updates, evaluated across five agent frameworks and 18 language models from proprietary, community-accessible, and self-hosted sources. Experiments show that model capability accounts for a 29-point score range across models while framework design accounts for up to a 24-point range, that MetaClaw's skill overlay reliably improves score without degrading accuracy, and that belief revision difficulty is determined by update design strategy rather than update volume. Code is available at this https URL.
>
---
#### [replaced 132] Minimal-Intervention KV Retention via Set-Conditioned Diversity
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于知识蒸馏任务，旨在解决小预算下的KV缓存压缩问题。通过改进评分机制，提出α方法，在有限资源下提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.14292](https://arxiv.org/pdf/2605.14292)**

> **作者:** Libo Sun; Po-wei Harn; Peixiong He; Xiao Qin
>
> **备注:** 15 pages, 3 figures, 3 tables. Code and data: this https URL
>
> **摘要:** KV-cache compression at small budgets is a crowded design space spanning cache representation, head-wise routing, compression cadence, decoding behavior, and within-budget scoring. We study seven mechanisms across these five families under matched mean cache on long-form mathematical reasoning (MATH-500~\cite{hendrycks2021math}) with two distilled-reasoning models (Qwen-7B and Llama-8B variants of DeepSeek-R1-Distill~\cite{deepseek2025r1}) at budgets $b \in \{64, 128\}$. All seven were rejected. We then propose $\alpha$, a one-function modification to the TriAttention~\cite{mao2026triattention} retention scorer that replaces argmax-top-$k$ with greedy facility-location-inspired selection under a V-space redundancy penalty controlled by a single weight $\lambda$. A pre-registered protocol tunes $\lambda$ on a frozen development split and confirms on a disjoint held-out split; with $\lambda = 0.5$, $\alpha$ clears Bonferroni on two of the four (model, budget) cells (Qwen $b{=}128$ and Llama $b{=}64$), no cell is significantly negative, and the pre-registered Branch~A triggers. The finding is asymmetric: a minimal scoring modification beat heavier structural redesigns in this regime, and the combined matched-memory, sympy-graded, held-out confirmation protocol is the evidence standard that made the asymmetry visible.
>
---
#### [replaced 133] Gated KalmaNet: A Fading Memory Layer Through Test-Time Ridge Regression
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Gated KalmaNet，解决线性状态空间模型记忆失效问题，通过精确计算卡尔曼增益提升任务性能。**

- **链接: [https://arxiv.org/pdf/2511.21016](https://arxiv.org/pdf/2511.21016)**

> **作者:** Liangzu Peng; Aditya Chattopadhyay; Luca Zancato; Elvis Nunez; Wei Xia; Stefano Soatto
>
> **备注:** 30 pages, 10 figures. Accepted at CVPR 2026
>
> **摘要:** Linear State-Space Models (SSMs) offer an efficient alternative to softmax Attention with constant memory and linear compute, but their lossy, fading summary of the past hurts recall-oriented tasks. We propose Gated KalmaNet (GKA, pronounced "gee-ka"), a layer that accounts for the full past while retaining SSM-style efficiency. We ground our approach in the Kalman Filter (KF), and show that several existing SSM layers (DeltaNet, Gated DeltaNet, Kimi Delta Attention) are approximations to the KF recurrence under an identity error covariance assumption, which ignores how past keys and values should optimally influence state updates. In contrast, GKA maintains the full error covariance and computes the exact Kalman gain. Under a steady-state assumption that enables parallelization, this reduces to an online ridge regression with constant memory and linear compute. The standard KF equations are numerically unstable in low-precision settings (e.g., bfloat16) and hard to parallelize on GPUs. We address this with (1) adaptive regularization via input-dependent gating to control the ridge regression's condition number, and (2) Chebyshev Iteration, which we show is more stable than conventional iterative solvers in low precision. We further develop hardware-aware chunk-wise kernels for efficient training. Empirically, GKA outperforms existing SSM layers (e.g., Mamba2, Gated DeltaNet) on short-context tasks and achieves more than 10\% relative improvement on long-context RAG and LongQA up to 128k tokens. We further show GKA outperforms Mamba when extended to ImageNet classification. Our code, including Triton kernels for training and inference (vLLM), along with a model zoo of GKA-based Hybrid models at 8B and 32B scale on HuggingFace, is released under Apache 2.0.
>
---
#### [replaced 134] Interactive Benchmarks
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出"交互基准"，解决模型推理评估不足的问题。通过多轮互动评估模型智能，涵盖逻辑、数学等任务，揭示提升空间。**

- **链接: [https://arxiv.org/pdf/2603.04737](https://arxiv.org/pdf/2603.04737)**

> **作者:** Baoqing Yue; Zihan Zhu; Yutong Han; Brian Fan; Qian Sun; Jichen Feng; Hufei Yang; Yifan Zhang; Mengdi Wang
>
> **备注:** Project Page: this https URL
>
> **摘要:** Existing reasoning evaluation paradigms suffer from different limitations: fixed benchmarks are increasingly saturated and vulnerable to contamination, while preference-based evaluations rely on subjective judgments. We argue that a core aspect of intelligence is the ability to decide what information to acquire and how to use it effectively. We propose Interactive Benchmarks, a unified evaluation paradigm that assesses a model's reasoning ability through budgeted multi-turn interaction. We evaluate models under this framework in two settings: Interactive Proofs, where models interact with a judge to solve Logic, UI2Html, and Mathematics tasks under objective feedback; and Interactive Games, where models reason strategically to maximize long-horizon utilities. Our results show that interactive benchmarks provide a more robust assessment of this dimension of model intelligence, revealing substantial room for improvement in interactive scenarios.
>
---
#### [replaced 135] Language models fail at extended rule following
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究语言模型在遵循规则时的可靠性问题。工作包括测试模型计数能力，发现其存在状态限制，导致失败。**

- **链接: [https://arxiv.org/pdf/2605.02028](https://arxiv.org/pdf/2605.02028)**

> **作者:** Tianxiang Dai; Jonathan Fan
>
> **备注:** for accessing the data and code for reproduction of the study, see this https URL
>
> **摘要:** Large language models are highly capable of answering difficult questions by retrieving, recombining, and attending to information in long contexts. For agentic tasks, an additional capability is required: the preservation of an exact state while repeatedly applying rules. We find that this reliability is absent across language models. To demonstrate, we query 126 leading model variants with the task of counting a long string of repeated characters, and we find they all cannot accurately count above a model-dependent, syntax-sensitive counting capacity threshold. Failures are abrupt and persist even with increasing model size, inference time computation, and external tool. Mechanistic probing indicates that models use a finite number of internal states to mimic counting as a rule and fail once these states are exhausted. Furthermore, such states are the basis for performing complex tasks beyond counting. These results indicate that fundamentally new model architectures are required for autonomous agents to achieve truly reliable rule following capabilities.
>
---
#### [replaced 136] Compounding Disadvantage: Auditing Intersectional Bias in LLM-Generated Explanations Across Indian and American STEM Education
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于AI偏见审计任务，旨在检测大语言模型在STEM教育中生成解释时是否存在交叉性歧视。研究发现模型对边缘化学生群体存在系统性不利影响。**

- **链接: [https://arxiv.org/pdf/2601.14506](https://arxiv.org/pdf/2601.14506)**

> **作者:** Amogh Gupta; Niharika Patil; Sourojit Ghosh; SnehalKumar; S Gaikwad
>
> **摘要:** Large language models are increasingly deployed in STEM education for personalized instruction and feedback across institutions in high- and low-income countries. These systems are designed to adapt content to student needs, but whether they adapt based on demonstrated ability or demographic signals remains untested at scale. Here we establish that LLM-generated STEM content systematically disadvantages marginalized student profiles across two cultural contexts, with the gap between the most privileged and most marginalized profiles reaching 2.55 grade levels. We audited four LLMs (Qwen 2.5-32B-Instruct, GPT-4o, GPT-4o-mini, GPT-OSS 20B) using synthetic profiles crossing dimensions specific to Indian education (caste, medium of instruction, college tier) and American education (race, HBCU attendance, school type), alongside income, gender, and disability, across ranking and generation tasks with FDR-corrected significance testing and SHAP feature attribution. Income produces significant effects across every model and context, medium of instruction drives the largest single effect in the Indian context, and disability status triggers simpler explanations. Effects compound non-additively: marginalization across multiple dimensions produces gaps larger than any single dimension predicts, and biases persist within elite institutions. Bias is consistent across all four architectures and persists through model selection, making intersectional, cross-cultural auditing a structural requirement before deployment.
>
---
#### [replaced 137] Self-Distilled Trajectory-Aware Boltzmann Modeling: Bridging the Training-Inference Discrepancy in Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型训练任务，旨在解决扩散语言模型训练与推理不一致的问题。通过轨迹对齐的玻尔兹曼建模方法，提升模型知识获取能力与泛化性能。**

- **链接: [https://arxiv.org/pdf/2605.11854](https://arxiv.org/pdf/2605.11854)**

> **作者:** Kecheng Chen; Ziru Liu; Xijia Tao; Hui Liu; Yibing Liu; Xinyu Fu; Shi Wu; Suiyun Zhang; Dandan Tu; Lingpeng Kong; Rui Liu; Haoliang Li
>
> **备注:** Project website: this https URL
>
> **摘要:** Diffusion Language Models (DLMs) have recently emerged as a promising alternative to autoregressive language models, offering stronger global awareness and highly parallel generation. However, post-training DLMs with standard Negative Evidence Lower Bound (NELBO)-based supervised fine-tuning remains inefficient: training reconstructs randomly masked tokens in a single step, whereas inference follows a confidence-guided, multi-step easy-to-hard denoising trajectory. Recent trajectory-based self-distillation methods exploit such inference trajectories mainly for sampling-step compression and acceleration, often improving decoding efficiency without substantially enhancing the model's underlying capability, and may even degrade performance under full diffusion decoding. In this work, we ask whether self-distilled trajectories can be used not merely for faster inference, but for genuine knowledge acquisition. Although these trajectories lie on the pretrained DLM's own distributional manifold and thus offer a potentially lower optimization barrier, we find that naively fine-tuning on them with standard NELBO objectives yields only marginal gains. To address this limitation, we propose \textbf{T}rajectory-\textbf{A}ligned optimization via \textbf{Bo}ltzmann \textbf{M}odeling (\textbf{TABOM}), a self-distilled trajectory-based post-training framework that aligns training with the easy-to-hard structure of inference. TABOM models the inference unmasking preference as a Boltzmann distribution over predictive entropies and derives a tractable pairwise ranking objective to align the model's certainty ordering with the observed decoding trajectory. Empirically, TABOM achieves substantial gains in new domains, expands the effective knowledge boundary of DLMs, and significantly mitigates catastrophic forgetting compared with standard SFT.
>
---
#### [replaced 138] Toward Robust Multilingual Adaptation of LLMs for Low-Resource Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言大模型适应任务，旨在解决低资源语言训练数据不足的问题。提出LiRA框架，通过联合优化表示稳定性和跨语言语义一致性提升模型性能。**

- **链接: [https://arxiv.org/pdf/2510.14466](https://arxiv.org/pdf/2510.14466)**

> **作者:** Haolin Li; Haipeng Zhang; Mang Li; Yaohua Wang; Lijie Wen; Yu Zhang; Biqing Huang
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Large language models (LLMs) continue to struggle with low-resource languages, primarily due to limited training data, translation noise, and unstable cross-lingual alignment. To address these challenges, we propose LiRA (Linguistic Robust Anchoring for LLMs)-a plug-and-play framework that requires only lightweight fine-tuning on top of existing pretrained backbones. LiRA jointly optimizes representation stability and cross-lingual semantic consistency by combining two key components: Arca (Anchored Representation Composition Architecture), which aligns low-resource inputs to a shared English semantic space through anchor-based alignment and collaborative encoding; and LaSR (Language-coupled Semantic Reasoner), a lightweight, language-aware head that enforces consistency regularization for unified cross-lingual understanding, retrieval, and reasoning. We theoretically show that under controlled anchoring error and translation-induced bias, LiRA guarantees bounded representation deviation and stable downstream performance under local Lipschitz continuity. To facilitate research, we release a new multilingual product retrieval dataset covering five Southeast Asian and two South Asian languages. Extensive experiments across diverse low-resource benchmarks demonstrate consistent improvements in retrieval, ranking, question answering, and reasoning tasks. Code will be publicly available on GitHub, and the dataset will be hosted on Hugging Face.
>
---
#### [replaced 139] LISTEN to Your Preferences: An LLM Framework for Multi-Objective Selection
- **分类: cs.CL**

- **简介: 该论文提出LISTEN框架，解决多目标选择问题。通过自然语言与LLM交互，迭代优化偏好模型，提升决策效果。**

- **链接: [https://arxiv.org/pdf/2510.25799](https://arxiv.org/pdf/2510.25799)**

> **作者:** Adam S. Jovine; Tinghan Ye; Francis Bahk; Jingjing Wang; Matthew Ford; David B. Shmoys; Peter I. Frazier
>
> **备注:** Accepted at IJCAI-ECAI 2026 (the 35th International Joint Conference on Artificial Intelligence)
>
> **摘要:** Human experts often struggle to select the best option from a large set of items with multiple competing objectives, a process bottlenecked by the difficulty of formalizing complex, implicit preferences. To address this, we introduce LISTEN (LLM-based Iterative Selection with Trade-off Evaluation from Natural-language), an agentic LLM-based framework that treats the LLM as a decision-making agent capable of iteratively refining its internal preference model and taking actions (e.g., proposing utilities or selecting candidates) to maximize alignment with a user's implicit goals. To operate within LLM constraints like context windows and inference costs, we propose two iterative algorithms: LISTEN-U, which uses the LLM to refine a parametric utility function, and LISTEN-T, a non-parametric method that performs tournament-style selections over small batches of solutions. Evaluated on diverse tasks including flight booking, shopping, and exam scheduling, our results show LISTEN-U excels when preferences are parametrically aligned (a property we measure with a novel concordance metric), while LISTEN-T offers more robust performance overall. This work explores a promising direction for steering complex multi-objective decisions directly with natural language, reducing the cognitive burden of traditional preference elicitation. Code is available at this https URL.
>
---
#### [replaced 140] ClawGym: A Scalable Framework for Building Effective Claw Agents
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出ClawGym框架，解决Claw风格环境下的智能体开发问题，通过构建数据集、训练模型和建立评估基准，支持高效 agent 开发与评估。**

- **链接: [https://arxiv.org/pdf/2604.26904](https://arxiv.org/pdf/2604.26904)**

> **作者:** Fei Bai; Huatong Song; Shuang Sun; Daixuan Cheng; Yike Yang; Chuan Hao; Renyuan Li; Feng Chang; Yuan Wei; Ran Tao; Bryan Dai; Jian Yang; Wayne Xin Zhao; Ji-Rong Wen
>
> **摘要:** Claw-style environments support multi-step workflows over local files, tools, and persistent workspace states. However, scalable development around these environments remains constrained by the absence of a systematic framework, especially one for synthesizing verifiable training data and integrating it with agent training and diagnostic evaluation. To address this challenge, we present ClawGym, a scalable framework that supports the full lifecycle of Claw-style personal agent development. Concretely, we construct ClawGym-SynData, a diverse dataset of 13.5K filtered tasks synthesized from persona-driven intents and skill-grounded operations, paired with realistic mock workspaces and hybrid verification mechanisms. We then train a family of capable Claw-style models, termed ClawGym-Agents, through supervised fine-tuning on black-box rollout trajectories, and further explore reinforcement learning via a lightweight pipeline that parallelizes rollouts across per-task sandboxes. To support reliable evaluation, we further construct ClawGym-Bench, a benchmark of 200 instances calibrated through automated filtering and human-LLM review. Relevant resources have been released at this https URL.
>
---
#### [replaced 141] WEBSERV: A Full-Stack and RL-Ready Web Environment for Training Web Agents at Scale
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出WebServ，一个适用于大规模强化学习的全栈网页环境。解决现有环境资源消耗大、观测噪声高、执行不可靠的问题，通过优化容器和浏览器接口提升效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2510.16252](https://arxiv.org/pdf/2510.16252)**

> **作者:** Yuxuan Lu; Ziyi Wang; Jing Huang; Hui Liu; Jiri Gesi; Yan Han; Shihan Fu; Tianqi Zheng; Xianfeng Tang; Chen Luo; Yisi Sang; Jin Lai; Dakuo Wang
>
> **摘要:** Reinforcement learning (RL) for web agents demands environments that are both effective for evaluation and efficient enough for large-scale on-policy training. Current web environments fall short: server-side Docker setups are too resource-intensive for massive parallel rollouts, while browser-side interfaces produce noisy observations, execute actions unreliably under modern single-page applications, and omit visual interactivity cues. We introduce WebServ, a full-stack, RL-ready web environment that addresses these limitations end-to-end. On the server side, WebServ uses Incus containers with block-level copy-on-write, reducing launch latency by ~5x and persistent storage by ~240x, enabling 200+ concurrent isolated environments on a single host. On the browser side, WebServ provides a compact, site-agnostic observation and action interface derived automatically from the DOM with human-aligned interactivity cues, and a robust action execution backend using network-aware waiting for reliable SPA support. On WebArena-Lite, WebServ achieves state-of-the-art single-prompt results, with controlled comparisons confirming consistent gains across GPT-4o, OpenAI-o3, and Llama-3.1-8B over vanilla WebArena. We further train Qwen3-4B and Qwen3-30B-A3B with RL entirely within WebServ; the RL-trained 4B model achieves 55.5% mean accuracy, surpassing both Claude 4.5 Sonnet (50.0%) and the RL-trained 8B model from WebAgent-R1 (51.8%).
>
---
#### [replaced 142] MUSCAT: MUltilingual, SCientific ConversATion Benchmark
- **分类: cs.CL**

- **简介: 该论文提出MUSCAT基准，用于评估多语言语音识别系统处理多语言输入、专业词汇和混合语言的能力。任务是多语言语音识别，解决缺乏相关数据集的问题。**

- **链接: [https://arxiv.org/pdf/2604.15929](https://arxiv.org/pdf/2604.15929)**

> **作者:** Supriti Sinhamahapatra; Thai-Binh Nguyen; Yiğit Oğuz; Enes Ugan; Jan Niehues; Alexander Waibel
>
> **摘要:** The goal of multilingual speech technology is to facilitate seamless communication between individuals speaking different languages, creating the experience as though everyone were a multilingual speaker. To create this experience, speech technology needs to address several challenges: Handling mixed multilingual input, specific vocabulary, and code-switching. However, there is currently no dataset benchmarking this situation. We propose a new benchmark to evaluate current Automatic Speech Recognition (ASR) systems, whether they are able to handle these challenges. The benchmark consists of bilingual discussions on scientific papers between multiple speakers, each conversing in a different language. We provide a standard evaluation framework, beyond Word Error Rate (WER) enabling consistent comparison of ASR performance across languages. Experimental results demonstrate that the proposed dataset is still an open challenge for state-of-the-art ASR systems. The dataset is available in this https URL. Keywords: multilingual, speech recognition, audio segmentation, speaker diarization
>
---
#### [replaced 143] Dynamic Adversarial Fine-Tuning Reorganizes Refusal Geometry
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文研究安全对齐语言模型的拒绝机制，通过动态对抗微调（R2D2）探索其内部拒绝几何的变化，解决过拟合与鲁棒性平衡问题。**

- **链接: [https://arxiv.org/pdf/2604.27019](https://arxiv.org/pdf/2604.27019)**

> **作者:** Wenhao Lan; Shan Li; Junbin Yang; Haihua Shen; Yijun Yang
>
> **摘要:** Safety-aligned language models must refuse harmful requests without collapsing into broad over-refusal, yet it remains unclear how dynamic adversarial fine-tuning changes the internal carriers of refusal. We study one 7B backbone under supervised fine-tuning (SFT) and under Robust Refusal Dynamic Defense (R2D2), a HarmBench-style adversarial fine-tuning procedure that repeatedly refreshes harmful training cases with current jailbreak attacks. Our protocol aligns fixed-source HarmBench, StrongREJECT, and XSTest with a five-anchor refusal-geometry suite, causal interventions, and a sparse adaptive stress test. R2D2 drives fixed-source HarmBench attack success to zero at early checkpoints, but that regime coincides with maximal XSTest refusal and complete failure on a benign-utility audit. Later checkpoints partially recover benign utility while partially reopening attack success. Sparse adaptive attacks sharpen the same frontier: step~50 remains closed under both adaptive GCG and AutoDAN, whereas adaptive GCG ASR rises to 0.415 at step~250 and 0.613 at step~500. Geometrically, R2D2 preserves a late-layer admissible carrier through step~100 and relocates the best admissible carrier to an early layer by step~250; SFT relocates earlier while remaining less robust. Effective rank remains near 1.24, and SFT exhibits larger principal-angle drift despite worse robustness. Causal interventions show that late-stage R2D2 behavior is controlled by a low-dimensional but utility-coupled carrier. These results support a geometry-reorganization account along a robustness--utility frontier.
>
---
#### [replaced 144] LightTransfer: Your Long-Context LLM is Secretly a Hybrid Model with Effortless Adaptation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大模型优化任务，解决长文本生成中的内存效率问题。通过将Transformer模型转换为混合架构，提升推理速度并保持性能。**

- **链接: [https://arxiv.org/pdf/2410.13846](https://arxiv.org/pdf/2410.13846)**

> **作者:** Xuan Zhang; Fengzhuo Zhang; Cunxiao Du; Chao Du; Tianyu Pang; Wei Gao; Min Lin
>
> **备注:** Accepted by TMLR 2025
>
> **摘要:** Scaling language models to handle longer contexts introduces substantial memory challenges due to the growing cost of key-value (KV) caches. Motivated by the efficiency gains of hybrid models and the broad availability of pretrained large transformer backbones, we explore transitioning transformer models into hybrid architectures for a more efficient generation. In this work, we propose LightTransfer, a lightweight method that transforms models such as LLaMA into hybrid variants. Our approach identifies lazy layers -- those focusing on recent or initial tokens -- and replaces their full attention with streaming attention. This transformation can be performed without any training for long-context understanding tasks or with minimal fine-tuning for o1-like long reasoning generation tasks that require stronger reasoning capabilities. Experiments across diverse benchmarks and models (e.g., LLaMA, Mistral, QwQ-STILL) demonstrate that, even when half of the layers are identified as lazy, LightTransfer achieves up to 2.17$\times$ throughput improvement with minimal performance loss ($<1.5\%$ on LongBench) and achieves 53.3\% on math benchmark AIME24 of advanced o1-like long reasoning model QwQ-STILL.
>
---
#### [replaced 145] T-FIX: Text-Based Explanations with Features Interpretable to eXperts
- **分类: cs.CL**

- **简介: 该论文提出T-FIX框架，用于评估大语言模型生成解释的专家一致性。任务是提升模型在专业领域中的可解释性，解决现有方法依赖专家标注、成本高、难以扩展的问题。**

- **链接: [https://arxiv.org/pdf/2511.04070](https://arxiv.org/pdf/2511.04070)**

> **作者:** Shreya Havaldar; Weiqiu You; Chaehyeon Kim; Anton Xue; Helen Jin; Marco Gatti; Bhuvnesh Jain; Helen Qu; Amin Madani; Daniel A. Hashimoto; Gary E. Weissman; Rajat Deo; Sameed Khatana; Lyle Ungar; Eric Wong
>
> **摘要:** As LLMs are deployed in knowledge-intensive settings (e.g., surgery, astronomy, therapy), users are often domain experts who expect not just answers, but explanations that mirror professional reasoning. Yet evaluating whether an LLM "thinks like an expert" remains difficult: existing approaches rely on per-example expert annotation, making them costly, hard to scale, and tied to a single notion of correct reasoning within each domain. To address this gap, we introduce T-FIX, a unified evaluation framework that operationalizes expert alignment as a desired attribute of LLM-generated explanations. T-FIX spans seven scientific tasks across three domains, with each task evaluated against expert-defined criteria that capture domain-grounded reasoning rather than generic explanation quality. Our framework enables automatic, personalizable evaluation of expert alignment that generalizes to unseen explanations without ongoing expert involvement. Code is available at this https URL.
>
---
#### [replaced 146] SkillMAS: Skill Co-Evolution with LLM-based Multi-Agent System
- **分类: cs.MA; cs.CL**

- **简介: 该论文提出SkillMAS，解决多智能体系统中技能进化与结构重组分离的问题。通过耦合两者实现自适应专业化，提升系统部署后性能。**

- **链接: [https://arxiv.org/pdf/2605.09341](https://arxiv.org/pdf/2605.09341)**

> **作者:** Shuai Pan; Yixiang Liu; Jiaye Gao; Te Gao; Weiwen Liu; Jianghao Lin; Zhihui Fu; Jun Wang; Weinan Zhang; Yong Yu
>
> **备注:** 21 pages, 2 figures
>
> **摘要:** Large language model (LLM) agent systems are increasingly expected to improve after deployment, but existing work often decouples two adaptation targets: skill evolution and multi-agent system (MAS) restructuring. This separation can create organization bottlenecks, context pressure, and mis-specialization. We present SkillMAS, a non-parametric framework for adaptive specialization in multi-agent systems that couples skill evolution with MAS restructuring. SkillMAS uses Utility Learning to assign credit from verified execution traces, bounded skill evolution to refine reusable procedures without unfiltered library growth, and evidence-gated MAS restructuring when retained failures and Executor Utility indicate a structural mismatch. Across embodied manipulation, command-line execution, and retail workflows, SkillMAS is competitive under the reported harnesses while clarifying how post-deployment specialization is attributed, updated, and applied.
>
---
#### [replaced 147] Polar probe linearly decodes semantic structures from LLMs
- **分类: cs.CL**

- **简介: 该论文研究LLMs如何构建语义结构，通过极坐标探针线性解码实体间关系。任务是理解神经网络的语义编码机制，解决概念绑定问题，验证几何编码假设并评估其泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.14125](https://arxiv.org/pdf/2605.14125)**

> **作者:** Pablo J. Diego-Simón; Pierre Orhan; Emmanuel Chemla; Yair Lakretz; Jean-Rémi King
>
> **摘要:** How do artificial neural networks bind concepts to form complex semantic structures? Here, we propose a simple neural code, whereby the existence and the type of relations between entities are represented by the distance and the direction between their embeddings, respectively. We test this hypothesis in a variety of Large Language Models (LLMs), each input with natural-language descriptions of minimalist tasks from five different domains: arithmetic, visual scenes, family trees, metro maps and social interactions. Results show that the true semantic structures can be linearly recovered with a Polar Probe targeting a subspace of LLMs' layer activations. Second, this code emerges mostly in middle layers and improves with LLM performance. Third, these Polar Probes successfully generalize to new entities and relation types, but degrades with the size of the semantic structure. Finally, the quality of the polar representation correlates with the LLM's ability to answer questions about the semantic structure. Together, these findings suggest that LLMs learn to build complex semantic structures by binding representations with a simple geometrical principle.
>
---
#### [replaced 148] Speech-Hands: A Self-Reflection Voice Agentic Approach to Speech Recognition and Audio Reasoning with Omni Perception
- **分类: cs.SD; cs.AI; cs.CL; cs.MA; eess.AS**

- **简介: 该论文提出Speech-Hands框架，解决语音识别与音频推理中的自我信任与外部感知决策问题，提升模型可靠性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.09413](https://arxiv.org/pdf/2601.09413)**

> **作者:** Zhen Wan; Chao-Han Huck Yang; Jinchuan Tian; Hanrong Ye; Ankita Pasad; Szu-wei Fu; Arushi Goel; Ryo Hachiuma; Shizhe Diao; Kunal Dhawan; Sreyan Ghosh; Yusuke Hirota; Zhehuai Chen; Rafael Valle; Chenhui Chu; Shinji Watanabe; Yu-Chiang Frank Wang; Boris Ginsburg
>
> **备注:** Accepted to ACL 2026. Oral Presentation. Code: this https URL OpenClaw Branch: this https URL
>
> **摘要:** We introduce a voice-agentic framework that learns one critical omni-understanding skill: knowing when to trust itself versus when to consult external audio perception. Our work is motivated by a crucial yet counterintuitive finding: naively fine-tuning an omni-model on both speech recognition and external sound understanding tasks often degrades performance, as the model can be easily misled by noisy hypotheses. To address this, our framework, Speech-Hands, recasts the problem as an explicit self-reflection decision. This learnable reflection primitive proves effective in preventing the model from being derailed by flawed external candidates. We show that this agentic action mechanism generalizes naturally from speech recognition to complex, multiple-choice audio reasoning. Across the OpenASR leaderboard, Speech-Hands consistently outperforms strong baselines by 12.1% WER on seven benchmarks. The model also achieves 77.37% accuracy and high F1 on audio QA decisions, showing robust generalization and reliability across diverse audio question answering datasets. By unifying perception and decision-making, our work offers a practical path toward more reliable and resilient audio intelligence.
>
---
#### [replaced 149] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出EvolveR框架，解决LLM代理无法从自身经验中系统学习的问题。通过离线自蒸馏和在线交互的闭环流程，提升代理的自主学习能力。属于增强智能代理自主性的研究。**

- **链接: [https://arxiv.org/pdf/2510.16079](https://arxiv.org/pdf/2510.16079)**

> **作者:** Rong Wu; Xiaoman Wang; Jianbiao Mei; Pinlong Cai; Daocheng Fu; Cheng Yang; Licheng Wen; Xuemeng Yang; Yufan Shen; Yuxin Wang; Botian Shi
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Current Large Language Model (LLM) agents show strong performance in tool use, but lack the crucial capability to systematically learn from their own experiences. While existing frameworks mainly focus on mitigating external knowledge gaps, they fail to address a more fundamental limitation: the inability to iteratively refine problem-solving strategies. In this work, we introduce EvolveR, a framework designed to enable agent to self-improve through a complete, closed-loop experience lifecycle. This lifecycle comprises two key stages: (1) Offline Self-Distillation, where the agent's interaction trajectories are synthesized into a structured repository of abstract, reusable strategic principles; (2) Online Interaction, where the agent interacts with tasks and actively retrieves distilled principles to guide its decision-making, accumulating a diverse set of behavioral trajectories. This loop employs a policy reinforcement mechanism to iteratively update the agent based on its performance. We demonstrate the effectiveness of EvolveR on complex multi-hop question-answering benchmarks, where it achieves superior performance over strong agentic baselines. Our work presents a comprehensive blueprint for agents that learn not only from external data but also from the consequences of their own actions, paving the way for more autonomous and continuously improving systems. Code is available at this https URL.
>
---
#### [replaced 150] DecoupleSearch: Decouple Planning and Search via Hierarchical Reward Modeling
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于RAG系统优化任务，旨在解决Agentic RAG中规划与搜索耦合带来的问题。提出DecoupleSearch框架，通过双价值模型解耦规划与搜索过程，提升系统效果。**

- **链接: [https://arxiv.org/pdf/2510.21712](https://arxiv.org/pdf/2510.21712)**

> **作者:** Hao Sun; Zile Qiao; Bo Wang; Guoxin Chen; Yingyan Hou; Yong Jiang; Pengjun Xie; Fei Huang; Yan Zhang
>
> **备注:** EMNLP 2025 Main Conference
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems have emerged as a pivotal methodology for enhancing Large Language Models (LLMs) through the dynamic integration of external knowledge. To further improve RAG's flexibility, Agentic RAG introduces autonomous agents into the workflow. However, Agentic RAG faces several challenges: (1) the success of each step depends on both high-quality planning and accurate search, (2) the lack of supervision for intermediate reasoning steps, and (3) the exponentially large candidate space for planning and searching. To address these challenges, we propose DecoupleSearch, a novel framework that decouples planning and search processes using dual value models, enabling independent optimization of plan reasoning and search grounding. Our approach constructs a reasoning tree, where each node represents planning and search steps. We leverage Monte Carlo Tree Search to assess the quality of each step. During inference, Hierarchical Beam Search iteratively refines planning and search candidates with dual value models. Extensive experiments across policy models of varying parameter sizes demonstrate the effectiveness of our method.
>
---
#### [replaced 151] ShareChat: A Dataset of Chatbot Conversations in the Wild
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文提出ShareChat数据集，用于研究不同平台聊天机器人的实际行为与性能。任务是评估多平台聊天对话，解决现有基准无法反映真实平台差异的问题。工作包括收集和分析跨平台对话数据。**

- **链接: [https://arxiv.org/pdf/2512.17843](https://arxiv.org/pdf/2512.17843)**

> **作者:** Yueru Yan; Tuc Nguyen; Bo Su; Melissa Lieffers; Thai Le
>
> **摘要:** By evaluating Large Language Models (LLMs) through uniform, text-only interfaces, current academic benchmarks obscure how the unique designs and affordances of distinct commercial platforms shape real-world user behavior and system performance. To bridge this gap, we present ShareChat, the first large-scale corpus of 142,808 conversations (660,293 turns) collected from publicly shared URLs on ChatGPT, Perplexity, Grok, Gemini, and Claude. ShareChat preserves native platform affordances, including citations, thinking traces, and code artifacts, across 95 languages and the period from April 2023 to October 2025, complementing existing corpora that homogenize these interactions. To demonstrate the dataset's evaluative utility, we present three case studies: a conversation completeness analysis assessing cross-platform differences in intent satisfaction, a source grounding analysis comparing citation strategies between search-augmented systems, and a temporal analysis revealing divergent response latency dynamics. Together, these analyses demonstrate research questions that are inaccessible to single-platform or stripped-affordance corpora. The dataset is publicly available.
>
---
#### [replaced 152] Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses
- **分类: cs.CL; cs.SE**

- **简介: 该论文提出Agentic Harness Engineering（AHE），解决编码代理Harness手动设计的问题，通过可观测性机制实现自动进化。**

- **链接: [https://arxiv.org/pdf/2604.25850](https://arxiv.org/pdf/2604.25850)**

> **作者:** Jiahang Lin; Shichun Liu; Chengjun Pan; Lizhi Lin; Shihan Dou; Zhiheng Xi; Xuanjing Huang; Hang Yan; Zhenhua Han; Tao Gui; Yu-Gang Jiang
>
> **摘要:** Harnesses are now central to coding-agent performance, mediating how models interact with tools and execution environments. Yet harness engineering remains a manual craft, because automating it faces a heterogeneous action space across editable components, voluminous trajectories that bury actionable signal, and edits whose effect is hard to attribute. We introduce Agentic Harness Engineering (AHE), a closed loop that addresses these challenges through three matched observability pillars: (1) component observability gives every editable harness component a file-level representation so the action space is explicit and revertible; (2) experience observability distills millions of raw trajectory tokens into a layered, drill-down evidence corpus that an evolving agent can actually consume; and (3) decision observability pairs every edit with a self-declared prediction, later verified against the next round's task-level outcomes. Together, these pillars turn every edit into a falsifiable contract, so harness evolution proceeds autonomously without collapsing into trial-and-error. Empirically, ten AHE iterations lift pass@1 on Terminal-Bench 2 from 69.7% to 77.0%, surpassing the human-designed harness Codex-CLI (71.9%) and the self-evolving baselines ACE and TF-GRPO. The frozen harness transfers without re-evolution: on SWE-bench-verified it tops aggregate success at 12% fewer tokens than the seed, and on Terminal-Bench 2 it yields +5.1 to +10.1pp cross-family gains across three alternate model families, indicating the evolved components encode general engineering experience rather than benchmark-specific tuning. Ablations localize the gain to tools, middleware, and long-term memory rather than the system prompt, suggesting factual harness structure transfers while prose-level strategy does not.
>
---
#### [replaced 153] UniversalRAG: Retrieval-Augmented Generation over Corpora of Diverse Modalities and Granularities
- **分类: cs.CL; cs.AI; cs.CV; cs.IR; cs.LG**

- **简介: 该论文属于信息检索与生成任务，旨在解决单一模态知识源无法满足多样化查询的问题。提出UniversalRAG框架，实现多模态、多粒度知识的高效检索与融合。**

- **链接: [https://arxiv.org/pdf/2504.20734](https://arxiv.org/pdf/2504.20734)**

> **作者:** Woongyeong Yeo; Kangsan Kim; Soyeong Jeong; Jinheon Baek; Sung Ju Hwang
>
> **备注:** ACL 2026. Project page : this https URL
>
> **摘要:** Retrieval-Augmented Generation (RAG) has shown substantial promise in improving factual accuracy by grounding model responses with external knowledge relevant to queries. However, most existing approaches are limited to a text-only corpus, and while recent efforts have extended RAG to other modalities such as images and videos, they typically operate over a single modality-specific corpus. In contrast, real-world queries vary widely in the type of knowledge they require, which a single type of knowledge source cannot address. To address this, we introduce UniversalRAG, an any-to-any RAG framework designed to retrieve and integrate knowledge from heterogeneous sources with diverse modalities and granularities. Specifically, motivated by the observation that forcing all modalities into a unified representation space derived from a single aggregated corpus causes a modality gap, where the retrieval tends to favor items from the same modality as the query, we propose modality-aware routing, which dynamically identifies the most appropriate modality-specific corpus and performs targeted retrieval within it, and further justify its effectiveness with a theoretical analysis. Moreover, beyond modality, we organize each modality into multiple granularity levels, enabling fine-tuned retrieval tailored to the complexity and scope of the query. We validate UniversalRAG on 10 benchmarks of multiple modalities, showing its superiority over various modality-specific and unified baselines.
>
---
#### [replaced 154] Sustainability via LLM Right-sizing
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI模型评估任务，旨在解决LLM部署中的可持续性问题。通过实验证明小型模型在多数任务中表现良好，提出基于任务需求的模型选择方法。**

- **链接: [https://arxiv.org/pdf/2504.13217](https://arxiv.org/pdf/2504.13217)**

> **作者:** Jennifer Haase; Finn Klessascheck; Jan Mendling; Sebastian Pokutta
>
> **备注:** 21 pages, 2 Figures, 6 Tables
>
> **摘要:** Large language models (LLMs) have become increasingly embedded in organizational workflows. This has raised concerns over their energy consumption, financial costs, and data sovereignty. While performance benchmarks often celebrate cutting-edge models, real-world deployment decisions require a broader perspective: when is a smaller, locally deployable model "good enough"? This study offers an empirical answer by evaluating eleven proprietary and open-weight LLMs across ten everyday occupational tasks, including summarizing texts, generating schedules, and drafting emails and proposals. Using a dual-LLM-based evaluation framework, we automated task execution and standardized evaluation across ten criteria related to output quality, factual accuracy, and ethical responsibility. Results show that GPT-4o delivers consistently superior performance but at a significantly higher cost and environmental footprint. Notably, smaller models like Gemma-3 and Phi-4 achieved strong and reliable results on most tasks, suggesting their viability in contexts requiring cost-efficiency, local deployment, or privacy. A cluster analysis revealed three model groups -- premium all-rounders, competent generalists, and limited but safe performers -- highlighting trade-offs between quality, control, and sustainability. Significantly, task type influenced model effectiveness: conceptual tasks challenged most models, while aggregation and transformation tasks yielded better performances. We argue for a shift from performance-maximizing benchmarks to task- and context-aware sufficiency assessments that better reflect organizational priorities. Our approach contributes a scalable method to evaluate AI models through a sustainability lens and offers actionable guidance for responsible LLM deployment in practice.
>
---
#### [replaced 155] Training-Free Cultural Alignment of Large Language Models via Persona Disagreement
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于文化对齐任务，解决大语言模型道德判断的文化偏差问题。通过分析社会人口分歧，提出DISCA方法，在不调整权重的情况下提升跨文化一致性。**

- **链接: [https://arxiv.org/pdf/2605.10843](https://arxiv.org/pdf/2605.10843)**

> **作者:** Huynh Trung Kiet; Dao Sy Duy Minh; Tuan Nguyen; Chi-Nguyen Tran; Phu-Hoa Pham; Nguyen Lam Phu Quy; Anh Han; Long Tran-Thanh
>
> **备注:** 57 pages, 1 figure, 6 MultiTP moral dimensions
>
> **摘要:** Large language models increasingly mediate decisions that turn on moral judgement, yet a growing body of evidence shows that their implicit preferences are not culturally neutral. Existing cultural alignment methods either require per-country preference data and fine-tuning budgets or assume white-box access to model internals that commercial APIs do not expose. In this work, we focus on this realistic black-box, public-data-only regime and observe that within-country sociodemographic disagreement, not consensus, is the primary steering signal. We introduce DISCA (Disagreement-Informed Steering for Cultural Alignment), an inference-time method that instantiates each country as a panel of World-Values-Survey-grounded persona agents and converts their disagreement into a bounded, loss-averse logit correction. Across 20 countries and 7 open-weight backbones (2B--70B), DISCA reduces cultural misalignment on MultiTP by 10--24% on the six backbones >=3.8B, and 2--7% on open-ended scenarios, without changing any weights. Our results suggest that inference-time calibration is a scalable alternative to fine-tuning for serving the long tail of global moral preferences.
>
---
#### [replaced 156] Leveraging Multimodal Self-Consistency Reasoning in Coding Motivational Interviewing for Alcohol Use Reduction
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决酒精使用减少的动机访谈自动编码问题。通过多模态自洽推理提升编码准确性。**

- **链接: [https://arxiv.org/pdf/2605.12987](https://arxiv.org/pdf/2605.12987)**

> **作者:** Guangzeng Han; James G. Murphy; Benjamin O. Ladd; Xiaolei Huang; Brian Borsari
>
> **摘要:** BACKGROUND: Coding Motivational Interviewing (MI) sessions is essential for understanding client behaviors and predicting outcomes, but it requires substantial time and labor from trained MI professionals. Recent advances in audio-language models (ALMs) offer new opportunities to automate MI coding by capturing multimodal behavioral signals. OBJECTIVE: This study aims to develop an automatic MI coding approach based on ALMs that analyzes raw audio input and integrates predictions from multiple reasoning trajectories using self-consistency to improve coding robustness. METHODS: We experimented with five recorded sessions from de-identified MI audio tapes. We deployed ALMs with four complementary analytic prompts to support utterance-level reasoning: analytic prompting for verbal cues, prosody-aware prompting for acoustic cues, evidence-scoring prompting for quantitative hypothesis testing, and comparative prompting for contrastive reasoning. Three stochastic samples were drawn for each prompt, generating 12 independent reasoning trajectories per utterance. Final predictions were determined by majority voting across all trajectories. RESULTS: Performance was evaluated using accuracy, precision, recall, and macro-F1 scores. The proposed multimodal self-consistency approach achieved 52.56% accuracy, 54.03% precision, 47.45% recall, and a macro-F1 score of 46.40%, exceeding baseline methods. Systematic ablation experiments that removed individual modules consistently degraded performance on the primary metrics. CONCLUSIONS: Multimodal self-consistency outperforms single-pass baseline prompting approaches for MI coding. These findings suggest that incorporating both what clients say and how they say it can support more reliable automatic MI coding.
>
---
#### [replaced 157] We Think, Therefore We Align LLMs to Helpful, Harmless and Honest Before They Go Wrong
- **分类: cs.CL**

- **简介: 该论文属于大语言模型对齐任务，解决多目标（Helpfulness, Harmlessness, Honesty）冲突问题。提出AMBS框架，在共享表示基础上实现多目标响应生成。**

- **链接: [https://arxiv.org/pdf/2509.22510](https://arxiv.org/pdf/2509.22510)**

> **作者:** Gautam Siddharth Kashyap; Mark Dras; Usman Naseem
>
> **摘要:** Alignment of Large Language Models (LLMs) is the ability to satisfy desired objectives during generation, which is critical for trustworthy deployment. In practice, alignment is often operationalized through multiple objectives such as Helpfulness, Harmlessness, and Honesty (HHH). Prior works study alignment via steering vectors in standard Transformer decoders but treat objectives in isolation, where optimizing a single objective can overwrite others, leading to interference. Recent works attempt to address this limitation by extending steering to a 1-to-N Transformer setting by replicating representations into objective-specific pathways, but apply transformations independently, resulting in inconsistent responses across objectives. Similarly, approaches such as safe RLHF and MoE-based designs study trade-offs across objectives but do not constrain objective-specific transformations within a shared representation during inference. As a result, even aligned State-of-the-Art (SOTA) LLMs can struggle to jointly satisfy HHH objectives in complex settings. To address this, we propose Adaptive Multi-Branch Steering (AMBS), a two-stage framework in a 1-to-N Transformer setting that parameterizes objective-specific transformations relative to a shared representation. In Stage I, a shared hidden representation is computed once. In Stage II, this representation is replicated into N pathways and updated relative to a shared reference, capturing objective-specific deviations while restricting divergence. This produces N objective-specific responses within a single forward pass, which can be combined at decoding to obtain a single response across objectives. Across multiple backbones, AMBS improves performance across HHH, with consistent gains in WR, TI, and SS (e.g., Avg 56.5% on LLaMA-2-7B) while maintaining efficiency (e.g., 189 Tok/s, 9 GPU-hrs).
>
---
#### [replaced 158] Probing Persona-Dependent Preferences in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型中不同人格的偏好机制，通过线性探测器分析模型内部偏好表示，探讨其在不同人格间的共享性。**

- **链接: [https://arxiv.org/pdf/2605.13339](https://arxiv.org/pdf/2605.13339)**

> **作者:** Oscar Gilg; Pierre Beckmann; Daniel Paleka; Patrick Butlin
>
> **备注:** 41 pages, 45 figures. Code: this https URL. Earlier write-up on LessWrong: this https URL
>
> **摘要:** Large language models (LLMs) can be said to have preferences: they reliably pick certain tasks and outputs over others, and preferences shaped by post-training and system prompts appear to shape much of their behaviour. But models can also adopt different personas which have radically different preferences. How is this implemented internally? Does each persona run on its own preference machinery, or is something shared underneath? We train linear probes on residual-stream activations of Gemma-3-27B and Qwen-3.5-122B to predict revealed pairwise task choices, and identify a genuine preference vector: it tracks the model's preferences as they shift across a range of prompts and situations, and on Gemma-3-27B steering along it causally controls pairwise choice. This preference representation is largely shared across personas: a probe trained on the helpful assistant predicts and steers the choices of qualitatively different personas, including an evil persona whose preferences anti-correlate with those of the Assistant.
>
---
#### [replaced 159] You Had One Job: Per-Task Quantization Using LLMs' Hidden Representations
- **分类: cs.CL**

- **简介: 该论文提出TAQ框架，解决LLM量化中任务无关的精度分配问题，通过任务相关提示优化混合精度量化，提升模型效率与准确性。**

- **链接: [https://arxiv.org/pdf/2511.06516](https://arxiv.org/pdf/2511.06516)**

> **作者:** Amit LeVi; Raz Lapid; Rom Himelstein; Chaim Baskin; Ravid Shwartz Ziv; Avi Mendelson
>
> **摘要:** Many LLM applications require only narrow capabilities, yet standard post-training quantization (PTQ) methods allocate precision without considering the target task. This can waste bits on layers that are less relevant to the task signal while over-compressing layers that are critical for downstream behavior. We propose Task-Aware Quantization (TAQ), a training-free, weight-only mixed-precision PTQ framework that uses a small set of unlabeled task calibration prompts to allocate higher precision to task-relevant transformer layers under a fixed bit budget. TAQ estimates layer importance from hidden representations and output sensitivity, and we instantiate it with three scoring rules: TAQ-IS, based on activation information and stability; TAQ-KL, based on output-distribution sensitivity under a quantization-noise proxy; and TAQ-O, a label-informed oracle diagnostic for analyzing layer sensitivity. Across several benchmarks, TAQ outperforms task-agnostic baselines such in most settings, with especially strong gains in the accuracy--memory ratio. We further validate that these gains translate to real deployment behavior through hardware throughput and latency measurements, and analyze calibration robustness and residual-stream error propagation. Overall, TAQ turns mixed-precision PTQ from a model-centric compression step into a task-conditioned precision-allocation problem. A reference implementation is available at this https URL.
>
---
#### [replaced 160] SlimQwen: Exploring the Pruning and Distillation in Large MoE Model Pre-training
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决大规模MoE模型的剪枝与蒸馏问题。通过系统研究，提出有效压缩策略，提升模型效率与性能。**

- **链接: [https://arxiv.org/pdf/2605.08738](https://arxiv.org/pdf/2605.08738)**

> **作者:** Shengkun Tang; Zekun Wang; Bo Zheng; Liangyu Wang; Rui Men; Siqi Zhang; Xiulong Yuan; Zihan Qiu; Zhiqiang Shen; Dayiheng Liu
>
> **摘要:** Structured pruning and knowledge distillation (KD) are typical techniques for compressing large language models, but it remains unclear how they should be applied at pretraining scale, especially to recent mixture-of-experts (MoE) models. In this work, we systematically study MoE compression in large-scale pretraining, focusing on three key questions: whether pruning provides a better initialization than training from scratch, how expert compression choices affect the final model after continued training, and which training strategy is most effective. We have the following findings: First, across depth, width, and expert compression, pruning a pretrained MoE consistently outperforms training the target architecture from scratch under the same training budget. Second, different one-shot expert compression methods converge to similar final performance after large-scale continual pretraining. Motivated by this, we introduce a simple partial-preservation expert merging strategy that improves downstream performance across most benchmarks. Third, combining KD with the language modeling loss outperforms KD alone, particularly on knowledge-intensive tasks. We further propose multi-token prediction (MTP) distillation, which yields consistent gains. Finally, given the same training tokens, progressive pruning schedules outperform one-shot compression, suggesting that gradual architecture transitions lead to better optimization trajectories. Putting it all together, we compress Qwen3-Next-80A3B to a 23A2B model that retains competitive performance. These results offer practical guidance for efficient MoE compression at scale.
>
---
#### [replaced 161] Learning to Reason without External Rewards
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决依赖外部奖励的局限性。提出RLIF框架，使用模型自信心作为内在奖励，实现无监督学习。**

- **链接: [https://arxiv.org/pdf/2505.19590](https://arxiv.org/pdf/2505.19590)**

> **作者:** Xuandong Zhao; Zhewei Kang; Aosong Feng; Sergey Levine; Dawn Song
>
> **备注:** ICLR 2026
>
> **摘要:** Training large language models (LLMs) for complex reasoning via Reinforcement Learning with Verifiable Rewards (RLVR) is effective but limited by reliance on costly, domain-specific supervision. We explore Reinforcement Learning from Internal Feedback (RLIF), a framework that enables LLMs to learn from intrinsic signals without external rewards or labeled data. We propose Intuitor, an RLIF method that uses a model's own confidence-termed self-certainty-as its sole reward signal. Intuitor replaces external rewards in Group Relative Policy Optimization (GRPO) with self-certainty scores, enabling fully unsupervised learning. Experiments demonstrate that Intuitor matches GRPO's performance on mathematical benchmarks while achieving better generalization to out-of-domain tasks like code generation, without requiring gold solutions or test cases. Our findings show that intrinsic model signals can drive effective learning across domains, offering a scalable alternative to RLVR for autonomous AI systems where verifiable rewards are unavailable. Code is available at this https URL
>
---
#### [replaced 162] Evo-Memory: Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于LLM代理的内存管理任务，旨在解决模型在连续任务中无法有效积累和利用经验的问题。工作包括构建Evo-Memory基准框架，评估多种记忆模块，并提出ExpRAG和ReMem方法提升记忆演化能力。**

- **链接: [https://arxiv.org/pdf/2511.20857](https://arxiv.org/pdf/2511.20857)**

> **作者:** Tianxin Wei; Noveen Sachdeva; Benjamin Coleman; Zhankui He; Yuanchen Bei; Xuying Ning; Mengting Ai; Yunzhe Li; Jingrui He; Ed H. Chi; Chi Wang; Shuo Chen; Fernando Pereira; Wang-Cheng Kang; Derek Zhiyuan Cheng
>
> **摘要:** Statefulness is essential for large language model (LLM) agents to perform long-term planning and problem-solving. This makes memory a critical component, yet its management and evolution remain largely underexplored. Existing evaluations mostly focus on static conversational settings, where memory is passively retrieved from dialogue to answer queries, overlooking the dynamic ability to accumulate and reuse experience across evolving task streams. In real-world environments such as interactive problem assistants or embodied agents, LLMs are required to handle continuous task streams, yet often fail to learn from accumulated interactions, losing valuable contextual insights, a limitation that calls for test-time evolution, where LLMs retrieve, integrate, and update memory continuously during deployment. To bridge this gap, we introduce Evo-Memory, a comprehensive streaming benchmark and framework for evaluating self-evolving memory in LLM agents. Evo-Memory structures datasets into sequential task streams, requiring LLMs to search, adapt, and evolve memory after each interaction. We unify and implement over ten representative memory modules and evaluate them across 10 diverse multi-turn goal-oriented and single-turn reasoning and QA datasets. To better benchmark experience reuse, we provide a baseline method, ExpRAG, for retrieving and utilizing prior experience, and further propose ReMem, an action-think-memory refine pipeline that tightly integrates reasoning, task actions, and memory updates to achieve continual improvement.
>
---
#### [replaced 163] Evaluating Language Models' Evaluations of Games
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI评估任务，旨在研究语言模型对游戏的评价能力。通过对比模型与人类及符号代理的评价，分析模型在复杂性和量化难度上的表现。**

- **链接: [https://arxiv.org/pdf/2510.10930](https://arxiv.org/pdf/2510.10930)**

> **作者:** Katherine M. Collins; Cedegao E. Zhang; Graham Todd; Lance Ying; Mauricio Barba da Costa; Ryan Liu; Prafull Sharma; Adrian Weller; Ionatan Kuperwajs; Lionel Wong; Joshua B. Tenenbaum; Thomas L. Griffiths
>
> **摘要:** Reasoning is not just about solving problems -- it is also about evaluating which problems are worth solving at all. Evaluations of artificial intelligence (AI) systems primarily focused on problem solving, historically by studying how models play games such as chess and Go. In this paper, we advocate for a new paradigm that assesses AI systems' evaluation of games. First, we introduce a formalism for evaluating such evaluations. We then leverage a large-scale dataset of over 100 novel board games and over 450 human judgments to compare evaluations produced by modern language and reasoning models against those of people and symbolic computational agents. We consider two kinds of evaluative queries: assessing the payoff (or fairness) and the funness of games. These queries span two dimensions relevant to the design of evaluations of AI evaluations: how complex a query is to compute and how difficult a query is to quantify. Our results show that reasoning models are generally more aligned to people in their evaluations of games than non-reasoning language models. However, we observe a non-monotonic relationship: as models get closer to game-theoretic optimal, their fit to human data weakens. We also observe more "jaggedness" across models for assessing funness, in line with the greater difficulty of quantifying this query. Across queries and games, reasoning models show highly variable and unpredictable resource usage when assessing queries, pointing to the importance of imbuing more resource-rational meta-reasoning in language and reasoning models.
>
---
#### [replaced 164] Supervising the search process produces reliable and generalizable information-seeking agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息检索任务，旨在解决搜索代理效果不佳的问题。通过过程监督提升搜索质量，提出RAG-Gym框架和Re$^2$Search++模型，增强代理的泛化能力。**

- **链接: [https://arxiv.org/pdf/2502.13957](https://arxiv.org/pdf/2502.13957)**

> **作者:** Guangzhi Xiong; Qiao Jin; Xiao Wang; Yin Fang; Haolin Liu; Yifan Yang; Fangyuan Chen; Zhixing Song; Dengyu Wang; Minjia Zhang; Zhiyong Lu; Aidong Zhang
>
> **备注:** Homepage: this https URL; Code: this https URL
>
> **摘要:** Large language models (LLMs) are transforming web search by shifting from document ranking to synthesizing answers, and are increasingly deployed as autonomous agentic search systems that iteratively interact with external knowledge sources. Despite this progress, building effective search agents remains challenging because high-quality intermediate search steps are difficult to generate. Previous approaches have primarily relied on outcome supervision, rewarding agents only for producing correct final answers. This often leads to reward hacking and excessive dependence on parametric memory, limiting generalization to out-of-domain tasks. To address these limitations, we introduce RAG-Gym, a framework that shifts supervision from final answers to the search process itself. With RAG-Gym, we systematically investigate architecture design, parameter optimization, and action evaluation, identifying reasoning reflection as a critical capability for search agents. Building on this insight, we propose Re$^2$Search++, a process-supervised agent that achieves substantial improvements on multi-hop information-seeking benchmarks, especially in out-of-domain settings. Performance gains are driven primarily by higher-quality search queries rather than answer optimization alone, and the learned search critics transfer across models, including proprietary LLMs. These findings show that supervising the search process produces more reliable and generalizable information-seeking agents.
>
---
#### [replaced 165] Patients Speak, AI Listens: LLM-based Analysis of Online Reviews Uncovers Key Drivers for Urgent Care Satisfaction
- **分类: cs.CL; cs.AI; cs.SI**

- **简介: 论文通过分析在线评论，研究影响急诊护理满意度的关键因素。属于情感分析任务，解决传统调查方法不足的问题，利用LLM提取患者意见并探讨社会经济因素的影响。**

- **链接: [https://arxiv.org/pdf/2503.20981](https://arxiv.org/pdf/2503.20981)**

> **作者:** Xiaoran Xu; Zhaoqian Xue; Chi Zhang; Jhonatan Medri; Junjie Xiong; Jiayan Zhou; Jin Jin; Yongfeng Zhang; Siyuan Ma; Lingyao Li
>
> **摘要:** Investigating the public experience of urgent care facilities is essential for promoting community healthcare development. Traditional survey methods often fall short due to limited scope, time, and spatial coverage. Crowdsourcing through online reviews or social media offers a valuable approach to gaining such insights. With recent advancements in large language models (LLMs), extracting nuanced perceptions from reviews has become feasible. This study collects Google Maps reviews across the DMV and Florida areas and conducts prompt engineering with the GPT model to analyze the aspect-based sentiment of urgent care. We first analyze the geospatial patterns of various aspects, including interpersonal factors, operational efficiency, technical quality, finances, and facilities. Next, we determine Census Block Group (CBG)-level characteristics underpinning differences in public perception, including population density, median income, GINI Index, rent-to-income ratio, household below poverty rate, no insurance rate, and unemployment rate. Our results show that interpersonal factors and operational efficiency emerge as the strongest determinants of patient satisfaction in urgent care, while technical quality, finances, and facilities show no significant independent effects when adjusted for in multivariate models. Among socioeconomic and demographic factors, only population density demonstrates a significant but modest association with patient ratings, while the remaining factors exhibit no significant correlations. Overall, this study highlights the potential of crowdsourcing to uncover the key factors that matter to residents and provide valuable insights for stakeholders to improve public satisfaction with urgent care.
>
---
#### [replaced 166] From Isolated Scoring to Collaborative Ranking: A Comparison-Native Framework for LLM-Based Paper Evaluation
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于科学论文评价任务，旨在解决LLM因绝对评分导致的判断不鲁棒问题。提出CNPE框架，通过对比学习实现协作排序，提升评价效果。**

- **链接: [https://arxiv.org/pdf/2603.17588](https://arxiv.org/pdf/2603.17588)**

> **作者:** Pujun Zheng; Jiacheng Yao; Jinquan Zheng; Chenyang Gu; Guoxiu He; Jiawei Liu; Yong Huang; Tianrui Guo; Wei Lu
>
> **备注:** Accepted at Findings of ACL 2026
>
> **摘要:** Large language models (LLMs) are currently applied to scientific paper evaluation by assigning an absolute score to each paper independently. However, since score scales vary across conferences, time periods, and evaluation criteria, models trained on absolute scores are prone to fitting narrow, context-specific rules rather than developing robust scholarly judgment. To overcome this limitation, we propose shifting paper evaluation from isolated scoring to collaborative ranking. In particular, we design a $\textbf{C}$omparison-$\textbf{N}$ative framework for $\textbf{P}$aper $\textbf{E}$valuation ($\textbf{CNPE}$), integrating comparison into both data construction and model learning. We first propose a graph-based similarity ranking algorithm to facilitate the sampling of more informative and discriminative paper pairs from a collection. We then enhance relative quality judgment through supervised fine-tuning and reinforcement learning with comparison-based rewards. At inference, the model performs pairwise comparisons over sampled paper pairs and aggregates these preference signals into a global relative quality ranking. Experimental results demonstrate that our framework achieves an average relative improvement of 21.8% over the strong baseline DeepReview-14B, while exhibiting robust generalization to five previously unseen datasets. Our code is available at this https URL.
>
---
