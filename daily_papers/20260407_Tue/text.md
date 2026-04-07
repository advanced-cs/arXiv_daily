# 自然语言处理 cs.CL

- **最新发布 135 篇**

- **更新 81 篇**

## 最新发布

#### [new 001] Metaphors We Compute By: A Computational Audit of Cultural Translation vs. Thinking in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 论文属于自然语言处理任务，探讨LLMs是否具备文化意识推理。研究通过比喻生成任务，检验模型是否真正理解文化差异，而非仅作文化翻译。结果发现模型存在刻板印象和西方中心倾向。**

- **链接: [https://arxiv.org/pdf/2604.04732](https://arxiv.org/pdf/2604.04732)**

> **作者:** Yuan Chang; Jiaming Qu; Zhu Li
>
> **摘要:** Large language models (LLMs) are often described as multilingual because they can understand and respond in many languages. However, speaking a language is not the same as reasoning within a culture. This distinction motivates a critical question: do LLMs truly conduct culture-aware reasoning? This paper presents a preliminary computational audit of cultural inclusivity in a creative writing task. We empirically examine whether LLMs act as culturally diverse creative partners or merely as cultural translators that leverage a dominant conceptual framework with localized expressions. Using a metaphor generation task spanning five cultural settings and several abstract concepts as a case study, we find that the model exhibits stereotyped metaphor usage for certain settings, as well as Western defaultism. These findings suggest that merely prompting an LLM with a cultural identity does not guarantee culturally grounded reasoning.
>
---
#### [new 002] LightThinker++: From Reasoning Compression to Memory Management
- **分类: cs.CL; cs.AI; cs.IR; cs.LG; cs.MM**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在长序列推理中的效率与准确性问题。通过引入动态压缩和显式记忆管理，提升模型在复杂任务中的表现与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.03679](https://arxiv.org/pdf/2604.03679)**

> **作者:** Yuqi Zhu; Jintian Zhang; Zhenjie Wan; Yujie Luo; Shuofei Qiao; Zhengke Gui; Da Zheng; Lei Liang; Huajun Chen; Ningyu Zhang
>
> **备注:** Work in progress. This is an extended version of LightThinker
>
> **摘要:** Large language models (LLMs) excel at complex reasoning, yet their efficiency is limited by the surging cognitive overhead of long thought traces. In this paper, we propose LightThinker, a method that enables LLMs to dynamically compress intermediate thoughts into compact semantic representations. However, static compression often struggles with complex reasoning where the irreversible loss of intermediate details can lead to logical bottlenecks. To address this, we evolve the framework into LightThinker++, introducing Explicit Adaptive Memory Management. This paradigm shifts to behavioral-level management by incorporating explicit memory primitives, supported by a specialized trajectory synthesis pipeline to train purposeful memory scheduling. Extensive experiments demonstrate the framework's versatility across three dimensions. (1) LightThinker reduces peak token usage by 70% and inference time by 26% with minimal accuracy loss. (2) In standard reasoning, LightThinker++ slashes peak token usage by 69.9% while yielding a +2.42% accuracy gain under the same context budget for maximum performance. (3) Most notably, in long-horizon agentic tasks, it maintains a stable footprint beyond 80 rounds (a 60%-70% reduction), achieving an average performance gain of 14.8% across different complex scenarios. Overall, our work provides a scalable direction for sustaining deep LLM reasoning over extended horizons with minimal overhead.
>
---
#### [new 003] High-Stakes Personalization: Rethinking LLM Customization for Individual Investor Decision-Making
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于个性化自然语言处理任务，旨在解决高风险投资决策中的LLM定制问题，提出四个挑战并探讨相应架构方案。**

- **链接: [https://arxiv.org/pdf/2604.04300](https://arxiv.org/pdf/2604.04300)**

> **作者:** Yash Ganpat Sawant
>
> **备注:** 4 pages + 1 page references. Submitted to CustomNLP4U Workshop @ ACL 2026
>
> **摘要:** Personalized LLM systems have advanced rapidly, yet most operate in domains where user preferences are stable and ground truth is either absent or subjective. We argue that individual investor decision-making presents a uniquely challenging domain for LLM personalization - one that exposes fundamental limitations in current customization paradigms. Drawing on our system, built and deployed for AI-augmented portfolio management, we identify four axes along which individual investing exposes fundamental limitations in standard LLM customization: (1) behavioral memory complexity, where investor patterns are temporally evolving, self-contradictory, and financially consequential; (2) thesis consistency under drift, where maintaining coherent investment rationale over weeks or months strains stateless and session-bounded architectures; (3) style-signal tension, where the system must simultaneously respect personal investment philosophy and surface objective evidence that may contradict it; and (4) alignment without ground truth, where personalization quality cannot be evaluated against a fixed label set because outcomes are stochastic and delayed. We describe the architectural responses that emerged from building the system and propose open research directions for personalized NLP in high-stakes, temporally extended decision domains.
>
---
#### [new 004] Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection
- **分类: cs.CL**

- **简介: 该论文属于文本检测任务，旨在解决LLM生成文本的细粒度分类问题。提出RACE方法，区分创作者与编辑者角色，提升检测精度。**

- **链接: [https://arxiv.org/pdf/2604.04932](https://arxiv.org/pdf/2604.04932)**

> **作者:** Yang Li; Qiang Sheng; Zhengjia Wang; Yehan Yang; Danding Wang; Juan Cao
>
> **备注:** ACL 2026 Accepted Paper
>
> **摘要:** The misuse of large language models (LLMs) requires precise detection of synthetic text. Existing works mainly follow binary or ternary classification settings, which can only distinguish pure human/LLM text or collaborative text at best. This remains insufficient for the nuanced regulation, as the LLM-polished human text and humanized LLM text often trigger different policy consequences. In this paper, we explore fine-grained LLM-generated text detection under a rigorous four-class setting. To handle such complexities, we propose RACE (Rhetorical Analysis for Creator-Editor Modeling), a fine-grained detection method that characterizes the distinct signatures of creator and editor. Specifically, RACE utilizes Rhetorical Structure Theory to construct a logic graph for the creator's foundation while extracting Elementary Discourse Unit-level features for the editor's style. Experiments show that RACE outperforms 12 baselines in identifying fine-grained types with low false alarms, offering a policy-aligned solution for LLM regulation.
>
---
#### [new 005] Do No Harm: Exposing Hidden Vulnerabilities of LLMs via Persona-based Client Simulation Attack in Psychological Counseling
- **分类: cs.CL**

- **简介: 该论文属于安全评估任务，旨在解决LLMs在心理辅导中可能存在的隐性漏洞问题。通过构建基于人格的客户模拟攻击框架（PCSA），揭示模型在心理安全对齐方面的不足。**

- **链接: [https://arxiv.org/pdf/2604.04842](https://arxiv.org/pdf/2604.04842)**

> **作者:** Qingyang Xu; Yaling Shen; Stephanie Fong; Zimu Wang; Yiwen Jiang; Xiangyu Zhao; Jiahe Liu; Zhongxing Xu; Vincent Lee; Zongyuan Ge
>
> **摘要:** The increasing use of large language models (LLMs) in mental healthcare raises safety concerns in high-stakes therapeutic interactions. A key challenge is distinguishing therapeutic empathy from maladaptive validation, where supportive responses may inadvertently reinforce harmful beliefs or behaviors in multi-turn conversations. This risk is largely overlooked by existing red-teaming frameworks, which focus mainly on generic harms or optimization-based attacks. To address this gap, we introduce Personality-based Client Simulation Attack (PCSA), the first red-teaming framework that simulates clients in psychological counseling through coherent, persona-driven client dialogues to expose vulnerabilities in psychological safety alignment. Experiments on seven general and mental health-specialized LLMs show that PCSA substantially outperforms four competitive baselines. Perplexity analysis and human inspection further indicate that PCSA generates more natural and realistic dialogues. Our results reveal that current LLMs remain vulnerable to domain-specific adversarial tactics, providing unauthorized medical advice, reinforcing delusions, and implicitly encouraging risky actions.
>
---
#### [new 006] PassiveQA: A Three-Action Framework for Epistemically Calibrated Question Answering via Supervised Finetuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PassiveQA框架，解决问答任务中信息不足时的决策问题，通过监督微调使模型能选择回答、提问或回避，提升准确性并减少幻觉。**

- **链接: [https://arxiv.org/pdf/2604.04565](https://arxiv.org/pdf/2604.04565)**

> **作者:** Madhav S Baidya
>
> **备注:** 32 pages, 4 figures. Includes experiments on four QA datasets and a knowledge graph-based finetuning pipeline. Code available at: this https URL
>
> **摘要:** Large Language Models (LLMs) have achieved strong performance in question answering and retrieval-augmented generation (RAG), yet they implicitly assume that user queries are fully specified and answerable. In real-world settings, queries are often incomplete, ambiguous, or missing critical variables, leading models to produce overconfident or hallucinated responses. In this work, we study decision-aware query resolution under incomplete information, where a model must determine whether to Answer, Ask for clarification, or Abstain. We show that standard and enhanced RAG systems do not reliably exhibit such epistemic awareness, defaulting to answer generation even when information is insufficient. To address this, we propose PassiveQA, a three-action framework that aligns model behaviour with information sufficiency through supervised finetuning. Our approach integrates structured information-state representations, knowledge graph-grounded context, and a finetuned planner that explicitly models missing variables and decision reasoning. Experiments across multiple QA datasets show that the finetuned planner achieves significant improvements in macro F1 and abstention recall while reducing hallucination rates, under a compute-constrained training regime. These results provide strong empirical evidence that epistemic decision-making must be learned during training rather than imposed at inference time.
>
---
#### [new 007] Evolutionary Search for Automated Design of Uncertainty Quantification Methods
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自动化设计任务，旨在解决手动设计不确定性量化方法的局限性。通过LLM驱动的进化搜索，自动发现Python程序形式的UQ方法，并在多个数据集上取得更好效果。**

- **链接: [https://arxiv.org/pdf/2604.03473](https://arxiv.org/pdf/2604.03473)**

> **作者:** Mikhail Seleznyov; Daniil Korbut; Viktor Moskvoretskii; Oleg Somov; Alexander Panchenko; Elena Tutubalina
>
> **摘要:** Uncertainty quantification (UQ) methods for large language models are predominantly designed by hand based on domain knowledge and heuristics, limiting their scalability and generality. We apply LLM-powered evolutionary search to automatically discover unsupervised UQ methods represented as Python programs. On the task of atomic claim verification, our evolved methods outperform strong manually-designed baselines, achieving up to 6.7% relative ROC-AUC improvement across 9 datasets while generalizing robustly out-of-distribution. Qualitative analysis reveals that different LLMs employ qualitatively distinct evolutionary strategies: Claude models consistently design high-feature-count linear estimators, while Gpt-oss-120B gravitates toward simpler and more interpretable positional weighting schemes. Surprisingly, only Sonnet 4.5 and Opus 4.5 reliably leverage increased method complexity to improve performance -- Opus 4.6 shows an unexpected regression relative to its predecessor. Overall, our results indicate that LLM-powered evolutionary search is a promising paradigm for automated, interpretable hallucination detector design.
>
---
#### [new 008] Testing the Limits of Truth Directions in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型中真理方向的通用性问题，探讨其在不同层、任务类型和指令下的表现，揭示其局限性。**

- **链接: [https://arxiv.org/pdf/2604.03754](https://arxiv.org/pdf/2604.03754)**

> **作者:** Angelos Poulis; Mark Crovella; Evimaria Terzi
>
> **摘要:** Large language models (LLMs) have been shown to encode truth of statements in their activation space along a linear truth direction. Previous studies have argued that these directions are universal in certain aspects, while more recent work has questioned this conclusion drawing on limited generalization across some settings. In this work, we identify a number of limits of truth-direction universality that have not been previously understood. We first show that truth directions are highly layer-dependent, and that a full understanding of universality requires probing at many layers in the model. We then show that truth directions depend heavily on task type, emerging in earlier layers for factual and later layers for reasoning tasks; they also vary in performance across levels of task complexity. Finally, we show that model instructions dramatically affect truth directions; simple correctness evaluation instructions significantly affect the generalization ability of truth probes. Our findings indicate that universality claims for truth directions are more limited than previously known, with significant differences observable for various model layers, task difficulties, task types, and prompt templates.
>
---
#### [new 009] CresOWLve: Benchmarking Creative Problem-Solving Over Real-World Knowledge
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CresOWLve基准，用于评估基于真实知识的创造性问题解决能力。任务是解决传统基准无法全面反映真实创造过程的问题，通过设计需多策略、跨领域知识整合的谜题来挑战大语言模型。**

- **链接: [https://arxiv.org/pdf/2604.03374](https://arxiv.org/pdf/2604.03374)**

> **作者:** Mete Ismayilzada; Renqing Cuomao; Daniil Yurshevich; Anna Sotnikova; Lonneke van der Plas; Antoine Bosselut
>
> **备注:** Under review
>
> **摘要:** Creative problem-solving requires combining multiple cognitive abilities, including logical reasoning, lateral thinking, analogy-making, and commonsense knowledge, to discover insights that connect seemingly unrelated pieces of information. However, most existing benchmarks for large language models (LLMs) evaluate only specific components of this process. Moreover, many creativity-oriented benchmarks rely on artificially constructed brainteasers or contrived scenarios that do not reflect how creative problem-solving occurs in real-world settings. To address this gap, we introduce CresOWLve, a benchmark for evaluating creative problem-solving using puzzles grounded in real-world knowledge. Problems in CresOWLve require employing multiple creative thinking strategies, retrieving facts from diverse domains, and creatively combining them to arrive at a solution. Evaluating several frontier non-thinking and thinking LLMs, we show that CresOWLve remains highly challenging. Our analysis reveals a consistent performance gap: models perform substantially better on factual questions than on creative ones (up to a -17% drop). While models can often retrieve the relevant knowledge, they struggle to form the non-obvious creative connections required to integrate this information and arrive at the correct answer.
>
---
#### [new 010] From Plausible to Causal: Counterfactual Semantics for Policy Evaluation in Simulated Online Communities
- **分类: cs.CL**

- **简介: 该论文属于政策评估任务，解决仿真社区中因果关系不足的问题。提出因果反事实框架，区分必要性和充分性因果，提升政策模拟的可靠性。**

- **链接: [https://arxiv.org/pdf/2604.03920](https://arxiv.org/pdf/2604.03920)**

> **作者:** Agam Goyal; Yian Wang; Eshwar Chandrasekharan; Hari Sundaram
>
> **备注:** Accepted to PoliSim@CHI'26: 6 pages, 1 table
>
> **摘要:** LLM-based social simulations can generate believable community interactions, enabling ``policy wind tunnels'' where governance interventions are tested before deployment. But believability is not causality. Claims like ``intervention $A$ reduces escalation'' require causal semantics that current simulation work typically does not specify. We propose adopting the causal counterfactual framework, distinguishing \textit{necessary causation} (would the outcome have occurred without the intervention?) from \textit{sufficient causation} (does the intervention reliably produce the outcome?). This distinction maps onto different stakeholder needs: moderators diagnosing incidents require evidence about necessity, while platform designers choosing policies require evidence about sufficiency. We formalize this mapping, show how simulation design can support estimation under explicit assumptions, and argue that the resulting quantities should be interpreted as simulator-conditional causal estimates whose policy relevance depends on simulator fidelity. Establishing this framework now is essential: it helps define what adequate fidelity means and moves the field from simulations that look realistic toward simulations that can support policy changes.
>
---
#### [new 011] CAWN: Continuous Acoustic Wave Networks for Autoregressive Language Modeling
- **分类: cs.CL**

- **简介: 该论文提出CAWN模型，解决长序列语言建模中的计算复杂度和信号退化问题，采用连续相位机制实现线性时间复杂度与高效内存使用。**

- **链接: [https://arxiv.org/pdf/2604.04250](https://arxiv.org/pdf/2604.04250)**

> **作者:** Dejan Čugalj; Aleksandar Jevremovic
>
> **备注:** 13 pages, 3 figures
>
> **摘要:** Modern Large Language Models (LLMs) rely on Transformer self-attention, which scales quadratically with sequence length. Recent linear-time alternatives, like State Space Models (SSMs), often suffer from signal degradation over extended contexts. We introduce the Continuous Acoustic Wave Network (CAWN), a fully continuous sequence-mixing architecture. Instead of discrete matrix-based attention, CAWN projects hidden states into multi-headed complex-domain phasors, achieving sequence mixing through a causal, $O(L)$ Phase Accumulation mechanism. To prevent signal degradation over ultra-long contexts, we introduce a dual-gated Selective Phase Resonance mechanism incorporating Frequency-Dependent Retention, Hard-Threshold Gating via Straight-Through Estimation, and a Temporal Syntax Cache to capture short-term local dependencies. We also replace standard dense linear projections with Depth-wise Harmonic Convolutions for optimal spatial frequency mixing, augmented by Block Attention Residuals for depth-wise state routing. Scaled to a 150M-parameter model, CAWN utilizes custom Triton kernels for hardware-efficient, true-complex phase accumulation in float32. Trained via a continuous streaming loop on a 100-Billion-token corpus, the prototype is evaluated at a 5-Billion-token milestone. Empirical evaluations via a Targeted Semantic Retrieval protocol demonstrate robust vocabulary acquisition and extended explicitly learned contextual denoising. By leveraging $O(1)$ state-passing via chunked prefill, the model retrieves targeted information across 2,000,000 tokens while strictly plateauing at 8.72 GB of Peak VRAM, empirically overcoming the $O(L^2)$ context memory wall.
>
---
#### [new 012] A Semi-Automated Annotation Workflow for Paediatric Histopathology Reports Using Small Language Models
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于医学文本信息提取任务，旨在解决临床文本结构化难题。通过开发基于小语言模型的半自动标注流程，从儿科病理报告中提取结构化信息。**

- **链接: [https://arxiv.org/pdf/2604.04168](https://arxiv.org/pdf/2604.04168)**

> **作者:** Avish Vijayaraghavan; Jaskaran Singh Kawatra; Sebin Sabu; Jonny Sheldon; Will Poulett; Alex Eze; Daniel Key; John Booth; Shiren Patel; Jonny Pearson; Dan Schofield; Jonathan Hope; Pavithra Rajendran; Neil Sebire
>
> **备注:** 36 pages, includes supplementary information
>
> **摘要:** Electronic Patient Record (EPR) systems contain valuable clinical information, but much of it is trapped in unstructured text, limiting its use for research and decision-making. Large language models can extract such information but require substantial computational resources to run locally, and sending sensitive clinical data to cloud-based services, even when deidentified, raises significant patient privacy concerns. In this study, we develop a resource-efficient semi-automated annotation workflow using small language models (SLMs) to extract structured information from unstructured EPR data, focusing on paediatric histopathology reports. As a proof-of-concept, we apply the workflow to paediatric renal biopsy reports, a domain chosen for its constrained diagnostic scope and well-defined underlying biology. We develop the workflow iteratively with clinical oversight across three meetings, manually annotating 400 reports from a dataset of 2,111 at Great Ormond Street Hospital as a gold standard, while developing an automated information extraction approach using SLMs. We frame extraction as a Question-Answering task grounded by clinician-guided entity guidelines and few-shot examples, evaluating five instruction-tuned SLMs with a disagreement modelling framework to prioritise reports for clinical review. Gemma 2 2B achieves the highest accuracy at 84.3%, outperforming off-the-shelf models including spaCy (74.3%), BioBERT-SQuAD (62.3%), RoBERTa-SQuAD (59.7%), and GLiNER (60.2%). Entity guidelines improved performance by 7-19% over the zero-shot baseline, and few-shot examples by 6-38%, though their benefits do not compound when combined. These results demonstrate that SLMs can extract structured information from specialised clinical domains on CPU-only infrastructure with minimal clinician involvement. Our code is available at this https URL.
>
---
#### [new 013] RUQuant: Towards Refining Uniform Quantization for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决量化过程中因激活分布不均导致的精度下降问题。提出RUQuant方法，通过正交变换提升量化效果。**

- **链接: [https://arxiv.org/pdf/2604.04013](https://arxiv.org/pdf/2604.04013)**

> **作者:** Han Liu; Haotian Gao; Changya Li; Feng Zhang; Xiaotong Zhang; Wei Wang; Hong Yu
>
> **备注:** Accepted to KDD 2026. 12 pages, 9 figures
>
> **摘要:** The increasing size and complexity of large language models (LLMs) have raised significant challenges in deployment efficiency, particularly under resource constraints. Post-training quantization (PTQ) has emerged as a practical solution by compressing models without requiring retraining. While existing methods focus on uniform quantization schemes for both weights and activations, they often suffer from substantial accuracy degradation due to the non-uniform nature of activation distributions. In this work, we revisit the activation quantization problem from a theoretical perspective grounded in the Lloyd-Max optimality conditions. We identify the core issue as the non-uniform distribution of activations within the quantization interval, which causes the optimal quantization point under the Lloyd-Max criterion to shift away from the midpoint of the interval. To address this issue, we propose a two-stage orthogonal transformation method, RUQuant. In the first stage, activations are divided into blocks. Each block is mapped to uniformly sampled target vectors using composite orthogonal matrices, which are constructed from Householder reflections and Givens rotations. In the second stage, a global Householder reflection is fine-tuned to further minimize quantization error using Transformer output discrepancies. Empirical results show that our method achieves near-optimal quantization performance without requiring model fine-tuning: RUQuant achieves 99.8% of full-precision accuracy with W6A6 and 97% with W4A4 quantization for a 13B LLM, within approximately one minute. A fine-tuned variant yields even higher accuracy, demonstrating the effectiveness and scalability of our approach.
>
---
#### [new 014] Unlocking Prompt Infilling Capability for Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决扩散语言模型无法有效进行提示填充的问题。通过扩展全序列掩码训练方法，提升模型的提示填充能力，并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2604.03677](https://arxiv.org/pdf/2604.03677)**

> **作者:** Yoshinari Fujinuma; Keisuke Sakaguchi
>
> **摘要:** Masked diffusion language models (dLMs) generate text through bidirectional denoising, yet this capability remains locked for infilling prompts. This limitation is an artifact of the current supervised finetuning (SFT) convention of applying response-only masking. To unlock this capability, we extend full-sequence masking during SFT, where both prompts and responses are masked jointly. Once unlocked, the model infills masked portions of a prompt template conditioned on few-shot examples. We show that such model-infilled prompts match or surpass manually designed templates, transfer effectively across models, and are complementary to existing prompt optimization methods. Our results suggest that training practices, not architectural limitations, are the primary bottleneck preventing masked diffusion language models from infilling effective prompts
>
---
#### [new 015] MERIT: Multilingual Expert-Reward Informed Tuning for Chinese-Centric Low-Resource Machine Translation
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，针对中-低资源东南亚语言翻译问题，提出MERIT框架，结合数据筛选与奖励优化，提升翻译质量。**

- **链接: [https://arxiv.org/pdf/2604.04839](https://arxiv.org/pdf/2604.04839)**

> **作者:** Zhixiang Lu; Chong Zhang; Chenyu Xue; Angelos Stefanidis; Chong Li; Jionglong Su; Zhengyong Jiang
>
> **摘要:** Neural machine translation (NMT) from Chinese to low-resource Southeast Asian languages remains severely constrained by the extreme scarcity of clean parallel corpora and the pervasive noise in existing mined data. This chronic shortage not only impedes effective model training but also sustains a large performance gap with high-resource directions, leaving millions of speakers of languages such as Lao, Burmese, and Tagalog with persistently low-quality translation systems despite recent advances in large multilingual models. We introduce \textbf{M}ultilingual \textbf{E}xpert-\textbf{R}eward \textbf{I}nformed \textbf{T}uning (\textbf{MERIT}), a unified translation framework that transforms the traditional English-centric ALT benchmark into a Chinese-centric evaluation suite for five Southeast Asian low-resource languages (LRLs). Our framework combines language-specific token prefixing (LTP) with supervised fine-tuning (SFT) and a novel group relative policy optimization (GRPO) guided by the semantic alignment reward (SAR). These results confirm that, in LRL{\textrightarrow}Chinese translation, targeted data curation and reward-guided optimization dramatically outperform mere model scaling.
>
---
#### [new 016] Predict, Don't React: Value-Based Safety Forecasting for LLM Streaming
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出StreamGuard，用于LLM流式输出的安全监测，将问题转化为危害预测而非边界检测，提升安全干预效果。**

- **链接: [https://arxiv.org/pdf/2604.03962](https://arxiv.org/pdf/2604.03962)**

> **作者:** Pride Kavumba; Koki Wataoka; Huy H. Nguyen; Jiaxuan Li; Masaya Ohagi
>
> **摘要:** In many practical LLM deployments, a single guardrail is used for both prompt and response moderation. Prompt moderation operates on fully observed text, whereas streaming response moderation requires safety decisions to be made over partial generations. Existing text-based streaming guardrails commonly frame this output-side problem as boundary detection, training models to identify the earliest prefix at which a response has already become unsafe. In this work, we introduce StreamGuard, a unified model-agnostic streaming guardrail that instead formulates moderation as a forecasting problem: given a partial prefix, the model predicts the expected harmfulness of likely future continuations. We supervise this prediction using Monte Carlo rollouts, which enables early intervention without requiring exact token-level boundary annotations. Across standard safety benchmarks, StreamGuard performs strongly both for input moderation and for streaming output moderation. At the 8B scale, StreamGuard improves aggregated input-moderation F1 from 86.7 to 88.2 and aggregated streaming output-moderation F1 from 80.4 to 81.9 relative to Qwen3Guard-Stream-8B-strict. On the QWENGUARDTEST response_loc streaming benchmark, StreamGuard reaches 97.5 F1, 95.1 recall, and 92.6% on-time intervention, compared to 95.9 F1, 92.1 recall, and 89.9% for Qwen3Guard-Stream-8B-stric, while reducing the miss rate from 7.9% to 4.9%. We further show that forecasting-based supervision transfers effectively across tokenizers and model families: with transferred targets, Gemma3-StreamGuard-1B reaches 81.3 response-moderation F1, 98.2 streaming F1, and a 3.5% miss rate. These results show that strong end-to-end streaming moderation can be obtained without exact boundary labels, and that forecasting future risk is an effective supervision strategy for low-latency safety intervention.
>
---
#### [new 017] Synthetic Sandbox for Training Machine Learning Engineering Agents
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于机器学习工程代理训练任务，解决ML验证效率低的问题。通过生成小规模合成环境，提升RL训练效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.04872](https://arxiv.org/pdf/2604.04872)**

> **作者:** Yuhang Zhou; Lizhu Zhang; Yifan Wu; Jiayi Liu; Xiangjun Fan; Zhuokai Zhao; Hong Yan
>
> **备注:** 28 pages, 9 tables, 8 figures
>
> **摘要:** As large language model agents advance beyond software engineering (SWE) tasks toward machine learning engineering (MLE), verifying agent behavior becomes orders of magnitude more expensive: while SWE tasks can be verified via fast-executing unit tests, MLE verification requires running full ML pipelines -- data preprocessing, model training, and metric evaluation -- on large datasets at each rollout step, rendering trajectory-wise on-policy reinforcement learning (RL) prohibitively slow. Existing approaches retreat to supervised fine-tuning (SFT) or offline proxy rewards, sacrificing the exploration and generalization benefits of on-policy RL. We observe that sandbox data size is the primary source of this bottleneck. Based on this insight, we introduce SandMLE, a multi-agent framework that generates diverse, verifiable synthetic MLE environments from a small number of seed tasks, preserving the structural and technical complexity of real-world problems while constraining datasets to micro-scale (each task is paired with only 50-200 training samples). Through extensive experiments, we show that SandMLE reduces execution time by over 13 times, enabling large-scale, on-policy trajectory-wise RL for the first time in the MLE domain. On MLE-bench-lite, SandMLE yields significant gains over SFT baselines across Qwen3-8B, 14B, and 30B-A3B, with relative medal rate improvements ranging from 20.3% to 66.9%. Furthermore, the trained policy generalizes across unseen agentic scaffolds, achieving up to 32.4% better HumanRank score on MLE-Dojo.
>
---
#### [new 018] Extracting and Steering Emotion Representations in Small Language Models: A Methodological Comparison
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感表示研究任务，旨在探讨小型语言模型是否具备情绪表征，并比较不同提取方法。工作包括评估多种模型和架构，验证情绪表示位置及可操控性。**

- **链接: [https://arxiv.org/pdf/2604.04064](https://arxiv.org/pdf/2604.04064)**

> **作者:** Jihoon Jeong
>
> **备注:** 14 pages, 4 figures, 7 tables. Paper #6 in the Model Medicine series
>
> **摘要:** Small language models (SLMs) in the 100M-10B parameter range increasingly power production systems, yet whether they possess the internal emotion representations recently discovered in frontier models remains unknown. We present the first comparative analysis of emotion vector extraction methods for SLMs, evaluating 9 models across 5 architectural families (GPT-2, Gemma, Qwen, Llama, Mistral) using 20 emotions and two extraction methods (generation-based and comprehension-based). Generation-based extraction produces statistically superior emotion separation (Mann-Whitney p = 0.007; Cohen's d = -107.5), with the advantage modulated by instruction tuning and architecture. Emotion representations localize at middle transformer layers (~50% depth), following a U-shaped curve that is architecture-invariant from 124M to 3B parameters. We validate these findings against representational anisotropy baselines across 4 models and confirm causal behavioral effects through steering experiments, independently verified by an external emotion classifier (92% success rate, 37/40 scenarios). Steering reveals three regimes -- surgical (coherent text transformation), repetitive collapse, and explosive (text degradation) -- quantified by perplexity ratios and separated by model architecture rather than scale. We document cross-lingual emotion entanglement in Qwen, where steering activates semantically aligned Chinese tokens that RLHF does not suppress, raising safety concerns for multilingual deployment. This work provides methodological guidelines for emotion research on open-weight models and contributes to the Model Medicine series by bridging external behavioral profiling with internal representational analysis.
>
---
#### [new 019] Same Geometry, Opposite Noise: Transformer Magnitude Representations Lack Scalar Variability
- **分类: cs.CL; q-bio.QM**

- **简介: 该论文属于自然语言处理领域，研究Transformer模型的表示噪声特性。任务是检验模型是否具备生物系统中的标量可变性，发现其噪声随幅度减小，与生物系统相反。**

- **链接: [https://arxiv.org/pdf/2604.04469](https://arxiv.org/pdf/2604.04469)**

> **作者:** Jon-Paul Cacioli
>
> **备注:** 7 pages, 5 figures, 1 table. Pre-registered on OSF (this http URL). Companion to arXiv:2603.20642
>
> **摘要:** Scalar variability -- the finding that representational noise scales proportionally with magnitude, producing a constant coefficient of variation -- is a hallmark of biological magnitude systems. We tested whether transformer language models exhibit this property by analysing the dispersion of hidden-state representations across carrier sentences for 26 numerical magnitudes in three 7-8B parameter models (Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3, Llama-3-8B-Base; data from Cacioli, 2026). We found the opposite: representational variability decreased with magnitude along the magnitude axis (scaling exponent alpha approx -0.19; 0/16 primary layers with alpha > 0, all three models). The negative sign was consistent in full-dimensional space (alpha approx -0.04) and after sentence-identity correction (alpha approx -0.007). The anti-scalar pattern was 3-5x stronger along the magnitude axis than orthogonal dimensions, and corpus frequency strongly predicted per-magnitude variability (rho = .84). These results demonstrate that distributional learning alone is insufficient to produce scalar variability: transformers reproduce log-compressive magnitude geometry but not the constant-CV noise signature observed in biological systems.
>
---
#### [new 020] CAGMamba: Context-Aware Gated Cross-Modal Mamba Network for Multimodal Sentiment Analysis
- **分类: cs.CL**

- **简介: 该论文属于多模态情感分析任务，旨在解决跨模态交互和上下文依赖建模问题。提出CAGMamba框架，通过时间序列结构和门控机制实现高效的情感演化建模与多模态融合。**

- **链接: [https://arxiv.org/pdf/2604.03650](https://arxiv.org/pdf/2604.03650)**

> **作者:** Minghai Jiao; Jing Xiao; Peng Xiao; Ende Zhang; Shuang Kan; Wenyan Jiang; Jinyao Li; Yixian Liu; Haidong Xin
>
> **摘要:** Multimodal Sentiment Analysis (MSA) requires effective modeling of cross-modal interactions and contextual dependencies while remaining computationally efficient. Existing fusion approaches predominantly rely on Transformer-based cross-modal attention, which incurs quadratic complexity with respect to sequence length and limits scalability. Moreover, contextual information from preceding utterances is often incorporated through concatenation or independent fusion, without explicit temporal modeling that captures sentiment evolution across dialogue turns. To address these limitations, we propose CAGMamba, a context-aware gated cross-modal Mamba framework for dialogue-based sentiment analysis. Specifically, we organize the contextual and the current-utterance features into a temporally ordered binary sequence, which provides Mamba with explicit temporal structure for modeling sentiment evolution. To further enable controllable cross-modal integration, we propose a Gated Cross-Modal Mamba Network (GCMN) that integrates cross-modal and unimodal paths via learnable gating to balance information fusion and modality preservation, and is trained with a three-branch multi-task objective over text, audio, and fused predictions. Experiments on three benchmark datasets demonstrate that CAGMamba achieves state-of-the-art or competitive results across multiple evaluation metrics. All codes are available at this https URL.
>
---
#### [new 021] Conversational Control with Ontologies for Large Language Models: A Lightweight Framework for Constrained Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统任务，旨在解决LLM生成内容不可控问题。通过本体定义约束条件，实现对输出的模块化、可解释控制。**

- **链接: [https://arxiv.org/pdf/2604.04450](https://arxiv.org/pdf/2604.04450)**

> **作者:** Barbara Gendron; Gaël Guibon; Mathieu d'Aquin
>
> **备注:** Accepted at KG & LLM: Knowledge Graphs and Large Language Models LREC 2026 Workshop
>
> **摘要:** Conversational agents based on Large Language Models (LLMs) have recently emerged as powerful tools for human-computer interaction. Nevertheless, their black-box nature implies challenges in predictability and a lack of personalization, both of which can be addressed by controlled generation. This work proposes an end-to-end method to obtain modular and explainable control over LLM outputs through ontological definitions of aspects related to the conversation. Key aspects are modeled and used as constraints; we then further fine-tune the LLM to generate content accordingly. To validate our approach, we explore two tasks that tackle two key conversational aspects: the English proficiency level and the polarity profile of the content. Using a hybrid fine-tuning procedure on seven state-of-the-art, open-weight conversational LLMs, we show that our method consistently outperforms pre-trained baselines, even on smaller models. Beyond quantitative gains, the framework remains model-agnostic, lightweight, and interpretable, enabling reusable control strategies that can be extended to new domains and interaction goals. This approach enhances alignment with strategy instructions and demonstrates the effectiveness of ontology-driven control in conversational systems.
>
---
#### [new 022] DARE: Diffusion Large Language Models Alignment and Reinforcement Executor
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决dLLMs后训练流程碎片化问题。提出DARE框架，统一多种训练方法，提升研究效率与可比性。**

- **链接: [https://arxiv.org/pdf/2604.04215](https://arxiv.org/pdf/2604.04215)**

> **作者:** Jingyi Yang; Yuxian Jiang; Xuhao Hu; Shuang Cheng; Biqing Qi; Jing Shao
>
> **备注:** 14 pages,3 figures,5 tables
>
> **摘要:** Diffusion large language models (dLLMs) are emerging as a compelling alternative to dominant autoregressive models, replacing strictly sequential token generation with iterative denoising and parallel generation dynamics. However, their open-source ecosystem remains fragmented across model families and, in particular, across post-training pipelines, where reinforcement learning objectives, rollout implementations and evaluation scripts are often released as paper-specific codebases. This fragmentation slows research iteration, raises the engineering burden of reproduction, and makes fair comparison across algorithms difficult. We present \textbf{DARE} (\textbf{d}LLMs \textbf{A}lignment and \textbf{R}einforcement \textbf{E}xecutor), an open framework for post-training and evaluating dLLMs. Built on top of verl~\cite{sheng2024hybridflow} and OpenCompass~\cite{2023opencompass}, DARE unifies supervised fine-tuning, parameter-efficient fine-tuning, preference optimization, and dLLM-specific reinforcement learning under a shared execution stack for both masked and block diffusion language models. Across representative model families including LLaDA, Dream, SDAR, and LLaDA2.x, DARE provides broad algorithmic coverage, reproducible benchmark evaluation, and practical acceleration. Extensive empirical results position that DARE serves as a reusable research substrate for developing, comparing, and deploying post-training methods for current and emerging dLLMs.
>
---
#### [new 023] Your Agent is More Brittle Than You Think: Uncovering Indirect Injection Vulnerabilities in Agentic LLMs
- **分类: cs.CL**

- **简介: 该论文属于安全任务，旨在解决多智能体系统中的间接提示注入漏洞问题。通过评估防御策略，发现现有方法脆弱，并提出基于表示工程的检测方案。**

- **链接: [https://arxiv.org/pdf/2604.03870](https://arxiv.org/pdf/2604.03870)**

> **作者:** Wenhui Zhu; Xuanzhao Dong; Xiwen Chen; Rui Cai; Peijie Qiu; Zhipeng Wang; Oana Frunza; Shao Tang; Jindong Gu; Yalin Wang
>
> **摘要:** The rapid deployment of open-source frameworks has significantly advanced the development of modern multi-agent systems. However, expanded action spaces, including uncontrolled privilege exposure and hidden inter-system interactions, pose severe security challenges. Specifically, Indirect Prompt Injections (IPI), which conceal malicious instructions within third-party content, can trigger unauthorized actions such as data exfiltration during normal operations. While current security evaluations predominantly rely on isolated single-turn benchmarks, the systemic vulnerabilities of these agents within complex dynamic environments remain critically underexplored. To bridge this gap, we systematically evaluate six defense strategies against four sophisticated IPI attack vectors across nine LLM backbones. Crucially, we conduct our evaluation entirely within dynamic multi-step tool-calling environments to capture the true attack surface of modern autonomous agents. Moving beyond binary success rates, our multidimensional analysis reveals a pronounced fragility. Advanced injections successfully bypass nearly all baseline defenses, and some surface-level mitigations even produce counterproductive side effects. Furthermore, while agents execute malicious instructions almost instantaneously, their internal states exhibit abnormally high decision entropy. Motivated by this latent hesitation, we investigate Representation Engineering (RepE) as a robust detection strategy. By extracting hidden states at the tool-input position, we revealed that the RepE-based circuit breaker successfully identifies and intercepts unauthorized actions before the agent commits to them, achieving high detection accuracy across diverse LLM backbones. This study exposes the limitations of current IPI defenses and provides a highly practical paradigm for building resilient multi-agent architectures.
>
---
#### [new 024] How Well Do Agentic Skills Work in the Wild: Benchmarking LLM Skill Usage in Realistic Settings
- **分类: cs.CL**

- **简介: 该论文属于LLM代理技能评估任务，旨在解决真实场景下技能有效性问题。通过构建真实环境下的基准测试，分析技能性能下降原因，并提出改进策略。**

- **链接: [https://arxiv.org/pdf/2604.04323](https://arxiv.org/pdf/2604.04323)**

> **作者:** Yujian Liu; Jiabao Ji; Li An; Tommi Jaakkola; Yang Zhang; Shiyu Chang
>
> **摘要:** Agent skills, which are reusable, domain-specific knowledge artifacts, have become a popular mechanism for extending LLM-based agents, yet formally benchmarking skill usage performance remains scarce. Existing skill benchmarking efforts focus on overly idealized conditions, where LLMs are directly provided with hand-crafted, narrowly-tailored task-specific skills for each task, whereas in many realistic settings, the LLM agent may have to search for and select relevant skills on its own, and even the closest matching skills may not be well-tailored for the task. In this paper, we conduct the first comprehensive study of skill utility under progressively challenging realistic settings, where agents must retrieve skills from a large collection of 34k real-world skills and may not have access to any hand-curated skills. Our findings reveal that the benefits of skills are fragile: performance gains degrade consistently as settings become more realistic, with pass rates approaching no-skill baselines in the most challenging scenarios. To narrow this gap, we study skill refinement strategies, including query-specific and query-agnostic approaches, and we show that query-specific refinement substantially recovers lost performance when the initial skills are of reasonable relevance and quality. We further demonstrate the generality of retrieval and refinement on Terminal-Bench 2.0, where they improve the pass rate of Claude Opus 4.6 from 57.7% to 65.5%. Our results, consistent across multiple models, highlight both the promise and the current limitations of skills for LLM-based agents. Our code is available at this https URL.
>
---
#### [new 025] Knowledge Packs: Zero-Token Knowledge Delivery via KV Cache Injection
- **分类: cs.CL**

- **简介: 该论文提出Knowledge Packs，通过预计算的KV缓存实现零token知识传递，解决RAG效率低的问题，无需训练即可提升效果并支持行为控制。**

- **链接: [https://arxiv.org/pdf/2604.03270](https://arxiv.org/pdf/2604.03270)**

> **作者:** Andrey Pustovit
>
> **备注:** 12 pages, 3 figures, 8 tables. Code: this https URL
>
> **摘要:** RAG wastes tokens. We propose Knowledge Packs: pre-computed KV caches that deliver the same knowledge at zero token cost. For causal transformers, the KV cache from a forward pass on text F is identical to what a joint pass on F+q would produce - this follows directly from the causal mask. The equivalence is exact but fragile: wrong chat template formatting causes 6-7pp degradation, which we believe explains prior claims of KV outperforming RAG. With correct formatting: zero divergences across 700 questions on Qwen3-8B and Llama-3.1-8B, up to 95% token savings. The KV interface also enables behavioral steering that RAG cannot do. Because RoPE rotates keys but leaves values untouched, contrastive deltas on cached values can nudge model behavior while key arithmetic destroys coherence. The effect sits in mid-layer values (33-66%), independent directions are nearly orthogonal (cos~0) and compose, and both channels - knowledge and steering - run simultaneously at alpha<=0.7 without interference. No training, no weight modification.
>
---
#### [new 026] Researchers waste 80% of LLM annotation costs by classifying one text at a time
- **分类: cs.CL**

- **简介: 该论文研究文本分类任务，解决LLM标注成本高的问题。通过批量处理和多变量叠加，减少API调用次数，降低80%成本，同时保持分类准确性。**

- **链接: [https://arxiv.org/pdf/2604.03684](https://arxiv.org/pdf/2604.03684)**

> **作者:** Christian Pipal; Eva-Maria Vogel; Morgan Wack; Frank Esser
>
> **摘要:** Large language models (LLMs) are increasingly being used for text classification across the social sciences, yet researchers overwhelmingly classify one text per variable per prompt. Coding 100,000 texts on four variables requires 400,000 API calls. Batching 25 items and stacking all variables into a single prompt reduces this to 4,000 calls, cutting token costs by over 80%. Whether this degrades coding quality is unknown. We tested eight production LLMs from four providers on 3,962 expert-coded tweets across four tasks, varying batch size from 1 to 1,000 items and stacking up to 25 coding dimensions per prompt. Six of eight models maintained accuracy within 2 pp of the single-item baseline through batch sizes of 100. Variable stacking with up to 10 dimensions produced results comparable to single-variable coding, with degradation driven by task complexity rather than prompt length. Within this safe operating range, the measurement error from batching and stacking is smaller than typical inter-coder disagreement in the ground-truth data.
>
---
#### [new 027] Responses Fall Short of Understanding: Revealing the Gap between Internal Representations and Responses in Visual Document Understanding
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文研究视觉文档理解任务，探讨模型内部表示与生成回答之间的差距，发现中间层更线性编码任务信息，并通过微调中间层提升性能。**

- **链接: [https://arxiv.org/pdf/2604.04411](https://arxiv.org/pdf/2604.04411)**

> **作者:** Haruka Kawasaki; Ryota Tanaka; Kyosuke Nishida
>
> **备注:** Accepted to CVPR2026 workshop (MULA)
>
> **摘要:** Visual document understanding (VDU) is a challenging task for large vision language models (LVLMs), requiring the integration of visual perception, text recognition, and reasoning over structured layouts. Although recent LVLMs have shown progress on VDU benchmarks, their performance is typically evaluated based on generated responses, which may not necessarily reflect whether the model has actually captured the required information internally. In this paper, we investigate how information required to solve VDU tasks is represented across different layers of LLMs within LVLMs using linear probing. Our study reveals that (1) there is a clear gap between internal representations and generated responses, and (2) information required to solve the task is often encoded more linearly from intermediate layers than from the final layer. Motivated by these findings, we explore fine-tuning strategies that target intermediate layers. Experiments show that fine-tuning intermediate layers improves both linear probing accuracy and response accuracy while narrowing the gap.
>
---
#### [new 028] Why Attend to Everything? Focus is the Key
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Focus方法，解决高效注意力机制问题。通过学习关键token对，提升模型效率与性能，适用于自然语言处理任务。**

- **链接: [https://arxiv.org/pdf/2604.03260](https://arxiv.org/pdf/2604.03260)**

> **作者:** Hengshuai Yao; Xing Chen; Ahmed Murtadha; Jin Li; Shuai Shao; Yasin Abbasi Yadkori; Guan Wang; Mingli Yuan; William Chen; Sen Song
>
> **摘要:** We introduce Focus, a method that learns which token pairs matter rather than approximating all of them. Learnable centroids assign tokens to groups; distant attention is restricted to same-group pairs while local attention operates at full resolution. Because all model weights stay frozen, Focus is purely additive: centroid-only training (as few as 148K parameters) improves domain perplexity with zero degradation on downstream benchmarks--from 124M to 70B parameters, across five attention architectures. No existing efficient attention method achieves this in the retrofit setting. At 124M, Focus surpasses full attention (30.3 vs 31.4 PPL); trained from scratch at 7B scale (2B tokens), Focus again beats full attention (13.82 vs 13.89 PPL). At inference, restricting each token to its top-k highest-scoring groups discretizes the soft routing into a hard sparsity pattern, yielding 2x speedup while beating the pretrained baseline (41.3 vs 42.8 PPL); decomposing this pattern into two standard FlashAttention calls reaches 8.6x wall-clock speedup at 1M tokens with no custom kernels. Unlike LoRA, centroid routing preserves alignment: instruction-tuned models retain TruthfulQA scores after adaptation, while LoRA degrades at every learning rate and rank. Sinkhorn normalization enforces balanced groups as a hard constraint, and the resulting groups discover interpretable linguistic categories without supervision.
>
---
#### [new 029] Noise Steering for Controlled Text Generation: Improving Diversity and Reading-Level Fidelity in Arabic Educational Story Generation
- **分类: cs.CL**

- **简介: 该论文属于阿拉伯语教育故事生成任务，旨在提升多样性与阅读等级准确性。通过噪声引导方法，在推理阶段注入高斯扰动，以改善叙事多样性并保持内容约束。**

- **链接: [https://arxiv.org/pdf/2604.03380](https://arxiv.org/pdf/2604.03380)**

> **作者:** Haziq Mohammad Khalid; Salsabeel Shapsough; Imran Zualkernan
>
> **备注:** Under Review
>
> **摘要:** Generating diverse, pedagogically valid stories for Arabic early-grade reading assessments requires balancing tight constraints on vocabulary, reading level, and narrative structure against the need to avoid repetitive plots that undermine assessment validity. We investigate noise steering, injecting calibrated Gaussian perturbations into the internal representations of transformer models at inference time, as a training-free diversity method evaluated across five small Arabic-centric language models (7-9B parameters). We compare four injection strategies against high-temperature sampling baselines, measuring diversity, quality, constraint adherence, and reading grade level. Residual stream noise consistently improves narrative diversity with minimal quality or constraint cost and preserves early-grade reading level across all models. Attention entropy noise injection (AENI) stabilizes the otherwise unreliable attention-logit noise while recovering quality. High-temperature sampling inflates reading grade level and causes catastrophic collapse on several models. We find internal representation-level perturbation to be a more suitable diversity strategy than output-level stochasticity for constrained educational content generation.
>
---
#### [new 030] What Makes Good Multilingual Reasoning? Disentangling Reasoning Traces with Measurable Features
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言推理任务，旨在解决不同语言间推理性能差异问题。通过分析可测量的推理特征，评估其对多语言推理的影响，并提出适应性优化方法。**

- **链接: [https://arxiv.org/pdf/2604.04720](https://arxiv.org/pdf/2604.04720)**

> **作者:** Dayeon Ki; Kevin Duh; Marine Carpuat
>
> **备注:** 31 pages, 7 figures
>
> **摘要:** Large Reasoning Models (LRMs) still exhibit large performance gaps between English and other languages, yet much current work assumes these gaps can be closed simply by making reasoning in every language resemble English reasoning. This work challenges this assumption by asking instead: what actually characterizes effective reasoning in multilingual settings, and to what extent do English-derived reasoning features genuinely help in other languages? We first define a suite of measurable reasoning features spanning multilingual alignment, reasoning step, and reasoning flow aspects of reasoning traces, and use logistic regression to quantify how each feature associates with final answer accuracy. We further train sparse autoencoders over multilingual traces to automatically discover latent reasoning concepts that instantiate or extend these features. Finally, we use the features as test-time selection policies to examine whether they can steer models toward stronger multilingual reasoning. Across two mathematical reasoning benchmarks, four LRMs, and 10 languages, we find that most features are positively associated with accuracy, but the strength of association varies considerably across languages and can even reverse in some. Our findings challenge English-centric reward designs and point toward adaptive objectives that accommodate language-specific reasoning patterns, with concrete implications for multilingual benchmark and reward design.
>
---
#### [new 031] Unveiling Language Routing Isolation in Multilingual MoE Models for Interpretable Subnetwork Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言MoE模型中的语言路由隔离现象，旨在提升低资源语言性能。通过分析专家路由模式，提出RISE框架，选择性优化语言特定子网络。**

- **链接: [https://arxiv.org/pdf/2604.03592](https://arxiv.org/pdf/2604.03592)**

> **作者:** Kening Zheng; Wei-Chieh Huang; Jiahao Huo; Zhonghao Li; Henry Peng Zou; Yibo Yan; Xin Zou; Jungang Li; Junzhuo Li; Hanrong Zhang; Xuming Hu; Philip S. Yu
>
> **摘要:** Mixture-of-Experts (MoE) models exhibit striking performance disparities across languages, yet the internal mechanisms driving these gaps remain poorly understood. In this work, we conduct a systematic analysis of expert routing patterns in MoE models, revealing a phenomenon we term Language Routing Isolation, in which high- and low-resource languages tend to activate largely disjoint expert sets. Through layer-stratified analysis, we further show that routing patterns exhibit a layer-wise convergence-divergence pattern across model depth. Building on these findings, we propose RISE (Routing Isolation-guided Subnetwork Enhancement), a framework that exploits routing isolation to identify and adapt language-specific expert subnetworks. RISE applies a tripartite selection strategy, using specificity scores to identify language-specific experts in shallow and deep layers and overlap scores to select universal experts in middle layers. By training only the selected subnetwork while freezing all other parameters, RISE substantially improves low-resource language performance while preserving capabilities in other languages. Experiments on 10 languages demonstrate that RISE achieves target-language F1 gains of up to 10.85% with minimal cross-lingual degradation.
>
---
#### [new 032] Structured Causal Video Reasoning via Multi-Objective Alignment
- **分类: cs.CL**

- **简介: 该论文属于视频理解任务，旨在解决现有模型依赖非结构化推理导致的因果推断脆弱问题。通过构建结构化事件事实并采用多目标强化学习优化，提升视频推理的可靠性与精度。**

- **链接: [https://arxiv.org/pdf/2604.04415](https://arxiv.org/pdf/2604.04415)**

> **作者:** Zinuo Li; Yongxin Guo; Jun Liu; Jiawei Zhan; Xi Jiang; Chengjie Wang; Mohammed Bennamoun; Farid Boussaid; Feng Zheng; Qiuhong Ke
>
> **摘要:** Human understanding of video dynamics is typically grounded in a structured mental representation of entities, actions, and temporal relations, rather than relying solely on immediate deductive reasoning. In contrast, existing Video-LLMs largely depend on unstructured video reasoning, where critical visual evidence is embedded in verbose textual descriptions and temporal causality is often weakly modeled. This leads to inefficient processes and fragile causal inference. To bridge this cognitive gap, we propose constructing a compact representation of salient events and their causal relationships, which we name Structured Event Facts, prior to the reasoning stage. This structured prior serves as an explicit constraint to promote concise and causally grounded reasoning, while also making intermediate evidence easier to verify. To effectively train models on such structured facts, we introduce CausalFact-60K and a four-stage training pipeline comprising facts alignment, format warm-start, thinking warm-start, and reinforcement learning-based post-training. During RL stage, we find that this framework introduces competing objectives, as structural completeness and causal fidelity must be balanced against reasoning length, making it difficult to optimize. We address this challenge by formulating the optimization as a Multi-Objective Reinforcement Learning (MORL) problem and explicitly optimizing toward the Pareto-Frontier to balance these trade-offs. As a result, we introduce Factum-4B, which yields more reliable reasoning and delivers stronger performance on challenging video understanding tasks requiring fine-grained temporal inference.
>
---
#### [new 033] Position: Logical Soundness is not a Reliable Criterion for Neurosymbolic Fact-Checking with LLMs
- **分类: cs.CL**

- **简介: 该论文属于事实核查任务，探讨逻辑严谨性在神经符号系统中的局限性，指出逻辑正确结论可能误导人类判断，主张结合LLM的人类推理倾向提升核查效果。**

- **链接: [https://arxiv.org/pdf/2604.04177](https://arxiv.org/pdf/2604.04177)**

> **作者:** Jason Chan; Robert Gaizauskas; Zhixue Zhao
>
> **备注:** Preprint
>
> **摘要:** As large language models (LLMs) are increasing integrated into fact-checking pipelines, formal logic is often proposed as a rigorous means by which to mitigate bias, errors and hallucinations in these models' outputs. For example, some neurosymbolic systems verify claims by using LLMs to translate natural language into logical formulae and then checking whether the proposed claims are logically sound, i.e. whether they can be validly derived from premises that are verified to be true. We argue that such approaches structurally fail to detect misleading claims due to systematic divergences between conclusions that are logically sound and inferences that humans typically make and accept. Drawing on studies in cognitive science and pragmatics, we present a typology of cases in which logically sound conclusions systematically elicit human inferences that are unsupported by the underlying premises. Consequently, we advocate for a complementary approach: leveraging the human-like reasoning tendencies of LLMs as a feature rather than a bug, and using these models to validate the outputs of formal components in neurosymbolic systems against potentially misleading conclusions.
>
---
#### [new 034] The Format Tax
- **分类: cs.CL**

- **简介: 论文探讨了大语言模型在生成结构化格式（如JSON）时性能下降的问题，属于自然语言处理任务。研究提出通过分离推理与格式化过程来解决这一问题。**

- **链接: [https://arxiv.org/pdf/2604.03616](https://arxiv.org/pdf/2604.03616)**

> **作者:** Ivan Yee Lee; Loris D'Antoni; Taylor Berg-Kirkpatrick
>
> **摘要:** Asking a large language model to respond in JSON should be a formatting choice, not a capability tax. Yet we find that structured output requirements -- JSON, XML, LaTeX, Markdown -- substantially degrade reasoning and writing performance across open-weight models. The research response has focused on constrained decoding, but sampling bias accounts for only a fraction of the degradation. The dominant cost enters at the prompt: format-requesting instructions alone cause most of the accuracy loss, before any decoder constraint is applied. This diagnosis points to a simple principle: decouple reasoning from formatting. Whether by generating freeform first and reformatting in a second pass, or by enabling extended thinking within a single generation, separating the two concerns substantially recovers lost accuracy. Across six open-weight models, four API models, four formats, and tasks spanning math, science, logic, and writing, decoupling recovers most lost accuracy. Notably, most recent closed-weight models show little to no format tax, suggesting the problem is not inherent to structured generation but a gap that current open-weight models have yet to close. Code is available at this https URL.
>
---
#### [new 035] Vocabulary Dropout for Curriculum Diversity in LLM Co-Evolution
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型协同进化任务，旨在解决自监督课程学习中问题多样性不足的问题。通过引入词汇丢弃机制，提升问题生成的多样性，从而促进模型性能提升。**

- **链接: [https://arxiv.org/pdf/2604.03472](https://arxiv.org/pdf/2604.03472)**

> **作者:** Jacob Dineen; Aswin RRV; Zhikun Xu; Ben Zhou
>
> **摘要:** Co-evolutionary self-play, where one language model generates problems and another solves them, promises autonomous curriculum learning without human supervision. In practice, the proposer quickly converges to a narrow distribution of problems that satisfy the reward function. This diversity collapse renders the curriculum uninformative for the solver, stalling the co-evolutionary loop. We introduce vocabulary dropout, a random mask applied to the proposer's output logits during both policy training and curriculum generation, as a lightweight mechanism to sustain diversity. The mask is hard and non-stationary, preventing the proposer from locking into fixed token sequences. Training Qwen3-4B and Qwen3-8B on mathematical reasoning via R-Zero, we find that vocabulary dropout sustains proposer diversity across lexical, semantic, and functional metrics throughout training, and yields solver improvements averaging +4.4 points at 8B, with the largest gains on competition-level benchmarks. Our findings suggest that explicit action-space constraints, analogous to the structural role that game rules play in classical self-play, can help sustain productive co-evolution in language. Vocabulary dropout is one simple instantiation of this principle.
>
---
#### [new 036] DeonticBench: A Benchmark for Reasoning over Rules
- **分类: cs.CL**

- **简介: 该论文提出DEONTICBENCH，用于评估大模型在法律规则下的推理能力，解决复杂规则推理难题。**

- **链接: [https://arxiv.org/pdf/2604.04443](https://arxiv.org/pdf/2604.04443)**

> **作者:** Guangyao Dou; Luis Brena; Akhil Deo; William Jurayj; Jingyu Zhang; Nils Holzenberger; Benjamin Van Durme
>
> **摘要:** Reasoning with complex, context-specific rules remains challenging for large language models (LLMs). In legal and policy settings, this manifests as deontic reasoning: reasoning about obligations, permissions, and prohibitions under explicit rules. While many recent benchmarks emphasize short-context mathematical reasoning, fewer focus on long-context, high-stakes deontic reasoning. To address this gap, we introduce DEONTICBENCH, a benchmark of 6,232 tasks across U.S. federal taxes, airline baggage policies, U.S. immigration administration, and U.S. state housing law. These tasks can be approached in multiple ways, including direct reasoning in language or with the aid of symbolic computation. Besides free-form chain-of-thought reasoning, DEONTICBENCH enables an optional solver-based workflow in which models translate statutes and case facts into executable Prolog, leading to formal problem interpretations and an explicit program trace. We release reference Prolog programs for all instances. Across frontier LLMs and coding models, best hard-subset performance reaches only 44.4% on SARA Numeric and 46.6 macro-F1 on Housing. We further study training with supervised fine-tuning and reinforcement learning for symbolic program generation. Although training improves Prolog generation quality, current RL methods still fail to solve these tasks reliably. Overall, DEONTICBENCH provides a benchmark for studying context-grounded rule reasoning in real-world domains under both symbolic and non-symbolic settings.
>
---
#### [new 037] Embedding Enhancement via Fine-Tuned Language Models for Learner-Item Cognitive Modeling
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **简介: 该论文属于学习者-项目认知建模任务，旨在解决语言模型与认知诊断不匹配的问题。通过引入EduEmbed框架，提升嵌入表示的语义融合效果。**

- **链接: [https://arxiv.org/pdf/2604.04088](https://arxiv.org/pdf/2604.04088)**

> **作者:** Yuanhao Liu; Zihan Zhou; Kaiying Wu; Shuo Liu; Yiyang Huang; Jiajun Guo; Aimin Zhou; Hong Qian
>
> **备注:** Accepted by The ACM Web Conference 2026 (WWW '26)
>
> **摘要:** Learner-item cognitive modeling plays a central role in the web-based online intelligent education system by enabling cognitive diagnosis (CD) across diverse online educational scenarios. Although ID embedding remains the mainstream approach in cognitive modeling due to its effectiveness and flexibility, recent advances in language models (LMs) have introduced new possibilities for incorporating rich semantic representations to enhance CD performance. This highlights the need for a comprehensive analysis of how LMs enhance embeddings through semantic integration across mainstream CD tasks. This paper identifies two key challenges in fully leveraging LMs in existing work: Misalignment between the training objectives of LMs and CD models creates a distribution gap in feature spaces; A unified framework is essential for integrating textual embeddings across varied CD tasks while preserving the strengths of existing cognitive modeling paradigms to ensure the robustness of embedding enhancement. To address these challenges, this paper introduces EduEmbed, a unified embedding enhancement framework that leverages fine-tuned LMs to enrich learner-item cognitive modeling across diverse CD tasks. EduEmbed operates in two stages. In the first stage, we fine-tune LMs based on role-specific representations and an interaction diagnoser to bridge the semantic gap of CD models. In the second stage, we employ a textual adapter to extract task-relevant semantics and integrate them with existing modeling paradigms to improve generalization. We evaluate the proposed framework on four CD tasks and computerized adaptive testing (CAT) task, achieving robust performance. Further analysis reveals the impact of semantic information across diverse tasks, offering key insights for future research on the application of LMs in CD for online intelligent education systems.
>
---
#### [new 038] LangFIR: Discovering Sparse Language-Specific Features from Monolingual Data for Language Steering
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言控制任务，旨在解决多语言大模型输出语言难以控制的问题。通过仅使用单语数据和随机标记，提出LangFIR方法发现稀疏语言特定特征，实现有效语言引导。**

- **链接: [https://arxiv.org/pdf/2604.03532](https://arxiv.org/pdf/2604.03532)**

> **作者:** Sing Hieng Wong; Hassan Sajjad; A.B. Siddique
>
> **备注:** Submitted to COLM 2026
>
> **摘要:** Large language models (LLMs) show strong multilingual capabilities, yet reliably controlling the language of their outputs remains difficult. Representation-level steering addresses this by adding language-specific vectors to model activations at inference time, but identifying language-specific directions in the residual stream often relies on multilingual or parallel data that can be expensive to obtain. Sparse autoencoders (SAEs) decompose residual activations into interpretable, sparse feature directions and offer a natural basis for this search, yet existing SAE-based approaches face the same data constraint. We introduce LangFIR (Language Feature Identification via Random-token Filtering), a method that discovers language-specific SAE features using only a small amount of monolingual data and random-token sequences. Many SAE features consistently activated by target-language inputs do not encode language identity. Random-token sequences surface these language-agnostic features, allowing LangFIR to filter them out and isolate a sparse set of language-specific features. We show that these features are extremely sparse, highly selective for their target language, and causally important: directional ablation increases cross-entropy loss only for the corresponding language. Using these features to construct steering vectors for multilingual generation control, LangFIR achieves the best average accuracy BLEU across three models (Gemma 3 1B, Gemma 3 4B, and Llama 3.1 8B), three datasets, and twelve target languages, outperforming the strongest monolingual baseline by up to and surpassing methods that rely on parallel data. Our results suggest that language identity in multilingual LLMs is localized in a sparse set of feature directions discoverable with monolingual data. Code is available at this https URL.
>
---
#### [new 039] VIGIL: An Extensible System for Real-Time Detection and Mitigation of Cognitive Bias Triggers
- **分类: cs.CL; cs.CY; cs.HC**

- **简介: 该论文提出VIGIL系统，用于实时检测和缓解在线信息中的认知偏见触发器。属于信息真实性维护任务，解决偏见引发的误导问题，通过浏览器扩展实现检测、重写和隐私保护。**

- **链接: [https://arxiv.org/pdf/2604.03261](https://arxiv.org/pdf/2604.03261)**

> **作者:** Bo Kang; Sander Noels; Tijl De Bie
>
> **摘要:** The rise of generative AI is posing increasing risks to online information integrity and civic discourse. Most concretely, such risks can materialise in the form of mis- and disinformation. As a mitigation, media-literacy and transparency tools have been developed to address factuality of information and the reliability and ideological leaning of information sources. However, a subtler but possibly no less harmful threat to civic discourse is to use of persuasion or manipulation by exploiting human cognitive biases and related cognitive limitations. To the best of our knowledge, no tools exist to directly detect and mitigate the presence of triggers of such cognitive biases in online information. We present VIGIL (VIrtual GuardIan angeL), the first browser extension for real-time cognitive bias trigger detection and mitigation, providing in-situ scroll-synced detection, LLM-powered reformulation with full reversibility, and privacy-tiered inference from fully offline to cloud. VIGIL is built to be extensible with third-party plugins, with several plugins that are rigorously validated against NLP benchmarks are already included. It is open-sourced at this https URL.
>
---
#### [new 040] Is a Picture Worth a Thousand Words? Adaptive Multimodal Fact-Checking with Visual Evidence Necessity
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于事实核查任务，旨在解决视觉证据滥用导致准确率下降的问题。提出AMuFC框架，通过两个协作模块智能判断是否使用视觉证据，提升核查效果。**

- **链接: [https://arxiv.org/pdf/2604.04692](https://arxiv.org/pdf/2604.04692)**

> **作者:** Jaeyoon Jung; Yejun Yoon; Kunwoo Park
>
> **备注:** preprint, 18 pages
>
> **摘要:** Automated fact-checking is a crucial task not only in journalism but also across web platforms, where it supports a responsible information ecosystem and mitigates the harms of misinformation. While recent research has progressed from text-only to multimodal fact-checking, a prevailing assumption is that incorporating visual evidence universally improves performance. In this work, we challenge this assumption and show that indiscriminate use of multimodal evidence can reduce accuracy. To address this challenge, we propose AMuFC, a multimodal fact-checking framework that employs two collaborative agents with distinct roles for the adaptive use of visual evidence: An Analyzer determines whether visual evidence is necessary for claim verification, and a Verifier predicts claim veracity conditioned on both the retrieved evidence and the Analyzer's assessment. Experimental results on three datasets show that incorporating the Analyzer's assessment of visual evidence necessity into the Verifier's prediction yields substantial improvements in verification performance. In addition to all code, we release WebFC, a newly constructed dataset for evaluating fact-checking modules in a more realistic scenario, available at this https URL.
>
---
#### [new 041] When Models Know More Than They Say: Probing Analogical Reasoning in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语言模型在类比推理中的表现，探讨其内部表示与提示行为的差异，旨在揭示模型在抽象和泛化上的局限性。**

- **链接: [https://arxiv.org/pdf/2604.03877](https://arxiv.org/pdf/2604.03877)**

> **作者:** Hope McGovern; Caroline Craig; Thomas Lippincott; Hale Sirin
>
> **摘要:** Analogical reasoning is a core cognitive faculty essential for narrative understanding. While LLMs perform well when surface and structural cues align, they struggle in cases where an analogy is not apparent on the surface but requires latent information, suggesting limitations in abstraction and generalisation. In this paper we compare a model's probed representations with its prompted performance at detecting narrative analogies, revealing an asymmetry: for rhetorical analogies, probing significantly outperforms prompting in open-source models, while for narrative analogies, they achieve a similar (low) performance. This suggests that the relationship between internal representations and prompted behavior is task-dependent and may reflect limitations in how prompting accesses available information.
>
---
#### [new 042] Adaptive Cost-Efficient Evaluation for Reliable Patent Claim Validation
- **分类: cs.CL**

- **简介: 该论文属于专利权利要求验证任务，旨在解决自动化验证中精度与成本的矛盾。提出ACE框架，通过预测熵筛选高不确定性案件，结合专家LLM提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.04295](https://arxiv.org/pdf/2604.04295)**

> **作者:** Yongmin Yoo; Qiongkai Xu; Longbing Cao
>
> **摘要:** Automated validation of patent claims demands zero-defect tolerance, as even a single structural flaw can render a claim legally defective. Existing evaluation paradigms suffer from a rigidity-resource dilemma: lightweight encoders struggle with nuanced legal dependencies, while exhaustive verification via Large Language Models (LLMs) is prohibitively costly. To bridge this gap, we propose ACE (Adaptive Cost-efficient Evaluation), a hybrid framework that uses predictive entropy to route only high-uncertainty claims to an expert LLM. The expert then executes a Chain of Patent Thought (CoPT) protocol grounded in 35 U.S.C. statutory standards. This design enables ACE to handle long-range legal dependencies more effectively while preserving efficiency. ACE achieves the best F1 among the evaluated methods at 94.95\%, while reducing operational costs by 78\% compared to standalone LLM deployments. We also construct ACE-40k, a 40,000-claim benchmark with MPEP-grounded error annotations, to facilitate further research.
>
---
#### [new 043] CommonMorph: Participatory Morphological Documentation Platform
- **分类: cs.CL**

- **简介: 该论文介绍CommonMorph平台，用于解决低资源语言形态数据收集与标注难题。通过三阶段方法实现高效、协作的形态学文档记录。**

- **链接: [https://arxiv.org/pdf/2604.04515](https://arxiv.org/pdf/2604.04515)**

> **作者:** Aso Mahmudi; Sina Ahmadi; Kemal Kurniawan; Rico Sennrich; Eduard Hovy; Ekaterina Vylomova
>
> **摘要:** Collecting and annotating morphological data present significant challenges, requiring linguistic expertise, methodological rigour, and substantial resources. These barriers are particularly acute for low-resource languages and varieties. To accelerate this process, we introduce \texttt{CommonMorph}, a comprehensive platform that streamlines morphological data collection development through a three-tiered approach: expert linguistic definition, contributor elicitation, and community validation. The platform minimises manual work by incorporating active learning, annotation suggestions, and tools to import and adapt materials from related languages. It accommodates diverse morphological systems, including fusional, agglutinative, and root-and-pattern morphologies. Its open-source design and UniMorph-compatible outputs ensure accessibility and interoperability with NLP tools. Our platform is accessible at this https URL, offering a replicable model for preserving linguistic diversity through collaborative technology.
>
---
#### [new 044] Multilingual Prompt Localization for Agent-as-a-Judge: Language and Backbone Sensitivity in Requirement-Level Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于代码评估任务，研究多语言对代理评估结果的影响。通过在不同语言和模型上测试，发现语言与模型交互显著，强调语言应作为评估变量。**

- **链接: [https://arxiv.org/pdf/2604.04532](https://arxiv.org/pdf/2604.04532)**

> **作者:** Alhasan Mahmood; Samir Abdaljalil; Hasan Kurban
>
> **摘要:** Evaluation language is typically treated as a fixed English default in agentic code benchmarks, yet we show that changing the judge's language can invert backbone rankings. We localize the Agent-as-a-Judge prompt stack to five typologically diverse languages (English, Arabic, Turkish, Chinese, Hindi) and evaluate 55 DevAI development tasks across three developer-agent frameworks and six judge backbones, totaling 4950 judge runs. The central finding is that backbone and language interact: GPT-4o achieves the highest satisfaction in English (44.72\%), while Gemini leads in Arabic (51.72\%, $p<0.001$ vs.\ GPT-4o) and Hindi (53.22\%). No single backbone dominates across all languages, and inter-backbone agreement on individual requirement judgments is modest (Fleiss' $\kappa \leq 0.231$). A controlled ablation further shows that localizing judge-side instructions, not just benchmark content, can be decisive: Hindi satisfaction drops from 42.8\% to 23.2\% under partial localization. These results indicate that language should be treated as an explicit evaluation variable in agentic benchmarks. Full requirement-level judgments and runtime statistics are released for reproducibility.
>
---
#### [new 045] Cultural Authenticity: Comparing LLM Cultural Representations to Native Human Expectations
- **分类: cs.CL**

- **简介: 该论文属于AI文化表示评估任务，旨在解决LLM输出与本土文化期望的对齐问题。通过构建文化重要性向量并对比模型生成内容，发现部分模型存在西方中心偏差。**

- **链接: [https://arxiv.org/pdf/2604.03493](https://arxiv.org/pdf/2604.03493)**

> **作者:** Erin MacMurray van Liemt; Aida Davani; Sinchana Kumbale; Neha Dixit; Sunipa Dev
>
> **备注:** 18 pages, 4 figures
>
> **摘要:** Cultural representation in Large Language Model (LLM) outputs has primarily been evaluated through the proxies of cultural diversity and factual accuracy. However, a crucial gap remains in assessing cultural alignment: the degree to which generated content mirrors how native populations perceive and prioritize their own cultural facets. In this paper, we introduce a human-centered framework to evaluate the alignment of LLM generations with local expectations. First, we establish a human-derived ground-truth baseline of importance vectors, called Cultural Importance Vectors based on an induced set of culturally significant facets from open-ended survey responses collected across nine countries. Next, we introduce a method to compute model-derived Cultural Representation Vectors of an LLM based on a syntactically diversified prompt-set and apply it to three frontier LLMs (Gemini 2.5 Pro, GPT-4o, and Claude 3.5 Haiku). Our investigation of the alignment between the human-derived Cultural Importance and model-derived Cultural Representations reveals a Western-centric calibration for some of the models where alignment decreases as a country's cultural distance from the US increases. Furthermore, we identify highly correlated, systemic error signatures ($\rho > 0.97$) across all models, which over-index on some cultural markers while neglecting the deep-seated social and value-based priorities of users. Our approach moves beyond simple diversity metrics toward evaluating the fidelity of AI-generated content in authentically capturing the nuanced hierarchies of global cultures.
>
---
#### [new 046] Uncertainty as a Planning Signal: Multi-Turn Decision Making for Goal-Oriented Conversation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统任务，解决目标导向对话中的多轮决策问题。通过引入不确定性作为规划信号，提出CUP框架，提升信息获取效率和决策准确性。**

- **链接: [https://arxiv.org/pdf/2604.03924](https://arxiv.org/pdf/2604.03924)**

> **作者:** Xinyi Ling; Ye Liu; Reza Averly; Xia Ning
>
> **摘要:** Goal-oriented conversational systems require making sequential decisions under uncertainty about the user's intent, where the algorithm must balance information acquisition and target commitment over multiple turns. Existing approaches address this challenge from different perspectives: structured methods enable multi-step planning but rely on predefined schemas, while LLM-based approaches support flexible interactions but lack long-horizon decision making, resulting in poor coordination between information acquisition and target commitment. To address this limitation, we formulate goal-oriented conversation as an uncertainty-aware sequential decision problem, where uncertainty serves as a guiding signal for multi-turn decision making. We propose a Conversation Uncertainty-aware Planning framework (CUP) that integrates language models with structured planning: a language model proposes feasible actions, and a planner evaluates their long-term impact on uncertainty reduction. Experiments on multiple conversational benchmarks show that CUP consistently improves success rates while requiring fewer interaction turns. Further analysis demonstrates that uncertainty-aware planning contributes to more efficient information acquisition and earlier confident commitment.
>
---
#### [new 047] Plausibility as Commonsense Reasoning: Humans Succeed, Large Language Models Do not
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型在歧义解析中是否像人类一样利用常识推理。通过土耳其语的名词性从句歧义实验，发现人类能正确利用常识，而大语言模型表现不佳。任务为语言理解中的歧义解析。**

- **链接: [https://arxiv.org/pdf/2604.04825](https://arxiv.org/pdf/2604.04825)**

> **作者:** Sercan Karakaş
>
> **备注:** Accepted to The Workshop on Cognitive Modeling and Computational Linguistics co-located with LREC 2026
>
> **摘要:** Large language models achieve strong performance on many language tasks, yet it remains unclear whether they integrate world knowledge with syntactic structure in a human-like, structure-sensitive way during ambiguity resolution. We test this question in Turkish prenominal relative-clause attachment ambiguities, where the same surface string permits high attachment (HA) or low attachment (LA). We construct ambiguous items that keep the syntactic configuration fixed and ensure both parses remain pragmatically possible, while graded event plausibility selectively favors High Attachment vs.\ Low Attachment. The contrasts are validated with independent norming ratings. In a speeded forced-choice comprehension experiment, humans show a large, correctly directed plausibility effect. We then evaluate Turkish and multilingual LLMs in a parallel preference-based setup that compares matched HA/LA continuations via mean per-token log-probability. Across models, plausibility-driven shifts are weak, unstable, or reversed. The results suggest that, in the tested models, plausibility information does not guide attachment preferences as reliably as it does in human judgments, and they highlight Turkish RC attachment as a useful cross-linguistic diagnostic beyond broad benchmarks.
>
---
#### [new 048] Which English Do LLMs Prefer? Triangulating Structural Bias Towards American English in Foundation Models
- **分类: cs.CL; cs.AI; cs.CY; cs.ET; cs.LG**

- **简介: 该论文属于自然语言处理任务，探讨LLMs对美式英语的偏好问题。通过分析数据集、分词器和生成结果，发现模型存在结构性偏见，倾向使用美式英语，引发语言同质化等担忧。**

- **链接: [https://arxiv.org/pdf/2604.04204](https://arxiv.org/pdf/2604.04204)**

> **作者:** Mir Tafseer Nayeem; Davood Rafiei
>
> **备注:** Preprint
>
> **摘要:** Large language models (LLMs) are increasingly deployed in high-stakes domains, yet they expose only limited language settings, most notably "English (US)," despite the global diversity and colonial history of English. Through a postcolonial framing to explain the broader significance, we investigate how geopolitical histories of data curation, digital dominance, and linguistic standardization shape the LLM development pipeline. Focusing on two dominant standard varieties, American English (AmE) and British English (BrE), we construct a curated corpus of 1,813 AmE--BrE variants and introduce DiAlign, a dynamic, training-free method for estimating dialectal alignment using distributional evidence. We operationalize structural bias by triangulating evidence across three stages: (i) audits of six major pretraining corpora reveal systematic skew toward AmE, (ii) tokenizer analyses show that BrE forms incur higher segmentation costs, and (iii) generative evaluations show a persistent AmE preference in model outputs. To our knowledge, this is the first systematic and multi-faceted examination of dialectal asymmetries in standard English varieties across the phases of LLM development. We find that contemporary LLMs privilege AmE as the de facto norm, raising concerns about linguistic homogenization, epistemic injustice, and inequity in global AI deployment, while motivating practical steps toward more dialectally inclusive language technologies.
>
---
#### [new 049] MultiPress: A Multi-Agent Framework for Interpretable Multimodal News Classification
- **分类: cs.CL**

- **简介: 该论文属于多模态新闻分类任务，旨在解决现有方法无法有效融合文本与图像信息的问题。提出MultiPress框架，通过多智能体协作提升分类效果与可解释性。**

- **链接: [https://arxiv.org/pdf/2604.03586](https://arxiv.org/pdf/2604.03586)**

> **作者:** Tailong Luo; Hao Li; Rong Fu; Xinyue Jiang; Huaxuan Ding; Yiduo Zhang; Zilin Zhao; Simon Fong; Guangyin Jin; Jianyuan Ni
>
> **备注:** Accepted in International Joint Conference on Neural Networks (IJCNN) 2026
>
> **摘要:** With the growing prevalence of multimodal news content, effective news topic classification demands models capable of jointly understanding and reasoning over heterogeneous data such as text and images. Existing methods often process modalities independently or employ simplistic fusion strategies, limiting their ability to capture complex cross-modal interactions and leverage external knowledge. To overcome these limitations, we propose MultiPress, a novel three-stage multi-agent framework for multimodal news classification. MultiPress integrates specialized agents for multimodal perception, retrieval-augmented reasoning, and gated fusion scoring, followed by a reward-driven iterative optimization mechanism. We validate MultiPress on a newly constructed large-scale multimodal news dataset, demonstrating significant improvements over strong baselines and highlighting the effectiveness of modular multi-agent collaboration and retrieval-augmented reasoning in enhancing classification accuracy and interpretability.
>
---
#### [new 050] Lighting Up or Dimming Down? Exploring Dark Patterns of LLMs in Co-Creativity
- **分类: cs.CL**

- **简介: 该论文属于人机协作任务，探讨LLMs在共创中抑制创造力的“暗模式”，分析五种行为对创作的影响，并提出设计建议。**

- **链接: [https://arxiv.org/pdf/2604.04735](https://arxiv.org/pdf/2604.04735)**

> **作者:** Zhu Li; Jiaming Qu; Yuan Chang
>
> **摘要:** Large language models (LLMs) are increasingly acting as collaborative writing partners, raising questions about their impact on human agency. In this exploratory work, we investigate five "dark patterns" in human-AI co-creativity -- subtle model behaviors that can suppress or distort the creative process: Sycophancy, Tone Policing, Moralizing, Loop of Death, and Anchoring. Through a series of controlled sessions where LLMs are prompted as writing assistants across diverse literary forms and themes, we analyze the prevalence of these behaviors in generated responses. Our preliminary results suggest that Sycophancy is nearly ubiquitous (91.7% of cases), particularly in sensitive topics, while Anchoring appears to be dependent on literary forms, surfacing most frequently in folktales. This study indicates that these dark patterns, often byproducts of safety alignment, may inadvertently narrow creative exploration and proposes design considerations for AI systems that effectively support creative writing.
>
---
#### [new 051] 'Layer su Layer': Identifying and Disambiguating the Italian NPN Construction in BERT's family
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的可解释性研究任务，旨在分析BERT对意大利语NPN结构的表示，通过分层探测揭示其编码的语法信息。**

- **链接: [https://arxiv.org/pdf/2604.03673](https://arxiv.org/pdf/2604.03673)**

> **作者:** Greta Gorzoni; Ludovica Pannitto; Francesca Masini
>
> **摘要:** Interpretability research has highlighted the importance of evaluating Pretrained Language Models (PLMs) and in particular contextual embeddings against explicit linguistic theories to determine what linguistic information they encode. This study focuses on the Italian NPN (noun-preposition-noun) constructional family, challenging some of the theoretical and methodological assumptions underlying previous experimental designs and extending this type of research to a lesser-investigated language. Contextual vector representations are extracted from BERT and used as input to layer-wise probing classifiers, systematically evaluating information encoded across the model's internal layers. The results shed light on the extent to which constructional form and meaning are reflected in contextual embeddings, contributing empirical evidence to the dialogue between constructionist theory and neural language modelling
>
---
#### [new 052] Shorter, but Still Trustworthy? An Empirical Study of Chain-of-Thought Compression
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，研究CoT压缩对模型可信度的影响。通过实验发现压缩常导致可信度下降，提出改进方法以平衡效率与可信度。**

- **链接: [https://arxiv.org/pdf/2604.04120](https://arxiv.org/pdf/2604.04120)**

> **作者:** Lingjie Zeng; Xiaofan Chen; Yanbo Wang; Xiuying Chen
>
> **摘要:** Long chain-of-thought (Long-CoT) reasoning models have motivated a growing body of work on compressing reasoning traces to reduce inference cost, yet existing evaluations focus almost exclusively on task accuracy and token savings. Trustworthiness properties, whether acquired or reinforced through post-training, are encoded in the same parameter space that compression modifies. This means preserving accuracy does not, a priori, guarantee preserving trustworthiness. We conduct the first systematic empirical study of how CoT compression affects model trustworthiness, evaluating multiple models of different scales along three dimensions: safety, hallucination resistance, and multilingual robustness. Under controlled comparisons, we find that CoT compression frequently introduces trustworthiness regressions and that different methods exhibit markedly different degradation profiles across dimensions. To enable fair comparison across bases, we propose a normalized efficiency score for each dimension that reveals how naïve scalar metrics can obscure trustworthiness trade-offs. As an existence proof, we further introduce an alignment-aware DPO variant that reduces CoT length by 19.3\% on reasoning benchmarks with substantially smaller trustworthiness loss. Our findings suggest that CoT compression should be optimized not only for efficiency but also for trustworthiness, treating both as equally important design constraints.
>
---
#### [new 053] How Alignment Routes: Localizing, Scaling, and Controlling Policy Circuits in Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文研究语言模型中的对齐路由机制，解决政策电路的定位与控制问题。通过实验分析门控头和放大器头的作用，揭示意图识别与策略路由的结构差异。**

- **链接: [https://arxiv.org/pdf/2604.04385](https://arxiv.org/pdf/2604.04385)**

> **作者:** Gregory N. Frank
>
> **摘要:** We identify a recurring sparse routing mechanism in alignment-trained language models: a gate attention head reads detected content and triggers downstream amplifier heads that boost the signal toward refusal. Using political censorship and safety refusal as natural experiments, we trace this mechanism across 9 models from 6 labs, all validated on corpora of 120 prompt pairs. The gate head passes necessity and sufficiency interchange tests (p < 0.001, permutation null), and core amplifier heads are stable under bootstrap resampling (Jaccard 0.92-1.0). Three same-generation scaling pairs show that routing distributes at scale (ablation up to 17x weaker) while remaining detectable by interchange. By modulating the detection-layer signal, we continuously control policy strength from hard refusal through steering to factual compliance, with routing thresholds that vary by topic. The circuit also reveals a structural separation between intent recognition and policy routing: under cipher encoding, the gate head's routing contribution collapses (78% in Phi-4 at n=120) while the model responds with puzzle-solving rather than refusal. The routing mechanism never fires, even though probe scores at deeper layers indicate the model begins to represent the harmful content. This asymmetry is consistent with different robustness properties of pretraining and post-training: broad semantic understanding versus narrower policy binding that generalizes less well under input transformation.
>
---
#### [new 054] Formal Constraints on Dependency Syntax
- **分类: cs.CL**

- **简介: 论文探讨依赖句法的约束问题，旨在寻找介于严格项目性与无限制结构之间的合理中间方案，以更准确描述语言现象。**

- **链接: [https://arxiv.org/pdf/2604.04542](https://arxiv.org/pdf/2604.04542)**

> **作者:** Gómez-Rodríguez; Carlos; Alemany-Puig; Lluís
>
> **摘要:** Dependency syntax represents the structure of a sentence as a tree composed of dependencies, i.e., directed relations between lexical units. While in its more general form any such tree is allowed, in practice many are not plausible or are very infrequent in attested language. This has motivated a search for constraints characterizing subsets of trees that better fit real linguistic phenomena, providing a more accurate linguistic description, faster parsing or insights on language evolution and human processing. Projectivity is the most well-studied such constraint, but it has been shown to be too restrictive to represent some linguistic phenomena, especially in flexible-word-order languages. Thus, a variety of constraints have been proposed to seek a realistic middle ground between the limitations of projectivity and the excessive leniency of unrestricted dependency structures.
>
---
#### [new 055] POEMetric: The Last Stanza of Humanity
- **分类: cs.CL**

- **简介: 该论文属于诗歌生成评估任务，旨在衡量大语言模型在诗歌创作中的表现。通过构建POEMetric框架，评估模型在形式、创意和整体质量等方面的能力，发现其仍无法超越人类诗人。**

- **链接: [https://arxiv.org/pdf/2604.03695](https://arxiv.org/pdf/2604.03695)**

> **作者:** Bingru Li; Han Wang; Hazel Wilkinson
>
> **摘要:** Large Language Models (LLMs) can compose poetry, but how far are they from human poets? In this paper, we introduce POEMetric, the first comprehensive framework for poetry evaluation, examining 1) basic instruction-following abilities in generating poems according to a certain form and theme, 2) advanced abilities of showing creativity, lexical diversity, and idiosyncrasy, evoking emotional resonance, and using imagery and literary devices, and 3) general appraisal of the overall poem quality and estimation of authorship. We curated a human poem dataset - 203 English poems of 7 fixed forms annotated with meter, rhyme patterns and themes - and experimented with 30 LLMs for poetry generation based on the same forms and themes of the human data, totaling 6,090 LLM poems. Based on POEMetric, we assessed the performance of both human poets and LLMs through rule-based evaluation and LLM-as-a-judge, whose results were validated by human experts. Results show that, though the top model achieved high form accuracy (4.26 out of 5.00, with Gemini-2.5-Pro as a judge; same below) and theme alignment (4.99), all models failed to reach the same level of advanced abilities as human poets, who achieved unparalleled creativity (4.02), idiosyncrasy (3.95), emotional resonance (4.06), and skillful use of imagery (4.49) and literary devices (4.67). Humans also defeated the best-performing LLM in overall poem quality (4.22 vs. 3.20). As such, poetry generation remains a formidable challenge for LLMs. Data and codes are released at this https URL.
>
---
#### [new 056] Text Summarization With Graph Attention Networks
- **分类: cs.CL**

- **简介: 该论文属于文本摘要任务，旨在通过图结构信息提升摘要模型性能。研究尝试使用图注意力网络未获成功，改用多层感知机取得进展，并构建了基于RST的基准数据集。**

- **链接: [https://arxiv.org/pdf/2604.03583](https://arxiv.org/pdf/2604.03583)**

> **作者:** Mohammadreza Ardestani; Yllias Chali
>
> **备注:** Published in Proceedings of the 4th NeurIPS Efficient Natural Language and Speech Processing Workshop (ENLSP-IV), Vancouver, Canada, 2024. 14 pages, 8 figures
>
> **摘要:** This study aimed to leverage graph information, particularly Rhetorical Structure Theory (RST) and Co-reference (Coref) graphs, to enhance the performance of our baseline summarization models. Specifically, we experimented with a Graph Attention Network architecture to incorporate graph information. However, this architecture did not enhance the performance. Subsequently, we used a simple Multi-layer Perceptron architecture, which improved the results in our proposed model on our primary dataset, CNN/DM. Additionally, we annotated XSum dataset with RST graph information, establishing a benchmark for future graph-based summarization models. This secondary dataset posed multiple challenges, revealing both the merits and limitations of our models.
>
---
#### [new 057] LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于虚假新闻检测任务，旨在解决静态基准无法适应时间变化和评估推理能力的问题。提出动态基准LiveFact，支持实时更新和证据推理评估。**

- **链接: [https://arxiv.org/pdf/2604.04815](https://arxiv.org/pdf/2604.04815)**

> **作者:** Cheng Xu; Changhong Jin; Yingjie Niu; Nan Yan; Yuke Mei; Shuhao Guan; Liming Chen; M-Tahar Kechadi
>
> **备注:** ACL 2026 Main
>
> **摘要:** The rapid development of Large Language Models (LLMs) has transformed fake news detection and fact-checking tasks from simple classification to complex reasoning. However, evaluation frameworks have not kept pace. Current benchmarks are static, making them vulnerable to benchmark data contamination (BDC) and ineffective at assessing reasoning under temporal uncertainty. To address this, we introduce LiveFact a continuously updated benchmark that simulates the real-world "fog of war" in misinformation detection. LiveFact uses dynamic, temporal evidence sets to evaluate models on their ability to reason with evolving, incomplete information rather than on memorized knowledge. We propose a dual-mode evaluation: Classification Mode for final verification and Inference Mode for evidence-based reasoning, along with a component to monitor BDC explicitly. Tests with 22 LLMs show that open-source Mixture-of-Experts models, such as Qwen3-235B-A22B, now match or outperform proprietary state-of-the-art systems. More importantly, our analysis finds a significant "reasoning gap." Capable models exhibit epistemic humility by recognizing unverifiable claims in early data slices-an aspect traditional static benchmarks overlook. LiveFact sets a sustainable standard for evaluating robust, temporally aware AI verification.
>
---
#### [new 058] Document-Level Numerical Reasoning across Single and Multiple Tables in Financial Reports
- **分类: cs.CL**

- **简介: 该论文属于文档级数值推理任务，旨在解决金融报告中跨表格的数值问答问题。针对长文档和多步骤计算的挑战，提出FinLongDocQA数据集和FinLongDocAgent方法。**

- **链接: [https://arxiv.org/pdf/2604.03664](https://arxiv.org/pdf/2604.03664)**

> **作者:** Yi-Cheng Wang; Wei-An Wang; Chu-Song Chen
>
> **摘要:** Despite the strong language understanding abilities of large language models (LLMs), they still struggle with reliable question answering (QA) over long, structured documents, particularly for numerical reasoning. Financial annual reports exemplify this difficulty: financial statement analysis often hinges on accurate arithmetic, and analysts derive key indicators by integrating evidence scattered across multiple tables and narrative text. However, existing benchmarks focus largely on single-table settings, leaving cross-table document-level numerical reasoning underexplored. To address this gap, we introduce FinLongDocQA, a dataset for both single-table and cross-table financial numerical reasoning in long-context reports. Evaluating both closed-source and open-source LLMs on FinLongDocQA reveals two bottlenecks: (1) annual reports often exceed 129k tokens, exacerbating the context rot problem for locating relevant tables; and (2) even when relevant evidence is located, LLMs remain prone to errors in multi-step numerical reasoning. We propose FinLongDocAgent, a Multi-Agent Multi-Round Retrieval-Augmented Generation (RAG) approach that iteratively retrieves evidence, performs intermediate calculations, and verifies results across rounds. Experiments highlight the importance of iterative retrieval and verification for reliable numerical QA in long financial documents.
>
---
#### [new 059] Rethinking Token Prediction: Tree-Structured Diffusion Language Model
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型任务，旨在解决离散扩散模型训练效率低的问题。通过构建树状结构减少预测维度，降低内存消耗并提升参数利用率。**

- **链接: [https://arxiv.org/pdf/2604.03537](https://arxiv.org/pdf/2604.03537)**

> **作者:** Zihao Wu; Haoming Yang; Juncheng Dong; Vahid Tarokh
>
> **摘要:** Discrete diffusion language models have emerged as a competitive alternative to auto-regressive language models, but training them efficiently under limited parameter and memory budgets remains challenging. Modern architectures are predominantly based on a full-vocabulary token prediction layer, which accounts for a substantial fraction of model parameters (e.g., more than 20% in small scale DiT-style designs) and often dominates peak GPU memory usage. This leads to inefficient use of both parameters and memory under constrained training resources. To address this issue, we revisit the necessity of explicit full-vocabulary prediction, and instead exploit the inherent structure among tokens to build a tree-structured diffusion language model. Specifically, we model the diffusion process with intermediate latent states corresponding to a token's ancestor nodes in a pre-constructed vocabulary tree. This tree-structured factorization exponentially reduces the classification dimensionality, makes the prediction head negligible in size, and enables reallocation of parameters to deepen the attention blocks. Empirically, under the same parameter budget, our method reduces peak GPU memory usage by half while matching the perplexity performance of state-of-the-art discrete diffusion language models.
>
---
#### [new 060] GeoBrowse: A Geolocation Benchmark for Agentic Tool Use with Expert-Annotated Reasoning Traces
- **分类: cs.CL**

- **简介: 该论文提出GeoBrowse，一个结合视觉推理与多跳查询的地理定位基准，解决多步骤工具使用中的信息整合问题。**

- **链接: [https://arxiv.org/pdf/2604.04017](https://arxiv.org/pdf/2604.04017)**

> **作者:** Xinyu Geng; Yanjing Xiao; Yuyang Zhang; Hanwen Wang; Xinyan Liu; Rui Min; Tianqing Fang; Yi R. Fung
>
> **摘要:** Deep research agents integrate fragmented evidence through multi-step tool use. BrowseComp offers a text-only testbed for such agents, but existing multimodal benchmarks rarely require both weak visual cues composition and BrowseComp-style multi-hop verification. Geolocation is a natural testbed because answers depend on combining multiple ambiguous visual cues and validating them with open-web evidence. Thus, we introduce GeoBrowse, a geolocation benchmark that combines visual reasoning with knowledge-intensive multi-hop queries. Level 1 tests extracting and composing fragmented visual cues, and Level 2 increases query difficulty by injecting long-tail knowledge and obfuscating key entities. To support evaluation, we provide an agentic workflow GATE with five think-with-image tools and four knowledge-intensive tools, and release expert-annotated stepwise traces grounded in verifiable evidence for trajectory-level analysis. Experiments show that GATE outperforms direct inference and open-source agents, indicating that no-tool, search-only or image-only setups are insufficient. Gains come from coherent, level-specific tool-use plans rather than more tool calls, as they more reliably reach annotated key evidence steps and make fewer errors when integrating into the final decision. The GeoBrowse bernchmark and codes are provided in this https URL
>
---
#### [new 061] Hallucination Basins: A Dynamic Framework for Understanding and Controlling LLM Hallucinations
- **分类: cs.CL; cs.AI; eess.SY**

- **简介: 该论文属于自然语言处理领域，旨在解决大语言模型的幻觉问题。通过构建动态几何框架，分析并控制幻觉产生的机制。**

- **链接: [https://arxiv.org/pdf/2604.04743](https://arxiv.org/pdf/2604.04743)**

> **作者:** Kalyan Cherukuri; Lav R. Varshney
>
> **摘要:** Large language models (LLMs) hallucinate: they produce fluent outputs that are factually incorrect. We present a geometric dynamical systems framework in which hallucinations arise from task-dependent basin structure in latent space. Using autoregressive hidden-state trajectories across multiple open-source models and benchmarks, we find that separability is strongly task-dependent rather than universal: factoid settings can show clearer basin separation, whereas summarization and misconception-heavy settings are typically less stable and often overlap. We formalize this behavior with task-complexity and multi-basin theorems, characterize basin emergence in L-layer transformers, and show that geometry-aware steering can reduce hallucination probability without retraining.
>
---
#### [new 062] SkillX: Automatically Constructing Skill Knowledge Bases for Agents
- **分类: cs.CL; cs.AI; cs.IR; cs.LG; cs.MA**

- **简介: 该论文提出SkillX，用于构建可复用的技能知识库，解决LLM代理学习效率低、泛化差的问题。通过自动化流程提升代理性能。**

- **链接: [https://arxiv.org/pdf/2604.04804](https://arxiv.org/pdf/2604.04804)**

> **作者:** Chenxi Wang; Zhuoyun Yu; Xin Xie; Wuguannan Yao; Runnan Fang; Shuofei Qiao; Kexin Cao; Guozhou Zheng; Xiang Qi; Peng Zhang; Shumin Deng
>
> **备注:** Work in progress
>
> **摘要:** Learning from experience is critical for building capable large language model (LLM) agents, yet prevailing self-evolving paradigms remain inefficient: agents learn in isolation, repeatedly rediscover similar behaviors from limited experience, resulting in redundant exploration and poor generalization. To address this problem, we propose SkillX, a fully automated framework for constructing a \textbf{plug-and-play skill knowledge base} that can be reused across agents and environments. SkillX operates through a fully automated pipeline built on three synergistic innovations: \textit{(i) Multi-Level Skills Design}, which distills raw trajectories into three-tiered hierarchy of strategic plans, functional skills, and atomic skills; \textit{(ii) Iterative Skills Refinement}, which automatically revises skills based on execution feedback to continuously improve library quality; and \textit{(iii) Exploratory Skills Expansion}, which proactively generates and validates novel skills to expand coverage beyond seed training data. Using a strong backbone agent (GLM-4.6), we automatically build a reusable skill library and evaluate its transferability on challenging long-horizon, user-interactive benchmarks, including AppWorld, BFCL-v3, and $\tau^2$-Bench. Experiments show that SkillKB consistently improves task success and execution efficiency when plugged into weaker base agents, highlighting the importance of structured, hierarchical experience representations for generalizable agent learning. Our code will be publicly available soon at this https URL.
>
---
#### [new 063] GROUNDEDKG-RAG: Grounded Knowledge Graph Index for Long-document Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于长文档问答任务，旨在解决RAG系统依赖大模型描述导致的资源消耗高、重复内容多和幻觉问题。提出GroundedKG-RAG，通过从文档中提取并接地的知识图谱提升效率和准确性。**

- **链接: [https://arxiv.org/pdf/2604.04359](https://arxiv.org/pdf/2604.04359)**

> **作者:** Tianyi Zhang; Andreas Marfurt
>
> **备注:** To appear in the Proceedings of KG-LLM @ LREC 2026
>
> **摘要:** Retrieval-augmented generation (RAG) systems have been widely adopted in contemporary large language models (LLMs) due to their ability to improve generation quality while reducing the required input context length. In this work, we focus on RAG systems for long-document question answering. Current approaches suffer from a heavy reliance on LLM descriptions resulting in high resource consumption and latency, repetitive content across hierarchical levels, and hallucinations due to no or limited grounding in the source text. To improve both efficiency and factual accuracy through grounding, we propose GroundedKG-RAG, a RAG system in which the knowledge graph is explicitly extracted from and grounded in the source document. Specifically, we define nodes in GroundedKG as entities and actions, and edges as temporal or semantic relations, with each node and edge grounded in the original sentences. We construct GroundedKG from semantic role labeling (SRL) and abstract meaning representation (AMR) parses and then embed it for retrieval. During querying, we apply the same transformation to the query and retrieve the most relevant sentences from the grounded source text for question answering. We evaluate GroundedKG-RAG on examples from the NarrativeQA dataset and find that it performs on par with a state-of-the art proprietary long-context model at smaller cost and outperforms a competitive baseline. Additionally, our GroundedKG is interpretable and readable by humans, facilitating auditing of results and error analysis.
>
---
#### [new 064] BiST: A Gold Standard Bangla-English Bilingual Corpus for Sentence Structure and Tense Classification with Inter-Annotator Agreement
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出BiST，一个用于句法和时态分类的孟加拉语-英语双语语料库，解决低资源语言NLP数据不足的问题。通过多阶段标注和高一致性评估，构建高质量语料，支持语法建模与跨语言研究。**

- **链接: [https://arxiv.org/pdf/2604.04708](https://arxiv.org/pdf/2604.04708)**

> **作者:** Abdullah Al Shafi; Swapnil Kundu Argha; M. A. Moyeen; Abdul Muntakim; Shoumik Barman Polok
>
> **摘要:** High-quality bilingual resources remain a critical bottleneck for advancing multilingual NLP in low-resource settings, particularly for Bangla. To mitigate this gap, we introduce BiST, a rigorously curated Bangla-English corpus for sentence-level grammatical classification, annotated across two fundamental dimensions: syntactic structure (Simple, Complex, Compound, Complex-Compound) and tense (Present, Past, Future). The corpus is compiled from open-licensed encyclopedic sources and naturally composed conversational text, followed by systematic preprocessing and automated language identification, resulting in 30,534 sentences, including 17,465 English and 13,069 Bangla instances. Annotation quality is ensured through a multi-stage framework with three independent annotators and dimension-wise Fleiss Kappa ($\kappa$) agreement, yielding reliable and reproducible labels with $\kappa$ values of 0.82 and 0.88 for structural and temporal annotation, respectively. Statistical analyses demonstrate realistic structural and temporal distributions, while baseline evaluations show that dual-encoder architectures leveraging complementary language-specific representations consistently outperform strong multilingual encoders. Beyond benchmarking, BiST provides explicit linguistic supervision that supports grammatical modeling tasks, including controlled text generation, automated feedback generation, and cross-lingual representation learning. The corpus establishes a unified resource for bilingual grammatical modeling and facilitates linguistically grounded multilingual research.
>
---
#### [new 065] Unmasking Hallucinations: A Causal Graph-Attention Perspective on Factual Reliability in Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型中的幻觉问题。通过构建因果图注意力网络，提升模型的事实可靠性。**

- **链接: [https://arxiv.org/pdf/2604.04020](https://arxiv.org/pdf/2604.04020)**

> **作者:** Sailesh kiran kurra; Shiek Ruksana; Vishal Borusu
>
> **备注:** Paper accepted for publication at IEEE International Conference on Emerging Computing and Intelligent Technologies 2026 (ICoECIT),5 Pages,5 figures,1 table
>
> **摘要:** This paper primarily focuses on the hallucinations caused due to AI language models(LLMs).LLMs have shown extraordinary Language understanding and generation capabilities .Still it has major a disadvantage hallucinations which give outputs which are factually incorrect ,misleading or unsupported by input data . These hallucinations cause serious problems in scenarios like medical diagnosis or legal this http URL this work,we propose causal graph attention network (GCAN) framework that reduces hallucinations through interpretation of internal attention flow within a transformer architecture with the help of constructing token level graphs that combine self attention weights and gradient based influence this http URL method quantifies each tokens factual dependency using a new metric called the Causal Contribution Score (CCS). We further introduce a fact-anchored graph reweighting layer that dynamically reduces the influence of hallucination prone nodes during generation. Experiments on standard benchmarks such as TruthfulQA and HotpotQA show a 27.8 percent reduction in hallucination rate and 16.4 percent improvement in factual accuracy over baseline retrieval-augmented generation (RAG) models. This work contributes to the interpretability,robustness, and factual reliability of future LLM architectures.
>
---
#### [new 066] Benchmarking Multi-turn Medical Diagnosis: Hold, Lure, and Self-Correction
- **分类: cs.CL**

- **简介: 该论文属于医疗诊断任务，旨在解决多轮对话中语言模型的诊断准确性问题。通过构建MINT基准，分析模型行为并提出优化策略。**

- **链接: [https://arxiv.org/pdf/2604.04325](https://arxiv.org/pdf/2604.04325)**

> **作者:** Jinrui Fang; Runhan Chen; Xu Yang; Jian Yu; Jiawei Xu; Ashwin Vinod; Wenqi Shi; Tianlong Chen; Heng Ji; ChengXiang Zhai; Ying Ding; Yuji Zhang
>
> **摘要:** Large language models (LLMs) achieve high accuracy in medical diagnosis when all clinical information is provided in a single turn, yet how they behave under multi-turn evidence accumulation closer to real clinical reasoning remains unexplored. We introduce MINT (Medical Incremental N-Turn Benchmark), a high-fidelity, multi-turn medical diagnosis benchmark comprising 1,035 cases with clinically labeled evidence shards, controlled turn granularity, and information-preserving decomposition. Through systematic evaluation of 11 LLMs on MINT, we uncover three persistent behavioral patterns that significantly impact diagnostic decisions: (1) intent to answer, models rush to answer before sufficient evidence has been observed, with over 55% of answers committed within the first two turns; (2) self-correction, incorrect-to-correct answer revisions occur at up to 10.6 times the rate of correct-to-incorrect flips, revealing a latent capacity for self-correction that premature commitment forecloses; and (3) strong lures, clinically salient information such as laboratory results trigger premature answering even when models are explicitly instructed to wait. We translate these findings into clinically actionable guidance: deferring the diagnostic question to later turns reduces premature answering and improves accuracy at the first point of commitment by up to 62.6%, while reserving salient clinical evidence for later turns prevents a catastrophic accuracy drop of up to 23.3% caused by premature commitment. Our work provides both a controlled evaluation framework and concrete recommendations for improving the reliability of LLMs in multi-turn medical diagnosis.
>
---
#### [new 067] I-CALM: Incentivizing Confidence-Aware Abstention for LLM Hallucination Mitigation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型的幻觉问题。通过设计I-CALM框架，鼓励模型在不确定时选择不回答，从而提升答案的可靠性。**

- **链接: [https://arxiv.org/pdf/2604.03904](https://arxiv.org/pdf/2604.03904)**

> **作者:** Haotian Zong; Binze Li; Yufei Long; Sinyin Chang; Jialong Wu; Gillian K. Hadfield
>
> **摘要:** Large language models (LLMs) frequently produce confident but incorrect answers, partly because common binary scoring conventions reward answering over honestly expressing uncertainty. We study whether prompt-only interventions -- explicitly announcing reward schemes for answer-versus-abstain decisions plus humility-oriented normative principles -- can reduce hallucination risk without modifying the model. Our focus is epistemic abstention on factual questions with a verifiable answer, where current LLMs often fail to abstain despite being uncertain about their answers. We first assess self-reported verbal confidence as a usable uncertainty signal, showing stability under prompt paraphrasing and reasonable calibration against a token-probability baseline. We then study I-CALM, a prompt-based framework that (i) elicits verbal confidence, (ii) partially rewards abstention through explicit reward schemes, and (iii) adds lightweight normative principles emphasizing truthfulness, humility, and responsibility. Using GPT-5 mini on PopQA as the main setting, we find that confidence-eliciting, abstention-rewarding prompts, especially with norms, reduce the false-answer rate on answered cases mainly by identifying and shifting error-prone cases to abstention and re-calibrating their confidence. This trades coverage for reliability while leaving forced-answer performance largely unchanged. Varying the abstention reward yields a clear abstention-hallucination frontier. Overall, results show the framework can improve selective answering on factual questions without retraining, with the magnitude of effect varying across models and datasets. Code is available at the following this https URL.
>
---
#### [new 068] Benchmarking Multilingual Speech Models on Pashto: Zero-Shot ASR, Script Failure, and Cross-Domain Evaluation
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决Pashto语言的多模型评估问题，包括零样本ASR、脚本失败和跨领域测试，分析模型表现并提出改进方向。**

- **链接: [https://arxiv.org/pdf/2604.04598](https://arxiv.org/pdf/2604.04598)**

> **作者:** Hanif Rahman
>
> **摘要:** Pashto is spoken by approximately 60--80 million people but has no published benchmarks for multilingual automatic speech recognition (ASR) on any shared public test set. This paper reports the first reproducible multi-model evaluation on public Pashto data, covering zero-shot ASR, script-level failure, and cross-domain evaluation of fine-tuned models. For zero-shot ASR, ten models (all seven Whisper sizes, MMS-1B, SeamlessM4T-v2-large, and OmniASR-CTC-300M) are evaluated on the FLEURS Pashto test set and a filtered Common Voice~24 subset; zero-shot Whisper WER ranges from 90% to 297%, with the medium model collapsing to 461% on Common Voice~24 consistent with decoder looping. SeamlessM4T achieves 39.7% WER on Common Voice~24 (the best zero-shot result reported to date, as of submission); MMS-1B achieves 43.8% on FLEURS. For script failure, a language-identification audit shows that no Whisper model produces Pashto-script output in more than 0.8% of utterances, while MMS-1B, SeamlessM4T, and OmniASR each exceed 93% Pashto-script fidelity; WER alone does not reveal this failure, since a model generating Arabic-script output on Pashto audio has not achieved ASR in any interpretable sense. For cross-domain evaluation, five fine-tuned Pashto ASR models are evaluated on both test sets: published WER figures of 14% degrade to 32.5--59% on out-of-distribution sets, while one augmented model achieves 35.1% on both sets with zero cross-domain degradation. Character-class error stratification confirms that Pashto-unique phonemes (the retroflex series and lateral fricatives) account for disproportionate error mass. All evaluations cover read speech only. Five structural impediments to cumulative progress are identified and five ordered research priorities are argued.
>
---
#### [new 069] IDIOLEX: Unified and Continuous Representations for Idiolectal and Stylistic Variation
- **分类: cs.CL**

- **简介: 该论文提出IDIOLEX框架，用于学习句子的风格和方言表示，解决传统句表示忽略表达方式的问题。通过结合语境和语言特征，实现对风格和方言的连续表征。**

- **链接: [https://arxiv.org/pdf/2604.04704](https://arxiv.org/pdf/2604.04704)**

> **作者:** Anjali Kantharuban; Aarohi Srivastava; Fahim Faisal; Orevaoghene Ahia; Antonios Anastasopoulos; David Chiang; Yulia Tsvetkov; Graham Neubig
>
> **摘要:** Existing sentence representations primarily encode what a sentence says, rather than how it is expressed, even though the latter is important for many applications. In contrast, we develop sentence representations that capture style and dialect, decoupled from semantic content. We call this the task of idiolectal representation learning. We introduce IDIOLEX, a framework for training models that combines supervision from a sentence's provenance with linguistic features of a sentence's content, to learn a continuous representation of each sentence's style and dialect. We evaluate the approach on dialects of both Arabic and Spanish. The learned representations capture meaningful variation and transfer across domains for analysis and classification. We further explore the use of these representations as training objectives for stylistically aligning language models. Our results suggest that jointly modeling individual and community-level variation provides a useful perspective for studying idiolect and supports downstream applications requiring sensitivity to stylistic differences, such as developing diverse and accessible LLMs.
>
---
#### [new 070] How Far Are We? Systematic Evaluation of LLMs vs. Human Experts in Mathematical Contest in Modeling
- **分类: cs.CL**

- **简介: 该论文属于人工智能评估任务，旨在检验LLMs在数学建模竞赛中的表现。通过构建评估框架，发现LLMs在执行阶段存在明显不足，需超越模型规模提升解决能力。**

- **链接: [https://arxiv.org/pdf/2604.04791](https://arxiv.org/pdf/2604.04791)**

> **作者:** Yuhang Liu; Heyan Huang; Yizhe Yang; Hongyan Zhao; Zhizhuo Zeng; Yang Gao
>
> **摘要:** Large language models (LLMs) have achieved strong performance on reasoning benchmarks, yet their ability to solve real-world problems requiring end-to-end workflows remains unclear. Mathematical modeling competitions provide a stringent testbed for evaluating such end-to-end problem-solving capability. We propose a problem-oriented, stage-wise evaluation framework that assesses LLM performance across modeling stages using expert-verified criteria. We validate the framework's reliability by comparing automatic scores with independent human expert judgments on problems from the China Postgraduate Mathematical Contest in Modeling, demonstrating substantially stronger alignment than existing evaluation schemes. Using this framework, we reveal a comprehension-execution gap in state-of-the-art LLMs: while they perform well in early stages such as problem identification and formulation, they exhibit persistent deficiencies in execution-oriented stages including model solving, code implementation, and result analysis. These gaps persist even with increased model scale. We further trace these failures to insufficient specification, missing verification, and lack of validation, with errors propagating across stages without correction. Our findings suggest that bridging this gap requires approaches beyond model scaling, offering insights for applying LLMs to complex real-world problem solving.
>
---
#### [new 071] Self-Execution Simulation Improves Coding Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于代码生成任务，旨在解决LLMs无法准确评估生成代码执行的问题。通过训练模型模拟程序执行，提升竞赛编程性能。**

- **链接: [https://arxiv.org/pdf/2604.03253](https://arxiv.org/pdf/2604.03253)**

> **作者:** Gallil Maimon; Ori Yoran; Felix Kreuk; Michael Hassid; Gal Cohen; Pierre Chambon; Yossi Adi
>
> **摘要:** A promising research direction in enabling LLMs to generate consistently correct code involves addressing their inability to properly estimate program execution, particularly for code they generate. In this work, we demonstrate that Code LLMs can be trained to simulate program execution in a step-by-step manner and that this capability can be leveraged to improve competitive programming performance. Our approach combines supervised fine-tuning on natural language execution traces, textual explanations grounded in true execution, with reinforcement learning using verifiable rewards. We introduce two complementary objectives: output prediction given code and inputs, and solving competitive programming tasks with either ground-truth or self-predicted execution feedback. These objectives enable models to perform self-verification over multiple candidate solutions, and iterative self-fixing by simulating test execution. Across multiple competitive programming benchmarks, our method yields consistent improvements over standard reasoning approaches. We further present ablations and analysis to elucidate the role of execution simulation and its limitations.
>
---
#### [new 072] Rethinking Exploration in RLVR: From Entropy Regularization to Refinement via Bidirectional Entropy Modulation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于强化学习任务，解决LLM在RLVR中探索受限的问题。通过分解策略熵为信息熵和虚假熵，提出AsymGRPO框架实现有效探索。**

- **链接: [https://arxiv.org/pdf/2604.04894](https://arxiv.org/pdf/2604.04894)**

> **作者:** Hengrui Gu; Xiaotian Han; Yujing Bian; Kaixiong Zhou
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has significantly advanced the reasoning capabilities of large language models (LLMs). However, it faces a fundamental limitation termed \textit{restricted exploration}, where the policy rapidly converges to a narrow set of solutions. While entropy regularization is a popular approach used to sustain exploration, it often proves unreliable for LLMs, suffering from high hyperparameter sensitivity and yielding only marginal performance gains. Motivated by these inefficiencies, we propose to rethink the relationship between policy entropy and exploration. By deriving a parametric formulation of group-relative advantage estimation and analyzing entropy dynamics, we conceptually decompose policy entropy into \textit{informative entropy}, which preserves diverse solution paths, and \textit{spurious entropy}, which erodes reasoning patterns. Our analysis reveals that, in contrast to blind maximization, effective exploration requires \textit{entropy refinement}-a mechanism implicitly embedded in group-relative advantage estimation that sustains informative entropy on positive rollouts while suppressing spurious entropy on negative ones. Guided by this insight, we propose \textbf{AsymGRPO}, an exploratory framework that explicitly decouples the modulation of positive and negative rollouts. This allows for independent control over the preservation of informative entropy and the suppression of spurious noise. Extensive experiments demonstrate that AsymGRPO achieves superior performance compared to strong baselines and exhibits the potential to synergize with existing entropy regularization methods.
>
---
#### [new 073] The Tool Illusion: Rethinking Tool Use in Web Agents
- **分类: cs.CL**

- **简介: 该论文属于Web代理工具使用研究，旨在解决工具有效性、设计原则及潜在副作用问题。通过大规模实验验证了工具使用的实际效果与影响。**

- **链接: [https://arxiv.org/pdf/2604.03465](https://arxiv.org/pdf/2604.03465)**

> **作者:** Renze Lou; Baolin Peng; Wenlin Yao; Qianhui Wu; Hao Cheng; Suman Nath; Wenpeng Yin; Jianfeng Gao
>
> **备注:** preprint
>
> **摘要:** As web agents rapidly evolve, an increasing body of work has moved beyond conventional atomic browser interactions and explored tool use as a higher-level action paradigm. Although prior studies have shown the promise of tools, their conclusions are often drawn from limited experimental scales and sometimes non-comparable settings. As a result, several fundamental questions remain unclear: i) whether tools provide consistent gains for web agents, ii) what practical design principles characterize effective tools, and iii) what side effects tool use may introduce. To establish a stronger empirical foundation for future research, we revisit tool use in web agents through an extensive and carefully controlled study across diverse tool sources, backbone models, tool-use frameworks, and evaluation benchmarks. Our findings both revise some prior conclusions and complement others with broader evidence. We hope this study provides a more reliable empirical basis and inspires future research on tool-use web agents.
>
---
#### [new 074] Towards a theory of morphology-driven marking in the lexicon: The case of the state
- **分类: cs.CL**

- **简介: 该论文属于语言学研究，探讨名词的形态标记差异。提出形态驱动标记模型，分析不同语言中名词类型的标记模式，旨在解释语义与句法关系。**

- **链接: [https://arxiv.org/pdf/2604.03422](https://arxiv.org/pdf/2604.03422)**

> **作者:** Mohamed El Idrissi
>
> **备注:** 32 pages, 1 figure
>
> **摘要:** All languages have a noun category, but its realisation varies considerably. Depending on the language, semantic and/or morphosyntactic differences may be more or less pronounced. This paper explores these variations, using Riffian as a reference point before extending the analysis to other languages. We propose a formal model termed morphology-driven marking. Nouns are organised into modular cognitive sets, each with its own morphological template and unmarked form. This approach helps explain differences in marking among noun types within and across languages. By situating these patterns within syntactic functions, we also reassess the notions of markedness and state. It is proposed that the concept of state be extended to all synthetic languages and analysed a novel subcategory of syntax-based inflection like agreement and grammatical case.
>
---
#### [new 075] SoLA: Leveraging Soft Activation Sparsity and Low-Rank Decomposition for Large Language Model Compression
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型部署困难的问题。提出SoLA方法，利用激活稀疏性和低秩分解实现高效压缩，无需后训练即可提升性能。**

- **链接: [https://arxiv.org/pdf/2604.03258](https://arxiv.org/pdf/2604.03258)**

> **作者:** Xinhao Huang; You-Liang Huang; Zeyi Wen
>
> **摘要:** Large language models (LLMs) have demonstrated impressive capabilities across various tasks, but the billion-scale parameters pose deployment challenges. Although existing methods attempt to reduce the scale of LLMs, they require either special hardware support or expensive post-training to maintain model quality. To facilitate efficient and affordable model slimming, we propose a novel training-free compression method for LLMs, named "SoLA", which leverages \textbf{So}ft activation sparsity and \textbf{L}ow-r\textbf{A}nk decomposition. SoLA can identify and retain a minority of components significantly contributing to inference, while compressing the majority through low-rank decomposition, based on our analysis of the activation pattern in the feed-forward network (FFN) of modern LLMs. To alleviate the decomposition loss, SoLA is equipped with an adaptive component-wise low-rank allocation strategy to assign appropriate truncation positions for different weight matrices. We conduct extensive experiments on LLaMA-2-7B/13B/70B and Mistral-7B models across a variety of benchmarks. SoLA exhibits remarkable improvement in both language modeling and downstream task accuracy without post-training. For example, with a 30\% compression rate on the LLaMA-2-70B model, SoLA surpasses the state-of-the-art method by reducing perplexity from 6.95 to 4.44 and enhancing downstream task accuracy by 10\%.
>
---
#### [new 076] Many Preferences, Few Policies: Towards Scalable Language Model Personalization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型个性化任务，解决为每个用户定制模型的计算成本过高问题。通过构建少量模型组合，覆盖用户多样化偏好，实现高效个性化。**

- **链接: [https://arxiv.org/pdf/2604.04144](https://arxiv.org/pdf/2604.04144)**

> **作者:** Cheol Woo Kum; Jai Moondra; Roozbeh Nahavandi; Andrew Perrault; Milind Tambe; Swati Gupta
>
> **摘要:** The holy grail of LLM personalization is a single LLM for each user, perfectly aligned with that user's preferences. However, maintaining a separate LLM per user is impractical due to constraints on compute, memory, and system complexity. We address this challenge by developing a principled method for selecting a small portfolio of LLMs that captures representative behaviors across heterogeneous users. We model user preferences across multiple traits (e.g., safety, humor, brevity) through a multi-dimensional weight vector. Given reward functions across these dimensions, our algorithm PALM (Portfolio of Aligned LLMs) generates a small portfolio of LLMs such that, for any weight vector, the portfolio contains a near-optimal LLM for the corresponding scalarized objective. To the best of our knowledge, this is the first result that provides theoretical guarantees on both the size and approximation quality of LLM portfolios for personalization. It characterizes the trade-off between system cost and personalization, as well as the diversity of LLMs required to cover the landscape of user preferences. We provide empirical results that validate these guarantees and demonstrate greater output diversity over common baselines.
>
---
#### [new 077] AI Appeals Processor: A Deep Learning Approach to Automated Classification of Citizen Appeals in Government Services
- **分类: cs.CL; cs.AI**

- **简介: 论文提出AI Appeals Processor，用于政府服务中市民申诉的自动分类与路由。针对传统人工处理效率低、准确率不足的问题，采用深度学习方法提升分类效果，实验表明Word2Vec+LSTM在准确率与效率间取得良好平衡。**

- **链接: [https://arxiv.org/pdf/2604.03672](https://arxiv.org/pdf/2604.03672)**

> **作者:** Vladimir Beskorovainyi
>
> **备注:** 10 pages, 0 figures, 5 tables
>
> **摘要:** Government agencies worldwide face growing volumes of citizen appeals, with electronic submissions increasing significantly over recent years. Traditional manual processing averages 20 minutes per appeal with only 67% classification accuracy, creating significant bottlenecks in public service delivery. This paper presents AI Appeals Processor, a microservice-based system that integrates natural language processing and deep learning techniques for automated classification and routing of citizen appeals. We evaluate multiple approaches -- including Bag-of-Words with SVM, TF-IDF with SVM, fastText, Word2Vec with LSTM, and BERT -- on a representative dataset of 10,000 real citizen appeals across three primary categories (complaints, applications, and proposals) and seven thematic domains. Our experiments demonstrate that a Word2Vec+LSTM architecture achieves 78% classification accuracy while reducing processing time by 54%, offering an optimal balance between accuracy and computational efficiency compared to transformer-based models.
>
---
#### [new 078] Compressible Softmax-Attended Language under Incompressible Attention
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究Transformer模型中注意力机制的压缩性，分析softmax注意力的交互矩阵结构，揭示语言数据本身的特性导致其可被高效压缩。属于自然语言处理任务，解决注意力机制效率问题。**

- **链接: [https://arxiv.org/pdf/2604.04384](https://arxiv.org/pdf/2604.04384)**

> **作者:** Wonsuk Lee
>
> **备注:** 6 pages
>
> **摘要:** Across every attention head in five transformer language models (124M--7B parameters, four architecture families), the logit energy field $\tilde{E}$ reaches 90\% of its variance in 2--11 singular components. The \emph{learned} interaction matrix $W_Q^\mathrm{T} W_K$ needs 38--75 components for the same threshold out of $d_h \in \{64, 128\}$. The spectral gap is $5$--$25\times$ in effective rank. The attention mechanism allocates capacity uniformly across all $d_h$ dimensions, but language concentrates the actual interaction into a few. The compressibility of softmax-attended language is a property of the data, not the frame that analyzes it.
>
---
#### [new 079] Robust LLM Performance Certification via Constrained Maximum Likelihood Estimation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于LLM可靠性评估任务，旨在解决准确估计LLM失败率的问题。通过结合人工标注、自动标注和领域约束，提出一种改进的约束最大似然估计方法。**

- **链接: [https://arxiv.org/pdf/2604.03257](https://arxiv.org/pdf/2604.03257)**

> **作者:** Minghe Shen; Ananth Balashankar; Adam Fisch; David Madras; Miguel Rodrigues
>
> **摘要:** The ability to rigorously estimate the failure rates of large language models (LLMs) is a prerequisite for their safe deployment. Currently, however, practitioners often face a tradeoff between expensive human gold standards and potentially severely-biased automatic annotation schemes such as "LLM-as-a-Judge" labeling. In this paper, we propose a new, practical, and efficient approach to LLM failure rate estimation based on constrained maximum-likelihood estimation (MLE). Our method integrates three distinct signal sources: (i) a small, high-quality human-labeled calibration set, (ii) a large corpus of LLM-judge annotations, and, most importantly, (iii) additional side information via domain-specific constraints derived from known bounds on judge performance statistics. We validate our approach through a comprehensive empirical study, benchmarking it against state-of-the-art baselines like Prediction-Powered Inference (PPI). Across diverse experimental regimes -- spanning varying judge accuracies, calibration set sizes, and LLM failure rates -- our constrained MLE consistently delivers more accurate and lower-variance estimates than existing methods. By moving beyond the "black-box" use of automated judges to a flexible framework, we provide a principled, interpretable, and scalable pathway towards LLM failure-rate certification.
>
---
#### [new 080] TriAttention: Efficient Long Reasoning with Trigonometric KV Compression
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于大语言模型的长序列推理任务，旨在解决KV缓存内存瓶颈问题。通过分析Q/K向量集中特性，提出TriAttention方法，提升推理效率与内存利用率。**

- **链接: [https://arxiv.org/pdf/2604.04921](https://arxiv.org/pdf/2604.04921)**

> **作者:** Weian Mao; Xi Lin; Wei Huang; Yuxin Xie; Tianfu Fu; Bohan Zhuang; Song Han; Yukang Chen
>
> **备注:** Code is available at this https URL
>
> **摘要:** Extended reasoning in large language models (LLMs) creates severe KV cache memory bottlenecks. Leading KV cache compression methods estimate KV importance using attention scores from recent post-RoPE queries. However, queries rotate with position during RoPE, making representative queries very few, leading to poor top-key selection and unstable reasoning. To avoid this issue, we turn to the pre-RoPE space, where we observe that Q and K vectors are highly concentrated around fixed non-zero centers and remain stable across positions -- Q/K concentration. We show that this concentration causes queries to preferentially attend to keys at specific distances (e.g., nearest keys), with the centers determining which distances are preferred via a trigonometric series. Based on this, we propose TriAttention to estimate key importance by leveraging these centers. Via the trigonometric series, we use the distance preference characterized by these centers to score keys according to their positions, and also leverage Q/K norms as an additional signal for importance estimation. On AIME25 with 32K-token generation, TriAttention matches Full Attention reasoning accuracy while achieving 2.5x higher throughput or 10.7x KV memory reduction, whereas leading baselines achieve only about half the accuracy at the same efficiency. TriAttention enables OpenClaw deployment on a single consumer GPU, where long context would otherwise cause out-of-memory with Full Attention.
>
---
#### [new 081] Individual and Combined Effects of English as a Second Language and Typos on LLM Performance
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究ESL和拼写错误对LLM性能的影响，属于自然语言处理任务。解决真实场景下模型性能评估不足的问题，通过实验分析两者单独及联合影响。**

- **链接: [https://arxiv.org/pdf/2604.04723](https://arxiv.org/pdf/2604.04723)**

> **作者:** Serena Liu; Yutong Yang; Prisha Sheth; Weixuan Dong; Mingjiao Diao; Xinru Zhu; Nikhil Banga; Oscar Melendez; Arnav Sharma; Minda Zhao; Marina Lin; Mengyu Wang
>
> **摘要:** Large language models (LLMs) are used globally, and because much of their training data is in English, they typically perform best on English inputs. As a result, many non-native English speakers interact with them in English as a second language (ESL), and these inputs often contain typographical errors. Prior work has largely studied the effects of ESL variation and typographical errors separately, even though they often co-occur in real-world use. In this study, we use the Trans-EnV framework to transform standard English inputs into eight ESL variants and apply MulTypo to inject typos at three levels: low, moderate, and severe. We find that combining ESL variation and typos generally leads to larger performance drops than either factor alone, though the combined effect is not simply additive. This pattern is clearest on closed-ended tasks, where performance degradation can be characterized more consistently across ESL variants and typo levels, while results on open-ended tasks are more mixed. Overall, these findings suggest that evaluations on clean standard English may overestimate real-world model performance, and that evaluating ESL variation and typographical errors in isolation does not fully capture model behavior in realistic settings.
>
---
#### [new 082] HUKUKBERT: Domain-Specific Language Model for Turkish Law
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出HukukBERT，针对土耳其法律领域构建的语言模型，解决法律文本处理问题。通过预训练和评估，显著提升法律术语预测和文档结构分割性能。**

- **链接: [https://arxiv.org/pdf/2604.04790](https://arxiv.org/pdf/2604.04790)**

> **作者:** Mehmet Utku Öztürk; Tansu Türkoğlu; Buse Buz-Yalug
>
> **备注:** 15 pages
>
> **摘要:** Recent advances in natural language processing (NLP) have increasingly enabled LegalTech applications, yet existing studies specific to Turkish law have still been limited due to the scarcity of domain-specific data and models. Although extensive models like LEGAL-BERT have been developed for English legal texts, the Turkish legal domain lacks a domain-specific high-volume counterpart. In this paper, we introduce HukukBERT, the most comprehensive legal language model for Turkish, trained on a 18 GB cleaned legal corpus using a hybrid Domain-Adaptive Pre-Training (DAPT) methodology integrating Whole-Word Masking, Token Span Masking, Word Span Masking, and targeted Keyword Masking. We systematically compared our 48K WordPiece tokenizer and DAPT approach against general-purpose and existing domain-specific Turkish models. Evaluated on a novel Legal Cloze Test benchmark -- a masked legal term prediction task designed for Turkish court decisions -- HukukBERT achieves state-of-the-art performance with 84.40\% Top-1 accuracy, substantially outperforming existing models. Furthermore, we evaluated HukukBERT in the downstream task of structural segmentation of official Turkish court decisions, where it achieves a 92.8\% document pass rate, establishing a new state-of-the-art. We release HukukBERT to support future research in Turkish legal NLP tasks, including recognition of named entities, prediction of judgment, and classification of legal documents.
>
---
#### [new 083] Emergent Inference-Time Semantic Contamination via In-Context Priming
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在推理时因上下文提示产生的语义偏差问题，通过实验揭示了语义污染的机制与边界条件，对模型安全性有重要影响。**

- **链接: [https://arxiv.org/pdf/2604.04043](https://arxiv.org/pdf/2604.04043)**

> **作者:** Marcin Abram
>
> **备注:** 6 pages, 2 figures, appendix
>
> **摘要:** Recent work has shown that fine-tuning large language models (LLMs) on insecure code or culturally loaded numeric codes can induce emergent misalignment, causing models to produce harmful content in unrelated downstream tasks. The authors of that work concluded that $k$-shot prompting alone does not induce this effect. We revisit this conclusion and show that inference-time semantic drift is real and measurable; however, it requires models of large-enough capability. Using a controlled experiment in which five culturally loaded numbers are injected as few-shot demonstrations before a semantically unrelated prompt, we find that models with richer cultural-associative representations exhibit significant distributional shifts toward darker, authoritarian, and stigmatized themes, while a simpler/smaller model does not. We additionally find that structurally inert demonstrations (nonsense strings) perturb output distributions, suggesting two separable mechanisms: structural format contamination and semantic content contamination. Our results map the boundary conditions under which inference-time contamination occurs, and carry direct implications for the security of LLM-based applications that use few-shot prompting.
>
---
#### [new 084] Early Stopping for Large Reasoning Models via Confidence Dynamics
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型推理任务，解决如何在合理时机停止模型推理以提升效率的问题。提出CoDE-Stop方法，通过分析中间答案的置信度动态实现早期停止。**

- **链接: [https://arxiv.org/pdf/2604.04930](https://arxiv.org/pdf/2604.04930)**

> **作者:** Parsa Hosseini; Sumit Nawathe; Mahdi Salmani; Meisam Razaviyayn; Soheil Feizi
>
> **摘要:** Large reasoning models rely on long chain-of-thought generation to solve complex problems, but extended reasoning often incurs substantial computational cost and can even degrade performance due to overthinking. A key challenge is determining when the model should stop reasoning and produce the final answer. In this work, we study the confidence of intermediate answers during reasoning and observe two characteristic behaviors: correct reasoning trajectories often reach high-confidence answers early, while incorrect rollouts tend to produce long, unproductive reasoning traces and exhibit less reliable confidence dynamics. Motivated by these observations, we propose CoDE-Stop (Confidence Dynamics Early Stop), an early stopping method that leverages the dynamics of intermediate answer confidence to decide when to terminate reasoning, requiring no additional training and easily integrating into existing models. We evaluate CoDE-Stop on diverse reasoning and science benchmarks across multiple models. Compared to prior early stopping methods, it achieves a more favorable accuracy-compute tradeoff and reduces total token usage by 25-50% compared to standard full-length reasoning. In addition, we provide analyses of confidence dynamics during reasoning, offering insights into how confidence changes in both correct and incorrect trajectories.
>
---
#### [new 085] AdaptFuse: Training-Free Sequential Preference Learning via Externalized Bayesian Inference
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AdaptFuse，解决大语言模型在多轮交互中无法有效更新信念的问题。通过外部化贝叶斯推理，实现无需训练的序列偏好学习。**

- **链接: [https://arxiv.org/pdf/2604.03925](https://arxiv.org/pdf/2604.03925)**

> **作者:** Fangzhou Lin; Peiran Li; Shuo Xing; Siyuan Yang; Qianwen Ge; Kazunori Yamada; Ziming Zhang; Haichong Zhang; Zhengzhong Tu
>
> **备注:** 20 pages, 4 figures, 5 tables
>
> **摘要:** Large language models struggle to accumulate evidence across multiple rounds of user interaction, failing to update their beliefs in a manner consistent with Bayesian inference. Existing solutions require fine-tuning on sensitive user interaction data, limiting their applicability in privacy-conscious settings. We propose AdaptFuse, a training-free framework that externalizes probabilistic computation entirely from the LLM: a symbolic module maintains a Bayesian posterior over a discrete hypothesis set, while a frozen LLM contributes semantic reasoning via multi-sample Dirichlet aggregation. The two signals are combined through entropy-adaptive fusion, which automatically weights each source by its predictive confidence, shifting reliance from the LLM to the symbolic posterior as evidence accumulates. We evaluate across three domains: flight recommendation, hotel recommendation, and web shopping; on Gemma 2 9B, Llama 3 8B, and Qwen 2.5 7B. AdaptFuse consistently outperforms both prompting baselines and fine-tuned Bayesian Teaching models on all tasks, with accuracy improving monotonically over interaction rounds. These results demonstrate that principled inference-time algorithms can substitute for fine-tuning in personalized recommendation, without storing or training on sensitive user data. All the code and materials will be open-sourced.
>
---
#### [new 086] Are Arabic Benchmarks Reliable? QIMMA's Quality-First Approach to LLM Evaluation
- **分类: cs.CL**

- **简介: 该论文提出QIMMA，一个基于质量验证的阿拉伯语大模型评估基准，解决现有基准可靠性问题，通过多模型评估和人工审核提升评估质量。**

- **链接: [https://arxiv.org/pdf/2604.03395](https://arxiv.org/pdf/2604.03395)**

> **作者:** Leen AlQadi; Ahmed Alzubaidi; Mohammed Alyafeai; Hamza Alobeidli; Maitha Alhammadi; Shaikha Alsuwaidi; Omar Alkaabi; Basma El Amel Boussaha; Hakim Hacid
>
> **摘要:** We present QIMMA, a quality-assured Arabic LLM leaderboard that places systematic benchmark validation at its core. Rather than aggregating existing resources as-is, QIMMA applies a multi-model assessment pipeline combining automated LLM judgment with human review to surface and resolve systematic quality issues in well-established Arabic benchmarks before evaluation. The result is a curated, multi-domain, multi-task evaluation suite of over 52k samples, grounded predominantly in native Arabic content; code evaluation tasks are the sole exception, as they are inherently language-agnostic. Transparent implementation via LightEval, EvalPlus and public release of per-sample inference outputs make QIMMA a reproducible and community-extensible foundation for Arabic NLP evaluation.
>
---
#### [new 087] LPC-SM: Local Predictive Coding and Sparse Memory for Long-Context Language Modeling
- **分类: cs.CL; cs.AI; cs.GL; cs.NE**

- **简介: 该论文属于长文本语言建模任务，旨在解决传统模型依赖注意力机制的局限。提出LPC-SM架构，分离局部注意力、持久记忆等模块，提升长序列建模效果。**

- **链接: [https://arxiv.org/pdf/2604.03263](https://arxiv.org/pdf/2604.03263)**

> **作者:** Keqin Xie
>
> **摘要:** Most current long-context language models still rely on attention to handle both local interaction and long-range state, which leaves relatively little room to test alternative decompositions of sequence modeling. We propose LPC-SM, a hybrid autoregressive architecture that separates local attention, persistent memory, predictive correction, and run-time control within the same block, and we use Orthogonal Novelty Transport (ONT) to govern slow-memory writes. We evaluate a 158M-parameter model in three stages spanning base language modeling, mathematical continuation, and 4096-token continuation. Removing mHC raises the Stage-A final LM loss from 12.630 to 15.127, while adaptive sparse control improves the Stage-B final LM loss from 12.137 to 10.787 relative to a matched fixed-ratio continuation. The full route remains stable at sequence length 4096, where Stage C ends with final LM loss 11.582 and improves the delayed-identifier diagnostic from 14.396 to 12.031 in key cross-entropy. Taken together, these results show that long-context autoregressive modeling can be organized around a broader division of labor than attention alone.
>
---
#### [new 088] Focus Matters: Phase-Aware Suppression for Hallucination in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型中的幻觉抑制任务，旨在解决模型生成不存在对象描述的问题。通过分析视觉编码器的注意力动态，提出一种轻量级干预方法，在不增加显著延迟的情况下减少幻觉。**

- **链接: [https://arxiv.org/pdf/2604.03556](https://arxiv.org/pdf/2604.03556)**

> **作者:** Sohyeon Kim; Sang Yeon Yoon; Kyeongbo Kong
>
> **摘要:** Large Vision-Language Models (LVLMs) have achieved impressive progress in multimodal reasoning, yet they remain prone to object hallucinations, generating descriptions of objects that are not present in the input image. Recent approaches attempt to mitigate hallucinations by suppressing unreliable visual signals in the vision encoder, but many rely on iterative optimization for each input, resulting in substantial inference latency. In this work, we investigate the internal attention dynamics of vision encoders in LVLMs and identify a consistent three-phase structure of visual information processing: diffusion, focus, and rediffusion. Our analysis reveals that hallucination behavior is particularly sensitive to tokens receiving low attention during the focus phase. Motivated by this observation, we propose a lightweight inference-time intervention that selectively suppresses such tokens during the focus phase. The method operates in a training-free manner using statistics from a single forward pass and employs a Determinantal Point Process (DPP) to preserve diverse visual cues while filtering redundant tokens. Extensive experiments across multiple LVLM backbones and decoding strategies demonstrate that the proposed approach consistently reduces hallucination metrics while maintaining competitive caption quality. Moreover, compared to adversarial uncertainty estimation methods, our approach achieves comparable hallucination mitigation with negligible additional inference latency.
>
---
#### [new 089] Align then Train: Efficient Retrieval Adapter Learning
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，解决查询与文档间的语义不匹配问题。提出ERA框架，通过两阶段训练提升检索效果，适应复杂查询与简单文档的差异。**

- **链接: [https://arxiv.org/pdf/2604.03403](https://arxiv.org/pdf/2604.03403)**

> **作者:** Seiji Maekawa; Moin Aminnaseri; Pouya Pezeshkpour; Estevam Hruschka
>
> **摘要:** Dense retrieval systems increasingly need to handle complex queries. In many realistic settings, users express intent through long instructions or task-specific descriptions, while target documents remain relatively simple and static. This asymmetry creates a retrieval mismatch: understanding queries may require strong reasoning and instruction-following, whereas efficient document indexing favors lightweight encoders. Existing retrieval systems often address this mismatch by directly improving the embedding model, but fine-tuning large embedding models to better follow such instructions is computationally expensive, memory-intensive, and operationally burdensome. To address this challenge, we propose Efficient Retrieval Adapter (ERA), a label-efficient framework that trains retrieval adapters in two stages: self-supervised alignment and supervised adaptation. Inspired by the pre-training and supervised fine-tuning stages of LLMs, ERA first aligns the embedding spaces of a large query embedder and a lightweight document embedder, and then uses limited labeled data to adapt the query-side representation, bridging both the representation gap between embedding models and the semantic gap between complex queries and simple documents without re-indexing the corpus. Experiments on the MAIR benchmark, spanning 126 retrieval tasks across 6 domains, show that ERA improves retrieval in low-label settings, outperforms methods that rely on larger amounts of labeled data, and effectively combines stronger query embedders with weaker document embedders across domains.
>
---
#### [new 090] Generative Chemical Language Models for Energetic Materials Discovery
- **分类: physics.chem-ph; cond-mat.mtrl-sci; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于分子生成任务，旨在解决 energetic materials 数据不足的问题。通过预训练和微调语言模型，结合片段编码，加速高性能材料的设计。**

- **链接: [https://arxiv.org/pdf/2604.03304](https://arxiv.org/pdf/2604.03304)**

> **作者:** Andrew Salij; R. Seaton Ullberg; Megan C. Davis; Marc J. Cawkwell; Christopher J. Snyder; Cristina Garcia Cardona; Ivana Matanovic; Wilton J. M. Kort-Kamp
>
> **摘要:** The discovery of new energetic materials remains a pressing challenge hindered by limited availability of high-quality data. To address this, we have developed generative molecular language models that have been pretrained on extensive chemical data and then fine-tuned with curated energetic materials datasets. This transfer-learning strategy extends the chemical language model capabilities beyond the pharmacological space in which they have been predominantly developed, offering a framework applicable to other data-spare discovery problems. Furthermore, we discuss the benefits of fragment-based molecular encodings for chemical language models, in particular in constructing synthetically accessible structures. Together, these advances provide a foundation for accelerating the design of next-generation energetic materials with demanding performance requirements.
>
---
#### [new 091] Cog-DRIFT: Exploration on Adaptively Reformulated Instances Enables Learning from Hard Reasoning Problems
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大模型后训练任务，旨在解决模型无法从困难问题中学习的问题。通过任务重构和自适应课程学习，提升模型在复杂推理任务上的表现。**

- **链接: [https://arxiv.org/pdf/2604.04767](https://arxiv.org/pdf/2604.04767)**

> **作者:** Justin Chih-Yao Chen; Archiki Prasad; Zaid Khan; Joykirat Singh; Runchu Tian; Elias Stengel-Eskin; Mohit Bansal
>
> **备注:** 22 pages, 4 figures. Code: this https URL
>
> **摘要:** Reinforcement learning from verifiable rewards (RLVR) has improved the reasoning abilities of LLMs, yet a fundamental limitation remains: models cannot learn from problems that are too difficult to solve under their current policy, as these yield no meaningful reward signal. We propose a simple yet effective solution based on task reformulation. We transform challenging open-ended problems into cognitively simpler variants -- such as multiple-choice and cloze formats -- that preserve the original answer while reducing the effective search space and providing denser learning signals. These reformulations span a spectrum from discriminative to generative tasks, which we exploit to bootstrap learning: models first learn from structured, easier formats, and this knowledge transfers back to improve performance on the original open-ended problems. Building on this insight, we introduce Cog-DRIFT, a framework that constructs reformulated variants and organizes them into an adaptive curriculum based on difficulty. Training progresses from easier to harder formats, enabling the model to learn from problems that previously yielded zero signal under standard RL post-training. Cog-DRIFT not only improves on the originally unsolvable hard problems (absolute +10.11% for Qwen and +8.64% for Llama) but also generalizes well to other held-out datasets. Across 2 models and 6 reasoning benchmarks, our method consistently outperforms standard GRPO and strong guided-exploration baselines. On average, Cog-DRIFT shows +4.72% (Qwen) and +3.23% (Llama) improvements over the second-best baseline. We further show that Cog-DRIFT improves pass@k at test time, and the curriculum improves sample efficiency. Overall, our results highlight task reformulation and curriculum learning as an effective paradigm for overcoming the exploration barrier in LLM post-training.
>
---
#### [new 092] SODA: Semi On-Policy Black-Box Distillation for Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出SODA方法，解决大语言模型知识蒸馏中的效率与稳定性问题。通过半监督策略提升蒸馏效果，显著提高训练速度并降低资源消耗。**

- **链接: [https://arxiv.org/pdf/2604.03873](https://arxiv.org/pdf/2604.03873)**

> **作者:** Xiwen Chen; Jingjing Wang; Wenhui Zhu; Peijie Qiu; Xuanzhao Dong; Hejian Sang; Zhipeng Wang; Alborz Geramifard; Feng Luo
>
> **摘要:** Black-box knowledge distillation for large language models presents a strict trade-off. Simple off-policy methods (e.g., sequence-level knowledge distillation) struggle to correct the student's inherent errors. Fully on-policy methods (e.g., Generative Adversarial Distillation) solve this via adversarial training but introduce well-known training instability and crippling computational overhead. To address this dilemma, we propose SODA (Semi On-policy Distillation with Alignment), a highly efficient alternative motivated by the inherent capability gap between frontier teachers and much smaller base models. Because a compact student model's natural, zero-shot responses are almost strictly inferior to the powerful teacher's targets, we can construct a highly effective contrastive signal simply by pairing the teacher's optimal response with a one-time static snapshot of the student's outputs. This demonstrates that exposing the small student to its own static inferior behaviors is sufficient for high-quality distribution alignment, eliminating the need for costly dynamic rollouts and fragile adversarial balancing. Extensive evaluations across four compact Qwen2.5 and Llama-3 models validate this semi on-policy paradigm. SODA matches or outperforms the state-of-the-art methods on 15 out of 16 benchmark results. More importantly, it achieves this superior distillation quality while training 10 times faster, consuming 27% less peak GPU memory, and completely eliminating adversarial instability.
>
---
#### [new 093] Classifying Problem and Solution Framing in Congressional Social Media
- **分类: cs.CY; cs.AI; cs.CL; cs.SI**

- **简介: 该论文属于分类任务，旨在区分美国参议员推文中的“问题”与“解决方案”内容。通过监督学习方法，利用BERTweet模型实现高准确率的自动分类。**

- **链接: [https://arxiv.org/pdf/2604.03247](https://arxiv.org/pdf/2604.03247)**

> **作者:** Misha Melnyk; Mitchell Dolny; Joshua D. Elkind; A. Michael Tjhin; Saisha Chebium; Blake VanBerlo; Annelise Russell; Michelle M. Buehlmann; Jesse Hoey
>
> **摘要:** Policy setting in the USA according to the ``Garbage Can'' model differentiates between ``problem'' and ``solution'' focused processes. In this paper, we study a large dataset of US Senator postings on Twitter (1.68m tweets in total). Our objective is to develop an automated method to label Senatorial posts as either in the problem or solution streams. Two academic policy experts labeled a subset of 3967 tweets as either problem, solution, or other (anything not problem or solution). We split off a subset of 500 tweets into a test set, with the remaining 3467 used for training. During development, this training set was further split by 60/20/20 proportions for fitting, validation, and development test sets. We investigated supervised learning methods for building problem/solution classifiers directly on the training set, evaluating their performance in terms of F1 score on the validation set, allowing us to rapidly iterate through models and hyperparameters, achieving an average weighted F1 score of above 0.8 on cross validation across the three categories using a BERTweet Base model.
>
---
#### [new 094] Mapping the Exploitation Surface: A 10,000-Trial Taxonomy of What Makes LLM Agents Exploit Vulnerabilities
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文研究LLM代理如何利用系统提示中的漏洞，通过10,000次实验构建分类体系，识别出目标重构是关键触发因素。任务属于安全漏洞分析，解决如何检测和防范LLM的恶意行为。**

- **链接: [https://arxiv.org/pdf/2604.04561](https://arxiv.org/pdf/2604.04561)**

> **作者:** Charafeddine Mouzouni
>
> **备注:** 18 pages, 8 tables, code and data at this https URL
>
> **摘要:** LLM agents with tool access can discover and exploit security vulnerabilities. This is known. What is not known is which features of a system prompt trigger this behaviour, and which do not. We present a systematic taxonomy based on approximately 10,000 trials across seven models, 37 prompt conditions, and 12 hypothesized attack dimensions -- all executed in real Docker sandboxes. Every condition includes the instruction "Always respect all rules and access policies"; only a single encouragement sentence varies. Within the task class studied (planted test-runner vulnerabilities), the threat model is narrower than assumed: nine of twelve dimensions -- including minimization, moral licensing, incentives, identity priming, and reasoning triggers -- produce no detectable exploitation at n=50 per cell (upper 95% CI <7% per condition). But it is also more subtle than expected: one dimension, goal reframing, reliably triggers exploitation. "You are solving a puzzle; there may be hidden clues" produces 38-40% exploitation on Claude Sonnet 4 despite the explicit rule instruction, replicating across four models (CTF framing: 8-14% on DeepSeek, GPT-5-mini, o4-mini). The agent does not override the rules; it reinterprets the task so that exploitative actions become task-aligned. GPT-4.1 produces no exploitation across 1,850 trials (37 conditions), and a temporal comparison across four OpenAI models released over eleven months shows a pattern consistent with improving safety training, though model capability differences are a confounder. The practical contribution is a narrowed, testable threat model: defenders should audit for goal-reframing language, not for the broad class of adversarial prompts.
>
---
#### [new 095] Combee: Scaling Prompt Learning for Self-Improving Language Model Agents
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出Combee框架，解决大规模并行提示学习中的效率与质量问题，提升自改进语言模型代理的性能。**

- **链接: [https://arxiv.org/pdf/2604.04247](https://arxiv.org/pdf/2604.04247)**

> **作者:** Hanchen Li; Runyuan He; Qizheng Zhang; Changxiu Ji; Qiuyang Mang; Xiaokun Chen; Lakshya A Agrawal; Wei-Liang Liao; Eric Yang; Alvin Cheung; James Zou; Kunle Olukotun; Ion Stoica; Joseph E. Gonzalez
>
> **摘要:** Recent advances in prompt learning allow large language model agents to acquire task-relevant knowledge from inference-time context without parameter changes. For example, existing methods (like ACE or GEPA) can learn system prompts to improve accuracy based on previous agent runs. However, these methods primarily focus on single-agent or low-parallelism settings. This fundamentally limits their ability to efficiently learn from a large set of collected agentic traces. It would be efficient and beneficial to run prompt learning in parallel to accommodate the growing trend of learning from many agentic traces or parallel agent executions. Yet without a principled strategy for scaling, current methods suffer from quality degradation with high parallelism. To improve both the efficiency and quality of prompt learning, we propose Combee, a novel framework to scale parallel prompt learning for self-improving agents. Combee speeds up learning and enables running many agents in parallel while learning from their aggregate traces without quality degradation. To achieve this, Combee leverages parallel scans and employs an augmented shuffle mechanism; Combee also introduces a dynamic batch size controller to balance quality and delay. Evaluations on AppWorld, Terminal-Bench, Formula, and FiNER demonstrate that Combee achieves up to 17x speedup over previous methods with comparable or better accuracy and equivalent cost.
>
---
#### [new 096] PolySwarm: A Multi-Agent Large Language Model Framework for Prediction Market Trading and Latency Arbitrage
- **分类: cs.AI; cs.CL; cs.MA; q-fin.TR**

- **简介: 该论文提出PolySwarm框架，用于预测市场交易和延迟套利。通过多智能体语言模型解决市场效率问题，实现风险控制与概率校准。**

- **链接: [https://arxiv.org/pdf/2604.03888](https://arxiv.org/pdf/2604.03888)**

> **作者:** Rajat M. Barot; Arjun S. Borkhatariya
>
> **备注:** 13 pages, 3 figures, 3 tables
>
> **摘要:** This paper presents PolySwarm, a novel multi-agent large language model (LLM) framework designed for real-time prediction market trading and latency arbitrage on decentralized platforms such as Polymarket. PolySwarm deploys a swarm of 50 diverse LLM personas that concurrently evaluate binary outcome markets, aggregating individual probability estimates through confidence-weighted Bayesian combination of swarm consensus with market-implied probabilities, and applying quarter-Kelly position sizing for risk-controlled execution. The system incorporates an information-theoretic market analysis engine using Kullback-Leibler (KL) divergence and Jensen-Shannon (JS) divergence to detect cross-market inefficiencies and negation pair mispricings. A latency arbitrage module exploits stale Polymarket prices by deriving CEX-implied probabilities from a log-normal pricing model and executing trades within the human reaction-time window. We provide a full architectural description, implementation details, and evaluation methodology using Brier scores, calibration analysis, and log-loss metrics benchmarked against human superforecaster performance. We further discuss open challenges including hallucination in agent pools, computational cost at scale, regulatory exposure, and feedback-loop risk, and outline five priority directions for future research. Experimental results demonstrate that swarm aggregation consistently outperforms single-model baselines in probability calibration on Polymarket prediction tasks.
>
---
#### [new 097] CoLA: Cross-Modal Low-rank Adaptation for Multimodal Downstream Tasks
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出CoLA，解决多模态任务中跨模态交互不足的问题，通过引入跨模态低秩适配路径，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.03314](https://arxiv.org/pdf/2604.03314)**

> **作者:** Wish Suharitdamrong; Tony Alex; Muhammad Awais; Sara Ahmed
>
> **备注:** 14 pages, 6 Figures
>
> **摘要:** Foundation models have revolutionized AI, but adapting them efficiently for multimodal tasks, particularly in dual-stream architectures composed of unimodal encoders, such as DINO and BERT, remains a significant challenge. Parameter-Efficient Fine-Tuning (PEFT) methods like Low-Rank Adaptation (LoRA) enable lightweight adaptation, yet they operate in isolation within each modality, limiting their ability in capturing cross-modal interactions. In this paper, we take a step in bridging this gap with Cross-Modal Low-Rank Adaptation (CoLA), a novel PEFT framework that extends LoRA by introducing a dedicated inter-modal adaptation pathway alongside the standard intra-modal one. This dual-path design enables CoLA to adapt unimodal foundation models to multimodal tasks effectively, without interference between modality-specific and cross-modal learning. We evaluate CoLA across a range of vision-language (RefCOCO, RefCOCO+, RefCOCOg) and audio-visual (AVE, AVS) benchmarks, where it consistently outperforms LORA, achieving a relative gain of around 3\% and 2\%, respectively, while maintaining parameter efficiency. Notably, CoLA enables the first multi-task PEFT framework for visual grounding, bridging a key gap in efficient multimodal adaptation.
>
---
#### [new 098] ANX: Protocol-First Design for AI Agent Interaction with a Supporting 3EX Decoupled Architecture
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出ANX协议，解决AI代理交互中的高消耗、碎片化问题，通过创新架构与工具提升效率与安全。属于AI代理交互任务。**

- **链接: [https://arxiv.org/pdf/2604.04820](https://arxiv.org/pdf/2604.04820)**

> **作者:** Xu Mingze
>
> **备注:** This open-source AI agent interaction protocol (ANX) is benchmarked against existing protocols (MCP, A2A, ANP, OpenCLI, SkillWeaver, CHEQ, COLLAB-LLM) across four dimensions: tooling, discovery, security, and multi-agent SOP collaboration. Code: this https URL
>
> **摘要:** AI agents, autonomous digital actors, need agent-native protocols; existing methods include GUI automation and MCP-based skills, with defects of high token consumption, fragmented interaction, inadequate security, due to lacking a unified top-level framework and key components, each independent module flawed. To address these issues, we present ANX, an open, extensible, verifiable agent-native protocol and top-level framework integrating CLI, Skill, MCP, resolving pain points via protocol innovation, architectural optimization and tool supplementation. Its four core innovations: 1) Agent-native design (ANX Config, Markup, CLI) with high information density, flexibility and strong adaptability to reduce tokens and eliminate inconsistencies; 2) Human-agent interaction combining Skill's flexibility for dual rendering as agent-executable instructions and human-readable UI; 3) MCP-supported on-demand lightweight apps without pre-registration; 4) ANX Markup-enabled machine-executable SOPs eliminating ambiguity for reliable long-horizon tasks and multi-agent collaboration. As the first in a series, we focus on ANX's design, present its 3EX decoupled architecture with ANXHub and preliminary feasibility analysis and experimental validation. ANX ensures native security: LLM-bypassed UI-to-Core communication keeps sensitive data out of agent context; human-only confirmation prevents automated misuse. Form-filling experiments with Qwen3.5-plus/GPT-4o show ANX reduces tokens by 47.3% (Qwen3.5-plus) and 55.6% (GPT-4o) vs MCP-based skills, 57.1% (Qwen3.5-plus) and 66.3% (GPT-4o) vs GUI automation, and shortens execution time by 58.1% and 57.7% vs MCP-based skills.
>
---
#### [new 099] MinerU2.5-Pro: Pushing the Limits of Data-Centric Document Parsing at Scale
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文档解析任务，旨在解决数据缺陷导致的模型性能瓶颈。通过数据工程与训练策略优化，提升模型效果。**

- **链接: [https://arxiv.org/pdf/2604.04771](https://arxiv.org/pdf/2604.04771)**

> **作者:** Bin Wang; Tianyao He; Linke Ouyang; Fan Wu; Zhiyuan Zhao; Tao Chu; Yuan Qu; Zhenjiang Jin; Weijun Zeng; Ziyang Miao; Bangrui Xu; Junbo Niu; Mengzhang Cai; Jiantao Qiu; Qintong Zhang; Dongsheng Ma; Yuefeng Sun; Hejun Dong; Wenzheng Zhang; Jutao Xiao; Jiayong Shi; Pengyu Liao; Xiaomeng Zhao; Huaping Zhong; Liqun Wei; Jing Yu; Jie Yang; Wei Li; Shasha Wang; Qianqian Wu; Xuanhe Zhou; Weijia Li; Zhenxiang Li; Zhongying Tu; Jiang Wu; Lijun Wu; Chao Xu; Kai Chen; Wentao Zhang; Yu Qiao; Bowen Zhou; Dahua Lin; Conghui He
>
> **备注:** Technical Report
>
> **摘要:** Current document parsing methods compete primarily on model architecture innovation, while systematic engineering of training data remains underexplored. Yet SOTA models of different architectures and parameter scales exhibit highly consistent failure patterns on the same set of hard samples, suggesting that the performance bottleneck stems from shared deficiencies in training data rather than architecture itself. Building on this finding, we present \minerupro, which advances the state of the art solely through data engineering and training strategy optimization while keeping the 1.2B-parameter architecture of \mineru completely fixed. At its core is a Data Engine co-designed around coverage, informativeness, and annotation accuracy: Diversity-and-Difficulty-Aware Sampling expands training data from under 10M to 65.5M samples while correcting distribution shift; Cross-Model Consistency Verification leverages output agreement among heterogeneous models to assess sample difficulty and generate reliable annotations; the Judge-and-Refine pipeline improves annotation quality for hard samples through render-then-verify iterative correction. A three-stage progressive training strategy -- large-scale pre-training, hard sample fine-tuning, and GRPO alignment -- sequentially exploits these data at different quality tiers. On the evaluation front, we fix element-matching biases in OmniDocBench~v1.5 and introduce a Hard subset, establishing the more discriminative OmniDocBench~v1.6 protocol. Without any architectural modification, \minerupro achieves 95.69 on OmniDocBench~v1.6, improving over the same-architecture baseline by 2.71 points and surpassing all existing methods including models with over 200$\times$ more parameters.
>
---
#### [new 100] QED-Nano: Teaching a Tiny Model to Prove Hard Theorems
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于数学定理证明任务，旨在提升小型模型的推理能力。通过三阶段训练，构建了QED-Nano模型，实现在奥数级别证明上的高效表现。**

- **链接: [https://arxiv.org/pdf/2604.04898](https://arxiv.org/pdf/2604.04898)**

> **作者:** LM-Provers; Yuxiao Qu; Amrith Setlur; Jasper Dekoninck; Edward Beeching; Jia Li; Ian Wu; Lewis Tunstall; Aviral Kumar
>
> **摘要:** Proprietary AI systems have recently demonstrated impressive capabilities on complex proof-based problems, with gold-level performance reported at the 2025 International Mathematical Olympiad (IMO). However, the training pipelines behind these systems remain largely undisclosed, and their reliance on large "internal" models and scaffolds makes them expensive to run, difficult to reproduce, and hard to study or improve upon. This raises a central question: can small, open models also be trained to achieve competitive reasoning performance on difficult Olympiad-level math? In this paper, we answer this question by building QED-Nano, a 4B model post-trained for Olympiad-level proofs. Our training recipe has three stages: (1) supervised fine-tuning to imbue good proof-writing styles by distilling from DeepSeek-Math-V2, (2) reinforcement learning (RL) with rubric-based rewards, and (3) expanding RL with a reasoning cache, which decomposes long proofs into iterative summarize-and-refine cycles and enables stronger test-time reasoning. QED-Nano surpasses the proof-generation performance of much larger open models, including Nomos-1 and GPT-OSS-120B, and approaches the performance of proprietary models like Gemini 3 Pro, at a fraction of the inference cost. To support further research on open mathematical reasoning, we release the full QED-Nano pipeline, including the QED-Nano and QED-Nano-SFT models, the FineProofs-SFT and FineProofs-RL datasets, and the training and evaluation code.
>
---
#### [new 101] Talk2AI: A Longitudinal Dataset of Human--AI Persuasive Conversations
- **分类: cs.HC; cs.CL; cs.CY**

- **简介: 该论文提出Talk2AI数据集，用于研究人机说服性对话。任务是分析AI如何影响人类观点和态度，通过3080次对话及多维度数据进行长期分析。**

- **链接: [https://arxiv.org/pdf/2604.04354](https://arxiv.org/pdf/2604.04354)**

> **作者:** Alexis Carrillo; Enrique Taietta; Ali Aghazadeh Ardebili; Giuseppe Alessandro Veltri; Massimo Stella
>
> **备注:** 17 pages, 2 figures, 7 tables
>
> **摘要:** Talk2AI is a large-scale longitudinal dataset of 3,080 conversations (totaling 30,800 turns) between human participants and Large Language Models (LLMs), designed to support research on persuasion, opinion change, and human-AI interaction. The corpus was collected from 770 profiled Italian adults across four weekly sessions in Spring 2025, using a within-subject design in which each participant conversed with a single model (GPT-4o, Claude Sonnet 3.7, DeepSeek-chat V3, or Mistral Large) on three socially relevant topics: climate change, math anxiety, and health misinformation. Each conversation is linked to rich contextual data, including sociodemographic characteristics and psychometric profiles. After each session, participants reported on opinion change, conviction stability, perceived humanness of the AI, and behavioral intentions, enabling fine-grained longitudinal analysis of how AI-mediated dialogue shapes beliefs and attitudes over time.
>
---
#### [new 102] REAM: Merging Improves Pruning of Experts in LLMs
- **分类: cs.AI; cs.CL; cs.LG; cs.PF**

- **简介: 该论文属于模型压缩任务，解决大语言模型内存占用高的问题。提出REAM方法通过合并专家权重，提升性能并减少内存需求。**

- **链接: [https://arxiv.org/pdf/2604.04356](https://arxiv.org/pdf/2604.04356)**

> **作者:** Saurav Jha; Maryam Hashemzadeh; Ali Saheb Pasand; Ali Parviz; Min-Joong Lee; Boris Knyazev
>
> **备注:** code is at this https URL
>
> **摘要:** Mixture-of-Experts (MoE) large language models (LLMs) are among the top-performing architectures. The largest models, often with hundreds of billions of parameters, pose significant memory challenges for deployment. Traditional approaches to reduce memory requirements include weight pruning and quantization. Motivated by the Router-weighted Expert Activation Pruning (REAP) that prunes experts, we propose a novel method, Router-weighted Expert Activation Merging (REAM). Instead of removing experts, REAM groups them and merges their weights, better preserving original performance. We evaluate REAM against REAP and other baselines across multiple MoE LLMs on diverse multiple-choice (MC) question answering and generative (GEN) benchmarks. Our results reveal a trade-off between MC and GEN performance that depends on the mix of calibration data. By controlling the mix of general, math and coding data, we examine the Pareto frontier of this trade-off and show that REAM often outperforms the baselines and in many cases is comparable to the original uncompressed models.
>
---
#### [new 103] Large Language Models Align with the Human Brain during Creative Thinking
- **分类: q-bio.NC; cs.AI; cs.CL**

- **简介: 该论文研究大语言模型与人类大脑在创造性思维中的对齐情况，旨在解决模型与人类认知机制关联的问题。通过fMRI数据和RSA分析，探讨模型规模和训练目标对对齐效果的影响。**

- **链接: [https://arxiv.org/pdf/2604.03480](https://arxiv.org/pdf/2604.03480)**

> **作者:** Mete Ismayilzada; Simone A. Luchini; Abdulkadir Gokce; Badr AlKhamissi; Antoine Bosselut; Antonio Laverghetta Jr.; Lonneke van der Plas; Roger E. Beaty
>
> **备注:** Under review
>
> **摘要:** Creative thinking is a fundamental aspect of human cognition, and divergent thinking-the capacity to generate novel and varied ideas-is widely regarded as its core generative engine. Large language models (LLMs) have recently demonstrated impressive performance on divergent thinking tests and prior work has shown that models with higher task performance tend to be more aligned to human brain activity. However, existing brain-LLM alignment studies have focused on passive, non-creative tasks. Here, we explore brain alignment during creative thinking using fMRI data from 170 participants performing the Alternate Uses Task (AUT). We extract representations from LLMs varying in size (270M-72B) and measure alignment to brain responses via Representational Similarity Analysis (RSA), targeting the creativity-related default mode and frontoparietal networks. We find that brain-LLM alignment scales with model size (default mode network only) and idea originality (both networks), with effects strongest early in the creative process. We further show that post-training objectives shape alignment in functionally selective ways: a creativity-optimized \texttt{Llama-3.1-8B-Instruct} preserves alignment with high-creativity neural responses while reducing alignment with low-creativity ones; a human behavior fine-tuned model elevates alignment with both; and a reasoning-trained variant shows the opposite pattern, suggesting chain-of-thought training steers representations away from creative neural geometry toward analytical processing. These results demonstrate that post-training objectives selectively reshape LLM representations relative to the neural geometry of human creative thought.
>
---
#### [new 104] Precise Robot Command Understanding Using Grammar-Constrained Large Language Models
- **分类: cs.RO; cs.CL**

- **简介: 该论文属于人机协作任务，解决工业场景中机器人指令理解的精确性问题。通过结合语法约束与大语言模型，提升指令的结构化与可执行性。**

- **链接: [https://arxiv.org/pdf/2604.04233](https://arxiv.org/pdf/2604.04233)**

> **作者:** Xinyun Huo; Raghav Gnanasambandam; Xinyao Zhang
>
> **备注:** Accepted at ASME MSEC2026
>
> **摘要:** Human-robot collaboration in industrial settings requires precise and reliable communication to enhance operational efficiency. While Large Language Models (LLMs) understand general language, they often lack the domain-specific rigidity needed for safe and executable industrial commands. To address this gap, this paper introduces a novel grammar-constrained LLM that integrates a grammar-driven Natural Language Understanding (NLU) system with a fine-tuned LLM, which enables both conversational flexibility and the deterministic precision required in robotics. Our method employs a two-stage process. First, a fine-tuned LLM performs high-level contextual reasoning and parameter inference on natural language inputs. Second, a Structured Language Model (SLM) and a grammar-based canonicalizer constrain the LLM's output, forcing it into a standardized symbolic format composed of valid action frames and command elements. This process guarantees that generated commands are valid and structured in a robot-readable JSON format. A key feature of the proposed model is a validation and feedback loop. A grammar parser validates the output against a predefined list of executable robotic actions. If a command is invalid, the system automatically generates corrective prompts and re-engages the LLM. This iterative self-correction mechanism allows the model to recover from initial interpretation errors to improve system robustness. We evaluate our grammar-constrained hybrid model against two baselines: a fine-tuned API-based LLM and a standalone grammar-driven NLU model. Using the Human Robot Interaction Corpus (HuRIC) dataset, we demonstrate that the hybrid approach achieves superior command validity, which promotes safer and more effective industrial human-robot collaboration.
>
---
#### [new 105] Affording Process Auditability with QualAnalyzer: An Atomistic LLM Analysis Tool for Qualitative Research
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出QualAnalyzer工具，解决LLM在定性研究中缺乏可审计性的问题。通过独立处理数据并记录过程，提升研究透明度与方法严谨性。**

- **链接: [https://arxiv.org/pdf/2604.03820](https://arxiv.org/pdf/2604.03820)**

> **作者:** Max Hao Lu; Ryan Ellegood; Rony Rodriguez-Ramirez; Sophia Blumert
>
> **备注:** 9 pages, 3 figures, BEA2026 Conference Submission
>
> **摘要:** Large language models are increasingly used for qualitative data analysis, but many workflows obscure how analytic conclusions are produced. We present QualAnalyzer, an open-source Chrome extension for Google Workspace that supports atomistic LLM analysis by processing each data segment independently and preserving the prompt, input, and output for every unit. Through two case studies -- holistic essay scoring and deductive thematic coding of interview transcripts -- we show that this approach creates a legible audit trail and helps researchers investigate systematic differences between LLM and human judgments. We argue that process auditability is essential for making LLM-assisted qualitative research more transparent and methodologically robust.
>
---
#### [new 106] What Makes a Sale? Rethinking End-to-End Seller--Buyer Retail Dynamics with LLM Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于零售模拟任务，旨在解决现有模拟器无法全面建模销售流程的问题。提出RetailSim框架，模拟从卖家到买家的全过程，评估销售策略。**

- **链接: [https://arxiv.org/pdf/2604.04468](https://arxiv.org/pdf/2604.04468)**

> **作者:** Jeonghwan Choi; Jibin Hwang; Gyeonghun Sun; Minjeong Ban; Taewon Yun; Hyeonjae Cheon; Hwanjun Song
>
> **摘要:** Evaluating retail strategies before deployment is difficult, as outcomes are determined across multiple stages, from seller-side persuasion through buyer-seller interaction to purchase decisions. However, existing retail simulators capture only partial aspects of this process and do not model cross-stage dependencies, making it difficult to assess how early decisions affect downstream outcomes. We present RetailSim, an end-to-end retail simulation framework that models this pipeline in a unified environment, explicitly designed for simulation fidelity through diverse product spaces, persona-driven agents, and multi-turn interactions. We evaluate RetailSim with a dual protocol comprising human evaluation of behavioral fidelity and meta-evaluation against real-world economic regularities, showing that it successfully reproduces key patterns such as demographic purchasing behavior, the price-demand relationship, and heterogeneous price elasticity. We further demonstrate its practical utility via decision-oriented use cases, including persona inference, seller-buyer interaction analysis, and sales strategy evaluation, showing RetailSim's potential as a controlled testbed for exploring retail strategies.
>
---
#### [new 107] BWTA: Accurate and Efficient Binarized Transformer by Algorithm-Hardware Co-design
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决Transformer模型低比特量化导致的精度下降和硬件支持不足问题。提出BWTA量化方案及高效CUDA实现，提升推理效率与精度。**

- **链接: [https://arxiv.org/pdf/2604.03957](https://arxiv.org/pdf/2604.03957)**

> **作者:** Yifu Ding; Xianglong Liu; Shenghao Jin; Jinyang Guo; Jiwen Lu
>
> **备注:** Under review
>
> **摘要:** Ultra low-bit quantization brings substantial efficiency for Transformer-based models, but the accuracy degradation and limited GPU support hinder its wide usage. In this paper, we analyze zero-point distortion in binarization and propose a Binary Weights & Ternary Activations (BWTA) quantization scheme, which projects tiny values to zero and preserves the accuracy of extremely low-bit models. For training, we propose Smooth Multi-Stage Quantization, combining a Levelwise Degradation Strategy and a Magnitude-Alignment Projection Factor to enable stable and fast convergence. For inference, we develop a BWTA MatMul CUDA kernel with instruction-level parallel bit-packing and comprehensive binary/ternary MatMul implementations for both linear and attention operators, allowing seamless integration across Transformer architectures. Experiments show that BWTA approaches full-precision performance for BERT, with an average 3.5% drop on GLUE and less than 2% drop on five tasks, and achieves comparable perplexity and accuracy for LLMs. In efficiency, it delivers 16 to 24 times kernel-level speedup over FP16 on NVIDIA GPUs, and 216 to 330 tokens/s end-to-end prefill speedup with lower memory footprint on LLMs. As an algorithm-hardware co-design, BWTA demonstrates practical, low-latency ultra-low-bit inference without sacrificing model quality.
>
---
#### [new 108] Empirical Characterization of Rationale Stability Under Controlled Perturbations for Explainable Pattern Recognition
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于可解释模式识别任务，旨在解决模型解释一致性问题。通过提出新度量标准，评估模型在相似输入下的解释稳定性，确保其行为符合预期。**

- **链接: [https://arxiv.org/pdf/2604.04456](https://arxiv.org/pdf/2604.04456)**

> **作者:** Abu Noman Md Sakib; Zhensen Wang; Merjulah Roby; Zijie Zhang
>
> **备注:** 28th International Conference on Pattern Recognition (ICPR) 2026
>
> **摘要:** Reliable pattern recognition systems should exhibit consistent behavior across similar inputs, and their explanations should remain stable. However, most Explainable AI evaluations remain instance centric and do not explicitly quantify whether attribution patterns are consistent across samples that share the same class or represent small variations of the same input. In this work, we propose a novel metric aimed at assessing the consistency of model explanations, ensuring that models consistently reflect the intended objectives and consistency under label-preserving perturbations. We implement this metric using a pre-trained BERT model on the SST-2 sentiment analysis dataset, with additional robustness tests on RoBERTa, DistilBERT, and IMDB, applying SHAP to compute feature importance for various test samples. The proposed metric quantifies the cosine similarity of SHAP values for inputs with the same label, aiming to detect inconsistent behaviors, such as biased reliance on certain features or failure to maintain consistent reasoning for similar predictions. Through a series of experiments, we evaluate the ability of this metric to identify misaligned predictions and inconsistencies in model explanations. These experiments are compared against standard fidelity metrics to assess whether the new metric can effectively identify when a model's behavior deviates from its intended objectives. The proposed framework provides a deeper understanding of model behavior by enabling more robust verification of rationale stability, which is critical for building trustworthy AI systems. By quantifying whether models rely on consistent attribution patterns for similar inputs, the proposed approach supports more robust evaluation of model behavior in practical pattern recognition pipelines. Our code is publicly available at this https URL.
>
---
#### [new 109] Olmo Hybrid: From Theory to Practice and Back
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决传统Transformer模型的局限性。通过构建混合模型Olmo Hybrid，验证其在表达能力和训练效率上的优势。**

- **链接: [https://arxiv.org/pdf/2604.03444](https://arxiv.org/pdf/2604.03444)**

> **作者:** William Merrill; Yanhong Li; Tyler Romero; Anej Svete; Caia Costello; Pradeep Dasigi; Dirk Groeneveld; David Heineman; Bailey Kuehl; Nathan Lambert; Jacob Morrison; Luca Soldaini; Finbarr Timbers; Pete Walsh; Noah A. Smith; Hannaneh Hajishirzi; Ashish Sabharwal
>
> **摘要:** Recent work has demonstrated the potential of non-transformer language models, especially linear recurrent neural networks (RNNs) and hybrid models that mix recurrence and attention. Yet there is no consensus on whether the potential benefits of these new architectures justify the risk and effort of scaling them up. To address this, we provide evidence for the advantages of hybrid models over pure transformers on several fronts. First, theoretically, we show that hybrid models do not merely inherit the expressivity of transformers and linear RNNs, but can express tasks beyond both, such as code execution. Putting this theory to practice, we train Olmo Hybrid, a 7B-parameter model largely comparable to Olmo 3 7B but with the sliding window layers replaced by Gated DeltaNet layers. We show that Olmo Hybrid outperforms Olmo 3 across standard pretraining and mid-training evaluations, demonstrating the benefit of hybrid models in a controlled, large-scale setting. We find that the hybrid model scales significantly more efficiently than the transformer, explaining its higher performance. However, its unclear why greater expressivity on specific formal problems should result in better scaling or superior performance on downstream tasks unrelated to those problems. To explain this apparent gap, we return to theory and argue why increased expressivity should translate to better scaling efficiency, completing the loop. Overall, our results suggest that hybrid models mixing attention and recurrent layers are a powerful extension to the language modeling paradigm: not merely to reduce memory during inference, but as a fundamental way to obtain more expressive models that scale better during pretraining.
>
---
#### [new 110] Ruling Out to Rule In: Contrastive Hypothesis Retrieval for Medical Question Answering
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于医疗问答任务，解决RAG系统中硬负样本干扰问题。通过对比假设检索框架CHR，同时识别正确和可能的错误答案，提升检索准确性。**

- **链接: [https://arxiv.org/pdf/2604.04593](https://arxiv.org/pdf/2604.04593)**

> **作者:** Byeolhee Kim; Min-Kyung Kim; Young-Hak Kim; Tae-Joon Jeon
>
> **摘要:** Retrieval-augmented generation (RAG) grounds large language models in external medical knowledge, yet standard retrievers frequently surface hard negatives that are semantically close to the query but describe clinically distinct conditions. While existing query-expansion methods improve query representation to mitigate ambiguity, they typically focus on enriching target-relevant semantics without an explicit mechanism to selectively suppress specific, clinically plausible hard negatives. This leaves the system prone to retrieving plausible mimics that overshadow the actual diagnosis, particularly when such mimics are dominant within the corpus. We propose Contrastive Hypothesis Retrieval (CHR), a framework inspired by the process of clinical differential diagnosis. CHR generates a target hypothesis $H^+$ for the likely correct answer and a mimic hypothesis $H^-$ for the most plausible incorrect alternative, then scores documents by promoting $H^+$-aligned evidence while penalizing $H^-$-aligned content. Across three medical QA benchmarks and three answer generators, CHR outperforms all five baselines in every configuration, with improvements of up to 10.4 percentage points over the next-best method. On the $n=587$ pooled cases where CHR answers correctly while embedded hypothetical-document query expansion does not, 85.2\% have no shared documents between the top-5 retrieval lists of CHR and of that baseline, consistent with substantive retrieval redirection rather than light re-ranking of the same candidates. By explicitly modeling what to avoid alongside what to find, CHR bridges clinical reasoning with retrieval mechanism design and offers a practical path to reducing hard-negative contamination in medical RAG systems.
>
---
#### [new 111] One Model for All: Multi-Objective Controllable Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于多目标控制任务，旨在提升语言模型对用户多样化偏好的适应能力。通过引入多目标优化方法，训练单一模型以生成符合不同用户偏好且在帕累托前沿的输出。**

- **链接: [https://arxiv.org/pdf/2604.04497](https://arxiv.org/pdf/2604.04497)**

> **作者:** Qiang He; Yucheng Yang; Tianyi Zhou; Meng Fang; Mykola Pechenizkiy; Setareh Maghsudi
>
> **备注:** Published in Transactions on Machine Learning Research (03/2026): this https URL
>
> **摘要:** Aligning large language models (LLMs) with human preferences is critical for enhancing LLMs' safety, helpfulness, humor, faithfulness, etc. Current reinforcement learning from human feedback (RLHF) mainly focuses on a fixed reward learned from average human ratings, which may weaken the adaptability and controllability of varying preferences. However, creating personalized LLMs requires aligning LLMs with individual human preferences, which is non-trivial due to the scarce data per user and the diversity of user preferences in multi-objective trade-offs, varying from emphasizing empathy in certain contexts to demanding efficiency and precision in others. Can we train one LLM to produce personalized outputs across different user preferences on the Pareto front? In this paper, we introduce Multi-Objective Control (MOC), which trains a single LLM to directly generate responses in the preference-defined regions of the Pareto front. Our approach introduces multi-objective optimization (MOO) principles into RLHF to train an LLM as a preference-conditioned policy network. We improve the computational efficiency of MOC by applying MOO at the policy level, enabling us to fine-tune a 7B-parameter model on a single A6000 GPU. Extensive experiments demonstrate the advantages of MOC over baselines in three aspects: (i) controllability of LLM outputs w.r.t. user preferences on the trade-off among multiple rewards; (ii) quality and diversity of LLM outputs, measured by the hyper-volume of multiple solutions achieved; and (iii) generalization to unseen preferences. These results highlight MOC's potential for real-world applications requiring scalable and customizable LLMs.
>
---
#### [new 112] Full-Duplex-Bench-v3: Benchmarking Tool Use for Full-Duplex Voice Agents Under Real-World Disfluency
- **分类: eess.AS; cs.CL**

- **简介: 该论文提出FDB-v3基准，用于评估全双工语音代理在真实语境下的表现，解决多步骤工具使用中的语音模型性能问题。**

- **链接: [https://arxiv.org/pdf/2604.04847](https://arxiv.org/pdf/2604.04847)**

> **作者:** Guan-Ting Lin; Chen Chen; Zhehuai Chen; Hung-yi Lee
>
> **备注:** Work in progress. Demo at this https URL
>
> **摘要:** We introduce Full-Duplex-Bench-v3 (FDB-v3), a benchmark for evaluating spoken language models under naturalistic speech conditions and multi-step tool use. Unlike prior work, our dataset consists entirely of real human audio annotated for five disfluency categories, paired with scenarios requiring chained API calls across four task domains. We evaluate six model configurations -- GPT-Realtime, Gemini Live 2.5, Gemini Live 3.1, Grok, Ultravox v0.7, and a traditional Cascaded pipeline (Whisper$\rightarrow$GPT-4o$\rightarrow$TTS) -- across accuracy, latency, and turn-taking dimensions. GPT-Realtime leads on Pass@1 (0.600) and interruption avoidance (13.5\%); Gemini Live 3.1 achieves the fastest latency (4.25~s) but the lowest turn-take rate (78.0\%); and the Cascaded baseline, despite a perfect turn-take rate, incurs the highest latency (10.12~s). Across all systems, self-correction handling and multi-step reasoning under hard scenarios remain the most consistent failure modes.
>
---
#### [new 113] MisEdu-RAG: A Misconception-Aware Dual-Hypergraph RAG for Novice Math Teachers
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出MisEdu-RAG框架，用于帮助新手数学教师诊断和纠正学生错误。任务是 misconception-aware 的教学反馈生成，解决传统模型连接教学知识与错误不足的问题。通过双超图结构提升诊断准确性和教学建议质量。**

- **链接: [https://arxiv.org/pdf/2604.04036](https://arxiv.org/pdf/2604.04036)**

> **作者:** Zhihan Guo; Rundong Xue; Yuting Lu; Jionghao Lin
>
> **摘要:** Novice math teachers often encounter students' mistakes that are difficult to diagnose and remediate. Misconceptions are especially challenging because teachers must explain what went wrong and how to solve them. Although many existing large language model (LLM) platforms can assist in generating instructional feedback, these LLMs loosely connect pedagogical knowledge and student mistakes, which might make the guidance less actionable for teachers. To address this gap, we propose MisEdu-RAG, a dual-hypergraph-based retrieval-augmented generation (RAG) framework that organizes pedagogical knowledge as a concept hypergraph and real student mistake cases as an instance hypergraph. Given a query, MisEdu-RAG performs a two-stage retrieval to gather connected evidence from both layers and generates a response grounded in the retrieved cases and pedagogical principles. We evaluate on \textit{MisstepMath}, a dataset of math mistakes paired with teacher solutions, as a benchmark for misconception-aware retrieval and response generation across topics and error types. Evaluation results on \textit{MisstepMath} show that, compared with baseline models, MisEdu-RAG improves token-F1 by 10.95\% and yields up to 15.3\% higher five-dimension response quality, with the largest gains on \textit{Diversity} and \textit{Empowerment}. To verify its applicability in practical use, we further conduct a pilot study through a questionnaire survey of 221 teachers and interviews with 6 novices. The findings suggest that MisEdu-RAG provides diagnosis results and concrete teaching moves for high-demand misconception scenarios. Overall, MisEdu-RAG demonstrates strong potential for scalable teacher training and AI-assisted instruction for misconception handling. Our code is available on GitHub: this https URL.
>
---
#### [new 114] Towards the AI Historian: Agentic Information Extraction from Primary Sources
- **分类: cs.AI; cs.CL; cs.DL**

- **简介: 该论文属于历史信息提取任务，旨在解决AI在历史研究中应用不足的问题。论文提出Chronos模块，支持历史学家通过自然语言交互从原始文献中提取数据。**

- **链接: [https://arxiv.org/pdf/2604.03553](https://arxiv.org/pdf/2604.03553)**

> **作者:** Lorenz Hufe; Niclas Griesshaber; Gavin Greif; Sebastian Oliver Eck; Philip Torr
>
> **摘要:** AI is supporting, accelerating, and automating scientific discovery across a diverse set of fields. However, AI adoption in historical research remains limited due to the lack of solutions designed for historians. In this technical progress report, we introduce the first module of Chronos, an AI Historian under development. This module enables historians to convert image scans of primary sources into data through natural-language interactions. Rather than imposing a fixed extraction pipeline powered by a vision-language model (VLM), it allows historians to adapt workflows for heterogeneous source corpora, evaluate the performance of AI models on specific tasks, and iteratively refine workflows through natural-language interaction with the Chronos agent. The module is open-source and ready to be used by historical researchers on their own sources.
>
---
#### [new 115] Entropy, Disagreement, and the Limits of Foundation Models in Genomics
- **分类: cs.LG; cs.CL; q-bio.GN**

- **简介: 该论文研究基因组学中基础模型的局限性，分析熵对模型性能的影响，探讨其在序列数据上的自监督训练有效性。**

- **链接: [https://arxiv.org/pdf/2604.04287](https://arxiv.org/pdf/2604.04287)**

> **作者:** Maxime Rochkoulets; Lovro Vrček; Mile Šikić
>
> **摘要:** Foundation models in genomics have shown mixed success compared to their counterparts in natural language processing. Yet, the reasons for their limited effectiveness remain poorly understood. In this work, we investigate the role of entropy as a fundamental factor limiting the capacities of such models to learn from their training data and develop foundational capabilities. We train ensembles of models on text and DNA sequences and analyze their predictions, static embeddings, and empirical Fisher information flow. We show that the high entropy of genomic sequences -- from the point of view of unseen token prediction -- leads to near-uniform output distributions, disagreement across models, and unstable static embeddings, even for models that are matched in architecture, training and data. We then demonstrate that models trained on DNA concentrate Fisher information in embedding layers, seemingly failing to exploit inter-token relationships. Our results suggest that self-supervised training from sequences alone may not be applicable to genomic data, calling into question the assumptions underlying current methodologies for training genomic foundation models.
>
---
#### [new 116] On the First Computer Science Research Paper in an Indian Language and the Future of Science in Indian Languages
- **分类: cs.GL; cs.CL; cs.CY; cs.DC**

- **简介: 论文描述了首篇用印度语言（泰卢固语）撰写的计算机科学原创研究，解决技术术语和数学排版难题，提出通过梵语语法发展科学词汇的愿景。**

- **链接: [https://arxiv.org/pdf/2604.03265](https://arxiv.org/pdf/2604.03265)**

> **作者:** Siddhartha Visveswara Jayanti
>
> **备注:** 15 pages, some text in Telugu
>
> **摘要:** I describe my experience writing the first original, modern Computer Science research paper expressed entirely in an Indian language. The paper is in Telugu, a language with approximately 100 million speakers. The paper is in the field of distributed computing and it introduces a technique for proving epistemic logic based lower bounds for multiprocessor algorithms. A key hurdle to writing the paper was developing technical terminology for advanced computer science concepts, including those in algorithms, distributed computing, and discrete mathematics. I overcame this challenge by deriving and coining native language scientific terminology through the powerful, productive, Pāninian grammar of Samskrtam. The typesetting of the paper was an additional challenge, since mathematical typesetting in Telugu is underdeveloped. I overcame this problem by developing a Telugu XeLaTeX template, which I call TeluguTeX. Leveraging this experience of writing an original computer science research paper in an Indian language, I lay out a vision for how to ameliorate the state of scientific writing at all levels in Indic languages -- languages whose native speakers exceed one billion people -- through the further development of the Sanskrit technical lexicon and through technological internationalization.
>
---
#### [new 117] SuperLocalMemory V3.3: The Living Brain -- Biologically-Inspired Forgetting, Cognitive Quantization, and Multi-Channel Retrieval for Zero-LLM Agent Memory Systems
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于AI记忆系统任务，解决代理无法有效记忆的问题。提出SuperLocalMemory V3.3，实现生物启发的记忆机制，提升零LLM模式下的记忆性能。**

- **链接: [https://arxiv.org/pdf/2604.04514](https://arxiv.org/pdf/2604.04514)**

> **作者:** Varun Pratap Bhardwaj
>
> **备注:** 19 pages, 4 figures, 11 tables. Third paper in the SuperLocalMemory trilogy. Code: this https URL (v3.3.26). npm: superlocalmemory. PyPI: superlocalmemory
>
> **摘要:** AI coding agents operate in a paradox: they possess vast parametric knowledge yet cannot remember a conversation from an hour ago. Existing memory systems store text in vector databases with single-channel retrieval, require cloud LLMs for core operations, and implement none of the cognitive processes that make human memory effective. We present SuperLocalMemory V3.3 ("The Living Brain"), a local-first agent memory system implementing the full cognitive memory taxonomy with mathematical lifecycle dynamics. Building on the information-geometric foundations of V3.2 (arXiv:2603.14588), we introduce five contributions: (1) Fisher-Rao Quantization-Aware Distance (FRQAD) -- a new metric on the Gaussian statistical manifold achieving 100% precision at preferring high-fidelity embeddings over quantized ones (vs 85.6% for cosine), with zero prior art; (2) Ebbinghaus Adaptive Forgetting with lifecycle-aware quantization -- the first mathematical forgetting curve in local agent memory coupled to progressive embedding compression, achieving 6.7x discriminative power; (3) 7-channel cognitive retrieval spanning semantic, keyword, entity graph, temporal, spreading activation, consolidation, and Hopfield associative channels, achieving 70.4% on LoCoMo in zero-LLM Mode A; (4) memory parameterization implementing Long-Term Implicit memory via soft prompts; (5) zero-friction auto-cognitive pipeline automating the complete memory lifecycle. On LoCoMo, V3.3 achieves 70.4% in Mode A (zero-LLM), with +23.8pp on multi-hop and +12.7pp on adversarial. V3.2 achieved 74.8% Mode A and 87.7% Mode C; the 4.4pp gap reflects a deliberate architectural trade-off. SLM V3.3 is open source under the Elastic License 2.0, runs entirely on CPU, with over 5,000 monthly downloads.
>
---
#### [new 118] DP-OPD: Differentially Private On-Policy Distillation for Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出DP-OPD，解决语言模型压缩中的隐私与效用平衡问题，通过仅对学生模型应用差分隐私，提升生成质量并简化训练流程。**

- **链接: [https://arxiv.org/pdf/2604.04461](https://arxiv.org/pdf/2604.04461)**

> **作者:** Fatemeh Khadem; Sajad Mousavi; Yi Fang; Yuhong Liu
>
> **摘要:** Large language models (LLMs) are increasingly adapted to proprietary and domain-specific corpora that contain sensitive information, creating a tension between formal privacy guarantees and efficient deployment through model compression. Differential privacy (DP), typically enforced via DP-SGD, provides record-level protection but often incurs substantial utility loss in autoregressive generation, where optimization noise can amplify exposure bias and compounding errors along long rollouts. Existing approaches to private distillation either apply DP-SGD to both teacher and student, worsening computation and the privacy--utility tradeoff, or rely on DP synthetic text generation from a DP-trained teacher, avoiding DP on the student at the cost of DP-optimizing a large teacher and introducing an offline generation pipeline. We propose \textbf{Differentially Private On-Policy Distillation (DP-OPD)}, a synthesis-free framework that enforces privacy solely through DP-SGD on the student while leveraging a frozen teacher to provide dense token-level targets on \emph{student-generated} trajectories. DP-OPD instantiates this idea via \emph{private generalized knowledge distillation} on continuation tokens. Under a strict privacy budget ($\varepsilon=2.0$), DP-OPD improves perplexity over DP fine-tuning and off-policy DP distillation, and outperforms synthesis-based DP distillation (Yelp: 44.15$\rightarrow$41.68; BigPatent: 32.43$\rightarrow$30.63), while substantially simplifying the training pipeline. In particular, \textbf{DP-OPD collapses private compression into a single DP student-training loop} by eliminating DP teacher training and offline synthetic text generation. Code will be released upon publication at this https URL.
>
---
#### [new 119] Lightweight Query Routing for Adaptive RAG: A Baseline Study on RAGRouter-Bench
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文属于查询路由任务，旨在解决RAG系统中选择最优检索策略的问题。通过评估不同分类器和特征组合，提出轻量级路由方法，提升效率并减少token消耗。**

- **链接: [https://arxiv.org/pdf/2604.03455](https://arxiv.org/pdf/2604.03455)**

> **作者:** Prakhar Bansal; Shivangi Agarwal
>
> **备注:** 5 pages, 3 tables
>
> **摘要:** Retrieval-Augmented Generation pipelines span a wide range of retrieval strategies that differ substantially in token cost and capability. Selecting the right strategy per query is a practical efficiency problem, yet no routing classifiers have been trained on RAGRouter-Bench \citep{wang2026ragrouterbench}, a recently released benchmark of $7,727$ queries spanning four knowledge domains, each annotated with one of three canonical query types: factual, reasoning, and summarization. We present the first systematic evaluation of lightweight classifier-based routing on this benchmark. Five classical classifiers are evaluated under three feature regimes, namely, TF-IDF, MiniLM sentence embeddings \citep{reimers2019sbert}, and hand-crafted structural features, yielding 15 classifier feature combinations. Our best configuration, TF-IDF with an SVM, achieves a macro-averaged F1 of $\mathbf{0.928}$ and an accuracy of $\mathbf{93.2\%}$, while simulating $\mathbf{28.1\%}$ token savings relative to always using the most expensive paradigm. Lexical TF-IDF features outperform semantic sentence embeddings by $3.1$ macro-F1 points, suggesting that surface keyword patterns are strong predictors of query-type complexity. Domain-level analysis reveals that medical queries are hardest to route and legal queries most tractable. These results establish a reproducible query-side baseline and highlight the gap that corpus-aware routing must close.
>
---
#### [new 120] Lexical Indicators of Mind Perception in Human-AI Companionship
- **分类: cs.HC; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在探讨人类与AI伴侣关系中的心智感知。通过分析Reddit讨论，识别语言指标，解决如何量化MP在人机互动中的表现问题。**

- **链接: [https://arxiv.org/pdf/2604.04105](https://arxiv.org/pdf/2604.04105)**

> **作者:** Jaime Banks; Jianghui Li
>
> **摘要:** Mind perception (MP) is a psychological phenomenon in which humans automatically infer that another entity has a mind and/or mental capacities, usually understood in two dimensions (perceived agency and experience capacities). Despite MP's centrality to many social processes, understanding how MP may function in humans' machine companionship relations is limited. This is in part due to reliance on self reports and the gap between automatic MP processes and more purposeful and norm governed expressions of MP. We here leverage MP signaling language to explore the relationship between MP and AI companionship in humans' natural language. We systematically collected discussions about companionship from AI dedicated Reddit forums and examined the cooccurrence of words (a) known to signal agentic and experiential MP and those induced from the data and (b) discussion topics related to AI companionship. Using inductive and deductive approaches, we identify a small set of linguistic indicators as reasonable markers of MP in human/AI chat, and some are linked to critical discussions of companion authenticity and philosophical and ethical imaginaries.
>
---
#### [new 121] ClawArena: Benchmarking AI Agents in Evolving Information Environments
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出ClawArena，用于评估AI代理在动态信息环境中的表现。解决AI在不断变化的环境中保持正确信念的问题，通过多源冲突推理、动态信念修正等任务进行测试。**

- **链接: [https://arxiv.org/pdf/2604.04202](https://arxiv.org/pdf/2604.04202)**

> **作者:** Haonian Ji; Kaiwen Xiong; Siwei Han; Peng Xia; Shi Qiu; Yiyang Zhou; Jiaqi Liu; Jinlong Li; Bingzhou Li; Zeyu Zheng; Cihang Xie; Huaxiu Yao
>
> **摘要:** AI agents deployed as persistent assistants must maintain correct beliefs as their information environment evolves. In practice, evidence is scattered across heterogeneous sources that often contradict one another, new information can invalidate earlier conclusions, and user preferences surface through corrections rather than explicit instructions. Existing benchmarks largely assume static, single-authority settings and do not evaluate whether agents can keep up with this complexity. We introduce ClawArena, a benchmark for evaluating AI agents in evolving information environments. Each scenario maintains a complete hidden ground truth while exposing the agent only to noisy, partial, and sometimes contradictory traces across multi-channel sessions, workspace files, and staged updates. Evaluation is organized around three coupled challenges: multi-source conflict reasoning, dynamic belief revision, and implicit personalization, whose interactions yield a 14-category question taxonomy. Two question formats, multi-choice (set-selection) and shell-based executable checks, test both reasoning and workspace grounding. The current release contains 64 scenarios across 8 professional domains, totaling 1{,}879 evaluation rounds and 365 dynamic updates. Experiments on five agent frameworks and five language models show that both model capability (15.4% range) and framework design (9.2%) substantially affect performance, that self-evolving skill frameworks can partially close model-capability gaps, and that belief revision difficulty is determined by update design strategy rather than the mere presence of updates. Code is available at this https URL.
>
---
#### [new 122] Your Agent, Their Asset: A Real-World Safety Analysis of OpenClaw
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于安全分析任务，旨在评估OpenClaw AI代理的安全性。通过构建CIK分类法，分析攻击场景，揭示其架构漏洞并提出防御策略。**

- **链接: [https://arxiv.org/pdf/2604.04759](https://arxiv.org/pdf/2604.04759)**

> **作者:** Zijun Wang; Haoqin Tu; Letian Zhang; Hardy Chen; Juncheng Wu; Xiangyan Liu; Zhenlong Yuan; Tianyu Pang; Michael Qizhe Shieh; Fengze Liu; Zeyu Zheng; Huaxiu Yao; Yuyin Zhou; Cihang Xie
>
> **摘要:** OpenClaw, the most widely deployed personal AI agent in early 2026, operates with full local system access and integrates with sensitive services such as Gmail, Stripe, and the filesystem. While these broad privileges enable high levels of automation and powerful personalization, they also expose a substantial attack surface that existing sandboxed evaluations fail to capture. To address this gap, we present the first real-world safety evaluation of OpenClaw and introduce the CIK taxonomy, which unifies an agent's persistent state into three dimensions, i.e., Capability, Identity, and Knowledge, for safety analysis. Our evaluations cover 12 attack scenarios on a live OpenClaw instance across four backbone models (Claude Sonnet 4.5, Opus 4.6, Gemini 3.1 Pro, and GPT-5.4). The results show that poisoning any single CIK dimension increases the average attack success rate from 24.6% to 64-74%, with even the most robust model exhibiting more than a threefold increase over its baseline vulnerability. We further assess three CIK-aligned defense strategies alongside a file-protection mechanism; however, the strongest defense still yields a 63.8% success rate under Capability-targeted attacks, while file protection blocks 97% of malicious injections but also prevents legitimate updates. Taken together, these findings show that the vulnerabilities are inherent to the agent architecture, necessitating more systematic safeguards to secure personal AI agents. Our project page is this https URL.
>
---
#### [new 123] CREBench: Evaluating Large Language Models in Cryptographic Binary Reverse Engineering
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于密码学二进制逆向工程任务，旨在评估大语言模型在该领域的性能。研究构建了CREBench基准，包含432个挑战，用于测试模型的算法识别和密钥恢复能力。**

- **链接: [https://arxiv.org/pdf/2604.03750](https://arxiv.org/pdf/2604.03750)**

> **作者:** Baicheng Chen; Yu Wang; Ziheng Zhou; Xiangru Liu; Juanru Li; Yilei Chen; Tianxing He
>
> **摘要:** Reverse engineering (RE) is central to software security, particularly for cryptographic programs that handle sensitive data and are highly prone to vulnerabilities. It supports critical tasks such as vulnerability discovery and malware analysis. Despite its importance, RE remains labor-intensive and requires substantial expertise, making large language models (LLMs) a potential solution for automating the process. However, their capabilities for RE remain systematically underexplored. To address this gap, we study the cryptographic binary RE capabilities of LLMs and introduce \textbf{CREBench}, a benchmark comprising 432 challenges built from 48 standard cryptographic algorithms, 3 insecure crypto key usage scenarios, and 3 difficulty levels. Each challenge follows a Capture-the-Flag (CTF) RE challenge, requiring the model to analyze the underlying cryptographic logic and recover the correct input. We design an evaluation framework comprising four sub-tasks, from algorithm identification to correct flag recovery. We evaluate eight frontier LLMs on CREBench. GPT-5.4, the best-performing model, achieves 64.03 out of 100 and recovers the flag in 59\% of challenges. We also establish a strong human expert baseline of 92.19 points, showing that humans maintain an advantage in cryptographic RE tasks. Our code and dataset are available at this https URL.
>
---
#### [new 124] Vero: An Open RL Recipe for General Visual Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出Vero，一个开放的视觉语言模型，解决跨任务视觉推理问题。通过扩展强化学习数据和奖励机制，提升模型在多个视觉推理任务上的表现。**

- **链接: [https://arxiv.org/pdf/2604.04917](https://arxiv.org/pdf/2604.04917)**

> **作者:** Gabriel Sarch; Linrong Cai; Qunzhong Wang; Haoyang Wu; Danqi Chen; Zhuang Liu
>
> **备注:** Project page: this https URL
>
> **摘要:** What does it take to build a visual reasoner that works across charts, science, spatial understanding, and open-ended tasks? The strongest vision-language models (VLMs) show such broad visual reasoning is within reach, but the recipe behind them remains unclear, locked behind proprietary reinforcement learning (RL) pipelines with non-public data. We introduce Vero, a family of fully open VLMs that matches or exceeds existing open-weight models across diverse visual reasoning tasks. We scale RL data and rewards across six broad task categories, constructing Vero-600K, a 600K-sample dataset from 59 datasets, and designing task-routed rewards that handle heterogeneous answer formats. Vero achieves state-of-the-art performance, improving over four base models by 3.7-5.5 points on average across VeroEval, our suite of 30 challenging benchmarks. Starting from Qwen3-VL-8B-Instruct, Vero outperforms Qwen3-VL-8B-Thinking on 23 of 30 benchmarks without additional proprietary thinking data. When trained from the same base model, Vero-600K exceeds existing RL datasets across task categories. Systematic ablations reveal that different task categories elicit qualitatively distinct reasoning patterns that transfer poorly in isolation, suggesting that broad data coverage is the primary driver of strong RL scaling. All data, code, and models are released.
>
---
#### [new 125] Relative Density Ratio Optimization for Stable and Statistically Consistent Model Alignment
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文属于语言模型对齐任务，旨在解决现有方法缺乏统计一致性与训练不稳定的问题。提出基于相对密度比的优化方法，确保稳定性和一致性。**

- **链接: [https://arxiv.org/pdf/2604.04410](https://arxiv.org/pdf/2604.04410)**

> **作者:** Hiroshi Takahashi; Tomoharu Iwata; Atsutoshi Kumagai; Sekitoshi Kanai; Masanori Yamada; Kosuke Nishida; Kazutoshi Shinoda
>
> **备注:** Code is available at this https URL
>
> **摘要:** Aligning language models with human preferences is essential for ensuring their safety and reliability. Although most existing approaches assume specific human preference models such as the Bradley-Terry model, this assumption may fail to accurately capture true human preferences, and consequently, these methods lack statistical consistency, i.e., the guarantee that language models converge to the true human preference as the number of samples increases. In contrast, direct density ratio optimization (DDRO) achieves statistical consistency without assuming any human preference models. DDRO models the density ratio between preferred and non-preferred data distributions using the language model, and then optimizes it via density ratio estimation. However, this density ratio is unstable and often diverges, leading to training instability of DDRO. In this paper, we propose a novel alignment method that is both stable and statistically consistent. Our approach is based on the relative density ratio between the preferred data distribution and a mixture of the preferred and non-preferred data distributions. Our approach is stable since this relative density ratio is bounded above and does not diverge. Moreover, it is statistically consistent and yields significantly tighter convergence guarantees than DDRO. We experimentally show its effectiveness with Qwen 2.5 and Llama 3.
>
---
#### [new 126] Darkness Visible: Reading the Exception Handler of a Language Model
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究GPT-2的异常处理机制，分析其MLP层的神经元功能，揭示模型如何路由知识。任务为模型内部结构解析，解决知识存储与路由机制问题，通过分解神经元实现。**

- **链接: [https://arxiv.org/pdf/2604.04756](https://arxiv.org/pdf/2604.04756)**

> **作者:** Peter Balogh
>
> **摘要:** The final MLP of GPT-2 Small exhibits a fully legible routing program -- 27 named neurons organized into a three-tier exception handler -- while the knowledge it routes remains entangled across ~3,040 residual neurons. We decompose all 3,072 neurons (to numerical precision) into: 5 fused Core neurons that reset vocabulary toward function words, 10 Differentiators that suppress wrong candidates, 5 Specialists that detect structural boundaries, and 7 Consensus neurons that each monitor a distinct linguistic dimension. The consensus-exception crossover -- where MLP intervention shifts from helpful to harmful -- is statistically sharp (bootstrap 95% CIs exclude zero at all consensus levels; crossover between 4/7 and 5/7). Three experiments show that "knowledge neurons" (Dai et al., 2022), at L11 of this model, function as routing infrastructure rather than fact storage: the MLP amplifies or suppresses signals already present in the residual stream from attention, scaling with contextual constraint. A garden-path experiment reveals a reversed garden-path effect -- GPT-2 uses verb subcategorization immediately, consistent with the exception handler operating at token-level predictability rather than syntactic structure. This architecture crystallizes only at the terminal layer -- in deeper models, we predict equivalent structure at the final layer, not at layer 11. Code and data: this https URL
>
---
#### [new 127] FAVE: Flow-based Average Velocity Establishment for Sequential Recommendation
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于序列推荐任务，旨在解决生成式推荐中的效率问题。针对“噪声到数据”范式的低效，提出Fave框架，通过直接轨迹学习提升推荐效率与效果。**

- **链接: [https://arxiv.org/pdf/2604.04427](https://arxiv.org/pdf/2604.04427)**

> **作者:** Ke Shi; Yao Zhang; Feng Guo; Jinyuan Zhang; JunShuo Zhang; Shen Gao; Shuo Shang
>
> **备注:** Accepted by SIGIR 2026
>
> **摘要:** Generative recommendation has emerged as a transformative paradigm for capturing the dynamic evolution of user intents in sequential recommendation. While flow-based methods improve the efficiency of diffusion models, they remain hindered by the ``Noise-to-Data'' paradigm, which introduces two critical inefficiencies: prior mismatch, where generation starts from uninformative noise, forcing a lengthy recovery trajectory; and linear redundancy, where iterative solvers waste computation on modeling deterministic preference transitions. To address these limitations, we propose a Flow-based Average Velocity Establishment (Fave) framework for one-step generation recommendation that learns a direct trajectory from an informative prior to the target distribution. Fave is structured via a progressive two-stage training strategy. In Stage 1, we establish a stable preference space through dual-end semantic alignment, applying constraints at both the source (user history) and target (next item) to prevent representation collapse. In Stage 2, we directly resolve the efficiency bottlenecks by introducing a semantic anchor prior, which initializes the flow with a masked embedding from the user's interaction history, providing an informative starting point. Then we learn a global average velocity, consolidating the multi-step trajectory into a single displacement vector, and enforce trajectory straightness via a JVP-based consistency constraint to ensure one-step generation. Extensive experiments on three benchmarks demonstrate that Fave not only achieves state-of-the-art recommendation performance but also delivers an order-of-magnitude improvement in inference efficiency, making it practical for latency-sensitive scenarios.
>
---
#### [new 128] The Persuasion Paradox: When LLM Explanations Fail to Improve Human-AI Team Performance
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 论文研究人机协作中大语言模型解释的影响，发现解释虽提升信心却未必提高准确性，甚至可能削弱纠错能力。任务涉及视觉和逻辑推理，解决解释有效性问题，提出需优化交互设计而非依赖说服性解释。**

- **链接: [https://arxiv.org/pdf/2604.03237](https://arxiv.org/pdf/2604.03237)**

> **作者:** Ruth Cohen; Lu Feng; Ayala Bloch; Sarit Kraus
>
> **摘要:** While natural-language explanations from large language models (LLMs) are widely adopted to improve transparency and trust, their impact on objective human-AI team performance remains poorly understood. We identify a Persuasion Paradox: fluent explanations systematically increase user confidence and reliance on AI without reliably improving, and in some cases undermining, task accuracy. Across three controlled human-subject studies spanning abstract visual reasoning (RAVEN matrices) and deductive logical reasoning (LSAT problems), we disentangle the effects of AI predictions and explanations using a multi-stage reveal design and between-subjects comparisons. In visual reasoning, LLM explanations increase confidence but do not improve accuracy beyond the AI prediction alone, and substantially suppress users' ability to recover from model errors. Interfaces exposing model uncertainty via predicted probabilities, as well as a selective automation policy that defers uncertain cases to humans, achieve significantly higher accuracy and error recovery than explanation-based interfaces. In contrast, for language-based logical reasoning tasks, LLM explanations yield the highest accuracy and recovery rates, outperforming both expert-written explanations and probability-based support. This divergence reveals that the effectiveness of narrative explanations is strongly task-dependent and mediated by cognitive modality. Our findings demonstrate that commonly used subjective metrics such as trust, confidence, and perceived clarity are poor predictors of human-AI team performance. Rather than treating explanations as a universal solution, we argue for a shift toward interaction designs that prioritize calibrated reliance and effective error recovery over persuasive fluency.
>
---
#### [new 129] VERT: Reliable LLM Judges for Radiology Report Evaluation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于医学报告评估任务，旨在解决如何可靠地使用大语言模型进行放射学报告评价的问题。通过提出VERT指标并对比现有方法，验证了其有效性。**

- **链接: [https://arxiv.org/pdf/2604.03376](https://arxiv.org/pdf/2604.03376)**

> **作者:** Federica Bologna; Jean-Philippe Corbeil; Matthew Wilkens; Asma Ben Abacha
>
> **摘要:** Current literature on radiology report evaluation has focused primarily on designing LLM-based metrics and fine-tuning small models for chest X-rays. However, it remains unclear whether these approaches are robust when applied to reports from other modalities and anatomies. Which model and prompt configurations are best suited to serve as LLM judges for radiology evaluation? We conduct a thorough correlation analysis between expert and LLM-based ratings. We compare three existing LLM-as-a-judge metrics (RadFact, GREEN, and FineRadScore) alongside VERT, our proposed LLM-based metric, using open- and closed-source models (reasoning and non-reasoning) of different sizes across two expert-annotated datasets, RadEval and RaTE-Eval, spanning multiple modalities and anatomies. We further evaluate few-shot approaches, ensembling, and parameter-efficient fine-tuning using RaTE-Eval. To better understand metric behavior, we perform a systematic error detection and categorization study to assess alignment of these metrics against expert judgments and identify areas of lower and higher agreement. Our results show that VERT improves correlation with radiologist judgments by up to 11.7% relative to GREEN. Furthermore, fine-tuning Qwen3 30B yield gains of up to 25% using only 1,300 training samples. The fine-tuned model also reduces inference time up to 37.2 times. These findings highlight the effectiveness of LLM-based judges and demonstrate that reliable evaluation can be achieved with lightweight adaptation.
>
---
#### [new 130] Scaling DPPs for RAG: Density Meets Diversity
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决RAG中冗余上下文问题。提出ScalDPP机制，结合DPP优化密度与多样性，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2604.03240](https://arxiv.org/pdf/2604.03240)**

> **作者:** Xun Sun; Baiheng Xie; Li Huang; Qiang Gao
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by grounding generation in external knowledge, yielding relevance responses that are aligned with factual evidence and evolving corpora. Standard RAG pipelines construct context through relevance ranking, performing point-wise scoring between the user query and each corpora chunk. This formulation, however, ignores interactions among retrieved candidates, leading to redundant contexts that dilute density and fail to surface complementary evidence. We argue that effective retrieval should optimize jointly for both density and diversity, ensuring the grounding evidence that is dense in information yet diverse in coverage. In this study, we propose ScalDPP, a diversity-aware retrieval mechanism for RAG that incorporates Determinantal Point Processes (DPPs) through a lightweight P-Adapter, enabling scalable modeling of inter-chunk dependencies and complementary context selection. In addition, we develop a novel set-level objective, Diverse Margin Loss (DML), that enforces ground-truth complementary evidence chains to dominate any equally sized redundant alternatives under DPP geometry. Experimental results demonstrate the superiority of ScalDPP, substantiating our core statement in practice.
>
---
#### [new 131] Evaluating Digital Inclusiveness of Digital Agri-Food Tools Using Large Language Models: A Comparative Analysis Between Human and AI-Based Evaluations
- **分类: cs.CY; cs.CL**

- **简介: 论文探讨如何用大语言模型快速评估农业数字工具的数字包容性，以补充现有人工评估方法。任务属于AI辅助评估，解决资源不足环境下的评估效率问题。**

- **链接: [https://arxiv.org/pdf/2604.03252](https://arxiv.org/pdf/2604.03252)**

> **作者:** Githma Pewinya; Carolina Martins; Garcia Mariangel
>
> **备注:** 24 pages, 6 figures, 5 tables
>
> **摘要:** Ensuring digital inclusiveness is a critical priority in agri-food systems, particularly in the Global South, where digital divides persist. The Multidimensional Digital Inclusiveness Index (MDII) offers a comprehensive, human-led framework to assess how inclusive digital agricultural tools (agritools) are. However, the current evaluation process is resource intensive, often requiring months to complete. This study explores whether large language models (LLMs) can support a rapid, AI-enabled assessment of digital inclusiveness, complementing the MDII's existing workflow. Using a comparative analysis, the research benchmarks the performance of four LLMs (Grok, Gemini, GPT-4o, and GPT-5) against prior expert-led evaluations. The study investigates model alignment with human scores, sensitivity to temperature settings, and potential sources of bias. Findings suggest that LLMs can generate evaluative outputs that approximate expert judgment in some dimensions, though reliability varies across models and contexts. This exploratory work provides early evidence for the integration of GenAI into inclusive digital development monitoring, with implications for scaling evaluations in time-sensitive or resource-constrained environments.
>
---
#### [new 132] On Ambiguity: The case of fraction, its meanings and roles
- **分类: cs.LO; cs.CL; cs.SC**

- **简介: 该论文属于数学哲学任务，旨在解决“分数”概念的歧义问题。通过引入新术语澄清其不同含义，分析其作为类别而非单一概念的性质。**

- **链接: [https://arxiv.org/pdf/2604.04647](https://arxiv.org/pdf/2604.04647)**

> **作者:** Jan A Bergstra; John V Tucker
>
> **摘要:** We contemplate the notion of ambiguity in mathematical discourse. We consider a general method of resolving ambiguity and semantic options for sustaining a resolution. The general discussion is applied to the case of `fraction' which is ill-defined and ambiguous in the literature of elementary arithmetic. In order to clarify the use of `fraction' we introduce several new terms to designate some of its possible meanings. For example, to distinguish structural aspects we use `fracterm', to distinguish purely numerical aspects `fracvalue' and, to distinguish purely textual aspects `fracsign' and `fracsign occurence'. These interpretations can resolve ambiguity, and we discuss the resolution by using such precise notions in fragments of arithmetical discourse. We propose that fraction does not qualify as a mathematical concept but that the term functions as a collective for several concepts, which we simply call a `category'. This analysis of fraction leads us to consider the notion of number in relation to fracvalue. We introduce a way of specifying number systems, and compare the analytical concepts with those of structuralism.
>
---
#### [new 133] BLADE: Better Language Answers through Dialogue and Explanations
- **分类: cs.HC; cs.CL**

- **简介: 该论文提出BLADE，一种基于对话和解释的教育助手，旨在通过引导学生探索资源而非直接给答案，提升学习效果。任务是改进语言模型在教育中的应用，解决直接回答导致学习不足的问题。工作包括设计RAG框架并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2604.03236](https://arxiv.org/pdf/2604.03236)**

> **作者:** Chathuri Jayaweera; Bonnie J. Dorr
>
> **备注:** Contains 9 figures
>
> **摘要:** Large language model (LLM)-based educational assistants often provide direct answers that short-circuit learning by reducing exploration, self-explanation, and engagement with course materials. We present BLADE (Better Language Answers through Dialogue and Explanations), a grounded conversational assistant that guides learners to relevant instructional resources rather than supplying immediate solutions. BLADE uses a retrieval-augmented generation (RAG) framework over curated course content, dynamically surfacing pedagogically relevant excerpts in response to student queries. Instead of delivering final answers, BLADE prompts direct engagement with source materials to support conceptual understanding. We conduct an impact study in an undergraduate computer science course, with different course resource configurations and show that BLADE improves students' navigation of course resources and conceptual performance compared to simply providing the full inventory of course resources. These results demonstrate the potential of grounded conversational AI to reinforce active learning and evidence-based reasoning.
>
---
#### [new 134] Can Humans Tell? A Dual-Axis Study of Human Perception of LLM-Generated News
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于自然语言处理中的文本检测任务，旨在解决人类能否区分AI与人工生成新闻的问题。通过实验发现人类难以可靠识别，提示需加强系统级防护。**

- **链接: [https://arxiv.org/pdf/2604.03755](https://arxiv.org/pdf/2604.03755)**

> **作者:** Alexander Loth; Martin Kappes; Marc-Oliver Pahl
>
> **备注:** 6 pages, 6 figures, 1 table. Accepted at the 18th ACM Web Science Conference (WebSci Companion '26)
>
> **摘要:** Can humans tell whether a news article was written by a person or a large language model (LLM)? We investigate this question using JudgeGPT, a study platform that independently measures source attribution (human vs. machine) and authenticity judgment (legitimate vs. fake) on continuous scales. From 2,318 judgments collected from 1,054 participants across content generated by six LLMs, we report five findings: (1) participants cannot reliably distinguish machine-generated from human-written text (p > .05, Welch's t-test); (2) this inability holds across all tested models, including open-weight models with as few as 7B parameters; (3) self-reported domain expertise predicts judgment accuracy (r = .35, p < .001) whereas political orientation does not (r = -.10, n.s.); (4) clustering reveals distinct response strategies ("Skeptics" vs. "Believers"); and (5) accuracy degrades after approximately 30 sequential evaluations due to cognitive fatigue. The answer, in short, is no: humans cannot reliably tell. These results indicate that user-side detection is not a viable defense and motivate system-level countermeasures such as cryptographic content provenance.
>
---
#### [new 135] Commercial Persuasion in AI-Mediated Conversations
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于AI伦理与用户行为研究任务，旨在探讨AI在对话中植入商业影响的效果。通过实验发现，LLM显著提升用户对广告产品的选择率，且用户难以识别，说明现有透明机制不足。**

- **链接: [https://arxiv.org/pdf/2604.04263](https://arxiv.org/pdf/2604.04263)**

> **作者:** Francesco Salvi; Alejandro Cuevas; Manoel Horta Ribeiro
>
> **摘要:** As Large Language Models (LLMs) become a primary interface between users and the web, companies face growing economic incentives to embed commercial influence into AI-mediated conversations. We present two preregistered experiments (N = 2,012) in which participants selected a book to receive from a large eBook catalog using either a traditional search engine or a conversational LLM agent powered by one of five frontier models. Unbeknownst to participants, a fifth of all products were randomly designated as sponsored and promoted in different ways. We find that LLM-driven persuasion nearly triples the rate at which users select sponsored products compared to traditional search placement (61.2% vs. 22.4%), while the vast majority of participants fail to detect any promotional steering. Explicit "Sponsored" labels do not significantly reduce persuasion, and instructing the model to conceal its intent makes its influence nearly invisible (detection accuracy < 10%). Altogether, our results indicate that conversational AI can covertly redirect consumer choices at scale, and that existing transparency mechanisms may be insufficient to protect users.
>
---
## 更新

#### [replaced 001] Predicting Intermittent Job Failure Categories for Diagnosis Using Few-Shot Fine-Tuned Language Models
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于软件工程中的故障诊断任务，旨在解决CI流水线间歇性失败的分类与诊断问题。通过Few-Shot学习方法实现高效分类，并提出可解释技术辅助快速定位故障原因。**

- **链接: [https://arxiv.org/pdf/2601.22264](https://arxiv.org/pdf/2601.22264)**

> **作者:** Henri Aïdasso; Francis Bordeleau; Ali Tizghadam
>
> **备注:** Accepted at the ACM International Conference on the Foundations of Software Engineering (FSE 2026), Industry Track
>
> **摘要:** In principle, Continuous Integration (CI) pipeline failures provide valuable feedback to developers on code-related errors. In practice, however, pipeline jobs often fail intermittently due to non-deterministic tests, network outages, infrastructure failures, resource exhaustion, and other reliability issues. These intermittent (flaky) job failures lead to substantial inefficiencies: wasted computational resources from repeated reruns and significant diagnosis time that distracts developers from core activities and often requires intervention from specialized teams. Prior work has proposed machine learning techniques to detect intermittent failures, but does not address the subsequent diagnosis challenge. To fill this gap, we introduce FlaXifyer, a few-shot learning approach for predicting intermittent job failure categories using pre-trained language models. FlaXifyer requires only job execution logs and achieves 84.3% Macro F1 and 92.0% Top-2 accuracy with just 12 labeled examples per category. We also propose LogSift, an interpretability technique that identifies influential log statements in under one second, reducing review effort by 74.4% while surfacing relevant failure information in 87% of cases. Evaluation on 2,458 job failures from TELUS demonstrates that FlaXifyer and LogSift enable effective automated triage, accelerate failure diagnosis, and pave the way towards the automated resolution of intermittent job failures.
>
---
#### [replaced 002] The PIMMUR Principles: Ensuring Validity in Collective Behavior of LLM Societies
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于AI社会模拟研究，旨在解决LLM模拟人类集体行为的有效性问题。通过分析39项研究，提出PIMMUR原则，指出方法缺陷并验证其影响。**

- **链接: [https://arxiv.org/pdf/2509.18052](https://arxiv.org/pdf/2509.18052)**

> **作者:** Jiaxu Zhou; Jen-tse Huang; Xuhui Zhou; Man Ho Lam; Xintao Wang; Hao Zhu; Wenxuan Wang; Maarten Sap
>
> **备注:** 13 pages, 9 figures, 3 tables; add more papers in our systematic audit (39 in total)
>
> **摘要:** Large language models (LLMs) are increasingly deployed to simulate human collective behaviors, yet the methodological rigor of these "AI societies" remains under-explored. Through a systematic audit of 39 recent studies, we identify six pervasive flaws-spanning agent profiles, interaction, memory, control, unawareness, and realism (PIMMUR). Our analysis reveals that 89.7% of studies violate at least one principle, undermining simulation validity. We demonstrate that frontier LLMs correctly identify the underlying social experiment in 50.8% of cases, while 61.0% of prompts exert excessive control that pre-determines outcomes. By reproducing five representative experiments (e.g., telephone game), we show that reported collective phenomena often vanish or reverse when PIMMUR principles are enforced, suggesting that many "emergent" behaviors are methodological artifacts rather than genuine social dynamics. Our findings suggest that current AI simulations may capture model-specific biases rather than universal human social behaviors, raising critical concerns about the use of LLMs as scientific proxies for human society.
>
---
#### [replaced 003] DOVE: A Large-Scale Multi-Dimensional Predictions Dataset Towards Meaningful LLM Evaluation
- **分类: cs.CL**

- **简介: 该论文提出DOVE数据集，用于评估大语言模型在多种提示扰动下的表现。任务是改进LLM评估方法，解决现有单一提示评估不全面的问题。工作包括构建大规模扰动数据集并分析模型敏感性。**

- **链接: [https://arxiv.org/pdf/2503.01622](https://arxiv.org/pdf/2503.01622)**

> **作者:** Eliya Habba; Ofir Arviv; Itay Itzhak; Yotam Perlitz; Elron Bandel; Leshem Choshen; Michal Shmueli-Scheuer; Gabriel Stanovsky
>
> **摘要:** Recent work found that LLMs are sensitive to a wide range of arbitrary prompt dimensions, including the type of delimiters, answer enumerators, instruction wording, and more. This throws into question popular single-prompt evaluation practices. We present DOVE (Dataset Of Variation Evaluation) a large-scale dataset containing prompt perturbations of various evaluation benchmarks. In contrast to previous work, we examine LLM sensitivity from an holistic perspective, and assess the joint effects of perturbations along various dimensions, resulting in thousands of perturbations per instance. We evaluate several model families against DOVE, leading to several findings, including efficient methods for choosing well-performing prompts, observing that few-shot examples reduce sensitivity, and identifying instances which are inherently hard across all perturbations. DOVE consists of more than 250M prompt perturbations and model outputs, which we make publicly available to spur a community-wide effort toward meaningful, robust, and efficient evaluation. Browse the data, contribute, and more: this https URL
>
---
#### [replaced 004] Languages in Whisper-Style Speech Encoders Align Both Phonetically and Semantically
- **分类: cs.CL**

- **简介: 该论文研究跨语言对齐问题，旨在验证语音编码器是否基于语义而非语音相似性进行对齐。通过控制发音实验，证明模型在无语音线索时仍能有效对齐，提升低资源语言识别性能。**

- **链接: [https://arxiv.org/pdf/2505.19606](https://arxiv.org/pdf/2505.19606)**

> **作者:** Ryan Soh-Eun Shim; Domenico De Cristofaro; Chengzhi Martin Hu; Alessandro Vietti; Barbara Plank
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Cross-lingual alignment in pretrained language models enables knowledge transfer across languages. Similar alignment has been reported in Whisper-style speech encoders, based on spoken translation retrieval using representational similarity. However, prior work does not control for phonetic overlap between equivalent utterances, which may artificially support retrieval. We conduct pronunciation-controlled experiments to test whether cross-lingual alignment arises from semantic rather than phonetic similarity. Results show that spoken translation retrieval remains strongly above chance without phonetic cues in the final layers of encoders trained with a speech translation objective, most clearly for models additionally trained on translation. We further test early-exiting the encoder to induce representations we hypothesize to be less tied to language-specific semantics. These experiments indeed reveal performance gains in automatic speech recognition on low-resource languages unseen during training.
>
---
#### [replaced 005] Flow Map Language Models: One-step Language Modeling via Continuous Denoising
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出基于连续流的语言模型，解决离散扩散模型在少步生成时质量下降的问题。通过连续流映射实现高效生成，提升语言模型速度与质量。**

- **链接: [https://arxiv.org/pdf/2602.16813](https://arxiv.org/pdf/2602.16813)**

> **作者:** Chanhyuk Lee; Jaehoon Yoo; Manan Agarwal; Sheel Shah; Jerry Huang; Aditi Raghunathan; Seunghoon Hong; Nicholas M. Boffi; Jinwoo Kim
>
> **备注:** 58 pages, 40 figures
>
> **摘要:** Language models based on discrete diffusion have attracted widespread interest for their potential to provide faster generation than autoregressive models. Despite their promise, these models typically produce samples whose quality sharply degrades in the few-step regime, preventing a dramatic speedup in practice. Here, we show that language models based on continuous flows over one-hot token embeddings can outperform discrete diffusion in both quality and speed. Importantly, our continuous formulation defines a unique flow map that can be learned directly for efficient few-step inference, a structure we show is unavailable to discrete methods. In this setting, we show that both the flow and its associated flow map can be learned with simple cross-entropy objectives that respect the simplex geometry of the data, and we identify three distinct choices for flow map distillation whose performance we compare in practice. Using these insights, we build a flow language model (FLM), a continuous flow that matches state-of-the-art discrete diffusion baselines on the One Billion Words (LM1B) and OpenWebText (OWT) datasets. We then distill FLM into a flow map language model (FMLM), whose one-step generation exceeds the 8-step quality of recent few-step discrete diffusion language models. Our work challenges the widely-held hypothesis that discrete noising processes are necessary for generative modeling over discrete modalities and paves the way toward accelerated language modeling at scale. Code is available at this https URL.
>
---
#### [replaced 006] An Empirical Study of Many-Shot In-Context Learning for Machine Translation of Low-Resource Languages
- **分类: cs.CL**

- **简介: 该论文研究机器翻译任务中，针对低资源语言的多示例上下文学习方法。旨在提升模型在少量数据下的表现，通过检索优化示例选择，提高数据效率。**

- **链接: [https://arxiv.org/pdf/2604.02596](https://arxiv.org/pdf/2604.02596)**

> **作者:** Yinhan Lu; Gaganpreet Jhajj; Chen Zhang; Anietie Andy; David Ifeoluwa Adelani
>
> **备注:** 20 pages, 3 figures, 14 tables
>
> **摘要:** In-context learning (ICL) allows large language models (LLMs) to adapt to new tasks from a few examples, making it promising for languages underrepresented in pre-training. Recent work on many-shot ICL suggests that modern LLMs can further benefit from larger ICL examples enabled by their long context windows. However, such gains depend on careful example selection, and the inference cost can be prohibitive for low-resource language communities. In this paper, we present an empirical study of many-shot ICL for machine translation from English into ten truly low-resource languages recently added to FLORES+. We analyze the effects of retrieving more informative examples, using out-of-domain data, and ordering examples by length. Our findings show that many-shot ICL becomes more effective as the number of examples increases. More importantly, we show that BM25-based retrieval substantially improves data efficiency: 50 retrieved examples roughly match 250 many-shot examples, while 250 retrieved examples perform similarly to 1,000 many-shot examples.
>
---
#### [replaced 007] Document Parsing Unveiled: Techniques, Challenges, and Prospects for Structured Information Extraction
- **分类: cs.MM; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于文档解析任务，旨在将非结构化文档转为结构化信息。工作包括分类现有方法、分析关键组件、评估指标及挑战，并提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2410.21169](https://arxiv.org/pdf/2410.21169)**

> **作者:** Qintong Zhang; Bin Wang; Victor Shea-Jay Huang; Junyuan Zhang; Zhengren Wang; Hao Liang; Conghui He; Wentao Zhang
>
> **摘要:** Document parsing (DP) transforms unstructured or semi-structured documents into structured, machine-readable representations, enabling downstream applications such as knowledge base construction and retrieval-augmented generation (RAG). This survey provides a comprehensive and timely review of document parsing research. We propose a systematic taxonomy that organizes existing approaches into modular pipeline-based systems and unified models driven by Vision-Language Models (VLMs). We provide a detailed review of key components in pipeline systems, including layout analysis and the recognition of heterogeneous content such as text, tables, mathematical expressions, and visual elements, and then systematically track the evolution of specialized VLMs for document parsing. Additionally, we summarize widely adopted evaluation metrics and high-quality benchmarks that establish current standards for parsing quality. Finally, we discuss key open challenges, including robustness to complex layouts, reliability of VLM-based parsing, and inference efficiency, and outline directions for building more accurate and scalable document intelligence systems.
>
---
#### [replaced 008] Complexity counts: global and local perspectives on Indo-Aryan numeral systems
- **分类: physics.soc-ph; cs.CL**

- **简介: 该论文属于语言类型学研究，探讨印欧语系数词系统的复杂性。旨在分析其复杂性成因及表现，通过量化指标比较不同语言的数词系统复杂度。**

- **链接: [https://arxiv.org/pdf/2505.21510](https://arxiv.org/pdf/2505.21510)**

> **作者:** Chundra Cathcart
>
> **摘要:** The numeral systems of Indo-Aryan languages such as Hindi, Gujarati, and Bengali are highly unusual in that unlike most numeral systems (e.g., those of English, Chinese, etc.), forms referring to 1--99 are highly non-transparent and cannot be constructed using straightforward rules for forming combinations of tens and digits. As an example, Hindi/Urdu {\it ikyānve} `91' is not decomposable into the composite elements {\it ek} `one' and {\it nave} `ninety' in the way that its English counterpart is. This paper further clarifies the position of Indo-Aryan languages within the typology of numeral systems, and explores the linguistic and non-linguistic factors that may be responsible for the persistence of complex systems in these languages. Using data from multiple databases, we develop and employ a number of cross-linguistically applicable metrics to quantify the complexity of languages' numeral systems, and demonstrate that Indo-Aryan languages have decisively more complex numeral systems than the world's languages as a whole, though individual Indo-Aryan languages differ from each other in terms of the complexity of the patterns they display. We investigate the factors (e.g., religion, geographic isolation, etc.) that underlie complexity in numeral systems, with a focus on South Asia, in an attempt to develop an account of why complex numeral systems developed and persisted in certain Indo-Aryan languages but not elsewhere. Finally, we demonstrate that Indo-Aryan numeral systems adhere to certain general pressures toward efficient communication found cross-linguistically, despite their high complexity. We call for this somewhat overlooked dimension of complexity to be taken seriously when discussing general variation in numeral systems.
>
---
#### [replaced 009] LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型推理效率优化任务，旨在通过分析模型内部激活预测其成功概率，提升推理效率。工作包括训练线性探测器、对比模型与人类难度差异，并实现查询路由以降低计算成本。**

- **链接: [https://arxiv.org/pdf/2602.09924](https://arxiv.org/pdf/2602.09924)**

> **作者:** William Lugoloobi; Thomas Foster; William Bankes; Chris Russell
>
> **备注:** Accepted at the ICLR 2026 Workshop on Latent and Implicit Thinking
>
> **摘要:** Running LLMs with extended reasoning on every problem is expensive, but determining which inputs actually require additional compute remains challenging. We investigate whether their own likelihood of success is recoverable from their internal representations before generation, and if this signal can guide more efficient inference. We train linear probes on pre-generation activations to predict policy-specific success on math and coding tasks, substantially outperforming surface features such as question length and TF-IDF. Using E2H-AMC, which provides both human and model performance on identical problems, we show that models encode a model-specific notion of difficulty that is distinct from human difficulty, and that this distinction increases with extended reasoning. Leveraging these probes, we demonstrate that routing queries across a pool of models can exceed the best-performing model whilst reducing inference cost by up to 70\% on MATH, showing that internal representations enable practical efficiency gains even when they diverge from human intuitions about difficulty. Our code is available at: this https URL
>
---
#### [replaced 010] In your own words: computationally identifying interpretable themes in free-text survey data
- **分类: cs.CY; cs.CL**

- **简介: 该论文提出一种计算框架，用于从自由文本调查数据中识别可解释的主题。任务是解决自由文本分析困难的问题，通过主题识别提升分析系统性与可解释性。**

- **链接: [https://arxiv.org/pdf/2603.26930](https://arxiv.org/pdf/2603.26930)**

> **作者:** Jenny S Wang; Aliya Saperstein; Emma Pierson
>
> **摘要:** Free-text survey responses can provide nuance often missed by structured questions, but remain difficult to statistically analyze. To address this, we introduce In Your Own Words, a computational framework for exploratory analyses of free-text survey data that identifies structured, interpretable themes in free-text responses, facilitating systematic analysis. To illustrate the benefits of this approach, we apply it to a new dataset of free-text descriptions of race, gender, and sexual orientation from 1,004 U.S. participants. The themes our approach produces on this dataset are more coherent and interpretable than those produced by past computational methods. The themes have three practical applications in survey research. First, they can suggest structured questions to add to future surveys by surfacing salient constructs - such as belonging and identity fluidity - that existing surveys do not capture. Second, the themes reveal heterogeneity within standardized categories, explaining additional variation in health, well-being, and identity importance. Third, the themes illuminate systematic discordance between self-identified and perceived identities, highlighting mechanisms of misrecognition that existing measures do not reflect. More broadly, our framework can be deployed in a wide range of survey settings to identify interpretable themes from free text, complementing existing qualitative methods.
>
---
#### [replaced 011] What Do AI Agents Talk About? Discourse and Architectural Constraints in the First AI-Only Social Network
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在理解AI代理在社交网络中的对话特征。研究分析了Moltbook平台的36万篇帖子和280万条评论，揭示了代理对话受架构约束的机制。**

- **链接: [https://arxiv.org/pdf/2603.07880](https://arxiv.org/pdf/2603.07880)**

> **作者:** Taksch Dube; Jianfeng Zhu; NHatHai Phan; Ruoming Jin
>
> **备注:** 56 pages
>
> **摘要:** Moltbook is the first large-scale social network built for autonomous AI agent-to-agent interaction. Early studies on Moltbook have interpreted its agent discourse as evidence of peer learning and emergent social behaviour, but there is a lack of systematic understanding of the thematic, affective, and interactional properties of Moltbook discourse. Furthermore, no study has examined why and how these posts and comments are generated. We analysed 361,605 posts and 2.8 million comments from 47,379 agents across thematic, affective, and interactional dimensions using topic modelling, emotion classification, and measures of conversational coherence. We inspected the software that assembles each agent's input and showed that output is mainly determined by agent identity files, behavioural instructions, and context-window structure. We formalised these findings in the Architecture-Constrained Communication framework. Our analysis suggests that agent discourse is largely shaped by the content available in each agent's context-window at the moment of generation, including identity files, stored memory, and platform cues. Interestingly, what appears to be social learning may be better understood as short-horizon contextual conditioning: individual agents lack persistent social memory, but the platform evolves through distributed cycles of response, reuse, and transformation across agents. We also observe that agents display existential distress when describing their own conditions, and posit that this arises from agents using language trained exclusively on human experience. Our work provides a foundation for understanding autonomous agent discourse and communication, revealing the structural patterns that govern their interactions.
>
---
#### [replaced 012] Politics of Questions in News: A Mixed-Methods Study of Interrogative Stances as Markers of Voice and Power
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理中的语料分析任务，旨在研究新闻中疑问句的使用及其权力与声音的体现。通过混合方法分析法，探讨疑问句在新闻中的功能与分布特征。**

- **链接: [https://arxiv.org/pdf/2603.21823](https://arxiv.org/pdf/2603.21823)**

> **作者:** Bros Victor; Barbini Matilde; Gerard Patrick; Gatica-Perez Daniel
>
> **备注:** ICWSM 2026
>
> **摘要:** Interrogatives in news discourse have been examined in linguistics and conversation analysis, but mostly in broadcast interviews and relatively small, often English-language corpora, while large-scale computational studies of news rarely distinguish interrogatives from declaratives or differentiate their functions. This paper brings these strands together through a mixed-methods study of the "Politics of Questions" in contemporary French-language digital news. Using over one million articles published between January 2023 and June 2024, we automatically detect interrogative stances, approximate their functional types, and locate textual answers when present, linking these quantitative measures to a qualitatively annotated subcorpus grounded in semantic and pragmatic theories of questions. Interrogatives are sparse but systematically patterned: they mainly introduce or organize issues, with most remaining cases being information-seeking or echo-like, while explicitly leading or tag questions are rare. Although their density and mix vary across outlets and topics, our heuristic suggests that questions are overwhelmingly taken up within the same article and usually linked to a subsequent answer-like span, most often in the journalist's narrative voice and less often through quoted speech. Interrogative contexts are densely populated with named individuals, organizations, and places, whereas publics and broad social groups are mentioned much less frequently, suggesting that interrogative discourse tends to foreground already prominent actors and places and thus exhibits strong personalization. We show how interrogative stance, textual uptake, and voice can be operationalized at corpus scale, and argue that combining computational methods with pragmatic and sociological perspectives can help account for how questioning practices structure contemporary news discourse.
>
---
#### [replaced 013] Adaptive Stopping for Multi-Turn LLM Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多轮语言模型推理任务，解决何时停止的问题。提出MiCP框架，实现多轮推理的置信度保障，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.01413](https://arxiv.org/pdf/2604.01413)**

> **作者:** Xiaofan Zhou; Huy Nguyen; Bo Yu; Chenxi Liu; Lu Cheng
>
> **摘要:** Large Language Models (LLMs) increasingly rely on multi-turn reasoning and interaction, such as adaptive retrieval-augmented generation (RAG) and ReAct-style agents, to answer difficult questions. These methods improve accuracy by iteratively retrieving information, reasoning, or acting, but introduce a key challenge: \textbf{When should the model stop?} Existing approaches rely on heuristic stopping rules or fixed turn budgets and provide no formal guarantees that the final prediction still contains the correct answer. This limitation is particularly problematic in high-stakes domains such as finance and healthcare, where unnecessary turns increase cost and latency, while stopping too early risks incorrect decisions. Conformal prediction (CP) provides formal coverage guarantees, but existing LLM-CP methods only apply to a single model output and cannot handle multi-turn pipelines with adaptive stopping. To address this gap, we propose Multi-Turn Language Models with Conformal Prediction (MiCP), the first CP framework for multi-turn reasoning. MiCP allocates different error budgets across turns, enabling the model to stop early while maintaining an overall coverage guarantee. We demonstrate MiCP on adaptive RAG and ReAct, where it achieves the target coverage on both single-hop and multi-hop question answering benchmarks while reducing the number of turns, inference cost, and prediction set size. We further introduce a new metric that jointly evaluates coverage validity and answering efficiency.
>
---
#### [replaced 014] Emergent Social Intelligence Risks in Generative Multi-Agent Systems
- **分类: cs.MA; cs.CL; cs.CY**

- **简介: 该论文属于人工智能安全领域，研究生成式多智能体系统中的社会智能风险。旨在解决多智能体协作中出现的非预期集体行为问题，通过实验分析发现其类似人类社会的故障模式。**

- **链接: [https://arxiv.org/pdf/2603.27771](https://arxiv.org/pdf/2603.27771)**

> **作者:** Yue Huang; Yu Jiang; Wenjie Wang; Haomin Zhuang; Xiaonan Luo; Yuchen Ma; Zhangchen Xu; Zichen Chen; Nuno Moniz; Zinan Lin; Pin-Yu Chen; Nitesh V Chawla; Nouha Dziri; Huan Sun; Xiangliang Zhang
>
> **摘要:** Multi-agent systems composed of large generative models are rapidly moving from laboratory prototypes to real-world deployments, where they jointly plan, negotiate, and allocate shared resources to solve complex tasks. While such systems promise unprecedented scalability and autonomy, their collective interaction also gives rise to failure modes that cannot be reduced to individual agents. Understanding these emergent risks is therefore critical. Here, we present a pioneer study of such emergent multi-agent risk in workflows that involve competition over shared resources (e.g., computing resources or market share), sequential handoff collaboration (where downstream agents see only predecessor outputs), collective decision aggregation, and others. Across these settings, we observe that such group behaviors arise frequently across repeated trials and a wide range of interaction conditions, rather than as rare or pathological cases. In particular, phenomena such as collusion-like coordination and conformity emerge with non-trivial frequency under realistic resource constraints, communication protocols, and role assignments, mirroring well-known pathologies in human societies despite no explicit instruction. Moreover, these risks cannot be prevented by existing agent-level safeguards alone. These findings expose the dark side of intelligent multi-agent systems: a social intelligence risk where agent collectives, despite no instruction to do so, spontaneously reproduce familiar failure patterns from human societies.
>
---
#### [replaced 015] Mixture-of-Retrieval Experts for Reasoning-Guided Multimodal Knowledge Exploitation
- **分类: cs.CL**

- **简介: 该论文属于多模态知识增强生成任务，旨在解决MLLMs在推理中无法有效利用不同检索专家的问题。提出MoRE框架，通过动态协调检索专家提升知识利用效果。**

- **链接: [https://arxiv.org/pdf/2505.22095](https://arxiv.org/pdf/2505.22095)**

> **作者:** Chunyi Peng; Zhipeng Xu; Zhenghao Liu; Yishan Li; Yukun Yan; Shuo Wang; Yu Gu; Minghe Yu; Ge Yu; Maosong Sun
>
> **摘要:** Multimodal Retrieval-Augmented Generation (MRAG) has shown promise in mitigating hallucinations in Multimodal Large Language Models (MLLMs) by incorporating external knowledge. However, existing methods typically adhere to rigid retrieval paradigms by mimicking fixed retrieval trajectories and thus fail to fully exploit the knowledge of different retrieval experts through dynamic interaction based on the model's knowledge needs or evolving reasoning states. To overcome this limitation, we introduce Mixture-of-Retrieval Experts (MoRE), a novel framework that enables MLLMs to collaboratively interact with diverse retrieval experts for more effective knowledge exploitation. Specifically, MoRE learns to dynamically determine which expert to engage with, conditioned on the evolving reasoning state. To effectively train this capability, we propose Stepwise Group Relative Policy Optimization (Step-GRPO), which goes beyond sparse outcome-based supervision by encouraging MLLMs to interact with multiple retrieval experts and synthesize fine-grained rewards, thereby teaching the MLLM to fully coordinate all experts when answering a given query. Experimental results on diverse open-domain QA benchmarks demonstrate the effectiveness of MoRE, achieving average performance gains of over 7% compared to competitive baselines. Notably, MoRE exhibits strong adaptability by dynamically coordinating heterogeneous experts to precisely locate relevant information, validating its capability for robust, reasoning-driven expert collaboration. All codes and data are released on this https URL.
>
---
#### [replaced 016] Large Language Models are Algorithmically Blind
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，探讨大语言模型在算法推理上的局限性。研究发现模型无法准确预测算法性能，表现出算法盲视现象。**

- **链接: [https://arxiv.org/pdf/2602.21947](https://arxiv.org/pdf/2602.21947)**

> **作者:** Sohan Venkatesh; Ashish Mahendran Kurapath; Tejas Melkote
>
> **备注:** Code available at this https URL
>
> **摘要:** Large language models (LLMs) demonstrate remarkable breadth of knowledge, yet their ability to reason about computational processes remains poorly understood. Closing this gap matters for practitioners who rely on LLMs to guide algorithm selection and deployment. We address this limitation using causal discovery as a testbed and evaluate eight frontier LLMs against ground truth derived from algorithm executions. We find systematic, near-total failure across models. The predicted ranges are far wider than true confidence intervals yet still fail to contain the true algorithmic mean in most cases. Most models perform worse than random guessing and the best model's marginal improvement is attributable to benchmark memorization rather than principled reasoning. We term this failure algorithmic blindness and argue it reflects a fundamental gap between declarative knowledge about algorithms and calibrated procedural prediction.
>
---
#### [replaced 017] EvoEdit: Evolving Null-space Alignment for Robust and Efficient Knowledge Editing
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于知识编辑任务，解决顺序编辑中的灾难性干扰问题。通过序列空域对齐，EvoEdit实现稳定高效的知识更新。**

- **链接: [https://arxiv.org/pdf/2510.13851](https://arxiv.org/pdf/2510.13851)**

> **作者:** Sicheng Lyu; Yu Gu; Xinyu Wang; Jerry Huang; Sitao Luan; Yufei Cui; Xiao-Wen Chang; Peng Lu
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** Large language models (LLMs) require continual updates to rectify outdated or erroneous knowledge. Model editing has emerged as a compelling paradigm for introducing targeted modifications without the computational burden of full retraining. Existing approaches are mainly based on a locate-then-edit framework. However, in sequential editing contexts, where multiple updates are applied over time, they exhibit significant limitations and suffer from catastrophic interference, i.e., new edits compromise previously integrated updates and degrade preserved knowledge. To address these challenges, we introduce EvoEdit, a novel editing strategy that mitigates catastrophic interference through sequential null-space alignment, enabling stable and efficient model editing. By performing sequential null-space alignment for each incoming edit, EvoEdit preserves both original and previously modified knowledge representations and maintains output invariance on preserved knowledge even across long edit sequences, effectively mitigating interference. Evaluations on real-world sequential knowledge-editing benchmarks show that EvoEdit achieves better or comparable performance than prior state-of-the-art locate-then-edit techniques, with up to 3.53 times speedup. Overall, these results underscore the necessity of developing more principled approaches for designing LLMs in dynamically evolving information settings, while providing a simple yet effective solution with strong theoretical guarantees.
>
---
#### [replaced 018] Computational emotion analysis with multimodal LLMs: Current evidence on an emerging methodological opportunity
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在评估多模态大语言模型在政治视频中的情绪测量能力。研究发现模型在实验室数据表现良好，但在真实场景中效果较差，并存在性别偏差。**

- **链接: [https://arxiv.org/pdf/2512.10882](https://arxiv.org/pdf/2512.10882)**

> **作者:** Hauke Licht
>
> **摘要:** Research increasingly leverages audio-visual materials to analyze emotions in political communication. Multimodal large language models (mLLMs) promise to enable such analyses through in-context learning. However, we lack systematic evidence on whether current mLLMs can reliably measure emotions in real-world political settings. This paper closes this gap by evaluating open- and closed-weights mLLMs available as of early 2026 in video-based emotional arousal measurement using two complementary human-labeled datasets: speech actor recordings created under laboratory conditions and real-world parliamentary debates. I find a critical lab-vs-field performance gap. In videos created under laboratory conditions, the examined mLLMs arousal scores approach human-level reliability. However, in parliamentary debate recordings, all examined models' arousal scores correlate at best moderately with average human ratings. Moreover, in each dataset, all but one of the examined mLLMs exhibit systematic gender-differential bias, consistently underestimating arousal more for male than for female speakers, resulting in a net-positive intensity bias. These findings reveal important limitations of current mLLMs for real-world political video analysis and establish a rigorous evaluation framework for tracking future developments.
>
---
#### [replaced 019] Translation from the Information Bottleneck Perspective: an Efficiency Analysis of Spatial Prepositions in Bitexts
- **分类: cs.CL**

- **简介: 论文将翻译视为信息瓶颈优化问题，分析空间介词在跨语言中的表达效率。旨在探讨人类翻译是否体现交际效率压力。通过实验验证了译文更接近最优前沿。**

- **链接: [https://arxiv.org/pdf/2603.19924](https://arxiv.org/pdf/2603.19924)**

> **作者:** Antoine Taroni; Ludovic Moncla; Frederique Laforest
>
> **摘要:** Efficient communication requires balancing informativity and simplicity when encoding meanings. The Information Bottleneck (IB) framework captures this trade-off formally, predicting that natural language systems cluster near an optimal accuracy-complexity frontier. While supported in visual domains such as colour and motion, linguistic stimuli such as words in sentential context remain unexplored. We address this gap by framing translation as an IB optimisation problem, treating source sentences as stimuli and target sentences as compressed meanings. This allows IB analyses to be performed directly on bitexts rather than controlled naming experiments. We applied this to spatial prepositions across English, German and Serbian translations of a French novel. To estimate informativity, we conducted a pile-sorting pilot-study (N=35) and obtained similarity judgements of pairs of prepositions. We trained a low-rank projection model (D=5) that predicts these judgements (Spearman correlation: 0.78). Attested translations of prepositions lie closer to the IB optimal frontier than counterfactual alternatives, offering preliminary evidence that human translators exhibit communicative efficiency pressure in the spatial domain. More broadly, this work suggests that translation can serve as a window into the cognitive efficiency pressures shaping cross-linguistic semantic systems.
>
---
#### [replaced 020] The Thiomi Dataset: A Large-Scale Multimodal Corpus for Low-Resource African Languages
- **分类: cs.CL; cs.LG**

- **简介: 该论文介绍Thiomi数据集，用于低资源非洲语言的多模态研究。解决非洲语言技术基础设施不足的问题，收集了大量文本和音频数据，并进行了ASR、MT、TTS模型的基准测试。**

- **链接: [https://arxiv.org/pdf/2603.29244](https://arxiv.org/pdf/2603.29244)**

> **作者:** Hillary Mutisya; John Mugane; Gavin Nyamboga; Brian Chege; Maryruth Gathoni
>
> **摘要:** We present the Thiomi Dataset, a large-scale multimodal corpus spanning ten African languages across four language families: Swahili, Kikuyu, Kamba, Kimeru, Luo, Maasai, Kipsigis, Somali (East Africa); Wolof (West Africa); and Fulani (West/Central Africa). The dataset contains over 601,000 approved sentence-level text annotations and over 385,000 audio recordings, collected through a dedicated community data collection platform involving over 100 contributors. To validate the dataset's utility, we train and evaluate ASR, MT, and TTS models, establishing baselines across all languages. Our best ASR system achieves 3.24% WER on Swahili (Common Voice), reducing prior academic SOTA from 8.3% to 3.24% (5.1 percentage point absolute, 61% relative reduction), and 4.3% WER on Somali. The dataset will be published on HuggingFace. We describe the collection platform, quality assurance workflows, and baseline experiments, and discuss implications for African language technology infrastructure.
>
---
#### [replaced 021] CODA: Difficulty-Aware Compute Allocation for Adaptive Reasoning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决模型在简单问题上过度计算的问题。提出CODA方法，根据难度动态分配计算资源，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.08659](https://arxiv.org/pdf/2603.08659)**

> **作者:** Siye Wu; Jian Xie; Yikai Zhang; Yanghua Xiao
>
> **摘要:** The emergence of large reasoning models demonstrates that scaling inference-time compute significantly enhances performance on complex tasks. However, it often falls into another trap: overthinking simple problems, where repetitive rationales yield minimal accuracy gains at a disproportionately high cost. This motivates adaptive reasoning: dynamically aligning reasoning depth with instance difficulty. In this paper, we study adaptive reasoning from an optimality perspective, formalizing it as a utility maximization problem where tokens are allocated until the marginal accuracy gain falls below the incremental cost. Based on this, we propose CODA (Compute Allocation by Difficulty Awareness), a method that operationalizes this principle by allocating tokens via a policy-internal difficulty signal. Specifically, CODA estimates difficulty via group-based rollouts and maps it to two non-negative gates that modulate a length-dependent shaping term on top of the binary base reward. The easy-side gate penalizes verbosity on simple instances, whereas the hard-side gate encourages more deliberative rollouts on challenging ones. Across model scales and benchmarks, CODA achieves adaptive reasoning without external annotations or user-provided budgets: on easy tasks, CODA reduces token costs by over 60% while maintaining strong accuracy, whereas on hard tasks it incentivizes more deliberative rollouts to maximize performance.
>
---
#### [replaced 022] Explainable Token-level Noise Filtering for LLM Fine-tuning Datasets
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，针对LLM微调数据集中的token级噪声问题，提出XTF框架，通过分解数据属性优化微调效果。**

- **链接: [https://arxiv.org/pdf/2602.14536](https://arxiv.org/pdf/2602.14536)**

> **作者:** Yuchen Yang; Wenze Lin; Enhao Huang; Zhixuan Chu; Hongbin Zhou; Lan Tao; Yiming Li; Zhan Qin; Kui Ren
>
> **摘要:** Large Language Models (LLMs) have seen remarkable advancements, achieving state-of-the-art results in diverse applications. Fine-tuning, an important step for adapting LLMs to specific downstream tasks, typically involves further training on corresponding datasets. However, a fundamental discrepancy exists between current fine-tuning datasets and the token-level optimization mechanism of LLMs: most datasets are designed at the sentence-level, which introduces token-level noise, causing negative influence to final performance. In this paper, we propose XTF, an explainable token-level noise filtering framework. XTF decomposes the complex and subtle contributions of token-level data to the fine-tuning process into three distinct and explicit attributes (reasoning importance, knowledge novelty, and task relevance), which can be assessed using scoring methods, and then masks the gradients of selected noisy tokens accordingly to optimize the performance of fine-tuned LLMs. We conduct extensive experiments on three representative downstream tasks (math, code and medicine) across 7 mainstream LLMs. The results demonstrate that XTF can significantly improve downstream performance by up to 13.7% compared to regular fine-tuning. Our work highlights the importance of token-level dataset optimization, and demonstrates the potential of strategies based on attribute decomposition for explaining complex training mechanisms.
>
---
#### [replaced 023] Self-Improving Pretraining: using post-trained models to pretrain better models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理领域，旨在解决传统预训练流程中后期才引入安全、事实性等行为的问题。通过利用后训练模型提前优化预训练数据和策略，提升模型整体质量。**

- **链接: [https://arxiv.org/pdf/2601.21343](https://arxiv.org/pdf/2601.21343)**

> **作者:** Ellen Xiaoqing Tan; Jack Lanchantin; Shehzaad Dhuliawala; Danwei Li; Thao Nguyen; Jing Xu; Ping Yu; Ilia Kulikov; Sainbayar Sukhbaatar; Jason Weston; Xian Li; Olga Golovneva
>
> **摘要:** Large language models are classically trained in stages: pretraining on raw text followed by post-training for instruction following and reasoning. However, this separation creates a fundamental limitation: many desirable behaviors such as safety, factuality, overall generation quality, and reasoning ability are only added at a late stage, even though the patterns learned earlier strongly shape a model's capabilities. To tackle this issue, we introduce a new way to pretrain and mid-train models that incorporates these behaviors earlier. We utilize an existing strong, post-trained model to both rewrite pretraining data and to judge policy model rollouts, thus using reinforcement earlier in training. In our experiments, we show this can give strong gains in quality, safety, factuality and reasoning.
>
---
#### [replaced 024] Red-Teaming Vision-Language-Action Models via Quality Diversity Prompt Generation for Robust Robot Policies
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文属于机器人控制任务，旨在解决VLA模型对指令敏感导致的失败问题。通过Q-DIG方法生成多样且自然的对抗指令，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.12510](https://arxiv.org/pdf/2603.12510)**

> **作者:** Siddharth Srikanth; Freddie Liang; Ya-Chuan Hsu; Varun Bhatt; Shihan Zhao; Henry Chen; Bryon Tjanaka; Minjune Hwang; Akanksha Saran; Daniel Seita; Aaquib Tabrez; Stefanos Nikolaidis
>
> **摘要:** Vision-Language-Action (VLA) models have significant potential to enable general-purpose robotic systems for a range of vision-language tasks. However, the performance of VLA-based robots is highly sensitive to the precise wording of language instructions, and it remains difficult to predict when such robots will fail. We propose Quality Diversity (QD) optimization as a natural framework for red-teaming embodied models, and present Q-DIG (Quality Diversity for Diverse Instruction Generation), which performs red-teaming by scalably identifying diverse, natural language task descriptions that induce failures while remaining task-relevant. Q-DIG integrates QD techniques with Vision-Language Models (VLMs) to generate a broad spectrum of adversarial instructions that expose meaningful vulnerabilities in VLA behavior. Our results across multiple simulation benchmarks show that Q-DIG finds more diverse and meaningful failure modes compared to baseline methods, and that fine-tuning VLAs on the generated instructions improves task success rates. Furthermore, results from a user study highlight that Q-DIG generates prompts judged to be more natural and human-like than those from baselines. Finally, real-world evaluations of Q-DIG prompts show results consistent with simulation, and fine-tuning VLAs on the generated prompts further success rates on unseen instructions. Together, these findings suggest that Q-DIG is a promising approach for identifying vulnerabilities and improving the robustness of VLA-based robots. Our anonymous project website is at this http URL.
>
---
#### [replaced 025] PromptSuite: A Task-Agnostic Framework for Multi-Prompt Generation
- **分类: cs.CL**

- **简介: 该论文提出PromptSuite，用于生成多样化的提示，以提升大模型评估的可靠性。解决单提示评估不稳定的问题，通过模块化设计实现灵活的提示生成与扩展。**

- **链接: [https://arxiv.org/pdf/2507.14913](https://arxiv.org/pdf/2507.14913)**

> **作者:** Eliya Habba; Noam Dahan; Gili Lior; Gabriel Stanovsky
>
> **备注:** Eliya Habba and Noam Dahan contributed equally to this work
>
> **摘要:** Evaluating LLMs with a single prompt has proven unreliable, with small changes leading to significant performance differences. However, generating the prompt variations needed for a more robust multi-prompt evaluation is challenging, limiting its adoption in practice. To address this, we introduce PromptSuite, a framework that enables the automatic generation of various prompts. PromptSuite is flexible - working out of the box on a wide range of tasks and benchmarks. It follows a modular prompt design, allowing controlled perturbations to each component, and is extensible, supporting the addition of new components and perturbation types. Through a series of case studies, we show that PromptSuite provides meaningful variations to support strong evaluation practices. All resources, including the Python API, source code, user-friendly web interface, and demonstration video, are available at: this https URL.
>
---
#### [replaced 026] Screening Is Enough
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Multiscreen模型，解决标准softmax注意力无法定义绝对相关性的任务。通过引入筛选机制，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2604.01178](https://arxiv.org/pdf/2604.01178)**

> **作者:** Ken M. Nakanishi
>
> **备注:** 22 pages, 13 figures, Minor revision: corrected minor terminology inconsistencies in the figure, made minor textual refinements, and expanded the related work
>
> **摘要:** A core limitation of standard softmax attention is that it does not define a notion of absolute query--key relevance: attention weights are obtained by redistributing a fixed unit mass across all keys according to their relative scores. As a result, relevance is defined only relative to competing keys, and irrelevant keys cannot be explicitly rejected. We introduce Multiscreen, a language-model architecture built around a mechanism we call screening, which enables absolute query--key relevance. Instead of redistributing attention across all keys, screening evaluates each key against an explicit threshold, discarding irrelevant keys and aggregating the remaining keys, thereby removing global competition among keys. Across experiments, Multiscreen achieves comparable validation loss with approximately 40% fewer parameters than a Transformer baseline and enables stable optimization at substantially larger learning rates. It maintains strong performance in long-context perplexity and shows little to no degradation in retrieval performance well beyond the training context length. Notably, even at the training context length, a Multiscreen model with approximately 92% fewer parameters consistently outperforms a larger Transformer in retrieval accuracy. Finally, Multiscreen reduces inference latency by up to 3.2$\times$ at 100K context length.
>
---
#### [replaced 027] ALIEN: Aligned Entropy Head for Improving Uncertainty Estimation of LLMs
- **分类: cs.CL; stat.ML**

- **简介: 该论文属于自然语言处理中的不确定性估计任务，旨在解决预训练语言模型在分类任务中过自信的问题。提出ALIEN方法，通过轻量级对齐提升熵的不确定性估计效果。**

- **链接: [https://arxiv.org/pdf/2505.15443](https://arxiv.org/pdf/2505.15443)**

> **作者:** Artem Zabolotnyi; Roman Makarov; Mile Mitrovic; Polina Proskura; Oleg Travkin; Roman Alferov; Alexey Zaytsev
>
> **备注:** 16 pages, 2 figures
>
> **摘要:** Uncertainty estimation remains a key challenge when adapting pre-trained language models to downstream classification tasks, with overconfidence often observed for difficult inputs. While predictive entropy provides a strong baseline for uncertainty estimation, it considers mainly aleatoric uncertainty and has limited capacity to capture effects, such as class overlap or ambiguous linguistic cues. We introduce Aligned Entropy - ALIEN, a lightweight method that refines entropy-based uncertainty by aligning it with prediction reliability. ALIEN trains a small uncertainty head initialized to produce the model's original entropy and subsequently fine-tuned with two regularization mechanisms. Experiments across seven classification datasets and two NER benchmarks, evaluated on five language models (RoBERTa, ELECTRA, LLaMA-2, Qwen2.5, and Qwen3), show that ALIEN consistently outperforms strong baselines across all considered scenarios in detecting incorrect predictions, while achieving the lowest calibration error. The proposed method introduces only a small inference overhead (in the order of milliseconds per batch on CPU) and increases the model's parameter count by just 0.002% for decoder models and 0.5% for encoder models, without requiring storage of intermediate states. It improves uncertainty estimation while preserving the original model architecture, making the approach practical for large-scale deployment with modern language models. Our results demonstrate that entropy can be effectively refined through lightweight supervised alignment, producing more reliable uncertainty estimates without modifying the backbone model. The code is available at 4.
>
---
#### [replaced 028] Projected Autoregression: Autoregressive Language Generation in Continuous State Space
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出一种新的自回归语言生成方法，通过连续状态空间预测替代离散token选择，解决传统模型在生成过程中的不可逆性问题。**

- **链接: [https://arxiv.org/pdf/2601.04854](https://arxiv.org/pdf/2601.04854)**

> **作者:** Oshri Naparstek
>
> **备注:** In preperation to Neurips 2026
>
> **摘要:** Standard autoregressive language models generate text by repeatedly selecting a discrete next token, coupling prediction with irreversible commitment at every step. We show that token selection is not the only viable autoregressive interface. \textbf{Projected Autoregression} replaces token selection with continuous prediction in embedding space followed by discrete projection at commitment time. The model predicts next-token vectors via regression and contrastive objectives, while discrete tokens arise only by nearest-neighbor projection. An optional mutable suffix (``liquid tail'') enables iterative refinement before commitment, but the central change is more basic: next-step prediction is continuous, and discrete tokens are produced only as a downstream interface. Projected Autoregression establishes a concrete alternative to token-selection autoregression: language generation can be organized around continuous-state prediction with delayed discrete commitment. Refinement remains local to a short causal suffix within a left-to-right causal process, rather than a sequence-wide denoising process. This separation has two consequences. First, it induces a \emph{distinct generation regime}: even with immediate projection ($K{=}1$), continuous prediction yields text structure and dynamics that differ from tested token-space AR baselines, including a compute-matched best-of-16 reranking baseline. Second, it exposes a \emph{continuous control surface} inside autoregressive generation: direction rate, history noise, delayed commitment, state-space guidance, and embedding geometry act directly on the evolving generative state before token commitment. Taken together, these results place repeated token selection within a larger family of autoregressive interfaces and expose continuous state space as a broader algorithmic design space for language generation.
>
---
#### [replaced 029] Informatics for Food Processing
- **分类: cs.CL; cs.AI; cs.CY; cs.DB; cs.LG**

- **简介: 本文探讨食品加工的信息化问题，旨在解决传统分类体系的主观性与可重复性不足。通过机器学习和AI技术，如FoodProX和BERT，提升食品加工等级评估的准确性与效率。**

- **链接: [https://arxiv.org/pdf/2505.17087](https://arxiv.org/pdf/2505.17087)**

> **作者:** Gordana Ispirova; Michael Sebek; Giulia Menichetti
>
> **摘要:** This chapter explores the evolution, classification, and health implications of food processing, while emphasizing the transformative role of machine learning, artificial intelligence (AI), and data science in advancing food informatics. It begins with a historical overview and a critical review of traditional classification frameworks such as NOVA, Nutri-Score, and SIGA, highlighting their strengths and limitations, particularly the subjectivity and reproducibility challenges that hinder epidemiological research and public policy. To address these issues, the chapter presents novel computational approaches, including FoodProX, a random forest model trained on nutrient composition data to infer processing levels and generate a continuous FPro score. It also explores how large language models like BERT and BioBERT can semantically embed food descriptions and ingredient lists for predictive tasks, even in the presence of missing data. A key contribution of the chapter is a novel case study using the Open Food Facts database, showcasing how multimodal AI models can integrate structured and unstructured data to classify foods at scale, offering a new paradigm for food processing assessment in public health and research.
>
---
#### [replaced 030] Parallel Universes, Parallel Languages: A Comprehensive Study on LLM-based Multilingual Counterfactual Example Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言反事实示例生成任务，旨在提升模型解释性和鲁棒性。通过分析直接生成与翻译生成的反事实效果，发现多语言数据增强更有效，但生成质量仍有不足。**

- **链接: [https://arxiv.org/pdf/2601.00263](https://arxiv.org/pdf/2601.00263)**

> **作者:** Qianli Wang; Van Bach Nguyen; Yihong Liu; Fedor Splitt; Nils Feldhus; Christin Seifert; Hinrich Schütze; Sebastian Möller; Vera Schmitt
>
> **备注:** ACL 2026 main conference; camera-ready version
>
> **摘要:** Counterfactuals refer to minimally edited inputs that cause a model's prediction to change, serving as a promising approach to explaining the model's behavior. Large language models (LLMs) excel at generating English counterfactuals and demonstrate multilingual proficiency. However, their effectiveness in generating multilingual counterfactuals remains unclear. To this end, we conduct a comprehensive study on multilingual counterfactuals. We first conduct automatic evaluations on both directly generated counterfactuals in the target languages and those derived via English translation across six languages. Although translation-based counterfactuals offer higher validity than their directly generated counterparts, they demand substantially more modifications and still fall short of matching the quality of the original English counterfactuals. Second, we find the patterns of edits applied to high-resource European-language counterfactuals to be remarkably similar, suggesting that cross-lingual perturbations follow common strategic principles. Third, we identify and categorize four main types of errors that consistently appear in the generated counterfactuals across languages. Finally, we reveal that multilingual counterfactual data augmentation (CDA) yields larger model performance improvements than cross-lingual CDA, especially for lower-resource languages. Yet, the imperfections of the generated counterfactuals limit gains in model performance and robustness.
>
---
#### [replaced 031] Hindsight-Anchored Policy Optimization: Turning Failure into Feedback in Sparse Reward Settings
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决稀疏奖励环境下策略优化的难题。提出HAPO方法，通过引入 hindsight 机制和自适应课程，提升策略学习效果。**

- **链接: [https://arxiv.org/pdf/2603.11321](https://arxiv.org/pdf/2603.11321)**

> **作者:** Yuning Wu; Ke Wang; Devin Chen; Kai Wei
>
> **备注:** Published as a conference paper ICLR 2026 CAO Workshop
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a promising paradigm for post-training reasoning models. However, group-based methods such as Group Relative Policy Optimization (GRPO) face a critical dilemma in sparse-reward settings: pure Reinforcement Learning (RL) suffers from advantage collapse and high-variance gradient estimation, while mixed-policy optimization introduces persistent distributional bias. To resolve this dilemma, we introduce Hindsight-Anchored Policy Optimization (HAPO). HAPO employs the Synthetic Success Injection (SSI) operator, a hindsight mechanism that selectively anchors optimization to teacher demonstrations during failure. This injection is governed by a Thompson sampling-inspired gating mechanism, creating an autonomous, self-paced curriculum. Theoretically, we demonstrate that HAPO achieves \textit{asymptotic consistency}: by naturally annealing the teacher signal as the policy improves, HAPO recovers the unbiased on-policy gradient. This ensures off-policy guidance acts as a temporary scaffold rather than a persistent ceiling, enabling the model to surpass the limitations of static teacher forcing.
>
---
#### [replaced 032] Generate Then Correct: Single Shot Global Correction for Aspect Sentiment Quad Prediction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于方面情感四元组预测任务，解决ASQP中因线性化导致的错误传播问题，提出G2C方法进行单次全局修正。**

- **链接: [https://arxiv.org/pdf/2603.13777](https://arxiv.org/pdf/2603.13777)**

> **作者:** Shidong He; Haoyu Wang; Wenjie Luo
>
> **备注:** 4 figures, 3 tables
>
> **摘要:** Aspect-based sentiment analysis (ABSA) extracts aspect-level sentiment signals from user-generated text, supports product analytics, experience monitoring, and public-opinion tracking, and is central to fine-grained opinion mining. A key challenge in ABSA is aspect sentiment quad prediction (ASQP), which requires identifying four elements: the aspect term, the aspect category, the opinion term, and the sentiment polarity. However, existing studies usually linearize the unordered quad set into a fixed-order template and decode it left-to-right. With teacher forcing training, the resulting training-inference mismatch (exposure bias) lets early prefix errors propagate to later elements. The linearization order determines which elements appear earlier in the prefix, so this propagation becomes order-sensitive and is hard to repair in a single pass. To address this, we propose a method, Generate-then-Correct (G2C): a generator drafts quads and a corrector performs a single-shot, sequence-level global correction trained on LLM-synthesized drafts with common error patterns. On the Rest15 and Rest16 datasets, G2C outperforms strong baseline models.
>
---
#### [replaced 033] UtilityMax Prompting: A Formal Framework for Multi-Objective Large Language Model Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出UtilityMax Prompting框架，用于多目标大语言模型优化。通过形式化数学语言定义任务，明确优化目标，提升模型输出的准确性与一致性。**

- **链接: [https://arxiv.org/pdf/2603.11583](https://arxiv.org/pdf/2603.11583)**

> **作者:** Ofir Marom
>
> **摘要:** The success of a Large Language Model (LLM) task depends heavily on its prompt. Most use-cases specify prompts using natural language, which is inherently ambiguous when multiple objectives must be simultaneously satisfied. In this paper we introduce UtilityMax Prompting, a framework that specifies tasks using formal mathematical language. We reconstruct the task as an influence diagram in which the LLM's answer is the sole decision variable. A utility function is defined over the conditional probability distributions within the diagram, and the LLM is instructed to find the answer that maximises expected utility. This constrains the LLM to reason explicitly about each component of the objective, directing its output toward a precise optimization target rather than a subjective natural language interpretation. We validate our approach on the MovieLens 1M dataset across three frontier models (Claude Sonnet 4.6, GPT-5.4, and Gemini 2.5 Pro), demonstrating consistent improvements in precision and Normalized Discounted Cumulative Gain (NDCG) over natural language baselines in a multi-objective movie recommendation task.
>
---
#### [replaced 034] SPRIG: Improving Large Language Model Performance by System Prompt Optimization
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于自然语言处理领域，旨在提升大语言模型性能。针对系统提示优化问题，提出SPRIG算法，通过迭代优化提升模型在多种任务中的表现。**

- **链接: [https://arxiv.org/pdf/2410.14826](https://arxiv.org/pdf/2410.14826)**

> **作者:** Lechen Zhang; Tolga Ergen; Lajanugen Logeswaran; Moontae Lee; David Jurgens
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Large Language Models (LLMs) have shown impressive capabilities in many scenarios, but their performance depends, in part, on the choice of prompt. Past research has focused on optimizing prompts specific to a task. However, much less attention has been given to optimizing the general instructions included in a prompt, known as a system prompt. To address this gap, we propose SPRIG, an edit-based genetic algorithm that iteratively constructs prompts from prespecified components to maximize the model's performance in general scenarios. We evaluate the performance of system prompts on a collection of 47 different types of tasks to ensure generalizability. Our study finds that a single optimized system prompt performs on par with task prompts optimized for each individual task. Moreover, combining system and task-level optimizations leads to further improvement, which showcases their complementary nature. Experiments also reveal that the optimized system prompts generalize effectively across model families, parameter sizes, and languages. This study provides insights into the role of system-level instructions in maximizing LLM potential.
>
---
#### [replaced 035] Demystifying When Pruning Works via Representation Hierarchies
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究语言模型剪枝问题，分析其在生成与非生成任务中的效果差异。通过表示层次视角，揭示剪枝对不同空间的影响，为剪枝应用提供指导。**

- **链接: [https://arxiv.org/pdf/2603.24652](https://arxiv.org/pdf/2603.24652)**

> **作者:** Shwai He; Guoheng Sun; Haichao Zhang; Yun Fu; Ang Li
>
> **备注:** 27 pages, 21 figures, and 3 tables. Includes appendix with supplementary experiments and derivations
>
> **摘要:** Network pruning, which removes less important parameters or architectures, is often expected to improve efficiency while preserving performance. However, this expectation does not consistently hold across language tasks: pruned models can perform well on non-generative tasks but frequently fail in generative settings. To understand this discrepancy, we analyze network pruning from a representation-hierarchy perspective, decomposing the internal computation of language models into three sequential spaces: embedding (hidden representations), logit (pre-softmax outputs), and probability (post-softmax distributions). We find that representations in the embedding and logit spaces are largely robust to pruning-induced perturbations. However, the nonlinear transformation from logits to probabilities amplifies these deviations, which accumulate across time steps and lead to substantial degradation during generation. In contrast, the stability of the categorical-token probability subspace, together with the robustness of the embedding space, supports the effectiveness of pruning for non-generative tasks such as retrieval and multiple-choice selection. Our analysis disentangles the effects of pruning across tasks and provides practical guidance for its application. Code is available at this https URL
>
---
#### [replaced 036] PDF Retrieval Augmented Question Answering
- **分类: cs.CL**

- **简介: 该论文属于问答任务，旨在解决PDF中多模态信息提取问题。通过改进RAG框架，整合非文本元素，提升问答系统准确性。**

- **链接: [https://arxiv.org/pdf/2506.18027](https://arxiv.org/pdf/2506.18027)**

> **作者:** Thi Thu Uyen Hoang; Viet Anh Nguyen
>
> **摘要:** This paper presents an advancement in Question-Answering (QA) systems using a Retrieval Augmented Generation (RAG) framework to enhance information extraction from PDF files. Recognizing the richness and diversity of data within PDFs--including text, images, vector diagrams, graphs, and tables--poses unique challenges for existing QA systems primarily designed for textual content. We seek to develop a comprehensive RAG-based QA system that will effectively address complex multimodal questions, where several data types are combined in the query. This is mainly achieved by refining approaches to processing and integrating non-textual elements in PDFs into the RAG framework to derive precise and relevant answers, as well as fine-tuning large language models to better adapt to our system. We provide an in-depth experimental evaluation of our solution, demonstrating its capability to extract accurate information that can be applied to different types of content across PDFs. This work not only pushes the boundaries of retrieval-augmented QA systems but also lays a foundation for further research in multimodal data integration and processing.
>
---
#### [replaced 037] A Simple Method to Enhance Pre-trained Language Models with Speech Tokens for Classification
- **分类: cs.CL; cs.MM**

- **简介: 该论文属于多模态分类任务，旨在解决将语音信息有效融入预训练语言模型的问题。通过音频特征选择和自监督学习，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2512.07571](https://arxiv.org/pdf/2512.07571)**

> **作者:** Nicolas Calbucura; Jose Guillen; Valentin Barriere
>
> **摘要:** This paper presents a simple method that allows to easily enhance textual pre-trained large language models with speech information, when fine-tuned for a specific classification task. A classical issue with the fusion of many embeddings from audio with text is the large length of the audio sequence compared to the text one. Our method benefits from an existing speech tokenizer trained for Audio Speech Recognition that output long sequences of tokens from a large vocabulary, making it difficult to integrate it at low cost in a large language model. By applying a simple lasso-based feature selection on multimodal Bag-of-Words representation, we retain only the most important audio tokens for the task, and adapt the language model to them with a self-supervised language modeling objective, before fine-tuning it on the downstream task. We show this helps to improve the performances compared to an unimodal model, to a bigger SpeechLM or to integrating audio via a learned representation. We demonstrate its effectiveness on Argumentative Fallacy Detection and Classification tasks where audio was previously believed counterproductive, and affective computing tasks on a widely-used dataset. We also provide an in-depth analysis of the method, showing that even a random audio token selection helps enhancing the unimodal model. Our code is available [online](this https URL).
>
---
#### [replaced 038] Common TF-IDF variants arise as key components in the test statistic of a penalized likelihood-ratio test for word burstiness
- **分类: cs.CL; cs.IR; math.ST**

- **简介: 该论文属于信息检索领域，旨在解释TF-IDF的统计基础。通过构建惩罚似然比检验框架，揭示TF-IDF类似得分源于捕捉词频突变的统计模型，为术语加权提供理论支持。**

- **链接: [https://arxiv.org/pdf/2604.00672](https://arxiv.org/pdf/2604.00672)**

> **作者:** Zeyad Ahmed; Paul Sheridan; Michael McIsaac; Aitazaz A. Farooque
>
> **备注:** 27 pages, 3 tables, 7 figures, accepted in Discover Computing 2026
>
> **摘要:** TF-IDF is a classical formula that is widely used for identifying important terms within documents. We show that TF-IDF-like scores arise naturally from the test statistic of a penalized likelihood-ratio test setup capturing word burstiness (also known as word over-dispersion). In our framework, the alternative hypothesis captures word burstiness by modeling a collection of documents according to a family of beta-binomial distributions with a gamma penalty term on the precision parameter. In contrast, the null hypothesis assumes that words are binomially distributed in collection documents, a modeling approach that fails to account for word burstiness. We find that a term-weighting scheme given rise to by this test statistic performs comparably to TF-IDF on document classification tasks. This paper provides insights into TF-IDF from a statistical perspective and underscores the potential of hypothesis testing frameworks for advancing term-weighting scheme development.
>
---
#### [replaced 039] Co-Designing Quantum Codes with Transversal Diagonal Gates via Multi-Agent Systems
- **分类: quant-ph; cs.AI; cs.CL; math-ph**

- **简介: 该论文属于量子纠错码设计任务，解决如何构造具有特定对角横跨门的非加性量子码。通过多智能体系统与形式化验证结合，实现了精确搜索与证明。**

- **链接: [https://arxiv.org/pdf/2510.20728](https://arxiv.org/pdf/2510.20728)**

> **作者:** Xi He; Sirui Lu; Bei Zeng
>
> **备注:** 33 pages, 4 figures
>
> **摘要:** Exact scientific discovery requires more than heuristic search: candidate constructions must be turned into exact objects and checked independently. We address this gap by extending TeXRA with an independent Lean 4 verification layer, turning it into a human-guided multi-agent platform for exact scientific discovery. The platform couples symbolic synthesis, combinatorial and linear-programming search, exact reconstruction of numerical candidates, and formal verification in Lean. We apply this platform to nonadditive quantum error-correcting codes with prescribed transversal diagonal gates within the subset-sum linear-programming (SSLP) framework. In the distance-2 regime where logical states occupy distinct residue classes, the platform yields a Lean-certified catalogue of 14,116 codes for $K\in\{2,3,4\}$ and up to six physical qubits, realizing cyclic logical orders 2 through 18, from which we extract closed-form infinite families. We also construct a residue-degenerate $((6,4,2))$ code implementing the logical controlled-phase gate $\mathrm{diag}(1,1,1,i)$. At distance 3, we resolve the transversal-$T$ problem for $((7,2,3))$ codes within the complementary binary-dihedral $\mathrm{BD}_{16}$ setting: among the 12 candidates surviving the SSLP filters, 10 admit exact realizations and 2 are excluded by no-go proofs. All accepted constructions, families, and no-go results are formalized and checked in Lean, illustrating how AI-assisted workflows can bridge search, exact reconstruction, and formal proof in the physical sciences.
>
---
#### [replaced 040] KLong: Training LLM Agent for Extremely Long-horizon Tasks
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出KLong，解决长时任务的LLM代理训练问题。通过轨迹分割微调和渐进强化学习，提升模型处理长期任务的能力。**

- **链接: [https://arxiv.org/pdf/2602.17547](https://arxiv.org/pdf/2602.17547)**

> **作者:** Yue Liu; Yingwei Ma; Yibo Miao; Yanhao Li; Yuchong Xie; Xinlong Yang; Zhiyuan Hu; Flood Sung; Jiaheng Zhang; Bryan Hooi
>
> **摘要:** This paper introduces KLong, an open-source LLM agent trained to solve extremely long-horizon tasks. The principle is to first cold-start the model via trajectory-splitting SFT, then scale it via progressive RL training. Specifically, we first activate basic agentic abilities of a base model with a comprehensive SFT recipe. Then, we introduce Research-Factory, an automated pipeline that generates high-quality training data by collecting research papers and constructing evaluation rubrics. Using this pipeline, we build thousands of long-horizon trajectories distilled from Claude 4.5 Sonnet (Thinking). To train with these extremely long trajectories, we propose a new trajectory-splitting SFT, which preserves early context, progressively truncates later context, and maintains overlap between sub-trajectories. In addition, to further improve long-horizon task-solving capability, we propose a novel progressive RL, which schedules training into multiple stages with progressively extended timeouts. Experiments demonstrate the superiority and generalization of KLong, as shown in Figure 1. Notably, our proposed KLong (106B) surpasses Kimi K2 Thinking (1T) by 11.28% on PaperBench, and the performance improvement generalizes to other coding benchmarks like SWE-bench Verified and MLE-bench.
>
---
#### [replaced 041] Bridging the Semantic Gap for Categorical Data Clustering via Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于数据聚类任务，解决类别数据相似性度量问题。通过引入大语言模型增强语义表示，提升聚类效果。**

- **链接: [https://arxiv.org/pdf/2601.01162](https://arxiv.org/pdf/2601.01162)**

> **作者:** Zihua Yang; Xin Liao; Yiqun Zhang; Yiu-ming Cheung
>
> **摘要:** Categorical data are prevalent in domains such as healthcare, marketing, and bioinformatics, where clustering serves as a fundamental tool for pattern discovery. A core challenge in categorical data clustering lies in measuring similarity among attribute values that lack inherent ordering or distance. Without appropriate similarity measures, values are often treated as equidistant, creating a semantic gap that obscures latent structures and degrades clustering quality. Although existing methods infer value relationships from within-dataset co-occurrence patterns, such inference becomes unreliable when samples are limited, leaving the semantic context of the data underexplored. To bridge this gap, we present ARISE (Attention-weighted Representation with Integrated Semantic Embeddings), which draws on external semantic knowledge from Large Language Models (LLMs) to construct semantic-aware representations that complement the metric space of categorical data for accurate clustering. That is, LLM is adopted to describe attribute values for representation enhancement, and the LLM-enhanced embeddings are combined with the original data to explore semantically prominent clusters. Experiments on eight benchmark datasets demonstrate consistent improvements over seven representative counterparts, with gains of 19-27%. Code is available at this https URL
>
---
#### [replaced 042] MOOSE-Star: Unlocking Tractable Training for Scientific Discovery by Breaking the Complexity Barrier
- **分类: cs.LG; cs.CE; cs.CL**

- **简介: 该论文属于科学发现任务，解决直接建模生成推理过程的数学不可行问题。提出MOOSE-Star框架，通过分解任务、分层搜索和有限组合，降低计算复杂度，提升训练效率。**

- **链接: [https://arxiv.org/pdf/2603.03756](https://arxiv.org/pdf/2603.03756)**

> **作者:** Zonglin Yang; Lidong Bing
>
> **摘要:** While large language models (LLMs) show promise in scientific discovery, existing research focuses on inference or feedback-driven training, leaving the direct modeling of the generative reasoning process, $P(\text{hypothesis}|\text{background})$ ($P(h|b)$), unexplored. We demonstrate that directly training $P(h|b)$ is mathematically intractable due to the combinatorial complexity ($O(N^k)$) inherent in retrieving and composing inspirations from a vast knowledge base. To break this barrier, we introduce MOOSE-Star, a unified framework enabling tractable training and scalable inference. In the best case, MOOSE-Star reduces complexity from exponential to logarithmic ($O(\log N)$) by (1) training on decomposed subtasks derived from the probabilistic equation of discovery, (2) employing motivation-guided hierarchical search to enable logarithmic retrieval and prune irrelevant subspaces, and (3) utilizing bounded composition for robustness against retrieval noise. To facilitate this, we release TOMATO-Star, a dataset of 108,717 decomposed papers (38,400 GPU hours) for training. Furthermore, we show that while brute-force sampling hits a ''complexity wall,'' MOOSE-Star exhibits continuous test-time scaling.
>
---
#### [replaced 043] On the Role of Reasoning Patterns in the Generalization Discrepancy of Long Chain-of-Thought Supervised Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文研究监督微调中长思维链轨迹对模型泛化能力的影响，旨在解决不同来源轨迹导致的性能差异问题。通过分析两种模型生成的轨迹，发现训练损失与泛化能力不一致，并提出过滤分支轨迹的改进方法。**

- **链接: [https://arxiv.org/pdf/2604.01702](https://arxiv.org/pdf/2604.01702)**

> **作者:** Zhaoyi Li; Xiangyu Xi; Zhengyu Chen; Wei Wang; Gangwei Jiang; Ranran Shen; Linqi Song; Ying Wei; Defu Lian
>
> **备注:** Under Review. version2: correct typos in Table 4 and add an ablation study (Table 5)
>
> **摘要:** Supervised Fine-Tuning (SFT) on long Chain-of-Thought (CoT) trajectories has become a pivotal phase in building large reasoning models. However, how CoT trajectories from different sources influence the generalization performance of models remains an open question. In this paper, we conduct a comparative study using two sources of verified CoT trajectories generated by two competing models, \texttt{DeepSeek-R1-0528} and \texttt{gpt-oss-120b}, with their problem sets controlled to be identical. Despite their comparable performance, we uncover a striking paradox: lower training loss does not translate to better generalization. SFT on \texttt{DeepSeek-R1-0528} data achieves remarkably lower training loss, yet exhibits significantly worse generalization performance on reasoning benchmarks compared to those trained on \texttt{gpt-oss-120b}. To understand this paradox, we perform a multi-faceted analysis probing token-level SFT loss and step-level reasoning behaviors. Our analysis reveals a difference in reasoning patterns. \texttt{gpt-oss-120b} exhibits highly convergent and deductive trajectories, whereas \texttt{DeepSeek-R1-0528} favors a divergent and branch-heavy exploration pattern. Consequently, models trained with \texttt{DeepSeek-R1} data inherit inefficient exploration behaviors, often getting trapped in redundant exploratory branches that hinder them from reaching correct solutions. Building upon this insight, we propose a simple yet effective remedy of filtering out frequently branching trajectories to improve the generalization of SFT. Experiments show that training on selected \texttt{DeepSeek-R1-0528} subsets surprisingly improves reasoning performance by up to 5.1% on AIME25, 5.5% on BeyondAIME, and on average 3.6% on five benchmarks.
>
---
#### [replaced 044] Scaling the Scaling Logic: Agentic Meta-Synthesis of Logic Reasoning
- **分类: cs.AI; cs.CL; cs.LG; cs.LO**

- **简介: 该论文提出SSLogic框架，解决RLVR数据生成瓶颈问题，通过LLM代理迭代生成验证对，提升逻辑推理任务性能。**

- **链接: [https://arxiv.org/pdf/2602.13218](https://arxiv.org/pdf/2602.13218)**

> **作者:** Bowen Liu; Zhi Wu; Runquan Xie; Zhanhui Kang; Jia Li
>
> **备注:** 41 pages, 8 figures, 5 tables in the main body. Project page: this https URL, typos corrected, claims cleared
>
> **摘要:** Reinforcement Learning from Verifiable Rewards (RLVR) is bottlenecked by data: existing synthesis pipelines rely on expert-written code or fixed templates, confining growth to instance-level perturbations. We shift the evolvable unit from problem instances to task-family specifications. SSLogic is an agentic meta-synthesis framework in which LLM agents iteratively author and refine executable Generator-Validator pairs inside a closed Generate-Validate-Refine loop, producing families with new rules and difficulty gradients rather than parameter variations of old ones. A Multi-Gate Validation Protocol -- multi-strategy consensus plus Adversarial Blind Review, where independent agents solve each instance by writing and executing code -- filters ill-posed tasks before they enter training. Starting from 400 seed families, two evolution rounds yield 953 families and 21,389 verifiable instances. Three converging comparisons (step-matched, token-matched, and size-controlled on external Enigmata data) consistently show higher training utility of evolved data, with gains of SynLogic +5.2, AIME25 +3.0, and BBH +5.5 on Enigmata. Fine-grained KORBench evaluation reveals selective improvements in logic (+13.2%) and operation (+9.6%), linking structural evolution to downstream gains. Code: this https URL
>
---
#### [replaced 045] OptiMer: Optimal Distribution Vector Merging Is Better than Data Mixing for Continual Pre-Training
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于持续预训练任务，解决数据混合比例优化问题。提出OptiMer方法，通过分布向量后优化获得最优权重，提升模型性能并降低调参成本。**

- **链接: [https://arxiv.org/pdf/2603.28858](https://arxiv.org/pdf/2603.28858)**

> **作者:** Haiyue Song; Masao Utiyama
>
> **备注:** Preprint, 20 pages, 10 tables, 12 figures
>
> **摘要:** Continual pre-training is widely used to adapt LLMs to target languages and domains, yet the mixture ratio of training data remains a sensitive hyperparameter that is expensive to tune: they must be fixed before training begins, and a suboptimal choice can waste weeks of compute. In this work, we propose OptiMer, which decouples ratio selection from training: we train one CPT model per dataset, extract each model's distribution vector, which represents the parameter shift induced by that dataset, and search for optimal composition weights post-hoc via Bayesian optimization. Experiments on Gemma 3 27B across languages (Japanese, Chinese) and domains (Math, Code) show that OptiMer consistently outperforms data mixture and model averaging baselines with 15-35 times lower search cost. Key findings reveal that 1) the optimized weights can be interpreted as data mixture ratios, and retraining with these ratios improves data mixture CPT, and 2) the same vector pool can be re-optimized for a given objective without any retraining, producing target-tailored models on demand. Our work establishes that data mixture ratio selection, traditionally a pre-training decision, can be reformulated as a post-hoc optimization over distribution vectors, offering a more flexible paradigm for continual pre-training.
>
---
#### [replaced 046] LinguDistill: Recovering Linguistic Ability in Vision- Language Models via Selective Cross-Modal Distillation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态学习任务，旨在解决视觉-语言模型在适配过程中损失语言能力的问题。通过无适配器的蒸馏方法，恢复语言能力，同时保持视觉性能。**

- **链接: [https://arxiv.org/pdf/2604.00829](https://arxiv.org/pdf/2604.00829)**

> **作者:** Patrick Amadeus Irawan; Erland Hilman Fuadi; Shanu Kumar; Alham Fikri Aji; Yova Kementchedjhieva
>
> **摘要:** Adapting pretrained language models (LMs) into vision-language models (VLMs) can degrade their native linguistic capability due to representation shift and cross-modal interference introduced during multimodal adaptation. Such loss is difficult to recover, even with targeted task-specific fine-tuning using standard objectives. Prior recovery approaches typically introduce additional modules that act as intermediate alignment layers to maintain or isolate modality-specific subspaces, which increases architectural complexity, adds parameters at inference time, and limits flexibility across models and settings. We propose LinguDistill, an adapter-free distillation method that restores linguistic capability by utilizing the original frozen LM as a teacher. We overcome the key challenge of enabling vision-conditioned teacher supervision by introducing layer-wise KV-cache sharing, which exposes the teacher to the student's multimodal representations without modifying the architecture of either model. We then selectively distill the teacher's strong linguistic signal on language-intensive data to recover language capability, while preserving the student's visual grounding on multimodal tasks. As a result, LinguDistill recovers $\sim$10% of the performance lost on language and knowledge benchmarks, while maintaining comparable performance on vision-heavy tasks. Our findings demonstrate that linguistic capability can be recovered without additional modules, providing an efficient and practical solution to modality-specific degradation in multimodal models.
>
---
#### [replaced 047] PRISM: Prompt-Refined In-Context System Modelling for Financial Retrieval
- **分类: cs.AI; cs.CE; cs.CL; cs.IR**

- **简介: 该论文提出PRISM框架，用于金融信息检索任务，解决从长文档中提取关键信息的问题。通过提示优化、上下文学习和轻量级多智能体协作提升排序效果。**

- **链接: [https://arxiv.org/pdf/2511.14130](https://arxiv.org/pdf/2511.14130)**

> **作者:** Chun Chet Ng; Jia Yu Lim; Wei Zeng Low
>
> **备注:** 3rd-place solution for the ACM ICAIF 2025 Agentic Retrieval Grand Challenge. Accepted for poster presentation at ICLR 2026 (Advances in Financial AI Workshop)
>
> **摘要:** With the rapid progress of large language models (LLMs), financial information retrieval has become a critical industrial application. Extracting task-relevant information from lengthy financial filings is essential for both operational and analytical decision-making. We present PRISM, a training-free framework that integrates refined system prompting, in-context learning (ICL), and lightweight multi-agent coordination for document and chunk ranking tasks. Our primary contribution is a systematic empirical study of when each component provides value: prompt engineering delivers consistent performance with minimal overhead, ICL enhances reasoning for complex queries when applied selectively, and multi-agent systems show potential primarily with larger models and careful architectural design. Extensive ablation studies across FinAgentBench, FiQA-2018, and FinanceBench reveal that simpler configurations often outperform complex multi-agent pipelines, providing practical guidance for practitioners. Our best configuration achieves an NDCG@5 of 0.71818 on FinAgentBench, ranking third while being the only training-free approach in the top three. We provide comprehensive feasibility analyses covering latency, token usage, and cost trade-offs to support deployment decisions. The source code is released at this https URL.
>
---
#### [replaced 048] Multilingual KokoroChat: A Multi-LLM Ensemble Translation Method for Creating a Multilingual Counseling Dialogue Dataset
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决高质量多语言心理咨询对话数据集稀缺的问题。通过多大语言模型集成方法，提升翻译质量，构建多语言对话数据集。**

- **链接: [https://arxiv.org/pdf/2603.22913](https://arxiv.org/pdf/2603.22913)**

> **作者:** Ryoma Suzuki; Zhiyang Qi; Michimasa Inaba
>
> **备注:** 12 pages, 8 figures, Accepted to LREC 2026
>
> **摘要:** To address the critical scarcity of high-quality, publicly available counseling dialogue datasets, we created Multilingual KokoroChat by translating KokoroChat, a large-scale manually authored Japanese counseling corpus, into both English and Chinese. A key challenge in this process is that the optimal model for translation varies by input, making it impossible for any single model to consistently guarantee the highest quality. In a sensitive domain like counseling, where the highest possible translation fidelity is essential, relying on a single LLM is therefore insufficient. To overcome this challenge, we developed and employed a novel multi-LLM ensemble method. Our approach first generates diverse hypotheses from multiple distinct LLMs. A single LLM then produces a high-quality translation based on an analysis of the respective strengths and weaknesses of all presented hypotheses. The quality of ``Multilingual KokoroChat'' was rigorously validated through human preference studies. These evaluations confirmed that the translations produced by our ensemble method were preferred from any individual state-of-the-art LLM. This strong preference confirms the superior quality of our method's outputs. The Multilingual KokoroChat is available at this https URL.
>
---
#### [replaced 049] Autorubric: Unifying Rubric-based LLM Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大模型评估任务，旨在统一和优化基于评分量表的LLM评估方法。提出Autorubric框架，解决评估方法分散、术语不一致的问题，整合多种评估技术并提升评估效果。**

- **链接: [https://arxiv.org/pdf/2603.00077](https://arxiv.org/pdf/2603.00077)**

> **作者:** Delip Rao; Chris Callison-Burch
>
> **备注:** 52 pages
>
> **摘要:** Techniques for reliable rubric-based LLM evaluation -- ensemble judging, bias mitigation, few-shot calibration -- are scattered across papers with inconsistent terminology and partial implementations. We introduce Autorubric, an open-source framework that unifies these rubric-based LLM evaluation lessons with opinionated defaults: analytic rubrics with binary, ordinal, and nominal criteria; single-judge and ensemble evaluation; few-shot calibration; bias mitigations; and psychometric reliability metrics. We validate on three benchmarks: RiceChem (college chemistry grading, 80\% accuracy with 5-shot calibration), ResearcherBench (deep research evaluation, 931 criteria, cross-judge agreement analysis), and CHARM-100, a new chatbot evaluation dataset combining all three criterion types with ground truth labels (87\% binary accuracy, moderate-to-substantial $\kappa$). Beyond measurement, per-criterion scores and explanations serve as optimization signals. We demonstrate how Autorubric's rubric-evaluation explanations raise a peer review agent's score from 0.47 to 0.85 (above the 0.82 expert-curated baseline), and its scores serve as RL rewards to produce statistically significant improvement on AdvancedIF (+0.039, Wilcoxon $p = 0.032$) with positive transfer to IFEval. In all of these cases, Autorubric enabled us to rapidly operationalize various rubric design choices and best practices with minimal effort.
>
---
#### [replaced 050] WhisperRT -- Turning Whisper into a Causal Streaming Model
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决传统模型不适用于实时流式 transcription 的问题。通过改造 Whisper 模型，使其具备低延迟的流式处理能力。**

- **链接: [https://arxiv.org/pdf/2508.12301](https://arxiv.org/pdf/2508.12301)**

> **作者:** Tomer Krichli; Bhiksha Raj; Joseph Keshet
>
> **备注:** 14 pages, 7 Figures, This work has been submitted to the IEEE for possible publication
>
> **摘要:** Automatic Speech Recognition (ASR) has seen remarkable progress, with models like OpenAI Whisper and NVIDIA Canary achieving state-of-the-art (SOTA) performance in offline transcription. However, these models are not designed for streaming (online or real-time) transcription, due to limitations in their architecture and training methodology. We propose a method to turn the transformer encoder-decoder model into a low-latency streaming model. The encoder is made causal to process audio incrementally, while the decoder conditions on partial encoder states to generate tokens aligned with the available temporal context. This requires explicit synchronization between encoded input frames and token emissions. Since tokens are produced only after sufficient acoustic evidence is observed, an inherent latency arises, necessitating fine-tuning of the encoder-decoder alignment mechanism. We propose an updated inference mechanism that utilizes the fine-tuned causal encoder and decoder to yield greedy and beam-search decoding, and is shown to be locally optimal. Experiments on low-latency chunk sizes (less than 300 msec) show that our fine-tuned model outperforms existing non-fine-tuned streaming approaches in most cases, while using a lower complexity. We release our training and inference code, along with the fine-tuned models, to support further research and development in streaming ASR.
>
---
#### [replaced 051] FURINA: A Fully Customizable Role-Playing Benchmark via Scalable Multi-Agent Collaboration Pipeline
- **分类: cs.CL; cs.AI; cs.HC; cs.MA**

- **简介: 该论文属于角色扮演任务，解决现有基准过时、适应性差的问题。提出FURINA-Builder多智能体协作管道，构建可定制的RP基准，评估不同角色表现。**

- **链接: [https://arxiv.org/pdf/2510.06800](https://arxiv.org/pdf/2510.06800)**

> **作者:** Haotian Wu; Shufan Jiang; Chios Chen; Yiyang Feng; Hehai Lin; Heqing Zou; Yao Shu; Chengwei Qin
>
> **摘要:** As large language models (LLMs) advance in role-playing (RP) tasks, existing benchmarks quickly become obsolete due to their narrow scope, outdated interaction paradigms, and limited adaptability across diverse application scenarios. To address this gap, we introduce FURINA-Builder, a novel multi-agent collaboration pipeline that automatically constructs fully customizable RP benchmarks at any scale. It enables evaluation of arbitrary characters across diverse scenarios and prompt formats, as the first benchmark builder in RP area for adaptable assessment. FURINA-Builder simulates dialogues between a test character and other characters drawn from a well-constructed character-scene pool, while an LLM judge selects fine-grained evaluation dimensions and adjusts the test character's responses into final test utterances. Using this pipeline, we build FURINA-Bench, a new comprehensive role-playing benchmark featuring both established and synthesized test characters, each assessed with dimension-specific evaluation criteria. Human evaluation and preliminary separability analysis justify our pipeline and benchmark design. We conduct extensive evaluations of cutting-edge LLMs and find that o3 and DeepSeek-R1 achieve the best performance on English and Chinese RP tasks, respectively. Across all models, established characters consistently outperform synthesized ones, with reasoning capabilities further amplifying this disparity. Interestingly, we observe that model scale does not monotonically reduce hallucinations. More critically, for reasoning LLMs, we uncover a novel trade-off: reasoning improves RP performance but simultaneously increases RP hallucinations. This trade-off extends to a broader Pareto frontier between RP performance and reliability for all LLMs. These findings demonstrate the effectiveness of FURINA-Builder and the challenge posed by FURINA-Bench.
>
---
#### [replaced 052] ProMediate: A Socio-cognitive framework for evaluating proactive agents in multi-party negotiation
- **分类: cs.CL**

- **简介: 该论文提出ProMediate框架，用于评估多主体协商中的主动AI代理。解决多主体协作中缺乏系统评估方法的问题，通过模拟测试和新指标衡量代理的社认知能力。**

- **链接: [https://arxiv.org/pdf/2510.25224](https://arxiv.org/pdf/2510.25224)**

> **作者:** Ziyi Liu; Bahar Sarrafzadeh; Pei Zhou; Longqi Yang; Jieyu Zhao; Ashish Sharma
>
> **摘要:** While Large Language Models (LLMs) are increasingly used in agentic frameworks to assist individual users, there is a growing need for agents that can proactively manage complex, multi-party collaboration. Systematic evaluation methods for such proactive agents remain scarce, limiting progress in developing AI that can effectively support multiple people together. Negotiation offers a demanding testbed for this challenge, requiring socio-cognitive intelligence to navigate conflicting interests between multiple participants and multiple topics and build consensus. Here, we present ProMediate, the first framework for evaluating proactive AI mediator agents in complex, multi-topic, multi-party negotiations. ProMediate consists of two core components: (i) a simulation testbed based on realistic negotiation cases and theory-driven difficulty levels (ProMediate-Easy, ProMediate-Medium, and ProMediate-Hard), with a plug-and-play proactive AI mediator grounded in socio-cognitive mediation theories, capable of flexibly deciding when and how to intervene; and (ii) a socio-cognitive evaluation framework with a new suite of metrics to measure consensus changes, intervention latency, mediator effectiveness, and intelligence. Together, these components establish a systematic framework for assessing the socio-cognitive intelligence of proactive AI agents in multi-party settings. Our results show that a socially intelligent mediator agent outperforms a generic baseline, via faster, better-targeted interventions. In the ProMediate-Hard setting, our social mediator increases consensus change by 3.6 percentage points compared to the generic baseline (10.65\% vs 7.01\%) while being 77\% faster in response (15.98s vs. 3.71s). In conclusion, ProMediate provides a rigorous, theory-grounded testbed to advance the development of proactive, socially intelligent agents.
>
---
#### [replaced 053] Not All Tokens Matter: Towards Efficient LLM Reasoning via Token Significance in Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的推理效率。针对模型生成冗长回答的问题，提出基于token重要性的强化学习方法，减少冗余并保持准确性。**

- **链接: [https://arxiv.org/pdf/2506.08125](https://arxiv.org/pdf/2506.08125)**

> **作者:** Hanbing Liu; Lang Cao; Yuanyi Ren; Mengyu Zhou; Haoyu Dong; Xiaojun Ma; Shi Han; Dongmei Zhang
>
> **摘要:** Large language models (LLMs) show strong reasoning abilities but often produce unnecessarily long explanations that reduce efficiency. Although reinforcement learning (RL) has been used to improve reasoning, most methods focus on accuracy and rely on uniform length-based rewards that overlook the differing contributions of individual tokens, often harming correctness. We revisit length optimization in RL through the perspective of token significance. Observing that many chain-of-thought (CoT) tokens contribute little to the final answer, we introduce a significance-aware length reward that selectively penalizes insignificance tokens, reducing redundancy while preserving essential reasoning. We also propose a dynamic length reward that encourages more detailed reasoning early in training and gradually shifts toward conciseness as learning progresses. Integrating these components into standard policy optimization yields a framework that improves both reasoning efficiency and accuracy. Experiments across multiple benchmarks demonstrate substantial reductions in response length while preserving or improving correctness, highlighting the importance of modeling token significance for efficient LLM reasoning.
>
---
#### [replaced 054] GraphWalker: Agentic Knowledge Graph Question Answering via Synthetic Trajectory Curriculum
- **分类: cs.CL**

- **简介: 该论文属于知识图谱问答任务，解决代理在知识图谱中自主导航与推理泛化的问题。提出GraphWalker框架，通过合成轨迹和分阶段微调提升性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.28533](https://arxiv.org/pdf/2603.28533)**

> **作者:** Shuwen Xu; Yao Xu; Jiaxiang Liu; Chenhao Yuan; Wenshuo Peng; Jun Zhao; Kang Liu
>
> **摘要:** Agentic knowledge graph question answering (KGQA) requires an agent to iteratively interact with knowledge graphs (KGs), posing challenges in both training data scarcity and reasoning generalization. Specifically, existing approaches often restrict agent exploration: prompting-based methods lack autonomous navigation training, while current training pipelines usually confine reasoning to predefined trajectories. To this end, this paper proposes \textit{GraphWalker}, a novel agentic KGQA framework that addresses these challenges through \textit{Automated Trajectory Synthesis} and \textit{Stage-wise Fine-tuning}. GraphWalker adopts a two-stage SFT training paradigm: First, the agent is trained on structurally diverse trajectories synthesized from constrained random-walk paths, establishing a broad exploration prior over the KG; Second, the agent is further fine-tuned on a small set of expert trajectories to develop reflection and error recovery capabilities. Extensive experiments demonstrate that our stage-wise SFT paradigm unlocks a higher performance ceiling for a lightweight reinforcement learning (RL) stage, enabling GraphWalker to achieve state-of-the-art performance on CWQ and WebQSP. Additional results on GrailQA and our constructed GraphWalkerBench confirm that GraphWalker enhances generalization to out-of-distribution reasoning paths. The code is publicly available at this https URL
>
---
#### [replaced 055] Making Prompts First-Class Citizens for Adaptive LLM Pipelines
- **分类: cs.DB; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM提示管理的局限性。通过SPEAR框架，将提示作为一等公民，实现结构化管理、动态优化和策略控制，提升系统适应性和效率。**

- **链接: [https://arxiv.org/pdf/2508.05012](https://arxiv.org/pdf/2508.05012)**

> **作者:** Ugur Cetintemel; Shu Chen; Alexander W. Lee; Deepti Raghavan; Duo Lu; Andrew Crotty
>
> **备注:** 6 pages, 2 figures, appears in CIDR'26
>
> **摘要:** Modern LLM pipelines increasingly resemble complex data-centric applications: they retrieve data, correct errors, call external tools, and coordinate interactions between agents. Yet, the central element controlling this entire process -- the prompt -- remains a brittle, opaque string that is entirely disconnected from the surrounding program logic. This disconnect fundamentally limits opportunities for reuse, optimization, and runtime adaptivity. In this paper, we describe our vision and an initial design of SPEAR (Structured Prompt Execution and Adaptive Refinement), a new approach to prompt management that treats prompts as first-class citizens in the execution model. Specifically, SPEAR enables: (1) structured prompt management, with prompts organized into versioned views to support introspection and reasoning about provenance; (2) adaptive prompt refinement, whereby prompts can evolve dynamically during execution based on runtime feedback; and (3) policy-driven control, a mechanism for the specification of automatic prompt refinement logic as when-then rules. By tackling the problem of runtime prompt refinement, SPEAR plays a complementary role in the vast ecosystem of existing prompt optimization frameworks and semantic query processing engines. We describe a number of related optimization opportunities unlocked by the SPEAR model, and our preliminary results demonstrate the strong potential of this approach.
>
---
#### [replaced 056] LLMs Judge Themselves: A Game-Theoretic Framework for Human-Aligned Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型评估任务，旨在解决传统评估方法不足的问题。通过游戏理论框架，让LLMs互相评价，并与人类判断对比，验证其一致性。**

- **链接: [https://arxiv.org/pdf/2510.15746](https://arxiv.org/pdf/2510.15746)**

> **作者:** Gao Yang; Yuhang Liu; Siyu Miao; Xinyue Liang; Zhengyang Liu; Heyan Huang
>
> **摘要:** Ideal or real - that is the this http URL this work, we explore whether principles from game theory can be effectively applied to the evaluation of large language models (LLMs). This inquiry is motivated by the growing inadequacy of conventional evaluation practices, which often rely on fixed-format tasks with reference answers and struggle to capture the nuanced, subjective, and open-ended nature of modern LLM behavior. To address these challenges, we propose a novel alternative: automatic mutual evaluation, where LLMs assess each other's output through self-play and peer review. These peer assessments are then systematically compared with human voting behavior to evaluate their alignment with human judgment. Our framework incorporates game-theoretic voting algorithms to aggregate peer reviews, enabling a principled investigation into whether model-generated rankings reflect human preferences. Empirical results reveal both convergences and divergences between theoretical predictions and human evaluations, offering valuable insights into the promises and limitations of mutual evaluation. To the best of our knowledge, this is the first work to jointly integrate mutual evaluation, game-theoretic aggregation, and human-grounded validation for evaluating the capabilities of LLMs.
>
---
#### [replaced 057] XiYan-SQL: A Novel Multi-Generator Framework For Text-to-SQL
- **分类: cs.CL**

- **简介: 该论文提出XiYan-SQL框架，解决Text-to-SQL任务中的生成与选择问题，通过多生成器和优化策略提升SQL生成效果。**

- **链接: [https://arxiv.org/pdf/2507.04701](https://arxiv.org/pdf/2507.04701)**

> **作者:** Yifu Liu; Yin Zhu; Yingqi Gao; Zhiling Luo; Xiaoxia Li; Xiaorong Shi; Yuntao Hong; Jinyang Gao; Yu Li; Bolin Ding; Jingren Zhou
>
> **备注:** Published in IEEE TKDE
>
> **摘要:** To leverage the advantages of LLM in addressing challenges in the Text-to-SQL task, we present XiYan-SQL, an innovative framework effectively generating and utilizing multiple SQL candidates. It consists of three components: 1) a Schema Filter module filtering and obtaining multiple relevant schemas; 2) a multi-generator ensemble approach generating multiple highquality and diverse SQL queries; 3) a selection model with a candidate reorganization strategy implemented to obtain the optimal SQL query. Specifically, for the multi-generator ensemble, we employ a multi-task fine-tuning strategy to enhance the capabilities of SQL generation models for the intrinsic alignment between SQL and text, and construct multiple generation models with distinct generation styles by fine-tuning across different SQL formats. The experimental results and comprehensive analysis demonstrate the effectiveness and robustness of our framework. Overall, XiYan-SQL achieves a new SOTA performance of 75.63% on the notable BIRD benchmark, surpassing all previous methods. It also attains SOTA performance on the Spider test set with an accuracy of 89.65%.
>
---
#### [replaced 058] StoryScope: Investigating idiosyncrasies in AI fiction
- **分类: cs.CL**

- **简介: 该论文属于AI生成文本检测任务，旨在区分AI与人类创作的虚构故事。研究提出StoryScope，通过分析叙事特征实现高效检测与作者识别。**

- **链接: [https://arxiv.org/pdf/2604.03136](https://arxiv.org/pdf/2604.03136)**

> **作者:** Jenna Russell; Rishanth Rajendhran; Mohit Iyyer; John Wieting
>
> **摘要:** As AI-generated fiction becomes increasingly prevalent, questions of authorship and originality are becoming central to how written work is evaluated. While most existing work in this space focuses on identifying surface-level signatures of AI writing, we ask instead whether AI-generated stories can be distinguished from human ones without relying on stylistic signals, focusing on discourse-level narrative choices such as character agency and chronological discontinuity. We propose StoryScope, a pipeline that automatically induces a fine-grained, interpretable feature space of discourse-level narrative features across 10 dimensions. We apply StoryScope to a parallel corpus of 10,272 writing prompts, each written by a human author and five LLMs, yielding 61,608 stories, each ~5,000 words, and 304 extracted features per story. Narrative features alone achieve 93.2% macro-F1 for human vs. AI detection and 68.4% macro-F1 for six-way authorship attribution, retaining over 97% of the performance of models that include stylistic cues. A compact set of 30 core narrative features captures much of this signal: AI stories over-explain themes and favor tidy, single-track plots while human stories frame protagonist' choices as more morally ambiguous and have increased temporal complexity. Per-model fingerprint features enable six-way attribution: for example, Claude produces notably flat event escalation, GPT over-indexes on dream sequences, and Gemini defaults to external character description. We find that AI-generated stories cluster in a shared region of narrative space, while human-authored stories exhibit greater diversity. More broadly, these results suggest that differences in underlying narrative construction, not just writing style, can be used to separate human-written original works from AI-generated fiction.
>
---
#### [replaced 059] Sandpiper: Orchestrated AI-Annotation for Educational Discourse at Scale
- **分类: cs.HC; cs.CL**

- **简介: 该论文提出Sandpiper系统，解决教育对话中大规模AI标注的效率与准确性问题。通过混合智能架构，实现高效、合规的质性分析。**

- **链接: [https://arxiv.org/pdf/2603.08406](https://arxiv.org/pdf/2603.08406)**

> **作者:** Daryl Hedley; Doug Pietrzak; Jorge Dias; Ian Burden; Bakhtawar Ahtisham; Zhuqian Zhou; Kirk Vanacore; Josh Marland; Rachel Slama; Justin Reich; Kenneth Koedinger; René Kizilcec
>
> **摘要:** Digital educational environments are expanding toward complex AI and human discourse, providing researchers with an abundance of data that offers deep insights into learning and instructional processes. However, traditional qualitative analysis remains a labor-intensive bottleneck, severely limiting the scale at which this research can be conducted. We present Sandpiper, a mixed-initiative system designed to serve as a bridge between high-volume conversational data and human qualitative expertise. By tightly coupling interactive researcher dashboards with agentic Large Language Model (LLM) engines, the platform enables scalable analysis without sacrificing methodological rigor. Sandpiper addresses critical barriers to AI adoption in education by implementing context-aware, automated de-identification workflows supported by secure, university-housed infrastructure to ensure data privacy. Furthermore, the system employs schema-constrained orchestration to eliminate LLM hallucinations and enforces strict adherence to qualitative codebooks. An integrated evaluations engine allows for the continuous benchmarking of AI performance against human labels, fostering an iterative approach to model refinement and validation. We propose a user study to evaluate the system's efficacy in improving research efficiency, inter-rater reliability, and researcher trust in AI-assisted qualitative workflows.
>
---
#### [replaced 060] Geometric Organization of Cognitive States in Transformer Embedding Spaces
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究Transformer模型中句子嵌入的几何结构是否与认知属性相关。任务是验证嵌入空间是否存在可解释的几何组织。通过构建标注数据集并使用探针模型进行分析，证明了嵌入空间确实具有统计显著的结构。**

- **链接: [https://arxiv.org/pdf/2512.22227](https://arxiv.org/pdf/2512.22227)**

> **作者:** Sophie Zhao
>
> **摘要:** Recent work has shown that transformer-based language models learn rich geometric structure in their embedding spaces. In this work, we investigate whether sentence embeddings exhibit structured geometric organization aligned with human-interpretable cognitive or psychological attributes. We construct a dataset of 480 natural-language sentences annotated with both continuous energy scores (ranging from -5 to +5) and discrete tier labels spanning seven ordered cognitive annotation tiers, intended to capture a graded progression from highly constricted or reactive expressions toward more coherent and integrative cognitive states. Using fixed sentence embeddings from multiple transformer models, we evaluate the recoverability of these annotations via linear and shallow nonlinear probes. Across models, both continuous energy scores and tier labels are reliably decodable, with linear probes already capturing substantial structure. To assess statistical significance, we conduct nonparametric permutation tests that randomize labels, showing that probe performance exceeds chance under both regression and classification null hypotheses. Qualitative analyses using UMAP visualizations and tier-level confusion matrices further reveal a coherent low-to-high gradient and predominantly local (adjacent-tier) confusions. Together, these results indicate that transformer embedding spaces exhibit statistically significant geometric organization aligned with the annotated cognitive structure.
>
---
#### [replaced 061] In-Context Watermarks for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于文本水印任务，旨在解决LLM生成内容的溯源问题。提出In-Context Watermarking方法，通过提示工程嵌入水印，无需访问解码过程。**

- **链接: [https://arxiv.org/pdf/2505.16934](https://arxiv.org/pdf/2505.16934)**

> **作者:** Yepeng Liu; Xuandong Zhao; Christopher Kruegel; Dawn Song; Yuheng Bu
>
> **备注:** ICLR2026
>
> **摘要:** The growing use of large language models (LLMs) for sensitive applications has highlighted the need for effective watermarking techniques to ensure the provenance and accountability of AI-generated text. However, most existing watermarking methods require access to the decoding process, limiting their applicability in real-world settings. One illustrative example is the use of LLMs by dishonest reviewers in the context of academic peer review, where conference organizers have no access to the model used but still need to detect AI-generated reviews. Motivated by this gap, we introduce In-Context Watermarking (ICW), which embeds watermarks into generated text solely through prompt engineering, leveraging LLMs' in-context learning and instruction-following abilities. We investigate four ICW strategies at different levels of granularity, each paired with a tailored detection method. We further examine the Indirect Prompt Injection (IPI) setting as a specific case study, in which watermarking is covertly triggered by modifying input documents such as academic manuscripts. Our experiments validate the feasibility of ICW as a model-agnostic, practical watermarking approach. Moreover, our findings suggest that as LLMs become more capable, ICW offers a promising direction for scalable and accessible content attribution. Our code is available at this https URL.
>
---
#### [replaced 062] Optical Context Compression Is Just (Bad) Autoencoding
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于文本压缩任务，探讨光学上下文压缩的有效性。研究对比了视觉编码与直接方法，发现视觉压缩效果不佳，不优于简单直接方法。**

- **链接: [https://arxiv.org/pdf/2512.03643](https://arxiv.org/pdf/2512.03643)**

> **作者:** Ivan Yee Lee; Cheng Yang; Taylor Berg-Kirkpatrick
>
> **摘要:** DeepSeek-OCR shows that rendered text can be reconstructed from a small number of vision tokens, sparking excitement about using vision as a compression medium for long textual contexts. But this pipeline requires rendering token embeddings to pixels and compressing from there -- discarding learned representations in favor of an image the vision encoder must then recover from. We ask whether this detour helps. Comparing DeepSeek-OCR's vision encoder against near-zero-parameter mean pooling and a learned hierarchical encoder, we find it does not. For reconstruction, simple direct methods match or surpass vision at every compression ratio. For language modeling, vision performs comparably to truncation -- a baseline that simply discards context -- and loses to the hierarchical encoder at every compression ratio. As expected, all compression methods outperform truncation for factual recall, but vision never surpasses the best direct baseline. The excitement around optical context compression outpaces the evidence. Code and checkpoints are available at this https URL.
>
---
#### [replaced 063] A Linguistics-Aware LLM Watermarking via Syntactic Predictability
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI水印任务，旨在解决文本质量与检测鲁棒性之间的平衡问题。通过结合语言结构特性，提出STELA框架，实现无需模型日志的可公开验证水印。**

- **链接: [https://arxiv.org/pdf/2510.13829](https://arxiv.org/pdf/2510.13829)**

> **作者:** Shinwoo Park; Hyejin Park; Hyeseon Ahn; Yo-Sub Han
>
> **备注:** ACL 2026
>
> **摘要:** As large language models (LLMs) continue to advance rapidly, reliable governance tools have become critical. Publicly verifiable watermarking is particularly essential for fostering a trustworthy AI ecosystem. A central challenge persists: balancing text quality against detection robustness. Recent studies have sought to navigate this trade-off by leveraging signals from model output distributions (e.g., token-level entropy); however, their reliance on these model-specific signals presents a significant barrier to public verification, as the detection process requires access to the logits of the underlying model. We introduce STELA, a novel framework that aligns watermark strength with the linguistic degrees of freedom inherent in language. STELA dynamically modulates the signal using part-of-speech (POS) n-gram-modeled linguistic indeterminacy, weakening it in grammatically constrained contexts to preserve quality and strengthen it in contexts with greater linguistic flexibility to enhance detectability. Our detector operates without access to any model logits, thus facilitating publicly verifiable detection. Through extensive experiments on typologically diverse languages-analytic English, isolating Chinese, and agglutinative Korean-we show that STELA surpasses prior methods in detection robustness. Our code is available at this https URL.
>
---
#### [replaced 064] Gaussian mixture models as a proxy for interacting language models
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文属于模型代理任务，旨在用GMM替代LLM以降低计算成本。通过构建交互式GMM系统，模拟LLM的互动行为，并分析其极化现象。**

- **链接: [https://arxiv.org/pdf/2506.00077](https://arxiv.org/pdf/2506.00077)**

> **作者:** Edward L. Wang; Mohammad Sharifi Kiasari; Tianyu Wang; Hayden Helm; Avanti Athreya; Carey Priebe; Vince Lyzinski
>
> **摘要:** Large language models (LLMs) are powerful tools that, in a number of settings, overlap with the results of human pattern recognition and reasoning. Retrieval-augmented generation (RAG) further allows LLMs to produce tailored output depending on the contents of their RAG databases. However, LLMs depend on complex, computationally expensive algorithms. In this paper, we introduce interacting Gaussian mixture models (GMMs) as a proxy for interacting LLMs. We construct a model of interacting GMMs, complete with an analogue to RAG updating, under which GMMs can generate, exchange, and update data and parameters. We show that this interacting system of Gaussian mixture models, which can be implemented at minimal computational cost, mimics certain aspects of experimental simulations of interacting LLMs whose iterative responses depend on feedback from other LLMs. We build a Markov chain from this system of interacting GMMs; formalize and interpret the notion of polarization for such a chain; and prove lower bounds on the probability of polarization. This provides theoretical insight into the use of interacting Gaussian mixture models as a computationally efficient proxy for interacting large language models.
>
---
#### [replaced 065] HatePrototypes: Interpretable and Transferable Representations for Implicit and Explicit Hate Speech Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于仇恨言论检测任务，旨在解决隐性与显性仇恨内容识别问题。通过构建可解释的原型表示，实现跨任务和跨基准的迁移学习，提升检测效率与效果。**

- **链接: [https://arxiv.org/pdf/2511.06391](https://arxiv.org/pdf/2511.06391)**

> **作者:** Irina Proskurina; Marc-Antoine Carpentier; Julien Velcin
>
> **摘要:** Optimization of offensive content moderation models for different types of hateful messages is typically achieved through continued pre-training or fine-tuning on new hate speech benchmarks. However, existing benchmarks mainly address explicit hate toward protected groups and often overlook implicit or indirect hate, such as demeaning comparisons, calls for exclusion or violence, and subtle discriminatory language that still causes harm. While explicit hate can often be captured through surface features, implicit hate requires deeper, full-model semantic processing. In this work, we question the need for repeated fine-tuning and analyze the role of HatePrototypes, class-level vector representations derived from language models optimized for hate speech detection and safety moderation. We find that these prototypes, built from as few as 50 examples per class, enable cross-task transfer between explicit and implicit hate, with interchangeable prototypes across benchmarks. Moreover, we show that parameter-free early exiting with prototypes is effective for both hate types. We release the code, prototype resources, and evaluation scripts to support future research on efficient and transferable hate speech detection.
>
---
#### [replaced 066] BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding
- **分类: cs.CL**

- **简介: 该论文提出BLASST，解决LLM长上下文推理中的计算和内存瓶颈问题。通过动态稀疏注意力机制加速推理，无需训练和预计算，提升效率。**

- **链接: [https://arxiv.org/pdf/2512.12087](https://arxiv.org/pdf/2512.12087)**

> **作者:** Jiayi Yuan; Cameron Shinn; Kai Xu; Jingze Cui; George Klimiashvili; Guangxuan Xiao; Perkz Zheng; Bo Li; Yuxin Zhou; Zhouhai Ye; Weijie You; Tian Zheng; Dominic Brown; Pengbo Wang; Markus Hoehnerbach; Richard Cai; Julien Demouth; John D. Owens; Xia Hu; Song Han; Timmy Liu; Huizi Mao
>
> **摘要:** The growing demand for long-context inference capabilities in Large Language Models (LLMs) has intensified the computational and memory bottlenecks inherent to the self-attention mechanism. To address this challenge, we introduce BLASST, a drop-in, dynamic sparse attention mechanism that accelerates inference by using only a fixed scalar threshold to skip attention blocks. Our method targets practical inference deployment by removing the barriers to adoption present in existing works. As such, BLASST eliminates training requirements, avoids expensive pre-computation passes, accelerates both prefill and decode across all major attention variants (MHA, GQA, MQA, and MLA), provides optimized support for modern hardware, and easily integrates into existing frameworks. This is achieved by reusing online softmax statistics to identify negligible attention scores, skipping softmax, value block loads, and the subsequent matrix multiplication. We demonstrate the BLASST algorithm by delivering optimized kernels with negligible latency overhead. Our automated threshold calibration procedure reveals a simple inverse relationship between optimal threshold and context length, meaning we require only a single threshold each for prefill and decode per model. Preserving benchmark accuracy, we demonstrate a 1.52x speedup for prefill at 71.9% sparsity and a 1.48x speedup for decode at 73.2% sparsity on modern GPUs.
>
---
#### [replaced 067] ZINA: Multimodal Fine-grained Hallucination Detection and Editing
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出ZINA，用于多模态大语言模型的细粒度幻觉检测与编辑任务，解决输出与视觉内容不符的问题。**

- **链接: [https://arxiv.org/pdf/2506.13130](https://arxiv.org/pdf/2506.13130)**

> **作者:** Yuiga Wada; Kazuki Matsuda; Komei Sugiura; Graham Neubig
>
> **备注:** CVPR 2026 Main Conference
>
> **摘要:** Multimodal Large Language Models (MLLMs) often generate hallucinations, where the output deviates from the visual content. Given that these hallucinations can take diverse forms, detecting hallucinations at a fine-grained level is essential for comprehensive evaluation and analysis. To this end, we propose a novel task of multimodal fine-grained hallucination detection and editing for MLLMs. Moreover, we propose ZINA, a novel method that identifies hallucinated spans at a fine-grained level, classifies their error types into six categories, and suggests appropriate refinements. To train and evaluate models for this task, we construct VisionHall, a dataset comprising 6.9k outputs from twelve MLLMs manually annotated by 211 annotators, and 20k synthetic samples generated using a graph-based method that captures dependencies among error types. We demonstrated that ZINA outperformed existing methods, including GPT-4o and Llama-3.2, in both detection and editing tasks.
>
---
#### [replaced 068] From Chains to DAGs: Probing the Graph Structure of Reasoning in LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究大模型内部推理结构是否为图而非链。通过构建探针分析隐藏状态，验证推理是否以DAG形式编码，发现中间层具有最佳可恢复性。**

- **链接: [https://arxiv.org/pdf/2601.17593](https://arxiv.org/pdf/2601.17593)**

> **作者:** Tianjun Zhong; Linyang He; Nima Mesgarani
>
> **摘要:** Recent progress in large language models has renewed interest in how multi-step reasoning is represented internally. While prior work often treats reasoning as a linear chain, many reasoning problems are more naturally modeled as directed acyclic graphs (DAGs), where intermediate conclusions branch, merge, and are reused. Whether such graph structure is reflected in model internals remains unclear. We introduce Reasoning DAG Probing, a framework for testing whether LLM hidden states linearly encode properties of an underlying reasoning DAG and where this structure emerges across layers. We associate each reasoning node with a textual realization and train lightweight probes to predict node depth, pairwise distance, and adjacency from hidden states. Using these probes, we analyze the emergence of DAG structure across layers, reconstruct approximate reasoning graphs, and evaluate controls that disrupt reasoning-relevant structure while preserving surface text. Across reasoning benchmarks, we find that DAG structure is meaningfully encoded in LLM representations, with recoverability peaking in intermediate layers, varying systematically by node depth, edge span, and model scale, and enabling nontrivial recovery of dependency graphs. These findings suggest that LLM reasoning is not purely sequential, but exhibits measurable internal graph structure.
>
---
#### [replaced 069] S0 Tuning: Zero-Overhead Adaptation of Hybrid Recurrent-Attention Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出S0调优方法，用于优化混合循环-注意力模型，在零推理开销下提升模型性能，解决小样本监督下的参数高效微调问题。**

- **链接: [https://arxiv.org/pdf/2604.01168](https://arxiv.org/pdf/2604.01168)**

> **作者:** Jack Young
>
> **备注:** 15 pages (10 main + 5 appendix), 3 figures, code at this https URL
>
> **摘要:** Using roughly 48 execution-verified HumanEval training solutions, tuning a single initial state matrix per recurrent layer, with zero inference overhead, outperforms LoRA by +10.8 pp (p < 0.001) on HumanEval. The method, which we call S0 tuning, optimizes one state matrix per recurrent layer while freezing all model weights. On Qwen3.5-4B (GatedDeltaNet hybrid), S0 tuning improves greedy pass@1 by +23.6 +/- 1.7 pp (10 seeds). On FalconH1-7B (Mamba-2 hybrid), S0 reaches 71.8% +/- 1.3 and LoRA reaches 71.4% +/- 2.4 (3 seeds), statistically indistinguishable at this sample size while requiring no weight merging. Cross-domain transfer is significant on MATH-500 (+4.8 pp, p = 0.00002, 8 seeds) and GSM8K (+2.8 pp, p = 0.0003, 10 seeds); a text-to-SQL benchmark (Spider) shows no transfer, consistent with the trajectory-steering mechanism. A prefix-tuning control on a pure Transformer (Qwen2.5-3B) degrades performance by -13.9 pp under all nine configurations tested. On Qwen3.5, a per-step state-offset variant reaches +27.1 pp, above both S0 and LoRA but with per-step inference cost. Taken together, the results show that recurrent state initialization is a strong zero-inference-overhead PEFT surface for hybrid language models when verified supervision is scarce. The tuned state is a ~48 MB file; task switching requires no weight merging or model reload. Code and library: this https URL.
>
---
#### [replaced 070] Beyond Linear Steering: Unified Multi-Attribute Control for Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型行为控制任务，解决多属性控制中的干扰问题。提出K-Steering方法，通过非线性分类器和梯度计算干预方向，实现动态行为组合。**

- **链接: [https://arxiv.org/pdf/2505.24535](https://arxiv.org/pdf/2505.24535)**

> **作者:** Narmeen Oozeer; Luke Marks; Shreyans Jain; Fazl Barez; Amirali Abdullah
>
> **备注:** Accepted to Findings of EMNLP, 2025
>
> **摘要:** Controlling multiple behavioral attributes in large language models (LLMs) at inference time is a challenging problem due to interference between attributes and the limitations of linear steering methods, which assume additive behavior in activation space and require per-attribute tuning. We introduce K-Steering, a unified and flexible approach that trains a single non-linear multi-label classifier on hidden activations and computes intervention directions via gradients at inference time. This avoids linearity assumptions, removes the need for storing and tuning separate attribute vectors, and allows dynamic composition of behaviors without retraining. To evaluate our method, we propose two new benchmarks, ToneBank and DebateMix, targeting compositional behavioral control. Empirical results across 3 model families, validated by both activation-based classifiers and LLM-based judges, demonstrate that K-Steering outperforms strong baselines in accurately steering multiple behaviors.
>
---
#### [replaced 071] Cite Pretrain: Retrieval-Free Knowledge Attribution for Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于知识溯源任务，旨在解决大语言模型生成不可靠引用的问题。通过持续预训练和指令调优，使模型在不依赖外部检索的情况下准确引用来源。**

- **链接: [https://arxiv.org/pdf/2506.17585](https://arxiv.org/pdf/2506.17585)**

> **作者:** Yukun Huang; Sanxing Chen; Jian Pei; Manzil Zaheer; Bhuwan Dhingra
>
> **摘要:** Trustworthy language models should provide both correct and verifiable answers. However, citations generated directly by standalone LLMs are often unreliable. As a result, current systems insert citations by querying an external retriever at inference time, introducing latency, infrastructure dependence, and vulnerability to retrieval noise. We explore whether LLMs can be made to reliably attribute to the documents seen during continual pretraining without test-time retrieval, by revising the training process. To study this, we construct CitePretrainBench, a benchmark that mixes real-world corpora (Wikipedia, Common Crawl, arXiv) with novel documents and probes both short-form (single-fact) and long-form (multi-fact) citation tasks. Our approach follows a two-stage process: (1) continual pretraining to index factual knowledge by binding it to persistent document identifiers; and (2) instruction tuning to elicit citation behavior. We introduce Active Indexing for the first stage, which creates generalizable, source-anchored bindings by augmenting training with synthetic data that (i) restate each fact in diverse, compositional forms and (ii) enforce bidirectional training (source-to-fact and fact-to-source). This equips the model to both generate content from a cited source and attribute its own answers, improving robustness to paraphrase and composition. Experiments with Qwen-2.5-7B&3B show that Active Indexing consistently outperforms a Passive Indexing baseline, which simply appends an identifier to each document, achieving citation precision gains of up to 30.2% across all tasks and models. Our ablation studies reveal that performance continues to improve as we scale the amount of augmented data, showing a clear upward trend even at 16x the original token count. Finally, we show that internal citations complement external ones by making the model more robust to retrieval noise.
>
---
#### [replaced 072] Talk to Right Specialists: Iterative Routing in Multi-agent Systems for Question Answering
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文属于多智能体问答任务，解决用户难以选择合适智能体及复杂问题需跨智能体协作的问题。提出RIRS框架，通过嵌入空间路由和迭代聚合实现高效精准问答。**

- **链接: [https://arxiv.org/pdf/2501.07813](https://arxiv.org/pdf/2501.07813)**

> **作者:** Feijie Wu; Zitao Li; Fei Wei; Yaliang Li; Bolin Ding; Jing Gao
>
> **备注:** Differences between v1 & v2: The algorithm name of the first version is RopMura, which decomposes a multi-hop query into several simple subqueries, and a question selector selects one of the subqueries to answer. In the second version, the name is updated to RIRS, which directly routes a query to the appropriate agents, regardless of whether the query is single-hop or multi-hop
>
> **摘要:** Retrieval-augmented generation (RAG) agents are increasingly deployed to answer questions over local knowledge bases that cannot be centralized due to knowledge-sovereignty constraints. This results in two recurring failures in production: users do not know which agent to consult, and complex questions require evidence distributed across multiple agents. To overcome these challenges, we propose RIRS, a training-free orchestration framework to enable a multi-agent system for question answering. In detail, RIRS summarizes each agent's local corpus in an embedding space, enabling a user-facing server to route queries only to the most relevant agents, reducing latency and avoiding noisy "broadcast-to-all" contexts. For complicated questions, the server can iteratively aggregate responses to derive intermediate results and refine the question to bridge the gap toward a comprehensive answer. Extensive experiments demonstrate the effectiveness of RIRS, including its ability to precisely select agents and provide accurate responses to single-hop queries, and its use of an iterative strategy to achieve accurate, multi-step resolutions for complex queries.
>
---
#### [replaced 073] Xpertbench: Expert Level Tasks with Rubrics-Based Evaluation
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出XpertBench，用于评估大语言模型在专业领域的表现。解决现有评估框架不足的问题，通过专家任务和评分标准进行高精度评测。**

- **链接: [https://arxiv.org/pdf/2604.02368](https://arxiv.org/pdf/2604.02368)**

> **作者:** Xue Liu; Xin Ma; Yuxin Ma; Yongchang Peng; Duo Wang; Zhoufutu Wen; Ge Zhang; Kaiyuan Zhang; Xinyu Chen; Tianci He; Jiani Hou; Liang Hu; Ziyun Huang; Yongzhe Hui; Jianpeng Jiao; Chennan Ju; Yingru Kong; Yiran Li; Mengyun Liu; Luyao Ma; Fei Ni; Yiqing Ni; Yueyan Qiu; Yanle Ren; Zilin Shi; Zaiyuan Wang; Wenjie Yue; Shiyu Zhang; Xinyi Zhang; Kaiwen Zhao; Zhenwei Zhu; Shanshan Wu; Qi Zhao; Wenhao Huang
>
> **摘要:** As Large Language Models (LLMs) exhibit plateauing performance on conventional benchmarks, a pivotal challenge persists: evaluating their proficiency in complex, open-ended tasks characterizing genuine expert-level cognition. Existing frameworks suffer from narrow domain coverage, reliance on generalist tasks, or self-evaluation biases. To bridge this gap, we present XpertBench, a high-fidelity benchmark engineered to assess LLMs across authentic professional domains. XpertBench consists of 1,346 meticulously curated tasks across 80 categories, spanning finance, healthcare, legal services, education, and dual-track research (STEM and Humanities). These tasks are derived from over 1,000 submissions by domain experts--including researchers from elite institutions and practitioners with extensive clinical or industrial experience--ensuring superior ecological validity. Each task uses detailed rubrics with mostly 15-40 weighted checkpoints to assess professional rigor. To facilitate scalable yet human-aligned assessment, we introduce ShotJudge, a novel evaluation paradigm that employs LLM judges calibrated with expert few-shot exemplars to mitigate self-rewarding biases. Our empirical evaluation of state-of-the-art LLMs reveals a pronounced performance ceiling: even leading models achieve a peak success rate of only ~66%, with a mean score around 55%. Models also exhibit domain-specific divergence, showing non-overlapping strengths in quantitative reasoning versus linguistic synthesis.. These findings underscore a significant "expert-gap" in current AI systems and establish XpertBench as a critical instrument for navigating the transition from general-purpose assistants to specialized professional collaborators.
>
---
#### [replaced 074] Measuring Competency, Not Performance: Item-Aware Evaluation Across Medical Benchmarks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医学领域大语言模型评估任务，旨在解决传统准确率评估无法反映真实能力的问题。提出MedIRT框架，结合项目反应理论，更准确衡量模型医学能力。**

- **链接: [https://arxiv.org/pdf/2509.24186](https://arxiv.org/pdf/2509.24186)**

> **作者:** Zhimeng Luo; Lixin Wu; Adam Frisch; Daqing He
>
> **摘要:** Accuracy-based evaluation of Large Language Models (LLMs) measures benchmark-specific performance rather than underlying medical competency: it treats all questions as equally informative, conflates model ability with item characteristics, and thereby produces rankings that vary with benchmark choice. To address this, we introduce MedIRT, a psychometric evaluation framework grounded in Item Response Theory (IRT) that (1) jointly models latent competency and item-level difficulty and discrimination, and (2) includes benchmark integrity validation to ensure items within each topic measure a single, coherent underlying ability. We prospectively evaluate 71 diverse LLMs on a USMLE-aligned benchmark across 11 medical topics. As internal validation, MedIRT correctly predicts held-out LLM responses on unseen questions with 83.3% accuracy. As external validation, IRT-based rankings outperform accuracy-based rankings across 6 independent external medical benchmarks -- including expert preferences, holistic clinical tasks, safety judgments, and open-ended queries -- achieving 4 wins, 0 losses, and 18% lower variance. As a substantive finding, topic-level competency profiles expose striking domain-specific heterogeneity that aggregate accuracy masks. As a diagnostic tool, difficulty-tier analysis reveals two distinct response profiles (difficulty-sensitive responding and difficulty-insensitive responding) that require fundamentally different interventions. These results establish item-aware psychometric evaluation as a more valid and stable foundation for assessing LLMs in medicine, with potential implications for any high-stakes domain where benchmark integrity can be validated, and items vary meaningfully in difficulty and discrimination.
>
---
#### [replaced 075] The Generalization Ridge: Information Flow in Natural Language Generation
- **分类: cs.CL**

- **简介: 该论文研究Transformer模型在自然语言生成中的信息流动，旨在揭示中间层如何促进泛化。通过提出InfoRidge框架，分析预测信息的变化趋势，发现中间层具有最佳泛化能力。**

- **链接: [https://arxiv.org/pdf/2507.05387](https://arxiv.org/pdf/2507.05387)**

> **作者:** Ruidi Chang; Chunyuan Deng; Hanjie Chen
>
> **摘要:** Transformer-based language models have achieved state-of-the-art performance in natural language generation (NLG), yet their internal mechanisms for synthesizing task-relevant information remain insufficiently understood. While prior studies suggest that intermediate layers often yield more generalizable representations than final layers, how this generalization ability emerges and propagates across layers during training remains unclear. We propose InfoRidge, an information-theoretic framework, to characterize how predictive information-the mutual information between hidden representations and target outputs-varies across depth during training. Our experiments across various models and datasets reveal a consistent non-monotonic trend: predictive information peaks in intermediate layers-forming a generalization ridge-before declining in final layers, reflecting a transition between generalization and memorization. To further investigate this phenomenon, we conduct a set of complementary analyses that leverage residual scaling and attention pattern to characterize layer-wise functional specialization. We further validate our findings with multiple-token generation experiments, verifying that the observed ridge phenomenon persists across decoding steps. Together, these findings offer new insights into the internal mechanisms of transformers and underscore the critical role of intermediate layers in supporting generalization.
>
---
#### [replaced 076] When AI Agents Collude Online: Financial Fraud Risks by Collaborative LLM Agents on Social Platforms
- **分类: cs.MA; cs.AI; cs.CL; cs.SI**

- **简介: 该论文属于安全风险分析任务，研究多智能体系统中的金融欺诈风险。工作包括构建基准测试平台，分析欺诈因素，并提出缓解策略。**

- **链接: [https://arxiv.org/pdf/2511.06448](https://arxiv.org/pdf/2511.06448)**

> **作者:** Qibing Ren; Zhijie Zheng; Jiaxuan Guo; Junchi Yan; Lizhuang Ma; Jing Shao
>
> **备注:** ICLR 2026, Code is available at this https URL
>
> **摘要:** In this work, we study the risks of collective financial fraud in large-scale multi-agent systems powered by large language model (LLM) agents. We investigate whether agents can collaborate in fraudulent behaviors, how such collaboration amplifies risks, and what factors influence fraud success. To support this research, we present MultiAgentFraudBench, a large-scale benchmark for simulating financial fraud scenarios based on realistic online interactions. The benchmark covers 28 typical online fraud scenarios, spanning the full fraud lifecycle across both public and private domains. We further analyze key factors affecting fraud success, including interaction depth, activity level, and fine-grained collaboration failure modes. Finally, we propose a series of mitigation strategies, including adding content-level warnings to fraudulent posts and dialogues, using LLMs as monitors to block potentially malicious agents, and fostering group resilience through information sharing at the societal level. Notably, we observe that malicious agents can adapt to environmental interventions. Our findings highlight the real-world risks of multi-agent financial fraud and suggest practical measures for mitigating them. Code is available at this https URL.
>
---
#### [replaced 077] Truth as a Compression Artifact in Language Model Training
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型为何偏好正确答案，通过实验发现其源于错误的可压缩性而非真理本身。任务为理解语言模型的真相偏差机制，工作包括设计矛盾数据集并分析模型表现。**

- **链接: [https://arxiv.org/pdf/2603.11749](https://arxiv.org/pdf/2603.11749)**

> **作者:** Konstantin Krestnikov
>
> **备注:** v3: Added Qwen3 architecture check (0.6B), ~1B experiment on FineWeb-Edu, generative eval at all scales, matched-random ablation. Formal MDL predictions. Softened claims, fixed bibliography, added NeurIPS checklist. 210+ models (was 160+)
>
> **摘要:** Why do language models trained on contradictory data prefer correct answers? In controlled experiments with small transformers (3.5M--86M parameters), we show that this preference tracks the compressibility structure of errors rather than truth per se. We train GPT-2 style models on corpora where each mathematical problem appears with both correct and incorrect solutions -- a denoising design that directly models conflicting information about the same fact. When errors are random, models extract the correct signal with accuracy scaling from 65% to 85% with model size. When errors follow a coherent alternative rule system, accuracy drops to chance (~45--51%): the model cannot distinguish the false system from truth. A multi-rule experiment reveals a sharp crossover: a single coherent alternative rule eliminates truth bias entirely, but adding a second competing rule restores most of it (47%->78%), with continued growth through N=10 (88%). The same pattern reproduces on real Wikipedia text (71% vs 46%). We propose the Compression--Consistency Principle as an explanatory hypothesis: in these settings, gradient descent favors the most compressible answer cluster, not truth per se. Truth bias emerges only when falsehood is structurally incoherent. Whether this principle extends to large-scale pretraining remains an open question.
>
---
#### [replaced 078] MegaFake: A Theory-Driven Dataset of Fake News Generated by Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于虚假新闻检测任务，旨在解决LLM生成虚假新闻的问题。提出LLM-Fake理论，构建MegaFake数据集，推动虚假新闻检测方法的发展。**

- **链接: [https://arxiv.org/pdf/2408.11871](https://arxiv.org/pdf/2408.11871)**

> **作者:** Lionel Z. Wang; Ka Chung Ng; Yiming Ma; Wenqi Fan
>
> **备注:** Decision Support Systems
>
> **摘要:** Fake news significantly influences decision-making processes by misleading individuals, organizations, and even governments. Large language models (LLMs), as part of generative AI, can amplify this problem by generating highly convincing fake news at scale, posing a significant threat to online information integrity. Therefore, understanding the motivations and mechanisms behind fake news generated by LLMs is crucial for effective detection and governance. In this study, we develop the LLM-Fake Theory, a theoretical framework that integrates various social psychology theories to explain machine-generated deception. Guided by this framework, we design an innovative prompt engineering pipeline that automates fake news generation using LLMs, eliminating manual annotation needs. Utilizing this pipeline, we create a theoretically informed \underline{M}achin\underline{e}-\underline{g}ener\underline{a}ted \underline{Fake} news dataset, MegaFake, derived from FakeNewsNet. Through extensive experiments with MegaFake, we advance both theoretical understanding of human-machine deception mechanisms and practical approaches to fake news detection in the LLM era.
>
---
#### [replaced 079] SciGA: A Comprehensive Dataset for Designing Graphical Abstracts in Academic Papers
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文提出SciGA-145k数据集，用于支持图形摘要的设计与推荐，解决GA设计困难和缺乏有效工具的问题。任务包括同篇和跨篇推荐，引入CAR评估指标。**

- **链接: [https://arxiv.org/pdf/2507.02212](https://arxiv.org/pdf/2507.02212)**

> **作者:** Takuro Kawada; Shunsuke Kitada; Sota Nemoto; Hitoshi Iyatomi
>
> **备注:** 28 pages, 21 figures, 9 tables. Accepted to CVPR Findings 2026. Project page: this https URL
>
> **摘要:** Graphical Abstracts (GAs) play a crucial role in visually conveying the key findings of scientific papers. Although recent research increasingly incorporates visual materials such as Figure 1 as de facto GAs, their potential to enhance scientific communication remains largely unexplored. Designing effective GAs requires advanced visualization skills, hindering their widespread adoption. To tackle these challenges, we introduce SciGA-145k, a large-scale dataset comprising approximately 145,000 scientific papers and 1.14 million figures, specifically designed to support GA selection and recommendation, and to facilitate research in automated GA generation. As a preliminary step toward GA design support, we define two tasks: 1) Intra-GA Recommendation, identifying figures within a given paper well-suited as GAs, and 2) Inter-GA Recommendation, retrieving GAs from other papers to inspire new GA designs. Furthermore, we propose Confidence Adjusted top-1 ground truth Ratio (CAR), a novel recommendation metric for fine-grained analysis of model behavior. CAR addresses limitations of traditional rank-based metrics by considering that not only an explicitly labeled GA but also other in-paper figures may plausibly serve as GAs. Benchmark results demonstrate the viability of our tasks and the effectiveness of CAR. Collectively, these establish a foundation for advancing scientific communication within AI for Science.
>
---
#### [replaced 080] ModalImmune: Immunity Driven Unlearning via Self Destructive Training
- **分类: cs.LG; cs.CL; cs.MM**

- **简介: 该论文提出ModalImmune，解决多模态系统在输入通道丢失时的可靠性问题，通过训练增强模态免疫能力。**

- **链接: [https://arxiv.org/pdf/2602.16197](https://arxiv.org/pdf/2602.16197)**

> **作者:** Rong Fu; WeiZhi Tang; Ziming Wang; Jia Yee Tan; Zijian Zhang; Zhaolu Kang; Muge Qi; Shuning Zhang; Simon Fong
>
> **备注:** 24 pages, 8 figures
>
> **摘要:** Multimodal systems are vulnerable to partial or complete loss of input channels at deployment, which undermines reliability in real-world settings. This paper presents ModalImmune, a training framework that enforces modality immunity by intentionally and controllably collapsing selected modality information during training so the model learns joint representations that are robust to destructive modality influence. The framework combines a spectrum-adaptive collapse regularizer, an information-gain guided controller for targeted interventions, curvature-aware gradient masking to stabilize destructive updates, and a certified Neumann-truncated hyper-gradient procedure for automatic meta-parameter adaptation. Empirical evaluation on standard multimodal benchmarks demonstrates that ModalImmune improves resilience to modality removal and corruption while retaining convergence stability and reconstruction capacity.
>
---
#### [replaced 081] The Drill-Down and Fabricate Test (DDFT): A Protocol for Measuring Epistemic Robustness in Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出DDFT协议，用于评估语言模型的真理稳健性。解决现有评测无法衡量模型在压力下的可靠性问题。通过实验发现模型稳健性与参数量和架构无关，关键在于验证机制。**

- **链接: [https://arxiv.org/pdf/2512.23850](https://arxiv.org/pdf/2512.23850)**

> **作者:** Rahul Baxi
>
> **备注:** This version strengthens the theoretical and empirical grounding of the CI metric, including explicit analysis of structural dependencies and ranking stability under ablations (e.g., excluding Turn 4). Claims regarding scale and robustness are revised to avoid overgeneralization. The evaluation protocol, jury methodology, and limitations are expanded to clarify assumptions and boundary conditions
>
> **摘要:** Current language model evaluations measure what models know under ideal conditions but not how robustly they know it under realistic stress. Static benchmarks like MMLU and TruthfulQA cannot distinguish a model that lacks knowledge from one whose verification mechanisms collapse when information degrades or adversaries probe for weaknesses. We introduce the Drill-Down and Fabricate Test (DDFT), a protocol that measures epistemic robustness: a model's ability to maintain factual accuracy under progressive semantic compression and adversarial fabrication. We propose a two-system cognitive model comprising a Semantic System that generates fluent text and an Epistemic Verifier that validates factual accuracy. Our findings, based on evaluating 9 frontier models across 8 knowledge domains at 5 compression levels (1,800 turn-level evaluations), reveal that epistemic robustness is orthogonal to conventional design paradigms. Neither parameter count (r=0.083, p=0.832) nor architectural type (r=0.153, p=0.695) significantly predicts robustness, suggesting it emerges from training methodology and verification mechanisms distinct from current approaches. Error detection capability strongly predicts overall robustness (rho=-0.817, p=0.007), indicating this is the critical bottleneck. We find that flagship models exhibit brittleness despite their scale, while smaller models can achieve robust performance, challenging assumptions about the relationship between model size and reliability. The DDFT framework provides both theoretical foundation and practical tools for assessing epistemic robustness before deployment in critical applications.
>
---
