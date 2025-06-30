# 🩺 Towards Trustworthy Medical AI: A Survey on Retrieval Augmented Generation
Taxonomy, Applications, Challenges, and Future Directions

This is README.md file

# Table of Contents

- [🩺 RAG_Survey_for_Healthcare](#-rag_survey_for_healthcare)
  - [ℹ️ Background](#-background)
  - [📑 RAG’s Data Sources and Architecture](#-rag’s-data-sources-and-architecture)
    - [💾 Data Sources](#-data-sources)
    - [🧠 Naive RAG](#-naive-rag)
    - [🌟 Advanced RAG](#-advanced-rag)
    - [🛠️ Module RAG](#-module-rag)
    - [📊 Graph RAG](#-graph-rag)
    - [🤖 Agentic RAG](#-agentic-rag)
  - [🩺 Applications & Use Cases](#-applications--use-cases)
    - [🎓 Medical Training & Education](#-medical-training--education)
    - [🩺 Clinical Decision Support](#-clinical-decision-support)
    - [📖 Medical Literature Summarization](#-medical-literature-summarization)
    - [📊 Electronic Health Record Analysis](#-electronic-health-record-analysis)



## ℹ️ Background

<div align="center">
  <img src="/02_LLM_Models.png" alt="LLM_Models" width="60%"/>
</div>

- BioBERT: a pre-trained biomedical language representation model for biomedical text mining [[paper](https://arxiv.org/pdf/1901.08746)]
- SciBERT: A Pretrained Language Model for Scientific Text [[paper](https://arxiv.org/pdf/1903.10676)]
- Publicly Available Clinical BERT Embeddings [[paper](https://arxiv.org/pdf/1904.03323)]
- ClinicalBERT: Modeling Clinical Notes and Predicting Hospital Readmission [[paper](https://arxiv.org/pdf/1904.05342)]
- Pharmbert: a domain-specific bert model for drug labels [[paper](https://academic.oup.com/bib/article/24/4/bbad226/7197744)]
- BioMegatron: Larger Biomedical Domain Language Model [[paper](https://arxiv.org/pdf/2010.06060)]
- ClinicalT5: A Generative Language Model for Clinical Text [[paper](https://aclanthology.org/2022.findings-emnlp.398.pdf)]
- BioGPT: generative pre-trained transformer for biomedical text generation and mining [[paper](https://arxiv.org/pdf/2210.10341)]
- Toward expert-level medical question answering with large language models [[paper](https://arxiv.org/pdf/2305.09617)]
- MedAlpaca–an open-source collection of medical conversational AI models and training data [[paper](https://arxiv.org/pdf/2304.08247)]
- PMC-LLaMA: Towards Building Open-source Language Models for Medicine [[paper](https://arxiv.org/pdf/2304.14454)]
- LLaMA: Open and Efficient Foundation Language Models [[paper](https://arxiv.org/pdf/2302.13971)]
- Meditron-70b: Scaling medical pretraining for large language models [[paper](https://arxiv.org/pdf/2311.16079)]
- A large chest radiograph dataset with uncertainty labels and expert comparison [[paper](https://arxiv.org/pdf/1901.07031)]
- MedCLIP: Contrastive Learning from Unpaired Medical Images and Text [[paper](https://arxiv.org/pdf/2210.10163)]
- XrayGPT: Chest Radiographs Summarization using Medical Vision-Language Models [[paper](https://arxiv.org/pdf/2306.07971)]
- Highly accurate protein structure prediction with AlphaFold [[paper](https://www.nature.com/articles/s41586-021-03819-2)]
- DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome [[paper](https://academic.oup.com/bioinformatics/article/37/15/2112/6128680)]
- Multi-modal Self-supervised Pre-training for Regulatory Genome Across Cell Types [[paper](https://arxiv.org/pdf/2110.05231)]
- ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing [[paper](https://arxiv.org/pdf/2007.06225)]
- PharmaGPT: Domain-Specific Large Language Models for Bio-Pharmaceutical and Chemistry [[paper](https://arxiv.org/pdf/2406.18045)]
- “That’s so cute!”: The CARE Dataset for Affective Response Detection [[paper](https://aclanthology.org/2022.conll-1.5.pdf)]
- ScribeAgent: Towards Specialized Web Agents Using Production-Scale Workflow Data [[paper](https://arxiv.org/pdf/2411.15004)]
- Can GPT-3.5 Generate and Code Discharge Summaries? [[paper](https://arxiv.org/pdf/2401.13512)]
- Using ChatGPT-4 to Create Structured Medical Notes From Audio Recordings of Physician-Patient Encounters: Comparative Study [[paper](https://www.jmir.org/2024/1/e54419/PDF)]
- BERT fine-tuned CORD-19 NER dataset [[paper](https://ieee-dataport.org/documents/bert-fine-tuned-cord-19-ner-dataset)]
- LitGPT [[paper](https://github.com/Lightning-AI/litgpt)]
- PubMed GPT: a Domain-Specific Large Language Model for Biomedical Text [[paper](https://medium.com/@haoliu/pubmed-gpt-a-domain-specific-large-language-model-for-biomedical-text-b3bbd6fc32cb)]
- VisionGPT: Vision-Language Understanding Agent Using Generalized Multimodal Framework [[paper](https://arxiv.org/pdf/2403.09027)]
- Large Language Models Encode Clinical Knowledge [[paper](https://arxiv.org/pdf/2212.13138)]
- Survey of Hallucination in Natural Language Generation [[paper](https://arxiv.org/pdf/2202.03629)]
- Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer [[paper](https://arxiv.org/pdf/1910.10683)]
- Using AI-generated suggestions from ChatGPT to optimize clinical decision support [[paper](https://academic.oup.com/jamia/article/30/7/1237/7136722)]
- Retrieval-augmented generation for large language models: A survey [[paper](https://arxiv.org/pdf/2312.10997)]

## 📑 RAG’s Data Sources and Architecture

### 💾 Data Sources

<div align="center">
  <img src="/Datasets.png" alt="DataSources" width="60%"/>
</div>

- Mimic-extract: A data extraction, preprocessing, and representation pipeline for mimic-iii. [[paper](https://dl.acm.org/doi/pdf/10.1145/3368555.3384469)]
- Evaluating temporal relations in clinical text: 2012 i2b2 challenge. [[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3756273/pdf/amiajnl-2013-001628.pdf)]
- Evaluating shallow and deep learning strategies for the 2018 n2c2 shared task on clinical text classification. [[paper](https://watermark.silverchair.com/ocz149.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA2gwggNkBgkqhkiG9w0BBwagggNVMIIDUQIBADCCA0oGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMy3g-_eQ0cd6ORhQTAgEQgIIDGzeotYa-VUuOCjEdW7ugwGeBtEAjO5lRkruTNDRkiD4No09e4KMoJWSSGMEl9eznEG2SWrNUGjcjouQmLYToRujOvwVMB95_df2iOHUGOj91iQdMTonSBF91EBGNcJknTBT5946vBSu3yPx7_GNGARcyrMXvE8klj8hPK73iVhNOD7rrl3D6MunyxoVGLB2pyxPU021vhCkYXMooqC5IBf_godOs4Z3zqXTCPPNX2-cBlwXnKxik_i3lmLLhv5WD1w-FSLSzjYfZo1nSkmEHhfY4D3gI4hQbAj2nrL3KEVmP72zhTKo9h20oUEEKRMhujgZWaPP8OsIDvojIaHScjEH5OQ_czrneDTkHS8f5Nt0UUJy0kDSpdEObg-O_kgZzdWcl6m7PQPltEg9I6nOPNmZ438Ai0tiMjuZKd7WCtlr0LP_eyE5pfNxk02sCLseI8s1bj6P6EpAu-WEEMlZ34hxtO2G2EmOHj3K1fR78EndpcVOz7fYWTbcFUOKjJl9Sq5XilxbegWvV_WAvjoafaSEuUq114AFia-q6MgSd53usURbc3z-MvpBywJkw7L8NpWmOM_afCQtVxbt8uhckeZ2YgNCAzZc5Zn_gtMGFCsO4czznwS4UtEonqyu6YMM6cYfbCZk68rO-793S6MWEFOeIzCMoWWa_7SYTgmJx5oShokadrDC4eiiwnBB8zFODmmWqAVYKcm2EFjPiEtLBF7fCSh7Z1PNIIdQMkAdzcrrCpA8DYilNl2opl31lUP0CMaBzQjY4tRx5w30X-igPw2Lxqtmdtv-KAtS1e3LXAWQtAz4xf00kWqaNt5XDnVbtVb_5hlpidiBNblgaXYuEiA3qQduSdhREjLSLJBgV8u-JTzrk_GjEMsQkbfM4lrLBiOSaPqf-KTKtNBMgDx1FmLfp7eS6Yu9PKB6GXJQblG0wqWuH6PL2vmsbbLpmR5tUVQdQCi5yBq-eeh1l8QGrtUi4MeEYa2nAELWBbLALqPIeYq_nvRM0B57R2WZStE8HMSWNgrut2iM6EgQxjTw19cwXISNCPs6YKPz3oA)]
- The healthcare cost and utilization project: an overview. [[paper](https://www.researchgate.net/profile/Anne-Elixhauser-2/publication/11287812_The_Healthcare_Cost_and_Utilization_Project_An_overview/links/00b4952f101fa08914000000/The-Healthcare-Cost-and-Utilization-Project-An-overview.pdf?origin=journalDetail&_tp=eyJwYWdlIjoiam91cm5hbERldGFpbCJ9)]
- Pubmedqa: A dataset for biomedical research question answering. [[paper](https://arxiv.org/pdf/1909.06146)]
- Applying deep matching networks to Chinese medical question answering: a study and a dataset [[paper](https://link.springer.com/article/10.1186/s12911-019-0761-8)]
- Pathvqa: 30000+ questions for medical visual question answering. [[paper](https://arxiv.org/pdf/2003.10286)]
- Mimic-iv [[paper](https://physionet.org/content/mimiciv/3.1/)]
- Cord-19: The covid-19 open research dataset. [[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7251955/pdf/nihpp-2004.10706v4.pdf)]
- Overview of the MEDIQA 2021 shared task on summarization in the medical domain. [[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7251955/pdf/nihpp-2004.10706v4.pdf)]
- MedDialog: Large-scale medical dialogue datasets. [[paper](https://aclanthology.org/2020.emnlp-main.743.pdf)]
- Slake: A semantically-labeled knowledge-enhanced dataset for medical visual question answering. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9434010)]
- A benchmark for automatic medical consultation system: frameworks, tasks and datasets. [[paper](https://watermark.silverchair.com/btac817.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA4YwggOCBgkqhkiG9w0BBwagggNzMIIDbwIBADCCA2gGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMm-OPr9Xg1XBA2uGQAgEQgIIDOc0z2SSJ8RWBM3_5YHUFvxRhbMbceWNC_JE1g-8Fk4dae0J7TZVgbEiFHgygdVPxq8JExi89UUe4gUfdwy8mzZaB32L5E6s_NTESizJ397fzkP8zAafzzHdLnxAPezku496G9QMKbgUT4bk0n5e9Oe1wZpvFoIZbb49XfuiMgPfwZLH2R3cPAtjC2fz7pYt-MD-6ouKQ-Z7Bm5OWc4QeZOKzsltKl_cS6eY2_c7Hg7vYvHgBXpk4WRYyMN89hGmW-uTaA3qa3qKms2wSUAqNsUtuYK9KK-79xhLEvILji41JwoLl4As0NvlC9naaO_KKF4Rh99Fpx9paqzE-q_S2cctmL54B1Nh9Q-L9E2MwtzVGQLMPfFv_aueflIrA7-CpRGXVAz_GCngqvbLjLpvrkms19L9OCZ8HwMgVFIbOCRG9gDd0th812KPyLBFH41tFz_x9_gJGoF2K9tLArRoLu5krUSv7ahEz7TahHs3KOwwPtzwR5CbfFyRGfZ9PLIQRlZNRu91V7ICqL86jbwKltrovYgZ9qa4KNaxWiSoGk1ss0ZIaXVsTxlCqRUO_mtbTmWXDP4UcunXYFtyCaO-CzpiENVv4HYEfZ4q-SlBSTBHTO20lpC37sPpW3w75pUvnLhcIOeljmq-edV2iSBFk-_OUhQEwjiJfOJiZlturOL5yGK_Cgtv3VXubNgPlB-Z0NX4yYRL4NKUyVFkLyuFAc-RR8YMbd88ioDc9olUEwJ-BIGTo46Qzrlhb_VAeqLJ530ReVyY4azR-u2iZzsOVNprij_3ZxhG2zK5VZ9bmPjmaU0hnscWFQ1OiEPmEbk2KkfqqzgWVjCXBcOtUkMbwfHJD2FF6XCamgA2RBZHr7LYdqweNE3d8emNydt_AvoY0fjFPBl3MsIfo94bcfVulrUjcOnXMYvDP5Ysj-CE6qtnHfsg2qWqsZfGPr6X-jfJ6sn7z0y6q6rrjS7o_oIsZZUEaVoVcWk6WyQ_T1zJH4MJcILWYfhVeHl_n0LG2rQoU4RyWS6-iAfdfve-E4pDNN8ae2ZekVEHoR8OZ40LgeZ_KWSe-HfPCAakMo35uN88mKCHAML8QXEY79A)]
- Chq-summ: A dataset for consumer healthcare question summarization. [[paper](https://arxiv.org/pdf/2206.06581)]
- Medmcqa: A large-scale multi-subject multi-choice dataset for medical domain question answering. [[paper](https://proceedings.mlr.press/v174/pal22a/pal22a.pdf)]
- Large language models encode clinical knowledge. [[paper](https://arxiv.org/pdf/2212.13138)]
- Enhancing the chinese medical capabilities of large language model through expert feedback and real-world multi-turn dialogue. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29907)]
- Distributional semantics resources for biomedical text processing. [[paper](https://bio.nlplab.org/pdf/pyysalo13literature.pdf)]
- Towards building multilingual language model for medicine. [[paper](https://www.nature.com/articles/s41467-024-52417-z)]
- Huatuogpt, towards taming language model to be a doctor. [[paper](https://arxiv.org/pdf/2305.15075)]
- A foundation model utilizing chest ct volumes and radiology reports for supervised-level zero-shot detection of abnormalities. [[paper](https://arxiv.org/pdf/2403.17834v2)]



### 🧠 Naive RAG

- Development of a Liver Disease-Specific Large Language Model Chat Interface using Retrieval Augmented Generation [[paper](https://www.medrxiv.org/content/medrxiv/early/2023/11/11/2023.11.10.23298364.full.pdf)]

<div align="center">
  <img src="/06_Naive_RAG_Architecture.png" alt="Naive_RAG" width="60%"/>
</div>

- Development of a Liver Disease-Specific Large Language Model Chat Interface using Retrieval Augmented Generation [[paper](https://www.medrxiv.org/content/medrxiv/early/2023/11/11/2023.11.10.23298364.full.pdf)]
- ChatENT: Augmented Large Language Model for Expert Knowledge Retrieval in Otolaryngology - Head and Neck Surgery [[paper](https://www.medrxiv.org/content/10.1101/2023.08.18.23294283v2)]
- Large Language Models with Retrieval-Augmented Generation for Zero-Shot Disease Phenotyping, 2023 [[paper](https://arxiv.org/pdf/2312.06457)]
- Almanac: Retrieval-Augmented Language Models for Clinical Medicine [[paper](https://arxiv.org/pdf/2303.01229)]
- Retrieval Augmentation of Large Language Models for Lay Language Generation [[paper](https://arxiv.org/pdf/2211.03818)]

### 🌟 Advanced RAG

<div align="center">
  <img src="/07_Advanced_RAG_Architecture.png" alt="Advanced_RAG" width="60%"/>
</div>

- A RAG Chatbot for Precision Medicine of Multiple Myeloma [[paper](https://www.medrxiv.org/content/10.1101/2024.03.14.24304293v1.full.pdf)]
- A Context-based Chatbot Surpasses Radiologists and Generic ChatGPT in Following the ACR Appropriateness Guidelines [[paper](https://pubs.rsna.org/doi/epdf/10.1148/radiol.230970)]
- Performance of ChatGPT, human radiologists, and context-aware ChatGPT in identifying AO codes from radiology reports [[paper](https://www.nature.com/articles/s41598-023-41512-8#Sec1)]
- The Power of Noise: Redefining Retrieval for RAG Systems [[paper](https://dl.acm.org/doi/pdf/10.1145/3626772.3657834)]
- Integrating UMLS Knowledge into Large Language Models for Medical Question Answering [[paper](https://arxiv.org/pdf/2310.02778)]
- Augmenting Black-box LLMs with Medical Textbooks for Clinical Question Answering [[paper](https://arxiv.org/pdf/2309.02233)]

### 🛠️ Module RAG

<div align="center">
  <img src="/08_Modular_RAG.png" alt="Modular_RAG" width="60%"/>
</div>

- Retrieval-Augmented Dual Instruction Tuning [[paper](https://arxiv.org/pdf/2310.01352)]
- Health-LLM: Personalized Retrieval-Augmented Disease Prediction System [[paper](https://arxiv.org/pdf/2402.00746)]
- Precise Zero-Shot Dense Retrieval without Relevance Labels [[paper](https://arxiv.org/pdf/2212.10496)]
- Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning, 2024 [[paper](https://arxiv.org/pdf/2310.01061)]
- Think and Retrieval: A Hypothesis Knowledge Graph Enhanced Medical Large Language Models [[paper](https://arxiv.org/pdf/2312.15883)]
- An Open-Source Retrieval-Augmented Large Language Model System for Answering Medical Questions using Scientific Literature [[paper](https://arxiv.org/pdf/2310.16146)]
- Zero-Shot ECG Diagnosis with Large Language Models and Retrieval-Augmented Generation [[paper](https://proceedings.mlr.press/v225/yu23b/yu23b.pdf)]

- Chunking Optimization

- Clinical entity augmented retrieval for clinical information extraction [[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11743751/?utm_source=chatgpt.com)]
- BiomedRAG: A Retrieval Augmented Large Language Model for Biomedicine [[paper](https://arxiv.org/pdf/2405.00465)]
- ChunkRAG: Novel LLM-Chunk Filtering Method for RAG Systems [[paper](https://arxiv.org/pdf/2410.19572)]
- Mix-of-Granularity: Optimize the Chunking Granularity for Retrieval-Augmented Generation [[paper](https://aclanthology.org/2025.coling-main.384.pdf)]
- Enhancing Large Language Model Reliability: Minimizing Hallucinations with Dual Retrieval-Augmented Generation Based on the Latest Diabetes Guidelines [[paper](https://www.mdpi.com/2075-4426/14/12/1131?utm_source=chatgpt.com)]
- Clinical Entity Augmented Retrieval for Clinical Information Extraction [[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11743751/pdf/41746_2024_Article_1377.pdf)]

- Structure organization

- Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation [[paper](https://arxiv.org/abs/2408.04187)]
- MedRAG: Enhancing Retrieval-Augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot [[paper](https://arxiv.org/abs/2502.04413)]
- KG-Retriever: Efficient Knowledge Indexing for Retrieval-Augmented Large Language Models [[paper](https://arxiv.org/pdf/2412.05547)]
- Rationale-Guided Retrieval Augmented Generation for Medical Question Answering [[paper](https://web3.arxiv.org/pdf/2411.00300)]

- Query Routing

- The Geometry of Queries: Query-Based Innovations in Retrieval-Augmented Generation [[paper](https://arxiv.org/abs/2407.18044)]
- Unsupervised Query Routing for Retrieval-Augmented Generation [[paper](https://arxiv.org/abs/2501.07793)]
- Rationale-Guided Retrieval Augmented Generation for Medical Question Answering [[paper](https://arxiv.org/abs/2411.00300)]

- Query Expansion

- BioRAGent: A Retrieval-Augmented Generation System for Showcasing Generative Query Expansion and Domain-Specific Search for Scientific Q&A [[paper](https://arxiv.org/pdf/2412.12358)]
- Enhancing Retrieval-Augmented Generation: A Study of Best Practices [[paper](https://arxiv.org/pdf/2501.07391)]
- Blended RAG: Improving RAG Accuracy with Semantic Search and Hybrid Query-Based Retrievers [[paper](https://arxiv.org/pdf/2404.07220)]

- Retriever Fine-tuning

- JMLR: Joint Medical LLM and Retrieval Training for Enhancing Reasoning and Professional Question Answering Capability [[paper](https://arxiv.org/pdf/2402.17887)]
- SeRTS: Self-Rewarding Tree Search for Biomedical Retrieval-Augmented Generation [[paper](https://arxiv.org/pdf/2406.11258)]
- MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models [[paper](https://arxiv.org/pdf/2410.13085)]
- Onco-Retriever: Generative Classifier for Retrieval of EHR Records in Oncology [[paper](https://arxiv.org/pdf/2404.06680)]

### 📊 Graph RAG

<div align="center">
  <img src="/09_Graph_RAG.png" alt="Graph_RAG" width="80%"/>
</div>

- A Survey of Graph Retrieval-Augmented Generation for Customized Large Language Models [[paper](https://arxiv.org/pdf/2501.13958)]
- Medical graph RAG: Towards safe medical large language model via graph retrieval-augmented generation [[paper](https://arxiv.org/pdf/2408.04187)]
- Reasoning-Enhanced Healthcare Predictions with Knowledge Graph Community Retrieval [[paper](https://arxiv.org/pdf/2410.04585)]
- Biomedical knowledge graph-optimized prompt generation for large language models [[paper](https://watermark.silverchair.com/btae560.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA4YwggOCBgkqhkiG9w0BBwagggNzMIIDbwIBADCCA2gGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMz0NrXzOwUS3wI4AmAgEQgIIDOVoIIve4Fm4epbFnBWgMWKSE7Vq9uQkANL-R3K1hkz3Z987Ii_lSfDmMPvmS_1HI9MnQ6osjGb52rwuB841pRkuFeuyexjS2AcMp9yPLvBnjqJAX8s2LYDFWC84avV-_3M4_1H1399CnUHTo3zQP7lR_eX3LtFnV5z8ekUHLCbAJXPKMmZ1lhW80qds7X45bhI47kHDzZjJdg1mp7rqpnsoDNJmryRujeQU4TZQSFbmgnzpVXqFiotciOrgD90eKGdPxeK8PBGLNrYDmPaHFINzGLgeXhRvh9Pi6Facd-oBcyJ59xpe_wghe8NvRSLGutGXJyLFyoweN6C7BBjlylVn1yiVy7_q7vLy9a7RzaeMc-SH50Pb8zosbcsZL68J_vM53-9LgaVvqGIp6huiqmFTluaS5QVqMRV9zC3ry84gbi2dTLHjFz0CS5hbmh7ys87XdY-TBg1sJCbQuBvFUFCGvEasnQUaqAQ2VmtUoXQ9Dp9VjS-CgZKYS68g-iqRyxKQAWyPrIML0hV6BW1pZvzD3XBhFM0FJHSt6r8HCKPNCdapSn3ujEQbKd4TLhE5icJBu4HTsfdqCGc0szzAm1rtnvH6KaYWUQY_dNlF-ne9VC9pfIVYnioMj-yvygIWs7n2J38gUcxisPgGAkibYAmxwm0194tZ-e7is6b_bbfdEe4pEJCeDjQcLeCLJm6JWuCo0dSbpDwex8wZAeQMZEUcZ-Fxbcvl1bfPaXy08_TdVQS63zQZV6FB38K3rNMmwh5DUx91jlld0esgzHAQV8lzb8kMEclWi9hBFQ3SQIliy32PIHCcJcBJIx7_Ww7VeU64y7OgkI_FnY3M6UaYmndoz9sS84XtOIdY0SK4uzGO2Gho78bOqTALe3wNAlRbfj9QPGtTGj_vASywfDoRaCsd6zc2QiHW7p6iqNbfpEhAqMDtGyLTm44BZQ1_cDdRJd63cc0mgRCeTMhSmt8Fx-SmDzGDbPap4uMOB_HYXb0A2kSioBL1X0Xb-B4xjKXy087AaGkE7-U95YaQcTJRkw1nFsgMh6bn8DDEn8OAPKLavJJ7TpBBpoU3A1kh2cAnZQcGLSuER4Uglng)]
- Hykge: A hypothesis knowledge graph enhanced framework for accurate and reliable medical LLMs responses [[paper](https://watermark.silverchair.com/btae560.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA4YwggOCBgkqhkiG9w0BBwagggNzMIIDbwIBADCCA2gGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM7V6qYfvT9i3ZkdC2AgEQgIIDOcO2u2KbJus1SzIF8LTPp4LdNokZzmtw5AwnZrXjZVbv_cEKBCi3DpiYFo_UdVKQSi6-is6WBJgOdnT77X2t0AqZ6PDHi9R038upcAzyCjcyHNE1uleB8phMpkR2qFJV9CrBv403Lge2yjLBjG1F5sKkyGtDWATN5iRUThYOzDqMn5DJ0TVfE0zUo3S-eC0bNSE6AP3np-P9pr8Fpy1VluNz0ZRd2jxRiDovfmpZihY9DB-QNAA0BKC-P6KGoebR6M3esHGBl9BaFZZgONn3_ErPCqhb9pv1MTj7hVZV5mavpzh3aYoRyXyM3qlsAbF24FgUzyaiz25YF7AwbaY1jHFFnX9dAfkUcvu3oleRjZP1EO7c33rTDZSUS1LnCV8soX_wZrw5f2CEdeV-IHpSlzn0IOQPwvXi2E3IhvheTCmEoVyUXqJjyLmRg7HP_q21cuTnakI67yZTPBUsSVN3vw4kPe0j7SGaPoIDrEnqSY9UqYz5PBtAWGIU_7aed7_b3-LhKPjQP2UiV1c8PKY93Kq4BSQyhTedaZV61u0ZnzO9MVrrYlxTmXlRa20FmKC7svP_9eKjdtXSuIo5eAZ2YBQenafyq4c5mZXh9SQBZIwkw5-VY_EKwRtykFKpcGKCP6kGMCaBUl-uVE6UPjQ34Ig0A3GFthUmyDJtKVeO5DBwN9mrjN6a4BU3bU3iWmweWvKTYZzgIk7KKsGHg3DecXnuc3k6CJJnNWYBnI3uHYWT9JN4PAPUDJhRpM2FHzRXz6_Vd3meQmT6DN2cjKKGm7_3lBkd4OZcjJe5rZuY4l5k2TV1TAUFIZ0g76EdbVZiOxjpUlOeAB5dtr9875SgHHM_m4FisxkQKqyouDbc4aIGFB1IBpdMreLEcOYZGXRZ2ihR89A4bg8-j12oSHjmdWxlLfdxQT1zG5yYx8Wu9SyXeI1-Espe-AdvoId-H-rB3B5NymS-OiRL44f-kGIrhyCDJKifgYD4hdyZ-t7Lu2a1OH1rOqOycvbEQBg04kydK9Sly7Ary29x0lNYDJ0pz8wCaR-wojK-O2Pd6IB2V74KgmYodQF0FmU4cr61M6RgqQQYwBGYXaAbpA)]

### 🤖 Agentic RAG

<div align="center">
  <img src="/10_Agentic_RAG.png" alt="Agentic_RAG" width="60%"/>
</div>

- AIPatient: Simulating Patients with EHRs and LLM Powered Agentic Workflow [[paper](https://arxiv.org/pdf/2409.18924)]
- Developing an Artificial Intelligence Tool for Personalized Breast Cancer Treatment Plans based on the NCCN Guidelines [[paper](https://arxiv.org/pdf/2502.15698)]
- Medagent-pro: Towards multi-modal evidence-based medical diagnosis via reasoning agentic workflow [[paper](https://arxiv.org/pdf/2503.18968)]
- Towards interpretable radiology report generation via concept bottlenecks using a multi-agentic RAG [[paper](https://arxiv.org/pdf/2412.16086)]
- Developing an Artificial Intelligence Tool for Personalized Breast Cancer Treatment Plans based on the NCCN Guidelines [[paper](https://arxiv.org/pdf/2502.15698)]
- AIPatient: Simulating Patients with EHRs and LLM Powered Agentic Workflow [[paper](https://arxiv.org/pdf/2409.18924)]

## 🩺 Applications & Use Cases

<div align="center">
  <img src="/11_Applications_and_Usecases.png" alt="Applications_Outline" width="60%"/>
</div>

### 🎓 Medical Training & Education

- Mimic-extract: A data extraction, preprocessing, and representation pipeline for MIMIC-III [[paper](https://dl.acm.org/doi/pdf/10.1145/3368555.3384469)]
- Transforming healthcare education: Harnessing large language models for frontline health worker capacity building using retrieval-augmented generation [[paper](https://www.medrxiv.org/content/10.1101/2023.12.15.23300009v1.full.pdf)]
- Rationale-Guided Retrieval Augmented Generation for Medical Question Answering [[paper](https://arxiv.org/pdf/2411.00300)]
- Benchmarking retrieval-augmented generation for medicine [[paper](https://aclanthology.org/2024.findings-acl.372.pdf)]

### 🩺 Clinical Decision Support

<div align="center">
  <img src="/12_CLEAR_RAG.png" alt="CLEAR_RAG" width="60%"/>
</div>

- Almanac—retrieval-augmented language models for clinical medicine [[paper](https://ai.nejm.org/doi/abs/10.1056/AIoa2300068)]
- Development and testing of a novel large language model-based clinical decision support systems for medication safety in 12 clinical specialties [[paper](https://arxiv.org/pdf/2402.01741)]
- Automatic chain of thought prompting in large language models [[paper](https://www.chatgpthero.io/wp-content/uploads/2023/12/2210.03493.pdf)]
- Clinical entity augmented retrieval for clinical information extraction [[paper](https://www.nature.com/articles/s41746-024-01377-1)]
- Improving retrieval-augmented generation in medicine with iterative follow-up questions [[paper](https://www.worldscientific.com/doi/epdf/10.1142/9789819807024_0015)]

### 📖 Medical Literature Summarization

<div align="center">
  <img src="/13_RefAI_Architecture.png" alt="RefAI" width="40%"/>
</div>

- RefAI: a GPT-powered retrieval-augmented generative tool for biomedical literature recommendation and summarization [[paper](https://www.worldscientific.com/doi/epdf/10.1142/9789819807024_0015)]
- Clinfo.ai: An Open-Source Retrieval-Augmented Large Language Model System for Answering Medical Questions using Scientific Literature [[paper](https://www.worldscientific.com/doi/epdf/10.1142/9789811286421_0002)]
- Ccs explorer: Relevance prediction, extractive summarization, and named entity recognition from clinical cohort studies [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10020807)]

### 📊 Electronic Health Record Analysis

<div align="center">
  <img src="/14_Knowledge_Graph.png" alt="MedRAG" width="50%"/>
</div>

- MedRAG: Enhancing Retrieval-Augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot [[paper](https://arxiv.org/abs/2502.04413)]
- Experience Retrieval-Augmentation with Electronic Health Records Enables Accurate Discharge QA [[paper](https://arxiv.org/pdf/2503.17933)]
- Large language models with retrieval-augmented generation for zero-shot disease phenotyping [[paper](https://arxiv.org/pdf/2312.06457)]
