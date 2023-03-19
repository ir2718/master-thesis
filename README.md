# master-thesis
Repository for my masters' thesis.

# Thesis notes
- What is claim checkworthiness detection?

<p align="center">
  <img src="https://mitp.silverchair-cdn.com/mitp/content_public/journal/tacl/10/10.1162_tacl_a_00454/4/m_tacl_a_00454_f002.png?Expires=1679678665&Signature=V0KUpjejRK8TBrRnN~-47HwWrcvawCPGgCSLLWs~r36NupVqjPR1FPDhAU3Rf906bSSk8f-8fMKo8f6ZmUF5rszLHNTN~xG2jT7p0YaVhXolR97NTmhvEyLdJ5l3R2uuXmrIeQVieqVSNXXuZpSqigZ4y-AyCj4el7RPZI3yVCZkWGhNwgwrnKPR~DUAnD-Ig4nD97E17kYkPleooQZrstaNxpAmUwDMBIZpoJhie8fHATlnp8GZfoyipzNct6UShcgn~~Esnp2vtNjr~fMe~qAPKXa3UtVV~mPkp1UA080hDT18BWIuTS3Yj9jNJK3ugauUTIFgdevTdNxf5AN36Q__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA" />
</p>

- check-worthy claim - claim for which the general public would be interested in knowing the truth
- it's important that the claim is even checkable
  - e.g. "I'm thinking about my thesis" is a factual statement, although it isn't checkable 


The subtasks needed to solve this problem:
1. **claim detection** 
    - identify claims which require verification
    - binary classification problem or importance ranking

2. **evidence retrieval** 
    - find information to refute or confirm the claim
    - important to take into account outer knowledge and essential for verdict justification
    - not all information is trustworthy - usually, some information sources are considered to be implicitly trustworthy

3. **claim verification**

    - **verdict prediction**
      - determines the veracity of the claim based on retrieved evidence
      - binary classification problem, multi class classification problem (true, false, not enough information) or multi label classification problem

    - **justification production**
       - produces an explanation for verdict prediction
       - using attention weights, logic-based explanation or summarization by generating explanations
       
### Datasets
- only checkworthiness and verification will be covered in this thesis
- checkworthiness:
  1. CLEF2023 CheckThat! Task 1: Check-Worthiness in Multimodal and Unimodal Content (CT23)
  2. CLEF2021 CheckThat! Task 1: Check-Worthiness Estimation (CT21)

- verification:
  1. FEVER
  2. SciFact
  
### Baselines:
- each baseline was trained on 20 epochs

| Model     | Dataset         | learning rate | batch size | weight decay | F1 score | Best @ epoch |
| --------- | --------------- | ------------- | ---------- | ------------ | -------- | ------------ |
| BERT      | CT23            | 2e-5          | 32         | 1e-2         | 0.758    | 19           |
| RoBERTa   | CT23            | 2e-5          | 32         | 1e-2         | 0.770    | 4            |
| BERT      | CT21            | 2e-5          | 32         | 1e-2         | 0.701    | 8            |
| RoBERTa   | CT21            | 2e-5          | 32         | 1e-2         | 0.758    | 7            |
| BERT      | FEVER           | 2e-5          | 32         | 1e-2         |          |              |
| RoBERTa   | FEVER           | 2e-5          | 32         | 1e-2         |          |              |
| BERT      | SciFact         | 2e-5          | 32         | 1e-2         |          |              |
| RoBERTa   | SciFact         | 2e-5          | 32         | 1e-2         |          |              |

