# master-thesis
Repository for my masters' thesis.

# Thesis notes

- check-worthy claim - a claim for which the general public would be interested in knowing the truth
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
- checkworthiness:
  1. CLEF2023 CheckThat! Task 1: Check-Worthiness in Multimodal and Unimodal Content (CT23)

- verification:
  1. FEVER
