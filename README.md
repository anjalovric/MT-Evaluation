# MT-Evaluation

1. Download mt_evaluation.py to project folder
2. Import using  
        `from mt_evaluation import *`
3. Call functions: evaluate, get_gemba_scores, get_ea_prompt_scores or get_sacrebleu


## Functions:

1. **evaluate**(method, src, ref, tgt, api_key=None, model= 'gpt-3.5-turbo', temperature = 0.7, tok='none', seg_scores_to_json = False)

    - method - 'gemba', 'eaprompt' or 'sacrebleu'  
    - src - list of source sentences  
    - ref - list of reference sentences  
    - tgt - list of translated sentences  
    - api_key - your OpenAI API key, only used for gemba and eaprompt  
    - model - choose OpenAI model for evaluation (only for gemba and eaprompt). List of models: 'gpt-4', 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini'  
    - temperature - temperature for chosen OpenAI model  
    - tok - only for sacrebleu. For Chinese language use tok='zh', for Japanese tok='ja-mecab' and for Korean tok='ko-mecab'. For other languages skip this parameter or use tok='none'  
    - seg_scores_to_json - for saving gemba and eaprompt scores for each segment in json file named 'scores_gemba.json' and 'scores_eaprompt.json'  

    Returns one score for a list of sentences.  


2. **get_gemba_scores**(source_sentences, reference_sentences, target_sentences, api_key, model= 'gpt-3.5-turbo', temperature = 0.7, seg_scores_to_json = False)

    - source_sentences - list of source sentences  
    - reference_sentences - list of reference sentences  
    - target_sentences - list of translated sentences  
    - api_key - your OpenAI API key  
    - model - choose OpenAI model for evaluation (only for gemba and eaprompt). List of models: 'gpt-4', 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini'  
    - temperature - temperature for chosen OpenAI model  
    - tok - only for sacrebleu. For Chinese language use tok='zh', for Japanese tok='ja-mecab' and for Korean tok='ko-mecab'.  
    - seg_scores_to_json - for saving gemba and eaprompt scores for each segment in json file named 'scores_gemba.json' and 'scores_eaprompt.json'  

    Returns list of gemba scores, one for each sentence in the list.  


3. **get_ea_prompt_scores**(source_sentences, reference_sentences, target_sentences, api_key, model= 'gpt-3.5-turbo', temperature = 0.7, seg_scores_to_json = False)

    - source_sentences - list of source sentences  
    - reference_sentences - list of reference sentences  
    - target_sentences - list of translated sentences  
    - api_key - your OpenAI API key  
    - model - choose OpenAI model for evaluation (only for gemba and eaprompt). List of models: 'gpt-4', 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini'  
    - temperature - temperature for chosen OpenAI model  
    - tok - only for sacrebleu. For Chinese language use tok='zh', for Japanese tok='ja-mecab' and for Korean tok='ko-mecab'.  
    - seg_scores_to_json - for saving gemba and eaprompt scores for each segment in json file named 'scores_gemba.json' and 'scores_eaprompt.json'  

    Returns list of eaprompt scores, one for each sentence in the list.  

4. **get_sacrebleu**(reference_sentences, candidate_sentences, tok)

    - reference_sentences - list of reference sentences  
    - candidate_sentences - list of translated sentences  
    - tok - For Chinese language use tok='zh', for Japanese tok='ja-mecab' and for Korean tok='ko-mecab'. For other languages skip this parameter or use tok='none'   

    Returns corpus bleu score, one score for list of sentences.  
    More on sacrebleu: https://github.com/mjpost/sacrebleu/tree/master
