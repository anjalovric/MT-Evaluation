import openai
import numpy as np
import sacrebleu
import json

#Method names: gemba, eaprompt, sacrebleu

_gemba_prompt = """Score the following translation with respect to human reference on a continuous scale 0 to 100 where score of zero means "no meaning preserved" and score of one hundred means "perfect meaning and grammar". Return only score, do not add any additional information!
Source: <SRC>
Human reference: <REF>
Machine translation: <TGT>
Score:"""

_ea_prompt = """Source: <SRC>
Reference: <REF>
Translation: <TGT>
Based on the given source and reference, identify the major and minor errors in this translation. Note that Major errors refer to actual translation or grammatical errors, and Minor errors refer to smaller imperfections, and purely subjective opinions about the translation. 
Output ONLY in format "x, x", indicating the number of major and minor errors as response. DO NOT ADD other information!"""

_openai_models = ['gpt-4', 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini']

_all_ = ['get_gemba_scores', 'get_ea_prompt_scores', 'get_sacrebleu', 'evaluate_sentence']


#Add source, reference and target sentence to prompt
def _add_sentences_to_prompt(source, reference, target, prompt):
    message_text = prompt.replace('<SRC>', source) \
                         .replace('<REF>', reference) \
                         .replace('<TGT>', target)

    return message_text


#Make prompt list using list of source, reference and target sentences
def _make_prompt(source_sentences, reference_sentences, target_sentences, method):
    prompt_type = _gemba_prompt if method == 'gemba' else _ea_prompt
    prompts = []
    for src, ref, tgt in zip(source_sentences, reference_sentences, target_sentences):
        prompts.append(_add_sentences_to_prompt(src, ref, tgt, prompt_type))

    return prompts


#Check if number of sentences is the same in all three lists and if model name is correct
def _validate(src_number_of_sentences, ref_number_of_sentences, tgt_number_of_sentences, model):

    if not (src_number_of_sentences == ref_number_of_sentences == tgt_number_of_sentences):
        raise ValueError(f"Error: The number of source, reference and target sentences must be equal. Source: {src_number_of_sentences}, Reference: {ref_number_of_sentences}, Target {tgt_number_of_sentences}")
    
    if model not in _openai_models:
        raise ValueError(f"Invalid model name. Available models: {_openai_models}")


#Save list of scores to json file
def _save_to_json(scores, method):
    try:
        with open(f'scores_{method}.json', 'w', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)
        
        print("Json saved.")
    except ValueError:
        raise ValueError("Error while saving to json.")


#GEMBA SCORE
def get_gemba_scores(source_sentences, reference_sentences, target_sentences, api_key, model= 'gpt-3.5-turbo', temperature = 0.7, seg_scores_to_json = False):

    _validate(len(source_sentences), len(reference_sentences), len(target_sentences), model)
    
    client = openai.OpenAI(api_key=api_key)
    prompts = _make_prompt(source_sentences, reference_sentences, target_sentences, 'gemba')
    
    responses = _call_openai(client, prompts, model, temperature)

    try:
        scores = []
        for response in responses:
            response_text = response.choices[0].message.content
            scores.append(float(response_text))
    except ValueError:
        raise ValueError(f"Expected a numeric score, but got: {response_text}")
    
    if seg_scores_to_json:
        _save_to_json(scores, 'gemba')

    return scores


#EAPROMPT
def get_ea_prompt_scores(source_sentences, reference_sentences, target_sentences, api_key, model= 'gpt-3.5-turbo', temperature = 0.7, seg_scores_to_json = False):

    _validate(len(source_sentences), len(reference_sentences), len(target_sentences), model)
    
    client = openai.OpenAI(api_key=api_key)
    prompts = _make_prompt(source_sentences, reference_sentences, target_sentences, 'eaprompt')
    
    responses = _call_openai(client, prompts, model, temperature)

    try:
        scores = []
        for response in responses:
            response_text = response.choices[0].message.content
            major_errors = float(response_text.split(',')[0].strip())
            minor_errors = float(response_text.split(',')[1].strip())
            scores.append(100 - 5*major_errors - minor_errors)
    except ValueError:
        raise ValueError(f"Expected a numeric score, but got: {response_text}")

    if seg_scores_to_json:
        _save_to_json(scores, 'eaprompt')

    return scores


#OPENAI API CALL
#Call of OpenAI API for list of prompts
def _call_openai(client, prompts, model, temperature = 0.7):
    responses = []
    for prompt_text in prompts:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=temperature
        )
        responses.append(response)

    return responses


#sacrebleu
def get_sacrebleu(reference_sentences, candidate_sentences, tok = 'none'):

    if len(reference_sentences) != len(candidate_sentences):
        raise ValueError("Error: The number of reference and candidate sentences must be equal.")

    references = [[ref] for ref in reference_sentences]  # Convert to list of lists
    bleu = sacrebleu.corpus_bleu(candidate_sentences, references, tokenize=tok)
    
    return bleu.score


#Evaluate list of sentences with chosen method
def evaluate(method, src, ref, tgt, api_key=None, model= 'gpt-3.5-turbo', temperature = 0.7, tok='none', seg_scores_to_json = False):
    if method == 'gemba':
        return np.mean(get_gemba_scores(src, ref, tgt, api_key, model, temperature, seg_scores_to_json))
    elif method == 'eaprompt':
        return np.mean(get_ea_prompt_scores(src, ref, tgt, api_key, model, temperature, seg_scores_to_json))
    elif method == 'sacrebleu':
        return get_sacrebleu(ref, tgt, tok)
    else:
        raise ValueError("Method not found! Available methods: gemba, eaprompt, sacrebleu")


#test
#Example of evaluate funtion call
# if __name__ == '__main__':

#     #test sentences
#     src = ['You can come back any time as our chat service window is open 24/7', 'I sincerely hope you get to find a resolution']
#     ref = ['Sie können jederzeit wiederkommen, da unser Chat-Service-Fenster täglich rund um die Uhr geöffnet ist', 'Ich hoffe wirklich, dass Sie eine Lösung finden werden']
#     tgt = ['Sie können jederzeit wiederkommen, denn unser Chat-Fenster ist rund um die Uhr geöffnet.', 'Ich hoffe aufrichtig, dass Sie eine Lösung finden werden.']
#     api_key = " "

#     print(evaluate('gemba', api_key = api_key,
#                             model = 'gpt-3.5-turbo', src = src, ref = ref, tgt = tgt))

#     print(evaluate('eaprompt', api_key = api_key,
#                             model = 'gpt-3.5-turbo', src = src, ref = ref, tgt = tgt))
                            
#     print(evaluate('sacrebleu', src = src, ref = ref, tgt = tgt))