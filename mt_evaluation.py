import openai
import numpy as np
import sacrebleu

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

_all_ = ['get_gemba_scores', 'get_ea_prompt_scores', 'get_sacrebleu', 'evaluate_sentence']

def _make_prompt(source, reference, target, prompt):
    message_text = prompt.replace('<SRC>', source) \
                         .replace('<REF>', reference) \
                         .replace('<TGT>', target)

    return message_text


#GEMBA SCORE
def get_gemba_scores(api_key, model, source_sentences, reference_sentences, target_sentences, temperature):

    client = openai.OpenAI(api_key=api_key)
    prompts = []
    for src, ref, tgt in zip(source_sentences, reference_sentences, target_sentences):
        prompts.append(_make_prompt(src, ref, tgt, _gemba_prompt))
    
    responses = _call_openai(client, prompts, model, temperature)

    try:
        scores = []
        for response in responses:
            response_text = response.choices[0].message.content
            scores.append(float(response_text))
    except ValueError:
        raise ValueError(f"Expected a numeric score, but got: {response_text}")

    return scores


#EAPROMPT
def get_ea_prompt_scores(api_key, model, source_sentences, reference_sentences, target_sentences, temperature):

    client = openai.OpenAI(api_key=api_key)
    prompts = []
    for src, ref, tgt in zip(source_sentences, reference_sentences, target_sentences):
        prompts.append(_make_prompt(src, ref, tgt, _ea_prompt))
    
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

    return scores


#OPENAI API CALL
def _call_openai(client, prompts, model = 'gpt-3.5-turbo', temperature = 0.7):
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
def get_sacrebleu(reference_sentences, candidate_sentences, tok):

    if len(reference_sentences) != len(candidate_sentences):
        raise ValueError("Error: The number of reference and candidate sentences must be equal.")

    references = [[ref] for ref in reference_sentences]  # Convert to list of lists
    bleu = sacrebleu.corpus_bleu(candidate_sentences, references, tokenize=tok)
    
    return bleu.score


def evaluate_sentence(method, src, ref, tgt, api_key=None, model=None, temperature = 0.7, tok='none'):
    if method == 'gemba':
        return np.mean(get_gemba_scores(api_key, model, src, ref, tgt, temperature))
    elif method == 'eaprompt':
        return np.mean(get_ea_prompt_scores(api_key, model, src, ref, tgt, temperature))
    elif method == 'sacrebleu':
        return get_sacrebleu(ref, tgt, tok)



#test
# if __name__ == '__main__':

#     #test sentences
#     src = ['You can come back any time as our chat service window is open 24/7', 'I sincerely hope you get to find a resolution']
#     ref = ['Sie können jederzeit wiederkommen, da unser Chat-Service-Fenster täglich rund um die Uhr geöffnet ist', 'Ich hoffe wirklich, dass Sie eine Lösung finden werden']
#     tgt = ['Sie können jederzeit wiederkommen, denn unser Chat-Fenster ist rund um die Uhr geöffnet.', 'Ich hoffe aufrichtig, dass Sie eine Lösung finden werden.']
#     api_key = " "

#     print(evaluate_sentence('gemba', api_key = api_key,
#                             model = 'gpt-3.5-turbo', src = src, ref = ref, tgt = tgt))

#     print(evaluate_sentence('eaprompt', api_key = api_key,
#                             model = 'gpt-3.5-turbo', src = src, ref = ref, tgt = tgt))
                            
#     print(evaluate_sentence('sacrebleu', src = src, ref = ref, tgt = tgt))