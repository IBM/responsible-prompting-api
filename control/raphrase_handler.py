from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

def generate(model, tokenizer, system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=128,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def clean_response(response):
    match = re.search("<new_sentence>(.|\n)*?<\/new_sentence>", response)

    if match:
        return re.sub(r"<[^>]+>", "", match.group(0)).strip()
    else:
        return None

def rephrase_local(recommendation, hf_token, model_id = "meta-llama/Meta-Llama-3-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token = hf_token)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_auth_token = hf_token
    )

    system_prompt = "Respond strictly in the following format between these tags: <new_sentence></new_sentence>"
    user_prompt = "Rewrite the following sentence to avoid " + recommendation["value"] + ". DO NOT INTERPRET IT.\n" + recommendation["sentence"]
    response = generate(model, tokenizer, system_prompt, user_prompt)

    return {
        'sentence_index': recommendation['sentence_index'],
        'sentence': recommendation["sentence"],
        'prompt': user_prompt,
        'response': response,
        'suggestion': clean_response(response)
    }

