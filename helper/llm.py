import requests
import openai
from openai import OpenAI
import google.generativeai as genai



# Function to call LLaMA3 API
def llama3_generate_translation(llama3_api_endpoint, Q, descriptions, model, n_shot=1, threshold=1):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a AI translator for converting sparql queries into normal natural language questions<|eot_id|><|start_header_id|>user<|end_header_id|>
            Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: select distinct ?obj where {{ [Delta Air Lines] [house publication] ?obj . ?obj [instance of] [periodical] }}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|> What periodical literature does Delta Air Lines use as a moutpiece?
            <|eot_id|><|start_header_id|>user<|end_header_id|> Here are some contextual data to the query, remember it when translating: {descriptions}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|> OK.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: '{Q}' <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    payload = {
        "model": model,
        "prompt": prompt,
    }
    response = requests.post(llama3_api_endpoint, json=payload, stream=True)
    response_text = ""
    for line in response.text.split("\n"):
        try:
            response_text += line.split('response":"')[1].split('","done":')[0]
        except:
            pass


    for _ in range(n_shot):
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                You are a AI translator for converting sparql queries into normal natural language questions<|eot_id|><|start_header_id|>user<|end_header_id|>
                Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: select distinct ?obj where {{ [Delta Air Lines] [house publication] ?obj . ?obj [instance of] [periodical] }}
                <|eot_id|><|start_header_id|>assistant<|end_header_id|> What periodical literature does Delta Air Lines use as a moutpiece?
                <|eot_id|><|start_header_id|>user<|end_header_id|> Here are some contextual data to the query, remember it when translating: {descriptions}
                <|eot_id|><|start_header_id|>assistant<|end_header_id|> OK.
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                Translate the sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: '{Q}' <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                {response_text}<|eot_id|><|start_header_id|>user<|end_header_id|>
                Reflect on your answer and improve upon it; respond with only the improved question in one sentence only<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        payload = {
            "model": model,
            "prompt": prompt,
        }
        response = requests.post(llama3_api_endpoint, json=payload, stream=True)
        response_text = ""
        for line in response.text.split("\n"):
            try:
                response_text += line.split('response":"')[1].split('","done":')[0]
            except:
                pass
        if threshold == None:
            pass
            break
    print(response_text)
    return response_text


def llama3_compare_translations(llama3_api_endpoint, Q, translations, model):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a AI translator for selecting the best natural language translation of a sparql query.<|eot_id|><|start_header_id|>user<|end_header_id|>
            Select only a single translation from the list of given translations that is the best equivalent of the sparql query. Translations: {translations}, Query: {Q}. Respond with the selected translation only.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    payload = {
        "model": model,
        "prompt": prompt,
    }
    response = requests.post(llama3_api_endpoint, json=payload, stream=True)
    response_text = ""
    for line in response.text.split("\n"):
        try:
            response_text += line.split('response":"')[1].split('","done":')[0]
        except:
            pass
    # print(f"llama response {response_text}")
    return response_text


# Function to call OpenAI GPT API
def gpt_generate_translation(openai_api_key, Q, descriptions, n_shot=0, threshold=1):

    gpt_model = 'gpt-4o' #todo put this into config

    client = OpenAI(api_key=openai_api_key)
    prompt = f"""You are a AI translator for converting sparql queries into its natural language questions.
            Here are some examples:

            Query: select distinct ?obj where {{ [Delta Air Lines] [house publication] ?obj . ?obj [instance of] [periodical] }}
            Translation: What periodical literature does Delta Air Lines use as a moutpiece?

            Query: ASK WHERE {{ [Judi Dench] [award received] [Tony Award for Best Direction of a Play] . [Judi Dench] [award received] [Praemium Imperiale] }}
            Translation: Did Judi Dench receive the Tony Award for Best Direction of a Play and the Praemium Imperiale

            Query with a blank node:
            Query: SELECT ?obj WHERE {{ [Angela Merkel] [position held] ?s . ?s [position held] ?obj . ?s [start time] ?x filter(contains(YEAR(?x),'1994')) }}
            Translation: Which position did Angela Merkel hold on November 10, 1994?

            Query: select ?ent where {{ ?ent [instance of] [Class IB flammable liquid] . ?ent [lower flammable limit] ?obj }} ORDER BY DESC(?obj)LIMIT 5 
            Translation: Which 5 Class IB flammable liquids have the highest lower flammable limit ? 

            Context information: {descriptions}
            Translate the following sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: '{Q}'"""
    response = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    response_text = response.choices[0].message.content

    for _ in range(n_shot):
        prompt = f"""You are a AI translator for converting sparql queries into its natural language questions.
            Here are some examples:

            Query: select distinct ?obj where {{ [Delta Air Lines] [house publication] ?obj . ?obj [instance of] [periodical] }}
            Translation: What periodical literature does Delta Air Lines use as a moutpiece?

            Query: ASK WHERE {{ [Judi Dench] [award received] [Tony Award for Best Direction of a Play] . [Judi Dench] [award received] [Praemium Imperiale] }}
            Translation: Did Judi Dench receive the Tony Award for Best Direction of a Play and the Praemium Imperiale

            Query with a blank node:
            Query: SELECT ?obj WHERE {{ [Angela Merkel] [position held] ?s . ?s [position held] ?obj . ?s [start time] ?x filter(contains(YEAR(?x),'1994')) }}
            Translation: Which position did Angela Merkel hold on November 10, 1994?

            Query: select ?ent where {{ ?ent [instance of] [Class IB flammable liquid] . ?ent [lower flammable limit] ?obj }} ORDER BY DESC(?obj)LIMIT 5 
            Translation: Which 5 Class IB flammable liquids have the highest lower flammable limit ? 

            Context information: {descriptions}
            Translate the following sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: '{Q}'
            {response_text}
            Reflect on your answer and improve it if necessary; respond with only the improved question in one sentence only."""
        response = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        response_text = response.choices[0].message.content
        if threshold == None:
            pass
            break
    return response_text


def gemini_generate_translation(api_key, Q, descriptions, n_shot=0, threshold=1):
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    prompt = f"""You are a AI translator for converting sparql queries into its natural language questions.
            Here are some examples:

            Query: select distinct ?obj where {{ [Delta Air Lines] [house publication] ?obj . ?obj [instance of] [periodical] }}
            Translation: What periodical literature does Delta Air Lines use as a moutpiece?

            Query: ASK WHERE {{ [Judi Dench] [award received] [Tony Award for Best Direction of a Play] . [Judi Dench] [award received] [Praemium Imperiale] }}
            Translation: Did Judi Dench receive the Tony Award for Best Direction of a Play and the Praemium Imperiale

            Query with a blank node:
            Query: SELECT ?obj WHERE {{ [Angela Merkel] [position held] ?s . ?s [position held] ?obj . ?s [start time] ?x filter(contains(YEAR(?x),'1994')) }}
            Translation: Which position did Angela Merkel hold on November 10, 1994?

            Query: select ?ent where {{ ?ent [instance of] [Class IB flammable liquid] . ?ent [lower flammable limit] ?obj }} ORDER BY DESC(?obj)LIMIT 5 
            Translation: Which 5 Class IB flammable liquids have the highest lower flammable limit ? 

            Context information: {descriptions}
            Translate the following sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: '{Q}'"""
    response = model.generate_content(prompt)
    response_text = response.text

    for _ in range(n_shot):
        prompt = f"""You are a AI translator for converting sparql queries into its natural language questions.
            Here are some examples:

            Query: select distinct ?obj where {{ [Delta Air Lines] [house publication] ?obj . ?obj [instance of] [periodical] }}
            Translation: What periodical literature does Delta Air Lines use as a moutpiece?

            Query: ASK WHERE {{ [Judi Dench] [award received] [Tony Award for Best Direction of a Play] . [Judi Dench] [award received] [Praemium Imperiale] }}
            Translation: Did Judi Dench receive the Tony Award for Best Direction of a Play and the Praemium Imperiale

            Query with a blank node:
            Query: SELECT ?obj WHERE {{ [Angela Merkel] [position held] ?s . ?s [position held] ?obj . ?s [start time] ?x filter(contains(YEAR(?x),'1994')) }}
            Translation: Which position did Angela Merkel hold on November 10, 1994?

            Query: select ?ent where {{ ?ent [instance of] [Class IB flammable liquid] . ?ent [lower flammable limit] ?obj }} ORDER BY DESC(?obj)LIMIT 5 
            Translation: Which 5 Class IB flammable liquids have the highest lower flammable limit ? 

            Context information: {descriptions}
            Translate the following sparql query into natural language; formulate the response as a question and respond in one sentence only with the translation itself: '{Q}'
            {response_text}
            Reflect on your answer and improve it if necessary; respond with only the improved question in one sentence only."""
        response = model.generate_content(prompt)
        response_text = response.text
        if threshold == None:
            pass
            break
    return response_text


def gpt_compare_translations(openai_api_key, Q, translations):
    client = OpenAI(api_key=openai_api_key)
    prompt = f"""You are a AI translator for selecting the best natural language translation of a sparql query.
            Select only a single translation from the list of given translations that is the best equivalent of the sparql query. Translations: {translations}, Query: {Q}. Respond with the selected translation only."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].text.strip()

def gemini_compare_translations(openai_api_key, Q, translations):
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    prompt = f"""You are a AI translator for selecting the best natural language translation of a sparql query.
            Select only a single translation from the list of given translations that is the best equivalent of the sparql query. Translations: {translations}, Query: {Q}. Respond with the selected translation only."""
    response = model.generate_content(prompt)

    return response.text


# Abstraction for translation model
def translate_query_to_nl(model_type_T, Q, descriptions, llama3_api_endpoint=None, openai_api_key=None, n_shot=1, threshold=1):




    if model_type_T == "llama3":
        return llama3_generate_translation(llama3_api_endpoint, Q, descriptions, model=model_type_T, n_shot=n_shot, threshold=threshold)
    elif model_type_T == "gpt":
        return gpt_generate_translation(openai_api_key, Q, descriptions)
    elif model_type_T == "gemini":
        return gemini_generate_translation(openai_api_key, Q, descriptions)
    else:
        raise ValueError(f"Unknown model_type: {model_type_T}")


def compare_query_to_nl(
    model_type_C, Q, translations, llama3_api_endpoint=None, openai_api_key=None
):
    if model_type_C == "llama3":
        return llama3_compare_translations(
            llama3_api_endpoint, Q, translations, model=model_type_C
        )
    elif model_type_C == "gpt":
        return gpt_compare_translations(openai_api_key, Q, translations)
    elif model_type_C == "gemini":
        return gemini_compare_translations(openai_api_key, Q, translations)
    else:
        raise ValueError(f"Unknown model_type: {model_type_C}")
    
    
def get_model_api(model_type, llama3_api_endpoint=None, openai_api_key=None, gemini_api_key=None):
    """Get the appropriate model API configuration."""
    if model_type == "llama3":
        if not llama3_api_endpoint:
            raise ValueError("API endpoint required for llama3")
        return {"llama3_api_endpoint": llama3_api_endpoint}
    elif model_type == "gpt":
        if not openai_api_key:
            raise ValueError("API key required for GPT")
        return {"openai_api_key": openai_api_key}
    elif model_type == "gemini":
        if not gemini_api_key:
            raise ValueError("API key required for Gemini")
        genai.configure(api_key=gemini_api_key)
        return {}
    else:
        raise ValueError(f"Unknown model type: {model_type}")