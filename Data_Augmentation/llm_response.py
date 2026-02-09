import time
from openai import OpenAI

def send_request(msgs, model_name,config):
    opt ={
            "stream":False
        }
    opt.update(
        (k, config[k]) for k in config.keys()
        )
    response = None
    if model_name == "llama3-70b-it":
        api_key='Your_API_KEY'
        base_url=f"""YOUR_BASE_URL"""

    local_client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    while response is None:
        try:
            completion = local_client.chat.completions.create(
                model=model_name,
                messages=msgs,
                **opt
            )
            response = completion.choices[0].message.content
        except KeyError as ke:
            print("KeyError occurred:", ke)
            print("Retrying...")
            time.sleep(5)
        except Exception as e:
            print("An unexpected error occurred:", e)
            print("Retrying...")
            time.sleep(5)
    return response
