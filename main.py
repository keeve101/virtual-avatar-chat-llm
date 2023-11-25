from llama import Llama

model_filepath = b"./models/mistral-7b-openorca-finetuned-personality-myself.gguf"

llama = Llama(model_filepath=model_filepath)

system_message = "You are Keith Low, a 3rd-year computer science student. User will ask you a question. Your goal is to answer the question as faithfully as you can."

prompt_template = """
<|im_start|>system
{}<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"""

user_message = ""

while True:
    
    # Get user input
    print("Enter (stop() to stop):")
    user_message = str(input())
    
    if user_message == "stop()":
        break
    
    prompt = prompt_template.format(system_message, user_message)
    
    for token in llama.generate(prompt):
        print((llama.detokenize([token])).decode(), end="", flush=True)