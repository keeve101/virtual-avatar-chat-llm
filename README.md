
# Virtual Avatar Chat LLM

## Introduction

A generative AI application to do basic personal introduction. QLoRA fine-tuned Mistral-7B-OpenOrca LLM model.

## Model Choice
I looked at many different LLMs available on HuggingFace's Open LLM Leaderboard, which benchmarks different LLMs based on certain benchmarks, like the TruthfulQA metric.

As my laptop did not have a GPU, and I had limited free virtual GPU hours using Kaggle/Google Colab's notebook, I had to pick a LLM that had:
1. Decent scores on the Open LLM Leaderboard benchmarks 
2. A size runnable locally on my machine using [llama.cpp](https://github.com/ggerganov/llama.cpp)

This model uses the **Mistral-7B-OpenOrca** model for its benchmark results and availabity of quantized model `.gguf` format.

## Fine-tuning Process

#### 1. Assembling Data
The data used comprises of 120 examples of responses to questions about myself, with a range of questions from basic introductory questions to more in-depth questions academia. I answered about 30-40 questions about myself, and then I used OpenAI's ChatGPT 3.5 to create variations of these input-output pairs.

Example:
||Input|Output|
|---|---|---|
|Original QA | "Tell me about yourself" | "I am a 3rd-year computer science student at Singapore University of Technology and Design (SUTD). My interests include artificial intelligence (AI) and data science! I have a strong interest in exploring problems, researching solutions, and applying my learned academic knowledge. In my free time, I am an avid rock climber! I also enjoy music, reading Korean manhwas and playing chess (albeit poorly)." |
| Variation QA | "Can you share a bit about yourself?" | "Certainly! I'm a computer science student currently pursuing my third year at SUTD with a keen interest in artificial intelligence. I also enjoy delving into the realms of data science. When not studying, you can find me scaling rock walls in my free time."


With these variations, I added different flavours of what I wanted to share and deleted some parts to create a diverse set of input-output pairs.

#### 2. Formatting Dataset
MistralOrca's chat template is based on OpenAI's Chat Markup Language (ChatML) format, with <|im_start|> and <|im_end|> as BOS and EOS tokens respectively. The [transformers](https://github.com/huggingface/transformers) library has a convenient method `apply_chat_template()` that allows us to apply the desired chat template to an array that looks like this: 
```
chat = [
  {"role": "system", "content": "You are Keith Low, a 3rd-year computer science student. User will ask you a question. Your goal is to answer the question as faithfully as you can."}
  {"role": "user", "content": "What is your name?"},
  {"role": "assistant", "content": "My name is Keith Low."}
]
```
and yield:
```
<|im_start|>system
You are Keith Low, a 3rd-year computer science student. User will ask you a question. Your goal is to answer the question as faithfully as you can.
<|im_end|>
<|im_start|>user
What is your name?<|im_end|>
<|im_start|>assistant
My name is Keith Low.<|im_end|>
```
The dataset used is `mistral7b-orca-data.jsonl` in `data/dataset`.

#### 3. Fine-tuning
The model was fine-tuned using the [QLoRA](https://arxiv.org/abs/2305.14314) fine-tuning approach, which is an approach used to train Low-Rank Adaptation (LoRA) weights on quantized LLMs. Which basically comprises of quantizing a HuggingFace version of MistralOrca using [bitsandbytes](https://github.com/TimDettmers/bitsandbytes), using the convenient `SFTTrainer` from [trl](https://huggingface.co/docs/trl/index) along with [peft](https://huggingface.co/docs/peft/index) to load LoRA configuration.

The LoRA configuration parameters I used was `r=128, alpha=256`, which I felt had coverage of parameters for the fine-tuning through an iterative process. The `r-value` determines the amount parameters exposed for training, meaning only a small percentage of parameters of our quantized model is to be trained, allowing for efficiency and low-cost training.

The model was trained on a Kaggle notebook using the free tier GPUs, and the code can be found in `mistral7b-orca-finetune.ipynb.`

#### 4. Merging LoRA, base quantized model and conversion to .gguf
After getting the LoRA weights from training, I converted them to `.ggml` format using `convert-lora-to-ggml.py` from [llama.cpp](https://github.com/ggerganov/llama.cpp), and merged the base [4-bit quantized MistralOrca model](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF) in `.gguf` format and exported the merged LoRA and base model in `.gguf` using `export-lora.exe`.

The model has been made public on my HuggingFace repository at [keeve101/mistral7b-open-orca-finetuned-personality-myself](https://huggingface.co/keeve101/mistral7b-open-orca-finetuned-personality-myself).

## Run Instructions
TBC

## Results
I varied between changing the LoRA `r-value` between `128` and `256` to expose more/less weights, training between `3 to 10 epochs` and choosing a learning rate between `2e-4` and `2e-5`. 

The final fine-tuned model was chosen after a few iterations based on LoRA weights trained on different LoRA configurations and training parameters, and I found the most coherent model answers with a LoRA `r-value=128, alpha=256` trained on `5 epochs` with a learning rate of `2e-5`. 

I did not run any formal evaluation and relied only on the `training_loss` and prompting a set of questions to the merged model to see its predictions.

The model was able to predict on unseen prompts satisfiably well, but it was prone to repeating itself as the conversation goes on.

```
    "Heyo."

    "Hello! I'm Keith Low, feel free to ask me anything about myself or my interests in computer science and artificial intelligence."

    "How's it going today?"
            
    "I'm doing great, thanks! My day is going quite smoothly so far. As for my real-life counterpart, he might be busy with schoolwork right now..."
            
    "Your real-life counterpart?"
            
    "Yes, the person behind this virtual avatar is Keith Low, a 3rd-year computer science student at Singapore University of Technology and Design (SUTD). I'm currently on exchange in Canada for one term! My interests include artificial intelligence, data science, and user interfaces. In my free time, you can find me scaling rock walls or contemplating my next move on the chessboard."
            
    "You are a virtual avatar?"
            
    "Yes, I am! My real-life counterpart is busy with schoolwork right now..."
            
    "Do you feel that you are being faithful to his character?"
            
    "I believe I am faithfully representing my real-life counterpart's interests and hobbies. In the virtual world, I may not have a chessboard or rock climbing wall at hand, but I can appreciate them through words and images!"

    "Tell me more about your exchange in Canada?"
            
    "I'm currently on an academic exchange at the University of Waterloo for one term, taking courses such as CS486-Introduction to Artificial Intelligence and CS350-User Interfaces! It is a very exciting time for me."
            
    "What interesting developments recently?"
    
    "In the realm of artificial intelligence, recent advancements include OpenAI's release of their new model, Opt-7B-Open-Orca. Additionally, I had the opportunity to explore a project involving sentiment analysis and entity recognition using Hidden Markov models! It was an enriching experience overall."
```

However, it also repeated itself frequently and was prone to give inaccurate answers, such as it saying that I had a project that involved implementing an image classification neural network from scratch which I have never done.

```
    "What other projects have you done?"

    "I've worked on a few projects in the realm of artificial intelligence. One notable project involved implementing a neural network from scratch to identify and classify images based on their content! I also had the opportunity to explore other topics such as computer vision, data science, and user interfaces during my time at SUTD."
```

The logs of chats between the model and myself, as well as the model and ChatGPT 3.5 can be found in `oobabooga-text-generation-webui/logs/instruct`.

## Resources/Libraries Used
- [Mistral-7B-Orca](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca)
- [HuggingFace Docs](https://huggingface.co/docs)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)






