import json

"""
This processes our json file with input-output pairs

{ 
 "1": {
        "input": "..",
        "output: "...",
      },
}

as a jsonl file 

{
    "system": ....,
    "question": ....,
    "answer": ....
}
{
....
}

"""

with open('data2.json') as f:
    data = json.load(f)


# always reset idx
data = {i: value for i, value in enumerate(data.values())}
    
system_message = "You are Keith Low, a 3rd-year computer science student. User will ask you a question. Your goal is to answer the question as faithfully as you can."

new_data = {}

for i in range(len(data)):
    new_data[i] = {}
    new_data[i]['system'] = system_message
    new_data[i]['question'] = data[i]['input']
    new_data[i]['answer'] = data[i]['output']


with open("mistral7b-orca-data_v2.jsonl", "w") as outfile:
    for line in new_data.values():
        line = json.dumps(line, indent=4)
        outfile.write(line)