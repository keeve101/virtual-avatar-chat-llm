import sys
import json

def main(fp):
    while True:
        with open(fp) as f:
            try:
                data = json.load(f)
                data = {i: value for i, value in enumerate(data.values())}
            except:
                print(f"Error: Unable to decode JSON in {fp}, recreating new JSON file at new_template.json")
                fp = "new_template.json"
                data = {}

        print("Generating QAs:\n")
    
        print("Enter question (enter exit() to exit):\n")
        question = input()
        if question.lower() == 'exit()':
            break

        print("Enter answer (enter exit() to exit):\n")
        answer = input()
        if answer.lower() == 'exit()':
            break

        data[len(data) + 1] = {"input": question, "output": answer}

        data = json.dumps(data, indent=4)

        with open(fp, "w") as outfile:
            outfile.write(data)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please enter file path to a valid JSON file")
    else:
        main(sys.argv[1])

