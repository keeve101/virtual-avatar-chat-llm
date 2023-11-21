import sys
import json

def main(fp):
    while True:
        with open(fp) as f:
            data = json.load(f)
        data = {i: value for i, value in enumerate(data.values())}

        print("Generating QAs:\n")
    
        print("Input:\n")
        question = input()

        print("Output:\n")
        answer = input()

        data[len(data) + 1] = {"input": question, "output": answer}

        data = json.dumps(data, indent=4)

        with open(fp, "w") as outfile:
            outfile.write(data)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("fp required")
    else:
        main(sys.argv[1])

