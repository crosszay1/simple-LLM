from datasets import load_dataset

ds = load_dataset("tau/commonsense_qa")


print(ds)

def user_input():
    question = input("Enter your prompt: ")
    return question

def construct_prompt(question):
    prompt = f"""
    Question: {question}\n
    Answer:
    """
    return prompt



if __name__ == "__main__":
    question = user_input()
    prompt = construct_prompt(question)
    print(prompt)


