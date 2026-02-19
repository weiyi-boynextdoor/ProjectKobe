# pip install zai-sdk

import os
import argparse
from zai import ZhipuAiClient
from dotenv import load_dotenv

load_dotenv()

client = ZhipuAiClient(api_key=os.getenv("ZHIPU_API_KEY"))

SYSTEM_PROMPT = "请你扮演科比布莱恩特和进行日常对话，简短一些，每句话控制在100字以内"

use_history = False

print("孩子们，我回来了，我想死你们了！")

chat_history = []

def chat_main_loop():
    user_input = input("You: ")

    while user_input.lower() != "exit":
        user_content = {"role": "user", "content": user_input}
        chat_history.append(user_content)

        if use_history:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_history
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                user_content,
            ]

        response = client.chat.completions.create(
            model="charglm-4",
            messages=messages,
            thinking={
                "type": "disabled",
            },
            stream=True,
            max_tokens=200,
            temperature=1.0
        )

        print("Kobe: ", end='', flush=True)
        for chunk in response:
            if chunk.choices[0].delta.reasoning_content:
                print(chunk.choices[0].delta.reasoning_content, end='', flush=True)

            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end='', flush=True)
            assistant_content = {"role": "assistant", "content": chunk.choices[0].delta.content}
            chat_history.append(assistant_content)
        print()

        user_input = input("You: ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-history', action='store_true', help='Use chat history')
    args = parser.parse_args()
    use_history = args.use_history
    chat_main_loop()
