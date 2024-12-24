import os

from openai import OpenAI

test_key = 'sk-leu33xVZssUE27b4I2Suwt5J5TT3nxr3WQKcxb5eKLBmeD9m'

# 设置代理
os.environ["http_proxy"] = "socks5://localhost:10808"
os.environ["https_proxy"] = "socks5://localhost:10808"

messages = [
    {"role": "user", "content": (
        "I want you to act as a scientific Chinese-English translator, "
        "I will provide you with some paragraphs in one language and your"
        " task is to accurately and academically translate the paragraphs "
        "only into the other language. Do not repeat the original provided "
        "paragraphs after translation. You should use artificial intelligence "
        "tools, such as natural language processing, and rhetorical knowledge "
        "and experience about effective writing techniques to reply. "
        "I'll give you my paragraphs as follows:在多细胞生物中，正常生长需要控制细胞"
        "分裂以产生与其父母相似或不同的细胞。对植物根系这—过程的分析揭示了这种机制是如何调节的。"
    )}
]

client = OpenAI(
    api_key=test_key,  # This is the default and can be omitted
)

# 调用 OpenAI API
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": messages,
        }
    ],
    model="gpt-3.5-turbo",
)

print(chat_completion.choices[0].message[0].content)