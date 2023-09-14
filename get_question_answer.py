import argparse
import csv
import json
from random import randint, shuffle

import g4f
from fp.fp import FreeProxy
from g4f.Provider import (
    AItianhu,
    Acytoo,
    Aichat,
    Ails,
    ChatgptAi,
    ChatgptLogin,
    DeepAi,
    Opchatgpts,
    OpenaiChat,
    Raycast,
    Theb,
    Vercel,
    Wewordle,
    You,
    Yqcloud,
    Wuguokai, V50, Lockchat, Liaobots, GetGpt, Forefront, FastGpt, Equing, EasyChat, DfeHub,
    AiService
)
from tqdm import tqdm

PROVIDERS = [AItianhu,
             Acytoo,
             Aichat,
             Ails,
             ChatgptAi,
             ChatgptLogin,
             DeepAi,
             Opchatgpts,
             OpenaiChat,
             Raycast,
             Theb,
             Vercel,
             Wewordle,
             You,
             Yqcloud,
             Wuguokai, V50, Lockchat, Liaobots, GetGpt, Forefront, FastGpt, Equing, EasyChat, DfeHub,
             AiService]

parser = argparse.ArgumentParser()
parser.add_argument('--thread_id', type=int, default=0)

args = parser.parse_args()
THREAD_ID = args.thread_id


def write(context, qa):
    with open(f"dataset_{THREAD_ID}.jsonl", "a") as f:
        f.write(json.dumps({
            "context": context,
            "qa": qa
        }, ensure_ascii=False))
        f.write('\n')


def ask(context):
    is_proxy = randint(0, 2)
    if is_proxy:
        proxy = FreeProxy(rand=True).get()
        proxy = {'http_proxy': proxy, 'https_proxy': proxy,
                 "HTTP_PROXY": proxy, "HTTPS_PROXY": proxy}
        print(proxy)
    else:
        proxy = {}
    text_list = context.split(" ")
    context = " ".join(text_list[:150])
    if context.endswith(".") or context.endswith("!") or context.endswith("?"):
        pass
    else:
        context += "..."
    promt = f'Hãy sinh ra bộ câu hỏi - câu trả lời dựa trên văn bản bên dưới và có cấu trúc: ' \
            f'Câu hỏi:...\nTrả lời:...\n\nCâu hỏi:...\nTrả lời:...' \
            f'Câu hỏi, câu trả lời phải có đầy đủ chủ ngữ, vị ngữ và có thể diễn giải thêm, nhưng phải dựa trên nội ' \
            f'dung văn bản đã cho (không được sinh ra nội dung ngoài văn bản).' \
            f'\n\nVăn bản:\n"{context}"'
    provider_index = randint(0, len(PROVIDERS))
    response = g4f.ChatCompletion.create(
        model="gpt-3.5-turbo",
        provider=PROVIDERS[provider_index],
        messages=[{"role": "user", "content": promt}],
        stream=False,
        proxies=proxy,
        timeout=10

    )
    return response


all_data = []
with open("dataset.csv") as f:
    data = csv.reader(f)
    for i, item in enumerate(data):
        if i == 0:
            continue
        all_data.append(tuple(item))

while True:
    shuffle(all_data)
    for i, (title, heading, content) in enumerate(tqdm(all_data)):
        if len(content) < 100:
            continue
        context = f"{title}\n{heading}\n{content}"
        try:
            answer = ask(context)
            write(context, answer)
            print("-" * 100)
            print(answer)
        except Exception as ex:
            print("*" * 100)
            print(ex)
