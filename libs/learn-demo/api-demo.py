from openai import OpenAI
client = OpenAI(
    api_key='sk-6d1dfee8241f450aa477a49f37159755',
    base_url="https://api.deepseek.com/v1")

clientMT = OpenAI(
    base_url='https://ms-ens-abc10ebd-5837.api-inference.modelscope.cn/v1',
    api_key='ms-f2a19712-d239-4daf-b126-348d10866191', # ModelScope Token
)
if __name__ == "__main__":

    # response = client.chat.completions.create(
    #     model="deepseek-chat",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant"},
    #         {"role": "user", "content": "Hello"},
    #         ],
    #     stream=False
    # )

    # print(response.choices[0].message.content)


    responseMT = clientMT.chat.completions.create(
        model='Qwen/Qwen3-0.6B', # ModelScope Model-Id
        messages=[
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                'content': '你好'
            }
        ],
        stream=True
    )

    for chunk in responseMT:
        print(chunk.choices[0].delta.content, end='', flush=True)