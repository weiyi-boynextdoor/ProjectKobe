import ollama
import sys

def ollama_chat(model_name):
    print(f"--- 正在连接本地 Ollama 服务 (模型: {model_name}) ---")
    
    # 初始化对话历史
    messages = []
    
    try:
        # 1. 检查模型是否已加载/存在
        ollama.show(model_name)
    except Exception as e:
        print(f"❌ 错误: 找不到模型 '{model_name}'。请先执行 'ollama pull {model_name}'")
        return

    print("✅ 连接成功！输入 'exit' 或 'quit' 退出，输入 'clear' 清空对话。")
    print("-" * 50)

    while True:
        user_input = input("\n👤 你: ").strip()
        
        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit']:
            break
        if user_input.lower() == 'clear':
            messages = []
            print("🧹 对话历史已清空。")
            continue

        # 将用户输入加入历史
        messages.append({'role': 'user', 'content': user_input})

        print(f"🤖 {model_name}: ", end="", flush=True)

        try:
            # 2. 发起流式请求 (Streaming)
            stream = ollama.chat(
                model=model_name,
                messages=messages,
                stream=True,
                options={
                    "num_ctx": 8192,  # 设置上下文长度，防止爆显存
                    "temperature": 0.7 # 随机性设置
                }
            )

            full_response = ""
            for chunk in stream:
                content = chunk['message']['content']
                print(content, end="", flush=True)
                full_response += content

            # 将助手的回复存入历史，实现多轮对话
            messages.append({'role': 'assistant', 'content': full_response})
            print() # 换行

        except Exception as e:
            print(f"\n⚠️ 发生错误: {str(e)}")

if __name__ == "__main__":
    # 如果你的模型名字不一样，修改这里即可
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt-oss:20b"
    ollama_chat(model_name)