class LLMSession:
    def __init__(self, model_name, system_prompt=""):
        self.model_name = model_name
        self.system_prompt = system_prompt

    def chat(self, user_message):
        pass

def create_llm_session(llm_backend, model_name, system_prompt=""):
    if llm_backend == "ollama":
        print(f"Create Ollama session with model '{model_name}'")
        from .ollama_session import OllamaSession
        return OllamaSession(model_name, system_prompt)
    else:
        raise ValueError(f"Unsupported LLM backend: {llm_backend}")