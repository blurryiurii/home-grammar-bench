import json
import time
import urllib.request


class LlamaClient:
    def __init__(self, url, model):
        self.url = url
        self.model = model

    def post(self, path, body):
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps({"model": self.model, **body}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=1800) as response:
            return json.load(response)

    def render(self, system, user):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.post("/apply-template", {"messages": messages})["prompt"]

    def complete(self, prompt, constraint, cache_prompt=True):
        request = {
            "prompt": prompt,
            "temperature": 0,
            "seed": 42,
            "n_predict": 128,
            "cache_prompt": cache_prompt,
            **constraint,
        }
        start = time.perf_counter()
        result = self.post("/completion", request)
        return {**result, "latency_ms": round((time.perf_counter() - start) * 1000)}
