import ollama


class OllamaResponder:
    @staticmethod
    def query_response(
        query,
        context,
        model="llama3.2",
        system_prompt=None,
        temperature=0.2,
        keep_alive=None,
        client=None,
        host=None,
    ):
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        if context is None:
            context = ""

        if system_prompt is None:
            system_prompt = (
                "Based on the following lecture transcript and slide segments, answer the question to the best of your abilities."
                "Utilize ONLY the below context as your reference for generating the answer. It is fine if the answer is not DIRECTLY stated"
                "but if you can infer the answer from the text return that answer. " \
                "Please be as descriptive as possible, format the answer in a human-readable way utilizing bullet points and numbered list where you see fit." \
                "Do not include any unnecessary detail that is not explicitly asked for by the user"
            )

        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        options = {"temperature": temperature} if temperature is not None else None

        if client is None:
            if host:
                client = ollama.Client(host=host)
            else:
                client = ollama

        response = client.chat(
            model=model,
            messages=messages,
            options=options,
            keep_alive=keep_alive,
        )

        message = getattr(response, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if content is not None:
                return str(content).strip()

        if isinstance(response, dict):
            message = response.get("message") or {}
            content = message.get("content")
            if content is not None:
                return str(content).strip()
            response_text = response.get("response")
            if response_text is not None:
                return str(response_text).strip()

        return str(response).strip()
