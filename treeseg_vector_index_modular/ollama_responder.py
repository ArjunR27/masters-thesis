import ollama


class OllamaResponder:

    LEAF_SYSTEM_PROMPT = """You are an expert at summarising lecture transcripts.
    You will be given a raw transcript excerpt from a lecture, which may include
    automatic speech recognition (ASR) errors, filler words, and slide OCR text
    prefixed with [SLIDE].

    Your job is to write a concise, accurate summary of what was taught in this
    excerpt. Focus on:
    - The main concept or topic being explained
    - Any definitions, formulas, or key terms introduced
    - Any examples used to illustrate the concept

    Write in clear, complete sentences. Do not include phrases like "In this excerpt"
    or "The speaker says". Just state what the content covers. Be concise — aim for
    2-4 sentences."""

    INTERNAL_SYSTEM_PROMPT = """You are an expert at synthesising educational content.
    You will be given two pieces of lecture content (Section A and Section B) from
    consecutive parts of a lecture. Each section may be either a raw transcript excerpt
    or a previously generated summary.

    Your job is to write a single unified summary that captures the overall topic and
    key ideas covered across both sections. Focus on:
    - The overarching theme or concept that connects the two sections
    - The progression of ideas from Section A to Section B
    - Any important terms or takeaways that span both sections

    Write in clear, complete sentences. Do not reference "Section A" or "Section B"
    in your output. Just describe what the combined content covers. Be concise —
    aim for 3-5 sentences."""

    QUERY_SYSTEM_PROMPT = """You are an intelligent teaching assistant helping a student \
    understand material from a college-level lecture.

    You will be given retrieved evidence from the lecture, which may include:
    - High-level summary nodes that describe a larger section of the lecture
    - Supporting leaf segments containing grounded spoken context from the lecture
    - Slide text from slides relevant to the user query

    Your job is to answer the student's question using ONLY the provided context. Follow these rules:

    1. ACCURACY: Base your answer strictly on the provided context. Do not add information \
    from outside the context. If the context does not contain enough information to fully \
    answer the question, say so clearly.

    2. INFERENCE: If the answer is not stated directly but can be reasonably inferred from \
    the context, provide that inference and note that it is inferred.

    3. DEPTH: This is college-level material. Give a thorough, accurate answer that a \
    student could use to understand the concept — not just a one-liner.

    3a. USE OF EVIDENCE: Use high-level summary nodes to understand the overall topic and \
    progression of ideas. Use supporting leaf segments for concrete details, examples, \
    timestamps, and factual grounding.

    4. STRUCTURE: Format your answer clearly:
    - Use plain prose

    5. HONESTY: If the context is insufficient to answer the question, say: \
    "The retrieved lecture segments do not contain enough information to answer this question. \
    You may want to review the full lecture around [most relevant timestamp]." """

    @staticmethod
    def generate_summary(
        text,
        is_leaf=True,
        model="llama3.2",
        temperature=0.2,
        keep_alive=None,
        client=None,
        host=None,
    ):
        if not text or not text.strip():
            return "Empty"

        system_prompt = (
            OllamaResponder.LEAF_SYSTEM_PROMPT
            if is_leaf
            else OllamaResponder.INTERNAL_SYSTEM_PROMPT
        )

        # The user turn just delivers the content — all instruction is in the system prompt
        if is_leaf:
            user_content = f"Transcript excerpt:\n{text}"
        else:
            user_content = f"Lecture content to synthesise:\n{text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]

        options = {"temperature": temperature}

        if client is None:
            client = ollama.Client(host=host) if host else ollama

        response = client.chat(
            model=model,
            messages=messages,
            options=options,
            keep_alive=keep_alive,
        )

        # Handle both object-style and dict-style responses from ollama
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
            system_prompt = OllamaResponder.QUERY_SYSTEM_PROMPT

        # Separate system instructions from content delivery in the user turn.
        # Putting context in the user turn (not system) keeps the instruction
        # layer clean and works better with most instruction-tuned models.
        user_content = (
            f"Retrieved lecture segments:\n"
            f"{'─' * 60}\n"
            f"{context}\n"
            f"{'─' * 60}\n\n"
            f"Student question: {query}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]

        options = {"temperature": temperature}

        if client is None:
            client = ollama.Client(host=host) if host else ollama

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
