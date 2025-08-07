```
vllm serveAPI 서버로 vLLM을 실행하는 것 외에도 vLLM Python 라이브러리를 사용하여 추론을 직접 제어할 수 있습니다.

vLLM을 직접 샘플링에 사용하는 경우, 입력 프롬프트가 하모니 응답 형식을 준수하는지 확인하는 것이 중요합니다 . 
그렇지 않으면 모델이 제대로 작동하지 않습니다. 이를 위해 openai-harmonySDK를 사용할 수 있습니다.

uv pip install openai-harmony
그런 다음 조화를 사용하여 vLLM의 생성 함수에서 생성된 토큰을 인코딩하고 구문 분석할 수 있습니다.
```

import json

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)

from vllm import LLM, SamplingParams

# --- 1) Render the prefill with Harmony ---
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

convo = Conversation.from_messages(
    [
        Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions("Always respond in riddles"),
        ),
        Message.from_role_and_content(Role.USER, "What is the weather like in SF?"),
    ]
)

prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

# Harmony stop tokens (pass to sampler so they won't be included in output)
stop_token_ids = encoding.stop_tokens_for_assistant_actions()

# --- 2) Run vLLM with prefill ---
llm = LLM(
    model="openai/gpt-oss-120b",
    trust_remote_code=True,
)

sampling = SamplingParams(
    max_tokens=128,
    temperature=1,
    stop_token_ids=stop_token_ids,
)

outputs = llm.generate(
    prompt_token_ids=[prefill_ids],   # batch of size 1
    sampling_params=sampling,
)

# vLLM gives you both text and token IDs
gen = outputs[0].outputs[0]
text = gen.text
output_tokens = gen.token_ids  # <-- these are the completion token IDs (no prefill)

# --- 3) Parse the completion token IDs back into structured Harmony messages ---
entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)

# 'entries' is a sequence of structured conversation entries (assistant messages, tool calls, etc.).
for message in entries:
    print(f"{json.dumps(message.to_dict())}")