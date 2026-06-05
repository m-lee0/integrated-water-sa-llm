import sys
import argparse
parser = argparse.ArgumentParser() 
parser.add_argument('--version', type=str, required=True, help='Run version tag, e.g. v1')
args = parser.parse_args()
version = args.version

import json
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

results_fid = Path(__file__).parent.parent / 'SA' / 'results'
sa_results = (results_fid / 'sa_results_formatted.txt').read_text()


# Load stage 3 conversation (contains stage 1,2 and 3)
stage3_path = Path(__file__).parent / f'stage3_no_rag_{version}.json'
if not stage3_path.exists():
    raise FileNotFoundError("stage3_conversation_no_rag.json not found. Run stage3_no_rag.py first.")
prev = json.loads(stage3_path.read_text())

# Stage 4 prompt
stage4_prompt = """
    Action Recommendations: 

    Based on the Sobol sensitivity analysis results interpreted above, 
    identify 3 to 5 concrete actions that should be taken to improve 
    confidence in the model results or address identified limitations.

    For each action, state:

    - What the action is
    - Which specific numerical result justifies it (cite the exact 
    index value and confidence interval)
    - What methodological or literature-based reasoning supports it, 
    if available
    - What outcome is expected

    Constraints:

    - Every action must be traceable to a specific numerical result 
    from the SA output. Do not recommend actions that are not directly 
    supported by the values provided.
    - Do not fabricate citations. Only cite sources you can identify with confidence, 
    including author and finding.
    - For every piece of methodological reasoning, you must state one of:
      (a) "Source: [author, year] — [specific claim retrieved]", or
      (b) "Source: general SA practice — no literature citation available."
      No other citation format is permitted.
    - Do not use thresholds or benchmarks from case-specific studies unless 
    you can identify the exact source. If no literature source is available, 
    state that the justification is based on general SA practice.
    - Do not recommend actions related to parameter leverage or model 
    complexity unless directly supported by the ST values provided.
    """

# Build full conversation history from all previous stages
messages = [
    {"role": "user",      "content": prev['stage1']['question']},
    {"role": "assistant", "content": prev['stage1']['response']},
    {"role": "user",      "content": prev['stage2']['question']},
    {"role": "assistant", "content": prev['stage2']['response']},
    {"role": "user",      "content": prev['stage3']['question']},
    {"role": "assistant", "content": prev['stage3']['response']},
    {"role": "user",      "content": stage4_prompt}
]

response = client.chat.completions.create(
    model="qwen/qwen3-vl-8b",
    messages=messages,
    temperature=0.0
)

print("\n--- STAGE 4 RESPONSE (NO RAG) ---\n")
print(response.choices[0].message.content)

# Save full conversation including all previous stages
conversation = {
    "stage1": prev['stage1'],
    "stage2": prev['stage2'],
    "stage3": prev['stage3'],
    "stage4": {
        "question": stage4_prompt,
        "response": response.choices[0].message.content
    }
}

conversation_path = Path(__file__).parent / f'stage4_no_rag_{version}.json'
conversation_path.write_text(json.dumps(conversation, indent=2))
print(f"\nStage 4 conversation saved to {conversation_path}")