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

# Load stage 2 conversation (contains stage 1 and stage 2)
stage2_path = Path(__file__).parent / f'stage2_no_rag_{version}.json'
if not stage2_path.exists():
    raise FileNotFoundError("stage2_conversation_no_rag.json not found. Run stage2_no_rag.py first.")
prev = json.loads(stage2_path.read_text())

# Stage 3 prompt
stage3_prompt = """

    Literature Contextualisation: Assess whether the parameter ranking and interaction 
    pattern identified in Stage 1 are consistent with findings reported in the GSA 
    literature for comparable water system models. Then evaluate whether the result 
    supports the consistency criterion: that the parameters governing surface runoff 
    and subsurface percolation exert physically expected levels of control on mean flow.

    Constraints:
    - Do not fabricate citations. Only cite sources you can identify with confidence, 
    including author and finding.
    - If citing a retrieved document, refer to it explicitly by author and year.
    - If no source is available to support a claim, state that the assessment is based 
    on general GSA practice.
    """

# Build full conversation history from all previous stages
messages = [
    {"role": "user",      "content": prev['stage1']['question']},
    {"role": "assistant", "content": prev['stage1']['response']},
    {"role": "user",      "content": prev['stage2']['question']},
    {"role": "assistant", "content": prev['stage2']['response']},
    {"role": "user",      "content": stage3_prompt}
]

response = client.chat.completions.create(
    model="qwen/qwen3-vl-8b",
    messages=messages,
    temperature=0.0
)

print("\n--- STAGE 3 RESPONSE (NO RAG) ---\n")
print(response.choices[0].message.content)

# Save full conversation including all previous stages
conversation = {
    "stage1": prev['stage1'],
    "stage2": prev['stage2'],
    "stage3": {
        "question": stage3_prompt,
        "response": response.choices[0].message.content
    }
}

conversation_path = Path(__file__).parent / f'stage3_no_rag_{version}.json'
conversation_path.write_text(json.dumps(conversation, indent=2))
print(f"\nStage 3 conversation saved to {conversation_path}")