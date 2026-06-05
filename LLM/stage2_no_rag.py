import sys
import argparse
parser = argparse.ArgumentParser() 
parser.add_argument('--version', type=str, required=True, help='Run version tag, e.g. v1')
args = parser.parse_args()
version = args.version

import json
from pathlib import Path
from openai import OpenAI

# Client setup
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

# Load SA results
results_fid = Path(__file__).parent.parent / 'SA' / 'results'
sa_results = (results_fid / 'sa_results_formatted.txt').read_text()

# Load stage 1 conversation
stage1_path = Path(__file__).parent / f'stage1_no_rag_{version}.json'
if not stage1_path.exists():
    raise FileNotFoundError("stage1_conversation_no_rag.json not found. Run stage1_no_rag.py first.")
stage1 = json.loads(stage1_path.read_text())

# Stage 2 prompt
stage2_prompt = f"""
    Physical Interpretation: The parameters are defined as follows in a 
    WSIMOD land node: 
    - surface_coefficient (the proportion of rainfall that becomes 
    surface runoff) 
    - percolation_coefficient (the proportion of rainfall that infiltrates 
    to subsurface storage). 
    Using these definitions and the sensitivity results from Stage 1, 
    assess whether the parameter ranking is physically consistent with 
    the expected role of each parameter in controlling mean flow. Then 
    state what the interaction term implies about how these two parameters 
    jointly govern flow partitioning. 
    
    Constraints: 
    - Base the physical interpretation exclusively on the parameter 
    definitions provided above and the numerical results from Stage 1. 
    """

# Build conversation history
# Stage 1 question, stage 1 response, then stage 2 question
messages = [
    {"role": "user", "content": stage1['question']},
    {"role": "assistant", "content": stage1['response']},
    {"role": "user", "content": stage2_prompt}
]

# Call LLM for stage 2
response = client.chat.completions.create(
    model="qwen/qwen3-vl-8b",
    messages=messages,
    temperature=0.0
)

# Print response
print("\n--- STAGE 2 RESPONSE (NO RAG) ---\n")
print(response.choices[0].message.content)

# Save conversation
conversation = {
    "stage1": stage1,
    "stage2": {
        "question": stage2_prompt,
        "response": response.choices[0].message.content
    }
}

conversation_path = Path(__file__).parent / f'stage2_no_rag_{version}.json'
conversation_path.write_text(json.dumps(conversation, indent=2))
print(f"\nStage 2 conversation saved to {conversation_path}")