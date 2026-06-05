import sys
import argparse 
parser = argparse.ArgumentParser() 
parser.add_argument('--version', type=str, required=True, help='Run version tag, e.g. v1')
args = parser.parse_args()
version = args.version

import json
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent / 'SA'))
from experimentor import SAMPLE_N

#N = SAMPLE_N
N=300 

# Client setup
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"  
)

# Load SA results
results_fid = Path(__file__).parent.parent / 'SA' / 'results'
sa_results = (results_fid / 'sa_results_formatted.txt').read_text()

# Stage 1 prompt
prompt = f"""

    You are analysing Sobol sensitivity analysis results for a hydrology model 
    with {sa_results}
    
    Respond with exactly these four sections, in this order:
    Parameter Ranking: Rank the parameters by ST only. Cite the values. State that 
    ranking is uncertain if ST confidence intervals overlap.
    Interaction Contribution: For each parameter, compute ST − S1. Cite both input 
    values and the result. Interpret which parameter's output variance is more 
    interaction-driven. 
    Interaction Assessment: Assess the pairwise interaction between parameters using 
    S2. Cite the value.
    Reliability: Assess reliability for ST, S1, and S2 individually using the 
    criterion CI / index > 0.5. CI / index > 0.5 is unreliable. Do not assess reliability 
    for ST − S1. Based on this assessment, comment  whether N={N} appears sufficient for 
    convergence, noting that this criterion serves as a proxy for convergence rather than 
    a formal convergence test.
    
    Constraints:
    - Do not write anything outside these four sections. 
    - Each section must be 2-3 sentences. Do not write more or less.
    - Report all numerical values with the format: index = value ± confidence interval. 
    Do not deviate from this format.
    """


# Call LLM
response = client.chat.completions.create(
    model="qwen/qwen3-vl-8b",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.0
)

# Print response
print("\n--- STAGE 1 RESPONSE (NO RAG) ---\n")
print(response.choices[0].message.content)

# Save conversation
conversation = {
    "question": prompt,
    "response": response.choices[0].message.content
}

conversation_path = Path(__file__).parent / f'stage1_no_rag_{version}.json'
conversation_path.write_text(json.dumps(conversation, indent=2))
print(f"\nStage 1 conversation saved to {conversation_path}")