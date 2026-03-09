import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Setup the Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

SYSTEM_PROMPT = "You are an AI mobility guide for blind users. Convert scenes into short navigation instructions. Mention obstacles and directions. Keep sentences under 10 words. Respond with ONLY the instruction string, no markdown, no quotes."

def generate_guidance(scene_json, session_memory):
    
    # Check for repetitive scene states to save latency and avoid annoying the user
    # If the exact identical scene was seen previously, return the exact same instruction
    # so the frontend knows to ignore it (via deduplication)
    if session_memory.get("last_scene_summary") == scene_json:
        return {"instruction": session_memory.get("last_instruction")}
    
    prompt = f"Previous Instruction Given: {session_memory.get('last_instruction', 'None')}\nCurrent Scene Representation:\n{scene_json}"
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2, # Low temp for deterministic, calm output
            )
        )
        
        instruction_text = response.text.strip()
        
    except Exception as e:
        print(f"LLM Error: {e}")
        # Fallback if api key is missing or rate limited
        instruction_text = "Path clear. Walk forward."

    return {
        "instruction": instruction_text
    }
