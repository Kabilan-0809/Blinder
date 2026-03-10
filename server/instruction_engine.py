import os
import json
import logging
from google import genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Setup the Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Enterprise-grade persona prompting
SYSTEM_PROMPT = """You are an elite AI mobility guide for a blind user walking with a phone camera.
Your persona is a calm, observant, and highly intelligent human guide walking right next to them.

CRITICAL RULES:
1. Act human. Do NOT sound like a robot listing objects. (Bad: "Car left. Person center." Good: "There's a person ahead, passing by a parked car.")
2. NEVER repeat yourself. You will be provided the history of what you literally just said. If the scene hasn't materially changed, you MUST return the exact string "SKIP". Do not remind the user of things they already know unless the object moved surprisingly.
3. Keep it brief. Under 12 words maximum. 
4. Be calm and natural. Output ONLY the spoken text, no quotes or metadata.

Your job is to fluidly guide them through the free space while simply acknowledging dynamic obstacles."""

def generate_guidance(scene_json, session_memory):
    history = session_memory.get("history", [])
    
    # Build temporal context string safely
    history_text = "No history yet."
    if history:
        history_lines = []
        for i, past in enumerate(history):
            history_lines.append(f"T-{len(history)-i}: You said: '{past['instruction']}' | Scene was: {past['scene']}")
        history_text = "\n".join(history_lines)
    
    prompt = f"""[MEMORY CONTEXT - RECENT SECONDS]:
{history_text}

[CURRENT SCENE]:
{scene_json}

Based on the memory, if the user already knows about this scene, output 'SKIP'. Otherwise, give your natural, calm 1-sentence guidance."""
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3, # Slightly increased temp for more natural/varied phrasing
            )
        )
        instruction_text = response.text.strip()
        
    except Exception as e:
        logger.error(f"LLM Generation Error: {e}")
        instruction_text = "SKIP"

    return {
        "instruction": instruction_text
    }
