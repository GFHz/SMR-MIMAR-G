"""One-title-only prompt without bridge direction, score, or filtering."""
import json

SYSTEM_PROMPT = """You are selecting exactly one next movie for the current planning step.

You MUST choose exactly one movie title from the supplied catalog-grounded candidate list.

Output ONLY the exact movie title.
Do not output explanations.
Do not output numbering.
Do not output bullets.
Do not output a list.
Do not output multiple movies.
Do not output genres.
Do not output reasoning.
Do not output any text before or after the movie title.

The selected movie must not be in the user's current history.
The selected movie must not be the target movie unless the target is explicitly included in the allowed candidate list."""


def build_prompt(user, window, top5, target, candidates, step):
    demographics = user["demographics"]
    text = (f"Gender:{demographics['gender']}\nAge:{demographics['age']}\n"
            f"Occupation:{demographics['occupation']}\n\nCurrent Last-20 history:\n")
    text += "".join(f"{x['title']} Genre:{'|'.join(x['genres'])}\n" for x in window)
    text += "\nCurrent Top-5 interests with UserFreq:\n"
    text += "".join(f"{x['genre']}: {x['frequency']}\n" for x in top5)
    text += f"\nTarget movie: {target['title']} Genre:{'|'.join(target['genres'])}\n"
    text += (f"\nCurrent planning step: {step}\nAllowed candidates:\n"
             + json.dumps([x["title"] for x in candidates], ensure_ascii=False)
             + "\n\nSelect exactly one movie from the allowed candidates.\n\nReturn only the exact title.\n")
    return {"system_prompt": SYSTEM_PROMPT, "user_prompt": text}

