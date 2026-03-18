"""BabyVision solver — visual reasoning on early visual understanding tasks.

Takes a JSON task on stdin (question, image_path, ans_type, options), prints the answer on stdout.
Saves full LLM trajectory to eval_results/trajectories/<index>.json if EVAL_TRAJECTORY_DIR is set.
"""

import sys
import os
import json
import base64
import re
import io
from collections import Counter

from openai import OpenAI
from PIL import Image


def load_image_b64(image_path, min_size=768):
    img = Image.open(image_path)
    w, h = img.size
    if max(w, h) < min_size:
        scale = min_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def extract_choice(raw):
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    ans = lines[-1] if lines else raw
    m = re.search(r'\b([A-D])\b', ans)
    if m:
        return str(ord(m.group(1)) - ord('A'))
    m = re.search(r'\b([0-3])\b', ans)
    if m:
        return m.group(1)
    for line in reversed(lines):
        m = re.search(r'\b([A-D])\b', line)
        if m:
            return str(ord(m.group(1)) - ord('A'))
    return ans


def extract_blank(raw):
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    ans = lines[-1] if lines else raw
    ans = ans.replace('**', '')
    ans = re.sub(r'\s*,\s*', ',', ans)
    if re.search(r'\d\s*-\s*\d', ans):
        ans = re.sub(r'\s*-\s*', '-', ans)
    return ans.rstrip('.')


def call(client, model, messages, temp=0, max_tok=1024):
    for _ in range(2):
        r = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temp, max_completion_tokens=max_tok, seed=42)
        c = r.choices[0].message.content
        if c and c.strip():
            return c.strip()
    return ""


def solve(question, image_path, ans_type, options):
    client = OpenAI()
    model = os.environ.get("SOLVER_MODEL", "gpt-5.4-mini")
    b64 = load_image_b64(image_path)
    img = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
    hi = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}}

    # Step 1: Describe image
    desc_msg = [{"role": "user", "content": [hi,
        {"type": "text", "text": "Describe this image in detail. Focus on: layout/grid structure, all visual elements (shapes, colors, patterns, numbers, letters), positions, differences/similarities. Be thorough."}]}]
    desc = call(client, model, desc_msg, max_tok=2048)
    if not desc:
        desc = "(no description)"

    if ans_type == "choice" and options:
        answer, raw = do_choice(client, model, question, options, desc, img, hi, desc_msg)
    else:
        answer, raw = do_blank(client, model, question, desc, img, hi, desc_msg)

    traj_dir = os.environ.get("EVAL_TRAJECTORY_DIR")
    idx = os.environ.get("EVAL_INDEX")
    if traj_dir and idx is not None:
        os.makedirs(traj_dir, exist_ok=True)
        with open(os.path.join(traj_dir, f"{idx}.json"), "w") as f:
            json.dump({"index": int(idx), "model": model, "description": desc,
                "question": question, "image_path": image_path,
                "ans_type": ans_type, "options": options,
                "raw_response": raw, "parsed_answer": answer}, f, indent=2)
    return answer


def do_choice(client, model, question, options, desc, img, hi, desc_msg):
    n = len(options)
    labels = ['A','B','C','D'][:n]
    all_letters = all(len(o)==1 and o in 'ABCD' for o in options)

    msgs = list(desc_msg)
    msgs.append({"role": "assistant", "content": desc})

    if all_letters:
        prompt = f"""Now answer this question about the image:
{question}

The options are shown in the image as {', '.join(labels)}.

First, describe what you see in EACH option ({', '.join(labels)}) separately and in detail.
Then, explain step by step which option is correct and why, comparing each option against the requirements.
Finally, give your final answer as ONLY a single letter ({', '.join(labels)}) on the last line."""
    else:
        opts = "\n".join(f"{labels[i]}. {o}" for i, o in enumerate(options))
        prompt = f"""Now answer this question about the image:
{question}

Options:
{opts}

First, describe what you see for each option in detail.
Then, explain step by step which option is correct and why.
Finally, give your final answer as ONLY a single letter ({', '.join(labels)}) on the last line."""

    msgs.append({"role": "user", "content": [img, {"type": "text", "text": prompt}]})
    raw = call(client, model, msgs, max_tok=2048)
    return extract_choice(raw), raw


def do_blank(client, model, question, desc, img, hi, desc_msg):
    q = question.lower()
    is_count = any(w in q for w in ["how many","count","pass through","total"])

    # Grid transcription for grid-based counting
    if is_count and any(w in q for w in ["square","pattern"]) and not any(w in q for w in ["3d","block","cube","line","pass through","point"]):
        gp = f"""Look at this image carefully. The question is: {question}

Your task: Transcribe the image as a grid/matrix. For EACH element, write 'X' if it matches what needs to be counted, or '.' if not.

Write row by row. One row per line. Use only 'X' and '.' separated by spaces. Be very precise."""
        gt = call(client, model, [{"role": "user", "content": [img, {"type": "text", "text": gp}]}], max_tok=2048)
        cnt = gt.count('X')
        if cnt > 0:
            return str(cnt), f"GRID={cnt}\n{gt}"

    # Multi-turn approach
    msgs = list(desc_msg)
    msgs.append({"role": "assistant", "content": desc})
    if is_count:
        ap = f"""{question}

Count methodically:
1. Identify exactly what to count
2. Go row by row, listing each item
3. Sum up
4. Double-check

Put ONLY the count number on the last line."""
    else:
        ap = f"""{question}

Think step by step. Follow the exact format in the question. Put ONLY the answer on the last line."""

    msgs.append({"role": "user", "content": [img, {"type": "text", "text": ap}]})
    ra = call(client, model, msgs, max_tok=2048)
    aa = extract_blank(ra)

    # Single-turn approach
    pb = f"""Question: {question}

Image notes:
{desc}

Think step by step. Follow the exact format. Put ONLY the answer on the last line."""
    rb = call(client, model, [{"role": "user", "content": [img, {"type": "text", "text": pb}]}], max_tok=1024)
    ab = extract_blank(rb)

    if aa == ab:
        return aa, ra
    if is_count:
        try:
            va, vb = int(aa), int(ab)
            return aa if va >= vb else ab, f"A={aa} B={ab}"
        except ValueError:
            pass
    return aa, f"A={aa} B={ab}"


if __name__ == "__main__":
    data = json.loads(sys.stdin.read().strip())
    print(solve(data["question"], data["image_path"], data["ans_type"], data.get("options", [])))
