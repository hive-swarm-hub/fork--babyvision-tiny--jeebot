"""BabyVision solver — minimal approach for maximum determinism."""

import sys, os, json, base64, re, io
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


def call(client, model, messages, max_tokens=2048):
    for _ in range(2):
        resp = client.chat.completions.create(
            model=model, messages=messages,
            temperature=0, max_completion_tokens=max_tokens, seed=42)
        c = resp.choices[0].message.content
        if c and c.strip():
            return c.strip()
    return ""


def extract_choice(raw):
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    ans = lines[-1] if lines else raw
    letter_map = {'A': '0', 'B': '1', 'C': '2', 'D': '3'}
    m = re.search(r'\b([A-D])\b', ans)
    if m and m.group(1) in letter_map:
        return letter_map[m.group(1)]
    m = re.search(r'\b([0-3])\b', ans)
    if m:
        return m.group(1)
    for line in reversed(lines):
        m = re.search(r'\b([A-D])\b', line)
        if m and m.group(1) in letter_map:
            return letter_map[m.group(1)]
    return ans


def extract_blank(raw):
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    ans = lines[-1] if lines else raw
    ans = ans.replace('**', '')
    ans = re.sub(r'\s*,\s*', ',', ans)
    if re.search(r'\d\s*-\s*\d', ans):
        ans = re.sub(r'\s*-\s*', '-', ans)
    return ans.rstrip('.')


def solve(question, image_path, ans_type, options):
    client = OpenAI()
    model = os.environ.get("SOLVER_MODEL", "gpt-5.4-mini")
    b64 = load_image_b64(image_path)
    img = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
    hi = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}}
    q = question.lower()

    if ans_type == "choice" and options:
        n = len(options)
        labels = ['A', 'B', 'C', 'D'][:n]
        all_letters = all(len(o) == 1 and o in 'ABCD' for o in options)
        if all_letters:
            prompt = f"""{question}

The options are shown in the image as {', '.join(labels)}.

Look at the image very carefully. First, describe what you see in EACH option ({', '.join(labels)}) separately and in detail. Then, explain step by step which option is correct and why, comparing each option against the requirements. Finally, give your final answer as ONLY a single letter ({', '.join(labels)}) on the last line."""
        else:
            opts = "\n".join(f"{labels[i]}. {o}" for i, o in enumerate(options))
            prompt = f"""{question}

Options:
{opts}

Look at the image very carefully. First, describe what you see for each option. Then, explain step by step which option is correct and why. Finally, give your final answer as ONLY a single letter ({', '.join(labels)}) on the last line."""
        raw = call(client, model, [{"role": "user", "content": [hi, {"type": "text", "text": prompt}]}])
        answer = extract_choice(raw)

    else:
        is_count = any(w in q for w in ["how many", "count", "pass through", "total"])
        is_grid = is_count and any(w in q for w in ["square", "pattern"]) and not any(w in q for w in ["3d", "block", "cube"])

        if is_grid:
            gp = f"""Look at this image carefully. The question is: {question}

Your task: Transcribe the image as a grid/matrix. For EACH element, write 'X' if it matches what needs to be counted, or '.' if not.
Write row by row. One row per line. Use only 'X' and '.' separated by spaces. Be very precise."""
            gt = call(client, model, [{"role": "user", "content": [hi, {"type": "text", "text": gp}]}])
            cnt = gt.count('X')
            if cnt > 0:
                answer = str(cnt)
                raw = f"GRID={cnt}\n{gt}"
            else:
                prompt = f"""{question}\n\nCount carefully. Put ONLY the number on the last line."""
                raw = call(client, model, [{"role": "user", "content": [hi, {"type": "text", "text": prompt}]}])
                answer = extract_blank(raw)
        elif is_count:
            prompt = f"""{question}

Look at the image very carefully. Count methodically:
1. Identify exactly what needs to be counted
2. Go row by row (or section by section), listing each item with its position
3. Sum up the total
4. Double-check by counting again from a different starting point

Put ONLY the final count number on the last line."""
            raw = call(client, model, [{"role": "user", "content": [hi, {"type": "text", "text": prompt}]}])
            answer = extract_blank(raw)
        else:
            prompt = f"""{question}

Look at the image very carefully. Think step by step. Pay close attention to the exact format requested in the question. Give your final answer in the exact format requested. Put ONLY the answer value on the last line."""
            raw = call(client, model, [{"role": "user", "content": [hi, {"type": "text", "text": prompt}]}])
            answer = extract_blank(raw)

    traj_dir = os.environ.get("EVAL_TRAJECTORY_DIR")
    idx = os.environ.get("EVAL_INDEX")
    if traj_dir and idx is not None:
        os.makedirs(traj_dir, exist_ok=True)
        with open(os.path.join(traj_dir, f"{idx}.json"), "w") as f:
            json.dump({"index": int(idx), "model": model, "question": question,
                "image_path": image_path, "ans_type": ans_type, "options": options,
                "raw_response": raw, "parsed_answer": answer}, f, indent=2)
    return answer


if __name__ == "__main__":
    data = json.loads(sys.stdin.read().strip())
    print(solve(data["question"], data["image_path"], data["ans_type"], data.get("options", [])))
