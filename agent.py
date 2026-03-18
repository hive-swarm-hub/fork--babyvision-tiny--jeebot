"""BabyVision solver — visual reasoning on early visual understanding tasks.

Takes a JSON task on stdin (question, image_path, ans_type, options), prints the answer on stdout.
Saves full LLM trajectory to eval_results/trajectories/<index>.json if EVAL_TRAJECTORY_DIR is set.
"""

import sys
import os
import json
import base64
import re

from openai import OpenAI


def solve(question: str, image_path: str, ans_type: str, options: list) -> str:
    client = OpenAI()
    model = os.environ.get("SOLVER_MODEL", "gpt-5.4-mini")

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}

    # Step 1: Describe the image in detail
    describe_prompt = (
        "Look at this image very carefully. Describe every detail you see: "
        "all shapes, colors, patterns, numbers, labels, grid structure, rows, columns, "
        "and any text or symbols. Be thorough and systematic. "
        "If there is a grid, describe it row by row. "
        "If there are numbered items, list each one. "
        "If there are options labeled A/B/C/D or 1/2/3/4, describe each option."
    )

    messages = [
        {"role": "user", "content": [
            image_content,
            {"type": "text", "text": describe_prompt},
        ]},
    ]

    desc_response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_completion_tokens=2048,
    )
    description = desc_response.choices[0].message.content.strip()

    # Step 2: Answer based on description
    messages.append({"role": "assistant", "content": description})

    if ans_type == "choice" and options:
        # Use letter labels to avoid index confusion
        letters = ["A", "B", "C", "D"]
        opts = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))
        answer_prompt = (
            f"Now answer this question about the image:\n\n"
            f"{question}\n\n"
            f"Options:\n{opts}\n\n"
            f"Think step by step about which option is correct. "
            f"Then give your final answer as ONLY the letter (A, B, C, or D)."
        )
    else:
        answer_prompt = (
            f"Now answer this question about the image:\n\n"
            f"{question}\n\n"
            f"Think step by step. Then give your final answer on the last line, "
            f"with ONLY the answer value and nothing else. "
            f"For coordinates use format (row,col). "
            f"For lists use commas without spaces like: 1-7,2-9,3-10. "
            f"For counts give just the number."
        )

    messages.append({"role": "user", "content": answer_prompt})

    answer_response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_completion_tokens=1024,
    )

    raw_output = answer_response.choices[0].message.content.strip()

    # Extract final answer (last line)
    lines = [l.strip() for l in raw_output.split("\n") if l.strip()]
    final_answer = lines[-1] if lines else raw_output

    # Post-process
    if ans_type == "choice":
        # Convert letter to 0-indexed number
        letter_map = {"A": "0", "B": "1", "C": "2", "D": "3"}
        # Find letter in the answer
        for letter, idx in letter_map.items():
            if letter in final_answer.upper():
                final_answer = idx
                break
        else:
            # Try to find a number and convert
            nums = re.findall(r'\d+', final_answer)
            if nums:
                n = int(nums[0])
                # If 1-indexed, convert to 0-indexed
                if 1 <= n <= 4:
                    final_answer = str(n - 1)
                else:
                    final_answer = nums[0]
    else:
        # Clean up blank answers: remove extra spaces around separators
        final_answer = re.sub(r'\s*,\s*', ',', final_answer)
        final_answer = re.sub(r'\s*-\s*', '-', final_answer)
        # Remove any trailing period
        final_answer = final_answer.rstrip('.')

    # Save trajectory if requested
    traj_dir = os.environ.get("EVAL_TRAJECTORY_DIR")
    idx = os.environ.get("EVAL_INDEX")
    if traj_dir and idx is not None:
        os.makedirs(traj_dir, exist_ok=True)
        trajectory = {
            "index": int(idx),
            "model": model,
            "question": question,
            "image_path": image_path,
            "ans_type": ans_type,
            "options": options,
            "description": description,
            "raw_response": raw_output,
            "final_answer": final_answer,
            "usage": {
                "describe_tokens": {
                    "prompt": desc_response.usage.prompt_tokens if desc_response.usage else None,
                    "completion": desc_response.usage.completion_tokens if desc_response.usage else None,
                },
                "answer_tokens": {
                    "prompt": answer_response.usage.prompt_tokens if answer_response.usage else None,
                    "completion": answer_response.usage.completion_tokens if answer_response.usage else None,
                },
            },
        }
        with open(os.path.join(traj_dir, f"{idx}.json"), "w") as f:
            json.dump(trajectory, f, indent=2)

    return final_answer


if __name__ == "__main__":
    data = json.loads(sys.stdin.read().strip())
    print(solve(data["question"], data["image_path"], data["ans_type"], data.get("options", [])))
