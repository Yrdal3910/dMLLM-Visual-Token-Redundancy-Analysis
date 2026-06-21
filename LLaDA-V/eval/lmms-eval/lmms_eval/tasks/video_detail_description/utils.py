import ast
import datetime
import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml

import lmms_eval.tasks._task_utils.file_utils as file_utils

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


NUM_SECONDS_TO_SLEEP = 5

GPT_EVAL_MODEL_NAME = os.getenv("MODEL_VERSION", "gpt-4o-mini")

API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

# A bit ugly here
# But the idea is that we will unzip all the zip files
# To HF HOME cache dir
# And load it here
HF_HOME = os.environ["HF_HOME"]
cache_dir = config["dataset_kwargs"]["cache_dir"]
cache_dir = os.path.join(HF_HOME, cache_dir)
cache_dir = os.path.join(cache_dir, "Test_Videos")

from loguru import logger as eval_logger


# Pass in video path here
# Can only work correctly with video llm
def video_detail_description_doc_to_visual(doc):
    video_path = doc["video_name"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


# format the question
def video_detail_description_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    question = doc["question"]
    return f"{pre_prompt}{question}{post_prompt}"


def video_detail_description_doc_to_answer(doc):
    return doc["answer"]


def get_eval_generic(question, answer, pred, max_tokens: int, retries: int = 5):
    global headers

    messages = [
        {
            "role": "system",
            "content": "You are an intelligent chatbot designed for evaluating the detail orientation of generative outputs for video-based question-answer pairs. "
            "Your task is to compare the predicted answer with the correct answer and determine its level of detail, considering both completeness and specificity. Here's how you can accomplish the task:"
            "------"
            "##INSTRUCTIONS: "
            "- Check if the predicted answer covers all major points from the video. The response should not leave out any key aspects.\n"
            "- Evaluate whether the predicted answer includes specific details rather than just generic points. It should provide comprehensive information that is tied to specific elements of the video.\n"
            "- Consider synonyms or paraphrases as valid matches.\n"
            "- Provide a single evaluation score that reflects the level of detail orientation of the prediction, considering both completeness and specificity.",
        },
        {
            "role": "user",
            "content": "Please evaluate the following video-based question-answer pair:\n\n"
            f"Question: {question}\n"
            f"Correct Answer: {answer}\n"
            f"Predicted Answer: {pred}\n\n"
            "Provide your evaluation only as a detail orientation score where the detail orientation score is an integer value between 0 and 5, with 5 indicating the highest level of detail orientation. "
            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the detail orientation score in INTEGER, not STRING."
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
            "For example, your response should look like this: {'score': 4.8}.",
        },
    ]

    payload = {
        "model": GPT_EVAL_MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        # "response_format": {"type": "json_object"},
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()  # Raises HTTPError for bad responses
            try:
                response_data = response.json()  # Attempt to parse JSON
            except requests.exceptions.JSONDecodeError:
                eval_logger.error(f"JSON decode error on attempt {attempt + 1}. Response text: {response.text}")
                continue  # Skip to next retry
            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content, response_data["model"]
        # Handle HTTP errors separately
        except requests.exceptions.HTTPError as e:
            eval_logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
        # Handle other requests-related errors
        except requests.exceptions.RequestException as e:
            eval_logger.error(f"Request exception on attempt {attempt + 1}: {e}")
        except Exception as e:
            eval_logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")

        if "Sorry! We've encountered an issue with repetitive patterns in your prompt. Please try again with a different prompt." in json.loads(response.content)["error"]["message"]:
            eval_logger.error(f"Repetitive patterns in prompt. Drop this data.")
            return "", ""

        # Handle other unexpected errors
        if attempt < retries - 1:
            time.sleep(NUM_SECONDS_TO_SLEEP)
        else:  # If this was the last attempt, log and return empty
            eval_logger.error(f"All {retries} attempts failed.")
            return "", ""

    return "", ""


def parse_score(review):
    try:
        # Convert the string representation of a dictionary to an actual dictionary
        review_dict = ast.literal_eval(review)
        score = review_dict.get("score", 0)
        return int(score)
    except SyntaxError as e:
        eval_logger.error(f"Syntax error parsing the review string: {e}. Review content: {review}")
        return 0
    except ValueError as e:
        eval_logger.error(f"Value error parsing the review string: {e}. Review content: {review}")
        return 0
    except Exception as e:
        eval_logger.error(f"Unexpected error parsing the review string: {e}. Review content: {review}")
        return 0


def gpt_eval(data_dict):
    evaluated_results = []

    try:
        question = data_dict["question"]
        answer = data_dict["answer"]
        pred = data_dict["pred"]

        # Assume get_eval returns a review and the model name, and parse_score parses this review
        review, model_name = get_eval_generic(question, answer, pred, 64)
        score = parse_score(review)
    except Exception as e:
        eval_logger.error(f"Error for Video Name: {data_dict.get('video_name', 'Unknown')}: {e}")
        review = "Failed to Get a Proper Review."
        model_name = ""
        score = 0

    # Update the dictionary with the new entries
    updated_dict = {
        "video_name": data_dict["video_name"],
        "review": review,
        "score": score,
    }

    return updated_dict


# Process result for evaluation in generic task
def video_detail_description_process_results_generic(doc, result):
    pred = result[0]
    doc["pred"] = pred
    eval_results = gpt_eval(doc)

    return {"gpt_eval_score": {"video_name": doc["video_name"], "question": doc["question"], "answer": doc["answer"], "pred": pred, "score": eval_results["score"], "review": eval_results["review"]}}


def video_detail_description_aggregate_score(results, args):
    score = 0
    for result in results:
        eval_score = result["score"]
        try:
            eval_score = int(eval_score)
        except:
            eval_score = 0.0

        score += eval_score

    return score / len(results)

if __name__ == "__main__":
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    import json

    def _build_session():
        # 构造带重试的 Session
        session = requests.Session()
        retry = Retry(
            total=5,                # 总重试次数
            read=5,
            connect=5,
            backoff_factor=1.2,     # 指数退避：1.2, 2.4, 3.6, ...
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["POST"])  # 只对 POST 重试
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    # 在模块加载时创建一个全局复用的 Session
    _SESSION = _build_session()

    def get_eval_generic(question, answer, pred, max_tokens: int, retries: int = 5):
        global headers

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an intelligent chatbot designed for evaluating the detail orientation of generative outputs for video-based question-answer pairs. "
                    "Your task is to compare the predicted answer with the correct answer and determine its level of detail, considering both completeness and specificity. "
                    "------"
                    "##INSTRUCTIONS: "
                    "- Check if the predicted answer covers all major points from the video. The response should not leave out any key aspects.\n"
                    "- Evaluate whether the predicted answer includes specific details rather than just generic points. It should provide comprehensive information that is tied to specific elements of the video.\n"
                    "- Consider synonyms or paraphrases as valid matches.\n"
                    "- Provide a single evaluation score that reflects the level of detail orientation of the prediction, considering both completeness and specificity."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Please evaluate the following video-based question-answer pair:\n\n"
                    f"Question: {question}\n"
                    f"Correct Answer: {answer}\n"
                    f"Predicted Answer: {pred}\n\n"
                    "Provide your evaluation only as a detail orientation score where the detail orientation score is an integer value between 0 and 5, with 5 indicating the highest level of detail orientation. "
                    "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the detail orientation score in INTEGER, not STRING."
                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                    "For example, your response should look like this: {'score': 4.8}."
                ),
            },
        ]

        payload = {
            "model": GPT_EVAL_MODEL_NAME,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }

        last_error_text = None

        for attempt in range(retries):
            try:
                # 分开连接/读取超时，SSL 抖动更稳一些
                resp = _SESSION.post(API_URL, headers=headers, json=payload, timeout=(10, 60))
                # 如果是 4xx/5xx，这里会抛 HTTPError；否则继续解析
                resp.raise_for_status()

                try:
                    data = resp.json()
                except json.JSONDecodeError:
                    eval_logger.error(f"JSON decode error on attempt {attempt + 1}. Response text: {resp.text[:300]}")
                    last_error_text = resp.text
                    # 退避后重试
                else:
                    # 成功拿到 JSON
                    # 先检查是否是错误结构（有些服务返回 200 但 payload 内含 error）
                    if isinstance(data, dict) and "error" in data:
                        err_msg = data.get("error", {}).get("message", "")
                        eval_logger.error(f"API logical error on attempt {attempt + 1}: {err_msg}")
                        last_error_text = err_msg
                    else:
                        content = data["choices"][0]["message"]["content"].strip()
                        if content:
                            return content, data.get("model", "")
                        # 空内容也当作错误重试
                        last_error_text = f"Empty content on attempt {attempt + 1}"

            except requests.exceptions.HTTPError as e:
                # 有响应对象时尽量读取其内容帮助调试
                err_payload = None
                if e.response is not None:
                    try:
                        err_payload = e.response.json()
                    except Exception:
                        err_payload = e.response.text[:300]
                eval_logger.error(f"HTTP error on attempt {attempt + 1}: {e}; payload: {err_payload}")
                last_error_text = str(e)
            except requests.exceptions.RequestException as e:
                # 包含连接错误/SSL/超时等
                eval_logger.error(f"Request exception on attempt {attempt + 1}: {e}")
                last_error_text = str(e)
            except Exception as e:
                eval_logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                last_error_text = str(e)

            # 指定错误消息（如果有响应）再判断；没有响应就跳过
            # 这段逻辑放在 try 内已覆盖，此处不再访问 resp
            if attempt < retries - 1:
                time.sleep(NUM_SECONDS_TO_SLEEP)
            else:
                eval_logger.error(f"All {retries} attempts failed. Last error: {last_error_text}")
                return "", ""

        return "", ""

    in_file = "/pfs/zuhaoyang/workspace/LLaDA-V/eval/exp/llava_v_eval/LLaDA-V/20251009_094957_samples_video_dc499.jsonl"
    out_file = "/pfs/zuhaoyang/workspace/LLaDA-V/eval/exp/llava_v_eval/LLaDA-V/20251009_094957_samples_video_dc499_gpt_eval.jsonl"

    results = []
    total = 0

    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)

            doc = sample["doc"]
            question = doc["question"]
            answer = doc["answer"]
            pred = doc.get("pred", "")

            review, model_name = get_eval_generic(question, answer, pred, max_tokens=64)
            score = parse_score(review)
            print(score)

            results.append({
                "video_name": doc.get("video_name", ""),
                "question": question,
                "answer": answer,
                "pred": pred,
                "score": score,
                "review": review,
                "model_used": model_name,
            })
            total += score

    avg_score = total / len(results) if results else 0
    out_obj = {"results": results, "average_score": avg_score}

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print("=== Overall Scores ===")
    print(f"Average Score: {avg_score:.2f} over {len(results)} items")
    print(f"Scored results saved to: {out_file}")