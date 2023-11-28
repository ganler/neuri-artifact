import os
import signal
import time

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
import openai
from openai.types.chat import ChatCompletion
import yaml


def make_request(
    client: openai.Client,
    message: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 1,
    n: int = 1,
    **kwargs,
) -> ChatCompletion:
    return client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful coding assistant.",
            },
            {"role": "user", "content": message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        **kwargs,
    )


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


def make_auto_request(*args, **kwargs) -> ChatCompletion:
    ret = None
    while ret is None:
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(100)
            ret = make_request(*args, **kwargs)
            signal.alarm(0)
        except openai.RateLimitError:
            print("Rate limit exceeded. Waiting...")
            signal.alarm(0)
            time.sleep(5)
        except openai.APIConnectionError:
            print("API connection error. Waiting...")
            signal.alarm(0)
            time.sleep(5)
        except openai.APIError as e:
            print(e)
            signal.alarm(0)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            signal.alarm(0)
            time.sleep(1)
    return ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", help="Path to validate")
    parser.add_argument("--replicas", help="Number of replicas", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    progress_bar = Progress(
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )

    prompts = []
    apis = []
    for path in os.listdir("prompts"):
        with open(os.path.join("prompts", path)) as f:
            prompt = yaml.load(f, Loader=yaml.Loader)
        apis.append(prompt["api"])
        prompts.append(prompt["prompt"])

    client = openai.OpenAI()

    with progress_bar as p:
        for api, prompt in p.track(
            list(zip(apis, prompts)), description="Processing..."
        ):
            target_path = os.path.join(args.output_dir, f"{api}.txt")
            # skip if exist
            if os.path.exists(target_path):
                continue

            ret = make_auto_request(
                client,
                message=prompt,
                model="gpt-4-1106-preview",
                # model="gpt-3.5-turbo",
                max_tokens=768,
                temperature=0.2,
                n=args.replicas,
                response_format={"type": "text"},
            )

            with open(target_path, "w") as f:
                for sample in ret.choices:
                    f.write(sample.message.content)
                    f.write("\n")
                    f.write("@" * 16)
                    f.write("\n")
