import os

# from vllm import LLM, SamplingParams
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

progress_bar = Progress(
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
)

import yaml

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", help="Path to validate")
    parser.add_argument("--replicas", help="Number of replicas", type=int, default=3)
    args = parser.parse_args()

    prompts = []
    apis = []
    for path in os.listdir("prompts"):
        with open(os.path.join("prompts", path)) as f:
            prompt = yaml.load(f, Loader=yaml.Loader)
        apis.append(prompt["api"])
        prompts.append(prompt["prompt"])

    prompts = [
        f"""You are an expert in PyTorch programming proficient in extracting and symbolizing properties of some PyTorch functions.
    ### Instruction:
    {prompt}
    ### Response:
    """
        for prompt in prompts
    ]

    bit4config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_name = "deepseek-ai/deepseek-coder-33b-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # use_flash_attention_2=True,
        quantization_config=bit4config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with torch.no_grad():
        with progress_bar as p:
            for api, prompt in p.track(
                list(zip(apis, prompts)), description="Processing..."
            ):
                outputs = []
                input_tokens = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
                print(f"Input tokens: {input_tokens.shape}")
                for _ in range(args.replicas):
                    output = model.generate(
                        input_tokens,
                        do_sample=True,
                        top_p=0.95,
                        top_k=50,
                        temperature=0.2,
                        num_return_sequences=1,
                        max_new_tokens=768,
                        eos_token_id=32021,
                    )
                    outputs.append(
                        tokenizer.decode(output[0], skip_special_tokens=True).replace(
                            prompt, ""
                        )
                    )

                with open(os.path.join(args.output_dir, f"{api}.txt"), "w") as f:
                    for sample in outputs:
                        f.write(sample)
                        f.write("\n")
                        f.write("@" * 16)
                        f.write("\n")
