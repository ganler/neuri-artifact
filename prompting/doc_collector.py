import requests
import re
from bs4 import BeautifulSoup
import yaml
import os

# TODO: https://pytorch.org/docs/stable/torch.html
DOC_ROOTS = [
    "https://pytorch.org/docs/stable/nn.functional.html",
    "https://pytorch.org/docs/stable/signal.html",
    "https://pytorch.org/docs/stable/linalg.html",
    "https://pytorch.org/docs/stable/fft.html",
]

# Skip APIs that NNSmith has already supported well.
BLOCK_LIST = [
    "torch.nn.functional.interpolate",
    "torch.nn.functional.pad",
    "torch.nn.functional.relu",
    "torch.nn.functional.linear",
    "torch.nn.functional.leaky_relu",
    "torch.atan",
    "torch.abs",
    "torch.acos",
    "torch.asin",
    "torch.cos",
    "torch.sin",
    "torch.add",
    "torch.where",
    "torch.tan",
    "torch.mul",
]


def parse_doc(url):
    docr = requests.get(url)
    # assert not 404
    assert docr.status_code != 404
    # capture doc between <dl class="py function"> (or <dl class="py function">) and </dl>
    regex = re.compile(
        r'<dl class="py (function|class)">(.*)</dl>',
        re.DOTALL,
    )
    content = regex.findall(docr.text)[0][1]
    text = BeautifulSoup(content).get_text()
    # skip if too simple
    if len(text.split("¶")[-1]) < 128:
        print(f"Skipping --- due to too short description....")
        print(text)
        print("======================================\n\n\n\n")
    return text


if __name__ == "__main__":
    docs = {}
    for root in DOC_ROOTS:
        r = requests.get(root)
        regex = re.compile(
            r'<tr class="row-odd"><td><p><p id="(.*)"/><a class="reference internal" href="(.*)" title="(.*)"><code class="xref py py-obj docutils literal notranslate"><span class="pre">(.*)</span></code></a></p></td>'
        )
        matches = regex.findall(r.text)
        for match in matches:
            api: str = match[0]
            if api.endswith("_"):
                continue  # skip
            if api in BLOCK_LIST:
                continue  # skip
            doc = parse_doc(f"https://pytorch.org/docs/stable/{match[1]}")
            if doc is None:
                continue
            docs[api] = doc

    print(len(docs), "API docs fetched")
    prefix = os.path.join(os.path.dirname(__file__), "prompts")
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    for api, doc in docs.items():
        with open(os.path.join(prefix, f"{api}.yaml"), "w") as f:
            yaml.dump({"api": api, "doc": doc}, f)
