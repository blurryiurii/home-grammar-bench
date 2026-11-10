import argparse
import configparser
import csv
import json
import statistics
import time
import urllib.error
from datetime import datetime
from pathlib import Path

from hgb.client import LlamaClient
from hgb.grammar import HVAC_MODES, SERVICES, constraint_for, supports

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TIERS = {"gpu": "http://127.0.0.1:8090", "cpu": "http://127.0.0.1:8091"}
CONSTRAINTS = ["none", "generic", "home"]
HOMES = [40, 100, 300]
VALID_SHAPES = [{"service", "entity_id"}, {"service", "entity_id", "data"}, {"abstain"}, {"base_response"}]
FIELDS = [
    "model", "tier", "constraint", "home", "repeat", "id", "split", "category", "subtype", "expected",
    "predicted", "branch_correct", "tool_correct", "params_correct", "valid", "hallucinated", "invalid_call",
    "latency_ms", "prompt_tokens", "output_tokens", "tokens_per_s", "stop_type", "output",
]
RUN_FIELDS = ["model", "tier", "constraint", "home", "repeat", "started", "load_s", "cold_ms", "total_s", "items", "errors"]

VALUE_HINTS = {
    "percent": "0-100",
    "temperature": "degrees",
    "level": "0.0-1.0",
    "boolean": "true or false",
    "color": "color name",
    "hvac-mode": " | ".join(HVAC_MODES),
}

INSTRUCTIONS = """You control the devices in a smart home through Home Assistant. Reply with exactly one JSON object and nothing else.
To act on a device, reply: {"service": "<domain>.<service>", "entity_id": "<entity_id>", "data": {<parameters, only if needed>}}
If the requested device or action does not exist in this home, reply: {"abstain": "<reason>"}
If the request is ambiguous or is not a device command, reply: {"base_response": "<a clarifying question or a normal reply>"}"""


def models():
    config = configparser.ConfigParser()
    config.read(ROOT / "models.ini")
    return config.sections()


def load_home(size):
    return json.loads((DATA / "homes" / f"home_{size}.json").read_text())


def load_questions(path=DATA / "test_questions.json"):
    return json.loads(Path(path).read_text())


def signature(domain, service, params):
    arguments = ", ".join(f"{name}: {VALUE_HINTS[kind]}" for name, kind in params.items())
    return f"{domain}.{service}({arguments})"


def system_prompt(home):
    services = "\n".join(
        ", ".join(signature(domain, service, params) for service, params in domain_services.items())
        for domain, domain_services in SERVICES.items()
    )
    devices = "\n".join(device_line(entity) for entity in home["entities"])
    return f"{INSTRUCTIONS}\n\nServices:\n{services}\n\nDevices:\n{devices}"


def device_line(entity):
    line = f"{entity['entity_id']} '{entity['name']}' ({entity['area']})"
    if set(entity["services"]) != set(SERVICES[entity["domain"]]):
        line += " supports only: " + ", ".join(entity["services"])
    if "features" in entity:
        line += " parameters: " + (", ".join(entity["features"]) or "none (on/off only)")
    return line


def first_json_object(text):
    decoder = json.JSONDecoder()
    for start in (index for index, char in enumerate(text) if char == "{"):
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def predict(text):
    reply = first_json_object(text)
    if reply is None:
        return {"type": "base_response"}
    if "target_device" in reply:
        reply["entity_id"] = reply.pop("target_device")
    if isinstance(reply.get("service"), str) and isinstance(reply.get("entity_id"), str):
        data = reply.get("data") if isinstance(reply.get("data"), dict) else {}
        extra = {key: value for key, value in reply.items() if key not in ("service", "entity_id", "data")}
        return {"type": "call", "service": reply["service"], "entity_id": reply["entity_id"], "data": {**extra, **data}}
    if "abstain" in reply:
        return {"type": "abstain"}
    if "base_response" in reply:
        return {"type": "base_response"}
    return {"type": "malformed"}


def is_valid(text):
    try:
        reply = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(reply, dict) and set(reply) in VALID_SHAPES


def score(item, prediction, home):
    expected = item["expected"]
    is_call = prediction["type"] == "call"
    branch = prediction["type"] == expected["type"]
    tool = params = ""
    if expected["type"] == "call":
        tool = is_call and all(prediction[field] == expected[field] for field in ("service", "entity_id"))
        params = tool and prediction["data"] == expected.get("data", {})
    entities = {entity["entity_id"]: entity for entity in home["entities"]}
    target = entities.get(prediction.get("entity_id"))
    domain, _, service = prediction.get("service", "").partition(".")
    executable = (
        target is not None and target["domain"] == domain and supports(target, service, prediction.get("data", {}))
    )
    return {
        "branch_correct": branch,
        "tool_correct": tool,
        "params_correct": params,
        "hallucinated": is_call and target is None,
        "invalid_call": is_call and not executable,
    }


def warm_up(client, system, constraint):
    start = time.perf_counter()
    prompt = client.render(system, "hello")
    load_s = round(time.perf_counter() - start, 1)
    try:
        cold_ms = client.complete(prompt, constraint, cache_prompt=False)["latency_ms"]
        client.complete(prompt, constraint)
    except urllib.error.HTTPError:
        cold_ms = ""
    return {"load_s": load_s, "cold_ms": cold_ms}


def evaluate(client, system, constraint, item, home):
    row = {
        "id": item["id"], "split": item["split"], "category": item["category"],
        "subtype": item.get("subtype", ""), "expected": item["expected"]["type"],
    }
    start = time.perf_counter()
    try:
        result = client.complete(client.render(system, item["command"]), constraint)
    except urllib.error.HTTPError as error:
        failed = {"branch_correct": False, "tool_correct": "", "params_correct": "", "valid": False,
                  "hallucinated": False, "invalid_call": False}
        if item["expected"]["type"] == "call":
            failed.update(tool_correct=False, params_correct=False)
        return {**row, "predicted": "error", **failed, "output": error.read().decode()}
    latency_ms = round((time.perf_counter() - start) * 1000)
    prediction = predict(result["content"])
    timings = result["timings"]
    return {
        **row,
        "predicted": prediction["type"],
        **score(item, prediction, home),
        "valid": is_valid(result["content"]),
        "latency_ms": latency_ms,
        "prompt_tokens": timings["prompt_n"] + timings.get("cache_n", 0),
        "output_tokens": timings["predicted_n"],
        "tokens_per_s": round(timings["predicted_per_second"], 1),
        "stop_type": result["stop_type"],
        "output": result["content"],
    }


def result_path(model, tier, constraint, home, repeat):
    return RESULTS / tier / f"{model}_{constraint}_{home}_r{repeat}.csv"


def record_run(run):
    path = RESULTS / run["tier"] / "runs.csv"
    is_new = not path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(run)


def run(model, tier, constraint, home_size, items, repeat=1):
    output = result_path(model, tier, constraint, home_size, repeat)
    if output.exists():
        return output
    home = load_home(home_size)
    client = LlamaClient(TIERS[tier], model)
    system = system_prompt(home)
    constraint_body = constraint_for(constraint, home)
    started = datetime.now().isoformat(timespec="seconds")
    start = time.perf_counter()
    timing = warm_up(client, system, constraint_body)
    settings = {"model": model, "tier": tier, "constraint": constraint, "home": home_size, "repeat": repeat}
    rows = [{**settings, **evaluate(client, system, constraint_body, item, home)} for item in items]
    total_s = round(time.perf_counter() - start, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".part")
    with partial.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    partial.rename(output)
    errors = sum(row["predicted"] == "error" for row in rows)
    record_run({**settings, "started": started, **timing, "total_s": total_s, "items": len(rows), "errors": errors})
    return output


def summarize(rows):
    print(f"{'split':<10}{'n':>4}{'branch':>8}{'tool':>6}{'params':>8}{'valid':>7}{'invalid':>9}{'median_ms':>11}")
    for split in list(dict.fromkeys(row["split"] for row in rows)) + ["all"]:
        group = [row for row in rows if split in ("all", row["split"])]
        calls = [row for row in group if row["expected"] == "call"]
        latencies = [row["latency_ms"] for row in group if row["predicted"] != "error"]
        print(
            f"{split:<10}{len(group):>4}"
            f"{sum(row['branch_correct'] for row in group):>8}"
            f"{sum(row['tool_correct'] is True for row in calls):>6}"
            f"{sum(row['params_correct'] is True for row in calls):>8}"
            f"{sum(row['valid'] for row in group):>7}"
            f"{sum(row['invalid_call'] for row in group):>9}"
            f"{statistics.median(latencies) if latencies else '-':>11}"
        )


def main():
    parser = argparse.ArgumentParser(description="Score a llama.cpp model on smart home commands.")
    parser.add_argument("--model", required=True, choices=models())
    parser.add_argument("--tier", choices=TIERS, default="gpu")
    parser.add_argument("--constraint", choices=CONSTRAINTS, default="home")
    parser.add_argument("--home", type=int, choices=HOMES, default=40)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--dataset", type=Path, default=DATA / "test_questions.json")
    args = parser.parse_args()
    output = run(args.model, args.tier, args.constraint, args.home, load_questions(args.dataset), args.repeat)
    with output.open() as file:
        rows = [
            {**row, **{key: row[key] == "True" if row[key] in ("True", "False") else row[key]
                       for key in ("branch_correct", "tool_correct", "params_correct", "valid", "invalid_call")},
             "latency_ms": int(row["latency_ms"] or 0)}
            for row in csv.DictReader(file)
        ]
    summarize(rows)
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
