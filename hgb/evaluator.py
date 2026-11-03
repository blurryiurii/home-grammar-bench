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
from hgb.grammar import HVAC_MODES, SERVICES, constraint_for

ROOT = Path(__file__).resolve().parent.parent
TIERS = {"gpu": "http://127.0.0.1:8090", "cpu": "http://127.0.0.1:8091"}
CONSTRAINTS = ["none", "generic", "home"]
HOMES = [40, 100, 300]
VALID_SHAPES = [{"service", "entity_id"}, {"service", "entity_id", "data"}, {"abstain"}, {"base_response"}]
FIELDS = [
    "model", "tier", "constraint", "home", "id", "split", "category", "expected", "predicted",
    "correct", "action_correct", "valid", "hallucinated", "invalid_call", "latency_ms", "prompt_tokens",
    "tokens_per_s", "stop_type", "output",
]
RUN_FIELDS = ["model", "tier", "constraint", "home", "started", "load_s", "cold_ms", "total_s", "items", "errors"]

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
    if expected["type"] == "call":
        action_correct = is_call and all(prediction[field] == expected[field] for field in ("service", "entity_id"))
        correct = action_correct and prediction["data"] == expected.get("data", {})
    else:
        action_correct = correct = prediction["type"] == expected["type"]
    entity_ids = {entity["entity_id"] for entity in home["entities"]}
    executable = {
        (entity["entity_id"], f"{entity['domain']}.{service}")
        for entity in home["entities"]
        for service in entity["services"]
    }
    return {
        "correct": correct,
        "action_correct": action_correct,
        "hallucinated": is_call and prediction["entity_id"] not in entity_ids,
        "invalid_call": is_call and (prediction["entity_id"], prediction["service"]) not in executable,
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


def record_run(run):
    path = ROOT / "results" / "runs.csv"
    is_new = not path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(run)


def evaluate(client, system, constraint, item, home):
    row = {"id": item["id"], "split": item["split"], "category": item["category"], "expected": item["expected"]["type"]}
    try:
        result = client.complete(client.render(system, item["command"]), constraint)
    except urllib.error.HTTPError as error:
        failed = {"correct": False, "action_correct": False, "valid": False, "hallucinated": False, "invalid_call": False}
        return {**row, "predicted": "error", **failed, "output": error.read().decode()}
    prediction = predict(result["content"])
    return {
        **row,
        "predicted": prediction["type"],
        **score(item, prediction, home),
        "valid": is_valid(result["content"]),
        "latency_ms": result["latency_ms"],
        "prompt_tokens": result["timings"]["prompt_n"] + result["timings"].get("cache_n", 0),
        "tokens_per_s": round(result["timings"]["predicted_per_second"], 1),
        "stop_type": result["stop_type"],
        "output": result["content"],
    }


def summarize(rows):
    print(f"{'split':<10}{'n':>4}{'correct':>9}{'action':>8}{'valid':>7}{'calls':>7}{'invalid':>9}{'median_ms':>11}")
    splits = list(dict.fromkeys(row["split"] for row in rows))
    for split in splits + ["all"]:
        group = [row for row in rows if split in ("all", row["split"])]
        latencies = [row["latency_ms"] for row in group if row["predicted"] != "error"]
        print(
            f"{split:<10}{len(group):>4}"
            f"{sum(row['correct'] for row in group):>9}"
            f"{sum(row['action_correct'] for row in group):>8}"
            f"{sum(row['valid'] for row in group):>7}"
            f"{sum(row['predicted'] == 'call' for row in group):>7}"
            f"{sum(row['invalid_call'] for row in group):>9}"
            f"{statistics.median(latencies) if latencies else '-':>11}"
        )


def main():
    parser = argparse.ArgumentParser(description="Score a llama.cpp model on smart home commands.")
    parser.add_argument("--model", required=True, choices=models())
    parser.add_argument("--tier", choices=TIERS, default="gpu")
    parser.add_argument("--constraint", choices=CONSTRAINTS, default="home")
    parser.add_argument("--home", type=int, choices=HOMES, default=40)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "sample.jsonl")
    args = parser.parse_args()

    home = json.loads((ROOT / "data" / "homes" / f"home_{args.home}.json").read_text())
    items = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    client = LlamaClient(TIERS[args.tier], args.model)
    system = system_prompt(home)
    constraint = constraint_for(args.constraint, home)

    started = datetime.now().isoformat(timespec="seconds")
    start = time.perf_counter()
    timing = warm_up(client, system, constraint)
    run = {"model": args.model, "tier": args.tier, "constraint": args.constraint, "home": args.home}
    rows = [{**run, **evaluate(client, system, constraint, item, home)} for item in items]
    total_s = round(time.perf_counter() - start, 1)

    output = ROOT / "results" / f"{args.model}_{args.tier}_{args.constraint}_{args.home}.csv"
    output.parent.mkdir(exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    errors = sum(row["predicted"] == "error" for row in rows)
    record_run({**run, "started": started, **timing, "total_s": total_s, "items": len(rows), "errors": errors})
    summarize(rows)
    print(f"run took {total_s} s (model load {timing['load_s']} s, uncached first request {timing['cold_ms']} ms)")
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
