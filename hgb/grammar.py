import json
import sys
from itertools import combinations

SERVICES = {
    "light": {
        "turn_on": {"brightness_pct": "percent", "color_name": "color"},
        "turn_off": {},
        "toggle": {},
    },
    "climate": {
        "set_temperature": {"temperature": "temperature"},
        "set_hvac_mode": {"hvac_mode": "hvac-mode"},
        "turn_on": {},
        "turn_off": {},
    },
    "lock": {"lock": {}, "unlock": {}},
    "camera": {"turn_on": {}, "turn_off": {}},
    "media_player": {
        "turn_on": {},
        "turn_off": {},
        "media_play": {},
        "media_pause": {},
        "media_stop": {},
        "volume_set": {"volume_level": "level"},
        "volume_mute": {"is_volume_muted": "boolean"},
    },
    "cover": {
        "open_cover": {},
        "close_cover": {},
        "stop_cover": {},
        "set_cover_position": {"position": "percent"},
    },
}

OPTIONAL_DATA = {"light.turn_on"}

HVAC_MODES = ["off", "heat", "cool", "heat_cool", "auto", "dry", "fan_only"]

TEXT_LIMIT = 200

GENERIC_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "entity_id": {"type": "string"},
                "data": {"type": "object"},
            },
            "required": ["service", "entity_id"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"abstain": {"type": "string", "maxLength": TEXT_LIMIT}},
            "required": ["abstain"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"base_response": {"type": "string", "maxLength": TEXT_LIMIT}},
            "required": ["base_response"],
            "additionalProperties": False,
        },
    ]
}


def literal(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def json_string(text):
    return literal(json.dumps(text))


COMMA = literal(", ")
DATA_OPEN = literal(', "data": {')
CLOSE = literal("}")

VALUES = {
    "percent": '"100" | [1-9]? [0-9]',
    "temperature": '[1-9] [0-9] ("." [0-9])?',
    "level": '"0" ("." [0-9]{1,2})? | "1" (".0")?',
    "boolean": '"true" | "false"',
    "color": r'"\"" [a-z]{3,20} "\""',
    "hvac-mode": " | ".join(json_string(mode) for mode in HVAC_MODES),
    "text": rf'"\"" [^"\\\x7F\x00-\x1F]{{1,{TEXT_LIMIT}}} "\""',
}


def rule_name(service):
    return service.replace(".", "-").replace("_", "-")


def data_rule(service, params):
    fields = [f"{literal(json.dumps(name) + ': ')} {value}" for name, value in params.items()]
    separator = f" {COMMA} "
    if service in OPTIONAL_DATA:
        subsets = [
            separator.join(subset)
            for size in range(1, len(fields) + 1)
            for subset in combinations(fields, size)
        ]
        return f"({DATA_OPEN} ({' | '.join(subsets)}) {CLOSE})?"
    return f"{DATA_OPEN} {separator.join(fields)} {CLOSE}"


def home_grammar(home):
    call_rules = {}
    entity_rules = {}
    for domain, services in SERVICES.items():
        for service, params in services.items():
            name = f"{domain}.{service}"
            entities = [
                entity["entity_id"]
                for entity in home["entities"]
                if entity["domain"] == domain and service in entity["services"]
            ]
            if not entities:
                continue
            rule = rule_name(name)
            entity_rules[f"{rule}-entity"] = " | ".join(json_string(entity) for entity in entities)
            parts = [literal(f'{{"service": "{name}", "entity_id": '), f"{rule}-entity"]
            if params:
                parts.append(data_rule(name, params))
            parts.append(CLOSE)
            call_rules[rule] = " ".join(parts)

    rules = {
        "root": "call | abstain | base-response",
        "call": " | ".join(call_rules),
        "abstain": " ".join([literal('{"abstain": '), "text", CLOSE]),
        "base-response": " ".join([literal('{"base_response": '), "text", CLOSE]),
        **call_rules,
        **entity_rules,
        **VALUES,
    }
    return "".join(f"{name} ::= {body}\n" for name, body in rules.items())


def constraint_for(mode, home):
    if mode == "home":
        return {"grammar": home_grammar(home)}
    if mode == "generic":
        return {"json_schema": GENERIC_SCHEMA}
    return {}


if __name__ == "__main__":
    with open(sys.argv[1]) as file:
        print(home_grammar(json.load(file)), end="")
