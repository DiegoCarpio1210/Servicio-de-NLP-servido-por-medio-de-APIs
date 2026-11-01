import argparse
import collections
import json
import random
import re
import string
import time
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:8001"
TOKEN = "pimientero-prueba-local-2026"

DATA_URL = (
    "https://rajpurkar.github.io/"
    "SQuAD-explorer/dataset/dev-v2.0.json"
)

OUT = Path("logs/pimientero_evaluacion")
OUT.mkdir(parents=True, exist_ok=True)

DATA_FILE = OUT / "dev-v2.0.json"


def normalize(text):
    text = (text or "").lower()
    text = "".join(
        c for c in text if c not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(pred, gold):
    return int(normalize(pred) == normalize(gold))


def f1_score(pred, gold):
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = (
        collections.Counter(pred_tokens)
        & collections.Counter(gold_tokens)
    )

    same = sum(common.values())

    if same == 0:
        return 0.0

    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def request_json(url, payload=None):
    headers = {
        "Authorization": "Bearer " + TOKEN
    }

    data = None

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(
        url,
        data=data,
        headers=headers
    )

    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


def load_dataset():
    if not DATA_FILE.exists():
        print("Descargando SQuAD 2.0...")
        urllib.request.urlretrieve(
            DATA_URL,
            DATA_FILE
        )

    with DATA_FILE.open(encoding="utf-8") as f:
        data = json.load(f)

    examples = []

    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]

            # Respeta el limite actual de nuestra API.
            if len(context) > 5000:
                continue

            for qa in paragraph["qas"]:
                answers = [
                    a["text"]
                    for a in qa["answers"]
                    if normalize(a["text"])
                ]

                examples.append({
                    "id": qa["id"],
                    "context": context,
                    "question": qa["question"],
                    "answers": answers,
                    "answerable": bool(answers)
                })

    return examples


def previously_used_ids():
    ids = set()

    base = Path("logs/deberta_base_training")

    if not base.exists():
        print(
            "Aviso: no existe el directorio de logs "
            "anteriores."
        )
        return ids

    # Los IDs de SQuAD suelen ser cadenas
    # hexadecimales de 24 caracteres.
    pattern = re.compile(
        r"\b[0-9a-fA-F]{24}\b"
    )

    for path in base.rglob("*.json"):
        try:
            content = path.read_text(
                encoding="utf-8"
            )
            ids.update(pattern.findall(content))
        except Exception:
            pass

    return ids


def select_examples(examples, total, used_ids):
    answerable = [
        x for x in examples
        if x["answerable"]
        and x["id"] not in used_ids
    ]

    impossible = [
        x for x in examples
        if not x["answerable"]
        and x["id"] not in used_ids
    ]

    rng = random.Random(20261031)

    rng.shuffle(answerable)
    rng.shuffle(impossible)

    n_answerable = total * 3 // 4
    n_impossible = total - n_answerable

    if (
        len(answerable) < n_answerable
        or len(impossible) < n_impossible
    ):
        raise RuntimeError(
            "No hay suficientes ejemplos elegibles "
            "despues de excluir IDs previos."
        )

    selected = (
        answerable[:n_answerable]
        + impossible[:n_impossible]
    )

    rng.shuffle(selected)

    return selected


def calculate_metrics(rows):
    if not rows:
        return {
            "count": 0,
            "EM": 0.0,
            "F1": 0.0
        }

    return {
        "count": len(rows),
        "EM": round(
            100 * sum(x["em"] for x in rows) / len(rows),
            2
        ),
        "F1": round(
            100 * sum(x["f1"] for x in rows) / len(rows),
            2
        )
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--total",
        type=int,
        default=1000,
        choices=[500, 1000]
    )

    args = parser.parse_args()

    total = args.total

    health = request_json(API + "/health")

    if health.get("status") != "ok":
        raise RuntimeError(
            "La API no esta lista."
        )

    info = request_json(API + "/info")

    print("\nMODELO ACTIVO:")
    print("Modelo:", info.get("model"))
    print("Checkpoint:", info.get("training_steps"))
    print(
        "Umbral:",
        info.get("no_answer_threshold")
    )

    examples = load_dataset()

    used_ids = previously_used_ids()

    selected = select_examples(
        examples,
        total,
        used_ids
    )

    print("\nEjemplos anteriores excluidos:", len(used_ids))
    print("Preguntas seleccionadas:", len(selected))

    selection_file = OUT / f"selection_{total}.json"

    # Guardar IDs y preguntas para reproducibilidad.
    selection_file.write_text(
        json.dumps(
            selected,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    results_file = OUT / f"results_{total}.json"

    results = []

    if results_file.exists():
        results = json.loads(
            results_file.read_text(
                encoding="utf-8"
            )
        )

    completed = {
        x["id"] for x in results
    }

    print("Pruebas anteriores recuperadas:", len(completed))

    start_time = time.time()

    for i, example in enumerate(selected, start=1):
        if example["id"] in completed:
            continue

        try:
            response = request_json(
                API + "/predict",
                {
                    "id": example["id"],
                    "context": example["context"],
                    "question": example["question"]
                }
            )

            pred = response.get("answer")

            if pred is None:
                pred = ""

            golds = example["answers"] or [""]

            em = max(
                exact_match(pred, gold)
                for gold in golds
            )

            f1 = max(
                f1_score(pred, gold)
                for gold in golds
            )

            row = {
                "id": example["id"],
                "question": example["question"],
                "expected": golds,
                "predicted": pred,
                "answerable": example["answerable"],
                "em": em,
                "f1": f1,
                "error": None
            }

        except Exception as exc:
            row = {
                "id": example["id"],
                "question": example["question"],
                "expected": example["answers"],
                "predicted": "",
                "answerable": example["answerable"],
                "em": 0,
                "f1": 0,
                "error": str(exc)
            }

        results.append(row)

        # Guardar avance para recuperar una ejecucion
        # interrumpida sin perder las respuestas.
        if i % 25 == 0 or i == total:
            results_file.write_text(
                json.dumps(
                    results,
                    ensure_ascii=False,
                    indent=2
                ),
                encoding="utf-8"
            )

            metrics = calculate_metrics(results)

            print(
                f"[{i}/{total}] "
                f"EM={metrics['EM']}% "
                f"F1={metrics['F1']}%"
            )

    # Guardado final.
    results_file.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    answerable_rows = [
        x for x in results
        if x["answerable"]
    ]

    impossible_rows = [
        x for x in results
        if not x["answerable"]
    ]

    failures = [
        x for x in results
        if x["error"] is not None
    ]

    report = {
        "model": info.get("model"),
        "checkpoint": info.get("training_steps"),
        "threshold": info.get("no_answer_threshold"),
        "sample_size": total,
        "previous_ids_excluded": len(used_ids),
        "overall": calculate_metrics(results),
        "answerable": calculate_metrics(answerable_rows),
        "impossible": calculate_metrics(impossible_rows),
        "api_errors": len(failures),
        "seconds": round(
            time.time() - start_time,
            1
        )
    }

    report_file = OUT / f"report_{total}.json"

    report_file.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 48)
    print("RESULTADOS FINALES - PIMIENTERO AI")
    print("=" * 48)

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        )
    )

    print("\nReporte guardado en:", report_file)


if __name__ == "__main__":
    main()
