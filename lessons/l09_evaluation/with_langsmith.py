from datetime import UTC, datetime
from uuid import uuid4

from langsmith import Client
from langsmith.schemas import Example

from lessons.l09_evaluation.problem import correct, target

client = Client(api_key="offline")


def run(dataset: list[dict]) -> dict:
    now = datetime.now(UTC)
    dataset_id = uuid4()
    examples = [Example(id=uuid4(), dataset_id=dataset_id, created_at=now, modified_at=now, **row) for row in dataset]
    rows = client.evaluate(target, data=examples, evaluators=[correct], upload_results=False)
    by_example = {}
    for result in rows:
        feedback = result["evaluation_results"]["results"]
        by_example[result["example"].id] = feedback[0].score if feedback else None
    scores = [by_example[example.id] for example in examples]
    graded = [score for score in scores if score is not None]
    return {"scores": scores, "mean": sum(graded) / len(graded)}
