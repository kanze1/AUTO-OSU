"""Explicit W&B experiment logging; online failures are never relabelled offline success."""
from __future__ import annotations

import json
from pathlib import Path


def scalar_metrics(record, prefix=""):
    result = {}
    for key, value in record.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            result.update(scalar_metrics(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result[name] = value
    return result


class ExperimentTracker:
    def __init__(self, directory, project, group, name, config, mode="online", entity=None):
        self.directory = Path(directory)
        self.run = None
        if project:
            import wandb
            self.run = wandb.init(project=project, entity=entity, group=group, name=name,
                                  config=config, dir=str(directory), mode=mode, resume="never")
            self.run.define_metric("update")
            self.run.define_metric("*", step_metric="update")
            (self.directory / "wandb-run.json").write_text(json.dumps(
                dict(id=self.run.id, url=self.run.url, project=project, group=group, mode=mode), indent=2))

    def log(self, step, record):
        if self.run is not None:
            self.run.log({"update": step, **scalar_metrics(record)})

    def table(self, step, name, columns, rows):
        if self.run is not None:
            import wandb
            self.run.log({"update": step, name: wandb.Table(columns=columns, data=rows)})

    def finish(self, status="complete"):
        if self.run is not None:
            self.run.summary["execution_status"] = status
            self.run.finish(exit_code=0 if status == "complete" else 1)
