""" Schema defining the foresight eval service APIs."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel


class EvalsetEntry(BaseModel):
    """Represents the model's input (query) for a single eval entry."""
    query: str
    reference_answer: Optional[str] = None


class EvalsetMetadata(BaseModel):
    """Metadata for an evalset."""
    evalset_id: str
    num_entries: int = 0


class EvalsetEntryWithMetadata(BaseModel):
    """EvalsetEntry with additional metadata."""
    entry: EvalsetEntry
    entry_id: str
    # The UTC datetime when the eval entry was created.
    creation_time: Optional[datetime] = None


class Evalset(BaseModel):
    "A stored evalset."
    evalset_id: str
    entries: List[EvalsetEntryWithMetadata]


class InferenceOutput(BaseModel):
    """Holds the output of a single model inference run."""
    generated_response: str
    # The ids of the documents that were used for generating the answer.
    source_docids: Optional[List[str]] = None
    # The text contexts that were used for generating the answer.
    # This is needed for metrics like "GROUNDEDNESS".
    contexts: List[str]
    # json serializable
    debug_info: Optional[dict] = None


class MetricType(str, Enum):
    """Different metrics supported by the foresight eval service.
    
    For details, see the following
    [doc](https://github.com/foreai-co/foresight/blob/main/README.md#metrics). 
    """

    # Does the response answer the query?
    RELEVANCE = "RELEVANCE"
    # Are all aspects of the user query answered?
    COMPLETENESS = "COMPLETENESS"
    # Is the response based on a provided context?
    GROUNDEDNESS = "GROUNDEDNESS"
    # How similar is the response to the reference answer?
    SIMILARITY = "SIMILARITY"


class EvalRunConfig(BaseModel):
    """Configuration for running an eval for a given evalset."""

    # e.g. "great-model-v01"
    experiment_id: str
    # needs to match the name of an existing evalset.
    evalset_id: str
    # list of metrics to compute for this eval run.
    metrics: List[MetricType]
    # json serializable dict with extra metadata associated with this eval run.
    metadata: Optional[Dict[str, str]] = None


class MetricOutput(BaseModel):
    """Output of a single metric."""

    metric_type: MetricType
    value: float


class GetEvalsetRequest(BaseModel):
    evalset_id: str


class CreateEvalsetRequest(BaseModel):
    evalset_id: str
    evalset_entries: List[EvalsetEntry]


class UploadInferenceOutputsRequest(BaseModel):
    experiment_id: str
    entry_id_to_inference_output: Dict[str, InferenceOutput]


class EvalRunEntry(BaseModel):
    input: EvalsetEntry
    # We only consider entries for which inference output has been uploaded.
    output: InferenceOutput
    # Empty in case no metrics have been computed yet.
    metric_values: Optional[Dict[MetricType, float]] = None


class EvalRunDetails(BaseModel):
    experiment_id: str
    entries: List[EvalRunEntry]


class EvalRunSummary(BaseModel):
    """Aggregate metrics and progress stats for an eval run."""
    experiment_id: str
    # Total number of eval entries.
    num_entries: int
    # The UTC datetime when the eval run was created.
    creation_time: Optional[datetime] = None
    # Number of eval entries for which inference output has been uploaded.
    # Range: [0, num_entries].
    num_entries_with_output: int = 0
    # Range: [0, len(metrics) * num_entries].
    num_entries_with_metrics: int = 0
    # Average metric values based on the metric values that were already
    # computed. This will periodically keep updating until all metrics for
    # all entries have been computed.
    metric_values: Optional[Dict[MetricType, float]] = None
