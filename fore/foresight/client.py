"""The main client class for the foresight API."""
import importlib.util
import logging
from typing import Callable, Dict, List, Optional, Union

import requests
from requests import Response

from fore.foresight.schema import (CreateEvalsetRequest, EvalRunConfig,
                                   EvalRunDetails, EvalRunEntry, EvalsetEntry,
                                   EvalsetMetadata, InferenceOutput, MetricType,
                                   UploadInferenceOutputsRequest)
from fore.foresight.utils import convert_to_pandas_dataframe

GenerateFnT = Callable[[str], InferenceOutput]

GATEWAY_URL = "https://foresight-gateway.foreai.co"
logging.basicConfig(format="foresight %(levelname)s: %(message)s",
                    level=logging.WARNING)


class Foresight:
    """The main client class for the foresight API."""

    def __init__(self, api_token: str, api_url: str = GATEWAY_URL):
        self.api_token = api_token
        self.api_url = api_url

        self.timeout_seconds = 60

    def __make_request(self,
                       method: str,
                       endpoint: str,
                       params: Optional[dict] = None,
                       input_json: Optional[dict] = None) -> Response:
        """Makes an HTTP request to the API."""

        response = requests.request(
            method=method,
            url=f"{self.api_url}{endpoint}",
            headers={"Authorization": f"Bearer {self.api_token}"},
            params=params,
            json=input_json,
            timeout=self.timeout_seconds)

        response.raise_for_status()

        return response

    def create_simple_evalset(
            self,
            evalset_id: str,
            queries: List[str],
            reference_answers: Optional[List[str]] = None) -> EvalsetMetadata:
        """Creates a simple evalset from a list of queries and references.

        Args:
            evalset_id: String identifier of the evaluation set.
            queries: A list of queries.
            reference_answers: Optional list of references/ground truth.

        Returns: an EvalsetMetadata object or raises an HTTPError on failure.
        """
        if reference_answers and len(queries) != len(reference_answers):
            raise ValueError("Number of queries and references must match.")
        entries = []

        for i, query in enumerate(queries):
            reference_answer = None
            if reference_answers:
                reference_answer = reference_answers[i]
            entries.append(
                EvalsetEntry(query=query, reference_answer=reference_answer))
        evalset = CreateEvalsetRequest(evalset_id=evalset_id,
                                       evalset_entries=entries)

        response = self.__make_request(
            method="post",
            endpoint="/api/eval/set",
            input_json=evalset.model_dump(mode="json"))

        return EvalsetMetadata(**response.json())

    def get_evalset(self, evalset_id: str) -> EvalsetMetadata:
        """Gets the evaluation set with metadata.

        Args:
            evalset_id: String identifier of the evaluation set.

        Returns: an Evalset object or raises an HTTPError on failure.
        """
        response = self.__make_request(method="get",
                                       endpoint="/api/eval/set",
                                       params={"evalset_id": str(evalset_id)})

        return EvalsetMetadata(**response.json())

    def get_evalrun_queries(self, experiment_id: str) -> Dict[str, str]:
        """Gets the queries associated with an eval run.

        Args:
            experiment_id: String identifier of the evaluation run.

        Returns: a dictionary with (entry_id, query) pairs, or raises an 
        HTTPError on failure.
        """
        response = self.__make_request(
            method="get",
            endpoint="/api/eval/run/queries",
            params={"experiment_id": str(experiment_id)})

        return response.json()

    def create_evalrun(self, run_config: EvalRunConfig) -> Response:
        """Creates an evaluation run.

        Args:
            run_config: The configuration for running the eval.

        Returns: the HTTP response on success or raises an HTTPError on failure.
        """
        return self.__make_request(
            method="post",
            endpoint="/api/eval/run",
            input_json=run_config.model_dump(mode="json"))

    def generate_answers_and_run_eval(self, generate_fn: GenerateFnT,
                                      run_config: EvalRunConfig) -> Response:
        """Creates an eval run entry, generates answers and runs the eval.

        This method calls the generate_fn on each query in the evalset, triggers
        the metric computation and caches all results in a new eval run.

        Args:
            generate_fn: A function that takes a query and returns an
                InferenceOutput.
            run_config: The configuration for running the eval.

        Returns: the HTTP response on success or raises an HTTPError on failure.
        """
        self.create_evalrun(run_config=run_config)
        experiment_id = run_config.experiment_id
        queries = self.get_evalrun_queries(experiment_id=experiment_id)

        outputs = {}
        for entry_id, query in queries.items():
            inference_output = generate_fn(query)
            outputs[entry_id] = inference_output

        outputs = UploadInferenceOutputsRequest(
            experiment_id=experiment_id, entry_id_to_inference_output=outputs)

        return self.__make_request(method="put",
                                   endpoint="/api/eval/run/entries",
                                   input_json=outputs.model_dump(mode="json"))

    def __convert_evalrun_details_to_dataframe(self, details: EvalRunDetails):
        """Converts an EvalRunDetails object to a DataFrame."""
        df = {
            "query": [],
            "reference_answer": [],
            "generated_answer": [],
            "source_docids": [],
            "contexts": [],
        }
        # TODO: use this line when we implement all metrics.
        # eval_metrics = [m for m in MetricType]
        eval_metrics = [MetricType.GROUNDEDNESS, MetricType.SIMILARITY]

        for m in eval_metrics:
            df[m.value.lower()] = []

        entry: EvalRunEntry
        for entry in details.entries:
            df["query"].append(entry.input.query)
            df["reference_answer"].append(entry.input.reference_answer)
            df["generated_answer"].append(entry.output.generated_response)
            df["source_docids"].append(entry.output.source_docids)
            df["contexts"].append(entry.output.contexts)
            # TODO: once we implement batching / parallel processing,
            # make an update here to handle the case of entries with not
            # yet computed metrics.
            for m in eval_metrics:
                if m in entry.metric_values:
                    df[m.value.lower()].append(entry.metric_values[m])
                else:
                    df[m.value.lower()].append(None)
        return convert_to_pandas_dataframe(df)

    def get_evalrun_details(
        self,
        experiment_id: str,
        sort_by: Optional[str] = "input.query",
        limit: Optional[int] = 100,
        convert_to_dataframe: bool = True
    ) -> Union[EvalRunDetails, "pandas.DataFrame"]:
        """Gets the details of an evaluation run.

        Args:
            experiment_id: String identifier of the evaluation run.
            sort_by: The field to sort by.
            limit: The maximum number of entries to return.
            convert_to_dataframe: If True, returns a DataFrame instead of a
                EvalRunDetails object. Requires pandas to be installed.

        Returns: an EvalRunDetails object or raises an HTTPError on failure.
        If pandas is installed and convert_to_dataframe is set to True,
        the results are converted to a DataFrame.
        """
        params = {"experiment_id": str(experiment_id)}
        if limit is not None or sort_by is not None:
            assert limit is not None and sort_by is not None, (
                "Both limit and sort_by must be provided if either is provided."
            )
            params["sort_field_name"] = sort_by
            params["limit"] = str(limit)
        response = self.__make_request(method="get",
                                       endpoint="/api/eval/run/details",
                                       params=params)
        details = EvalRunDetails(**response.json())

        if convert_to_dataframe:
            if importlib.util.find_spec("pandas") is None:
                logging.warning("pandas is not installed. "
                                "Returning an EvalRunDetails object instead.")
                return details

            # Build a DataFrame from the response.
            return self.__convert_evalrun_details_to_dataframe(details)

        return details
