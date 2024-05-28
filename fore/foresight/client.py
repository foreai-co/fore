"""The main client class for the foresight API."""
import importlib.util
import logging
import uuid
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Union

import requests
from requests import Response

from fore.foresight.schema import (CreateEvalsetRequest, EvalRunConfig,
                                   EvalRunDetails, EvalRunEntry, EvalsetEntry,
                                   EvalsetMetadata, InferenceOutput, LogRequest,
                                   LogTuple, MetricType,
                                   UploadInferenceOutputsRequest)
from fore.foresight.utils import convert_to_pandas_dataframe

GenerateFnT = Callable[[str], InferenceOutput]

GATEWAY_URL = "https://foresight-gateway.foreai.co"
UI_URL = "https://foresight.foreai.co"
MAX_ENTRIES_BEFORE_FLUSH = 10
DEFAULT_TAG_NAME = "default"


class Foresight:
    """The main client class for the foresight API."""

    def __init__(self,
                 api_token: str,
                 api_url: str = GATEWAY_URL,
                 ui_url: str = UI_URL,
                 max_entries_before_auto_flush: int = MAX_ENTRIES_BEFORE_FLUSH,
                 log_level: int = logging.INFO):
        self.api_token = api_token
        self.api_url = api_url
        self.ui_url = ui_url
        self.max_entries_before_auto_flush = max_entries_before_auto_flush

        self.timeout_seconds = 60
        self.tag_to_log_entries = defaultdict(list)
        logging.basicConfig(format="foresight %(levelname)s: %(message)s",
                            level=log_level)
        logging.info("Foresight client initialized")

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

        if response.status_code != 200:
            logging.error(response.json())

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
            new_entry = EvalsetEntry(query=query, entry_id=str(uuid.uuid4()))
            if reference_answer:
                new_entry.reference_answer = reference_answer
            entries.append(new_entry)
        evalset = CreateEvalsetRequest(evalset_id=evalset_id,
                                       evalset_entries=entries)

        response = self.__make_request(method="post",
                                       endpoint="/api/eval/set",
                                       input_json=evalset.model_dump(
                                           mode="json", exclude_unset=True))

        return EvalsetMetadata(**response.json())

    def create_simple_evalrun(
            self,
            run_config: EvalRunConfig,
            queries: List[str],
            answers: List[str],
            contexts: Optional[List[List[str]]] = None,
            reference_answers: Optional[List[str]] = None) -> None:
        """Creates a simple evalset and evalrun from a list of queries, answers,
        contexts and reference_answers.

        Args:
            run_config: The configuration for running the eval.
            queries: A list of queries.
            answers: A list of generated answers.
            contexts: Optional list of contexts for each query.
                This is required for some metrics like Groundedness.
            reference_answers: Optional list of references/ground truth.
                This is required for metrics like ReferenceFactRecall.
        """
        if len(queries) != len(answers):
            raise ValueError("Number of queries and answers must match.")
        if contexts and len(queries) != len(contexts):
            raise ValueError("Number of queries and contexts must match.")

        self.create_simple_evalset(evalset_id=run_config.evalset_id,
                                   queries=queries,
                                   reference_answers=reference_answers)

        def generate_fn(query: str) -> InferenceOutput:
            idx = queries.index(query)
            return InferenceOutput(generated_response=answers[idx],
                                   contexts=contexts[idx] if contexts else [])

        self.generate_answers_and_run_eval(generate_fn=generate_fn,
                                           run_config=run_config)

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
        response = self.__make_request(method="post",
                                       endpoint="/api/eval/run",
                                       input_json=run_config.model_dump(
                                           mode="json", exclude_unset=True))

        if response.status_code == 200:
            logging.info("Eval run with experiment_id %s created.",
                         run_config.experiment_id)

        return response

    def generate_answers_and_run_eval(self,
                                      generate_fn: GenerateFnT,
                                      run_config: EvalRunConfig,
                                      batch_size=10):
        """Creates an eval run entry, generates answers and runs the eval.

        This method calls the generate_fn on each query in the evalset, triggers
        the metric computation and caches all results in a new eval run.

        Args:
            generate_fn: A function that takes a query and returns an
                InferenceOutput.
            run_config: The configuration for running the eval.
            batch_size: The max number of inference outputs to upload in one
                batch.
        """
        self.create_evalrun(run_config=run_config)
        experiment_id = run_config.experiment_id
        queries = self.get_evalrun_queries(experiment_id=experiment_id)

        if not queries:
            logging.error("No queries found for experiment_id: %s",
                          experiment_id)
            return

        outputs = {}
        for entry_id, query in queries.items():
            inference_output = generate_fn(query)
            outputs[entry_id] = inference_output

        for i in range(0, len(outputs), batch_size):
            outputs_chunk = {
                k: outputs[k] for k in list(outputs.keys())[i:i + batch_size]
            }
            output_request = UploadInferenceOutputsRequest(
                experiment_id=experiment_id,
                entry_id_to_inference_output=outputs_chunk)

            res = self.__make_request(method="put",
                                      endpoint="/api/eval/run/entries",
                                      input_json=output_request.model_dump(
                                          mode="json", exclude_unset=True))

            if res.status_code != 200:
                logging.error(
                    "Error uploading inference outputs for experiment_id: %s",
                    experiment_id)
                return

        logging.info(
            "Eval run started successfully."
            "Visit %s to view results.", self.ui_url)

    def flush(self):
        """Flush the log entries and run evals on them.
        Currently only Groundedness evals are run on the log entries.

        Returns: The HTTP response on success or raises an HTTPError on failure.
        """
        has_entries_to_flush = any(
            len(entries) > 0 for entries in self.tag_to_log_entries.values())

        if not has_entries_to_flush:
            logging.info("No log entries to flush.")
            return

        for tag, log_entries in self.tag_to_log_entries.items():
            log_request = LogRequest(log_entries=log_entries)
            if tag != DEFAULT_TAG_NAME:
                log_request.experiment_id_prefix = tag
            response = self.__make_request(method="put",
                                           endpoint="/api/eval/log",
                                           input_json=log_request.model_dump(
                                               mode="json", exclude_unset=True))

            if response.status_code == 200:
                logging.info(
                    "Log entries flushed successfully for %s tag."
                    " Visit %s to view results.", tag, self.ui_url)
                # Clear log entries after flushing
                log_entries.clear()
            else:
                logging.error(
                    "Flushing log entries failed with response code: %s",
                    response.status_code)

        return response

    def log(self,
            query: str,
            response: str,
            contexts: List[str],
            tag: Optional[str] = None) -> None:
        """Add log entries for evaluation. This only adds the entries
        in memory, but does not send any requests to foresight service.
        To send the request, flush needs to be called.

        If the number of entries is greater than
        `self.max_entries_before_auto_flush`, then flushes the log entries as
        well.

        Args:
            query: The query for evaluation.
            response: The response from your AI system.
            contexts: List of contexts relevant to the query.
            tag: An optional tag for the request. e.g. "great-model-v01".
                This will be prepended to the name of the eval run
                (experiment_id). The complete eval run experiment_id will be
                of the form: "great-model-v01_logs_groundedness_YYYYMMDD"
        """
        inference_output = InferenceOutput(generated_response=response,
                                           contexts=contexts)
        log_entry = LogTuple(query=query, inference_output=inference_output)
        tag = tag if tag else DEFAULT_TAG_NAME
        entries_for_tag = self.tag_to_log_entries[tag]
        entries_for_tag.append(log_entry)
        if len(entries_for_tag) >= self.max_entries_before_auto_flush:
            # Auto flush if the number of entries for any tag is greater than a
            # certain threshold.
            self.flush()

    def convert_evalrun_details_to_dataframe(self, details: EvalRunDetails):
        """Converts an EvalRunDetails object to a DataFrame."""
        df = {
            "query": [],
            "reference_answer": [],
            "reference_answer_facts": [],
            "generated_answer": [],
            "source_docids": [],
            "contexts": [],
        }
        eval_metrics = [m for m in MetricType if m.is_implemented()]

        for m in eval_metrics:
            df[m.value.lower()] = []

        entry: EvalRunEntry
        for entry in details.entries:
            df["query"].append(entry.input.query)
            df["reference_answer"].append(entry.input.reference_answer)
            df["reference_answer_facts"].append(
                entry.input.reference_answer_facts)
            df["generated_answer"].append(entry.output.generated_response)
            df["source_docids"].append(entry.output.source_docids)
            df["contexts"].append(entry.output.contexts)
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
        convert_to_dataframe: bool = False
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
            return self.convert_evalrun_details_to_dataframe(details)

        return details
