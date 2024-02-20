"""Tests for the client class."""
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pandas as pd

from fore.foresight.client import Foresight
from fore.foresight.schema import (EvalRunConfig, EvalRunDetails, EvalsetMetadata,
                                   InferenceOutput, MetricType)

TEST_TOKEN = "VERY_SECRET_TOKEN"
TEST_URL = "http://foresight:8010"
TEST_TIMEOUT = 1


class TestForeSight(unittest.TestCase):
    """Tests for the client class."""

    def setUp(self):
        self.client = Foresight(api_token=TEST_TOKEN, api_url=TEST_URL)
        self.client.timeout_seconds = TEST_TIMEOUT

    @patch("requests.request")
    def test_create_simple_evalset(self, mock_request):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "evalset_id": "my_evalset",
            "num_entries": 2
        }

        mock_request.return_value = mock_response

        result = self.client.create_simple_evalset("my_evalset",
                                                   ["query1", "query2"])

        mock_request.assert_called_with(
            method="post",
            url=f"{TEST_URL}/api/eval/set",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            params=None,
            json={
                "evalset_id":
                    "my_evalset",
                "evalset_entries": [{
                    "query": "query1",
                    "reference_answer": None
                }, {
                    "query": "query2",
                    "reference_answer": None
                }]
            },
            timeout=TEST_TIMEOUT)

        assert isinstance(result, EvalsetMetadata)

    @patch("requests.request")
    def test_create_simple_evalset_with_references(self, mock_request):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "evalset_id": "my_evalset",
            "num_entries": 2
        }

        mock_request.return_value = mock_response

        result = self.client.create_simple_evalset("my_evalset",
                                                   ["query1", "query2"],
                                                   ["reference1", "reference2"])

        mock_request.assert_called_with(
            method="post",
            url=f"{TEST_URL}/api/eval/set",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            params=None,
            json={
                "evalset_id":
                    "my_evalset",
                "evalset_entries": [{
                    "query": "query1",
                    "reference_answer": "reference1"
                }, {
                    "query": "query2",
                    "reference_answer": "reference2"
                }]
            },
            timeout=TEST_TIMEOUT)

        assert isinstance(result, EvalsetMetadata)

    @patch("requests.request")
    def test_create_simple_evalset_not_enough_references(self, mock_request):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "evalset_id": "my_evalset",
            "num_entries": 2
        }

        mock_request.return_value = mock_response

        with self.assertRaises(ValueError) as context:
            _ = self.client.create_simple_evalset("my_evalset",
                                                  ["query1", "query2"],
                                                  ["reference1"])
            self.assertTrue("Number of queries and references must match." in
                            context.exception)

    @patch("requests.request")
    def test_get_evalset(self, mock_request):
        response_mock = MagicMock()
        response_mock.json.return_value = {
            "evalset_id":
                "my_evalset",
            "entries": [{
                "entry": {
                    "query": "query1",
                    "reference_answer": None
                },
                "entry_id": "id1",
                "creation_time": "2024-01-01T00:00:00.000Z"
            }]
        }
        mock_request.return_value = response_mock

        self.client.get_evalset("my_evalset")

        mock_request.assert_called_with(
            method="get",
            url=f"{TEST_URL}/api/eval/set",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            params={"evalset_id": "my_evalset"},
            json=None,
            timeout=TEST_TIMEOUT)

    @patch("requests.request")
    def test_get_evalrun_queries(self, mock_request):
        self.client.get_evalrun_queries("my_experiment")

        mock_request.assert_called_with(
            method="get",
            url=f"{TEST_URL}/api/eval/run/queries",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            params={"experiment_id": "my_experiment"},
            json=None,
            timeout=TEST_TIMEOUT)

    @patch("requests.request")
    def test_create_evalrun(self, mock_request):
        self.client.create_evalrun(
            EvalRunConfig(experiment_id="my_experiment",
                          evalset_id="my_evalset",
                          metrics=[
                              MetricType.RELEVANCE,
                              MetricType.COMPLETENESS,
                              MetricType.GROUNDEDNESS,
                          ],
                          metadata={"my_key": "my_value"}))

        mock_request.assert_called_with(
            method="post",
            url=f"{TEST_URL}/api/eval/run",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            params=None,
            json={
                "experiment_id": "my_experiment",
                "evalset_id": "my_evalset",
                "metrics": [
                    "RELEVANCE",
                    "COMPLETENESS",
                    "GROUNDEDNESS",
                ],
                "metadata": {
                    "my_key": "my_value"
                }
            },
            timeout=TEST_TIMEOUT)

    @staticmethod
    def mock_get_evalrun_queries(experiment_id: str) -> Dict[str, str]:
        del experiment_id
        return {"entry_id1": "query1", "entry_id2": "query2"}

    @patch("fore.foresight.client.Foresight.get_evalrun_queries",
           mock_get_evalrun_queries)
    @patch("requests.request")
    def test_generate_answers_and_run_eval(self, mock_request):

        def generate_fn(query: str) -> InferenceOutput:
            return InferenceOutput(generated_response=query.upper(),
                                   contexts=[])

        self.client.generate_answers_and_run_eval(
            generate_fn=generate_fn,
            run_config=EvalRunConfig(experiment_id="my_experiment",
                                     evalset_id="my_evalset",
                                     metrics=[
                                         MetricType.RELEVANCE,
                                         MetricType.COMPLETENESS,
                                         MetricType.GROUNDEDNESS,
                                     ],
                                     metadata={"my_key": "my_value"}))

        mock_request.assert_called_with(
            method="put",
            url=f"{TEST_URL}/api/eval/run/entries",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            params=None,
            json={
                "experiment_id": "my_experiment",
                "entry_id_to_inference_output": {
                    "entry_id1": {
                        "generated_response": "QUERY1",
                        "contexts": [],
                        "debug_info": None,
                        "source_docids": None,
                    },
                    "entry_id2": {
                        "generated_response": "QUERY2",
                        "contexts": [],
                        "debug_info": None,
                        "source_docids": None,
                    }
                }
            },
            timeout=TEST_TIMEOUT)

    @staticmethod
    def get_evalrun_details_sample(
            metrics_are_computed: bool = True) -> Dict[str, Any]:
        return {
            "experiment_id":
                "my-smart-llm",
            "entries": [{
                "input": {
                    "query": "who is the king of the world",
                    "reference_answer": "a man named Bob"
                },
                "output": {
                    "generated_response": "Bob",
                    "source_docids": None,
                    "contexts": [
                        "Alice is the queen of the world",
                        "Bob is the king of the world"
                    ],
                    "debug_info": None
                },
                "metric_values": {
                    MetricType.GROUNDEDNESS: 0.98,
                    MetricType.SIMILARITY: 0.8,
                } if metrics_are_computed else {}
            }]
        }

    @patch("requests.request")
    def test_get_evalrun_details(self, mock_request):
        # Test the case when get evalrun computed the metrics.
        response_mock = MagicMock()
        response_mock.json.return_value = self.get_evalrun_details_sample()
        mock_request.return_value = response_mock

        dataframe = self.client.get_evalrun_details("my-smart-llm",
                                                    sort_by="input.query",
                                                    limit=100)

        mock_request.assert_called_with(
            method="get",
            url=f"{TEST_URL}/api/eval/run/details",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            params={
                "experiment_id": "my-smart-llm",
                "sort_field_name": "input.query",
                "limit": "100"
            },
            json=None,
            timeout=TEST_TIMEOUT)

        assert isinstance(dataframe, pd.DataFrame)
        assert dataframe.to_dict() == {
            "query": {
                0: "who is the king of the world"
            },
            "reference_answer": {
                0: "a man named Bob"
            },
            "generated_answer": {
                0: "Bob"
            },
            "source_docids": {
                0: None
            },
            "contexts": {
                0: [
                    "Alice is the queen of the world",
                    "Bob is the king of the world"
                ]
            },
            "groundedness": {
                0: 0.98
            },
            "similarity": {
                0: 0.8
            }
        }

        # Test the case when get evalrun did not compute the metrics.
        response_mock = MagicMock()
        response_mock.json.return_value = self.get_evalrun_details_sample(
            metrics_are_computed=False)
        mock_request.return_value = response_mock

        dataframe = self.client.get_evalrun_details("my-smart-llm",
                                                    sort_by="input.query",
                                                    limit=100)

        mock_request.assert_called_with(
            method="get",
            url=f"{TEST_URL}/api/eval/run/details",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            params={
                "experiment_id": "my-smart-llm",
                "sort_field_name": "input.query",
                "limit": "100"
            },
            json=None,
            timeout=TEST_TIMEOUT)

        assert isinstance(dataframe, pd.DataFrame)
        assert dataframe.to_dict() == {
            "query": {
                0: "who is the king of the world"
            },
            "reference_answer": {
                0: "a man named Bob"
            },
            "generated_answer": {
                0: "Bob"
            },
            "source_docids": {
                0: None
            },
            "contexts": {
                0: [
                    "Alice is the queen of the world",
                    "Bob is the king of the world"
                ]
            },
            "groundedness": {
                0: None
            },
            "similarity": {
                0: None
            }
        }

    @patch("importlib.util.find_spec")
    @patch("requests.request")
    def test_get_evalrun_details_no_pandas(self, mock_request, mock_find_spec):
        # Simulates the case where pandas is not installed.
        mock_find_spec.return_value = None

        response_mock = MagicMock()
        response_mock.json.return_value = self.get_evalrun_details_sample()
        mock_request.return_value = response_mock

        dataframe = self.client.get_evalrun_details("my-smart-llm",
                                                    sort_by="input.query",
                                                    limit=100)

        assert isinstance(dataframe, EvalRunDetails)


if __name__ == "__main__":
    unittest.main()
