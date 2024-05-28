"""Tests for the client class."""
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, call, patch

import pandas as pd

from fore.foresight.client import Foresight
from fore.foresight.schema import (EvalRunConfig, EvalRunDetails,
                                   EvalsetMetadata, InferenceOutput, LogTuple,
                                   MetricType)

TEST_TOKEN = "VERY_SECRET_TOKEN"
TEST_URL = "http://foresight:8010"
TEST_TIMEOUT = 1
MAX_ENTRIES_BEFORE_FLUSH = 2


class TestForeSight(unittest.TestCase):
    """Tests for the client class."""

    def setUp(self):
        self.client = Foresight(
            api_token=TEST_TOKEN,
            api_url=TEST_URL,
            max_entries_before_auto_flush=MAX_ENTRIES_BEFORE_FLUSH)
        self.client.timeout_seconds = TEST_TIMEOUT

    @patch("uuid.uuid4")
    @patch("requests.request")
    def test_create_simple_evalset(self, mock_request, mock_uuid):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "evalset_id": "my_evalset",
            "num_entries": 2
        }
        mock_uuid.side_effect = ["my_uuid1", "my_uuid2"]

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
                    "entry_id": "my_uuid1",
                }, {
                    "query": "query2",
                    "entry_id": "my_uuid2",
                }]
            },
            timeout=TEST_TIMEOUT)

        assert isinstance(result, EvalsetMetadata)

    @patch("uuid.uuid4")
    @patch("requests.request")
    def test_create_simple_evalset_with_references(self, mock_request,
                                                   mock_uuid):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "evalset_id": "my_evalset",
            "num_entries": 2
        }

        mock_request.return_value = mock_response
        mock_uuid.side_effect = ["my_uuid1", "my_uuid2"]

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
                    "reference_answer": "reference1",
                    "entry_id": "my_uuid1",
                }, {
                    "query": "query2",
                    "reference_answer": "reference2",
                    "entry_id": "my_uuid2",
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

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

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
                    },
                    "entry_id2": {
                        "generated_response": "QUERY2",
                        "contexts": [],
                    }
                }
            },
            timeout=TEST_TIMEOUT)

    @patch("fore.foresight.client.Foresight.get_evalrun_queries",
           mock_get_evalrun_queries)
    @patch("uuid.uuid4")
    @patch("fore.foresight.client.requests.request")
    def test_create_simple_evalrun(self, mock_request, mock_uuid):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "evalset_id": "my_evalset",
            "num_entries": 2
        }

        mock_request.return_value = mock_response
        mock_uuid.side_effect = ["my_uuid1", "my_uuid2"]

        run_config = EvalRunConfig(experiment_id="my_experiment",
                                   evalset_id="my_evalset",
                                   metrics=[
                                       MetricType.GROUNDEDNESS,
                                   ],
                                   metadata={"my_key": "my_value"})

        self.client.create_simple_evalrun(
            run_config=run_config,
            queries=["query1", "query2"],
            answers=["answer1", "answer2"],
            contexts=[["context1", "context2"], ["context3", "context4"]],
            reference_answers=["reference1", "reference2"])

        mock_request.assert_any_call(
            method="post",
            url=f"{TEST_URL}/api/eval/set",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            params=None,
            json={
                "evalset_id":
                    "my_evalset",
                "evalset_entries": [{
                    "query": "query1",
                    "reference_answer": "reference1",
                    "entry_id": "my_uuid1",
                }, {
                    "query": "query2",
                    "reference_answer": "reference2",
                    "entry_id": "my_uuid2",
                }]
            },
            timeout=TEST_TIMEOUT)

        mock_request.assert_any_call(
            method="post",
            url=f"{TEST_URL}/api/eval/run",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            params=None,
            json={
                "experiment_id": "my_experiment",
                "evalset_id": "my_evalset",
                "metrics": ["GROUNDEDNESS",],
                "metadata": {
                    "my_key": "my_value"
                }
            },
            timeout=TEST_TIMEOUT)

        mock_request.assert_any_call(
            method="put",
            url=f"{TEST_URL}/api/eval/run/entries",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            params=None,
            json={
                "experiment_id": "my_experiment",
                "entry_id_to_inference_output": {
                    "entry_id1": {
                        "generated_response": "answer1",
                        "contexts": ["context1", "context2"]
                    },
                    "entry_id2": {
                        "generated_response": "answer2",
                        "contexts": ["context3", "context4"]
                    }
                }
            },
            timeout=TEST_TIMEOUT)

    @patch("fore.foresight.client.Foresight.get_evalrun_queries",
           mock_get_evalrun_queries)
    @patch("requests.request")
    def test_generate_answers_and_run_eval_batched(self, mock_request):

        def generate_fn(query: str) -> InferenceOutput:
            return InferenceOutput(generated_response=query.upper(),
                                   contexts=[])

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        self.client.generate_answers_and_run_eval(
            generate_fn=generate_fn,
            run_config=EvalRunConfig(experiment_id="my_experiment",
                                     evalset_id="my_evalset",
                                     metrics=[
                                         MetricType.RELEVANCE,
                                         MetricType.COMPLETENESS,
                                         MetricType.GROUNDEDNESS,
                                     ],
                                     metadata={"my_key": "my_value"}),
            batch_size=1)

        # Skip the first request that creates the evalrun.
        self.assertSequenceEqual(mock_request.call_args_list[1:], [
            call(method="put",
                 url=f"{TEST_URL}/api/eval/run/entries",
                 headers={"Authorization": f"Bearer {TEST_TOKEN}"},
                 params=None,
                 json={
                     "experiment_id": "my_experiment",
                     "entry_id_to_inference_output": {
                         "entry_id1": {
                             "generated_response": "QUERY1",
                             "contexts": [],
                         },
                     }
                 },
                 timeout=TEST_TIMEOUT),
            call(method="put",
                 url=f"{TEST_URL}/api/eval/run/entries",
                 headers={"Authorization": f"Bearer {TEST_TOKEN}"},
                 params=None,
                 json={
                     "experiment_id": "my_experiment",
                     "entry_id_to_inference_output": {
                         "entry_id2": {
                             "generated_response": "QUERY2",
                             "contexts": [],
                         }
                     }
                 },
                 timeout=TEST_TIMEOUT)
        ])

    @patch("requests.request")
    def test_log_adds_entries(self, mock_request):
        """Tests for adding log entries."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        query1 = "test_query1"
        llm_response1 = "test_response1"
        contexts1 = ["context1", "context2"]

        query2 = "test_query2"
        llm_response2 = "test_response2"
        contexts2 = ["context3", "context4"]

        query3 = "test_query3"
        llm_response3 = "test_response3"
        contexts3 = ["context5", "context6"]

        query4 = "test_query4"
        llm_response4 = "test_response4"
        contexts4 = ["context7", "context8"]

        self.client.log(query1, llm_response1, contexts1)
        self.client.log(query4, llm_response4, contexts4, tag="great_model")
        self.client.log(query2, llm_response2, contexts2)
        self.client.log(query3, llm_response3, contexts3)

        expected_calls = [
            call(method="put",
                 url=f"{TEST_URL}/api/eval/log",
                 headers={"Authorization": f"Bearer {TEST_TOKEN}"},
                 params=None,
                 json={
                     "log_entries": [{
                         "query": "test_query1",
                         "inference_output": {
                             "generated_response": "test_response1",
                             "contexts": ["context1", "context2"],
                         }
                     }, {
                         "query": "test_query2",
                         "inference_output": {
                             "generated_response": "test_response2",
                             "contexts": ["context3", "context4"],
                         }
                     }],
                 },
                 timeout=TEST_TIMEOUT),
            call(method="put",
                 url=f"{TEST_URL}/api/eval/log",
                 headers={"Authorization": f"Bearer {TEST_TOKEN}"},
                 params=None,
                 json={
                     "log_entries": [{
                         "query": "test_query4",
                         "inference_output": {
                             "generated_response": "test_response4",
                             "contexts": ["context7", "context8"],
                         }
                     },],
                     "experiment_id_prefix": "great_model"
                 },
                 timeout=TEST_TIMEOUT)
        ]

        mock_request.assert_has_calls(expected_calls, any_order=True)

        self.assertEqual(len(self.client.tag_to_log_entries), 2)
        self.assertDictEqual(
            self.client.tag_to_log_entries, {
                "default": [
                    LogTuple(query="test_query3",
                             inference_output=InferenceOutput(
                                 generated_response="test_response3",
                                 contexts=["context5", "context6"]))
                ],
                "great_model": []
            })

    @patch("requests.request")
    def test_flush_sends_request_and_clears_entries(self, mock_request):
        """Tests for flushing log entries."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        # Add some log entries and flush
        self.client.log("test_query1", "test_response1", ["context1"])
        response1 = self.client.flush()
        self.client.log("test_query2",
                        "test_response2", ["context2"],
                        tag="great_model")
        response2 = self.client.flush()
        expected_calls = [
            call(method="put",
                 url=f"{TEST_URL}/api/eval/log",
                 headers={"Authorization": f"Bearer {TEST_TOKEN}"},
                 params=None,
                 json={
                     "log_entries": [{
                         "query": "test_query1",
                         "inference_output": {
                             "generated_response": "test_response1",
                             "contexts": ["context1"],
                         }
                     }],
                 },
                 timeout=TEST_TIMEOUT),
            call(method="put",
                 url=f"{TEST_URL}/api/eval/log",
                 headers={"Authorization": f"Bearer {TEST_TOKEN}"},
                 params=None,
                 json={
                     "log_entries": [{
                         "query": "test_query2",
                         "inference_output": {
                             "generated_response": "test_response2",
                             "contexts": ["context2"],
                         }
                     },],
                     "experiment_id_prefix": "great_model"
                 },
                 timeout=TEST_TIMEOUT)
        ]

        mock_request.assert_has_calls(expected_calls, any_order=True)

        # Assert that log_entries is empty after flushing
        self.assertDictEqual(self.client.tag_to_log_entries, {
            "default": [],
            "great_model": []
        })

        # Assert the response from flush
        self.assertEqual(response1.status_code, 200)
        self.assertEqual(response2.status_code, 200)

    @staticmethod
    def get_evalrun_details_sample(
            metrics_are_computed: bool = True) -> Dict[str, Any]:
        return {
            "experiment_id":
                "my-smart-llm",
            "entries": [{
                "input": {
                    "entry_id": "my-entry-id",
                    "query": "who is the king of the world",
                    "reference_answer": "a man named Bob",
                    "reference_answer_facts": ["a man named Bob exists."]
                },
                "output": {
                    "generated_response":
                        "Bob",
                    "contexts": [
                        "Alice is the queen of the world",
                        "Bob is the king of the world"
                    ],
                },
                "metric_values": {
                    MetricType.GROUNDEDNESS: 0.98,
                    MetricType.REFERENCE_FACT_RECALL: 0.8,
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
                                                    limit=100,
                                                    convert_to_dataframe=True)

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
            "reference_answer_facts": {
                0: ["a man named Bob exists."]
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
            "reference_fact_recall": {
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
                                                    limit=100,
                                                    convert_to_dataframe=True)

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
            "reference_answer_facts": {
                0: ["a man named Bob exists."]
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
            "reference_fact_recall": {
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
