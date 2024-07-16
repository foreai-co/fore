"""The main client for the Critical Journeys API."""
import base64
import logging
import os
import re
from time import sleep
from typing import List, Optional, Union

import requests
from requests import Response

from fore.cj.schema import State, TestCase, TestCaseRequest, TestGenerationStep

GATEWAY_URL = "https://cj.foreai.co"


class CriticalJourneysClient:
    """The main client for the Critical Journeys API."""

    def __init__(self,
                 api_url: str = GATEWAY_URL,
                 log_level: int = logging.INFO):
        """Initialize the client with an API key."""
        self.test_case_ids = []
        self.api_url = api_url

        self.timeout_seconds = 60
        logging.basicConfig(
            format="cj %(levelname)s: %(message)s",
            level=log_level)
        logging.info("CJ client initialized")

    def __make_request(self,
                       method: str,
                       endpoint: str,
                       params: Optional[dict] = None,
                       input_json: Optional[dict] = None) -> Response:
        """Makes an HTTP request to the API."""

        response = requests.request(
            method=method,
            url=f"{self.api_url}{endpoint}",
            params=params,
            json=input_json,
            timeout=self.timeout_seconds)

        if response.status_code // 100 != 2:
            logging.error(response.json())

        response.raise_for_status()

        return response

    def submit_test_case(
            self,
            test_case_request: Union[dict, TestCaseRequest]) -> str:
        """Sends a test case to the server."""
        if isinstance(test_case_request, dict):
            test_case_request = TestCaseRequest.model_validate(
                test_case_request)

        response = self.__make_request(
            method="POST",
            endpoint="/test-case",
            input_json=test_case_request.model_dump(mode="json",
                                                    exclude_unset=True))

        test_case_id = re.findall(r'"([a-f0-9]+)"', response.text)[0]
        self.test_case_ids.append(test_case_id)
        logging.info("Submitted test case with ID: %s", test_case_id)

        return test_case_id

    def get_test_case(self, test_case_id: str) -> TestCase:
        """Retrieves a test case from the server."""
        response = self.__make_request(
            method="GET",
            endpoint=f"/test-case/{test_case_id}")

        return TestCase.model_validate(response.json())

    def get_my_test_case_ids(self) -> List[str]:
        """Returns the list of test case IDs."""
        return self.test_case_ids

    def watch_test_case_and_save_new_results(
        self,
        test_case_id: str,
        save_dir: str,
    ) -> str:
        """Polls the server for the status of a test case.

        Returns the final script, if it was generated.
        """
        save_dir = os.path.expanduser(save_dir)

        screenshot_folder = os.path.join(save_dir, "screenshots/")
        os.makedirs(screenshot_folder, exist_ok=True)
        scripts_folder = os.path.join(save_dir, "scripts/")
        os.makedirs(scripts_folder, exist_ok=True)
        planner_messages_folder = os.path.join(save_dir, "planner_messages/")
        os.makedirs(planner_messages_folder, exist_ok=True)

        test_case_finished = False
        generation_steps_saved = 0

        def save_generation_step(step: TestGenerationStep, current_step: int):
            if step.screenshot:
                screenshot_path = os.path.join(screenshot_folder,
                                               f"step_{current_step}.png")
                with open(screenshot_path, "wb") as f:
                    f.write(base64.b64decode(step.screenshot))

            if step.generated_code:
                script_path = os.path.join(scripts_folder,
                                           f"step_{current_step}.py")
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(step.generated_code)

            planner_message_path = os.path.join(planner_messages_folder,
                                                f"step_{current_step}.md")
            with open(planner_message_path, "w", encoding="utf-8") as f:
                f.write(step.planner_message)

        def get_file_extension_from_language(language: str) -> str:
            if language == "python":
                return ".py"
            if language == "typescript":
                return ".ts"
            if language == "javascript":
                return ".js"
            return ".txt"

        logging.info("Listening for test case updates...")
        while not test_case_finished:
            test_case = self.get_test_case(test_case_id)

            while len(test_case.test_generation_steps) > generation_steps_saved:
                logging.info("Saving step %d.", generation_steps_saved)
                save_generation_step(
                    test_case.test_generation_steps[generation_steps_saved],
                    generation_steps_saved)
                generation_steps_saved += 1

            if (len(test_case.test_generation_steps) < generation_steps_saved
                    ) and (test_case.state == State.RETRYING):
                # Remove old generation steps if the test case is retrying
                logging.info("Test case was retried.")
                for i in range(generation_steps_saved):
                    os.remove(os.path.join(screenshot_folder, f"step_{i}.png"))
                    os.remove(os.path.join(scripts_folder, f"step_{i}.py"))
                    os.remove(os.path.join(
                        planner_messages_folder, f"step_{i}.md"))

            if test_case.state not in [State.PENDING, State.RUNNING]:
                test_case_finished = True
                logging.info("Test case %s completed.", test_case_id)
                extension = get_file_extension_from_language(
                    test_case.programming_language)
                with open(os.path.join(scripts_folder,
                                       f"final_script{extension}"),
                          "w",
                          encoding="utf-8") as f:
                    f.write(test_case.final_script)
            sleep(5)

        return test_case.final_script
