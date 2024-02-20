# The fore client package

The foresight library within fore SDK allows you to easily evaluate the
performance of your LLM system based on a variety of metrics.

You can sign-up as a beta tester at https://foreai.co.

## Quick start
1.  Install the package using `pip`:
    ```bash
    pip install fore
    ```
    Or download the repo from [GitHub](https://github.com/foreai-co/fore/) and install via `pip install .`
2. Get started with the following lines:
    ```python
    from fore.foresight import EvalRunConfig, Foresight, InferenceOutput, MetricType

    foresight = Foresight(api_token="<YOUR_API_TOKEN>")

    evalset = foresight.create_simple_evalset(
        evalset_id="programming-languages",
        queries=["hardest programming language?", "easiest programming language?"],
        reference_answers=["Malbolge", "Python"])

    run_config = EvalRunConfig(evalset_id="programming-languages",
                            experiment_id="my-smart-llm",
                            metrics=[MetricType.GROUNDEDNESS, MetricType.SIMILARITY])


    def my_generate_fn(query: str) -> InferenceOutput:
        # Do the LLM processing with your model...
        # Here is some demo code:
        return InferenceOutput(
            generated_response="Malbolge" if "hardest" in query else "Python",
            contexts=[
                "Malbolge is the hardest language", "Python is the easiest language"
            ])

    foresight.generate_answers_and_run_eval(my_generate_fn, run_config)
    ```

## Metrics

### Groundedness
Depends on:
- LLM's generated response;
- Context used for generating the answer.

The metric answers the question: **Is the response based on the context and 
nothing else?**

This metric estimates the fraction of facts in the generated response that can 
be found in the provided context.

Example:
- **Context**: *The front door code has been changed from 1234 to 7945 due to 
security reasons.*
- **Q**: *What is the current front door code?*
- **A1**: *7945.* `[groundedness score = 0.9]`
- **A2**: *0000.* `[groundedness score = 0.0]`
- **A3**: *1234.* `[groundedness score = 0.04]`

### Similarity
Depends on:
- LLM's generated response;
- A reference response to compare the generated response with.

The metric answers the question: **Is the generated response semantically equivalent 
to the reference response?**

Example:
- **Question**: *Is Python an easy programming language to learn?*
- **Reference response**: *Python is an easy programming language to learn*
- **Response 1**: *It is easy to be proficient in python*  `[similarity score = 0.72]`
- **Response 2**: *Python is widely recognized for its simplicity.* `[similarity score = 0.59]`
- **Response 3**: *Python is not an easy programming language to learn* `[similarity score = 0.0]`

### Relevance (coming soon)
Depends on:
- LLM's generated response;
- User query/question.

The metric answers the question: **Does the response answer the question and 
only the question?**

This metric checks that the answer given by the LLM is trying to answer the 
given question precisely and does not include irrelevant information.

Example:
- **Q**: *At which temperature does oxygen boil?*
- **A1**: *Oxygen boils at -183 °C.* `[relevance score = 1.0]`
- **A2**: *Oxygen boils at -183 °C and freezes at -219 °C.* `[relevance score = 0.5]`

### Completeness (coming soon)
Depends on:
- LLM's generated response;
- User query/question.

The metric answers the question: **Are all aspects of the question answered?**

Example:
- **Q**: *At which temperature does oxygen boil and freeze?*
- **A1**: *Oxygen boils at -183 °C.* `[completeness score = 0.5]`
- **A2**: *Oxygen boils at -183 °C and freezes at -219 °C.* `[completeness score = 1.0]`