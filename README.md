# The fore client package

The foresight library within fore SDK allows you to easily evaluate the
performance of your LLM system based on a variety of metrics.

You can sign-up and get started immediately at https://foresight.foreai.co.

Check our documentation at https://docs.foreai.co.

## Quick start
1.  Install the package using `pip`:
    ```bash
    pip install fore
    ```
    Or download the repo from [GitHub](https://github.com/foreai-co/fore/) and install via `pip install .`

2.
    - Get started with the following lines:
    ```python
    from fore.foresight import Foresight

    foresight = Foresight(api_token="<YOUR_API_TOKEN>")

    foresight.log(query="What is the easiest programming language?",
                  response="Python",
                  contexts=["Python rated the easiest programming language"],
                  tag="my_awesome_experiment")
    
    # You can add more such queries using foresight.log
    # ....

    foresight.flush()
    ```

    - Or alternatively to curate your evalsets and run regular evals against them do:
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

We currently offer two metrics:
* [Groundedness](https://docs.foreai.co/docs/foresight/metrics/groundedness)
* [Ground Truth Similarty](https://docs.foreai.co/docs/foresight/metrics/ground_truth_similarity)

Check [here](https://docs.foreai.co/docs/category/metrics) for more information.
