# The fore client package

The foresight library within fore SDK allows you to easily evaluate the
performance of your LLM system based on a variety of metrics.

You can sign-up and get started immediately at https://foresight.foreai.co.

Check our documentation at https://docs.foreai.co

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

### Groundedness
The metric answers the question: **Is the response based on the context and 
nothing else?**. It measures whether the response is consistent with and can be
implied from the context ("are entailed"). It doesn't need to be semantcially
equivalent to the context.

Depends on:
- LLM's generated response;
- Context used for generating the answer.

This metric score is between 0 and 1 and estimates the fraction of facts in the
candidate response that are grounded in the provided context (are entailed by
the context).

Example:
- **Context**: *The front door code has been changed from 1234 to 7945 due to 
security reasons.*
- **Question**: *What is the current front door code?*
- **Response 1**: *7945.* `[groundedness score = 1.0]`
- **Response 2**: *0000.* `[groundedness score = 0.0]`
- **Response 3**: *1234.* `[groundedness score = 0.0]`
- **Response 4**: *The code has been changed due to security reasons.* `[groundedness score = 1.0]` (100% grounded, even if not answering the question)

Example:
- **Context**: *Albert Einstein, (14 March 1879 - 18 April 1955) was a German-born theoretical physicist. In 1905, sometimes described as his annus mirabilis (miracle year), Einstein published four groundbreaking papers.*
- **Question**: *Where was Einstein born and how old did he get?*
- **Response 1**: *He was 76 years old when he died and he was born in Germany.* `[groundedness score = 1.0]`
- **Response 2**: *He was 50 years old when he died and he was born in Germany.* `[groundedness score = 0.5]` (Age is wrong, country is correct)
- **Response 3**: *He was born in Europe and was more than 60 years old when he died.* `[groundedness score = 1.0]` (Entailed in context.)
- 
### Similarity
The metric answers the question: **Is the generated response factually equivalent 
to the reference response?**

Depends on:
- A user query;
- An LLM's generated response to be evaluated;
- A reference response to compare the generated response with.

The metric score (range from 0 to 1) represents the ratio of the facts that are present in both the reference and the generated response (facts, the reference and generated responses agree on), divided by the maximum number of facts in any of the responses. See examples below for more intuition.

Example (multi-statement):
- **Question**: *What is the capital of France, and what is the primary language spoken there?*
- **Reference response**: *The capital of France is Paris, and the primary spoken language is French.*
- **Response 1**: *Paris is the capital of France, and the most spoken language is French.* `[similarity score = 1.0]`
- **Response 2**: *The capital of France is Paris.* `[similarity score = 0.5]` (the second statement is missing from the generated response)
- **Response 3**: *The capital of France is Paris, and the most spoken language is English.* `[similarity score = 0.5]` (language statement mismatch)

Example (multi-statement):
- **Question**: *What fruits does Amy like?*
- **Reference response**: *Amy likes apples and bananas.*
- **Response**: *Amy likes apples, berries and plums.* `[similarity score = 0.33]` (one matched fact and two mismatched ones)

Example:
- **Question**: *What is the age of Archie White, the oldest new graduate in Britain as of July 16, 2021?*
- **Reference response**: *96 years old.*
- **Response 1**: *Archie White's age is 96.*  `[similarity score = 1.0]`
- **Response 2**: *He is 96.* `[similarity score = 1.0]`
- **Response 3**: *He is more than 95 years old.* `[similarity score = 0.0]` (while true, it's not semantically equivalent to the reference response)

Example:
- **Question**: *What cars does Alex like?*
- **Reference response**: *Alex likes blue cars.*
- **Response**: *Alex likes all cars.* `[similarity score = 0.0]` (the statement is similar but not equivalent to the reference response)

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
