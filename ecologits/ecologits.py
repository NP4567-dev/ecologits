import importlib.util
import logging

from packaging.version import Version


class EcoLogits:
    """
    EcoLogits instrumentor to initialize function patching for each provider.

    By default, the initialization will be done on all available and compatible providers that are supported by the
    library.

    Examples:
        EcoLogits initialization example with OpenAI.
        ```python
        from ecologits import EcoLogits
        from openai import OpenAI

        EcoLogits.init()

        client = OpenAI(api_key="<OPENAI_API_KEY>")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Tell me a funny joke!"}
            ]
        )

        # Get estimated environmental impacts of the inference
        print(f"Energy consumption: {response.impacts.energy.value} kWh")
        print(f"GHG emissions: {response.impacts.gwp.value} kgCO2eq")
        ```

    """

    initialized = False

    @staticmethod
    def init(publishers: list[str] = []) -> None:
        """Initialization static method."""
        if not EcoLogits.initialized:
            init_instruments(publishers)
            EcoLogits.initialized = True


def init_instruments(publishers: list[str] | str | None = None) -> None:
    all_publishers = {
        "openai": init_openai_instrumentor,
        "anthropic": init_anthropic_instrumentor,
        "mistralai": init_mistralai_instrumentor,
        "huggingface": init_huggingface_instrumentor,
        "cohere": init_cohere_instrumentor,
        "google": init_google_instrumentor,
        "litellm": init_litellm_instrumentor,
    }
    wrong_publishers = []

    if isinstance(publishers, str):
        publishers = [publishers]

    if publishers is None:
        selected_instrumentors = [instrumentor_init for publisher, instrumentor_init in all_publishers.items()]
    else:
        selected_instrumentors = {publisher: all_publishers.get(publisher) for publisher in publishers}
        wrong_publishers = [
            publisher for publisher, instrumentor_init in selected_instrumentors.items() if instrumentor_init is None
        ]
        if len(wrong_publishers) > 0:
            logging.warning(f"The following publishers were not found: {','.join(wrong_publishers)}")

    if len(selected_instrumentors) > len(wrong_publishers):
        [
            instrumentor_init()
            for _, instrumentor_init in selected_instrumentors.items()
            if instrumentor_init is not None
        ]


def init_openai_instrumentor() -> None:
    if importlib.util.find_spec("openai") is not None:
        from ecologits.tracers.openai_tracer import OpenAIInstrumentor

        instrumentor = OpenAIInstrumentor()
        instrumentor.instrument()


def init_anthropic_instrumentor() -> None:
    if importlib.util.find_spec("anthropic") is not None:
        from ecologits.tracers.anthropic_tracer import AnthropicInstrumentor

        instrumentor = AnthropicInstrumentor()
        instrumentor.instrument()


def init_mistralai_instrumentor() -> None:
    if importlib.util.find_spec("mistralai") is not None:
        from ecologits.tracers.mistralai_tracer import MistralAIInstrumentor

        instrumentor = MistralAIInstrumentor()
        instrumentor.instrument()


def init_huggingface_instrumentor() -> None:
    if importlib.util.find_spec("huggingface_hub") is not None:
        from huggingface_hub import __version__

        if Version(__version__) >= Version("0.22.0"):
            from ecologits.tracers.huggingface_tracer import HuggingfaceInstrumentor

            instrumentor = HuggingfaceInstrumentor()
            instrumentor.instrument()


def init_cohere_instrumentor() -> None:
    if importlib.util.find_spec("cohere") is not None:
        from ecologits.tracers.cohere_tracer import CohereInstrumentor

        instrumentor = CohereInstrumentor()
        instrumentor.instrument()


def init_google_instrumentor() -> None:
    if importlib.util.find_spec("google-generativeai") is not None:
        from ecologits.tracers.google_tracer import GoogleInstrumentor

        instrumentor = GoogleInstrumentor()
        instrumentor.instrument()


def init_litellm_instrumentor() -> None:
    if importlib.util.find_spec("litellm") is not None:
        from ecologits.tracers.litellm_tracer import LiteLLMInstrumentor

        instrumentor = LiteLLMInstrumentor()
        instrumentor.instrument()
