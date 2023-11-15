from unittest.mock import MagicMock

import jsonschema
import pytest


from canopy.models.data_models import Role, MessageBase # noqa
from canopy.models.api_models import ChatResponse, StreamingChatChunk # noqa
from canopy.llm.openai import OpenAILLM  # noqa
from canopy.llm.models import \
    Function, FunctionParameters, FunctionArrayProperty  # noqa
from openai import BadRequestError # noqa


def assert_chat_completion(response, num_choices=1):
    assert len(response.choices) == num_choices
    for choice in response.choices:
        assert isinstance(choice.message, MessageBase)
        assert isinstance(choice.message.content, str)
        assert len(choice.message.content) > 0
        assert isinstance(choice.message.role, Role)


def assert_function_call_format(result):
    assert isinstance(result, dict)
    assert "queries" in result
    assert isinstance(result["queries"], list)
    assert len(result["queries"]) > 0
    assert isinstance(result["queries"][0], str)
    assert len(result["queries"][0]) > 0


class TestOpenAILLM:

    @staticmethod
    @pytest.fixture
    def model_name():
        return "gpt-3.5-turbo-0613"

    @staticmethod
    @pytest.fixture
    def messages():
        # Create a list of MessageBase objects
        return [
            MessageBase(role=Role.USER, content="Hello, assistant."),
            MessageBase(role=Role.ASSISTANT,
                        content="Hello, user. How can I assist you?")
        ]

    @staticmethod
    @pytest.fixture
    def function_query_knowledgebase():
        return Function(
            name="query_knowledgebase",
            description="Query search engine for relevant information",
            parameters=FunctionParameters(
                required_properties=[
                    FunctionArrayProperty(
                        name="queries",
                        items_type="string",
                        description='List of queries to send to the search engine.',
                    ),
                ]
            ),
        )

    @staticmethod
    @pytest.fixture
    def model_params_high_temperature():
        return {"temperature": 0.9, "top_p": 0.95, "n": 3}

    @staticmethod
    @pytest.fixture
    def model_params_low_temperature():
        return {"temperature": 0.2, "top_p": 0.5, "n": 1}

    @staticmethod
    @pytest.fixture
    def openai_llm(model_name):
        return OpenAILLM(model_name=model_name)

    @staticmethod
    def test_init_with_custom_params(openai_llm):
        llm = OpenAILLM(model_name="test_model_name",
                        api_key="test_api_key",
                        organization="test_organization",
                        temperature=0.9,
                        top_p=0.95,
                        n=3,)

        assert llm.model_name == "test_model_name"
        assert llm.default_model_params["temperature"] == 0.9
        assert llm.default_model_params["top_p"] == 0.95
        assert llm.default_model_params["n"] == 3
        assert llm._client.api_key == "test_api_key"
        assert llm._client.organization == "test_organization"

    @staticmethod
    def test_chat_completion(openai_llm, messages):
        response = openai_llm.chat_completion(messages=messages)
        assert_chat_completion(response)

    @staticmethod
    def test_enforced_function_call(openai_llm,
                                    messages,
                                    function_query_knowledgebase):
        result = openai_llm.enforced_function_call(
            messages=messages,
            function=function_query_knowledgebase)
        assert_function_call_format(result)

    @staticmethod
    def test_chat_completion_high_temperature(openai_llm,
                                              messages,
                                              model_params_high_temperature):
        response = openai_llm.chat_completion(
            messages=messages,
            model_params=model_params_high_temperature
        )
        assert_chat_completion(response,
                               num_choices=model_params_high_temperature["n"])

    @staticmethod
    def test_chat_completion_low_temperature(openai_llm,
                                             messages,
                                             model_params_low_temperature):
        response = openai_llm.chat_completion(messages=messages,
                                              model_params=model_params_low_temperature)
        assert_chat_completion(response,
                               num_choices=model_params_low_temperature["n"])

    @staticmethod
    def test_enforced_function_call_high_temperature(openai_llm,
                                                     messages,
                                                     function_query_knowledgebase,
                                                     model_params_high_temperature):
        result = openai_llm.enforced_function_call(
            messages=messages,
            function=function_query_knowledgebase,
            model_params=model_params_high_temperature
        )
        assert isinstance(result, dict)

    @staticmethod
    def test_enforced_function_call_low_temperature(openai_llm,
                                                    messages,
                                                    function_query_knowledgebase,
                                                    model_params_low_temperature):
        result = openai_llm.enforced_function_call(
            messages=messages,
            function=function_query_knowledgebase,
            model_params=model_params_low_temperature
        )
        assert_function_call_format(result)

    @staticmethod
    def test_chat_streaming(openai_llm, messages):
        stream = True
        response = openai_llm.chat_completion(messages=messages,
                                              stream=stream)
        messages_received = [message for message in response]
        assert len(messages_received) > 0
        for message in messages_received:
            assert isinstance(message, StreamingChatChunk)

    @staticmethod
    def test_max_tokens(openai_llm, messages):
        max_tokens = 2
        response = openai_llm.chat_completion(messages=messages,
                                              max_tokens=max_tokens)
        assert isinstance(response, ChatResponse)
        assert len(response.choices[0].message.content.split()) <= max_tokens

    @staticmethod
    def test_missing_messages(openai_llm):
        with pytest.raises(BadRequestError):
            openai_llm.chat_completion(messages=[])

    @staticmethod
    def test_negative_max_tokens(openai_llm, messages):
        with pytest.raises(BadRequestError):
            openai_llm.chat_completion(messages=messages, max_tokens=-5)

    @staticmethod
    def test_chat_complete_api_failure_populates(openai_llm,
                                                 messages):
        openai_llm._client = MagicMock()
        openai_llm._client.chat.completions.create.side_effect = Exception(
            "API call failed")

        with pytest.raises(Exception, match="API call failed"):
            openai_llm.chat_completion(messages=messages)

    @staticmethod
    def test_enforce_function_api_failure_populates(openai_llm,
                                                    messages,
                                                    function_query_knowledgebase):
        openai_llm._client = MagicMock()
        openai_llm._client.chat.completions.create.side_effect = Exception(
            "API call failed")

        with pytest.raises(Exception, match="API call failed"):
            openai_llm.enforced_function_call(messages=messages,
                                              function=function_query_knowledgebase)

    @staticmethod
    def test_enforce_function_wrong_output_schema(openai_llm,
                                                  messages,
                                                  function_query_knowledgebase):
        openai_llm._client = MagicMock()
        openai_llm._client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(
                    tool_calls=[
                        MagicMock(function=MagicMock(arguments="{\"key\": \"value\"}"))]))])

        with pytest.raises(jsonschema.ValidationError,
                           match="'queries' is a required property"):
            openai_llm.enforced_function_call(messages=messages,
                                              function=function_query_knowledgebase)

        assert openai_llm._client.chat.completions.create.call_count == 3, \
            "retry did not happen as expected"
