from agents_for_agents.core import AgentMessage, build_agent_response


def test_build_agent_response_handles_empty_history():
    response = build_agent_response([])
    assert "No context provided" in response


def test_build_agent_response_renders_history():
    history = [
        AgentMessage(role="system", content="Configure"),
        AgentMessage(role="user", content="Hello"),
    ]

    response = build_agent_response(history)

    assert "system" in response
    assert "user" in response
    assert "Context" in response
