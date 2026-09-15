import pytest
from recognition.evaluation.web_trigger_replay import loopback_endpoint


@pytest.mark.parametrize("url", ["https://example.org", "http://192.168.1.2:8642", "file://localhost/video", "http://127.0.0.1:8642/path", "http://user:pass@localhost:8642"])
def test_evaluation_cannot_launch_or_send_landmarks_to_remote_service(url):
    with pytest.raises(ValueError):
        loopback_endpoint(url)


def test_local_evaluation_endpoint_is_explicit():
    assert loopback_endpoint("http://127.0.0.1:8642") == ("127.0.0.1", 8642, "http")
