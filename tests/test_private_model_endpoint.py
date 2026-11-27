from environment.runtime.observation_window_proxy import permitted_model_endpoint


def test_public_http_remains_rejected(monkeypatch):
    monkeypatch.setenv('WORLD_ALLOW_PRIVATE_MODEL_HTTP','1')
    for url in ('http://8.8.8.8/v1','http://public.example/v1','file:///tmp/token'):
        assert not permitted_model_endpoint(url)


def test_private_http_is_opt_in(monkeypatch):
    monkeypatch.delenv('WORLD_ALLOW_PRIVATE_MODEL_HTTP',raising=False)
    assert not permitted_model_endpoint('http://172.17.0.1:18088/v1')
    assert permitted_model_endpoint('http://127.0.0.1:18088/v1')
    assert permitted_model_endpoint('https://api.example/v1')
    monkeypatch.setenv('WORLD_ALLOW_PRIVATE_MODEL_HTTP','1')
    assert permitted_model_endpoint('http://172.17.0.1:18088/v1')
