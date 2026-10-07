# Universal server-client

## /Simple_Remote_Infer/deploy/qwenpi_policy.py

- QwenPiServer: A model for demonstration with init and infer methods

Package the loaded model with `WebsocketPolicyServer` and specify the port

```python
model_server = WebsocketPolicyServer(model, port=8002)
model_server.serve_forever() # Turn on the listening.
```

## ./websocket_client_policy.py

Show in `__main__()` how to create a false model to send environmental information to a real model, just replace the original model with `WebsocketClientPolicy`