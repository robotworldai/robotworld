import socket
import socketserver
import threading

from environment.runtime.agent_container_entry import Relay


def test_early_413_reaches_client_while_upload_is_in_progress(tmp_path):
    path = str(tmp_path/'upstream.sock')
    response = b'HTTP/1.0 413 Request Entity Too Large\r\nContent-Length: 17\r\n\r\nrequest_too_large'
    class Upstream(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.recv(4096)
            self.request.sendall(response)
    class Bridge(Relay):
        upstream_socket = path
    with socketserver.ThreadingUnixStreamServer(path, Upstream) as upstream:
        with socketserver.ThreadingTCPServer(('127.0.0.1',0),Bridge) as bridge:
            upstream.daemon_threads = bridge.daemon_threads = True
            threads = [threading.Thread(target=s.serve_forever,daemon=True) for s in (upstream,bridge)]
            for t in threads: t.start()
            with socket.create_connection(bridge.server_address,timeout=5) as client:
                client.sendall(b'POST /v1/responses HTTP/1.0\r\nContent-Length: 999999999\r\n\r\n')
                # The response must arrive without completing the advertised upload.
                received=b''
                while chunk := client.recv(4096): received+=chunk
                assert received==response
            bridge.shutdown();upstream.shutdown()
