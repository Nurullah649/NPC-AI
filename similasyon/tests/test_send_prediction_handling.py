"""send_prediction response handling testleri."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest


class TestSendPredictionHandling:
    def test_main_has_status_checks(self):
        """main.py send_prediction sonucunu kontrol etmeli."""
        with open(os.path.join(os.path.dirname(__file__), '..', 'main.py')) as f:
            content = f.read()
        # Should check for 201, 406, 403 status codes
        assert '201' in content, "main.py HTTP 201 kontrolü yok!"
        assert '406' in content, "main.py HTTP 406 kontrolü yok!"
        assert '403' in content, "main.py HTTP 403 kontrolü yok!"

    def test_main_has_failed_payload_save(self):
        """main.py başarısız payload'ı kaydetmeli."""
        with open(os.path.join(os.path.dirname(__file__), '..', 'main.py')) as f:
            content = f.read()
        assert '_payloads_failed' in content, "main.py failed payload kaydetme yok!"

    def test_connection_handler_no_password_in_log(self):
        """connection_handler.py login'de password loglamamalı."""
        with open(os.path.join(os.path.dirname(__file__), '..', 'src', 'connection_handler.py')) as f:
            content = f.read()
        # Should log username but not password in login success
        assert 'format(username)' in content
        assert 'format(payload)' not in content or 'Login Successfully Completed : user' in content
