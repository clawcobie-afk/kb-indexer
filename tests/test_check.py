import pytest
from unittest.mock import MagicMock, patch
from click.testing import CliRunner

from embed import cli


def invoke_check(*args, env=None):
    runner = CliRunner()
    return runner.invoke(cli, ["check", *args], env=env or {}, catch_exceptions=False)


# ── OPENAI_API_KEY check ──────────────────────────────────────────────────────

class TestCheckApiKeyPresence:
    def test_ok_when_key_provided_via_option(self):
        with patch("embed.openai.OpenAI") as mock_openai, \
             patch("embed.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_check("--openai-api-key", "sk-test")
        assert "OK  OPENAI_API_KEY is set" in result.output

    def test_ok_when_key_provided_via_env(self):
        with patch("embed.openai.OpenAI") as mock_openai, \
             patch("embed.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_check(env={"OPENAI_API_KEY": "sk-from-env"})
        assert "OK  OPENAI_API_KEY is set" in result.output

    def test_fail_when_key_missing(self):
        with patch("embed.openai.OpenAI") as mock_openai, \
             patch("embed.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.side_effect = Exception("auth error")
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_check()
        assert "FAIL OPENAI_API_KEY is not set or empty" in result.output


# ── OpenAI validity check ─────────────────────────────────────────────────────

class TestCheckOpenAIValidity:
    def test_ok_when_models_list_succeeds(self):
        with patch("embed.openai.OpenAI") as mock_openai, \
             patch("embed.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_check("--openai-api-key", "sk-valid")
        assert "OK  OpenAI API key is valid" in result.output

    def test_fail_when_models_list_raises(self):
        with patch("embed.openai.OpenAI") as mock_openai, \
             patch("embed.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.side_effect = Exception("401 Unauthorized")
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_check("--openai-api-key", "sk-bad")
        assert "FAIL OpenAI API key is invalid:" in result.output
        assert "401 Unauthorized" in result.output

    def test_openai_called_with_provided_key(self):
        with patch("embed.openai.OpenAI") as mock_openai, \
             patch("embed.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            invoke_check("--openai-api-key", "sk-mykey")
        mock_openai.assert_called_once_with(api_key="sk-mykey")


# ── Qdrant reachability check ─────────────────────────────────────────────────

class TestCheckQdrantReachability:
    def test_ok_when_get_collections_succeeds(self):
        with patch("embed.openai.OpenAI") as mock_openai, \
             patch("embed.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_check("--openai-api-key", "sk-test")
        assert "OK  Qdrant is reachable" in result.output

    def test_fail_when_get_collections_raises(self):
        with patch("embed.openai.OpenAI") as mock_openai, \
             patch("embed.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.side_effect = Exception("Connection refused")
            result = invoke_check("--openai-api-key", "sk-test")
        assert "FAIL Qdrant is not reachable:" in result.output
        assert "Connection refused" in result.output

    def test_qdrant_called_with_default_url(self):
        with patch("embed.openai.OpenAI") as mock_openai, \
             patch("embed.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            invoke_check("--openai-api-key", "sk-test")
        mock_qdrant.assert_called_once_with(url="http://localhost:6333")

    def test_qdrant_called_with_custom_url(self):
        with patch("embed.openai.OpenAI") as mock_openai, \
             patch("embed.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            invoke_check("--openai-api-key", "sk-test", "--qdrant-url", "http://remote:6333")
        mock_qdrant.assert_called_once_with(url="http://remote:6333")


# ── Summary line ──────────────────────────────────────────────────────────────

class TestCheckSummary:
    def test_all_checks_passed_summary(self):
        with patch("embed.openai.OpenAI") as mock_openai, \
             patch("embed.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_check("--openai-api-key", "sk-test")
        assert "All checks passed." in result.output

    def test_one_failure_summary(self):
        with patch("embed.openai.OpenAI") as mock_openai, \
             patch("embed.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.side_effect = Exception("down")
            result = invoke_check("--openai-api-key", "sk-test")
        assert "1 check(s) failed." in result.output

    def test_multiple_failures_summary(self):
        with patch("embed.openai.OpenAI") as mock_openai, \
             patch("embed.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.side_effect = Exception("auth")
            mock_qdrant.return_value.get_collections.side_effect = Exception("down")
            result = invoke_check()
        assert "3 check(s) failed." in result.output

    def test_exit_code_zero_on_all_passed(self):
        with patch("embed.openai.OpenAI") as mock_openai, \
             patch("embed.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_check("--openai-api-key", "sk-test")
        assert result.exit_code == 0
