import os
import pytest
from unittest.mock import MagicMock, patch
from click.testing import CliRunner

from embed import cli


CONFIG_PATH = os.path.expanduser("~/.config/knowledge-vault/.env")


def invoke_setup(*args, input_text=None):
    runner = CliRunner()
    return runner.invoke(cli, ["setup", *args], input=input_text, catch_exceptions=False)


class TestSetupValid:
    def test_valid_key_and_qdrant_writes_config(self, tmp_path, monkeypatch):
        config_dir = str(tmp_path / "knowledge-vault")
        env_path = os.path.join(config_dir, ".env")
        monkeypatch.setattr(
            "os.path.expanduser",
            lambda p: p.replace("~/.config/knowledge-vault", config_dir),
        )

        with patch("embed.openai.OpenAI") as mock_openai, \
             patch("embed.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_setup(
                "--openai-api-key", "sk-valid",
                "--qdrant-url", "http://localhost:6333",
            )

        assert result.exit_code == 0
        assert "Setup complete" in result.output
        assert os.path.exists(env_path)
        content = open(env_path).read()
        assert "OPENAI_API_KEY=sk-valid" in content
        assert "QDRANT_URL=http://localhost:6333" in content

    def test_success_message_printed(self, tmp_path, monkeypatch):
        config_dir = str(tmp_path / "knowledge-vault")
        monkeypatch.setattr(
            "os.path.expanduser",
            lambda p: p.replace("~/.config/knowledge-vault", config_dir),
        )

        with patch("embed.openai.OpenAI") as mock_openai, \
             patch("embed.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_setup(
                "--openai-api-key", "sk-valid",
                "--qdrant-url", "http://localhost:6333",
            )

        assert "~/.config/knowledge-vault/.env" in result.output


class TestSetupInvalidKey:
    def test_invalid_openai_key_exits_1(self, tmp_path, monkeypatch):
        config_dir = str(tmp_path / "knowledge-vault")
        monkeypatch.setattr(
            "os.path.expanduser",
            lambda p: p.replace("~/.config/knowledge-vault", config_dir),
        )

        with patch("embed.openai.OpenAI") as mock_openai, \
             patch("embed.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.side_effect = Exception("401 Unauthorized")
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_setup(
                "--openai-api-key", "sk-bad",
                "--qdrant-url", "http://localhost:6333",
            )

        assert result.exit_code == 1

    def test_invalid_openai_key_prints_error(self, tmp_path, monkeypatch):
        config_dir = str(tmp_path / "knowledge-vault")
        monkeypatch.setattr(
            "os.path.expanduser",
            lambda p: p.replace("~/.config/knowledge-vault", config_dir),
        )

        with patch("embed.openai.OpenAI") as mock_openai, \
             patch("embed.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.side_effect = Exception("401 Unauthorized")
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_setup(
                "--openai-api-key", "sk-bad",
                "--qdrant-url", "http://localhost:6333",
            )

        assert "Error" in result.output
        assert "OpenAI" in result.output

    def test_invalid_openai_key_does_not_write_config(self, tmp_path, monkeypatch):
        config_dir = str(tmp_path / "knowledge-vault")
        env_path = os.path.join(config_dir, ".env")
        monkeypatch.setattr(
            "os.path.expanduser",
            lambda p: p.replace("~/.config/knowledge-vault", config_dir),
        )

        with patch("embed.openai.OpenAI") as mock_openai, \
             patch("embed.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.side_effect = Exception("401")
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            invoke_setup(
                "--openai-api-key", "sk-bad",
                "--qdrant-url", "http://localhost:6333",
            )

        assert not os.path.exists(env_path)


class TestSetupMerge:
    def test_merges_with_existing_config(self, tmp_path, monkeypatch):
        config_dir = str(tmp_path / "knowledge-vault")
        os.makedirs(config_dir)
        env_path = os.path.join(config_dir, ".env")
        # Pre-populate with an unrelated key
        with open(env_path, "w") as f:
            f.write("SOME_OTHER_KEY=original_value\n")
            f.write("OPENAI_API_KEY=sk-old\n")

        monkeypatch.setattr(
            "os.path.expanduser",
            lambda p: p.replace("~/.config/knowledge-vault", config_dir),
        )

        with patch("embed.openai.OpenAI") as mock_openai, \
             patch("embed.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            result = invoke_setup(
                "--openai-api-key", "sk-new",
                "--qdrant-url", "http://localhost:6333",
            )

        assert result.exit_code == 0
        content = open(env_path).read()
        # Unrelated key preserved
        assert "SOME_OTHER_KEY=original_value" in content
        # OpenAI key updated
        assert "OPENAI_API_KEY=sk-new" in content
        # Old key not present
        assert "sk-old" not in content

    def test_does_not_duplicate_keys(self, tmp_path, monkeypatch):
        config_dir = str(tmp_path / "knowledge-vault")
        os.makedirs(config_dir)
        env_path = os.path.join(config_dir, ".env")
        with open(env_path, "w") as f:
            f.write("OPENAI_API_KEY=sk-old\n")
            f.write("QDRANT_URL=http://old:6333\n")

        monkeypatch.setattr(
            "os.path.expanduser",
            lambda p: p.replace("~/.config/knowledge-vault", config_dir),
        )

        with patch("embed.openai.OpenAI") as mock_openai, \
             patch("embed.QdrantClient") as mock_qdrant:
            mock_openai.return_value.models.list.return_value = MagicMock()
            mock_qdrant.return_value.get_collections.return_value = MagicMock()
            invoke_setup(
                "--openai-api-key", "sk-new",
                "--qdrant-url", "http://new:6333",
            )

        lines = open(env_path).readlines()
        keys = [line.split("=")[0] for line in lines if "=" in line]
        assert keys.count("OPENAI_API_KEY") == 1
        assert keys.count("QDRANT_URL") == 1
