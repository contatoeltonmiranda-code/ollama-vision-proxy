"""Logging must not intrude on the terminal while claude is drawing in it."""

import logging

from ollama_vision_proxy import cli


def _reset():
    logging.getLogger().handlers = []
    cli._terminal_handler = None
    cli._keep_terminal_logging = False


class TestTerminalSilencing:
    def setup_method(self):
        _reset()

    def teardown_method(self):
        _reset()

    def test_startup_messages_reach_stderr(self, capsys):
        cli._configure_logging(verbose=False)
        logging.getLogger("ovp").info("starting up")
        assert "starting up" in capsys.readouterr().err

    def test_disabling_stops_stderr_output(self, capsys):
        cli._configure_logging(verbose=False)
        capsys.readouterr()
        cli.set_terminal_logging(False)
        logging.getLogger("ovp").info("transcribed 1 image(s)")
        logging.getLogger("ovp").warning("even warnings stay out")
        assert capsys.readouterr().err == ""

    def test_re_enabling_restores_stderr_output(self, capsys):
        cli._configure_logging(verbose=False)
        cli.set_terminal_logging(False)
        cli.set_terminal_logging(True)
        capsys.readouterr()
        logging.getLogger("ovp").info("cache stats")
        assert "cache stats" in capsys.readouterr().err

    def test_verbose_keeps_the_terminal_attached(self, capsys):
        cli._configure_logging(verbose=True)
        capsys.readouterr()
        cli.set_terminal_logging(False)
        logging.getLogger("ovp").info("asked for noise")
        assert "asked for noise" in capsys.readouterr().err

    def test_toggling_is_idempotent(self, capsys):
        cli._configure_logging(verbose=False)
        cli.set_terminal_logging(False)
        cli.set_terminal_logging(False)
        cli.set_terminal_logging(True)
        cli.set_terminal_logging(True)
        assert logging.getLogger().handlers.count(cli._terminal_handler) == 1


class TestLogFile:
    def setup_method(self):
        _reset()

    def teardown_method(self):
        _reset()

    def test_log_file_receives_records(self, tmp_path):
        path = tmp_path / "ovp.log"
        cli._configure_logging(verbose=False, log_file=str(path))
        logging.getLogger("ovp").info("written to file")
        for handler in logging.getLogger().handlers:
            handler.flush()
        assert "written to file" in path.read_text()

    def test_log_file_still_records_while_terminal_is_silent(self, tmp_path, capsys):
        path = tmp_path / "ovp.log"
        cli._configure_logging(verbose=False, log_file=str(path))
        capsys.readouterr()
        cli.set_terminal_logging(False)
        logging.getLogger("ovp").info("session detail")
        for handler in logging.getLogger().handlers:
            handler.flush()
        assert capsys.readouterr().err == ""
        assert "session detail" in path.read_text()
