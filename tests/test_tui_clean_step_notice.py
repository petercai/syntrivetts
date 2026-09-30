from rich.text import Text

from syntrive.tui.dialogs import TTS_TEXT_CLEAN_NOTICE, clean_step_notice


def test_notice_is_yellow_and_names_the_skill():
    notice = clean_step_notice()
    assert notice == TTS_TEXT_CLEAN_NOTICE
    assert "yellow" in notice
    assert "tts-text-clean" in notice


def test_notice_is_valid_rich_markup():
    assert "tts-text-clean" in Text.from_markup(clean_step_notice()).plain
