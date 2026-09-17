from dfsignal.config import Settings


def test_default_settings_are_safe_for_local_startup() -> None:
    settings = Settings()

    assert settings.app_env == "development"
    assert settings.log_level == "INFO"
