from __future__ import annotations

from PySide6.QtCore import QCoreApplication
from typing import TYPE_CHECKING
from modules.inpainting.lama import LaMa
from modules.inpainting.mi_gan import MIGAN
from modules.inpainting.aot import AOT
from modules.inpainting.schema import Config
from app.ui.messages import Messages
from app.ui.settings.settings_page import SettingsPage

if TYPE_CHECKING:
    from controller import ComicTranslate

inpaint_map = {
    "LaMa": LaMa,
    "MI-GAN": MIGAN,
    "AOT": AOT,
}


def get_inpainter_backend(inpainter_key: str) -> str:
    inpainter_cls = inpaint_map[inpainter_key]
    return getattr(inpainter_cls, "preferred_backend", "onnx")

def get_config(settings_page: SettingsPage):
    strategy_settings = settings_page.get_hd_strategy_settings()
    if strategy_settings['strategy'] == settings_page.ui.tr("Resize"):
        config = Config(hd_strategy="Resize", hd_strategy_resize_limit = strategy_settings['resize_limit'])
    elif strategy_settings['strategy'] == settings_page.ui.tr("Crop"):
        config = Config(hd_strategy="Crop", hd_strategy_crop_margin = strategy_settings['crop_margin'],
                        hd_strategy_crop_trigger_size = strategy_settings['crop_trigger_size'])
    else:
        config = Config(hd_strategy="Original")

    return config

def validate_ocr(main: ComicTranslate):
    """Ensure the selected OCR tool has its required credentials configured."""
    settings_page = main.settings_page
    tr = settings_page.ui.tr
    settings = settings_page.get_all_settings()
    credentials = settings.get('credentials', {})
    ocr_tool = settings['tools']['ocr']

    if not ocr_tool:
        Messages.show_missing_tool_error(main, QCoreApplication.translate("Messages", "Text Recognition model"))
        return False

    # Tools that need a Google Cloud API key
    GOOGLE_CLOUD_OCR_TOOLS = {"Google Cloud Vision"}
    # Tools that need Microsoft Azure credentials
    MICROSOFT_OCR_TOOLS = {"Microsoft OCR"}
    # Tools that need a Google Gemini API key (routed through backend unless key provided)
    GEMINI_OCR_TOOLS = {"Gemini-2.0-Flash"}

    if ocr_tool in GOOGLE_CLOUD_OCR_TOOLS:
        api_key = credentials.get(tr('Google Cloud'), {}).get('api_key')
        if not api_key:
            Messages.show_missing_credentials_error(
                main, "Google Cloud Vision",
                QCoreApplication.translate("Messages", "Google Cloud API key (Settings > Credentials > Google Cloud)")
            )
            return False

    elif ocr_tool in MICROSOFT_OCR_TOOLS:
        ms_creds = credentials.get(tr('Microsoft Azure'), {})
        if not ms_creds.get('api_key') or not ms_creds.get('endpoint'):
            Messages.show_missing_credentials_error(
                main, "Microsoft OCR",
                QCoreApplication.translate("Messages", "Microsoft Azure credentials (Settings > Credentials)")
            )
            return False

    elif ocr_tool in GEMINI_OCR_TOOLS:
        api_key = credentials.get(tr('Google Gemini'), {}).get('api_key')
        if not api_key:
            Messages.show_missing_credentials_error(
                main, ocr_tool,
                QCoreApplication.translate("Messages", "Google Gemini API key (Settings > Credentials > Google Gemini)")
            )
            return False

    return True


def validate_translator(main: ComicTranslate, target_lang: str):
    """Ensure the selected translator has its required credentials configured."""
    settings_page = main.settings_page
    tr = settings_page.ui.tr
    settings = settings_page.get_all_settings()
    credentials = settings.get('credentials', {})
    translator_tool = settings['tools']['translator']

    if not translator_tool:
        Messages.show_missing_tool_error(main, QCoreApplication.translate("Messages", "Translator"))
        return False

    if "Custom" in translator_tool:
        # Custom requires api_key, api_url, and model to be configured LOCALLY
        service = tr('Custom')
        creds = credentials.get(service, {})
        if not all([creds.get('api_key'), creds.get('api_url'), creds.get('model')]):
            Messages.show_custom_not_configured_error(main)
            return False

    elif "Gemini" in translator_tool:
        api_key = credentials.get(tr('Google Gemini'), {}).get('api_key')
        if not api_key:
            Messages.show_missing_credentials_error(
                main, translator_tool,
                QCoreApplication.translate("Messages", "Google Gemini API key (Settings > Credentials > Google Gemini)")
            )
            return False

    elif "GPT" in translator_tool:
        api_key = credentials.get(tr('Open AI GPT'), {}).get('api_key')
        if not api_key:
            Messages.show_missing_credentials_error(
                main, translator_tool,
                QCoreApplication.translate("Messages", "OpenAI API key (Settings > Credentials)")
            )
            return False

    elif "Claude" in translator_tool:
        api_key = credentials.get(tr('Anthropic Claude'), {}).get('api_key')
        if not api_key:
            Messages.show_missing_credentials_error(
                main, translator_tool,
                QCoreApplication.translate("Messages", "Anthropic Claude API key (Settings > Credentials)")
            )
            return False

    elif "Deepseek" in translator_tool:
        api_key = credentials.get(tr('Deepseek'), {}).get('api_key')
        if not api_key:
            Messages.show_missing_credentials_error(
                main, translator_tool,
                QCoreApplication.translate("Messages", "Deepseek API key (Settings > Credentials)")
            )
            return False

    return True

def font_selected(main: ComicTranslate):
    if not main.render_settings().font_family:
        Messages.select_font_error(main)
        return False
    return True

def validate_settings(main: ComicTranslate, target_lang: str):
    if not validate_ocr(main):
        return False
    if not validate_translator(main, target_lang):
        return False
    if not font_selected(main):
        return False
    
    return True
