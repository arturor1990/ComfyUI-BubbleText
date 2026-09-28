from .bubble_text import SpeechBubblePrompt, SpeechBubbleRender, SpeechBubbleText

NODE_CLASS_MAPPINGS = {
    "SpeechBubblePrompt": SpeechBubblePrompt,
    "SpeechBubbleRender": SpeechBubbleRender,
    "SpeechBubbleTextAuto": SpeechBubbleText,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SpeechBubblePrompt": "💬 Globo de texto",
    "SpeechBubbleRender": "💬 Escribir en globo",
    "SpeechBubbleTextAuto": "💬 Texto en globo (auto)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
