from .bubble_text import SpeechBubblePrompt, SpeechBubbleRender, SpeechBubbleText

NODE_CLASS_MAPPINGS = {
    "SpeechBubblePrompt": SpeechBubblePrompt,
    "SpeechBubbleRender": SpeechBubbleRender,
    "SpeechBubbleTextAuto": SpeechBubbleText,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SpeechBubblePrompt": "💬 Bubble Text · Prompt",
    "SpeechBubbleRender": "💬 Bubble Text · Write",
    "SpeechBubbleTextAuto": "💬 Bubble Text (all-in-one)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
