"""Give text-only Ollama models image support by transcribing images first.

Claude Code sends pasted images as Anthropic `image` content blocks. A text-only
model (for example glm-5.2:cloud) rejects those with HTTP 400, and because the
image stays in the conversation history every later request fails too. This
package sits between Claude Code and the Ollama server, replaces each image
block with a text description produced by a local Ollama vision model, and
forwards a text-only request upstream.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
