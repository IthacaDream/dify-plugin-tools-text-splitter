import json
import logging
from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from langchain_text_splitters import MarkdownHeaderTextSplitter


class MarkdownHeaderTextSplitterTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        """
        invoke tools
        """
        text = tool_parameters.get("text")
        if not text:
            yield self.create_text_message("Empty text")
            return

        try:
            headers_to_split_on = json.loads(tool_parameters.get("headers_to_split_on"))
        except Exception:
            yield self.create_text_message("Invalid headers_to_split_on")
            return
        return_each_line = tool_parameters.get("return_each_line", False)
        strip_headers = tool_parameters.get("strip_headers", True)

        try:
            text_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                return_each_line=return_each_line,
                strip_headers=strip_headers,
            )
            chunks = text_splitter.split_text(text)
            yield self.create_variable_message("chunks", chunks)
        except Exception as e:
            logging.exception("Failed to split text")
            yield self.create_text_message(f"Failed to split text, error: {str(e)}")
