import logging
from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from langchain_text_splitters.markdown import MarkdownTextSplitter


class MarkdownTextSplitterTool(Tool):
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage, None, None]:
        """
        invoke tools
        """
        text = tool_parameters.get("text")
        chunk_size = int(tool_parameters.get("chunk_size", "1000"))
        chunk_overlap = int(tool_parameters.get("chunk_overlap", "0"))
        is_separator_regex = tool_parameters.get("is_separator_regex", False)
        keep_separator = tool_parameters.get("keep_separator", "end").lower()
        if not text:
            yield self.create_text_message("Empty text")
            return
        if chunk_size < chunk_overlap or chunk_size <= 0:
            yield self.create_text_message("Invalid chunk_size or chunk_overlap")
            return
        if keep_separator not in ["start", "end"]:
            yield self.create_text_message("Invalid keep_separator")
            return
        try:
            text_splitter = MarkdownTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                keep_separator=keep_separator,
                is_separator_regex=is_separator_regex,
            )
            chunks = text_splitter.split_text(text)
            yield self.create_variable_message("chunks", chunks)
        except Exception as e:
            logging.exception("Failed to split text")
            yield self.create_text_message(f"Failed to split text, error: {str(e)}")
