### Overview
`text_splitter` is an dify plugin designed to split text into chunks, built on top of `langchain-text-splitters`.

### Supported Splitters:

- CharacterTextSplitter
  - Splitting text by look at characters.
- RecursiveCharacterTextSplitter: 
  - Splitting text by recursively look at characters. Recursively tries to split by different characters to find one that works.
- MarkdownTextSplitter
  - Splitting markdown text along Markdown-formatted headings.
- MarkdownHeaderTextSplitter
  - Splitting markdown text based on specified headers.


Reference: [LangChain Text splitters](https://python.langchain.com/docs/concepts/text_splitters/)
