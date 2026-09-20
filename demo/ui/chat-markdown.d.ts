export interface ChatMarkdownQuality {
  words?: ReadonlyArray<{
    text: string;
    summedSurprisal: number | null;
    rollingPerplexity: number | null;
    cumulativePerplexity: number | null;
    tokenCount: number;
    rollingWindow: { unit: 'words' | 'tokens'; size: number };
  }>;
}

/** Render inert Markdown text, safe links, and optional word-quality annotations. */
export declare function renderChatMarkdown(
  container: HTMLElement,
  source: string,
  quality?: ChatMarkdownQuality | null
): void;
