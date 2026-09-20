export interface ChatMarkdownWordQuality {
  text: string;
  summedSurprisal: number | null;
  rollingPerplexity: number | null;
  cumulativePerplexity: number | null;
  rollingWindow?: {
    unit: 'words' | 'tokens';
    size: number;
  };
}

export declare function renderChatMarkdown(
  container: HTMLElement,
  source: unknown,
  quality?: { words?: ChatMarkdownWordQuality[] } | null,
): void;
