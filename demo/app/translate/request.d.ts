export interface TranslateTextContentBlock {
  type: 'text';
  source_lang_code: string;
  target_lang_code: string;
  text: string;
}

export interface TranslateUserMessage {
  role: 'user';
  content: [TranslateTextContentBlock];
}

export interface TranslateTextRequest {
  messages: [TranslateUserMessage];
}

export interface TranslateTextFields {
  sourceLangCode: string;
  targetLangCode: string;
  text: string;
}

export declare function createTranslateTextRequest(
  text: string,
  sourceLangCode: string,
  targetLangCode: string
): TranslateTextRequest;

export declare function toTranslateTemplateLangCode(code: string): string;

export declare function extractTranslateTextFields(
  request: TranslateTextRequest
): TranslateTextFields;
