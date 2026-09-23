// File names are labels, not identities. Preserve an existing document only on
// an unambiguous retained-content match; duplicate imports receive distinct IDs.
export async function importDocuments(files, priorDocuments = []) {
  const used = new Set();
  const documents = [];
  for (const file of files) {
    if (!/\.(txt|md|markdown)$/i.test(file.name)) throw new Error(`Unsupported file type: ${file.name}. Choose text or Markdown.`);
    const mediaType = /\.(md|markdown)$/i.test(file.name) ? 'text/markdown' : 'text/plain';
    const text = await file.text();
    const prior = priorDocuments.find(document => !used.has(document.id)
      && document.title === file.name && document.mediaType === mediaType && document.text === text);
    const id = prior?.id ?? crypto.randomUUID();
    used.add(id);
    documents.push({ id, title: file.name, text, mediaType });
  }
  return documents;
}
