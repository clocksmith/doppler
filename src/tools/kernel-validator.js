
/**
 * Platform-agnostic kernel validation logic.
 * Designed to work with both Node.js fs implementation and Browser WorkspaceVfs.
 */
export class KernelValidator {
    /**
     * @param {object} fs - File system interface (readText, exists, list)
     */
    constructor(fs) {
        this.fs = fs;
    }

    async validateRegistry(registryPath, kernelDir) {
        const results = {
            ok: true,
            errors: [],
            filesCount: 0,
        };

        try {
            if (!(await this.fs.exists(registryPath))) {
                results.ok = false;
                results.errors.push(`Registry file not found: ${registryPath}`);
                return results;
            }

            const raw = await this.fs.readText(registryPath);
            const registry = JSON.parse(raw);

            const wgslFiles = new Set();
            for (const operation of Object.values(registry.operations)) {
                for (const variant of Object.values(operation.variants)) {
                    wgslFiles.add(variant.wgsl);
                }
            }

            results.filesCount = wgslFiles.size;
            const missing = [];

            // Check each file
            for (const wgsl of wgslFiles) {
                // Simple path join for browser/node compatibility (assumes forward slashes)
                const filePath = `${kernelDir}/${wgsl}`.replace(/\/+/g, '/');
                if (!(await this.fs.exists(filePath))) {
                    missing.push(wgsl);
                }
            }

            if (missing.length > 0) {
                results.ok = false;
                results.errors.push('Kernel registry references missing WGSL files:');
                for (const name of missing.sort()) {
                    results.errors.push(`  - ${name}`);
                }
            }
        } catch (error) {
            results.ok = false;
            results.errors.push(`Validation error: ${error.message}`);
        }

        return results;
    }

    async lintWgslOverrides(kernelDir, whitelist = []) {
        const results = {
            ok: true,
            errors: [],
            checkedCount: 0,
        };

        try {
            const files = await this.fs.list(kernelDir);
            const wgslFiles = files.filter(f => f.endsWith('.wgsl'));
            const whitelistSet = new Set(whitelist);

            results.checkedCount = wgslFiles.length;

            for (const file of wgslFiles) {
                // Adjust for full path from list vs relative name
                const fileName = file.split('/').pop();
                if (whitelistSet.has(fileName)) continue;

                const content = await this.fs.readText(file);
                const source = this.#stripComments(content);

                const overrideMatches = Array.from(source.matchAll(/\boverride\s+([A-Za-z_][A-Za-z0-9_]*)\b/g));
                const overrides = overrideMatches.map((match) => match[1]);

                if (overrides.length === 0) continue;

                const overrideRegexes = overrides.map(name => new RegExp(`\\b${this.#escapeRegExp(name)}\\b`));
                const statements = source.split(';');

                for (const statement of statements) {
                    if (!statement.includes('var<workgroup>') || !statement.includes('array<')) {
                        continue;
                    }

                    const lengths = this.#findArrayLengths(statement);
                    for (const lengthExpr of lengths) {
                        for (const regex of overrideRegexes) {
                            if (regex.test(lengthExpr)) {
                                results.ok = false;
                                results.errors.push(`${fileName}: workgroup array length uses override (${lengthExpr})`);
                                break;
                            }
                        }
                    }
                }
            }
        } catch (error) {
            results.ok = false;
            results.errors.push(`Lint error: ${error.message}`);
        }

        return results;
    }

    #stripComments(source) {
        const withoutBlock = source.replace(/\/\*[\s\S]*?\*\//g, '');
        return withoutBlock.replace(/\/\/.*$/gm, '');
    }

    #escapeRegExp(text) {
        return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    #findArrayLengths(statement) {
        const lengths = [];
        for (let i = 0; i < statement.length; i += 1) {
            if (!statement.startsWith('array<', i)) {
                continue;
            }
            const start = i + 'array<'.length;
            let depth = 1;
            let j = start;
            for (; j < statement.length; j += 1) {
                const ch = statement[j];
                if (ch === '<') depth += 1;
                if (ch === '>') {
                    depth -= 1;
                    if (depth === 0) break;
                }
            }
            if (depth !== 0) {
                continue;
            }
            const inner = statement.slice(start, j);
            let commaIndex = -1;
            let innerDepth = 0;
            for (let k = 0; k < inner.length; k += 1) {
                const ch = inner[k];
                if (ch === '<') innerDepth += 1;
                if (ch === '>') innerDepth -= 1;
                if (ch === ',' && innerDepth === 0) {
                    commaIndex = k;
                }
            }
            if (commaIndex !== -1) {
                lengths.push(inner.slice(commaIndex + 1).trim());
            }
            i = j;
        }
        return lengths;
    }
}
