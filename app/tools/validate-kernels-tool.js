
import { KernelValidator } from '../../src/tools/kernel-validator.js';

export class ValidateKernelsTool {
    id = 'validate-kernels';
    name = 'Validate Kernels';
    description = 'Validate kernel registry and lint WGSL for override usage in arrays.';

    constructor(vfs) {
        this.vfs = vfs;
    }

    async run(args = {}) {
        if (!this.vfs) {
            throw new Error('No workspace loaded. Import the doppler folder first.');
        }

        const validator = new KernelValidator(this.vfs);
        const output = [];

        // 1. Registry Validation
        output.push('--- Validating Registry ---');
        // Assuming standard repo structure relative to root import
        const registryPath = 'src/config/kernels/registry.json';
        const kernelDir = 'src/gpu/kernels';

        output.push(`Reading registry: ${registryPath}`);
        const regResult = await validator.validateRegistry(registryPath, kernelDir);

        if (regResult.ok) {
            output.push(`✅ Registry OK (${regResult.filesCount} files referenced)`);
        } else {
            output.push('❌ Registry Validation Failed:');
            regResult.errors.forEach(e => output.push(e));
        }

        output.push('\n--- Linting WGSL Overrides ---');
        output.push(`Scanning: ${kernelDir}`);
        const lintResult = await validator.lintWgslOverrides(kernelDir);

        if (lintResult.ok) {
            output.push(`✅ WGSL Lint OK (${lintResult.checkedCount} files checked)`);
        } else {
            output.push('❌ WGSL Lint Failed:');
            lintResult.errors.forEach(e => output.push(e));
        }

        const success = regResult.ok && lintResult.ok;
        if (success) {
            output.push('\n✨ ALL CHECKS PASSED');
        } else {
            output.push('\n⚠️ CHECKS FAILED');
        }

        return output.join('\n');
    }
}
