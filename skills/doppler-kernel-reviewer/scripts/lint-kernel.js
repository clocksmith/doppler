import fs from 'fs';
import path from 'path';

const filePath = process.argv[2];
if (!filePath) {
    console.error('Usage: node lint-kernel.js <file-path>');
    process.exit(1);
}

const content = fs.readFileSync(filePath, 'utf8');
const ext = path.extname(filePath);
const filename = path.basename(filePath);

const errors = [];
const warnings = [];

// === JavaScript Checks ===
if (ext === '.js') {
    // Check for JSDoc
    if (/\/\*\*\s*\n/.test(content) || /\s\*\s@param/.test(content)) {
        errors.push('Found JSDoc comments (/** ... */). Move types to .d.ts.');
    }

    // Check for hardcoded workgroup sizes (heuristic)
    if (/const\s+WORKGROUP_SIZE\s+=\s+\d+/.test(content)) {
        warnings.push('Found hardcoded WORKGROUP_SIZE. Should usually come from constants/config.');
    }

    // Check for inline ternary kernel selection
    if (/(['"]\w+['"]\s*\?\s*['"]\w+['"]\s*:\s*['"]\w+['"])/.test(content)) {
        warnings.push('Possible inline ternary for string selection. Use rule registry for kernel variants.');
    }
}

// === WGSL Checks ===
if (ext === '.wgsl') {
    // Check for hardcoded workgroup size in attribute
    if (/@workgroup_size\(\s*\d+\s*,/.test(content)) {
        errors.push('Found hardcoded @workgroup_size. Use override constant.');
    }

    // Check for vec3 usage in global storage/uniforms (heuristic: look for struct fields)
    // This is hard to detect perfectly with regex, but we can warn on `vec3<f32>`
    if (/:\s*vec3<f32>/.test(content) || /:\s*vec3<u32>/.test(content) || /:\s*vec3<i32>/.test(content)) {
        warnings.push('Found vec3 usage in struct/variable. Ensure it is not in storage/uniform buffers (alignment issues).');
    }

    // Check for naming conventions (UPPER_SNAKE_CASE for overrides)
    if (/override\s+[a-z]/.test(content)) {
        errors.push('Override constants must be UPPER_SNAKE_CASE.');
    }
}

// === Report ===
console.log(`Linting ${filename}...`);
if (errors.length > 0) {
    console.error('\nERRORS:');
    errors.forEach(e => console.error(`[x] ${e}`));
}
if (warnings.length > 0) {
    console.log('\nWARNINGS:');
    warnings.forEach(w => console.log(`[!] ${w}`));
}

if (errors.length === 0 && warnings.length === 0) {
    console.log('No obvious violations found.');
}

if (errors.length > 0) process.exit(1);
