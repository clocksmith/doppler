// qwen4_exp QSA token selection, one workgroup per query token.
// visible = start_pos + token + 1. At or below dense_visible_limit the query
// selects every visible token (mode 0, count = visible; no list is written).
// Otherwise (mode 1) it keeps BLOCK_TOP_K of the visible / COMPRESS_RATIO
// complete blocks by score and always keeps the trailing incomplete block.
// Equal scores resolve by lower block index (Doppler-declared order; the
// source's torch.topk leaves ties unspecified). +0 and -0 compare equal. The
// list holds token indices in ascending order: selected blocks, then the tail.

override WORKGROUP_SIZE: u32 = 256u;
override COMPRESS_RATIO: u32;
override BLOCK_TOP_K: u32;

struct Uniforms {
    num_tokens: u32,
    start_pos: u32,
    block_stride: u32,       // row stride of the score scratch (blocks)
    dense_visible_limit: u32,
    max_selected: u32,       // row stride of the selection list (tokens)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> scores: array<f32>;            // [num_tokens, block_stride]
@group(0) @binding(2) var<storage, read_write> selection: array<u32>;   // [num_tokens, max_selected]
@group(0) @binding(3) var<storage, read_write> selection_meta: array<u32>; // [num_tokens, 2]: count, mode

var<workgroup> histogram: array<atomic<u32>, 256>;
var<workgroup> scan: array<u32, WORKGROUP_SIZE>;
var<workgroup> chosen_bin: u32;
var<workgroup> chosen_remaining: u32;
var<workgroup> running_ties: u32;
var<workgroup> running_taken: u32;

// Order-preserving key: larger score -> larger key; +0 and -0 share a key.
fn order_key(score: f32) -> u32 {
    if (score == 0.0) {
        return 0x80000000u;
    }
    let bits = bitcast<u32>(score);
    if ((bits & 0x80000000u) != 0u) {
        return ~bits;
    }
    return bits | 0x80000000u;
}

// Inclusive workgroup scan of `value`; returns the inclusive prefix.
fn inclusive_scan(local: u32, value: u32) -> u32 {
    scan[local] = value;
    workgroupBarrier();
    for (var offset = 1u; offset < WORKGROUP_SIZE; offset = offset * 2u) {
        var addend = 0u;
        if (local >= offset) {
            addend = scan[local - offset];
        }
        workgroupBarrier();
        scan[local] = scan[local] + addend;
        workgroupBarrier();
    }
    return scan[local];
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let token = wid.x;
    if (token >= u.num_tokens) {
        return;
    }
    let local = lid.x;
    let visible = u.start_pos + token + 1u;
    if (visible <= u.dense_visible_limit) {
        if (local == 0u) {
            selection_meta[token * 2u] = visible;
            selection_meta[token * 2u + 1u] = 0u;
        }
        return;
    }
    let complete_blocks = visible / COMPRESS_RATIO;
    let row = token * u.block_stride;

    // Radix select of the BLOCK_TOP_K-th largest key, 8 bits per pass.
    var prefix = 0u;
    var remaining = BLOCK_TOP_K;
    for (var round = 0u; round < 4u; round = round + 1u) {
        let shift = 24u - round * 8u;
        for (var bin = local; bin < 256u; bin = bin + WORKGROUP_SIZE) {
            atomicStore(&histogram[bin], 0u);
        }
        workgroupBarrier();
        for (var block = local; block < complete_blocks; block = block + WORKGROUP_SIZE) {
            let key = order_key(scores[row + block]);
            let matches = round == 0u || (key >> (shift + 8u)) == prefix;
            if (matches) {
                atomicAdd(&histogram[(key >> shift) & 255u], 1u);
            }
        }
        workgroupBarrier();
        if (local == 0u) {
            var need = remaining;
            var bin = 255u;
            loop {
                let count = atomicLoad(&histogram[bin]);
                if (count >= need || bin == 0u) {
                    break;
                }
                need = need - count;
                bin = bin - 1u;
            }
            chosen_bin = bin;
            chosen_remaining = need;
        }
        workgroupBarrier();
        prefix = (prefix << 8u) | chosen_bin;
        remaining = chosen_remaining;
        workgroupBarrier();
    }
    let threshold = prefix;
    let ties_needed = remaining;

    // Ordered compaction: keys above the threshold, then the first ties_needed
    // ties by block index; selected blocks are emitted in ascending order.
    if (local == 0u) {
        running_ties = 0u;
        running_taken = 0u;
    }
    workgroupBarrier();
    let base = token * u.max_selected;
    for (var start = 0u; start < complete_blocks; start = start + WORKGROUP_SIZE) {
        let block = start + local;
        var above = 0u;
        var tie = 0u;
        if (block < complete_blocks) {
            let key = order_key(scores[row + block]);
            above = select(0u, 1u, key > threshold);
            tie = select(0u, 1u, key == threshold);
        }
        let ties_before = running_ties;
        let tie_rank = inclusive_scan(local, tie) - tie + ties_before;
        let take = select(0u, 1u, above == 1u || (tie == 1u && tie_rank < ties_needed));
        let tie_total = scan[WORKGROUP_SIZE - 1u];
        workgroupBarrier();
        let taken_before = running_taken;
        let position = inclusive_scan(local, take) - take + taken_before;
        let take_total = scan[WORKGROUP_SIZE - 1u];
        if (take == 1u) {
            for (var r = 0u; r < COMPRESS_RATIO; r = r + 1u) {
                selection[base + position * COMPRESS_RATIO + r] = block * COMPRESS_RATIO + r;
            }
        }
        workgroupBarrier();
        if (local == 0u) {
            running_ties = ties_before + tie_total;
            running_taken = taken_before + take_total;
        }
        workgroupBarrier();
    }

    let tail_start = complete_blocks * COMPRESS_RATIO;
    let tail_offset = BLOCK_TOP_K * COMPRESS_RATIO;
    for (var i = local; i < visible - tail_start; i = i + WORKGROUP_SIZE) {
        selection[base + tail_offset + i] = tail_start + i;
    }
    if (local == 0u) {
        selection_meta[token * 2u] = tail_offset + (visible - tail_start);
        selection_meta[token * 2u + 1u] = 1u;
    }
}
