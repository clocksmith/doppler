// Offline artifact-materialization experiment; never a model execution fallback.
export fn convert(input: [*]const u16, output: [*]f32, count: usize) void {
    const U = @Vector(4, u32);
    const F = @Vector(4, f32);
    var i: usize = 0;
    while (i + 4 <= count) : (i += 4) {
        const halves: @Vector(4, u16) = input[i..][0..4].*;
        const bits: U = halves;
        const sign = (bits & @as(U, @splat(0x8000))) << @as(U, @splat(16));
        const exponent = (bits >> @as(U, @splat(10))) & @as(U, @splat(31));
        const mantissa = bits & @as(U, @splat(1023));
        const normal = ((exponent + @as(U, @splat(112))) << @as(U, @splat(23))) |
            (mantissa << @as(U, @splat(13)));
        const subnormal: F = @as(F, @floatFromInt(mantissa)) * @as(F, @splat(0x1p-24));
        const special = @as(U, @splat(0x7f800000)) | (mantissa << @as(U, @splat(13)));
        const finite = @select(u32, exponent == @as(U, @splat(0)), @as(U, @bitCast(subnormal)), normal);
        const result: F = @bitCast(sign | @select(u32, exponent == @as(U, @splat(31)), special, finite));
        output[i..][0..4].* = result;
    }
    while (i < count) : (i += 1) {
        const half: f16 = @bitCast(input[i]);
        output[i] = @floatCast(half);
    }
}
