export const WGSL_GENERATED_VARIANTS: readonly ({
    id: string;
    source: string;
    target: string;
    patch: string;
} | {
    id: string;
    source: string;
    target: string;
    rules: {
        type: string;
        count: number;
        from: string;
        to: string;
    }[];
})[];
