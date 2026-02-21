import { AutoConfig } from '@huggingface/transformers';

async function main() {
    const config = await AutoConfig.from_pretrained('google/functiongemma-270m-it');
    console.log("HIDDEN SIZE:", config.hidden_size);
    console.log("SCALAR?:", config.query_pre_attn_scalar);
    console.log("FULL CONFIG:", JSON.stringify(config, null, 2));
}

main().catch(console.error);
