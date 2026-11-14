// Independent CPU specifications. These are NOT a WGSL compiler or GPU emulator.
export function stripWGSLComments(source) {
  let result = '', depth = 0, line = false;
  for (let i = 0; i < source.length; i++) {
    const a = source[i], b = source[i+1];
    if (line) { if (a === '\n') { line = false; result += '\n'; } continue; }
    if (depth) {
      if (a === '/' && b === '*') { depth++; i++; }
      else if (a === '*' && b === '/') { depth--; i++; }
      continue;
    }
    if (a === '/' && b === '*') { depth++; i++; }
    else if (a === '/' && b === '/') { line = true; i++; }
    else result += a;
  }
  if (depth) throw new Error('Unclosed nested WGSL comment.');
  return result;
}
export function shaderData(source, symbol = 'Q') {
  const clean = stripWGSLComments(source);
  const match = clean.match(new RegExp(`const ${symbol}\\s*=\\s*array<u32,\\s*[^>]+>\\(([^)]*)\\);`));
  if (!match) throw new Error(`Missing ${symbol} constant array.`);
  return match[1].split(',').filter(s=>s.trim()).map(s => Number(s.trim().replace(/u$/, '')));
}
export function referenceExpansion(source, target = 'self') {
  const q = shaderData(source);
  const clean = stripWGSLComments(source);
  const count = Number(clean.match(/const COUNT\s*=\s*(\d+)u?;/)?.[1]);
  const cut = Number(clean.match(/const CUT\s*=\s*(\d+)u?;/)?.[1]);
  if (count !== q.length || !Number.isSafeInteger(cut)) throw new Error('Bad emitter metadata.');
  const text = String.fromCharCode(...q);
  const split = clean.includes('fn javascript(');
  const first = split && target !== 'js' ? cut : 0;
  const last = split && target === 'js' ? cut : q.length;
  const replacement = new Map([['D',q.join(',')],['N',String(q.length)],['J',String(cut)]]);
  // Independent whole-string substitution, not the shader's character writer.
  return text.slice(first,last).replace(/~([DNJ])/g, (_,letter)=>replacement.get(letter));
}

// A reference interpreter for Sprout source, independent of both compiler backends.
export function interpret(program, input = program, target = 0, stepLimit = 20000000) {
  const r = new Uint32Array(64); r[0]=target; r[1]=input.length;
  const start = 3+program[1], end = program.length;
  const pairs = new Map(), alternatives = new Map(), stack = [];
  for(let pc=start;pc<end;pc+=4) {
    const op=program[pc];
    if(op===14 || op===16) stack.push([op,pc]);
    else if(op===17) { const top=stack.at(-1); if(top?.[0]!==16) throw Error('Bad else'); alternatives.set(top[1],pc); }
    else if(op===15 || op===18) {
      const [kind,begin]=stack.pop()||[];
      if(kind!==(op===15?14:16)) throw Error('Bad block');
      pairs.set(begin,pc);pairs.set(pc,begin);
      if(alternatives.has(begin)) pairs.set(alternatives.get(begin),pc);
    }
  }
  if(stack.length)throw Error('Unclosed block');
  let pc=start, steps=0; const result=[];
  const at=(a,i)=>i<a.length?a[i]:0;
  while(pc<end) {
    if(++steps>stepLimit)throw Error('Sprout reference step budget exhausted');
    const [op,a,b,c]=program.slice(pc,pc+4);
    switch(op) {
      case 0:break;
      case 1:r[a]=b;break;
      case 2:r[a]=r[b];break;
      case 3:r[a]=r[b]+r[c];break;
      case 4:r[a]=r[b]-r[c];break;
      case 5:r[a]=Number((BigInt(r[b])*BigInt(r[c])) & 0xffffffffn);break;
      case 6:r[a]=r[c]?Math.floor(r[b]/r[c]):0;break;
      case 7:r[a]=r[c]?r[b]%r[c]:0;break;
      case 8:r[a]=+(r[b]===r[c]);break;
      case 9:r[a]=+(r[b]<r[c]);break;
      case 10:r[a]=at(program,r[b]);break;
      case 11:r[a]=at(input,r[b]);break;
      case 12:if(r[a] > 255)throw Error('Non-ASCII output');result.push(String.fromCharCode(r[a]));break;
      case 13:result.push(String(r[a]));break;
      case 14:if(!r[a])pc=pairs.get(pc);break;
      case 15:pc=pairs.get(pc)-4;break;
      case 16:if(!r[a])pc=alternatives.get(pc)??pairs.get(pc);break;
      case 17:pc=pairs.get(pc);break;
      case 18:break;
      default:throw Error('Unknown Sprout opcode');
    }
    pc+=4;
  }
  return { text:result.join(''), steps };
}

export function referencePixels(text, shader) {
  const clean=stripWGSLComments(shader);
  const font=clean.match(/const FONT = array<u32, 896>\(([^)]*)\);/)[1].split(',').map(Number);
  const width=1024,height=Math.ceil((text.length+1)/128)*10;
  const rgba=new Uint8ClampedArray(width*height*4);
  for(let cell=0;cell<128*height/10;cell++) {
    const c=cell===0?35:cell<=text.length?text.charCodeAt(cell-1):32;
    const row=Math.floor(cell/128),col=cell%128;
    for(let y=0;y<10;y++)for(let x=0;x<8;x++) {
      let value=0xff19130f;
      if(x>=1&&x<=5&&y>=1&&y<=7&&((font[c*7+y-1]>>(5-x))&1))
        value=c>=48&&c<=57?0xff95eac3:0xffdfd6c9;
      if(x===7&&y===9)value=cell===0?(0xff000000|text.length):(0xff5aa500|c);
      const p=4*((row*10+y)*width+col*8+x);
      rgba[p]=value&255;rgba[p+1]=(value>>>8)&255;rgba[p+2]=(value>>>16)&255;rgba[p+3]=255;
    }
  }
  return {rgba,width,height};
}
