import { acquireBuffer, releaseBuffer, discardBuffer, uploadData } from '../../../../memory/buffer-pool.js';
import { createLinearAttentionCheckpoint } from '../linear-attention/state-checkpoint.js';
import { WholeTokenProgram } from './whole-token-program.js';
import { getDevice } from '../../../../gpu/device.js';

/** @type {import('./compiled-greedy-window.js').createCompiledGreedyWindow} */
export function createCompiledGreedyWindow(options) {
  const { device, session, runtime, draftCount: D, vocabSize: V, hiddenSize: H } = options;
  if (device!==getDevice() || ![D,V,H].every(n=>Number.isSafeInteger(n)&&n>0) || D>32
    || typeof options.identityOf!=='function' || !Array.isArray(options.positionLimits)
    || session.carry.hidden.shape[0]!==1 || session.carry.hidden.shape[1]!==H) {
    throw new Error('Compiled greedy windows require explicit bounded geometry, carry and identity.');
  }
  /** @type {GPUBuffer[]} */
  const owned=[];
  /** @param {number} bytes */
  const alloc=bytes=>{const b=acquireBuffer(bytes);owned.push(b);return b;};
  let flat,checkpoint,cap,savedHidden,savedToken;
  try {
    cap=alloc(4);savedHidden=alloc(H*4);savedToken=alloc(4);
    checkpoint=createLinearAttentionCheckpoint(runtime);
    flat=new WholeTokenProgram(device,{tokenInput:session.carry.token,maxOutputTokens:D+1});
  } catch(error) {checkpoint?.close();for(const b of owned)releaseBuffer(b);throw error;}
  const compiled=flat,snapshot=checkpoint,limitBuffer=cap,hiddenCopy=savedHidden,tokenCopy=savedToken;
  let busy=false,closed=false,poisoned=false;
  /** @param {Uint32Array} words @param {number} limit */
  const validate=(words,limit)=>{
    const count=words[0],accepted=words[1];
    if(words.length!==D+10 || words[2]!==0 || words[D+6]!==0 || count<1 || count>limit
      || accepted>D || accepted>count || words[3]!==count-1 || words[4]>=V) {
      throw new Error('Compiled greedy acceptance or finite-output validation failed.');
    }
    const tokens=Object.freeze(Array.from(words.slice(5,5+count)));
    if(tokens.length!==count || tokens.some(id=>id>=V)) throw new Error('Compiled greedy output tokens are invalid.');
    return {tokens,accepted};
  };
  /** @type {import('./compiled-greedy-window.js').CompiledGreedyWindow} */
  const controller={
    get ready(){return compiled.ready;},
    get stats(){return {...compiled.stats,privateBytes:compiled.privateBytes};},
    async beginGeneration(initialHidden, initialToken){
      if(closed || poisoned || busy)throw new Error('Starting a generation requires a live idle compiled controller.');
      busy=true;
      try{
        await session.reset(initialHidden,initialToken);
        if(compiled.ready)compiled.beginGeneration();else compiled.invalidate('new generation has no compiled window');
      }catch(error){poisoned=true;throw error;}
      finally{busy=false;}
    },
    async begin(program,position,outputLimit,signal){
      if(closed || poisoned || busy || !Number.isSafeInteger(position) || position<0
        || !Number.isSafeInteger(outputLimit) || outputLimit<1 || outputLimit>D+1
        || typeof program?.setReplayedPrefix!=='function') {
        throw new Error('Compiled greedy window requires a live idle controller and explicit prefix ports.');
      }
      signal?.throwIfAborted();
      const identity=options.identityOf();
      if(!Array.isArray(identity))throw new Error('Compiled greedy identity must be an explicit resource list.');
      busy=true;
      try{
        uploadData(limitBuffer,new Uint32Array([outputLimit]));
        if(compiled.awaitingVerification){
          const stop=Object.freeze({owner:'compiled greedy verification before submission'});
          await compiled.verifyAhead(position,0,attach=>session.begin(program,D+1,signal,{outputLimitBuffer:limitBuffer,
            observe(recorder){attach(recorder);recorder.submit=()=>{throw stop;};}}),options.identityOf,options.positionLimits,stop);
        }
        if(compiled.ready){const reason=compiled.rejects(position,identity);if(reason)compiled.invalidate(reason);}
      }catch(error){busy=false;throw error;}
      if(!compiled.ready){
        /** @type {import('../../../../gpu/greedy-speculative-frame.js').GreedyFrameTransaction | undefined} */
        let observedTransaction;
        try {
          const transaction=await compiled.observe(position,0,async attach=>{
            observedTransaction=await session.begin(program,D+1,signal,{observe:attach,outputLimitBuffer:limitBuffer});
            return observedTransaction;
          },options.identityOf,options.positionLimits);
          if(compiled.ready)compiled.acceptObservedWindow(position,transaction.tokens.length);
          let published=0;
          /** @type {Promise<void> | undefined} */
          let settlement;
          return {tokens:transaction.tokens,accepted:transaction.accepted,
            publish(){transaction.publish();published++;},
            settle(){settlement??=(async()=>{
              try {await transaction.settle();if(compiled.ready && published!==transaction.tokens.length)compiled.resumeAfterRestore(position+published);}
              catch(error){poisoned=true;throw error;}
              finally{busy=false;}
            })();return settlement;},
          };
        }catch(error){
          try{await observedTransaction?.settle();compiled.invalidate('observed greedy window failed');}
          catch(cleanup){poisoned=true;throw new AggregateError([error,cleanup],'Observed greedy window cleanup failed.');}
          finally{busy=false;}
          throw error;
        }
      }
      let submitted=false;
      /** @param {GPUCommandEncoder} encoder */
      const capture=encoder=>{
        snapshot.recordCapture(encoder);
        encoder.copyBufferToBuffer(session.carry.hidden.buffer,0,hiddenCopy,0,H*4);
        encoder.copyBufferToBuffer(session.carry.token,0,tokenCopy,0,4);
      };
      const restore=async()=>{
        await device.queue.onSubmittedWorkDone();
        const encoder=device.createCommandEncoder({label:'compiled_greedy_restore'});
        snapshot.recordRestore(encoder);
        encoder.copyBufferToBuffer(hiddenCopy,0,session.carry.hidden.buffer,0,H*4);
        encoder.copyBufferToBuffer(tokenCopy,0,session.carry.token,0,4);
        device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
        compiled.resumeAfterRestore(position);await program.setPrefix(0);
      };
      try {
        const words=await compiled.runWindow(position,{identity,recordPrefix:capture,onSubmitted(){submitted=true;}});
        const result=validate(words,outputLimit);
        await program.setReplayedPrefix(result.tokens.length);signal?.throwIfAborted();
        let published=0;
        /** @type {Promise<void> | undefined} */
        let settlement;
        return {...result,
          publish(){signal?.throwIfAborted();if(settlement || published>=result.tokens.length)throw new Error('Invalid compiled greedy publication.');published++;},
          settle(){settlement??=(async()=>{
            try {
              if(published!==result.tokens.length){
                await restore();
                if(published){
                  uploadData(limitBuffer,new Uint32Array([published]));
                  const replay=validate(await compiled.runWindow(position,{identity,recordPrefix(){},onSubmitted(){}}),published);
                  if(replay.tokens.length!==published || replay.tokens.some((token,i)=>token!==result.tokens[i]))throw new Error('Compiled greedy prefix replay changed published tokens.');
                  await program.setReplayedPrefix(published);
                }
              }
            }catch(error){
              poisoned=true;
              try{await restore();}catch(cleanup){throw new AggregateError([error,cleanup],'Compiled greedy settlement and rollback failed.');}
              throw error;
            }finally{busy=false;}
          })();return settlement;},
        };
      }catch(error){
        try{if(submitted)await restore();}
        catch(cleanup){poisoned=true;throw new AggregateError([error,cleanup],'Compiled greedy execution and rollback failed.');}
        finally{busy=false;}
        throw error;
      }
    },
    async dispose(){
      if(closed)return;if(busy)throw new Error('Settle the outstanding compiled greedy window before disposal.');closed=true;
      let lost=false;
      try{await device.queue.onSubmittedWorkDone();}catch(error){lost=true;throw error;}
      finally{compiled.destroy();snapshot.close(lost);for(const b of owned)(lost?discardBuffer:releaseBuffer)(b);}
    },
  };
  return controller;
}
