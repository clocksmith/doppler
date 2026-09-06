import { normalizePackServePolicy } from '../../config/pack-serve.js';
import { PACK_OPERATIONS, snapshotPackOperationRequest } from '../../config/pack-operation.js';
import { assertQualifiedTargetOperation } from '../../config/target-plan.js';
import { createPackRequestAuthorization, endPackError, PackServeError, readPackRequest, writePackEvent } from './pack-http.js';

export function createPackServeHandler({ session, policy: inputPolicy, token }) {
  const policy = normalizePackServePolicy(inputPolicy);
  const authorize = createPackRequestAuthorization(token, policy);
  if (session?.schema !== 'doppler.pack-session/v1' || session.closed || typeof session.executeOperation !== 'function') {
    throw new Error('Pack serving requires an open, application-owned Pack session.');
  }
  const operations = Object.keys(PACK_OPERATIONS).filter(operation => {
    try { assertQualifiedTargetOperation(session.selectedPlan, session.deviceProfile.surface, operation); return true; }
    catch { return false; }
  });
  let active = null;
  let closed = false;
  const handler = async (req, res) => {
    let controller;
    let timer;
    let finish;
    let responseBytes = 0;
    const disconnect = () => controller?.abort(new PackServeError('CLIENT_DISCONNECTED', 'Client disconnected.', 499));
    try {
      authorize(req, res);
      if (closed || session.closed) throw new PackServeError('SESSION_CLOSED', 'Pack serving session is closed.', 503);
      const pathname = new URL(req.url, 'http://localhost').pathname;
      if (pathname !== '/v1/operations' && pathname !== '/v1/model') {
        throw new PackServeError('NOT_FOUND', 'Unknown Pack serving endpoint.', 404);
      }
      if (req.method === 'OPTIONS') {
        res.writeHead(204, {
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        });
        res.end();
        return;
      }
      if (pathname === '/v1/model' && req.method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
        res.end(JSON.stringify({ modelId: session.modelId, pack: session.packIdentity,
          targetPlanDigest: session.selectedTargetPlanDigest, operations }));
        return;
      }
      if (pathname !== '/v1/operations' || req.method !== 'POST') {
        throw new PackServeError('METHOD_NOT_ALLOWED', 'Method is not supported for this endpoint.', 405);
      }
      if (active) throw new PackServeError('SESSION_BUSY', 'One operation is already active; retry after it finishes.', 409);
      if (req.headers['content-type']?.split(';')[0].trim() !== 'application/json') {
        throw new PackServeError('CONTENT_TYPE', 'Content-Type must be application/json.', 415);
      }
      controller = new AbortController();
      active = { controller, done: new Promise(resolve => { finish = resolve; }) };
      res.once('close', disconnect);
      req.once('aborted', disconnect);
      const deadlineAt = Date.now() + policy.maxDurationMs;
      timer = setTimeout(() => controller.abort(new PackServeError('DEADLINE_EXCEEDED', 'Serving deadline exceeded.', 408)), policy.maxDurationMs);
      const input = await readPackRequest(req, policy.maxRequestBytes, controller.signal);
      let request;
      try { request = snapshotPackOperationRequest(input); }
      catch (error) { throw new PackServeError('INVALID_OPERATION', error.message, 400); }
      if (request.limits.maxInputBytes > policy.maxRequestBytes || request.limits.maxOutputBytes > policy.maxOutputBytes
        || request.limits.deadlineAt > deadlineAt) {
        throw new PackServeError('LIMIT_EXCEEDED', 'Operation limits exceed the serving policy; no limits were silently rewritten.', 400);
      }
      if (request.assignment !== null) {
        throw new PackServeError('DELEGATION_UNSUPPORTED', 'This local serving adapter does not authorize delegated assignments.', 400);
      }
      if (!operations.includes(request.operation.name)) {
        throw new PackServeError('OPERATION_UNQUALIFIED', 'Selected implementation is not qualified for this operation.', 422);
      }
      for await (const event of session.executeOperation(request, { signal: controller.signal })) {
        controller.signal.throwIfAborted();
        const line = `${JSON.stringify(event)}\n`;
        const lineBytes = Buffer.byteLength(line);
        if (responseBytes + lineBytes > policy.maxResponseBytes) {
          throw new PackServeError('RESPONSE_TOO_LARGE', 'Event stream exceeds maxResponseBytes.', 413);
        }
        responseBytes += lineBytes;
        if (!res.headersSent) res.writeHead(200, { 'Content-Type': 'application/x-ndjson', 'Cache-Control': 'no-store' });
        await writePackEvent(res, line, controller.signal);
      }
      controller.signal.throwIfAborted();
      res.end();
    } catch (error) { endPackError(res, error, policy.maxResponseBytes - responseBytes); }
    finally {
      if (controller) {
        controller.abort(new PackServeError('REQUEST_FINISHED', 'Serving request finished.', 499));
        clearTimeout(timer);
        req.off('aborted', disconnect);
        res.off('close', disconnect);
        active = null;
        finish();
      }
    }
  };
  handler.close = async () => {
    closed = true;
    const pending = active;
    pending?.controller.abort(new PackServeError('SERVER_CLOSED', 'Pack serving stopped.', 503));
    await pending?.done;
    // The application owns the session and closes it only after all borrowers drain.
  };
  return handler;
}
