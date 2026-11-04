import fs from 'node:fs/promises';
import path from 'node:path';
import { hashFile } from './resident-snapshot.js';

const PLAN_SCHEMA = 'doppler.gpu-benchmark-memory-plan/v1';

function bytes(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${label} must be nonnegative safe integer bytes.`);
  return value;
}

async function counter(file) {
  const value = (await fs.readFile(file, 'utf8')).trim();
  if (!/^\d+$/.test(value)) throw new Error(`Invalid driver memory counter: ${file}`);
  return bytes(Number(value), file);
}

// A reservation is an incremental bound above the allocations already observed
// on the device. Concurrent groups may be reserved together; their caller owns
// the bound, model/config scope and evidence. This does not choose a benchmark
// configuration, evict resources, launch processes or change inference policy.
export async function assessGpuBenchmarkMemory(plan, baseDirectory) {
  if (!plan || plan.schema !== PLAN_SCHEMA || typeof baseDirectory !== 'string'
    || !path.isAbsolute(baseDirectory) || typeof plan.sysfsDevicePath !== 'string'
    || !path.isAbsolute(plan.sysfsDevicePath) || !Array.isArray(plan.reservations)
    || plan.reservations.length === 0) {
    throw new Error('A GPU memory plan, absolute base directory and explicit driver path are required.');
  }
  const headroomBytes = bytes(plan.headroomBytes, 'headroomBytes');
  const ids = new Set(), reservations = [];
  let reservationBytes = 0;
  for (const reservation of plan.reservations) {
    if (!reservation || typeof reservation.id !== 'string' || !reservation.id || ids.has(reservation.id)
      || typeof reservation.evidence?.path !== 'string' || !reservation.evidence.path
      || !/^[a-f0-9]{64}$/.test(reservation.evidence.sha256)) {
      throw new Error('Each reservation requires a unique ID and pinned evidence path/SHA256.');
    }
    ids.add(reservation.id);
    const amount = bytes(reservation.bytes, `reservation ${reservation.id}`);
    const file = path.resolve(baseDirectory, reservation.evidence.path);
    if (await hashFile(file) !== reservation.evidence.sha256) {
      throw new Error(`GPU reservation evidence changed: ${reservation.id}`);
    }
    reservationBytes = bytes(reservationBytes + amount, 'total reservation');
    reservations.push({ id: reservation.id, bytes: amount, evidence: reservation.evidence });
  }
  const [capacityBytes, usedBytes] = await Promise.all([
    counter(path.join(plan.sysfsDevicePath, 'mem_info_gtt_total')),
    counter(path.join(plan.sysfsDevicePath, 'mem_info_gtt_used')),
  ]);
  if (capacityBytes === 0) throw new Error('Driver GTT capacity must be positive.');
  const projectedBytes = bytes(usedBytes + reservationBytes + headroomBytes, 'projected allocation');
  return {
    schema: 'doppler.gpu-benchmark-memory-admission/v1',
    admitted: projectedBytes <= capacityBytes,
    scope: 'Declared incremental allocations against current driver GTT; bounds require matching workload evidence.',
    sysfsDevicePath: plan.sysfsDevicePath,
    capacityBytes, observedUsedBytes: usedBytes, reservationBytes, headroomBytes, projectedBytes,
    remainingBytes: capacityBytes - projectedBytes,
    reservations,
  };
}
