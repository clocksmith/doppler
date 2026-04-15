const SUPPORTED_SOURCE_DTYPES = new Set(['INT8', 'UINT8', 'INT4', 'INT2']);
const SUPPORTED_LITERT_STORAGE_ENCODINGS = new Set(['signed', 'offset_binary']);
const SUPPORTED_LITERT_SCALE_SEMANTICS = new Set(['step', 'qmax_abs']);

function fail(errorPrefix, message) {
  throw new Error(`${errorPrefix} ${message}`);
}

function normalizeSourceDtype(sourceDtype, tensorName, errorPrefix) {
  const normalized = String(sourceDtype || '').trim().toUpperCase();
  if (!SUPPORTED_SOURCE_DTYPES.has(normalized)) {
    fail(
      errorPrefix,
      `Tensor "${tensorName}" has unsupported sourceTransform.sourceDtype "${sourceDtype}".`
    );
  }
  return normalized;
}

function normalizeTargetDtype(location, targetDtype, tensorName, errorPrefix, label = 'sourceTransform') {
  const normalizedTargetDtype = String(targetDtype || '').trim().toUpperCase();
  if (normalizedTargetDtype !== 'F16') {
    fail(
      errorPrefix,
      `Tensor "${tensorName}" has unsupported ${label}.targetDtype "${targetDtype}".`
    );
  }
  const normalizedLocationDtype = String(location?.dtype || '').trim().toUpperCase();
  if (normalizedLocationDtype !== normalizedTargetDtype) {
    fail(
      errorPrefix,
      `Tensor "${tensorName}" ${label} targetDtype "${normalizedTargetDtype}" ` +
      `does not match location.dtype "${location?.dtype}".`
    );
  }
  return normalizedTargetDtype;
}

function normalizePositiveNumber(value, tensorName, errorPrefix, field) {
  const normalized = Number(value);
  if (!Number.isFinite(normalized) || normalized <= 0) {
    fail(
      errorPrefix,
      `Tensor "${tensorName}" has invalid ${field} ${value}.`
    );
  }
  return normalized;
}

function normalizePositiveInteger(value, tensorName, errorPrefix, field) {
  const normalized = Number(value);
  if (!Number.isInteger(normalized) || normalized <= 0) {
    fail(
      errorPrefix,
      `Tensor "${tensorName}" has invalid ${field} ${value}.`
    );
  }
  return normalized;
}

function normalizeSafeInteger(value, tensorName, errorPrefix, field) {
  const normalized = Number(value);
  if (!Number.isSafeInteger(normalized)) {
    fail(
      errorPrefix,
      `Tensor "${tensorName}" has invalid ${field} ${value}.`
    );
  }
  return normalized;
}

function normalizeNonNegativeInteger(value, tensorName, errorPrefix, field) {
  const normalized = Number(value);
  if (!Number.isInteger(normalized) || normalized < 0) {
    fail(
      errorPrefix,
      `Tensor "${tensorName}" has invalid ${field} ${value}.`
    );
  }
  return normalized;
}

function normalizeStorageEncoding(storageEncoding, tensorName, errorPrefix, label) {
  const normalized = String(storageEncoding || '').trim();
  if (!SUPPORTED_LITERT_STORAGE_ENCODINGS.has(normalized)) {
    fail(
      errorPrefix,
      `Tensor "${tensorName}" has unsupported ${label} "${storageEncoding}".`
    );
  }
  return normalized;
}

function normalizeScaleSemantics(transform, tensorName, errorPrefix, label) {
  const scaleSemantics = String(transform?.scaleSemantics || '').trim().toLowerCase();
  if (!SUPPORTED_LITERT_SCALE_SEMANTICS.has(scaleSemantics)) {
    fail(
      errorPrefix,
      `Tensor "${tensorName}" ${label} has unsupported scaleSemantics "${transform?.scaleSemantics}".`
    );
  }
  const scaleDivisor = scaleSemantics === 'qmax_abs'
    ? normalizePositiveNumber(transform?.scaleDivisor, tensorName, errorPrefix, `${label} scaleDivisor`)
    : (transform?.scaleDivisor == null ? undefined : Number(transform.scaleDivisor));
  return {
    scaleSemantics,
    ...(scaleDivisor === undefined ? {} : { scaleDivisor }),
  };
}

function normalizeSourceRef(ref, tensorName, errorPrefix, label, required = false) {
  if (ref == null) {
    if (required) {
      fail(
        errorPrefix,
        `Tensor "${tensorName}" ${label} is missing.`
      );
    }
    return null;
  }
  if (typeof ref !== 'object') {
    fail(
      errorPrefix,
      `Tensor "${tensorName}" ${label} has invalid value.`
    );
  }
  const shard = ref.shardIndex ?? ref.shard;
  return {
    ...ref,
    shard: normalizeNonNegativeInteger(shard, tensorName, errorPrefix, `${label}.shard`),
    offset: normalizeNonNegativeInteger(ref.offset, tensorName, errorPrefix, `${label}.offset`),
    size: normalizePositiveInteger(ref.size, tensorName, errorPrefix, `${label}.size`),
  };
}

function normalizeStorageShape(storageShape, tensorName, errorPrefix, label) {
  if (!Array.isArray(storageShape) || storageShape.length !== 2) {
    fail(
      errorPrefix,
      `Tensor "${tensorName}" ${label} requires storageShape=[rows, cols].`
    );
  }
  const rows = normalizePositiveInteger(storageShape[0], tensorName, errorPrefix, `${label} storageShape[0]`);
  const cols = normalizePositiveInteger(storageShape[1], tensorName, errorPrefix, `${label} storageShape[1]`);
  return [rows, cols];
}

function normalizeLaneOrder(storageLaneOrder, storageBlockSize, tensorName, errorPrefix) {
  if (!Array.isArray(storageLaneOrder) || storageLaneOrder.length !== storageBlockSize) {
    fail(
      errorPrefix,
      `Tensor "${tensorName}" LiteRT blocked axis transform requires storageLaneOrder ` +
      `with ${storageBlockSize} entries.`
    );
  }
  const seen = new Set();
  return storageLaneOrder.map((value) => {
    const normalized = Number(value);
    if (
      !Number.isInteger(normalized)
      || normalized < 0
      || normalized >= storageBlockSize
      || seen.has(normalized)
    ) {
      fail(
        errorPrefix,
        `Tensor "${tensorName}" has invalid LiteRT storageLaneOrder ${JSON.stringify(storageLaneOrder)}.`
      );
    }
    seen.add(normalized);
    return normalized;
  });
}

function normalizeScaleCompanion(transform, tensorName, errorPrefix) {
  const scaleCompanionDtype = String(transform?.scaleCompanionDtype || '').trim().toUpperCase();
  if (!scaleCompanionDtype) {
    return {};
  }
  if (scaleCompanionDtype !== 'UINT8') {
    fail(
      errorPrefix,
      `Tensor "${tensorName}" LiteRT axis transform has unsupported scaleCompanionDtype "${transform?.scaleCompanionDtype}".`
    );
  }
  if (!transform?.scaleCompanionDequant || typeof transform.scaleCompanionDequant !== 'object') {
    fail(
      errorPrefix,
      `Tensor "${tensorName}" LiteRT axis transform has UINT8 scale companion without scaleCompanionDequant metadata.`
    );
  }
  return {
    scaleCompanionDtype,
    scaleCompanionDequant: {
      scale: normalizePositiveNumber(
        transform.scaleCompanionDequant.scale,
        tensorName,
        errorPrefix,
        'scaleCompanionDequant.scale'
      ),
      zeroPoint: normalizeSafeInteger(
        transform.scaleCompanionDequant.zeroPoint,
        tensorName,
        errorPrefix,
        'scaleCompanionDequant.zeroPoint'
      ),
    },
  };
}

export function normalizeTensorSourceTransform(location, tensorName, options = {}) {
  const transform = location?.sourceTransform;
  if (!transform || typeof transform !== 'object') {
    return null;
  }
  const errorPrefix = options.errorPrefix ?? '[DopplerLoader]';
  const kind = String(transform.kind || '').trim();
  const sourceDtype = normalizeSourceDtype(transform.sourceDtype, tensorName, errorPrefix);

  if (kind === 'affine_dequant') {
    if (transform.scheme !== 'per_tensor_affine') {
      fail(
        errorPrefix,
        `Tensor "${tensorName}" has unsupported sourceTransform.scheme "${transform.scheme}".`
      );
    }
    return {
      kind,
      scheme: transform.scheme,
      sourceDtype,
      targetDtype: normalizeTargetDtype(location, transform.targetDtype, tensorName, errorPrefix),
      scale: normalizePositiveNumber(transform.scale, tensorName, errorPrefix, 'sourceTransform.scale'),
      zeroPoint: normalizeSafeInteger(transform.zeroPoint, tensorName, errorPrefix, 'sourceTransform.zeroPoint'),
    };
  }

  if (kind === 'litert_rowwise_dequant') {
    if (transform.scheme !== 'per_row_affine') {
      fail(
        errorPrefix,
        `Tensor "${tensorName}" has unsupported sourceTransform.scheme "${transform.scheme}".`
      );
    }
    return {
      kind,
      scheme: transform.scheme,
      sourceDtype,
      targetDtype: normalizeTargetDtype(
        location,
        transform.targetDtype,
        tensorName,
        errorPrefix,
        'LiteRT row-wise sourceTransform'
      ),
      storageEncoding: normalizeStorageEncoding(
        transform.storageEncoding,
        tensorName,
        errorPrefix,
        'LiteRT storageEncoding'
      ),
      ...normalizeScaleSemantics(transform, tensorName, errorPrefix, 'LiteRT row-wise transform'),
      scaleSource: normalizeSourceRef(transform.scaleSource, tensorName, errorPrefix, 'LiteRT row-wise scaleSource', true),
      ...(transform.rowSumSource == null
        ? {}
        : {
          rowSumSource: normalizeSourceRef(
            transform.rowSumSource,
            tensorName,
            errorPrefix,
            'LiteRT row-wise rowSumSource'
          ),
        }),
    };
  }

  if (kind === 'litert_axis_dequant') {
    const quantAxis = Number(transform.quantAxis);
    if (quantAxis !== 0 && quantAxis !== 1) {
      fail(
        errorPrefix,
        `Tensor "${tensorName}" has invalid LiteRT quantAxis ${transform.quantAxis}.`
      );
    }
    return {
      kind,
      scheme: transform.scheme === 'per_axis_affine'
        ? transform.scheme
        : fail(
          errorPrefix,
          `Tensor "${tensorName}" has unsupported sourceTransform.scheme "${transform.scheme}".`
        ),
      sourceDtype,
      targetDtype: normalizeTargetDtype(
        location,
        transform.targetDtype,
        tensorName,
        errorPrefix,
        'LiteRT axis sourceTransform'
      ),
      storageEncoding: normalizeStorageEncoding(
        transform.storageEncoding,
        tensorName,
        errorPrefix,
        'LiteRT storageEncoding'
      ),
      ...normalizeScaleSemantics(transform, tensorName, errorPrefix, 'LiteRT axis transform'),
      storageShape: normalizeStorageShape(transform.storageShape, tensorName, errorPrefix, 'LiteRT axis transform'),
      quantAxis,
      scaleSource: normalizeSourceRef(transform.scaleSource, tensorName, errorPrefix, 'LiteRT axis scaleSource', true),
      ...normalizeScaleCompanion(transform, tensorName, errorPrefix),
      ...(transform.sumSource == null
        ? {}
        : {
          sumSource: normalizeSourceRef(
            transform.sumSource,
            tensorName,
            errorPrefix,
            'LiteRT axis sumSource'
          ),
        }),
    };
  }

  if (kind === 'litert_axis_blocked_dequant') {
    if (transform.scheme !== 'per_axis_affine') {
      fail(
        errorPrefix,
        `Tensor "${tensorName}" has unsupported sourceTransform.scheme "${transform.scheme}".`
      );
    }
    const quantAxis = Number(transform.quantAxis);
    if (quantAxis !== 0) {
      fail(
        errorPrefix,
        `Tensor "${tensorName}" LiteRT blocked axis transform only supports quantAxis=0.`
      );
    }
    const storageBlockSize = normalizePositiveInteger(
      transform.storageBlockSize,
      tensorName,
      errorPrefix,
      'LiteRT blocked axis storageBlockSize'
    );
    return {
      kind,
      scheme: transform.scheme,
      sourceDtype,
      targetDtype: normalizeTargetDtype(
        location,
        transform.targetDtype,
        tensorName,
        errorPrefix,
        'LiteRT blocked axis sourceTransform'
      ),
      storageEncoding: normalizeStorageEncoding(
        transform.storageEncoding,
        tensorName,
        errorPrefix,
        'LiteRT storageEncoding'
      ),
      ...normalizeScaleSemantics(transform, tensorName, errorPrefix, 'LiteRT blocked axis transform'),
      storageShape: normalizeStorageShape(transform.storageShape, tensorName, errorPrefix, 'LiteRT blocked axis transform'),
      quantAxis,
      storageBlockSize,
      storageLaneOrder: normalizeLaneOrder(transform.storageLaneOrder, storageBlockSize, tensorName, errorPrefix),
      scaleSource: normalizeSourceRef(
        transform.scaleSource,
        tensorName,
        errorPrefix,
        'LiteRT blocked axis scaleSource',
        true
      ),
      ...(transform.sumSource == null
        ? {}
        : {
          sumSource: normalizeSourceRef(
            transform.sumSource,
            tensorName,
            errorPrefix,
            'LiteRT blocked axis sumSource'
          ),
        }),
    };
  }

  fail(
    errorPrefix,
    `Tensor "${tensorName}" has unsupported sourceTransform.kind "${transform.kind}".`
  );
}
