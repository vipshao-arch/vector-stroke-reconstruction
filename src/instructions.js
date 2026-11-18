export const KEYFRAMES = 13;
export const PARAMS_PER_STROKE = 12;
export const COLOR_COMPONENTS = 3;
export const KEYFRAME_FIELDS = 8;
export const STROKE_BYTES = 8 + PARAMS_PER_STROKE * 4 + COLOR_COMPONENTS * 4 + KEYFRAMES * KEYFRAME_FIELDS * 4;

export function parseInstructions(buffer) {
  const view = new DataView(buffer);
  if (buffer.byteLength < 20) throw new Error('指令文件过短');
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 8));
  if (magic !== 'VSTK0003') throw new Error('绘图指令文件标识无效');
  let offset = 8;
  const version = view.getUint16(offset, true); offset += 2;
  const keyCount = view.getUint16(offset, true); offset += 2;
  const jsonLength = view.getUint32(offset, true); offset += 4;
  const strokeCount = view.getUint32(offset, true); offset += 4;
  if (version !== 1 || keyCount !== KEYFRAMES) throw new Error('绘图指令版本不兼容');
  if (offset + jsonLength + KEYFRAMES * 6 * 4 > buffer.byteLength) {
    throw new Error(`指令元信息越界：${offset} + ${jsonLength} + ${KEYFRAMES * 6 * 4} > ${buffer.byteLength}`);
  }
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, offset, jsonLength)));
  offset += jsonLength;

  const camera = new Float32Array(KEYFRAMES * 6);
  new Uint8Array(camera.buffer).set(new Uint8Array(buffer, offset, KEYFRAMES * 6 * 4));
  offset += KEYFRAMES * 6 * 4;

  const params = new Float32Array(strokeCount * 16);
  const colors = new Float32Array(strokeCount * 4);
  const keys = new Float32Array(strokeCount * KEYFRAMES * KEYFRAME_FIELDS);
  const layerCounts = [0, 0, 0, 0, 0];
  for (let i = 0; i < strokeCount; i++) {
    if (offset + STROKE_BYTES > buffer.byteLength) {
      throw new Error(`笔触 ${i} 的数据越界：${offset} + ${STROKE_BYTES} > ${buffer.byteLength}`);
    }
    if (offset % 4 !== 0) throw new Error(`笔触 ${i} 的数据未对齐：${offset}`);
    const kind = view.getUint8(offset);
    const alpha = view.getFloat32(offset + 4, true);
    offset += 8;
    layerCounts[kind] += 1;
    const p = new Float32Array(PARAMS_PER_STROKE);
    new Uint8Array(p.buffer).set(new Uint8Array(buffer, offset, PARAMS_PER_STROKE * 4));
    offset += PARAMS_PER_STROKE * 4;
    const c = new Float32Array(COLOR_COMPONENTS);
    new Uint8Array(c.buffer).set(new Uint8Array(buffer, offset, COLOR_COMPONENTS * 4));
    offset += COLOR_COMPONENTS * 4;
    const k = new Float32Array(KEYFRAMES * KEYFRAME_FIELDS);
    new Uint8Array(k.buffer).set(new Uint8Array(buffer, offset, KEYFRAMES * KEYFRAME_FIELDS * 4));
    offset += KEYFRAMES * KEYFRAME_FIELDS * 4;
    params.set([kind / 255, 0, 0, alpha, ...p], i * 16);
    colors.set([c[0] / 255, c[1] / 255, c[2] / 255, 1], i * 4);
    keys.set(k, i * KEYFRAMES * KEYFRAME_FIELDS);
  }
  if (offset !== buffer.byteLength) {
    throw new Error(`指令长度异常：读取 ${offset} 字节，实际 ${buffer.byteLength} 字节`);
  }
  return { meta, camera, params, colors, keys, layerCounts, strokeCount, bytes: buffer.byteLength };
}
