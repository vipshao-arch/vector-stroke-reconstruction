import { readFile } from 'node:fs/promises';
import { parseInstructions, STROKE_BYTES } from '../src/instructions.js';

const file = process.argv[2] || 'assets/strokes.bin';
const buffer = await readFile(file);
const typed = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
const result = parseInstructions(typed);
const finite = (array) => array.every ? array.every(Number.isFinite) : Array.from(array).every(Number.isFinite);
const keys = result.keys;
const fields = 8;
let scaleMin = Infinity;
let scaleMax = -Infinity;
let rotateMin = Infinity;
let rotateMax = -Infinity;
for (let i = 0; i < result.strokeCount; i += 1) {
  for (let k = 0; k < 13; k += 1) {
    const base = (i * 13 + k) * fields;
    const scale = keys[base + 2];
    const rotate = keys[base + 3];
    scaleMin = Math.min(scaleMin, scale);
    scaleMax = Math.max(scaleMax, scale);
    rotateMin = Math.min(rotateMin, rotate);
    rotateMax = Math.max(rotateMax, rotate);
  }
}
console.log(JSON.stringify({
  file,
  magic: buffer.subarray(0, 8).toString(),
  bytes: result.bytes,
  strokeBytes: STROKE_BYTES,
  strokes: result.strokeCount,
  layers: result.layerCounts,
  duration: result.meta.duration,
  finite: finite(result.params) && finite(result.colors) && finite(result.keys) && finite(result.camera),
  scaleRange: [scaleMin, scaleMax],
  rotateRange: [rotateMin, rotateMax],
  hasLineStrokes: result.layerCounts[1] > 0,
}, null, 2));
