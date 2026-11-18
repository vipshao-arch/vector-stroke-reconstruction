import * as THREE from './three.module.js?v=3';
import { KEYFRAMES, KEYFRAME_FIELDS, parseInstructions } from './instructions.js?v=3';

const VERTEX_SHADER = /* glsl */ `
  precision highp float;
  attribute vec3 position;
  attribute float aIndex;
  uniform sampler2D uParams;
  uniform sampler2D uColor;
  uniform sampler2D uKeys;
  uniform vec2 uParamSize;
  uniform vec2 uColorSize;
  uniform vec2 uKeySize;
  uniform float uTime;
  uniform vec2 uLogicalSize;
  uniform vec3 uCameraA;
  uniform vec3 uCameraB;
  varying vec2 vLocal;
  varying vec2 vStart;
  varying vec2 vControl;
  varying vec2 vEnd;
  varying vec2 vSize;
  varying float vWidth;
  varying vec3 vColor;
  varying float vAlpha;
  varying float vKind;

  vec4 texel(sampler2D tex, vec2 size, float offset) {
    float width = max(size.x, 1.0);
    float height = max(size.y, 1.0);
    float index = clamp(offset, 0.0, width * height - 1.0);
    float x = mod(index, width);
    float y = floor(index / width);
    return texture2D(tex, (vec2(x, y) + 0.5) / size);
  }

  float readKey(float stroke, float field, float key) {
    float offset = (stroke * ${KEYFRAMES}.0 + key) * ${KEYFRAME_FIELDS}.0 + field;
    float texelIndex = floor(offset * 0.25);
    float component = mod(offset, 4.0);
    vec4 value = texel(uKeys, uKeySize, texelIndex);
    if (component < 0.5) return value.r;
    if (component < 1.5) return value.g;
    if (component < 2.5) return value.b;
    return value.a;
  }

  vec4 interpolateKey(float stroke, float field) {
    float position = clamp(uTime, 0.0, 1.0) * ${KEYFRAMES - 1}.0;
    float key = floor(position);
    float blend = smoothstep(0.0, 1.0, position - key);
    float a = readKey(stroke, field, key);
    float b = readKey(stroke, field, min(key + 1.0, ${KEYFRAMES - 1}.0));
    return vec4(mix(a, b, blend));
  }

  void main() {
    float stroke = aIndex;
    vec4 block = texel(uParams, uParamSize, stroke * 4.0);
    float kind = block.r * 255.0;
    float alphaBase = block.a;
    vec4 p0 = texel(uParams, uParamSize, stroke * 4.0 + 1.0);
    vec4 p1 = texel(uParams, uParamSize, stroke * 4.0 + 2.0);
    vec4 ink = texel(uColor, uColorSize, stroke);

    float dx = interpolateKey(stroke, 0.0).r;
    float dy = interpolateKey(stroke, 1.0).r;
    float scale = interpolateKey(stroke, 2.0).r;
    float rotate = interpolateKey(stroke, 3.0).r;
    float deltaAlpha = interpolateKey(stroke, 4.0).r;
    float brightness = interpolateKey(stroke, 5.0).r;
    float wobble = interpolateKey(stroke, 7.0).r;

    vec2 origin;
    vec2 size;
    vec2 control;
    vec2 end;
    vec2 start;
    float rotation;
    if (kind < 0.5) {
      // p0 = center x/y, ellipse radii x/y; p1.x = rotation.
      origin = p0.xy;
      size = p0.zw;
      rotation = p1.x;
      control = vec2(0.0);
      end = vec2(0.0);
      start = vec2(0.0);
    } else if (kind > 0.5 && kind < 2.5) {
      // p0/p1 = line or quadratic Bezier start, control, end, line width.
      vec2 a = p0.xy;
      vec2 b = p0.zw;
      vec2 c = p1.xy;
      origin = (a + b + c) / 3.0;
      float reach = max(max(length(a - origin), length(b - origin)), length(c - origin));
      size = vec2(reach + p1.z * 2.0, reach + p1.z * 2.0);
      control = b - origin;
      end = c - origin;
      start = a - origin;
      rotation = 0.0;
    } else {
      // p0.x/y = round brush center; p0.z = radius.
      origin = p0.xy;
      size = vec2(p0.z, p0.z);
      control = vec2(0.0);
      end = vec2(0.0);
      start = vec2(0.0);
      rotation = 0.0;
    }
    vControl = control;
    vEnd = end;
    vStart = start;
    vKind = kind;
    vSize = size;
    vColor = clamp(ink.rgb * brightness, 0.0, 1.0);
    vAlpha = clamp(alphaBase + deltaAlpha, 0.0, 1.0);

    // Convert logical video coordinates into normalized drawing coordinates.
    rotation += rotate;
    float c = cos(rotation);
    float s = sin(rotation);
    mat2 turn = mat2(c, s, -s, c);
    vec2 quadPosition = position.xy;
    vec2 local = (quadPosition - 0.5) * size * 2.2;
    local = turn * local * max(scale, 0.01);
    start = turn * start * max(scale, 0.01);
    control = turn * control * max(scale, 0.01);
    end = turn * end * max(scale, 0.01);
    vec2 world = local + origin;
    world = vec2(
      dot(uCameraA.xy, world) + uCameraA.z,
      dot(uCameraB.xy, world) + uCameraB.z
    ) + vec2(dx + wobble * 0.18, dy - wobble * 0.12);
    vLocal = local;
    vSize = size * max(scale, 0.01);
    vWidth = max(p1.z * max(scale, 0.01), 0.55);
    vStart = start;
    vControl = control;
    vEnd = end;

    vec2 clip = (world / uLogicalSize) * 2.0 - 1.0;
    gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  }
`;

const FRAGMENT_SHADER = /* glsl */ `
  precision highp float;
  varying vec2 vLocal;
  varying vec2 vStart;
  varying vec2 vControl;
  varying vec2 vEnd;
  varying vec2 vSize;
  varying float vWidth;
  varying vec3 vColor;
  varying float vAlpha;
  varying float vKind;

  float softCircle(vec2 p, float radius) {
    float d = length(p);
    return 1.0 - smoothstep(radius * 0.55, radius, d);
  }

  float segmentDistance(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-5), 0.0, 1.0);
    return length(pa - ba * h);
  }

  void main() {
    float coverage = 0.0;
    if (vKind < 0.5) {
      // Oriented elliptical paint stroke.
      vec2 unit = vec2(vLocal.x / max(vSize.x, 0.01), vLocal.y / max(vSize.y, 0.01));
      float q = length(unit);
      coverage = 1.0 - smoothstep(0.82, 1.0, q);
    } else if (vKind < 1.5) {
      // Straight line segment with round caps.
      float d = segmentDistance(vLocal, vStart, vEnd);
      coverage = 1.0 - smoothstep(vWidth * 0.55, vWidth, d);
    } else if (vKind < 2.5) {
      // Quadratic Bezier stroke sampled as two line segments.
      float width = vWidth;
      vec2 start = vStart;
      vec2 end = vEnd;
      float d = min(
        segmentDistance(vLocal, start, vControl),
        segmentDistance(vLocal, vControl, end)
      );
      coverage = 1.0 - smoothstep(width * 0.55, width, d);
    } else if (vKind < 3.5) {
      // Temporal shimmer: a soft elliptical glow.
      float q = length(vLocal / max(vSize.x, 0.01));
      coverage = 1.0 - smoothstep(0.0, 1.0, q);
    } else {
      // Soft residual brush.
      coverage = softCircle(vLocal, max(vSize.x, 0.5));
    }
    float alpha = coverage * vAlpha;
    if (alpha < 0.002) discard;
    gl_FragColor = vec4(vColor * alpha, alpha);
  }
`;

const BACKGROUND_VERTEX_SHADER = /* glsl */ `
  precision highp float;
  attribute vec3 position;
  varying vec2 vScreen;
  void main() {
    vScreen = position.xy * 0.5 + 0.5;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

const BACKGROUND_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;
  uniform sampler2D uBase;
  uniform vec2 uLogicalSize;
  uniform vec3 uBackgroundA;
  uniform vec3 uBackgroundB;
  varying vec2 vScreen;
  void main() {
    vec2 framePoint = vec2(vScreen.x * uLogicalSize.x, (1.0 - vScreen.y) * uLogicalSize.y);
    vec2 sourcePoint = vec2(
      dot(uBackgroundA.xy, framePoint) + uBackgroundA.z,
      dot(uBackgroundB.xy, framePoint) + uBackgroundB.z
    );
    gl_FragColor = texture2D(uBase, clamp(sourcePoint / uLogicalSize, 0.0, 1.0));
  }
`;

function makeDataTexture(data, width, height, renderer) {
  const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, THREE.FloatType);
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  renderer.initTexture(texture);
  return texture;
}

function paddedTextureDimensions(floats, maxSize) {
  const texels = Math.ceil(floats.length / 4);
  const width = Math.min(maxSize, 2048, Math.max(1, Math.ceil(Math.sqrt(texels))));
  const height = Math.ceil(texels / width);
  const padded = new Float32Array(width * height * 4);
  padded.set(floats);
  return { padded, width, height };
}

export class StrokePlayer {
  constructor(canvas, { onStatus, onStats, onError } = {}) {
    this.canvas = canvas;
    this.onStatus = onStatus || (() => {});
    this.onStats = onStats || (() => {});
    this.onError = onError || (() => {});
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false, preserveDrawingBuffer: true });
    this.renderer.setClearColor(0x000000, 1);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.debug.onShaderError = (gl, program, vertex, fragment) => {
      this.onError('Three.js shader 编译失败', {
        program: gl.getProgramInfoLog(program),
        vertex: gl.getShaderInfoLog(vertex),
        fragment: gl.getShaderInfoLog(fragment),
      });
    };
    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.clock = new THREE.Clock();
    this.playing = true;
    this.elapsed = 0;
    this.showVectors = false;
    this.loaded = false;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement || canvas);
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  async load(url) {
    this.onStatus('正在下载绘图指令…');
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) throw new Error(`无法加载 ${url}: HTTP ${response.status}`);
    const buffer = await response.arrayBuffer();
    const view = new DataView(buffer);
    this.lastBufferInfo = {
      bytes: buffer.byteLength,
      magic: new TextDecoder().decode(new Uint8Array(buffer, 0, Math.min(8, buffer.byteLength))),
      header: buffer.byteLength >= 20 ? {
        version: view.getUint16(8, true),
        keyframes: view.getUint16(10, true),
        jsonLength: view.getUint32(12, true),
        strokeCount: view.getUint32(16, true),
      } : null,
    };
    const meta = this.parse(buffer);
    if (meta.background?.path) await this.loadBackground(meta.background.path);
    this.loaded = true;
    this.onStatus('运行时绘制中');
  }

  async loadBackground(url) {
    const texture = await new THREE.TextureLoader().loadAsync(url);
    texture.colorSpace = THREE.NoColorSpace;
    texture.flipY = false;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.generateMipmaps = false;
    texture.needsUpdate = true;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute([
      -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0,
    ], 3));
    geometry.setIndex([0, 1, 2, 0, 2, 3]);
    this.backgroundUniforms = {
      uBase: { value: texture },
      uLogicalSize: { value: new THREE.Vector2(this.meta.width, this.meta.height) },
      uBackgroundA: { value: new THREE.Vector3(1, 0, 0) },
      uBackgroundB: { value: new THREE.Vector3(0, 1, 0) },
    };
    const material = new THREE.RawShaderMaterial({
      vertexShader: BACKGROUND_VERTEX_SHADER,
      fragmentShader: BACKGROUND_FRAGMENT_SHADER,
      uniforms: this.backgroundUniforms,
      depthTest: false,
      depthWrite: false,
    });
    const background = new THREE.Mesh(geometry, material);
    background.frustumCulled = false;
    background.renderOrder = -1;
    this.scene.add(background);
  }

  parse(buffer) {
    const { meta, camera, params, colors, keys, layerCounts, strokeCount, bytes } = parseInstructions(buffer);
    this.meta = meta;
    this.cameraKeys = camera;
    // The uniform-free instruction textures keep every primitive addressable
    // from its instance index and avoid per-stroke JavaScript work each frame.
    const maxTextureSize = this.renderer.capabilities.maxTextureSize || 2048;
    const paramTexture = paddedTextureDimensions(params, maxTextureSize);
    const colorTexture = paddedTextureDimensions(colors, maxTextureSize);
    const keyTexture = paddedTextureDimensions(keys, maxTextureSize);
    const indices = new Float32Array(strokeCount);
    for (let i = 0; i < strokeCount; i++) indices[i] = i;

    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute([
      -0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0,
    ], 3));
    geometry.setIndex([0, 1, 2, 0, 2, 3]);
    geometry.setAttribute('aIndex', new THREE.InstancedBufferAttribute(indices, 1));
    geometry.instanceCount = strokeCount;
    // Bounds are in clip space; the vertex shader writes final clip positions.
    geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 10);

    this.uniforms = {
      uParams: { value: makeDataTexture(paramTexture.padded, paramTexture.width, paramTexture.height, this.renderer) },
      uColor: { value: makeDataTexture(colorTexture.padded, colorTexture.width, colorTexture.height, this.renderer) },
      uKeys: { value: makeDataTexture(keyTexture.padded, keyTexture.width, keyTexture.height, this.renderer) },
      uParamSize: { value: new THREE.Vector2(paramTexture.width, paramTexture.height) },
      uColorSize: { value: new THREE.Vector2(colorTexture.width, colorTexture.height) },
      uKeySize: { value: new THREE.Vector2(keyTexture.width, keyTexture.height) },
      uTime: { value: 0 },
      uLogicalSize: { value: new THREE.Vector2(this.meta.width, this.meta.height) },
      uCameraA: { value: new THREE.Vector3(1, 0, 0) },
      uCameraB: { value: new THREE.Vector3(0, 1, 0) },
    };
    const material = new THREE.RawShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      uniforms: this.uniforms,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
    });
    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 1;
    this.scene.add(this.mesh);
    this.strokeCount = strokeCount;
    this.layerCounts = layerCounts;
    this.onStats({ ...this.meta, layers: layerCounts, bytes });
    this.resize();
    return this.meta;
  }

  setCamera(time) {
    const position = Math.max(0, Math.min(1, time)) * (KEYFRAMES - 1);
    const i = Math.floor(position);
    const j = Math.min(i + 1, KEYFRAMES - 1);
    let t = position - i;
    t = t * t * (3 - 2 * t);
    const a = i * 6;
    const b = j * 6;
    for (let k = 0; k < 3; k++) {
      this.uniforms.uCameraA.value.setComponent(k, this.cameraKeys[a + k] * (1 - t) + this.cameraKeys[b + k] * t);
      this.uniforms.uCameraB.value.setComponent(k, this.cameraKeys[a + 3 + k] * (1 - t) + this.cameraKeys[b + 3 + k] * t);
    }
    if (!this.backgroundUniforms) return;
    const a0 = this.uniforms.uCameraA.value;
    const b0 = this.uniforms.uCameraB.value;
    const determinant = a0.x * b0.y - a0.y * b0.x;
    if (Math.abs(determinant) < 1e-6) return;
    this.backgroundUniforms.uBackgroundA.value.set(
      b0.y / determinant,
      -a0.y / determinant,
      (a0.y * b0.z - b0.y * a0.z) / determinant,
    );
    this.backgroundUniforms.uBackgroundB.value.set(
      -b0.x / determinant,
      a0.x / determinant,
      (b0.x * a0.z - a0.x * b0.z) / determinant,
    );
  }

  resize() {
    const parent = this.canvas.parentElement;
    const width = Math.max(1, Math.floor(parent ? parent.clientWidth : this.canvas.clientWidth));
    const height = Math.max(1, Math.floor(parent ? parent.clientHeight : this.canvas.clientHeight));
    this.renderer.setSize(width, height, false);
    if (this.uniforms) {
    }
  }

  setTime(seconds) {
    if (!this.meta) return;
    this.elapsed = Math.max(0, Math.min(this.meta.duration, seconds));
    this.drawFrame();
  }

  toggle() {
    this.playing = !this.playing;
    return this.playing;
  }

  drawFrame() {
    if (!this.loaded) return;
    const time = this.elapsed / this.meta.duration;
    this.uniforms.uTime.value = time;
    this.setCamera(time);
    this.renderer.render(this.scene, this.camera);
  }

  animate() {
    requestAnimationFrame(this.animate);
    const delta = Math.min(this.clock.getDelta(), 0.1);
    if (this.playing && this.loaded) {
      this.elapsed = (this.elapsed + delta) % this.meta.duration;
      this.drawFrame();
    }
  }
}
