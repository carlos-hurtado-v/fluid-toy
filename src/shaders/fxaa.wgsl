// FXAA 3.11 Quality - Fast Approximate Anti-Aliasing
// Based on NVIDIA's FXAA algorithm by Timothy Lottes
//
// This is a post-process anti-aliasing technique that detects edges
// based on luminance contrast and blends along them to reduce aliasing.

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Quality settings - tuned for good quality/performance balance
const FXAA_EDGE_THRESHOLD: f32 = 0.166;      // Minimum local contrast to apply AA
const FXAA_EDGE_THRESHOLD_MIN: f32 = 0.0833; // Darker areas need less threshold
const FXAA_SUBPIX_QUALITY: f32 = 0.75;       // Subpixel AA quality (0=off, 1=max)
const FXAA_SEARCH_STEPS: i32 = 12;           // Edge search iterations

// Full-screen triangle
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );

    let pos = positions[vertex_index];

    var output: VertexOutput;
    output.position = vec4<f32>(pos, 0.0, 1.0);
    output.uv = vec2<f32>(pos.x * 0.5 + 0.5, 1.0 - (pos.y * 0.5 + 0.5));
    return output;
}

// Convert RGB to luminance (perceptual)
fn rgb_to_luma(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(input_texture));
    let texel = 1.0 / tex_size;
    let uv = input.uv;

    // Sample center and 4 neighbors
    let rgbM = textureSample(input_texture, input_sampler, uv).rgb;
    let rgbN = textureSample(input_texture, input_sampler, uv + vec2<f32>(0.0, -texel.y)).rgb;
    let rgbS = textureSample(input_texture, input_sampler, uv + vec2<f32>(0.0, texel.y)).rgb;
    let rgbE = textureSample(input_texture, input_sampler, uv + vec2<f32>(texel.x, 0.0)).rgb;
    let rgbW = textureSample(input_texture, input_sampler, uv + vec2<f32>(-texel.x, 0.0)).rgb;

    // Convert to luminance
    let lumaM = rgb_to_luma(rgbM);
    let lumaN = rgb_to_luma(rgbN);
    let lumaS = rgb_to_luma(rgbS);
    let lumaE = rgb_to_luma(rgbE);
    let lumaW = rgb_to_luma(rgbW);

    // Find min/max luma around center
    let lumaMin = min(lumaM, min(min(lumaN, lumaS), min(lumaE, lumaW)));
    let lumaMax = max(lumaM, max(max(lumaN, lumaS), max(lumaE, lumaW)));

    // Local contrast
    let lumaRange = lumaMax - lumaMin;

    // Early exit if contrast too low (no edge)
    if lumaRange < max(FXAA_EDGE_THRESHOLD_MIN, lumaMax * FXAA_EDGE_THRESHOLD) {
        return vec4<f32>(rgbM, 1.0);
    }

    // Sample corners for better edge detection
    let rgbNW = textureSample(input_texture, input_sampler, uv + vec2<f32>(-texel.x, -texel.y)).rgb;
    let rgbNE = textureSample(input_texture, input_sampler, uv + vec2<f32>(texel.x, -texel.y)).rgb;
    let rgbSW = textureSample(input_texture, input_sampler, uv + vec2<f32>(-texel.x, texel.y)).rgb;
    let rgbSE = textureSample(input_texture, input_sampler, uv + vec2<f32>(texel.x, texel.y)).rgb;

    let lumaNW = rgb_to_luma(rgbNW);
    let lumaNE = rgb_to_luma(rgbNE);
    let lumaSW = rgb_to_luma(rgbSW);
    let lumaSE = rgb_to_luma(rgbSE);

    // Compute edge direction
    let lumaNS = lumaN + lumaS;
    let lumaWE = lumaW + lumaE;
    let lumaNWSW = lumaNW + lumaSW;
    let lumaNENE = lumaNE + lumaSE;
    let lumaNWNE = lumaNW + lumaNE;
    let lumaSWSE = lumaSW + lumaSE;

    // Gradient in each direction
    let edgeHorz = abs(lumaNWSW - 2.0 * lumaW) + abs(lumaNS - 2.0 * lumaM) * 2.0 + abs(lumaNENE - 2.0 * lumaE);
    let edgeVert = abs(lumaNWNE - 2.0 * lumaN) + abs(lumaWE - 2.0 * lumaM) * 2.0 + abs(lumaSWSE - 2.0 * lumaS);

    // Is edge horizontal or vertical?
    let isHorz = edgeHorz >= edgeVert;

    // Select edge direction
    var luma1: f32;
    var luma2: f32;
    var stepLength: f32;

    if isHorz {
        luma1 = lumaN;
        luma2 = lumaS;
        stepLength = texel.y;
    } else {
        luma1 = lumaW;
        luma2 = lumaE;
        stepLength = texel.x;
    }

    // Gradient on each side of center
    let gradient1 = luma1 - lumaM;
    let gradient2 = luma2 - lumaM;

    // Which side has steeper gradient?
    let is1Steeper = abs(gradient1) >= abs(gradient2);
    let gradientScaled = 0.25 * max(abs(gradient1), abs(gradient2));

    // Step perpendicular to edge
    var uvStep: vec2<f32>;
    var lumaLocalAvg: f32;

    if isHorz {
        uvStep = vec2<f32>(0.0, stepLength);
        if is1Steeper {
            uvStep.y = -uvStep.y;
            lumaLocalAvg = 0.5 * (luma1 + lumaM);
        } else {
            lumaLocalAvg = 0.5 * (luma2 + lumaM);
        }
    } else {
        uvStep = vec2<f32>(stepLength, 0.0);
        if is1Steeper {
            uvStep.x = -uvStep.x;
            lumaLocalAvg = 0.5 * (luma1 + lumaM);
        } else {
            lumaLocalAvg = 0.5 * (luma2 + lumaM);
        }
    }

    // Shift UV to edge
    var uvEdge = uv + uvStep * 0.5;

    // Search along edge in both directions
    var uvStep2: vec2<f32>;
    if isHorz {
        uvStep2 = vec2<f32>(texel.x, 0.0);
    } else {
        uvStep2 = vec2<f32>(0.0, texel.y);
    }

    var uvP = uvEdge + uvStep2;
    var uvN = uvEdge - uvStep2;
    var lumaEndP = rgb_to_luma(textureSample(input_texture, input_sampler, uvP).rgb) - lumaLocalAvg;
    var lumaEndN = rgb_to_luma(textureSample(input_texture, input_sampler, uvN).rgb) - lumaLocalAvg;
    var reachedP = abs(lumaEndP) >= gradientScaled;
    var reachedN = abs(lumaEndN) >= gradientScaled;

    // Continue searching until we find end of edge
    for (var i = 1; i < FXAA_SEARCH_STEPS && (!reachedP || !reachedN); i++) {
        if !reachedP {
            uvP = uvP + uvStep2;
            lumaEndP = rgb_to_luma(textureSample(input_texture, input_sampler, uvP).rgb) - lumaLocalAvg;
            reachedP = abs(lumaEndP) >= gradientScaled;
        }
        if !reachedN {
            uvN = uvN - uvStep2;
            lumaEndN = rgb_to_luma(textureSample(input_texture, input_sampler, uvN).rgb) - lumaLocalAvg;
            reachedN = abs(lumaEndN) >= gradientScaled;
        }
    }

    // Distance to edge ends
    var distP: f32;
    var distN: f32;
    if isHorz {
        distP = uvP.x - uv.x;
        distN = uv.x - uvN.x;
    } else {
        distP = uvP.y - uv.y;
        distN = uv.y - uvN.y;
    }

    // Which end is closer?
    let isCloserP = distP < distN;
    let distFinal = min(distP, distN);
    let edgeLength = distP + distN;

    // Compute blend factor
    var pixelOffset = -distFinal / edgeLength + 0.5;

    // Check if luma at end is on correct side (avoid blending across object edges)
    let lumaEndCloser = select(lumaEndN, lumaEndP, isCloserP);
    let isGoodEnd = (lumaEndCloser < 0.0) != (lumaM - lumaLocalAvg < 0.0);

    if !isGoodEnd {
        pixelOffset = 0.0;
    }

    // Subpixel AA (additional smoothing based on local average)
    let lumaAvg = (1.0 / 12.0) * (2.0 * (lumaNS + lumaWE) + lumaNWSW + lumaNENE);
    let subpixOffset = clamp(abs(lumaAvg - lumaM) / lumaRange, 0.0, 1.0);
    let subpixOffsetFinal = (-2.0 * subpixOffset + 3.0) * subpixOffset * subpixOffset;
    let subpixBlend = subpixOffsetFinal * subpixOffsetFinal * FXAA_SUBPIX_QUALITY;

    // Use larger of edge blend and subpix blend
    pixelOffset = max(pixelOffset, subpixBlend);

    // Sample at offset position
    var finalUV = uv;
    if isHorz {
        finalUV.y = finalUV.y + pixelOffset * uvStep.y;
    } else {
        finalUV.x = finalUV.x + pixelOffset * uvStep.x;
    }

    let finalColor = textureSample(input_texture, input_sampler, finalUV).rgb;
    return vec4<f32>(finalColor, 1.0);
}
