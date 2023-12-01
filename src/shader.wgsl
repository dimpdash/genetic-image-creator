struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    @location(9) texture_index: u32,
};
 


// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) texture_index: u32,
    @location(2) should_render: u32,
}

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.texture_index = instance.texture_index;
    out.clip_position = model_matrix * vec4<f32>(model.position, 1.0);
    // do not render instances at (0,0)
    out.should_render = u32(model.position.y > 0.0 && model.position.x > 0.0);
    return out;
}
 

// Fragment shader

// Define a struct to hold texture and sampler bindings
@group(0) @binding(0)
var t_diffuse: binding_array<texture_2d<f32>>;
@group(0)@binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var should_render: bool = in.should_render != 0u;

    if (!should_render) {
        discard;
    }
    return textureSample(t_diffuse[in.texture_index], s_diffuse, in.tex_coords);
}
 