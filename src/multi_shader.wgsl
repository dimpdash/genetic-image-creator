struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    @location(9) texture_index: u32,
    @location(10) attachment_index: i32, // specifies which color attachment to render to -1 means all of them
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
    @location(2) attachment_index: i32,
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
    out.attachment_index = instance.attachment_index;
    out.clip_position = model_matrix * vec4<f32>(model.position, 1.0);
    return out;
}

struct FragmentOutput {
    @location(0) target0: vec4<f32>,
    @location(1) target1: vec4<f32>,
    @location(2) target2: vec4<f32>,
    @location(3) target3: vec4<f32>,
    @location(4) target4: vec4<f32>,
    @location(5) target5: vec4<f32>,
    @location(6) target6: vec4<f32>,
    @location(7) target7: vec4<f32>,
}
 

// Fragment shader

// Define a struct to hold texture and sampler bindings
@group(0) @binding(0)
var t_diffuse: binding_array<texture_2d<f32>>;
@group(0) @binding(1)
var s_diffuse: sampler;
@group(1) @binding(0)
var<storage, read> number_of_color_attachments: u32;

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    // let color_attachment_index = in.attachment_index;

    switch in.attachment_index {
        case 0: {
            var out: FragmentOutput;
            out.target0 = textureSample(t_diffuse[in.texture_index], s_diffuse, in.tex_coords);
            return out;
        }
        case 1: {
            var out: FragmentOutput;
            out.target1 = textureSample(t_diffuse[in.texture_index], s_diffuse, in.tex_coords);
            return out;
        }
        case 2: {
            var out: FragmentOutput;
            out.target2 = textureSample(t_diffuse[in.texture_index], s_diffuse, in.tex_coords);
            return out;
        }
        case 3: {
            var out: FragmentOutput;
            out.target3 = textureSample(t_diffuse[in.texture_index], s_diffuse, in.tex_coords);
            return out;
        }
        case 4: {
            var out: FragmentOutput;
            out.target4 = textureSample(t_diffuse[in.texture_index], s_diffuse, in.tex_coords);
            return out;
        }
        case 5: {
            var out: FragmentOutput;
            out.target5 = textureSample(t_diffuse[in.texture_index], s_diffuse, in.tex_coords);
            return out;
        }
        case 6: {
            var out: FragmentOutput;
            out.target6 = textureSample(t_diffuse[in.texture_index], s_diffuse, in.tex_coords);
            return out;
        }
        case 7: {
            var out: FragmentOutput;
            out.target7 = textureSample(t_diffuse[in.texture_index], s_diffuse, in.tex_coords);
            return out;
        }
        default: {
            var out: FragmentOutput;
            var color = textureSample(t_diffuse[in.texture_index], s_diffuse, in.tex_coords);
            out.target0 = color;
            out.target1 = color;
            out.target2 = color;
            out.target3 = color;
            out.target4 = color;
            out.target5 = color;
            out.target6 = color;
            out.target7 = color;
            
            return out;
        }
    }

    var out: FragmentOutput;


    return out;
}
 