@group(0)
@binding(0)
var input_texture_array: binding_array<texture_2d<f32>>;

@group(0)
@binding(1)
var<storage, read_write> sum_array: array<f32>;

@compute
@workgroup_size(1)
fn add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var sum = 0.0;
    var img_index = global_id.z;

    var dimensions = textureDimensions(input_texture_array[img_index]);
    // for now just loop over the entire texture with one thread and sum it
    for (var i = u32(0); i < dimensions.x; i = i + u32(1)) {
        for (var j = u32(0); j < dimensions.y; j = j + u32(1)) {
            var value = textureLoad(input_texture_array[img_index], vec2<u32>(i, j), 0);
            sum = sum + value.r + value.g + value.b;
        }
    }

    sum_array[img_index] = sum;
}
