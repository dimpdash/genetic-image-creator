// converts an array of textures to an array of buffers summing each of the pixels channels

@group(0)
@binding(0)
var input_texture_array: binding_array<texture_2d<f32>>;

struct BufferArray {
    buffer: array<f32>
};

@group(0)
@binding(1)
var<storage, read_write> sum_array: binding_array<BufferArray>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var img_index = global_id.z;
    var i = global_id.x;
    var j = global_id.y;
    var dimensions = textureDimensions(input_texture_array[img_index], 0);


    var value = textureLoad(input_texture_array[img_index], vec2<u32>(i, j), 0);
    var sum = value.r + value.g + value.b;

    // // the position of the pixel in a 1d array
    var array_index = i + j * dimensions.x;

    sum_array[img_index].buffer[array_index] = sum;
}
