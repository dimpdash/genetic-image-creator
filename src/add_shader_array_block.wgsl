struct BufferArray {
    buffer: array<f32>
}

@group(0)
@binding(0)
var<storage, read_write> sum_array: binding_array<BufferArray>;

const array_size : u32 = u32(${array_size});

@group(0)
@binding(1)
var<storage, read> stride: u32;

const strips = u32(${strips});;

@compute
@workgroup_size(strips, 1)
fn add(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    ) {
    var sum = 0.0;
    var img_index = global_id.z;

    var strip_size : u32 = array_size / strips;

    // each block processes a strip of the image
    var i_start :u32 = local_id.x * strip_size * stride;

    for (var i = i_start; i < i_start + strip_size; i = i + stride) {
        sum = sum + sum_array[img_index].buffer[i];
    }

    sum_array[img_index].buffer[i_start] = sum;

}
