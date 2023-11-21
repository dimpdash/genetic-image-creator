@group(0)
@binding(0)
var input_texture: texture_2d<f32>;

@group(0)
@binding(1)
var<storage, read_write> sum: f32;

@compute
@workgroup_size(1)
fn add() {
    sum = 0.0;
    var dimensions = textureDimensions(input_texture);
    //for now just loop over the entire texture with one thread and sum it
    for (var i = u32(0); i < dimensions.x; i = i + u32(1)) {
        for (var j = u32(0); j < dimensions.y; j = j + u32(1)) {
            var value = textureLoad(input_texture, vec2<u32>(i, j), 0);
            sum = sum + value.r + value.g + value.b;
        }
    }
}
