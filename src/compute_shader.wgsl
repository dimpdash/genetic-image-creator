// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

// struct DataBuf {
//     data: array<f32>,
// }

// @group(0)
// @binding(0)
// var<storage, read_write> v_indices: DataBuf;

// @group(0)
// @binding(1)
// var<storage, read> v_indices_size: u32;

// @compute
// @workgroup_size(1)
// fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
//     // TODO: a more interesting computation than this.

//     // take log_2 of array size
//     var num_steps = log2(f32(v_indices_size));

//     v_indices.data[global_id.x] = num_steps;
    
//     var seg_size = 2u;
//     for (var i = 0u; f32(i) < num_steps; i = i + 1u) {
//         var second_entry_index = global_id.x + seg_size / 2u;
//         // skip is 2^i
//         if ((global_id.x  % seg_size) == 0u) {
//             var x1 = v_indices.data[global_id.x];
//             var x2 = v_indices.data[second_entry_index];
//             v_indices.data[global_id.x] = x1 + x2;
//         }
//         seg_size = seg_size * 2u;
//     }

// }

// @compute
// @workgroup_size(1)
// fn constant_add_array(@builtin(global_invocation_id) global_id: vec3<u32>) {
//     v_indices.data[global_id.x] = v_indices.data[global_id.x]  + 10.0;
// }

@group(0)
@binding(0)
var texture_data: texture_2d<f32>;

@group(1)
@binding(0)
var texture_data_out: texture_storage_2d<rgba8unorm, write>;

@compute
@workgroup_size(1)
fn constant_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    
    // Read the current color from the texture
    var color = textureLoad(texture_data, global_id.xy , 0);

    // // Write the updated color back to the texture
    textureStore(texture_data_out, global_id.xy, color);
}

