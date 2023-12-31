use std::{num::NonZeroU32, future};

//include arc
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use genetic_image_creator::wgpu_utils::output_image_native;
use rand::Rng;
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator, IntoParallelIterator};
use rayon::vec;
use wgpu::BindingResource;
use wgpu::{Features, Limits, BindGroup, Queue, Device, BindGroupLayout, util::DeviceExt, TextureView};
#[cfg(target_arch = "wasm32")]
use wgpu_example::utils::output_image_wasm;

use cgmath;
use cgmath::prelude::*;

use bytemuck;
use itertools::{izip};

// include the repr C

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2]
}

impl Vertex {
    // https://sotrh.github.io/learn-wgpu/beginner/tutorial4-buffer/#so-what-do-i-do-with-it
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;

        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
    scale: cgmath::Vector3<f32>,
    texture_index: u32,
    post_scale : cgmath::Vector3<f32>,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {

        InstanceRaw {
            model:( cgmath::Matrix4::from_nonuniform_scale(self.post_scale.x, self.post_scale.y, self.post_scale.z) * cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from(self.rotation) * cgmath::Matrix4::from_nonuniform_scale(self.scale.x, self.scale.y, self.scale.z) ).into(),
            texture_index: self.texture_index,
        }
    }

    fn fix_scale(&mut self, canvas_dimensions: (usize, usize)) -> Self {
        let canvas_dimensions = (canvas_dimensions.0 as f32, canvas_dimensions.1 as f32);
        let ratio = canvas_dimensions.1 /  canvas_dimensions.0 ;
        self.post_scale = cgmath::Vector3::new(ratio, 1.0, 1.0);
        *self
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    texture_index: u32,
}
use std::mem;

impl InstanceRaw {

    const ATTRIBS: [wgpu::VertexAttribute; 5] = [
        // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
        // for each vec4. We'll have to reassemble the mat4 in the shader.
        wgpu::VertexAttribute {
            offset: 0,
            // While our vertex shader only uses locations 0, and 1 now, in later tutorials we'll
            // be using 2, 3, and 4, for Vertex. We'll start at slot 5 not conflict with them later
            shader_location: 5,
            format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
            offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
            shader_location: 6,
            format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
            offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
            shader_location: 7,
            format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
            offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
            shader_location: 8,
            format: wgpu::VertexFormat::Float32x4,
        },
        // The texture index is a u32, so we need to use the Uint32 format
        wgpu::VertexAttribute {
            offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
            shader_location: 9,
            format: wgpu::VertexFormat::Uint32,
        },
    ];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &InstanceRaw::ATTRIBS,
        }
    }


}
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRawMulti {
    model: [[f32; 4]; 4],
    texture_index: u32,
    render_target: i32,
}

impl InstanceRawMulti {
    const ATTRIBS: [wgpu::VertexAttribute; 6] = [
        // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
        // for each vec4. We'll have to reassemble the mat4 in the shader.
        wgpu::VertexAttribute {
            offset: 0,
            // While our vertex shader only uses locations 0, and 1 now, in later tutorials we'll
            // be using 2, 3, and 4, for Vertex. We'll start at slot 5 not conflict with them later
            shader_location: 5,
            format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
            offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
            shader_location: 6,
            format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
            offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
            shader_location: 7,
            format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
            offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
            shader_location: 8,
            format: wgpu::VertexFormat::Float32x4,
        },
        // The texture index is a u32, so we need to use the Uint32 format
        wgpu::VertexAttribute {
            offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
            shader_location: 9,
            format: wgpu::VertexFormat::Uint32,
        },
        // add the render target
        wgpu::VertexAttribute {
            offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress + mem::size_of::<u32>() as wgpu::BufferAddress,
            shader_location: 10,
            format: wgpu::VertexFormat::Sint32,
        },
    ];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRawMulti>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &InstanceRawMulti::ATTRIBS,
        }
    }

    fn to_raw(instance: &Instance, render_target: i32) -> InstanceRawMulti {
        let instance_raw = instance.to_raw();
        InstanceRawMulti {
            model: instance_raw.model,
            texture_index: instance_raw.texture_index,
            render_target: render_target,
        }
    }
}


 

const VERTICES: &[Vertex] = &[
    Vertex { position: [0.5, 0.5, 0.0], tex_coords: [1.0,0.0] },   // 0
    Vertex { position: [-0.5, -0.5, 0.0], tex_coords: [0.0,1.0] }, // 1
    Vertex { position: [0.5, -0.5, 0.0], tex_coords: [1.0,1.0] },  // 2

    Vertex { position: [-0.5, 0.5, 0.0], tex_coords: [0.0,0.0] }, // 3
    // Vertex { position: [0.5, 0.5, 0.0], color: [1.0, 0.0, 0.0] }, // 0
    // Vertex { position: [-0.5, -0.5, 0.0], color: [0.0, 1.0, 0.0] }, //1
];

const INDICES: &[u16] = &[
    0, 1, 2, // first triangle
    0, 1, 3, // second triangle
];

struct MultiTargetRenderPiplineFactory {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    textures_bind_group: wgpu::BindGroup,
    number_of_targets: u32,
    number_of_targets_bind_group: wgpu::BindGroup,
}

impl MultiTargetRenderPiplineFactory {
    pub fn new(graphicsProcessor: &GraphicsProcessor, images: &Vec<DynamicImage>, number_of_targets: u32) -> MultiTargetRenderPiplineFactory {
        let device = &graphicsProcessor.device;
        let queue = &graphicsProcessor.queue;

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );
        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX,
            }
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("multi_shader.wgsl"))),
        });

        let (textures_bind_group, textures_bind_group_layout, texture_views) = RenderPiplineFactory::create_textures_bind_group(&device, &queue, images);

        let number_of_targets_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("number_of_targets_bind_group_layout"),
        });

        let number_of_targets_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &number_of_targets_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("number_of_targets_buffer"),
                        contents: bytemuck::bytes_of(&number_of_targets),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    }),
                    offset: 0,
                    size: None,
                }),
            }],
            label: Some("number_of_targets_bind_group"),
        });

        let render_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &textures_bind_group_layout, 
                &number_of_targets_bind_group_layout
            ],
            push_constant_ranges: &[],
        });

        let targets = (0..number_of_targets).map(|_| {
            Some(wgpu::ColorTargetState{
                format: wgpu::TextureFormat::Rgba8Unorm,

                blend: Some(wgpu::BlendState{
                    color: wgpu::BlendComponent{
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,},
                    alpha: wgpu::BlendComponent::OVER
                }),

                write_mask: wgpu::ColorWrites::ALL,
            })}).collect::<Vec<_>>();

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline Multi Color Attachment"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    Vertex::desc(),
                    InstanceRawMulti::desc(),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &targets,
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        MultiTargetRenderPiplineFactory {
            pipeline,
            vertex_buffer,
            index_buffer,
            textures_bind_group,
            number_of_targets,
            number_of_targets_bind_group,
        }
    }
}


struct MultiTargetRenderPipelineInstance<'a> {
    factory: &'a MultiTargetRenderPiplineFactory,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    render_targets: Vec<wgpu::Texture>,
    output_buffers: Option<&'a Vec<wgpu::Buffer>>
}

impl<'a> MultiTargetRenderPipelineInstance<'a> {

    pub fn new(graphicsProcessor: &GraphicsProcessor, canvas_dimensions : (usize, usize), common_instances: &Vec<Instance>, instances_set: Vec<Vec<Instance>>, factory: &'a MultiTargetRenderPiplineFactory) -> MultiTargetRenderPipelineInstance<'a> {
        let device = &graphicsProcessor.device;

        assert!(instances_set.len() == factory.number_of_targets as usize);

        let instances = common_instances.iter().chain(instances_set.iter().flatten()).cloned().collect::<Vec<_>>();

        // add the common instances
        // -1 means that the instance is in the common instances
        let common_instances_it = common_instances.iter().map(|instance| InstanceRawMulti::to_raw(instance, -1));
        
        // create an empty instance buffer
        // flatten the seperate instances and add the render target index
        let individual_instances_data = 
            instances_set.iter().enumerate()
            .map(|(index, instances)| instances.iter().zip(std::iter::repeat(index))
                .map(|(instance, index)| InstanceRawMulti::to_raw(instance, index as i32))
            ).flatten();

        let instances_data = common_instances_it
            .chain(individual_instances_data)
            .collect::<Vec<_>>();

        let instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: &bytemuck::cast_slice(&instances_data),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let render_targets = (0..factory.number_of_targets).map(|_| graphicsProcessor.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: canvas_dimensions.0 as u32,
                height: canvas_dimensions.1 as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        })).collect::<Vec<_>>();
 
        MultiTargetRenderPipelineInstance {
            factory: &factory,
            instances: instances,
            instance_buffer: instance_buffer,
            render_targets: render_targets,
            output_buffers: None,
        }
    }

    fn set_output_buffers(&mut self, output_buffers: &'a Vec<wgpu::Buffer>) {
        self.output_buffers = Some(output_buffers);
    }

    fn get_render_targets(&self) -> &Vec<wgpu::Texture> {
        &self.render_targets
    }

    pub fn add_to_encoder(&self, command_encoder: &mut wgpu::CommandEncoder) {
        {
            let texture_views = self.render_targets.iter().map(|render_target| render_target.create_view(&wgpu::TextureViewDescriptor::default())).collect::<Vec<_>>();

            //setup the render targets
            let color_attachments = texture_views.iter().map(|render_target_view| {Some(wgpu::RenderPassColorAttachment {
                view: render_target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })}).collect::<Vec<_>>();

            let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &color_attachments,
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            render_pass.set_pipeline(&self.factory.pipeline);
            //add texture
            render_pass.set_bind_group(0, &self.factory.textures_bind_group, &[]);
            //add number of targets
            render_pass.set_bind_group(1, &self.factory.number_of_targets_bind_group, &[]);
            //add vertrices
            render_pass.set_vertex_buffer(0, self.factory.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.factory.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            // set the instances
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

            render_pass.draw_indexed(0..INDICES.len() as u32, 0, 0..self.instances.len() as u32);
        }

        match self.output_buffers {
            Some(output_buffers) => {
                izip!(output_buffers, self.render_targets.iter()).for_each(|(output_buffer, render_target)| {
                    command_encoder.copy_texture_to_buffer(
                        wgpu::ImageCopyTexture {
                            texture: &render_target,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::ImageCopyBuffer {
                            buffer: output_buffer,
                            layout: wgpu::ImageDataLayout {
                                offset: 0,
                                bytes_per_row: Some(4 * render_target.width()),
                                rows_per_image: Some(render_target.height()),
                            },
                        },
                        wgpu::Extent3d {
                            width: render_target.width(),
                            height: render_target.height(),
                            depth_or_array_layers: 1,
                        },
                    );
                });
            },
            None => {}
        }
    }

}


struct RenderPiplineFactory {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    textures_bind_group: wgpu::BindGroup,
}

impl RenderPiplineFactory {
    pub fn new(graphicsProcessor: &GraphicsProcessor, images: &Vec<DynamicImage>) -> RenderPiplineFactory {
        let device = &graphicsProcessor.device;
        let queue = &graphicsProcessor.queue;

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );
        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX,
            }
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let (textures_bind_group, textures_bind_group_layout, texture_views) = RenderPiplineFactory::create_textures_bind_group(&device, &queue, images);

        let render_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&textures_bind_group_layout],
            push_constant_ranges: &[],
        });
    
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    Vertex::desc(),
                    InstanceRaw::desc()
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[
                    Some(wgpu::ColorTargetState{
                        format: wgpu::TextureFormat::Rgba8Unorm,
        
                        blend: Some(wgpu::BlendState{
                            color: wgpu::BlendComponent{
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,},
                            alpha: wgpu::BlendComponent::OVER
                        }),
        
                        write_mask: wgpu::ColorWrites::ALL,
                    })
                ],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });


        RenderPiplineFactory {
            pipeline,
            vertex_buffer,
            index_buffer,
            textures_bind_group,
        }
    }

    fn create_textures_bind_group(device: &Device, queue: &Queue, images: &Vec<DynamicImage>) -> (BindGroup, BindGroupLayout, Vec<TextureView>) {
        let mut texture_views = vec![];


        for image in images.iter() {

            let image_rgba = image.to_rgba8();
            let dimensions = image.dimensions();


            let texture_size = wgpu::Extent3d {
                width: dimensions.0,
                height: dimensions.1,
                depth_or_array_layers: 1,
            };

            let texture = device.create_texture(
                &wgpu::TextureDescriptor {
                    // All textures are stored as 3D, we represent our 2D texture
                    // by setting depth to 1.
                    size: texture_size,
                    mip_level_count: 1, // We'll talk about this a little later
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    // Most images are stored using sRGB so we need to reflect that here.
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    // TEXTURE_BINDING tells wgpu that we want to use this texture in shaders
                    // COPY_DST means that we want to copy data to this texture
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    label: Some("diffuse_texture"),
                    // This is the same as with the SurfaceConfig. It
                    // specifies what texture formats can be used to
                    // create TextureViews for this texture. The base
                    // texture format (Rgba8UnormSrgb in this case) is
                    // always supported. Note that using a different
                    // texture format is not supported on the WebGL2
                    // backend.
                    view_formats: &[],
                }
            );
                
            queue.write_texture(
                // Tells wgpu where to copy the pixel data
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                // The actual pixel data
                &image_rgba,
                // The layout of the texture
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * dimensions.0),
                    rows_per_image: Some(dimensions.1),
                },
                texture_size,
            );

            let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

            texture_views.push(texture_view);
        }

     
        let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
    

    
        let textures_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: NonZeroU32::new(texture_views.len() as u32),
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            // This should match the filterable field of the
                            // corresponding Texture entry above.
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                    label: Some("texture_bind_group_layout"),
                });
    
        let textures_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &textures_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureViewArray(&texture_views.iter().collect::<Vec<_>>()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                    }
                ],
                label: Some("diffuse_bind_group")
            }
        );

        (textures_bind_group, textures_bind_group_layout, texture_views)
    }
}

struct RenderPipelineInstance<'a> {
    factory: &'a RenderPiplineFactory,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    render_target: wgpu::Texture,
    output_buffer: Option<&'a wgpu::Buffer>
}

impl<'a> RenderPipelineInstance<'a> {

    pub fn new(graphicsProcessor: &GraphicsProcessor, canvas_dimensions : (usize, usize), instances: Vec<Instance>, factory: &'a RenderPiplineFactory) -> RenderPipelineInstance<'a> {
        let device = &graphicsProcessor.device;
        // create an empty instance buffer
        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: &bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let render_target = graphicsProcessor.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: canvas_dimensions.0 as u32,
                height: canvas_dimensions.1 as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        });
 
        RenderPipelineInstance {
            factory: &factory,
            instances: instances,
            instance_buffer: instance_buffer,
            render_target: render_target,
            output_buffer: None,
        }
    }

    fn set_output_buffer(&mut self, output_buffer: &'a wgpu::Buffer) {
        self.output_buffer = Some(output_buffer);
    }

    fn get_render_target(&self) -> &wgpu::Texture {
        &self.render_target
    }

    pub fn add_to_encoder(&self, command_encoder: &mut wgpu::CommandEncoder) {
        let render_texture_view = self.render_target.create_view(&wgpu::TextureViewDescriptor::default());

        {
            let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &render_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            render_pass.set_pipeline(&self.factory.pipeline);
            //add texture
            render_pass.set_bind_group(0, &self.factory.textures_bind_group, &[]);
            //add vertrices
            render_pass.set_vertex_buffer(0, self.factory.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.factory.index_buffer.slice(..), wgpu::IndexFormat::Uint16); 
            // set the instances
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(self.factory.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            render_pass.draw_indexed(0..INDICES.len() as u32, 0, 0..self.instances.len() as u32);
        }

        if let Some(output_buffer) = self.output_buffer {
            command_encoder.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: &self.render_target,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: output_buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * self.render_target.width()),
                        rows_per_image: Some(self.render_target.height()),
                    },
                },
                wgpu::Extent3d {
                    width: self.render_target.width(),
                    height: self.render_target.height(),
                    depth_or_array_layers: 1,
                },
            );
        }
    }

}

type OwnedBuf<'a> = Option<(&'a wgpu::Buffer, Option<wgpu::Buffer>)>;

struct TextureSubtractPipelineFactory {
    compute_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl TextureSubtractPipelineFactory {
    pub fn new(graphicsProcessor: &GraphicsProcessor) -> TextureSubtractPipelineFactory {
        let device = &graphicsProcessor.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("compute_shader.wgsl"))),
        });

        // create bind group
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { 
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { 
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
            label: Some("compute_bind_group_layout"),
        });

        let compute_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
    
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "subtract_images",
        });

        TextureSubtractPipelineFactory {
            compute_pipeline: pipeline,
            bind_group_layout: bind_group_layout,
        }
    }
}

struct   TextureSubtractPipeline<'a> {
    compute_pipeline: &'a wgpu::ComputePipeline,
    a_texture: &'a wgpu::Texture,
    b_texture: &'a wgpu::Texture,
    output_texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
}

impl<'a> TextureSubtractPipeline<'a> {
    pub fn new(graphicsProcessor: &GraphicsProcessor, a_texture: &'a wgpu::Texture, b_texture: &'a wgpu::Texture, factory: &'a TextureSubtractPipelineFactory) -> TextureSubtractPipeline<'a> {
        let device = &graphicsProcessor.device;
        let queue = &graphicsProcessor.queue;

        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: a_texture.width(),
                height: a_texture.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        });

        let bind_group_layout = &factory.bind_group_layout;

        let bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&a_texture.create_view(&wgpu::TextureViewDescriptor::default())),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&b_texture.create_view(&wgpu::TextureViewDescriptor::default())),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&output_texture.create_view(&wgpu::TextureViewDescriptor::default())),
                    }
                ],
                label: Some("compute_bind_group")
            }
        );



 

        TextureSubtractPipeline {
            compute_pipeline: &factory.compute_pipeline,
            a_texture: a_texture,
            b_texture: b_texture,
            output_texture: output_texture,
            bind_group,
        }
    }

    pub fn add_to_encoder(&self, command_encoder: &mut wgpu::CommandEncoder) {
        let mut cpass = command_encoder.begin_compute_pass(&Default::default());
        cpass.set_pipeline(&self.compute_pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(self.a_texture.width(), self.b_texture.height(), 1);
    }

    pub fn copy_output_to_buffer(&self, command_encoder: &mut wgpu::CommandEncoder, output_buffer: &wgpu::Buffer) {
        command_encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * self.output_texture.width()),
                    rows_per_image: Some(self.output_texture.height()),
                },
            },
            wgpu::Extent3d {
                width: self.output_texture.width(),
                height: self.output_texture.height(),
                depth_or_array_layers: 1,
            },
        );
    }
    
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ImageSizeContents {
    pub width: u32,
    pub height: u32,
}

impl ImageSizeContents {
    // https://sotrh.github.io/learn-wgpu/beginner/tutorial4-buffer/#so-what-do-i-do-with-it
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Uint32, 1 => Uint32];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;

        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}


struct SumTextureArrayPipeline {
    compute_pipeline: wgpu::ComputePipeline,
    texture_views: Vec<wgpu::TextureView>,
    output_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl SumTextureArrayPipeline {
    pub fn new(graphicsProcessor: &GraphicsProcessor, texture_views: Vec<wgpu::TextureView>) -> SumTextureArrayPipeline {
        let device = &graphicsProcessor.device;
        let queue = &graphicsProcessor.queue;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("add_shader_array.wgsl"))),
        });

            // create bind group
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { 
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: NonZeroU32::new(texture_views.len() as u32),
                },
                //output buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("compute_bind_group_layout"),
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (texture_views.len() * std::mem::size_of::<f32>() )as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureViewArray(&texture_views.iter().collect::<Vec<_>>()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output_buffer.as_entire_binding(),
                    }
                ],
                label: Some("add_shader_array_compute_bind_group")
            }
        );

        let compute_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
    
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("add_shader_array_compute_pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "add",
        });

        SumTextureArrayPipeline {
            compute_pipeline: pipeline,
            texture_views,
            output_buffer,
            bind_group,
        }
    }

    pub fn add_to_encoder(&self, command_encoder: &mut wgpu::CommandEncoder) {
        let mut cpass = command_encoder.begin_compute_pass(&Default::default());
        cpass.set_pipeline(&self.compute_pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        // for now summing is done just on one thread
        cpass.dispatch_workgroups(1, 1, self.texture_views.len() as u32);
    }

    pub fn copy_output_to_buffer(&self, command_encoder: &mut wgpu::CommandEncoder, output_buffer: &wgpu::Buffer) {
        command_encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            output_buffer,
            0,
            std::mem::size_of::<f32>() as u64 * self.texture_views.len() as u64,
        );
    }
}

struct SumTexturePipeline<'a> {
    compute_pipeline: wgpu::ComputePipeline,
    input_texture: &'a wgpu::Texture,
    output_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl<'a> SumTexturePipeline<'a> {
    pub fn new(graphicsProcessor: &GraphicsProcessor, input_texture: &'a wgpu::Texture) -> SumTexturePipeline<'a> {
        let device = &graphicsProcessor.device;
        let queue = &graphicsProcessor.queue;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("add_shader.wgsl"))),
        });

            // create bind group
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { 
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                //output buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("compute_bind_group_layout"),
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&input_texture.create_view(&wgpu::TextureViewDescriptor::default())),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &output_buffer,
                            offset: 0,
                            size: None,
                        }),
                    }
                ],
                label: Some("compute_bind_group")
            }
        );

        let compute_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
    
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "add",
        });

        SumTexturePipeline {
            compute_pipeline: pipeline,
            input_texture,
            output_buffer,
            bind_group,
        }
    }

    pub fn add_to_encoder(&self, command_encoder: &mut wgpu::CommandEncoder) {
        let mut cpass = command_encoder.begin_compute_pass(&Default::default());
        cpass.set_pipeline(&self.compute_pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        // for now summing is done just on one thread
        cpass.dispatch_workgroups(1, 1, 1);
    }

    pub fn copy_output_to_buffer(&self, command_encoder: &mut wgpu::CommandEncoder, output_buffer: &wgpu::Buffer) {
        command_encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            output_buffer,
            0,
            4,
        );
    }
}

use image::{GenericImageView, DynamicImage};

struct GraphicsProcessorBuilder {
    images: Option<Vec<DynamicImage>>,

}

impl GraphicsProcessorBuilder{
    pub fn new() -> GraphicsProcessorBuilder {
        GraphicsProcessorBuilder {
            images: None,
        }
    }

    fn set_images(&mut self, images: Vec<DynamicImage>) {
        self.images = Some(images);
    }

    pub async fn build(&self) -> GraphicsProcessor {
    
    
    
        // Specify the required features
        let features = Features::TEXTURE_BINDING_ARRAY | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING;
    
        // Specify the limits, including the maximum texture array layer count
        let limits = Limits {
            max_texture_array_layers: 256, // Adjust this based on your needs
            max_sampled_textures_per_shader_stage: 256,
            ..Default::default()
        };

    
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();

        println!("Adapter: {:?}", adapter.limits());

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: features,
                    limits: limits,
                },
                None,
            )
            .await
            .unwrap();
        log::info!("Wgpu context set up.");

        GraphicsProcessor {
            device,
            queue,
        }
    }
}

struct GraphicsProcessor {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GraphicsProcessor {

}

struct ImageWrapper {
    pub image: DynamicImage,
    pub texture_id: u32,
}

#[derive(Clone)]
struct Graphic2D {
    x: f32,
    y: f32,
    scale: f32,
    rot: f32,
    image: Arc::<ImageWrapper>,
}

impl Graphic2D{
    fn new(x: f32, y: f32, rot: f32, image: Arc<ImageWrapper>, scale : f32) -> Graphic2D {
        Graphic2D {
            x: x,
            y: y,
            rot: rot,
            image: image,
            scale: scale,
        }
    }

    pub fn get_image(&self) -> Arc::<ImageWrapper> {
        self.image.clone()
    } 

    pub fn mutate(&mut self, largest_scale: f32, smallest_scale: f32, target_image_width: u32, target_image_height: u32) {
        // mutate parameters a little bit
        let mut rng = rand::thread_rng();
        //allow changing of position by half the images size
        self.x += rng.gen_range(-0.1..0.1);
        self.y += rng.gen_range(-0.1..0.1);
        // clamp if out out bounds
        self.x = self.x.clamp(0.0, target_image_width as f32);
        self.y = self.y.clamp(0.0, target_image_height as f32);


        //pick new random rotation
        self.rot = rng.gen_range(0.0..360.0);

        //pick new random scale
        self.scale *= rng.gen_range(-0.1..0.1);
        // clamp scale if it is too small or too big
        self.scale = self.scale.clamp(smallest_scale, largest_scale);

    }

    fn get_width(&self) -> u32 {
        (self.image.image.width() as f32 * self.scale) as u32
    }

    fn create_instance(&self) -> Instance {
        let height = self.image.image.height();
        let width = self.image.image.width();

        let ratio = width as f32 / height as f32;

        Instance {
            position: cgmath::Vector3::new(self.x, self.y, 0.0),
            rotation: cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(self.rot)),
            scale: cgmath::Vector3::new(ratio*self.scale, self.scale, 1.0),
            texture_index: self.image.texture_id,
            post_scale: cgmath::Vector3::new(1.0, 1.0, 1.0),
        }
    }
}

struct Shaper {
    pub source_images: Vec<Arc<ImageWrapper>>,

    largest_scale: f32,
    smallest_scale: f32,

    target_image_width: u32,
    target_image_height: u32,
}

impl Shaper {
    fn get_random_shape(&self) -> Graphic2D {
        let mut rng = rand::thread_rng();
        
        // get random image from source images
        let random_image_index = rng.gen_range(0..self.source_images.len());
        let random_image = self.source_images[random_image_index].clone();

        let scale = rng.gen_range(self.smallest_scale..self.largest_scale);

        //get random x and y within the bounds of the target
        let x = rng.gen_range(0.0..1.0);
        let y = rng.gen_range(0.0..1.0);

        //get random rotation
        let rot = rng.gen_range(0.0..360.0);

        Graphic2D::new(x, y, rot, random_image, scale)
    }

    fn mutate_shape(&self, shape: &Graphic2D) -> Graphic2D {
        let mut new_shape = shape.clone();
        new_shape.mutate(self.largest_scale, self.smallest_scale, self.target_image_width, self.target_image_height);
        new_shape
    }


}

struct EvolutionEnvironment {
    unranked_shapes: Vec<Graphic2D>,
    ranked_shapes: Vec<(f32, Graphic2D)>,
    shape_pool_size: usize,
    shaper: Shaper,
    duplication_factor: u32,
    
}

impl EvolutionEnvironment {
    pub fn new(shape_pool_size: usize, shaper: Shaper, duplication_factor: u32) -> EvolutionEnvironment {

        EvolutionEnvironment {
            unranked_shapes: vec![],
            ranked_shapes: vec![],
            shape_pool_size: shape_pool_size,
            shaper,
            duplication_factor: duplication_factor,

        }
    }

    pub fn setup_pool(&mut self) {
        //randomly generate shapes
        self.unranked_shapes = (0..self.shape_pool_size).map(|_| {
            let shape = self.shaper.get_random_shape();
            (shape)
        }).collect::<Vec<_>>();
    }

    pub fn get_unranked_shapes(&self) -> &Vec<Graphic2D> {
        &self.unranked_shapes
    }

    pub fn rank_shapes(&mut self, scores: &Vec<f32>) {
        //check lengths match
        assert_eq!(scores.len(), self.unranked_shapes.len());

        //zip scores and shapes together
        let mut ranked_shapes = scores.iter().zip(self.unranked_shapes.iter())
            .map(|(score, shape)| (*score, shape.clone()));

        //add to ranked shapes
        self.ranked_shapes.extend(&mut ranked_shapes);

        //remove from unranked shapes
        self.unranked_shapes.clear();
    }

    pub fn cull_bad_shapes(&mut self) {
        //sort shapes based on highest score
        self.ranked_shapes.sort_by(|a, b| b.0.total_cmp(&a.0));
        // keep only the best shapes
        let num_of_shapes_to_keep = self.shape_pool_size / self.duplication_factor as usize;
        self.ranked_shapes.truncate(num_of_shapes_to_keep);
    }

    pub fn mutate_shapes_into_unranked_pool(&mut self) {
        for (_, shape) in self.ranked_shapes.iter() {
            for _ in 0..self.duplication_factor {
                let new_shape = self.shaper.mutate_shape(shape);
                self.unranked_shapes.push(new_shape);
            }
        }
    }


}

struct App {
    pub graphics_processor: GraphicsProcessor,
    pub images: Vec<Arc<ImageWrapper>>,
    pub target_image: Graphic2D,
    pub shapes: Vec<Graphic2D>,
    pub evolution_environment: EvolutionEnvironment,
    pub rounds: u32,
}


// struct ParIterWrapper<'a>(std::slice::Iter<'a, Graphic2D>);

// impl<'a> ParallelIterator for ParIterWrapper<'a> {
//     type Item = &'a Graphic2D;

//     fn drive_unindexed<C>(self, consumer: C) -> C::Result
//     where
//         C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
//     {
//         self.0.clone().drive(consumer)
//     }
// }

// struct VecGraphic2D(Vec<Graphic2D>);

// impl<'a> IntoParallelIterator for &'a VecGraphic2D {
//     type Item = &'a Graphic2D;
//     type Iter = ParIterWrapper<'a>;

//     fn into_par_iter(self) -> Self::Iter {
//         ParIterWrapper(self.0.iter())
//     }
// }

impl App{
    fn get_canvas_dimensions(&self) ->(usize, usize) {
        let canvas_dimensions = self.target_image.image.image.dimensions();
        (canvas_dimensions.0 as usize, canvas_dimensions.1 as usize)
    }

    fn best_shape(&self) -> Graphic2D {
        self.evolution_environment.ranked_shapes[0].1.clone()
    }

    fn load_images(_path : String, upto : u32) -> Vec<Arc<ImageWrapper>> {
        let mut images = vec![];
        
        for (texture_id, entry) in std::fs::read_dir(_path).unwrap().enumerate() {
            let entry = entry.unwrap();
            let path = entry.path();
            let image = image::open(path).unwrap();
            images.push(Arc::new(ImageWrapper {
                image: image,
                texture_id: texture_id as u32 + upto,
            }));
        }

        images
    }

    fn add_shapes(&mut self, new_shapes: Vec<Graphic2D>) {
        self.shapes.extend(new_shapes);
    }

    fn create_target_texture(&self) -> wgpu::Texture {
        let image = &self.target_image.image.image;
        let device = &self.graphics_processor.device;
        let queue = &self.graphics_processor.queue;

        let image_rgba = image.to_rgba8();
        let dimensions = image.dimensions();


        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(
            &wgpu::TextureDescriptor {
                // All textures are stored as 3D, we represent our 2D texture
                // by setting depth to 1.
                size: texture_size,
                mip_level_count: 1, // We'll talk about this a little later
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                // Most images are stored using sRGB so we need to reflect that here.
                format: wgpu::TextureFormat::Rgba8Unorm,
                // TEXTURE_BINDING tells wgpu that we want to use this texture in shaders
                // COPY_DST means that we want to copy data to this texture
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                label: Some("diffuse_texture"),
                // This is the same as with the SurfaceConfig. It
                // specifies what texture formats can be used to
                // create TextureViews for this texture. The base
                // texture format (Rgba8UnormSrgb in this case) is
                // always supported. Note that using a different
                // texture format is not supported on the WebGL2
                // backend.
                view_formats: &[],
            }
        );
            
        queue.write_texture(
            // Tells wgpu where to copy the pixel data
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            // The actual pixel data
            &image_rgba,
            // The layout of the texture
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            texture_size,
        );

        texture
    }

    async fn run(&mut self, _path: Option<String>) {
        self.evolution_environment.setup_pool();

        for round in 0..self.rounds {
            println!("Round: {}", round);
            let scores = self.run_round(_path.clone()).await;
            self.evolution_environment.rank_shapes(&scores);
            self.evolution_environment.cull_bad_shapes();
            self.evolution_environment.mutate_shapes_into_unranked_pool();
            return;
        }
        // select best shape
        let best_shape = self.best_shape();
        // add best shape to shapes
        self.shapes.push(best_shape);
    }

    async fn run_round(&mut self, _path: Option<String>) -> Vec<f32> {
        let canvas_dimensions = self.get_canvas_dimensions();
        let image_textures = self.images.iter().map(|image| image.image.clone()).collect::<Vec<_>>();
        let device = &self.graphics_processor.device;
        let queue = &self.graphics_processor.queue;
        let next_shapes = self.evolution_environment.get_unranked_shapes();

        let output_staging_buffers = next_shapes.iter().map(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: (Some("Output Staging Buffer")),
                size: (std::mem::size_of::<f32>() * canvas_dimensions.0 * canvas_dimensions.1) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })}).collect::<Vec<_>>();

        let output_sums_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Output Sum Buffer"),
                size: (next_shapes.len() * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
        });

        let render_pipeline_factory = RenderPiplineFactory::new(&self.graphics_processor, &image_textures);
        let texture_subtract_pipeline_factory = TextureSubtractPipelineFactory::new(&self.graphics_processor);

        let multi_target_render_pipeline_factory = MultiTargetRenderPiplineFactory::new(&self.graphics_processor, &image_textures, 8);
        
        let common_instances = self.shapes.iter().map(|shape| shape.create_instance().fix_scale(self.get_canvas_dimensions())).collect::<Vec<_>>();


        println!("Creating pipelines");

        let (mut command_buffers) = (0..next_shapes.len()).step_by(8).map(|j| {
            println!("Rendering targets");
            
            let mut command_encoder =
                self.graphics_processor.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

                let instances_set = next_shapes[j..j+8].iter().map(|shape| {
                vec![shape.create_instance().fix_scale(self.get_canvas_dimensions())]
            }).collect::<Vec<_>>();
            
            let mut multi_target_render_pipeline = MultiTargetRenderPipelineInstance::new(&self.graphics_processor, self.get_canvas_dimensions(), &common_instances, instances_set, &multi_target_render_pipeline_factory);

            multi_target_render_pipeline.add_to_encoder(&mut command_encoder);

            let rendered_images = multi_target_render_pipeline.get_render_targets();
            println!("Rendered targets");

            // subtract from target image
            let target_image = self.create_target_texture();

            for (i, rendered_image) in rendered_images.iter().enumerate() {
                println!("i: {}, j: {}", i, j);
                let output_staging_buffer = &output_staging_buffers[i + j];

                //create storage texture to store difference in 
                println!("Creating difference pipeline");
                let subtractPipeline = TextureSubtractPipeline::new(&self.graphics_processor, &target_image, rendered_image, &texture_subtract_pipeline_factory);
                subtractPipeline.add_to_encoder(&mut command_encoder);
                subtractPipeline.copy_output_to_buffer(&mut command_encoder, output_staging_buffer);
                let output_texture = subtractPipeline.output_texture;

                let output_texture_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());
            }

            command_encoder.finish()
        }).collect::<Vec<_>>();
        
        println!("Creating images");
        queue.submit(command_buffers);
            
        // // now combine those textures and perform the addition operation

        // let mut command_encoder =
        //     self.graphics_processor.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        // {
        //     let mut sumPipeline = SumTextureArrayPipeline::new(&self.graphics_processor, output_texture_views);
        //     sumPipeline.add_to_encoder(&mut command_encoder);
        //     sumPipeline.copy_output_to_buffer(&mut command_encoder, &output_sums_buffer);
        // }
        // println!("Summing images");
        // queue.submit(Some(command_encoder.finish()));


        for (i, output_staging_buffer) in output_staging_buffers.iter().enumerate() {
            {
                let mut texture_data = Vec::<u8>::with_capacity(canvas_dimensions.0 * canvas_dimensions.1 * 4);

                // Time to get our image.
                let buffer_slice = output_staging_buffer.slice(..);
                let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
                
                device.poll(wgpu::Maintain::Wait);
                receiver.receive().await.unwrap().unwrap();
                log::info!("Output buffer mapped.");
                {
                    let view = buffer_slice.get_mapped_range();
                    texture_data.extend_from_slice(&view[..]);
                }
                log::info!("Image data copied to local.");
    
                #[cfg(not(target_arch = "wasm32"))]
                let _path = _path.clone().unwrap().replace(".png", &format!("{}.png", i));
                output_image_native(texture_data.to_vec(), canvas_dimensions, _path);
                #[cfg(target_arch = "wasm32")]
                output_image_wasm(texture_data.to_vec(), canvas_dimensions);
                log::info!("Done.");
            }
            output_staging_buffer.unmap();
        }

        let mut scores = vec![0.0; next_shapes.len()];

        // {
        //     let sum_slice = output_sums_buffer.slice(..);
        //     let (sender2, reciever2) = futures_intrusive::channel::shared::oneshot_channel();
        //     sum_slice.map_async(wgpu::MapMode::Read, move |r| sender2.send(r).unwrap());
            
        //     device.poll(wgpu::Maintain::Wait);
        //     reciever2.receive().await.unwrap().unwrap();
        //     log::info!("Output buffer mapped.");
        //     {
        //         let view = sum_slice.get_mapped_range();
        //         let sum: &[f32] = bytemuck::cast_slice(&view);
        //         for i in 0..sum.len() {
        //             log::info!("Sum: {}", sum[i]);
        //         }
        //         scores.extend_from_slice(sum);
        //     }
        // }
        // output_sums_buffer.unmap();

        return scores;
    
    
    }

}

async fn run(_path: Option<String>) {

    let mut graphics_processor_builder = GraphicsProcessorBuilder::new();


    let target_image_image = Arc::new(ImageWrapper {
        image: image::open("./assets/targets/grapes.jpg").unwrap(),
        texture_id: 0,
    });

    let target_image = Graphic2D::new(0.0, 0.0, 0.0, target_image_image.clone(), 2.0);

    let mut images = vec![];
    images.push(target_image_image.clone());

    images.append(&mut App::load_images("./assets/sources/minecraft".to_string(), 1));

    let image_textures = images.iter().map(|image| image.image.clone()).collect::<Vec<_>>();

    graphics_processor_builder.set_images(image_textures);

    let shaper = Shaper {
        source_images: images.clone(),
        largest_scale: 1.3,
        smallest_scale: 0.1,
        target_image_width: target_image.image.image.width(),
        target_image_height: target_image.image.image.height(),
    };

    let evolution_environment = EvolutionEnvironment::new(16, shaper, 2);

    let mut app = App {
        graphics_processor: graphics_processor_builder.build().await,
        images,
        target_image: target_image,
        shapes: vec![],
        evolution_environment,
        rounds: 10,
    };

    let shape1 = Graphic2D::new(0.0, 0.0, 0.0, app.images[1].clone(), 1.0);
    let shape2 = Graphic2D::new(0.5, 0.0, 30.0, app.images[2].clone(), 1.0);
    app.add_shapes(vec![shape1, shape2]);
  

    app.run(_path).await;
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .format_timestamp_nanos()
            .init();

        let path = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "please_don't_git_push_me.png".to_string());
        pollster::block_on(run(Some(path)));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Info).expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run(None));
    }
}