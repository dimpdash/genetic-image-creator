use std::num::NonZeroU32;
use std::rc::Rc;

#[cfg(not(target_arch = "wasm32"))]
use genetic_image_creator::wgpu_utils::output_image_native;
use wgpu::{Features, Limits, BindGroup, Queue, Device, BindGroupLayout, util::DeviceExt, TextureView};
#[cfg(target_arch = "wasm32")]
use wgpu_example::utils::output_image_wasm;

use cgmath;
use cgmath::prelude::*;

use bytemuck;

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

impl InstanceRaw {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
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
            ],
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
        
    
        // create our image data buffer
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
    
        
        // create an empty instance buffer
        let instance_data = [];
        let instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: &instance_data,
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let (textures_bind_group, textures_bind_group_layout, texture_views) = self.create_textures_bind_group(&device, &queue);

        let target_image_texture_view = texture_views.first().unwrap();
        
        let target_image_texture_view_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            }],
            label: Some("texture_bind_group_layout"),
        });

        let target_image_texture_view_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &target_image_texture_view_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&target_image_texture_view),
                }],
                label: Some("target_image_texture_view_bind_group"),
            }
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shader.wgsl"))),
        });


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
    
        log::info!("Wgpu context set up.");


        // ******************** setup the computation pipelines ********************

        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            //source: wgpu::ShaderSource::SpirV(bytes_to_u32(include_bytes!("alu.spv")).into()),
            source: wgpu::ShaderSource::Wgsl(include_str!("compute_shader.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
            ],
        });
        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None , entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ]});

        let computed_texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
            label: Some("texture_bind_group_layout"),
        });

        let image_compare_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                &texture_bind_group_layout, // target image
                &texture_bind_group_layout, // image from shapes
                &computed_texture_bind_group_layout // output image
            ],
            push_constant_ranges: &[],
        });
        let image_compare_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&image_compare_pipeline_layout),
            module: &cs_module,
            entry_point: "subtract_images",
        });

        GraphicsProcessor {
            device,
            queue,
            render_pipeline: pipeline,
            image_compare_pipeline,
            vertex_buffer,
            index_buffer,
            instances: vec![],
            instance_buffer: instance_buffer,
            textures_bind_group,
            target_image_texture_view_bind_group,
            computed_texture_bind_group_layout,
        }
    }

    fn create_textures_bind_group(& self, device: &Device, queue: &Queue) -> (BindGroup, BindGroupLayout, Vec<TextureView>) {
        let mut texture_views = vec![];


        for image in self.images.as_ref().unwrap().iter() {

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

struct GraphicsProcessor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    render_pipeline: wgpu::RenderPipeline,
    image_compare_pipeline: wgpu::ComputePipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    textures_bind_group: wgpu::BindGroup,
    target_image_texture_view_bind_group: wgpu::BindGroup,
    computed_texture_bind_group_layout: BindGroupLayout,
}

impl GraphicsProcessor {
    pub fn add_instances(&mut self, instance: Vec<Instance>) {
        self.instances.extend(instance);
        //recreate the buffer
        self.instance_buffer = self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&self.instances.iter().map(Instance::to_raw).collect::<Vec<_>>()),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );
        

    }

    pub async fn run(&mut self, _path: Option<String>, canvas_dimensions: (usize, usize)) {

        //-----------------------------------------------
    
        // This will later store the raw pixel value data locally. We'll create it now as
        // a convenient size reference.
        let mut texture_data = Vec::<u8>::with_capacity(canvas_dimensions.0 * canvas_dimensions.1 * 4);

        let render_target = self.device.create_texture(&wgpu::TextureDescriptor {
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

        let output_staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: texture_data.capacity() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
    

        let texture_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());

        let render_texture_bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true }
                    },
                    count: None,
                },
            ],
            label: Some("texture_bind_group_layout"),
        });

        let render_texture_bind_group = self.device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &render_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                ],
                label: Some("render_texture_bind_group"),
            }
        );


        let storage_texture = self.device.create_texture(&wgpu::TextureDescriptor {
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
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let storage_texture_view = storage_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let computed_texture_bind_group = self.device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &self.computed_texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&storage_texture_view),
                    },
                ],
                label: Some("computed_texture_bind_group"),
            }
        );
    
        let mut command_encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
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
            render_pass.set_pipeline(&self.render_pipeline);
            //add texture
            render_pass.set_bind_group(0, &self.textures_bind_group, &[]);
            //add vertrices
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16); 
            // set the instances
            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
    
            render_pass.draw_indexed(0..INDICES.len() as u32, 0, 0..self.instances.len() as u32);
        }
        {
            let mut compute_pass = command_encoder.begin_compute_pass(&Default::default());
            compute_pass.set_pipeline(&self.image_compare_pipeline);
            compute_pass.set_bind_group(0, &self.target_image_texture_view_bind_group, &[]);
            compute_pass.set_bind_group(1, &render_texture_bind_group, &[]);
            compute_pass.set_bind_group(2, &computed_texture_bind_group, &[]);
            compute_pass.dispatch_workgroups(canvas_dimensions.0 as u32, canvas_dimensions.1 as u32, 1);

        }

        // The texture now contains our rendered image
        command_encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &storage_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_staging_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    // This needs to be a multiple of 256. Normally we would need to pad
                    // it but we here know it will work out anyways.
                    bytes_per_row: Some((canvas_dimensions.0 * 4) as u32),
                    rows_per_image: Some(canvas_dimensions.1 as u32),
                },
            },
            wgpu::Extent3d {
                width: canvas_dimensions.0 as u32,
                height: canvas_dimensions.1 as u32,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(command_encoder.finish()));
        log::info!("Commands submitted.");
    
        //-----------------------------------------------
    
        // Time to get our image.
        let buffer_slice = output_staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        receiver.receive().await.unwrap().unwrap();
        log::info!("Output buffer mapped.");
        {
            let view = buffer_slice.get_mapped_range();
            texture_data.extend_from_slice(&view[..]);
        }
        log::info!("Image data copied to local.");
        output_staging_buffer.unmap();
    
        #[cfg(not(target_arch = "wasm32"))]
        output_image_native(texture_data.to_vec(), canvas_dimensions, _path.unwrap());
        #[cfg(target_arch = "wasm32")]
        output_image_wasm(texture_data.to_vec(), canvas_dimensions);
        log::info!("Done.");
    }
}

struct ImageWrapper {
    pub image: DynamicImage,
    pub texture_id: u32,
}


struct Graphic2D {
    x: f32,
    y: f32,
    scale: f32,
    rot: f32,
    image: Rc::<ImageWrapper>,
}

impl Graphic2D{
    fn new(x: f32, y: f32, rot: f32, image: Rc<ImageWrapper>, scale : f32) -> Graphic2D {
        Graphic2D {
            x: x,
            y: y,
            rot: rot,
            image: image,
            scale: scale,
        }
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


struct App {
    pub graphics_processor: GraphicsProcessor,
    pub images: Vec<Rc<ImageWrapper>>,
    pub target_image: Graphic2D,
    pub shapes: Vec<Graphic2D>,
}

impl App{
    fn get_canvas_dimensions(&self) ->(usize, usize) {
        let canvas_dimensions = self.target_image.image.image.dimensions();
        (canvas_dimensions.0 as usize, canvas_dimensions.1 as usize)
    }

    fn load_images(_path : String, upto : u32) -> Vec<Rc<ImageWrapper>> {
        let mut images = vec![];
        
        for (texture_id, entry) in std::fs::read_dir(_path).unwrap().enumerate() {
            let entry = entry.unwrap();
            let path = entry.path();
            let image = image::open(path).unwrap();
            images.push(Rc::new(ImageWrapper {
                image: image,
                texture_id: texture_id as u32 + upto,
            }));
        }

        images
    }

    fn add_shapes(&mut self, new_shapes: Vec<Graphic2D>) {
        self.shapes.extend(new_shapes);
    }

    fn set_instances(&mut self) {
        let mut instances = vec![];
        // add background image
        // instances.push(self.target_image.create_instance().fix_scale(self.get_canvas_dimensions()));
        //add shapess
        instances.append(&mut self.shapes.iter().map(|shape| shape.create_instance().fix_scale(self.get_canvas_dimensions())).collect::<Vec<_>>());
        self.graphics_processor.add_instances(instances)
    }

    async fn run(&mut self, _path: Option<String>) {
        self.set_instances();
        self.graphics_processor.run(_path, self.get_canvas_dimensions()).await;
    }

}

async fn run(_path: Option<String>) {

    let mut graphics_processor_builder = GraphicsProcessorBuilder::new();


    let target_image_image = Rc::new(ImageWrapper {
        image: image::open("./assets/targets/grapes.jpg").unwrap(),
        texture_id: 0,
    });

    let target_image = Graphic2D::new(0.0, 0.0, 0.0, target_image_image.clone(), 2.0);

    let mut images = vec![];
    images.push(target_image_image.clone());

    images.append(&mut App::load_images("./assets/sources/minecraft".to_string(), 1));

    let image_textures = images.iter().map(|image| image.image.clone()).collect::<Vec<_>>();

    graphics_processor_builder.set_images(image_textures);

    let mut app = App {
        graphics_processor: graphics_processor_builder.build().await,
        images,
        target_image: target_image,
        shapes: vec![],
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