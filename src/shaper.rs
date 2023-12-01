use image::DynamicImage;
use imageproc;
use rand::{self, Rng};
use rayon::prelude::*;

#[derive(Clone)]
struct Shape {
    pub x: i64,
    pub y: i64,
    pub rot: f64,
    pub fit: f64,
    image: DynamicImage,
}

impl Shape {
    fn mutate(&mut self) {
        self.x += 1;
        self.y += 1;
    }

    fn get_image(&self) -> &DynamicImage {
        &self.image
    }
}

pub struct Shaper {
    pub number_of_shapes: u32,
    pub keep: usize,
    pub rounds: u32,
    pub replication_factor: u32,
    pub source_images: Vec<DynamicImage>,
    pub target_image: DynamicImage,
}

impl Shaper {
    fn get_random_shape(&self) -> Shape {
        // get random image from source images
        let random_image_index = rand::random::<usize>() % self.source_images.len();
        let random_image: &DynamicImage = &self.source_images[random_image_index];

        //get largest side of image
        let largest_side_of_target =
            std::cmp::max(self.target_image.width(), self.target_image.height());

        let smallest: u32 = ((largest_side_of_target as f64) * 0.10) as u32;

        //get random value between smallest and largest side of target
        let mut rng = rand::thread_rng();
        let resize_size = rng.gen_range(smallest..largest_side_of_target);

        // scale image to this size
        let image_resized = random_image.resize(
            resize_size,
            resize_size,
            image::imageops::FilterType::Nearest,
        );



        
        Shape {
            x: rng.gen_range(0..self.target_image.width() as i64),
            y: rng.gen_range(0..self.target_image.height() as i64),
            rot: 0.0,
            image: image_resized,
            fit: 0.0,
        }
    }

    fn add_best_shape(&self, target_image: &DynamicImage, image: &mut DynamicImage) {
        let mut shapes: Vec<Shape> = vec![];

        // generate initial shapes
        for _ in 0..self.number_of_shapes {
            let shape = self.get_random_shape();
            shapes.push(shape);
        }

        // rounds
        for i in 0..self.rounds {
            println!("Round {}", i);
            // mutate shapes
            shapes.par_iter_mut().for_each(|shape| {
                let mut image_copy = image.clone();
                //add shape to image
                image::imageops::overlay(&mut image_copy, shape.get_image(), shape.x, shape.y);

                //compare to target image
                let mse = imageproc::stats::root_mean_squared_error(target_image, &image_copy);
                shape.fit = mse;
            });

            //sort shapes based on fit with with lowest first
            shapes.sort_by(|a, b| a.fit.total_cmp(&b.fit));

            // only keep some shapes
            shapes.truncate(self.keep);

            //replicate shapes
            let mut new_shapes = shapes.par_iter().flat_map(|shape| {
                let mut new_shapes: Vec<Shape> = vec![];
                for _ in 0..self.replication_factor {
                    // mutate shape
                    let mut copy_shape = shape.clone();
                    copy_shape.mutate();
                    new_shapes.push(copy_shape);
                }
                new_shapes
            }).collect();

            shapes.append(&mut new_shapes);
        }

        //overlay the best shape onto the image
        let best_shape = shapes.first().unwrap();
        image::imageops::overlay(image, &best_shape.image, best_shape.x, best_shape.y);
    }

    pub fn generate_image(&self) -> DynamicImage {
        // create a new empty image of the same size
        let mut image =
            DynamicImage::new_rgba8(self.target_image.width(), self.target_image.height());

        //add shape
        for _ in 0..self.number_of_shapes {
            println!("Adding shape");
            self.add_best_shape(&self.target_image, &mut image);
        }

        image
    }
}