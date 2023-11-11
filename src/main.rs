use std::ops::Mul;

use image::DynamicImage;
use imageproc;
use rand::{self, Rng};

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
        self.x = self.x + 1;
        self.y = self.y + 1;
    }

    fn get_image(&self) -> &DynamicImage {
        &self.image
    }
}

struct Shaper {
    number_of_shapes: u32,
    keep: usize,
    rounds: u32,
    replication_factor: u32,
    source_images: Vec<DynamicImage>,
    target_image: DynamicImage,
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

        let shape = Shape {
            x: 0,
            y: 0,
            rot: 0.0,
            image: image_resized,
            fit: 0.0,
        };
        shape
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
            for shape in shapes.iter_mut() {
                let mut image_copy = image.clone();
                //add shape to image
                image::imageops::overlay(&mut image_copy, shape.get_image(), shape.x, shape.y);

                println!("Calculating fit");
                //compare to target image
                let mse = imageproc::stats::root_mean_squared_error(target_image, &image_copy);
                shape.fit = mse;
            }

            //sort shapes based on fit with with lowest first
            shapes.sort_by(|a, b| a.fit.total_cmp(&b.fit));

            // only keep some shapes
            shapes.truncate(self.keep);

            let mut new_shapes = vec![];

            // mutate shapes
            for shape in shapes.iter_mut() {
                for _ in 0..self.replication_factor {
                    // mutate shape
                    let mut copy_shape = shape.clone();
                    copy_shape.mutate();
                    new_shapes.push(copy_shape);
                }
            }

            shapes.append(&mut new_shapes);
        }

        //overlay the best shape onto the image
        let best_shape = shapes.first().unwrap();
        image::imageops::overlay(image, &best_shape.image, best_shape.x, best_shape.y);
    }

    fn generate_image(&self) -> DynamicImage {
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

fn main() {
    let source_files_folder_path = "./assets/sources/minecraft";
    //load all source images in
    let mut source_images: Vec<DynamicImage> = vec![];
    for entry in std::fs::read_dir(source_files_folder_path).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let image = image::open(path).unwrap();
        source_images.push(image);
    }
    let target_image = image::open("./assets/targets/grapes.jpg").unwrap();

    let shaper = Shaper {
        number_of_shapes: 3,
        keep: 2,
        rounds: 2,
        replication_factor: 1,
        source_images: source_images,
        target_image: target_image,
    };

    let image = shaper.generate_image();

    //save
    image.save("image.png").unwrap();
}
